"""Track E4-v1 CLI: improved oracle-evidence completion vs the E3 control (E3 code and results untouched).

Commands: build-occ, verify-occ, preflight, train-ablations, eval-ablation, choose-main, train-main,
eval-table, generative-check, leak-check, run. Results: ``paths.results_json``.
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from hready.baselines.e3_heuristic import heuristic_foot_contact
from hready.body.joint_indices import LEG_BODY_AA_INDICES, LOWER_BODY_JOINTS
from hready.data.amass import clip_flags, load_index, load_paths_config
from hready.data.e3_dataset import (
    E3TrainDataset,
    E3TrainSampler,
    collate_e3,
    window_starts,
)
from hready.data.e3_motion_memmap import load_cache_index, load_clip_memmap
from hready.data.e4_occlusion import (
    E4TrainDataset,
    build_occlusion_cache,
    clip_world_evidence_cached,
)
from hready.eval.e3_cohorts import CohortMasks
from hready.eval.e3_oracle import (
    Ctx,
    _gt_window_evidence,
    _json_default,
    _worker_single_thread,
    clip_world_evidence,
    load_trained,
    predict_clip_heuristic,
    predict_clip_model,
    print_rows,
)
from hready.metrics.e3_eval import (
    ClipEval,
    CohortAccumulator,
    accumulate_clip,
    per_subject_table,
    proxy_foot_contact,
    summarize_cohort,
)
from hready.models.ego_complete_e4 import EgoCompleteMotionE4
from hready.train.e3_oracle_engine import (
    load_checkpoint,
    save_checkpoint,
    seed_everything,
    stitch_windows,
)
from hready.train.e4_engine import compute_loss_e4, output_joints, predict_windows_e4

MAIN = "main"
TABLE_ROWS = ("heuristic", "e3_learned", "e4_same_budget", "e4_main", "gt_reference")


def same_budget_run(ctx: E4Ctx) -> str:
    """Ablation run with the main arm's config at E3's budget (the pre-registered rule picks w_phys)."""
    dec = ctx.results().get("main_decision")
    if dec is None:
        raise RuntimeError("run `choose-main` first")
    return "phys" if float(dec["w_phys"]) > 0 else "ref_locked_gen_phys0"


def deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = (
            deep_merge(out[k], v)
            if isinstance(v, dict) and isinstance(out.get(k), dict)
            else v
        )
    return out


class E4Ctx:
    def __init__(
        self, cfg: dict[str, Any], config_path: Path, device: torch.device
    ) -> None:
        self.cfg = cfg
        self.config_path = config_path
        self.device = device
        cfg3 = yaml.safe_load(Path(cfg["e3_config"]).read_text(encoding="utf-8"))
        self.c3 = Ctx(cfg3, Path(cfg["e3_config"]), device)
        self.disclaimer = " ".join(cfg["disclaimer"].split())
        if self.disclaimer != self.c3.disclaimer:
            raise ValueError("E4 disclaimer must match E3")
        self.occ_dir = cfg["occlusion_cache"]["rel_dir"]
        self.sole_t = torch.as_tensor(self.c3.sole, device=device)
        self.entries = {e.rel_path: e for e in load_index()}
        self._flags: dict[str, dict[str, Any]] = {}

    def flags(self, rel: str) -> dict[str, Any]:
        """``clip_flags`` via one index dict (E3's ``Ctx.flags`` reloads the index per call)."""
        if rel not in self._flags:
            self._flags[rel] = clip_flags(self.entries[rel])
        return self._flags[rel]

    def contact_valid(self, rels: list[str]) -> dict[str, bool]:
        return {rel: not bool(self.flags(rel).get("exclude_contact")) for rel in rels}

    # ----------------------------------------------------------------- run configs / checkpoints
    def run_cfg(self, name: str) -> dict[str, Any]:
        base = {k: self.cfg[k] for k in ("model", "lock", "loss", "train")}
        if name == MAIN:
            dec = self.results().get("main_decision")
            if dec is None:
                raise RuntimeError(
                    "run `choose-main` first (main w_phys is decided on VAL ablations)"
                )
            rc = deep_merge(base, {"loss": {"w_phys": dec["w_phys"]}})
            rc["train"]["max_steps"] = int(self.cfg["main"]["max_steps"])
            return rc
        rc = deep_merge(base, self.cfg["ablation"]["runs"][name])
        rc["train"]["max_steps"] = int(self.cfg["ablation"]["max_steps"])
        return rc

    def ckpt_dir(self, name: str) -> Path:
        root = self.cfg["train"].get("checkpoint_root")
        root = (
            Path(root)
            if root
            else Path(load_paths_config()["data_root"]) / "checkpoints" / "e4"
        )
        return root / name

    def new_model(self, rc: dict[str, Any]) -> EgoCompleteMotionE4:
        return EgoCompleteMotionE4(**rc["model"]).to(self.device)

    def load_run(
        self, name: str
    ) -> tuple[EgoCompleteMotionE4, dict[str, Any], dict[str, Any]]:
        path = self.ckpt_dir(name) / "best.pt"
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}; train run '{name}' first")
        st = load_checkpoint(path)
        rc = st["run_cfg"]
        model = self.new_model(rc)
        model.load_state_dict(st["model"])
        model.eval()
        return (
            model,
            rc,
            {
                "checkpoint": path.as_posix(),
                "step": st["step"],
                "best_selection_metric": st["best"],
            },
        )

    # ----------------------------------------------------------------- results file
    def results(self) -> dict[str, Any]:
        p = Path(self.cfg["paths"]["results_json"])
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}

    def update(self, key: str, payload: Any) -> None:
        p = Path(self.cfg["paths"]["results_json"])
        data = self.results()
        data.update(
            disclaimer=self.disclaimer,
            config=self.config_path.as_posix(),
            seed=self.cfg["seed"],
        )
        data[key] = payload
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(data, indent=1, default=_json_default) + "\n", encoding="utf-8"
        )
        print(f"[results] wrote section '{key}' -> {p.as_posix()}")

    def evidence(self, rel: str):
        return clip_world_evidence_cached(
            rel,
            motion_dir=self.c3.rel_dir,
            occ_dir=self.occ_dir,
            ev_cfg=self.c3.ev_cfg,
            eval_noise_seed=int(self.c3.cfg["eval_noise_seed"]),
        )


# --------------------------------------------------------------------------- occlusion cache


def cmd_build_occ(ctx: E4Ctx) -> dict[str, Any]:
    rels = ctx.c3.cache_rels()
    oc = ctx.cfg["occlusion_cache"]
    ego = yaml.safe_load(
        Path(ctx.c3.cfg["paths"]["ego_observation_yaml"]).read_text(encoding="utf-8")
    )
    out = build_occlusion_cache(
        rels,
        motion_dir=ctx.c3.rel_dir,
        rel_dir=ctx.occ_dir,
        occ_cfg=ego.get("occlusion") or {},
        workers=int(oc["build_workers"]),
    )
    print(json.dumps(out, indent=1))
    ctx.update("build_occ", out)
    return out


def cmd_verify_occ(ctx: E4Ctx) -> dict[str, Any]:
    """Cached-occlusion evidence must be bit-identical to the E3 (uncached) evidence."""
    oc = ctx.cfg["occlusion_cache"]
    rng = np.random.default_rng(int(ctx.cfg["seed"]))
    val_test = ctx.c3.lists["val"] + ctx.c3.lists["test"]
    clips = [
        val_test[i]
        for i in rng.choice(
            len(val_test), size=int(oc["verify_n_clips"]), replace=False
        )
    ]
    eval_bad = []
    n_frames = 0
    for rel in clips:
        _, o1, r1 = clip_world_evidence(ctx.c3, rel)
        _, o2, r2 = ctx.evidence(rel)
        n_frames += o1["joint_pos_3d"].shape[0]
        for d1, d2 in ((o1, o2), (r1, r2)):
            for k in d1:
                if not torch.equal(d1[k], d2[k]):
                    eval_bad.append(f"{rel}:{k}")
    c3 = ctx.c3
    t_by_rel = load_cache_index(c3.rel_dir)
    windows = [
        (r, st)
        for r in c3.lists["train"]
        for st in window_starts(t_by_rel[r], c3.window, c3.stride)
    ]
    common_kw = {
        "window": c3.window,
        "rel_dir": c3.rel_dir,
        "ev_cfg": c3.ev_cfg,
        "seed": int(c3.cfg["seed"]),
        "z_rot_max_rad": float(c3.cfg["train"]["z_rot_max_rad"]),
        "j0_neutral": c3.j0,
        "contact_valid": ctx.contact_valid(c3.lists["train"]),
    }
    ds3 = E3TrainDataset(windows, **common_kw)
    ds4 = E4TrainDataset(windows, occ_dir=ctx.occ_dir, **common_kw)
    train_bad = []
    keys = [
        (int(rng.integers(len(ds3))), int(rng.integers(1 << 30)))
        for _ in range(int(oc["verify_n_train_items"]))
    ]
    for key in keys:
        a, b = ds3[key], ds4[key]
        for g in ("obs", "rig", "targets"):
            for k in a[g]:
                if not torch.equal(a[g][k], b[g][k]):
                    train_bad.append(f"{key}:{g}.{k}")
    out = {
        "eval_clips_checked": len(clips),
        "eval_frames_checked": n_frames,
        "eval_bit_identical": not eval_bad,
        "train_items_checked": len(keys),
        "train_bit_identical": not train_bad,
        "pass": not eval_bad and not train_bad,
        "mismatches": (eval_bad + train_bad)[:20],
    }
    print(json.dumps(out, indent=1))
    ctx.update("verify_occ", out)
    return out


# --------------------------------------------------------------------------- training


def make_loader(
    ctx: E4Ctx, rc: dict[str, Any], start_draw: int, num_workers: int | None = None
) -> tuple[DataLoader, int]:
    c3 = ctx.c3
    t_by_rel = load_cache_index(c3.rel_dir)
    windows = [
        (rel, s)
        for rel in c3.lists["train"]
        for s in window_starts(t_by_rel[rel], c3.window, c3.stride)
    ]
    contact_valid = ctx.contact_valid(c3.lists["train"])
    ds = E4TrainDataset(
        windows,
        window=c3.window,
        rel_dir=c3.rel_dir,
        ev_cfg=c3.ev_cfg,
        seed=int(ctx.cfg["seed"]),
        z_rot_max_rad=float(c3.cfg["train"]["z_rot_max_rad"]),
        j0_neutral=c3.j0,
        contact_valid=contact_valid,
        occ_dir=ctx.occ_dir,
    )
    nw = int(rc["train"]["num_workers"]) if num_workers is None else num_workers
    loader = DataLoader(
        ds,
        batch_size=int(rc["train"]["batch_size"]),
        sampler=E3TrainSampler(len(windows), int(ctx.cfg["seed"]), start_draw),
        collate_fn=collate_e3,
        num_workers=nw,
        persistent_workers=nw > 0,
        prefetch_factor=4 if nw > 0 else None,
        pin_memory=ctx.device.type == "cuda",
        worker_init_fn=_worker_single_thread,
    )
    return loader, len(windows)


def make_optim(model: torch.nn.Module, rc: dict[str, Any]):
    tc = rc["train"]
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(tc["lr"]), weight_decay=float(tc["weight_decay"])
    )
    warm, total = int(tc["warmup_steps"]), int(tc["max_steps"])

    def lr_at(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        p = min(1.0, (step - warm) / max(1, total - warm))
        return 0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * p))

    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lr_at)


def step_fn(ctx: E4Ctx, model, opt, sched, batch, rc, gen) -> dict[str, float]:
    model.train()
    opt.zero_grad(set_to_none=True)
    loss, stats = compute_loss_e4(
        model,
        ctx.c3.fk,
        batch,
        loss_w=rc["loss"],
        lock=rc["lock"],
        sole_offsets=ctx.sole_t,
        device=ctx.device,
        amp=bool(rc["train"]["bf16"]) and ctx.device.type == "cuda",
        gen=gen,
    )
    if not rc["train"].get("skip_nonfinite", False):  # v1 path, unchanged
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(rc["train"]["grad_clip"])
        )
        opt.step()
        if sched is not None:
            sched.step()
        return stats
    # E4-v2: skip (and log) any step whose loss or gradient norm is not finite.
    finite = bool(torch.isfinite(loss))
    if finite:
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(rc["train"]["grad_clip"])
        )
        finite = bool(torch.isfinite(norm))
    if finite:
        opt.step()
    else:
        opt.zero_grad(set_to_none=True)
        ctx.__dict__.setdefault("nonfinite_steps", []).append(
            {k: stats[k] for k in ("loss", "kl_nats")}
        )
        print(
            f"[skip-nonfinite] loss={stats['loss']} kl={stats['kl_nats']}", flush=True
        )
    if sched is not None:
        sched.step()
    return {**stats, "skipped": 0.0 if finite else 1.0}


def predict_clip_e4(
    ctx: E4Ctx, model, rc, obs_w, rig_w, *, n_samples: int = 0, gen=None
):
    c3 = ctx.c3
    t_len = obs_w["joint_pos_3d"].shape[0]
    p = predict_windows_e4(
        model,
        c3.fk,
        obs_w,
        rig_w,
        window=c3.window,
        stride=c3.stride,
        lock=rc["lock"],
        device=ctx.device,
        n_samples=n_samples,
        generator=gen,
    )
    pred, dis, n_dis = stitch_windows(
        p["joints"], p["starts"], p["valid"], t_len, p["origins"]
    )
    prob, _, _ = stitch_windows(p["probs"], p["starts"], p["valid"], t_len)
    samples = None
    if n_samples:
        samples = np.stack(
            [
                stitch_windows(s, p["starts"], p["valid"], t_len, p["origins"])[0]
                for s in p["samples"]
            ]
        )
    return pred, prob, dis, n_dis, samples


def selection_items(ctx: E4Ctx) -> list[dict[str, Any]]:
    c3 = ctx.c3
    masks = CohortMasks(
        Path(c3.cfg["paths"]["ego_splits_yaml"]),
        Path(c3.cfg["paths"]["floor_work_csv"]),
        c3.lists["selection"],
    )
    items = []
    for rel in c3.lists["selection"]:
        clip, obs, rig = ctx.evidence(rel)
        items.append(
            {
                "gt": np.asarray(clip["joints_22"], dtype=np.float32),
                "obs": obs,
                "rig": rig,
                "masks": {
                    c: masks.mask(rel, clip["T"], c)
                    for c in c3.cfg["selection"]["cohorts"]
                },
            }
        )
    return items


def selection_eval(ctx: E4Ctx, model, rc, items) -> dict[str, Any]:
    """Same metric as E3: mean of pooled MPJPE (mm) over VAL `all` and `ordinary_locomotion`."""
    sums = {c: [0.0, 0] for c in ctx.c3.cfg["selection"]["cohorts"]}
    for it in items:
        pred = predict_clip_e4(ctx, model, rc, it["obs"], it["rig"])[0]
        err = np.linalg.norm(pred - it["gt"], axis=-1) * 1000.0
        for c, m in it["masks"].items():
            sums[c][0] += float(err[m].sum())
            sums[c][1] += int(m.sum()) * err.shape[1]
    per = {c: (s / n if n else None) for c, (s, n) in sums.items()}
    vals = [v for v in per.values() if v is not None]
    return {"mpjpe_mm": per, "metric": float(np.mean(vals)) if vals else float("inf")}


def train_run(ctx: E4Ctx, name: str) -> dict[str, Any]:
    rc = ctx.run_cfg(name)
    tc = rc["train"]
    seed_everything(int(ctx.cfg["seed"]))
    gen = torch.Generator().manual_seed(int(ctx.cfg["seed"]))
    model = ctx.new_model(rc)
    opt, sched = make_optim(model, rc)
    d = ctx.ckpt_dir(name)
    last_p, best_p = d / "last.pt", d / "best.pt"
    step, best, history, wall0 = 0, float("inf"), [], 0.0
    if last_p.is_file():
        st = load_checkpoint(last_p)
        if st["run_cfg"] != rc:
            raise RuntimeError(
                f"{last_p} was trained with a different run config; refusing to resume"
            )
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        step, best, history, wall0 = st["step"], st["best"], st["history"], st["wall_s"]
        torch.set_rng_state(st["torch_rng"])
        gen.set_state(st["gen_state"])
        print(f"[{name}] resumed at step {step}", flush=True)
    max_steps = int(tc["max_steps"])
    if step >= max_steps:
        print(f"[{name}] already at max_steps={max_steps}")
    items = selection_items(ctx)
    loader, n_windows = make_loader(ctx, rc, step * int(tc["batch_size"]))
    print(
        f"[{name}] windows={n_windows} max_steps={max_steps} selection_clips={len(items)}",
        flush=True,
    )
    t0 = time.perf_counter()
    acc: list[dict[str, float]] = []

    def _save(path: Path) -> None:
        save_checkpoint(
            path,
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "step": step,
                "best": best,
                "history": history,
                "wall_s": wall0 + time.perf_counter() - t0,
                "torch_rng": torch.get_rng_state(),
                "gen_state": gen.get_state(),
                "run_cfg": rc,
                "train_clip_list_sha256": ctx.c3.lists["train_meta"][
                    "train_clip_list_sha256"
                ],
            },
        )

    for batch in loader if step < max_steps else []:
        acc.append(step_fn(ctx, model, opt, sched, batch, rc, gen))
        step += 1
        if step % int(tc["log_every"]) == 0:
            mean = {k: float(np.mean([s[k] for s in acc])) for k in acc[0]}
            acc = []
            print(
                f"[{name}] step {step} "
                + " ".join(f"{k}={v:.4f}" for k, v in mean.items())
                + f" lr={sched.get_last_lr()[0]:.2e} wall_s={time.perf_counter() - t0:.0f}",
                flush=True,
            )
        if step % int(tc["val_every"]) == 0 or step == max_steps:
            sv = selection_eval(ctx, model, rc, items)
            history.append(
                {"step": step, **sv, "wall_s": wall0 + time.perf_counter() - t0}
            )
            if sv["metric"] < best:
                best = sv["metric"]
                _save(best_p)
            _save(last_p)
            print(
                f"[{name}][select] step {step} VAL {sv['mpjpe_mm']} metric={sv['metric']:.2f} best={best:.2f}",
                flush=True,
            )
        if step >= max_steps:
            break
    del loader
    best_rec = min(history, key=lambda r: r["metric"]) if history else None
    tail = [h["metric"] for h in history if h["step"] > 0.8 * max_steps]
    out = {
        "run_cfg": rc,
        "steps": step,
        "n_train_windows": n_windows,
        "selection": {"split": "val", "n_clips": len(items), **ctx.c3.cfg["selection"]},
        "best": best_rec,
        "best_is_last_step": bool(best_rec and best_rec["step"] == max_steps),
        "last_20pct_metric_range_mm": [min(tail), max(tail)] if tail else None,
        "history": history,
        "checkpoint": best_p.as_posix(),
        "wall_s": wall0 + time.perf_counter() - t0,
    }
    print(
        json.dumps(
            {k: v for k, v in out.items() if k not in ("history", "run_cfg")}, indent=1
        )
    )
    return out


def cmd_train_ablations(ctx: E4Ctx) -> dict[str, Any]:
    res = ctx.results().get("train_ablations", {})
    for name in ctx.cfg["ablation"]["runs"]:
        res[name] = train_run(ctx, name)
        ctx.update("train_ablations", res)
    return res


def cmd_train_main(ctx: E4Ctx) -> dict[str, Any]:
    out = train_run(ctx, MAIN)
    ctx.update("train_main", out)
    return out


# --------------------------------------------------------------------------- evaluation


def _common(ctx: E4Ctx, rel: str, clip, obs) -> dict[str, Any]:
    entry = ctx.entries[rel]
    fl = ctx.flags(rel)
    return {
        "rel_path": rel,
        "subject": f"{entry.subset}/{entry.subject}",
        "gt": np.asarray(clip["joints_22"], dtype=np.float32),
        "visible": obs["joint_visible"].numpy().astype(bool),
        "contact_gt": np.asarray(clip["contact"], dtype=bool),
        "exclude_contact": bool(fl.get("exclude_contact")),
        "exclude_physical_eval": bool(fl.get("exclude_physical_eval")),
    }


def learned_eval(pred, prob, dis, n_dis, common, thr) -> ClipEval:
    return ClipEval(
        pred=pred,
        contact_pred=prob > thr,
        contact_prob=prob,
        overlap_disagree_mm=dis,
        n_overlap_frames=n_dis,
        **common,
    )


def evaluate(
    ctx: E4Ctx,
    split: str,
    makers: dict[str, Callable[..., ClipEval]],
    cohorts: list[str],
) -> tuple[dict[str, dict[str, CohortAccumulator]], float]:
    c3 = ctx.c3
    ev = c3.cfg["eval"]
    tau, tau_s = (
        float(ev["ground_consistency_tolerance_m"]),
        float(ev["ground_consistency_tolerance_sensitivity_m"]),
    )
    rels = c3.lists[split]
    masks = CohortMasks(
        Path(c3.cfg["paths"]["ego_splits_yaml"]),
        Path(c3.cfg["paths"]["floor_work_csv"]),
        rels,
    )
    accs = {b: {c: CohortAccumulator() for c in cohorts} for b in makers}
    t0 = time.perf_counter()
    for i, rel in enumerate(rels):
        if i % 250 == 0:
            print(
                f"eval {split} {i}/{len(rels)} elapsed_s={time.perf_counter() - t0:.0f}",
                flush=True,
            )
        clip, obs, rig = ctx.evidence(rel)
        common = _common(ctx, rel, clip, obs)
        for b, mk in makers.items():
            ce = mk(obs, rig, common)
            for c in cohorts:
                accumulate_clip(
                    accs[b][c],
                    ce,
                    masks.mask(rel, clip["T"], c),
                    sole_offsets=c3.sole,
                    tau_m=tau,
                    tau_sensitivity_m=tau_s,
                )
    return accs, time.perf_counter() - t0


def summarize(ctx: E4Ctx, accs, learned: set[str]) -> dict[str, Any]:
    ev = ctx.c3.cfg["eval"]
    return {
        b: {
            c: {
                "disclaimer": ctx.disclaimer,
                **summarize_cohort(
                    a,
                    with_ece=b in learned,
                    n_boot=int(ev["bootstrap_n"]),
                    seed=int(ev["bootstrap_seed"]),
                ),
                **(
                    {"descriptive_only": True}
                    if c in ("sit_floor", "sit_support")
                    else {}
                ),
                **({"label": "indicative"} if c == "floor_work_eligible" else {}),
            }
            for c, a in by_c.items()
        }
        for b, by_c in accs.items()
    }


def model_maker(ctx: E4Ctx, model, rc) -> Callable[..., ClipEval]:
    thr = float(ctx.c3.cfg["eval"]["contact_threshold"])

    def mk(obs, rig, common):
        pred, prob, dis, n_dis, _ = predict_clip_e4(ctx, model, rc, obs, rig)
        return learned_eval(pred, prob, dis, n_dis, common, thr)

    return mk


def cmd_eval_ablation(ctx: E4Ctx) -> dict[str, Any]:
    """VAL only (no TEST in any design decision): one row per ablation run, identical budget."""
    makers, metas = {}, {}
    for name in ctx.cfg["ablation"]["runs"]:
        model, rc, meta = ctx.load_run(name)
        makers[name] = model_maker(ctx, model, rc)
        metas[name] = {
            **meta,
            "lock": rc["lock"] is not None,
            "generative": rc["model"]["generative"],
            "w_phys": rc["loss"]["w_phys"],
        }
    accs, wall = evaluate(ctx, "val", makers, ["all", "ordinary_locomotion"])
    table = summarize(ctx, accs, set(makers))
    print_rows(ctx.c3, "val", table)
    out = {
        "disclaimer": ctx.disclaimer,
        "split": "val",
        "budget_steps": int(ctx.cfg["ablation"]["max_steps"]),
        "runs": metas,
        "table": table,
        "wall_s": wall,
    }
    ctx.update("ablation", out)
    return out


def cmd_choose_main(ctx: E4Ctx) -> dict[str, Any]:
    ab = ctx.results()["ablation"]["table"]
    ref, phys = (
        ab["ref_locked_gen_phys0"]["all"]["metrics"],
        ab["phys"]["all"]["metrics"],
    )
    lim = float(ctx.cfg["main"]["w_phys_rule_max_mpjpe_increase_mm"])
    checks = {
        "penetration_not_worse": phys["penetration_mean_mm"]
        <= ref["penetration_mean_mm"],
        "skate_not_worse": phys["foot_skate_m_s"] <= ref["foot_skate_m_s"],
        "full_mpjpe_within_limit": phys["mpjpe_full_all_mm"]
        <= ref["mpjpe_full_all_mm"] + lim,
    }
    use = all(checks.values())
    out = {
        "rule": ctx.cfg["main"],
        "inputs_val_all": {
            k: {
                m: d[m]
                for m in ("mpjpe_full_all_mm", "penetration_mean_mm", "foot_skate_m_s")
            }
            for k, d in (("ref", ref), ("phys", phys))
        },
        "checks": checks,
        "w_phys": float(ctx.cfg["ablation"]["runs"]["phys"]["loss"]["w_phys"])
        if use
        else 0.0,
    }
    print(json.dumps(out, indent=1))
    ctx.update("main_decision", out)
    return out


def cmd_eval_table(ctx: E4Ctx) -> dict[str, Any]:
    c3 = ctx.c3
    print(ctx.disclaimer)
    m3, meta3 = load_trained(c3)
    sb = same_budget_run(ctx)
    m4b, rc4b, meta4b = ctx.load_run(sb)
    m4, rc4, meta4 = ctx.load_run(MAIN)
    for k in ("model", "lock", "loss"):
        if rc4b[k] != rc4[k]:
            raise RuntimeError(f"same-budget run '{sb}' differs from main in '{k}'")
    thr = float(c3.cfg["eval"]["contact_threshold"])

    def mk_heur(obs, rig, common):
        hp, dis, n_dis, _ = predict_clip_heuristic(c3, obs, rig)
        return ClipEval(
            pred=hp,
            contact_pred=heuristic_foot_contact(hp, c3.sole),
            contact_prob=None,
            overlap_disagree_mm=dis,
            n_overlap_frames=n_dis,
            **common,
        )

    def mk_e3(obs, rig, common):
        pred, prob, dis, n_dis = predict_clip_model(c3, m3, obs, rig)
        return learned_eval(pred, prob, dis, n_dis, common, thr)

    def mk_gt(obs, rig, common):
        return ClipEval(
            pred=common["gt"],
            contact_pred=proxy_foot_contact(common["gt"], c3.sole),
            contact_prob=None,
            overlap_disagree_mm=0.0,
            n_overlap_frames=0,
            **common,
        )

    makers = {
        "heuristic": mk_heur,
        "e3_learned": mk_e3,
        "e4_same_budget": model_maker(ctx, m4b, rc4b),
        "e4_main": model_maker(ctx, m4, rc4),
        "gt_reference": mk_gt,
    }
    cohorts = list(c3.cfg["eval"]["cohorts"])
    result: dict[str, Any] = {
        "disclaimer": ctx.disclaimer,
        "models": {
            "e3_learned": meta3,
            "e4_same_budget": {
                "run": sb,
                "max_steps": rc4b["train"]["max_steps"],
                **meta4b,
                "run_cfg": rc4b,
            },
            "e4_main": {
                "run": MAIN,
                "max_steps": rc4["train"]["max_steps"],
                **meta4,
                "run_cfg": rc4,
            },
        },
        "note": "e4_same_budget = main-arm config trained for E3's budget (40000 steps, batch 64, "
        "windows 64/32, seed 0); e4_main = same config, longer schedule",
        "ground_consistency_tolerance_m": c3.cfg["eval"][
            "ground_consistency_tolerance_m"
        ],
        "ground_consistency_tolerance_sensitivity_m": c3.cfg["eval"][
            "ground_consistency_tolerance_sensitivity_m"
        ],
        "splits": {},
    }
    floor_accs = {}
    for split in ("val", "test"):
        accs, wall = evaluate(ctx, split, makers, cohorts)
        result["splits"][split] = summarize(
            ctx, accs, {"e3_learned", "e4_same_budget", "e4_main"}
        )
        result[f"wall_s_{split}"] = wall
        print_rows(c3, split, result["splits"][split])
        floor_accs[split] = {b: accs[b]["floor_work_eligible"] for b in makers}
    keys = (
        "mpjpe_full_all_mm",
        "mpjpe_full_visible_mm",
        "mpjpe_full_hidden_mm",
        "mpjpe_lower_all_mm",
        "penetration_mean_mm",
        "foot_skate_m_s",
        "ground_consistency_violation_frac",
    )
    result["floor_work_eligible"] = {
        "label": "indicative",
        "disclaimer": ctx.disclaimer,
        "note": "kneel + lie only; VAL and TEST separate; TEST has 2 segments; never used for selection",
        "per_subject": {
            s: {b: per_subject_table(a, keys) for b, a in fa.items()}
            for s, fa in floor_accs.items()
        },
    }
    ctx.update("eval_table", result)
    return result


def cmd_generative_check(ctx: E4Ctx) -> dict[str, Any]:
    """Prior samples on VAL: spread of hidden joints and best-of-K (diagnostic, not a point estimate)."""
    gc = ctx.cfg["generative_check"]
    m4, rc4, meta = ctx.load_run(MAIN)
    rels = ctx.c3.lists["selection"][: int(gc["n_clips"])]
    gen = torch.Generator(device=ctx.device).manual_seed(int(ctx.cfg["seed"]))
    s = {
        "point_hidden": [0.0, 0],
        "best_of_k_hidden": [0.0, 0],
        "sample_mean_hidden": [0.0, 0],
        "spread_hidden": [0.0, 0],
        "spread_visible": [0.0, 0],
    }
    for rel in rels:
        clip, obs, rig = ctx.evidence(rel)
        gt = np.asarray(clip["joints_22"], dtype=np.float32)
        pred, _, _, _, samp = predict_clip_e4(
            ctx, m4, rc4, obs, rig, n_samples=int(gc["n_samples"]), gen=gen
        )
        hid = ~obs["joint_visible"].numpy().astype(bool)
        e_pt = np.linalg.norm(pred - gt, axis=-1) * 1000
        e_s = np.linalg.norm(samp - gt[None], axis=-1) * 1000  # (K, T, J)
        e_mean = np.linalg.norm(samp.mean(0) - gt, axis=-1) * 1000
        spread = (
            np.linalg.norm(samp - samp.mean(0, keepdims=True), axis=-1).mean(0) * 1000
        )
        for key, arr, m in (
            ("point_hidden", e_pt, hid),
            ("best_of_k_hidden", e_s.min(0), hid),
            ("sample_mean_hidden", e_mean, hid),
            ("spread_hidden", spread, hid),
            ("spread_visible", spread, ~hid),
        ):
            s[key][0] += float(arr[m].sum())
            s[key][1] += int(m.sum())
    out = {
        "disclaimer": ctx.disclaimer,
        "model": meta,
        "split": "val",
        "n_clips": len(rels),
        "n_samples": int(gc["n_samples"]),
        "mm": {k: v / n if n else None for k, (v, n) in s.items()},
        "note": "best_of_k uses GT to pick a sample: a diversity diagnostic, never a reported accuracy",
    }
    print(json.dumps(out, indent=1))
    ctx.update("generative_check", out)
    return out


def cmd_leak_check(ctx: E4Ctx) -> dict[str, Any]:
    """E3 leak protocol on both reported E4 models (same-budget and main)."""
    res = {name: _leak_check_run(ctx, name) for name in (same_budget_run(ctx), MAIN)}
    res["pass"] = all(r["pass"] for r in res.values())
    print("leak_check pass:", res["pass"])
    ctx.update("leak_check", res)
    return res


def _leak_check_run(ctx: E4Ctx, name: str) -> dict[str, Any]:
    """E3 leak protocol on one trained E4 model; outputs include the evidence-locked joints."""
    c3 = ctx.c3
    model, rc, meta = ctx.load_run(name)
    c3.body._model.to("cpu")
    seed = int(ctx.cfg["seed"])
    t_by_rel = load_cache_index(c3.rel_dir)
    rng = np.random.default_rng(seed)
    found = None
    for k in range(int(c3.cfg["leak_check"]["max_tries"])):
        rel = c3.lists["val"][int(rng.integers(len(c3.lists["val"])))]
        if t_by_rel[rel] < c3.window:
            continue
        clip = load_clip_memmap(rel, rel_dir=c3.rel_dir)
        s = int(rng.integers(0, t_by_rel[rel] - c3.window + 1))
        base = np.array(clip["pose_body"][s : s + c3.window], dtype=np.float32)
        obs0, rig0 = _gt_window_evidence(c3, clip, s, base, seed + k)
        vis0 = obs0["joint_visible"]
        # Legs fully hidden, but some visible joint for the positive control.
        if vis0[:, list(LOWER_BODY_JOINTS)].any() or not vis0.any():
            continue
        pert = base.reshape(-1, 21, 3).copy()
        pert[:, list(LEG_BODY_AA_INDICES), :] += 0.37
        obs1, _ = _gt_window_evidence(c3, clip, s, pert.reshape(-1, 63), seed + k)
        if torch.equal(obs0["joint_visible"], obs1["joint_visible"]):
            found = (rel, s, obs0, rig0, obs1)
            break
    if found is None:
        out = {"pass": False, "message": "no all-legs-hidden VAL window found"}
        print(json.dumps(out, indent=1))
        return out
    rel, s, obs0, rig0, obs1 = found
    dev = ctx.device

    def run(o, r):
        ob = {k: v.unsqueeze(0).to(dev) for k, v in o.items()}
        with torch.no_grad():
            out = {
                k: v.float()
                for k, v in model(
                    ob, {k: v.unsqueeze(0).to(dev) for k, v in r.items()}
                ).items()
            }
            out["joints_out"] = output_joints(out, c3.fk, ob, rc["lock"])
        return out

    def diff(a, b):
        return max(
            float((a[k] - b[k]).abs().max())
            for k in (
                "transl",
                "root_rot_6d",
                "body_rot_6d",
                "contact_logits",
                "joints_out",
            )
        )

    out0 = run(obs0, rig0)
    hid = ~obs0["joint_visible"]
    kp_hidden_diff = float(
        (obs1["keypoints_2d"] - obs0["keypoints_2d"])[hid].abs().max()
    )
    hmask = hid.unsqueeze(-1)
    gen = torch.Generator().manual_seed(seed)
    swap = {k: v.clone() for k, v in obs0.items()}
    rand = {k: v.clone() for k, v in obs0.items()}
    for k in ("joint_pos_3d", "keypoints_2d"):
        swap[k] = torch.where(hmask, obs1[k], obs0[k])
        rand[k] = torch.where(
            hmask, (torch.rand(obs0[k].shape, generator=gen) - 0.5) * 10.0, obs0[k]
        )
    swap["joint_confidence"] = torch.where(
        hid, obs1["joint_confidence"], obs0["joint_confidence"]
    )
    rand["joint_confidence"] = torch.where(
        hid, torch.rand(hid.shape, generator=gen), obs0["joint_confidence"]
    )
    swap_d, rand_d = diff(out0, run(swap, rig0)), diff(out0, run(rand, rig0))
    vis = obs0["joint_visible"]
    ti, ji = (int(x) for x in torch.nonzero(vis)[0])
    pc = {k: v.clone() for k, v in obs0.items()}
    pc["joint_pos_3d"][ti, ji, 0] += 0.05
    pos_ctrl = diff(out0, run(pc, rig0))
    b_obs = {k: v.unsqueeze(0).to(dev) for k, v in obs0.items()}
    b_rig = {k: v.unsqueeze(0).to(dev) for k, v in rig0.items()}
    z = torch.zeros(1, c3.window, 21, 3, device=dev)
    attempts = {
        "targets_kwarg": lambda: model(b_obs, b_rig, targets={"body_aa": z}),
        "init_pose_kwarg": lambda: model(b_obs, b_rig, init_body_aa=z),
        "gt_key_in_obs": lambda: model(
            {**b_obs, "joints_gt_22": torch.zeros(1, c3.window, 22, 3, device=dev)},
            b_rig,
        ),
        "pose_key_in_obs": lambda: model(
            {**b_obs, "pose_body": torch.zeros(1, c3.window, 63, device=dev)}, b_rig
        ),
        "target_key_in_rig": lambda: model(
            b_obs, {**b_rig, "transl": torch.zeros(1, c3.window, 3, device=dev)}
        ),
        "rig_missing_head_pose": lambda: model(
            b_obs, {k: v for k, v in b_rig.items() if k != "head_pos_world"}
        ),
        "rig_missing_gravity": lambda: model(
            b_obs, {k: v for k, v in b_rig.items() if k != "gravity_world"}
        ),
    }
    neg = {}
    for label, fn in attempts.items():
        try:
            fn()
            neg[label] = "ACCEPTED"
        except (TypeError, ValueError) as exc:
            neg[label] = f"rejected: {type(exc).__name__}: {exc}"
    params = list(inspect.signature(model.forward).parameters)
    res = {
        "model": meta,
        "window": {
            "rel_path": rel,
            "start": s,
            "length": c3.window,
            "all_lower_body_hidden": True,
        },
        "forward_parameters": params,
        "a_signature_obs_rig_only": params == ["obs", "rig"],
        "b_negative_control_raw_hidden_keypoints_2d_max_abs_diff": kp_hidden_diff,
        "b_hidden_slots_from_perturbed_gt_output_max_abs_diff": swap_d,
        "b_hidden_slots_random_output_max_abs_diff": rand_d,
        "b_output_bit_identical": swap_d == 0.0 and rand_d == 0.0,
        "c_positive_control_visible_joint_5cm_output_diff": pos_ctrl,
        "d_negative_controls": neg,
        "note": "outputs compared: transl, rotations, contact logits and evidence-locked joints",
    }
    res["pass"] = bool(
        res["a_signature_obs_rig_only"]
        and res["b_output_bit_identical"]
        and kp_hidden_diff > 0
        and pos_ctrl > 0
        and all(v.startswith("rejected") for v in neg.values())
    )
    print(name, json.dumps(res, indent=1))
    return res


def cmd_preflight(ctx: E4Ctx) -> dict[str, Any]:
    """Overfit one batch, exact resume, and throughput for each ablation config."""
    pf = ctx.cfg["preflight"]
    out: dict[str, Any] = {}
    for name in ctx.cfg["ablation"]["runs"]:
        rc = ctx.run_cfg(name)
        seed_everything(int(ctx.cfg["seed"]))
        loader, n_windows = make_loader(ctx, rc, 0, num_workers=0)
        batch = next(iter(loader))
        model = ctx.new_model(rc)
        opt, _ = make_optim(model, rc)
        # Constant base LR (the scheduler starts at the warm-up LR).
        for g in opt.param_groups:
            g["lr"] = float(rc["train"]["lr"])
        gen = torch.Generator().manual_seed(0)
        first = step_fn(ctx, model, opt, None, batch, rc, gen)
        last = first
        for _ in range(int(pf["overfit_steps"]) - 1):
            last = step_fn(ctx, model, opt, None, batch, rc, gen)
        st = copy.deepcopy({"model": model.state_dict(), "opt": opt.state_dict()})
        model_b = ctx.new_model(rc)
        opt_b, _ = make_optim(model_b, rc)
        for g in opt_b.param_groups:
            g["lr"] = float(rc["train"]["lr"])
        model_b.load_state_dict(st["model"])
        opt_b.load_state_dict(st["opt"])
        for m, o in ((model, opt), (model_b, opt_b)):
            torch.manual_seed(1)
            g = torch.Generator().manual_seed(1)
            for _ in range(int(pf["resume_steps"])):
                m.eval()  # dropout off; posterior noise seeded identically
                o.zero_grad(set_to_none=True)
                loss, _ = compute_loss_e4(
                    m,
                    ctx.c3.fk,
                    batch,
                    loss_w=rc["loss"],
                    lock=rc["lock"],
                    sole_offsets=ctx.sole_t,
                    device=ctx.device,
                    amp=False,
                    gen=g,
                )
                loss.backward()
                o.step()
        resume_diff = max(
            float((p - q).detach().abs().max())
            for p, q in zip(model.parameters(), model_b.parameters())
        )
        model = ctx.new_model(rc)
        opt, sched = make_optim(model, rc)
        loader, _ = make_loader(ctx, rc, 0)
        it = iter(loader)
        for _ in range(10):
            step_fn(ctx, model, opt, sched, next(it), rc, gen)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(int(pf["benchmark_steps"])):
            step_fn(ctx, model, opt, sched, next(it), rc, gen)
        torch.cuda.synchronize()
        sps = (time.perf_counter() - t0) / int(pf["benchmark_steps"])
        del it, loader
        out[name] = {
            "n_train_windows": n_windows,
            "overfit_first_loss": first["loss"],
            "overfit_last_loss": last["loss"],
            "overfit_first_joints_m": first["joints_m"],
            "overfit_last_joints_m": last["joints_m"],
            "resume_max_param_diff": resume_diff,
            "s_per_step": sps,
            "hours_at_ablation_budget": sps
            * int(ctx.cfg["ablation"]["max_steps"])
            / 3600,
            "hours_at_main_budget": sps * int(ctx.cfg["main"]["max_steps"]) / 3600,
        }
        print(name, json.dumps(out[name], indent=1), flush=True)
    ctx.update("preflight", out)
    return out


def cmd_run(ctx: E4Ctx) -> None:
    for fn in (
        cmd_build_occ,
        cmd_verify_occ,
        cmd_preflight,
        cmd_train_ablations,
        cmd_eval_ablation,
        cmd_choose_main,
        cmd_train_main,
        cmd_eval_table,
        cmd_generative_check,
        cmd_leak_check,
    ):
        print(f"=== {fn.__name__} ===", flush=True)
        fn(ctx)


COMMANDS = {
    "build-occ": cmd_build_occ,
    "verify-occ": cmd_verify_occ,
    "preflight": cmd_preflight,
    "train-ablations": cmd_train_ablations,
    "eval-ablation": cmd_eval_ablation,
    "choose-main": cmd_choose_main,
    "train-main": cmd_train_main,
    "eval-table": cmd_eval_table,
    "generative-check": cmd_generative_check,
    "leak-check": cmd_leak_check,
    "run": cmd_run,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Track E4-v1 completion (oracle control)"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/e4_completion.yaml")
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("cmd", choices=sorted(COMMANDS))
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    COMMANDS[args.cmd](E4Ctx(cfg, args.config, torch.device(args.device)))


if __name__ == "__main__":
    main()
