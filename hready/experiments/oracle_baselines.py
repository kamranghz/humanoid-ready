"""Track E3 oracle completion baselines CLI (control path only).

Every table row carries the disclaimer from the config. Commands:
train-list, build-cache, verify-cache, preflight, train, eval-table, heuristic-check, leak-check, run.
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from hready.baselines.e3_heuristic import (
    HeuristicConfig,
    clip_headings,
    heuristic_foot_contact,
    heuristic_joint_positions,
    neutral_template,
)
from hready.body.batch_forward import smpl_forward_bt
from hready.body.joint_indices import (
    LEG_BODY_AA_INDICES,
    LOWER_BODY_JOINTS,
    UPPER_BODY_JOINTS,
)
from hready.body.smplx_wrapper import load_body
from hready.data.amass import _entry_by_rel, clip_flags, load_index, load_paths_config
from hready.data.e3_clips import resolve_clip_lists
from hready.data.e3_dataset import (
    E3TrainDataset,
    E3TrainSampler,
    EvidenceConfig,
    canonicalize,
    clip_noise_rng,
    collate_e3,
    eval_window_batch,
    rotate_about_z,
    simulate_evidence,
    slice_window,
    window_starts,
    z_rotation,
)
from hready.data.e3_heading import (
    SOURCE_BACKFILL,
    SOURCE_HOLD,
    SOURCE_LOOK,
    SOURCE_PELVIS_HEAD,
    SOURCE_WORLD_Y,
)
from hready.data.e3_motion_memmap import (
    build_memmap,
    cache_disk_bytes,
    load_cache_index,
    load_clip_memmap,
    measure_window_reads,
    verify_memmap,
)
from hready.data.ego_observation import (
    camera_config_from_dict,
    load_ego_config,
    noise_config_from_dict,
    occlusion_config_from_dict,
)
from hready.data.ego_splits import split_clip_list_hashes
from hready.eval.e3_cohorts import CohortMasks
from hready.metrics.e3_eval import (
    ClipEval,
    CohortAccumulator,
    accumulate_clip,
    foot_channel_sole_offsets,
    per_subject_table,
    proxy_foot_contact,
    summarize_cohort,
)
from hready.models.ego_complete_motion import EgoCompleteMotion
from hready.train.e3_oracle_engine import (
    NeutralJointFK,
    aa_to_matrix,
    compute_loss,
    load_checkpoint,
    neutral_pelvis_rest,
    predict_clip_learned,
    save_checkpoint,
    seed_everything,
    stitch_windows,
)

BASELINES = ("heuristic", "learned", "gt_reference")


class Ctx:
    """Shared, lazily built state for one CLI invocation."""

    def __init__(
        self, cfg: dict[str, Any], config_path: Path, device: torch.device
    ) -> None:
        self.cfg = cfg
        self.config_path = config_path
        self.device = device
        ego = load_ego_config(cfg["paths"]["ego_observation_yaml"])
        self.ev_cfg = EvidenceConfig(
            cam=camera_config_from_dict(ego.get("camera")),
            noise=noise_config_from_dict(ego.get("noise")),
            occ=occlusion_config_from_dict(ego.get("occlusion")),
        )
        self.hcfg = HeuristicConfig(**cfg["heuristic"])
        self.window = int(cfg["window"]["length"])
        self.stride = int(cfg["window"]["stride"])
        self.rel_dir = cfg["memmap"]["rel_dir"]
        self.disclaimer = " ".join(cfg["disclaimer"].split())
        self._lists: dict[str, Any] | None = None
        self.body = load_body("locked_head")
        self.fk = NeutralJointFK(self.body).to(device)
        self.sole = foot_channel_sole_offsets(self.body)
        self.tpl = neutral_template(self.body)
        self.j0 = neutral_pelvis_rest(self.fk)

    @property
    def lists(self) -> dict[str, Any]:
        if self._lists is None:
            self._lists = resolve_clip_lists(self.cfg)
        return self._lists

    def cache_rels(self) -> list[str]:
        lst = self.lists
        return sorted(set(lst["train"]) | set(lst["val"]) | set(lst["test"]))

    def checkpoint_dir(self) -> Path:
        p = self.cfg["train"].get("checkpoint_dir")
        return (
            Path(p)
            if p
            else Path(load_paths_config()["data_root"]) / "checkpoints" / "e3_oracle"
        )

    def new_model(self) -> EgoCompleteMotion:
        return EgoCompleteMotion(**self.cfg["model"]).to(self.device)

    def flags(self, rel: str) -> dict[str, Any]:
        return clip_flags(_entry_by_rel(load_index(), rel))


def update_results(ctx: Ctx, key: str, payload: Any) -> None:
    path = Path(ctx.cfg["paths"]["results_json"])
    data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    data["disclaimer"] = ctx.disclaimer
    data["config"] = ctx.config_path.as_posix()
    data["seed"] = ctx.cfg["seed"]
    data["eval_noise_seed"] = ctx.cfg["eval_noise_seed"]
    data[key] = payload
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=1, default=_json_default) + "\n", encoding="utf-8"
    )
    print(f"[results] wrote section '{key}' -> {path.as_posix()}")


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _doc_split_sha(doc: Path) -> dict[str, str]:
    text = Path(doc).read_text(encoding="utf-8")
    return {
        s: re.search(rf"{s} `([0-9a-f]{{64}})`", text).group(1)
        for s in ("train", "val", "test")
    }


# --------------------------------------------------------------------------- clip lists / cache


def cmd_train_list(ctx: Ctx) -> dict[str, Any]:
    meta = ctx.lists["train_meta"]
    sj = json.loads(Path(ctx.cfg["paths"]["splits_json"]).read_text(encoding="utf-8"))
    got = sj["split_clip_list_sha256_after_tune"]
    doc = _doc_split_sha(Path(ctx.cfg["paths"]["ego_splits_doc"]))
    live = split_clip_list_hashes(load_index())
    split = {
        "recomputed_from_index": live,
        "splits_json": got,
        "docs_ego_splits_md": doc,
        "match": live == doc and got == doc,
    }
    print(
        f"train clip list: n_clips={meta['n_clips']} sha256={meta['train_clip_list_sha256']}"
    )
    print(json.dumps(meta, indent=1))
    for s in ("train", "val", "test"):
        print(
            f"split {s}: recomputed={live[s]} splits.json={got[s]} docs={doc[s]} "
            f"equal={live[s] == got[s] == doc[s]}"
        )
    out = {"train_clip_list": meta, "split_sha256": split}
    if ctx.lists["subset"]:
        out["subset"] = {
            "spec": ctx.lists["subset"],
            **{k: len(ctx.lists[k]) for k in ("train", "val", "test")},
        }
    update_results(ctx, "clip_lists", out)
    return out


def cmd_build_cache(ctx: Ctx) -> dict[str, Any]:
    rels = ctx.cache_rels()
    mm = ctx.cfg["memmap"]
    print(
        f"build-cache n_clips={len(rels)} workers={mm['build_workers']} rel_dir={ctx.rel_dir}",
        flush=True,
    )
    payload = build_memmap(rels, rel_dir=ctx.rel_dir, workers=int(mm["build_workers"]))
    summary = {k: v for k, v in payload.items() if k != "entries"}
    summary["disk_bytes"] = cache_disk_bytes(rels, rel_dir=ctx.rel_dir)
    summary["n_frames"] = int(sum(e["T"] for e in payload["entries"]))
    ok = [e["rel_path"] for e in payload["entries"]]
    if ok:
        summary["window_reads"] = measure_window_reads(
            ok, rel_dir=ctx.rel_dir, window=ctx.window
        )
    print(json.dumps(summary, indent=1))
    update_results(ctx, "build_cache", summary)
    return summary


def cmd_verify_cache(ctx: Ctx) -> dict[str, Any]:
    out = verify_memmap(
        ctx.cache_rels(),
        rel_dir=ctx.rel_dir,
        n_sample=int(ctx.cfg["memmap"]["verify_n_clips"]),
        seed=int(ctx.cfg["seed"]),
    )
    print(json.dumps(out, indent=1))
    update_results(ctx, "verify_cache", out)
    return out


# --------------------------------------------------------------------------- per-clip prediction


def clip_world_evidence(
    ctx: Ctx, rel: str
) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Clip arrays + deterministic whole-clip world evidence (one noise draw per clip, all windows share it)."""
    clip = load_clip_memmap(rel, rel_dir=ctx.rel_dir)
    obs, rig = simulate_evidence(
        clip["joints_22"],
        clip["joints_55"],
        clip_noise_rng(rel, int(ctx.cfg["eval_noise_seed"])),
        ctx.ev_cfg,
    )
    return clip, obs, rig


def predict_clip_heuristic(
    ctx: Ctx,
    obs_w: dict[str, torch.Tensor],
    rig_w: dict[str, torch.Tensor],
    *,
    world_x_offsets: bool = False,
) -> tuple[np.ndarray, float, int, np.ndarray]:
    """``world_x_offsets`` replaces the heading by world +x (negative control for the rotation test)."""
    t_len = obs_w["joint_pos_3d"].shape[0]
    headings, src = clip_headings(
        obs_w["joint_pos_3d"].numpy(),
        obs_w["joint_visible"].numpy(),
        rig_w["head_pos_world"].numpy(),
        rig_w["camera_R"].numpy(),
        ctx.tpl,
        ctx.hcfg,
    )
    if world_x_offsets:
        headings = np.tile(np.array([1.0, 0.0]), (t_len, 1))
    starts = window_starts(t_len, ctx.window, ctx.stride)
    obs, rig, origins, valid = eval_window_batch(obs_w, rig_w, starts, ctx.window)
    preds = np.zeros((len(starts), ctx.window, 22, 3), dtype=np.float32)
    for w, s in enumerate(starts):
        h = headings[s : s + ctx.window]
        h = np.concatenate([h, np.repeat(h[-1:], ctx.window - h.shape[0], axis=0)])
        preds[w] = heuristic_joint_positions(
            obs["joint_pos_3d"][w].numpy(),
            obs["joint_visible"][w].numpy(),
            rig["head_pos_world"][w].numpy(),
            h,
            ctx.tpl,
        )
    pred, dis_mm, dis_n = stitch_windows(
        preds, starts, valid.numpy(), t_len, origins.numpy()
    )
    return pred, dis_mm, dis_n, src


def predict_clip_model(
    ctx: Ctx,
    model: EgoCompleteMotion,
    obs_w: dict[str, torch.Tensor],
    rig_w: dict[str, torch.Tensor],
) -> tuple[np.ndarray, np.ndarray, float, int]:
    t_len = obs_w["joint_pos_3d"].shape[0]
    joints, probs, starts, origins, valid = predict_clip_learned(
        model,
        ctx.fk,
        obs_w,
        rig_w,
        window=ctx.window,
        stride=ctx.stride,
        device=ctx.device,
    )
    pred, dis_mm, dis_n = stitch_windows(joints, starts, valid, t_len, origins)
    prob, _, _ = stitch_windows(probs, starts, valid, t_len)
    return pred, prob, dis_mm, dis_n


def clip_evals(
    ctx: Ctx, rel: str, model: EgoCompleteMotion | None
) -> dict[str, ClipEval]:
    clip, obs_w, rig_w = clip_world_evidence(ctx, rel)
    entry = _entry_by_rel(load_index(), rel)
    fl = ctx.flags(rel)
    gt = np.asarray(clip["joints_22"], dtype=np.float32)
    common = {
        "rel_path": rel,
        "subject": f"{entry.subset}/{entry.subject}",
        "gt": gt,
        "visible": obs_w["joint_visible"].numpy().astype(bool),
        "contact_gt": np.asarray(clip["contact"], dtype=bool),
        "exclude_contact": bool(fl.get("exclude_contact")),
        "exclude_physical_eval": bool(fl.get("exclude_physical_eval")),
    }
    hp, h_dis, h_n, _ = predict_clip_heuristic(ctx, obs_w, rig_w)
    out = {
        "heuristic": ClipEval(
            pred=hp,
            contact_pred=heuristic_foot_contact(hp, ctx.sole),
            contact_prob=None,
            overlap_disagree_mm=h_dis,
            n_overlap_frames=h_n,
            **common,
        ),
        "gt_reference": ClipEval(
            pred=gt,
            contact_pred=proxy_foot_contact(gt, ctx.sole),
            contact_prob=None,
            overlap_disagree_mm=0.0,
            n_overlap_frames=0,
            **common,
        ),
    }
    if model is not None:
        lp, prob, l_dis, l_n = predict_clip_model(ctx, model, obs_w, rig_w)
        out["learned"] = ClipEval(
            pred=lp,
            contact_pred=prob > float(ctx.cfg["eval"]["contact_threshold"]),
            contact_prob=prob,
            overlap_disagree_mm=l_dis,
            n_overlap_frames=l_n,
            **common,
        )
    return out


# --------------------------------------------------------------------------- training


def _worker_single_thread(_: int) -> None:
    """One intra-op thread per loader worker (evidence simulation is many small CPU ops)."""
    torch.set_num_threads(1)


def make_train_loader(
    ctx: Ctx, start_draw: int, num_workers: int | None = None
) -> tuple[DataLoader, int]:
    t_by_rel = load_cache_index(ctx.rel_dir)
    windows = [
        (rel, s)
        for rel in ctx.lists["train"]
        for s in window_starts(t_by_rel[rel], ctx.window, ctx.stride)
    ]
    contact_valid = {
        rel: not bool(ctx.flags(rel).get("exclude_contact"))
        for rel in ctx.lists["train"]
    }
    ds = E3TrainDataset(
        windows,
        window=ctx.window,
        rel_dir=ctx.rel_dir,
        ev_cfg=ctx.ev_cfg,
        seed=int(ctx.cfg["seed"]),
        z_rot_max_rad=float(ctx.cfg["train"]["z_rot_max_rad"]),
        j0_neutral=ctx.j0,
        contact_valid=contact_valid,
    )
    nw = int(ctx.cfg["train"]["num_workers"]) if num_workers is None else num_workers
    loader = DataLoader(
        ds,
        batch_size=int(ctx.cfg["train"]["batch_size"]),
        sampler=E3TrainSampler(len(windows), int(ctx.cfg["seed"]), start_draw),
        collate_fn=collate_e3,
        num_workers=nw,
        persistent_workers=nw > 0,
        prefetch_factor=4 if nw > 0 else None,
        pin_memory=ctx.device.type == "cuda",
        worker_init_fn=_worker_single_thread,
    )
    return loader, len(windows)


def make_optim(
    ctx: Ctx, model: EgoCompleteMotion
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    tc = ctx.cfg["train"]
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(tc["lr"]), weight_decay=float(tc["weight_decay"])
    )
    warm, total = int(tc["warmup_steps"]), int(tc["max_steps"])

    def lr_at(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        p = min(1.0, (step - warm) / max(1, total - warm))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * p))

    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lr_at)


def train_step(ctx: Ctx, model, opt, sched, batch) -> dict[str, float]:
    model.train()
    opt.zero_grad(set_to_none=True)
    loss, stats = compute_loss(
        model,
        ctx.fk,
        batch,
        loss_w=ctx.cfg["loss"],
        sole_offsets=torch.as_tensor(ctx.sole, device=ctx.device),
        device=ctx.device,
        amp=bool(ctx.cfg["train"]["bf16"]) and ctx.device.type == "cuda",
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), float(ctx.cfg["train"]["grad_clip"])
    )
    opt.step()
    sched.step()
    return stats


def selection_set(ctx: Ctx, masks: CohortMasks) -> list[dict[str, Any]]:
    out = []
    for rel in ctx.lists["selection"]:
        clip, obs_w, rig_w = clip_world_evidence(ctx, rel)
        t_len = clip["T"]
        out.append(
            {
                "rel": rel,
                "gt": np.asarray(clip["joints_22"], dtype=np.float32),
                "obs": obs_w,
                "rig": rig_w,
                "masks": {
                    c: masks.mask(rel, t_len, c)
                    for c in ctx.cfg["selection"]["cohorts"]
                },
            }
        )
    return out


def selection_eval(
    ctx: Ctx, model: EgoCompleteMotion, sel: list[dict[str, Any]]
) -> dict[str, Any]:
    sums = {c: [0.0, 0] for c in ctx.cfg["selection"]["cohorts"]}
    for item in sel:
        pred, _, _, _ = predict_clip_model(ctx, model, item["obs"], item["rig"])
        err = np.linalg.norm(pred - item["gt"], axis=-1) * 1000.0
        for c, m in item["masks"].items():
            sums[c][0] += float(err[m].sum())
            sums[c][1] += int(m.sum()) * err.shape[1]
    per = {c: (s / n if n else None) for c, (s, n) in sums.items()}
    vals = [v for v in per.values() if v is not None]
    return {"mpjpe_mm": per, "metric": float(np.mean(vals)) if vals else float("inf")}


def cmd_preflight(ctx: Ctx) -> dict[str, Any]:
    seed_everything(int(ctx.cfg["seed"]))
    pf = ctx.cfg["preflight"]
    loader, n_windows = make_train_loader(ctx, 0, num_workers=0)
    batch = next(iter(loader))
    # (1) neutral FK exactness vs SmplxBody.forward on GT rotations of a real batch.
    b, t = batch["valid"].shape
    root_R = batch["targets"]["root_R"].reshape(b * t, 3, 3).to(ctx.device)
    from hready.body.rotations import matrix_to_rotation_6d

    body_R = aa_to_matrix(batch["targets"]["body_aa"].reshape(b * t, 21, 3)).to(
        ctx.device
    )
    tr = batch["targets"]["transl"].reshape(b * t, 3).to(ctx.device)
    ctx.body._model.to(ctx.device)
    with torch.no_grad():
        ref = ctx.body.forward(
            matrix_to_rotation_6d(root_R),
            matrix_to_rotation_6d(body_R).reshape(b * t, -1),
            torch.zeros(b * t, 16, device=ctx.device),
            tr,
            rot_repr="6d",
        ).joints[:, :22]
        fk_diff = float((ref - ctx.fk(tr, root_R, body_R)).abs().max())
    ctx.body._model.to("cpu")
    # (2) overfit one batch.
    model = ctx.new_model()
    opt, sched = make_optim(ctx, model)
    for g in opt.param_groups:
        g["lr"] = float(ctx.cfg["train"]["lr"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    t0 = time.perf_counter()
    first = train_step(ctx, model, opt, sched, batch)
    last = first
    for _ in range(int(pf["overfit_steps"]) - 1):
        last = train_step(ctx, model, opt, sched, batch)
    overfit_s = time.perf_counter() - t0
    # (3) resume: continue in memory vs. save -> load -> continue, same batch.
    ckpt = ctx.checkpoint_dir() / "preflight_resume.pt"
    save_checkpoint(
        ckpt,
        {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "step": int(pf["overfit_steps"]),
        },
    )
    model_b = ctx.new_model()
    opt_b, _ = make_optim(ctx, model_b)
    state = load_checkpoint(ckpt)
    model_b.load_state_dict(state["model"])
    opt_b.load_state_dict(state["opt"])
    sched_b = torch.optim.lr_scheduler.LambdaLR(opt_b, lambda s: 1.0)
    model.eval(), model_b.eval()  # dropout off for the comparison
    for _ in range(int(pf["resume_steps"])):
        for m, o, s in ((model, opt, sched), (model_b, opt_b, sched_b)):
            o.zero_grad(set_to_none=True)
            loss, _ = compute_loss(
                m,
                ctx.fk,
                batch,
                loss_w=ctx.cfg["loss"],
                sole_offsets=torch.as_tensor(ctx.sole, device=ctx.device),
                device=ctx.device,
                amp=False,
            )
            loss.backward()
            o.step()
    resume_diff = max(
        float((p - q).detach().abs().max())
        for p, q in zip(model.parameters(), model_b.parameters())
    )
    ckpt.unlink(missing_ok=True)
    # (4) throughput with the real loader (workers as configured).
    model = ctx.new_model()
    opt, sched = make_optim(ctx, model)
    loader, _ = make_train_loader(ctx, 0)
    it = iter(loader)
    for _ in range(5):
        train_step(ctx, model, opt, sched, next(it))
    if ctx.device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    n_bench = int(pf["benchmark_steps"])
    for _ in range(n_bench):
        train_step(ctx, model, opt, sched, next(it))
    if ctx.device.type == "cuda":
        torch.cuda.synchronize()
    s_per_step = (time.perf_counter() - t0) / n_bench
    del it, loader
    # Data-only rate (loader without the model) separates input pipeline from GPU time.
    loader, _ = make_train_loader(ctx, 10_000)
    it = iter(loader)
    next(it)
    t0 = time.perf_counter()
    for _ in range(n_bench):
        next(it)
    data_s_per_step = (time.perf_counter() - t0) / n_bench
    del it, loader
    tc = ctx.cfg["train"]
    out = {
        "n_train_clips": len(ctx.lists["train"]),
        "n_train_windows": n_windows,
        "fk_vs_smplx_forward_max_abs_m": fk_diff,
        "overfit_steps": int(pf["overfit_steps"]),
        "overfit_first": first,
        "overfit_last": last,
        "overfit_wall_s": overfit_s,
        "resume_steps": int(pf["resume_steps"]),
        "resume_max_param_diff": resume_diff,
        "benchmark_steps": n_bench,
        "s_per_train_step": s_per_step,
        "s_per_batch_data_only": data_s_per_step,
        "batch_size": int(tc["batch_size"]),
        "num_workers": int(tc["num_workers"]),
        "max_steps": int(tc["max_steps"]),
        "epochs_at_max_steps": int(tc["max_steps"])
        * int(tc["batch_size"])
        / max(1, n_windows),
        "train_hours_estimate_excl_selection": s_per_step
        * int(tc["max_steps"])
        / 3600.0,
    }
    print(json.dumps(out, indent=1))
    update_results(ctx, "preflight", out)
    return out


def cmd_train(ctx: Ctx) -> dict[str, Any]:
    tc = ctx.cfg["train"]
    seed_everything(int(ctx.cfg["seed"]))
    ckpt_dir = ctx.checkpoint_dir()
    model = ctx.new_model()
    opt, sched = make_optim(ctx, model)
    step, best, history, wall0 = 0, float("inf"), [], 0.0
    last_path, best_path = ckpt_dir / "last.pt", ckpt_dir / "best.pt"
    if last_path.is_file():
        st = load_checkpoint(last_path)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        step, best, history, wall0 = (
            st["step"],
            st["best"],
            st["history"],
            st.get("wall_s", 0.0),
        )
        torch.set_rng_state(st["torch_rng"])
        print(f"resumed from {last_path} at step {step}", flush=True)
    masks = CohortMasks(
        Path(ctx.cfg["paths"]["ego_splits_yaml"]),
        Path(ctx.cfg["paths"]["floor_work_csv"]),
        ctx.lists["selection"],
    )
    sel = selection_set(ctx, masks)
    loader, n_windows = make_train_loader(ctx, step * int(tc["batch_size"]))
    max_steps = int(tc["max_steps"])
    print(
        f"train windows={n_windows} clips={len(ctx.lists['train'])} max_steps={max_steps} selection_clips={len(sel)}",
        flush=True,
    )
    t0 = time.perf_counter()
    run_stats: list[dict[str, float]] = []

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
                "model_cfg": ctx.cfg["model"],
                "train_clip_list_sha256": ctx.lists["train_meta"][
                    "train_clip_list_sha256"
                ],
            },
        )

    for batch in loader:
        if step >= max_steps:
            break
        run_stats.append(train_step(ctx, model, opt, sched, batch))
        step += 1
        if step % int(tc["log_every"]) == 0:
            mean = {k: float(np.mean([s[k] for s in run_stats])) for k in run_stats[0]}
            run_stats = []
            el = time.perf_counter() - t0
            print(
                f"step {step} "
                + " ".join(f"{k}={v:.4f}" for k, v in mean.items())
                + f" lr={sched.get_last_lr()[0]:.2e} wall_s={el:.0f}",
                flush=True,
            )
        if step % int(tc["val_every"]) == 0 or step == max_steps:
            sv = selection_eval(ctx, model, sel)
            rec = {"step": step, **sv, "wall_s": wall0 + time.perf_counter() - t0}
            history.append(rec)
            if sv["metric"] < best:
                best = sv["metric"]
                _save(best_path)
            _save(last_path)
            print(
                f"[select] step {step} VAL mpjpe_mm={sv['mpjpe_mm']} metric={sv['metric']:.2f} best={best:.2f}",
                flush=True,
            )
    best_rec = min(history, key=lambda r: r["metric"]) if history else None
    out = {
        "train_clip_list_sha256": ctx.lists["train_meta"]["train_clip_list_sha256"],
        "n_train_clips": len(ctx.lists["train"]),
        "n_train_windows": n_windows,
        "steps": step,
        "selection": {"split": "val", "n_clips": len(sel), **ctx.cfg["selection"]},
        "best": best_rec,
        "history": history,
        "checkpoint": best_path.as_posix(),
        "wall_s": wall0 + time.perf_counter() - t0,
    }
    print(json.dumps({k: v for k, v in out.items() if k != "history"}, indent=1))
    update_results(ctx, "train", out)
    return out


def load_trained(ctx: Ctx) -> tuple[EgoCompleteMotion, dict[str, Any]]:
    path = ctx.checkpoint_dir() / "best.pt"
    if not path.is_file():
        raise FileNotFoundError(
            f"trained checkpoint missing: {path} (run `train` first)"
        )
    st = load_checkpoint(path)
    model = ctx.new_model()
    model.load_state_dict(st["model"])
    model.eval()
    return model, {
        "checkpoint": path.as_posix(),
        "step": st["step"],
        "best_selection_metric": st["best"],
    }


# --------------------------------------------------------------------------- evaluation table


def _fmt(v: Any, nd: int = 1) -> str:
    if v is None:
        return "-"
    if isinstance(v, str):
        return v
    return f"{v:.{nd}f}"


def print_rows(ctx: Ctx, split: str, table: dict[str, Any]) -> None:
    hdr = (
        "split | baseline | cohort | clips | MPJPE full/upper/lower/foot mm | full vis/hid mm | "
        "skate m/s | pen mm | GC viol (tau) | GC viol (tau sens.) | contact F1 | ECE | overlap mm | lost c/p | disclaimer"
    )
    print(hdr)
    for base, cohorts in table.items():
        for c, row in cohorts.items():
            if row.get("n_clips", 0) == 0:
                print(
                    f"{split} | {base} | {c} | 0 | no frames | | | | | | | | | | {ctx.disclaimer}"
                )
                continue
            m = row["metrics"]
            print(
                " | ".join(
                    [
                        split,
                        base,
                        c,
                        str(row["n_clips"]),
                        "/".join(
                            _fmt(m.get(f"mpjpe_{g}_all_mm"))
                            for g in ("full", "upper", "lower", "foot")
                        ),
                        f"{_fmt(m.get('mpjpe_full_visible_mm'))}/{_fmt(m.get('mpjpe_full_hidden_mm'))}",
                        _fmt(m.get("foot_skate_m_s"), 3),
                        _fmt(m.get("penetration_mean_mm"), 2),
                        _fmt(m.get("ground_consistency_violation_frac"), 3),
                        _fmt(
                            m.get("ground_consistency_violation_frac_tau_sensitivity"),
                            3,
                        ),
                        _fmt(m.get("contact_f1"), 3),
                        _fmt(m.get("contact_ece"), 3),
                        _fmt(row.get("overlap_disagree_mm"), 2),
                        f"{row['clips_lost_exclude_contact']}/{row['clips_lost_exclude_physical_eval']}",
                        ctx.disclaimer,
                    ]
                )
            )


def cmd_eval_table(ctx: Ctx) -> dict[str, Any]:
    print(ctx.disclaimer)
    model, mmeta = load_trained(ctx)
    ev = ctx.cfg["eval"]
    tau = float(ev["ground_consistency_tolerance_m"])
    tau_s = float(ev["ground_consistency_tolerance_sensitivity_m"])
    cohorts = list(ev["cohorts"])
    rels_all = sorted(set(ctx.lists["val"]) | set(ctx.lists["test"]))
    masks = CohortMasks(
        Path(ctx.cfg["paths"]["ego_splits_yaml"]),
        Path(ctx.cfg["paths"]["floor_work_csv"]),
        rels_all,
    )
    floor_union = {b: CohortAccumulator() for b in BASELINES}
    result: dict[str, Any] = {
        "disclaimer": ctx.disclaimer,
        "model": mmeta,
        "ground_consistency_tolerance_m": tau,
        "ground_consistency_tolerance_sensitivity_m": tau_s,
        "tau_note": "primary tau = recorded 0.04947 (docs/e0_audit.md); sensitivity tau = single floor "
        "grounding (docs/pivot_log.md 2026-10-07)",
        "splits": {},
    }
    t0 = time.perf_counter()
    for split in ("val", "test"):
        accs = {b: {c: CohortAccumulator() for c in cohorts} for b in BASELINES}
        rels = ctx.lists[split]
        for i, rel in enumerate(rels):
            if i % 100 == 0:
                print(
                    f"eval {split} {i}/{len(rels)} elapsed_s={time.perf_counter() - t0:.0f}",
                    flush=True,
                )
            evs = clip_evals(ctx, rel, model)
            t_len = evs["heuristic"].gt.shape[0]
            for c in cohorts:
                m = masks.mask(rel, t_len, c)
                for b in BASELINES:
                    accumulate_clip(
                        accs[b][c],
                        evs[b],
                        m,
                        sole_offsets=ctx.sole,
                        tau_m=tau,
                        tau_sensitivity_m=tau_s,
                    )
                    if c == "floor_work_eligible":
                        accumulate_clip(
                            floor_union[b],
                            evs[b],
                            m,
                            sole_offsets=ctx.sole,
                            tau_m=tau,
                            tau_sensitivity_m=tau_s,
                        )
        table = {
            b: {
                c: {
                    "disclaimer": ctx.disclaimer,
                    **summarize_cohort(
                        accs[b][c],
                        with_ece=b == "learned",
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
                for c in cohorts
            }
            for b in BASELINES
        }
        result["splits"][split] = table
        print_rows(ctx, split, table)
    keys = (
        "mpjpe_full_all_mm",
        "mpjpe_lower_all_mm",
        "mpjpe_foot_all_mm",
        "mpjpe_full_hidden_mm",
        "foot_skate_m_s",
        "penetration_mean_mm",
        "ground_consistency_violation_frac",
        "ground_consistency_violation_frac_tau_sensitivity",
    )
    result["floor_work_eligible_val_union_test"] = {
        "label": "indicative",
        "disclaimer": ctx.disclaimer,
        "note": "kneel + lie only; 8 subjects in VAL+TEST (E1); kneel concentrated in Eyes_Japan; TEST has 2 segments",
        "per_subject": {b: per_subject_table(floor_union[b], keys) for b in BASELINES},
        "pooled": {
            b: summarize_cohort(
                floor_union[b],
                with_ece=b == "learned",
                n_boot=int(ev["bootstrap_n"]),
                seed=int(ev["bootstrap_seed"]),
            )
            for b in BASELINES
        },
    }
    print("floor_work_eligible per-subject (VAL+TEST, indicative):")
    print(
        json.dumps(
            result["floor_work_eligible_val_union_test"]["per_subject"], indent=1
        )
    )
    result["model_selection"] = {"split": "val", **ctx.cfg["selection"]}
    result["wall_s"] = time.perf_counter() - t0
    update_results(ctx, "eval_table", result)
    return result


# --------------------------------------------------------------------------- heuristic check


def _rotate_world_evidence(obs_w, rig_w, rz):
    obs = {k: v.clone() for k, v in obs_w.items()}
    rig = {k: v.clone() for k, v in rig_w.items()}
    rotate_about_z(obs, rig, None, rz)
    rig["camera_t"] = -torch.einsum(
        "tij,tj->ti", rig["camera_R"], rig["head_pos_world"]
    )
    return obs, rig


def cmd_heuristic_check(ctx: Ctx) -> dict[str, Any]:
    """Heuristic MPJPE on VAL ordinary_locomotion frames vs. a naive fill; world-rotation invariance."""
    hc = ctx.cfg["heuristic_check"]
    val = ctx.lists["val"]
    masks = CohortMasks(
        Path(ctx.cfg["paths"]["ego_splits_yaml"]),
        Path(ctx.cfg["paths"]["floor_work_csv"]),
        val,
    )
    t_by_rel = load_cache_index(ctx.rel_dir)
    loco = [r for r in val if masks.mask(r, t_by_rel[r], "ordinary_locomotion").any()]
    rng = np.random.default_rng(int(ctx.cfg["seed"]))
    pick = sorted(
        rng.choice(
            loco, size=min(int(hc["n_clips"]), len(loco)), replace=False
        ).tolist()
    )
    groups = {
        "full": list(range(22)),
        "upper": list(UPPER_BODY_JOINTS),
        "lower": list(LOWER_BODY_JOINTS),
        "foot": [7, 8, 10, 11],
    }
    sums = {
        f"{name}_{g}_{p}": [0.0, 0]
        for name in ("heuristic", "naive_head_fill")
        for g in groups
        for p in ("all", "hidden")
    }
    src_count = np.zeros(5, dtype=np.int64)
    max_dmpjpe, max_dpos, max_overlap, neg_dmpjpe = 0.0, 0.0, 0.0, 0.0
    for rel in pick:
        clip, obs_w, rig_w = clip_world_evidence(ctx, rel)
        gt = np.asarray(clip["joints_22"], dtype=np.float32)
        m = masks.mask(rel, clip["T"], "ordinary_locomotion")
        pred, dis_mm, _, src = predict_clip_heuristic(ctx, obs_w, rig_w)
        max_overlap = max(max_overlap, dis_mm)
        src_count += np.bincount(src, minlength=5)
        vis = obs_w["joint_visible"].numpy().astype(bool)
        naive = np.where(
            vis[..., None],
            obs_w["joint_pos_3d"].numpy(),
            rig_w["head_pos_world"].numpy()[:, None, :],
        )
        for name, p in (("heuristic", pred), ("naive_head_fill", naive)):
            err = np.linalg.norm(p[m] - gt[m], axis=-1) * 1000.0
            hid = ~vis[m]
            for g, idx in groups.items():
                sums[f"{name}_{g}_all"][0] += float(err[:, idx].sum())
                sums[f"{name}_{g}_all"][1] += err[:, idx].size
                sums[f"{name}_{g}_hidden"][0] += float(err[:, idx][hid[:, idx]].sum())
                sums[f"{name}_{g}_hidden"][1] += int(hid[:, idx].sum())
        base_mpjpe = float(np.linalg.norm(pred[m] - gt[m], axis=-1).mean() * 1000.0)
        pred_x, _, _, _ = predict_clip_heuristic(
            ctx, obs_w, rig_w, world_x_offsets=True
        )
        base_x = float(np.linalg.norm(pred_x[m] - gt[m], axis=-1).mean() * 1000.0)
        for deg in hc["rotations_deg"]:
            rz = z_rotation(math.radians(float(deg)))
            obs_r, rig_r = _rotate_world_evidence(obs_w, rig_w, rz)
            pred_r, _, _, _ = predict_clip_heuristic(ctx, obs_r, rig_r)
            gt_r = gt @ rz.numpy().T
            mp_r = float(np.linalg.norm(pred_r[m] - gt_r[m], axis=-1).mean() * 1000.0)
            max_dmpjpe = max(max_dmpjpe, abs(mp_r - base_mpjpe))
            max_dpos = max(max_dpos, float(np.abs(pred_r - pred @ rz.numpy().T).max()))
            px_r, _, _, _ = predict_clip_heuristic(
                ctx, obs_r, rig_r, world_x_offsets=True
            )
            mx_r = float(np.linalg.norm(px_r[m] - gt_r[m], axis=-1).mean() * 1000.0)
            neg_dmpjpe = max(neg_dmpjpe, abs(mx_r - base_x))
    mp = {k: (s / n if n else None) for k, (s, n) in sums.items()}
    total = int(src_count.sum())
    out = {
        "disclaimer": ctx.disclaimer,
        "cohort": "ordinary_locomotion (VAL)",
        "template": ctx.tpl.as_dict(),
        "n_clips": len(pick),
        "n_val_clips_with_ordinary_locomotion": len(loco),
        "mpjpe_mm": mp,
        "rotation_invariance": {
            "rotations_deg": hc["rotations_deg"],
            "max_abs_mpjpe_change_mm": max_dmpjpe,
            "max_abs_position_diff_m": max_dpos,
            "negative_control_world_x_offsets_max_abs_mpjpe_change_mm": neg_dmpjpe,
        },
        "heading_source_fraction": {
            name: float(src_count[code] / total)
            for name, code in (
                ("look_axis", SOURCE_LOOK),
                ("hold_last", SOURCE_HOLD),
                ("pelvis_to_head", SOURCE_PELVIS_HEAD),
                ("world_plus_y", SOURCE_WORLD_Y),
                ("backfill_first_stable", SOURCE_BACKFILL),
            )
        },
        "max_overlap_disagree_mm": max_overlap,
    }
    print(json.dumps(out, indent=1))
    update_results(ctx, "heuristic_check", out)
    return out


# --------------------------------------------------------------------------- leak check


def _gt_window_evidence(
    ctx: Ctx, clip: dict[str, Any], s: int, body_aa: np.ndarray, seed: int
):
    """Evidence for window ``[s, s+W)`` regenerated from GT parameters (FK with GT betas), canonical frame."""
    sl = slice(s, s + ctx.window)
    t_len = body_aa.shape[0]
    with torch.no_grad():
        j55, _ = smpl_forward_bt(
            ctx.body,
            torch.as_tensor(np.array(clip["transl"][sl])).unsqueeze(0),
            torch.as_tensor(np.array(clip["root_orient"][sl])).unsqueeze(0),
            torch.as_tensor(body_aa.reshape(t_len, 21, 3)).unsqueeze(0),
            torch.as_tensor(np.array(clip["betas"])).unsqueeze(0),
        )
    j55 = j55[0].numpy()
    obs, rig = simulate_evidence(
        j55[:, :22], j55, np.random.default_rng(seed), ctx.ev_cfg
    )
    obs, rig, _ = slice_window(obs, rig, 0, ctx.window)
    canonicalize(obs, rig)
    return obs, rig


def cmd_leak_check(ctx: Ctx) -> dict[str, Any]:
    model, mmeta = load_trained(ctx)
    ctx.body._model.to("cpu")
    seed = int(ctx.cfg["seed"])
    t_by_rel = load_cache_index(ctx.rel_dir)
    rng = np.random.default_rng(seed)
    cands = ctx.lists["val"]
    found = None
    for k in range(int(ctx.cfg["leak_check"]["max_tries"])):
        rel = cands[int(rng.integers(len(cands)))]
        if t_by_rel[rel] < ctx.window:
            continue
        clip = load_clip_memmap(rel, rel_dir=ctx.rel_dir)
        s = int(rng.integers(0, t_by_rel[rel] - ctx.window + 1))
        base_aa = np.array(clip["pose_body"][s : s + ctx.window], dtype=np.float32)
        obs0, rig0 = _gt_window_evidence(ctx, clip, s, base_aa, seed + k)
        if obs0["joint_visible"][:, list(LOWER_BODY_JOINTS)].any():
            continue
        pert = base_aa.reshape(-1, 21, 3).copy()
        pert[:, list(LEG_BODY_AA_INDICES), :] += 0.37
        obs1, rig1 = _gt_window_evidence(ctx, clip, s, pert.reshape(-1, 63), seed + k)
        if not torch.equal(obs0["joint_visible"], obs1["joint_visible"]):
            continue
        found = (rel, s, obs0, rig0, obs1, rig1)
        break
    if found is None:
        out = {"pass": False, "message": "no all-legs-hidden VAL window found"}
        print(json.dumps(out, indent=1))
        return out
    rel, s, obs0, rig0, obs1, rig1 = found
    dev = ctx.device

    def run(o, r):
        with torch.no_grad():
            return model(
                {k: v.unsqueeze(0).to(dev) for k, v in o.items()},
                {k: v.unsqueeze(0).to(dev) for k, v in r.items()},
            )

    def max_diff(a, b):
        return max(
            float((a[k] - b[k]).abs().max())
            for k in ("transl", "root_rot_6d", "body_rot_6d", "contact_logits")
        )

    out0, out1 = run(obs0, rig0), run(obs1, rig1)
    hid = ~obs0["joint_visible"]
    # Regenerated evidence: visible joints move slightly too (joints are regressed from the posed
    # mesh, so hip rotations shift pelvis/spine) - reported, not a leak criterion.
    regen_visible_diff = float(
        (obs1["joint_pos_3d"] - obs0["joint_pos_3d"])[~hid].abs().max()
    )
    # Negative control: the raw hidden slots do carry GT-dependent data (2D projections).
    kp_hidden_diff = float(
        (obs1["keypoints_2d"] - obs0["keypoints_2d"])[hid].abs().max()
    )
    # Leak test: visible slots from obs0, hidden slots from the perturbed evidence / random values.
    hmask = hid.unsqueeze(-1)
    obs_swap = {k: v.clone() for k, v in obs0.items()}
    gen = torch.Generator().manual_seed(seed)
    obs_rand = {k: v.clone() for k, v in obs0.items()}
    for k in ("joint_pos_3d", "keypoints_2d"):
        obs_swap[k] = torch.where(hmask, obs1[k], obs0[k])
        noise = (torch.rand(obs0[k].shape, generator=gen) - 0.5) * 10.0
        obs_rand[k] = torch.where(hmask, noise, obs0[k])
    obs_swap["joint_confidence"] = torch.where(
        hid, obs1["joint_confidence"], obs0["joint_confidence"]
    )
    obs_rand["joint_confidence"] = torch.where(
        hid, torch.rand(hid.shape, generator=gen), obs0["joint_confidence"]
    )
    swap_diff = max_diff(out0, run(obs_swap, rig0))
    rand_diff = max_diff(out0, run(obs_rand, rig0))
    # Positive control: move one visible joint 5 cm -> output must change.
    vis = obs0["joint_visible"]
    ti, ji = [int(x) for x in torch.nonzero(vis)[0]]
    obs_pc = {k: v.clone() for k, v in obs0.items()}
    obs_pc["joint_pos_3d"][ti, ji, 0] += 0.05
    pos_ctrl = max_diff(out0, run(obs_pc, rig0))
    # Negative controls: the forward must refuse anything but obs + rig.
    neg: dict[str, str] = {}
    b_obs = {k: v.unsqueeze(0).to(dev) for k, v in obs0.items()}
    b_rig = {k: v.unsqueeze(0).to(dev) for k, v in rig0.items()}
    tgt = {"body_aa": torch.zeros(1, ctx.window, 21, 3, device=dev)}
    attempts = {
        "targets_kwarg": lambda: model(b_obs, b_rig, targets=tgt),
        "init_pose_kwarg": lambda: model(b_obs, b_rig, init_body_aa=tgt["body_aa"]),
        "gt_key_in_obs": lambda: model(
            {**b_obs, "joints_gt_22": torch.zeros(1, ctx.window, 22, 3, device=dev)},
            b_rig,
        ),
        "pose_key_in_obs": lambda: model(
            {**b_obs, "pose_body": torch.zeros(1, ctx.window, 63, device=dev)}, b_rig
        ),
        "target_key_in_rig": lambda: model(
            b_obs, {**b_rig, "transl": torch.zeros(1, ctx.window, 3, device=dev)}
        ),
        "rig_missing_head_pose": lambda: model(
            b_obs, {k: v for k, v in b_rig.items() if k != "head_pos_world"}
        ),
        "rig_missing_gravity": lambda: model(
            b_obs, {k: v for k, v in b_rig.items() if k != "gravity_world"}
        ),
    }
    for name, fn in attempts.items():
        try:
            fn()
            neg[name] = "ACCEPTED"
        except (TypeError, ValueError) as exc:
            neg[name] = f"rejected: {type(exc).__name__}: {exc}"
    params = list(inspect.signature(model.forward).parameters)
    res = {
        "model": mmeta,
        "window": {
            "rel_path": rel,
            "start": s,
            "length": ctx.window,
            "all_lower_body_hidden": True,
        },
        "forward_parameters": params,
        "a_signature_obs_rig_only": params == ["obs", "rig"],
        "b_hidden_leg_gt_perturbation_rad": 0.37,
        "b_regenerated_evidence_visible_joint_pos_max_abs_diff_m": regen_visible_diff,
        "b_regenerated_model_output_max_abs_diff": max_diff(out0, out1),
        "b_negative_control_raw_hidden_keypoints_2d_max_abs_diff": kp_hidden_diff,
        "b_hidden_slots_from_perturbed_gt_output_max_abs_diff": swap_diff,
        "b_hidden_slots_random_output_max_abs_diff": rand_diff,
        "b_output_bit_identical": swap_diff == 0.0 and rand_diff == 0.0,
        "c_positive_control_visible_joint_5cm_output_diff": pos_ctrl,
        "d_negative_controls": neg,
    }
    res["pass"] = bool(
        res["a_signature_obs_rig_only"]
        and res["b_output_bit_identical"]
        and kp_hidden_diff > 0.0
        and pos_ctrl > 0.0
        and all(v.startswith("rejected") for v in neg.values())
    )
    print(json.dumps(res, indent=1))
    update_results(ctx, "leak_check", res)
    return res


def cmd_run(ctx: Ctx) -> None:
    for fn in (
        cmd_train_list,
        cmd_build_cache,
        cmd_verify_cache,
        cmd_preflight,
        cmd_train,
        cmd_eval_table,
        cmd_heuristic_check,
        cmd_leak_check,
    ):
        print(f"=== {fn.__name__} ===", flush=True)
        fn(ctx)


COMMANDS = {
    "train-list": cmd_train_list,
    "build-cache": cmd_build_cache,
    "verify-cache": cmd_verify_cache,
    "preflight": cmd_preflight,
    "train": cmd_train,
    "eval-table": cmd_eval_table,
    "heuristic-check": cmd_heuristic_check,
    "leak-check": cmd_leak_check,
    "run": cmd_run,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Track E3 oracle completion baselines (control)"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/e3_oracle.yaml"))
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("cmd", choices=sorted(COMMANDS))
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    COMMANDS[args.cmd](Ctx(copy.deepcopy(cfg), args.config, torch.device(args.device)))


if __name__ == "__main__":
    main()
