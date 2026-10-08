"""Track E4-v2 CLI (post-hoc design after E4-v1; config ``configs/e4_v2_completion.yaml``, frozen rule).

Commands: verify-occ, preflight, train, eval-val, decide, eval-test, leak-check.
Reuses the E4-v1 / E3 code paths; v1 and E3 outputs are not touched.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from hready.baselines.e3_heuristic import heuristic_foot_contact
from hready.data.amass import load_paths_config
from hready.eval.e3_cohorts import CohortMasks
from hready.eval.e3_oracle import (
    load_trained,
    predict_clip_heuristic,
    predict_clip_model,
    print_rows,
)
from hready.eval.e4_completion import (
    E4Ctx,
    _common,
    _leak_check_run,
    cmd_verify_occ,
    deep_merge,
    learned_eval,
    make_loader,
    make_optim,
    model_maker,
    selection_eval,
    selection_items,
    step_fn,
    summarize,
    train_run,
)
from hready.metrics.e3_eval import (
    ClipEval,
    CohortAccumulator,
    accumulate_clip,
    proxy_foot_contact,
)
from hready.metrics.paired_bootstrap import ece_bins, paired_bootstrap
from hready.train.e3_oracle_engine import (
    load_checkpoint,
    save_checkpoint,
    seed_everything,
)
from hready.train.e4_engine import compute_loss_e4

GATE_KEYS = [
    "mpjpe_full_hidden_mm",
    "mpjpe_full_all_mm",
    "penetration_mean_mm",
    "foot_skate_m_s",
    "contact_ece",
]
REPORT_KEYS = GATE_KEYS + [
    "mpjpe_full_visible_mm",
    "ground_consistency_violation_frac",
    "ground_consistency_violation_frac_tau_sensitivity",
]


class E4V2Ctx(E4Ctx):
    def __init__(
        self, cfg: dict[str, Any], config_path: Path, device: torch.device
    ) -> None:
        super().__init__(cfg, config_path, device)
        self.data_root = Path(load_paths_config()["data_root"])
        self.anchor = cfg["anchor"]["name"]
        self.ckpt_override: dict[str, Path] = {}

    def run_cfg(self, name: str) -> dict[str, Any]:
        rc = deep_merge(self.cfg["base"], self.cfg["arms"][name])
        return copy.deepcopy(rc)

    def ckpt_dir(self, name: str) -> Path:
        if name == self.anchor:
            return (self.data_root / self.cfg["anchor"]["checkpoint"]).parent
        return self.data_root / self.cfg["paths"]["checkpoint_root"] / name

    def load_run(self, name: str):
        if name in self.ckpt_override:
            st = load_checkpoint(self.ckpt_override[name])
            model = self.new_model(st["run_cfg"])
            model.load_state_dict(st["model"])
            model.eval()
            return (
                model,
                st["run_cfg"],
                {
                    "checkpoint": self.ckpt_override[name].as_posix(),
                    "step": st["step"],
                    "best_selection_metric": None,
                },
            )
        return super().load_run(name)


# --------------------------------------------------------------------------- preflight


def cmd_preflight(ctx: E4V2Ctx) -> dict[str, Any]:
    """Per new arm: 1-batch overfit, exact resume, leak check on the overfit model, throughput, hours."""
    pf = ctx.cfg["preflight"]
    out: dict[str, Any] = {"verify_occ": cmd_verify_occ(ctx)}
    items = selection_items(ctx)
    for name in ctx.cfg["arms"]:
        rc = ctx.run_cfg(name)
        seed_everything(int(ctx.cfg["seed"]))
        loader, n_windows = make_loader(ctx, rc, 0, num_workers=0)
        batch = next(iter(loader))
        model = ctx.new_model(rc)
        opt, _ = make_optim(model, rc)
        for (
            g
        ) in opt.param_groups:  # constant base LR (the scheduler starts at warm-up LR)
            g["lr"] = float(rc["train"]["lr"])
        gen = torch.Generator().manual_seed(0)
        first = step_fn(ctx, model, opt, None, batch, rc, gen)
        last = first
        for _ in range(int(pf["overfit_steps"]) - 1):
            last = step_fn(ctx, model, opt, None, batch, rc, gen)
        # Exact resume: save -> load -> continue must equal continuing in memory.
        path = ctx.ckpt_dir(name) / "preflight.pt"
        save_checkpoint(
            path,
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "step": int(pf["overfit_steps"]),
                "run_cfg": rc,
                "best": None,
            },
        )
        st = load_checkpoint(path)
        model_b = ctx.new_model(rc)
        opt_b, _ = make_optim(model_b, rc)
        model_b.load_state_dict(st["model"])
        opt_b.load_state_dict(st["opt"])
        for m, o in ((model, opt), (model_b, opt_b)):
            g = torch.Generator().manual_seed(1)
            torch.manual_seed(1)
            for _ in range(int(pf["resume_steps"])):
                m.eval()
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
        # Leak check on the (overfit) arm model, full E3 protocol.
        ctx.ckpt_override[name] = path
        leak = _leak_check_run(ctx, name)
        t0 = time.perf_counter()
        selection_eval(ctx, ctx.load_run(name)[0], rc, items)
        sel_s = time.perf_counter() - t0
        del ctx.ckpt_override[name]
        path.unlink()
        # Throughput with the real loader.
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
        steps = int(rc["train"]["max_steps"])
        n_sel = steps // int(rc["train"]["val_every"])
        out[name] = {
            "n_train_windows": n_windows,
            "overfit_first": {k: first[k] for k in ("loss", "joints_m", "kl_nats")},
            "overfit_last": {k: last[k] for k in ("loss", "joints_m", "kl_nats")},
            "resume_max_param_diff": resume_diff,
            "leak_check_pass": leak["pass"],
            "leak_check": leak,
            "s_per_step": sps,
            "steps_per_s": 1.0 / sps,
            "selection_eval_s": sel_s,
            "hours_total_estimate": (sps * steps + n_sel * sel_s) / 3600.0,
        }
        print(
            name,
            json.dumps(
                {k: v for k, v in out[name].items() if k != "leak_check"}, indent=1
            ),
            flush=True,
        )
    out["hours_total_estimate_all_arms"] = sum(
        out[n]["hours_total_estimate"] for n in ctx.cfg["arms"]
    )
    out["nonfinite_steps_seen"] = ctx.__dict__.get("nonfinite_steps", [])
    ctx.update("preflight", out)
    return out


# --------------------------------------------------------------------------- training


def cmd_train(ctx: E4V2Ctx) -> dict[str, Any]:
    res = ctx.results().get("train", {})
    for name in ctx.cfg["arms"]:
        ctx.__dict__["nonfinite_steps"] = []
        res[name] = {
            **train_run(ctx, name),
            "nonfinite_steps": ctx.__dict__["nonfinite_steps"],
        }
        ctx.update("train", res)
    return res


# --------------------------------------------------------------------------- evaluation


def evaluate_with_bins(
    ctx: E4V2Ctx,
    split: str,
    makers: dict[str, Callable[..., ClipEval]],
    cohorts: list[str],
):
    """v1 ``evaluate`` plus per-subject calibration bins for cohort ``all`` (for the paired bootstrap)."""
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
    bins: dict[str, dict[str, np.ndarray]] = {b: {} for b in makers}
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
                m = masks.mask(rel, clip["T"], c)
                added = accumulate_clip(
                    accs[b][c],
                    ce,
                    m,
                    sole_offsets=c3.sole,
                    tau_m=tau,
                    tau_sensitivity_m=tau_s,
                )
                if (
                    added
                    and c == "all"
                    and ce.contact_prob is not None
                    and not ce.exclude_contact
                ):
                    prev = bins[b].get(ce.subject, 0.0)
                    bins[b][ce.subject] = prev + ece_bins(
                        ce.contact_prob[m], ce.contact_gt[m]
                    )
    return accs, bins, time.perf_counter() - t0


def _makers(
    ctx: E4V2Ctx, names: list[str], *, with_fixed_rows: bool
) -> tuple[dict[str, Callable[..., ClipEval]], dict[str, Any]]:
    c3 = ctx.c3
    thr = float(c3.cfg["eval"]["contact_threshold"])
    m3, meta3 = load_trained(c3)

    def mk_e3(obs, rig, common):
        pred, prob, dis, n_dis = predict_clip_model(c3, m3, obs, rig)
        return learned_eval(pred, prob, dis, n_dis, common, thr)

    makers: dict[str, Callable[..., ClipEval]] = {}
    metas: dict[str, Any] = {"e3_learned": meta3}
    if with_fixed_rows:

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

        makers["heuristic"] = mk_heur
    makers["e3_learned"] = mk_e3
    for n in names:
        model, rc, meta = ctx.load_run(n)
        makers[n] = model_maker(ctx, model, rc)
        metas[n] = {
            **meta,
            "run_cfg": rc,
            **({"label": ctx.cfg["anchor"]["label"]} if n == ctx.anchor else {}),
        }
    if with_fixed_rows:

        def mk_gt(obs, rig, common):
            return ClipEval(
                pred=common["gt"],
                contact_pred=proxy_foot_contact(common["gt"], c3.sole),
                contact_prob=None,
                overlap_disagree_mm=0.0,
                n_overlap_frames=0,
                **common,
            )

        makers["gt_reference"] = mk_gt
    return makers, metas


def _paired_vs_e3(ctx: E4V2Ctx, accs, bins, names: list[str]) -> dict[str, Any]:
    dr = ctx.cfg["decision_rule"]["comparison"]
    ref = {
        "per_subject": accs["e3_learned"]["all"].per_subject,
        "ece_bins": bins["e3_learned"],
    }
    return {
        n: paired_bootstrap(
            {"per_subject": accs[n]["all"].per_subject, "ece_bins": bins[n]},
            ref,
            REPORT_KEYS,
            n_boot=int(dr["n_boot"]),
            seed=int(dr["seed"]),
        )
        for n in names
    }


def cmd_eval_val(ctx: E4V2Ctx) -> dict[str, Any]:
    names = [ctx.anchor, *ctx.cfg["arms"]]
    makers, metas = _makers(ctx, names, with_fixed_rows=True)
    cohorts = list(ctx.c3.cfg["eval"]["cohorts"])
    accs, bins, wall = evaluate_with_bins(ctx, "val", makers, cohorts)
    table = summarize(ctx, accs, {"e3_learned", *names})
    print_rows(ctx.c3, "val", table)
    paired = _paired_vs_e3(ctx, accs, bins, names)
    # Informational only (no decision uses it): does the retrained det_w0 reproduce the v1 anchor?
    repro = None
    if "det_w0" in names:
        mw, ma = table["det_w0"]["all"]["metrics"], table[ctx.anchor]["all"]["metrics"]
        repro = {
            k: {"det_w0": mw[k], "v1_anchor": ma[k], "diff": mw[k] - ma[k]}
            for k in REPORT_KEYS
        }
    out = {
        "disclaimer": ctx.disclaimer,
        "split": "val",
        "det_w0_vs_v1_anchor_informational": repro,
        "models": metas,
        "table": table,
        "paired_vs_e3_learned": paired,
        "wall_s": wall,
    }
    print(json.dumps(paired, indent=1))
    ctx.update("eval_val", out)
    return out


def apply_rule(cfg: dict[str, Any], paired: dict[str, Any]) -> dict[str, Any]:
    dr = cfg["decision_rule"]
    per_arm = {}
    for n in dr["candidates"]:
        checks = {}
        for key, gate in dr["gates"].items():
            p = paired[n][key]
            if "upper_bound_lt" in gate:
                checks[key] = {
                    "upper_bound": p["hi"],
                    "limit": gate["upper_bound_lt"],
                    "op": "<",
                    "pass": p["hi"] < gate["upper_bound_lt"],
                }
            else:
                lim = gate["upper_bound_le_frac_of_e3"] * p["reference"]
                checks[key] = {
                    "upper_bound": p["hi"],
                    "limit": lim,
                    "op": "<=",
                    "pass": p["hi"] <= lim,
                }
        per_arm[n] = {"checks": checks, "pass": all(c["pass"] for c in checks.values())}
    passing = [n for n in dr["candidates"] if per_arm[n]["pass"]]
    passing.sort(
        key=lambda n: (
            paired[n]["mpjpe_full_hidden_mm"]["model"],
            paired[n]["penetration_mean_mm"]["model"],
        )
    )
    return {
        "rule_frozen_on": dr["frozen_on"],
        "per_arm": per_arm,
        "passing": passing,
        "selected": passing[0] if passing else None,
        "v2_failed": not passing,
    }


def cmd_decide(ctx: E4V2Ctx) -> dict[str, Any]:
    out = apply_rule(ctx.cfg, ctx.results()["eval_val"]["paired_vs_e3_learned"])
    print(json.dumps(out, indent=1))
    ctx.update("decision", out)
    return out


def cmd_eval_test(ctx: E4V2Ctx) -> dict[str, Any]:
    """TEST once: selected arm + e3_learned only. Refuses to run twice or without a selection."""
    if "eval_test" in ctx.results():
        raise RuntimeError("TEST already evaluated for E4-v2; it is evaluated once")
    sel = ctx.results().get("decision", {}).get("selected")
    if sel is None:
        raise RuntimeError(
            "no selected arm (run `decide`; if v2 failed, TEST is not evaluated)"
        )
    makers, metas = _makers(ctx, [sel], with_fixed_rows=False)
    accs, bins, wall = evaluate_with_bins(
        ctx, "test", makers, ["all", "ordinary_locomotion", "floor_work_eligible"]
    )
    table = summarize(ctx, accs, {"e3_learned", sel})
    print_rows(ctx.c3, "test", table)
    paired = _paired_vs_e3(ctx, accs, bins, [sel])
    out = {
        "disclaimer": ctx.disclaimer,
        "split": "test",
        "note": "evaluated once; not used for any choice",
        "models": metas,
        "table": table,
        "paired_vs_e3_learned": paired,
        "wall_s": wall,
    }
    ctx.update("eval_test", out)
    return out


def cmd_leak_check(ctx: E4V2Ctx) -> dict[str, Any]:
    res = {n: _leak_check_run(ctx, n) for n in ctx.cfg["arms"]}
    res["pass"] = all(r["pass"] for r in res.values())
    ctx.update("leak_check", res)
    return res


COMMANDS = {
    "verify-occ": cmd_verify_occ,
    "preflight": cmd_preflight,
    "train": cmd_train,
    "eval-val": cmd_eval_val,
    "decide": cmd_decide,
    "eval-test": cmd_eval_test,
    "leak-check": cmd_leak_check,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Track E4-v2 (oracle control; frozen rule)"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/e4_v2_completion.yaml")
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("cmd", choices=sorted(COMMANDS))
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    COMMANDS[args.cmd](E4V2Ctx(cfg, args.config, torch.device(args.device)))


if __name__ == "__main__":
    main()
