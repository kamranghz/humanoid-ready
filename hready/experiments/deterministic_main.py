"""Deterministic evidence-locked completion at the main budget (oracle control path).

Pre-registered in ``docs/decoupled_contact_grounding.md`` (second study). Commands: train, eval.
Reuses the ablation study's deterministic arm unchanged except for the budget.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from hready.data.amass import load_paths_config
from hready.experiments.evidence_locked_ablation import (
    EvidenceLockedContext,
    evaluate,
    learned_eval,
    model_maker,
    summarize,
    train_run,
)
from hready.experiments.oracle_baselines import load_trained, predict_clip_model, print_rows


def load_context(config_path: Path, device: torch.device) -> tuple[dict[str, Any], EvidenceLockedContext]:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    ab = Path(cfg["ablation_config"])
    abl = yaml.safe_load(ab.read_text(encoding="utf-8"))
    name = cfg["run_name"]
    abl["ablation"]["runs"][name] = abl["ablation"]["runs"][cfg["reference_run"]]
    abl["ablation"]["max_steps"] = int(cfg["max_steps"])
    abl["train"]["checkpoint_root"] = (Path(load_paths_config()["data_root"]) / cfg["checkpoint_root"]).as_posix()
    abl["paths"]["results_json"] = cfg["paths"]["results_json"]
    return cfg, EvidenceLockedContext(abl, ab, device)


def cmd_train(cfg: dict[str, Any], ctx: EvidenceLockedContext) -> dict[str, Any]:
    name = cfg["run_name"]
    rc = ctx.run_cfg(name)
    if rc["model"]["generative"] or rc["loss"]["w_phys"] != 0 or rc["train"]["max_steps"] != int(cfg["max_steps"]):
        raise RuntimeError("run config is not the pre-registered deterministic, w_phys 0, main-budget run")
    out = train_run(ctx, name)
    ctx.update("train", out)
    return out


def _reference_ckpt(cfg: dict[str, Any]) -> Path:
    return Path(load_paths_config()["data_root"]) / "checkpoints" / "evidence_locked_ablation" / cfg["reference_run"] / "best.pt"


def _checks(cfg: dict[str, Any], train: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    c = cfg["criteria"]
    hist = [{"step": h["step"], "metric": h["metric"]} for h in train["history"]]
    worst = max(hist, key=lambda h: h["metric"])
    checks = {
        "primary_skate": {
            "value": row["foot_skate_m_s"],
            "limit": c["primary_skate_m_s_max"],
            "pass": bool(row["foot_skate_m_s"] <= c["primary_skate_m_s_max"]),
        },
        "secondary_mpjpe_full_all": {
            "value": row["mpjpe_full_all_mm"],
            "limit": c["secondary_mpjpe_full_all_mm_max"],
            "pass": bool(row["mpjpe_full_all_mm"] <= c["secondary_mpjpe_full_all_mm_max"]),
        },
        "tertiary_no_divergence": {
            "value": worst["metric"],
            "at_step": worst["step"],
            "n_checkpoints": len(hist),
            "limit": c["tertiary_selection_metric_mm_max"],
            "pass": bool(
                len(hist) > 0 and all(h["metric"] <= c["tertiary_selection_metric_mm_max"] for h in hist)
            ),
        },
    }
    checks["all_pass"] = bool(all(v["pass"] for v in checks.values()))
    return checks


def cmd_eval(cfg: dict[str, Any], ctx: EvidenceLockedContext) -> dict[str, Any]:
    oracle = ctx.oracle
    name = cfg["run_name"]
    train = ctx.results().get("train")
    if train is None:
        raise RuntimeError("run `train` first")
    thr = float(oracle.cfg["eval"]["contact_threshold"])
    model_tf, meta_tf = load_trained(oracle)
    model, rc, meta = ctx.load_run(name)

    def mk_transformer(obs, rig, common):
        pred, prob, dis, n_dis = predict_clip_model(oracle, model_tf, obs, rig)
        return learned_eval(pred, prob, dis, n_dis, common, thr)

    makers = {"oracle_transformer": mk_transformer, name: model_maker(ctx, model, rc)}
    metas = {"oracle_transformer": meta_tf, name: {**meta, "run_cfg": rc}}
    ref_path = _reference_ckpt(cfg)
    if ref_path.is_file():
        from hready.train.completion_engine import load_checkpoint
        from hready.models.evidence_locked_completion import EvidenceLockedCompletion

        st = load_checkpoint(ref_path)
        ref = EvidenceLockedCompletion(**st["run_cfg"]["model"]).to(ctx.device)
        ref.load_state_dict(st["model"])
        ref.eval()
        ref_name = "deterministic_40000_steps"
        makers[ref_name] = model_maker(ctx, ref, st["run_cfg"])
        metas[ref_name] = {"checkpoint": ref_path.as_posix(), "step": st["step"], "best_selection_metric": st["best"]}
    cohorts = list(cfg["cohorts"])
    accs, wall = evaluate(ctx, "val", makers, cohorts)
    table = summarize(ctx, accs, set(makers))
    print_rows(oracle, "val", table)
    row = table[name]["all"]["metrics"]
    checks = _checks(cfg, train, row)
    out = {
        "disclaimer": ctx.disclaimer,
        "split": "val",
        "models": metas,
        "selection_history": [{"step": h["step"], "metric": h["metric"], "mpjpe_mm": h["mpjpe_mm"]} for h in train["history"]],
        "table": table,
        "criteria": checks,
        "wall_s": wall,
    }
    ctx.update("eval", out)
    print("\nAcceptance criteria")
    for k, v in checks.items():
        if k != "all_pass":
            extra = f" (step {v['at_step']})" if "at_step" in v else ""
            print(f"  {k:28s} value={v['value']:.4f} limit={v['limit']}{extra} -> {'PASS' if v['pass'] else 'FAIL'}")
    print(f"  all criteria pass: {checks['all_pass']}")
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Deterministic evidence-locked completion at the main budget (oracle control)")
    p.add_argument("--config", type=Path, default=Path("configs/deterministic_main.yaml"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("cmd", choices=["train", "eval"])
    a = p.parse_args(argv)
    cfg, ctx = load_context(a.config, torch.device(a.device))
    {"train": cmd_train, "eval": cmd_eval}[a.cmd](cfg, ctx)


if __name__ == "__main__":
    main()
