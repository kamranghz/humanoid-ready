"""Decoupled contact grounding CLI (oracle control path): temperature-calibrated contact + foot-only grounding stage.

Pre-registered in ``docs/decoupled_contact_grounding.md``. Commands: build-tune-cache, fit-temperature, train, eval.
The completion model (an ablation run of ``configs/evidence_locked_ablation.yaml``) is frozen and unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from hready.data.amass import clip_flags, load_index, load_paths_config
from hready.data.completion_clips import assign_split, load_tune_subject_ids
from hready.data.completion_windows import (
    clip_noise_rng,
    eval_window_batch,
    simulate_evidence,
    window_starts,
)
from hready.data.motion_cache import build_memmap, load_clip_memmap
from hready.experiments.cohorts import CohortMasks
from hready.experiments.evidence_locked_ablation import EvidenceLockedContext, make_loader, summarize
from hready.losses.physics import foot_skating, ground_penetration
from hready.metrics.completion_metrics import ClipEval, CohortAccumulator, accumulate_clip
from hready.models.contact_calibration import apply_temperature, brier, ece, fit_temperature, nll
from hready.models.grounding_stage import GroundingStage
from hready.train.completion_engine import save_checkpoint, seed_everything, stitch_windows
from hready.train.evidence_locked_engine import FOOT_CHANNELS, foot_points, output_joints

FPS = 30.0
VARIANTS = ("completion_uncalibrated", "completion_calibrated", "grounded_calibrated")


# --------------------------------------------------------------------------- context and clip lists


def load_context(config_path: Path, device: torch.device) -> tuple[dict[str, Any], EvidenceLockedContext]:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    ab = Path(cfg["ablation_config"])
    ctx = EvidenceLockedContext(yaml.safe_load(ab.read_text(encoding="utf-8")), ab, device)
    return cfg, ctx


def tune_clip_lists(cfg: dict[str, Any], ctx: EvidenceLockedContext) -> tuple[list[str], list[str]]:
    """(calibration clips, monitor clips): seeded split of the tune-subject clips (never in the completion training list)."""
    tune = load_tune_subject_ids(Path(ctx.oracle.cfg["paths"]["splits_json"]))
    rels = sorted(
        e.rel_path
        for e in load_index()
        if assign_split(e) == "train"
        and f"{e.subset}/{e.subject}" in tune
        and not clip_flags(e).get("skate_flag")
        and not ctx.flags(e.rel_path).get("exclude_contact")
    )
    cc = cfg["calibration"]
    perm = np.random.default_rng(int(cc["seed"])).permutation(len(rels))
    n_cal, n_mon = int(cc["n_clips"]), int(cc["monitor_clips"])
    return [rels[i] for i in perm[:n_cal]], [rels[i] for i in perm[n_cal : n_cal + n_mon]]


def tune_evidence(ctx: EvidenceLockedContext, rel_dir: str, rel: str):
    """Clip arrays and whole-clip world evidence from the tune cache (same simulator and noise rule as the oracle path)."""
    clip = load_clip_memmap(rel, rel_dir=rel_dir)
    obs, rig = simulate_evidence(
        clip["joints_22"],
        clip["joints_55"],
        clip_noise_rng(rel, int(ctx.oracle.cfg["eval_noise_seed"])),
        ctx.oracle.ev_cfg,
    )
    return clip, obs, rig


def cmd_build_tune_cache(cfg: dict[str, Any], ctx: EvidenceLockedContext) -> dict[str, Any]:
    cal, mon = tune_clip_lists(cfg, ctx)
    cc = cfg["calibration"]
    payload = build_memmap(cal + mon, rel_dir=cc["tune_motion_rel_dir"], workers=int(cc["build_workers"]))
    out = {k: v for k, v in payload.items() if k != "entries"}
    print(json.dumps(out, indent=1))
    return out


# --------------------------------------------------------------------------- frozen completion model


@torch.no_grad()
def predict_windows(ctx: EvidenceLockedContext, model, rc, obs_w, rig_w, max_batch: int = 256) -> dict[str, Any]:
    """Canonical per-window locked joints and contact logits (point estimate), as the evaluation path computes them."""
    window, stride = ctx.oracle.window, ctx.oracle.stride
    t_len = obs_w["joint_pos_3d"].shape[0]
    starts = window_starts(t_len, window, stride)
    obs, rig, origins, valid = eval_window_batch(obs_w, rig_w, starts, window)
    model.eval()
    joints, logits = [], []
    for i0 in range(0, len(starts), max_batch):
        sl = slice(i0, i0 + max_batch)
        o = {k: v[sl].to(ctx.device) for k, v in obs.items()}
        r = {k: v[sl].to(ctx.device) for k, v in rig.items()}
        out = {k: v.float() for k, v in model(o, r).items()}
        joints.append(output_joints(out, ctx.oracle.fk, o, rc["lock"]).cpu().numpy())
        logits.append(out["contact_logits"].cpu().numpy())
    return {
        "joints": np.concatenate(joints),
        "logits": np.concatenate(logits),
        "starts": starts,
        "origins": origins.numpy(),
        "valid": valid.numpy(),
        "t_len": t_len,
    }


def stitch_joints(win: dict[str, Any]) -> tuple[np.ndarray, float, int]:
    return stitch_windows(win["joints"], win["starts"], win["valid"], win["t_len"], win["origins"])


def stitch_probs(win: dict[str, Any], temperature: float) -> np.ndarray:
    p = 1.0 / (1.0 + np.exp(-win["logits"].astype(np.float64) / float(temperature)))
    return stitch_windows(p.astype(np.float32), win["starts"], win["valid"], win["t_len"])[0]


# --------------------------------------------------------------------------- temperature scaling


def _calibration_metrics(probs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {"ece": ece(probs, labels), "brier": brier(probs, labels), "nll": nll(probs, labels)}


def cmd_fit_temperature(cfg: dict[str, Any], ctx: EvidenceLockedContext) -> dict[str, Any]:
    model, rc, meta = ctx.load_run(cfg["completion_run"])
    cal, mon = tune_clip_lists(cfg, ctx)
    cc = cfg["calibration"]
    store = []
    t0 = time.perf_counter()
    for rel in cal:
        clip, obs, rig = tune_evidence(ctx, cc["tune_motion_rel_dir"], rel)
        store.append((predict_windows(ctx, model, rc, obs, rig), np.asarray(clip["contact"], dtype=bool)))

    def pooled(temperature: float) -> tuple[np.ndarray, np.ndarray]:
        p = np.concatenate([stitch_probs(w, temperature).reshape(-1) for w, _ in store])
        y = np.concatenate([lab.reshape(-1) for _, lab in store])
        return p, y

    pre = _calibration_metrics(*pooled(1.0))
    temperature = fit_temperature(lambda t: nll(*pooled(t)), tuple(cc["log_temperature_bounds"]))
    post = _calibration_metrics(*pooled(temperature))
    out = {
        "temperature": temperature,
        "completion_model": cfg["completion_run"],
        "completion_checkpoint": meta["checkpoint"],
        "fit": "single scalar T on the contact logits, p = sigmoid(logit / T), minimising the NLL of the overlap-averaged probabilities",
        "calibration_set": {
            "source": "seeded draw from the clips of the 19 tune subjects (not in the completion training list, VAL or TEST)",
            "n_clips": len(cal),
            "seed": int(cc["seed"]),
            "n_frame_channels": int(sum(lab.size for _, lab in store)),
            "positive_rate": float(np.mean(np.concatenate([lab.reshape(-1) for _, lab in store]))),
            "before": pre,
            "after": post,
        },
        "n_monitor_clips": len(mon),
        "wall_s": time.perf_counter() - t0,
    }
    path = Path(cfg["paths"]["temperature_json"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))
    return out


def load_temperature(cfg: dict[str, Any]) -> float:
    return float(json.loads(Path(cfg["paths"]["temperature_json"]).read_text(encoding="utf-8"))["temperature"])


# --------------------------------------------------------------------------- grounding stage: loss and training


def new_stage(cfg: dict[str, Any]) -> GroundingStage:
    sc = cfg["stage"]
    return GroundingStage(
        channels=int(sc["channels"]),
        kernel=int(sc["kernel"]),
        dilations=tuple(sc["dilations"]),
        displacement_scale=float(sc["displacement_scale"]),
        fps=FPS,
    )


def apply_stage(stage: GroundingStage, joints: torch.Tensor, prob: torch.Tensor, sole: torch.Tensor):
    """Correct the foot-channel joints of ``joints`` ``(B, T, 22, 3)``; returns (corrected joints, delta)."""
    delta = stage(foot_points(joints, sole), prob)
    out = joints.clone()
    out[..., FOOT_CHANNELS, :] = out[..., FOOT_CHANNELS, :] + delta
    return out, delta


def stage_loss(
    stage: GroundingStage,
    joints: torch.Tensor,
    logits: torch.Tensor,
    gt: torch.Tensor,
    contact: torch.Tensor,
    valid: torch.Tensor,
    contact_valid: torch.Tensor,
    sole: torch.Tensor,
    temperature: float,
    w: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    prob = torch.sigmoid(logits / temperature)
    corrected, delta = apply_stage(stage, joints, prob, sole)
    feet = foot_points(corrected, sole)
    cmask = valid & contact_valid.view(-1, 1)
    c = contact * cmask.unsqueeze(-1)
    n_valid = valid.sum().clamp_min(1) * 4
    l_skate = foot_skating(feet, c, FPS)
    l_pen = ground_penetration(feet[valid])
    l_height = (feet[..., 2].abs() * c).sum() / c.sum().clamp_min(1.0)
    l_pos = ((corrected[..., FOOT_CHANNELS, :] - gt[..., FOOT_CHANNELS, :]).norm(dim=-1) * valid.unsqueeze(-1)).sum() / n_valid
    l_disp = (delta.norm(dim=-1) * valid.unsqueeze(-1)).sum() / n_valid
    loss = (
        float(w["skate"]) * l_skate
        + float(w["penetration"]) * l_pen
        + float(w["contact_height"]) * l_height
        + float(w["foot_position"]) * l_pos
        + float(w["displacement"]) * l_disp
    )
    stats = {
        "skate": l_skate.item(),
        "pen": l_pen.item(),
        "height": l_height.item(),
        "foot_pos": l_pos.item(),
        "disp": l_disp.item(),
        "loss": loss.item(),
    }
    return loss, stats


@torch.no_grad()
def frozen_batch_outputs(ctx: EvidenceLockedContext, model, rc, batch, device: torch.device):
    """Locked joints and contact logits of the frozen completion model for one training batch (canonical frame)."""
    obs = {k: v.to(device, non_blocking=True) for k, v in batch["obs"].items()}
    rig = {k: v.to(device, non_blocking=True) for k, v in batch["rig"].items()}
    out = {k: v.float() for k, v in model(obs, rig).items()}
    joints = output_joints(out, ctx.oracle.fk, obs, rc["lock"])
    return joints, out["contact_logits"]


def build_monitor_set(cfg: dict[str, Any], ctx: EvidenceLockedContext, model, rc, clips: list[str]) -> dict[str, torch.Tensor]:
    """Frozen-model windows of the monitor clips: joints, logits, GT joints, GT contact, validity (canonical frame)."""
    window = ctx.oracle.window
    acc: dict[str, list[np.ndarray]] = {k: [] for k in ("joints", "logits", "gt", "contact", "valid", "cvalid")}
    for rel in clips:
        clip, obs, rig = tune_evidence(ctx, cfg["calibration"]["tune_motion_rel_dir"], rel)
        win = predict_windows(ctx, model, rc, obs, rig)
        gt_all = np.asarray(clip["joints_22"], dtype=np.float32)
        ct_all = np.asarray(clip["contact"], dtype=np.float32)
        for k, s in enumerate(win["starts"]):
            n = int(win["valid"][k].sum())
            gt = np.zeros((window, 22, 3), np.float32)
            ct = np.zeros((window, 4), np.float32)
            gt[:n] = gt_all[s : s + n]
            gt[:n, :, 0] -= win["origins"][k, 0]
            gt[:n, :, 1] -= win["origins"][k, 1]
            ct[:n] = ct_all[s : s + n]
            acc["joints"].append(win["joints"][k])
            acc["logits"].append(win["logits"][k])
            acc["gt"].append(gt)
            acc["contact"].append(ct)
            acc["valid"].append(win["valid"][k])
            acc["cvalid"].append(np.asarray(True))
    return {k: torch.as_tensor(np.stack(v)) for k, v in acc.items()}


@torch.no_grad()
def monitor_loss(stage: GroundingStage, mon: dict[str, torch.Tensor], ctx, temperature: float, w, device) -> dict[str, float]:
    stage.eval()
    sums: dict[str, float] = {}
    n = mon["joints"].shape[0]
    bs = 128
    for i0 in range(0, n, bs):
        sl = slice(i0, i0 + bs)
        b = {k: v[sl].to(device) for k, v in mon.items()}
        _, st = stage_loss(stage, b["joints"], b["logits"], b["gt"], b["contact"], b["valid"].bool(),
                           b["cvalid"].bool(), ctx.sole_t, temperature, w)
        for k, v in st.items():
            sums[k] = sums.get(k, 0.0) + v * b["joints"].shape[0]
    stage.train()
    return {k: v / n for k, v in sums.items()}


def cmd_train(cfg: dict[str, Any], ctx: EvidenceLockedContext, *, max_steps: int | None = None, out_dir: Path | None = None) -> dict[str, Any]:
    device = ctx.device
    seed_everything(int(cfg["seed"]))
    model, rc, meta = ctx.load_run(cfg["completion_run"])
    for p in model.parameters():
        p.requires_grad_(False)
    temperature = load_temperature(cfg)
    tc, w = cfg["train"], cfg["loss"]
    total = int(max_steps if max_steps is not None else tc["max_steps"])
    stage = new_stage(cfg).to(device)
    opt = torch.optim.AdamW(stage.parameters(), lr=float(tc["lr"]), weight_decay=0.0)
    warm, floor = int(tc["warmup_steps"]), float(tc["lr_floor_fraction"])

    def lr_at(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        p = min(1.0, (step - warm) / max(1, total - warm))
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    _, mon_clips = tune_clip_lists(cfg, ctx)
    mon = build_monitor_set(cfg, ctx, model, rc, mon_clips)
    history = [{"step": 0, **monitor_loss(stage, mon, ctx, temperature, w, device)}]
    print(f"[stage] params={sum(p.numel() for p in stage.parameters())} receptive_field={stage.receptive_field} "
          f"monitor_windows={mon['joints'].shape[0]} T={temperature:.4f} step0 monitor={history[0]}", flush=True)
    loader, n_windows = make_loader(ctx, rc, 0, int(tc["num_workers"]))
    print(f"[stage] training windows={n_windows} steps={total}", flush=True)
    t0 = time.perf_counter()
    acc: list[dict[str, float]] = []
    step = 0
    diverged = False
    for batch in loader:
        joints, logits = frozen_batch_outputs(ctx, model, rc, batch, device)
        tg = batch["targets"]
        loss, st = stage_loss(
            stage, joints, logits, tg["joints_gt_22"].to(device), tg["contact"].to(device),
            batch["valid"].to(device), batch["contact_valid"].to(device), ctx.sole_t, temperature, w,
        )
        if not torch.isfinite(loss):
            diverged = True
            print(f"[stage] non-finite loss at step {step}: {st}", flush=True)
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(stage.parameters(), float(tc["grad_clip"]))
        opt.step()
        sched.step()
        acc.append(st)
        step += 1
        if step % int(tc["log_every"]) == 0:
            mean = {k: float(np.mean([s[k] for s in acc])) for k in acc[0]}
            acc = []
            print(f"[stage] step {step} " + " ".join(f"{k}={v:.5f}" for k, v in mean.items())
                  + f" lr={sched.get_last_lr()[0]:.2e} wall_s={time.perf_counter() - t0:.0f}", flush=True)
        if step % int(tc["monitor_every"]) == 0 or step == total:
            rec = {"step": step, **monitor_loss(stage, mon, ctx, temperature, w, device)}
            history.append(rec)
            print(f"[stage][monitor] {rec}", flush=True)
        if step >= total:
            break
    del loader
    final = history[-1]
    finite = all(np.isfinite(v) for h in history for v in h.values())
    summary = {
        "steps": step,
        "diverged": bool(diverged or not finite or final["loss"] > history[0]["loss"]),
        "divergence_rule": "any non-finite loss, or monitor loss at the last step above its step-0 value (identity stage)",
        "monitor_history": history,
        "monitor_clips": len(mon_clips),
        "n_train_windows": n_windows,
        "temperature": temperature,
        "wall_s": time.perf_counter() - t0,
        "completion_checkpoint": meta["checkpoint"],
    }
    root = out_dir or Path(cfg["train"]["checkpoint_dir"] or Path(load_paths_config()["data_root"]) / "checkpoints" / "contact_grounding")
    save_checkpoint(root / "stage.pt", {"stage": stage.state_dict(), "cfg": cfg, "summary": summary, "step": step})
    print(json.dumps({k: v for k, v in summary.items() if k != "monitor_history"}, indent=1), flush=True)
    return summary


def load_stage(cfg: dict[str, Any], device: torch.device, path: Path | None = None) -> tuple[GroundingStage, dict[str, Any]]:
    root = path or Path(cfg["train"]["checkpoint_dir"] or Path(load_paths_config()["data_root"]) / "checkpoints" / "contact_grounding")
    st = torch.load(root / "stage.pt", map_location="cpu", weights_only=False)
    stage = new_stage(cfg).to(device)
    stage.load_state_dict(st["stage"])
    stage.eval()
    return stage, st["summary"]


# --------------------------------------------------------------------------- evaluation


def _clip_eval(pred, prob, dis, n_dis, common, thr) -> ClipEval:
    return ClipEval(pred=pred, contact_pred=prob > thr, contact_prob=prob, overlap_disagree_mm=dis,
                    n_overlap_frames=n_dis, **common)


def cmd_eval(cfg: dict[str, Any], ctx: EvidenceLockedContext) -> dict[str, Any]:
    """VAL `all` (+ `ordinary_locomotion`): the frozen model, calibrated, and calibrated + grounded; one pass, one evaluation."""
    device = ctx.device
    oracle = ctx.oracle
    model, rc, meta = ctx.load_run(cfg["completion_run"])
    stage, train_summary = load_stage(cfg, device)
    temperature = load_temperature(cfg)
    thr = float(oracle.cfg["eval"]["contact_threshold"])
    ev = oracle.cfg["eval"]
    tau, tau_s = float(ev["ground_consistency_tolerance_m"]), float(ev["ground_consistency_tolerance_sensitivity_m"])
    cohorts = ["all", "ordinary_locomotion"]
    rels = oracle.lists["val"]
    masks = CohortMasks(Path(oracle.cfg["paths"]["ego_splits_yaml"]), Path(oracle.cfg["paths"]["floor_work_csv"]), rels)
    accs = {v: {c: CohortAccumulator() for c in cohorts} for v in VARIANTS}
    t0 = time.perf_counter()
    for i, rel in enumerate(rels):
        if i % 250 == 0:
            print(f"eval val {i}/{len(rels)} elapsed_s={time.perf_counter() - t0:.0f}", flush=True)
        clip, obs, rig = ctx.evidence(rel)
        entry, fl = ctx.entries[rel], ctx.flags(rel)
        common = {
            "rel_path": rel, "subject": f"{entry.subset}/{entry.subject}",
            "gt": np.asarray(clip["joints_22"], dtype=np.float32),
            "visible": obs["joint_visible"].numpy().astype(bool),
            "contact_gt": np.asarray(clip["contact"], dtype=bool),
            "exclude_contact": bool(fl.get("exclude_contact")),
            "exclude_physical_eval": bool(fl.get("exclude_physical_eval")),
        }
        win = predict_windows(ctx, model, rc, obs, rig)
        pred, dis, n_dis = stitch_joints(win)
        prob_raw, prob_cal = stitch_probs(win, 1.0), stitch_probs(win, temperature)
        with torch.no_grad():
            j = torch.as_tensor(pred, device=device).unsqueeze(0)
            grounded, _ = apply_stage(stage, j, torch.as_tensor(prob_cal, device=device).unsqueeze(0), ctx.sole_t)
        evs = {
            "completion_uncalibrated": _clip_eval(pred, prob_raw, dis, n_dis, common, thr),
            "completion_calibrated": _clip_eval(pred, prob_cal, dis, n_dis, common, thr),
            "grounded_calibrated": _clip_eval(grounded[0].cpu().numpy().astype(np.float32), prob_cal, dis, n_dis, common, thr),
        }
        for v, ce in evs.items():
            for c in cohorts:
                accumulate_clip(accs[v][c], ce, masks.mask(rel, clip["T"], c), sole_offsets=oracle.sole,
                                tau_m=tau, tau_sensitivity_m=tau_s)
    table = summarize(ctx, accs, set(VARIANTS))
    for v in VARIANTS:
        a = accs[v]["all"]
        p, y = np.concatenate(a.probs), np.concatenate(a.labels)
        table[v]["all"]["metrics"]["contact_brier"] = brier(p, y)
    res = _checks(cfg, table)
    out = {
        "disclaimer": ctx.disclaimer,
        "split": "val",
        "n_clips": len(rels),
        "completion_model": cfg["completion_run"],
        "completion_checkpoint": meta["checkpoint"],
        "temperature": temperature,
        "stage_training": train_summary,
        "table": table,
        **res,
        "wall_s": time.perf_counter() - t0,
    }
    path = Path(cfg["paths"]["results_json"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8")
    tpath = Path(cfg["paths"]["temperature_json"])
    tj = json.loads(tpath.read_text(encoding="utf-8"))
    tj["val_all"] = {
        "n_clips": table["completion_uncalibrated"]["all"]["n_clips"],
        "before": {k: table["completion_uncalibrated"]["all"]["metrics"][f"contact_{k}"] for k in ("ece", "brier")},
        "after": {k: table["completion_calibrated"]["all"]["metrics"][f"contact_{k}"] for k in ("ece", "brier")},
    }
    tpath.write_text(json.dumps(tj, indent=1) + "\n", encoding="utf-8")
    print_report(out)
    return out


def _checks(cfg: dict[str, Any], table: dict[str, Any]) -> dict[str, Any]:
    crit = cfg["criteria"]
    g = table["grounded_calibrated"]["all"]["metrics"]
    ref = json.loads(Path(cfg["paths"]["reference_json"]).read_text(encoding="utf-8"))["eval_table"]["splits"]["val"]
    unc = table["completion_uncalibrated"]["all"]["metrics"]
    rec = ref["evidence_locked_same_budget"]["all"]["metrics"]
    checks = {
        "primary_skate": {"value_m_s": g["foot_skate_m_s"], "limit": crit["skate_max_m_s"],
                          "pass": bool(g["foot_skate_m_s"] <= crit["skate_max_m_s"])},
        "secondary_contact_ece": {"value": g["contact_ece"], "limit": crit["contact_ece_max"],
                                  "pass": bool(g["contact_ece"] <= crit["contact_ece_max"])},
        "tertiary_full_mpjpe": {"value_mm": g["mpjpe_full_all_mm"], "limit": crit["full_mpjpe_max_mm"],
                                "pass": bool(g["mpjpe_full_all_mm"] <= crit["full_mpjpe_max_mm"])},
    }
    reproduction = {k: {"this_run": unc[k], "recorded": rec[k]} for k in
                    ("mpjpe_full_all_mm", "foot_skate_m_s", "penetration_mean_mm", "contact_ece")}
    return {
        "criteria": checks,
        "all_criteria_pass": all(c["pass"] for c in checks.values()),
        "reference_values": {
            "oracle_transformer": {k: ref["oracle_transformer"]["all"]["metrics"][k] for k in
                                   ("mpjpe_full_all_mm", "foot_skate_m_s", "penetration_mean_mm", "contact_ece")},
            "evidence_locked_same_budget_recorded": {k: rec[k] for k in
                                                     ("mpjpe_full_all_mm", "foot_skate_m_s", "penetration_mean_mm", "contact_ece")},
        },
        "reproduction_of_recorded_same_budget_row": reproduction,
    }


def print_report(out: dict[str, Any]) -> None:
    keys = ("mpjpe_full_all_mm", "mpjpe_full_visible_mm", "mpjpe_full_hidden_mm", "penetration_mean_mm", "foot_skate_m_s",
            "ground_consistency_violation_frac", "contact_f1", "contact_ece", "contact_brier")
    print("\nVAL all (2223 clips)")
    print(f"{'variant':26s}" + "".join(f"{k.replace('_mm','').replace('_m_s','')[:14]:>15s}" for k in keys))
    for v, by in out["table"].items():
        m = by["all"]["metrics"]
        print(f"{v:26s}" + "".join(f"{m.get(k, float('nan')):15.4f}" for k in keys))
    print("\nAcceptance criteria")
    for name, c in out["criteria"].items():
        val = c.get("value_m_s", c.get("value", c.get("value_mm")))
        print(f"  {name:24s} value={val:.4f} limit={c['limit']} -> {'PASS' if c['pass'] else 'FAIL'}")
    print(f"  all criteria pass: {out['all_criteria_pass']}")


COMMANDS = {"build-tune-cache": cmd_build_tune_cache, "fit-temperature": cmd_fit_temperature,
            "train": cmd_train, "eval": cmd_eval}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Decoupled contact grounding (oracle control)")
    ap.add_argument("--config", type=Path, default=Path("configs/contact_grounding.yaml"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("cmd", choices=sorted(COMMANDS))
    args = ap.parse_args(argv)
    cfg, ctx = load_context(args.config, torch.device(args.device))
    COMMANDS[args.cmd](cfg, ctx)


if __name__ == "__main__":
    main()
