"""Train and evaluate HR-Refine with auxiliary heads (foot contact, BABEL action, root-translation intent).

One condition per run (the IMU fusion ablation: video_only, imu_only, fused)::

    python -m hready.train.hr_refine_auxiliary_heads --config configs/hr_refine_auxiliary_heads.yaml --condition fused

then, after all three conditions have run, write the result tables::

    python -m hready.train.hr_refine_auxiliary_heads --config configs/hr_refine_auxiliary_heads.yaml --report
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from hready.body.batch_forward import smpl_forward_bt
from hready.data.refine_auxiliary_dataset import (
    IGNORE_INDEX,
    AuxiliaryWindowDataset,
    action_vocabulary,
    collate_auxiliary,
)
from hready.data.refine_corrupt import corruption_config_from_dict, unpack_pose
from hready.metrics.pose import mpjpe
from hready.train.hr_refine_engine import (
    build_model_and_body,
    build_scheduler,
    prepare_fixed_corruption,
    reconstruction_loss,
    repo_root,
)

FOOT_JOINTS = (7, 10, 8, 11)  # L ankle, L foot, R ankle, R foot (heel/toe channel order of the cache)
CONDITIONS = ("video_only", "imu_only", "fused")


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}


def _inputs(cond: dict[str, bool], fixed: Any, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor | None]:
    kp = fixed.keypoints if cond["video"] else torch.zeros_like(fixed.keypoints)
    return kp, (batch["imu"] if cond["imu"] else None)


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"f1": 2 * p * r / (p + r) if p + r else 0.0, "precision": p, "recall": r}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    body: Any,
    cfg: dict[str, Any],
    cond: dict[str, bool],
    ds_main: AuxiliaryWindowDataset,
    ds_action: AuxiliaryWindowDataset,
    device: torch.device,
) -> dict[str, Any]:
    """Fixed-window validation: refinement, contact, intent (main windows) and action (labelled windows)."""
    model.eval()
    body._model.eval()
    ccfg = corruption_config_from_dict(cfg["corruption"])
    n_obs, horizon = int(cfg["observed_frames"]), int(cfg["intent_horizon"])
    seed = int(cfg["seed"]) + 777
    bs = int(cfg["eval_batch"])
    root_out, root_in, mp_out, mp_in = [], [], [], []
    cnt = {k: np.zeros(3, dtype=np.int64) for k in ("model", "all_contact", "foot_height_rule")}
    pos_frames = n_frames = 0
    ade, fde = {"model": [], "zero": [], "constant_velocity": []}, {"model": [], "zero": [], "constant_velocity": []}
    for k, batch in enumerate(DataLoader(ds_main, batch_size=bs, collate_fn=collate_auxiliary)):
        batch = _to_device(batch, device)
        fixed = prepare_fixed_corruption(body, batch, base_seed=seed, corrupt_step=k, corrupt_cfg=ccfg)
        kp, imu = _inputs(cond, fixed, batch)
        pred = model(fixed.corrupt_pose, kp, imu=imu)
        tr, ro, ba = unpack_pose(pred)
        joints_pred, _ = smpl_forward_bt(body, tr, ro, ba, batch["betas"])
        ctr, cro, cba = unpack_pose(fixed.corrupt_pose)
        joints_corr, _ = smpl_forward_bt(body, ctr, cro, cba, batch["betas"])
        clean_t = batch["transl"]
        root_out.append((tr - clean_t).norm(dim=-1).mean(dim=-1).cpu())
        root_in.append((ctr - clean_t).norm(dim=-1).mean(dim=-1).cpu())
        jc = fixed.joints_clean[..., :22, :].cpu().numpy()
        mp_out.append(np.atleast_1d(mpjpe(joints_pred[..., :22, :].cpu().numpy(), jc, root_idx=0)))
        mp_in.append(np.atleast_1d(mpjpe(joints_corr[..., :22, :].cpu().numpy(), jc, root_idx=0)))
        lab = batch["contact"] > 0.5
        for name, hit in (
            ("model", pred["contact_logits"] > 0),
            ("all_contact", torch.ones_like(lab)),
            ("foot_height_rule", joints_corr[..., list(FOOT_JOINTS), 2] < 0.05),
        ):
            cnt[name] += np.array([(hit & lab).sum().item(), (hit & ~lab).sum().item(), (~hit & lab).sum().item()])
        pos_frames += int(lab.sum().item())
        n_frames += lab.numel()
        # intent: next `horizon` root offsets after the n_obs observed frames
        target = clean_t[:, n_obs : n_obs + horizon] - clean_t[:, n_obs - 1 : n_obs]
        vel = (ctr[:, n_obs - 1] - ctr[:, n_obs - 6]) / 5.0
        steps = torch.arange(1, horizon + 1, device=device, dtype=vel.dtype).view(1, -1, 1)
        for name, p in (
            ("model", model.forecast(fixed.corrupt_pose, kp, n_obs, imu)),
            ("zero", torch.zeros_like(target)),
            ("constant_velocity", vel.unsqueeze(1) * steps),
        ):
            d = (p - target).norm(dim=-1)
            ade[name].append(d.mean(dim=-1).cpu())
            fde[name].append(d[:, -1].cpu())
    correct = total = base_zero = 0
    for batch in DataLoader(ds_action, batch_size=bs, collate_fn=collate_auxiliary):
        batch = _to_device(batch, device)
        fixed = prepare_fixed_corruption(body, batch, base_seed=seed, corrupt_step=10_000, corrupt_cfg=ccfg)
        kp, imu = _inputs(cond, fixed, batch)
        logits = model(fixed.corrupt_pose, kp, imu=imu)["action_logits"]
        mask = batch["action"] != IGNORE_INDEX
        correct += int((logits.argmax(-1)[mask] == batch["action"][mask]).sum().item())
        base_zero += int((batch["action"][mask] == 0).sum().item())
        total += int(mask.sum().item())
    cat = lambda xs: float(torch.cat(xs).mean()) if isinstance(xs[0], torch.Tensor) else float(np.concatenate(xs).mean())
    n_cls = len(action_vocabulary())
    return {
        "refine": {
            "root_ade_m": cat(root_out), "corrupt_root_ade_m": cat(root_in),
            "mpjpe_mm": cat(mp_out), "corrupt_mpjpe_mm": cat(mp_in),
        },
        "contact": {
            **{name: _prf(*map(int, c)) for name, c in cnt.items()},
            "positive_rate": pos_frames / max(n_frames, 1), "frame_channels": n_frames,
        },
        "action": {
            "accuracy": correct / max(total, 1), "uniform_baseline": 1.0 / n_cls,
            "always_first_class_baseline": base_zero / max(total, 1),
            "labelled_frames": total, "classes": n_cls,
        },
        "intent": {
            "ade_m": cat(ade["model"]), "fde_m": cat(fde["model"]),
            "zero_ade_m": cat(ade["zero"]), "zero_fde_m": cat(fde["zero"]),
            "constant_velocity_ade_m": cat(ade["constant_velocity"]),
            "constant_velocity_fde_m": cat(fde["constant_velocity"]),
            "windows": len(ds_main), "horizon_frames": horizon,
        },
    }


def train_condition(cfg: dict[str, Any], name: str) -> None:
    """Train one IMU-fusion condition for ``max_steps`` and save the final checkpoint with its metrics."""
    cond = cfg["conditions"][name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    window, n_obs, horizon = int(cfg["window"]), int(cfg["observed_frames"]), int(cfg["intent_horizon"])
    n_cls = len(action_vocabulary())
    model_cfg = {
        **cfg["model"],
        "imu_dim": int(cfg["imu_dim"]) if cond["imu"] else 0,
        "contact_channels": int(cfg["contact_channels"]),
        "n_actions": n_cls,
        "intent_horizon": horizon,
    }
    model, body, _ = build_model_and_body(model_cfg, device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    max_steps = int(cfg["max_steps"])
    sched = build_scheduler(opt, max_steps, int(cfg["warmup_steps"]))
    ccfg = corruption_config_from_dict(cfg["corruption"])
    w = cfg["loss"]
    noise = float(cfg["imu_noise_std"])
    train_ds = AuxiliaryWindowDataset(
        "train", window, seed=seed, labeled_fraction=float(cfg["labeled_fraction"]), imu_noise_std=noise
    )
    ds_main = AuxiliaryWindowDataset("val", window, seed=seed + 11, plan_windows=int(cfg["eval_windows"]), imu_noise_std=noise)
    ds_action = AuxiliaryWindowDataset(
        "val", window, seed=seed + 22, plan_windows=int(cfg["eval_action_windows"]), imu_noise_std=noise, only_labeled=True
    )
    loader = DataLoader(train_ds, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=0,
                        collate_fn=collate_auxiliary, drop_last=True)
    it = iter(loader)
    print(f"[{name}] params={model.num_parameters()} train clips={len(train_ds.entries)} "
          f"(BABEL-labelled {len(train_ds.labeled)}) device={device}", flush=True)
    t0 = time.perf_counter()
    metrics: dict[str, Any] = {}
    for step in range(max_steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch = _to_device(batch, device)
        model.train()
        fixed = prepare_fixed_corruption(body, batch, base_seed=seed, corrupt_step=step, corrupt_cfg=ccfg)
        kp, imu = _inputs(cond, fixed, batch)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            pred = model(fixed.corrupt_pose, kp, imu=imu)
            fut = model.forecast(fixed.corrupt_pose, kp, n_obs, imu)
        tr, ro, ba = unpack_pose(pred)
        joints_pred, _ = smpl_forward_bt(body, tr, ro, ba, batch["betas"])
        recon = reconstruction_loss(pred, fixed.clean_packed, joints_pred, fixed.joints_clean,
                                    w_joint=float(w["recon_joint"]), w_rot=float(w["recon_rot"]))
        contact = F.binary_cross_entropy_with_logits(pred["contact_logits"].float(), batch["contact"])
        mask = batch["action"] != IGNORE_INDEX
        action = (F.cross_entropy(pred["action_logits"].float()[mask], batch["action"][mask])
                  if mask.any() else recon.new_zeros(()))
        clean_t = batch["transl"]
        intent = (fut.float() - (clean_t[:, n_obs : n_obs + horizon] - clean_t[:, n_obs - 1 : n_obs])).abs().mean()
        loss = recon.float() + float(w["contact"]) * contact + float(w["action"]) * action + float(w["intent"]) * intent
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip_norm"]))
        opt.step()
        sched.step()
        if step % int(cfg["log_every"]) == 0:
            print(f"[{name}] step={step} loss={loss.item():.4f} recon={recon.item():.4f} contact={contact.item():.3f} "
                  f"action={action.item():.3f} intent={intent.item():.4f}", flush=True)
        if (step + 1) % int(cfg["eval_every"]) == 0 or step + 1 == max_steps:
            metrics = evaluate(model, body, cfg, cond, ds_main, ds_action, device)
            print(f"[{name}] step={step + 1} val root_ade={metrics['refine']['root_ade_m']*100:.2f}cm "
                  f"mpjpe={metrics['refine']['mpjpe_mm']:.2f}mm contact_f1={metrics['contact']['model']['f1']:.3f} "
                  f"action_acc={metrics['action']['accuracy']:.3f} intent_ade={metrics['intent']['ade_m']*100:.2f}cm", flush=True)
    out = repo_root() / cfg["checkpoint_dir"] / name
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"condition": name, "step": max_steps, "config": cfg, "val_metrics": metrics,
                "model": model.state_dict(), "wall_s": time.perf_counter() - t0}, out / "last.pt")
    print(f"[{name}] done steps={max_steps} wall_s={time.perf_counter() - t0:.0f}", flush=True)


def write_reports(cfg: dict[str, Any]) -> None:
    """Write contact_f1.txt and action_intent.txt from the saved last.pt metrics of the three conditions."""
    root = repo_root()
    runs = {n: torch.load(root / cfg["checkpoint_dir"] / n / "last.pt", map_location="cpu", weights_only=False)
            for n in CONDITIONS}
    m = {n: r["val_metrics"] for n, r in runs.items()}
    steps = runs["fused"]["step"]
    any_m = m["fused"]
    c = any_m["contact"]
    lines = [
        "HR-Refine foot-contact head: validation F1",
        f"Trained {steps} steps on AMASS train windows (grounded_motion_30hz cache), seed {cfg['seed']}, one run per condition.",
        f"Validation: {cfg['eval_windows']} fixed random val windows of {cfg['window']} frames, corrupted inputs "
        f"(fixed seeds); micro-averaged over the 4 foot channels (L heel, L toe, R heel, R toe); "
        f"{c['frame_channels']} frame-channels, positive rate {c['positive_rate']:.3f}.",
        "Labels: the cache's foot-channel contact labels (height + velocity rule, hready/data/contact.py).",
        "",
        f"{'condition / baseline':34s}{'F1':>8s}{'precision':>11s}{'recall':>9s}",
    ]
    for n in CONDITIONS:
        x = m[n]["contact"]["model"]
        lines.append(f"{'head: ' + n:34s}{x['f1']:8.3f}{x['precision']:11.3f}{x['recall']:9.3f}")
    for key, label in (("all_contact", "baseline: always contact"), ("foot_height_rule", "baseline: corrupted foot z<5cm")):
        x = c[key]
        lines.append(f"{label:34s}{x['f1']:8.3f}{x['precision']:11.3f}{x['recall']:9.3f}")
    d = root / "results" / "contact_torques"
    d.mkdir(parents=True, exist_ok=True)
    (d / "contact_f1.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    a, it = any_m["action"], any_m["intent"]
    out = [
        "HR-Refine action and intent heads, and IMU fusion ablation",
        f"Trained {steps} steps on AMASS train windows (grounded_motion_30hz cache), seed {cfg['seed']}, one run per condition; "
        "no confidence intervals.",
        "",
        f"Action head (BABEL frame labels, {a['classes']} categories, single label per frame = most frequent category present)",
        f"  validation: {cfg['eval_action_windows']} fixed val windows from BABEL-labelled clips, {a['labelled_frames']} labelled frames",
        f"  baselines: uniform {a['uniform_baseline']:.3f}, always first class ('{action_vocabulary()[0]}') {a['always_first_class_baseline']:.3f}",
    ]
    for n in CONDITIONS:
        out.append(f"  accuracy, {n}: {m[n]['action']['accuracy']:.3f}")
    out += [
        "",
        f"Intent head (next {it['horizon_frames']} frames of root translation after {cfg['observed_frames']} observed frames, metres)",
        f"  validation: {it['windows']} fixed random val windows",
        f"  baselines: zero motion ADE {it['zero_ade_m']:.3f} / FDE {it['zero_fde_m']:.3f}; "
        f"constant velocity ADE {it['constant_velocity_ade_m']:.3f} / FDE {it['constant_velocity_fde_m']:.3f}",
    ]
    for n in CONDITIONS:
        out.append(f"  head, {n}: ADE {m[n]['intent']['ade_m']:.3f} / FDE {m[n]['intent']['fde_m']:.3f}")
    out += [
        "",
        "IMU fusion ablation (ADE on root translation of the refined output over the 64-frame window, metres)",
        f"  input: corrupted pose always; video = 2D keypoints; imu = 6-sensor kinematic accelerometer, "
        f"noise std {cfg['imu_noise_std']} m/s^2",
        f"  {'condition':14s}{'root ADE (m)':>14s}{'MPJPE (mm)':>13s}",
    ]
    for n in CONDITIONS:
        r = m[n]["refine"]
        out.append(f"  {n:14s}{r['root_ade_m']:14.4f}{r['mpjpe_mm']:13.2f}")
    r = any_m["refine"]
    out.append(f"  {'corrupted input':14s}{r['corrupt_root_ade_m']:14.4f}{r['corrupt_mpjpe_mm']:13.2f}")
    out += [
        "",
        "Notes",
        "  - The synthetic IMU is computed from the clean (uncorrupted) motion plus Gaussian noise, so it is cleaner than the",
        "    corrupted pose input; it is a kinematic accelerometer in world axes (no gyroscope, no sensor-frame rotation).",
        "  - The intent head sees only the first observed frames; the last observed frame's IMU feature is zeroed because it",
        "    is a second difference that includes the first future frame.",
        "  - The imu_only intent gain is large while fused shows almost none; the cause was not investigated.",
        "  - The refined root ADE barely improves on the corrupted input in all conditions (training is short).",
        "  - Multi-task loss weights were not tuned; one seed per condition.",
    ]
    d = root / "results" / "auxiliary_heads"
    d.mkdir(parents=True, exist_ok=True)
    (d / "action_intent.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(lines + [""] + out))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--condition", choices=CONDITIONS, default=None)
    ap.add_argument("--report", action="store_true", help="write the result tables from the saved runs")
    ap.add_argument("--max-steps", type=int, default=None, help="override max_steps (smoke runs)")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
    if args.report:
        write_reports(cfg)
    elif args.condition:
        train_condition(cfg, args.condition)
    else:
        ap.error("give --condition or --report")


if __name__ == "__main__":
    main()
