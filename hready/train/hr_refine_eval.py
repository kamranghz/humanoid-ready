"""Fixed-window val eval for HR-Refine (item 8 gate + ablation)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from hready.body.batch_forward import smpl_forward_bt
from hready.body.smplx_wrapper import SmplxBody
from hready.data.amass import load_clip
from hready.data.refine_corrupt import (
    CorruptionConfig,
    apply_corruption,
    make_corruption_rng,
    unpack_pose,
)
from hready.data.refine_dataset import collate_windows
from hready.data.refine_eligible_cache import eligible_entries
from hready.metrics.physical import foot_skate, ground_penetration, jitter
from hready.metrics.pose import mpjpe, pa_mpjpe


@dataclass
class WindowMetrics:
    mpjpe_mm: float
    pa_mpjpe_mm: float
    foot_skate: float
    penetration_mm: float
    jitter: float
    subject: str


def _window_batch(entry_idx: int, offset: int, window: int, split: str = "val") -> dict[str, Any]:
    entry = eligible_entries(split)[entry_idx]
    clip = load_clip(entry, ground=True)
    t = clip["root_orient"].shape[0]
    i0 = min(offset, max(0, t - window))
    root = clip["root_orient"][i0 : i0 + window]
    body = clip["pose_body"][i0 : i0 + window]
    transl = clip["transl"][i0 : i0 + window]
    if root.shape[0] < window:
        pad = window - root.shape[0]
        root = np.concatenate([root, np.repeat(root[-1:], pad, axis=0)], axis=0)
        body = np.concatenate([body, np.repeat(body[-1:], pad, axis=0)], axis=0)
        transl = np.concatenate([transl, np.repeat(transl[-1:], pad, axis=0)], axis=0)
    return {
        "rel_path": entry.rel_path,
        "subject": entry.subject,
        "root_orient": torch.as_tensor(root, dtype=torch.float32),
        "pose_body": torch.as_tensor(body, dtype=torch.float32),
        "transl": torch.as_tensor(transl, dtype=torch.float32),
        "betas": torch.as_tensor(clip["betas"], dtype=torch.float32),
        "fps": float(clip["fps"]),
    }


def eval_plan(n_windows: int, eval_seed: int, split: str = "val") -> list[tuple[int, int, str]]:
    """(entry_idx, time_offset, subject) per window."""
    entries = eligible_entries(split)
    rng = np.random.default_rng(eval_seed)
    plan: list[tuple[int, int, str]] = []
    for _ in range(n_windows):
        ei = int(rng.integers(0, len(entries)))
        entry = entries[ei]
        clip = load_clip(entry, ground=True)
        t = clip["root_orient"].shape[0]
        i0 = int(rng.integers(0, max(1, t - 63))) if t > 63 else 0
        plan.append((ei, i0, entry.subject))
    return plan


def _metrics_from_joints(
    joints_pred: np.ndarray,
    joints_clean: np.ndarray,
    verts_pred: np.ndarray,
    fps: float,
) -> WindowMetrics:
    j22p = joints_pred[..., :22, :]
    j22c = joints_clean[..., :22, :]
    mp = float(np.nanmean(mpjpe(j22p, j22c, root_idx=0)))
    pa = float(np.nanmean(pa_mpjpe(j22p, j22c)))
    fp = joints_pred[0, :, [7, 10, 8, 11], :]
    ct = (fp[..., 2] < 0.05).astype(np.float32)
    pen = ground_penetration(verts_pred[0])[0] * 1000.0
    return WindowMetrics(
        mpjpe_mm=mp,
        pa_mpjpe_mm=pa,
        foot_skate=foot_skate(fp, ct, fps),
        penetration_mm=float(pen),
        jitter=jitter(joints_pred, fps),
        subject="",
    )


def eval_windows(
    *,
    body: SmplxBody,
    corrupt_cfg: CorruptionConfig,
    plan: list[tuple[int, int, str]],
    window: int,
    device: torch.device,
    model: nn.Module | None = None,
    mode: str = "corrupt",
    base_seed: int = 0,
    corrupt_step: int = 0,
) -> list[WindowMetrics]:
    """mode: corrupt | output | clean (GT joints, zero error)."""
    body._model.eval()
    if model is not None:
        model.eval()
    out: list[WindowMetrics] = []
    for k, (ei, i0, subject) in enumerate(plan):
        item = _window_batch(ei, i0, window)
        batch = collate_windows([item])
        batch = {key: val.to(device) if hasattr(val, "to") else val for key, val in batch.items()}
        clean = {
            "transl": batch["transl"],
            "root_aa": batch["root_orient"],
            "body_aa": batch["pose_body"],
        }
        with torch.no_grad():
            joints_clean, verts_clean = smpl_forward_bt(
                body, clean["transl"], clean["root_aa"], clean["body_aa"], batch["betas"]
            )
        if mode == "clean":
            m = _metrics_from_joints(
                joints_clean.cpu().numpy(),
                joints_clean.cpu().numpy(),
                verts_clean.cpu().numpy(),
                float(batch["fps"]),
            )
            m.subject = subject
            out.append(m)
            continue
        corr = apply_corruption(
            clean,
            joints_clean,
            rng=make_corruption_rng(base_seed, corrupt_step + k),
            cfg=corrupt_cfg,
        )
        if mode == "corrupt":
            tr, ro, ba = unpack_pose(corr["corrupt_pose"])
            with torch.no_grad():
                ji, vi = smpl_forward_bt(body, tr, ro, ba, batch["betas"])
            m = _metrics_from_joints(
                ji.cpu().numpy(), joints_clean.cpu().numpy(), vi.cpu().numpy(), float(batch["fps"])
            )
            m.subject = subject
            out.append(m)
            continue
        assert model is not None and mode == "output"
        with torch.no_grad():
            pred = model(corr["corrupt_pose"], corr["keypoints"])
            tr, ro, ba = unpack_pose(pred)
            jo, vo = smpl_forward_bt(body, tr, ro, ba, batch["betas"])
        m = _metrics_from_joints(
            jo.cpu().numpy(), joints_clean.cpu().numpy(), vo.cpu().numpy(), float(batch["fps"])
        )
        m.subject = subject
        out.append(m)
    return out


def aggregate_metrics(rows: list[WindowMetrics]) -> dict[str, float]:
    if not rows:
        return {}
    return {
        "mpjpe_mm": float(np.mean([r.mpjpe_mm for r in rows])),
        "pa_mpjpe_mm": float(np.mean([r.pa_mpjpe_mm for r in rows])),
        "foot_skate": float(np.mean([r.foot_skate for r in rows])),
        "penetration_mm": float(np.mean([r.penetration_mm for r in rows])),
        "jitter": float(np.mean([r.jitter for r in rows])),
    }
