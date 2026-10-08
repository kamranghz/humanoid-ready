"""E4-v1 training/inference: evidence lock, loss (geodesic, transl, joints, contact, KL, optional
ground-aware terms), and per-clip windowed prediction on the E3 evidence path."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from hready.data.e3_dataset import eval_window_batch, window_starts
from hready.losses.physics import (
    balance_com_in_support,
    com_from_joints,
    foot_skating,
    ground_penetration,
)
from hready.models.ego_complete_e4 import EgoCompleteMotionE4
from hready.models.ego_complete_motion import geodesic_pose_loss
from hready.train.e3_oracle_engine import (
    NeutralJointFK,
    _masked_mean,
    aa_to_matrix,
    predict_joints,
)

FOOT_CHANNELS = [7, 10, 8, 11]


def gaussian_kernel(sigma: float, radius: int) -> Tensor:
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def evidence_lock(
    joints: Tensor, obs: dict[str, Tensor], kernel: Tensor, kappa: float
) -> Tensor:
    """``J + smooth_t(vis * (obs - J)) / (smooth_t(vis) + kappa)`` per joint and axis.

    Visible stretches: output ~ temporally smoothed evidence (denoised); hidden frames near visible
    ones take a decaying share of the neighbouring residual; isolated hidden stretches keep ``J``.
    ``joints`` ``(B, T, 22, 3)`` in the window's canonical frame (same frame as ``obs``).
    """
    b, t, j, _ = joints.shape
    vis = obs["joint_visible"].to(joints.dtype)  # (B, T, J)
    res = (obs["joint_pos_3d"].to(joints.dtype) - joints) * vis.unsqueeze(-1)
    k = kernel.to(joints.device, joints.dtype).view(1, 1, -1)
    pad = kernel.numel() // 2
    num = nn.functional.conv1d(
        res.permute(0, 2, 3, 1).reshape(-1, 1, t), k, padding=pad
    )
    den = nn.functional.conv1d(vis.permute(0, 2, 1).reshape(-1, 1, t), k, padding=pad)
    num = num.reshape(b, j, 3, t).permute(0, 3, 1, 2)
    den = den.reshape(b, j, t).permute(0, 2, 1).unsqueeze(-1)
    return joints + num / (den + kappa)


def foot_points(joints: Tensor, sole_offsets: Tensor) -> Tensor:
    """Foot contact channels (L heel, L toe, R heel, R toe) with sole-proxy heights."""
    feet = joints[..., FOOT_CHANNELS, :]
    shift = torch.zeros(4, 3, device=joints.device, dtype=joints.dtype)
    shift[:, 2] = sole_offsets.to(joints.dtype)
    return feet - shift


def output_joints(
    model_out: dict[str, Tensor],
    fk: NeutralJointFK,
    obs: dict[str, Tensor],
    lock: dict[str, Any] | None,
) -> Tensor:
    j = predict_joints(fk, model_out)
    if lock is None:
        return j
    return evidence_lock(
        j,
        obs,
        gaussian_kernel(lock["sigma_frames"], lock["radius_frames"]),
        lock["kappa"],
    )


def compute_loss_e4(
    model: EgoCompleteMotionE4,
    fk: NeutralJointFK,
    batch: dict[str, Any],
    *,
    loss_w: dict[str, Any],
    lock: dict[str, Any] | None,
    sole_offsets: Tensor,
    device: torch.device,
    amp: bool,
    gen: torch.Generator | None = None,
) -> tuple[Tensor, dict[str, float]]:
    obs = {k: v.to(device, non_blocking=True) for k, v in batch["obs"].items()}
    rig = {k: v.to(device, non_blocking=True) for k, v in batch["rig"].items()}
    tgt = {k: v.to(device, non_blocking=True) for k, v in batch["targets"].items()}
    valid = batch["valid"].to(device)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
        out, kl = model.forward_train(obs, rig, tgt)
    out = {k: v.float() for k, v in out.items()}
    kl = kl.float()
    b = valid.shape[0]
    l_geo = geodesic_pose_loss(
        out["root_rot_6d"],
        out["body_rot_6d"],
        tgt["root_R"],
        aa_to_matrix(tgt["body_aa"]),
        valid,
    )
    l_tr = _masked_mean((out["transl"] - tgt["transl"]).abs(), valid)
    joints = output_joints(out, fk, obs, lock)
    l_j = _masked_mean((joints - tgt["joints_gt_22"]).norm(dim=-1), valid)
    cvalid = batch["contact_valid"].to(device).view(b, 1)
    cmask = valid & cvalid
    bce = nn.functional.binary_cross_entropy_with_logits(
        out["contact_logits"], tgt["contact"], reduction="none"
    )
    l_c = _masked_mean(bce, cmask)
    l_kl = _masked_mean(kl, valid)
    loss = (
        float(loss_w["w_geodesic"]) * l_geo
        + float(loss_w["w_transl"]) * l_tr
        + float(loss_w["w_joints"]) * l_j
        + float(loss_w["w_contact"]) * l_c
        + float(loss_w["beta_kl"]) * l_kl
    )
    stats = {
        "geodesic_rad": l_geo.item(),
        "transl_m": l_tr.item(),
        "joints_m": l_j.item(),
        "contact_bce": l_c.item(),
        "kl_nats": l_kl.item(),
    }
    w_phys = float(loss_w.get("w_phys", 0.0))
    if w_phys > 0:
        terms = loss_w["phys_terms"]
        feet = foot_points(joints, sole_offsets)
        contact = tgt["contact"] * cmask.unsqueeze(-1)
        l_pen = ground_penetration(feet[valid])
        l_sk = foot_skating(feet, contact, 30.0)
        # Balance loss loops per frame in Python: evaluated on a seeded subsample of frames.
        n_bal = int(terms["balance_frames_per_step"])
        flat = torch.nonzero(
            cmask.reshape(-1) & (contact.reshape(-1, 4).sum(-1) > 0)
        ).squeeze(1)
        if flat.numel() > 0 and float(terms["balance"]) > 0:
            pick = flat[
                torch.randperm(flat.numel(), generator=gen)[:n_bal].to(flat.device)
            ]
            com = com_from_joints(joints.reshape(-1, 22, 3)[pick])
            l_bal = balance_com_in_support(
                com.unsqueeze(0),
                feet.reshape(-1, 4, 3)[pick].unsqueeze(0),
                contact.reshape(-1, 4)[pick].unsqueeze(0),
            )
        else:
            l_bal = torch.zeros((), device=device)
        l_p = (
            float(terms["penetration"]) * l_pen
            + float(terms["skate"]) * l_sk
            + float(terms["balance"]) * l_bal
        )
        loss = loss + w_phys * l_p
        stats.update(
            phys_pen=l_pen.item(), phys_skate=l_sk.item(), phys_balance=float(l_bal)
        )
    stats["loss"] = loss.item()
    return loss, stats


@torch.no_grad()
def predict_windows_e4(
    model: EgoCompleteMotionE4,
    fk: NeutralJointFK,
    obs_w: dict[str, Tensor],
    rig_w: dict[str, Tensor],
    *,
    window: int,
    stride: int,
    lock: dict[str, Any] | None,
    device: torch.device,
    n_samples: int = 0,
    generator: torch.Generator | None = None,
    max_batch: int = 256,
) -> dict[str, Any]:
    """Canonical per-window joints (point estimate, optional prior samples) and contact probs."""
    t_len = obs_w["joint_pos_3d"].shape[0]
    starts = window_starts(t_len, window, stride)
    obs, rig, origins, valid = eval_window_batch(obs_w, rig_w, starts, window)
    model.eval()
    joints, probs, samples = [], [], []
    for i0 in range(0, len(starts), max_batch):
        sl = slice(i0, i0 + max_batch)
        o = {k: v[sl].to(device) for k, v in obs.items()}
        r = {k: v[sl].to(device) for k, v in rig.items()}
        out = {k: v.float() for k, v in model(o, r).items()}
        joints.append(output_joints(out, fk, o, lock).cpu().numpy())
        probs.append(torch.sigmoid(out["contact_logits"]).cpu().numpy())
        if n_samples:
            draws = model.sample(o, r, n_samples, generator)
            samples.append(
                np.stack(
                    [
                        output_joints({k: v.float() for k, v in d.items()}, fk, o, lock)
                        .cpu()
                        .numpy()
                        for d in draws
                    ]
                )
            )
    return {
        "joints": np.concatenate(joints),
        "probs": np.concatenate(probs),
        "samples": np.concatenate(samples, axis=1) if samples else None,
        "starts": starts,
        "origins": origins.numpy(),
        "valid": valid.numpy(),
    }
