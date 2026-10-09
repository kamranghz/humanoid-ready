"""Oracle completion transformer engine: neutral-shape joint FK, loss, inference, stitched prediction, checkpoints."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from smplx.lbs import batch_rigid_transform
from torch import Tensor, nn

from hready.body.rotations import axis_angle_to_matrix, rotation_6d_to_matrix
from hready.data.completion_windows import eval_window_batch, window_starts
from hready.losses.physics import foot_skating, ground_penetration
from hready.models.ego_complete_motion import (
    NUM_BODY,
    EgoCompleteMotion,
    geodesic_pose_loss,
)

NUM_KP = 22


class NeutralJointFK(nn.Module):
    """First 22 FK joints of locked_head SMPL-X with betas = 0 (expression 0, hands/jaw/eyes at 0).

    Same math as ``SmplxBody.forward`` + ``J_regressor`` on posed vertices, restricted to the
    vertices with non-zero 22-joint regressor weight, so it is exact and far cheaper.
    """

    def __init__(self, body: Any) -> None:
        super().__init__()
        m = body._model
        jreg = m.J_regressor.detach().float()  # (55, V)
        sel = torch.nonzero(jreg[:NUM_KP].abs().sum(dim=0) > 0).squeeze(1)
        v_t = m.v_template.detach().float()
        n_pose = m.posedirs.shape[0]
        self.register_buffer("parents", m.parents.detach().clone())
        self.register_buffer("j_rest", jreg @ v_t)  # (55, 3), shape-neutral rest joints
        self.register_buffer("v_rest", v_t[sel])
        self.register_buffer(
            "posedirs",
            m.posedirs.detach()
            .float()
            .reshape(n_pose, -1, 3)[:, sel]
            .reshape(n_pose, -1),
        )
        self.register_buffer("weights", m.lbs_weights.detach().float()[sel])
        self.register_buffer("jreg", jreg[:NUM_KP, sel])
        self.n_joints = int(jreg.shape[0])

    def forward(self, transl: Tensor, root_R: Tensor, body_R: Tensor) -> Tensor:
        """``transl`` (N,3), ``root_R`` (N,3,3), ``body_R`` (N,21,3,3) -> joints (N,22,3)."""
        n = transl.shape[0]
        eye = torch.eye(3, device=transl.device, dtype=transl.dtype)
        rest = eye.expand(n, self.n_joints - 1 - NUM_BODY, 3, 3)
        rot = torch.cat([root_R.unsqueeze(1), body_R, rest], dim=1)
        pose_feature = (rot[:, 1:] - eye).reshape(n, -1)
        v_posed = self.v_rest + (pose_feature @ self.posedirs).reshape(n, -1, 3)
        _, a = batch_rigid_transform(
            rot, self.j_rest.expand(n, -1, -1), self.parents, dtype=transl.dtype
        )
        t = (self.weights @ a.reshape(n, self.n_joints, 16)).reshape(n, -1, 4, 4)
        verts = (
            (t[..., :3, :3] @ v_posed.unsqueeze(-1)).squeeze(-1)
            + t[..., :3, 3]
            + transl.unsqueeze(1)
        )
        return self.jreg @ verts


def aa_to_matrix(aa: Tensor) -> Tensor:
    """Axis-angle ``(..., 3)`` -> ``(..., 3, 3)`` (``axis_angle_to_matrix`` takes ``(B, 3)`` only)."""
    return axis_angle_to_matrix(aa.reshape(-1, 3)).reshape(*aa.shape[:-1], 3, 3)


def neutral_pelvis_rest(fk: NeutralJointFK) -> np.ndarray:
    """Pelvis rest position J0 for betas = 0 (``pelvis = transl + J0`` at any root rotation)."""
    return fk.j_rest[0].detach().cpu().numpy().astype(np.float32)


def predict_joints(fk: NeutralJointFK, out: dict[str, Tensor]) -> Tensor:
    b, t = out["transl"].shape[:2]
    root_R = rotation_6d_to_matrix(out["root_rot_6d"].float()).reshape(b * t, 3, 3)
    body_R = rotation_6d_to_matrix(
        out["body_rot_6d"].float().reshape(b * t, NUM_BODY, 6)
    )
    return fk(out["transl"].float().reshape(b * t, 3), root_R, body_R).reshape(
        b, t, NUM_KP, 3
    )


def _masked_mean(x: Tensor, w: Tensor) -> Tensor:
    while w.ndim < x.ndim:
        w = w.unsqueeze(-1)
    w = w.expand_as(x).to(x.dtype)
    return (x * w).sum() / w.sum().clamp_min(1.0)


def compute_loss(
    model: EgoCompleteMotion,
    fk: NeutralJointFK,
    batch: dict[str, Any],
    *,
    loss_w: dict[str, float],
    sole_offsets: Tensor,
    device: torch.device,
    amp: bool,
) -> tuple[Tensor, dict[str, float]]:
    obs = {k: v.to(device, non_blocking=True) for k, v in batch["obs"].items()}
    rig = {k: v.to(device, non_blocking=True) for k, v in batch["rig"].items()}
    tgt = {k: v.to(device, non_blocking=True) for k, v in batch["targets"].items()}
    valid = batch["valid"].to(device)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
        out = model(obs, rig)
    out = {k: v.float() for k, v in out.items()}
    b = valid.shape[0]
    gt_body_R = aa_to_matrix(tgt["body_aa"])
    l_geo = geodesic_pose_loss(
        out["root_rot_6d"], out["body_rot_6d"], tgt["root_R"], gt_body_R, valid
    )
    l_tr = _masked_mean((out["transl"] - tgt["transl"]).abs(), valid)
    joints = predict_joints(fk, out)
    l_j = _masked_mean((joints - tgt["joints_gt_22"]).norm(dim=-1), valid)
    cmask = valid & batch["contact_valid"].to(device).view(b, 1)
    bce = nn.functional.binary_cross_entropy_with_logits(
        out["contact_logits"], tgt["contact"], reduction="none"
    )
    l_c = _masked_mean(bce, cmask)
    loss = (
        float(loss_w["w_geodesic"]) * l_geo
        + float(loss_w["w_transl"]) * l_tr
        + float(loss_w["w_joints"]) * l_j
        + float(loss_w["w_contact"]) * l_c
    )
    stats = {
        "geodesic_rad": l_geo.item(),
        "transl_m": l_tr.item(),
        "joints_m": l_j.item(),
        "contact_bce": l_c.item(),
    }
    w_phys = float(loss_w.get("w_phys", 0.0))
    if (
        w_phys > 0
    ):  # off by default (refinement comparison "none" arm); item-3 losses on sole-proxy foot channels
        feet = joints[:, :, [7, 10, 8, 11], :] - torch.stack(
            [
                torch.zeros_like(sole_offsets),
                torch.zeros_like(sole_offsets),
                sole_offsets,
            ],
            dim=-1,
        )
        l_p = foot_skating(
            feet, tgt["contact"] * cmask.unsqueeze(-1), 30.0
        ) + ground_penetration(feet)
        loss = loss + w_phys * l_p
        stats["phys"] = l_p.item()
    stats["loss"] = loss.item()
    return loss, stats


@torch.no_grad()
def predict_clip_learned(
    model: EgoCompleteMotion,
    fk: NeutralJointFK,
    obs_w: dict[str, Tensor],
    rig_w: dict[str, Tensor],
    *,
    window: int,
    stride: int,
    device: torch.device,
    max_batch: int = 256,
) -> tuple[np.ndarray, np.ndarray, list[int], np.ndarray, np.ndarray]:
    """Per-window canonical predictions for one clip: joints, contact probs, starts, origins, valid."""
    t_len = obs_w["joint_pos_3d"].shape[0]
    starts = window_starts(t_len, window, stride)
    obs, rig, origins, valid = eval_window_batch(obs_w, rig_w, starts, window)
    model.eval()
    j_l, p_l = [], []
    for i0 in range(0, len(starts), max_batch):
        sl = slice(i0, i0 + max_batch)
        o = {k: v[sl].to(device) for k, v in obs.items()}
        r = {k: v[sl].to(device) for k, v in rig.items()}
        out = {k: v.float() for k, v in model(o, r).items()}
        j_l.append(predict_joints(fk, out).cpu().numpy())
        p_l.append(torch.sigmoid(out["contact_logits"]).cpu().numpy())
    return (
        np.concatenate(j_l),
        np.concatenate(p_l),
        starts,
        origins.numpy(),
        valid.numpy(),
    )


def stitch_windows(
    values: np.ndarray,
    starts: list[int],
    valid: np.ndarray,
    t_len: int,
    origins: np.ndarray | None = None,
) -> tuple[np.ndarray, float, int]:
    """Average overlapping windows into ``(T, ...)``; positions get ``origin_xy`` added first.

    Returns the stitched array, the mean per-joint L2 (mm) between consecutive windows on their
    shared frames (positions only), and the number of shared frames.
    """
    vals = values.astype(np.float64).copy()
    if origins is not None:
        vals[..., 0] += origins[:, None, None, 0]
        vals[..., 1] += origins[:, None, None, 1]
    acc = np.zeros((t_len,) + vals.shape[2:], dtype=np.float64)
    cnt = np.zeros(t_len, dtype=np.float64)
    dis_sum, dis_n = 0.0, 0
    prev: tuple[int, np.ndarray] | None = None
    for w, s in enumerate(starts):
        n = int(valid[w].sum())
        acc[s : s + n] += vals[w, :n]
        cnt[s : s + n] += 1.0
        if origins is not None and prev is not None:
            ps, pv = prev
            lo, hi = s, min(ps + pv.shape[0], s + n)
            if hi > lo:
                d = np.linalg.norm(
                    vals[w, lo - s : hi - s] - pv[lo - ps : hi - ps], axis=-1
                ).mean(axis=-1)
                dis_sum += float(d.sum()) * 1000.0
                dis_n += hi - lo
        prev = (s, vals[w, :n])
    out = acc / cnt.reshape((-1,) + (1,) * (acc.ndim - 1))
    return out.astype(np.float32), (dis_sum / dis_n if dis_n else 0.0), dis_n


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)
