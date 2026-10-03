"""Online corruption + virtual pinhole camera for HR-Refine (item 7)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from hready.body.rotations import (
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)

NUM_BODY_JOINTS = 21
NUM_KP_JOINTS = 22
_FPS = 30.0


@dataclass
class CorruptionConfig:
    rot_jitter_std_rad: float = 0.08
    root_trans_std_m: float = 0.03
    foot_sink_m: float = 0.04
    kp_noise_px: float = 4.0
    frame_dropout_prob: float = 0.05
    joint_dropout_prob: float = 0.1
    kp_dropout_prob: float = 0.15


@dataclass
class VirtualCameraConfig:
    distance_m: tuple[float, float] = (2.5, 5.5)
    height_m: tuple[float, float] = (1.0, 2.2)
    yaw_rad: tuple[float, float] = (-3.14159, 3.14159)
    focal_px: float = 900.0
    img_size: tuple[int, int] = (512, 512)


def make_corruption_rng(base_seed: int, step: int) -> np.random.Generator:
    return np.random.default_rng(int(base_seed) + int(step) * 1_000_003)


def axis_angle_to_rot6d(aa: Tensor) -> Tensor:
    """(…, 3) -> (…, 6)."""
    m = axis_angle_to_matrix(aa.reshape(-1, 3)).reshape(*aa.shape[:-1], 3, 3)
    return matrix_to_rotation_6d(m)


def rot6d_to_axis_angle(r6: Tensor) -> Tensor:
    from hready.body.rotations import matrix_to_axis_angle

    m = rotation_6d_to_matrix(r6.reshape(-1, 6)).reshape(*r6.shape[:-1], 3, 3)
    return matrix_to_axis_angle(m.reshape(-1, 3, 3)).reshape(*r6.shape[:-1], 3)


def pack_pose(transl: Tensor, root_aa: Tensor, body_aa: Tensor) -> dict[str, Tensor]:
    """``body_aa`` (B,T,21,3) axis-angle."""
    root6 = axis_angle_to_rot6d(root_aa)
    body6 = axis_angle_to_rot6d(body_aa)
    return {
        "transl": transl,
        "root_rot_6d": root6,
        "body_rot_6d": body6.reshape(*body_aa.shape[:-2], NUM_BODY_JOINTS * 6),
    }


def unpack_pose(packed: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
    transl = packed["transl"]
    root_aa = rot6d_to_axis_angle(packed["root_rot_6d"])
    body6 = packed["body_rot_6d"].reshape(*packed["body_rot_6d"].shape[:-1], NUM_BODY_JOINTS, 6)
    body_aa = rot6d_to_axis_angle(body6)
    return transl, root_aa, body_aa


def sample_virtual_camera(
    rng: np.random.Generator, cfg: VirtualCameraConfig, device: torch.device, batch: int
) -> dict[str, Tensor]:
    dist = rng.uniform(cfg.distance_m[0], cfg.distance_m[1], size=batch)
    height = rng.uniform(cfg.height_m[0], cfg.height_m[1], size=batch)
    yaw = rng.uniform(cfg.yaw_rad[0], cfg.yaw_rad[1], size=batch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cam_pos = np.stack([-dist * sy, dist * cy, height], axis=-1)
    target = np.zeros((batch, 3))
    forward = target - cam_pos
    forward /= np.linalg.norm(forward, axis=-1, keepdims=True) + 1e-8
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right, axis=-1, keepdims=True) + 1e-8
    up = np.cross(right, forward)
    r = np.stack([right, up, -forward], axis=1)
    t = -np.einsum("bij,bj->bi", r, cam_pos)
    return {
        "R": torch.as_tensor(r, dtype=torch.float32, device=device),
        "t": torch.as_tensor(t, dtype=torch.float32, device=device),
        "focal": torch.full((batch,), cfg.focal_px, device=device),
        "cx": torch.full((batch,), cfg.img_size[0] * 0.5, device=device),
        "cy": torch.full((batch,), cfg.img_size[1] * 0.5, device=device),
    }


def project_joints(
    joints: Tensor,
    cam: dict[str, Tensor],
    *,
    noise_std_px: float = 0.0,
    rng: np.random.Generator | None = None,
) -> Tensor:
    """``joints`` (B,T,J,3) world -> (B,T,J,3) normalized xy in [-1,1] + confidence."""
    b, t, j, _ = joints.shape
    x = joints.reshape(b, t * j, 3)
    xc = torch.matmul(x, cam["R"].transpose(-1, -2)) + cam["t"].unsqueeze(1)
    z = xc[..., 2].clamp(min=0.05)
    focal = cam["focal"].view(b, 1)
    cx = cam["cx"].view(b, 1)
    cy = cam["cy"].view(b, 1)
    u = focal * xc[..., 0] / z + cx
    v = focal * xc[..., 1] / z + cy
    if noise_std_px > 0 and rng is not None:
        u = u + torch.as_tensor(rng.normal(0, noise_std_px, u.shape), device=u.device, dtype=u.dtype)
        v = v + torch.as_tensor(rng.normal(0, noise_std_px, v.shape), device=v.device, dtype=v.dtype)
    w = (cam["cx"] * 2).view(b, 1)
    h = (cam["cy"] * 2).view(b, 1)
    nx = (u / w) * 2.0 - 1.0
    ny = (v / h) * 2.0 - 1.0
    conf = (z > 0.1).float()
    nx = nx.reshape(b, t, j)
    ny = ny.reshape(b, t, j)
    conf = conf.reshape(b, t, j)
    return torch.stack([nx, ny, conf], dim=-1)


def apply_corruption(
    clean: dict[str, Tensor],
    joints_clean: Tensor,
    *,
    rng: np.random.Generator,
    cfg: CorruptionConfig | None = None,
    cam_cfg: VirtualCameraConfig | None = None,
) -> dict[str, Any]:
    """Corrupt pose + project 2D keypoints. ``clean`` has transl, root_aa, body_aa."""
    cfg = cfg or CorruptionConfig()
    cam_cfg = cam_cfg or VirtualCameraConfig()
    device = clean["transl"].device
    b, t = clean["transl"].shape[:2]
    transl = clean["transl"].clone()
    root_aa = clean["root_aa"].clone()
    body_aa = clean["body_aa"].clone()

    jitter = torch.as_tensor(
        rng.normal(0, cfg.rot_jitter_std_rad, body_aa.shape), device=device, dtype=body_aa.dtype
    )
    body_aa = body_aa + jitter
    root_aa = root_aa + torch.as_tensor(
        rng.normal(0, cfg.rot_jitter_std_rad, root_aa.shape), device=device, dtype=root_aa.dtype
    )
    transl = transl + torch.as_tensor(
        rng.normal(0, cfg.root_trans_std_m, transl.shape), device=device, dtype=transl.dtype
    )
    transl[..., 2] = transl[..., 2] - float(cfg.foot_sink_m)

    frame_mask = torch.as_tensor(
        rng.random(t) > cfg.frame_dropout_prob, device=device
    ).float().view(1, t, 1, 1)
    joint_mask = torch.as_tensor(
        rng.random(body_aa.shape[:-1]) > cfg.joint_dropout_prob, device=device
    ).float().unsqueeze(-1)
    body_aa = body_aa * frame_mask * joint_mask

    corrupt_joints = joints_clean.clone()
    corrupt_joints[..., 2] = corrupt_joints[..., 2] - float(cfg.foot_sink_m)

    cam = sample_virtual_camera(rng, cam_cfg, device, b)
    kp = project_joints(
        corrupt_joints[..., :NUM_KP_JOINTS, :],
        cam,
        noise_std_px=cfg.kp_noise_px,
        rng=rng,
    )
    bb, tt, jj, _ = kp.shape
    kp_mask = torch.as_tensor(
        rng.random((bb, tt, jj)) > cfg.kp_dropout_prob, device=device, dtype=torch.float32
    )
    kp = kp.clone()
    kp[..., 2] = kp[..., 2] * kp_mask

    packed = pack_pose(transl, root_aa, body_aa)
    return {
        "corrupt_pose": packed,
        "keypoints": kp,
        "camera": cam,
        "clean_pose": pack_pose(clean["transl"], clean["root_aa"], clean["body_aa"]),
    }


def corruption_mpjpe_report(
    body: Any,
    clean: dict[str, Tensor],
    betas: Tensor,
    base_seed: int,
    step: int,
) -> dict[str, float]:
    """Per-corruption mean MPJPE (mm, root-aligned) vs clean SMPL joints."""
    from hready.body.batch_forward import smpl_forward_bt
    from hready.metrics.pose import mpjpe

    rng = make_corruption_rng(base_seed, step)
    cfg = CorruptionConfig()
    device = clean["transl"].device
    b = clean["transl"].shape[0]

    with torch.inference_mode():
        joints_clean, _ = smpl_forward_bt(
            body, clean["transl"], clean["root_aa"], clean["body_aa"], betas
        )

    def _mpj(c_joints: Tensor) -> float:
        p = mpjpe(
            c_joints.detach().cpu().numpy(),
            joints_clean.detach().cpu().numpy(),
            root_idx=0,
        )
        return float(np.nanmean(p))

    out: dict[str, float] = {"clean": 0.0}

    c = {k: v.clone() for k, v in clean.items()}
    c["body_aa"] = c["body_aa"] + torch.as_tensor(
        rng.normal(0, cfg.rot_jitter_std_rad, c["body_aa"].shape),
        device=device,
        dtype=c["body_aa"].dtype,
    )
    c["root_aa"] = c["root_aa"] + torch.as_tensor(
        rng.normal(0, cfg.rot_jitter_std_rad, c["root_aa"].shape),
        device=device,
        dtype=c["root_aa"].dtype,
    )
    with torch.inference_mode():
        out["rot_jitter"] = _mpj(smpl_forward_bt(body, c["transl"], c["root_aa"], c["body_aa"], betas)[0])

    c = {k: v.clone() for k, v in clean.items()}
    c["transl"] = c["transl"] + torch.as_tensor(
        rng.normal(0, cfg.root_trans_std_m, c["transl"].shape),
        device=device,
        dtype=c["transl"].dtype,
    )
    with torch.inference_mode():
        out["root_noise"] = _mpj(smpl_forward_bt(body, c["transl"], c["root_aa"], c["body_aa"], betas)[0])

    c = {k: v.clone() for k, v in clean.items()}
    c["transl"] = c["transl"].clone()
    c["transl"][..., 2] -= cfg.foot_sink_m
    with torch.inference_mode():
        out["foot_sink"] = _mpj(smpl_forward_bt(body, c["transl"], c["root_aa"], c["body_aa"], betas)[0])

    full = apply_corruption(clean, joints_clean, rng=make_corruption_rng(base_seed, step), cfg=cfg)
    c_pack = full["corrupt_pose"]
    with torch.inference_mode():
        tr, ro, ba = unpack_pose(c_pack)
        out["full"] = _mpj(smpl_forward_bt(body, tr, ro, ba, betas)[0])

    cam = sample_virtual_camera(rng, VirtualCameraConfig(), device, b)
    kp = project_joints(joints_clean, cam, noise_std_px=cfg.kp_noise_px, rng=rng)
    out["kp_noise_px_rms"] = float((kp[..., :2] - project_joints(joints_clean, cam)[..., :2]).square().mean().sqrt().item() * 512)
    return out
