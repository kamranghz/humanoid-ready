"""Online corruption + virtual pinhole camera for HR-Refine (item 7)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

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
CorruptionMode = Literal["full", "rot_jitter", "root_noise", "foot_sink", "none"]


@dataclass
class CorruptionConfig:
    rot_jitter_std_rad: float = 0.018
    root_trans_std_m: float = 0.035
    foot_sink_m: float = 0.025
    kp_noise_px: float = 3.0
    frame_dropout_prob: float = 0.03
    joint_dropout_prob: float = 0.05
    kp_dropout_prob: float = 0.08


@dataclass
class VirtualCameraConfig:
    distance_m: tuple[float, float] = (2.5, 5.5)
    height_m: tuple[float, float] = (1.0, 2.2)
    yaw_rad: tuple[float, float] = (-3.14159, 3.14159)
    focal_px: float = 900.0
    img_size: tuple[int, int] = (512, 512)


def corruption_config_from_dict(data: dict[str, Any] | None) -> CorruptionConfig:
    if not data:
        return CorruptionConfig()
    return CorruptionConfig(
        rot_jitter_std_rad=float(data.get("rot_jitter_std_rad", 0.018)),
        root_trans_std_m=float(data.get("root_trans_std_m", 0.035)),
        foot_sink_m=float(data.get("foot_sink_m", 0.025)),
        kp_noise_px=float(data.get("kp_noise_px", 3.0)),
        frame_dropout_prob=float(data.get("frame_dropout_prob", 0.03)),
        joint_dropout_prob=float(data.get("joint_dropout_prob", 0.05)),
        kp_dropout_prob=float(data.get("kp_dropout_prob", 0.08)),
    )


def make_corruption_rng(base_seed: int, step: int) -> np.random.Generator:
    return np.random.default_rng(int(base_seed) + int(step) * 1_000_003)


def axis_angle_to_rot6d(aa: Tensor) -> Tensor:
    m = axis_angle_to_matrix(aa.reshape(-1, 3)).reshape(*aa.shape[:-1], 3, 3)
    return matrix_to_rotation_6d(m)


def rot6d_to_axis_angle(r6: Tensor) -> Tensor:
    from hready.body.rotations import matrix_to_axis_angle

    m = rotation_6d_to_matrix(r6.reshape(-1, 6)).reshape(*r6.shape[:-1], 3, 3)
    return matrix_to_axis_angle(m.reshape(-1, 3, 3)).reshape(*r6.shape[:-1], 3)


def pack_pose(transl: Tensor, root_aa: Tensor, body_aa: Tensor) -> dict[str, Tensor]:
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
    mode: CorruptionMode = "full",
) -> dict[str, Any]:
    cfg = cfg or CorruptionConfig()
    cam_cfg = cam_cfg or VirtualCameraConfig()
    device = clean["transl"].device
    b, t = clean["transl"].shape[:2]
    transl = clean["transl"].clone()
    root_aa = clean["root_aa"].clone()
    body_aa = clean["body_aa"].clone()

    if mode in ("full", "rot_jitter"):
        jitter = torch.as_tensor(
            rng.normal(0, cfg.rot_jitter_std_rad, body_aa.shape), device=device, dtype=body_aa.dtype
        )
        body_aa = body_aa + jitter
        root_aa = root_aa + torch.as_tensor(
            rng.normal(0, cfg.rot_jitter_std_rad, root_aa.shape), device=device, dtype=root_aa.dtype
        )
    if mode in ("full", "root_noise"):
        transl = transl + torch.as_tensor(
            rng.normal(0, cfg.root_trans_std_m, transl.shape), device=device, dtype=transl.dtype
        )
    if mode in ("full", "foot_sink"):
        transl[..., 2] = transl[..., 2] - float(cfg.foot_sink_m)

    if mode == "full":
        frame_mask = torch.as_tensor(
            rng.random(t) > cfg.frame_dropout_prob, device=device
        ).float().view(1, t, 1, 1)
        joint_mask = torch.as_tensor(
            rng.random(body_aa.shape[:-1]) > cfg.joint_dropout_prob, device=device
        ).float().unsqueeze(-1)
        body_aa = body_aa * frame_mask * joint_mask

    corrupt_joints = joints_clean.clone()
    if mode in ("full", "foot_sink"):
        corrupt_joints[..., 2] = corrupt_joints[..., 2] - float(cfg.foot_sink_m)

    cam = sample_virtual_camera(rng, cam_cfg, device, b)
    kp = project_joints(
        corrupt_joints[..., :NUM_KP_JOINTS, :],
        cam,
        noise_std_px=cfg.kp_noise_px if mode == "full" else 0.0,
        rng=rng if mode == "full" else None,
    )
    if mode == "full":
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


def _mpjpe_root_mm(pred: np.ndarray, gt: np.ndarray) -> float:
    from hready.metrics.pose import mpjpe

    return float(np.nanmean(mpjpe(pred, gt, root_idx=0)))


def _mpjpe_abs_mm(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    return float(np.linalg.norm(pred - gt, axis=-1).mean() * 1000.0)


def _foot_phys(
    joints: np.ndarray,
    verts: np.ndarray | None = None,
    fps: float = 30.0,
) -> tuple[float, float]:
    from hready.metrics.physical import foot_skate, ground_penetration

    j = np.asarray(joints, dtype=np.float64)
    v = np.asarray(verts if verts is not None else joints, dtype=np.float64)
    if j.ndim == 4:
        pens, skates = [], []
        for bi in range(j.shape[0]):
            for ti in range(j.shape[1]):
                frame = j[bi, ti]
                fp = frame[[7, 10, 8, 11], :]
                ct = (fp[:, 2] < 0.05).astype(np.float32)
                pens.append(ground_penetration(v[bi, ti][np.newaxis, ...])[0])
                skates.append(foot_skate(fp[np.newaxis, ...], ct[np.newaxis, ...], fps))
        return float(np.mean(pens)), float(np.mean(skates))
    fp = j[:, [7, 10, 8, 11], :]
    ct = (fp[..., 2] < 0.05).astype(np.float32)
    pen = ground_penetration(v[np.newaxis, ...] if v.ndim == 2 else v)[0]
    skate = foot_skate(fp[np.newaxis, ...], ct[np.newaxis, ...], fps)
    return float(pen), float(skate)


def corruption_metrics_table(
    body: Any,
    clean: dict[str, Tensor],
    betas: Tensor,
    cfg: CorruptionConfig,
    base_seed: int,
    step: int,
) -> list[dict[str, float | str]]:
    from hready.body.batch_forward import smpl_forward_bt

    rows: list[dict[str, float | str]] = []
    with torch.inference_mode():
        joints_clean, _verts_clean = smpl_forward_bt(
            body, clean["transl"], clean["root_aa"], clean["body_aa"], betas
        )
    jc = joints_clean.detach().cpu().numpy()
    rows.append(
        {
            "type": "clean",
            "mpjpe_root_mm": 0.0,
            "mpjpe_abs_mm": 0.0,
            "penetration_mm": 0.0,
            "foot_skate_m_s": 0.0,
        }
    )
    _seed_off = {"rot_jitter": 11, "root_noise": 22, "foot_sink": 33, "full": 0}
    for mode in ("rot_jitter", "root_noise", "foot_sink", "full"):
        rng = make_corruption_rng(base_seed + _seed_off[mode], step)
        corrupted = apply_corruption(
            clean, joints_clean, rng=rng, cfg=cfg, mode=mode  # type: ignore[arg-type]
        )
        tr, ro, ba = unpack_pose(corrupted["corrupt_pose"])
        with torch.inference_mode():
            joints_c, verts_c = smpl_forward_bt(body, tr, ro, ba, betas)
        jn = joints_c.detach().cpu().numpy()
        vn = verts_c.detach().cpu().numpy()
        pen, skate = _foot_phys(jn, vn)
        rows.append(
            {
                "type": mode,
                "mpjpe_root_mm": _mpjpe_root_mm(jn, jc),
                "mpjpe_abs_mm": _mpjpe_abs_mm(jn, jc),
                "penetration_mm": pen,
                "foot_skate_m_s": skate,
            }
        )
    return rows
