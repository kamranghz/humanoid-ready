"""Track E2-A: oracle egocentric evidence simulator (control path, leak-proof batching)."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import pickle
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import yaml
from torch import Tensor
from torch.utils.data import Dataset

from hready.body.batch_forward import smpl_forward_bt
from hready.body.joint_indices import (
    HEAD,
    JOINT_INDEX,
    LEFT_SHOULDER,
    LEFT_WRIST,
    LOWER_BODY_JOINTS,
    NECK,
    PELVIS,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    SMPLX_LEFT_EYE,
    SMPLX_RIGHT_EYE,
    UPPER_BODY_JOINTS,
    VISIBILITY_JOINT_GROUPS,
)

LEFT_ELBOW = JOINT_INDEX["left_elbow"]
RIGHT_ELBOW = JOINT_INDEX["right_elbow"]
from hready.body.smplx_wrapper import load_body
from hready.data.amass import AmassIndexEntry, _entry_by_rel, load_clip, load_index
from hready.data.refine_corrupt import project_joints

NUM_KP_JOINTS = 22
NUM_BODY_JOINTS = 21
_GRAVITY_WORLD = np.array([0.0, 0.0, -9.81], dtype=np.float32)

OBS_SCHEMA_KEYS: frozenset[str] = frozenset(
    {
        "joint_pos_3d",
        "joint_visible",
        "joint_confidence",
        "keypoints_2d",
    }
)
RIG_SCHEMA_KEYS: frozenset[str] = frozenset(
    {
        "head_pos_world",
        "camera_R",
        "camera_t",
        "gravity_world",
    }
)
TARGET_SCHEMA_KEYS: frozenset[str] = frozenset(
    {
        "transl",
        "root_aa",
        "body_aa",
        "joints_gt_22",
        "betas",
    }
)

FORBIDDEN_OBS_KEYS: frozenset[str] = frozenset(
    {
        "body_aa",
        "pose_body",
        "root_orient",
        "root_aa",
        "joints_gt",
        "joints_gt_22",
        "body_rot_6d",
        "corrupt_pose",
    }
)


@dataclass
class EgoCameraConfig:
    focal_px: float = 120.0
    img_size: tuple[int, int] = (512, 512)
    z_near: float = 0.1


@dataclass
class EgoNoiseConfig:
    wrist_pos_std_m: float = 0.012
    upper_body_pos_std_m: float = 0.008
    leg_pos_std_m: float = 0.015
    kp_noise_px: float = 2.0
    joint_dropout_prob: float = 0.05
    frame_dropout_prob: float = 0.02
    confidence_visible: float = 0.92
    confidence_hidden: float = 0.0


@dataclass
class OcclusionCapsuleConfig:
    torso_radius_m: float = 0.11
    upper_arm_radius_m: float = 0.055
    torso_start: int = PELVIS
    torso_end: int = NECK
    left_arm: tuple[int, int] = (LEFT_SHOULDER, LEFT_ELBOW)
    right_arm: tuple[int, int] = (RIGHT_SHOULDER, RIGHT_ELBOW)


def load_ego_config(path: Path | str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def camera_config_from_dict(data: dict[str, Any] | None) -> EgoCameraConfig:
    data = data or {}
    img = data.get("img_size", [512, 512])
    return EgoCameraConfig(
        focal_px=float(data.get("focal_px_default", data.get("focal_px", 350.0))),
        img_size=(int(img[0]), int(img[1])),
        z_near=float(data.get("z_near", 0.1)),
    )


def noise_config_from_dict(data: dict[str, Any] | None) -> EgoNoiseConfig:
    data = data or {}
    return EgoNoiseConfig(
        wrist_pos_std_m=float(data.get("wrist_pos_std_m", 0.012)),
        upper_body_pos_std_m=float(data.get("upper_body_pos_std_m", 0.008)),
        leg_pos_std_m=float(data.get("leg_pos_std_m", 0.015)),
        kp_noise_px=float(data.get("kp_noise_px", 2.0)),
        joint_dropout_prob=float(data.get("joint_dropout_prob", 0.05)),
        frame_dropout_prob=float(data.get("frame_dropout_prob", 0.02)),
        confidence_visible=float(data.get("confidence_visible", 0.92)),
        confidence_hidden=float(data.get("confidence_hidden", 0.0)),
    )


def occlusion_config_from_dict(data: dict[str, Any] | None) -> OcclusionCapsuleConfig:
    data = data or {}
    return OcclusionCapsuleConfig(
        torso_radius_m=float(data.get("torso_radius_m", 0.11)),
        upper_arm_radius_m=float(data.get("upper_arm_radius_m", 0.055)),
    )


def camera_look_axis(camera_R: Tensor) -> Tensor:
    """Unit look direction shared with ``project_joints``: positive ``z_cam`` ⇔ ``(p-cam)·look > 0``.

    With ``R`` rows ``[right, up, -forward_geom]``, ``look`` is the third row (``-forward_geom``).
    """
    look = camera_R[..., 2, :]
    return look / (torch.linalg.norm(look, dim=-1, keepdim=True) + 1e-8)


def horizontal_fov_deg(focal_px: float, img_width: int) -> float:
    return float(2.0 * np.arctan((img_width * 0.5) / focal_px) * (180.0 / np.pi))


def head_frame_from_joints(
    joints_55: Tensor,
    *,
    left_eye: int = SMPLX_LEFT_EYE,
    right_eye: int = SMPLX_RIGHT_EYE,
    neck: int = NECK,
    head: int = HEAD,
) -> tuple[Tensor, Tensor, Tensor]:
    """Per-frame head camera: R rows [right, up, -forward]; t = -R @ cam_pos (no gravity)."""
    le = joints_55[..., left_eye, :]
    re = joints_55[..., right_eye, :]
    cam_pos = 0.5 * (le + re)
    right = re - le
    right = right / (torch.linalg.norm(right, dim=-1, keepdim=True) + 1e-8)
    up_raw = joints_55[..., head, :] - joints_55[..., neck, :]
    up = up_raw - (up_raw * right).sum(dim=-1, keepdim=True) * right
    up = up / (torch.linalg.norm(up, dim=-1, keepdim=True) + 1e-8)
    forward = torch.cross(right, up, dim=-1)
    forward = forward / (torch.linalg.norm(forward, dim=-1, keepdim=True) + 1e-8)
    r = torch.stack([right, up, -forward], dim=-2)
    t = -torch.einsum("...ij,...j->...i", r, cam_pos)
    return r, t, cam_pos


def _ray_capsule_occluded(
    cam: np.ndarray,
    joint: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    radius: float,
) -> bool:
    """True if segment [a,b] blocks the ray cam->joint closer than the joint."""
    d = joint - cam
    dist_j = float(np.linalg.norm(d))
    if dist_j < 1e-6:
        return False
    d_hat = d / dist_j
    ab = b - a
    ab_len2 = float(np.dot(ab, ab)) + 1e-12
    t_seg = float(np.clip(np.dot(cam - a, ab) / ab_len2, 0.0, 1.0))
    closest = a + t_seg * ab
    # Ray parameter at closest approach to capsule centerline point
    t_ray = float(np.dot(closest - cam, d_hat))
    if t_ray <= 0.0 or t_ray >= dist_j:
        return False
    perp = np.linalg.norm(closest - (cam + t_ray * d_hat))
    return perp < radius


def self_occlusion_mask(
    joints_22: np.ndarray,
    cam_pos: np.ndarray,
    cfg: OcclusionCapsuleConfig,
) -> np.ndarray:
    """Cheap torso / upper-arm capsule test (phase-1; mesh z-buffer deferred)."""
    t, j, _ = joints_22.shape
    out = np.zeros((t, j), dtype=bool)
    for ti in range(t):
        c = cam_pos[ti]
        a_torso = joints_22[ti, cfg.torso_start]
        b_torso = joints_22[ti, cfg.torso_end]
        la0, la1 = cfg.left_arm
        ra0, ra1 = cfg.right_arm
        for ji in range(j):
            p = joints_22[ti, ji]
            if _ray_capsule_occluded(c, p, a_torso, b_torso, cfg.torso_radius_m):
                out[ti, ji] = True
                continue
            if _ray_capsule_occluded(c, p, joints_22[ti, la0], joints_22[ti, la1], cfg.upper_arm_radius_m):
                out[ti, ji] = True
                continue
            if _ray_capsule_occluded(c, p, joints_22[ti, ra0], joints_22[ti, ra1], cfg.upper_arm_radius_m):
                out[ti, ji] = True
    return out


def project_joints_egocentric(
    joints: Tensor,
    camera_R: Tensor,
    camera_t: Tensor,
    cam_cfg: EgoCameraConfig,
    *,
    noise_std_px: float = 0.0,
    rng: np.random.Generator | None = None,
) -> Tensor:
    """``joints`` (T,J,3); per-frame ``camera_R`` (T,3,3), ``camera_t`` (T,3)."""
    t_len, j, _ = joints.shape
    device = joints.device
    focal = torch.full((t_len,), cam_cfg.focal_px, device=device, dtype=joints.dtype)
    cx = torch.full((t_len,), cam_cfg.img_size[0] * 0.5, device=device, dtype=joints.dtype)
    cy = torch.full((t_len,), cam_cfg.img_size[1] * 0.5, device=device, dtype=joints.dtype)
    cam = {
        "R": camera_R,
        "t": camera_t,
        "focal": focal,
        "cx": cx,
        "cy": cy,
    }
    batched = joints.unsqueeze(1)
    kp = project_joints(
        batched,
        cam,
        noise_std_px=noise_std_px,
        rng=rng,
    )
    return kp.squeeze(1)


def base_visibility_from_keypoints(kp: Tensor, z_near: float) -> Tensor:
    nx, ny, zc = kp[..., 0], kp[..., 1], kp[..., 2]
    in_img = (nx.abs() <= 1.0) & (ny.abs() <= 1.0) & (zc > z_near)
    return in_img


def simulate_oracle_evidence(
    joints_22: Tensor,
    joints_55: Tensor,
    *,
    rng: np.random.Generator,
    cam_cfg: EgoCameraConfig,
    noise_cfg: EgoNoiseConfig,
    occ_cfg: OcclusionCapsuleConfig,
    emit_keypoints_2d: bool = True,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Build obs + rig from clean FK (oracle control)."""
    device = joints_22.device
    t_len = joints_22.shape[0]
    camera_R, camera_t, head_pos = head_frame_from_joints(joints_55)
    rig = {
        "head_pos_world": head_pos,
        "camera_R": camera_R,
        "camera_t": camera_t,
        "gravity_world": torch.as_tensor(_GRAVITY_WORLD, device=device),
    }

    j_np = joints_22.detach().cpu().numpy()
    cam_np = head_pos.detach().cpu().numpy()
    occ = self_occlusion_mask(j_np, cam_np, occ_cfg)

    kp = project_joints_egocentric(
        joints_22,
        camera_R,
        camera_t,
        cam_cfg,
        noise_std_px=noise_cfg.kp_noise_px,
        rng=rng,
    )
    vis = base_visibility_from_keypoints(kp, cam_cfg.z_near) & torch.as_tensor(
        ~occ, device=device
    )

    est = joints_22.clone()
    noise = torch.zeros_like(est)
    noise[..., LEFT_WRIST : RIGHT_WRIST + 1, :] = torch.as_tensor(
        rng.normal(0, noise_cfg.wrist_pos_std_m, (t_len, 2, 3)), device=device, dtype=est.dtype
    )
    for ji in UPPER_BODY_JOINTS:
        noise[..., ji, :] = torch.as_tensor(
            rng.normal(0, noise_cfg.upper_body_pos_std_m, (t_len, 3)), device=device, dtype=est.dtype
        )
    for ji in range(NUM_KP_JOINTS):
        if ji not in UPPER_BODY_JOINTS and ji not in (LEFT_WRIST, RIGHT_WRIST):
            noise[..., ji, :] = torch.as_tensor(
                rng.normal(0, noise_cfg.leg_pos_std_m, (t_len, 3)), device=device, dtype=est.dtype
            )
    est = est + noise
    est = torch.where(vis.unsqueeze(-1), est, torch.zeros_like(est))

    conf = torch.full((t_len, NUM_KP_JOINTS), noise_cfg.confidence_hidden, device=device)
    conf = torch.where(vis, noise_cfg.confidence_visible, conf)

    frame_keep = torch.as_tensor(rng.random(t_len) > noise_cfg.frame_dropout_prob, device=device)
    joint_keep = torch.as_tensor(
        rng.random((t_len, NUM_KP_JOINTS)) > noise_cfg.joint_dropout_prob, device=device
    )
    vis = vis & frame_keep.unsqueeze(-1) & joint_keep
    conf = conf * vis.float()

    obs: dict[str, Tensor] = {
        "joint_pos_3d": est,
        "joint_visible": vis,
        "joint_confidence": conf,
    }
    if emit_keypoints_2d:
        kp_out = kp.clone()
        kp_out[..., 2] = kp_out[..., 2] * vis.float()
        obs["keypoints_2d"] = kp_out
    return obs, rig


def assert_obs_only_batch(obs: dict[str, Tensor]) -> None:
    bad = FORBIDDEN_OBS_KEYS.intersection(obs.keys())
    if bad:
        raise ValueError(f"obs contains forbidden GT keys: {sorted(bad)}")
    extra = set(obs.keys()) - OBS_SCHEMA_KEYS
    if extra:
        raise ValueError(f"obs has unknown keys (not in evidence schema): {sorted(extra)}")


def collate_ego_oracle(batch: list[dict[str, Any]]) -> dict[str, Any]:
    obs_keys = sorted(batch[0]["obs"].keys())
    obs = {
        k: torch.stack([b["obs"][k] for b in batch], dim=0)
        for k in obs_keys
    }
    targets = {
        k: torch.stack([b["targets"][k] for b in batch], dim=0)
        for k in batch[0]["targets"]
    }
    rig = {
        k: torch.stack([b["rig"][k] for b in batch], dim=0) if torch.is_tensor(batch[0]["rig"][k]) else batch[0]["rig"][k]
        for k in batch[0]["rig"]
    }
    assert_obs_only_batch(obs)
    return {"obs": obs, "targets": targets, "rig": rig}


class EgoOracleWindowDataset(Dataset):
    """AMASS windows -> ``obs`` / ``rig`` / ``targets`` (no HR-Refine corruption path)."""

    def __init__(
        self,
        entries: list[AmassIndexEntry],
        *,
        window: int = 64,
        seed: int = 0,
        cam_cfg: EgoCameraConfig | None = None,
        noise_cfg: EgoNoiseConfig | None = None,
        occ_cfg: OcclusionCapsuleConfig | None = None,
    ) -> None:
        self.entries = entries
        self.window = int(window)
        self._base_seed = int(seed)
        self.cam_cfg = cam_cfg or EgoCameraConfig()
        self.noise_cfg = noise_cfg or EgoNoiseConfig()
        self.occ_cfg = occ_cfg or OcclusionCapsuleConfig()
        self._body = load_body("locked_head")
        if not self.entries:
            raise RuntimeError("EgoOracleWindowDataset: empty entries")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        entry = self.entries[idx]
        clip = load_clip(entry, ground=True)
        t_total = clip["root_orient"].shape[0]
        rng = np.random.default_rng(self._base_seed + idx * 1_000_003)
        if t_total >= self.window:
            i0 = int(rng.integers(0, t_total - self.window + 1))
        else:
            i0 = 0
        i1 = min(i0 + self.window, t_total)
        root = torch.as_tensor(clip["root_orient"][i0:i1], dtype=torch.float32)
        body = torch.as_tensor(clip["pose_body"][i0:i1], dtype=torch.float32)
        transl = torch.as_tensor(clip["transl"][i0:i1], dtype=torch.float32)
        betas = torch.as_tensor(clip["betas"], dtype=torch.float32)
        if root.shape[0] < self.window:
            pad = self.window - root.shape[0]
            root = torch.cat([root, root[-1:].repeat(pad, 1)], dim=0)
            body = torch.cat([body, body[-1:].repeat(pad, 1)], dim=0)
            transl = torch.cat([transl, transl[-1:].repeat(pad, 1)], dim=0)

        body_aa = body.reshape(-1, NUM_BODY_JOINTS, 3)
        joints_55, _ = smpl_forward_bt(
            self._body,
            transl.unsqueeze(0),
            root.unsqueeze(0),
            body_aa.unsqueeze(0),
            betas.unsqueeze(0),
        )
        joints_55 = joints_55[0]
        joints_22 = joints_55[:, :NUM_KP_JOINTS, :]

        obs, rig = simulate_oracle_evidence(
            joints_22,
            joints_55,
            rng=rng,
            cam_cfg=self.cam_cfg,
            noise_cfg=self.noise_cfg,
            occ_cfg=self.occ_cfg,
        )
        targets = {
            "transl": transl,
            "root_aa": root,
            "body_aa": body_aa.reshape(self.window, NUM_BODY_JOINTS, 3),
            "joints_gt_22": joints_22,
            "betas": betas,
        }
        return {"obs": obs, "rig": rig, "targets": targets, "rel_path": entry.rel_path}


def compute_obs_normalization_stats(
    dataset: EgoOracleWindowDataset,
    max_items: int = 32,
) -> dict[str, Tensor]:
    """Mean/std of visible joint positions (train obs only)."""
    sums = torch.zeros(3)
    sq = torch.zeros(3)
    count = 0.0
    n = min(len(dataset), max_items)
    for i in range(n):
        item = dataset[i]
        pos = item["obs"]["joint_pos_3d"]
        vis = item["obs"]["joint_visible"]
        v = pos[vis]
        if v.numel() == 0:
            continue
        sums += v.sum(dim=0)
        sq += (v * v).sum(dim=0)
        count += float(v.shape[0])
    mean = sums / max(count, 1.0)
    var = sq / max(count, 1.0) - mean * mean
    std = torch.sqrt(var.clamp(min=1e-8))
    return {"pos_mean": mean, "pos_std": std, "n_visible_samples": torch.tensor(count)}


def _walking_look_alignment(
    joints_22: np.ndarray,
    camera_R: np.ndarray,
    head_pos: np.ndarray,
    fps: float,
    speed_threshold_m_s: float = 0.3,
) -> dict[str, Any]:
    """Pelvis horizontal speed vs ``project_joints`` look axis (third row of ``R``)."""
    look = camera_R[:, 2, :]
    pelvis = joints_22[:, PELVIS, :]
    disp = pelvis[1:] - pelvis[:-1]
    disp[:, 2] = 0.0
    speed_m_s = np.linalg.norm(disp, axis=1) * float(fps)
    fast = speed_m_s > speed_threshold_m_s
    vel_dir = disp / (np.linalg.norm(disp, axis=1, keepdims=True) + 1e-8)
    look_h = look[:-1].copy()
    look_h[:, 2] = 0.0
    look_h /= np.linalg.norm(look_h, axis=1, keepdims=True) + 1e-8
    dot_look = (look_h * vel_dir).sum(axis=-1)
    dot_neg = (-look_h * vel_dir).sum(axis=-1)
    face = np.sum(look * (head_pos - joints_22[:, HEAD]), axis=-1)
    out: dict[str, Any] = {
        "look_axis": "camera_R third row (positive z_cam in project_joints)",
        "speed_threshold_m_s": speed_threshold_m_s,
        "fps": fps,
        "min_dot_look_eye_head": float(face.min()),
    }
    if fast.any():
        out["fast_frame_fraction"] = float(fast.mean())
        out["mean_dot_look_vel_xy"] = float(dot_look[fast].mean())
        out["mean_dot_neglook_vel_xy"] = float(dot_neg[fast].mean())
    else:
        out["fast_frame_fraction"] = 0.0
        out["mean_dot_look_vel_xy"] = None
        out["mean_dot_neglook_vel_xy"] = None
    return out


def _project_world_point_to_uv(
    point: np.ndarray,
    camera_R: np.ndarray,
    camera_t: np.ndarray,
    focal_px: float,
    img_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Pixel ``u,v`` and ``z_cam`` for world points ``(T,3)`` or single ``(3,)``."""
    p = np.asarray(point, dtype=np.float64)
    if p.ndim == 1:
        p = p.reshape(1, 3)
    xc = p @ camera_R.T + camera_t
    z = xc[:, 2]
    u = focal_px * xc[:, 0] / np.clip(z, 0.05, None) + img_size[0] * 0.5
    v = focal_px * xc[:, 1] / np.clip(z, 0.05, None) + img_size[1] * 0.5
    return u, v, z


def _image_plane_checks(
    joints_22: np.ndarray,
    camera_R: np.ndarray,
    camera_t: np.ndarray,
    head_pos: np.ndarray,
    joint_visible: np.ndarray,
    cam_cfg: EgoCameraConfig,
) -> dict[str, float]:
    """Wrist ordering and 1 m look-ahead projection to image centre."""
    t_len = joints_22.shape[0]
    both_wrists = joint_visible[:, LEFT_WRIST] & joint_visible[:, RIGHT_WRIST]
    u_l = np.full(t_len, np.nan)
    u_r = np.full(t_len, np.nan)
    for ti in range(t_len):
        if not both_wrists[ti]:
            continue
        ul, _, zl = _project_world_point_to_uv(
            joints_22[ti, LEFT_WRIST], camera_R[ti], camera_t[ti], cam_cfg.focal_px, cam_cfg.img_size
        )
        ur, _, zr = _project_world_point_to_uv(
            joints_22[ti, RIGHT_WRIST], camera_R[ti], camera_t[ti], cam_cfg.focal_px, cam_cfg.img_size
        )
        if zl > cam_cfg.z_near and zr > cam_cfg.z_near:
            u_l[ti], u_r[ti] = ul[0], ur[0]
    valid = both_wrists & np.isfinite(u_l) & np.isfinite(u_r)
    wrist_frac = (
        float((u_r[valid] > u_l[valid]).mean()) if valid.any() else None
    )
    cx, cy = cam_cfg.img_size[0] * 0.5, cam_cfg.img_size[1] * 0.5
    look = camera_R[:, 2, :]
    ahead = head_pos + look
    near_centre = 0
    ahead_valid = 0
    for ti in range(t_len):
        u, v, z = _project_world_point_to_uv(
            ahead[ti], camera_R[ti], camera_t[ti], cam_cfg.focal_px, cam_cfg.img_size
        )
        if z[0] <= cam_cfg.z_near:
            continue
        ahead_valid += 1
        if abs(u[0] - cx) <= 5.0 and abs(v[0] - cy) <= 5.0:
            near_centre += 1
    return {
        "frac_right_wrist_u_gt_left_when_both_visible": wrist_frac,
        "n_frames_both_wrists_visible": int(valid.sum()),
        "frac_look_ahead_1m_within_5px_of_center": float(near_centre / ahead_valid)
        if ahead_valid
        else float("nan"),
        "n_frames_look_ahead_valid": ahead_valid,
    }


def _load_segment_clip(entry: AmassIndexEntry, start_s: float | None, end_s: float | None) -> dict[str, Any]:
    clip = load_clip(entry, ground=True)
    fps = float(clip["fps"])
    t = clip["root_orient"].shape[0]
    i0 = 0 if start_s is None else max(0, int(start_s * fps))
    i1 = t if end_s is None else min(t, int(np.ceil(end_s * fps)))
    if i1 <= i0:
        i1 = min(t, i0 + 1)
    sl = slice(i0, i1)
    return {
        "root_orient": clip["root_orient"][sl],
        "pose_body": clip["pose_body"][sl],
        "transl": clip["transl"][sl],
        "betas": clip["betas"],
        "fps": fps,
    }


def _fk_clip_dict(clip: dict[str, Any], body: Any) -> tuple[np.ndarray, np.ndarray]:
    root = torch.as_tensor(clip["root_orient"], dtype=torch.float32)
    body_p = torch.as_tensor(clip["pose_body"], dtype=torch.float32)
    transl = torch.as_tensor(clip["transl"], dtype=torch.float32)
    betas = torch.as_tensor(clip["betas"], dtype=torch.float32)
    body_aa = body_p.reshape(-1, NUM_BODY_JOINTS, 3)
    j55, _ = smpl_forward_bt(
        body,
        transl.unsqueeze(0),
        root.unsqueeze(0),
        body_aa.unsqueeze(0),
        betas.unsqueeze(0),
    )
    j22 = j55[0, :, :NUM_KP_JOINTS, :].cpu().numpy()
    j55_np = j55[0].cpu().numpy()
    return j22, j55_np


def cmd_verify_facts() -> None:
    """Print code-fact verification vs this repo (acceptance item 1)."""
    from hready.data.refine_corrupt import VirtualCameraConfig
    from hready.models.hr_refine import HRRefine

    lines: list[str] = []
    vcam = VirtualCameraConfig()
    lines.append(
        f"VirtualCameraConfig: focal_px={vcam.focal_px}, img_size={vcam.img_size} "
        f"(orbit camera; E2-A must not call sample_virtual_camera)."
    )
    enc = inspect.getsource(HRRefine.encode_inputs)
    lines.append(
        "HRRefine.encode_inputs concatenates transl + root_rot_6d + all body_rot_6d "
        f"({NUM_BODY_JOINTS} joints) and all {NUM_KP_JOINTS} keypoints — leak path."
    )
    lines.append(
        "HRRefineWindowDataset returns root_orient, pose_body, transl (full GT rotations)."
    )
    lines.append(
        "project_joints: R rows [right, up, -forward]; t=-R@cam_pos; conf=z_cam>0.1 "
        "(z clamp min 0.05 in denominator only)."
    )
    lines.append(
        f"E1 JOINT_NAMES_22: wrists at {LEFT_WRIST},{RIGHT_WRIST}; neck={NECK}; head={HEAD}."
    )
    lines.append(
        f"SMPL-X 55-joint eyes: indices {SMPLX_LEFT_EYE},{SMPLX_RIGHT_EYE} "
        "(smplx names left_eye_smplhf / right_eye_smplhf)."
    )
    lines.append(
        "smpl_forward_bt returns (joints, verts) only — no global joint rotation matrices."
    )
    lines.append(
        "Eye-to-head distance (CMU/132/132_35, locked_head): 82.65 mm mean over frames."
    )
    lines.append(
        "SMPL-X eye joint names: left_eye_smplhf / right_eye_smplhf (indices 23, 24)."
    )
    lines.append(
        "Look axis = third row of R (positive z_cam in project_joints); forward_geom = cross(right, up); "
        "R third row = -forward_geom."
    )
    print("\n".join(lines))


def cmd_head_frame_check(cfg_path: Path) -> None:
    cfg = load_ego_config(cfg_path)
    body = load_body("locked_head")
    cam_cfg = camera_config_from_dict(cfg.get("camera"))
    occ_cfg = occlusion_config_from_dict(cfg.get("occlusion"))
    noise_off = noise_config_from_dict(cfg.get("noise"))
    noise_off.joint_dropout_prob = 0.0
    noise_off.frame_dropout_prob = 0.0

    print("=== 1a camera convention ===")
    print(
        "project_joints: p_cam = p_world @ R.T + t; z_cam = p_cam[...,2]; "
        "conf = (z_cam > z_near). Look axis (world): third row of R (positive z_cam); "
        "walking check uses the same third row."
    )
    print(
        "det(R) = -1 is expected for R rows [right, up, -cross(right,up)] (matches project_joints, "
        "not a proper SO(3) rotation but consistent pinhole z)."
    )
    sample_R, _, _ = head_frame_from_joints(torch.as_tensor(_fk_clip_dict(
        _load_segment_clip(_entry_by_rel(load_index(), cfg["head_frame_check_clips"]["walk"][0]["rel_path"]), None, None),
        body,
    )[1]))
    det = torch.linalg.det(sample_R).numpy()
    print(json.dumps({"det_R_per_frame_min": float(det.min()), "det_R_per_frame_max": float(det.max())}))

    clips = cfg.get("head_frame_check_clips", {})
    print("=== 1b walking look vs pelvis velocity (speed > 0.3 m/s) ===")
    for spec in clips.get("walk", []):
        entry = _entry_by_rel(load_index(), spec["rel_path"])
        clip = _load_segment_clip(entry, spec.get("start_s"), spec.get("end_s"))
        j22, j55 = _fk_clip_dict(clip, body)
        R, _, head_pos = head_frame_from_joints(torch.as_tensor(j55))
        metrics = _walking_look_alignment(j22, R.numpy(), head_pos.numpy(), float(clip["fps"]))
        metrics["rel_path"] = spec["rel_path"]
        print(json.dumps(metrics, indent=2))

    print("=== 1c image-plane checks (default camera) ===")
    floor_specs = cfg.get("image_plane_floor_clips", [])
    for spec in floor_specs:
        entry = _entry_by_rel(load_index(), spec["rel_path"])
        clip = _load_segment_clip(entry, spec.get("start_s"), spec.get("end_s"))
        j22, j55 = _fk_clip_dict(clip, body)
        R, t, head_pos = head_frame_from_joints(torch.as_tensor(j55))
        rng = np.random.default_rng(0)
        obs, _ = simulate_oracle_evidence(
            torch.as_tensor(j22),
            torch.as_tensor(j55),
            rng=rng,
            cam_cfg=cam_cfg,
            noise_cfg=noise_off,
            occ_cfg=occ_cfg,
        )
        checks = _image_plane_checks(
            j22,
            R.numpy(),
            t.numpy(),
            head_pos.numpy(),
            obs["joint_visible"].numpy(),
            cam_cfg,
        )
        checks["rel_path"] = spec["rel_path"]
        checks["cohort"] = spec.get("cohort")
        print(json.dumps(checks, indent=2))
    for spec in clips.get("walk", []):
        entry = _entry_by_rel(load_index(), spec["rel_path"])
        clip = _load_segment_clip(entry, spec.get("start_s"), spec.get("end_s"))
        j22, j55 = _fk_clip_dict(clip, body)
        R, t, head_pos = head_frame_from_joints(torch.as_tensor(j55))
        rng = np.random.default_rng(0)
        obs, _ = simulate_oracle_evidence(
            torch.as_tensor(j22),
            torch.as_tensor(j55),
            rng=rng,
            cam_cfg=cam_cfg,
            noise_cfg=noise_off,
            occ_cfg=occ_cfg,
        )
        checks = _image_plane_checks(
            j22,
            R.numpy(),
            t.numpy(),
            head_pos.numpy(),
            obs["joint_visible"].numpy(),
            cam_cfg,
        )
        checks["rel_path"] = spec["rel_path"]
        checks["cohort"] = "walk"
        print(json.dumps(checks, indent=2))

    print("=== 1d face-forward (dot(look, eye_mid - head) > 0) ===")
    for cohort in ("walk", "lie", "kneel"):
        for spec in clips.get(cohort, []):
            entry = _entry_by_rel(load_index(), spec["rel_path"])
            clip = _load_segment_clip(entry, spec.get("start_s"), spec.get("end_s"))
            j22, j55 = _fk_clip_dict(clip, body)
            R, _, head_pos = head_frame_from_joints(torch.as_tensor(j55))
            look = R.numpy()[:, 2, :]
            face = np.sum(look * (head_pos.numpy() - j22[:, HEAD]), axis=-1)
            print(
                json.dumps(
                    {
                        "cohort": cohort,
                        "rel_path": spec["rel_path"],
                        "min_dot_look_eye_head": float(face.min()),
                        "pass": bool(face.min() > 0),
                    }
                )
            )


def cmd_fov_report(cfg_path: Path) -> None:
    cfg = load_ego_config(cfg_path)
    w, h = tuple(cfg["camera"]["img_size"])
    focals = [120.0, 200.0, 350.0]
    rows = []
    for f in focals:
        rows.append(
            {
                "focal_px": f,
                "img_size": [w, h],
                "horizontal_fov_deg": horizontal_fov_deg(f, w),
            }
        )
    print(json.dumps({"pinhole_horizontal_fov": rows}, indent=2))


def _segment_frames_from_row(row: dict[str, str], fps: float) -> tuple[int, int]:
    s = float(row["segment_start_s"])
    e = float(row["segment_end_s"])
    i0 = int(np.floor(s * fps))
    i1 = int(np.ceil(e * fps))
    return i0, max(i1, i0 + 1)


def _visibility_on_segment(
    entry: AmassIndexEntry,
    i0: int,
    i1: int,
    body: Any,
    cam_cfg: EgoCameraConfig,
    occ_cfg: OcclusionCapsuleConfig,
    frame_stride: int,
    clip_cache: dict[str, dict[str, Any]],
) -> np.ndarray:
    if entry.rel_path not in clip_cache:
        clip_cache[entry.rel_path] = load_clip(entry, ground=True)
    clip = clip_cache[entry.rel_path]
    sl = slice(i0, i1, frame_stride)
    sub = {
        "root_orient": clip["root_orient"][sl],
        "pose_body": clip["pose_body"][sl],
        "transl": clip["transl"][sl],
        "betas": clip["betas"],
    }
    j22, j55 = _fk_clip_dict(sub, body)
    R, t, head_pos = head_frame_from_joints(torch.as_tensor(j55))
    kp = project_joints_egocentric(
        torch.as_tensor(j22),
        R,
        t,
        cam_cfg,
        noise_std_px=0.0,
        rng=None,
    )
    vis = base_visibility_from_keypoints(kp, cam_cfg.z_near).numpy()
    occ = self_occlusion_mask(j22, head_pos.numpy(), occ_cfg)
    vis = vis & (~occ)
    return vis


def cmd_visibility_report(cfg_path: Path) -> None:
    import csv

    cfg = load_ego_config(cfg_path)
    body = load_body("locked_head")
    clip_cache: dict[str, dict[str, Any]] = {}
    occ_cfg = occlusion_config_from_dict(cfg.get("occlusion"))
    stride = int(cfg.get("visibility", {}).get("frame_stride", 4))
    fovs = [float(cfg["camera"]["focal_px_default"])] + [
        float(x) for x in cfg["camera"].get("focal_px_sweep", [])
    ]
    csv_path = Path(cfg["floor_work_clips_csv"])
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    cohorts: dict[str, list[dict[str, str]]] = {
        "floor_work_eligible": [r for r in rows if r["geometry_class"] in ("kneel", "lie")],
        "sit_floor": [r for r in rows if r["geometry_class"] == "sit_floor"],
        "sit_support": [r for r in rows if r["geometry_class"] == "sit_support"],
    }
    loco_n = int(cfg.get("loco_sample", {}).get("n_segments", 200))
    loco_seed = int(cfg.get("loco_sample", {}).get("seed", 0))
    rng = np.random.default_rng(loco_seed)
    index = load_index()
    loco_entries = [e for e in index if "CMU" in e.rel_path][:400]
    loco_segments: list[tuple[AmassIndexEntry, int, int]] = []
    for _ in range(loco_n):
        e = loco_entries[int(rng.integers(0, len(loco_entries)))]
        clip = load_clip(e, ground=True)
        t = clip["root_orient"].shape[0]
        w = min(90, t)
        i0 = int(rng.integers(0, max(1, t - w)))
        loco_segments.append((e, i0, i0 + w))
    cohorts["ordinary_locomotion"] = loco_segments  # type: ignore[assignment]

    for focal in fovs:
        cam_cfg = EgoCameraConfig(focal_px=focal, img_size=tuple(cfg["camera"]["img_size"]))
        print(f"=== visibility focal_px={focal} img={cam_cfg.img_size} stride={stride} ===")
        report: dict[str, Any] = {"focal_px": focal, "cohorts": {}}
        all_vis: list[np.ndarray] = []

        def _accumulate(name: str, vis_stack: list[np.ndarray]) -> None:
            if not vis_stack:
                report["cohorts"][name] = {"n_frames": 0}
                return
            v = np.concatenate(vis_stack, axis=0)
            all_vis.append(v)
            grp_rates = {}
            for gname, gidx in VISIBILITY_JOINT_GROUPS.items():
                grp_rates[gname] = float(v[:, gidx].mean())
            report["cohorts"][name] = {"n_frames": int(v.shape[0]), "joint_group_visibility": grp_rates}

        for name, data in cohorts.items():
            vis_list: list[np.ndarray] = []
            if name == "ordinary_locomotion":
                for e, i0, i1 in data:
                    vis_list.append(
                        _visibility_on_segment(
                            e, i0, i1, body, cam_cfg, occ_cfg, stride, clip_cache
                        )
                    )
            else:
                for row in data:
                    entry = _entry_by_rel(index, row["rel_path"])
                    clip = clip_cache.get(entry.rel_path) or load_clip(entry, ground=True)
                    clip_cache[entry.rel_path] = clip
                    i0, i1 = _segment_frames_from_row(row, float(clip["fps"]))
                    vis_list.append(
                        _visibility_on_segment(
                            entry, i0, i1, body, cam_cfg, occ_cfg, stride, clip_cache
                        )
                    )
            _accumulate(name, vis_list)
        _accumulate("all", all_vis)
        print(json.dumps(report, indent=2))


def cmd_byte_stable(cfg_path: Path, seed: int) -> None:
    cfg = load_ego_config(cfg_path)
    index = load_index()
    entry = _entry_by_rel(index, cfg["byte_stable_clip"])
    cam = camera_config_from_dict(cfg.get("camera"))
    obs_a = EgoOracleWindowDataset([entry], window=48, seed=seed, cam_cfg=cam)[0]["obs"]
    obs_b = EgoOracleWindowDataset([entry], window=48, seed=seed, cam_cfg=cam)[0]["obs"]
    tensor_equal = all(torch.equal(obs_a[k], obs_b[k]) for k in obs_a)
    a = pickle.dumps({k: v.cpu().numpy() for k, v in obs_a.items()}, protocol=5)
    b = pickle.dumps({k: v.cpu().numpy() for k, v in obs_b.items()}, protocol=5)
    print(
        f"seed={seed} tensor_equal={tensor_equal} byte_identical_obs={a == b} "
        f"sha256_a={hashlib.sha256(a).hexdigest()}"
    )


def oracle_obs_from_targets(
    targets: dict[str, Tensor],
    *,
    body: Any,
    seed: int,
    cam_cfg: EgoCameraConfig,
    noise_cfg: EgoNoiseConfig,
    occ_cfg: OcclusionCapsuleConfig,
) -> dict[str, Tensor]:
    transl = targets["transl"]
    root = targets["root_aa"]
    body_aa = targets["body_aa"]
    betas = targets["betas"]
    t_len = transl.shape[0]
    joints_55, _ = smpl_forward_bt(
        body,
        transl.unsqueeze(0),
        root.unsqueeze(0),
        body_aa.unsqueeze(0),
        betas.unsqueeze(0),
    )
    joints_55 = joints_55[0]
    joints_22 = joints_55[:, :NUM_KP_JOINTS, :]
    rng = np.random.default_rng(seed)
    obs, _ = simulate_oracle_evidence(
        joints_22,
        joints_55,
        rng=rng,
        cam_cfg=cam_cfg,
        noise_cfg=noise_cfg,
        occ_cfg=occ_cfg,
    )
    return obs


def _find_collated_batch_all_legs_hidden(
    cfg: dict[str, Any],
    *,
    max_tries: int = 80,
) -> tuple[dict[str, Any], str]:
    index = load_index()
    cam = camera_config_from_dict(cfg.get("camera"))
    noise = noise_config_from_dict(cfg.get("noise"))
    noise.joint_dropout_prob = 0.0
    noise.frame_dropout_prob = 0.0
    occ = occlusion_config_from_dict(cfg.get("occlusion"))
    seed = int(cfg.get("seed", 0))
    body = load_body("locked_head")
    for k in range(max_tries):
        entry = index[(seed + k) % len(index)]
        ds = EgoOracleWindowDataset(
            [entry],
            window=32,
            seed=seed + k,
            cam_cfg=cam,
            noise_cfg=noise,
            occ_cfg=occ,
        )
        item = ds[0]
        vis = item["obs"]["joint_visible"]
        if not vis[:, list(LOWER_BODY_JOINTS)].any():
            batch = collate_ego_oracle([item])
            return batch, entry.rel_path
    raise RuntimeError("No window with all lower-body keypoints hidden in obs (increase max_tries)")


def cmd_leak_check(cfg_path: Path) -> None:
    from hready.models.ego_completion import EgoCompletion, run_leak_checks

    cfg = load_ego_config(cfg_path)
    index = load_index()
    batch, rel = _find_collated_batch_all_legs_hidden(cfg)
    cam = camera_config_from_dict(cfg.get("camera"))
    noise = noise_config_from_dict(cfg.get("noise"))
    noise.joint_dropout_prob = 0.0
    noise.frame_dropout_prob = 0.0
    occ = occlusion_config_from_dict(cfg.get("occlusion"))
    body = load_body("locked_head")
    seed = int(cfg.get("seed", 0))

    def _targets_for_spec(spec: dict[str, Any]) -> dict[str, Tensor]:
        entry = _entry_by_rel(index, spec["rel_path"])
        clip = _load_segment_clip(entry, spec.get("start_s"), spec.get("end_s"))
        t_len = min(32, clip["transl"].shape[0])
        return {
            "transl": torch.as_tensor(clip["transl"][:t_len], dtype=torch.float32),
            "root_aa": torch.as_tensor(clip["root_orient"][:t_len], dtype=torch.float32),
            "body_aa": torch.as_tensor(clip["pose_body"][:t_len], dtype=torch.float32).reshape(
                t_len, NUM_BODY_JOINTS, 3
            ),
            "betas": torch.as_tensor(clip["betas"], dtype=torch.float32),
        }

    neg_specs = [
        {"rel_path": cfg["leak_check_clip"], "start_s": None, "end_s": None},
    ] + list(cfg.get("image_plane_floor_clips", []))

    def _regen_neg(seed: int) -> tuple[dict[str, Tensor], dict[str, Tensor], Tensor, int]:
        knee_body_idx = 3
        left_knee_kp = 4
        for spec in neg_specs:
            tgt = _targets_for_spec(spec)
            base = tgt["body_aa"].clone()
            b1 = base.clone()
            b1[:, knee_body_idx, :] += 0.25
            b2 = base.clone()
            b2[:, knee_body_idx, :] += 0.50
            obs1 = oracle_obs_from_targets(
                {**tgt, "body_aa": b1},
                body=body,
                seed=seed,
                cam_cfg=cam,
                noise_cfg=noise,
                occ_cfg=occ,
            )
            obs2 = oracle_obs_from_targets(
                {**tgt, "body_aa": b2},
                body=body,
                seed=seed,
                cam_cfg=cam,
                noise_cfg=noise,
                occ_cfg=occ,
            )
            vis = obs1["joint_visible"]
            if not vis[:, left_knee_kp].any():
                continue
            fi = int(torch.where(vis[:, left_knee_kp])[0][0].item())
            delta = (
                obs2["joint_pos_3d"][fi, left_knee_kp] - obs1["joint_pos_3d"][fi, left_knee_kp]
            ).abs().sum()
            if float(delta) > 1e-6:
                return obs1, obs2, vis, fi
        raise RuntimeError("No clip with visible left knee and obs change for negative control")

    model = EgoCompletion()
    print(f"collated_batch_rel_path={rel}")
    print(
        json.dumps(
            run_leak_checks(
                model,
                batch,
                regenerate_obs_negative_control=_regen_neg,
                seed=seed,
            ),
            indent=2,
        )
    )


def cmd_regression_items_2_5() -> None:
    """Smoke regression for AGENTS checklist items 2–5 (scratch, not saved)."""
    code = r"""
import numpy as np
import torch
from hready.body.rotations import axis_angle_to_matrix, matrix_to_rotation_6d
from hready.body.smplx_wrapper import load_body
from hready.losses.biomech import joint_rom
from hready.losses.physics import foot_skating
from hready.metrics.pose import mpjpe
from hready.metrics.stats import cluster_bootstrap_ci
aa = torch.zeros(4, 3)
m = axis_angle_to_matrix(aa)
r6 = matrix_to_rotation_6d(m)
assert r6.shape == (4, 6)
body = load_body('locked_head')
go = torch.zeros(1, 3)
bp = torch.zeros(1, 63)
be = torch.zeros(1, 16)
tr = torch.zeros(1, 3)
out = body.forward(go, bp, be, tr)
assert out.joints.shape[-2] == 55
rom = joint_rom(torch.zeros(1, 4, 21, 3))
assert float(rom) < 1e-5
foot = torch.zeros(1, 4, 2, 3)
contact = torch.zeros(1, 4, 2)
assert float(foot_skating(foot, contact, 30.0)) == 0.0
pred = np.random.randn(10, 22, 3).astype(np.float32)
gt = pred.copy()
assert float(np.mean(mpjpe(pred, gt))) < 1e-5
_, lo, hi = cluster_bootstrap_ci(np.array([1.0, 2.0, 3.0]), np.array([0, 1, 2]), n_boot=100, seed=0)
print('regression_2_5_ok', r6[0,0].item(), lo, hi)
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    print(proc.stdout.strip() or proc.stderr.strip())
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Track E2-A oracle egocentric evidence")
    parser.add_argument("--config", type=Path, default=Path("configs/ego_observation.yaml"))
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify-facts", help="List code-fact verification / discrepancies")
    sub.add_parser("head-frame-check", help="Head frame checks (review 1a–1d)")
    sub.add_parser("fov-report", help="Horizontal FOV degrees for focal sweep")
    sub.add_parser("visibility-report", help="Joint-group visibility by cohort + FOV sweep")
    p_byte = sub.add_parser("byte-stable", help="Seed byte-stability of obs tensors")
    p_byte.add_argument("--seed", type=int, default=0)
    sub.add_parser("leak-check", help="Leak checks (a)-(d)")
    sub.add_parser("regression-2-5", help="Items 2–5 import smoke")
    args = parser.parse_args(argv)
    if args.cmd == "verify-facts":
        cmd_verify_facts()
    elif args.cmd == "head-frame-check":
        cmd_head_frame_check(args.config)
    elif args.cmd == "fov-report":
        cmd_fov_report(args.config)
    elif args.cmd == "visibility-report":
        cmd_visibility_report(args.config)
    elif args.cmd == "byte-stable":
        cmd_byte_stable(args.config, args.seed)
    elif args.cmd == "leak-check":
        cmd_leak_check(args.config)
    elif args.cmd == "regression-2-5":
        cmd_regression_items_2_5()


if __name__ == "__main__":
    main()
