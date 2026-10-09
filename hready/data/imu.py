"""Synthetic body-worn IMU from AMASS clips (item 5, synthetic IMU and QA)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

# SMPL-X joint indices (vertices2joints / 55-joint layout).
DEFAULT_SENSOR_JOINT_IDS: tuple[int, ...] = (0, 15, 20, 21, 7, 8)
DEFAULT_SENSOR_NAMES: tuple[str, ...] = (
    "pelvis",
    "head",
    "left_wrist",
    "right_wrist",
    "left_ankle",
    "right_ankle",
)
_GRAVITY_WORLD = np.array([0.0, 0.0, -9.81], dtype=np.float64)


def synthetic_imu_from_clip(
    clip: dict[str, Any],
    *,
    sensor_joint_ids: Sequence[int] = DEFAULT_SENSOR_JOINT_IDS,
    sensor_names: Sequence[str] = DEFAULT_SENSOR_NAMES,
    acc_noise_std: float = 0.0,
    gyro_bias: float | np.ndarray = 0.0,
    device: str | None = "cpu",
    body: Any | None = None,
    chunk_size: int = 128,
) -> dict[str, Any]:
    """Synthesize accelerometer and gyroscope streams from a grounded AMASS clip.

    Expects ``load_clip(..., ground=True)`` fields: ``root_orient`` (T,3),
    ``pose_body`` (T,21,3), ``transl`` (T,3), ``betas``, ``fps`` (30).

    **World frame:** AMASS loader Z-up, floor at ``z=0`` after grounding.

    **Sensor frame:** IMU axes match the SMPL-X *global* joint orientation at each
    frame (rotation matrix ``R`` from forward kinematics). Accelerometer reports
    *specific force* in the sensor frame::

        f_s = R^T (a_world - g)

    with ``g = (0, 0, -9.81)`` m/s². A static upright sensor therefore reads
    ``+9.81`` m/s² along world +Z when ``R = I``.

    **Differentiation:** ``a_world`` is the second time derivative of joint
    position in world coordinates using central differences on interior frames;
    first and last frames use one-sided quadratic stencils (no NaN padding):
    ``t=0`` uses ``(x[2]-2*x[1]+x[0])/dt²``, ``t=T-1`` uses
    ``(x[T-1]-2*x[T-2]+x[T-3])/dt²``; for ``T=2`` both frames use
    ``(x[1]-x[0])/dt²``. At **30 fps** this amplifies mocap noise (~mm-level
    position jitter becomes tens of m/s²); treat as kinematic IMU without sensor
    dynamics.

    **Gyroscope:** ``omega_s = log(R_t^T R_{t+1}) * fps`` (axis-angle log map,
    rad/s) for ``t=0…T-2``. The last frame **reuses** ``gyro[T-2]`` (no NaN pad).

    Returns dict with ``acc`` (T, S, 3), ``gyro`` (T, S, 3) float64,
    ``sensor_joint_ids``, ``sensor_names``, ``fps``.
    """
    ids = tuple(int(i) for i in sensor_joint_ids)
    names = tuple(sensor_names)
    if len(ids) != len(names):
        raise ValueError("sensor_joint_ids and sensor_names length mismatch")

    root = np.asarray(clip["root_orient"], dtype=np.float64)
    body_aa = np.asarray(clip["pose_body"], dtype=np.float64).reshape(-1, 21, 3)
    transl = np.asarray(clip["transl"], dtype=np.float64)
    betas = np.asarray(clip["betas"], dtype=np.float64).reshape(-1)
    fps = float(clip["fps"])
    t = root.shape[0]
    if body_aa.shape[0] != t or transl.shape[0] != t:
        raise ValueError("clip pose/transl length mismatch")

    pos_w, rot_w = _joint_kinematics_series(
        root, body_aa, transl, betas, device=device, body=body, chunk_size=chunk_size
    )
    dt = 1.0 / fps
    a_world = _central_accel_2d(pos_w, dt)
    g = _GRAVITY_WORLD
    acc = np.zeros((t, len(ids), 3), dtype=np.float64)
    for si, ji in enumerate(ids):
        aw = a_world[:, ji, :]
        rw = rot_w[:, ji, :, :]
        acc[:, si, :] = np.einsum("tij,tj->ti", rw.transpose(0, 2, 1), aw - g)

    gyro = _gyro_from_rotations(rot_w[:, ids, :, :], fps)

    if acc_noise_std > 0.0:
        acc += np.random.randn(*acc.shape) * acc_noise_std
    bias = np.asarray(gyro_bias, dtype=np.float64)
    if bias.ndim == 0:
        gyro = gyro + float(bias)
    else:
        gyro = gyro + bias.reshape(1, -1, 3)

    return {
        "acc": acc,
        "gyro": gyro,
        "sensor_joint_ids": ids,
        "sensor_names": names,
        "fps": fps,
    }


def _joint_kinematics_series(
    root_orient: np.ndarray,
    pose_body: np.ndarray,
    transl: np.ndarray,
    betas: np.ndarray,
    *,
    device: str | None,
    body: Any | None,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from smplx.lbs import batch_rigid_transform, blend_shapes, vertices2joints

    from hready.body.rotations import axis_angle_to_matrix
    from hready.body.smplx_wrapper import load_body
    from hready.data.amass import _resolve_torch_device

    dev = _resolve_torch_device(device)
    if body is None:
        body = load_body("locked_head")
        body._model.to(dev)
    model = body._model
    parents = model.parents.to(dev)
    n = root_orient.shape[0]
    pos_chunks: list[np.ndarray] = []
    rot_chunks: list[np.ndarray] = []
    be0 = torch.as_tensor(betas, device=dev, dtype=torch.float32).reshape(1, -1)
    v_t = model.v_template.unsqueeze(0)
    shaped = v_t + blend_shapes(be0, model.shapedirs)
    j0 = vertices2joints(model.J_regressor, shaped)

    for i0 in range(0, n, chunk_size):
        i1 = min(n, i0 + chunk_size)
        b = i1 - i0
        go = torch.as_tensor(root_orient[i0:i1], device=dev, dtype=torch.float32)
        bp = torch.as_tensor(pose_body[i0:i1], device=dev, dtype=torch.float32).reshape(
            b, 63
        )
        tr = torch.as_tensor(transl[i0:i1], device=dev, dtype=torch.float32)
        z3 = torch.zeros(b, 3, device=dev, dtype=torch.float32)
        z45 = torch.zeros(b, 45, device=dev, dtype=torch.float32)
        full_pose = torch.cat([go, bp, z3, z3, z3, z45, z45], dim=1)
        rot_mats = axis_angle_to_matrix(full_pose.reshape(-1, 3)).reshape(b, 55, 3, 3)
        j = j0.expand(b, -1, -1)
        _, a_mats = batch_rigid_transform(rot_mats, j, parents)
        rot_g = a_mats[:, :, :3, :3].detach().cpu().numpy().astype(np.float64)
        pos = (
            body.forward(go, bp, be0.expand(b, -1), tr)
            .joints.detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        pos_chunks.append(pos)
        rot_chunks.append(rot_g)
    return np.concatenate(pos_chunks, axis=0), np.concatenate(rot_chunks, axis=0)


def _central_accel_2d(pos: np.ndarray, dt: float) -> np.ndarray:
    """Second derivative of ``pos`` (T, J, 3) w.r.t. time; explicit end stencils."""
    t = pos.shape[0]
    out = np.zeros_like(pos)
    if t == 1:
        return out
    inv_dt2 = 1.0 / (dt * dt)
    if t == 2:
        out[0] = out[1] = (pos[1] - pos[0]) / (dt * dt)
        return out
    out[1:-1] = (pos[2:] - 2.0 * pos[1:-1] + pos[:-2]) * inv_dt2
    # Forward quadratic at t=0: same formula using phantom x[-1] extrapolated
    out[0] = (pos[2] - 2.0 * pos[1] + pos[0]) * inv_dt2
    out[-1] = (pos[-1] - 2.0 * pos[-2] + pos[-3]) * inv_dt2
    return out


def _gyro_from_rotations(rot: np.ndarray, fps: float) -> np.ndarray:
    """``rot`` (T, S, 3, 3) global -> gyro (T, S, 3) rad/s in sensor frame."""
    import torch

    from hready.body.rotations import matrix_to_axis_angle

    t, s, _, _ = rot.shape
    gyro = np.zeros((t, s, 3), dtype=np.float64)
    if t < 2:
        return gyro
    r0 = torch.as_tensor(rot[:-1], dtype=torch.float64)
    r1 = torch.as_tensor(rot[1:], dtype=torch.float64)
    rel = torch.matmul(r0.transpose(-1, -2), r1)
    omega = matrix_to_axis_angle(rel.reshape(-1, 3, 3)).reshape(t - 1, s, 3) * fps
    gyro[:-1] = omega.detach().cpu().numpy()
    gyro[-1] = gyro[-2]
    return gyro


def hold_clip_at_frame(
    clip: dict[str, Any],
    frame: int = 0,
    n_frames: int = 90,
) -> dict[str, Any]:
    """Repeat one grounded frame to synthesize a stationary segment (verification)."""
    fi = int(frame)
    t = int(clip["root_orient"].shape[0])
    if fi < 0 or fi >= t:
        raise IndexError(frame)
    n = int(n_frames)
    out = dict(clip)
    out["root_orient"] = np.repeat(clip["root_orient"][fi : fi + 1], n, axis=0)
    out["transl"] = np.repeat(clip["transl"][fi : fi + 1], n, axis=0)
    out["pose_body"] = np.repeat(clip["pose_body"][fi : fi + 1], n, axis=0)
    return out


def static_verification_clip(
    *,
    cache_dir: Any | None = None,
    n_frames: int = 90,
) -> tuple[str, int, dict[str, Any]]:
    """Idle / stand source clip + held frame for |acc| ≈ 9.81 m/s² checks."""
    from hready.data.amass import load_index

    cache_dir = cache_dir or __import__(
        "hready.data.amass", fromlist=["cache_dir_from_config"]
    ).cache_dir_from_config()
    indexed = {e.rel_path for e in load_index(cache_dir)}
    stand_ref = "CMU/91/91_48_stageii.npz"
    rel = stand_ref if stand_ref in indexed else find_near_static_clip_rel(cache_dir=cache_dir)
    clip = __import__("hready.data.amass", fromlist=["load_clip"]).load_clip(
        rel, ground=True, cache_dir=cache_dir
    )
    pose = clip["pose_body"].reshape(clip["pose_body"].shape[0], -1)
    if pose.shape[0] > 1:
        step = np.linalg.norm(np.diff(pose, axis=0), axis=1)
        fi = int(np.argmin(step))
    else:
        fi = 0
    held = hold_clip_at_frame(clip, fi, n_frames=n_frames)
    return rel, fi, held


def find_near_static_clip_rel(
    *,
    cache_dir: Any | None = None,
    max_candidates: int = 40,
) -> str:
    """Pick a short AMASS clip with low pelvis motion (standing / idle proxy)."""
    from hready.data.amass import load_clip, load_index
    from hready.data.babel import load_babel_index_payload

    cache_dir = cache_dir or __import__(
        "hready.data.amass", fromlist=["cache_dir_from_config"]
    ).cache_dir_from_config()
    payload = load_babel_index_payload(cache_dir)
    indexed = {e.rel_path for e in load_index(cache_dir)}
    cands: list[str] = []
    for rec in payload["entries"]:
        cats = []
        for seg in rec.get("segments") or []:
            cats.extend(seg.get("act_cat") or [])
        if any("stand" in str(c).lower() for c in cats):
            rel = rec["rel_path"]
            if rel in indexed:
                cands.append(rel)
    if not cands:
        cands = []
    if "CMU/91/91_48_stageii.npz" in indexed:
        cands.insert(0, "CMU/91/91_48_stageii.npz")
    if not cands:
        cands = ["CMU/91/91_48_stageii.npz"]
    best_rel = cands[0]
    best_score = float("inf")
    for rel in cands[:max_candidates]:
        try:
            clip = load_clip(rel, ground=True, cache_dir=cache_dir)
        except (KeyError, OSError, ValueError):
            continue
        v = np.linalg.norm(np.diff(clip["transl"], axis=0), axis=1)
        score = float(np.median(v))
        if score < best_score:
            best_score = score
            best_rel = rel
    return best_rel


def ballistic_imu_free_fall(
    n_frames: int = 60,
    fps: float = 30.0,
    z0: float = 1.0,
    vz0: float = 0.0,
) -> dict[str, np.ndarray]:
    """Point mass under gravity for IMU sanity (specific force ~ 0 in free fall)."""
    t = np.arange(n_frames, dtype=np.float64) / fps
    z = z0 + vz0 * t - 0.5 * 9.81 * t * t
    pos = np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=-1)[:, None, :]
    dt = 1.0 / fps
    a_world = _central_accel_2d(pos, dt)[:, 0, :]
    rot = np.tile(np.eye(3), (n_frames, 1, 1, 1))
    g = _GRAVITY_WORLD
    spec = a_world - g
    acc = np.einsum("tij,tj->ti", rot[:, 0].transpose(0, 2, 1), spec)[:, None, :]
    gyro = np.zeros((n_frames, 1, 3), dtype=np.float64)
    return {"acc": acc, "gyro": gyro, "a_world": a_world, "fps": fps}
