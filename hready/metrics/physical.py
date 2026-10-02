"""Interpretable physical QA metrics (numpy-only, non-differentiable).

Conventions match :mod:`hready.losses.physics`: Z-up world, floor ``z = 0``, meters in.
Uses :func:`hready.losses._constants.com_from_joints_numpy` for center of mass.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import numpy as np

from hready.losses._constants import com_from_joints_numpy

_EPS = 1e-8


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _as_time_series(x: np.ndarray) -> np.ndarray:
    """Return ``(T, ...)`` from ``(T, ...)`` or batched ``(1, T, ...)`` (4D only)."""
    if x.ndim == 4 and x.shape[0] == 1:
        return x[0]
    return x


def foot_skate(
    foot_pos: Any,
    contact: Any,
    fps: float,
    *,
    contact_thresh: float = 0.5,
) -> float:
    """Mean horizontal foot speed (m/s) while in contact.

    Averages over **all in-contact feet** at each velocity step (not only the
    moving foot). ``foot_pos``: ``(T, F, 3)`` meters; ``contact``: ``(T, F)``.
    """
    pos = _as_time_series(_to_numpy(foot_pos))
    w = _as_time_series(_to_numpy(contact))
    t = pos.shape[0]
    if t < 2:
        return 0.0
    vel = (pos[1:] - pos[:-1]) * fps
    horiz = np.linalg.norm(vel[..., :2], axis=-1)
    mask = w[1:] > contact_thresh
    if not np.any(mask):
        return 0.0
    return float(horiz[mask].mean())


def ground_penetration(
    verts_or_joints: Any,
) -> Tuple[float, float]:
    """Ground penetration depth: ``(mean_mm, max_mm)`` for points with ``z < 0``.

    ``verts_or_joints``: ``(T, P, 3)`` meters.
    """
    pts = _as_time_series(_to_numpy(verts_or_joints))
    z = pts[..., 2]
    pen = np.maximum(0.0, -z)
    if pen.size == 0:
        return 0.0, 0.0
    return float(pen.mean() * 1000.0), float(pen.max() * 1000.0)


def foot_float(
    foot_pos: Any,
    contact: Any,
    *,
    contact_thresh: float = 0.5,
) -> float:
    """Mean foot height ``z`` (mm) while contact is active (should be ~0 on floor)."""
    pos = _as_time_series(_to_numpy(foot_pos))
    w = _as_time_series(_to_numpy(contact))
    mask = w > contact_thresh
    if not np.any(mask):
        return 0.0
    return float(pos[..., 2][mask].mean() * 1000.0)


def _joint_accel_jerk(joints: np.ndarray, fps: float) -> Tuple[np.ndarray, np.ndarray]:
    """``joints`` ``(T, J, 3)`` -> accel ``(T-2, J)``, jerk ``(T-3, J)`` magnitudes."""
    vel = (joints[1:] - joints[:-1]) * fps
    acc = (vel[1:] - vel[:-1]) * fps
    jerk = (acc[1:] - acc[:-1]) * fps
    acc_mag = np.linalg.norm(acc, axis=-1)
    jerk_mag = np.linalg.norm(jerk, axis=-1)
    return acc_mag, jerk_mag


def accel_error(joints: Any, fps: float) -> float:
    """Mean joint acceleration magnitude (m/s²) over frames and joints."""
    j = _as_time_series(_to_numpy(joints))
    if j.shape[0] < 3:
        return 0.0
    acc_mag, _ = _joint_accel_jerk(j, fps)
    return float(acc_mag.mean())


def jitter(joints: Any, fps: float) -> float:
    """Mean joint jerk magnitude (m/s³) over frames and joints."""
    j = _as_time_series(_to_numpy(joints))
    if j.shape[0] < 4:
        return 0.0
    _, jerk_mag = _joint_accel_jerk(j, fps)
    return float(jerk_mag.mean())


def _cross_2d(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
    pts = points
    if pts.shape[0] <= 2:
        return pts
    pts = pts[np.argsort(pts[:, 0])]
    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and _cross_2d(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[np.ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and _cross_2d(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.stack(lower[:-1] + upper[:-1], axis=0)
    return hull


def _dist_point_segment_2d(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = max(float(ab @ ab), _EPS * _EPS)
    t = float(np.clip(((p - a) @ ab) / denom, 0.0, 1.0))
    closest = a + t * ab
    return float(np.linalg.norm(p - closest))


def _point_on_segment_2d(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> bool:
    ab = b - a
    t = float((p - a) @ ab) / max(float(ab @ ab), _EPS * _EPS)
    on = 0.0 <= t <= 1.0
    perp = np.linalg.norm(p - (a + np.clip(t, 0.0, 1.0) * ab))
    return bool(on and perp < 1e-4)


def _inside_convex_ccw(p: np.ndarray, hull: np.ndarray) -> bool:
    k = hull.shape[0]
    if k < 3:
        return False
    sign_ok = True
    for i in range(k):
        c = _cross_2d(hull[i], hull[(i + 1) % k], p)
        sign_ok = sign_ok and c >= -1e-6
    return sign_ok


def _dist_to_support_xy(p: np.ndarray, pts: np.ndarray) -> float:
    k = pts.shape[0]
    if k == 1:
        return float(np.linalg.norm(p - pts[0]))
    if k == 2:
        d = _dist_point_segment_2d(p, pts[0], pts[1])
        if _point_on_segment_2d(p, pts[0], pts[1]):
            return 0.0
        return d
    hull = _convex_hull_2d(pts)
    if _inside_convex_ccw(p, hull):
        return 0.0
    edges = [
        _dist_point_segment_2d(p, hull[i], hull[(i + 1) % hull.shape[0]])
        for i in range(hull.shape[0])
    ]
    return float(min(edges))


def _com_numpy(joints: np.ndarray) -> np.ndarray:
    return com_from_joints_numpy(joints)


def balance_violation(
    joints: Any,
    foot_pts: Any,
    contact: Any,
    *,
    contact_thresh: float = 0.5,
) -> Tuple[float, float]:
    """Balance metrics on contact frames.

    Returns ``(fraction_outside, mean_distance_outside_mm)`` where distance is
    CoM XY distance to the support polygon (0 if inside). Uses
    :func:`~hready.losses._constants.com_from_joints_numpy` on ``joints`` ``(T, J, 3)``.
    """
    joints = _to_numpy(joints)
    if joints.ndim == 2:
        joints = joints.reshape(1, joints.shape[0], joints.shape[1])
    joints = _as_time_series(joints)
    foot_pts = _to_numpy(foot_pts)
    contact = _to_numpy(contact)
    if foot_pts.ndim == 2:
        foot_pts = foot_pts.reshape(1, foot_pts.shape[0], foot_pts.shape[1])
    if contact.ndim == 1:
        contact = contact.reshape(1, -1)
    foot_pts = _as_time_series(foot_pts)
    contact = _as_time_series(contact)
    t = min(joints.shape[0], foot_pts.shape[0], contact.shape[0])
    com = _com_numpy(joints)
    if com.ndim == 1:
        com = com.reshape(1, 3)
    com_xy = com[:, :2]
    outside_dists: list[float] = []
    contact_frames = 0
    for ti in range(t):
        mask = contact[ti] > contact_thresh
        if not np.any(mask):
            continue
        contact_frames += 1
        pts = foot_pts[ti, mask, :2]
        d = _dist_to_support_xy(com_xy[ti], pts)
        if d > 1e-5:
            outside_dists.append(d)
    if contact_frames == 0:
        return 0.0, 0.0
    frac = len(outside_dists) / contact_frames
    mean_out = float(np.mean(outside_dists) * 1000.0) if outside_dists else 0.0
    return float(frac), mean_out
