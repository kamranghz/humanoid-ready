"""Physics losses for human motion (kinematic inputs).

Conventions
-----------
- World frame: Z-up, floor at ``z = 0``.
- Units: meters, seconds.
- Joint tensors: ``(B, T, J, 3)`` positions in the world frame.
- ``fps``: frames per second (scalar or 0-dim tensor).
- Gravity magnitude: ``g = 9.81`` m/s² (downward → negative Z acceleration in flight).

All losses are pure PyTorch, batched, and differentiable. No body-model imports.
"""

from __future__ import annotations

import torch
from torch import Tensor

G: float = 9.81
_EPS: float = 1e-8

# Segment mass (% total body mass), male/female from de Leva (1996) Table 1 /
# Visual3D documentation (Zatsiorsky-Seluyanov adjusted), averaged (F+M)/2:
# head 6.81%, trunk 43.015%, upper arm 2.63%, forearm 1.50%, hand 0.585%,
# thigh 14.47%, shank 4.57%, foot 1.33% (each side where bilateral).
COM_MASS_FRAC_SOURCE = (
    "de Leva (1996) segment mass fractions (%TBW), averaged male/female tables; "
    "mapped to SMPL-X body joint positions 0..21 (see com_from_joints docstring)."
)
# Joint order matches SMPL-X kinematic tree indices 0..21 (pelvis … wrists).
# Mapping: trunk split equally across pelvis+spine1–3+neck; thigh→hip, shank→knee,
# foot→foot joint; upper arm→shoulder, forearm→elbow, hand→wrist; collar/ankle=0.
_COM_RAW_22: tuple[float, ...] = (
    0.43015 / 5,  # 0 pelvis — trunk share
    0.1447,  # 1 L hip — L thigh
    0.1447,  # 2 R hip — R thigh
    0.43015 / 5,  # 3 spine1
    0.0457,  # 4 L knee — L shank
    0.0457,  # 5 R knee — R shank
    0.43015 / 5,  # 6 spine2
    0.0,  # 7 L ankle (mass on foot joint)
    0.0,  # 8 R ankle
    0.43015 / 5,  # 9 spine3
    0.0133,  # 10 L foot
    0.0133,  # 11 R foot
    0.43015 / 5,  # 12 neck — trunk share
    0.0,  # 13 L collar
    0.0,  # 14 R collar
    0.0681,  # 15 head
    0.0263,  # 16 L shoulder — L upper arm
    0.0263,  # 17 R shoulder
    0.0150,  # 18 L elbow — L forearm
    0.0150,  # 19 R elbow
    0.00585,  # 20 L wrist — L hand
    0.00585,  # 21 R wrist
)
_s = sum(_COM_RAW_22)
COM_MASS_FRAC_22: tuple[float, ...] = tuple(v / _s for v in _COM_RAW_22)


def com_mass_fractions(device=None, dtype=torch.float32) -> Tensor:
    """Return the 22 segment mass fractions (sum = 1)."""
    frac = torch.tensor(COM_MASS_FRAC_22, device=device, dtype=dtype)
    return frac / frac.sum().clamp_min(_EPS)


def _safe_norm(x: Tensor, dim: int = -1) -> Tensor:
    return torch.sqrt((x * x).sum(dim=dim) + _EPS * _EPS)


def com_from_joints(
    joints: Tensor,
    segment_mass_frac: Tensor | None = None,
) -> Tensor:
    """Center of mass from the first 22 SMPL-X body joint positions.

    Mass fractions follow de Leva (1996) segment masses (% total body mass),
    averaged over published male and female columns, then mapped to joint
    indices 0..21 (pelvis, L/R hip, spine1, L/R knee, spine2, L/R ankle,
    spine3, L/R foot, neck, L/R collar, head, L/R shoulder, L/R elbow,
    L/R wrist). Trunk mass is split equally across pelvis, spine1–3, and
    neck; bilateral segments use hip/knee/foot and shoulder/elbow/wrist pairs;
    collar and ankle joints carry zero (foot mass on toe/foot joint). The
    vector is renormalized to sum 1. Each segment mass is placed at the
    proximal joint of that segment (kinematic approximation). Not
    subject-specific anthropometry.
    """
    if joints.shape[-2] < 22:
        raise ValueError(f"com_from_joints expects J >= 22, got {joints.shape[-2]}")
    j22 = joints[..., :22, :]
    if segment_mass_frac is None:
        frac = com_mass_fractions(device=joints.device, dtype=joints.dtype)
    else:
        frac = segment_mass_frac
        if frac.shape[-1] != 22:
            raise ValueError("segment_mass_frac must have length 22")
        frac = frac / frac.sum().clamp_min(_EPS)
    frac = frac.reshape(*(1,) * (joints.dim() - 2), 22, 1)
    return (frac * j22).sum(dim=-2)


def foot_skating(foot_pos: Tensor, contact: Tensor, fps: Tensor | float) -> Tensor:
    """Contact-weighted squared horizontal foot speed (m²/s² mean)."""
    fps_t = torch.as_tensor(fps, device=foot_pos.device, dtype=foot_pos.dtype)
    vel = (foot_pos[:, 1:] - foot_pos[:, :-1]) * fps_t
    horiz = vel[..., :2]
    speed_sq = horiz.square().sum(dim=-1)
    w = contact[:, 1:].clamp(0.0, 1.0)
    return (w * speed_sq).mean()


def ground_penetration(verts_or_joints: Tensor) -> Tensor:
    """Mean squared hinge penalty for points with ``z < 0`` (m²)."""
    z = verts_or_joints[..., 2]
    pen = torch.relu(-z).square()
    return pen.mean()


def flight_consistency(
    com: Tensor,
    contact: Tensor,
    fps: Tensor | float,
    g: float = G,
) -> Tensor:
    """Ballistic prior when no foot has meaningful contact.

    Penalizes vertical acceleration deviating from ``-g`` and horizontal
    velocity changes between consecutive frames (momentum).
    """
    fps_t = torch.as_tensor(fps, device=com.device, dtype=com.dtype)
    flight = contact.clamp(0.0, 1.0).sum(dim=-1) < 0.5
    if com.shape[1] < 3:
        return torch.zeros((), device=com.device, dtype=com.dtype)
    vel = (com[:, 1:] - com[:, :-1]) * fps_t
    acc = (vel[:, 1:] - vel[:, :-1]) * fps_t
    flight_mid = flight[:, 1:-1]
    if not flight_mid.any():
        return torch.zeros((), device=com.device, dtype=com.dtype)
    az = acc[..., 2]
    vert_err = (az + g).square()
    horiz_vel = vel[..., :2]
    horiz_change = (horiz_vel[:, 1:] - horiz_vel[:, :-1]).square().sum(dim=-1)
    return (flight_mid.float() * (vert_err + horiz_change)).mean()


def _dist_point_to_segment_2d(p: Tensor, a: Tensor, b: Tensor) -> Tensor:
    ab = b - a
    denom = (ab * ab).sum(dim=-1, keepdim=True).clamp_min(_EPS * _EPS)
    t = ((p - a) * ab).sum(dim=-1, keepdim=True) / denom
    t = t.clamp(0.0, 1.0)
    closest = a + t * ab
    return _safe_norm(p - closest, dim=-1)


def balance_com_in_support(
    com: Tensor,
    foot_pts: Tensor,
    contact: Tensor,
    contact_thresh: float = 0.5,
) -> Tensor:
    """Distance from CoM ground projection to the support polygon (m).

    Support is the 2D convex hull of in-contact feet on the floor plane.
    Zero when the projection lies inside the hull, when only one support point
    (distance to that point), or when no foot is in contact (flight).
    """
    if com.dim() == 2:
        com = com.unsqueeze(0)
    if foot_pts.dim() == 3:
        foot_pts = foot_pts.unsqueeze(0)
        contact = contact.unsqueeze(0)
    com_xy = com[..., :2]
    feet_xy = foot_pts[..., :2]
    active = contact > contact_thresh
    b, t, f, _ = feet_xy.shape
    dists = []
    for bi in range(b):
        for ti in range(t):
            mask = active[bi, ti]
            if not mask.any():
                continue
            pts = feet_xy[bi, ti, mask]
            p = com_xy[bi, ti]
            k = pts.shape[0]
            if k == 1:
                d = _safe_norm(p - pts[0])
            elif k == 2:
                d = _dist_point_to_segment_2d(p, pts[0], pts[1])
                inside = _point_in_triangle_or_segment(p, pts[0], pts[1])
                d = torch.where(inside, torch.zeros_like(d), d)
            else:
                hull = _convex_hull_2d(pts)
                d = _dist_to_convex_polygon_2d(p, hull)
            dists.append(d)
    if not dists:
        return torch.zeros((), device=com.device, dtype=com.dtype)
    return torch.stack(dists).mean()


def _point_in_triangle_or_segment(p: Tensor, a: Tensor, b: Tensor) -> Tensor:
    """True if ``p`` projects onto the segment ``ab`` (including endpoints)."""
    ab = b - a
    ap = p - a
    t = (ap * ab).sum() / (ab.square().sum() + _EPS * _EPS)
    on_seg = (t >= 0.0) & (t <= 1.0)
    perp = _safe_norm(ap - t.clamp(0.0, 1.0) * ab)
    return on_seg & (perp < 1e-4)


def _convex_hull_2d(points: Tensor) -> Tensor:
    """Monotone-chain convex hull, CCW, shape ``(K, 2)``."""
    pts = points
    if pts.shape[0] <= 2:
        return pts
    pts = pts[torch.argsort(pts[:, 0])]
    lower: list[Tensor] = []
    for p in pts:
        while len(lower) >= 2 and _cross_2d(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[Tensor] = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross_2d(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return torch.stack(hull)


def _cross_2d(o: Tensor, a: Tensor, b: Tensor) -> Tensor:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _point_in_convex_ccw(p: Tensor, hull: Tensor) -> Tensor:
    k = hull.shape[0]
    if k < 3:
        return torch.tensor(False, device=p.device)
    sign = None
    for i in range(k):
        c = _cross_2d(hull[i], hull[(i + 1) % k], p)
        if sign is None:
            sign = c >= -1e-6
        else:
            sign = sign & (c >= -1e-6)
    return sign


def _dist_to_convex_polygon_2d(p: Tensor, hull: Tensor) -> Tensor:
    if hull.shape[0] == 1:
        return _safe_norm(p - hull[0])
    if hull.shape[0] == 2:
        d = _dist_point_to_segment_2d(p, hull[0], hull[1])
        inside = _point_in_triangle_or_segment(p, hull[0], hull[1])
        return torch.where(inside, torch.zeros_like(d), d)
    if _point_in_convex_ccw(p, hull):
        return torch.zeros((), device=p.device, dtype=p.dtype)
    k = hull.shape[0]
    edge_d = [_dist_point_to_segment_2d(p, hull[i], hull[(i + 1) % k]) for i in range(k)]
    d = torch.stack(edge_d).min()
    on_edge = d < 1e-5
    return torch.where(on_edge, torch.zeros_like(d), d)


def smoothness_terms(pos: Tensor, fps: Tensor | float) -> tuple[Tensor, Tensor]:
    """Mean squared acceleration and jerk (finite differences, forward)."""
    fps_t = torch.as_tensor(fps, device=pos.device, dtype=pos.dtype)
    if pos.shape[1] < 3:
        z = torch.zeros((), device=pos.device, dtype=pos.dtype)
        return z, z
    vel = (pos[:, 1:] - pos[:, :-1]) * fps_t
    acc = (vel[:, 1:] - vel[:, :-1]) * fps_t
    jerk = (acc[:, 1:] - acc[:, :-1]) * fps_t
    return acc.square().mean(), jerk.square().mean()


def smoothness(
    pos: Tensor,
    fps: Tensor | float,
    *,
    accel_weight: float = 1.0,
    jerk_weight: float = 1.0,
) -> Tensor:
    """Acceleration and jerk penalties via finite differences."""
    la, lj = smoothness_terms(pos, fps)
    return accel_weight * la + jerk_weight * lj
