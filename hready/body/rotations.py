"""Differentiable SO(3) utilities (pure PyTorch)."""

from __future__ import annotations

import torch
from torch import Tensor

_EPS = 1e-8


def _normalize(v: Tensor, dim: int = -1) -> Tensor:
    return v / (v.norm(dim=dim, keepdim=True).clamp_min(_EPS))


def axis_angle_to_matrix(axis_angle: Tensor) -> Tensor:
    """Axis-angle (B, 3) -> rotation matrix (B, 3, 3)."""
    aa = axis_angle.reshape(-1, 3)
    angle = aa.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    axis = aa / angle
    x, y, z = axis.unbind(-1)
    ca = torch.cos(angle.squeeze(-1))
    sa = torch.sin(angle.squeeze(-1))
    one_c = 1.0 - ca
    b = axis_angle.shape[:-1]
    out = torch.zeros(*b, 3, 3, device=axis_angle.device, dtype=axis_angle.dtype)
    out[..., 0, 0] = ca + x * x * one_c
    out[..., 0, 1] = x * y * one_c - z * sa
    out[..., 0, 2] = x * z * one_c + y * sa
    out[..., 1, 0] = y * x * one_c + z * sa
    out[..., 1, 1] = ca + y * y * one_c
    out[..., 1, 2] = y * z * one_c - x * sa
    out[..., 2, 0] = z * x * one_c - y * sa
    out[..., 2, 1] = z * y * one_c + x * sa
    out[..., 2, 2] = ca + z * z * one_c
    return out


def matrix_to_axis_angle(matrix: Tensor) -> Tensor:
    """Rotation matrix (..., 3, 3) -> axis-angle (..., 3)."""
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def matrix_to_rotation_6d(matrix: Tensor) -> Tensor:
    """Rotation matrix (..., 3, 3) -> 6D (..., 6) (Zhou et al., first two columns)."""
    c0 = matrix[..., :, 0]
    c1 = matrix[..., :, 1]
    return torch.cat((c0, c1), dim=-1)


def rotation_6d_to_matrix(rot_6d: Tensor) -> Tensor:
    """6D (..., 6) -> rotation matrix (..., 3, 3)."""
    a1 = rot_6d[..., 0:3]
    a2 = rot_6d[..., 3:6]
    b1 = _normalize(a1, dim=-1)
    dot = (b1 * a2).sum(dim=-1, keepdim=True)
    b2 = _normalize(a2 - dot * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def matrix_to_quaternion(matrix: Tensor) -> Tensor:
    """Rotation matrix (..., 3, 3) -> quaternion (..., 4) in (w, x, y, z) order."""
    m = matrix.reshape(-1, 3, 3)
    batch = m.shape[0]
    qw = torch.zeros(batch, device=m.device, dtype=m.dtype)
    qx = torch.zeros(batch, device=m.device, dtype=m.dtype)
    qy = torch.zeros(batch, device=m.device, dtype=m.dtype)
    qz = torch.zeros(batch, device=m.device, dtype=m.dtype)
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    cond0 = trace > 0
    s0 = torch.sqrt(trace[cond0] + 1.0) * 2.0
    qw[cond0] = 0.25 * s0
    qx[cond0] = (m[cond0, 2, 1] - m[cond0, 1, 2]) / s0
    qy[cond0] = (m[cond0, 0, 2] - m[cond0, 2, 0]) / s0
    qz[cond0] = (m[cond0, 1, 0] - m[cond0, 0, 1]) / s0
    cond1 = (~cond0) & (m[:, 0, 0] > m[:, 1, 1]) & (m[:, 0, 0] > m[:, 2, 2])
    s1 = torch.sqrt(1.0 + m[cond1, 0, 0] - m[cond1, 1, 1] - m[cond1, 2, 2]) * 2.0
    qw[cond1] = (m[cond1, 2, 1] - m[cond1, 1, 2]) / s1
    qx[cond1] = 0.25 * s1
    qy[cond1] = (m[cond1, 0, 1] + m[cond1, 1, 0]) / s1
    qz[cond1] = (m[cond1, 0, 2] + m[cond1, 2, 0]) / s1
    cond2 = (~cond0) & (~cond1) & (m[:, 1, 1] > m[:, 2, 2])
    s2 = torch.sqrt(1.0 + m[cond2, 1, 1] - m[cond2, 0, 0] - m[cond2, 2, 2]) * 2.0
    qw[cond2] = (m[cond2, 0, 2] - m[cond2, 2, 0]) / s2
    qx[cond2] = (m[cond2, 0, 1] + m[cond2, 1, 0]) / s2
    qy[cond2] = 0.25 * s2
    qz[cond2] = (m[cond2, 1, 2] + m[cond2, 2, 1]) / s2
    cond3 = (~cond0) & (~cond1) & (~cond2)
    s3 = torch.sqrt(1.0 + m[cond3, 2, 2] - m[cond3, 0, 0] - m[cond3, 1, 1]) * 2.0
    qw[cond3] = (m[cond3, 1, 0] - m[cond3, 0, 1]) / s3
    qx[cond3] = (m[cond3, 0, 2] + m[cond3, 2, 0]) / s3
    qy[cond3] = (m[cond3, 1, 2] + m[cond3, 2, 1]) / s3
    qz[cond3] = 0.25 * s3
    quat = torch.stack((qw, qx, qy, qz), dim=-1)
    quat = _normalize(quat, dim=-1)
    return quat.reshape(*matrix.shape[:-2], 4)


def quaternion_to_matrix(quaternion: Tensor) -> Tensor:
    """Quaternion (..., 4) (w, x, y, z) -> rotation matrix (..., 3, 3)."""
    q = _normalize(quaternion, dim=-1)
    w, x, y, z = q.unbind(-1)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    row0 = torch.stack((1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)), dim=-1)
    row1 = torch.stack((2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)), dim=-1)
    row2 = torch.stack((2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def quaternion_to_axis_angle(quaternion: Tensor) -> Tensor:
    """Quaternion (..., 4) (w, x, y, z) -> axis-angle (..., 3)."""
    q = _normalize(quaternion, dim=-1)
    q = torch.where(q[..., :1] < 0, -q, q)
    w = q[..., 0].clamp(0.0, 1.0)
    xyz = q[..., 1:]
    sin_half = xyz.norm(dim=-1)
    small = sin_half < 1e-6
    sin_safe = torch.sqrt(sin_half * sin_half + _EPS * _EPS)
    angle = 2.0 * torch.atan2(sin_safe, torch.sqrt(w * w + _EPS * _EPS))
    axis = xyz / sin_safe.unsqueeze(-1)
    aa = axis * angle.unsqueeze(-1)
    return torch.where(small.unsqueeze(-1), torch.zeros_like(aa), aa)


def axis_angle_to_quaternion(axis_angle: Tensor) -> Tensor:
    """Axis-angle (..., 3) -> quaternion (..., 4) (w, x, y, z)."""
    aa = axis_angle.reshape(-1, 3)
    angle = aa.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    half = angle * 0.5
    axis = aa / angle
    w = torch.cos(half)
    xyz = axis * torch.sin(half)
    quat = torch.cat((w, xyz), dim=-1)
    return _normalize(quat, dim=-1).reshape(*axis_angle.shape[:-1], 4)


def geodesic_distance(r1: Tensor, r2: Tensor) -> Tensor:
    """Geodesic angle (radians) between rotation matrices (..., 3, 3). Returns (...,)."""
    r_rel = torch.matmul(r1.transpose(-1, -2), r2)
    skew = r_rel - r_rel.transpose(-1, -2)
    vee = torch.stack((skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]), dim=-1)
    sin_term = 0.5 * torch.sqrt((vee * vee).sum(dim=-1) + _EPS * _EPS)
    cos_term = (r_rel[..., 0, 0] + r_rel[..., 1, 1] + r_rel[..., 2, 2] - 1.0) * 0.5
    return torch.atan2(sin_term, cos_term)


def poses_to_axis_angle(poses: Tensor, rot_repr: str) -> Tensor:
    """Convert pose blocks (..., K, 3) or (..., K, 6) to axis-angle (..., K*3)."""
    if rot_repr == "axis_angle":
        return poses.reshape(*poses.shape[:-2], -1)
    if rot_repr == "6d":
        k = poses.shape[-2]
        mats = rotation_6d_to_matrix(poses.reshape(-1, 6)).reshape(*poses.shape[:-2], k, 3, 3)
        aa = matrix_to_axis_angle(mats)
        return aa.reshape(*poses.shape[:-2], k * 3)
    raise ValueError(f"rot_repr must be 'axis_angle' or '6d', got {rot_repr!r}")
