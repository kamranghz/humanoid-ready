"""Biomechanical losses (bone lengths, joint ROM from body_pose rotations).

Conventions match ``physics.py``: Z-up world frame, meters, seconds.
``body_pose``: ``(B, T, 21, 3)`` axis-angle or ``(B, T, 21, 3, 3)`` rotation matrices
(SMPL-X body joints, excluding global root).

``joint_rom`` flexion direction (locked_head, SMPL-X local Y-up, body +Z forward):
left/right knee — local X, sign +1: flexion moves ankle backward (−Z).
left elbow — local Y, sign −1: flexion moves wrist forward (+Z).
right elbow — local Y, sign +1: flexion moves wrist forward (+Z).
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

_EPS: float = 1e-8

# Conservative anatomical hinge flexion limits (degrees).
# Calibrate on AMASS percentiles after checklist item 5.
#
# | Joint        | body_pose idx | flexion range (deg) |
# |--------------|---------------|---------------------|
# | left_knee    | 3             | [0, 150]            |
# | right_knee   | 4             | [0, 150]            |
# | left_elbow   | 17            | [0, 150]            |
# | right_elbow  | 18            | [0, 150]            |
_ROM_LIMITS_DEG = {
    "left_knee": (0.0, 150.0),
    "right_knee": (0.0, 150.0),
    "left_elbow": (0.0, 150.0),
    "right_elbow": (0.0, 150.0),
}

# body_pose joint index (0..20) → hinge axis (0=x,1=y,2=z) and sign for positive flexion.
_ROM_JOINTS = {
    "left_knee": {"idx": 3, "axis": 0, "sign": 1.0},
    "right_knee": {"idx": 4, "axis": 0, "sign": 1.0},
    "left_elbow": {"idx": 17, "axis": 1, "sign": -1.0},
    "right_elbow": {"idx": 18, "axis": 1, "sign": 1.0},
}

RotRepr = Literal["axis_angle", "matrix"]


def _safe_norm(x: Tensor, dim: int = -1) -> Tensor:
    return torch.sqrt((x * x).sum(dim=dim) + _EPS * _EPS)


def _signed_hinge_flexion(
    body_pose: Tensor,
    rot_repr: RotRepr,
    joint_idx: int,
    hinge_axis: int,
    flex_sign: float,
) -> Tensor:
    """Signed flexion (rad): 0 = straight, positive = flexion, negative = hyperextension."""
    if rot_repr == "axis_angle":
        return flex_sign * body_pose[..., joint_idx, hinge_axis]
    R = body_pose[..., joint_idx, :, :]
    if hinge_axis == 0:
        flex = torch.atan2(R[..., 2, 1], R[..., 2, 2])
    elif hinge_axis == 1:
        flex = torch.atan2(-R[..., 2, 0], R[..., 2, 2])
    else:
        flex = torch.atan2(R[..., 1, 0], R[..., 0, 0])
    return flex * flex_sign


def _hinge_rom_penalty_deg(flex: Tensor, min_deg: float, max_deg: float) -> Tensor:
    min_r = min_deg * (torch.pi / 180.0)
    max_r = max_deg * (torch.pi / 180.0)
    return torch.relu(min_r - flex).square() + torch.relu(flex - max_r).square()


def bone_length_consistency(joints: Tensor, parents: Tensor) -> Tensor:
    """Penalize variance of each bone length over time (m²)."""
    parents = parents.to(device=joints.device)
    child_idx = torch.arange(joints.shape[-2], device=joints.device)
    mask = parents >= 0
    c = child_idx[mask]
    p = parents[mask]
    bone = joints[..., c, :] - joints[..., p, :]
    length = _safe_norm(bone, dim=-1)
    mean = length.mean(dim=1, keepdim=True)
    return (length - mean).square().mean()


def joint_rom(body_pose: Tensor, *, rot_repr: RotRepr = "axis_angle") -> Tensor:
    """Hinge ROM on knees and elbows from ``body_pose`` rotations (not positions).

    The ``axis_angle`` branch uses the raw axis component (``sign * angle``) per hinge;
    the ``matrix`` branch recovers the hinge angle with ``atan2`` on rotation-matrix
    entries. They agree for pure single-axis hinge rotations only. Use
    ``rot_repr='matrix'`` in training (e.g. after 6D → matrix).
    """
    terms = []
    for name, spec in _ROM_JOINTS.items():
        lo, hi = _ROM_LIMITS_DEG[name]
        flex = _signed_hinge_flexion(
            body_pose,
            rot_repr,
            spec["idx"],
            spec["axis"],
            spec["sign"],
        )
        terms.append(_hinge_rom_penalty_deg(flex, lo, hi).mean())
    return torch.stack(terms).sum() / len(terms)
