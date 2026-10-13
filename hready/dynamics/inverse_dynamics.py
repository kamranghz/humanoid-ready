"""Joint torques of an SMPL-X body from motion alone: recursive Newton-Euler over the kinematic tree.

NVIDIA Newton is not used here (rule 0 default); this is the documented fallback, see ``docs/pivot_log.md``.

Model
-----
- The 22 body joints of SMPL-X. Segment ``j`` is attached to joint ``j`` and runs from joint ``j`` to the
  mean of its child joints (leaf segments are extended, see ``_LEAF_EXTENSION``). Its centre of mass (COM) is
  the segment midpoint, except the pelvis (COM at the pelvis joint).
- Segment masses come from anthropometric fractions (``SEGMENT_FRACTIONS``). The fractions given by the task
  sum to 94.8 %, so they are rescaled to ``total_mass`` (70 kg by default); the raw sum is reported by
  :func:`mass_table_sum`.
- Each segment is an isotropic rigid body with moment of inertia ``m L^2 / 12`` (thin-rod value about a
  perpendicular axis, ``L`` = segment length). The gyroscopic term then vanishes.
- Linear acceleration of each COM and angular velocity / acceleration of each segment frame come from central
  finite differences of the supplied joint positions and global joint rotations.
- Ground reaction: the whole-body Newton-Euler residual (force and moment about the whole-body COM) is the
  ground wrench. It is split over the labelled contact points by the minimum-norm solution of the 6 wrench
  equations, weighted to prefer vertical force (no friction cone, no non-negativity constraint). With no
  contact the ground force is zero.

Torque at joint ``j`` is the moment, about joint ``j``, that the parent exerts on the subtree of ``j``::

    tau_j = sum_{i in subtree(j)} [ (c_i - p_j) x m_i (a_i - g) + I_i alpha_i ] - sum_{k in contacts(j)} (x_k - p_j) x f_k

World-frame vectors, N m. The root joint (0) has no torque (free base) and edge frames that need
second differences of the angular velocity are NaN.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from hready.body.rotations import axis_angle_to_matrix, matrix_to_axis_angle

GRAVITY = 9.81
NUM_JOINTS = 22
# SMPL-X body joint order.
JOINT_NAMES = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle",
    "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
)
# Foot-channel contact points, in the order of the cache labels (L heel, L toe, R heel, R toe).
CONTACT_JOINTS = (7, 10, 8, 11)

# Fraction of body mass per segment (attached to the joint at its proximal end). The trunk's 43 % is split
# over pelvis / spine1 / spine2 / spine3; joints not listed (neck, collars, toes) carry no mass.
SEGMENT_FRACTIONS = {
    0: 0.13, 3: 0.10, 6: 0.10, 9: 0.10,  # trunk 0.43
    15: 0.08,  # head
    16: 0.033, 17: 0.033,  # upper arms
    18: 0.019, 19: 0.019,  # forearms
    20: 0.006, 21: 0.006,  # hands
    1: 0.10, 2: 0.10,  # thighs
    4: 0.0465, 5: 0.0465,  # shanks
    7: 0.0145, 8: 0.0145,  # feet
}
# Leaf segments have no child joint: the distal end is proximal + factor * (proximal - parent).
_LEAF_EXTENSION = {15: 2.0, 20: 0.7, 21: 0.7, 10: 0.0, 11: 0.0}
_PELVIS_LENGTH_JOINTS = (1, 2)  # pelvis inertia length = hip width
# Ground-force split: minimise |f_z|^2 + P^2 |f_xy|^2, so horizontal force is used only where the wrench needs it.
_HORIZONTAL_PENALTY = 10.0
_MIN_SEGMENT_LENGTH = 1e-6  # m


def mass_table_sum() -> float:
    """Sum of the given segment fractions (0.948: the table does not cover 100 % of the body)."""
    return float(sum(SEGMENT_FRACTIONS.values()))


def segment_masses(total_mass: float = 70.0) -> np.ndarray:
    """Per-joint segment mass (22,) in kg, fractions rescaled so the masses sum to ``total_mass``."""
    m = np.zeros(NUM_JOINTS)
    for j, frac in SEGMENT_FRACTIONS.items():
        m[j] = frac
    return m * (total_mass / m.sum())


def global_rotations(root_aa: torch.Tensor, body_aa: torch.Tensor, parents: torch.Tensor) -> torch.Tensor:
    """Global rotation matrices (..., 22, 3, 3) from root (..., 3) and body (..., 21, 3) axis-angle."""
    aa = torch.cat([root_aa.unsqueeze(-2), body_aa], dim=-2).to(torch.float64)
    local = axis_angle_to_matrix(aa.reshape(-1, 3)).reshape(*aa.shape[:-1], 3, 3)  # the helper takes (B, 3)
    out = [local[..., 0, :, :]]
    for j in range(1, NUM_JOINTS):
        out.append(out[int(parents[j])] @ local[..., j, :, :])
    return torch.stack(out, dim=-3)


@dataclass
class InverseDynamicsResult:
    torque: np.ndarray  # (T, 22, 3) N m, world frame; NaN at the root and at unusable edge frames
    ground_force: np.ndarray  # (T, 4, 3) N per contact point (L heel, L toe, R heel, R toe)
    masses: np.ndarray  # (22,) kg
    com: np.ndarray  # (T, 3) whole-body COM (NaN at edge frames)
    wrench_residual: np.ndarray  # (T,) |A f - b| of the ground-wrench fit (NaN at edge frames)


def _children(parents: np.ndarray) -> list[list[int]]:
    kids: list[list[int]] = [[] for _ in range(NUM_JOINTS)]
    for j in range(1, NUM_JOINTS):
        kids[int(parents[j])].append(j)
    return kids


def _subtree(kids: list[list[int]], j: int) -> list[int]:
    out = [j]
    for c in kids[j]:
        out += _subtree(kids, c)
    return out


def _skew(r: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]])


def inverse_dynamics(
    joints: np.ndarray,
    rot_global: np.ndarray,
    contact: np.ndarray,
    fps: float,
    parents: np.ndarray,
    total_mass: float = 70.0,
) -> InverseDynamicsResult:
    """Joint torques for one window.

    ``joints`` (T, >=22, 3) world positions (Z up), ``rot_global`` (T, 22, 3, 3), ``contact`` (T, 4) bool-like
    for the contact points ``CONTACT_JOINTS``, ``parents`` (22,). Needs ``T >= 5``.
    """
    pos = np.asarray(joints, dtype=np.float64)[:, :NUM_JOINTS]
    rot = np.asarray(rot_global, dtype=np.float64)
    contact = np.asarray(contact) > 0.5
    parents = np.asarray(parents, dtype=np.int64)
    t_len = pos.shape[0]
    if t_len < 5:
        raise ValueError("inverse_dynamics needs at least 5 frames")
    if not (np.isfinite(pos).all() and np.isfinite(rot).all()):
        raise ValueError("inverse_dynamics: non-finite joint positions or rotations in the input")
    dt = 1.0 / float(fps)
    mass = segment_masses(total_mass)
    kids = _children(parents)

    # segment geometry
    com = np.zeros((t_len, NUM_JOINTS, 3))
    length = np.zeros((t_len, NUM_JOINTS))
    for j in np.flatnonzero(mass > 0):
        if kids[j]:
            distal = pos[:, kids[j]].mean(axis=1)
        else:
            distal = pos[:, j] + _LEAF_EXTENSION[j] * (pos[:, j] - pos[:, parents[j]])
        com[:, j] = 0.5 * (pos[:, j] + distal)
        length[:, j] = np.linalg.norm(distal - pos[:, j], axis=-1)
    com[:, 0] = pos[:, 0]
    length[:, 0] = np.linalg.norm(pos[:, _PELVIS_LENGTH_JOINTS[0]] - pos[:, _PELVIS_LENGTH_JOINTS[1]], axis=-1)
    length = np.maximum(length, _MIN_SEGMENT_LENGTH)  # zero-length segments (coincident joints) keep a tiny finite length
    inertia = mass[None, :] * length**2 / 12.0  # (T, 22)

    # kinematics by central differences
    acc = np.full((t_len, NUM_JOINTS, 3), np.nan)
    acc[1:-1] = (com[2:] - 2.0 * com[1:-1] + com[:-2]) / dt**2
    rt = torch.as_tensor(rot)
    omega = np.full((t_len, NUM_JOINTS, 3), np.nan)
    rel = rt[2:] @ rt[:-2].transpose(-1, -2)  # R_{t+1} R_{t-1}^T
    omega[1:-1] = matrix_to_axis_angle(rel).numpy() / (2.0 * dt)
    alpha = np.full((t_len, NUM_JOINTS, 3), np.nan)
    alpha[2:-2] = (omega[3:-1] - omega[1:-3]) / (2.0 * dt)

    g_vec = np.array([0.0, 0.0, -GRAVITY])
    f_seg = mass[None, :, None] * (acc - g_vec)  # m (a - g), (T, 22, 3)
    ia = inertia[:, :, None] * alpha  # I alpha

    # whole-body wrench and its split over the contact points
    com_total = (mass[None, :, None] * com).sum(axis=1) / mass.sum()
    total_force = f_seg.sum(axis=1)
    total_moment = (np.cross(com - com_total[:, None], mass[None, :, None] * acc) + ia).sum(axis=1)
    ground = np.zeros((t_len, 4, 3))
    resid = np.full(t_len, np.nan)
    for t in range(2, t_len - 2):
        idx = np.flatnonzero(contact[t])
        if idx.size == 0:
            resid[t] = float(np.linalg.norm(np.concatenate([total_force[t], total_moment[t]])))
            continue
        a_mat = np.zeros((6, 3 * idx.size))
        for n, c in enumerate(idx):
            a_mat[:3, 3 * n : 3 * n + 3] = np.eye(3)
            a_mat[3:, 3 * n : 3 * n + 3] = _skew(pos[t, CONTACT_JOINTS[c]] - com_total[t])
        b = np.concatenate([total_force[t], total_moment[t]])
        scale = np.tile([_HORIZONTAL_PENALTY, _HORIZONTAL_PENALTY, 1.0], idx.size)
        f = np.linalg.lstsq(a_mat / scale, b, rcond=None)[0] / scale
        resid[t] = float(np.linalg.norm(a_mat @ f - b))
        ground[t, idx] = f.reshape(-1, 3)

    # joint torques: subtree sums
    torque = np.full((t_len, NUM_JOINTS, 3), np.nan)
    contact_pos = pos[:, list(CONTACT_JOINTS)]
    for j in range(1, NUM_JOINTS):
        sub = _subtree(kids, j)
        tau = np.zeros((t_len, 3))
        for i in sub:
            tau += np.cross(com[:, i] - pos[:, j], f_seg[:, i]) + ia[:, i]
        for c, cj in enumerate(CONTACT_JOINTS):
            if cj in sub:
                tau -= np.cross(contact_pos[:, c] - pos[:, j], ground[:, c])
        torque[:, j] = tau
    torque[:2] = np.nan
    torque[-2:] = np.nan
    com_out = com_total.copy()
    com_out[:1] = np.nan
    com_out[-1:] = np.nan
    return InverseDynamicsResult(torque=torque, ground_force=ground, masses=mass, com=com_out, wrench_residual=resid)
