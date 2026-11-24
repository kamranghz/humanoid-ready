"""Tracking metrics of one simulated clip against its reference (numpy only; shared by the open-loop baseline and the policy runs).

Positions are the 22 joint positions of the SMPL-X-skeleton humanoid in metres, sampled at the control rate; the root is joint 0.
Termination follows the published recipe (anchor or end-effector height error above 0.25 m, anchor orientation error above 0.8 rad);
success follows PHC (mean joint distance below 0.5 m in every frame). The anchor is the pelvis.
"""

from __future__ import annotations

import numpy as np

Z_TOL_M = 0.25
ANGLE_TOL_RAD = 0.8
SUCC_TOL_M = 0.5


def quat_angle(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Angle in radians between unit quaternions (..., 4); sign-insensitive."""
    d = np.abs(np.sum(q1 * q2, axis=-1)).clip(0.0, 1.0)
    return 2.0 * np.arccos(d)


def _difference_error(sim: np.ndarray, ref: np.ndarray, dt: float, order: int) -> float:
    if len(sim) <= order:
        return float("nan")
    s, r = np.diff(sim, n=order, axis=0) / dt**order, np.diff(ref, n=order, axis=0) / dt**order
    return float(np.linalg.norm(s - r, axis=-1).mean())


def evaluate_clip(sim_joints: np.ndarray, ref_joints: np.ndarray, sim_quat: np.ndarray, ref_quat: np.ndarray, dt: float,
                  end_effectors: list[int]) -> dict:
    """sim_joints, ref_joints (T,22,3); sim_quat, ref_quat (T,4) xyzw root orientation; dt the control step in seconds."""
    t_len = len(ref_joints)
    finite = np.isfinite(sim_joints).all(axis=(1, 2)) & np.isfinite(sim_quat).all(axis=1)
    sim_j = np.nan_to_num(sim_joints, nan=1e3, posinf=1e3, neginf=-1e3)
    err = np.linalg.norm(sim_j - ref_joints, axis=-1)  # (T,22)
    mean_err = err.mean(axis=1)
    rel_err = np.linalg.norm((sim_j - sim_j[:, :1]) - (ref_joints - ref_joints[:, :1]), axis=-1).mean(axis=1)
    anchor_z = np.abs(sim_j[:, 0, 2] - ref_joints[:, 0, 2])
    anchor_angle = np.where(finite, quat_angle(np.nan_to_num(sim_quat), ref_quat), np.inf)
    ee_z = np.abs(sim_j[:, end_effectors, 2] - ref_joints[:, end_effectors, 2]).max(axis=1)
    reasons = {"anchor_height": anchor_z > Z_TOL_M, "anchor_orientation": anchor_angle > ANGLE_TOL_RAD,
               "end_effector_height": ee_z > Z_TOL_M, "not_finite": ~finite}
    any_term = np.any(np.stack(list(reasons.values())), axis=0)
    term_step = int(np.argmax(any_term)) if any_term.any() else None
    until = slice(0, term_step if term_step is not None else t_len)
    return {
        "steps": t_len, "duration_s": (t_len - 1) * dt,
        "completed": term_step is None, "termination_step": term_step,
        "termination_time_s": None if term_step is None else term_step * dt,
        "termination_reasons": None if term_step is None else [k for k, v in reasons.items() if v[term_step]],
        "success_phc": bool(finite.all() and mean_err.max() < SUCC_TOL_M),
        "mpjpe_global_mm": float(1000 * mean_err.mean()), "mpjpe_root_relative_mm": float(1000 * rel_err.mean()),
        "mpjpe_global_until_termination_mm": float(1000 * mean_err[until].mean()) if mean_err[until].size else float("nan"),
        "max_mean_joint_error_m": float(mean_err.max()),
        "velocity_error_m_s": _difference_error(sim_j, ref_joints, dt, 1),
        "acceleration_error_m_s2": _difference_error(sim_j, ref_joints, dt, 2),
        "final_pelvis_height_error_m": float(anchor_z[-1]),
    }
