"""Robot rollout metrics from Isaac Newton run .npz (pure NumPy, item 6C)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

DEFINITIONS: dict[str, str] = {
    "time_to_fall_s": (
        "Free mode only: sim time at fall_step (base_z < 0.4 m or |roll|/|pitch| > 45 deg); "
        "null if no fall or kin_root mode. Not comparable across solver settings "
        "(e.g. pass3 jump free 0.55 s at 8 substeps + scratch motion vs 6C 0.97 s at "
        "1 substep + retargeted motion); see sim_settings in metrics JSON."
    ),
    "joint_rmse_rad": (
        "RMS over time and joints of |q - q_ref| (GMR 29-DoF order). "
        "Free mode: only steps t <= fall_step (inclusive); kin_root: full clip."
    ),
    "root_pos_err_cm_mean": (
        "Mean L2 distance (cm) between sim root_pos and root_pos_ref. "
        "Free mode: truncated at fall_step; kin_root: full clip."
    ),
    "root_pos_err_cm_p95": "95th percentile of per-step root position error (cm); same truncation as mean.",
    "root_rot_err_deg_mean": (
        "Mean geodesic angle (deg) between sim and ref root quaternions (xyzw); same truncation as root pos."
    ),
    "root_rot_err_deg_p95": "95th percentile of root orientation error (deg).",
    "pd_demand_exceeds_limit_{legs,waist,arms}_step_fraction": (
        "Per actuator group (12 legs / 3 waist / 14 arms GMR indices): fraction of steps where "
        "ANY joint in the group has |tau_pd|/effort_limit >= 0.95. tau_pd is the logged "
        "analytic PD demand K*(q*-q)-D*qdot (same as Isaac computed_torque before clip)."
    ),
    "pd_demand_ratio_{legs,waist,arms}_step_max_p95": (
        "95th percentile of per-step max_j |tau_pd|/limit within the group."
    ),
    "actuator_torque_at_limit_{legs,waist,arms}_step_fraction": (
        "Per group: fraction of steps where ANY joint has |tau_applied|/effort_limit >= 0.95 "
        "(Isaac applied_torque after actuator clipping). Null if run .npz has no tau_applied_gmr."
    ),
    "actuator_clamped_pd_demand_{legs,waist,arms}_step_fraction": (
        "Per group: fraction of steps where ANY joint has |tau_computed| > |tau_applied| + 1e-3 N·m "
        "(PD demand was clipped by the actuator model). Null without tau_applied_gmr."
    ),
    "deprecated_pooled_all29_pd_demand_step_fraction": (
        "DEPRECATED headline: any-of-29-joints PD demand step rate (was 0.380 on ACCAD jump kin_root); "
        "dominated by unclipped waist ImplicitActuator PD vs limit — use per-group pd_demand_* instead."
    ),
    "torque_saturation_step_fraction": "Alias of deprecated_pooled_all29_pd_demand_step_fraction.",
    "torque_saturation_joint_step_fraction": (
        "Fraction of (step, joint) pairs over 29 GMR joints with |tau_pd|/effort_limit >= 0.95."
    ),
    "torque_saturation_fraction": "Alias of torque_saturation_joint_step_fraction.",
    "foot_slip_rms_m_s": (
        "EXPERIMENTAL, NOT INTERPRETED: contact-aware slip on simulated ankle positions at sim dt; "
        "reference vs sim contact sample counts differ widely (e.g. walk 106 vs 3032). "
        "Do not use as evidence."
    ),
    "foot_slip_contact_frames": (
        "EXPERIMENTAL: count of (step, foot) velocity samples in foot_slip_rms_m_s; not interpreted."
    ),
    "sim_settings": (
        "Physics/control metadata: dt, num_substeps, njmax, nconmax, initial_root_velocity_convention."
    ),
    "assist_wrench": "Not reported — not available on this Newton path (see pivot_log).",
}

_TORQUE_THRESHOLD = 0.95
_CLAMP_EPS_NM = 1e-3

# GMR 29-DoF order (matches isaaclab_newton.GMR_JOINT_NAMES).
JOINT_GROUP_INDICES: dict[str, np.ndarray] = {
    "legs": np.arange(0, 12, dtype=int),
    "waist": np.arange(12, 15, dtype=int),
    "arms": np.arange(15, 29, dtype=int),
}


def _quat_geodesic_deg(qa: np.ndarray, qb: np.ndarray) -> float:
    qa = qa / np.linalg.norm(qa)
    qb = qb / np.linalg.norm(qb)
    dot = float(np.clip(np.abs(np.dot(qa, qb)), -1.0, 1.0))
    return float(2.0 * np.degrees(np.arccos(dot)))


def _truncate_free(steps: int, fall_step: int, mode: str) -> int:
    if mode != "free" or fall_step < 0:
        return steps
    return min(steps, fall_step + 1)


def foot_slip_rms_from_ankles(ankle: np.ndarray, time: np.ndarray) -> tuple[float, int]:
    """Shared slip metric: ankle (T, 2, 3), time (T,) -> (rms_m_s, n_samples)."""
    slip_speeds: list[float] = []
    n_contact = 0
    if ankle.shape[0] > 1 and ankle.ndim == 3:
        for foot in range(min(2, ankle.shape[1])):
            z = ankle[:, foot, 2]
            z_min = float(z.min())
            contact = z <= z_min + 0.03
            for i in range(1, ankle.shape[0]):
                if contact[i] and contact[i - 1]:
                    dt_i = max(float(time[i] - time[i - 1]), 1e-9)
                    vxy = ankle[i, foot, :2] - ankle[i - 1, foot, :2]
                    slip_speeds.append(float(np.linalg.norm(vxy) / dt_i))
                    n_contact += 1
    slip_rms = float(np.sqrt(np.mean(np.square(slip_speeds)))) if slip_speeds else 0.0
    return slip_rms, n_contact


def step_saturation_fraction(
    tau: np.ndarray, lim: np.ndarray, joint_indices: np.ndarray | None = None
) -> float:
    """Fraction of steps where ANY joint in subset has |tau|/limit >= threshold."""
    if joint_indices is not None:
        tau = tau[:, joint_indices]
        lim = lim[joint_indices]
    ratio = np.abs(tau) / np.maximum(lim[None, :], 1e-6)
    if ratio.size == 0:
        return 0.0
    return float(np.mean(ratio.max(axis=1) >= _TORQUE_THRESHOLD))


def _group_step_max_p95(tau: np.ndarray, lim: np.ndarray, idx: np.ndarray) -> float:
    t_sub = tau[:, idx]
    l_sub = lim[idx]
    ratio = np.abs(t_sub) / np.maximum(l_sub[None, :], 1e-6)
    step_max = ratio.max(axis=1)
    return float(np.percentile(step_max, 95)) if step_max.size else 0.0


def torque_metrics_by_group(
    tau_pd: np.ndarray,
    lim: np.ndarray,
    tau_applied: np.ndarray | None,
    tau_computed: np.ndarray | None,
) -> dict[str, Any]:
    """Per-group PD demand vs actuator-applied torque metrics."""
    out: dict[str, Any] = {}
    pooled_step = step_saturation_fraction(tau_pd, lim, None)
    out["deprecated_pooled_all29_pd_demand_step_fraction"] = pooled_step
    out["torque_saturation_step_fraction"] = pooled_step
    sat_mask = np.abs(tau_pd) / np.maximum(lim[None, :], 1e-6) >= _TORQUE_THRESHOLD
    out["torque_saturation_joint_step_fraction"] = float(np.mean(sat_mask))
    out["torque_saturation_fraction"] = out["torque_saturation_joint_step_fraction"]

    has_applied = (
        tau_applied is not None
        and tau_computed is not None
        and tau_applied.shape == tau_pd.shape
    )
    for name, idx in JOINT_GROUP_INDICES.items():
        out[f"pd_demand_exceeds_limit_{name}_step_fraction"] = step_saturation_fraction(
            tau_pd, lim, idx
        )
        out[f"pd_demand_ratio_{name}_step_max_p95"] = _group_step_max_p95(tau_pd, lim, idx)
        if has_applied:
            out[f"actuator_torque_at_limit_{name}_step_fraction"] = step_saturation_fraction(
                tau_applied, lim, idx
            )
            comp = np.abs(tau_computed[:, idx])
            appl = np.abs(tau_applied[:, idx])
            clamped = comp > (appl + _CLAMP_EPS_NM)
            out[f"actuator_clamped_pd_demand_{name}_step_fraction"] = float(
                np.mean(clamped.any(axis=1))
            )
        else:
            out[f"actuator_torque_at_limit_{name}_step_fraction"] = None
            out[f"actuator_clamped_pd_demand_{name}_step_fraction"] = None
    return out


def compute_run_metrics(run_npz: Path | str) -> dict[str, Any]:
    """Load a ``isaaclab_newton.py`` run archive and return metric numbers + definitions."""
    path = Path(run_npz)
    data = np.load(path, allow_pickle=True)
    t = np.asarray(data["time"], dtype=np.float64)
    q = np.asarray(data["q"], dtype=np.float64)
    q_ref = np.asarray(data["q_ref"], dtype=np.float64)
    root_pos = np.asarray(data["root_pos"], dtype=np.float64)
    root_ref = np.asarray(data["root_pos_ref"], dtype=np.float64)
    quat = np.asarray(data["root_quat_xyzw"], dtype=np.float64)
    quat_ref = np.asarray(data["root_quat_xyzw_ref"], dtype=np.float64)
    tau = np.asarray(data["tau"], dtype=np.float64)
    tau_computed = (
        np.asarray(data["tau_computed_gmr"], dtype=np.float64)
        if "tau_computed_gmr" in data
        else None
    )
    tau_applied = (
        np.asarray(data["tau_applied_gmr"], dtype=np.float64)
        if "tau_applied_gmr" in data
        else None
    )
    lim = np.asarray(data["effort_limit_gmr"], dtype=np.float64)
    ankle = np.asarray(data["ankle_pos"], dtype=np.float64)
    fall_step = int(data["fall_step"])
    dt = float(data["dt"])
    mode = str(data["mode"].item() if data["mode"].ndim == 0 else data["mode"][0])

    n_use = _truncate_free(q.shape[0], fall_step, mode)
    q, q_ref = q[:n_use], q_ref[:n_use]
    root_pos, root_ref = root_pos[:n_use], root_ref[:n_use]
    quat, quat_ref = quat[:n_use], quat_ref[:n_use]
    tau = tau[:n_use]
    if tau_computed is not None:
        tau_computed = tau_computed[:n_use]
    if tau_applied is not None:
        tau_applied = tau_applied[:n_use]
    ankle = ankle[:n_use]
    t = t[:n_use]

    joint_rmse = float(np.sqrt(np.mean((q - q_ref) ** 2)))
    waist_idx = JOINT_GROUP_INDICES["waist"]
    waist_rmse_rad = float(np.sqrt(np.mean((q[:, waist_idx] - q_ref[:, waist_idx]) ** 2)))
    waist_max_abs_err_rad = float(np.max(np.abs(q[:, waist_idx] - q_ref[:, waist_idx])))
    pos_err_m = np.linalg.norm(root_pos - root_ref, axis=1)
    pos_err_cm = pos_err_m * 100.0
    rot_err = np.array([_quat_geodesic_deg(quat[i], quat_ref[i]) for i in range(quat.shape[0])])

    tau_pd = tau_computed if tau_computed is not None else tau
    tq = torque_metrics_by_group(tau_pd, lim, tau_applied, tau_computed)

    ttf: float | None = None
    if mode == "free" and fall_step >= 0:
        ttf = float(fall_step * dt)

    slip_rms, n_contact = foot_slip_rms_from_ankles(ankle, t)

    out: dict[str, Any] = {
        "run_npz": str(path.resolve()),
        "mode": mode,
        "sim_steps": int(q.shape[0]),
        "sim_steps_recorded": int(data["q"].shape[0]),
        "sim_duration_s": float(t[-1]) if t.size else 0.0,
        "dt": dt,
        "time_to_fall_s": ttf,
        "joint_rmse_rad": joint_rmse,
        "waist_tracking_rmse_rad": waist_rmse_rad,
        "waist_tracking_max_abs_err_rad": waist_max_abs_err_rad,
        "root_pos_err_cm_mean": float(np.mean(pos_err_cm)),
        "root_pos_err_cm_p95": float(np.percentile(pos_err_cm, 95)),
        "root_rot_err_deg_mean": float(np.mean(rot_err)),
        "root_rot_err_deg_p95": float(np.percentile(rot_err, 95)),
        **tq,
        "foot_slip_rms_m_s": slip_rms,
        "foot_slip_contact_frames": n_contact,
        "definitions": DEFINITIONS,
        "assist_wrench": None,
        "assist_wrench_note": DEFINITIONS["assist_wrench"],
    }
    if mode == "free" and fall_step >= 0:
        out["free_mode_metrics_truncated_at_fall"] = True
        out["fall_step_index"] = fall_step
    if "versions" in data:
        out["versions"] = json.loads(str(data["versions"]))
    if "mass_kg" in data:
        out["mass_kg"] = float(data["mass_kg"])
    if "sim_settings" in data:
        out["sim_settings"] = json.loads(str(data["sim_settings"]))
    else:
        out["sim_settings"] = {
            "dt_s": dt,
            "control_rate_hz": 1.0 / dt if dt > 0 else 0.0,
            "physics_num_substeps": 1,
            "njmax": 250,
            "nconmax": 80,
            "initial_root_velocity_convention": (
                "6C default: zero at spawn (inferred; re-run tracker for sim_settings blob)"
            ),
            "fall_detector": "z<0.4 m or tilt>45 deg",
        }
    if vers := out.get("versions"):
        out["sim_settings"].setdefault("njmax", vers.get("njmax", 250))
        out["sim_settings"].setdefault("nconmax", vers.get("nconmax", 80))
    return out


def write_metrics_json(run_npz: Path | str, out_json: Path | str) -> dict[str, Any]:
    metrics = compute_run_metrics(run_npz)
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
