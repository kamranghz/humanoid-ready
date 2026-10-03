"""WSL-only Isaac Lab + Newton G1 tracker (item 6C). Run as ``__main__``; do not import from hready."""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any

if __name__ != "__main__":
    raise ImportError(
        "hready.robot.isaaclab_newton is a WSL Isaac Lab entrypoint; execute with Isaac Lab's Python."
    )

print(sys.executable, flush=True)

from isaaclab.app import AppLauncher

_parser = argparse.ArgumentParser(description="G1 Newton MJWarp motion tracking (free or kin_root).")
_parser.add_argument("--motion", type=str, required=True, help="Retarget motion .npz from hready.robot.retarget")
_parser.add_argument("--mode", choices=["free", "kin_root"], required=True)
_parser.add_argument("--out", type=str, required=True, help="Output run .npz path")
AppLauncher.add_app_launcher_args(_parser)
_args_cli = _parser.parse_args()
_app_launcher = AppLauncher(_args_cli)
simulation_app = _app_launcher.app

import numpy as np
import torch
import isaaclab
import isaaclab.sim as sim_utils
import isaaclab_newton
import newton
import warp as wp
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab_assets import G1_29DOF_CFG
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

SIM_DT = 0.005
NJMAX = 250
NCONMAX = 80
FALL_Z = 0.4
FALL_TILT_DEG = 45.0
STAND_DURATION_S = 10.0

GMR_TO_ISAAC_IDX = list(range(22)) + list(range(29, 36))
GMR_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


def _versions() -> dict[str, Any]:
    return {
        "isaaclab": getattr(isaaclab, "__version__", "unknown"),
        "isaaclab_newton": isaaclab_newton.__version__,
        "newton": getattr(newton, "__version__", "unknown"),
        "warp": wp.__version__,
        "backend": "newton_mjwarp",
        "dt": SIM_DT,
        "njmax": NJMAX,
        "nconmax": NCONMAX,
    }


def _newton_sim_cfg() -> SimulationCfg:
    return SimulationCfg(
        dt=SIM_DT,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=NJMAX,
                nconmax=NCONMAX,
                cone="pyramidal",
                impratio=1.0,
                integrator="implicitfast",
                iterations=100,
                use_mujoco_contacts=False,
            ),
            num_substeps=1,
        ),
    )


def _make_robot_cfg() -> Any:
    cfg = G1_29DOF_CFG.copy()
    cfg.prim_path = "/World/Robot"
    cfg.spawn.activate_contact_sensors = True
    return cfg


def _quat_rpy_deg(q: torch.Tensor) -> tuple[float, float, float]:
    x, y, z, w = [float(q[i].item()) for i in range(4)]
    sinr = 2 * (w * x + y * z)
    cosr = 1 - 2 * (x * x + y * y)
    roll = math.degrees(math.atan2(sinr, cosr))
    sinp = 2 * (w * y - z * x)
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, sinp))))
    siny = 2 * (w * z + x * y)
    cosy = 1 - 2 * (y * y + z * z)
    yaw = math.degrees(math.atan2(siny, cosy))
    return roll, pitch, yaw


def _fall_detect(robot: Articulation) -> bool:
    z = float(robot.data.root_pos_w.torch[0, 2].item())
    r, p, _ = _quat_rpy_deg(robot.data.root_quat_w.torch[0])
    return z < FALL_Z or max(abs(r), abs(p)) > FALL_TILT_DEG


def _implicit_pd_torque(
    robot: Articulation, q_target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q = robot.data.joint_pos.torch[0]
    qd = robot.data.joint_vel.torch[0]
    if q_target.ndim == 2:
        q_target = q_target[0]
    tau = torch.zeros_like(q)
    stiff = torch.zeros_like(q)
    damp = torch.zeros_like(q)
    for act in robot.actuators.values():
        idx = act._joint_indices
        stiff[idx] = act.stiffness[0]
        damp[idx] = act.damping[0]
        tau[idx] = stiff[idx] * (q_target[idx] - q[idx]) - damp[idx] * qd[idx]
    lim = torch.full_like(q, 300.0)
    for act in robot.actuators.values():
        idx = act._joint_indices
        lim[idx] = act.effort_limit[0].abs().clamp(min=1e-3, max=1e4)
    return tau, lim, stiff, damp


def _gmr_to_full_q(default_q: torch.Tensor, gmr_row: np.ndarray) -> torch.Tensor:
    q = default_q.clone()
    for gi, ii in enumerate(GMR_TO_ISAAC_IDX):
        q[0, ii] = float(gmr_row[gi])
    return q


def _full_to_gmr(q_isaac: torch.Tensor) -> np.ndarray:
    row = np.zeros(29, dtype=np.float64)
    for gi, ii in enumerate(GMR_TO_ISAAC_IDX):
        row[gi] = float(q_isaac[ii].item())
    return row


def _load_motion(path: str) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    meta_raw = data["metadata"]
    meta = json.loads(str(meta_raw.item() if hasattr(meta_raw, "item") else meta_raw))
    return {
        "fps": float(data["fps"]),
        "root_pos": np.asarray(data["root_pos"], dtype=np.float64),
        "root_quat_xyzw": np.asarray(data["root_quat_xyzw"], dtype=np.float64),
        "dof_pos": np.asarray(data["dof_pos"], dtype=np.float64),
        "joint_names": [str(x) for x in data["joint_names"]],
        "metadata": meta,
    }


def _yaw_from_quat(q: np.ndarray) -> float:
    x, y, z, w = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def _quat_yaw(yaw: float) -> np.ndarray:
    return np.array([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)], dtype=np.float64)


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        return out / np.linalg.norm(out)
    th0 = math.acos(dot)
    th = th0 * t
    s0 = math.sin(th0 - th) / math.sin(th0)
    s1 = math.sin(th) / math.sin(th0)
    return s0 * q0 + s1 * q1


def _register_motion_to_spawn(
    motion: dict[str, Any], spawn_pos: np.ndarray, spawn_quat: np.ndarray
) -> dict[str, Any]:
    rp = motion["root_pos"].copy()
    rq = motion["root_quat_xyzw"].copy()
    p0 = rp[0].copy()
    q0 = rq[0].copy()
    spawn_xy = spawn_pos[:2].copy()
    ref_xy0 = p0[:2]
    dyaw = _yaw_from_quat(spawn_quat) - _yaw_from_quat(q0)
    qdy = _quat_yaw(dyaw)
    c, s = math.cos(dyaw), math.sin(dyaw)
    rz = np.array([[c, -s], [s, c]], dtype=np.float64)
    for i in range(rp.shape[0]):
        dxy = rp[i, :2] - ref_xy0
        rp[i, :2] = spawn_xy + rz @ dxy
        rp[i, 2] = motion["root_pos"][i, 2]
        rq[i] = _quat_mul(qdy, rq[i])
        n = np.linalg.norm(rq[i])
        if n > 1e-8:
            rq[i] /= n
    dt = 1.0 / float(motion["fps"])
    lin_v = np.zeros_like(rp)
    ang_v = np.zeros((rp.shape[0], 3))
    if rp.shape[0] > 1:
        lin_v[1:] = (rp[1:] - rp[:-1]) / dt
        lin_v[0] = lin_v[1]
        for i in range(1, rp.shape[0]):
            y0 = _yaw_from_quat(rq[i - 1])
            y1 = _yaw_from_quat(rq[i])
            ang_v[i, 2] = (y1 - y0) / dt
        ang_v[0] = ang_v[1]
    return {
        **motion,
        "root_pos": rp,
        "root_quat_xyzw": rq,
        "root_lin_vel_w": lin_v,
        "root_ang_vel_w": ang_v,
    }


def _sample_motion(motion: dict[str, Any], t: float) -> tuple[np.ndarray, ...]:
    fps = float(motion["fps"])
    t = max(0.0, t)
    dof = motion["dof_pos"]
    t_end = (dof.shape[0] - 1) / fps
    t = min(t, t_end)
    u = t * fps
    i0 = int(min(max(u, 0), dof.shape[0] - 2))
    a = u - i0
    p = (1 - a) * motion["root_pos"][i0] + a * motion["root_pos"][i0 + 1]
    q = _slerp(motion["root_quat_xyzw"][i0], motion["root_quat_xyzw"][i0 + 1], a)
    d = (1 - a) * dof[i0] + a * dof[i0 + 1]
    v = (1 - a) * motion["root_lin_vel_w"][i0] + a * motion["root_lin_vel_w"][i0 + 1]
    w = (1 - a) * motion["root_ang_vel_w"][i0] + a * motion["root_ang_vel_w"][i0 + 1]
    return p, q, d, v, w


def _standing_motion(
    spawn_pos: np.ndarray, spawn_quat: np.ndarray, gmr_dof: np.ndarray, fps: float = 30.0
) -> dict[str, Any]:
    n = int(STAND_DURATION_S * fps) + 1
    rp = np.tile(spawn_pos[None, :], (n, 1))
    rq = np.tile(spawn_quat[None, :], (n, 1))
    return {
        "fps": fps,
        "root_pos": rp,
        "root_quat_xyzw": rq,
        "dof_pos": np.tile(gmr_dof[None, :], (n, 1)),
        "joint_names": [],
        "metadata": {"clip": "standing_default_pose", "synthetic": True},
        "root_lin_vel_w": np.zeros_like(rp),
        "root_ang_vel_w": np.zeros((n, 3), dtype=np.float64),
    }


def _gmr_gains(robot: Articulation) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, effort, stiff, damp = _implicit_pd_torque(robot, robot.data.default_joint_pos.torch)
    k = np.zeros(29, dtype=np.float64)
    d_out = np.zeros(29, dtype=np.float64)
    lim = np.zeros(29, dtype=np.float64)
    for gi, ii in enumerate(GMR_TO_ISAAC_IDX):
        k[gi] = float(stiff[ii].item())
        d_out[gi] = float(damp[ii].item())
        lim[gi] = float(effort[ii].item())
    return k, d_out, lim


def _run() -> None:
    motion_raw = _load_motion(_args_cli.motion)
    sim = SimulationContext(_newton_sim_cfg())
    mat = sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6)
    sim_utils.GroundPlaneCfg(physics_material=mat).func(
        "/World/defaultGroundPlane", sim_utils.GroundPlaneCfg(physics_material=mat)
    )
    sim_utils.DomeLightCfg(intensity=2000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2000.0))
    robot = Articulation(_make_robot_cfg())
    sim.reset()
    robot.reset()
    mass = float(robot.data.default_mass.torch[0].sum().item())
    default_q = robot.data.default_joint_pos.torch.clone()
    spawn_pose = robot.data.default_root_pose.torch[0].detach().cpu().numpy()
    spawn_pos = spawn_pose[:3]
    spawn_quat = spawn_pose[3:7]

    if motion_raw["metadata"].get("synthetic") and motion_raw["metadata"].get("clip") == "standing_default_pose":
        gmr0 = np.array([default_q[0, i].item() for i in GMR_TO_ISAAC_IDX])
        motion = _standing_motion(spawn_pos, spawn_quat, gmr0, fps=float(motion_raw["fps"]))
    else:
        motion = _register_motion_to_spawn(motion_raw, spawn_pos, spawn_quat)

    duration = (motion["dof_pos"].shape[0] - 1) / float(motion["fps"])
    if motion["metadata"].get("clip") == "standing_default_pose":
        duration = STAND_DURATION_S

    p0, q0, d0, _, _ = _sample_motion(motion, 0.0)
    q_init = _gmr_to_full_q(default_q, d0)
    pose0 = torch.zeros((1, 7), device=sim.device)
    pose0[0, :3] = torch.tensor(p0, device=sim.device, dtype=torch.float32)
    pose0[0, 3:7] = torch.tensor(q0, device=sim.device, dtype=torch.float32)
    robot.write_root_link_pose_to_sim_index(root_pose=pose0)
    robot.write_joint_position_to_sim_index(position=q_init)
    robot.write_data_to_sim()
    robot.update(SIM_DT)

    ankle_ids, _ = robot.find_bodies([".*ankle_roll_link"])
    stiff_gmr, damp_gmr, lim_gmr = _gmr_gains(robot)
    n_steps = int(duration / SIM_DT) + 1
    mode = _args_cli.mode

    times: list[float] = []
    qs: list[np.ndarray] = []
    q_refs: list[np.ndarray] = []
    root_pos: list[np.ndarray] = []
    root_quat: list[np.ndarray] = []
    root_pos_ref: list[np.ndarray] = []
    root_quat_ref: list[np.ndarray] = []
    taus: list[np.ndarray] = []
    tau_applied_gmr: list[np.ndarray] = []
    tau_computed_gmr: list[np.ndarray] = []
    ankles: list[np.ndarray] = []
    fall_step = -1

    for step in range(n_steps):
        t = step * SIM_DT
        rp, rq, gd, rv, rw = _sample_motion(motion, t)
        q_tgt = _gmr_to_full_q(default_q, gd)
        if mode == "kin_root":
            pose = torch.zeros((1, 7), device=sim.device)
            pose[0, :3] = torch.tensor(rp, device=sim.device, dtype=torch.float32)
            pose[0, 3:7] = torch.tensor(rq, device=sim.device, dtype=torch.float32)
            vel = torch.zeros((1, 6), device=sim.device)
            vel[0, :3] = torch.tensor(rv, device=sim.device, dtype=torch.float32)
            vel[0, 3:6] = torch.tensor(rw, device=sim.device, dtype=torch.float32)
            robot.write_root_link_pose_to_sim_index(root_pose=pose)
            robot.write_root_velocity_to_sim_index(root_velocity=vel)
        robot.set_joint_position_target_index(target=q_tgt)
        robot.write_data_to_sim()
        sim.step()
        robot.update(SIM_DT)

        if mode == "free" and fall_step < 0 and _fall_detect(robot):
            fall_step = step

        q_sim = robot.data.joint_pos.torch[0]
        tau, _, _, _ = _implicit_pd_torque(robot, q_tgt)
        tau_gmr = np.array([float(tau[ii].item()) for ii in GMR_TO_ISAAC_IDX])
        comp = robot.data.computed_torque.torch[0]
        appl = robot.data.applied_torque.torch[0]
        tau_computed_gmr.append(
            np.array([float(comp[ii].item()) for ii in GMR_TO_ISAAC_IDX], dtype=np.float64)
        )
        tau_applied_gmr.append(
            np.array([float(appl[ii].item()) for ii in GMR_TO_ISAAC_IDX], dtype=np.float64)
        )
        ankle_xy = []
        for bi in ankle_ids[:2]:
            ankle_xy.append(robot.data.body_pos_w.torch[0, bi].detach().cpu().numpy())
        while len(ankle_xy) < 2:
            ankle_xy.append(np.zeros(3))

        times.append(t)
        qs.append(_full_to_gmr(q_sim))
        q_refs.append(gd.astype(np.float64))
        root_pos.append(robot.data.root_pos_w.torch[0].detach().cpu().numpy())
        root_quat.append(robot.data.root_quat_w.torch[0].detach().cpu().numpy())
        root_pos_ref.append(rp)
        root_quat_ref.append(rq)
        taus.append(tau_gmr)
        ankles.append(np.stack(ankle_xy, axis=0))

        if mode == "free" and fall_step >= 0:
            break

    vers = _versions()
    sim_settings = {
        "dt_s": SIM_DT,
        "control_rate_hz": 1.0 / SIM_DT,
        "physics_num_substeps": 1,
        "njmax": NJMAX,
        "nconmax": NCONMAX,
        "initial_root_velocity_convention": (
            "zero at spawn (write_root_velocity not called at init); "
            "kin_root mode overwrites root pose+velocity from motion reference each step"
        ),
        "fall_detector": f"z<{FALL_Z} m or tilt>{FALL_TILT_DEG} deg",
    }
    np.savez_compressed(
        _args_cli.out,
        time=np.asarray(times, dtype=np.float64),
        q=np.asarray(qs, dtype=np.float64),
        q_ref=np.asarray(q_refs, dtype=np.float64),
        root_pos=np.asarray(root_pos, dtype=np.float64),
        root_quat_xyzw=np.asarray(root_quat, dtype=np.float64),
        root_pos_ref=np.asarray(root_pos_ref, dtype=np.float64),
        root_quat_xyzw_ref=np.asarray(root_quat_ref, dtype=np.float64),
        tau=np.asarray(taus, dtype=np.float64),
        tau_computed_gmr=np.asarray(tau_computed_gmr, dtype=np.float64),
        tau_applied_gmr=np.asarray(tau_applied_gmr, dtype=np.float64),
        stiffness_gmr=stiff_gmr,
        damping_gmr=damp_gmr,
        effort_limit_gmr=lim_gmr,
        ankle_pos=np.asarray(ankles, dtype=np.float64),
        fall_step=np.int64(fall_step),
        mode=np.asarray(mode),
        mass_kg=np.float64(mass),
        joint_names=np.asarray(
            motion.get("joint_names") or list(GMR_JOINT_NAMES), dtype=object
        ),
        motion_metadata=np.asarray(json.dumps(motion["metadata"]), dtype=object),
        versions=np.asarray(json.dumps(vers), dtype=object),
        sim_settings=np.asarray(json.dumps(sim_settings), dtype=object),
        dt=np.float64(SIM_DT),
        assist_wrench_note=np.asarray(
            "Not measured: Newton path has no reliable foot contact force readback; "
            "pelvis pin / assist wrench dropped after 6B (see docs/pivot_log.md).",
            dtype=object,
        ),
    )
    print(
        json.dumps(
            {
                "out": _args_cli.out,
                "mode": mode,
                "steps": len(times),
                "fall_step": int(fall_step),
                "duration_s": float(times[-1]) if times else 0.0,
                **vers,
            },
            indent=2,
        ),
        flush=True,
    )
    simulation_app.close()


if __name__ == "__main__":
    _run()
