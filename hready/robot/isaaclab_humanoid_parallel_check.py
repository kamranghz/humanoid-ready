"""SMPL-X-skeleton humanoid in many parallel copies on Isaac Lab + Newton: speed, memory and the 5 s stand at the tracking step.

One process runs one copy count (Isaac Lab keeps a single simulation context per process); a shell loop covers several counts.
The control step is the one used for tracking: simulation step 1/180 s, control at 30 Hz. All copies start in the same standing
pose, so the copies must also agree with each other, which is reported as a check on the cloning.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

if __name__ != "__main__":
    raise ImportError("hready.robot.isaaclab_humanoid_parallel_check is an Isaac Lab entrypoint; run it with Isaac Lab's Python.")

print(sys.executable, flush=True)

from isaaclab.app import AppLauncher

_parser = argparse.ArgumentParser(description="Many parallel copies of the SMPL-X-skeleton humanoid on Isaac Lab + Newton.")
_parser.add_argument("--asset", type=str, required=True, help="folder written by scripts/humanoid_asset_build.py")
_parser.add_argument("--out", type=str, required=True, help="result json")
_parser.add_argument("--num-envs", type=int, required=True, help="number of parallel copies in this process")
_parser.add_argument("--env-spacing", type=float, default=3.0)
_parser.add_argument("--seconds", type=float, default=5.0)
_parser.add_argument("--sim-dt", type=float, default=1.0 / 180.0, help="physics step in seconds")
_parser.add_argument("--control-hz", type=float, default=30.0)
AppLauncher.add_app_launcher_args(_parser)
_args = _parser.parse_args()
_app = AppLauncher(_args)
simulation_app = _app.app

import isaaclab
import isaaclab.sim as sim_utils
import isaaclab_newton
import newton
import numpy as np
import torch
import warp as wp
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hready.robot import humanoid_asset as H
from hready.robot import usd_drives as U

NJMAX = 600  # per world, as in the single-copy stand check
NCONMAX = 150
FALL_Z_FRAC = 0.5
FALL_TILT_DEG = 45.0
FEET = ("left_ankle", "right_ankle", "left_foot", "right_foot")


def _gpu_used_gb() -> float:
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1e9


def _primed_usd_copy(asset_dir: Path) -> tuple[str, str]:
    """A private copy of the newest importer output (the importer writes numbered folders under usd/); the copy gets primed."""
    usd_root = asset_dir / "usd"
    target = usd_root / "smplx_humanoid_primed"
    if not target.is_dir():
        sources = sorted(usd_root.glob("smplx_humanoid_sim_*"), key=lambda p: p.stat().st_mtime)
        if not sources:
            raise FileNotFoundError(f"no importer output under {usd_root}; run hready/robot/isaaclab_humanoid_stand.py once")
        shutil.copytree(sources[-1], target)
        source = str(sources[-1].name)
    else:
        source = "existing copy"
    files = sorted(target.glob("*.usd*"))
    if len(files) != 1:
        raise FileNotFoundError(f"expected one top-level USD file in {target}, found {[f.name for f in files]}")
    return str(files[0]), source


def _make_scene_cfg(meta: dict, usd_path: str, n: int):
    groups: dict[str, list[str]] = {}
    for name, g in meta["group_of_joint"].items():
        if name != "pelvis":
            groups.setdefault(g, []).append(f"{name}_(x:[0-2]|[xyz])")
    actuators = {
        g: ImplicitActuatorCfg(joint_names_expr=exprs, stiffness=meta["pd"][g]["kp"], damping=meta["pd"][g]["kd"],
                               effort_limit_sim=meta["pd"][g]["torque_limit"])
        for g, exprs in groups.items()
    }
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=usd_path),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.5)), actuators=actuators)
    mat = sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0)

    @configclass
    class _Scene(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg(physics_material=mat))
        light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))

    cfg = _Scene(num_envs=n, env_spacing=_args.env_spacing, replicate_physics=True)
    cfg.robot = robot
    return cfg


def _run() -> None:
    asset_dir = Path(_args.asset)
    meta = json.loads((asset_dir / "smplx_humanoid.json").read_text(encoding="utf-8"))
    usd_path, usd_source = _primed_usd_copy(asset_dir)
    prime = U.prime_usd_file(usd_path)  # before anything is built; the cloned model reads this file
    prime["copied_from"] = usd_source

    n, dt = _args.num_envs, _args.sim_dt
    decimation = max(1, round(1.0 / (dt * _args.control_hz)))
    n_control = int(_args.seconds / (dt * decimation))
    gpu_before = _gpu_used_gb()
    t0 = time.time()
    sim = SimulationContext(SimulationCfg(
        dt=dt, physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(njmax=NJMAX, nconmax=NCONMAX, cone="pyramidal", impratio=1.0,
                                                          integrator="implicitfast", iterations=100), num_substeps=1)))
    scene = InteractiveScene(_make_scene_cfg(meta, usd_path, n))
    sim.reset()
    robot = scene["robot"]
    robot.reset()
    build_s = time.time() - t0
    gpu_built = _gpu_used_gb()
    dev = sim.device

    r_root = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
    local = np.stack([np.eye(3)] + [H.hinges_to_matrix(np.zeros(3)) for _ in range(21)])
    root_z = -H.lowest_height(meta, r_root, local) + 0.003
    pose = torch.zeros((n, 7), device=dev)
    pose[:, :3] = scene.env_origins[:, :3].to(dev) + torch.tensor([0.0, 0.0, root_z], device=dev)
    pose[:, 3:7] = torch.tensor(H.matrix_to_quat_xyzw(r_root), device=dev, dtype=torch.float32)
    q_tgt = torch.zeros((n, len(robot.joint_names)), device=dev)
    robot.write_root_link_pose_to_sim_index(root_pose=pose)
    robot.write_joint_position_to_sim_index(position=q_tgt)
    robot.write_data_to_sim()
    robot.update(dt)

    ids_p = robot.find_bodies(["pelvis"])[0][0]
    feet_ids = [robot.find_bodies([nm])[0][0] for nm in FEET]
    origins = scene.env_origins[:, :3].to(dev)
    feet0 = robot.data.body_pos_w.torch[:, feet_ids, :2].clone()
    pz0 = robot.data.body_pos_w.torch[:, ids_p, 2].clone()
    pz_min, tilt_max = pz0.clone(), torch.zeros(n, device=dev)
    pelvis_xyz = []
    warm = min(5, n_control - 1)
    t_start = None
    for k in range(n_control):
        if k == warm:
            torch.cuda.synchronize()
            t_start = time.time()
        robot.set_joint_position_target_index(target=q_tgt)
        for _ in range(decimation):
            robot.write_data_to_sim()
            sim.step()
            robot.update(dt)
        p = robot.data.body_pos_w.torch[:, ids_p]
        x, y, z, w = robot.data.root_quat_w.torch.unbind(-1)
        tilt = torch.rad2deg(torch.acos(torch.clamp(2 * (y * z + x * w), -1.0, 1.0)))
        pz_min = torch.minimum(pz_min, p[:, 2])
        tilt_max = torch.maximum(tilt_max, torch.nan_to_num(tilt, nan=180.0))
        pelvis_xyz.append((p - origins).clone())
    torch.cuda.synchronize()
    timed = time.time() - t_start
    timed_control = n_control - warm
    gpu_after = _gpu_used_gb()

    traj = torch.stack(pelvis_xyz, dim=1)  # (n, steps, 3), relative to each copy's origin
    last = traj[:, -int(2.0 / (dt * decimation)):]
    finite = torch.isfinite(traj).all(dim=(1, 2))
    drop = (pz0 - pz_min).cpu().numpy()
    tilt_np = tilt_max.cpu().numpy()
    xy_range = (last[..., :2].amax(1) - last[..., :2].amin(1)).amax(-1).cpu().numpy()
    z_range = (last[..., 2].amax(1) - last[..., 2].amin(1)).cpu().numpy()
    feet1 = robot.data.body_pos_w.torch[:, feet_ids, :2]
    foot_drift = (feet1 - feet0).norm(dim=-1).amax(-1).cpu().numpy()
    fell = ((pz_min < FALL_Z_FRAC * pz0) | (tilt_max > FALL_TILT_DEG) | ~finite).cpu().numpy()
    spread = float((traj[:, -1] - traj[0:1, -1]).abs().amax().item())  # copies against copy 0, final pelvis position
    ok = ((drop <= 0.03) & (tilt_np < 5.0) & (foot_drift <= 0.01) & (xy_range <= 0.005) & (z_range <= 0.002)
          & finite.cpu().numpy() & ~fell)
    sim_steps = timed_control * decimation
    origin_xy = origins[:, :2].cpu().numpy()
    origin_dist = np.linalg.norm(origin_xy, axis=1)
    final_xy = traj[:, -1, :2].cpu().numpy()

    def _corr(v: np.ndarray) -> float | None:
        return float(np.corrcoef(origin_dist, v)[0, 1]) if n > 2 and v.std() > 0 and origin_dist.std() > 0 else None

    worst = np.argsort(-foot_drift)[:5]
    copy_spread = {
        "max_origin_distance_m": float(origin_dist.max()),
        "worst_5_copies_by_foot_drift": [{"copy": int(i), "origin_xy": origin_xy[i].round(2).tolist(), "foot_drift_m": float(foot_drift[i]),
                                          "xy_range_last_2s_m": float(xy_range[i]), "tilt_deg": float(tilt_np[i])} for i in worst],
        "corr_foot_drift_vs_origin_distance": _corr(foot_drift), "corr_xy_range_vs_origin_distance": _corr(xy_range),
        "final_pelvis_xy_relative_to_origin_mm_min": (1000 * final_xy.min(0)).round(2).tolist(),
        "final_pelvis_xy_relative_to_origin_mm_max": (1000 * final_xy.max(0)).round(2).tolist(),
        "copy_0": {"foot_drift_m": float(foot_drift[0]), "xy_range_last_2s_m": float(xy_range[0]), "tilt_deg": float(tilt_np[0])},
        "per_copy": None if n > 64 else [{"copy": i, "origin_xy": origin_xy[i].round(2).tolist(), "foot_drift_m": float(foot_drift[i]),
                                          "xy_range_last_2s_m": float(xy_range[i]), "tilt_deg": float(tilt_np[i]),
                                          "final_pelvis_xy_mm": (1000 * final_xy[i]).round(2).tolist()} for i in range(n)],
    }
    report = {
        "num_envs": n, "sim_dt": dt, "control_hz": 1.0 / (dt * decimation), "decimation": decimation, "seconds": _args.seconds,
        "control_steps": n_control, "joint_count": len(robot.joint_names), "prime": prime,
        "build_seconds": build_s, "gpu_used_gb_before": gpu_before, "gpu_used_gb_after_build": gpu_built, "gpu_used_gb_after_run": gpu_after,
        "timed_seconds": timed, "sim_steps_timed": sim_steps,
        "sim_steps_per_second_all_copies": sim_steps / timed, "copy_steps_per_second": n * sim_steps / timed,
        "faster_than_real_time_factor": (sim_steps * dt) / timed,
        "stand": {"copies_passing_all": int(ok.sum()), "copies_fell": int(fell.sum()), "worst_drop_m": float(drop.max()),
                  "worst_tilt_deg": float(tilt_np.max()), "worst_foot_drift_m": float(foot_drift.max()),
                  "worst_pelvis_xy_range_last_2s_m": float(xy_range.max()), "worst_pelvis_z_range_last_2s_m": float(z_range.max()),
                  "criteria": "drop<=3cm, tilt<5deg, feet drift<=1cm, pelvis xy range last 2s<=5mm, z range last 2s<=2mm, no fall, no NaN "
                              "(the centre-of-mass criterion of the single-copy check is not repeated here)"},
        "copies_agree_max_abs_pelvis_difference_m": spread, "copy_spread": copy_spread, "env_spacing_m": _args.env_spacing,
        "passed": bool(ok.all()),
        "versions": {"isaaclab": getattr(isaaclab, "__version__", "unknown"), "isaaclab_newton": isaaclab_newton.__version__,
                     "newton": getattr(newton, "__version__", "unknown"), "warp": wp.__version__},
    }
    Path(_args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(_args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1), flush=True)
    print(f"SUMMARY num_envs={n} dt={dt:.5f} control_hz={report['control_hz']:.1f} passed={report['passed']} fell={int(fell.sum())}/{n} "
          f"build_s={build_s:.1f} copy_steps_per_s={report['copy_steps_per_second']:.0f} real_time_x={report['faster_than_real_time_factor']:.2f} "
          f"gpu_gb_run={gpu_after:.1f} worst_drop_cm={100 * float(drop.max()):.1f} worst_tilt_deg={float(tilt_np.max()):.1f} "
          f"worst_xy_range_mm={1000 * float(xy_range.max()):.1f} copies_disagree_mm={1000 * spread:.2f}", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    _run()
