"""Open-loop PD baseline of the tracking study (E6, stage 1): the reference joint angles go to the PD controllers with no policy.

One copy of the SMPL-X-skeleton humanoid per reference window, all in one scene on Isaac Lab + Newton. Each copy starts at the first
reference frame (root pose and joint angles from the reference, zero velocity) and receives, at every control step, the reference
joint angles of that time as PD targets. Nothing else acts on the body. Simulation step 0.005 s, control step 6 simulation steps
(33.3 Hz); the 30 Hz reference is interpolated at the control times (linear, normalised linear for the root quaternion).
Metrics: hready/robot/tracking_metrics.py. No reset after a termination: the termination time is only recorded.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if __name__ != "__main__":
    raise ImportError("hready.robot.isaaclab_open_loop_baseline is an Isaac Lab entrypoint; run it with Isaac Lab's Python.")

print(sys.executable, flush=True)

from isaaclab.app import AppLauncher

_parser = argparse.ArgumentParser(description="Open-loop PD baseline on the reference windows.")
_parser.add_argument("--asset", type=str, required=True, help="folder written by scripts/humanoid_asset_build.py")
_parser.add_argument("--reference-dir", type=str, required=True, help="folder with manifest.json and the window npz files")
_parser.add_argument("--out", type=str, required=True, help="result json (an npz with the simulated joints is written beside it)")
_parser.add_argument("--sim-dt", type=float, default=0.005)
_parser.add_argument("--decimation", type=int, default=6)
_parser.add_argument("--env-spacing", type=float, default=3.0)
AppLauncher.add_app_launcher_args(_parser)
_args = _parser.parse_args()
_app = AppLauncher(_args)
simulation_app = _app.app

import isaaclab
import isaaclab_newton
import newton
import numpy as np
import torch
import warp as wp
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hready.robot import isaaclab_parallel_scene as S
from hready.robot import tracking_metrics as M
from hready.robot import tracking_reference as R
from hready.robot import usd_drives as U

END_EFFECTORS = ("left_ankle", "right_ankle", "left_wrist", "right_wrist")


def _run() -> None:
    asset_dir = Path(_args.asset)
    meta = json.loads((asset_dir / "smplx_humanoid.json").read_text(encoding="utf-8"))
    ref_dir = Path(_args.reference_dir)
    manifest = json.loads((ref_dir / "manifest.json").read_text(encoding="utf-8"))
    windows = manifest["windows"]
    n = len(windows)
    dt, dec = _args.sim_dt, _args.decimation
    ctrl_dt = dt * dec

    refs = []
    for w in windows:
        z = np.load(ref_dir / w["file"])
        fps = float(z["fps"])
        n_ctrl = R.control_steps(len(z["hinges"]), fps, ctrl_dt)
        t_ctrl = np.arange(n_ctrl) * ctrl_dt
        refs.append({"hinges": R.resample(z["hinges"], t_ctrl, fps), "root_pos": R.resample(z["root_pos"], t_ctrl, fps),
                     "quat": R.resample(z["root_quat_xyzw"], t_ctrl, fps, normalise=True), "joints": R.resample(z["joints"], t_ctrl, fps),
                     "steps": n_ctrl})
    n_max = max(r["steps"] for r in refs)

    def _pad(key: str) -> np.ndarray:  # repeat the last frame so every copy can run n_max steps; metrics use only the valid steps
        return np.stack([np.concatenate([r[key], np.repeat(r[key][-1:], n_max - r["steps"], axis=0)]) for r in refs])

    hing, root_pos, quat, ref_joints = _pad("hinges"), _pad("root_pos"), _pad("quat"), _pad("joints")

    usd_path, usd_source = S.primed_usd_copy(asset_dir)
    prime = U.prime_usd_file(usd_path)
    prime["copied_from"] = usd_source
    t0 = time.time()
    sim = SimulationContext(S.newton_sim_cfg(dt))
    scene = InteractiveScene(S.make_scene_cfg(meta, usd_path, n, _args.env_spacing))
    sim.reset()
    robot = scene["robot"]
    robot.reset()
    dev = sim.device
    build_s = time.time() - t0

    dof_to_hinge = np.array(S.hinge_index_of_dofs(list(robot.joint_names), meta))
    if len(dof_to_hinge) != 63:
        raise RuntimeError(f"expected 63 DOFs, found {len(dof_to_hinge)}")
    origins = scene.env_origins[:, :3].to(dev)
    body_ids = [robot.find_bodies([nm])[0][0] for nm in meta["joint_names"]]
    ee = [meta["joint_names"].index(nm) for nm in END_EFFECTORS]

    q_all = torch.tensor(hing[:, :, dof_to_hinge], dtype=torch.float32, device=dev)  # (n, n_max, 63) in simulation DOF order
    pose = torch.zeros((n, 7), device=dev)
    pose[:, :3] = origins + torch.tensor(root_pos[:, 0], dtype=torch.float32, device=dev)
    pose[:, 3:7] = torch.tensor(quat[:, 0], dtype=torch.float32, device=dev)
    robot.write_root_link_pose_to_sim_index(root_pose=pose)
    robot.write_joint_position_to_sim_index(position=q_all[:, 0])
    robot.write_data_to_sim()
    robot.update(dt)

    sim_joints = torch.zeros((n, n_max, 22, 3), device=dev)
    sim_quat = torch.zeros((n, n_max, 4), device=dev)
    t_run = time.time()
    for k in range(n_max):
        sim_joints[:, k] = robot.data.body_pos_w.torch[:, body_ids] - origins[:, None]
        sim_quat[:, k] = robot.data.root_quat_w.torch
        robot.set_joint_position_target_index(target=q_all[:, k])
        for _ in range(dec):
            robot.write_data_to_sim()
            sim.step()
            robot.update(dt)
    torch.cuda.synchronize()
    run_s = time.time() - t_run
    sj, sq = sim_joints.cpu().numpy().astype(np.float64), sim_quat.cpu().numpy().astype(np.float64)

    results = []
    for i, w in enumerate(windows):
        s = refs[i]["steps"]
        m = M.evaluate_clip(sj[i, :s], ref_joints[i, :s], sq[i, :s], quat[i, :s], ctrl_dt, ee)
        m["initial_pose_error_mm"] = float(1000 * np.linalg.norm(sj[i, 0] - ref_joints[i, 0], axis=-1).mean())
        m.update({"file": w["file"], "category": w["category"], "rank_in_category": w["rank_in_category"], "stage2_clip": w["stage2_clip"],
                  "rel_path": w["rel_path"]})
        results.append(m)

    def _agg(rows: list[dict]) -> dict:
        return {"windows": len(rows), "completed": int(sum(r["completed"] for r in rows)),
                "success_phc": int(sum(r["success_phc"] for r in rows)),
                "median_termination_time_s": float(np.median([r["termination_time_s"] for r in rows if not r["completed"]]))
                if any(not r["completed"] for r in rows) else None,
                "mean_mpjpe_global_mm": float(np.mean([r["mpjpe_global_mm"] for r in rows])),
                "mean_mpjpe_root_relative_mm": float(np.mean([r["mpjpe_root_relative_mm"] for r in rows])),
                "mean_velocity_error_m_s": float(np.mean([r["velocity_error_m_s"] for r in rows])),
                "mean_acceleration_error_m_s2": float(np.mean([r["acceleration_error_m_s2"] for r in rows]))}

    report = {
        "num_windows": n, "sim_dt": dt, "decimation": dec, "control_hz": 1.0 / ctrl_dt, "control_steps_max": n_max,
        "target_rule": "reference joint angles at the current control time, no lead; start from rest at the first reference frame",
        "anchor": "pelvis", "end_effectors": list(END_EFFECTORS),
        "thresholds": {"height_error_m": M.Z_TOL_M, "orientation_error_rad": M.ANGLE_TOL_RAD, "success_mean_joint_error_m": M.SUCC_TOL_M},
        "initial_pose_error_mm_max": max(r["initial_pose_error_mm"] for r in results),
        "all": _agg(results), "by_category": {c: _agg([r for r in results if r["category"] == c]) for c in manifest["windows_taken"]},
        "stage2_clips": [r for r in results if r["stage2_clip"]], "windows": results,
        "build_seconds": build_s, "run_seconds": run_s, "prime": prime,
        "versions": {"isaaclab": getattr(isaaclab, "__version__", "unknown"), "isaaclab_newton": isaaclab_newton.__version__,
                     "newton": getattr(newton, "__version__", "unknown"), "warp": wp.__version__},
    }
    out = Path(_args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    np.savez_compressed(out.with_suffix(".npz"), sim_joints=sj.astype(np.float32), sim_quat=sq.astype(np.float32),
                        ref_joints=ref_joints.astype(np.float32), steps=np.array([r["steps"] for r in refs]))
    print(json.dumps({k: report[k] for k in ("all", "by_category", "initial_pose_error_mm_max")}, indent=1), flush=True)
    print(f"SUMMARY windows={n} completed={report['all']['completed']}/{n} success_phc={report['all']['success_phc']}/{n} "
          f"initial_pose_error_mm_max={report['initial_pose_error_mm_max']:.2f} mpjpe_global_mm={report['all']['mean_mpjpe_global_mm']:.0f} "
          f"run_s={run_s:.1f}", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    _run()
