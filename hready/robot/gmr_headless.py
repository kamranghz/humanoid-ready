"""Run inside conda env ``hready-gmr`` only (subprocess from ``hready.robot.retarget``).

Uses third-party `general_motion_retargeting` (MIT) from ``gmr_root``; does not import ``hready``.
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

print(sys.executable, flush=True)

GMR_JOINT_NAMES = [
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
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Headless GMR: AMASS SMPL-X npz -> G1 pickle.")
    parser.add_argument("--clip", type=Path, required=True, help="Absolute path to AMASS *_stageii.npz")
    parser.add_argument("--out-pkl", type=Path, required=True, help="Output pickle path")
    parser.add_argument(
        "--smplx-body-dir",
        type=Path,
        required=True,
        help="SMPL-X body models directory (GMR assets/body_models)",
    )
    parser.add_argument("--robot", type=str, default="unitree_g1")
    parser.add_argument("--tgt-fps", type=float, default=30.0)
    args = parser.parse_args()

    clip = args.clip.resolve()
    out_pkl = args.out_pkl.resolve()
    smplx_body = args.smplx_body_dir.resolve()
    if not clip.is_file():
        raise FileNotFoundError(clip)
    out_pkl.parent.mkdir(parents=True, exist_ok=True)

    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting.utils.smpl import get_smplx_data_offline_fast, load_smplx_file

    t0 = time.perf_counter()
    smplx_data, body_model, smplx_output, human_h = load_smplx_file(str(clip), str(smplx_body))
    frames, fps = get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=args.tgt_fps)
    retarget = GMR(actual_human_height=human_h, src_human="smplx", tgt_robot=args.robot)
    qpos_list = []
    for fr in frames:
        qpos_list.append(retarget.retarget(fr))
    elapsed = time.perf_counter() - t0

    import numpy as np

    root_pos = np.asarray([q[:3] for q in qpos_list], dtype=np.float64)
    # MuJoCo free joint quat in qpos is wxyz; store xyzw for HumanoidReady.
    root_quat_xyzw = np.asarray([q[3:7][[1, 2, 3, 0]] for q in qpos_list], dtype=np.float64)
    dof_pos = np.asarray([q[7:] for q in qpos_list], dtype=np.float64)

    dof_names = [k for k in retarget.robot_dof_names.keys() if k != "pelvis"]
    if dof_names != GMR_JOINT_NAMES:
        print("[gmr_headless] WARN: robot_dof_names order differs from expected GMR 29-DoF list", flush=True)
        print("[gmr_headless] got:", dof_names, flush=True)

    motion = {
        "fps": float(fps),
        "root_pos": root_pos,
        "root_rot": root_quat_xyzw,
        "dof_pos": dof_pos,
        "joint_names": GMR_JOINT_NAMES,
        "robot": args.robot,
        "gmr_wall_s": elapsed,
    }
    with out_pkl.open("wb") as f:
        pickle.dump(motion, f)

    print(
        f"[gmr_headless] n_frames={dof_pos.shape[0]} fps={fps} n_dof={dof_pos.shape[1]} "
        f"wall_s={elapsed:.2f} out={out_pkl}",
        flush=True,
    )


if __name__ == "__main__":
    main()
