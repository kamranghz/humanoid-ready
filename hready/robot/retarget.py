"""CLI bridge: AMASS clip -> GMR (subprocess) -> HumanoidReady motion npz."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

print(sys.executable, flush=True)

from hready.data.amass import amass_root_from_config, clip_flags, load_index, load_paths_config

GMR_JOINT_NAMES: tuple[str, ...] = (
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

GMR_LICENSE = "MIT"
DEFAULT_GMR_ROBOT = "unitree_g1"


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "configs" / "paths.example.yaml").is_file():
            return parent
    raise FileNotFoundError("Could not locate repo root.")


def _gmr_paths(cfg: dict[str, Any]) -> tuple[Path, Path]:
    data_root = Path(cfg["data_root"])
    gmr_root = Path(cfg.get("gmr_root", data_root.parent / "third_party" / "GMR"))
    gmr_python = Path(
        cfg.get(
            "gmr_python",
            r"C:\Users\kghol072\AppData\Local\anaconda3\envs\hready-gmr\python.exe",
        )
    )
    return gmr_root.resolve(), gmr_python.resolve()


def _git_commit(repo: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _normalize_rel_clip(clip: str) -> str:
    clip = clip.replace("\\", "/").strip("/")
    if clip.endswith(".npz"):
        return clip
    if not clip.endswith("_stageii.npz"):
        return f"{clip}_stageii.npz"
    return clip


def retarget_clip(
    rel_clip: str,
    out_npz: Path,
    *,
    allow_excluded: bool = False,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Retarget one AMASS clip; write ``out_npz`` and return summary dict."""
    cfg = cfg or load_paths_config()
    amass_root = amass_root_from_config(cfg)
    gmr_root, gmr_python = _gmr_paths(cfg)
    rel = _normalize_rel_clip(rel_clip)
    npz_path = amass_root / rel
    if not npz_path.is_file():
        load_index()  # ensure index exists for clearer errors
        raise FileNotFoundError(f"Clip not under amass_root: {rel} -> {npz_path}")

    flags = clip_flags(rel)
    if flags.get("exclude_contact") and not allow_excluded:
        raise ValueError(
            f"Clip {rel} has exclude_contact=True ({flags.get('skate_reason', '')}); "
            "pass --allow-excluded to override."
        )

    gmr_script = _repo_root() / "hready" / "robot" / "gmr_headless.py"
    smplx_body = gmr_root / "assets" / "body_models"
    if not gmr_script.is_file():
        raise FileNotFoundError(gmr_script)
    if not gmr_python.is_file():
        raise FileNotFoundError(f"gmr_python not found: {gmr_python}")

    out_npz = out_npz.resolve()
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="hready_gmr_") as tmp:
        pkl_path = Path(tmp) / "gmr_motion.pkl"
        cmd = [
            str(gmr_python),
            str(gmr_script),
            "--clip",
            str(npz_path.resolve()),
            "--out-pkl",
            str(pkl_path),
            "--smplx-body-dir",
            str(smplx_body),
            "--robot",
            DEFAULT_GMR_ROBOT,
            "--tgt-fps",
            "30",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        if proc.returncode != 0:
            raise RuntimeError(f"GMR subprocess failed (code {proc.returncode})")

        import pickle

        with pkl_path.open("rb") as f:
            motion = pickle.load(f)

    gmr_commit = _git_commit(gmr_root)
    metadata = {
        "clip": rel,
        "gmr_commit": gmr_commit,
        "gmr_robot": motion.get("robot", DEFAULT_GMR_ROBOT),
        "gmr_ik_config": "smplx_to_g1.json",
        "gmr_license": GMR_LICENSE,
        "retargeter": "general_motion_retargeting",
        "smoothing": "none",
    }

    fps = float(motion["fps"])
    root_pos = np.asarray(motion["root_pos"], dtype=np.float64)
    root_quat_xyzw = np.asarray(motion["root_rot"], dtype=np.float64)
    dof_pos = np.asarray(motion["dof_pos"], dtype=np.float64)
    joint_names = np.asarray(GMR_JOINT_NAMES, dtype=object)

    if dof_pos.shape[1] != len(GMR_JOINT_NAMES):
        raise ValueError(f"Expected {len(GMR_JOINT_NAMES)} DoF, got {dof_pos.shape[1]}")

    np.savez_compressed(
        out_npz,
        fps=np.float64(fps),
        root_pos=root_pos,
        root_quat_xyzw=root_quat_xyzw,
        dof_pos=dof_pos,
        joint_names=joint_names,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    wall_s = time.perf_counter() - t0
    summary = {
        "out": str(out_npz),
        "n_frames": int(root_pos.shape[0]),
        "fps": fps,
        "root_z_min": float(root_pos[:, 2].min()),
        "root_z_max": float(root_pos[:, 2].max()),
        "wall_s": wall_s,
        "metadata": metadata,
    }
    print(
        f"[retarget] wrote {out_npz} frames={summary['n_frames']} fps={fps} "
        f"root_z=[{summary['root_z_min']:.3f},{summary['root_z_max']:.3f}] wall_s={wall_s:.2f}",
        flush=True,
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Retarget AMASS SMPL-X clip to Unitree G1 via GMR.")
    parser.add_argument("--clip", required=True, help="Path relative to amass_root (e.g. CMU/132/132_35)")
    parser.add_argument("--out", required=True, type=Path, help="Output .npz motion file")
    parser.add_argument(
        "--allow-excluded",
        action="store_true",
        help="Allow clips flagged exclude_contact in amass_clip_flags.json",
    )
    args = parser.parse_args(argv)
    retarget_clip(args.clip, args.out, allow_excluded=args.allow_excluded)


if __name__ == "__main__":
    main()
