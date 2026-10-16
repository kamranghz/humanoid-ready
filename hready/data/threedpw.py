"""3DPW test-set loader (SMPL parameters, 3D joints, 2D keypoints, camera).

Reads ``<data_root>/datasets/3dpw/sequenceFiles/test/*.pkl`` (``data_root`` from ``configs/paths.yaml``).

Format as found in the released pkl files (checked on all 24 test sequences):

- ``poses``, ``trans``, ``betas``, ``jointPositions``, ``poses2d``, ``campose_valid`` are **lists with one
  entry per person** (1 or 2 people per sequence), not arrays with a people axis.
- ``poses`` (N, 72) SMPL axis-angle; ``trans`` (N, 3); ``betas`` (300,) clothed shape, of which the first 10 are
  the SMPL shape coefficients (one sequence stores 10).
- ``jointPositions`` (N, 72) = 24 SMPL joints x 3, in the **world** frame of ``trans`` (y up), not camera space;
  the pelvis joint minus ``trans`` is a constant offset in every frame.
- ``poses2d`` (N, 3, 18): rows are x, y, confidence for the 18 OpenPose keypoints.
- ``cam_poses`` (N, 4, 4) world-to-camera extrinsics, shared by the people; ``cam_intrinsics`` (3, 3).
- ``campose_valid`` (N,) per person: 0 where the camera pose is not valid for that frame.
- ``img_frame_ids`` (N,) image frame index of each row.

SMPL (not SMPL-X) parameters; ``smpl_to_smplx_pose`` is a stub (see its docstring).
"""

from __future__ import annotations

import argparse
import pickle
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import yaml

N_SMPL_JOINTS = 24
N_SHARED_JOINTS = 22  # pelvis + 21 body joints, common to SMPL and SMPL-X
SMPLX_POSE_DIM = 165  # global 3 + body 63 + jaw 3 + leye 3 + reye 3 + lhand 45 + rhand 45


class ThreeDPWSequence:
    """One 3DPW sequence file; per-frame access for one person (default person 0)."""

    def __init__(self, path: str | Path, person: int = 0) -> None:
        self.path = Path(path)
        with self.path.open("rb") as f:
            raw: dict[str, Any] = pickle.load(f, encoding="latin1")
        self.name = str(raw["sequence"])
        self.num_people = len(raw["poses"])
        if not 0 <= person < self.num_people:
            raise IndexError(f"{self.name}: person {person} not in 0..{self.num_people - 1}")
        self.person = person
        self.cam_intrinsics = np.asarray(raw["cam_intrinsics"], dtype=np.float64)
        self._frame_ids = np.asarray(raw["img_frame_ids"], dtype=np.int64)
        self._cam = np.asarray(raw["cam_poses"], dtype=np.float64)
        self._poses = np.asarray(raw["poses"][person], dtype=np.float64)
        self._trans = np.asarray(raw["trans"][person], dtype=np.float64)
        self._betas = np.asarray(raw["betas"][person], dtype=np.float64)[:10]
        self._joints = np.asarray(raw["jointPositions"][person], dtype=np.float64).reshape(-1, N_SMPL_JOINTS, 3)
        self._kp2d = np.asarray(raw["poses2d"][person], dtype=np.float64).transpose(0, 2, 1)  # (N, 18, 3)
        self._valid = np.asarray(raw["campose_valid"][person], dtype=bool)
        self.gender = str(raw["genders"][person])
        n = len(self._frame_ids)
        for key, arr in (
            ("poses", self._poses),
            ("trans", self._trans),
            ("jointPositions", self._joints),
            ("poses2d", self._kp2d),
            ("cam_poses", self._cam),
            ("campose_valid", self._valid),
        ):
            if arr.shape[0] != n:
                raise ValueError(f"{self.name}: {key} has {arr.shape[0]} rows, img_frame_ids has {n}")

    def __len__(self) -> int:
        return len(self._frame_ids)

    def __getitem__(self, i: int) -> dict[str, Any]:
        if not -len(self) <= i < len(self):
            raise IndexError(i)
        return {
            "smpl_pose": self._poses[i],  # (72,)
            "smpl_betas": self._betas,  # (10,)
            "trans": self._trans[i],  # (3,)
            "joints_3d": self._joints[i],  # (24, 3) world frame
            "keypoints_2d": self._kp2d[i],  # (18, 3) x, y, confidence
            "cam_extrinsic": self._cam[i],  # (4, 4)
            "campose_valid": bool(self._valid[i]),
            "sequence_name": self.name,
            "frame_idx": int(self._frame_ids[i]),
        }

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for i in range(len(self)):
            yield self[i]


class ThreeDPWLoader:
    """All test sequences under ``<data_root>/datasets/3dpw/sequenceFiles/test/``, sorted by file name."""

    def __init__(self, config: str | Path | None = None, person: int = 0) -> None:
        if config is None:
            from hready.data.amass import load_paths_config

            cfg = load_paths_config()
        else:
            cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
        self.root = Path(cfg["data_root"]) / "datasets" / "3dpw" / "sequenceFiles" / "test"
        self.files = sorted(self.root.glob("*.pkl"))
        if not self.files:
            raise FileNotFoundError(f"no 3DPW test pkl files under {self.root}")
        self.person = person

    def __len__(self) -> int:
        return len(self.files)

    def __iter__(self) -> Iterator[ThreeDPWSequence]:
        for f in self.files:
            yield ThreeDPWSequence(f, self.person)


def smpl_to_smplx_pose(smpl_pose_aa: np.ndarray) -> np.ndarray:
    """SMPL 72-dim axis-angle -> SMPL-X 165-dim axis-angle (stub).

    Copies the 22 shared joints (global orientation + 21 body joints = the first 66 values) and zero-fills the
    rest (jaw, leye, reye, both hands). SMPL joints 22 and 23 (hands) are dropped.

    Full SMPL->SMPL-X transfer requires model_transfer from smplx extras; this stub is sufficient for pose-error
    evaluation on the 14-joint common subset.
    """
    pose = np.asarray(smpl_pose_aa)
    if pose.shape != (N_SMPL_JOINTS * 3,):
        raise ValueError(f"expected SMPL pose of shape (72,), got {pose.shape}")
    out = np.zeros(SMPLX_POSE_DIM, dtype=pose.dtype)
    out[: N_SHARED_JOINTS * 3] = pose[: N_SHARED_JOINTS * 3]
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="3DPW test-set loader summary")
    p.add_argument("--config", type=Path, default=Path("configs/paths.yaml"))
    a = p.parse_args(argv)
    loader = ThreeDPWLoader(a.config)
    n_seq, total, first = 0, 0, None
    for seq in loader:
        n_seq += 1
        total += len(seq)
        if first is None:
            first = (seq.name, len(seq))
    print(f"sequences: {n_seq}")
    print(f"total frames (person 0): {total}")
    print(f"first sequence: {first[0]} ({first[1]} frames)")


if __name__ == "__main__":
    main()
