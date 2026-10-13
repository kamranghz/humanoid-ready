"""Window dataset for the HR-Refine auxiliary heads: foot contact, BABEL action and synthetic IMU.

Reads the ``grounded_motion_30hz`` memmap cache (see ``motion_cache.py``). Per window it returns the
refiner inputs of ``refine_dataset.py`` plus:

- ``contact`` (W, 4): foot-channel labels from the cache (L heel, L toe, R heel, R toe), float 0/1;
- ``action`` (W,): class index into the BABEL vocabulary, ``-100`` where the frame has no label;
- ``imu`` (W, 18): kinematic accelerometer, see :func:`window_specific_force`.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from hready.data.amass import AmassIndexEntry
from hready.data.babel import load_babel_index_payload
from hready.data.imu import DEFAULT_SENSOR_JOINT_IDS
from hready.data.motion_cache import load_clip_memmap
from hready.data.refine_dataset import MOTION_CACHE_DIR, SplitName, _eligible_entries

IGNORE_INDEX = -100
IMU_SCALE = 10.0  # m/s^2; features are divided by this so they are O(1)
_GRAVITY_Z = -9.81


def action_vocabulary() -> list[str]:
    """BABEL action categories in the cached index order (class index = position)."""
    return list(load_babel_index_payload()["meta"]["vocabulary"]["vocabulary"])


def _segments_by_rel() -> dict[str, list[tuple[float, float, list[str]]]]:
    out: dict[str, list[tuple[float, float, list[str]]]] = {}
    for rec in load_babel_index_payload()["entries"]:
        segs = [(float(s["start_t"]), float(s["end_t"]), list(s["act_cat"])) for s in rec["segments"]]
        if segs:
            out[rec["rel_path"]] = segs
    return out


def window_specific_force(joints: np.ndarray, i0: int, window: int, fps: float) -> np.ndarray:
    """World-axis specific force (acceleration minus gravity) at the 6 IMU joints, (W, 6, 3).

    ``joints`` is the cached (T, 55, 3) grounded joint array. Acceleration is the second central
    difference of joint position (one extra frame of context on each side where available, edge frames
    repeated). Unlike ``imu.synthetic_imu_from_clip`` there is no sensor-frame rotation and no
    gyroscope, because the cache holds positions but not global joint rotations.
    """
    t = joints.shape[0]
    idx = np.clip(np.arange(i0 - 1, i0 + window + 1), 0, t - 1)
    pos = np.asarray(joints[idx][:, list(DEFAULT_SENSOR_JOINT_IDS), :], dtype=np.float64)
    acc = (pos[2:] - 2.0 * pos[1:-1] + pos[:-2]) * fps * fps
    acc[..., 2] -= _GRAVITY_Z
    return acc


class AuxiliaryWindowDataset(Dataset):
    """Random (training) or fixed-plan (evaluation) windows with contact, action and IMU targets."""

    def __init__(
        self,
        split: SplitName,
        window: int = 64,
        *,
        seed: int = 0,
        labeled_fraction: float = 0.0,
        imu_noise_std: float = 0.0,
        plan_windows: int | None = None,
        only_labeled: bool = False,
    ) -> None:
        self.window = int(window)
        self.seed = int(seed)
        self.imu_noise_std = float(imu_noise_std)
        self.labeled_fraction = float(labeled_fraction)
        self.vocab = action_vocabulary()
        self._cat_to_class = {c: i for i, c in enumerate(self.vocab)}
        self._segments = _segments_by_rel()
        entries = _eligible_entries(split)
        self.entries: list[AmassIndexEntry] = entries
        self.labeled = [e for e in entries if e.rel_path in self._segments]
        if not self.entries:
            raise RuntimeError(f"No eligible clips for split={split}")
        if only_labeled and not self.labeled:
            raise RuntimeError(f"No BABEL-labelled clips for split={split}")
        self._rng = random.Random(self.seed)
        self._plan: list[tuple[AmassIndexEntry, int]] | None = None
        if plan_windows is not None:
            pool = self.labeled if only_labeled else self.entries
            rng = random.Random(self.seed)
            plan = []
            for _ in range(int(plan_windows)):
                entry = pool[rng.randrange(len(pool))]
                n = int(load_clip_memmap(entry.rel_path, rel_dir=MOTION_CACHE_DIR)["T"])
                plan.append((entry, rng.randint(0, max(0, n - self.window))))
            self._plan = plan

    def __len__(self) -> int:
        return len(self._plan) if self._plan is not None else len(self.entries) * 4

    def _action_labels(self, rel_path: str, frame_idx: np.ndarray, n_frames: int, fps: float) -> np.ndarray:
        """Per frame, the lowest class index (most frequent BABEL category) of the segments covering it."""
        none = np.iinfo(np.int64).max
        labels = np.full(self.window, none, dtype=np.int64)
        for start_t, end_t, cats in self._segments.get(rel_path, ()):
            a = max(0, min(int(np.floor(start_t * fps)), n_frames))
            b = max(0, min(int(np.floor(end_t * fps)), n_frames))
            classes = [self._cat_to_class[c] for c in cats if c in self._cat_to_class]
            if b <= a or not classes:
                continue
            hit = (frame_idx >= a) & (frame_idx < b)
            labels[hit] = np.minimum(labels[hit], min(classes))
        labels[labels == none] = IGNORE_INDEX
        return labels

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self._plan is not None:
            entry, i0 = self._plan[idx]
            noise_rng = np.random.default_rng(self.seed * 1_000_003 + idx)
        else:
            pool = self.labeled if (self.labeled and self._rng.random() < self.labeled_fraction) else self.entries
            entry = pool[self._rng.randrange(len(pool))]
            i0 = None
            noise_rng = np.random.default_rng(self._rng.randrange(2**31))
        clip = load_clip_memmap(entry.rel_path, rel_dir=MOTION_CACHE_DIR)
        n = int(clip["T"])
        fps = float(clip["fps"])
        if i0 is None:
            i0 = self._rng.randint(0, max(0, n - self.window))
        raw = np.arange(i0, i0 + self.window)
        frames = np.minimum(raw, n - 1)
        imu = window_specific_force(clip["joints_55"], i0, self.window, fps)
        if self.imu_noise_std > 0.0:
            imu = imu + noise_rng.normal(0.0, self.imu_noise_std, imu.shape)
        action = self._action_labels(entry.rel_path, raw, n, fps)
        action[raw >= n] = IGNORE_INDEX
        return {
            "root_orient": torch.as_tensor(np.array(clip["root_orient"][frames]), dtype=torch.float32),
            "pose_body": torch.as_tensor(np.array(clip["pose_body"][frames]).reshape(-1, 21, 3), dtype=torch.float32),
            "transl": torch.as_tensor(np.array(clip["transl"][frames]), dtype=torch.float32),
            "betas": torch.as_tensor(np.array(clip["betas"]), dtype=torch.float32),
            "contact": torch.as_tensor(np.array(clip["contact"][frames]), dtype=torch.float32),
            "action": torch.as_tensor(action),
            "imu": torch.as_tensor((imu / IMU_SCALE).reshape(self.window, -1), dtype=torch.float32),
            "fps": fps,
        }


def collate_auxiliary(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack windows; ``fps`` stays a scalar."""
    out: dict[str, Any] = {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0] if k != "fps"}
    out["fps"] = batch[0]["fps"]
    return out
