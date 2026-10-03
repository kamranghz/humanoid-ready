"""AMASS window dataset for HR-Refine (train split filters)."""

from __future__ import annotations

import random
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from hready.data.amass import AmassIndexEntry, load_clip
from hready.data.refine_eligible_cache import eligible_entries
from hready.data.refine_eligible_cache import split_counts as cached_split_counts

SplitName = Literal["train", "val", "test"]
WINDOW = 64


def _eligible_entries(split: SplitName) -> list[AmassIndexEntry]:
    return eligible_entries(split)


class HRRefineWindowDataset(Dataset):
    """Random ``window``-frame slices from eligible AMASS clips."""

    def __init__(
        self,
        split: SplitName = "train",
        window: int = WINDOW,
        *,
        entries: list[AmassIndexEntry] | None = None,
        seed: int = 0,
    ) -> None:
        self.window = int(window)
        self.entries = entries if entries is not None else _eligible_entries(split)
        self._rng = random.Random(seed)
        if not self.entries:
            raise RuntimeError(f"No eligible clips for split={split}")

    def __len__(self) -> int:
        return len(self.entries) * 4

    def __getitem__(self, idx: int) -> dict[str, Any]:
        entry = self.entries[idx % len(self.entries)]
        clip = load_clip(entry, ground=True)
        t = clip["root_orient"].shape[0]
        if t >= self.window:
            i0 = self._rng.randint(0, t - self.window)
            root = clip["root_orient"][i0 : i0 + self.window]
            body = clip["pose_body"][i0 : i0 + self.window]
            transl = clip["transl"][i0 : i0 + self.window]
        else:
            root = clip["root_orient"]
            body = clip["pose_body"]
            transl = clip["transl"]
            pad = self.window - t
            root = np.concatenate([root, np.repeat(root[-1:], pad, axis=0)], axis=0)
            body = np.concatenate([body, np.repeat(body[-1:], pad, axis=0)], axis=0)
            transl = np.concatenate([transl, np.repeat(transl[-1:], pad, axis=0)], axis=0)
        return {
            "rel_path": entry.rel_path,
            "root_orient": torch.as_tensor(root, dtype=torch.float32),
            "pose_body": torch.as_tensor(body, dtype=torch.float32),
            "transl": torch.as_tensor(transl, dtype=torch.float32),
            "betas": torch.as_tensor(clip["betas"], dtype=torch.float32),
            "fps": float(clip["fps"]),
        }


def collate_windows(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    betas = torch.stack([b["betas"] for b in batch], dim=0)
    return {
        "root_orient": torch.stack([b["root_orient"] for b in batch], dim=0),
        "pose_body": torch.stack([b["pose_body"] for b in batch], dim=0),
        "transl": torch.stack([b["transl"] for b in batch], dim=0),
        "betas": betas,
        "fps": batch[0]["fps"],
    }


def split_counts() -> dict[str, int]:
    return cached_split_counts()
