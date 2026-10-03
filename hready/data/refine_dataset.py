"""AMASS window dataset for HR-Refine (train split filters)."""

from __future__ import annotations

import random
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from hready.data.amass import (
    AmassIndexEntry,
    assign_split,
    clip_flags,
    load_clip,
    load_index,
)
from hready.data.foot_height_rise import load_foot_height_rise_cache

SplitName = Literal["train", "val", "test"]
WINDOW = 64
_ELIGIBLE_CACHE: dict[SplitName, list[AmassIndexEntry]] = {}


def _eligible_entries(split: SplitName) -> list[AmassIndexEntry]:
    if split in _ELIGIBLE_CACHE:
        return _ELIGIBLE_CACHE[split]
    rise_cache = load_foot_height_rise_cache()
    rise_entries = rise_cache.get("entries", {})
    out: list[AmassIndexEntry] = []
    for e in load_index():
        if assign_split(e) != split:
            continue
        flags = clip_flags(e)
        if flags.get("skate_flag"):
            continue
        rec = rise_entries.get(e.rel_path)
        if rec and rec.get("status") == "assessable" and float(rec.get("rise_cm", 0)) > 5.0:
            continue
        out.append(e)
    _ELIGIBLE_CACHE[split] = out
    return out


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
            i1 = i0 + self.window
            root = clip["root_orient"][i0:i1]
            body = clip["pose_body"][i0:i1]
            transl = clip["transl"][i0:i1]
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
    return {s: len(_eligible_entries(s)) for s in ("train", "val", "test")}
