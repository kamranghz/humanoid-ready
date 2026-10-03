"""Foot-height rise detector (floor-level inconsistency) from ``foot_traj`` cache."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from hready.data.amass import AmassIndexEntry, cache_dir_from_config, load_index
from hready.data.contact import _foot_traj_path

_FOOT_HEIGHT_RISE_JSON = "amass_foot_height_rise.json"
_TARGET_FPS = 30.0
_WIN_FRAMES = 30
_MIN_DURATION_S = 2.0
_NEAR_FLOOR_M = 0.05


def foot_height_rise_cm_from_positions(positions: np.ndarray) -> tuple[float | None, str]:
    """Return (rise_cm, status) with status assessable|short|no_near_floor."""
    pos = np.asarray(positions, dtype=np.float64)
    if pos.shape[0] < int(_MIN_DURATION_S * _TARGET_FPS):
        return None, "short"
    z_min = pos[:, :, 2].min(axis=1)
    if float(z_min.min()) > _NEAR_FLOOR_M:
        return None, "no_near_floor"
    if z_min.size < _WIN_FRAMES:
        return None, "short"
    roll = np.array([z_min[i : i + _WIN_FRAMES].min() for i in range(z_min.size - _WIN_FRAMES + 1)])
    return float((roll.max() - roll.min()) * 100.0), "assessable"


def _cache_path(cache_dir: Path) -> Path:
    return cache_dir / _FOOT_HEIGHT_RISE_JSON


def build_foot_height_rise_cache(
    *,
    cache_dir: Path | None = None,
    entries: list[AmassIndexEntry] | None = None,
) -> dict[str, Any]:
    cache_dir = (cache_dir or cache_dir_from_config()).resolve()
    entries = entries or load_index(cache_dir)
    t0 = time.perf_counter()
    out_entries: dict[str, dict[str, Any]] = {}
    missing = 0
    for i, entry in enumerate(entries):
        path = _foot_traj_path(cache_dir, entry.rel_path)
        if not path.is_file():
            missing += 1
            continue
        with np.load(path) as data:
            pos = data["positions"]
        rise_cm, status = foot_height_rise_cm_from_positions(pos)
        out_entries[entry.rel_path] = {
            "subset": entry.subset,
            "rise_cm": rise_cm,
            "status": status,
        }
        if (i + 1) % 2000 == 0:
            print(f"  foot_height_rise: {i + 1}/{len(entries)}", flush=True)
    payload = {
        "version": 1,
        "metric": "foot_height_rise_cm",
        "definition": (
            "Foot-height rise (floor-level inconsistency): 100*(max roll_min_1s - min roll_min_1s). "
            "Cannot separate capture drift from stairs/platforms/handstands; upper bound only."
        ),
        "n_index_clips": len(entries),
        "n_with_foot_traj": len(out_entries),
        "n_missing_foot_traj": missing,
        "wall_s": time.perf_counter() - t0,
        "entries": out_entries,
    }
    _cache_path(cache_dir).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_foot_height_rise_cache(cache_dir: Path | None = None) -> dict[str, Any]:
    cache_dir = (cache_dir or cache_dir_from_config()).resolve()
    path = _cache_path(cache_dir)
    legacy = cache_dir / "amass_foot_height_drift.json"
    if not path.is_file() and legacy.is_file():
        data = json.loads(legacy.read_text(encoding="utf-8"))
        data["metric"] = "foot_height_rise_cm"
        for row in data.get("entries", {}).values():
            if "drift_cm" in row and "rise_cm" not in row:
                row["rise_cm"] = row.pop("drift_cm")
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return data
    if not path.is_file():
        return build_foot_height_rise_cache(cache_dir=cache_dir)
    return json.loads(path.read_text(encoding="utf-8"))


def foot_height_rise_summary(cache: dict[str, Any] | None = None) -> dict[str, int]:
    cache = cache or load_foot_height_rise_cache()
    entries = cache["entries"]
    gt3 = gt5 = na = 0
    for e in entries.values():
        if e["status"] != "assessable":
            na += 1
            continue
        r = float(e["rise_cm"])
        if r > 3.0:
            gt3 += 1
        if r > 5.0:
            gt5 += 1
    return {
        "n_entries": len(entries),
        "not_assessable": na,
        "gt_3cm": gt3,
        "gt_5cm": gt5,
    }
