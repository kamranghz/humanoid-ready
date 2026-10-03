"""Disk cache for HR-Refine eligible AMASS clips (skate + foot-height filters)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Literal

from hready.data.amass import (
    AmassIndexEntry,
    assign_split,
    cache_dir_from_config,
    clip_flags,
    load_index,
)
from hready.data.foot_height_rise import load_foot_height_rise_cache

SplitName = Literal["train", "val", "test"]
_CACHE_JSON = "hr_refine_eligible_index.json"
_RISE_JSON = "amass_foot_height_rise.json"
_FLAGS_JSON = "amass_clip_flags.json"
_INDEX_JSON = "amass_index.json"


def _fingerprint(cache_dir: Path) -> dict[str, Any]:
    def _stat(name: str) -> dict[str, Any]:
        p = cache_dir / name
        if not p.is_file():
            return {"path": name, "exists": False}
        st = p.stat()
        return {"path": name, "exists": True, "mtime_ns": st.st_mtime_ns, "size": st.st_size}

    return {
        "clip_flags": _stat(_FLAGS_JSON),
        "foot_height_rise": _stat(_RISE_JSON),
        "index": _stat(_INDEX_JSON),
    }


def _build_split(split: SplitName, rise_entries: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for e in load_index():
        if assign_split(e) != split:
            continue
        flags = clip_flags(e)
        if flags.get("skate_flag"):
            continue
        rec = rise_entries.get(e.rel_path)
        if rec and rec.get("status") == "assessable" and float(rec.get("rise_cm", 0)) > 5.0:
            continue
        out.append(e.rel_path)
    return sorted(out)


def build_eligible_cache(cache_dir: Path | None = None) -> dict[str, Any]:
    cache_dir = (cache_dir or cache_dir_from_config()).resolve()
    t0 = time.perf_counter()
    rise = load_foot_height_rise_cache(cache_dir)
    rise_entries = rise.get("entries", {})
    splits = {s: _build_split(s, rise_entries) for s in ("train", "val", "test")}
    payload = {
        "version": 1,
        "built_wall_s": time.perf_counter() - t0,
        "fingerprints": _fingerprint(cache_dir),
        "splits": splits,
    }
    path = cache_dir / _CACHE_JSON
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_eligible_cache(cache_dir: Path | None = None) -> dict[str, Any]:
    cache_dir = (cache_dir or cache_dir_from_config()).resolve()
    path = cache_dir / _CACHE_JSON
    fp_now = _fingerprint(cache_dir)
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("fingerprints") == fp_now:
            return data
    return build_eligible_cache(cache_dir)


def eligible_rel_paths(split: SplitName, cache_dir: Path | None = None) -> list[str]:
    data = load_eligible_cache(cache_dir)
    return list(data["splits"][split])


def eligible_entries(split: SplitName, cache_dir: Path | None = None) -> list[AmassIndexEntry]:
    rels = set(eligible_rel_paths(split, cache_dir))
    return [e for e in load_index() if e.rel_path in rels]


def split_counts(cache_dir: Path | None = None) -> dict[str, int]:
    data = load_eligible_cache(cache_dir)
    return {s: len(data["splits"][s]) for s in ("train", "val", "test")}
