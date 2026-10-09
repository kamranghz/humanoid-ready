"""Descriptive BABEL frame_ann gait statistics (not a contact regression test)."""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from hready.data.amass import cache_dir_from_config, clip_flags, load_index
from hready.data.babel import (
    _entries_by_rel,
    act_cat_matches_keyword,
    ann_source_for_segmentation,
    load_babel_index_payload,
)
from hready.data.contact import (
    CONTACT_H_OFF_M,
    CONTACT_H_ON_M,
    CONTACT_V_OFF_M_S,
    CONTACT_V_ON_M_S,
    compute_foot_contact_from_positions,
    load_foot_positions,
)

_BABEL_GAIT_MEDIAN_CACHE = "babel_gait_medians.json"


def _gait_stats_from_contact_slice(contact: np.ndarray) -> dict[str, float]:
    sl = np.asarray(contact, dtype=bool)
    lf = sl[:, 0] | sl[:, 1]
    rf = sl[:, 2] | sl[:, 3]
    return {
        "L_pct": 100.0 * float(lf.mean()),
        "R_pct": 100.0 * float(rf.mean()),
        "dbl_pct": 100.0 * float((lf & rf).mean()),
        "alt_pct": 100.0 * float((lf ^ rf).mean()),
    }


def babel_frame_ann_gait_medians(
    keyword: str,
    *,
    h_on: float = CONTACT_H_ON_M,
    v_on: float = CONTACT_V_ON_M_S,
    cache_dir: Path | None = None,
    use_cache: bool = True,
) -> dict[str, float]:
    """Median per-segment gait metrics over BABEL ``frame_ann`` non-excluded segments."""
    cache_dir = (cache_dir or cache_dir_from_config()).resolve()
    cache_path = cache_dir / _BABEL_GAIT_MEDIAN_CACHE
    if use_cache and cache_path.is_file():
        blob = json.loads(cache_path.read_text(encoding="utf-8"))
        key = f"{keyword}|h_on={h_on}|v_on={v_on}"
        if key in blob.get("entries", {}):
            return {k: float(v) for k, v in blob["entries"][key].items()}
    indexed = {e.rel_path: e for e in load_index(cache_dir)}
    entries = ann_source_for_segmentation(
        list(_entries_by_rel(load_babel_index_payload()).values())
    )
    by_rel: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in entries:
        ent = indexed.get(e.rel_path)
        if ent is None or clip_flags(ent, cache_dir=cache_dir).get("exclude_contact"):
            continue
        for seg in e.segments:
            cats = seg.get("act_cat") or []
            if any(act_cat_matches_keyword(c, keyword) for c in cats):
                by_rel[e.rel_path].append(seg)
    vals: list[dict[str, float]] = []
    for rel, segs in by_rel.items():
        try:
            pos = load_foot_positions(indexed[rel], cache_dir=cache_dir)
        except FileNotFoundError:
            continue
        contact = compute_foot_contact_from_positions(
            pos, h_on=h_on, v_on=v_on, h_off=CONTACT_H_OFF_M, v_off=CONTACT_V_OFF_M_S
        )
        t_len = pos.shape[0]
        for seg in segs:
            i0 = int(float(seg["start_t"]) * 30.0)
            i1 = int(float(seg["end_t"]) * 30.0)
            i0, i1 = max(0, i0), min(t_len, i1)
            if i1 <= i0:
                continue
            vals.append(_gait_stats_from_contact_slice(contact[i0:i1]))
    if not vals:
        raise RuntimeError(f"No BABEL frame_ann segments for keyword={keyword!r}")
    out = {
        "n_segments": float(len(vals)),
        "L_pct": float(st.median([v["L_pct"] for v in vals])),
        "R_pct": float(st.median([v["R_pct"] for v in vals])),
        "dbl_pct": float(st.median([v["dbl_pct"] for v in vals])),
        "alt_pct": float(st.median([v["alt_pct"] for v in vals])),
    }
    if use_cache:
        blob: dict[str, Any] = {"version": 1, "entries": {}}
        if cache_path.is_file():
            blob = json.loads(cache_path.read_text(encoding="utf-8"))
        key = f"{keyword}|h_on={h_on}|v_on={v_on}"
        blob.setdefault("entries", {})[key] = out
        cache_path.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="BABEL frame_ann descriptive gait medians")
    p.add_argument("keyword", default="walk", nargs="?")
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args()
    out = babel_frame_ann_gait_medians(args.keyword, use_cache=not args.no_cache)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
