"""Cohort frame masks, reproducing the recorded subject-split cohort definitions (``hready.data.ego_splits``).

- ``ordinary_locomotion``: BABEL segments >= ``min_segment_duration_s`` whose first matching cohort
  (cohort order: floor_work, ordinary_locomotion, other_labelled) is ordinary_locomotion.
- ``floor_work_eligible`` (kneel + lie), ``sit_floor``, ``sit_support``: confirmed segments in
  ``results/splits/floor_work_clips.csv``.
Segment -> frame indices use the split module's ``_segment_frame_indices`` at 30 Hz.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from hready.data.amass import load_index
from hready.data.babel import BabelIndexEntry, load_babel_index_payload
from hready.data.completion_clips import FLOOR_WORK_ELIGIBLE_CLASSES
from hready.data.ego_splits import _collect_babel_proposals, _segment_frame_indices

COHORTS: tuple[str, ...] = (
    "all",
    "ordinary_locomotion",
    "floor_work_eligible",
    "sit_floor",
    "sit_support",
)


class CohortMasks:
    def __init__(self, ego_splits_yaml: Path, floor_csv: Path, rels: list[str]) -> None:
        cfg = yaml.safe_load(Path(ego_splits_yaml).read_text(encoding="utf-8"))
        self.fps = float(cfg.get("fps", 30.0))
        cohort_kw = {
            "floor_work": dict(cfg["cohorts"]["floor_work_babel_keywords"]),
            "ordinary_locomotion": dict(
                cfg["cohorts"]["ordinary_locomotion_babel_keywords"]
            ),
            "other_labelled": dict(cfg["cohorts"]["other_labelled_babel_keywords"]),
        }
        min_dur = float(cfg["floor_work"]["min_segment_duration_s"])
        babel_by_rel: dict[str, BabelIndexEntry] = {}
        for row in load_babel_index_payload()["entries"]:
            row = dict(row)
            if "playback_duration" not in row:
                bd = float(row.get("babel_dur", 0.0))
                row.setdefault(
                    "playback_duration", float(row.get("mocap_time_length", bd))
                )
            row.pop("mocap_time_length", None)
            row.pop("time_scale", None)
            babel_by_rel[row["rel_path"]] = BabelIndexEntry(**row)
        amass_by_rel = {e.rel_path: e for e in load_index()}
        props = _collect_babel_proposals(
            babel_by_rel, set(rels), amass_by_rel, cohort_kw, min_dur
        )
        self._segments: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for p in props:
            if p.cohort == "ordinary_locomotion":
                self._segments[p.rel_path]["ordinary_locomotion"].append(
                    (p.start_s, p.end_s)
                )
        with Path(floor_csv).open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                g = r["geometry_class"]
                seg = (float(r["segment_start_s"]), float(r["segment_end_s"]))
                if g in FLOOR_WORK_ELIGIBLE_CLASSES:
                    self._segments[r["rel_path"]]["floor_work_eligible"].append(seg)
                elif g in ("sit_floor", "sit_support"):
                    self._segments[r["rel_path"]][g].append(seg)

    def mask(self, rel: str, t_len: int, cohort: str) -> np.ndarray:
        if cohort == "all":
            return np.ones(t_len, dtype=bool)
        m = np.zeros(t_len, dtype=bool)
        for s, e in self._segments.get(rel, {}).get(cohort, []):
            idx = _segment_frame_indices(self.fps, s, e, t_len)
            m[idx.start : idx.stop] = True
        return m
