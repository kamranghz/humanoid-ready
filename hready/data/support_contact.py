"""Support-contact labels v2 (Track E1b): LBS-region lowest-vertex contact beside foot channels."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

from hready.data.amass import (
    AmassIndexEntry,
    _entry_by_rel,
    _load_npz_raw,
    _resample_axis_angle_series,
    _target_frame_count,
    amass_root_from_config,
    assign_split,
    cache_dir_from_config,
    clip_flags,
    floor_offset,
    load_index,
    resample_translation,
)
from hready.data.babel import act_cat_matches_keyword, load_babel_index_payload
from hready.data.contact import (
    CONTACT_H_OFF_M,
    CONTACT_H_ON_M,
    CONTACT_V_OFF_M_S,
    CONTACT_V_ON_M_S,
    _CONTACT_MIN_RUN,
    compute_foot_contact_from_positions,
    load_foot_positions,
    vertex_contact,
)

_TARGET_FPS = 30.0
_NUM_MESH_VERTS = 10475
_FLOOR_UNCERTAIN_MEDIAN_M = 0.07

REGION_JOINT_GROUPS: dict[str, tuple[int, ...]] = {
    "shins": (4, 5),
    "thighs": (1, 2),
    "pelvis_seat": (0,),
    "back_torso": (3, 6, 9),
    "head": (12, 15),
    "forearms": (18, 19),
    "wrists": (20, 21),
    "hands": tuple(range(25, 55)),
}

NON_FOOT_REGION_NAMES: tuple[str, ...] = tuple(REGION_JOINT_GROUPS.keys())
ALL_REGION_NAMES: tuple[str, ...] = NON_FOOT_REGION_NAMES + ("feet",)

_STAND_EXCLUDE_KEYWORDS: tuple[str, ...] = (
    "stand up",
    "standup",
    "transition",
    "sitdown",
    "sit down",
)


@dataclass(frozen=True)
class SegmentRow:
    rel_path: str
    subset: str
    geometry_class: str
    start_s: float
    end_s: float


class FkClipCache:
    """One grounded FK + foot_traj load per clip per run."""

    def __init__(self, body: Any, device: Optional[str]) -> None:
        self._body = body
        self._device = device
        self._verts: dict[str, np.ndarray] = {}
        self._foot: dict[str, Optional[np.ndarray]] = {}

    def grounded_vertices(self, entry: AmassIndexEntry) -> np.ndarray:
        rel = entry.rel_path
        if rel not in self._verts:
            floor_off = float(floor_offset(entry)["floor_offset"])
            self._verts[rel] = _forward_grounded_vertices(
                entry, floor_off, body=self._body, device=self._device
            )
            cache_dir = cache_dir_from_config()
            try:
                self._foot[rel] = load_foot_positions(entry, cache_dir=cache_dir)
            except FileNotFoundError:
                self._foot[rel] = None
        return self._verts[rel]

    def foot_positions(self, entry: AmassIndexEntry) -> Optional[np.ndarray]:
        self.grounded_vertices(entry)
        return self._foot.get(entry.rel_path)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _save_yaml(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)


def _babel_category_from_segment(
    act_cats: list[str], keyword_map: dict[str, str]
) -> tuple[str, str] | None:
    for kw, cat in keyword_map.items():
        for ac in act_cats:
            if act_cat_matches_keyword(ac, kw):
                return cat, kw
    return None


def lbs_argmax_joint_ids(body: Any) -> np.ndarray:
    w = body._model.lbs_weights.detach().cpu().numpy()[:_NUM_MESH_VERTS]
    return w.argmax(axis=1).astype(np.int64)


def resolve_region_vertex_sets(body: Any) -> dict[str, np.ndarray]:
    j_id = lbs_argmax_joint_ids(body)
    out: dict[str, np.ndarray] = {}
    for name, joints in REGION_JOINT_GROUPS.items():
        out[name] = np.where(np.isin(j_id, np.asarray(joints, dtype=np.int64)))[0].astype(
            np.int64
        )
    return out


def print_region_inventory(body: Any, device: Optional[str] = None) -> dict[str, Any]:
    import torch
    from hready.data.amass import _resolve_torch_device

    dev = _resolve_torch_device(device)
    out_mesh = body.forward(
        torch.zeros(1, 3, device=dev),
        torch.zeros(1, 63, device=dev),
        torch.zeros(1, 16, device=dev),
        torch.zeros(1, 3, device=dev),
    )
    v = out_mesh.vertices[0].detach().cpu().numpy()
    joints = out_mesh.joints[0].detach().cpu().numpy()
    sets = resolve_region_vertex_sets(body)
    report: dict[str, Any] = {}
    print("REGION_VERTEX_INVENTORY (max-LBS-weight assignment)")
    for name, joint_ids in REGION_JOINT_GROUPS.items():
        idx = sets[name]
        jlist = list(joint_ids)
        vz = v[idx, 2]
        jz = [float(joints[j, 2]) for j in jlist]
        rec = {
            "joints": jlist,
            "n_vertices": int(idx.size),
            "neutral_vert_z_min_m": float(vz.min()) if idx.size else None,
            "neutral_vert_z_median_m": float(np.median(vz)) if idx.size else None,
            "neutral_joint_z_m": jz,
        }
        report[name] = rec
        print(
            f"  {name}: joints={jlist} n_verts={idx.size} "
            f"vert_z[min,med]={rec['neutral_vert_z_min_m']}, {rec['neutral_vert_z_median_m']} "
            f"joint_z={jz}"
        )
    return report


def _frame_range(n_frames: int, fps: float, start_s: float, end_s: float) -> slice:
    i0 = max(0, int(math.floor(start_s * fps)))
    i1 = min(n_frames, int(math.ceil(end_s * fps)))
    if i1 <= i0:
        i1 = min(n_frames, i0 + 1)
    return slice(i0, i1)


def _forward_grounded_vertices(
    entry: AmassIndexEntry,
    floor_off: float,
    *,
    device: Optional[str] = None,
    body: Optional[Any] = None,
    frame_slice: Optional[slice] = None,
) -> np.ndarray:
    import torch
    from hready.body.smplx_wrapper import load_body
    from hready.data.amass import _resolve_torch_device

    amass_root = amass_root_from_config().resolve()
    raw = _load_npz_raw(amass_root / entry.rel_path)
    fps_src = float(entry.fps)
    n_src = int(raw["root_orient"].shape[0])
    root = _resample_axis_angle_series(
        raw["root_orient"].reshape(-1, 1, 3), fps_src, _TARGET_FPS
    ).reshape(-1, 3)
    body_aa = _resample_axis_angle_series(
        raw["pose_body"].reshape(-1, 21, 3), fps_src, _TARGET_FPS
    )
    transl = resample_translation(raw["trans"], fps_src, _TARGET_FPS)

    if frame_slice is not None:
        root = root[frame_slice]
        body_aa = body_aa[frame_slice]
        transl = transl[frame_slice]

    n_tgt = root.shape[0]
    dev = _resolve_torch_device(device)
    dtype = torch.float32
    if body is None:
        body = load_body("locked_head")
        body._model.to(dev)
    parts: list[np.ndarray] = []
    chunk = 128
    for i0 in range(0, n_tgt, chunk):
        i1 = min(n_tgt, i0 + chunk)
        go = torch.as_tensor(root[i0:i1], device=dev, dtype=dtype)
        bp = torch.as_tensor(body_aa[i0:i1], device=dev, dtype=dtype).reshape(i1 - i0, 63)
        tr = torch.as_tensor(transl[i0:i1], device=dev, dtype=dtype)
        be = (
            torch.as_tensor(raw["betas"], device=dev, dtype=dtype)
            .unsqueeze(0)
            .expand(i1 - i0, -1)
        )
        with torch.inference_mode():
            verts = body.forward(go, bp, be, tr).vertices.detach().cpu().numpy()
        verts[:, :, 2] -= floor_off
        parts.append(verts.astype(np.float64))
    return np.concatenate(parts, axis=0)


def lowest_vertex_argmin_indices(verts: np.ndarray, region_ids: np.ndarray) -> np.ndarray:
    sub = verts[:, region_ids, :]
    return sub[:, :, 2].argmin(axis=1).astype(np.int64)


_LEGACY_NON_FOOT_SPEED = "argmin_vertex_horiz_speed"


def lowest_vertex_height_speed(
    verts: np.ndarray, region_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Legacy non-foot speed: horizontal speed of the lowest vertex (argmin switches)."""
    sub = verts[:, region_ids, :]
    z = sub[:, :, 2]
    j = z.argmin(axis=1)
    t = np.arange(verts.shape[0])
    pts = sub[t, j, :]
    h = pts[:, 2].copy()
    xy = pts[:, :2]
    dt = 1.0 / _TARGET_FPS
    speed = np.zeros(verts.shape[0], dtype=np.float64)
    if verts.shape[0] > 1:
        speed[1:] = np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt
        speed[0] = speed[1]
    return h, speed


def lowest_vertex_height_patch_median_speed(
    verts: np.ndarray, region_ids: np.ndarray, patch_band_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Lowest-vertex height; speed = median same-vertex horiz speed over near-floor patch."""
    sub = verts[:, region_ids, :]
    z = sub[:, :, 2]
    z_min = z.min(axis=1)
    h = z_min.astype(np.float64, copy=True)
    in_patch = z <= (z_min[:, None] + patch_band_m)
    xy = sub[:, :, :2]
    t_len = verts.shape[0]
    speed = np.zeros(t_len, dtype=np.float64)
    if t_len > 1:
        disp = np.linalg.norm(np.diff(xy, axis=0), axis=2) * _TARGET_FPS
        for t in range(1, t_len):
            mask = in_patch[t]
            if mask.any():
                speed[t] = float(np.median(disp[t - 1, mask]))
        speed[0] = speed[1]
    return h, speed


def region_height_speed(
    verts: np.ndarray,
    region_ids: np.ndarray,
    cfg: dict[str, Any],
    *,
    speed_mode: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    mode = speed_mode if speed_mode is not None else str(
        cfg.get("speed_definition", _LEGACY_NON_FOOT_SPEED)
    )
    if mode == "patch_median_same_vertex":
        band = float(cfg.get("patch_band_m", 0.02))
        return lowest_vertex_height_patch_median_speed(verts, region_ids, band)
    return lowest_vertex_height_speed(verts, region_ids)


def segment_duration_s(n_frames: int, start_s: float, end_s: float) -> float:
    sl = _frame_range(n_frames, _TARGET_FPS, start_s, end_s)
    return float(sl.stop - sl.start) / _TARGET_FPS


def contact_runs_inclusive(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive frame index ranges where ``mask`` is True."""
    m = np.asarray(mask, dtype=bool)
    runs: list[tuple[int, int]] = []
    i = 0
    while i < m.size:
        if not m[i]:
            i += 1
            continue
        j = i + 1
        while j < m.size and m[j]:
            j += 1
        runs.append((i, j - 1))
        i = j
    return runs


def _contact_mask_no_speed_gate(
    height: np.ndarray,
    speed: np.ndarray,
    *,
    h_on: float,
    h_off: float,
    min_run: int,
) -> np.ndarray:
    """Same hysteresis/min_run as production, but speed never blocks ON/OFF."""
    big = 1.0e9
    return region_contact_mask(
        height,
        speed,
        h_on=h_on,
        h_off=h_off,
        v_on=big,
        v_off=big,
        min_run=min_run,
    )


def region_contact_mask(
    height: np.ndarray,
    speed: np.ndarray,
    *,
    h_on: float,
    h_off: float,
    v_on: float,
    v_off: float,
    min_run: int,
) -> np.ndarray:
    return vertex_contact(
        height,
        speed,
        h_on=h_on,
        h_off=h_off,
        v_on=v_on,
        v_off=v_off,
        min_run=min_run,
    ).astype(bool)


def foot_contact_mask(positions: np.ndarray, z_shift_m: float = 0.0, **kw: Any) -> np.ndarray:
    pos = positions.copy()
    if z_shift_m != 0.0:
        pos[..., 2] += z_shift_m
    return compute_foot_contact_from_positions(pos, **kw)


def _babel_segments(rel_path: str) -> list[dict[str, Any]]:
    for row in load_babel_index_payload()["entries"]:
        if row.get("rel_path") == rel_path:
            return list(row.get("segments") or [])
    return []


def _segment_has_excluded_stand_label(act_cats: list[str]) -> bool:
    for ac in act_cats:
        low = ac.strip().lower()
        for ex in _STAND_EXCLUDE_KEYWORDS:
            if act_cat_matches_keyword(ac, ex) or ex in low:
                return True
        if "transition" in low:
            return True
    return False


def _pure_stand_segment_heights(
    entry: AmassIndexEntry, pos: np.ndarray
) -> list[float]:
    """Min foot-channel z per frame on BABEL stand segments (excl. stand-up / transition)."""
    vals: list[float] = []
    for seg in _babel_segments(entry.rel_path):
        cats = list(seg.get("act_cat") or [])
        if not any(act_cat_matches_keyword(c, "stand") for c in cats):
            continue
        if _segment_has_excluded_stand_label(cats):
            continue
        sl = _frame_range(pos.shape[0], _TARGET_FPS, float(seg["start_t"]), float(seg["end_t"]))
        block = pos[sl]
        if block.size == 0:
            continue
        zmin = block[:, :, 2].min(axis=1)
        vals.extend(float(x) for x in zmin)
    return vals


def characterize_standing_foot_residual_val() -> dict[str, Any]:
    """Per-clip median stand-segment foot residual; clip-level distributions."""
    cache_dir = cache_dir_from_config()
    index = {e.rel_path: e for e in load_index(cache_dir)}
    per_clip: list[dict[str, Any]] = []
    by_subset_medians: dict[str, list[float]] = defaultdict(list)
    all_medians: list[float] = []

    for rel, entry in sorted(index.items()):
        if assign_split(entry) != "val":
            continue
        try:
            pos = load_foot_positions(entry, cache_dir=cache_dir)
        except FileNotFoundError:
            continue
        frame_vals = _pure_stand_segment_heights(entry, pos)
        if not frame_vals:
            continue
        med = float(np.median(frame_vals))
        stand_labels: list[str] = []
        for seg in _babel_segments(rel):
            cats = list(seg.get("act_cat") or [])
            if any(act_cat_matches_keyword(c, "stand") for c in cats):
                if not _segment_has_excluded_stand_label(cats):
                    stand_labels.extend(cats)
        rec = {
            "rel_path": rel,
            "subset": entry.subset,
            "median_stand_foot_min_channel_m": med,
            "floor_uncertain": med > _FLOOR_UNCERTAIN_MEDIAN_M,
            "babel_stand_act_cat": sorted(set(stand_labels)),
        }
        per_clip.append(rec)
        by_subset_medians[entry.subset].append(med)
        all_medians.append(med)

    def _dist(vals: list[float]) -> dict[str, Any]:
        if not vals:
            return {"n_clips": 0}
        a = np.asarray(vals, dtype=np.float64)
        return {
            "n_clips": int(a.size),
            "p50_m": float(np.median(a)),
            "p95_m": float(np.quantile(a, 0.95)),
            "max_m": float(a.max()),
        }

    return {
        "per_clip": per_clip,
        "overall": _dist(all_medians),
        "per_subset": {k: _dist(v) for k, v in sorted(by_subset_medians.items())},
        "floor_uncertain_threshold_median_m": _FLOOR_UNCERTAIN_MEDIAN_M,
    }


def _loco_keyword_map(cfg: dict[str, Any]) -> dict[str, str]:
    return dict(cfg["ordinary_locomotion_babel_keywords"])


def collect_loco_val_segments(
    cfg: dict[str, Any], *, exclude_rels: set[str]
) -> list[SegmentRow]:
    """BABEL locomotion segments on VAL (>= min duration), ego_splits keyword map."""
    anti = cfg.get("anti_circularity") or {}
    seed = int(anti.get("loco_sample_seed", 0))
    max_seg = int(anti.get("loco_max_segments", 200))
    min_dur = float(cfg.get("min_segment_duration_s", 1.0))
    kw = _loco_keyword_map(cfg)
    index = {e.rel_path: e for e in load_index()}
    pool: list[SegmentRow] = []
    for row in load_babel_index_payload()["entries"]:
        rel = row.get("rel_path") or ""
        if rel not in index or assign_split(index[rel]) != "val":
            continue
        if rel in exclude_rels:
            continue
        entry = index[rel]
        for seg in row.get("segments") or []:
            start_s = float(seg["start_t"])
            end_s = float(seg["end_t"])
            if end_s - start_s < min_dur:
                continue
            cats = list(seg.get("act_cat") or [])
            if _babel_category_from_segment(cats, kw) is None:
                continue
            pool.append(
                SegmentRow(rel, entry.subset, "ordinary_locomotion", start_s, end_s)
            )
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:max_seg]


def derive_loco_region_envelope(
    cfg: dict[str, Any],
    region_sets: dict[str, np.ndarray],
    segments: list[SegmentRow],
    fk: FkClipCache,
    *,
    frame_stride: int,
) -> dict[str, dict[str, float]]:
    pools: dict[str, list[float]] = {n: [] for n in NON_FOOT_REGION_NAMES}
    h_on = float(cfg["foot"]["h_on_m"])
    for seg in segments:
        entry = _entry_by_rel(load_index(), seg.rel_path)
        if clip_flags(entry).get("exclude_contact"):
            continue
        verts_full = fk.grounded_vertices(entry)
        sl = _frame_range(verts_full.shape[0], _TARGET_FPS, seg.start_s, seg.end_s)
        verts = verts_full[sl]
        for fi in range(0, verts.shape[0], frame_stride):
            frame = verts[fi : fi + 1]
            for name, ids in region_sets.items():
                if ids.size == 0:
                    continue
                h, _ = lowest_vertex_height_speed(frame, ids)
                pools[name].append(float(h[0]))
    out: dict[str, dict[str, float]] = {}
    for name, vals in pools.items():
        if not vals:
            out[name] = {"p0.5": None, "p1": None, "p50": None, "n_frames": 0}
            continue
        a = np.asarray(vals, dtype=np.float64)
        out[name] = {
            "n_frames": int(a.size),
            "p0.5": float(np.quantile(a, 0.005)),
            "p1": float(np.quantile(a, 0.01)),
            "p50": float(np.quantile(a, 0.50)),
            "contact_fraction_at_recorded_h_on": float(np.mean(a < h_on)),
        }
    return out


def record_config(cfg: dict[str, Any], *, standing_char: dict[str, Any], envelope: dict) -> None:
    """All non-foot regions use foot contact constants (world z=0 floor)."""
    cfg["recorded_date"] = date.today().isoformat()
    foot = cfg["foot"]
    for name in NON_FOOT_REGION_NAMES:
        reg = cfg["regions"][name]
        reg["h_on_m"] = foot["h_on_m"]
        reg["h_off_m"] = foot["h_off_m"]
        reg["v_on_m_s"] = foot["v_on_m_s"]
        reg["v_off_m_s"] = foot["v_off_m_s"]
        reg["min_run"] = foot["min_run"]
    cfg["standing_foot_characterisation_val"] = standing_char
    cfg["loco_negative_envelope"] = envelope
    cfg["threshold_note"] = (
        "Non-foot h_on/h_off identical to foot (0.05/0.06 m); heights are lowest-vertex "
        "world z after clip floor_offset (pipeline floor z=0). Not fitted on floor-work eval."
    )


def _floor_work_eval_rels(csv_path: Path) -> set[str]:
    rels: set[str] = set()
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("geometry_class") in ("kneel", "lie"):
                rels.add(row["rel_path"])
    return rels


def load_floor_work_segments(csv_path: Path) -> list[SegmentRow]:
    rows: list[SegmentRow] = []
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gc = row["geometry_class"]
            if gc not in ("kneel", "lie", "sit_floor", "sit_support"):
                continue
            rows.append(
                SegmentRow(
                    rel_path=row["rel_path"],
                    subset=row["subset"],
                    geometry_class=gc,
                    start_s=float(row["segment_start_s"]),
                    end_s=float(row["segment_end_s"]),
                )
            )
    return rows


def _cohort_for_geometry(gc: str) -> str:
    if gc in ("kneel", "lie"):
        return "floor_work_eligible"
    return gc


def build_report_cohort_segments(
    floor_segments: list[SegmentRow], loco_segments: list[SegmentRow]
) -> dict[str, list[SegmentRow]]:
    cohorts: dict[str, list[SegmentRow]] = defaultdict(list)
    for seg in floor_segments:
        if seg.geometry_class in ("kneel", "lie"):
            cohorts[seg.geometry_class].append(seg)
            cohorts["floor_work_eligible"].append(seg)
        elif seg.geometry_class in ("sit_floor", "sit_support"):
            cohorts[seg.geometry_class].append(seg)
    cohorts["ordinary_locomotion"] = list(loco_segments)
    return cohorts


_MEAN_FRAC_DEF = (
    "Per-segment unweighted mean of each segment's per-frame contact fraction "
    "(ordinary_locomotion uses loco_frame_stride on segment frames)."
)


def aggregate_cohort_rates(
    cohort: str,
    segs: list[SegmentRow],
    cfg: dict[str, Any],
    region_sets: dict[str, np.ndarray],
    fk: FkClipCache,
    uncertain_map: dict[str, bool],
    *,
    frame_stride: int,
    non_foot_speed_mode: Optional[str] = None,
) -> dict[str, Any]:
    rels = {s.rel_path for s in segs}
    attr = attrition_counts(rels)
    fu = sum(1 for r in rels if uncertain_map.get(r, False))
    region_sums: dict[str, list[float]] = {n: [] for n in ALL_REGION_NAMES}
    subjects: set[str] = set()
    clips: set[str] = set()
    total_s = 0.0
    n_proc = 0
    stride = frame_stride if cohort == "ordinary_locomotion" else 1
    for seg in segs:
        entry = _entry_by_rel(load_index(), seg.rel_path)
        if clip_flags(entry).get("exclude_contact"):
            continue
        verts_full = fk.grounded_vertices(entry)
        total_s += segment_duration_s(verts_full.shape[0], seg.start_s, seg.end_s)
        subjects.add(entry.subject)
        clips.add(seg.rel_path)
        fr = compute_segment_region_fractions_cached(
            seg,
            cfg,
            region_sets,
            fk,
            frame_stride=stride,
            non_foot_speed_mode=non_foot_speed_mode,
        )
        for k, v in fr.items():
            if np.isfinite(v):
                region_sums[k].append(v)
        n_proc += 1
    out: dict[str, Any] = {
        "n_seg": n_proc,
        "n_clip": len(clips),
        "n_subj": len(subjects),
        "total_s": round(total_s, 3),
        "n_segments_processed": n_proc,
        "attrition": attr,
        "floor_uncertain_clips": fu,
        "mean_contact_fraction_per_region": {
            k: float(np.mean(v)) if v else None for k, v in region_sums.items()
        },
    }
    if cohort == "sit_support":
        out["floor_assuming_metrics"] = "excluded — no seat channel"
    return out


def accumulate_speed_gate_stats(
    seg: SegmentRow,
    cfg: dict[str, Any],
    region_sets: dict[str, np.ndarray],
    fk: FkClipCache,
    *,
    frame_stride: int,
    non_foot_speed_mode: Optional[str] = None,
) -> dict[str, dict[str, int]]:
    """Per-region frame counts for speed-gate characterisation (report only)."""
    entry = _entry_by_rel(load_index(), seg.rel_path)
    verts_full = fk.grounded_vertices(entry)
    sl = _frame_range(verts_full.shape[0], _TARGET_FPS, seg.start_s, seg.end_s)
    verts = verts_full[sl]
    if frame_stride > 1:
        verts = verts[::frame_stride]
    counts: dict[str, dict[str, int]] = {
        n: {
            "n_frames": 0,
            "n_low_h": 0,
            "n_low_h_no_contact": 0,
            "n_low_h_no_contact_speed": 0,
            "n_speed_block_argmin_changed": 0,
        }
        for n in NON_FOOT_REGION_NAMES
    }
    for name, ids in region_sets.items():
        if ids.size == 0:
            continue
        th = _region_thresholds(cfg, name)
        h, sp = region_height_speed(verts, ids, cfg, speed_mode=non_foot_speed_mode)
        mask = region_contact_mask(
            h,
            sp,
            h_on=th["h_on"],
            h_off=th["h_off"],
            v_on=th["v_on"],
            v_off=th["v_off"],
            min_run=th["min_run"],
        )
        mask_ns = _contact_mask_no_speed_gate(
            h,
            sp,
            h_on=th["h_on"],
            h_off=th["h_off"],
            min_run=th["min_run"],
        )
        low_h = h < th["h_on"]
        no_c = ~mask
        speed_block = low_h & no_c & mask_ns
        c = counts[name]
        c["n_frames"] += int(h.size)
        c["n_low_h"] += int(low_h.sum())
        c["n_low_h_no_contact"] += int((low_h & no_c).sum())
        c["n_low_h_no_contact_speed"] += int(speed_block.sum())
    return counts


def merge_speed_gate_counts(
    acc: dict[str, dict[str, int]], part: dict[str, dict[str, int]]
) -> None:
    for name, c in part.items():
        for k, v in c.items():
            acc[name][k] += v


def speed_gate_report_row(
    counts: dict[str, dict[str, int]],
) -> dict[str, dict[str, Optional[float]]]:
    out: dict[str, dict[str, Optional[float]]] = {}
    for name, c in counts.items():
        nf = c["n_frames"]
        n_low = c["n_low_h"]
        n_lh_nc = c["n_low_h_no_contact"]
        n_sp = c["n_low_h_no_contact_speed"]
        n_chg = c["n_speed_block_argmin_changed"]
        out[name] = {
            "frac_low_h_no_contact_due_to_speed": (
                float(n_sp / n_low) if n_low else None
            ),
            "frac_low_h_no_contact_due_to_speed_of_no_contact": (
                float(n_sp / n_lh_nc) if n_lh_nc else None
            ),
            "frac_speed_block_argmin_changed": (
                float(n_chg / n_sp) if n_sp else None
            ),
            "n_frames": nf,
            "n_low_h": n_low,
            "n_low_h_no_contact_speed": n_sp,
        }
    return out


def lie_head_shin_height_distribution(
    lie_segments: list[SegmentRow],
    cfg: dict[str, Any],
    region_sets: dict[str, np.ndarray],
    fk: FkClipCache,
) -> dict[str, Any]:
    h_on = float(cfg["foot"]["h_on_m"])
    pools: dict[str, list[float]] = {"head": [], "shins": []}
    for seg in lie_segments:
        entry = _entry_by_rel(load_index(), seg.rel_path)
        if clip_flags(entry).get("exclude_contact"):
            continue
        verts_full = fk.grounded_vertices(entry)
        sl = _frame_range(verts_full.shape[0], _TARGET_FPS, seg.start_s, seg.end_s)
        verts = verts_full[sl]
        for name in ("head", "shins"):
            ids = region_sets[name]
            h, _ = lowest_vertex_height_speed(verts, ids)
            pools[name].extend(float(x) for x in h)
    report: dict[str, Any] = {"h_on_m": h_on, "n_lie_segments": len(lie_segments)}
    for name, vals in pools.items():
        if not vals:
            report[name] = {"n_frames": 0}
            continue
        a = np.asarray(vals, dtype=np.float64)
        report[name] = {
            "n_frames": int(a.size),
            "min_m": float(a.min()),
            "p10_m": float(np.quantile(a, 0.10)),
            "p50_m": float(np.quantile(a, 0.50)),
            "frac_below_h_on": float(np.mean(a < h_on)),
            "mean_contact_fraction_segments": None,
        }
    return report


def validation_clip_summary(
    verts: np.ndarray,
    cfg: dict[str, Any],
    region_sets: dict[str, np.ndarray],
    fk: FkClipCache,
    entry: AmassIndexEntry,
    sl: slice,
) -> dict[str, Any]:
    region_stats: dict[str, Any] = {}
    pos = fk.foot_positions(entry)
    for name, ids in region_sets.items():
        th = _region_thresholds(cfg, name)
        h, sp = region_height_speed(verts, ids, cfg)
        mask = region_contact_mask(
            h,
            sp,
            h_on=th["h_on"],
            h_off=th["h_off"],
            v_on=th["v_on"],
            v_off=th["v_off"],
            min_run=th["min_run"],
        )
        region_stats[name] = {
            "lowest_height_min_m": float(h.min()) if h.size else None,
            "lowest_height_median_m": float(np.median(h)) if h.size else None,
            "contact_fraction": float(mask.mean()) if mask.size else 0.0,
            "contact_runs_inclusive": contact_runs_inclusive(mask),
        }
    foot_frac = 0.0
    foot_runs: list[tuple[int, int]] = []
    if pos is not None:
        foot = foot_contact_mask(pos[sl])
        foot_1d = np.asarray(foot).any(axis=-1) if foot.ndim > 1 else foot
        foot_frac = float(foot_1d.mean()) if foot_1d.size else 0.0
        foot_runs = contact_runs_inclusive(foot_1d)
    region_stats["feet"] = {
        "contact_fraction": foot_frac,
        "contact_runs_inclusive": foot_runs,
    }
    return region_stats


def _region_thresholds(cfg: dict[str, Any], name: str) -> dict[str, float]:
    reg = cfg["regions"][name]
    return {
        "h_on": float(reg["h_on_m"]),
        "h_off": float(reg["h_off_m"]),
        "v_on": float(reg["v_on_m_s"]),
        "v_off": float(reg["v_off_m_s"]),
        "min_run": int(reg["min_run"]),
    }


def compute_segment_region_fractions_cached(
    seg: SegmentRow,
    cfg: dict[str, Any],
    region_sets: dict[str, np.ndarray],
    fk: FkClipCache,
    *,
    z_shift_m: float = 0.0,
    h_on_override: Optional[float] = None,
    frame_stride: int = 1,
    non_foot_speed_mode: Optional[str] = None,
) -> dict[str, float]:
    entry = _entry_by_rel(load_index(), seg.rel_path)
    verts_full = fk.grounded_vertices(entry)
    sl = _frame_range(verts_full.shape[0], _TARGET_FPS, seg.start_s, seg.end_s)
    verts = verts_full[sl]
    if frame_stride > 1:
        verts = verts[::frame_stride]
    if z_shift_m != 0.0:
        verts = verts.copy()
        verts[:, :, 2] += z_shift_m
    fracs: dict[str, float] = {}
    for name, ids in region_sets.items():
        th = _region_thresholds(cfg, name)
        h_on = float(h_on_override if h_on_override is not None else th["h_on"])
        h_off = float(th["h_off"] if h_on_override is None else h_on + 0.01)
        h, sp = region_height_speed(verts, ids, cfg, speed_mode=non_foot_speed_mode)
        mask = region_contact_mask(
            h,
            sp,
            h_on=h_on,
            h_off=h_off,
            v_on=th["v_on"],
            v_off=th["v_off"],
            min_run=th["min_run"],
        )
        fracs[name] = float(mask.mean()) if mask.size else 0.0
    pos = fk.foot_positions(entry)
    if pos is None:
        fracs["feet"] = float("nan")
    else:
        foot = foot_contact_mask(pos[sl], z_shift_m=z_shift_m)
        if frame_stride > 1:
            foot = foot[::frame_stride]
        fracs["feet"] = float(foot.mean()) if foot.size else 0.0
    return fracs


def attrition_counts(rels: set[str]) -> dict[str, int]:
    ex_c = ex_p = 0
    for rel in rels:
        fl = clip_flags(_entry_by_rel(load_index(), rel))
        if fl.get("exclude_contact"):
            ex_c += 1
        if fl.get("exclude_physical_eval"):
            ex_p += 1
    return {
        "unique_clips": len(rels),
        "exclude_contact": ex_c,
        "exclude_physical_eval": ex_p,
    }


def _floor_uncertain_by_rel(standing_char: dict[str, Any]) -> dict[str, bool]:
    return {
        r["rel_path"]: bool(r["floor_uncertain"])
        for r in standing_char.get("per_clip", [])
    }


def hand_pose_non_default(entry: AmassIndexEntry) -> bool:
    path = amass_root_from_config() / entry.rel_path
    with np.load(path, allow_pickle=True) as data:
        if "pose_hand" not in data:
            return False
        ph = np.asarray(data["pose_hand"], dtype=np.float64)
        return bool(np.max(np.abs(ph)) > 1e-6)


def _aggregate_cohort_speed_gate(
    cohort: str,
    segs: list[SegmentRow],
    cfg: dict[str, Any],
    region_sets: dict[str, np.ndarray],
    fk: FkClipCache,
    *,
    loco_stride: int,
    non_foot_speed_mode: Optional[str] = None,
) -> dict[str, dict[str, Optional[float]]]:
    acc = {
        n: {
            "n_frames": 0,
            "n_low_h": 0,
            "n_low_h_no_contact": 0,
            "n_low_h_no_contact_speed": 0,
            "n_speed_block_argmin_changed": 0,
        }
        for n in NON_FOOT_REGION_NAMES
    }
    stride = loco_stride if cohort == "ordinary_locomotion" else 1
    for seg in segs:
        entry = _entry_by_rel(load_index(), seg.rel_path)
        if clip_flags(entry).get("exclude_contact"):
            continue
        merge_speed_gate_counts(
            acc,
            accumulate_speed_gate_stats(
                seg,
                cfg,
                region_sets,
                fk,
                frame_stride=stride,
                non_foot_speed_mode=non_foot_speed_mode,
            ),
        )
    return speed_gate_report_row(acc)


def _non_foot_mean_contact_changes(
    before: dict[str, Optional[float]], after: dict[str, Optional[float]]
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for name in NON_FOOT_REGION_NAMES:
        b = float(before.get(name) or 0.0)
        a = float(after.get(name) or 0.0)
        if abs(a - b) > 1e-12:
            rows.append({"region": name, "before": b, "after": a, "delta": a - b})
    return rows


def build_speed_amendment_2026_10_04(
    cfg: dict[str, Any],
    cohort_segments: dict[str, list[SegmentRow]],
    region_sets: dict[str, np.ndarray],
    fk: FkClipCache,
    uncertain_map: dict[str, bool],
    loco_stride: int,
    after_cohorts: dict[str, Any],
    after_speed_gate: dict[str, Any],
) -> dict[str, Any]:
    """Before/after tables: legacy argmin speed vs patch-median (production)."""
    ba = ("kneel", "lie", "sit_floor", "sit_support", "ordinary_locomotion")
    cohort_tables: dict[str, Any] = {}
    for cohort in ba:
        segs = cohort_segments.get(cohort, [])
        before_rates = aggregate_cohort_rates(
            cohort,
            segs,
            cfg,
            region_sets,
            fk,
            uncertain_map,
            frame_stride=loco_stride,
            non_foot_speed_mode=_LEGACY_NON_FOOT_SPEED,
        )
        before_sg = _aggregate_cohort_speed_gate(
            cohort,
            segs,
            cfg,
            region_sets,
            fk,
            loco_stride=loco_stride,
            non_foot_speed_mode=_LEGACY_NON_FOOT_SPEED,
        )
        after_sg = after_speed_gate.get(cohort)
        if after_sg is None:
            after_sg = _aggregate_cohort_speed_gate(
                cohort, segs, cfg, region_sets, fk, loco_stride=loco_stride
            )
        after_means = after_cohorts[cohort]["mean_contact_fraction_per_region"]
        before_means = before_rates["mean_contact_fraction_per_region"]
        regions: dict[str, Any] = {}
        for name in ALL_REGION_NAMES:
            rec: dict[str, Any] = {
                "mean_contact_fraction": {
                    "before": before_means.get(name),
                    "after": after_means.get(name),
                }
            }
            if name in NON_FOOT_REGION_NAMES:
                rec["frac_low_h_no_contact_due_to_speed"] = {
                    "before": before_sg[name]["frac_low_h_no_contact_due_to_speed"],
                    "after": after_sg[name]["frac_low_h_no_contact_due_to_speed"],
                }
            regions[name] = rec
        cohort_tables[cohort] = {
            "regions": regions,
            "non_foot_mean_contact_changes": _non_foot_mean_contact_changes(
                {k: before_means.get(k) for k in NON_FOOT_REGION_NAMES},
                {k: after_means.get(k) for k in NON_FOOT_REGION_NAMES},
            ),
        }

    squat_seg = SegmentRow(
        "Eyes_Japan_Dataset/aita/sitdown_standup-09-squat_down-aita_stageii.npz",
        "",
        "sit_floor",
        3.334,
        8.474,
    )
    squat_before = compute_segment_region_fractions_cached(
        squat_seg,
        cfg,
        region_sets,
        fk,
        non_foot_speed_mode=_LEGACY_NON_FOOT_SPEED,
    )
    squat_after = compute_segment_region_fractions_cached(squat_seg, cfg, region_sets, fk)
    squat_non_foot = {
        n: {"before": squat_before[n], "after": squat_after[n]}
        for n in ("shins", "thighs", "pelvis_seat")
    }

    lie_segs = cohort_segments.get("lie", [])
    band_sens: dict[str, Any] = {}
    for band in (0.01, 0.03):
        cfg_band = {**cfg, "patch_band_m": band}
        sg = _aggregate_cohort_speed_gate(
            "lie", lie_segs, cfg_band, region_sets, fk, loco_stride=1
        )
        band_sens[str(band)] = {
            n: sg[n]["frac_low_h_no_contact_due_to_speed"] for n in NON_FOOT_REGION_NAMES
        }

    return {
        "date": "2026-10-04",
        "reason": (
            "Argmin-vertex horizontal speed spiked when the lowest vertex index switched "
            "(lie A9 thighs/pelvis gaps despite h << h_on). Non-foot speed only; feet unchanged."
        ),
        "before_non_foot_speed": _LEGACY_NON_FOOT_SPEED,
        "after_non_foot_speed": cfg.get("speed_definition"),
        "patch_band_m": cfg.get("patch_band_m"),
        "patch_band_m_sensitivity_lie_speed_gate_only": band_sens,
        "cohorts": cohort_tables,
        "validation_squat_down_non_foot": squat_non_foot,
    }


def run_foot_regression_27cc2da(seed: int = 0, n_babel: int = 300) -> int:
    from hready.data.contact import foot_traj_build_clip_list

    repo_root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        ["git", "show", "27cc2da:hready/data/contact.py"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    ns_old: dict[str, Any] = {"np": np}
    exec(proc.stdout, ns_old)  # noqa: S102
    compute_old = ns_old["compute_foot_contact_from_positions"]
    mism = 0
    for entry in foot_traj_build_clip_list(seed=seed, n_babel=n_babel):
        pos = load_foot_positions(entry)
        if not np.array_equal(compute_foot_contact_from_positions(pos), compute_old(pos)):
            mism += 1
    return mism


def print_validation_frames(
    cfg: dict[str, Any],
    region_sets: dict[str, np.ndarray],
    fk: FkClipCache,
) -> list[dict[str, Any]]:
    """Stdout per-frame dump; return summary rows for JSON."""
    summaries: list[dict[str, Any]] = []
    val_cfg = cfg.get("validation_clips") or {}
    for cohort, clips in val_cfg.items():
        for spec in clips:
            rel = spec["rel_path"]
            entry = _entry_by_rel(load_index(), rel)
            start_s = spec.get("start_s")
            end_s = spec.get("end_s")
            verts_full = fk.grounded_vertices(entry)
            n = verts_full.shape[0]
            if start_s is None or end_s is None:
                sl = slice(0, n)
                t0, t1 = 0.0, (n - 1) / _TARGET_FPS
            else:
                t0, t1 = float(start_s), float(end_s)
                sl = _frame_range(n, _TARGET_FPS, t0, t1)
            verts = verts_full[sl]
            hand_nd = hand_pose_non_default(entry)
            region_masks: dict[str, np.ndarray] = {}
            region_heights: dict[str, np.ndarray] = {}
            for name, ids in region_sets.items():
                th = _region_thresholds(cfg, name)
                h, sp = region_height_speed(verts, ids, cfg)
                region_heights[name] = h
                region_masks[name] = region_contact_mask(
                    h,
                    sp,
                    h_on=th["h_on"],
                    h_off=th["h_off"],
                    v_on=th["v_on"],
                    v_off=th["v_off"],
                    min_run=th["min_run"],
                )
            pos = fk.foot_positions(entry)
            foot_mask: Optional[np.ndarray] = None
            if pos is not None:
                foot_raw = foot_contact_mask(pos[sl])
                foot_mask = (
                    np.asarray(foot_raw).any(axis=-1)
                    if np.asarray(foot_raw).ndim > 1
                    else foot_raw
                )
            print(
                f"\nVALIDATION_CLIP [{cohort}] {rel} t=[{t0:.3f},{t1:.3f}] "
                f"hand_pose_non_default={hand_nd}"
            )
            for fi in range(verts.shape[0]):
                parts: list[str] = [f"fi={fi}"]
                for name in region_sets:
                    parts.append(f"{name}_h={region_heights[name][fi]:.4f}")
                    parts.append(f"{name}_c={int(region_masks[name][fi])}")
                if foot_mask is not None:
                    parts.append(f"feet_c={int(np.asarray(foot_mask[fi]).any())}")
                print("  " + " ".join(parts))
            vsum = validation_clip_summary(verts, cfg, region_sets, fk, entry, sl)
            summaries.append(
                {
                    "cohort": cohort,
                    "rel_path": rel,
                    "start_s": t0,
                    "end_s": t1,
                    "hand_pose_non_default": hand_nd,
                    "regions": vsum,
                }
            )
    return summaries


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Support-contact v2 (Track E1b)")
    p.add_argument("--config", type=Path, default=Path("configs/support_contact_v2.yaml"))
    p.add_argument("--derive-config", action="store_true")
    p.add_argument("--report", action="store_true")
    p.add_argument("--all", action="store_true")
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)
    cfg_path = args.config.resolve()
    cfg = _load_yaml(cfg_path)
    repo_root = Path(__file__).resolve().parents[2]
    csv_path = (repo_root / cfg["floor_work_clips_csv"]).resolve()
    exclude_eval = _floor_work_eval_rels(csv_path)

    from hready.body.smplx_wrapper import load_body
    from hready.data.amass import _resolve_torch_device

    body = load_body("locked_head")
    dev = _resolve_torch_device(args.device)
    body._model.to(dev)
    fk = FkClipCache(body, str(dev))

    print("ORDER_OF_OPERATIONS:")
    print("  1) derive-config: characterise standing-foot residual, loco envelope, record yaml")
    print("  2) commit recorded config (manual step between CLI phases)")
    print("  3) report: cohort rates / sensitivity / validation stdout (uses recorded thresholds)")
    print("")

    region_sets = resolve_region_vertex_sets(body)
    print_region_inventory(body, device=str(dev))

    do_derive = args.derive_config or args.all or cfg["regions"]["shins"]["h_on_m"] is None
    do_report = args.report or args.all

    if do_derive:
        print("\n=== PHASE 1: derive-config ===")
        standing_char = characterize_standing_foot_residual_val()
        print("STANDING_FOOT_CHARACTERISATION overall:", json.dumps(standing_char["overall"], indent=2))
        print("STANDING_FOOT_CHARACTERISATION per_subset:", json.dumps(standing_char["per_subset"], indent=2))
        uncertain = [c for c in standing_char["per_clip"] if c["floor_uncertain"]]
        print(f"floor_uncertain clips (median>{_FLOOR_UNCERTAIN_MEDIAN_M}): n={len(uncertain)}")
        worst = sorted(
            standing_char["per_clip"],
            key=lambda x: x["median_stand_foot_min_channel_m"],
            reverse=True,
        )[:10]
        print("WORST_10_STAND_FOOT_RESIDUAL_CLIPS:")
        for w in worst:
            print(
                f"  {w['median_stand_foot_min_channel_m']:.4f} m  {w['subset']}  {w['rel_path']}  "
                f"babel={w['babel_stand_act_cat']}"
            )
        loco_segments = collect_loco_val_segments(cfg, exclude_rels=exclude_eval)
        stride = int((cfg.get("anti_circularity") or {}).get("loco_frame_stride", 4))
        envelope = derive_loco_region_envelope(
            cfg, region_sets, loco_segments, fk, frame_stride=stride
        )
        print(
            f"LOCO_ENVELOPE sample: n_segments={len(loco_segments)} seed="
            f"{cfg['anti_circularity']['loco_sample_seed']} stride={stride}"
        )
        for name, st in envelope.items():
            print(f"  {name}: {st}")
        record_config(cfg, standing_char=standing_char, envelope=envelope)
        _save_yaml(cfg_path, cfg)
        print(f"Wrote recorded config: {cfg_path} recorded_date={cfg['recorded_date']}")
        if not do_report:
            return

    if not do_report:
        p.print_help()
        return

    if cfg["regions"]["shins"]["h_on_m"] is None:
        print("Config not recorded — run --derive-config first", file=sys.stderr)
        sys.exit(1)

    print("\n=== PHASE 2: report (post-recording) ===")
    standing_char = cfg.get("standing_foot_characterisation_val") or {}
    uncertain_map = _floor_uncertain_by_rel(standing_char)

    segments = load_floor_work_segments(csv_path)
    loco_segments = collect_loco_val_segments(cfg, exclude_rels=exclude_eval)
    cohort_segments = build_report_cohort_segments(segments, loco_segments)
    anti = cfg.get("anti_circularity") or {}
    loco_stride = int(anti.get("loco_frame_stride", 4))
    loco_subjects = {
        _entry_by_rel(load_index(), s.rel_path).subject for s in loco_segments
    }
    loco_clips = {s.rel_path for s in loco_segments}

    rates: dict[str, Any] = {
        "recorded_date": cfg.get("recorded_date"),
        "production_thresholds": cfg["foot"],
        "mean_contact_fraction_definition": _MEAN_FRAC_DEF,
        "total_duration_s_note": (
            "total_s sums segment frame-window lengths "
            "(floor(start_s*fps)..ceil(end_s*fps) at 30 fps), not raw end-start; "
            "E1 cohort_counts uses end-start only (e.g. floor_work_eligible 100.803 vs 100.700)."
        ),
        "loco_sample": {
            "seed": anti.get("loco_sample_seed"),
            "frame_stride": loco_stride,
            "n_segments_sampled": len(loco_segments),
            "n_segments": len(loco_segments),
            "n_clips": len(loco_clips),
            "n_subjects": len(loco_subjects),
            "segment_skip": "segments on clips with exclude_contact are omitted from rates",
        },
        "cohorts": {},
        "speed_gate_characterisation": {},
        "lie_head_shin_heights": {},
    }

    cohort_order = (
        "kneel",
        "lie",
        "floor_work_eligible",
        "sit_floor",
        "sit_support",
        "ordinary_locomotion",
    )
    speed_gate_cohorts = (
        "kneel",
        "lie",
        "sit_floor",
        "sit_support",
        "ordinary_locomotion",
    )
    for cohort in cohort_order:
        segs = cohort_segments.get(cohort, [])
        rates["cohorts"][cohort] = aggregate_cohort_rates(
            cohort,
            segs,
            cfg,
            region_sets,
            fk,
            uncertain_map,
            frame_stride=loco_stride,
        )
        if cohort in speed_gate_cohorts:
            acc = {
                n: {
                    "n_frames": 0,
                    "n_low_h": 0,
                    "n_low_h_no_contact": 0,
                    "n_low_h_no_contact_speed": 0,
                    "n_speed_block_argmin_changed": 0,
                }
                for n in NON_FOOT_REGION_NAMES
            }
            stride = loco_stride if cohort == "ordinary_locomotion" else 1
            for seg in segs:
                entry = _entry_by_rel(load_index(), seg.rel_path)
                if clip_flags(entry).get("exclude_contact"):
                    continue
                merge_speed_gate_counts(
                    acc, accumulate_speed_gate_stats(
                        seg, cfg, region_sets, fk, frame_stride=stride
                    )
                )
            rates["speed_gate_characterisation"][cohort] = speed_gate_report_row(acc)

    rates["amendment_2026_10_04_non_foot_speed"] = build_speed_amendment_2026_10_04(
        cfg,
        cohort_segments,
        region_sets,
        fk,
        uncertain_map,
        loco_stride,
        rates["cohorts"],
        rates["speed_gate_characterisation"],
    )

    lie_segs = cohort_segments.get("lie", [])
    lie_dist = lie_head_shin_height_distribution(lie_segs, cfg, region_sets, fk)
    for name in ("head", "shins"):
        fracs = []
        for seg in lie_segs:
            entry = _entry_by_rel(load_index(), seg.rel_path)
            if clip_flags(entry).get("exclude_contact"):
                continue
            fr = compute_segment_region_fractions_cached(
                seg, cfg, region_sets, fk, frame_stride=1
            )
            fracs.append(fr[name])
        if fracs and name in lie_dist:
            lie_dist[name]["mean_contact_fraction_segments"] = float(np.mean(fracs))
    rates["lie_head_shin_heights"] = lie_dist

    rates["cohorts"]["crawl"] = {"n_confirmed_segments": 0, "not_evaluable": True}
    rates["cohorts"]["yoga_like"] = {"n_confirmed_segments": 0, "not_evaluable": True}

    val_specs = cfg.get("validation_clips") or {}
    grid_h = [0.05, 0.07, 0.10]
    grid_shift = [0.0, 0.035, 0.07]
    sens: list[dict[str, Any]] = []
    for cohort, clips in val_specs.items():
        for spec in clips:
            start_s = spec.get("start_s")
            end_s = spec.get("end_s")
            seg = SegmentRow(
                spec["rel_path"],
                "",
                cohort,
                float(start_s if start_s is not None else 0.0),
                float(end_s if end_s is not None else 1e9),
            )
            entry = _entry_by_rel(load_index(), seg.rel_path)
            fk.grounded_vertices(entry)
            for h_on in grid_h:
                for shift in grid_shift:
                    fr = compute_segment_region_fractions_cached(
                        seg,
                        cfg,
                        region_sets,
                        fk,
                        z_shift_m=shift,
                        h_on_override=h_on,
                    )
                    sens.append(
                        {
                            "rel_path": spec["rel_path"],
                            "cohort": cohort,
                            "h_on_m": h_on,
                            "z_shift_m": shift,
                            "contact_fraction": fr,
                        }
                    )
    rates["sensitivity_validation_clips"] = sens
    rates["validation_summaries"] = print_validation_frames(cfg, region_sets, fk)

    mism = run_foot_regression_27cc2da()
    rates["foot_regression_contact_mism_27cc2da"] = mism
    print(f"FOOT_REGRESSION contact_mism={mism} (27cc2da, seed=0, n=300)")

    out_path = (repo_root / cfg["output_json"]).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(rates, f, indent=2)
    print(f"\nWrote {out_path}")
    print("VALIDATION_CLIP_NAMES:")
    for cohort, clips in val_specs.items():
        for spec in clips:
            print(f"  [{cohort}] {spec['rel_path']} {spec.get('start_s')}-{spec.get('end_s')}")


if __name__ == "__main__":
    main()
