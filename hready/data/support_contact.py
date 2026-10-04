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


def lowest_vertex_height_speed(
    verts: np.ndarray, region_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
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
            "contact_fraction_at_frozen_h_on": float(np.mean(a < h_on)),
        }
    return out


def freeze_config(cfg: dict[str, Any], *, standing_char: dict[str, Any], envelope: dict) -> None:
    """All non-foot regions use foot contact constants (world z=0 floor)."""
    cfg["frozen_date"] = date.today().isoformat()
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
        h, sp = lowest_vertex_height_speed(verts, ids)
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
            print(
                f"\nVALIDATION_CLIP [{cohort}] {rel} t=[{t0:.3f},{t1:.3f}] "
                f"hand_pose_non_default={hand_nd}"
            )
            region_contact_rates: dict[str, float] = {}
            for fi in range(verts.shape[0]):
                frame = verts[fi : fi + 1]
                parts: list[str] = [f"fi={fi}"]
                for name, ids in region_sets.items():
                    th = _region_thresholds(cfg, name)
                    h, sp = lowest_vertex_height_speed(frame, ids)
                    mask = region_contact_mask(
                        h,
                        sp,
                        h_on=th["h_on"],
                        h_off=th["h_off"],
                        v_on=th["v_on"],
                        v_off=th["v_off"],
                        min_run=th["min_run"],
                    )
                    parts.append(f"{name}_h={h[0]:.4f}")
                    parts.append(f"{name}_c={int(mask[0])}")
                print("  " + " ".join(parts))
            for name in region_sets:
                th = _region_thresholds(cfg, name)
                h, sp = lowest_vertex_height_speed(verts, region_sets[name])
                m = region_contact_mask(
                    h, sp, h_on=th["h_on"], h_off=th["h_off"], v_on=th["v_on"],
                    v_off=th["v_off"], min_run=th["min_run"],
                )
                region_contact_rates[name] = float(m.mean())
            summaries.append(
                {
                    "cohort": cohort,
                    "rel_path": rel,
                    "start_s": t0,
                    "end_s": t1,
                    "hand_pose_non_default": hand_nd,
                    "frozen_threshold_contact_fraction": region_contact_rates,
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
    print("  1) derive-config: characterise standing-foot residual, loco envelope, freeze yaml")
    print("  2) commit frozen config (manual step between CLI phases)")
    print("  3) report: cohort rates / sensitivity / validation stdout (uses frozen thresholds)")
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
        freeze_config(cfg, standing_char=standing_char, envelope=envelope)
        _save_yaml(cfg_path, cfg)
        print(f"Wrote frozen config: {cfg_path} frozen_date={cfg['frozen_date']}")
        if not do_report:
            return

    if not do_report:
        p.print_help()
        return

    if cfg["regions"]["shins"]["h_on_m"] is None:
        print("Config not frozen — run --derive-config first", file=sys.stderr)
        sys.exit(1)

    print("\n=== PHASE 2: report (post-freeze) ===")
    standing_char = cfg.get("standing_foot_characterisation_val") or {}
    uncertain_map = _floor_uncertain_by_rel(standing_char)

    segments = load_floor_work_segments(csv_path)
    cohort_segments: dict[str, list[SegmentRow]] = defaultdict(list)
    for seg in segments:
        cohort_segments[_cohort_for_geometry(seg.geometry_class)].append(seg)

    loco_segments = collect_loco_val_segments(cfg, exclude_rels=exclude_eval)
    cohort_segments["ordinary_locomotion"] = loco_segments
    anti = cfg.get("anti_circularity") or {}
    loco_stride = int(anti.get("loco_frame_stride", 4))
    loco_subjects = {
        _entry_by_rel(load_index(), s.rel_path).subject for s in loco_segments
    }
    loco_clips = {s.rel_path for s in loco_segments}

    rates: dict[str, Any] = {
        "frozen_date": cfg.get("frozen_date"),
        "production_thresholds": cfg["foot"],
        "loco_sample": {
            "seed": anti.get("loco_sample_seed"),
            "frame_stride": loco_stride,
            "n_segments": len(loco_segments),
            "n_clips": len(loco_clips),
            "n_subjects": len(loco_subjects),
        },
        "cohorts": {},
    }

    for cohort, segs in cohort_segments.items():
        rels = {s.rel_path for s in segs}
        attr = attrition_counts(rels)
        fu = sum(1 for r in rels if uncertain_map.get(r, False))
        region_sums: dict[str, list[float]] = {n: [] for n in ALL_REGION_NAMES}
        n_proc = 0
        stride = loco_stride if cohort == "ordinary_locomotion" else 1
        for seg in segs:
            entry = _entry_by_rel(load_index(), seg.rel_path)
            if clip_flags(entry).get("exclude_contact"):
                continue
            fr = compute_segment_region_fractions_cached(
                seg, cfg, region_sets, fk, frame_stride=stride
            )
            for k, v in fr.items():
                if np.isfinite(v):
                    region_sums[k].append(v)
            n_proc += 1
        rates["cohorts"][cohort] = {
            "n_segments_processed": n_proc,
            "attrition": attr,
            "floor_uncertain_clips": fu,
            "mean_contact_fraction_per_region": {
                k: float(np.mean(v)) if v else None for k, v in region_sums.items()
            },
        }
        if cohort == "sit_support":
            rates["cohorts"][cohort]["floor_assuming_metrics"] = "excluded — no seat channel"

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
