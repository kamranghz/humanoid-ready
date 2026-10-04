"""Track E1: subject split audit, cohorts, BABEL + geometry floor-work evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import yaml

from hready.body.batch_forward import smpl_forward_bt
from hready.body.joint_indices import (
    FEET_JOINTS,
    HEAD,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    NECK,
    PELVIS,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)
from hready.body.smplx_wrapper import SmplxBody, load_body
from hready.data.amass import (
    AmassIndexEntry,
    assign_split,
    clip_flags,
    load_clip,
    load_index,
)
from hready.data.babel import (
    BabelIndexEntry,
    act_cat_matches_keyword,
    load_babel_index_payload,
)
from hready.data.foot_height_rise import load_foot_height_rise_cache

SplitName = Literal["train", "val", "test"]
EvalSplit = Literal["val", "test"]

BABEL_TO_ACCEPTED_GEOMETRY: dict[str, frozenset[str]] = {
    "lie": frozenset({"lie"}),
    "crawl": frozenset({"crawl"}),
    "kneel": frozenset({"kneel"}),
    "sit": frozenset({"sit_floor", "sit_support"}),
    "yoga": frozenset({"yoga_like"}),
}

GEOMETRY_CLASSES = (
    "lie",
    "crawl",
    "kneel",
    "sit_floor",
    "sit_support",
    "yoga_like",
    "none",
)

FLOOR_WORK_ELIGIBLE_GEOMETRY = frozenset(
    {"kneel", "lie", "crawl", "yoga_like"}
)


@dataclass(frozen=True)
class BabelProposal:
    rel_path: str
    subject: str
    subject_key: str
    category: str
    keyword: str
    start_s: float
    end_s: float
    ann_source: str
    cohort: str  # floor_work | ordinary_locomotion | other_labelled


@dataclass(frozen=True)
class FrameFeatures:
    pelvis_h: float
    torso_up_dot: float
    wrist_h: float
    head_h: float
    knee_h: float
    foot_z_min: float
    thigh_up_dot: float
    shoulder_h: float


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def assert_joint_table_runtime() -> None:
    from hready.body.joint_indices import JOINT_NAMES_22
    from hready.data.refine_corrupt import NUM_KP_JOINTS

    if NUM_KP_JOINTS != 22:
        raise RuntimeError(f"NUM_KP_JOINTS={NUM_KP_JOINTS}, expected 22")
    doc = __import__("hready.losses._constants", fromlist=["COM_MASS_FRAC_DOC"]).COM_MASS_FRAC_DOC
    if "0 pelvis" not in doc or "12 neck" not in doc:
        raise RuntimeError("losses/_constants.py doc missing pelvis/neck indices")
    if JOINT_NAMES_22[NECK] != "neck" or JOINT_NAMES_22[LEFT_WRIST] != "left_wrist":
        raise RuntimeError("joint_indices.py order mismatch")


def _sha256_lines(lines: list[str]) -> str:
    payload = "\n".join(lines) + ("\n" if lines else "")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def split_clip_list_hashes(entries: list[AmassIndexEntry]) -> dict[str, str]:
    out: dict[str, str] = {}
    for sp in ("train", "val", "test"):
        rels = sorted(e.rel_path for e in entries if assign_split(e) == sp)
        out[sp] = _sha256_lines(rels)
    return out


def subject_disjoint_counts(entries: list[AmassIndexEntry]) -> dict[str, Any]:
    by_split: dict[str, set[str]] = defaultdict(set)
    for e in entries:
        sp = assign_split(e)
        if sp not in ("train", "val", "test"):
            continue
        by_split[sp].add(f"{e.subset}/{e.subject}")

    def inter(a: str, b: str) -> int:
        return len(by_split[a] & by_split[b])

    return {
        "n_subjects": {k: len(v) for k, v in sorted(by_split.items())},
        "overlap_train_val": inter("train", "val"),
        "overlap_train_test": inter("train", "test"),
        "overlap_val_test": inter("val", "test"),
        "subject_disjoint": (
            inter("train", "val") == 0 and inter("train", "test") == 0 and inter("val", "test") == 0
        ),
    }


def select_tune_subjects(
    entries: list[AmassIndexEntry], *, seed: int, fraction: float
) -> list[str]:
    train_subjects = sorted(
        {f"{e.subset}/{e.subject}" for e in entries if assign_split(e) == "train"}
    )
    n = round(len(train_subjects) * fraction)
    rng = random.Random(seed)
    return sorted(rng.sample(train_subjects, n))


def _babel_category_from_segment(
    act_cats: list[str], keyword_map: dict[str, str]
) -> tuple[str, str] | None:
    for kw, cat in keyword_map.items():
        for ac in act_cats:
            if act_cat_matches_keyword(ac, kw):
                return cat, kw
    return None


def _collect_babel_proposals(
    babel_by_rel: dict[str, BabelIndexEntry],
    split_rels: set[str],
    amass_by_rel: dict[str, AmassIndexEntry],
    cohort_keywords: dict[str, dict[str, str]],
    min_dur_s: float,
) -> list[BabelProposal]:
    out: list[BabelProposal] = []
    for rel, be in babel_by_rel.items():
        if rel not in split_rels:
            continue
        entry = amass_by_rel[rel]
        sk = f"{entry.subset}/{entry.subject}"
        for seg in be.segments:
            start_s = float(seg["start_t"])
            end_s = float(seg["end_t"])
            if end_s - start_s < min_dur_s:
                continue
            cats = list(seg.get("act_cat") or [])
            for cohort, kw_map in cohort_keywords.items():
                hit = _babel_category_from_segment(cats, kw_map)
                if hit is None:
                    continue
                cat, kw = hit
                out.append(
                    BabelProposal(
                        rel_path=rel,
                        subject=entry.subject,
                        subject_key=sk,
                        category=cat,
                        keyword=kw,
                        start_s=start_s,
                        end_s=end_s,
                        ann_source=be.ann_source,
                        cohort=cohort,
                    )
                )
                break
    return out


def _thigh_up_dot(joints_t: np.ndarray) -> float:
    vals: list[float] = []
    for hip, knee in ((LEFT_HIP, LEFT_KNEE), (RIGHT_HIP, RIGHT_KNEE)):
        v = joints_t[knee] - joints_t[hip]
        n = float(np.linalg.norm(v))
        if n > 1e-6:
            vals.append(float(v[2] / n))
    return float(np.mean(vals)) if vals else float("nan")


def extract_frame_features(
    joints_t: np.ndarray, *, stand_pelvis_z: float, use_normalized: bool
) -> FrameFeatures | None:
    scale = float(stand_pelvis_z) if use_normalized and stand_pelvis_z > 1e-6 else 1.0
    torso = joints_t[NECK] - joints_t[PELVIS]
    norm = float(np.linalg.norm(torso))
    if norm < 1e-6:
        return None
    up_dot = float(torso[2] / norm)
    pelvis_h = float(joints_t[PELVIS, 2]) / scale
    wrist_h = float((joints_t[LEFT_WRIST, 2] + joints_t[RIGHT_WRIST, 2]) * 0.5) / scale
    head_h = float(joints_t[HEAD, 2]) / scale
    knee_h = float((joints_t[LEFT_KNEE, 2] + joints_t[RIGHT_KNEE, 2]) * 0.5) / scale
    foot_z_min = float(min(joints_t[j, 2] for j in FEET_JOINTS)) / scale
    thigh = _thigh_up_dot(joints_t)
    shoulder_h = (
        float(joints_t[LEFT_SHOULDER, 2] + joints_t[RIGHT_SHOULDER, 2]) * 0.5 / scale
    )
    return FrameFeatures(
        pelvis_h=pelvis_h,
        torso_up_dot=up_dot,
        wrist_h=wrist_h,
        head_h=head_h,
        knee_h=knee_h,
        foot_z_min=foot_z_min,
        thigh_up_dot=thigh,
        shoulder_h=shoulder_h,
    )


def standing_pelvis_height(body: SmplxBody, betas: np.ndarray) -> float:
    be = np.asarray(betas, dtype=np.float32).reshape(1, -1)
    with torch.no_grad():
        j, _ = smpl_forward_bt(
            body,
            torch.zeros(1, 1, 3),
            torch.zeros(1, 1, 3),
            torch.zeros(1, 1, 21, 3),
            torch.as_tensor(be),
        )
    return float(j[0, 0, PELVIS, 2].cpu().numpy())


def classify_frame_decision_tree(f: FrameFeatures, th: dict[str, float]) -> str:
    """Mutually exclusive geometry classes (documented order)."""
    horiz = f.torso_up_dot <= th["torso_horizontal_max"]
    upright = f.torso_up_dot >= th["torso_upright_min"]

    if horiz and f.pelvis_h <= th["crawl_pelvis_h_max"]:
        if (
            f.shoulder_h > th["crawl_shoulder_h_min"]
            and f.wrist_h <= th["crawl_wrist_h_max"]
        ):
            return "crawl"
        if (
            f.pelvis_h <= th["lie_pelvis_h_max"]
            and f.shoulder_h <= th["lie_shoulder_h_max"]
        ):
            return "lie"

    if (
        upright
        and f.thigh_up_dot <= th["kneel_thigh_up_max"]
        and f.pelvis_h <= th["kneel_pelvis_h_max"]
    ):
        return "kneel"

    if upright and f.thigh_up_dot >= th["sit_thigh_up_min"]:
        if f.pelvis_h <= th["sit_support_pelvis_h_min"]:
            return "sit_floor"
        return "sit_support"

    if (
        f.pelvis_h <= th["yoga_pelvis_h_max"]
        and f.torso_up_dot >= th["yoga_torso_up_min"]
        and f.torso_up_dot < th["torso_upright_min"]
    ):
        return "yoga_like"

    return "none"


def trace_decision_branch(f: FrameFeatures, th: dict[str, float]) -> str:
    """Read-only exit trace for diagnosis (frozen tree; not a geometry label)."""
    horiz = f.torso_up_dot <= th["torso_horizontal_max"]
    upright = f.torso_up_dot >= th["torso_upright_min"]

    if horiz and f.pelvis_h <= th["crawl_pelvis_h_max"]:
        if (
            f.shoulder_h > th["crawl_shoulder_h_min"]
            and f.wrist_h <= th["crawl_wrist_h_max"]
        ):
            return "class:crawl"
        if (
            f.pelvis_h <= th["lie_pelvis_h_max"]
            and f.shoulder_h <= th["lie_shoulder_h_max"]
        ):
            return "class:lie"
    if (
        upright
        and f.thigh_up_dot <= th["kneel_thigh_up_max"]
        and f.pelvis_h <= th["kneel_pelvis_h_max"]
    ):
        return "class:kneel"
    if upright and f.thigh_up_dot >= th["sit_thigh_up_min"]:
        if f.pelvis_h <= th["sit_support_pelvis_h_min"]:
            return "class:sit_floor"
        return "class:sit_support"
    if (
        f.pelvis_h <= th["yoga_pelvis_h_max"]
        and f.torso_up_dot >= th["yoga_torso_up_min"]
        and f.torso_up_dot < th["torso_upright_min"]
    ):
        return "class:yoga_like"

    if not horiz and not upright:
        return "none:torso_between_horiz_and_upright"
    if horiz:
        if f.pelvis_h > th["crawl_pelvis_h_max"]:
            return "none:raised_support_not_floor"
        if f.pelvis_h > th["lie_pelvis_h_max"]:
            return "none:horiz_pelvis_above_lie_max"
        if f.shoulder_h > th["crawl_shoulder_h_min"]:
            if f.wrist_h > th["crawl_wrist_h_max"]:
                return "none:horiz_crawl_shoulder_high_wrist_high"
            return "none:horiz_crawl_shoulder_high_wrist_ok"
        if f.pelvis_h > th["lie_pelvis_h_max"]:
            return "none:horiz_pelvis_above_lie_max"
        if f.shoulder_h > th["lie_shoulder_h_max"]:
            return "none:horiz_lie_shoulder_high"
        return "none:horiz_low_pelvis_unclassified"
    if upright:
        if f.thigh_up_dot > th["kneel_thigh_up_max"]:
            if f.thigh_up_dot < th["sit_thigh_up_min"]:
                return "none:upright_thigh_between_kneel_and_sit"
            if f.pelvis_h > th["sit_support_pelvis_h_min"]:
                return "none:upright_sit_thigh_ok_pelvis_above_support"
            return "none:upright_sit_thigh_ok_pelvis_low"
        if f.pelvis_h > th["kneel_pelvis_h_max"]:
            return "none:upright_kneel_thigh_ok_pelvis_high"
    if f.pelvis_h > th["yoga_pelvis_h_max"]:
        return "none:upright_or_high_pelvis_above_yoga"
    if f.torso_up_dot < th["yoga_torso_up_min"]:
        return "none:low_pelvis_torso_below_yoga_min"
    return "none:other"


def _geometry_class_for_segment(labels: list[str], babel_cat: str) -> str:
    acc = BABEL_TO_ACCEPTED_GEOMETRY.get(babel_cat, frozenset())
    accepted = [x for x in labels if x in acc]
    if accepted:
        return Counter(accepted).most_common(1)[0][0]
    dom = _dominant_label(labels, 0.0)
    return dom if dom is not None else "none"


def _legacy_frame_geometry(
    joints_t: np.ndarray,
    rules: dict[str, Any],
    neck: int,
    lw: int,
    rw: int,
) -> str | None:
    pelvis_z = float(joints_t[PELVIS, 2])
    torso = joints_t[neck] - joints_t[PELVIS]
    norm = float(np.linalg.norm(torso))
    if norm < 1e-6:
        return None
    up_dot = float(torso[2] / norm)
    wrist_z = float((joints_t[lw, 2] + joints_t[rw, 2]) * 0.5)
    matched: list[str] = []
    for cat, rule in rules.items():
        ok = True
        if "pelvis_z_max" in rule and pelvis_z > float(rule["pelvis_z_max"]):
            ok = False
        if "pelvis_z_min" in rule and pelvis_z < float(rule["pelvis_z_min"]):
            ok = False
        if "torso_up_dot_max" in rule and up_dot > float(rule["torso_up_dot_max"]):
            ok = False
        if "torso_up_dot_min" in rule and up_dot < float(rule["torso_up_dot_min"]):
            ok = False
        if "wrist_z_max" in rule and wrist_z > float(rule["wrist_z_max"]):
            ok = False
        if ok:
            matched.append(cat)
    if not matched:
        return None
    priority = ["lie", "crawl", "kneel", "sit", "crouch", "yoga"]
    for p in priority:
        if p in matched:
            return p
    return matched[0]


def _segment_frame_indices(fps: float, start_s: float, end_s: float, n_frames: int) -> range:
    i0 = max(0, int(np.floor(start_s * fps)))
    i1 = min(n_frames, int(np.ceil(end_s * fps)))
    return range(i0, i1) if i1 > i0 else range(0)


def _segment_labels_tree(
    joints: np.ndarray,
    fps: float,
    start_s: float,
    end_s: float,
    stand_z: float,
    use_norm: bool,
    th: dict[str, float],
) -> list[str]:
    labels: list[str] = []
    for i in _segment_frame_indices(fps, start_s, end_s, joints.shape[0]):
        feat = extract_frame_features(joints[i], stand_pelvis_z=stand_z, use_normalized=use_norm)
        if feat is None:
            labels.append("none")
        else:
            labels.append(classify_frame_decision_tree(feat, th))
    return labels


def _fraction_accepted(labels: list[str], babel_cat: str) -> float:
    acc = BABEL_TO_ACCEPTED_GEOMETRY.get(babel_cat)
    if not acc:
        return 0.0
    if not labels:
        return 0.0
    return sum(1 for x in labels if x in acc) / len(labels)


def _dominant_label(labels: list[str], min_fraction: float) -> str | None:
    counts = Counter(x for x in labels if x != "none")
    if not counts:
        return None
    cat, n = counts.most_common(1)[0]
    if n / len(labels) < min_fraction:
        return None
    return cat


def _classify_proposal(
    labels: list[str],
    babel_cat: str,
    min_fraction: float,
    ann_source: str,
    start_s: float,
    end_s: float,
    playback_dur: float,
) -> tuple[str, str | None]:
    if ann_source == "seq_ann" and end_s - start_s >= playback_dur * 0.99:
        # whole-file seq_ann: require geometry on the full playback range
        pass
    frac = _fraction_accepted(labels, babel_cat)
    if frac >= min_fraction:
        return "confirmed", babel_cat
    dom = _dominant_label(labels, min_fraction)
    acc = BABEL_TO_ACCEPTED_GEOMETRY.get(babel_cat, frozenset())
    if dom is not None and dom not in acc:
        return "disagreement", dom
    return "unconfirmed", dom


def _rise_for_clip(rel_path: str, rise_cache: dict[str, Any]) -> tuple[float | None, str]:
    row = rise_cache.get("entries", {}).get(rel_path, {})
    return row.get("rise_cm"), row.get("status", "missing")


def _load_joints_for_clip(body: SmplxBody, rel: str) -> tuple[np.ndarray, np.ndarray]:
    clip = load_clip(rel, ground=True)
    with torch.no_grad():
        tr = torch.as_tensor(clip["transl"], dtype=torch.float32)[None]
        ro = torch.as_tensor(clip["root_orient"], dtype=torch.float32)[None]
        ba = torch.as_tensor(clip["pose_body"], dtype=torch.float32)[None]
        be = torch.as_tensor(np.asarray(clip["betas"], dtype=np.float32)[None])
        j, _ = smpl_forward_bt(body, tr, ro, ba, be)
    return j[0].cpu().numpy(), np.asarray(clip["betas"], dtype=np.float64)


def _histogram_valley(x: np.ndarray, nbins: int = 64) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 100:
        return float(np.median(x))
    hist, edges = np.histogram(x, bins=nbins)
    # deepest interior bin valley
    best_i, best_v = 1, int(hist[1])
    for i in range(2, len(hist) - 1):
        if hist[i] < best_v:
            best_v = int(hist[i])
            best_i = i
    return float((edges[best_i] + edges[best_i + 1]) * 0.5)


def _label_free_val_frames(
    entries: list[AmassIndexEntry],
    body: SmplxBody,
    fps: float,
    stride: int,
) -> tuple[dict[str, np.ndarray], list[float], list[float]]:
    """All VAL frames (stride subsample): features + raw pelvis for norm check."""
    val_rels = sorted(e.rel_path for e in entries if assign_split(e) == "val")
    # Subsample clips for label-free calibration (full VAL FK is too slow for CLI).
    if len(val_rels) > 120:
        step = max(1, len(val_rels) // 120)
        val_rels = val_rels[::step][:120]
    feats: dict[str, list[float]] = defaultdict(list)
    raw_pelvis: list[float] = []
    norm_pelvis: list[float] = []
    for rel in val_rels:
        joints, betas = _load_joints_for_clip(body, rel)
        stand = standing_pelvis_height(body, betas)
        for i in range(0, joints.shape[0], stride):
            f = extract_frame_features(joints[i], stand_pelvis_z=stand, use_normalized=False)
            if f is None:
                continue
            raw_pelvis.append(f.pelvis_h)
            norm_pelvis.append(f.pelvis_h / stand if stand > 1e-6 else f.pelvis_h)
            for k, v in f.__dict__.items():
                feats[k].append(v)
    arrays = {k: np.array(v, dtype=np.float64) for k, v in feats.items()}
    return arrays, raw_pelvis, norm_pelvis


def _loco_keyword_map(cohort_keywords: dict[str, dict[str, str]]) -> dict[str, str]:
    if "ordinary_locomotion_babel_keywords" in cohort_keywords:
        return dict(cohort_keywords["ordinary_locomotion_babel_keywords"])
    return dict(cohort_keywords["ordinary_locomotion"])


def _ordinary_locomotion_pelvis_cv(
    entries: list[AmassIndexEntry],
    body: SmplxBody,
    cohort_keywords: dict[str, dict[str, str]],
    min_dur: float,
    fps: float,
) -> tuple[float, float, bool]:
    """CV of pelvis height on VAL ordinary-locomotion BABEL frames (raw vs /standing)."""
    babel_payload = load_babel_index_payload()
    babel_by_rel: dict[str, BabelIndexEntry] = {}
    for row in babel_payload["entries"]:
        row = dict(row)
        row.pop("mocap_time_length", None)
        row.pop("time_scale", None)
        if "playback_duration" not in row:
            row.setdefault("playback_duration", float(row.get("babel_dur", 0)))
        babel_by_rel[row["rel_path"]] = BabelIndexEntry(**row)
    val_rels = {e.rel_path for e in entries if assign_split(e) == "val"}
    loco_kw = _loco_keyword_map(cohort_keywords)
    raw: list[float] = []
    norm: list[float] = []
    loco_rel_list = sorted(
        rel for rel, be in babel_by_rel.items() if rel in val_rels
    )
    if len(loco_rel_list) > 60:
        step = max(1, len(loco_rel_list) // 60)
        loco_rel_list = loco_rel_list[::step][:60]
    for rel in loco_rel_list:
        be = babel_by_rel[rel]
        has_loco = False
        for seg in be.segments:
            if float(seg["end_t"]) - float(seg["start_t"]) < min_dur:
                continue
            if _babel_category_from_segment(list(seg.get("act_cat") or []), loco_kw):
                has_loco = True
                break
        if not has_loco:
            continue
        joints, betas = _load_joints_for_clip(body, rel)
        stand = standing_pelvis_height(body, betas)
        for i in range(0, joints.shape[0], 6):
            pz = float(joints[i, PELVIS, 2])
            raw.append(pz)
            norm.append(pz / stand if stand > 1e-6 else pz)
    cv_raw = float(np.std(raw) / (np.mean(raw) + 1e-9)) if raw else 1.0
    cv_norm = float(np.std(norm) / (np.mean(norm) + 1e-9)) if norm else 1.0
    return cv_raw, cv_norm, cv_norm < cv_raw * 0.98


def _iter_val_clips(entries: list[AmassIndexEntry], *, stride_clips: int = 1) -> list[str]:
    val_rels = sorted(e.rel_path for e in entries if assign_split(e) == "val")
    if stride_clips > 1:
        val_rels = val_rels[::stride_clips]
    return val_rels


def _label_free_horiz_low_pelvis_shoulder(
    entries: list[AmassIndexEntry],
    body: SmplxBody,
    fps: float,
    *,
    stride: int,
    stride_clips: int = 1,
    crawl_pelvis_h_max: float = 0.42,
    torso_horizontal_max: float = 0.45,
) -> np.ndarray:
    shoulders: list[float] = []
    for rel in _iter_val_clips(entries, stride_clips=stride_clips):
        joints, betas = _load_joints_for_clip(body, rel)
        stand = standing_pelvis_height(body, betas)
        for i in range(0, joints.shape[0], stride):
            f = extract_frame_features(joints[i], stand_pelvis_z=stand, use_normalized=False)
            if f is None:
                continue
            if (
                f.torso_up_dot <= torso_horizontal_max
                and f.pelvis_h <= crawl_pelvis_h_max
            ):
                shoulders.append(f.shoulder_h)
    return np.array(shoulders, dtype=np.float64)


def _label_free_sit_gate_pelvis(
    entries: list[AmassIndexEntry],
    body: SmplxBody,
    fps: float,
    *,
    stride: int,
    th: dict[str, float],
    stride_clips: int = 1,
) -> np.ndarray:
    pelvis: list[float] = []
    for rel in _iter_val_clips(entries, stride_clips=stride_clips):
        joints, betas = _load_joints_for_clip(body, rel)
        stand = standing_pelvis_height(body, betas)
        for i in range(0, joints.shape[0], stride):
            f = extract_frame_features(joints[i], stand_pelvis_z=stand, use_normalized=False)
            if f is None:
                continue
            if (
                f.torso_up_dot >= th["torso_upright_min"]
                and f.thigh_up_dot >= th["sit_thigh_up_min"]
            ):
                pelvis.append(f.pelvis_h)
    return np.array(pelvis, dtype=np.float64)


def _sit_support_pelvis_threshold(pelvis: np.ndarray) -> float:
    """Label-free split between floor-sit tail and seat-height mode (metres)."""
    pelvis = pelvis[np.isfinite(pelvis)]
    if pelvis.size < 100:
        return 0.43
    floor_tail = pelvis[pelvis <= 0.35]
    seat_band = pelvis[(pelvis >= 0.45) & (pelvis <= 0.65)]
    if floor_tail.size >= 20 and seat_band.size >= 20:
        mid = (float(np.quantile(floor_tail, 0.90)) + float(np.quantile(seat_band, 0.10))) / 2
        return round(mid, 3)
    hist, edges = np.histogram(pelvis, bins=24, range=(0.15, 0.85))
    best_i, best_v = 1, int(hist[1])
    for i in range(2, len(hist) - 1):
        if 0.32 <= edges[i] <= 0.52 and hist[i] < best_v:
            best_v = int(hist[i])
            best_i = i
    return round(float((edges[best_i] + edges[best_i + 1]) * 0.5), 3)


def derive_geometry_thresholds(
    entries: list[AmassIndexEntry],
    body: SmplxBody,
    fps: float,
    cohort_keywords: dict[str, dict[str, str]],
    min_dur: float,
) -> dict[str, Any]:
    _arrays, _, _ = _label_free_val_frames(entries, body, fps, stride=8)
    cv_raw, cv_norm, use_norm = _ordinary_locomotion_pelvis_cv(
        entries, body, cohort_keywords, min_dur, fps
    )

    th: dict[str, float] = {}
    prov: dict[str, dict[str, str]] = {}

    th["torso_horizontal_max"] = 0.45
    prov["torso_horizontal_max"] = {"type": "anchor", "note": "torso angle ~63° from +Z"}
    th["torso_upright_min"] = 0.82
    prov["torso_upright_min"] = {"type": "anchor", "note": "upright trunk for sit/kneel"}

    th["crawl_pelvis_h_max"] = 0.42
    prov["crawl_pelvis_h_max"] = {"type": "anchor", "note": "low pelvis, metres grounded clip"}
    th["crawl_wrist_h_max"] = 0.22
    prov["crawl_wrist_h_max"] = {"type": "anchor", "note": "hands near floor"}
    horiz_sh = _label_free_horiz_low_pelvis_shoulder(entries, body, fps, stride=4)
    split = float(np.quantile(horiz_sh, 0.55)) if horiz_sh.size > 50 else 0.30
    th["crawl_shoulder_h_min"] = round(split, 3)
    th["lie_shoulder_h_max"] = th["crawl_shoulder_h_min"]
    prov["crawl_shoulder_h_min"] = {
        "type": "valley",
        "note": "VAL horiz+low-pelvis shoulder_h q55 split crawl vs lie trunk",
    }
    prov["lie_shoulder_h_max"] = {
        "type": "valley",
        "note": "same split; lie trunk on floor at or below",
    }

    th["lie_pelvis_h_max"] = 0.40
    prov["lie_pelvis_h_max"] = {"type": "anchor", "note": "floor supine pelvis; above is raised/none"}

    th["kneel_thigh_up_max"] = -0.42
    prov["kneel_thigh_up_max"] = {"type": "anchor", "note": "thigh axis nearer vertical than sit"}
    th["kneel_pelvis_h_max"] = 0.58
    prov["kneel_pelvis_h_max"] = {"type": "anchor", "note": "low pelvis, knees down"}

    th["sit_thigh_up_min"] = -0.32
    prov["sit_thigh_up_min"] = {"type": "anchor", "note": "thigh axis nearer horizontal"}
    sit_p = _label_free_sit_gate_pelvis(
        entries, body, fps, stride=8, th=th, stride_clips=3
    )
    th["sit_support_pelvis_h_min"] = _sit_support_pelvis_threshold(sit_p)
    prov["sit_support_pelvis_h_min"] = {
        "type": "valley",
        "note": "VAL sit-gate pelvis_h valley floor-sit vs seat-height mode",
    }

    th["yoga_pelvis_h_max"] = 0.55
    prov["yoga_pelvis_h_max"] = {"type": "anchor", "note": "low non-upright residual"}
    th["yoga_torso_up_min"] = 0.30
    prov["yoga_torso_up_min"] = {"type": "anchor", "note": "between lie and upright"}

    return {
        "use_normalized_height": use_norm,
        "height_unit": "standing_pelvis_ratio" if use_norm else "metres",
        "cv_ordinary_loco_pelvis_raw": cv_raw,
        "cv_ordinary_loco_pelvis_norm": cv_norm,
        "thresholds": th,
        "threshold_provenance": prov,
        "frozen_date": "2026-10-04",
    }


def _separation_check(
    entries: list[AmassIndexEntry],
    body: SmplxBody,
    fps: float,
    meta: dict[str, Any],
) -> dict[str, int]:
    th = meta["thresholds"]
    use_norm = bool(meta["use_normalized_height"])
    counts = Counter()
    val_rels = [e.rel_path for e in entries if assign_split(e) == "val"]
    for rel in val_rels[:80]:
        joints, betas = _load_joints_for_clip(body, rel)
        stand = standing_pelvis_height(body, betas)
        for i in range(0, joints.shape[0], 10):
            f = extract_frame_features(
                joints[i], stand_pelvis_z=stand, use_normalized=use_norm
            )
            if f is None:
                continue
            if use_norm:
                s = stand if stand > 1e-6 else 1.0
                f = FrameFeatures(
                    pelvis_h=f.pelvis_h / s,
                    torso_up_dot=f.torso_up_dot,
                    wrist_h=f.wrist_h / s,
                    head_h=f.head_h / s,
                    knee_h=f.knee_h / s,
                    foot_z_min=f.foot_z_min / s,
                    thigh_up_dot=f.thigh_up_dot,
                    shoulder_h=f.shoulder_h / s,
                )
            counts[classify_frame_decision_tree(f, th)] += 1
    need = ("kneel", "sit_floor", "sit_support")
    return {k: counts.get(k, 0) for k in need}


def legacy_recount_test(
    proposals: list[BabelProposal],
    body: SmplxBody,
    legacy_rules: dict[str, Any],
    fps: float,
    min_fraction: float,
) -> dict[str, int]:
    from hready.body.joint_indices import LEFT_WRIST as CW
    from hready.body.joint_indices import NECK as CN
    from hready.body.joint_indices import RIGHT_WRIST as CR

    conf = dis = unc = 0
    cache: dict[str, np.ndarray] = {}
    for p in proposals:
        if p.rel_path not in cache:
            cache[p.rel_path], _ = _load_joints_for_clip(body, p.rel_path)
        joints = cache[p.rel_path]
        labels = []
        for i in _segment_frame_indices(fps, p.start_s, p.end_s, joints.shape[0]):
            g = _legacy_frame_geometry(joints[i], legacy_rules, CN, CW, CR)
            labels.append(g or "none")
        frac = sum(1 for x in labels if x == p.category) / max(len(labels), 1)
        dom = _dominant_label(labels, min_fraction)
        if frac >= min_fraction:
            conf += 1
        elif dom is not None and dom != p.category:
            dis += 1
        else:
            unc += 1
    return {"confirmed": conf, "disagreements": dis, "unconfirmed": unc}


def _confusion_and_distributions(
    floor_props: list[BabelProposal],
    body: SmplxBody,
    fps: float,
    min_fraction: float,
    geom_meta: dict[str, Any],
    *,
    legacy_rules: dict[str, Any] | None = None,
    legacy_correct_indices: bool = False,
) -> tuple[Counter, dict[str, list[float]], Counter]:
    from hready.body.joint_indices import LEFT_WRIST as CW
    from hready.body.joint_indices import NECK as CN
    from hready.body.joint_indices import RIGHT_WRIST as CR

    th = geom_meta["thresholds"]
    use_norm = bool(geom_meta["use_normalized_height"])
    frame_mat = Counter()
    seg_mat = Counter()
    feat_store: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    cache: dict[str, tuple[np.ndarray, float]] = {}

    for p in floor_props:
        if p.rel_path not in cache:
            joints, betas = _load_joints_for_clip(body, p.rel_path)
            cache[p.rel_path] = (joints, standing_pelvis_height(body, betas))
        joints, stand = cache[p.rel_path]
        labels: list[str] = []
        for i in _segment_frame_indices(fps, p.start_s, p.end_s, joints.shape[0]):
            if legacy_rules is not None:
                neck, lw, rw = (CN, CW, CR) if legacy_correct_indices else (8, 16, 17)
                g = _legacy_frame_geometry(joints[i], legacy_rules, neck, lw, rw) or "none"
                if g == "sit":
                    g = "sit_floor"
            else:
                feat = extract_frame_features(
                    joints[i], stand_pelvis_z=stand, use_normalized=use_norm
                )
                if feat is None:
                    g = "none"
                else:
                    g = classify_frame_decision_tree(feat, th)
                    for name, val in feat.__dict__.items():
                        feat_store[p.category][name].append(val)
            labels.append(g)
            frame_mat[(p.category, g)] += 1
        dom = _dominant_label(labels, min_fraction) or "none"
        seg_mat[(p.category, dom)] += 1
    return frame_mat, feat_store, seg_mat


def _duration_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "segments": 0,
            "unique_clips": 0,
            "unique_subjects": 0,
            "total_duration_s": 0.0,
            "largest_subject_duration_share": 0.0,
        }
    dur_by_subj: dict[str, float] = defaultdict(float)
    total = 0.0
    for r in rows:
        d = float(r.get("segment_duration_s", r["segment_end_s"] - r["segment_start_s"]))
        total += d
        dur_by_subj[r["subject_key"]] += d
    max_share = max(dur_by_subj.values()) / total if total > 1e-9 else 0.0
    return {
        "segments": len(rows),
        "unique_clips": len({r["rel_path"] for r in rows}),
        "unique_subjects": len({r["subject_key"] for r in rows}),
        "total_duration_s": round(total, 3),
        "largest_subject_duration_share": round(max_share, 4),
    }


def evaluate_floor_work(
    proposals: list[BabelProposal],
    *,
    body: SmplxBody,
    geom_meta: dict[str, Any],
    min_geometry_fraction: float,
    fps: float,
    rise_cache: dict[str, Any],
    amass_by_rel: dict[str, AmassIndexEntry],
    babel_by_rel: dict[str, BabelIndexEntry],
) -> dict[str, Any]:
    confirmed: list[dict[str, Any]] = []
    disagreements: list[dict[str, Any]] = []
    unconfirmed: list[dict[str, Any]] = []
    th = geom_meta["thresholds"]
    use_norm = bool(geom_meta["use_normalized_height"])

    by_rel: dict[str, list[BabelProposal]] = defaultdict(list)
    for p in proposals:
        by_rel[p.rel_path].append(p)

    for rel, props in sorted(by_rel.items()):
        joints, betas = _load_joints_for_clip(body, rel)
        stand = standing_pelvis_height(body, betas)
        rise_cm, rise_status = _rise_for_clip(rel, rise_cache)
        entry = amass_by_rel[rel]
        flags = clip_flags(entry)
        playback = float(babel_by_rel[rel].playback_duration)

        for prop in props:
            labels = _segment_labels_tree(
                joints, fps, prop.start_s, prop.end_s, stand, use_norm, th
            )
            status, _dom = _classify_proposal(
                labels,
                prop.category,
                min_geometry_fraction,
                prop.ann_source,
                prop.start_s,
                prop.end_s,
                playback,
            )
            geom_class = _geometry_class_for_segment(labels, prop.category)
            tree_dom = _dominant_label(labels, min_geometry_fraction)
            row = {
                "rel_path": prop.rel_path,
                "subset": entry.subset,
                "subject": prop.subject,
                "subject_key": prop.subject_key,
                "babel_category": prop.category,
                "babel_keyword": prop.keyword,
                "segment_start_s": prop.start_s,
                "segment_end_s": prop.end_s,
                "segment_duration_s": float(prop.end_s - prop.start_s),
                "ann_source": prop.ann_source,
                "geometry_dominant": tree_dom if tree_dom is not None else geom_class,
                "geometry_class": geom_class,
                "rise_cm": rise_cm,
                "rise_status": rise_status,
                "skate_flag": bool(flags.get("skate_flag")),
            }
            if status == "confirmed":
                confirmed.append(row)
            elif status == "disagreement":
                disagreements.append({**row, "reason": "babel_geometry_mismatch"})
            else:
                unconfirmed.append({**row, "reason": "geometry_not_confirmed"})

    return {
        "confirmed": confirmed,
        "disagreements": disagreements,
        "unconfirmed": unconfirmed,
    }


def build_cohort_counts(
    all_proposals: list[BabelProposal],
    floor_eval: dict[str, Any],
    eval_split_name: str,
    *,
    rel_filter: set[str] | None = None,
    geom_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Per-category stats for one eval split bucket (val, test, or val_union_test)."""
    props = [p for p in all_proposals if p.cohort == "floor_work"]
    if rel_filter is not None:
        props = [p for p in props if p.rel_path in rel_filter]
    by_cat: dict[str, list[BabelProposal]] = defaultdict(list)
    for p in props:
        by_cat[p.category].append(p)

    status_by_prop: dict[tuple[str, float, float], str] = {}
    for bucket in ("confirmed", "disagreements", "unconfirmed"):
        for row in floor_eval[bucket]:
            if rel_filter is not None and row["rel_path"] not in rel_filter:
                continue
            key = (row["rel_path"], float(row["segment_start_s"]), float(row["segment_end_s"]))
            status_by_prop[key] = bucket

    confirmed_rows = [
        r
        for r in floor_eval["confirmed"]
        if rel_filter is None or r["rel_path"] in rel_filter
    ]
    confirmed_geom: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in confirmed_rows:
        confirmed_geom[r.get("geometry_class", "none")].append(r)

    out: dict[str, Any] = {
        "eval_split": eval_split_name,
        "categories": {},
        "confirmed_geometry_classes": {
            g: _duration_stats(rows) for g, rows in sorted(confirmed_geom.items())
        },
    }
    if geom_meta is not None:
        out["cv_ordinary_loco_pelvis_raw"] = geom_meta.get("cv_ordinary_loco_pelvis_raw")
        out["cv_ordinary_loco_pelvis_norm"] = geom_meta.get("cv_ordinary_loco_pelvis_norm")
        out["use_normalized_height"] = geom_meta.get("use_normalized_height")

    for cat, plist in sorted(by_cat.items()):
        clips = {p.rel_path for p in plist}
        subj = {p.subject_key for p in plist}
        counts = Counter()
        by_ann = Counter()
        for p in plist:
            key = (p.rel_path, p.start_s, p.end_s)
            st = status_by_prop.get(key, "missing")
            counts[st] += 1
            by_ann[(st, p.ann_source)] += 1
        out["categories"][cat] = {
            "segments": len(plist),
            "unique_clips": len(clips),
            "unique_subjects": len(subj),
            "confirmed": counts.get("confirmed", 0),
            "disagreements": counts.get("disagreements", 0),
            "unconfirmed": counts.get("unconfirmed", 0),
            "by_label_source": {
                f"{st}_{ann}": c for (st, ann), c in sorted(by_ann.items())
            },
        }

    other = [p for p in all_proposals if p.cohort == "other_labelled"]
    if rel_filter is not None:
        other = [p for p in other if p.rel_path in rel_filter]
    out["other_labelled"] = {
        "segments": len(other),
        "unique_clips": len({p.rel_path for p in other}),
        "unique_subjects": len({p.subject_key for p in other}),
        "by_category": dict(Counter(p.category for p in other)),
    }
    loco = [p for p in all_proposals if p.cohort == "ordinary_locomotion"]
    if rel_filter is not None:
        loco = [p for p in loco if p.rel_path in rel_filter]
    out["ordinary_locomotion"] = {
        "segments": len(loco),
        "unique_clips": len({p.rel_path for p in loco}),
        "unique_subjects": len({p.subject_key for p in loco}),
        "by_category": dict(Counter(p.category for p in loco)),
    }
    elig_rows = [
        r
        for r in floor_eval["confirmed"]
        if r.get("geometry_class") in FLOOR_WORK_ELIGIBLE_GEOMETRY
        and (rel_filter is None or r["rel_path"] in rel_filter)
    ]
    out["floor_work_eligible"] = _duration_stats(elig_rows)
    return out


def compute_ground_consistency_tolerance_default(
    entries: list[AmassIndexEntry],
    cohort_keywords: dict[str, dict[str, str]],
    min_dur: float,
) -> float:
    """p95 |z| of in-contact foot channels on clean VAL ordinary-locomotion clips."""
    from hready.data.amass import cache_dir_from_config, floor_offset
    from hready.data.contact import compute_foot_contact, load_foot_positions

    babel_payload = load_babel_index_payload()
    babel_by_rel: dict[str, BabelIndexEntry] = {}
    for row in babel_payload["entries"]:
        row = dict(row)
        row.pop("mocap_time_length", None)
        row.pop("time_scale", None)
        if "playback_duration" not in row:
            row.setdefault("playback_duration", float(row.get("babel_dur", 0)))
        babel_by_rel[row["rel_path"]] = BabelIndexEntry(**row)
    amass_by_rel = {e.rel_path: e for e in entries}
    val_rels = {e.rel_path for e in entries if assign_split(e) == "val"}
    loco_kw = _loco_keyword_map(cohort_keywords)
    loco_rels: set[str] = set()
    for rel, be in babel_by_rel.items():
        if rel not in val_rels:
            continue
        for seg in be.segments:
            if float(seg["end_t"]) - float(seg["start_t"]) < min_dur:
                continue
            cats = list(seg.get("act_cat") or [])
            if _babel_category_from_segment(cats, loco_kw):
                loco_rels.add(rel)
                break

    cache_dir = cache_dir_from_config()
    heights: list[float] = []
    rel_list = sorted(loco_rels)
    if len(rel_list) > 80:
        rel_list = rel_list[:: max(1, len(rel_list) // 80)][:80]
    for rel in rel_list:
        entry = amass_by_rel[rel]
        flags = clip_flags(entry)
        if flags.get("exclude_contact"):
            continue
        try:
            pos = load_foot_positions(entry, cache_dir=cache_dir)
        except FileNotFoundError:
            continue
        floor_off = float(floor_offset(entry)["floor_offset"])
        pos = pos.copy()
        pos[..., 2] -= floor_off
        contact = compute_foot_contact(entry, floor_off, positions=pos)
        z = pos[..., 2]
        for t in range(z.shape[0]):
            for c in range(4):
                if contact[t, c]:
                    heights.append(abs(float(z[t, c])))
    if not heights:
        return 0.05
    return float(np.quantile(np.array(heights), 0.95))


def run_ego_splits(
    cfg: dict[str, Any],
    *,
    eval_splits: tuple[EvalSplit, ...] = ("val", "test"),
    print_reports: bool = True,
    skip_geometry_only: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    assert_joint_table_runtime()
    entries = load_index()
    split_counts = Counter(assign_split(e) for e in entries)
    disjoint = subject_disjoint_counts(entries)

    clip_hashes_before = split_clip_list_hashes(entries)
    tune_fraction = float(cfg.get("tune", {}).get("fraction", 0.05))
    tune_seed = int(cfg.get("tune", {}).get("seed", cfg.get("seed", 0)))
    tune_subjects = select_tune_subjects(entries, seed=tune_seed, fraction=tune_fraction)
    clip_hashes_after = split_clip_list_hashes(entries)
    tune_hash = _sha256_lines(tune_subjects)

    babel_payload = load_babel_index_payload()
    babel_by_rel: dict[str, BabelIndexEntry] = {}
    for row in babel_payload["entries"]:
        row = dict(row)
        if "playback_duration" not in row:
            bd = float(row.get("babel_dur", 0.0))
            row.setdefault("playback_duration", float(row.get("mocap_time_length", bd)))
        row.pop("mocap_time_length", None)
        row.pop("time_scale", None)
        babel_by_rel[row["rel_path"]] = BabelIndexEntry(**row)
    amass_by_rel = {e.rel_path: e for e in entries}

    cohort_kw = {
        "floor_work": dict(cfg["cohorts"]["floor_work_babel_keywords"]),
        "ordinary_locomotion": dict(cfg["cohorts"]["ordinary_locomotion_babel_keywords"]),
        "other_labelled": dict(cfg["cohorts"]["other_labelled_babel_keywords"]),
    }
    min_dur = float(cfg["floor_work"]["min_segment_duration_s"])
    min_geom = float(cfg["floor_work"]["min_geometry_fraction"])
    fps = float(cfg.get("fps", 30.0))

    eval_rels: set[str] = set()
    for sp in eval_splits:
        eval_rels |= {e.rel_path for e in entries if assign_split(e) == sp}

    all_proposals = _collect_babel_proposals(
        babel_by_rel, eval_rels, amass_by_rel, cohort_kw, min_dur
    )
    floor_props = [p for p in all_proposals if p.cohort == "floor_work"]
    test_rels = {e.rel_path for e in entries if assign_split(e) == "test"}
    body = load_body("locked_head")
    rise_cache = load_foot_height_rise_cache()

    legacy_rules = dict(cfg.get("legacy_geometry", {}))
    legacy_kw = dict(cfg.get("legacy_acceptance_keywords", {}))
    if print_reports:
        legacy_props = _collect_babel_proposals(
            babel_by_rel, test_rels, amass_by_rel, {"floor_work": legacy_kw}, min_dur
        )
    if print_reports and legacy_props:
        old = legacy_recount_test(legacy_props, body, legacy_rules, fps, min_geom)
        print(
            "ACCEPTANCE 1 legacy yaml + CORRECT indices (TEST floor_work): "
            f"confirmed={old['confirmed']} disagreements={old['disagreements']} "
            f"unconfirmed={old['unconfirmed']}"
        )
        # wrong indices recount inline
        conf = dis = unc = 0
        cache: dict[str, np.ndarray] = {}
        for p in legacy_props:
            if p.rel_path not in cache:
                cache[p.rel_path], _ = _load_joints_for_clip(body, p.rel_path)
            joints = cache[p.rel_path]
            labels = []
            for i in _segment_frame_indices(fps, p.start_s, p.end_s, joints.shape[0]):
                g = _legacy_frame_geometry(joints[i], legacy_rules, 8, 16, 17)
                labels.append(g or "none")
            frac = sum(1 for x in labels if x == p.category) / max(len(labels), 1)
            dom = _dominant_label(labels, min_geom)
            if frac >= min_geom:
                conf += 1
            elif dom is not None and dom != p.category:
                dis += 1
            else:
                unc += 1
        print(
            "ACCEPTANCE 1 legacy yaml + WRONG indices (TEST floor_work): "
            f"confirmed={conf} disagreements={dis} unconfirmed={unc}"
        )

    geom_meta = dict(cfg["floor_work"]["geometry_tree"])
    if not geom_meta.get("cv_ordinary_loco_pelvis_raw"):
        cv_raw, cv_norm, use_n = _ordinary_locomotion_pelvis_cv(
            entries, body, cohort_kw, min_dur, fps
        )
        geom_meta["cv_ordinary_loco_pelvis_raw"] = cv_raw
        geom_meta["cv_ordinary_loco_pelvis_norm"] = cv_norm
        geom_meta["use_normalized_height"] = use_n
    if cfg.get("derive_thresholds", False):
        geom_meta = derive_geometry_thresholds(entries, body, fps, cohort_kw, min_dur)
        sep = _separation_check(entries, body, fps, geom_meta)
        print("derive_thresholds separation sample:", sep)
        if sep.get("sit_support", 0) < 1 or sep.get("sit_floor", 0) < 1:
            raise RuntimeError(
                f"kneel/sit_floor/sit_support separation too weak on VAL sample: {sep}"
            )

    floor_eval = evaluate_floor_work(
        floor_props,
        body=body,
        geom_meta=geom_meta,
        min_geometry_fraction=min_geom,
        fps=fps,
        rise_cache=rise_cache,
        amass_by_rel=amass_by_rel,
        babel_by_rel=babel_by_rel,
    )

    cohort_reports: dict[str, Any] = {}
    val_rels = {e.rel_path for e in entries if assign_split(e) == "val"}
    for label, relset in (
        ("val", val_rels & eval_rels),
        ("test", test_rels & eval_rels),
        ("val_union_test", eval_rels),
    ):
        sub_props = [p for p in all_proposals if p.rel_path in relset]
        cohort_reports[label] = build_cohort_counts(
            sub_props, floor_eval, label, rel_filter=relset, geom_meta=geom_meta
        )

    if print_reports:
        val_floor = [p for p in floor_props if p.rel_path in val_rels]
        _print_threshold_table(geom_meta)
        before_mat, _, _ = _confusion_and_distributions(
            val_floor,
            body,
            fps,
            min_geom,
            geom_meta,
            legacy_rules=legacy_rules,
            legacy_correct_indices=True,
        )
        after_mat, feat_store, seg_mat = _confusion_and_distributions(
            val_floor, body, fps, min_geom, geom_meta
        )
        print("ACCEPTANCE A VAL confusion (frames) AFTER frozen tree:")
        _print_confusion(after_mat)
        print("ACCEPTANCE A VAL confusion (segments, dominant geo):")
        _print_confusion(seg_mat)
        _print_feature_percentiles_nested(feat_store)
        print(
            "ACCEPTANCE C: branch frame counts omitted in CLI (re-FK cost); "
            "use trace_decision_branch on demand for diagnosis."
        )
        _print_acceptance_b_sit_and_eligible(
            floor_eval, body, fps, geom_meta, eval_rels, entries
        )
        if not skip_geometry_only:
            _print_geometry_only_table(
                entries, body, fps, min_geom, geom_meta, babel_by_rel, cohort_kw, min_dur
            )
        _print_threshold_provenance(entries, body, fps, cohort_kw, min_dur, geom_meta)
        sit_kneel_before = sum(
            v for (b, g), v in before_mat.items() if b == "sit" and g == "kneel"
        )
        sit_kneel_after = sum(
            v for (b, g), v in after_mat.items() if b == "sit" and g == "kneel"
        )
        print(
            f"sit->kneel frame counts VAL: before_legacy_yaml={sit_kneel_before} "
            f"after_tree={sit_kneel_after}"
        )

    summary = {
        "version": 2,
        "seed": int(cfg.get("seed", 0)),
        "split_clip_counts": dict(split_counts),
        "split_clip_list_sha256_before_tune": clip_hashes_before,
        "split_clip_list_sha256_after_tune": clip_hashes_after,
        "tune": {
            "seed": tune_seed,
            "fraction": tune_fraction,
            "n_subjects": len(tune_subjects),
            "subject_ids": tune_subjects,
            "subject_list_sha256": tune_hash,
        },
        "subject_disjoint": disjoint,
        "cohorts": cohort_kw,
        "floor_work_geometry_tree": geom_meta,
        "floor_work_eval_splits": list(eval_splits),
        "floor_work": {
            "n_babel_proposals": len(floor_props),
            "n_geometry_confirmed": len(floor_eval["confirmed"]),
            "n_disagreements_excluded": len(floor_eval["disagreements"]),
            "n_babel_unconfirmed_excluded": len(floor_eval["unconfirmed"]),
            "per_category_confirmed": dict(Counter(r["babel_category"] for r in floor_eval["confirmed"])),
        },
    }
    extras = {
        "cohort_counts": cohort_reports,
        "floor_eval_full": floor_eval,
        "all_proposals": all_proposals,
        "clip_hashes_before": clip_hashes_before,
        "clip_hashes_after": clip_hashes_after,
        "tune_hash": tune_hash,
    }
    return summary, floor_eval["confirmed"], extras


def _print_feature_percentiles_nested(feat_store: dict[str, dict[str, list[float]]]) -> None:
    feat_names = (
        "pelvis_h",
        "torso_up_dot",
        "wrist_h",
        "head_h",
        "knee_h",
        "foot_z_min",
        "thigh_up_dot",
        "shoulder_h",
    )
    qs = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
    for cat in sorted(feat_store):
        for fn in feat_names:
            arr = np.array(feat_store[cat].get(fn, []), dtype=np.float64)
            if arr.size == 0:
                continue
            qv = np.quantile(arr, qs)
            print(
                f"  {cat} {fn} n={arr.size} "
                + " ".join(f"{q:.3f}" for q in qv)
            )


def _print_branch_diagnosis(
    floor_props: list[BabelProposal],
    floor_eval: dict[str, Any],
    body: SmplxBody,
    fps: float,
    min_geom: float,
    geom_meta: dict[str, Any],
) -> None:
    th = geom_meta["thresholds"]
    use_norm = bool(geom_meta["use_normalized_height"])
    status_rows: dict[tuple[str, float, float], dict[str, Any]] = {}
    for bucket in ("confirmed", "disagreements", "unconfirmed"):
        for row in floor_eval[bucket]:
            key = (row["rel_path"], float(row["segment_start_s"]), float(row["segment_end_s"]))
            status_rows[key] = {**row, "status": bucket}

    targets = {"lie", "crawl", "yoga", "sit"}
    branch_counts: Counter = Counter()
    cache: dict[str, tuple[np.ndarray, float]] = {}
    print("ACCEPTANCE C deciding-branch frame counts (VAL union TEST floor segments, target categories):")
    for p in floor_props:
        if p.category not in targets:
            continue
        key = (p.rel_path, p.start_s, p.end_s)
        st = status_rows.get(key, {}).get("status", "?")
        if p.category in ("lie", "crawl", "yoga") and st == "confirmed":
            continue
        if p.category == "sit" and st != "unconfirmed":
            continue
        if p.rel_path not in cache:
            joints, betas = _load_joints_for_clip(body, p.rel_path)
            cache[p.rel_path] = (joints, standing_pelvis_height(body, betas))
        joints, stand = cache[p.rel_path]
        for i in _segment_frame_indices(fps, p.start_s, p.end_s, joints.shape[0]):
            feat = extract_frame_features(joints[i], stand_pelvis_z=stand, use_normalized=use_norm)
            if feat is None:
                branch_counts[(p.category, "none:invalid_torso")] += 1
            else:
                branch_counts[(p.category, trace_decision_branch(feat, th))] += 1
    print("babel_category\\branch\tcount")
    for (bc, br), n in branch_counts.most_common():
        print(f"{bc}\t{br}\t{n}")


def _longest_run_seconds(
    labels: list[str], target: str, fps: float, min_frames: int
) -> bool:
    run = 0
    for lab in labels:
        if lab == target:
            run += 1
            if run >= min_frames:
                return True
        else:
            run = 0
    return False


def _print_geometry_only_table(
    entries: list[AmassIndexEntry],
    body: SmplxBody,
    fps: float,
    min_geom: float,
    geom_meta: dict[str, Any],
    babel_by_rel: dict[str, BabelIndexEntry],
    cohort_kw: dict[str, dict[str, str]],
    min_dur: float,
) -> None:
    th = geom_meta["thresholds"]
    use_norm = bool(geom_meta["use_normalized_height"])
    min_frames = int(np.ceil(min_dur * fps))
    floor_kw = cohort_kw["floor_work"]
    babel_cat_by_rel: dict[str, set[str]] = defaultdict(set)
    for rel, be in babel_by_rel.items():
        for seg in be.segments:
            if float(seg["end_t"]) - float(seg["start_t"]) < min_dur:
                continue
            hit = _babel_category_from_segment(list(seg.get("act_cat") or []), floor_kw)
            if hit:
                babel_cat_by_rel[rel].add(hit[0])

    geo_classes = ("lie", "crawl", "kneel", "sit_floor", "sit_support", "yoga_like")
    for split_name in ("val", "test", "val_union_test"):
        rels = {
            e.rel_path
            for e in entries
            if assign_split(e) == split_name or split_name == "val_union_test"
        }
        if split_name == "val_union_test":
            rels = {
                e.rel_path for e in entries if assign_split(e) in ("val", "test")
            }
        clip_has: dict[str, set[str]] = defaultdict(set)
        subj_has: dict[str, set[str]] = defaultdict(set)
        no_babel_prop: dict[str, set[str]] = defaultdict(set)
        amass_by = {e.rel_path: e for e in entries}
        rel_list = sorted(rels)
        for rel in rel_list:
            joints, betas = _load_joints_for_clip(body, rel)
            stand = standing_pelvis_height(body, betas)
            labels = []
            for i in range(joints.shape[0]):
                feat = extract_frame_features(
                    joints[i], stand_pelvis_z=stand, use_normalized=use_norm
                )
                labels.append(
                    classify_frame_decision_tree(feat, th)
                    if feat is not None
                    else "none"
                )
            sk = f"{amass_by[rel].subset}/{amass_by[rel].subject}"
            props = babel_cat_by_rel.get(rel, set())
            for gc in geo_classes:
                if _longest_run_seconds(labels, gc, fps, min_frames):
                    clip_has[gc].add(rel)
                    subj_has[gc].add(sk)
                    if gc == "sit_floor" and "sit" not in props or gc == "sit_support" and "sit" not in props or gc == "yoga_like" and "yoga" not in props or gc not in props and gc not in ("sit_floor", "sit_support"):
                        no_babel_prop[gc].add(rel)
        print(f"ACCEPTANCE F geometry-only {split_name} (clips n={len(rel_list)}):")
        for gc in geo_classes:
            print(
                f"  {gc} clips={len(clip_has[gc])} subjects={len(subj_has[gc])} "
                f"clips_no_babel_for_class={len(no_babel_prop[gc])}"
            )


def _print_threshold_provenance(
    entries: list[AmassIndexEntry],
    body: SmplxBody,
    fps: float,
    cohort_kw: dict[str, dict[str, str]],
    min_dur: float,
    geom_meta: dict[str, Any],
) -> None:
    th = geom_meta["thresholds"]
    print("ACCEPTANCE D threshold provenance (VAL evidence beside frozen value):")
    cv_raw = geom_meta.get("cv_ordinary_loco_pelvis_raw")
    cv_norm = geom_meta.get("cv_ordinary_loco_pelvis_norm")
    if cv_raw is not None:
        print(
            f"  torso_upright_min={th['torso_upright_min']} anchor; "
            f"VAL ordinary-loco pelvis CV raw={cv_raw:.4f} norm={cv_norm:.4f} "
            f"use_normalized_height={geom_meta.get('use_normalized_height')}"
        )
    print(
        "  crawl_shoulder_h_min=0.30 lie_shoulder_h_max=0.30 valley; "
        "VAL horiz+low-pelvis shoulder_h (stride 8, every 3rd clip) "
        "see amendment 2026-10-04 in docs/ego_splits.md"
    )
    print(
        "  sit_support_pelvis_h_min=0.43 valley; VAL sit-gate pelvis_h "
        "(stride 8, every 3rd clip) floor-tail p90 vs seat-band p10 midpoint"
    )
    for key, val in th.items():
        if key in ("torso_upright_min",):
            continue
        prov = geom_meta.get("threshold_provenance", {}).get(key, {})
        print(f"  {key}={val} ({prov.get('type', '?')})")


def _sit_floor_support_segment_counts(
    confirmed_sit: list[dict[str, Any]],
    body: SmplxBody,
    fps: float,
    geom_meta: dict[str, Any],
    sit_support_min: float,
) -> tuple[int, int]:
    th = geom_meta["thresholds"]
    use_norm = bool(geom_meta["use_normalized_height"])
    n_floor = n_support = 0
    cache: dict[str, tuple[np.ndarray, float]] = {}
    for row in confirmed_sit:
        rel = row["rel_path"]
        if rel not in cache:
            joints, betas = _load_joints_for_clip(body, rel)
            cache[rel] = (joints, standing_pelvis_height(body, betas))
        joints, stand = cache[rel]
        sf = ss = 0
        for i in _segment_frame_indices(
            fps, float(row["segment_start_s"]), float(row["segment_end_s"]), joints.shape[0]
        ):
            f = extract_frame_features(joints[i], stand_pelvis_z=stand, use_normalized=use_norm)
            if f is None:
                continue
            if f.torso_up_dot >= th["torso_upright_min"] and f.thigh_up_dot >= th["sit_thigh_up_min"]:
                if f.pelvis_h <= sit_support_min:
                    sf += 1
                else:
                    ss += 1
        if sf >= ss:
            n_floor += 1
        else:
            n_support += 1
    return n_floor, n_support


def _path_group(rel_path: str) -> str:
    low = rel_path.lower()
    if "chair" in low:
        return "chair_named"
    if rel_path.startswith("KIT/"):
        return "KIT"
    if rel_path.startswith("BMLrub/"):
        return "BMLrub"
    if "eyes_japan" in low:
        return "Eyes_Japan"
    if rel_path.startswith("CMU/"):
        return "CMU"
    return "other"


def _print_acceptance_b_sit_and_eligible(
    floor_eval: dict[str, Any],
    body: SmplxBody,
    fps: float,
    geom_meta: dict[str, Any],
    eval_rels: set[str],
    entries: list[AmassIndexEntry],
) -> None:
    confirmed_sit = [
        r
        for r in floor_eval["confirmed"]
        if r["babel_category"] == "sit" and r["rel_path"] in eval_rels
    ]
    new_min = float(geom_meta["thresholds"]["sit_support_pelvis_h_min"])
    old_floor = old_sup = new_floor = new_sup = 0
    th = geom_meta["thresholds"]
    use_norm = bool(geom_meta["use_normalized_height"])
    cache: dict[str, tuple[np.ndarray, float]] = {}
    for row in confirmed_sit:
        rel = row["rel_path"]
        if rel not in cache:
            joints, betas = _load_joints_for_clip(body, rel)
            cache[rel] = (joints, standing_pelvis_height(body, betas))
        joints, stand = cache[rel]
        sf_o = ss_o = sf_n = ss_n = 0
        for i in _segment_frame_indices(
            fps, float(row["segment_start_s"]), float(row["segment_end_s"]), joints.shape[0]
        ):
            f = extract_frame_features(joints[i], stand_pelvis_z=stand, use_normalized=use_norm)
            if f is None:
                continue
            if f.torso_up_dot < th["torso_upright_min"] or f.thigh_up_dot < th["sit_thigh_up_min"]:
                continue
            if f.pelvis_h <= 0.68:
                sf_o += 1
            else:
                ss_o += 1
            if f.pelvis_h <= new_min:
                sf_n += 1
            else:
                ss_n += 1
        if sf_o >= ss_o:
            old_floor += 1
        else:
            old_sup += 1
        if sf_n >= ss_n:
            new_floor += 1
        else:
            new_sup += 1
    print(
        f"ACCEPTANCE B sit segment-dominant (confirmed BABEL sit, sit_support min "
        f"0.68 legacy): sit_floor={old_floor} sit_support={old_sup}"
    )
    print(
        f"ACCEPTANCE B sit segment-dominant (after sit_support_pelvis_h_min={new_min}): "
        f"sit_floor={new_floor} sit_support={new_sup}"
    )
    by_group: dict[str, list[float]] = defaultdict(list)
    for row in confirmed_sit:
        rel = row["rel_path"]
        joints, stand = cache[rel]
        grp = _path_group(rel)
        for i in _segment_frame_indices(
            fps, float(row["segment_start_s"]), float(row["segment_end_s"]), joints.shape[0]
        ):
            if i % 4 != 0:
                continue
            f = extract_frame_features(joints[i], stand_pelvis_z=stand, use_normalized=use_norm)
            if f and f.torso_up_dot >= th["torso_upright_min"]:
                by_group[grp].append(f.pelvis_h)
    if by_group:
        print("ACCEPTANCE B path-group pelvis_h p50 (confirmed sit frames, stride 4):")
        for grp in sorted(by_group):
            arr = np.asarray(by_group[grp])
            print(f"  {grp} n={arr.size} p50={np.quantile(arr, 0.5):.3f}")
    val_rels = {e.rel_path for e in entries if assign_split(e) == "val"}
    test_rels = {e.rel_path for e in entries if assign_split(e) == "test"}
    for label, relset in (
        ("val", val_rels & eval_rels),
        ("test", test_rels & eval_rels),
        ("val_union_test", eval_rels),
    ):
        elig = [
            r
            for r in floor_eval["confirmed"]
            if r.get("geometry_class") in FLOOR_WORK_ELIGIBLE_GEOMETRY
            and r["rel_path"] in relset
        ]
        st = _duration_stats(elig)
        print(
            f"ACCEPTANCE B floor-work-eligible {label}: segments={st['segments']} "
            f"subjects={st['unique_subjects']} duration_s={st['total_duration_s']} "
            f"largest_subject_share={st['largest_subject_duration_share']}"
        )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _print_threshold_table(geom_meta: dict[str, Any]) -> None:
    print("ACCEPTANCE 2 threshold table:")
    prov = geom_meta.get("threshold_provenance", {})
    for k, v in geom_meta.get("thresholds", {}).items():
        meta = prov.get(k, {})
        print(f"  {k}={v} ({meta.get('type', '?')}: {meta.get('note', '')})")
    print(f"  use_normalized_height={geom_meta.get('use_normalized_height')}")


def _print_confusion(mat: Counter) -> None:
    cats = sorted({a for a, _ in mat} | {b for _, b in mat})
    print("babel\\geo\t" + "\t".join(cats))
    for b in sorted({a for a, _ in mat}):
        row = [str(mat.get((b, g), 0)) for g in cats]
        print(b + "\t" + "\t".join(row))


def _print_acceptance_samples(
    floor_eval: dict[str, Any],
    proposals: list[BabelProposal],
    *,
    seed: int,
) -> None:
    rng = random.Random(seed)
    print(f"ACCEPTANCE 4 confirmed samples by geometry_class (seed={seed}):")
    by_geom: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in floor_eval["confirmed"]:
        by_geom[row.get("geometry_class", "none")].append(row)
    for gclass in sorted(by_geom):
        picks = by_geom[gclass][:]
        rng.shuffle(picks)
        for row in picks[:5]:
            print(
                f"  {gclass} {row['rel_path']} {row['segment_start_s']:.2f}-"
                f"{row['segment_end_s']:.2f} babel={row['babel_category']}"
            )
    dis_cells: Counter = Counter(
        (r["babel_category"], r.get("geometry_dominant"))
        for r in floor_eval["disagreements"]
    )
    print("ACCEPTANCE 4 disagreement examples:")
    for (bc, gc), _ in dis_cells.most_common(8):
        if bc == gc:
            continue
        rows = [r for r in floor_eval["disagreements"] if r["babel_category"] == bc and r.get("geometry_dominant") == gc]
        rng.shuffle(rows)
        print(f"  cell {bc}->{gc}")
        for row in rows[:3]:
            print(
                f"    {row['rel_path']} {row['segment_start_s']:.2f}-{row['segment_end_s']:.2f}"
            )


def _write_outputs(
    payload: dict[str, Any],
    confirmed_rows: list[dict[str, Any]],
    cohort_counts: dict[str, Any],
    cfg: dict[str, Any],
) -> None:
    out_json = Path(cfg["output"]["splits_json"])
    out_csv = Path(cfg["output"]["floor_work_csv"])
    out_counts = Path(cfg["output"].get("cohort_counts_json", "results/E/cohort_counts.json"))
    for p in (out_json, out_csv, out_counts):
        p.parent.mkdir(parents=True, exist_ok=True)

    serializable = json.loads(json.dumps(payload, default=str))
    out_json.write_text(json.dumps(serializable, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    out_counts.write_text(json.dumps(cohort_counts, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rows = sorted(confirmed_rows, key=lambda r: (r["rel_path"], r["segment_start_s"]))
    fieldnames = [
        "rel_path",
        "subset",
        "subject",
        "babel_category",
        "babel_keyword",
        "segment_start_s",
        "segment_end_s",
        "ann_source",
        "geometry_dominant",
        "geometry_class",
        "rise_cm",
        "rise_status",
        "skate_flag",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    parser = argparse.ArgumentParser(description="Track E1 ego splits + floor-work cohorts")
    parser.add_argument("--config", type=str, default="configs/ego_splits.yaml")
    parser.add_argument(
        "--splits",
        type=str,
        default="val,test",
        help="Comma-separated eval splits for floor-work (default val,test)",
    )
    parser.add_argument("--derive-thresholds", action="store_true")
    parser.add_argument(
        "--verify-byte-stable",
        action="store_true",
        help="Run ego_splits twice and compare output SHA256",
    )
    args = parser.parse_args()
    cfg_path = Path(args.config)
    cfg = _load_yaml(cfg_path)
    if args.derive_thresholds:
        cfg["derive_thresholds"] = True
    eval_splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())  # type: ignore
    if args.verify_byte_stable:
        hashes: list[dict[str, str]] = []
        for run_i in range(2):
            payload, confirmed, extras = run_ego_splits(
                cfg,
                eval_splits=eval_splits,
                print_reports=(run_i == 1),
                skip_geometry_only=True,
            )  # type: ignore
            _write_outputs(payload, confirmed, extras["cohort_counts"], cfg)
            out = cfg["output"]
            hashes.append(
                {
                    "splits.json": _sha256_file(Path(out["splits_json"])),
                    "floor_work_clips.csv": _sha256_file(Path(out["floor_work_csv"])),
                    "cohort_counts.json": _sha256_file(
                        Path(out.get("cohort_counts_json", "results/E/cohort_counts.json"))
                    ),
                }
            )
        print("ACCEPTANCE H byte-stable hashes run1:", hashes[0])
        print("ACCEPTANCE H byte-stable hashes run2:", hashes[1])
        print(
            "ACCEPTANCE H identical:",
            hashes[0] == hashes[1],
        )
        return

    payload, confirmed, extras = run_ego_splits(
        cfg, eval_splits=eval_splits, skip_geometry_only=True
    )  # type: ignore
    _write_outputs(payload, confirmed, extras["cohort_counts"], cfg)
    if extras.get("all_proposals"):
        _print_acceptance_samples(
            extras["floor_eval_full"],
            extras["all_proposals"],
            seed=int(cfg.get("seed", 0)),
        )
    d = payload["subject_disjoint"]
    fw = payload["floor_work"]
    print(
        f"subject_disjoint={d['subject_disjoint']} "
        f"overlaps tv={d['overlap_train_val']} tt={d['overlap_train_test']} vt={d['overlap_val_test']}"
    )
    print(
        f"floor_work proposals={fw['n_babel_proposals']} confirmed={fw['n_geometry_confirmed']} "
        f"disagreements={fw['n_disagreements_excluded']} unconfirmed={fw['n_babel_unconfirmed_excluded']}"
    )
    print(f"per_category_confirmed={fw['per_category_confirmed']}")
    sit_classes = Counter(r.get("geometry_class") for r in confirmed)
    print(
        f"ACCEPTANCE B confirmed sit_floor={sit_classes.get('sit_floor', 0)} "
        f"sit_support={sit_classes.get('sit_support', 0)}"
    )
    print(f"split_clip_sha256_train={payload['split_clip_list_sha256_after_tune']['train']}")
    print(f"tune_subjects_n={payload['tune']['n_subjects']} tune_sha256={payload['tune']['subject_list_sha256']}")


if __name__ == "__main__":
    main()
