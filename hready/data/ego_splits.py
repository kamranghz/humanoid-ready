"""Track E1: subject split audit + BABEL+geometry floor-work test subset."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from hready.body.batch_forward import smpl_forward_bt
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

# SMPL-X body 22-joint layout (first 22 of SMPL-X joints tensor).
_PELVIS, _NECK = 0, 8
_L_WRIST, _R_WRIST = 16, 17


@dataclass(frozen=True)
class BabelProposal:
    rel_path: str
    subject: str
    category: str
    keyword: str
    start_s: float
    end_s: float
    ann_source: str


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def subject_disjoint_counts(entries: list[AmassIndexEntry]) -> dict[str, Any]:
    """Unique ``subset/subject`` folders per split; report pairwise intersections."""
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
    test_rels: set[str],
    amass_by_rel: dict[str, AmassIndexEntry],
    keyword_map: dict[str, str],
    min_dur_s: float,
) -> list[BabelProposal]:
    out: list[BabelProposal] = []
    for rel, be in babel_by_rel.items():
        if rel not in test_rels:
            continue
        entry = amass_by_rel[rel]
        for seg in be.segments:
            start_s = float(seg["start_t"])
            end_s = float(seg["end_t"])
            if end_s - start_s < min_dur_s:
                continue
            cats = list(seg.get("act_cat") or [])
            hit = _babel_category_from_segment(cats, keyword_map)
            if hit is None:
                continue
            cat, kw = hit
            out.append(
                BabelProposal(
                    rel_path=rel,
                    subject=entry.subject,
                    category=cat,
                    keyword=kw,
                    start_s=start_s,
                    end_s=end_s,
                    ann_source=be.ann_source,
                )
            )
    return out


def _frame_geometry_category(joints_t: np.ndarray, rules: dict[str, Any]) -> str | None:
    """Classify one frame into a floor-work geometry category (or None)."""
    pelvis_z = float(joints_t[_PELVIS, 2])
    torso = joints_t[_NECK] - joints_t[_PELVIS]
    norm = float(np.linalg.norm(torso))
    if norm < 1e-6:
        return None
    up_dot = float(torso[2] / norm)
    wrist_z = float((joints_t[_L_WRIST, 2] + joints_t[_R_WRIST, 2]) * 0.5)

    scores: list[tuple[str, bool]] = []
    for cat, th in rules.items():
        ok = True
        if "pelvis_z_max" in th and pelvis_z > float(th["pelvis_z_max"]):
            ok = False
        if "pelvis_z_min" in th and pelvis_z < float(th["pelvis_z_min"]):
            ok = False
        if "torso_up_dot_max" in th and up_dot > float(th["torso_up_dot_max"]):
            ok = False
        if "torso_up_dot_min" in th and up_dot < float(th["torso_up_dot_min"]):
            ok = False
        if "wrist_z_max" in th and wrist_z > float(th["wrist_z_max"]):
            ok = False
        scores.append((cat, ok))
    matched = [c for c, ok in scores if ok]
    if not matched:
        return None
    # Prefer more specific labels when multiple match.
    priority = ["lie", "crawl", "kneel", "sit", "crouch", "yoga"]
    for p in priority:
        if p in matched:
            return p
    return matched[0]


def _segment_geometry_labels(
    joints: np.ndarray,
    fps: float,
    start_s: float,
    end_s: float,
    rules: dict[str, Any],
) -> list[str]:
    t = joints.shape[0]
    i0 = max(0, int(np.floor(start_s * fps)))
    i1 = min(t, int(np.ceil(end_s * fps)))
    if i1 <= i0:
        return []
    return [_frame_geometry_category(joints[i], rules) or "none" for i in range(i0, i1)]


def _dominant_floor_label(labels: list[str], min_fraction: float) -> str | None:
    if not labels:
        return None
    counts = Counter(labels)
    counts.pop("none", None)
    if not counts:
        return None
    cat, n = counts.most_common(1)[0]
    if n / len(labels) < min_fraction:
        return None
    return cat


def _geometry_confirms_babel(
    labels: list[str], babel_cat: str, min_fraction: float
) -> tuple[bool, str | None]:
    if not labels:
        return False, None
    frac_babel = sum(1 for x in labels if x == babel_cat) / len(labels)
    if frac_babel >= min_fraction:
        return True, babel_cat
    dom = _dominant_floor_label(labels, min_fraction)
    return False, dom


def _rise_for_clip(rel_path: str, rise_cache: dict[str, Any]) -> tuple[float | None, str]:
    row = rise_cache.get("entries", {}).get(rel_path, {})
    return row.get("rise_cm"), row.get("status", "missing")


def build_floor_work_subset(
    proposals: list[BabelProposal],
    *,
    body: SmplxBody,
    geometry_rules: dict[str, Any],
    min_geometry_fraction: float,
    fps: float,
    rise_cache: dict[str, Any],
) -> dict[str, Any]:
    confirmed: list[dict[str, Any]] = []
    disagreements: list[dict[str, Any]] = []
    babel_unconfirmed: list[dict[str, Any]] = []
    per_cat = Counter()
    rise_gt3_clips: set[str] = set()
    rise_gt5_clips: set[str] = set()

    by_rel: dict[str, list[BabelProposal]] = defaultdict(list)
    for p in proposals:
        by_rel[p.rel_path].append(p)

    for rel, props in sorted(by_rel.items()):
        clip = load_clip(props[0].rel_path, ground=True)
        root = clip["root_orient"]
        body_aa = clip["pose_body"]
        transl = clip["transl"]
        betas = clip["betas"]
        with np.errstate(all="ignore"):
            tr = transl.astype(np.float32)[None]
            ro = root.astype(np.float32)[None]
            ba = body_aa.astype(np.float32)[None]
            be = np.asarray(betas, dtype=np.float32)[None]
            joints, _ = smpl_forward_bt(
                body,
                torch.as_tensor(tr),
                torch.as_tensor(ro),
                torch.as_tensor(ba),
                torch.as_tensor(be),
            )
        joints_np = joints[0].cpu().numpy()

        rise_cm, rise_status = _rise_for_clip(rel, rise_cache)

        for prop in props:
            labels = _segment_geometry_labels(
                joints_np, fps, prop.start_s, prop.end_s, geometry_rules
            )
            ok, dom = _geometry_confirms_babel(labels, prop.category, min_geometry_fraction)
            row_base = {
                "rel_path": prop.rel_path,
                "subject": prop.subject,
                "babel_category": prop.category,
                "babel_keyword": prop.keyword,
                "segment_start_s": prop.start_s,
                "segment_end_s": prop.end_s,
                "ann_source": prop.ann_source,
                "geometry_dominant": dom,
                "rise_cm": rise_cm,
                "rise_status": rise_status,
            }
            if ok:
                per_cat[prop.category] += 1
                if rise_cm is not None and float(rise_cm) > 3.0:
                    rise_gt3_clips.add(prop.rel_path)
                if rise_cm is not None and float(rise_cm) > 5.0:
                    rise_gt5_clips.add(prop.rel_path)
                confirmed.append(row_base)
            elif dom is not None and dom != prop.category:
                disagreements.append({**row_base, "reason": "babel_geometry_mismatch"})
            else:
                babel_unconfirmed.append({**row_base, "reason": "geometry_not_confirmed"})

    return {
        "n_babel_proposals": len(proposals),
        "n_geometry_confirmed": len(confirmed),
        "n_disagreements_excluded": len(disagreements),
        "n_babel_unconfirmed_excluded": len(babel_unconfirmed),
        "per_category_confirmed": dict(per_cat),
        "confirmed_rise_gt_3cm_clips": len(rise_gt3_clips),
        "confirmed_rise_gt_5cm_clips": len(rise_gt5_clips),
        "confirmed_clips": confirmed,
        "disagreements": disagreements,
        "babel_unconfirmed": babel_unconfirmed,
    }


def run_ego_splits(cfg: dict[str, Any]) -> dict[str, Any]:
    _ = cfg.get("paths_config", "configs/paths.yaml")
    entries = load_index()
    split_counts = Counter(assign_split(e) for e in entries)
    disjoint = subject_disjoint_counts(entries)

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
    test_rels = {e.rel_path for e in entries if assign_split(e) == "test"}

    kw_map = dict(cfg["floor_work_babel_keywords"])
    min_dur = float(cfg["floor_work"]["min_segment_duration_s"])
    proposals = _collect_babel_proposals(
        babel_by_rel, test_rels, amass_by_rel, kw_map, min_dur
    )

    body = load_body("locked_head")
    rise_cache = load_foot_height_rise_cache()
    floor = build_floor_work_subset(
        proposals,
        body=body,
        geometry_rules=dict(cfg["floor_work"]["geometry"]),
        min_geometry_fraction=float(cfg["floor_work"]["min_geometry_fraction"]),
        fps=float(cfg.get("fps", 30.0)),
        rise_cache=rise_cache,
    )

    # Fix clip_flags calls — need entry
    for bucket in ("confirmed_clips", "disagreements", "babel_unconfirmed"):
        for row in floor[bucket]:
            entry = amass_by_rel[row["rel_path"]]
            fl = clip_flags(entry)
            row["skate_flag"] = bool(fl.get("skate_flag"))
            row["subset"] = entry.subset

    summary = {
        "version": 1,
        "seed": int(cfg.get("seed", 0)),
        "split_clip_counts": dict(split_counts),
        "subject_disjoint": disjoint,
        "floor_work_babel_keywords": kw_map,
        "floor_work_rules": {
            "min_segment_duration_s": min_dur,
            "min_geometry_fraction": float(cfg["floor_work"]["min_geometry_fraction"]),
            "geometry": cfg["floor_work"]["geometry"],
        },
        "floor_work": {
            k: v
            for k, v in floor.items()
            if k not in ("confirmed_clips", "disagreements", "babel_unconfirmed")
        },
    }
    return summary, floor["confirmed_clips"]


def _write_outputs(payload: dict[str, Any], confirmed_rows: list[dict[str, Any]], cfg: dict[str, Any]) -> None:
    out_json = Path(cfg["output"]["splits_json"])
    out_csv = Path(cfg["output"]["floor_work_csv"])
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    serializable = json.loads(json.dumps(payload, default=str))
    out_json.write_text(json.dumps(serializable, indent=2, sort_keys=True) + "\n", encoding="utf-8")

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
    parser = argparse.ArgumentParser(description="Track E1 ego splits + floor-work subset")
    parser.add_argument("--config", type=str, default="configs/ego_splits.yaml")
    args = parser.parse_args()
    cfg_path = Path(args.config)
    cfg = _load_yaml(cfg_path)
    payload, confirmed = run_ego_splits(cfg)
    _write_outputs(payload, confirmed, cfg)
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


if __name__ == "__main__":
    main()
