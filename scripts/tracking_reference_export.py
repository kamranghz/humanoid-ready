"""Select validation windows for the tracking study and export their reference for the SMPL-X-skeleton humanoid (E6, stage 1).

    python scripts/tracking_reference_export.py [--list-only]

Categories: walk and fast (BABEL walk / run segments), sit_support and floor_work (kneel, lie) from results/splits/floor_work_clips.csv.
Rule (docs/simulation_tracking_protocol.md): validation split, clips not excluded from physical evaluation and not treadmill, segments of at least 5 s, seeded shuffle per category, one segment
per subject, the first 8 usable windows of each category (the first window of each category is the stage 2 clip); a window is the
first 10 s of its segment. Output: <data_root>/tracking_reference/<name>/*.npz and manifest.json, plus a copy of the manifest
in results/E/tracking_infrastructure/reference_manifest.json.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from hready.body.batch_forward import smpl_forward_bt
from hready.body.smplx_wrapper import load_body
from hready.data.amass import clip_flags, load_clip, load_index, load_paths_config
from hready.data.babel import BabelIndexEntry, load_babel_index_payload
from hready.data.completion_clips import FLOOR_WORK_ELIGIBLE_CLASSES, split_rels
from hready.data.ego_splits import _collect_babel_proposals

from hready.robot import tracking_reference as R


def _babel_candidates(cfg: dict, val_rels: set[str]) -> list[R.Candidate]:
    kw = {k: dict(cfg["cohorts"][f"{k}_babel_keywords"]) for k in ("floor_work", "ordinary_locomotion", "other_labelled")}
    babel = {}
    for row in load_babel_index_payload()["entries"]:
        row = dict(row)
        if "playback_duration" not in row:
            row.setdefault("playback_duration", float(row.get("mocap_time_length", row.get("babel_dur", 0.0))))
        row.pop("mocap_time_length", None)
        row.pop("time_scale", None)
        babel[row["rel_path"]] = BabelIndexEntry(**row)
    amass = {e.rel_path: e for e in load_index()}
    props = _collect_babel_proposals(babel, val_rels, amass, kw, 1.0)
    cat_of = {"walk": "walk", "run": "fast"}
    return [R.Candidate(cat_of[p.category], p.rel_path, p.subject_key, p.start_s, p.end_s)
            for p in props if p.cohort == "ordinary_locomotion" and p.category in cat_of]


def _floor_candidates(csv_path: Path, val_rels: set[str]) -> list[R.Candidate]:
    amass = {e.rel_path: e for e in load_index()}
    out = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rel, g = r["rel_path"], r["geometry_class"]
            if rel not in val_rels or r["skate_flag"] == "True":
                continue
            cat = "sit_support" if g == "sit_support" else "floor_work" if g in FLOOR_WORK_ELIGIBLE_CLASSES else None
            if cat is not None:
                e = amass[rel]
                out.append(R.Candidate(cat, rel, f"{e.subset}/{e.subject}", float(r["segment_start_s"]), float(r["segment_end_s"])))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-only", action="store_true", help="print the candidate counts and the chosen order, write nothing")
    ap.add_argument("--asset-json", type=Path, default=None)
    ap.add_argument("--name", default="validation_windows")
    ap.add_argument("--ego-splits", type=Path, default=Path("configs/ego_splits.yaml"))
    ap.add_argument("--floor-csv", type=Path, default=Path("results/splits/floor_work_clips.csv"))
    ap.add_argument("--manifest-copy", type=Path, default=Path("results/E/tracking_infrastructure/reference_manifest.json"))
    args = ap.parse_args(argv)

    data_root = Path(load_paths_config()["data_root"])
    meta = json.loads((args.asset_json or data_root / "robot_assets" / "smplx_humanoid" / "smplx_humanoid.json").read_text(encoding="utf-8"))
    cfg = yaml.safe_load(args.ego_splits.read_text(encoding="utf-8"))
    val_rels = set(split_rels("val"))
    cands = _babel_candidates(cfg, val_rels) + _floor_candidates(args.floor_csv, val_rels)
    flags = {rel: clip_flags(rel) for rel in sorted({c.rel_path for c in cands})}
    physical = {rel for rel, f in flags.items() if not (f["exclude_physical_eval"] or f["treadmill_name"])}
    kept = [c for c in cands if c.rel_path in physical]  # clips the project already excludes from physical evaluation, and treadmill clips
    order = R.shuffled_order(kept, seed=R.SEED)

    print("segments in the validation split: all, after the clip quality flags, usable (5 s minimum, one per subject):")
    for cat in R.CATEGORIES:
        print(f"  {cat:12s} all {sum(c.category == cat for c in cands):4d}   flags {sum(c.category == cat for c in kept):4d}   "
              f"usable {len(order[cat]):4d}")
    if args.list_only:
        for cat in R.CATEGORIES:
            for c in order[cat][: R.WINDOWS_PER_CATEGORY + 4]:
                print(f"  {cat:12s} {c.rel_path} {c.start_s:.1f}-{c.end_s:.1f} s")
        return

    body = load_body("locked_head")
    out_dir = data_root / "tracking_reference" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    entries, skipped = [], []
    for cat in R.CATEGORIES:
        taken = 0
        for c in order[cat]:
            if taken >= R.WINDOWS_PER_CATEGORY:
                break
            clip = load_clip(c.rel_path, ground=True)
            n = len(clip["transl"])
            f0, f1 = R.window_frames(c, n)
            if clip.get("floor_outlier"):
                skipped.append({"rel_path": c.rel_path, "category": cat, "reason": "floor_outlier"})
                continue
            if f1 - f0 < R.MIN_SEGMENT_S * R.FPS - 1:
                skipped.append({"rel_path": c.rel_path, "category": cat, "reason": "window shorter than 5 s inside the clip"})
                continue
            with torch.no_grad():
                joints, _ = smpl_forward_bt(
                    body, torch.as_tensor(clip["transl"], dtype=torch.float32)[None],
                    torch.as_tensor(clip["root_orient"], dtype=torch.float32)[None],
                    torch.as_tensor(clip["pose_body"], dtype=torch.float32)[None],
                    torch.as_tensor(np.asarray(clip["betas"], dtype=np.float32)[None]))
            pelvis = joints[0, f0:f1, 0].cpu().numpy().astype(np.float64)
            ref = R.build_reference(meta, clip["root_orient"][f0:f1], clip["pose_body"][f0:f1], pelvis)
            name = f"{cat}_{taken:02d}.npz"
            np.savez(out_dir / name, fps=R.FPS, root_pos=ref["root_pos"], root_quat_xyzw=ref["root_quat_xyzw"], hinges=ref["hinges"],
                     joints=ref["joints"], lowest_m=ref["lowest_m"])
            entries.append({"file": name, "category": cat, "rank_in_category": taken, "stage2_clip": taken == 0, "rel_path": c.rel_path,
                            "subject": c.subject_key, "segment_s": [c.start_s, c.end_s], "frames": [f0, f1], "steps": f1 - f0,
                            "duration_s": (f1 - f0 - 1) / R.FPS, **ref["stats"]})
            taken += 1
        if taken == 0:
            raise SystemExit(f"no usable window for category {cat}; stop and report, do not change the rule after seeing this")
    manifest = {"rule": "docs/simulation_tracking_protocol.md, amendment on reference windows", "seed": R.SEED, "fps": R.FPS,
                "min_segment_s": R.MIN_SEGMENT_S, "max_window_s": R.MAX_WINDOW_S, "windows_per_category": R.WINDOWS_PER_CATEGORY,
                "usable_segments": {cat: len(order[cat]) for cat in R.CATEGORIES},
                "excluded_clip_flags": "exclude_physical_eval or treadmill name",
                "windows_taken": dict(Counter(e["category"] for e in entries)), "skipped": skipped, "windows": entries}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    args.manifest_copy.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_copy.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    by = defaultdict(list)
    for e in entries:
        by[e["category"]].append(e)
    for cat, rows in by.items():
        print(f"{cat:12s} windows {len(rows)}  duration {min(r['duration_s'] for r in rows):.1f}-{max(r['duration_s'] for r in rows):.1f} s  "
              f"clipped hinge values {100 * np.mean([r['hinge_values_clipped_fraction'] for r in rows]):.1f} %")
    print("written", out_dir)


if __name__ == "__main__":
    main()
