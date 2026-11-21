"""Which convention makes the projected Xsens hands land on the clicked gloves? (Nymeria)

``nymeria_hand_eval.py`` found a median miss of hundreds of pixels. A miss that large, and one that does
not change with a clock shift, points at a convention (which Xsens segment is the hand, which side is
left, how the camera frame maps to the upright video, which way the device-to-camera rotation goes) and
not at a few centimetres of error. This script tests those conventions against the hand-clicked labels.

For every labelled hand it projects *every* Xsens segment into the frame under 8 image mappings (four
rotations, with or without a transpose) and 2 camera rotations (as stored, and transposed), and reports

1. the standard mapping (left = segment 14, right = segment 10, the current ``cw90`` convention) next
   to the same mapping with the sides swapped, for every image mapping and camera rotation;
2. for the left and the right label separately, the segment and conventions with the smallest median miss.

A point that does not project into the image counts as ``MISS_PX``. With few labels a search over
conventions can fit noise, so only a large drop (from hundreds of pixels to tens) is evidence; a small
one is not. Nothing here changes the alignment; it only reports.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

import nymeria_motion_sync as sync
import nymeria_project_check as npc

REVIEW = Path(r"D:\projects\hready_data\review\nymeria_projection")
MISS_PX = 1000.0  # error given to a point that does not project into the image
SIDES = ("left", "right")
STANDARD = {"left": npc.SEGMENTS["l_hand"], "right": npc.SEGMENTS["r_hand"]}
MAPPING_NAMES = (
    "none",
    "cw90",
    "rot180",
    "ccw90",
    "transpose",
    "transpose+cw90",
    "transpose+rot180",
    "transpose+ccw90",
)


def to_frame(uv, size: int, k: int) -> tuple[float, float]:
    """Calibration pixel to the video frame; k = 0..3 rotations, 4..7 the same after a transpose."""
    u, v = np.asarray(uv, dtype=float) * size / npc.CALIB_SIZE
    if k >= 4:
        u, v = v, u
    turn = k % 4
    if turn == 0:
        return u, v
    if turn == 1:
        return size - v, u
    if turn == 2:
        return size - u, size - v
    return v, size - u


def pixel_error(frame: dict, segment: int, point, k: int, transposed: bool) -> float:
    r_dc = frame["r_dc"].T if transposed else frame["r_dc"]
    r_wc = frame["r_wd"] @ r_dc
    origin = frame["pos"] + frame["r_wd"] @ frame["t_dc"]
    p_c = (frame["world"][segment] - origin) @ r_wc
    uv, valid = npc.project_fisheye624(p_c[None, :], frame["params"])
    if not valid[0]:
        return MISS_PX
    u, v = to_frame(uv[0], frame["size"], k)
    return float(np.hypot(u - point[0], v - point[1]))


def points_of(frames: list[dict]) -> list[tuple[dict, str, np.ndarray]]:
    return [
        (f, side, f["points"][side])
        for f in frames
        for side in SIDES
        if side in f["points"]
    ]


def convention_rows(frames: list[dict]) -> list[dict]:
    """Standard segments and the two sides swapped, under every image mapping and camera rotation."""
    pts = points_of(frames)
    rows = []
    for k, transposed, swap in itertools.product(
        range(8), (False, True), (False, True)
    ):
        errors = []
        for frame, side, point in pts:
            other = "right" if side == "left" else "left"
            segment = STANDARD[other if swap else side]
            errors.append(pixel_error(frame, segment, point, k, transposed))
        errors = np.array(errors)
        rows.append(
            {
                "mapping": MAPPING_NAMES[k],
                "camera_rotation": "transposed" if transposed else "as stored",
                "sides": "swapped" if swap else "standard",
                "points": len(errors),
                "median_px": float(np.median(errors)),
                "share_inside_image": float(np.mean(errors < MISS_PX)),
                "share_within_60px": float(np.mean(errors < 60.0)),
            }
        )
    return sorted(rows, key=lambda r: r["median_px"])


def segment_rows(frames: list[dict], side: str, count: int = 5) -> list[dict]:
    """For one label side: the segment and conventions with the smallest median miss."""
    pts = [(f, p) for f, s, p in points_of(frames) if s == side]
    if not pts:
        return []
    n_segments = len(pts[0][0]["world"])
    rows = []
    for segment, k, transposed in itertools.product(
        range(n_segments), range(8), (False, True)
    ):
        errors = np.array([pixel_error(f, segment, p, k, transposed) for f, p in pts])
        rows.append(
            {
                "segment": segment,
                "mapping": MAPPING_NAMES[k],
                "camera_rotation": "transposed" if transposed else "as stored",
                "median_px": float(np.median(errors)),
                "share_within_60px": float(np.mean(errors < 60.0)),
            }
        )
    return sorted(rows, key=lambda r: r["median_px"])[:count]


def prepare(labels: dict, report: dict, root: Path) -> tuple[list[dict], list[dict]]:
    """Per labelled frame: every Xsens segment in the SLAM world (local fit) and the camera pose."""
    frames, skipped = [], []
    size = labels["image_size"]
    by_sequence: dict[str, list] = {}
    for frame in labels["frames"]:
        by_sequence.setdefault(frame["sequence"], []).append(frame)
    for sequence, items in by_sequence.items():
        entry = report.get(sequence, {})
        offset_s = entry.get("refine", {}).get("offset_s")
        head_d = entry.get("shared_head_offset", {}).get("d_m")
        folder = root / sequence
        body, head_zip = (
            sync.find(folder, "body_motion.zip"),
            sync.find(folder, "recording_head.zip"),
        )
        if offset_s is None or head_d is None or body is None or head_zip is None:
            skipped.append(
                {
                    "sequence": sequence,
                    "reason": "no offset, shared head offset or data",
                }
            )
            continue
        print(f"{sequence} ...", flush=True)
        offset_us = offset_s * 1e6
        xs = npc.load_xsens(body)
        head = npc.load_trajectory(head_zip, "recording_head")
        camera = npc.load_camera(head_zip)
        n_segments = xs["pos"].shape[1]
        for item in items:
            points = {
                side: np.array(item[side], dtype=float) * size
                for side in SIDES
                if item.get(f"{side}_status") == "point"
            }
            if not points:
                continue
            t_s = item["time_s"]
            fit = npc.local_world(
                head, xs, npc.SEGMENTS["head"], offset_us, t_s, head_d=head_d
            )
            if fit is None:
                skipped.append({"id": item["id"], "reason": "no local fit"})
                continue
            rz = npc.rot_z(np.radians(fit["yaw_deg"]))
            translation = np.array(fit["translation_m"])
            t_xs = np.array([t_s * 1e6 - offset_us])
            world = np.array(
                [
                    translation + rz @ npc.xsens_at(xs, t_xs, s)[0][0]
                    for s in range(n_segments)
                ]
            )
            pos_s, rot_s = npc.interp_pose(head, np.array([t_s * 1e6]))
            frames.append(
                {
                    "sequence": sequence,
                    "id": item["id"],
                    "points": points,
                    "world": world,
                    "r_wd": rot_s[0],
                    "pos": pos_s[0],
                    "r_dc": camera["r_dev_cam"],
                    "t_dc": camera["t_dev_cam"],
                    "params": camera["params"],
                    "size": size,
                }
            )
    return frames, skipped


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--labels", type=Path, default=REVIEW / "labels" / "hand_labels.json"
    )
    parser.add_argument(
        "--report", type=Path, default=REVIEW / "nymeria_project_check.json"
    )
    parser.add_argument("--root", type=Path, default=sync.DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=REVIEW / "hand_diagnose.json")
    args = parser.parse_args(argv)
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    report = json.loads(args.report.read_text(encoding="utf-8"))
    frames, skipped = prepare(labels, report, args.root)
    if not frames:
        print("no labelled hands to test")
        return 1
    rows = convention_rows(frames)
    result = {
        "frames": len(frames),
        "points": {side: sum(side in f["points"] for f in frames) for side in SIDES},
        "skipped": skipped,
        "conventions": rows,
        "best_segment_per_side": {side: segment_rows(frames, side) for side in SIDES},
    }
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    standard = next(
        r
        for r in rows
        if r["mapping"] == "cw90"
        and r["camera_rotation"] == "as stored"
        and r["sides"] == "standard"
    )
    print("points", result["points"])
    print("current convention", json.dumps(standard))
    print("best 6 conventions (standard segments)")
    for row in rows[:6]:
        print(" ", json.dumps(row))
    for side in SIDES:
        print(f"best segment for the {side} label")
        for row in result["best_segment_per_side"][side]:
            print(" ", json.dumps(row))
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
