"""Find the time offset between the Nymeria body motion (Xsens) and the head trajectory, and check it.

The Xsens timestamps are not on the head SLAM clock, so the offset is estimated from the data: the
head rotation speed (the norm of the angular velocity, the same in any frame) of the Xsens head segment
is matched against the SLAM device angular speed over all offsets that keep most of the Xsens
recording inside the SLAM one. The result is checked three ways: the sharpness of the correlation
peak, the agreement of head height changes after alignment, and the Xsens head height during narrated
kneel and sit-on-ground spans (narration times are on the SLAM clock).

Reads without extracting anything to disk: ``body/xdata.npz`` from ``body_motion.zip``, the head SLAM
trajectory from ``recording_head.zip`` and the atomic-action narration CSV.

Usage (hready environment, Windows):
    python scripts/nymeria_motion_sync.py
    python scripts/nymeria_motion_sync.py --sequence 20230706_s0_breanna_reese_act0_cfg468
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np

DEFAULT_ROOT = Path(r"D:\projects\hready_data\datasets\nymeria")
DEFAULT_OUT = Path(r"D:\projects\hready_data\review\nymeria_motion_sync.json")
SEQUENCES = (
    "20230706_s0_breanna_reese_act0_cfg468",
    "20230815_s0_samantha_lester_act0_513kae",
    "20230628_s1_hayley_little_act0_z038r0",
    "20230731_s0_tammy_campos_act0_d3blev",
    "20230720_s0_robert_ballard_act0_7n4853",
    "20230707_s1_elizabeth_tucker_act0_df444f",
    "20230816_s1_jeffery_bryant_act0_p5w199",
)
GROUND = r"\b(floor|ground|carpet|rug|mat)\b"
SPANS = {
    "kneel": r"\bC\b[^.]{0,80}\b(kneels?|kneeling|knelt)\b",
    "sit_ground": r"\bC\b[^.]{0,80}\b(sits?|sitting)\b[^.]{0,40}" + GROUND,
}
GRID_HZ = 30.0
SLAM_STRIDE = 10
MIN_OVERLAP_FRACTION = 0.6
HEAD_SEGMENT = 6  # head in the standard 23-segment Xsens order; checked against the highest segment
SMOOTH_S = 0.1
PEAK_EXCLUDE_S = 1.0


def find(folder: Path, suffix: str) -> Path | None:
    hits = sorted(folder.glob("*_" + suffix))
    return hits[0] if hits else None


def load_xsens(zip_path: Path) -> dict:
    """The arrays needed from body/xdata.npz inside body_motion.zip (read into memory only)."""
    with zipfile.ZipFile(zip_path) as archive:
        raw = archive.read("body/xdata.npz")
    with np.load(io.BytesIO(raw)) as data:
        return {
            key: data[key]
            for key in (
                "timestamps_us",
                "segment_tXYZ",
                "segment_angularVelocity",
                "foot_contacts",
                "frameRate",
                "frameCount",
            )
        }


def load_slam(zip_path: Path) -> dict:
    """Time (both clocks) and world z of the head trajectory, every SLAM_STRIDE-th row."""
    member = "recording_head/mps/slam/closed_loop_trajectory.csv"
    track, utc, z, omega = [], [], [], []
    with zipfile.ZipFile(zip_path) as archive, archive.open(member) as handle:
        reader = csv.reader(io.TextIOWrapper(handle, encoding="utf-8", newline=""))
        header = next(reader)
        i_track = header.index("tracking_timestamp_us")
        i_utc = header.index("utc_timestamp_ns")
        i_z = header.index("tz_world_device")
        i_w = [header.index(f"angular_velocity_{axis}_device") for axis in "xyz"]
        for number, row in enumerate(reader):
            if number % SLAM_STRIDE:
                continue
            track.append(float(row[i_track]))
            utc.append(float(row[i_utc]) / 1000.0)
            z.append(float(row[i_z]))
            omega.append(float(np.linalg.norm([float(row[i]) for i in i_w])))
    return {
        "track_us": np.array(track),
        "utc_us": np.array(utc),
        "z": np.array(z),
        "omega": np.array(omega),
    }


def narration_spans(path: Path) -> dict:
    out = {name: [] for name in SPANS}
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            text = list(row.values())[-1] or ""
            for name, pattern in SPANS.items():
                if re.search(pattern, text, re.IGNORECASE):
                    out[name].append((float(row["start_time"]), float(row["end_time"])))
    return out


def to_grid(
    t_us: np.ndarray, values: np.ndarray, t0_us: float, count: int
) -> np.ndarray:
    """Box-smooth a series (SMOOTH_S) and sample it on a GRID_HZ grid starting at ``t0_us``."""
    rate = (len(t_us) - 1) / ((t_us[-1] - t_us[0]) / 1e6)
    width = max(1, round(float(rate * SMOOTH_S)))
    smooth = np.convolve(values, np.ones(width) / width, mode="same")
    return np.interp(t0_us + np.arange(count) * 1e6 / GRID_HZ, t_us, smooth)


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 30 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def offset_search(
    slam: np.ndarray, xs: np.ndarray, needed: int
) -> list[tuple[int, float]]:
    """Correlation for every shift k (slam index i matches xs index i - k) with at least ``needed`` samples overlapping."""
    out = []
    for k in range(-(len(xs) - needed), len(slam) - needed + 1):
        i0, i1 = max(0, k), min(len(slam), len(xs) + k)
        if i1 - i0 < needed:
            continue
        out.append((k, correlation(slam[i0:i1], xs[i0 - k : i1 - k])))
    return out


def analyse(folder: Path) -> dict:
    body = find(folder, "body_motion.zip")
    head = find(folder, "recording_head.zip")
    narration = find(folder, "narration_atomic_action.csv")
    if body is None or head is None or narration is None:
        return {"error": "a required file is missing"}
    xs = load_xsens(body)
    xs_t = xs["timestamps_us"].astype(np.float64)
    positions = xs["segment_tXYZ"].reshape(len(xs_t), -1, 3)
    head_z = positions[:, HEAD_SEGMENT, 2]
    highest = int(np.argmax(np.median(positions[:, :, 2], axis=0)))
    xs_omega = np.linalg.norm(
        xs["segment_angularVelocity"].reshape(len(xs_t), -1, 3)[:, HEAD_SEGMENT, :],
        axis=1,
    )
    slam = load_slam(head)
    result: dict = {
        "xsens_frames": len(xs_t),
        "xsens_frameRate": int(xs["frameRate"][0]),
        "xsens_duration_s": float((xs_t[-1] - xs_t[0]) / 1e6),
        "slam_duration_s": float((slam["track_us"][-1] - slam["track_us"][0]) / 1e6),
        "xsens_t_start_us": float(xs_t[0]),
        "slam_track_start_us": float(slam["track_us"][0]),
        "head_segment_checked": {"used": HEAD_SEGMENT, "highest_median_z": highest},
        "foot_contacts_mean": [float(v) for v in xs["foot_contacts"].mean(axis=0)],
    }
    s0, x0 = slam["track_us"][0], xs_t[0]
    n_slam = int((slam["track_us"][-1] - s0) / 1e6 * GRID_HZ)
    n_xs = int((xs_t[-1] - x0) / 1e6 * GRID_HZ)
    slam_w = to_grid(slam["track_us"], slam["omega"], s0, n_slam)
    xs_w = to_grid(xs_t, xs_omega, x0, n_xs)
    needed = int(min(n_slam, n_xs) * MIN_OVERLAP_FRACTION)
    scan = [(k, c) for k, c in offset_search(slam_w, xs_w, needed) if not np.isnan(c)]
    if not scan:
        result["offset"] = "no usable overlap"
        return result
    best_k, best_c = max(scan, key=lambda kc: kc[1])
    exclude = PEAK_EXCLUDE_S * GRID_HZ
    others = [c for k, c in scan if abs(k - best_k) > exclude]
    offset_us = (s0 - x0) + best_k * 1e6 / GRID_HZ  # slam time = xsens time + offset
    result["offset"] = {
        "slam_minus_xsens_s": offset_us / 1e6,
        "corr_angular_speed": best_c,
        "best_corr_elsewhere": max(others) if others else None,
        "shifts_tested": len(scan),
    }
    # check 2: head height changes after alignment (SLAM z has an arbitrary origin, so compare changes)
    slam_z = to_grid(slam["track_us"], slam["z"], s0, n_slam)
    xs_z = to_grid(xs_t, head_z, x0, n_xs)
    i0, i1 = max(0, best_k), min(len(slam_z), len(xs_z) + best_k)
    result["corr_head_height_after_alignment"] = correlation(
        slam_z[i0:i1], xs_z[i0 - best_k : i1 - best_k]
    )
    # check 3: head height during narrated spans, relative to a standing proxy
    standing = float(np.percentile(head_z, 90))
    result["xsens_head_z_p90_m"] = standing
    stats = {}
    for label, spans in narration_spans(narration).items():
        values = []
        for start, end in spans:
            t = np.arange(start, end, 0.1) * 1e6 - offset_us
            inside = (t >= xs_t[0]) & (t <= xs_t[-1])
            if inside.any():
                values.append(float(np.median(np.interp(t[inside], xs_t, head_z))))
        stats[label] = {
            "spans": len(spans),
            "spans_with_data": len(values),
            "median_head_z_m": float(np.median(values)) if values else None,
            "ratio_to_standing_p90": float(np.median(values) / standing)
            if values
            else None,
        }
    result["narration_spans"] = stats
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--sequence", action="append", help="default: the 7 floor-heavy sequences"
    )
    args = parser.parse_args(argv)
    report = {}
    for sequence in args.sequence or SEQUENCES:
        print(f"{sequence} ...", flush=True)
        try:
            report[sequence] = analyse(args.root / sequence)
        except (OSError, KeyError, ValueError, zipfile.BadZipFile) as error:
            report[sequence] = {"error": f"{type(error).__name__}: {error}"}
        print(json.dumps(report[sequence], indent=1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
