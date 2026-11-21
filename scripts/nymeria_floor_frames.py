"""Pull frames from the Nymeria preview videos at the moments the narration calls floor work.

For each sequence: read the per-frame timestamps stored in the video, check that the narration
times fall inside the video's time range (same clock), pick narrated kneel, sit-on-ground,
squat and lie moments spread over the sequence, and cut one small JPEG at the middle of each.
Needs ``ffmpeg`` on PATH. Writes only under ``--out`` (outside the repository by default).

Usage (hready environment, Windows):
    python scripts/nymeria_floor_frames.py
    python scripts/nymeria_floor_frames.py --per-label 3 --size 512
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mp4_probe import frame_times_ns

DEFAULT_ROOT = Path(r"D:\projects\hready_data\datasets\nymeria")
DEFAULT_OUT = Path(r"D:\projects\hready_data\review\nymeria_floor_frames")
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
LABELS = {
    "kneel": r"\bC\b[^.]{0,80}\b(kneels?|kneeling|knelt)\b",
    "sit_ground": r"\bC\b[^.]{0,80}\b(sits?|sitting)\b[^.]{0,40}" + GROUND,
    "squat": r"\bC\b[^.]{0,80}\b(squats?|squatting|crouch(es|ing)?)\b",
    "lie": r"\bC\b[^.]{0,80}\b(lies|lie down|lying|lays down|laying down|lay down)\b",
}


def narration_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as handle:
        return [
            {
                "start": float(row["start_time"]),
                "end": float(row["end_time"]),
                "text": list(row.values())[-1] or "",
            }
            for row in csv.DictReader(handle)
        ]


def spread(items: list, count: int) -> list:
    """``count`` items spread evenly over a list."""
    if len(items) <= count:
        return items
    return [items[int((i + 0.5) * len(items) / count)] for i in range(count)]


def cut_frame(video: Path, seconds: float, out: Path, size: int) -> bool:
    command = [
        "ffmpeg", "-loglevel", "error", "-y", "-ss", f"{seconds:.3f}", "-i", str(video),
        "-frames:v", "1", "-update", "1", "-vf", f"scale={size}:{size}", "-q:v", "3", str(out),
    ]  # fmt: skip
    return subprocess.run(command, check=False).returncode == 0 and out.exists()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--per-label", type=int, default=2, help="frames per label per sequence"
    )
    parser.add_argument("--size", type=int, default=480, help="side of the output JPEG")
    parser.add_argument(
        "--sequence",
        action="append",
        help="sequence folder name; default: the 7 floor-heavy ones",
    )
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    report = []
    for sequence in args.sequence or SEQUENCES:
        folder = args.root / sequence
        videos = sorted(folder.glob("*_preview_rgb.mp4"))
        narration = sorted(folder.glob("*_narration_atomic_action.csv"))
        if not videos or not narration:
            print(f"{sequence}: video or narration missing, skipped")
            continue
        times = frame_times_ns(videos[0])
        if not times:
            print(f"{sequence}: no frame times in the video, skipped")
            continue
        first_s, last_s = times[0] / 1e9, times[-1] / 1e9
        rows = narration_rows(narration[0])
        inside = sum(1 for r in rows if first_s <= r["start"] and r["end"] <= last_s)
        print(
            f"{sequence}: video {first_s:.1f}-{last_s:.1f} s, narration "
            f"{min(r['start'] for r in rows):.1f}-{max(r['end'] for r in rows):.1f} s, "
            f"{inside}/{len(rows)} narration rows inside the video"
        )
        entry = {
            "sequence": sequence,
            "video_s": [first_s, last_s],
            "rows": len(rows),
            "rows_inside": inside,
            "frames": [],
        }
        for label, pattern in LABELS.items():
            hits = [
                r
                for r in rows
                if re.search(pattern, r["text"], re.IGNORECASE)
                and first_s <= r["start"] <= last_s
            ]
            for number, row in enumerate(spread(hits, args.per_label)):
                middle = (row["start"] + row["end"]) / 2
                name = f"{sequence[:15]}_{label}_{number}_t{middle - first_s:.0f}.jpg"
                done = cut_frame(
                    videos[0], middle - first_s, args.out / name, args.size
                )
                entry["frames"].append(
                    {
                        "file": name if done else None,
                        "label": label,
                        "narration_time_s": middle,
                        "video_time_s": middle - first_s,
                        "text": row["text"][:300],
                    }
                )
            print(f"   {label}: {len(hits)} narrated spans in range")
        report.append(entry)
    with open(args.out / "index.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nframes and index.json written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
