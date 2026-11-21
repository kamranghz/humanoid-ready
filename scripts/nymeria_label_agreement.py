"""Agreement between two people's hand labels (Nymeria glove centres) and a consensus label file.

Inputs are two ``hand_labels.json`` files from ``nymeria_label_tool.html`` (same frames, labelled
independently). The script reports

1. how often the two agree on "visible point" versus "not visible", per hand;
2. the distance between the two clicks on hands both marked (pixels of the 800 px frame): median,
   75th and 90th percentile, maximum, share within 20, 40 and 60 px, per side;
3. the mean signed difference (second minus first labeller), which shows a systematic difference in
   where each person puts "the centre";
4. the largest disagreements.

It also writes a consensus file: the mean of the two clicks where both marked a point, "not_visible"
where both did, and "disputed" (left out of any evaluation) where only one marked a point. The rule is
fixed here, before any evaluation on the consensus.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SIDES = ("left", "right")
REVIEW = Path(r"D:\projects\hready_data\review\nymeria_projection\labels")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compare(first: dict, second: dict) -> tuple[dict, list[dict]]:
    """Counts of status agreement and one row per hand both labellers marked."""
    if first["image_size"] != second["image_size"]:
        raise SystemExit("image sizes differ")
    size = first["image_size"]
    a = {f["id"]: f for f in first["frames"]}
    b = {f["id"]: f for f in second["frames"]}
    common = sorted(set(a) & set(b))
    counts = {
        "frames_first": len(a),
        "frames_second": len(b),
        "frames_common": len(common),
        "both_point": 0,
        "both_not_visible": 0,
        "first_point_second_not": 0,
        "second_point_first_not": 0,
    }
    rows = []
    for frame_id in common:
        for side in SIDES:
            sa, sb = a[frame_id][f"{side}_status"], b[frame_id][f"{side}_status"]
            if sa == "point" and sb == "point":
                counts["both_point"] += 1
                pa = np.array(a[frame_id][side]) * size
                pb = np.array(b[frame_id][side]) * size
                rows.append(
                    {
                        "id": frame_id,
                        "side": side,
                        "first_px": pa.tolist(),
                        "second_px": pb.tolist(),
                        "distance_px": float(np.linalg.norm(pa - pb)),
                    }
                )
            elif sa == "point":
                counts["first_point_second_not"] += 1
            elif sb == "point":
                counts["second_point_first_not"] += 1
            else:
                counts["both_not_visible"] += 1
    return counts, rows


def summarise(rows: list[dict]) -> dict:
    out = {}
    groups = {"all": rows} | {s: [r for r in rows if r["side"] == s] for s in SIDES}
    for name, group in groups.items():
        if not group:
            continue
        d = np.array([r["distance_px"] for r in group])
        out[name] = {
            "n": len(d),
            "median_px": float(np.median(d)),
            "p75_px": float(np.percentile(d, 75)),
            "p90_px": float(np.percentile(d, 90)),
            "max_px": float(d.max()),
            "within_20px": float(np.mean(d < 20)),
            "within_40px": float(np.mean(d < 40)),
            "within_60px": float(np.mean(d < 60)),
        }
    diff = np.array([np.array(r["second_px"]) - np.array(r["first_px"]) for r in rows])
    out["mean_second_minus_first_px"] = diff.mean(axis=0).tolist() if len(diff) else None
    return out


def consensus(first: dict, second: dict) -> dict:
    size = first["image_size"]
    b = {f["id"]: f for f in second["frames"]}
    frames = []
    for fa in first["frames"]:
        fb = b.get(fa["id"])
        if fb is None:
            continue
        out = {k: fa[k] for k in ("id", "sequence", "time_s", "label")}
        for side in SIDES:
            sa, sb = fa[f"{side}_status"], fb[f"{side}_status"]
            if sa == "point" and sb == "point":
                mean = (np.array(fa[side]) + np.array(fb[side])) / 2
                out[side], out[f"{side}_status"] = mean.tolist(), "point"
            elif sa == "not_visible" and sb == "not_visible":
                out[side], out[f"{side}_status"] = None, "not_visible"
            else:
                out[side], out[f"{side}_status"] = None, "disputed"
        frames.append(out)
    return {
        "version": first["version"],
        "image_size": size,
        "note": "consensus: mean of two independent labellers where both marked a point; "
        "'disputed' where only one did (not evaluated)",
        "frames": frames,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--first", type=Path, default=REVIEW / "hand_labels.json")
    parser.add_argument("--second", type=Path, default=REVIEW / "hand_labels_monica.json")
    parser.add_argument(
        "--consensus-out", type=Path, default=REVIEW / "hand_labels_consensus.json"
    )
    parser.add_argument("--report-out", type=Path, default=REVIEW / "label_agreement.json")
    args = parser.parse_args(argv)
    first, second = load(args.first), load(args.second)
    counts, rows = compare(first, second)
    summary = summarise(rows)
    worst = sorted(rows, key=lambda r: -r["distance_px"])[:10]
    args.report_out.write_text(
        json.dumps(
            {"counts": counts, "distance": summary, "largest": worst, "rows": rows},
            indent=1,
        ),
        encoding="utf-8",
    )
    cons = consensus(first, second)
    args.consensus_out.write_text(json.dumps(cons, indent=1), encoding="utf-8")
    print("status", json.dumps(counts))
    print("click distance", json.dumps(summary))
    print("largest disagreements")
    for r in worst:
        print(f"  {r['distance_px']:.0f} px  {r['id']}  {r['side']}")
    points = sum(f[f"{s}_status"] == "point" for f in cons["frames"] for s in SIDES)
    disputed = sum(f[f"{s}_status"] == "disputed" for f in cons["frames"] for s in SIDES)
    print(f"consensus: {points} points, {disputed} disputed hands (not evaluated)")
    print(f"written to {args.report_out} and {args.consensus_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
