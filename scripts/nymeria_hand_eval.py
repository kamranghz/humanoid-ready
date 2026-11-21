"""Hand projection error against hand-clicked glove centres (Nymeria, Xsens vs the head camera).

Inputs are ``hand_labels.json`` (from ``nymeria_label_tool.html``) and the report of
``nymeria_project_check.py`` (offset and head offset d of each sequence). For every labelled hand the
Xsens hand segment is projected into the video with the local world fit, and the distance to the
clicked point is reported in pixels and as an equivalent length at the hand.

Three numbers are reported: the error with the raw Xsens hand origin, the error after a constant
offset of the clicked point in the Xsens hand frame (fitted on the other sequences, so the held-out
sequence never sees its own labels), and how the error changes when the Xsens clock is shifted
(a check of the time offset against spatial evidence).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

import nymeria_motion_sync as sync
import nymeria_project_check as npc

REVIEW = Path(r"D:\projects\hready_data\review\nymeria_projection")
SIDE_SEGMENT = {"left": "l_hand", "right": "r_hand"}
SHIFTS_S = (-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2)
RIDGE = 1.0e3  # px^2 per m^2: only keeps directions the data cannot see at zero


def sequence_tag(sequence: str) -> str:
    parts = sequence.split("_")
    return "_".join(parts[:3]) if len(parts) > 2 else sequence


def sample_at(xs, head_t_us, offset_us, shift_s, segment, fit):
    """Xsens hand origin in the SLAM world and the matrix that carries a hand-frame offset there."""
    p_x, r_x = npc.xsens_at(
        xs, np.array([head_t_us - offset_us - shift_s * 1e6]), segment
    )
    rz = npc.rot_z(math.radians(fit["yaw_deg"]))
    return np.array(fit["translation_m"]) + rz @ p_x[0], rz @ r_x[0]


def collect(labels: dict, report: dict, root: Path, rotation: str) -> tuple[list, list]:
    """One sample per labelled hand that can be projected; a list of what was left out and why."""
    samples, skipped = [], []
    size = labels["image_size"]
    by_sequence: dict[str, list] = {}
    for frame in labels["frames"]:
        by_sequence.setdefault(frame["sequence"], []).append(frame)
    for sequence, frames in by_sequence.items():
        entry = report.get(sequence, {})
        offset_s = entry.get("refine", {}).get("offset_s")
        head_d = entry.get("shared_head_offset", {}).get("d_m")
        folder = root / sequence
        body, head_zip = (
            sync.find(folder, "body_motion.zip"),
            sync.find(folder, "recording_head.zip"),
        )
        if offset_s is None or body is None or head_zip is None:
            skipped.append({"sequence": sequence, "reason": "no offset or data"})
            continue
        # without a shared head offset (too few good windows) the local fit keeps d free: usable,
        # but reported on its own because d and the translation are then less well determined
        fit_mode = "fixed_d" if head_d is not None else "free_d"
        print(f"{sequence} ...", flush=True)
        offset_us = offset_s * 1e6
        xs = npc.load_xsens(body)
        head = npc.load_trajectory(head_zip, "recording_head")
        camera = npc.load_camera(head_zip)
        for frame in frames:
            t_s = frame["time_s"]
            fit = npc.local_world(
                head, xs, npc.SEGMENTS["head"], offset_us, t_s, head_d=head_d
            )
            if fit is None:
                skipped.append({"id": frame["id"], "reason": "no local fit"})
                continue
            pos_s, rot_s = npc.interp_pose(head, np.array([t_s * 1e6]))
            r_wc = rot_s[0] @ camera["r_dev_cam"]
            origin = pos_s[0] + rot_s[0] @ camera["t_dev_cam"]
            for side, segment_name in SIDE_SEGMENT.items():
                if frame.get(f"{side}_status") != "point":
                    continue
                segment = npc.SEGMENTS[segment_name]
                shifts = {
                    shift: sample_at(xs, t_s * 1e6, offset_us, shift, segment, fit)
                    for shift in SHIFTS_S
                }
                samples.append(
                    {
                        "sequence": sequence,
                        "id": frame["id"],
                        "frame_label": frame["label"],
                        "side": side,
                        "t_s": t_s,
                        "label_px": np.array(frame[side]) * size,
                        "shifts": shifts,
                        "origin": origin,
                        "r_wc": r_wc,
                        "params": camera["params"],
                        "size": size,
                        "rotation": rotation,
                        "fit_mode": fit_mode,
                        "fit_rms_m": fit["position_rms_m"],
                        "fit_spread_deg": fit["orientation_spread_deg_at_yaw"],
                    }
                )
    return samples, skipped


def project(sample: dict, h: np.ndarray, shift: float = 0.0):
    """Pixel of the Xsens hand (plus the hand-frame offset h) and its range [m]; None outside view."""
    base, mat = sample["shifts"][shift]
    p_c = (base + mat @ h - sample["origin"]) @ sample["r_wc"]
    uv, valid = npc.project_fisheye624(p_c[None, :], sample["params"])
    if not valid[0]:
        return None, float(np.linalg.norm(p_c))
    u, v = npc.to_image(uv[0], sample["size"], sample["rotation"])
    return np.array([u, v]), float(np.linalg.norm(p_c))


def errors(samples: list, offsets: dict, shift: float = 0.0) -> list[dict]:
    """Pixel and length error of each sample; ``offsets`` maps a side to its hand-frame offset."""
    rows = []
    for s in samples:
        px, rng = project(s, offsets[s["side"]], shift)
        row = {k: s[k] for k in ("sequence", "id", "frame_label", "side", "t_s")}
        row["range_m"] = rng
        if px is None:
            row["in_view"] = False
        else:
            err = float(np.linalg.norm(px - s["label_px"]))
            focal = s["params"][0] * s["size"] / npc.CALIB_SIZE
            row.update(in_view=True, error_px=err, error_cm=100.0 * err / focal * rng)
        rows.append(row)
    return rows


def fit_offset(samples: list, iterations=30) -> np.ndarray:
    """Hand-frame offset h that moves the projected hand onto the clicked point (robust, IRLS)."""
    h = np.zeros(3)
    for _ in range(iterations):
        jac_rows, res_rows, weights = [], [], []
        for s in samples:
            px, _ = project(s, h)
            if px is None:
                continue
            jac = np.zeros((2, 3))
            for k in range(3):
                step = np.zeros(3)
                step[k] = 1e-3
                plus, _ = project(s, h + step)
                minus, _ = project(s, h - step)
                if plus is None or minus is None:
                    break
                jac[:, k] = (plus - minus) / 2e-3
            else:
                res = s["label_px"] - px
                scale = 20.0  # px, Huber knee
                weight = (
                    1.0 if np.linalg.norm(res) <= scale else scale / np.linalg.norm(res)
                )
                jac_rows.append(jac)
                res_rows.append(res)
                weights.append(weight)
        if len(jac_rows) < 6:
            return h
        jac_all, res_all = np.concatenate(jac_rows), np.concatenate(res_rows)
        w = np.repeat(weights, 2)
        normal = jac_all.T @ (w[:, None] * jac_all) + RIDGE * np.eye(3)
        delta = np.linalg.solve(normal, jac_all.T @ (w * res_all) - RIDGE * h)
        h = h + delta
        if np.linalg.norm(delta) < 1e-5:
            break
    return h


def stats(rows: list[dict], rng: np.random.Generator) -> dict:
    seen = [r for r in rows if r["in_view"]]
    out = {"labelled": len(rows), "in_view": len(seen)}
    if not seen:
        return out
    for key in ("error_px", "error_cm"):
        values = np.array([r[key] for r in seen])
        boot = [np.median(rng.choice(values, len(values))) for _ in range(1000)]
        out[key] = {
            "median": float(np.median(values)),
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "p75": float(np.percentile(values, 75)),
            "p90": float(np.percentile(values, 90)),
        }
    return out


def loso(samples: list) -> tuple[list[dict], dict]:
    """Errors with an offset fitted on the other sequences, and the offsets of every fold."""
    rows, folds = [], {}
    for held in sorted({s["sequence"] for s in samples}):
        train = [s for s in samples if s["sequence"] != held]
        test = [s for s in samples if s["sequence"] == held]
        offsets = {
            side: fit_offset([s for s in train if s["side"] == side])
            for side in SIDE_SEGMENT
        }
        folds[held] = {
            side: [round(float(v), 4) for v in h] for side, h in offsets.items()
        }
        rows += errors(test, offsets)
    return rows, folds


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--labels", type=Path, default=REVIEW / "labels" / "hand_labels.json"
    )
    parser.add_argument(
        "--report", type=Path, default=REVIEW / "nymeria_project_check.json"
    )
    parser.add_argument("--root", type=Path, default=sync.DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=REVIEW / "hand_eval.json")
    parser.add_argument("--rotation", choices=("cw90", "ccw90", "none"), default="cw90")
    args = parser.parse_args(argv)
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    report = json.loads(args.report.read_text(encoding="utf-8"))
    samples, skipped = collect(labels, report, args.root, args.rotation)
    free = [s for s in samples if s["fit_mode"] == "free_d"]
    samples = [s for s in samples if s["fit_mode"] == "fixed_d"]
    if not samples:
        print("no labelled hands to evaluate")
        return 1
    rng = np.random.default_rng(0)
    zero = {side: np.zeros(3) for side in SIDE_SEGMENT}
    raw = errors(samples, zero)
    free_raw = errors(free, zero)
    cross, folds = loso(samples)
    in_sample = {
        side: fit_offset([s for s in samples if s["side"] == side])
        for side in SIDE_SEGMENT
    }
    result = {
        "labelled_hands": len(samples),
        "sequences": sorted({s["sequence"] for s in samples}),
        "skipped": skipped,
        "raw_xsens_hand_origin": stats(raw, rng),
        "free_d_sequences_raw_xsens_hand_origin": {
            seq: stats([r for r in free_raw if r["sequence"] == seq], rng)
            for seq in sorted({s["sequence"] for s in free})
        },
        "constant_offset_leave_one_sequence_out": stats(cross, rng),
        "offset_folds_m": folds,
        "offset_in_sample_m": {
            side: [round(float(v), 4) for v in h] for side, h in in_sample.items()
        },
        "per_sequence_raw": {
            seq: stats([r for r in raw if r["sequence"] == seq], rng)
            for seq in sorted({s["sequence"] for s in samples})
        },
        "clock_shift_median_error_px": {
            f"{shift:+.2f}s": stats(errors(samples, zero, shift), rng)
            .get("error_px", {})
            .get("median")
            for shift in SHIFTS_S
        },
        "rows_raw": raw,
        "rows_leave_one_out": cross,
    }
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    for key in ("raw_xsens_hand_origin", "constant_offset_leave_one_sequence_out"):
        print(key, json.dumps(result[key]))
    print(
        "free-d sequences (raw only, not in the headline)",
        json.dumps(result["free_d_sequences_raw_xsens_hand_origin"]),
    )
    print("offset folds", json.dumps(folds))
    print("clock shift", json.dumps(result["clock_shift_median_error_px"]))
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
