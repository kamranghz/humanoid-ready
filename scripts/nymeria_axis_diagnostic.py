"""Camera-axis diagnostic for the Nymeria foot-lock run with the repo model.

Uses only the head pose from the glasses (the cached inputs of nymeria_foot_lock_repo_model.py) and the upright
frames of the data check. No model output and no foot label is read. Nothing is scored.

For every sequence it prints numbers that show whether the axes of the frame stored in the inputs (x down, y left, z forward) hold,
and that do not depend only on how far the wearer looks down: the three gravity components, the roll, the look
elevation, and how well each of the six signed axes lines up with the walking direction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import nymeria_foot_lock_experiment as e1
import nymeria_foot_lock_repo_model as rm
import nymeria_foot_lock_data_check as s0

HZ = e1.HZ
MIN_PROJECTION = 0.3  # an axis is "horizontal enough" to give a direction if its horizontal part is this long
AXES = ("+x", "-x", "+y", "-y", "+z", "-z")


def device_axes(camera_r: np.ndarray) -> dict[str, np.ndarray]:
    """Axes of the stored camera frame in the world frame from the cached camera rotation (rows right, up, look)."""
    r = camera_r.astype(np.float64)
    x, y, z = -r[:, 1], -r[:, 0], r[:, 2]
    return {"+x": x, "-x": -x, "+y": y, "-y": -y, "+z": z, "-z": -z}


def horizontal_cos(axis_w: np.ndarray, heading: np.ndarray) -> np.ndarray:
    h = axis_w[:, :2]
    length = np.linalg.norm(h, axis=1)
    out = np.full(len(axis_w), np.nan)
    ok = length >= MIN_PROJECTION
    out[ok] = (h[ok] * heading[ok, :2]).sum(axis=1) / length[ok]
    return out


def diagnose(rec: dict, camera_r: np.ndarray, upright: np.ndarray) -> dict:
    ax = device_axes(camera_r)
    on_upright = int(upright.sum()) >= 30 * HZ
    use = upright if on_upright else np.ones(len(upright), bool)
    g = {
        k: -ax[k][:, 2] for k in ("+x", "+y", "+z")
    }  # gravity (down) along the camera axis
    elev = np.degrees(np.arcsin(np.clip(ax["+z"][:, 2], -1, 1)))[use]
    out = {
        "upright_frames": int(use.sum()),
        "on_upright_frames": bool(on_upright),
        "gravity_mean_xyz": [float(g[k][use].mean()) for k in ("+x", "+y", "+z")],
        "gravity_median_xyz": [float(np.median(g[k][use])) for k in ("+x", "+y", "+z")],
        "roll_median_abs_gravity_y": float(np.median(np.abs(g["+y"][use]))),
        "sagittal_fraction": float((g["+x"][use] ** 2 + g["+z"][use] ** 2).mean()),
        "look_elevation_deg": {
            "mean": float(elev.mean()),
            "p10": float(np.percentile(elev, 10)),
            "p50": float(np.percentile(elev, 50)),
            "p90": float(np.percentile(elev, 90)),
            "share_below_minus_60": float((elev < -60).mean()),
        },
    }
    k = HZ
    pos = rec["head_pos"]
    vel = (pos[k:, :2] - pos[:-k, :2]) / (k / HZ)
    speed = np.linalg.norm(vel, axis=1)
    walk = speed > rm.AXIS_WALK_SPEED
    out["walking_frames"] = int(walk.sum())
    if walk.sum() >= rm.AXIS_MIN_WALK_FRAMES:
        heading = np.zeros((len(vel), 3))
        heading[:, :2] = vel / np.maximum(speed, 1e-9)[:, None]
        cos, count = {}, {}
        for name in AXES:
            c = horizontal_cos(ax[name][:-k], heading)[walk]
            c = c[np.isfinite(c)]
            cos[name] = float(c.mean()) if len(c) else None
            count[name] = int(len(c))
        out["walk_cos"] = cos
        out["walk_cos_frames"] = count
        cz = horizontal_cos(ax["+z"][:-k], heading)[walk]
        cz = cz[np.isfinite(cz)]
        out["z_walk_share_forward"] = float((cz > 0).mean()) if len(cz) else None
        out["z_walk_median_offset_deg"] = (
            float(np.median(np.degrees(np.arccos(np.clip(cz, -1, 1)))))
            if len(cz)
            else None
        )
        # which of the four horizontal-ish axes lines up best with the walking direction
        cand = {
            k_: v
            for k_, v in cos.items()
            if k_ in ("+y", "-y", "+z", "-z") and v is not None
        }
        out["best_forward_axis"] = max(cand, key=cand.get) if cand else None
    else:
        out["walk_cos"] = None
        out["best_forward_axis"] = None
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", type=Path, default=s0.sync.DEFAULT_ROOT)
    p.add_argument("--data-check", type=Path, default=e1.DATA_CHECK)
    p.add_argument("--experiment-cache", type=Path, default=e1.CACHE)
    p.add_argument("--cache", type=Path, default=rm.CACHE)
    p.add_argument(
        "--out",
        type=Path,
        default=rm.OUT.with_name(f"nymeria_axis_diagnostic_{rm.INPUT_VERSION}.json"),
    )
    args = p.parse_args(argv)
    prepared = e1.prepare(
        args.root, args.data_check, args.experiment_cache, False, None
    )
    rows = {}
    print(
        f"{'sequence':34s} {'up s':>5s} {'g_x':>5s} {'|g_y|':>5s} {'sag':>4s} {'elev mean':>9s} {'p10':>5s} {'p90':>5s}"
        f" {'walk s':>6s} {'cos+z':>6s} {'cos-x':>6s} {'cos+y':>6s} {'cos-y':>6s} {'best':>4s} low"
    )
    for name in prepared["names"]:
        file = args.cache / "inputs" / f"{name}.npz"
        if not file.exists():
            print(f"{name[:34]:34s} no cached inputs (excluded or not built)")
            continue
        rec = prepared["recs"][name]
        with np.load(file) as data:
            camera_r = data["camera_R"]
            strict = json.loads(str(data["axis"]))
        idx, valid = e1.lookup(rec, 0.0)
        upright = valid & rec["slam_ok"] & rec["upright"][idx]
        d = diagnose(rec, camera_r, upright)
        d["walking_along_look_in_inputs"] = strict.get("walking_along_look")
        rows[name] = d
        w = d["walk_cos"]
        e = d["look_elevation_deg"]
        f = lambda v: "   n/a" if v is None else f"{v:6.2f}"  # noqa: E731
        print(
            f"{name[:34]:34s} {d['upright_frames'] / HZ:5.0f} {d['gravity_mean_xyz'][0]:5.2f} {d['roll_median_abs_gravity_y']:5.2f}"
            f" {d['sagittal_fraction']:4.2f} {e['mean']:9.1f} {e['p10']:5.0f} {e['p90']:5.0f} {d['walking_frames'] / HZ:6.0f}"
            f" {f(w and w['+z'])} {f(w and w['-x'])} {f(w and w['+y'])} {f(w and w['-y'])}"
            f" {(d['best_forward_axis'] or 'n/a'):>4s} {'LOW' if (strict.get('walking_along_look') is not None and strict['walking_along_look'] < rm.AXIS_LOW_FORWARD) else '-'}"
        )
    summary = {}

    def stat(label, values):
        v = np.array([x for x in values if x is not None], float)
        if len(v):
            summary[label] = {
                "n": int(len(v)),
                "min": float(v.min()),
                "median": float(np.median(v)),
                "max": float(v.max()),
            }
            print(
                f"{label:34s} n={len(v):3d} min {v.min():6.2f} median {np.median(v):6.2f} max {v.max():6.2f}"
            )

    print()
    stat("gravity along +x (mean)", [r["gravity_mean_xyz"][0] for r in rows.values()])
    stat("gravity along +y (mean)", [r["gravity_mean_xyz"][1] for r in rows.values()])
    stat("gravity along +z (mean)", [r["gravity_mean_xyz"][2] for r in rows.values()])
    stat(
        "|gravity y| median (roll)",
        [r["roll_median_abs_gravity_y"] for r in rows.values()],
    )
    stat("gravity in the x-z plane", [r["sagittal_fraction"] for r in rows.values()])
    stat(
        "look elevation mean (deg)",
        [r["look_elevation_deg"]["mean"] for r in rows.values()],
    )
    stat(
        "walking along +z",
        [r["walk_cos"]["+z"] for r in rows.values() if r["walk_cos"]],
    )
    stat(
        "walking along -z",
        [r["walk_cos"]["-z"] for r in rows.values() if r["walk_cos"]],
    )
    stat(
        "walking along +y",
        [r["walk_cos"]["+y"] for r in rows.values() if r["walk_cos"]],
    )
    stat(
        "walking along -y",
        [r["walk_cos"]["-y"] for r in rows.values() if r["walk_cos"]],
    )
    stat(
        "share of walking frames, z forward",
        [r["z_walk_share_forward"] for r in rows.values() if r["walk_cos"]],
    )
    best = [r["best_forward_axis"] for r in rows.values() if r["best_forward_axis"]]
    counts = {a: best.count(a) for a in ("+z", "-z", "+y", "-y")}
    summary["best_forward_axis_counts"] = counts
    print(
        f"best forward axis among +-y, +-z over {len(best)} sequences with enough walking: {counts}"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"sequences": rows, "summary": summary}, indent=1))
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
