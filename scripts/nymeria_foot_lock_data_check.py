"""Data check of the foot-lock protocol (Nymeria): cohort, id checks, labels, prevalence.

Implements section 2 to 4 of ``docs/nymeria_foot_lock_protocol.md`` and nothing else: no detector,
no metric of a detector. For every sequence, in the order of the SHA-256 of its name:

1. the clock-offset step: the same statistic as ``nymeria_motion_sync.py`` (Pearson correlation of the
   smoothed head angular speed at 30 Hz, every shift with at least 60% overlap, computed here with FFT
   sums so that 200 sequences are feasible), then ``nymeria_project_check.refine_offset``; the sequence
   enters the cohort when the refined correlation is at least ``TIMING_MIN_CORR`` (0.95);
2. the Xsens segment-order check (foot below knee below pelvis; right foot on the right of the walking
   direction), the side check of the four ``foot_contacts`` columns, and the labels with their prevalence.

Sequences that fail the offset step are listed and never retried (a file-reading error is not an
offset failure and is retried on the next run). The run stops when the cohort has
``--cohort`` sequences (60). The result is written after every sequence; a re-run continues where it
stopped, and refuses to continue with changed code unless ``--fresh`` archives the old file first.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

import nymeria_motion_sync as sync
import nymeria_project_check as npc

OUT = Path(r"D:\projects\hready_data\review\nymeria_foot_lock_data_check.json")
GRID_HZ = 30.0
V_ON, V_OFF, MIN_RUN = 0.20, 0.25, 3  # the constants of the repository's contact labels
BOX_S = 0.1
HEAD_HARD_SPEED = 0.05  # m/s, the "hard stratum" of the protocol
ORDER_MIN_SHARE = 0.95
SIDE_MIN_SHARE = (
    0.80  # a swapped left/right gives about 1 - share, so 0.8 separates the two clearly
)
STRAIGHT_TURN_DEG = (
    20.0  # heading change over +-0.5 s below which walking counts as straight
)
UPRIGHT_PELVIS_M = (
    0.80  # the height order is only checked while the pelvis is above this
)
MIN_UPRIGHT_FRAMES = 30 * 30  # 30 s of upright frames needed for the height check
WALK_SPEED = 0.8  # m/s of the pelvis for the left/right check
MIN_WALK_FRAMES = 200
MIN_COHORT_FOR_CLAIM = 30
MIN_BOTH_PLANTED_MIN = 60.0
SEG = npc.SEGMENTS
GRID_SEGMENTS = (
    SEG["pelvis"],
    SEG["head"],
    SEG["r_knee"],
    SEG["r_foot"],
    SEG["r_toe"],
    SEG["l_knee"],
    SEG["l_foot"],
    SEG["l_toe"],
)


# ---------------------------------------------------------------- cohort order and offset step


def hash_key(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def fold_of(name: str) -> int:
    return int(hash_key(name)[:8], 16) % 5


def ordered_sequences(root: Path) -> list[str]:
    names = [
        p.name
        for p in root.iterdir()
        if p.is_dir()
        and sync.find(p, "body_motion.zip")
        and sync.find(p, "recording_head.zip")
    ]
    return sorted(names, key=hash_key)


def pearson_scan(a: np.ndarray, b: np.ndarray, needed: int) -> list[tuple[int, float]]:
    """Same result as ``nymeria_motion_sync.offset_search`` (Pearson correlation of a[i0:i1] with
    b[i0-k:i1-k] for every shift k with at least ``needed`` overlapping samples), in O(n log n)."""
    a = a - a.mean()
    b = b - b.mean()
    na, nb = len(a), len(b)
    size = 1 << int(np.ceil(np.log2(na + nb - 1)))
    cross = np.fft.irfft(np.fft.rfft(a, size) * np.fft.rfft(b[::-1], size), size)
    cum = {
        name: np.concatenate([[0.0], np.cumsum(x)])
        for name, x in (("a", a), ("aa", a * a), ("b", b), ("bb", b * b))
    }
    out = []
    for k in range(-(nb - needed), na - needed + 1):
        i0, i1 = max(0, k), min(na, nb + k)
        n = i1 - i0
        if n < needed or n < 30:
            continue
        sa = cum["a"][i1] - cum["a"][i0]
        saa = cum["aa"][i1] - cum["aa"][i0]
        sb = cum["b"][i1 - k] - cum["b"][i0 - k]
        sbb = cum["bb"][i1 - k] - cum["bb"][i0 - k]
        va, vb = saa - sa * sa / n, sbb - sb * sb / n
        if va <= 0 or vb <= 0:
            continue
        out.append((k, float((cross[k + nb - 1] - sa * sb / n) / np.sqrt(va * vb))))
    return out


def offset_step(xs: dict, head: dict) -> dict:
    """Coarse offset (all shifts) and the refined offset of one sequence."""
    t_x = xs["t_us"]
    s0, x0 = head["t_us"][0], t_x[0]
    n_slam = int((head["t_us"][-1] - s0) / 1e6 * sync.GRID_HZ)
    n_xs = int((t_x[-1] - x0) / 1e6 * sync.GRID_HZ)
    slam_w = sync.to_grid(head["t_us"], head["omega"], s0, n_slam)
    xs_w = sync.to_grid(t_x, xs["omega"][:, SEG["head"]], x0, n_xs)
    needed = int(min(n_slam, n_xs) * sync.MIN_OVERLAP_FRACTION)
    scan = pearson_scan(slam_w, xs_w, needed)
    if not scan:
        return {"error": "no usable overlap", "passes": False}
    best_k, best_c = max(scan, key=lambda kc: kc[1])
    exclude = sync.PEAK_EXCLUDE_S * sync.GRID_HZ
    others = [c for k, c in scan if abs(k - best_k) > exclude]
    coarse_s = ((s0 - x0) + best_k * 1e6 / sync.GRID_HZ) / 1e6
    refined = npc.refine_offset(head, xs, SEG["head"], coarse_s * 1e6)
    corr = refined.get("corr")
    return {
        "coarse_offset_s": coarse_s,
        "coarse_corr": best_c,
        "best_corr_elsewhere": max(others) if others else None,
        "refined": refined,
        "refined_offset_s": refined.get("offset_s"),
        "refined_corr": corr,
        "passes": bool(corr is not None and corr >= npc.TIMING_MIN_CORR),
    }


# ---------------------------------------------------------------- Xsens on a 30 Hz grid


def to_grid_30(t_us: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linear resampling of [T, ...] values onto a 30 Hz grid starting at the first Xsens time."""
    count = int((t_us[-1] - t_us[0]) / 1e6 * GRID_HZ)
    grid = t_us[0] + np.arange(count) * 1e6 / GRID_HZ
    flat = values.reshape(len(t_us), -1)
    out = np.stack([np.interp(grid, t_us, flat[:, j]) for j in range(flat.shape[1])], 1)
    return grid, out.reshape((count,) + values.shape[1:])


def horizontal_speed(t_us: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Horizontal speed of one segment at the native rate, box-smoothed over BOX_S seconds."""
    velocity = np.gradient(pos[:, :2], t_us / 1e6, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    # a frame without a valid speed (missing position) cannot be called stationary: count it as moving
    speed = np.where(np.isfinite(speed), speed, 1.0)
    return npc.smooth(speed, t_us, BOX_S)


def hysteresis(speed: np.ndarray) -> np.ndarray:
    """True while a foot is stationary: on below V_ON, off above V_OFF, runs shorter than MIN_RUN flipped."""
    state = np.empty(len(speed), dtype=bool)
    current = bool(speed[0] < V_ON)
    for i, v in enumerate(speed):
        if current and v > V_OFF:
            current = False
        elif not current and v < V_ON:
            current = True
        state[i] = current
    for _ in range(3):
        change = np.flatnonzero(np.diff(state.astype(np.int8))) + 1
        edges = np.concatenate([[0], change, [len(state)]])
        short = [(a, b) for a, b in itertools.pairwise(edges) if b - a < MIN_RUN]
        if not short:
            break
        for a, b in short:
            state[a:b] = ~state[a:b]
    return state


def foot_labels(t_us: np.ndarray, pos: np.ndarray) -> dict:
    """Stationary flags of both feet on the 30 Hz grid (a foot is stationary when foot and toe are)."""
    flags = {}
    for side in ("l", "r"):
        speeds = np.maximum(
            horizontal_speed(t_us, pos[:, SEG[f"{side}_foot"]]),
            horizontal_speed(t_us, pos[:, SEG[f"{side}_toe"]]),
        )
        _, on_grid = to_grid_30(t_us, speeds)
        flags[side] = hysteresis(on_grid)
    return flags


# ---------------------------------------------------------------- data checks


def check_segments(grid_pos: np.ndarray) -> dict:
    """Id check (the gate): foot and toe below knee below pelvis while the wearer is upright.

    Side check (reported only, not a gate): the right foot on the right of the walking direction.
    ``grid_pos`` is [N, 23, 3] on the 30 Hz grid (only GRID_SEGMENTS are filled), z up. With a forward
    direction f and up z, f x z points to the right, so (right foot - left foot) . (f x z) is positive
    while walking forward. Walking backward or sideways inverts it, so this test cannot tell swapped
    left/right ids from such walking. The labels (both feet stationary) are symmetric in left and
    right, so a swap could not change them (Amendment A4).
    """
    z = grid_pos[:, :, 2]
    upright = z[:, SEG["pelvis"]] > UPRIGHT_PELVIS_M
    height = None
    if upright.sum() >= MIN_UPRIGHT_FRAMES:
        height = {
            side: float(
                np.mean(
                    (z[upright, SEG[f"{side}_foot"]] < z[upright, SEG[f"{side}_knee"]])
                    & (z[upright, SEG[f"{side}_toe"]] < z[upright, SEG[f"{side}_knee"]])
                    & (z[upright, SEG[f"{side}_knee"]] < z[upright, SEG["pelvis"]])
                )
            )
            for side in ("l", "r")
        }
    pelvis_xy = grid_pos[:, SEG["pelvis"], :2]
    velocity = np.gradient(pelvis_xy, 1.0 / GRID_HZ, axis=0)
    width = int(GRID_HZ * 0.5) | 1
    kernel = np.ones(width) / width
    velocity = np.stack(
        [np.convolve(velocity[:, j], kernel, mode="same") for j in range(2)], 1
    )
    speed = np.linalg.norm(velocity, axis=1)
    walking = speed > WALK_SPEED
    result = {
        "upright_frames": int(upright.sum()),
        "pelvis_z_median_m": float(np.median(z[:, SEG["pelvis"]])),
        "height_order_share": height,
        "walking_frames": int(walking.sum()),
        "right_of_heading_share": None,
        "right_of_heading_share_straight": None,
    }
    if walking.sum() >= MIN_WALK_FRAMES:
        forward = velocity[walking] / speed[walking][:, None]
        right = np.stack([forward[:, 1], -forward[:, 0]], 1)
        lateral = (
            grid_pos[walking, SEG["r_foot"], :2] - grid_pos[walking, SEG["l_foot"], :2]
        )
        correct = np.sum(lateral * right, axis=1) > 0
        result["right_of_heading_share"] = float(np.mean(correct))
        heading = np.arctan2(velocity[:, 1], velocity[:, 0])
        half = int(GRID_HZ * 0.5)
        turn = np.full(len(heading), np.inf)
        change = np.abs(
            np.angle(np.exp(1j * (heading[2 * half :] - heading[: -2 * half])))
        )
        turn[half : len(heading) - half] = np.degrees(change)
        straight = (speed > 1.0) & (turn < STRAIGHT_TURN_DEG)
        result["straight_walking_frames"] = int(straight.sum())
        if straight.sum() >= 100:
            result["right_of_heading_share_straight"] = float(
                np.mean(
                    np.sum(
                        (
                            grid_pos[straight, SEG["r_foot"], :2]
                            - grid_pos[straight, SEG["l_foot"], :2]
                        )
                        * np.stack([velocity[straight, 1], -velocity[straight, 0]], 1)
                        / speed[straight][:, None],
                        axis=1,
                    )
                    > 0
                )
            )
    height_ok = None if height is None else min(height.values()) >= ORDER_MIN_SHARE
    side = result["right_of_heading_share"]
    side_ok = None if side is None else side >= SIDE_MIN_SHARE
    result["height_ok"] = height_ok
    result["side_ok"] = side_ok
    result["ids_tested"] = height_ok is not None
    result["ids_ok"] = height_ok is not False
    return result


def check_contacts(t_us: np.ndarray, contacts: np.ndarray, flags: dict) -> dict:
    """Agreement of each ``foot_contacts`` column with the stationary flag of each foot (side check only)."""
    _, grid = to_grid_30(t_us, contacts.astype(float))
    n = min(len(grid), len(flags["l"]))
    columns = []
    for j in range(grid.shape[1]):
        on = grid[:n, j] > 0.5
        columns.append(
            {
                "mean": float(grid[:n, j].mean()),
                "agree_left_stationary": float(np.mean(on == flags["l"][:n])),
                "agree_right_stationary": float(np.mean(on == flags["r"][:n])),
            }
        )
    return {"columns": columns}


def prevalence(flags: dict, head_speed: np.ndarray, pelvis_z: np.ndarray) -> dict:
    n = min(len(flags["l"]), len(flags["r"]), len(head_speed), len(pelvis_z))
    left, right = flags["l"][:n], flags["r"][:n]
    both = left & right
    hard = both & (head_speed[:n] > HEAD_HARD_SPEED)
    upright = pelvis_z[:n] > UPRIGHT_PELVIS_M
    return {
        "seconds_total": float(n / GRID_HZ),
        "seconds_both_planted": float(both.sum() / GRID_HZ),
        "seconds_one_planted": float((left ^ right).sum() / GRID_HZ),
        "seconds_none_planted": float((~left & ~right).sum() / GRID_HZ),
        "seconds_hard_stratum": float(hard.sum() / GRID_HZ),
        "seconds_upright": float(upright.sum() / GRID_HZ),
        "seconds_both_planted_upright": float((both & upright).sum() / GRID_HZ),
    }


# ---------------------------------------------------------------- one sequence and the run


def load_sequence(folder: Path) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    body, head_zip = (
        sync.find(folder, "body_motion.zip"),
        sync.find(folder, "recording_head.zip"),
    )
    raw = sync.load_xsens(body)
    t_us = raw["timestamps_us"].astype(np.float64)
    pos = raw["segment_tXYZ"].reshape(len(t_us), -1, 3)
    omega = np.linalg.norm(
        raw["segment_angularVelocity"].reshape(len(t_us), -1, 3), axis=2
    )
    head = npc.load_trajectory(head_zip, "recording_head")
    contacts = raw["foot_contacts"]
    keep = np.concatenate(
        [[True], np.diff(t_us) > 0]
    )  # repeated or backward timestamps
    dropped = int((~keep).sum())
    if dropped:
        t_us, pos, omega, contacts = t_us[keep], pos[keep], omega[keep], contacts[keep]
    return {"t_us": t_us, "omega": omega, "dropped": dropped}, head, pos, contacts


def analyse(folder: Path) -> dict:
    xs, head, pos, contacts = load_sequence(folder)
    result: dict = {
        "xsens_frames": len(xs["t_us"]),
        "xsens_frames_dropped_bad_timestamp": xs["dropped"],
        "offset": offset_step(xs, head),
    }
    if not result["offset"]["passes"]:
        return result
    t_us = xs["t_us"]
    grid_t, grid_pos = to_grid_30(t_us, pos[:, list(GRID_SEGMENTS)])
    full = np.zeros((len(grid_t), pos.shape[1], 3))
    full[:, list(GRID_SEGMENTS)] = grid_pos
    result["checks"] = check_segments(full)
    if not result["checks"]["ids_ok"]:
        result["note"] = "segment-id check failed on this sequence: no label built"
        return result
    if not result["checks"]["ids_tested"]:
        result["note"] = (
            "segment ids could not be tested here (too little upright time)"
        )
    flags = foot_labels(t_us, pos)
    _, head_speed = to_grid_30(t_us, horizontal_speed(t_us, pos[:, SEG["head"]]))
    result["prevalence"] = prevalence(flags, head_speed, full[:, SEG["pelvis"], 2])
    result["foot_contacts"] = check_contacts(t_us, contacts, flags)
    return result


def provenance(args) -> dict:
    here = Path(__file__)
    code = hashlib.sha256()
    for path in (here, Path(npc.__file__), Path(sync.__file__)):
        code.update(path.read_bytes())
    return {
        "code_sha256_16": code.hexdigest()[:16],
        "cohort_target": args.cohort,
        "v_on": V_ON,
        "v_off": V_OFF,
        "min_run": MIN_RUN,
        "grid_hz": GRID_HZ,
        "timing_min_corr": npc.TIMING_MIN_CORR,
        "numpy": np.__version__,
    }


def summarise(report: dict, target: int) -> dict:
    sequences = report["sequences"]
    passed = [s for s, r in sequences.items() if r.get("offset", {}).get("passes")]
    failed = {
        s: r["offset"].get("refined_corr", r["offset"].get("error"))
        for s, r in sequences.items()
        if "offset" in r and not r["offset"].get("passes")
    }
    errors = {s: r["error"] for s, r in sequences.items() if "error" in r}
    labelled = [s for s in passed if "prevalence" in sequences[s]]
    totals = {
        key: sum(sequences[s]["prevalence"][key] for s in labelled) / 60.0
        for key in (
            "seconds_total",
            "seconds_both_planted",
            "seconds_one_planted",
            "seconds_none_planted",
            "seconds_hard_stratum",
            "seconds_upright",
            "seconds_both_planted_upright",
        )
    }
    totals = {k.replace("seconds", "minutes"): v for k, v in totals.items()}
    ids_failed = [s for s in passed if not sequences[s]["checks"]["ids_ok"]]
    ids_untested = [
        s
        for s in passed
        if sequences[s]["checks"]["ids_ok"] and not sequences[s]["checks"]["ids_tested"]
    ]
    side_inconsistent = [
        s for s in passed if sequences[s]["checks"].get("side_ok") is False
    ]
    indicative = (
        len(passed) < MIN_COHORT_FOR_CLAIM
        or totals["minutes_both_planted"] < MIN_BOTH_PLANTED_MIN
    )
    return {
        "attempted": len(sequences),
        "passed_offset_step": len(passed),
        "cohort_target": target,
        "cohort_complete": len(passed) >= target,
        "failed_offset_step": failed,
        "load_errors": errors,
        "segment_ids_failed": ids_failed,
        "segment_ids_not_testable": ids_untested,
        "side_check_inconsistent": side_inconsistent,
        "labelled_sequences": len(labelled),
        "totals_minutes": totals,
        "indicative_only": bool(indicative),
    }


def write_report(path: Path, report: dict, target: int) -> None:
    report["summary"] = summarise(report, target)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=1), encoding="utf-8")
    tmp.replace(path)


def load_or_start(args, prov: dict) -> dict:
    if args.out.exists():
        old = json.loads(args.out.read_text(encoding="utf-8"))
        if old.get("run", {}).get("code_sha256_16") == prov["code_sha256_16"]:
            return old
        if not args.fresh:
            raise SystemExit(
                f"{args.out} was made by other code ({old.get('run', {}).get('code_sha256_16')}"
                f" vs {prov['code_sha256_16']}); use --fresh to archive it and start again"
            )
        archive = args.out.parent / "_previous"
        archive.mkdir(exist_ok=True)
        args.out.replace(
            archive / f"{args.out.stem}_{time.strftime('%Y%m%dT%H%M%S')}.json"
        )
    return {"run": prov, "sequences": {}}


def line(name: str, result: dict) -> str:
    offset = result.get("offset", {})
    corr = offset.get("refined_corr")
    head = f"{name[:44]:44s} corr {corr if corr is None else round(corr, 3)}"
    if not offset.get("passes"):
        return head + "  OFFSET STEP FAILED"
    checks = result["checks"]
    if "prevalence" not in result:
        return head + f"  ID CHECK FAILED {json.dumps(checks)}"
    p = result["prevalence"]
    total = p["seconds_total"]
    return (
        head
        + f"  both {100 * p['seconds_both_planted'] / total:4.0f}%"
        + f" one {100 * p['seconds_one_planted'] / total:4.0f}%"
        + f" none {100 * p['seconds_none_planted'] / total:4.0f}%"
        + f"  hard {p['seconds_hard_stratum'] / 60:5.1f} min"
        + f"  upright {p['seconds_upright'] / 60:5.1f} min"
        + f"  walk {checks['walking_frames']} lr {checks['right_of_heading_share']}"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, default=sync.DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--cohort", type=int, default=60)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument(
        "--sequence", action="append", help="only these (default: all, in hash order)"
    )
    args = parser.parse_args(argv)
    prov = provenance(args)
    report = load_or_start(args, prov)
    names = args.sequence or ordered_sequences(args.root)
    print(
        f"{len(names)} sequences in hash order, cohort target {args.cohort}", flush=True
    )
    for name in names:
        if summarise(report, args.cohort)["passed_offset_step"] >= args.cohort:
            break
        if name in report["sequences"] and "error" not in report["sequences"][name]:
            continue  # done; a file-reading error is retried, an offset-step failure never is
        started = time.time()
        try:
            result = analyse(args.root / name)
        except (OSError, KeyError, ValueError, zipfile.BadZipFile) as error:
            result = {"error": f"{type(error).__name__}: {error}"}
        result["fold"] = fold_of(name)
        result["seconds_taken"] = round(time.time() - started, 1)
        report["sequences"][name] = result
        write_report(args.out, report, args.cohort)
        print(
            line(name, result)
            if "error" not in result
            else f"{name} {result['error']}",
            flush=True,
        )
    write_report(args.out, report, args.cohort)
    print(json.dumps(report["summary"], indent=1))
    if report["summary"]["segment_ids_failed"]:
        print(
            "STOP: the segment-id check failed on some sequences; no label is trusted"
        )
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
