"""Main experiment of the foot-lock protocol (Nymeria): detectors, held-out metrics, controls.

Implements sections 5 to 7 of ``docs/nymeria_foot_lock_protocol.md`` as fixed by Amendment A5
(definitions that the protocol left open, recorded before the first run). Input is the report of
``nymeria_foot_lock_data_check.py`` (cohort and refined clock offsets); nothing is re-selected here.

Steps (``--step``): ``prepare`` builds one cache file per sequence (head pose on a 30 Hz SLAM-clock
grid, the Xsens-derived labels on a 30 Hz Xsens grid, and the clock offset); ``run`` fits and evaluates
everything from the caches; ``all`` does both. A cache is rebuilt when the data-check report changed.

Detectors (inputs: the SLAM head pose only): B0 always both-planted, B1 a speed threshold, B2 the same
speed with hysteresis, M1 a logistic regression on window features. Every fitted quantity is fitted on
the four other folds of the grouped 5-fold split (fold = first 8 hex digits of SHA-256 of the name,
modulo 5). Each detector has a causal row (windows end at the frame) and a lookahead row (the same
windows end 0.5 s later).
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np

import nymeria_foot_lock_data_check as s0
import nymeria_project_check as npc

DATA_CHECK = s0.OUT
OUT = Path(r"D:\projects\hready_data\review\nymeria_foot_lock_experiment.json")
CACHE = Path(r"D:\projects\hready_data\review\nymeria_foot_lock_experiment_cache")
SEG = npc.SEGMENTS
HZ = 30
MIN_RUN = s0.MIN_RUN
MIN_CLASS_FRAMES = 30  # a rate over fewer than 1 s of frames is not reported
FALSE_LOCK_LIMIT = 0.05
L2_WEIGHT = (
    1.0  # as the C of an L2 logistic regression: C * sum(log loss) + 0.5 * |w|^2
)
LOG_FLOOR = 1e-3
BAND_FLOOR = 1e-7  # a position power in m^2, far smaller than the other features
MAX_SLAM_GAP_S = 0.1  # frames inside a longer gap of the SLAM trajectory are not used
GAP_MARGIN_FRAMES = 75  # nor the 2.5 s around such a gap
PREP_VERSION = "A6"
NOISE_SETTINGS = (
    (0.0, 0.0),
    (0.01, 0.5),
    (0.02, 1.0),
    (0.05, 2.0),
)  # (position sigma m, yaw sigma deg)
CLOCK_SHIFTS_S = (-0.1, -0.05, 0.0, 0.05, 0.1)
SHUFFLE_FRAMES = 60 * HZ
B2_LEVELS = np.linspace(
    0.01, 0.90, 40
)  # quantiles of the training speeds used as thresholds (plus B1's two taus)
SIGNAL_MIN_BA, SIGNAL_MIN_SHARE = 0.60, 0.80
MAIN_MIN_SHARE, MAIN_MIN_DIFF = 0.60, 0.05
BOOT = 1000
BLOCK_BOOT = 200
BLOCK_FRAMES = 10 * HZ
FEATURE_NAMES = (
    "speed_1s",
    "vertical_speed_1s",
    "spread_1s",
    "spread_2s",
    "angular_speed_1s",
    "band_1_3hz_2s",
    "speed_last_0_5s",
)
FEATURE_NAMES_LOOKAHEAD = (*FEATURE_NAMES, "speed_next_0_5s")


def sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


# ---------------------------------------------------------------- prepare: one cache per sequence


def trailing_mean(x: np.ndarray, width: int) -> np.ndarray:
    """Mean of the last ``width`` samples (fewer at the start); uses no later sample."""
    cum = np.concatenate([[0.0], np.cumsum(x)])
    i = np.arange(len(x))
    a = np.maximum(i + 1 - width, 0)
    return (cum[i + 1] - cum[a]) / (i + 1 - a)


def prepare_sequence(folder: Path, offset_s: float, expect_both_s: float) -> dict:
    xs, head, pos, _ = s0.load_sequence(folder)
    t_x = xs["t_us"]
    flags, speeds = {}, []
    for side in ("l", "r"):
        sp = np.maximum(
            s0.horizontal_speed(t_x, pos[:, SEG[f"{side}_foot"]]),
            s0.horizontal_speed(t_x, pos[:, SEG[f"{side}_toe"]]),
        )
        _, on_grid = s0.to_grid_30(t_x, sp)
        flags[side] = s0.hysteresis(on_grid)
        speeds.append(on_grid)
    n = min(len(flags["l"]), len(flags["r"]))
    both = flags["l"][:n] & flags["r"][:n]
    if abs(both.sum() / HZ - expect_both_s) > 1e-6:
        raise RuntimeError(
            f"labels differ from the data-check report ({both.sum() / HZ:.3f} s vs {expect_both_s:.3f} s):"
            " rerun the data check"
        )
    _, head_speed = s0.to_grid_30(t_x, s0.horizontal_speed(t_x, pos[:, SEG["head"]]))
    _, pelvis_z = s0.to_grid_30(t_x, pos[:, SEG["pelvis"], 2])
    offset_us = offset_s * 1e6
    lo = max(head["t_us"][0], t_x[0] + offset_us)
    hi = min(head["t_us"][-1], t_x[-1] + offset_us)
    count = int((hi - lo) / 1e6 * HZ)
    t = lo + np.arange(count) * 1e6 / HZ
    slam_pos = np.stack(
        [np.interp(t, head["t_us"], head["pos"][:, j]) for j in range(3)], 1
    )
    rate_hz = (len(head["t_us"]) - 1) / ((head["t_us"][-1] - head["t_us"][0]) / 1e6)
    omega = trailing_mean(head["omega"], max(1, round(rate_hz * 0.1)))
    after = np.clip(np.searchsorted(head["t_us"], t), 1, len(head["t_us"]) - 1)
    gap_s = (head["t_us"][after] - head["t_us"][after - 1]) / 1e6
    near_gap = (
        window_mean(
            (gap_s > MAX_SLAM_GAP_S).astype(float),
            -GAP_MARGIN_FRAMES,
            GAP_MARGIN_FRAMES,
        )
        > 0
    )
    return {
        "slam_ok": ~near_gap,
        "max_slam_gap_s": np.array(float(np.max(np.diff(head["t_us"])) / 1e6)),
        "slam_t_us": t,
        "head_pos": slam_pos,
        "head_omega": np.interp(t, head["t_us"], omega),
        "x0_us": np.array(t_x[0]),
        "offset_us": np.array(offset_us),
        "both": both,
        "hard": both & (head_speed[:n] > s0.HEAD_HARD_SPEED),
        "xs_head_speed": head_speed[:n],
        "upright": pelvis_z[:n] > s0.UPRIGHT_PELVIS_M,
        "foot_speed": 0.5 * (speeds[0][:n] + speeds[1][:n]),
    }


def cohort_from_data_check(path: Path) -> tuple[list[str], dict, str]:
    report = json.loads(path.read_text(encoding="utf-8"))
    sequences = report["sequences"]
    names = [n for n, v in sequences.items() if "prevalence" in v]
    names.sort(key=s0.hash_key)
    return names, report, sha16(path)


def prepare(
    root: Path, data_check: Path, cache: Path, rebuild: bool, limit: int | None
) -> dict:
    names, report, check_sha = cohort_from_data_check(data_check)
    prep_code = hashlib.sha256(
        (
            inspect.getsource(prepare_sequence)
            + inspect.getsource(trailing_mean)
            + PREP_VERSION
        ).encode()
    ).hexdigest()[:16]
    key = f"{check_sha}-{prep_code}"
    if limit:
        names = names[:limit]
    cache.mkdir(parents=True, exist_ok=True)
    recs = {}
    for number, name in enumerate(names, 1):
        file = cache / f"{name}.npz"
        rec = None
        if file.exists() and not rebuild:
            with np.load(file) as data:
                if str(data["cache_key"]) == key:
                    rec = {k: data[k] for k in data.files if k != "cache_key"}
        if rec is None:
            started = time.time()
            seq = report["sequences"][name]
            rec = prepare_sequence(
                root / name,
                seq["offset"]["refined_offset_s"],
                seq["prevalence"]["seconds_both_planted"],
            )
            np.savez_compressed(file, cache_key=np.array(key), **rec)
            print(
                f"[{number}/{len(names)}] {name[:46]} cached in {time.time() - started:.0f} s",
                flush=True,
            )
        recs[name] = rec
    return {"names": names, "recs": recs, "check_sha": check_sha, "data_check": report}


# ---------------------------------------------------------------- features


def window_mean(x: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Mean of x over frames i+lo .. i+hi (inclusive), clipped to the series, for every i."""
    n = len(x)
    cum = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x, axis=0)])
    i = np.arange(n)
    a, b = np.clip(i + lo, 0, n - 1), np.clip(i + hi, 0, n - 1)
    return (cum[b + 1] - cum[a]) / (b - a + 1).reshape((-1,) + (1,) * (x.ndim - 1))


def base_speeds(pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal and absolute vertical speed of frame i over the 0.2 s before it (frame i minus i-6)."""
    pad = np.pad(pos, ((6, 0), (0, 0)), mode="edge")
    d = (pad[6:] - pad[:-6]) / 0.2
    return np.linalg.norm(d[:, :2], axis=1), np.abs(d[:, 2])


def spread(xy: np.ndarray, length: int, end: int) -> np.ndarray:
    """Root of the summed horizontal variances in the ``length``-frame window that ends ``end`` frames ahead."""
    padded = np.pad(xy, ((length - 1 - end, end), (0, 0)), mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(padded, length, axis=0)
    return np.sqrt(win.var(axis=2).sum(axis=1))


def band_energy(z: np.ndarray, end: int) -> np.ndarray:
    """Power of the vertical head motion at 1 to 3 Hz in the 2 s window that ends ``end`` frames ahead."""
    length = 2 * HZ
    padded = np.pad(z, (length - 1 - end, end), mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(padded, length)
    win = win - win.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(win * np.hanning(length), axis=1)) ** 2
    return power[:, 2:7].sum(axis=1) / length  # bins 2..6 are 1.0 to 3.0 Hz


def features(pos: np.ndarray, omega: np.ndarray, causal: bool) -> np.ndarray:
    """Window features. The causal windows end at the frame and use no later frame. The lookahead
    windows are the same windows ending 0.5 s (15 frames) later; no feature uses a frame beyond that."""
    end = 0 if causal else HZ // 2
    one = (-(HZ - 1) + end, end)
    vh, vz = base_speeds(pos)
    cols = [
        window_mean(vh, *one),
        window_mean(vz, *one),
        spread(pos[:, :2], HZ, end),
        spread(pos[:, :2], 2 * HZ, end),
        window_mean(omega, *one),
        band_energy(pos[:, 2], end),
        window_mean(vh, -(HZ // 2 - 1), 0),
    ]
    if not causal:
        cols.append(window_mean(vh, 1, HZ // 2))
    floors = np.full(len(cols), LOG_FLOOR)
    floors[5] = BAND_FLOOR
    return np.log(floors + np.stack(cols, axis=1))


def add_noise(
    rec: dict, noise: tuple[float, float], name: str, salt: int
) -> tuple[np.ndarray, np.ndarray]:
    pos, omega = rec["head_pos"], rec["head_omega"]
    sigma_pos, sigma_yaw = noise
    if sigma_pos == 0 and sigma_yaw == 0:
        return pos, omega
    rng = np.random.default_rng(int(s0.hash_key(name)[:8], 16) + salt)
    pos = pos + rng.normal(0.0, sigma_pos, pos.shape)
    yaw = rng.normal(0.0, np.radians(sigma_yaw), len(pos))
    rate = np.diff(yaw, prepend=yaw[0]) * HZ
    return pos, np.sqrt(omega**2 + rate**2)


def lookup(rec: dict, shift_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Index into the Xsens-grid labels for every SLAM-grid frame (nearest frame; positive shift delays labels)."""
    idx = np.rint(
        (
            rec["slam_t_us"]
            - float(rec["offset_us"])
            - shift_s * 1e6
            - float(rec["x0_us"])
        )
        / (1e6 / HZ)
    )
    idx = idx.astype(int)
    valid = (idx >= 0) & (idx < len(rec["both"]))
    return np.clip(idx, 0, len(rec["both"]) - 1), valid


# ---------------------------------------------------------------- detectors


def fix_short(state: np.ndarray) -> np.ndarray:
    """Flip runs shorter than MIN_RUN (three passes, as the labels do)."""
    state = state.copy()
    for _ in range(3):
        edges = np.concatenate(
            [[0], np.flatnonzero(np.diff(state.astype(np.int8))) + 1, [len(state)]]
        )
        lengths = np.diff(edges)
        short = lengths < MIN_RUN
        if not short.any():
            break
        state = np.repeat(state[edges[:-1]] ^ short, lengths)
    return state


def hysteresis(score: np.ndarray, on: float, off: float) -> np.ndarray:
    """True (locked) after the score fell below ``on`` and until it rises above ``off`` (on <= off)."""
    event = np.where(score < on, 1, np.where(score > off, -1, 0)).astype(np.int8)
    last = np.maximum.accumulate(np.where(event != 0, np.arange(len(event)), -1))
    state = np.where(last >= 0, event[np.maximum(last, 0)] == 1, False)
    return fix_short(state)


def thresholds(score: np.ndarray, y: np.ndarray) -> dict:
    """Lock when score < tau. tau_ba maximises balanced accuracy; tau_fl5 is the most permissive tau
    whose false-lock rate on these frames is at most 5%. Only cuts between different scores count."""
    order = np.argsort(score, kind="stable")
    s, ys = score[order], y[order]
    pos = ys.sum()
    neg = len(ys) - pos
    if pos == 0 or neg == 0:
        raise RuntimeError("a training set without both classes")
    cum_pos, cum_neg = np.cumsum(ys) / pos, np.cumsum(~ys) / neg
    cut = np.append(s[:-1] < s[1:], True)
    i = int(np.argmax(np.where(cut, 0.5 * (cum_pos + 1 - cum_neg), -np.inf)))
    ok = np.flatnonzero((cum_neg <= FALSE_LOCK_LIMIT) & cut)
    j = int(ok[-1]) if len(ok) else -1

    def mid(k: int) -> float:
        return float(s[k] + 1.0) if k == len(s) - 1 else float(0.5 * (s[k] + s[k + 1]))

    return {"ba": mid(i), "fl5": mid(j) if j >= 0 else float(s[0] - 1.0)}


def search_b2(
    scores: list[np.ndarray], ys: list[np.ndarray], extra: list[float]
) -> dict:
    """(on, off) with the best balanced accuracy, and with the lowest missed-lock rate at false-lock <= 5%.
    The candidates are quantiles of the training speeds plus ``extra`` (B1's two taus), so that B2 can
    always do at least as well as B1 on the training folds."""
    levels = np.unique(
        np.concatenate([np.quantile(np.concatenate(scores), B2_LEVELS), extra])
    )
    pos = sum(int(y.sum()) for y in ys)
    neg = sum(int((~y).sum()) for y in ys)
    best_ba = (-1.0, None)
    best_fl = (2.0, None)
    least_fl = (2.0, None)
    for a, on in enumerate(levels):
        for off in levels[a:]:
            miss = fl = 0
            for s, y in zip(scores, ys, strict=True):
                lock = hysteresis(s, on, off)
                miss += int((~lock & y).sum())
                fl += int((lock & ~y).sum())
            miss_r, fl_r = miss / pos, fl / neg
            ba = 0.5 * (2 - miss_r - fl_r)
            if ba > best_ba[0]:
                best_ba = (ba, (float(on), float(off)))
            if fl_r <= FALSE_LOCK_LIMIT and miss_r < best_fl[0]:
                best_fl = (miss_r, (float(on), float(off)))
            if fl_r < least_fl[0]:
                least_fl = (fl_r, (float(on), float(off)))
    return {
        "ba": best_ba[1],
        "fl5": best_fl[1] or least_fl[1],
        "fl5_feasible": best_fl[1] is not None,
    }


def fit_logistic(x: np.ndarray, y: np.ndarray, iterations: int = 60) -> dict:
    mu, sd = x.mean(axis=0), x.std(axis=0) + 1e-9
    z = np.hstack([(x - mu) / sd, np.ones((len(x), 1))])
    w = np.zeros(z.shape[1])
    reg = np.eye(len(w))
    reg[-1, -1] = 0.0  # the bias is not penalised
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-(z @ w)))
        grad = L2_WEIGHT * z.T @ (p - y) + reg @ w
        hess = L2_WEIGHT * (z.T * (p * (1 - p))) @ z + reg + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-6:
            break
    return {"mu": mu, "sd": sd, "w": w}


def predict(model: dict, x: np.ndarray) -> np.ndarray:
    z = np.hstack([(x - model["mu"]) / model["sd"], np.ones((len(x), 1))])
    return 1.0 / (1.0 + np.exp(-(z @ model["w"])))


# ---------------------------------------------------------------- metrics


def rate(count: int, total: int) -> float | None:
    return float(count / total) if total >= MIN_CLASS_FRAMES else None


def transitions(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = np.diff(state.astype(np.int8))
    return np.flatnonzero(d == 1) + 1, np.flatnonzero(d == -1) + 1


def timing(y: np.ndarray, lock: np.ndarray, max_frames: int = 45) -> dict:
    out = {}
    for name, true_t, det_t in (
        ("onset", transitions(y)[0], transitions(lock)[0]),
        ("offset", transitions(y)[1], transitions(lock)[1]),
    ):
        if len(true_t) == 0 or len(det_t) == 0:
            out[name] = {
                "events": len(true_t),
                "median_abs_ms": None,
                "matched_share": None,
            }
            continue
        k = (
            np.clip(np.searchsorted(det_t, true_t), 1, len(det_t) - 1)
            if len(det_t) > 1
            else np.zeros(len(true_t), int)
        )
        cand = np.stack([det_t[np.maximum(k - 1, 0)], det_t[k]])
        diff = np.abs(cand - true_t).min(axis=0)
        matched = diff <= max_frames
        out[name] = {
            "events": len(true_t),
            "median_abs_ms": float(np.median(diff[matched]) * 1000.0 / HZ)
            if matched.any()
            else None,
            "matched_share": float(matched.mean()),
        }
    return out


def metrics(lock: np.ndarray, d: dict) -> dict:
    y, hard, up, fs = d["y"], d["hard"], d["up"], d["fs"]
    pos, neg = int(y.sum()), int((~y).sum())
    miss = rate(int((~lock & y).sum()), pos)
    false = rate(int((lock & ~y).sum()), neg)
    ba = None if miss is None or false is None else 0.5 * ((1 - miss) + (1 - false))
    tp = int((lock & y).sum())
    f1 = (
        float(2 * tp / (2 * tp + int((~lock & y).sum()) + int((lock & ~y).sum())))
        if tp
        else 0.0
    )
    out = {
        "frames": len(y),
        "planted_frames": pos,
        "balanced_accuracy": ba,
        "f1": f1,
        "missed_lock": miss,
        "false_lock": false,
        "slip_speed_mps": float(fs[lock].mean()) if lock.any() else None,
        "hard_missed_lock": rate(int((~lock & hard).sum()), int(hard.sum())),
        "timing": timing(y, lock),
    }
    for name, mask in (("upright", up),):
        yy, ll = y[mask], lock[mask]
        m = rate(int((~ll & yy).sum()), int(yy.sum()))
        f = rate(int((ll & ~yy).sum()), int((~yy).sum()))
        out[f"{name}_missed_lock"] = m
        out[f"{name}_false_lock"] = f
        out[f"{name}_balanced_accuracy"] = (
            None if m is None or f is None else 0.5 * (2 - m - f)
        )
    return out


def block_ci(
    y: np.ndarray, lock: np.ndarray, rng: np.random.Generator
) -> list[float] | None:
    """95% interval of the missed-lock rate, resampling 10 s blocks of one sequence."""
    k = len(y) // BLOCK_FRAMES
    if k < 3:
        return None
    n = k * BLOCK_FRAMES
    miss = (~lock[:n] & y[:n]).reshape(k, -1).sum(axis=1)
    pos = y[:n].reshape(k, -1).sum(axis=1)
    draw = rng.integers(0, k, (BLOCK_BOOT, k))
    total = pos[draw].sum(axis=1)
    ok = total >= MIN_CLASS_FRAMES
    if not ok.any():
        return None
    r = miss[draw].sum(axis=1)[ok] / total[ok]
    return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = which == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


# ---------------------------------------------------------------- one setting over all folds


def evaluate(
    recs: dict,
    names: list[str],
    *,
    causal: bool,
    noise=(0.0, 0.0),
    shift=0.0,
    roll=0,
    dets=("B0", "B1", "B2", "M1"),
    keep=False,
    mask=None,
) -> dict:
    """``mask`` (optional): name -> bool array on the SLAM grid; only frames where it is true are used
    (the repo-model protocol uses it to compare every detector on the same frames)."""
    data = {}
    for number, name in enumerate(names):
        rec = recs[name]
        pos, omega = add_noise(rec, noise, name, number)
        x = features(pos, omega, causal)
        idx, valid = lookup(rec, shift)
        valid = valid & rec["slam_ok"]
        if mask is not None:
            valid = valid & mask[name]
        lab = {k: rec[k][idx] for k in ("both", "hard", "upright", "foot_speed")}
        if roll:
            lab = {k: np.roll(v, roll) for k, v in lab.items()}
        data[name] = {
            "x": x[valid],
            "y": lab["both"][valid],
            "hard": lab["hard"][valid],
            "up": lab["upright"][valid],
            "fs": lab["foot_speed"][valid],
        }
    fold = {n: s0.fold_of(n) for n in names}
    out = {n: {} for n in names}
    extra = {"M1_prob": [], "M1_label": [], "params": {}}
    rng = np.random.default_rng(0)
    for f in range(5):
        train = [n for n in names if fold[n] != f]
        test = [n for n in names if fold[n] == f]
        if not test or not train:
            continue
        xtr = np.vstack([data[n]["x"] for n in train])
        ytr = np.concatenate([data[n]["y"] for n in train])
        ops: dict[str, dict] = {}
        if "B1" in dets:
            ops["B1"] = thresholds(xtr[:, 0], ytr)
        if "B2" in dets:
            ops["B2"] = search_b2(
                [data[n]["x"][:, 0] for n in train],
                [data[n]["y"] for n in train],
                list(ops["B1"].values()) if "B1" in ops else [],
            )
        if "M1" in dets:
            model = fit_logistic(xtr, ytr.astype(float))
            neg_p = thresholds(-predict(model, xtr), ytr)
            ops["M1"] = {k: -v for k, v in neg_p.items()}  # lock when p > thr
        extra["params"][f] = {k: v for k, v in ops.items()}
        for n in test:
            d = data[n]
            p = predict(model, d["x"]) if "M1" in dets else None
            for det in dets:
                out[n][det] = {}
                for op in ("ba", "fl5"):
                    if det == "B0":
                        lock = np.ones(len(d["y"]), bool)
                    elif det == "B1":
                        lock = d["x"][:, 0] < ops["B1"][op]
                    elif det == "B2":
                        on, off = ops["B2"][op]
                        lock = hysteresis(d["x"][:, 0], on, off)
                    else:
                        lock = p > ops["M1"][op]
                    m = metrics(lock, d)
                    if keep and op == "fl5" and det in ("B2", "M1"):
                        m["missed_lock_block_ci95"] = block_ci(d["y"], lock, rng)
                    out[n][det][op] = m
            if keep and p is not None:
                extra["M1_prob"].append(p)
                extra["M1_label"].append(d["y"])
    if keep and extra["M1_prob"]:
        extra["M1_ece"] = ece(
            np.concatenate(extra["M1_prob"]), np.concatenate(extra["M1_label"])
        )
    extra.pop("M1_prob")
    extra.pop("M1_label")
    return {"per_sequence": out, "extra": extra}


# ---------------------------------------------------------------- summaries and the success rule


def summarise_values(values: list) -> dict:
    v = np.array([x for x in values if x is not None], dtype=float)
    if len(v) == 0:
        return {"n": 0}
    return {
        "n": len(v),
        "n_missing": int(len(values) - len(v)),
        "median": float(np.median(v)),
        "q25": float(np.percentile(v, 25)),
        "q75": float(np.percentile(v, 75)),
    }


def table(result: dict, dets: tuple[str, ...]) -> dict:
    out = {}
    for det in dets:
        out[det] = {}
        for op in ("ba", "fl5"):
            out[det][op] = {
                key: summarise_values(
                    [
                        result["per_sequence"][n][det][op][key]
                        for n in result["per_sequence"]
                    ]
                )
                for key in (
                    "balanced_accuracy",
                    "f1",
                    "missed_lock",
                    "false_lock",
                    "slip_speed_mps",
                    "hard_missed_lock",
                    "upright_missed_lock",
                    "upright_false_lock",
                    "upright_balanced_accuracy",
                )
            }
    return out


def rule(result: dict, names: list[str], exclude: set[str], decision: bool) -> dict:
    use = [n for n in names if n not in exclude]
    ba = [result["per_sequence"][n]["B1"]["ba"]["balanced_accuracy"] for n in use]
    ok = [b for b in ba if b is not None]
    share = float(np.mean([b >= SIGNAL_MIN_BA for b in ok])) if ok else None
    signal = {
        "evaluable": len(ok),
        "share_at_least_0_60": share,
        "passes": bool(share is not None and share >= SIGNAL_MIN_SHARE),
    }
    diffs = []
    for n in use:
        b2 = result["per_sequence"][n]["B2"]["fl5"]["missed_lock"]
        m1 = result["per_sequence"][n]["M1"]["fl5"]["missed_lock"]
        if b2 is not None and m1 is not None:
            diffs.append(b2 - m1)
    held_out_false_lock = {
        det: summarise_values(
            [result["per_sequence"][n][det]["fl5"]["false_lock"] for n in use]
        ).get("median")
        for det in ("B1", "B2", "M1")
    }
    main = {
        "sequences": len(diffs),
        "dropped_for_missing_value": len(use) - len(diffs),
        "median_held_out_false_lock_at_fl5": held_out_false_lock,
    }
    if diffs:
        d = np.array(diffs)
        rng = np.random.default_rng(1)
        boot = np.median(d[rng.integers(0, len(d), (BOOT, len(d)))], axis=1)
        main |= {
            "share_m1_lower": float(np.mean(d > 0)),
            "median_difference_pp": float(100 * np.median(d)),
            "bootstrap95_pp": [
                float(100 * np.percentile(boot, 2.5)),
                float(100 * np.percentile(boot, 97.5)),
            ],
        }
        main["passes"] = bool(
            main["share_m1_lower"] > MAIN_MIN_SHARE
            and np.median(d) >= MAIN_MIN_DIFF - 1e-9
            and np.percentile(boot, 2.5) > 0
        )
    else:
        main["passes"] = False
    if not signal["passes"]:
        outcome = "signal check failed: the head pose alone is reported as insufficient on this data"
    elif main["passes"]:
        outcome = "rule 2 met"
    else:
        outcome = "rule 2 not met: a learned detector did not beat the hysteresis speed threshold on this data"
    if not decision:
        outcome = (
            "lookahead row, reported only (the rule is applied to the causal rows): "
            + outcome
        )
    return {
        "sequences_used": len(use),
        "signal_check": signal,
        "main_comparison": main,
        "outcome": outcome,
    }


def noise_floor(recs: dict, names: list[str]) -> dict:
    values, medians = [], []
    for n in names:
        rec = recs[n]
        idx, valid = lookup(rec, 0.0)
        mask = (
            valid
            & rec["slam_ok"]
            & rec["both"][idx]
            & (rec["xs_head_speed"][idx] < 0.02)
        )
        if mask.sum() < MIN_CLASS_FRAMES:
            continue
        vh, _ = base_speeds(rec["head_pos"])
        values.append(vh[mask])
        medians.append(float(np.median(vh[mask])))
    if not values:
        return {"sequences": 0}
    allv = np.concatenate(values)
    return {
        "sequences": len(values),
        "frames": len(allv),
        "slam_head_speed_quantiles_mps": {
            q: float(np.percentile(allv, q)) for q in (50, 90, 95, 99)
        },
        "per_sequence_median_mps": summarise_values(medians),
        "note": "combined noise of SLAM and Xsens head sway, on both-planted frames with Xsens head speed below 0.02 m/s",
    }


def control_row(result: dict, dets: tuple[str, ...]) -> dict:
    return {
        det: {
            op: {
                key: summarise_values(
                    [
                        result["per_sequence"][n][det][op][key]
                        for n in result["per_sequence"]
                    ]
                ).get("median")
                for key in ("balanced_accuracy", "missed_lock", "false_lock")
            }
            for op in ("ba", "fl5")
        }
        for det in dets
    }


def clean(obj):
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.generic):
        return clean(obj.item())
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def write_report(report: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(clean(report), indent=1), encoding="utf-8")
    tmp.replace(out)


def run(prepared: dict, out: Path) -> dict:
    names, recs = prepared["names"], prepared["recs"]
    side = set(prepared["data_check"]["summary"].get("side_check_inconsistent", []))
    report: dict = {
        "provenance": {
            "data_check_report_sha256_16": prepared["check_sha"],
            "code_sha256_16": sha16(Path(__file__)),
            "data_check_code_sha256_16": sha16(Path(s0.__file__)),
            "sequences": len(names),
            "features_causal": FEATURE_NAMES,
            "features_lookahead": FEATURE_NAMES_LOOKAHEAD,
            "prep_version": PREP_VERSION,
            "slam_gaps": {
                n: {
                    "max_gap_s": float(recs[n]["max_slam_gap_s"]),
                    "share_frames_excluded": float(1 - recs[n]["slam_ok"].mean()),
                }
                for n in names
            },
        },
        "main": {},
    }
    started = time.time()
    for causal in (True, False):
        label = "causal" if causal else "lookahead"
        print(f"main run, {label} ...", flush=True)
        result = evaluate(recs, names, causal=causal, keep=True)
        dets = ("B0", "B1", "B2", "M1")
        report["main"][label] = {
            "per_sequence": result["per_sequence"],
            "table": table(result, dets),
            "rule": rule(result, names, set(), causal),
            "rule_without_side_check_inconsistent": rule(result, names, side, causal),
            "M1_ece_pooled": result["extra"].get("M1_ece"),
            "operating_points_by_fold": result["extra"]["params"],
        }
        print(
            f"  done in {time.time() - started:.0f} s; {report['main'][label]['rule']['outcome']}",
            flush=True,
        )
    write_report(report, out)
    controls: dict = {"noise_floor": noise_floor(recs, names)}
    print("controls: head-pose noise sweep (causal) ...", flush=True)
    controls["noise_sweep"] = {
        f"position_{p * 100:g}cm_yaw_{y:g}deg": control_row(
            evaluate(recs, names, causal=True, noise=(p, y)), ("B1", "B2", "M1")
        )
        for p, y in NOISE_SETTINGS
    }
    print("controls: clock shift (causal and lookahead rows) ...", flush=True)
    controls["clock_shift"] = {
        row: {
            f"{round(s * 1000):+d}ms": control_row(
                evaluate(
                    recs, names, causal=(row == "causal"), shift=s, dets=("B1", "M1")
                ),
                ("B1", "M1"),
            )
            for s in CLOCK_SHIFTS_S
        }
        for row in ("causal", "lookahead")
    }
    print("controls: labels circularly shifted by 60 s (causal) ...", flush=True)
    controls["shuffled_labels"] = control_row(
        evaluate(recs, names, causal=True, roll=SHUFFLE_FRAMES, dets=("B1", "M1")),
        ("B1", "M1"),
    )
    report["controls"] = controls
    write_report(report, out)
    return report


def med_text(v: float | None) -> str:
    return "   -  " if v is None else f"{v:6.3f}"


def show(report: dict) -> None:
    for label, block in report["main"].items():
        print(f"\n== {label} rows (median over held-out sequences) ==")
        print(
            f"{'det':4s} {'op':4s} {'BA':>6s} {'missed':>7s} {'false':>7s} {'slip m/s':>9s} {'hard miss':>9s} {'upright BA':>10s}"
        )
        for det, ops in block["table"].items():
            for op, t in ops.items():
                cells = {k: med_text(t[k].get("median")) for k in t}
                print(
                    f"{det:4s} {op:4s} {cells['balanced_accuracy']} {cells['missed_lock']}  {cells['false_lock']}  "
                    f"{cells['slip_speed_mps']}   {cells['hard_missed_lock']}    {cells['upright_balanced_accuracy']}"
                )
        r = block["rule"]
        print("signal check:", json.dumps(r["signal_check"]))
        print("main comparison:", json.dumps(r["main_comparison"]))
        print("outcome:", r["outcome"])
        r2 = block["rule_without_side_check_inconsistent"]
        print(
            "without the left/right-inconsistent sequences:",
            json.dumps(r2["main_comparison"]),
        )
        print("M1 pooled ECE:", block["M1_ece_pooled"])
        points = block["operating_points_by_fold"]
        infeasible = [f for f, v in points.items() if not v["B2"]["fl5_feasible"]]
        print(
            "folds where B2 could not reach false-lock <= 5% on training:",
            infeasible or "none",
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--step", choices=("prepare", "run", "all"), default="all")
    parser.add_argument("--root", type=Path, default=s0.sync.DEFAULT_ROOT)
    parser.add_argument("--data-check", type=Path, default=DATA_CHECK)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument(
        "--rebuild", action="store_true", help="rebuild every cache file"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="use only the first N cohort sequences (testing)",
    )
    args = parser.parse_args(argv)
    prepared = prepare(args.root, args.data_check, args.cache, args.rebuild, args.limit)
    if args.step == "prepare":
        print(f"{len(prepared['names'])} sequences cached in {args.cache}")
        return 0
    report = run(prepared, args.out)
    show(report)
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
