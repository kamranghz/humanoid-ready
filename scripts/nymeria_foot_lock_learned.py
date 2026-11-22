"""Learned foot-lock detector on raw head-pose windows (Nymeria): plan, data, training, test, controls.

Implements ``docs/nymeria_foot_lock_learned_protocol.md`` (sections 2 to 5, amendments A1 to A3). The label,
the clock offsets, the hard stratum, the upright rule, the SLAM-gap rule, the operating-point search, the
metrics and the first-protocol baselines are NOT re-implemented: they are imported from
``nymeria_foot_lock_experiment`` (and ``nymeria_foot_lock_data_check``), unchanged.

Steps (``--step``):
  plan      split of the training pool (validation groups, held-out locations, nested subsets); no model, no test data
  prepare   one cache file per sequence (head pose and rotation on the 30 Hz SLAM grid, labels); pool and test
  train     train one variant (main / newplace / shuffle / frac25 / frac50 / noise) for one or both sizes and seeds;
            reads ONLY the training and validation sequences of the plan (never a test sequence)
  choose    choose the size on the validation loss (protocol 2.4); reads training logs only
  test      evaluate the chosen size once on the test sequences (scope main) or on the held-out-location test
            sequences (scope newplace); applies the recorded rule
  controls  C1 to C6 of protocol section 4 (scope main)

Everything that touches a test sequence is in ``run_test`` and ``run_controls``; they are the only functions that
load the test caches.
"""

from __future__ import annotations

import argparse
import copy
import csv
import glob
import hashlib
import inspect
import io
import json
import math
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

import nymeria_foot_lock_data_check as s0
import nymeria_foot_lock_experiment as exp
import nymeria_project_check as npc

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError:  # plan and prepare need no torch
    torch = None
    nn = None

DATA = Path(r"D:\projects\hready_data")
REVIEW = DATA / "review"
POOL_CHECK = REVIEW / "nymeria_learned_training_data_check.json"
TEST_CHECK = s0.OUT
PLAN = REVIEW / "nymeria_learned_plan.json"
CACHE = REVIEW / "nymeria_learned_cache"
MODELS = REVIEW / "nymeria_learned_models"
CHOICE = REVIEW / "nymeria_learned_size_choice.json"
RESULT = REVIEW / "nymeria_learned_test_result.json"
RESULT_NEW = REVIEW / "nymeria_learned_newplace_result.json"
CONTROLS = REVIEW / "nymeria_learned_controls.json"
FIRST_PROTOCOL_RESULT = exp.OUT

HZ = exp.HZ
PREP_VERSION = "L1"
WINDOW = 64  # frames (2.13 s)
LOOK = 15  # lookahead row: the window ends 15 frames after the labelled frame
STEP_SCALE = 30.0  # a per-frame step in m becomes m/s
CHANNELS = 15
KERNEL = 5
DROPOUT = 0.1
LR = 1e-3
WEIGHT_DECAY = 1e-2
BATCH = 256
EPOCH_STEPS = 400  # one "epoch" = 400 steps of 256 windows drawn at random
MAX_EPOCHS = 30
PATIENCE = 5
VAL_STRIDE = 10  # validation loss: every 10th window end
SEEDS = (0, 1, 2)
SIZE_TIE = 0.002
SIZES = {
    "S": {"blocks": 4, "width": 64, "dilations": (1, 2, 4, 8)},
    "M": {"blocks": 6, "width": 192, "dilations": (1, 2, 4, 8, 16, 32)},
}
VAL_MOD = 5
ROWS = ("causal", "lookahead")
OPS = ("ba", "fl5")
META_KEYS = (
    "participant_gender",
    "participant_height_cm",
    "participant_weight_kg",
    "participant_age_group",
)
EXPECT = {  # fixed in protocol 8.0 and 8.0a before any model
    "pool": 117,
    "test": 60,
    "held_out_locations": 9,
    "held_out_test": 15,
    "removed_training": 24,
    "remaining_training": 93,
}
NEW_PLACE_MIN_TEST = 15
FRACTIONS = (0.25, 0.5, 1.0)
SHUFFLES = {"shuffle": 60 * HZ, "shuffle90": 90 * HZ, "shuffle120": 120 * HZ}  # C1: three label shifts
C1_MIN_GAIN = 0.02
VARIANTS = ("main", "newplace", *SHUFFLES, "frac25", "frac50", "noise")
# Network trained with noise (docs/nymeria_foot_lock_noise_protocol.md, section 2): the only change against the main run
NOISE_RESULT = REVIEW / "nymeria_learned_noise_result.json"
NOISE_CLEAN_SHARE = 0.2  # share of training sequences (per epoch) that stay clean
NOISE_SIGMA_CM = (0.3, 8.0)  # position sigma, log-uniform
NOISE_YAW_DEG_PER_CM = 0.45  # yaw sigma in degrees = 0.45 x position sigma in cm
NOISE_VAL_SEED = 20261121  # one fixed noisy copy of the validation sequences
CORR_FRAMES = 30  # correlation time of the descriptive correlated noise: 1 s
CORR_LEVELS = ((0.01, 0.5), (0.02, 1.0), (0.05, 2.0))
NOISE_RULE_LEVELS = ((0.01, 0.5), (0.02, 1.0))  # rule item 1
NOISE_CLEAN_TOLERANCE = 0.03  # rule item 2: at most 3 points above the clean-trained network on clean poses
NOISE_SPEC = {
    "clean_share": NOISE_CLEAN_SHARE,
    "sigma_cm_log_uniform": list(NOISE_SIGMA_CM),
    "yaw_deg_per_cm": NOISE_YAW_DEG_PER_CM,
    "validation_noise_seed": NOISE_VAL_SEED,
}


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def labelled_names(report_path: Path) -> list[str]:
    report = read_json(report_path)
    names = [n for n, v in report["sequences"].items() if "prevalence" in v]
    names.sort(key=s0.hash_key)
    return names


# ---------------------------------------------------------------- plan (no model, no test data)


def metadata(root: Path, name: str) -> dict:
    files = glob.glob(str(root / name / "*_metadata.json"))
    if not files:
        raise RuntimeError(f"no metadata file in {name}")
    return json.loads(Path(files[0]).read_text(encoding="utf-8"))


def group_key(meta: dict) -> str:
    return "|".join(str(meta.get(k)) for k in META_KEYS)


def group_hash(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def in_validation(key: str) -> bool:
    return int(group_hash(key)[:8], 16) % VAL_MOD == 0


def split_names(names: list[str], groups: dict) -> tuple[list[str], list[str]]:
    val = [n for n in names if in_validation(groups[n])]
    train = [n for n in names if not in_validation(groups[n])]
    if not val or not train:
        raise RuntimeError("the hash rule left an empty training or validation set")
    return train, val


def nested_subsets(train: list[str], groups: dict) -> dict:
    """Prefixes of the training groups in hash order that cover at least 25%, 50% and 100% of the sequences."""
    order = sorted({groups[n] for n in train}, key=group_hash)
    out = {}
    for frac in FRACTIONS:
        chosen: list[str] = []
        for g in order:
            if len(chosen) >= frac * len(train) - 1e-9:
                break
            chosen += [n for n in train if groups[n] == g]
        out[str(frac)] = chosen
    return out


def held_out_locations(test: list[str], pool: list[str], loc: dict) -> dict:
    """Protocol 3A item 1: test locations in SHA-256 order until they hold at least 15 test sequences."""
    test_locs = sorted(
        {loc[n] for n in test},
        key=lambda s: hashlib.sha256(str(s).encode("utf-8")).hexdigest(),
    )
    held, count = [], 0
    for place in test_locs:
        if count >= NEW_PLACE_MIN_TEST:
            break
        held.append(place)
        count += sum(1 for n in test if loc[n] == place)
    held_set = set(held)
    removed = [n for n in pool if loc[n] in held_set]
    remaining = [n for n in pool if loc[n] not in held_set]
    return {
        "held_out_locations": held,
        "test_sequences": [n for n in test if loc[n] in held_set],
        "removed_training": removed,
        "remaining_training": remaining,
    }


def make_plan(root: Path) -> dict:
    pool = labelled_names(POOL_CHECK)
    test = labelled_names(TEST_CHECK)
    if set(pool) & set(test):
        raise RuntimeError("STOP: a sequence is in both the training pool and the test set")
    if (len(pool), len(test)) != (EXPECT["pool"], EXPECT["test"]):
        raise RuntimeError(
            f"STOP: expected {EXPECT['pool']} pool and {EXPECT['test']} test sequences (protocol 8.0), "
            f"found {len(pool)} and {len(test)}. Send this line to Claude."
        )
    meta = {n: metadata(root, n) for n in pool + test}
    missing = sorted({k for n in pool + test for k in META_KEYS if meta[n].get(k) is None})
    if missing:
        raise RuntimeError(f"STOP: metadata keys missing or None in some sequences: {missing}. Send this line to Claude.")
    groups = {n: group_key(meta[n]) for n in pool + test}
    shared = {groups[n] for n in pool} & {groups[n] for n in test}
    if shared:
        raise RuntimeError(
            f"STOP: {len(shared)} metadata combinations occur in both the training pool and the test set (protocol 8.0 "
            "excluded these). Send this line to Claude."
        )
    loc = {n: meta[n].get("location") for n in pool + test}
    train, val = split_names(pool, groups)
    held = held_out_locations(test, pool, loc)
    got = (
        len(held["held_out_locations"]),
        len(held["test_sequences"]),
        len(held["removed_training"]),
        len(held["remaining_training"]),
    )
    want = tuple(
        EXPECT[k]
        for k in ("held_out_locations", "held_out_test", "removed_training", "remaining_training")
    )
    if got != want:
        raise RuntimeError(
            f"STOP: held-out locations give {got}, protocol 8.0a says {want}. Send this line to Claude."
        )
    new_train, new_val = split_names(held["remaining_training"], groups)
    plan = {
        "pool": pool,
        "test": test,
        "group": groups,
        "location": loc,
        "main": {"train": train, "validation": val},
        "newplace": {**held, "train": new_train, "validation": new_val},
        "nested_train_subsets": nested_subsets(train, groups),
        "counts": {
            "pool": len(pool),
            "test": len(test),
            "main_train": len(train),
            "main_validation": len(val),
            "main_validation_groups": len({groups[n] for n in val}),
            "main_train_groups": len({groups[n] for n in train}),
            "newplace_train": len(new_train),
            "newplace_validation": len(new_val),
            "held_out": dict(zip(
                ("locations", "test_sequences", "training_removed", "training_remaining"), got, strict=True
            )),
            "test_locations_in_pool": len({loc[n] for n in test} & {loc[n] for n in pool}),
            "test_locations": len({loc[n] for n in test}),
        },
    }
    return plan


def show_plan(plan: dict) -> None:
    print(json.dumps(plan["counts"], indent=1))
    for k, v in plan["nested_train_subsets"].items():
        print(f"  nested training subset {k}: {len(v)} sequences")


# ---------------------------------------------------------------- prepare: one cache per sequence


def read_gravity(folder: Path) -> np.ndarray:
    """Gravity in the world frame from the first row of the head trajectory (must point down: world z is up)."""
    head_zip = s0.sync.find(folder, "recording_head.zip")
    member = "recording_head/mps/slam/closed_loop_trajectory.csv"
    with zipfile.ZipFile(head_zip) as archive, archive.open(member) as handle:
        reader = csv.reader(io.TextIOWrapper(handle, encoding="utf-8", newline=""))
        header = next(reader)
        row = next(reader)
    names = [f"gravity_{a}_world" for a in "xyz"]
    if not all(n in header for n in names):
        raise RuntimeError(f"no gravity columns in the trajectory file; columns: {header}")
    g = np.array([float(row[header.index(n)]) for n in names])
    if not (np.abs(g[:2]).max() < 0.5 and g[2] < -9.0):
        raise RuntimeError(f"world frame is not z-up (gravity {g.tolist()}); STOP, send this line to Claude")
    return g


def prepare_sequence_learned(folder: Path, offset_s: float, expect_both_s: float) -> dict:
    """The first protocol's cache record (unchanged) plus the head rotation on the same SLAM grid."""
    captured: dict = {}
    original = s0.load_sequence

    def spy(path):  # the head trajectory is loaded once, by the unchanged first-protocol code
        out = original(path)
        captured["head"] = out[1]
        return out

    s0.load_sequence = spy
    try:
        rec = exp.prepare_sequence(folder, offset_s, expect_both_s)
    finally:
        s0.load_sequence = original
    pos, rot = npc.interp_pose(captured["head"], rec["slam_t_us"])
    err = float(np.abs(pos - rec["head_pos"]).max())
    if err > 1e-6:
        raise RuntimeError(f"interpolated position differs from the cache by {err:.2e} m")
    rec["head_rot"] = rot.reshape(len(rot), 9).astype(np.float32)
    rec["gravity_world"] = read_gravity(folder)
    return rec


def prep_key(check_path: Path) -> str:
    code = hashlib.sha256(
        (
            inspect.getsource(prepare_sequence_learned)
            + inspect.getsource(read_gravity)
            + inspect.getsource(exp.prepare_sequence)
            + inspect.getsource(exp.trailing_mean)
            + exp.PREP_VERSION
            + PREP_VERSION
        ).encode()
    ).hexdigest()[:16]
    return f"{exp.sha16(check_path)}-{code}"


def load_caches(
    names: list[str],
    check_path: Path,
    root: Path,
    cache: Path = CACHE,
    build: bool = False,
    rebuild: bool = False,
    limit: int | None = None,
) -> dict:
    report = read_json(check_path)["sequences"]
    key = prep_key(check_path)
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
            if not build:
                raise RuntimeError(f"no valid cache for {name}: run --step prepare first")
            started = time.time()
            seq = report[name]
            rec = prepare_sequence_learned(
                root / name,
                seq["offset"]["refined_offset_s"],
                seq["prevalence"]["seconds_both_planted"],
            )
            np.savez_compressed(file, cache_key=np.array(key), **rec)
            print(f"[{number}/{len(names)}] {name[:46]} cached in {time.time() - started:.0f} s", flush=True)
        recs[name] = rec
    return recs


# ---------------------------------------------------------------- windows and network


def make_windows(P, R, base, n, e):
    """Windows of WINDOW frames ending at frame ``e`` of sequences stored back to back in P [T,3] and R [T,3,3].

    ``base``/``n``: first row and length of each window's sequence; frames outside a sequence are the nearest
    frame of that sequence (edge padding). Channels per frame, all in the device frame of the LAST frame of the
    window: position relative to the last frame (3), step from the previous frame times 30 (3), the first two
    columns of the rotation relative to the last frame (6), and gravity direction in the device frame of the
    frame itself (3). No speed, spread or band power is computed."""
    ar = torch.arange(-(WINDOW - 1), 1, device=e.device)
    loc = torch.minimum(torch.clamp(e[:, None] + ar, min=0), (n[:, None] - 1))
    gi = base[:, None] + loc
    gp = base[:, None] + torch.clamp(loc - 1, min=0)
    p, pp, r = P[gi], P[gp], R[gi]
    re = r[:, -1]
    dp = torch.einsum("bji,bkj->bki", re, p - p[:, -1:, :])
    step = torch.einsum("bji,bkj->bki", re, p - pp) * STEP_SCALE
    rel = torch.einsum("bji,bkjl->bkil", re, r)[..., :2].reshape(len(e), WINDOW, 6)
    grav = r[:, :, 2, :]  # R^T applied to the world up axis
    return torch.cat([dp, step, rel, grav], dim=2).transpose(1, 2).contiguous()


if torch is not None:

    class Block(nn.Module):
        def __init__(self, width: int, dilation: int):
            super().__init__()
            self.conv = nn.Conv1d(
                width, width, KERNEL, dilation=dilation, padding=dilation * (KERNEL - 1) // 2
            )
            self.norm = nn.LayerNorm(width)
            self.drop = nn.Dropout(DROPOUT)

        def forward(self, x):
            h = F.gelu(self.conv(x))
            h = self.norm(h.transpose(1, 2)).transpose(1, 2)
            return x + self.drop(h)

    class Net(nn.Module):
        """Temporal convolutional network; two outputs per frame: 0 is read at the last frame (causal row),
        1 is read LOOK frames before the last frame (lookahead row)."""

        def __init__(self, size: str):
            super().__init__()
            cfg = SIZES[size]
            self.inp = nn.Conv1d(CHANNELS, cfg["width"], 1)
            self.blocks = nn.ModuleList(Block(cfg["width"], d) for d in cfg["dilations"])
            self.out = nn.Conv1d(cfg["width"], 2, 1)

        def forward(self, x):
            h = self.inp(x)
            for b in self.blocks:
                h = b(h)
            o = self.out(h)
            return torch.stack([o[:, 0, -1], o[:, 1, -1 - LOOK]], dim=1)  # [B, 2] logits


def _no_grad(fn):
    return torch.no_grad()(fn) if torch is not None else fn


def count_parameters(net) -> int:
    return sum(p.numel() for p in net.parameters())


# ---------------------------------------------------------------- tables of windows

class Table:
    """All sequences of a list stored back to back, with the window ends and the two targets per end.

    Window ends run over 0 .. N+LOOK-1 of each sequence. End e has a causal target (label of frame e) and a
    lookahead target (label of frame e-LOOK); each has its own validity mask (frame labelled, SLAM ok)."""

    def __init__(self, names, recs, device, roll: int = 0):
        P, R, base, length = [], [], [], []
        seq, end, tc, mc, tl, ml = [], [], [], [], [], []
        offset = 0
        self.ineffective_shift = 0
        for i, name in enumerate(names):
            rec = recs[name]
            pos = rec["head_pos"]
            n = len(pos)
            P.append((pos - pos.mean(axis=0)).astype(np.float32))
            R.append(rec["head_rot"].reshape(n, 3, 3).astype(np.float32))
            idx, valid = exp.lookup(rec, 0.0)
            elig = valid & rec["slam_ok"]
            y = rec["both"][idx]
            if roll:
                y = np.roll(y, roll)
                if not 0.1 * n <= roll % n <= 0.9 * n:
                    self.ineffective_shift += 1  # the circular shift lands too close to no shift at all
            total = n + LOOK
            c_t = np.zeros(total, np.float32)
            c_m = np.zeros(total, bool)
            l_t = np.zeros(total, np.float32)
            l_m = np.zeros(total, bool)
            c_t[:n], c_m[:n] = y, elig
            l_t[LOOK:], l_m[LOOK:] = y, elig
            tc.append(c_t), mc.append(c_m), tl.append(l_t), ml.append(l_m)
            seq.append(np.full(total, i))
            end.append(np.arange(total))
            base.append(offset)
            length.append(n)
            offset += n
        if self.ineffective_shift > 0.1 * len(names):
            raise RuntimeError(
                f"the label shift of {roll} frames is within 10% of no shift on {self.ineffective_shift} of {len(names)} sequences: "
                "these sequences are too short for this control (send this line to Claude)"
            )
        dev = device
        self.P = torch.as_tensor(np.concatenate(P), device=dev)
        self.R = torch.as_tensor(np.concatenate(R), device=dev)
        base_a, len_a = np.array(base), np.array(length)
        seq_a = np.concatenate(seq)
        self.base = torch.as_tensor(base_a[seq_a], device=dev)
        self.n = torch.as_tensor(len_a[seq_a], device=dev)
        self.end = torch.as_tensor(np.concatenate(end), device=dev)
        self.tc = torch.as_tensor(np.concatenate(tc), device=dev)
        self.mc = torch.as_tensor(np.concatenate(mc).astype(np.float32), device=dev)
        self.tl = torch.as_tensor(np.concatenate(tl), device=dev)
        self.ml = torch.as_tensor(np.concatenate(ml).astype(np.float32), device=dev)
        usable = (self.mc > 0) | (self.ml > 0)
        self.usable = torch.nonzero(usable).squeeze(1)
        self.val_ends = self.usable[(self.end[self.usable] % VAL_STRIDE) == 0]

    def set_inputs(self, names, recs):
        """Replace the head pose of every sequence (same names, same order, same lengths), e.g. by a noisy copy."""
        P, R = [], []
        for name in names:
            pos = recs[name]["head_pos"]
            P.append((pos - pos.mean(axis=0)).astype(np.float32))
            R.append(recs[name]["head_rot"].reshape(len(pos), 3, 3).astype(np.float32))
        self.P = torch.as_tensor(np.concatenate(P), device=self.P.device)
        self.R = torch.as_tensor(np.concatenate(R), device=self.R.device)

    def batch(self, sel):
        x = make_windows(self.P, self.R, self.base[sel], self.n[sel], self.end[sel])
        return x, (self.tc[sel], self.mc[sel], self.tl[sel], self.ml[sel])


def loss_parts(logits, tc, mc, tl, ml):
    bc = F.binary_cross_entropy_with_logits(logits[:, 0], tc, reduction="none")
    bl = F.binary_cross_entropy_with_logits(logits[:, 1], tl, reduction="none")
    return (bc * mc).sum(), mc.sum(), (bl * ml).sum(), ml.sum()


def mean_loss(sc, nc, sl, nl):
    return 0.5 * (sc / nc.clamp(min=1) + sl / nl.clamp(min=1))


@_no_grad
def validation_loss(net, table, chunk: int = 4096) -> dict:
    net.eval()
    tot = torch.zeros(4, device=table.P.device, dtype=torch.float64)
    for i in range(0, len(table.val_ends), chunk):
        sel = table.val_ends[i : i + chunk]
        x, tgt = table.batch(sel)
        parts = loss_parts(net(x), *tgt)
        tot += torch.stack([p.double() for p in parts])
    sc, nc, sl, nl = (float(v) for v in tot)
    lc, ll = sc / max(nc, 1), sl / max(nl, 1)
    return {"loss": 0.5 * (lc + ll), "loss_causal": lc, "loss_lookahead": ll, "windows": int(nc + nl)}


def prior_loss(train: Table, val: Table) -> float:
    """Loss of the constant base-rate predictor (fitted on the training targets), on the validation windows."""
    out = []
    for t_name, m_name in (("tc", "mc"), ("tl", "ml")):
        mt, tt = getattr(train, m_name), getattr(train, t_name)
        p0 = float((tt * mt).sum() / mt.sum().clamp(min=1))
        p0 = min(max(p0, 1e-6), 1 - 1e-6)
        sel = val.val_ends
        t, m = getattr(val, t_name)[sel], getattr(val, m_name)[sel]
        ll = -(t * math.log(p0) + (1 - t) * math.log(1 - p0))
        out.append(float((ll * m).sum() / m.sum().clamp(min=1)))
    return 0.5 * sum(out)


def train_one(train, val, recs, size, seed, roll, device, out_path: Path, label: dict, noise_train: bool = False) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr, va = Table(train, recs, device, roll), Table(val, recs, device, roll)
    va_clean = None
    if noise_train:  # protocol noise 2.3: one fixed noisy copy of the validation sequences decides early stopping
        va_clean = va
        va = Table(val, mixture_recs(recs, val, np.random.default_rng(NOISE_VAL_SEED)), device, roll)
    net = Net(size).to(device)
    params = count_parameters(net)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    prior = prior_loss(tr, va)
    best = {"loss": float("inf"), "epoch": 0, "state": None, "parts": None}
    bad, history = 0, []
    started = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        if noise_train:  # protocol noise 2.2: a fresh noisy copy of every training sequence each epoch
            tr.set_inputs(train, mixture_recs(recs, train, np.random.default_rng([seed, epoch, 71])))
        net.train()
        total = torch.zeros((), device=device)
        for _ in range(EPOCH_STEPS):
            sel = tr.usable[torch.randint(len(tr.usable), (BATCH,), generator=gen, device=device)]
            x, tgt = tr.batch(sel)
            loss = mean_loss(*loss_parts(net(x), *tgt))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.detach()
        v = validation_loss(net, va)
        row = {"epoch": epoch, "train_loss": float(total) / EPOCH_STEPS, **v, "seconds": round(time.time() - started)}
        clean_note = ""
        if noise_train:
            row["clean_validation_loss"] = validation_loss(net, va_clean)["loss"]
            clean_note = f", clean val {row['clean_validation_loss']:.4f}"
        history.append(row)
        print(
            f"  {label['variant']} {size} seed {seed} epoch {epoch:2d}: train {row['train_loss']:.4f} "
            f"val {v['loss']:.4f} (causal {v['loss_causal']:.4f}, lookahead {v['loss_lookahead']:.4f}{clean_note}) "
            f"[{row['seconds']} s]",
            flush=True,
        )
        if v["loss"] < best["loss"]:
            best = {"loss": v["loss"], "epoch": epoch, "parts": v,
                    "clean_loss": row.get("clean_validation_loss"),
                    "state": {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    summary = {
        **label,
        "size": size,
        "seed": seed,
        "parameters": params,
        "best_epoch": best["epoch"],
        "best_validation_loss": best["loss"],
        "best_validation_parts": best["parts"],
        "best_epoch_clean_validation_loss": best.get("clean_loss"),
        "possibly_not_converged": bool(best["epoch"] > MAX_EPOCHS - 3),  # best epoch within the last three of the maximum
        "prior_validation_loss": prior,
        "trained": bool(best["loss"] < prior - 0.001),
        "epochs_run": len(history),
        "train_sequences": len(train),
        "validation_sequences": len(val),
        "history": history,
        "torch": torch.__version__,
        "device": str(device),
        "code_sha256_16": exp.sha16(Path(__file__)),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": best["state"], "size": size, "seed": seed, "label": label, "train": train, "validation": val}, out_path)
    out_path.with_suffix(".json").write_text(json.dumps(exp.clean(summary), indent=1), encoding="utf-8")
    flag = "" if summary["trained"] else "  *** did not train: validation loss not below the base-rate loss ***"
    print(
        f"  saved {out_path.name}: best epoch {best['epoch']}, validation loss {best['loss']:.4f} "
        f"(base-rate loss {prior:.4f}), {params:,} parameters{flag}",
        flush=True,
    )
    return summary


def load_net(path: Path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    net = Net(ck["size"]).to(device)
    net.load_state_dict(ck["state"])
    net.eval()
    return net, ck


# ---------------------------------------------------------------- the network on whole sequences


def yaw_rotate(rot: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Rotate every head rotation by a yaw angle (radians, one per frame) about the vertical axis."""
    c, s = np.cos(yaw), np.sin(yaw)
    rz = np.zeros((len(yaw), 3, 3))
    rz[:, 0, 0], rz[:, 0, 1], rz[:, 1, 0], rz[:, 1, 1], rz[:, 2, 2] = c, -s, s, c, 1.0
    return rz @ rot


def ar1(e: np.ndarray, a: float, x0) -> np.ndarray:
    """First-order autoregression along axis 0 with unit stationary variance: x[t] = a x[t-1] + sqrt(1-a^2) e[t],
    x[-1] = x0 (unit variance). Computed in blocks of 512 frames so the powers of a stay in range."""
    b = math.sqrt(1.0 - a * a)
    out = np.empty_like(e)
    prev = x0
    for s in range(0, len(e), 512):
        seg = e[s : s + 512]
        k = np.arange(len(seg), dtype=np.float64).reshape((-1,) + (1,) * (e.ndim - 1))
        ak = a**k
        out[s : s + len(seg)] = a * ak * prev + b * ak * np.cumsum(seg / ak, axis=0)
        prev = out[s + len(seg) - 1]
    return out


def corr_noise_arrays(n: int, sp: float, sy_deg: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """Position noise (n, 3) in m and yaw noise (n,) in rad with a correlation time of CORR_FRAMES frames."""
    a = math.exp(-1.0 / CORR_FRAMES)
    dpos = sp * ar1(rng.normal(0.0, 1.0, (n, 3)), a, rng.normal(0.0, 1.0, 3))
    yaw = np.radians(sy_deg) * ar1(rng.normal(0.0, 1.0, n), a, float(rng.normal()))
    return dpos, yaw


def corr_add_noise(rec: dict, noise: tuple[float, float], name: str, salt: int):
    """``exp.add_noise`` with correlated noise (used only to rerun B2 and M1 under the same noise as the network)."""
    pos, omega = rec["head_pos"], rec["head_omega"]
    sp, sy = noise
    if sp == 0 and sy == 0:
        return pos, omega
    rng = np.random.default_rng(int(s0.hash_key(name)[:8], 16) + salt)
    dpos, yaw = corr_noise_arrays(len(pos), sp, sy, rng)
    rate = np.diff(yaw, prepend=yaw[0]) * HZ
    return pos + dpos, np.sqrt(omega**2 + rate**2)


def draw_noise_level(rng) -> tuple[float, float]:
    """One training level: clean with probability NOISE_CLEAN_SHARE, otherwise sigma_pos log-uniform in NOISE_SIGMA_CM."""
    if rng.random() < NOISE_CLEAN_SHARE:
        return 0.0, 0.0
    cm = math.exp(rng.uniform(math.log(NOISE_SIGMA_CM[0]), math.log(NOISE_SIGMA_CM[1])))
    return cm / 100.0, NOISE_YAW_DEG_PER_CM * cm


def mixture_recs(recs: dict, names: list[str], rng) -> dict:
    """Copy of ``recs`` for ``names`` with independent per-frame noise of a level drawn per sequence (protocol noise 2.1, 2.2)."""
    out = {}
    for n in names:
        rec = recs[n]
        sp, sy = draw_noise_level(rng)
        if sp == 0 and sy == 0:
            out[n] = rec
            continue
        pos = rec["head_pos"]
        rot = rec["head_rot"].reshape(len(pos), 3, 3).astype(np.float64)
        pos = pos + rng.normal(0.0, sp, pos.shape)
        yaw = rng.normal(0.0, np.radians(sy), len(pos))
        out[n] = {**rec, "head_pos": pos, "head_rot": yaw_rotate(rot, yaw).reshape(len(pos), 9).astype(np.float32)}
    return out


def noisy_input(rec: dict, noise: tuple[float, float], name: str, salt: int, corr: bool = False):
    """Head position and rotation with the noise of the first protocol's sweep (same random numbers as
    ``exp.add_noise``): position sigma in m per frame, yaw sigma in degrees per frame about the vertical axis.
    ``corr`` switches to the correlated noise of the descriptive control."""
    pos = rec["head_pos"]
    rot = rec["head_rot"].reshape(len(pos), 3, 3).astype(np.float64)
    sp, sy = noise
    if sp == 0 and sy == 0:
        return pos, rot
    rng = np.random.default_rng(int(s0.hash_key(name)[:8], 16) + salt)
    if corr:
        dpos, yaw = corr_noise_arrays(len(pos), sp, sy, rng)
        return pos + dpos, yaw_rotate(rot, yaw)
    pos = pos + rng.normal(0.0, sp, pos.shape)
    yaw = rng.normal(0.0, np.radians(sy), len(pos))
    c, s = np.cos(yaw), np.sin(yaw)
    rz = np.zeros((len(pos), 3, 3))
    rz[:, 0, 0], rz[:, 0, 1], rz[:, 1, 0], rz[:, 1, 1], rz[:, 2, 2] = c, -s, s, c, 1.0
    return pos, rz @ rot


@_no_grad
def sequence_logits(net, pos, rot, device, chunk: int = 8192) -> np.ndarray:
    """Logits [N+LOOK, 2] for every window end of one sequence (column 0: causal, column 1: lookahead)."""
    n = len(pos)
    P = torch.as_tensor((pos - pos.mean(axis=0)).astype(np.float32), device=device)
    R = torch.as_tensor(rot.astype(np.float32), device=device)
    out = np.empty((n + LOOK, 2), np.float32)
    for i in range(0, n + LOOK, chunk):
        e = torch.arange(i, min(i + chunk, n + LOOK), device=device)
        x = make_windows(P, R, torch.zeros_like(e), torch.full_like(e, n), e)
        out[i : i + len(e)] = net(x).cpu().numpy()
    return out


def all_logits(net, recs, names, device, noise=(0.0, 0.0), corr: bool = False) -> dict:
    net.eval()
    out = {}
    for number, name in enumerate(names):
        pos, rot = noisy_input(recs[name], noise, name, number, corr)
        out[name] = sequence_logits(net, pos, rot, device)
    return out


def row_logits(out: np.ndarray, row: str) -> np.ndarray:
    n = len(out) - LOOK
    return out[:n, 0] if row == "causal" else out[LOOK : LOOK + n, 1]


# ---------------------------------------------------------------- operating points, metrics, comparisons


def frame_data(rec: dict, shift: float):
    idx, valid = exp.lookup(rec, shift)
    valid = valid & rec["slam_ok"]
    lab = {k: rec[k][idx] for k in ("both", "hard", "upright", "foot_speed")}
    d = {"y": lab["both"][valid], "hard": lab["hard"][valid], "up": lab["upright"][valid], "fs": lab["foot_speed"][valid]}
    return d, valid


def validation_points(outs: dict, recs: dict, names: list[str], roll: int = 0) -> dict:
    """Operating points ("ba", "fl5") of each row from the validation sequences only: lock when -logit < tau."""
    points = {}
    for row in ROWS:
        s_all, y_all = [], []
        for n in names:
            rec = recs[n]
            idx, valid = exp.lookup(rec, 0.0)
            valid = valid & rec["slam_ok"]
            y = rec["both"][idx]
            if roll:
                y = np.roll(y, roll)
            s_all.append(-row_logits(outs[n], row).astype(np.float64)[valid])
            y_all.append(y[valid])
        points[row] = exp.thresholds(np.concatenate(s_all), np.concatenate(y_all))
    return points


def test_metrics(outs: dict, recs: dict, names: list[str], points: dict, shift: float = 0.0, rows=ROWS) -> dict:
    res = {}
    for n in names:
        d, valid = frame_data(recs[n], shift)
        res[n] = {}
        for row in rows:
            s = -row_logits(outs[n], row).astype(np.float64)[valid]
            res[n][row] = {op: exp.metrics(s < points[row][op], d) for op in OPS}
    return res


def seed_mean(per_seed: list[dict], n: str, row: str, op: str, key: str):
    vals = [s[n][row][op][key] for s in per_seed]
    return None if any(v is None for v in vals) else float(np.mean(vals))


def mean_over_seeds(per_seed: list[dict], names, row, op, key) -> dict:
    return {n: seed_mean(per_seed, n, row, op, key) for n in names}


def compare(reference: dict, network: dict, names: list[str]) -> dict:
    """Rule item 2: per-sequence difference reference minus network in missed-lock at fl5 (positive: network better)."""
    diffs = [reference[n] - network[n] for n in names if reference.get(n) is not None and network.get(n) is not None]
    out = {"sequences": len(diffs), "dropped_for_missing_value": len(names) - len(diffs)}
    if not diffs:
        out["passes"] = False
        return out
    d = np.array(diffs)
    rng = np.random.default_rng(1)
    boot = np.median(d[rng.integers(0, len(d), (exp.BOOT, len(d)))], axis=1)
    out |= {
        "share_network_lower": float(np.mean(d > 0)),
        "median_difference_pp": float(100 * np.median(d)),
        "bootstrap95_pp": [float(100 * np.percentile(boot, 2.5)), float(100 * np.percentile(boot, 97.5))],
    }
    out["passes"] = bool(
        out["share_network_lower"] > exp.MAIN_MIN_SHARE
        and np.median(d) >= exp.MAIN_MIN_DIFF - 1e-9
        and np.percentile(boot, 2.5) > 0
    )
    return out


def auc_score(score: np.ndarray, y: np.ndarray) -> float | None:
    """Area under the ROC curve of a score for the positive class (rank sum; ties are not averaged)."""
    pos = int(y.sum())
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        return None
    ranks = np.empty(len(score))
    ranks[np.argsort(score, kind="stable")] = np.arange(1, len(score) + 1)
    return float((ranks[y].sum() - pos * (pos + 1) / 2) / (pos * neg))


def signal_check(ba: dict, names: list[str]) -> dict:
    ok = [ba[n] for n in names if ba.get(n) is not None]
    share = float(np.mean([b >= exp.SIGNAL_MIN_BA for b in ok])) if ok else None
    return {
        "evaluable": len(ok),
        "share_at_least_0_60": share,
        "passes": bool(share is not None and share >= exp.SIGNAL_MIN_SHARE),
    }


def outcome_text(signal: dict, main: dict, new_places: bool = False) -> str:
    if not signal["passes"]:
        return "signal check failed: the head pose alone is reported as insufficient on this data"
    if main["passes"]:
        return "rule 2 met" + (" on the held-out locations" if new_places else "")
    return "rule 2 not met: the network did not beat the hysteresis speed threshold on this data"


def decide(per_seed: list[dict], names: list[str], reference: dict, other: dict | None = None,
           new_places: bool = False, baseline_false_lock: dict | None = None) -> dict:
    """The recorded rule on the causal rows, for the given test sequences. ``reference`` is B2 as recorded
    (name -> missed-lock at fl5); ``other`` optionally another baseline for the beyond-data comparison."""
    net_missed = mean_over_seeds(per_seed, names, "causal", "fl5", "missed_lock")
    net_ba = mean_over_seeds(per_seed, names, "causal", "ba", "balanced_accuracy")
    signal = signal_check(net_ba, names)
    main = compare(reference, net_missed, names)
    false_lock = {
        "network": exp.summarise_values(list(mean_over_seeds(per_seed, names, "causal", "fl5", "false_lock").values())).get("median"),
        **(baseline_false_lock or {}),
    }
    out = {
        "sequences_used": len(names),
        "signal_check": signal,
        "main_comparison_vs_B2": main,
        "median_held_out_false_lock_at_fl5": false_lock,
        "outcome": outcome_text(signal, main, new_places),
    }
    if other:
        out["comparison_vs_other"] = {k: compare(v, net_missed, names) for k, v in other.items()}
    return out


# ---------------------------------------------------------------- baselines fitted on the training pool (B2-137, M1-137)


def baselines_137(recs: dict, train: list[str], val: list[str], test_names: list[str], test_recs: dict) -> dict:
    """B2 and M1 of the first protocol, fitted on the training sequences (M1) and with the operating point
    chosen on the validation sequences (both); applied to the test sequences. Returns row -> name -> det -> op -> metrics."""
    out = {}
    for row in ROWS:
        causal = row == "causal"

        def feat(rec, shift=0.0):
            x = exp.features(rec["head_pos"], rec["head_omega"], causal)
            d, valid = frame_data(rec, shift)
            return x[valid], d

        tr = [feat(recs[n]) for n in train]
        va = [feat(recs[n]) for n in val]
        xtr, ytr = np.vstack([a for a, _ in tr]), np.concatenate([d["y"] for _, d in tr])
        xva, yva = np.vstack([a for a, _ in va]), np.concatenate([d["y"] for _, d in va])
        b1 = exp.thresholds(xtr[:, 0], ytr)
        b2 = exp.search_b2([a[:, 0] for a, _ in va], [d["y"] for _, d in va], list(b1.values()))
        model = exp.fit_logistic(xtr, ytr.astype(float))
        neg = exp.thresholds(-exp.predict(model, xva), yva)
        m1 = {k: -v for k, v in neg.items()}
        out[row] = {"operating_points": {"B2": b2, "M1": m1}, "per_sequence": {}}
        for n in test_names:
            x, d = feat(test_recs[n])
            p = exp.predict(model, x)
            out[row]["per_sequence"][n] = {
                "B2-137": {op: exp.metrics(exp.hysteresis(x[:, 0], *b2[op]), d) for op in OPS},
                "M1-137": {op: exp.metrics(p > m1[op], d) for op in OPS},
            }
    return out


def first_protocol_baselines(recs: dict, names: list[str]) -> dict:
    """B1, B2, M1 of the first protocol on the test sequences, with its own 5-fold split over these sequences."""
    out = {}
    for row in ROWS:
        res = exp.evaluate(recs, names, causal=(row == "causal"), dets=("B1", "B2", "M1"))
        out[row] = res["per_sequence"]
    return out


def check_against_recorded(base: dict, names: list[str]) -> dict:
    """The rerun of the first protocol must give the recorded B2 / M1 numbers (same code, same data)."""
    if not FIRST_PROTOCOL_RESULT.exists():
        return {"checked": False, "note": f"{FIRST_PROTOCOL_RESULT} not found"}
    rec = read_json(FIRST_PROTOCOL_RESULT)["main"]
    worst = 0.0
    for row in ROWS:
        for n in names:
            for det in ("B2", "M1"):
                a = base[row][n][det]["fl5"]["missed_lock"]
                b = rec[row]["per_sequence"][n][det]["fl5"]["missed_lock"]
                if a is None and b is None:
                    continue
                worst = max(worst, abs((a if a is not None else 9) - (b if b is not None else 9)))
    return {"checked": True, "max_abs_difference_missed_lock_fl5": worst, "identical": bool(worst < 1e-9)}


# ---------------------------------------------------------------- the test step


def table_rows(per_seed: list[dict], names: list[str], row: str) -> dict:
    out = {}
    for op in OPS:
        out[op] = {
            key: exp.summarise_values(list(mean_over_seeds(per_seed, names, row, op, key).values()))
            for key in ("balanced_accuracy", "f1", "missed_lock", "false_lock", "slip_speed_mps",
                        "hard_missed_lock", "upright_missed_lock", "upright_false_lock", "upright_balanced_accuracy")
        }
    return out


def pooled_ece(outs: dict, recs: dict, names: list[str], row: str) -> float:
    ps, ys = [], []
    for n in names:
        d, valid = frame_data(recs[n], 0.0)
        ps.append(1.0 / (1.0 + np.exp(-row_logits(outs[n], row).astype(np.float64)[valid])))
        ys.append(d["y"])
    return exp.ece(np.concatenate(ps), np.concatenate(ys))


def block_intervals(outs: dict, recs: dict, names: list[str], points: dict) -> dict:
    rng = np.random.default_rng(0)
    res = {}
    for n in names:
        d, valid = frame_data(recs[n], 0.0)
        s = -row_logits(outs[n], "causal").astype(np.float64)[valid]
        res[n] = exp.block_ci(d["y"], s < points["causal"]["fl5"], rng)
    return res


def model_files(variant: str, size: str, models: Path) -> list[Path]:
    return [models / f"{variant}_{size}_seed{s}.pt" for s in SEEDS]


def run_test(scope: str, device, root: Path, models: Path = MODELS) -> dict:
    """Evaluate the chosen size ONCE: scope main on the 60 test sequences, scope newplace on the test sequences
    at the held-out locations. Loads the test caches. Refuses to run twice."""
    target = RESULT if scope == "main" else RESULT_NEW
    if target.exists():
        raise RuntimeError(f"{target} exists: the protocol evaluates once. Rename it yourself if a rerun is allowed.")
    plan, choice = read_json(PLAN), read_json(CHOICE)
    size = choice["chosen_size"]
    variant = "main" if scope == "main" else "newplace"
    sets = plan["main"] if scope == "main" else plan["newplace"]
    train, val = sets["train"], sets["validation"]
    all_test = plan["test"]
    names = all_test if scope == "main" else plan["newplace"]["test_sequences"]
    if set(train + val) & set(all_test):
        raise RuntimeError("a test sequence is in the training or validation list")
    print(f"scope {scope}: size {size}, {len(train)} training, {len(val)} validation, {len(names)} test sequences", flush=True)
    recs_tv = load_caches(train + val, POOL_CHECK, root)
    recs_test = load_caches(all_test, TEST_CHECK, root)  # all 60: the first protocol's folds need all of them
    nets = [load_net(f, device)[0] for f in model_files(variant, size, models)]

    per_seed, points, ece_by_seed, ci_by_seed, outs_all = [], [], [], [], []
    for s, net in enumerate(nets):
        outs_val = all_logits(net, recs_tv, val, device)
        pts = validation_points(outs_val, recs_tv, val)
        outs = all_logits(net, recs_test, all_test, device)
        per_seed.append(test_metrics(outs, recs_test, all_test, pts))
        points.append(pts)
        ece_by_seed.append(pooled_ece(outs, recs_test, names, "causal"))
        ci_by_seed.append(block_intervals(outs, recs_test, names, pts))
        outs_all.append(outs)
        print(f"  seed {s}: operating points {json.dumps(pts)}", flush=True)

    if scope == "main":
        base = first_protocol_baselines(recs_test, all_test)
        recorded = check_against_recorded(base, all_test)
        if recorded.get("checked") and not recorded.get("identical"):
            raise RuntimeError(
                "STOP: the rerun of the first protocol differs from the recorded numbers "
                f"(max difference {recorded['max_abs_difference_missed_lock_fl5']:.3g}); nothing was written. Send this to Claude."
            )
        extra = baselines_137(recs_tv, train, val, all_test, recs_test)
    else:
        main_result = read_json(RESULT)
        base = main_result["first_protocol"]
        recorded = main_result["first_protocol_check"]
        extra = baselines_137(recs_tv, train, val, all_test, recs_test)

    ref = {n: base["causal"][n]["B2"]["fl5"]["missed_lock"] for n in names}
    m1 = {n: base["causal"][n]["M1"]["fl5"]["missed_lock"] for n in names}
    m137 = {n: extra["causal"]["per_sequence"][n]["M1-137"]["fl5"]["missed_lock"] for n in names}
    b137 = {n: extra["causal"]["per_sequence"][n]["B2-137"]["fl5"]["missed_lock"] for n in names}
    def med_false(rows):
        return exp.summarise_values(rows).get("median")

    base_false = {
        "B2_first_protocol": med_false([base["causal"][n]["B2"]["fl5"]["false_lock"] for n in names]),
        "M1_first_protocol": med_false([base["causal"][n]["M1"]["fl5"]["false_lock"] for n in names]),
        "B2_137": med_false([extra["causal"]["per_sequence"][n]["B2-137"]["fl5"]["false_lock"] for n in names]),
        "M1_137": med_false([extra["causal"]["per_sequence"][n]["M1-137"]["fl5"]["false_lock"] for n in names]),
    }
    not_trained = [
        f"{variant}_{size}_seed{i}"
        for i, f in enumerate(model_files(variant, size, models))
        if not read_json(f.with_suffix(".json"))["trained"]
    ]
    if not_trained:
        print(f"WARNING: runs flagged 'did not train' (reported, not retried): {not_trained}", flush=True)
    report = {
        "scope": scope,
        "note": None if scope == "main" else (
            "indicative only: 15 test sequences; the training pool was reduced before the validation split (protocol 3A)"
        ),
        "runs_flagged_not_trained": not_trained,
        "chosen_size": size,
        "provenance": {
            "code_sha256_16": exp.sha16(Path(__file__)),
            "experiment_code_sha256_16": exp.sha16(Path(exp.__file__)),
            "plan_sha256_16": exp.sha16(PLAN),
            "torch": torch.__version__,
            "device": str(device),
            "models": [str(f) for f in model_files(variant, size, models)],
            "training_sequences": len(train),
            "validation_sequences": len(val),
            "test_sequences": len(names),
        },
        "operating_points_by_seed": points,
        "first_protocol_check": recorded,
        "rule": {
            "causal": decide(per_seed, names, ref, {"M1_first_protocol": m1, "M1-137": m137, "B2-137": b137},
                             new_places=(scope == "newplace"), baseline_false_lock=base_false),
        },
        "lookahead_reported_only": {
            "median_missed_lock_fl5_network": exp.summarise_values(list(mean_over_seeds(per_seed, names, "lookahead", "fl5", "missed_lock").values())),
            "median_missed_lock_fl5_B2": exp.summarise_values([base["lookahead"][n]["B2"]["fl5"]["missed_lock"] for n in names]),
        },
        "table": {row: table_rows(per_seed, names, row) for row in ROWS},
        "baseline_tables": {
            "B2_first_protocol": exp.summarise_values([base["causal"][n]["B2"]["fl5"]["missed_lock"] for n in names]),
            "M1_first_protocol": exp.summarise_values([base["causal"][n]["M1"]["fl5"]["missed_lock"] for n in names]),
            "M1_137": exp.summarise_values(list(m137.values())),
            "B2_137": exp.summarise_values(list(b137.values())),
            "hard_stratum_missed_lock_fl5": {
                "network": exp.summarise_values(list(mean_over_seeds(per_seed, names, "causal", "fl5", "hard_missed_lock").values())),
                "B2": exp.summarise_values([base["causal"][n]["B2"]["fl5"]["hard_missed_lock"] for n in names]),
                "M1": exp.summarise_values([base["causal"][n]["M1"]["fl5"]["hard_missed_lock"] for n in names]),
                "M1_137": exp.summarise_values([extra["causal"]["per_sequence"][n]["M1-137"]["fl5"]["hard_missed_lock"] for n in names]),
                "B2_137": exp.summarise_values([extra["causal"]["per_sequence"][n]["B2-137"]["fl5"]["hard_missed_lock"] for n in names]),
            },
        },
        "operating_points_baselines_137": {row: extra[row]["operating_points"] for row in ROWS},
        "pooled_ece_causal_by_seed": ece_by_seed,
        "missed_lock_block_interval_by_seed": ci_by_seed,
        "per_sequence_by_seed": per_seed,
        "per_sequence_baselines_137": {row: extra[row]["per_sequence"] for row in ROWS},
        "first_protocol": {row: {n: {d: v[d] for d in ("B1", "B2", "M1")} for n, v in base[row].items()} for row in ROWS},
    }
    exp.write_report(report, target)
    show_test(report)
    print(f"\nwritten to {target}")
    return report


def med(v) -> str:
    return "  -  " if v is None else f"{v:6.3f}"


def show_test(report: dict) -> None:
    r = report["rule"]["causal"]
    print(f"\n== {report['scope']}: chosen size {report['chosen_size']}, {r['sequences_used']} test sequences ==")
    for row in ROWS:
        print(f"\n{row} rows, network (mean of 3 seeds per sequence, median over sequences)")
        for op, t in report["table"][row].items():
            print(f"  {op:4s} BA {med(t['balanced_accuracy'].get('median'))}  missed {med(t['missed_lock'].get('median'))}  "
                  f"false {med(t['false_lock'].get('median'))}  hard missed {med(t['hard_missed_lock'].get('median'))}")
    bt = report["baseline_tables"]
    print("\nmissed-lock at fl5, median over sequences:")
    for k in ("B2_first_protocol", "M1_first_protocol", "B2_137", "M1_137"):
        print(f"  {k:18s} {med(bt[k].get('median'))}")
    print("hard stratum:", json.dumps({k: v.get("median") for k, v in bt["hard_stratum_missed_lock_fl5"].items()}))
    print("signal check:", json.dumps(r["signal_check"]))
    print("network vs B2 (rule 2):", json.dumps(r["main_comparison_vs_B2"]))
    for k, v in r.get("comparison_vs_other", {}).items():
        print(f"network vs {k}:", json.dumps(v))
    print("median held-out false-lock at fl5 (network and baselines):", json.dumps(r["median_held_out_false_lock_at_fl5"]))
    if report.get("note"):
        print("note:", report["note"])
    if report.get("runs_flagged_not_trained"):
        print("WARNING, runs that did not train:", report["runs_flagged_not_trained"])
    print("outcome:", r["outcome"])
    print("rerun of the first protocol equals the recorded numbers:", json.dumps(report["first_protocol_check"]))


# ---------------------------------------------------------------- controls (scope main)


def run_controls(device, root: Path, models: Path = MODELS) -> dict:
    plan, choice = read_json(PLAN), read_json(CHOICE)
    size = choice["chosen_size"]
    train, val = plan["main"]["train"], plan["main"]["validation"]
    test = plan["test"]
    if set(train + val) & set(test):
        raise RuntimeError("a test sequence is in the training or validation list")
    recs_tv = load_caches(train + val, POOL_CHECK, root)
    recs_test = load_caches(test, TEST_CHECK, root)
    main_nets = [load_net(f, device)[0] for f in model_files("main", size, models)]
    base = read_json(RESULT)["first_protocol"]
    side = set(read_json(TEST_CHECK)["summary"].get("side_check_inconsistent", []))
    b2 = {n: base["causal"][n]["B2"]["fl5"]["missed_lock"] for n in test}
    out: dict = {"chosen_size": size}

    outs_val = [all_logits(net, recs_tv, val, device) for net in main_nets]
    points = [validation_points(o, recs_tv, val) for o in outs_val]
    outs_main = [all_logits(net, recs_test, test, device) for net in main_nets]
    per_seed = [test_metrics(o, recs_test, test, p) for o, p in zip(outs_main, points, strict=True)]

    # C1 label shuffle, C2 learning curve (one seed each, trained earlier with --variant shuffle / frac25 / frac50)
    def one(variant: str, roll: int = 0):
        f = models / f"{variant}_{size}_seed0.pt"
        if not f.exists():
            return {"note": f"{f.name} not trained yet"}
        net, ck = load_net(f, device)
        pts = validation_points(all_logits(net, recs_tv, val, device), recs_tv, val, roll)
        outs_c = all_logits(net, recs_test, test, device)
        res = test_metrics(outs_c, recs_test, test, pts, rows=("causal",))
        log = read_json(f.with_suffix(".json"))
        scores, truth = [], []
        for n in test:
            d, valid = frame_data(recs_test[n], 0.0)
            scores.append(row_logits(outs_c[n], "causal").astype(np.float64)[valid])
            truth.append(d["y"])
        score, y = np.concatenate(scores), np.concatenate(truth)
        return {
            "trained": log["trained"],
            "best_validation_loss": log["best_validation_loss"],
            "base_rate_validation_loss": log["prior_validation_loss"],
            "logit_std_on_test": float(score.std()),
            "pooled_auc_on_true_test_labels": auc_score(score, y),
            "train_sequences": len(ck["train"]),
            "operating_points": pts["causal"],
            "median_balanced_accuracy_ba": exp.summarise_values([res[n]["causal"]["ba"]["balanced_accuracy"] for n in test]).get("median"),
            "median_missed_lock_fl5": exp.summarise_values([res[n]["causal"]["fl5"]["missed_lock"] for n in test]).get("median"),
            "median_false_lock_fl5": exp.summarise_values([res[n]["causal"]["fl5"]["false_lock"] for n in test]).get("median"),
        }

    c1 = {name: one(name, roll) for name, roll in SHUFFLES.items()}
    gains = [
        v["base_rate_validation_loss"] - v["best_validation_loss"] for v in c1.values() if "best_validation_loss" in v
    ]
    c1["reading"] = {
        "rule": f"leakage is suspected only if, for all three shifts, the run trained on shifted labels to a validation loss at least "
        f"{C1_MIN_GAIN} below the base-rate loss (shifted labels carry no information about the head motion; ranking measures such as "
        "balanced accuracy or the area under the curve are not used for the verdict because a model that learned nothing can still rank "
        "frames by head speed with a tiny weight of random sign)",
        "validation_loss_gain_by_shift": gains,
        "leakage_suspected": bool(len(gains) == len(SHUFFLES) and all(g >= C1_MIN_GAIN for g in gains)),
    }
    out["C1_label_shuffle"] = c1
    out["C2_learning_curve"] = {"25%": one("frac25"), "50%": one("frac50"), "100%": one("main")}

    # C3 noise sweep (test input only)
    out["C3_noise"] = {}
    for sp, sy in exp.NOISE_SETTINGS:
        key = f"position_{sp * 100:g}cm_yaw_{sy:g}deg"
        res = [test_metrics(all_logits(net, recs_test, test, device, (sp, sy)), recs_test, test, p, rows=("causal",))
               for net, p in zip(main_nets, points, strict=True)]
        base_noise = exp.evaluate(recs_test, test, causal=True, noise=(sp, sy), dets=("B1", "B2", "M1"))["per_sequence"]
        out["C3_noise"][key] = {
            "network": {op: {k: exp.summarise_values(list(mean_over_seeds(res, test, "causal", op, k).values())).get("median")
                             for k in ("balanced_accuracy", "missed_lock", "false_lock")} for op in OPS},
            **{det: {op: {k: exp.summarise_values([base_noise[n][det][op][k] for n in test]).get("median")
                          for k in ("balanced_accuracy", "missed_lock", "false_lock")} for op in OPS} for det in ("B2", "M1")},
        }
        print(f"C3 {key} done", flush=True)

    # C4 clock shift of the test labels
    out["C4_clock_shift"] = {}
    for row in ROWS:
        out["C4_clock_shift"][row] = {}
        for shift in exp.CLOCK_SHIFTS_S:
            res = [test_metrics(o, recs_test, test, p, shift=shift, rows=(row,)) for o, p in zip(outs_main, points, strict=True)]
            out["C4_clock_shift"][row][f"{round(shift * 1000):+d}ms"] = {
                op: {k: exp.summarise_values(list(mean_over_seeds(res, test, row, op, k).values())).get("median")
                     for k in ("balanced_accuracy", "missed_lock", "false_lock")} for op in OPS}

    # C5 without the left/right-inconsistent sequences
    use = [n for n in test if n not in side]
    out["C5_without_side_check_inconsistent"] = decide(per_seed, use, {n: b2[n] for n in use})

    # C6 location split (descriptive)
    pool_locs = {plan["location"][n] for n in train + val}
    known = [n for n in test if plan["location"][n] in pool_locs]
    unknown = [n for n in test if plan["location"][n] not in pool_locs]
    out["C6_location_split"] = {"counts": {"location_in_training_pool": len(known), "location_not_in_training_pool": len(unknown)}}
    for label, part in (("location_in_training_pool", known), ("location_not_in_training_pool", unknown)):
        if len(part) < 5:
            out["C6_location_split"][label] = {"note": "fewer than 5 sequences: not tested"}
        else:
            out["C6_location_split"][label] = decide(per_seed, part, {n: b2[n] for n in part})
    exp.write_report(out, CONTROLS)
    print(json.dumps(exp.clean(out), indent=1)[:6000])
    print(f"\nwritten to {CONTROLS}")
    return out


# ---------------------------------------------------------------- the noise test (docs/nymeria_foot_lock_noise_protocol.md)


def level_key(sp: float, sy: float) -> str:
    return f"position_{sp * 100:g}cm_yaw_{sy:g}deg"


def level_summary(per_seed: list[dict], names: list[str]) -> dict:
    keys = ("balanced_accuracy", "missed_lock", "false_lock", "hard_missed_lock")
    return {
        op: {k: exp.summarise_values(list(mean_over_seeds(per_seed, names, "causal", op, k).values())).get("median") for k in keys}
        for op in OPS
    }


def baseline_summary(base: dict, names: list[str], det: str) -> dict:
    keys = ("balanced_accuracy", "missed_lock", "false_lock", "hard_missed_lock")
    return {op: {k: exp.summarise_values([base[n][det][op][k] for n in names]).get("median") for k in keys} for op in OPS}


def noise_level_block(res_noise: list[dict], res_clean: list[dict], base: dict, names: list[str]) -> dict:
    net = mean_over_seeds(res_noise, names, "causal", "fl5", "missed_lock")
    clean = mean_over_seeds(res_clean, names, "causal", "fl5", "missed_lock")
    b2 = {n: base[n]["B2"]["fl5"]["missed_lock"] for n in names}
    m1 = {n: base[n]["M1"]["fl5"]["missed_lock"] for n in names}
    return {
        "noise_trained": level_summary(res_noise, names),
        "clean_trained": level_summary(res_clean, names),
        "B2": baseline_summary(base, names, "B2"),
        "M1": baseline_summary(base, names, "M1"),
        "noise_trained_vs_B2": compare(b2, net, names),
        "noise_trained_vs_M1": compare(m1, net, names),
        "clean_trained_vs_B2": compare(b2, clean, names),
        "noise_trained_vs_clean_trained": compare(clean, net, names),
        "per_sequence_missed_lock_fl5": {"noise_trained": net, "clean_trained": clean, "B2": b2, "M1": m1},
    }


def noise_rule(levels: dict, clean_recorded_missed: float) -> dict:
    """Protocol noise section 4, items 1 and 2."""
    item1 = {level_key(*lv): levels[level_key(*lv)]["noise_trained_vs_B2"] for lv in NOISE_RULE_LEVELS}
    zero = levels[level_key(0.0, 0.0)]["noise_trained"]["fl5"]
    zero_clean = levels[level_key(0.0, 0.0)]["clean_trained"]["fl5"]
    limit = clean_recorded_missed + NOISE_CLEAN_TOLERANCE
    item2 = {
        "noise_trained_median_missed_lock_fl5": zero["missed_lock"],
        "clean_trained_recorded_median_missed_lock_fl5": clean_recorded_missed,
        "limit": limit,
        "noise_trained_median_false_lock_fl5": zero["false_lock"],
        "clean_trained_median_false_lock_fl5": zero_clean["false_lock"],
        "passes": bool(zero["missed_lock"] is not None and zero["missed_lock"] <= limit + 1e-12),
    }
    passes = [bool(v.get("passes")) for v in item1.values()]
    if all(passes) and item2["passes"]:
        outcome = "rule met: the lead over B2 holds at 1 cm and 2 cm and the clean result is kept"
    elif all(passes):
        outcome = "item 1 met at 1 cm and 2 cm, item 2 not met: robustness cost accuracy on clean poses"
    elif passes[0]:
        outcome = "item 1 met only at 1 cm" + ("" if item2["passes"] else "; item 2 not met")
    else:
        outcome = "item 1 not met: training with noise did not make the network robust at 1 cm"
    return {"item1_by_level": item1, "item2_clean_level": item2, "outcome": outcome}


def run_test_noise(device, root: Path, models: Path = MODELS) -> dict:
    """Noise test of the noise-trained network (size S, three seeds) on the 60 test sequences, once; second use of the test
    sequences (protocol noise, section 3). Loads the test caches. Refuses to run twice."""
    if NOISE_RESULT.exists():
        raise RuntimeError(f"{NOISE_RESULT} exists: the protocol evaluates once. Rename it yourself if a rerun is allowed.")
    if not RESULT.exists():
        raise RuntimeError(f"{RESULT} not found: the main test result of the clean-trained network comes first")
    plan, choice = read_json(PLAN), read_json(CHOICE)
    if choice["chosen_size"] != "S":
        raise RuntimeError("the noise protocol is written for size S; the size choice says otherwise")
    train, val, test = plan["main"]["train"], plan["main"]["validation"], plan["test"]
    if set(train + val) & set(test):
        raise RuntimeError("a test sequence is in the training or validation list")
    noise_files, clean_files = model_files("noise", "S", models), model_files("main", "S", models)
    for f in noise_files + clean_files:
        if not f.exists():
            raise RuntimeError(f"{f.name} is not trained yet")
    logs = [read_json(f.with_suffix(".json")) for f in noise_files]
    not_trained = [i for i, lg in enumerate(logs) if not lg["trained"]]
    if not_trained:
        print(f"WARNING: runs flagged 'did not train' (reported, not retried): seeds {not_trained}", flush=True)
    recs_tv = load_caches(train + val, POOL_CHECK, root)
    recs_test = load_caches(test, TEST_CHECK, root)
    noise_nets = [load_net(f, device)[0] for f in noise_files]
    clean_nets = [load_net(f, device)[0] for f in clean_files]

    val_noisy = mixture_recs(recs_tv, val, np.random.default_rng(NOISE_VAL_SEED))
    pts_noise = [validation_points(all_logits(n_, val_noisy, val, device), val_noisy, val) for n_ in noise_nets]
    pts_clean = [validation_points(all_logits(n_, recs_tv, val, device), recs_tv, val) for n_ in clean_nets]
    for s, p in enumerate(pts_noise):
        print(f"  noise-trained seed {s}: operating points {json.dumps(p)}", flush=True)

    def run_level(noise, corr: bool, base):
        res_n = [test_metrics(all_logits(n_, recs_test, test, device, noise, corr), recs_test, test, p, rows=("causal",))
                 for n_, p in zip(noise_nets, pts_noise, strict=True)]
        res_c = [test_metrics(all_logits(n_, recs_test, test, device, noise, corr), recs_test, test, p, rows=("causal",))
                 for n_, p in zip(clean_nets, pts_clean, strict=True)]
        return noise_level_block(res_n, res_c, base, test)

    levels = {}
    for sp, sy in exp.NOISE_SETTINGS:
        base = exp.evaluate(recs_test, test, causal=True, noise=(sp, sy), dets=("B1", "B2", "M1"))["per_sequence"]
        levels[level_key(sp, sy)] = run_level((sp, sy), False, base)
        print(f"noise level {level_key(sp, sy)} done", flush=True)

    correlated = {}
    original = exp.add_noise
    exp.add_noise = corr_add_noise  # B2 and M1 of the first protocol, rerun under the same correlated noise
    try:
        for sp, sy in CORR_LEVELS:
            base = exp.evaluate(recs_test, test, causal=True, noise=(sp, sy), dets=("B1", "B2", "M1"))["per_sequence"]
            correlated[level_key(sp, sy)] = run_level((sp, sy), True, base)
            print(f"correlated noise level {level_key(sp, sy)} done", flush=True)
    finally:
        exp.add_noise = original

    clean_recorded = read_json(RESULT)["table"]["causal"]["fl5"]["missed_lock"]["median"]
    report = {
        "note": "second use of the 60 test sequences (the first is nymeria_learned_test_result.json); nothing was chosen on them",
        "runs_flagged_not_trained": not_trained,
        "provenance": {
            "code_sha256_16": exp.sha16(Path(__file__)),
            "experiment_code_sha256_16": exp.sha16(Path(exp.__file__)),
            "plan_sha256_16": exp.sha16(PLAN),
            "torch": torch.__version__,
            "device": str(device),
            "noise_models": [str(f) for f in noise_files],
            "clean_models": [str(f) for f in clean_files],
            "noise_training": NOISE_SPEC,
            "test_sequences": len(test),
        },
        "training_logs": [
            {k: lg.get(k) for k in ("seed", "best_epoch", "best_validation_loss", "best_epoch_clean_validation_loss",
                                    "prior_validation_loss", "trained", "possibly_not_converged", "epochs_run")}
            for lg in logs
        ],
        "operating_points_noise_trained_by_seed": pts_noise,
        "operating_points_clean_trained_by_seed": pts_clean,
        "rule": noise_rule(levels, clean_recorded),
        "levels_independent_noise": levels,
        "levels_correlated_noise_descriptive": correlated,
    }
    exp.write_report(report, NOISE_RESULT)
    show_noise(report)
    print(f"\nwritten to {NOISE_RESULT}")
    return report


def show_noise(report: dict) -> None:
    print(f"\n== noise-trained network against the clean-trained one and the baselines (causal, fl5, median over "
          f"{report['provenance']['test_sequences']} test sequences) ==")
    for title, key in (("independent noise", "levels_independent_noise"), ("correlated noise (descriptive)", "levels_correlated_noise_descriptive")):
        print(f"\n{title}: missed-lock (false-lock) of noise-trained | clean-trained | B2 | M1")
        for lv, b in report[key].items():
            cells = [b["noise_trained"]["fl5"], b["clean_trained"]["fl5"], b["B2"]["fl5"], b["M1"]["fl5"]]
            print(f"  {lv:28s} " + " | ".join(f"{c['missed_lock']:.3f} ({c['false_lock']:.3f})" for c in cells))
    r = report["rule"]
    print("\nrule item 1 (noise-trained against B2, same noise):")
    for lv, c in r["item1_by_level"].items():
        print(f"  {lv}: lower on {c.get('share_network_lower', 0):.3f} of sequences, median difference "
              f"{c.get('median_difference_pp', float('nan')):.1f} points, interval {json.dumps(c.get('bootstrap95_pp'))}, passes {c.get('passes')}")
    print("rule item 2 (clean level):", json.dumps(r["item2_clean_level"]))
    print("outcome:", r["outcome"])
    for lv, b in report["levels_independent_noise"].items():
        c = b["noise_trained_vs_clean_trained"]
        print(f"  noise-trained against clean-trained at {lv}: lower on {c.get('share_network_lower', 0):.3f}, "
              f"median difference {c.get('median_difference_pp', float('nan')):.1f} points")
    if report["runs_flagged_not_trained"]:
        print("WARNING, runs that did not train:", report["runs_flagged_not_trained"])
    for lg in report["training_logs"]:
        note = "  (possibly not converged)" if lg.get("possibly_not_converged") else ""
        print(f"  seed {lg['seed']}: best epoch {lg['best_epoch']}, validation loss {lg['best_validation_loss']:.4f}, "
              f"clean validation loss {lg['best_epoch_clean_validation_loss']:.4f}{note}")


# ---------------------------------------------------------------- choose and train steps


def choose_size(models: Path = MODELS) -> dict:
    res = {}
    for size in SIZES:
        logs = [read_json(f.with_suffix(".json")) for f in model_files("main", size, models)]
        res[size] = {
            "validation_loss_by_seed": [l["best_validation_loss"] for l in logs],
            "mean_validation_loss": float(np.mean([l["best_validation_loss"] for l in logs])),
            "best_epoch_by_seed": [l["best_epoch"] for l in logs],
            "trained_by_seed": [l["trained"] for l in logs],
            "parameters": logs[0]["parameters"],
        }
    s_loss, m_loss = res["S"]["mean_validation_loss"], res["M"]["mean_validation_loss"]
    chosen = "S" if (s_loss - m_loss) < SIZE_TIE else "M"  # M only if it is lower by at least 0.002
    out = {"chosen_size": chosen, "sizes": res, "rule": "lower mean validation loss; a difference under 0.002 keeps S"}
    CHOICE.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    flagged = [(sz, i) for sz in SIZES for i, ok in enumerate(res[sz]["trained_by_seed"]) if not ok]
    if flagged:
        print(f"WARNING: runs flagged 'did not train' (reported, not retried): {flagged}")
    return out


def variant_sets(plan: dict, variant: str) -> tuple[list[str], list[str], int]:
    if variant == "main":
        return plan["main"]["train"], plan["main"]["validation"], 0
    if variant == "newplace":
        return plan["newplace"]["train"], plan["newplace"]["validation"], 0
    if variant == "noise":
        return plan["main"]["train"], plan["main"]["validation"], 0
    if variant in SHUFFLES:
        return plan["main"]["train"], plan["main"]["validation"], SHUFFLES[variant]
    frac = {"frac25": "0.25", "frac50": "0.5"}[variant]
    return plan["nested_train_subsets"][frac], plan["main"]["validation"], 0


def run_train(variant: str, sizes: list[str], seeds: list[int], device, root: Path, models: Path, limit: int | None) -> None:
    plan = read_json(PLAN)
    train, val, roll = variant_sets(plan, variant)
    if set(train + val) & set(plan["test"]):
        raise RuntimeError("STOP: a test sequence is in the training or validation list")
    if limit:
        train, val = train[:limit], val[: max(1, limit // 4)]
    recs = load_caches(train + val, POOL_CHECK, root)
    for size in sizes:
        for seed in seeds:
            out = models / f"{variant}_{size}_seed{seed}.pt"
            if out.exists():
                print(f"{out.name} exists, skipped")
                continue
            print(f"training {variant} {size} seed {seed}: {len(train)} training, {len(val)} validation sequences", flush=True)
            label = {"variant": variant, "label_roll_frames": roll}
            if variant == "noise":
                label["noise_training"] = NOISE_SPEC
            train_one(train, val, recs, size, seed, roll, device, out, label, noise_train=(variant == "noise"))


def pick_device(name: str):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return torch.device(name)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--step", required=True, choices=("plan", "prepare", "train", "choose", "test", "controls"))
    ap.add_argument("--root", type=Path, default=s0.sync.DEFAULT_ROOT)
    ap.add_argument("--which", choices=("pool", "test", "both"), default="both", help="prepare: which sequences")
    ap.add_argument("--variant", choices=VARIANTS, default="main")
    ap.add_argument("--size", choices=("S", "M", "both"), default="both")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    ap.add_argument("--scope", choices=("main", "newplace", "noise"), default="main")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=None, help="prepare/train: only the first N sequences (testing)")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny settings and a separate model folder, to check the pipeline")
    a = ap.parse_args(argv)
    if not a.seeds:
        ap.error("--seeds needs at least one seed")
    if (a.smoke or a.limit) and a.step in ("choose", "test", "controls"):
        ap.error("--smoke and --limit are for plan/prepare/train checks only; they would overwrite real result files")
    if a.limit and a.step == "train" and not a.smoke:
        ap.error("--limit on train needs --smoke (otherwise models trained on part of the data get real names)")

    if a.step == "plan":
        plan = make_plan(a.root)
        PLAN.write_text(json.dumps(plan, indent=1), encoding="utf-8")
        show_plan(plan)
        print(f"written to {PLAN}")
        return 0
    if a.step == "prepare":
        plan = read_json(PLAN)
        if a.which in ("pool", "both"):
            load_caches(plan["pool"], POOL_CHECK, a.root, build=True, rebuild=a.rebuild, limit=a.limit)
        if a.which in ("test", "both"):
            load_caches(plan["test"], TEST_CHECK, a.root, build=True, rebuild=a.rebuild, limit=a.limit)
        print("caches ready")
        return 0
    if torch is None:
        print("torch is not installed in this environment")
        return 1
    device = pick_device(a.device)
    print(f"torch {torch.__version__}, device {device}", flush=True)
    models = MODELS
    if a.smoke:
        global EPOCH_STEPS, MAX_EPOCHS, VAL_STRIDE
        EPOCH_STEPS, MAX_EPOCHS, VAL_STRIDE = 20, 2, 200
        models = MODELS / "smoke"
    if a.step == "train":
        if a.variant == "main":
            sizes = list(SIZES) if a.size == "both" else [a.size]
        elif a.variant == "noise":
            sizes = ["S"]  # the noise protocol is written for size S
        else:
            sizes = [read_json(CHOICE)["chosen_size"]]
        seeds = a.seeds if a.variant in ("main", "newplace", "noise") else [0]
        run_train(a.variant, sizes, seeds, device, a.root, models, a.limit)
        return 0
    if a.step == "choose":
        choose_size(models)
        return 0
    if a.step == "test":
        if a.scope == "noise":
            run_test_noise(device, a.root, models)
        else:
            run_test(a.scope, device, a.root, models)
        return 0
    run_controls(device, a.root, models)
    return 0


if __name__ == "__main__":
    sys.exit(main())
