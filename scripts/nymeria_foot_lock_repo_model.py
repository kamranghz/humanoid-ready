"""Second foot-lock protocol (Nymeria): the repo's trained completion model on the head pose only.

Implements ``docs/nymeria_foot_lock_repo_model_protocol.md`` and its amendments A1 and A2 (definitions that
the protocol left open, recorded before any model output existed). The cohort, labels, folds, frames and
the success rule are those of the first protocol (``nymeria_foot_lock_experiment.py``), which is imported
and not changed in behaviour.

What the model gets: only the rig of a 64-frame window (head position with the floor at z = 0, camera
rotation world to camera, gravity); every body joint is marked not seen. The camera is the glasses' RGB
camera: its rotation in the device frame is read from the recording's own calibration (A2). The score is
the minimum of the four contact probabilities (left and right heel and toe). The causal row is the output
at the last frame of the window; the lookahead row is the output 15 frames before the last frame.

Steps (``--step``): ``prepare`` (camera rotation, floor height and the camera-frame check, per sequence),
``score`` (the model on every window, per condition), ``run`` (evaluation, rule, controls), ``all``.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

import nymeria_foot_lock_data_check as s0
import nymeria_foot_lock_experiment as e1
import nymeria_project_check as npc

REVIEW = Path(r"D:\projects\hready_data\review")
CKPT = Path(r"D:\projects\hready_data\checkpoints\deterministic_main\best.pt")
CKPT_SHA256 = "13b1ded8e3a0e93500ebe0e082198ae3a8ffe253b1f31e63c74fec5c8896840a"
OUT = REVIEW / "nymeria_foot_lock_repo_model.json"
CACHE = REVIEW / "nymeria_foot_lock_repo_model_cache"
SEG = npc.SEGMENTS
HZ = e1.HZ
WINDOW = 64  # the model's max_len
LOOKAHEAD = HZ // 2  # 15 frames
INPUT_VERSION = "R1-A2"
GRAVITY = (0.0, 0.0, -9.81)
NUM_KP = 22
FLOOR_SHIFTS_M = (
    -0.10,
    -0.05,
    -0.03,
    0.03,
    0.05,
    0.10,
)  # floor assumed this much higher
CONSTANT_HEIGHT_M = 1.6  # control C4: head height above the floor, constant
# Camera-frame check (A2). Pooled over the sequences; per-sequence values are reported, not gated.
AXIS_MIN_DOWN = (
    0.7  # median over sequences of the mean gravity along the camera's down axis
)
AXIS_MIN_FORWARD = 0.6  # median over testable sequences of the mean walking direction along the look direction
AXIS_LOW_FORWARD = 0.3  # a sequence below this is listed in the report
AXIS_MAX_SHARE_LOW = 0.10  # more than this share of testable sequences below AXIS_LOW_FORWARD stops the run
AXIS_MAX_STEREO_DIFF_DEG = (
    10.0  # camera yaw from the RGB extrinsics vs from the SLAM-camera baseline
)
AXIS_MIN_PROJECTION = (
    0.3  # the look direction must be this far from vertical to give a heading
)
AXIS_WALK_SPEED = 0.5  # m/s
AXIS_MIN_WALK_FRAMES = (
    300  # 10 s of walking at 30 Hz, else the sequence is "not testable"
)
CONDITIONS = (
    "main",
    *[f"floor{int(round(s * 100)):+d}cm" for s in FLOOR_SHIFTS_M],
    "xsens_height",
    "constant_height",
)
CLOCK_SHIFTS_S = (-0.1, -0.05, 0.05, 0.1)


# ---------------------------------------------------------------- inputs per sequence


class InputError(RuntimeError):
    """A sequence cannot give its inputs. It is left out and listed in the report; the run goes on."""


class FloorError(InputError):
    """A sequence has too few usable frames for the floor estimate."""


class CalibrationError(InputError):
    """The recording's own camera calibration is missing or not usable."""


def quaternion_to_matrix(q) -> np.ndarray:
    """Rotation matrix from an Aria unit quaternion written as [w, [x, y, z]]."""
    w, (x, y, z) = float(q[0]), [float(v) for v in q[1]]
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def camera_extrinsics(folder: Path) -> dict:
    """The RGB camera's rotation in the device frame, from the first line of the online calibration of the
    head recording (``T_Device_Camera``: its columns are the camera axes in device coordinates). The device
    frame of the Aria glasses is the frame of the left SLAM camera, which points about 40 degrees away from
    the direction the wearer faces (A2). Also returned: the facing direction implied by the baseline between
    the two SLAM cameras, as an independent check of the RGB camera's yaw."""
    member = "recording_head/mps/slam/online_calibration.jsonl"
    try:
        with (
            zipfile.ZipFile(s0.sync.find(folder, "recording_head.zip")) as archive,
            archive.open(member) as handle,
        ):
            entry = json.loads(handle.readline())
        cams = {c["Label"]: c for c in entry["CameraCalibrations"]}
        names = ("camera-rgb", "camera-slam-left", "camera-slam-right")
        if not all(cams[n].get("Calibrated", False) for n in names):
            raise CalibrationError("a camera is marked not calibrated")
        rot = {
            n: quaternion_to_matrix(cams[n]["T_Device_Camera"]["UnitQuaternion"])
            for n in names
        }
        tr = {
            n: np.array(cams[n]["T_Device_Camera"]["Translation"], float) for n in names
        }
    except CalibrationError:
        raise
    except (
        OSError,
        KeyError,
        ValueError,
        TypeError,
        IndexError,
        zipfile.BadZipFile,
    ) as err:
        raise CalibrationError(f"no usable online calibration: {err!r}") from err
    r = rot["camera-rgb"]
    if not (np.allclose(r @ r.T, np.eye(3), atol=1e-6) and np.linalg.det(r) > 0):
        raise CalibrationError("the RGB camera rotation is not a rotation")
    down = np.array([1.0, 0.0, 0.0])  # device x is down (gravity check)

    def horizontal(v):  # remove the vertical part and normalise
        h = v - v.dot(down) * down
        return h / np.linalg.norm(h)

    base = (
        tr["camera-slam-right"] - tr["camera-slam-left"]
    )  # points to the wearer's right
    left = horizontal(-base)
    stereo_forward = np.cross(down, left)
    look = horizontal(r[:, 2])
    return {
        "r_dc": r,
        "rgb_yaw_right_of_device_z_deg": float(
            np.degrees(np.arctan2(-look[1], look[2]))
        ),
        "stereo_vs_rgb_yaw_diff_deg": float(
            np.degrees(np.arccos(np.clip(stereo_forward.dot(look), -1, 1)))
        ),
    }


def eye_rotation(r_wd: np.ndarray, r_dc: np.ndarray) -> np.ndarray:
    """Camera rotation world -> camera with rows [right, up, look] for the RGB camera. ``r_wd`` is the
    device rotation (world from device), ``r_dc`` the RGB camera in the device frame. The RGB camera is
    mounted turned by 90 degrees: its axes are x down, y left, z forward (checked per sequence)."""
    r_wc = r_wd @ r_dc
    right = -r_wc[:, :, 1]
    up = -r_wc[:, :, 0]
    look = r_wc[:, :, 2]
    return np.stack([right, up, look], axis=1)


def axis_check(
    r_wc: np.ndarray, pos: np.ndarray, upright: np.ndarray | None = None
) -> dict:
    """Numbers that show whether the camera axes (x down, y left, z forward) hold on one sequence:
    the mean gravity along the camera's down axis over the upright frames (all frames if there are fewer
    than 30 s of them), and the mean of the cosine between the walking direction and the horizontal part
    of the look direction (frames walking above AXIS_WALK_SPEED, look direction not near vertical).
    ``r_wc`` is the camera rotation, world from camera. Nothing is gated here; see ``axis_gate``."""
    g_cam = np.einsum("tji,j->ti", r_wc, np.array([0.0, 0.0, -1.0]))
    on_upright = upright is not None and int(upright.sum()) >= 30 * HZ
    use = upright if on_upright else np.ones(len(g_cam), bool)
    look_z = np.clip(r_wc[:, 2, 2], -1.0, 1.0)
    k = HZ  # one second
    vel = (pos[k:, :2] - pos[:-k, :2]) / (k / HZ)
    speed = np.linalg.norm(vel, axis=1)
    walk = speed > AXIS_WALK_SPEED
    out = {
        "gravity_along_down": float(g_cam[use, 0].mean()),
        "gravity_frames": int(use.sum()),
        "gravity_on_upright_frames": bool(on_upright),
        "mean_look_elevation_deg": float(np.degrees(np.arcsin(look_z[use])).mean()),
        "walking_frames": int(walk.sum()),
        "walking_along_look": None,
    }
    look_h = r_wc[:-k, :2, 2]
    length = np.linalg.norm(look_h, axis=1)
    sel = walk & (length >= AXIS_MIN_PROJECTION)
    if sel.sum() >= AXIS_MIN_WALK_FRAMES:
        heading = vel[sel] / speed[sel][:, None]
        cos = (look_h[sel] * heading).sum(axis=1) / length[sel]
        out["walking_along_look"] = float(cos.mean())
    return out


def axis_gate(axes: dict[str, dict]) -> dict:
    """The camera-frame check, pooled over the sequences (A2). It stops the run only when the frame is wrong
    for the set as a whole; single sequences where the wearer often looks away from the walking direction
    are listed, not removed."""
    down = [a["gravity_along_down"] for a in axes.values()]
    fwd = [
        a["walking_along_look"]
        for a in axes.values()
        if a["walking_along_look"] is not None
    ]
    diff = [
        a["stereo_vs_rgb_yaw_diff_deg"]
        for a in axes.values()
        if "stereo_vs_rgb_yaw_diff_deg" in a
    ]
    low = sorted(
        n
        for n, a in axes.items()
        if a["walking_along_look"] is not None
        and a["walking_along_look"] < AXIS_LOW_FORWARD
    )
    out: dict = {
        "sequences": len(axes),
        "tested_forward": len(fwd),
        "median_gravity_along_down": float(np.median(down)) if down else None,
        "median_walking_along_look": float(np.median(fwd)) if fwd else None,
        "low_forward": low,
        "share_low_forward": (len(low) / len(fwd)) if fwd else None,
        "max_stereo_vs_rgb_yaw_diff_deg": float(max(diff)) if diff else None,
        "reasons": [],
    }
    if axes:
        if out["median_gravity_along_down"] < AXIS_MIN_DOWN:
            out["reasons"].append(
                f"median gravity along the camera's down axis {out['median_gravity_along_down']:.2f} < {AXIS_MIN_DOWN}"
            )
        if not fwd:
            out["reasons"].append(
                "the walking direction could not be tested on any sequence"
            )
        else:
            if out["median_walking_along_look"] < AXIS_MIN_FORWARD:
                out["reasons"].append(
                    f"median walking direction along the look direction {out['median_walking_along_look']:.2f} < {AXIS_MIN_FORWARD}"
                )
            if out["share_low_forward"] > AXIS_MAX_SHARE_LOW:
                out["reasons"].append(
                    f"{out['share_low_forward']:.0%} of the testable sequences are below {AXIS_LOW_FORWARD} (limit {AXIS_MAX_SHARE_LOW:.0%})"
                )
        if diff and out["max_stereo_vs_rgb_yaw_diff_deg"] > AXIS_MAX_STEREO_DIFF_DEG:
            out["reasons"].append(
                f"camera yaw from the RGB extrinsics and from the SLAM-camera baseline differ by up to {out['max_stereo_vs_rgb_yaw_diff_deg']:.1f} degrees (limit {AXIS_MAX_STEREO_DIFF_DEG})"
            )
    out["passes"] = not out["reasons"]
    return out


def prepare_inputs_one(folder: Path, rec: dict) -> dict:
    xs, head, pos, _ = s0.load_sequence(folder)
    t_x = xs["t_us"]
    _, r_wd = npc.interp_pose(head, rec["slam_t_us"])
    calib = camera_extrinsics(folder)
    r_dc = calib["r_dc"]
    idx, valid = e1.lookup(rec, 0.0)
    upright_slam = valid & rec["slam_ok"] & rec["upright"][idx]
    axis = axis_check(r_wd @ r_dc, rec["head_pos"], upright_slam)
    axis["rgb_yaw_right_of_device_z_deg"] = calib["rgb_yaw_right_of_device_z_deg"]
    axis["stereo_vs_rgb_yaw_diff_deg"] = calib["stereo_vs_rgb_yaw_diff_deg"]
    n = len(rec["both"])
    _, head_z = s0.to_grid_30(t_x, pos[:, SEG["head"], 2])
    toes = [s0.to_grid_30(t_x, pos[:, SEG[f"{s}_toe"], 2])[1] for s in ("l", "r")]
    head_z = head_z[:n]
    toe_low = np.minimum(toes[0][:n], toes[1][:n])
    finite = np.isfinite(head_z) & np.isfinite(toe_low)
    sel = rec["both"] & rec["upright"] & finite
    if sel.sum() < 30 * HZ:
        raise FloorError(
            "fewer than 30 s of finite upright both-planted frames: no floor estimate"
        )
    xs_floor = float(np.median(toe_low[sel]))
    ok = valid & rec["slam_ok"] & sel[idx]
    if ok.sum() < 30 * HZ:
        raise FloorError("fewer than 30 s of usable frames for the floor estimate")
    offset = rec["head_pos"][:, 2] - head_z[idx]  # SLAM z minus Xsens head z
    c = float(np.median(offset[ok]))
    floor_z = c + xs_floor
    if not np.isfinite(floor_z):
        raise FloorError("the floor estimate is not finite")
    xs_z = np.where(
        valid & np.isfinite(head_z[idx]), head_z[idx] + c, np.nan
    )  # Xsens head height in the SLAM z frame
    return {
        "camera_R": eye_rotation(r_wd, r_dc).astype(np.float32),
        "floor_z": np.array(floor_z),
        "xsens_head_z": xs_z.astype(np.float32),
        "axis": np.array(json.dumps(axis)),
    }


def input_key(prepared: dict) -> str:
    code = (
        inspect.getsource(prepare_inputs_one)
        + inspect.getsource(eye_rotation)
        + inspect.getsource(camera_extrinsics)
        + inspect.getsource(quaternion_to_matrix)
        + inspect.getsource(axis_check)
        + inspect.getsource(InputError)
        + inspect.getsource(FloorError)
        + inspect.getsource(CalibrationError)
        + INPUT_VERSION
    )
    return f"{prepared['check_sha']}-{hashlib.sha256(code.encode()).hexdigest()[:16]}"


def prepare_inputs(
    root: Path,
    prepared: dict,
    cache: Path,
    rebuild: bool,
    allow_axis_failure: bool,
) -> dict:
    key = input_key(prepared)
    folder = cache / "inputs"
    folder.mkdir(parents=True, exist_ok=True)
    inputs, excluded = {}, {}
    names = prepared["names"]
    for number, name in enumerate(names, 1):
        file = folder / f"{name}.npz"
        inp = None
        if file.exists() and not rebuild:
            with np.load(file) as data:
                if str(data["cache_key"]) == key:
                    inp = {k: data[k] for k in data.files if k != "cache_key"}
        if inp is None:
            started = time.time()
            try:
                inp = prepare_inputs_one(root / name, prepared["recs"][name])
            except InputError as err:
                excluded[name] = str(err)
                print(
                    f"[{number}/{len(names)}] {name[:46]} EXCLUDED: {err}", flush=True
                )
                continue
            np.savez_compressed(file, cache_key=np.array(key), **inp)
            print(
                f"[{number}/{len(names)}] {name[:46]} inputs in {time.time() - started:.0f} s",
                flush=True,
            )
        inputs[name] = inp
    axes = {n: json.loads(str(v["axis"])) for n, v in inputs.items()}
    gate = axis_gate(axes)
    if not gate["passes"] and not allow_axis_failure:
        raise RuntimeError(
            "the camera-frame check (x down, z forward, pooled over the sequences) failed; "
            "nothing was scored:\n  " + "\n  ".join(gate["reasons"])
        )
    return {
        "inputs": inputs,
        "key": key,
        "axis_gate": gate,
        "excluded": excluded,
    }


# ---------------------------------------------------------------- the model


def load_model(path: Path, device: str):
    import torch
    from hready.models.evidence_locked_completion import EvidenceLockedCompletion

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != CKPT_SHA256:
        raise RuntimeError(
            f"checkpoint {path} has SHA-256 {digest}, expected {CKPT_SHA256}"
        )
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = EvidenceLockedCompletion(**state["run_cfg"]["model"])
    model.load_state_dict(state["model"])
    return model.to(device).eval(), {"step": int(state["step"]), "sha256": digest}


def window_ok(finite: np.ndarray) -> np.ndarray:
    """True at index e when frames e-63..e are all true."""
    c = np.concatenate([[0], np.cumsum(finite.astype(np.int64))])
    ok = np.zeros(len(finite), bool)
    if len(finite) >= WINDOW:
        e = np.arange(WINDOW - 1, len(finite))
        ok[e] = (c[e + 1] - c[e + 1 - WINDOW]) == WINDOW
    return ok


def score_windows(
    model, device: str, pos: np.ndarray, cam_r: np.ndarray, ends: np.ndarray, batch: int
):
    """Contact score (minimum of the four contact probabilities) for the windows ending at ``ends``.
    ``pos`` is the head position with the floor at z = 0. Returns the outputs at the last frame and at
    the frame LOOKAHEAD frames before it. Windows are made relative to their first frame in x and y,
    as the model's training windows were."""
    import torch

    last = np.full(len(pos), np.nan, np.float32)
    early = np.full(len(pos), np.nan, np.float32)
    offsets = np.arange(-(WINDOW - 1), 1)
    g = torch.tensor(GRAVITY, dtype=torch.float32, device=device)
    for start in range(0, len(ends), batch):
        e = ends[start : start + batch]
        idx = e[:, None] + offsets[None, :]
        p = pos[idx].astype(np.float32).copy()  # (B, 64, 3)
        p[:, :, :2] -= p[:, :1, :2]
        r = cam_r[idx].astype(np.float32)  # (B, 64, 3, 3)
        t = -np.einsum("btij,btj->bti", r, p)
        b = len(e)
        rig = {
            "head_pos_world": torch.from_numpy(p).to(device),
            "camera_R": torch.from_numpy(r).to(device),
            "camera_t": torch.from_numpy(t.astype(np.float32)).to(device),
            "gravity_world": g.expand(b, 3),
        }
        obs = {
            "joint_pos_3d": torch.zeros(b, WINDOW, NUM_KP, 3, device=device),
            "joint_visible": torch.zeros(
                b, WINDOW, NUM_KP, dtype=torch.bool, device=device
            ),
            "joint_confidence": torch.zeros(b, WINDOW, NUM_KP, device=device),
        }
        with torch.no_grad():
            prob = torch.sigmoid(model(obs, rig)["contact_logits"])  # (B, 64, 4)
        score = prob.min(dim=-1).values.cpu().numpy()
        last[e] = score[:, WINDOW - 1]
        early[e - LOOKAHEAD] = score[:, WINDOW - 1 - LOOKAHEAD]
    return last, early


def condition_pos(rec: dict, inp: dict, condition: str) -> np.ndarray:
    """Head position with the floor at z = 0 for one condition (x and y are the SLAM ones)."""
    pos = rec["head_pos"].astype(np.float64).copy()
    floor = float(inp["floor_z"])
    if condition == "main":
        pos[:, 2] -= floor
    elif condition.startswith("floor"):
        shift = int(condition[5:-2]) / 100.0
        pos[:, 2] -= floor + shift
    elif condition == "xsens_height":
        pos[:, 2] = inp["xsens_head_z"].astype(np.float64) - floor
    elif condition == "constant_height":
        pos[:, 2] = CONSTANT_HEIGHT_M
    else:
        raise ValueError(condition)
    return pos


def score_key(ins: dict, info: dict, stride: int) -> str:
    code = (
        inspect.getsource(score_windows)
        + inspect.getsource(condition_pos)
        + inspect.getsource(window_ok)
        + INPUT_VERSION
        + str(GRAVITY)
        + str(CONSTANT_HEIGHT_M)
        + str(WINDOW)
        + str(LOOKAHEAD)
    )
    return f"{ins['key']}-{info['sha256'][:16]}-{hashlib.sha256(code.encode()).hexdigest()[:16]}-s{stride}"


def score_all(
    prepared: dict,
    ins: dict,
    cache: Path,
    ckpt: Path,
    device: str,
    batch: int,
    conditions,
    rebuild: bool,
    stride: int = 1,
) -> tuple[dict, dict]:
    model, info = load_model(ckpt, device)
    key = score_key(ins, info, stride)
    scores: dict = {c: {} for c in conditions}
    started = time.time()
    done = 0
    for condition in conditions:
        folder = cache / "scores" / condition
        folder.mkdir(parents=True, exist_ok=True)
        for number, name in enumerate(prepared["names"], 1):
            file = folder / f"{name}.npz"
            if file.exists() and not rebuild:
                with np.load(file) as data:
                    if str(data["cache_key"]) == key:
                        scores[condition][name] = (data["causal"], data["lookahead"])
                        continue
            rec, inp = prepared["recs"][name], ins["inputs"][name]
            pos = condition_pos(rec, inp, condition)
            finite = rec["slam_ok"] & np.isfinite(pos).all(axis=1)
            ends = np.flatnonzero(window_ok(finite))[::stride]
            causal, lookahead = score_windows(
                model, device, pos, inp["camera_R"], ends, batch
            )
            np.savez_compressed(
                file, cache_key=np.array(key), causal=causal, lookahead=lookahead
            )
            scores[condition][name] = (causal, lookahead)
            done += 1
            print(
                f"[{condition} {number}/{len(prepared['names'])}] {name[:40]} {len(ends)} windows, "
                f"{time.time() - started:.0f} s elapsed",
                flush=True,
            )
    info["scored_sequences_this_run"] = done
    return scores, info


# ---------------------------------------------------------------- evaluation


def r1_data(
    recs: dict, names: list[str], scores: dict, row: int, shift: float, roll: int
) -> dict:
    data = {}
    for name in names:
        rec = recs[name]
        score = scores[name][row].astype(np.float64)
        idx, valid = e1.lookup(rec, shift)
        valid = valid & rec["slam_ok"] & np.isfinite(score)
        lab = {k: rec[k][idx] for k in ("both", "hard", "upright", "foot_speed")}
        if roll:
            lab = {k: np.roll(v, roll) for k, v in lab.items()}
        data[name] = {
            "x": score[valid],
            "y": lab["both"][valid],
            "hard": lab["hard"][valid],
            "up": lab["upright"][valid],
            "fs": lab["foot_speed"][valid],
        }
    return data


def r1_evaluate(
    recs: dict,
    names: list[str],
    scores: dict,
    *,
    causal: bool,
    shift=0.0,
    roll=0,
    keep=False,
) -> dict:
    row = 0 if causal else 1
    data = r1_data(recs, names, scores, row, shift, roll)
    out = {n: {} for n in names}
    points = {}
    rng = np.random.default_rng(0)
    probs, labels = [], []
    for f in range(5):
        train = [n for n in names if s0.fold_of(n) != f]
        test = [n for n in names if s0.fold_of(n) == f]
        if not test or not train:
            continue
        x = np.concatenate([data[n]["x"] for n in train])
        y = np.concatenate([data[n]["y"] for n in train])
        ops = {
            k: -v for k, v in e1.thresholds(-x, y).items()
        }  # lock when score > threshold
        points[f] = ops
        for n in test:
            d = data[n]
            out[n]["R1"] = {}
            for op in ("ba", "fl5"):
                lock = d["x"] > ops[op]
                m = e1.metrics(lock, d)
                if keep and op == "fl5":
                    m["missed_lock_block_ci95"] = e1.block_ci(d["y"], lock, rng)
                out[n]["R1"][op] = m
            if keep:
                probs.append(d["x"])
                labels.append(d["y"])
    extra = {"operating_points_by_fold": points}
    if keep and probs:
        extra["R1_ece_pooled_min_of_four"] = e1.ece(
            np.concatenate(probs), np.concatenate(labels)
        )
    return {"per_sequence": out, "extra": extra}


def masks(scores: dict, row: int) -> dict:
    return {n: np.isfinite(v[row]) for n, v in scores.items()}


def merged(base: dict, r1: dict) -> dict:
    per = {
        n: {**base["per_sequence"][n], **r1["per_sequence"][n]}
        for n in r1["per_sequence"]
    }
    return {"per_sequence": per, "extra": {**base["extra"], **r1["extra"]}}


def rule_r1(result: dict, names: list[str], decision: bool) -> dict:
    """The rule of the first protocol with R1 in place of the learned detector: the signal check on R1's
    balanced accuracy and the main comparison R1 against B2 at "fl5" (the same code, e1.rule)."""
    per = {
        n: {"B1": v["R1"], "B2": v["B2"], "M1": v["R1"]}
        for n, v in result["per_sequence"].items()
    }
    r = e1.rule({"per_sequence": per}, names, set(), decision)
    held = r["main_comparison"].pop("median_held_out_false_lock_at_fl5")
    real = {
        det: e1.summarise_values(
            [result["per_sequence"][n][det]["fl5"]["false_lock"] for n in names]
        ).get("median")
        for det in ("B1", "B2", "M1")
    }
    r["main_comparison"]["median_held_out_false_lock_at_fl5"] = {
        "R1": held["M1"],
        "B2": held["B2"],
        "B1": real["B1"],
        "M1": real["M1"],
    }
    r["main_comparison"]["share_r1_lower"] = r["main_comparison"].pop(
        "share_m1_lower", None
    )
    pairs = [
        (v["R1"]["fl5"]["false_lock"], v["B2"]["fl5"]["false_lock"])
        for v in result["per_sequence"].values()
        if v["R1"]["fl5"]["false_lock"] is not None
        and v["B2"]["fl5"]["false_lock"] is not None
    ]
    r["main_comparison"]["share_r1_false_lock_not_higher_than_b2"] = (
        float(np.mean([a <= b for a, b in pairs])) if pairs else None
    )
    r["outcome"] = (
        r["outcome"]
        .replace(
            "a learned detector did not beat", "the repo's trained model did not beat"
        )
        .replace(
            "the head pose alone is reported as insufficient",
            "the model's signal on the head pose is insufficient",
        )
    )
    return r


def summary_row(result: dict, names: list[str], det: str = "R1") -> dict:
    row = {}
    for op in ("ba", "fl5"):
        row[op] = {
            k: e1.summarise_values(
                [result["per_sequence"][n][det][op][k] for n in names]
            ).get("median")
            for k in (
                "balanced_accuracy",
                "missed_lock",
                "false_lock",
                "hard_missed_lock",
            )
        }
    return row


def source_hashes() -> dict:
    """Hashes of the model code and versions, so that the report says what ran."""
    import hready
    import torch

    base = Path(hready.__file__).resolve().parent
    files = (
        "models/evidence_locked_completion.py",
        "models/ego_complete_motion.py",
        "data/ego_observation.py",
    )
    protocol = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "nymeria_foot_lock_repo_model_protocol.md"
    )
    return {
        "hready_sources_sha256_16": {f: e1.sha16(base / f) for f in files},
        "protocol_sha256_16": e1.sha16(protocol) if protocol.exists() else None,
        "torch": torch.__version__,
        "numpy": np.__version__,
    }


def run(
    prepared: dict, ins: dict, scores: dict, info: dict, out: Path, controls: bool
) -> dict:
    names, recs = prepared["names"], prepared["recs"]
    report: dict = {
        "provenance": {
            "model": info,
            "code_sha256_16": e1.sha16(Path(__file__)),
            "experiment_code_sha256_16": e1.sha16(Path(e1.__file__)),
            "data_check_report_sha256_16": prepared["check_sha"],
            "sequences": len(names),
            "axis_gate": ins["axis_gate"],
            "excluded_no_floor_estimate": ins["excluded"],
            "input_cache_key": ins["key"],
            **source_hashes(),
        },
        "inputs": {
            n: {
                "axis": json.loads(str(ins["inputs"][n]["axis"])),
                "floor_z_slam": float(ins["inputs"][n]["floor_z"]),
            }
            for n in names
        },
        "main": {},
        "controls": {"home_domain_check": "not run"},
    }
    for label, causal in (("causal", True), ("lookahead", False)):
        row = 0 if causal else 1
        mask = masks(scores["main"], row)
        base = e1.evaluate(recs, names, causal=causal, keep=True, mask=mask)
        r1 = r1_evaluate(recs, names, scores["main"], causal=causal, keep=True)
        result = merged(base, r1)
        report["main"][label] = {
            "table": {
                det: summary_row(result, names, det) for det in ("B1", "B2", "M1", "R1")
            },
            "table_full": e1.table(result, ("B1", "B2", "M1", "R1")),
            "rule": rule_r1(result, names, decision=causal),
            "extra": result["extra"],
            "per_sequence": result["per_sequence"],
        }
        e1.write_report(report, out)
    if controls:
        causal_rows = {}
        for condition in CONDITIONS[1:]:
            if condition in scores:
                r = r1_evaluate(recs, names, scores[condition], causal=True)
                causal_rows[condition] = summary_row(r, names)
        report["controls"]["inputs_causal"] = causal_rows
        shifts = {}
        for shift in CLOCK_SHIFTS_S:
            for label, causal in (("causal", True), ("lookahead", False)):
                r = r1_evaluate(recs, names, scores["main"], causal=causal, shift=shift)
                shifts[f"{label}_{shift:+.2f}s"] = summary_row(r, names)
        report["controls"]["clock_shift"] = shifts
        r = r1_evaluate(
            recs, names, scores["main"], causal=True, roll=e1.SHUFFLE_FRAMES
        )
        report["controls"]["label_roll_60s_causal"] = summary_row(r, names)
        e1.write_report(report, out)
    return report


def med_text(v) -> str:
    return "   n/a" if v is None else f"{v:6.3f}"


def show(report: dict) -> None:
    for label, block in report["main"].items():
        print(f"\n== {label} rows (median over held-out sequences) ==")
        print(
            f"{'det':4s} {'op':4s} {'BA':>6s} {'missed':>7s} {'false':>7s} {'hard miss':>9s}"
        )
        for det, ops in block["table"].items():
            for op, t in ops.items():
                print(
                    f"{det:4s} {op:4s} {med_text(t['balanced_accuracy'])} {med_text(t['missed_lock'])} "
                    f"{med_text(t['false_lock'])} {med_text(t['hard_missed_lock'])}"
                )
        r = block["rule"]
        print("signal check:", json.dumps(r["signal_check"]))
        print("main comparison:", json.dumps(r["main_comparison"]))
        print("outcome:", r["outcome"])
    if report["controls"]:
        print("\n== controls (R1, causal rows, medians) ==")
        for name, row in report["controls"].get("inputs_causal", {}).items():
            print(
                f"{name:18s} fl5 missed {med_text(row['fl5']['missed_lock'])}  false {med_text(row['fl5']['false_lock'])}"
            )
        for name, row in report["controls"].get("clock_shift", {}).items():
            print(f"clock {name:18s} fl5 missed {med_text(row['fl5']['missed_lock'])}")
        row = report["controls"].get("label_roll_60s_causal")
        if row:
            print(
                f"label roll 60 s    ba balanced accuracy {med_text(row['ba']['balanced_accuracy'])}"
            )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--step", choices=("prepare", "score", "run", "all"), default="all"
    )
    parser.add_argument("--root", type=Path, default=s0.sync.DEFAULT_ROOT)
    parser.add_argument("--data-check", type=Path, default=e1.DATA_CHECK)
    parser.add_argument("--experiment-cache", type=Path, default=e1.CACHE)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--ckpt", type=Path, default=CKPT)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument(
        "--device", default=None, help="default: cuda if available, else cpu"
    )
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="score every N-th window (quick look only; the protocol rule is valid at 1)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="first N cohort sequences (testing)"
    )
    parser.add_argument(
        "--no-controls", action="store_true", help="main condition only"
    )
    parser.add_argument(
        "--allow-axis-failure",
        action="store_true",
        help="TEST ONLY: continue when the camera-frame check fails (the failure is written to the report)",
    )
    args = parser.parse_args(argv)
    prepared = e1.prepare(
        args.root, args.data_check, args.experiment_cache, False, args.limit
    )
    ins = prepare_inputs(
        args.root, prepared, args.cache, args.rebuild, args.allow_axis_failure
    )
    if ins["excluded"]:
        prepared = {
            **prepared,
            "names": [n for n in prepared["names"] if n not in ins["excluded"]],
        }
        print(
            f"{len(ins['excluded'])} sequence(s) excluded (no floor estimate): "
            f"{', '.join(ins['excluded'])}"
        )
    gate = ins["axis_gate"]
    print(
        f"camera-frame check: {'pass' if gate['passes'] else 'FAIL'}; {gate['sequences']} sequences, "
        f"walking direction tested on {gate['tested_forward']}; median gravity along the down axis "
        f"{gate['median_gravity_along_down']:.2f}, median walking along the look direction "
        f"{gate['median_walking_along_look']:.2f}, {len(gate['low_forward'])} sequence(s) below {AXIS_LOW_FORWARD}; "
        f"largest yaw difference between RGB extrinsics and SLAM baseline {gate['max_stereo_vs_rgb_yaw_diff_deg']:.1f} deg"
        if gate["sequences"] and gate["tested_forward"]
        else "camera-frame check: not applicable (no sequences or no walking)"
    )
    if args.step == "prepare":
        return 0
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    conditions = ("main",) if args.no_controls else CONDITIONS
    scores, info = score_all(
        prepared,
        ins,
        args.cache,
        args.ckpt,
        device,
        args.batch,
        conditions,
        args.rebuild,
        args.stride,
    )
    info["device"] = device
    info["window_stride"] = args.stride
    info["axis_check_ignored"] = bool(args.allow_axis_failure and not gate["passes"])
    if args.step == "score":
        return 0
    report = run(prepared, ins, scores, info, args.out, controls=not args.no_controls)
    show(report)
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
