"""Check the timing and the geometry between Nymeria body motion, head SLAM and the RGB video.

Everything is read in place (no extraction): ``body/xdata.npz`` from ``body_motion.zip``, the head
(and, when present, wrist) SLAM trajectories and calibration from the ``recording_*.zip`` files, the
preview video through ``ffmpeg``. No package beyond numpy. The fisheye model is
``FisheyeRadTanThinPrism`` (15 parameters) and was checked against ``projectaria_tools`` 2.4.0.

Steps (``--steps``, comma separated, default all except ``scan``):
  fit      refine the Xsens-to-SLAM time offset to ~1 ms and fit the world transform: a rotation
           about the vertical axis plus a translation, plus a constant offset between the Xsens head
           segment origin and the SLAM device origin. Reports the residual of the head position and
           the spread of the head orientation (both would be ~0 for a perfect alignment).
  drift    the same offset in 30 s windows: does it move over the recording?
  yaw      the world fit in 120 s windows: does the Xsens heading drift?
  wrist    when ``recording_lwrist.zip`` / ``recording_rwrist.zip`` are present: the offset of each
           wrist device against the matching Xsens hand, and the Xsens-to-UTC time for head and
           wrists, which must agree if every clock is consistent.
  video    video-to-SLAM timing: the image change per frame against the SLAM rotation speed.
  project  project the Xsens elbows, hands, knees, feet and toes into the video frames at the
           narrated kneel / sit-on-ground moments and at evenly spaced moments; PNG overlays.
  scan     (not in the default) for a sequence whose head-segment peak is weak: the best correlation
           of every Xsens segment against the SLAM rotation speed.

Usage (hready environment, Windows):
    python scripts/nymeria_project_check.py
    python scripts/nymeria_project_check.py --sequence 20230706_s0_breanna_reese_act0_cfg468 --steps fit,project
    python scripts/nymeria_project_check.py --sequence 20230816_s1_jeffery_bryant_act0_p5w199 --steps scan
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import shutil
import struct
import subprocess
import sys
import time
import zipfile
import zlib
from pathlib import Path

import numpy as np

import nymeria_motion_sync as sync
from mp4_probe import frame_times_ns

PROJECTION_OUT = Path(r"D:\projects\hready_data\review\nymeria_projection")
FRAMES_INDEX = Path(r"D:\projects\hready_data\review\nymeria_floor_frames\index.json")
CALIB_SIZE = (
    2880  # assumed square calibration frame of camera-rgb (principal point ~1440)
)
THETA_MAX = math.radians(
    65.0
)  # the radial polynomial is only trusted inside this angle
SLAM_STRIDE = 10  # ~100 Hz from the ~1 kHz trajectory
MIN_RELIABLE_CORR = (
    0.5  # below this the coarse offset is not trusted and the sequence is skipped
)
FFMPEG_TIMEOUT_S = (
    600.0  # one decode of a 60 s window takes seconds; this only stops a hang
)
SEGMENTS = {
    "pelvis": 0,
    "head": 6,
    "r_elbow": 9,
    "r_hand": 10,
    "l_elbow": 13,
    "l_hand": 14,
    "r_knee": 16,
    "r_foot": 17,
    "r_toe": 18,
    "l_knee": 20,
    "l_foot": 21,
    "l_toe": 22,
}
COLOURS = {  # right side red tones, left side blue tones, pelvis and head green
    "pelvis": (0, 200, 0),
    "head": (0, 255, 0),
    "r_elbow": (255, 160, 160),
    "r_hand": (255, 0, 0),
    "r_knee": (255, 128, 0),
    "r_foot": (255, 255, 0),
    "r_toe": (255, 200, 0),
    "l_elbow": (160, 200, 255),
    "l_hand": (0, 90, 255),
    "l_knee": (0, 200, 255),
    "l_foot": (0, 255, 255),
    "l_toe": (140, 255, 255),
}
HAND = {"recording_lwrist": SEGMENTS["l_hand"], "recording_rwrist": SEGMENTS["r_hand"]}


# ---------------------------------------------------------------- rotations and camera


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Rotation matrices from quaternions ordered (w, x, y, z); shape [..., 4] -> [..., 3, 3]."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    rot = np.empty((*q.shape[:-1], 3, 3))
    rot[..., 0, 0] = 1 - 2 * (y * y + z * z)
    rot[..., 0, 1] = 2 * (x * y - z * w)
    rot[..., 0, 2] = 2 * (x * z + y * w)
    rot[..., 1, 0] = 2 * (x * y + z * w)
    rot[..., 1, 1] = 1 - 2 * (x * x + z * z)
    rot[..., 1, 2] = 2 * (y * z - x * w)
    rot[..., 2, 0] = 2 * (x * z - y * w)
    rot[..., 2, 1] = 2 * (y * z + x * w)
    rot[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return rot


def rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def project_fisheye624(
    points: np.ndarray, params: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``FisheyeRadTanThinPrism`` projection of camera-frame points [M, 3].

    ``params`` = [f, cx, cy, k0..k5, p0, p1, s0..s3] in calibration pixels. Returns the pixel
    coordinates [M, 2] and a validity mask (in front, inside the trusted angle, inside the frame).
    Matches ``projectaria_tools`` 2.4.0 (CameraModelType.FISHEYE624) to 1e-12 px.
    """
    f, cx, cy = params[:3]
    k = params[3:9]
    p0, p1 = params[9:11]
    s = params[11:15]
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.hypot(x, y)
    theta = np.arctan2(r, z)
    t2 = theta * theta
    poly = 1 + t2 * (
        k[0] + t2 * (k[1] + t2 * (k[2] + t2 * (k[3] + t2 * (k[4] + t2 * k[5]))))
    )
    scale = np.where(
        r > 1e-9, theta * poly / np.maximum(r, 1e-9), 1.0 / np.maximum(z, 1e-9)
    )
    xr, yr = x * scale, y * scale
    r2 = xr * xr + yr * yr
    r4 = r2 * r2
    xd = xr + (2 * xr * xr + r2) * p0 + 2 * xr * yr * p1 + s[0] * r2 + s[1] * r4
    yd = yr + 2 * xr * yr * p0 + (2 * yr * yr + r2) * p1 + s[2] * r2 + s[3] * r4
    uv = np.stack([f * xd + cx, f * yd + cy], axis=1)
    valid = (
        (z > 0)
        & (theta < THETA_MAX)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < CALIB_SIZE)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < CALIB_SIZE)
    )
    return uv, valid


def to_image(uv: np.ndarray, size: int, rotation: str) -> tuple[float, float]:
    """Map a calibration pixel to the video frame of ``size`` pixels.

    The RGB sensor is mounted sideways and the preview video is rotated upright, so the
    calibration frame has to be rotated the same way: ``cw90`` (checked on the overlays),
    ``ccw90`` or ``none``.
    """
    u, v = uv * size / CALIB_SIZE
    if rotation == "cw90":
        return size - v, u
    if rotation == "ccw90":
        return v, size - u
    return u, v


def find_key(node, key: str):
    if isinstance(node, dict):
        if key in node:
            return node[key]
        for value in node.values():
            found = find_key(value, key)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = find_key(value, key)
            if found is not None:
                return found
    return None


def load_camera(head_zip: Path, label: str = "camera-rgb") -> dict:
    """Intrinsics and device-to-camera pose from the first line of the head online calibration."""
    with zipfile.ZipFile(head_zip) as archive:
        member = next(
            n for n in archive.namelist() if n.endswith("online_calibration.jsonl")
        )
        with archive.open(member) as handle:
            first = json.loads(handle.readline())
    for cam in find_key(first, "CameraCalibrations"):
        if cam.get("Label") == label:
            pose = cam["T_Device_Camera"]
            w, (qx, qy, qz) = pose["UnitQuaternion"]
            return {
                "params": np.array(cam["Projection"]["Params"], dtype=np.float64),
                "model": cam["Projection"]["Name"],
                "r_dev_cam": quat_to_rot(np.array([w, qx, qy, qz], dtype=np.float64)),
                "t_dev_cam": np.array(pose["Translation"], dtype=np.float64),
            }
    raise KeyError(f"camera {label} not in the calibration")


# ---------------------------------------------------------------- loading


def load_xsens(body_zip: Path) -> dict:
    """Times, positions, orientations and rotation speeds of all segments (memory only)."""
    with zipfile.ZipFile(body_zip) as archive:
        raw = archive.read("body/xdata.npz")
    with np.load(io.BytesIO(raw)) as data:
        t = data["timestamps_us"].astype(np.float64)
        pos = data["segment_tXYZ"].reshape(len(t), -1, 3)
        quat = data["segment_qWXYZ"].reshape(len(t), -1, 4)
        omega = np.linalg.norm(
            data["segment_angularVelocity"].reshape(len(t), -1, 3), axis=2
        )
    return {"t_us": t, "pos": pos, "quat": quat, "omega": omega}


def load_trajectory(zip_path: Path, device: str, stride: int = SLAM_STRIDE) -> dict:
    """Pose and rotation speed of one device (every ``stride``-th row of the SLAM trajectory)."""
    member = f"{device}/mps/slam/closed_loop_trajectory.csv"
    track, utc, pos, quat, omega = [], [], [], [], []
    with zipfile.ZipFile(zip_path) as archive, archive.open(member) as handle:
        reader = csv.reader(io.TextIOWrapper(handle, encoding="utf-8", newline=""))
        header = next(reader)
        col = {name: header.index(name) for name in header}
        i_p = [col[f"t{a}_world_device"] for a in "xyz"]
        i_q = [col[f"q{a}_world_device"] for a in "wxyz"]
        i_w = [col[f"angular_velocity_{a}_device"] for a in "xyz"]
        for number, row in enumerate(reader):
            if number % stride:
                continue
            track.append(float(row[col["tracking_timestamp_us"]]))
            utc.append(float(row[col["utc_timestamp_ns"]]) / 1000.0)
            pos.append([float(row[i]) for i in i_p])
            quat.append([float(row[i]) for i in i_q])
            omega.append(math.sqrt(sum(float(row[i]) ** 2 for i in i_w)))
    quat_arr = np.array(quat)
    flips = np.where(np.sum(quat_arr[1:] * quat_arr[:-1], axis=1) < 0)[0]
    for i in flips:  # make the sign continuous so that interpolation is safe
        quat_arr[i + 1 :] *= -1
    return {
        "t_us": np.array(track),
        "utc_us": np.array(utc),
        "pos": np.array(pos),
        "quat": quat_arr,
        "omega": np.array(omega),
    }


def interp_pose(traj: dict, t_us: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Position [M, 3] and rotation [M, 3, 3] (world from device) at the given times."""
    t = traj["t_us"]
    idx = np.clip(np.searchsorted(t, t_us) - 1, 0, len(t) - 2)
    w = ((t_us - t[idx]) / (t[idx + 1] - t[idx]))[:, None]
    pos = traj["pos"][idx] * (1 - w) + traj["pos"][idx + 1] * w
    quat = traj["quat"][idx] * (1 - w) + traj["quat"][idx + 1] * w
    return pos, quat_to_rot(quat)


def xsens_at(xs: dict, t_us: np.ndarray, segment: int) -> tuple[np.ndarray, np.ndarray]:
    """Segment position (linear) and rotation (nearest frame) at Xsens times [us]."""
    t = xs["t_us"]
    idx = np.clip(np.searchsorted(t, t_us) - 1, 0, len(t) - 2)
    w = ((t_us - t[idx]) / (t[idx + 1] - t[idx]))[:, None]
    pos = xs["pos"][idx, segment] * (1 - w) + xs["pos"][idx + 1, segment] * w
    near = np.where(w[:, 0] < 0.5, idx, idx + 1)
    return pos, quat_to_rot(xs["quat"][near, segment])


def smooth(values: np.ndarray, t_us: np.ndarray, seconds: float) -> np.ndarray:
    rate = (len(t_us) - 1) / ((t_us[-1] - t_us[0]) / 1e6)
    width = (
        max(1, round(float(rate * seconds))) | 1
    )  # odd, so the box does not shift the series
    return np.convolve(values, np.ones(width) / width, mode="same")


def grid(t_us, values, t0_us, count, hz, seconds):
    return np.interp(
        t0_us + np.arange(count) * 1e6 / hz, t_us, smooth(values, t_us, seconds)
    )


# ---------------------------------------------------------------- offset refinement and drift


def parabolic(lags: np.ndarray, scores: np.ndarray, i: int) -> float:
    """Sub-step position of the maximum around index i."""
    if 0 < i < len(scores) - 1:
        a, b, c = scores[i - 1], scores[i], scores[i + 1]
        denominator = a - 2 * b + c
        if denominator != 0:
            return float(lags[i] + 0.5 * (a - c) / denominator * (lags[1] - lags[0]))
    return float(lags[i])


def common_times(slam_t, xs_t, offset_us, margin_s=1.0):
    """SLAM-clock times (100 Hz) that have Xsens data at ``t - offset`` with a margin."""
    lo = max(slam_t[0], xs_t[0] + offset_us) + margin_s * 1e6
    hi = min(slam_t[-1], xs_t[-1] + offset_us) - margin_s * 1e6
    return slam_t[(slam_t >= lo) & (slam_t <= hi)]


def refine_offset(
    slam: dict, xs: dict, segment: int, offset_us: float, half_range_s=0.25, step_ms=2.0
) -> dict:
    """Fine search of the offset around a coarse one, correlating rotation speeds at ~100 Hz."""
    t = common_times(slam["t_us"], xs["t_us"], offset_us, margin_s=half_range_s + 1)
    if len(t) < 3000:
        return {"error": "too little overlap"}
    s_smooth = smooth(slam["omega"], slam["t_us"], 0.05)
    x_smooth = smooth(xs["omega"][:, segment], xs["t_us"], 0.05)
    s_values = np.interp(t, slam["t_us"], s_smooth)
    lags = np.arange(-half_range_s, half_range_s + 1e-9, step_ms / 1000.0)
    scores = np.array(
        [
            sync.correlation(
                s_values, np.interp(t - offset_us - lag * 1e6, xs["t_us"], x_smooth)
            )
            for lag in lags
        ]
    )
    i = int(np.nanargmax(scores))
    best = parabolic(lags, scores, i)
    return {
        "offset_s": offset_us / 1e6 + best,
        "delta_from_coarse_ms": best * 1000.0,
        "corr": float(scores[i]),
        "samples": len(t),
    }


def drift(
    slam: dict, xs: dict, segment: int, offset_us: float, window_s=30.0, max_lag_s=0.5
) -> dict:
    """The residual offset in consecutive windows."""
    t = common_times(slam["t_us"], xs["t_us"], offset_us, margin_s=max_lag_s + 1)
    if len(t) == 0:
        return {"error": "no overlap"}
    s_smooth = smooth(slam["omega"], slam["t_us"], 0.05)
    x_smooth = smooth(xs["omega"][:, segment], xs["t_us"], 0.05)
    lags = np.arange(-max_lag_s, max_lag_s + 1e-9, 0.005)
    rows = []
    start = t[0]
    while start + window_s * 1e6 <= t[-1]:
        sel = t[(t >= start) & (t < start + window_s * 1e6)]
        s_values = np.interp(sel, slam["t_us"], s_smooth)
        scores = np.array(
            [
                sync.correlation(
                    s_values,
                    np.interp(sel - offset_us - lag * 1e6, xs["t_us"], x_smooth),
                )
                for lag in lags
            ]
        )
        if not np.all(np.isnan(scores)):
            i = int(np.nanargmax(scores))
            rows.append(
                (
                    float((start - t[0]) / 1e6),
                    parabolic(lags, scores, i) * 1000.0,
                    float(scores[i]),
                )
            )
        start += window_s * 1e6
    good = [(a, b) for a, b, c in rows if c >= 0.8]
    out = {
        "windows": len(rows),
        "windows_corr_ge_0.8": len(good),
        "residual_ms": [round(b, 1) for _, b, _ in rows],
        "window_corr": [round(c, 2) for _, _, c in rows],
    }
    if len(good) >= 3:
        times = np.array([a for a, _ in good])
        values = np.array([b for _, b in good])
        slope = np.polyfit(times, values, 1)[0] * 3600.0
        out |= {
            "median_ms": float(np.median(values)),
            "iqr_ms": float(np.percentile(values, 75) - np.percentile(values, 25)),
            "slope_ms_per_hour": float(slope),
        }
    return out


# ---------------------------------------------------------------- world transform


def spread_deg(rotations: np.ndarray) -> float:
    """Mean angle of rotations [M,3,3] from their chordal mean."""
    u, _, vt = np.linalg.svd(rotations.sum(axis=0))
    mean = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    trace = np.einsum("mij,ij->m", rotations, mean)
    return float(np.degrees(np.mean(np.arccos(np.clip((trace - 1) / 2, -1, 1)))))


def fit_world(slam_pos, slam_rot, head_pos, head_rot, step_deg=1.0) -> dict:
    """Yaw angle, translation and head-frame offset: slam = T + Rz(theta) (x + R_xh d)."""
    count = len(slam_pos)
    eye = np.tile(np.eye(3), (count, 1, 1))

    def solve(theta):
        rz = rot_z(theta)
        target = slam_pos - head_pos @ rz.T
        design = np.concatenate([eye, rz @ head_rot], axis=2).reshape(3 * count, 6)
        sol, *_ = np.linalg.lstsq(design, target.reshape(-1), rcond=None)
        resid = target - np.einsum("mij,j->mi", design.reshape(count, 3, 6), sol)
        return float(np.sqrt(np.mean(np.sum(resid**2, axis=1)))), sol

    angles = np.radians(np.arange(0, 360, step_deg))
    costs = np.array([solve(a)[0] for a in angles])
    best = float(angles[int(np.argmin(costs))])
    fine = best + np.radians(np.arange(-step_deg, step_deg + 1e-9, 0.05))
    fine_costs = np.array([solve(a)[0] for a in fine])
    theta = float(fine[int(np.argmin(fine_costs))])
    rms, sol = solve(theta)
    far = np.abs((angles - theta + np.pi) % (2 * np.pi) - np.pi) > np.radians(10)
    rot_costs = np.array(
        [
            spread_deg(np.einsum("mji,jk,mkl->mil", slam_rot, rot_z(a), head_rot))
            for a in angles
        ]
    )
    theta_rot = float(angles[int(np.argmin(rot_costs))])
    return {
        "yaw_deg": math.degrees(theta) % 360,
        "translation_m": [float(v) for v in sol[:3]],
        "head_offset_m": [float(v) for v in sol[3:]],
        "head_offset_norm_m": float(np.linalg.norm(sol[3:])),
        "position_rms_m": rms,
        "second_best_rms_m_over_10deg_away": float(costs[far].min()),
        "yaw_from_orientation_deg": math.degrees(theta_rot) % 360,
        "orientation_spread_deg_at_yaw": spread_deg(
            np.einsum("mji,jk,mkl->mil", slam_rot, rot_z(theta), head_rot)
        ),
        "orientation_spread_deg_min": float(rot_costs.min()),
    }


def window_fits(
    head: dict, xs: dict, segment: int, offset_us: float, window_s=120.0
) -> list[dict]:
    """The world fit in consecutive windows: a drifting Xsens heading shows as a moving yaw."""
    t_all = common_times(head["t_us"], xs["t_us"], offset_us)[::20]
    rows = []
    start = t_all[0]
    while start + window_s * 1e6 <= t_all[-1]:
        t = t_all[(t_all >= start) & (t_all < start + window_s * 1e6)]
        pos_s, rot_s = interp_pose(head, t)
        pos_x, rot_x = xsens_at(xs, t - offset_us, segment)
        good = (
            np.isfinite(pos_s).all(axis=1)
            & np.isfinite(pos_x).all(axis=1)
            & np.isfinite(rot_s).all(axis=(1, 2))
            & np.isfinite(rot_x).all(axis=(1, 2))
        )
        if good.sum() < 100:
            rows.append(
                {
                    "start_s": round(float((start - t_all[0]) / 1e6)),
                    "skipped": "non-finite data",
                }
            )
            start += window_s * 1e6
            continue
        fit = fit_world(
            pos_s[good], rot_s[good], pos_x[good], rot_x[good], step_deg=2.0
        )
        rows.append(
            {
                "start_s": round(float((start - t_all[0]) / 1e6)),
                "yaw_deg": round(fit["yaw_deg"], 1),
                "position_rms_m": round(fit["position_rms_m"], 3),
                "orientation_spread_deg": round(
                    fit["orientation_spread_deg_at_yaw"], 1
                ),
                "head_offset_norm_m": round(fit["head_offset_norm_m"], 3),
            }
        )
        start += window_s * 1e6
    return rows


def fit_fixed_d(slam_pos, head_pos, head_rot, d, step_deg=2.0) -> dict:
    """Yaw and translation with the head-frame offset d given: slam = T + Rz(theta)(x + R d)."""
    v = head_pos + np.einsum("mij,j->mi", head_rot, d)

    def cost(theta):
        resid = slam_pos - v @ rot_z(theta).T
        resid = resid - resid.mean(axis=0)
        return float(np.sqrt(np.mean(np.sum(resid**2, axis=1))))

    angles = np.radians(np.arange(0, 360, step_deg))
    best = float(angles[int(np.argmin([cost(a) for a in angles]))])
    fine = best + np.radians(np.arange(-step_deg, step_deg + 1e-9, 0.05))
    theta = float(fine[int(np.argmin([cost(a) for a in fine]))])
    translation = (slam_pos - v @ rot_z(theta).T).mean(axis=0)
    return {
        "yaw_deg": math.degrees(theta) % 360,
        "translation_m": [float(t) for t in translation],
        "position_rms_m": cost(theta),
    }


def window_samples(head, xs, segment, offset_us, window_s=120.0) -> list[dict]:
    """Finite samples (5 Hz) of the head device and the Xsens segment, one dict per window."""
    t_all = common_times(head["t_us"], xs["t_us"], offset_us)[::20]
    out = []
    start = t_all[0]
    while start + window_s * 1e6 <= t_all[-1]:
        t = t_all[(t_all >= start) & (t_all < start + window_s * 1e6)]
        pos_s, rot_s = interp_pose(head, t)
        pos_x, rot_x = xsens_at(xs, t - offset_us, segment)
        good = (
            np.isfinite(pos_s).all(axis=1)
            & np.isfinite(pos_x).all(axis=1)
            & np.isfinite(rot_s).all(axis=(1, 2))
            & np.isfinite(rot_x).all(axis=(1, 2))
        )
        if good.sum() >= 100:
            out.append(
                {
                    "start_s": round(float((start - t_all[0]) / 1e6)),
                    "slam_pos": pos_s[good],
                    "slam_rot": rot_s[good],
                    "x_pos": pos_x[good],
                    "x_rot": rot_x[good],
                }
            )
        start += window_s * 1e6
    return out


def solve_shared_d(windows: list[dict], d0: np.ndarray, iterations=4) -> tuple:
    """One head offset d for all windows, a yaw and a translation for each (alternating)."""
    d = np.array(d0, dtype=np.float64)
    count = len(windows)
    fits = []
    for _ in range(iterations):
        fits = [fit_fixed_d(w["slam_pos"], w["x_pos"], w["x_rot"], d) for w in windows]
        rows, targets = [], []
        for k, (w, fit) in enumerate(zip(windows, fits, strict=True)):
            rz = rot_z(math.radians(fit["yaw_deg"]))
            n = len(w["slam_pos"])
            block = np.zeros((n, 3, 3 * count + 3))
            block[:, :, 3 * k : 3 * k + 3] = np.eye(3)
            block[:, :, -3:] = rz @ w["x_rot"]
            rows.append(block.reshape(3 * n, -1))
            targets.append((w["slam_pos"] - w["x_pos"] @ rz.T).reshape(-1))
        sol, *_ = np.linalg.lstsq(
            np.concatenate(rows), np.concatenate(targets), rcond=None
        )
        d = sol[-3:]
    fits = [fit_fixed_d(w["slam_pos"], w["x_pos"], w["x_rot"], d) for w in windows]
    return d, fits


def shared_head_offset(head, xs, segment, offset_us, window_s=120.0) -> dict | None:
    """The head-to-device offset d of one person, from all good windows together.

    Windows with a poor free fit (position rms > 0.10 m or orientation spread > 8 deg) are left
    out. ``half_difference_m`` is the distance between d from the even and from the odd windows:
    it says how well d is determined.
    """
    windows = window_samples(head, xs, segment, offset_us, window_s)
    starts, free = [], []
    for w in windows:
        fit = fit_world(
            w["slam_pos"], w["slam_rot"], w["x_pos"], w["x_rot"], step_deg=2.0
        )
        if (
            fit["position_rms_m"] <= 0.10
            and fit["orientation_spread_deg_at_yaw"] <= 8.0
        ):
            starts.append(w)
            free.append(fit)
    if len(starts) < 4:
        return None
    d0 = np.median([f["head_offset_m"] for f in free], axis=0)
    d, fits = solve_shared_d(starts, d0)
    d_even, _ = solve_shared_d(starts[0::2], d)
    d_odd, _ = solve_shared_d(starts[1::2], d)
    return {
        "d_m": [float(v) for v in d],
        "d_norm_m": float(np.linalg.norm(d)),
        "windows_used": len(starts),
        "windows_total": len(windows),
        "half_difference_m": float(np.linalg.norm(d_even - d_odd)),
        "free_d_norm_m_range": [
            float(min(np.linalg.norm(f["head_offset_m"]) for f in free)),
            float(max(np.linalg.norm(f["head_offset_m"]) for f in free)),
        ],
        "position_rms_m_median": float(np.median([f["position_rms_m"] for f in fits])),
    }


# ---------------------------------------------------------------- wrists


def xs_to_utc_s(traj: dict, offset_s: float) -> float:
    """Xsens time to UTC [s]: slam = xs + offset, utc = slam + median(utc - slam)."""
    return offset_s + float(np.median(traj["utc_us"] - traj["t_us"])) / 1e6


def wrist_check(folder: Path, xs: dict, head: dict, head_offset_s: float) -> dict:
    out = {
        "head": {
            "xsens_to_utc_s": xs_to_utc_s(head, head_offset_s),
            "utc_minus_track_std_ms": float(np.std(head["utc_us"] - head["t_us"]))
            / 1000.0,
        }
    }
    for device, segment in HAND.items():
        archive = sync.find(
            folder, device.removeprefix("recording_") + ".zip"
        ) or sync.find(folder, device + ".zip")
        if archive is None:
            out[device] = {"skipped": "zip not present"}
            continue
        try:
            traj = load_trajectory(archive, device)
        except (KeyError, OSError, zipfile.BadZipFile) as error:
            out[device] = {"skipped": f"{type(error).__name__}: {error}"}
            continue
        s0, x0 = traj["t_us"][0], xs["t_us"][0]
        hz = 10.0
        n_s = int((traj["t_us"][-1] - s0) / 1e6 * hz)
        n_x = int((xs["t_us"][-1] - x0) / 1e6 * hz)
        s_w = grid(traj["t_us"], traj["omega"], s0, n_s, hz, 0.2)
        x_w = grid(xs["t_us"], xs["omega"][:, segment], x0, n_x, hz, 0.2)
        needed = int(min(n_s, n_x) * 0.6)
        scan = [
            (k, c) for k, c in sync.offset_search(s_w, x_w, needed) if not np.isnan(c)
        ]
        if not scan:
            out[device] = {"skipped": "no usable overlap"}
            continue
        k, c = max(scan, key=lambda kc: kc[1])
        coarse_us = (s0 - x0) + k * 1e6 / hz
        refined = refine_offset(traj, xs, segment, coarse_us)
        entry = {
            "corr_coarse": c,
            "utc_minus_track_std_ms": float(np.std(traj["utc_us"] - traj["t_us"]))
            / 1000.0,
            **refined,
        }
        if "offset_s" in refined:
            entry["xsens_to_utc_s"] = xs_to_utc_s(traj, refined["offset_s"])
            entry["utc_difference_to_head_ms"] = (
                entry["xsens_to_utc_s"] - out["head"]["xsens_to_utc_s"]
            ) * 1000.0
        out[device] = entry
    return out


# ---------------------------------------------------------------- video


def run_ffmpeg(args: list[str], size: int) -> bytes | None:
    """Raw output of ffmpeg; tries the new and the old name of the no-drop option."""
    for option in (["-fps_mode", "passthrough"], ["-vsync", "0"]):
        try:
            done = subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-loglevel",
                    "error",
                    *args[:-1],
                    *option,
                    args[-1],
                ],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                check=False,
                timeout=FFMPEG_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            say(f"ffmpeg did not finish in {FFMPEG_TIMEOUT_S:.0f} s, window skipped")
            return None
        if done.returncode == 0:
            return done.stdout
    return None


def video_timing(video: Path, head: dict, windows=3, seconds=60.0, side=96) -> dict:
    """Lag of the image change against the SLAM rotation speed, in several windows.

    The video frame timestamps are on the SLAM clock. The signal is the mean absolute difference of
    consecutive small grey frames. ``lag_ms`` > 0 means the picture moves later than SLAM says.
    """
    times = frame_times_ns(video)
    if not times:
        return {"error": "no frame times in the video"}
    frame_t = np.array(times, dtype=np.float64) / 1000.0  # us
    rel = (frame_t - frame_t[0]) / 1e6
    s_smooth = smooth(head["omega"], head["t_us"], 0.1)
    results = []
    for number in range(windows):
        index = int(len(rel) * (0.1 + 0.35 * number))
        start = float(rel[index] + 0.5 * np.median(np.diff(rel)))  # between two frames
        data = run_ffmpeg(
            [
                "-ss",
                f"{start:.3f}",
                "-t",
                f"{seconds:.1f}",
                "-i",
                str(video),
                "-vf",
                f"scale={side}:{side},format=gray",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "gray",
                "-",
            ],
            side,
        )
        if not data:
            results.append({"start_s": start, "error": "ffmpeg failed"})
            continue
        frames = np.frombuffer(data, dtype=np.uint8)
        frames = frames[: len(frames) // (side * side) * side * side]
        frames = frames.reshape(-1, side, side).astype(np.float32)
        first = index + 1
        count = min(len(frames), len(rel) - first)
        if count < 300:
            results.append({"start_s": start, "error": "too few frames"})
            continue
        motion = np.abs(np.diff(frames[:count], axis=0)).mean(axis=(1, 2))
        mid = 0.5 * (
            frame_t[first : first + count - 1] + frame_t[first + 1 : first + count]
        )
        motion = smooth(motion, mid, 0.1)
        lags = np.arange(-0.5, 0.5 + 1e-9, 0.005)
        scores = np.array(
            [
                sync.correlation(
                    motion, np.interp(mid - lag * 1e6, head["t_us"], s_smooth)
                )
                for lag in lags
            ]
        )
        i = int(np.nanargmax(scores))
        results.append(
            {
                "start_s": round(start, 1),
                "frames": count,
                "lag_ms": parabolic(lags, scores, i) * 1000.0,
                "corr": float(scores[i]),
                "corr_at_zero": float(scores[len(lags) // 2]),
            }
        )
    good = [r["lag_ms"] for r in results if r.get("corr", 0) >= 0.5]
    return {
        "windows": results,
        "median_lag_ms": float(np.median(good)) if good else None,
        "note": "one frame is 33.3 ms; the first frame after the seek may be off by one",
    }


def write_png(path: Path, rgb: np.ndarray) -> None:
    height, width, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(height))

    def chunk(kind: bytes, body: bytes) -> bytes:
        return (
            struct.pack(">I", len(body))
            + kind
            + body
            + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)
        )

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def cut_frame(video: Path, seconds: float, size: int) -> np.ndarray | None:
    data = run_ffmpeg(
        [
            "-ss",
            f"{seconds:.3f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-vf",
            f"scale={size}:{size}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        size,
    )
    if not data or len(data) < size * size * 3:
        return None
    return (
        np.frombuffer(data[: size * size * 3], dtype=np.uint8)
        .reshape(size, size, 3)
        .copy()
    )


def local_world(
    head: dict,
    xs: dict,
    segment: int,
    offset_us: float,
    t_s: float,
    half_s=30.0,
    head_d=None,
) -> dict | None:
    """World fit on the +-half_s seconds around one frame: follows the slowly wandering heading.

    With ``head_d`` (the head offset of this person) only yaw and translation are free.
    """
    t_all = common_times(head["t_us"], xs["t_us"], offset_us)[::10]
    t = t_all[np.abs(t_all - t_s * 1e6) <= half_s * 1e6]
    if len(t) < 100:
        return None
    pos_s, rot_s = interp_pose(head, t)
    pos_x, rot_x = xsens_at(xs, t - offset_us, segment)
    good = (
        np.isfinite(pos_s).all(axis=1)
        & np.isfinite(pos_x).all(axis=1)
        & np.isfinite(rot_s).all(axis=(1, 2))
        & np.isfinite(rot_x).all(axis=(1, 2))
    )
    if good.sum() < 100:
        return None
    if head_d is not None:
        fit = fit_fixed_d(
            pos_s[good], pos_x[good], rot_x[good], np.array(head_d), step_deg=2.0
        )
        spread = spread_deg(
            np.einsum(
                "mji,jk,mkl->mil",
                rot_s[good],
                rot_z(math.radians(fit["yaw_deg"])),
                rot_x[good],
            )
        )
        fit["orientation_spread_deg_at_yaw"] = spread
        return fit
    return fit_world(pos_s[good], rot_s[good], pos_x[good], rot_x[good], step_deg=2.0)


def draw_square(image: np.ndarray, x: float, y: float, half: int, colour) -> None:
    height, width, _ = image.shape
    x0, x1 = round(x) - half, round(x) + half + 1
    y0, y1 = round(y) - half, round(y) + half + 1
    image[max(0, y0) : max(0, min(height, y1)), max(0, x0) : max(0, min(width, x1))] = (
        colour
    )


def project_frames(
    video: Path,
    camera: dict,
    head: dict,
    xs: dict,
    offset_us: float,
    world: dict,
    times_s: list[tuple[str, float]],
    out_dir: Path,
    tag: str,
    size: int,
    rotation: str,
    head_d=None,
) -> list[dict]:
    """Overlay projected Xsens points on video frames at the given SLAM-clock times [s]."""
    first_s = (frame_times_ns(video) or [0])[0] / 1e9
    records = []
    for label, t_s in times_s:
        t_us = np.array([t_s * 1e6])
        pos_s, rot_s = interp_pose(head, t_us)
        r_wc = rot_s[0] @ camera["r_dev_cam"]
        origin = pos_s[0] + rot_s[0] @ camera["t_dev_cam"]
        image = cut_frame(video, t_s - first_s, size)
        if image is None:
            records.append({"label": label, "time_s": t_s, "error": "no frame"})
            continue
        local = local_world(head, xs, SEGMENTS["head"], offset_us, t_s, head_d=head_d)
        entry: dict = {"label": label, "time_s": t_s, "points": {}}
        if local is not None:
            entry["local_fit"] = {
                "yaw_deg": round(local["yaw_deg"], 1),
                "position_rms_m": round(local["position_rms_m"], 3),
                "orientation_spread_deg": round(
                    local["orientation_spread_deg_at_yaw"], 1
                ),
            }
        fit = local if local is not None else world
        for name, segment in SEGMENTS.items():
            p_x, _ = xsens_at(xs, t_us - offset_us, segment)
            placed = {}
            for kind, w in (("local", fit), ("global", world)):
                p_w = (
                    np.array(w["translation_m"])
                    + p_x[0] @ rot_z(math.radians(w["yaw_deg"])).T
                )
                p_c = (p_w - origin) @ r_wc
                uv, valid = project_fisheye624(p_c[None, :], camera["params"])
                placed[kind] = (p_c, uv[0], bool(valid[0]))
            p_c, uv, ok = placed["local"]
            entry["points"][name] = {"visible": ok, "depth_m": float(p_c[2])}
            if ok:
                u, v = to_image(uv, size, rotation)
                entry["points"][name]["uv_frac"] = [u / size, v / size]
                if placed["global"][2]:
                    ug, vg = to_image(placed["global"][1], size, rotation)
                    entry["points"][name]["global_minus_local_px"] = round(
                        math.hypot(ug - u, vg - v), 1
                    )
                big = name.endswith(("hand", "foot"))
                draw_square(image, u, v, 8 if big else 4, (255, 255, 255))
                draw_square(image, u, v, 6 if big else 3, COLOURS[name])
        file = f"{tag}_{label}_t{int(t_s)}.png"
        write_png(out_dir / file, image)
        entry["file"] = file
        records.append(entry)
    return records


# ---------------------------------------------------------------- per sequence


def scan_segments(slam: dict, xs: dict, hz=10.0) -> list[dict]:
    s0, x0 = slam["t_us"][0], xs["t_us"][0]
    n_s = int((slam["t_us"][-1] - s0) / 1e6 * hz)
    n_x = int((xs["t_us"][-1] - x0) / 1e6 * hz)
    s_w = grid(slam["t_us"], slam["omega"], s0, n_s, hz, 0.2)
    needed = int(min(n_s, n_x) * 0.6)
    rows = []
    for segment in range(xs["omega"].shape[1]):
        x_w = grid(xs["t_us"], xs["omega"][:, segment], x0, n_x, hz, 0.2)
        scan = [
            (k, c) for k, c in sync.offset_search(s_w, x_w, needed) if not np.isnan(c)
        ]
        if scan:
            k, c = max(scan, key=lambda kc: kc[1])
            rows.append(
                {
                    "segment": segment,
                    "corr": c,
                    "offset_s": ((s0 - x0) + k * 1e6 / hz) / 1e6,
                }
            )
    return sorted(rows, key=lambda r: -r["corr"])


def say(message: str) -> None:
    """Progress line with the wall clock, flushed so a stuck step is visible."""
    print(f"  [{time.strftime('%H:%M:%S')}] {message}", flush=True)


def analyse(sequence: str, folder: Path, coarse: dict, steps: set, args) -> dict:
    body, head_zip = (
        sync.find(folder, "body_motion.zip"),
        sync.find(folder, "recording_head.zip"),
    )
    videos = sorted(folder.glob("*_preview_rgb.mp4"))
    if body is None or head_zip is None:
        return {"error": "body_motion.zip or recording_head.zip missing"}
    say("loading Xsens")
    xs = load_xsens(body)
    say("loading head SLAM trajectory")
    head = load_trajectory(head_zip, "recording_head")
    result: dict = {}
    if "scan" in steps:
        result["scan_top5"] = scan_segments(head, xs)[:5]
    reliable = coarse.get("corr_angular_speed", 0) >= MIN_RELIABLE_CORR
    if not reliable:
        result["note"] = (
            f"coarse offset not reliable (corr {coarse.get('corr_angular_speed')}); "
            "fit, drift, wrist and project are skipped"
        )
        if "video" in steps and videos:
            result["video"] = video_timing(videos[0], head)
        return result
    segment = SEGMENTS["head"]
    offset_us = coarse["slam_minus_xsens_s"] * 1e6
    say("refining the offset")
    result["refine"] = refine_offset(head, xs, segment, offset_us)
    if "offset_s" in result["refine"]:
        offset_us = result["refine"]["offset_s"] * 1e6
    world = None
    if steps & {"fit", "project", "labels"}:
        say("fitting the world transform")
        t = common_times(head["t_us"], xs["t_us"], offset_us)[::20]  # 5 Hz
        pos_s, rot_s = interp_pose(head, t)
        pos_x, rot_x = xsens_at(xs, t - offset_us, segment)
        good = (
            np.isfinite(pos_s).all(axis=1)
            & np.isfinite(pos_x).all(axis=1)
            & np.isfinite(rot_s).all(axis=(1, 2))
            & np.isfinite(rot_x).all(axis=(1, 2))
        )
        result["non_finite_samples"] = int((~good).sum())
        world = fit_world(pos_s[good], rot_s[good], pos_x[good], rot_x[good])
        world["samples"] = int(good.sum())
        result["world"] = world
    if "drift" in steps:
        say("drift windows")
        result["drift"] = drift(head, xs, segment, offset_us)
    if "yaw" in steps:
        say("yaw windows")
        result["yaw_windows"] = window_fits(head, xs, segment, offset_us)
    if "wrist" in steps:
        say("wrist devices")
        result["wrist"] = wrist_check(folder, xs, head, offset_us / 1e6)
    if "video" in steps and videos:
        say("video timing")
        result["video"] = video_timing(videos[0], head)
    shared = None
    if steps & {"project", "labels"} and world is not None:
        say("shared head offset")
        shared = shared_head_offset(head, xs, segment, offset_us)
        result["shared_head_offset"] = (
            shared if shared else {"error": "too few good windows"}
        )
    head_d = shared["d_m"] if shared else None
    if "project" in steps and videos and world is not None:
        camera = load_camera(head_zip)
        result["camera"] = {
            "model": camera["model"],
            "f_calib_px": float(camera["params"][0]),
        }
        picks = [
            (f["label"], f["narration_time_s"])
            for item in args.frames_index
            if item["sequence"] == sequence
            for f in item["frames"]
        ]
        lo, hi = (
            max(head["t_us"][0], xs["t_us"][0] + offset_us) / 1e6,
            min(head["t_us"][-1], xs["t_us"][-1] + offset_us) / 1e6,
        )
        picks += [
            ("uniform", lo + (hi - lo) * (i + 0.5) / args.uniform)
            for i in range(args.uniform)
        ]
        args.out.mkdir(parents=True, exist_ok=True)
        parts = sequence.split("_")
        tag = "_".join(parts[:3]) if len(parts) > 2 else sequence
        say("overlays")
        result["project"] = project_frames(
            videos[0],
            camera,
            head,
            xs,
            offset_us,
            world,
            picks,
            args.out,
            tag,
            args.size,
            args.rotation,
            head_d,
        )
    if "labels" in steps and videos and world is not None:
        say("frames for hand labelling")
        narrated = [
            (f["label"], f["narration_time_s"])
            for item in args.frames_index
            if item["sequence"] == sequence
            for f in item["frames"]
        ]
        lo, hi = (
            max(head["t_us"][0], xs["t_us"][0] + offset_us) / 1e6,
            min(head["t_us"][-1], xs["t_us"][-1] + offset_us) / 1e6,
        )
        times = pick_label_times(
            head, xs, segment, offset_us, head_d, narrated, lo, hi, args.label_frames
        )
        parts = sequence.split("_")
        tag = "_".join(parts[:3]) if len(parts) > 2 else sequence
        result["label_frames"] = write_label_frames(
            videos[0],
            times,
            args.out / "labels",
            tag,
            sequence,
            args.label_size,
        )
    return result


def pick_label_times(
    head, xs, segment, offset_us, head_d, narrated, lo, hi, total
) -> list[tuple[str, float]]:
    """Frames to label: floor poses from the narration plus evenly spread ones, all with a good fit."""

    def usable(t_s: float) -> bool:
        fit = local_world(head, xs, segment, offset_us, t_s, head_d=head_d)
        return (
            fit is not None
            and fit["position_rms_m"] <= 0.10
            and fit["orientation_spread_deg_at_yaw"] <= 10.0
        )

    chosen: list[tuple[str, float]] = []
    seen = set()
    for label, t_s in narrated:
        if label in seen or len(chosen) >= total // 2 or not lo <= t_s <= hi:
            continue
        if usable(t_s):
            chosen.append((label, t_s))
            seen.add(label)
    need = total - len(chosen)
    for k in range(need):
        centre = lo + (hi - lo) * (k + 0.5) / need
        for shift in [0.0] + [s * sign for s in range(15, 200, 15) for sign in (1, -1)]:
            t_s = centre + shift
            if (
                lo <= t_s <= hi
                and all(abs(t_s - other) > 10.0 for _, other in chosen)
                and usable(t_s)
            ):
                chosen.append(("uniform", t_s))
                break
    return sorted(chosen, key=lambda item: item[1])


def write_label_frames(
    video: Path, times: list[tuple[str, float]], out_dir: Path, tag, sequence, size
) -> list[dict]:
    """Raw upright frames (no overlay) for hand labelling, one PNG each."""
    out_dir.mkdir(parents=True, exist_ok=True)
    first_s = (frame_times_ns(video) or [0])[0] / 1e9
    entries = []
    for label, t_s in times:
        image = cut_frame(video, t_s - first_s, size)
        if image is None:
            continue
        name = f"{tag}_{label}_t{int(t_s)}"
        write_png(out_dir / f"{name}.png", image)
        entries.append(
            {
                "id": name,
                "sequence": sequence,
                "label": label,
                "time_s": t_s,
                "file": f"{name}.png",
                "size": size,
            }
        )
    return entries


def write_label_manifest(report: dict, out_dir: Path) -> int:
    """label_frames.json and label_frames.js for all sequences of this run, plus the page."""
    frames = [f for entry in report.values() for f in entry.get("label_frames", [])]
    if not frames:
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "label_frames.json").write_text(
        json.dumps(frames, indent=1), encoding="utf-8"
    )
    (out_dir / "label_frames.js").write_text(
        "window.LABEL_FRAMES = " + json.dumps(frames, indent=1) + ";\n",
        encoding="utf-8",
    )
    page = Path(__file__).with_name("nymeria_label_tool.html")
    if page.exists():
        (out_dir / "index.html").write_text(
            page.read_text(encoding="utf-8"), encoding="utf-8"
        )
    return len(frames)


def offset_of(record: dict) -> dict:
    """The offset block of the sync report, or an empty dict when there was no usable overlap."""
    block = record.get("offset")
    return block if isinstance(block, dict) else {}


def write_merged(target: Path, report: dict) -> None:
    """Write the report, keeping the sequences of earlier runs that this run did not touch.

    A sequence that was run again gets the new keys on top of the old ones (so a run with only
    ``--steps scan`` adds to a full run); an old ``error`` is dropped when the new run has none.
    """
    merged = {}
    if target.exists():
        try:
            old = json.loads(target.read_text(encoding="utf-8"))
            merged = old if isinstance(old, dict) else {}
        except (OSError, ValueError):
            merged = {}
    for sequence, new in report.items():
        old_entry = merged.get(sequence)
        if isinstance(old_entry, dict) and isinstance(new, dict) and "error" not in new:
            old_entry.pop("error", None)
            old_run = old_entry.get("run")
            old_entry.update(new)
            if isinstance(old_run, dict) and old_run != new.get("run"):
                # keys of an earlier run stay next to the new ones: the audit flags this
                old_entry.setdefault("earlier_runs", []).append(old_run.get("run_id"))
            elif not isinstance(old_run, dict) and new.get("run"):
                old_entry.setdefault("earlier_runs", []).append("unrecorded")
        else:
            merged[sequence] = new
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    temporary.replace(target)


TIMING_MIN_CORR = 0.95  # refined head-to-Xsens correlation needed for any timing claim


def provenance(args, steps: set, run_id: str) -> dict:
    """What produced an entry: run id, a hash of the code, the steps and the settings that matter."""
    here = Path(__file__).resolve().parent
    return {
        "run_id": run_id,
        "code_sha256_16": {
            name: hashlib.sha256((here / name).read_bytes()).hexdigest()[:16]
            for name in ("nymeria_project_check.py", "nymeria_motion_sync.py")
        },
        "steps": sorted(steps),
        "uniform": args.uniform,
        "label_frames": args.label_frames,
        "rotation": args.rotation,
        "numpy": np.__version__,
    }


def archive_previous(out: Path, run_id: str) -> int:
    """Move the report and the overlay images of earlier runs to out/_previous/<run_id>.

    Nothing is deleted, and the ``labels`` folder is left alone. The new run starts from an empty
    report, so no key of an older run can sit next to a newer one.
    """
    if not out.exists():
        return 0
    items = [
        p
        for p in out.iterdir()
        if p.is_file()
        and (p.suffix == ".png" or p.name == "nymeria_project_check.json")
    ]
    if not items:
        return 0
    store = out / "_previous" / run_id
    store.mkdir(parents=True)
    for item in items:
        shutil.move(str(item), str(store / item.name))
    return len(items)


def audit_report(report: dict) -> tuple[list[dict], list[str]]:
    """One row per sequence (what it may be used for and why not) and a list of uniformity problems."""
    rows = []
    for sequence, entry in sorted(report.items()):
        if not isinstance(entry, dict):
            continue
        run = entry.get("run") if isinstance(entry.get("run"), dict) else {}
        refine = entry.get("refine", {})
        shared = entry.get("shared_head_offset", {})
        corr = refine.get("corr")
        timing_ok = corr is not None and corr >= TIMING_MIN_CORR
        spatial_ok = timing_ok and "d_m" in shared
        why = []
        if "refine" not in entry:
            why.append(entry.get("note") or entry.get("error") or "no offset")
        elif not timing_ok:
            why.append(f"refined correlation {corr:.3f} below {TIMING_MIN_CORR}")
        elif not spatial_ok:
            why.append("no shared head offset: " + str(shared.get("error", "not run")))
        rows.append(
            {
                "sequence": sequence,
                "run_id": run.get("run_id"),
                "code": run.get("code_sha256_16", {}).get("nymeria_project_check.py"),
                "steps": run.get("steps"),
                "earlier_runs": entry.get("earlier_runs", []),
                "refined_corr": corr,
                "timing_ok": timing_ok,
                "spatial_ok": spatial_ok,
                "windows": (
                    f"{shared['windows_used']}/{shared['windows_total']}"
                    if "windows_used" in shared
                    else None
                ),
                "excluded_because": why,
            }
        )
    problems = []
    for key in ("run_id", "code"):
        values = {r[key] for r in rows}
        if len(values) > 1 or None in values:
            problems.append(
                f"{key} differs or is missing across sequences: {sorted(map(str, values))}"
            )
    steps = {json.dumps(r["steps"]) for r in rows}
    if len(steps) > 1:
        problems.append(f"different step sets across sequences: {sorted(steps)}")
    for r in rows:
        if r["earlier_runs"]:
            problems.append(
                f"{r['sequence']}: keys of earlier runs {r['earlier_runs']} are mixed in"
            )
    return rows, problems


def print_audit(report: dict) -> None:
    rows, problems = audit_report(report)
    for r in rows:
        use = (
            "timing+spatial"
            if r["spatial_ok"]
            else "timing only"
            if r["timing_ok"]
            else "excluded"
        )
        corr = f"{r['refined_corr']:.3f}" if r["refined_corr"] is not None else "-"
        print(
            f"{r['sequence'][:40]:40s} corr {corr:>6s}  windows {r['windows']!s:>5s}  {use:15s} {'; '.join(r['excluded_because'])}"
        )
    print("UNIFORM" if not problems else "NOT UNIFORM:")
    for problem in problems:
        print("  -", problem)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, default=sync.DEFAULT_ROOT)
    parser.add_argument("--sync-json", type=Path, default=sync.DEFAULT_OUT)
    parser.add_argument("--out", type=Path, default=PROJECTION_OUT)
    parser.add_argument("--frames-index", type=Path, default=FRAMES_INDEX)
    parser.add_argument("--sequence", action="append")
    parser.add_argument("--steps", default="fit,drift,yaw,wrist,video,project")
    parser.add_argument(
        "--uniform", type=int, default=4, help="extra evenly spaced frames"
    )
    parser.add_argument("--size", type=int, default=640, help="overlay image size")
    parser.add_argument(
        "--label-frames", type=int, default=8, help="frames to label per sequence"
    )
    parser.add_argument(
        "--label-size", type=int, default=800, help="size of the frames to label"
    )
    parser.add_argument(
        "--rotation",
        choices=("cw90", "ccw90", "none"),
        default="cw90",
        help="how the calibration frame is turned to match the upright video",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="start from an empty report: move the old report and overlay images to _previous/",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="only check the existing report: one run, one code version, who can be used for what",
    )
    args = parser.parse_args(argv)
    if args.audit:
        target = args.out / "nymeria_project_check.json"
        print_audit(json.loads(target.read_text(encoding="utf-8")))
        return 0
    if args.fresh and args.sequence:
        parser.error(
            "--fresh replaces the whole report, so it cannot be used with --sequence"
        )
    steps = set(args.steps.split(","))
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    if args.fresh:
        moved = archive_previous(args.out, run_id)
        print(
            f"{moved} files of earlier runs moved to {args.out / '_previous' / run_id}"
        )
    coarse = json.loads(args.sync_json.read_text(encoding="utf-8"))
    args.frames_index = (
        json.loads(args.frames_index.read_text(encoding="utf-8"))
        if args.frames_index.exists()
        else []
    )
    report = {}
    for sequence in args.sequence or sync.SEQUENCES:
        print(f"{sequence} ...", flush=True)
        try:
            report[sequence] = analyse(
                sequence,
                args.root / sequence,
                offset_of(coarse.get(sequence, {})),
                steps,
                args,
            )
        except (OSError, KeyError, ValueError, zipfile.BadZipFile) as error:
            report[sequence] = {"error": f"{type(error).__name__}: {error}"}
        report[sequence]["run"] = provenance(args, steps, run_id)
        summary = {
            k: v
            for k, v in report[sequence].items()
            if k not in ("project", "label_frames", "run")
        }
        print(json.dumps(summary, indent=1))
        if "project" in report[sequence]:
            seen = [r for r in report[sequence]["project"] if "points" in r]
            print(f"  {len(seen)} overlay frames written to {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "nymeria_project_check.json"
    write_merged(target, report)
    print(f"written to {target}")
    print_audit(json.loads(target.read_text(encoding="utf-8")))
    count = write_label_manifest(report, args.out / "labels")
    if count:
        print(f"{count} frames to label: open {args.out / 'labels' / 'index.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
