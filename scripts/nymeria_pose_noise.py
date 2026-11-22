"""Error of a head pose estimated from the Nymeria preview video (protocol: docs/nymeria_pose_noise_protocol.md).

No foot-lock label is read. For the 7 sequences that have the preview video, 40 windows of 64 consecutive
frames are cut, the head pose of each window is estimated by two feed-forward models (VGGT-Omega on every
window, MonST3R on every fourth) and compared with the head SLAM trajectory after a per-window alignment.
The result is the error in the units of the simulated noise of the detector tests (centimetres, degrees,
step error, correlation in time, scale).

Steps (``--step``):
  prepare   cut the windows: upright pinhole views (512 x 512, 90 degrees) of the fisheye frames as PNG
            files, and the SLAM reference of every window. Needs numpy and ffmpeg only.
  estimate  run one model (``--model omega`` or ``--model monst3r``) on the prepared windows. Run each in the
            Python environment in which that model works. Resumable: finished windows are skipped.
  report    alignment, statistics, the reading of the protocol; written once to
            ``review/nymeria_pose_noise.json``.

Usage (Windows):
    python scripts/nymeria_pose_noise.py --step prepare --limit 1 --sequence 20230706_s0_breanna_reese_act0_cfg468
    python scripts/nymeria_pose_noise.py --step estimate --model omega --limit 1 --sequence 20230706_s0_breanna_reese_act0_cfg468
    python scripts/nymeria_pose_noise.py --step prepare
    python scripts/nymeria_pose_noise.py --step estimate --model omega
    python scripts/nymeria_pose_noise.py --step estimate --model monst3r      (in the MonST3R environment)
    python scripts/nymeria_pose_noise.py --step report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

import nymeria_motion_sync as sync
import nymeria_project_check as pc
from mp4_probe import frame_times_ns

DATA = Path(r"D:\projects\hready_data\datasets\nymeria")
REVIEW = Path(r"D:\projects\hready_data\review")
WORK = REVIEW / "nymeria_pose_noise"
RESULT = REVIEW / "nymeria_pose_noise.json"
LEARNED_PLAN = REVIEW / "nymeria_learned_plan.json"
MODELS = Path(r"C:\Users\kghol072\physical_fidelity\models")

WINDOWS = 40  # windows per sequence, evenly spaced (protocol 4.1)
LENGTH = 64  # frames per window (the length of the detector's input)
MARGIN_S = 1.0  # seconds kept free at both ends of the overlap of video and SLAM
MONST3R_EVERY = 4  # MonST3R runs on every fourth window
VIEW = 512  # side of the pinhole view in pixels (the input size of both models)
VIEW_FOV_DEG = 90.0  # horizontal field of view of the pinhole view
VIDEO_SIZE = 1408  # side of the preview video
ROTATION = "cw90"  # sensor to upright video, as checked on the overlays in nymeria_project_check.py
MOVING_PATH_M = 0.15  # a window is "moving" if the camera path in it is at least this long
GUARD_DEG = 10.0  # median rotation error after alignment above this: the sequence is reported as failed
EQUIVALENT_SIGMA_LIMIT_M = 0.02  # protocol 6
CORRELATION_TIME_FRAMES = 30.0  # the correlated control of the noise protocol: coefficient exp(-1/30)
LAGS = (1, 15, 30)  # frames
FFMPEG_TIMEOUT_S = 900.0
SEED = 20261121

# columns: the upright camera axes written in the sensor camera frame (x_up = -y_sensor, y_up = x_sensor)
R_SENSOR_FROM_UP = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def say(message: str) -> None:
    print(message, flush=True)


# ---------------------------------------------------------------- pinhole view of the fisheye frame


def view_maps(params: np.ndarray, size: int = VIEW, fov_deg: float = VIEW_FOV_DEG):
    """For every pixel of the upright pinhole view: the video pixel to read (float x, y) and a validity mask.

    The view is a pinhole camera in the upright camera frame (x right, y down, z forward), focal length from
    the horizontal field of view. A ray of the view is moved to the sensor frame (``R_SENSOR_FROM_UP``),
    projected with the fisheye model into calibration pixels and mapped to the video frame with the same
    rotation as ``pc.to_image``. No inverse of the fisheye model is needed.
    """
    focal = 0.5 * size / math.tan(math.radians(fov_deg) / 2.0)
    j, i = np.meshgrid(np.arange(size), np.arange(size))
    x = (j + 0.5 - size / 2.0) / focal
    y = (i + 0.5 - size / 2.0) / focal
    rays_up = np.stack([x.ravel(), y.ravel(), np.ones(x.size)], axis=1)
    rays_sensor = rays_up @ R_SENSOR_FROM_UP.T
    uv, valid = pc.project_fisheye624(rays_sensor, params)
    u = uv[:, 0] * VIDEO_SIZE / pc.CALIB_SIZE
    v = uv[:, 1] * VIDEO_SIZE / pc.CALIB_SIZE
    if ROTATION == "cw90":
        xv, yv = VIDEO_SIZE - v, u
    elif ROTATION == "ccw90":
        xv, yv = v, VIDEO_SIZE - u
    else:
        xv, yv = u, v
    valid &= (xv >= 0) & (xv <= VIDEO_SIZE - 1) & (yv >= 0) & (yv <= VIDEO_SIZE - 1)
    return xv.reshape(size, size), yv.reshape(size, size), valid.reshape(size, size)


def prepare_sampler(xv: np.ndarray, yv: np.ndarray, valid: np.ndarray):
    """Bilinear sampling indices and weights for ``sample``."""
    xc = np.clip(xv, 0, VIDEO_SIZE - 1.001)
    yc = np.clip(yv, 0, VIDEO_SIZE - 1.001)
    x0 = np.floor(xc).astype(np.int32)
    y0 = np.floor(yc).astype(np.int32)
    wx = (xc - x0).astype(np.float32)[..., None]
    wy = (yc - y0).astype(np.float32)[..., None]
    return x0, y0, wx, wy, valid


def sample(frame: np.ndarray, sampler) -> np.ndarray:
    """One upright pinhole view [VIEW, VIEW, 3] uint8 from a video frame [VIDEO_SIZE, VIDEO_SIZE, 3]."""
    x0, y0, wx, wy, valid = sampler
    f = frame.astype(np.float32)
    top = f[y0, x0] * (1 - wx) + f[y0, x0 + 1] * wx
    bottom = f[y0 + 1, x0] * (1 - wx) + f[y0 + 1, x0 + 1] * wx
    out = top * (1 - wy) + bottom * wy
    out[~valid] = 0
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- windows and the SLAM reference


def plan_windows(frame_t_us: np.ndarray, slam_t_us: np.ndarray) -> list[int]:
    """First video-frame index of each window: evenly spaced where video and SLAM overlap (margin kept)."""
    lo = max(frame_t_us[0], slam_t_us[0]) + MARGIN_S * 1e6
    hi = min(frame_t_us[-1], slam_t_us[-1]) - MARGIN_S * 1e6
    ok = np.where((frame_t_us >= lo) & (frame_t_us <= hi))[0]
    if len(ok) < LENGTH + WINDOWS:
        raise ValueError("video and SLAM overlap too little for the windows")
    first, last = int(ok[0]), int(ok[-1]) - LENGTH + 1
    return [int(k) for k in np.round(np.linspace(first, last, WINDOWS)).astype(int)]


def build_reference(head: dict, camera: dict, t_us: np.ndarray) -> dict:
    """SLAM camera centres and upright-camera rotations (world from camera) at the given times."""
    pos, rot = pc.interp_pose(head, t_us)
    centre = pos + np.einsum("kij,j->ki", rot, camera["t_dev_cam"])
    r_up = rot @ camera["r_dev_cam"] @ R_SENSOR_FROM_UP
    path = float(np.linalg.norm(np.diff(centre, axis=0), axis=1).sum())
    return {"t_us": t_us, "centre": centre, "r_up": r_up, "path_m": path}


def frame_lag(gray: np.ndarray, t_us: np.ndarray, head: dict) -> dict:
    """Diagnostic only: shift in frames (-2..2) at which the image change best follows the SLAM rotation speed."""
    motion = np.abs(np.diff(gray, axis=0)).mean(axis=(1, 2))
    mid = 0.5 * (t_us[1:] + t_us[:-1])
    dt = float(np.median(np.diff(t_us)))
    best, scores = 0, {}
    for s in range(-2, 3):
        omega = np.interp(mid + s * dt, head["t_us"], head["omega"])
        scores[s] = float(np.corrcoef(motion, omega)[0, 1]) if motion.std() > 0 else float("nan")
    valid = {s: c for s, c in scores.items() if np.isfinite(c)}
    if valid:
        best = max(valid, key=valid.get)
    return {"best_shift_frames": int(best), "corr_best": valid.get(best, float("nan")), "corr_zero": scores[0]}


def read_frames(video: Path, start_s: float, count: int) -> np.ndarray | None:
    """``count`` frames from ``start_s`` as uint8 [count, VIDEO_SIZE, VIDEO_SIZE, 3] (no frame dropped)."""
    data = pc.run_ffmpeg(
        ["-ss", f"{start_s:.4f}", "-i", str(video), "-frames:v", str(count), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        VIDEO_SIZE,
    )
    size = VIDEO_SIZE * VIDEO_SIZE * 3
    if not data or len(data) < count * size:
        return None
    return np.frombuffer(data[: count * size], dtype=np.uint8).reshape(count, VIDEO_SIZE, VIDEO_SIZE, 3)


def write_image(path: Path, rgb: np.ndarray) -> None:
    try:
        from PIL import Image

        Image.fromarray(rgb).save(path)
    except ImportError:
        pc.write_png(path, rgb)


def window_dir(seq: str, w: int) -> Path:
    return WORK / "frames" / seq / f"w{w:02d}"


def sequences_with_video() -> list[str]:
    return sorted(p.name for p in DATA.iterdir() if p.is_dir() and any(p.glob("*_preview_rgb.mp4")))


def groups_of(names: list[str]) -> dict[str, str]:
    plan = json.loads(LEARNED_PLAN.read_text())
    test, pool = set(plan["test"]), set(plan["pool"])
    return {n: "test" if n in test else "pool" if n in pool else "other" for n in names}


def run_prepare(args) -> None:
    names = [args.sequence] if args.sequence else sequences_with_video()
    (WORK / "ref").mkdir(parents=True, exist_ok=True)
    plan_path = WORK / "plan.json"
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else {"sequences": {}}
    group = groups_of(sequences_with_video())
    for seq in names:
        folder = DATA / seq
        video = sorted(folder.glob("*_preview_rgb.mp4"))[0]
        head_zip = sync.find(folder, "recording_head.zip")
        say(f"{seq}: loading SLAM trajectory and camera")
        head = pc.load_trajectory(head_zip, "recording_head")
        camera = pc.load_camera(head_zip)
        times = frame_times_ns(video)
        if not times:
            raise SystemExit(f"{seq}: no frame times in the video")
        frame_t = np.array(times, dtype=np.float64) / 1000.0
        starts_all = plan_windows(frame_t, head["t_us"])
        starts = starts_all[: args.limit] if args.limit else starts_all
        dt = float(np.median(np.diff(frame_t)))
        sampler = prepare_sampler(*view_maps(camera["params"]))
        entry = {"group": group[seq], "video": video.name, "starts": starts_all, "windows": {}}
        for w, k0 in enumerate(starts):
            ref_path = WORK / "ref" / f"{seq}_w{w:02d}.npz"
            wdir = window_dir(seq, w)
            if ref_path.exists() and len(list(wdir.glob("f*.png"))) == LENGTH:
                with np.load(ref_path) as done:
                    entry["windows"][str(w)] = {"k0": k0, "path_m": float(done["path_m"]), "lag": json.loads(str(done["lag"]))}
                continue
            t_us = frame_t[k0 : k0 + LENGTH]
            start_s = (frame_t[k0] - frame_t[0]) / 1e6 - 0.5 * dt / 1e6  # first output frame is frame k0
            frames = read_frames(video, start_s, LENGTH)
            if frames is None:
                say(f"  window {w}: ffmpeg gave too few frames, skipped")
                continue
            wdir.mkdir(parents=True, exist_ok=True)
            views = []
            for k in range(LENGTH):
                view = sample(frames[k], sampler)
                write_image(wdir / f"f{k:03d}.png", view)
                views.append(view)
            gray = np.stack(views).astype(np.float32).mean(axis=3).reshape(LENGTH, 64, 8, 64, 8).mean(axis=(2, 4))
            ref = build_reference(head, camera, t_us)
            lag = frame_lag(gray, t_us, head)
            np.savez(ref_path, **ref, lag=json.dumps(lag))
            entry["windows"][str(w)] = {"k0": k0, "path_m": ref["path_m"], "lag": lag}
            say(f"  window {w:2d} (frame {k0}): path {ref['path_m'] * 100:5.1f} cm, lag shift {lag['best_shift_frames']} (corr {lag['corr_best']:.2f})")
        entry["windows"] = {**plan["sequences"].get(seq, {}).get("windows", {}), **entry["windows"]}
        plan["sequences"][seq] = entry
        plan_path.write_text(json.dumps(plan, indent=1))
    say(f"plan written to {plan_path}")


# ---------------------------------------------------------------- estimators


def todo_windows(args, every: int = 1):
    plan = json.loads((WORK / "plan.json").read_text())
    for seq, entry in plan["sequences"].items():
        if args.sequence and seq != args.sequence:
            continue
        for w in sorted(int(x) for x in entry["windows"]):
            if w % every:
                continue
            if args.limit and w >= args.limit:
                continue
            yield seq, w


def save_estimate(model: str, seq: str, w: int, c2w: np.ndarray, seconds: float, extra: dict) -> None:
    out = WORK / f"est_{model}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{seq}_w{w:02d}.npz", r_c2w=c2w[:, :3, :3], centre=c2w[:, :3, 3], seconds=seconds, info=json.dumps(extra))


def extrinsics_to_c2w(ext: np.ndarray) -> np.ndarray:
    """Camera-to-world matrices [S, 4, 4] from camera-from-world extrinsics [S, 3, 4] (OpenCV axes)."""
    r_cw, t_cw = ext[:, :, :3], ext[:, :, 3]
    r_c2w = r_cw.transpose(0, 2, 1)
    c2w = np.tile(np.eye(4), (len(ext), 1, 1))
    c2w[:, :3, :3] = r_c2w
    c2w[:, :3, 3] = -np.einsum("kij,kj->ki", r_c2w, t_cw)
    return c2w


def load_views(paths: list[str]):
    """The views as one float tensor [S, 3, VIEW, VIEW] in 0..1.

    This is what ``vggt_omega.utils.load_fn.load_and_preprocess_images`` returns for 512 x 512 images
    (no resize, no crop, no padding: the target shape of a square image at resolution 512 is 512 x 512);
    it is written out here because that module imports torchvision, which the environment may not have.
    """
    import torch
    from PIL import Image

    arr = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in paths])
    return torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0


def run_omega(args) -> None:
    import torch

    sys.path.insert(0, str(args.omega_dir))
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.pose_enc import encoding_to_camera

    device = torch.device(args.device)
    model = VGGTOmega().to(device).eval()
    model.load_state_dict(torch.load(args.omega_checkpoint, map_location="cpu"))
    for seq, w in todo_windows(args):
        target = WORK / "est_omega" / f"{seq}_w{w:02d}.npz"
        if target.exists():
            continue
        paths = sorted(str(p) for p in window_dir(seq, w).glob("f*.png"))
        if len(paths) != LENGTH:
            say(f"{seq} w{w}: {len(paths)} frames, skipped")
            continue
        t0 = time.time()
        images = load_views(paths).to(device)
        with torch.inference_mode():
            predictions = model(images)
            extrinsics, _ = encoding_to_camera(predictions["pose_enc"], predictions["images"].shape[-2:])
        c2w = extrinsics_to_c2w(extrinsics[0].float().cpu().numpy())
        save_estimate("omega", seq, w, c2w, time.time() - t0, {"checkpoint": Path(args.omega_checkpoint).name, "dtype": "float32"})
        say(f"{seq} w{w:02d}: {time.time() - t0:.1f} s")


def run_monst3r(args) -> None:
    import os

    import torch

    sys.path.insert(0, str(args.monst3r_dir))
    os.chdir(args.monst3r_dir)
    from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference
    from dust3r.model import AsymmetricCroCo3DStereo
    from dust3r.utils.image import load_images

    device = args.device
    model = AsymmetricCroCo3DStereo.from_pretrained(str(args.monst3r_weights)).to(device)
    model.eval()
    for seq, w in todo_windows(args, every=MONST3R_EVERY):
        target = WORK / "est_monst3r" / f"{seq}_w{w:02d}.npz"
        if target.exists():
            continue
        paths = sorted(str(p) for p in window_dir(seq, w).glob("f*.png"))
        if len(paths) != LENGTH:
            say(f"{seq} w{w}: {len(paths)} frames, skipped")
            continue
        t0 = time.time()
        # the settings of the command line of demo.py of MonST3R (no setting is changed)
        imgs = load_images(paths, size=512, verbose=False)
        pairs = make_pairs(imgs, scene_graph="swinstride-5-noncyclic", prefilter=None, symmetrize=True)
        output = inference(pairs, model, device, batch_size=16, verbose=False)
        scene = global_aligner(
            output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer, verbose=True, shared_focal=True,  # verbose only prints progress
            temporal_smoothing_weight=0.01, translation_weight=1.0, flow_loss_weight=0.01, flow_loss_start_epoch=0.1,
            flow_loss_thre=25, use_self_mask=True, num_total_iter=300, empty_cache=len(paths) > 72, batchify=True,
            window_wise=False, window_size=100, window_overlap_ratio=0.5, prev_video_results=None,
        )
        scene.compute_global_alignment(init="mst", niter=300, schedule="linear", lr=0.01)
        c2w = scene.get_im_poses().detach().float().cpu().numpy()  # camera to world
        save_estimate("monst3r", seq, w, c2w, time.time() - t0, {"weights": str(args.monst3r_weights)})
        say(f"{seq} w{w:02d}: {time.time() - t0:.1f} s")
        del scene, output, pairs
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- alignment and statistics


def chordal_mean(rots: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(rots.sum(axis=0))
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def window_errors(ref_c, ref_r, est_c, est_r, scale: float | None = None) -> dict:
    """Errors of one window after the alignment (amendment A1 of the protocol).

    The rotation ``q`` is the chordal mean of the per-frame rotation between the SLAM and the estimated
    orientations (stable also when the head hardly moves); the translation matches the window means; the
    scale is the least-squares one (``scale=None``) or given. Positions are the camera centres.
    """
    q = chordal_mean(np.einsum("kij,klj->kil", ref_r, est_r))
    ce = est_c - est_c.mean(axis=0)
    cr = ref_c - ref_c.mean(axis=0)
    qe = ce @ q.T
    s_free = float((cr * qe).sum() / max((qe * qe).sum(), 1e-18))
    s = s_free if scale is None else float(scale)
    e = s * qe - cr  # fitted minus SLAM, metres (the window means of both are removed)
    err = np.einsum("kij,klj->kil", ref_r, np.einsum("ij,kjl->kil", q, est_r))
    yaw = np.degrees(np.arctan2(err[:, 1, 0], err[:, 0, 0]))
    tilt = np.degrees(np.arccos(np.clip(err[:, 2, 2], -1, 1)))
    theta = np.degrees(np.arccos(np.clip((np.trace(err, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    return {"e": e, "yaw": yaw, "tilt": tilt, "theta": theta, "s_free": s_free}


def autocorrelation(errors: list[np.ndarray]) -> dict:
    out = {}
    for lag in LAGS:
        num = d0 = d1 = 0.0
        for e in errors:
            a, b = e[:-lag], e[lag:]
            num += float((a * b).sum())
            d0 += float((a * a).sum())
            d1 += float((b * b).sum())
        out[str(lag)] = num / math.sqrt(d0 * d1) if d0 > 0 and d1 > 0 else float("nan")
    return out


def pooled_stats(results: list[dict]) -> dict:
    """Statistics over a list of ``window_errors`` results (metres and degrees converted to cm and degrees)."""
    if not results:
        return {"windows": 0}
    e_all = [r["e"] for r in results]
    n = sum(len(e) for e in e_all)
    pos2 = sum(float((e**2).sum()) for e in e_all)
    vert2 = sum(float((e[:, 2] ** 2).sum()) for e in e_all)
    step2 = sum(float(((e[1:] - e[:-1]) ** 2).sum()) for e in e_all)
    m = sum(len(e) - 1 for e in e_all)
    per_window = np.array([math.sqrt(float(((e[1:] - e[:-1]) ** 2).sum()) / (6 * (len(e) - 1))) for e in e_all])
    yaw = np.concatenate([r["yaw"] for r in results])
    tilt = np.concatenate([r["tilt"] for r in results])
    theta = np.array([float(np.median(r["theta"])) for r in results])
    return {
        "windows": len(results),
        "position_rms_cm": 100 * math.sqrt(pos2 / n),
        "vertical_rms_cm": 100 * math.sqrt(vert2 / n),
        "step_error_rms_cm": 100 * math.sqrt(step2 / m),
        "equivalent_sigma_cm": 100 * math.sqrt(step2 / (6 * m)),
        "equivalent_sigma_window_quartiles_cm": [100 * float(x) for x in np.percentile(per_window, [25, 50, 75])],
        "yaw_std_deg": float(yaw.std()),
        "tilt_rms_deg": float(math.sqrt((tilt**2).mean())),
        "rotation_error_median_over_windows_deg": float(np.median(theta)),
        "autocorrelation": autocorrelation(e_all),
    }


def reference_autocorrelation(refs: list[dict]) -> dict:
    """Autocorrelation the same procedure gives for simulated noise: independent and correlated (1 s), 1 cm."""
    rng = np.random.default_rng(SEED)
    a = math.exp(-1.0 / CORRELATION_TIME_FRAMES)
    out = {}
    for kind in ("independent", "correlated"):
        errors = []
        for ref in refs:
            noise = rng.standard_normal((LENGTH, 3))
            if kind == "correlated":
                for k in range(1, LENGTH):
                    noise[k] = a * noise[k - 1] + math.sqrt(1 - a * a) * noise[k]
            est_c = ref["centre"] + 0.01 * noise
            errors.append(window_errors(ref["centre"], ref["r_up"], est_c, ref["r_up"], scale=1.0)["e"])
        out[kind] = autocorrelation(errors)
    return out


def load_windows(model: str, names: list[str], plan: dict) -> dict:
    """Reference and estimate arrays of every window that has both, per sequence."""
    out: dict = {}
    for seq in names:
        rows = []
        for w in sorted(int(x) for x in plan["sequences"][seq]["windows"]):
            ref_path = WORK / "ref" / f"{seq}_w{w:02d}.npz"
            est_path = WORK / f"est_{model}" / f"{seq}_w{w:02d}.npz"
            if not (ref_path.exists() and est_path.exists()):
                continue
            with np.load(ref_path) as ref, np.load(est_path) as est:
                rows.append(
                    {
                        "w": w,
                        "centre": ref["centre"], "r_up": ref["r_up"], "path_m": float(ref["path_m"]),
                        "est_centre": est["centre"], "est_r": est["r_c2w"], "seconds": float(est["seconds"]),
                    }
                )
        out[seq] = rows
    return out


def analyse_sequence(rows: list[dict], all_paths: list[float]) -> dict:
    """Variants A (free scale per window, moving windows) and B (scale of the sequence) for one sequence."""
    free = [window_errors(r["centre"], r["r_up"], r["est_centre"], r["est_r"]) for r in rows]
    moving = [i for i, r in enumerate(rows) if r["path_m"] >= MOVING_PATH_M]
    scales = np.array([free[i]["s_free"] for i in moving])
    s_seq = float(np.median(scales)) if len(scales) else None
    fixed = (
        [window_errors(r["centre"], r["r_up"], r["est_centre"], r["est_r"], scale=s_seq) for r in rows]
        if s_seq is not None
        else None
    )
    median_path = float(np.median(all_paths)) if all_paths else 0.0
    slow = [i for i, r in enumerate(rows) if r["path_m"] < median_path]
    fast = [i for i, r in enumerate(rows) if r["path_m"] >= median_path]
    mov_set = set(moving)
    pick = lambda results, idx: [results[i] for i in idx] if results is not None else []
    variants = {
        "A_moving": {
            "all": pick(free, moving),
            "slow": pick(free, [i for i in slow if i in mov_set]),
            "fast": pick(free, [i for i in fast if i in mov_set]),
        },
        "B_all": {"all": pick(fixed, range(len(rows))), "slow": pick(fixed, slow), "fast": pick(fixed, fast)},
        "B_moving": {"all": pick(fixed, moving)},
    }
    rotation = float(np.median([float(np.median(f["theta"])) for f in free])) if free else float("nan")
    scale_summary = None
    if len(scales) >= 3:
        scale_summary = {
            "moving_windows": len(scales),
            "median": s_seq,
            "cv": float(scales.std() / scales.mean()) if scales.mean() != 0 else float("nan"),
            "quartile_range_over_median": float((np.percentile(scales, 75) - np.percentile(scales, 25)) / s_seq),
        }
    return {"variants": variants, "rotation_median_deg": rotation, "scale": scale_summary, "n_windows": len(rows), "n_moving": len(moving), "free": free}


def reading(summary: dict, reference: dict) -> dict:
    """The reading of protocol section 6 (amendment A1), for one estimator.

    Correlation in time is read at a lag of one frame: inside a 64-frame window after the fit, the
    autocorrelation at 0.5 s and 1 s is removed by the fit even for slowly varying error (a simulated
    correlated error shows 0.0 and -0.37 there), so those lags are reported but not used.
    """
    a = summary["A_moving"]["equivalent_sigma_cm"] if summary.get("A_moving", {}).get("windows") else None
    b = summary["B_all"]["equivalent_sigma_cm"] if summary.get("B_all", {}).get("windows") else None
    if a is None:
        return {"statement": "no moving window with an estimate: nothing to read"}
    rho = summary["A_moving"]["autocorrelation"]["1"]
    ind, cor = reference["independent"]["1"], reference["correlated"]["1"]
    correlated_like = rho > 0.5 * (ind + cor)
    ratio = summary["A_moving"]["equivalent_sigma_cm"] / (summary["A_moving"]["position_rms_cm"] / math.sqrt(3))
    verdict = {
        "sigma_A_moving_cm": a, "sigma_B_all_cm": b, "autocorrelation_lag1_frame": rho,
        "reference_autocorrelation_lag1_frame": {"independent": ind, "correlated": cor},
        "step_sigma_over_position_sigma": ratio, "correlated_like": bool(correlated_like),
    }
    limit = 100 * EQUIVALENT_SIGMA_LIMIT_M
    if correlated_like or a > limit:
        verdict["statement"] = "above 2 cm or slowly varying: the simulation under-states the error; next protocol is a different input"
    elif b is not None and b > limit:
        verdict["statement"] = "within 2 cm and independent only with a scale fitted per window (which a single camera does not give); a scale source is needed"
    else:
        verdict["statement"] = "within the simulated conditions (equivalent sigma at most 2 cm, not slowly varying): the detector can be tried on estimated poses"
    return verdict


def run_report(args) -> None:
    if RESULT.exists():
        raise SystemExit(f"{RESULT} exists; the result is written once")
    plan = json.loads((WORK / "plan.json").read_text())
    names = sorted(plan["sequences"])
    groups = {n: plan["sequences"][n]["group"] for n in names}
    report: dict = {
        "protocol": "docs/nymeria_pose_noise_protocol.md",
        "script_sha1": hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:16],
        "run": time.strftime("%Y-%m-%d %H:%M:%S"),
        "windows_per_sequence": WINDOWS, "window_frames": LENGTH, "moving_path_m": MOVING_PATH_M, "estimators": {},
    }
    only = getattr(args, "model", None)  # amendment A2: the report can be limited to one estimator
    for model in ((only,) if only else ("omega", "monst3r")):
        data = load_windows(model, names, plan)
        if not any(data.values()):
            continue
        refs_moving = [r for rows in data.values() for r in rows if r["path_m"] >= MOVING_PATH_M]
        reference = reference_autocorrelation(refs_moving) if refs_moving else None
        per_seq, pools = {}, {k: {"A_moving": [], "B_all": [], "B_moving": [], "A_slow": [], "A_fast": [], "B_slow": [], "B_fast": []} for k in ("pool", "test", "other", "outside_test", "all")}
        scale_rows = {}
        for seq in names:
            rows = data[seq]
            if not rows:
                continue
            all_paths = [v["path_m"] for v in plan["sequences"][seq]["windows"].values() if "path_m" in v]
            an = analyse_sequence(rows, all_paths)
            failed = not (an["rotation_median_deg"] <= GUARD_DEG)
            lags = [v["lag"] for v in plan["sequences"][seq]["windows"].values() if "lag" in v]
            shifts = [x["best_shift_frames"] for x in lags if x.get("corr_best", 0) >= 0.5]
            per_seq[seq] = {
                "frame_shift_diagnostic": {"windows_with_corr_at_least_0.5": len(shifts),
                                           "histogram": {str(k): shifts.count(k) for k in range(-2, 3)}},
                "group": groups[seq], "windows": an["n_windows"], "moving_windows": an["n_moving"],
                "failed_guard": failed, "rotation_median_deg": an["rotation_median_deg"], "scale": an["scale"],
                "seconds_per_window": float(np.mean([r["seconds"] for r in rows])),
                "stats": {
                    v: {part: pooled_stats(res) for part, res in parts.items()} for v, parts in an["variants"].items()
                },
            }
            if failed:
                continue
            for g in (groups[seq], "outside_test" if groups[seq] != "test" else None, "all"):
                if g is None:
                    continue
                pools[g]["A_moving"] += an["variants"]["A_moving"]["all"]
                pools[g]["B_all"] += an["variants"]["B_all"]["all"]
                pools[g]["B_moving"] += an["variants"]["B_moving"]["all"]
                pools[g]["A_slow"] += an["variants"]["A_moving"]["slow"]
                pools[g]["A_fast"] += an["variants"]["A_moving"]["fast"]
                pools[g]["B_slow"] += an["variants"]["B_all"]["slow"]
                pools[g]["B_fast"] += an["variants"]["B_all"]["fast"]
            if an["scale"]:
                scale_rows[seq] = an["scale"]
        pooled = {g: {k: pooled_stats(v) for k, v in d.items()} for g, d in pools.items()}
        report["estimators"][model] = {
            "windows_with_estimate": int(sum(len(v) for v in data.values())),
            "per_sequence": per_seq, "pooled": pooled, "scale_by_sequence": scale_rows,
            "reference_autocorrelation_1cm_simulated": reference,
            "reading_all_sequences": reading(pooled["all"], reference) if reference and pooled["all"]["A_moving"].get("windows") else None,
            "reading_outside_test": reading(pooled["outside_test"], reference) if reference and pooled["outside_test"]["A_moving"].get("windows") else None,
        }
    RESULT.write_text(json.dumps(report, indent=1))
    show(report)
    say(f"\nwritten to {RESULT}")


def show(report: dict) -> None:
    for model, block in report["estimators"].items():
        say(f"\n== {model}: {block['windows_with_estimate']} windows with an estimate ==")
        for seq, d in block["per_sequence"].items():
            a, b = d["stats"]["A_moving"]["all"], d["stats"]["B_all"]["all"]
            fmt = lambda s: f"{s['equivalent_sigma_cm']:.2f} cm (pos rms {s['position_rms_cm']:.2f})" if s.get("windows") else "n/a"
            h = d["frame_shift_diagnostic"]
            say(f"  {seq[:30]:30s} frame shift (-2..2) over {h['windows_with_corr_at_least_0.5']} windows: {h['histogram']}")
            say(f"  {seq[:30]:30s} [{d['group']:5s}] windows {d['windows']:2d} (moving {d['moving_windows']:2d}) rot {d['rotation_median_deg']:.1f} deg"
                f"{' FAILED' if d['failed_guard'] else ''}  sigma A {fmt(a)} | B {fmt(b)}")
        for key in ("reading_all_sequences", "reading_outside_test"):
            r = block[key]
            if r:
                if "sigma_A_moving_cm" in r:
                    b = r["sigma_B_all_cm"]
                    say(f"  {key}: sigma A {r['sigma_A_moving_cm']:.2f} cm, B {'n/a' if b is None else f'{b:.2f} cm'}, "
                        f"lag-1 autocorrelation {r['autocorrelation_lag1_frame']:.2f} (simulated independent "
                        f"{r['reference_autocorrelation_lag1_frame']['independent']:.2f}, correlated {r['reference_autocorrelation_lag1_frame']['correlated']:.2f})")
                say(f"    -> {r['statement']}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--step", required=True, choices=["prepare", "estimate", "report"])
    p.add_argument("--model", choices=["omega", "monst3r"])
    p.add_argument("--sequence")
    p.add_argument("--limit", type=int, default=0, help="only the first N windows of each sequence (smoke test)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--omega-dir", type=Path, default=MODELS / "vggt-omega")
    p.add_argument("--omega-checkpoint", type=Path, default=MODELS / "vggt-omega" / "checkpoints" / "vggt_omega_1b_512.pt")
    p.add_argument("--monst3r-dir", type=Path, default=MODELS / "monst3r")
    p.add_argument("--monst3r-weights", type=Path, default=MODELS / "monst3r" / "checkpoints")
    args = p.parse_args(argv)
    if args.step == "prepare":
        run_prepare(args)
    elif args.step == "estimate":
        if not args.model:
            p.error("--model is required for --step estimate")
        (run_omega if args.model == "omega" else run_monst3r)(args)
    else:
        run_report(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
