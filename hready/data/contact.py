"""Foot contact labels on the 30 Hz AMASS grid (item 5c-1)."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import time
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from hready.data.amass import (
    AmassIndexEntry,
    _entry_by_rel,
    _load_floor_sidecar,
    _load_npz_raw,
    _resample_axis_angle_series,
    _target_frame_count,
    amass_root_from_config,
    build_clip_flags_cache,
    build_floor_cache,
    cache_dir_from_config,
    load_index,
    load_paths_config,
)

_TARGET_FPS = 30.0
_FOOT_TRAJ_DIR = "foot_traj"
_FOOT_TRAJ_INDEX = "foot_traj_index.json"
_SKATE_SCORES = "amass_skate_scores.json"

CHANNEL_NAMES: tuple[str, ...] = ("L_heel", "L_toe", "R_heel", "R_toe")
_SOLE_BAND_M = 0.015

# Native SMPL-X neutral mesh axes (pelvis 0 -> head 15 is +Y; ankle 7.x > ankle 8.x => +X left;
# ankle 7 -> foot joint 10 is predominantly +Z). AMASS grounding uses world Z = mesh Z.
FOOT_NATIVE_LATERAL_AXIS = 0  # +X left
FOOT_NATIVE_UP_AXIS = 1  # +Y up
FOOT_NATIVE_FORWARD_AXIS = 2  # +Z forward
_LEFT_FOOT_LBS_JOINTS = (7, 10)  # left_ankle, left_foot
_RIGHT_FOOT_LBS_JOINTS = (8, 11)  # right_ankle, right_foot

# BABEL frame_ann grid (Oct 2026): h_on=0.05 passes walk/stand/jump targets; floor offset is
# 1st-percentile vertex z (AMASS penetration on GT up to ~5 mm), so h_on=0.05 absorbs that band.
CONTACT_H_ON_M = 0.05
CONTACT_H_OFF_M = 0.06
CONTACT_V_ON_M_S = 0.2
CONTACT_V_OFF_M_S = 0.25
_CONTACT_MIN_RUN = 3
SKATE_NEAR_FLOOR_M = 0.03
SKATE_MAX_FRAMES_30HZ = 120  # 4 s center crop at 30 Hz for belt-skate score
# Recorded at the BMLrub skate_score histogram valley (Oct 2026); > p99 non-BMLrub (~0.265 m/s).
T_SKATE = 0.35

_foot_channel_clusters: Optional[dict[str, np.ndarray]] = None
_foot_vertex_sets: Optional[tuple[np.ndarray, np.ndarray]] = None


def _foot_traj_root(cache_dir: Path) -> Path:
    return cache_dir / _FOOT_TRAJ_DIR


def _foot_traj_path(cache_dir: Path, rel_path: str) -> Path:
    safe = rel_path.replace("/", "__")
    return _foot_traj_root(cache_dir) / f"{safe}.npz"


def _neutral_pose_mesh(body: Any) -> tuple[np.ndarray, np.ndarray]:
    import torch

    out_mesh = body.forward(
        torch.zeros(1, 3),
        torch.zeros(1, 63),
        torch.zeros(1, 16),
        torch.zeros(1, 3),
    )
    v = out_mesh.vertices[0].detach().cpu().numpy()
    joints = out_mesh.joints[0].detach().cpu().numpy()
    return v, joints


def _foot_sets_from_lbs(body: Any) -> tuple[np.ndarray, np.ndarray]:
    w = body._model.lbs_weights.detach().cpu().numpy()[:10475]
    j_id = w.argmax(axis=1)
    left = np.where(np.isin(j_id, _LEFT_FOOT_LBS_JOINTS))[0].astype(np.int64)
    right = np.where(np.isin(j_id, _RIGHT_FOOT_LBS_JOINTS))[0].astype(np.int64)
    return left, right


def _sole_heel_toe_clusters(
    foot_idx: np.ndarray, ankle_i: int, v: np.ndarray, joints: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Sole band along native up; split at ankle forward (Z) into heel (behind) / toe (ahead)."""
    up = FOOT_NATIVE_UP_AXIS
    fwd = FOOT_NATIVE_FORWARD_AXIS
    ankle = joints[ankle_i]
    y_min = float(v[foot_idx, up].min())
    sole = foot_idx[v[foot_idx, up] <= y_min + _SOLE_BAND_M]
    rel = v[sole] - ankle
    along = rel[:, fwd]
    heel = sole[along < 0.0]
    toe = sole[along >= 0.0]
    if heel.size == 0:
        med = float(np.median(along))
        heel = sole[along <= med]
    if toe.size == 0:
        med = float(np.median(along))
        toe = sole[along > med]
    return np.unique(heel), np.unique(toe)


def resolve_foot_channel_clusters(body: Optional[Any] = None) -> dict[str, np.ndarray]:
    """LBS foot sets + sole heel/toe clusters (all sole vertices kept per channel)."""
    global _foot_channel_clusters, _foot_vertex_sets
    if _foot_channel_clusters is not None:
        return _foot_channel_clusters

    from hready.body.smplx_wrapper import load_body

    if body is None:
        body = load_body("locked_head")
    v, joints = _neutral_pose_mesh(body)
    left, right = _foot_sets_from_lbs(body)
    lh, lt = _sole_heel_toe_clusters(left, 7, v, joints)
    rh, rt = _sole_heel_toe_clusters(right, 8, v, joints)
    _foot_vertex_sets = (left, right)
    _foot_channel_clusters = {
        "L_heel": lh,
        "L_toe": lt,
        "R_heel": rh,
        "R_toe": rt,
    }
    return _foot_channel_clusters


def foot_vertex_sets() -> tuple[np.ndarray, np.ndarray]:
    resolve_foot_channel_clusters()
    assert _foot_vertex_sets is not None
    return _foot_vertex_sets


def _cluster_offset_mm(
    vertices: np.ndarray, ankle: np.ndarray, cluster: np.ndarray
) -> tuple[float, float, float]:
    """Centroid vs ankle in native frame: lateral (+X), forward (+Z), up (+Y), mm."""
    lat = FOOT_NATIVE_LATERAL_AXIS
    up = FOOT_NATIVE_UP_AXIS
    fwd = FOOT_NATIVE_FORWARD_AXIS
    c = vertices[cluster].mean(axis=0) - ankle
    return float(c[lat] * 1000.0), float(c[fwd] * 1000.0), float(c[up] * 1000.0)


def write_foot_channel_pngs(
    out_dir: Path, body: Optional[Any] = None
) -> tuple[Path, Path]:
    """Side + top views per foot; heel red, toe blue, other foot verts gray, ankle cross."""
    import matplotlib.pyplot as plt
    from hready.body.smplx_wrapper import load_body

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if body is None:
        body = load_body("locked_head")
    v, j = _neutral_pose_mesh(body)
    clusters = resolve_foot_channel_clusters(body)
    left, right = foot_vertex_sets()
    paths: list[Path] = []

    def _plot_foot(
        side: str, foot_idx: np.ndarray, ankle_i: int, heel: np.ndarray, toe: np.ndarray
    ) -> Path:
        ankle = j[ankle_i]
        fig, (ax_side, ax_top) = plt.subplots(1, 2, figsize=(10, 5))
        rest = np.setdiff1d(foot_idx, np.concatenate([heel, toe]))
        for ax, ix, iy, xlabel, ylabel in (
            (
                ax_side,
                FOOT_NATIVE_FORWARD_AXIS,
                FOOT_NATIVE_UP_AXIS,
                "forward (+Z)",
                "up (+Y)",
            ),
            (
                ax_top,
                FOOT_NATIVE_LATERAL_AXIS,
                FOOT_NATIVE_FORWARD_AXIS,
                "lateral (+X)",
                "forward (+Z)",
            ),
        ):
            if rest.size:
                ax.scatter(v[rest, ix], v[rest, iy], c="#bbbbbb", s=8, alpha=0.6)
            ax.scatter(v[heel, ix], v[heel, iy], c="red", s=20, label="heel")
            ax.scatter(v[toe, ix], v[toe, iy], c="blue", s=20, label="toe")
            ax.scatter(
                ankle[ix],
                ankle[iy],
                c="k",
                marker="x",
                s=80,
                linewidths=2,
                label="ankle",
            )
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_aspect("equal", adjustable="box")
            ax.legend(loc="best", fontsize=8)
        fig.suptitle(f"{side} foot (LBS verts)")
        fig.tight_layout()
        path = out_dir / f"foot_{side.lower()}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    paths.append(_plot_foot("Left", left, 7, clusters["L_heel"], clusters["L_toe"]))
    paths.append(_plot_foot("Right", right, 8, clusters["R_heel"], clusters["R_toe"]))
    return paths[0], paths[1]


def print_foot_channel_report() -> None:
    """LBS clusters: acceptance table, foot-set sizes, CMU stand heights (no stance plane)."""
    from hready.body.smplx_wrapper import load_body
    from hready.data.amass import _entry_by_rel, _resolve_torch_device, load_index

    body = load_body("locked_head")
    v, j = _neutral_pose_mesh(body)
    clusters = resolve_foot_channel_clusters(body)
    left, right = foot_vertex_sets()
    print(
        "Native frame: +X lateral (left), +Y up, +Z forward; "
        "channel height = min cluster Z minus floor_offset (AMASS Z-up)"
    )
    print(f"foot set sizes: left={left.size} right={right.size}")
    print("acceptance (centroid vs ankle, mm):")
    print("channel  n   lateral  forward   up")
    stats: dict[str, tuple[int, float, float, float]] = {}
    for ch, ankle_i in zip(CHANNEL_NAMES, (7, 7, 8, 8)):
        c = clusters[ch]
        lat, fwd, up = _cluster_offset_mm(v, j[ankle_i], c)
        stats[ch] = (c.size, lat, fwd, up)
        print(f"  {ch:6} {c.size:3} {lat:8.1f} {fwd:8.1f} {up:8.1f}")
    lh = stats["L_heel"]
    rh = stats["R_heel"]
    lt = stats["L_toe"]
    rt = stats["R_toe"]
    print(
        f"  L/R lateral |diff|: heel {abs(abs(lh[1]) - abs(rh[1])):.1f} mm "
        f"toe {abs(abs(lt[1]) - abs(rt[1])):.1f} mm (target <=10)"
    )
    rel = "CMU/91/91_48_stageii.npz"
    entry = _entry_by_rel(load_index(), rel)
    cache_dir = cache_dir_from_config()
    floor = float(_load_floor_sidecar(cache_dir)["entries"][rel]["floor_offset"])
    dev = _resolve_torch_device(None)
    body._model.to(dev)
    pos = forward_foot_channel_positions(
        entry, floor, amass_root=amass_root_from_config(), device=str(dev), body=body
    )
    z_cm = pos[:, :, 2] * 100.0
    for fi in (0, 10, 20, 30, 40):
        if fi < z_cm.shape[0]:
            print(
                f"  CMU stand {rel} frame {fi} channel heights cm: {z_cm[fi].round(1)}"
            )


def vertex_contact(
    height_above_floor: np.ndarray,
    speed_horiz: np.ndarray,
    *,
    h_on: float = CONTACT_H_ON_M,
    h_off: float = CONTACT_H_OFF_M,
    v_on: float = CONTACT_V_ON_M_S,
    v_off: float = CONTACT_V_OFF_M_S,
    min_run: int = _CONTACT_MIN_RUN,
) -> np.ndarray:
    """Hysteresis contact rule on height (m) and horizontal speed (m/s).

    ``height_above_floor`` and ``speed_horiz`` share shape ``(T,)`` or ``(T, C)``.
    Returns bool contact mask of the same shape.
    """
    h = np.asarray(height_above_floor, dtype=np.float64)
    s = np.asarray(speed_horiz, dtype=np.float64)
    if h.ndim == 1:
        h = h[:, None]
        s = s[:, None]
    t, c = h.shape
    state = np.zeros(c, dtype=bool)
    out = np.zeros((t, c), dtype=bool)
    for i in range(t):
        for j in range(c):
            if state[j]:
                if h[i, j] > h_off or s[i, j] > v_off:
                    state[j] = False
            else:
                if h[i, j] < h_on and s[i, j] < v_on:
                    state[j] = True
            out[i, j] = state[j]
    if min_run > 1 and t > 0:
        for j in range(c):
            out[:, j] = _enforce_min_run(out[:, j], min_run)
    return out if out.shape[1] > 1 else out[:, 0]


def _enforce_min_run(mask: np.ndarray, min_run: int) -> np.ndarray:
    m = mask.astype(bool).copy()
    i = 0
    while i < m.size:
        if not m[i]:
            i += 1
            continue
        j = i + 1
        while j < m.size and m[j]:
            j += 1
        if j - i < min_run:
            m[i:j] = False
        i = j
    return m


def heights_and_speeds_from_positions(
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """``positions`` grounded ``(T, 4, 3)`` -> heights ``(T,4)``, horiz speed ``(T,4)``.

    Speed uses finite differences **per channel** along time (``diff`` on axis 0 for each
    channel's own ``xy`` trajectory); skate_score then takes the speed of whichever channel
    is lowest at each frame when within ``SKATE_NEAR_FLOOR_M``.
    """
    pos = np.asarray(positions, dtype=np.float64)
    z = pos[:, :, 2]
    xy = pos[:, :, :2]
    dt = 1.0 / _TARGET_FPS
    speed = np.zeros_like(z)
    if pos.shape[0] > 1:
        speed[1:] = np.linalg.norm(np.diff(xy, axis=0), axis=2) / dt
        speed[0] = speed[1]
    return z, speed


def compute_skate_score_from_positions(positions: np.ndarray) -> float:
    z, speed = heights_and_speeds_from_positions(positions)
    return skate_score_from_height_speed(z, speed)


def compute_foot_contact_from_positions(
    positions: np.ndarray,
    *,
    h_on: float = CONTACT_H_ON_M,
    h_off: float = CONTACT_H_OFF_M,
    v_on: float = CONTACT_V_ON_M_S,
    v_off: float = CONTACT_V_OFF_M_S,
) -> np.ndarray:
    z, speed = heights_and_speeds_from_positions(positions)
    return vertex_contact(
        z, speed, h_on=h_on, h_off=h_off, v_on=v_on, v_off=v_off
    ).astype(np.bool_)


def forward_foot_channel_positions(
    entry: AmassIndexEntry,
    floor_offset: float,
    *,
    amass_root: Optional[Path] = None,
    device: Optional[str] = None,
    body: Optional[Any] = None,
    max_frames: Optional[int] = None,
) -> np.ndarray:
    """Single SMPL-X pass: grounded channel positions ``(T, 4, 3)`` float32."""
    import torch
    from hready.body.smplx_wrapper import load_body
    from hready.data.amass import _resolve_torch_device, resample_translation

    clusters = resolve_foot_channel_clusters()
    all_ids = np.unique(np.concatenate([clusters[ch] for ch in CHANNEL_NAMES]))
    id_to_j = {int(vid): j for j, vid in enumerate(all_ids)}
    channel_cols = [
        np.array([id_to_j[int(v)] for v in clusters[ch]], dtype=np.int64)
        for ch in CHANNEL_NAMES
    ]

    amass_root = (amass_root or amass_root_from_config()).resolve()
    raw = _load_npz_raw(amass_root / entry.rel_path)
    fps_src = float(entry.fps)
    n_src = int(raw["root_orient"].shape[0])
    if max_frames is not None:
        max_src = min(n_src, int(math.ceil(max_frames * fps_src / _TARGET_FPS)) + 2)
        if n_src > max_src:
            s0 = (n_src - max_src) // 2
            s1 = s0 + max_src
            raw = {
                "root_orient": raw["root_orient"][s0:s1],
                "pose_body": raw["pose_body"][s0:s1],
                "trans": raw["trans"][s0:s1],
                "betas": raw["betas"],
            }
            n_src = max_src
    n_tgt = _target_frame_count(n_src, fps_src, _TARGET_FPS)
    root = _resample_axis_angle_series(
        raw["root_orient"].reshape(-1, 1, 3), fps_src, _TARGET_FPS
    ).reshape(-1, 3)
    body_aa = _resample_axis_angle_series(
        raw["pose_body"].reshape(-1, 21, 3), fps_src, _TARGET_FPS
    )
    transl = resample_translation(raw["trans"], fps_src, _TARGET_FPS)
    betas = raw["betas"]
    if max_frames is not None and n_tgt > max_frames:
        s0 = (n_tgt - max_frames) // 2
        s1 = s0 + max_frames
        root = root[s0:s1]
        body_aa = body_aa[s0:s1]
        transl = transl[s0:s1]
        n_tgt = max_frames

    dev = _resolve_torch_device(device)
    dtype = torch.float32
    if body is None:
        body = load_body("locked_head")
        body._model.to(dev)
    chunk = 128
    out_list: list[np.ndarray] = []
    for i0 in range(0, n_tgt, chunk):
        i1 = min(n_tgt, i0 + chunk)
        go = torch.as_tensor(root[i0:i1], device=dev, dtype=dtype)
        bp = torch.as_tensor(body_aa[i0:i1], device=dev, dtype=dtype).reshape(
            i1 - i0, 63
        )
        tr = torch.as_tensor(transl[i0:i1], device=dev, dtype=dtype)
        be = (
            torch.as_tensor(betas, device=dev, dtype=dtype)
            .unsqueeze(0)
            .expand(i1 - i0, -1)
        )
        with torch.inference_mode():
            verts = (
                body.forward(go, bp, be, tr)
                .vertices[:, all_ids, :]
                .detach()
                .cpu()
                .numpy()
            )
        frame_pos = np.zeros((verts.shape[0], 4, 3), dtype=np.float32)
        for ci, cols in enumerate(channel_cols):
            sub = verts[:, cols, :]
            frame_pos[:, ci, 0] = sub[:, :, 0].mean(axis=1)
            frame_pos[:, ci, 1] = sub[:, :, 1].mean(axis=1)
            frame_pos[:, ci, 2] = sub[:, :, 2].min(axis=1) - floor_offset
        out_list.append(frame_pos)
    return np.concatenate(out_list, axis=0)


def load_foot_positions(
    entry: Union[AmassIndexEntry, str],
    *,
    cache_dir: Optional[Path] = None,
) -> np.ndarray:
    cache_dir = (cache_dir or cache_dir_from_config()).resolve()
    if isinstance(entry, str):
        entry = _entry_by_rel(load_index(cache_dir), entry)
    path = _foot_traj_path(cache_dir, entry.rel_path)
    if not path.is_file():
        raise FileNotFoundError(f"No foot trajectory cache for {entry.rel_path}")
    with np.load(path) as data:
        return data["positions"].astype(np.float32)


def foot_traj_build_clip_list(
    *, seed: int = 0, n_babel: int = 300
) -> list[AmassIndexEntry]:
    """All BMLrub + stratified BABEL-labeled non-BMLrub sample."""
    import random
    from collections import defaultdict

    from hready.data.babel import act_cat_matches_keyword, load_babel_index_payload

    index = {e.rel_path: e for e in load_index()}
    bml = [e for e in index.values() if e.subset == "BMLrub"]
    by_kw: dict[str, list[AmassIndexEntry]] = defaultdict(list)
    for row in load_babel_index_payload()["entries"]:
        rel = row.get("rel_path") or row.get("feat_p", "")
        if rel not in index or index[rel].subset == "BMLrub":
            continue
        ent = index[rel]
        cats: list[str] = []
        for seg in row.get("segments") or []:
            cats.extend(seg.get("act_cat") or [])
        for kw in ("walk", "run", "stand", "sit", "jump", "step"):
            if any(act_cat_matches_keyword(c, kw) for c in cats):
                by_kw[kw].append(ent)
                break
    random.seed(seed)
    per = max(1, n_babel // max(1, len(by_kw)))
    sample: list[AmassIndexEntry] = []
    for kw in sorted(by_kw):
        pool = by_kw[kw]
        random.shuffle(pool)
        sample.extend(pool[:per])
    random.shuffle(sample)
    sample = sample[:n_babel]
    rels = {e.rel_path for e in bml} | {e.rel_path for e in sample}
    return [index[r] for r in sorted(rels) if r in index]


def entries_missing_foot_traj(
    cache_dir: Optional[Path] = None,
) -> list[AmassIndexEntry]:
    cache_dir = (cache_dir or cache_dir_from_config()).resolve()
    return [
        e
        for e in load_index(cache_dir)
        if not _foot_traj_path(cache_dir, e.rel_path).is_file()
    ]


def _foot_traj_clip_build_target(
    rel_path: str,
    floor_offset: float,
    amass_root_str: str,
    cache_dir_str: str,
    device: Optional[str],
) -> None:
    """Spawn worker: forward foot channels and write ``foot_traj`` npz."""
    amass_root = Path(amass_root_str)
    cache_dir = Path(cache_dir_str)
    entry = _entry_by_rel(load_index(cache_dir), rel_path)
    out_path = _foot_traj_path(cache_dir, rel_path)
    from hready.body.smplx_wrapper import load_body
    from hready.data.amass import _resolve_torch_device

    import sys

    try:
        body = load_body("locked_head")
        body._model.to(_resolve_torch_device(device))
        pos = forward_foot_channel_positions(
            entry, floor_offset, amass_root=amass_root, device=device, body=body
        )
        np.savez_compressed(out_path, positions=pos.astype(np.float32))
    except Exception:
        sys.exit(1)


def build_foot_traj_cache(
    entries: Optional[list[AmassIndexEntry]] = None,
    *,
    amass_root: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    device: Optional[str] = None,
    resume: bool = True,
    force: bool = False,
    clip_timeout_s: Optional[float] = None,
) -> dict[str, Any]:
    amass_root = (amass_root or amass_root_from_config()).resolve()
    cache_dir = (cache_dir or cache_dir_from_config()).resolve()
    _foot_traj_root(cache_dir).mkdir(parents=True, exist_ok=True)
    floor_side = json.loads(
        (cache_dir / "amass_floor.json").read_text(encoding="utf-8")
    )
    entries = entries or foot_traj_build_clip_list()
    from hready.body.smplx_wrapper import load_body
    from hready.data.amass import _resolve_torch_device

    body = load_body("locked_head")
    body._model.to(dev := _resolve_torch_device(device))
    index_entries: dict[str, Any] = {}
    index_path = cache_dir / _FOOT_TRAJ_INDEX
    if resume and index_path.is_file():
        with index_path.open(encoding="utf-8") as f:
            index_entries = json.load(f).get("entries", {})
    t0 = time.perf_counter()
    built = 0
    corrupt: list[str] = []
    timeouts: list[str] = []
    mp_ctx = (
        multiprocessing.get_context("spawn") if clip_timeout_s is not None else None
    )
    for i, entry in enumerate(entries):
        out_path = _foot_traj_path(cache_dir, entry.rel_path)
        if resume and not force and out_path.is_file():
            index_entries[entry.rel_path] = {
                "path": str(out_path.relative_to(cache_dir)),
                "n_frames": int(np.load(out_path)["positions"].shape[0]),
            }
            continue
        floor_rec = floor_side["entries"].get(entry.rel_path)
        if floor_rec is None:
            continue
        floor_off = float(floor_rec["floor_offset"])
        try:
            if mp_ctx is not None:
                proc = mp_ctx.Process(
                    target=_foot_traj_clip_build_target,
                    args=(
                        entry.rel_path,
                        floor_off,
                        str(amass_root),
                        str(cache_dir),
                        device,
                    ),
                )
                proc.start()
                proc.join(timeout=clip_timeout_s)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(10.0)
                    if out_path.is_file():
                        out_path.unlink(missing_ok=True)
                    timeouts.append(entry.rel_path)
                    continue
                if proc.exitcode != 0 or not out_path.is_file():
                    corrupt.append(f"{entry.rel_path}: worker exitcode={proc.exitcode}")
                    continue
                pos = np.load(out_path)["positions"]
            else:
                pos = forward_foot_channel_positions(
                    entry,
                    floor_off,
                    amass_root=amass_root,
                    device=device,
                    body=body,
                )
                np.savez_compressed(out_path, positions=pos.astype(np.float32))
        except Exception as exc:
            corrupt.append(f"{entry.rel_path}: {type(exc).__name__}: {exc}")
            continue
        index_entries[entry.rel_path] = {
            "path": str(out_path.relative_to(cache_dir)),
            "n_frames": int(pos.shape[0]),
        }
        built += 1
        if built % 50 == 0:
            elapsed = time.perf_counter() - t0
            rate = built / elapsed if elapsed > 0 else 0.0
            print(
                f"  foot_traj: {built} built ({i + 1}/{len(entries)}) {rate:.2f} clips/s",
                flush=True,
            )
    payload = {
        "version": 1,
        "channels": list(CHANNEL_NAMES),
        "entries": index_entries,
    }
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    elapsed = time.perf_counter() - t0
    return {
        "n_clips": len(entries),
        "n_built": built,
        "n_indexed": len(index_entries),
        "corrupt": corrupt,
        "timeouts": timeouts,
        "wall_s": elapsed,
        "clips_per_s": built / elapsed if elapsed > 0 else 0.0,
    }


def skate_score_from_height_speed(
    z_grounded: np.ndarray,
    speed_horiz: np.ndarray,
    *,
    near_floor_m: float = SKATE_NEAR_FLOOR_M,
) -> float:
    """Median horizontal speed of the lowest foot channel when within ``near_floor_m`` of the floor."""
    if z_grounded.size == 0:
        return float("nan")
    z = np.asarray(z_grounded, dtype=np.float64)
    vals: list[float] = []
    for i in range(z.shape[0]):
        j = int(np.argmin(z[i]))
        if z[i, j] <= near_floor_m:
            vals.append(float(speed_horiz[i, j]))
    if not vals:
        return float("nan")
    return float(np.median(vals))


def compute_skate_score(
    entry: AmassIndexEntry,
    floor_offset: float,
    *,
    amass_root: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    device: Optional[str] = None,
    body: Optional[Any] = None,
    positions: Optional[np.ndarray] = None,
) -> float:
    if positions is None:
        try:
            positions = load_foot_positions(
                entry, cache_dir=cache_dir or cache_dir_from_config()
            )
        except FileNotFoundError:
            positions = None
    if positions is not None:
        if (
            SKATE_MAX_FRAMES_30HZ is not None
            and positions.shape[0] > SKATE_MAX_FRAMES_30HZ
        ):
            s0 = (positions.shape[0] - SKATE_MAX_FRAMES_30HZ) // 2
            positions = positions[s0 : s0 + SKATE_MAX_FRAMES_30HZ]
        return compute_skate_score_from_positions(positions)
    amass_root = (amass_root or amass_root_from_config()).resolve()
    pos = forward_foot_channel_positions(
        entry,
        floor_offset,
        amass_root=amass_root,
        device=device,
        body=body,
        max_frames=SKATE_MAX_FRAMES_30HZ,
    )
    return compute_skate_score_from_positions(pos)


def compute_foot_contact(
    entry: AmassIndexEntry,
    floor_offset: float,
    *,
    amass_root: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    device: Optional[str] = None,
    body: Optional[Any] = None,
    positions: Optional[np.ndarray] = None,
    **contact_kw: Any,
) -> np.ndarray:
    """Bool contact ``(T, 4)`` channels L heel, L toe, R heel, R toe."""
    if positions is None:
        try:
            positions = load_foot_positions(
                entry, cache_dir=cache_dir or cache_dir_from_config()
            )
        except FileNotFoundError:
            positions = None
    if positions is not None:
        return compute_foot_contact_from_positions(positions, **contact_kw)
    amass_root = (amass_root or amass_root_from_config()).resolve()
    pos = forward_foot_channel_positions(
        entry, floor_offset, amass_root=amass_root, device=device, body=body
    )
    return compute_foot_contact_from_positions(pos, **contact_kw)


def _skate_scores_path(cache_dir: Path) -> Path:
    return cache_dir / _SKATE_SCORES


def build_skate_scores_cache(
    *,
    amass_root: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    device: Optional[str] = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Per-clip belt-skate scores for clip flags (incremental JSON sidecar)."""
    amass_root = (amass_root or amass_root_from_config()).resolve()
    cache_dir = (cache_dir or cache_dir_from_config()).resolve()
    floor_side = json.loads(
        (cache_dir / "amass_floor.json").read_text(encoding="utf-8")
    )
    path = _skate_scores_path(cache_dir)
    payload: dict[str, Any] = {"version": 1, "entries": {}}
    if resume and path.is_file():
        with path.open(encoding="utf-8") as f:
            payload = json.load(f)
    entries_map = payload.setdefault("entries", {})
    entries = load_index(cache_dir)
    from hready.body.smplx_wrapper import load_body
    from hready.data.amass import _resolve_torch_device

    dev = _resolve_torch_device(device)
    body = load_body("locked_head")
    body._model.to(dev)
    t0 = time.perf_counter()
    built = 0
    for i, entry in enumerate(entries):
        if entry.rel_path in entries_map and resume:
            continue
        floor_rec = floor_side["entries"].get(entry.rel_path)
        if floor_rec is None:
            entries_map[entry.rel_path] = float("nan")
        else:
            positions = None
            try:
                positions = load_foot_positions(entry, cache_dir=cache_dir)
            except FileNotFoundError:
                positions = None
            entries_map[entry.rel_path] = compute_skate_score(
                entry,
                float(floor_rec["floor_offset"]),
                amass_root=amass_root,
                device=device,
                body=body,
                positions=positions,
            )
        built += 1
        if built % 100 == 0:
            with path.open("w", encoding="utf-8") as f:
                json.dump(payload, f)
            print(f"  skate scores: {i + 1}/{len(entries)}", flush=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)
    elapsed = time.perf_counter() - t0
    return {
        "n_entries": len(entries_map),
        "n_built_this_run": built,
        "wall_s": elapsed,
        "clips_per_s": built / elapsed if elapsed > 0 else 0.0,
    }


def load_contact(
    entry: Union[AmassIndexEntry, str],
    *,
    cache_dir: Optional[Path] = None,
    device: Optional[str] = None,
    **contact_kw: Any,
) -> np.ndarray:
    """Foot contact ``(T, 4)`` bool at 30 Hz (from ``foot_traj`` when cached)."""
    cache_dir = (cache_dir or cache_dir_from_config()).resolve()
    if isinstance(entry, str):
        entry = _entry_by_rel(load_index(cache_dir), entry)
    floor_rec = _load_floor_sidecar(cache_dir)["entries"].get(entry.rel_path)
    if floor_rec is None:
        raise KeyError(f"No floor cache for {entry.rel_path}")
    return compute_foot_contact(
        entry,
        float(floor_rec["floor_offset"]),
        cache_dir=cache_dir,
        device=device,
        **contact_kw,
    )


def _print_cluster_summary() -> None:
    c = resolve_foot_channel_clusters()
    left, right = foot_vertex_sets()
    print(
        f"Foot channels (LBS argmax joints L{{7,10}} R{{8,11}}): left_set={left.size} right_set={right.size}"
    )
    for ch in CHANNEL_NAMES:
        print(f"  {ch}: cluster n={c[ch].size}")


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description="AMASS foot contact cache")
    p.add_argument(
        "--build", action="store_true", help="Build floor cache and clip flags"
    )
    p.add_argument(
        "--build-skate", action="store_true", help="Fill amass_skate_scores.json only"
    )
    p.add_argument(
        "--build-foot-traj",
        action="store_true",
        help="Build foot_traj cache (BMLrub + 300 BABEL)",
    )
    p.add_argument(
        "--build-foot-traj-all",
        action="store_true",
        help="Build foot_traj for every indexed clip",
    )
    p.add_argument(
        "--build-foot-traj-missing",
        action="store_true",
        help="Build only index clips with no foot_traj npz (resume skips existing)",
    )
    p.add_argument(
        "--clip-timeout-s",
        type=float,
        default=None,
        help="Per-clip wall limit; on timeout skip, log, and continue",
    )
    p.add_argument(
        "--force-foot-traj", action="store_true", help="Rebuild all foot_traj files"
    )
    p.add_argument(
        "--print-foot-channels", action="store_true", help="Print cluster diagnostics"
    )
    p.add_argument("--force-floor", action="store_true")
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)
    if args.print_foot_channels:
        print_foot_channel_report()
        return
    if args.build_skate:
        stats = build_skate_scores_cache(device=args.device)
        print("Skate scores cache:", stats)
        return
    if args.build_foot_traj or args.build_foot_traj_all or args.build_foot_traj_missing:
        cache_dir = cache_dir_from_config()
        if args.build_foot_traj_missing:
            clip_list = entries_missing_foot_traj(cache_dir)
            print(f"foot_traj missing: {len(clip_list)} clips", flush=True)
            for ent in clip_list:
                print(ent.rel_path)
        elif args.build_foot_traj_all:
            clip_list = load_index(cache_dir)
        else:
            clip_list = foot_traj_build_clip_list()
        stats = build_foot_traj_cache(
            clip_list,
            device=args.device,
            force=args.force_foot_traj,
            clip_timeout_s=args.clip_timeout_s,
        )
        print("Foot traj cache:", stats)
        if stats.get("timeouts"):
            print(f"  timeouts ({len(stats['timeouts'])}):")
            for line in stats["timeouts"][:20]:
                print(f"    {line}")
        if stats.get("corrupt"):
            print(f"  corrupt ({len(stats['corrupt'])}):")
            for line in stats["corrupt"][:20]:
                print(f"    {line}")
        return
    if not args.build:
        p.print_help()
        return
    _print_cluster_summary()
    floor_stats = build_floor_cache(force=args.force_floor, device=args.device)
    print("Floor cache:", floor_stats)
    print(
        f"  global median floor offset: {floor_stats['global_median_floor_offset_m']:.4f} m"
    )
    print(
        f"  floor outliers (|offset-med|>{0.35}): {len(floor_stats['floor_outliers'])}"
    )
    for rel in floor_stats["floor_outliers"]:
        print(f"    {rel}")
    flag_stats = build_clip_flags_cache(device=args.device, compute_missing_skate=True)
    print("Clip flags counts:", flag_stats["counts"])
    print(
        "  name-flag among skate+belt:",
        flag_stats.get("name_flag_among_skate_belt"),
        "/",
        flag_stats.get("skate_belt_total"),
    )


# Foot-height rise cache (informational QA; not used by loaders).
from hready.data.foot_height_rise import (  # noqa: E402
    build_foot_height_rise_cache,
    foot_height_rise_cm_from_positions,
    foot_height_rise_summary,
    load_foot_height_rise_cache,
)

__all_foot_height_rise__ = (
    "foot_height_rise_cm_from_positions",
    "build_foot_height_rise_cache",
    "load_foot_height_rise_cache",
    "foot_height_rise_summary",
)


if __name__ == "__main__":
    main()
