"""Reference motion for the simulation tracking study (E6): window selection and the reference of the SMPL-X-skeleton humanoid.

The humanoid is the neutral-shape asset driven by the clip's hinge angles (clipped to the asset's joint limits) and the clip's
root orientation. Its root sits at the subject's pelvis joint plus one constant vertical offset per clip, chosen so that the
lowest point of the humanoid touches the floor at its lowest frame. Horizontal position starts at the origin.
Numpy only; the data access (body model, clip index, BABEL) lives in ``scripts/tracking_reference_export.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hready.robot import humanoid_asset as H

CATEGORIES = ("walk", "fast", "sit_support", "floor_work")
SEED = 0
FPS = 30.0
MIN_SEGMENT_S = 5.0
MAX_WINDOW_S = 10.0
WINDOWS_PER_CATEGORY = 8
CLEARANCE_M = 0.003  # lowest point of the humanoid above the floor at its lowest frame, as in the stand check


@dataclass(frozen=True)
class Candidate:
    category: str
    rel_path: str
    subject_key: str
    start_s: float
    end_s: float


def shuffled_order(candidates: list[Candidate], seed: int = SEED) -> dict[str, list[Candidate]]:
    """Per category: segments of at least MIN_SEGMENT_S, sorted by (rel_path, start), shuffled with a generator seeded by
    (seed, category index), then the first segment of every subject is kept. The first entry of each list is the stage 2 clip."""
    out: dict[str, list[Candidate]] = {}
    for ci, cat in enumerate(CATEGORIES):
        rows = sorted((c for c in candidates if c.category == cat and c.end_s - c.start_s >= MIN_SEGMENT_S),
                      key=lambda c: (c.rel_path, c.start_s))
        seen: set[str] = set()
        kept: list[Candidate] = []
        for i in np.random.default_rng([seed, ci]).permutation(len(rows)):
            c = rows[int(i)]
            if c.subject_key not in seen:
                seen.add(c.subject_key)
                kept.append(c)
        out[cat] = kept
    return out


def window_frames(c: Candidate, n_frames: int, fps: float = FPS) -> tuple[int, int]:
    """Frame range [f0, f1) of the window: the start of the segment, at most MAX_WINDOW_S long, inside the clip."""
    f0 = max(0, int(np.floor(c.start_s * fps)))
    f1 = min(n_frames, int(np.ceil(c.end_s * fps)), f0 + round(MAX_WINDOW_S * fps))
    return f0, max(f0, f1)


def control_steps(n_frames: int, fps: float, ctrl_dt: float) -> int:
    """Number of control times k * ctrl_dt that fall inside a reference of n_frames frames at fps."""
    return int(np.floor((n_frames - 1) / fps / ctrl_dt + 1e-9)) + 1


def resample(arr: np.ndarray, t_ctrl: np.ndarray, fps: float, normalise: bool = False) -> np.ndarray:
    """Linear interpolation of (T, ...) frames at fps onto the control times; normalise for quaternions."""
    pos = t_ctrl * fps
    lo = np.clip(np.floor(pos).astype(int), 0, len(arr) - 1)
    hi = np.minimum(lo + 1, len(arr) - 1)
    w = (pos - lo).reshape((-1,) + (1,) * (arr.ndim - 1))
    out = (1 - w) * arr[lo] + w * arr[hi]
    return out / np.linalg.norm(out, axis=-1, keepdims=True) if normalise else out


def hinge_limits(meta: dict) -> np.ndarray:
    """(63, 2) lower and upper limit of every hinge, in the order of the SMPL-X body pose (joints 1..21, a b c each)."""
    return np.concatenate([np.asarray(meta["limits_rad"][meta["joint_names"][j]], dtype=float) for j in range(1, 22)], axis=0)


def _quat_series(root_rot: np.ndarray) -> np.ndarray:
    q = np.stack([H.matrix_to_quat_xyzw(r) for r in root_rot])
    for t in range(1, len(q)):
        if np.dot(q[t], q[t - 1]) < 0:
            q[t] = -q[t]
    return q


def build_reference(meta: dict, root_orient: np.ndarray, pose_body: np.ndarray, pelvis_pos: np.ndarray) -> dict:
    """root_orient (T,3) and pose_body (T,21,3) axis-angle, pelvis_pos (T,3) the subject's floor-grounded pelvis joint.
    Returns the reference arrays and statistics on the hinge clipping."""
    rest = np.asarray(meta["rest_joints"], dtype=float)
    parents = np.asarray(meta["parents"])
    lim = hinge_limits(meta)
    t_len = len(root_orient)
    raw = np.stack([H.pose_to_hinges(p) for p in pose_body])
    hing = np.clip(raw, lim[:, 0], lim[:, 1])
    clipped = np.abs(raw - hing) > 1e-6
    root_rot = H.aa_to_matrix(root_orient)
    local = np.empty((t_len, 22, 3, 3))
    low = np.empty(t_len)
    for t in range(t_len):
        local[t] = np.concatenate([np.eye(3)[None], H.hinges_to_matrix(hing[t].reshape(21, 3))])
        low[t] = H.lowest_height(meta, root_rot[t], local[t])
    pelvis = np.asarray(pelvis_pos, dtype=float).copy()
    pelvis[:, :2] -= pelvis[0, :2]
    offset = -float(np.min(pelvis[:, 2] + low)) + CLEARANCE_M
    root_pos = pelvis + np.array([0.0, 0.0, offset])
    joints = np.stack([H.forward_joints(rest, parents, root_rot[t], root_pos[t], local[t]) for t in range(t_len)])
    return {
        "root_pos": root_pos, "root_quat_xyzw": _quat_series(root_rot), "hinges": hing, "joints": joints,
        "lowest_m": low + root_pos[:, 2], "offset_m": offset,
        "stats": {"hinge_values_clipped_fraction": float(clipped.mean()), "frames_with_any_clipped_hinge": float(clipped.any(1).mean()),
                  "max_clip_rad": float(np.abs(raw - hing).max()), "offset_m": offset,
                  "root_height_min_m": float(root_pos[:, 2].min()), "root_height_max_m": float(root_pos[:, 2].max()),
                  "horizontal_path_m": float(np.linalg.norm(np.diff(root_pos[:, :2], axis=0), axis=1).sum())},
    }
