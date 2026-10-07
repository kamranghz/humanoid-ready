"""Ego heading in the horizontal plane (Track E3 heuristic)."""

from __future__ import annotations

import numpy as np

_FALLBACK_WORLD = np.array([0.0, 1.0], dtype=np.float64)

# Per-frame heading source codes returned by ``heading_sequence``.
SOURCE_LOOK, SOURCE_HOLD, SOURCE_PELVIS_HEAD, SOURCE_WORLD_Y, SOURCE_BACKFILL = (
    0,
    1,
    2,
    3,
    4,
)


def heading_sequence(
    camera_R: np.ndarray,
    head_pos: np.ndarray,
    pelvis_pos: np.ndarray,
    *,
    min_horiz: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Unit xy heading ``(T, 2)`` and source code ``(T,)``.

    ``camera_R`` rows are camera axes in world (E2-A: world->camera), so row 2 is the look axis.
    If ``|look_xy| < min_horiz``: hold the last stable heading, else pelvis->head xy, else the first
    later stable heading (clip start), else world +y. Only the last fallback depends on the world axes.
    """
    look_xy = np.asarray(camera_R[:, 2, :2], dtype=np.float64)
    norm = np.linalg.norm(look_xy, axis=-1)
    d_xy = np.asarray(head_pos[:, :2] - pelvis_pos[:, :2], dtype=np.float64)
    d_norm = np.linalg.norm(d_xy, axis=-1)
    t_len = look_xy.shape[0]
    out = np.zeros((t_len, 2), dtype=np.float64)
    src = np.zeros(t_len, dtype=np.int8)
    last: np.ndarray | None = None
    for ti in range(t_len):
        if norm[ti] >= min_horiz:
            h, s = look_xy[ti] / norm[ti], SOURCE_LOOK
        elif last is not None:
            h, s = last, SOURCE_HOLD
        elif d_norm[ti] >= min_horiz:
            h, s = d_xy[ti] / d_norm[ti], SOURCE_PELVIS_HEAD
        else:
            h, s = _FALLBACK_WORLD, SOURCE_WORLD_Y
        out[ti], src[ti] = h, s
        if s != SOURCE_WORLD_Y:
            last = h
    stable = np.flatnonzero(src != SOURCE_WORLD_Y)
    if stable.size and stable[0] > 0:
        out[: stable[0]] = out[stable[0]]
        src[: stable[0]] = SOURCE_BACKFILL
    return out, src
