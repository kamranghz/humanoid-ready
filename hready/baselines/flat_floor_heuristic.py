"""E3 flat-floor ankle-offset heuristic (heading frame; no IK; nothing fitted to data).

Offsets come from the neutral locked_head rest skeleton (betas = 0), symmetrised left/right, with
the arms hanging (rest pose is a T-pose). They are expressed in the per-frame heading frame
(right, forward, up) and anchored at the given head position (eye midpoint, D1):

1. visible joints keep their evidence;
2. pelvis: evidence if visible, else head + template offset;
3. hips: pelvis + template hip offset (lateral / forward / vertical);
4. knees, ankles: under their hip with the template ankle offset in xy, at the template height
   above the flat floor (z = 0 given);
5. feet: ankle xy + template forward offset, template height;
6. any other hidden joint: head + template offset (upright body).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hready.body.joint_indices import (
    JOINT_INDEX,
    PELVIS,
    SMPLX_LEFT_EYE,
    SMPLX_RIGHT_EYE,
)
from hready.data.e3_heading import heading_sequence
from hready.metrics.e3_eval import proxy_foot_contact

NUM_KP = 22
HIPS = (JOINT_INDEX["left_hip"], JOINT_INDEX["right_hip"])
LEG_CHAINS = (  # (hip, knee, ankle, foot)
    (
        JOINT_INDEX["left_hip"],
        JOINT_INDEX["left_knee"],
        JOINT_INDEX["left_ankle"],
        JOINT_INDEX["left_foot"],
    ),
    (
        JOINT_INDEX["right_hip"],
        JOINT_INDEX["right_knee"],
        JOINT_INDEX["right_ankle"],
        JOINT_INDEX["right_foot"],
    ),
)
ARM_CHAINS = (  # (shoulder, elbow, wrist)
    (
        JOINT_INDEX["left_shoulder"],
        JOINT_INDEX["left_elbow"],
        JOINT_INDEX["left_wrist"],
    ),
    (
        JOINT_INDEX["right_shoulder"],
        JOINT_INDEX["right_elbow"],
        JOINT_INDEX["right_wrist"],
    ),
)
_MIRROR = {
    JOINT_INDEX[f"left_{n}"]: JOINT_INDEX[f"right_{n}"]
    for n in ("hip", "knee", "ankle", "foot", "collar", "shoulder", "elbow", "wrist")
}


@dataclass
class HeuristicConfig:
    heading_min_horiz: float = 0.15


@dataclass(frozen=True)
class HeuristicTemplate:
    """Per-joint offsets from the eye midpoint (right, forward, up) and heights above the floor, metres."""

    offset_rfu: np.ndarray  # (22, 3)
    height: np.ndarray  # (22,)

    def as_dict(self) -> dict[str, Any]:
        from hready.body.joint_indices import JOINT_NAMES_22

        return {
            n: {
                "right_fwd_up_from_eyes_m": [
                    round(float(x), 4) for x in self.offset_rfu[i]
                ],
                "height_above_floor_m": round(float(self.height[i]), 4),
            }
            for i, n in enumerate(JOINT_NAMES_22)
        }


def neutral_template(body: Any) -> HeuristicTemplate:
    """Template from the neutral rest mesh (SMPL-X rest: +Y up, +Z forward, subject's left = +X)."""
    from hready.data.contact import _neutral_pose_mesh

    v, j = _neutral_pose_mesh(body)
    floor = float(v[:, 1].min())
    eye = 0.5 * (j[SMPLX_LEFT_EYE] + j[SMPLX_RIGHT_EYE])
    d = j[:NUM_KP] - eye
    rfu = np.stack([-d[:, 0], d[:, 2], d[:, 1]], axis=-1).astype(np.float64)
    h = (j[:NUM_KP, 1] - floor).astype(np.float64)
    for (
        li,
        ri,
    ) in _MIRROR.items():  # symmetrise: mean |lateral|, mean forward / up / height
        lat = 0.5 * (abs(rfu[li, 0]) + abs(rfu[ri, 0]))
        fu = 0.5 * (rfu[li, 1:] + rfu[ri, 1:])
        hh = 0.5 * (h[li] + h[ri])
        rfu[li] = [-lat, *fu]
        rfu[ri] = [lat, *fu]
        h[li] = h[ri] = hh
    for s, e, w in ARM_CHAINS:  # arms hanging: segment lengths kept, pointing down
        upper = float(np.linalg.norm(j[e] - j[s]))
        fore = float(np.linalg.norm(j[w] - j[e]))
        rfu[e] = rfu[s] - [0.0, 0.0, upper]
        rfu[w] = rfu[e] - [0.0, 0.0, fore]
        h[e], h[w] = h[s] - upper, h[s] - upper - fore
    rfu[:, 0] = np.where(np.abs(rfu[:, 0]) < 1e-9, 0.0, rfu[:, 0])
    for i in range(NUM_KP):  # midline joints: no lateral offset
        if i not in _MIRROR and i not in _MIRROR.values():
            rfu[i, 0] = 0.0
    return HeuristicTemplate(
        offset_rfu=rfu.astype(np.float32), height=h.astype(np.float32)
    )


def _frame_axes(headings: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h = np.asarray(headings, dtype=np.float32)
    zero = np.zeros_like(h[:, 0])
    fwd = np.stack([h[:, 0], h[:, 1], zero], axis=-1)
    right = np.stack([h[:, 1], -h[:, 0], zero], axis=-1)
    up = np.broadcast_to(np.array([0.0, 0.0, 1.0], dtype=np.float32), fwd.shape)
    return right, fwd, up


def _offset(
    rfu: np.ndarray, right: np.ndarray, fwd: np.ndarray, up: np.ndarray
) -> np.ndarray:
    """``rfu`` (3,) in the heading frame -> world vectors ``(T, 3)``."""
    return rfu[0] * right + rfu[1] * fwd + rfu[2] * up


def clip_headings(
    obs_pos: np.ndarray,
    obs_vis: np.ndarray,
    head_pos: np.ndarray,
    camera_R: np.ndarray,
    tpl: HeuristicTemplate,
    cfg: HeuristicConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Clip-global headings (computed once per clip, then sliced per window).

    The pelvis->head fallback uses the observed pelvis only (a template pelvis would need a heading).
    """
    pelvis = np.where(obs_vis[:, PELVIS, None], obs_pos[:, PELVIS], head_pos)
    return heading_sequence(camera_R, head_pos, pelvis, min_horiz=cfg.heading_min_horiz)


def heuristic_joint_positions(
    obs_pos: np.ndarray,
    obs_vis: np.ndarray,
    head_pos: np.ndarray,
    headings: np.ndarray,
    tpl: HeuristicTemplate,
) -> np.ndarray:
    """Complete ``(T, 22, 3)`` joints from evidence + head pose + heading (rules 1-6, module doc)."""
    vis = np.asarray(obs_vis, dtype=bool)
    out = np.asarray(obs_pos, dtype=np.float32).copy()
    head = np.asarray(head_pos, dtype=np.float32)
    right, fwd, up = _frame_axes(headings)
    off = tpl.offset_rfu
    fill = np.zeros_like(out)
    fill[:] = head[:, None, :]
    for ji in range(NUM_KP):  # rule 6 default: upright template at the head
        fill[:, ji] += _offset(off[ji], right, fwd, up)
    pelvis = np.where(vis[:, PELVIS, None], out[:, PELVIS], fill[:, PELVIS])
    fill[:, PELVIS] = pelvis
    for hip, knee, ankle, foot in LEG_CHAINS:
        fill[:, hip] = pelvis + _offset(off[hip] - off[PELVIS], right, fwd, up)
        hip_pos = np.where(vis[:, hip, None], out[:, hip], fill[:, hip])
        for ji in (knee, ankle):
            p = hip_pos + _offset(off[ji] - off[hip], right, fwd, up)
            p[:, 2] = tpl.height[ji]
            fill[:, ji] = p
        ankle_pos = np.where(vis[:, ankle, None], out[:, ankle], fill[:, ankle])
        p = ankle_pos + _offset(off[foot] - off[ankle], right, fwd, up)
        p[:, 2] = tpl.height[foot]
        fill[:, foot] = p
    return np.where(vis[..., None], out, fill)


def heuristic_foot_contact(
    joints_22: np.ndarray, sole_offsets: np.ndarray
) -> np.ndarray:
    """Contact ``(T, 4)`` from the completed joints (item-5 rule on sole-proxy channel heights)."""
    return proxy_foot_contact(joints_22, sole_offsets)
