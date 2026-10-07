"""E3 oracle windows: memmap motion -> E2-A evidence + rig -> canonical frame (+ train z-rotation).

Canonical frame: subtract the window-start head xy from every world xy (evidence, head, targets);
z is kept (floor at z = 0 is given). Training may rotate obs, rig and targets together about z.
Targets never enter the evidence path; they are returned in a separate dict for the loss only.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from hready.body.rotations import axis_angle_to_matrix
from hready.data.e3_motion_memmap import load_clip_memmap
from hready.data.ego_observation import (
    EgoCameraConfig,
    EgoNoiseConfig,
    OcclusionCapsuleConfig,
    simulate_oracle_evidence,
)

NUM_KP = 22
NUM_BODY = 21
EYES = (23, 24)


@dataclass(frozen=True)
class EvidenceConfig:
    cam: EgoCameraConfig
    noise: EgoNoiseConfig
    occ: OcclusionCapsuleConfig


def window_starts(t_len: int, window: int, stride: int) -> list[int]:
    """Starts covering ``[0, t_len)``; the last window is aligned to the clip end."""
    if t_len <= window:
        return [0]
    starts = list(range(0, t_len - window + 1, stride))
    if starts[-1] != t_len - window:
        starts.append(t_len - window)
    return starts


def clip_noise_rng(rel_path: str, eval_noise_seed: int) -> np.random.Generator:
    """Process-independent per-clip RNG (``hash()`` is salted per process; sha256 is not)."""
    key = int(hashlib.sha256(rel_path.encode("utf-8")).hexdigest()[:8], 16)
    return np.random.default_rng([int(eval_noise_seed), key])


def simulate_evidence(
    joints_22: np.ndarray,
    joints_55: np.ndarray,
    rng: np.random.Generator,
    ev_cfg: EvidenceConfig,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """World-frame E2-A evidence for a contiguous frame range."""
    return simulate_oracle_evidence(
        torch.from_numpy(np.ascontiguousarray(joints_22, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(joints_55, dtype=np.float32)),
        rng=rng,
        cam_cfg=ev_cfg.cam,
        noise_cfg=ev_cfg.noise,
        occ_cfg=ev_cfg.occ,
    )


def _pad(x: Tensor, window: int) -> Tensor:
    if x.shape[0] >= window:
        return x[:window]
    return torch.cat([x, x[-1:].expand(window - x.shape[0], *x.shape[1:])], dim=0)


def slice_window(
    obs: dict[str, Tensor], rig: dict[str, Tensor], start: int, window: int
) -> tuple[dict[str, Tensor], dict[str, Tensor], Tensor]:
    """Slice world evidence ``[start, start+window)``, pad by repeating the last frame."""
    t_len = obs["joint_pos_3d"].shape[0]
    n = min(window, t_len - start)
    o = {k: _pad(v[start : start + n], window) for k, v in obs.items()}
    r = {
        k: (_pad(v[start : start + n], window) if v.ndim >= 2 else v.clone())
        for k, v in rig.items()
    }
    valid = torch.zeros(window, dtype=torch.bool)
    valid[:n] = True
    return o, r, valid


def canonicalize(obs: dict[str, Tensor], rig: dict[str, Tensor]) -> Tensor:
    """In place: subtract window-start head xy from evidence + head; returns ``origin_xy`` (2,)."""
    origin = rig["head_pos_world"][0, :2].clone()
    shift = torch.zeros(3, dtype=origin.dtype)
    shift[:2] = origin
    vis = obs["joint_visible"].unsqueeze(-1)
    obs["joint_pos_3d"] = torch.where(
        vis, obs["joint_pos_3d"] - shift, torch.zeros_like(obs["joint_pos_3d"])
    )
    rig["head_pos_world"] = rig["head_pos_world"] - shift
    rig["camera_t"] = -torch.einsum(
        "tij,tj->ti", rig["camera_R"], rig["head_pos_world"]
    )
    return origin


def z_rotation(angle: float) -> Tensor:
    c, s = float(np.cos(angle)), float(np.sin(angle))
    return torch.tensor(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
    )


def rotate_about_z(
    obs: dict[str, Tensor],
    rig: dict[str, Tensor],
    targets: dict[str, Tensor] | None,
    rz: Tensor,
) -> None:
    """In place world rotation ``p' = Rz p`` of evidence, rig and (optionally) targets.

    ``camera_R`` maps world->camera, so ``R' = R Rz^T``; ``camera_t`` and ``keypoints_2d`` are invariant.
    """
    obs["joint_pos_3d"] = obs["joint_pos_3d"] @ rz.T
    rig["head_pos_world"] = rig["head_pos_world"] @ rz.T
    rig["camera_R"] = rig["camera_R"] @ rz.T
    if targets is None:
        return
    for k in ("joints_gt_22", "pelvis"):
        targets[k] = targets[k] @ rz.T
    targets["root_R"] = rz @ targets["root_R"]


class E3TrainSampler(Sampler):
    """Infinite seeded window stream yielding ``(window_index, draw_id)``; resumable by draw count."""

    def __init__(self, n_windows: int, seed: int, start_draw: int = 0) -> None:
        self.n = n_windows
        self.seed = seed
        self.start_draw = start_draw

    def __iter__(self) -> Iterator[tuple[int, int]]:
        draw = self.start_draw
        epoch, pos = divmod(draw, self.n)
        while True:
            perm = np.random.default_rng([self.seed, epoch]).permutation(self.n)
            for i in range(pos, self.n):
                yield int(perm[i]), draw
                draw += 1
            epoch, pos = epoch + 1, 0


class E3TrainDataset(Dataset):
    """Training windows: evidence simulated per draw (fresh noise each draw), z-rotation augmentation."""

    def __init__(
        self,
        windows: list[tuple[str, int]],
        *,
        window: int,
        rel_dir: str,
        ev_cfg: EvidenceConfig,
        seed: int,
        z_rot_max_rad: float,
        j0_neutral: np.ndarray,
        contact_valid: dict[str, bool],
        lru_clips: int = 128,
    ) -> None:
        self.windows = windows
        self.window = window
        self.rel_dir = rel_dir
        self.ev_cfg = ev_cfg
        self.seed = seed
        self.z_rot_max_rad = float(z_rot_max_rad)
        self.j0_neutral = torch.as_tensor(j0_neutral, dtype=torch.float32)
        self.contact_valid = contact_valid
        self.lru_clips = lru_clips
        self._lru: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.windows)

    def _clip(self, rel: str) -> dict[str, Any]:
        if rel in self._lru:
            self._lru.move_to_end(rel)
            return self._lru[rel]
        clip = load_clip_memmap(rel, rel_dir=self.rel_dir)
        self._lru[rel] = clip
        if len(self._lru) > self.lru_clips:
            self._lru.popitem(last=False)
        return clip

    def __getitem__(self, key: tuple[int, int] | int) -> dict[str, Any]:
        wi, draw = key if isinstance(key, tuple) else (key, key)
        rel, start = self.windows[wi]
        rng = np.random.default_rng([self.seed, int(draw)])
        clip = self._clip(rel)
        end = min(start + self.window, clip["T"])
        obs, rig = simulate_evidence(
            clip["joints_22"][start:end], clip["joints_55"][start:end], rng, self.ev_cfg
        )
        obs, rig, valid = slice_window(obs, rig, 0, self.window)
        origin = canonicalize(obs, rig)
        shift = torch.cat([origin, torch.zeros(1)])
        sl = slice(start, end)
        joints = (
            _pad(
                torch.from_numpy(np.array(clip["joints_22"][sl], dtype=np.float32)),
                self.window,
            )
            - shift
        )
        root_aa = _pad(
            torch.from_numpy(np.array(clip["root_orient"][sl], dtype=np.float32)),
            self.window,
        )
        body_aa = _pad(
            torch.from_numpy(np.array(clip["pose_body"][sl], dtype=np.float32)),
            self.window,
        )
        contact = _pad(torch.from_numpy(np.array(clip["contact"][sl])), self.window)
        targets = {
            "joints_gt_22": joints,
            "pelvis": joints[:, 0].clone(),
            "root_R": axis_angle_to_matrix(root_aa),
            "body_aa": body_aa.reshape(self.window, NUM_BODY, 3),
            "contact": contact.float(),
        }
        if self.z_rot_max_rad > 0:
            rotate_about_z(
                obs,
                rig,
                targets,
                z_rotation(float(rng.uniform(-self.z_rot_max_rad, self.z_rot_max_rad))),
            )
        # transl for the neutral-shape body (betas = 0): pelvis = transl + J0_neutral.
        targets["transl"] = targets.pop("pelvis") - self.j0_neutral
        return {
            "obs": obs,
            "rig": rig,
            "targets": targets,
            "valid": valid,
            "contact_valid": torch.tensor(self.contact_valid[rel]),
        }


def collate_e3(items: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for group in ("obs", "rig", "targets"):
        out[group] = {
            k: torch.stack([it[group][k] for it in items]) for k in items[0][group]
        }
    out["valid"] = torch.stack([it["valid"] for it in items])
    out["contact_valid"] = torch.stack([it["contact_valid"] for it in items])
    return out


def eval_window_batch(
    obs_w: dict[str, Tensor], rig_w: dict[str, Tensor], starts: list[int], window: int
) -> tuple[dict[str, Tensor], dict[str, Tensor], Tensor, Tensor]:
    """Canonical windows of one clip's world evidence: obs, rig ``(N, W, ...)``, origins ``(N, 2)``, valid."""
    obs_l, rig_l, org_l, val_l = [], [], [], []
    for s in starts:
        o, r, v = slice_window(obs_w, rig_w, s, window)
        org_l.append(canonicalize(o, r))
        obs_l.append(o)
        rig_l.append(r)
        val_l.append(v)
    obs = {k: torch.stack([o[k] for o in obs_l]) for k in obs_l[0]}
    rig = {k: torch.stack([r[k] for r in rig_l]) for k in rig_l[0]}
    return obs, rig, torch.stack(org_l), torch.stack(val_l)
