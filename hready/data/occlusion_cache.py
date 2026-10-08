"""E4 data path: per-frame self-occlusion masks precomputed once, E2-A evidence otherwise unchanged.

The E2-A capsule occlusion test depends only on the clean joints and head camera of each frame
and dominates simulation time. E4 stores it per clip (``<cache_dir>/<rel_dir>/clips/<clip>/occ.npy``,
``(T, 22)`` bool) and runs the unchanged ``simulate_oracle_evidence`` with
``self_occlusion_mask`` served from that array, so evidence is bit-identical to the E3 path
(``verify-occ`` checks this against the uncached simulator).
"""

from __future__ import annotations

import contextlib
import json
import multiprocessing as mp
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hready.data import ego_observation as eo
from hready.data.e3_dataset import (
    E3TrainDataset,
    EvidenceConfig,
    clip_noise_rng,
    simulate_evidence,
)
from hready.data.e3_motion_memmap import (
    cache_dir_from_config,
    clip_dir,
    load_clip_memmap,
)

_W: dict[str, Any] = {}


def occ_root(rel_dir: str) -> Path:
    return (cache_dir_from_config() / rel_dir).resolve()


def clip_occlusion(
    joints_22: np.ndarray, joints_55: np.ndarray, occ_cfg: eo.OcclusionCapsuleConfig
) -> np.ndarray:
    """Same inputs as inside ``simulate_oracle_evidence`` (float32 joints, eye-midpoint camera)."""
    j22 = torch.from_numpy(np.ascontiguousarray(joints_22, dtype=np.float32))
    j55 = torch.from_numpy(np.ascontiguousarray(joints_55, dtype=np.float32))
    _, _, cam = eo.head_frame_from_joints(j55)
    return eo.self_occlusion_mask(j22.numpy(), cam.numpy(), occ_cfg)


def _init(motion_dir: str, out_root: str, occ_cfg: dict[str, Any]) -> None:
    torch.set_num_threads(1)
    _W.update(
        motion_dir=motion_dir,
        out_root=Path(out_root),
        occ=eo.occlusion_config_from_dict(occ_cfg),
    )


def _build_one(rel: str) -> tuple[str, int, str | None]:
    path = clip_dir(_W["out_root"], rel) / "occ.npy"
    if path.is_file():
        return rel, -1, None
    try:
        clip = load_clip_memmap(rel, rel_dir=_W["motion_dir"])
        occ = clip_occlusion(clip["joints_22"], clip["joints_55"], _W["occ"])
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, occ)
        tmp.replace(path)
        return rel, int(occ.shape[0]), None
    except Exception as exc:  # noqa: BLE001 - recorded in the build summary
        return rel, 0, f"{type(exc).__name__}: {exc}"


def build_occlusion_cache(
    rels: list[str],
    *,
    motion_dir: str,
    rel_dir: str,
    occ_cfg: dict[str, Any],
    workers: int,
) -> dict[str, Any]:
    out_root = occ_root(rel_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    built = skipped = 0
    errors: list[str] = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        workers, initializer=_init, initargs=(motion_dir, str(out_root), occ_cfg)
    ) as pool:
        for i, (rel, n, err) in enumerate(
            pool.imap_unordered(_build_one, rels, chunksize=8)
        ):
            if err:
                errors.append(f"{rel}: {err}")
            elif n < 0:
                skipped += 1
            else:
                built += 1
            if (i + 1) % 1000 == 0 or i + 1 == len(rels):
                print(
                    f"build-occ {i + 1}/{len(rels)} elapsed_s={time.perf_counter() - t0:.0f}",
                    flush=True,
                )
    out = {
        "n_requested": len(rels),
        "n_built_now": built,
        "n_skipped_existing": skipped,
        "n_errors": len(errors),
        "errors": errors[:20],
        "wall_s": time.perf_counter() - t0,
    }
    (out_root / "build.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def load_occ(rel: str, rel_dir: str) -> np.ndarray:
    return np.load(clip_dir(occ_root(rel_dir), rel) / "occ.npy", mmap_mode="r")


@contextlib.contextmanager
def cached_occlusion(occ: np.ndarray) -> Iterator[None]:
    """Serve ``self_occlusion_mask`` from ``occ`` for the duration of one simulation call."""
    orig = eo.self_occlusion_mask

    def _lookup(joints_22: np.ndarray, cam_pos: np.ndarray, cfg: Any) -> np.ndarray:
        if joints_22.shape[0] != occ.shape[0]:
            raise ValueError(
                f"occlusion cache length {occ.shape[0]} != frames {joints_22.shape[0]}"
            )
        return np.array(occ, dtype=bool)

    eo.self_occlusion_mask = _lookup
    try:
        yield
    finally:
        eo.self_occlusion_mask = orig


def clip_world_evidence_cached(
    rel: str,
    *,
    motion_dir: str,
    occ_dir: str,
    ev_cfg: EvidenceConfig,
    eval_noise_seed: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """E3 eval evidence (same per-clip seed) with cached occlusion."""
    clip = load_clip_memmap(rel, rel_dir=motion_dir)
    with cached_occlusion(load_occ(rel, occ_dir)):
        obs, rig = simulate_evidence(
            clip["joints_22"],
            clip["joints_55"],
            clip_noise_rng(rel, eval_noise_seed),
            ev_cfg,
        )
    return clip, obs, rig


class E4TrainDataset(E3TrainDataset):
    """E3 training windows (same sampler, noise, canonical frame, z-rotation) with cached occlusion."""

    def __init__(self, *args: Any, occ_dir: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.occ_dir = occ_dir

    def __getitem__(self, key: tuple[int, int] | int) -> dict[str, Any]:
        wi = key[0] if isinstance(key, tuple) else key
        rel, start = self.windows[wi]
        clip = self._clip(rel)
        end = min(start + self.window, clip["T"])
        with cached_occlusion(load_occ(rel, self.occ_dir)[start:end]):
            return super().__getitem__(key)
