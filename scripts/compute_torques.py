"""Joint torques (recursive Newton-Euler) of raw vs HR-Refine-refined motion on grounded_motion_30hz clips.

    python scripts/compute_torques.py --out results/B2/torque_comparison.npz

For each of ``--clips`` seeded validation clips, ``--windows`` consecutive 64-frame windows around the clip
middle are taken. Three versions of each window go through :func:`hready.dynamics.inverse_dynamics`:

- ``raw``: the corrupted input the refiner receives (the standard HR-Refine corruption, fixed seed);
- ``refined``: the output of the ``video_only`` auxiliary-heads checkpoint (2000 steps) on that input;
- ``clean``: the cached AMASS motion (reference).

The ground-force split uses the cache's foot-contact labels for all three versions, so the versions differ
only in the motion.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from hready.body.batch_forward import smpl_forward_bt
from hready.data.motion_cache import load_clip_memmap
from hready.data.refine_corrupt import corruption_config_from_dict, unpack_pose
from hready.data.refine_dataset import MOTION_CACHE_DIR, _eligible_entries
from hready.dynamics.inverse_dynamics import (
    JOINT_NAMES,
    NUM_JOINTS,
    global_rotations,
    inverse_dynamics,
    mass_table_sum,
)
from hready.train.hr_refine_engine import build_model_and_body, prepare_fixed_corruption, repo_root

WINDOW = 64
VERSIONS = ("raw", "refined", "clean")


def _window_batch(rel_path: str, windows: int, device: torch.device) -> tuple[dict[str, torch.Tensor], list[int], float]:
    clip = load_clip_memmap(rel_path, rel_dir=MOTION_CACHE_DIR)
    n = int(clip["T"])
    first = max(0, (n - windows * WINDOW) // 2)
    starts = [first + k * WINDOW for k in range(windows)]
    sl = lambda key, shape=None: np.stack([np.array(clip[key][s : s + WINDOW]) for s in starts])
    batch = {
        "root_orient": torch.as_tensor(sl("root_orient"), dtype=torch.float32),
        "pose_body": torch.as_tensor(sl("pose_body").reshape(windows, WINDOW, 21, 3), dtype=torch.float32),
        "transl": torch.as_tensor(sl("transl"), dtype=torch.float32),
        "contact": torch.as_tensor(sl("contact"), dtype=torch.float32),
        "betas": torch.as_tensor(np.stack([np.array(clip["betas"])] * windows), dtype=torch.float32),
    }
    for key, v in batch.items():
        if not torch.isfinite(v).all():
            raise ValueError(f"non-finite values in {key} of clip {rel_path}")
    return {k: v.to(device) for k, v in batch.items()}, starts, float(clip["fps"])


def _select_clips(split: str, n_clips: int, seed: int, min_frames: int) -> list[str]:
    entries = _eligible_entries(split)
    rng = random.Random(seed)
    rng.shuffle(entries)
    out = []
    for e in entries:
        if int(load_clip_memmap(e.rel_path, rel_dir=MOTION_CACHE_DIR)["T"]) >= min_frames:
            out.append(e.rel_path)
        if len(out) == n_clips:
            break
    return out


def compute(
    checkpoint: Path, config: Path, n_clips: int = 3, windows: int = 4, seed: int = 0, split: str = "val",
    total_mass: float = 70.0,
) -> dict[str, Any]:
    """Run the refiner and the inverse dynamics; returns the arrays that are saved to the npz."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if ck["condition"] != "video_only":
        raise ValueError(f"expected the video_only checkpoint, got {ck['condition']}")
    model_cfg = {**cfg["model"], "imu_dim": 0, "contact_channels": int(cfg["contact_channels"]),
                 "n_actions": int(ck["model"]["head_action.weight"].shape[0]),
                 "intent_horizon": int(cfg["intent_horizon"])}
    model, body, parents = build_model_and_body(model_cfg, device)
    model.load_state_dict(ck["model"])
    model.eval()
    parents22 = parents[:NUM_JOINTS].cpu().numpy()
    ccfg = corruption_config_from_dict(cfg["corruption"])
    clips = _select_clips(split, n_clips, seed, min_frames=windows * WINDOW)
    if len(clips) < n_clips:
        raise RuntimeError(f"only {len(clips)} {split} clips with >= {windows * WINDOW} frames")
    out: dict[str, list] = {f"{k}_{v}": [] for k in ("torque", "ground_force", "wrench_residual") for v in VERSIONS}
    out.update({"clip_index": [], "window_start": []})
    out.update({f"{m}_{v}": [] for m in ("mpjpe_mm", "root_ade_m") for v in ("raw", "refined")})
    fps = 30.0
    for ci, rel in enumerate(clips):
        batch, starts, fps = _window_batch(rel, windows, device)
        fixed = prepare_fixed_corruption(body, batch, base_seed=seed + 5000, corrupt_step=ci, corrupt_cfg=ccfg)
        with torch.no_grad():
            pred = model(fixed.corrupt_pose, fixed.keypoints, imu=None)
            poses = {"raw": unpack_pose(fixed.corrupt_pose), "refined": unpack_pose(pred),
                     "clean": (batch["transl"], batch["root_orient"], batch["pose_body"])}
            joints = {v: smpl_forward_bt(body, *poses[v], batch["betas"])[0][..., :NUM_JOINTS, :] for v in VERSIONS}
        for v in ("raw", "refined"):
            rel = joints[v] - joints[v][..., :1, :]
            rel_clean = joints["clean"] - joints["clean"][..., :1, :]
            out[f"mpjpe_mm_{v}"].append((rel - rel_clean).norm(dim=-1).mean(dim=(1, 2)).cpu().numpy() * 1000.0)
            out[f"root_ade_m_{v}"].append((poses[v][0] - batch["transl"]).norm(dim=-1).mean(dim=-1).cpu().numpy())
        for w in range(windows):
            for v in VERSIONS:
                rot = global_rotations(poses[v][1][w], poses[v][2][w], parents[:NUM_JOINTS]).cpu().numpy()
                res = inverse_dynamics(joints[v][w].double().cpu().numpy(), rot, batch["contact"][w].cpu().numpy(),
                                       fps, parents22, total_mass)
                out[f"torque_{v}"].append(res.torque)
                out[f"ground_force_{v}"].append(res.ground_force)
                out[f"wrench_residual_{v}"].append(res.wrench_residual)
            out["clip_index"].append(ci)
            out["window_start"].append(starts[w])
    arrays: dict[str, Any] = {
        k: np.stack(v) if k.startswith(("torque", "ground", "wrench")) else
        np.concatenate(v) if k.startswith(("mpjpe", "root_ade")) else np.asarray(v)
        for k, v in out.items()
    }
    arrays.update({
        "clip_names": np.asarray(clips), "joint_names": np.asarray(JOINT_NAMES), "fps": np.asarray(fps),
        "total_mass_kg": np.asarray(total_mass), "mass_table_fraction_sum": np.asarray(mass_table_sum()),
        "checkpoint": np.asarray(str(checkpoint)), "seed": np.asarray(seed), "split": np.asarray(split),
    })
    return arrays


def main(argv: list[str] | None = None) -> None:
    root = repo_root()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=root / "configs" / "hr_refine_auxiliary_heads.yaml")
    ap.add_argument("--checkpoint", type=Path, default=root / "results" / "auxiliary_heads" / "video_only" / "last.pt")
    ap.add_argument("--out", type=Path, default=root / "results" / "B2" / "torque_comparison.npz")
    ap.add_argument("--clips", type=int, default=3)
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--total-mass", type=float, default=70.0)
    args = ap.parse_args(argv)
    arrays = compute(args.checkpoint, args.config, args.clips, args.windows, args.seed, total_mass=args.total_mass)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    print(f"wrote {args.out}: {len(arrays['clip_index'])} windows from {list(arrays['clip_names'])}")


if __name__ == "__main__":
    main()
