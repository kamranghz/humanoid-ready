"""Write the egocentric render set (E2-B1): flat-colour head-camera images of AMASS clips plus 2D joint labels.

Per clip: frames at a fixed stride (offset fixed by the clip path and the seed), rendered with the torch rasteriser
(hready/data/ego_render.py), saved as PNG, with per-clip labels (joint pixel position, depth, in-image flag, z-buffer
visibility). Split comes from the AMASS index; a clip is in exactly one split. Floor height per clip is the 1st
percentile of the per-frame lowest vertex over the rendered frames (the repo's floor rule on the sampled frames).

Images are not committed (out-root is outside the repo). A finished clip has its labels file; a second run skips it.

    python scripts/ego_render_set.py --profile aria_rect --max-clips-per-split 20 --out-root D:\\...\\ego_render_trial
    python scripts/ego_render_set.py --profile aria_rect          # full set, default stride train 20, val/test 30
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from hready.body.smplx_wrapper import load_body
from hready.data import ego_render as R
from hready.data.amass import load_clip, load_index, load_paths_config
from hready.data.ego_observation import head_frame_from_joints

SCRIPT_VERSION = 1
SPLITS = ("train", "val", "test")


def clip_key(e) -> str:
    return f"{e.subset}__{e.clip_id}__{hashlib.sha1(e.rel_path.encode()).hexdigest()[:8]}"


def clip_seed(e, seed: int) -> int:
    return int(hashlib.sha1(f"{e.rel_path}|{seed}".encode()).hexdigest()[:8], 16)


def frame_ids(n_frames: int, stride: int, offset: int) -> np.ndarray:
    return np.arange(offset % stride, n_frames, stride)


@torch.no_grad()
def forward_frames(body, clip, ids, device, chunk=64):
    nb = body.num_betas
    verts, joints = [], []
    for s in range(0, len(ids), chunk):
        i = ids[s : s + chunk]
        go = torch.as_tensor(clip["root_orient"][i], dtype=torch.float32, device=device)
        bp = torch.as_tensor(clip["pose_body"][i].reshape(len(i), -1), dtype=torch.float32, device=device)
        tr = torch.as_tensor(clip["transl"][i], dtype=torch.float32, device=device)
        be = torch.as_tensor(clip["betas"][:nb], dtype=torch.float32, device=device).unsqueeze(0).expand(len(i), -1)
        out = body.forward(go, bp, be, tr)
        verts.append(out.vertices)
        joints.append(out.joints)
    return torch.cat(verts), torch.cat(joints), out.faces.to(device)


@torch.no_grad()
def render_clip(body, weights, e, clip, ids, prof, out_dir: Path, device, seed: int, pool) -> dict:
    v, j, faces = forward_frames(body, clip, ids, device)
    up = np.array([0.0, 0.0, 1.0])  # AMASS is z-up
    floor_h = float(np.percentile(v[:, :, 2].min(dim=1).values.cpu().numpy(), 1))
    floor = R.Floor(up=up, height=floor_h)
    freg = R.face_regions(weights, faces)
    r, t, cp = head_frame_from_joints(j)
    cseed = clip_seed(e, seed) % 1000003
    out_dir.mkdir(parents=True, exist_ok=True)
    uv, zz, inim, vis, bfrac, futures = [], [], [], [], [], []
    for k in range(len(ids)):
        rcv, tcv = R.cam_cv_from_head_frame(r[k], cp[k])
        o = R.render_frame(v[k], faces, freg, rcv, tcv, cp[k], prof, floor, seed=cseed)
        jv = R.joint_visibility(j[k, :22], rcv, tcv, prof, o["body_depth"])
        uv.append(jv["uv"]); zz.append(jv["z"]); inim.append(jv["in_image"]); vis.append(jv["visible"])
        bfrac.append(float((o["region"] < R.NUM_REGIONS).mean()))
        futures.append(pool.submit(Image.fromarray(o["rgb"]).save, out_dir / f"{int(ids[k]):06d}.png", compress_level=3))
    for f in futures:
        f.result()
    return {
        "frame": ids.astype(np.int32), "uv": np.stack(uv).astype(np.float32), "z": np.stack(zz).astype(np.float32),
        "in_image": np.stack(inim), "visible": np.stack(vis), "body_frac": np.asarray(bfrac, np.float32),
        "floor_height": np.float32(floor_h),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="aria_rect", choices=sorted(R.PROFILES))
    ap.add_argument("--stride-train", type=int, default=20)
    ap.add_argument("--stride-val", type=int, default=30)
    ap.add_argument("--stride-test", type=int, default=30)
    ap.add_argument("--max-clips-per-split", type=int, default=0, help="trial: seeded subset of clips per split")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)

    cfg = load_paths_config()
    out_root = args.out_root or Path(cfg["data_root"]) / "ego_render" / args.profile
    prof = R.PROFILES[args.profile]
    strides = {"train": args.stride_train, "val": args.stride_val, "test": args.stride_test}
    device = torch.device(args.device)
    body = load_body("locked_head")
    body._model.to(device)
    weights = body._model.lbs_weights.to(device)
    index = [e for e in load_index() if e.split in SPLITS]

    by_split = defaultdict(list)
    for e in sorted(index, key=lambda x: x.rel_path):
        by_split[e.split].append(e)
    if args.max_clips_per_split:
        rng = np.random.default_rng(args.seed)
        for s in SPLITS:
            items = by_split[s]
            pick = rng.choice(len(items), size=min(args.max_clips_per_split, len(items)), replace=False)
            by_split[s] = [items[int(i)] for i in sorted(pick)]
    keys = defaultdict(set)  # leak check: no clip key in two splits
    for s in SPLITS:
        for e in by_split[s]:
            keys[clip_key(e)].add(s)
    assert all(len(v) == 1 for v in keys.values()), "a clip is in two splits"

    manifest_path = out_root / "manifest.json"
    total = sum(len(v) for v in by_split.values())
    print(f"profile {args.profile} {prof.width}x{prof.height}, {total} clips, strides {strides}, device {device}, out {out_root}")
    t0, done_frames, n = time.time(), 0, 0
    clips_meta = []
    with ThreadPoolExecutor(args.threads) as pool:
        for s in SPLITS:
            for e in by_split[s]:
                n += 1
                key = clip_key(e)
                lab_path = out_root / "labels" / s / f"{key}.npz"
                img_dir = out_root / "images" / s / key
                if lab_path.exists():
                    meta_n = int(np.load(lab_path)["frame"].shape[0])
                else:
                    clip = load_clip(e)
                    ids = frame_ids(clip["root_orient"].shape[0], strides[s], clip_seed(e, args.seed))
                    if len(ids) == 0:
                        continue
                    lab = render_clip(body, weights, e, clip, ids, prof, img_dir, device, args.seed, pool)
                    lab_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(lab_path, **lab)
                    meta_n = len(ids)
                done_frames += meta_n
                clips_meta.append({"key": key, "split": s, "rel_path": e.rel_path, "subset": e.subset, "subject": e.subject, "frames": meta_n})
                if n % 20 == 0 or n == total:
                    print(f"  {n}/{total} clips, {done_frames} frames, {time.time() - t0:.0f} s", flush=True)

    counts = {s: sum(c["frames"] for c in clips_meta if c["split"] == s) for s in SPLITS}
    subj = {s: {c["subject"] for c in clips_meta if c["split"] == s} for s in SPLITS}
    manifest = {
        "script_version": SCRIPT_VERSION, "profile": args.profile, "width": prof.width, "height": prof.height,
        "strides": strides, "seed": args.seed, "max_clips_per_split": args.max_clips_per_split,
        "frames_per_split": counts, "clips_per_split": {s: sum(c["split"] == s for c in clips_meta) for s in SPLITS},
        "subjects_in_more_than_one_split": sorted((subj["train"] & subj["val"]) | (subj["train"] & subj["test"]) | (subj["val"] & subj["test"])),
        "floor_rule": "1st percentile of the per-frame lowest vertex z over the rendered frames of the clip",
        "labels": "labels/<split>/<clip key>.npz: frame, uv (T,22,2) pixels, z (T,22), in_image, visible (z-buffer, 0.10 m), body_frac, floor_height",
        "clips": clips_meta,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print("frames per split", counts, "| subjects in more than one split:", len(manifest["subjects_in_more_than_one_split"]))
    print("written", manifest_path)


if __name__ == "__main__":
    main()
