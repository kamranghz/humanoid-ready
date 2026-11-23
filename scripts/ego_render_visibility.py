"""How much of the body does the head camera see? (E2-B1, sizing of the render set)

For a seeded sample of AMASS clips, render the body with the torch rasteriser from
the head camera and count, per camera profile, how often joints / hands / feet are
visible by the z-buffer rule and how much of the image the body covers.

Nothing is written to the repo except one JSON. Run in the hready env:

    python scripts/ego_render_visibility.py --clips-per-subset 40 --frames-per-clip 30

Output: results/E/perception/render_visibility.json (path settable with --out).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from hready.body.smplx_wrapper import load_body
from hready.data import ego_render as R
from hready.data.amass import load_clip, load_index
from hready.data.ego_observation import head_frame_from_joints

PROFILES = ("hl2_like", "ego_wide")
GROUPS = {
    "wrists": (20, 21),
    "arms": (18, 19, 20, 21),
    "feet": (7, 8, 10, 11),
    "legs": (1, 2, 4, 5, 7, 8, 10, 11),
}
SCRIPT_VERSION = 1


def sample_clips(index, per_subset: int, seed: int, min_frames: int):
    by_subset = defaultdict(list)
    for e in index:
        if e.n_frames and e.duration * 30.0 >= min_frames:
            by_subset[e.subset].append(e)
    rng = np.random.default_rng(seed)
    out = []
    for sub in sorted(by_subset):
        items = sorted(by_subset[sub], key=lambda e: e.rel_path)
        take = min(per_subset, len(items))
        pick = rng.choice(len(items), size=take, replace=False)
        out.extend(items[int(i)] for i in sorted(pick))
    return out


def frames_of_clip(clip, n: int) -> np.ndarray:
    t = clip["root_orient"].shape[0]
    return np.unique(np.linspace(0, t - 1, num=min(n, t)).round().astype(int))


@torch.no_grad()
def count_clip(body, weights, clip, frame_ids, profiles, device, chunk=32):
    """Return {profile: dict of arrays over the sampled frames}."""
    res = {p: {"vis": [], "body_frac": []} for p in profiles}
    nb = body.num_betas
    freg = None
    for s in range(0, len(frame_ids), chunk):
        ids = frame_ids[s : s + chunk]
        go = torch.as_tensor(clip["root_orient"][ids], dtype=torch.float32, device=device)
        bp = torch.as_tensor(clip["pose_body"][ids].reshape(len(ids), -1), dtype=torch.float32, device=device)
        tr = torch.as_tensor(clip["transl"][ids], dtype=torch.float32, device=device)
        be = torch.as_tensor(clip["betas"][:nb], dtype=torch.float32, device=device).unsqueeze(0).expand(len(ids), -1)
        out = body.forward(go, bp, be, tr)
        v, j, f = out.vertices, out.joints, out.faces.to(device)
        if freg is None:
            freg = R.face_regions(weights, f)
        r, t, cp = head_frame_from_joints(j)
        for k in range(len(ids)):
            rcv, tcv = R.cam_cv_from_head_frame(r[k], cp[k])
            for p in profiles:
                prof = R.PROFILES[p]
                o = R.render_frame(v[k], f, freg, rcv, tcv, cp[k], prof, None)
                vis = R.joint_visibility(j[k, :22], rcv, tcv, prof, o["body_depth"])
                res[p]["vis"].append(vis["visible"])
                res[p]["body_frac"].append(float((o["region"] < R.NUM_REGIONS).mean()))
    return {p: {"vis": np.stack(d["vis"]), "body_frac": np.asarray(d["body_frac"])} for p, d in res.items()}


def summarise(vis: np.ndarray, body_frac: np.ndarray) -> dict:
    n = int(vis.shape[0])
    if n == 0:
        return {"n_frames": 0}
    out = {
        "n_frames": n,
        "p_any_joint": float((vis.sum(1) >= 1).mean()),
        "p_3plus_joints": float((vis.sum(1) >= 3).mean()),
        "mean_joints_visible": float(vis.sum(1).mean()),
        "p_body_px_over_1pct": float((body_frac > 0.01).mean()),
        "median_body_px_frac": float(np.median(body_frac)),
    }
    for g, idx in GROUPS.items():
        out[f"p_any_{g}"] = float(vis[:, list(idx)].any(1).mean())
        out[f"rate_{g}"] = float(vis[:, list(idx)].mean())
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-per-subset", type=int, default=40)
    ap.add_argument("--frames-per-clip", type=int, default=30)
    ap.add_argument("--min-frames", type=int, default=60, help="skip clips shorter than this at 30 fps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/E/perception/render_visibility.json")
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    body = load_body("locked_head")
    body._model.to(device)
    weights = body._model.lbs_weights.to(device)
    index = load_index()
    clips = sample_clips(index, args.clips_per_subset, args.seed, args.min_frames)
    print(f"{len(clips)} clips from {len({e.subset for e in clips})} subsets, device {device}")

    acc = {p: defaultdict(lambda: {"vis": [], "body_frac": []}) for p in PROFILES}
    t0 = time.time()
    for n, e in enumerate(clips):
        clip = load_clip(e, ground=True)
        ids = frames_of_clip(clip, args.frames_per_clip)
        got = count_clip(body, weights, clip, ids, PROFILES, device)
        for p in PROFILES:
            for key in (f"subset:{e.subset}", f"split:{e.split}", "all"):
                acc[p][key]["vis"].append(got[p]["vis"])
                acc[p][key]["body_frac"].append(got[p]["body_frac"])
        if (n + 1) % 20 == 0 or n + 1 == len(clips):
            print(f"  {n + 1}/{len(clips)} clips, {time.time() - t0:.0f} s", flush=True)

    result = {
        "script_version": SCRIPT_VERSION,
        "args": {k: v for k, v in vars(args).items() if k != "out"},
        "n_clips": len(clips),
        "visibility_rule": "joint visible if inside the image, farther than z_near, and the nearest body surface at its pixel is at most 0.10 m in front of it",
        "profiles": {},
    }
    for p in PROFILES:
        rows = {}
        for key, d in sorted(acc[p].items()):
            rows[key] = summarise(np.concatenate(d["vis"]), np.concatenate(d["body_frac"]))
        result["profiles"][p] = rows
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    for p in PROFILES:
        a = result["profiles"][p]["all"]
        print(f"[{p}] frames {a['n_frames']}: any joint {a['p_any_joint']:.2f}, any wrist {a['p_any_wrists']:.2f}, "
              f"any foot {a['p_any_feet']:.2f}, 3+ joints {a['p_3plus_joints']:.2f}, body pixels >1% {a['p_body_px_over_1pct']:.2f}")
    print("written", out)


if __name__ == "__main__":
    main()
