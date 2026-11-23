"""B2: pretrained baseline on the egocentric render set (E2-B). Measurement only, not a gate.

Model: torchvision Keypoint R-CNN (ResNet-50 FPN, COCO_V1 weights), used as released. Per image the highest-scoring
person detection is taken. The 12 joints that exist in both COCO and SMPL-X are scored on the joints that are visible
by the z-buffer rule and inside the image. Protocol: docs/ego_perception_b2_protocol.md.

    python scripts/ego_perception_b2.py                      # val split, all frames, weights downloaded by torchvision
    python scripts/ego_perception_b2.py --every 10           # quick check on every 10th frame (writes another file)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from hready.data.amass import load_paths_config

SCRIPT_VERSION = 1
# SMPL-X body joint -> COCO keypoint index (nose, eyes, ears are not scored)
PAIRS = {  # name: (smplx_joint, coco_index, partner_coco_index)
    "l_shoulder": (16, 5, 6), "r_shoulder": (17, 6, 5), "l_elbow": (18, 7, 8), "r_elbow": (19, 8, 7),
    "l_wrist": (20, 9, 10), "r_wrist": (21, 10, 9), "l_hip": (1, 11, 12), "r_hip": (2, 12, 11),
    "l_knee": (4, 13, 14), "r_knee": (5, 14, 13), "l_ankle": (7, 15, 16), "r_ankle": (8, 16, 15),
}
NAMES = list(PAIRS)
GROUPS = {g: [n for n in NAMES if n.endswith(g)] for g in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle")}
EXCLUDE = {("DFaust", "50021"), ("DFaust", "50022"), ("DFaust", "50027"), ("MoSh", "50021"), ("MoSh", "50022"), ("MoSh", "50027")}
PCK_PX = (12.8, 25.6)  # 2.5% and 5% of the 512 px image side


def score_frames(gt_uv, gt_vis, pred, pscore, detected):
    """gt_uv (N,12,2), gt_vis (N,12) bool (visible and in image), pred (N,12,2), pscore (N,12), detected (N,) bool.

    Returns per-joint arrays: labelled error, side-agnostic error (min over the joint and its left/right partner),
    both NaN where the joint is not visible or nothing was detected.
    """
    n = gt_uv.shape[0]
    err = np.full((n, len(NAMES)), np.nan)
    err_any = np.full((n, len(NAMES)), np.nan)
    ok = gt_vis & detected[:, None]
    for k, name in enumerate(NAMES):
        d_own = np.linalg.norm(pred[:, k] - gt_uv[:, k], axis=-1)
        partner = NAMES.index(_partner(name))
        d_par = np.linalg.norm(pred[:, partner] - gt_uv[:, k], axis=-1)
        err[:, k] = np.where(ok[:, k], d_own, np.nan)
        err_any[:, k] = np.where(ok[:, k], np.minimum(d_own, d_par), np.nan)
    return err, err_any


def _partner(name: str) -> str:
    return ("r" if name[0] == "l" else "l") + name[1:]


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score of a visible joint > score of a not-visible joint), ties count half (rank statistic)."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def summarise(gt_vis, err, err_any, pscore, detected, in_image):
    out = {"frames": int(gt_vis.shape[0]), "frames_with_detection": float(detected.mean()),
           "visible_joint_instances": int(gt_vis.sum())}

    def block(cols):
        v = gt_vis[:, cols]
        n = int(v.sum())
        r: dict = {"visible_instances": n}
        if n == 0:
            return r
        for tag, e in (("labelled", err[:, cols]), ("side_agnostic", err_any[:, cols])):
            ev = e[v & ~np.isnan(e)]
            r[tag] = {"median_px": float(np.median(ev)) if len(ev) else None, "mean_px": float(ev.mean()) if len(ev) else None}
            for t in PCK_PX:  # recall over all visible joints: no detection counts as a miss
                hit = (~np.isnan(e)) & (np.nan_to_num(e, nan=1e9) <= t) & v
                r[tag][f"recall_at_{t}px"] = float(hit.sum() / n)
        return r

    out["all_12"] = block(list(range(len(NAMES))))
    out["by_group"] = {g: block([NAMES.index(x) for x in xs]) for g, xs in GROUPS.items()}
    det = detected[:, None] & np.ones_like(gt_vis)
    pos = pscore[det & gt_vis]
    neg = pscore[det & ~gt_vis]
    out["visibility_auc"] = {"score_visible_vs_not_visible": auc(pos, neg), "visible": int(len(pos)), "not_visible": int(len(neg))}
    return out


def load_items(root: Path, split: str, every: int, exclude: bool):
    manifest = json.loads((root / "manifest.json").read_text())
    meta = {c["key"]: c for c in manifest["clips"]}
    items = []
    for lab in sorted((root / "labels" / split).glob("*.npz")):
        key = lab.stem
        c = meta[key]
        if exclude and (c["subset"], str(c["subject"])) in EXCLUDE:
            continue
        d = np.load(lab)
        for i in range(0, len(d["frame"]), every):
            items.append((key, int(d["frame"][i]), d["uv"][i], d["visible"][i] & d["in_image"][i]))
    return items


def run(args) -> dict:
    from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights, keypointrcnn_resnet50_fpn

    root = args.render_root or Path(load_paths_config()["data_root"]) / "ego_render" / "aria_rect"
    device = torch.device(args.device)
    weights = None if args.no_pretrained else KeypointRCNN_ResNet50_FPN_Weights.COCO_V1
    model = keypointrcnn_resnet50_fpn(weights=weights, weights_backbone=None).to(device).eval()

    res = {}
    for tag, exclude in (("all_clips", False), ("without_dfaust_mosh_shared_ids", True)):
        if exclude and args.skip_exclusion:
            continue
        items = load_items(root, args.split, args.every, exclude)
        n = len(items)
        smpl = [PAIRS[k][0] for k in NAMES]
        coco = [PAIRS[k][1] for k in NAMES]
        gt_uv = np.stack([it[2][smpl] for it in items])
        gt_vis = np.stack([it[3][smpl] for it in items])
        pred = np.zeros((n, len(NAMES), 2))
        pscore = np.zeros((n, len(NAMES)))
        detected = np.zeros(n, bool)
        t0 = time.time()

        def load(i):
            key, fr, _, _ = items[i]
            return np.asarray(Image.open(root / "images" / args.split / key / f"{fr:06d}.png").convert("RGB"))

        with ThreadPoolExecutor(4) as pool, torch.no_grad():
            for s in range(0, n, args.batch):
                idx = list(range(s, min(n, s + args.batch)))
                imgs = list(pool.map(load, idx))
                x = [torch.from_numpy(a).permute(2, 0, 1).float().div(255.0).to(device) for a in imgs]
                out = model(x)
                for i, o in zip(idx, out):
                    if len(o["scores"]):
                        kp = o["keypoints"][0].cpu().numpy()
                        ks = o["keypoints_scores"][0].cpu().numpy()
                        pred[i] = kp[coco, :2]
                        pscore[i] = ks[coco]
                        detected[i] = True
                if (s // args.batch) % 50 == 0:
                    print(f"  [{tag}] {min(n, s + args.batch)}/{n} frames, {time.time() - t0:.0f} s", flush=True)
        err, err_any = score_frames(gt_uv, gt_vis, pred, pscore, detected)
        res[tag] = summarise(gt_vis, err, err_any, pscore, detected, gt_vis)
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render-root", type=Path, default=None)
    ap.add_argument("--split", default="val", choices=["train", "val"], help="test is reserved for the selected model (B4)")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-pretrained", action="store_true", help="smoke test of the code path only; numbers are meaningless")
    ap.add_argument("--skip-exclusion", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    out = args.out or Path("results/E/perception") / f"b2_baseline_keypointrcnn_{args.split}{'' if args.every == 1 else f'_every{args.every}'}.json"
    if out.exists():
        raise SystemExit(f"{out} exists; give another --out (a result file is not replaced by accident)")
    res = run(args)
    report = {"script_version": SCRIPT_VERSION, "script_sha1": hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:16],
              "protocol": "docs/ego_perception_b2_protocol.md", "model": "torchvision keypointrcnn_resnet50_fpn",
              "weights": "none (smoke test)" if args.no_pretrained else "COCO_V1", "split": args.split, "every": args.every,
              "pck_px": PCK_PX, "image_px": 512, "results": res, "run": time.strftime("%Y-%m-%d %H:%M:%S")}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    for tag, r in res.items():
        a = r["all_12"]
        print(f"[{tag}] frames {r['frames']}, detected {r['frames_with_detection']:.2f}, visible joints {a['visible_instances']}")
        if a.get("visible_instances"):
            for name in ("labelled", "side_agnostic"):
                m = a[name]
                med = "n/a" if m["median_px"] is None else f"{m['median_px']:.1f}"
                print(f"   {name:13s} median {med} px, recall@25.6 {m['recall_at_25.6px']:.2f}")
        print(f"   visibility AUC {r['visibility_auc']['score_visible_vs_not_visible']:.2f}")
    print("written", out)


if __name__ == "__main__":
    main()
