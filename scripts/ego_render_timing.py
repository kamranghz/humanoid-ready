"""Where does the render time go? One clip, timed pieces (E2-B1). Not a result, a diagnosis.

    python scripts/ego_render_timing.py            # first CMU clip of the index, cuda if present
    python scripts/ego_render_timing.py --device cpu
"""

from __future__ import annotations

import argparse
import time

import torch

from hready.body.smplx_wrapper import load_body
from hready.data import ego_render as R
from hready.data.amass import load_clip, load_index
from hready.data.ego_observation import head_frame_from_joints


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--frames", type=int, default=30)
    args = ap.parse_args()
    dev = torch.device(args.device)

    t = time.time()
    index = load_index()
    print(f"load_index            {time.time() - t:7.2f} s")
    e = next(x for x in index if x.subset == "CMU")
    t = time.time()
    clip = load_clip(e, ground=True)
    print(f"load_clip (ground)    {time.time() - t:7.2f} s")
    t = time.time()
    clip = load_clip(e, ground=True)
    print(f"load_clip again       {time.time() - t:7.2f} s")

    body = load_body("locked_head")
    body._model.to(dev)
    w = body._model.lbs_weights.to(dev)
    n = min(args.frames, clip["root_orient"].shape[0])
    ids = torch.linspace(0, clip["root_orient"].shape[0] - 1, n).round().long().numpy()
    go = torch.as_tensor(clip["root_orient"][ids], dtype=torch.float32, device=dev)
    bp = torch.as_tensor(clip["pose_body"][ids].reshape(n, -1), dtype=torch.float32, device=dev)
    tr = torch.as_tensor(clip["transl"][ids], dtype=torch.float32, device=dev)
    be = torch.as_tensor(clip["betas"][: body.num_betas], dtype=torch.float32, device=dev).unsqueeze(0).expand(n, -1)
    with torch.no_grad():
        body.forward(go, bp, be, tr)  # warm-up
        sync(dev)
        t = time.time()
        out = body.forward(go, bp, be, tr)
        sync(dev)
        print(f"body forward, {n} fr    {time.time() - t:7.2f} s")
        v, j, f = out.vertices, out.joints, out.faces.to(dev)
        freg = R.face_regions(w, f)
        r, tt, cp = head_frame_from_joints(j)
        for name in ("hl2_like", "ego_wide"):
            prof = R.PROFILES[name]
            rcv, tcv = R.cam_cv_from_head_frame(r[0], cp[0])
            R.render_frame(v[0], f, freg, rcv, tcv, cp[0], prof, None)  # warm-up
            sync(dev)
            t = time.time()
            for k in range(n):
                rcv, tcv = R.cam_cv_from_head_frame(r[k], cp[k])
                vc = v[k] @ rcv.T + tcv
                R.rasterise(vc, f, prof)
            sync(dev)
            a = (time.time() - t) / n
            t = time.time()
            for k in range(n):
                rcv, tcv = R.cam_cv_from_head_frame(r[k], cp[k])
                R.render_frame(v[k], f, freg, rcv, tcv, cp[k], prof, None)
            sync(dev)
            b = (time.time() - t) / n
            print(f"{name:9s} rasterise only {a * 1000:7.1f} ms/frame, render_frame {b * 1000:7.1f} ms/frame")


if __name__ == "__main__":
    main()
