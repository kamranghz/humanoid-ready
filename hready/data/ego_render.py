"""Egocentric render of the body mesh with a small torch rasteriser (E2-B1).

Why an own rasteriser: no render package is installed on the project machine and
nothing is to be installed. Pure torch, flat colours, z-buffer. Result is exactly
repeatable (z-buffer uses amin scatter, ties broken by the lowest face id).

Camera convention (same as ``ego_observation.head_frame_from_joints`` +
``project_joints_egocentric``): the camera sits at the midpoint of the two eyes,
``R_cv`` rows are [right, down, look], ``x_cam = R_cv @ (p - cam_pos)``,
``u = fx * x / z + cx``, ``v = fy * y / z + cy``. Pixel (row i, col j) has its
centre at (j + 0.5, i + 0.5).

Outputs per frame: rgb uint8 (H, W, 3); depth float32 (H, W) (inf = background
in the body layer; floor depth where the floor is the nearest thing); region
uint8 (H, W) (0..21 body region, 254 floor, 255 nothing).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

REGION_FLOOR = 254
REGION_NONE = 255
NUM_REGIONS = 22

# SMPL-X joint order: 0..21 body, 22 jaw, 23/24 eyes, 25..39 left hand, 40..54 right hand.
_HEAD, _LWRIST, _RWRIST = 15, 20, 21


def region_of_joint55() -> np.ndarray:
    """(55,) map from SMPL-X joint index to one of the 22 body regions."""
    m = np.zeros(55, dtype=np.int64)
    m[:22] = np.arange(22)
    m[22:25] = _HEAD
    m[25:40] = _LWRIST
    m[40:55] = _RWRIST
    return m


def region_palette() -> np.ndarray:
    """(22, 3) uint8 fixed colours, one per body region (no randomness)."""
    pal = [
        (200, 200, 200),  # 0 pelvis
        (230, 120, 40),   # 1 left_hip
        (40, 120, 230),   # 2 right_hip
        (170, 170, 120),  # 3 spine1
        (240, 170, 80),   # 4 left_knee
        (90, 160, 240),   # 5 right_knee
        (150, 150, 100),  # 6 spine2
        (250, 210, 120),  # 7 left_ankle
        (140, 200, 250),  # 8 right_ankle
        (130, 130, 90),   # 9 spine3
        (255, 235, 160),  # 10 left_foot
        (190, 225, 255),  # 11 right_foot
        (220, 180, 150),  # 12 neck
        (200, 60, 120),   # 13 left_collar
        (60, 170, 120),   # 14 right_collar
        (235, 200, 170),  # 15 head
        (220, 60, 60),    # 16 left_shoulder
        (60, 190, 90),    # 17 right_shoulder
        (240, 110, 110),  # 18 left_elbow
        (110, 215, 130),  # 19 right_elbow
        (255, 160, 160),  # 20 left_wrist
        (170, 245, 185),  # 21 right_wrist
    ]
    return np.asarray(pal, dtype=np.uint8)


@dataclass(frozen=True)
class RenderProfile:
    name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    z_near: float = 0.1
    model: str = "pinhole"


PROFILES: dict[str, RenderProfile] = {
    # HoloLens-2 research-mode-like colour view: 640x360, about 64 deg horizontal.
    "hl2_like": RenderProfile("hl2_like", 640, 360, 486.3, 486.3, 315.3, 170.0),
    # Aria-like RGB view as a rectified pinhole: square image, 110 deg horizontal (focal = 256 / tan(55 deg)).
    # The real Aria camera is a fisheye; this is the rectified stand-in used for the main perception set.
    "aria_rect": RenderProfile("aria_rect", 512, 512, 179.3, 179.3, 256.0, 256.0),
    # Wide pinhole, used only to compare with the oracle camera (focal 120 px, 512 px).
    "ego_wide": RenderProfile("ego_wide", 512, 512, 120.0, 120.0, 256.0, 256.0),
}


@dataclass
class Floor:
    """Plane through ``height`` along unit ``up`` (world)."""

    up: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    height: float = 0.0
    tile_m: float = 0.5


def cam_cv_from_head_frame(r: Tensor, cam_pos: Tensor) -> tuple[Tensor, Tensor]:
    """``r`` rows [right, up, -forward] -> OpenCV-style rows [right, down, look]."""
    flip = torch.tensor([1.0, -1.0, 1.0], dtype=r.dtype, device=r.device)
    r_cv = flip.view(3, 1) * r
    t_cv = -(r_cv @ cam_pos.unsqueeze(-1)).squeeze(-1)
    return r_cv, t_cv


def project_points(
    points_world: Tensor, r_cv: Tensor, t_cv: Tensor, prof: RenderProfile
) -> tuple[Tensor, Tensor]:
    """(N,3) world -> (N,2) pixel (u, v) and (N,) z; no clamping."""
    xc = points_world @ r_cv.T + t_cv
    z = xc[:, 2]
    zs = torch.where(z.abs() < 1e-9, torch.full_like(z, 1e-9), z)
    u = prof.fx * xc[:, 0] / zs + prof.cx
    v = prof.fy * xc[:, 1] / zs + prof.cy
    return torch.stack([u, v], dim=-1), z


def face_regions(lbs_weights: Tensor, faces: Tensor) -> Tensor:
    """(F,) region id per face: majority of its vertices' dominant region, first vertex on a tie."""
    m = torch.as_tensor(region_of_joint55(), device=lbs_weights.device)
    vreg = m[lbs_weights.argmax(dim=1)]
    r = vreg[faces]
    a, b, c = r[:, 0], r[:, 1], r[:, 2]
    out = a.clone()
    out = torch.where((b == c) & (b != a), b, out)
    return out


def rasterise(
    verts_cam: Tensor, faces: Tensor, prof: RenderProfile, *, max_cells: int = 3_000_000
) -> tuple[Tensor, Tensor]:
    """Return depth (H,W) float32 (inf = empty) and face id (H,W) int64 (-1 = empty)."""
    h, w = prof.height, prof.width
    dev = verts_cam.device
    z = verts_cam[:, 2]
    zc = torch.where(z > 1e-6, z, torch.ones_like(z))
    x = prof.fx * verts_cam[:, 0] / zc + prof.cx
    y = prof.fy * verts_cam[:, 1] / zc + prof.cy
    f = faces
    keep = (z[f] > prof.z_near).all(dim=1)
    fx_ = x[f]
    fy_ = y[f]
    fz_ = z[f]
    x0, x1, x2 = fx_[:, 0], fx_[:, 1], fx_[:, 2]
    y0, y1, y2 = fy_[:, 0], fy_[:, 1], fy_[:, 2]
    area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    keep &= area.abs() > 1e-9
    j0 = torch.ceil(fx_.min(dim=1).values - 0.5).clamp(min=0)
    j1 = torch.floor(fx_.max(dim=1).values - 0.5).clamp(max=w - 1)
    i0 = torch.ceil(fy_.min(dim=1).values - 0.5).clamp(min=0)
    i1 = torch.floor(fy_.max(dim=1).values - 0.5).clamp(max=h - 1)
    keep &= (j1 >= j0) & (i1 >= i0)
    idx = torch.nonzero(keep).squeeze(1)
    empty_d = torch.full((h * w,), float("inf"), dtype=torch.float32, device=dev)
    empty_i = torch.full((h * w,), -1, dtype=torch.int64, device=dev)
    if idx.numel() == 0:
        return empty_d.view(h, w), empty_i.view(h, w)

    bw = (j1[idx] - j0[idx] + 1).long()
    bh = (i1[idx] - i0[idx] + 1).long()
    side = torch.maximum(bw, bh)
    # buckets: window side 1,2,4,... (face fits in side x side window)
    k_of = torch.ceil(torch.log2(side.float())).long().clamp(min=0)
    pix_all, dep_all, fid_all = [], [], []
    for k in torch.unique(k_of).tolist():
        sel = idx[k_of == k]
        kk = 1 << k
        chunk = max(1, max_cells // (kk * kk))
        for s in range(0, sel.numel(), chunk):
            fs = sel[s : s + chunk]
            n = fs.numel()
            o = torch.arange(kk, device=dev)
            jj = j0[fs].long().view(n, 1, 1) + o.view(1, 1, kk)
            ii = i0[fs].long().view(n, 1, 1) + o.view(1, kk, 1)
            inwin = (jj <= j1[fs].long().view(n, 1, 1)) & (ii <= i1[fs].long().view(n, 1, 1))
            px = jj.to(torch.float32) + 0.5
            py = ii.to(torch.float32) + 0.5
            a = area[fs].view(n, 1, 1)
            xa, xb, xc_ = x0[fs].view(n, 1, 1), x1[fs].view(n, 1, 1), x2[fs].view(n, 1, 1)
            ya, yb, yc_ = y0[fs].view(n, 1, 1), y1[fs].view(n, 1, 1), y2[fs].view(n, 1, 1)
            w0 = ((xb - px) * (yc_ - py) - (xc_ - px) * (yb - py)) / a
            w1 = ((xc_ - px) * (ya - py) - (xa - px) * (yc_ - py)) / a
            w2 = 1.0 - w0 - w1
            eps = -1e-6
            ins = inwin & (w0 >= eps) & (w1 >= eps) & (w2 >= eps)
            za, zb, zc_ = fz_[fs, 0].view(n, 1, 1), fz_[fs, 1].view(n, 1, 1), fz_[fs, 2].view(n, 1, 1)
            invz = w0 / za + w1 / zb + w2 / zc_
            ins &= invz > 0
            depth = 1.0 / invz.clamp(min=1e-12)
            flat = (ii * w + jj)
            fid = fs.view(n, 1, 1).expand(n, kk, kk)
            pix_all.append(flat[ins])
            dep_all.append(depth[ins])
            fid_all.append(fid[ins])
    if not pix_all:
        return empty_d.view(h, w), empty_i.view(h, w)
    pix = torch.cat(pix_all)
    dep = torch.cat(dep_all).to(torch.float32)
    fid = torch.cat(fid_all)
    zbuf = empty_d.scatter_reduce(0, pix, dep, reduce="amin", include_self=True)
    win = dep == zbuf[pix]
    big = torch.iinfo(torch.int64).max
    ibuf = torch.full((h * w,), big, dtype=torch.int64, device=dev).scatter_reduce(
        0, pix[win], fid[win], reduce="amin", include_self=True
    )
    ibuf = torch.where(ibuf == big, torch.full_like(ibuf, -1), ibuf)
    return zbuf.view(h, w), ibuf.view(h, w)


def _hash01(a: Tensor, b: Tensor, seed: int) -> Tensor:
    """Deterministic integer hash of cell indices -> [0,1)."""
    h = (a * 73856093) ^ (b * 19349663) ^ (seed * 83492791)
    h = (h ^ (h >> 13)) * 1274126177
    h = h ^ (h >> 16)
    return (h & 0xFFFF).to(torch.float32) / 65536.0


def floor_layer(
    r_cv: Tensor, cam_pos: Tensor, prof: RenderProfile, floor: Floor, seed: int
) -> tuple[Tensor, Tensor]:
    """Depth (H,W) (inf where the ray does not hit the floor in front) and rgb float (H,W,3) 0..255."""
    h, w = prof.height, prof.width
    dev = r_cv.device
    jj = torch.arange(w, device=dev, dtype=torch.float32) + 0.5
    ii = torch.arange(h, device=dev, dtype=torch.float32) + 0.5
    dx = ((jj - prof.cx) / prof.fx).view(1, w).expand(h, w)
    dy = ((ii - prof.cy) / prof.fy).view(h, 1).expand(h, w)
    d_cam = torch.stack([dx, dy, torch.ones_like(dx)], dim=-1)  # z = 1
    d_world = d_cam @ r_cv  # R^T d
    up = torch.as_tensor(floor.up, dtype=torch.float32, device=dev)
    denom = d_world @ up
    num = floor.height - float(cam_pos @ up)
    t = torch.where(denom.abs() > 1e-9, num / denom, torch.full_like(denom, -1.0))
    hit = t > prof.z_near
    pt = cam_pos.view(1, 1, 3) + t.unsqueeze(-1) * d_world
    # two in-plane axes
    a0 = torch.tensor([1.0, 0.0, 0.0], device=dev)
    if abs(float(up @ a0)) > 0.9:
        a0 = torch.tensor([0.0, 1.0, 0.0], device=dev)
    e1 = torch.linalg.cross(up, a0)
    e1 = e1 / torch.linalg.norm(e1)
    e2 = torch.linalg.cross(up, e1)
    s = (pt @ e1) / floor.tile_m
    q = (pt @ e2) / floor.tile_m
    ci = torch.floor(s).long()
    cj = torch.floor(q).long()
    checker = ((ci + cj) & 1).to(torch.float32)
    noise = _hash01(ci, cj, seed)
    base = 95.0 + 35.0 * checker + 18.0 * noise
    rgb = torch.stack([base, base * 0.96, base * 0.9], dim=-1)
    depth = torch.where(hit, t, torch.full_like(t, float("inf")))
    return depth, rgb


def render_frame(
    vertices: Tensor,
    faces: Tensor,
    f_region: Tensor,
    r_cv: Tensor,
    t_cv: Tensor,
    cam_pos: Tensor,
    prof: RenderProfile,
    floor: Floor | None,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """One frame. ``vertices`` (V,3) world, ``faces`` (F,3), ``f_region`` (F,)."""
    vc = vertices @ r_cv.T + t_cv
    zbuf, ibuf = rasterise(vc, faces, prof)
    h, w = prof.height, prof.width
    body = ibuf >= 0
    pal = torch.as_tensor(region_palette(), dtype=torch.float32, device=vc.device)
    rgb = torch.zeros((h, w, 3), dtype=torch.float32, device=vc.device)
    region = torch.full((h, w), REGION_NONE, dtype=torch.uint8, device=vc.device)
    depth = zbuf.clone()
    if floor is not None:
        fdep, frgb = floor_layer(r_cv, cam_pos, prof, floor, seed)
        use_floor = torch.isfinite(fdep) & (fdep < zbuf)
        rgb = torch.where(use_floor.unsqueeze(-1), frgb, rgb)
        region = torch.where(use_floor, torch.full_like(region, REGION_FLOOR), region)
        depth = torch.where(use_floor, fdep, depth)
        body = body & ~use_floor
    if body.any():
        fid = ibuf[body]
        reg = f_region[fid]
        # flat colour times a mild facing term from the face normal
        v0, v1, v2 = vc[faces[fid, 0]], vc[faces[fid, 1]], vc[faces[fid, 2]]
        n = torch.linalg.cross(v1 - v0, v2 - v0, dim=-1)
        n = n / (torch.linalg.norm(n, dim=-1, keepdim=True) + 1e-12)
        cen = (v0 + v1 + v2) / 3.0
        vdir = cen / (torch.linalg.norm(cen, dim=-1, keepdim=True) + 1e-12)
        facing = (n * vdir).sum(dim=-1).abs()
        shade = 0.65 + 0.35 * facing
        rgb[body] = pal[reg.long()] * shade.unsqueeze(-1)
        region[body] = reg.to(torch.uint8)
    return {
        "rgb": rgb.round().clamp(0, 255).to(torch.uint8).cpu().numpy(),
        "depth": depth.cpu().numpy().astype(np.float32),
        "region": region.cpu().numpy(),
        "body_depth": zbuf.cpu().numpy().astype(np.float32),
    }


def joint_visibility(
    joints22: Tensor,
    r_cv: Tensor,
    t_cv: Tensor,
    prof: RenderProfile,
    body_depth: np.ndarray,
    *,
    tol_m: float = 0.10,
) -> dict[str, np.ndarray]:
    """Per joint: pixel position, depth, in-image flag and z-buffer visibility.

    A joint lies inside the body, so it is visible when the nearest body surface at
    its pixel is at most ``tol_m`` in front of the joint (``tol_m`` is about a limb
    radius). Joints outside the image or nearer than z_near are not visible.
    """
    uv, z = project_points(joints22, r_cv, t_cv, prof)
    u = uv[:, 0].cpu().numpy()
    v = uv[:, 1].cpu().numpy()
    zz = z.cpu().numpy()
    inside = (u >= 0) & (u < prof.width) & (v >= 0) & (v < prof.height) & (zz > prof.z_near)
    col = np.clip(np.floor(u).astype(int), 0, prof.width - 1)
    row = np.clip(np.floor(v).astype(int), 0, prof.height - 1)
    surf = body_depth[row, col]
    vis = inside & (zz - surf <= tol_m)
    return {"uv": np.stack([u, v], axis=-1), "z": zz, "in_image": inside, "visible": vis}
