"""Checks for the torch rasteriser (synthetic data only, no body model needed)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # CI installs numpy only; these checks need torch

from hready.data import ego_render as R  # noqa: E402
from hready.data.ego_observation import (  # noqa: E402
    EgoCameraConfig,
    head_frame_from_joints,
    project_joints_egocentric,
)

PROF = R.RenderProfile("t", 64, 48, 50.0, 50.0, 32.0, 24.0)
TRI = torch.tensor([[-0.4, -0.3, 2.0], [0.4, -0.3, 2.0], [0.0, 0.4, 2.0]])
FACES = torch.tensor([[0, 1, 2]])


def _expected_coverage(verts, prof):
    u = (prof.fx * verts[:, 0] / verts[:, 2] + prof.cx).numpy()
    v = (prof.fy * verts[:, 1] / verts[:, 2] + prof.cy).numpy()
    out = np.zeros((prof.height, prof.width), bool)
    for r in range(prof.height):
        for c in range(prof.width):
            x, y = c + 0.5, r + 0.5

            def e(a, b, x=x, y=y):
                return (u[b] - u[a]) * (y - v[a]) - (v[b] - v[a]) * (x - u[a])

            s = [e(0, 1), e(1, 2), e(2, 0)]
            out[r, c] = all(k >= -1e-6 for k in s) or all(k <= 1e-6 for k in s)
    return out


def test_triangle_coverage_and_depth():
    z, i = R.rasterise(TRI, FACES, PROF)
    cov = (i >= 0).numpy()
    assert (cov != _expected_coverage(TRI, PROF)).sum() <= 2
    assert torch.allclose(z[i >= 0], torch.full((int(cov.sum()),), 2.0))


def test_nearer_triangle_wins_in_either_order():
    v = torch.cat([TRI, TRI - torch.tensor([0.0, 0.0, 0.5])], 0)
    za, _ = R.rasterise(v, torch.tensor([[0, 1, 2], [3, 4, 5]]), PROF)
    zb, _ = R.rasterise(v, torch.tensor([[3, 4, 5], [0, 1, 2]]), PROF)
    assert float(za[24, 32]) < 1.51 and float(zb[24, 32]) < 1.51


def test_triangle_crossing_near_plane_is_dropped():
    v = TRI.clone()
    v[2, 2] = -1.0
    _, i = R.rasterise(v, FACES, PROF)
    assert (i < 0).all()


def test_camera_matches_project_joints_egocentric():
    torch.manual_seed(0)
    j55 = torch.randn(5, 55, 3) * 0.3 + torch.tensor([0.0, 0.0, 1.5])
    j55[:, 23] = j55[:, 15] + torch.tensor([-0.03, 0.08, 0.05])
    j55[:, 24] = j55[:, 23] + torch.tensor([0.06, 0.0, 0.0])
    r, t, cp = head_frame_from_joints(j55)
    kp = project_joints_egocentric(j55[:, :22], r, t, EgoCameraConfig(focal_px=50.0, img_size=(64, 48)))
    worst, used = 0.0, 0
    for k in range(5):
        rcv, tcv = R.cam_cv_from_head_frame(r[k], cp[k])
        uv, z = R.project_points(j55[k, :22], rcv, tcv, PROF)
        ref_u = (kp[k, :, 0] + 1) / 2 * 64
        ref_v = (kp[k, :, 1] + 1) / 2 * 48
        ok = z > 0.1
        d = torch.maximum((uv[:, 0] - ref_u).abs(), (uv[:, 1] - ref_v).abs())[ok]
        worst = max(worst, float(d.max()))
        used += int(ok.sum())
    assert used > 20 and worst < 1e-3


def test_floor_and_frame_are_byte_stable():
    v = TRI.clone()
    f = R.Floor(up=np.array([0.0, 1.0, 0.0]), height=1.0)
    eye = torch.eye(3)
    kw = {"r_cv": eye, "t_cv": torch.zeros(3), "cam_pos": torch.zeros(3), "prof": PROF, "floor": f, "seed": 3}
    a = R.render_frame(v, FACES, torch.tensor([5]), **kw)
    b = R.render_frame(v, FACES, torch.tensor([5]), **kw)
    for key in ("rgb", "depth", "region"):
        assert a[key].tobytes() == b[key].tobytes()
    assert (a["region"] == 5).any() and (a["region"] == R.REGION_FLOOR).any()
