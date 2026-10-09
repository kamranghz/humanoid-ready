"""Pose and physical metrics against hand-computed values (numpy only)."""

import numpy as np

from hready.metrics.physical import foot_skate, ground_penetration
from hready.metrics.pose import mpjpe, pa_mpjpe


def _rng_joints(t=5, j=22, seed=0):
    return np.random.default_rng(seed).normal(size=(t, j, 3))


def test_mpjpe_zero_for_identical_and_root_aligned():
    gt = _rng_joints()
    assert np.allclose(mpjpe(gt, gt), 0.0)
    assert np.allclose(mpjpe(gt + np.array([0.5, -0.2, 1.0]), gt), 0.0)  # global shift removed by root alignment


def test_mpjpe_single_joint_offset():
    gt = _rng_joints()
    pred = gt.copy()
    pred[:, 5, 0] += 0.03  # one of 22 joints off by 3 cm
    assert np.allclose(mpjpe(pred, gt), 30.0 / 22)


def test_pa_mpjpe_invariant_to_similarity_transform():
    gt = _rng_joints()
    c, s = np.cos(0.7), np.sin(0.7)
    r = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    pred = 1.3 * gt @ r.T + np.array([0.1, 0.2, 0.3])
    assert np.allclose(pa_mpjpe(pred, gt), 0.0, atol=1e-6)


def test_ground_penetration_hand_computed():
    pts = np.array([[[0.0, 0.0, -0.002], [0.0, 0.0, 0.01]]])  # (T=1, P=2, 3)
    mean_mm, max_mm = ground_penetration(pts)
    assert np.isclose(mean_mm, 1.0) and np.isclose(max_mm, 2.0)


def test_foot_skate_constant_velocity():
    t = np.arange(10) / 30.0
    pos = np.zeros((10, 1, 3))
    pos[:, 0, 0] = 0.3 * t  # 0.3 m/s along x
    assert np.isclose(foot_skate(pos, np.ones((10, 1)), fps=30.0), 0.3)
    assert foot_skate(pos, np.zeros((10, 1)), fps=30.0) == 0.0
