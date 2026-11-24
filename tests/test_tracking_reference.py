"""Window selection, reference construction and tracking metrics (numpy only)."""

import numpy as np

from hready.robot import humanoid_asset as H
from hready.robot import tracking_metrics as M
from hready.robot import tracking_reference as R
from tests.test_humanoid_asset import _rest


def _meta():
    return H.build_humanoid(_rest()).meta


def _candidates():
    out = []
    for cat in R.CATEGORIES:
        for i in range(12):
            out.append(R.Candidate(cat, f"set/subj{i // 2}/clip{i}.npz", f"set/subj{i // 2}", 1.0, 1.0 + 4.0 + i))  # i = 0 is 4 s long
    return out


def test_selection_is_reproducible_and_filtered():
    a = R.shuffled_order(_candidates())
    b = R.shuffled_order(list(reversed(_candidates())))
    assert a == b  # input order does not matter
    for rows in a.values():
        assert all(c.end_s - c.start_s >= R.MIN_SEGMENT_S for c in rows)
        assert len({c.subject_key for c in rows}) == len(rows)  # one segment per subject
    assert a != R.shuffled_order(_candidates(), seed=1)


def test_categories_are_shuffled_independently():
    a = R.shuffled_order(_candidates())
    assert [c.rel_path for c in a["walk"]] != [c.rel_path for c in a["fast"]]


def test_window_frames():
    c = R.Candidate("walk", "x", "s", 2.0, 30.0)
    assert R.window_frames(c, 2000) == (60, 360)  # capped at 10 s
    assert R.window_frames(c, 100) == (60, 100)  # clip ends first
    assert R.window_frames(R.Candidate("walk", "x", "s", 2.0, 6.0), 2000) == (60, 180)


def test_reference_stands_on_the_floor_and_starts_at_the_origin():
    meta = _meta()
    t = 6
    root_orient = np.tile([np.pi / 2, 0.0, 0.0], (t, 1))  # canonical Y-up body into the Z-up world
    pose = np.zeros((t, 21, 3))
    pelvis = np.tile([3.0, -2.0, 0.95], (t, 1)) + np.linspace(0, 0.5, t)[:, None] * [1.0, 0.0, 0.0]
    ref = R.build_reference(meta, root_orient, pose, pelvis)
    assert np.allclose(ref["root_pos"][0, :2], 0.0)
    assert abs(ref["lowest_m"].min() - R.CLEARANCE_M) < 1e-9
    assert np.allclose(ref["joints"][:, 0], ref["root_pos"])  # root joint is the pelvis
    assert ref["stats"]["hinge_values_clipped_fraction"] == 0.0
    assert ref["stats"]["clip_mean_joint_shift_mm"] < 1e-6 and abs(ref["stats"]["offset_without_clipping_m"] - ref["offset_m"]) < 1e-9
    assert ref["root_quat_xyzw"].shape == (t, 4)


def test_reference_clips_hinges_to_the_limits():
    meta = _meta()
    pose = np.zeros((2, 21, 3))
    pose[:, 2] = [0.0, 0.0, 0.0]
    pose[1, 2] = [0.0, 0.0, 0.0]
    pose[:, 17 - 1] = [0.0, 3.0, 0.0]  # a left elbow at 3 rad about y is far beyond its 150 deg limit
    ref = R.build_reference(meta, np.zeros((2, 3)), pose, np.tile([0.0, 0.0, 1.0], (2, 1)))
    assert ref["stats"]["hinge_values_clipped_fraction"] > 0.0
    assert ref["stats"]["clip_max_joint_shift_m"] > 0.1  # clipping a 3 rad elbow moves the hand a long way
    lim = R.hinge_limits(meta)
    assert (ref["hinges"] >= lim[:, 0] - 1e-12).all() and (ref["hinges"] <= lim[:, 1] + 1e-12).all()


def test_quaternion_series_is_continuous():
    q = R._quat_series(H.aa_to_matrix(np.stack([[0.0, 0.0, a] for a in np.linspace(0, 6.0, 40)])))
    assert (np.sum(q[1:] * q[:-1], axis=1) > 0).all()


def test_metrics_perfect_tracking():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 0.3, (30, 22, 3))
    quat = np.tile([0.0, 0.0, 0.0, 1.0], (30, 1))
    m = M.evaluate_clip(ref.copy(), ref, quat, quat, 0.03, [7, 8, 20, 21])
    assert m["completed"] and m["success_phc"]
    assert m["mpjpe_global_mm"] == 0.0 and m["velocity_error_m_s"] == 0.0 and m["acceleration_error_m_s2"] == 0.0


def test_metrics_termination_and_rules():
    ref = np.zeros((30, 22, 3))
    quat = np.tile([0.0, 0.0, 0.0, 1.0], (30, 1))
    sim = ref.copy()
    sim[12:, 0, 2] = -0.4  # pelvis drops 0.4 m from step 12
    m = M.evaluate_clip(sim, ref, quat, quat, 0.03, [7, 8, 20, 21])
    assert not m["completed"] and m["termination_step"] == 12
    assert "anchor_height" in m["termination_reasons"]
    assert m["success_phc"]  # mean joint distance stays far below 0.5 m
    sq = quat.copy()
    sq[5:] = [np.sin(0.5), 0.0, 0.0, np.cos(0.5)]  # 1.0 rad rotation of the root
    m2 = M.evaluate_clip(ref, ref, sq, quat, 0.03, [7, 8, 20, 21])
    assert m2["termination_step"] == 5 and m2["termination_reasons"] == ["anchor_orientation"]
    bad = ref.copy()
    bad[3, 5, 0] = np.nan
    m3 = M.evaluate_clip(bad, ref, quat, quat, 0.03, [7, 8, 20, 21])
    assert m3["termination_step"] == 3 and not m3["success_phc"] and np.isfinite(m3["mpjpe_global_mm"])


def test_root_relative_error_ignores_a_common_shift():
    ref = np.random.default_rng(1).normal(0, 0.3, (10, 22, 3))
    quat = np.tile([0.0, 0.0, 0.0, 1.0], (10, 1))
    m = M.evaluate_clip(ref + np.array([0.2, 0.0, 0.0]), ref, quat, quat, 0.03, [7, 8, 20, 21])
    assert m["mpjpe_root_relative_mm"] < 1e-6 and abs(m["mpjpe_global_mm"] - 200.0) < 1e-6


def test_resampling_onto_the_control_times():
    ctrl_dt = 0.03  # 33.3 Hz
    n = R.control_steps(31, 30.0, ctrl_dt)  # one second of reference
    assert n == 34  # times 0 .. 0.99 s
    t = np.arange(n) * ctrl_dt
    ramp = np.linspace(0.0, 1.0, 31)[:, None] * np.ones((1, 2))
    out = R.resample(ramp, t, 30.0)
    assert np.allclose(out[:, 0], np.minimum(t, 1.0))  # linear data stays linear
    q = np.tile([0.0, 0.0, 0.0, 1.0], (5, 1))
    q[3:] = [0.0, 0.0, 1.0, 0.0]
    assert np.allclose(np.linalg.norm(R.resample(q, np.array([0.05]), 30.0, normalise=True), axis=-1), 1.0)


def test_elbow_flexion_beyond_ninety_degrees_is_kept():
    meta = _meta()
    j = meta["joint_names"].index("left_elbow")
    pose = np.zeros((3, 21, 3))
    pose[:, j - 1] = [0.0, -np.deg2rad(125.0), 0.0]  # left flexion is negative b, limit 150 degrees
    ref = R.build_reference(meta, np.zeros((3, 3)), pose, np.tile([0.0, 0.0, 1.0], (3, 1)))
    assert ref["stats"]["hinge_values_clipped_fraction"] == 0.0
    assert np.allclose(ref["hinges"][0].reshape(21, 3)[j - 1], [0.0, -np.deg2rad(125.0), 0.0], atol=1e-9)
    assert ref["stats"]["second_euler_solution_fraction"] > 0.0
    assert np.abs(H.hinges_to_matrix(ref["hinges"][0].reshape(21, 3)) - H.aa_to_matrix(pose[0])).max() < 1e-9


def test_both_euler_solutions_are_the_same_rotation():
    rng = np.random.default_rng(3)
    lim = np.tile([-10.0, 10.0], (63, 1))  # wide limits: the first solution always wins
    for _ in range(20):
        aa = rng.normal(0, 1.5, (21, 3))
        main = H.pose_to_hinges(aa).reshape(21, 3)
        alt = main.copy()
        alt[:, 0] += np.pi
        alt[:, 1] = np.where(main[:, 1] >= 0, np.pi - main[:, 1], -np.pi - main[:, 1])
        alt[:, 2] += np.pi
        assert np.abs(H.hinges_to_matrix(alt) - H.hinges_to_matrix(main)).max() < 1e-9
    h, used = R.hinges_within_limits(aa, lim)
    assert not used.any() and np.allclose(h, H.pose_to_hinges(aa))
