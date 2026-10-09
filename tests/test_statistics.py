"""Calibration error, cluster bootstrap and the paired subject-cluster bootstrap (numpy only)."""

import numpy as np

from hready.metrics.paired_bootstrap import ece_bins, ece_from_bins, paired_bootstrap
from hready.metrics.stats import cluster_bootstrap_ci, expected_calibration_error


def test_ece_hand_computed():
    # 10 predictions of 0.05 that are all negative: one bin, |accuracy 0 - confidence 0.05| = 0.05
    assert np.isclose(expected_calibration_error(np.full(10, 0.05), np.zeros(10, bool)), 0.05)
    # perfectly calibrated two-bin case
    p = np.array([0.25] * 4 + [0.75] * 4)
    y = np.array([1, 0, 0, 0, 1, 1, 1, 0], bool)
    assert np.isclose(expected_calibration_error(p, y), 0.0)


def test_ece_from_bins_matches_reference_estimator():
    rng = np.random.default_rng(1)
    p = rng.random(5000)
    y = rng.random(5000) < p**1.5
    assert np.isclose(ece_from_bins(ece_bins(p, y)), expected_calibration_error(p, y), atol=1e-12)


def test_cluster_bootstrap_ci_degenerate_when_clusters_identical():
    point, lo, hi = cluster_bootstrap_ci(np.ones(12), np.repeat(np.arange(4), 3), n_boot=200, seed=0)
    assert point == lo == hi == 1.0


def _subjects(sums, counts, key="m"):
    return {"per_subject": {f"s{i}": {key: [s, n]} for i, (s, n) in enumerate(zip(sums, counts))}, "ece_bins": {}}


def test_paired_bootstrap_constant_offset_is_exact():
    counts = np.array([10.0, 20.0, 30.0, 40.0])
    ref_sums = np.array([5.0, 9.0, 18.0, 21.0])
    model = _subjects(ref_sums + 2.0 * counts, counts)  # +2 per unit on every subject
    out = paired_bootstrap(model, _subjects(ref_sums, counts), ["m"], n_boot=300, seed=0)["m"]
    assert np.isclose(out["diff"], 2.0) and np.isclose(out["lo"], 2.0) and np.isclose(out["hi"], 2.0)
    assert out["n_subjects"] == 4


def test_paired_bootstrap_interval_is_below_zero_when_model_better_everywhere():
    counts = np.full(8, 50.0)
    ref = np.random.default_rng(2).uniform(40, 60, 8) * counts
    model = ref - np.random.default_rng(3).uniform(1, 3, 8) * counts
    out = paired_bootstrap(_subjects(model, counts), _subjects(ref, counts), ["m"], n_boot=500, seed=0)["m"]
    assert out["hi"] < 0 and out["lo"] <= out["diff"] <= out["hi"]
