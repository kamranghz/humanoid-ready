"""Paired subject-cluster bootstrap of pooled-metric differences between two models on the same clips.

Each model contributes, per subject, the numerator/denominator sums of every pooled metric
(``CohortAccumulator.per_subject``) and contact-calibration bin statistics. One bootstrap draw
resamples subjects with replacement; both models are pooled over the same draw and differenced,
so the comparison is paired per clip (same clips, same evidence). The metric definitions are
exactly the pooled ones in the tables (``hready.metrics.completion_metrics``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

N_BINS = 10
_EDGES = np.linspace(0.0, 1.0, N_BINS + 1)


def ece_bins(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """``(3, N_BINS)``: count, sum of p, sum of y; same bins as ``expected_calibration_error``."""
    p = np.asarray(probs, dtype=float).reshape(-1)
    y = np.asarray(labels, dtype=bool).reshape(-1)
    idx = np.clip(np.searchsorted(_EDGES, p, side="right") - 1, 0, N_BINS - 1)
    out = np.zeros((3, N_BINS))
    np.add.at(out[0], idx, 1.0)
    np.add.at(out[1], idx, p)
    np.add.at(out[2], idx, y.astype(float))
    return out


def ece_from_bins(bins: np.ndarray) -> float:
    """``sum_b |sum_y - sum_p| / N`` = ``sum_b n_b/N * |acc_b - conf_b|``."""
    n = bins[0].sum()
    return float(np.abs(bins[2] - bins[1]).sum() / n) if n > 0 else float("nan")


def _sums(
    src: dict[str, Any], subs: list[str], key: str
) -> tuple[np.ndarray, np.ndarray]:
    v = [src["per_subject"][s].get(key, [0.0, 0.0]) for s in subs]
    return np.array([x[0] for x in v]), np.array([x[1] for x in v])


def paired_bootstrap(
    model: dict[str, Any],
    reference: dict[str, Any],
    keys: list[str],
    *,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, dict[str, float]]:
    """``model``/``reference``: ``{"per_subject": {subj: {key: [sum, n]}}, "ece_bins": {subj: (3, B)}}``.

    Returns, per key, the full-data pooled values, their difference (model - reference) and the
    two-sided ``1 - alpha`` percentile interval of the difference over subject resamples.
    ``contact_ece`` is recomputed from pooled bins on every resample.
    """
    subs = sorted(set(model["per_subject"]) & set(reference["per_subject"]))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(subs), size=(n_boot, len(subs)))
    out: dict[str, dict[str, float]] = {}
    for key in keys:
        if key == "contact_ece":
            bm = np.stack(
                [model["ece_bins"].get(s, np.zeros((3, N_BINS))) for s in subs]
            )
            br = np.stack(
                [reference["ece_bins"].get(s, np.zeros((3, N_BINS))) for s in subs]
            )
            pm, pr = ece_from_bins(bm.sum(0)), ece_from_bins(br.sum(0))
            diffs = np.array(
                [
                    ece_from_bins(bm[d].sum(0)) - ece_from_bins(br[d].sum(0))
                    for d in draws
                ]
            )
        else:
            sm, nm = _sums(model, subs, key)
            sr, nr = _sums(reference, subs, key)
            pm, pr = sm.sum() / nm.sum(), sr.sum() / nr.sum()
            diffs = sm[draws].sum(1) / nm[draws].sum(1) - sr[draws].sum(1) / nr[
                draws
            ].sum(1)
        out[key] = {
            "model": float(pm),
            "reference": float(pr),
            "diff": float(pm - pr),
            "lo": float(np.percentile(diffs, 100 * alpha / 2)),
            "hi": float(np.percentile(diffs, 100 * (1 - alpha / 2))),
            "n_subjects": len(subs),
        }
    return out
