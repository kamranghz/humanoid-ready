"""Temperature scaling and calibration metrics for the contact head.

The contact head outputs logits; calibrated probabilities are ``sigmoid(logit / T)`` with one scalar ``T > 0``.
``ece`` is the plain 10-bin estimator used in the completion tables (``hready.metrics.stats``);
``brier`` is the mean squared difference between probability and label.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.optimize import minimize_scalar

from hready.metrics.stats import expected_calibration_error

_EPS = 1e-7


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1.0 - _EPS)
    return np.log(p) - np.log1p(-p)


def apply_temperature(p: np.ndarray, temperature: float) -> np.ndarray:
    """Probabilities rescaled as ``sigmoid(logit(p) / T)`` (float32, same shape)."""
    z = logit(p) / float(temperature)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def ece(probs: np.ndarray, labels: np.ndarray) -> float:
    return expected_calibration_error(probs, labels)


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=bool).reshape(-1).astype(np.float64)
    return float(np.mean((p - y) ** 2)) if p.size else float("nan")


def nll(probs: np.ndarray, labels: np.ndarray) -> float:
    p = np.clip(np.asarray(probs, dtype=np.float64).reshape(-1), _EPS, 1.0 - _EPS)
    y = np.asarray(labels, dtype=bool).reshape(-1)
    return float(-np.mean(np.where(y, np.log(p), np.log1p(-p))))


def fit_temperature(
    nll_of_temperature: Callable[[float], float], log_bounds: tuple[float, float] = (-3.0, 3.0)
) -> float:
    """Minimiser of ``nll_of_temperature`` over ``T = exp(s)``, ``s`` in ``log_bounds`` (bounded 1-D search)."""
    res = minimize_scalar(
        lambda s: nll_of_temperature(float(np.exp(s))),
        bounds=log_bounds,
        method="bounded",
        options={"xatol": 1e-4},
    )
    return float(np.exp(res.x))
