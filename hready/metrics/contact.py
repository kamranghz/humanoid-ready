"""Binary contact classification metrics (numpy-only).

Conventions match :mod:`hready.metrics.pose` for optional masks (not used here;
frame-level vectors are aggregated by the caller).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _binarize(x: np.ndarray, threshold: float) -> np.ndarray:
    return (x > threshold).astype(bool)


def contact_precision_recall_f1(
    pred: Any,
    gt: Any,
    *,
    threshold: float = 0.5,
) -> Tuple[float, float, float]:
    """Precision, recall, and F1 for binary contact.

    ``pred`` and ``gt`` are flattened to 1D booleans after thresholding ``pred``.

    **No-positives convention:** if there are no positive labels in **both** prediction
    and ground truth (TP+FP=0 and TP+FN=0), F1 is defined as **0** (not NaN).
    """
    p = _binarize(_to_numpy(pred), threshold)
    g = _to_numpy(gt).astype(bool).reshape(-1)
    p = p.reshape(-1)
    tp = int(np.sum(p & g))
    fp = int(np.sum(p & ~g))
    fn = int(np.sum(~p & g))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if (tp + fp) == 0 and (tp + fn) == 0:
        f1 = 0.0
    elif (prec + rec) == 0:
        f1 = 0.0
    else:
        f1 = 2 * prec * rec / (prec + rec)
    return float(prec), float(rec), float(f1)
