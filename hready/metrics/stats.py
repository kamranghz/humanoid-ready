"""Cluster bootstrap, rank correlations, and QA-flag evaluation metrics (numpy-only)."""

from __future__ import annotations

import math
from typing import Any, Callable, Literal, Optional, Tuple, Union

import numpy as np

StatFn = Callable[[np.ndarray], float]

BootstrapMethod = Literal["percentile", "t"]

# scipy.stats.t.ppf(0.975, df) for df = 1..30 (two-sided 95%).
_STUDENT_T_975: tuple[float, ...] = (
    12.7062047364,
    4.3026527297,
    3.1824463053,
    2.7764451052,
    2.5705818356,
    2.4469118511,
    2.3646242516,
    2.3060041352,
    2.2621571629,
    2.2281388520,
    2.2009851601,
    2.1788128297,
    2.1603686565,
    2.1447866879,
    2.1314495456,
    2.1199052992,
    2.1098155778,
    2.1009220402,
    2.0930240544,
    2.0859634473,
    2.0796138447,
    2.0738730679,
    2.0686576104,
    2.0638985616,
    2.0595385528,
    2.0555294386,
    2.0518305165,
    2.0484071418,
    2.0452296421,
    2.0422724563,
)


def _erfinv(x: float) -> float:
    """Inverse error function on ``(-1, 1)`` (Winitzki 2008 approximation)."""
    if x <= -1.0:
        return float("-inf")
    if x >= 1.0:
        return float("inf")
    if x == 0.0:
        return 0.0
    sign = 1.0 if x > 0 else -1.0
    a = 0.147
    ln = math.log(1.0 - x * x)
    t1 = 2.0 / (math.pi * a) + ln / 2.0
    t2 = ln / a
    return sign * math.sqrt(math.sqrt(t1 * t1 - t2) - t1)


def _normal_icdf(p: float) -> float:
    return math.sqrt(2.0) * _erfinv(2.0 * p - 1.0)


def _t_critical(df: int, alpha: float) -> float:
    """Two-sided Student-t critical value for ``df`` degrees of freedom."""
    q = 1.0 - alpha / 2.0
    if df < 1:
        return float("inf")
    if df > 30:
        return _normal_icdf(q)
    if abs(q - 0.975) < 1e-9 and df <= 30:
        return _STUDENT_T_975[df - 1]
    # Non-default alpha: normal approximation (exact table is for alpha=0.05).
    return _normal_icdf(q)


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def per_clip_mean(
    values: Any, clip_ids: Any
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-clip means from frame-level ``values`` and parallel ``clip_ids``.

    Returns ``(means, unique_clip_ids)`` in sorted clip-id order.
    """
    v = _to_numpy(values).reshape(-1)
    c = _to_numpy(clip_ids).reshape(-1)
    clips = np.unique(c)
    means = np.array([v[c == cl].mean() for cl in clips], dtype=float)
    return means, clips


def _aggregate_by_cluster(
    values: np.ndarray, cluster_ids: np.ndarray, stat_fn: StatFn
) -> Tuple[np.ndarray, np.ndarray]:
    clusters = np.unique(cluster_ids)
    agg = np.array([stat_fn(values[cluster_ids == cl]) for cl in clusters], dtype=float)
    return agg, clusters


def cluster_bootstrap_ci(
    values: Any,
    cluster_ids: Any,
    *,
    stat_fn: StatFn = np.mean,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
    method: BootstrapMethod = "percentile",
) -> Tuple[float, float, float]:
    """Confidence interval for the cluster-level mean.

    ``values`` are aggregated with ``stat_fn`` within each cluster first.
    Returns ``(point_estimate, lo, hi)``.

    ``method="percentile"`` (default): bootstrap resample clusters with
    replacement and take percentile bounds.

    ``method="t"``: normal-theory interval on cluster means:
    ``mean ± t_{K-1, 1-α/2} · sd / sqrt(K)`` where ``K`` is the number of
    clusters (uses an embedded ``t`` table for df 1..30 at α=0.05, normal
    quantile for df > 30).

    **Coverage note:** with few clusters, the percentile bootstrap often
    **undercovers** the nominal ``1 - alpha`` level; the ``t`` interval is
    often closer to nominal for small ``K`` but assumes Gaussian cluster means.
    """
    v = _to_numpy(values).reshape(-1)
    c = _to_numpy(cluster_ids).reshape(-1)
    agg, _ = _aggregate_by_cluster(v, c, stat_fn)
    point = float(stat_fn(agg))
    n = len(agg)
    if method == "t":
        if n < 2:
            return point, float("nan"), float("nan")
        se = float(np.std(agg, ddof=1) / math.sqrt(n))
        tc = _t_critical(n - 1, alpha)
        return point, point - tc * se, point + tc * se
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = stat_fn(agg[idx])
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return point, lo, hi


def paired_cluster_bootstrap_diff(
    a: Any,
    b: Any,
    cluster_ids: Any,
    *,
    stat_fn: StatFn = np.mean,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Bootstrap CI for the mean paired difference ``stat_fn(a) - stat_fn(b)`` per cluster."""
    a = _to_numpy(a).reshape(-1)
    b = _to_numpy(b).reshape(-1)
    c = _to_numpy(cluster_ids).reshape(-1)
    clusters = np.unique(c)
    diffs = np.array(
        [
            stat_fn(a[c == cl]) - stat_fn(b[c == cl])
            for cl in clusters
        ],
        dtype=float,
    )
    point = float(np.mean(diffs))
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = float(np.mean(diffs[idx]))
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return point, lo, hi


def _rank_average(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=float)
    n = x.size
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def spearman(x: Any, y: Any) -> float:
    """Spearman rho with average ranks for ties."""
    x = _to_numpy(x).reshape(-1).astype(float)
    y = _to_numpy(y).reshape(-1).astype(float)
    rx = _rank_average(x)
    ry = _rank_average(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if denom < 1e-15:
        return float("nan")
    return float((rx * ry).sum() / denom)


def kendall_tau_b(x: Any, y: Any) -> float:
    """Kendall tau-b with tie correction (matches ``scipy.stats.kendalltau``)."""
    x = _to_numpy(x).reshape(-1)
    y = _to_numpy(y).reshape(-1)
    n = x.size
    if n < 2:
        return float("nan")
    conc = 0
    discord = 0
    tx = 0
    ty = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            prod = dx * dy
            if prod > 0:
                conc += 1
            elif prod < 0:
                discord += 1
            else:
                if dx == 0:
                    tx += 1
                if dy == 0:
                    ty += 1
    n0 = n * (n - 1) / 2
    denom = np.sqrt((n0 - tx) * (n0 - ty))
    if denom < 1e-15:
        return float("nan")
    return float((conc - discord) / denom)


def cluster_bootstrap_corr_ci(
    x: Any,
    y: Any,
    cluster_ids: Any,
    *,
    kind: Literal["spearman", "kendall"] = "spearman",
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float, float, int]:
    """Cluster bootstrap CI for a rank correlation.

    Cluster-level means of ``x`` and ``y`` are correlated; resamples with constant
    ``x`` or ``y`` are dropped (returns ``n_dropped`` count).
    """
    x = _to_numpy(x).reshape(-1)
    y = _to_numpy(y).reshape(-1)
    c = _to_numpy(cluster_ids).reshape(-1)
    clusters = np.unique(c)
    cx = np.array([x[c == cl].mean() for cl in clusters], dtype=float)
    cy = np.array([y[c == cl].mean() for cl in clusters], dtype=float)
    corr_fn = spearman if kind == "spearman" else kendall_tau_b
    point = corr_fn(cx, cy)
    rng = np.random.default_rng(seed)
    n = len(clusters)
    boots: list[float] = []
    dropped = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bx = cx[idx]
        by = cy[idx]
        if np.std(bx) < 1e-15 or np.std(by) < 1e-15:
            dropped += 1
            continue
        boots.append(corr_fn(bx, by))
    if not boots:
        return float(point), float("nan"), float("nan"), dropped
    arr = np.array(boots, dtype=float)
    lo = float(np.percentile(arr, 100 * alpha / 2))
    hi = float(np.percentile(arr, 100 * (1 - alpha / 2)))
    return float(point), lo, hi, dropped


def auroc(scores: Any, labels: Any) -> float:
    """Area under ROC via Mann–Whitney / rank formulation.

    Ties in ``scores`` receive 0.5 credit. Returns ``nan`` if ``labels`` contain
    only one class (no positive or no negative).
    """
    scores = _to_numpy(scores).reshape(-1).astype(float)
    labels = _to_numpy(labels).reshape(-1).astype(bool)
    pos = scores[labels]
    neg = scores[~labels]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    if np.all(scores == scores[0]):
        return 0.5
    count = 0.0
    total = pos.size * neg.size
    for s_p in pos:
        for s_n in neg:
            if s_p > s_n:
                count += 1.0
            elif s_p == s_n:
                count += 0.5
    return float(count / total)


def expected_calibration_error(
    probs: Any,
    labels: Any,
    *,
    n_bins: int = 10,
) -> float:
    """Mean absolute calibration gap over bins (binary, flattened)."""
    p = _to_numpy(probs).reshape(-1).astype(float)
    y = _to_numpy(labels).reshape(-1).astype(bool)
    if p.size == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        if not np.any(mask):
            continue
        acc = float(y[mask].mean())
        conf = float(p[mask].mean())
        ece += abs(acc - conf) * (mask.sum() / p.size)
    return float(ece)


def flag_precision_recall(
    flags: Any, bad: Any
) -> Tuple[float, float, float]:
    """Precision, recall, F1 of binary ``flags`` vs binary ``bad`` clip labels.

    Same **no-positives** convention as :func:`hready.metrics.contact.contact_precision_recall_f1`.
    """
    from hready.metrics.contact import contact_precision_recall_f1

    return contact_precision_recall_f1(flags, bad, threshold=0.5)


def cluster_bootstrap_auroc_ci(
    scores: Any,
    labels: Any,
    cluster_ids: Any,
    *,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float, float, int]:
    """Cluster bootstrap CI for :func:`auroc`.

    Per cluster: mean score and any-positive label. Resamples with a single class
    are dropped (count returned).
    """
    scores = _to_numpy(scores).reshape(-1)
    labels = _to_numpy(labels).reshape(-1).astype(bool)
    c = _to_numpy(cluster_ids).reshape(-1)
    clusters = np.unique(c)
    cs = np.array([scores[c == cl].mean() for cl in clusters], dtype=float)
    cl = np.array([labels[c == cl].any() for cl in clusters], dtype=bool)
    point = auroc(cs, cl)
    rng = np.random.default_rng(seed)
    n = len(clusters)
    boots: list[float] = []
    dropped = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bs = cs[idx]
        bl = cl[idx]
        if not bl.any() or bl.all():
            dropped += 1
            continue
        val = auroc(bs, bl)
        if not np.isnan(val):
            boots.append(val)
        else:
            dropped += 1
    if not boots:
        return float(point), float("nan"), float("nan"), dropped
    arr = np.array(boots, dtype=float)
    lo = float(np.percentile(arr, 100 * alpha / 2))
    hi = float(np.percentile(arr, 100 * (1 - alpha / 2)))
    return float(point), lo, hi, dropped
