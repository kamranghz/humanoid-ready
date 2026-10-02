"""Evaluation metrics (lazy imports — no torch at ``import hready``)."""

from __future__ import annotations

__all__ = [
    "mpjpe",
    "pa_mpjpe",
    "pve",
    "w_mpjpe",
    "wa_mpjpe",
    "rte",
    "exceed_rate",
    "masked_mean",
    "foot_skate",
    "ground_penetration",
    "foot_float",
    "accel_error",
    "jitter",
    "balance_violation",
    "contact_precision_recall_f1",
    "per_clip_mean",
    "cluster_bootstrap_ci",
    "paired_cluster_bootstrap_diff",
    "spearman",
    "kendall_tau_b",
    "cluster_bootstrap_corr_ci",
    "auroc",
    "flag_precision_recall",
    "cluster_bootstrap_auroc_ci",
]

_LAZY_EXPORTS = {
    "mpjpe": ("hready.metrics.pose", "mpjpe"),
    "pa_mpjpe": ("hready.metrics.pose", "pa_mpjpe"),
    "pve": ("hready.metrics.pose", "pve"),
    "w_mpjpe": ("hready.metrics.pose", "w_mpjpe"),
    "wa_mpjpe": ("hready.metrics.pose", "wa_mpjpe"),
    "rte": ("hready.metrics.pose", "rte"),
    "exceed_rate": ("hready.metrics.pose", "exceed_rate"),
    "masked_mean": ("hready.metrics.pose", "masked_mean"),
    "foot_skate": ("hready.metrics.physical", "foot_skate"),
    "ground_penetration": ("hready.metrics.physical", "ground_penetration"),
    "foot_float": ("hready.metrics.physical", "foot_float"),
    "accel_error": ("hready.metrics.physical", "accel_error"),
    "jitter": ("hready.metrics.physical", "jitter"),
    "balance_violation": ("hready.metrics.physical", "balance_violation"),
    "contact_precision_recall_f1": (
        "hready.metrics.contact",
        "contact_precision_recall_f1",
    ),
    "per_clip_mean": ("hready.metrics.stats", "per_clip_mean"),
    "cluster_bootstrap_ci": ("hready.metrics.stats", "cluster_bootstrap_ci"),
    "paired_cluster_bootstrap_diff": (
        "hready.metrics.stats",
        "paired_cluster_bootstrap_diff",
    ),
    "spearman": ("hready.metrics.stats", "spearman"),
    "kendall_tau_b": ("hready.metrics.stats", "kendall_tau_b"),
    "cluster_bootstrap_corr_ci": (
        "hready.metrics.stats",
        "cluster_bootstrap_corr_ci",
    ),
    "auroc": ("hready.metrics.stats", "auroc"),
    "flag_precision_recall": ("hready.metrics.stats", "flag_precision_recall"),
    "cluster_bootstrap_auroc_ci": (
        "hready.metrics.stats",
        "cluster_bootstrap_auroc_ci",
    ),
}


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        module_path, attr = _LAZY_EXPORTS[name]
        import importlib

        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
