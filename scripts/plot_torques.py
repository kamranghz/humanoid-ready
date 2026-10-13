"""Plot mean +- std joint-torque magnitude, raw vs HR-Refine-refined (clean motion as a reference marker).

    python scripts/plot_torques.py --npz results/B2/torque_comparison.npz --out results/B2/torque_plots.png

Torque magnitude is the norm of the world-frame joint torque vector (N m) per frame; mean and std are taken over
all valid frames of all windows. The y axis is logarithmic and the lower error bar is clipped at 1 % of the mean.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _magnitude_stats(torque: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(mean, std) over valid frames of |tau| per joint; torque (windows, T, 22, 3)."""
    mag = np.linalg.norm(torque, axis=-1).reshape(-1, torque.shape[2])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # the root column is all NaN by design
        return np.nanmean(mag, axis=0), np.nanstd(mag, axis=0)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", type=Path, default=Path("results/B2/torque_comparison.npz"))
    ap.add_argument("--out", type=Path, default=Path("results/B2/torque_plots.png"))
    args = ap.parse_args(argv)
    data = np.load(args.npz)
    names = [str(n) for n in data["joint_names"]]
    joints = np.arange(1, len(names))  # the root has no joint torque
    stats = {v: _magnitude_stats(data[f"torque_{v}"]) for v in ("raw", "refined", "clean")}

    fig, ax = plt.subplots(figsize=(13, 5.5))
    width = 0.27
    colors = {"raw": "#c0504d", "refined": "#2e75b6", "clean": "#6aa84f"}
    labels = {"raw": "raw (corrupted input)", "refined": "HR-Refine refined", "clean": "clean AMASS (reference)"}
    for k, v in enumerate(("raw", "refined", "clean")):
        mean, std = (s[joints] for s in stats[v])
        lower = np.minimum(std, 0.99 * mean)
        ax.bar(joints + (k - 1) * width, mean, width, yerr=[lower, std], capsize=2, color=colors[v],
               label=labels[v], error_kw={"elinewidth": 0.8})
    ax.set_yscale("log")
    ax.set_xticks(joints)
    ax.set_xticklabels([names[j] for j in joints], rotation=60, ha="right")
    ax.set_ylabel("joint torque magnitude (N m), mean +- std")
    clips = int(len(set(data["clip_index"].tolist())))
    ax.set_title(
        f"Inverse-dynamics joint torques: raw vs refined ({clips} {str(data['split'])} clips, "
        f"{len(data['clip_index'])} windows of 64 frames, {float(data['total_mass_kg']):.0f} kg body)"
    )
    ax.legend()
    ax.grid(axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    for v in ("raw", "refined", "clean"):
        print(f"{v:8s} mean |tau| over joints 1..21: {stats[v][0][joints].mean():.2f} N m")


if __name__ == "__main__":
    main()
