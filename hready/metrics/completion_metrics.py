"""E3 evaluation: pooled MPJPE slices, foot-contact and physical metrics on the 22-joint feet.

Physical metrics use the FK foot joints (L ankle 7, L foot 10, R ankle 8, R foot 11 = channels
L heel, L toe, R heel, R toe). A joint is not a contact point, so each channel height is the joint
height minus its rest-pose joint-to-sole offset (neutral locked_head, item-5 sole clusters). The
same proxy is applied to GT, heuristic and learned joints; the GT row is reported as a reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hready.body.joint_indices import (
    FEET_JOINTS,
    FOOT_CONTACT_CHANNEL_JOINTS,
    LOWER_BODY_JOINTS,
    UPPER_BODY_JOINTS,
)
from hready.data.contact import (
    CHANNEL_NAMES,
    FOOT_NATIVE_UP_AXIS,
    compute_foot_contact_from_positions,
    resolve_foot_channel_clusters,
)
from hready.metrics.stats import expected_calibration_error

MPJPE_GROUPS: dict[str, tuple[int, ...] | None] = {
    "full": None,
    "upper": UPPER_BODY_JOINTS,
    "lower": LOWER_BODY_JOINTS,
    "foot": FEET_JOINTS,
}
BOOTSTRAP_KEYS: tuple[str, ...] = ("mpjpe_full_all_mm", "mpjpe_lower_hidden_mm")


def foot_channel_sole_offsets(body: Any) -> np.ndarray:
    """Rest-pose height of each channel joint above its sole cluster minimum, metres ``(4,)``."""
    from hready.data.contact import _neutral_pose_mesh

    v, joints = _neutral_pose_mesh(body)
    clusters = resolve_foot_channel_clusters(body)
    up = FOOT_NATIVE_UP_AXIS
    return np.array(
        [
            float(joints[j, up] - v[clusters[ch], up].min())
            for ch, j in zip(CHANNEL_NAMES, FOOT_CONTACT_CHANNEL_JOINTS)
        ],
        dtype=np.float32,
    )


def foot_channel_points(joints_22: np.ndarray, sole_offsets: np.ndarray) -> np.ndarray:
    """``(T, 4, 3)`` channel points: joint xy, joint z minus sole offset (floor at z = 0)."""
    pts = np.asarray(
        joints_22[:, FOOT_CONTACT_CHANNEL_JOINTS, :], dtype=np.float64
    ).copy()
    pts[..., 2] -= sole_offsets[None, :]
    return pts


def proxy_foot_contact(joints_22: np.ndarray, sole_offsets: np.ndarray) -> np.ndarray:
    """Item-5 hysteresis rule (recorded thresholds) on the joint channel proxies, ``(T, 4)`` bool."""
    return compute_foot_contact_from_positions(
        foot_channel_points(joints_22, sole_offsets)
    )


@dataclass
class ClipEval:
    rel_path: str
    subject: str
    pred: np.ndarray  # (T, 22, 3) clip world frame
    gt: np.ndarray
    visible: np.ndarray  # (T, 22) bool, evidence visibility
    contact_pred: np.ndarray  # (T, 4) bool
    contact_gt: np.ndarray  # (T, 4) bool, item-5 labels
    contact_prob: np.ndarray | None  # (T, 4) learned only
    overlap_disagree_mm: float
    n_overlap_frames: int
    exclude_contact: bool
    exclude_physical_eval: bool


@dataclass
class CohortAccumulator:
    sums: dict[str, float] = field(default_factory=dict)
    counts: dict[str, float] = field(default_factory=dict)
    maxes: dict[str, float] = field(default_factory=dict)
    per_subject: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    probs: list[np.ndarray] = field(default_factory=list)
    labels: list[np.ndarray] = field(default_factory=list)
    n_clips: int = 0
    n_frames: int = 0
    lost_exclude_contact: int = 0
    lost_exclude_physical_eval: int = 0
    overlap: list[tuple[float, int]] = field(default_factory=list)

    def _add(self, subject: str, key: str, s: float, n: float) -> None:
        if n <= 0:
            return
        self.sums[key] = self.sums.get(key, 0.0) + s
        self.counts[key] = self.counts.get(key, 0.0) + n
        subj = self.per_subject.setdefault(subject, {})
        acc = subj.setdefault(key, [0.0, 0.0])
        acc[0] += s
        acc[1] += n


def accumulate_clip(
    acc: CohortAccumulator,
    ev: ClipEval,
    frame_mask: np.ndarray,
    *,
    sole_offsets: np.ndarray,
    tau_m: float,
    tau_sensitivity_m: float | None = None,
    fps: float = 30.0,
) -> bool:
    """Add one clip's cohort frames. Returns False when the clip has no frames in the cohort.

    ``tau_m`` is the primary (recorded) tolerance; ``tau_sensitivity_m`` adds a second ground-consistency
    column (``ground_consistency_violation_frac_tau_sensitivity``) with the same frames.
    """
    m = np.asarray(frame_mask, dtype=bool)
    if not m.any():
        return False
    acc.n_clips += 1
    acc.n_frames += int(m.sum())
    acc.overlap.append((ev.overlap_disagree_mm, ev.n_overlap_frames))
    subj = ev.subject

    err = np.linalg.norm(ev.pred[m] - ev.gt[m], axis=-1) * 1000.0  # (F, 22) mm
    vis = ev.visible[m]
    for g, idx in MPJPE_GROUPS.items():
        e = err if idx is None else err[:, list(idx)]
        v = vis if idx is None else vis[:, list(idx)]
        acc._add(subj, f"mpjpe_{g}_all_mm", float(e.sum()), e.size)
        acc._add(subj, f"mpjpe_{g}_visible_mm", float(e[v].sum()), int(v.sum()))
        acc._add(subj, f"mpjpe_{g}_hidden_mm", float(e[~v].sum()), int((~v).sum()))

    if ev.exclude_contact:
        acc.lost_exclude_contact += 1
    else:
        cp, cg = ev.contact_pred[m], ev.contact_gt[m]
        acc._add(subj, "contact_tp", float((cp & cg).sum()), 1)
        acc._add(subj, "contact_fp", float((cp & ~cg).sum()), 1)
        acc._add(subj, "contact_fn", float((~cp & cg).sum()), 1)
        if ev.contact_prob is not None:
            acc.probs.append(ev.contact_prob[m].reshape(-1))
            acc.labels.append(cg.reshape(-1))

    if ev.exclude_physical_eval:
        acc.lost_exclude_physical_eval += 1
        return True
    pts = foot_channel_points(ev.pred, sole_offsets)  # (T, 4, 3)
    h = pts[..., 2]
    pen = np.maximum(0.0, -h[m]) * 1000.0
    acc._add(subj, "penetration_mean_mm", float(pen.sum()), pen.size)
    acc.maxes["penetration_max_mm"] = max(
        acc.maxes.get("penetration_max_mm", 0.0), float(pen.max())
    )
    if not ev.exclude_contact:
        # Skate: horizontal channel speed while the GT label says that channel is in contact.
        speed = np.zeros_like(h)
        if h.shape[0] > 1:
            speed[1:] = np.linalg.norm(np.diff(pts[..., :2], axis=0), axis=-1) * fps
        sk = np.zeros_like(m)
        sk[1:] = m[1:]
        c = ev.contact_gt & sk[:, None]
        acc._add(subj, "foot_skate_m_s", float(speed[c].sum()), int(c.sum()))
        # Ground consistency (docs/e0_audit.md): frames with any GT in-contact channel |h| > tau.
        cg = ev.contact_gt & m[:, None]
        frames = cg.any(axis=1)
        viol = (cg & (np.abs(h) > tau_m)).any(axis=1)
        acc._add(
            subj,
            "ground_consistency_violation_frac",
            float(viol[frames].sum()),
            int(frames.sum()),
        )
        if tau_sensitivity_m is not None:
            viol_s = (cg & (np.abs(h) > tau_sensitivity_m)).any(axis=1)
            acc._add(
                subj,
                "ground_consistency_violation_frac_tau_sensitivity",
                float(viol_s[frames].sum()),
                int(frames.sum()),
            )
    return True


def _ratio_bootstrap(
    per_subject: dict[str, dict[str, list[float]]], key: str, *, n_boot: int, seed: int
) -> dict[str, Any]:
    subs = sorted(s for s, d in per_subject.items() if key in d and d[key][1] > 0)
    if len(subs) < 2:
        return {"lo": None, "hi": None, "n_subjects": len(subs)}
    s = np.array([per_subject[x][key][0] for x in subs])
    n = np.array([per_subject[x][key][1] for x in subs])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(subs), size=(n_boot, len(subs)))
    boots = s[idx].sum(axis=1) / n[idx].sum(axis=1)
    return {
        "lo": float(np.percentile(boots, 2.5)),
        "hi": float(np.percentile(boots, 97.5)),
        "n_subjects": len(subs),
    }


def summarize_cohort(
    acc: CohortAccumulator, *, with_ece: bool, n_boot: int, seed: int
) -> dict[str, Any]:
    if acc.n_clips == 0:
        return {"n_clips": 0, "n_frames": 0, "note": "no frames in cohort"}
    metrics: dict[str, Any] = {}
    for k, s in acc.sums.items():
        if k.startswith("contact_"):
            continue
        metrics[k] = s / acc.counts[k]
    metrics.update(acc.maxes)
    if "contact_tp" in acc.sums:
        tp, fp, fn = (acc.sums[f"contact_{x}"] for x in ("tp", "fp", "fn"))
        p = tp / (tp + fp) if tp + fp > 0 else 0.0
        r = tp / (tp + fn) if tp + fn > 0 else 0.0
        metrics.update(
            contact_precision=p,
            contact_recall=r,
            contact_f1=2 * p * r / (p + r) if p + r > 0 else 0.0,
        )
    if with_ece:
        metrics["contact_ece"] = (
            expected_calibration_error(
                np.concatenate(acc.probs), np.concatenate(acc.labels)
            )
            if acc.probs
            else None
        )
    else:
        metrics["contact_ece"] = "n/a"
    ov = [(d, n) for d, n in acc.overlap if n > 0]
    out: dict[str, Any] = {
        "n_clips": acc.n_clips,
        "n_frames": acc.n_frames,
        "n_subjects": len(acc.per_subject),
        "clips_lost_exclude_contact": acc.lost_exclude_contact,
        "clips_lost_exclude_physical_eval": acc.lost_exclude_physical_eval,
        "metrics": metrics,
        "overlap_disagree_mm": (
            float(sum(d * n for d, n in ov) / sum(n for _, n in ov)) if ov else None
        ),
        "bootstrap_95ci_subject_cluster": {
            k: _ratio_bootstrap(acc.per_subject, k, n_boot=n_boot, seed=seed)
            for k in BOOTSTRAP_KEYS
        },
    }
    return out


def per_subject_table(
    acc: CohortAccumulator, keys: tuple[str, ...]
) -> dict[str, dict[str, float]]:
    return {
        s: {k: d[k][0] / d[k][1] for k in keys if k in d and d[k][1] > 0}
        for s, d in sorted(acc.per_subject.items())
    }
