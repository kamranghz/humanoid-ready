"""Pose error metrics (numpy-only; torch inputs converted via ``.detach().cpu().numpy()``).

Conventions
-----------
- ``pred`` / ``gt`` joint positions: ``(T, J, 3)`` or ``(B, T, J, 3)``, meters, Z-up world.
- Errors are reported in **millimeters**.
- Per-frame outputs have shape ``(..., T)`` (batch dimensions preserved).
- Optional ``valid`` mask ``(..., T)`` or ``(T,)``: ``False`` frames are excluded from
  scalar summaries via :func:`masked_mean` and must not be treated as zero error.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import numpy as np

_MM = 1000.0


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _broadcast_valid(valid: Optional[np.ndarray], shape: tuple[int, ...]) -> Optional[np.ndarray]:
    if valid is None:
        return None
    v = np.asarray(valid, dtype=bool)
    if v.shape == shape:
        return v
    if v.ndim == 1 and len(shape) >= 1 and v.shape[0] == shape[-1]:
        return np.broadcast_to(v, shape)
    raise ValueError(f"valid shape {v.shape} incompatible with {shape}")


def masked_mean(values: np.ndarray, valid: Optional[np.ndarray] = None) -> float:
    """Mean over all elements where ``valid`` is True (or all if ``valid`` is None)."""
    if valid is None:
        return float(np.mean(values))
    v = np.asarray(valid, dtype=bool)
    if v.shape != values.shape:
        v = _broadcast_valid(v, values.shape)
    if not np.any(v):
        return float("nan")
    return float(np.mean(values[v]))


def _select_joints(
    x: np.ndarray, joint_idx: Optional[Union[Sequence[int], np.ndarray]]
) -> np.ndarray:
    if joint_idx is None:
        return x
    idx = np.asarray(joint_idx, dtype=int)
    return x[..., idx, :]


def _root_align(pred: np.ndarray, gt: np.ndarray, root_idx: int) -> tuple[np.ndarray, np.ndarray]:
    pr = pred[..., root_idx : root_idx + 1, :]
    gr = gt[..., root_idx : root_idx + 1, :]
    return pred - pr, gt - gr


def mpjpe(
    pred: Any,
    gt: Any,
    *,
    root_idx: int = 0,
    joint_idx: Optional[Union[Sequence[int], np.ndarray]] = None,
) -> np.ndarray:
    """Mean per-joint position error (mm), root-aligned at ``root_idx``.

    Returns ``(..., T)`` mean Euclidean error over selected joints per frame.
    """
    pred = _to_numpy(pred)
    gt = _to_numpy(gt)
    pred, gt = _root_align(pred, gt, root_idx)
    pred = _select_joints(pred, joint_idx)
    gt = _select_joints(gt, joint_idx)
    err = np.linalg.norm(pred - gt, axis=-1).mean(axis=-1) * _MM
    return err


def _procrustes_single(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Similarity transform (scale, rotation det=+1, translation) aligning ``pred`` to ``gt``."""
    n = pred.shape[0]
    mx = pred.mean(axis=0)
    my = gt.mean(axis=0)
    xc = pred - mx
    yc = gt - my
    cov = yc.T @ xc / max(n, 1)
    u, d, vt = np.linalg.svd(cov)
    s_fix = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_fix[2, 2] = -1.0
    r = u @ s_fix @ vt
    var_x = float((xc * xc).sum() / max(n, 1))
    scale = float(np.trace(np.diag(d) @ s_fix) / max(var_x, 1e-12))
    t = my - scale * (r @ mx)
    return scale * (pred @ r.T) + t


def _procrustes_align_no_reflection(
    pred: np.ndarray, gt: np.ndarray
) -> np.ndarray:
    """Align ``pred`` to ``gt`` with optimal scale, rotation (det=+1), translation."""
    lead = pred.shape[:-2]
    j = pred.shape[-2]
    flat_p = pred.reshape(-1, j, 3)
    flat_g = gt.reshape(-1, j, 3)
    out = np.empty_like(flat_p)
    for i in range(flat_p.shape[0]):
        out[i] = _procrustes_single(flat_p[i], flat_g[i])
    return out.reshape(*lead, j, 3)


def pa_mpjpe(
    pred: Any,
    gt: Any,
    *,
    joint_idx: Optional[Union[Sequence[int], np.ndarray]] = None,
) -> np.ndarray:
    """PA-MPJPE (mm): Procrustes per frame (scale + rotation + translation, det(R)=+1).

    Returns ``(..., T)`` mean joint error after alignment.
    """
    pred = _to_numpy(pred)
    gt = _to_numpy(gt)
    pred = _select_joints(pred, joint_idx)
    gt = _select_joints(gt, joint_idx)
    aligned = _procrustes_align_no_reflection(pred, gt)
    err = np.linalg.norm(aligned - gt, axis=-1).mean(axis=-1) * _MM
    return err


def pve(
    pred_verts: Any,
    gt_verts: Any,
    *,
    root_idx: int = 0,
    pred_joints: Optional[Any] = None,
    gt_joints: Optional[Any] = None,
) -> np.ndarray:
    """Per-vertex error (mm) after root joint translation alignment.

    Provide ``pred_joints`` / ``gt_joints`` for root subtraction (pelvis ``root_idx``).
    Vertices: ``(..., T, V, 3)``.
    Returns ``(..., T)`` mean vertex error per frame.
    """
    pv = _to_numpy(pred_verts)
    gv = _to_numpy(gt_verts)
    if pred_joints is None or gt_joints is None:
        raise ValueError("pve requires pred_joints and gt_joints for root alignment")
    pj = _to_numpy(pred_joints)
    gj = _to_numpy(gt_joints)
    pr = pj[..., root_idx : root_idx + 1, :]
    gr = gj[..., root_idx : root_idx + 1, :]
    pv = pv - pr
    gv = gv - gr
    err = np.linalg.norm(pv - gv, axis=-1).mean(axis=-1) * _MM
    return err


def _align_pcl(
    target_pts: np.ndarray,
    source_pts: np.ndarray,
    *,
    fixed_scale: bool = False,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Umeyama similarity aligning ``source_pts`` to ``target_pts`` (WHAM ``align_pcl``).

    Both inputs ``(N, 3)``. Returns ``(scale, R, t)`` with ``det(R)=+1``.
    """
    y = np.asarray(target_pts, dtype=np.float64)
    x = np.asarray(source_pts, dtype=np.float64)
    n = y.shape[0]
    my = y.mean(axis=0)
    mx = x.mean(axis=0)
    y0 = y - my
    x0 = x - mx
    c = y0.T @ x0 / max(n, 1)
    u, d, vt = np.linalg.svd(c)
    s_fix = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_fix[2, 2] = -1.0
    r = u @ s_fix @ vt
    var = float((x0 * x0).sum() / max(n, 1))
    if fixed_scale:
        scale = 1.0
    else:
        scale = float(np.trace(np.diag(d) @ s_fix) / max(var, 1e-12))
    t = my - scale * (r @ mx)
    return scale, r, t


def _apply_sim(joints: np.ndarray, scale: float, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    return scale * (joints @ r.T) + t


def _world_mpjpe_segment(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    n_align_frames: int | None,
    fixed_scale: bool,
    joint_idx: Optional[Union[Sequence[int], np.ndarray]],
) -> np.ndarray:
    """Per-frame MPJPE (mm) on one ``(T, J, 3)`` segment."""
    pred = _select_joints(pred, joint_idx)
    gt = _select_joints(gt, joint_idx)
    t = pred.shape[0]
    if n_align_frames is None or n_align_frames >= t:
        y = gt.reshape(-1, 3)
        x = pred.reshape(-1, 3)
    else:
        y = gt[:n_align_frames].reshape(-1, 3)
        x = pred[:n_align_frames].reshape(-1, 3)
    s, r, tt = _align_pcl(y, x, fixed_scale=fixed_scale)
    pred_a = _apply_sim(pred, s, r, tt)
    return np.linalg.norm(pred_a - gt, axis=-1).mean(axis=-1) * _MM


def _world_mpjpe_full(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    n_init_frames: int | None,
    segment_length: Optional[int],
    joint_idx: Optional[Union[Sequence[int], np.ndarray]],
) -> np.ndarray:
    t = pred.shape[-3]
    seg = segment_length or t
    out = np.zeros(pred.shape[:-2], dtype=np.float64)
    for start in range(0, t, seg):
        end = min(t, start + seg)
        chunk = _world_mpjpe_segment(
            pred[..., start:end, :, :],
            gt[..., start:end, :, :],
            n_align_frames=n_init_frames,
            fixed_scale=False,
            joint_idx=joint_idx,
        )
        out[..., start:end] = chunk
    return out


def w_mpjpe(
    pred: Any,
    gt: Any,
    *,
    joint_idx: Optional[Union[Sequence[int], np.ndarray]] = None,
    n_init_frames: int = 2,
    segment_length: Optional[int] = None,
) -> np.ndarray:
    """World MPJPE (mm) after similarity alignment from the first ``n_init_frames``.

    Definition follows WHAM / GVHMR (Shin et al., CVPR 2024; Shen et al., SIGGRAPH
    Asia 2024): within each segment, estimate scale, rotation, and translation by
    aligning ``pred`` to ``gt`` using all joints in the first two frames (WHAM
    ``first_align_joints`` / ``align_pcl``), then report mean joint error per frame.
    Returns ``(..., T)`` mm.

    EMDB / RICH protocols report **W-MPJPE100** on **100-frame** segments;
    ``segment_length=None`` evaluates the **whole** sequence (not paper-default).
    Use ``segment_length=100`` for paper-comparable W-MPJPE. These definitions have
    not been validated against published EMDB numbers; that check is planned in
    checklist item **R4**.
    """
    pred = _to_numpy(pred)
    gt = _to_numpy(gt)
    if pred.shape != gt.shape:
        raise ValueError("pred and gt must have the same shape")
    if pred.ndim == 3:
        return _world_mpjpe_full(
            pred, gt,
            n_init_frames=n_init_frames,
            segment_length=segment_length,
            joint_idx=joint_idx,
        )
    if pred.ndim == 4:
        return np.stack(
            [
                _world_mpjpe_full(
                    pred[i], gt[i],
                    n_init_frames=n_init_frames,
                    segment_length=segment_length,
                    joint_idx=joint_idx,
                )
                for i in range(pred.shape[0])
            ],
            axis=0,
        )
    raise ValueError("expected shape (T,J,3) or (B,T,J,3)")


def wa_mpjpe(
    pred: Any,
    gt: Any,
    *,
    joint_idx: Optional[Union[Sequence[int], np.ndarray]] = None,
    segment_length: Optional[int] = None,
) -> np.ndarray:
    """World-aligned MPJPE (mm) after **whole-segment** similarity alignment.

    WHAM ``global_align_joints``: Umeyama similarity (scale + rotation +
    translation, ``det(R)=+1``) estimated from **all frames and joints** in the
    segment, then per-frame MPJPE in mm. Returns ``(..., T)``.

    Papers report **WA-MPJPE100** on **100-frame** segments; ``segment_length=None``
    uses the full sequence. Set ``segment_length=100`` for paper-comparable WA-MPJPE.
    Not yet cross-checked against published results (planned **R4**).
    """
    pred = _to_numpy(pred)
    gt = _to_numpy(gt)
    if pred.ndim == 3:
        return _world_mpjpe_full(
            pred, gt,
            n_init_frames=None,
            segment_length=segment_length,
            joint_idx=joint_idx,
        )
    if pred.ndim == 4:
        return np.stack(
            [
                _world_mpjpe_full(
                    pred[i], gt[i],
                    n_init_frames=None,
                    segment_length=segment_length,
                    joint_idx=joint_idx,
                )
                for i in range(pred.shape[0])
            ],
            axis=0,
        )
    raise ValueError("expected shape (T,J,3) or (B,T,J,3)")


_RTE_MIN_DISP_M = 1e-3


def _rte_single(
    pred: np.ndarray, gt: np.ndarray, min_disp_m: float
) -> np.ndarray:
    _, r, t = _align_pcl(gt, pred, fixed_scale=True)
    pr_hat = pred @ r.T + t
    err = np.linalg.norm(gt - pr_hat, axis=-1)
    if gt.shape[0] < 2:
        disp = min_disp_m
    else:
        disp = float(np.linalg.norm(gt[1:] - gt[:-1], axis=-1).sum())
        disp = max(disp, min_disp_m)
    return err / disp * 100.0


def rte(
    pred_root: Any,
    gt_root: Any,
    *,
    min_disp_m: float = _RTE_MIN_DISP_M,
) -> np.ndarray:
    """Root translation error (percent) after whole-sequence rigid alignment.

    WHAM ``compute_rte``: align ``pred_root`` to ``gt_root`` with ``align_pcl``
    (rotation + translation, **no scale**, ``fixed_scale=True``), then
    ``||gt - pred_aligned||_2`` at each frame divided by total ground-truth path
    length (sum of root step norms). Near-stationary clips use
    ``max(total_disp, min_disp_m)`` so the metric does not explode. Returns
    per-frame percentages ``(T,)`` or ``(..., T)``.

    Reported on the **full** trajectory in WHAM/GVHMR EMDB evaluation (not
    chunked like W/WA-MPJPE). Not yet validated against published RTE (planned **R4**).
    """
    pred = _to_numpy(pred_root).astype(np.float64)
    gt = _to_numpy(gt_root).astype(np.float64)
    if pred.shape != gt.shape:
        raise ValueError("pred_root and gt_root must match shape")
    if pred.ndim == 2:
        return _rte_single(pred, gt, min_disp_m)
    if pred.ndim == 3:
        return np.stack(
            [_rte_single(pred[i], gt[i], min_disp_m) for i in range(pred.shape[0])],
            axis=0,
        )
    raise ValueError("expected shape (T,3) or (B,T,3)")


def exceed_rate(errors: Any, threshold: float) -> float:
    """Fraction of scalar error entries strictly above ``threshold``."""
    e = _to_numpy(errors).reshape(-1)
    if e.size == 0:
        return float("nan")
    return float(np.mean(e > threshold))
