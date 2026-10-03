#!/usr/bin/env python3
"""Visual QA for one AMASS clip: contact sheet + time-series (item 5c-2).

Requires optional deps: ``pip install -e '.[viz]'`` (matplotlib). Not imported by CI or the
``hready`` package — run manually only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Body skeleton edges (SMPL-X 22 body joints).
_BODY_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 4),
    (2, 5),
    (3, 6),
    (4, 7),
    (5, 8),
    (6, 9),
    (7, 10),
    (8, 11),
    (9, 12),
    (9, 13),
    (9, 14),
    (12, 15),
    (13, 16),
    (14, 17),
    (16, 18),
    (17, 19),
    (18, 20),
    (19, 21),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_rel_path(clip_arg: str) -> str:
    from hready.data.amass import _entry_by_rel, load_index

    raw = clip_arg.replace("\\", "/").strip("/")
    index = load_index()
    if raw.endswith(".npz"):
        _entry_by_rel(index, raw)
        return raw
    cand = f"{raw}_stageii.npz"
    try:
        _entry_by_rel(index, cand)
        return cand
    except KeyError:
        pass
    tail = raw.split("/")[-1]
    matches = [e.rel_path for e in index if e.rel_path.endswith(f"{tail}_stageii.npz")]
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise KeyError(f"Ambiguous clip {clip_arg}: {matches[:5]}")
    raise KeyError(clip_arg)


def clip_output_name(clip_arg: str) -> str:
    return clip_arg.replace("\\", "/").strip("/").replace("/", "_")


def _babel_segments(rel_path: str, n_frames: int, fps: float) -> list[tuple[int, int, str]]:
    from hready.data.babel import _entries_by_rel, load_babel_index_payload

    payload = load_babel_index_payload()
    ent = _entries_by_rel(payload).get(rel_path)
    if ent is None:
        return []
    out: list[tuple[int, int, str]] = []
    for seg in ent.segments:
        cats = seg.get("act_cat") or []
        label = cats[0] if cats else "?"
        i0 = max(0, round(float(seg["start_t"]) * fps))
        i1 = min(n_frames, round(float(seg["end_t"]) * fps))
        if i1 > i0:
            out.append((i0, i1, label))
    return out


def _joints_at_frames(clip: dict, frame_ids: np.ndarray, device: str = "cpu") -> np.ndarray:
    import torch

    from hready.body.smplx_wrapper import load_body

    body = load_body("locked_head")
    body._model.to(device)
    betas = torch.as_tensor(clip["betas"], device=device, dtype=torch.float32).reshape(1, -1)
    chunks = []
    for fi in frame_ids:
        go = torch.as_tensor(clip["root_orient"][fi : fi + 1], device=device, dtype=torch.float32)
        bp = torch.as_tensor(clip["pose_body"][fi : fi + 1], device=device, dtype=torch.float32).reshape(
            1, 63
        )
        tr = torch.as_tensor(clip["transl"][fi : fi + 1], device=device, dtype=torch.float32)
        j = body.forward(go, bp, betas, tr).joints[0, :22].detach().cpu().numpy()
        chunks.append(j)
    return np.stack(chunks, axis=0)


def _draw_skeleton(ax, joints: np.ndarray, contact_feet: bool, view: str) -> None:
    if view == "front":
        xs, ys = joints[:, 0], joints[:, 2]
        ax.axhline(0.0, color="0.5", linewidth=0.8, linestyle="--")
    else:
        xs, ys = joints[:, 1], joints[:, 2]
        ax.axhline(0.0, color="0.5", linewidth=0.8, linestyle="--")
    for i, j in _BODY_EDGES:
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]], color="0.35", linewidth=1.0, zorder=1)
    colors = ["C0"] * 22
    if contact_feet:
        colors[7] = colors[8] = colors[10] = colors[11] = "C2"
    else:
        colors[7] = colors[8] = colors[10] = colors[11] = "C3"
    ax.scatter(xs, ys, c=colors, s=12, zorder=2)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def render_contact_sheet(
    clip: dict,
    contact: np.ndarray,
    foot_pos: np.ndarray,
    out_path: Path,
    *,
    device: str = "cpu",
) -> None:
    t = clip["root_orient"].shape[0]
    frame_ids = np.linspace(0, t - 1, 8, dtype=int)
    joints_seq = _joints_at_frames(clip, frame_ids, device=device)
    fig, axes = plt.subplots(8, 2, figsize=(6.0, 16.0), constrained_layout=True)
    for row, fi in enumerate(frame_ids):
        c_any = bool(contact[fi].any())
        _draw_skeleton(axes[row, 0], joints_seq[row], c_any, "front")
        _draw_skeleton(axes[row, 1], joints_seq[row], c_any, "side")
        fp = foot_pos[fi]
        for ch in range(4):
            col = "C2" if contact[fi, ch] else "C3"
            axes[row, 0].scatter(fp[ch, 0], fp[ch, 2], c=col, s=8, marker="s", zorder=3)
            axes[row, 1].scatter(fp[ch, 1], fp[ch, 2], c=col, s=8, marker="s", zorder=3)
        axes[row, 0].set_title(f"f={fi}", fontsize=8)
    fig.suptitle("Contact sheet (green=contact, red=air; floor z=0)", fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def render_timeseries(
    clip: dict,
    rel_path: str,
    contact: np.ndarray,
    foot_pos: np.ndarray,
    imu: dict,
    out_path: Path,
) -> None:
    fps = float(clip["fps"])
    t = foot_pos.shape[0]
    time_s = np.arange(t) / fps
    z, speed = __import__(
        "hready.data.contact", fromlist=["heights_and_speeds_from_positions"]
    ).heights_and_speeds_from_positions(foot_pos)
    skate_speed = speed.max(axis=1)
    names = list(imu["sensor_names"])
    acc = imu["acc"]
    idx_pelvis = names.index("pelvis")
    idx_la = names.index("left_ankle")
    idx_ra = names.index("right_ankle")
    mag_p = np.linalg.norm(acc[:, idx_pelvis], axis=1)
    mag_la = np.linalg.norm(acc[:, idx_la], axis=1)
    mag_ra = np.linalg.norm(acc[:, idx_ra], axis=1)
    segs = _babel_segments(rel_path, t, fps)

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    ch_labels = ["L_heel", "L_toe", "R_heel", "R_toe"]
    for ch in range(4):
        axes[0].plot(time_s, z[:, ch], label=ch_labels[ch], linewidth=0.9)
        on = np.where(contact[:, ch])[0]
        if on.size:
            axes[0].scatter(time_s[on], z[on, ch], s=4, alpha=0.35)
    axes[0].set_ylabel("foot z (m)")
    axes[0].legend(loc="upper right", fontsize=7, ncol=2)
    axes[0].set_title("Heights + contact samples")

    axes[1].plot(time_s, skate_speed, color="C4", linewidth=0.9)
    axes[1].axhline(0.35, color="0.5", linestyle=":", linewidth=0.8, label="T_SKATE=0.35")
    axes[1].set_ylabel("max foot horiz speed (m/s)")
    axes[1].legend(loc="upper right", fontsize=7)

    axes[2].plot(time_s, mag_p, label="|acc| pelvis")
    axes[2].plot(time_s, mag_la, label="|acc| L ankle", alpha=0.85)
    axes[2].plot(time_s, mag_ra, label="|acc| R ankle", alpha=0.85)
    axes[2].axhline(9.81, color="0.5", linestyle="--", linewidth=0.7)
    axes[2].set_ylabel("m/s²")
    axes[2].legend(loc="upper right", fontsize=7)

    axes[3].set_ylim(0, 1)
    axes[3].set_yticks([])
    for i0, i1, lab in segs:
        axes[3].axvspan(i0 / fps, i1 / fps, alpha=0.25)
        axes[3].text(
            (i0 + i1) / 2 / fps,
            0.5,
            lab,
            ha="center",
            va="center",
            fontsize=6,
            rotation=0,
        )
    axes[3].set_xlabel("time (s)")
    axes[3].set_title("BABEL segments (if indexed)")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Inspect one AMASS clip (contact + IMU plots).")
    p.add_argument("--clip", required=True, help="e.g. CMU/132/132_35 or full rel_path")
    p.add_argument("--out", type=Path, default=Path("results/checks"))
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    sys.path.insert(0, str(_repo_root()))
    from hready.data.amass import load_clip
    from hready.data.contact import load_contact, load_foot_positions
    from hready.data.imu import synthetic_imu_from_clip

    rel = resolve_rel_path(args.clip)
    clip = load_clip(rel, ground=True)
    contact = load_contact(rel, device=args.device)
    foot_pos = load_foot_positions(rel)
    imu = synthetic_imu_from_clip(clip, device=args.device)

    out_dir = args.out / clip_output_name(args.clip)
    sheet = out_dir / "contact_sheet.png"
    series = out_dir / "timeseries.png"
    render_contact_sheet(clip, contact, foot_pos, sheet, device=args.device)
    render_timeseries(clip, rel, contact, foot_pos, imu, series)
    print(sheet.resolve())
    print(series.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
