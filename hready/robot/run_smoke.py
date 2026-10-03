"""Item 6 smoke orchestrator (Windows hready env): retarget → WSL Newton → metrics → replay."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from hready.data.amass import load_paths_config
from hready.robot.metrics import compute_run_metrics
from hready.robot.retarget import GMR_JOINT_NAMES, retarget_clip

CONTROLLER_LABEL = (
    "PD-only joint tracking with default G1_29DOF_CFG gains; not a balance controller"
)

SMOKE_ROWS: tuple[tuple[str, str | None, bool], ...] = (
    ("standing_calibration", None, False),
    ("CMU_132_132_35", "CMU/132/132_35", False),
    (
        "ACCAD_run_to_jump",
        "ACCAD/Female1Running_c3d/C20_-__run_to_jump_to_walk",
        False,
    ),
    (
        "BMLrub_rub001_treadmill",
        "BMLrub/rub001/0000_treadmill_norm_stageii.npz",
        True,
    ),
)
MODES = ("kin_root", "free")


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "configs" / "paths.example.yaml").is_file():
            return parent
    raise FileNotFoundError("repo root")


def _win_to_wsl(path: Path) -> str:
    p = str(path.resolve())
    if len(p) >= 2 and p[1] == ":":
        return "/mnt/" + p[0].lower() + p[2:].replace("\\", "/")
    return p.replace("\\", "/")


def _isaac_cfg(cfg: dict[str, Any]) -> tuple[str, str]:
    isaaclab_root = Path(
        cfg.get("isaaclab_root", "D:/RoboticsResearch/repositories/IsaacLab-Newton")
    )
    wsl_python = cfg.get(
        "wsl_python",
        "/mnt/d/RoboticsResearch/environments/isaaclab-newton-beta2/bin/python",
    )
    if not isaaclab_root.exists():
        raise FileNotFoundError(f"isaaclab_root missing: {isaaclab_root}")
    return str(wsl_python), _win_to_wsl(isaaclab_root)


def _gmr_python(cfg: dict[str, Any]) -> Path:
    return Path(
        cfg.get(
            "gmr_python",
            r"C:\Users\kghol072\AppData\Local\anaconda3\envs\hready-gmr\python.exe",
        )
    )


def write_standing_motion_npz(out: Path) -> None:
    meta = {"clip": "standing_default_pose", "synthetic": True}
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        fps=np.float64(30.0),
        root_pos=np.zeros((1, 3), dtype=np.float64),
        root_quat_xyzw=np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
        dof_pos=np.zeros((1, 29), dtype=np.float64),
        joint_names=np.asarray(GMR_JOINT_NAMES, dtype=object),
        metadata=np.asarray(json.dumps(meta), dtype=object),
    )


def run_wsl_tracker(
    motion_npz: Path,
    mode: str,
    out_npz: Path,
    cfg: dict[str, Any],
    *,
    headless: bool = True,
) -> subprocess.CompletedProcess[str]:
    wsl_python, isaac_cwd = _isaac_cfg(cfg)
    script = _win_to_wsl(_repo_root() / "hready" / "robot" / "isaaclab_newton.py")
    motion_wsl = _win_to_wsl(motion_npz)
    out_wsl = _win_to_wsl(out_npz)
    cmd = [
        "wsl",
        "-e",
        "bash",
        "-lc",
        (
            f"export OMNI_KIT_ACCEPT_EULA=YES; cd {isaac_cwd}; "
            f"{wsl_python} {script} --motion {motion_wsl} --mode {mode} "
            f"--out {out_wsl}"
            + (" --headless" if headless else "")
        ),
    ]
    print("[run_smoke] WSL:", " ".join(cmd), flush=True)
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def run_replay(run_npz: Path, out_mp4: Path, cfg: dict[str, Any]) -> None:
    gmr_py = _gmr_python(cfg)
    script = _repo_root() / "hready" / "robot" / "replay.py"
    subprocess.run(
        [str(gmr_py), str(script), "--run", str(run_npz.resolve()), "--out", str(out_mp4.resolve())],
        check=True,
    )


def summary_row_from_metrics(
    slug: str,
    rel_clip: str | None,
    mode: str,
    run_npz: Path,
    metrics: dict[str, Any],
    metrics_path: Path,
    replay_mp4: Path | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "slug": slug,
        "amass_clip": rel_clip or "standing_default_pose",
        "mode": mode,
        "run_npz": str(run_npz.resolve()),
        "metrics_json": str(metrics_path.resolve()),
        "time_to_fall_s": metrics.get("time_to_fall_s"),
        "joint_rmse_rad": metrics["joint_rmse_rad"],
        "root_pos_err_cm_mean": metrics["root_pos_err_cm_mean"],
        "root_rot_err_deg_mean": metrics["root_rot_err_deg_mean"],
        "pd_demand_exceeds_limit_legs_step_fraction": metrics[
            "pd_demand_exceeds_limit_legs_step_fraction"
        ],
        "pd_demand_exceeds_limit_waist_step_fraction": metrics[
            "pd_demand_exceeds_limit_waist_step_fraction"
        ],
        "pd_demand_exceeds_limit_arms_step_fraction": metrics[
            "pd_demand_exceeds_limit_arms_step_fraction"
        ],
        "waist_tracking_max_abs_err_rad": metrics["waist_tracking_max_abs_err_rad"],
    }
    if replay_mp4 is not None:
        row["replay_mp4"] = str(replay_mp4.resolve())
    return row


def regenerate_smoke_summary(out_root: Path) -> dict[str, Any]:
    """Rebuild ``summary.json`` from saved ``run_*.npz`` (no simulator)."""
    out_root = out_root.resolve()
    table: list[dict[str, Any]] = []
    for slug, rel_clip, _ in SMOKE_ROWS:
        clip_dir = out_root / slug
        for mode in MODES:
            run_npz = clip_dir / f"run_{mode}.npz"
            if not run_npz.is_file():
                continue
            metrics = compute_run_metrics(run_npz)
            metrics_path = clip_dir / f"metrics_{mode}.json"
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            mp4 = clip_dir / f"replay_{mode}.mp4"
            table.append(
                summary_row_from_metrics(
                    slug,
                    rel_clip,
                    mode,
                    run_npz,
                    metrics,
                    metrics_path,
                    mp4 if mp4.is_file() else None,
                )
            )
    summary = {
        "controller_label": CONTROLLER_LABEL,
        "out_root": str(out_root),
        "rows": table,
        "notes": {
            "assist_wrench": "Not measured (Newton contact readback unavailable; pin dropped).",
            "physics_engine": "Isaac Lab + Newton MJWarp in WSL only.",
            "free_mode_tracking": (
                "joint_rmse_rad and root_* errors in free mode are computed only for "
                "steps t <= fall_step (inclusive); see metrics JSON definitions."
            ),
            "torque_saturation": (
                "Per-group pd_demand_exceeds_limit_*_step_fraction; "
                "deprecated_pooled_all29_pd_demand_step_fraction in metrics JSON only."
            ),
            "foot_slip": "EXPERIMENTAL, NOT INTERPRETED — values in metrics JSON only; omitted from this table.",
            "time_to_fall": (
                "Depends on sim_settings (substeps, dt, motion registration). "
                "See sim_settings in each metrics_*.json."
            ),
        },
    }
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_smoke(out_root: Path, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_paths_config()
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    table: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    for slug, rel_clip, allow_excluded in SMOKE_ROWS:
        clip_dir = out_root / slug
        clip_dir.mkdir(parents=True, exist_ok=True)
        motion_npz = clip_dir / "motion.npz"
        if rel_clip is None:
            write_standing_motion_npz(motion_npz)
        else:
            retarget_clip(
                rel_clip, motion_npz, cfg=cfg, allow_excluded=allow_excluded
            )

        for mode in MODES:
            run_npz = clip_dir / f"run_{mode}.npz"
            proc = run_wsl_tracker(motion_npz, mode, run_npz, cfg)
            sys.stdout.write(proc.stdout)
            sys.stderr.write(proc.stderr)
            if proc.returncode != 0:
                raise RuntimeError(f"WSL tracker failed {slug} {mode} code={proc.returncode}")
            if not run_npz.is_file():
                raise FileNotFoundError(run_npz)

            metrics = compute_run_metrics(run_npz)
            metrics_path = clip_dir / f"metrics_{mode}.json"
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

            mp4 = clip_dir / f"replay_{mode}.mp4"
            run_replay(run_npz, mp4, cfg)

            row = summary_row_from_metrics(
                slug, rel_clip, mode, run_npz, metrics, metrics_path, mp4
            )
            table.append(row)
            print(json.dumps(row, indent=2), flush=True)

    summary = {
        "controller_label": CONTROLLER_LABEL,
        "wall_s": time.perf_counter() - t0,
        "out_root": str(out_root),
        "rows": table,
        "notes": {
            "assist_wrench": "Not measured (Newton contact readback unavailable; pin dropped).",
            "physics_engine": "Isaac Lab + Newton MJWarp in WSL only.",
            "free_mode_tracking": (
                "joint_rmse_rad and root_* errors in free mode are computed only for "
                "steps t <= fall_step (inclusive); see metrics JSON definitions."
            ),
            "torque_saturation": (
                "Headline: per-group pd_demand_exceeds_limit_*_step_fraction (PD demand vs effort_limit). "
                "deprecated_pooled_all29_pd_demand_step_fraction kept in metrics JSON only (was 0.380 ACCAD jump)."
            ),
            "foot_slip": "EXPERIMENTAL, NOT INTERPRETED — values in metrics JSON only; omitted from this table.",
            "time_to_fall": (
                "Depends on sim_settings (substeps, dt, motion registration). See sim_settings in each metrics_*.json."
            ),
        },
    }
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Item 6 G1 smoke: retarget + Newton + metrics + replay.")
    p.add_argument("--clip", default="CMU/132/132_35", help="Primary clip id (smoke suite is fixed)")
    p.add_argument("--out", type=Path, default=Path("results/D/smoke"))
    args = p.parse_args(argv)
    if args.clip.replace("\\", "/") not in ("CMU/132/132_35",):
        print(f"[run_smoke] note: full suite runs regardless of --clip={args.clip}", flush=True)
    summary = run_smoke(args.out)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
