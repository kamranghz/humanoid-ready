"""Generate the SMPL-X-skeleton humanoid asset (E6a) and write a short summary under results/E/asset/.

    python scripts/humanoid_asset_build.py [--total-mass 75]

The asset (MJCF + metadata) goes to <data_root>/robot_assets/smplx_humanoid/; the repo gets only the summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from hready.data.amass import load_paths_config

from hready.robot import humanoid_asset as H

SMPLX_NEUTRAL = Path("models/smplx/locked_head/models_lockedhead/smplx/SMPLX_NEUTRAL.npz")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-mass", type=float, default=H.DEFAULT_TOTAL_MASS_KG)
    ap.add_argument("--npz", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--summary", type=Path, default=Path("results/E/asset/asset_summary.json"))
    args = ap.parse_args(argv)

    data_root = Path(load_paths_config()["data_root"])
    npz = args.npz or data_root / SMPLX_NEUTRAL
    out = args.out_dir or data_root / "robot_assets" / "smplx_humanoid"
    rest = H.load_smplx_rest(npz)
    asset = H.build_humanoid(rest, args.total_mass)
    xml_path, meta_path = H.write_asset(asset, out)

    j = np.array(asset.meta["rest_joints"])
    summary = {
        "asset": str(xml_path), "metadata": str(meta_path), "body_model": str(npz),
        "joints": 22, "hinges": 63, "total_mass_kg": asset.meta["total_mass_kg"],
        "mass_sum_kg": float(np.sum(asset.meta["body_mass_kg"])), "mass_source": asset.meta["mass_source"],
        "root_height_standing_m": asset.meta["root_height_standing_m"],
        "bone_lengths_m": {n: float(np.linalg.norm(j[i] - j[p])) for i, (n, p) in
                           enumerate(zip(asset.meta["joint_names"], asset.meta["parents"])) if p >= 0},
        "pd_groups": asset.meta["pd"],
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("asset:", xml_path)
    print("mass sum / total:", summary["mass_sum_kg"], summary["total_mass_kg"], "kg; standing root height:",
          round(summary["root_height_standing_m"], 3), "m")
    print("written", args.summary)


if __name__ == "__main__":
    main()
