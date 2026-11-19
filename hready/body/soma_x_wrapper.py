"""SOMA-X subprocess bridge: SMPL-X ↔ MHR round-trip conversion.

SMPL-X ↔ MHR conversion goes through py-soma-x, which requires chumpy and an
older NumPy ABI incompatible with the hready env (NumPy 2.x).  The conversion
therefore runs in a separate conda env ``hready-soma`` (py-soma-x, chumpy,
NumPy ≤1.26) via a subprocess, exactly like the GMR retargeting bridge.

Usage
-----
    from hready.body.soma_x_wrapper import SomaXBridge

    bridge = SomaXBridge()                        # reads paths.yaml automatically
    result = bridge.smplx_to_mhr(poses, transl)   # (B,J_smplx,3) → MHR poses
    rt     = bridge.round_trip(poses, transl)      # SMPL-X → MHR → SMPL-X

The round-trip vertex RMSE is logged and returned; item-17 acceptance criterion
is that it is reported (the number itself is data).

Pose convention: poses crossing this bridge are ABSOLUTE joint rotations
(axis-angle), not parent-relative parameters; MHR has 77 joints, SMPL-X 55.
See ``docs/body_models.md`` (round-trip error) for the method and its limits.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import torch
import yaml
from torch import Tensor

# ---------------------------------------------------------------------------
# Paths helpers (mirrors smplx_wrapper._repo_root / _load_data_root)
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "configs" / "paths.example.yaml").is_file():
            return parent
    raise FileNotFoundError("Could not locate repo root (configs/paths.example.yaml).")


def _load_paths_cfg() -> dict:
    root = _repo_root()
    paths_yaml = root / "configs" / "paths.yaml"
    if not paths_yaml.is_file():
        raise FileNotFoundError(
            f"Missing {paths_yaml}; copy configs/paths.example.yaml → configs/paths.yaml."
        )
    with paths_yaml.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _soma_python(cfg: dict) -> Path:
    key = "soma_python"
    if key not in cfg:
        raise KeyError(
            f"'soma_python' missing from configs/paths.yaml. "
            f"Add the path to the hready-soma env Python, e.g.:\n"
            f"  soma_python: C:/Users/.../anaconda3/envs/hready-soma/python.exe\n"
            f"See env/soma_env.txt for setup instructions."
        )
    p = Path(cfg[key])
    if not p.is_file():
        raise FileNotFoundError(
            f"soma_python not found: {p}\n"
            f"Create the hready-soma env: see env/soma_env.txt."
        )
    return p


def _worker_script() -> Path:
    return _repo_root() / "scripts" / "soma_x_worker.py"


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SmplxToMhrResult:
    """Output of a SMPL-X → MHR conversion."""
    mhr_poses: Tensor        # (B, J_mhr, 3)  axis-angle, MHR joint order
    mhr_transl: Tensor       # (B, 3)
    per_vertex_rmse_m: float # RMS of per-vertex distance, metres (MHR topology, fit error)


@dataclass
class MhrToSmplxResult:
    """Output of an MHR → SMPL-X conversion."""
    smplx_poses: Tensor      # (B, J_smplx, 3)  axis-angle, SMPL-X joint order
    smplx_transl: Tensor     # (B, 3)
    per_vertex_rmse_m: float # RMS of per-vertex distance, metres (SMPL-X topology, fit error)


@dataclass
class RoundTripResult:
    """SMPL-X → MHR → SMPL-X round-trip."""
    smplx_to_mhr: SmplxToMhrResult
    mhr_to_smplx: MhrToSmplxResult
    round_trip_vertex_rmse_m: float  # RMSE between original and recovered SMPL-X vertices


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------

class SomaXBridge:
    """Subprocess bridge to the hready-soma env for py-soma-x operations.

    Parameters
    ----------
    soma_python:
        Path to the Python executable in the ``hready-soma`` conda env.
        If *None*, read from ``configs/paths.yaml`` (key ``soma_python``).
    data_root:
        Path to ``hready_data/``.  If *None*, read from ``configs/paths.yaml``.
    device:
        ``'cpu'`` or ``'cuda'``.  The worker runs on CPU by default because
        hready-soma targets lightweight conversion, not training.
    """

    def __init__(
        self,
        soma_python: Union[str, Path, None] = None,
        data_root: Union[str, Path, None] = None,
        device: str = "cpu",
    ) -> None:
        cfg = _load_paths_cfg()
        self._soma_python: Path = Path(soma_python) if soma_python is not None else _soma_python(cfg)
        self._data_root: Path = Path(data_root) if data_root is not None else Path(cfg["data_root"])
        self._device = device
        self._worker = _worker_script()
        if not self._worker.is_file():
            raise FileNotFoundError(f"Worker script not found: {self._worker}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def smplx_to_mhr(
        self,
        poses: Tensor,
        transl: Tensor,
        *,
        num_betas: int = 16,
    ) -> SmplxToMhrResult:
        """Convert SMPL-X axis-angle poses to MHR poses.

        Parameters
        ----------
        poses:
            (B, J_smplx, 3) or (B, J_smplx*3) axis-angle tensors.
            J_smplx = 55 (22 body + 1 jaw + 2 eyes + 15+15 hand joints).
        transl:
            (B, 3) root translation in metres.
        num_betas:
            Number of SMPL-X shape coefficients (must match the loaded model).

        Returns
        -------
        SmplxToMhrResult
        """
        payload = {
            "op": "smplx_to_mhr",
            "poses": poses.detach().cpu().numpy().reshape(poses.shape[0], -1).tolist(),
            "transl": transl.detach().cpu().numpy().tolist(),
            "num_betas": num_betas,
            "data_root": str(self._data_root),
            "device": self._device,
        }
        out = self._run_worker(payload)
        return SmplxToMhrResult(
            mhr_poses=torch.as_tensor(out["mhr_poses"], dtype=poses.dtype),
            mhr_transl=torch.as_tensor(out["mhr_transl"], dtype=transl.dtype),
            per_vertex_rmse_m=float(out["per_vertex_rmse_m"]),
        )

    def mhr_to_smplx(
        self,
        mhr_poses: Tensor,
        mhr_transl: Tensor,
        *,
        num_betas: int = 16,
    ) -> MhrToSmplxResult:
        """Convert MHR axis-angle poses to SMPL-X poses.

        Parameters
        ----------
        mhr_poses:
            (B, J_mhr, 3) or (B, J_mhr*3) axis-angle tensors.
        mhr_transl:
            (B, 3) root translation in metres.
        num_betas:
            Number of SMPL-X shape coefficients for the target SMPL-X layer.
        """
        payload = {
            "op": "mhr_to_smplx",
            "poses": mhr_poses.detach().cpu().numpy().reshape(mhr_poses.shape[0], -1).tolist(),
            "transl": mhr_transl.detach().cpu().numpy().tolist(),
            "num_betas": num_betas,
            "data_root": str(self._data_root),
            "device": self._device,
        }
        out = self._run_worker(payload)
        return MhrToSmplxResult(
            smplx_poses=torch.as_tensor(out["smplx_poses"], dtype=mhr_poses.dtype),
            smplx_transl=torch.as_tensor(out["smplx_transl"], dtype=mhr_transl.dtype),
            per_vertex_rmse_m=float(out["per_vertex_rmse_m"]),
        )

    def round_trip(
        self,
        poses: Tensor,
        transl: Tensor,
        *,
        num_betas: int = 16,
    ) -> RoundTripResult:
        """SMPL-X → MHR → SMPL-X round-trip; reports vertex RMSE at each step."""
        fwd = self.smplx_to_mhr(poses, transl, num_betas=num_betas)
        inv = self.mhr_to_smplx(fwd.mhr_poses, fwd.mhr_transl, num_betas=num_betas)

        payload = {
            "op": "vertex_rmse",
            "poses_a": poses.detach().cpu().numpy().reshape(poses.shape[0], -1).tolist(),
            "transl_a": transl.detach().cpu().numpy().tolist(),
            "poses_b": inv.smplx_poses.detach().cpu().numpy().reshape(inv.smplx_poses.shape[0], -1).tolist(),
            "transl_b": inv.smplx_transl.detach().cpu().numpy().tolist(),
            "num_betas": num_betas,
            "data_root": str(self._data_root),
            "device": self._device,
        }
        out = self._run_worker(payload)
        return RoundTripResult(
            smplx_to_mhr=fwd,
            mhr_to_smplx=inv,
            round_trip_vertex_rmse_m=float(out["rmse_m"]),
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_worker(self, payload: dict) -> dict:
        """Serialise *payload* to a temp file, run the worker, return result."""
        with tempfile.TemporaryDirectory(prefix="hready_soma_") as tmp:
            tmp_path = Path(tmp)
            in_file = tmp_path / "input.json"
            out_file = tmp_path / "output.json"
            in_file.write_text(json.dumps(payload), encoding="utf-8")

            cmd = [
                str(self._soma_python),
                str(self._worker),
                "--input", str(in_file),
                "--output", str(out_file),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"soma_x_worker failed (exit {proc.returncode}):\n"
                    f"stdout: {proc.stdout[-2000:]}\n"
                    f"stderr: {proc.stderr[-2000:]}"
                )
            if not out_file.is_file():
                raise RuntimeError("soma_x_worker exited 0 but wrote no output file.")
            return json.loads(out_file.read_text(encoding="utf-8"))
