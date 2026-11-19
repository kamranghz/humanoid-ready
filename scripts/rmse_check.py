"""Round-trip RMSE check for soma_x_worker.py.

Run in the hready conda env (not hready-soma):
    conda activate hready
    python %TEMP%\rmse_check.py

Adjust DATA_ROOT and WORKER_SCRIPT below if needed.
"""

import json, subprocess, sys, tempfile, os
from pathlib import Path

# ── config ────────────────────────────────────────────────────────────────────
DATA_ROOT    = r"D:\projects\hready_data"
WORKER      = r"D:\projects\humanoid-ready\scripts\soma_x_worker.py"
CONDA_ENV   = "hready-soma"
PYTHON      = rf"C:\Users\kghol072\AppData\Local\anaconda3\envs\{CONDA_ENV}\python.exe"
NUM_BETAS   = 16
B           = 4   # batch frames
DEVICE      = "cpu"
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np

rng = np.random.default_rng(42)
smplx_poses  = rng.normal(0, 0.1, (B, 55 * 3)).astype(np.float32)
smplx_transl = rng.normal(0, 0.01, (B, 3)).astype(np.float32)

def call_worker(payload: dict) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as fi:
        json.dump(payload, fi)
        fi_path = fi.name
    fo_path = fi_path.replace(".json", "_out.json")
    try:
        r = subprocess.run(
            [PYTHON, WORKER, "--input", fi_path, "--output", fo_path],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            print("STDERR:\n", r.stderr)
            raise RuntimeError(f"worker failed (code {r.returncode})")
        return json.loads(Path(fo_path).read_text())
    finally:
        os.unlink(fi_path)
        if Path(fo_path).exists():
            os.unlink(fo_path)

print("── smplx → mhr ──────────────────────────────────────────")
res1 = call_worker({
    "op": "smplx_to_mhr",
    "data_root": DATA_ROOT,
    "num_betas": NUM_BETAS,
    "device": DEVICE,
    "poses":  smplx_poses.tolist(),
    "transl": smplx_transl.tolist(),
})
rmse_s2m = res1["per_vertex_rmse_m"]
print(f"  SMPL-X → MHR vertex RMSE : {rmse_s2m*100:.4f} cm  ({rmse_s2m:.6f} m)")
print(f"    mean vertex error      : {res1['mean_vertex_error_m']*100:.4f} cm")
print(f"    bridge NN dist (rest)  : median {res1['bridge_nn_median_m']*100:.2f} cm, p95 {res1['bridge_nn_p95_m']*100:.2f} cm")

mhr_poses  = np.array(res1["mhr_poses"],  dtype=np.float32)
mhr_transl = np.array(res1["mhr_transl"], dtype=np.float32)

# flatten to (B, J*3) for the reverse op
J_mhr = mhr_poses.shape[1] if mhr_poses.ndim == 3 else (mhr_poses.shape[1] // 3)
if mhr_poses.ndim == 3:
    mhr_poses_flat = mhr_poses.reshape(B, -1)
else:
    mhr_poses_flat = mhr_poses

print("── mhr → smplx ──────────────────────────────────────────")
res2 = call_worker({
    "op": "mhr_to_smplx",
    "data_root": DATA_ROOT,
    "num_betas": NUM_BETAS,
    "device": DEVICE,
    "poses":  mhr_poses_flat.tolist(),
    "transl": mhr_transl.tolist(),
})
rmse_m2s = res2["per_vertex_rmse_m"]
print(f"  MHR → SMPL-X vertex RMSE : {rmse_m2s*100:.4f} cm  ({rmse_m2s:.6f} m)")
print(f"    mean vertex error      : {res2['mean_vertex_error_m']*100:.4f} cm")
print(f"    bridge NN dist (rest)  : median {res2['bridge_nn_median_m']*100:.2f} cm, p95 {res2['bridge_nn_p95_m']*100:.2f} cm")

smplx_poses_rt  = np.array(res2["smplx_poses"],  dtype=np.float32)
smplx_transl_rt = np.array(res2["smplx_transl"], dtype=np.float32)

# flatten to (B, J*3) for vertex_rmse op
if smplx_poses_rt.ndim == 3:
    smplx_poses_rt_flat = smplx_poses_rt.reshape(B, -1)
else:
    smplx_poses_rt_flat = smplx_poses_rt
smplx_poses_flat = smplx_poses.reshape(B, -1)

print("── round-trip RMSE ──────────────────────────────────────")
res3 = call_worker({
    "op": "vertex_rmse",
    "data_root": DATA_ROOT,
    "num_betas": NUM_BETAS,
    "device": DEVICE,
    "poses_a":  smplx_poses_flat.tolist(),
    "transl_a": smplx_transl.tolist(),
    "poses_b":  smplx_poses_rt_flat.tolist(),
    "transl_b": smplx_transl_rt.tolist(),
})
rmse_rt = res3["rmse_m"]
print(f"  Round-trip vertex RMSE   : {rmse_rt*100:.4f} cm  ({rmse_rt:.6f} m)")
print(f"    mean vertex error      : {res3['mean_vertex_error_m']*100:.4f} cm")

print()
print("══ SUMMARY ══════════════════════════════════════════════")
print(f"  SMPL-X → MHR : {rmse_s2m*100:.4f} cm")
print(f"  MHR → SMPL-X : {rmse_m2s*100:.4f} cm")
print(f"  Round-trip   : {rmse_rt*100:.4f} cm")
print("══════════════════════════════════════════════════════════")
