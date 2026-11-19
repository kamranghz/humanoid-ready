"""SOMA-X worker script — runs in the hready-soma conda env.

Receives a JSON payload via --input, executes the requested py-soma-x
operation, and writes the result to --output as JSON.

Operations
----------
smplx_to_mhr
    SMPL-X poses -> MHR poses.  Pose SMPL-X, bridge its surface onto the MHR
    topology, then fit MHR rotations with ``soma.smpl.transfer.PoseInversion``.

mhr_to_smplx
    The reverse direction, symmetric to the above.

vertex_rmse
    Vertex error between two SMPL-X pose sequences (round-trip evaluation).

Conventions
-----------
* All poses exchanged by this worker are ABSOLUTE joint rotations (axis-angle,
  shape (B, J, 3)), which is what ``PoseInversion`` returns and what
  ``layer.pose(..., absolute_pose=True)`` consumes.  SMPL-X: J = 55.
  MHR: J = 77 public SOMA joints (the extra root joint PoseInversion returns
  for MHR is stripped; its motion is carried by the root translation).
* Translations and vertices are in metres.

Why PoseInversion directly
--------------------------
``transfer_smpl_family_pose_parameters`` needs ``.obj`` base meshes that the
locked-head SMPL-X distribution does not ship, and for target=MHR it calls
``SOMALayer.pose(global_translation=...)`` although that layer's keyword is
``transl``.  Both are py-soma-x issues we cannot fix from outside.

Error metrics
-------------
``per_vertex_rmse_m`` is a true RMS, sqrt(mean(d^2)) over all vertices and
frames, where d is the per-vertex Euclidean distance.  ``mean_vertex_error_m``
is the plain mean of d.  Each directional value is the inversion fit error
measured in the TARGET topology against the bridged surface, so it does not
include the nearest-neighbour bridge error itself; that is reported separately
as ``bridge_nn_median_m`` / ``bridge_nn_p95_m`` (rest-pose distance from each
target vertex to its matched source vertex).

This file must stay dependency-free for hready (it only imports py-soma-x,
torch/numpy/scipy and the standard library) and must not import any hready
module.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

MHR_IDENTITY_DIM = 45
MHR_PUBLIC_JOINTS = 77


def _smplx_layer(data_root: str, num_betas: int, device: str):
    import soma
    smplx_model_dir = str(
        Path(data_root) / "models" / "smplx" / "locked_head" / "models_lockedhead"
    )
    return soma.SMPLXLayer(
        data_root=smplx_model_dir,
        gender="neutral",
        device=device,
    )


def _mhr_layer(data_root: str, device: str):
    import soma
    # MHR is built into py-soma-x; no external model files needed.
    # data_root is passed so SOMA_neutral.npz can be located (required by
    # SOMALayer even when identity_model_type='mhr').
    soma_assets = str(Path(data_root) / "models" / "soma")
    return soma.SOMALayer(
        data_root=soma_assets if Path(soma_assets).is_dir() else None,
        identity_model_type="mhr",
        device=device,
    )


def _patch_mhr_layer(layer, device: str) -> None:
    """Make an MHR SOMALayer inject scale_params=(B, 68) when it is absent.

    py-soma-x 0.3.6 requires scale_params for MHR in every prepare_identity
    call, but PoseInversion calls prepare_identity without it.
    """
    import functools
    import types

    import torch as _t

    _orig = layer.prepare_identity

    @functools.wraps(_orig)
    def _patched(identity_coeffs, scale_params=None, **kwargs):
        if scale_params is None:
            B = identity_coeffs.shape[0] if identity_coeffs.ndim > 1 else 1
            scale_params = _t.zeros(B, 68, device=device)
        return _orig(identity_coeffs, scale_params=scale_params, **kwargs)

    layer.prepare_identity = types.MethodType(
        lambda self, identity_coeffs, scale_params=None, **kw: _patched(
            identity_coeffs, scale_params, **kw
        ),
        layer,
    )


def _verts(out):
    return out["vertices"] if isinstance(out, dict) else out.vertices


def _rest_vertices(layer, identity_dim: int, n_joints: int, device: str):
    """Rest-pose vertices (V, 3) in metres, pelvis frame, zero identity.

    Uses zero RELATIVE rotations.  Note that layer.bind_shape is NOT this mesh:
    for MHR it is in centimetres, feet at y=0, in a different (A-like) pose.
    """
    import torch

    layer.prepare_identity(torch.zeros(1, identity_dim, device=device))
    with torch.no_grad():
        out = layer.pose(
            torch.zeros(1, n_joints, 3, device=device),
            pose2rot=True,
            apply_correctives=False,
            absolute_pose=False,
        )
    return _verts(out)[0]


def _nn_index(src_rest, dst_rest, chunk: int = 2048):
    """For every dst vertex, the index of the nearest src vertex (chunked).

    Returns (idx (V_dst,), dist (V_dst,)).
    """
    import torch

    idx, dist = [], []
    for i in range(0, dst_rest.shape[0], chunk):
        d = torch.cdist(dst_rest[i:i + chunk], src_rest)
        m, a = d.min(dim=1)
        idx.append(a)
        dist.append(m)
    return torch.cat(idx), torch.cat(dist)


def _bridge_stats(dist) -> dict:
    return {
        "bridge_nn_median_m": float(dist.median().item()),
        "bridge_nn_p95_m": float(dist.quantile(0.95).item()),
    }


def _errors(a, b) -> dict:
    """RMS and mean of per-vertex Euclidean distance between two vertex sets."""
    d = (a - b).norm(dim=-1)
    return {
        "per_vertex_rmse_m": float((d ** 2).mean().sqrt().item()),
        "mean_vertex_error_m": float(d.mean().item()),
    }


def _rotmat_to_aa(R):
    """(B, J, 3, 3) torch rotation matrices -> (B, J, 3) numpy axis-angle."""
    import scipy.spatial.transform as st

    arr = R.detach().cpu().numpy()
    sh = arr.shape
    aa = st.Rotation.from_matrix(arr.reshape(-1, 3, 3)).as_rotvec()
    return aa.reshape(sh[0], sh[1], 3)


def _op_smplx_to_mhr(payload: dict) -> dict:
    import numpy as np
    import torch
    from soma.smpl.transfer import PoseInversion

    device = payload.get("device", "cpu")
    data_root = payload["data_root"]
    num_betas = int(payload.get("num_betas", 16))

    poses_np = np.array(payload["poses"], dtype=np.float32)    # (B, 55*3)
    transl_np = np.array(payload["transl"], dtype=np.float32)  # (B, 3)
    B = poses_np.shape[0]

    smplx_layer = _smplx_layer(data_root, num_betas, device)
    mhr_layer = _mhr_layer(data_root, device)
    _patch_mhr_layer(mhr_layer, device)

    J_smplx = smplx_layer.num_joints
    poses_t = torch.as_tensor(poses_np.reshape(B, J_smplx, 3), device=device)
    transl_t = torch.as_tensor(transl_np, device=device)

    # Bridge from rest-pose meshes (both metres, pelvis frame, T-pose).
    smplx_rest = _rest_vertices(smplx_layer, num_betas, J_smplx, device)
    mhr_rest = _rest_vertices(mhr_layer, MHR_IDENTITY_DIM, MHR_PUBLIC_JOINTS, device)
    idx, nn_dist = _nn_index(smplx_rest, mhr_rest)  # one SMPL-X vertex per MHR vertex

    # Step 1: pose SMPL-X -> vertices (B, 10475, 3)
    smplx_layer.prepare_identity(torch.zeros(1, num_betas, device=device))
    with torch.no_grad():
        smplx_out = smplx_layer.pose(
            poses_t,
            pose2rot=True,
            apply_correctives=False,
            absolute_pose=True,
            global_translation=transl_t,
        )
    fit_verts = _verts(smplx_out)[:, idx, :]  # (B, 18056, 3) on the MHR topology

    # Step 2: fit MHR rotations to the bridged surface.
    mhr_layer.prepare_identity(torch.zeros(1, MHR_IDENTITY_DIM, device=device))
    inv = PoseInversion(mhr_layer, low_lod=False)
    with torch.no_grad():
        inv.prepare_identity(torch.zeros(1, MHR_IDENTITY_DIM, device=device))
        inv_result = inv.fit(fit_verts)
    rotations = inv_result["rotations"]          # (B, 78, 3, 3) absolute
    root_translation = inv_result["root_translation"]  # (B, 3)
    poses_77 = rotations[:, 1:, :, :]            # drop the extra root joint

    # Step 3: reconstruct and measure the fit error on the MHR topology.
    with torch.no_grad():
        recon = _verts(mhr_layer.pose(
            poses_77, transl=root_translation, pose2rot=False,
            apply_correctives=False, absolute_pose=True,
        ))

    return {
        "mhr_poses": _rotmat_to_aa(poses_77).tolist(),   # (B, 77, 3)
        "mhr_transl": root_translation.cpu().numpy().tolist(),
        **_errors(recon, fit_verts),
        **_bridge_stats(nn_dist),
    }


def _op_mhr_to_smplx(payload: dict) -> dict:
    import numpy as np
    import torch
    from soma.smpl.transfer import PoseInversion

    device = payload.get("device", "cpu")
    data_root = payload["data_root"]
    num_betas = int(payload.get("num_betas", 16))

    poses_np = np.array(payload["poses"], dtype=np.float32)    # (B, 77*3)
    transl_np = np.array(payload["transl"], dtype=np.float32)  # (B, 3)
    B = poses_np.shape[0]

    mhr_layer = _mhr_layer(data_root, device)
    smplx_layer = _smplx_layer(data_root, num_betas, device)
    _patch_mhr_layer(mhr_layer, device)

    J_mhr = poses_np.shape[1] // 3
    if J_mhr != MHR_PUBLIC_JOINTS:
        raise ValueError(f"Expected {MHR_PUBLIC_JOINTS} MHR joints, got {J_mhr}")
    poses_t = torch.as_tensor(poses_np.reshape(B, J_mhr, 3), device=device)
    transl_t = torch.as_tensor(transl_np, device=device)

    J_smplx = smplx_layer.num_joints
    mhr_rest = _rest_vertices(mhr_layer, MHR_IDENTITY_DIM, MHR_PUBLIC_JOINTS, device)
    smplx_rest = _rest_vertices(smplx_layer, num_betas, J_smplx, device)
    idx, nn_dist = _nn_index(mhr_rest, smplx_rest)  # one MHR vertex per SMPL-X vertex

    # Step 1: pose MHR -> vertices (B, 18056, 3), absolute rotations.
    mhr_layer.prepare_identity(torch.zeros(1, MHR_IDENTITY_DIM, device=device))
    with torch.no_grad():
        mhr_out = mhr_layer.pose(
            poses_t, transl=transl_t, pose2rot=True,
            apply_correctives=False, absolute_pose=True,
        )
    fit_verts = _verts(mhr_out)[:, idx, :]  # (B, 10475, 3) on the SMPL-X topology

    # Step 2: fit SMPL-X rotations to the bridged surface.
    smplx_layer.prepare_identity(torch.zeros(1, num_betas, device=device))
    inv = PoseInversion(smplx_layer, low_lod=False)
    with torch.no_grad():
        inv.prepare_identity(torch.zeros(1, num_betas, device=device))
        inv_result = inv.fit(fit_verts)
    rotations = inv_result["rotations"]                # (B, 55, 3, 3) absolute
    root_translation = inv_result["root_translation"]  # (B, 3)
    if rotations.shape[1] != J_smplx:
        raise ValueError(
            f"PoseInversion returned {rotations.shape[1]} SMPL-X joints; expected {J_smplx}"
        )

    # Step 3: reconstruct and measure the fit error on the SMPL-X topology.
    with torch.no_grad():
        recon = _verts(smplx_layer.pose(
            rotations, pose2rot=False, apply_correctives=False,
            absolute_pose=True, global_translation=root_translation,
        ))

    return {
        "smplx_poses": _rotmat_to_aa(rotations).tolist(),  # (B, 55, 3)
        "smplx_transl": root_translation.cpu().numpy().tolist(),
        **_errors(recon, fit_verts),
        **_bridge_stats(nn_dist),
    }


def _op_vertex_rmse(payload: dict) -> dict:
    """Vertex error between two SMPL-X pose sequences (absolute rotations)."""
    import numpy as np
    import torch

    device = payload.get("device", "cpu")
    data_root = payload["data_root"]
    num_betas = int(payload.get("num_betas", 16))

    poses_a = torch.as_tensor(np.array(payload["poses_a"], dtype=np.float32), device=device)
    transl_a = torch.as_tensor(np.array(payload["transl_a"], dtype=np.float32), device=device)
    poses_b = torch.as_tensor(np.array(payload["poses_b"], dtype=np.float32), device=device)
    transl_b = torch.as_tensor(np.array(payload["transl_b"], dtype=np.float32), device=device)

    layer = _smplx_layer(data_root, num_betas, device)
    layer.prepare_identity(torch.zeros(1, num_betas, device=device))

    with torch.no_grad():
        # Same convention as the transfer ops: absolute rotations.
        va = _verts(layer.pose(poses_a, pose2rot=True, apply_correctives=False,
                               absolute_pose=True, global_translation=transl_a))
        vb = _verts(layer.pose(poses_b, pose2rot=True, apply_correctives=False,
                               absolute_pose=True, global_translation=transl_b))
    err = _errors(va, vb)
    return {"rmse_m": err["per_vertex_rmse_m"],
            "mean_vertex_error_m": err["mean_vertex_error_m"]}


_OPS = {
    "smplx_to_mhr": _op_smplx_to_mhr,
    "mhr_to_smplx": _op_mhr_to_smplx,
    "vertex_rmse": _op_vertex_rmse,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="SOMA-X worker (hready-soma env)")
    parser.add_argument("--input", required=True, help="Path to JSON input file")
    parser.add_argument("--output", required=True, help="Path to write JSON result")
    args = parser.parse_args()

    try:
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
        op = payload.get("op")
        if op not in _OPS:
            raise ValueError(f"Unknown op {op!r}; expected one of {list(_OPS)}")
        result = _OPS[op](payload)
        Path(args.output).write_text(json.dumps(result), encoding="utf-8")
    except Exception:
        sys.stderr.write(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
