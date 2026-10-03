"""Batched SMPL-X forward over (B, T) sequences."""

from __future__ import annotations

from torch import Tensor

from hready.body.smplx_wrapper import SmplxBody


def smpl_forward_bt(
    body: SmplxBody,
    transl: Tensor,
    root_aa: Tensor,
    body_aa: Tensor,
    betas: Tensor,
) -> tuple[Tensor, Tensor]:
    """``transl`` (B,T,3), ``root_aa`` (B,T,3), ``body_aa`` (B,T,21,3), ``betas`` (B,16)."""
    b, t = transl.shape[:2]
    bt = b * t
    go = root_aa.reshape(bt, 3)
    bp = body_aa.reshape(bt, 21, 3).reshape(bt, 63)
    be = betas.unsqueeze(1).expand(b, t, -1).reshape(bt, -1)
    tr = transl.reshape(bt, 3)
    out = body.forward(go, bp, be, tr, rot_repr="axis_angle")
    joints = out.joints.reshape(b, t, -1, 3)
    verts = out.vertices.reshape(b, t, -1, 3)
    return joints, verts
