"""E3 learned completion: evidence + rig -> SMPL-X motion (transl, 6D rotations) + 4 foot-contact logits.

``forward(obs, rig)`` is the only input path: evidence-schema tensors plus the given rig (head pose,
camera rotation, gravity; D1). No pose initialisation, no targets, no body shape.
Per-frame encoder (all 22 joints + rig) -> temporal Transformer (HR-Refine width) -> heads.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from hready.body.rotations import (
    geodesic_distance,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)
from hready.data.ego_observation import RIG_SCHEMA_KEYS, assert_obs_only_batch

NUM_KP = 22
NUM_BODY = 21
NUM_CONTACT = 4
REQUIRED_RIG_KEYS: tuple[str, ...] = ("head_pos_world", "camera_R", "gravity_world")
_JOINT_FEATS = (
    3 + 1 + 1 + 3
)  # head-relative pos, visible, confidence, keypoint (nx, ny, depth)
_RIG_FEATS = 3 + 6 + 3


def check_rig(rig: dict[str, Tensor]) -> None:
    missing = [k for k in REQUIRED_RIG_KEYS if k not in rig]
    if missing:
        raise ValueError(f"rig is mandatory; missing keys: {missing}")
    extra = set(rig) - RIG_SCHEMA_KEYS
    if extra:
        raise ValueError(f"rig has keys outside the rig schema: {sorted(extra)}")


def evidence_features(obs: dict[str, Tensor], rig: dict[str, Tensor]) -> Tensor:
    """``(B, T, 22*8 + 12)``. Every per-joint channel is zeroed where the joint is not visible."""
    assert_obs_only_batch(obs)
    check_rig(rig)
    head = rig["head_pos_world"]  # (B, T, 3)
    b, t = head.shape[:2]
    vis = obs["joint_visible"].to(head.dtype).unsqueeze(-1)  # (B, T, 22, 1)
    rel = (obs["joint_pos_3d"] - head.unsqueeze(2)) * vis
    conf = obs["joint_confidence"].to(head.dtype).unsqueeze(-1) * vis
    kp = obs.get("keypoints_2d")
    # E2-A keeps (nx, ny) of hidden joints; masking here keeps hidden-joint projections out.
    kp = torch.zeros_like(rel) if kp is None else kp.to(head.dtype) * vis
    joint = torch.cat([rel, vis, conf, kp], dim=-1).reshape(b, t, NUM_KP * _JOINT_FEATS)
    g = rig["gravity_world"].to(head.dtype)
    g = g.reshape(b, 1, 3).expand(b, t, 3) if g.ndim < 3 else g
    rig_f = torch.cat(
        [head, matrix_to_rotation_6d(rig["camera_R"].to(head.dtype)), g / 9.81], dim=-1
    )
    return torch.cat([joint, rig_f], dim=-1)


class EgoCompleteMotion(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
        max_len: int = 64,
    ) -> None:
        super().__init__()
        in_dim = NUM_KP * _JOINT_FEATS + _RIG_FEATS
        self.frame_in = nn.Sequential(
            nn.Linear(in_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model,
            n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(
            layer, n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)
        self.head_transl = nn.Linear(d_model, 3)
        self.head_root = nn.Linear(d_model, 6)
        self.head_body = nn.Linear(d_model, NUM_BODY * 6)
        self.head_contact = nn.Linear(d_model, NUM_CONTACT)
        self.register_buffer(
            "rot6d_identity", matrix_to_rotation_6d(torch.eye(3)), persistent=False
        )

    def forward(
        self, obs: dict[str, Tensor], rig: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        x = evidence_features(obs, rig)
        b, t, _ = x.shape
        if t > self.pos.shape[1]:
            raise ValueError(f"window {t} > max_len {self.pos.shape[1]}")
        h = self.norm(self.temporal(self.frame_in(x) + self.pos[:, :t]))
        ident = self.rot6d_identity.to(h.dtype)
        return {
            # transl of the neutral-shape body, predicted as an offset from the given head position.
            "transl": rig["head_pos_world"].to(h.dtype) + self.head_transl(h),
            "root_rot_6d": self.head_root(h) + ident,
            "body_rot_6d": (
                self.head_body(h).reshape(b, t, NUM_BODY, 6) + ident
            ).reshape(b, t, NUM_BODY * 6),
            "contact_logits": self.head_contact(h),
        }


def geodesic_pose_loss(
    pred_root6: Tensor,
    pred_body6: Tensor,
    gt_root_R: Tensor,
    gt_body_R: Tensor,
    valid: Tensor,
) -> Tensor:
    """Mean geodesic angle (rad) over valid frames: root + mean over the 21 body joints."""
    b, t = valid.shape
    pr = rotation_6d_to_matrix(pred_root6)
    pb = rotation_6d_to_matrix(pred_body6.reshape(b, t, NUM_BODY, 6))
    w = valid.to(pr.dtype)
    lr = (geodesic_distance(pr, gt_root_R) * w).sum() / w.sum().clamp_min(1.0)
    lb = (geodesic_distance(pb, gt_body_R).mean(dim=-1) * w).sum() / w.sum().clamp_min(
        1.0
    )
    return lr + lb
