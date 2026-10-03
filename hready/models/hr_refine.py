"""HR-Refine: spatio-temporal transformer (random init, item 7)."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from hready.data.refine_corrupt import NUM_BODY_JOINTS, NUM_KP_JOINTS


class HRRefine(nn.Module):
    """Fuse corrupted 6D pose + 2D keypoints -> clean root/body 6D + translation."""

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        pose_dim = 3 + 6 + NUM_BODY_JOINTS * 6
        kp_dim = NUM_KP_JOINTS * 3
        self.pose_proj = nn.Linear(pose_dim, d_model)
        self.kp_proj = nn.Linear(kp_dim, d_model)
        self.fuse = nn.Linear(d_model * 2, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model,
            n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.head_trans = nn.Linear(d_model, 3)
        self.head_root = nn.Linear(d_model, 6)
        self.head_body = nn.Linear(d_model, NUM_BODY_JOINTS * 6)
        self._init_weights()
        for head in (self.head_trans, self.head_root, self.head_body):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode_inputs(self, corrupt_pose: dict[str, Tensor], keypoints: Tensor) -> Tensor:
        b, t, _ = corrupt_pose["transl"].shape
        pose_cat = torch.cat(
            [
                corrupt_pose["transl"],
                corrupt_pose["root_rot_6d"],
                corrupt_pose["body_rot_6d"],
            ],
            dim=-1,
        )
        kp = keypoints.reshape(b, t, -1)
        h = self.fuse(torch.cat([self.pose_proj(pose_cat), self.kp_proj(kp)], dim=-1))
        return self.encoder(h)

    def forward(
        self,
        corrupt_pose: dict[str, Tensor],
        keypoints: Tensor,
    ) -> dict[str, Tensor]:
        h = self.encode_inputs(corrupt_pose, keypoints)
        return {
            "transl": corrupt_pose["transl"] + self.head_trans(h),
            "root_rot_6d": corrupt_pose["root_rot_6d"] + self.head_root(h),
            "body_rot_6d": corrupt_pose["body_rot_6d"] + self.head_body(h),
        }

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# Extension points (not implemented): multi-view cameras, egocentric head cam, IMU tokens.
MULTIVIEW_PLACEHOLDER = "hr_refine.multiview_tokens"
EGO_PLACEHOLDER = "hr_refine.ego_camera_token"
IMU_PLACEHOLDER = "hr_refine.imu_encoder"
