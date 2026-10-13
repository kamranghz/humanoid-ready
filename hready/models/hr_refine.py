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
        imu_dim: int = 0,
        contact_channels: int = 0,
        n_actions: int = 0,
        intent_horizon: int = 0,
    ) -> None:
        """Optional extras (all off by default, so the refiner is unchanged):

        ``imu_dim`` per-frame IMU feature size (0 = no IMU input); ``contact_channels`` per-frame
        foot-contact logits; ``n_actions`` per-frame action-class logits; ``intent_horizon`` number of
        future root-translation frames predicted by :meth:`forecast`.
        """
        super().__init__()
        pose_dim = 3 + 6 + NUM_BODY_JOINTS * 6
        kp_dim = NUM_KP_JOINTS * 3
        self.pose_proj = nn.Linear(pose_dim, d_model)
        self.kp_proj = nn.Linear(kp_dim, d_model)
        self.imu_proj = nn.Linear(imu_dim, d_model) if imu_dim > 0 else None
        self.fuse = nn.Linear(d_model * (3 if imu_dim > 0 else 2), d_model)
        self.intent_horizon = int(intent_horizon)
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
        self.head_contact = nn.Linear(d_model, contact_channels) if contact_channels > 0 else None
        self.head_action = nn.Linear(d_model, n_actions) if n_actions > 0 else None
        self.head_intent = nn.Linear(d_model, intent_horizon * 3) if intent_horizon > 0 else None
        self._init_weights()
        for head in (self.head_trans, self.head_root, self.head_body, self.head_contact, self.head_action, self.head_intent):
            if head is not None:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode_inputs(
        self, corrupt_pose: dict[str, Tensor], keypoints: Tensor, imu: Tensor | None = None
    ) -> Tensor:
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
        parts = [self.pose_proj(pose_cat), self.kp_proj(kp)]
        if self.imu_proj is not None:
            if imu is None:
                raise ValueError("this HRRefine was built with imu_dim > 0; pass imu")
            parts.append(self.imu_proj(imu.reshape(b, t, -1)))
        return self.encoder(self.fuse(torch.cat(parts, dim=-1)))

    def forward(
        self,
        corrupt_pose: dict[str, Tensor],
        keypoints: Tensor,
        imu: Tensor | None = None,
    ) -> dict[str, Tensor]:
        h = self.encode_inputs(corrupt_pose, keypoints, imu)
        out = {
            "transl": corrupt_pose["transl"] + self.head_trans(h),
            "root_rot_6d": corrupt_pose["root_rot_6d"] + self.head_root(h),
            "body_rot_6d": corrupt_pose["body_rot_6d"] + self.head_body(h),
        }
        if self.head_contact is not None:
            out["contact_logits"] = self.head_contact(h)
        if self.head_action is not None:
            out["action_logits"] = self.head_action(h)
        return out

    def forecast(
        self,
        corrupt_pose: dict[str, Tensor],
        keypoints: Tensor,
        n_observed: int,
        imu: Tensor | None = None,
    ) -> Tensor:
        """Predict the next ``intent_horizon`` root-translation offsets (B, H, 3).

        Only the first ``n_observed`` frames are given to the encoder, so no future frame is seen.
        Offsets are relative to the last observed frame's translation. The IMU feature at frame ``t``
        is a second difference of positions ``t-1..t+1``, so the last observed frame's IMU feature would
        reveal the first future frame; it is zeroed here.
        """
        if self.head_intent is None:
            raise RuntimeError("built with intent_horizon=0")
        cut = {k: v[:, :n_observed] for k, v in corrupt_pose.items()}
        imu_obs = None
        if imu is not None:
            imu_obs = imu[:, :n_observed].clone()
            imu_obs[:, -1] = 0.0
        h = self.encode_inputs(cut, keypoints[:, :n_observed], imu_obs)
        return self.head_intent(h[:, -1]).reshape(-1, self.intent_horizon, 3)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# Extension points (not implemented): multi-view cameras, egocentric head cam.
MULTIVIEW_PLACEHOLDER = "hr_refine.multiview_tokens"
EGO_PLACEHOLDER = "hr_refine.ego_camera_token"
