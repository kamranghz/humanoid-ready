"""Decoupled foot-grounding stage: a small temporal convolution that corrects the foot-channel joints only.

Input: the four foot-channel points (L heel, L toe, R heel, R toe; sole-proxy heights, floor at z = 0) of a motion
``(B, T, 4, 3)`` and the calibrated contact probabilities ``(B, T, 4)``. Output: an additive 3-D displacement
``(B, T, 4, 3)`` for those joints. Horizontal position enters only through velocities, so the stage does not depend on
the window's xy origin. The output layer is zero-initialised: the untrained stage is the identity.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

NUM_FOOT_CHANNELS = 4
FEATURES_PER_CHANNEL = 5  # height, vx, vy, vz, contact probability
_HEIGHT_SCALE = 10.0  # m -> O(1)


def foot_features(foot_pts: Tensor, prob: Tensor, fps: float) -> Tensor:
    """``(B, T, 4 * 5)`` features: height (x10), velocity (m/s) per axis, contact probability."""
    vel = torch.zeros_like(foot_pts)
    vel[:, 1:] = (foot_pts[:, 1:] - foot_pts[:, :-1]) * fps
    feats = torch.cat(
        [foot_pts[..., 2:3] * _HEIGHT_SCALE, vel, prob.unsqueeze(-1)], dim=-1
    )  # (B, T, 4, 5)
    return feats.flatten(2)


class GroundingStage(nn.Module):
    def __init__(
        self,
        channels: int = 64,
        kernel: int = 5,
        dilations: tuple[int, ...] = (1, 2, 4, 1),
        displacement_scale: float = 0.1,
        fps: float = 30.0,
    ) -> None:
        super().__init__()
        self.kernel = kernel
        self.dilations = tuple(dilations)
        self.displacement_scale = float(displacement_scale)
        self.fps = float(fps)
        dims = [NUM_FOOT_CHANNELS * FEATURES_PER_CHANNEL] + [channels] * len(self.dilations)
        self.convs = nn.ModuleList(
            nn.Conv1d(dims[i], dims[i + 1], kernel, dilation=d) for i, d in enumerate(self.dilations)
        )
        self.out = nn.Conv1d(channels, NUM_FOOT_CHANNELS * 3, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    @property
    def receptive_field(self) -> int:
        return 1 + (self.kernel - 1) * sum(self.dilations)

    def forward(self, foot_pts: Tensor, prob: Tensor) -> Tensor:
        b, t = foot_pts.shape[:2]
        x = foot_features(foot_pts, prob, self.fps).transpose(1, 2)  # (B, F, T)
        for conv, d in zip(self.convs, self.dilations):
            pad = d * (self.kernel - 1) // 2
            x = F.gelu(conv(F.pad(x, (pad, pad), mode="replicate")))
        delta = self.out(x).transpose(1, 2).reshape(b, t, NUM_FOOT_CHANNELS, 3)
        return self.displacement_scale * delta
