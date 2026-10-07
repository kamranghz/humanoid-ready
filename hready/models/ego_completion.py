"""Track E2-A: completion encoder (obs-only forward; training in E3/E4)."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from hready.body.joint_indices import LEG_BODY_AA_INDICES, LOWER_BODY_JOINTS
from hready.data.ego_observation import FORBIDDEN_OBS_KEYS, assert_obs_only_batch

NUM_BODY_JOINTS = 21
NUM_KP_JOINTS = 22


class EgoCompletion(nn.Module):
    """Fuse evidence tensors only -> per-joint 3D predictions (E3/E4 placeholder head)."""

    def __init__(self, d_model: int = 256, n_layers: int = 2) -> None:
        super().__init__()
        self.joint_proj = nn.Linear(3 + 1, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead=4,
            dim_feedforward=d_model * 2,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.head = nn.Linear(d_model, 3)

    def forward(self, obs: dict[str, Tensor]) -> Tensor:
        assert_obs_only_batch(obs)
        b, t, j, _ = obs["joint_pos_3d"].shape
        conf = obs["joint_confidence"].unsqueeze(-1)
        vis = obs["joint_visible"].float().unsqueeze(-1)
        x = torch.cat([obs["joint_pos_3d"], conf * vis], dim=-1)
        h = self.joint_proj(x.reshape(b, t * j, -1))
        h = self.encoder(h)
        return self.head(h).reshape(b, t, j, 3)


def forward_rejects_full_clip(model: nn.Module) -> tuple[bool, str]:
    try:
        corrupt_pose = {
            "transl": torch.zeros(1, 64, 3),
            "root_rot_6d": torch.zeros(1, 64, 6),
            "body_rot_6d": torch.zeros(1, 64, 21 * 6),
        }
        keypoints = torch.zeros(1, 64, 22, 3)
        model.forward(corrupt_pose, keypoints)  # type: ignore[arg-type]
        return False, "forward accepted corrupt_pose+keypoints (HR-Refine leak)"
    except TypeError as exc:
        return True, f"TypeError: {exc}"
    except Exception as exc:
        return True, f"rejected full clip: {type(exc).__name__}: {exc}"


def run_leak_checks(
    model: EgoCompletion,
    batch: dict[str, Any],
    *,
    regenerate_obs_negative_control,
    seed: int = 0,
) -> dict[str, Any]:
    """Leak checks (a)–(d) on a collated batch and randomly initialised ``EgoCompletion``."""
    torch.manual_seed(seed)
    model = model.train()
    leg_kp = list(LOWER_BODY_JOINTS)
    leg_ba = list(LEG_BODY_AA_INDICES)

    vis = batch["obs"]["joint_visible"][0]
    all_legs_hidden = not bool(vis[:, leg_kp].any().item())

    model.eval()
    obs0 = batch["obs"]
    out_a = model(obs0)
    targets_leg = batch["targets"]["body_aa"].clone()
    targets_leg[..., leg_ba, :] += 0.37
    out_b = model(obs0)
    check_a = {
        "module": "hready.models.ego_completion.EgoCompletion",
        "all_leg_keypoints_hidden_in_obs": all_legs_hidden,
        "perturbation": f"targets.body_aa[{leg_ba}] += 0.37 (obs unchanged)",
        "bit_identical_output": bool(torch.equal(out_a, out_b)),
        "max_abs_diff": float((out_a - out_b).abs().max().item()),
    }

    knee_body_idx = 3
    left_knee_kp = 4
    obs_clean, obs_pert, neg_vis, fi = regenerate_obs_negative_control(seed=seed)
    delta = float(
        (
            obs_pert["joint_pos_3d"][fi, left_knee_kp]
            - obs_clean["joint_pos_3d"][fi, left_knee_kp]
        )
        .abs()
        .sum()
        .item()
    )
    check_a_neg = {
        "perturbation": "body_aa[:,3,:]+=0.5 vs +0.25, re-run simulate_oracle_evidence (same seed)",
        "left_knee_kp_index": left_knee_kp,
        "frame": fi,
        "obs_joint_pos_l1_delta": delta,
        "obs_changed": bool(delta > 1e-6),
    }

    model.train()
    obs_grad = {
        k: (v.detach().clone().requires_grad_(k == "joint_pos_3d"))
        for k, v in batch["obs"].items()
        if torch.is_tensor(v)
    }
    targets_grad = {
        k: v.detach().clone().requires_grad_(True) for k, v in batch["targets"].items() if torch.is_tensor(v)
    }
    out = model(obs_grad)
    out.sum().backward()
    target_grads = {
        k: (None if targets_grad[k].grad is None else float(targets_grad[k].grad.abs().sum().item()))
        for k in targets_grad
    }
    obs_pos_grad = float(obs_grad["joint_pos_3d"].grad.abs().sum().item())
    check_b = {
        "module": "hready.models.ego_completion.EgoCompletion",
        "loss": "output.sum()",
        "target_grad_abs_sum": target_grads,
        "targets_all_zero_or_none": all(g is None or g == 0.0 for g in target_grads.values()),
        "obs_joint_pos_3d_grad_abs_sum": obs_pos_grad,
        "obs_grad_nonzero": obs_pos_grad > 0.0,
    }

    forbidden = FORBIDDEN_OBS_KEYS.intersection(batch["obs"].keys())
    check_c = {"forbidden_keys_in_obs": sorted(forbidden), "pass": len(forbidden) == 0}
    ok_d, msg_d = forward_rejects_full_clip(model)

    return {
        "a_gt_leg_invisible": check_a,
        "a_negative_visible_leg_obs": check_a_neg,
        "b_backward": check_b,
        "c_forbidden_keys": check_c,
        "d_reject_full_clip": {"pass": ok_d, "message": msg_d},
    }
