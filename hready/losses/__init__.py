"""Physics and biomechanics losses (lazy imports — no torch at ``import hready``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

__all__ = [
    "LossConfig",
    "LossTermConfig",
    "compute_losses",
    "bone_length_consistency",
    "com_from_joints",
    "flight_consistency",
    "foot_skating",
    "ground_penetration",
    "balance_com_in_support",
    "joint_rom",
    "smoothness",
]

_LAZY_EXPORTS = {
    "bone_length_consistency": ("hready.losses.biomech", "bone_length_consistency"),
    "joint_rom": ("hready.losses.biomech", "joint_rom"),
    "com_from_joints": ("hready.losses.physics", "com_from_joints"),
    "flight_consistency": ("hready.losses.physics", "flight_consistency"),
    "foot_skating": ("hready.losses.physics", "foot_skating"),
    "ground_penetration": ("hready.losses.physics", "ground_penetration"),
    "balance_com_in_support": ("hready.losses.physics", "balance_com_in_support"),
    "smoothness": ("hready.losses.physics", "smoothness"),
    "compute_losses": ("hready.losses", "_compute_losses_impl"),
}


@dataclass
class LossTermConfig:
    weight: float = 1.0
    enabled: bool = True


@dataclass
class LossConfig:
    foot_skating: LossTermConfig = field(default_factory=LossTermConfig)
    ground_penetration: LossTermConfig = field(default_factory=LossTermConfig)
    flight_consistency: LossTermConfig = field(default_factory=LossTermConfig)
    balance_com_in_support: LossTermConfig = field(default_factory=LossTermConfig)
    smoothness: LossTermConfig = field(default_factory=LossTermConfig)
    smoothness_accel_weight: float = 1.0
    smoothness_jerk_weight: float = 1.0
    bone_length_consistency: LossTermConfig = field(default_factory=LossTermConfig)
    joint_rom: LossTermConfig = field(default_factory=LossTermConfig)


def loss_config_from_dict(data: Mapping[str, Any] | None) -> LossConfig:
    """Build ``LossConfig`` from optional YAML ``physics_loss`` block."""
    cfg = LossConfig()
    if not data:
        return cfg
    term_names = (
        "foot_skating",
        "ground_penetration",
        "flight_consistency",
        "balance_com_in_support",
        "smoothness",
        "bone_length_consistency",
        "joint_rom",
    )
    for name in term_names:
        block = data.get(name)
        if block is None:
            continue
        term = getattr(cfg, name)
        if "enabled" in block:
            term.enabled = bool(block["enabled"])
        if "weight" in block:
            term.weight = float(block["weight"])
    if "smoothness_accel_weight" in data:
        cfg.smoothness_accel_weight = float(data["smoothness_accel_weight"])
    if "smoothness_jerk_weight" in data:
        cfg.smoothness_jerk_weight = float(data["smoothness_jerk_weight"])
    return cfg


def _compute_losses_impl(inputs: Mapping[str, Any], config: LossConfig) -> dict:
    import torch

    from hready.losses.biomech import bone_length_consistency, joint_rom
    from hready.losses.physics import (
        G,
        balance_com_in_support,
        com_from_joints,
        flight_consistency,
        foot_skating,
        ground_penetration,
        smoothness,
    )

    device = inputs["joints"].device
    dtype = inputs["joints"].dtype
    out: dict[str, torch.Tensor] = {}
    total = torch.zeros((), device=device, dtype=dtype)

    fps = inputs["fps"]
    joints = inputs["joints"]
    contact = inputs["contact"]
    foot_pos = inputs["foot_pos"]
    foot_pts = inputs.get("foot_pts", foot_pos)
    verts = inputs.get("verts_or_joints", joints)
    com = inputs.get("com")
    if com is None:
        com = com_from_joints(joints)

    if config.foot_skating.enabled:
        v = foot_skating(foot_pos, contact, fps)
        out["foot_skating"] = v
        total = total + config.foot_skating.weight * v
    else:
        out["foot_skating"] = total.new_zeros(())

    if config.ground_penetration.enabled:
        v = ground_penetration(verts)
        out["ground_penetration"] = v
        total = total + config.ground_penetration.weight * v
    else:
        out["ground_penetration"] = total.new_zeros(())

    if config.flight_consistency.enabled:
        v = flight_consistency(com, contact, fps, g=G)
        out["flight_consistency"] = v
        total = total + config.flight_consistency.weight * v
    else:
        out["flight_consistency"] = total.new_zeros(())

    if config.balance_com_in_support.enabled:
        v = balance_com_in_support(com, foot_pts, contact)
        out["balance_com_in_support"] = v
        total = total + config.balance_com_in_support.weight * v
    else:
        out["balance_com_in_support"] = total.new_zeros(())

    if config.smoothness.enabled:
        v = smoothness(
            joints,
            fps,
            accel_weight=config.smoothness_accel_weight,
            jerk_weight=config.smoothness_jerk_weight,
        )
        out["smoothness"] = v
        total = total + config.smoothness.weight * v
    else:
        out["smoothness"] = total.new_zeros(())

    parents = inputs["parents"]
    if config.bone_length_consistency.enabled:
        v = bone_length_consistency(joints, parents)
        out["bone_length_consistency"] = v
        total = total + config.bone_length_consistency.weight * v
    else:
        out["bone_length_consistency"] = total.new_zeros(())

    if config.joint_rom.enabled:
        body_pose = inputs["body_pose"]
        rot_repr = inputs.get("body_pose_rot_repr", "axis_angle")
        v = joint_rom(body_pose, rot_repr=rot_repr)
        out["joint_rom"] = v
        total = total + config.joint_rom.weight * v
    else:
        out["joint_rom"] = total.new_zeros(())

    out["total"] = total
    return out


def compute_losses(inputs: Mapping[str, Any], config: LossConfig) -> dict:
    return _compute_losses_impl(inputs, config)


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        module_path, attr = _LAZY_EXPORTS[name]
        if module_path == "hready.losses" and attr == "_compute_losses_impl":
            return _compute_losses_impl
        import importlib

        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
