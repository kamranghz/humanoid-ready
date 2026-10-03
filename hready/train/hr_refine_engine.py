"""HR-Refine training loop utilities (DDP, checkpoint, loss)."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

from hready.body.batch_forward import smpl_forward_bt
from hready.body.smplx_wrapper import SmplxBody, load_body
from hready.data.refine_corrupt import (
    CorruptionConfig,
    VirtualCameraConfig,
    apply_corruption,
    make_corruption_rng,
    unpack_pose,
)
from hready.losses import LossConfig, compute_losses
from hready.models.hr_refine import HRRefine

FOOT_CHANNEL_JOINTS = (7, 10, 8, 11)


@dataclass
class TrainHyper:
    lr: float = 3e-4
    weight_decay: float = 0.01
    recon_joint: float = 1.0
    recon_rot: float = 1.0
    physics_scale: float = 1.0
    grad_clip: float = 1.0
    smpl_joint_recon: bool = True
    joints_only_recon: bool = False


@dataclass
class FixedCorruptionBatch:
    corrupt_pose: dict[str, Tensor]
    keypoints: Tensor
    clean_packed: dict[str, Tensor]
    joints_clean: Tensor


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "configs" / "paths.example.yaml").is_file():
            return parent
    raise FileNotFoundError("repo root not found")


def setup_distributed() -> tuple[int, int, int, torch.device]:
    if os.name == "nt":
        os.environ.setdefault("USE_LIBUV", "0")
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        world_size = int(os.environ["WORLD_SIZE"])
        if not dist.is_initialized():
            backend = "gloo" if os.name == "nt" else "nccl"
            dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        return rank, local_rank, world_size, device
    return 0, 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def set_deterministic_training(enabled: bool = True) -> None:
    if enabled:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def smpl_parents(body: SmplxBody) -> Tensor:
    return body._model.parents.detach().clone().to(dtype=torch.long)


def foot_tensors(joints: Tensor, fps: float) -> tuple[Tensor, Tensor]:
    foot_pos = joints[..., list(FOOT_CHANNEL_JOINTS), :]
    z = foot_pos[..., 2]
    contact = (z < 0.05).float()
    return foot_pos, contact


def reconstruction_loss(
    pred: dict[str, Tensor],
    clean_packed: dict[str, Tensor],
    joints_pred: Tensor,
    joints_clean: Tensor,
    *,
    w_joint: float,
    w_rot: float,
    use_joint_smpl: bool = True,
    joints_only: bool = False,
) -> Tensor:
    r_loss = (
        (pred["root_rot_6d"] - clean_packed["root_rot_6d"]).abs().mean()
        + (pred["body_rot_6d"] - clean_packed["body_rot_6d"]).abs().mean()
    )
    t_loss = (pred["transl"] - clean_packed["transl"]).abs().mean()
    if use_joint_smpl:
        j_loss = (joints_pred[..., :22, :] - joints_clean[..., :22, :]).abs().mean()
        if joints_only:
            return w_joint * j_loss
        return w_joint * j_loss + w_rot * r_loss + w_joint * t_loss
    return w_rot * r_loss + w_joint * t_loss


def prepare_fixed_corruption(
    body: SmplxBody,
    batch: dict[str, Tensor],
    *,
    base_seed: int,
    corrupt_step: int,
    corrupt_cfg: CorruptionConfig,
    cam_cfg: VirtualCameraConfig | None = None,
) -> FixedCorruptionBatch:
    clean = {
        "transl": batch["transl"],
        "root_aa": batch["root_orient"],
        "body_aa": batch["pose_body"],
    }
    with torch.no_grad():
        joints_clean, _ = smpl_forward_bt(
            body, clean["transl"], clean["root_aa"], clean["body_aa"], batch["betas"]
        )
    corrupted = apply_corruption(
        clean,
        joints_clean,
        rng=make_corruption_rng(base_seed, corrupt_step),
        cfg=corrupt_cfg,
        cam_cfg=cam_cfg or VirtualCameraConfig(),
    )
    return FixedCorruptionBatch(
        corrupt_pose=corrupted["corrupt_pose"],
        keypoints=corrupted["keypoints"],
        clean_packed=corrupted["clean_pose"],
        joints_clean=joints_clean,
    )


def forward_loss(
    model: nn.Module,
    body: SmplxBody,
    batch: dict[str, Tensor],
    *,
    parents: Tensor,
    loss_cfg: LossConfig,
    hyper: TrainHyper,
    corrupt_cfg: CorruptionConfig,
    base_seed: int = 0,
    corrupt_step: int = 0,
    fixed: FixedCorruptionBatch | None = None,
    use_bf16: bool = False,
) -> dict[str, Tensor | float]:
    device = batch["transl"].device
    b, t = batch["transl"].shape[:2]
    if fixed is None:
        fixed = prepare_fixed_corruption(
            body,
            batch,
            base_seed=base_seed,
            corrupt_step=corrupt_step,
            corrupt_cfg=corrupt_cfg,
        )
    joints_clean = fixed.joints_clean
    corrupt_pose = fixed.corrupt_pose
    keypoints = fixed.keypoints
    clean_packed = fixed.clean_packed

    with torch.amp.autocast("cuda", enabled=use_bf16 and device.type == "cuda"):
        pred = model(corrupt_pose, keypoints)
    transl_p, root_aa_p, body_aa_p = unpack_pose(pred)
    need_pred_smpl = hyper.smpl_joint_recon or hyper.physics_scale > 0
    if need_pred_smpl:
        joints_pred, verts_pred = smpl_forward_bt(body, transl_p, root_aa_p, body_aa_p, batch["betas"])
    else:
        joints_pred, verts_pred = joints_clean, joints_clean
    with torch.amp.autocast("cuda", enabled=use_bf16 and device.type == "cuda"):
        recon = reconstruction_loss(
            pred,
            clean_packed,
            joints_pred,
            joints_clean,
            w_joint=hyper.recon_joint,
            w_rot=hyper.recon_rot,
            use_joint_smpl=hyper.smpl_joint_recon,
            joints_only=hyper.joints_only_recon,
        )
    if hyper.physics_scale > 0:
        foot_pos, contact = foot_tensors(joints_pred, float(batch["fps"]))
        phys = compute_losses(
            {
                "joints": joints_pred.float(),
                "verts_or_joints": verts_pred.float(),
                "foot_pos": foot_pos.float(),
                "foot_pts": foot_pos.float(),
                "contact": contact.float(),
                "fps": batch["fps"],
                "parents": parents.to(device),
                "body_pose": body_aa_p.reshape(b, t, 21, 3).float(),
                "body_pose_rot_repr": "axis_angle",
            },
            loss_cfg,
        )["total"]
        loss = recon.float() + hyper.physics_scale * phys
    else:
        phys = torch.zeros((), device=device)
        loss = recon.float()
    return {"loss": loss, "recon": recon, "physics": phys}


def backward_step(
    model: nn.Module,
    body: SmplxBody,
    batch: dict[str, Tensor],
    optimizer: torch.optim.Optimizer,
    *,
    parents: Tensor,
    loss_cfg: LossConfig,
    hyper: TrainHyper,
    corrupt_cfg: CorruptionConfig,
    base_seed: int = 0,
    corrupt_step: int = 0,
    fixed: FixedCorruptionBatch | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    use_bf16: bool = True,
) -> dict[str, float]:
    out = forward_loss(
        model,
        body,
        batch,
        parents=parents,
        loss_cfg=loss_cfg,
        hyper=hyper,
        corrupt_cfg=corrupt_cfg,
        base_seed=base_seed,
        corrupt_step=corrupt_step,
        fixed=fixed,
        use_bf16=use_bf16,
    )
    loss = out["loss"]
    optimizer.zero_grad(set_to_none=True)
    if scaler is not None and use_bf16 and batch["transl"].device.type == "cuda":
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), hyper.grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), hyper.grad_clip)
        optimizer.step()
    return {
        "loss": float(loss.detach().cpu()),
        "recon": float(out["recon"].detach().cpu()),
        "physics": float(out["physics"].detach().cpu()),
    }


def wrap_ddp(model: HRRefine, device: torch.device, local_rank: int) -> nn.Module:
    if dist.is_initialized() and dist.get_world_size() > 1:
        return DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
    return model


def unwrap(model: nn.Module) -> HRRefine:
    return model.module if isinstance(model, DDP) else model


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    step: int,
    base_seed: int,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "base_seed": base_seed,
        "config": config,
        "model": unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_random": torch.get_rng_state(),
        "cuda_random": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    unwrap(model).load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if restore_rng:
        random.setstate(ckpt["python_random"])
        np.random.set_state(ckpt["numpy_random"])
        torch.set_rng_state(ckpt["torch_random"])
        if ckpt.get("cuda_random") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ckpt["cuda_random"])
    return ckpt


def build_model_and_body(
    model_cfg: dict[str, Any], device: torch.device
) -> tuple[HRRefine, SmplxBody, Tensor]:
    model = HRRefine(**model_cfg).to(device)
    body = load_body("locked_head")
    body._model.to(device)
    body._model.eval()
    parents = smpl_parents(body)
    return model, body, parents


def max_param_diff(a: nn.Module, b: nn.Module) -> float:
    d = 0.0
    for (ka, pa), (kb, pb) in zip(a.state_dict().items(), b.state_dict().items()):
        if ka != kb:
            raise KeyError(ka, kb)
        diff = (pa.float() - pb.float()).abs().max().item()
        d = max(d, diff)
    return d
