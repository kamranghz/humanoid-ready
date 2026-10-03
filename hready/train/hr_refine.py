"""CLI: ``python -m hready.train.hr_refine --config configs/hr_refine.yaml``."""

from __future__ import annotations

import argparse
import os

os.environ["USE_LIBUV"] = "0"
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from hready.data.refine_corrupt import corruption_config_from_dict
from hready.data.refine_dataset import HRRefineWindowDataset, collate_windows
from hready.losses import LossConfig, loss_config_from_dict
from hready.train.hr_refine_engine import (
    TrainHyper,
    backward_step,
    build_model_and_body,
    cleanup_distributed,
    load_checkpoint,
    repo_root,
    save_checkpoint,
    setup_distributed,
    unwrap,
    wrap_ddp,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _make_loader(split: str, batch_size: int, window: int, seed: int) -> DataLoader:
    ds = HRRefineWindowDataset(split=split, window=window, seed=seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_windows,
        drop_last=True,
    )


def train_main(cfg: dict[str, Any]) -> None:
    rank, local_rank, world_size, device = setup_distributed()
    base_seed = int(cfg.get("seed", 0))
    torch.manual_seed(base_seed + rank)
    window = int(cfg.get("window", 64))
    batch_size = int(os.environ.get("HR_REFINE_BATCH_SIZE", cfg.get("batch_size", 8)))
    max_steps = int(os.environ.get("HR_REFINE_MAX_STEPS", cfg.get("max_steps", 10_000)))
    log_every = int(cfg.get("log_every", 50))
    save_every = int(cfg.get("save_every", 500))
    use_bf16 = bool(cfg.get("bf16", True))
    model_cfg = dict(cfg.get("model", {}))
    hyper = TrainHyper(
        lr=float(cfg.get("lr", 3e-4)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        recon_joint=float(cfg.get("loss", {}).get("recon_joint", 1.0)),
        recon_rot=float(cfg.get("loss", {}).get("recon_rot", 1.0)),
        physics_scale=float(cfg.get("loss", {}).get("physics", 1.0)),
    )
    loss_cfg = loss_config_from_dict(cfg.get("physics_loss"))
    corrupt_cfg = corruption_config_from_dict(cfg.get("corruption"))
    ckpt_dir = repo_root() / cfg.get("checkpoint_dir", "results/A2/hr_refine")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model, body, parents = build_model_and_body(model_cfg, device)
    model = wrap_ddp(model, device, local_rank)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=hyper.lr, weight_decay=hyper.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=use_bf16 and device.type == "cuda")

    step = 0
    resume_path = cfg.get("resume")
    if resume_path:
        ckpt = load_checkpoint(Path(resume_path), model=model, optimizer=optimizer, scheduler=scheduler)
        step = int(ckpt["step"])

    loader = _make_loader("train", batch_size, window, base_seed)
    data_iter = iter(loader)
    t0 = time.perf_counter()
    if rank == 0:
        print(f"HR-Refine train world_size={world_size} device={device} params={unwrap(model).num_parameters()}")

    while step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
        stats = backward_step(
            model,
            body,
            batch,
            optimizer,
            parents=parents,
            loss_cfg=loss_cfg,
            hyper=hyper,
            corrupt_cfg=corrupt_cfg,
            base_seed=base_seed,
            corrupt_step=step,
            scaler=scaler,
            use_bf16=use_bf16,
        )
        scheduler.step()
        if rank == 0 and step % log_every == 0:
            print(f"step={step} loss={stats['loss']:.6f} recon={stats['recon']:.6f} phys={stats['physics']:.6f}")
        if rank == 0 and save_every > 0 and step > 0 and step % save_every == 0:
            save_checkpoint(
                ckpt_dir / f"step_{step:06d}.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                base_seed=base_seed,
                config=cfg,
            )
        step += 1

    if rank == 0:
        print(f"done steps={step} wall_s={time.perf_counter() - t0:.1f}")
        save_checkpoint(
            ckpt_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            base_seed=base_seed,
            config=cfg,
        )
    cleanup_distributed()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train HR-Refine on AMASS windows.")
    parser.add_argument("--config", type=Path, required=True, help="YAML config path")
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint to resume")
    args = parser.parse_args(argv)
    cfg = _load_yaml(args.config.resolve())
    if args.resume is not None:
        cfg["resume"] = str(args.resume)
    train_main(cfg)


if __name__ == "__main__":
    main()
