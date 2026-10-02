# Environment decisions

### 2b. Environment decision (S3, Oct 1 2026)

- Machine: Windows 11, **RTX 4090 24 GB** (driver 610.60, CUDA 13.3 runtime available), data on `D:\projects\hready_data`.
- **Windows-native conda** for all project code: env `hready` (PyTorch, smplx, training, data, metrics). Reasons: Isaac Sim/Lab already run natively on Windows, all existing envs are Windows conda, and reading `D:` from WSL2 (`/mnt/d`) is slow.
- **Isaac Lab + Newton:** reuse the existing Isaac Lab installation (separate env); `hready` talks to it through files (retargeted trajectories in, metrics/videos out), not imports.
- **WSL2 only as fallback**, per baseline, if a public HMR method (GVHMR, WHAM, TRAM) does not install on Windows (Linux-only CUDA extensions). Log every such case in `docs/pivot_log.md`.
- DDP on Windows uses the `gloo` backend (no NCCL); multi-GPU NCCL runs happen on Kaggle (Linux). `torch.compile` is optional on Windows.
