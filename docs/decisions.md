# Environment decisions

### 2b. Environment decision (S3, Oct 1 2026)

- Machine: Windows 11, **RTX 4090 24 GB** (driver 610.60, CUDA 13.3 runtime available), data on `D:\projects\hready_data`.
- **Windows-native conda** for all project code: env `hready` (PyTorch, smplx, training, data, metrics). Reasons: Isaac Sim/Lab already run natively on Windows, all existing envs are Windows conda, and reading `D:` from WSL2 (`/mnt/d`) is slow.
- **Isaac Lab + Newton:** reuse the existing Isaac Lab installation (separate env); `hready` talks to it through files (retargeted trajectories in, metrics/videos out), not imports.
- **WSL2 only as fallback**, per baseline, if a public HMR method (GVHMR, WHAM, TRAM) does not install on Windows (Linux-only CUDA extensions). Log every such case in `docs/pivot_log.md`.
- DDP on Windows uses the `gloo` backend (no NCCL); multi-GPU NCCL runs happen on Kaggle (Linux). `torch.compile` is optional on Windows.

### Item 6C — G1 Newton smoke (Oct 2026)

- **Isaac Lab + Newton** run only in **WSL** (`hready/robot/isaaclab_newton.py`); Windows `hready` orchestrates via `run_smoke.py` (retarget → run `.npz` → metrics → MuJoCo replay mp4).
- **PhysX** dropped for item 6 (AGENTS rule 0: Newton is the single-engine path for this smoke).
- **Assist wrench / pelvis pin** not shipped: Newton applies body-frame wrenches (`f_body = R @ F_world` verified on a box); stable 6D pin of floating G1 not achieved (no gravity feedforward + rot pin diverged; COM pin with feedforward still **8.9 cm / 3.1°** on stand). Foot **contact force readback unavailable** on this Newton path.
- **Metrics:** `kin_root` (root pose/velocity written each step + joint PD to reference) and `free` (joint PD only, time-to-fall). Controller label: *PD-only joint tracking with default G1_29DOF_CFG gains; not a balance controller*.
- **Torque saturation (Oct 2026 close-out):** Smoke metrics use **per actuator group** (legs 12 / waist 3 / arms 14 GMR joints): **`pd_demand_exceeds_limit_{group}_step_fraction`** and **`pd_demand_ratio_{group}_step_max_p95`** from logged PD demand `K*(q*-q)-D*qdot` (Isaac `computed_torque` when archived). When `tau_applied_gmr` is present, also **`actuator_torque_at_limit_{group}_step_fraction`** and **`actuator_clamped_pd_demand_{group}_step_fraction`** from Isaac `applied_torque` vs `computed_torque`. **Do not use pooled any-of-29 demand as a headline:** use per-group `pd_demand_*`; re-tracked ACCAD jump waist **`actuator_clamped_pd_demand` ≈ 0.232**, legs **≈ 0.005**. Waist **max |q−q_ref| ≈ 0.79 rad** on that clip is real tracking error under kin_root (pelvis scripted, waist must catch up) but **not** deliverable actuator torque.
- **Foot slip:** 6C uses ankle world positions at **sim dt=5 ms** with contact rule `z <= min_run_z + 3 cm`. Pass4 scratch used **instantaneous** `body_lin_vel_w` when `z < 5 cm` — not comparable. Reference slip should FK the retargeted motion in MuJoCo (`g1_mocap_29dof.xml`) at the **same dt** and the same ankle-based rule.
- **Free mode vs pass3 jump (0.55 s vs 0.97 s):** Pass3 used **8 physics substeps**, wrote **initial root linear/angular velocity** from the registered motion, and continued sim after fall for logging. 6C `isaaclab_newton.py` uses **1 substep**, **zero initial root velocity**, and **stops** the rollout at fall — same fall detector (`z<0.4 m` or `|tilt|>45°`) but different contact transients and clip alignment (retargeted `motion.npz` vs scratch `jump.npz`).
