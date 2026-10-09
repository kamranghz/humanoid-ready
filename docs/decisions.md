# Environment decisions

### 2b. Environment decision (environment setup, Oct 1 2026)

- Machine: Windows 11, **RTX 4090 24 GB** (driver 610.60, CUDA 13.3 runtime available), data on `D:\projects\hready_data`.
- **Windows-native conda** for all project code: env `hready` (PyTorch, smplx, training, data, metrics). Reasons: Isaac Sim/Lab already run natively on Windows, all existing envs are Windows conda, and reading `D:` from WSL2 (`/mnt/d`) is slow.
- **Isaac Lab + Newton:** reuse the existing Isaac Lab installation (separate env); `hready` talks to it through files (retargeted trajectories in, metrics/videos out), not imports.
- **WSL2 only as fallback**, per baseline, if a public HMR method (GVHMR, WHAM, TRAM) does not install on Windows (Linux-only CUDA extensions). Log every such case in `docs/pivot_log.md`.
- DDP on Windows uses the `gloo` backend (no NCCL); multi-GPU NCCL runs happen on Kaggle (Linux). `torch.compile` is optional on Windows.

### HR-Refine item 7 / item 8 physics losses (Oct 2026)

- **Disabled by default in `configs/hr_refine.yaml` `physics_loss`:** `smoothness` and `flight_consistency`. On corrupted AMASS windows, jerk/flight penalties on joint trajectories reach **O(1e8)** and dominate the batch loss (NaN within a few Adam steps). Foot skating, ground penetration, balance, bone length, and joint ROM stay enabled for ablation item 8.
- **Refinement smoke regression (item 7 acceptance):** 300 train steps with `shuffle=True`, `corrupt_step=step`, dropout **0.1**, and only **50/200** val windows evaluated left val MPJPE **worse than corrupt** (~116 mm vs ~36 mm). Root cause: **under-trained** model plus **train/eval corruption mismatch** (not zero-init: residual heads verified at **max |Δtransl| < 1e-5** at init). Item 8 gate uses **reconstruction-only**, **dropout 0**, longer training, and fixed **200**-window eval.
- **`hr_refine_eval` units:** `mpjpe` / `pa_mpjpe` already return **mm**; eval must not multiply again (the item-8 gate CSV, unfinished work kept only in a local backup, was ~1000× inflated before that fix).

### Module audit (Oct 4, 2026)

| Area | Reuse | Conflicts / gaps |
|------|--------|------------------|
| AMASS index + `assign_split` | `hready/data/amass.py` — beta-group splits, `load_clip`, flags, foot-height-rise sidecar | Split key is **beta-connected folder group**, not raw subject string alone; `assert_no_subject_leakage` is group-level. |
| BABEL | `hready/data/babel.py` — `act_cat_matches_keyword`, cached `babel_index.json` | Many test clips are `seq_ann` only; floor-work rules prefer `frame_ann` segments ≥1 s. |
| Contact / skate / rise | `contact.py`, `foot_height_rise.py`, `refine_eligible_cache.py` | Rise >5 cm excluded from HR-Refine **eligible** train set; floor-work subset **annotates** rise, does not auto-drop. |
| Ego sensors (egocentric evidence) | `refine_corrupt.py`, virtual cam, `imu.py` | Corruption models **full-body** noise, not partial leg dropout; needs new encoder + observation mask. |
| Regression baseline (oracle completion baselines) | `hready/models/hr_refine.py` backbone | Must **not** take corrupted legs; new observation encoder on egocentric-evidence features only. |
| HR-Refine trainer | `hr_refine_engine.py`, `configs/hr_refine.yaml` | Item 8 / fast-loader work kept only in a local backup (motion memmap unfinished). |
| Losses / metrics | `hready/losses/*`, `hready/metrics/*` | Lower-body/feet MPJPE slices and contact **ECE** not wired in one eval CLI yet. |
| SMPL-X | `smplx_wrapper.load_body`, `batch_forward.smpl_forward_bt` | — |
| G1 / Isaac | `hready/robot/*` | **Out of scope** for the egocentric-track preview (robot module deferred). |
| Missing for splits, evidence and completion | — | floor-work CLI (this commit); `ego_observation_model.md` + synth; flat-floor heuristic + ego-conditioned regression + shared metrics table. |

### Item 6 — G1 Newton smoke test (Oct 2026)

- **Isaac Lab + Newton** run only in **WSL** (`hready/robot/isaaclab_newton.py`); Windows `hready` orchestrates via `run_smoke.py` (retarget → run `.npz` → metrics → MuJoCo replay mp4).
- **PhysX** dropped for item 6 (AGENTS rule 0: Newton is the single-engine path for this smoke).
- **Assist wrench / pelvis pin** not shipped: Newton applies body-frame wrenches (`f_body = R @ F_world` verified on a box); stable 6D pin of floating G1 not achieved (no gravity feedforward + rot pin diverged; COM pin with feedforward still **8.9 cm / 3.1°** on stand). Foot **contact force readback unavailable** on this Newton path.
- **Metrics:** `kin_root` (root pose/velocity written each step + joint PD to reference) and `free` (joint PD only, time-to-fall). Controller label: *PD-only joint tracking with default G1_29DOF_CFG gains; not a balance controller*.
- **Torque saturation (Oct 2026 close-out):** Smoke metrics use **per actuator group** (legs 12 / waist 3 / arms 14 GMR joints): **`pd_demand_exceeds_limit_{group}_step_fraction`** and **`pd_demand_ratio_{group}_step_max_p95`** from logged PD demand `K*(q*-q)-D*qdot` (Isaac `computed_torque` when archived). When `tau_applied_gmr` is present, also **`actuator_torque_at_limit_{group}_step_fraction`** and **`actuator_clamped_pd_demand_{group}_step_fraction`** from Isaac `applied_torque` vs `computed_torque`. **Do not use pooled any-of-29 demand as a headline:** use per-group `pd_demand_*`; re-tracked ACCAD jump waist **`actuator_clamped_pd_demand` ≈ 0.232**, legs **≈ 0.005**. Waist **max |q−q_ref| ≈ 0.79 rad** on that clip is real tracking error under kin_root (pelvis scripted, waist must catch up) but **not** deliverable actuator torque.
- **Foot slip:** the smoke test uses ankle world positions at **sim dt=5 ms** with contact rule `z <= min_run_z + 3 cm`. An earlier scratch run used **instantaneous** `body_lin_vel_w` when `z < 5 cm` — not comparable. Reference slip should FK the retargeted motion in MuJoCo (`g1_mocap_29dof.xml`) at the **same dt** and the same ankle-based rule.
- **Free mode vs the earlier scratch jump run (0.55 s vs 0.97 s):** that run used **8 physics substeps**, wrote **initial root linear/angular velocity** from the registered motion, and continued sim after fall for logging. The smoke-test `isaaclab_newton.py` uses **1 substep**, **zero initial root velocity**, and **stops** the rollout at fall — same fall detector (`z<0.4 m` or `|tilt|>45°`) but different contact transients and clip alignment (retargeted `motion.npz` vs scratch `jump.npz`).
