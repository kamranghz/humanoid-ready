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

### Naming rules (2026-10-09)

- Naming rules recorded in `AGENTS.md` §2a: descriptive names only (no step, attempt, version or track codes), one `docs/experiments.md` row per experiment, no version suffixes in file names, external cache/checkpoint folders named like the config and code that reference them, "pre-registered"/"recorded" wording, renames as separate string-only commits with proof and fingerprint check, and no employer-specific or application-process wording in the repository or its history.

### EgoBody — download exception, files taken, evaluation-only use (2026-10-09)

- **Dataset:** EgoBody (ETH Zurich, ECCV 2022), source `egobody.ethz.ch`: 125 sequences, 36 participants, HoloLens2 egocentric RGB with head/hand/gaze tracking, SMPL-X ground truth fitted from a multi-Kinect rig. Format findings: `docs/egobody_data_format.md`.
- **Licence terms as signed: NOT RECORDED YET (gap).** No licence copy exists under `D:\projects\hready_data\review\` (searched; the only EgoBody mentions there are planning notes, not terms). The owner must add the signed terms here (permitted use, redistribution, derived data, citation text). Until then the data is treated as evaluation-only, never redistributed, and nothing derived from it is committed.
- **Download exception:** `AGENTS.md` §1/§8 pause downloads until the real-egocentric-validation decision, and pending decision (2) there still reads "pending". The owner took the files below on 2026-10-09 (file times 15:45–15:53 that day); this entry records that exception for EgoBody only. `AGENTS.md` itself was not edited in this step; the owner updates the pending-decision text.
- **Files taken** (`D:\projects\hready_data\datasets\egobody\raw`, 75 GB on disk): `egocentric_color.zip` (79.2 GB, 199,111 JPEG frames plus per-sequence pose and 2D-keypoint files), `egocentric_gaze.zip` (0.47 GB), `scene_mesh.zip` (0.23 GB, 15 scenes), `calibrations.zip`, `transf_matrices_all_seqs.pkl` (84 MB), `data_info_release.csv`, `data_splits.csv`, `smplx_camera_wearer_{train,val,test}.zip` (0.23 / 0.06 / 0.15 GB). **Added 2026-10-11:** `smplx_interactee_{train,val,test}.zip` (0.23 / 0.06 / 0.15 GB, file times 2026-10-11 20:23–20:24), the SMPL-X ground truth of the person visible in the egocentric view; it is what the 2D keypoints in `keypoints.npz` belong to. Same exception, same evaluation-only rule.
- **Files deliberately skipped (not downloaded, not inspected):** `kinect_color`, `kinect_depth`, `human3d_*`, `Egohmr_*`, the non-X `smpl_*` fits, depth, and every interactee file other than the SMPL-X fits above. Reason: the plan needs the camera wearer's egocentric view, the wearer's body ground truth, and the visible person's body ground truth to relate the 2D keypoints to 3D.
- **Extraction:** only the small zips were extracted (the six above, plus the three interactee zips on 2026-10-11), into `datasets\egobody\extracted\` (about 5.2 GB; never inside the repo). `egocentric_color.zip` stays unextracted: its central directory was listed, and a few small members were read in memory (per-sequence `_pv.txt`, `keypoints.npz`, `valid_frame.npz`, one JPEG header, and the 143 MB `PV.tar` listing) without writing them to disk.
- **Use: evaluation and noise calibration only.** Never for training, never for threshold tuning on test.
- **Pre-registered protocol (recorded 2026-10-09, before any calibration or evaluation on this data):** noise and visibility calibration of the realistic observation model uses the official `train` and `val` splits only; final numbers are reported once on `test`; nothing is re-tuned after test is seen. The pre-registered cohort-tree thresholds in `configs/ego_splits.yaml` are not changed by this data.
- **What was already touched in this step (disclosure):** format and geometry checks on three sequences, one per split (the test one is `recording_20220318_S31_S32_01`; checks were transform chain, forward kinematics, pelvis and foot heights against the scene floor; no perception output, evidence statistic or threshold was computed from it). The geometry cohort tree was run on `val` only. Raw outputs are in `docs/egobody_data_format.md`. On 2026-10-11 the interactee files were verified and the identity of the 25 keypoint slots was measured on 4,958 frames sampled from all 125 recordings, test included; this used frame counts, camera projection and nearest-point identity only. No detector error was computed and nothing was tuned.
- **Finding that bears on the protocol (open, owner to decide):** the official splits are not participant-disjoint. Counting the camera wearer only, `S05` appears in train (9 sequences) and test (7), and `S03` in val (5) and test (3); counting every participant in any role, 10 are shared by train and val, 13 by train and test, 1 by val and test. Options: (a) keep the official splits and disclose the overlap next to every test number; (b) also report test restricted to sequences whose participants are absent from train and val. Neither is chosen here; the calibration-on-train/val, report-on-test rule above stands either way. *(Superseded 2026-10-11, see the next entry.)*

### EgoBody — evaluation protocol and subject-disjoint split (2026-10-11)

- Full protocol: `docs/egobody_evaluation_protocol.md` (documentation only; no code added). Two protocols: the camera wearer (primary, full-body completion against the wearer's SMPL-X) and the visible person (secondary, real 2D-keypoint noise against the other person's SMPL-X, never presented as a result on the wearer).
- **The official train/val/test split is not used for any EgoBody number.** It is not participant-disjoint (previous entry). Replacement: connected components of the participant graph (36 participants, every participant in any role linked per recording) assigned whole to a calibration set (61 recordings, 1.03 h, 18 participants) and a report set (64 recordings, 1.01 h, 18 participants), by a deterministic frame-balancing rule fixed before any method result existed. The graph has 11 components (largest 45 recordings), so nothing was dropped; zero participants are shared in any role. This replaces the "calibration on train/val, report on test" assignment of 2026-10-09; its principle (noise parameters from the calibration set only, final numbers once on the report set, no tuning on the report set, no training on EgoBody) stands.
- Evaluation-only exception to the single-training-body-model rule: EgoBody ground truth uses the gendered `locked_head` files with per-sequence median betas; ground truth is never converted to the neutral model (measured gap 25.7-45.6 mm mean over 22 joints). Comparison on the 22 body joints.
- Floor rules: ±5 cm sensitivity for every floor-dependent quantity; scenes with fewer than 10% of mesh vertices in the floor bin (`seminar_g110`, 4 recordings) are excluded from floor-dependent quantities; scenes with a mesh minimum more than 5 cm below the floor bin also get ±10 cm.
- Finding recorded: on the new split the real data contains no lie or crawl, essentially no kneel, and almost no sit_floor, so the floor-work slice cannot be evaluated on EgoBody; only sit_support versus the rest can.
- Correction to the data-format doc made the same day: the gendered-versus-neutral gap figures compared two joint definitions and were 1-3 mm off; the rechecked values are in the doc.
- Wording: the older term for the cohort-tree thresholds in the 2026-10-09 entry and in the data-format doc was replaced by "pre-registered" (`AGENTS.md` §2a rule 5). Other repository hits for the old term are Python's `frozenset` / `frozen=True` and the old-wording row in `docs/experiments.md`, which are left as they are.

### Item 8 ablation finding (2026-10-12)

- **Setup:** four arms of 2000 steps each on the AMASS HR-Refine train windows (11,260 of 11,664 eligible clips, the ones present in the `grounded_motion_30hz` cache), seed 0, one run per arm, warmup 20 steps, dropout 0.1, validation on 200 fixed val windows with fixed corruption seeds (corrupt-input baseline 29.12 mm). Table: `results/physics_loss_ablation/ablation.csv`.
- **Finding:** the full physics-loss arm (weight 0.01) is worse than reconstruction-only on MPJPE, 26.25 mm against 24.10 mm (+2.1 mm), and also on foot skating, penetration and jitter. The arm without the contact-weighted foot-skating term (`minus_contact`) matches reconstruction-only on MPJPE (24.12 mm) and has the best physical metrics (penetration 82.9 mm against 118.2 mm for reconstruction-only; skating 4.12 against 4.29).
- **Balance term:** the arm without the balance term (`minus_balance`, 26.08 mm) is within 0.2 mm of the full arm, so removing balance did not recover the MPJPE loss. The MPJPE cost is not attributable to the balance term in this run. The foot-skating term is the more likely source, which is a hypothesis to test, not a result of this run.
- **Open:** the weights of the physics terms (the skating term first) need tuning in future runs; no rerun is needed for this record, the finding is the result. One seed per arm and no confidence intervals, so the 2 mm gaps are not established as significant. The absolute physical-metric values (skating about 4.3, jitter about 9000) are implausibly large and their definitions and units were not checked in this run; only the differences between arms are used.
- **Deviations recorded:** the training and validation windows were read from the `grounded_motion_30hz` memmap cache instead of `load_clip` (values equal to 2e-7); dropout stayed at 0.1; `minus_contact` means the foot-skating term off; the 200-step overfit criterion was relaxed to under 20% of the step-0 loss within 1000 steps (measured 14.6%, warmup 20).

### HR-Refine auxiliary heads and IMU fusion ablation (2026-10-12)

- **What:** `HRRefine` gained optional heads (foot contact, BABEL action, root-translation intent) and an optional IMU input; all are off by default, so the refiner itself is unchanged. Three conditions of 2000 steps each (`video_only`, `imu_only`, `fused`), one seed, validation on fixed val windows. Results: `results/contact_torques/contact_f1.txt`, `results/auxiliary_heads/action_intent.txt`.
- **Findings (as measured):** contact F1 0.936 / 0.936 / 0.938 against 0.826 for always-contact; action accuracy 0.500 / 0.493 / 0.504 against 0.281 for always-walk and 0.033 uniform; intent ADE 0.133 / 0.067 / 0.130 m against 0.140 m for zero motion. Refined root ADE is 0.0559 m in every condition against 0.0607 m for the corrupted input, so IMU did not help the refinement, and `fused` did not reproduce the `imu_only` intent gain (cause not investigated).
- **Choices:** the IMU is a world-axis kinematic accelerometer from the cached clean joint positions (no gyroscope: the cache has no global joint rotations), plus 0.5 m/s^2 noise, so it is cleaner than the corrupted pose. The intent head sees only the first 49 frames and its last IMU frame is zeroed (the second difference there includes frame 49; a first run without this showed the same large gain, but the fix was kept as correct). Loss weights were not tuned. Training windows come from the 11,260 cached train clips, half of them from BABEL-labelled clips. The IMU ablation uses short real training instead of overfitting one batch, and the result folders follow `AGENTS.md`, not the retired `B2` / `C1` codes.

### Paper track: G1 tracking policy for the Newton / PhysX / MuJoCo comparison (2026-10-16)

**Status: research only.** Nothing was downloaded, installed or run for this entry. No candidate below is confirmed to run
in all three engines, so the policy chosen for now is the **PD-only fallback**, labeled as such. The paper-track row
"Tracking-policy selection" stays open (its done-when is "runs one AMASS clip in all three").

**(1) Choice.**

- **Controller used for the cross-engine study until a learned policy is verified: PD-only joint tracking of the
  retargeted reference** (default G1 29-DoF gains, `hready/robot/isaaclab_newton.py`), as in the item 6 smoke test.
  Label on every row: *PD-only joint tracking, not a balance controller; engine-independent by construction, not a
  learned policy.* No source URL: it is our own code.
- **Learned-policy candidate to verify next (not yet verified): a BeyondMimic-style G1 motion tracker**
  (paper: https://arxiv.org/pdf/2508.08241; it reports sim-to-sim evaluation and real G1 deployment, 25 LAFAN1
  references). I did not confirm the location or license of the official code in this search.
- **Retargeter feeding any policy: GMR** (paper https://arxiv.org/abs/2510.02252, code github.com/YanjieZe/GMR, MIT per
  its repository page). Its experiments used only the G1 and LAFAN1 data.

**(2) Why this choice, and the alternatives.**

- **Cross-engine policy transfer is the open risk, so a learned policy confounds the comparison.** PolySim
  (https://arxiv.org/pdf/2510.01708) reports a policy trained in one simulator can fail in another (Genesis-trained
  policy: 0.043 success in Isaac Gym; Isaac Gym-trained: 1.000 there). A learned policy trained in one engine would
  measure that engine mismatch together with the clip quality we want to measure. The fallback has no training engine.
- **Cost of the fallback:** it is not a balance controller, so on a free-base robot it falls (item 6 reports
  time-to-fall under joint PD only), and verdicts must use the `kin_root` mode or a held-base variant. Results are
  executability under tracking, not a statement about learned-controller feasibility.
- **Alternatives found (none verified in all three engines):**
  - mjlab-based G1 tracking (MuJoCo Warp backend, BeyondMimic tasks), e.g. Unitree's
    https://github.com/unitreerobotics/unitree_rl_mjlab (motion-imitation section; mainly velocity tracking
    otherwise). Trains and runs in MuJoCo Warp. The upstream mjlab URL was not among the pages I saw (only forks).
  - MimicLite, https://github.com/roboparty/mimiclite: released G1 tracker with ONNX inference and MuJoCo sim2sim;
    its comparison to another tracker is self-reported.
  - ProtoMotions GTP G1 tracker, https://huggingface.co/cagataydev/protomotions-gtp-unitree-g1: community ONNX model,
    IsaacLab or MuJoCo Warp, MuJoCo rollout example; the card calls real-hardware integration a work in progress.
  - TWIST (github.com/YanjieZe/TWIST) and TWIST2: teleoperation-oriented stacks; not chosen, as the repository's purpose
    is real-time teleoperation, not clip-level evaluation.
  - PolySim: trains across simulators, which is closest to the goal, but I could not confirm that code or
    checkpoints are released.
- **Engine support found:** Newton is integrated in Isaac Lab 3.0 beta, focused on the MuJoCo-Warp solver; its docs say
  Newton was validated against PhysX by transferring locomotion policies in both directions on the G1. That is a
  locomotion policy, not a motion-tracking one. I found no published G1 motion-tracking policy validated on Newton.

**(3) Confirmation that it runs in all three engines: NOT confirmed.** What exists in this repository:

| engine | PD-only G1 tracking |
|---|---|
| Newton (Isaac Lab, WSL) | run in item 6 (`results/robot_smoke_test/`) |
| PhysX (Isaac Lab) | not run; dropped for item 6 (rule 0) |
| MuJoCo | used only to replay the retargeted motion to video (`replay.py`); no dynamics run |

For a learned policy, the verification still to do is one AMASS clip in all three engines with the same checkpoint and
configuration, recorded here; until then the fallback label above applies to every cross-engine row. The BeyondMimic
checkpoint would have to be trained or obtained (a download exception under `AGENTS.md` §1), which this entry does not do.

**Open items.** (a) Locate the official BeyondMimic code and checkpoint license. (b) Decide whether a trained checkpoint
is obtained or trained here. (c) Run the three-engine check. Sources: arXiv 2508.08241, 2510.02252, 2510.01708;
Isaac Lab Newton backend docs (isaac-sim.github.io/IsaacLab, v3.0.0-beta2).

### Nymeria: files on disk, licence, audit results (2026-11-19)

Full format notes: `docs/nymeria_data_format.md`. Inventory tool: `scripts/nymeria_inventory.py` (read-only; run 2026-11-19).

- **Files on disk** (`D:\projects\hready_data\datasets\nymeria`): 206 sequence folders, 499.9 GB of zips, file times 2026-10-12. This
  download was **not recorded** in this file when it was made; this entry records it now.
- **Licence as stated in all 206 `metadata.json`:** CC BY-NC 4.0 (non-commercial, attribution). The licence text and the dataset's own
  terms were not re-read here; the owner confirms them before any redistribution or derived release.
- **What each sequence has:** all 206 have `body_motion.zip`, `recording_head.zip`, `recording_lwrist.zip`, `recording_rwrist.zip` and
  `metadata.json`. 37 lack `narration_atomic_action.csv` and 31 of those also lack `narration_activity_summarization.csv`, so 169
  sequences have all seven files.
- **The egocentric video is not in this download.** `recording_head.zip` holds `motion.vrs`, `et.vrs` and the SLAM and eye-gaze outputs,
  but no `data.vrs`. The dataset's own definitions list `recording_head/data/data.vrs` as a separate download group
  (`recording_head_data_data_vrs`); that group was not taken. Whether `data.vrs` carries the RGB stream is inferred from the
  dataset's viewer, which shows RGB from the participant's glasses, not stated in the file definitions. Consequence: today Nymeria
  gives real head trajectories (SLAM), body motion and narrations, but no frames to run a detector on.
- **Body motion:** `body_motion.zip` holds `body/xdata.npz` and `body/xdata_blueman.glb` (3 members; no `.mvnx`). It is the Xsens
  export (23 segments, 22 joint angles, `foot_contacts`, N = 197,843 and 233,661 frames in the two sequences read), not an SMPL-X fit;
  array names and shapes are in `docs/nymeria_data_format.md`.
- **Size of the set:** 206 sequences, 53.92 h of action time, 45 locations, 29 sequences with two participants, 16 scripted scenarios.
- **Floor work in the narrations (169 sequences with atomic-action text; sentences with the wearer as subject, word patterns, not
  verified frame by frame; sentence counts, not durations):** kneeling 889, squatting or crouching 782, sitting on the ground 379,
  lying 191 (166 name a bed, sofa or chair, 3 name the floor, 22 name no surface), crawling 11, push-up or plank 20. The floor-heavy
  sequences are concentrated: 4 of the 5 "Game night" sequences (kneeling 16 to 205 sentences each, sitting on the ground up to 163) and
  3 "Body stretch" sequences (kneeling, sitting on the ground, 20 push-up or plank sentences in one). Kneel and sit_floor therefore
  look evaluable on real data if video is added; lying on the floor is not supported by these counts.
- **Head camera (read later the same day):** `camera-rgb` is a fisheye (`FisheyeRadTanThinPrism`), focal 1218.0 px, principal point
  (1455.8, 1451.6); under stated assumptions (radial terms only, 2880 x 2880 calibration frame) about 115 degrees horizontal. Details and
  caveats in `docs/nymeria_data_format.md`, section 6.
- **Not known yet:** `frameRate` and `frameCount` values, the clock of the video frame list, and the alignment between video and motion.
- **Intended use (proposal, not decided):** a second real egocentric dataset next to EgoBody (evaluation-only, unchanged). Whether
  Nymeria is also used for training or adaptation, and under which subject-disjoint split, is an owner decision. Adding the video
  group is a download exception under `AGENTS.md` section 1 and needs the owner's decision and a size check on one sequence first.

### Nymeria: preview video download, camera profile and calibration hold-out (2026-11-19, later)

- **Download exception (owner decision, in chat).** The owner approved adding the preview RGB video group (`video_main_rgb`) for
  the 7 floor-heavy sequences listed in `docs/nymeria_data_format.md`: `20230706_s0_breanna_reese_act0_cfg468`,
  `20230815_s0_samantha_lester_act0_513kae`, `20230628_s1_hayley_little_act0_z038r0`, `20230731_s0_tammy_campos_act0_d3blev`,
  `20230720_s0_robert_ballard_act0_7n4853`, `20230707_s1_elizabeth_tucker_act0_df444f`, `20230816_s1_jeffery_bryant_act0_p5w199`.
  Size: 0.82 to 1.42 GB each, 7.5 GB in total (1.42, 0.96, 0.85, 0.82, 1.14, 1.20, 1.13), against about 519 GB free on the drive afterwards.
  All seven were downloaded on 2026-11-19 and every SHA-1 matched the list. The raw `recording_head_data_data_vrs` group (20.75 GB for the first sequence) was **not** taken; it would be
  a further decision, with the `projectaria_tools` licence and a separate environment checked first.
- **Why this group first.** It is read with ordinary video tools, needs no new package in the pinned `hready` environment, and one
  frame was checked by eye (fisheye, upright, wearer's hands and thighs visible).
- **Links.** The first URL list (file time 2026-10-11) had expired (HTTP 403, 14-day validity); a new list was requested from the
  dataset page by the owner on 2026-11-19. The list is a credential-like file and stays outside the repository.
- **Camera profile (owner accepted the recommendation).** `hl2_like` is the primary synthetic profile, because the real
  validation set (EgoBody) is that device class; a smaller `aria_like` subset follows the Nymeria camera (fisheye, about 115 degrees
  horizontal, parameters in `docs/nymeria_data_format.md`); `ego_wide` (129.8 degrees) stays only so the oracle results remain comparable.
  A fisheye render needs a renderer that accepts a custom projection; if the renderer probe shows it does not, `aria_like` becomes a
  rectified pinhole view and the document says so.
- **Noise calibration hold-out (owner accepted the recommendation).** The 19 tune subjects of `results/splits/splits.json` (fraction
  0.05, seed 0). The perception model trains on train minus tune, model selection uses VAL, TEST is used once; real-image noise is
  calibrated on the EgoBody calibration set only. A leak check must show no tune subject in perception training.
- **Stated limit.** One frame of one sequence has been looked at. No alignment between video and motion exists yet, and no detector
  has been run on Nymeria video.
- **Alignment result (same day).** The Xsens `timestamps_us` are not on the SLAM clock (no overlap in 5 of 7 sequences); the video frame list and
  the narration are on the SLAM clock (every one of 1,400 narration rows lies inside its video's time range). Per-sequence offsets were estimated by
  matching head rotation speed (`scripts/nymeria_motion_sync.py`, coarse stage): 6 of 7 sequences give a peak (0.98 to 0.995 against at most
  0.28 elsewhere, except one at 0.63 that the refined run later puts at 0.986), 1 none (0.07, unresolved). Details in `docs/nymeria_data_format.md`, section 7.2. Stated limit: this is a
  timing check on head motion; no joint has been projected into an image yet, so no evaluation-grade ground truth is claimed.

### Nymeria: one uniform alignment run, inclusion rule and next experiment (2026-11-20)

- **Why a new run.** The earlier Nymeria numbers came from several runs of one script (different code versions, keys from different runs in one
  report, overlays from two generations). The owner had the old projection results deleted and asked for one consistent set. The script now stores a run id and a
  code hash in every entry, `--fresh` starts from an empty report, and `--audit` prints one line per sequence and UNIFORM or the list of problems.
  The run is id 20261120T170201Z; numbers are in `docs/nymeria_data_format.md`, section 7.3.
- **Inclusion rule (in the script).** Timing claims need a refined head-to-Xsens correlation of at least 0.95 (6 of 7 sequences). Spatial claims also need a
  shared head offset (5 of 7: breanna, samantha, hayley, tammy, elizabeth). Excluded: jeffery (no peak), and robert from spatial claims (3 good windows of 7).
- **Corrections to earlier statements.** The global fit rms is 5.7 to 35.9 cm (earlier text said 9.5 to 36); drift residuals reach +-12.5 ms (earlier +-12);
  the head offset d is not shared across persons (x 2 to 14 cm); elizabeth's alignment is not weak (refined 0.986), only her coarse value was low.
- **Hand accuracy, measured with two independent labellers.** The window rms is head-position consistency, not hand accuracy. Hand points were clicked on the wearer's own glove by two people (click distance median 20.5 px of 800, 90th percentile 35.5 px). Headline = consensus of the two: the raw Xsens hand lands a median 58 px (about 10.1 cm) from the glove centre, 90th percentile about 18 cm; after a leave-one-sequence-out constant offset 47 px (about 7.6 cm). Either labeller alone gives the same within 4 px. The projection convention is the best of 32 tested. The first labelling pass (315 px) is withdrawn: some points were on another person. Details and limits in `docs/nymeria_data_format.md` section 7.3.
- **Real-data experiment chosen (owner agreed on the priority).** Standing and foot slip from a noisy head pose on Nymeria, with the recorded protocol
  `docs/nymeria_foot_lock_protocol.md` (baselines B0, B1, B2, M1; success rule fixed before any run; cohort rule). Run once on 60 sequences, results in section 11 of that file.
  Body motion and head trajectory are already on disk for all 206 sequences, so no download is needed; the owner's approval of a download was not used.
- **Older baselines.** The oracle completion baselines stay as controls for the simulated task only (`docs/oracle_completion_baselines.md` says so); they make no real-data claim.
- **Standing and foot slip from the head pose alone, measured (first run, 60 Nymeria sequences).** Rule 2 not met: M1 beat the hysteresis baseline on 58 of 60 sequences but by a median 2.6 points of missed-lock against the recorded margin of 5. At no more than 5% false-lock, 62% to 65% of both-feet-planted frames are locked from the head pose alone (72% to 76% with 0.5 s of lookahead); in the hard case (planted while the head moves) 68% are missed by every detector. A single speed threshold is as good as hysteresis. Under strong simulated noise the learned detector degrades more slowly. Allowed claim: head-pose speed alone locks most but not all planted frames, and fails mainly when the head moves while the feet are planted; not a benchmark. Details, controls and limits in `docs/nymeria_foot_lock_protocol.md` section 11.

### Nymeria: the repo's trained model on the head pose, second protocol (2026-11-21)

- **Question and protocol.** Does the repo's trained completion model (`checkpoints/deterministic_main/best.pt`, step 140000) tell when both feet are planted better
  than a speed threshold, given only the head pose of the glasses (all body joints marked not seen)? Recorded in `docs/nymeria_foot_lock_repo_model_protocol.md`
  (rule in section 3, amendments A1 and A2, result in section 9). Same cohort, labels, folds and rule as the first protocol. Run once, 58 of the 60 sequences
  (2 had no floor estimate).
- **A mistake found before anything was scored.** The first call of the preparation step stopped on the axis check. A read-only diagnostic (no model, no
  labels) showed that the device z axis of the Aria glasses is not the direction the wearer faces: the device frame is the left SLAM camera frame, and the
  facing direction is about 40 degrees to the right of device z (walking directions in all 58 sequences: median 39.6 degrees, spread 7.8; the RGB camera's
  own calibration: 38 degrees; the baseline between the two SLAM cameras: 0.3 degrees from that). The earlier assumption "z is forward" came from two sequences seen by hand.
  Amendment A2 (written before any model output) takes the camera from each recording's calibration. Loosening the thresholds would have fed the model a camera rotated by 40 degrees.
- **Result: rule not met, signal check failed.** At no more than 5% false-lock (training folds), the repo's model misses 86% of the both-feet-planted frames
  (median over held-out sequences), the speed threshold B2 38%, the logistic regression M1 36%. The model's balanced accuracy is 0.56 (B2 0.80); it reaches 0.60
  on 28% of the sequences (rule: 80%). It is better than B2 on 2 of 58 sequences (rule: more than 60%); median difference -41.5 points
  (interval -49.6 to -28.6). The lookahead row is the same (80% against 28%). Hard case: 87% missed against 68%.
- **Controls.** No floor shift of 3, 5 or 10 cm in either direction helps (82% to 89% missed); the Xsens head height instead of the SLAM height gives the same
  (86.7%), so the SLAM pose is not the cause; a constant head height of 1.6 m is better (79%), so the model does not use the height; clock shifts are flat;
  a 60 s label shift gives chance (0.497). The home-domain check on AMASS was not run.
- **Allowed claim.** On 58 Nymeria sequences, with the floor height given, the repo's trained model fed only the head pose does not beat a speed threshold and is far worse.
- **Not allowed.** That head pose has no contact information: a speed threshold reaches balanced accuracy 0.80 on the same frames. The failure is this model on this input, and
  the run does not say why (input outside the training distribution; contact output trained on body evidence; camera conventions: the RGB camera looks 23 degrees
  below the horizon on average and the head position is the left SLAM camera; none tested).
- **Consequence for the project.** The completion model as trained is not a head-pose-only foot-contact detector. Anything claimed about contact from head pose on real
  data rests on the first protocol (speed threshold and logistic regression), not on this model. The model has not been tested with the evidence it was built for
  (visible joints) on real data.
- **Open (owner's choice, each needs its own protocol):** the home-domain check (same model, head-only input, AMASS windows) to separate "outside the training
  distribution" from the rest; a run with hand evidence after a valid hand-projection check; a new detector for the hard case on fresh sequences (137 of 206 untried);
  the perception stage E2-B. Nymeria stays test-only.


### Nymeria: the 137 untried sequences may be used for training (2026-11-21)

- **Decision (owner, 2026-11-21).** The 137 sequences that no experiment has touched (206 on disk, minus the 69 that went through the data check) may be used for
  training and adaptation of a learned detector. This replaces "Nymeria stays test-only" (entries of 2026-11-20 and 2026-11-21) **for these 137 only**. The 60
  sequences of the foot-lock cohort stay test-only and are not used for fitting, tuning or selecting anything.
- **Reason (from the files, `docs/dataset_sheets.md`).** Nymeria is the only dataset on disk with head pose, wrist devices, eye gaze and full-body ground truth
  together on real data. The simple detectors reach balanced accuracy 0.80 from head pose, so the signal holds contact information; the repo's completion model,
  trained on simulated evidence, did not transfer. The direction is to learn from the raw signals (no hand-made features), not reinforcement learning.
- **Facts checked before this entry (read-only, from the data-check report and the metadata files).**
  - 206 sequences: 60 in the test cohort, 9 attempted but failed the offset step, 137 never attempted.
  - 3 of the 137 have the same gender, height, weight and age group as a test sequence. The files do not say whether they are the same person, so these 3 are **not**
    used for training. 5 combinations repeat inside the 137: sequences with the same combination stay on the same side of any fold.
  - The 9 that failed the offset step stay out of both sets unless their offset step later passes the same rule as the cohort.
  - The 137 have not been through the offset step. Any of them that fails it is left out, as in the cohort; the number used is reported.
  - 28 of the 30 locations of the test sequences also occur in the 137, and the scripted activities are mostly the same. A split by sequence therefore does **not**
    show how a detector behaves in a new place; results are reported as "new sequences, mostly known places".
  - 9 of the 137 have floor-heavy scripts (stretching, dance, game night, workout); the test cohort has 3 (game night only; no stretching, dance or workout).
- **Not yet done.** The wrist devices' world frame and clock have not been checked against the head device and the suit; no wrist signal is used as an input until
  that check passes (reported per sequence). The protocol of the first learned detector (input only the head pose, raw windows, same labels and rule as the first
  protocol) is not written; it is written and read by the owner before any training.

### Nymeria: wrist devices checked against the head device and the suit (2026-11-21)

- **What was checked (read-only, owner ran it on the 60 test-cohort sequences).** Whether the left and right wrist devices share the head device's world frame and
  clock. Method: the distance between head and wrist from the devices' own SLAM logs (aligned by UTC time) against the distance between the suit's head and hand
  segments at the known head-to-suit offset, with a lag search of -4 to +4 s. Distance does not depend on the axes, so it tests a shared frame and a shared clock
  together. Script: `hready_data/wip/wrist_frame_test.py` (outside the repo); per-sequence output: `hready_data/wip/wrist_frame_test.jsonl`.
- **Result.** Median head-to-wrist distance 0.58 m (left) and 0.53 m (right). Correlation at lag 0: median 0.60 for both wrists. At the best lag: median 0.66 (left) and
  0.67 (right); 48% (left) and 38% (right) of the 60 sequences reach 0.7. About 15 sequences reach 0.8 to 0.89; several are near zero or negative, and in 3 wrist records (2 sequences) the best lag sits at
  the edge of the search (+4 s), meaning no match. The best lag has a median near 0 but spreads from about -1 s (10th percentile) to +1.7 s (90th percentile), and the left
  and right wrist of one sequence usually have different lags. The head-to-suit correlation for the head device itself is 0.98 to 0.99.
- **Reading.** The wrist devices are **not** shown to share the head device's clock and frame. The test cannot say which of the two fails: a clock off by more than the
  search range, an unstable clock, or a different world frame would each give this picture. A separate clock fit per wrist device from its own rotation speed against the suit
  hand (the method of `nymeria_project_check.py`, which gave correlations of 0.94 to 0.98 for 6 of 7 sequences) followed by the same distance test would separate clock from frame.
  That has not been run.
- **Consequence.** No wrist signal is used as an input of any learned detector until that two-part check passes. Wrist-based claims on real data are not made. Without the
  suit, a method to align the wrist clocks from the devices themselves would also be needed outside Nymeria (the UTC stamps of the devices differed by 0.3 to 1.6 s in the
  earlier clock check); that is stated as a limit. Step 1 of the new order (head pose only) does not depend on the wrist devices.
- **Written next:** `docs/nymeria_foot_lock_learned_protocol.md` (approved by the owner 2026-11-21; nothing run yet).

### Nymeria: data check of the training pool (2026-11-21)

- **What was run (owner, Windows machine).** The repo's data check, unchanged, on the 134 sequences of the training pool of `docs/nymeria_foot_lock_learned_protocol.md`
  (approved by the owner the same day). Report: `review/nymeria_learned_training_data_check.json`.
- **Result.** 118 of 134 passed the offset step; 16 failed and are left out. One of the 118 failed the height-order id gate and is left out. **117 sequences are labelled and
  usable for training.** 1831 min labelled, 1239 min both feet planted, 604 min hard stratum, 841 min both planted while upright. Size rule met; not indicative.
- **Locations.** 41 locations in the usable pool; 3 of the 30 test locations are not in it.
- **New-place test (protocol section 3A).** Held-out locations fixed by hash order before any model: 9 locations, 15 test sequences; 24 of the 117 training sequences are removed for
  that test, 93 remain. The interval of that test will be wide.
- **Not run:** any model. Next: the training and evaluation script, then an independent code review that checks no test sequence reaches training, validation, early
  stopping or an operating point, before any run.

### Nymeria: training script of the learned detector, independent review (2026-11-21)

- **Written.** `scripts/nymeria_foot_lock_learned.py` (plan, prepare, train, choose, test, controls) and amendment A3 of `docs/nymeria_foot_lock_learned_protocol.md`. Labels, offsets,
  strata, metrics, operating-point search and the first-protocol baselines are imported unchanged from the first protocol's code.
- **Checks on made-up data.** Window building against an independent numpy version; the causal output does not change when later frames are changed, the lookahead output
  sees at most 15 frames after the labelled frame; training reaches a validation loss near zero on an easy synthetic task; the whole chain (both sizes, three seeds, newplace,
  controls) runs without error. On real data, owner's machine (RTX 4090): plan counts match protocol 8.0 and 8.0a; the caches of all 177 sequences (117 training, 60 test)
  were built; rotations are valid and gravity in the world frame is (0, 0, -9.81), so z is up; a two-epoch smoke training of both sizes ran (smoke models only, not results).
- **Change to the label-shuffle control (A3 item 10).** On made-up data, a network that learned nothing still ranked frames by head speed with a tiny weight of random sign, so
  its balanced accuracy on the true labels was 0.99 or 0.02. "About 0.5" is therefore not a reliable leakage test. Now three shifts (60, 90, 120 s) and the verdict uses the
  validation-loss gain over the base rate (leakage suspected only if at least 0.02 for all three).
- **Independent review** (separate agent, did not write the code, traced test isolation, causality, label alignment, metrics and the rule against the protocol, with its own checks).
  No blocker. Three major findings, all fixed before any model: the smoke and limit options could overwrite the one-shot test result (now refused for choose, test, controls);
  false-lock of the baselines was not printed next to the rule (now printed and stored); a rerun of the first protocol that differs from the recorded numbers only warned (now stops).
  Minor fixes: metadata keys and pool/test group overlap are checked in the plan; runs that did not train are listed; the newplace report is marked indicative.
  Recorded as properties, not defects: the size S network's causal output reaches 30 frames back (about 1 s), M the whole window; inputs are not normalised (protocol); the
  reference B2 and M1 of the first protocol have thresholds picked in-sample on the test folds, so B2-137 and M1-137 are the like-for-like comparison.
- **Not run:** any real training, the size choice, the test.


### Nymeria: learned head-pose foot-lock detector, test results and controls (2026-11-21)

- **Main test (60 test sequences, size S chosen on validation: mean loss 0.2475 against 0.2572 for M, three seeds each).** Causal row at the "fl5" point: missed-lock 0.131
  (B2 0.375, M1 0.355, B2-137 0.344, M1-137 0.363), hard-stratum missed 0.330 (baselines 0.58 to 0.70), balanced accuracy 0.878. Lower than B2 on 60 of 60 sequences, median
  difference 24.3 points, 95% interval 22.6 to 27.0: the rule (section 3 item 2) and the beyond-data comparison (item 3, against M1-137: 23.4 points, 19.4 to 26.1) are met. The
  first-protocol rerun is identical to the recorded numbers. Allowed claim: on 60 sequences, from head pose only, a network trained on 95 other sequences beats a speed threshold by
  at least 5 points of missed-lock at 5% false-lock (new sequences, mostly known places).
- **Caveat recorded with it.** False-lock at "fl5" is 0.081 on the test sequences, above the 5% it was set to on validation; B2-137 is 0.093, M1-137 0.039. Part of the lead over
  M1-137 may come from the higher false-lock. About one third of the hard stratum is still missed.
- **New-place test (15 sequences, 9 held-out locations, trained on 75).** Indicative. Missed-lock 0.143 against 0.36 to 0.38 for the baselines; 15 of 15 sequences lower than B2,
  24.4 points (17.4 to 28.5): rule met on the held-out locations. False-lock 0.110.
- **Controls.** C2: more data still helps (missed-lock 0.174, 0.139, 0.131 at 25%, 50%, 100%). C4: lookahead optimum at 0 ms; labels aligned to about 50 ms. C5 (56 sequences) and C6
  (52 in known locations, 8 in others): rule met in each, no location effect.
- **C3, noise: negative.** At 1 cm and 0.5 degrees the network is at chance (missed-lock 0.97); B2 and M1 degrade slowly (0.385 and 0.386). The result holds for clean head poses
  (Aria closed-loop SLAM), not for noisy ones such as poses estimated from video. The likely cause (the input is the frame step times 30) is untested. This limits the claim for
  the real-video goal and must appear in any paper.
- **C1, label shift: incomplete, leakage neither found nor excluded.** The guard stopped the 60 s and 90 s runs: sequences are 11 to 21 minutes, and the guard judged a 60 s shift
  as "no shift" because it compared the shift with 10% of the length. That is a mistake of mine in A3 item 10. The 120 s run trained and kept 8% of the main model's validation gain
  (0.033 of 0.41), probably slow context. A4 (proposed, not applied): guard in seconds, verdict at half of the main gain. Owner has not yet decided.
- **Not done.** The 60 s and 90 s runs; a model trained with noise (needs its own protocol and counts as a second look at the 60 test sequences); `docs/dataset_sheets.md` and the
  tracker (E5b) not updated.

### Nymeria: network trained with noise, protocol approved and code added (2026-11-21)

- **Why.** The clean-trained network is at chance with 1 cm of per-frame noise (previous entry, C3), and head poses estimated from video carry that much noise. The owner asked to train
  the network so that it works under noise and to compare it with the clean-trained one.
- **Protocol.** `docs/nymeria_foot_lock_noise_protocol.md`, approved the same day. One change against the main run: independent per-frame noise added to the training input (a fresh copy
  every epoch; 20% clean, otherwise position sigma log-uniform from 0.3 to 8 cm, yaw sigma 0.45 degrees per cm); size S, three seeds, same settings. Rule, recorded before any run: at 1 cm and at
  2 cm the missed-lock at "fl5" is lower than B2's at the same noise (more than 60% of the sequences, median at least 5 points, interval excludes 0), and on clean poses at most 3 points above the
  clean-trained network (0.131). Second use of the 60 test sequences, stated as such; correlated noise is a descriptive control.
- **Code.** `scripts/nymeria_foot_lock_learned.py` gained `--step train --variant noise` and `--step test --scope noise` (written once to `review/nymeria_learned_noise_result.json`). Nothing existing
  was changed in behaviour: training with the default settings gives bit-identical weights and loss to the version that produced the results (checked on made-up data), and the independent-noise
  input equals the old one. Checks on made-up data: the autoregression against a plain loop (difference 3e-15; standard deviation 0.99; autocorrelation at one second 0.365 against 0.368),
  yaw noise leaves the gravity input unchanged, the validation noisy copy is identical when regenerated, no test sequence is loaded while training, the whole chain (train, noise test, refusal of a
  second run) runs. The results of the previous entries come from the earlier version of the script (code hash 4c8bf9a575bb98e6), archived as
  `review/nymeria_foot_lock_learned_4c8bf9a575bb98e6.py`. No independent review of the new code has been done (the owner decides).
- **Not run:** training of the noise models and the noise test.

### Nymeria: network trained with noise, test result (2026-11-21)

- **What was run.** Protocol `docs/nymeria_foot_lock_noise_protocol.md` as approved, no change to settings: three seeds of size S trained with noise (best epochs 10, 6, 9; none "possibly not
  converged"), one test run on the 60 test sequences (second use of them, stated as such), written once to `review/nymeria_learned_noise_result.json`. Full tables in section 8 of the protocol.
- **Result (causal, "fl5", median missed-lock, independent per-frame noise).** Noise-trained / clean-trained / B2: 0 cm 0.237 / 0.131 / 0.375; 1 cm 0.284 / 0.971 / 0.385; 2 cm 0.353 / 0.980 /
  0.463; 5 cm 0.366 / 0.976 / 0.739.
- **Rule.** Item 1 met at 1 cm (lower on 93.3% of the sequences, median 10.0 points, interval [8.3, 12.0]) and at 2 cm (85.0%, 11.6 points, [9.4, 14.3]). Item 2 not met: clean-level missed-lock
  0.237 against the limit 0.161 (clean-trained 0.131); hard stratum 0.542 against 0.330.
- **Allowed claim.** Trained with simulated independent per-frame noise, the network keeps a lead of at least 5 points of missed-lock over a speed threshold at 1 and 2 cm of noise (60
  sequences, second use); robustness cost accuracy on clean poses. Neither network is better everywhere: the clean-trained one on exact poses, the noise-trained one under any noise.
- **Limits.** Correlated noise (descriptive): lead over B2 11.8 points at 1 cm, 4.5 points at 2 cm (below the 5-point margin), and worse than B2 and M1 at 5 cm (missed-lock 0.839 against 0.555
  and 0.506). The noise is simulated; the error of a pose estimated from video is not measured, so nothing is shown about video. No setting was changed after the result.
- **Next (each needs its own protocol and the owner's approval):** measure the error of an actual pose estimator on these sequences (E2-B) and compare with the levels used here; a different
  input (longer-span step, smoothed positions); a larger network; a different noise model; the open C1 decision (A4 and the two missing models). `docs/dataset_sheets.md`, `docs/experiments.md`
  and the tracker (E5b) not yet updated.

### Nymeria: error of a head pose estimated from video (2026-11-21)

- **What was run.** Protocol `docs/nymeria_pose_noise_protocol.md` with amendments A1 and A2: VGGT-Omega (1B 512, as released) on 40 windows of 64 frames in each of the 7 sequences that have preview
  video (280 windows), compared with the SLAM head trajectory after a per-window similarity fit. No foot-lock label read. Written once to `review/nymeria_pose_noise.json`. MonST3R dropped (A2): 2 h 50 min per
  window, 8 days for the plan, and a larger error on the one window run.
- **Result.** Equivalent independent sigma of the step error 0.13 cm (A, 145 moving windows; 0.15 cm outside the test cohort) and 0.21 cm (B, one scale per sequence); rotation error 0.1 to 0.3 degrees; position
  error 1.2 cm (A) and 5.7 cm (B). Lag-1 autocorrelation 0.97 (simulated independent -0.01, correlated 0.93). Scale spread within a sequence 24 to 52%. Video and SLAM time alignment: 90% of windows at shift 0, no systematic shift.
- **Reading by the pre-registered rule.** Sigma below 2 cm, but the error is slowly varying and the scale is not stable, so: the simulation of independent noise does not describe this error, the next step is a
  different input and a scale source, not more noise training.
- **Allowed claim.** For VGGT-Omega on 7 preview sequences, the frame-to-frame jitter of the head position is about 0.1 cm, well below the simulated 1 to 5 cm; the remaining error is smooth and the scale changes by
  24 to 52% between windows. Nothing about other estimators; nothing about the detector on these poses.
- **Limits.** One estimator; the fit uses SLAM (scale per window, rotation, offset), so real error is at least this large; 64-frame windows; SLAM error not measured; compressed preview video; MonST3R not run beyond one window.
- **Next (each needs its own protocol and the owner's approval):** a scale source (wearer height, gravity plus a known height, IMU) and a test of the detector on VGGT-Omega poses with that scale; a detector
  input that is robust to smooth error and scale change; the open C1 decision. `docs/dataset_sheets.md`, `docs/experiments.md` and the tracker (E5b) not yet updated.

### Nymeria: supplementary MonST3R check (2026-11-23)

- **What was run.** Amendment A3 of `docs/nymeria_pose_noise_protocol.md`: MonST3R and VGGT-Omega on 7 moving windows, one per sequence (`review/nymeria_pose_noise_supplement.json`, written once). MonST3R took 48 min to 10 h 44 min per window.
- **Result.** Equivalent independent sigma 0.09 cm (VGGT-Omega) and 0.19 cm (MonST3R); lag-1 autocorrelation 0.86 and 0.97; rotation error 0.1 to 0.6 degrees (VGGT-Omega) and 0.1 to 3.0 degrees (MonST3R). MonST3R has the smaller sigma in 3 of 7 windows, the smaller rotation error in 0 of 7, the larger position error in 7 of 7.
- **Reading by the rule fixed before the run.** Both estimators: sigma below 2 cm and slowly varying, no disagreement. The claim in the entry above may say "VGGT-Omega and MonST3R, on 7 windows" instead of VGGT-Omega only.
- **Not claimed.** Which estimator is better; anything about scale (one window per sequence; no scale reported); anything beyond 7 moving windows.

### Egocentric render set built (2026-11-23)

- **Decision (owner).** The primary perception view is `aria_rect`: a rectified pinhole stand-in for the Aria camera (512x512, 110 degrees horizontal); `hl2_like` stays as the narrow view. Reason: on 25,320 AMASS frames the body (any joint, z-buffer rule) is visible in 6% of frames with `hl2_like`, 36% with `aria_rect`, 57% with `ego_wide` (`results/E/perception/render_visibility.json`). The real Aria camera is a fisheye; the fisheye projection is not implemented.
- **What was built.** Own torch rasteriser (nothing installed; flat colour per body region, z-buffer, floor plane), `hready/data/ego_render.py`; writer `scripts/ego_render_set.py`. Train every 20th frame, val and test every 30th (start offset fixed by clip path and seed 0): 247,027 / 31,013 / 16,494 frames from 13,856 / 2,222 / 1,271 clips, written outside the repo (`ego_render/aria_rect`, manifest with the clip list). Labels per frame: 22 joint pixel positions, depth, in-image flag, z-buffer visibility.
- **Subject ids in more than one split (7).** Four are name collisions between different datasets (CMU/KIT 11, 12, 63; GRAB/TotalCapture `s1`), not the same person. Three are subject ids that DFaust and MoSh both carry (50021, 50027: DFaust train, MoSh val; 50022: MoSh train, DFaust test). Whether these are the same people is not confirmed: the first ten shape parameters of the same id in the two datasets are not closer than those of different ids. If they are, the affected frames are 78 of 16,494 test frames and 104 of 31,013 val frames. The split itself is unchanged (pre-registered). Rule for B3/B4: report val and test results with and without the clips of these three subject ids.


### B2 pretrained keypoint baseline measured (2026-11-23)

- **What was run.** `scripts/ego_perception_b2.py`: COCO-pretrained Keypoint R-CNN, as released, on the val split of the `aria_rect` render set (31,013 frames; test untouched). Protocol and result: `docs/ego_perception_b2_protocol.md` sections 1 to 8.
- **Result.** A person is detected in 4.9% of frames; labelled recall at 25.6 px is 0.077 (0.08 without the DFaust/MoSh shared ids); median error of detected joints 85 px, wrists 21 px; visibility AUC 0.755.
- **Reading by the rules fixed before the run.** Recall below 0.5: B3 must include P-C (adapted pretrained) next to P-A (own model) and P-B (pretrained as released). Side-agnostic recall is not 0.2 higher: left/right naming is not the main error.
- **Not claimed.** Nothing about real images, other pretrained models, or test.

### B3 protocol approved; own BEDLAM-pretrained model kept as a fourth variant (2026-11-23)

- **Decision (owner).** Go ahead with B3 as written in `docs/ego_perception_b3_protocol.md`: P-C (COCO Keypoint R-CNN backbone, adapted) and P-A (ViT-S from random init, ego renders only, labelled as such), two seeds, 30,000 steps, fixed hyperparameters, VAL only.
- **Items 12 and 13 stay on the plan.** The own model "BEDLAM-pretrained by us" is added later as P-A' (same encoder, head and budget; only the initial weights differ) by an amendment to the B3 protocol. B4 (selection, the single use of TEST) is not run until P-A' exists or the owner records that it is dropped, so TEST is not used twice. A claim "own model, BEDLAM-pretrained" is made only after P-A' is trained and evaluated.
- **Code.** `hready/models/ego_perception.py`, `hready/data/ego_perception_data.py`, `scripts/ego_perception_b3_train.py`, `scripts/ego_perception_b3_eval.py`, `tests/test_ego_perception.py`. Checked on random labels only (full train, resume and eval path for both variants on CPU, 6 tests); no real number exists yet.

### B3 two-seed result; B4 runs now, own BEDLAM-pretrained model becomes a later extra row (2026-11-24)

- **B3 seed 1 done** (`docs/ego_perception_b3_protocol.md` section 9): VAL visible-joint 3D mean, P-C 4.253 cm and P-A 5.552 cm; two-seed means 4.23 and 5.57 cm, seed differences 0.05 and 0.04 cm. By the pre-registered criterion P-C is ahead.
- **Decision (owner delegated, taken by Claude, 2026-11-24).** B4 is run now with P-C and P-A (and P-B, 2D only). Items 12 and 13 stay on the plan; their model P-A' is added later as an extra row "added after selection" with VAL numbers only. This reverses the earlier rule "B4 waits for P-A'" (2026-11-23). Reason: the numbers on synthetic frames are close to saturation, the real test of perception is E5b on real data, and the critical path (E4, E5) was blocked. The selection rule and TEST single use are unchanged.
- **Protocol:** `docs/ego_perception_b4_protocol.md` (written before TEST is read). Script `scripts/ego_perception_b4.py`, converter `hready/data/ego_evidence.py`, checked on synthetic geometry and on random labels only.

### B4 run: P-C selected, TEST used once (2026-11-24)

- **What was run.** `scripts/ego_perception_b4.py` once (protocol `docs/ego_perception_b4_protocol.md`): selection P-C (VAL mean 4.229 cm against 5.573 cm for P-A), checkpoint P-C seed 0; val, test and train passes; per-clip evidence in `ego_perception_b4/`.
- **Result.** TEST visible-joint 3D error mean 4.17 cm (median 3.13), VAL 4.21 cm; B2-comparable recall at 25.6 px 0.997; no val-to-test gap; the DFaust/MoSh shared ids change nothing. Result file `results/E/perception/b4_selected_val_test.json`, written once.
- **Not claimed.** Anything about real egocentric images; P-A on TEST; the trunk and head group (6 visible instances on TEST).
- **Open for completion (E4, E5).** The evidence is per frame (render set strided); perceived-evidence windows need contiguous renders of chosen windows.
