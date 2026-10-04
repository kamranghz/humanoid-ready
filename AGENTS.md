# HumanoidReady v8.1 — project specification + CVPR 2027 paper track

> Save as `AGENTS.md` at the repo root (Cursor reads it; same file works as `CLAUDE.md`).
> **Agent: read §0–§2 the paper track (§6) and the checklist (§7) before any task. Work on one checklist item at a time.
> Mark it ☑ with its evidence path when done. Do not add scope that is not in this file.**

This version is organized by work packages (WP); each work package has one piece of evidence. Nothing else is in scope.
Items marked *(later)* need data that is not downloaded yet and are done only if time allows.

---

> **Naming:** project **HumanoidReady** (repo `humanoid-ready`, package `hready`, conda env `hready`). Image→SMPL-X model **HR-HMR**; physics refinement model **HR-Refine**. "HumanoidReady" (arXiv 2411.17189) and "PhysHMR" (arXiv 2510.02566) are existing works — never use those names.
> Tagline: *From human video to physically verified, humanoid-ready motion.*

## 0. One-paragraph goal

Build **HumanoidReady**: a from-scratch human-motion perception stack that goes from images/video to SMPL-X bodies,
cleans the motion with physics and biomechanics losses, estimates contact and joint torques, and checks whether the
motion is executable by a Unitree G1 humanoid in Isaac Lab + Newton. Main message:*pose accuracy is not
physical usability — this stack measures and enforces the latter.*

## 1. Data and models available (frozen — no more downloads unless a WP marked *(later)* is started)

| Asset | Path (under `D:\projects\hready_data`) | Used by |
|---|---|---|
| SMPL-X `locked_head` (neutral, 16 betas) — **single training body model** | `models/smplx/locked_head/` | all |
| SMPL-X `v1_1` — only for the BEDLAM-CLIFF baseline output | `models/smplx/v1_1/` | WP-A1 baseline |
| SMPL-X extras (segm, flip, model_transfer), MANO | `models/smplx/extras/`, `models/mano/` | WP-A1, WP-D |
| AMASS SMPL-X N (21 subsets + MOYO) | `datasets/amass/smplx_n/` | WP-A2, A3, B2, B3, C1, D |
| BABEL v1.0 | `datasets/babel/babel_v1.0_release/` | WP-B3, C1 |
| BEDLAM labels, locked-head 16b (training) | `datasets/bedlam/labels/lockedhead_16b/` | WP-A1 |
| BEDLAM images: 2 tars (`..._closeup_suburb_a_6fps`, `..._orbit_bigOffice_6fps`) | `datasets/bedlam/images/` | WP-A1, B1 |
| BEDLAM-CLIFF / BEDLAM-HMR checkpoints (pretrained baseline) | `data/BEDLAM/checkpoints/` | WP-A1 |
| 3DPW test video + GT *(to download, paper item R3)*; EMDB if obtainable | `datasets/3dpw/`, `datasets/emdb/` | §6 paper |

The repo **never** contains these files. Code reads paths from `configs/paths.yaml` (git-ignored; a
`paths.example.yaml` is committed).

## 2. Rules

0. **Physics engine is fixed: NVIDIA Isaac Lab with the Newton backend** for all robot simulation (WP-D, B3) and Newton for human inverse dynamics (WP-B2). PhysX (via Isaac Lab) and MuJoCo are used **only** for the paper's cross-engine study (§6), under the configuration-fair protocol; every single-engine result is reported in Newton.

1. No fabricated numbers. Every number in README/slides/paper comes from a script that ran; outputs in `results/`.
2. "From scratch" = random init for HR-HMR and HR-Refine. Pretrained models appear only as labeled baselines.
3. Verify library APIs from installed packages (Isaac Lab, Newton, smplx, SOMA-X change often); pin versions in `env/versions.lock`.
4. Never commit licensed models or data. `.gitignore` covers `models/`, `datasets/`, `data/`, checkpoints, shards.
5. Each WP has its own CLI entry point. Checks are run but not saved: the agent may run any test, smoke test or sanity script to verify its work, but it does so from a scratch location (e.g. a temp dir or `python -c`), shows the raw output, and deletes every such file before finishing. No test files, mock files, placeholder reports, demo scripts or other extra files are committed. `git status --short --untracked-files=all` must list only files the current checklist item calls for.
6. Blocked > 2 h → use the WP's fallback and log it in `docs/pivot_log.md` (it also records how the plan changed).
7. Small scale is fine. State the scale honestly (e.g. "trained on 2 BEDLAM scenes").

---

### 2b. Environment decision (S3, Oct 1 2026)

- Machine: Windows 11, **RTX 4090 24 GB** (driver 610.60, CUDA 13.3 runtime available), data on `D:\projects\hready_data`.
- **Windows-native conda** for all project code: env `hready` (PyTorch, smplx, training, data, metrics). Reasons: Isaac Sim/Lab already run natively on Windows, all existing envs are Windows conda, and reading `D:` from WSL2 (`/mnt/d`) is slow.
- **Isaac Lab + Newton:** reuse the existing Isaac Lab installation (separate env); `hready` talks to it through files (retargeted trajectories in, metrics/videos out), not imports.
- **WSL2 only as fallback**, per baseline, if a public HMR method (GVHMR, WHAM, TRAM) does not install on Windows (Linux-only CUDA extensions). Log every such case in `docs/pivot_log.md`.
- DDP on Windows uses the `gloo` backend (no NCCL); multi-GPU NCCL runs happen on Kaggle (Linux). `torch.compile` is optional on Windows.
- **S4 (done):** conda/package pins recorded in `env/versions.lock` (see rule 3).

## 3. Work packages

### A. Architecting Proprietary Articulation Models

**WP-A1 — Zero-to-One model: 3D pose, dense mesh, kinematic tracking** *(scope: 3D human pose estimation, dense full-body mesh recovery and kinematic tracking; vision models trained from scratch)*
- **HR-HMR:** ViT-S/16 encoder, random init, on person crops → transformer decoder → SMPL-X (6D joint rotations, 16 betas, camera) → full 10,475-vertex mesh.
- **Tracking:** temporal transformer over per-frame tokens for video; simple IoU/ID tracker for multiple people.
- **Data:** BEDLAM 2 tars + locked-head labels. Handle rotated `closeup` images. Split by subject.
- **Baseline:** BEDLAM-CLIFF checkpoint (pretrained HRNet backbone, labeled as such), same held-out frames, compared on vertices/joints.
- **Evidence:** `results/A1/` — MPJPE, PA-MPJPE, PVE vs baseline; training curves; overlay images. State honestly that HR-HMR is trained on 2 scenes from random init.
- *Fallback:* ResNet-18-sized CNN encoder from scratch.

**WP-A2 — Distributed, multi-view, temporal training** *(scope: multi-view and temporal architectures on multiple GPUs and multi-modal data; PyTorch scaling)*
- **HR-Refine:** spatio-temporal transformer that takes noisy SMPL-X sequences + 2D keypoints from 1–4 **virtual cameras** (incl. a head-mounted egocentric camera) + optional synthetic IMU, and outputs clean world-frame motion.
- Training data: AMASS, corrupted on the fly (jitter, occlusion masks, dropout, foot sinking).
- One code path for 1→N GPUs: `torchrun` + DDP (FSDP option), bf16, checkpoint/resume.
- Multi-GPU evidence: DDP equivalence test (2 processes vs 1, same result) + one Kaggle 2×T4 run (1 vs 2 GPU throughput).
- **Evidence:** `results/A2/` — scaling table; 1/2/4-view ablation.

**WP-A3 — Loss innovation** *(scope: loss functions for biomechanical constraints, temporal smoothness, postural balance and physical plausibility)*
- Losses, each switchable: foot skating (contact-weighted), ground penetration, bone-length consistency, anatomical ROM, CoM-in-support-polygon (balance), gravity/momentum consistency in flight, acceleration/jerk smoothness.
- Each loss has a unit test: 0 on valid motion, > 0 with correct gradient sign on violating motion.
- **Evidence:** `results/A3/ablation.csv` — full vs reconstruction-only vs minus each loss; physical metrics + PA-MPJPE cost.

### B. Human-Scene Interaction & Complex Motion

**WP-B1 — Motion blur, self-occlusion, multi-person crowding** *(scope: dynamic scene understanding)*
- HR-HMR augmentations: motion-blur kernels, synthetic occluders, crop truncation; the multi-person BEDLAM tar (orbit_bigOffice, 3 people) for crowding.
- HR-Refine robustness: occlusion-rate and jitter sweeps.
- **Evidence:** `results/B1/` — error vs blur level, occlusion level, number of people.

**WP-B2 — Allocentric & egocentric tracking; foot contact, joint torques, affordances** *(scope: tracking bodies through complex spaces; foot-ground contact, joint torques and environmental affordances)*
- Allocentric = fixed/orbit cameras; egocentric = virtual head-mounted camera in HR-Refine (same model, ego-only ablation).
- **Contact:** per-vertex contact head (labels from clean AMASS: height + velocity thresholds); report foot-contact F1.
- **Joint torques:** inverse dynamics on SMPL-X with segment masses, computed with **NVIDIA Newton** (default); a simple recursive Newton–Euler implementation is the fallback; optional OpenSim cross-check.
- **Affordances (scoped):** support-surface estimation — classify **stand / sit / lean** plus **support height** from foot–body contacts. BABEL action labels (e.g. sit, lean) are a **proxy** for support surfaces; report **agreement with BABEL action labels** (affordance F1), not ground-truth surface accuracy. Real scene meshes (PROX / EgoBody) remain item **20b** / paper *(later)*.
- **Evidence:** `results/B2/` — contact F1, torque plots raw vs refined, ego-only vs allocentric error.
- *(later)* PROX / EgoBody for real scene meshes and real egocentric video.

**WP-B3 — Regularization for action-conditioned humanoid models** *(scope: regularization for downstream action-conditioned humanoid models)*
- Small action-conditioned motion prior on AMASS + BABEL labels, two variants: plain vs regularized with HumanoidReady signals (contact, torque bounds, balance).
- Compare generated motion on physical metrics and on G1 (WP-D).
- **Evidence:** `results/B3/`.

### C. Emergent Perception R&D

**WP-C1 — Rapid prototyping: action segmentation, intent prediction, sensor integration** *(scope: action segmentation, intent prediction and sensor integration)*
- Action segmentation head on HR-Refine (BABEL frame labels): frame accuracy, edit score.
- Intent head: predict next 0.5–1.0 s of root + pose: ADE/FDE.
- Sensor integration: synthetic body-worn IMU from AMASS fused with video keypoints: video-only vs IMU-only vs fused.
- **Evidence:** `results/C1/`.

**WP-C2 — Agile problem solving in the data engine** *(scope: resolving data-pipeline bottlenecks)*
- Profile the AMASS/BEDLAM loader, fix the top bottleneck, report before/after throughput.
- Keep `docs/pivot_log.md` with dated problem → decision → outcome entries.
- **Evidence:** `results/C2/bottleneck.md`, `docs/pivot_log.md`.

### D. Dense Contact & Physics-Aware Tracking

**WP-D — Pipeline to robot control** *(scope: contact surfaces, gravity and momentum, linking human video to robot control and locomotion)*
- Retarget SMPL-X → Unitree G1 (verified open-source retargeter or IK). Track in **Isaac Lab + Newton** with PD + feedforward.
- Metrics: time-to-fall, tracking error, foot slip, torque saturation, assist wrench (force needed to keep the robot on the reference).
- Main result: raw vs HR-Refine-cleaned motion on G1; and correlation between pose error and robot feasibility (*pose accuracy ≠ physical usability*).
- Physics QA flag per clip (PASS / REVIEW / FAIL with reason) from the physical metrics.
- **Evidence:** `results/D/` — table + rollout videos.
- *Optional differentiator (thesis link):* same clips in PhysX and MuJoCo to check engine dependence.
- *Fallback:* standalone Newton with G1 MJCF.

---

## 4. "Who You Are" → evidence

| Capability | Evidence |
|---|---|
| Deep learning, 3D CV, articulated tracking | WP-A1, A2, B1. **Scope:** HR-Refine **egocentric camera is virtual** until EgoBody eval (item 20b); **body-worn IMU is synthetic** (AMASS-derived, not hardware). |
| Training large-scale vision models from scratch | WP-A1 (random-init ViT) + WP-A2 (DDP/multi-GPU path). **Scope:** **HR-HMR trained on 2 BEDLAM scenes** from random init (not full BEDLAM). |
| SMPL / SMPL-X / GHUM / MHR / SOMA-X, IK, dense mesh | SMPL-X throughout; **SOMA-X pivot** SMPL-X ↔ MHR (item 17); IK in retargeting; dense 10,475-vertex mesh; `docs/body_models.md` comparing models (**GHUM documented only** — not supported by SOMA-X). |
| PyTorch and scaling frameworks | DDP/FSDP, bf16, `torch.compile`, checkpoint/resume (WP-A2). **Scope:** DDP path verified on **1 GPU (Windows gloo) + Kaggle 2×T4**, not a large NCCL cluster. |
| Multi-TB image/video data | Streaming shard pipeline + rolling-window processing (download → process → delete), measured throughput. **Scope:** multi-TB handled as a **measured streaming pipeline**, not a multi-TB training run *(full mirror run later)*. |
| Fast-paced, shifting priorities | `docs/pivot_log.md` |
| **Strong signal:** first-author top-tier paper | `paper/` draft from these results, working title "Pose Accuracy Is Not Physical Usability" |
| **Strong signal:** AMASS, Human3.6M, EgoBody, PROX and their optimization challenges | AMASS used throughout; `docs/dataset_challenges.md` (SMPL-X version mismatch, frame rates, GT artifacts in AMASS, BEDLAM rotated images, motion leakage between AMASS and BEDLAM). *(later)* Human3.6M, EgoBody, PROX evaluation (item **20b**) |

---

## 5. Repository layout

```
humanoid-ready/
  AGENTS.md  README.md  pyproject.toml
  configs/   (paths.example.yaml, one yaml per WP)
  hready/            (Python package)
    body/      (smplx wrapper, rotations, soma-x wrapper)
    data/      (amass, babel, bedlam loaders; corruption; synthetic IMU; shards)
    losses/    (physics + biomechanics losses)
    metrics/   (pose, physical, contact, stats/bootstrap)
    models/    (hr_hmr, hr_refine, heads)
    train/     (DDP trainer)
    robot/     (retarget, isaaclab_newton tracker, metrics)
    dynamics/  (inverse dynamics / torques)
  scripts/   tests/   results/   docs/   paper/
```

---

## 6. Paper track — CVPR 2027 (registration Nov 10, submission Nov 16, 2026, AoE)

The paper is written from the same code and results as the work packages. It does **not** need every checklist item.

**Working title:** *Is Humanoid Executability a Reliable Benchmark for Human Motion Recovery?*

**Positioning (from the literature check, Oct 1 2026).**
- Physics losses for HMR are well covered (PhysCap, SimPoE, PhysDiff, PhysPT, PhysHMR, body-momentum). HR-Refine is a tool here, not the contribution.
- Simulation-based plausibility metrics already exist for simulated human characters (Measuring Physical Plausibility…, 2025), and BeyondRetarget (arXiv 2609.29850) reports method-level agreement between pose error and G1 execution success in MuJoCo. The paper must cite both and not claim "MPJPE is unrelated to executability" in general.
- PolySim and GMR show simulators disagree for humanoid **policies**; no work found tests whether **evaluation verdicts on human-motion data** (method rankings, per-clip PASS/FAIL) survive a change of physics engine. This is the open gap and the paper's core.

**Contributions.**
1. A configuration-fair protocol to evaluate HMR outputs by humanoid executability (retarget → fixed open-source G1 tracking policy → executability metrics), with HMR error and retargeting error separated.
2. Method-level vs **clip-level** analysis: agreement between MPJPE/PA-MPJPE, physical metrics, and executability (Spearman/Kendall with subject-cluster bootstrap CIs).
3. **Cross-engine stability** of these verdicts in Newton, PhysX and MuJoCo (Kendall τ of method and clip rankings, PASS/FAIL agreement, within-engine repeatability, record of what could not be matched).
4. HR-Refine as a plug-in: does physical refinement change executability, and is that change engine-stable?

**Subjects of evaluation:** public HMR methods GVHMR, WHAM, TRAM, BEDLAM-CLIFF (and HR-HMR if ready), all with their own released checkpoints, labeled as such.

**Data:** 3DPW test (video + GT); EMDB if obtainable in week 1; AMASS ground truth as the "perfect HMR" upper bound.

**Controller:** one open-source general motion-tracking policy for Unitree G1 that runs in all three engines (choose in week 1; PD-only tracking is a fallback and must be labeled as such).

**Rules:** thresholds and analysis plan frozen in `paper/claims_map.md` before the cross-engine runs; negative or mixed findings are reported as found.

## 7. Checklist (work in this order)

| # | Item | Work package | Data | Done when | ☐/☑ |
|---|---|---|---|---|---|
| 1 | Repo skeleton, `pyproject`, `.gitignore`, CPU CI, `paths.example.yaml` | — | none | CI green | ☑ evidence: `.github/workflows/ci.yml`, CI run on commit `10cab7b` |
| 2 | Rotations + SMPL-X wrapper (locked_head; v1_1 only for baseline; mixing raises error) | Required: body models | SMPL-X | checks pass; raw output shown, check files not saved | ☑ evidence: `hready/body/rotations.py`, `hready/body/smplx_wrapper.py` |
| 3 | Physics/biomech losses | WP-A3 | none | each loss checked analytically (0 on valid motion, >0 with correct gradient sign on violation); raw output shown, check files not saved | ☑ evidence: `hready/losses/physics.py`, `hready/losses/biomech.py`, `hready/losses/__init__.py` |
| 4 | Metrics + bootstrap CI | all | none | checked against hand-computed values; raw output shown, check files not saved | ☑ evidence: `hready/metrics/pose.py`, `hready/metrics/physical.py`, `hready/metrics/contact.py`, `hready/metrics/stats.py`, `hready/metrics/__init__.py`, `hready/losses/_constants.py` |
| 5 | AMASS + BABEL loader (30 fps, Z-up, floor z=0), contact labels, synthetic IMU | WP-A2, C1 | AMASS, BABEL | 3 clips visually checked | ☑ evidence: `hready/data/amass.py`, `hready/data/babel.py`, `hready/data/contact.py`, `hready/data/imu.py`, `hready/data/foot_height_rise.py`, `scripts/inspect_clip.py`, `scripts/babel_gait_stats.py`, `results/checks/` (CMU/132/132_35, ACCAD C20 run_to_jump, BMLrub treadmill), `docs/dataset_challenges.md`, `docs/pivot_log.md`; contact core vs `27cc2da` on 300 `foot_traj` clips (seed 0) |
| 6 | **G1 smoke test:** one AMASS walk → G1 in Isaac Lab + Newton, metrics + video | WP-D | AMASS | video + JSON | ☑ evidence: `hready/robot/isaaclab_newton.py`, `hready/robot/run_smoke.py`, `hready/robot/metrics.py`, `hready/robot/replay.py`, `results/D/smoke/summary.json`, `results/D/smoke/*/metrics_*.json`, `results/D/smoke/*/replay_*.mp4`, `docs/decisions.md`, `docs/pivot_log.md`; visual sign-off CMU walk / ACCAD jump / BMLrub treadmill (`76bf7e2`) |
| 7 | HR-Refine model + corruption + virtual cams (incl. ego) + DDP trainer | WP-A2, B2 | AMASS | overfits 1 batch; resume works | ☑ evidence: `hready/models/hr_refine.py`, `hready/train/`, `configs/hr_refine.yaml`, `docs/decisions.md` |
| 8 | HR-Refine training + loss ablation | WP-A3 | AMASS | `results/A3/ablation.csv` | ☐ |
| 9 | Contact, action, intent heads; IMU fusion ablation | WP-B2, C1 | AMASS, BABEL | `results/B2`, `results/C1`; **affordance F1** (agreement with BABEL action labels) | ☐ |
| 10 | Joint torques (inverse dynamics) raw vs refined | WP-B2 | AMASS | torque plots | ☐ |
| 11 | G1 on raw vs refined clips; pose-error vs feasibility; QA flag | WP-D | AMASS | `results/D/`; end-to-end: one real video (**3DPW** test sequence) → HMR → HR-Refine → G1 rollout video in `results/D/e2e/` | ☐ |
| 12 | BEDLAM loader (2 tars, rotated closeups) + HR-HMR from scratch + augmentations | WP-A1, B1 | BEDLAM | training curves, overlays | ☐ |
| 13 | BEDLAM-CLIFF baseline on same frames; robustness by blur/occlusion/people | WP-A1, B1 | BEDLAM + ckpt | `results/A1`, `results/B1`; **tracking:** ID-switch count and jitter on `orbit_bigOffice`, temporal transformer vs per-frame baseline | ☐ |
| 14 | Action-conditioned prior ± physics regularization, evaluated on G1 | WP-B3 | AMASS, BABEL | `results/B3/` | ☐ |
| 15 | DDP equivalence test + Kaggle 2×T4 run | WP-A2 | AMASS | scaling table | ☐ |
| 16 | Loader bottleneck before/after; shard + rolling-window pipeline | WP-C2, Req. multi-TB | AMASS/BEDLAM | `results/C2/` | ☐ |
| 17 | SOMA-X pivot SMPL-X ↔ MHR + `body_models.md` | Req. body models | SMPL-X | round-trip error reported. **SMPL-X ↔ MHR** goes through the **SOMA-X pivot** (`py-soma-x` tools convert **to** SOMA; no direct SMPL-X↔MHR API documented). Implement in a **separate conda env `hready-soma`** (chumpy conflicts with NumPy 2.x in `hready`); API verified from the installed package (rule 3). **GHUM** in `docs/body_models.md` only (not supported by SOMA-X). | ☐ |
| 18 | README with §4 table linked to evidence; `dataset_challenges.md`; `pivot_log.md` | all | — | every number traced to `results/` | ☐ |
| 19 | Paper draft + slides | Strong signal | — | compiles | ☐ |
| 20a | File access requests for Human3.6M, EgoBody and PROX *(no downloads yet)* | Strong signal | — | requests filed | ☐ |
| 20b | *(later)* Human3.6M / EgoBody / PROX eval on a small subset; multi-TB run | Strong signal | needs download | — | ☐ |

### Paper track checklist (interleaved with the items above; see the week plan)

| # | Item | Done when | ☐/☑ |
|---|---|---|---|
| R1 | Read BeyondRetarget, Measuring Physical Plausibility, PolySim, GMR, PHUMA, PhysHMR in full; `paper/related_work.md` | each paper: setup, metrics, engines, overlap with us | ☐ |
| R2 | Choose the G1 tracking policy that runs in Newton, PhysX and MuJoCo; record in `docs/decisions.md` | runs one AMASS clip in all three | ☐ |
| R3 | Download 3DPW (and EMDB if available) | loader test passes | ☐ |
| R4 | Run GVHMR, WHAM, TRAM, BEDLAM-CLIFF on the test videos; convert to SMPL-X locked_head | per-method outputs + MPJPE/PA-MPJPE match published numbers within tolerance | ☐ |
| R5 | Freeze analysis plan + thresholds in `paper/claims_map.md` | dated commit before R7 | ☐ |
| R6 | Executability in Newton for all methods + AMASS GT upper bound; method- and clip-level analysis | `results/paper/newton/` | ☐ |
| R7 | Same in PhysX and MuJoCo; configuration-fairness record; Kendall τ, PASS/FAIL agreement, repeatability | `results/paper/cross_engine/` | ☐ |
| R8 | HR-Refine on all method outputs; effect on executability per engine | `results/paper/refine/`, `results/D/e2e/` | ☐ |
| R9 | Draft: intro, related work, method, experiments, limitations; every number from `results/paper/` | compiles in CVPR template | ☐ |
| R10 | Register abstract (Nov 10) and submit (Nov 16); supplementary (Nov 23) | submitted | ☐ |

### Week plan to the CVPR deadline

| Week | Dates | Checklist items | Paper items |
|---|---|---|---|
| 1 | Oct 1–8 | S1–S4, 1–4 | R1, R2, R3 |
| 2 | Oct 9–15 | 5, 6 | R4 |
| 3 | Oct 16–22 | 10 | R5, R6 |
| 4 | Oct 23–29 | — | R7 |
| W5 | Oct 30–Nov 5 | 7, 8 | R8, R9 (draft) |
| 6 | Nov 6–16 | — | R9 (final), R10 |
| after | Nov 17 → | 9, 11–19 | camera-ready / workshop / arXiv |


**Minimum milestone:** items 1–6 (and 7–8 if time). Items 1–11 cover the physics, loss, contact,
torque and robot clauses; 12–13 cover from-scratch vision; 14–19 complete the rest. Track E starts now (decided Oct 4, 2026); item 8 ablation is paused (branch `wip/fast-loader` holds the unfinished fast loader and ablation CLI).

## 8. Extension track E — egocentric motion prior + physics tracking (queued; do not start before item 8 is closed and Oct 8, 2026)

**Status:** active (Oct 4, 2026). Item 8 ablation paused on `wip/fast-loader`. **Priority for Oct 7:** E0, E1, E2, E3 first; E4 only if time; no G1 in Track E.

**Why.** The problem:from a head-mounted rig, hands/wrists, upper-body keypoints and the metric head/camera trajectory are observed; legs and feet are rarely visible. Today the lower body is inferred, feet are an ankle offset on an assumed flat floor, and contact is a frame-rejection check. Track E builds a model that outputs a full-body pose per frame in one metric world frame that stands on the floor, does not slide or sink, handles floor work (kneel, sit, lie, crawl), and reports calibrated per-foot contact; a physics stage then makes the result physically consistent.

**Mapping to existing work (reuse, do not rewrite or delete working modules).**
- Module A reuses: AMASS/BABEL loaders and contact labels (item 5), synthetic IMU/virtual-camera code (item 7), losses (item 3), metrics + bootstrap (item 4), smplx wrapper (item 2). It extends WP-B2 (ego-only) and replaces nothing.
- Module B reuses: the Isaac Lab + Newton runner, robot metrics and WSL orchestration from item 6 (hready/robot/). It does NOT use the G1 or the GMR retargeter; the G1 pipeline stays as is for WP-D.

**Fixed constraints.** Single local GPU (RTX 4090); no cloud, no new downloads; everything trains and evaluates on this machine. Newton is the only physics backend (rule 0): if a feature is missing, stop and report with evidence; never silently switch to PhysX. Checkpoints/logs live under hready_data, not the repo. No multi-GPU / multi-TB claims.

### Module A — conditional full-body motion prior
- Body model: SMPL-X locked_head. MHR/SOMA-X adapter only if model files exist locally; else "future work".
- Data: AMASS subset already in hready_data. Deterministic split by SUBJECT (train/val/test) plus a separate "floor work" test subset (kneel, sit, lie, crawl, yoga-like) chosen by documented rules (BABEL labels + geometric rules). One command reproduces splits and subset. Exclude skate-flagged clips and clips with foot-height rise > 5 cm unless a documented reason says otherwise.
- Synthetic egocentric observations (seeded, configurable, documented in docs/ego_observation_model.md): head 6-DoF trajectory, wrist 6-DoF poses, upper-body 3D keypoints, partial/noisy/intermittently visible leg keypoints with realistic noise and dropout.
- Model: ONE conditional generative model (diffusion OR masked transformer; choice justified in this file before training). Outputs full-body pose sequence in metric world frame + per-frame per-foot contact probability. Trainable on one GPU in < ~24 h.
- Baselines on identical splits/metrics: (1) industry heuristic: inferred lower body + ankle offset on flat floor; (2) deterministic regression model of similar size.
- Metrics: MPJPE and lower-body/feet MPJPE (world frame), foot skate, ground penetration, jitter, contact P/R/F1, calibration (ECE + reliability plot). Floor-work subset reported separately.

### Module B — physics-based tracking (Isaac Lab + Newton), SMPL-X-skeleton humanoid
- Humanoid: a simulated humanoid whose skeleton matches the SMPL-X body model (22 body joints, 3-DoF spherical joints as in SMPL-X rotations), so Module A output is tracked directly with no retargeting. The asset is GENERATED programmatically from SMPL-X (bone lengths from the locked_head shape, segment masses/inertias from the de Leva tables already used in the repo, simple capsule/box collision geometry, joint limits from the item-3 ROM table, PD gains documented) into MJCF/URDF under hready/robot/. No downloads. A public SMPL humanoid asset (e.g. from ProtoMotions/MimicKit) may be used as a REFERENCE only if already installed locally; record this in the audit.
- Asset acceptance before any RL: loads in Isaac Lab + Newton; stands passively under PD holding a static AMASS pose for >= 5 s; mass matches the de Leva total; joint ordering round-trips to SMPL-X pose parameters (error reported).
- DeepMimic-style imitation reward (pose, velocity, end-effector, root) + PPO with parallel envs sized to the GPU. ProtoMotions / MimicKit as dependency only if they actually run on this Isaac Lab + Newton setup; verify and record here.
- Compare kinematic-only vs physics-tracked: foot skate, penetration, fall/termination rate, tracking error to GT, contact agreement with Module A. Include floor-work subset. Log number of envs and wall-clock time.
- Scope honesty: state how many clips the policy was trained on; a policy trained on a few clips is not a general tracker. If the humanoid cannot be stabilised, report it with evidence; do not fall back to G1 silently (a G1 fallback needs user approval and is labelled as such).

### Track E checklist (work in order; one item at a time)
| # | Item | Done when | ☐/☑ |
|---|---|---|---|
| E0 | Audit section appended to this file: what exists, what is reused, what conflicts | written, reviewed by user | ☑ evidence: `docs/e0_audit.md` |
| E1 | Subject split + floor-work subset, one command | reproducible, documented | ☑ evidence: `results/E/splits.json`, `results/E/cohort_counts.json`, `results/E/floor_work_clips.csv`, `docs/ego_splits.md` |
| E2 | Ego observation synthesis, seeded + configurable | docs/ego_observation_model.md | ☐ |
| E3 | Baselines (heuristic, regression) on identical splits | results table | ☐ |
| E4 | Generative prior trains end to end from one command | logs/ckpts under hready_data | ☐ |
| E5 | One eval command: full metrics table (test + floor-work) + reliability plot | results/E/ | ☐ |
| E6a | SMPL-X-skeleton humanoid asset generated; passes the asset acceptance above in Newton | evidence in results/E/asset/ | ☐ |
| E6 | PPO tracking trains on Newton with parallel envs | envs + wall-clock logged | ☐ |
| E7 | Kinematic vs physics-tracked table, one command | results/E/ | ☐ |
| E8 | Qualitative clips: GT vs baseline vs ours vs physics-tracked, >= 1 floor-work | local mp4 | ☐ |
| E9 | Claims section lists only logged results with exact command + config | claims_map | ☐ |

**Integrity rules (apply to every E item).** Existing modules, commands and results keep working exactly as before (re-run the regression of §7 items 2–6). No fabricated, estimated or placeholder numbers; a result not run is "not run". Negative or weak results are reported as found. Scratch checks follow rule 5. No large data, checkpoints or renders in the repo. All seeds fixed and recorded.
