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

HumanoidReady is ONE integrated system that goes from **egocentric RGB/video** to a **physically refined full-body human motion**, and evaluates every stage: egocentric RGB/video (+ given head/camera pose and gravity in v1) → trained visual perception → partial 3D evidence (shared schema) → full-body completion → contact prediction and ground plane (foot channel + E1b regions) → physics-aware refinement (**K** kinematic/contact-aware, then **S** SMPL-X humanoid simulation tracking in Newton) → evaluation (E5, E1 cohort slices). Vision is mandatory on the main path. G1 executability (items 6, 11, paper track) is **downstream/supporting**, not the Track E path. Authoritative detail: `docs/project_definition.md` (APPROVED 2026-10-04).

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
`paths.example.yaml` is committed). **Downloads frozen** for the current stage; real egocentric validation (Track E **E5b**) is decided after the first synthetic end-to-end result (`docs/project_definition.md` §7).

## 2. Rules

0. **Physics engine is fixed: NVIDIA Isaac Lab with the Newton backend** for all robot simulation (WP-D, B3) and Newton for human inverse dynamics (WP-B2). PhysX (via Isaac Lab) and MuJoCo are used **only** for the paper's cross-engine study (§6), under the configuration-fair protocol; every single-engine result is reported in Newton.

1. No fabricated numbers. Every number in README/slides/paper comes from a script that ran; outputs in `results/`.
2. "From scratch" = random init for HR-HMR and HR-Refine. Other models (pretrained baselines, and the adapted variant P-C) are allowed, always labeled **pretrained** in every table row and figure.
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
    eval/      (Track E3 oracle baseline CLI)
    baselines/ (E3 heuristic completion)
    robot/     (retarget, isaaclab_newton tracker, metrics)
    dynamics/  (inverse dynamics / torques)
  scripts/   tests/   results/   docs/   paper/
```

---

## 6. Paper track — CVPR 2027 (registration Nov 10, submission Nov 16, 2026, AoE)

> **ON HOLD (2026-10-09).** Paper work is paused; the Track E checklist in §8 drives the work. This section and its checklist are kept unchanged for history. The paper direction will be decided after the literature check listed in §8 (pending decisions).

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
| 7 | HR-Refine model + corruption + virtual cams (incl. ego) + DDP trainer — **candidate learned instance of refinement K** (not the Track E main path) | WP-A2, B2 | AMASS | overfits 1 batch; resume works | ☑ evidence: `hready/models/hr_refine.py`, `hready/train/`, `configs/hr_refine.yaml`, `docs/decisions.md` |
| 8 | HR-Refine training + item-3 loss ablation (supporting evidence for K losses) | WP-A3 | AMASS | `results/A3/ablation.csv` | ☐ |
| 9 | Contact, action, intent heads; IMU fusion ablation — contact overlaps E4; action/intent/IMU supporting (work package C1) | WP-B2, C1 | AMASS, BABEL | `results/B2`, `results/C1`; **affordance F1** (agreement with BABEL action labels) | ☐ |
| 10 | Joint torques (inverse dynamics) on reconstructed body — supporting metric | WP-B2 | AMASS | torque plots | ☐ |
| 11 | G1 on raw vs refined clips; pose-error vs feasibility; QA flag (**downstream**, not Track E) | WP-D | AMASS | `results/D/`; end-to-end: one real video (**3DPW** test sequence) → HMR → HR-Refine → G1 rollout video in `results/D/e2e/` | ☐ |
| 12 | **Perception pretraining (P-A):** BEDLAM loader (2 tars, rotated closeups) + HR-HMR from scratch + augmentations — prerequisite of E2-B3 | WP-A1, B1 | BEDLAM | training curves, overlays | ☐ |
| 13 | **Perception pretraining (P-A):** BEDLAM-CLIFF baseline on same frames; robustness by blur/occlusion/people — prerequisite of E2-B3 | WP-A1, B1 | BEDLAM + ckpt | `results/A1`, `results/B1`; **tracking:** ID-switch count and jitter on `orbit_bigOffice`, temporal transformer vs per-frame baseline | ☐ |
| 14 | Action-conditioned prior ± physics regularization, evaluated on G1 (**downstream**) | WP-B3 | AMASS, BABEL | `results/B3/` | ☐ |
| 15 | DDP equivalence test + Kaggle 2×T4 run | WP-A2 | AMASS | scaling table | ☐ |
| 16 | Loader bottleneck before/after; shard + rolling-window pipeline | WP-C2, Req. multi-TB | AMASS/BEDLAM | `results/C2/` | ☐ |
| 17 | SOMA-X pivot SMPL-X ↔ MHR + `body_models.md` | Req. body models | SMPL-X | round-trip error reported. **SMPL-X ↔ MHR** goes through the **SOMA-X pivot** (`py-soma-x` tools convert **to** SOMA; no direct SMPL-X↔MHR API documented). Implement in a **separate conda env `hready-soma`** (chumpy conflicts with NumPy 2.x in `hready`); API verified from the installed package (rule 3). **GHUM** in `docs/body_models.md` only (not supported by SOMA-X). | ☐ |
| 18 | README with §4 table linked to evidence; `dataset_challenges.md`; `pivot_log.md` | all | — | every number traced to `results/` | ☐ |
| 19 | Paper draft + slides | Strong signal | — | compiles | ☐ |
| 20a | File access requests for Human3.6M, EgoBody and PROX *(no downloads yet)* | Strong signal | — | requests filed | ☐ |
| 20b | *(later)* Human3.6M / PROX eval on a small subset; multi-TB run; **EgoBody → Track E E5b** | Strong signal | needs download | — | ☐ |

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
torque and robot clauses; 12–13 cover perception pretraining (E2-B3 prerequisites); 14–19 complete the rest. Track E main path is `docs/project_definition.md` (APPROVED 2026-10-04); item 8 ablation paused on `wip/fast-loader`.

## 8. Extension track E — egocentric RGB to physics-refined motion (main path)

**Status:** active (Oct 4, 2026). **No G1** on this path (items 6/11 remain downstream).

**Decision record (2026-10-09): switch from oracle evidence to realistic observations.** E3/E4 were developed and judged on oracle evidence (visible-joint 3D positions with small Gaussian noise, exact head pose, gravity and floor height). That path stays as a labelled upper-bound control, as defined in `docs/project_definition.md` (E2-A, [C2]). The main path moves to realistic observations, as already specified by D2/D3/[C3]: 2D keypoints with confidence and visibility, head pose from SLAM-like tracking, gravity, and floor height estimated rather than given. E4 is closed as a completed oracle-evidence study (see its row below). No acceptance rule of a completed experiment is changed.

**Authoritative spec:** `docs/project_definition.md` (binding decisions **D1–D4**, stage contracts, evidence schema, controls, Track E order). Summary chain:

```
egocentric RGB/video (+ given head pose + gravity, v1)
  -> visual perception (E2-B)
  -> evidence schema (E2-A defines; oracle + perceived share it)
  -> completion (E3 baselines, E4 generative prior)
  -> contact + ground (foot + E1b region channels)
  -> refinement K then S (E7 arms: none, K, S, K+S)
  -> evaluation E5 (+ E5b real ego when scheduled)
```

- **D1:** metric head/camera pose and gravity are **given** in v1 (rig); stated on every results table.
- **D2:** perception variants P-A (our HR-HMR-style, random init + ego adapt), P-B (labeled pretrained), P-C (adapted if needed); weak HR-HMR does not block the pipeline — **E2-B4** selects the main-path model.
- **D3:** synthetic egocentric RGB from AMASS first; **E5b** = real egocentric validation (explicit step, not indefinite later).
- **D4:** refinement **K** (kinematic/contact-aware, item-3 losses; HR-Refine a candidate learned K) then **S** (SMPL-X humanoid simulation tracking, Newton only).

**Controls:** **E2-A** oracle path (AMASS → observation simulator → schema → …) is a **control**, not the product interface. **E3** runs oracle mode first for diagnosis only; perceived RGB path is the main end-to-end system. **E5** main rows = perceived; oracle rows = control; perception-only row + decomposition.

**Evidence schema (E2-A):** head 6-DoF; visible-joint 3D in world frame; per-joint visibility mask and confidence; optional 2D keypoints from virtual head camera. Completion consumes **only** schema tensors (+ given rig signals per [C1]).

**E2-B sub-steps (critical path perception):** **B1** ego render set; **B2** baseline measurement (~50 frames then VAL, not a go/no-go gate); **B3** train/adapt P-A/P-B/P-C (items **12–13** prerequisites for P-A); **B4** select on VAL (criterion frozen before TEST), emit schema for splits.

**Stage S (E6a/E6/E7):** SMPL-X-skeleton humanoid asset generated in-repo; PPO DeepMimic-style tracking in Newton; **E7** compares none vs K vs S vs K+S on reconstructed motion. Reuses `hready/robot/` Newton runner patterns from item 6 but **not** G1/GMR.

**Floor-work cohorts (frozen E1):** evaluable floor-work metrics = **kneel + lie** only (`floor_work_eligible`); **crawl** and **yoga_like** have **0** confirmed segments; **sit_floor** / **sit_support** reported separately (sit_floor **not** eligible). E1b region contact labels are GT-derived training targets; at test time contact is predicted from perceived/completed evidence.

**Completion training direction [C3] (open, prompts for E3/E4):** final training targets evidence whose error distribution approximates perception (calibrated from held-out perception errors; oracle evidence is a control, not the final training distribution). See `docs/project_definition.md` §8.

**Fixed constraints.** Single local GPU (RTX 4090); no cloud; **no new downloads** until E5b decision. Newton only (rule 0). Checkpoints/logs under `hready_data`. No multi-GPU / multi-TB claims beyond items 15–16 measured paths.

**Reuse (do not delete working modules):** items 2–5 loaders/contact/metrics/losses; item 7 virtual-camera/IMU code for controls; item 6 Newton orchestration patterns for stage S only.

### Track E checklist (work in order; one item at a time)
| # | Item | Done when | ☐/☑ |
|---|---|---|---|
| E0 | Audit: what exists, what is reused, what conflicts | written, reviewed by user | ☑ evidence: `docs/e0_audit.md` |
| E1 | Subject split + floor-work subset (kneel/lie eligible; crawl/yoga 0 confirmed; sit_floor separate) | reproducible, documented | ☑ evidence: `results/E/splits.json`, `results/E/cohort_counts.json`, `results/E/floor_work_clips.csv`, `docs/ego_splits.md` |
| E1b | Support-region contact labels v2 (E1b; foot channel unchanged) | frozen thresholds + rates JSON | ☑ evidence: `docs/contact_labels_v2.md` (commit `b9dd8ce`) |
| E2-A | Oracle-evidence control + evidence schema + leak-proof observation simulator | `docs/ego_observation_model.md` + oracle path config | ☑ evidence: `docs/ego_observation_model.md`, `configs/ego_observation.yaml`, `hready/data/ego_observation.py`, `hready/models/ego_completion.py`, `hready/body/joint_indices.py` |
| E2-C | Realistic observation model: 2D keypoints + confidence + visibility, head pose from SLAM-like tracking, estimated floor height; noise/visibility model calibrated from measured detector errors (synthetic first); uncertainty-weighted evidence lock; leak check for the new schema; design note written and dated before any training | design doc + leak check + frozen pre-registration (primary arm, decision set, seeds, noise-derived margins) | ☐ |
| E2-B1 | Ego render set (virtual head camera; E1 splits; camera/appearance spec recorded) | render manifest + spec doc | ☐ |
| E2-B2 | Perception baseline measurement (pretrained zero-shot; ~50 frames then VAL) | measurement table (not a gate) | ☐ |
| E2-B3 | Train/adapt perception P-A / P-B / P-C (items 12–13 for P-A) | checkpoints under hready_data | ☐ |
| E2-B4 | Select main-path model on VAL; emit schema evidence train/val/test | selection record + emitted schema | ☐ |
| E3 | Completion baselines (heuristic, regression) — **oracle mode first** (control), then perceived after B4 | `results/E/` tables (oracle labeled control) | ☑ oracle mode, evidence: `results/E/e3_oracle_run.json` (VAL+TEST tables, heuristic check, leak check; commit `72d5c38`), `docs/e3_oracle.md`. Perceived mode after E2-B4: ☐ |
| E4 | Generative prior on schema (visibility + confidence); trains on perceived evidence | logs/ckpts under hready_data | CLOSED as an oracle-evidence study (2026-10-09). Attempt 1: acceptance not met, `docs/e4_completion.md`. Attempt 2 (E4-v2, frozen in commit `8e0adc5`): no arm passed the frozen rule (deterministic `det_w0` passed 4 of 5 gates and missed the contact-ECE gate; TEST not evaluated), recorded in `results/E/e4_v2_run.json`, `docs/e4_v2_plan.md`. Training on realistic evidence continues under E2-C / E4b. ☐ |
| E4b | Decoupled grounding: accuracy-trained completion + contact-conditioned minimal-displacement foot-ground stage (candidate kinematic refinement K, D4) + calibrated contact probabilities; contact metrics computed on the predicted mesh with the same definition as the labels; calibration metric and margin fixed from measured noise before training | dated pre-registration, then `results/E/` | ☐ |
| E5 | Eval command: perceived rows main; oracle control; perception-only row; decomposition; floor-work slice (kneel+lie) | `results/E/` + reliability plot | ☐ |
| E5b | Real egocentric validation (dataset decision + download exception; data access and a 2D-detector pilot may start in parallel with E2-C, after the owner's decision) | planned eval under `results/E/` | ☐ |
| E6a | SMPL-X-skeleton humanoid asset; Newton acceptance (static pose ≥5 s) | `results/E/asset/` | ☐ |
| E6 | PPO stage-S tracking trains on Newton | envs + wall-clock logged | ☐ |
| E7 | Refinement arms none / K / S / K+S on reconstructed motion | `results/E/` | ☐ |
| E8 | Qualitative: ego RGB, perceived evidence, GT vs completion vs physics; ≥1 kneel/lie clip | local mp4 (not in repo) | ☐ |
| E9 | Claims: perception vs completion vs physics separate; oracle vs perceived separate; D2 labeling | claims map with commands | ☐ |

**Pending decisions (2026-10-09).** (1) Literature check outcome: closest prior work and what remains different. (2) Real egocentric dataset choice, license, size and the download exception (§1 freeze stays until then). (3) Calibration metric for gating (Brier score vs debiased ECE) and how its margin is derived from measured noise. (4) Decision set: the 19 untouched tune subjects vs more data; AMASS VAL is a development set only. (5) Training-seed policy for compared arms. (6) Paper direction (§6 on hold).

**Integrity rules (apply to every E item).** Existing modules, commands and results keep working exactly as before (re-run the regression of §7 items 2–5; item 6 when robot code changes). No fabricated, estimated or placeholder numbers; a result not run is "not run". Negative or weak results are reported as found. Scratch checks follow rule 5. No large data, checkpoints or renders in the repo. All seeds fixed and recorded.
