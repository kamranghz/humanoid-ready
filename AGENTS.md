# HumanoidReady — project specification and tracker (CVPR 2027 paper track on hold)

> Save as `AGENTS.md` at the repo root (Cursor reads it; same file works as `CLAUDE.md`).
> **Agent: read §0–§2 the paper track (§6) and the checklist (§7) before any task. Work on one checklist item at a time.
> Mark it ☑ with its evidence path when done. Do not add scope that is not in this file.**

This version is organized by work packages; each work package has one piece of evidence. Nothing else is in scope.
Items marked *(later)* need data that is not downloaded yet and are done only if time allows. Old step codes are
listed only in `docs/experiments.md`.

---

> **Naming:** project **HumanoidReady** (repo `humanoid-ready`, package `hready`, conda env `hready`). Image→SMPL-X model **HR-HMR**; physics refinement model **HR-Refine**. "HumanoidReady" (arXiv 2411.17189) and "PhysHMR" (arXiv 2510.02566) are existing works — never use those names.
> Tagline: *From human video to physically verified, humanoid-ready motion.*

## 0. One-paragraph goal

HumanoidReady is ONE integrated system that goes from **egocentric RGB/video** to a **physically refined full-body human motion**, and evaluates every stage: egocentric RGB/video (+ given head/camera pose and gravity in the first system version) → trained visual perception → partial 3D evidence (shared schema) → full-body completion → contact prediction and ground plane (foot channel + support-contact region channels) → physics-aware refinement (**kinematic** contact-aware refinement, then **simulation** tracking of an SMPL-X humanoid in Newton) → evaluation (end-to-end evaluation, sliced by subject-split cohorts). Vision is mandatory on the main path. G1 executability (items 6, 11, paper track) is **downstream/supporting**, not the egocentric motion track. Authoritative detail: `docs/project_definition.md` (APPROVED 2026-10-04).

## 1. Data and models available (paused — no more downloads unless a work package marked *(later)* is started)

| Asset | Path (under `D:\projects\hready_data`) | Used by |
|---|---|---|
| SMPL-X `locked_head` (neutral, 16 betas) — **single training body model** | `models/smplx/locked_head/` | all |
| SMPL-X `v1_1` — only for the BEDLAM-CLIFF baseline output | `models/smplx/v1_1/` | 3D pose model baseline |
| SMPL-X extras (segm, flip, model_transfer), MANO | `models/smplx/extras/`, `models/mano/` | 3D pose model, robot pipeline |
| AMASS SMPL-X N (21 subsets + MOYO) | `datasets/amass/smplx_n/` | refinement training, physics-loss study, contact and torques, motion prior, auxiliary heads, robot pipeline |
| BABEL v1.0 | `datasets/babel/babel_v1.0_release/` | motion prior, auxiliary heads |
| BEDLAM labels, locked-head 16b (training) | `datasets/bedlam/labels/lockedhead_16b/` | 3D pose model |
| BEDLAM images: 2 tars (`..._closeup_suburb_a_6fps`, `..._orbit_bigOffice_6fps`) | `datasets/bedlam/images/` | 3D pose model, robustness |
| BEDLAM-CLIFF / BEDLAM-HMR checkpoints (pretrained baseline) | `data/BEDLAM/checkpoints/` | 3D pose model |
| 3DPW test video + GT *(to download, paper task "external video data")*; EMDB if obtainable | `datasets/3dpw/`, `datasets/emdb/` | §6 paper |

The repo **never** contains these files. Code reads paths from `configs/paths.yaml` (git-ignored; a
`paths.example.yaml` is committed). **Downloads paused** for the current stage; real egocentric validation (egocentric motion track, **real egocentric validation**) is decided after the first synthetic end-to-end result (`docs/project_definition.md` §7).

## 2. Rules

0. **Physics engine is fixed: NVIDIA Isaac Lab with the Newton backend** for all robot simulation (robot pipeline, motion prior) and Newton for human inverse dynamics (contact and torques). PhysX (via Isaac Lab) and MuJoCo are used **only** for the paper's cross-engine study (§6), under the configuration-fair protocol; every single-engine result is reported in Newton.

1. No fabricated numbers. Every number in README/slides/paper comes from a script that ran; outputs in `results/`.
2. "From scratch" = random init for HR-HMR and HR-Refine. Other models (pretrained baselines, and the adapted pretrained backbone) are allowed, always labeled **pretrained** in every table row and figure.
3. Verify library APIs from installed packages (Isaac Lab, Newton, smplx, SOMA-X change often); pin versions in `env/versions.lock`.
4. Never commit licensed models or data. `.gitignore` covers `models/`, `datasets/`, `data/`, checkpoints, shards.
5. Each work package has its own CLI entry point. Checks are run but not saved: the agent may run any test, smoke test or sanity script to verify its work, but it does so from a scratch location (e.g. a temp dir or `python -c`), shows the raw output, and deletes every such file before finishing. No test files, mock files, placeholder reports, demo scripts or other extra files are committed. `git status --short --untracked-files=all` must list only files the current checklist item calls for.
6. Blocked > 2 h → use the work package's fallback and log it in `docs/pivot_log.md` (it also records how the plan changed).
7. Small scale is fine. State the scale honestly (e.g. "trained on 2 BEDLAM scenes").

---

### 2b. Environment decision (environment setup, Oct 1 2026)

- Machine: Windows 11, **RTX 4090 24 GB** (driver 610.60, CUDA 13.3 runtime available), data on `D:\projects\hready_data`.
- **Windows-native conda** for all project code: env `hready` (PyTorch, smplx, training, data, metrics). Reasons: Isaac Sim/Lab already run natively on Windows, all existing envs are Windows conda, and reading `D:` from WSL2 (`/mnt/d`) is slow.
- **Isaac Lab + Newton:** reuse the existing Isaac Lab installation (separate env); `hready` talks to it through files (retargeted trajectories in, metrics/videos out), not imports.
- **WSL2 only as fallback**, per baseline, if a public HMR method (GVHMR, WHAM, TRAM) does not install on Windows (Linux-only CUDA extensions). Log every such case in `docs/pivot_log.md`.
- DDP on Windows uses the `gloo` backend (no NCCL); multi-GPU NCCL runs happen on Kaggle (Linux). `torch.compile` is optional on Windows.
- **Package pins (done):** conda/package pins recorded in `env/versions.lock` (see rule 3).

## 3. Work packages

### Pose and motion models

**3D pose model — 3D pose, dense mesh, kinematic tracking** *(scope: 3D human pose estimation, dense full-body mesh recovery and kinematic tracking; vision models trained from scratch)*
- **HR-HMR:** ViT-S/16 encoder, random init, on person crops → transformer decoder → SMPL-X (6D joint rotations, 16 betas, camera) → full 10,475-vertex mesh.
- **Tracking:** temporal transformer over per-frame tokens for video; simple IoU/ID tracker for multiple people.
- **Data:** BEDLAM 2 tars + locked-head labels. Handle rotated `closeup` images. Split by subject.
- **Baseline:** BEDLAM-CLIFF checkpoint (pretrained HRNet backbone, labeled as such), same held-out frames, compared on vertices/joints.
- **Evidence:** `results/pose_model/` — MPJPE, PA-MPJPE, PVE vs baseline; training curves; overlay images. State honestly that HR-HMR is trained on 2 scenes from random init.
- *Fallback:* ResNet-18-sized CNN encoder from scratch.

**Distributed temporal training — multi-view, temporal refinement model** *(scope: multi-view and temporal architectures on multiple GPUs and multi-modal data; PyTorch scaling)*
- **HR-Refine:** spatio-temporal transformer that takes noisy SMPL-X sequences + 2D keypoints from 1–4 **virtual cameras** (incl. a head-mounted egocentric camera) + optional synthetic IMU, and outputs clean world-frame motion.
- Training data: AMASS, corrupted on the fly (jitter, occlusion masks, dropout, foot sinking).
- One code path for 1→N GPUs: `torchrun` + DDP (FSDP option), bf16, checkpoint/resume.
- Multi-GPU evidence: DDP equivalence test (2 processes vs 1, same result) + one Kaggle 2×T4 run (1 vs 2 GPU throughput).
- **Evidence:** `results/refinement_scaling/` — scaling table; 1/2/4-view ablation.

**Physics-loss study — biomechanical and physical losses** *(scope: loss functions for biomechanical constraints, temporal smoothness, postural balance and physical plausibility)*
- Losses, each switchable: foot skating (contact-weighted), ground penetration, bone-length consistency, anatomical ROM, CoM-in-support-polygon (balance), gravity/momentum consistency in flight, acceleration/jerk smoothness.
- Each loss has a unit test: 0 on valid motion, > 0 with correct gradient sign on violating motion.
- **Evidence:** `results/physics_loss_ablation/ablation.csv` — full vs reconstruction-only vs minus each loss; physical metrics + PA-MPJPE cost.

### Human-scene interaction and complex motion

**Robustness — motion blur, self-occlusion, multi-person crowding** *(scope: dynamic scene understanding)*
- HR-HMR augmentations: motion-blur kernels, synthetic occluders, crop truncation; the multi-person BEDLAM tar (orbit_bigOffice, 3 people) for crowding.
- HR-Refine robustness: occlusion-rate and jitter sweeps.
- **Evidence:** `results/robustness/` — error vs blur level, occlusion level, number of people.

**Contact, torques and support surfaces — allocentric and egocentric tracking** *(scope: tracking bodies through complex spaces; foot-ground contact, joint torques and environmental affordances)*
- Allocentric = fixed/orbit cameras; egocentric = virtual head-mounted camera in HR-Refine (same model, ego-only ablation).
- **Contact:** per-vertex contact head (labels from clean AMASS: height + velocity thresholds); report foot-contact F1.
- **Joint torques:** inverse dynamics on SMPL-X with segment masses, computed with **NVIDIA Newton** (default); a simple recursive Newton–Euler implementation is the fallback; optional OpenSim cross-check.
- **Affordances (scoped):** support-surface estimation — classify **stand / sit / lean** plus **support height** from foot–body contacts. BABEL action labels (e.g. sit, lean) are a **proxy** for support surfaces; report **agreement with BABEL action labels** (affordance F1), not ground-truth surface accuracy. Real scene meshes (PROX / EgoBody) remain item **20b** / paper *(later)*.
- **Evidence:** `results/contact_torques/` — contact F1, torque plots raw vs refined, ego-only vs allocentric error.
- *(later)* PROX / EgoBody for real scene meshes and real egocentric video.

**Motion-prior regularization — action-conditioned humanoid models** *(scope: regularization for downstream action-conditioned humanoid models)*
- Small action-conditioned motion prior on AMASS + BABEL labels, two variants: plain vs regularized with HumanoidReady signals (contact, torque bounds, balance).
- Compare generated motion on physical metrics and on G1 (robot pipeline).
- **Evidence:** `results/motion_prior/`.

### Perception prototyping

**Auxiliary heads and sensor fusion — action segmentation, intent prediction, sensor integration** *(scope: action segmentation, intent prediction and sensor integration)*
- Action segmentation head on HR-Refine (BABEL frame labels): frame accuracy, edit score.
- Intent head: predict next 0.5–1.0 s of root + pose: ADE/FDE.
- Sensor integration: synthetic body-worn IMU from AMASS fused with video keypoints: video-only vs IMU-only vs fused.
- **Evidence:** `results/auxiliary_heads/`.

**Data-pipeline bottlenecks — problem solving in the data engine** *(scope: resolving data-pipeline bottlenecks)*
- Profile the AMASS/BEDLAM loader, fix the top bottleneck, report before/after throughput.
- Keep `docs/pivot_log.md` with dated problem → decision → outcome entries.
- **Evidence:** `results/data_pipeline/bottleneck.md`, `docs/pivot_log.md`.

### Contact and physics-aware tracking

**Robot pipeline — human motion to robot control** *(scope: contact surfaces, gravity and momentum, linking human video to robot control and locomotion)*
- Retarget SMPL-X → Unitree G1 (verified open-source retargeter or IK). Track in **Isaac Lab + Newton** with PD + feedforward.
- Metrics: time-to-fall, tracking error, foot slip, torque saturation, assist wrench (force needed to keep the robot on the reference).
- Main result: raw vs HR-Refine-cleaned motion on G1; and correlation between pose error and robot feasibility (*pose accuracy ≠ physical usability*).
- Physics QA flag per clip (PASS / REVIEW / FAIL with reason) from the physical metrics.
- **Evidence:** `results/robot_comparison/` — table + rollout videos.
- *Optional differentiator (thesis link):* same clips in PhysX and MuJoCo to check engine dependence.
- *Fallback:* standalone Newton with G1 MJCF.

---

## 4. Capabilities → evidence

| Capability | Evidence |
|---|---|
| Deep learning, 3D CV, articulated tracking | 3D pose model, distributed temporal training, robustness. **Scope:** HR-Refine **egocentric camera is virtual** until EgoBody eval (item 20b); **body-worn IMU is synthetic** (AMASS-derived, not hardware). |
| Training large-scale vision models from scratch | 3D pose model (random-init ViT) + distributed temporal training (DDP/multi-GPU path). **Scope:** **HR-HMR trained on 2 BEDLAM scenes** from random init (not full BEDLAM). |
| SMPL / SMPL-X / GHUM / MHR / SOMA-X, IK, dense mesh | SMPL-X throughout; **SOMA-X pivot** SMPL-X ↔ MHR (item 17); IK in retargeting; dense 10,475-vertex mesh; `docs/body_models.md` comparing models (**GHUM documented only** — not supported by SOMA-X). |
| PyTorch and scaling frameworks | DDP/FSDP, bf16, `torch.compile`, checkpoint/resume (distributed temporal training). **Scope:** DDP path verified on **1 GPU (Windows gloo) + Kaggle 2×T4**, not a large NCCL cluster. |
| Multi-TB image/video data | Streaming shard pipeline + rolling-window processing (download → process → delete), measured throughput. **Scope:** multi-TB handled as a **measured streaming pipeline**, not a multi-TB training run *(full mirror run later)*. |
| Fast-paced, shifting priorities | `docs/pivot_log.md` |
| First-author paper | `paper/` draft from these results, working title "Pose Accuracy Is Not Physical Usability" |
| AMASS, Human3.6M, EgoBody, PROX and their optimization challenges | AMASS used throughout; `docs/dataset_challenges.md` (SMPL-X version mismatch, frame rates, GT artifacts in AMASS, BEDLAM rotated images, motion leakage between AMASS and BEDLAM). *(later)* Human3.6M, EgoBody, PROX evaluation (item **20b**) |

---

## 5. Repository layout

```
humanoid-ready/
  AGENTS.md  README.md  pyproject.toml
  configs/   (paths.example.yaml, one yaml per experiment; index in configs/README.md)
  hready/            (Python package)
    body/      (smplx wrapper, rotations, soma-x wrapper)
    data/      (amass, babel, bedlam loaders; corruption; synthetic IMU; shards)
    losses/    (physics + biomechanics losses)
    metrics/   (pose, physical, contact, stats/bootstrap)
    models/    (hr_hmr, hr_refine, heads)
    train/     (DDP trainer)
    experiments/ (oracle completion baselines and evidence-locked completion CLIs)
    baselines/ (flat-floor heuristic completion)
    robot/     (retarget, isaaclab_newton tracker, metrics)
    dynamics/  (inverse dynamics / torques)
  scripts/   tests/   results/   docs/   paper/
```

---

## 6. Paper track — CVPR 2027 (registration Nov 10, submission Nov 16, 2026, AoE)

> **ON HOLD (2026-10-09).** Paper work is paused; the egocentric motion track checklist in §8 drives the work. This section and its checklist are kept for history. The paper direction will be decided after the literature check listed in §8 (pending decisions).

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

**Rules:** thresholds and analysis plan pre-registered in `paper/claims_map.md` before the cross-engine runs; negative or mixed findings are reported as found.

## 7. Checklist (work in this order)

| # | Item | Work package | Data | Done when | ☐/☑ |
|---|---|---|---|---|---|
| 1 | Repo skeleton, `pyproject`, `.gitignore`, CPU CI, `paths.example.yaml` | — | none | CI green | ☑ evidence: `.github/workflows/ci.yml`, CI run on the repository-setup commit |
| 2 | Rotations + SMPL-X wrapper (locked_head; v1_1 only for baseline; mixing raises error) | Body models | SMPL-X | checks pass; raw output shown, check files not saved | ☑ evidence: `hready/body/rotations.py`, `hready/body/smplx_wrapper.py` |
| 3 | Physics/biomech losses | Physics-loss study | none | each loss checked analytically (0 on valid motion, >0 with correct gradient sign on violation); raw output shown, check files not saved | ☑ evidence: `hready/losses/physics.py`, `hready/losses/biomech.py`, `hready/losses/__init__.py` |
| 4 | Metrics + bootstrap CI | all | none | checked against hand-computed values; raw output shown, check files not saved | ☑ evidence: `hready/metrics/pose.py`, `hready/metrics/physical.py`, `hready/metrics/contact.py`, `hready/metrics/stats.py`, `hready/metrics/__init__.py`, `hready/losses/_constants.py` |
| 5 | AMASS + BABEL loader (30 fps, Z-up, floor z=0), contact labels, synthetic IMU | Distributed temporal training, auxiliary heads | AMASS, BABEL | 3 clips visually checked | ☑ evidence: `hready/data/amass.py`, `hready/data/babel.py`, `hready/data/contact.py`, `hready/data/imu.py`, `hready/data/foot_height_rise.py`, `scripts/inspect_clip.py`, `scripts/babel_gait_stats.py`, `results/checks/` (CMU/132/132_35, ACCAD C20 run_to_jump, BMLrub treadmill), `docs/dataset_challenges.md`, `docs/pivot_log.md`; contact core vs the earlier `contact.py` reference version on 300 `foot_traj` clips (seed 0) |
| 6 | **G1 smoke test:** one AMASS walk → G1 in Isaac Lab + Newton, metrics + video | Robot pipeline | AMASS | video + JSON | ☑ evidence: `hready/robot/isaaclab_newton.py`, `hready/robot/run_smoke.py`, `hready/robot/metrics.py`, `hready/robot/replay.py`, `results/robot_smoke_test/summary.json`, `results/robot_smoke_test/*/metrics_*.json`, `results/robot_smoke_test/*/replay_*.mp4`, `docs/decisions.md`, `docs/pivot_log.md`; visual sign-off CMU walk / ACCAD jump / BMLrub treadmill (robot smoke-test sign-off commit) |
| 7 | HR-Refine model + corruption + virtual cams (incl. ego) + DDP trainer — **candidate learned instance of kinematic refinement** (not the egocentric-track main path) | Distributed temporal training, contact and torques | AMASS | overfits 1 batch; resume works | ☑ evidence: `hready/models/hr_refine.py`, `hready/train/`, `configs/hr_refine.yaml`, `docs/decisions.md` |
| 8 | HR-Refine training + item-3 loss ablation (supporting evidence for kinematic-refinement losses) | Physics-loss study | AMASS | `results/physics_loss_ablation/ablation.csv` | ☐ |
| 9 | Contact, action, intent heads; IMU fusion ablation — contact overlaps evidence-locked completion; action/intent/IMU supporting (auxiliary heads) | Contact and torques, auxiliary heads | AMASS, BABEL | `results/contact_torques`, `results/auxiliary_heads`; **affordance F1** (agreement with BABEL action labels) | ☐ |
| 10 | Joint torques (inverse dynamics) on reconstructed body — supporting metric | Contact and torques | AMASS | torque plots | ☐ |
| 11 | G1 on raw vs refined clips; pose-error vs feasibility; QA flag (**downstream**, not the egocentric motion track) | Robot pipeline | AMASS | `results/robot_comparison/`; end-to-end: one real video (**3DPW** test sequence) → HMR → HR-Refine → G1 rollout video in `results/robot_comparison/end_to_end/` | ☐ |
| 12 | **Perception pretraining (own perception model):** BEDLAM loader (2 tars, rotated closeups) + HR-HMR from scratch + augmentations — prerequisite of perception training | 3D pose model, robustness | BEDLAM | training curves, overlays | ☐ |
| 13 | **Perception pretraining (own perception model):** BEDLAM-CLIFF baseline on same frames; robustness by blur/occlusion/people — prerequisite of perception training | 3D pose model, robustness | BEDLAM + ckpt | `results/pose_model`, `results/robustness`; **tracking:** ID-switch count and jitter on `orbit_bigOffice`, temporal transformer vs per-frame baseline | ☐ |
| 14 | Action-conditioned prior ± physics regularization, evaluated on G1 (**downstream**) | Motion-prior regularization | AMASS, BABEL | `results/motion_prior/` | ☐ |
| 15 | DDP equivalence test + Kaggle 2×T4 run | Distributed temporal training | AMASS | scaling table | ☐ |
| 16 | Loader bottleneck before/after; shard + rolling-window pipeline | Data-pipeline bottlenecks, large-scale data | AMASS/BEDLAM | `results/data_pipeline/` | ☐ |
| 17 | SOMA-X pivot SMPL-X ↔ MHR + `body_models.md` | Body models | SMPL-X | round-trip error reported. **SMPL-X ↔ MHR** goes through the **SOMA-X pivot** (`py-soma-x` tools convert **to** SOMA; no direct SMPL-X↔MHR API documented). Implement in a **separate conda env `hready-soma`** (chumpy conflicts with NumPy 2.x in `hready`); API verified from the installed package (rule 3). **GHUM** in `docs/body_models.md` only (not supported by SOMA-X). | ☐ |
| 18 | README with §4 table linked to evidence; `dataset_challenges.md`; `pivot_log.md` | all | — | every number traced to `results/` | ☐ |
| 19 | Paper draft + slides | Publication | — | compiles | ☐ |
| 20a | File access requests for Human3.6M, EgoBody and PROX *(no downloads yet)* | External datasets | — | requests filed | ☐ |
| 20b | *(later)* Human3.6M / PROX eval on a small subset; multi-TB run; **EgoBody → real egocentric validation** | External datasets | needs download | — | ☐ |

### Paper track checklist (interleaved with the items above; see the week plan)

| Task | Item | Done when | ☐/☑ |
|---|---|---|---|
| Literature review | Read BeyondRetarget, Measuring Physical Plausibility, PolySim, GMR, PHUMA, PhysHMR in full; `paper/related_work.md` | each paper: setup, metrics, engines, overlap with us | ☐ |
| Tracking-policy selection | Choose the G1 tracking policy that runs in Newton, PhysX and MuJoCo; record in `docs/decisions.md` | runs one AMASS clip in all three | ☐ |
| External video data | Download 3DPW (and EMDB if available) | loader test passes | ☐ |
| Method runs | Run GVHMR, WHAM, TRAM, BEDLAM-CLIFF on the test videos; convert to SMPL-X locked_head | per-method outputs + MPJPE/PA-MPJPE match published numbers within tolerance | ☐ |
| Analysis plan | Pre-register analysis plan + thresholds in `paper/claims_map.md` | dated commit before the cross-engine comparison | ☐ |
| Newton runs | Executability in Newton for all methods + AMASS GT upper bound; method- and clip-level analysis | `results/paper/newton/` | ☐ |
| Cross-engine comparison | Same in PhysX and MuJoCo; configuration-fairness record; Kendall τ, PASS/FAIL agreement, repeatability | `results/paper/cross_engine/` | ☐ |
| Refinement effect | HR-Refine on all method outputs; effect on executability per engine | `results/paper/refine/`, `results/robot_comparison/end_to_end/` | ☐ |
| Draft | Draft: intro, related work, method, experiments, limitations; every number from `results/paper/` | compiles in CVPR template | ☐ |
| Submission | Register abstract (Nov 10) and submit (Nov 16); supplementary (Nov 23) | submitted | ☐ |

### Week plan to the CVPR deadline

| Week | Dates | Checklist items | Paper tasks |
|---|---|---|---|
| 1 | Oct 1–8 | environment setup, 1–4 | literature review, tracking-policy selection, external video data |
| 2 | Oct 9–15 | 5, 6 | method runs |
| 3 | Oct 16–22 | 10 | analysis plan, Newton runs |
| 4 | Oct 23–29 | — | cross-engine comparison |
| W5 | Oct 30–Nov 5 | 7, 8 | refinement effect, draft |
| 6 | Nov 6–16 | — | draft (final), submission |
| after | Nov 17 → | 9, 11–19 | camera-ready / workshop / arXiv |


**Minimum milestone:** items 1–6 (and 7–8 if time). Items 1–11 cover the physics, loss, contact,
torque and robot work; 12–13 cover perception pretraining (prerequisites of perception training); 14–19 complete the rest. The egocentric motion track main path is `docs/project_definition.md` (APPROVED 2026-10-04); item 8 ablation paused on `wip/fast-loader`.

## 8. Egocentric motion track — egocentric RGB to physics-refined motion (main path)

**Status:** active (Oct 4, 2026). **No G1** on this path (items 6/11 remain downstream).

**Decision record (2026-10-09): switch from oracle evidence to realistic observations.** The oracle completion baselines and evidence-locked completion were developed and judged on oracle evidence (visible-joint 3D positions with small Gaussian noise, exact head pose, gravity and floor height). That path stays as a labelled upper-bound control, as defined in `docs/project_definition.md` (observation simulator; oracle evidence as a control). The main path moves to realistic observations, as already specified by the perception-variants, egocentric-data and perception-matched-training decisions: 2D keypoints with confidence and visibility, head pose from SLAM-like tracking, gravity, and floor height estimated rather than given. Evidence-locked completion is closed as a completed oracle-evidence study (see its row below). No acceptance rule of a completed experiment is changed.

**Authoritative spec:** `docs/project_definition.md` (four binding decisions, stage contracts, evidence schema, controls, track order). Summary chain:

```
egocentric RGB/video (+ given head pose + gravity, first system version)
  -> visual perception (egocentric perception stage)
  -> evidence schema (defined by the observation simulator; oracle + perceived share it)
  -> completion (oracle completion baselines, evidence-locked completion)
  -> contact + ground (foot + support-contact region channels)
  -> refinement: kinematic, then simulation (refinement comparison arms: none, kinematic, simulation, both)
  -> end-to-end evaluation (+ real egocentric validation when scheduled)
```

- **Given head pose and gravity:** metric head/camera pose and gravity are **given** in the first system version (rig); stated on every results table.
- **Perception variants:** own perception model (our HR-HMR-style, random init + ego adapt), pretrained perception baseline (labeled pretrained), adapted pretrained backbone (if needed); weak HR-HMR does not block the pipeline — **perception model selection** selects the main-path model.
- **Egocentric data plan:** synthetic egocentric RGB from AMASS first; **real egocentric validation** is an explicit step, not indefinite later.
- **Two-stage physics refinement:** **kinematic** refinement (contact-aware, item-3 losses; HR-Refine a candidate learned instance) then **simulation** refinement (SMPL-X humanoid simulation tracking, Newton only).

**Controls:** the **observation simulator** oracle path (AMASS → observation simulator → schema → …) is a **control**, not the product interface. The **oracle completion baselines** run oracle mode first for diagnosis only; the perceived RGB path is the main end-to-end system. **End-to-end evaluation** main rows = perceived; oracle rows = control; perception-only row + decomposition.

**Evidence schema (observation simulator):** head 6-DoF; visible-joint 3D in world frame; per-joint visibility mask and confidence; optional 2D keypoints from virtual head camera. Completion consumes **only** schema tensors (+ given rig signals alongside the evidence).

**Egocentric perception stage (critical path):** **egocentric render set**; **perception baseline measurement** (~50 frames then VAL, not a go/no-go gate); **perception training** of the own model, the pretrained baseline and the adapted backbone (items **12–13** prerequisites for the own model); **perception model selection** on VAL (criterion pre-registered before TEST), emit schema for splits.

**Simulation refinement (humanoid asset, tracking training, refinement comparison):** SMPL-X-skeleton humanoid asset generated in-repo; PPO DeepMimic-style tracking in Newton; the **refinement comparison** compares none vs kinematic vs simulation vs both on reconstructed motion. Reuses `hready/robot/` Newton runner patterns from item 6 but **not** G1/GMR.

**Floor-work cohorts (recorded with the subject splits):** evaluable floor-work metrics = **kneel + lie** only (`floor_work_eligible`); **crawl** and **yoga_like** have **0** confirmed segments; **sit_floor** / **sit_support** reported separately (sit_floor **not** eligible). Support-contact region labels are GT-derived training targets; at test time contact is predicted from perceived/completed evidence.

**Completion training direction (perception-matched training evidence; open, prompts for the completion items):** final training targets evidence whose error distribution approximates perception (calibrated from held-out perception errors; oracle evidence is a control, not the final training distribution). See `docs/project_definition.md` §8.

**Fixed constraints.** Single local GPU (RTX 4090); no cloud; **no new downloads** until the real-egocentric-validation decision. Newton only (rule 0). Checkpoints/logs under `hready_data`. No multi-GPU / multi-TB claims beyond items 15–16 measured paths.

**Reuse (do not delete working modules):** items 2–5 loaders/contact/metrics/losses; item 7 virtual-camera/IMU code for controls; item 6 Newton orchestration patterns for simulation refinement only.

### Egocentric motion track checklist (work in order; one item at a time)
| Item | Description | Done when | ☐/☑ |
|---|---|---|---|
| Module audit | Audit: what exists, what is reused, what conflicts | written, reviewed by user | ☑ evidence: `docs/module_audit.md` |
| Subject splits and cohorts | Subject split + floor-work subset (kneel/lie eligible; crawl/yoga 0 confirmed; sit_floor separate) | reproducible, documented | ☑ evidence: `results/splits/splits.json`, `results/splits/cohort_counts.json`, `results/splits/floor_work_clips.csv`, `docs/ego_splits.md` |
| Support-contact labels | Support-region contact labels (foot channel unchanged) | recorded thresholds + rates JSON | ☑ evidence: `docs/support_contact_labels.md` (support-contact labels commit) |
| Observation simulator (oracle evidence) | Oracle-evidence control + evidence schema + leak-proof observation simulator | `docs/ego_observation_model.md` + oracle path config | ☑ evidence: `docs/ego_observation_model.md`, `configs/ego_observation.yaml`, `hready/data/ego_observation.py`, `hready/models/ego_completion.py`, `hready/body/joint_indices.py` |
| Realistic observation model | Realistic observation model: 2D keypoints + confidence + visibility, head pose from SLAM-like tracking, estimated floor height; noise/visibility model calibrated from measured detector errors (synthetic first); uncertainty-weighted evidence lock; leak check for the new schema; design note written and dated before any training | design doc + leak check + pre-registration (primary arm, decision set, seeds, noise-derived margins) | ☐ |
| Egocentric render set | Ego render set (virtual head camera; subject splits; camera/appearance spec recorded) | render manifest + spec doc | ☐ |
| Perception baseline measurement | Perception baseline measurement (pretrained zero-shot; ~50 frames then VAL) | measurement table (not a gate) | ☐ |
| Perception training | Train/adapt perception: own model / pretrained baseline / adapted backbone (items 12–13 for the own model) | checkpoints under hready_data | ☐ |
| Perception model selection | Select main-path model on VAL; emit schema evidence train/val/test | selection record + emitted schema | ☐ |
| Completion baselines | Completion baselines (heuristic, regression) — **oracle mode first** (control), then perceived after perception model selection | `results/completion/` tables (oracle labeled control) | ☑ oracle mode, evidence: `results/completion/oracle_baselines.json` (VAL+TEST tables, heuristic check, leak check; oracle-baselines results commit), `docs/oracle_completion_baselines.md`. Perceived mode after perception model selection: ☐ |
| Evidence-locked completion | Generative prior on schema (visibility + confidence); trains on perceived evidence | logs/ckpts under hready_data | CLOSED as an oracle-evidence study (2026-10-09). Ablation study: acceptance not met, `docs/evidence_locked_ablation.md`. Pre-registered comparison (pre-registered in the comparison's pre-registration commit): no arm passed the pre-registered rule (deterministic `det_w0` passed 4 of 5 gates and missed the contact-ECE gate; TEST not evaluated), recorded in `results/completion/evidence_locked_preregistered_comparison.json`, `docs/evidence_locked_preregistered_comparison.md`. Training on realistic evidence continues under the realistic observation model / decoupled contact grounding. ☐ |
| Decoupled contact grounding | Decoupled grounding: accuracy-trained completion + contact-conditioned minimal-displacement foot-ground stage (candidate kinematic refinement, two-stage physics refinement) + calibrated contact probabilities; contact metrics computed on the predicted mesh with the same definition as the labels; calibration metric and margin fixed from measured noise before training | dated pre-registration, then `results/contact_grounding/` | ☐ |
| End-to-end evaluation | Eval command: perceived rows main; oracle control; perception-only row; decomposition; floor-work slice (kneel+lie) | `results/end_to_end/` + reliability plot | ☐ |
| Real egocentric validation | Real egocentric validation (dataset decision + download exception; data access and a 2D-detector pilot may start in parallel with the realistic observation model, after the owner's decision) | planned eval under `results/real_egocentric/` | ☐ |
| Humanoid simulation asset | SMPL-X-skeleton humanoid asset; Newton acceptance (static pose ≥5 s) | `results/humanoid_asset/` | ☐ |
| Simulation tracking training | PPO simulation-refinement tracking trains on Newton | envs + wall-clock logged | ☐ |
| Refinement comparison | Refinement arms none / kinematic / simulation / both on reconstructed motion | `results/refinement_comparison/` | ☐ |
| Qualitative review | Qualitative: ego RGB, perceived evidence, GT vs completion vs physics; ≥1 kneel/lie clip | local mp4 (not in repo) | ☐ |
| Claims map | Claims: perception vs completion vs physics separate; oracle vs perceived separate; perception-variant labeling | claims map with commands | ☐ |

**Pending decisions (2026-10-09).** (1) Literature check outcome: closest prior work and what remains different. (2) Real egocentric dataset choice, license, size and the download exception (§1 download pause stays until then). (3) Calibration metric for gating (Brier score vs debiased ECE) and how its margin is derived from measured noise. (4) Decision set: the 19 untouched tune subjects vs more data; AMASS VAL is a development set only. (5) Training-seed policy for compared arms. (6) Paper direction (§6 on hold).

**Integrity rules (apply to every egocentric-track item).** Existing modules, commands and results keep working exactly as before (re-run the regression of §7 items 2–5; item 6 when robot code changes). No fabricated, estimated or placeholder numbers; a result not run is "not run". Negative or weak results are reported as found. Scratch checks follow rule 5. No large data, checkpoints or renders in the repo. All seeds fixed and recorded.
