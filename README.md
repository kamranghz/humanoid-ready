# HumanoidReady

*From egocentric human video to physically plausible full-body motion.*

HumanoidReady is a research codebase for recovering full-body 3D human motion (SMPL-X) from what a head-worn camera can see, and for checking that the recovered motion is physically plausible. The intended pipeline is: egocentric RGB → visual perception → partial 3D body evidence → full-body completion → foot-contact and ground estimation → physics-aware refinement → evaluation. The work so far covers the data and evaluation infrastructure, a simulator that produces the "partial evidence" a head-worn camera would provide, and completion models evaluated under that simulated evidence. Every number in this README comes from a result file in `results/` produced by a script in this repository.

## Status

| Component | Status | Evidence |
|---|---|---|
| SMPL-X body model wrapper, rotations | Built | `hready/body/` |
| Physics and biomechanics losses (foot skating, penetration, balance, smoothness, bone length, range of motion) | Built | `hready/losses/` |
| Pose, physical and contact metrics; subject-cluster bootstrap statistics | Built | `hready/metrics/` |
| AMASS + BABEL loaders, 30 Hz resampling, floor grounding, foot-contact labels, synthetic IMU | Built | `hready/data/`, `results/checks/` |
| Subject-disjoint splits and motion cohorts (locomotion, kneel/lie floor work, sitting) | Built | `results/splits/splits.json`, `results/splits/cohort_counts.json`, `docs/ego_splits.md` |
| Support-region contact labels | Built | `results/support_contact/label_rates.json`, `docs/support_contact_labels.md` |
| Simulated egocentric evidence (head camera, visibility, occlusion, noise) with leak checks | Built | `hready/data/ego_observation.py`, `docs/ego_observation_model.md` |
| Full-body completion baselines under simulated evidence (heuristic + learned) | Built | `results/completion/oracle_baselines.json`, `docs/oracle_completion_baselines.md` |
| Humanoid-robot smoke test (retarget one motion to a Unitree G1 in Isaac Lab + Newton) | Built | `results/robot_smoke_test/` |
| Spatio-temporal refinement model and trainer (DDP code path implemented, not yet tested on multiple GPUs) | Building (code written; no reported result) | `hready/models/hr_refine.py`, `hready/train/` |
| Improved completion model (evidence-locked; generative and deterministic variants) | Building | ablation study failed acceptance; pre-registered comparison failed its pre-registered rule (below) |
| Visual perception from egocentric RGB | Planned | — |
| Evaluation on perceived (not simulated) evidence | Planned | — |
| Physics-aware refinement (kinematic, then simulation tracking) | Planned | — |
| Validation on real egocentric recordings | Planned | — |

Built = code on `main` with evidence in `results/` and CI green. Building = started, not measured. Planned = not started.

## Headline results

All results below are an **oracle control**: the body evidence is simulated from motion-capture ground truth (no image perception), the floor height is given (world z = 0), and the head pose and gravity are given. They measure completion error with perception error removed, not end-to-end accuracy.

**Full-body completion baselines** (`results/completion/oracle_baselines.json`; validation split, 2223 clips; test split, 1272 clips):

| | Full MPJPE (mm) | Visible joints | Hidden joints | Penetration (mm) | Foot skate (m/s) | Ground-consistency violation |
|---|---|---|---|---|---|---|
| Heuristic (rest-skeleton offsets) — val | 141.3 | 20.3 | 153.5 | 0.37 | 0.605 | 0.073 |
| Learned (temporal Transformer) — val | 66.7 | 59.5 | 67.4 | 5.98 | 0.202 | 0.426 |
| Ground truth (metric reference) — val | 0.0 | 0.0 | 0.0 | 0.00 | 0.027 | 0.169 |
| Learned — test | 64.6 | 59.4 | 65.2 | 16.88 | 0.167 | 0.479 |

- The learned model roughly halves the heuristic's full-body error (66.7 vs 141.3 mm on validation) but does not reproduce the visible joints it is given (59.5 vs 20.3 mm), and it is less physically plausible than the ground truth on penetration, foot skate and ground consistency.
- The learned model's best checkpoint was its last (step 40000 of 40000), so it is likely under-trained.
- Floor-work results (kneel and lie) rest on 11 validation and 2 test clips and are indicative only. The ground-consistency metric is unreliable there: the ground truth itself scores 0.884.

**Improved completion model, attempt 1 — did not meet its acceptance criteria** (`results/completion/evidence_locked_ablation.json`, `docs/evidence_locked_ablation.md`). An evidence-locked conditional VAE trained for the same 40000-step budget improved validation full MPJPE (64.8 mm) and visible-joint MPJPE (12.2 mm) over the learned baseline, but had worse foot skate (0.235 vs 0.202 m/s) and worse hidden-joint MPJPE (70.1 vs 67.4 mm). Its longer 150000-step run diverged; the reported checkpoint is from step 25000. The attempt is documented as failed.

**Evidence-locked completion, pre-registered comparison — did not pass its pre-registered rule** (`results/completion/evidence_locked_preregistered_comparison.json`, `docs/evidence_locked_preregistered_comparison.md`). Arms, hyperparameters and a statistical decision rule (paired subject-cluster bootstrap against the learned baseline) were fixed in `configs/evidence_locked_preregistered_comparison.yaml` before training. No arm passed; TEST was not evaluated. The best arm (deterministic, no physics loss) improved validation full MPJPE (60.0 vs 66.7 mm) and hidden-joint MPJPE (64.8 vs 67.4 mm) over the learned baseline but missed the contact-calibration gate. Its design was informed by the earlier ablation study's validation results, which `docs/evidence_locked_preregistered_comparison.md` states explicitly.

## Scale and limitations

- Trained on 12309 AMASS motion clips (SMPL-X, 30 Hz) on a single RTX 4090; evaluated on 2223 validation and 1272 test clips from subjects disjoint from training.
- Evidence is simulated from ground truth; no image-based perception is part of any reported number.
- Physical metrics on the 22 FK joints use a foot-sole height proxy; a ground-truth row is reported so the proxy's own error is visible.
- Floor-work cohorts are small (kneel and lie only; 8 subjects across validation and test).
- A known issue in the recorded ground-consistency tolerance is recorded in `docs/pivot_log.md` (2026-10-07); tables report both the recorded and a corrected tolerance.

## What is not claimed

- No accuracy on real egocentric video or images.
- No end-to-end system: perception, physics refinement and real-data validation are planned, not built.
- No robot control result beyond a single smoke test of retargeting one motion to a simulated humanoid.
- No claim that the improved completion model works; both the ablation study and the pre-registered comparison failed their acceptance rules.

## Repository layout

```
hready/            Python package
  body/            SMPL-X wrapper, rotations, joint tables
  data/            AMASS/BABEL loaders, contact labels, splits and cohorts, evidence simulator, caches
  losses/          physics and biomechanics losses
  metrics/         pose, physical, contact metrics; bootstrap statistics
  models/          completion and refinement models
  baselines/       heuristic completion baseline
  train/           training engines (DDP code path implemented, not yet tested on multiple GPUs)
  eval/            evaluation CLIs for the completion experiments
  robot/           humanoid retargeting and simulation smoke test
configs/           experiment configs (paths.example.yaml is the template for local paths)
docs/              design notes, data issues, decision and pivot logs
results/           result files produced by the scripts (numbers above come from here)
scripts/           small inspection utilities
```

## Reproducing

Data and body models are **not** included (licensed): SMPL-X (locked-head neutral model), AMASS (SMPL-X N), BABEL. Paths are read from `configs/paths.yaml`.

```bash
conda activate hready            # Python environment used for all project code
pip install -e ".[dev]"
cp configs/paths.example.yaml configs/paths.yaml   # then edit data_root, amass_root, cache_dir
python -c "import hready"

# Full-body completion baselines (builds a motion cache under cache_dir, trains, evaluates)
python -m hready.experiments.oracle_baselines run --config configs/oracle_completion_baselines.yaml
```

Installed package versions are recorded in `env/versions.lock`. Design decisions and their reasons are logged in `docs/decisions.md` and `docs/pivot_log.md`.

## License

The code in this repository is released under the MIT License (see `LICENSE`). The license does not cover SMPL-X, AMASS, BABEL, BEDLAM or any other third-party data, body models or pretrained models used with this code; those remain under their own licenses and must be obtained from their providers.

## Citation

If you use this code, please cite it using the metadata in `CITATION.cff`.
