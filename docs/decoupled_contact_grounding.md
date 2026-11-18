# Decoupled contact grounding: design and pre-registration

**Date:** 2026-10-13. **Status:** pre-registered before any calibration fit, training run or evaluation of this study.
Oracle **control** path (same evidence, splits, harness and metrics as `docs/evidence_locked_ablation.md`); every
result row carries the disclaimer: oracle control; floor height given (world z = 0); head pose and gravity given (rig input).

## Why a decoupled stage

Two attempts to improve foot behaviour inside the completion objective did not meet their acceptance rules
(`docs/evidence_locked_ablation.md`, `docs/evidence_locked_preregistered_comparison.md`):

- Ground-aware losses added to the training objective (`phys` arm, `w_phys = 1`) lowered penetration and skate but cost
  +1.9 mm full MPJPE against the reference arm and raised contact ECE (0.077 against 0.013 for `oracle_transformer`);
  it was excluded by the pre-registered +1.0 mm rule.
- The generative reference arm (`evidence_locked_same_budget`) has VAL skate 0.235 m/s against 0.202 for
  `oracle_transformer`, and contact ECE 0.101 against 0.013. In the pre-registered comparison the deterministic arm
  `det_w0` failed the contact-ECE gate (upper bound above +5 % of the reference); no arm passed.

This study therefore leaves the completion model untouched and adds two things after it: a **temperature** that
calibrates the contact probabilities, and a small **grounding stage** that corrects foot joints only, using those
calibrated probabilities.

## Fixed inputs

- **Completion model:** `evidence_locked_same_budget` (ablation run `ref_locked_gen_phys0`, `best.pt`, 40000 steps),
  frozen. Its weights, training config, evidence schema and the observation simulator are not changed. Its forward
  pass, the evidence lock, window 64 / stride 32 stitching and the evaluation harness are used as they are.
- **Evaluation:** VAL `all`, all 2223 clips, the oracle-baselines metric definitions (full, visible and hidden MPJPE,
  penetration, skate, ground-consistency violation at both tolerances, contact F1, ECE). Brier score is added.
  TEST is not evaluated.
- **Reference values** (from `results/completion/evidence_locked_ablation.json`, VAL `all`): `oracle_transformer` skate
  0.2017 m/s, contact ECE 0.0130, full MPJPE 66.68 mm; `evidence_locked_same_budget` full MPJPE 64.82 mm, skate
  0.235 m/s, ECE 0.101.

## Contact calibration (temperature scaling)

- **Calibration set:** 200 clips drawn with seed 0 from the clips of the 19 tune subjects (`splits.json`, `tune`). These
  clips are in the train split but were removed from the completion training list (482 clips; see
  `train_clip_list` in `results/completion/oracle_baselines.json`), so the completion model has not seen them, and they are disjoint from VAL and
  TEST. Clips flagged `skate_flag` or `exclude_contact` are not drawn.
- **Fit:** one scalar temperature `T > 0` on the contact logits (`p = sigmoid(logit / T)`), applied to every window before
  overlap averaging, chosen by minimising the negative log-likelihood of the stitched probabilities against the item-5
  contact labels (bounded one-dimensional search over `log T`). No other parameter is fitted.
- **Report:** ECE (the existing plain 10-bin estimator) and Brier score before and after calibration, on the calibration
  set and on VAL `all`. `T` is saved to `results/completion/contact_temperature.json`.

## Grounding stage

- **Function:** a lightweight temporal convolution applied after the completion model (and its evidence lock) to the
  stitched clip. It corrects the four foot-channel joints (22-joint indices 7, 10, 8, 11) by an additive 3-D displacement
  and leaves every other joint untouched. It is joint-space only: the corrected feet are no longer exactly the forward
  kinematics of the SMPL-X parameters, which is also true of the evidence-locked joints that the tables already score.
- **Input per frame and foot channel:** sole-proxy height, horizontal velocity of the channel point, vertical velocity,
  and the calibrated contact probability; the floor is at z = 0 (rig, oracle control). Horizontal positions enter only as
  velocities, so the stage is invariant to the window's xy origin.
- **Architecture:** four dilated 1-D convolution layers (kernel 5, dilations 1, 2, 4, 1; 64 channels, receptive field 33
  frames), zero-initialised output layer so that the untrained stage is the identity (minimal displacement). The receptive
  field is wider than the 8 frames suggested in the plan because a stance phase lasts longer than 8 frames at 30 Hz.
- **Loss (weights fixed here, one configuration, no sweep):** GT-contact-weighted foot skating (the existing
  `foot_skating`, weight 5.0); ground penetration hinge (`ground_penetration`, 100.0); contact-height term, the mean
  absolute sole-proxy height on GT-contact frames (1.0); foot-joint position error to the ground truth (mean L2, 1.0);
  displacement penalty (mean |correction|, 0.1). No KL, no generative term.
- **Training data and budget:** windows of the 12309-clip training list through the frozen completion model (oracle
  evidence path, the baselines' z-rotation augmentation and sampler, seed 0); batch 64, 10000 steps, AdamW, learning
  rate 1e-3 with cosine decay to 2 %, no checkpoint selection (the checkpoint after the last step is evaluated). Divergence
  is monitored on the remaining tune-subject clips (those not in the calibration set, same exclusions) (loss every 1000 steps, never used to
  select or tune). The run is declared divergent if any loss is non-finite or the monitor loss at the end exceeds its
  value at step 0, where the stage is the identity.
- **Order of runs:** smoke test (one clip, 5 steps), temperature fit, training, one VAL evaluation. Nothing is re-run
  after the VAL evaluation.

## Acceptance criteria (pre-registered)

Evaluated on the pooled point estimate over VAL `all` (2223 clips); subject-cluster bootstrap intervals are reported but
do not decide.

1. **Primary:** foot skate of the grounded output **<= 0.202 m/s** (the `oracle_transformer` VAL value).
2. **Secondary:** contact ECE after temperature scaling **<= 0.015** (`oracle_transformer` 0.013 plus 0.002).
3. **Tertiary:** full MPJPE **<= 66.8 mm** (`evidence_locked_same_budget` 64.8 mm plus 2 mm).

Reported, not criteria: visible and hidden MPJPE, penetration, ground-consistency violation, contact F1, Brier score,
the same metrics for the uncalibrated and ungrounded model, and pre- and post-calibration ECE and Brier on both sets.

**Decision rule.** The study passes only if all three criteria hold. If the primary criterion fails, the failure is
logged in `docs/pivot_log.md` with a hypothesis and the study stops; no threshold, weight, budget or architecture is
changed after any result. If the secondary or tertiary criterion fails while the primary holds, the result is reported
as a partial failure in the same way.

## Not in this study

- No change to the completion model, its training, the evidence schema or the observation simulator.
- No retraining of the calibration of other models, no TEST evaluation, no perceived (non-oracle) evidence.
- No re-posing of SMPL-X parameters from the corrected feet.

---

# Second study: deterministic completion at the main budget

**Date:** 2026-10-13. **Status:** pre-registered before any code for this study was written and before any run. The
first study above is closed as a partial failure (contact ECE 0.0420 against 0.015); nothing in it is changed.

## Question

The deterministic ablation arm (latent fixed at 0, no posterior, no KL; evidence lock on; `w_phys = 0`), trained for
40000 steps, beat `oracle_transformer` on VAL `all` for full MPJPE (60.0 vs 66.7 mm), hidden MPJPE (64.8 vs 67.4),
penetration (4.43 vs 5.98 mm), skate (0.182 vs 0.202 m/s) and tied on contact ECE (0.0126 vs 0.0130). It lost only on
ground-consistency violation (0.432 vs 0.426). The generative arm of the same design, trained for 150000 steps with the
cosine schedule, diverged (VAL selection metric 66.1 mm at step 25000, 112.6 at 30000, 185.0 at 150000). The question
is whether the deterministic arm holds or diverges at the 150000-step budget.

## Run (fixed before any result)

- Model, evidence lock, loss weights (`w_phys = 0`), optimiser, batch size 64, learning rate 4e-4, weight decay 0.01,
  warmup 2000, gradient clip 1.0, bf16, windows 64 / stride 32, training list (12309 clips), sampler, z-rotation
  augmentation, cached-occlusion evidence and seed 0: all identical to the 40000-step deterministic ablation arm
  (`configs/evidence_locked_ablation.yaml`, which is read, not edited).
- Only changes: `max_steps = 150000` and the cosine schedule, which decays to 2% of the peak learning rate over the
  whole run (the schedule the generative main run used).
- VAL-only checkpoint selection every 5000 steps: best mean pooled MPJPE over VAL `all` + `ordinary_locomotion`
  (the 200 seeded VAL clips of the existing protocol). The best checkpoint is evaluated once on all 2223 VAL `all`
  clips. TEST is not evaluated.
- No change to the evidence schema, the observation simulator, the model code, or the grounding stage and calibration of
  the first study (not used here).
- Reference rows evaluated in the same harness on the same clips: `oracle_transformer`, and the 40000-step
  deterministic checkpoint (labelled as the shorter budget).

## Acceptance criteria (VAL `all`, pooled point estimate)

1. **Primary:** foot skate **<= 0.202 m/s** (`oracle_transformer` VAL).
2. **Secondary:** full MPJPE **<= 66.7 mm** (`oracle_transformer` VAL).
3. **Tertiary:** no divergence: the VAL selection metric is **<= 150 mm at every checkpoint** (all 30 validations,
   steps 5000 to 150000).

Reported, not criteria: visible and hidden MPJPE, penetration, ground-consistency violation, contact F1, ECE, Brier,
the selection-metric history, and which step was selected.

**Decision rule.** If the primary criterion fails, the failure is logged in `docs/pivot_log.md` with a hypothesis and
the study stops; nothing is committed. No threshold, budget, seed or architecture changes after any result. A
deterministic arm that passes does not show that the divergence of the generative run was caused by its KL term: the
generative run also differed in its latent, and no run isolates the learning-rate schedule.

Results: `results/completion/deterministic_main.json`. Command: `python -m hready.experiments.deterministic_main`
(`configs/deterministic_main.yaml`), subcommands `train` and `eval`.
