# Realistic perceived evidence: design note

**Date:** 2026-10-13. **Status:** design and pre-registration only. No code, no model, no run has been made for this note.

This note defines how completion is evaluated when its evidence comes from a perception stage instead of the
ground-truth simulator, and records the acceptance criteria before any experiment runs. Numbers are taken from
`docs/ego_observation_model.md` and `docs/evidence_locked_ablation.md`; nothing here is a new measurement.

## 1. Problem statement

The completion models so far are developed and judged on **oracle evidence**: the observation simulator
(`docs/ego_observation_model.md`) takes the ground-truth SMPL-X motion, keeps the joints that a head-mounted camera
would see, adds small independent Gaussian noise (`wrist_pos_std_m` 0.012 m, `upper_body_pos_std_m` 0.008 m,
`leg_pos_std_m` 0.015 m), drops joints at random (`joint_dropout_prob` 0.05, `frame_dropout_prob` 0.02) and reports
a fixed confidence (`confidence_visible` 0.92, `confidence_hidden` 0.0). The head pose, gravity and floor height are
given. Every one of these choices favours the model, so oracle evidence is an **upper bound** on what a perception
stage can deliver:

- the noise is independent per joint and per frame, whereas a detector's errors are correlated in time and biased
  under occlusion, motion blur and unusual poses;
- visibility is computed from geometry, so the model is never given a joint that is "visible" but wrong;
- the confidence is a constant, so it carries no information about which joints are actually reliable.

**"Realistic" in this note** means evidence of the same schema (`joint_pos_3d`, `joint_visible`,
`joint_confidence`, `keypoints_2d`) produced by a 2D pose detector or a 3D lifter applied to **synthetic
egocentric renders of AMASS clips**. Real egocentric video (EgoBody) is a later, separate step and is not used here.

Two questions follow:

1. How do we obtain realistic joint estimates? Through the perception stage (render set, baseline measurement and
   perception training), which is not part of this note.
2. Before that stage exists, how much does completion quality degrade as the evidence gets worse? This note answers
   only the second question, with an evaluation-only sensitivity analysis (sections 2 to 4). It is a sensitivity
   analysis, not a prediction: synthetic Gaussian degradation is not a perception model's error distribution.

## 2. Evidence degradation sweep (evaluation only, no retraining)

**Models, unchanged and used as recorded in `docs/evidence_locked_ablation.md`:** `oracle_transformer` and
`evidence_locked_same_budget`. Reference values on VAL `all` (2223 clips) with oracle evidence:

| model | full MPJPE (mm) | visible | hidden | contact F1 | ECE |
|---|---|---|---|---|---|
| oracle_transformer | 66.7 | 59.5 | 67.4 | 0.941 | 0.013 |
| evidence_locked_same_budget | 64.8 | 12.2 | 70.1 | 0.913 | 0.101 |

**Mechanism.** Each degraded variant is produced by the same observation simulator with a modified setting, or by a
post-processing of `obs` that leaves the rig signals (head pose, gravity, floor height) untouched. The models receive
only `obs`, so the leak check of `docs/ego_observation_model.md` still applies and is rerun for each variant. The
simulator's per-clip random seeds are kept (`seed: 0`), so a variant differs from the reference only in the degradation.

**Degradations** (each applied alone; the combined case is listed last):

| id | degradation | grid |
|---|---|---|
| noise | one position noise std replaces `wrist_pos_std_m`, `upper_body_pos_std_m` and `leg_pos_std_m` for all visible joints | 0.008, 0.015, 0.025, 0.035, 0.05 m |
| mask | a random fraction of the joints that are visible in each frame is set to invisible (`joint_visible` false, `joint_pos_3d` zero, `joint_confidence` set to `confidence_hidden`, 2D confidence zero) | 20 %, 40 %, 60 % |
| torso | every torso joint (`JOINT_GROUP_TORSO`: 0, 3, 6, 9, 12, 13, 14) is set to invisible in every frame (the leg-glimpse condition, section 3) | on |
| combined | the torso degradation plus the 0.05 m noise and the 40 % mask | one setting |

The config has no single `joint_pos_std_m` key; the noise grid therefore replaces the three existing std values by one
scalar. The grid starts at 0.008 m, which is below the default leg std (0.015 m), and its first two points bracket the
defaults. The default setting is also run as the reference (it must reproduce the table above).

**Metrics** (the oracle-baselines metric definitions, unchanged): full, visible and hidden MPJPE (mm, pooled over
clips), penetration (mm), foot skate (m/s), contact F1, and the calibration metrics of section 4.

**Cohorts and size.** Primary: VAL `all`, all 2223 clips. Secondary: VAL `ordinary_locomotion`. Indicative only:
VAL `floor_work_eligible` (11 clips). `sit_floor` and `sit_support` are reported descriptively. TEST is not evaluated.

**Statistics.** Each degraded variant is compared with the same model on default evidence by paired subject-cluster
bootstrap (2000 draws, seed 0, the estimator already used in
`docs/evidence_locked_preregistered_comparison.md`); results are reported as the pooled difference with its 95 %
interval. The two models are never ranked against each other except by the criteria of section 5.

**Expected size of the effects (stated before the run, not a result).** The torso degradation changes little in
isolation, because torso visibility is already about 0.004 at `focal_px = 120`. Noise and masking are where degradation
should appear. `evidence_locked_same_budget` has a visible MPJPE of 12.2 mm against 59.5 mm for `oracle_transformer`,
which suggests it follows visible evidence closely, so its error under large noise is expected to rise more than that of
`oracle_transformer`. This is a hypothesis to be tested, not an assumption used in the criteria.

## 3. Leg-glimpse slice

**Definition.** A frame belongs to the leg-glimpse slice when **no torso joint is visible** (`JOINT_GROUP_TORSO` all
`joint_visible` false after the simulator's visibility steps), while hands, forearms, thighs, shins and feet keep their
own visibility. Hands and legs are therefore present only when they are in view and not occluded; no other joint is
forced visible. A clip belongs to the slice-level report if at least half of its frames are in the slice (the
threshold is fixed here and reported with the clip count).

**Why it is the primary realistic condition.** The head-mounted camera at the default setting sees mostly the lower
visual field (hands, forearms, legs when visible), and "torso keypoints are almost never visible at the default FOV".
The leg-glimpse slice is that case made exact.

**Coverage of the VAL set.** `docs/ego_observation_model.md` gives torso visibility of about 0.004 (cohort `all`,
`focal_px = 120`, `frame_stride` 8, 7531 frames: sit_support 4671, ordinary_locomotion 2214, floor_work_eligible 389,
sit_floor 257). That figure is a per-joint, per-frame visibility rate over those four cohorts, not the share of VAL
frames that satisfy the slice definition, and it was not computed on the 2223 VAL clips. The coverage of the slice on
VAL (frames and clips, per cohort) is therefore **to be measured**, not quoted, as the first step of the run. Because
the torso rate is so low, the measured share is expected to be high; if it is not, the criteria below are unaffected,
since they are evaluated on all VAL clips under the forced torso degradation (section 5).

**Consequence for interpretation.** Under the default simulator the evidence is already close to leg-glimpse, so the
torso degradation alone is close to the oracle reference. The slice is a realistic condition for what is visible, but
the oracle estimates inside it are still unrealistically accurate; the realism gap is in noise and bias, which the noise
and mask sweeps explore, and which the perception stage will measure.

## 4. Calibration check

The contact head's output probability is evaluated against the item-5 contact labels (`contact_f1` in the tables). The
reported ECE in the existing docs is the plain 10-bin estimator (`hready/metrics/paired_bootstrap.py`): the sum over
bins of `|sum y - sum p| / N`, pooled over frames and channels. For realistic evidence, confidence may be unreliable
and the plain ECE is biased upward when bins are sparsely populated (which degraded evidence will make worse), so three
quantities are reported for every model and every degradation:

1. **ECE** (the existing estimator, for continuity with the tables above).
2. **Brier score**: the mean squared difference between probability and label, with its reliability, resolution and
   uncertainty decomposition.
3. **Debiased ECE**: per bin, the squared gap `(mean p - mean y)^2` minus the finite-sample term
   `mean y (1 - mean y) / (n_bin - 1)`, weighted by bin size, floored at zero and square-rooted. Bins with fewer
   than 2 frames are merged into their neighbour.

Comparison: oracle evidence against each degraded variant, for both models, with the paired subject-cluster bootstrap
of section 2. Reliability diagrams (observed frequency against mean probability per bin, with bin counts) are produced
for oracle evidence, the leg-glimpse condition and the combined degradation.

**Open decision, not made here:** which calibration metric gates a later training run, and how its margin is derived
from measured noise, is a decision for the owner (pending decision 3 of the project tracker). The secondary criterion
below uses the plain ECE because that is the number in the existing tables; the Brier score and the debiased ECE are
reported alongside it and gate nothing in this study.

## 5. Pre-registration

Recorded on 2026-10-13, before any run described in this note. The thresholds below are not changed after the sweep
runs; a failure is reported as a failure.

**Setting for both criteria:** VAL `all`, all 2223 clips, evidence = the torso degradation of section 2 (the
leg-glimpse condition) on top of the default simulator settings, the same seeds as the reference, evaluation-only
with the recorded checkpoints.

**Primary.** `evidence_locked_same_budget` under leg-glimpse evidence must not exceed the `oracle_transformer` full
MPJPE under oracle evidence (66.7 mm) by more than **+10 mm**. The criterion is met if the pooled VAL `all` full MPJPE
of `evidence_locked_same_budget` under leg-glimpse evidence is at most **76.7 mm**. The 95 % interval is reported but
the point estimate decides. The threshold is generous: `evidence_locked_same_budget` is already 1.9 mm below the
reference with oracle evidence (64.8 mm), so it has 11.9 mm of its own degradation available.

**Secondary.** Contact ECE (plain 10-bin estimator) of `evidence_locked_same_budget` under the same evidence must stay
below **0.15**. Its ECE with oracle evidence is 0.101, so the headroom is 0.049.

**Reported but not criteria:** every other cell of the sweep (noise, mask, combined), the same quantities for
`oracle_transformer`, hidden and visible MPJPE, penetration, skate, contact F1, Brier score and debiased ECE, and the
cohort breakdown.

**Decision rule.** If both criteria hold, the evidence-locked completion tolerates leg-glimpse evidence at this
simulator's noise level, and nothing else is concluded from it (see section 3: the oracle estimates in the slice are
still accurate). If either fails, the result is reported as found and the design of the training evidence is revisited
before any new training run. The criteria never refer to `oracle_transformer` under degraded evidence; it is a
descriptive comparison only.

## 6. What is not in this study

- No retraining and no new model; the two checkpoints are used as recorded.
- No change to the oracle observation simulator, its defaults, or any completed experiment's acceptance rule.
- No perception model: no 2D detector, no 3D lifter, no HR-HMR integration, no egocentric render set.
- No real video; real egocentric validation (EgoBody) is a later step.
- No evaluation on TEST.
- No new calibration method, and no choice of the gating calibration metric (section 4).
