# Nymeria: foot lock from the head pose with a network learned on raw windows, protocol

Status: APPROVED by the owner 2026-11-21 (sections 1 to 9 including A1 and A2). Amendment A3 (what sections 2.3 and 2.4 left open, written together with the
training script) is added at the end, before any model is trained. No model has been trained. Nothing in sections 1 to 9 is changed after the first run; any
later change is added as a dated amendment at the end.

This is a separate protocol. The first protocol (`docs/nymeria_foot_lock_protocol.md`: B1, B2, M1) and the second
(`docs/nymeria_foot_lock_repo_model_protocol.md`: the repo's model R1) stay as the record of their runs. Their results are not replaced by this one.

Decision it rests on: `docs/decisions.md`, entry "Nymeria: the 137 untried sequences may be used for training (2026-11-21)". Facts about the data:
`docs/dataset_sheets.md`, `docs/nymeria_data_format.md`.

## 1. Question

The first protocol showed that a speed threshold (B2) and a logistic regression on hand-made features (M1) reach a balanced accuracy of about 0.80 from the head
pose alone, but miss 38% of the planted frames at 5% false-lock, and 68% in the hard case (both feet planted while the head moves). The second protocol showed
that a model trained only on simulated evidence did not transfer.

> Can a network, trained on real Nymeria head-pose windows that no earlier experiment touched, with no hand-made features and no other signal, tell
> "both feet planted" better than the hysteresis speed threshold on the 60 held-out sequences, and does it help in the hard case?

This is supervised learning on raw signal windows. It is not reinforcement learning.

## 2. What is fixed

### 2.1 Sequences

- **Test set**: the 60 sequences of the foot-lock cohort, with the same labels (both_planted from Xsens foot and toe horizontal speed), the same clock offsets,
  the same hard stratum, the same upright rule and the same SLAM-gap exclusion as the first protocol. They are not used for fitting, tuning, early stopping or
  choosing an operating point. They are evaluated once, with the final model.
- **Training pool**: the 137 sequences no experiment has touched, minus the 3 that share gender, height, weight and age group with a test sequence
  (`docs/decisions.md`, 2026-11-21), minus every sequence that fails the data-check steps below. The 9 sequences that failed the offset step earlier stay out.
- **Data-check step on the training pool (first, before any model)**: run `scripts/nymeria_foot_lock_data_check.py` unchanged (offset step with refined correlation
  at least 0.95; height-order id gate of amendment A4; labels). Report per sequence the same numbers as for the cohort, and the counts: how many of the 134 pass,
  how many fail each step, the total both_planted time, stepping time and hard-stratum time. Failures are listed and never retried with other settings.
- **Size rule**: if fewer than 60 training sequences pass, or their total both_planted time is under 60 minutes, the result is labelled indicative.

### 2.2 Folds inside the training pool

- Group key: the metadata combination (gender, height, weight, age group). Sequences with the same combination always go to the same side.
- Validation: groups ordered by the SHA-256 of the combination string; the groups whose hash value modulo 5 equals 0 form the validation set (about 20%), the
  rest train. The validation set is used for early stopping and for the operating point. The test set is never used for either.

### 2.3 Input (head pose only)

- SLAM head pose on the 30 Hz grid in SLAM time, as in the first protocol (frames whose bracketing samples are more than 0.1 s apart, and the 75 frames around
  them, are left out of training and evaluation).
- Window: 64 frames (2.13 s) of raw pose, per frame: position relative to the last frame of the window (3 values), the same position difference between
  consecutive frames (3 values), and the orientation as the 6 numbers of the first two columns of the rotation matrix, all expressed in a frame that has gravity
  as the vertical axis and the horizontal heading of the last frame as the forward axis. No speed, spread, band power or other derived feature is given;
  the network computes what it needs. Absolute height above the floor is not given (SLAM has no floor; the second protocol found that height did not help).
- Two rows, as before: causal (the label is the frame at the end of the window) and lookahead (the window ends 15 frames after the labelled frame).

### 2.4 Network and training (two sizes fixed in advance, chosen on validation only)

- A temporal convolutional network: residual blocks of one 1-D dilated convolution (kernel 5), GELU and layer norm; the output is read at the labelled frame;
  one output, the probability of both_planted. Two sizes, both fixed here:
  - **S**: 4 blocks, 64 channels, dilations 1, 2, 4, 8 (about 0.1 million parameters).
  - **M**: 6 blocks, 192 channels, dilations 1, 2, 4, 8, 16, 32 (about 1.1 million parameters).
- Loss: binary cross entropy, no class weights. AdamW, learning rate 1e-3, weight decay 1e-2, batch 256 windows, at most 30 epochs over randomly drawn
  windows, early stopping on the validation loss with patience 5. Dropout 0.1. Augmentation: none in the main run (noise is a control, section 4).
- Three seeds (0, 1, 2) per size, identical settings otherwise.
- **Choice of size, on validation only**: the size with the lower validation loss at the best epoch, averaged over the three seeds, is the chosen size; on a tie
  (difference under 0.002) the smaller one. Only the chosen size is evaluated on the test set; the test result of the other size is not computed, so there is no
  choice between sizes made on the test set. Both validation losses are reported.
- The rule is applied to the chosen size, on the per-sequence mean of its three seeds' missed-lock; rows per seed are reported as well.
- These are the only settings. If a run fails to train (loss does not fall), that is reported as a failure, not fixed by trying other settings.

### 2.5 Operating points and baselines

- Operating points "ba" (best balanced accuracy) and "fl5" (most permissive with false-lock at most 5%) are chosen on the validation set only.
- B2 and M1 are rerun on the same 60 test sequences with the code of the first protocol, folds as in that protocol, so their numbers are the recorded ones.
- Extra baseline, for fairness: **B2-137 and M1-137**, the same B2 and M1, fitted on the training pool (not on folds of the 60), with the operating point on the
  validation set. This separates "more data" from "a learned representation". The rule below uses B2 as recorded, and the comparison with M1-137 is the question
  that says whether the network adds something beyond more data.

## 3. Rule (recorded before the run)

1. Signal check: the network's "ba" causal balanced accuracy is at least 0.60 on at least 80% of the 60 sequences.
2. Main comparison, "fl5", causal rows: the network's missed-lock is lower than B2's on more than 60% of the held-out sequences, the median per-sequence
   difference (B2 minus network) is at least 5 percentage points, and the 95% sequence-bootstrap interval of that median (1000 resamples) excludes 0. Held-out
   false-lock of both is printed next to it and read together.
3. Beyond-data comparison (reported, with its own wording): the same statistic against M1-137. If the network meets 2 but not this, the allowed claim is that
   more real training data helps, not that a learned network does.
4. Hard case: reported, not part of the rule: missed-lock in the hard stratum for the network, B2, M1 and M1-137.
5. If 2 is met, the allowed claim: "on 60 Nymeria sequences, from head pose only, a network trained on N other sequences beats a speed threshold by at least 5
   points of missed-lock at 5% false-lock (new sequences, mostly known places)". If not: "a network trained on N sequences did not beat a speed threshold".
   Either goes in the decisions file as it is.
6. No setting, threshold or operating point is changed after seeing a test result.

## 3A. New-place test (a second, separate result, recorded before the run)

The main result (section 3) is on new sequences in mostly known places (28 of the 30 test locations also occur in the training pool). To measure new places:

1. Locations: the location field of each sequence's metadata. Order the locations of the test sequences by the SHA-256 of the location name; take locations in that
   order until the held-out locations hold at least 15 test sequences (all of them if fewer). These are the held-out locations. The counts (locations, test
   sequences, and training sequences removed) are reported before any model is trained.
2. Training for this test: the training pool without every sequence at a held-out location. Same network size as chosen in 2.4 (not chosen again), same settings,
   three seeds, same validation rule (validation groups are redrawn from the reduced pool by the same hash rule).
3. Test: only the test sequences at the held-out locations, with the same metrics and the same rule as section 3 items 1 and 2 (B2 as recorded as the reference).
4. Reading: this is a second result, reported next to the first and never merged with it. It has fewer sequences, so the bootstrap interval is wider and the rule is
   stated with that count. If the reduced training pool has fewer than 60 sequences, or lost more than 30% of the pool, it is labelled indicative.
5. Allowed claim: "new places" is claimed only if this result meets the rule of section 3 item 2 on its own; if not, the claim stays "new sequences, mostly known
   places", and the number found here is reported as the drop.
6. Cost: one more set of three trainings.

## 4. Controls (reported, not part of the rule)

- C1, label shuffle in training: train with the training labels circularly shifted by 60 s inside each sequence; test balanced accuracy must fall to about 0.5.
  If it does not, there is leakage and the main result is withdrawn.
- C2, learning curve: the chosen size trained on 25%, 50% and 100% of the training sequences (nested, by group); report fl5 missed-lock on the test set for each (these are controls and are not used to change anything). This says
  whether more data would still help.
- C3, head-pose noise sweep as in the first protocol (position 0, 1, 2, 5 cm with yaw 0, 0.5, 1, 2 degrees), applied to the test input only.
- C4, clock shift of -100, -50, 0, +50, +100 ms on the test labels (the lookahead row is the one to read for an optimum).
- C5, test-sequence overlap: the same comparison without the sequences listed as `side_check_inconsistent` in the data check.
- C6, location split (descriptive only): results on test sequences whose location also occurs in the training pool and on those whose location does not; the
  counts are reported first and a part with fewer than 5 sequences is not tested. The real new-place test is section 3A.

## 5. Also reported

Balanced accuracy, F1, missed-lock, false-lock, slip speed (mean true foot speed over locked frames), upright variants, onset and offset timing, pooled ECE,
block bootstrap (10 s blocks, 200) and sequence bootstrap (1000), per seed and for the mean.

## 6. Known gaps and limits stated in advance

- Locations: 28 of the 30 locations of the test sequences also appear in the training pool, and the scripted activities are mostly the same. Results show new
  sequences, not new places. C6 gives a partial look.
- People: the files give no person identity. Sequences with the same metadata combination are kept together, but two sequences of the same person with
  different metadata cannot be found; test and training may share people.
- Floor-heavy scripts: 9 of the training pool and 3 of the test cohort. The network may do worse on floor work than on standing; the per-sequence numbers show it.
- Labels come from an IMU suit and define "planted" by foot speed only (first protocol, section 8).
- Aria closed-loop SLAM is cleaner than a real-time tracker; only the simulated noise sweep speaks to that.
- Only the head pose is used. Wrist devices (the check of 2026-11-21 did not pass: median correlation 0.66 at the best lag, lags spread from about -1 to +1.7 s),
  eye gaze, IMU and AMASS pretraining are later steps, each with its own recorded protocol; none is used here.
- Number of usable training sequences is not known until the data-check step has run.

## 7. Order of work

1. Owner reads and approves this protocol (or asks for changes; changes before the first run are made in the text and dated).
2. Data-check step on the training pool (read-only; report counts; no model).
3. Write the training and evaluation script; independent code review before any run (as for the first protocol), including a check that no test sequence reaches
   training, validation, early stopping or operating-point choice.
4. Train three seeds for each of the two sizes on the training pool; choose the size and the operating points with the validation set only.
5. Evaluate once on the 60 test sequences; run controls; then the new-place test of section 3A (train again without the held-out locations, test once on their sequences); record both results in section 8 and in `docs/decisions.md`.

## 8. Result

### 8.0 Data check of the training pool (2026-11-21, before any model)

The repo's data check (`scripts/nymeria_foot_lock_data_check.py`, unchanged) was run on the 134 sequences of the training pool; report
`review/nymeria_learned_training_data_check.json`.

- 134 attempted; 118 passed the offset step (refined correlation 0.95 or more); 16 failed it (correlations from -0.05 to 0.94) and are left out, never retried.
- Of the 118, one failed the height-order id gate (foot and toe below knee below pelvis: 0.94 and 0.92 against 0.95) and is left out; 3 had too little upright
  time to test the ids and are kept and labelled, as in the cohort; 4 have a left/right share below 0.80 (`side_check_inconsistent`) and are kept and flagged,
  as in the cohort. The run's final line "STOP" refers to the one sequence that failed the height gate; no label was built for it.
- **117 labelled training sequences.** Totals: 1831 min of labelled time, 1239 min both feet planted (68%), 553 min one foot planted, 39 min neither, 604 min of
  hard stratum, 1335 min upright, 841 min both planted while upright. The size rule of section 2.1 is met (at least 60 sequences and 60 min); the result is not
  labelled indicative.
- Locations: the 117 sequences cover 41 locations. Of the 30 test locations, 3 do not occur in the training pool (two earlier; one more because all its pool
  sequences failed a check).

### 8.0a Held-out locations for the new-place test (section 3A), fixed before any model

Test locations ordered by the SHA-256 of the location name; taken until the held-out locations hold at least 15 test sequences: 9 locations, 15 test sequences
(1 to 3 per location). Of the 117 training sequences, 24 (21%) are at these locations and are removed for that test, leaving 93 (more than 60 and a loss under
30%, so not labelled indicative). With 15 sequences over 9 locations the interval of that test will be wide; this is stated in advance.

### 8.1 Model results (2026-11-21)

Run on the owner's machine (RTX 4090, torch 2.11.0+cu128). Code hash 4c8bf9a575bb98e6, plan hash 94144e6a26c2d104. Files: `review/nymeria_learned_models/`,
`review/nymeria_learned_size_choice.json`, `review/nymeria_learned_test_result.json`, `review/nymeria_learned_newplace_result.json`,
`review/nymeria_learned_controls.json`. The main test and the new-place test were each run once; the first-protocol rerun inside both is identical to the recorded numbers.

**Size choice (validation only).** All six runs trained. Mean validation loss S 0.2475 (seeds 0.2467, 0.2460, 0.2498), M 0.2572 (0.2547, 0.2508, 0.2661); base-rate loss
0.6602. M was not lower by 0.002, so S (83,842 parameters) is the chosen size.

**Main test: 60 test sequences, S, mean of 3 seeds per sequence, median over sequences (quartiles).**

| row, operating point | balanced acc. | missed-lock | false-lock | hard-stratum missed |
|---|---|---|---|---|
| causal, ba | 0.875 (0.843-0.909) | 0.072 (0.048-0.150) | 0.139 (0.102-0.191) | 0.181 (0.137-0.259) |
| causal, fl5 | 0.878 (0.835-0.916) | 0.131 (0.086-0.266) | 0.081 (0.043-0.103) | 0.330 (0.260-0.436) |
| lookahead, ba | 0.904 (0.872-0.929) | 0.047 (0.031-0.097) | 0.111 (0.085-0.160) | 0.118 (0.091-0.172) |
| lookahead, fl5 | 0.912 (0.881-0.936) | 0.078 (0.050-0.161) | 0.068 (0.046-0.099) | 0.187 (0.150-0.275) |

Baselines, causal, fl5, median over the same 60 sequences:

| | missed-lock | hard-stratum missed | false-lock |
|---|---|---|---|
| network | 0.131 | 0.330 | 0.081 |
| B2, first protocol | 0.375 | 0.683 | 0.048 |
| M1, first protocol | 0.355 | 0.688 | 0.041 |
| B2-137 | 0.344 | 0.580 | 0.093 |
| M1-137 | 0.363 | 0.698 | 0.039 |

Rule (section 3): signal check passes (60 of 60 sequences at 0.60 or more). Item 2, network against B2: lower on 60 of 60 sequences, median difference 24.3 points, bootstrap 95%
interval 22.6 to 27.0: **met**. Item 3, beyond-data: against M1-137 lower on 60 of 60, 23.4 points (19.4 to 26.1); against B2-137 lower on 59 of 60, 20.2 points (15.2 to 24.1);
against M1, first protocol, 21.9 points (18.6 to 24.2): **met**, so the claim is not limited to "more real data helps". Pooled calibration error (causal) per seed 0.0064, 0.0036, 0.0029.

Reading, stated as it is. (a) Allowed claim (section 3 item 5): on 60 Nymeria sequences, from head pose only, a network trained on 95 other sequences (22 more for validation) beats a speed
threshold by at least 5 points of missed-lock at 5% false-lock; the 60 sequences are new sequences, mostly in known places. (b) The false-lock of the network at the "fl5" point is 0.081 on the
test sequences, above the 5% it was set to on validation; B2-137 (same selection procedure) is 0.093, M1-137 is 0.039, the first-protocol baselines 0.041 to 0.048. Against M1-137 and
the first-protocol baselines the network therefore runs at a higher false-lock, and part of its lower missed-lock may come from that; against B2-137 the false-lock is similar.
(c) Hard stratum (both feet planted while the head moves): the network misses 0.330, the baselines 0.580 to 0.698; one third is still missed.

**New-place test (section 3A): 15 test sequences at 9 held-out locations, trained on 75 sequences (18 validation), S, three seeds. Indicative: 15 sequences.**
Causal fl5: balanced accuracy 0.873 (0.797-0.896), missed-lock 0.143 (0.090-0.285), false-lock 0.110 (0.065-0.157), hard-stratum missed 0.333. Causal ba: 0.855, 0.075, 0.201. Lookahead fl5:
0.896, 0.073, 0.115. Baselines, missed-lock at fl5: B2 0.383, M1 0.366, B2-137 0.360, M1-137 0.365; hard stratum 0.684, 0.688, 0.623, 0.729; false-lock 0.056, 0.060, 0.084, 0.057.
Rule: signal check passes (15 of 15); against B2 lower on 15 of 15, median difference 24.4 points (17.4 to 28.5); against M1-137 24.3 (17.6 to 30.7); against B2-137 21.8 (17.3 to 36.2):
**met on the held-out locations**. Section 3A item 5 allows the words "new places" for this result; the number of sequences (15) and the reduced training pool must be stated next to it.
The false-lock is higher in new places (0.110 against 0.081); calibration error per seed 0.016, 0.023, 0.014.

**Controls (reported, not part of the rule).**

- **C1, label shift: incomplete.** The runs for 60 s and 90 s were stopped by the script's guard (A3 item 10: a shift within 10% of no shift on more than 10% of the sequences stops
  the run). The 95 training sequences are 11 to 21 minutes long (median 15.6), so a 60 s shift is 5% to 9% of the length, and the guard judged a shift of 60 s (and of 90 s on 85 of 95
  sequences) as "no shift", although it moves the labels by a full minute against the head motion. That is a design error in the guard, which should be in seconds; no model for
  these two shifts exists. The 120 s run trained: validation loss 0.6268 against a base-rate loss of 0.6602, gain 0.0334 (the main seed-0 model gains 0.4135, so 8%). On the true test labels
  its pooled AUC is 0.888 and its median balanced accuracy 0.797 at its own operating point (descriptive; as written in A3 item 10, ranking measures can be high for a network that learned
  little). The gain of 0.033 is probably slow context (minutes of standing or walking stay partly aligned after a short shift); this is a reading, not tested. The verdict of A3 item 10 needs all
  three shifts, so **leakage is neither found nor excluded**; the script's "leakage_suspected: false" means "not evaluated". A4 (below) is proposed, not applied.
- **C2, learning curve (seed 0, chosen size).** Validation loss 0.2743 / 0.2516 / 0.2467 for 25% / 50% / 100% of the training sequences (24 / 48 / 95). On the test sequences, causal fl5
  missed-lock 0.174 / 0.139 / 0.131 and false-lock 0.066 / 0.083 / 0.077; causal ba balanced accuracy 0.864 / 0.869 / 0.874. More data still helps, with a small step from 50% to 100%.
- **C3, noise on the test input, causal, median missed-lock at fl5 (balanced accuracy):**

| noise (position, yaw) | network | B2 | M1 |
|---|---|---|---|
| 0 cm, 0 deg | 0.131 (0.878) | 0.375 (0.777) | 0.355 (0.791) |
| 1 cm, 0.5 deg | 0.971 (0.512) | 0.385 (0.773) | 0.386 (0.774) |
| 2 cm, 1 deg | 0.980 (0.506) | 0.463 (0.725) | 0.444 (0.750) |
| 5 cm, 2 deg | 0.976 (0.503) | 0.739 (0.598) | 0.520 (0.712) |

  **The network does not survive the smallest noise tested**: at 1 cm and 0.5 deg it is at chance, while B2 and M1 degrade slowly. The noise-free row equals the main result. The noise is drawn per
  frame and independently, which is harsher than the correlated error of a real tracker, but the baselines see the same noise. A likely cause (not tested): the input holds the step between
  frames times 30, so 1 cm of per-frame noise is about 0.3 m/s of false speed, against a locking signal below 0.05 m/s. Consequence: the result holds for clean head poses of the kind this
  data set provides (closed-loop SLAM of the Aria glasses), and **does not support a claim for noisy head poses, for example from video**. Section 6 listed this as an open gap; it is now measured.
- **C4, clock shift of the test labels.** Lookahead: balanced accuracy at fl5 0.893 / 0.906 / 0.912 / 0.906 / 0.889 for -100 / -50 / 0 / +50 / +100 ms; the optimum is at 0 ms. Causal: 0.854 /
  0.865 / 0.878 / 0.884 / 0.887, still rising at +100 ms but within 1 point from +0 to +100. The label clock is aligned to within about 50 ms.
- **C5, without the 4 `side_check_inconsistent` sequences.** 56 sequences; signal check passes; against B2 lower on 56 of 56, median difference 24.8 points (21.7 to 27.6): rule met.
- **C6, location split (descriptive).** Locations in the training pool: 52 sequences, 23.6 points (20.7 to 27.1), network false-lock 0.070. Locations not in the pool: 8 sequences, 25.9 points
  (24.2 to 34.3), false-lock 0.090. No sign of a location effect on the missed-lock difference.

**What is not shown.** Anything about noisy head poses (C3 is negative); the C1 verdict; wrist devices, gaze, IMU (later steps); more than one data set.

## 9. Amendments

- A1, 2026-11-21, before the data-check step and before any model exists: section 2.4 first named one network of about 0.1 million parameters. At the owner's request it is now two sizes (S about 0.1 million, M about 1.1 million), chosen by validation loss as written in 2.4, because one size alone could not say whether a larger network would have helped. Only the chosen size is evaluated on the test set. Nothing else changed.
- A2, 2026-11-21, before the data-check step and before any model exists: added section 3A (new-place test with held-out locations, retrained once) and reduced C6 to a descriptive control. Reason (owner): the first result is mostly in known places, so it cannot support a claim about new places. Nothing else changed.
- A3, 2026-11-21, before any model exists: written together with the script `scripts/nymeria_foot_lock_learned.py`; it fixes what sections 2.3 and 2.4 left open. It
  replaces the description of the frame in 2.3 (gravity-aligned heading frame), nothing else is removed.
  1. **Input per frame (15 values), all in the device frame of the LAST frame of the window**: position relative to the last frame (3, metres); the step from the
     previous frame times 30 (3, metres per second); the first two columns of the rotation relative to the last frame (6); the direction of gravity in the
     device frame of the frame itself (3). Reason: the "heading" of a head that looks down at the floor is undefined, and the device frame has no such
     pose. The SLAM world frame is z-up; the script checks this from the gravity columns of every trajectory file and stops otherwise. Nothing is fitted or
     normalised on data (the only constant is the factor 30). Frames outside a sequence are the nearest frame of that sequence (edge padding), as in the first
     protocol's features. Window ends run from the first frame to 15 frames after the last frame.
  2. **One network, two outputs.** Output 0 is read at the last frame of the window (causal row). Output 1 is read 15 frames before the last frame (lookahead row).
     The loss is the mean of the two binary cross entropies, each averaged over its own valid frames: a window counts for an output only if the frame that output
     labels is labelled and not near a SLAM gap. Both outputs share all layers except the last 1x1 convolution.
  3. **Blocks**: 1x1 convolution to the width, then per block: dilated convolution (kernel 5, zero padding that keeps the length), GELU, layer norm over the channels,
     dropout 0.1, residual add; then a 1x1 convolution to the two outputs. Sizes as in 2.4 (S 0.084 million, M 1.113 million parameters).
  4. **Epoch**: 400 steps of 256 windows drawn at random (with replacement) from all valid window ends of the training sequences. At most 30 epochs.
  5. **Validation loss** for early stopping and for the size choice: every 10th window end of all validation sequences, the mean of the two outputs' losses; the
     best epoch's weights are kept. Operating points ("ba", "fl5") of each seed: all valid validation frames, with the weights of the best epoch, score = minus the
     logit, lock when the score is below the threshold.
  6. **Not trained**: a run whose best validation loss is not below the loss of a constant predictor (the base rate of the training labels) minus 0.001 is reported
     as "did not train" and nothing is tried instead.
  7. **Folds**: the group string is gender|height_cm|weight_kg|age_group (the metadata values written with `str`), joined by "|"; a group is validation if
     the first 8 hex digits of its SHA-256, read as a number, give 0 modulo 5 (the rule of the repo's fold function). Nested subsets for C2: training groups in
     ascending SHA-256 order, the shortest prefix that holds at least 25% (50%) of the training sequences.
  8. **Per-sequence values for the rule** are the mean over the three seeds of that sequence's value; the signal check uses the seed mean of the balanced accuracy.
  9. **B2-137, M1-137**: M1 (the first protocol's logistic regression and features, unchanged) is fitted on the training sequences; the operating points of M1 and
     the two thresholds of B2 (the first protocol's search, with the speed quantiles of the validation sequences and the two taus of B1 fitted on the training
     sequences as candidates) are chosen on the validation sequences. Applied to the test sequences without refitting.
  10. **Controls**: C1 shifts the labels of the training and the validation sequences (circular on the 30 Hz grid) by 60 s, and again by 90 s and by 120 s (section 4
      named 60 s only); one seed, the chosen size, operating points from the shifted validation labels, tested on the true test labels. Reason, found when the script
      was tried on made-up data: a network that learned nothing can still rank frames by head speed with a tiny weight of random sign, so its balanced accuracy on the
      true labels can land far from 0.5 in either direction (0.99 or 0.02 were seen) and "about 0.5" is not a reliable test of leakage. Reading, fixed here: for each
      shift the script reports whether the run trained (validation loss below the base-rate loss), the logit spread, the balanced accuracy and the area under the
      ROC curve on the true test labels, all descriptive. The verdict uses the validation loss only, because it measures what was learned, not just the ranking:
      leakage is suspected, and section 4's consequence (main result withdrawn pending a check) applies, only if for all three shifts the run on shifted labels
      reached a validation loss at least 0.02 below the base-rate loss (shifted labels carry no information about the head motion). A shift that lands within 10%
      of no shift on more than 10% of the sequences (a sequence shorter than the shift allows) stops the run. C2 uses seed 0; its 100% point is the seed-0 model of the main run. C3 adds the noise to the
      head position (independent per frame) and as a rotation about the vertical axis (independent yaw per frame), with the same random numbers as the first
      protocol's sweep, to the test input only, for all three seeds; B2 and M1 are rerun under the same noise with the first protocol's code. C4 moves the test
      labels, not the operating points.
  11. **The test set is evaluated once.** The script refuses to write a second test result for the same scope; a rerun needs the owner to rename the first file
      and is then recorded here as an amendment.
  12. **Reproducibility**: seeds fix initialisation and window sampling; on a GPU the result is not bitwise reproducible. Every result file records the code hash,
      the plan hash, the torch version and the device.
  13. **Receptive field.** The convolutions are centred. The causal output at the last frame therefore sees only the frames before it: the size S network (dilations
      1, 2, 4, 8) reaches 30 frames back (about 1 s of the 2.13 s window), the size M network the whole window. Recorded as a property, not a defect; it is part of the
      choice between the sizes.
  14. **Safeguards added after the independent review of the script (2026-11-21, before any model):** the held-out false-lock of the network and of B2, M1, B2-137
      and M1-137 is printed and stored next to the rule; if the rerun of the first protocol on the test sequences differs from the recorded numbers by 1e-9 or more
      the test step stops and writes nothing; runs flagged "did not train" are listed in the size choice and in the test report; the plan stops if a metadata key is
      missing or if a metadata combination occurs in both the training pool and the test set; `--smoke` and `--limit` are refused for the choose, test and controls
      steps. The newplace report carries the note that it is indicative (15 test sequences).
- A4, 2026-11-21, PROPOSED, not applied, not approved (made after the test results of 8.1, none of which it can change): (1) the guard of A3 item 10 measures the distance of the shift from "no
  shift" in seconds (at least 30 s) and not as a share of the sequence length; (2) leakage is suspected only if, for all three shifts, the gain of the shifted run is at least half of the
  gain of the main seed-0 model over the base-rate loss (instead of an absolute 0.02), and each gain is also reported as a share of it. The 50% was chosen after seeing that the 120 s run kept
  8%; a leak would keep most of the gain. The 120 s run needs no retraining; the 60 s and 90 s runs would be trained.
