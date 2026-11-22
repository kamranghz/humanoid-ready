# Nymeria: foot lock from the head pose, network trained with noise, protocol

Status: written on 2026-11-21 after the results of `docs/nymeria_foot_lock_learned_protocol.md` section 8.1; **approved by the owner on 2026-11-21** (his request: make the network work
under noise and compare it with the clean-trained one). Code added the same day to `scripts/nymeria_foot_lock_learned.py` (`--step train --variant noise`, `--step test --scope noise`);
the result was recorded the same day (section 8). From here on, changes are made only as dated amendments.

## 1. Question

The network of the learned-detector protocol (clean training) is at chance as soon as 1 cm of independent per-frame noise is added to the head position (8.1, control C3; missed-lock 0.97
against 0.385 for the speed threshold B2). Head poses estimated from ordinary video will carry noise of that size or more. Question: does the same network, trained with noise added to
its input, keep a lead over the speed threshold at 1 cm and 2 cm of noise, without losing its clean result?

This is a step towards the real-video goal, not the real-video result: the noise here is simulated.

## 2. What is fixed

Everything of the learned-detector protocol stays as it is (sequences, folds, labels, input of 15 values per frame, network size S with 83,842 parameters, block layout, optimiser, batch,
epoch of 400 steps, at most 30 epochs, patience 5, three seeds 0, 1, 2, validation loss, operating-point search, baselines). **The one change is noise added to the training input.**

1. **Noise generator.** As in `noisy_input` of the script: independent per frame, position sigma in metres on each axis, yaw sigma in degrees about the vertical axis (a rotation applied to
   the head rotation of that frame). Applied to the whole sequence before windows are cut, so the step input sees the noisy positions.
2. **Training noise.** At the start of every epoch each training sequence gets a fresh noisy copy (new random numbers each epoch, not those of the test). With probability 0.2 the copy is
   clean. Otherwise sigma_pos is drawn log-uniform between 0.3 cm and 8 cm, and sigma_yaw (degrees) = 0.45 x sigma_pos (centimetres), the ratio of the sweep. The drawn levels cover the
   test levels (1, 2, 5 cm) without being equal to them.
3. **Validation noise.** The validation sequences get one noisy copy, drawn once with a fixed seed from the same mixture (clean with probability 0.2, otherwise as in item 2) and kept for
   all epochs and seeds. Early stopping, "did not train" and the operating points use this copy. One pair of operating points ("ba", "fl5") per seed, used at every test noise level (a
   deployed detector does not know the noise level).
4. **Size.** S only; no size choice is made again. The best epoch is reported; if it is within the last three epochs the run is called "possibly not converged" in the result.
5. **Cost.** Three trainings of about one minute each on the owner's GPU.

## 3. Test (second use of the 60 test sequences)

The 60 test sequences were used once (8.1). This is a second use and is reported as such, next to the first, never merged with it. Nothing is chosen on the test sequences: the noise
recipe, the size, the rule and the reading below are fixed here before any run. The test noise levels are those of control C3 (position, yaw): (0 cm, 0), (1 cm, 0.5), (2 cm, 1), (5 cm, 2), with the same
random numbers as C3, the causal row at the "fl5" point, three seeds, per-sequence mean over seeds. B2 and M1 are the numbers of C3 (first-protocol code, thresholds picked in-sample on the
test folds under the same noise, which favours them). The network of the clean run is shown next to the new one at every level.

## 4. Rule (recorded before the run)

1. At 1 cm and at 2 cm separately: the network's missed-lock at "fl5" is lower than B2's at the same noise on more than 60% of the 60 sequences, the median per-sequence difference is at
   least 5 percentage points, and the 95% sequence-bootstrap interval of that median (1000 resamples) excludes 0. Held-out false-lock of both is printed next to it.
2. Clean level (0 cm): the median missed-lock of the noise-trained network is at most 3 points above that of the clean-trained network (0.131), and its false-lock is printed next to it.
   If not, the allowed wording says that robustness cost accuracy on clean poses.
3. Reported, not part of the rule: 5 cm; M1 and M1-137 at every level; the hard stratum.
4. Allowed claims. If items 1 and 2 are met: "trained with simulated independent per-frame noise, the network keeps a lead of at least 5 points of missed-lock over a speed threshold at 1 and
   2 cm of noise, on 60 sequences already used once". If item 1 is met only at 1 cm: say so, with that limit. If item 1 is not met: "training with noise did not make the network robust at
   1 cm". Either goes in the decisions file as it is. No setting is changed after a result.
5. If item 1 is not met, the next steps are named now and each needs its own protocol: a different input (step over a longer span, smoothed positions), a larger network, or a different
   noise model. None of them is tried inside this protocol.

## 5. Also reported

Balanced accuracy, F1, missed-lock, false-lock, hard-stratum missed-lock, per seed and for the mean; pooled calibration error; best epoch and validation loss per seed (noisy and clean
validation loss both); descriptive control: **correlated noise**, position noise with a correlation time of 1 s (first-order autoregression with coefficient exp(-1/30) per frame, same
stationary sigma of 1, 2 and 5 cm, yaw 0.5, 1 and 2 degrees), causal "fl5" row, network of both runs against B2 and M1 rerun with the first-protocol code under the same noise. This is
closer to the error of a tracker than independent noise and is not part of the rule.

## 6. Known gaps and limits stated in advance

- The noise is simulated. Errors of poses estimated from video can also drift, jump, change scale (single camera) and drop out; none of this is in the sweep. A pass here does not show that
  the detector works on video. The next measurement for that is the error of an actual pose estimator on these sequences, compared with the levels used here.
- The 60 test sequences are not fresh; see section 3.
- The input is unchanged, so a network that cannot use it under noise will not be rescued by training alone; that is a possible result (rule item 5).
- Sequences are mostly in known places; the new-place test is not repeated here.
- Labels come from an IMU suit and define "planted" by foot speed only.

## 7. Order of work

1. Owner reads and approves this protocol, or asks for changes (dated, before the first run).
2. Add the noise variant to `scripts/nymeria_foot_lock_learned.py` (training with noise, test at the noise levels, the correlated-noise control) without changing any existing step; an
   independent code review as before if the owner asks for one, including a check that no test sequence reaches training, validation or an operating point.
3. Train three seeds; evaluate once on the 60 test sequences; record the result in a section 8 of this file and in `docs/decisions.md`.

## 8. Result

Run on 2026-11-21 by the owner (three trainings, then one test run; output `review/nymeria_learned_noise_result.json`, written once). Second use of the 60 test sequences. Causal row at
"fl5", median over the 60 sequences of the per-sequence mean over three seeds. B2 and M1 are the numbers of control C3 (first-protocol code, thresholds in-sample under the same noise).

**Training.** Best epochs 10, 6 and 9 (none within the last three epochs, so no run is "possibly not converged"); validation loss on the noisy copy 0.3119, 0.3202, 0.3153 (base-rate
loss 0.6602); on the clean validation sequences 0.2957, 0.2989, 0.3015 (the clean-trained size S had a mean validation loss of 0.2475).

**Independent per-frame noise** (missed-lock, false-lock in brackets):

| noise (position, yaw) | noise-trained | clean-trained | B2 | M1 |
|---|---|---|---|---|
| 0 cm, 0 deg | 0.237 (0.067) | 0.131 (0.081) | 0.375 (0.048) | 0.355 (0.041) |
| 1 cm, 0.5 deg | 0.284 (0.059) | 0.971 (0.006) | 0.385 (0.044) | 0.386 (0.041) |
| 2 cm, 1 deg | 0.353 (0.053) | 0.980 (0.007) | 0.463 (0.055) | 0.444 (0.038) |
| 5 cm, 2 deg | 0.366 (0.068) | 0.976 (0.017) | 0.739 (0.053) | 0.520 (0.037) |

**Rule item 1 (noise-trained against B2, same noise): met at both levels.**
- 1 cm: lower on 93.3% of the sequences, median difference 10.0 points, 95% interval [8.3, 12.0].
- 2 cm: lower on 85.0% of the sequences, median difference 11.6 points, 95% interval [9.4, 14.3].
- Not part of the rule: 0 cm 13.3 points [11.9, 14.7] (100% of sequences); 5 cm 31.0 points [28.3, 34.2] (100%). Against M1: 10.6, 10.1, 9.9 and 16.0 points at 0, 1, 2 and 5 cm
  (lower on 96.7%, 90.0%, 80.0% and 93.3% of the sequences; every interval excludes 0).

**Rule item 2 (clean level): not met.** The median missed-lock of the noise-trained network on clean poses is 0.237 against the limit 0.161 (0.131 + 0.03); the median per-sequence
difference to the clean-trained network is 9.8 points in its disfavour, interval [8.7, 11.4], and the noise-trained network is lower on none of the sequences. False-lock
at the clean level: 0.067 against 0.081. Hard stratum (both feet planted while the head moves), missed-lock at the clean level: 0.542 against 0.330 for the clean-trained network (B2 0.683, M1 0.688).

**Outcome (as written by the script):** item 1 met at 1 cm and 2 cm, item 2 not met: robustness cost accuracy on clean poses. Allowed wording (section 4 item 4): the network trained with
simulated independent per-frame noise keeps a lead of at least 5 points of missed-lock over a speed threshold at 1 and 2 cm of noise, on 60 sequences already used once, and robustness cost
accuracy on clean poses (missed-lock 0.237 against 0.131).

**Against the clean-trained network under noise:** the noise-trained one is lower on all 60 sequences at 1, 2 and 5 cm, by 68.3, 62.8 and 60.9 points; the clean-trained one is at or near
chance (balanced accuracy 0.50 to 0.51) at all three levels.

**Correlated noise (descriptive, correlation time 1 s, same sigma; not part of the rule):**

| noise | noise-trained | clean-trained | B2 | M1 |
|---|---|---|---|---|
| 1 cm, 0.5 deg | 0.259 (0.060) | 0.741 (0.024) | 0.380 (0.046) | 0.361 (0.047) |
| 2 cm, 1 deg | 0.359 (0.046) | 0.930 (0.008) | 0.402 (0.046) | 0.400 (0.044) |
| 5 cm, 2 deg | 0.839 (0.010) | 0.984 (0.004) | 0.555 (0.051) | 0.506 (0.038) |

Noise-trained against B2: 1 cm lower by 11.8 points [9.7, 13.8] (96.7% of the sequences); 2 cm lower by 4.5 points [1.9, 7.4] (68.3%), which is below the 5-point margin of item 1 if it
had been applied here; 5 cm worse than B2 by 21.0 points (interval of the median difference [-27.0, -14.4]; lower on only 18.3% of the sequences) and worse than M1 by 27.4 points. So the lead
is much smaller for slowly drifting errors, which is closer to the error of a tracker, and is lost at 5 cm.

**Reading, within what the numbers carry.**
- Training with noise made the network usable at 1 and 2 cm of independent noise; the clean-trained network is not.
- It paid for this on clean poses: about 10 points more missed locks, and a hard-stratum missed-lock about 21 points higher. Neither network dominates the other: the clean-trained one is better on exact
  poses, the noise-trained one under any noise.
- At 5 cm of independent noise the noise-trained network still beats B2 and M1; at 5 cm of correlated noise it does not.
- The noise is simulated. The noise of a pose estimated from video (drift, jumps, scale, dropouts) is not measured here. Whether the detector works on video is not shown by this result.
- The 60 test sequences have now been used twice for the learned detector; nothing was chosen on them.

Next steps named by the protocol (section 4 item 5, each needs its own protocol and the owner's approval): a different input (step over a longer span, smoothed positions), a larger network, a
different noise model; and, for the real-video goal, the measurement of the error of an actual pose estimator on these sequences against the levels used here.

## 9. Amendments

None. Implementation details fixed in the script on 2026-11-21, before any run (they make the text above exact and change nothing in it): the noisy copy of an epoch is drawn with
random numbers seeded by (seed, epoch); the fixed validation copy by the number 20261121 and the order of the validation list; yaw noise is a rotation about the vertical axis applied to
the head rotation, so the gravity direction input is unchanged by it; the correlated noise is a first-order autoregression with coefficient exp(-1/30) per frame, unit stationary
variance scaled by the sigma, one draw per axis and one for yaw, and the baselines B2 and M1 are rerun with the same correlated noise through a copy of the first protocol's
noise function; the clean-trained network of the comparison is the seed 0 to 2 models of the main run with the operating points of the main run's validation set (clean); the
independent-noise levels use the random numbers of control C3; the output is `review/nymeria_learned_noise_result.json`, written once.
