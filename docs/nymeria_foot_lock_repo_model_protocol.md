# Nymeria: foot lock from the head pose with the repo's trained model, protocol

Status: written before the model is run on Nymeria (2026-11-20). Nothing below is changed after the first run; any later
change is added as a dated amendment at the end.

This is a separate protocol. The first protocol (`docs/nymeria_foot_lock_protocol.md`) stays as the record of the first
run (simple detectors B1, B2 and a logistic regression M1). Its result is not replaced by this one.

## 1. Question

The repo's own trained completion model has a contact output (four sole points: left heel, left toe, right heel, right
toe). Given only the head pose from the glasses (all body joints marked "not seen"), does this model tell when both feet
are planted better than the best simple detector (B2, hysteresis speed threshold)?

In plain words: is the learned model in this repo better than a speed threshold at this one job, on real data it has
never seen?

## 2. What is fixed

- Model: `checkpoints/deterministic_main/best.pt`, SHA-256 `13b1ded8e3a0e93500ebe0e082198ae3a8ffe253b1f31e63c74fec5c8896840a`
  (84,585,465 bytes). One model, used once, no tuning, no fine-tuning on Nymeria, no other checkpoint tried after seeing
  results. Repo commit at the time of writing: `fafe095`.
- Data: the same 60 labelled sequences as the first protocol, same fold rule (SHA-256 of name mod 5), same labels
  (both_planted from Xsens foot and toe horizontal speed), same upright rule, same hard stratum, same SLAM gap exclusion.
- Input to the model: only the rig. Head position in world, camera rotation (world to camera), gravity. Every body joint
  is marked not visible (`joint_visible` = 0, confidence 0, 2-D keypoints 0). No hand evidence is used, because the hand
  projection check on Nymeria is not valid yet.
- Window: 64 frames at 30 Hz (the model's `max_len`), 2.13 s. Windows with a SLAM gap > 0.1 s are excluded.
- Two rows, as before:
  - causal: the model output at the last frame of a window that ends at frame i;
  - lookahead: the model output at the frame 15 frames before the end of that window.
- Floor: the model expects the floor at z = 0. Floor height per sequence is taken from Xsens (median of the lowest sole
  point over upright frames). This is a help the real glasses do not have. It favours the model, so a failure is strong
  evidence and a pass needs this caveat.
- Score: the minimum of the four contact probabilities (all four sole points on the floor). Two operating points are chosen
  on training folds only, the same way as for M1: "ba" (best balanced accuracy) and "fl5" (most permissive with false-lock
  at most 5%). The model itself is not trained on any fold.

## 3. Rule (recorded before the run)

Signal check: the model's "ba" causal balanced accuracy is at least 0.60 on at least 80% of sequences.

Main comparison, on the "fl5" operating point (false-lock at most 5% on the training folds), causal rows: the model's
missed-lock rate is lower than B2's on more than 60% of the held-out sequences, the median per-sequence difference (B2
minus model) is at least 5 percentage points, and the 95% sequence-bootstrap interval of that median excludes 0. This is
the same rule and the same code as the first protocol, with the model in place of M1. Held-out false-lock of both is
printed next to it and must be read together, because it can differ from 5%. M1 is reported next to the model and is not
part of the rule.

If the rule is met, the allowed claim is: "on 60 Nymeria sequences, with the floor height given, the repo's model beats
a speed threshold by at least 5 points". If not met: "the repo's model does not beat a speed threshold from head pose".
Either result goes in the decisions file as it is.

## 4. Controls (reported, not part of the rule)

- C1, convention check: run the same model with a rig built from the Xsens head instead of the SLAM head. If this also
  fails, the problem is the model or the conventions, not the SLAM pose.
- C2, floor sensitivity: floor height shifted by -3 cm and +3 cm.
- C3, clock shift of +-50 and +-100 ms on both rows, and a 60 s circular label shift (should fall to chance).
- C4, no-body control: the same input with the head position replaced by a constant height, to see how much of the result
  comes from the height above the floor.

## 5. Also reported

Balanced accuracy, F1, missed-lock, false-lock, slip speed, hard-stratum missed-lock, upright variants, onset and offset
timing, pooled ECE of the model's probabilities, block bootstrap (10 s blocks, 200) and sequence bootstrap (1000).

## 6. Known gaps

- The model was trained on studio motion (AMASS) with a simulated rig. SLAM head pose from the glasses is a different
  domain. This is the thing being tested, not a flaw in the test.
- The model's contact definition (sole point close to the floor) differs from the label (low horizontal foot speed).
- The model sees a 64-frame window, longer than the features of the first protocol (up to 2 s). It gets no extra
  information, only a different way to use the same head pose.
- Floor from Xsens (section 2).
- Hand evidence is not used. A run with hands is a separate protocol after the hand projection check is valid.
