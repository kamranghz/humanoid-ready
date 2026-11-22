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

## 7. Amendment A1 (2026-11-21, before the first run): definitions the protocol left open

Written together with `scripts/nymeria_foot_lock_repo_model.py`. Nothing here was chosen after seeing a result on Nymeria.

1. Camera rotation. (Superseded by Amendment A2 in section 8: the device z axis is not the direction the wearer faces.)
   The model's camera matrix has the rows [right, up, look] (world to camera). From the Aria device rotation
   (world from device) the rows are [-y_device, -x_device, z_device]: device x points down, y left, z forward. This was seen on
   two sequences by hand and is checked on every sequence by the script: the mean gravity direction along device x, over the upright
   frames (pelvis above 0.80 m; all frames if there are fewer than 30 s of them), must be at least 0.7, and on sequences with at least 10 s of walking above 0.5 m/s the mean walking direction along device z must be at
   least 0.3. If one sequence fails, the script stops before scoring anything and lists the failing sequences. The result of the
   check is written to the report for every sequence.
2. Window. 64 frames at 30 Hz ending at the frame; x and y are taken relative to the first frame of the window (as the
   model's training windows are); z is the height above the assumed floor; gravity (0, 0, -9.81) in the world frame; all 22 body
   joints marked not visible with zero confidence and no 2-D keypoints. The model is used in evaluation mode, in single
   precision, once per window, with no sampling.
3. Rows. One window gives both rows: the causal score of frame f is the model output at the last frame of the window that ends
   at f; the lookahead score of frame f is the output at the frame 15 frames before the end of the window that ends at f + 15.
   A window is used only if none of its 64 frames lies inside a SLAM gap or the 2.5 s margin around one (the exclusion of the
   first protocol). The first 63 frames of a sequence have no causal score.
4. Frames. Every detector in the comparison (B1, B2, M1 and the model, called R1 in the files) is evaluated on exactly the frames
   where the model has a score. B1, B2 and M1 are therefore re-evaluated here and their numbers differ slightly from the first
   run. They are fitted on the training folds as before.
5. Score and operating points. The score is the minimum of the four contact probabilities. "ba" and "fl5" are chosen on the
   training folds by the code of the first protocol. The "ECE" in the report is the pooled calibration error of this score as if it
   were a probability of "both planted"; it is not the calibration of the model's own contact output.
6. Floor. Floor height in the SLAM z frame = (median over upright both-planted frames of SLAM device z minus Xsens head z) +
   (median over the same frames of the lower of the two Xsens toe-segment heights). The offset between the glasses and the Xsens
   head origin is not measured, so C2 shifts the floor by -10, -5, -3, +3, +5 and +10 cm (a positive shift = floor assumed higher).
7. Controls. C1 as written in section 4 cannot be built: the Xsens and SLAM coordinates share no yaw alignment in the cache.
   C1 is replaced by C1b: the SLAM horizontal motion and orientation with the Xsens head height (plus the constant offset of
   item 6). The home-domain check (the same model, head only, on AMASS windows) is not part of this run and is reported as
   "not run". C2: the floor shifts of item 6. C3: clock shifts of -100, -50, +50 and +100 ms on both rows and a 60 s circular
   label shift on the causal row. C4: head height constant at 1.6 m above the floor.
8. Rule. Applied by the code of the first protocol (`e1.rule`) with the model in place of the learned detector: the signal check
   on the model's "ba" balanced accuracy (at least 0.60 on at least 80% of sequences), and the main comparison model against B2
   at "fl5" on the causal rows (lower missed-lock on more than 60% of the sequences, median difference at least 5 points,
   bootstrap interval excluding 0). The lookahead row is reported next to it. A test shows that the same code with M1 in the
   model's place gives the first protocol's numbers exactly.
9. Compute. Every window is scored (stride 1). A `--stride` option exists only for quick looks and its results are not used for
   the rule.
10. Code changes to the first experiment script: one optional argument (`mask`, default none) that restricts the frames; with
    the default the output is identical to the output before the change (tested).
11. Sequences without a floor estimate. A sequence with fewer than 30 s of usable upright both-planted frames (Xsens heights that are
    missing do not count) cannot give a floor estimate. It is left out of every detector's comparison, listed in the report with
    the reason, and the run goes on. This is different from the axis check, which stops the run.
12. Floor and the word "causal". The floor height is one number per sequence, estimated from the whole sequence, including later
    frames and the label-defined frames (upright and both planted). The causal row is causal in the model's input windows but not in
    this one number. Any claim from the result says "with the floor height given".
13. Attribution. The input "all body joints not seen for 64 frames" may be rare in the model's training. The home-domain check (the
    same model on AMASS windows with the same head-only input) is not run (the report says "not run"). A negative result therefore
    mixes "the head pose holds no contact information" with "this input is outside what the model was trained on", and the
    decisions file must say so. The causal row also reads the output at the last position of a model that looks both ways inside
    its window, where outputs are usually worse; this is fixed by the protocol.
14. Reported next to the rule: the share of sequences where the model's held-out false-lock is not higher than B2's, because the
    rule compares missed-lock only; the full table (F1, slip speed, upright variants) of B1, B2, M1 and the model; the mean
    elevation of the look direction (a diagnostic of a possible fixed pitch offset between the glasses and the head); the SHA-256 of
    the three model source files, the torch and numpy versions and the protocol file hash.
15. The scores are kept in double precision when the operating points are chosen (single precision can put a threshold on the wrong
    side of a tie; a test shows 49 instead of 50 frames).
16. Use a fresh `--cache` folder for the real run. `--rebuild` does not rebuild the cache of the first experiment.

## 8. Amendment A2 (2026-11-21, before any model output): the camera rotation

Written after the first call of `--step prepare` stopped, and before anything was scored. No model output exists.

What happened.
- The check of A1 item 1 failed on 6 of the 58 prepared sequences (2 more sequences had already been left out for having no
  floor estimate, A1 item 11). As A1 says, the run stopped and nothing was scored.
- A read-only diagnostic (`scripts/nymeria_axis_diagnostic.py`; it runs no model and reads no foot label) was run on the 58
  sequences. Gravity points along device x on all of them (median 0.89), so "x is down" holds. The walking direction is not
  along device z: the mean cosine is only 0.53 (median over sequences), and the walking direction also has a steady component
  to the right (0.44 along -y). The angle that fits each sequence has a median of 39.6 degrees (spread 7.8; 10th to 90th
  percentile 31 to 48). The same sign and size in all 58 sequences cannot come from head turns. Device z points about 40
  degrees to the left of the direction the wearer faces.
- Cause, from the documentation and the recordings themselves. The device frame of the Aria glasses is the local frame of the left
  SLAM camera (https://facebookresearch.github.io/projectaria_tools/docs/data_formats/coordinate_convention/3d_coordinate_frame_convention),
  which looks outward. The online calibration of the head recording (first line of `online_calibration.jsonl`, inside the recording zip) gives the rotation of every camera in the device frame. In the first recording read by hand the RGB camera's
  optical axis is 38 degrees to the right of device z. The baseline between the two SLAM cameras points the same way (0.3
  degrees apart). So three independent sources agree: the walking directions (39.6), the RGB extrinsics (38) and the SLAM baseline.
- A1 item 1 took "z is forward" from two sequences seen by hand. That was wrong in yaw. The axes x down and y left were right.
  This is an input error, not a matter of thresholds. Loosening the thresholds and going on would have given the model a camera
  rotated by about 40 degrees.

What changes (all of it is decided before any model output).
1. Camera. The model's camera is the glasses' RGB camera. Its rotation is read for every recording from that recording's own
   calibration: world-from-camera = world-from-device times `T_Device_Camera` of `camera-rgb` (the camera axes are x down, y left,
   z forward, because the sensor is mounted turned). Rows [right, up, look] = [-y, -x, z] of the camera axes in the world, as
   before. The head position stays the device origin.
2. Check. It is now pooled over the sequences. Per-sequence values are reported and are not a reason to stop. The run stops only if
   (a) the median over sequences of the mean gravity along the camera's down axis is below 0.7, or
   (b) the median over sequences of the mean cosine between the walking direction and the horizontal part of the look direction
   (frames above 0.5 m/s, at least 10 s of them) is below 0.6, or more than 10% of the sequences that can be tested are below 0.3, or
   (c) on any sequence the camera yaw from the RGB extrinsics and from the SLAM-camera baseline differ by more than 10 degrees.
   The thresholds were chosen after the diagnostic above (which used the old, wrong frame) and before any model output. From the
   diagnostic I expect about 0.7 for (b) on the real data. The walking check alone is coarse: a clean 38 degree error still gives
   0.79. The protection against a yaw error is item 1 (the calibration) and check (c).
3. A recording whose calibration is missing or not a rotation is left out of every comparison and listed in the report, like item
   11 of A1.
4. Caveats written now. (i) The RGB camera is not necessarily level with the wearer's forward direction in pitch, and the repo's
   training camera looked along the head's forward direction; no pitch correction is made, and the mean look elevation is reported
   (A1 item 14). (ii) The head position is the device origin, which is the left SLAM camera, a few centimetres to the left of the
   head centre; it is not corrected. (iii) The attribution caveat of A1 item 13 stays: a negative result cannot separate "head pose
   holds no contact information" from "this input is outside what the model was trained on", and now also not from the two
   conventions in (i) and (ii).
5. Code: `INPUT_VERSION` is `R1-A2`, so the inputs cached by the stopped call are rebuilt; the detector keeps the name R1 in all
   files. `scripts/nymeria_axis_diagnostic.py` is part of the record. Its first output (old frame) stays in
   `review/nymeria_axis_diagnostic.json`; run on the new inputs it writes `nymeria_axis_diagnostic_R1-A2.json`.
6. Tests (synthetic data): the calibration is read (RGB yaw 38.1 degrees, 0.3 degrees from the baseline); a device pre-rotated by the
   calibration gives exactly the camera rotation of the identity case (difference 5e-10); the pooled check passes on it; on the
   same data the device frame alone shows the 38 degree error (0.78 against 0.79 expected); a wrong frame is stopped; a missing
   calibration is listed, not fatal; the quaternion convention; and the 13 earlier checks of A1 pass again.
7. Nothing else in sections 1 to 7 changes: same model, same data, same rule, same controls.

## 9. Result of the run (2026-11-21, once, 58 sequences)

Run once with the code of A1 and A2 (`scripts/nymeria_foot_lock_repo_model.py`, code hash 9c9edca8515a001e, protocol hash 54b7635f14660ccf,
model step 140000, SHA-256 as in section 2, torch 2.11.0, numpy 2.4.6), stride 1, on a GPU. 58 of the 60 sequences were used; 2 had no floor
estimate and were left out (A1 item 11). Report: `review/nymeria_foot_lock_repo_model.json`. The result is recorded as it came out.

Camera-frame check (A2): passed. Median gravity along the down axis 0.86; median walking direction along the look direction 0.71 (predicted
about 0.7); 1 of 58 sequences below 0.3; largest yaw difference between the RGB extrinsics and the SLAM-camera baseline 1.1 degrees. Median mean
look elevation -22.7 degrees (range -52 to +6).

Causal rows, median over the held-out sequences (the "ba" point is the best balanced accuracy on the training folds; "fl5" allows 5% false-lock
on the training folds):

| detector | ba: balanced accuracy | ba: missed-lock | ba: false-lock | fl5: missed-lock | fl5: false-lock | fl5: hard-case missed |
|---|---|---|---|---|---|---|
| B2 (hysteresis speed) | 0.797 | 0.144 | 0.173 | 0.380 | 0.048 | 0.684 |
| M1 (logistic, first protocol) | 0.809 | 0.154 | 0.169 | 0.364 | 0.040 | 0.688 |
| R1 (the repo's model) | 0.559 | 0.758 | 0.086 | 0.863 | 0.038 | 0.872 |

(B1 and B2 are the same, as in the first protocol.)

Rule, applied as recorded:
- Signal check: R1 balanced accuracy is at least 0.60 on 27.6% of the sequences (16 of 58); the rule needs 80%. NOT MET.
- Main comparison (R1 against B2 at fl5, causal): R1's missed-lock is lower on 3.4% of the sequences (2 of 58; the rule needs more than 60%); the
  median difference (B2 minus R1) is -41.5 points (95% sequence-bootstrap interval -49.6 to -28.6; the rule needs at least +5 and an interval
  above 0). NOT MET. Held-out false-lock at fl5: R1 3.8%, B2 4.8%. R1's false-lock is not higher than B2's on 53% of the sequences, so the
  gap is not an artefact of a higher false-lock.
- Lookahead row (reported): B2 misses 27.8% and R1 80.2% at fl5; R1 balanced accuracy at "ba" is 0.583 (at least 0.60 on 43% of the sequences).
  Same outcome.

Controls (R1, causal, fl5 missed-lock, median; main condition 0.863):
- C2, floor assumed 10, 5, 3 cm lower: 0.820, 0.855, 0.852; 3, 5, 10 cm higher: 0.866, 0.874, 0.894. No floor height in this range rescues it.
- C1b, Xsens head height with the SLAM horizontal motion and orientation: 0.867, the same as the main condition. The SLAM pose is not the cause.
- C4, head height constant at 1.6 m: 0.792, better than with the real height. The model does not use the height above the floor in a way that
  helps on this data.
- C3, clock shifts of 50 and 100 ms: 0.80 to 0.87, flat. A circular label shift of 60 s: balanced accuracy 0.497, chance. The pipeline
  itself is sound.
- Home-domain check (the same model, head only, on AMASS windows): not run.

What may be claimed. On 58 Nymeria sequences, with the floor height given, the repo's trained completion model, given only the head pose of
the glasses with all body joints marked not seen, does not beat a speed threshold from head pose; it is far worse (it misses 86% of the planted
frames at 5% false-lock, against 38% for the speed threshold).

What may NOT be claimed. That the head pose holds no information about foot contact: a speed threshold and a logistic regression reach a
balanced accuracy of about 0.80 on the same frames. The failure belongs to this model on this input. The run cannot say why. Three readings
fit the controls and none was tested: the input "all joints unseen for 64 frames" is outside what the model was trained on (A1 item 13); the
model's contact output was trained to read body evidence, not the head alone; the head-camera conventions of the glasses differ from the
training rig in a way the checks do not see (the RGB camera looks 23 degrees below the horizon on average; the head position is the left
SLAM camera, A2 item 4). Separating them needs the home-domain check and a run with evidence the model was trained on (for example hand
evidence once the hand-projection check is valid). Both are separate protocols.

