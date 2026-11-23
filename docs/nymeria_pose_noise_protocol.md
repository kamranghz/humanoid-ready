# Nymeria: error of a head pose estimated from video, protocol

Status: written on 2026-11-21; **approved by the owner on 2026-11-21** (VGGT-Omega as main estimator, MonST3R as second). **Run on 2026-11-21 with VGGT-Omega only (amendment A2); result in section 9. Supplementary MonST3R check on 7 windows added by amendment A3 (2026-11-22); run on 2026-11-23, result under A3 below.** From here on, changes are made only as dated amendments.

## 1. Question

The foot-lock detector of `docs/nymeria_foot_lock_learned_protocol.md` and `docs/nymeria_foot_lock_noise_protocol.md` was tested with simulated noise on the head pose (independent per
frame, 0 to 5 cm and 0 to 2 degrees). Head poses estimated from video will have a different error. Question: **how large is the error of a head pose estimated from the Nymeria preview
video, in the same units as the simulation, and is it independent from frame to frame or slowly varying?** This is a measurement of the pose estimator, not a test of the detector, and no
foot-lock label is read.

## 2. Data

The 7 sequences that have the preview video (`..._preview_rgb.mp4`, 1408 x 1408, 30 fps, H.264) and a head SLAM trajectory: 3 are in the 60 test sequences of the learned detector
(breanna, tammy, samantha), 3 are in its training pool (hayley, elizabeth, robert), 1 is in neither (jeffery). Reference: the head SLAM trajectory (`closed_loop_trajectory.csv`), the
same signal the detector was trained on, interpolated to the video frame times (video and SLAM clocks agree within 7.5 ms, `docs/nymeria_data_format.md` 7.3). The SLAM pose is itself an
estimate; its error is not measured here and is taken as small against the errors below.

## 3. Estimators

Two feed-forward models from `physical_fidelity/models` on the owner's machine, each used as released, no fine-tuning, no setting varied:

- **VGGT-Omega** (main; fast, runs on every window). The 1B 512 checkpoint, `models/vggt-omega/checkpoints/vggt_omega_1b_512.pt`, which is already on the machine; the folder `models/VGGT`
  has the code of the first VGGT but no weights, so the first VGGT is not used.
- **MonST3R** (second; built for video with moving things, which these videos have: a second person is often in view; it returns per-frame camera poses; slower because of its global
  alignment, so it runs on every fourth window, 10 per sequence). Checkpoint `models/monst3r/checkpoints/model.safetensors`.

MoVieS is not used: it is a novel-view model that takes camera poses as input (from another tool) and does not estimate them. DUSt3R and MASt3R are pairwise models with no advantage over
these two here.

## 4. Procedure

1. **Windows.** Per sequence, 40 windows of 64 consecutive frames (the length of the detector's input), start times evenly spaced over the part where video and SLAM overlap. Fixed here,
   not chosen by looking at results. About 18,000 frames in total.
2. **Input images.** Each fisheye frame is turned into an upright pinhole view of 512 x 512 pixels (the model's input size) with a horizontal field of view of 90 degrees around the camera axis, using the camera
   model already in `scripts/nymeria_project_check.py` (`project_fisheye624`, the sideways-mounted sensor rotated as there). Pixels are read by projecting the pinhole rays into the
   fisheye frame, so no inverse of the fisheye model is needed.
3. **Estimate.** Each model is run on the 64 frames of a window; camera-to-world poses are taken for every frame and moved to the head device origin with `T_Device_Camera` of the
   calibration.
4. **Alignment.** Single-camera poses have no metric scale and no world frame. Per window, a similarity transform (scale, rotation, translation) is fitted by least squares from the
   estimated head positions to the SLAM positions of the same frames. The fitted scale is kept. Errors are the residuals after this fit; the alignment is therefore generous to the
   estimator.
5. **Guard.** If the median over windows of the rotation error after alignment is above 10 degrees in a sequence, the camera model or the frame assumption (calibration frame of
   2880 pixels, section 6 of the data-format note) is wrong for it; the sequence is reported as failed and not averaged in.

## 5. What is reported

Per estimator, per sequence, per speed half (windows below and above the median head speed), and for the pool and test groups separately:

1. Position error per frame: RMS in cm (all axes, and the vertical axis alone).
2. Yaw error (rotation about the gravity direction of the estimate against SLAM), standard deviation in degrees; roll and pitch error likewise.
3. Step error: standard deviation of the difference between consecutive frames' position errors. For independent per-frame noise of sigma this equals sigma x sqrt(2), so the
   **equivalent independent sigma** = step error / sqrt(2), directly comparable with the 1, 2 and 5 cm of the simulation. The detector's input is the step, so this is the number that
   matters most.
4. Correlation of the position error: autocorrelation at lags of 1 frame, 0.5 s and 1 s (independent noise gives about 0; the correlated control of the noise protocol gives 0.37 at 1 s).
5. Scale: the fitted scale per window and its spread within a sequence (a detector fed metric steps needs a metric scale; a single camera does not give one).

## 6. Reading (fixed before the run)

Read for each estimator separately and both stated; no estimator is chosen afterwards by its result. A statement about "video poses" is made only if it holds for both.

- Equivalent independent sigma at or below 2 cm and lag-1 s autocorrelation below 0.3: the simulated conditions the noise-trained detector was tested under cover this error; the
  detector can be tried on estimated poses.
- Above 2 cm, or autocorrelation 0.3 or more: the simulation under-states the error, and the next protocol is a different input (steps over a longer span, smoothed positions) rather than
  more noise training.
- Scale spread within a sequence above 10%: a scale source (wearer height, IMU, gravity-plus-known height) is needed before video poses can feed the detector at all.
- No result here says that the detector works on video. It says how large the pose error is that such a test would have to live with.

## 7. Known limits stated in advance

- Two estimators, one video quality (the compressed preview), 7 sequences, mostly "Game night" and "Body stretch" scenes; low light, a second person often in view, which a model trained
  on static scenes does not expect.
- Windows of 64 frames only; no long-range drift or loop closure is measured.
- The error is measured after a per-window similarity fit that real use would not have; real error is at least this large.
- The 3 test-cohort sequences are used only as pose data. If the measured levels are later used to set a training noise, the number used comes from the 4 sequences outside the test
  cohort, and the test sequences are reported separately.

## 8. Order of work

1. Owner reads and approves this protocol, or asks for changes (dated, before any run).
2. Write `scripts/nymeria_pose_noise.py` (reuses the camera code of `nymeria_project_check.py`); test on made-up data with a known pose error; the owner runs it in three steps:
   `--step prepare` (frames and SLAM reference, numpy and ffmpeg only), `--step estimate --model omega|monst3r` (each in the environment where that model works; a one-window smoke test
   first with `--limit 1 --sequence ...`), `--step report` (output `review/nymeria_pose_noise.json`, written once).
3. Record the result in section 9 of this file and in `docs/decisions.md`.

## 9. Result

Run on 2026-11-21: `--step report --model omega`, written once to `review/nymeria_pose_noise.json` (script hash 80e93bb08a82a39f). 7 sequences, 40 windows of 64 frames each, 280 windows of VGGT-Omega
(1B 512 checkpoint, as released), about 5.8 s per window. MonST3R was not run beyond one window (amendment A2). Sequences are named by group and order of section 2 (pool 1 to 3, test 1 to 3, other).

**Time alignment (diagnostic).** Best frame shift of the image change against the SLAM rotation speed, over the 276 windows with correlation at least 0.5: -2: 2, -1: 9, 0: 248 (90%), +1: 15, +2: 2;
mean +0.02 frames. The non-zero shifts are scattered over all sequences, none is systematic.

**Pooled, all 7 sequences (none failed the 10 degree guard; the median rotation error is 0.1 to 0.3 degrees in every sequence).**

| Quantity | Variant A (free scale per window, 145 moving windows) | Variant B (one scale per sequence, all 280 windows) |
|---|---|---|
| Position error, RMS | 1.24 cm (vertical 0.72) | 5.67 cm (vertical 3.36); moving windows only 7.78 cm |
| Step error, RMS | 0.32 cm | 0.50 cm |
| **Equivalent independent sigma** | **0.13 cm** | **0.21 cm** |
| Yaw std / tilt RMS | 0.32 / 0.72 degrees | same rotation (the scale does not change it) |
| Lag-1 autocorrelation | 0.97 | 1.00 |

Outside the test cohort (4 sequences): equivalent sigma 0.15 cm (A) and 0.24 cm (B), position error 1.51 cm (A) and 5.80 cm (B), lag-1 autocorrelation 0.97. Test cohort alone (3 sequences): 0.09 cm (A), 0.15 cm (B).
Slow half against fast half of the windows (A): 0.07 and 0.14 cm. Per sequence, A: 0.06 to 0.19 cm; B position error 1.0 to 8.6 cm.

Simulated references through the same procedure on the same windows (1 cm, lag 1 frame): independent noise -0.01, correlated noise 0.93; the midpoint is 0.46. Ratio of step error to position error: 0.18.

**Scale.** Free scale per window, moving windows: median per sequence 0.78 to 2.05 (arbitrary units of the model), spread (coefficient of variation) 24%, 44%, 43%, 27%, 52%, 49%, 35% for
the seven sequences (pool 1, test 1, pool 2, pool 3, test 2, test 3, other).

**Reading by the rules of section 6 and A1 (fixed before the run).**

1. Equivalent independent sigma is 0.13 cm (0.15 cm outside the test cohort), far below 2 cm. B is 0.21 cm, also below 2 cm.
2. The lag-1 autocorrelation 0.97 is above the midpoint 0.46 (and close to the correlated reference 0.93): **slowly varying**. By the rule, the statement is "above 2 cm or slowly varying: the simulation
   under-states the error", and the next protocol is a different input rather than more noise training.
3. The scale spread within a sequence is 24 to 52%, above 10% in all seven: **a scale source is needed** before estimated poses can feed a detector that uses metric steps.

**Allowed wording.** For VGGT-Omega (1B 512, as released) on the preview video of 7 sequences, after a per-window similarity fit to the SLAM trajectory, the frame-to-frame error of the head position is
equivalent to independent noise of 0.13 cm, much smaller than the 1, 2 and 5 cm of the simulation; the error that remains is smooth (lag-1 autocorrelation 0.97), and the scale of the estimate changes
between windows of one sequence by 24 to 52%. Nothing is said about other estimators, and nothing about whether the detector works on these poses.

**Observation, not a conclusion.** The two failures of the reading are the same kind of error (smooth, not jitter). Whether a detector that reads steps is hurt by smooth error and a changing scale is not
measured here; the 1 cm independent test does not cover it and the correlated control of the noise protocol (4.5 points of lead at 2 cm, worse than B2 at 5 cm) only partly does.

**Limits of this result.** One estimator. Variant A takes its scale from the SLAM trajectory in every window and removes a per-window rotation and offset, which real use would not have, so the real
error is at least this large; windows with a SLAM camera path under 15 cm (135 of 280) are left out of A. The SLAM pose is itself an estimate and its error is not measured. 64-frame windows only, no drift
over longer spans. Mostly indoor scenes of two kinds, compressed preview video.

## 10. Amendments

**A1 (2026-11-21, before any run; written while writing the script, after tests on made-up data).** They make the procedure exact and replace the text above where they differ.

1. **Positions compared** are the RGB camera centres (SLAM device position plus the lever arm of `T_Device_Camera`) and the upright-camera rotations, not the device origin. The estimators give
   camera centres in an unknown scale, so a metric lever arm cannot be applied to them.
2. **Alignment (replaces 4.4).** Per window: the rotation is the chordal mean of the per-frame rotation between the SLAM and the estimated camera orientation (stable when the head hardly moves;
   a rotation fitted from positions is undefined then); the translation matches the window means; the scale is of two kinds. **Variant A**: least-squares scale per window, used only on
   *moving* windows, those whose SLAM camera path in the 64 frames is at least 15 cm. On near-static windows a free scale collapses towards 0 and hides the error (made-up check: 0.02 cm
   reported for a true 1 cm). **Variant B**: one scale per sequence, the median of the free scales of its moving windows, applied to all windows (and to the moving ones); it keeps the
   window-to-window scale changes in the error. Both are generous: a scale fitted to noisy data is attenuated (about 5% at 1 cm and 12% at 3 cm on made-up data), so the true error is at least
   what is reported.
3. **Correlation (replaces 5.4 and the autocorrelation bullet of 6).** Inside a 64-frame window after the fit, long-lag autocorrelation is removed even for slowly varying error (made-up
   check: simulated correlated error shows 0.92 at one frame, 0.0 at 0.5 s and -0.37 at 1 s). The correlation is therefore read at a lag of one frame and compared with the same number from
   simulated independent noise and from simulated correlated noise (coefficient exp(-1/30) per frame, 1 cm) put through the same procedure on the same windows: "slowly varying" if it is above
   the midpoint of the two. Lags 15 and 30 frames and the ratio of step error to position error are reported, not used.
4. **Reading (6).** The 2 cm limit is applied to the equivalent independent sigma of variant A on moving windows. If variant B is above 2 cm while A is not, the statement says so (a scale
   source is needed). The scale-spread bullet of 6 is read from the spread (coefficient of variation) of the free scales of the moving windows of each sequence, reported with the scale itself.
5. **Guard (4.5).** Per sequence: median over windows of the median rotation error after the alignment above 10 degrees means failed; the sequence is reported, not pooled.
6. **Diagnostic, not a filter.** Per window, the shift in frames (-2 to 2) at which the image change best follows the SLAM rotation speed is reported, because the first frame returned after a
   seek can be off by one (data-format note, 7.3).
7. **Estimator settings.** VGGT-Omega: 1B 512 checkpoint, float32, no autocast, batch of the 64 frames. MonST3R: the settings of the `demo.py` command line (512 input, scene graph `swinstride-5-noncyclic`,
   300 iterations, shared focal length, smoothing 0.01, flow loss 0.01, batch 16); its image loader crops square images to 512 x 384, which is left as released.
8. **Groups reported:** pool, test, other, outside the test cohort (pool and other), and all. The reading is given for all sequences and for the outside-test group.

**A2 (2026-11-21, after the smoke tests, before the report; approved by the owner).** MonST3R is dropped as second estimator. On one window (64 frames, the first window of the first test sequence) its global alignment took
10,215 s (2 h 50 min, 300 iterations at about 34 to 38 s per iteration); the 70 planned windows would take about 8 days. On that one window its error was larger than VGGT-Omega's (position error 6.8 cm
against 1.3 cm, rotation 5.6 against 0.9 degrees after the same alignment; one window, a plumbing check, not a result). The settings were not changed to speed it up. The report is made for VGGT-Omega only
(`--step report --model omega`); that window of MonST3R is not in the result file. Statements about "video poses" in section 6 are therefore not made: the result holds for VGGT-Omega only.

**A3 (2026-11-22, before any supplementary run; approved by the owner).** A supplementary MonST3R check, added because MonST3R is built for scenes with moving things (a second person is often in view) and A2 dropped it for time only. It does not
change the main result of section 9 and does not replace A2.

1. **Windows.** One window per sequence, 7 in all. Rule, from the SLAM reference only (no estimate was looked at): the window with a SLAM camera path of at least 15 cm (a moving window) whose index is closest to 20, the lower index if
   two are equally close. The windows, in the order of section 2 (pool 1 to 3, test 1 to 3, other): 19, 20, 20, 20, 20, 17, 20 (SLAM camera path 31, 20, 15, 38, 25, 120 and 42 cm). In the script: `--subset supplement`.
2. **Settings** as in A1 item 7; nothing is changed to make it faster. The one window run earlier (first window of the first test sequence, A2) is not in this set and is not used.
3. **Time.** About 2 h 50 min per window, about 20 hours in all; the run can be stopped and continued (finished windows are skipped).
4. **Report.** `--step report --subset supplement` writes `review/nymeria_pose_noise_supplement.json` (a separate file, written once); it holds MonST3R and VGGT-Omega on the same 7 windows, with the same statistics, the same reference
   autocorrelation procedure and the same reading as section 9. With one window per sequence, variant B equals variant A on these windows (the sequence scale is that window's scale), and the scale spread within a sequence cannot be measured; the 7 fitted scales are listed only.
5. **Reading (fixed before the run).** The numbers are descriptive: 7 windows are too few to call one estimator better. Reported: for each window the two estimators side by side, and the number of windows (of 7) in which MonST3R has the smaller equivalent
   independent sigma and the smaller rotation error. A statement about "video pose estimators" in general (not only VGGT-Omega) is made only if, for both, the equivalent independent sigma of the step error is below 2 cm and the error is slowly varying
   (lag-1 autocorrelation above the midpoint of the simulated references). The need for a scale source is not tested again: it follows from the single camera. If MonST3R and VGGT-Omega disagree, the section 9 statement stays limited to VGGT-Omega.
6. **Limits.** 7 windows, one per sequence, chosen to move; no long-range or sequence-level statement; the time of the run (1 to 2 nights) is the only reason the sample is small.

**A3 result (run 2026-11-23, `review/nymeria_pose_noise_supplement.json`, script hash 46c51a3155ce9e8e).** 7 windows, one per sequence; MonST3R took 48 min to 10 h 44 min per window (the 10 h window is the one with the largest error and flow loss). Per window, equivalent independent
sigma (cm) / median rotation error (deg) / position error (cm), VGGT-Omega then MonST3R, in sorted sequence order:

| window (sequence, group) | sigma Omega / MonST3R | rotation Omega / MonST3R | position Omega / MonST3R |
|---|---|---|---|
| hayley (pool) | 0.043 / 0.100 | 0.17 / 0.77 | 0.19 / 0.71 |
| breanna (test) | 0.068 / 0.046 | 0.27 / 0.98 | 0.37 / 0.48 |
| elizabeth (pool) | 0.048 / 0.065 | 0.17 / 0.29 | 0.21 / 0.74 |
| robert (pool) | 0.051 / 0.049 | 0.06 / 0.11 | 0.14 / 0.41 |
| tammy (test) | 0.060 / 0.083 | 0.18 / 1.37 | 0.21 / 1.62 |
| samantha (test) | 0.182 / 0.474 | 0.59 / 2.97 | 0.87 / 4.68 |
| jeffery (other) | 0.060 / 0.049 | 0.11 / 0.21 | 0.24 / 0.33 |

- MonST3R has the smaller sigma in 3 of 7 windows (breanna, robert, jeffery; robert by 0.002 cm) and the smaller rotation error in 0 of 7. Its position error is larger in 7 of 7. Pooled, sigma 0.09 cm (VGGT-Omega) and 0.19 cm (MonST3R); lag-1 autocorrelation 0.86 and 0.97
  (simulated independent 0.04, correlated 0.91, midpoint about 0.48).
- By the rule fixed in item 5: for both estimators sigma is below 2 cm and the error is slowly varying, and they do not disagree, so the section 9 statement may be widened from VGGT-Omega to "VGGT-Omega and MonST3R, on these 7 windows".
- Not shown: which is better (7 windows); anything about scale (one window per sequence). The 7 fitted scales were not reported by the script (field empty), so no scale statement is made from this run.
- Limits as in item 6.
