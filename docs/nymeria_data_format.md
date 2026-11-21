# Nymeria data format (audit of the downloaded files)

**Date:** 2026-11-19. **Status:** partial. Everything below was measured from the files on disk with
`scripts/nymeria_inventory.py` (read-only: file names, zip directories, `metadata.json`, narration CSVs, array headers and the first
line of a few text members), `scripts/nymeria_fetch.py` (list of download groups, sizes redacted of links) and `scripts/mp4_probe.py`
(video header, no decoding). Items not yet read are listed at the end. Licence, provenance and intended use are in `docs/decisions.md`.

## 1. What is on disk

`D:\projects\hready_data\datasets\nymeria\<sequence>\` for 206 sequences, 499.9 GB of zips in total. Sequence names follow
`<date>_<session>_<pseudonym>_<act>_<uid>`; files are `Nymeria_v0.0_<sequence>_<suffix>`.

| suffix | in sequences | content (measured) |
|---|---|---|
| `body_motion.zip` | 206 | `body/xdata.npz`, `body/xdata_blueman.glb` (about 0.9 to 1.5 GB per sequence) |
| `recording_head.zip` | 206 | `data/motion.vrs`, `data/et.vrs`, `mps/slam/` (`closed_loop_trajectory.csv`, `semidense_points.csv.gz`, `summary.json`, `online_calibration.jsonl`), `mps/eye_gaze/` (`general_eye_gaze.csv`, and more) |
| `recording_lwrist.zip`, `recording_rwrist.zip` | 206 | `data/motion.vrs` and `mps/slam/` files |
| `metadata.json` | 206 | licence, script, location, durations, availability flags, participant height and age group |
| `narration_atomic_action.csv` | 169 | free-text sentences with start and end times, the wearer written as "C" |
| `narration_activity_summarization.csv` | 175 | 30 s activity summaries |

**No `data.vrs` in the zips.** The raw head recording is a separate download group (`recording_head_data_data_vrs`); it is absent. The
download list also offers a preview RGB video as its own group (`video_main_rgb`). The original download took neither, so the 206
sequences had no egocentric frames. The preview video was added afterwards for 7 floor-heavy sequences (7.5 GB, section 7); the raw
group was not. See `docs/decisions.md` for the download decision.

## 2. Set statistics (from `metadata.json`, all 206)

- 53.92 h of action time; 45 locations; 29 sequences with two participants; 16 scripted scenarios.
- Every sequence flags body motion, head data and head SLAM as present. The head SLAM log reports a successful run for the two
  sequences whose summary was read; the VRS health check reported an error or warnings on some streams (IMU repeat checks), not examined further.

## 3. Floor-related behaviour in the narrations

Method: sentences of `narration_atomic_action.csv` in which "C" (the wearer) appears before the verb, simple word patterns,
counted per sentence, not per second, and not checked against motion. Counts over the 169 sequences that have the file:

| behaviour | sentences |
|---|---|
| kneeling | 889 |
| squatting or crouching | 782 |
| sitting on the ground | 379 |
| lying (all) | 191 |
| lying, bed or sofa or chair named | 166 |
| lying, floor named | 3 |
| lying, no surface named | 22 |
| push-up or plank | 20 |
| crawling | 11 |

The counts are concentrated: "Game night" (5 sequences) accounts for most kneeling and sitting on the ground, and "Body stretch"
(3 sequences) for most of the rest. Lying on the floor is not supported by these counts. By the repository's cohort rule, lying on
a bed or sofa is raised support, not floor work.

## 4. Body motion (`body/xdata.npz`, array headers of two sequences)

The file is the Xsens motion capture export, not a body-model fit. Arrays, with N frames (197,843 and 233,661 in the two sequences read):

| array | shape | note |
|---|---|---|
| `segment_qWXYZ`, `segment_tXYZ` | N x 92, N x 69 | 23 segments, quaternions and positions |
| `segment_velocity`, `segment_acceleration`, `segment_angularVelocity`, `segment_angularAcceleration` | N x 69 | |
| `joint_angleEulerZXY` | N x 66 | 22 joints |
| `foot_contacts` | N x 4 | a real ground-contact signal, per frame (meaning of the four columns not yet read) |
| `sensor_qWXYZ`, `sensor_freeAcceleration` | N x 68, N x 51 | 17 sensors |
| `timestamps_us`, `frame_index` | N | |
| `identity_*`, `tpose_*`, `tpose-isb_*` | 92 or 69 | calibration poses |
| `segmentCount`, `sensorCount`, `jointCount`, `frameRate`, `frameCount` | 1 | values not yet read |

Segment positions are Xsens segments (23), not the 55 SMPL-X joints; a mapping would be an owner decision, not made here.

## 5. Head SLAM trajectory (`mps/slam/closed_loop_trajectory.csv`)

Columns: `graph_uid, tracking_timestamp_us, utc_timestamp_ns, tx/ty/tz_world_device, qx/qy/qz/qw_world_device`, device linear and
angular velocity in the device frame, `gravity_x/y/z_world`, `quality_score`, `geo_available`, and the ECEF pose. The first row has
gravity (0, 0, -9.81) in the world frame. This gives the head pose and the gravity vector that the project definition treats as given rig signals.
The file is about 417 MB for the first sequence. The `online_calibration.jsonl` first line holds IMU and camera calibrations (next section).

## 6. Head camera calibration (first line of `online_calibration.jsonl`, first sequence)

Three cameras, all with the model `FisheyeRadTanThinPrism` (15 parameters: focal, cx, cy, six radial terms, two tangential, four thin-prism):

| camera | focal (px) | principal point (px) |
|---|---|---|
| `camera-slam-left` | 241.7 | (316.5, 241.2) |
| `camera-slam-right` | 241.8 | (321.1, 242.9) |
| `camera-rgb` | 1218.0 | (1455.8, 1451.6) |

For `camera-rgb`, using the radial terms only (tangential and thin-prism terms ignored) and assuming the calibration frame is 2880 x 2880
(inferred from the principal point, not stated): the half-width of the frame corresponds to 57.7 degrees from the axis, so the
horizontal field of view is about 115 degrees; the corner corresponds to about 70 degrees, a diagonal of about 141 degrees. The radial
polynomial is only meaningful inside the image circle. Whether the calibration was made for 2880 x 2880 is checked after the
alignment step.

## 7. Preview video (`video_main_rgb`)

One sequence checked (`20230706_s0_breanna_reese_act0_cfg468`): `..._preview_rgb.mp4`, 1.42 GB, SHA-1 matches the list. Header
(`scripts/mp4_probe.py`): H.264, 1408 x 1408, no rotation, 30 fps, 39,339 frames, 1311.3 s. The file metadata carries a list of
integers (first value 337531862175, spacing about 33.3 million units), consistent with one timestamp per frame in nanoseconds; which clock
they use has not been identified. One frame at 300 s was looked at: fisheye geometry with a dark rim, upright orientation, the
wearer's gloved hands, forearms and thighs visible, a second person in view, low light. The wearer wears an Xsens suit with cables and
gloves, which no synthetic render in this project reproduces.

### 7.1 Narration clock and floor-work frames (7 sequences, 34 frames)

`scripts/nymeria_floor_frames.py` read the frame-time list of each preview video and compared it with the narration times. In all 7
sequences every narration row lies inside the video's time range (1,400 of 1,400 rows), and the first sequence's video starts at 337.5 s
while its narration spans 499.5 to 1613.3 s. The narration clock and the video frame list therefore appear to be the same clock. This is
a range check; it does not give second-level accuracy. 34 frames were cut at the middle of narrated kneel, sit-on-ground, squat and lie
spans and looked at as contact sheets (kept local, not committed: they show people, with faces blurred in the dataset).

What the frames show (a visual check, not a measurement):

- Where the narration names a visible object (checkerboard, playing cards, puzzle pieces, laptop, yoga mat in the kitchen), the frame shows
  it, in the "Game night" sequences and in several "Body stretch" ones. The clock assumption looks right there; one frame (kneeling
  on a yoga mat, narrated in the living room) shows the kitchen, so the match is not perfect.
- Kneeling at a table ("Game night"): the table, the wearer's gloved hands and forearms are visible, the torso and legs mostly are not.
  Kneeling on a yoga mat during stretching: the camera looks across the room and the body is out of view. The kneel pose is not visible in
  the picture itself in most of these frames.
- Sitting on the ground (cross-legged, legs extended): thighs, knees and feet are clearly in view in 3 of the 4 sequences that have this label.
- Squatting and bending forward: forearms and hands on the floor are visible. Lying (a stretch): the image is rotated by about 90 degrees,
  so the head roll is large; this is why the head pose and gravity are given signals.
- Appearance gap to the synthetic render set: black Xsens suit with red and orange cables and gloves, low or uneven light, strong fisheye distortion with a dark rim, a second person often in view.

Consequence: real-video evaluation of sit_floor looks feasible; for kneel, the evidence in the image is mostly the hands and the head
height, which is the completion case but not a case where the pose can be read off the picture. Frame counts here are small (34 frames of 7 sequences); nothing was run on them.

### 7.2 Alignment of the Xsens motion with the head trajectory and the video

The Xsens `timestamps_us` are **not** on the head SLAM clock: they start between 480 s and 2808 s in the sequences read, while SLAM
starts between 338 s and 33,222 s, and the two do not overlap in 5 of the 7 sequences. The video frame list and the narration are on the
SLAM clock (section 7.1). The offset between the Xsens clock and the SLAM clock therefore has to be estimated per sequence.
`scripts/nymeria_motion_sync.py` does this by matching the head rotation speed (norm of the angular velocity; the same in any frame) of the
Xsens head segment with the SLAM device angular speed over all offsets that keep at least 60 percent of the Xsens recording inside the SLAM
one. On synthetic data with a known offset it returns that offset exactly. Results on the 7 sequences (Xsens covers 907 to 1114 s, SLAM 1069 to 1310 s):

| sequence | correlation at best offset | best elsewhere | head-height correlation after alignment | Xsens head height, kneel spans | Xsens head height, sit-on-ground spans |
|---|---|---|---|---|---|
| 20230706 breanna (Game night) | 0.984 | 0.267 | 0.961 | 1.10 m | no span |
| 20230815 samantha (Game night) | 0.988 | 0.282 | 0.981 | 0.90 m | 0.69 m |
| 20230628 hayley (Game night) | 0.988 | 0.223 | 0.971 | 1.07 m | 0.64 m |
| 20230731 tammy (Game night) | 0.992 | 0.181 | 0.976 | 0.97 m | no span |
| 20230720 robert (Body stretch) | 0.995 | 0.113 | 0.908 | 0.79 m | 0.61 m |
| 20230707 elizabeth (Body stretch) | 0.633 | 0.202 | 0.759 | 0.88 m | 0.73 m |
| 20230816 jeffery (Body stretch) | 0.072 | 0.072 | -0.010 | not trusted | not trusted |

Reading: five sequences are aligned with a sharp peak (0.98 to 0.995 against at most 0.28 elsewhere) and the head heights agree after
alignment; one (elizabeth) has a clear but weaker peak and is flagged; one (jeffery) has no peak, so its alignment is unresolved and its
motion-based numbers are not used. In the aligned sequences the narrated spans behave as the narration says: upright kneeling at the table
gives a head height of about 0.97 to 1.10 m, sitting on the ground 0.61 to 0.69 m (0.73 m in the weaker sequence), and the yoga-mat
kneeling in the stretch sequence 0.79 m. The 90th percentile of the head height is not a standing height in the Game-night sequences (it is 0.90 to 1.10 m
there, because the wearer is on the floor for most of the recording), so the ratio the script also prints is not used. Floor-work spans have narration and
motion on the same clock in 6 of 7 sequences, which gives a first real kneel and sit-on-ground set with video, motion and head pose together.
This table is the coarse stage (`scripts/nymeria_motion_sync.py`, 2026-11-19). The refined correlations, drift, video lag and the spatial fit of one uniform run are in section 7.3. The coarse value of elizabeth (0.633) is far below her refined correlation (0.986) and her scan value (0.951) at the same offset; the cause of that gap is not identified.

### 7.3 One uniform run (2026-11-20, run id 20261120T170201Z)

Every number below comes from one run of `scripts/nymeria_project_check.py --fresh --steps fit,drift,yaw,wrist,video,project,scan --uniform 6`: one code version, one step
set for all 7 sequences, old results removed first (the report stores the run id and a hash of the code for each sequence; `--audit` printed UNIFORM). Rule recorded in the script:
a timing claim needs a refined correlation of at least 0.95; a spatial claim also needs a shared head offset d (at least 4 good 120 s windows).
Older numbers in this repository that differ from this table are superseded.

| sequence | refined corr | change from coarse (ms) | drift windows: n, residual range (ms), slope (ms/h) | video lag vs SLAM, 3 windows (ms) | global fit rms (cm) | good 120 s windows, their rms (cm) | yaw range in good windows (deg) | shared d: windows, norm (m), even/odd difference (cm) |
|---|---|---|---|---|---|---|---|---|
| breanna | 0.984 | +5.1 | 37, -9.5 to +7.2, -30.4 | -7.4 to -0.6 | 5.7 | 9 of 9, 1.2 to 5.9 | 346 to 355 | 9 of 9, 0.181, 3.0 |
| samantha | 0.987 | -9.1 | 30, -9.2 to +6.2, -28.7 | -4.3 to -1.5 | 13.7 | 7 of 7, 0.9 to 5.0 | 99 to 108 | 7 of 7, 0.140, 0.8 |
| hayley | 0.986 | -13.3 | 31, -7.7 to +8.4, +0.5 | -4.5 to +1.9 | 9.5 | 7 of 7, 2.1 to 5.2 | 282 to 294 | 7 of 7, 0.180, 4.5 |
| tammy | 0.999 | +12.9 | 30, -3.8 to +2.3, -15.8 | -4.9 to -2.4 | 26.2 | 6 of 7, 2.0 to 5.1 | 275 to 292 | 6 of 7, 0.193, 3.9 |
| robert | 0.998 | +12.8 | 30, -2.5 to +2.5, -16.9 | -4.9 to +6.5 | 17.0 | 3 of 7, 5.8 to 7.1 | 74 to 76 | none (too few good windows) |
| elizabeth | 0.986 | +6.8 | 31, -12.2 to +12.5, -2.1 | -0.9 to +7.5 | 35.9 | 4 of 7, 4.4 to 8.6 | 348 to 353 | 4 of 7, 0.224, 3.7 |
| jeffery | no peak (coarse 0.072, best scan 0.178) | - | not run | -2.9 to +0.8 | - | - | - | - |

What this supports:

- **Timing.** The head-to-Xsens offset is resolved for 6 of 7 sequences. All 30 to 37 drift windows of each have a correlation of 0.89 or more (0.95 or more in five sequences); residuals stay within
  +-12.5 ms (elizabeth) and within +-9.5 ms in the other five; the slope reaches -30.4 ms per hour (about 8 ppm). The video timing against the SLAM clock is within -7.4 to +7.5 ms in all 21 windows of the 7 videos (a frame is 33.3 ms;
  the first frame after a seek can be one frame off). In all 6 aligned sequences the Xsens segment 6 gives the best rotation-speed match to the head device (0.95 to 0.997, against 0.73 to 0.86 for the next, segment 5).
- **Wrist devices.** Their own offsets can be found by content (correlation 0.93 to 0.98) except breanna (0.13 and 0.14, unusable). Their UTC clocks differ from the head's by -1.6 s to +1.2 s, so UTC is not used to align them.
- **Space.** The heading between the Xsens world and the SLAM world is not constant: within one sequence it moves by 2 to 17 degrees across the good windows (tammy 275 to 292), so one global transform is wrong
  (global fit rms 5.7 to 35.9 cm). Windows of 120 s fit with 0.9 to 8.6 cm. These are head-position consistencies, not hand accuracy. Windows excluded by the rule (rms above 10 cm or orientation spread above 8 degrees):
  robert 4 of 7, elizabeth 3 of 7, tammy 1 of 7.
- **Head offset d** (Xsens head origin to the SLAM device, in the Xsens head frame) is determined within a sequence to 0.8 to 4.5 cm (even and odd windows disagree by that much) but differs between persons: x 2 to 14 cm, y 6 to 8 cm,
  z 9 to 17 cm. A d shared across persons is not supported; the evaluation uses one d per sequence.

**Hand projection measured (2026-11-20, `scripts/nymeria_hand_eval.py`, two independent labellers).** The label is the centre of the back of the wearer's own glove, clicked in the same 48 frames by two people independently (labeller A: the owner; labeller B: a second person). A first labelling pass by A (48 points) is **invalid and withdrawn**: some points were clicked on another person's hands, which gave a median miss of 315 px; no number from that pass is used.
- Agreement (`scripts/nymeria_label_agreement.py`): both marked the same 54 hands as points and 37 as not visible; B marked 5 hands that A left unmarked (not evaluated). Distance between the two clicks on the 54 shared hands: median 20.5 px of 800 (75th percentile 28.4, 90th 35.5, maximum 44.9); 94% within 40 px. B clicks on average 15 px higher in the image than A, so the two people place "the centre" slightly differently.
- Headline labels = consensus (mean of the two clicks where both marked a point; the rule was fixed before it was evaluated): 51 hands in 5 sequences with a shared d (3 more in robert, which has no shared d, are reported apart); 45 of them project into the image.
- Raw Xsens hand origin, median miss: consensus 58.3 px (95% interval 50.0 to 82.7), about 10.1 cm (8.7 to 13.3), 90th percentile 18.4 cm. Labeller A alone: 62.0 px, 10.7 cm. Labeller B alone (56 hands, 48 in view): 62.1 px, 10.1 cm, 90th percentile 20.9 cm.
- After a constant hand-frame offset fitted on the other sequences (leave one sequence out, 47 in view), consensus: 47.0 px (36.0 to 58.8), about 7.6 cm (6.2 to 9.1), 90th percentile 14.3 cm. A alone: 49.2 px, 7.5 cm. B alone: 41.3 px, 7.7 cm. The fitted offsets are stable across folds (consensus: left 4.5 to 7.3 cm along the hand z axis; right 7.6 to 8.9 cm along z and 1.3 to 10.8 cm along x); the y component shifts by roughly 3 cm between labeller A and B, in line with their 15 px click difference.
- So the choice of labeller changes the result by at most 4 px or 0.7 cm, inside the intervals. The click disagreement (about 20 px, 3 to 4 cm) is about a third of the median miss, so most of the miss is not label noise.
- Clock shift of the Xsens motion, median miss in px (consensus): -0.2 s 62.0, -0.1 s 57.9, -0.05 s 56.2, 0 s 58.3, +0.05 s 63.1, +0.1 s 67.4, +0.2 s 67.1. There is no sharp optimum (a shift of +-0.1 s changes the median by at most 9 px, and B's curve alone has its minimum at -0.2 s): weak support for the clock offset, not a measurement of it.
- Conventions (`scripts/nymeria_hand_diagnose.py`, with A's second-pass labels): the current one (cw90, rotation as stored, left = segment 14, right = segment 10) is the best of 32 combinations, 67.2 px with outside-image points counted as misses; the next best is 161 px. So there is no convention error; the earlier suspicion of one came from the invalid first pass.
- Spread (consensus, raw, hands that project into the image; medians in cm): by sequence hayley 8.2 (11 hands), breanna 9.3 (14), tammy 13.9 (14), samantha 12.4 (2), elizabeth 17.6 (4); by side left 8.7 (23), right 13.6 (22); by narrated pose uniform 9.4 (37), kneel 15.0 (6), sit_ground 7.2 (1), squat 11.9 (1). The headline rests on three sequences with 11 to 14 hands; samantha, elizabeth and robert have too few hands for a statement of their own, and sit, squat and lie have no usable sample.
- Robert (free d, 3 points, not in the headline): median about 153 to 158 px (about 20 cm) with every label set.
- Six labelled hands project outside the image (not in the medians). With A's labels all six are in the lower part of the image (y 684 to 794 of 800), three of them within 40 px of the bottom edge and three 76 to 116 px from it. The medians are therefore optimistic for hands held low.

Limits: two labellers, 51 points, 5 sequences, one person's glove-centre click rather than a surveyed point. The residual includes the head-to-world alignment error (heading wanders 2 to 17 degrees, which at 50 to 60 cm arm length is several centimetres; this is an estimate, not measured on these points), the offset between the Xsens hand segment origin and the glove back (not checked), and the click error above. What it supports: an Xsens hand can be placed in the egocentric frame to roughly 10 cm median (7.6 cm after a constant offset), 18 cm at the 90th percentile, in these 5 sequences. What it does not support: a centimetre-accurate hand label, or any claim for other sequences. Robert and jeffery are excluded from spatial claims by the rule above.

## 8. Download groups of one sequence (from the URL list)

14 groups, sizes in bytes for `20230706_s0_breanna_reese_act0_cfg468`: `video_main_rgb` 1.42 GB, `recording_head_data_data_vrs` 20.75 GB,
`recording_observer_data_data_vrs` 18.82 GB, `semidense_observations` 10.17 GB, `body_xdata_mvnx` 2.17 GB, `body_motion` 1.14 GB,
`recording_head` 0.69 GB, `recording_observer` 0.50 GB, `recording_lwrist` 0.35 GB, `recording_rwrist` 0.23 GB, plus licence, metadata
and the two narration files. The download links are signed and valid for 14 days from issue.

## 9. Not yet read

- Values of `frameRate` and `frameCount` in `xdata.npz`, the meaning of the four `foot_contacts` columns, and the mapping from the Xsens
  segments to a body model.
- The alignment of the sequence `20230816_s1_jeffery_bryant_act0_p5w199` (no correlation peak; cause unknown), and the low coarse value of `20230707_s1_elizabeth_tucker_act0_df444f` (0.633 against a refined 0.986, section 7.3).
- Whether projecting the aligned Xsens joints into the image lands on the visible hands and feet (the check that would make the alignment an evaluation-grade ground truth).
- Whether the preview video covers the full calibration field of view (checked by projecting a known joint onto a frame).
- The reader needed for `.vrs` files, if the raw group is ever taken.
