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
sequences had no egocentric frames. The preview video was added afterwards for the floor-heavy sequences (section 6); the raw
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

## 8. Download groups of one sequence (from the URL list)

14 groups, sizes in bytes for `20230706_s0_breanna_reese_act0_cfg468`: `video_main_rgb` 1.42 GB, `recording_head_data_data_vrs` 20.75 GB,
`recording_observer_data_data_vrs` 18.82 GB, `semidense_observations` 10.17 GB, `body_xdata_mvnx` 2.17 GB, `body_motion` 1.14 GB,
`recording_head` 0.69 GB, `recording_observer` 0.50 GB, `recording_lwrist` 0.35 GB, `recording_rwrist` 0.23 GB, plus licence, metadata
and the two narration files. The download links are signed and valid for 14 days from issue.

## 9. Not yet read

- Values of `frameRate` and `frameCount` in `xdata.npz`, the meaning of the four `foot_contacts` columns, and the mapping from the Xsens
  segments to a body model.
- Alignment between video, body motion, head trajectory and narration timestamps (which clock the video frame list uses).
- Whether the preview video covers the full calibration field of view (checked by projecting a known joint onto a frame).
- Whether the video of the other six floor-heavy sequences shows the same quality; sizes of their groups.
- The reader needed for `.vrs` files, if the raw group is ever taken.
