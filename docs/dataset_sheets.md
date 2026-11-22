# Dataset sheets

**Date:** 2026-11-21. **Purpose:** one sheet per dataset on disk, so that decisions about training, testing and signals are made from what the
files contain. Every line is tagged:

- **[measured]** read from the files on disk on 2026-11-21 (read-only, throw-away scripts, nothing saved);
- **[repo doc]** taken from an earlier audit in this repository, with the document named (not re-measured today);
- **[not verified]** a statement that is plausible but was not checked in the files; it must not be used as a fact.

Paths are under `D:\projects\hready_data\datasets\`. Person names are not recorded anywhere in this file (counts only).

## 0. Overview: which signal exists where

| Dataset | Real or synthetic | Egocentric view | Head pose | Wrists / hands | IMU | Eye gaze | Full-body ground truth | Floor / scene | Video or images |
|---|---|---|---|---|---|---|---|---|---|
| Nymeria | real | yes (glasses) | yes, 1 kHz SLAM | yes, two wrist devices with their own 1 kHz SLAM pose; Xsens hand segments | inside `.vrs` files, not readable here | yes, 30 Hz | Xsens suit, 240 Hz, 23 segments (not SMPL-X) | no mesh; floor only from the suit | preview video for 7 sequences only |
| EgoBody | real | yes (HoloLens) | yes, per image and about 60 Hz | sparse device hand tracking (11 to 22 percent of sampled rows) | none found in the files | yes, about 60 Hz | SMPL-X fitted from Kinect cameras, 30 fps | scene meshes, 15 scenes | yes, 1920 x 1080 |
| AMASS (SMPL-X N) | real mocap, body only | no | no (can be computed from the joints) | finger poses (90 values) and wrist joints | no (this repo simulates it) | no | SMPL-X, 100 to 120 Hz | no | no |
| BABEL | labels only | no | no | no | no | no | action labels for AMASS clips | no | no |
| BEDLAM | synthetic renders | no (third-person cameras) | no | SMPL-X in the label files | no | no | SMPL-X, per image | no | yes, 2 scenes on disk |
| 3DPW (folders `3dpw` and `3spw`) | real, phone camera | no | no | SMPL has no fingers | raw IMU not in the files | no | SMPL fitted with IMUs and video, 30 Hz | no | yes (images in `3spw`) |

Reading this table: only Nymeria has head pose, wrist devices, gaze and full-body ground truth together on real data. AMASS is the only one with
a large body-motion volume. EgoBody is the only one with a floor mesh and a camera-wearer SMPL-X on real egocentric video.

---

## 1. Nymeria

**What it is.** Real egocentric recordings of people doing scripted daily activities, wearing smart glasses on the head, a device on each wrist
and a motion-capture suit. Licence CC BY-NC 4.0 [measured, `metadata.json`].

**On disk.** 206 sequences, 499.9 GB [repo doc, `nymeria_data_format.md`]. Per sequence: `body_motion.zip`, `recording_head.zip`,
`recording_lwrist.zip`, `recording_rwrist.zip`, `metadata.json`, and narrations for some. The full list of the dataset (`Nymeria_download_urls.json`)
names 1,100 sequences [measured]; the other 894 are not on disk.

**People.** The 206 sequences carry 206 different pseudonyms, and the pseudonyms of the full list number 236 [measured]. So the sequences on
disk are one per pseudonym. 193 different combinations of gender, height, weight and age group exist among the 206 [measured]; the files do not
say whether two pseudonyms are the same person. Gender 103 female and 103 male; height 149 to 199 cm (median 169); six age groups from 18 to 50 [measured].
29 sequences have two participants [measured].

**Size of the material.** Action time 53.9 h; per sequence 659 to 1,238 s (median 934 s). 16 scripts (the most frequent: cooking 50, "where is X" 33,
having a meal 31, welcome to my place 22) and 45 locations [measured].

**Signals and rates.**

| Signal | File | Rate | Notes |
|---|---|---|---|
| Head device pose (glasses) | `recording_head/mps/slam/closed_loop_trajectory.csv` | 1000 Hz [measured, first 2000 rows of one sequence] | position, quaternion, device velocities, gravity in the world frame, quality score |
| Head camera calibration | `online_calibration.jsonl` | per log line | RGB camera and two SLAM cameras; the facing direction is the RGB camera axis, about 38 degrees right of the device z axis (`decisions.md`, amendment A2) |
| Eye gaze | `mps/eye_gaze/general_eye_gaze.csv` | 30 Hz [measured] | yaw, pitch, depth in the head frame; personalized gaze exists in 184 of 206 sequences [measured] |
| Left and right wrist device pose | `recording_lwrist` and `recording_rwrist`, `mps/slam/closed_loop_trajectory.csv` | about 1000 Hz [measured, one sequence] | each wrist device has its own SLAM log. **Whether its world frame and clock equal the head's is not verified** |
| IMU of head and wrist devices | `data/motion.vrs` | not measured | `.vrs` files; no reader installed here [not verified: rate and axes] |
| Eye-tracking camera stream | `recording_head/data/et.vrs` | not measured | |
| Full-body motion | `body/xdata.npz` | `frameRate` value 240; median timestamp spacing 4,200 microseconds in the sequence read [measured] | 23 segments (position, quaternion, velocity, acceleration), 22 joints, 17 sensors [measured]; this is Xsens, not SMPL-X |
| Foot contacts of the suit | `foot_contacts` in `xdata.npz` | per body frame | 4 columns of 0 and 1 [measured]; the share of ones per column in the sequence read was 0.82, 0.04, 0.65, 0.24; **the meaning of the columns is not read** |
| Narrations | `narration_atomic_action.csv`, `narration_activity_summarization.csv` | sentences with start and end time | 169 and 175 sequences [measured]; the wearer is written as "C" |
| Egocentric video | preview RGB video | 30 fps, 1408 x 1408 | **only for 7 sequences** [repo doc]; the raw head video (`data.vrs`, about 21 GB per sequence) was not downloaded |

**Where the labels come from.** There is no label file for foot contact. The first Nymeria protocol builds a foot-lock label from the Xsens foot
speed after aligning the suit clock to the head clock (`nymeria_foot_lock_protocol.md`). Floor-work and activity words come from the narrations
(counted per sentence, not checked against motion, `nymeria_data_format.md` section 3).

**Timing and space (measured earlier, [repo doc] `nymeria_data_format.md` 7.2 and the tracker note).** The suit clock is not the head clock; the offset
is found per sequence by matching head rotation speed, and was resolved with a sharp peak in 6 of 7 sequences tried. Projecting the suit hand onto
the glove in the image misses by a median of about 10 cm (51 hands, 5 sequences, two labellers): the suit is **not** a centimetre-accurate hand ground truth.

**Limits.**
- Body ground truth is a suit with drift and a calibration; it has no mesh and no SMPL-X parameters.
- No scene mesh and no floor file; the floor height comes from the suit.
- IMU signals are not yet read. Wrist-device signals were checked on 2026-11-21 (see `docs/decisions.md`): the check did **not** show a shared clock and frame with the head device (median correlation 0.66 at the best lag; lags spread from about -1 s to +1.7 s); no wrist signal is used as an input yet.
- Most of the 206 sequences are standing, walking and kitchen work; floor work is concentrated in a few sequences (Game night, Body stretch) [repo doc].
- Current rule (decision 2026-11-21, `docs/decisions.md`): the 60 foot-lock cohort sequences stay **test only**; the 137 untried sequences may be used for training and adaptation (3 that share a metadata combination with a test sequence are excluded, and each must first pass the data check). Protocol: `docs/nymeria_foot_lock_learned_protocol.md` (approved 2026-11-21).

---

## 2. EgoBody

**What it is.** Real egocentric recordings with a HoloLens worn by one person while a second person interacts with them, with scene meshes and
SMPL-X fits from several Kinect cameras. All facts below are from `docs/egobody_data_format.md` (measured 2026-10-09 and 2026-10-11) [repo doc].

**On disk.** `datasets\egobody\extracted\`: calibrations, gaze, scene meshes, wearer and interactee SMPL-X for train, val and test [measured today: folder listing].
The colour images are a 79.2 GB zip that is not extracted [repo doc].

**People and size.** 125 recordings, 36 distinct participants, 2.03 h of wearer SMPL-X (219,731 frames at 30 fps); train 65, val 17, test 43 recordings.
Subjects overlap between splits (train and val share 10, train and test 13 participants in any role); the wearer identity is not stated in the files [repo doc].

**Signals and rates.**

| Signal | Rate | Notes |
|---|---|---|
| RGB images, 1920 x 1080 | about 30 fps, about 9 percent of frames missing | 199,111 images; the wearer's own body is in view in at most 0.3 percent of frames |
| Head (camera) pose | per image, and about 60 Hz in the gaze file | HoloLens tracking output, not an independent ground truth |
| Hand tracking | about 60 Hz, mostly empty | non-zero in 11 to 22 percent of sampled rows |
| Eye gaze | about 60 Hz | flag set in 91 to 100 percent of sampled rows; internal layout not verified |
| Wearer SMPL-X (10 betas, finger poses as 12 PCA values) | 30 fps | fitted from the Kinect views |
| Interactee SMPL-X | 30 fps | the person seen in the image |
| 2D keypoints (BODY_25) | per image | belong to the interactee, not the wearer |
| IMU | not found in the files | [not verified] |

**Where the labels come from.** Fitted SMPL-X bodies (multi-camera fits); floor from the scene mesh with an uncertainty of a few centimetres
(the lowest foot sits 1 to 6 cm above the chosen floor in train and val, 3.8 cm below in the test scene).

**Limits.**
- 2 hours is small; the floor-work cohorts (kneel, lie) are missing on the subject-disjoint split (`decisions.md`).
- Gravity is not a file: it is the scene up axis, assumed gravity-aligned.
- Evaluation-only by decision (download exception, licence gap).
- No wearer 2D annotations; hands are not reliable ground truth.

---

## 3. AMASS (SMPL-X N)

**What it is.** A large collection of optical motion-capture sequences fitted with SMPL-X, from 21 source datasets, plus a yoga set (MOYO) [measured: folders].

**On disk.** Clips found by file count [measured]: ACCAD 272, BMLmovi 1,953, BMLrub 3,172, CMU 2,079, CNRS 81, DFaust 139, EKUT 353,
Eyes_Japan_Dataset 762, GRAB 1,350, HDM05 219, HumanEva 31, KIT 4,287, MoSh 96, PosePrior 38, SFU 51, SOMA 71, SSM 33, TCDHands 63, TotalCapture 42,
Transitions 111, WEIZMANN 2,227 (sum 17,430), and the yoga folders train 301, val 36, test 42, extra 9 (sum 388). The repo's cache has 17,355
clips and the subject-disjoint split gives train 13,860, val 2,223, test 1,272 [measured, `results/splits/splits.json`]; 19 tune subjects are part of train.

**Naming note.** The AMASS subset called **SOMA** (71 clips, a mocap set) has nothing to do with the SOMA-X body-model converter in this repository.

**Signals and rates.** Fields per clip [measured]: `poses` (165 values per frame), `pose_body` 63, `pose_hand` 90 (finger rotations), `pose_jaw`, `pose_eye`,
`root_orient`, `trans`, `betas` (16, or 300 in the yoga files), `gender`, `mocap_frame_rate`. Frame rate by subset, from 12 random clips each [measured]:
120 Hz for most (ACCAD, BMLmovi, BMLrub, CMU mostly, DFaust, Eyes_Japan, GRAB, HDM05, HumanEva, PosePrior, SFU, SOMA, Transitions, TCDHands), 100 Hz for CNRS, EKUT, KIT, WEIZMANN, 60 Hz for TotalCapture and some CMU clips,
59 Hz for part of SSM, 150 Hz for one TCDHands clip; yoga files 60 Hz. Median clip length from 2 s (DFaust) to 70 s (TotalCapture).
The repo resamples everything to 30 Hz and puts the floor at z = 0.

**Where the labels come from.** Foot contact, foot lock and support contact are computed by this repo from foot height and speed (thresholds recorded in
`decisions.md`); action words come from BABEL.

**What is not in it.** No head sensor, no IMU, no gaze, no video, no glasses noise. Synthetic IMU, a virtual head camera and noise are produced by
this repo's own simulators (`hready/data/imu.py`, `ego_observation.py`), so every signal except the body motion is a model of reality, not a measurement.

**Limits.** Studio motion, many sources with different marker sets and quality; known artefacts (frame-rate metadata, treadmill clips, floor offsets;
`dataset_challenges.md`). Wrist and head tracks are available only as joints of the fitted body, with no sensor error.

---

## 4. BABEL

**What it is.** Text action labels on AMASS clips [measured: `babel_v1.0_release`].

**Content.** `train.json` 6,615 sequences (3,985 with frame-level labels, 36,137 labelled segments), `val.json` 2,193 (1,356; 13,333),
`test.json` 2,084 (1,322; 12,297); `extra_train.json` 7,921 and `extra_val.json` 2,636 sequences use a different field layout (multiple annotators) and carry no
single sequence label [measured]. Every sequence has a sequence-level label in the three main files.

**Limits.** Labels are free text from annotators; the repo found disagreements and unconfirmed segments when it used them for floor-work cohorts
(`results/splits/cohort_counts.json`). Only for AMASS clips; no signal of its own.

---

## 5. BEDLAM

**What it is.** Synthetic renders of people with SMPL-X ground truth [measured: file fields].

**On disk.** Labels: 30 `.npz` files in `lockedhead_16b` (about 1.06 million person-image rows in total) and one training archive folder in `v1_1_10b` [measured]. Images: **two** scenes only,
`closeup_suburb_a_6fps` (250 sequence folders, 9,836 PNG files) and `orbit_bigOffice_6fps` (250 folders, 7,619 PNG files) [measured].
The label files of these two scenes hold 9,066 and 22,370 rows; rows and images are not the same count (the images are the 6 fps subset), so only the rows whose image exists are usable [not verified: the match was not computed].

**Signals.** Per row [measured]: image name, person box (`center`, `scale`), `pose_cam` and `pose_world` (165 values), `shape` (16), `trans_cam`, `trans_world`,
`gtkps` (171 keypoints), camera intrinsics and extrinsics, gender, hand boxes, projected vertices. Images 6 fps. No IMU, no gaze, no head sensor.

**Where the labels come from.** Rendering: the ground truth is exact by construction.

**Limits.** Third-person cameras, synthetic appearance, two scenes on disk, rotated close-up images. Used only for the perception stage (not started).

---

## 6. 3DPW (folders `3dpw` and `3spw`)

**What it is.** The folder `3spw` holds the 3DPW images (`imageFiles.zip`, 4.61 GB, 53,796 image files in 61 folders) and the same sequence files as `3dpw`;
`3dpw` has only the sequence files. The two copies of the readme and the `test` file `downtown_bus_00.pkl` are byte-identical, and the file names of train, validation and test are the same [measured].

**Content [measured].** 60 sequences (train 24, validation 12, test 24), 87 person tracks (34 + 16 + 37; two-person sequences 10, 4 and 13), 52,653 frames at 30 Hz (17,904 + 8,509 + 26,240), about 29 min.
Per sequence: SMPL poses (72), translations, 60 Hz versions, betas, genders, 2D poses, 3D joint positions (24 joints), camera extrinsics and intrinsics per frame, camera validity.
The image count (53,796) and the pose frame count (52,653) differ; not investigated.

**Where the labels come from.** SMPL fitted with body-worn IMUs and a phone camera (readme). The raw IMU streams are not in the files [measured: field list].

**Limits.** Hand-held or phone camera, not egocentric; SMPL, not SMPL-X; no fingers; few people (the number of different people is not in the files [not verified]); outdoor and street scenes.
Used by `hready/data/threedpw.py` (loader test only); not on the egocentric path.

---

## 7. Body models (not datasets)

SMPL-X `locked_head` (the single training body model), SMPL-X `v1_1` (only for one pretrained baseline), MANO hands [`models/`, repo doc `AGENTS.md` section 1].

---

## 8. What the facts say about each dataset's role (implications, not measurements)

| Dataset | Role the facts support | Role the facts do not support |
|---|---|---|
| Nymeria | the only real source of head + wrist + gaze + body together; test of anything that must work on real glasses data; training source for the 137 untried sequences (owner decision 2026-11-21) | any claim about hands to the centimetre; any claim about floor work from the sequences on disk (few) |
| AMASS | large body-motion volume for pretraining, with signals simulated from joints | any claim that a result holds on real sensors without a real test |
| BABEL | action words for AMASS | signals |
| EgoBody | real egocentric test with a floor mesh and a wearer SMPL-X | training (evaluation-only), floor-work evaluation (cohorts missing) |
| BEDLAM | perception training (third-person bodies) | head, wrist, IMU, gaze signals |
| 3DPW | an outside real-video check of a body-pose model | anything egocentric |

## 9. Not verified, to be checked before a decision depends on it

1. Whether the wrist devices' SLAM world frame and clock equal the head device's (needed before wrist signals are used as inputs). Checked 2026-11-21 on 60 sequences: not shown. Open: whether the cause is the clock or the frame (a per-device clock fit from rotation speed, then the same distance test, would separate them).
2. IMU rate and axes inside the `.vrs` files, and a reader for them.
3. The meaning of the four `foot_contacts` columns of the suit.
4. Whether two Nymeria pseudonyms can be the same person (needed for any subject-disjoint split inside Nymeria).
5. How many BEDLAM label rows have an image on disk.
6. The number of different people in 3DPW, and why its image count differs from its pose frame count.
7. Frame rate of every AMASS clip (here: 12 random clips per subset).
