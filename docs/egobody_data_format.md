# EgoBody data format (inspected 2026-10-09)

Everything below was measured on the files under `D:\projects\hready_data\datasets\egobody\` with throw-away scripts (deleted after use; no loader or model code was added). Where a statement is an inference rather than a measurement it says so. The download decision, licence gap and evaluation-only protocol are in `docs/decisions.md` ("EgoBody — download exception").

The "world frame = Kinect camera 1" of the task text is, in the files, the frame called `kinect12` (calibration files `kinect_11to12_color.json`, `kinect_13to12_color.json`, `kinect12_to_world/<scene>.json`). This document calls it the **Kinect12 frame**; the SMPL-X ground truth lives in it.

## 1. Archives

| Archive | Size | Content |
|---|---|---|
| `egocentric_color.zip` (not extracted) | 79.2 GB | 199,487 files: 199,111 `.jpg`, 125 `_pv.txt`, 125 `keypoints.npz`, 125 `valid_frame.npz`, 1 `.tar` |
| `egocentric_gaze.zip` | 0.47 GB | one `*_head_hand_eye.csv` per recording (125 files) |
| `scene_mesh.zip` | 0.23 GB | 15 scenes, `<scene>/<scene>.obj` plus texture |
| `calibrations.zip` | 0.4 MB | per recording `cal_trans/*.json` |
| `transf_matrices_all_seqs.pkl` | 84 MB | per recording transforms (section 4) |
| `smplx_camera_wearer_{train,val,test}.zip` | 0.23 / 0.06 / 0.15 GB | one `.pkl` per frame, wearer only |
| `data_info_release.csv`, `data_splits.csv` | small | scene, body genders, frame range, split lists |

## 2. Layout

**Egocentric colour** (listed from the zip central directory):
`egocentric_color/<recording>/<YYYY-MM-DD-HHMMSS>/PV/<ticks>_frame_<NNNNN>.jpg`, plus in `<recording>/<timestamp>/`: `<timestamp>_pv.txt`, `keypoints.npz`, `valid_frame.npz`. Recording names are `recording_<date>_S<aa>_S<bb>_<nn>`; 125 recordings, one timestamp folder each.

- Images: 1920 x 1080 JPEG (about 350 KB each); `pv.txt` line 1 states width 1920 and height 1080 for all 125 recordings.
- `<ticks>` is a Windows FILETIME (100 ns units). `NNNNN` is the frame number, **identical to the SMPL-X frame number** (section 3).
- One recording (`recording_20211002_S18_S03_02`) also holds `PV/PV.tar` (143 MB, 329 members). All 329 members are also present as loose JPEGs, so the tar is redundant.
- `<timestamp>_pv.txt`: line 1 `cx,cy,w,h` (for example `945.911,510.013,1920,1080`; read as principal point and size, which fits the image centre). Each further line: `ticks, fx, fy, <16 numbers>`, the 16 numbers being a row-major 4x4 **PV-camera-to-HoloLens-world** pose (last row `0 0 0 1`). Over 4036 / 3336 / 4685 lines (three sequences checked) `pv.txt` equals the inverse of `trans_world2pv[ticks]` from the pickle to within 7e-16 / 2e-15 / 2e-15. `fx, fy` differ between recordings (first-row values 1451 to 1467 in the three checked); whether they change within a recording was not checked.
- `keypoints.npz`: `imgname`, `center` (N,2), `scale` (N,), `keypoints` (N,25,3) (x, y, confidence), `gender` (N,). **These belong to the interactee, not the wearer**: in all 65 recordings where wearer and interactee genders differ, `gender` equals the interactee's.
- `valid_frame.npz`: `imgname`, `valid` (bool). Its meaning is not documented in the files; keypoint rows equal the number of `valid == True` entries. Both files cover more frames than the SMPL-X range (see counts).

**Wearer SMPL-X:** `smplx_camera_wearer_<split>/<recording>/body_idx_<k>/results/frame_<NNNNN>/000.pkl`. Exactly one `body_idx_<k>` folder per recording, one `.pkl` per frame, no gaps in frame numbers in any of the 125 recordings, and the frame range equals `start_frame..end_frame` of `data_info_release.csv` in all 125. The folder index is always the opposite of `body_idx_fpv` in `data_info_release.csv`, and the wearer gender in the pickle equals the `data_info` gender of that index in 125/125; so `body_idx_fpv` names the person seen in the egocentric view (the interactee), not the wearer.

Fields of each pickle (shapes as stored):

| Field | Shape | Note |
|---|---|---|
| `betas` | (1,10) | 10 shape coefficients, **not constant within a sequence** (largest per-coefficient std over a sequence: 1.02 over all 125; 0.60 / 0.35 / 0.33 in the three checked) |
| `global_orient` | (1,3) | axis-angle, in the Kinect12 frame |
| `transl` | (1,3) | Kinect12 frame |
| `body_pose` | (1,63) | axis-angle, 21 body joints |
| `left_hand_pose`, `right_hand_pose` | (1,12) each | PCA coefficients (12), not the 45-d axis-angle the repo wrapper expects |
| `jaw_pose`, `leye_pose`, `reye_pose` | (1,3) each | |
| `expression` | (1,10) | |
| `pose_embedding` | (1,32) | latent of the pose prior used in fitting |
| `gender` | string | `male` (66 recordings) or `female` (59) |

There is no timestamp field; time comes from the frame number at 30 fps.

**Gaze / tracking:** `egocentric_gaze/<recording>/<timestamp>/<timestamp>_head_hand_eye.csv`, no header, 861 columns, about 60 Hz (median over recordings 58.4 Hz, min 29.9, max 60.0). Layout **inferred from column counts and zero patterns, not from documentation**: column 0 ticks; columns 1-16 a row-major 4x4 head pose (same layout as `pv.txt`; equality with `pv.txt` not checked); columns 17-850 two blocks of 417 = 1 + 26 x 16 (hand tracking, left then right); columns 851-860 the eye-gaze block (column 851 is 1 in 91-100% of sampled rows; internal layout not verified).

## 3. Frames, rates, alignment, counts

- SMPL-X frames: 30 fps (frame numbers consecutive). Images: 33.32 ms per +1 frame number (30.01 fps) in the three checked recordings; 95.4% / 73.5% / 96.7% of consecutive image pairs differ by one frame number, so frames are dropped in places (about 91% of SMPL-X frames have an image overall).
- Alignment: image frame `NNNNN` = SMPL-X `frame_NNNNN`. In every one of the 125 recordings all images lie inside the SMPL-X frame range; for the three checked recordings the first and last image frame equal the SMPL-X first and last frame. Every image tick (all 199,111) is present in its recording's `pv.txt`; in the three checked recordings every image tick is also a key of `trans_world2pv`.
- Sequence length: minimum 340, median 1800, maximum 3930 frames (11 s, 60 s, 131 s).

| | train | val | test | total |
|---|---|---|---|---|
| recordings | 65 | 17 | 43 | 125 |
| wearer SMPL-X frames | 116,630 | 29,140 | 73,961 | 219,731 (2.03 h) |
| images in the zip | 105,388 | 25,416 | 68,307 | 199,111 (90.6% of SMPL-X frames) |
| images flagged `valid` | 90,120 | 23,332 | 62,140 | 175,592 (88.2% of images) |
| `keypoints.npz` rows (interactee) | 120,745 | 31,129 | 83,643 | 235,517 |
| gaze CSV rows | 474,901 | 139,781 | 337,555 | 952,237 |
| wearer gender (recordings) | 43 male / 22 female | 8 / 9 | 15 / 28 | |
| participants in recording names | 28 | 10 | 21 | 36 distinct |

Scenes (recordings per split): train `seminar_h52` 12, `kitchen_gfloor` 12, `cab_h_tables` 6, `foodlab_0312` 6, `seminar_j716` 5, `seminar_d78` 4, `cab_g_benches` 4, `cnb_dlab_0215` 4, `seminar_g110` 3, 5 further scenes 1-2 each; val `seminar_g110_0315` 6, `cab_g_benches` 4, `cab_e` 2, `cnb_dlab_0215` 2, `seminar_h53_0218` 2, `cab_h_tables` 1; test `seminar_j716` 8, `seminar_d78_0318` 8, `cnb_dlab_0225` 7, `foodlab_0312` 5, `seminar_d78` 4, 5 further scenes 1-3 each.

### Subject overlap between splits (checked; the answer is: they overlap)

- Counting every participant named in a recording, in any role: train and val share 10 (`S02 S03 S10 S11 S14 S18 S21 S22 S23 S30`), train and test share 13, val and test share 1 (`S03`).
- Which name token is the wearer is not stated in any file. Measured: taking the **first** token as the wearer gives a consistent gender for all 35 wearer identities (0 conflicts) and the most consistent betas (mean within-person std of the first five median betas 0.156, against 0.179 for the second token and 0.280 for random choice); taking the second token gives 5 gender conflicts. Under the first-token reading the wearers per split are 19 / 4 / 14 (35 distinct), and **`S05` is a wearer in train (9 recordings) and test (7), `S03` in val (5) and test (3)**; train and val share no wearer. The first-token reading is the best-supported one, not a stated fact.

## 4. Transforms and gravity

Pickle `transf_matrices_all_seqs.pkl`: dict keyed by recording (125); each value has `trans_kinect2holo` (4x4) and `trans_world2pv`, a dict keyed by the **string** of the image tick (457,216 entries in total) holding a 4x4. Measured relations (three checked recordings):

- `trans_kinect2holo` equals `inv(calibrations/<rec>/cal_trans/holo_to_kinect12.json["trans"])` exactly (difference 0.0).
- `p_pv = trans_world2pv[tick] @ trans_kinect2holo @ p_kinect12` (points as homogeneous column vectors). Check: the camera centre obtained this way is 0.175 / 0.180 / 0.152 m (median) from the wearer's SMPL-X head joint (5th-95th percentile 0.14-0.22 m), which confirms that the SMPL-X body is the camera wearer and the chain is right.
- `p_scene = kinect12_to_world/<scene>.json["trans"] @ p_kinect12`, where "scene" is the frame of the `.obj` mesh. The file exists per recording (the scene name is the file name).
- **PV camera axes** (measured): the camera looks along **-z** (median cosine between the +z axis and the wearer's torso facing direction -0.99 in all three recordings) and +y points up (angle to scene +y 8-20 degrees). That is an OpenGL-style camera; an OpenCV-style projection needs y and z flipped.

**Gravity and floor.** The Kinect12 frame is a tilted y-down camera frame, not gravity-aligned: expressed in Kinect12 coordinates the scene up axis is `(-0.039, -0.985, -0.167)`, `(-0.030, -0.975, -0.219)`, `(-0.004, -0.994, -0.112)` for the three recordings, i.e. 9.9 / 12.8 / 6.4 degrees from Kinect12 -y. The **scene frame is y-up** (scene meshes: the vertical extent is 0.85-3.7 m and the tallest horizontal-plane peak sits at the lowest y; pelvis is 0.82-1.00 m above that plane when standing and 0.54-0.56 m when seated in the val and test recordings checked). Gravity direction is therefore not a file: it is `kinect12_to_world` row 1 (scene +y), under the assumption that the scene frame is gravity-aligned, which the pelvis and floor numbers below support but do not prove. A floor height per scene (the "floor" used here is the tallest 2 cm bin of mesh-vertex y within the lowest 0.6 m):

| scene | floor y (m) | mesh y min | share of vertices in the floor bin |
|---|---|---|---|
| cab_e | -0.713 | -0.743 | 0.159 |
| cab_g_benches | -0.154 | -0.204 | 0.352 |
| cab_h_tables | -0.316 | -0.366 | 0.278 |
| cnb_dlab_0215 | -0.112 | -0.142 | 0.449 |
| cnb_dlab_0225 | -0.150 | -0.180 | 0.356 |
| foodlab_0312 | -0.706 | -0.756 | 0.129 |
| kitchen_gfloor | -0.837 | -0.867 | 0.118 |
| seminar_d78 | -0.816 | -0.886 | 0.193 |
| seminar_d78_0318 | -1.009 | -1.099 | 0.117 |
| seminar_g110 | -1.686 | -1.816 | **0.051 (weak)** |
| seminar_g110_0315 | -0.728 | -0.798 | 0.164 |
| seminar_g110_0415 | -0.794 | -0.884 | 0.153 |
| seminar_h52 | -0.498 | -0.548 | 0.260 |
| seminar_h53_0218 | -0.781 | -0.891 | 0.138 |
| seminar_j716 | -0.891 | -0.921 | 0.182 |

Conversion to the repo convention (Z-up, floor at z = 0): `(x, y, z)_scene -> (x, -z, y - floor_y)`. The floor is an estimate good to a few centimetres: the lowest ankle/foot joint sits 1-6 cm above it in the train and val recordings and **3.8 cm below it** (median) in the test one (`seminar_d78_0318`; there the mesh minimum is 9 cm below the chosen floor bin). Any contact or height threshold computed from this floor inherits that uncertainty.

## 5. What is missing for the plan

| Need | Status | Where / note |
|---|---|---|
| Head (camera) pose | **present** | `pv.txt` and `trans_world2pv` per image, plus `*_head_hand_eye.csv` at about 60 Hz; HoloLens tracking output, so a measured pose with unknown error, not an independent ground truth. The only reference is the wearer's SMPL-X head joint 0.15-0.18 m from the camera centre |
| Hand tracking | **present, sparse** | gaze CSV, hand blocks non-zero in 11-22% of sampled rows (every 20th row); tracked by the device, not ground truth |
| Gaze | **present** | gaze CSV, flag column 851 set in 91-100% of sampled rows; layout of the block unverified |
| Wearer body visible in the image | **unknown / no annotation** | no wearer 2D keypoints or masks are released (`keypoints.npz` is the interactee). Geometric check on three recordings (project wrists and ankles with the chain above, no occlusion test, not checked against images): in the field of view in at most 0.3% of frames, i.e. the wearer's own body is essentially absent from the egocentric image |
| Contact labels (foot, support) | **absent** | none released; they would have to be derived from the SMPL-X ground truth and the mesh floor, with the floor uncertainty above |
| Gravity direction | **absent as a file** | derivable from `kinect12_to_world` (assumption in section 4) |
| Floor height | **absent as a file** | derivable from the scene mesh (estimate, a few cm) |
| Wearer ground truth quality | **unknown** | SMPL-X fitted from a multi-Kinect rig, not motion capture; penetration and foot-skate level not measured |
| Labels for ordinary locomotion or floor work | **absent** | no action labels (BABEL-like); see the cohort counts below |

## 6. Sanity checks (raw output)

Body model: the repo wrapper `load_body("locked_head")` is neutral with 16 betas and flat hands. EgoBody gives 10 betas, gendered, per frame. Handling used here: 10 betas zero-padded to 16 (identical to a 10-beta model when padded this way), per-sequence median betas, neutral model, hands flat (the body joints do not depend on hand pose), jaw/eyes/expression ignored. **The mismatch is not negligible**; against the gendered `locked_head` file with the same 10 betas, 40 frames per recording:

```
=== train recording_20210911_S08_S03_01 (male, seminar_h52)
 male model vs repo neutral+zero-padded-16: body-joint dist mean 47.3 mm max 95.8 mm | vertex mean 57.6 mm max 113.7 mm | pelvis position diff mean 30.9 mm
=== val recording_20211002_S03_S18_02 (female, cab_g_benches)
 female model vs repo neutral+zero-padded-16: body-joint dist mean 25.2 mm max 59.6 mm | vertex mean 17.6 mm max 57.7 mm | pelvis position diff mean 6.9 mm
=== test recording_20220318_S31_S32_01 (female, seminar_d78_0318)
 female model vs repo neutral+zero-padded-16: body-joint dist mean 30.4 mm max 70.3 mm | vertex mean 28.1 mm max 77.2 mm | pelvis position diff mean 7.9 mm
```

Forward kinematics and floor, three recordings (heights in metres above the mesh floor of section 4; `lowest foot` is the lowest of left/right ankle and foot joints; columns are [min, p5, median, p95, max]):

```
train recording_20210911_S08_S03_01  scene=seminar_h52  frames=2890
  pelvis      [0.818, 0.837, 0.963, 0.983, 1.004]
  lowest foot [0.03, 0.04, 0.06, 0.124, 0.13]
  camera centre to wearer head: median 0.175 m (p5 0.157, p95 0.192), 2736 image frames
val   recording_20211002_S03_S18_02  scene=cab_g_benches  frames=1300
  pelvis      [0.538, 0.546, 0.879, 0.908, 0.919]
  lowest foot [-0.022, -0.007, 0.012, 0.069, 0.138]
  camera centre to wearer head: median 0.180 m (p5 0.145, p95 0.223), 902 image frames
test  recording_20220318_S31_S32_01  scene=seminar_d78_0318  frames=2751
  pelvis      [0.558, 0.567, 0.923, 0.934, 0.944]
  lowest foot [-0.06, -0.054, -0.038, -0.017, -0.003]
  camera centre to wearer head: median 0.152 m (p5 0.140, p95 0.160), 2646 image frames
```

Existing geometry cohort tree (`hready.data.ego_splits.classify_frame_decision_tree` with the frozen thresholds in `configs/ego_splits.yaml`, metre units, no normalisation) applied to the wearer ground truth of **val** (17 recordings, 29,140 frames at 30 fps, floor from the mesh, per-sequence median betas). "Runs" are maximal same-class frame runs of at least 1.0 s; subjects are wearers under the first-token reading of section 3.

```
  class        frames   frac   runs>=1s  sequences  wearer-subjects
  lie               0  0.000         0          0          0
  crawl             0  0.000         0          0          0
  kneel             0  0.000         0          0          0
  sit_floor       159  0.005         2          1          1
  sit_support    8573  0.294        15          8          4
  yoga_like         8  0.000         0          0          0
  none          20400  0.700        22         15          4
```

Shifting the floor by -5 cm / +5 cm changes `sit_floor` to 149 / 164 frames, `kneel` to 0 / 6 and `yoga_like` to 8 / 10, and leaves the other rows within 15 frames. **"Ordinary locomotion" is not a class of the tree**: it is defined from BABEL keywords (walk, run, stand, turn), which EgoBody does not have, so it is not reported. Consequence: val contains no kneel, lie or crawl at all and a single `sit_floor` recording, so the floor-work slice cannot be evaluated on EgoBody val (test was deliberately not examined with the tree).

## 7. Reproduction notes

Inputs are the files listed in section 1 plus `D:\projects\hready_data\models\smplx\locked_head\models_lockedhead` (neutral, male, female). The mesh floor is defined in section 4; the wearer mapping and overlap are defined in section 3; the wearer joints are `smpl_forward_bt` on per-sequence median betas zero-padded to 16, then the rigid maps of section 4. The checking scripts were not kept (project rule: checks are run, not saved).
