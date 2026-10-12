# EgoBody evaluation protocol (recorded 2026-10-11)

Documentation only: no loader, metric or model code is added by this document. Every fact about the data comes from `docs/egobody_data_format.md`; the download and licence record is in `docs/decisions.md` ("EgoBody — download exception"). EgoBody is used **only for evaluation and noise calibration**: it is never used for training, and nothing is tuned on the report set. The licence terms as signed are still a recorded gap.

The task text calls the two protocols A and B; the descriptive names below are the ones to use in tables and file names.

## 1. Two protocols

**Wearer protocol (A, primary).** Subject: the camera wearer.

| | |
|---|---|
| Inputs (realistic main path) | device head pose per image (`pv.txt` / `trans_world2pv`, 30 Hz; the gaze CSV adds about 60 Hz), sparse hand tracking when present, floor height and gravity **estimated** from the evidence |
| Hand tracking presence (measured, every 20th gaze row, hand block non-zero) | calibration set: left 13.1%, right 16.7%, either 24.5%, both 5.4%; report set: left 13.5%, right 18.7%, either 25.7%, both 6.5%. Per recording, either hand: median 21.7% / 21.3%, min 4.9% / 3.3%, max 80.5% / 69.6%; no recording has zero hand rows. "Hand block non-zero" is an inference about the CSV layout (data-format doc, section 2), not a documented flag |
| Output | full-body completion (SMPL-X body joints), compared with the **wearer's** SMPL-X ground truth |
| Control rows | the same pipeline with floor and gravity **given** from the scene transform and mesh (section 4), labelled as given |
| Evaluation frames | frames that have both an egocentric image and a wearer SMPL-X pose (199,111 in total). `valid_frame` is **not** used as a filter because its meaning is undocumented; results restricted to `valid == True` are reported as a sensitivity |

**Visible-person protocol (B, secondary).** Subject: the **person seen in the egocentric image, who is another person than the wearer** (the `body_idx_fpv` person of `data_info_release.csv`, the interactee). This protocol does not evaluate the main-path system on the wearer; it is used to measure real detector noise.

| | |
|---|---|
| Inputs | the provided 2D keypoints (`keypoints.npz`, with confidence) and the camera pose of the same image |
| Reference | the interactee's SMPL-X ground truth projected into the image with the camera chain of the data-format doc (section 4), which that doc validates against these keypoints (60,749 of 60,763 confident keypoints fall inside the image) |
| Keypoint set | 25 slots in OpenPose BODY_25 order, **measured** (24 of 25 slots land on the expected SMPL-X point; slot 8, mid-hip, falls between the hips). The files do not document the order. The detector or annotation process that produced the values is **unknown** |
| Alignment | 175,592 of the 235,517 keypoint rows have an interactee SMPL-X frame and an image; 59,925 rows lie outside the SMPL-X frame range and are not used; 44,139 SMPL-X frames have no keypoint row. 34.6% of all keypoint entries have confidence exactly 0 |
| Used for | calibrating the observation model's 2D noise, missing-keypoint rate and confidence reliability (calibration set), then checking them once on the report set. These are noise measurements, not method scores |

Slots whose definition differs between the keypoint set and SMPL-X (1 neck, 8 mid-hip, and the hip slots 9 and 12, with median offsets of 35-47 px in the identity check) have their reference points fixed on the calibration set before the report set is used, and are reported separately from the other slots.

## 2. Subject-disjoint split

The official split is not participant-disjoint (data-format doc, section 3: 10 participants shared by train and val, 13 by train and test, 1 by val and test; wearer S05 in train and test, S03 in val and test). It is replaced for every EgoBody number by the split below; the "train/val calibration, test report" rule recorded on 2026-10-09 is superseded, its principle (calibration and report are separate) is kept.

**Construction.** Nodes are the 36 participant ids of the recording names (`recording_<date>_S<aa>_S<bb>_<nn>`). Every recording links its two participants, in any role, so the split does not depend on which name token is the wearer (that reading, "first token = wearer", is an inference from gender and beta consistency and is not used here). Connected components are assigned whole.

**Assignment rule (deterministic, fixed before any method result existed):** sort components by total wearer SMPL-X frames (end_frame - start_frame + 1 from `data_info_release.csv`) descending, ties by the smallest participant id; walk the list and give each component to the set with fewer frames so far, ties to calibration.

Raw output of the throw-away script (not committed):

```
participants 36, recordings 125, components 11
component recs=45 frames= 75780 participants=[S05 S06 S07 S09 S12 S13 S16 S19 S20] -> calibration
component recs=14 frames= 28350 participants=[S21 S22 S30]                         -> report
component recs=12 frames= 22220 participants=[S10 S11 S15 S17]                     -> report
component recs=18 frames= 21950 participants=[S03 S04 S08 S14 S18]                 -> report
component recs=11 frames= 20030 participants=[S28 S29]                             -> report
component recs= 5 frames= 10840 participants=[S33 S34]                             -> calibration
component recs= 5 frames= 10490 participants=[S01 S02 S23]                         -> calibration
component recs= 4 frames=  9850 participants=[S35 S36]                             -> report
component recs= 3 frames=  8761 participants=[S31 S32]                             -> calibration
component recs= 5 frames=  6590 participants=[S26 S27]                             -> report
component recs= 3 frames=  4870 participants=[S24 S25]                             -> calibration
calibration: recordings 61, frames 110741 (1.03 h), participants 18, scenes 10
report:      recordings 64, frames 108990 (1.01 h), participants 18, scenes 11
participants shared between sets (any role): []   union 36   recordings in both sets: none
```

The graph is **not** one giant component (the largest has 45 of 125 recordings, 36%), so no recordings are dropped and no fallback is needed. Consequences to keep in view: the largest component (S05 and eight partners, 45 recordings) is entirely in the calibration set; the calibration set has 39 female / 22 male wearers and the report set 20 / 44; the sets share six scenes (`cab_e`, `cab_h_tables`, `cnb_dlab_0225`, `seminar_d78`, `seminar_g110`, `seminar_h52`), so room-specific effects such as floor error are not independent between the sets; the report set contains 12 unordered participant pairs and the calibration set 13, so cluster bootstraps (section 5) have about a dozen clusters per set.

Recordings (the official split of each is not used):

```
calibration (61)
recording_20210910_S05_S06_01  recording_20210910_S06_S05_01  recording_20210910_S06_S05_02  recording_20210910_S06_S05_03
recording_20210911_S06_S07_01  recording_20210911_S06_S07_02  recording_20210911_S07_S06_01  recording_20210911_S07_S06_02
recording_20210911_S07_S06_03  recording_20210918_S05_S06_01  recording_20210918_S05_S06_02  recording_20210918_S05_S06_03
recording_20210918_S05_S06_04  recording_20210918_S05_S06_05  recording_20210918_S05_S09_01  recording_20210918_S06_S05_01
recording_20210918_S06_S05_02  recording_20210918_S06_S05_03  recording_20210918_S09_S05_01  recording_20210918_S09_S05_02
recording_20210918_S09_S05_03  recording_20210921_S05_S12_01  recording_20210921_S05_S12_02  recording_20210921_S05_S12_03
recording_20210923_S05_S13_01  recording_20210923_S05_S13_02  recording_20210923_S13_S05_01  recording_20210929_S05_S16_01
recording_20210929_S05_S16_02  recording_20210929_S05_S16_03  recording_20210929_S05_S16_04  recording_20210929_S16_S05_01
recording_20211004_S12_S20_01  recording_20211004_S12_S20_02  recording_20211004_S12_S20_03  recording_20211004_S12_S20_04
recording_20211004_S19_S06_01  recording_20211004_S19_S06_02  recording_20211004_S19_S06_03  recording_20211004_S19_S06_04
recording_20211004_S19_S06_05  recording_20211004_S20_S12_01  recording_20211004_S20_S12_02  recording_20211004_S20_S12_03
recording_20211004_S20_S12_04  recording_20220318_S33_S34_01  recording_20220318_S33_S34_02  recording_20220318_S34_S33_01
recording_20220318_S34_S33_02  recording_20220318_S34_S33_03  recording_20210907_S02_S01_01  recording_20220218_S02_S23_01
recording_20220218_S02_S23_02  recording_20220218_S23_S02_01  recording_20220218_S23_S02_02  recording_20220318_S31_S32_01
recording_20220318_S32_S31_01  recording_20220318_S32_S31_02  recording_20220225_S24_S25_01  recording_20220225_S24_S25_02
recording_20220225_S25_S24_01

report (64)
recording_20220215_S21_S22_01  recording_20220215_S21_S22_02  recording_20220215_S22_S21_01  recording_20220215_S22_S21_02
recording_20220215_S22_S21_03  recording_20220215_S22_S21_04  recording_20220315_S21_S30_01  recording_20220315_S21_S30_02
recording_20220315_S21_S30_03  recording_20220315_S21_S30_04  recording_20220315_S21_S30_05  recording_20220315_S21_S30_06
recording_20220315_S30_S21_01  recording_20220315_S30_S21_02  recording_20210921_S10_S11_01  recording_20210921_S10_S11_02
recording_20210921_S11_S10_01  recording_20210921_S11_S10_02  recording_20210929_S15_S11_01  recording_20210929_S15_S11_02
recording_20210929_S15_S11_03  recording_20211002_S15_S17_01  recording_20211002_S15_S17_02  recording_20211002_S17_S15_01
recording_20211002_S17_S15_02  recording_20211002_S17_S15_03  recording_20210907_S03_S04_01  recording_20210907_S04_S03_01
recording_20210907_S04_S03_02  recording_20210911_S03_S08_01  recording_20210911_S03_S08_02  recording_20210911_S08_S03_01
recording_20210911_S08_S03_02  recording_20210911_S08_S03_03  recording_20210923_S03_S14_01  recording_20210923_S14_S03_01
recording_20210923_S14_S03_02  recording_20210923_S14_S03_03  recording_20211002_S03_S18_01  recording_20211002_S03_S18_02
recording_20211002_S03_S18_03  recording_20211002_S03_S18_04  recording_20211002_S18_S03_01  recording_20211002_S18_S03_02
recording_20220312_S28_S29_01  recording_20220312_S28_S29_02  recording_20220312_S28_S29_03  recording_20220312_S28_S29_04
recording_20220312_S28_S29_05  recording_20220312_S28_S29_06  recording_20220312_S29_S28_01  recording_20220312_S29_S28_02
recording_20220312_S29_S28_03  recording_20220312_S29_S28_04  recording_20220312_S29_S28_05  recording_20220415_S35_S36_01
recording_20220415_S35_S36_02  recording_20220415_S36_S35_01  recording_20220415_S36_S35_02  recording_20220225_S26_S27_01
recording_20220225_S27_S26_01  recording_20220225_S27_S26_02  recording_20220225_S27_S26_03  recording_20220225_S27_S26_04
```

## 3. Ground-truth handling (evaluation-only exception)

`AGENTS.md` §1 makes SMPL-X `locked_head` (neutral, 16 betas) the single training body model. EgoBody ground truth is gendered SMPL-X with 10 betas, per frame. **Exception, evaluation only:** the ground-truth joints are computed with the **gendered `locked_head` files** (`SMPLX_MALE.npz` / `SMPLX_FEMALE.npz`, 10 betas, flat hands) chosen by the `gender` field, with the **per-sequence median** of the per-frame betas (the per-frame betas vary within a recording, largest per-coefficient std 1.02 wearer / 0.90 interactee), joints regressed from the posed vertices (the repo wrapper's definition), then mapped to the repo convention (section 4). The ground truth is never converted to the neutral model: the neutral model with zero-padded betas differs from the gendered one by 25.7-45.6 mm mean over the 22 body joints (up to 95 mm) in the three recordings checked, which would be a bias larger than several of the effects to be measured. The gendered model is used by evaluation code only and by no training or model code. The effect of per-frame versus median betas is measured once, on the calibration set, before the report set is used.

**Comparison.** Predictions come from the neutral `locked_head` model. They are compared on a **common joint set**: the 22 SMPL-X body joints (pelvis to wrists; hands, face and the extra landmarks are excluded). Pose metrics are computed root-relative (MPJPE, PA-MPJPE) and in the gravity-aligned floor frame (W-MPJPE, WA-MPJPE, RTE) with the existing `hready.metrics.pose` functions. No quantity that depends on body shape beyond joint positions (vertices, PVE) is reported on EgoBody.

## 4. Floor and gravity

Derivation (data-format doc, section 4): the scene frame is y-up; `p_scene = kinect12_to_world[scene] @ p_kinect12`; gravity is scene +y (the Kinect12 frame itself is tilted 6-13 degrees from vertical); the floor is the tallest 2 cm bin of mesh-vertex y within the lowest 0.6 m of each scene mesh; repo convention is `(x, y, z)_scene -> (x, -z, y - floor_y)`. The per-scene floor values, the share of vertices in the floor bin, and the gap between the mesh minimum and the floor bin are in the table of the data-format doc, section 4. This floor is an estimate good to a few centimetres; the lowest foot joint sat 1-6 cm above it in two checked recordings and 3.8 cm below it in the third.

Rules:

1. **Sensitivity.** Every quantity that depends on floor height (heights above floor, penetration, float, contact labels and contact F1, the geometry cohort classes, world-frame metrics) is reported at floor offsets of -5 cm, 0 and +5 cm. A comparison between two methods is called floor-robust only if its direction is the same at all three offsets.
2. **Weak floor.** A scene whose floor bin holds less than 10% of the mesh vertices has a weak floor. Only `seminar_g110` qualifies (5.1%; the next lowest are `seminar_d78_0318` 11.7% and `kitchen_gfloor` 11.8%). Its 4 recordings (1 calibration, 3 report) are **excluded from every floor-dependent quantity** and kept for floor-independent ones (root-relative pose metrics, keypoint noise). They are listed in every table that excludes them.
3. **Deep gap.** Scenes whose mesh minimum lies more than 5 cm below the floor bin (`seminar_d78` 7 cm, `seminar_d78_0318` 9 cm, `seminar_g110_0315` 7 cm, `seminar_g110_0415` 9 cm, `seminar_h53_0218` 11 cm, and the already excluded `seminar_g110`) may have a lower surface than the one picked; for these, floor-dependent quantities are also reported at -10 cm and +10 cm.
4. In the given-floor control rows the floor and gravity are exactly the values above; in the main-path rows they are estimated from the evidence and compared with the values above as the reference. The reference is itself an estimate and is described as such.

## 5. Metrics and what is reported on which set

| Quantity | Calibration set | Report set |
|---|---|---|
| Wearer protocol, pose: MPJPE, PA-MPJPE, W-MPJPE, WA-MPJPE, RTE (22 joints) | EgoBody is never trained on, so no development happens here; computed only to check the pipeline end to end | **final numbers**, once |
| Wearer protocol, physical: foot skate, ground penetration, foot float, jitter, acceleration error (`hready.metrics.physical`), contact precision/recall/F1 (`hready.metrics.contact`, labels from the repo's height-and-speed hysteresis rule applied to the GT body and the mesh floor) | as above | final, with the floor sensitivity of section 4 |
| Wearer protocol, support: sit_support detection (the cohort tree on predicted versus GT joints) | as above | final |
| Observation noise: hand-wrist offset against GT wrist when hand tracking is present, head-camera to GT-head offset statistics (device output versus the fitted head joint, **not** a head-pose error), floor/gravity estimate versus the section 4 reference | **fitted here**: these are the only parameters taken from EgoBody | checked once |
| Visible-person protocol: 2D keypoint error in pixels and relative to body size, missing rate (confidence 0), confidence reliability (`expected_calibration_error`), per slot | **fitted here** | checked once |

Intervals: `hready.metrics.stats.cluster_bootstrap_ci` and `paired_cluster_bootstrap_diff`, cluster = the unordered participant pair of the recording name (13 clusters in calibration, 12 in report). Because that is few clusters, the intervals are reported together with the cluster count, and a coarser bootstrap by connected component (6 clusters in the report set) is given as a sensitivity.

### Cohorts on the new split (ground-truth geometry only, no method output)

The pre-registered tree (`hready.data.ego_splits.classify_frame_decision_tree`, thresholds of `configs/ego_splits.yaml`, metres) applied to the GT after the assignment of section 2 was fixed. Weak-floor scene excluded; GT joints from the gendered model with median betas; "runs" are same-class runs of at least 1.0 s; floor offset 0, with the range over -5 / +5 cm in brackets.

Wearer ground truth (wearer-protocol subject):

| class | calibration frames (60 recordings, 109,451 frames) | runs / recordings | report frames (61 recordings, 106,140 frames) | runs / recordings |
|---|---|---|---|---|
| lie | 0 | 0 / 0 | 0 | 0 / 0 |
| crawl | 0 | 0 / 0 | 0 | 0 / 0 |
| kneel | 0 [0-47] | 0 / 0 | 12 [0-70] | 0 / 0 |
| sit_floor | 163 [72-362] | 1 / 1 [1-2 / 1-2] | 323 [13-935] | 3 / 2 [0-5 / 0-3] |
| sit_support | 27,284 [27,085-27,375] | 54 / 34 [53-55 / 33-34] | 17,077 [16,465-17,387] | 30 / 19 [25-30 / 17-20] |
| yoga_like | 1,217 [323-1,255] | 5 / 3 [1-5 / 1-3] | 1,043 [1,029-1,051] | 1 / 1 |
| none | 80,787 | 88 / 56 | 87,685 | 74 / 56 |

Visible-person ground truth (visible-person-protocol subject), calibration / report at floor offset 0: kneel 156 / 0 frames, sit_floor 122 / 182, sit_support 20,672 / 14,780, yoga_like 803 / 1,441, lie and crawl 0 / 0; the sit_floor report count ranges 46-1,502 and its runs 0-3 over +-5 cm.

**Which classes can be evaluated.** Not evaluable: **lie, crawl** (zero frames in both sets, both subjects); **kneel** (at most 12 wearer frames in the report set at offset 0, 70 at +5 cm, and no 1-second run at any of the three offsets); **sit_floor** (3 runs in 2 wearer recordings in the report set at offset 0, and the count moves between 0 and 5 runs with 5 cm); **yoga_like** (1 run in 1 wearer recording in the report set). The floor-work slice of the project (kneel plus lie) therefore **cannot be evaluated on EgoBody**; it stays a synthetic-data result. "Ordinary locomotion" is not a class of the tree: it is defined from BABEL keywords, which EgoBody lacks, so it is not reported; the `none` class mixes standing, walking and everything else and is not interpreted as locomotion. The only cohort distinction usable on real data is **sit_support versus the rest**: 30 runs in 19 report recordings (wearer), 25-30 runs under the +-5 cm floor shift.

## 6. Leak rules

1. The calibration set provides noise parameters only (section 5). The report set provides final numbers only.
2. No hyper-parameter, threshold, model choice, early stopping, floor rule or protocol detail is chosen by looking at report-set results. Rules in this document were fixed before any method result existed; a later change is a dated amendment that states why and which results were already seen.
3. No training on EgoBody, in any form, including fine-tuning, self-training on its images, or fitting a model on its keypoints. The visible person's keypoints are used for noise statistics only.
4. The report-set ground truth is looked at only through the section 5 cohort counts and the rules above. Disclosed so far: the identity of the 25 keypoint slots was measured on 4,958 frames sampled from all 125 recordings (identity only, no error computed); two of the three recordings used for the format checks of 2026-10-09 (`recording_20210911_S08_S03_01`, `recording_20211002_S03_S18_02`) are in the report set, and the cohort-tree counts of that date were computed on the 17 official-val recordings, 15 of which are in the report set. No method result existed at any of these points.
5. Calibration-set parameters are fixed by recording them (value, date, command) before the first report-set run; they are not refitted afterwards.
6. Shared scenes (section 2) mean floor-related findings on the calibration set partly transfer to the report set; this is stated next to any result that uses them.

## 7. Limits

- The ground truth is a fit to a multi-Kinect rig, not motion capture; its error is taken as about 4 cm (a figure supplied with the task, not measured here). It sets a floor on every joint-error comparison and on contact timing.
- Contact labels derived from the GT body and the mesh floor are expected to be **favourable by construction**: the fits and the floor come from the same scene reconstruction, so GT feet agree with the floor better than a real measurement would. Whether and by how much cannot be checked from the files taken; contact F1 on EgoBody is an upper-end estimate. The floor itself is good to a few centimetres (section 4).
- The head pose is device output (HoloLens tracking), not ground truth. The only independent reference is the wearer's SMPL-X head joint, 0.15-0.18 m from the camera centre; the offset statistics therefore mix a fixed mounting offset with fitting error and are not a head-pose error.
- Hand tracking is device output, present in about a quarter of sampled rows, and its layout in the CSV is inferred.
- `valid_frame.npz` has an undocumented meaning (175,592 of the 199,111 images are flagged); it is not used as a filter, only as a sensitivity.
- The keypoint values belong to another person than the wearer, their source is unknown, and the keypoint-set definition is measured, not documented; the reference points of slots 1, 8, 9 and 12 differ in definition from SMPL-X.
- The wearer's own body is almost never inside the egocentric image (wrists and ankles in view in at most 0.3% of frames in three recordings, projection validated, no occlusion test); the wearer protocol therefore tests inference from head and hand signals, not visual body recovery.
- 11 participant components, 36 participants and 125 recordings (about 2 hours): the sets are small, intervals are wide, and per-class results for rare classes do not exist.
- The licence terms as signed are not recorded (gap in `docs/decisions.md`).
