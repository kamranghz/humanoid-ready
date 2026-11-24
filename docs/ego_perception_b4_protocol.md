# B4 protocol: selection of the perception variant, one use of TEST, evidence

Status: written 2026-11-24, not run. Part of E2-B (B4). Script: `scripts/ego_perception_b4.py`. Evidence converter: `hready/data/ego_evidence.py`.

## 1. Decision recorded before TEST is read
- **Owner decision 2026-11-24:** B4 runs now with the variants trained in B3 (P-C, P-A) and the B2 baseline (P-B, 2D only, not eligible for the 3D evidence). The own BEDLAM-pretrained model (P-A', items 12 and 13) is not part of this selection. If it is built later it is reported as an extra row, labelled "added after selection", with VAL numbers only; it cannot replace the selected model, because TEST is used once.
- **Criterion (fixed in `docs/ego_perception_b3_protocol.md` section 5):** the variant with the lowest mean visible-joint 3D error on VAL, mean of the two seeds. VAL values: P-C 4.205 and 4.253 cm, P-A 5.593 and 5.552 cm (B3 section 9). The script recomputes the selection from the B3 result files: P-C is expected.
- **Checkpoint:** the seed of the selected variant with the lower VAL error (expected: P-C seed 0, 4.205 cm). Chosen on VAL, written here before TEST is read.

## 2. What one run does
1. Reads the four B3 VAL result files, applies section 1, prints the choice.
2. One forward pass of the selected checkpoint over every frame of the render set: val, test, train. No retraining, no tuning.
3. Metrics of B3 section 4 on val and on test (with and without the DFaust/MoSh shared ids of `docs/decisions.md`, 2026-11-23). Same code as B3 (`ego_perception_b3_eval.py`).
4. Per-clip evidence files in `<data_root>/ego_perception_b4/<split>/<clip key>.npz`: `frame`, `uv` (pixels), `z` (m), `vis_prob`, `xyz_cam` (camera frame, metres).
5. Result file `results/E/perception/b4_selected_val_test.json`, written once. If it exists the script stops before reading any test frame. TEST is therefore used once, in this run; if the run crashes before the file is written, it is started again with the same selection and the same checkpoint.

## 3. Evidence schema
`hready/data/ego_evidence.py: to_schema` turns the perception output and the given head pose into the schema of `docs/ego_observation_model.md`: `joint_pos_3d` (world, zero where not visible), `joint_visible` (visibility probability above 0.5, depth above 0.1 m; the 0.5 is fixed here, not tuned), `joint_confidence` (the probability where visible), `keypoints_2d` (normalised, same convention as the oracle path), and the rig (head position, camera rotation, camera translation, gravity). Test: on synthetic geometry the converter reproduces the world positions and agrees with `project_joints_egocentric` to 1e-4. The head pose is the given one (a rig input), not estimated.

Limit of this step: the render set holds every 20th train frame and every 30th val and test frame, so the evidence files are per-frame, not contiguous 64-frame windows. Perceived-evidence windows for completion (E5) need contiguous renders of chosen windows and are a separate step; B4 provides the per-frame perceived error that step is checked against.

## 4. Reading (fixed before the run)
- The numbers of val and test are reported side by side for the selected model only. A TEST error more than 25% above the VAL error is reported as a val-to-test gap; it does not trigger any change of the selection or retraining.
- Train-split evidence comes from a model trained on those frames and is more accurate than val and test evidence (B3: 3.4 against 4.2 cm for P-C). Anything trained on perceived train evidence must say so.

## 5. Allowed wording
"On synthetic flat-colour head-camera renders, the selected perception model (P-C, pretrained backbone, adapted) locates visible wearer joints with mean 3D error X cm on TEST." Not allowed: statements about real egocentric images, about the own BEDLAM-pretrained model, or about variants other than the selected one on TEST.

## 6. Limits
Synthetic flat-colour domain, one renderer for train, val and test, rectified pinhole camera (real Aria is a fisheye), per-frame evidence only, given head pose.

## 7. Result (run 2026-11-24 06:22; `results/E/perception/b4_selected_val_test.json`, script sha1 63c9d18da1037f70, checkpoint sha1 d9ad206cc478f909)
Selection by section 1: mean VAL 3D error P-C 4.229 cm, P-A 5.573 cm, so P-C; checkpoint P-C seed 0 (30,000 steps). One pass over 31,013 val, 16,494 test and 247,027 train frames; evidence files for 2,222 val, 1,271 test and 13,856 train clips in `ego_perception_b4/` outside the repo.

| P-C seed 0, visible joints | VAL | TEST | TEST without DFaust/MoSh shared ids |
|---|---|---|---|
| frames / visible joint instances | 31,013 / 21,514 | 16,494 / 14,496 | 16,416 / 14,469 |
| 3D error mean / median (cm) | 4.205 / 3.06 | 4.172 / 3.13 | 4.175 / 3.14 |
| by group, mean (cm): hands / arms / legs | 3.26 / 2.90 / 5.55 | 2.86 / 2.68 / 5.84 | 2.86 / 2.68 / 5.84 |
| 2D error mean / median (px) | 1.81 / 0.75 | 1.65 / 0.82 | 1.65 / 0.82 |
| recall at 25.6 px (12 joints of B2) | 0.994 | 0.997 | 0.997 |
| visibility AUC | 0.9998 | 0.9998 | 0.9997 |

Reading by section 4: the TEST error (4.17 cm) is not above the VAL error (4.21 cm), so there is no val-to-test gap; the shared ids change nothing. The trunk-and-head group has 33 visible instances on VAL and 6 on TEST (7.36 cm on TEST, 3.42 on VAL): it is practically never visible in this view and its number is not interpretable. The result is for the selected model only; P-A was not run on TEST.

Same limits as sections 5 and 6 and as B3 sections 8 and 9: synthetic flat-colour frames, one renderer for train, val and test, an upper bound for real footage.
