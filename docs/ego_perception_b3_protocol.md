# B3 protocol: training the egocentric perception variants

Status: written 2026-11-23, not run, waiting for owner approval. Part of E2-B (B3). Scripts: to be written after approval (`scripts/ego_perception_b3_train.py`, `scripts/ego_perception_b3_eval.py`).

## 1. Question
Which variant gives the best visible-joint evidence on the `aria_rect` render set: an own model from random init (P-A) or a pretrained backbone adapted on the renders (P-C)? B2 showed the pretrained detector as released (P-B) is far from enough (labelled recall 0.077 at 25.6 px, a person found in 4.9% of frames).

## 2. Common design (all trained variants)
- Input: the 512x512 render, whole frame, no person detector (the wearer's body is always seen from the same place; B2 lost most frames at the detection step). The given head pose and gravity enter later, when the output is converted to the evidence schema; they are not network inputs in this version.
- Output per frame, for the 22 joints: 2D heatmap (128x128, soft-argmax position in pixels), depth along the camera axis (metres), a visible/not-visible logit (visible = inside the image and z-buffer visible, as in B2), and a confidence taken from the visibility probability. The 3D point in the camera frame is ((u-cx) z/f, (v-cy) z/f, z) with the profile intrinsics; the given head pose turns it into the world frame (schema: visible-joint 3D, visibility, confidence, 2D keypoints).
- Loss: position and depth only on visible joints (L1 on pixels divided by 16, so a 16 px error counts 1.0; L1 on metres); binary cross-entropy on visibility for all 22 joints. Weights 1, 1, 1; not tuned. Gradient norm clipped at 1.0, weight decay 0.01.
- Data: train split of the render set (247,027 frames). VAL (31,013) for evaluation and selection. TEST (16,494) not used until B4.
- Augmentation: brightness and contrast, Gaussian noise, small blur. No flips, no crops (the pixel position is the label).
- Training budget, equal for every variant: batch 32, 30,000 steps, AdamW, cosine schedule with 1,000 warm-up steps, bf16, learning rate 1e-4 (backbone) and 1e-3 (new layers) for P-C, 5e-4 for P-A. These are fixed here and not tuned on VAL. Seeds 0 and 1 for each variant. A 200-step timing run comes first; if 2 seeds of 2 variants do not fit in the time the owner allows, the seed count is reduced and stated.

## 3. Variants
| Name | Backbone | Initialisation | Label in every table |
|---|---|---|---|
| P-B | Keypoint R-CNN as released | COCO | pretrained, as released; 2D only, no depth, no 3D schema; measured in B2, not retrained |
| P-C | ResNet-50 with FPN from the same COCO Keypoint R-CNN weights, new head on the stride-4 level | COCO, then adapted on the renders | pretrained, adapted |
| P-A | ViT-S/16 encoder (HR-HMR-style), two upsampling layers to stride 4, same head | random | own, from scratch, trained on the egocentric renders only |

P-A here is not BEDLAM-pretrained: items 12 and 13 (BEDLAM loader, HR-HMR from scratch) are not done, so the "BEDLAM-pretrained by us" step of the project definition is not part of this first run. The label "from scratch, ego renders only" must be used for it.

**Fourth variant P-A' (own, BEDLAM-pretrained by us).** Same encoder, same head, same training budget as P-A; only the initial weights come from our own BEDLAM pretraining (items 12 and 13). It is added by a written amendment to this protocol when the pretrained weights exist; nothing else in this protocol changes. **B4 is not run until P-A' exists or the owner records that it is dropped.** Reason: TEST is used once, by B4, for the final selection; a variant added after B4 would force a second use of TEST. Until then every result is VAL only. A claim of the form "own model, BEDLAM-pretrained" is made only after P-A' is trained and evaluated; before that the own model is P-A with its label.

## 4. Measures (all on VAL, visible joints by the z-buffer rule)
1. Visible-joint 3D error in the camera frame (cm): mean and median, all 22 joints and by group (hands/wrists, arms, legs, trunk and head).
2. 2D error (pixels) and recall at 12.8 and 25.6 px on the 12 joints of B2, labelled, to compare with B2 directly.
3. Visibility AUC over all 22 joints; share of visible joints with confidence above 0.5.
4. Gap: the same measures on 3,000 train frames (fixed by seed 0) against VAL.
Everything also without the clips of subject ids 50021, 50022, 50027 of DFaust and MoSh. Nothing is tuned on VAL; TEST untouched.

## 5. Reading (fixed before the run)
- If no trained variant reaches a labelled 2D recall at 25.6 px above the B2 value (0.077), the training pipeline is treated as broken and checked (labels, loss, input) before anything is reported.
- If the VAL mean 3D error of a variant is more than twice its train-frame error, that variant is reported with a generalisation gap and not described as converged.
- The selection criterion for B4 is fixed here: lowest mean visible-joint 3D error (cm, 22 joints, mean over the two seeds) on VAL, all clips. The seed spread is reported next to it; if the two best variants differ by less than that spread, B4 says so and selects by the criterion anyway. B4 is run once, after B3, and emits the evidence for train, val and test.

## 6. Allowed wording
"On synthetic flat-colour head-camera renders of AMASS motion, [variant, with its label] locates visible wearer joints with mean 3D error X cm on VAL." Not allowed: statements about real egocentric images, about the full BEDLAM-pretrained own model, or about test before B4.

## 7. Limits
Synthetic flat-colour domain, one rectified pinhole camera (real Aria is a fisheye), hands at the mean SMPL-X hand pose, one renderer for train and test (results are optimistic about real footage), head pose and gravity not network inputs, P-A not BEDLAM-pretrained, two seeds.

## 8. Result, seed 0 (run 2026-11-23; `results/E/perception/b3_PC_seed0_val.json`, `b3_PA_seed0_val.json`, script sha1 b74993d959ea4442)
Both variants trained 30,000 steps on the train split and evaluated on VAL (31,013 frames, 21,514 visible joint instances; 660,772 joint instances not visible). Seed 1 is not run yet; no selection is made (that is B4).

| | P-C (COCO backbone, adapted) | P-A (ViT-S from scratch, ego renders only) |
|---|---|---|
| 3D error, visible joints, mean / median (cm) | 4.20 / 3.06 | 5.59 / 4.16 |
| by group, mean (cm): hands / arms / legs / trunk and head | 3.3 / 2.9 / 5.5 / 3.4 | 4.5 / 3.6 / 7.3 / 5.7 |
| 2D error mean / median (px) | 1.8 / 0.7 | 4.4 / 2.1 |
| depth error mean (cm) | 2.9 | 3.8 |
| B2-comparable 12 joints, recall at 25.6 px (B2 as released: 0.077) | 0.994 | 0.971 |
| visibility AUC | 0.9998 | 0.9986 |
| 3D mean on 3,000 train frames (cm) | 3.41 | 4.88 |
| VAL / train 3D error | 1.23 | 1.15 |
| without DFaust/MoSh shared ids, 3D mean (cm) | 4.20 | 5.60 |

Reading by section 5: both variants are far above the B2 recall (0.077), so the pipeline is not broken; both VAL/train ratios are well below 2, so no generalisation gap is reported. On seed 0 P-C has the lower 3D error; the seed spread is unknown until seed 1.

Why the numbers are this high, and what they do not show: the renders are flat colours with one colour per body region, so the image almost names the joint; train and VAL use the same renderer and body model; 97% of the joint instances in the visibility AUC are far-out-of-image or plainly absent, which makes the AUC easy (it does not separate occluded from visible joints inside the image). These are numbers for synthetic flat-colour frames only, an upper bound for real footage; nothing here is evidence for real egocentric images.

## 9. Result, seed 1 and the two-seed summary (run 2026-11-24; `b3_PC_seed1_val.json`, `b3_PA_seed1_val.json`, same script sha1 b74993d959ea4442)
| VAL, visible joints | P-C seed 0 | P-C seed 1 | P-A seed 0 | P-A seed 1 |
|---|---|---|---|---|
| 3D error mean (cm) | 4.205 | 4.253 | 5.593 | 5.552 |
| 3D error median (cm) | 3.06 | 3.08 | 4.16 | 4.21 |
| 2D error mean (px) | 1.81 | 1.85 | 4.44 | 4.29 |
| recall at 25.6 px (12 joints of B2) | 0.994 | 0.994 | 0.971 | 0.970 |
| VAL / train 3D error | 1.23 | 1.25 | 1.15 | 1.12 |

Mean over the two seeds: P-C 4.23 cm (seeds differ by 0.05 cm), P-A 5.57 cm (seeds differ by 0.04 cm). The difference between the variants (1.34 cm) is about 30 times the seed difference. By the criterion fixed in section 5 (lowest mean visible-joint 3D error on VAL, mean of the two seeds) P-C is ahead of P-A; this is not the B4 selection, which also waits for P-A' (section 3). Both rules of section 5 pass for both seeds. Without the DFaust/MoSh shared ids the means are unchanged (4.20 to 4.25 and 5.60).

One difference between seeds: the trunk-and-head group of P-C is 3.42 cm (seed 0) and 5.26 cm (seed 1); the other groups differ by at most 0.1 cm. The group has few visible instances and is not analysed further here.

Same limits as section 8: synthetic flat-colour frames, one renderer for train and VAL, two seeds, P-A not BEDLAM-pretrained.
