# B2 protocol: a pretrained 2D pose model on the egocentric render set

Status: written 2026-11-23, not run. Part of E2-B (B2: baseline measurement, labelled pretrained, not a gate). Script: `scripts/ego_perception_b2.py`.

## 1. Question
How well does an off-the-shelf COCO keypoint model locate the wearer's own joints in the synthetic head-camera renders (`aria_rect`, `docs/decisions.md` entry of 2026-11-23)? The answer sets how much B3 has to do; it does not decide whether B3 is done.

## 2. Model and data
- Model: torchvision `keypointrcnn_resnet50_fpn`, weights `COCO_V1` (59.1 M parameters, about 237 MB, downloaded once by torchvision from `download.pytorch.org` into the torch cache), used as released, no tuning. Per image the highest-scoring person detection is used; an image with no detection (box score below torchvision's default 0.05) counts as a miss.
- Data: the val split of the render set (31,013 frames), every frame. TEST is not touched (reserved for the model selected in B4). The train split is not used here.
- Images are flat-colour renders (no texture, no real appearance): a domain far from the training images of the model. This is stated in every use of the result.

## 3. Scored joints
The 12 joints present in both COCO and SMPL-X: shoulders, elbows, wrists, hips, knees, ankles (SMPL-X joints 16 to 21, 1, 2, 4, 5, 7, 8). Nose, eyes, ears, spine, neck, collars, feet and the head are not scored. A joint is scored when it is inside the image and visible by the z-buffer rule (`render_visibility`: nearest body surface at its pixel at most 0.10 m in front of the joint).

## 4. Measures (fixed before the run)
For the scored joints, pooled and by joint group, in pixels of the 512 px image:
1. Labelled error: distance between the prediction for that joint (COCO left/right as named) and the ground truth; median, mean, and recall at 12.8 px (2.5% of the side) and 25.6 px (5%). Recall is over all visible joints; a missing detection is a miss.
2. Side-agnostic error: the smaller of the distances to the joint's own prediction and to its left/right partner's prediction. It separates localisation from left/right naming, which a model trained on third-person images may get wrong in a first-person view. It is optimistic by construction.
3. Visibility discrimination: AUC of the keypoint score for visible versus not-visible (outside the image or occluded) joints, on frames with a detection. 0.5 is chance.
4. Share of frames with a detection.
Everything is also reported without the clips of subject ids 50021, 50022, 50027 of DFaust and MoSh (`docs/decisions.md`, 2026-11-23). No threshold is tuned on val.

## 5. Reading (fixed before the run)
- If labelled recall at 25.6 px over visible joints is below 0.5, B3 must include an adapted pretrained variant (P-C) next to the own model (P-A) and the pretrained baseline (P-B). If it is 0.5 or higher, P-C is optional.
- If side-agnostic recall is at least 0.2 higher than labelled recall, left/right naming is a main source of error and B3 must say how it is handled.
- Nothing here ranks models: one model, one split, no selection.

## 6. Allowed wording
"On synthetic flat-colour head-camera renders of AMASS motion, a COCO-pretrained Keypoint R-CNN locates visible wearer joints with median error X px (recall Y at 25.6 px)." Not allowed: any statement about real egocentric images, about other pretrained models, or about test.

## 7. Limits
One pretrained model; synthetic flat-colour domain; hands are the mean SMPL-X hand pose; one rectified pinhole camera; the real Aria camera is a fisheye; the z-buffer visibility label has a 0.10 m tolerance; the pretrained model is not trained for a body seen from its own head.
