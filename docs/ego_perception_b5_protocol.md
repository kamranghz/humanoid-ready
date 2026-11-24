# B5 protocol: perceived evidence for whole clips (contiguous frames)

Status: written 2026-11-24, not run, waiting for owner approval. Part of E2-B (B5, added by owner decision 2026-11-24). Script: to be written after approval (`scripts/ego_perception_b5.py`).

## 1. Why
B4 emitted evidence only for the strided frames of the render set (every 20th train, every 30th val and test frame). The completion models (E3, E4, E5) read whole clips at 30 Hz in windows of 64 frames (stride 32). B5 gives them perceived evidence for every frame of whole clips, from the model selected in B4 (P-C seed 0). No selection is made and no model is trained here.

## 2. What is produced
Per clip (all frames, 30 Hz, the clip as `load_clip` returns it, the same clip and frame index space as the render set): the selected model's per-frame output (`uv` pixels, `z` metres, `vis_prob` for the 22 joints) and, from the same render, the renderer's ground truth for the same frames (`gt_uv`, `gt_z`, `gt_visible` by the z-buffer rule). Camera-frame quantities only: the world frame comes from the head pose of the completion's own clip (`load_clip_memmap`, grounded) through `hready/data/ego_evidence.py: to_schema`, as the oracle path does. Images are rendered in memory and not saved.

Same rendering as the render set: `aria_rect`, flat region colours, floor plane at the 1st percentile of the per-frame lowest vertex height, per-clip seed from the clip path (the floor height is taken over all frames of the clip here, over the strided frames in the render set; a small difference, checked below).

## 3. Clips
- VAL: all clips of the val split (same list as the render set: 2,222). TEST: all clips of the test split (1,271). About 0.93 million and 0.49 million frames.
- TRAIN: a seeded subset of 1,500 clips (seed 0) of the completion training list (12,309 clips, `docs/ego_splits.md`), for training on perceived evidence in E4 and for the noise statistics. About 0.54 million frames. Evidence from train clips comes from a model trained on those frames' neighbours and is more accurate than val and test evidence (B3: 3.4 against 4.2 cm); anything trained on it says so.
- A timing run on 20 clips comes first. If the estimated total exceeds 24 h, the train subset is reduced (never val or test) and the new size is stated.
- Output outside the repo: `<data_root>/ego_perception_b5/<split>/<clip key>.npz`, resumable (a finished clip is skipped).

## 4. Checks (reported, no thresholds tuned)
1. Frame count of every clip equals the frame count `T` of the completion's motion cache; a mismatch stops the run.
2. On the frames of the render set grid, B5 predictions agree with the B4 evidence files: mean absolute difference of `z` and of the 3D point reported (expected well below 0.1 cm, the floor height is the only difference).
3. Head pose: `camera_R` and `head_pos` rebuilt from the clip agree with the oracle path's rig for one clip to 1e-5.

## 5. What is reported
Per frame, on all frames of val and test: 3D error of visible joints (cm, by group), visibility rates (share of frames with at least one visible joint, with three or more, per group), and the distribution of runs of frames without any visible joint (which the completion model has to bridge), next to the same numbers for the oracle evidence on the same camera. Descriptive only; nothing is selected or thresholded.

## 6. Camera mismatch to be aware of
The oracle completion baselines (E3) and the evidence-locked ablation were trained and evaluated with oracle evidence on the wide camera (focal 120 px, 129.8 degrees; any joint visible in 57% of frames). The perceived evidence uses `aria_rect` (110 degrees; 36%). For the oracle-to-perceived decomposition of E5 the oracle must use the same camera: `gt_visible` and the ground truth positions in the B5 files give "oracle on the perception camera". Existing completion models trained on the wide-camera visibility pattern are a different row; training or adapting a completion model on `aria_rect` evidence is part of E4, not of B5.

## 7. Allowed wording
"Perceived evidence for N clips (all frames) from the selected perception model (P-C, pretrained backbone, adapted), on synthetic flat-colour renders." Not allowed: statements about real egocentric images or about completion quality (B5 contains no completion).

## 8. Limits
Synthetic flat-colour frames, one renderer, rectified pinhole camera, given head pose, per-clip floor height from the clip itself (for rendering only), train subset of 1,500 clips.
