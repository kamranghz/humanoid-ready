# Track E3 — oracle completion baselines

**Status:** oracle **control** only (`docs/project_definition.md` [C2]). The perceived-RGB completion table runs after E2-B4 on the same evidence schema. The oracle table is never described as the main system.

**Disclaimer on every table row:** `oracle control; floor height given (world z = 0); head pose and gravity given (D1)`. An unknown floor offset is a later ablation, not E3.

**Commands** (config `configs/e3_oracle.yaml`; every command merges its section into `results/E/e3_oracle_run.json`):

```bash
python -m hready.eval.e3_oracle train-list       # rule-derived training list, count + SHA256, split SHA256 check
python -m hready.eval.e3_oracle build-cache      # memmap cache (resumable, worker pool, progress log)
python -m hready.eval.e3_oracle verify-cache     # byte-equality vs load_clip + FK on 100 seeded clips
python -m hready.eval.e3_oracle preflight        # FK exactness, 1-batch overfit, resume, throughput -> runtime estimate
python -m hready.eval.e3_oracle train            # EgoCompleteMotion; selection on VAL; resumes from last.pt
python -m hready.eval.e3_oracle eval-table       # VAL + TEST tables, both baselines + GT reference row
python -m hready.eval.e3_oracle heuristic-check  # heuristic sanity on VAL ordinary_locomotion + rotation invariance
python -m hready.eval.e3_oracle leak-check       # on the trained checkpoint, with negative controls
python -m hready.eval.e3_oracle run              # all of the above in order
```

A config may add a `subset:` block (`seed`, `train_n`, `val_n`, `test_n`) and its own `paths.results_json` for a pipeline check on a few clips; subsets never move clips across splits and are never results.

## Training clip list (frozen rule)

1. `assign_split(entry) == "train"`
2. `clip_flags(entry)["skate_flag"]` is false
3. `{subset}/{subject}` is not one of the 19 tune subjects in `results/E/splits.json`

Count: **12309** clips (from `train-list`). The SHA256 of the sorted `rel_path` list (newline-terminated, same convention as the E1 split hashes) is stored in `results/E/e3_oracle_run.json` → `clip_lists.train_clip_list.train_clip_list_sha256`. `train-list` also recomputes the train/val/test split SHA256s from the AMASS index and compares them with `splits.json` and `docs/ego_splits.md`. `hr_refine_eligible_index.json` is not used.

Evaluation uses all VAL and TEST clips; clip flags never remove a clip from MPJPE, only from the metrics they invalidate (see below).

## Memmap cache

`<cache_dir>/e3_motion_30hz/clips/<rel_path>/`: uncompressed `.npy` per array, read with `np.load(..., mmap_mode="r")`:
`root_orient`, `pose_body`, `transl` (grounded, floor at z = 0), `betas`, `joints_22`, `joints_55` (locked_head FK, CPU, single thread, fixed 512-frame chunks), `contact` (T, 4 bool; item-5 foot-channel labels from the grounded `foot_traj` cache with the frozen hysteresis rule), and `meta.json` (written last, so an interrupted build resumes cleanly). All arrays are float32/bool at 30 Hz. `verify-cache` recomputes every array from `load_clip(ground=True)` + FK + item-5 contact on a seeded sample and requires byte equality and `np.memmap` reads.

## Evidence, rig and canonical frame

Evidence comes from the E2-A simulator with the camera, noise and occlusion settings of `configs/ego_observation.yaml` (not duplicated in the E3 config). Evaluation draws one evidence sample per whole clip from a process-independent seed (`eval_noise_seed`, sha256 of `rel_path`), so all windows of a clip share it. Training draws fresh evidence per sample (`seed`, draw counter).

Canonical frame per window: subtract the window-start head xy from all world xy (evidence, head, targets); keep z. Training applies one uniform z-rotation (±`z_rot_max_rad`) to evidence, rig and targets together (`camera_R` maps world→camera, so `R' = R Rz^T`; `camera_t` and `keypoints_2d` are invariant). Evaluation is unaugmented. Window predictions are mapped back to the clip frame (add the window origin) before stitching.

## Heuristic baseline (flat-floor ankle-offset, no IK)

All offsets live in the per-frame **heading frame** (right, forward, up), never along a world axis. Offsets come from the **neutral locked_head rest skeleton** (betas 0), symmetrised left/right, arms hanging (the rest pose is a T-pose); nothing is fitted to data. `heuristic-check` writes the template to the results file.

**Heading:** camera look axis (`camera_R` row 2) projected to xy. If `|look_xy| < heading_min_horiz` (0.15): hold the last stable heading; else observed pelvis→head xy; else the first later stable heading (clip start); else world +y. The heading is computed once per clip and sliced per window.

**Rules (hidden joints only; visible joints keep their evidence):** pelvis = observed or head + template offset; hips = pelvis + template offset; knees and ankles under their hip with the template ankle offset in xy, at the template height above the flat floor; feet = ankle + template forward offset at template height; every other hidden joint = head + template offset (upright body).

**Contact channels:** L_heel = joint 7 (left_ankle), L_toe = 10 (left_foot), R_heel = 8 (right_ankle), R_toe = 11 (right_foot). Heuristic contact applies the item-5 hysteresis rule to the sole-proxy channel heights (below).

`heuristic-check` reports MPJPE on VAL `ordinary_locomotion` frames against a naive fill (hidden joint = head position), the heading-source fractions, and world-rotation invariance (rotate evidence, rig and GT about z by several angles; MPJPE must not change). Its negative control is the same heuristic with offsets along world +x, which must fail that test.

## Learned baseline: `EgoCompleteMotion`

`forward(obs, rig)` is the only input path: evidence-schema tensors plus the mandatory rig (`head_pos_world`, `camera_R`, `gravity_world`; missing keys raise). Every per-joint channel is multiplied by visibility, including `keypoints_2d` (E2-A keeps the projected nx, ny of hidden joints). Per-frame encoder over all 22 joints + rig → temporal Transformer at HR-Refine width (d_model 256, 8 heads, 4 layers, ff 512) → `transl` (offset from the given head position), root and 21 body rotations in **6D**, 4 contact logits.

Loss: geodesic angle on root + body rotations, L1 on `transl`, mean per-joint L2 of FK joints, BCE on item-5 contact labels (masked by `exclude_contact`). `w_phys = 0` (E7 "none" arm). Predictions use the **neutral body shape** (betas 0): body shape is not given in the oracle control, so the `transl` target is `pelvis_gt − J0(betas 0)`. FK during training and evaluation uses `NeutralJointFK`, which applies the SMPL-X LBS only to the vertices the 22-joint regressor reads (same math as `SmplxBody.forward`; `preflight` reports the maximum difference).

Training is step-based with a seeded, resumable window sampler; `last.pt` / `best.pt` under `<data_root>/checkpoints/e3_oracle/`. **Model selection:** VAL only, cohorts `all` + `ordinary_locomotion` (mean of the two pooled MPJPEs) on a seeded subset of `selection.n_val_clips` VAL clips, whole clips stitched. Never floor-work, never TEST.

## Evaluation

- Whole clips, windows of 64 frames with stride 32 (last window aligned to the clip end), overlaps averaged. Reported: mean per-joint L2 between consecutive windows on shared frames (`overlap_disagree_mm`).
- MPJPE (mm, pooled over frames × joints): full / upper / lower / foot, each for all, visible and hidden joint-frames (visibility = evidence `joint_visible`).
- Contact precision / recall / F1 against item-5 labels; **ECE for the learned model only** (heuristic: `n/a`).
- Physical metrics for **both** baselines on the 22-joint FK feet (7, 10, 8, 11). A joint is not a contact point, so each channel height is the joint height minus its rest-pose joint-to-sole offset (neutral locked_head, item-5 sole clusters):
  - foot skate: horizontal channel speed (m/s) while the item-5 label marks that channel in contact;
  - penetration: mean / max depth below z = 0 of the channel heights (mm), all frames;
  - ground consistency (`docs/e0_audit.md`): fraction of frames with any labelled in-contact channel whose |height| > τ, τ = `ground_consistency_tolerance_m` (0.04947).
  A `gt_reference` row applies the same proxies to GT joints, so the proxy error itself is visible.
- Flags: `exclude_contact` removes a clip from contact metrics, skate and ground consistency; `exclude_physical_eval` removes it from all physical metrics. Clips lost to each flag are counted per cohort.
- Cohorts (frozen E1 definitions, reused code): `all`; `ordinary_locomotion` (BABEL segments ≥ 1 s whose first matching E1 cohort is ordinary locomotion); `floor_work_eligible` (kneel + lie confirmed segments); `sit_floor` and `sit_support` (descriptive only).
- VAL and TEST are reported separately. Subject-cluster bootstrap 95% CIs (ratio of pooled sums, `bootstrap_n` resamples). `floor_work_eligible` is additionally reported as a per-subject table over VAL ∪ TEST with CIs, labelled **indicative** (8 subjects; kneel concentrated in Eyes_Japan; TEST has 2 segments).

## Leak check (`leak-check`, trained checkpoint)

On a VAL window whose lower body is fully hidden: (a) `forward` parameters are exactly `obs, rig`; (b) leg rotations of the GT are perturbed and the evidence regenerated. Hidden slots of the raw evidence then differ (negative control: they do carry GT-dependent 2D projections), and the visible joints move slightly because joints are regressed from the posed mesh. The leak criterion: replacing only the hidden slots (from the perturbed evidence, or random values) must leave the model output bit-identical. (c) Positive control: moving one visible joint by 5 cm changes the output. (d) Negative controls: `targets=` / `init_body_aa=` keyword arguments, GT keys in `obs`, target keys in `rig`, and a missing rig must all be rejected.

## Known issue (not changed in E3)

The `foot_traj` cache stores floor-grounded channel positions, and `compute_ground_consistency_tolerance_default` (`hready/data/ego_splits.py`) subtracts the floor offset a second time before taking the 95th percentile. The frozen τ in `docs/e0_audit.md` is used unchanged here; changing it needs a dated owner decision recorded in `docs/ego_splits.md`.
