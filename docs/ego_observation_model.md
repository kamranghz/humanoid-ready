# Egocentric evidence schema and oracle simulator (Track E2-A)

**Status:** oracle **control** path only. Perceived RGB (E2-B4) must emit the same tensor schema so downstream completion cannot distinguish oracle from perceived evidence.

**Seeds:** `configs/ego_observation.yaml` → `seed: 0`; per-clip RNG `base_seed + idx * 1_000_003` (`EgoOracleWindowDataset`); locomotion visibility sample `loco_sample.seed: 0`.

**Eye–head geometry (measured):** on `CMU/132/132_35_stageii.npz`, mean distance eye midpoint (joints 23/24) to head joint (15) is **82.65 mm** (`locked_head` FK).

## Rig signals (given, not evidence)

Separate from evidence tensors and from GT targets (project definition D1):

| Field | Shape | Description |
|--------|--------|-------------|
| `head_pos_world` | `(T, 3)` | Midpoint of SMPL-X eye joints 23/24 (55-joint FK). |
| `camera_R` | `(T, 3, 3)` | World→camera rotation; rows `[right, up, -forward_geom]` (same as `project_joints`). |
| `camera_t` | `(T, 3)` | `t = -R @ head_pos_world`. |
| `gravity_world` | `(3,)` | `(0, 0, -9.81)` m/s², Z-up world. |

Head 6-DoF for tables is derived from `(camera_R, camera_t)`; it is **not** duplicated inside `obs`.

## Evidence schema (`obs`)

| Key | Shape | dtype | Description |
|-----|--------|--------|-------------|
| `joint_pos_3d` | `(T, 22, 3)` | float32 | Noisy 3D joint estimates in **world** frame; zero where not visible. |
| `joint_visible` | `(T, 22)` | bool | Visibility after FOV, `z_cam > z_near`, capsule self-occlusion, dropout. |
| `joint_confidence` | `(T, 22)` | float32 | `confidence_visible` if visible else `confidence_hidden` (then masked by dropout). |
| `keypoints_2d` | `(T, 22, 3)` | float32 | Normalized `(nx, ny, conf)` from `project_joints_egocentric` (bitmap `ny`, conf zero if not visible). |

Joint order: `hready.body.joint_indices.JOINT_NAMES_22`.

## Targets (`targets`, loss only)

| Key | Description |
|-----|-------------|
| `transl`, `root_aa`, `body_aa` | Ground-truth SMPL-X motion (never in `obs`). |
| `joints_gt_22` | Clean FK keypoints. |
| `betas` | Shape (16). |

## Head frame (no gravity)

From 55-joint positions:

- `cam_pos = 0.5 * (j23 + j24)`
- `right = normalize(j24 - j23)`
- `up = orthogonalize(j15 - j12, right)`
- `forward = right × up` (`numpy.cross(right, up)`)
- `R = stack([right, up, -forward])`, `t = -R @ cam_pos` (**`det(R) ≈ -1`**, same pinhole as HR-Refine’s `project_joints` input)

**Bitmap convention (E2-A only):** after `project_joints`, **`ny ← -ny`** (equivalently `v_bitmap = 2·cy − v_pinhole`). Row 0 of `R` → **+u** (right); row 2 → **+z_cam** (look); bitmap row 0 is top, **+v down**.

Does not use world gravity; supports lying / face-down poses.

## Camera / projection

- Reuses `hready.data.refine_corrupt.project_joints` (not `sample_virtual_camera`).
- Pinhole horizontal FOV at `img_size` **512×512** (degrees):

| `focal_px` | Horizontal FOV |
|------------|----------------|
| 120 | ≈ 129.8° |
| 200 | ≈ 104.0° |
| 350 | ≈ 72.4° |

(Default `focal_px_default: 120` in config; FOV via `2·atan((width/2)/focal_px)`.)

**Pinhole limitation at default FOV:** the head-mounted camera sees mostly the lower visual field (hands, forearms, legs when visible). **Torso keypoints are almost never visible** at the default FOV (cohort `all` torso visibility ≈ 0.004 at `focal_px=120`, `stride=8` in the visibility report) because spine/collar joints sit behind the head and outside the wide-but downward-centered cone.

- In-image: `|nx|, |ny| ≤ 1`; depth gate `z_cam > 0.1` (projection uses `z` clamp `min=0.05` in denominator only).
- **E2-B renderer contract:** raster with `camera_R`/`camera_t` from `head_frame_from_joints` and the E2-A bitmap rule above (`ny` flipped vs raw `project_joints`); `keypoints_2d` in emitted evidence use the flipped `nx, ny`.

## Visibility phase 1

1. Pinhole FOV + `z_cam > z_near`.
2. Cheap self-occlusion: ray `cam_pos → joint` blocked if it passes within **torso capsule** (pelvis–neck, radius `torso_radius_m` default **0.11 m**) or **upper-arm capsules** (shoulder–elbow, radius `upper_arm_radius_m` default **0.055 m**) at a closer ray parameter than the joint.

Mesh z-buffer occlusion: deferred until after the first metrics table.

## Noise and dropout (assumptions; sweepable in config)

| Parameter | Default | Applies to |
|-----------|---------|------------|
| `wrist_pos_std_m` | 0.012 m | Wrists 20, 21 |
| `upper_body_pos_std_m` | 0.008 m | `UPPER_BODY_JOINTS` |
| `leg_pos_std_m` | 0.015 m | Leg joints when visible |
| `kp_noise_px` | 2.0 px | 2D keypoints |
| `joint_dropout_prob` | 0.05 | Per joint per frame |
| `frame_dropout_prob` | 0.02 | Whole frame |
| `confidence_visible` / `confidence_hidden` | 0.92 / 0.0 | Confidence channel |

Not hardware-measured values; each key lives under `noise:` in `configs/ego_observation.yaml`.

## What the oracle obs looks like (default FOV)

At **`focal_px=120`**, evidence is dominated by **head rig signals** plus **intermittent hands and legs** in view; **torso groups are rarely observed**. Oracle `obs` is not a full-body point cloud.

**Why `thighs` visibility in cohort `all` exceeds subcohorts:** the visibility report builds `all` as the **concatenation of every frame** from `floor_work_eligible`, `sit_floor`, `sit_support`, and `ordinary_locomotion` (see `cmd_visibility_report`). With `stride=8`, one run had **`n_frames`**: sit_support **4671**, ordinary_locomotion **2214**, floor_work_eligible **389**, sit_floor **257** (total **7531**). **`sit_floor` and `sit_support` segments** (looking down at legs while sitting) drive a **much higher per-frame thigh visibility** (~0.57 and ~0.41) than walking or kneel/lie-only floor work (~0.13–0.15), so the **frame-weighted** `all` thigh rate (~0.33) exceeds each locomotion/floor-work subcohort.

## Leak-proof path

- **Do not** use `HRRefine.encode_inputs`, `apply_corruption`, or `HRRefineWindowDataset` for Track E completion.
- Use `EgoOracleWindowDataset` + `collate_ego_oracle` → `obs`, `rig`, `targets`.
- `EgoCompletion.forward(obs)` accepts **only** `obs` (random-init `nn.Module`; see `hready.models.ego_completion`).
- Normalization stats: `compute_obs_normalization_stats` on **train** `obs` only.

CLI leak checks: `python -m hready.data.ego_observation leak-check`.

## Module entry points

```bash
python -m hready.data.ego_observation verify-facts
python -m hready.data.ego_observation head-frame-check
python -m hready.data.ego_observation fov-report
python -m hready.data.ego_observation visibility-report
python -m hready.data.ego_observation byte-stable --seed 0
python -m hready.data.ego_observation leak-check
python -m hready.data.ego_observation regression-2-5
```

## Reporting groups

Visibility rates use `VISIBILITY_JOINT_GROUPS` in `joint_indices.py`: hands, forearms, torso, thighs, shins, feet.

Cohorts: `all`, `ordinary_locomotion` (CMU sample, same count seed as E1b-style table), `floor_work_eligible` (kneel+lie segments from `floor_work_clips.csv`), plus descriptive `sit_floor` / `sit_support`.
