# Egocentric render set: camera and appearance spec

**Date:** 2026-11-19. **Status:** the camera profile and the noise-calibration hold-out were decided by the owner on 2026-11-19
(`docs/decisions.md`); the renderer, appearance and sampling decisions stay open. No code and no render has been made for this note.

This is the review point that the tracker asks for before the render set step renders anything. It follows `docs/project_definition.md`
(section 5, "Egocentric render set") and the contract in `docs/ego_observation_model.md`. Every number below is either
quoted from a repo document (source named) or marked as a proposal or as to be measured.

## 1. What this step produces

Egocentric RGB frames of AMASS clips, rendered from a virtual head camera on the SMPL-X body, with per-frame ground truth
(2D and 3D joints, per-joint visibility, camera pose, gravity, floor height). The perception stage (baseline measurement, training, model selection) consumes only
the images plus the given head pose and gravity, and must emit the shared evidence schema of the oracle observation simulator.

Stated limit (kept on every results table): train and test use the same renderer, so synthetic-only results are
optimistic about real footage.

## 2. Camera

### 2.1 Pose (inherited, not redesigned)

- Position: midpoint of the SMPL-X left and right eye joints. Orientation: `head_frame_from_joints` in
  `hready/data/ego_observation.py` (right = eye line, up = head minus neck orthogonalised, forward = right x up).
  The renderer reuses this function, so the images and the oracle evidence share one camera.
- Convention: the oracle simulator work recorded that the renderer should use a det +1 rotation (rows right, down, look). The oracle path uses
  rows (right, up, -forward) and flips `ny`. The render set must convert once, in one function, and test it (section 7, check 1).
- The head pose and gravity are **given rig signals** (binding decision of 2026-10-04): the render uses the exact pose.
  Pose noise belongs to the realistic observation model, not to the renderer.

### 2.2 Intrinsics: two profiles

| profile | image | focal | horizontal x vertical FOV | source |
|---|---|---|---|---|
| `ego_wide` | 512 x 512 | `focal_px` 120 | 129.8 deg x 129.8 deg | `configs/ego_observation.yaml` default; all current oracle results |
| `hl2_like` | 640 x 360 | fx 486.3, fy 486.3, cx 315.3, cy 170.0 | 66.7 deg x 40.6 deg | EgoBody PV camera scaled by 1/3 (below) |
| `aria_like` | 1408 x 1408 (render size open) | fisheye, focal 1218.0 px and principal point (1455.8, 1451.6) in a 2880 x 2880 calibration frame; scaled to the render size | about 115 deg horizontal (radial terms only) | Nymeria `camera-rgb`, `docs/nymeria_data_format.md` section 6 |

`hl2_like` derivation: `docs/egobody_data_format.md` records the PV image as 1920 x 1080, principal point
(945.911, 510.013) in the one example given, and `fx, fy` between 1451 and 1467 in the three recordings checked.
Using fx = 1459 (midpoint of that range) gives a horizontal FOV of 66.7 deg (66.4 to 67.0 over the range) and a vertical
FOV of 40.6 deg. Whether `fx` varies inside a recording was not checked. Dividing by 3 gives the 640 x 360 intrinsics.

**Finding to carry forward.** The oracle simulator's default camera is about twice as wide as the only real egocentric
device planned for validation. The existing visibility sweep already contains `focal_px` 350 (72.4 deg at 512 px), the
closest point to `hl2_like`; the oracle results that used the default 130 deg view therefore describe a camera that
sees more of the body than a HoloLens-class camera does.

**Decision (owner, 2026-11-19).** `hl2_like` is the primary profile, because the synthetic-to-real gap is measured on EgoBody and the
real camera is the one the perception stage must work on. A small `aria_like` subset follows the Nymeria camera, so real video of
floor work can be compared with renders of a similar field of view. `ego_wide` is rendered only for a small subset so the oracle
results stay comparable. `aria_like` needs a fisheye projection; if the chosen renderer cannot do it, it becomes a rectified pinhole
view and this document says so.

### 2.3 Mount offset

The camera sits at the eye midpoint. On EgoBody the PV camera centre is 0.14 to 0.22 m from the wearer's SMPL-X head joint
(median 0.15 to 0.18 m; `docs/egobody_data_format.md`, section 4), while the eye midpoint is 82.65 mm from the head joint in
the repo's check (`cmd_verify_facts`). The real camera is therefore mounted further from the head joint than the virtual one.
The render set keeps the eye position and records the gap; a forward/up offset sweep is a later robustness check, not part of this step.

## 3. Body appearance

- Mesh: the repo's SMPL-X body (`locked_head`, neutral, 16 betas, flat hands), posed by the same wrapper that produces
  the AMASS joints, so rendered geometry and joint ground truth are the same body.
- **No texture is available.** SMPL-X UV textures need a separate download and licence, and the download pause is still in
  force. Proposal: flat-shaded albedo per body region, taken from the dominant skinning joint of each vertex. Colours are
  sampled per subject (skin tone for hands and forearms, a palette for torso, legs and feet) from the render seed.
- Head and face: vertices of the head region within the near plane distance of the camera are culled, so the camera never
  sits inside the head mesh. Hands stay visible.
- No clothing geometry, no hair, no accessories. This is the largest appearance gap to real video and is stated, not
  mitigated, in this step.
- Lighting: one directional light plus ambient, direction and intensity sampled per clip from the seed.

## 4. Scene and background

- No scene assets exist for AMASS. The floor is a plane at the gravity-aligned floor height of the clip with a random
  procedural texture (checker or noise, scale and colours from the seed). Walls and sky are a random vertical gradient.
- Consequence: `sit_support` clips show a body sitting on nothing, since AMASS carries no furniture. This is rendered as
  is and recorded; those frames are descriptive only, as in the existing cohort rules.
- BEDLAM scenes (items 12 and 13) and EgoBody scene meshes are not used: BEDLAM is a third-person set and EgoBody is
  evaluation-only.

## 5. Renderer

Nothing in the pinned `hready` environment (`env/versions.lock`: torch 2.11.0+cu128, numpy 2.4.6, smplx 0.1.28,
matplotlib 3.11.2) rasterises triangles. The choice needs one probe on the RTX 4090 Windows machine; it cannot be settled
from here.

| option | needs | note |
|---|---|---|
| nvdiffrast | CUDA build with MSVC on Windows | fastest, gives depth for the z-buffer visibility |
| pyrender | pip, an OpenGL context on Windows | simple, slower, depth buffer available |
| small torch rasteriser (own code) | nothing new | slowest to write, no dependency risk, depth buffer by construction |

**Proposal:** probe nvdiffrast first; fall back to pyrender; keep the own rasteriser as the last resort. Installing a
rendering package is a tooling install, not a dataset download, but it is a new dependency and its licence must be recorded
in `docs/decisions.md` first (the renderer decision, section 8).

## 6. Ground truth per frame and the manifest

Per rendered frame, stored in a manifest (JSON lines) next to the images under `hready_data` (not in the repo):

- clip id, AMASS relative path, frame index, split (from `results/splits/splits.json`), subject, cohort tags;
- `camera_R`, `camera_t`, intrinsics profile, gravity vector, floor height;
- 55-joint 3D positions (world frame), 2D positions in the image, and per-joint visibility;
- renderer, seed and software versions.

**Visibility.** The oracle simulator uses a capsule self-occlusion model and records that mesh z-buffer occlusion was
"deferred until after the first metrics table" (`docs/ego_observation_model.md`). The renderer gives mesh depth for free,
so the render set stores z-buffer visibility (a joint is visible if its depth is within 0.03 m of the rendered depth at its pixel;
the tolerance is a proposal). It also reports how often it disagrees with the capsule visibility, per cohort.

**Sampling.** Frame stride and clip subset are the sampling decision (section 8). The oracle sweeps used `frame_stride` 8 and a 200-segment
locomotion sample. Sizing needs the number of train clips and frames, which this note does not state because it was not
recomputed here; the manifest summary reports it in the manifest summary before any large render.

## 7. Acceptance checks

1. **Projection consistency.** On at least 3 walking and 3 floor-work clips (kneel and lie, the clips of the oracle head-frame check), the rendered
   2D position of every visible joint matches `project_joints_egocentric` within 1 px, and the walking look direction agrees
   with velocity as in that check (0.97 to 1.00).
2. **No split leakage.** The subject sets of train, VAL and TEST in the manifest are pairwise disjoint and equal to the
   sets in `results/splits/splits.json`; TEST frames are not rendered before perception model selection needs them.
3. **Byte stability.** Rendering the same clip twice with the same seed gives identical images.
4. **Visibility report.** Per joint group and cohort, z-buffer visibility rate and its disagreement with capsule visibility.
5. **Owner visual check** of contact sheets for kneel, lie, sit_support and walking. Sheets stay local, not committed.
6. **Leak rule.** The perception input contract is images plus the given head pose and gravity; the manifest fields that hold
   GT joints and body parameters are never passed to a model.
7. The regression of AGENTS.md section 7 items 2 to 5 still passes, and CI stays green.

Done when: this spec is approved, the manifest and its summary exist, checks 1 to 4 and 6 to 7 pass, and the owner has done check 5.

## 8. Decisions for the owner

Each decision has a recommendation; none is final until the owner confirms it.

- **Camera profile. Decided: `hl2_like` primary plus a small `aria_like` subset (`ego_wide` only for oracle comparability).** Earlier recommendation, kept for the record: `hl2_like` primary, plus a small wide subset. Reasons: the only real egocentric data
  planned (EgoBody) comes from this device class, so the synthetic-to-real gap is measured on a matching camera; the narrow
  view leaves the torso and often the legs unseen, which is the case where completion matters; and a result across two FOVs is
  a sensitivity statement that a single camera cannot give.
- **Renderer package and its licence**, after the probe in section 5.
- **Hold-out for noise calibration. Decided as recommended:** use the tune subjects already carved out of the train subjects in
  `results/splits/splits.json` (19 subjects, fraction 0.05, seed 0) as the synthetic calibration set. The perception model
  trains on train minus tune; model selection uses VAL; TEST is used once. Real-image noise is calibrated on the EgoBody
  calibration set only, as recorded in `docs/decisions.md`. A leak check must show that no tune subject enters perception training.
- **Appearance.** Keep flat per-region colours (proposed) or wait for a licensed texture.
- **Frame stride and clip subset**, after the manifest summary gives the real counts.

## 9. Not in this step

No perception model, no detector run (that is the baseline measurement step), no change to the oracle simulator or any completed result, no real video,
no TEST evaluation, and no new dataset download.
