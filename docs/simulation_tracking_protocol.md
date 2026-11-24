# Simulation tracking: protocol (E6)

Recorded 2026-11-24, before any tracking run. Numeric thresholds are added as a dated amendment at the end of this file
after the baseline check and before any training run. Whatever the results are, they are reported.

## Question

Does the BeyondMimic recipe (exponential tracking reward, early termination, adaptive sampling of the reference) train a
stable tracker for the SMPL-X-skeleton humanoid in Isaac Lab with the Newton backend, and which parts of the recipe are
needed? The recipe was published for a robot on a different engine, not for this body and engine.

## Stage 1: infrastructure and baseline check (no training)

- Run the humanoid in many parallel copies in one scene on Newton (10, 100, then as many as fit). Record steps per second
  and GPU memory per copy count.
- Repeat the 5 s stand check at the control step used for tracking (simulation step 1/180 s, control at 30 Hz).
- Open-loop baseline: give the reference joint targets to the PD controllers with no policy; record how many clips fall
  and the tracking error. This is the "no policy" arm.

## Stage 2: single-clip recipe study

- Clips: four, one per category (walking, sitting on a support, kneeling or other floor work, a fast motion). Selection rule,
  fixed before looking at any result: seeded shuffle of the validation clips of 5 to 15 s, first clip of each category.
- Arms: (1) full recipe; (2) uniform sampling of the reference instead of adaptive sampling; (3) PHC-style termination
  (mean joint error above 0.5 m) instead of the tighter BeyondMimic termination; (4) open-loop baseline from stage 1.
- Seeds: three per arm and clip. Report mean and range, not a single run. If stage 1 shows the budget is too small,
  arms 2 and 3 are run on two clips only, and this is stated.
- Metrics: success (no termination over the full clip), global and root-relative MPJPE, acceleration and velocity error
  (PHC definitions), foot skate, ground penetration, float.
- Claim scope: these clips and this asset only. A general claim waits for stage 3.

## Stage 3: tracker choice

One general tracker, one tracker per clip, or a general tracker adapted per clip. Chosen from the measured speed of stage 1
and the result of stage 2, and recorded in `docs/decisions.md` before the choice is built.

## Recipe values taken from the literature (starting values, not tuned here)

- Termination: anchor or end-effector height error above 0.25 m, or anchor orientation error above 0.8 rad.
- Adaptive sampling over one-second bins, with a uniform floor.
- Reward: exponential terms on body position, orientation, linear and angular velocity; penalties for joint limits,
  action smoothness and self-contact.
- Actions: PD targets around the default pose. PPO with an MLP policy (512, 256, 128), 24 steps per environment.

Sources: Luo et al., PHC (arXiv 2305.06456); Liao et al., BeyondMimic (arXiv 2508.08241); Isaac Lab warp environments page.

## Pre-registered decision rule (numbers added in the amendment)

The full recipe counts as working on a clip when it completes the clip without termination in at least two of three seeds
and its global MPJPE is below the open-loop baseline. The recipe counts as working overall when it works on all four clips.
Ablation results are descriptive.

## Amendment 2026-11-24: infrastructure check, first part (copies, speed, time step)

Decision rule fixed before the runs: of the tested simulation steps, take the largest at which all 64 copies pass the stand
criteria and agree with each other to less than 0.1 mm.

Results (files in `results/E/tracking_infrastructure/`, script `hready/robot/isaaclab_humanoid_parallel_check.py`, 5 s stand,
torch frontend, Newton with the MuJoCo-Warp solver, one process per copy count):

- Step 1/180 s (control 30 Hz): no copy fell, but the copies split into two discrete outcomes (feet drift 0.5 or 2.4 cm,
  pelvis xy range last 2 s 5.7 or 7.7 mm). The split does not depend on the distance of a copy from the origin (3, 30 and 300 m
  spacing give the same two values). Only 1 of 2048 copies met all criteria. Cause not known.
- Step 0.005 s (control 33.3 Hz, reference interpolated at the control times): all 64, 1024 and 4096 copies met all criteria,
  pelvis xy range 4.15 mm, no fall. Copies agree to 0.008 mm (64), 0.036 mm (1024), 0.16 mm (4096); the growth with the copy
  count is consistent with float rounding at larger distances from the origin, not tested. A single copy reproduces the
  earlier single-body stand check (4.1 mm against 4.2 mm).
- Step 1/240 s (control 30 Hz): all 64 copies met all criteria, pelvis xy range 0.58 mm. The 4 mm drift at 0.005 s therefore
  depends on the step size. Kept as the alternative if foot contact precision turns out to matter.
- Chosen by the rule: simulation step 0.005 s, control at 33.3 Hz (6 steps).
- Speed with 0.005 s, physics and a few metrics only (no policy, reward or observation computation): 4096 copies
  750 thousand physics steps per second in two runs (755 thousand in the repeat), about 125 thousand control steps per
  second, 0.92 times real time, GPU memory 3.9 GB in total. 1024 copies: 358 thousand. Earlier single runs at 16 copies
  varied by a factor of 3 between identical runs (cause not known), so small copy counts are not used for speed.
- Time estimates from these speeds (lower bounds, physics only): 30 000 iterations of 24 steps with 4096 copies about
  6.6 h; 3 000 iterations about 0.7 h; 10 billion control steps about 22 h. The cost of policy, reward and observations is
  measured in the single-clip study.

## Amendment 2026-11-24: reference windows and open-loop baseline (written before the export and before any run)

Reference windows (rule fixed here; code `scripts/tracking_reference_export.py`, `hready/robot/tracking_reference.py`):

- Source: validation split. Categories: walk and fast (BABEL walk and run segments), sit_support and floor_work (kneel and lie
  segments of `results/splits/floor_work_clips.csv`; segments flagged for foot skate are left out).
- This refines the earlier "validation clips of 5 to 15 s": the categories are labelled per segment, not per clip, so the unit
  is a segment of at least 5 s, and the window is its first 10 s.
- Per category: sort by clip path and start time, shuffle with a generator seeded by (0, category index), keep the first segment of
  each subject, take the first 8 usable windows. A window is unusable if the clip is a floor-outlier or the window is shorter
  than 5 s inside the clip. The first window of each category is the stage 2 clip. If a category has fewer than 8 usable
  segments, all are taken; if it has none, the export stops and the rule is not changed after seeing that.
- Reference of the humanoid: the neutral-shape asset driven by the clip's hinge angles, clipped to the asset's joint limits
  (the clipped fraction is reported per window), and the clip's root orientation. Root position is the subject's pelvis joint plus
  one constant vertical offset per window, chosen so that the lowest point of the humanoid is 3 mm above the floor at its lowest
  frame; horizontal position starts at the origin. The subject's body shape is not reproduced; this is the same asset for every clip.
- No support object is modelled. In sit_support windows the humanoid must hold the seated pose without the seat. This is a limit of
  that category: results there say how well the pose is held in the air, not how well a person sits.

Open-loop baseline (code `hready/robot/isaaclab_open_loop_baseline.py`, `hready/robot/tracking_metrics.py`):

- All windows of all categories, one copy each, simulation step 0.005 s, control step 6 simulation steps; reference interpolated at the
  control times. Each copy starts at the first reference frame with zero velocity. PD target at each control step = reference
  joint angles at that time (no lead). No policy, no reset.
- Termination (recorded, not acted on): pelvis or ankle or wrist height error above 0.25 m, or pelvis orientation error above
  0.8 rad. Success (PHC): mean joint distance below 0.5 m in every frame. Errors: global and root-relative MPJPE, velocity and
  acceleration error of the joint positions at the control rate.
- Check on the joint mapping: the pose error of all copies at the first frame, before any step, must be below 1 cm.
- Foot skate, ground penetration and float are defined with their thresholds in the amendment that adds the numeric stage 2
  thresholds, after this baseline.
