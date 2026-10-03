# Pivot log (agile / data-engine pivots)

| Date | Problem | Decision | Outcome |
| --- | --- | --- | --- |
| 2026-10-02 | DFaust NPZ `mocap_frame_rate=120` disagrees with BABEL `dur` at 60 Hz playback | Keep `mocap_frame_rate` as metadata; index `fps` / duration from `playback_fps_for_subset("DFaust")=60` | Resampling and BABEL gates use 60 Hz timeline; gravity jump windows only qualify at 60 Hz on DFaust |
| 2026-10-02 | BMLrub treadmill clips: filename flag misses belt motion; name-only exclude drops valid floor walking | Measure belt via `skate_score` on foot channels; `T_SKATE=0.35 m/s`; `exclude_contact` uses `skate_flag` only (name informational) | **1252** skate-flagged clips; **412/1213** name-flag among belt-signature clips |
| 2026-10-02 | Foot contact vertices wrong: `smplx_parts_segm` indices were **face** ids, not vertices | Replace with LBS argmax foot sets **{7,10}/{8,11}** + sole heel/toe clusters | CMU stand channel heights **0–2.3 cm**; penetration median **0.7 mm** on full mesh |
| 2026-10-02 | Partial `foot_traj` build: WindowsApps Python + `--build-foot-traj-all` passed `None` → default BMLrub+300 list | Fix CLI to pass full `load_index()`; require conda **`hready`** for SMPL-X; 20-clip cross-interpreter max diff **0 m** | **17,355** `foot_traj` npz files on disk |
| 2026-10-02 | `amass_contact` npz cache duplicated `foot_traj` and stalled sign-off | Drop contact cache; `load_contact` / `compute_foot_contact` on demand from `foot_traj` | Contact thresholds locked at **h_on=0.05**, **v_on=0.2** from full BABEL `frame_ann` grid |
| 2026-10-03 | Item 5c-2: synthetic IMU + clip viewer for sign-off | `hready/data/imu.py` (6-sensor FK + central-diff acc); `scripts/inspect_clip.py` → `results/checks/` PNGs | Quiet-window validation on **653** CMU stand/idle clips (**`CMU/13/13_03`** 1 s); `91_48` not quiet; ballistic helper = **g sign only**; docs in `dataset_challenges.md` |
| 2026-10-03 | Item 6: Isaac+Newton on Windows vs WSL | **WSL-only** Kit process; `hready` file bridge (`run_smoke.py`); PhysX dropped for smoke | `isaaclab_newton.py` + `metrics.py` + `replay.py` |
| 2026-10-03 | Newton `is_global` wrench semantics | Body frame: `f_body = R @ F_world` (hover tests on G1) | Documented; pin still unstable without feedforward |
| 2026-10-03 | Pelvis 6D pin / assist wrench for item 6 | **Dropped** after 6B: no stable floating pin; contact force readback N/A | **Assist wrench not reported**; `kin_root` + `free` PD tracking only |
| 2026-10-03 | Why `kin_root` mode | Separates retarget feasibility from balance: root kinematics injected, joints track under default G1 PD | Primary smoke metric alongside free-root time-to-fall |
| 2026-10-03 | 6C waist “saturation” | Logged PD demand can exceed effort_limit on ImplicitActuator waist; Isaac `applied_torque` clamps at limit | Per-group `pd_demand_*` vs `actuator_clamped_*`; archive `tau_applied_gmr` |
| 2026-10-03 | 5c-2 foot-height rise | `amass_foot_height_rise.json` over 17,355 `foot_traj`; ACCAD C20 ~6.9 cm rise | Contact empty after landing plateau; not flight |
