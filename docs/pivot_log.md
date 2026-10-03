# Pivot log (agile / data-engine pivots)

| Date | Problem | Decision | Outcome |
| --- | --- | --- | --- |
| 2026-10-02 | DFaust NPZ `mocap_frame_rate=120` disagrees with BABEL `dur` at 60 Hz playback | Keep `mocap_frame_rate` as metadata; index `fps` / duration from `playback_fps_for_subset("DFaust")=60` | Resampling and BABEL gates use 60 Hz timeline; gravity jump windows only qualify at 60 Hz on DFaust |
| 2026-10-02 | BMLrub treadmill clips: filename flag misses belt motion; name-only exclude drops valid floor walking | Measure belt via `skate_score` on foot channels; `T_SKATE=0.35 m/s`; `exclude_contact` uses `skate_flag` only (name informational) | **1252** skate-flagged clips; **412/1213** name-flag among belt-signature clips |
| 2026-10-02 | Foot contact vertices wrong: `smplx_parts_segm` indices were **face** ids, not vertices | Replace with LBS argmax foot sets **{7,10}/{8,11}** + sole heel/toe clusters | CMU stand channel heights **0–2.3 cm**; penetration median **0.7 mm** on full mesh |
| 2026-10-02 | Partial `foot_traj` build: WindowsApps Python + `--build-foot-traj-all` passed `None` → default BMLrub+300 list | Fix CLI to pass full `load_index()`; require conda **`hready`** for SMPL-X; 20-clip cross-interpreter max diff **0 m** | **17,355** `foot_traj` npz files on disk |
| 2026-10-02 | `amass_contact` npz cache duplicated `foot_traj` and stalled sign-off | Drop contact cache; `load_contact` / `compute_foot_contact` on demand from `foot_traj` | Contact thresholds locked at **h_on=0.05**, **v_on=0.2** from full BABEL `frame_ann` grid |
