# Module audit (Oct 4, 2026)

Moved from `docs/decisions.md` for the egocentric motion track handoff. Module reuse, gaps, and completion metric definitions.

## Module → stage → status

| Area | Reuse | Conflicts / gaps |
|------|--------|------------------|
| AMASS index + `assign_split` | `hready/data/amass.py` — beta-group splits, `load_clip`, flags, foot-height-rise sidecar | Split key is **beta-connected folder group**, not raw subject string alone; `assert_no_subject_leakage` is group-level. |
| BABEL | `hready/data/babel.py` — `act_cat_matches_keyword`, cached `babel_index.json` | Many test clips are `seq_ann` only; floor-work rules prefer `frame_ann` segments ≥1 s. |
| Contact / skate / rise | `contact.py`, `foot_height_rise.py`, `refine_eligible_cache.py` | Rise >5 cm excluded from HR-Refine **eligible** train set; floor-work subset **annotates** rise, does not auto-drop. |
| Ego sensors (egocentric evidence) | `refine_corrupt.py`, virtual cam, `imu.py` | Corruption models **full-body** noise, not partial leg dropout; needs new encoder + observation mask. |
| Regression baseline (oracle completion baselines) | `hready/models/hr_refine.py` backbone | Must **not** take corrupted legs; new observation encoder on egocentric-evidence features only. |
| HR-Refine trainer | `hr_refine_engine.py`, `configs/hr_refine.yaml` | Item 8 / fast-loader work kept only in a local backup (motion memmap unfinished). |
| Losses / metrics | `hready/losses/*`, `hready/metrics/*` | Lower-body/feet MPJPE slices and contact **ECE** not wired in one eval CLI yet. |
| SMPL-X | `smplx_wrapper.load_body`, `batch_forward.smpl_forward_bt` | — |
| G1 / Isaac | `hready/robot/*` | **Out of scope** for the egocentric-track preview (robot module deferred). |
| Split cohorts | `hready/data/ego_splits.py`, `configs/ego_splits.yaml` | Split correction: disjoint geometry tree, VAL∪TEST eval, tune subjects in `splits.json`. |
| Missing for egocentric evidence and completion | — | `ego_observation_model.md` + synth; flat-floor heuristic + ego-conditioned regression + shared metrics table. |

## Wrong-path / missing (do not use)

- **HR-Refine encoder on corrupted full-body pose** as a completion baseline without a new observation encoder.
- **Per-face `smplx_parts_segm.pkl` part ids** as body-region joint proxies (face indices ≠ body joints).
- **Filename-only treadmill exclusion** for contact validity (use `skate_flag` / `exclude_contact`).
- **Mixing SMPL-X `v1_1` with `locked_head`** in one training run.

## Completion metric definitions (wired in the oracle baselines; defined here)

### Ground consistency

For each frame, let `C` be the set of foot contact channels labelled **in contact** (bool mask from `compute_foot_contact` on grounded foot trajectories). For each channel `c ∈ C`, let `h(c)` be the contact-point height in the grounded world frame (floor at `z = 0`).

**Ground-consistency violation** at frame `t` if any in-contact channel has `|h(c)| > τ`.

**Ground-consistency score** (per clip): fraction of frames with at least one violation.

**Tolerance `τ`:** config key `ground_consistency_tolerance_m`. **Default (recorded Oct 4, 2026):** `0.04947` m — the 95th percentile of `|h|` over in-contact foot-channel samples on **VAL** clips that have ordinary-locomotion BABEL labels and `exclude_contact == false` (computed by `compute_ground_consistency_tolerance_default` in `ego_splits.py`).

This formula is authoritative for the oracle baselines; implementers read `τ` from config (default above) unless a dated override is recorded in `docs/ego_splits.md`.
