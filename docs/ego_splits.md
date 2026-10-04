# Track E1 — splits, cohorts, floor-work evaluation

**Command:** `python -m hready.data.ego_splits --config configs/ego_splits.yaml --splits val,test`  
**Outputs:** `results/E/splits.json`, `results/E/floor_work_clips.csv`, `results/E/cohort_counts.json`

**Frozen:** geometry thresholds in `configs/ego_splits.yaml` (`floor_work.geometry_tree`, dated **2026-10-05**) before any E3 result.

## Subject split

Uses existing `assign_split` on the AMASS index. Disjointness on `{subset}/{subject}` (pairwise overlaps **0**). Train/val/test clip lists are unchanged by tune subjects; `splits.json` records SHA256 of sorted `rel_path` lists before/after adding tune metadata.

**Clip-list SHA256 (must not change):** train `a3621dfbe0863558f0c1ed0ebd7b8c5af96533963ce3c9e80b9052093b459657`, val `19d97bd7cd23d2168627a17825721e102565394faf4616b5fcd2465ec4c7e957`, test `0ebd29fb002793b491d0d122c88e5171e8e96bf99961d2271a98a049f9f32832`.

## Cohorts (BABEL proposals)

| Cohort | Keywords | Geometry |
|--------|----------|----------|
| `floor_work` | lie, crawl, kneel, sit, yoga | Decision tree → `lie`, `crawl`, `kneel`, `sit_floor`, `sit_support`, `yoga_like`, `none` |
| `ordinary_locomotion` | walk, run, stand, turn | Listed for transparency (no floor geometry gate in E1) |
| `other_labelled` | crouch, stretch | Never floor-work proposals |

**BABEL → accepted geometry:** `sit` → `{sit_floor, sit_support}`; `yoga` → `yoga_like`; others map 1:1. Confirmed if ≥ `min_geometry_fraction` (0.5) of segment frames match the accepted set. **Geometry is authoritative;** disagreements excluded and counted.

### `sit_floor` vs `sit_support` (mesh pelvis height, not BABEL semantics)

On grounded SMPL-X FK, **pelvis joint height** (metres, floor z=0) splits seated postures:

- **`sit_floor`:** upright sit gates + `pelvis_h ≤ sit_support_pelvis_h_min` (low pelvis band — floor sitting in joint space).
- **`sit_support`:** same gates + `pelvis_h` above that threshold (seat-height mode ~0.5–0.6 m in AMASS; includes most KIT chair/wipe tasks).

**Amendment 2026-10-05:** `sit_support_pelvis_h_min` **0.68 → 0.43** (label-free VAL sit-gate pelvis valley / floor-tail vs seat-mode midpoint; see acceptance **D** print). At 0.68, ~96% of sit-gate VAL frames were `sit_floor`, which conflated chair sitting with floor sitting.

**Floor-work-eligible geometry** (metrics that assume floor support): `sit_floor`, `kneel`, `lie`, `crawl`, `yoga_like`. **`sit_support` is excluded** — reported separately in `cohort_counts.json` → `floor_work_eligible`.

## Height units

Ordinary-locomotion VAL frames: pelvis height CV raw vs normalized-by-standing-pelvis (zero pose, clip betas). **Use metres** when normalization does not reduce spread (`use_normalized_height: false` in frozen config).

## Geometry decision tree (mutually exclusive)

Order: **crawl** → **lie** → **kneel** → **sit_floor** / **sit_support** → **yoga_like** → **none**.

Features: `pelvis_h`, `torso_up_dot` (neck−pelvis vs +Z), `wrist_h`, `head_h`, `knee_h`, `foot_z_min` (ankles/feet), `thigh_up_dot`, **`shoulder_h`** (mean shoulders 16/17).

**Horizontal low-pelvis band** (`torso_up_dot ≤ torso_horizontal_max`, `pelvis_h ≤ crawl_pelvis_h_max`):

- **crawl:** `shoulder_h > crawl_shoulder_h_min` (trunk supported on arms) and `wrist_h ≤ crawl_wrist_h_max`.
- **lie:** `shoulder_h ≤ lie_shoulder_h_max` and `pelvis_h ≤ lie_pelvis_h_max` (trunk on floor).
- **Removed (2026-10-05):** `foot_z_min ≤ 0.06` (unsatisfiable on ankle joints ~0.12–0.15 m); lie wrist gate `wrist_h > 0.22` (excluded supine arms).

**Raised / non-floor horizontal:** `pelvis_h > crawl_pelvis_h_max` → trace `none:raised_support_not_floor` (not a geometry class).

Each threshold has provenance in `threshold_provenance` (physical **anchor** or label-free VAL **valley**). Thresholds were **not** tuned to maximize BABEL agreement.

## Floor-work evaluation set

**VAL ∪ TEST** (`--splits val,test`). Per category: segments, unique clips, unique subjects, confirmed / disagreement / unconfirmed, split by `ann_source` (`frame_ann` / `seq_ann`). Whole-file `seq_ann` segments require geometry on the same `[start_t, end_t]` range.

**Indicative, N small** (fewer than 10 unique subjects): TEST `kneel` (1), `crawl` (3), `yoga` (2); see `cohort_counts.json`.

## Tune set

`seed=0`, `fraction=0.05` → **19** train `{subset}/{subject}` ids in `splits.json` (`tune.subject_ids`), disjoint from val/test and from other train subjects.

## Foot-height rise

`rise_cm` attached per clip; **not** an exclusion for floor-work cohorts.

## Oct 5, 2026 run (E1 close)

Honest outcomes after tree fix (see mission acceptance prints). Lie/crawl/yoga confirmation may remain **low** — reported as findings. KIT subjects dominate **confirmed BABEL sit** duration (~82% at 0.68 split); after `sit_support_pelvis_h_min=0.43`, most confirmed sit segments classify as **`sit_support`** (chair-height pelvis).

Legacy overlapping yaml on **160** TEST proposals (with stretch): wrong indices 5/7/148; correct indices 52/61/47.

## Hashes (re-run `ego_splits` twice to verify)

Record SHA256 of `splits.json`, `floor_work_clips.csv`, `cohort_counts.json` after each mission run (`--verify-byte-stable`).

**ACCEPTANCE F (geometry-only scan):** skipped in CLI by default (`skip_geometry_only=True`) — full-index FK over VAL clips is multi-hour; BABEL-gated evaluation above is authoritative for E1.
