# Track E1 — splits, cohorts, floor-work evaluation

**Command:** `python -m hready.data.ego_splits --config configs/ego_splits.yaml --splits val,test`  
**Outputs:** `results/E/splits.json`, `results/E/floor_work_clips.csv`, `results/E/cohort_counts.json`

**Frozen:** geometry thresholds in `configs/ego_splits.yaml` (`floor_work.geometry_tree`, dated **2026-10-04**) before any E3 result.

## Subject split

Uses existing `assign_split` on the AMASS index. Disjointness on `{subset}/{subject}` (pairwise overlaps **0**). Train/val/test clip lists are unchanged by tune subjects; `splits.json` records SHA256 of sorted `rel_path` lists before/after adding tune metadata.

## Cohorts (BABEL proposals)

| Cohort | Keywords | Geometry |
|--------|----------|----------|
| `floor_work` | lie, crawl, kneel, sit, yoga | Decision tree → `lie`, `crawl`, `kneel`, `sit_floor`, `sit_support`, `yoga_like`, `none` |
| `ordinary_locomotion` | walk, run, stand, turn | Listed for transparency (no floor geometry gate in E1) |
| `other_labelled` | crouch, stretch | Never floor-work proposals |

**BABEL → accepted geometry:** `sit` → `{sit_floor, sit_support}`; `yoga` → `yoga_like`; others map 1:1. Confirmed if ≥ `min_geometry_fraction` (0.5) of segment frames match the accepted set. **Geometry is authoritative;** disagreements excluded and counted.

## Height units

Ordinary-locomotion VAL frames: pelvis height CV raw vs normalized-by-standing-pelvis (zero pose, clip betas). **Use metres** when normalization does not reduce spread (`use_normalized_height: false` in frozen config).

## Geometry decision tree (mutually exclusive)

Order: **crawl** → **lie** → **kneel** → **sit_floor** / **sit_support** → **yoga_like** → **none**.  
Features: `pelvis_h`, `torso_up_dot` (neck−pelvis vs +Z), `wrist_h`, `head_h`, `knee_h`, `foot_z_min` (ankles/feet), `thigh_up_dot` (mean hip→knee vs +Z).

Each threshold in `geometry_tree.thresholds` has provenance in `threshold_provenance` (physical **anchor** or label-free VAL **valley** where noted). Thresholds were **not** tuned to maximize BABEL agreement.

## Floor-work evaluation set

**VAL ∪ TEST** (`--splits val,test`). Per category: segments, unique clips, unique subjects, confirmed / disagreement / unconfirmed, split by `ann_source` (`frame_ann` / `seq_ann`). Whole-file `seq_ann` segments require geometry on the same `[start_t, end_t]` range.

**Indicative, N small** (fewer than 10 unique subjects): TEST `kneel` (1), `crawl` (3), `yoga` (2); see `cohort_counts.json`.

## Tune set

`seed=0`, `fraction=0.05` → **19** train `{subset}/{subject}` ids in `splits.json` (`tune.subject_ids`), disjoint from val/test and from other train subjects.

## Foot-height rise

`rise_cm` attached per clip; **not** an exclusion for floor-work cohorts.

## Oct 4, 2026 run (P1b v2)

| Metric | VAL∪TEST floor_work proposals |
|--------|------------------------------|
| Proposals (no stretch/crouch) | 197 |
| Geometry confirmed | 107 |
| Disagreements | 10 |
| Unconfirmed | 80 |
| Per-category confirmed | sit: 95, kneel: 12 |

Legacy overlapping yaml on **160** TEST proposals (with stretch): wrong indices 5/7/148; correct indices 52/61/47. New tree on VAL BABEL sit frames: sit→kneel **3** vs legacy **3745** (frame counts).

## Hashes (re-run `ego_splits` twice to verify)

Record SHA256 of `splits.json`, `floor_work_clips.csv`, `cohort_counts.json` after each mission run.
