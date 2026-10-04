# Track E1 — splits and floor-work test subset

**Command:** `python -m hready.data.ego_splits --config configs/ego_splits.yaml`  
**Outputs:** `results/E/splits.json`, `results/E/floor_work_clips.csv` (byte-stable across reruns; SHA256 logged in mission report).

## Subject split

Uses existing `assign_split` on the AMASS index (`hready/data/amass.py`). Disjointness is checked on **`{subset}/{subject}`** folder IDs (371 / 46 / 46 subjects in train / val / test; pairwise overlaps **0** on Oct 4 run). Beta-group leakage is separately guarded by `assert_no_subject_leakage` (split groups, not printed here).

## Floor-work rules (test split only)

1. **BABEL candidates:** segment `act_cat` matches a keyword in `configs/ego_splits.yaml` (`kneel`, `sit`, `lie`, `crawl`, `yoga`, `crouch`, `stretch` via `act_cat_matches_keyword`).
2. **Duration:** segment length ≥ `min_segment_duration_s` (1.0 s).
3. **Geometry confirm:** SMPL-X FK (`load_body` + `smpl_forward_bt`) on grounded clips; per-frame rules on pelvis height and torso upright dot (neck−pelvis vs +Z); category must match BABEL label on ≥ `min_geometry_fraction` (0.5) of segment frames.
4. **Disagreement:** BABEL category and dominant geometry category differ → **excluded**, counted in `n_disagreements_excluded`.
5. **Foot-height rise:** `amass_foot_height_rise.json` values attached per clip (`rise_cm`); **not** an exclusion criterion for this subset.

## Oct 4, 2026 counts (computed)

| Metric | Value |
|--------|------:|
| BABEL proposals (test) | 160 |
| Geometry confirmed | 5 |
| Disagreements excluded | 7 |
| BABEL unconfirmed excluded | 148 |
| Per-category confirmed | lie: 5 |
| Confirmed clips with rise >3 cm | 1 |
| Confirmed clips with rise >5 cm | 0 |

**Caveat:** Confirmed floor-work set is **too small** for strong test claims (only **lie**, 5 segments). Geometry thresholds are strict on upright sit/kneel vs AMASS grounding. **Proposal (not applied):** include **val**-split subjects for floor-work **reporting only**, or relax sit/kneel pelvis bounds after visual QC on `results/checks/` clips.

## Hashes (two consecutive runs)

- `splits.json`: `2C2DF09EF2A6E12A53FC3A4C0432A7779A4517B8EE09DB1D9378497E5593E77E`
- `floor_work_clips.csv`: `FAC437FB428C4A3A6694A3CC5ADBB41031889232620BF2FB6789EC73187259C3`
