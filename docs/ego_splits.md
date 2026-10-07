# Track E1 — splits, cohorts, floor-work evaluation

**Command:** `python -m hready.data.ego_splits --config configs/ego_splits.yaml --splits val,test`  
**Outputs:** `results/E/splits.json`, `results/E/floor_work_clips.csv`, `results/E/cohort_counts.json`

**Frozen:** geometry thresholds in `configs/ego_splits.yaml` (`floor_work.geometry_tree`, dated **2026-10-04**) before any E3 result.

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

**Amendment 2026-10-04:** `sit_support_pelvis_h_min` **0.68 → 0.43** (within **0.002 m** of the all-VAL percentile midpoint **0.4321 m**; kept as policy — see provenance below). At **0.68**, segment-dominant confirmed BABEL sit was **76** `sit_floor` / **19** `sit_support`; at **0.43** it is **17** / **78**.

**Floor-work-eligible geometry** (metrics that assume floor support): **`kneel` and `lie` only** (E1 final). **`crawl` and `yoga_like`** have **0** confirmed segments in VAL ∪ TEST — not evaluable floor-work classes. **`sit_floor` and `sit_support` are excluded** from eligible metrics — both reported separately in `cohort_counts.json`; only kneel+lie feed `floor_work_eligible`.

**Amendment 2026-10-04 (eligibility):** `sit_floor` removed from floor-work-eligible because the class is **impure** for floor-contact metrics. Owner visual check: Eyes_Japan **09-squat_down** is a deep squat (no seat) but is labelled **`sit_floor`** (3 confirmed segments; median `pelvis_h` **0.33–0.34 m**, median hip/thigh mesh **`support_h` ~0.15 m**). Across **17** confirmed `sit_floor` segments, median `support_h` spans **0.048–0.31 m** and median `pelvis_h` **0.21–0.45 m**. **`foot_z_min`**, **`thigh_up_dot`**, and a mesh **`support_h`** gate (hips + left/right up-leg vertices, grounded FK) **do not** separate floor sitting from squats without discarding most of the class (**τ = 0.0495 m** keeps **1/17**; **τ = 0.07 m** keeps **4/17**); confirmed **`sit_support`** segment medians are all **≥ 0.278 m**. Contact-with-floor metrics do not apply to squats. A mesh-based support anchor remains a possible later refinement; **not** adopted in E1 (no tree/threshold change).

## Height units

Ordinary-locomotion VAL frames: pelvis height CV raw **0.089** vs normalized **0.112** (`use_normalized_height: false` in frozen config). **Use metres** on raw pelvis height.

## Geometry decision tree (mutually exclusive)

Order: **crawl** → **lie** → **kneel** → **sit_floor** / **sit_support** → **yoga_like** → **none**.

Features: `pelvis_h`, `torso_up_dot` (neck−pelvis vs +Z), `wrist_h`, `head_h`, `knee_h`, `foot_z_min` (ankles/feet), `thigh_up_dot`, **`shoulder_h`** (mean shoulders 16/17).

**Horizontal low-pelvis band** (`torso_up_dot ≤ torso_horizontal_max`, `pelvis_h ≤ crawl_pelvis_h_max`):

- **crawl:** `shoulder_h > crawl_shoulder_h_min` (trunk supported on arms) and `wrist_h ≤ crawl_wrist_h_max`.
- **lie:** `shoulder_h ≤ lie_shoulder_h_max` and `pelvis_h ≤ lie_pelvis_h_max` (trunk on floor).
- **Removed (2026-10-04):** `foot_z_min ≤ 0.06` (unsatisfiable on ankle joints ~0.12–0.15 m); lie wrist gate `wrist_h > 0.22` (excluded supine arms).

**Raised / non-floor horizontal:** `pelvis_h > crawl_pelvis_h_max` → trace `none:raised_support_not_floor` (not a geometry class).

### Threshold amendments (old → new, provenance)

| Threshold | Frozen | Previous / removed | Provenance type |
|-----------|--------|-------------------|-----------------|
| `crawl_shoulder_h_min` | **0.30** | *(new)* | `policy` (see below) |
| `lie_shoulder_h_max` | **0.30** | *(new)* | `policy` (same as `crawl_shoulder_h_min`) |
| `sit_support_pelvis_h_min` | **0.43** | **0.68** | `midpoint_percentiles` (see below) |
| `crawl_foot_h_max` | — | **removed** | Unsatisfiable on ankle heights |
| Lie `wrist_h > 0.22` gate | — | **removed** | Excluded supine arms |

Anchors unchanged from P1 (e.g. `torso_horizontal_max=0.45`, `crawl_pelvis_h_max=0.42`, `crawl_wrist_h_max=0.22`, `lie_pelvis_h_max=0.40`, kneel/sit/yoga anchors in `configs/ego_splits.yaml`).

Each threshold has a matching entry in `threshold_provenance`. Thresholds were **not** tuned to maximize BABEL agreement.

#### Provenance detail (label-free VAL; frozen values unchanged)

**`crawl_shoulder_h_min` / `lie_shoulder_h_max` (0.30 m).** **Policy value** — not a histogram valley and **no derived physical anchor**. Reference horiz+low-pelvis pool (all VAL clips, frame stride 4): **q55 = 0.280 m**; histogram mode in **[0.25, 0.30)** (279 of 828 frames). This threshold **only** decides **lie vs crawl** within the horizontal low-pelvis band; per-class lie/crawl frame counts and BABEL confirmation of lie/crawl segments **depend on it**; **sensitivity to this value was not measured**.

**`sit_support_pelvis_h_min` (0.43 m).** **Primary evidence (all VAL, `stride_clips` 1, frame stride 8):** **1896** sit-gate frames; **floor_tail** (`pelvis_h ≤ 0.35`) **p90 = 0.3334 m** (**n = 389**); **seat_band** (`0.45 ≤ pelvis_h ≤ 0.65`) **p10 = 0.5307 m** (**n = 1162**); midpoint **0.4321 m**. Frozen **0.43 m** is **within 0.002 m** of this all-VAL midpoint and is **kept as policy**. **`derive_geometry_thresholds` code path** (every 3rd VAL clip): **598** frames, **37** clips, **17** subjects; midpoint **0.402 m** (floor_tail p90 **0.305 m**, seat_band p10 **0.499 m**).

## Floor-work evaluation set

**VAL ∪ TEST** (`--splits val,test`). Per category: segments, unique clips, unique subjects (proposals), confirmed / disagreement / unconfirmed, split by `ann_source` (`frame_ann` / `seq_ann`). Whole-file `seq_ann` segments require geometry on the same `[start_t, end_t]` range.

### Indicative, N small (VAL ∪ TEST proposals vs confirmed subjects)

Counts from `results/E/cohort_counts.json` (`val_union_test`) and confirmed rows in `floor_work_clips.csv` (unique `{subset}/{subject}`).

| Category | Proposal segments | Proposal subjects | Confirmed segments | Confirmed subjects |
|----------|-------------------|-------------------|--------------------|--------------------|
| crawl | 12 | 11 | 0 | 0 |
| kneel | 19 | 5 | 12 | 3 |
| lie | 20 | 9 | 9 | 5 |
| sit | 141 | 37 | 95 | 33 |
| yoga | 5 | 5 | 0 | 0 |

**TEST split:** **2** `floor_work_eligible` segments (**1** subject, **4.421 s**); VAL holds the other **19** (**96.382 s**, **7** subjects). Eligible set is **kneel + lie** only.

### E1 close run (`cohort_counts.json`, VAL ∪ TEST)

**Per BABEL category:** confirmed / disagreements / unconfirmed — crawl **0 / 3 / 9**, kneel **12 / 3 / 4**, lie **9 / 2 / 9**, sit **95 / 2 / 44**, yoga **0 / 0 / 5** (116 confirmed proposals total).

**Confirmed geometry classes (segments / unique subjects / duration s / largest-subject duration share):**

| Class | Segments | Subjects | Duration (s) | Largest-subject share |
|-------|----------|----------|--------------|------------------------|
| kneel | 12 | 3 | 70.792 | 0.581 |
| lie | 9 | 5 | 30.011 | 0.448 |
| sit_floor | 17 | 6 | 66.708 | 0.461 |
| sit_support | 78 | 30 | 1234.032 | 0.450 |

**`floor_work_eligible` (kneel + lie):** VAL **19** segments, **11** clips, **7** subjects, **96.382 s** (largest-subject share **0.426**); TEST **2** / **2** / **1** / **4.421 s** (share **1.0**); VAL ∪ TEST **21** / **13** / **8** / **100.803 s** (share **0.408**).

**KIT concentration (confirmed BABEL sit, diagnostic):** on VAL confirmed-sit frames (stride **4**), path-group pelvis p50 shows **KIT** dominates sample count (**8018** of ~10.4k frames); confirmed **`sit_support`** duration is spread across **30** subjects (largest share **~45%**), not a single KIT subject lock-in.

### Crawl = 0 confirmed (finding)

On **all 2223 VAL clips** (frame stride **8**, raw-metre features), crawl **pre-filter** frames (`torso_up_dot ≤ 0.45`, `pelvis_h ≤ 0.42`): **416** frames from **26** clips / **7** subjects. Of those, **`shoulder_h > 0.30`**: **145** frames from **17** clips / **7** subjects. `shoulder_h` on pre-filter frames (0.05 m bins): 54 in [0.15,0.20), 76 in [0.20,0.25), **141 in [0.25,0.30)**, 57 in [0.30,0.35), 43 in [0.35,0.40), 29 in [0.40,0.45), 8 in [0.45,0.50), 8 in [0.50,0.55).

On BABEL-**crawl** VAL segments (~**1440** labelled frames in ACCEPTANCE A), the frozen tree classifies most as **`lie`** (**419** frames) vs **`crawl`** (**6** frames) — BABEL crawl labels rarely match the crawl geometry gate. **0** crawl proposals reach segment confirmation (VAL ∪ TEST).

**Quadruped-like frames without BABEL crawl:** the stride-**8** diagnostic scan finds **145** frames (**17** clips, **7** subjects) with horizontal low pelvis and `shoulder_h > 0.30` that need not fall inside a BABEL crawl segment.

**Conclusion:** **not** “no quadruped-like horizontal low-pelvis frames in VAL.” **Crawl confirmed = 0** because **BABEL crawl segments do not match** the frozen crawl shoulder/wrist gates at ≥50% frame fraction (disagreements → `yoga_like` / `lie` / `none`), not because VAL lacks low horizontal posture.

## Tune set

`seed=0`, `fraction=0.05` → **19** train `{subset}/{subject}` ids in `splits.json` (`tune.subject_ids`), disjoint from val/test and from other train subjects.

## Foot-height rise

`rise_cm` attached per clip; **not** an exclusion for floor-work cohorts.

## Eyes_Japan floor offset

**E1b standing-foot characterisation (VAL, BABEL `stand` segments):** Eyes_Japan per-clip stand-foot medians are **p50 ≈ 0.0056 m**, **p95 ≈ 0.031 m**, **max ≈ 0.049 m** (`standing_foot_characterisation_val` in frozen `configs/support_contact_v2.yaml`, reported with `results/E/support_contact_v2_rates.json`). That does **not** support a blanket **0.07 m** standing residual on Eyes_Japan. With production thresholds, cohort **kneel** mean shin contact fraction is **≈ 0.911** from `results/E/support_contact_v2_rates.json` → `cohorts.kneel.mean_contact_fraction_per_region.shins` (**0.9107485747483427**, post patch-median non-foot speed amendment). **≈ 0.884** is the pre-amendment cohort mean: `amendment_2026_10_04_non_foot_speed.cohorts.kneel.regions.shins.mean_contact_fraction.before` (**0.8837845288980636**). **≈ 0.836** is the unweighted mean of the two frozen validation kneel clips at `h_on_m=0.05`, `z_shift_m=0` in `sensitivity_validation_clips` (`contact_fraction.shins` **0.6710526315789473** and **1.0**), not the full kneel cohort. Details: `docs/contact_labels_v2.md`.

## Legacy acceptance (joint-index fix)

Legacy overlapping yaml on **160** TEST proposals (with stretch): wrong indices **5 / 7 / 148**; correct indices **52 / 61 / 47** confirmed / disagreement / unconfirmed.

## Hashes (re-run `ego_splits` twice to verify)

Record SHA256 of `splits.json`, `floor_work_clips.csv`, `cohort_counts.json` after each mission run (`--verify-byte-stable`).

**ACCEPTANCE F (geometry-only scan):** skipped in CLI by default (`skip_geometry_only=True`) — full-index FK over VAL clips is multi-hour; BABEL-gated evaluation above is authoritative for E1.
