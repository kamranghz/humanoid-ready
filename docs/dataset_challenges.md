# Dataset challenges (measured facts)

## AMASS (SMPL-X N)

Data root: `smplx_n` under the project data tree (see `configs/paths.yaml`). Index built from on-disk scan; cached as `amass_index.json` (version 3).

### Frame rates and resampling to 30 Hz

FPS histogram over indexed clips (playback fps used for resampling): **120 Hz — 9307**; **100 Hz — 6920**; **60 Hz — 1062** (includes **129** DFaust clips with NPZ `mocap_frame_rate=120` but **60 Hz playback** override); **250 Hz — 57**; **150 Hz — 9**. **6977** clips have a playback fps that is not an integer multiple of 30 (notably 100 Hz and 250 Hz).

**DFaust playback fps:** NPZ stores `mocap_frame_rate=120`, but BABEL `dur` matches `(n-1)/60`. `amass.py` keeps `mocap_frame_rate` as metadata and sets index `fps` / `duration` from **`playback_fps_for_subset("DFaust") = 60`**. Resampling and BABEL duration gates use that **60 Hz** playback timeline, not 120 Hz.

**Gravity / flight check (scratch, one estimator):** On up to **5** BABEL jump/hop clips per subset, find flight windows as the intersection of (a) the middle **50%** of each jump/hop label span and (b) runs where SMPL-X ankle/foot joints are **> 6 cm** above the grounded floor; require **≥ 6 frames at 30 Hz** equivalent at native fps. Fit **z(t) = at² + bt + c** to grounded pelvis **z** (`trans[:,2] − floor_offset`) with **t** in seconds at **playback fps**; report **2a** (m/s²). Median **2a** by subset (native playback fps):

| Subset | Playback fps | n_windows | Median 2a (m/s²) | Min | Max | Note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| CMU | 120 | 8 | −9.60 | −10.16 | −9.37 | |
| ACCAD | 120 | 5 | −9.12 | −10.16 | −5.67 | |
| BMLmovi | 120 | 3 | −9.87 | −10.22 | −9.83 | |
| BMLrub | 120 | 20 | −9.27 | −10.62 | −1.68 | |
| DFaust | **60** | 1 | −9.68 | −9.68 | −9.68 | pooled jump clips; sparse ballistic windows |
| HDM05 | 120 | 20 | −9.23 | −10.28 | −4.14 | |
| KIT | 100 | 3 | −10.00 | −10.69 | −9.66 | |
| MoSh | 100 | 6 | −9.82 | −10.20 | −9.64 | |
| Eyes_Japan_Dataset | 120 | 37 | −10.07 | −10.72 | 1.35 | |
| SFU | 120 | 8 | −8.31 | −9.72 | −5.72 | |
| Transitions | 120 | 6 | −10.31 | −11.47 | −9.87 | |
| TotalCapture | 60 | 1 | −11.03 | −11.03 | −11.03 | |
| PosePrior | 120 | 2 | −4.49 | −9.00 | 0.01 | \|median+g\| > 2.5; try **half** fps |
| SSM | ~60 | 2 | +3.76 | +1.69 | +5.83 | \|median+g\| > 2.5; non-ballistic windows |
| EKUT | 100 | 0 | — | — | — | no qualifying windows in sample |

**DFaust 60 vs 120 (same estimator):** On five jump clips, ballistic windows appear only at **60 Hz** (example `50027_jumping_jacks`: **n=1**, median **−9.68 m/s²**); at **120 Hz** the same clips yield **n=0** qualifying windows (timestamps too short at double rate). No other subset fps overrides were changed.

Non-integer downsampling uses quaternion slerp on the time grid `t_k = k/30` s with output length `n_out = floor((n_src-1)*30/fps)+1`. Integer stride is used when `fps/30` is an integer (e.g. 120 Hz → stride 4).

Real-clip alignment vs source at mapped frames (max over checked joints):

| Source fps | Clip (example) | Method | Max rotation (geodesic, rad) | Max translation L2 (m) |
| --- | --- | --- | --- | --- |
| 120 | `ACCAD/Female1General_c3d/A10_-_lie_to_crouch_stageii.npz` | stride 4, output k vs source 4k | 5.96e-8 | 0.0 |
| 100 | `CNRS/283/-01_L_1_stageii.npz` | slerp; k=0,3,6,... vs source frame round(k*100/30) | 4.21e-8 | 0.0 |

`load_clip` returns poses as **axis-angle** (`root_orient` shape (T,3), `pose_body` shape (T,21,3)). Axis-angle can jump near pi; temporal smoothness and velocity losses should use **quaternion or rot6d** quantities, not raw axis-angle differences.

### Floor handling

Per-clip floor height: **1st percentile** of per-frame minimum vertex z on the locked_head mesh (every 10th frame plus last), using **playback fps** for subsampling. `load_clip(..., ground=True)` subtracts that offset from `transl[:, 2]`. `amass_floor.json` entries store **`playback_fps`**; cache rows without it or with a mismatch vs the index are recomputed on read (sidecar still version **1** on disk until the next write; **114** clips cached, **6** DFaust rows, **1** legacy DFaust row without `playback_fps` until touched).

Scratch check on **40** random clips (`random.seed(0)`, subsampled every 10th frame after grounding):

| Statistic | Per-frame min vertex z (m) | Per-clip min of those frame mins (m) |
| --- | --- | --- |
| min | -0.00760 | -0.00760 |
| median | 0.00895 | 1.03e-5 |
| max | 0.230 | 0.0205 |

The check uses every 10th frame of the **30 Hz** resampled clip, while the offset is estimated on every 10th frame of the **native-fps** clip (plus last), so per-clip minima can be slightly positive (max **+20.5 mm** here); penetration over all frames is measured later by the physical metrics.

Per-frame values up to about **+230 mm** come from **airborne or briefly elevated** poses (jump, step, etc.), not from the clip-wide floor offset. Across clips, the **minimum-over-frames** stays near the floor (median about **0 mm**; max about **21 mm** on this sample). Residual penetration (z < 0) on subsampled frames: **23** frames with any penetration, max depth about **7.6 mm** (smallest about **2.7e-6 m**).

**Limitation:** If the subject stays on an **elevated support for the whole clip**, the 1st-percentile rule makes that support **z = 0**; true ground is not recovered.

Floor outlier flag (`flag_floor_outliers`, default `z_thresh=0.35`): median `floor_offset` over the **50** cached floor entries in `amass_floor.json` is **-0.0246 m**; the outlier flag is **provisional** until floor height is computed for the whole index. Outlier when `abs(rec["floor_offset"] - med) > z_thresh`. **`DFaust/50002/50002_chicken_wings_stageii.npz`** has `floor_offset` **-0.612 m** (`abs(offset - med)` about **0.587 m** > 0.35; cause not investigated).

### MOYO

**388** clips, **one** subject id in the index (**`03596`**). Official MOYO train/val/test folders are **not** subject-disjoint. All MOYO clips are assigned to **train**; official `moyo_split` on each entry is kept as **metadata only**.

### Subject-level splits (beta-connected groups)

Within each subset, **group** = connected first-level folders that share the same shape vector (`betas` rounded to **1e-4**), via union-find on folder↔beta links.

- If the subset has **one** beta-connected group **or** **one distinct beta per clip** (no stable folder-level identity), the **entire subset** goes to a single split by hash of the subset name.
- Otherwise each group is hashed **80/10/10** on `SHA256(subset/group_id)`.

Final split sizes (indexed motion):

| Split | Clips | Groups | Hours |
| --- | ---: | ---: | ---: |
| train | 13860 | 368 | 45.66 |
| val | 2223 | 46 | 8.60 |
| test | 1272 | 46 | 4.58 |

Hour ratio is about **78% / 15% / 8%**, not 80/10/10, because assignment is by **group** hash (test is **4.58 h**, reported honestly as enough for evaluation-scale use).

Whole-subset assignment (hash of subset name → **train**): **MOYO**, **TCDHands**, **Transitions**.

**No leakage by folder and beta within each subset** (and globally, no rounded-beta vector appears in more than one split — see checks below). This does **not** prove absence of subject leakage in general: similar betas only show the fits differ numerically, not that people differ; the same person in two subsets or two folders with separate fits cannot be detected from betas alone.

**Independent leakage check (global, rounded betas):** **460** distinct beta vectors across the index; **0** appear in more than one of {train, val, test}.

**Cross-subset beta closeness (informational):** For each **test** group, mean beta vs nearest **train** group mean (L2 in 16-D beta space). Median nearest-train distance over **46** test groups: **2.79**. The five closest pairs are all inside **CMU** and **KIT** and may be either the same person fitted separately or similar builds; this is an **open leakage risk** for the test set. Five smallest distances:

| L2 | Test group | Nearest train group |
| --- | --- | --- |
| 0.675 | CMU / `CMU::77` | CMU / `CMU::139` |
| 1.021 | KIT / `KIT::1717` | KIT / `KIT::1747` |
| 1.341 | CMU / `CMU::40` | CMU / `CMU::06` |
| 1.422 | CMU / `CMU::63` | CMU / `CMU::138` |
| 1.466 | CMU / `CMU::09` | CMU / `CMU::138` |

### Corrupt / skipped file

One motion file failed to load: **`BMLmovi/Subject_49_F_MoSh/Subject_49_F_19_stageii.npz`** — `BadZipFile: File is not a zip file` (incomplete download). Skipped at index time.

Indexed corpus: **17,355** clips, **58.84** h total (playback timeline).

### SMPL-X version mismatch (locked_head vs v1_1)

Training uses **locked_head** (16 betas). BEDLAM-CLIFF baseline uses **v1_1**.

On **5 ACCAD** clips at frame 0, joint position difference (locked_head vs v1_1 with shared pose/trans/betas):

| Setting | Max per-clip (mm) | Mean per-clip (mm) |
| --- | --- | --- |
| v1_1 default `load_body` (10 betas) | 7.2–8.9 | 4.2–5.1 |
| v1_1 loaded with `num_betas=16` | 3.9–5.5 | 2.6–3.9 |

Going from 10 to 16 betas moves max from **7.2–8.9 mm** to **3.9–5.5 mm** and mean from **4.2–5.1 mm** to **2.6–3.9 mm**; the remaining gap at 16 betas is a **real model difference** between locked_head and v1_1, not beta count alone.

For item 13 (BEDLAM-CLIFF baseline): **v1_1 must be loaded with the beta count of the released checkpoint** (currently `load_body("v1_1")` uses **10** betas).

## BABEL (v1.0)

Release JSON under `babel_v1.0_release`: `train.json` (**6615** sequences), `val.json` (**2193**), `test.json` (**2084**), `extra_train.json` (**7921**), `extra_val.json` (**2636**). Top-level dict keyed by `babel_sid`; each value has keys `babel_sid`, `url`, `feat_p`, `dur`, `seq_ann`, `frame_ann` (and sometimes only one annotation block).

`seq_ann` / `frame_ann` are dicts with `labels` (list). Frame labels include `start_t`, `end_t`, `raw_label`, `proc_label`, `act_cat` (list of strings). Example frame segment: `walk`, `start_t=0`, `end_t=1.523`. `extra_*` files have **no** `seq_ann`/`frame_ann` content (7921 + 2636 sequences with neither usable labels).

Annotation coverage in official splits: `train` / `val` / `test` have both `seq_ann` and `frame_ann` for sequences with real labels; **3985 / 1356 / 1322** have both blocks in `train`/`val`/`test` respectively (remainder often `seq_ann` with `act_cat: null` in test).

### Mapping `feat_p` to AMASS `rel_path`

BABEL paths look like `<Prefix>/.../<name>_poses.npz`; our index uses `<Subset>/<folder>/<name>_stageii.npz`. Rules derived from data:

| BABEL prefix | AMASS subset | Strip after prefix |
| --- | --- | --- |
| BMLrub | BMLrub | `BioMotionLab_NTroje` |
| ACCAD | ACCAD | `ACCAD` |
| CMU | CMU | `CMU` |
| EyesJapanDataset | Eyes_Japan_Dataset | `Eyes_Japan_Dataset` |
| MPIHDM05 | HDM05 | `MPI_HDM05` |
| KIT | KIT | `KIT` |
| EKUT | EKUT | `EKUT` |
| MPImosh | MoSh | `MPI_mosh` |
| TCDhandMocap | TCDHands | `TCD_handMocap` |
| DFaust67 | DFaust | `DFaust_67` |
| MPILimits | PosePrior | `MPI_Limits` |
| SFU, TotalCapture, HumanEva, SSMsynced, BMLmovi, Transitionsmocap | same name (see code) | duplicate dataset folder |

Filename: `_poses.npz` or `_poses_poses.npz` (MoSh) → `_stageii.npz`. Fallbacks: unique `(subset, basename)`; then **`normalize_rel_path`** (lowercase tail, strip spaces/underscores/dashes) on the mapped AMASS-relative path if exactly one index hit.

Duration gate (required before accept): `|dur - playback_duration|` where `playback_duration` is the AMASS index duration (DFaust at **60 Hz**). On **20410** mapped pairs: median diff **0.0250 s**, max **0.1667 s**, **0** pairs above **0.5 s**. Segment times map 1:1 to `load_clip` (no `time_scale`).

**Path normalization recovery (scratch, pre-index):** among sequences that failed direct lookup, normalized paths would recover **378** ACCAD, **727** EyesJapanDataset, **0** CMU (remaining CMU `not_in_index` are absent from SMPL-X N, not naming).

### Coverage (build_babel_index)

| BABEL file | Mapped / total | With frame labels / total |
| --- | --- | --- |
| train.json | 6603 / 6615 | 6603 / 6615 |
| val.json | 2190 / 2193 | 2190 / 2193 |
| test.json | 2080 / 2084 | 1320 / 2084 |
| extra_train.json | 7904 / 7921 | 0 / 7921 |
| extra_val.json | 2633 / 2636 | 0 / 2636 |

**10113** labeled sequences cached; **39** unmatched `not_in_index` (e.g. CMU `22_23_justin/22_13` not in SMPL-X N); **11297** `no_labels` (mostly `extra_*`, plus test sequences with null `act_cat`).

Selected prefix mapped / total / labeled: KIT **6666 / 6666 / 3108**; BMLrub **5177 / 5177 / 2416**; CMU **3284 / 3319 / 1605**; BMLmovi **2812 / 2812 / 1262**; EyesJapanDataset **1273 / 1273 / 674**; ACCAD **386 / 386 / 176**; DFaust67 **199 / 199 / 90**.

### Frame labels API

`build_babel_index()` writes `cache_dir/babel_index.json` (version 3). `load_babel_labels(entry, n_frames)` returns multi-hot **uint8** `(T, K)` on the **30 Hz** grid matching `load_clip`, plus vocabulary list. Uses `frame_ann` timings when present; otherwise whole-clip `seq_ann` act categories. Overlaps are OR-ed; unlabeled frames are all-zero. Frame indices: `i0 = floor(start_t * 30)`, `i1 = floor(end_t * 30)` (**end exclusive**). Example segment `start_t=0`, `end_t=1.523` → frames **0:45**; `(end_t-start_t)*30 = 45.69` (fractional last frame not included).

**Segmentation vs clip labels:** `ann_source` is stored per entry. **Frame-level segmentation metrics (edit score, frame accuracy) must use only `ann_source == "frame_ann"`** — `seq_ann` tags span the whole clip. `print_annotation_source_report()` prints counts per BABEL file and per our train/val/test. Indexed labeled sequences: **frame_ann 6647**, **seq_ann 3466** (our train **5149 / 2840**, val **933 / 367**, test **565 / 259** frame vs seq).

**BMLrub treadmill:** **313** labeled sequences have `treadmill` in `feat_p` or `rel_path`. Root translation is nearly constant while the support foot moves at belt speed in the world frame, so velocity-based contact labels and foot-skating metrics on GT are invalid; these clips will be flagged (`is_bmlrub_treadmill_clip`) and excluded from contact labels and physical evaluation in item **5c**.

**Vocabulary:** **243** distinct `act_cat`; **K=30** = union of categories with **>= 30 min** labeled time (**25** categories) and fixed affordance channels **lean, lie, kneel, crouch, jump, stand up** (always present even below 30 min; **jump** also passes the 30 min rule, so union size is 30 not 31). Affordance support (hours, sequences with label): lean **0.18 h / 154**, lie **0.05 / 39**, kneel **0.10 / 94**, crouch **0.09 / 89**, jump **0.56 / 527**, stand up **0.23 / 420**. **F1 on channels with far below 30 min labeled time (lie, kneel, crouch, lean) is unreliable**; treat as affordance probes, not balanced classification.

Transl-only resampling (`resample_translation` at playback fps → 30 Hz) matches `load_clip` `transl` on **5** random clips (max abs difference **0 m**).

### Splits

We keep **our** beta-group train/val/test for training. On matched labeled sequences: our **train** **7989** seq (**27.13 h** BABEL `dur` sum), **val** **1300** (**5.54 h**), **test** **824** (**3.38 h**). BABEL official split vs ours (non-extra): **4450** sequences cross boundaries (e.g. BABEL **train** but our **val** **832**, BABEL **test** but our **train** **1020**).

### Semantic spot-check (scratch, 300 clips, seed 3)

Categories matched by exact name or whole token (`act_cat_matches_keyword`), not substring. Root horizontal speed from 30 Hz `transl`; pelvis height from locked_head joint 0.

| Label | n_frames | speed med (p25–p75) m/s | pelvis med m |
| --- | ---: | --- | ---: |
| walk | 17191 | 0.55 (0.12–0.88) | 0.93 |
| run | 2601 | 0.38 (0.15–1.41) | 0.92 |
| jog | 1105 | 0.88 (0.43–1.36) | 0.87 |
| stand | 9322 | 0.043 (0.013–0.13) | 0.91 |
| sit | 2473 | 0.020 (0.004–0.11) | 0.63 |
| lie / lie down | 634 | 0.043 (0.018–0.12) | 0.15 |
| jump | 2023 | 0.13 (0.065–0.26) | 0.94 |
| kneel | 288 | 0.20 (0.053–0.38) | 0.57 |
| crouch | 678 | 0.15 (0.059–0.41) | 0.80 |

Excluding treadmill `feat_p`, the same 300-clip sample gives run median **1.01 m/s** vs walk **0.66 m/s**. Global labeled-frame speeds: walk median **0.55 m/s**, run **0.20 m/s** (BMLrub treadmill walk dominates, **145815** frames at **0.088 m/s**); excluding treadmill: walk **0.72**, run **0.69**. Per-prefix run medians (n>=500 frames): CMU **1.80**, MPIHDM05 **1.26**, EyesJapanDataset **1.18**, HumanEva **1.19**; BMLrub run **0.15** (short steps on treadmill). Low global run median is mostly **in-place / treadmill** translation, not mocap timing error.

`extra_train.json` / `extra_val.json`: `seq_ann` and `frame_ann` are JSON **null** on all **7921** / **2636** sequences (e.g. `extra_train` first row: `feat_p=BMLrub/.../0020_lifting_heavy2_poses.npz`, `dur=6.77`, annotations null; **0** labels with non-null `act_cat`).
