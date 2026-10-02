# Dataset challenges (measured facts)

## AMASS (SMPL-X N)

Data root: `smplx_n` under the project data tree (see `configs/paths.yaml`). Index built from on-disk scan; cached as `amass_index.json` (version 2).

### Frame rates and resampling to 30 Hz

FPS histogram over indexed clips: **120 Hz — 9436**; **100 Hz — 6920**; **60 Hz — 933**; **250 Hz — 57**; **150 Hz — 9**. **6977** clips have a frame rate that is not an integer multiple of 30 (notably 100 Hz and 250 Hz).

Non-integer downsampling uses quaternion slerp on the time grid `t_k = k/30` s with output length `n_out = floor((n_src-1)*30/fps)+1`. Integer stride is used when `fps/30` is an integer (e.g. 120 Hz → stride 4).

Real-clip alignment vs source at mapped frames (max over checked joints):

| Source fps | Clip (example) | Method | Max rotation (geodesic, rad) | Max translation L2 (m) |
| --- | --- | --- | --- | --- |
| 120 | `ACCAD/Female1General_c3d/A10_-_lie_to_crouch_stageii.npz` | stride 4, output k vs source 4k | 5.96e-8 | 0.0 |
| 100 | `CNRS/283/-01_L_1_stageii.npz` | slerp; k=0,3,6,... vs source frame round(k*100/30) | 4.21e-8 | 0.0 |

`load_clip` returns poses as **axis-angle** (`root_orient` shape (T,3), `pose_body` shape (T,21,3)). Axis-angle can jump near pi; temporal smoothness and velocity losses should use **quaternion or rot6d** quantities, not raw axis-angle differences.

### Floor handling

Per-clip floor height: **1st percentile** of per-frame minimum vertex z on the locked_head mesh (every 10th frame plus last). `load_clip(..., ground=True)` subtracts that offset from `transl[:, 2]`.

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
| train | 13860 | 368 | 45.58 |
| val | 2223 | 46 | 8.60 |
| test | 1272 | 46 | 4.57 |

Hour ratio is about **78% / 15% / 8%**, not 80/10/10, because assignment is by **group** hash (test is **4.57 h**, reported honestly as enough for evaluation-scale use).

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

Indexed corpus: **17,355** clips, **58.74** h total.

### SMPL-X version mismatch (locked_head vs v1_1)

Training uses **locked_head** (16 betas). BEDLAM-CLIFF baseline uses **v1_1**.

On **5 ACCAD** clips at frame 0, joint position difference (locked_head vs v1_1 with shared pose/trans/betas):

| Setting | Max per-clip (mm) | Mean per-clip (mm) |
| --- | --- | --- |
| v1_1 default `load_body` (10 betas) | 7.2–8.9 | 4.2–5.1 |
| v1_1 loaded with `num_betas=16` | 3.9–5.5 | 2.6–3.9 |

Going from 10 to 16 betas moves max from **7.2–8.9 mm** to **3.9–5.5 mm** and mean from **4.2–5.1 mm** to **2.6–3.9 mm**; the remaining gap at 16 betas is a **real model difference** between locked_head and v1_1, not beta count alone.

For item 13 (BEDLAM-CLIFF baseline): **v1_1 must be loaded with the beta count of the released checkpoint** (currently `load_body("v1_1")` uses **10** betas).
