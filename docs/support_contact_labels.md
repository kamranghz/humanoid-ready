# Support-contact labels v2 (Track E1b)

**Module:** `hready/data/support_contact_v2.py`  
**Config:** `configs/support_contact_v2.yaml`  
**Rates:** `results/E/support_contact_v2_rates.json`

## Purpose

Foot contact (`hready.data.contact`) is foot-standing biased. v2 adds **support-region** contact bits from LBS-region lowest-vertex height/speed with the **same hysteresis as feet**. Labels are **rule-defined targets** for E3/E5 — **not** instrumented GT; **recall on kneel/lie is not validated** against hardware or force plates.

## Regions

Each region = vertices whose **max LBS weight** is on the listed joint index (same rule as foot channels). Names: `shins` (4,5), `thighs` (1,2), `pelvis_seat` (0), `back_torso` (3,6,9), `head` (12,15), `forearms` (18,19), `wrists` (20,21), `hands` (25–54). Feet stay the four `contact.py` channels.

### Knee border (shins vs thighs)

Vertices around the knee are **split by max-LBS assignment**: joint **4–5** → `shins`, joints **1–2** → `thighs`. There is no anatomical knee cap region. In kneel, the lowest points on the lower leg often sit on knee-joint-weighted geometry, so **shin vs thigh contact mostly reflects contact near the knee**, not independent shin and thigh surfaces.

## Thresholds (recorded)

**All non-foot regions use foot constants:** `h_on=0.05 m`, `h_off=0.06 m`, `v_on=0.2 m/s`, `v_off=0.25 m/s`, `min_run=3`. Heights are **lowest-vertex world z** after clip `floor_offset` (pipeline floor **z=0**). Thresholds were **not** changed on 2026-10-04.

**Rationale:** VAL stand-segment characterisation (`standing_foot_characterisation_val`, recorded with rates in `configs/support_contact_v2.yaml` / `results/E/support_contact_v2_rates.json`): subset **p50** medians — ACCAD **0.0055 m**, BMLmovi **0.0108 m**, MoSh **0.0038 m** (overall **p50 ≈ 0.0073 m**). Using **0.07 m** as `h_on` would risk labelling **body-on-body** support (e.g. thigh on calves in kneel) as floor contact.

### Amendment 2026-10-04 — non-foot speed only (feet unchanged)

**Reason:** Legacy speed followed the **argmin lowest vertex**; when that index switched frame-to-frame, horizontal speed spiked and opened spurious contact gaps (e.g. lie A9 thighs at 0.02–0.03 m with many short OFF runs; pelvis at ~0.005 m with ~0.27 s gaps).

**Config (two keys only):** `speed_definition: patch_median_same_vertex`, `patch_band_m: 0.02` (design constant fixed at amendment time; not tuned on cohort rates). **Non-foot** speed at frame `t`: patch `P_t = {v : z_{t,v} ≤ min_z_t + patch_band_m}`; `speed_t = median_{v∈P_t} ‖xy_{t,v} − xy_{t−1,v}‖ × fps`; `speed_0 = speed_1`. Heights unchanged (lowest vertex). Feet still use `contact.py` / foot channels only.

**Before/after evidence:** `results/E/support_contact_v2_rates.json` → `amendment_2026_10_04_non_foot_speed` (per cohort × region: mean contact fraction and `frac_low_h_no_contact_due_to_speed`). Example lie cohort means: thighs **0.550 → 0.613**, pelvis_seat **0.639 → 0.655**, back_torso **0.679 → 0.768**; lie `frac_low_h_no_contact_due_to_speed` on thighs **0.057 → 0.009**. `ordinary_locomotion` and `sit_support`: **no** non-foot mean-contact changes (all non-foot regions remain **0**). `patch_band_m` sensitivity **report-only** at 0.01 / 0.03 m (lie speed-gate only). Validation `squat_down`: shins/thighs/pelvis **0 before and after**.

## Floor calibration sensitivity

Validation kneel clips (recorded production thresholds except grid overrides): at `h_on_m=0.05`, `z_shift_m=0`, `results/E/support_contact_v2_rates.json` → `sensitivity_validation_clips` entries with `cohort=kneel` report per-clip `contact_fraction.shins` **0.671** (`Eyes_Japan_Dataset/aita/sitdown_standup-11-one_knee_drawn_up-aita_stageii.npz`) and **1.0** (`Eyes_Japan_Dataset/kaiwa/pose-11-bended_knees-kaiwa_stageii.npz`); unweighted mean **≈ 0.836** (not the full kneel cohort). At `z_shift_m=0.035` on the aita clip, shins **0.329**; at `z_shift_m=0.07`, shins **0** on that clip. **Interpretation:** support-region labels are only trustworthy when effective floor error is **below ~3 cm** relative to the grounded mesh; larger positive shifts remove most kneel shin contact.

Sensitivity grid uses **synthetic `z_shift` only** at **0, +0.035, +0.07 m** (report-only; production **`z_shift=0`**).

## Standing-foot characterisation (not a gate)

Per **VAL** clip: median of per-frame **min foot-channel z** over BABEL **`stand`** segment frames only, excluding segments whose labels include stand-up / transition-style act_cats. Clips with per-clip median **> 0.07 m** are flagged **`floor_uncertain`** for reporting/slicing only (counts per cohort in rates JSON; no automatic removal beyond E1 `exclude_contact`).

**Eyes_Japan / 0.07 m premise:** A blanket “0.07 m standing residual on Eyes_Japan” is **not** supported by the VAL stand-segment characterisation in the recorded config. On **ordinary-locomotion VAL** clips with usable stand segments, per-clip stand-foot medians on Eyes_Japan are **at most ~0.030 m** in this run (worst listed Eyes_Japan stand clip in `standing_foot_characterisation_val`). High residuals on some other subsets (e.g. KIT, CMU on worst clips) remain **unexplained** without further evidence.

## Anti-circularity

Floor-work eval kneel/lie segments are **not** used to set thresholds. Ordinary-locomotion **rates** and the **negative envelope** use BABEL locomotion **segment** ranges (`>= 1 s`), same keyword map as `ego_splits` (`walk`, `run`, `stand`, `turn`), seeded segment sample (see config). Segments on clips with `exclude_contact` are **skipped** in cohort means (sample size 200 → **192** processed for ordinary locomotion when eight sampled clips are excluded).

## Cohorts (rates JSON)

| Cohort | Role |
|--------|------|
| `kneel` / `lie` | Separate geometry classes from `floor_work_clips.csv` |
| `floor_work_eligible` | Pooled kneel + lie (E1 metrics set) |
| `sit_floor` / `sit_support` | Descriptive; `sit_support` excluded from floor-assuming metrics (no seat mesh) |
| `ordinary_locomotion` | Seeded BABEL loco segments on VAL (`loco_frame_stride=4`) |
| `crawl`, `yoga_like` | 0 confirmed — not evaluable |

**Mean contact fraction:** per-segment **unweighted** mean of each segment’s **per-frame** contact fraction (documented in JSON as `mean_contact_fraction_definition`).

**Duration:** `total_s` in the rates JSON sums **discretised frame windows** at 30 fps (`floor(start*fps)` … `ceil(end*fps)`). E1 `cohort_counts.json` uses `end_s - start_s` per row (e.g. floor_work_eligible **100.803 s** vs rates **100.700 s**).

## Lie: head and shins contact

On all **9** lie segments (production thresholds), **mean segment contact fraction is 0** for `head` and `shins` (see cohort `lie` in rates JSON). Per-frame lowest-height distributions over lie segment frames are in `lie_head_shin_heights` (min / p10 / p50). In this run, **both** regions stay **at or above `h_on`** on every lie-segment frame (head min ≈ **0.073 m**, shins min ≈ **0.065 m** vs `h_on=0.05 m`), so shin/head contact is **0 from height**, not from speed gating alone.

## Speed gate (characterisation only)

`speed_gate_characterisation` in the rates JSON quantifies frames with `h < h_on`, contact off, that would be on if speed did not gate (`v_on`/`v_off` disabled, same `min_run`). **Production definition unchanged.**

## Sensitivity (report only)

Validation clips: `h_on ∈ {0.05, 0.07, 0.10}` × synthetic `z_shift ∈ {0, 0.035, 0.07}` m. **Production** remains **0.05/0.06**.

## Commands

```text
python -m hready.data.support_contact_v2 --derive-config
# commit recorded configs/support_contact_v2.yaml
python -m hready.data.support_contact_v2 --report
```

Validation per-frame lines use **full-segment** hysteresis masks sliced per frame (stdout). JSON validation entries include contact **runs** on the segment window.

## Regression

`contact.py` unchanged; `contact_mism=0` vs `27cc2da` on 300 `foot_traj` clips (seed 0), recorded in rates JSON when `--report` runs.
