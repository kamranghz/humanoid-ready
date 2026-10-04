# Support-contact labels v2 (Track E1b)

**Module:** `hready/data/support_contact_v2.py`  
**Config:** `configs/support_contact_v2.yaml`  
**Rates:** `results/E/support_contact_v2_rates.json`

## Purpose

Foot contact (`hready.data.contact`) is foot-standing biased. v2 adds **support-region** contact bits from LBS-region lowest-vertex height/speed with the **same hysteresis as feet**. Labels are **rule-defined targets** for E3/E5 — **not** instrumented GT; **recall on kneel/lie is not validated** against hardware.

## Regions

Each region = vertices whose **max LBS weight** is on the listed joint index (same rule as foot channels). Names: `shins` (4,5), `thighs` (1,2), `pelvis_seat` (0), `back_torso` (3,6,9), `head` (12,15), `forearms` (18,19), `wrists` (20,21), `hands` (25–54). Feet stay the four `contact.py` channels.

## Thresholds (frozen)

**All non-foot regions use foot constants:** `h_on=0.05 m`, `h_off=0.06 m`, `v_on=0.2 m/s`, `v_off=0.25 m/s`, `min_run=3`. Heights are **lowest-vertex world z** after clip `floor_offset` (pipeline floor **z=0**). No extra free parameter.

**Rationale:** Standing-foot residual on reliably grounded subsets is small (e.g. Eyes_Japan stand-segment medians p95 ~0.021 m, max ~0.030 m; BMLrub ~0.026 m; MoSh ~0.005 m). Using **0.07 m** would risk labelling **body-on-body** support (e.g. thigh on calves in kneel, lowest thigh vertex ~0.06 m) as floor contact.

## Standing-foot characterisation (not a gate)

Per **VAL** clip: median of per-frame **min foot-channel z** over BABEL **`stand`** segment frames only, excluding segments whose labels include stand-up / transition-style act_cats. Reported distributions overall and per subset. Clips with per-clip median **> 0.07 m** are flagged **`floor_uncertain`** for reporting/slicing only (no removal unless already excluded by E1 `exclude_contact`).

High residuals on some subsets (e.g. KIT ~0.80 m, CMU ~0.58 m on worst clips) are listed for inspection; cause **unexplained** without further evidence.

## Anti-circularity

Floor-work eval kneel/lie segments are **not** used to set thresholds. Ordinary-locomotion **rates** and the **negative envelope** use BABEL locomotion **segment** ranges (`>= 1 s`), same keyword map as `ego_splits` (`walk`, `run`, `stand`, `turn`), seeded segment sample (see config).

## Cohorts

| Cohort | Role |
|--------|------|
| `floor_work_eligible` | kneel + lie segments from `floor_work_clips.csv` |
| `sit_floor` / `sit_support` | descriptive; `sit_support` excluded from floor-assuming metrics (no seat mesh) |
| `ordinary_locomotion` | seeded BABEL loco segments on VAL |
| `crawl`, `yoga_like` | 0 confirmed — not evaluable |

## Sensitivity (report only)

Validation clips: `h_on ∈ {0.05, 0.07, 0.10}` × synthetic `z_shift ∈ {0, 0.035, 0.07}` m. **Production** remains **0.05/0.06**.

## Commands

```text
python -m hready.data.support_contact_v2 --derive-config
# commit frozen configs/support_contact_v2.yaml
python -m hready.data.support_contact_v2 --report
```

Per-frame validation detail prints to **stdout**; JSON is summary-only.

## Regression

`contact.py` unchanged; `contact_mism=0` vs `27cc2da` on 300 `foot_traj` clips (seed 0).
