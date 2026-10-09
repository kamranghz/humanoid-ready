# E4-v2 — plan and pre-registered rule

**Status:** owner decisions taken 2026-10-08. Arms, hyperparameters and decision rule are pre-registered in `configs/e4_v2_completion.yaml` (pre-registered 2026-10-08; amended 2026-10-08 to add the eligible arm `det_w0`, before any full v2 run). Code: `hready/eval/e4_v2.py`, `hready/metrics/paired_bootstrap.py`. Only preflight has run; the full v2 run starts on owner go. The v1 config (`configs/e4_completion.yaml`) and its pre-registered rule stay untouched.

**Post-hoc design.** v2 is designed **after seeing E4-v1 VAL results** (`docs/e4_completion.md`, `results/E/e4_completion_run.json`). Its arm choices are informed by those results, so v2 VAL numbers carry selection bias and the single v2 TEST evaluation is the only unbiased comparison. The v1 `deterministic` arm numbers are **VAL only**; its TEST has not been evaluated.

Every table row keeps the disclaimer: oracle control; floor height given (world z = 0); head pose and gravity given (D1).

## What v1 showed (VAL `all`, from `results/E/e4_completion_run.json`)

- The evidence lock helped every metric in the ablation (locked vs unlocked: full MPJPE 64.8 vs 69.6 mm, visible 12.2 vs 52.4 mm, skate 0.235 vs 0.241 m/s).
- The conditional VAE (generative) arm was worse than the deterministic arm on full MPJPE (64.8 vs 60.0 mm), hidden MPJPE (70.1 vs 64.8 mm), penetration (5.63 vs 4.43 mm), skate (0.235 vs 0.182 m/s) and ECE (0.101 vs 0.0126); it was better on GC violation (0.397 vs 0.432). Its 150000-step run diverged.
- `w_phys = 1` (on top of the generative arm) reduced penetration to 1.03 mm and GC violation to 0.331 at a full-MPJPE cost of +1.9 mm vs the generative reference arm.

## Arms (all 40000 steps, E3 protocol)

Protocol identical to E3/E4-v1: 12309-clip training list, batch 64, window 64 / stride 32, seed 0, E3 sampler, z-rotation augmentation, cached-occlusion evidence (bit-identical to the E3 path), neutral-shape FK, VAL checkpoint selection on the same 200 clips every 5000 steps. All arms use the v1 evidence lock.

| arm | model | w_phys | eligible | notes |
|---|---|---|---|---|
| `det_w0` | deterministic | 0 | yes | v1 deterministic setting, trained fresh under this config |
| `det_wphys0p3` | deterministic | 0.3 | yes | v1 `phys_terms` weights |
| `det_wphys1p0` | deterministic | 1.0 | yes | v1 `phys_terms` weights |
| `gen_stab` | stabilised generative | 0 | yes | one latent per window; free bits 0.1 nats/dim; log-variance floor −4 (v1 −8); peak lr 1e-4 (v1 4e-4); grad clip 0.5 (v1 1.0); non-finite loss or gradient norm → step skipped and logged |
| `v1_det_wphys0` (reference row) | v1 deterministic checkpoint | 0 | **no** | "v1 anchor, post-hoc selected, not eligible, no TEST"; reused without retraining, VAL only |

`eval-val` reports, for information only, the difference between `det_w0` and the v1 anchor on VAL (same settings retrained; GPU nondeterminism may make them differ). Nothing is changed based on it.

## Decision rule (pre-registered; `decision_rule` in the config)

- **Checkpoint selection per arm:** VAL only; best E3 selection metric (mean pooled MPJPE over VAL `all` + `ordinary_locomotion`, 200 seeded clips).
- **Comparison:** VAL `all`, each eligible arm vs `e3_learned` (same clips, same evidence), paired subject-cluster bootstrap, 2000 draws, seed 0. On each draw subjects are resampled with replacement and both models are pooled over the same resampled clips; the statistic is pooled(arm) − pooled(E3), the same pooled estimator as the tables (contact ECE recomputed from pooled calibration bins). **"Upper bound" = the 97.5th percentile of the two-sided 95% bootstrap interval.**
- **Gates (an arm passes only if all hold):**
  1. hidden MPJPE: upper bound of (arm − E3) < 0;
  2. full MPJPE: upper bound of (arm − E3) < 0;
  3. penetration: upper bound ≤ +5% of the E3 value;
  4. foot skate: upper bound ≤ +5% of the E3 value;
  5. contact ECE: upper bound ≤ +5% of the E3 value.
- **Report-only (not gates):** GC violation (τ 0.04947 and 0.04453), visible MPJPE, contact F1.
- **Selection among passing arms:** lowest VAL hidden MPJPE; tie → lower VAL penetration.
- **If no arm passes:** E4-v2 fails, is reported as failed, and TEST is not evaluated.
- **TEST:** evaluated once at the end, for the selected arm and `e3_learned` only; never used for any choice.
- No change to arms, budgets, hyperparameters or this rule after seeing any v2 result.

## Owner decisions (2026-10-08)

1. A deterministic arm may be the v2 main candidate, named **"E4 point-estimate completion"**; the stabilised generative arm is also run and reported; the pre-registered rule picks between them.
2. **GC is reported but not a gate.** Reasons that predate E4 results: the τ double floor-offset issue (`docs/pivot_log.md` 2026-10-07) and `gt_reference` GC violation of 0.884 on VAL `floor_work_eligible` (`results/E/e3_oracle_run.json`).
3. **"Not worse than E3"** = paired subject-cluster bootstrap vs `e3_learned`; hidden MPJPE upper bound < 0; penetration, skate, ECE upper bound ≤ +5% of the E3 value; margins fixed in the config.
4. Generative-arm hyperparameters fixed in the config before the first run: one configuration, no sweep, no retune; a failure is reported as a failure.
5. The v1 deterministic `w_phys = 0` checkpoint is a reference row (no retraining), labelled post-hoc, not eligible, no TEST. New arms: `det_wphys0p3`, `det_wphys1p0`, `gen_stab`; amendment (2026-10-08, before any full run): `det_w0` added as an eligible arm trained fresh.

## Decisions taken on the two remaining questions (most conservative option)

6. **Full MPJPE is a gate**, in the strict form (upper bound of arm − E3 < 0), like hidden MPJPE.
7. **The v1 anchor is not eligible and is not evaluated on TEST.**

All open questions from the first draft are closed (owner decisions 1–5 and decisions 6–7 above).
