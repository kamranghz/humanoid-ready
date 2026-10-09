# Track E4-v1 — improved completion on the oracle evidence schema

**Status:** attempt 1 (E4-v1) — **acceptance NOT met** (see Results). Oracle **control** (same evidence path as E3, `docs/e3_oracle.md`). E3 code, config and results are the untouched reference; E4 imports them.

**Disclaimer on every table row:** `oracle control; floor height given (world z = 0); head pose and gravity given (D1)`.

**Command:** `python -m hready.eval.e4_completion run` (config `configs/e4_completion.yaml`; results `results/E/e4_completion_run.json`). Subcommands: `build-occ`, `verify-occ`, `preflight`, `train-ablations`, `eval-ablation`, `choose-main`, `train-main`, `eval-table`, `generative-check`, `leak-check`.

## What is identical to E3

Evidence schema (`obs`, `rig`), E2-A simulator settings, splits and the 12309-clip training list, canonical frame, z-rotation augmentation, window 64 / stride 32 with overlap averaging, training sampler and seed, batch size 64, neutral-shape FK, cohorts, metrics (pooled MPJPE slices, sole-proxy physical metrics, τ 0.04947 primary + 0.04453 sensitivity, subject-cluster bootstrap CIs), VAL-only model selection on `all` + `ordinary_locomotion` (200 seeded VAL clips), and the leak protocol.

**Occlusion cache.** The E2-A capsule self-occlusion test depends only on each frame's clean joints and head camera, and dominates simulation time. E4 stores it per clip (`<cache_dir>/e4_occlusion_30hz/`) and runs the unchanged E2-A simulator with the mask served from that array. `verify-occ` checks that evaluation evidence (whole clips) and training items are bit-identical to the E3 path.

## Design choices

1. **Evidence lock (residual on top of evidence).** Output joints `J = J_fk + smooth_t(vis·(obs − J_fk)) / (smooth_t(vis) + κ)`, a visibility-masked temporal Gaussian (σ 1.5 frames, radius 4, κ 0.05) per joint and axis, in the window's canonical frame. On visible stretches the output follows the temporally smoothed evidence (this also averages out per-frame evidence noise, which a hard copy would turn into foot skate); hidden frames next to visible ones take a decaying share of the residual; long hidden stretches keep the FK prediction. The lock reads only evidence tensors. SMPL-X parameters (transl, 6D rotations) stay the motion output; the locked joints are what the tables score. Ablation: lock off.
2. **Generative completion: conditional VAE.** Evidence encoder (4 layers, d 256) → learned conditional prior over per-frame latents (32-d); training-only posterior encoder reads GT joints, root rotation and contact; decoder (4 layers) → transl, root/body 6D, 4 contact logits. `forward(obs, rig)` decodes the prior mean (deterministic point estimate, used in every table); `sample(obs, rig, n)` draws from the prior. KL weight 0.001. Chosen over diffusion because sampling and training stay cheap at this scale. Ablation: deterministic (latent fixed at 0, no posterior, no KL).
3. **Contact head kept; ground-aware terms as an arm.** `w_phys > 0` adds item-3 losses on the sole-proxy foot channels of the output joints: squared penetration hinge (×100), GT-contact-weighted foot skating (×1), CoM-in-support balance (×1, on 16 seeded in-contact frames per step because `balance_com_in_support` loops in Python). Default `w_phys = 0`.
4. **Budgets.** Ablations: 40000 steps each (= E3's budget). Main: 150000 steps, cosine to 2% LR. Fixed before any ablation result.

## Rows reported

- Ablation table (VAL `all` + `ordinary_locomotion`, full VAL): `ref_locked_gen_phys0` and one row per changed factor (`unlocked`, `deterministic`, `phys`), identical 40000-step budget.
- **Main-arm rule (frozen before any ablation result):** the main arm uses the `phys` arm's `w_phys` if, on VAL `all`, its penetration and skate are not worse than the reference and its full MPJPE is at most +1.0 mm worse; otherwise `w_phys = 0` (`choose-main`).
- Main table (VAL and TEST separate): `heuristic`, `e3_learned` (E3 checkpoint), **`e4_same_budget`** (the ablation run with the main-arm config at E3's 40000-step budget, separating design from training length), **`e4_main`** (same config, 150000 steps), `gt_reference`.
- `floor_work_eligible`: indicative only (11 VAL clips, 2 TEST clips), per-subject table, never used for selection.
- `generative-check`: prior-sample spread on hidden vs visible joints and best-of-K (uses GT to pick a sample; a diversity diagnostic, never an accuracy claim).

## Results — E4-v1 (attempt 1)

All numbers below come from `results/E/e4_completion_run.json` except the KL history, which is only in the training log (not in the repository). Pooled over clips; full-MPJPE CIs are subject-cluster bootstrap 95%.

**Acceptance NOT met.**
- `e4_same_budget` (main-arm config `ref_locked_gen_phys0`, 40000 steps = E3 budget) beats `e3_learned` on VAL full MPJPE (64.8 vs 66.7 mm), visible MPJPE (12.2 vs 59.5) and penetration (5.63 vs 5.98 mm), but **fails skate** (0.235 vs 0.202 m/s). Its hidden MPJPE is also worse (70.1 vs 67.4 mm).
- `e4_main` is **not a 150000-step result**: it is the step-25000 checkpoint (best VAL selection metric 66.13 mm) of a 150000-step run that diverged. VAL selection metric by step: 5000: 74.3, 10000: 69.8, 15000: 68.3, 20000: 66.2, 25000: 66.1, 30000: 112.6, 35000: 191.1, 40000: 142.3, …, 150000: 185.0. It is worse than `e3_learned` on VAL full MPJPE (69.6), penetration (9.09) and skate (0.255).
- KL term of the main run (training log, 500-step means): 34 nats at step 30000, 294 at step 40000, maximum 64,927,806 at step 44500, still 18,268 at step 90000 (it fluctuated, e.g. 33 at step 70000). The two generative 40000-step ablation arms stayed between 11 and 24 nats. **Unverified hypothesis:** the 150000-step cosine keeps the learning rate high for longer (3.77e-4 at step 25000 vs 1.40e-4 in the 40000-step reference arm, from the log), and the prior log-variance floor of −8 lets the KL term grow without bound. Neither cause was tested.
- Main-arm rule (frozen before any ablation): the `phys` arm was excluded because its VAL full MPJPE (66.75 mm) exceeded the reference arm (64.82 mm) by more than +1.0 mm; its penetration and skate checks passed. Main arm therefore used `w_phys = 0`.

#### Main table — VAL `all`

| row | clips | full MPJPE mm [95% CI] | visible | hidden | penetration mm | skate m/s | GC viol (τ 0.04947 / 0.04453) | contact F1 | ECE | disclaimer |
|---|---|---|---|---|---|---|---|---|---|---|
| heuristic | 2223 | 141.3 [132.9, 153.5] | 20.3 | 153.5 | 0.37 | 0.605 | 0.073 / 0.098 | 0.578 | n/a | oracle control; floor height given (world z = 0); head pose and gravity given (D1) |
| e3_learned | 2223 | 66.7 [59.0, 76.0] | 59.5 | 67.4 | 5.98 | 0.202 | 0.426 / 0.492 | 0.941 | 0.013 | oracle control; floor height given (world z = 0); head pose and gravity given (D1) |
| e4_same_budget | 2223 | 64.8 [56.4, 74.5] | 12.2 | 70.1 | 5.63 | 0.235 | 0.397 / 0.457 | 0.913 | 0.101 | oracle control; floor height given (world z = 0); head pose and gravity given (D1) |
| e4_main | 2223 | 69.6 [61.6, 79.0] | 12.8 | 75.3 | 9.09 | 0.255 | 0.382 / 0.451 | 0.909 | 0.108 | oracle control; floor height given (world z = 0); head pose and gravity given (D1) |
| gt_reference | 2223 | 0.0 [0.0, 0.0] | 0.0 | 0.0 | 0.00 | 0.027 | 0.169 / 0.263 | 0.968 | n/a | oracle control; floor height given (world z = 0); head pose and gravity given (D1) |

#### Main table — TEST `all` (not used for any decision)

| row | clips | full MPJPE mm [95% CI] | visible | hidden | penetration mm | skate m/s | GC viol (τ 0.04947 / 0.04453) | contact F1 | ECE | disclaimer |
|---|---|---|---|---|---|---|---|---|---|---|
| heuristic | 1272 | 138.3 [125.8, 154.0] | 20.0 | 151.7 | 0.39 | 0.584 | 0.083 / 0.111 | 0.689 | n/a | oracle control; floor height given (world z = 0); head pose and gravity given (D1) |
| e3_learned | 1272 | 64.6 [61.0, 68.8] | 59.4 | 65.2 | 16.88 | 0.167 | 0.479 / 0.546 | 0.962 | 0.003 | oracle control; floor height given (world z = 0); head pose and gravity given (D1) |
| e4_same_budget | 1272 | 61.5 [57.3, 66.0] | 11.5 | 67.1 | 14.80 | 0.207 | 0.407 / 0.514 | 0.941 | 0.078 | oracle control; floor height given (world z = 0); head pose and gravity given (D1) |
| e4_main | 1272 | 66.5 [63.3, 70.2] | 12.1 | 72.7 | 21.05 | 0.230 | 0.447 / 0.516 | 0.939 | 0.080 | oracle control; floor height given (world z = 0); head pose and gravity given (D1) |
| gt_reference | 1272 | 0.0 [0.0, 0.0] | 0.0 | 0.0 | 0.01 | 0.019 | 0.095 / 0.168 | 0.983 | n/a | oracle control; floor height given (world z = 0); head pose and gravity given (D1) |

#### Ablations (hidden MPJPE and ECE included) — VAL `all`, 40000 steps each, same seed and protocol

| row | clips | full MPJPE mm [95% CI] | visible | hidden | penetration mm | skate m/s | GC viol (τ 0.04947 / 0.04453) | contact F1 | ECE | disclaimer |
|---|---|---|---|---|---|---|---|---|---|---|
| ref_locked_gen_phys0 | 2223 | 64.8 [56.4, 74.5] | 12.2 | 70.1 | 5.63 | 0.235 | 0.397 / 0.457 | 0.913 | 0.101 | oracle control; floor height given (world z = 0); head pose and gravity given (D1) |
| unlocked | 2223 | 69.6 [60.9, 79.5] | 52.4 | 71.3 | 6.38 | 0.241 | 0.411 / 0.464 | 0.913 | 0.103 | oracle control; floor height given (world z = 0); head pose and gravity given (D1) |
| deterministic | 2223 | 60.0 [52.1, 68.7] | 12.2 | 64.8 | 4.43 | 0.182 | 0.432 / 0.503 | 0.943 | 0.013 | oracle control; floor height given (world z = 0); head pose and gravity given (D1) |
| phys | 2223 | 66.7 [58.5, 76.1] | 12.5 | 72.2 | 1.03 | 0.192 | 0.331 / 0.393 | 0.926 | 0.077 | oracle control; floor height given (world z = 0); head pose and gravity given (D1) |

Every table row carries the disclaimer: oracle control; floor height given (world z = 0); head pose and gravity given (D1).

#### `deterministic` arm (VAL) vs `e3_learned` (VAL), metric by metric (lower is better)

| metric | deterministic | e3_learned | outcome |
|---|---|---|---|
| full MPJPE (mm) | 60.0 | 66.7 | beats |
| visible MPJPE (mm) | 12.2 | 59.5 | beats |
| hidden MPJPE (mm) | 64.8 | 67.4 | beats |
| penetration (mm) | 4.43 | 5.98 | beats |
| skate (m/s) | 0.182 | 0.202 | beats |
| GC violation (τ 0.04947) | 0.432 | 0.426 | loses |
| contact ECE | 0.0126 | 0.0130 | ties at 3 decimals (0.0004 lower) |

The deterministic arm is a v1 ablation, not the v1 main arm; its numbers are VAL only (TEST not evaluated).

**`w_phys = 1` arm vs `e3_learned` (VAL):** full MPJPE ties at 0.1 mm (66.75 vs 66.68 mm, 0.07 mm higher); beats E3 on penetration (1.03 vs 5.98 mm), skate (0.192 vs 0.202 m/s) and GC violation (0.331 vs 0.426); loses on hidden MPJPE (72.2 vs 67.4 mm) and ECE (0.077 vs 0.013). It was excluded only by the frozen +1.0 mm-vs-reference rule.

**floor_work_eligible (indicative only; VAL 11 clips, TEST 2 clips):** VAL full MPJPE e3_learned 125.3, e4_same_budget 136.0, e4_main 147.2 mm; VAL penetration 7.79, 27.09, 47.39 mm.

#### Generative check (`e4_main` step-25000 checkpoint, VAL, 50 clips, 8 prior samples)

Point estimate hidden MPJPE 74.6 mm; mean of samples 74.6 mm; best-of-8 hidden 51.2 mm (uses GT to pick a sample: a diversity diagnostic, never an accuracy claim). Sample spread: hidden joints 43.7 mm, visible joints 46.1 mm. Visible-joint spread is not reduced by the evidence lock: independent per-frame latents make samples differ frame to frame, and the temporal lock passes that high-frequency variation through. The point estimate (prior mean) is not affected.

#### Leak check (both reported models)

`ref_locked_gen_phys0` (= `e4_same_budget`) and `main`: forward parameters are exactly `obs, rig`; replacing hidden-joint slots with perturbed-GT or random values leaves the output (including evidence-locked joints) bit-identical (max diff 0.0); raw hidden 2D keypoints do differ (negative control, 11.73); moving one visible joint 5 cm changes the output (0.0405 / 0.0217); all 7 negative controls rejected. `verify-occ`: cached-occlusion evidence bit-identical to the E3 path on 40 eval clips (16254 frames) and 64 training items.

#### E3 regression

E3 code, config and `results/E/e3_oracle_run.json` are unchanged; the `e3_learned` rows recomputed inside the E4 table are identical to the committed E3 table for every VAL and TEST cohort.
