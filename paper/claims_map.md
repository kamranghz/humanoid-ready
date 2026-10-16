# paper/claims_map.md — Pre-registered Analysis Plan

**Pre-registered: 2026-10-16**
**Status: LOCKED — thresholds and analysis plan recorded before any R6/R7 runs.**
**Do not change thresholds or analysis decisions after R6 runs begin.**

---

## 0. Purpose

This file pre-registers the complete analysis plan for the paper
*"Is Humanoid Executability a Reliable Benchmark for Human Motion Recovery?"*
before any cross-engine simulation runs (R6, R7).
Every threshold, metric, and statistical test is recorded here.
Negative or mixed findings are reported as found.

---

## 1. Evaluation data

- **Test set:** 3DPW test split (video + GT SMPL-X, to be downloaded per R3).
- **Upper bound:** AMASS ground-truth sequences as "perfect HMR" (no estimation error).
- **HMR methods under evaluation:** GVHMR, WHAM, TRAM, BEDLAM-CLIFF (all with their
  own released checkpoints, labeled as baselines). HR-HMR is included if training is
  complete and honest results are available; otherwise reported as "not evaluated."
- **Retargeting:** SMPL-X locked_head 16-beta → Unitree G1 via the open-source retargeter
  chosen at R2 (path recorded in docs/decisions.md). Retargeting error is reported
  separately from HMR error using AMASS GT as input.

---

## 2. Controller

- **Primary:** open-source general motion-tracking policy for Unitree G1, chosen at R2,
  running in all three engines under a configuration-fair protocol (same PD gains,
  same timestep, same G1 MJCF/URDF, recorded in docs/decisions.md).
- **Fallback:** PD-only tracking (no learned policy). If the fallback is used, every
  result is labeled "PD-only tracking."

---

## 3. Physics engines

| Engine | Role | Notes |
|--------|------|-------|
| Newton (Isaac Lab) | **Primary** — all single-engine results reported here | default per AGENTS.md rule 0 |
| PhysX (Isaac Lab) | Cross-engine study (R7) | same G1 asset, same gains |
| MuJoCo | Cross-engine study (R7) | same G1 MJCF, same gains |

Configuration-fairness record: one committed file `paper/engine_config.md` documents
every parameter that was matched across engines before R7 runs begin.

---

## 4. Executability metrics and PASS/FAIL thresholds

Thresholds are set from physical reasoning and published humanoid-locomotion literature
before seeing any R6/R7 results. They are not tuned after the fact.

### 4.1 Primary metrics

| Metric | Symbol | Unit | PASS threshold | FAIL threshold | Rationale |
|--------|--------|------|---------------|----------------|-----------|
| Time to fall | TTF | s | ≥ 3.0 | < 1.0 | clip is ~3 s; full clip without fall = PASS |
| Root tracking error | RTE | m | ≤ 0.15 | > 0.40 | 15 cm ≈ one foot width; 40 cm = gross failure |
| Foot slip RMS | FS | m/s | ≤ 0.10 | > 0.30 | skating artifact; 10 cm/s is near-imperceptible |
| Torque saturation rate | TSR | fraction | ≤ 0.05 | > 0.20 | fraction of timesteps any joint at limit |
| Assist wrench norm | AW | N | ≤ 5.0 | > 50.0 | external force needed to keep robot on reference |

### 4.2 Per-clip verdict

```
PASS   : TTF >= 3.0  AND  RTE <= 0.15  AND  FS <= 0.10  AND  TSR <= 0.05
FAIL   : TTF <  1.0  OR   RTE >  0.40  OR   FS >  0.30  OR   TSR >  0.20
REVIEW : all others (between PASS and FAIL bands)
```

The verdict is written per clip before any statistical analysis.

### 4.3 Repeatability

Each clip is run **3 times** in each engine with the same seed.
Within-engine verdict agreement across the 3 runs is reported.
A clip with disagreeing verdicts across 3 same-engine runs is flagged
`UNSTABLE` and excluded from cross-engine agreement analysis.

---

## 5. Pose and retargeting metrics

| Metric | Symbol | Unit | Notes |
|--------|--------|------|-------|
| MPJPE | — | mm | 14-joint common subset with GT |
| PA-MPJPE | — | mm | Procrustes-aligned |
| Retargeting error | RE | mm | AMASS GT → G1 → back-project; quantifies IK loss |

These are computed per-method on the 3DPW test set
using the same GT alignment as published benchmarks.
Published numbers are cited; our numbers are reported alongside them.

**Note on AMASS GT as upper bound:** AMASS GT has zero HMR error but nonzero
retargeting error (RE > 0). HMR methods have nonzero HMR error and nonzero RE.
The two error sources are reported separately; executability of AMASS GT measures
the ceiling imposed by retargeting alone, not by pose estimation.

---

## 6. Analysis plan

### 6.1 Method-level analysis

For each of the N methods (GVHMR, WHAM, TRAM, BEDLAM-CLIFF, ± HR-HMR, + AMASS GT):

- Mean and std of each executability metric across all test clips.
- Mean MPJPE, PA-MPJPE.
- Method-level PASS/FAIL/REVIEW counts and fractions.

**Ranking correlation (method level):**
- Spearman ρ between method rank by MPJPE and method rank by each executability metric
  (TTF, RTE, FS, TSR) in Newton.
- 95 % bootstrap confidence intervals: 10,000 bootstrap resamples over clips,
  cluster-blocked by 3DPW subject (no subject appears in two bootstrap samples).
- Report ρ, 95 % CI, and p-value (two-tailed permutation test, 10,000 permutations).

### 6.2 Clip-level analysis

For each pair (MPJPE, executability metric):

- Spearman ρ over all (clip, method) pairs.
- Kendall τ over all (clip, method) pairs.
- Subject-cluster bootstrap CIs as above.
- Scatter plot: x = MPJPE, y = executability metric, colored by method.
- Highlight: clips where MPJPE rank and PASS/FAIL verdict disagree
  (low MPJPE + FAIL, or high MPJPE + PASS).

### 6.3 Cross-engine stability (R7)

For each pair of engines {Newton vs PhysX, Newton vs MuJoCo, PhysX vs MuJoCo}:

**Method-level:**
- Kendall τ of method rankings by each executability metric.
- Report τ, 95 % bootstrap CI, p-value.

**Clip-level:**
- PASS/FAIL/REVIEW agreement rate (fraction of clips with identical verdict).
- Cohen's κ for PASS/FAIL (REVIEW mapped to its nearest neighbor by metric value
  if needed; mapping rule recorded before R7 runs).
- Kendall τ of clip-level metric values between engines.

**Within-engine repeatability:**
- For each engine separately: Kendall τ of clip rankings across the 3 repeated runs.
- Expected to be high (> 0.95); deviations flagged.

### 6.4 HR-Refine effect (R8)

For each HMR method:
- Executability metrics before and after HR-Refine.
- Change in PASS/FAIL verdict count.
- Is the improvement (or non-improvement) engine-stable?
  Kendall τ of per-clip improvement rank across Newton/PhysX/MuJoCo.

---

## 7. Statistical decisions (recorded before results)

- Significance level α = 0.05 (not adjusted for multiple comparisons;
  all comparisons reported; multiplicity noted in paper).
- Minimum detectable correlation: |ρ| or |τ| ≥ 0.30 is considered
  practically meaningful for the paper's claims.
- If N methods < 5, method-level ranking correlation is reported
  descriptively only (no CI or p-value, as bootstrap is unreliable at N < 5).
- Clips where TTF < 0.5 s (robot collapses immediately) are reported separately
  as "immediate failure" and included in all analyses.

---

## 8. What we will claim (pre-registered)

The paper makes these claims only if the data support them:

| Claim | Pre-registered decision rule |
|-------|------------------------------|
| C1: MPJPE ranking does not fully predict executability ranking | Spearman ρ (method-level, Newton) has 95 % CI that does not exclude 0 OR |ρ| < 0.70 |
| C2: Clip-level MPJPE-executability correlation is weaker than method-level | clip-level |ρ| < method-level |ρ| − 0.15 |
| C3: Cross-engine verdicts are stable for method rankings | Kendall τ ≥ 0.80 for all engine pairs, Newton primary |
| C4: Cross-engine verdicts are less stable at clip level | clip-level PASS/FAIL agreement < 0.90 for at least one engine pair |
| C5: HR-Refine improves executability | PASS count increases for ≥ 3 of N methods; improvement is engine-stable (τ ≥ 0.70) |

If a claim's decision rule is not met, the finding is reported as-is
(e.g., "MPJPE and executability rankings are well-correlated at method level").
No claim is dropped or reworded after seeing results.

---

## 9. Reporting commitments

- All raw per-clip metrics saved to `results/paper/newton/`, `results/paper/cross_engine/`.
- Tables and figures generated from scripts in `scripts/paper/`; no manual edits to numbers.
- Every number in the paper draft traceable to a file in `results/paper/`.
- `paper/engine_config.md` committed before R7 runs.
- This file (`paper/claims_map.md`) committed with a dated entry before R6 runs.

---

## 10. Limitations (pre-acknowledged)

- 3DPW test set is small (~35 sequences); method-level analyses have low power.
- MuJoCo and PhysX configurations may not be perfectly matched despite best effort;
  any unmatched parameter is documented in `paper/engine_config.md`.
- HR-HMR is trained on only 2 BEDLAM scenes from random init; its results
  are reported at that scale and not compared to published benchmarks without caveat.

**Not limitations:**
- The open-source G1 tracking policy is a feature, not a limitation: it makes the
  evaluation fully reproducible by any reader. Noted in paper as a design choice.
- AMASS GT retargeting error is handled in §5 as a documented design choice,
  not an oversight.

---

*Pre-registered 2026-10-16. This file must not be modified after R6 runs begin.*
*Any amendment requires a dated entry below with reason.*

## Amendments

*(none)*
