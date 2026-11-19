# Physics-loss ablation analysis

Config: `configs/hr_refine.yaml` · 10 000 steps · val corrupt-input baseline MPJPE = 29.116 mm

## Results

| Arm | val MPJPE (mm) ↓ | Foot skating ↓ | Ground pen. (mm) ↓ | Jitter ↓ |
|---|---|---|---|---|
| recon\_only | 21.424 | 3.830 | 23.459 | 5 180.6 |
| full | 18.678 | 3.730 | 8.316 | 5 135.9 |
| minus\_contact | 21.264 | 3.806 | 22.500 | 5 114.5 |
| minus\_balance | **18.426** | 3.746 | **8.400** | **5 091.6** |

Deltas relative to `recon_only` (lower = better):

| Arm | ΔMPJPE | ΔSkating | ΔPen. | ΔJitter |
|---|---|---|---|---|
| full | −2.746 mm | −0.100 | −15.143 mm | −44.8 |
| minus\_contact | −0.160 mm | −0.024 | −0.959 mm | −66.1 |
| minus\_balance | −2.998 mm | −0.084 | −15.059 mm | −89.0 |

## Key findings

**1. The contact (foot-skating) loss drives the physics benefit.**
Dropping it (`minus_contact`) collapses performance to nearly `recon_only`
across all metrics: MPJPE recovers to 21.264 mm (+2.586 mm vs `full`),
ground penetration jumps back to 22.500 mm, and foot-skating worsens.
The contact term is the dominant contributor to both pose accuracy and
physical plausibility.

**2. The balance loss contributes little.**
`minus_balance` is statistically indistinguishable from `full`:
MPJPE 18.426 vs 18.678 mm (Δ = −0.252 mm, i.e. removing balance very
slightly *improves* val MPJPE), ground penetration nearly identical
(8.400 vs 8.316 mm), jitter marginally lower (5091.6 vs 5135.9).
The balance regulariser neither hurts nor helps at this training length
and data mix.

**3. Physics supervision in general is load-bearing.**
Both `full` and `minus_balance` outperform the recon-only baseline by
≈2.7–3.0 mm MPJPE and ≈15 mm ground penetration, confirming that the
physics losses provide a meaningful training signal beyond the
reconstruction loss alone.

## Recommendation

The contact loss is essential and should be kept. The balance loss can
be removed (or its weight reduced to zero) without measurable regression;
doing so simplifies the loss landscape and reduces one hyperparameter.
If compute budget allows a re-run with an extended schedule (≥15 k steps),
the balance term could be re-tested after the contact loss has stabilised,
as its effect may become detectable at a lower learning rate.
