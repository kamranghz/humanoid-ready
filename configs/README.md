# Config index

| Config | Used by | Purpose |
|---|---|---|
| `paths.example.yaml` | all | Template for the git-ignored `paths.yaml` (data root, AMASS root, cache directory) |
| `ego_splits.yaml` | `python -m hready.data.ego_splits` | Subject splits, motion cohorts and floor-work geometry thresholds |
| `support_contact_labels.yaml` | `python -m hready.data.support_contact` | Support-region contact label thresholds and recorded rates |
| `ego_observation.yaml` | `python -m hready.data.ego_observation` | Observation simulator: head camera, noise, occlusion, seeds |
| `oracle_completion_baselines.yaml` | `python -m hready.experiments.oracle_baselines` | Oracle completion baselines (control) |
| `evidence_locked_ablation.yaml` | `python -m hready.experiments.evidence_locked_ablation` | Evidence-locked completion, ablation study |
| `evidence_locked_preregistered_comparison.yaml` | `python -m hready.experiments.evidence_locked_preregistered` | Evidence-locked completion, pre-registered comparison (arms, budgets, decision rule) |
| `hr_refine.yaml` | `python -m hready.train.hr_refine` | Spatio-temporal refinement model (HR-Refine) training |

Configs of completed experiments are kept as recorded; changes after a run are limited to names, paths and labels
(see `docs/experiments.md`).

## External folders named in configs

Checkpoints and caches live outside the repository, under `data_root` and `cache_dir` from `paths.yaml`:
`checkpoints/oracle_completion_transformer/`, `checkpoints/evidence_locked_ablation/`,
`checkpoints/evidence_locked_preregistered_comparison/`, `<cache_dir>/grounded_motion_30hz/` (motion cache, config key
`memmap.rel_dir`) and `<cache_dir>/self_occlusion_30hz/` (occlusion cache, config key `occlusion_cache.rel_dir`).
