# Results index

Every number in the README and docs comes from a file listed here, produced by a script in this repository.

| Folder / file | Produced by | Contents |
|---|---|---|
| `splits/splits.json`, `splits/cohort_counts.json`, `splits/floor_work_clips.csv` | `python -m hready.data.ego_splits` | Subject-disjoint splits, cohort counts, confirmed floor-work clips |
| `support_contact/label_rates.json` | `python -m hready.data.support_contact --report` | Support-region contact label rates per cohort and region |
| `completion/oracle_baselines.json` | `python -m hready.experiments.oracle_baselines run` | Oracle completion baselines: VAL and TEST tables, checks (oracle control) |
| `completion/evidence_locked_ablation.json` | `python -m hready.experiments.evidence_locked_ablation run` | Evidence-locked completion, ablation study (acceptance not met) |
| `completion/evidence_locked_preregistered_comparison.json` | `python -m hready.experiments.evidence_locked_preregistered` | Evidence-locked completion, pre-registered comparison (no arm passed; TEST not evaluated) |
| `robot_smoke_test/` | `python -m hready.robot.run_smoke` | Humanoid-robot smoke test metrics (rollout videos and run files are local only) |
| `checks/` | `scripts/inspect_clip.py` | Contact sheets and time series of three visually checked AMASS clips |

Checkpoint paths recorded in `completion/*.json` point to `checkpoints/oracle_completion_transformer/`,
`checkpoints/evidence_locked_ablation/` and `checkpoints/evidence_locked_preregistered_comparison/` under the data
root (not in the repository).
