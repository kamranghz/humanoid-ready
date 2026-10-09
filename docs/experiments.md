# Experiment registry

One row per experiment, stage, decision or work package, under its descriptive name. The **old name / code** column
is the only place in the repository where the earlier step codes (track letters, step numbers, attempt and version
labels, decision and clarification codes, work-package letters, paper task numbers) still appear; they are kept so
that older notes and the git history can be read. Everything else in the repository uses the descriptive names.
Commit IDs are deliberately not recorded here.

## Egocentric motion track (main path)

| Descriptive name | Old name / code | Config | Results file | Status |
|---|---|---|---|---|
| Egocentric motion track | Track E, extension track E | — | — | active |
| Project scoping | P0 | — | — | done |
| Project definition document | E-doc, project definition v2 | — | `docs/project_definition.md` | approved |
| Tracker synchronization | E-sync | — | `AGENTS.md` | done |
| Split and cohort correction | P1, P1b, P1c | `configs/ego_splits.yaml` | `results/splits/` | done |
| Module audit | E0 | — | `docs/module_audit.md` | done |
| Subject splits and cohorts | E1 | `configs/ego_splits.yaml` | `results/splits/splits.json`, `results/splits/cohort_counts.json`, `results/splits/floor_work_clips.csv` | done |
| Support-contact labels | E1b, support-contact labels v2 | `configs/support_contact_labels.yaml` | `results/support_contact/label_rates.json` | done |
| Observation simulator (oracle evidence) | E2-A | `configs/ego_observation.yaml` | — (leak and byte-stability checks run on demand) | done |
| Egocentric perception stage | E2-B | — | — | not started |
| Egocentric render set | E2-B1, B1 | — | — | not started |
| Perception baseline measurement | E2-B2, B2 | — | — | not started |
| Perception training | E2-B3, B3 | — | — | not started |
| Perception model selection | E2-B4, B4 | — | — | not started |
| Realistic observation model | E2-C | — | — | not started |
| Oracle completion baselines | E3 (oracle mode) | `configs/oracle_completion_baselines.yaml` | `results/completion/oracle_baselines.json` | done (oracle control) |
| Completion baselines on realistic observations | E3 (perceived mode) | — | — | not started |
| Evidence-locked completion | E4 | — | — | closed as an oracle-evidence study |
| Evidence-locked completion, ablation study | E4 attempt 1, E4-v1, v1 | `configs/evidence_locked_ablation.yaml` | `results/completion/evidence_locked_ablation.json` | acceptance not met |
| Evidence-locked completion, pre-registered comparison | E4 attempt 2, E4-v2, v2 | `configs/evidence_locked_preregistered_comparison.yaml` | `results/completion/evidence_locked_preregistered_comparison.json` | no arm passed the pre-registered rule; TEST not evaluated |
| Decoupled contact grounding | E4b | — | — | not started |
| End-to-end evaluation | E5 | — | — | not started |
| Real egocentric validation | E5b | — | — | not started |
| Humanoid simulation asset | E6a | — | — | not started |
| Simulation tracking training | E6 | — | — | not started |
| Refinement comparison (none / kinematic / simulation / both) | E7 (arms none, K, S, K+S) | — | — | not started |
| Qualitative review | E8 | — | — | not started |
| Claims map | E9 | — | — | not started |
| Environment setup | S1–S4 (S3 environment decision, S4 package pins) | — | `env/versions.lock` | done |

## Decisions and clarifications (`docs/project_definition.md`)

| Descriptive name | Old name / code |
|---|---|
| Given head pose and gravity | D1 |
| Perception variants | D2 |
| Egocentric data plan | D3 |
| Two-stage physics refinement | D4 |
| Rig inputs alongside evidence | [C1] |
| Oracle evidence as a control | [C2] |
| Perception-matched training evidence | [C3] |
| Own perception model / pretrained perception baseline / adapted pretrained backbone | P-A / P-B / P-C |
| Kinematic refinement / simulation refinement | K / S |
| First system version | v1 (of the system) |

## Result row and arm labels

| Descriptive label (in configs, results and code) | Old label |
|---|---|
| `oracle_transformer` (oracle completion transformer) | `e3_learned` |
| `evidence_locked_same_budget` | `e4_same_budget` |
| `evidence_locked_main` | `e4_main` |
| `paired_vs_oracle_transformer` | `paired_vs_e3_learned` |
| `ablation_det_wphys0` (ablation-study deterministic reference row) | `v1_det_wphys0` ("v1 anchor") |
| `ablation_anchor`, `det_w0_vs_ablation_anchor_informational` | `v1_anchor`, `det_w0_vs_v1_anchor_informational` |
| `comparison_failed` | `v2_failed` |
| config key `baseline_config` | `e3_config` |
| gate key `upper_bound_le_frac_of_reference` | `upper_bound_le_frac_of_e3` |
| disclaimer "head pose and gravity given (rig input)" | "head pose and gravity given (D1)" |
| `foot_regression_contact_mism_reference` | old key named after the reference revision of `contact.py` |

Unchanged labels: `heuristic`, `gt_reference`, `ref_locked_gen_phys0`, `unlocked`, `deterministic`, `phys`, `main`,
`det_w0`, `det_wphys0p3`, `det_wphys1p0`, `gen_stab`.

## Work packages and paper tasks (`AGENTS.md` §3, §6)

| Descriptive name | Old name / code |
|---|---|
| 3D pose model | WP-A1 |
| Distributed temporal training | WP-A2 |
| Physics-loss study | WP-A3 |
| Robustness (motion blur, occlusion, crowding) | WP-B1 |
| Contact, torques and support surfaces | WP-B2 |
| Motion-prior regularization | WP-B3 |
| Auxiliary heads and sensor fusion | WP-C1 |
| Data-pipeline bottlenecks | WP-C2 |
| Robot pipeline | WP-D |
| Paper tasks: literature review, tracking-policy selection, external video data, method runs, analysis plan, Newton runs, cross-engine comparison, refinement effect, draft, submission | R1, R2, R3, R4, R5, R6, R7, R8, R9, R10 |

Planned result folders named in `AGENTS.md`: `results/pose_model/` (was `results/A1/`), `results/refinement_scaling/`
(`results/A2/`), `results/physics_loss_ablation/` (`results/A3/`), `results/robustness/` (`results/B1/`),
`results/contact_torques/` (`results/B2/`), `results/motion_prior/` (`results/B3/`), `results/auxiliary_heads/`
(`results/C1/`), `results/data_pipeline/` (`results/C2/`), `results/robot_comparison/` (`results/D/`),
`results/contact_grounding/`, `results/end_to_end/`, `results/real_egocentric/`, `results/humanoid_asset/`,
`results/refinement_comparison/` (all previously under `results/E/`). Refinement-model checkpoints default to
`results/hr_refine/` (was `results/A2/hr_refine/`).

## Checklist sub-steps and other labels

Checklist item numbers (item 1 … item 20b) are kept; only the letter sub-steps and informal labels were renamed.

| Descriptive name | Old name / code |
|---|---|
| item 5 (AMASS loader) | stage 5a, item 5a |
| item 5 (BABEL labels) | stage 5b, item 5b |
| item 5, foot-contact labels | item 5c-1, 5c |
| item 5, synthetic IMU and QA; foot-height rise QA | item 5c-2, 5c-2 |
| item 6 (retarget bridge) | item 6A |
| tracking-mode trials (item 6) | 6B |
| item 6 smoke test, smoke-test default | item 6C, 6C |
| refinement smoke regression | B6 |
| earlier scratch run, earlier scratch jump run | Pass4, pass3 |
| robot module (deferred) | Module B |
| head-frame checks: camera convention, walking look, image plane, face-forward | review 1a, 1b, 1c, 1d |
| split correction | P1c |
| wording "pre-registered" / "recorded" | "frozen" / "freeze" (replaced everywhere; see `AGENTS.md` §2a rule 5) |

The robot smoke-test metric files (`results/robot_smoke_test/*/metrics_*.json`) later had one note string edited
(`6C` and `pass3` labels in the `time_to_fall_s` description and the `initial_root_velocity_convention` value); no
number changed.

## Renamed modules

| New module | Old module |
|---|---|
| `hready/experiments/` | `hready/eval/` |
| `hready/experiments/oracle_baselines.py` | `hready/eval/e3_oracle.py` |
| `hready/experiments/cohorts.py` | `hready/eval/e3_cohorts.py` |
| `hready/experiments/evidence_locked_ablation.py` | `hready/eval/e4_completion.py` |
| `hready/experiments/evidence_locked_preregistered.py` | `hready/eval/e4_v2.py` |
| `hready/data/completion_clips.py` | `hready/data/e3_clips.py` |
| `hready/data/completion_windows.py` | `hready/data/e3_dataset.py` |
| `hready/data/ego_heading.py` | `hready/data/e3_heading.py` |
| `hready/data/motion_cache.py` | `hready/data/e3_motion_memmap.py` |
| `hready/data/occlusion_cache.py` | `hready/data/e4_occlusion.py` |
| `hready/data/support_contact.py` | `hready/data/support_contact_v2.py` |
| `hready/baselines/flat_floor_heuristic.py` | `hready/baselines/e3_heuristic.py` |
| `hready/metrics/completion_metrics.py` | `hready/metrics/e3_eval.py` |
| `hready/models/evidence_locked_completion.py` (`EvidenceLockedCompletion`) | `hready/models/ego_complete_e4.py` (`EgoCompleteMotionE4`) |
| `hready/train/completion_engine.py` | `hready/train/e3_oracle_engine.py` |
| `hready/train/evidence_locked_engine.py` | `hready/train/e4_engine.py` |
| `EvidenceLockedContext`, `PreregisteredContext` | `E4Ctx`, `E4V2Ctx` |

## Renamed config, doc and result files

Content sha256 of each file before it was renamed (LF line endings). These are content hashes, independent of commit
IDs. Only names, paths and labels changed in the renamed files; every number, threshold, seed, margin, count and outcome
is unchanged (checked by replaying the literal replacements on the old text and comparing every numeric value).

| Old path | New path | Old content sha256 |
|---|---|---|
| `configs/e3_oracle.yaml` | `configs/oracle_completion_baselines.yaml` | `d704ac48e7cc8aeb8dcb33ddb72befdad9c7d3b0f1d7ba5a1f8f0122d323f517` |
| `configs/e4_completion.yaml` | `configs/evidence_locked_ablation.yaml` | `521fe6c2c98511232fa7d5222d9482131fdcb157f8a0c26e9ead913603857a73` |
| `configs/e4_v2_completion.yaml` | `configs/evidence_locked_preregistered_comparison.yaml` | `a86441b1568f010186a7d8919f1052b4f0fcffe1c503f05717197d64fb708376` |
| `configs/support_contact_v2.yaml` | `configs/support_contact_labels.yaml` | `1fb758bee2863947ed22181211ff346be47b6b4fc858a33f2beab7975a912089` |
| `docs/e0_audit.md` | `docs/module_audit.md` | `56e1825abed0938de9a6bd5fcce3c9d44661408c00cce0aedff8a1eefef18f37` |
| `docs/contact_labels_v2.md` | `docs/support_contact_labels.md` | `0bcac158b36300775e620cc6805fe7f8ebb8d27aa364b162492cbb8534118124` |
| `docs/e3_oracle.md` | `docs/oracle_completion_baselines.md` | `ae81e5a448e792ec32b35e0368a2b91589e9f379135133bd45c43e7cb1cdbc55` |
| `docs/e4_completion.md` | `docs/evidence_locked_ablation.md` | `ef775494d3668888e7551581c793e2c86404735b3010fac0899e750cb90dd095` |
| `docs/e4_v2_plan.md` | `docs/evidence_locked_preregistered_comparison.md` | `c2cec1afc37c68caeefb9e73737028ec466a28ed2d2f22a2bf27c9187296ae4a` |
| `results/E/e3_oracle_run.json` | `results/completion/oracle_baselines.json` | `903ee606a4ee9d8d47d123cdb5ddfd11b7b70281cdb05f14d3016983fca2a0fc` |
| `results/E/e4_completion_run.json` | `results/completion/evidence_locked_ablation.json` | `c3ca37e8f98139d1d4a51f8c8194603055ac1f69aa5c2c5cc00af3ca50e7ec3c` |
| `results/E/e4_v2_run.json` | `results/completion/evidence_locked_preregistered_comparison.json` | `1ab84e8f1d6b6bf02a697e7a6b57d5dcb08f0931274ad7e45ed5749cbe1cadc2` |
| `results/E/support_contact_v2_rates.json` | `results/support_contact/label_rates.json` | `1d29d2744a792123e2bdd8c93301fb01005314be1cbdd9d2edd820a165734c93` |
| `results/E/splits.json` | `results/splits/splits.json` | `27dcfe2fcba97ce7957b2fb517a8e21029aa902f9fd8edaf9de8b713348a488d` (unchanged) |
| `results/E/cohort_counts.json` | `results/splits/cohort_counts.json` | `1ad8bd94cbe1325fa29d37e7b50a3c58628b939f1cc263ceb34dda5861f7f5c5` (unchanged) |
| `results/E/floor_work_clips.csv` | `results/splits/floor_work_clips.csv` | `5aadf878e8cad45ae8400d590b6d8251eb6e408f617fb90a442f3f55dbb575cb` (unchanged) |
| `results/D/smoke/summary.json` | `results/robot_smoke_test/summary.json` | `d6ce98f57452de5a064417e9a6fe1f5cd0f4f60541d8ef25968a31333ae9f95d` |
| `results/D/smoke/ACCAD_run_to_jump/metrics_free.json` | `results/robot_smoke_test/ACCAD_run_to_jump/metrics_free.json` | `c71f82811772687e16722a6827eb7e27c5f0a6246a9acb78aa61ca2751739c2b` (unchanged) |
| `results/D/smoke/ACCAD_run_to_jump/metrics_kin_root.json` | `results/robot_smoke_test/ACCAD_run_to_jump/metrics_kin_root.json` | `ad606bafcf50f7a52dd7b91f9e1a2e17e80ff3d836120267f18f3ffbe9fd2c38` (unchanged) |
| `results/D/smoke/BMLrub_rub001_treadmill/metrics_kin_root.json` | `results/robot_smoke_test/BMLrub_rub001_treadmill/metrics_kin_root.json` | `201cc3255bfcdeba297c39aad84de8b0475a1799513c93b1ef569a01b0ddce3a` (unchanged) |
| `results/D/smoke/CMU_132_132_35/metrics_free.json` | `results/robot_smoke_test/CMU_132_132_35/metrics_free.json` | `ce0c99e2a8c79995168416b449e7eefdd252a4473ed05d490ede777c156f3fc2` (unchanged) |
| `results/D/smoke/CMU_132_132_35/metrics_kin_root.json` | `results/robot_smoke_test/CMU_132_132_35/metrics_kin_root.json` | `e96ca857c77429b5495363200c03a63e67e182aeab7e870474c95f65810860f6` (unchanged) |
| `results/D/smoke/standing_calibration/metrics_free.json` | `results/robot_smoke_test/standing_calibration/metrics_free.json` | `6acaf371b190a5c879e56e30834a8689b1653ab67884f39ba6de86f98c14bac7` (unchanged) |
| `results/D/smoke/standing_calibration/metrics_kin_root.json` | `results/robot_smoke_test/standing_calibration/metrics_kin_root.json` | `2026349038cfb1368e5d18605ee31ec03743f4524546b8c893cdd7959cbc0d47` (unchanged) |

## Renamed external folders (under the data root, not in the repository)

Moved in place (no copy, nothing rebuilt). The only content change is the cache-format tag in each motion-cache
`meta.json` and in its `index.json`; nothing reads the tag back when loading.

| New name | Old name | Contents |
|---|---|---|
| `checkpoints/oracle_completion_transformer/` | `checkpoints/e3_oracle/` | oracle completion transformer (`best.pt`, `last.pt`) |
| `checkpoints/evidence_locked_ablation/` | `checkpoints/e4/` | ablation-study arms (`deterministic`, `main`, `phys`, `ref_locked_gen_phys0`, `unlocked`) |
| `checkpoints/evidence_locked_preregistered_comparison/` | `checkpoints/e4_v2/` | pre-registered comparison arms (`det_w0`, `det_wphys0p3`, `det_wphys1p0`, `gen_stab`) |
| `<cache_dir>/grounded_motion_30hz/` | `<cache_dir>/e3_motion_30hz/` | 30 Hz floor-grounded motion cache (memory-mapped `.npy` per clip) |
| `<cache_dir>/self_occlusion_30hz/` | `<cache_dir>/e4_occlusion_30hz/` | per-frame self-occlusion masks (`occ.npy` per clip) |
| `<cache_dir>/babel_gait_medians.json` | `<cache_dir>/babel_frame_ann_gait_median_v1.json` | BABEL gait statistics cache |
| cache-format tag `grounded_motion_npy_mmap` | `e3_npy_mmap_v3` | `format` field of the motion cache |
| `wip/split_correction_ego_splits_2026-10-04.patch` | `wip/p1c_ego_splits_2026-10-04.patch` | split-correction patch kept outside the repository (named in `docs/pivot_log.md`) |

Absolute checkpoint paths recorded in `results/completion/*.json` were updated to the new folder names; the
checkpoint files themselves are unchanged.
