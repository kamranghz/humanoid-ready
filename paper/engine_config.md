# paper/engine_config.md — configuration-fairness record for the cross-engine comparison

Committed before the cross-engine runs begin (`paper/claims_map.md` §9). It lists every parameter that must be matched
across Newton, PhysX and MuJoCo, the value as it is currently set, and whether the match is confirmed.

**Status as of 2026-10-16: the engines are not yet configured to match.** Only the Newton column has recorded values
(from `hready/robot/isaaclab_newton.py` and the item 6 smoke-test metrics in `results/robot_smoke_test/`). PhysX has not
been run for the G1; MuJoCo has been used only to replay the retargeted motion to video (`hready/robot/replay.py`), not to
simulate dynamics. Nothing below is therefore "confirmed matched across engines"; the status column says what is
recorded and what is to be filled in. No value is filled in that has not been set in code or read from a result file.

**Controller.** Fallback in force until a learned policy is verified in all three engines (`docs/decisions.md`,
2026-10-16 entry): *PD-only joint tracking, not a balance controller.* Every cross-engine row carries that label.

**Status key.** RECORDED = value read from code or a result file for that engine. TBD = unknown or not yet set; to be
filled before the cross-engine runs. MATCH = same value confirmed in two or more engines (none yet). DIFFERS = a
known, documented difference.

## 1. Time stepping

| parameter | Newton (Isaac Lab, MuJoCo-Warp solver) | PhysX (Isaac Lab) | MuJoCo | status |
|---|---|---|---|---|
| physics timestep | 0.005 s (`SIM_DT`) | TBD | TBD | RECORDED (Newton); target: 0.005 s in all |
| substeps per control step | 1 (`num_substeps`) | TBD | TBD | RECORDED (Newton) |
| control / PD target update rate | 200 Hz (= 1/dt, one target per step) | TBD | TBD | RECORDED (Newton) |
| reference motion rate | retargeted clip fps (`motion["fps"]`), interpolation rule to be recorded | TBD | TBD | TBD |
| integrator | `implicitfast` (MuJoCo-Warp) | TBD (PhysX TGS or PGS: to record) | TBD (`implicitfast` available in MuJoCo; the Warp port lacks some options) | TBD |
| solver iterations | 100 | TBD (position / velocity iteration counts) | TBD | RECORDED (Newton) |

## 2. Robot asset

| parameter | Newton | PhysX | MuJoCo | status |
|---|---|---|---|---|
| G1 model source | Isaac Lab `G1_29DOF_CFG` (`isaaclab_assets`), 29 DoF | TBD; same Isaac Lab asset expected | TBD; replay currently uses GMR `assets/unitree_g1/g1_mocap_29dof.xml` | TBD |
| joint order and names | Isaac order mapped to GMR order (`GMR_TO_ISAAC_IDX`) | TBD | TBD (MJCF order) | TBD |
| link masses and total mass | read at run time from the Isaac asset (`default_mass`); value not recorded in the repository | TBD | TBD | TBD (must be identical, or the difference documented) |
| inertias and centres of mass | from the asset, not recorded | TBD | TBD | TBD |
| joint position limits | from the asset, not recorded | TBD | TBD | TBD |
| joint velocity limits | from the asset, not recorded | TBD | TBD | TBD |
| joint effort limits | read from actuator config (`effort_limit`), per-joint values saved in each run `.npz` (`effort_limit_gmr`) | TBD | TBD | RECORDED (Newton, in run files) |
| joint armature / reflected inertia | from the asset, not recorded | TBD | TBD | TBD |
| joint friction (viscous / Coulomb) | from the asset, not recorded | TBD | TBD | TBD |
| collision geometry | Isaac asset collision shapes; contact sensors activated | TBD | TBD (MJCF geoms may differ from the Isaac asset) | TBD |
| self-collision | not recorded | TBD | TBD | TBD |
| spawn pose and initial root velocity | default root pose of the asset; zero velocity (smoke-test convention) | TBD | TBD | RECORDED (Newton) |

Whether the Isaac Lab asset and the GMR MJCF describe the same robot (masses, limits, geometry) has not been checked.
That check is required before the MJCF is used for the MuJoCo dynamics run.

## 3. PD controller

| parameter | Newton | PhysX | MuJoCo | status |
|---|---|---|---|---|
| control law | joint PD to the reference angle with implicit actuator, `tau = Kp (q* - q) - Kd qdot`, clamped to the effort limit | TBD | TBD (position actuators or explicit torque loop: to choose) | TBD |
| Kp per joint group (legs 12, waist 3, arms 14) | Isaac `G1_29DOF_CFG` defaults; per-joint values saved in each run `.npz` (`stiffness_gmr`), not yet tabulated here | TBD | TBD | TBD (tabulate from the run files, then copy to the other engines) |
| Kd per joint group | as above (`damping_gmr`) | TBD | TBD | TBD |
| feedforward torque | none (PD-only) | TBD | TBD | RECORDED (Newton) |
| actuator model (ideal vs implicit vs delay) | implicit PD with effort clamp | TBD | TBD | TBD |
| action / target filtering | none | TBD | TBD | RECORDED (Newton) |
| tracking modes | `kin_root` (root pose and velocity written each step, joint PD to the reference) and `free` (joint PD only) | TBD | TBD | RECORDED (Newton); the mode used for verdicts is to be fixed |

## 4. Gravity and environment

| parameter | Newton | PhysX | MuJoCo | status |
|---|---|---|---|---|
| gravity vector | not set in our code; engine default, value to be read from the running simulation | TBD | TBD | TBD |
| up axis | Z up | TBD | TBD | TBD |
| ground | flat plane at z = 0 (Isaac `GroundPlaneCfg`) | TBD | TBD | RECORDED (Newton) |
| air / joint damping, wind | none set | TBD | TBD | TBD |

## 5. Contact model

| parameter | Newton | PhysX | MuJoCo | status |
|---|---|---|---|---|
| static / dynamic friction (ground) | 0.8 / 0.6 (`RigidBodyMaterialCfg`) | TBD | TBD | RECORDED (Newton) |
| friction combine rule (ground x foot) | TBD | TBD | TBD | TBD |
| friction cone | pyramidal, `impratio` 1.0 | TBD | TBD | RECORDED (Newton) |
| restitution | not set | TBD | TBD | TBD |
| contact stiffness / softness (`solref`, `solimp` or PhysX contact offset) | MuJoCo-Warp defaults, not recorded | TBD | TBD | TBD |
| contact detection | Newton collision pipeline (`use_mujoco_contacts = False`) | TBD | TBD | RECORDED (Newton) |
| contact buffer limits | `njmax` 250, `nconmax` 80 | TBD | TBD | RECORDED (Newton); overflow must be checked in every run |
| penetration / rest offset, contact margin | not recorded | TBD | TBD | TBD |

## 6. Run protocol that can change executability metrics

| parameter | value | status |
|---|---|---|
| fall detector | z < 0.4 m or tilt > 45 degrees (`sim_settings`) | RECORDED (Newton); must be the same function in all engines |
| foot-slip rule | ankle world position at sim dt, contact when `z <= min_run_z + 3 cm` (`docs/decisions.md`, item 6) | RECORDED; to be applied identically to all engines |
| torque-saturation rule | per group, PD demand against effort limit, threshold 0.95 (`hready/robot/metrics.py`) | RECORDED |
| settle or warm-up before the clip | not recorded | TBD |
| clip length, resampling and retargeting | GMR retargeting; the same retargeted file for all engines | TBD (record the retargeting commit and file hashes) |
| random seeds and nondeterminism | engine nondeterminism, repeated-run rule: `paper/claims_map.md` §4.3 | TBD (record seeds) |
| numeric precision | Newton (Warp) float32 expected; others not recorded | TBD |

## 7. Software versions

| engine | version | status |
|---|---|---|
| Isaac Lab | 6.1.17 (as logged by the item 6 runs) | RECORDED |
| `isaaclab_newton` | 0.13.6 | RECORDED |
| Newton | 1.2.1 | RECORDED |
| Warp | 1.13.0 | RECORDED |
| PhysX (Isaac Lab backend) | TBD | TBD |
| MuJoCo (and MJCF compiler version) | TBD | TBD |
| GMR (retargeter) commit | TBD | TBD |

## 8. Known and expected differences (to be documented, not hidden)

- The Newton backend here uses the MuJoCo-Warp solver, so Newton and MuJoCo share a solver family but not an
  implementation; the Warp port lacks some MuJoCo options (implicit-fast midpoint variants, plugin actuators).
- PhysX uses a different contact and friction formulation, so friction values may not map one to one; any value
  that cannot be matched is recorded here as DIFFERS with the reason.
- The MuJoCo model currently available in the repository comes from the retargeter's assets, not from the Isaac Lab
  asset; their equivalence is unchecked (section 2).

## 9. Fill-in procedure before the cross-engine runs

1. Dump, from one Newton run, every TBD value that is readable (gravity, masses, limits, armature, friction, per-joint
   Kp and Kd) and enter it in the Newton column.
2. Set the other two engines to those values and enter what was set. Where an engine cannot take the value, mark
   DIFFERS with the reason.
3. Run one AMASS clip in all three engines with the same file and gains (the verification that `docs/decisions.md`
   still lists as open) and record the outcome.
4. Change every TBD above to RECORDED, MATCH or DIFFERS; do not leave a TBD in a published row.

Any later change to this file after the cross-engine runs begin needs a dated entry here with the reason.
