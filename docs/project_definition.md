# HumanoidReady — revised project definition and egocentric motion track interpretation (2026-10-04)

Status: APPROVED 2026-10-04 by the owner, with three clarifications (marked "rig inputs alongside evidence", "oracle evidence as a control" and "perception-matched training evidence" below). Docs-only. Implementation has not started.
Written against the tracker, AGENTS.md and the project history. The real AGENTS.md §8 text was not available to the author; the tracker synchronization must reconcile this file with it, keep every recorded module-audit and subject-split fact, and fix the existing §8 header contradiction.

---

## 1. Project definition (replaces the §0 goal paragraph and the egocentric motion track header)

HumanoidReady is ONE integrated system that goes from **egocentric RGB/video** to a **physically refined full-body human motion**, and evaluates every stage:

```
egocentric RGB/video  (+ head/camera pose and gravity, given in the first system version)
  -> visual body perception          trained visual model
  -> partial 3D body evidence        evidence schema (contract)
  -> full-body completion            oracle completion baselines, evidence-locked completion
  -> contact prediction + ground     foot channel + support-contact region channels
  -> physics-aware refinement        kinematic: contact-aware  ->  simulation: tracking (Newton)
  -> evaluation                      end-to-end evaluation, sliced by cohort
```

- Vision is mandatory on the main path. A weak perception result means the perception stage is adapted or trained; it is never removed or made optional.
- (Rig inputs alongside evidence.) The reconstruction path consumes the shared evidence schema plus the explicitly given rig signals (metric head/camera pose and gravity in the first system version). Perception output is not its only input.
- Contact and physics operate on the reconstructed human body (SMPL-X skeleton humanoid), not on a robot.
- Oracle mocap-derived evidence (observation simulator) is a diagnostic control, not the interface of the final system. (Oracle evidence as a control.) The oracle completion baselines run in oracle mode first only for diagnosis and development order. The oracle table is never described as the main system; the perceived RGB path is the main end-to-end system.
- G1 / robot executability and the CVPR paper are downstream or supporting. They are not on the egocentric motion track.
- The system covers two capabilities: body-pose perception (the perception stage) and full-body motion (completion, contact and physics refinement).

## 2. Binding decisions (owner, 2026-10-04)

- **Given head pose and gravity.** In the first system version, metric head/camera pose and gravity are given inputs (a rig with IMU/SLAM). The assumption is stated explicitly wherever results are reported. Estimating them from RGB is a later ablation, outside the first system. Whether the given pose carries declared noise is set in the observation-simulator config, not here.
- **Perception variants.** The tracker-level stage is architecture-agnostic: `egocentric RGB/video -> trained visual perception model -> visible/partial body evidence`. Compared variants:
  - **own perception model**: HR-HMR-style, BEDLAM-pretrained by us (random init) and then trained/adapted on egocentric renders;
  - **pretrained perception baseline(s)**: labeled pretrained, used as released;
  - **adapted pretrained backbone**: built only if needed for the strongest end-to-end system.
  The "from scratch" claim applies to our own HR-HMR/perception model and HR-Refine only, not to every model in the project. A weak HR-HMR result must not block RGB -> perception -> completion: the pipeline proceeds with the variant selected in perception model selection. HR-HMR is a candidate initialization, not the main path by decree; it is third-person and needs an ego-specific output head and ego training.
- **Egocentric data plan.** Start with synthetic egocentric RGB rendered from AMASS/SMPL-X with a virtual head camera (exact ground truth, controlled evaluation). Real egocentric validation (e.g. EgoBody) is an explicit planned step (real egocentric validation), not an indefinite "later". The no-download rule stays for the current stage; the dataset decision is made after the first synthetic end-to-end result.
- **Two-stage physics refinement.** One stage, two sub-stages, both on the reconstructed human: **kinematic** contact-aware refinement using the existing physics and biomechanics losses; **simulation**-based tracking/refinement of the SMPL-X humanoid in Newton.

## 3. Stage contracts

| Stage | Input | Output | Leak rule |
|---|---|---|---|
| Perception | ego frames + given head pose + gravity | evidence schema (see below) | no ground-truth body parameters as input; targets only from GT |
| Completion (oracle baselines, evidence-locked completion) | evidence schema (oracle or perceived) | full-body motion in world frame | consumes only schema tensors |
| Contact | evidence / completed motion | per-frame foot + region contact, ground plane | labels (subject splits, support-contact labels) used as targets only |
| Kinematic refinement | completed motion + predicted contact | refined motion | item-3 losses; HR-Refine is one possible learned instance (see §7) |
| Simulation refinement | motion to track | simulated humanoid motion (Newton) | Newton only (rule 0) |
| Evaluation | all of the above + GT | end-to-end evaluation tables | cohort slices from the subject splits |

Evidence schema (defined by the observation simulator, shared by oracle and perceived paths): head 6-DoF; visible-joint 3D estimates in the world frame; per-joint visibility mask; per-joint confidence; optional 2D keypoints from the virtual head camera. The same schema means completion cannot tell where the evidence came from.

## 4. Controls and decomposition

- **Oracle control (observation simulator):** AMASS GT -> observation simulator -> schema -> completion -> ... -> evaluation. Measures completion error with perception error removed.
- **Perception-only evaluation (egocentric perception stage):** ego frames -> perception -> visible-joint error vs GT.
- **Main path:** ego frames -> perception -> schema -> completion -> ...
- **Decomposition:** oracle row vs perceived row = the cost of perception; reported in the end-to-end evaluation. Oracle and perceived claims are kept separate in the claims map.

## 5. Egocentric motion track order (by dependency only; no dates)

`project scoping, project definition document, module audit (done), subject splits and cohorts (done), support-contact labels, tracker synchronization, observation simulator, oracle completion baselines (oracle mode: control, development order only), [items 12-13 perception pretraining], egocentric perception stage (render set, baseline measurement, training, model selection), completion baselines (perceived mode), evidence-locked completion, end-to-end evaluation, real egocentric validation, humanoid simulation asset, simulation tracking training, refinement comparison, qualitative review, claims map`

Egocentric perception sub-steps (each is one Cursor step):
- **Egocentric render set.** Virtual head camera on AMASS/SMPL-X; subject splits inherited (no leakage); per-frame GT for 2D/3D joints, per-joint visibility, camera pose, gravity. Camera spec and appearance/background spec recorded. Stated limit: train and test use the same renderer, so synthetic-only results are optimistic about real footage.
- **Perception baseline measurement.** Labeled pretrained detector(s) zero-shot: first ~50 frames, then VAL. Visible-joint error and visibility recall. This is a measurement, not a gate: the outcome sets how much perception training must do, not whether it exists.
- **Perception training.** Own model, pretrained baseline, and adapted backbone if needed. Items 12-13 (BEDLAM loader, HR-HMR from scratch, BEDLAM-CLIFF baseline) are prerequisites of the own model.
- **Perception model selection.** Select the main-path model on VAL with a criterion pre-registered before TEST; emit schema evidence for train/val/test; report all variants with their pretrained/from-scratch labels.

Real egocentric validation (new, planned): depends on the first synthetic end-to-end result in the end-to-end evaluation. Needs a dataset decision and a download exception, to be made then.

## 6. Reinterpretation / renaming map

| Item | Was | Becomes |
|---|---|---|
| Egocentric motion track header / §0 | oracle-style evidence -> perception -> ... | Section 1 chain, starting at egocentric RGB/video |
| Observation simulator | the observation interface | Oracle-evidence control + definition of the evidence schema and leak-proof path. Work content unchanged. |
| Egocentric perception stage | "feasibility gate", ~50 frames, go/no-go | **Egocentric perception stage** (render set, baseline measurement, training, model selection). The 50-frame test is the baseline measurement. On the critical path. |
| Completion baselines | baselines from observation-simulator observations | Same baselines in two evidence modes. Oracle table is still the first table, labeled a control. Perceived table follows perception model selection. |
| Evidence-locked completion | generative prior on partial evidence | Conditioned on the schema including visibility and confidence; must run on perceived evidence. |
| End-to-end evaluation | image rows only if the gate passed | Perceived rows are the main rows; oracle rows are the control; plus perception-only row and decomposition. |
| Humanoid simulation asset / simulation tracking training / refinement comparison | track the completed motion | Simulation refinement on the reconstructed motion. Refinement comparison "did physics help?" arms: none, kinematic, simulation, both. |
| Qualitative review | GT vs partial observation vs completion | Also shows ego RGB frames and perceived evidence. |
| Claims map | claims | Separate perception, completion and physics claims; oracle vs perceived kept separate; from-scratch wording scoped per the perception-variants decision. |
| Subject splits / support-contact labels | wording | Subject-split title and examples reflect kneel + lie evaluable, crawl/yoga empty, sit_floor excluded. Support-contact labels are GT-derived targets; at test time contact is predicted from perceived/completed evidence. |
| Items 12-13 / phase "Vision from scratch" | separate branch after the first table | "Perception pretraining and baselines": prerequisites of perception training, shown inside the egocentric motion track flow. |
| Items 7-8 (HR-Refine, loss ablation) | HR-Refine on corrupted AMASS | Candidate learned instance of kinematic refinement; item-3 ablation stays as support evidence for the losses. |
| Item 9 | contact/action/intent heads, IMU | Contact part overlaps evidence-locked completion contact; action/intent/IMU stay supporting (auxiliary heads and sensor fusion). |
| Item 10 | torques raw vs refined | Supporting evaluation metric on the reconstructed body. |
| Items 6, 11, 14, paper tracking-policy selection, Newton runs, cross-engine comparison, refinement effect | G1 work | Downstream/supporting; not on the egocentric motion track. |
| Item 20 | EgoBody/H36M/PROX "later" | EgoBody moves to real egocentric validation; Human3.6M and PROX stay later. |
| Tracker synchronization | structure sync | Must record this definition, the egocentric perception sub-steps, real egocentric validation, the rule amendments below, and the section 8 header fix. Its prompt is rewritten before use. |

Unchanged: project scoping, module audit, subject-split cohort definitions and recorded thresholds, support-contact label work content, the leak-proof principle, item 17, items 15-16, paper-last order.

## 7. Rule amendments needed (for the tracker synchronization to apply)

- **Rule 2** (from scratch): "From scratch = random init for HR-HMR and HR-Refine. Other models (pretrained baselines, and the adapted pretrained backbone) are allowed, always labeled 'pretrained' in every table row and figure."
- **Data/downloads note:** paused for the current stage; real egocentric data (real egocentric validation) is decided after the first synthetic end-to-end result.
- **Goal paragraph (section 0):** replaced by Section 1. The G1 executability check is described as downstream/supporting.
- Rule 0 (Newton only), rule 1 (no fabricated numbers), rule 5 (checks not saved) and the section 8 integrity rule are unchanged.

## 8. Open points (not decided here; each goes into the prompt of the item named)

1. **Completion training data** (oracle completion baselines, evidence-locked completion). (Perception-matched training evidence.) Recorded direction (owner): the final completion model is trained on evidence whose error distribution approximates the real perception stage, preferably simulated evidence with noise calibrated from held-out perception errors. Perfect oracle evidence is a control, not the final training distribution, and perception predictions on training clips must not introduce leakage. Constraints that follow (to be fixed in the perception-model-selection and completion prompts): the calibration errors must come from clips the perception model was not trained on (held-out subjects or cross-fitting) and never from TEST; and because a parametric noise model can miss structured errors (temporal correlation, visibility-dependent failures), perceived evidence on held-out data remains the main test row.
2. **Kinematic refinement implementation** (refinement comparison): learned HR-Refine (needs a new input path, because its current inputs include GT rotations and keypoints) or loss-based optimization. Stated in the refinement-comparison prompt.
3. **Perception model selection criterion**: exact metric, pre-registered before TEST.
4. **Virtual camera and appearance spec** (egocentric render set): placement, FOV, resolution, background.
5. **Scale statement**: all results are synthetic-egocentric, small-scale, floor-work cohort "indicative" (kneel + lie only, 8 subjects in VAL union TEST).

## 9. What this step does NOT do

No code and no AGENTS.md edit yet. The tracker text and the docs-only tracker-synchronization prompt follow this definition. Then support-contact labels, then the observation simulator.
