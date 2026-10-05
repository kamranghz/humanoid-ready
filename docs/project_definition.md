# HumanoidReady — project definition v2 and Track E interpretation (2026-10-04)

Status: APPROVED 2026-10-04 by the owner, with three clarifications (marked [C1], [C2], [C3] below). Docs-only. Implementation has not started.
Written against the tracker (v19), AGENTS.md v8.1 and the project history. The real AGENTS.md §8 text was not available to the author; E-sync must reconcile this file with it, keep every frozen E0/E1 fact, and fix the existing §8 header contradiction.

---

## 1. Project definition (replaces the §0 goal paragraph and the Track E header)

HumanoidReady is ONE integrated system that goes from **egocentric RGB/video** to a **physically refined full-body human motion**, and evaluates every stage:

```
egocentric RGB/video  (+ head/camera pose and gravity, given in v1)
  -> visual body perception          trained visual model
  -> partial 3D body evidence        evidence schema (contract)
  -> full-body completion            E3 baselines, E4 generative prior
  -> contact prediction + ground     foot channel + E1b region channels
  -> physics-aware refinement        K: kinematic/contact-aware  ->  S: simulation tracking (Newton)
  -> evaluation                      E5, sliced by cohort
```

- Vision is mandatory on the main path. A weak perception result means the perception stage is adapted or trained; it is never removed or made optional.
- [C1] The reconstruction path consumes the shared evidence schema plus the explicitly given rig signals (metric head/camera pose and gravity in v1). Perception output is not its only input.
- Contact and physics operate on the reconstructed human body (SMPL-X skeleton humanoid), not on a robot.
- Oracle mocap-derived evidence (E2-A) is a diagnostic control, not the interface of the final system. [C2] E3 runs in oracle mode first only for diagnosis and development order. The oracle table is never described as the main system; the perceived RGB path is the main end-to-end system.
- G1 / robot executability and the CVPR paper are downstream or supporting. They are not on the Track E path.
- The system covers two capabilities: body-pose perception (the perception stage) and full-body motion (completion, contact and physics refinement).

## 2. Binding decisions (owner, 2026-10-04)

- **D1 Head pose and gravity.** In v1, metric head/camera pose and gravity are given inputs (a rig with IMU/SLAM). The assumption is stated explicitly wherever results are reported. Estimating them from RGB is a later ablation, outside the first system. Whether the given pose carries declared noise is set in the E2-A config, not here.
- **D2 Perception backbone.** The tracker-level stage is architecture-agnostic: `egocentric RGB/video -> trained visual perception model -> visible/partial body evidence`. Compared variants:
  - **P-A** own HR-HMR-style model: BEDLAM-pretrained by us (random init) and then trained/adapted on egocentric renders;
  - **P-B** labeled pretrained perception baseline(s), used as released;
  - **P-C** adapted pretrained backbone, built only if needed for the strongest end-to-end system.
  The "from scratch" claim applies to our own HR-HMR/perception model and HR-Refine only, not to every model in the project. A weak HR-HMR result must not block RGB -> perception -> completion: the pipeline proceeds with the variant selected in E2-B4. HR-HMR is a candidate initialization, not the main path by decree; it is third-person and needs an ego-specific output head and ego training.
- **D3 Egocentric data.** Start with synthetic egocentric RGB rendered from AMASS/SMPL-X with a virtual head camera (exact ground truth, controlled evaluation). Real egocentric validation (e.g. EgoBody) is an explicit planned step (E5b), not an indefinite "later". The no-download rule stays for the current stage; the dataset decision is made after the first synthetic end-to-end result.
- **D4 Physics-aware refinement.** One stage, two sub-stages, both on the reconstructed human: **K** kinematic/contact-aware refinement using the existing physics and biomechanics losses; **S** simulation-based tracking/refinement of the SMPL-X humanoid in Newton.

## 3. Stage contracts

| Stage | Input | Output | Leak rule |
|---|---|---|---|
| Perception | ego frames + given head pose + gravity | evidence schema (see below) | no ground-truth body parameters as input; targets only from GT |
| Completion (E3/E4) | evidence schema (oracle or perceived) | full-body motion in world frame | consumes only schema tensors |
| Contact | evidence / completed motion | per-frame foot + region contact, ground plane | labels (E1/E1b) used as targets only |
| Refinement K | completed motion + predicted contact | refined motion | item-3 losses; HR-Refine is one possible learned instance (see §7) |
| Refinement S | motion to track | simulated humanoid motion (Newton) | Newton only (rule 0) |
| Evaluation | all of the above + GT | E5 tables | cohort slices from E1 splits |

Evidence schema (defined in E2-A, shared by oracle and perceived paths): head 6-DoF; visible-joint 3D estimates in the world frame; per-joint visibility mask; per-joint confidence; optional 2D keypoints from the virtual head camera. The same schema means completion cannot tell where the evidence came from.

## 4. Controls and decomposition

- **Oracle control (E2-A):** AMASS GT -> observation simulator -> schema -> completion -> ... -> evaluation. Measures completion error with perception error removed.
- **Perception-only evaluation (E2-B):** ego frames -> perception -> visible-joint error vs GT.
- **Main path:** ego frames -> perception -> schema -> completion -> ...
- **Decomposition:** oracle row vs perceived row = the cost of perception; reported in E5. Oracle and perceived claims are kept separate in E9.

## 5. Track E order (by dependency only; no dates)

`P0, E-doc, E0 (done), E1 (done), E1b, E-sync, E2-A, E3 (oracle mode: control, development order only), [items 12-13 perception pretraining], E2-B (B1-B4), E3 (perceived mode), E4, E5, E5b, E6a, E6, E7, E8, E9`

E2-B sub-steps (each is one Cursor step):
- **B1 Ego render set.** Virtual head camera on AMASS/SMPL-X; E1 splits inherited (no leakage); per-frame GT for 2D/3D joints, per-joint visibility, camera pose, gravity. Camera spec and appearance/background spec recorded. Stated limit: train and test use the same renderer, so synthetic-only results are optimistic about real footage.
- **B2 Baseline measurement.** Labeled pretrained detector(s) zero-shot: first ~50 frames, then VAL. Visible-joint error and visibility recall. This is a measurement, not a gate: the outcome sets how much B3 must do, not whether B3 exists.
- **B3 Train/adapt.** P-A, P-B, and P-C if needed. Items 12-13 (BEDLAM loader, HR-HMR from scratch, BEDLAM-CLIFF baseline) are prerequisites of P-A.
- **B4 Select and emit.** Select the main-path model on VAL with a criterion frozen before TEST; emit schema evidence for train/val/test; report all variants with their pretrained/from-scratch labels.

E5b (new, planned): real egocentric validation. Depends on the first synthetic end-to-end result in E5. Needs a dataset decision and a download exception, to be made then.

## 6. Reinterpretation / renaming map

| Item | Was | Becomes |
|---|---|---|
| Track E header / §0 | oracle-style evidence -> perception -> ... | Section 1 chain, starting at egocentric RGB/video |
| E2-A | the observation interface | Oracle-evidence control + definition of the evidence schema and leak-proof path. Work content unchanged. |
| E2-B | "feasibility gate", ~50 frames, go/no-go | **Egocentric perception stage** (B1-B4). The 50-frame test is B2. On the critical path. |
| E3 | baselines from E2-A observations | Same baselines in two evidence modes. Oracle table is still the first table, labeled a control. Perceived table follows B4. |
| E4 | generative prior on partial evidence | Conditioned on the schema including visibility and confidence; must run on perceived evidence. |
| E5 | image rows only if the gate passed | Perceived rows are the main rows; oracle rows are the control; plus perception-only row and decomposition. |
| E6a / E6 / E7 | track the completed motion | Stage S on the reconstructed motion. E7 "did physics help?" arms: none, K, S, K+S. |
| E8 | GT vs partial observation vs completion | Also shows ego RGB frames and perceived evidence. |
| E9 | claims | Separate perception, completion and physics claims; oracle vs perceived kept separate; from-scratch wording scoped per D2. |
| E1 / E1b | wording | E1 title and examples reflect kneel + lie evaluable, crawl/yoga empty, sit_floor excluded. E1b labels are GT-derived targets; at test time contact is predicted from perceived/completed evidence. |
| Items 12-13 / phase "Vision from scratch" | separate branch after the first table | "Perception pretraining and baselines": prerequisites of E2-B3, shown inside the Track E flow. |
| Items 7-8 (HR-Refine, loss ablation) | HR-Refine on corrupted AMASS | Candidate learned instance of K; item-3 ablation stays as support evidence for the losses. |
| Item 9 | contact/action/intent heads, IMU | Contact part overlaps E4 contact; action/intent/IMU stay supporting (work package C1). |
| Item 10 | torques raw vs refined | Supporting evaluation metric on the reconstructed body. |
| Items 6, 11, 14, R2, R6-R8 | G1 work | Downstream/supporting; not on the Track E path. |
| Item 20 | EgoBody/H36M/PROX "later" | EgoBody moves to E5b; Human3.6M and PROX stay later. |
| E-sync | structure sync | Must record this definition, the E2-B sub-steps, E5b, the rule amendments below, and the section 8 header fix. Its prompt is rewritten before use. |

Unchanged: P0, E0, E1 cohort definitions and frozen thresholds, E1b work content, the leak-proof principle, item 17, items 15-16, paper-last order.

## 7. Rule amendments needed (for E-sync to apply)

- **Rule 2** (from scratch): "From scratch = random init for HR-HMR and HR-Refine. Other models (pretrained baselines, and the adapted variant P-C) are allowed, always labeled 'pretrained' in every table row and figure."
- **Data/downloads note:** frozen for the current stage; real egocentric data (E5b) is decided after the first synthetic end-to-end result.
- **Goal paragraph (section 0):** replaced by Section 1. The G1 executability check is described as downstream/supporting.
- Rule 0 (Newton only), rule 1 (no fabricated numbers), rule 5 (checks not saved) and the section 8 integrity rule are unchanged.

## 8. Open points (not decided here; each goes into the prompt of the item named)

1. **Completion training data** (E3/E4). [C3] Recorded direction (owner): the final completion model is trained on evidence whose error distribution approximates the real perception stage, preferably simulated evidence with noise calibrated from held-out perception errors. Perfect oracle evidence is a control, not the final training distribution, and perception predictions on training clips must not introduce leakage. Constraints that follow (to be fixed in the E2-B4/E3 prompts): the calibration errors must come from clips the perception model was not trained on (held-out subjects or cross-fitting) and never from TEST; and because a parametric noise model can miss structured errors (temporal correlation, visibility-dependent failures), perceived evidence on held-out data remains the main test row.
2. **K implementation** (E7): learned HR-Refine (needs a new input path, because its current inputs include GT rotations and keypoints) or loss-based optimization. Stated in the E7 prompt.
3. **B4 selection criterion**: exact metric, frozen before TEST.
4. **Virtual camera and appearance spec** (B1): placement, FOV, resolution, background.
5. **Scale statement**: all results are synthetic-egocentric, small-scale, floor-work cohort "indicative" (kneel + lie only, 8 subjects in VAL union TEST).

## 9. What this step does NOT do

No code and no AGENTS.md edit yet. The tracker text and the docs-only E-sync prompt follow this definition. Then E1b, then E2-A.
