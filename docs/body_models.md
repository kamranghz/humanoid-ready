# Body models

This document compares the parametric body models relevant to this project
and records which are used, which are supported by SOMA-X, and which are
documented here for reference only.

## Summary

| Model | Vertices | Joints | Shape params | Hands | Face | Used in project | SOMA-X pivot | License |
|---|---|---|---|---|---|---|---|---|
| SMPL-X (locked\_head) | 10,475 | 55 | 16 (locked\_head spec) | ✓ full 15+15 | ✓ jaw, eyes | **primary** (training, eval) | source / target | [MPI-IS Research](https://smpl-x.is.tue.mpg.de/license.html) |
| SMPL-X (v1\_1) | 10,475 | 55 | 10 | ✓ full 15+15 | ✓ jaw, eyes | BEDLAM-CLIFF baseline only | — | [MPI-IS Research](https://smpl-x.is.tue.mpg.de/license.html) |
| MHR | 10,475 | 77 | — (data-driven) | ✓ detailed | ✓ jaw, eyes | SOMA-X pivot target | target (via SOMALayer) | NVIDIA Research |
| GHUM | 10,168 | 60 | 10 body + 6 expression | ✓ compact 6+6 PCA | ✓ | **not used** (documented only) | not supported | [Google Research](https://github.com/google-research/google-research/tree/master/ghum) |
| SOMA | 10,475 | 77 | — (learned) | ✓ detailed | ✓ jaw, eyes | SOMA-X internal representation | conversion hub | NVIDIA Research |

---

## SMPL-X

SMPL-X (Expressive Body Capture, Pavlakos et al. 2019) is the primary body
model throughout this project.

**Variants used:**

* **`locked_head`** (`smplx/locked_head/models_lockedhead/`): neutral gender,
  16 betas.  Matches the BEDLAM lockedhead\_16b label format.  All training
  data, evaluation, and physics losses use this variant.  Mixing it with
  v1\_1 raises `BodyModelMismatchError` (see `hready/body/smplx_wrapper.py`).
* **`v1_1`** (`smplx/v1_1/models/`): neutral gender, 10 betas.  Used only for
  the BEDLAM-CLIFF baseline (`purpose="baseline"` guard in `load_body`).

**Joint layout (55 joints):**

```
 0  Pelvis          22  Left index 1      37  Right index 1
 1  L Hip           23  Left index 2      38  Right index 2
 2  R Hip           24  Left index 3      39  Right index 3
 3  Spine 1         25  Left middle 1     40  Right middle 1
 4  L Knee          26  Left middle 2     41  Right middle 2
 5  R Knee          27  Left middle 3     42  Right middle 3
 6  Spine 2         28  Left pinky 1      43  Right pinky 1
 7  L Ankle         29  Left pinky 2      44  Right pinky 2
 8  R Ankle         30  Left pinky 3      45  Right pinky 3
 9  Spine 3         31  Left ring 1       46  Right ring 1
10  L Foot          32  Left ring 2       47  Right ring 2
11  R Foot          33  Left ring 3       48  Right ring 3
12  Neck            34  Left thumb 1      49  Right thumb 1
13  L Collar        35  Left thumb 2      50  Right thumb 2
14  R Collar        36  Left thumb 3      51  Right thumb 3
15  Head            (eye / jaw below)
16  L Shoulder      52  Jaw
17  R Shoulder      53  Left Eye
18  L Elbow         54  Right Eye
19  R Elbow
20  L Wrist
21  R Wrist
```

**Mesh:** 10,475 vertices, 20,908 triangles.  Dense vertex mesh used
throughout for contact detection (foot sole clusters) and physics losses.

**Model files:** `D:\projects\hready_data\models\smplx\`  (git-ignored; not
distributed with code).

---

## MHR (Mirrored Human Representation)

MHR is NVIDIA's internal body model that underpins SOMA.  It shares the
SMPL-X mesh topology (10,475 vertices) but uses a different joint hierarchy
(77 joints), co-located ankle distribution, and a data-driven shape space.

MHR model weights are bundled inside `py-soma-x` and do not require separate
licensed files.

**Usage in this project:**  SOMA-X pivot only (item 17).  SMPL-X poses are
converted to MHR format via `soma.transfer_smpl_family_pose_parameters`, and
the reverse direction (MHR → SMPL-X) is provided for round-trip evaluation.
MHR poses are **not** used as training targets.

**Joint count:** 77 (body + detailed hand + facial joints).

**Conversion entry point:** `hready/body/soma_x_wrapper.py` →
`SomaXBridge`.  Requires the separate `hready-soma` conda env (see
`env/soma_env.txt`).

---

## SOMA

SOMA (NVIDIA Research) is the full generative body model that MHR identity
parameters describe.  It runs in `py-soma-x` via `soma.SOMALayer` and is
used here as the **conversion hub**: SMPL-X vertices are bridged to SOMA
topology via `SMPLFamilyTopologyBridge`, then MHR pose parameters are
recovered via `PoseInversion`.

SOMA itself is **not** a training target in this project; it appears only
inside the SOMA-X worker script.

---

## GHUM

GHUM (Humans with Geometric Universals of Morphology, Xu et al. 2020,
Google Research) is documented here for completeness; it is **not used** in
this project.

**Why not:**  py-soma-x (v0.3.6) does not support GHUM as a source or target
for `transfer_smpl_family_pose_parameters`.  GHUM also uses a different
vertex topology (10,168 vertices, as opposed to the SMPL-X/MHR 10,475), so
the `SMPLFamilyTopologyBridge` does not apply without custom correspondence
mapping.

**Key properties for reference:**

* Vertices: 10,168 (body); separate hand models available.
* Shape space: 10 PCA betas (body) + 6 expression coefficients.
* Joint layout: 60 joints (24 body + 15+15 hand + jaw + eyes).
* Licence: Google Research non-commercial (requires registration).
* Code: `ghum-open` on GitHub (JAX/TensorFlow; no official PyTorch port as of
  August 2025).

---

## SOMA-X conversion environment

SMPL-X ↔ MHR conversion runs in a separate conda env, `hready-soma`, because
`chumpy` (a py-soma-x dependency) conflicts with NumPy ≥ 2.0 in the main
`hready` env.

Setup instructions are in `env/soma_env.txt`.  After creating the env, add
the Python executable path to `configs/paths.yaml`:

```yaml
soma_python: C:/Users/.../anaconda3/envs/hready-soma/python.exe
```

The bridge class `SomaXBridge` in `hready/body/soma_x_wrapper.py` starts the
worker subprocess automatically.

---

## Round-trip error (item 17)

The number is data, not a threshold: it documents the information loss
introduced by the SOMA-X pivot for downstream readers.  It is produced by
`scripts/rmse_check.py`, which drives the same worker (`scripts/soma_x_worker.py`)
that `SomaXBridge` uses.

### How the conversion is computed

`py-soma-x` 0.3.6 ships `transfer_smpl_family_pose_parameters`, but it is not
usable here: it needs `.obj` base meshes that the locked-head SMPL-X
distribution does not include, and for target=MHR it calls
`SOMALayer.pose(global_translation=...)` although that layer's keyword is
`transl`.  The worker therefore does the transfer itself:

1. Pose the source model (zero shape / zero identity, no pose correctives).
2. Give every target-topology vertex the position of its nearest source vertex,
   matched on the two layers' **rest-pose** meshes (both in metres, pelvis
   frame, T-pose).  This nearest-neighbour bridge is an approximation of
   SOMA's own topology bridge.
3. Fit the target model's joint rotations to that surface with
   `soma.smpl.transfer.PoseInversion`.

All poses crossing the worker are **absolute** joint rotations (axis-angle),
as `PoseInversion` returns them; MHR uses 77 public SOMA joints, SMPL-X 55.
Callers holding parent-relative SMPL-X parameters (AMASS style) must convert
before calling the bridge.

### Metrics

For each vertex, d is the Euclidean distance in metres; RMSE is
sqrt(mean(d^2)) over all vertices and frames.  The two directional rows are
the `PoseInversion` fit error in the **target** topology against the bridged
surface, so they do not include the bridge error itself; that is listed
separately.  The round-trip row compares the original SMPL-X vertices with
the SMPL-X vertices recovered after SMPL-X -> MHR -> SMPL-X (same topology).

| Step | RMSE (m) | Mean vertex error (m) | Bridge NN distance, rest pose (median / p95, m) |
|---|---|---|---|
| SMPL-X -> MHR | 0.0246 | 0.0217 | 0.0344 / 0.0936 |
| MHR -> SMPL-X | 0.0261 | 0.0233 | 0.0267 / 0.0672 |
| Round-trip (SMPL-X -> MHR -> SMPL-X) | 0.0206 | 0.0156 | n/a |

### Conditions and limits

* One run (2026-11-19) on **4 synthetic frames**: per-joint absolute
  rotations drawn from N(0, 0.1 rad), root translation from N(0, 0.01 m),
  NumPy seed 42, CPU, `py-soma-x` 0.3.6.  This is not real motion and not a
  distributional estimate; no confidence interval is claimed.  A run on real
  AMASS clips is needed before quoting a general figure.
* The bridge's own nearest-neighbour distance (median 2.7 to 3.4 cm) is of the
  same size as the fit errors.  It reflects differences between the two
  models' default bodies and their vertex layouts; this experiment does not
  separate those two causes.
* Shape is fixed at zero for both models; shape transfer is not evaluated.

Reproduce: `conda activate hready` then
`python scripts/rmse_check.py` (it starts the worker in `hready-soma`).
