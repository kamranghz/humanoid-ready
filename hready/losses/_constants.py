"""Shared constants (plain Python / NumPy only, no PyTorch).

Center-of-mass segment masses for SMPL-X body joints 0..21.
"""

from __future__ import annotations

import numpy as np

COM_MASS_FRAC_SOURCE = (
    "de Leva (1996) segment mass fractions (% total body mass), male/female "
    "tables averaged; mapped to SMPL-X joints 0..21 (see COM_MASS_FRAC_DOC)."
)

COM_MASS_FRAC_DOC = """
Joint index → SMPL-X body joint (0..21) and de Leva mapping
------------------------------------------------------------
 0 pelvis      — trunk mass / 5
 1 left_hip    — thigh (14.47% TBW, side)
 2 right_hip   — thigh
 3 spine1      — trunk / 5
 4 left_knee   — shank (4.57%)
 5 right_knee  — shank
 6 spine2      — trunk / 5
 7 left_ankle  — 0 (foot mass on joint 10)
 8 right_ankle — 0
 9 spine3      — trunk / 5
10 left_foot   — foot (1.33%)
11 right_foot  — foot
12 neck        — trunk / 5
13 left_collar — 0
14 right_collar— 0
15 head        — head (6.81%)
16 left_shoulder — upper arm (2.63%)
17 right_shoulder
18 left_elbow  — forearm (1.50%)
19 right_elbow
20 left_wrist  — hand (0.585%)
21 right_wrist
Trunk 43.015% TBW split equally across pelvis, spine1–3, neck.
Each segment mass is placed at the proximal joint (kinematic approximation).
""".strip()

# Raw de Leva-averaged masses before renormalization (sum ≈ 1).
_COM_RAW_22: tuple[float, ...] = (
    0.43015 / 5,
    0.1447,
    0.1447,
    0.43015 / 5,
    0.0457,
    0.0457,
    0.43015 / 5,
    0.0,
    0.0,
    0.43015 / 5,
    0.0133,
    0.0133,
    0.43015 / 5,
    0.0,
    0.0,
    0.0681,
    0.0263,
    0.0263,
    0.0150,
    0.0150,
    0.00585,
    0.00585,
)
_s = sum(_COM_RAW_22)
COM_MASS_FRAC_22: tuple[float, ...] = tuple(v / _s for v in _COM_RAW_22)

_COM_FRAC_NP = np.array(COM_MASS_FRAC_22, dtype=np.float64)


def com_from_joints_numpy(joints: np.ndarray) -> np.ndarray:
    """CoM from first 22 joint positions; ``joints`` ``(..., J, 3)`` meters."""
    j = np.asarray(joints, dtype=np.float64)
    if j.ndim == 2:
        j = j.reshape(1, j.shape[0], j.shape[1])
    if j.shape[-2] < 22:
        raise ValueError(f"expected J >= 22, got {j.shape[-2]}")
    j22 = j[..., :22, :]
    frac = _COM_FRAC_NP.reshape(*(1,) * (j22.ndim - 2), 22, 1)
    return (frac * j22).sum(axis=-2)
