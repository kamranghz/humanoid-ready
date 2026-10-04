"""22-joint hready keypoint order (first 22 SMPL-X body joints in FK output).

HR-Refine, ego splits, and losses that slice ``joints[..., :22, :]`` share this layout.
Full SMPL-X FK returns 55 joints; ``hready.data.imu`` uses 55-joint ids
``(0, 15, 20, 21, 7, 8)`` — do not confuse those indices with this 22-joint table.
"""

from __future__ import annotations

JOINT_NAMES_22: tuple[str, ...] = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)

JOINT_INDEX: dict[str, int] = {n: i for i, n in enumerate(JOINT_NAMES_22)}

PELVIS = JOINT_INDEX["pelvis"]
NECK = JOINT_INDEX["neck"]
HEAD = JOINT_INDEX["head"]
LEFT_WRIST = JOINT_INDEX["left_wrist"]
RIGHT_WRIST = JOINT_INDEX["right_wrist"]
LEFT_HIP = JOINT_INDEX["left_hip"]
RIGHT_HIP = JOINT_INDEX["right_hip"]
LEFT_KNEE = JOINT_INDEX["left_knee"]
RIGHT_KNEE = JOINT_INDEX["right_knee"]
LEFT_SHOULDER = JOINT_INDEX["left_shoulder"]
RIGHT_SHOULDER = JOINT_INDEX["right_shoulder"]

UPPER_BODY_JOINTS: tuple[int, ...] = (3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)
LOWER_BODY_JOINTS: tuple[int, ...] = (1, 2, 4, 5, 7, 8, 10, 11)
FEET_JOINTS: tuple[int, ...] = (7, 8, 10, 11)
OBSERVED_BY_DESIGN_JOINTS: tuple[int, ...] = (15, 20, 21) + UPPER_BODY_JOINTS
