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
# SMPL-X ``body_pose`` indices for leg articulation (leak tests).
LEG_BODY_AA_INDICES: tuple[int, ...] = (0, 1, 3, 4, 6, 7, 9, 10)
FEET_JOINTS: tuple[int, ...] = (7, 8, 10, 11)
# Foot-contact channels (L heel, L toe, R heel, R toe) for E3 heuristic + metrics.
FOOT_CONTACT_CHANNEL_JOINTS: tuple[int, ...] = (7, 10, 8, 11)
OBSERVED_BY_DESIGN_JOINTS: tuple[int, ...] = (15, 20, 21) + UPPER_BODY_JOINTS

# SMPL-X FK ``joints`` layout (55 joints); eyes are not in the 22-joint keypoint table.
SMPLX_LEFT_EYE = 23
SMPLX_RIGHT_EYE = 24

# Visibility / reporting groups (22-joint indices).
JOINT_GROUP_HANDS: tuple[int, ...] = (20, 21)
JOINT_GROUP_FOREARMS: tuple[int, ...] = (18, 19)
JOINT_GROUP_TORSO: tuple[int, ...] = (0, 3, 6, 9, 12, 13, 14)
JOINT_GROUP_THIGHS: tuple[int, ...] = (1, 2, 4, 5)
JOINT_GROUP_SHINS: tuple[int, ...] = (7, 8)
JOINT_GROUP_FEET: tuple[int, ...] = (10, 11)

VISIBILITY_JOINT_GROUPS: dict[str, tuple[int, ...]] = {
    "hands": JOINT_GROUP_HANDS,
    "forearms": JOINT_GROUP_FOREARMS,
    "torso": JOINT_GROUP_TORSO,
    "thighs": JOINT_GROUP_THIGHS,
    "shins": JOINT_GROUP_SHINS,
    "feet": JOINT_GROUP_FEET,
}
