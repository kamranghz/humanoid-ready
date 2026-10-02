"""SMPL-X body models with locked_head / v1_1 policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import torch
import yaml
from smplx.lbs import vertices2joints
from torch import Tensor

from hready.body.rotations import poses_to_axis_angle

RotRepr = Literal["axis_angle", "6d"]
Variant = Literal["locked_head", "v1_1"]

_NUM_JOINTS = 55
_NUM_VERTICES = 10475

# Training spec (AGENTS §1); num_betas for locked_head matches BEDLAM lockedhead_16b labels.
_LOCKED_HEAD_SPEC = {"gender": "neutral", "num_betas": 16}


class BodyModelPolicyError(Exception):
    """Raised when a body model variant is used outside its allowed policy."""


class BodyModelMismatchError(Exception):
    """Raised when mixing outputs or params from different body variants."""


@dataclass
class SmplxOutput:
    variant: Variant
    vertices: Tensor
    joints: Tensor
    faces: Tensor


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "configs" / "paths.example.yaml").is_file():
            return parent
    raise FileNotFoundError("Could not locate repo root (configs/paths.example.yaml).")


def _load_data_root() -> Path:
    root = _repo_root()
    paths_yaml = root / "configs" / "paths.yaml"
    if not paths_yaml.is_file():
        raise FileNotFoundError(
            f"Missing {paths_yaml}; copy configs/paths.example.yaml to configs/paths.yaml."
        )
    with paths_yaml.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return Path(cfg["data_root"])


def _model_dir(data_root: Path, variant: Variant) -> Path:
    if variant == "locked_head":
        return data_root / "models" / "smplx" / "locked_head" / "models_lockedhead"
    if variant == "v1_1":
        return data_root / "models" / "smplx" / "v1_1" / "models"
    raise ValueError(variant)


def _neutral_npz_path(model_dir: Path) -> Path:
    return model_dir / "smplx" / "SMPLX_NEUTRAL.npz"


def _read_gender_from_files(model_dir: Path) -> str:
    npz = _neutral_npz_path(model_dir)
    if not npz.is_file():
        raise FileNotFoundError(f"Expected neutral SMPL-X npz at {npz}")
    return "neutral"


def _max_betas_in_npz(model_dir: Path) -> int:
    npz = np.load(_neutral_npz_path(model_dir), allow_pickle=True)
    shapedirs = npz["shapedirs"]
    return int(shapedirs.shape[-1])


def _resolve_num_betas(variant: Variant, model_dir: Path) -> int:
    max_betas = _max_betas_in_npz(model_dir)
    if variant == "locked_head":
        num = _LOCKED_HEAD_SPEC["num_betas"]
        if num > max_betas:
            raise ValueError(f"locked_head requests num_betas={num} but npz has {max_betas}")
        return num
    if variant == "v1_1":
        import smplx

        probe = smplx.create(
            str(model_dir),
            model_type="smplx",
            gender="neutral",
            use_pca=False,
            flat_hand_mean=True,
            ext="npz",
        )
        num = int(probe.num_betas)
        del probe
        if num > max_betas:
            raise ValueError(f"v1_1 num_betas={num} exceeds npz capacity {max_betas}")
        return num
    raise ValueError(variant)


def _validate_locked_head_spec(model_dir: Path, num_betas: int) -> None:
    gender = _read_gender_from_files(model_dir)
    if gender != _LOCKED_HEAD_SPEC["gender"]:
        raise ValueError(
            f"locked_head gender from files is {gender!r}, spec requires "
            f"{_LOCKED_HEAD_SPEC['gender']!r}"
        )
    if num_betas != _LOCKED_HEAD_SPEC["num_betas"]:
        raise ValueError(
            f"locked_head num_betas={num_betas} differs from training spec "
            f"{_LOCKED_HEAD_SPEC['num_betas']}"
        )


class SmplxBody:
    """SMPL-X forward wrapper with variant tagging."""

    def __init__(self, variant: Variant, model, num_betas: int, faces: Tensor) -> None:
        self.variant = variant
        self._model = model
        self.num_betas = num_betas
        self.faces = faces

    @property
    def dtype(self) -> torch.dtype:
        return self._model.dtype

    def forward(
        self,
        global_orient: Tensor,
        body_pose: Tensor,
        betas: Tensor,
        transl: Tensor,
        *,
        rot_repr: RotRepr = "axis_angle",
        left_hand_pose: Optional[Tensor] = None,
        right_hand_pose: Optional[Tensor] = None,
    ) -> SmplxOutput:
        batch = global_orient.shape[0]
        device = global_orient.device
        dtype = global_orient.dtype

        go = poses_to_axis_angle(
            _as_pose_block(global_orient, rot_repr, 1), rot_repr
        ).reshape(batch, 3)
        bp = poses_to_axis_angle(
            _as_pose_block(body_pose, rot_repr, 21), rot_repr
        ).reshape(batch, 63)

        if left_hand_pose is None:
            lhp = torch.zeros(batch, 45, device=device, dtype=dtype)
        else:
            lhp = poses_to_axis_angle(
                _as_pose_block(left_hand_pose, rot_repr, 15), rot_repr
            ).reshape(batch, 45)
        if right_hand_pose is None:
            rhp = torch.zeros(batch, 45, device=device, dtype=dtype)
        else:
            rhp = poses_to_axis_angle(
                _as_pose_block(right_hand_pose, rot_repr, 15), rot_repr
            ).reshape(batch, 45)

        if betas.shape[-1] != self.num_betas:
            raise ValueError(f"betas dim {betas.shape[-1]} != num_betas {self.num_betas}")

        zeros3 = torch.zeros(batch, 3, device=device, dtype=dtype)
        zeros_expr = torch.zeros(batch, 10, device=device, dtype=dtype)
        out = self._model(
            betas=betas,
            global_orient=go,
            body_pose=bp,
            left_hand_pose=lhp,
            right_hand_pose=rhp,
            transl=transl,
            jaw_pose=zeros3,
            leye_pose=zeros3,
            reye_pose=zeros3,
            expression=zeros_expr,
            return_verts=True,
            pose2rot=True,
        )
        vertices = out.vertices
        if vertices.shape[1] != _NUM_VERTICES:
            raise RuntimeError(f"Expected {_NUM_VERTICES} vertices, got {vertices.shape[1]}")
        joints = vertices2joints(self._model.J_regressor, vertices)
        if joints.shape[1] != _NUM_JOINTS:
            raise RuntimeError(f"Expected {_NUM_JOINTS} joints, got {joints.shape[1]}")
        return SmplxOutput(
            variant=self.variant,
            vertices=vertices,
            joints=joints,
            faces=self.faces,
        )


def _as_pose_block(pose: Tensor, rot_repr: RotRepr, num_joints: int) -> Tensor:
    if rot_repr == "axis_angle":
        expected = num_joints * 3
        if pose.shape[-1] != expected:
            raise ValueError(f"axis_angle pose expects last dim {expected}, got {pose.shape[-1]}")
        return pose.reshape(*pose.shape[:-1], num_joints, 3)
    if rot_repr == "6d":
        expected = num_joints * 6
        if pose.shape[-1] != expected:
            raise ValueError(f"6d pose expects last dim {expected}, got {pose.shape[-1]}")
        return pose.reshape(*pose.shape[:-1], num_joints, 6)
    raise ValueError(rot_repr)


def load_body(
    variant: Union[str, Variant],
    *,
    purpose: Optional[str] = None,
) -> SmplxBody:
    import smplx

    if variant == "v1_1" and purpose != "baseline":
        raise BodyModelPolicyError(
            'load_body("v1_1") requires purpose="baseline" (BEDLAM-CLIFF baseline only).'
        )
    if variant not in ("locked_head", "v1_1"):
        raise ValueError(f"Unknown variant {variant!r}")

    data_root = _load_data_root()
    model_dir = _model_dir(data_root, variant)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"SMPL-X model directory not found: {model_dir}")

    gender = _read_gender_from_files(model_dir)
    num_betas = _resolve_num_betas(variant, model_dir)
    if variant == "locked_head":
        _validate_locked_head_spec(model_dir, num_betas)

    model = smplx.create(
        str(model_dir),
        model_type="smplx",
        gender=gender,
        num_betas=num_betas,
        use_pca=False,
        flat_hand_mean=True,
        ext="npz",
        create_global_orient=False,
        create_body_pose=False,
        create_left_hand_pose=False,
        create_right_hand_pose=False,
        create_betas=False,
        create_transl=False,
        create_jaw_pose=False,
        create_leye_pose=False,
        create_reye_pose=False,
        create_expression=False,
    )
    faces = torch.as_tensor(np.asarray(model.faces), dtype=torch.long)
    return SmplxBody(variant=variant, model=model, num_betas=num_betas, faces=faces)


def _body_variant(obj: Union[SmplxOutput, SmplxBody]) -> Variant:
    return obj.variant


def assert_same_body(
    a: Union[SmplxOutput, SmplxBody],
    b: Union[SmplxOutput, SmplxBody],
) -> None:
    if _body_variant(a) != _body_variant(b):
        raise BodyModelMismatchError(
            f"Body variant mismatch: {_body_variant(a)!r} vs {_body_variant(b)!r}"
        )
