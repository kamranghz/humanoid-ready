"""Body model utilities (lazy imports — no torch at `import hready`)."""

from __future__ import annotations

__all__ = [
    "BodyModelMismatchError",
    "BodyModelPolicyError",
    "SmplxBody",
    "SmplxOutput",
    "assert_same_body",
    "axis_angle_to_matrix",
    "axis_angle_to_quaternion",
    "geodesic_distance",
    "load_body",
    "matrix_to_axis_angle",
    "matrix_to_quaternion",
    "matrix_to_rotation_6d",
    "quaternion_to_axis_angle",
    "quaternion_to_matrix",
    "rotation_6d_to_matrix",
]

_LAZY_EXPORTS = {
    "BodyModelMismatchError": ("hready.body.smplx_wrapper", "BodyModelMismatchError"),
    "BodyModelPolicyError": ("hready.body.smplx_wrapper", "BodyModelPolicyError"),
    "SmplxBody": ("hready.body.smplx_wrapper", "SmplxBody"),
    "SmplxOutput": ("hready.body.smplx_wrapper", "SmplxOutput"),
    "assert_same_body": ("hready.body.smplx_wrapper", "assert_same_body"),
    "load_body": ("hready.body.smplx_wrapper", "load_body"),
    "axis_angle_to_matrix": ("hready.body.rotations", "axis_angle_to_matrix"),
    "axis_angle_to_quaternion": ("hready.body.rotations", "axis_angle_to_quaternion"),
    "geodesic_distance": ("hready.body.rotations", "geodesic_distance"),
    "matrix_to_axis_angle": ("hready.body.rotations", "matrix_to_axis_angle"),
    "matrix_to_quaternion": ("hready.body.rotations", "matrix_to_quaternion"),
    "matrix_to_rotation_6d": ("hready.body.rotations", "matrix_to_rotation_6d"),
    "quaternion_to_axis_angle": ("hready.body.rotations", "quaternion_to_axis_angle"),
    "quaternion_to_matrix": ("hready.body.rotations", "quaternion_to_matrix"),
    "rotation_6d_to_matrix": ("hready.body.rotations", "rotation_6d_to_matrix"),
}


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        module_path, attr = _LAZY_EXPORTS[name]
        import importlib

        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
