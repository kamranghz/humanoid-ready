"""Robot retargeting and simulation bridge (lazy imports)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

_LAZY: dict[str, tuple[str, str]] = {
    "retarget_clip": ("hready.robot.retarget", "retarget_clip"),
    "GMR_JOINT_NAMES": ("hready.robot.retarget", "GMR_JOINT_NAMES"),
}

__all__ = list(_LAZY.keys())


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        module_name, attr = _LAZY[name]
        import importlib

        mod = importlib.import_module(module_name)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    from hready.robot.retarget import GMR_JOINT_NAMES, retarget_clip  # noqa: F401
