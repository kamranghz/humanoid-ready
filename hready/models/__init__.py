"""Neural models (lazy imports)."""

from __future__ import annotations

from typing import Any

_LAZY = {"HRRefine": ("hready.models.hr_refine", "HRRefine")}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        import importlib

        return getattr(importlib.import_module(mod_name), attr)
    raise AttributeError(name)


__all__ = list(_LAZY.keys())
