"""Data loaders (lazy imports; no torch/numpy at package root)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

_LAZY: dict[str, tuple[str, str]] = {
    "build_index": ("hready.data.amass", "build_index"),
    "load_index": ("hready.data.amass", "load_index"),
    "load_clip": ("hready.data.amass", "load_clip"),
    "floor_offset": ("hready.data.amass", "floor_offset"),
    "assign_split": ("hready.data.amass", "assign_split"),
    "assert_no_subject_leakage": ("hready.data.amass", "assert_no_subject_leakage"),
    "AmassIndexEntry": ("hready.data.amass", "AmassIndexEntry"),
    "build_babel_index": ("hready.data.babel", "build_babel_index"),
    "load_babel_labels": ("hready.data.babel", "load_babel_labels"),
    "load_contact": ("hready.data.contact", "load_contact"),
    "clip_flags": ("hready.data.amass", "clip_flags"),
    "build_floor_cache": ("hready.data.amass", "build_floor_cache"),
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
    from hready.data.amass import (
        AmassIndexEntry,
        assert_no_subject_leakage,
        assign_split,
        build_index,
        floor_offset,
        load_clip,
        load_index,
    )
    from hready.data.babel import build_babel_index, load_babel_labels
    from hready.data.contact import load_contact
