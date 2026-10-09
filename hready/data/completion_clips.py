"""E3 clip lists: rule-derived training list (SHA256 fingerprint) and VAL/TEST eval lists."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from hready.data.amass import assign_split, clip_flags, load_index

FLOOR_WORK_ELIGIBLE_CLASSES: tuple[str, ...] = ("kneel", "lie")


def sha256_lines(lines: list[str]) -> str:
    """Same convention as the E1 split hashes (``ego_splits._sha256_lines``): newline-terminated."""
    payload = "\n".join(lines) + ("\n" if lines else "")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_tune_subject_ids(splits_json: Path) -> set[str]:
    data = json.loads(Path(splits_json).read_text(encoding="utf-8"))
    return set(data["tune"]["subject_ids"])


def build_train_clip_list(*, splits_json: Path) -> tuple[list[str], dict[str, Any]]:
    """Recorded rule: ``assign_split == train``, ``skate_flag`` false, not one of the tune subjects."""
    tune = load_tune_subject_ids(splits_json)
    rels: list[str] = []
    n_split = n_skate = n_tune = 0
    for e in load_index():
        if assign_split(e) != "train":
            continue
        n_split += 1
        if clip_flags(e).get("skate_flag"):
            n_skate += 1
            continue
        if f"{e.subset}/{e.subject}" in tune:
            n_tune += 1
            continue
        rels.append(e.rel_path)
    rels.sort()
    meta = {
        "rule": "assign_split == train AND skate_flag == false AND subject not in splits.json tune.subject_ids",
        "n_train_split": n_split,
        "n_removed_skate_flag": n_skate,
        "n_removed_tune_subject": n_tune,
        "n_tune_subjects": len(tune),
        "n_clips": len(rels),
        "train_clip_list_sha256": sha256_lines(rels),
    }
    return rels, meta


def split_rels(split: str) -> list[str]:
    return sorted(e.rel_path for e in load_index() if assign_split(e) == split)


def floor_work_eligible_rels(floor_csv: Path) -> list[str]:
    with Path(floor_csv).open(newline="", encoding="utf-8") as f:
        return sorted(
            {
                r["rel_path"]
                for r in csv.DictReader(f)
                if r["geometry_class"] in FLOOR_WORK_ELIGIBLE_CLASSES
            }
        )


def _seeded_subset(
    rels: list[str], n: int | None, rng: np.random.Generator
) -> list[str]:
    if n is None or n >= len(rels):
        return list(rels)
    return sorted(rng.choice(rels, size=n, replace=False).tolist())


def resolve_clip_lists(cfg: dict[str, Any]) -> dict[str, Any]:
    """Train / VAL / TEST clip lists. Optional ``subset`` config limits counts (pipeline checks only).

    A subset always keeps every ``floor_work_eligible`` clip of the chosen eval split so the
    floor-work slice is exercised; it never moves clips across splits.
    """
    paths = cfg["paths"]
    train, meta = build_train_clip_list(splits_json=Path(paths["splits_json"]))
    val, test = split_rels("val"), split_rels("test")
    floor = set(floor_work_eligible_rels(Path(paths["floor_work_csv"])))
    sub = cfg.get("subset")
    out: dict[str, Any] = {"train_meta": meta, "subset": sub}
    if sub:
        rng = np.random.default_rng(int(sub.get("seed", 0)))
        train = _seeded_subset(train, sub.get("train_n"), rng)
        val = sorted(
            set(_seeded_subset(val, sub.get("val_n"), rng)) | (floor & set(val))
        )
        test = sorted(
            set(_seeded_subset(test, sub.get("test_n"), rng)) | (floor & set(test))
        )
    out.update(train=train, val=val, test=test)
    out["selection"] = _seeded_subset(
        val,
        cfg["selection"].get("n_val_clips"),
        np.random.default_rng(int(cfg["seed"])),
    )
    return out
