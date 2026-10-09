"""BABEL v1.0 labels aligned to AMASS SMPL-X N index (item 5).

No PyTorch at import time; numpy only for label tensors.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from hready.data.amass import (
    AmassIndexEntry,
    _entry_by_rel,
    assign_split,
    cache_dir_from_config,
    load_index,
    load_index_meta,
    load_paths_config,
)

_TARGET_FPS = 30.0
_INDEX_VERSION = 3
_INDEX_NAME = "babel_index.json"
_BABEL_SPLITS = ("train.json", "val.json", "test.json", "extra_train.json", "extra_val.json")

# BABEL feat_p prefix -> (AMASS subset, path segments to strip after prefix).
_BABEL_PREFIX_MAP: dict[str, tuple[str, tuple[str, ...]]] = {
    "BMLrub": ("BMLrub", ("BioMotionLab_NTroje",)),
    "ACCAD": ("ACCAD", ("ACCAD",)),
    "CMU": ("CMU", ("CMU",)),
    "EyesJapanDataset": ("Eyes_Japan_Dataset", ("Eyes_Japan_Dataset",)),
    "MPIHDM05": ("HDM05", ("MPI_HDM05",)),
    "KIT": ("KIT", ("KIT",)),
    "EKUT": ("EKUT", ("EKUT",)),
    "MPImosh": ("MoSh", ("MPI_mosh",)),
    "TCDhandMocap": ("TCDHands", ("TCD_handMocap",)),
    "DFaust67": ("DFaust", ("DFaust_67",)),
    "MPILimits": ("PosePrior", ("MPI_Limits",)),
    "SFU": ("SFU", ("SFU",)),
    "TotalCapture": ("TotalCapture", ("TotalCapture",)),
    "HumanEva": ("HumanEva", ("HumanEva",)),
    "SSMsynced": ("SSM", ("SSM_synced",)),
    "BMLmovi": ("BMLmovi", ("BMLmovi",)),
    "Transitionsmocap": ("Transitions", ("Transitions_mocap",)),
}

_DURATION_TOL_S = 0.5
_VOCAB_MIN_HOURS = 0.5  # 30 minutes labeled time per act_cat
_AFFORDANCE_ACT_CAT: tuple[str, ...] = (
    "lean",
    "lie",
    "kneel",
    "crouch",
    "jump",
    "stand up",
)


def babel_root_from_config(cfg: Optional[dict[str, Any]] = None) -> Path:
    cfg = cfg or load_paths_config()
    data_root = Path(cfg["data_root"])
    if "babel_root" in cfg:
        return Path(cfg["babel_root"])
    return data_root / "datasets" / "babel" / "babel_v1.0_release"


def _index_path(cache_dir: Path) -> Path:
    return cache_dir / _INDEX_NAME


def _poses_basename_to_stageii(name: str) -> str:
    if name.endswith("_poses_poses.npz"):
        return name[: -len("_poses_poses.npz")] + "_stageii.npz"
    if name.endswith("_poses.npz"):
        return name[: -len("_poses.npz")] + "_stageii.npz"
    if name.endswith(".npz"):
        return name[: -len(".npz")] + "_stageii.npz"
    return f"{name}_stageii.npz"


def feat_p_to_rel_path(feat_p: str) -> Optional[str]:
    """Map BABEL ``feat_p`` to AMASS ``rel_path`` (subset/folder/..._stageii.npz)."""
    parts = feat_p.strip().split("/")
    if not parts:
        return None
    babel_prefix = parts[0]
    if babel_prefix not in _BABEL_PREFIX_MAP:
        return None
    subset, strip_segs = _BABEL_PREFIX_MAP[babel_prefix]
    rest = parts[1:]
    for seg in strip_segs:
        if rest and rest[0] == seg:
            rest = rest[1:]
    if not rest:
        return None
    fname = _poses_basename_to_stageii(rest[-1])
    return "/".join([subset, *rest[:-1], fname])


def normalize_rel_path(rel_path: str) -> str:
    """Lowercase; collapse spaces, underscores, dashes in the path after the subset."""
    parts = rel_path.strip().split("/")
    if not parts:
        return ""
    subset = parts[0]
    tail = "/".join(parts[1:]).lower()
    tail = re.sub(r"[\s_\-]+", "", tail)
    return f"{subset}/{tail}"


def _duration_diff_seconds(babel_dur: float, playback_duration: float) -> float:
    """BABEL ``dur`` vs AMASS index ``duration`` (playback timeline)."""
    return abs(babel_dur - playback_duration)


def act_cat_matches_keyword(act_cat: str, keyword: str) -> bool:
    """Exact category or whole-token match (not substring: ``sit`` != ``transition``)."""
    cat = act_cat.strip().lower()
    kw = keyword.strip().lower()
    if not cat or not kw:
        return False
    if cat == kw:
        return True
    tokens = re.split(r"[\s_/]+", cat)
    return kw in tokens


def is_bmlrub_treadmill_clip(feat_p: str, rel_path: str) -> bool:
    return "treadmill" in feat_p.lower() or "treadmill" in rel_path.lower()


@dataclass
class BabelSegment:
    start_t: float
    end_t: float
    act_cat: list[str]


@dataclass
class BabelIndexEntry:
    babel_file: str
    babel_sid: str
    rel_path: str
    feat_p: str
    babel_dur: float
    playback_duration: float
    ann_source: str  # "frame_ann" | "seq_ann"
    segments: list[dict[str, Any]]


def _extract_segments(seq: dict[str, Any]) -> tuple[str, list[BabelSegment]]:
    frame_ann = seq.get("frame_ann") or {}
    seq_ann = seq.get("seq_ann") or {}
    dur = float(seq.get("dur", 0.0))

    if isinstance(frame_ann, dict) and frame_ann.get("labels"):
        segs: list[BabelSegment] = []
        for lab in frame_ann["labels"]:
            cats = list(lab.get("act_cat") or [])
            if not cats:
                continue
            segs.append(
                BabelSegment(
                    start_t=float(lab["start_t"]),
                    end_t=float(lab["end_t"]),
                    act_cat=cats,
                )
            )
        return "frame_ann", segs

    if isinstance(seq_ann, dict) and seq_ann.get("labels"):
        cats: list[str] = []
        for lab in seq_ann["labels"]:
            for c in lab.get("act_cat") or []:
                if c not in cats:
                    cats.append(c)
        if cats:
            return "seq_ann", [BabelSegment(start_t=0.0, end_t=dur, act_cat=cats)]
    return "none", []


def _build_rel_lookups(
    amass_entries: list[AmassIndexEntry],
) -> tuple[set[str], dict[tuple[str, str], list[str]], dict[str, list[str]]]:
    rel_set = {e.rel_path for e in amass_entries}
    by_base: dict[tuple[str, str], list[str]] = defaultdict(list)
    by_norm: dict[str, list[str]] = defaultdict(list)
    for e in amass_entries:
        base = Path(e.rel_path).name
        by_base[(e.subset, base)].append(e.rel_path)
        by_norm[normalize_rel_path(e.rel_path)].append(e.rel_path)
    return rel_set, dict(by_base), dict(by_norm)


def _resolve_rel_path(
    feat_p: str,
    rel_set: set[str],
    by_base: dict[tuple[str, str], list[str]],
    by_norm: dict[str, list[str]],
) -> tuple[Optional[str], str]:
    mapped = feat_p_to_rel_path(feat_p)
    if mapped and mapped in rel_set:
        return mapped, "direct"
    if mapped:
        subset = mapped.split("/")[0]
        base = Path(mapped).name
        cands = by_base.get((subset, base), [])
        if len(cands) == 1:
            return cands[0], "basename_unique"
        norm = normalize_rel_path(mapped)
        nc = by_norm.get(norm, [])
        if len(nc) == 1:
            return nc[0], "normalized_path"
    prefix = feat_p.split("/")[0] if feat_p else ""
    if prefix not in _BABEL_PREFIX_MAP:
        return None, "unknown_prefix"
    return None, "not_in_index"


def build_babel_index(
    *,
    babel_root: Optional[Path] = None,
    amass_root: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    force: bool = False,
) -> dict[str, Any]:
    """Scan BABEL JSON, map to AMASS clips, validate duration, cache index."""
    from hready.data.amass import amass_root_from_config

    babel_root = (babel_root or babel_root_from_config()).resolve()
    amass_root = (amass_root or amass_root_from_config()).resolve()
    cache_dir = (cache_dir or cache_dir_from_config()).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = _index_path(cache_dir)
    if out_path.is_file() and not force:
        return load_babel_index_payload(cache_dir)

    amass_entries = load_index(cache_dir)
    rel_set, by_base, by_norm = _build_rel_lookups(amass_entries)
    amass_by_rel = {e.rel_path: e for e in amass_entries}

    entries: list[BabelIndexEntry] = []
    unmatched_reasons: Counter[str] = Counter()
    unmatched_examples: dict[str, list[str]] = defaultdict(list)
    per_file_total: dict[str, int] = {}
    per_file_matched: dict[str, int] = {}
    per_file_mapped: dict[str, int] = {}
    per_prefix_total: Counter[str] = Counter()
    per_prefix_matched: Counter[str] = Counter()
    dur_diffs: list[float] = []
    dur_failures: list[dict[str, Any]] = []
    bmlrub_treadmill_labeled = 0

    for babel_file in _BABEL_SPLITS:
        path = babel_root / babel_file
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        per_file_total[babel_file] = len(data)
        matched_n = 0
        mapped_n = 0
        for sid, seq in data.items():
            feat_p = seq.get("feat_p", "")
            prefix = feat_p.split("/")[0] if feat_p else "?"
            per_prefix_total[prefix] += 1

            rel, how = _resolve_rel_path(feat_p, rel_set, by_base, by_norm)
            if rel is None:
                reason = how
                unmatched_reasons[reason] += 1
                if len(unmatched_examples[reason]) < 3:
                    unmatched_examples[reason].append(feat_p)
                continue

            play_dur = float(amass_by_rel[rel].duration)
            bd = float(seq.get("dur", 0.0))
            dd = _duration_diff_seconds(bd, play_dur)
            dur_diffs.append(dd)
            if dd > _DURATION_TOL_S:
                dur_failures.append(
                    {
                        "feat_p": feat_p,
                        "rel_path": rel,
                        "babel_dur": bd,
                        "playback_duration": play_dur,
                        "diff_s": dd,
                    }
                )
                unmatched_reasons["duration_mismatch"] += 1
                if len(unmatched_examples["duration_mismatch"]) < 3:
                    unmatched_examples["duration_mismatch"].append(feat_p)
                continue

            mapped_n += 1
            per_prefix_matched[prefix] += 1

            ann_source, segs = _extract_segments(seq)
            if ann_source == "none":
                unmatched_reasons["no_labels"] += 1
                if len(unmatched_examples["no_labels"]) < 3:
                    unmatched_examples["no_labels"].append(feat_p)
                continue

            entries.append(
                BabelIndexEntry(
                    babel_file=babel_file,
                    babel_sid=str(sid),
                    rel_path=rel,
                    feat_p=feat_p,
                    babel_dur=bd,
                    playback_duration=play_dur,
                    ann_source=ann_source,
                    segments=[asdict(s) for s in segs],
                )
            )
            matched_n += 1
            if is_bmlrub_treadmill_clip(feat_p, rel):
                bmlrub_treadmill_labeled += 1
        per_file_matched[babel_file] = matched_n
        per_file_mapped[babel_file] = mapped_n

    if dur_failures:
        raise RuntimeError(
            f"BABEL duration validation failed for {len(dur_failures)} pairs; "
            f"first: {dur_failures[0]}"
        )

    vocab_meta = _compute_vocabulary_stats(entries)

    payload: dict[str, Any] = {
        "version": _INDEX_VERSION,
        "babel_root": str(babel_root),
        "meta": {
            "per_file_total": per_file_total,
            "per_file_matched": per_file_matched,
            "per_file_mapped": per_file_mapped,
            "per_prefix_total": dict(per_prefix_total),
            "per_prefix_matched": dict(per_prefix_matched),
            "unmatched_reasons": dict(unmatched_reasons),
            "unmatched_examples": dict(unmatched_examples),
            "duration_diff_median_s": float(np.median(dur_diffs)) if dur_diffs else 0.0,
            "duration_diff_max_s": float(max(dur_diffs)) if dur_diffs else 0.0,
            "duration_pairs_above_tol": sum(1 for d in dur_diffs if d > _DURATION_TOL_S),
            "duration_strict_above_0_5_s": sum(1 for d in dur_diffs if d > 0.5),
            "ann_source_counts": _ann_source_counts(entries),
            "bmlrub_treadmill_labeled_sequences": bmlrub_treadmill_labeled,
            "vocabulary": vocab_meta,
        },
        "entries": [asdict(e) for e in entries],
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    _print_build_summary(payload)
    return payload


def load_babel_index_payload(cache_dir: Optional[Path] = None) -> dict[str, Any]:
    cache_dir = cache_dir or cache_dir_from_config()
    path = _index_path(cache_dir)
    if not path.is_file():
        raise FileNotFoundError(f"BABEL index not found at {path}; run build_babel_index() first.")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _entries_by_rel(payload: dict[str, Any]) -> dict[str, BabelIndexEntry]:
    out: dict[str, BabelIndexEntry] = {}
    for row in payload["entries"]:
        row = dict(row)
        if "playback_duration" not in row:
            bd = float(row.get("babel_dur", 0.0))
            row.setdefault("playback_duration", float(row.get("mocap_time_length", bd)))
        row.pop("mocap_time_length", None)
        row.pop("time_scale", None)
        out[row["rel_path"]] = BabelIndexEntry(**row)
    return out


def _ann_source_counts(entries: list[BabelIndexEntry]) -> dict[str, int]:
    c: Counter[str] = Counter()
    for e in entries:
        c[e.ann_source] += 1
    return dict(c)


def ann_source_for_segmentation(entries: list[BabelIndexEntry]) -> list[BabelIndexEntry]:
    """Sequences safe for frame-level segmentation metrics (``frame_ann`` only)."""
    return [e for e in entries if e.ann_source == "frame_ann"]


def _iter_babel_entries(payload: dict[str, Any]) -> list[BabelIndexEntry]:
    return list(_entries_by_rel(payload).values())


def print_annotation_source_report(payload: Optional[dict[str, Any]] = None) -> None:
    """Per BABEL file and our split: ``frame_ann`` vs ``seq_ann`` labeled counts."""
    payload = payload or load_babel_index_payload()
    amass_entries = {e.rel_path: e for e in load_index()}
    idx_meta = load_index_meta()
    by_file: dict[str, Counter[str]] = defaultdict(Counter)
    by_our: dict[str, Counter[str]] = defaultdict(Counter)
    for e in _iter_babel_entries(payload):
        by_file[e.babel_file][e.ann_source] += 1
        if e.rel_path in amass_entries:
            sp = assign_split(amass_entries[e.rel_path], meta=idx_meta)
            by_our[sp][e.ann_source] += 1
    print("BABEL ann_source (labeled sequences in index):")
    for bf in _BABEL_SPLITS:
        if bf in by_file:
            print(f"  {bf}: {dict(by_file[bf])}")
    print("  our train/val/test:")
    for sp in ("train", "val", "test"):
        print(f"    {sp}: {dict(by_our.get(sp, {}))}")


def _compute_vocabulary_stats(entries: list[BabelIndexEntry]) -> dict[str, Any]:
    hours: Counter[str] = Counter()
    distinct: set[str] = set()
    for e in entries:
        for seg in e.segments:
            start_t = float(seg["start_t"])
            end_t = float(seg["end_t"])
            dt_h = max(0.0, end_t - start_t) / 3600.0
            for c in seg["act_cat"]:
                distinct.add(c)
                hours[c] += dt_h
    kept = sorted([c for c, h in hours.items() if h >= _VOCAB_MIN_HOURS])
    affordance = list(_AFFORDANCE_ACT_CAT)
    vocab_set = set(kept) | set(affordance)
    vocabulary = sorted(vocab_set, key=lambda c: (-hours.get(c, 0.0), c))
    top30 = hours.most_common(30)
    aff_stats = {
        c: {"hours": float(hours.get(c, 0.0)), "sequences": _affordance_sequence_count(entries, c)}
        for c in affordance
    }
    return {
        "distinct_act_cat": len(distinct),
        "categories_ge_30min": len(kept),
        "affordance_channels": affordance,
        "vocabulary": vocabulary,
        "vocabulary_k": len(vocabulary),
        "vocab_rule": (
            f">={_VOCAB_MIN_HOURS} h labeled time ({_VOCAB_MIN_HOURS*60:.0f} min) plus fixed "
            f"affordance channels {list(_AFFORDANCE_ACT_CAT)}"
        ),
        "hours_top30": [(c, float(h)) for c, h in top30],
        "hours_per_category": {c: float(hours[c]) for c in vocabulary if c in hours},
        "affordance_stats": aff_stats,
    }


def _affordance_sequence_count(entries: list[BabelIndexEntry], act_cat: str) -> int:
    n = 0
    for e in entries:
        found = False
        for seg in e.segments:
            if act_cat in (seg.get("act_cat") or []):
                found = True
                break
        if found:
            n += 1
    return n


def load_babel_labels(
    entry: Union[AmassIndexEntry, str, dict[str, Any]],
    n_frames: int,
    *,
    target_fps: float = _TARGET_FPS,
    cache_dir: Optional[Path] = None,
) -> tuple[np.ndarray, list[str]]:
    """Multi-hot frame labels on the 30 Hz grid (``n_frames`` from ``load_clip``).

    Returns ``(labels, vocabulary)`` with ``labels`` uint8 ``(T, K)`` and fixed
    ``vocabulary`` (length K) from the cached BABEL index.
    """
    cache_dir = cache_dir or cache_dir_from_config()
    payload = load_babel_index_payload(cache_dir)
    vocab: list[str] = payload["meta"]["vocabulary"]["vocabulary"]
    if not vocab:
        return np.zeros((n_frames, 0), dtype=np.uint8), vocab

    if isinstance(entry, dict):
        entry = AmassIndexEntry(**entry)
    elif isinstance(entry, str):
        entry = _entry_by_rel(load_index(cache_dir), entry)

    by_rel = _entries_by_rel(payload)
    rec = by_rel.get(entry.rel_path)
    if rec is None:
        return np.zeros((n_frames, len(vocab)), dtype=np.uint8), vocab

    cat_to_i = {c: i for i, c in enumerate(vocab)}
    labels = np.zeros((n_frames, len(vocab)), dtype=np.uint8)
    for seg in rec.segments:
        i0 = int(math.floor(float(seg["start_t"]) * target_fps))
        i1 = int(math.floor(float(seg["end_t"]) * target_fps))
        i0 = max(0, min(i0, n_frames))
        i1 = max(0, min(i1, n_frames))
        if i1 <= i0:
            continue
        for c in seg["act_cat"]:
            j = cat_to_i.get(c)
            if j is None:
                continue
            labels[i0:i1, j] = 1
    return labels, vocab


def _print_build_summary(payload: dict[str, Any]) -> None:
    meta = payload["meta"]
    print("BABEL index summary")
    for bf in _BABEL_SPLITS:
        tot = meta["per_file_total"].get(bf, 0)
        mapped = meta["per_file_mapped"].get(bf, 0)
        labeled = meta["per_file_matched"].get(bf, 0)
        print(f"  {bf}: mapped {mapped} / {tot}, labeled {labeled} / {tot}")
    print(
        f"  duration check: median diff {meta['duration_diff_median_s']:.4f} s, "
        f"max {meta['duration_diff_max_s']:.4f} s, "
        f">{_DURATION_TOL_S} s: {meta['duration_pairs_above_tol']}"
    )
    vm = meta["vocabulary"]
    print(f"  distinct act_cat: {vm['distinct_act_cat']}")
    print(
        f"  vocabulary K={vm.get('vocabulary_k', len(vm['vocabulary']))} "
        f"({vm['categories_ge_30min']} >=30min + affordance); {vm['vocab_rule']}"
    )
    print(
        f"  ann_source: {meta.get('ann_source_counts', {})}; "
        f"BMLrub treadmill labeled: {meta.get('bmlrub_treadmill_labeled_sequences', 0)}"
    )
    print("  unmatched reasons:", meta["unmatched_reasons"])
    for reason, ex in meta.get("unmatched_examples", {}).items():
        if ex:
            print(f"    {reason} examples: {ex[:3]}")
    labeled_by_prefix: Counter[str] = Counter()
    for row in payload["entries"]:
        labeled_by_prefix[row["feat_p"].split("/")[0]] += 1
    print("  per-prefix mapped / total / labeled:")
    for pr in sorted(
        meta["per_prefix_total"], key=lambda k: -meta["per_prefix_total"][k]
    ):
        print(
            f"    {pr}: {meta['per_prefix_matched'].get(pr, 0)} / "
            f"{meta['per_prefix_total'][pr]} / {labeled_by_prefix.get(pr, 0)}"
        )
    print_babel_coverage_reports(payload)


def print_babel_coverage_reports(payload: Optional[dict[str, Any]] = None) -> None:
    """Per our train/val/test and BABEL-vs-ours cross-split tables."""
    payload = payload or load_babel_index_payload()
    meta = payload["meta"]
    amass_entries = {e.rel_path: e for e in load_index()}
    idx_meta = load_index_meta()

    by_our: dict[str, list[BabelIndexEntry]] = defaultdict(list)
    for row in payload["entries"]:
        e = BabelIndexEntry(**row)
        if e.rel_path not in amass_entries:
            continue
        sp = assign_split(amass_entries[e.rel_path], meta=idx_meta)
        by_our[sp].append(e)

    print("BABEL labels vs our split (matched sequences with labels):")
    for sp in ("train", "val", "test"):
        es = by_our.get(sp, [])
        hours = sum(e.babel_dur for e in es) / 3600.0
        print(f"  our {sp}: {len(es)} sequences, {hours:.2f} labeled h (BABEL dur sum)")

    vocab = meta["vocabulary"]["vocabulary"]
    cat_h: Counter[str] = Counter()
    for e in by_our.get("train", []) + by_our.get("val", []) + by_our.get("test", []):
        for seg in e.segments:
            dt = max(0.0, float(seg["end_t"]) - float(seg["start_t"])) / 3600.0
            for c in seg["act_cat"]:
                if c in vocab:
                    cat_h[c] += dt
    print("  top-10 act_cat hours (our train+val+test, vocab only):")
    for c, h in cat_h.most_common(10):
        print(f"    {c}: {h:.3f} h")

    cross: Counter[tuple[str, str]] = Counter()
    for row in payload["entries"]:
        e = BabelIndexEntry(**row)
        if e.babel_file.startswith("extra"):
            continue
        if e.rel_path not in amass_entries:
            continue
        babel_sp = e.babel_file.replace(".json", "")
        our_sp = assign_split(amass_entries[e.rel_path], meta=idx_meta)
        cross[(babel_sp, our_sp)] += 1
    print("BABEL official split vs our split (non-extra, matched):")
    for (bs, os), n in sorted(cross.items()):
        if bs != os:
            print(f"  BABEL {bs} & our {os}: {n}")
    off = sum(n for (bs, os), n in cross.items() if bs != os)
    print(f"  sequences crossing BABEL/our boundary: {off}")


def segment_frame_count_check(
    start_t: float,
    end_t: float,
    target_fps: float = _TARGET_FPS,
) -> dict[str, float]:
    """Off-by-one helper: frames [floor(start*fps), floor(end*fps))."""
    i0 = int(math.floor(start_t * target_fps))
    i1 = int(math.floor(end_t * target_fps))
    return {
        "start_t": start_t,
        "end_t": end_t,
        "frame_start": i0,
        "frame_end_exclusive": i1,
        "n_frames_labeled": i1 - i0,
        "duration_times_fps": (end_t - start_t) * target_fps,
    }
