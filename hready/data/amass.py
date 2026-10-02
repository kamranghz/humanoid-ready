"""AMASS SMPL-X N loader (stage 5a): index, 30 Hz resampling, Z-up floor, splits.

No PyTorch import at module import time; SMPL-X / torch are loaded lazily for
floor height and optional checks.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Literal, Optional, Union

import numpy as np
import yaml

_TARGET_FPS = 30.0
_INDEX_VERSION = 2
_INDEX_NAME = "amass_index.json"
_FLOOR_NAME = "amass_floor.json"

# Top-level folders under smplx_n that are not motion subsets.
_NON_SUBSET_DIRS = frozenset({"extra", "train", "val", "test"})
_MOYO_SPLITS = ("train", "val", "test", "extra")
_MOTION_SUFFIX = "_stageii.npz"


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "configs" / "paths.example.yaml").is_file():
            return parent
    raise FileNotFoundError("Could not locate repo root.")


def load_paths_config() -> dict[str, Any]:
    paths_yaml = _repo_root() / "configs" / "paths.yaml"
    if not paths_yaml.is_file():
        raise FileNotFoundError(
            f"Missing {paths_yaml}; copy configs/paths.example.yaml to configs/paths.yaml."
        )
    with paths_yaml.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def amass_root_from_config(cfg: Optional[dict[str, Any]] = None) -> Path:
    cfg = cfg or load_paths_config()
    data_root = Path(cfg["data_root"])
    if "amass_root" in cfg:
        return Path(cfg["amass_root"])
    return data_root / "datasets" / "amass" / "smplx_n"


def cache_dir_from_config(cfg: Optional[dict[str, Any]] = None) -> Path:
    cfg = cfg or load_paths_config()
    data_root = Path(cfg["data_root"])
    return Path(cfg.get("cache_dir", data_root / "cache"))


@dataclass
class AmassIndexEntry:
    rel_path: str
    subset: str
    subject: str
    clip_id: str
    fps: float
    n_frames: int
    duration: float
    gender: str
    moyo_split: Optional[str] = None
    split_group: str = ""
    split: Optional[str] = None

    @property
    def hours(self) -> float:
        return self.duration / 3600.0


def _clip_id_from_path(npz_path: Path) -> str:
    return npz_path.stem.replace("_stageii", "")


def _motion_skip_reason(npz_path: Path) -> Optional[str]:
    if not npz_path.name.endswith(_MOTION_SUFFIX) or "_shape" in npz_path.name:
        return "not a motion stageii npz"
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            if "trans" not in data:
                return "missing trans"
            if "mocap_frame_rate" not in data:
                return "missing mocap_frame_rate"
            trans = data["trans"]
            fps = float(data["mocap_frame_rate"])
            n_frames = int(trans.shape[0])
            if n_frames < 1:
                return "empty trans"
            if fps <= 0:
                return "non-positive mocap_frame_rate"
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def _read_motion_header(npz_path: Path) -> Optional[dict[str, Any]]:
    reason = _motion_skip_reason(npz_path)
    if reason is not None:
        return None
    with np.load(npz_path, allow_pickle=True) as data:
        trans = data["trans"]
        fps = float(data["mocap_frame_rate"])
        n_frames = int(trans.shape[0])
        gender = str(data["gender"]) if "gender" in data else "unknown"
        duration = (n_frames - 1) / fps if n_frames > 1 else 0.0
        return {
            "fps": fps,
            "n_frames": n_frames,
            "duration": duration,
            "gender": gender,
        }


@dataclass
class ScanStats:
    skipped_corrupt_or_non_motion: int = 0
    skipped_paths: list[str] = None  # type: ignore[assignment]
    skipped_reasons: list[str] = None  # type: ignore[assignment]
    moyo_single_subject: bool = False
    moyo_distinct_subjects: int = 0

    def __post_init__(self) -> None:
        if self.skipped_paths is None:
            self.skipped_paths = []
        if self.skipped_reasons is None:
            self.skipped_reasons = []


def _iter_motion_npz(amass_root: Path) -> tuple[list[AmassIndexEntry], ScanStats]:
    entries: list[AmassIndexEntry] = []
    stats = ScanStats()
    amass_root = amass_root.resolve()

    def _consume(npz_path: Path, subset: str, subject: str, moyo_split: Optional[str]) -> None:
        if not npz_path.name.endswith(_MOTION_SUFFIX) or "_shape" in npz_path.name:
            return
        hdr = _read_motion_header(npz_path)
        if hdr is None:
            stats.skipped_corrupt_or_non_motion += 1
            stats.skipped_paths.append(npz_path.as_posix())
            stats.skipped_reasons.append(_motion_skip_reason(npz_path) or "unknown")
            return
        rel = npz_path.relative_to(amass_root).as_posix()
        entries.append(
            AmassIndexEntry(
                rel_path=rel,
                subset=subset,
                subject=subject,
                clip_id=_clip_id_from_path(npz_path),
                fps=hdr["fps"],
                n_frames=hdr["n_frames"],
                duration=hdr["duration"],
                gender=hdr["gender"],
                moyo_split=moyo_split,
            )
        )

    for split in _MOYO_SPLITS:
        split_root = amass_root / split
        if not split_root.is_dir():
            continue
        for npz_path in sorted(split_root.rglob(f"*{_MOTION_SUFFIX}")):
            parts = npz_path.relative_to(split_root).parts
            subject = parts[1] if len(parts) >= 2 else parts[0]
            _consume(npz_path, "MOYO", subject, split)
    for child in sorted(amass_root.iterdir()):
        if not child.is_dir() or child.name in _NON_SUBSET_DIRS:
            continue
        subset = child.name
        for npz_path in sorted(child.rglob(f"*{_MOTION_SUFFIX}")):
            parts = npz_path.relative_to(child).parts
            subject = parts[0] if parts else "unknown"
            _consume(npz_path, subset, subject, None)

    moyo_subjects = {e.subject for e in entries if e.subset == "MOYO"}
    stats.moyo_distinct_subjects = len(moyo_subjects)
    stats.moyo_single_subject = stats.moyo_distinct_subjects <= 1
    return entries, stats


def _index_path(cache_dir: Path) -> Path:
    return cache_dir / _INDEX_NAME


def _floor_path(cache_dir: Path) -> Path:
    return cache_dir / _FLOOR_NAME


def load_index(cache_dir: Optional[Path] = None) -> list[AmassIndexEntry]:
    return load_index_payload(cache_dir)["entries"]


def load_index_payload(cache_dir: Optional[Path] = None) -> dict[str, Any]:
    cache_dir = cache_dir or cache_dir_from_config()
    path = _index_path(cache_dir)
    if not path.is_file():
        raise FileNotFoundError(f"AMASS index not found at {path}; run build_index() first.")
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    entries = []
    for row in raw["entries"]:
        row = dict(row)
        row.setdefault("split_group", "")
        row.setdefault("split", None)
        entries.append(AmassIndexEntry(**row))
    meta = raw.get("meta", {})
    return {"entries": entries, "meta": meta}


def load_index_meta(cache_dir: Optional[Path] = None) -> dict[str, Any]:
    return load_index_payload(cache_dir)["meta"]


def _entry_by_rel(index: list[AmassIndexEntry], rel_path: str) -> AmassIndexEntry:
    for e in index:
        if e.rel_path == rel_path:
            return e
    raise KeyError(rel_path)


def build_index(
    *,
    amass_root: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    force: bool = False,
) -> list[AmassIndexEntry]:
    """Scan AMASS once, cache index JSON under ``cache_dir``."""
    amass_root = (amass_root or amass_root_from_config()).resolve()
    cache_dir = (cache_dir or cache_dir_from_config()).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = _index_path(cache_dir)
    if out_path.is_file() and not force:
        return load_index(cache_dir)

    entries, scan_stats = _iter_motion_npz(amass_root)
    split_meta = _assign_splits_beta_groups(entries, amass_root)
    payload = {
        "version": _INDEX_VERSION,
        "amass_root": str(amass_root),
        "meta": {
            "scan_skipped_corrupt_or_non_motion": scan_stats.skipped_corrupt_or_non_motion,
            "scan_skipped_paths": scan_stats.skipped_paths,
            "scan_skipped_reasons": scan_stats.skipped_reasons,
            "moyo_distinct_subjects": scan_stats.moyo_distinct_subjects,
            "moyo_single_subject": scan_stats.moyo_single_subject,
            **split_meta,
        },
        "entries": [asdict(e) for e in entries],
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    _print_index_summary(entries, scan_stats)
    audit_subject_grouping(entries, amass_root=amass_root)
    _print_split_summary(entries, meta=payload["meta"])
    return entries


def _print_index_summary(
    entries: list[AmassIndexEntry], scan_stats: Optional[ScanStats] = None
) -> None:
    from collections import Counter

    print("AMASS index summary")
    if scan_stats is not None:
        print(
            f"  scan skipped (corrupt / non-motion): "
            f"{scan_stats.skipped_corrupt_or_non_motion}"
        )
        for p, why in zip(scan_stats.skipped_paths, scan_stats.skipped_reasons):
            print(f"    skipped: {p} ({why})")
        print(
            f"  MOYO distinct subjects: {scan_stats.moyo_distinct_subjects} "
            f"(single_subject={scan_stats.moyo_single_subject})"
        )
    print(f"  total clips: {len(entries)}")
    total_h = sum(e.hours for e in entries)
    print(f"  total hours: {total_h:.2f}")

    by_sub: dict[str, list[AmassIndexEntry]] = {}
    for e in entries:
        by_sub.setdefault(e.subset, []).append(e)
    print("  per subset (clips, hours):")
    for subset in sorted(by_sub):
        es = by_sub[subset]
        print(f"    {subset}: {len(es)} clips, {sum(x.hours for x in es):.2f} h")

    fps_hist = Counter(round(e.fps, 4) for e in entries)
    print("  fps histogram (top):")
    for fps, cnt in fps_hist.most_common(15):
        print(f"    {fps}: {cnt}")

    not_mult = [e for e in entries if abs(e.fps / _TARGET_FPS - round(e.fps / _TARGET_FPS)) > 1e-3]
    print(f"  clips with fps not a multiple of 30: {len(not_mult)}")
    short = [e for e in entries if e.duration < 1.0]
    print(f"  clips shorter than 1 s: {len(short)}")


def _split_bucket_key(key: str) -> Literal["train", "val", "test"]:
    h = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16)
    slot = h % 10
    if slot < 8:
        return "train"
    if slot < 9:
        return "val"
    return "test"


def _split_bucket(subset: str, group_key: str) -> Literal["train", "val", "test"]:
    return _split_bucket_key(f"{subset}/{group_key}")


def _split_bucket_subset(subset: str) -> Literal["train", "val", "test"]:
    return _split_bucket_key(f"subset:{subset}")


def _beta_vector_key(betas: np.ndarray) -> tuple[float, ...]:
    b = np.asarray(betas, dtype=np.float64).reshape(-1)[:16]
    return tuple(np.round(b, 4).tolist())


def _read_npz_betas(npz_path: Path) -> np.ndarray:
    with np.load(npz_path, allow_pickle=True) as data:
        return np.asarray(data["betas"], dtype=np.float64).reshape(-1)[:16]


class _UnionFind:
    def __init__(self, items: list[str]) -> None:
        self._parent = {x: x for x in items}

    def find(self, x: str) -> str:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra


def _beta_connected_folder_groups(
    subset_entries: list[AmassIndexEntry], amass_root: Path
) -> tuple[dict[str, set[str]], dict[str, set[tuple[float, ...]]]]:
    """Folder (``subject``) groups = connected components sharing a beta vector."""
    from collections import defaultdict

    folder_betas: dict[str, set[tuple[float, ...]]] = defaultdict(set)
    beta_folders: dict[tuple[float, ...], set[str]] = defaultdict(set)
    for e in subset_entries:
        bk = _beta_vector_key(_read_npz_betas(amass_root / e.rel_path))
        folder_betas[e.subject].add(bk)
        beta_folders[bk].add(e.subject)

    folders = sorted(folder_betas.keys())
    uf = _UnionFind(folders)
    for flist in beta_folders.values():
        flist = sorted(flist)
        for other in flist[1:]:
            uf.union(flist[0], other)

    groups: dict[str, set[str]] = defaultdict(set)
    for f in folders:
        groups[uf.find(f)].add(f)
    return dict(groups), dict(folder_betas)


def _assign_splits_beta_groups(
    entries: list[AmassIndexEntry], amass_root: Path
) -> dict[str, Any]:
    """Assign train/val/test from beta-connected folder groups (per subset)."""
    from collections import defaultdict

    by_subset: dict[str, list[AmassIndexEntry]] = defaultdict(list)
    for e in entries:
        by_subset[e.subset].append(e)

    subset_modes: dict[str, str] = {}
    subset_whole_split: dict[str, str] = {}
    group_splits: dict[str, dict[str, str]] = {}

    for subset, es in sorted(by_subset.items()):
        groups, folder_betas = _beta_connected_folder_groups(es, amass_root)
        n_clips = len(es)
        n_folders = len(folder_betas)
        all_betas: set[tuple[float, ...]] = set()
        for s in folder_betas.values():
            all_betas |= s
        n_distinct_betas = len(all_betas)
        n_groups = len(groups)

        whole_subset = n_groups == 1 or n_distinct_betas >= n_clips
        if whole_subset:
            sp = _split_bucket_subset(subset)
            subset_whole_split[subset] = sp
            if n_groups == 1:
                subset_modes[subset] = "single_beta_group_whole_subset"
            else:
                subset_modes[subset] = "per_clip_betas_whole_subset"
            for e in es:
                e.split_group = f"{subset}::whole"
                e.split = sp
            continue

        subset_modes[subset] = "beta_connected_folder_groups"
        gmap: dict[str, str] = {}
        for root, members in groups.items():
            rep = min(members)
            gid = f"{subset}::{rep}"
            sp = _split_bucket(subset, gid)
            gmap[gid] = sp
            for e in es:
                if e.subject in members:
                    e.split_group = gid
                    e.split = sp
        group_splits[subset] = gmap

    return {
        "subset_split_modes": subset_modes,
        "subset_whole_split": subset_whole_split,
        "subset_group_splits": group_splits,
    }


def _beta_folders_from_folder_betas(
    folder_betas: dict[str, set[tuple[float, ...]]],
) -> dict[tuple[float, ...], set[str]]:
    from collections import defaultdict

    out: dict[tuple[float, ...], set[str]] = defaultdict(set)
    for folder, keys in folder_betas.items():
        for bk in keys:
            out[bk].add(folder)
    return dict(out)


def assign_split(
    entry: AmassIndexEntry, *, meta: Optional[dict[str, Any]] = None
) -> str:
    if entry.split is not None:
        return entry.split
    if meta is None:
        try:
            meta = load_index_meta()
        except FileNotFoundError:
            meta = {}
    whole = meta.get("subset_whole_split", {})
    if entry.subset in whole:
        return whole[entry.subset]
    return _split_bucket(entry.subset, entry.split_group or entry.subject)


def audit_subject_grouping(
    entries: list[AmassIndexEntry], *, amass_root: Optional[Path] = None
) -> None:
    """Print per-subset folder tables and beta-vector grouping diagnostics."""
    from collections import defaultdict

    amass_root = (amass_root or amass_root_from_config()).resolve()
    print("Subject grouping audit (folders = first-level subject field)")
    by_subset: dict[str, list[AmassIndexEntry]] = defaultdict(list)
    for e in entries:
        by_subset[e.subset].append(e)

    for subset in sorted(by_subset):
        es = by_subset[subset]
        folders = sorted({e.subject for e in es})
        examples = folders[:3]
        groups, folder_betas = _beta_connected_folder_groups(es, amass_root)
        folder_to_group = {}
        for root, members in groups.items():
            for f in members:
                folder_to_group[f] = root
        group_counts = [
            sum(1 for e in es if folder_to_group.get(e.subject, e.subject) == g)
            for g in groups
        ]
        cmin, cmax = min(group_counts), max(group_counts)
        cmed = float(np.median(group_counts))
        print(
            f"  {subset} | #folders={len(folders)} examples={examples} | "
            f"#split_groups={len(groups)} clips/group min/med/max="
            f"{cmin}/{cmed:.0f}/{cmax}"
        )

        clip_betas: list[tuple[float, ...]] = []
        for e in es:
            clip_betas.append(_beta_vector_key(_read_npz_betas(amass_root / e.rel_path)))
        n_distinct_betas = len(set(clip_betas))
        beta_to_folders = _beta_folders_from_folder_betas(folder_betas)
        shared_beta_vectors = sum(1 for fl in beta_to_folders.values() if len(fl) > 1)
        print(
            f"    betas: #distinct={n_distinct_betas} "
            f"beta_vectors_in_>1_folder={shared_beta_vectors}"
        )
        if n_distinct_betas == len(folders) and shared_beta_vectors == 0:
            print("    -> folder~subject (one beta per folder, no cross-folder sharing)")
        elif shared_beta_vectors > 0:
            print("    -> folders share betas (merge via union-find for splits)")
        elif n_distinct_betas > len(folders):
            print("    -> per-clip shape fits within folders (no stable subject identity)")
        elif len(folders) == 1:
            print("    -> single folder bucket in subset")


def _print_split_summary(
    entries: list[AmassIndexEntry], *, meta: Optional[dict[str, Any]] = None
) -> None:
    from collections import defaultdict

    if meta is None:
        try:
            meta = load_index_meta()
        except FileNotFoundError:
            meta = {}

    buckets: dict[str, list[AmassIndexEntry]] = defaultdict(list)
    groups: dict[str, set[str]] = defaultdict(set)
    subsets_by_split: dict[str, set[str]] = defaultdict(set)
    for e in entries:
        sp = assign_split(e, meta=meta)
        buckets[sp].append(e)
        groups[sp].add(e.split_group or f"{e.subset}/{e.subject}")
        subsets_by_split[sp].add(e.subset)
    whole = meta.get("subset_whole_split", {})
    print("Split summary (beta-group assign_split):")
    for sp in sorted(buckets):
        es = buckets[sp]
        print(
            f"  {sp}: {len(es)} clips, {len(groups[sp])} groups, "
            f"{sum(x.hours for x in es):.2f} h"
        )
    for sp in ("val", "test"):
        subs = sorted(subsets_by_split.get(sp, set()))
        print(f"  subsets in {sp}: {subs}")
    if whole:
        print("  whole-subset (single split by hash name):")
        for subset in sorted(whole):
            print(f"    {subset} -> {whole[subset]}")


def assert_no_subject_leakage(entries: list[AmassIndexEntry]) -> None:
    """No beta-connected split group appears in more than one train/val/test bucket."""
    by_key: dict[str, set[str]] = {}
    meta = load_index_meta()
    for e in entries:
        sp = assign_split(e, meta=meta)
        if sp not in ("train", "val", "test"):
            continue
        key = e.split_group or f"{e.subset}::{e.subject}"
        by_key.setdefault(key, set()).add(sp)
    leaks = {k: v for k, v in by_key.items() if len(v) > 1}
    if leaks:
        raise AssertionError(f"Split-group leakage across splits: {leaks}")
    print("assert_no_subject_leakage: OK (split groups disjoint)")


def _load_floor_sidecar(cache_dir: Path) -> dict[str, Any]:
    path = _floor_path(cache_dir)
    if not path.is_file():
        return {"version": 1, "entries": {}}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _save_floor_sidecar(cache_dir: Path, data: dict[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    with _floor_path(cache_dir).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _subsample_frame_indices(n_frames: int, every: int = 10) -> np.ndarray:
    idx = np.arange(0, n_frames, every, dtype=int)
    if idx.size == 0 or idx[-1] != n_frames - 1:
        idx = np.unique(np.append(idx, n_frames - 1))
    return idx


def _compute_floor_offset_mesh(
    root_orient: np.ndarray,
    pose_body: np.ndarray,
    transl: np.ndarray,
    betas: np.ndarray,
    fps: float,
) -> tuple[float, dict[str, float]]:
    """Floor height from locked_head mesh (1st percentile of per-frame min vertex z).

    Uses every 10th frame (plus last) to limit cost. The 1st percentile of
    per-frame minimum vertex ``z`` is more robust than a global minimum (single
    foot penetration / marker glitch) while still tracking the support surface.
    """
    import torch
    from hready.body.smplx_wrapper import load_body

    body = load_body("locked_head")
    device = torch.device("cpu")
    dtype = torch.float32
    idx = _subsample_frame_indices(root_orient.shape[0], every=10)
    mins: list[float] = []
    pelvis_z: list[float] = []
    head_above: list[float] = []
    for i in idx:
        go = torch.as_tensor(root_orient[i], device=device, dtype=dtype).unsqueeze(0)
        bp = torch.as_tensor(pose_body[i], device=device, dtype=dtype).reshape(1, 63)
        tr = torch.as_tensor(transl[i], device=device, dtype=dtype).unsqueeze(0)
        be = torch.as_tensor(betas, device=device, dtype=dtype).unsqueeze(0)
        out = body.forward(go, bp, be, tr)
        vz = out.vertices[0, :, 2].detach().cpu().numpy()
        mins.append(float(vz.min()))
        pelvis_z.append(float(out.joints[0, 0, 2].item()))
        head_z = float(out.joints[0, 15, 2].item())
        head_above.append(1.0 if head_z > pelvis_z[-1] else 0.0)
    floor = float(np.percentile(mins, 1))
    stats = {
        "pelvis_z_median": float(np.median(pelvis_z)),
        "head_above_pelvis_frac": float(np.mean(head_above)),
        "min_vertex_z_subsample_min": float(np.min(mins)),
    }
    return floor, stats


def floor_offset(
    entry: Union[AmassIndexEntry, str],
    *,
    amass_root: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    recompute: bool = False,
) -> dict[str, Any]:
    """Per-clip floor height (meters) cached in ``amass_floor.json`` sidecar."""
    amass_root = (amass_root or amass_root_from_config()).resolve()
    cache_dir = (cache_dir or cache_dir_from_config()).resolve()
    if isinstance(entry, str):
        entry = _entry_by_rel(load_index(cache_dir), entry)
    sidecar = _load_floor_sidecar(cache_dir)
    cached = sidecar["entries"].get(entry.rel_path)
    if cached is not None and not recompute:
        return cached

    raw = _load_npz_raw(amass_root / entry.rel_path)
    floor, stats = _compute_floor_offset_mesh(
        raw["root_orient"],
        raw["pose_body"].reshape(-1, 21, 3),
        raw["trans"],
        raw["betas"],
        entry.fps,
    )
    record = {
        "floor_offset": floor,
        "method": "p1_per_frame_min_vertex_z_subsample10",
        "outlier": False,
        **stats,
    }
    sidecar["entries"][entry.rel_path] = record
    _save_floor_sidecar(cache_dir, sidecar)
    return record


def flag_floor_outliers(
    cache_dir: Optional[Path] = None,
    *,
    z_thresh: float = 0.35,
) -> list[str]:
    """Mark clips whose floor offset is far from the global median (stairs, lying, etc.)."""
    cache_dir = cache_dir or cache_dir_from_config()
    sidecar = _load_floor_sidecar(cache_dir)
    floors = [v["floor_offset"] for v in sidecar["entries"].values()]
    if not floors:
        return []
    med = float(np.median(floors))
    flagged: list[str] = []
    for rel, rec in sidecar["entries"].items():
        out = abs(rec["floor_offset"] - med) > z_thresh
        rec["outlier"] = out
        if out:
            flagged.append(rel)
    _save_floor_sidecar(cache_dir, sidecar)
    return flagged


def _load_npz_raw(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        root_orient = np.asarray(data["root_orient"], dtype=np.float64)
        pose_body = np.asarray(data["pose_body"], dtype=np.float64)
        trans = np.asarray(data["trans"], dtype=np.float64)
        betas = np.asarray(data["betas"], dtype=np.float64).reshape(-1)[:16]
    return {
        "root_orient": root_orient,
        "pose_body": pose_body,
        "trans": trans,
        "betas": betas,
    }


def _target_frame_count(n_src: int, fps_src: float, fps_tgt: float) -> int:
    """Output frames so ``t_out[k]=k/fps_tgt`` never exceeds source span ``(n_src-1)/fps_src``."""
    if n_src <= 1:
        return 1
    span_tgt = (n_src - 1) * fps_tgt / fps_src
    return max(1, int(math.floor(span_tgt)) + 1)


def _integer_downsample_stride(fps_src: float, fps_tgt: float) -> Optional[int]:
    if fps_src < fps_tgt - 1e-9:
        return None
    ratio = fps_src / fps_tgt
    r_int = int(round(ratio))
    if abs(ratio - r_int) > 1e-9 or r_int < 1:
        return None
    return r_int


def _downsample_indices(n_src: int, stride: int) -> np.ndarray:
    return np.arange(0, n_src, stride, dtype=int)


def _source_times(n_src: int, fps_src: float) -> np.ndarray:
    if n_src <= 1:
        return np.zeros(1, dtype=np.float64)
    return np.arange(n_src, dtype=np.float64) / fps_src


def _target_times(n_tgt: int, fps_tgt: float) -> np.ndarray:
    if n_tgt <= 1:
        return np.zeros(1, dtype=np.float64)
    return np.arange(n_tgt, dtype=np.float64) / fps_tgt


def _resample_translation(trans: np.ndarray, fps_src: float, fps_tgt: float) -> np.ndarray:
    n_src = trans.shape[0]
    n_tgt = _target_frame_count(n_src, fps_src, fps_tgt)
    if n_src == n_tgt and abs(fps_src - fps_tgt) < 1e-6:
        return trans.copy()
    stride = _integer_downsample_stride(fps_src, fps_tgt)
    if stride is not None:
        return trans[_downsample_indices(n_src, stride)]
    t_src = _source_times(n_src, fps_src)
    t_tgt = _target_times(n_tgt, fps_tgt)
    out = np.empty((n_tgt, 3), dtype=np.float64)
    for d in range(3):
        out[:, d] = np.interp(t_tgt, t_src, trans[:, d])
    return out


def _resample_axis_angle_series(
    aa: np.ndarray, fps_src: float, fps_tgt: float
) -> np.ndarray:
    """``aa`` shaped ``(T, J, 3)`` -> resampled ``(T', J, 3)`` via quaternion slerp."""
    import torch
    from hready.body.rotations import (
        axis_angle_to_quaternion,
        quaternion_to_axis_angle,
    )

    n_src, j, _ = aa.shape
    n_tgt = _target_frame_count(n_src, fps_src, fps_tgt)
    if n_src == n_tgt and abs(fps_src - fps_tgt) < 1e-6:
        return aa.copy()
    stride = _integer_downsample_stride(fps_src, fps_tgt)
    if stride is not None:
        return aa[_downsample_indices(n_src, stride)]

    t_src = _source_times(n_src, fps_src)
    t_tgt = _target_times(n_tgt, fps_tgt)
    device = torch.device("cpu")
    dtype = torch.float64
    q_src = axis_angle_to_quaternion(torch.as_tensor(aa, device=device, dtype=dtype))
    q_src = q_src.reshape(n_src, j, 4)
    out = np.empty((n_tgt, j, 3), dtype=np.float64)
    for ji in range(j):
        qs = _quaternion_hemisphere_continuous(q_src[:, ji, :])
        q_tgt = _slerp_quaternion_series(qs, t_src, t_tgt)
        q_tgt = _quaternion_hemisphere_continuous(q_tgt)
        q_tgt = q_tgt / q_tgt.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        aa_tgt = quaternion_to_axis_angle(q_tgt)
        out[:, ji, :] = aa_tgt.detach().cpu().numpy()
    return out


def _quaternion_hemisphere_continuous(q: Any) -> Any:
    """Flip successive quaternions so dot products are non-negative (short arc)."""
    import torch

    out = q.clone()
    for i in range(1, out.shape[0]):
        if torch.dot(out[i - 1], out[i]) < 0.0:
            out[i] = -out[i]
    return out


def _slerp_quaternion_series(
    q_src: Any, t_src: np.ndarray, t_tgt: np.ndarray
) -> Any:
    import torch

    out = []
    for tt in t_tgt:
        if tt <= t_src[0]:
            out.append(q_src[0])
            continue
        if tt >= t_src[-1]:
            out.append(q_src[-1])
            continue
        k = int(np.searchsorted(t_src, tt, side="right") - 1)
        k = min(max(k, 0), len(t_src) - 2)
        t0, t1 = t_src[k], t_src[k + 1]
        if t1 <= t0:
            out.append(q_src[k])
            continue
        alpha = float((tt - t0) / (t1 - t0))
        if alpha <= 0.0:
            out.append(q_src[k])
        elif alpha >= 1.0:
            out.append(q_src[k + 1])
        else:
            out.append(_slerp(q_src[k], q_src[k + 1], alpha))
    return torch.stack(out, dim=0)


def _slerp(q0: Any, q1: Any, t: float) -> Any:
    import torch

    q0 = q0 / q0.norm().clamp_min(1e-12)
    q1 = q1 / q1.norm().clamp_min(1e-12)
    dot = torch.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = dot.clamp(-1.0, 1.0)
    if dot > 0.999999999:
        out = q0 + t * (q1 - q0)
        return out / out.norm().clamp_min(1e-12)
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    s0 = torch.sin((1.0 - t) * theta_0) / sin_theta_0
    s1 = torch.sin(t * theta_0) / sin_theta_0
    return s0 * q0 + s1 * q1


def load_clip(
    entry: Union[AmassIndexEntry, str, dict[str, Any]],
    *,
    target_fps: float = _TARGET_FPS,
    ground: bool = False,
    include_hands: bool = False,
    amass_root: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Load one AMASS clip resampled to ``target_fps`` (default 30)."""
    amass_root = (amass_root or amass_root_from_config()).resolve()
    cache_dir = cache_dir or cache_dir_from_config()
    if isinstance(entry, dict):
        entry = AmassIndexEntry(**entry)
    elif isinstance(entry, str):
        entry = _entry_by_rel(load_index(cache_dir), entry)

    raw = _load_npz_raw(amass_root / entry.rel_path)
    fps_src = float(entry.fps)
    root = _resample_axis_angle_series(
        raw["root_orient"].reshape(-1, 1, 3), fps_src, target_fps
    ).reshape(-1, 3)
    body = _resample_axis_angle_series(
        raw["pose_body"].reshape(-1, 21, 3), fps_src, target_fps
    )
    transl = _resample_translation(raw["trans"], fps_src, target_fps)

    out: dict[str, Any] = {
        "root_orient": root,
        "pose_body": body,
        "transl": transl,
        "betas": raw["betas"],
        "fps": target_fps,
        "source_fps": fps_src,
        "rel_path": entry.rel_path,
        "subset": entry.subset,
        "subject": entry.subject,
        "clip_id": entry.clip_id,
    }
    if include_hands:
        with np.load(amass_root / entry.rel_path, allow_pickle=True) as data:
            if "pose_hand" in data:
                ph = np.asarray(data["pose_hand"], dtype=np.float64)
                out["pose_hand"] = _resample_axis_angle_series(
                    ph.reshape(-1, 30, 3), fps_src, target_fps
                )

    if ground:
        floor_rec = floor_offset(entry, amass_root=amass_root, cache_dir=cache_dir)
        out["floor_offset"] = float(floor_rec["floor_offset"])
        out["floor_outlier"] = bool(floor_rec.get("outlier", False))
        out["transl"] = transl.copy()
        out["transl"][:, 2] -= out["floor_offset"]
    return out


def duration_seconds(n_frames: int, fps: float) -> float:
    if n_frames <= 1:
        return 0.0
    return (n_frames - 1) / fps


def duration_error(n_src: int, fps_src: float, n_tgt: int, fps_tgt: float) -> float:
    return abs(duration_seconds(n_src, fps_src) - duration_seconds(n_tgt, fps_tgt))


def source_frame_index_at_output_k(
    k: int, n_src: int, fps_src: float, fps_tgt: float = _TARGET_FPS
) -> int:
    """Source frame index at output time ``k / fps_tgt`` (nearest integer frame)."""
    if n_src <= 1:
        return 0
    j = (k / fps_tgt) * fps_src
    idx = int(round(j))
    return min(max(idx, 0), n_src - 1)


def _quaternion_geodesic_rad(q0: Any, q1: Any) -> float:
    import torch

    q0 = q0 / q0.norm().clamp_min(1e-12)
    q1 = q1 / q1.norm().clamp_min(1e-12)
    dot = torch.dot(q0, q1).abs().clamp(0.0, 1.0)
    return float(2.0 * torch.acos(dot))


def resampling_alignment_errors(
    entry: Union[AmassIndexEntry, str],
    *,
    amass_root: Optional[Path] = None,
    target_fps: float = _TARGET_FPS,
    check_ks: Optional[list[int]] = None,
) -> dict[str, Any]:
    """Max rotation (geodesic rad) / translation error vs source at mapped frames."""
    import torch
    from hready.body.rotations import axis_angle_to_quaternion

    amass_root = (amass_root or amass_root_from_config()).resolve()
    if isinstance(entry, str):
        entry = _entry_by_rel(load_index(), entry)
    raw = _load_npz_raw(amass_root / entry.rel_path)
    fps_src = float(entry.fps)
    n_src = raw["root_orient"].shape[0]
    loaded = load_clip(entry, target_fps=target_fps, amass_root=amass_root)
    n_tgt = loaded["root_orient"].shape[0]
    if check_ks is None:
        stride = _integer_downsample_stride(fps_src, target_fps)
        if stride is not None:
            check_ks = list(range(0, n_tgt))
        else:
            check_ks = list(range(0, n_tgt, 3))

    rot_max = 0.0
    trans_max = 0.0
    mappings: list[tuple[int, int]] = []
    for k in check_ks:
        if k < 0 or k >= n_tgt:
            continue
        j = source_frame_index_at_output_k(k, n_src, fps_src, target_fps)
        mappings.append((k, j))
        aa_out = np.concatenate(
            [
                loaded["root_orient"][k : k + 1].reshape(1, 1, 3),
                loaded["pose_body"][k : k + 1],
            ],
            axis=1,
        )
        aa_src = np.concatenate(
            [
                raw["root_orient"][j : j + 1].reshape(1, 1, 3),
                raw["pose_body"].reshape(-1, 21, 3)[j : j + 1],
            ],
            axis=1,
        )
        q_out = axis_angle_to_quaternion(torch.as_tensor(aa_out, dtype=torch.float64))
        q_src = axis_angle_to_quaternion(torch.as_tensor(aa_src, dtype=torch.float64))
        for ji in range(22):
            ang = _quaternion_geodesic_rad(q_out[0, ji], q_src[0, ji])
            rot_max = max(rot_max, ang)
        trans_max = max(
            trans_max,
            float(np.linalg.norm(loaded["transl"][k] - raw["trans"][j])),
        )
    return {
        "rel_path": entry.rel_path,
        "fps_src": fps_src,
        "n_src": n_src,
        "n_tgt": n_tgt,
        "integer_stride": _integer_downsample_stride(fps_src, target_fps),
        "max_rot_geodesic_rad": rot_max,
        "max_trans_l2": trans_max,
        "sample_mappings": mappings[:5],
    }
