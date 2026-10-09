"""30 Hz grounded motion cache for E3 (uncompressed ``.npy`` per array, read with ``mmap_mode="r"``).

One directory per clip under ``<cache_dir>/<rel_dir>/clips/``:
``root_orient`` (T,3), ``pose_body`` (T,63), ``transl`` (T,3, grounded), ``betas`` (16,),
``joints_22`` (T,22,3), ``joints_55`` (T,55,3) from locked_head FK, ``contact`` (T,4) bool
(item-5 foot-channel labels: L heel, L toe, R heel, R toe), and ``meta.json`` (written last).
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hready.body.batch_forward import smpl_forward_bt
from hready.body.smplx_wrapper import load_body
from hready.data.amass import (
    _entry_by_rel,
    amass_root_from_config,
    cache_dir_from_config,
    load_clip,
    load_index,
)
from hready.data.contact import compute_foot_contact_from_positions, load_foot_positions

FPS = 30.0
NUM_BODY = 21
NUM_KP = 22
FK_CHUNK = 512
ARRAYS = (
    "root_orient",
    "pose_body",
    "transl",
    "betas",
    "joints_22",
    "joints_55",
    "contact",
)
FORMAT = "e3_npy_mmap_v3"

_WORKER: dict[str, Any] = {}


def _safe_name(rel_path: str) -> str:
    return rel_path.replace("/", "__").replace("\\", "__")


def memmap_root(cache_dir: Path | None, rel_dir: str) -> Path:
    return ((cache_dir or cache_dir_from_config()) / rel_dir).resolve()


def clip_dir(root: Path, rel_path: str) -> Path:
    return root / "clips" / _safe_name(rel_path)


def fk_joints_55(body: Any, clip: dict[str, Any]) -> np.ndarray:
    """Locked-head FK joints ``(T, 55, 3)`` float32; fixed chunking so rebuilds are byte-equal."""
    root = torch.as_tensor(np.asarray(clip["root_orient"], dtype=np.float32))
    pose = torch.as_tensor(np.asarray(clip["pose_body"], dtype=np.float32)).reshape(
        -1, NUM_BODY, 3
    )
    transl = torch.as_tensor(np.asarray(clip["transl"], dtype=np.float32))
    betas = torch.as_tensor(
        np.asarray(clip["betas"], dtype=np.float32).reshape(-1)[:16]
    )
    out = []
    with torch.no_grad():
        for i0 in range(0, root.shape[0], FK_CHUNK):
            i1 = min(root.shape[0], i0 + FK_CHUNK)
            j55, _ = smpl_forward_bt(
                body,
                transl[i0:i1].unsqueeze(0),
                root[i0:i1].unsqueeze(0),
                pose[i0:i1].unsqueeze(0),
                betas.unsqueeze(0),
            )
            out.append(j55[0].numpy().astype(np.float32))
    return np.concatenate(out, axis=0)


def compute_clip_arrays(
    rel_path: str, *, body: Any, cache_dir: Path, amass_root: Path
) -> dict[str, Any]:
    """Reference arrays for one clip (used by both build and verify)."""
    entry = _entry_by_rel(load_index(cache_dir), rel_path)
    clip = load_clip(entry, ground=True, amass_root=amass_root, cache_dir=cache_dir)
    j55 = fk_joints_55(body, clip)
    t_len = j55.shape[0]
    # Item-5 contact labels: grounded vertex foot channels (foot_traj cache), recorded hysteresis.
    foot = load_foot_positions(entry, cache_dir=cache_dir)
    if foot.shape[0] != t_len:
        raise ValueError(f"{rel_path}: foot_traj T={foot.shape[0]} != clip T={t_len}")
    return {
        "root_orient": np.asarray(clip["root_orient"], dtype=np.float32),
        "pose_body": np.asarray(clip["pose_body"], dtype=np.float32).reshape(
            t_len, NUM_BODY * 3
        ),
        "transl": np.asarray(clip["transl"], dtype=np.float32),
        "betas": np.asarray(clip["betas"], dtype=np.float32).reshape(-1)[:16].copy(),
        "joints_22": np.ascontiguousarray(j55[:, :NUM_KP]),
        "joints_55": j55,
        "contact": compute_foot_contact_from_positions(foot).astype(np.bool_),
        "floor_offset": float(clip.get("floor_offset", 0.0)),
        "fps": float(clip.get("fps", FPS)),
        "T": t_len,
    }


def _init_worker(cache_dir_s: str, amass_root_s: str, out_root_s: str) -> None:
    torch.set_num_threads(1)
    _WORKER["body"] = load_body("locked_head")
    _WORKER["cache_dir"] = Path(cache_dir_s)
    _WORKER["amass_root"] = Path(amass_root_s)
    _WORKER["out_root"] = Path(out_root_s)


def _build_one(rel_path: str) -> dict[str, Any]:
    out_root = _WORKER["out_root"]
    cdir = clip_dir(out_root, rel_path)
    meta_path = cdir / "meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return {**meta, "skipped": True}
    try:
        rec = compute_clip_arrays(
            rel_path,
            body=_WORKER["body"],
            cache_dir=_WORKER["cache_dir"],
            amass_root=_WORKER["amass_root"],
        )
    except Exception as exc:  # noqa: BLE001 - recorded in the build index, never silently dropped
        return {"rel_path": rel_path, "error": f"{type(exc).__name__}: {exc}"}
    cdir.mkdir(parents=True, exist_ok=True)
    for name in ARRAYS:
        np.save(cdir / f"{name}.npy", rec[name])
    meta = {
        "rel_path": rel_path,
        "T": int(rec["T"]),
        "fps": rec["fps"],
        "floor_offset": rec["floor_offset"],
        "format": FORMAT,
    }
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return {**meta, "skipped": False}


def build_memmap(
    rel_paths: list[str],
    *,
    rel_dir: str,
    workers: int,
    cache_dir: Path | None = None,
    amass_root: Path | None = None,
    log_every: int = 200,
) -> dict[str, Any]:
    """Build (or resume) the cache for ``rel_paths``; writes ``index.json``."""
    cache_dir = (cache_dir or cache_dir_from_config()).resolve()
    amass_root = (amass_root or amass_root_from_config()).resolve()
    out_root = memmap_root(cache_dir, rel_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    init_args = (str(cache_dir), str(amass_root), str(out_root))
    t0 = time.perf_counter()
    metas: list[dict[str, Any]] = []
    n = len(rel_paths)

    def _log(i: int) -> None:
        if (i + 1) % log_every == 0 or i + 1 == n:
            el = time.perf_counter() - t0
            eta = el / (i + 1) * (n - i - 1)
            print(
                f"build-cache {i + 1}/{n} elapsed_s={el:.0f} eta_s={eta:.0f}",
                flush=True,
            )

    if workers <= 1:
        _init_worker(*init_args)
        for i, rel in enumerate(rel_paths):
            metas.append(_build_one(rel))
            _log(i)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, initializer=_init_worker, initargs=init_args) as pool:
            for i, meta in enumerate(
                pool.imap_unordered(_build_one, rel_paths, chunksize=4)
            ):
                metas.append(meta)
                _log(i)
    errors = sorted((m["rel_path"], m["error"]) for m in metas if "error" in m)
    ok = sorted((m for m in metas if "error" not in m), key=lambda m: m["rel_path"])
    payload = {
        "format": FORMAT,
        "fps": FPS,
        "n_requested": n,
        "n_clips": len(ok),
        "n_built_now": sum(1 for m in ok if not m["skipped"]),
        "n_errors": len(errors),
        "errors": [f"{r}: {e}" for r, e in errors],
        "workers": workers,
        "wall_s": time.perf_counter() - t0,
        "entries": [{"rel_path": m["rel_path"], "T": m["T"]} for m in ok],
    }
    (out_root / "index.json").write_text(
        json.dumps(payload, indent=1), encoding="utf-8"
    )
    return payload


def load_cache_index(rel_dir: str, cache_dir: Path | None = None) -> dict[str, int]:
    """``rel_path -> T`` from ``index.json``."""
    path = memmap_root(cache_dir, rel_dir) / "index.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return {e["rel_path"]: int(e["T"]) for e in data["entries"]}


def load_clip_memmap(
    rel_path: str, *, rel_dir: str, cache_dir: Path | None = None, mmap: bool = True
) -> dict[str, Any]:
    cdir = clip_dir(memmap_root(cache_dir, rel_dir), rel_path)
    mode = "r" if mmap else None
    out: dict[str, Any] = {
        name: np.load(cdir / f"{name}.npy", mmap_mode=mode) for name in ARRAYS
    }
    meta = json.loads((cdir / "meta.json").read_text(encoding="utf-8"))
    out.update(
        rel_path=rel_path,
        T=int(meta["T"]),
        fps=meta["fps"],
        floor_offset=meta["floor_offset"],
    )
    return out


def cache_disk_bytes(
    rel_paths: list[str], *, rel_dir: str, cache_dir: Path | None = None
) -> int:
    root = memmap_root(cache_dir, rel_dir)
    total = 0
    for rel in rel_paths:
        cdir = clip_dir(root, rel)
        total += sum(
            (cdir / f"{n}.npy").stat().st_size
            for n in ARRAYS
            if (cdir / f"{n}.npy").is_file()
        )
    return total


def measure_window_reads(
    rel_paths: list[str],
    *,
    rel_dir: str,
    window: int,
    n_reads: int = 500,
    seed: int = 0,
) -> dict[str, float]:
    """Random 64-frame window reads (joints_55 + pose) through the memmap, cold-open per read."""
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    for _ in range(n_reads):
        rel = rel_paths[int(rng.integers(len(rel_paths)))]
        clip = load_clip_memmap(rel, rel_dir=rel_dir)
        s = int(rng.integers(0, max(1, clip["T"] - window)))
        _ = (
            np.array(clip["joints_55"][s : s + window]),
            np.array(clip["pose_body"][s : s + window]),
        )
    dt = time.perf_counter() - t0
    return {
        "n_window_reads": n_reads,
        "wall_s": dt,
        "windows_per_s": n_reads / max(dt, 1e-9),
    }


def verify_memmap(
    rel_paths: list[str],
    *,
    rel_dir: str,
    n_sample: int,
    seed: int,
    cache_dir: Path | None = None,
    amass_root: Path | None = None,
) -> dict[str, Any]:
    """Byte-equality of every cached array vs ``load_clip`` + FK + item-5 contact on a seeded sample."""
    cache_dir = (cache_dir or cache_dir_from_config()).resolve()
    amass_root = (amass_root or amass_root_from_config()).resolve()
    rng = np.random.default_rng(seed)
    pick = sorted(
        rng.choice(len(rel_paths), size=min(n_sample, len(rel_paths)), replace=False)
    )
    sample = [rel_paths[i] for i in pick]
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)  # same as build workers
    body = load_body("locked_head")
    failures: list[str] = []
    not_memmap: list[str] = []
    fps_bad: list[str] = []
    try:
        for rel in sample:
            mm = load_clip_memmap(rel, rel_dir=rel_dir, cache_dir=cache_dir, mmap=True)
            ref = compute_clip_arrays(
                rel, body=body, cache_dir=cache_dir, amass_root=amass_root
            )
            for name in ARRAYS:
                arr = mm[name]
                if not isinstance(arr, np.memmap):
                    not_memmap.append(f"{rel}:{name}")
                if arr.dtype != ref[name].dtype or not np.array_equal(
                    np.asarray(arr), ref[name]
                ):
                    failures.append(f"{rel}:{name}")
            if mm["fps"] != FPS or ref["fps"] != FPS:
                fps_bad.append(rel)
    finally:
        torch.set_num_threads(prev_threads)
    return {
        "n_checked": len(sample),
        "arrays": list(ARRAYS),
        "byte_equal": not failures,
        "all_np_memmap": not not_memmap,
        "fps_30": not fps_bad,
        "pass": not failures and not not_memmap and not fps_bad,
        "failures": failures[:20],
        "not_memmap": not_memmap[:5],
        "fps_bad": fps_bad[:5],
    }
