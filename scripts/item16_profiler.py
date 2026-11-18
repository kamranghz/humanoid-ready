"""
Item 16 — DataLoader bottleneck profiler (numpy-only, no torch required).

Simulates HRRefineWindowDataset.__getitem__:
  open .npz -> read pose arrays -> random window slice

Benchmarks:
  - serial (num_workers=0 analog)  
  - parallel using multiprocessing.Pool (num_workers=4 analog)
"""
import sys, os, json, time, pathlib, multiprocessing as mp
import numpy as np

MOUNT_BASE = pathlib.Path("/sessions/rcw-01donjnzzbkiafjwlqubec7s/mnt")
AMASS_ROOT = MOUNT_BASE / "datasets" / "amass" / "smplx_n"
WINDOW = 64
BATCH_SIZE = 8
N_BATCHES = 50
SEED = 0


def collect_clips(amass_root: pathlib.Path, max_clips: int = 400) -> list[str]:
    out = []
    for subset in sorted(amass_root.iterdir()):
        if not subset.is_dir():
            continue
        for subject in subset.iterdir():
            if not subject.is_dir():
                continue
            for f in sorted(subject.iterdir()):
                if f.suffix == ".npz" and "shape" not in f.name and not f.stem.endswith("stagei"):
                    out.append(str(f))
                    if len(out) >= max_clips:
                        return out
    return out


def load_window(path: str):
    """Simulate HRRefineWindowDataset.__getitem__ on one clip."""
    data = np.load(path, allow_pickle=False)
    # AMASS smplx_n keys: root_orient/pose_body (smplx_n) or poses (older format)
    if "root_orient" not in data and "poses" not in data:
        return None  # shape-only or marker-only file, skip
    if "root_orient" in data:
        root = data["root_orient"]
        pose = data["pose_body"]
    else:
        poses = data["poses"]  # (T, 165) — first 3 = root, next 63 = body
        root = poses[:, :3]
        pose = poses[:, 3:66]
    transl = data["trans"] if "trans" in data else np.zeros((root.shape[0], 3))
    t = root.shape[0]
    if t >= WINDOW:
        s = int(np.random.randint(0, t - WINDOW + 1))
        return {
            "root": np.array(root[s:s + WINDOW], dtype=np.float32),
            "pose": np.array(pose[s:s + WINDOW, :63], dtype=np.float32),
            "transl": np.array(transl[s:s + WINDOW], dtype=np.float32),
        }
    else:
        pad = WINDOW - t
        return {
            "root": np.concatenate([root, np.repeat(root[-1:], pad, 0)], 0).astype(np.float32),
            "pose": np.concatenate([pose[:, :63], np.repeat(pose[-1:, :63], pad, 0)], 0).astype(np.float32),
            "transl": np.concatenate([transl, np.repeat(transl[-1:], pad, 0)], 0).astype(np.float32),
        }


def _worker_load(path: str) -> dict:
    return load_window(path)


def benchmark_serial(clips: list[str], n_batches: int) -> dict:
    rng = np.random.default_rng(SEED)
    # warmup
    for _ in range(2):
        batch_paths = [clips[int(rng.integers(len(clips)))] for _ in range(BATCH_SIZE)]
        for bp in batch_paths:
            r = load_window(bp)
            if r is None: continue

    t0 = time.perf_counter()
    for _ in range(n_batches):
        batch_paths = [clips[int(rng.integers(len(clips)))] for _ in range(BATCH_SIZE)]
        for bp in batch_paths:
            r = load_window(bp)
            if r is None: continue
    dt = time.perf_counter() - t0
    n = n_batches * BATCH_SIZE
    return {
        "mode": "serial_num_workers_0",
        "num_workers": 0,
        "n_batches": n_batches,
        "wall_s": round(dt, 3),
        "batches_per_s": round(n_batches / dt, 2),
        "samples_per_s": round(n / dt, 1),
        "s_per_batch": round(dt / n_batches, 4),
        "ms_per_sample": round(dt / n * 1000, 2),
    }


def benchmark_parallel(clips: list[str], n_workers: int, n_batches: int) -> dict:
    rng = np.random.default_rng(SEED + 1)
    ctx = mp.get_context("spawn")

    with ctx.Pool(n_workers) as pool:
        # warmup
        for _ in range(2):
            batch_paths = [clips[int(rng.integers(len(clips)))] for _ in range(BATCH_SIZE)]
            list(pool.map(_worker_load, batch_paths))

        t0 = time.perf_counter()
        for _ in range(n_batches):
            batch_paths = [clips[int(rng.integers(len(clips)))] for _ in range(BATCH_SIZE)]
            list(pool.map(_worker_load, batch_paths))
        dt = time.perf_counter() - t0

    n = n_batches * BATCH_SIZE
    return {
        "mode": f"parallel_num_workers_{n_workers}",
        "num_workers": n_workers,
        "n_batches": n_batches,
        "wall_s": round(dt, 3),
        "batches_per_s": round(n_batches / dt, 2),
        "samples_per_s": round(n / dt, 1),
        "s_per_batch": round(dt / n_batches, 4),
        "ms_per_sample": round(dt / n * 1000, 2),
    }


def main():
    out_dir = pathlib.Path("results/data_pipeline")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Collecting clips...", flush=True)
    clips = collect_clips(AMASS_ROOT, max_clips=400)
    print(f"  {len(clips)} clips", flush=True)

    # Quick per-file timing
    t0 = time.perf_counter()
    for cp in clips[:20]:
        load_window(cp)
    dt20 = (time.perf_counter() - t0) / 20
    print(f"  avg ms/file (20 samples): {dt20*1000:.1f}ms", flush=True)
    print(f"  serial batch time estimate: {dt20*BATCH_SIZE*1000:.0f}ms/batch", flush=True)

    print("\n--- BASELINE: serial (num_workers=0) ---", flush=True)
    r0 = benchmark_serial(clips, n_batches=N_BATCHES)
    print(f"  {r0['batches_per_s']:.2f} batches/s  "
          f"{r0['samples_per_s']:.1f} samples/s  "
          f"{r0['s_per_batch']*1000:.0f}ms/batch  "
          f"{r0['ms_per_sample']:.1f}ms/sample", flush=True)

    print("\n--- AFTER FIX: parallel (num_workers=4, spawn) ---", flush=True)
    r4 = benchmark_parallel(clips, n_workers=4, n_batches=N_BATCHES)
    print(f"  {r4['batches_per_s']:.2f} batches/s  "
          f"{r4['samples_per_s']:.1f} samples/s  "
          f"{r4['s_per_batch']*1000:.0f}ms/batch  "
          f"{r4['ms_per_sample']:.1f}ms/sample", flush=True)

    speedup = r4["batches_per_s"] / max(r0["batches_per_s"], 1e-9)
    print(f"\nSpeedup: {speedup:.2f}x", flush=True)

    r0["ms_per_sample"] = round(1000 / max(r0["samples_per_s"], 1e-9), 2)
    r4["ms_per_sample"] = round(1000 / max(r4["samples_per_s"], 1e-9), 2)

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "batch_size": BATCH_SIZE,
        "window_frames": WINDOW,
        "n_clips_measured": len(clips),
        "n_batches_timed": N_BATCHES,
        "ms_per_file_cold": round(dt20 * 1000, 1),
        "baseline": r0,
        "fixed": r4,
        "speedup_x": round(speedup, 2),
    }

    (out_dir / "profile_before.json").write_text(json.dumps(r0, indent=2))
    (out_dir / "profile_after.json").write_text(json.dumps(r4, indent=2))
    (out_dir / "profile_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nResults written to {out_dir}/", flush=True)
    return summary


if __name__ == "__main__":
    main()
