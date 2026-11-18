# Data Pipeline Bottleneck — Item 16

**Date:** 2026-11-18  
**Profiler:** `scripts/item16_profiler.py`  
**Profiling environment:** Linux VM, AMASS smplx_n on network-mounted NTFS volume  
**Setup:** 400 AMASS clips, batch_size=8, window=64 frames, 50 measured batches after warmup

---

## Root Cause

`_make_loader` in `hready/train/hr_refine.py` hardcoded `num_workers=0`.  
Every `__getitem__` call opens `.npy` memmap files synchronously on the main process.  
On a network-mounted drive each file read takes **~65 ms**, so a batch of 8 samples costs **~520 ms** of pure I/O — the GPU sits idle while the CPU fetches one sample at a time.

```python
# BEFORE (bottleneck)
return DataLoader(ds, batch_size=8, num_workers=0, ...)
```

---

## Profiling Results

| Mode | num_workers | Batches/s | Samples/s | ms/batch | ms/sample |
|------|-------------|-----------|-----------|----------|-----------|
| Baseline (serial) | 0 | 1.76 | 14.1 | 568 | 71.0 |
| Fixed (parallel, spawn) | 4 | **3.54** | **28.3** | 283 | 35.3 |

**Speedup: 2.01× on network-mounted files** (I/O-bound; GPU-local NVMe would show higher gains).

The parallel benchmark uses `multiprocessing.Pool(4, context="spawn")` with persistent workers — matching the `persistent_workers=True` behavior of PyTorch DataLoader.

---

## Fix Applied

### `hready/train/hr_refine.py`

`_make_loader` now accepts `num_workers` from config and sets `multiprocessing_context="spawn"` and `persistent_workers=True` when `num_workers > 0`:

```python
def _make_loader(split, batch_size, window, seed, num_workers=0):
    ds = HRRefineWindowDataset(split=split, window=window, seed=seed)
    kwargs = {}
    if num_workers > 0:
        kwargs["multiprocessing_context"] = "spawn"   # Windows-safe
        kwargs["persistent_workers"] = True            # avoids re-fork overhead
    return DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_windows,
        drop_last=True, **kwargs,
    )
```

`train_main` reads `num_workers` from the YAML config:

```python
num_workers = int(cfg.get("num_workers", 4))
loader = _make_loader("train", batch_size, window, base_seed, num_workers=num_workers)
```

### `configs/hr_refine.yaml`

```yaml
num_workers: 4  # parallel DataLoader workers; 0 = serial (bottleneck)
```

---

## Why `spawn` + `persistent_workers`

| Setting | Reason |
|---------|--------|
| `multiprocessing_context="spawn"` | Required on Windows (no `fork`). Safe and correct on Linux. |
| `persistent_workers=True` | Keeps worker processes alive between epochs. Avoids re-import and re-init overhead on each epoch boundary. Only valid when `num_workers > 0`. |

---

## Files Changed

| File | Change |
|------|--------|
| `hready/train/hr_refine.py` | `_make_loader` accepts `num_workers`; `train_main` reads it from config |
| `configs/hr_refine.yaml` | Added `num_workers: 4` |

## Measurement Artifacts

- `results/data_pipeline/profile_before.json` — serial baseline metrics
- `results/data_pipeline/profile_after.json` — parallel (num_workers=4) metrics
- `results/data_pipeline/profile_summary.json` — combined summary with speedup
