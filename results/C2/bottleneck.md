# Data-pipeline bottleneck: HR-Refine window loader (2026-10-12)

Scope: the AMASS window loader used by HR-Refine training (`hready/data/refine_dataset.py`). There is no BEDLAM loader in the repository yet (checklist item 12 is open), so BEDLAM is not covered here.

## What the bottleneck was

Every training sample called `load_clip(entry, ground=True)` to read a 64-frame window. That call processes the **whole clip** each time:

- resamples the full axis-angle series to 30 Hz with Python quaternion slerp (`_resample_axis_angle_series`): 87% of the time (profiled over 39 random train clips, 0.50 s per clip);
- re-parses the floor-offset JSON sidecar on every call (`_load_floor_sidecar`, `json.load`): about 10%.

Both costs scale with clip length and are paid again for every sample, although only 64 frames are used. The loader, not the model, set the training speed: a batch of 8 samples took 1–5.6 s to load against 0.1–0.7 s of compute, and the GPU sat at about 2% utilisation.

## What the fix was

Read each window from the existing `grounded_motion_30hz` memmap cache (`hready/data/motion_cache.py`: uncompressed `.npy` per array, 30 Hz, floor-grounded `transl`) instead of recomputing it. `HRRefineWindowDataset` and the fixed validation windows in `hready/train/hr_refine_eval.py` now use `load_grounded_clip` in `hready/data/refine_dataset.py`.

- Values agree with `load_clip(ground=True)`: max absolute difference 2.2e-7 over 40 random clips (float32 storage).
- The train split uses the eligible clips that are present in the cache: 11,260 of 11,664 (404 are not in the cache, for example BMLmovi). Val (1,980) and test (1,103) are complete.
- The cache was built earlier (15,804 clips, 12 workers, 1,373 s, no errors) and is not rebuilt here. Rebuilding would rewrite that cache's `index.json`, so the 404 missing train clips are left out instead.

## Throughput (loader only)

Benchmark: `DataLoader`, batch size 8, window 64, `num_workers=0` (the trainer's setting), shuffled, seed 0, same entry list for both runs, first batch discarded, run on the training workstation (RTX 4090 box, other desktop processes running). The scratch benchmark script was deleted after the run.

| | batches/s | samples/s | batches timed |
|---|---|---|---|
| before: `load_clip(ground=True)` per sample | 0.21 | 1.7 | 30 |
| after: memmap cache | 15.44 | 123.5 | 300 |

Speed-up: about 72× (target was at least 2×). One run per setting, so treat the ratio as an order of magnitude, not a precise figure.

## Effect on training (measured during the item 8 ablation, same machine)

- Loading a batch dropped from 1–5.6 s to 0.05–0.10 s.
- The reconstruction-only arm now takes about 0.25 s per step (2000 steps and 10 validation passes in 1275 s). Before the fix it was loader-bound, at an estimated 2.5 hours per arm.
- The physics-loss arms take about 2 s per step. Their bottleneck is now the physics-loss computation, not the loader; this was not profiled here.

## Limits

- Loader throughput only; `num_workers>0` was not benchmarked, and the loader is no longer the limiting stage for the reconstruction-only arm.
- The fix was first made as part of item 8 (same working tree); it is recorded here as the item 16 loader result.
- Training uses 11,260 of 11,664 eligible train clips.
