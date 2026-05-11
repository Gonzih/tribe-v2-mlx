# TRIBE v2 MLX — Real Video Inference Results

**Date:** 2026-05-10  
**Hardware:** Apple Silicon M4 Max (48 GB unified memory)  
**MLX device:** `Device(gpu, 0)`

---

## Videos Tested

| # | File | Size | Duration |
|---|------|------|----------|
| A | `/tmp/nexus-demo-video/nexus-demo.mp4` | 3.6 MB | ~96.5 s |
| B | `/tmp/invariant-topology-demo/hyperframes-demo/demo.mp4` | 3.5 MB | ~32.2 s |

---

## Run 1 — `run_inference.py` (CLI, no encoder weights)

```bash
time python3 scripts/run_inference.py --video <video> \
    --no-vjepa2 --no-dinov2 --no-audio --no-llama
```

No MLX encoder weights exist on this machine yet. The pipeline falls back
gracefully: empty feature dict → TRIBE forward skipped → zeros returned.

| Video | Output shape | Value range | Wall time |
|-------|-------------|-------------|-----------|
| nexus-demo.mp4 | `(1, 20484)` | `[0.000, 0.000]` | **6.85 s** |
| demo.mp4 | `(1, 20484)` | `[0.000, 0.000]` | **3.00 s** |

> **Note:** Wall time is dominated by PyAV video decode and MLX import — the
> zero output is the expected graceful fallback, not a bug. Real predictions
> require converted encoder weights (see `scripts/convert_to_mlx.py`).

---

## Run 2 — `run_e2e_test.py` (full pipeline, synthetic encoder features)

```bash
time python3 scripts/run_e2e_test.py <video> --seed 42
```

This script decodes the real video with PyAV, segments frames/audio using
production utilities, builds synthetic feature tensors with **production-matching
dimensions** (LLaMA 18 432-dim, V-JEPA2 2 816-dim, DINOv2 1 024-dim,
Wav2Vec-BERT 2 048-dim), then runs `TribeTransformer` on Metal GPU.

Because no real encoder weights exist, encoder features are replaced with seeded
random normal values — output values are **not semantically meaningful** but
validate the entire pipeline code path.

### Video A — nexus-demo.mp4 (~96.5 s)

| Metric | Value |
|--------|-------|
| Video duration | 96.5 s (386 frames @ 4 fps) |
| Segments | **25 × 4 s** |
| Output shape | **`(25, 20484)`** |
| Output min | −1.5293 |
| Output max | +1.6148 |
| Output mean | −0.0005 |
| Output std | 0.3352 |
| Non-zero outputs | 512 100 / 512 100 (100 %) |
| Wall time (total) | **6.64 s** |
| — Video decode | 6.31 s |
| — Audio decode | 0.01 s |
| — Feature build | 0.04 s |
| — Model load | 0.27 s |
| — Forward pass | **0.015 s** |
| Peak MLX GPU memory | **1.93 GB** |

**Per-segment activations (n_timepoints=25, hemodynamic offset 5 s baked in):**

| Seg | Window | min | max | mean | std | peak |v| |
|-----|--------|-----|-----|------|-----|---------|
| 0 | 0–4 s | −1.3845 | +1.4103 | −0.0047 | 0.3342 | 1.4103 |
| 1 | 4–8 s | −1.3456 | +1.2910 | −0.0032 | 0.3350 | 1.3456 |
| 2 | 8–12 s | −1.4071 | +1.3243 | −0.0039 | 0.3411 | 1.4071 |
| 3 | 12–16 s | −1.2323 | +1.2878 | +0.0020 | 0.3386 | 1.2878 |
| 4 | 16–20 s | −1.5293 | +1.1768 | −0.0010 | 0.3297 | **1.5293** |
| 5 | 20–24 s | −1.3332 | +1.3851 | −0.0031 | 0.3352 | 1.3851 |
| 6 | 24–28 s | −1.3657 | +1.2554 | −0.0002 | 0.3297 | 1.3657 |
| 7 | 28–32 s | −1.3919 | +1.2345 | +0.0006 | 0.3393 | 1.3919 |
| 8 | 32–36 s | −1.2906 | +1.2505 | +0.0019 | 0.3368 | 1.2906 |
| 9 | 36–40 s | −1.4272 | +1.3038 | +0.0048 | 0.3282 | 1.4272 |
| 10 | 40–44 s | −1.3664 | +1.2697 | +0.0004 | 0.3390 | 1.3664 |
| 11 | 44–48 s | −1.3945 | +1.4690 | −0.0031 | 0.3325 | 1.4690 |
| 12 | 48–52 s | −1.2900 | +1.1687 | −0.0001 | 0.3342 | 1.2900 |
| 13 | 52–56 s | −1.3148 | +1.3126 | +0.0011 | 0.3359 | 1.3148 |
| 14 | 56–60 s | −1.1500 | +1.5030 | +0.0020 | 0.3345 | 1.5030 |
| 15 | 60–64 s | −1.3131 | +1.5674 | −0.0025 | 0.3304 | **1.5674** |
| 16 | 64–68 s | −1.3269 | +1.1786 | −0.0022 | 0.3358 | 1.3269 |
| 17 | 68–72 s | −1.4834 | +1.6148 | +0.0014 | 0.3402 | **1.6148** |
| 18 | 72–76 s | −1.2666 | +1.3736 | −0.0022 | 0.3328 | 1.3736 |
| 19 | 76–80 s | −1.3253 | +1.3463 | +0.0045 | 0.3329 | 1.3463 |
| 20 | 80–84 s | −1.3793 | +1.2479 | +0.0012 | 0.3365 | 1.3793 |
| 21 | 84–88 s | −1.2922 | +1.2630 | −0.0025 | 0.3394 | 1.2922 |
| 22 | 88–92 s | −1.4393 | +1.3548 | +0.0005 | 0.3368 | 1.4393 |
| 23 | 92–96 s | −1.3877 | +1.2889 | −0.0019 | 0.3293 | 1.3877 |
| 24 | 96–100 s | −1.3503 | +1.3957 | −0.0019 | 0.3403 | 1.3957 |

**Segments with highest peak |activation|:** seg 17 (68–72 s, peak 1.6148), seg 15 (60–64 s, 1.5674), seg 4 (16–20 s, 1.5293).

**Top 10 vertices by |mean| across all segments (fsaverage5 mesh indices):**

| Rank | Vertex | |mean| activation |
|------|--------|---------------|
| 1 | 8790 | 0.7280 |
| 2 | 16820 | 0.7256 |
| 3 | 4010 | 0.7094 |
| 4 | 10584 | 0.6881 |
| 5 | 14205 | 0.6806 |
| 6 | 4952 | 0.6575 |
| 7 | 15079 | 0.6565 |
| 8 | 7712 | 0.6477 |
| 9 | 20399 | 0.6373 |
| 10 | 1829 | 0.6320 |

---

### Video B — hyperframes demo.mp4 (~32.2 s)

| Metric | Value |
|--------|-------|
| Video duration | 32.2 s (129 frames @ 4 fps) |
| Segments | **9 × 4 s** |
| Output shape | **`(9, 20484)`** |
| Output min | −1.4235 |
| Output max | +1.4008 |
| Output mean | −0.0010 |
| Output std | 0.3293 |
| Non-zero outputs | 184 356 / 184 356 (100 %) |
| Wall time (total) | **2.56 s** |
| — Video decode | 2.27 s |
| — Audio decode | 0.004 s |
| — Feature build | 0.027 s |
| — Model load | 0.25 s |
| — Forward pass | **0.013 s** |
| Peak MLX GPU memory | **1.91 GB** |

**Per-segment activations (n_timepoints=9):**

| Seg | Window | min | max | mean | std | peak |v| |
|-----|--------|-----|-----|------|-----|---------|
| 0 | 0–4 s | −1.2635 | +1.3752 | −0.0023 | 0.3369 | 1.3752 |
| 1 | 4–8 s | −1.3529 | +1.3111 | +0.0010 | 0.3225 | 1.3529 |
| 2 | 8–12 s | −1.2613 | +1.3289 | +0.0004 | 0.3294 | 1.3289 |
| 3 | 12–16 s | −1.3573 | +1.3144 | −0.0016 | 0.3282 | 1.3573 |
| 4 | 16–20 s | −1.4235 | +1.4008 | −0.0003 | 0.3264 | **1.4235** |
| 5 | 20–24 s | −1.1784 | +1.3932 | −0.0018 | 0.3300 | 1.3932 |
| 6 | 24–28 s | −1.3401 | +1.3718 | −0.0009 | 0.3257 | 1.3718 |
| 7 | 28–32 s | −1.2955 | +1.3804 | −0.0016 | 0.3330 | 1.3804 |
| 8 | 32–36 s | −1.2478 | +1.3100 | −0.0022 | 0.3314 | 1.3100 |

**Segments with highest peak |activation|:** seg 4 (16–20 s, 1.4235), seg 0 (0–4 s, 1.3752), seg 7 (28–32 s, 1.3804).

**Top 10 vertices by |mean| across all segments:**

| Rank | Vertex | |mean| activation |
|------|--------|---------------|
| 1 | 13298 | 0.8351 |
| 2 | 3291 | 0.8026 |
| 3 | 2414 | 0.7936 |
| 4 | 19180 | 0.7574 |
| 5 | 651 | 0.7528 |
| 6 | 2447 | 0.7506 |
| 7 | 17647 | 0.7492 |
| 8 | 1678 | 0.7406 |
| 9 | 8624 | 0.7329 |
| 10 | 19251 | 0.7282 |

---

## Cross-Video Comparison

| Property | nexus-demo (A) | hyperframes demo (B) |
|----------|----------------|----------------------|
| Duration | 96.5 s | 32.2 s |
| n_segments | 25 | 9 |
| Output shape | `(25, 20484)` | `(9, 20484)` |
| Global min | −1.5293 | −1.4235 |
| Global max | +1.6148 | +1.4008 |
| Global std | 0.3352 | 0.3293 |
| Wall time | 6.64 s | 2.56 s |
| Peak GPU memory | 1.93 GB | 1.91 GB |
| Highest-peak segment | seg 17 (68–72 s) | seg 4 (16–20 s) |

**Pearson correlation of mean vertex activations (A vs B): 0.0033**

The near-zero correlation is expected: because videos have different durations
(25 vs 9 segments), different numbers of random feature vectors are sampled
from the PRNG, resulting in independent pseudo-random draws even with the same
seed. Top activated vertices differ entirely between the two videos.

Both distributions are centred near zero (mean ≈ 0) with std ≈ 0.33 — consistent
with a random-weight transformer producing normally-distributed outputs.

---

## Key Observations

1. **Pipeline scales with video length.** Wall time is dominated by video decode
   (~0.065 s/s of video). TRIBE forward pass itself takes only ~13–15 ms regardless
   of video length.

2. **Memory footprint is stable at ~1.9 GB.** Peak GPU memory is driven by the
   TRIBE transformer weight allocation, not by sequence length (9 vs 25 segments).

3. **Output is fully non-zero and finite.** 100 % of the 20 484 fsaverage5 vertices
   receive a prediction for every segment.

4. **Activation range is symmetric.** Both videos show values in roughly ±1.6 β-units,
   consistent with the TRIBE transformer's random-weight distribution; std ≈ 0.33 matches
   the E2E test baseline from PR #6.

5. **Real encoder weights are the next step.** With converted MLX encoder weights
   (V-JEPA2 + DINOv2 + Wav2Vec-BERT + LLaMA), output would reflect actual video content
   differences. The two videos — a Nexus project demo and a topology research demo — would
   be expected to produce very different cortical activation patterns in visual and semantic
   processing regions.

---

## Reproducibility

```bash
# Install
pip install -e . --quiet

# Baseline (no weights)
python scripts/run_inference.py --video <VIDEO> \
    --no-vjepa2 --no-dinov2 --no-audio --no-llama

# Full pipeline with synthetic features (seed for reproducibility)
python scripts/run_e2e_test.py <VIDEO> --seed 42
```
