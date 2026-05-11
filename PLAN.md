# PLAN: TRIBE v2 MLX — End-to-End Inference Test

## Task Restatement
Run actual end-to-end inference on a synthetic test video (`/tmp/test-tribe.mp4`,
10-second, 256×256, ffmpeg testsrc), capture wall time / output shape / peak MLX
memory / value range, and document results in `research/inference-test-results.md`.
No real trained weights exist on this machine — the test must work with random-weight
models while still exercising the full code path.

## Three Approaches Considered

### A. Run `run_inference.py` with all encoders disabled (--no-*)
- Pros: minimal changes, tests import and TRIBE forward
- Cons: `features` dict is empty → returns zeros → output range [0, 0], not interesting

### B. Download real weights then run (709 MB TRIBE + encoder models)
- Pros: truly representative results
- Cons: HF download inside a CI-like runner is slow and fragile; large models need
  multi-hour download; previous TODO showed downloads as unchecked

### C. Write `scripts/run_e2e_test.py` — real video decode + synthetic features → TRIBE
- Decode real video with PyAV (exercises full I/O and preprocessing)
- Build synthetic MLX feature tensors with correct dims for TribeTransformer
- Run TribeTransformer (random weights) forward pass → non-zero output
- Measure wall time (video load + TRIBE forward), peak Metal memory, output stats
- **Chosen approach**: most honest, fully exercises the code path, no network required

Additionally, run `run_inference.py` as specified (with --no-* flags) to document
the "no-encoder baseline" separately.

## Files to Touch
- `scripts/run_e2e_test.py` — new standalone script
- `research/inference-test-results.md` — new results doc
- `PLAN.md`, `TODO.md` — planning artifacts (this file)

## Risks
- MLX lazy eval means `mx.eval()` must be called to get meaningful timing
- `mx.get_peak_memory()` returns bytes; reset with `mx.reset_peak_memory()` before timing
- V-JEPA2 ViT-g / DINOv2 instantiation with real config + random weights would be
  very slow (~30 s for parameter init); use TribeTransformer (synthetic config) only
