# PLAN: TRIBE v2 MLX — Real Video Inference on Hyperframes Videos

## Task Restatement
Run TRIBE v2 MLX inference on two real project videos:
1. `/tmp/nexus-demo-video/nexus-demo.mp4` — Nexus project demo
2. `/tmp/invariant-topology-demo/hyperframes-demo/demo.mp4` — Invariant Topology research demo

Capture output shape, wall time, peak MLX memory, activation stats, compare the two
videos, and write results to `research/real-video-results.md`.

## Situation
- No real encoder weights exist on this machine (previous agent confirmed)
- `run_inference.py` without weights → zeros (graceful fallback)
- `run_e2e_test.py` decodes real video → synthetic features → real TribeTransformer forward → non-zero output
- Both scripts work and produce valid (if not real-encoder) results

## Three Approaches

### A. run_inference.py only
- Pros: matches the exact CLI in the task instructions
- Cons: returns zeros (no real weights) — not informative about activation patterns

### B. run_e2e_test.py only
- Pros: real video decode, non-zero outputs, exercises full pipeline
- Cons: uses synthetic (random) features — not real encoder outputs

### C. Both: run_inference.py + run_e2e_test.py on both videos  ← CHOSEN
- run_inference.py documents the "no-weights baseline"
- run_e2e_test.py gives meaningful non-zero activations to compare
- Best for documentation showing what the pipeline produces with real video decoding

## Files to Touch
- `PLAN.md`, `TODO.md` — planning artifacts
- `research/real-video-results.md` — new results document
- (No code changes needed — existing scripts work)

## Risks
- Videos may have no audio track → graceful silence fallback already coded
- Video duration affects n_segments → captured in results
- Random-weight features mean activation patterns are random, not semantically meaningful
  (clearly noted in results doc)
