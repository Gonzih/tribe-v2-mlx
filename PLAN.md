# PLAN — README Rewrite

## Task
Rewrite README.md to be technically dense, impressive, direct. Engineer tone. Numbers everywhere. No fluff.

## Approach
Single-pass rewrite. All source material read: research/tribe-v2-mlx-research.md, research/quantization-results.md, existing README. Branch → PR → merge.

## Files to touch
- README.md (full rewrite)

## Key numbers
- DINOv2-Large: 1161 MB → 332.9 MB, 3.5×
- Wav2Vec-BERT 2.0: 2214.5 MB → 644 MB, 3.4×
- V-JEPA2 ViT-g: 3946.6 MB → 1099 MB, 3.6×
- TRIBE ckpt: 709 MB (loaded directly)
- Total: ~8.3 GB → ~2.1 GB
- 47 tests all passing

## Risks
None. Pure documentation task.
