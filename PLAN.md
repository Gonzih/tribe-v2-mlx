# PLAN: TRIBE v2 — Real Weight Download + MLX Conversion

## Task Restatement
Download real TRIBE v2 weights from HuggingFace and convert them to MLX format.
Verify converted weights produce correct output shapes. Document results.

## Model Sizes (determined by HF repo inspection)
- `facebook/tribev2` → `best.ckpt`: 709 MB — TOP PRIORITY
- `facebook/dinov2-large` → `model.safetensors`: 1.22 GB
- `facebook/w2v-bert-2.0` → `model.safetensors`: 2.32 GB (conformer_shaw.pt: 2.33 GB — skip)
- `facebook/vjepa2-vitg-fpc64-256` → `model.safetensors`: 4.14 GB (original/model.pth: 16 GB — skip)

## Download Strategy
- Download TRIBE + DINOv2 + Wav2Vec-BERT + V-JEPA2 safetensors only
- Ignore `*.pt`, `conformer_shaw.pt`, `original/` to avoid downloading multi-GB old-format files
- Run downloads in parallel (TRIBE + DINOv2 + Wav2Vec-BERT + V-JEPA2)

## Conversion Plan
1. TRIBE best.ckpt → load directly with torch, map to MLX via TribeTransformer.from_checkpoint()
2. DINOv2 → conversion/dinov2.py (already written)
3. Wav2Vec-BERT → conversion/wav2vec_bert.py (already written)
4. V-JEPA2 → conversion/vjepa2.py (already written) — if time allows
5. LLaMA → mlx-community/Llama-3.2-3B-Instruct-4bit (pre-converted, download via mlx_lm)

## Key Risks
- HF key names may differ from what conversion scripts expect → inspect actual keys, patch mappings
- TRIBE checkpoint may have different `model_build_args` structure than expected
- transformers 5.8.0 may have changed AutoModel API for some models

## Files to modify
- `scripts/download_weights.py` — add ignore_patterns for large .pt files
- `research/quantization-results.md` — new file documenting results
- `tests/` — add requires_weights tests if needed
