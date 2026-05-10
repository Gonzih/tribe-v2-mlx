# PLAN: TRIBE v2 MLX Quantization Research

## Task Restatement
Research and document a plan for running TRIBE v2 (Facebook Research's brain activity prediction model) with quantized inference on Apple Silicon (M4 Max, 48GB unified memory) using the MLX framework. The output is a comprehensive markdown research document covering architecture, conversion paths, and integration plan.

## Approaches

### Approach A: Full Parallel Research (chosen)
Launch multiple research agents in parallel to fetch TRIBE v2 source, encoder architectures, and MLX capabilities simultaneously, then synthesize into a single document.
- **Pro**: Fastest; all data gathered before writing begins
- **Con**: Some redundancy if sources overlap

### Approach B: Sequential Deep Dive
Research each component one-by-one, going deeper as needed.
- **Pro**: Can adapt based on what's found
- **Con**: Slow; likely to hit time limits

### Approach C: Minimal Viable Research
Just check README files and write a high-level plan.
- **Pro**: Fast
- **Con**: Too shallow for an implementation foundation

## Chosen Approach
**Approach A** — parallel research agents to fetch all sources, then synthesize.

## Files to Touch
- `PLAN.md` (this file)
- `TODO.md`
- `research/tribe-v2-mlx-research.md` (main output)

## Risks & Unknowns
- TRIBE v2 repo may have been renamed or is private
- V-JEPA2-Giant MLX port likely doesn't exist; need custom conversion plan
- Wav2Vec-BERT 2.0 MLX support unknown
- 4-bit quantization of vision encoders may degrade fMRI prediction quality significantly
- GitHub raw file URLs may differ from expected paths
