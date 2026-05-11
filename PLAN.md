# PLAN — Full Encoder Pipeline + 3D Brain Visualization

## Task Restatement

1. **Part 1**: Wire real MLX encoder weights and run full inference on both videos,
   saving semantically meaningful activations as `.npy` (and `.json` for visualization).
2. **Part 2**: Build a standalone `viz/index.html` Three.js brain visualization that
   renders TRIBE output as a per-vertex heatmap on the fsaverage5 cortical mesh,
   plus `scripts/export_viz.py` to generate the needed JSON data.

---

## Part 1 — Approach

**Chosen: convert scripts download only needed safetensors directly via hf_hub_download**
- Each conversion script calls `hf_hub_download` for just `model.safetensors`
- TRIBE checkpoint downloaded via `snapshot_download("facebook/tribev2")`
- LLaMA: attempt mlx-community/Llama-3.2-3B-Instruct-4bit; skip if slow

---

## Part 2 — Approach

- **nilearn** `fetch_surf_fsaverage(mesh='fsaverage5')` for mesh vertices + faces
- **Three.js r160** via CDN (standalone HTML, no build step)
- Diverging colormap: blue → white → red
- Dark background (#0a0a14), OrbitControls, time slider, play/pause

---

## Files to Create

- `scripts/export_viz.py` — export mesh + activations to JSON
- `output/nexus-demo-activations.npy`
- `output/invariant-topology-activations.npy`
- `output/brain-mesh.json`
- `output/nexus-demo-activations.json`
- `viz/index.html`
- `viz/README.md`

---

## Risks

1. TRIBE checkpoint may be gated → fall back to synthetic TRIBE weights
2. V-JEPA2 may have sharded safetensors → need to handle shard discovery
3. LLaMA ~1.7 GB download — skip if unavailable
