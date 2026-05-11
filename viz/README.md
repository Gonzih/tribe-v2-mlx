# TRIBE v2 — 3D Brain Activation Viewer

A standalone Three.js web app that renders TRIBE v2 fMRI predictions as a per-vertex
activation heatmap on the fsaverage5 cortical surface mesh.

## Quick start

### 1. Run inference to get activations

```bash
# Ensure encoder weights are downloaded and converted (see project README)
python scripts/run_inference.py \
    --video /tmp/nexus-demo-video/nexus-demo.mp4 \
    --output output/nexus-demo-activations.npy

python scripts/run_inference.py \
    --video /tmp/invariant-topology-demo/hyperframes-demo/demo.mp4 \
    --output output/invariant-topology-activations.npy
```

### 2. Export JSON for the visualizer

```bash
# Install nilearn (needed once)
pip install nilearn nibabel

# Export brain mesh + activation arrays to JSON
python scripts/export_viz.py \
    --activations output/nexus-demo-activations.npy \
                  output/invariant-topology-activations.npy \
    --output-dir output/
```

This creates:
- `output/brain-mesh.json` — fsaverage5 pial surface (LH + RH vertices & faces)
- `output/nexus-demo-activations.json`
- `output/invariant-topology-activations.json`

### 3. Open the viewer

The viewer is a **single HTML file** — no build step needed.

Open directly in a browser (Chrome/Firefox/Safari):

```bash
# macOS
open viz/index.html

# Linux
xdg-open viz/index.html
```

> **Note**: If loading local JSON files fails due to browser security policies,
> serve the project root with a simple HTTP server:
>
> ```bash
> python3 -m http.server 8080
> # Then open http://localhost:8080/viz/index.html
> ```

### 4. Using the viewer

1. Click **Brain Mesh JSON** → select `output/brain-mesh.json`
2. Click **Activations JSON** → select `output/nexus-demo-activations.json`
3. The brain renders with a blue→white→red heatmap (negative→zero→positive β)
4. Use the **time slider** to scrub through 4-second segments
5. Click **▶ Play** to animate through segments
6. Toggle **Left / Right / Both** hemispheres
7. Drag to rotate, scroll to zoom

## Colormap

| Color | Meaning |
|-------|---------|
| Deep blue | Strong negative BOLD prediction |
| White | Near-zero / baseline |
| Deep red | Strong positive BOLD prediction |

The scale is symmetric: `[-abs_max, +abs_max]` where `abs_max = max(|min|, |max|)`.

## File formats

### `brain-mesh.json`

```json
{
  "lh_vertices":   [x0, y0, z0, x1, y1, z1, ...],
  "lh_faces":      [i0, j0, k0, ...],
  "rh_vertices":   [...],
  "rh_faces":      [...],
  "lh_n_vertices": 10242,
  "rh_n_vertices": 10242
}
```

### `*-activations.json`

```json
{
  "n_segments":    25,
  "n_vertices":    20484,
  "lh_n_vertices": 10242,
  "rh_n_vertices": 10242,
  "global_min":    -1.531,
  "global_max":    1.531,
  "data": [[...20484 floats for t=0...], [...for t=1...], ...]
}
```

## Technical notes

- **Mesh**: [fsaverage5](https://nilearn.github.io/stable/modules/generated/nilearn.datasets.fetch_surf_fsaverage.html)
  pial surface — 10,242 vertices per hemisphere (20,484 total), ~40,000 faces per hemisphere.
- **Renderer**: Three.js r160, `MeshPhongMaterial` with `vertexColors: true`
- **No build step**: single HTML file with ES module import map pointing to unpkg CDN
- **Performance**: tested on Chrome 124 — smooth 60 fps for both hemispheres simultaneously
