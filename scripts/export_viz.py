#!/usr/bin/env python3
"""
Export TRIBE activations and fsaverage5 brain mesh to JSON for the web visualizer.

Usage
-----
    python scripts/export_viz.py \\
        --activations output/nexus-demo-activations.npy \\
        --output-dir output/

Outputs
-------
    output/brain-mesh.json          -- fsaverage5 vertices + faces (LH + RH)
    output/nexus-demo-activations.json  -- activations for viz

Requirements
------------
    pip install nilearn nibabel
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def get_fsaverage5_mesh() -> dict:
    """
    Fetch the fsaverage5 pial surface from nilearn and return as a dict with
    flat float arrays suitable for JSON / Three.js BufferGeometry.

    Returns
    -------
    dict with keys:
        lh_vertices  : list[float]   flat [x0,y0,z0, x1,y1,z1, ...]  (10242*3)
        lh_faces     : list[int]     flat [i0,j0,k0, i1,j1,k1, ...]  (N*3)
        rh_vertices  : same for right hemisphere
        rh_faces     : same for right hemisphere
        lh_n_vertices: int  (10242)
        rh_n_vertices: int  (10242)
    """
    try:
        from nilearn import datasets, surface as nisurf
    except ImportError:
        print("Installing nilearn…", flush=True)
        os.system(f"{sys.executable} -m pip install nilearn nibabel --quiet")
        from nilearn import datasets, surface as nisurf

    print("Fetching fsaverage5 mesh from nilearn…", flush=True)
    fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")

    lh_coords, lh_faces = nisurf.load_surf_mesh(fsaverage.pial_left)
    rh_coords, rh_faces = nisurf.load_surf_mesh(fsaverage.pial_right)

    lh_coords = np.asarray(lh_coords, dtype=np.float32)
    lh_faces  = np.asarray(lh_faces,  dtype=np.int32)
    rh_coords = np.asarray(rh_coords, dtype=np.float32)
    rh_faces  = np.asarray(rh_faces,  dtype=np.int32)

    print(f"  LH: {lh_coords.shape[0]} vertices, {lh_faces.shape[0]} faces")
    print(f"  RH: {rh_coords.shape[0]} vertices, {rh_faces.shape[0]} faces")

    return {
        "lh_vertices":   lh_coords.flatten().tolist(),
        "lh_faces":      lh_faces.flatten().tolist(),
        "rh_vertices":   rh_coords.flatten().tolist(),
        "rh_faces":      rh_faces.flatten().tolist(),
        "lh_n_vertices": int(lh_coords.shape[0]),
        "rh_n_vertices": int(rh_coords.shape[0]),
    }


def export_mesh(output_dir: Path) -> None:
    mesh_path = output_dir / "brain-mesh.json"
    if mesh_path.exists():
        print(f"brain-mesh.json already exists at {mesh_path} — skipping fetch.")
        return

    mesh = get_fsaverage5_mesh()
    with open(mesh_path, "w") as f:
        json.dump(mesh, f, separators=(",", ":"))
    size_mb = mesh_path.stat().st_size / 1e6
    print(f"Saved brain-mesh.json → {mesh_path}  ({size_mb:.1f} MB)")


def export_activations(npy_path: Path, output_dir: Path) -> None:
    print(f"Loading activations from {npy_path}…")
    data = np.load(str(npy_path)).astype(np.float32)  # (n_segments, 20484)
    print(f"  Shape: {data.shape}  range [{data.min():.3f}, {data.max():.3f}]")

    n_segments, n_vertices = data.shape
    lh_n = n_vertices // 2   # 10242 for fsaverage5
    rh_n = n_vertices - lh_n

    global_min = float(data.min())
    global_max = float(data.max())
    # Use symmetric range so zero maps to white in diverging colormap
    abs_max = max(abs(global_min), abs(global_max))

    out = {
        "n_segments":    n_segments,
        "n_vertices":    n_vertices,
        "lh_n_vertices": lh_n,
        "rh_n_vertices": rh_n,
        "global_min":    -abs_max,
        "global_max":    abs_max,
        "data": data.tolist(),  # list of lists — each row is one time segment
    }

    stem = npy_path.stem  # e.g. "nexus-demo-activations"
    out_path = output_dir / f"{stem}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, separators=(",", ":"))

    size_mb = out_path.stat().st_size / 1e6
    print(f"Saved {out_path.name} → {out_path}  ({size_mb:.1f} MB, {n_segments} segments × {n_vertices} vertices)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export TRIBE activations + brain mesh for web visualizer")
    parser.add_argument(
        "--activations", nargs="+", required=True,
        metavar="PATH",
        help="One or more .npy activation files to export",
    )
    parser.add_argument(
        "--output-dir", default="output",
        help="Directory to write JSON files (default: output/)",
    )
    parser.add_argument(
        "--skip-mesh", action="store_true",
        help="Skip exporting brain-mesh.json (use if already exported)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_mesh:
        export_mesh(out_dir)

    for npy_path_str in args.activations:
        npy_path = Path(npy_path_str)
        if not npy_path.exists():
            print(f"WARNING: {npy_path} not found — skipping.", file=sys.stderr)
            continue
        export_activations(npy_path, out_dir)

    print("\nDone. Open viz/index.html in a browser (serve from project root).")


if __name__ == "__main__":
    main()
