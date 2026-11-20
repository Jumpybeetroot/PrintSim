"""Isolated Gmsh subprocess worker for STEP -> 3D Tet4 meshing.

Running Gmsh in a dedicated child process prevents C++ OpenCASCADE (OCCT) shared-library
symbol collisions when `build123d` (cadquery-ocp) is also loaded in the parent process,
and guarantees full thread-safety inside multi-threaded FastAPI servers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import gmsh
import numpy as np


def mesh_step_to_npz(step_path: Path, out_npz: Path, mesh_size_mm: float | None) -> None:
    if not step_path.exists():
        raise FileNotFoundError(f"STEP file not found: {step_path}")

    gmsh.initialize(argv=["gmsh"], interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setString("Geometry.OCCTargetUnit", "MM")
        gmsh.model.add("printsim_part")

        gmsh.model.occ.importShapes(str(step_path))
        gmsh.model.occ.synchronize()

        volumes = gmsh.model.getEntities(dim=3)
        if not volumes:
            raise ValueError(f"No 3D solid volumes found in STEP file: {step_path}")

        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
        extents_mm = np.array([xmax - xmin, ymax - ymin, zmax - zmin], dtype=float)
        max_extent_mm = float(np.max(extents_mm))
        if max_extent_mm <= 1e-6:
            raise ValueError("STEP geometry has degenerate zero dimensions.")

        if mesh_size_mm is None:
            h_mm = float(np.clip(max_extent_mm / 10.0, 2.5, 16.0))
        else:
            h_mm = float(max(0.25, mesh_size_mm))

        h_min = max(0.25, min(h_mm * 0.4, 2.2 if h_mm >= 3.0 else h_mm * 0.45))
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h_min)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h_mm * 1.05)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
        gmsh.option.setNumber("Mesh.Optimize", 1)

        gmsh.model.mesh.generate(3)

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        elem_tags, elem_node_tags = gmsh.model.mesh.getElementsByType(4)
        if len(elem_tags) == 0:
            raise RuntimeError("Gmsh generated 0 tetrahedral elements from STEP geometry.")

        node_tags_arr = np.asarray(node_tags, dtype=np.int64)
        coords_mm = np.asarray(node_coords, dtype=np.float64).reshape(-1, 3)
        raw_tet_tags = np.asarray(elem_node_tags, dtype=np.int64).reshape(-1, 4)

        sort_order = np.argsort(node_tags_arr)
        sorted_tags = node_tags_arr[sort_order]
        pos = np.searchsorted(sorted_tags, raw_tet_tags.ravel())
        if not np.all(sorted_tags[pos] == raw_tet_tags.ravel()):
            raise RuntimeError("Gmsh returned tetrahedra referencing missing node tags.")
        raw_indices = sort_order[pos].reshape(-1, 4)

        used_raw, compact_inv = np.unique(raw_indices, return_inverse=True)
        points_mm = np.ascontiguousarray(coords_mm[used_raw], dtype=np.float64)
        tets = np.ascontiguousarray(compact_inv.reshape(-1, 4), dtype=np.int32)

        np.savez(out_npz, points_mm=points_mm, tets=tets, h_mm=np.array([h_mm], dtype=np.float64))
    finally:
        if gmsh.isInitialized():
            gmsh.finalize()


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        sys.stderr.write("Usage: python -m printsim.fea._gmsh_worker <step_path> <out_npz> [mesh_size_mm]\n")
        return 2
    step_path = Path(argv[1])
    out_npz = Path(argv[2])
    mesh_size_mm = None if len(argv) < 4 or argv[3] in ("", "none", "None") else float(argv[3])
    mesh_step_to_npz(step_path, out_npz, mesh_size_mm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
