"""STL / OBJ / GLB / STEP CAD import and procedural assembly mesh export."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from printsim.geometry.builder import AssemblyModel
from printsim.schema import CustomMeshSpec, get_material

if TYPE_CHECKING:
    import trimesh


def _hex_to_rgba(hex_color: str, alpha: int = 255) -> list[int]:
    h = hex_color.lstrip("#")
    return [int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), alpha]


def _beam_mesh(
    p0_mm: np.ndarray,
    p1_mm: np.ndarray,
    width_mm: float,
    height_mm: float,
    shape: str,
    color_hex: str,
) -> trimesh.Trimesh:
    import trimesh

    vec = p1_mm - p0_mm
    length = float(np.linalg.norm(vec))
    if length < 1e-3:
        length = 1.0
    direction = vec / length

    if shape == "rod":
        mesh = trimesh.creation.cylinder(radius=width_mm * 0.5, height=length, sections=16)
    else:
        mesh = trimesh.creation.box(extents=[width_mm, height_mm, length])

    # Align local +Z with beam direction
    z_axis = np.array([0.0, 0.0, 1.0])
    align = trimesh.geometry.align_vectors(z_axis, direction)
    if align is None:
        align = np.eye(4)
        if np.dot(z_axis, direction) < 0:
            align[0, 0] = -1.0
            align[2, 2] = -1.0
    mid = 0.5 * (p0_mm + p1_mm)
    transform = np.eye(4)
    transform[:3, :3] = align[:3, :3]
    transform[:3, 3] = mid
    mesh.apply_transform(transform)
    mesh.visual.face_colors = _hex_to_rgba(color_hex)
    return mesh


def build_trimesh_scene(assembly: AssemblyModel) -> trimesh.Scene:
    """Create a 3D trimesh Scene (units: mm) of the full printer assembly."""
    import trimesh

    scene = trimesh.Scene()
    nmap = assembly.node_map

    for member in assembly.members:
        p0 = nmap[member.start_id].position_m * 1000.0
        p1 = nmap[member.end_id].position_m * 1000.0
        mesh = _beam_mesh(
            p0,
            p1,
            member.profile.width_m * 1000.0,
            member.profile.height_m * 1000.0,
            member.profile.shape,
            member.material.color_hex,
        )
        scene.add_geometry(mesh, node_name=member.id)

    # Print bed plate
    bx, by = assembly.spec.bed.size_mm
    bt = assembly.spec.bed.plate_thickness_mm
    bed_mesh = trimesh.creation.box(extents=[bx, by, bt])
    bed_pos = assembly.bed_center_m * 1000.0
    bed_mesh.apply_translation(bed_pos - np.array([0.0, 0.0, bt * 0.5]))
    bed_mesh.visual.face_colors = _hex_to_rgba("#3b4252")
    scene.add_geometry(bed_mesh, node_name="print_bed")

    # Toolhead bounding volume
    th_node = nmap[assembly.toolhead_node_id]
    ex, ey, ez = assembly.spec.toolhead.envelope_mm
    th_mesh = trimesh.creation.box(extents=[ex, ey, ez])
    th_mesh.apply_translation(th_node.position_m * 1000.0 + np.array([0.0, 0.0, -ez * 0.25]))
    th_mesh.visual.face_colors = _hex_to_rgba("#e63946")
    scene.add_geometry(th_mesh, node_name="toolhead")

    return scene


def export_assembly_stl(assembly: AssemblyModel, output_path: str | Path) -> Path:
    """Export the full printer assembly to an STL (or OBJ/GLB based on extension)."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    scene = build_trimesh_scene(assembly)
    combined = scene.to_geometry()
    combined.export(str(out))
    return out


def inspect_custom_mesh(mesh_spec: CustomMeshSpec) -> dict:
    """Load an external STL/OBJ (or STEP if build123d is installed) and compute mass properties."""
    path = Path(mesh_spec.file_path)
    if not path.exists():
        raise FileNotFoundError(f"Custom mesh file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in (".step", ".stp"):
        try:
            import build123d as b3d  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "STEP import requires optional CAD extra (`pip install build123d`). "
                "Export your part to STL/OBJ or install build123d."
            ) from exc
        shape = b3d.import_step(str(path))
        bbox = shape.bounding_box()
        vol_mm3 = float(shape.volume)
        mat = get_material(mesh_spec.material)
        mass_g = (
            mesh_spec.override_mass_g
            if mesh_spec.override_mass_g is not None
            else (vol_mm3 * 1e-9) * mat.density_kg_m3 * 1000.0
        )
        return {
            "id": mesh_spec.id,
            "format": "step",
            "volume_mm3": vol_mm3,
            "mass_g": mass_g,
            "bounds_mm": [
                [bbox.min.X, bbox.min.Y, bbox.min.Z],
                [bbox.max.X, bbox.max.Y, bbox.max.Z],
            ],
            "watertight": True,
        }

    import trimesh

    loaded = trimesh.load(str(path), force="mesh")
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.dump(concatenate=True)
    vol_mm3 = abs(float(loaded.volume)) if loaded.is_volume else float(np.prod(loaded.extents) * 0.35)
    mat = get_material(mesh_spec.material)
    mass_g = (
        mesh_spec.override_mass_g
        if mesh_spec.override_mass_g is not None
        else (vol_mm3 * 1e-9) * mat.density_kg_m3 * 1000.0
    )
    return {
        "id": mesh_spec.id,
        "format": suffix.lstrip("."),
        "volume_mm3": vol_mm3,
        "mass_g": mass_g,
        "bounds_mm": loaded.bounds.tolist(),
        "extents_mm": loaded.extents.tolist(),
        "watertight": bool(loaded.is_watertight),
    }
