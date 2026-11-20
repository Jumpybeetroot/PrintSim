"""OpenCASCADE B-Rep parametric CAD part library and STEP export powered by build123d.

All geometry is constructed in millimeters (mm) with exact analytical OpenCASCADE
B-Rep surfaces (planes, cylinders, fillets) so exported STEP files can be edited
in FreeCAD/Fusion 360/SolidWorks or meshed conformally by Gmsh for continuum FEA.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Literal

import numpy as np

from printsim.geometry.builder import AssemblyModel

PartPreset = Literal["corner_gusset", "xy_motor_mount", "toolhead_carriage", "cantilever_bar"]
FaceName = Literal["min_x", "max_x", "min_y", "max_y", "min_z", "max_z"]

_PART_DEFAULT_BCS: dict[PartPreset, tuple[FaceName, FaceName, tuple[float, float, float]]] = {
    "corner_gusset": ("min_y", "max_y", (60.0, 0.0, 0.0)),
    "xy_motor_mount": ("min_y", "max_y", (0.0, 45.0, -25.0)),
    "toolhead_carriage": ("min_z", "max_z", (35.0, -20.0, -10.0)),
    "cantilever_bar": ("min_x", "max_x", (0.0, 0.0, -50.0)),
}


def is_cad_available() -> bool:
    """Return True if the optional build123d/OpenCASCADE CAD stack is importable."""
    try:
        import build123d  # noqa: F401
    except ImportError:
        return False
    return True


def _require_build123d() -> Any:
    """Lazily import build123d or raise an actionable installation error."""
    try:
        import build123d as b3d
    except ImportError as exc:
        raise RuntimeError(
            "STEP/B-Rep CAD requires the optional CAD extra: uv sync --extra cad"
        ) from exc
    return b3d


def get_part_default_bcs(
    preset: PartPreset,
) -> tuple[FaceName, FaceName, tuple[float, float, float]]:
    """Return (fixed_face, load_face, default_force_n) matching PrintSim solid FEA conventions."""
    if preset not in _PART_DEFAULT_BCS:
        raise ValueError(f"Unknown solid part preset: {preset}")
    return _PART_DEFAULT_BCS[preset]


def _ensure_single_solid(shape: Any, label: str) -> Any:
    """Clean a boolean result and return its single valid OpenCASCADE Solid."""
    try:
        cleaned = shape.clean()
    except Exception:
        cleaned = shape

    solids = cleaned.solids()
    if len(solids) != 1:
        raise RuntimeError(
            f"Expected a single connected B-Rep solid for '{label}', got {len(solids)} solids."
        )
    solid = solids[0]
    if not solid.is_valid or float(solid.volume) <= 0.0:
        raise RuntimeError(f"Generated B-Rep solid for '{label}' is invalid or empty.")
    solid.label = label
    return solid


def _safe_fillet(b3d: Any, solid: Any, edges: list[Any], radius: float) -> Any:
    """Apply an OpenCASCADE fillet to selected edges, falling back cleanly if rejected."""
    if not edges or radius <= 0.0:
        return solid
    try:
        filleted = solid.fillet(radius, edges)
        candidate = _ensure_single_solid(filleted, solid.label or "part")
        if candidate.is_valid and float(candidate.volume) > 0.0:
            return candidate
    except Exception:
        return solid
    return solid


def _select_vertical_edges(
    b3d: Any,
    solid: Any,
    expected_length: float,
    xy_targets: list[tuple[float, float]],
    tol: float = 0.15,
) -> list[Any]:
    """Select straight vertical edges matching target (x, y) coordinates and length."""
    selected: list[Any] = []
    for edge in solid.edges():
        if edge.geom_type != b3d.GeomType.LINE:
            continue
        if abs(float(edge.length) - expected_length) > tol:
            continue
        center = edge.center()
        for tx, ty in xy_targets:
            if abs(float(center.X) - tx) <= tol and abs(float(center.Y) - ty) <= tol:
                selected.append(edge)
                break
    return selected


def build_part_brep(preset: PartPreset) -> Any:
    """Construct a single watertight OpenCASCADE B-Rep Solid (in mm) in the positive octant.

    All parts are placed with their minimum bounding-box corner at ``(0, 0, 0)`` so
    their ``min_x/max_x/min_y/max_y/min_z/max_z`` boundary-condition faces match
    :func:`printsim.fea.solid_fem.generate_parametric_part_grid`.
    """
    b3d = _require_build123d()

    if preset == "cantilever_bar":
        # Exact 60 x 12 x 12 mm solid bar [0..60] x [0..12] x [0..12] (no fillets for analytical FEA)
        bar = b3d.Solid.make_box(60.0, 12.0, 12.0, plane=b3d.Plane.XY)
        return _ensure_single_solid(bar, "cantilever_bar")

    if preset == "corner_gusset":
        # 40 x 40 x 20 mm L-bracket:
        # - 8mm X-flange (y in [0..8]) and Y-flange (x in [0..8])
        # - 8mm-thick central triangular gusset web (z in [6..14]) embedded into flanges
        # - 2x M5 clearance bolt holes (Ø5.5mm) at x=26, z=10 (along Y) and y=26, z=10 (along X)
        flange_x = b3d.Solid.make_box(40.0, 8.0, 20.0, plane=b3d.Plane.XY)
        flange_y = b3d.Solid.make_box(8.0, 40.0, 20.0, plane=b3d.Plane.XY)
        web_wire = b3d.Wire.make_polygon(
            [(4.0, 4.0, 6.0), (25.5, 4.0, 6.0), (4.0, 25.5, 6.0)]
        )
        web = b3d.Solid.extrude(b3d.Face(web_wire), (0.0, 0.0, 8.0))

        body = flange_x.fuse(flange_y, web)
        body = _ensure_single_solid(body, "corner_gusset")

        hole_x = b3d.Solid.make_cylinder(
            2.75,
            10.0,
            plane=b3d.Plane(origin=(26.0, -1.0, 10.0), z_dir=(0.0, 1.0, 0.0)),
        )
        hole_y = b3d.Solid.make_cylinder(
            2.75,
            10.0,
            plane=b3d.Plane(origin=(-1.0, 26.0, 10.0), z_dir=(1.0, 0.0, 0.0)),
        )
        body = body.cut(hole_x, hole_y)
        solid = _ensure_single_solid(body, "corner_gusset")

        fillet_edges = _select_vertical_edges(
            b3d,
            solid,
            expected_length=20.0,
            xy_targets=[(0.0, 0.0), (40.0, 8.0), (8.0, 40.0)],
        )
        return _safe_fillet(b3d, solid, fillet_edges, radius=2.0)

    if preset == "xy_motor_mount":
        # 60 x 52 x 16 mm CoreXY NEMA17 motor mount plate:
        # - 8mm top deck (z in [8..16]), 8mm rear flange (y in [0..8]), 8mm left/right ribs
        # - Central NEMA17 pilot bore (Ø22.5mm) at (30, 30) through Z
        # - 4x NEMA17 M3 mounting holes (Ø3.4mm) on 31 x 31 mm square centered at (30, 30)
        # - 2x M5 rear frame mounting holes (Ø5.5mm) at x=15, 45, z=6 through Y
        top_deck = b3d.Solid.make_box(
            60.0, 52.0, 8.0, plane=b3d.Plane(origin=(0.0, 0.0, 8.0))
        )
        rear_flange = b3d.Solid.make_box(
            60.0, 8.0, 16.0, plane=b3d.Plane(origin=(0.0, 0.0, 0.0))
        )
        left_rib = b3d.Solid.make_box(
            8.0, 52.0, 16.0, plane=b3d.Plane(origin=(0.0, 0.0, 0.0))
        )
        right_rib = b3d.Solid.make_box(
            8.0, 52.0, 16.0, plane=b3d.Plane(origin=(52.0, 0.0, 0.0))
        )

        body = top_deck.fuse(rear_flange, left_rib, right_rib)
        body = _ensure_single_solid(body, "xy_motor_mount")

        pilot_bore = b3d.Solid.make_cylinder(
            11.25, 20.0, plane=b3d.Plane(origin=(30.0, 30.0, -2.0))
        )
        m3_holes = [
            b3d.Solid.make_cylinder(
                1.7,
                20.0,
                plane=b3d.Plane(origin=(30.0 + dx, 30.0 + dy, -2.0)),
            )
            for dx in (-15.5, 15.5)
            for dy in (-15.5, 15.5)
        ]
        m5_rear_holes = [
            b3d.Solid.make_cylinder(
                2.75,
                11.0,
                plane=b3d.Plane(origin=(hx, -1.0, 6.0), z_dir=(0.0, 1.0, 0.0)),
            )
            for hx in (15.0, 45.0)
        ]

        body = body.cut(pilot_bore, *m3_holes, *m5_rear_holes)
        solid = _ensure_single_solid(body, "xy_motor_mount")

        fillet_edges = _select_vertical_edges(
            b3d,
            solid,
            expected_length=16.0,
            xy_targets=[(0.0, 52.0), (60.0, 52.0)],
        )
        return _safe_fillet(b3d, solid, fillet_edges, radius=2.0)

    if preset == "toolhead_carriage":
        # 56 x 48 x 12 mm direct-drive X-carriage plate:
        # - 8mm base plate (z in [0..8]) + 4mm-tall, 8mm-wide perimeter stiffening ribs (z in [8..12])
        # - Central 16 x 16 mm weight-relief cutout (x in [20..36], y in [16..32])
        # - 4x MGN12H M3 mounting holes (Ø3.4mm) on 20 x 20 mm square centered at (28, 24)
        outer = b3d.Solid.make_box(56.0, 48.0, 12.0, plane=b3d.Plane.XY)
        rib_pocket = b3d.Solid.make_box(
            40.0, 32.0, 6.0, plane=b3d.Plane(origin=(8.0, 8.0, 8.0))
        )
        center_cutout = b3d.Solid.make_box(
            16.0, 16.0, 16.0, plane=b3d.Plane(origin=(20.0, 16.0, -2.0))
        )
        mgn12_holes = [
            b3d.Solid.make_cylinder(
                1.7,
                16.0,
                plane=b3d.Plane(origin=(28.0 + dx, 24.0 + dy, -2.0)),
            )
            for dx in (-10.0, 10.0)
            for dy in (-10.0, 10.0)
        ]

        body = outer.cut(rib_pocket, center_cutout, *mgn12_holes)
        solid = _ensure_single_solid(body, "toolhead_carriage")

        fillet_edges = _select_vertical_edges(
            b3d,
            solid,
            expected_length=12.0,
            xy_targets=[(0.0, 0.0), (56.0, 0.0), (0.0, 48.0), (56.0, 48.0)],
        )
        return _safe_fillet(b3d, solid, fillet_edges, radius=2.0)

    raise ValueError(f"Unknown solid part preset: {preset}")


def export_part_step(preset: PartPreset, output_path: str | Path) -> Path:
    """Build a parametric B-Rep part and write it to an ISO-10303-21 STEP file (units: mm)."""
    b3d = _require_build123d()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    solid = build_part_brep(preset)
    b3d.export_step(solid, str(out))
    return out


def export_part_step_bytes(preset: PartPreset) -> bytes:
    """Build a parametric B-Rep part and return its STEP file payload as bytes."""
    with tempfile.TemporaryDirectory(prefix="printsim_step_") as tmpdir:
        out = export_part_step(preset, Path(tmpdir) / f"{preset}.step")
        return out.read_bytes()


def export_part_stl_bytes(
    preset: PartPreset,
    tolerance: float = 0.02,
    angular_tolerance: float = 0.1,
) -> bytes:
    """Build a parametric B-Rep part and return a tessellated binary STL payload as bytes."""
    b3d = _require_build123d()
    solid = build_part_brep(preset)
    with tempfile.TemporaryDirectory(prefix="printsim_stl_") as tmpdir:
        out = Path(tmpdir) / f"{preset}.stl"
        b3d.export_stl(
            solid,
            str(out),
            tolerance=tolerance,
            angular_tolerance=angular_tolerance,
            ascii_format=False,
        )
        return out.read_bytes()


def _member_plane(b3d: Any, p0_mm: np.ndarray, direction: np.ndarray) -> Any:
    """Build a right-handed build123d Plane at p0_mm with +Z aligned to member direction."""
    if abs(float(direction[2])) < 0.99:
        ref = np.array([0.0, 0.0, 1.0], dtype=float)
        x_dir = np.cross(ref, direction)
    else:
        x_dir = np.array([1.0, 0.0, 0.0], dtype=float)
        x_dir = x_dir - float(np.dot(x_dir, direction)) * direction

    x_norm = float(np.linalg.norm(x_dir))
    if x_norm < 1e-12:
        x_dir = np.array([0.0, 1.0, 0.0], dtype=float)
    else:
        x_dir = x_dir / x_norm

    return b3d.Plane(
        origin=(float(p0_mm[0]), float(p0_mm[1]), float(p0_mm[2])),
        x_dir=(float(x_dir[0]), float(x_dir[1]), float(x_dir[2])),
        z_dir=(float(direction[0]), float(direction[1]), float(direction[2])),
    )


def _set_color_safe(b3d: Any, shape: Any, color_hex: str) -> None:
    try:
        shape.color = b3d.Color(color_hex)
    except Exception:
        pass


def build_assembly_brep(assembly: AssemblyModel) -> Any:
    """Construct a build123d Compound (units: mm) of the physical printer assembly."""
    b3d = _require_build123d()
    nmap = assembly.node_map
    children: list[Any] = []

    for member in assembly.members:
        # Skip virtual FEA shear-diaphragm links used to model enclosure panels
        if member.id.startswith("panel_shear_"):
            continue
        p0_mm = nmap[member.start_id].position_m * 1000.0
        p1_mm = nmap[member.end_id].position_m * 1000.0
        vec = p1_mm - p0_mm
        length = float(np.linalg.norm(vec))
        if length < 1e-6:
            continue
        direction = vec / length
        plane = _member_plane(b3d, p0_mm, direction)

        width_mm = float(member.profile.width_m * 1000.0)
        height_mm = float(member.profile.height_m * 1000.0)

        if member.profile.shape == "rod":
            solid = b3d.Solid.make_cylinder(
                radius=max(width_mm * 0.5, 0.5),
                height=length,
                plane=plane,
            )
        else:
            rect = b3d.Face.make_rect(
                width=max(width_mm, 1.0),
                height=max(height_mm, 1.0),
                plane=plane,
            )
            solid = b3d.Solid.extrude(
                rect,
                direction=(float(vec[0]), float(vec[1]), float(vec[2])),
            )

        solid.label = member.id
        _set_color_safe(b3d, solid, member.material.color_hex)
        children.append(solid)

    # Print bed plate (top surface at bed_center_m.z)
    bx, by = (float(assembly.spec.bed.size_mm[0]), float(assembly.spec.bed.size_mm[1]))
    bt = float(assembly.spec.bed.plate_thickness_mm)
    bed_pos = assembly.bed_center_m * 1000.0
    bed_origin = (
        float(bed_pos[0]) - bx * 0.5,
        float(bed_pos[1]) - by * 0.5,
        float(bed_pos[2]) - bt,
    )
    bed_solid = b3d.Solid.make_box(bx, by, bt, plane=b3d.Plane(origin=bed_origin))
    bed_solid.label = "print_bed"
    _set_color_safe(b3d, bed_solid, "#3b4252")
    children.append(bed_solid)

    # Toolhead bounding block
    if assembly.toolhead_node_id in nmap:
        th_pos = nmap[assembly.toolhead_node_id].position_m * 1000.0
        ex, ey, ez = (float(v) for v in assembly.spec.toolhead.envelope_mm)
        th_origin = (
            float(th_pos[0]) - ex * 0.5,
            float(th_pos[1]) - ey * 0.5,
            float(th_pos[2]) - ez * 0.75,
        )
        th_solid = b3d.Solid.make_box(ex, ey, ez, plane=b3d.Plane(origin=th_origin))
        th_solid.label = "toolhead"
        _set_color_safe(b3d, th_solid, "#e63946")
        children.append(th_solid)

    asm_label = (assembly.spec.name or "printsim_assembly").strip()
    return b3d.Compound(label=asm_label, children=children)


def export_assembly_step(assembly: AssemblyModel, output_path: str | Path) -> Path:
    """Export the full physical printer assembly to an ISO-10303-21 STEP file (units: mm)."""
    b3d = _require_build123d()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    compound = build_assembly_brep(assembly)
    b3d.export_step(compound, str(out))
    return out


def export_assembly_step_bytes(assembly: AssemblyModel) -> bytes:
    """Export the full physical printer assembly to STEP format and return the bytes."""
    with tempfile.TemporaryDirectory(prefix="printsim_asm_step_") as tmpdir:
        out = export_assembly_step(assembly, Path(tmpdir) / "printer_assembly.step")
        return out.read_bytes()
