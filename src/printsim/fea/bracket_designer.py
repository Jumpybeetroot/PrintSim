"""Parametric 3D printer bracket designer & fast Hex8 / conformal Tet10 FEA evaluator.

Designed for interactive AI agent discovery loops: given high-level geometric parameters
(arm length, height, wall thickness, gusset rib thickness, bolt hole diameter, fillet radius)
and material/load inputs, constructs both a fast deterministic 3D Hex8 voxel model (~0.1s)
and an exact OpenCASCADE B-Rep Solid via ``build123d`` (for conformal Tet10 FEA and STEP/STL CAD).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Literal

import numpy as np

from printsim.fea.solid_fem import FaceName, solve_solid_voxel_fea
from printsim.fea.tet_fem import solve_step_bytes_tet_fea
from printsim.fea.topology import voxel_grid_to_trimesh
from printsim.schema import get_material

BracketType = Literal["corner_gusset", "motor_mount", "xy_joint_l_bracket"]
BracketEngine = Literal["hex8", "tet10"]

_VALID_BRACKET_TYPES: tuple[BracketType, ...] = (
    "corner_gusset",
    "motor_mount",
    "xy_joint_l_bracket",
)
_VALID_ENGINES: tuple[BracketEngine, ...] = ("hex8", "tet10")


def _validate_bracket_inputs(
    bracket_type: BracketType,
    arm_length_mm: float,
    height_mm: float,
    wall_thickness_mm: float,
    rib_thickness_mm: float,
    bolt_hole_mm: float,
    fillet_mm: float,
    force_n: tuple[float, float, float],
    engine: BracketEngine,
) -> None:
    if bracket_type not in _VALID_BRACKET_TYPES:
        raise ValueError(
            f"Unknown bracket_type '{bracket_type}'. Expected one of {_VALID_BRACKET_TYPES}."
        )
    if engine not in _VALID_ENGINES:
        raise ValueError(f"Unknown engine '{engine}'. Expected one of {_VALID_ENGINES}.")
    if not (25.0 <= arm_length_mm <= 90.0):
        raise ValueError(f"arm_length_mm must be within [25, 90], got {arm_length_mm}")
    if not (10.0 <= height_mm <= 60.0):
        raise ValueError(f"height_mm must be within [10, 60], got {height_mm}")
    if not (2.5 <= wall_thickness_mm <= 18.0):
        raise ValueError(f"wall_thickness_mm must be within [2.5, 18], got {wall_thickness_mm}")
    if not (0.0 <= rib_thickness_mm <= 18.0):
        raise ValueError(f"rib_thickness_mm must be within [0, 18], got {rib_thickness_mm}")
    if not (0.0 <= bolt_hole_mm <= 10.0):
        raise ValueError(f"bolt_hole_mm must be within [0, 10], got {bolt_hole_mm}")
    if not (0.0 <= fillet_mm <= 6.0):
        raise ValueError(f"fillet_mm must be within [0, 6], got {fillet_mm}")
    if len(force_n) != 3:
        raise ValueError("force_n must be a 3-element [Fx, Fy, Fz] vector in Newtons")
    if float(np.linalg.norm(np.asarray(force_n, dtype=float))) <= 1e-9:
        raise ValueError("force_n must be a non-zero 3D force vector")


def _bracket_layout_dims(
    bracket_type: BracketType,
    arm_length_mm: float,
    height_mm: float,
    wall_thickness_mm: float,
    rib_thickness_mm: float,
    bolt_hole_mm: float,
    fillet_mm: float,
) -> dict[str, float]:
    """Compute consistent geometric dimensions (mm) shared by voxel and B-Rep builders."""
    H = float(height_mm)
    w = float(wall_thickness_mm)
    r = float(rib_thickness_mm)
    d_eff = min(float(bolt_hole_mm), max(0.0, H - 3.0)) if bolt_hole_mm > 0.0 else 0.0

    if bracket_type == "corner_gusset":
        Lx = float(arm_length_mm)
        Ly = float(arm_length_mm)
        p0 = 0.25 * w
        px_end = min(Lx - 1.5, w + 0.62 * max(Lx - w, 0.0))
        py_end = min(Ly - 1.5, w + 0.62 * max(Ly - w, 0.0))
        r_eff = min(r, H) if r > 0.0 else 0.0
        hole_pos = min(Lx - d_eff * 0.5 - 1.8, max(w + d_eff * 0.5 + 1.5, Lx - max(6.5, 0.18 * Lx)))
        fillet_eff = min(float(fillet_mm), w * 0.45, max(0.0, Lx - w - 1.5))
        return {
            "Lx": Lx,
            "Ly": Ly,
            "Lz": H,
            "w": w,
            "r_eff": r_eff,
            "p0": p0,
            "px_end": px_end,
            "py_end": py_end,
            "d_eff": d_eff,
            "hole_x": hole_pos,
            "hole_y": hole_pos,
            "fillet_eff": fillet_eff,
        }

    if bracket_type == "motor_mount":
        Lx = max(52.0, float(arm_length_mm) * 1.35)
        Ly = max(46.0, float(arm_length_mm) * 1.20)
        plate_t = min(w, H * 0.72)
        flange_t = min(w, Ly * 0.35)
        rib_t = min(r, Lx * 0.24) if r > 0.0 else 0.0
        cx = 0.5 * Lx
        cy = min(Ly - 14.5, max(flange_t + 14.0, 0.58 * Ly))
        pilot_r = 11.1
        fillet_eff = min(float(fillet_mm), flange_t * 0.45, plate_t * 0.45)
        return {
            "Lx": Lx,
            "Ly": Ly,
            "Lz": H,
            "plate_t": plate_t,
            "flange_t": flange_t,
            "rib_t": rib_t,
            "cx": cx,
            "cy": cy,
            "pilot_r": pilot_r,
            "d_eff": d_eff,
            "fillet_eff": fillet_eff,
        }

    # xy_joint_l_bracket
    Lx = float(arm_length_mm) * 1.25
    Ly = float(arm_length_mm)
    p0 = 0.25 * w
    px_end = min(Lx - 1.5, w + 0.65 * max(Lx - w, 0.0))
    py_end = min(Ly - 1.5, w + 0.65 * max(Ly - w, 0.0))
    r_eff = min(r, H) if r > 0.0 else 0.0
    hx1 = min(Lx - d_eff * 0.5 - 1.8, max(w + d_eff * 0.5 + 1.5, 0.68 * Lx))
    hx2 = min(Lx - d_eff * 0.5 - 1.8, max(w + d_eff * 0.5 + 1.5, 0.86 * Lx))
    hy = min(Ly - d_eff * 0.5 - 1.8, max(w + d_eff * 0.5 + 1.5, 0.78 * Ly))
    fillet_eff = min(float(fillet_mm), w * 0.45, max(0.0, Ly - w - 1.5))
    return {
        "Lx": Lx,
        "Ly": Ly,
        "Lz": H,
        "w": w,
        "r_eff": r_eff,
        "p0": p0,
        "px_end": px_end,
        "py_end": py_end,
        "d_eff": d_eff,
        "hx1": hx1,
        "hx2": hx2,
        "hy": hy,
        "fillet_eff": fillet_eff,
    }


def _choose_voxel_pitch_mm(Lx: float, Ly: float, Lz: float) -> float:
    """Choose deterministic cubic voxel size h_mm purely from bounding-box dims (~0.08-0.18s solve)."""
    max_dim = max(Lx, Ly, Lz)
    h_box = ((Lx * Ly * Lz) / 4200.0) ** (1.0 / 3.0)
    h_mm = max(max_dim / 20.0, h_box)
    # Keep at least 4 voxels along the thinnest bounding dimension when total cells stay reasonable
    min_dim = min(Lx, Ly, Lz)
    if (Lx / (min_dim / 4.0)) * (Ly / (min_dim / 4.0)) * (Lz / (min_dim / 4.0)) <= 7500.0:
        h_mm = min(h_mm, min_dim / 4.0)
    return float(max(1.25, h_mm))


def _build_bracket_voxel_grid(
    bracket_type: BracketType,
    dims: dict[str, float],
) -> tuple[np.ndarray, float, FaceName, FaceName]:
    """Construct exact 3D boolean voxel occupancy grid + (voxel_size_mm, fixed_face, load_face)."""
    Lx, Ly, Lz = dims["Lx"], dims["Ly"], dims["Lz"]
    h_mm = _choose_voxel_pitch_mm(Lx, Ly, Lz)

    nx = max(6, int(round(Lx / h_mm)))
    ny = max(6, int(round(Ly / h_mm)))
    nz = max(4, int(round(Lz / h_mm)))
    # Re-scale h slightly so grid spans [0..Lx] evenly along max axis while keeping cubic voxels
    hx = Lx / nx
    hy = Ly / ny
    hz = Lz / nz
    # Cubic voxels use h_mm; map normalized coordinates into [0..Lx], [0..Ly], [0..Lz]
    xc = (np.arange(nx, dtype=float) + 0.5) * hx
    yc = (np.arange(ny, dtype=float) + 0.5) * hy
    zc = (np.arange(nz, dtype=float) + 0.5) * hz
    X, Y, Z = np.meshgrid(xc, yc, zc, indexing="ij")

    grid = np.zeros((nx, ny, nz), dtype=bool)
    fixed_face: FaceName = "min_y"
    load_face: FaceName = "max_y"

    if bracket_type == "corner_gusset":
        w = dims["w"]
        w_eff = max(w, hx * 1.05, hy * 1.05)
        grid |= Y <= w_eff
        grid |= X <= w_eff

        # Concave root fillet at (w_eff, w_eff)
        rf = dims["fillet_eff"]
        if rf > 0.25 * max(hx, hy):
            in_fillet_box = (X >= w_eff) & (X <= w_eff + rf) & (Y >= w_eff) & (Y <= w_eff + rf)
            outside_arc = ((w_eff + rf - X) ** 2 + (w_eff + rf - Y) ** 2) >= (rf**2)
            grid |= in_fillet_box & outside_arc

        # Triangular diagonal stiffening rib centered at Z = Lz/2
        r_eff = dims["r_eff"]
        if r_eff > 0.0:
            p0 = dims["p0"]
            px_end = dims["px_end"]
            py_end = dims["py_end"]
            tri_x = max(px_end - p0, 1e-6)
            tri_y = max(py_end - p0, 1e-6)
            in_tri = (X >= p0) & (Y >= p0) & (((X - p0) / tri_x + (Y - p0) / tri_y) <= 1.0)
            half_z = max(0.5 * r_eff, 0.55 * hz)
            in_z = np.abs(Z - 0.5 * Lz) <= half_z
            grid |= in_tri & in_z

        # Bolt clearance holes on arms
        d_eff = dims["d_eff"]
        if d_eff > 0.5:
            rad = 0.5 * d_eff
            z_mid = 0.5 * Lz
            hole_x_arm = (Y <= w_eff) & (((X - dims["hole_x"]) ** 2 + (Z - z_mid) ** 2) <= rad**2)
            hole_y_arm = (X <= w_eff) & (((Y - dims["hole_y"]) ** 2 + (Z - z_mid) ** 2) <= rad**2)
            grid &= ~(hole_x_arm | hole_y_arm)

    elif bracket_type == "motor_mount":
        plate_t = max(dims["plate_t"], hz * 1.05)
        flange_t = max(dims["flange_t"], hy * 1.05)
        rib_t = dims["rib_t"]

        # Mid-plane motor deck (symmetric shear center -> zero twist under lateral X load) & rear flange
        grid |= np.abs(Z - 0.5 * Lz) <= 0.5 * plate_t
        grid |= Y <= flange_t

        # Symmetric upper & lower triangular side gussets (when rib_thickness_mm > 0)
        if rib_t > 0.0:
            rib_eff = max(rib_t, hx * 1.05)
            y0 = 0.5 * flange_t
            half_span_z = 0.5 * Lz - (0.5 * Lz - 0.5 * plate_t) * np.clip(
                (Y - y0) / max(Ly - y0, 1e-6), 0.0, 1.0
            )
            in_gusset_profile = (Y >= y0) & (np.abs(Z - 0.5 * Lz) <= half_span_z)
            grid |= ((X <= rib_eff) | (X >= (Lx - rib_eff))) & in_gusset_profile

        # Central NEMA17 pilot bore cutout through motor deck
        pilot_r = dims["pilot_r"]
        cx, cy = dims["cx"], dims["cy"]
        grid &= ~(((X - cx) ** 2 + (Y - cy) ** 2) <= pilot_r**2)

        # Rear M5 mounting bolt holes (symmetric about Z = Lz/2)
        d_eff = dims["d_eff"]
        if d_eff > 0.5:
            rad = 0.5 * d_eff
            z_hole = 0.5 * Lz
            for bx in (0.25 * Lx, 0.75 * Lx):
                grid &= ~((Y <= flange_t) & (((X - bx) ** 2 + (Z - z_hole) ** 2) <= rad**2))

    else:  # xy_joint_l_bracket
        w = dims["w"]
        w_eff = max(w, hx * 1.05, hy * 1.05)
        grid |= Y <= w_eff
        grid |= X <= w_eff

        rf = dims["fillet_eff"]
        if rf > 0.25 * max(hx, hy):
            in_fillet_box = (X >= w_eff) & (X <= w_eff + rf) & (Y >= w_eff) & (Y <= w_eff + rf)
            outside_arc = ((w_eff + rf - X) ** 2 + (w_eff + rf - Y) ** 2) >= (rf**2)
            grid |= in_fillet_box & outside_arc

        # Bottom-flush triangular gusset web (ideal flat print-bed orientation)
        r_eff = dims["r_eff"]
        if r_eff > 0.0:
            p0 = dims["p0"]
            px_end = dims["px_end"]
            py_end = dims["py_end"]
            tri_x = max(px_end - p0, 1e-6)
            tri_y = max(py_end - p0, 1e-6)
            in_tri = (X >= p0) & (Y >= p0) & (((X - p0) / tri_x + (Y - p0) / tri_y) <= 1.0)
            z_rib = max(r_eff, hz * 1.05)
            grid |= in_tri & (Z <= z_rib)

        d_eff = dims["d_eff"]
        if d_eff > 0.5:
            rad = 0.5 * d_eff
            z_mid = 0.5 * Lz
            for bx in (dims["hx1"], dims["hx2"]):
                grid &= ~((Y <= w_eff) & (((X - bx) ** 2 + (Z - z_mid) ** 2) <= rad**2))
            grid &= ~((X <= w_eff) & (((Y - dims["hy"]) ** 2 + (Z - z_mid) ** 2) <= rad**2))

    return grid, float(h_mm), fixed_face, load_face


def _build_parametric_bracket_brep(
    bracket_type: BracketType,
    dims: dict[str, float],
) -> tuple[Any, Any, FaceName, FaceName]:
    """Construct exact parametric OpenCASCADE B-Rep Solid via build123d.

    Returns ``(b3d_module, solid, fixed_face, load_face)``.
    """
    from printsim.geometry.cad_b3d import (
        _ensure_single_solid,
        _require_build123d,
        _safe_fillet,
        _select_vertical_edges,
    )

    b3d = _require_build123d()
    Lx, Ly, Lz = dims["Lx"], dims["Ly"], dims["Lz"]

    if bracket_type == "corner_gusset":
        w = dims["w"]
        flange_x = b3d.Solid.make_box(Lx, w, Lz, plane=b3d.Plane.XY)
        flange_y = b3d.Solid.make_box(w, Ly, Lz, plane=b3d.Plane.XY)
        parts = [flange_x, flange_y]

        r_eff = dims["r_eff"]
        if r_eff > 0.0:
            z0 = 0.5 * (Lz - r_eff)
            p0 = dims["p0"]
            web_wire = b3d.Wire.make_polygon(
                [(p0, p0, z0), (dims["px_end"], p0, z0), (p0, dims["py_end"], z0)]
            )
            web = b3d.Solid.extrude(b3d.Face(web_wire), (0.0, 0.0, r_eff))
            parts.append(web)

        body = parts[0].fuse(*parts[1:])
        body = _ensure_single_solid(body, "corner_gusset")

        d_eff = dims["d_eff"]
        if d_eff > 0.2:
            rad = 0.5 * d_eff
            z_mid = 0.5 * Lz
            hole_x = b3d.Solid.make_cylinder(
                rad,
                w + 2.0,
                plane=b3d.Plane(origin=(dims["hole_x"], -1.0, z_mid), z_dir=(0.0, 1.0, 0.0)),
            )
            hole_y = b3d.Solid.make_cylinder(
                rad,
                w + 2.0,
                plane=b3d.Plane(origin=(-1.0, dims["hole_y"], z_mid), z_dir=(1.0, 0.0, 0.0)),
            )
            body = body.cut(hole_x, hole_y)
            body = _ensure_single_solid(body, "corner_gusset")

        rf = dims["fillet_eff"]
        if rf > 0.2:
            targets = [(0.0, 0.0), (Lx, w), (w, Ly)]
            if r_eff <= 0.0:
                targets.append((w, w))
            edges = _select_vertical_edges(b3d, body, expected_length=Lz, xy_targets=targets)
            body = _safe_fillet(b3d, body, edges, radius=rf)
        return b3d, body, "min_y", "max_y"

    if bracket_type == "motor_mount":
        plate_t = dims["plate_t"]
        flange_t = dims["flange_t"]
        rib_t = dims["rib_t"]
        z_lo = 0.5 * (Lz - plate_t)
        z_hi = 0.5 * (Lz + plate_t)

        mid_deck = b3d.Solid.make_box(
            Lx, Ly, plate_t, plane=b3d.Plane(origin=(0.0, 0.0, z_lo))
        )
        rear_flange = b3d.Solid.make_box(Lx, flange_t, Lz, plane=b3d.Plane.XY)
        parts = [mid_deck, rear_flange]
        if rib_t > 0.0:
            y0 = 0.5 * flange_t
            left_wire = b3d.Wire.make_polygon(
                [(0.0, y0, 0.0), (0.0, Ly, z_lo), (0.0, Ly, z_hi), (0.0, y0, Lz)]
            )
            right_x = Lx - rib_t
            right_wire = b3d.Wire.make_polygon(
                [
                    (right_x, y0, 0.0),
                    (right_x, Ly, z_lo),
                    (right_x, Ly, z_hi),
                    (right_x, y0, Lz),
                ]
            )
            parts.append(b3d.Solid.extrude(b3d.Face(left_wire), (rib_t, 0.0, 0.0)))
            parts.append(b3d.Solid.extrude(b3d.Face(right_wire), (rib_t, 0.0, 0.0)))

        body = parts[0].fuse(*parts[1:])
        body = _ensure_single_solid(body, "motor_mount")

        cx, cy = dims["cx"], dims["cy"]
        cutters: list[Any] = [
            b3d.Solid.make_cylinder(
                dims["pilot_r"], Lz + 4.0, plane=b3d.Plane(origin=(cx, cy, -2.0))
            )
        ]
        # NEMA17 M3 31x31mm bolt pattern where clear of side walls and rear flange
        for dx in (-15.5, 15.5):
            for dy in (-15.5, 15.5):
                hx, hy = cx + dx, cy + dy
                if (rib_t + 2.5) < hx < (Lx - rib_t - 2.5) and (flange_t + 2.5) < hy < (Ly - 2.5):
                    cutters.append(
                        b3d.Solid.make_cylinder(
                            1.7, Lz + 4.0, plane=b3d.Plane(origin=(hx, hy, -2.0))
                        )
                    )

        d_eff = dims["d_eff"]
        if d_eff > 0.2:
            rad = 0.5 * d_eff
            z_hole = 0.5 * Lz
            for bx in (0.25 * Lx, 0.75 * Lx):
                cutters.append(
                    b3d.Solid.make_cylinder(
                        rad,
                        flange_t + 2.0,
                        plane=b3d.Plane(origin=(bx, -1.0, z_hole), z_dir=(0.0, 1.0, 0.0)),
                    )
                )

        body = body.cut(*cutters)
        body = _ensure_single_solid(body, "motor_mount")

        rf = dims["fillet_eff"]
        if rf > 0.2:
            edges = _select_vertical_edges(
                b3d, body, expected_length=plate_t, xy_targets=[(0.0, Ly), (Lx, Ly)]
            )
            body = _safe_fillet(b3d, body, edges, radius=rf)
        return b3d, body, "min_y", "max_y"

    # xy_joint_l_bracket
    w = dims["w"]
    flange_x = b3d.Solid.make_box(Lx, w, Lz, plane=b3d.Plane.XY)
    flange_y = b3d.Solid.make_box(w, Ly, Lz, plane=b3d.Plane.XY)
    parts = [flange_x, flange_y]

    r_eff = dims["r_eff"]
    if r_eff > 0.0:
        p0 = dims["p0"]
        web_wire = b3d.Wire.make_polygon(
            [(p0, p0, 0.0), (dims["px_end"], p0, 0.0), (p0, dims["py_end"], 0.0)]
        )
        web = b3d.Solid.extrude(b3d.Face(web_wire), (0.0, 0.0, r_eff))
        parts.append(web)

    body = parts[0].fuse(*parts[1:])
    body = _ensure_single_solid(body, "xy_joint_l_bracket")

    d_eff = dims["d_eff"]
    if d_eff > 0.2:
        rad = 0.5 * d_eff
        z_mid = 0.5 * Lz
        hole_cutters = [
            b3d.Solid.make_cylinder(
                rad,
                w + 2.0,
                plane=b3d.Plane(origin=(bx, -1.0, z_mid), z_dir=(0.0, 1.0, 0.0)),
            )
            for bx in (dims["hx1"], dims["hx2"])
        ]
        hole_cutters.append(
            b3d.Solid.make_cylinder(
                rad,
                w + 2.0,
                plane=b3d.Plane(origin=(-1.0, dims["hy"], z_mid), z_dir=(1.0, 0.0, 0.0)),
            )
        )
        body = body.cut(*hole_cutters)
        body = _ensure_single_solid(body, "xy_joint_l_bracket")

    rf = dims["fillet_eff"]
    if rf > 0.2:
        targets = [(0.0, 0.0), (Lx, w), (w, Ly)]
        if r_eff <= 0.0:
            targets.append((w, w))
        edges = _select_vertical_edges(b3d, body, expected_length=Lz, xy_targets=targets)
        body = _safe_fillet(b3d, body, edges, radius=rf)
    return b3d, body, "min_y", "max_y"


def design_and_evaluate_bracket(
    bracket_type: BracketType = "corner_gusset",
    arm_length_mm: float = 40.0,
    height_mm: float = 20.0,
    wall_thickness_mm: float = 6.0,
    rib_thickness_mm: float = 6.0,
    bolt_hole_mm: float = 5.5,
    fillet_mm: float = 2.0,
    material_id: str = "ppa_cf_printed",
    force_n: tuple[float, float, float] = (60.0, 0.0, 0.0),
    fdm_anisotropic: bool | None = None,
    engine: BracketEngine = "hex8",
    output_stl_path: str | Path | None = None,
    output_step_path: str | Path | None = None,
) -> dict[str, Any]:
    """Parametrically generate a structural bracket, run 3D Hex8 or Tet10 FEA, and optionally export STL/STEP."""
    _validate_bracket_inputs(
        bracket_type=bracket_type,
        arm_length_mm=arm_length_mm,
        height_mm=height_mm,
        wall_thickness_mm=wall_thickness_mm,
        rib_thickness_mm=rib_thickness_mm,
        bolt_hole_mm=bolt_hole_mm,
        fillet_mm=fillet_mm,
        force_n=force_n,
        engine=engine,
    )
    mat = get_material(material_id)
    is_fdm = fdm_anisotropic if fdm_anisotropic is not None else ("printed" in material_id)
    f_vec = (float(force_n[0]), float(force_n[1]), float(force_n[2]))
    f_mag = float(np.linalg.norm(np.asarray(f_vec, dtype=float)))

    dims = _bracket_layout_dims(
        bracket_type=bracket_type,
        arm_length_mm=arm_length_mm,
        height_mm=height_mm,
        wall_thickness_mm=wall_thickness_mm,
        rib_thickness_mm=rib_thickness_mm,
        bolt_hole_mm=bolt_hole_mm,
        fillet_mm=fillet_mm,
    )

    brep_tuple: tuple[Any, Any, FaceName, FaceName] | None = None
    if engine == "tet10" or output_step_path is not None:
        brep_tuple = _build_parametric_bracket_brep(bracket_type, dims)

    exported_step: str | None = None
    step_bytes: bytes | None = None
    if brep_tuple is not None:
        b3d, solid, brep_fixed, brep_load = brep_tuple
        if output_step_path is not None:
            step_out = Path(output_step_path)
            step_out.parent.mkdir(parents=True, exist_ok=True)
            b3d.export_step(solid, str(step_out))
            exported_step = str(step_out.resolve())
            if engine == "tet10":
                step_bytes = step_out.read_bytes()
        elif engine == "tet10":
            with tempfile.TemporaryDirectory(prefix="printsim_bkt_step_") as tmpdir:
                tmp_step = Path(tmpdir) / f"{bracket_type}.step"
                b3d.export_step(solid, str(tmp_step))
                step_bytes = tmp_step.read_bytes()

    voxel_grid: np.ndarray | None = None
    voxel_h_mm: float = 0.0

    if engine == "tet10":
        assert brep_tuple is not None and step_bytes is not None
        _, _, brep_fixed, brep_load = brep_tuple
        tet_res = solve_step_bytes_tet_fea(
            step_bytes=step_bytes,
            material_id=material_id,
            fixed_face=brep_fixed,
            load_face=brep_load,
            force_n=f_vec,
            order=2,
            part_name=f"Parametric {bracket_type.replace('_', ' ').title()}",
            fdm_anisotropic=is_fdm,
        )
        mass_g = float(tet_res.mass_g)
        volume_cm3 = float(tet_res.volume_mm3) * 1e-3
        max_disp_um = float(tet_res.max_displacement_um)
        max_vm_mpa = float(tet_res.max_von_mises_mpa)
        safety_factor = float(tet_res.safety_factor)
        failure_mode = str(tet_res.failure_mode)
        num_elements = int(tet_res.num_elements)
        num_dofs = int(tet_res.num_dofs)
    else:
        voxel_grid, voxel_h_mm, vox_fixed, vox_load = _build_bracket_voxel_grid(bracket_type, dims)
        hex_res = solve_solid_voxel_fea(
            grid=voxel_grid,
            voxel_size_mm=voxel_h_mm,
            material_id=material_id,
            fixed_face=vox_fixed,
            load_face=vox_load,
            force_n=f_vec,
            part_name=f"Parametric {bracket_type.replace('_', ' ').title()}",
            fdm_anisotropic=is_fdm,
        )
        mass_g = float(hex_res.mass_g)
        # Convert mass to exact material volume (cm^3) -> mass_g / (density_kg_m3 * 1e-3)
        volume_cm3 = mass_g / max(mat.density_kg_m3 * 1e-3, 1e-9)
        max_disp_um = float(hex_res.max_displacement_um)
        max_vm_mpa = float(hex_res.max_von_mises_mpa)
        safety_factor = float(hex_res.safety_factor)
        failure_mode = str(hex_res.failure_mode)
        num_elements = int(hex_res.num_elements)
        num_dofs = int(hex_res.num_dofs)

    exported_stl: str | None = None
    if output_stl_path is not None:
        stl_out = Path(output_stl_path)
        stl_out.parent.mkdir(parents=True, exist_ok=True)
        if brep_tuple is not None:
            b3d, solid, _, _ = brep_tuple
            b3d.export_stl(
                solid,
                str(stl_out),
                tolerance=0.02,
                angular_tolerance=0.1,
                ascii_format=False,
            )
        else:
            assert voxel_grid is not None
            mesh = voxel_grid_to_trimesh(voxel_grid, voxel_size_mm=voxel_h_mm, smooth_iterations=0)
            mesh.export(str(stl_out))
        exported_stl = str(stl_out.resolve())

    stiffness_n_per_mm = f_mag / max(max_disp_um * 1e-3, 1e-9)
    specific_stiffness = stiffness_n_per_mm / max(mass_g, 0.1)

    return {
        "bracket_type": bracket_type,
        "params_mm": {
            "arm_length_mm": round(float(arm_length_mm), 2),
            "height_mm": round(float(height_mm), 2),
            "wall_thickness_mm": round(float(wall_thickness_mm), 2),
            "rib_thickness_mm": round(float(rib_thickness_mm), 2),
            "bolt_hole_mm": round(float(bolt_hole_mm), 2),
            "fillet_mm": round(float(fillet_mm), 2),
        },
        "material_id": material_id,
        "engine": engine,
        "mass_g": round(mass_g, 2),
        "volume_cm3": round(volume_cm3, 3),
        "max_displacement_um": round(max_disp_um, 2),
        "stiffness_n_per_mm": round(stiffness_n_per_mm, 2),
        "specific_stiffness_n_per_mm_per_g": round(specific_stiffness, 2),
        "max_von_mises_mpa": round(max_vm_mpa, 3),
        "safety_factor": round(min(safety_factor, 999.0), 2),
        "failure_mode": failure_mode,
        "fdm_anisotropic": bool(is_fdm),
        "num_elements": num_elements,
        "num_dofs": num_dofs,
        "exported_stl_path": exported_stl,
        "exported_step_path": exported_step,
    }
