#!/usr/bin/env python3
"""3D Tet10 Continuum FEA Calibration for Timing-Belt Single Pitch Cells (2GT, 3GT, AT3).

Constructs 3D periodic unit cells (one tooth pitch p along X, one cord pitch p_c along Z,
full belt thickness H along Y) using ``build123d`` OpenCASCADE B-Rep modeling, imports and
conformally fragments the elastomer body and embedded tensile cord into ``gmsh``, and solves
3D linear elasticity with ``scikit-fem`` Tet10 (P2) elements.

Physics extracted per belt family (``2GT``, ``3GT``, ``AT3``):
  1. **Tooth-mesh shear stiffness ``k_t(q0)`` (N/(m*mm)) & radial flank seating**:
     - Step 1: Iterated active-set unilateral normal contact of the belt tooth seated into a
       rigid conjugate pulley groove (nominal 20T pulley, radial wrap pressure p_r = q0 / R)
       across line tensions q0 in [0.5, 1.0, 2.0, 4.0, 7.0, 12.0] N/mm.
     - Step 2: Small-amplitude tangential shear perturbation of the embedded cord cylinder
       holding the converged active driving-flank and seated root-land contact patch,
       extracting exact continuum tooth-mesh shear stiffness k_t(q0) per tooth per mm width.
  2. **Composite bending rigidity ``EI_b(r_m)`` (N*m^2 per m width) & energy partition**:
     - Exact 2x2 ABD static condensation (axial extension + pure curvature load cases) via
       Clapeyron strain-energy inner products: EI_eff = D - B^2 / A, eliminating neutral-axis
       assumptions and boundary-traction noise.
     - Sweeps matrix elastomer modulus retention r_m = E_matrix(T) / E_matrix(25 C) across
       [1.0, 0.8, 0.6, 0.4, 0.25] and partitions bending strain energy between cord and matrix.
  3. **Analytical & mesh-convergence verification**:
     - Plane-strain homogeneous rectangular bar benchmark (exact E*h^3 / (12*(1 - nu^2))).
     - Coarse vs. fine Tet10 mesh-convergence check on both k_t and EI_b for every profile.

Outputs calibrated constants to ``data/touchstones/belt_pitch_cell_fea.json``.

Run:
    uv run python scripts/calibrate_belt_pitch_cell_fea.py
    uv run python scripts/calibrate_belt_pitch_cell_fea.py --profiles 2gt
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# 1. Belt Pitch-Cell Geometry & Material Specifications (Datasheet-Backed)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BeltPitchCellSpec:
    """First-principles geometric and constitutive parameters for one periodic belt pitch cell."""

    profile_id: str
    label: str
    tooth_shape: str  # "curvilinear" (2GT/3GT) or "trapezoidal" (AT3)
    pitch_mm: float
    total_height_mm: float
    tooth_height_mm: float
    pld_mm: float  # Pitch Line Differential above tooth root
    tooth_root_width_mm: float
    tooth_tip_width_mm: float
    cord_pitch_mm: float
    cord_diameter_mm: float
    cord_material: str  # "fiberglass" or "steel"
    matrix_material: str  # "epdm" or "tpu_92a"
    # Elastomer constitutive properties at 25 C (small-amplitude 40-150 Hz dynamic modulus)
    e_matrix_25c_pa: float
    nu_matrix: float
    # Tooth-facing nylon fabric skin factor (1.0 for bare TPU AT3; ~1.12 for Gates woven nylon)
    facing_shear_boost: float
    # Stranded cord micromechanical parameters (for Costello + shear-lag flexural modulus)
    e_filament_pa: float
    nu_filament: float
    filament_diameter_mm: float
    n_filaments_per_cord: int
    helix_angle_deg: float
    interfilament_g_pa: float  # RFL dip shear modulus (glass) or TPU matrix shear modulus (steel)
    interfilament_softens_with_matrix: bool
    # Rigid pulley groove mating clearances (mm) and nominal pulley tooth count
    nominal_pulley_teeth: int
    flank_clearance_max_mm: float  # Clearance at tooth tip/root extremes before radial seating
    land_clearance_mm: float  # Initial radial land seating clearance (roughness/fabric bedding)
    flank_friction_mu: float  # Static/micro-slip friction on active pulley contact
    # Mesh sizing (mm)
    h_coarse_mm: float
    h_fine_mm: float


BELT_CELL_SPECS: dict[str, BeltPitchCellSpec] = {
    "2gt": BeltPitchCellSpec(
        profile_id="2gt",
        label="Gates PowerGrip 2GT (2 mm pitch, EPDM + E-glass 135 tex)",
        tooth_shape="curvilinear",
        pitch_mm=2.00,
        total_height_mm=1.38,
        tooth_height_mm=0.75,
        pld_mm=0.254,
        tooth_root_width_mm=1.36,
        tooth_tip_width_mm=0.62,
        cord_pitch_mm=0.82,
        cord_diameter_mm=0.30,
        cord_material="fiberglass",
        matrix_material="epdm",
        # 75-80 ShA EPDM/CR dynamic modulus at 40-150 Hz (Gent ~7.5 MPa static x ~1.6 Payne/visco)
        e_matrix_25c_pa=12.0e6,
        nu_matrix=0.47,
        facing_shear_boost=1.12,
        # NGF E-glass 135 tex cord: A_glass = 0.0529 mm^2 (~830 x 9 um G-filaments), alpha ~ 16 deg
        e_filament_pa=72.0e9,
        nu_filament=0.22,
        filament_diameter_mm=0.009,
        n_filaments_per_cord=830,
        helix_angle_deg=16.0,
        # Cured RFL thermoset dip (E_RFL ~ 120 MPa -> G_RFL ~ 45 MPa, Tg > 110 C)
        interfilament_g_pa=45.0e6,
        interfilament_softens_with_matrix=False,
        nominal_pulley_teeth=20,
        flank_clearance_max_mm=0.028,
        land_clearance_mm=0.014,
        flank_friction_mu=0.35,
        h_coarse_mm=0.30,
        h_fine_mm=0.21,
    ),
    "3gt": BeltPitchCellSpec(
        profile_id="3gt",
        label="Gates PowerGrip 3GT (3 mm pitch, EPDM + E-glass 270 tex)",
        tooth_shape="curvilinear",
        pitch_mm=3.00,
        total_height_mm=2.41,
        tooth_height_mm=1.14,
        pld_mm=0.381,
        tooth_root_width_mm=2.04,
        tooth_tip_width_mm=0.94,
        cord_pitch_mm=1.00,
        cord_diameter_mm=0.44,
        cord_material="fiberglass",
        matrix_material="epdm",
        # Slightly firmer 78-80 ShA EPDM compound used in 3MGT belts
        e_matrix_25c_pa=14.0e6,
        nu_matrix=0.47,
        facing_shear_boost=1.12,
        # Heavier 3GT E-glass cord (~270 tex -> ~1500 x 9 um filaments, alpha ~ 16 deg)
        e_filament_pa=72.0e9,
        nu_filament=0.22,
        filament_diameter_mm=0.009,
        n_filaments_per_cord=1500,
        helix_angle_deg=16.0,
        interfilament_g_pa=50.0e6,
        interfilament_softens_with_matrix=False,
        nominal_pulley_teeth=20,
        flank_clearance_max_mm=0.040,
        land_clearance_mm=0.020,
        flank_friction_mu=0.35,
        h_coarse_mm=0.45,
        h_fine_mm=0.31,
    ),
    "at3": BeltPitchCellSpec(
        profile_id="at3",
        label="Breco/Elatech AT3 (3 mm pitch, 92 ShA TPU + Steel 7x3 cord)",
        tooth_shape="trapezoidal",
        pitch_mm=3.00,
        total_height_mm=1.90,
        tooth_height_mm=1.10,
        pld_mm=0.300,
        tooth_root_width_mm=1.95,
        tooth_tip_width_mm=0.92,
        cord_pitch_mm=1.00,
        cord_diameter_mm=0.30,
        cord_material="steel",
        matrix_material="tpu_92a",
        # 92 ShA TPU (BASF Elastollan / Desmopan 92A dynamic modulus ~42 MPa at 25 C)
        e_matrix_25c_pa=42.0e6,
        nu_matrix=0.45,
        facing_shear_boost=1.00,
        # Steel 7x3 wire strand (21 x 0.053 mm galvanized steel wires, A_f = 0.0463 mm^2, alpha ~ 11 deg)
        e_filament_pa=205.0e9,
        nu_filament=0.29,
        filament_diameter_mm=0.053,
        n_filaments_per_cord=21,
        helix_angle_deg=11.0,
        # Bare steel wires embedded directly in 92 ShA TPU (G_TPU = 42 / (2 * 1.45) = 14.5 MPa)
        interfilament_g_pa=14.5e6,
        interfilament_softens_with_matrix=True,
        nominal_pulley_teeth=20,
        flank_clearance_max_mm=0.032,
        land_clearance_mm=0.016,
        flank_friction_mu=0.45,
        h_coarse_mm=0.43,
        h_fine_mm=0.30,
    ),
}


# ---------------------------------------------------------------------------
# 2. First-Principles Constitutive Tensors & Cord Flexural Homogenization
# ---------------------------------------------------------------------------


def iso_constitutive_3d(e_pa: float, nu: float) -> np.ndarray:
    """Build SPD 6x6 3D isotropic elasticity tensor D (Voigt [xx, yy, zz, xy, yz, xz])."""
    c = e_pa / ((1.0 + nu) * (1.0 - 2.0 * nu))
    d_mat = np.zeros((6, 6), dtype=np.float64)
    d_mat[0:3, 0:3] = c * np.array(
        [
            [1.0 - nu, nu, nu],
            [nu, 1.0 - nu, nu],
            [nu, nu, 1.0 - nu],
        ],
        dtype=np.float64,
    )
    g = e_pa / (2.0 * (1.0 + nu))
    d_mat[3, 3] = d_mat[4, 4] = d_mat[5, 5] = g
    return d_mat


def cord_bundle_effective_bending_modulus_pa(
    spec: BeltPitchCellSpec,
    matrix_retention: float = 1.0,
) -> tuple[float, float]:
    """Compute effective flexural Young's modulus E_c_bend (Pa) of a stranded cord cylinder.

    Uses Costello (1990) helical-wire bending lower bound + Newmark/Papailiou shear-lag
    inter-filament coupling through the RFL sizing (glass) or TPU matrix (steel) over the
    characteristic pulley-wrap bending length L_b = sqrt(R_pulley * h_backing).

    Returns:
        (e_cord_bend_pa, ei_single_cord_nm2)
    """
    r_f = 0.5 * spec.filament_diameter_mm * 1e-3
    r_c = 0.5 * spec.cord_diameter_mm * 1e-3
    alpha = math.radians(spec.helix_angle_deg)
    ca = math.cos(alpha)
    sa = math.sin(alpha)
    n_f = float(spec.n_filaments_per_cord)

    # Costello (1990) Eq. 2.27: unbonded helical filament bundle bending stiffness
    i_fil = math.pi * (r_f**4) / 4.0
    ei_min = n_f * spec.e_filament_pa * i_fil * (2.0 * ca / (2.0 + spec.nu_filament * (sa**2)))

    # Monolithic bonded bundle upper bound (using actual solid filament fill fraction V_f)
    a_fil_tot = n_f * math.pi * (r_f**2)
    a_cord = math.pi * (r_c**2)
    v_f = min(0.85, a_fil_tot / max(a_cord, 1e-18))
    i_cord_geom = math.pi * (r_c**4) / 4.0
    ei_max = spec.e_filament_pa * v_f * i_cord_geom * (ca**3)

    # Shear-lag coupling length across the multi-filament bundle
    g_inter = spec.interfilament_g_pa * (matrix_retention if spec.interfilament_softens_with_matrix else (0.90 + 0.10 * matrix_retention))
    g_inter = max(g_inter, 1.0e5)
    l_shear = r_c * math.sqrt((spec.e_filament_pa * v_f) / (4.0 * g_inter))

    # Characteristic pulley wrap bending length L_b = sqrt(R_pulley * h_back)
    r_pulley = (spec.nominal_pulley_teeth * spec.pitch_mm * 1e-3) / (2.0 * math.pi)
    h_back = (spec.total_height_mm - spec.tooth_height_mm) * 1e-3
    l_bend = math.sqrt(max(r_pulley * h_back, 1e-9))

    phi_bond = 1.0 / (1.0 + (math.pi * l_shear / l_bend) ** 2)
    ei_cord = ei_min + phi_bond * max(ei_max - ei_min, 0.0)
    e_cord_bend = ei_cord / max(i_cord_geom, 1e-24)
    return float(e_cord_bend), float(ei_cord)


def cord_axial_stiffness_modulus_pa(spec: BeltPitchCellSpec) -> float:
    """Effective longitudinal Young's modulus of the cord cylinder in pure axial tension (Pa)."""
    r_f = 0.5 * spec.filament_diameter_mm * 1e-3
    r_c = 0.5 * spec.cord_diameter_mm * 1e-3
    alpha = math.radians(spec.helix_angle_deg)
    a_fil_tot = spec.n_filaments_per_cord * math.pi * (r_f**2)
    a_cord = math.pi * (r_c**2)
    return float(spec.e_filament_pa * (a_fil_tot / a_cord) * (math.cos(alpha) ** 3))


# ---------------------------------------------------------------------------
# 3. build123d 3D Pitch-Cell Geometry -> Conformal Gmsh Tet10 Mesh
# ---------------------------------------------------------------------------


def _belt_profile_points_mm(spec: BeltPitchCellSpec) -> list[tuple[float, float]]:
    """Generate clean, symmetric XY cross-section profile points for one pitch cell (mm).

    Coordinate convention:
      - x in [0, pitch_mm], tooth centered at x = pitch_mm / 2
      - y = 0 at tooth crown (tip), y = tooth_height_mm at root land, y = total_height_mm at back
    """
    p = spec.pitch_mm
    ht = spec.tooth_height_mm
    h_tot = spec.total_height_mm
    b_root = spec.tooth_root_width_mm
    b_tip = spec.tooth_tip_width_mm

    xc = 0.5 * p
    x_rl = xc - 0.5 * b_root
    x_rr = xc + 0.5 * b_root
    x_tl = xc - 0.5 * b_tip
    x_tr = xc + 0.5 * b_tip

    pts: list[tuple[float, float]] = [(0.0, ht), (x_rl, ht)]

    if spec.tooth_shape == "curvilinear":
        # Smooth Gates modified-curvilinear (2GT/3GT) tooth flank & rounded crown
        # Left flank: smooth S-curve with root fillet transitioning into conjugate arc
        dx_flank = x_tl - x_rl
        n_flank = 6
        for i in range(1, n_flank + 1):
            t = i / n_flank
            # Smooth cosine-blended curvilinear flank (steep near pitch line, rounded root/tip)
            s_x = 0.5 * (1.0 - math.cos(math.pi * t))
            x_pt = x_rl + dx_flank * (0.25 * t + 0.75 * s_x)
            y_pt = ht * (1.0 - math.sin(0.5 * math.pi * t) ** 1.15)
            # Keep crown tangent slightly above y=0 for smooth rounded crown arc
            y_pt = max(y_pt, 0.06 * ht * (1.0 - t) + 0.04 * ht) if i < n_flank else 0.04 * ht
            pts.append((x_pt, y_pt))
        # Rounded curvilinear crown
        pts.append((xc - 0.25 * b_tip, 0.010 * ht))
        pts.append((xc, 0.0))
        pts.append((xc + 0.25 * b_tip, 0.010 * ht))
        # Right flank (exact mirror of left flank)
        left_flank = pts[2 : 2 + n_flank]
        for x_pt, y_pt in reversed(left_flank):
            pts.append((p - x_pt, y_pt))
    else:
        # AT3 trapezoidal profile (DIN AT-profile: 50 deg included angle with root/tip fillets)
        r_fil = 0.14 * ht
        # Left root fillet -> straight flank -> left tip fillet
        pts.append((x_rl + 0.05 * b_root, ht - 0.10 * ht))
        pts.append((x_rl + 0.14 * b_root, ht - 0.35 * ht))
        pts.append((x_tl - 0.06 * b_tip, 0.18 * ht))
        pts.append((x_tl + 0.04 * b_tip, 0.02 * ht))
        # Flat AT3 tooth crown
        pts.append((xc, 0.0))
        pts.append((x_tr - 0.04 * b_tip, 0.02 * ht))
        pts.append((x_tr + 0.06 * b_tip, 0.18 * ht))
        pts.append((x_rr - 0.14 * b_root, ht - 0.35 * ht))
        pts.append((x_rr - 0.05 * b_root, ht - 0.10 * ht))
        _ = r_fil

    pts.extend(
        [
            (x_rr, ht),
            (p, ht),
            (p, h_tot),
            (0.0, h_tot),
        ]
    )
    return pts


def _build_pitch_cell_step_files(
    spec: BeltPitchCellSpec,
    tmpdir: Path,
) -> tuple[Path, Path]:
    """Construct exact 3D elastomer body (with cord bore) and cord cylinder using build123d."""
    import build123d as b3d

    pts = _belt_profile_points_mm(spec)
    w_cell = spec.cord_pitch_mm
    y_cord = spec.tooth_height_mm + spec.pld_mm
    r_cord = 0.5 * spec.cord_diameter_mm

    # 1. Belt body extrusion along +Z from 0 to w_cell
    with b3d.BuildPart() as body_builder:
        with b3d.BuildSketch(b3d.Plane.XY):
            with b3d.BuildLine():
                b3d.Polyline(*pts, close=True)
            b3d.make_face()
        b3d.extrude(amount=w_cell)

    # 2. Tensile cord along +X from 0 to pitch_mm, centered at (y=y_cord, z=w_cell/2).
    # Use an inertia-matched regular 12-gon (R_12 = 1.02321 * r_cord: exact I = pi*r^4/4 and
    # area within 0.02% of pi*r^2) so planar facets avoid coarse-mesh inscribed-polygon I shrinkage.
    n_sides = 12
    r_poly = r_cord * 1.023210
    cord_pts = [
        (
            y_cord + r_poly * math.cos(2.0 * math.pi * k / n_sides),
            0.5 * w_cell + r_poly * math.sin(2.0 * math.pi * k / n_sides),
        )
        for k in range(n_sides)
    ]
    with b3d.BuildPart() as cord_builder:
        with b3d.BuildSketch(b3d.Plane.YZ.offset(0.0)):
            with b3d.BuildLine():
                b3d.Polyline(*cord_pts, close=True)
            b3d.make_face()
        b3d.extrude(amount=spec.pitch_mm)

    # Cut cord volume from body cleanly in OpenCASCADE so STEP interfaces match exactly
    body_solid = body_builder.part - cord_builder.part
    cord_solid = cord_builder.part

    body_path = tmpdir / f"{spec.profile_id}_body.step"
    cord_path = tmpdir / f"{spec.profile_id}_cord.step"
    b3d.export_step(body_solid, str(body_path))
    b3d.export_step(cord_solid, str(cord_path))
    return body_path, cord_path


def _export_all_steps_isolated(out_dir: Path, profile_ids: list[str]) -> None:
    """Export bar benchmark and requested pitch-cell STEP files using build123d only."""
    import build123d as b3d

    out_dir.mkdir(parents=True, exist_ok=True)
    bar_step = out_dir / "bar.step"
    b3d.export_step(
        b3d.Box(3.0, 1.0, 1.0, align=(b3d.Align.MIN, b3d.Align.MIN, b3d.Align.MIN)),
        str(bar_step),
    )
    for pid in profile_ids:
        _build_pitch_cell_step_files(BELT_CELL_SPECS[pid], out_dir)


def _ensure_gmsh_clean_model(name: str) -> None:
    """Initialize Gmsh once per process and reset to a fresh model."""
    import gmsh

    if not gmsh.isInitialized():
        gmsh.initialize(["gmsh"], interruptible=False)
    else:
        gmsh.clear()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(name)


def build_pitch_cell_mesh(
    spec: BeltPitchCellSpec,
    h_mm: float,
    order: int = 2,
    step_paths: tuple[Path, Path] | None = None,
) -> tuple[Any, Any, np.ndarray]:
    """Build 3D build123d pitch cell and generate conformal Gmsh -> skfem Tet10 mesh.

    Physical group tags:
      - gid 1: elastomer body (backing + tooth)
      - gid 2: tensile cord cylinder
    """
    import gmsh
    import skfem

    tmp_ctx = None
    if step_paths is None:
        tmp_ctx = tempfile.TemporaryDirectory(prefix=f"printsim_belt_{spec.profile_id}_")
        body_step, cord_step = _build_pitch_cell_step_files(spec, Path(tmp_ctx.name))
    else:
        body_step, cord_step = step_paths

    try:
        _ensure_gmsh_clean_model(f"belt_{spec.profile_id}")
        occ = gmsh.model.occ

        body_ents = occ.importShapes(str(body_step))
        cord_ents = occ.importShapes(str(cord_step))
        body_vols = [e for e in body_ents if e[0] == 3]
        cord_vols = [e for e in cord_ents if e[0] == 3]

        # Conformal boolean fragment across elastomer body and cord cylinder
        _, out_map = occ.fragment(cord_vols, body_vols)
        occ.synchronize()

        cord_out = sorted({tg for k in range(len(cord_vols)) for d, tg in out_map[k] if d == 3})
        body_raw = {tg for k in range(len(cord_vols), len(out_map)) for d, tg in out_map[k] if d == 3}
        body_out = sorted(body_raw - set(cord_out))

        gmsh.model.addPhysicalGroup(3, body_out, 1)
        gmsh.model.addPhysicalGroup(3, cord_out, 2)

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", max(0.25 * h_mm, 0.04))
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h_mm * 1.15)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h_mm)
        # Refine slightly around the cord boundary for smooth shear transfer
        pts_c = gmsh.model.getBoundary(
            [(3, tg) for tg in cord_out], combined=False, oriented=False, recursive=True
        )
        gmsh.model.mesh.setSize([pt for pt in pts_c if pt[0] == 0], 0.72 * h_mm)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.model.mesh.generate(3)

        ntags, ncoords, _ = gmsh.model.mesh.getNodes()
        all_tets: list[np.ndarray] = []
        all_gids: list[np.ndarray] = []
        for _, gid in gmsh.model.getPhysicalGroups(3):
            for ent in gmsh.model.getEntitiesForPhysicalGroup(3, gid):
                _, enodes = gmsh.model.mesh.getElementsByType(4, ent)
                if len(enodes) > 0:
                    arr = np.asarray(enodes, dtype=np.int64).reshape(-1, 4)
                    all_tets.append(arr)
                    all_gids.append(np.full(len(arr), gid, dtype=np.int32))
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()

    raw_tets = np.vstack(all_tets)
    elem_gid = np.concatenate(all_gids)
    ntags_arr = np.asarray(ntags, dtype=np.int64)
    coords_mm = np.asarray(ncoords, dtype=np.float64).reshape(-1, 3)
    order_idx = np.argsort(ntags_arr)
    pos = np.searchsorted(ntags_arr[order_idx], raw_tets.ravel())
    idx = order_idx[pos].reshape(-1, 4)
    used, inv = np.unique(idx, return_inverse=True)

    mesh = skfem.MeshTet((coords_mm[used] * 1e-3).T, inv.reshape(-1, 4).T).oriented()
    scalar_elem = skfem.ElementTetP2() if order == 2 else skfem.ElementTetP1()
    element = skfem.ElementVector(scalar_elem)
    intorder = 2 if order == 2 else 1
    basis = skfem.CellBasis(mesh, element, intorder=intorder)
    return mesh, basis, elem_gid


def assemble_domain_stiffness(
    mesh: Any,
    basis: Any,
    elem_gid: np.ndarray,
    gid: int,
    d_mat: np.ndarray,
    intorder: int = 2,
) -> Any:
    """Assemble 3D elasticity stiffness matrix K for a single physical group gid."""
    import skfem

    elems = np.nonzero(elem_gid == gid)[0]
    if len(elems) == 0:
        raise RuntimeError(f"No tetrahedral elements found for physical group {gid}")
    nz_rows = [[(j, float(d_mat[i, j])) for j in range(6) if abs(d_mat[i, j]) > 0.0] for i in range(6)]

    @skfem.BilinearForm
    def elas(u, v, w):
        g_u, g_v = u.grad, v.grad
        eu = (
            g_u[0, 0],
            g_u[1, 1],
            g_u[2, 2],
            g_u[0, 1] + g_u[1, 0],
            g_u[1, 2] + g_u[2, 1],
            g_u[0, 2] + g_u[2, 0],
        )
        ev = (
            g_v[0, 0],
            g_v[1, 1],
            g_v[2, 2],
            g_v[0, 1] + g_v[1, 0],
            g_v[1, 2] + g_v[2, 1],
            g_v[0, 2] + g_v[2, 0],
        )
        out = None
        for i in range(6):
            terms = nz_rows[i]
            if not terms:
                continue
            s_i = terms[0][1] * eu[terms[0][0]]
            for j, d_ij in terms[1:]:
                s_i = s_i + d_ij * eu[j]
            term = s_i * ev[i]
            out = term if out is None else out + term
        return out

    sub_basis = skfem.CellBasis(mesh, basis.elem, elements=elems, intorder=intorder)
    return elas.assemble(sub_basis)


# ---------------------------------------------------------------------------
# 4. Analytical Benchmark: Plane-Strain Homogeneous Bar Bending
# ---------------------------------------------------------------------------


def run_analytical_bar_benchmark(step_path: Path | None = None) -> dict[str, Any]:
    """Verify 3D Tet10 ABD bending extractor against exact plane-strain Euler-Bernoulli EI."""
    import gmsh
    import skfem

    l_mm, h_mm, w_mm = 3.0, 1.0, 1.0
    e_pa, nu = 25.0e6, 0.45
    tmp_ctx = None
    if step_path is None:
        import build123d as b3d

        tmp_ctx = tempfile.TemporaryDirectory(prefix="printsim_bar_bench_")
        step_path = Path(tmp_ctx.name) / "bar.step"
        bar = b3d.Box(l_mm, h_mm, w_mm, align=(b3d.Align.MIN, b3d.Align.MIN, b3d.Align.MIN))
        b3d.export_step(bar, str(step_path))

    try:
        _ensure_gmsh_clean_model("bar_bench")
        ents = gmsh.model.occ.importShapes(str(step_path))
        vols = [e[1] for e in ents if e[0] == 3]
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(3, vols, 1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.35)
        gmsh.model.mesh.generate(3)
        ntags, ncoords, _ = gmsh.model.mesh.getNodes()
        _, enodes = gmsh.model.mesh.getElementsByType(4)
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()

    raw_tets = np.asarray(enodes, dtype=np.int64).reshape(-1, 4)
    ntags_arr = np.asarray(ntags, dtype=np.int64)
    coords_mm = np.asarray(ncoords, dtype=np.float64).reshape(-1, 3)
    order_idx = np.argsort(ntags_arr)
    idx = order_idx[np.searchsorted(ntags_arr[order_idx], raw_tets.ravel())].reshape(-1, 4)
    used, inv = np.unique(idx, return_inverse=True)
    mesh = skfem.MeshTet((coords_mm[used] * 1e-3).T, inv.reshape(-1, 4).T).oriented()
    basis = skfem.CellBasis(mesh, skfem.ElementVector(skfem.ElementTetP2()), intorder=2)
    elem_gid = np.ones(mesh.nelements, dtype=np.int32)
    k_mat = assemble_domain_stiffness(mesh, basis, elem_gid, 1, iso_constitutive_3d(e_pa, nu))

    ei_fea, _, _ = _solve_abd_bending(
        mesh=mesh,
        basis=basis,
        k_body=k_mat,
        k_cord=None,
        p_m=l_mm * 1e-3,
        w_m=w_mm * 1e-3,
        y_ref_m=0.5 * h_mm * 1e-3,
    )
    e_ps = e_pa / (1.0 - nu**2)
    ei_exact = e_ps * ((h_mm * 1e-3) ** 3) / 12.0
    rel_err_pct = 100.0 * abs(ei_fea - ei_exact) / ei_exact
    return {
        "ei_fea_nm2_per_m": ei_fea,
        "ei_exact_nm2_per_m": ei_exact,
        "rel_error_pct": rel_err_pct,
        "n_tets": int(mesh.nelements),
        "n_dofs": int(basis.N),
    }


# ---------------------------------------------------------------------------
# 5. BVP 1: 3D Tet10 Tooth-Mesh Shear & Active-Set Radial Flank Seating
# ---------------------------------------------------------------------------


def _classify_pulley_contact_facets(
    mesh: Any,
    spec: BeltPitchCellSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Identify boundary facets on the tooth flanks & root lands and compute initial clearances.

    Returns:
        (contact_facets, facet_normals_3xN, gap_m, is_driving_flank, is_root_land)
    """
    bfacets = mesh.boundary_facets()
    p_m = spec.pitch_mm * 1e-3
    ht_m = spec.tooth_height_mm * 1e-3
    h_tot_m = spec.total_height_mm * 1e-3
    w_m = spec.cord_pitch_mm * 1e-3
    xc_m = 0.5 * p_m

    # Facet centroids and outward unit normals
    # mesh.facets[:, bfacets] has shape (3, n_bf)
    tri = mesh.p[:, mesh.facets[:, bfacets]]  # (3, 3, n_bf)
    cen = tri.mean(axis=1)  # (3, n_bf)
    e1 = tri[:, 1, :] - tri[:, 0, :]
    e2 = tri[:, 2, :] - tri[:, 0, :]
    nrm = np.cross(e1.T, e2.T).T
    nrm_len = np.linalg.norm(nrm, axis=0)
    nrm = nrm / np.maximum(nrm_len, 1e-18)

    # Ensure normals point outward from the adjacent tetrahedron centroid
    adj_tets = mesh.f2t[0, bfacets]
    tet_cen = mesh.p[:, mesh.t[:, adj_tets]].mean(axis=1)
    dot_out = np.sum(nrm * (cen - tet_cen), axis=0)
    flip = dot_out < 0.0
    nrm[:, flip] *= -1.0

    tol = 1e-6
    on_x0 = cen[0] < tol
    on_xp = cen[0] > (p_m - tol)
    on_z0 = cen[2] < tol
    on_zw = cen[2] > (w_m - tol)
    on_top = cen[1] > (h_tot_m - tol)
    is_bottom_profile = ~(on_x0 | on_xp | on_z0 | on_zw | on_top)

    # Exclude tooth crown tip clearance region (real pulleys clear the belt tooth crest)
    y_tip_clear = 0.14 * ht_m
    is_root_land = is_bottom_profile & (cen[1] > ht_m - 0.04 * ht_m) & (nrm[1] < -0.90)
    is_left_flank = is_bottom_profile & (~is_root_land) & (cen[1] >= y_tip_clear) & (cen[0] < xc_m) & (nrm[0] < -0.05)
    is_right_flank = is_bottom_profile & (~is_root_land) & (cen[1] >= y_tip_clear) & (cen[0] > xc_m) & (nrm[0] > 0.05)

    mask = is_root_land | is_left_flank | is_right_flank
    c_facets = bfacets[mask]
    c_nrm = nrm[:, mask]
    c_cen = cen[:, mask]
    c_land = is_root_land[mask]
    # Driving flank when cord pulls in -X: left flank (n_x < 0) pushes into pulley wall
    c_drive = is_left_flank[mask]

    # Initial flank/land clearance profile g0(y) (m):
    # Zero gap at the mid-flank seating pitch line (y_seat ~ 0.62 * ht), growing smoothly
    # toward tip and root fillet per Gates / DIN AT tooth-to-groove clearance tolerances.
    y_seat = 0.62 * ht_m
    u_flank = np.abs(c_cen[1] - y_seat) / max(0.55 * ht_m, 1e-9)
    gap_m = np.where(
        c_land,
        spec.land_clearance_mm * 1e-3,
        (spec.flank_clearance_max_mm * 1e-3) * (u_flank**2),
    )
    return c_facets, c_nrm, gap_m, c_drive, c_land


def solve_tooth_mesh_seating_and_shear(
    mesh: Any,
    basis: Any,
    elem_gid: np.ndarray,
    spec: BeltPitchCellSpec,
    q0_list_n_per_mm: list[float],
) -> list[dict[str, float]]:
    """Run active-set radial seating + tangent tooth-mesh shear FEA across line tensions q0."""
    from scipy.sparse.linalg import splu
    import skfem

    p_m = spec.pitch_mm * 1e-3
    w_m = spec.cord_pitch_mm * 1e-3
    h_tot_m = spec.total_height_mm * 1e-3
    r_pulley_m = (spec.nominal_pulley_teeth * p_m) / (2.0 * math.pi)

    # Assemble base stiffness: elastomer body + stiff axial tensile cord
    e_m_eff = spec.e_matrix_25c_pa * spec.facing_shear_boost
    e_c_ax = cord_axial_stiffness_modulus_pa(spec)
    k_body = assemble_domain_stiffness(mesh, basis, elem_gid, 1, iso_constitutive_3d(e_m_eff, spec.nu_matrix))
    k_cord = assemble_domain_stiffness(mesh, basis, elem_gid, 2, iso_constitutive_3d(e_c_ax, 0.28))
    k_base = (k_body + k_cord).tocsc()

    # Classify pulley contact boundary facets
    c_facets, c_nrm, gap_m, is_drive, is_land = _classify_pulley_contact_facets(mesh, spec)
    n_cf = len(c_facets)
    if n_cf == 0:
        raise RuntimeError(f"No pulley contact facets found for {spec.profile_id}")

    # Precompute facet areas for area-fraction reporting
    tri_c = mesh.p[:, mesh.facets[:, c_facets]]
    c_areas = 0.5 * np.linalg.norm(
        np.cross((tri_c[:, 1, :] - tri_c[:, 0, :]).T, (tri_c[:, 2, :] - tri_c[:, 0, :]).T),
        axis=1,
    )
    drive_area_tot = float(np.sum(c_areas[is_drive]))
    land_area_tot = float(np.sum(c_areas[is_land]))

    tol = 1e-6
    # Width symmetry: u_z = 0 on z = 0 and z = w_m
    z_sym_facets = mesh.facets_satisfying(lambda x: (x[2] < tol) | (x[2] > w_m - tol))
    z_sym_dofs = basis.get_dofs(z_sym_facets).all("u^3")

    # End faces x = 0 and x = p_m (used for symmetric u_x = 0 during radial seating)
    x_end_facets = mesh.facets_satisfying(lambda x: (x[0] < tol) | (x[0] > p_m - tol))
    x_end_ux_dofs = basis.get_dofs(x_end_facets).all("u^1")

    # Top backing face facets (y = H) where wrap radial pressure p_r = q0 / R is applied
    top_facets = mesh.facets_satisfying(lambda x: x[1] > h_tot_m - tol)
    top_fbasis = skfem.FacetBasis(mesh, basis.elem, facets=top_facets, intorder=2)

    @skfem.LinearForm
    def lf_unit_radial_pressure(v, w):
        # Downward pressure (-Y direction) of 1 Pa over the belt back
        return -1.0 * v[1]

    f_unit_pr = lf_unit_radial_pressure.assemble(top_fbasis)

    # Cord cylinder DOFs for applying tangential shear displacement in Step 2
    cord_elems = np.nonzero(elem_gid == 2)[0]
    cord_basis = skfem.CellBasis(mesh, basis.elem, elements=cord_elems, intorder=2)
    cord_dofs_all = np.unique(cord_basis.element_dofs.ravel())
    # Extract u^1 (X-component) and u^2 (Y-component) DOFs belonging to the cord cylinder
    all_u1_dofs = np.unique(np.concatenate([basis.nodal_dofs[0], basis.edge_dofs[0]]))
    all_u2_dofs = np.unique(np.concatenate([basis.nodal_dofs[1], basis.edge_dofs[1]]))
    cord_ux_dofs = np.intersect1d(cord_dofs_all, all_u1_dofs)
    cord_uy_dofs = np.intersect1d(cord_dofs_all, all_u2_dofs)

    # Penalty stiffness for rigid pulley normal contact (k_pen >> E_m / h_t)
    ht_m = spec.tooth_height_mm * 1e-3
    k_pen_n = 2.5e3 * (e_m_eff / ht_m)  # Pa/m (~4e13 Pa/m for 2GT)
    # Full-stick tangential micro-slip contact stiffness ceiling on fully-pressurized interfaces
    k_pen_t_max = 40.0 * (e_m_eff / ht_m)
    # Characteristic elastomer asperity shear-saturation pressure (Mindlin/Coulomb interfacial grip):
    # scales naturally with elastomer modulus (~0.022 * E_m -> ~295 kPa for EPDM, ~920 kPa for 92 ShA TPU)
    tau_sat_pa = 0.022 * e_m_eff

    # Helper to build contact bilinear/linear forms for a boolean active mask over c_facets
    def _assemble_contact_step1(active_mask: np.ndarray) -> tuple[Any, np.ndarray]:
        act_idx = np.nonzero(active_mask)[0]
        if len(act_idx) == 0:
            return None, np.zeros(basis.N, dtype=np.float64)
        fb = skfem.FacetBasis(mesh, basis.elem, facets=c_facets[act_idx], intorder=2)
        g_vals = gap_m[act_idx]  # (n_act,)
        # Broadcast per-facet constant gap to quadrature points (n_act, n_qp)
        g_qp = np.broadcast_to(g_vals[:, None], (len(act_idx), fb.dx.shape[1])).copy()

        @skfem.BilinearForm
        def bf_pen(u, v, w):
            un = u[0] * w.n[0] + u[1] * w.n[1] + u[2] * w.n[2]
            vn = v[0] * w.n[0] + v[1] * w.n[1] + v[2] * w.n[2]
            return k_pen_n * un * vn

        @skfem.LinearForm
        def lf_pen(v, w):
            vn = v[0] * w.n[0] + v[1] * w.n[1] + v[2] * w.n[2]
            return k_pen_n * g_qp * vn

        return bf_pen.assemble(fb), lf_pen.assemble(fb)

    def _assemble_shear_contact(
        active_drive: np.ndarray,
        active_land: np.ndarray,
        p_n_facet_pa: np.ndarray,
    ) -> Any:
        # On active driving flank: normal rigid pulley wall (k_pen_n) + pressure-dependent Mindlin
        # tangential friction stiffness k_t(p_n) = k_pen_t_max * (mu * p_n) / (mu * p_n + tau_sat).
        # On active root land: normal pulley OD support + pressure-dependent Coulomb/Mindlin friction
        # (prevents unphysical zero-pressure land gluing at low belt tension q0).
        act_all = np.nonzero(active_drive | active_land)[0]
        if len(act_all) == 0:
            raise RuntimeError("Zero active contact facets in shear perturbation")
        fb = skfem.FacetBasis(mesh, basis.elem, facets=c_facets[act_all], intorder=2)
        pn_act = np.maximum(p_n_facet_pa[act_all], 0.0)
        grip_ratio = (spec.flank_friction_mu * pn_act) / (spec.flank_friction_mu * pn_act + tau_sat_pa)
        kt_vals = k_pen_t_max * grip_ratio
        kt_qp = np.broadcast_to(kt_vals[:, None], (len(act_all), fb.dx.shape[1])).copy()

        @skfem.BilinearForm
        def bf_shear_contact(u, v, w):
            un = u[0] * w.n[0] + u[1] * w.n[1] + u[2] * w.n[2]
            vn = v[0] * w.n[0] + v[1] * w.n[1] + v[2] * w.n[2]
            ut0 = u[0] - un * w.n[0]
            ut1 = u[1] - un * w.n[1]
            ut2 = u[2] - un * w.n[2]
            vt0 = v[0] - vn * w.n[0]
            vt1 = v[1] - vn * w.n[1]
            vt2 = v[2] - vn * w.n[2]
            return k_pen_n * un * vn + kt_qp * (ut0 * vt0 + ut1 * vt1 + ut2 * vt2)

        return bf_shear_contact.assemble(fb)

    # Precompute facet basis over ALL contact facets to evaluate facet-average u . n fast
    fb_all_c = skfem.FacetBasis(mesh, basis.elem, facets=c_facets, intorder=2)
    fixed_step1 = np.unique(np.concatenate([z_sym_dofs, x_end_ux_dofs]))
    fixed_step2 = np.unique(np.concatenate([z_sym_dofs, cord_ux_dofs]))

    # Track convergence of active contact set as q0 ramps up
    results: list[dict[str, float]] = []
    # Seed initial active set with minimum-gap mid-flank facets
    active = gap_m <= (0.18 * spec.flank_clearance_max_mm * 1e-3)
    if not np.any(active & is_drive):
        best_drive = np.nonzero(is_drive)[0][np.argmin(gap_m[is_drive])]
        active[best_drive] = True

    for q0_n_mm in q0_list_n_per_mm:
        q0_n_m = q0_n_mm * 1e3  # N/mm -> N/m
        p_r_pa = q0_n_m / r_pulley_m  # radial wrap pressure (Pa)
        f_rad = p_r_pa * f_unit_pr

        # Active-set contact loop with 2-cycle chattering guard (typically converges in 2-5 steps)
        u_rad = np.zeros(basis.N, dtype=np.float64)
        un_mean = np.zeros(n_cf, dtype=np.float64)
        seen_masks: set[bytes] = set()
        for _it in range(7):
            seen_masks.add(active.tobytes())
            k_c, f_c = _assemble_contact_step1(active)
            k_tot = k_base if k_c is None else (k_base + k_c)
            f_tot = f_rad + f_c
            k_ff, f_f, _, idx_free = skfem.condense(k_tot, f_tot, D=fixed_step1)
            u_rad = np.zeros(basis.N, dtype=np.float64)
            u_rad[idx_free] = splu(k_ff.tocsc(), permc_spec="COLAMD").solve(f_f)

            # Evaluate mean normal displacement u . n on every candidate contact facet
            uh_c = fb_all_c.interpolate(u_rad)  # (3, n_cf, n_qp)
            un_qp = uh_c[0] * fb_all_c.normals[0] + uh_c[1] * fb_all_c.normals[1] + uh_c[2] * fb_all_c.normals[2]
            un_mean = np.sum(un_qp * fb_all_c.dx, axis=1) / np.maximum(np.sum(fb_all_c.dx, axis=1), 1e-24)

            new_active = un_mean > (gap_m - 1e-11)
            if not np.any(new_active & is_drive):
                best_d = np.nonzero(is_drive)[0][np.argmin(gap_m[is_drive] - un_mean[is_drive])]
                new_active[best_d] = True
            if np.array_equal(new_active, active):
                break
            if new_active.tobytes() in seen_masks:
                active = active | new_active
                break
            active = new_active

        active_drive = active & is_drive
        active_land = active & is_land
        flank_frac = float(np.sum(c_areas[active_drive]) / max(drive_area_tot, 1e-18))
        land_frac = float(np.sum(c_areas[active_land]) / max(land_area_tot, 1e-18))

        # Local interfacial normal contact pressure (Pa) from converged penalty indentation
        p_n_facet = np.where(active, k_pen_n * np.maximum(un_mean - gap_m, 0.0), 0.0)

        # Mean radial cord seating displacement (um, positive = deeper into pulley groove)
        seat_disp_um = float(-np.mean(u_rad[cord_uy_dofs]) * 1e6)

        # Step 2: Tangential shear perturbation on converged contact patch
        delta_x = -1.0e-6
        k_sh_c = _assemble_shear_contact(active_drive, active_land, p_n_facet)
        k_sh = (k_base + k_sh_c).tocsc()

        u_presc = np.zeros(basis.N, dtype=np.float64)
        u_presc[cord_ux_dofs] = delta_x
        k_ff, f_f, _, idx_free = skfem.condense(k_sh, np.zeros(basis.N), x=u_presc, D=fixed_step2)
        u_sh = u_presc.copy()
        u_sh[idx_free] = splu(k_ff.tocsc(), permc_spec="COLAMD").solve(f_f)

        # Exact reaction force F_x via Clapeyron strain energy: U = 0.5 * F_x * |delta_x|
        strain_energy = 0.5 * float(u_sh @ (k_sh @ u_sh))
        k_tooth_cell_n_m = (2.0 * strain_energy) / (delta_x**2)  # N/m for cell width cord_pitch_mm
        k_t_n_per_m_mm = k_tooth_cell_n_m / spec.cord_pitch_mm

        results.append(
            {
                "q0_n_per_mm": float(q0_n_mm),
                "radial_pressure_kpa": float(p_r_pa * 1e-3),
                "k_t_n_per_m_mm": float(k_t_n_per_m_mm),
                "flank_contact_fraction": float(flank_frac),
                "land_contact_fraction": float(land_frac),
                "radial_seating_um": float(seat_disp_um),
            }
        )

    # Fully-seated high-pressure asymptotic upper bound k_t_full (all driving flank + root land at stick)
    p_n_full = np.full(n_cf, 1.0e12, dtype=np.float64)
    k_sh_full = (k_base + _assemble_shear_contact(is_drive, is_land, p_n_full)).tocsc()
    delta_x = -1.0e-6
    u_presc = np.zeros(basis.N, dtype=np.float64)
    u_presc[cord_ux_dofs] = delta_x
    k_ff, f_f, _, idx_free = skfem.condense(k_sh_full, np.zeros(basis.N), x=u_presc, D=fixed_step2)
    u_sh = u_presc.copy()
    u_sh[idx_free] = splu(k_ff.tocsc(), permc_spec="COLAMD").solve(f_f)
    k_t_full = (float(u_sh @ (k_sh_full @ u_sh)) / (delta_x**2)) / spec.cord_pitch_mm
    for row in results:
        row["seating_ratio"] = float(row["k_t_n_per_m_mm"] / max(k_t_full, 1.0))
        row["k_t_full_n_per_m_mm"] = float(k_t_full)

    return results


# ---------------------------------------------------------------------------
# 6. BVP 2: 3D Tet10 Composite Bending Rigidity EI_b & Energy Partition
# ---------------------------------------------------------------------------


def _solve_abd_bending(
    mesh: Any,
    basis: Any,
    k_body: Any,
    k_cord: Any | None,
    p_m: float,
    w_m: float,
    y_ref_m: float,
) -> tuple[float, float, float]:
    """Extract exact plane-strain flexural rigidity EI_b (N*m^2/m), neutral axis y_na (m),
    and cord bending energy fraction via 2x2 ABD static condensation.
    """
    from scipy.sparse.linalg import splu
    import skfem

    k_tot = k_body if k_cord is None else (k_body + k_cord)
    k_tot = k_tot.tocsc()

    tol = 1e-6
    z_sym_facets = mesh.facets_satisfying(lambda x: (x[2] < tol) | (x[2] > w_m - tol))
    z_sym_dofs = basis.get_dofs(z_sym_facets).all("u^3")

    x0_facets = mesh.facets_satisfying(lambda x: x[0] < tol)
    xp_facets = mesh.facets_satisfying(lambda x: x[0] > p_m - tol)
    x0_ux_dofs = basis.get_dofs(x0_facets).all("u^1")
    xp_ux_dofs = basis.get_dofs(xp_facets).all("u^1")

    # Pin u_y = 0 at the single closest node to (0, y_ref, 0) and (p_m, y_ref, 0)
    # to eliminate rigid Y translation and rigid Z rotation while keeping Poisson contraction free
    all_u2_dofs = np.unique(np.concatenate([basis.nodal_dofs[1], basis.edge_dofs[1]]))
    locs_u2 = basis.doflocs[:, all_u2_dofs]
    pin_l = all_u2_dofs[np.argmin((locs_u2[0] - 0.0) ** 2 + (locs_u2[1] - y_ref_m) ** 2 + locs_u2[2] ** 2)]
    pin_r = all_u2_dofs[np.argmin((locs_u2[0] - p_m) ** 2 + (locs_u2[1] - y_ref_m) ** 2 + locs_u2[2] ** 2)]
    pin_uy_dofs = np.unique(np.array([pin_l, pin_r], dtype=np.int64))

    fixed_dofs = np.unique(np.concatenate([z_sym_dofs, x0_ux_dofs, xp_ux_dofs, pin_uy_dofs]))

    # Factor K_ff once and solve both load cases (axial extension eps0=1, curvature kap0=1)
    k_ff, _, _, idx_free = skfem.condense(k_tot, np.zeros(basis.N), D=fixed_dofs)
    lu = splu(k_ff.tocsc(), permc_spec="COLAMD")

    def _solve_prescribed(u_p: np.ndarray) -> np.ndarray:
        rhs = -(k_tot @ u_p)
        u = u_p.copy()
        u[idx_free] = lu.solve(rhs[idx_free])
        return u

    # Case 1: unit axial strain eps0 = 1.0 -> u_x(0) = -p/2, u_x(p) = +p/2
    u_p_eps = np.zeros(basis.N, dtype=np.float64)
    u_p_eps[x0_ux_dofs] = -0.5 * p_m
    u_p_eps[xp_ux_dofs] = +0.5 * p_m
    u_eps = _solve_prescribed(u_p_eps)

    # Case 2: unit curvature kap0 = 1.0 (1/m) about y_ref -> u_x(p) - u_x(0) = p * (y - y_ref)
    u_p_kap = np.zeros(basis.N, dtype=np.float64)
    y_x0 = basis.doflocs[1, x0_ux_dofs]
    y_xp = basis.doflocs[1, xp_ux_dofs]
    u_p_kap[x0_ux_dofs] = -0.5 * p_m * (y_x0 - y_ref_m)
    u_p_kap[xp_ux_dofs] = +0.5 * p_m * (y_xp - y_ref_m)
    u_kap = _solve_prescribed(u_p_kap)

    # Exact 2x2 ABD matrix per meter of belt width via Clapeyron inner products
    ku_eps = k_tot @ u_eps
    ku_kap = k_tot @ u_kap
    a_val = float(u_eps @ ku_eps) / (p_m * w_m)
    b_val = float(u_eps @ ku_kap) / (p_m * w_m)
    d_val = float(u_kap @ ku_kap) / (p_m * w_m)

    # Static condensation N = 0 -> eps0 = -(B/A) * kap0
    ei_per_m = d_val - (b_val**2) / max(a_val, 1e-24)
    y_na_m = y_ref_m + b_val / max(a_val, 1e-24)

    # Pure free-bending displacement field and exact material strain-energy partition
    u_bend = u_kap - (b_val / max(a_val, 1e-24)) * u_eps
    e_body = float(u_bend @ (k_body @ u_bend))
    e_cord = 0.0 if k_cord is None else float(u_bend @ (k_cord @ u_bend))
    f_cord = e_cord / max(e_body + e_cord, 1e-24)
    return float(ei_per_m), float(y_na_m), float(f_cord)


def solve_bending_retention_sweep(
    mesh: Any,
    basis: Any,
    elem_gid: np.ndarray,
    spec: BeltPitchCellSpec,
    retentions: list[float],
) -> list[dict[str, float]]:
    """Compute composite EI_b (N*m^2/m) and cord-vs-elastomer energy partition vs E_matrix(T)."""
    p_m = spec.pitch_mm * 1e-3
    w_m = spec.cord_pitch_mm * 1e-3
    y_cord_m = (spec.tooth_height_mm + spec.pld_mm) * 1e-3

    # Unit-modulus body and cord stiffness matrices (linear scaling avoids re-assembly!)
    k_body_unit = assemble_domain_stiffness(mesh, basis, elem_gid, 1, iso_constitutive_3d(1.0, spec.nu_matrix))
    k_cord_unit = assemble_domain_stiffness(mesh, basis, elem_gid, 2, iso_constitutive_3d(1.0, 0.28))

    rows: list[dict[str, float]] = []
    for r_m in retentions:
        e_m = spec.e_matrix_25c_pa * r_m
        e_c_bend, ei_single_cord = cord_bundle_effective_bending_modulus_pa(spec, matrix_retention=r_m)
        ei_per_m, y_na_m, f_cord = _solve_abd_bending(
            mesh=mesh,
            basis=basis,
            k_body=e_m * k_body_unit,
            k_cord=e_c_bend * k_cord_unit,
            p_m=p_m,
            w_m=w_m,
            y_ref_m=y_cord_m,
        )
        rows.append(
            {
                "matrix_modulus_retention": float(r_m),
                "e_matrix_mpa": float(e_m * 1e-6),
                "e_cord_bend_mpa": float(e_c_bend * 1e-6),
                "ei_per_m_nm2_per_m": float(ei_per_m),
                "ei_cord_analytical_per_m": float(ei_single_cord / w_m),
                "neutral_axis_y_mm": float(y_na_m * 1e3),
                "cord_bending_fraction": float(f_cord),
                "elastomer_bending_fraction": float(1.0 - f_cord),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# 7. Closed-Form Seating Law Fit & Full Calibration Runner
# ---------------------------------------------------------------------------


def _fit_seating_law(sweep: list[dict[str, float]]) -> dict[str, float]:
    """Fit s(q0) = s0 + (1 - s0) * (1 - exp(-q0 / q_seat)) to FEA seating_ratio vs q0."""
    q = np.array([r["q0_n_per_mm"] for r in sweep], dtype=np.float64)
    y = np.array([r["seating_ratio"] for r in sweep], dtype=np.float64)

    best_err = float("inf")
    best_s0 = 0.25
    best_qs = 2.0
    # Deterministic coarse-to-fine grid search (zero scipy.optimize dependency)
    for s0 in np.linspace(0.05, 0.85, 81):
        for qs in np.linspace(0.4, 8.0, 153):
            pred = s0 + (1.0 - s0) * (1.0 - np.exp(-q / qs))
            err = float(np.mean((pred - y) ** 2))
            if err < best_err:
                best_err = err
                best_s0 = float(s0)
                best_qs = float(qs)

    pred = best_s0 + (1.0 - best_s0) * (1.0 - np.exp(-q / best_qs))
    max_abs_err = float(np.max(np.abs(pred - y)))
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    return {
        "seating_floor_s0": round(best_s0, 4),
        "seating_scale_q_n_per_mm": round(best_qs, 4),
        "fit_rmse": round(rmse, 5),
        "fit_max_abs_err": round(max_abs_err, 5),
    }


def calibrate_profile(
    spec: BeltPitchCellSpec,
    step_paths: tuple[Path, Path] | None = None,
) -> dict[str, Any]:
    """Run coarse + fine Tet10 FEA, mesh convergence, tension seating, and bending sweeps."""
    t_start = time.perf_counter()
    q0_sweep = [0.5, 1.0, 2.0, 4.0, 7.0, 12.0]
    ret_sweep = [1.0, 0.8, 0.6, 0.4, 0.25]

    print(f"\n[{spec.profile_id.upper()}] Building coarse Tet10 mesh (h={spec.h_coarse_mm} mm)...", flush=True)
    mesh_c, basis_c, gid_c = build_pitch_cell_mesh(spec, h_mm=spec.h_coarse_mm, order=2, step_paths=step_paths)
    tooth_c = solve_tooth_mesh_seating_and_shear(mesh_c, basis_c, gid_c, spec, [4.0])
    bend_c = solve_bending_retention_sweep(mesh_c, basis_c, gid_c, spec, [1.0])

    print(
        f"[{spec.profile_id.upper()}] Coarse Tet10: {mesh_c.nelements} tets, {basis_c.N} DOFs | "
        f"k_t(4 N/mm)={tooth_c[0]['k_t_n_per_m_mm']:.0f} N/(m*mm), "
        f"k_t_full={tooth_c[0]['k_t_full_n_per_m_mm']:.0f}, "
        f"EI_b(25C)={bend_c[0]['ei_per_m_nm2_per_m']:.5f} N*m^2/m",
        flush=True,
    )

    print(f"[{spec.profile_id.upper()}] Building fine Tet10 mesh (h={spec.h_fine_mm} mm)...", flush=True)
    mesh_f, basis_f, gid_f = build_pitch_cell_mesh(spec, h_mm=spec.h_fine_mm, order=2, step_paths=step_paths)
    tooth_f = solve_tooth_mesh_seating_and_shear(mesh_f, basis_f, gid_f, spec, q0_sweep)
    bend_f = solve_bending_retention_sweep(mesh_f, basis_f, gid_f, spec, ret_sweep)

    # Compare fine vs coarse at identical operating point (q0=4.0 N/mm, r_m=1.0)
    kt_c_4 = tooth_c[0]["k_t_n_per_m_mm"]
    kt_f_4 = next(r["k_t_n_per_m_mm"] for r in tooth_f if abs(r["q0_n_per_mm"] - 4.0) < 1e-9)
    kt_full_c = tooth_c[0]["k_t_full_n_per_m_mm"]
    kt_full_f = tooth_f[0]["k_t_full_n_per_m_mm"]
    ei_c_1 = bend_c[0]["ei_per_m_nm2_per_m"]
    ei_f_1 = bend_f[0]["ei_per_m_nm2_per_m"]

    kt_full_diff_pct = 100.0 * abs(kt_full_f - kt_full_c) / max(kt_full_f, 1.0)
    kt_4_diff_pct = 100.0 * abs(kt_f_4 - kt_c_4) / max(kt_f_4, 1.0)
    ei_diff_pct = 100.0 * abs(ei_f_1 - ei_c_1) / max(ei_f_1, 1e-12)

    seating_fit = _fit_seating_law(tooth_f)

    # Linear retention fit for EI_b(r_m) / EI_b(1.0) = f_cord_eff + (1 - f_cord_eff) * r_m
    r_arr = np.array([row["matrix_modulus_retention"] for row in bend_f], dtype=np.float64)
    ei_norm = np.array([row["ei_per_m_nm2_per_m"] / ei_f_1 for row in bend_f], dtype=np.float64)
    # Least-squares slope m_ret for y = 1 - m_ret * (1 - r_m)
    dr = 1.0 - r_arr
    elast_slope = float(np.dot(dr, 1.0 - ei_norm) / max(np.dot(dr, dr), 1e-12))
    elast_slope = max(0.0, min(1.0, elast_slope))

    elapsed_s = time.perf_counter() - t_start
    print(
        f"[{spec.profile_id.upper()}] Fine Tet10:   {mesh_f.nelements} tets, {basis_f.N} DOFs ({elapsed_s:.1f}s) | "
        f"k_t_full={kt_full_f:.0f} N/(m*mm) (conv {kt_full_diff_pct:.2f}%), "
        f"k_t(4)={kt_f_4:.0f} (conv {kt_4_diff_pct:.2f}%), "
        f"EI_b(25C)={ei_f_1:.5f} (conv {ei_diff_pct:.2f}%), "
        f"f_cord={bend_f[0]['cord_bending_fraction']:.3f}, elast_slope={elast_slope:.3f}",
        flush=True,
    )

    return {
        "profile_id": spec.profile_id,
        "label": spec.label,
        "spec": asdict(spec),
        "mesh_convergence": {
            "coarse_h_mm": spec.h_coarse_mm,
            "coarse_n_tets": int(mesh_c.nelements),
            "coarse_n_dofs": int(basis_c.N),
            "fine_h_mm": spec.h_fine_mm,
            "fine_n_tets": int(mesh_f.nelements),
            "fine_n_dofs": int(basis_f.N),
            "k_t_full_coarse_n_per_m_mm": round(kt_full_c, 1),
            "k_t_full_fine_n_per_m_mm": round(kt_full_f, 1),
            "k_t_full_diff_pct": round(kt_full_diff_pct, 3),
            "k_t_q4_coarse_n_per_m_mm": round(kt_c_4, 1),
            "k_t_q4_fine_n_per_m_mm": round(kt_f_4, 1),
            "k_t_q4_diff_pct": round(kt_4_diff_pct, 3),
            "ei_coarse_nm2_per_m": round(ei_c_1, 7),
            "ei_fine_nm2_per_m": round(ei_f_1, 7),
            "ei_diff_pct": round(ei_diff_pct, 3),
        },
        "calibrated_summary": {
            "k_t_full_n_per_m_mm": round(kt_full_f, 1),
            "k_t_at_4n_mm_n_per_m_mm": round(kt_f_4, 1),
            "seating_floor_s0": seating_fit["seating_floor_s0"],
            "seating_scale_q_n_per_mm": seating_fit["seating_scale_q_n_per_mm"],
            "seating_fit_rmse": seating_fit["fit_rmse"],
            "ei_25c_nm2_per_m": round(ei_f_1, 7),
            "cord_bending_fraction_25c": round(bend_f[0]["cord_bending_fraction"], 4),
            "elastomer_bending_fraction_25c": round(bend_f[0]["elastomer_bending_fraction"], 4),
            "ei_matrix_retention_slope": round(elast_slope, 4),
            "neutral_axis_y_25c_mm": round(bend_f[0]["neutral_axis_y_mm"], 4),
        },
        "tooth_seating_sweep": tooth_f,
        "bending_retention_sweep": bend_f,
        "solve_time_s": round(elapsed_s, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="3D Tet10 single-pitch-cell FEA calibration for timing belts.")
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["2gt", "3gt", "at3"],
        choices=list(BELT_CELL_SPECS.keys()),
        help="Belt profiles to calibrate",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default="data/touchstones/belt_pitch_cell_fea.json",
        help="Output JSON summary path",
    )
    parser.add_argument(
        "--export-steps-dir",
        type=str,
        default="",
        help="Internal worker mode: export build123d STEP files only and exit",
    )
    args = parser.parse_args()

    if args.export_steps_dir:
        _export_all_steps_isolated(Path(args.export_steps_dir), list(args.profiles))
        return

    t0 = time.perf_counter()
    # Export all build123d OpenCASCADE STEP solids in an isolated worker process so OCP and Gmsh's
    # bundled OpenCASCADE never share an address space on macOS.
    with tempfile.TemporaryDirectory(prefix="printsim_belt_steps_") as td_str:
        td = Path(td_str)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--export-steps-dir",
            str(td),
            "--profiles",
            *args.profiles,
        ]
        subprocess.run(cmd, check=True)
        bar_step = td / "bar.step"
        profile_steps: dict[str, tuple[Path, Path]] = {
            pid: (td / f"{pid}_body.step", td / f"{pid}_cord.step") for pid in args.profiles
        }

        import gmsh

        print("=== Running Analytical Plane-Strain Bar Benchmark (Tet10) ===", flush=True)
        bar_bench = run_analytical_bar_benchmark(step_path=bar_step)
        print(
            f"  Bar EI_fea={bar_bench['ei_fea_nm2_per_m']:.6e} vs exact={bar_bench['ei_exact_nm2_per_m']:.6e} "
            f"(err={bar_bench['rel_error_pct']:.4f}%, {bar_bench['n_tets']} tets)",
            flush=True,
        )

        profiles_out: dict[str, Any] = {}
        try:
            for pid in args.profiles:
                profiles_out[pid] = calibrate_profile(BELT_CELL_SPECS[pid], step_paths=profile_steps[pid])
        finally:
            if gmsh.isInitialized():
                gmsh.finalize()

    payload = {
        "schema_version": 1,
        "description": (
            "First-principles 3D Tet10 continuum FEA calibration of single-pitch timing-belt unit cells "
            "(build123d OpenCASCADE B-Rep + conformal Gmsh Tet10 + scikit-fem elasticity)."
        ),
        "analytical_bar_benchmark": bar_bench,
        "profiles": profiles_out,
        "total_runtime_s": round(time.perf_counter() - t0, 2),
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved calibrated belt pitch-cell FEA summary to {out_path} ({payload['total_runtime_s']}s)")


if __name__ == "__main__":
    main()
