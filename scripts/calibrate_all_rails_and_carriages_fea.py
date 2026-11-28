"""3D Tet10 (P2) continuum FEA calibration for ALL solid linear rails, carriages & bushings.

Constructs exact watertight 3D B-Rep models in ``build123d`` / OpenCASCADE, meshes with Gmsh,
and solves 3D continuum elasticity via ``scikit-fem`` Tet10 (P2) elements for:

1. **All 4 Solid Profiled Linear Rails** (``mgn9_rail``, ``mgn12_rail``, ``mgn15_rail``,
   ``hgh15_rail``):
   - Exact HIWIN outer envelope, Gothic-arch (MGN) or twin circular-arc waisted (HGR15)
     ball-raceway grooves, and M3/M4 counterbored mounting holes on exact catalog pitch
     (20 / 25 / 40 / 60 mm).
   - Computes true homogenized ``area_m2``, ``iy_m4``, ``iz_m4``, ``j_m4`` (free & clamped
     warping), and short-span double-cantilever Timoshenko shear factors ``shear_kappa_y``,
     ``shear_kappa_z`` (replacing naive ``_solid_rect`` approximations).

2. **All 7 HIWIN Profiled Linear Carriages** (``mgn9c``, ``mgn9h``, ``mgn12c``, ``mgn12h``,
   ``mgn15c``, ``mgn15h``, ``hgh15ca``):
   - Exact HIWIN steel C-block saddle B-Rep geometry (width W, steel body length L1,
     height H - H1, inner U-slot, Gothic-arch / 4-row circular-arc raceway grooves,
     longitudinal ball-return bores, 4x tapped M3/M4 mounting holes on top B x C bolt pad).
   - Two-run differential compliance separation (rigid saddle vs 207 GPa bearing steel saddle
     over calibrated Hertzian ball-contact strips with free axial rolling along X) extracts
     true preload-independent 6-DOF C-block steel saddle structural stiffness:
     ``k_skirt_yy``, ``k_skirt_zz``, ``k_skirt_rx`` (roll skirt opening), ``k_skirt_ry``
     (pitch overhang/web flex), and ``k_skirt_rz`` (yaw overhang/skirt shear).
   - Replaces the unsourced generic 15% Hung scalar (``k_hertz_z1 / 0.15``) and flat ``1.25``
     roll opening factor in ``src/printsim/fea/components.py``.

3. **Round Linear Ball Bushings** (``lm8uu``, ``lm8luu``, ``lm10uu``, ``lm10luu``, ``lm12uu``,
   ``lm12luu``):
   - Exact Misumi hardened bearing-steel cylindrical sleeve B-Rep geometry over 4-row Hertzian
     ball-contact strips, validating axisymmetric sleeve-housing structural compliance.

4. **Mesh Convergence & Preload Invariance**:
   - Verifies Tet10 coarse vs medium vs fine convergence on ``mgn12_rail`` and ``mgn12h``,
     plus preload invariance of ``k_skirt_*`` across ``Z0 / Z1 / ZA``.

Run:
    uv run python scripts/calibrate_all_rails_and_carriages_fea.py
    uv run python scripts/calibrate_all_rails_and_carriages_fea.py --only rails carriages bushings convergence
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
import math
from pathlib import Path
import tempfile
import time
from typing import Any

# NOTE: Import gmsh BEFORE build123d so Gmsh's bundled OCCT symbols resolve before OCP on macOS.
import gmsh
import build123d as bd
import numpy as np
from scipy.sparse.linalg import splu
import skfem

from printsim.fea.components import _RAIL_CARRIAGE_CATALOG

RHO_STEEL = 7850.0  # kg/m^3 (SUJ2 / GCr15 / 4140 bearing steel)
E_STEEL = 207.0e9  # Pa (hardened bearing steel)
NU_STEEL = 0.29
G_STEEL = E_STEEL / (2.0 * (1.0 + NU_STEEL))
E_RIGID = 20000.0e9  # Pa (rigid reference platen/saddle, ~100x steel)

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 1. Exact HIWIN Solid Rail Geometry Specs (MGN9R, MGN12R, MGN15R, HGR15R)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SolidRailSpec:
    """Exact HIWIN solid linear rail outer envelope, raceway grooves, and bolt pitch."""

    profile_id: str
    display_name: str
    rail_style: str  # "mgn_2row" | "hg_4row"
    width_mm: float  # W_R (along local Y)
    height_mm: float  # H_R (along local Z)
    pitch_mm: float  # Bolt hole pitch P (along local X)
    thru_d_mm: float  # Bolt clearance bore d
    cbore_d_mm: float  # Counterbore D
    cbore_depth_mm: float  # Counterbore depth h from top face
    # For MGN (2-row Gothic arch): single groove per side at race_z_from_bottom_mm
    # For HG (4-row circular arc): twin grooves per side at race_z_low_mm & race_z_high_mm
    race_z_from_bottom_mm: float
    race_groove_h_mm: float
    race_groove_d_mm: float
    race_z_low_mm: float = 0.0
    race_z_high_mm: float = 0.0
    waist_depth_mm: float = 0.0  # HG side-waist recess depth between upper & lower tracks
    chamfer_mm: float = 0.30
    bare_fea_length_mm: float = 200.0
    shear_pitches: int = 2
    catalog_mass_g_per_m: float = 650.0


SOLID_RAIL_SPECS: dict[str, SolidRailSpec] = {
    "mgn9_rail": SolidRailSpec(
        profile_id="mgn9_rail",
        display_name="MGN9H Solid Linear Rail (9x6.5mm)",
        rail_style="mgn_2row",
        width_mm=9.0,
        height_mm=6.5,
        pitch_mm=20.0,
        thru_d_mm=3.5,
        cbore_d_mm=6.0,
        cbore_depth_mm=3.5,
        race_z_from_bottom_mm=3.9,
        race_groove_h_mm=1.85,
        race_groove_d_mm=0.85,
        chamfer_mm=0.25,
        bare_fea_length_mm=200.0,
        shear_pitches=2,
        catalog_mass_g_per_m=380.0,
    ),
    "mgn12_rail": SolidRailSpec(
        profile_id="mgn12_rail",
        display_name="MGN12H Solid Linear Rail (12x8mm)",
        rail_style="mgn_2row",
        width_mm=12.0,
        height_mm=8.0,
        pitch_mm=25.0,
        thru_d_mm=3.5,
        cbore_d_mm=6.0,
        cbore_depth_mm=4.5,
        race_z_from_bottom_mm=4.8,
        race_groove_h_mm=2.65,
        race_groove_d_mm=1.25,
        chamfer_mm=0.30,
        bare_fea_length_mm=200.0,
        shear_pitches=2,
        catalog_mass_g_per_m=650.0,
    ),
    "mgn15_rail": SolidRailSpec(
        profile_id="mgn15_rail",
        display_name="MGN15H Solid Linear Rail (15x10mm)",
        rail_style="mgn_2row",
        width_mm=15.0,
        height_mm=10.0,
        pitch_mm=40.0,
        thru_d_mm=3.5,
        cbore_d_mm=6.0,
        cbore_depth_mm=4.5,
        race_z_from_bottom_mm=6.0,
        race_groove_h_mm=3.50,
        race_groove_d_mm=1.65,
        chamfer_mm=0.35,
        bare_fea_length_mm=240.0,
        shear_pitches=2,
        catalog_mass_g_per_m=1060.0,
    ),
    # HIWIN HG Series HGR15R: W_R = 15 mm, H_R = 15 mm, P = 60 mm, M4 bolt (d=4.5, D=7.5, h=5.3),
    # 4-row circular-arc tracks (two tracks per side at z=7.3 & 11.7 mm with recessed waist),
    # catalog rail weight = 1.45 kg/m (184.71 mm^2 @ 7850 kg/m^3).
    "hgh15_rail": SolidRailSpec(
        profile_id="hgh15_rail",
        display_name="HGR15R Heavy Linear Rail (15x15mm)",
        rail_style="hg_4row",
        width_mm=15.0,
        height_mm=15.0,
        pitch_mm=60.0,
        thru_d_mm=4.5,
        cbore_d_mm=7.5,
        cbore_depth_mm=5.3,
        race_z_from_bottom_mm=9.5,
        race_groove_h_mm=3.10,
        race_groove_d_mm=1.85,
        race_z_low_mm=7.3,
        race_z_high_mm=11.7,
        waist_depth_mm=2.25,
        chamfer_mm=0.50,
        bare_fea_length_mm=360.0,
        shear_pitches=1,  # 1 pitch = 60 mm (L/H = 4.0, ideal Timoshenko shear ratio)
        catalog_mass_g_per_m=1450.0,
    ),
}


# ---------------------------------------------------------------------------
# 2. Exact HIWIN Profiled Carriage Steel Saddle Specs (All 7 Carriages)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CarriageSaddleSpec:
    """Exact HIWIN steel C-block saddle B-Rep dimensions (mm)."""

    carriage_id: str  # e.g. "mgn12h"
    rail_profile_id: str  # e.g. "mgn12_rail"
    block_type: str  # "C" | "H"
    rail_style: str  # "mgn_2row" | "hg_4row"
    width_mm: float  # Block outer width W (along Y)
    l1_mm: float  # Hardened steel saddle length L1 (along X, excluding plastic endcaps)
    stack_h_mm: float  # Assembly stack height H from rail bottom to carriage top
    bottom_clr_h1_mm: float  # Bottom clearance H1 above rail bottom face
    rail_w_mm: float  # Rail width W_R
    rail_h_mm: float  # Rail height H_R
    bolt_b_mm: float  # Transverse bolt spacing B (along Y)
    bolt_c_mm: float  # Axial bolt spacing C (along X)
    bolt_tap_d_mm: float  # Tapped minor/nominal hole diameter
    bolt_tap_depth_mm: float  # Thread engagement depth from top face
    bolt_pad_r_mm: float  # Clamped SHCS bolt-head seating pad radius on top face
    ball_d_mm: float  # HIWIN ball diameter d_b
    return_bore_d_mm: float  # Longitudinal ball return tube bore inside each skirt
    race_z_mm: float  # Ball raceway centerline height above rail bottom
    race_z_low_mm: float = 0.0  # Lower track Z (HGH15 4-row only)
    race_z_high_mm: float = 0.0  # Upper track Z (HGH15 4-row only)
    slot_clr_mm: float = 0.30  # Radial/top air clearance between rail & saddle U-slot
    chamfer_mm: float = 0.50
    catalog_mass_g: float = 54.0


CARRIAGE_SPECS: dict[str, CarriageSaddleSpec] = {
    "mgn9c": CarriageSaddleSpec(
        carriage_id="mgn9c",
        rail_profile_id="mgn9_rail",
        block_type="C",
        rail_style="mgn_2row",
        width_mm=20.0,
        l1_mm=18.9,
        stack_h_mm=10.0,
        bottom_clr_h1_mm=2.0,
        rail_w_mm=9.0,
        rail_h_mm=6.5,
        bolt_b_mm=15.0,
        bolt_c_mm=10.0,
        bolt_tap_d_mm=2.6,
        bolt_tap_depth_mm=3.0,
        bolt_pad_r_mm=2.25,
        ball_d_mm=1.588,
        return_bore_d_mm=2.0,
        race_z_mm=3.9,
        slot_clr_mm=0.25,
        chamfer_mm=0.35,
        catalog_mass_g=16.0,
    ),
    "mgn9h": CarriageSaddleSpec(
        carriage_id="mgn9h",
        rail_profile_id="mgn9_rail",
        block_type="H",
        rail_style="mgn_2row",
        width_mm=20.0,
        l1_mm=28.9,
        stack_h_mm=10.0,
        bottom_clr_h1_mm=2.0,
        rail_w_mm=9.0,
        rail_h_mm=6.5,
        bolt_b_mm=15.0,
        bolt_c_mm=16.0,
        bolt_tap_d_mm=2.6,
        bolt_tap_depth_mm=3.0,
        bolt_pad_r_mm=2.25,
        ball_d_mm=1.588,
        return_bore_d_mm=2.0,
        race_z_mm=3.9,
        slot_clr_mm=0.25,
        chamfer_mm=0.35,
        catalog_mass_g=26.0,
    ),
    "mgn12c": CarriageSaddleSpec(
        carriage_id="mgn12c",
        rail_profile_id="mgn12_rail",
        block_type="C",
        rail_style="mgn_2row",
        width_mm=27.0,
        l1_mm=21.7,
        stack_h_mm=13.0,
        bottom_clr_h1_mm=3.0,
        rail_w_mm=12.0,
        rail_h_mm=8.0,
        bolt_b_mm=20.0,
        bolt_c_mm=15.0,
        bolt_tap_d_mm=2.6,
        bolt_tap_depth_mm=3.5,
        bolt_pad_r_mm=2.75,
        ball_d_mm=2.381,
        return_bore_d_mm=2.8,
        race_z_mm=4.8,
        slot_clr_mm=0.30,
        chamfer_mm=0.45,
        catalog_mass_g=34.0,
    ),
    "mgn12h": CarriageSaddleSpec(
        carriage_id="mgn12h",
        rail_profile_id="mgn12_rail",
        block_type="H",
        rail_style="mgn_2row",
        width_mm=27.0,
        l1_mm=32.4,
        stack_h_mm=13.0,
        bottom_clr_h1_mm=3.0,
        rail_w_mm=12.0,
        rail_h_mm=8.0,
        bolt_b_mm=20.0,
        bolt_c_mm=20.0,
        bolt_tap_d_mm=2.6,
        bolt_tap_depth_mm=3.5,
        bolt_pad_r_mm=2.75,
        ball_d_mm=2.381,
        return_bore_d_mm=2.8,
        race_z_mm=4.8,
        slot_clr_mm=0.30,
        chamfer_mm=0.45,
        catalog_mass_g=54.0,
    ),
    "mgn15c": CarriageSaddleSpec(
        carriage_id="mgn15c",
        rail_profile_id="mgn15_rail",
        block_type="C",
        rail_style="mgn_2row",
        width_mm=32.0,
        l1_mm=26.7,
        stack_h_mm=16.0,
        bottom_clr_h1_mm=4.0,
        rail_w_mm=15.0,
        rail_h_mm=10.0,
        bolt_b_mm=25.0,
        bolt_c_mm=20.0,
        bolt_tap_d_mm=2.6,
        bolt_tap_depth_mm=4.0,
        bolt_pad_r_mm=2.75,
        ball_d_mm=3.175,
        return_bore_d_mm=3.6,
        race_z_mm=6.0,
        slot_clr_mm=0.35,
        chamfer_mm=0.50,
        catalog_mass_g=59.0,
    ),
    "mgn15h": CarriageSaddleSpec(
        carriage_id="mgn15h",
        rail_profile_id="mgn15_rail",
        block_type="H",
        rail_style="mgn_2row",
        width_mm=32.0,
        l1_mm=43.4,
        stack_h_mm=16.0,
        bottom_clr_h1_mm=4.0,
        rail_w_mm=15.0,
        rail_h_mm=10.0,
        bolt_b_mm=25.0,
        bolt_c_mm=25.0,
        bolt_tap_d_mm=2.6,
        bolt_tap_depth_mm=4.0,
        bolt_pad_r_mm=2.75,
        ball_d_mm=3.175,
        return_bore_d_mm=3.6,
        race_z_mm=6.0,
        slot_clr_mm=0.35,
        chamfer_mm=0.50,
        catalog_mass_g=92.0,
    ),
    "hgh15ca": CarriageSaddleSpec(
        carriage_id="hgh15ca",
        rail_profile_id="hgh15_rail",
        block_type="C",
        rail_style="hg_4row",
        width_mm=34.0,
        l1_mm=39.4,
        stack_h_mm=28.0,
        bottom_clr_h1_mm=4.3,
        rail_w_mm=15.0,
        rail_h_mm=15.0,
        bolt_b_mm=26.0,
        bolt_c_mm=26.0,
        bolt_tap_d_mm=3.4,
        bolt_tap_depth_mm=5.0,
        bolt_pad_r_mm=3.50,
        ball_d_mm=3.175,
        return_bore_d_mm=3.6,
        race_z_mm=9.5,
        race_z_low_mm=7.3,
        race_z_high_mm=11.7,
        slot_clr_mm=0.45,
        chamfer_mm=0.70,
        catalog_mass_g=180.0,
    ),
}


# ---------------------------------------------------------------------------
# 3. Round Linear Ball Bushing Sleeve Specs (Misumi LM8UU / LM10UU / LM12UU)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LinearBushingSpec:
    """Misumi LM-UU / LM-LUU hardened steel outer sleeve dimensions (mm)."""

    bushing_id: str
    rail_profile_id: str
    block_type: str  # "C" (UU) | "H" (LUU)
    shaft_d_mm: float
    outer_d_mm: float
    ball_d_mm: float
    l1_mm: float  # Active loaded ball track length
    sleeve_len_mm: float  # Steel outer cylinder length
    catalog_mass_g: float


BUSHING_SPECS: dict[str, LinearBushingSpec] = {
    "lm8uu": LinearBushingSpec("lm8uu", "steel_rod_8mm", "C", 8.0, 15.0, 1.588, 17.0, 24.0, 16.0),
    "lm8luu": LinearBushingSpec("lm8luu", "steel_rod_8mm", "H", 8.0, 15.0, 1.588, 24.0, 45.0, 31.0),
    "lm10uu": LinearBushingSpec("lm10uu", "steel_rod_10mm", "C", 10.0, 19.0, 1.984, 20.0, 29.0, 31.0),
    "lm10luu": LinearBushingSpec("lm10luu", "steel_rod_10mm", "H", 10.0, 19.0, 1.984, 29.0, 55.0, 56.0),
    "lm12uu": LinearBushingSpec("lm12uu", "steel_rod_12mm", "C", 12.0, 21.0, 2.381, 24.0, 30.0, 41.0),
    "lm12luu": LinearBushingSpec("lm12luu", "steel_rod_12mm", "H", 12.0, 21.0, 2.381, 36.0, 57.0, 76.0),
}


# ---------------------------------------------------------------------------
# 4. Shared 3D Elasticity, Gmsh & scikit-fem Tet10 (P2) Solver Infrastructure
# ---------------------------------------------------------------------------


def iso_constitutive_3d(e_pa: float, nu: float) -> np.ndarray:
    c = e_pa / ((1.0 + nu) * (1.0 - 2.0 * nu))
    d_mat = np.zeros((6, 6), dtype=np.float64)
    d_mat[0:3, 0:3] = c * np.array(
        [[1.0 - nu, nu, nu], [nu, 1.0 - nu, nu], [nu, nu, 1.0 - nu]],
        dtype=np.float64,
    )
    g = e_pa / (2.0 * (1.0 + nu))
    d_mat[3, 3] = d_mat[4, 4] = d_mat[5, 5] = g
    return d_mat


def raceway_strip_constitutive_3d(e_lat_gpa: float, g_vert_gpa: float) -> np.ndarray:
    """Orthotropic Hertzian ball-contact strip (Y=lateral compression, Z=vertical shear/45-deg).

    Axial stiffness along X (E_x, G_xy, G_xz) is kept at 0.2% of E_y so balls roll freely
    along X without falsely stiffening axial bending of the carriage saddle.
    """
    ey = max(e_lat_gpa, 1e-4) * 1e9
    gz = max(g_vert_gpa, 1e-4) * 1e9
    ex = 0.002 * ey
    g_roll = 0.002 * min(ey, gz)
    s_mat = np.diag([1.0 / ex, 1.0 / ey, 1.0 / max(ey * 0.9, 1e6), 1.0 / g_roll, 1.0 / gz, 1.0 / g_roll])
    return np.linalg.inv(s_mat)


def radial_bushing_strip_constitutive_3d(e_rad_gpa: float) -> np.ndarray:
    """Axisymmetric transverse-isotropic Hertzian ball strip for cylindrical LMxUU bushings.

    Equal radial/tangential YZ stiffness (E_y = E_z = G_yz = E_rad) and free axial rolling along X.
    """
    er = max(e_rad_gpa, 1e-4) * 1e9
    ex = 0.002 * er
    g_roll = 0.002 * er
    s_mat = np.diag([1.0 / ex, 1.0 / er, 1.0 / er, 1.0 / g_roll, 1.0 / er, 1.0 / g_roll])
    return np.linalg.inv(s_mat)


def _import_solid_to_gmsh(solid: bd.Solid) -> list[tuple[int, int]]:
    with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as tf:
        tmp_path = Path(tf.name)
    try:
        bd.export_step(solid, str(tmp_path))
        ents = gmsh.model.occ.importShapes(str(tmp_path))
        return [e for e in ents if e[0] == 3]
    finally:
        tmp_path.unlink(missing_ok=True)


def _fragment_and_tag(group_lists: list[tuple[int, list[tuple[int, int]]]]) -> None:
    occ = gmsh.model.occ
    flat_inputs: list[tuple[int, int]] = []
    counts: list[tuple[int, int]] = []
    for gid, vols in group_lists:
        if vols:
            counts.append((gid, len(vols)))
            flat_inputs.extend(vols)
    if len(flat_inputs) == 1:
        occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [flat_inputs[0][1]], counts[0][0])
        return
    _, out_map = occ.fragment(flat_inputs[:1], flat_inputs[1:])
    occ.synchronize()
    claimed: set[int] = set()
    off = 0
    for gid, cnt in counts:
        raw: set[int] = set()
        for k in range(off, off + cnt):
            for d, tg in out_map[k]:
                if d == 3:
                    raw.add(tg)
        off += cnt
        owned = sorted(raw - claimed)
        claimed |= set(owned)
        if owned:
            gmsh.model.addPhysicalGroup(3, owned, gid)


def _mesh_and_assemble(
    mats_by_gid: dict[int, np.ndarray],
    h_bulk_mm: float = 2.0,
    h_min_mm: float = 0.40,
    order: int = 2,
    curvature_pts: int = 8,
) -> tuple[skfem.MeshTet, skfem.CellBasis, Any, np.ndarray]:
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h_min_mm)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h_bulk_mm)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", curvature_pts)
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
    gmsh.finalize()

    raw_tets = np.vstack(all_tets)
    elem_gid = np.concatenate(all_gids)
    ntags = np.asarray(ntags, dtype=np.int64)
    coords_mm = np.asarray(ncoords, dtype=np.float64).reshape(-1, 3)
    order_idx = np.argsort(ntags)
    pos = np.searchsorted(ntags[order_idx], raw_tets.ravel())
    idx = order_idx[pos].reshape(-1, 4)
    used, inv = np.unique(idx, return_inverse=True)
    mesh = skfem.MeshTet((coords_mm[used] * 1e-3).T, inv.reshape(-1, 4).T).oriented()

    scalar_elem = skfem.ElementTetP2() if order == 2 else skfem.ElementTetP1()
    intorder = 2 if order == 2 else 1
    element = skfem.ElementVector(scalar_elem)
    basis_full = skfem.CellBasis(mesh, element, intorder=intorder)

    k_tot = None
    for gid, d_mat in mats_by_gid.items():
        elems = np.nonzero(elem_gid == gid)[0]
        if len(elems) == 0:
            continue
        nz_rows = [[(j, float(d_mat[i, j])) for j in range(6) if abs(d_mat[i, j]) > 0.0] for i in range(6)]

        @skfem.BilinearForm
        def elas(u, v, w):
            g_u, g_v = u.grad, v.grad
            eu = (g_u[0, 0], g_u[1, 1], g_u[2, 2], g_u[0, 1] + g_u[1, 0], g_u[1, 2] + g_u[2, 1], g_u[0, 2] + g_u[2, 0])
            ev = (g_v[0, 0], g_v[1, 1], g_v[2, 2], g_v[0, 1] + g_v[1, 0], g_v[1, 2] + g_v[2, 1], g_v[0, 2] + g_v[2, 0])
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

        k_sub = elas.assemble(skfem.CellBasis(mesh, element, elements=elems, intorder=intorder))
        k_tot = k_sub if k_tot is None else (k_tot + k_sub)

    return mesh, basis_full, k_tot, elem_gid


def _solve_factored(k_mat: Any, fixed_dofs: np.ndarray, rhs_list: list[np.ndarray]) -> list[np.ndarray]:
    dummy_f = np.zeros(k_mat.shape[0], dtype=np.float64)
    k_ff, _, _, idx_free = skfem.condense(k_mat, dummy_f, D=fixed_dofs)
    lu = splu(k_ff.tocsc(), permc_spec="COLAMD")
    sols = []
    for f_vec in rhs_list:
        u = np.zeros_like(f_vec)
        u[idx_free] = lu.solve(f_vec[idx_free])
        sols.append(u)
    return sols


def _pure_hertz_target_stiffness(rail_profile_id: str, block_type: str, preload: str = "Z1") -> tuple[float, float]:
    """Return analytical pure Hertzian radial contact stiffness (k_yy_hertz, k_zz_hertz) in N/m."""
    from printsim.fea.components import _profiled_rail_c_hertz
    from printsim.schema.printer import RAIL_PRELOAD_FRACTIONS

    d_b, r_y, n_rows, l1_c, c_c, l1_h, c_h, _ = _RAIL_CARRIAGE_CATALOG[rail_profile_id]
    is_h = block_type.upper() == "H"
    l1 = l1_h if is_h else l1_c
    c_dyn = c_h if is_h else c_c
    f_pre = RAIL_PRELOAD_FRACTIONS[preload.upper()]
    n_b = max(4, int(round(l1 / (1.05 * d_b))))
    sin_alpha = math.sin(math.radians(45.0))
    e_star = E_STEEL / (2.0 * (1.0 - NU_STEEL**2))

    if rail_profile_id.startswith("steel_rod_"):
        d_shaft = 2.0 * r_y
        r_e_in = 0.5 * d_b * math.sqrt(d_shaft / (d_b + d_shaft))
        r_e_out = 0.5 * d_b * math.sqrt((d_shaft + 2.0 * d_b) / (d_shaft + d_b))
        c_in = (9.0 / (16.0 * r_e_in * (e_star**2))) ** (1.0 / 3.0)
        c_out = (9.0 / (16.0 * r_e_out * (e_star**2))) ** (1.0 / 3.0)
        c_hertz = 0.5 * (c_in + c_out)
    else:
        c_hertz = _profiled_rail_c_hertz(d_b, e_star, 0.525)

    q_ball = max((f_pre * c_dyn) / (n_rows * n_b * sin_alpha), 0.15)
    k_n = (1.5 / c_hertz) * (q_ball ** (1.0 / 3.0))
    row_factor = 0.25 * n_rows if n_rows >= 4 else 0.5 * n_rows
    k_rad = row_factor * n_b * k_n
    return k_rad, k_rad


# ---------------------------------------------------------------------------
# 5. Phase 1: Exact Solid Profiled Rail B-Rep & Tet10 Section Calibration
# ---------------------------------------------------------------------------


def bolt_positions_mm(length_mm: float, pitch_mm: float, center_on_hole: bool = False) -> list[float]:
    """Symmetric bolt X-coordinates along [-L/2, +L/2].

    When ``center_on_hole=False``, bolts sit at ``+-(k + 0.5) * pitch`` (mid-span at x = 0).
    When ``center_on_hole=True``, a bolt sits at ``x = 0`` and ``+-k * pitch``.
    """
    half = 0.5 * length_mm
    margin = 0.25 * pitch_mm
    xs: list[float] = []
    if center_on_hole:
        xs.append(0.0)
        k = 1
        while True:
            x = k * pitch_mm
            if x > half - margin + 1e-6:
                break
            xs.extend([-x, x])
            k += 1
    else:
        k = 0
        while True:
            x = (k + 0.5) * pitch_mm
            if x > half - margin + 1e-6:
                break
            xs.extend([-x, x])
            k += 1
    return sorted(xs)


def build_solid_rail_mm(
    spec: SolidRailSpec, length_mm: float, center_on_hole: bool = False
) -> bd.Solid:
    """Build exact watertight HIWIN solid linear rail B-Rep in mm (MGN9/12/15 or HGR15)."""
    w = spec.width_mm
    h = spec.height_mm
    ch = spec.chamfer_mm

    with bd.BuildSketch(bd.Plane.YZ) as sk:
        pts_outer = [
            (-0.5 * w + ch, -0.5 * h),
            (0.5 * w - ch, -0.5 * h),
            (0.5 * w, -0.5 * h + ch),
            (0.5 * w, 0.5 * h - ch),
            (0.5 * w - ch, 0.5 * h),
            (-0.5 * w + ch, 0.5 * h),
            (-0.5 * w, 0.5 * h - ch),
            (-0.5 * w, -0.5 * h + ch),
        ]
        with bd.BuildLine():
            bd.Polyline(*pts_outer, close=True)
        bd.make_face()

        if spec.rail_style == "mgn_2row":
            z_race = -0.5 * h + spec.race_z_from_bottom_mm
            gh = spec.race_groove_h_mm
            gd = spec.race_groove_d_mm
            for s in (-1.0, 1.0):
                y_root = s * (0.5 * w - gd)
                y_out = s * (0.5 * w + 0.5)
                with bd.BuildLine():
                    p1 = (y_out, z_race - 0.5 * gh)
                    p2 = (s * (0.5 * w - 0.35 * gd), z_race - 0.5 * gh)
                    p3 = (y_root, z_race)
                    p4 = (s * (0.5 * w - 0.35 * gd), z_race + 0.5 * gh)
                    p5 = (y_out, z_race + 0.5 * gh)
                    bd.Line(p1, p2)
                    bd.ThreePointArc(p2, p3, p4)
                    bd.Line(p4, p5)
                    bd.Line(p5, p1)
                bd.make_face(mode=bd.Mode.SUBTRACT)
        else:
            # HGR15R 4-row circular-arc waisted profile: twin grooves at z_low and z_high
            # plus recessed central waist and lower skirt relief matching HIWIN 1.45 kg/m.
            z_lo = -0.5 * h + spec.race_z_low_mm
            z_hi = -0.5 * h + spec.race_z_high_mm
            gh = spec.race_groove_h_mm
            gd = spec.race_groove_d_mm
            wd = spec.waist_depth_mm
            for s in (-1.0, 1.0):
                y_out = s * (0.5 * w + 0.5)
                y_waist = s * (0.5 * w - wd)
                # Central waist recess between lower groove bottom and upper groove top
                with bd.BuildLine():
                    bd.Polyline(
                        (y_out, z_lo - 0.5 * gh - 0.3),
                        (s * (0.5 * w - 0.5 * wd), z_lo - 0.5 * gh - 0.3),
                        (y_waist, z_lo - 0.2),
                        (y_waist, z_hi + 0.2),
                        (s * (0.5 * w - 0.5 * wd), z_hi + 0.5 * gh + 0.3),
                        (y_out, z_hi + 0.5 * gh + 0.3),
                        close=True,
                    )
                bd.make_face(mode=bd.Mode.SUBTRACT)
                # Twin circular-arc ball grooves at z_lo and z_hi
                for z_g in (z_lo, z_hi):
                    y_root = s * (0.5 * w - max(gd, wd))
                    with bd.BuildLine():
                        p1 = (y_out, z_g - 0.5 * gh)
                        p2 = (s * (0.5 * w - 0.45 * gd), z_g - 0.5 * gh)
                        p3 = (y_root, z_g)
                        p4 = (s * (0.5 * w - 0.45 * gd), z_g + 0.5 * gh)
                        p5 = (y_out, z_g + 0.5 * gh)
                        bd.Line(p1, p2)
                        bd.ThreePointArc(p2, p3, p4)
                        bd.Line(p4, p5)
                        bd.Line(p5, p1)
                    bd.make_face(mode=bd.Mode.SUBTRACT)

    face_sec = sk.sketch.faces()[0]
    rail = bd.extrude(face_sec, amount=length_mm, both=False).translate(bd.Vector(-0.5 * length_mm, 0.0, 0.0))

    # Subtract exact M3/M4 counterbored mounting holes on exact catalog pitch
    bolts = bolt_positions_mm(length_mm, spec.pitch_mm, center_on_hole=center_on_hole)
    cutters: list[bd.Solid] = []
    z_top = 0.5 * h
    for xb in bolts:
        cutters.append(
            bd.Cylinder(
                radius=0.5 * spec.thru_d_mm,
                height=h + 1.0,
                align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.CENTER),
            ).translate(bd.Vector(xb, 0.0, 0.0))
        )
        cutters.append(
            bd.Cylinder(
                radius=0.5 * spec.cbore_d_mm,
                height=spec.cbore_depth_mm + 0.5,
                align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MAX),
            ).translate(bd.Vector(xb, 0.0, z_top + 0.5))
        )
    if cutters:
        rail = rail - bd.Compound(cutters)

    solids = rail.solids()
    if len(solids) != 1:
        raise RuntimeError(f"Expected 1 solid for {spec.profile_id}, got {len(solids)}")
    return solids[0]


def run_solid_rail_calibration(
    profile_id: str,
    h_bulk_mm: float | None = None,
    h_min_mm: float = 0.45,
    order: int = 2,
) -> dict[str, Any]:
    """Run full 3D Tet10 section calibration on a solid HIWIN rail (A, Iy, Iz, J, kappa_y, kappa_z)."""
    t0 = time.perf_counter()
    spec = SOLID_RAIL_SPECS[profile_id]
    if h_bulk_mm is None:
        h_bulk_mm = 2.4 if spec.rail_style == "hg_4row" else 2.0
    l_mm = spec.bare_fea_length_mm
    l_m = l_mm * 1e-3

    solid = build_solid_rail_mm(spec, length_mm=l_mm)
    area_eff_mm2 = float(solid.volume) / l_mm
    mass_g_per_m = area_eff_mm2 * 1e-6 * RHO_STEEL * 1e3

    gmsh.initialize(["gmsh"], interruptible=False)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(f"solid_rail_{profile_id}")
    _fragment_and_tag([(1, _import_solid_to_gmsh(solid))])

    d_steel = iso_constitutive_3d(E_STEEL, NU_STEEL)
    mesh, basis, k_mat, _ = _mesh_and_assemble(
        {1: d_steel}, h_bulk_mm=h_bulk_mm, h_min_mm=h_min_mm, order=order, curvature_pts=6
    )

    dl = basis.doflocs
    tol = 1.5e-4
    half_l = 0.5 * l_m
    comp = np.arange(basis.N) % 3

    # Simply supported bending BCs at x = +-L/2
    fix_ends_yz = np.nonzero((np.abs(np.abs(dl[0]) - half_l) < tol) & (comp != 0))[0]
    mid_ux = np.nonzero((np.abs(dl[0]) < 0.004) & (np.abs(dl[1]) < 0.002) & (comp == 0))[0]
    fix_bend = np.unique(np.concatenate([fix_ends_yz, mid_ux[:2]]))

    f_val = 20.0
    pitch_m = spec.pitch_mm * 1e-3
    n_p_half = max(2, int(round(half_l / pitch_m)))
    a_pitches = max(2, n_p_half // 2)
    q_pitches = max(1, a_pitches - 1)
    a_4pt = a_pitches * pitch_m
    x_q = q_pitches * pitch_m
    z_top_m = 0.5 * spec.height_mm * 1e-3
    y_side_m = 0.5 * spec.width_mm * 1e-3

    # 4-pt Z bending (Iy) and 4-pt Y bending (Iz)
    f_4z = basis.zeros()
    f_4y = basis.zeros()
    for xs in (-a_4pt, a_4pt):
        d4z = np.nonzero((np.abs(dl[0] - xs) <= 0.003) & (dl[2] >= z_top_m - 0.0015) & (comp == 2))[0]
        f_4z[d4z] = -(0.5 * f_val) / len(d4z)
        d4y = np.nonzero((np.abs(dl[0] - xs) <= 0.003) & (np.abs(dl[1]) >= y_side_m - 0.0015) & (comp == 1))[0]
        f_4y[d4y] = -(0.5 * f_val) / len(d4y)

    u_4z, u_4y = _solve_factored(k_mat, fix_bend, [f_4z, f_4y])
    v_mid = np.nonzero(np.abs(mesh.p[0]) <= 0.003)[0]
    v_q = np.nonzero(np.abs(np.abs(mesh.p[0]) - x_q) <= 0.003)[0]

    m0_4pt = (0.5 * f_val) * (half_l - a_4pt)
    sag_z = abs(float(np.mean(u_4z[basis.nodal_dofs[2, v_mid]])) - float(np.mean(u_4z[basis.nodal_dofs[2, v_q]])))
    sag_y = abs(float(np.mean(u_4y[basis.nodal_dofs[1, v_mid]])) - float(np.mean(u_4y[basis.nodal_dofs[1, v_q]])))
    ei_y_4pt = m0_4pt * (x_q**2) / (2.0 * max(sag_z, 1e-15))
    ei_z_4pt = m0_4pt * (x_q**2) / (2.0 * max(sag_y, 1e-15))

    # Free-warping St. Venant torsion over interior periodic span [-L/4, +L/4]
    fix_tor_yz = np.nonzero((np.abs(dl[0] + half_l) < tol) & (comp != 0))[0]
    fix_tor_x1 = np.nonzero((np.abs(dl[0] + half_l) < tol) & (comp == 0))[0][:1]
    fix_tor_free = np.unique(np.concatenate([fix_tor_yz, fix_tor_x1]))
    fix_tor_clamp = np.nonzero(np.abs(dl[0] + half_l) < tol)[0]

    f_right = np.nonzero(np.all(np.abs(mesh.p[0, mesh.facets] - half_l) < tol, axis=0))[0]
    fb_right = skfem.FacetBasis(mesh, skfem.ElementVector(basis.elem.elem), facets=f_right, intorder=2)
    area_end = float(np.sum(fb_right.dx))

    @skfem.Functional
    def zc_num(w):
        return w.x[2]

    z_c = float(zc_num.assemble(fb_right)) / max(area_end, 1e-12)

    @skfem.Functional
    def ip_fn(w):
        return w.x[1] ** 2 + (w.x[2] - z_c) ** 2

    ip_end = float(ip_fn.assemble(fb_right))
    t_x = 2.0

    @skfem.LinearForm
    def form_tor(v, w):
        return (t_x / ip_end) * (-(w.x[2] - z_c) * v[1] + w.x[1] * v[2])

    rhs_tor = form_tor.assemble(fb_right)
    (u_tor_free,) = _solve_factored(k_mat, fix_tor_free, [rhs_tor])
    (u_tor_clamp,) = _solve_factored(k_mat, fix_tor_clamp, [rhs_tor])

    def _twist_at_x(u_vec: np.ndarray, x_target: float) -> float:
        vs = np.nonzero(np.abs(mesh.p[0] - x_target) <= 0.0025)[0]
        yv = mesh.p[1, vs]
        zv = mesh.p[2, vs] - z_c
        uyv = u_vec[basis.nodal_dofs[1, vs]]
        uzv = u_vec[basis.nodal_dofs[2, vs]]
        return float(np.sum(yv * uzv - zv * uyv) / np.sum(yv**2 + zv**2))

    dtheta_free = _twist_at_x(u_tor_free, +0.25 * l_m) - _twist_at_x(u_tor_free, -0.25 * l_m)
    gj_free = t_x * (0.5 * l_m) / max(abs(dtheta_free), 1e-15)
    theta_clamp = _twist_at_x(u_tor_clamp, half_l)
    gj_clamp = t_x * l_m / max(abs(theta_clamp), 1e-15)

    # Self-consistent 1-pitch periodic shear stub (exact 1 bolt hole per pitch at x = 0):
    # Measure BOTH pure-bending EI_stub and double-cantilever shear on the EXACT SAME mesh
    # so mesh & hole-phase differences cancel identically!
    l_sh_mm = spec.pitch_mm
    l_sh_m = l_sh_mm * 1e-3
    sh_solid = build_solid_rail_mm(spec, length_mm=l_sh_mm, center_on_hole=True)
    gmsh.initialize(["gmsh"], interruptible=False)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(f"solid_shear_{profile_id}")
    _fragment_and_tag([(1, _import_solid_to_gmsh(sh_solid))])
    mesh_s, basis_s, k_s_mat, _ = _mesh_and_assemble(
        {1: d_steel}, h_bulk_mm=min(h_bulk_mm, 1.6), h_min_mm=h_min_mm, order=order, curvature_pts=6
    )
    dls = basis_s.doflocs
    comps = np.arange(basis_s.N) % 3
    half_sh = 0.5 * l_sh_m
    on_left = np.abs(dls[0] + half_sh) < tol
    on_right = np.abs(dls[0] - half_sh) < tol

    f_sh_r = np.nonzero(np.all(np.abs(mesh_s.p[0, mesh_s.facets] - half_sh) < tol, axis=0))[0]
    fb_sh_r = skfem.FacetBasis(mesh_s, skfem.ElementVector(basis_s.elem.elem), facets=f_sh_r, intorder=2)
    area_sh_r = float(np.sum(fb_sh_r.dx))

    @skfem.Functional
    def zc_sh_fn(w):
        return w.x[2]

    z_c_sh = float(zc_sh_fn.assemble(fb_sh_r)) / max(area_sh_r, 1e-12)

    @skfem.Functional
    def iy_end_fn(w):
        return (w.x[2] - z_c_sh) ** 2

    @skfem.Functional
    def iz_end_fn(w):
        return w.x[1] ** 2

    iy_end = float(iy_end_fn.assemble(fb_sh_r))
    iz_end = float(iz_end_fn.assemble(fb_sh_r))

    # Free-Poisson cantilever BCs on left face (x = -L/2): u_x = 0 across face, u_y/u_z pinned on centerlines
    fix_cant_y = np.unique(
        np.concatenate(
            [
                np.nonzero(on_left & (comps == 0))[0],
                np.nonzero(on_left & (comps == 1))[0],
                np.nonzero(on_left & (np.abs(dls[1]) < 0.0012) & (comps == 2))[0],
            ]
        )
    )
    fix_cant_z = np.unique(
        np.concatenate(
            [
                np.nonzero(on_left & (comps == 0))[0],
                np.nonzero(on_left & (comps == 2))[0],
                np.nonzero(on_left & (np.abs(dls[2] - z_c_sh) < 0.0012) & (comps == 1))[0],
            ]
        )
    )
    # Guided double-cantilever BCs add u_x = 0 on right face (zero rotation -> V*L^3/(12*EI) + V*L/GA)
    fix_sh_y = np.unique(np.concatenate([fix_cant_y, np.nonzero(on_right & (comps == 0))[0]]))
    fix_sh_z = np.unique(np.concatenate([fix_cant_z, np.nonzero(on_right & (comps == 0))[0]]))

    # Pure end moments M_z = 1 N*m (bends along Y) & M_y = 1 N*m (bends along Z) on right end face
    @skfem.LinearForm
    def form_mz(v, w):
        return (1.0 / iz_end) * w.x[1] * v[0]

    @skfem.LinearForm
    def form_my(v, w):
        return (1.0 / iy_end) * (w.x[2] - z_c_sh) * v[0]

    @skfem.LinearForm
    def form_sy(v, w):
        return (100.0 / area_sh_r) * v[1]

    @skfem.LinearForm
    def form_sz(v, w):
        return (100.0 / area_sh_r) * v[2]

    (u_mz,) = _solve_factored(k_s_mat, fix_cant_y, [form_mz.assemble(fb_sh_r)])
    (u_my,) = _solve_factored(k_s_mat, fix_cant_z, [form_my.assemble(fb_sh_r)])
    (u_sy,) = _solve_factored(k_s_mat, fix_sh_y, [form_sy.assemble(fb_sh_r)])
    (u_sz,) = _solve_factored(k_s_mat, fix_sh_z, [form_sz.assemble(fb_sh_r)])

    @skfem.Functional
    def th_z_fn(w):
        return w.x[1] * w.u[0]

    @skfem.Functional
    def th_y_fn(w):
        return (w.x[2] - z_c_sh) * w.u[0]

    @skfem.Functional
    def uy_int(w):
        return w.u[1]

    @skfem.Functional
    def uz_int(w):
        return w.u[2]

    th_z_stub = abs(float(th_z_fn.assemble(fb_sh_r, u=fb_sh_r.interpolate(u_mz))) / iz_end)
    th_y_stub = abs(float(th_y_fn.assemble(fb_sh_r, u=fb_sh_r.interpolate(u_my))) / iy_end)
    ei_z_stub = 1.0 * l_sh_m / max(th_z_stub, 1e-15)
    ei_y_stub = 1.0 * l_sh_m / max(th_y_stub, 1e-15)

    dy_sh_tot = abs(float(uy_int.assemble(fb_sh_r, u=fb_sh_r.interpolate(u_sy))) / area_sh_r)
    dy_sh_pure = max(dy_sh_tot - 100.0 * (l_sh_m**3) / (12.0 * ei_z_stub), 1e-15)
    ga_y = 100.0 * l_sh_m / dy_sh_pure

    # For vertical Z-shear kappa_z, use a short 2*H grooved section stub (free of hole-phase
    # M(0)=0 bias where the top counterbore softens EI_y_stub relative to the un-drilled ends).
    l_z_mm = 2.0 * spec.height_mm
    l_z_m = l_z_mm * 1e-3
    spec_grooved = SolidRailSpec(**{**spec.__dict__, "pitch_mm": 9999.0})
    gz_solid = build_solid_rail_mm(spec_grooved, length_mm=l_z_mm, center_on_hole=False)
    area_grooved_m2 = (float(gz_solid.volume) / l_z_mm) * 1e-6
    gmsh.initialize(["gmsh"], interruptible=False)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(f"solid_shearz_{profile_id}")
    _fragment_and_tag([(1, _import_solid_to_gmsh(gz_solid))])
    mesh_z, basis_z, k_z_mat, _ = _mesh_and_assemble(
        {1: d_steel}, h_bulk_mm=min(h_bulk_mm, 1.6), h_min_mm=min(h_min_mm, 0.35), order=order, curvature_pts=6
    )
    dlz = basis_z.doflocs
    compz = np.arange(basis_z.N) % 3
    half_z = 0.5 * l_z_m
    on_lz = np.abs(dlz[0] + half_z) < tol
    on_rz = np.abs(dlz[0] - half_z) < tol
    f_zr = np.nonzero(np.all(np.abs(mesh_z.p[0, mesh_z.facets] - half_z) < tol, axis=0))[0]
    fb_zr = skfem.FacetBasis(mesh_z, skfem.ElementVector(basis_z.elem.elem), facets=f_zr, intorder=2)
    area_zr = float(np.sum(fb_zr.dx))

    @skfem.Functional
    def zc_zr_fn(w):
        return w.x[2]

    z_c_zr = float(zc_zr_fn.assemble(fb_zr)) / max(area_zr, 1e-12)

    @skfem.Functional
    def iy_zr_fn(w):
        return (w.x[2] - z_c_zr) ** 2

    iy_zr = float(iy_zr_fn.assemble(fb_zr))
    fix_cz = np.unique(
        np.concatenate(
            [
                np.nonzero(on_lz & (compz == 0))[0],
                np.nonzero(on_lz & (compz == 2))[0],
                np.nonzero(on_lz & (np.abs(dlz[2] - z_c_zr) < 0.0012) & (compz == 1))[0],
            ]
        )
    )
    fix_sz_g = np.unique(np.concatenate([fix_cz, np.nonzero(on_rz & (compz == 0))[0]]))

    @skfem.LinearForm
    def form_my_z(v, w):
        return (1.0 / iy_zr) * (w.x[2] - z_c_zr) * v[0]

    @skfem.LinearForm
    def form_sz_z(v, w):
        return (100.0 / area_zr) * v[2]

    (u_my_z,) = _solve_factored(k_z_mat, fix_cz, [form_my_z.assemble(fb_zr)])
    (u_sz_z,) = _solve_factored(k_z_mat, fix_sz_g, [form_sz_z.assemble(fb_zr)])

    @skfem.Functional
    def thy_z_fn(w):
        return (w.x[2] - z_c_zr) * w.u[0]

    @skfem.Functional
    def uz_z_fn(w):
        return w.u[2]

    ei_y_grooved = l_z_m / max(abs(float(thy_z_fn.assemble(fb_zr, u=fb_zr.interpolate(u_my_z))) / iy_zr), 1e-15)
    dz_g_tot = abs(float(uz_z_fn.assemble(fb_zr, u=fb_zr.interpolate(u_sz_z))) / area_zr)
    dz_g_pure = max(dz_g_tot - 100.0 * (l_z_m**3) / (12.0 * ei_y_grooved), 1e-15)
    kappa_z = (100.0 * l_z_m / dz_g_pure) / (G_STEEL * area_grooved_m2)

    # Compute kappa_y relative to HIWIN catalog steel area (used in ProfileSpec) & B-Rep area
    area_cat_m2 = (spec.catalog_mass_g_per_m * 1e-3) / RHO_STEEL
    area_brep_m2 = area_eff_mm2 * 1e-6
    kappa_y = ga_y / (G_STEEL * area_cat_m2)
    ga_z = kappa_z * G_STEEL * area_cat_m2

    iy_mm4 = (ei_y_4pt / E_STEEL) * 1e12
    iz_mm4 = (ei_z_4pt / E_STEEL) * 1e12
    j_free_mm4 = (gj_free / G_STEEL) * 1e12
    j_clamp_mm4 = (gj_clamp / G_STEEL) * 1e12

    # Naive solid rectangle baseline for comparison
    rect_iy = spec.width_mm * (spec.height_mm**3) / 12.0
    rect_iz = spec.height_mm * (spec.width_mm**3) / 12.0

    return {
        "profile_id": profile_id,
        "order": order,
        "h_bulk_mm": h_bulk_mm,
        "tets": int(mesh.nelements),
        "dofs": int(basis.N),
        "area_brep_mm2": round(area_eff_mm2, 3),
        "area_catalog_mm2": round(area_cat_m2 * 1e6, 3),
        "mass_brep_g_per_m": round(mass_g_per_m, 2),
        "mass_catalog_g_per_m": spec.catalog_mass_g_per_m,
        "iy_mm4": round(iy_mm4, 2),
        "iz_mm4": round(iz_mm4, 2),
        "j_mm4": round(j_free_mm4, 2),
        "j_clamp_mm4": round(j_clamp_mm4, 2),
        "iy_vs_rect_pct": round(100.0 * iy_mm4 / rect_iy, 1),
        "iz_vs_rect_pct": round(100.0 * iz_mm4 / rect_iz, 1),
        "ga_y_mn": round(ga_y * 1e-6, 3),
        "ga_z_mn": round(ga_z * 1e-6, 3),
        "shear_kappa_y": round(min(kappa_y, 1.0), 3),
        "shear_kappa_z": round(min(kappa_z, 1.0), 3),
        "kappa_y_brep_area": round(ga_y / (G_STEEL * area_brep_m2), 3),
        "kappa_z_brep_area": round(ga_z / (G_STEEL * area_brep_m2), 3),
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }


# ---------------------------------------------------------------------------
# 6. Phase 2: All 7 HIWIN Profiled Carriages — 3D Tet10 Saddle Compliance
# ---------------------------------------------------------------------------


def build_carriage_saddle_mm(spec: CarriageSaddleSpec) -> bd.Solid:
    """Build exact watertight HIWIN steel C-block carriage saddle B-Rep in mm.

    Coordinate frame:
    - X in [-L1/2, +L1/2]: along rail travel
    - Y in [-W/2, +W/2]: transverse carriage width
    - Z in [H1, H]: vertical height above rail bottom face (z = 0 is rail mounting base)
    """
    w = spec.width_mm
    l1 = spec.l1_mm
    z_bot = spec.bottom_clr_h1_mm
    z_top = spec.stack_h_mm
    ch = spec.chamfer_mm
    u_half_w = 0.5 * spec.rail_w_mm + spec.slot_clr_mm
    u_top_z = spec.rail_h_mm + spec.slot_clr_mm

    with bd.BuildSketch(bd.Plane.YZ) as sk:
        if spec.rail_style == "hg_4row":
            # Classic HIWIN HGH15CA shouldered block profile: full W=34 mm upper crown,
            # slight lower skirt bevel below mid-height
            z_step = z_bot + 0.42 * (z_top - z_bot)
            w_low = w - 2.4
            pts = [
                (-0.5 * w_low + ch, z_bot),
                (0.5 * w_low - ch, z_bot),
                (0.5 * w_low, z_bot + ch),
                (0.5 * w_low, z_step),
                (0.5 * w, z_step + 1.2),
                (0.5 * w, z_top - ch),
                (0.5 * w - ch, z_top),
                (-0.5 * w + ch, z_top),
                (-0.5 * w, z_top - ch),
                (-0.5 * w, z_step + 1.2),
                (-0.5 * w_low, z_step),
                (-0.5 * w_low, z_bot + ch),
            ]
        else:
            pts = [
                (-0.5 * w + ch, z_bot),
                (0.5 * w - ch, z_bot),
                (0.5 * w, z_bot + ch),
                (0.5 * w, z_top - ch),
                (0.5 * w - ch, z_top),
                (-0.5 * w + ch, z_top),
                (-0.5 * w, z_top - ch),
                (-0.5 * w, z_bot + ch),
            ]
        with bd.BuildLine():
            bd.Polyline(*pts, close=True)
        bd.make_face()

        # Subtract inner U-slot clearing the rail
        with bd.BuildLine():
            bd.Polyline(
                (-u_half_w, z_bot - 1.0),
                (u_half_w, z_bot - 1.0),
                (u_half_w, u_top_z),
                (-u_half_w, u_top_z),
                close=True,
            )
        bd.make_face(mode=bd.Mode.SUBTRACT)

        # Subtract inner ball-raceway grooves in both skirts
        gd = 0.45 * spec.ball_d_mm
        gh = 1.05 * spec.ball_d_mm
        z_tracks = (spec.race_z_low_mm, spec.race_z_high_mm) if spec.rail_style == "hg_4row" else (spec.race_z_mm,)
        for z_t in z_tracks:
            for s in (-1.0, 1.0):
                y_in = s * (u_half_w - 0.5)
                y_edge = s * u_half_w
                y_root = s * (u_half_w + gd)
                with bd.BuildLine():
                    p1 = (y_in, z_t - 0.5 * gh)
                    p2 = (y_edge, z_t - 0.5 * gh)
                    p3 = (y_root, z_t)
                    p4 = (y_edge, z_t + 0.5 * gh)
                    p5 = (y_in, z_t + 0.5 * gh)
                    bd.Line(p1, p2)
                    bd.ThreePointArc(p2, p3, p4)
                    bd.Line(p4, p5)
                    bd.Line(p5, p1)
                bd.make_face(mode=bd.Mode.SUBTRACT)

        # Subtract longitudinal ball-return bores inside the two side skirts
        r_ret = 0.5 * spec.return_bore_d_mm
        if spec.rail_style == "hg_4row":
            y_ret_c = 0.5 * (u_half_w + gd + 0.5 * (w - 2.4))
            for s in (-1.0, 1.0):
                for z_b in (spec.race_z_low_mm + 0.8, spec.race_z_high_mm + 2.4):
                    with bd.Locations((s * y_ret_c, z_b)):
                        bd.Circle(r_ret, mode=bd.Mode.SUBTRACT)
        else:
            y_ret_c = 0.5 * (u_half_w + gd + 0.5 * w)
            z_ret_c = spec.race_z_mm + 0.25 * (z_top - spec.race_z_mm)
            for s in (-1.0, 1.0):
                with bd.Locations((s * y_ret_c, z_ret_c)):
                    bd.Circle(r_ret, mode=bd.Mode.SUBTRACT)

    face_sec = sk.sketch.faces()[0]
    saddle = bd.extrude(face_sec, amount=l1, both=False).translate(bd.Vector(-0.5 * l1, 0.0, 0.0))

    # Subtract 4x blind tapped mounting holes at (+-C/2, +-B/2) from top face z = H
    cutters: list[bd.Solid] = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            xh = sx * 0.5 * spec.bolt_c_mm
            yh = sy * 0.5 * spec.bolt_b_mm
            cutters.append(
                bd.Cylinder(
                    radius=0.5 * spec.bolt_tap_d_mm,
                    height=spec.bolt_tap_depth_mm + 0.5,
                    align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MAX),
                ).translate(bd.Vector(xh, yh, z_top + 0.5))
            )
    saddle = saddle - bd.Compound(cutters)
    solids = saddle.solids()
    if len(solids) != 1:
        raise RuntimeError(f"Expected 1 solid for carriage {spec.carriage_id}, got {len(solids)}")
    return solids[0]


def _solve_carriage_saddle_once(
    spec: CarriageSaddleSpec,
    e_saddle_pa: float,
    e_lat_gpa: float,
    g_vert_gpa: float,
    h_bulk_mm: float = 1.6,
    h_min_mm: float = 0.35,
    order: int = 2,
) -> dict[str, float]:
    """Solve 5-DOF carriage+Hertzian contact compliance referenced to ball-raceway center (0,0,z_race)."""
    saddle_solid = build_carriage_saddle_mm(spec)
    saddle_mass_g = float(saddle_solid.volume) * 1e-9 * RHO_STEEL * 1e3

    gmsh.initialize(["gmsh"], interruptible=False)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(f"car_{spec.carriage_id}")
    occ = gmsh.model.occ

    saddle_vols = _import_solid_to_gmsh(saddle_solid)

    l1 = spec.l1_mm
    z_top = spec.stack_h_mm
    u_half_w = 0.5 * spec.rail_w_mm + spec.slot_clr_mm
    gd = 0.45 * spec.ball_d_mm
    strip_h = 0.85 * spec.ball_d_mm

    # Grounded rigid rail core inside the U-slot, sized so the Hertzian ball-contact strips are
    # tightly centered on the exact ball-track pitch line y = +-W_R / 2 (preserving exact r_y = W_R/2).
    core_w = spec.rail_w_mm - 2.0 * (0.45 * spec.ball_d_mm)
    core_z0 = spec.bottom_clr_h1_mm - 1.5
    core_h = (spec.rail_h_mm - 0.4) - core_z0
    rail_core = occ.addBox(-0.5 * l1 - 1.0, -0.5 * core_w, core_z0, l1 + 2.0, core_w, core_h)

    # Calibrated Hertzian ball-contact strips bridging rail core <-> carriage inner raceway grooves
    z_tracks = (spec.race_z_low_mm, spec.race_z_high_mm) if spec.rail_style == "hg_4row" else (spec.race_z_mm,)
    strip_vols: list[tuple[int, int]] = []
    for z_t in z_tracks:
        y_span = (u_half_w + gd + 0.25) - (0.5 * core_w - 0.25)
        s_neg = occ.addBox(-0.5 * l1, -(u_half_w + gd + 0.25), z_t - 0.5 * strip_h, l1, y_span, strip_h)
        s_pos = occ.addBox(-0.5 * l1, 0.5 * core_w - 0.25, z_t - 0.5 * strip_h, l1, y_span, strip_h)
        strip_vols.extend([(3, s_neg), (3, s_pos)])

    # Rigid top mounting platen seated flush across the precision-ground 4-bolt mounting pad footprint
    # ((C + 2*r_pad) x (B + 2*r_pad)) on z = H, with a 0.40 mm relief gap outside the bolted pad so
    # overhanging saddle ends (L1 > C + 2*r_pad) and downward-hanging C-block skirts flex freely!
    pad_gap = 0.40
    platen_h = 3.0
    pad_lx = min(spec.bolt_c_mm + 2.0 * spec.bolt_pad_r_mm, l1 - 0.4)
    pad_wy = min(spec.bolt_b_mm + 2.0 * spec.bolt_pad_r_mm, spec.width_mm - 2.0 * spec.chamfer_mm - 0.2)
    platen_box = occ.addBox(-0.5 * l1, -0.5 * spec.width_mm, z_top + pad_gap, l1, spec.width_mm, platen_h)
    bolted_pad = occ.addBox(-0.5 * pad_lx, -0.5 * pad_wy, z_top - 0.10, pad_lx, pad_wy, pad_gap + 0.40)
    platen_fused, _ = occ.fuse([(3, platen_box)], [(3, bolted_pad)])

    # Priority: saddle (1) > rigid rail core (2) > rigid top platen/pads (3) > Hertzian strips (10)
    # Wait: give platen pads exact flush contact at z = z_top by letting saddle claim z <= z_top!
    _fragment_and_tag(
        [
            (1, saddle_vols),
            (2, [(3, rail_core)]),
            (3, platen_fused),
            (10, strip_vols),
        ]
    )

    mats = {
        1: iso_constitutive_3d(e_saddle_pa, NU_STEEL),
        2: iso_constitutive_3d(E_RIGID, 0.30),
        3: iso_constitutive_3d(E_RIGID, 0.30),
        10: raceway_strip_constitutive_3d(e_lat_gpa, g_vert_gpa),
    }
    mesh, basis, k_mat, elem_gid = _mesh_and_assemble(
        mats, h_bulk_mm=h_bulk_mm, h_min_mm=h_min_mm, order=order, curvature_pts=6
    )

    dl = basis.doflocs
    tol = 1.2e-4
    comp = np.arange(basis.N) % 3

    # Clamp grounded rigid rail core completely (u_x = u_y = u_z = 0)
    core_dofs = np.unique(basis.element_dofs[:, elem_gid == 2])
    # Also fix 1 axial DOF on platen center only when needed? No — Hertzian strip has 0.2% axial
    # stiffness so u_x is weakly grounded (well-conditioned SPD) without stiffening bending/pitch!
    fix_dofs = core_dofs

    platen_dofs = np.unique(basis.element_dofs[:, elem_gid == 3])
    is_platen = np.zeros(basis.N, dtype=bool)
    is_platen[platen_dofs] = True
    z_platen_top = float(np.max(dl[2, platen_dofs]))
    top_mask = is_platen & (np.abs(dl[2] - z_platen_top) < tol)

    d_fx = np.nonzero(top_mask & (comp == 0))[0]
    d_fy = np.nonzero(top_mask & (comp == 1))[0]
    d_fz = np.nonzero(top_mask & (comp == 2))[0]

    x_top_z = dl[0, d_fz]
    y_top_z = dl[1, d_fz]
    x_top_y = dl[0, d_fy]
    y_top_x = dl[1, d_fx]
    dz_arm = z_platen_top - (spec.race_z_mm * 1e-3)

    # 1. Pure lateral F_y = 50 N at z = z_race (compensate moment +M_x = +F_y * dz_arm on top face)
    f_lat = basis.zeros()
    f_lat[d_fy] = 50.0 / len(d_fy)
    f_lat[d_fz] = (50.0 * dz_arm / np.sum(y_top_z**2)) * y_top_z

    # 2. Pure vertical F_z = 50 N (pull-up tension opening C-block arch & skirts)
    f_vert = basis.zeros()
    f_vert[d_fz] = 50.0 / len(d_fz)

    # 3. Pure roll M_x = 5.0 N*m about X axis through (0, 0, z_race)
    f_roll = basis.zeros()
    f_roll[d_fz] = (5.0 / np.sum(y_top_z**2)) * y_top_z

    # 4. Pure pitch M_y = 5.0 N*m about Y axis through (0, 0, z_race): (x,0,0)x(0,0,Fz)=(0,-x*Fz,0)
    f_pitch = basis.zeros()
    f_pitch[d_fz] = -(5.0 / np.sum(x_top_z**2)) * x_top_z

    # 5. Pure yaw M_z = 5.0 N*m about Z axis: f_y = +M_z * x / sum(x^2 + y^2), f_x = -M_z * y / sum(...)
    f_yaw = basis.zeros()
    ip_xy = float(np.sum(x_top_y**2) + np.sum(y_top_x**2))
    f_yaw[d_fy] = +(5.0 / ip_xy) * x_top_y
    f_yaw[d_fx] = -(5.0 / ip_xy) * y_top_x

    u_lat, u_vert, u_roll, u_pitch, u_yaw = _solve_factored(
        k_mat, fix_dofs, [f_lat, f_vert, f_roll, f_pitch, f_yaw]
    )

    platen_verts = np.unique(mesh.t[:, elem_gid == 3])
    top_v = platen_verts[np.abs(mesh.p[2, platen_verts] - z_platen_top) < tol]
    xt = mesh.p[0, top_v]
    yt = mesh.p[1, top_v]

    # Project rigid-body platen motion back to ball-raceway center (0, 0, z_race)
    th_x_lat = float(np.sum(yt * u_lat[basis.nodal_dofs[2, top_v]]) / np.sum(yt**2))
    uy_race_lat = float(np.mean(u_lat[basis.nodal_dofs[1, top_v]])) + th_x_lat * dz_arm
    uz_vert = float(np.mean(u_vert[basis.nodal_dofs[2, top_v]]))
    th_roll = float(np.sum(yt * u_roll[basis.nodal_dofs[2, top_v]]) / np.sum(yt**2))
    th_pitch = -float(np.sum(xt * u_pitch[basis.nodal_dofs[2, top_v]]) / np.sum(xt**2))
    th_yaw = float(
        np.sum(xt * u_yaw[basis.nodal_dofs[1, top_v]] - yt * u_yaw[basis.nodal_dofs[0, top_v]])
        / np.sum(xt**2 + yt**2)
    )

    return {
        "k_yy_n_m": 50.0 / max(abs(uy_race_lat), 1e-15),
        "k_zz_n_m": 50.0 / max(abs(uz_vert), 1e-15),
        "k_rx_nm_rad": 5.0 / max(abs(th_roll), 1e-15),
        "k_ry_nm_rad": 5.0 / max(abs(th_pitch), 1e-15),
        "k_rz_nm_rad": 5.0 / max(abs(th_yaw), 1e-15),
        "saddle_mass_g": saddle_mass_g,
        "tets": int(mesh.nelements),
        "dofs": int(basis.N),
    }


def run_carriage_calibration(
    carriage_id: str,
    preload: str = "Z1",
    h_bulk_mm: float | None = None,
    h_min_mm: float = 0.40,
    order: int = 2,
) -> dict[str, Any]:
    """Calibrate Hertzian strip on rigid saddle, then extract exact 5-DOF 207 GPa steel saddle compliance."""
    t0 = time.perf_counter()
    spec = CARRIAGE_SPECS[carriage_id]
    if h_bulk_mm is None:
        h_bulk_mm = 2.2 if spec.rail_style == "hg_4row" else 1.75
    k_yy_tgt, k_zz_tgt = _pure_hertz_target_stiffness(spec.rail_profile_id, spec.block_type, preload=preload)

    # Step 1: Calibrate Hertzian strip (e_lat, g_vert) on rigid saddle to hit analytical pure Hertzian k_yy & k_zz
    e_lat, g_vert = 5.0, 2.5
    r_rigid: dict[str, float] = {}
    for _ in range(2):
        r_rigid = _solve_carriage_saddle_once(
            spec,
            e_saddle_pa=E_RIGID,
            e_lat_gpa=e_lat,
            g_vert_gpa=g_vert,
            h_bulk_mm=max(h_bulk_mm, 1.9),
            h_min_mm=h_min_mm,
            order=order,
        )
        e_lat *= k_yy_tgt / r_rigid["k_yy_n_m"]
        g_vert *= k_zz_tgt / r_rigid["k_zz_n_m"]

    # Step 2: Paired high-resolution runs at identical mesh size (rigid saddle vs real 207 GPa steel saddle)
    r_rigid = _solve_carriage_saddle_once(
        spec,
        e_saddle_pa=E_RIGID,
        e_lat_gpa=e_lat,
        g_vert_gpa=g_vert,
        h_bulk_mm=h_bulk_mm,
        h_min_mm=h_min_mm,
        order=order,
    )
    r_steel = _solve_carriage_saddle_once(
        spec,
        e_saddle_pa=E_STEEL,
        e_lat_gpa=e_lat,
        g_vert_gpa=g_vert,
        h_bulk_mm=h_bulk_mm,
        h_min_mm=h_min_mm,
        order=order,
    )

    out: dict[str, Any] = {
        "carriage_id": carriage_id,
        "rail_profile_id": spec.rail_profile_id,
        "block_type": spec.block_type,
        "preload": preload,
        "h_bulk_mm": h_bulk_mm,
        "order": order,
        "tets": r_steel["tets"],
        "dofs": r_steel["dofs"],
        "steel_saddle_mass_g": round(r_steel["saddle_mass_g"], 2),
        "catalog_block_mass_g": spec.catalog_mass_g,
        "strip_e_gpa": round(e_lat, 3),
        "strip_g_gpa": round(g_vert, 3),
    }

    for key, label, scale in (
        ("k_yy_n_m", "yy", 1e-6),  # MN/m = N/um
        ("k_zz_n_m", "zz", 1e-6),
        ("k_rx_nm_rad", "rx", 1.0),  # N*m/rad
        ("k_ry_nm_rad", "ry", 1.0),
        ("k_rz_nm_rad", "rz", 1.0),
    ):
        k_rig = r_rigid[key]
        k_stl = r_steel[key]
        dc = max(1.0 / k_stl - 1.0 / k_rig, 1e-18)
        k_skirt = 1.0 / dc
        eta = k_stl / k_rig
        skirt_share_pct = 100.0 * (1.0 - eta)
        unit_suffix = "mn_m" if scale < 1.0 else "nm_rad"
        out[f"k_rigid_{label}_{unit_suffix}"] = round(k_rig * scale, 2)
        out[f"k_steel_{label}_{unit_suffix}"] = round(k_stl * scale, 2)
        out[f"k_skirt_{label}_{unit_suffix}"] = round(k_skirt * scale, 2)
        out[f"eta_{label}"] = round(eta, 4)
        out[f"skirt_share_{label}_pct"] = round(skirt_share_pct, 1)

    # Effective roll opening ratio vs pure rigid-saddle k_zz * r_y^2 (compare against old flat 1.25)
    r_y_m = 0.5 * spec.rail_w_mm * 1e-3
    out["equiv_roll_opening_factor_z1"] = round(
        (r_steel["k_zz_n_m"] * (r_y_m**2)) / max(r_steel["k_rx_nm_rad"], 1e-9), 3
    )
    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    return out


# ---------------------------------------------------------------------------
# 7. Phase 3: Round Linear Ball Bushings (LM8UU / LM10UU / LM12UU) Validation
# ---------------------------------------------------------------------------


def _solve_bushing_sleeve_once(
    spec: LinearBushingSpec,
    e_sleeve_pa: float,
    e_rad_gpa: float,
    h_bulk_mm: float = 1.8,
    h_min_mm: float = 0.40,
    order: int = 2,
) -> dict[str, float]:
    """Solve 5-DOF axisymmetric steel sleeve compliance for Misumi LMxUU / LMxLUU bushings.

    Uses exact concentric cylindrical shells (zero grazing box-cylinder intersections) with
    a full annular Hertzian ball-contact layer of length L1 (4-row axisymmetric ball support).
    """
    r_shaft = 0.5 * spec.shaft_d_mm
    r_bore = r_shaft + spec.ball_d_mm
    r_out = 0.5 * spec.outer_d_mm
    l_slv = spec.sleeve_len_mm
    l1 = spec.l1_mm

    gmsh.initialize(["gmsh"], interruptible=False)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(f"bush_{spec.bushing_id}")
    occ = gmsh.model.occ

    # 1. Hardened cylindrical steel outer sleeve (exact concentric radii [r_bore, r_out])
    c_out = occ.addCylinder(-0.5 * l_slv, 0.0, 0.0, l_slv, 0.0, 0.0, r_out)
    c_in = occ.addCylinder(-0.5 * l_slv - 0.5, 0.0, 0.0, l_slv + 1.0, 0.0, 0.0, r_bore)
    sleeve_vols, _ = occ.cut([(3, c_out)], [(3, c_in)])

    # 2. Grounded rigid shaft core (exact radius r_shaft)
    core = occ.addCylinder(-0.5 * l_slv - 1.0, 0.0, 0.0, l_slv + 2.0, 0.0, 0.0, r_shaft)

    # 3. Exact concentric annular Hertzian ball-contact sleeve spanning [r_shaft, r_bore] over L1
    s_out = occ.addCylinder(-0.5 * l1, 0.0, 0.0, l1, 0.0, 0.0, r_bore)
    s_in = occ.addCylinder(-0.5 * l1 - 0.5, 0.0, 0.0, l1 + 1.0, 0.0, 0.0, r_shaft)
    strip_vols, _ = occ.cut([(3, s_out)], [(3, s_in)])

    # 4. Rigid outer pillow-block mounting collar wrapping central 75% of sleeve OD
    l_clamp = 0.75 * l_slv
    h_out = occ.addCylinder(-0.5 * l_clamp, 0.0, 0.0, l_clamp, 0.0, 0.0, r_out + 2.5)
    h_in = occ.addCylinder(-0.5 * l_clamp - 0.5, 0.0, 0.0, l_clamp + 1.0, 0.0, 0.0, r_out)
    housing_vols, _ = occ.cut([(3, h_out)], [(3, h_in)])

    _fragment_and_tag(
        [
            (1, sleeve_vols),
            (2, [(3, core)]),
            (3, housing_vols),
            (10, strip_vols),
        ]
    )

    mats = {
        1: iso_constitutive_3d(e_sleeve_pa, NU_STEEL),
        2: iso_constitutive_3d(E_RIGID, 0.30),
        3: iso_constitutive_3d(E_RIGID, 0.30),
        10: radial_bushing_strip_constitutive_3d(e_rad_gpa),
    }
    mesh, basis, k_mat, elem_gid = _mesh_and_assemble(
        mats, h_bulk_mm=h_bulk_mm, h_min_mm=h_min_mm, order=order, curvature_pts=8
    )

    dl = basis.doflocs
    comp = np.arange(basis.N) % 3
    fix_dofs = np.unique(basis.element_dofs[:, elem_gid == 2])

    h_dofs = np.unique(basis.element_dofs[:, elem_gid == 3])
    is_h = np.zeros(basis.N, dtype=bool)
    is_h[h_dofs] = True
    r_h = np.hypot(dl[1], dl[2])
    r_max = float(np.max(r_h[h_dofs]))
    outer_mask = is_h & (r_h >= r_max - 3.0e-4)

    d_fy = np.nonzero(outer_mask & (comp == 1))[0]
    d_fz = np.nonzero(outer_mask & (comp == 2))[0]

    # Radial F_y = 50 N, F_z = 50 N, Pitch M_y = 5 N*m
    f_y = basis.zeros()
    f_y[d_fy] = 50.0 / len(d_fy)
    f_z = basis.zeros()
    f_z[d_fz] = 50.0 / len(d_fz)
    x_z = dl[0, d_fz]
    f_pitch = basis.zeros()
    f_pitch[d_fz] = -(5.0 / np.sum(x_z**2)) * x_z

    u_y, u_z, u_pitch = _solve_factored(k_mat, fix_dofs, [f_y, f_z, f_pitch])
    h_verts = np.unique(mesh.t[:, elem_gid == 3])
    rv = np.hypot(mesh.p[1, h_verts], mesh.p[2, h_verts])
    ov = h_verts[rv >= r_max - 3.0e-4]
    xv = mesh.p[0, ov]

    uy = float(np.mean(u_y[basis.nodal_dofs[1, ov]]))
    uz = float(np.mean(u_z[basis.nodal_dofs[2, ov]]))
    th_p = -float(np.sum(xv * u_pitch[basis.nodal_dofs[2, ov]]) / np.sum(xv**2))

    return {
        "k_yy_n_m": 50.0 / max(abs(uy), 1e-15),
        "k_zz_n_m": 50.0 / max(abs(uz), 1e-15),
        "k_ry_nm_rad": 5.0 / max(abs(th_p), 1e-15),
        "tets": int(mesh.nelements),
        "dofs": int(basis.N),
    }


def run_bushing_calibration(bushing_id: str) -> dict[str, Any]:
    """Validate Misumi LMxUU / LMxLUU hardened cylindrical steel sleeve compliance via Tet10 FEA."""
    t0 = time.perf_counter()
    spec = BUSHING_SPECS[bushing_id]
    k_rad_tgt, _ = _pure_hertz_target_stiffness(spec.rail_profile_id, spec.block_type, preload="Z1")

    e_rad = 1.5
    for _ in range(2):
        r0 = _solve_bushing_sleeve_once(spec, e_sleeve_pa=E_RIGID, e_rad_gpa=e_rad, h_bulk_mm=2.0)
        e_rad *= k_rad_tgt / (0.5 * (r0["k_yy_n_m"] + r0["k_zz_n_m"]))

    r_rig = _solve_bushing_sleeve_once(spec, e_sleeve_pa=E_RIGID, e_rad_gpa=e_rad, h_bulk_mm=1.8)
    r_stl = _solve_bushing_sleeve_once(spec, e_sleeve_pa=E_STEEL, e_rad_gpa=e_rad, h_bulk_mm=1.8)

    k_rad_rig = 0.5 * (r_rig["k_yy_n_m"] + r_rig["k_zz_n_m"])
    k_rad_stl = 0.5 * (r_stl["k_yy_n_m"] + r_stl["k_zz_n_m"])
    dc_rad = max(1.0 / k_rad_stl - 1.0 / k_rad_rig, 1e-18)
    dc_ry = max(1.0 / r_stl["k_ry_nm_rad"] - 1.0 / r_rig["k_ry_nm_rad"], 1e-18)

    return {
        "bushing_id": bushing_id,
        "rail_profile_id": spec.rail_profile_id,
        "block_type": spec.block_type,
        "tets": r_stl["tets"],
        "dofs": r_stl["dofs"],
        "k_rad_rigid_mn_m": round(k_rad_rig * 1e-6, 2),
        "k_rad_steel_mn_m": round(k_rad_stl * 1e-6, 2),
        "k_skirt_rad_mn_m": round((1.0 / dc_rad) * 1e-6, 1),
        "eta_rad": round(k_rad_stl / k_rad_rig, 4),
        "sleeve_share_rad_pct": round(100.0 * (1.0 - k_rad_stl / k_rad_rig), 2),
        "k_ry_rigid_nm_rad": round(r_rig["k_ry_nm_rad"], 1),
        "k_ry_steel_nm_rad": round(r_stl["k_ry_nm_rad"], 1),
        "k_skirt_ry_nm_rad": round(1.0 / dc_ry, 1),
        "eta_ry": round(r_stl["k_ry_nm_rad"] / r_rig["k_ry_nm_rad"], 4),
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }


# ---------------------------------------------------------------------------
# 8. Phase 4: Mesh Convergence & Preload-Invariance Verification
# ---------------------------------------------------------------------------


def run_convergence_checks() -> dict[str, Any]:
    """Verify Tet10 mesh convergence (rail + carriage) and preload invariance of k_skirt."""
    rail_conv = [
        run_solid_rail_calibration("mgn12_rail", h_bulk_mm=2.6, h_min_mm=0.55, order=2),
        run_solid_rail_calibration("mgn12_rail", h_bulk_mm=2.0, h_min_mm=0.45, order=2),
        run_solid_rail_calibration("mgn12_rail", h_bulk_mm=1.5, h_min_mm=0.35, order=2),
    ]
    car_conv = [
        run_carriage_calibration("mgn12h", preload="Z1", h_bulk_mm=2.2, h_min_mm=0.45, order=2),
        run_carriage_calibration("mgn12h", preload="Z1", h_bulk_mm=1.6, h_min_mm=0.35, order=2),
        run_carriage_calibration("mgn12h", preload="Z1", h_bulk_mm=1.25, h_min_mm=0.30, order=2),
    ]
    preload_inv = [
        run_carriage_calibration("mgn12h", preload=pre, h_bulk_mm=1.6, h_min_mm=0.35, order=2)
        for pre in ("Z0", "Z1", "ZA")
    ]
    return {
        "mgn12_rail_mesh_convergence": rail_conv,
        "mgn12h_carriage_mesh_convergence": car_conv,
        "mgn12h_preload_invariance": preload_inv,
    }


# ---------------------------------------------------------------------------
# 9. CLI Driver
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        nargs="+",
        choices=["rails", "carriages", "bushings", "convergence"],
        default=["rails", "carriages", "bushings", "convergence"],
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out-json", type=str, default="")
    args = parser.parse_args()

    t_start = time.perf_counter()
    report: dict[str, Any] = {}

    if "rails" in args.only:
        print("\n=== Phase 1: Solid Profiled Linear Rails — 3D Tet10 (P2) Calibration ===")
        rail_ids = ["mgn9_rail", "mgn12_rail", "mgn15_rail", "hgh15_rail"]
        with ProcessPoolExecutor(max_workers=min(args.workers, len(rail_ids))) as ex:
            rail_res = list(ex.map(run_solid_rail_calibration, rail_ids))
        report["solid_rails"] = rail_res
        for r in rail_res:
            print(
                f"  {r['profile_id']:12s} | DOFs={r['dofs']:6d} | "
                f"A={r['area_brep_mm2']:6.2f} (cat {r['area_catalog_mm2']:6.2f}) mm^2 | "
                f"m={r['mass_brep_g_per_m']:6.1f} (cat {r['mass_catalog_g_per_m']:6.1f}) g/m | "
                f"Iy={r['iy_mm4']:7.2f} ({r['iy_vs_rect_pct']:4.1f}% rect) | "
                f"Iz={r['iz_mm4']:7.2f} ({r['iz_vs_rect_pct']:4.1f}% rect) | "
                f"J={r['j_mm4']:7.2f} mm^4 | "
                f"ky={r['shear_kappa_y']:.3f} kz={r['shear_kappa_z']:.3f} ({r['elapsed_s']}s)"
            )

    if "carriages" in args.only:
        print("\n=== Phase 2: All 7 HIWIN Profiled Linear Carriages — 3D Tet10 (P2) Saddle Calibration ===")
        car_ids = ["mgn9c", "mgn9h", "mgn12c", "mgn12h", "mgn15c", "mgn15h", "hgh15ca"]
        with ProcessPoolExecutor(max_workers=min(args.workers, len(car_ids))) as ex:
            car_res = list(ex.map(run_carriage_calibration, car_ids))
        report["carriages"] = car_res
        for c in car_res:
            print(
                f"  {c['carriage_id']:8s} | m_stl={c['steel_saddle_mass_g']:5.1f}/{c['catalog_block_mass_g']:5.1f}g | "
                f"k_sk_yy={c['k_skirt_yy_mn_m']:6.1f} MN/m ({c['skirt_share_yy_pct']:4.1f}%) | "
                f"k_sk_zz={c['k_skirt_zz_mn_m']:6.1f} MN/m ({c['skirt_share_zz_pct']:4.1f}%) | "
                f"k_sk_rx={c['k_skirt_rx_nm_rad']:7.0f} ({c['skirt_share_rx_pct']:4.1f}%, open={c['equiv_roll_opening_factor_z1']:.3f}) | "
                f"k_sk_ry={c['k_skirt_ry_nm_rad']:7.0f} ({c['skirt_share_ry_pct']:4.1f}%) | "
                f"k_sk_rz={c['k_skirt_rz_nm_rad']:7.0f} ({c['skirt_share_rz_pct']:4.1f}%) ({c['elapsed_s']}s)"
            )

    if "bushings" in args.only:
        print("\n=== Phase 3: Misumi LM8UU/LM10UU/LM12UU Cylindrical Sleeve Validation ===")
        bush_ids = ["lm8uu", "lm8luu", "lm10uu", "lm10luu", "lm12uu", "lm12luu"]
        with ProcessPoolExecutor(max_workers=min(args.workers, len(bush_ids))) as ex:
            bush_res = list(ex.map(run_bushing_calibration, bush_ids))
        report["bushings"] = bush_res
        for b in bush_res:
            print(
                f"  {b['bushing_id']:8s} | DOFs={b['dofs']:6d} | "
                f"k_rad={b['k_rad_steel_mn_m']:5.2f}/{b['k_rad_rigid_mn_m']:5.2f} MN/m | "
                f"k_sk_rad={b['k_skirt_rad_mn_m']:6.1f} MN/m ({b['sleeve_share_rad_pct']:4.2f}%) | "
                f"k_sk_ry={b['k_skirt_ry_nm_rad']:7.1f} Nm/rad (eta={b['eta_ry']:.4f}) ({b['elapsed_s']}s)"
            )

    if "convergence" in args.only:
        print("\n=== Phase 4: Mesh Convergence & Preload Invariance ===")
        conv = run_convergence_checks()
        report["convergence"] = conv
        print("  MGN12 solid rail mesh convergence (coarse -> medium -> fine):")
        for r in conv["mgn12_rail_mesh_convergence"]:
            print(
                f"    h={r['h_bulk_mm']:.2f}mm ({r['dofs']:6d} DOFs): "
                f"Iy={r['iy_mm4']:.2f}, Iz={r['iz_mm4']:.2f}, J={r['j_mm4']:.2f}, "
                f"ky={r['shear_kappa_y']:.3f}, kz={r['shear_kappa_z']:.3f}"
            )
        print("  MGN12H carriage mesh convergence (coarse -> medium -> fine):")
        for c in conv["mgn12h_carriage_mesh_convergence"]:
            print(
                f"    h={c['h_bulk_mm']:.2f}mm ({c['dofs']:6d} DOFs): "
                f"k_sk_yy={c['k_skirt_yy_mn_m']:.1f}, k_sk_zz={c['k_skirt_zz_mn_m']:.1f}, "
                f"k_sk_rx={c['k_skirt_rx_nm_rad']:.0f}, k_sk_ry={c['k_skirt_ry_nm_rad']:.0f}, "
                f"k_sk_rz={c['k_skirt_rz_nm_rad']:.0f}"
            )
        print("  MGN12H saddle k_skirt preload invariance (Z0 / Z1 / ZA):")
        for c in conv["mgn12h_preload_invariance"]:
            print(
                f"    preload={c['preload']}: k_sk_yy={c['k_skirt_yy_mn_m']:.1f}, "
                f"k_sk_zz={c['k_skirt_zz_mn_m']:.1f}, k_sk_rx={c['k_skirt_rx_nm_rad']:.0f}, "
                f"k_sk_ry={c['k_skirt_ry_nm_rad']:.0f}, k_sk_rz={c['k_skirt_rz_nm_rad']:.0f}"
            )

    report["total_elapsed_s"] = round(time.perf_counter() - t_start, 1)
    if args.out_json:
        out_p = Path(args.out_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved full JSON calibration report to {out_p}")
    print(f"\nDone in {report['total_elapsed_s']:.1f}s.")


if __name__ == "__main__":
    main()
