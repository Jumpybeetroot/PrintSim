"""3D Tet10 (P2) continuum FEA calibration & parametric CAD baker for milled/pocketed linear rails.

Constructs exact watertight 3D B-Rep rails in ``build123d`` / OpenCASCADE, meshes with Gmsh,
and solves 3D continuum elasticity via ``scikit-fem`` Tet10 (P2) elements for:

- ``mgn12_hollow_rail``: Monolith/ULS continuous bottom-channel hollow MGN12H (~440 g/m target)
- ``mgn12_pocketed_rail``: Periodic bottom-milled pockets between M3 bolt bridges on MGN12H
- ``mgn9_pocketed_rail``: Periodic bottom-milled pockets between M3 bolt bridges on MGN9H
- ``mgn15_pocketed_rail``: Periodic bottom-milled pockets between M3 bolt bridges on MGN15H
- Solid HIWIN reference rails (``mgn9_rail``, ``mgn12_rail``, ``mgn15_rail``) for apples-to-apples
  differential calibration.

Calibration phases (selectable via ``--only``):
1. ``bare``: Exact effective mass/m, pure 4-pt ``EI_y`` & ``EI_z``, 3-pt vs 4-pt Timoshenko
   ``GA_y`` & ``GA_z``, free-warping & clamped St. Venant ``GJ``, minimum raceway wall thickness,
   and peak pocket-fillet von Mises stress concentration.
2. ``raceway``: Direct Gothic-arch raceway backing compliance (``k_yy``, ``k_zz``, ``k_rx``) both
   **mid-pocket** and **over bolt bridge** vs solid reference rail (replacing guessed scalars).
3. ``composite``: Bolted composite Newmark coupling (``EI_y``, ``GJ``, ``gamma_3d``, ``k_s``)
   when mounted on 2020 Al extrusion, Carbonara 2-wall CF tube, and CNC solid-flange Al X-beam.
4. ``convergence``: Tet4 vs Tet10 coarse/medium/fine mesh convergence verification.
5. ``bake``: Export decimated PBR ``visual.glb`` + ``manifest.json`` under
   ``src/printsim/assets/hardware/x_beams/<profile_id>/``.

Run:
    uv run python scripts/calibrate_pocketed_rails_fea.py --only bare raceway composite convergence
    uv run python scripts/calibrate_pocketed_rails_fea.py --only bake
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

from printsim.fea.components import (
    hertzian_rail_carriage_stiffness,
    newmark_rail_composite_gamma,
)
from printsim.schema import get_material, get_profile

RHO_STEEL = 7850.0  # kg/m^3 (4140 / GCr15 bearing steel)
E_STEEL = 207.0e9  # Pa (hardened bearing steel)
NU_STEEL = 0.29
G_STEEL = E_STEEL / (2.0 * (1.0 + NU_STEEL))
SIGMA_Y_STEEL_MPA = 655.0  # Pa -> MPa (hardened 4140 / GCr15 yield reference)

REPO_ROOT = Path(__file__).resolve().parent.parent
XBEAM_ASSETS_DIR = REPO_ROOT / "src" / "printsim" / "assets" / "hardware" / "x_beams"


# ---------------------------------------------------------------------------
# 1. Parametric MGN Rail Geometry Specifications (HIWIN MG Series + Milled)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RailGeomSpec:
    """Exact HIWIN MG outer envelope, raceway, bolt pattern, and underside milling parameters."""

    profile_id: str
    base_rail_id: str
    display_name: str
    kind: str  # "solid" | "continuous_hollow" | "pocketed"
    width_mm: float  # W_R (along local Y)
    height_mm: float  # H_R (along local Z)
    pitch_mm: float  # Bolt hole pitch P (along local X)
    thru_d_mm: float  # M3 clearance bore d
    cbore_d_mm: float  # Counterbore D
    cbore_depth_mm: float  # Counterbore depth h from top face
    race_z_from_bottom_mm: float  # Ball track centerline height above bottom face
    race_groove_h_mm: float  # Gothic-arch groove opening height
    race_groove_d_mm: float  # Gothic-arch groove depth from outer side face
    ball_d_mm: float  # HIWIN ball diameter d_b
    carriage_l1_mm: float  # H-block active raceway length L1
    # Underside milling parameters (0 for solid)
    cavity_width_mm: float = 0.0  # Channel/pocket width along Y
    cavity_depth_mm: float = 0.0  # Channel/pocket depth upward from bottom face (z = -H/2)
    bridge_len_mm: float = 0.0  # Solid bolt-bridge length along X (pocketed only)
    pocket_corner_r_mm: float = 0.0  # Vertical endmill corner radius in XY
    pocket_roof_r_mm: float = 0.0  # Bull-nose fillet at cavity roof corners
    chamfer_mm: float = 0.30
    bake_length_mm: float = 375.0
    bare_fea_length_mm: float = 200.0
    target_mass_g_per_m: float | None = None

    @property
    def pocket_len_mm(self) -> float:
        if self.kind != "pocketed":
            return 0.0
        return max(self.pitch_mm - self.bridge_len_mm, 0.0)

    @property
    def min_raceway_wall_mm(self) -> float:
        """Minimum steel wall thickness between underside cavity and Gothic-arch raceway root."""
        if self.kind == "solid" or self.cavity_width_mm <= 0.0:
            return 0.5 * self.width_mm - self.race_groove_d_mm
        y_root = 0.5 * self.width_mm - self.race_groove_d_mm
        y_cav = 0.5 * self.cavity_width_mm
        z_race_bot = self.race_z_from_bottom_mm - 0.5 * self.race_groove_h_mm
        if self.cavity_depth_mm >= z_race_bot:
            # Cavity reaches into the raceway height band -> horizontal ligament governs
            return max(y_root - y_cav, 0.0)
        # Diagonal distance from cavity upper corner to raceway root lower corner
        dy = max(y_root - y_cav, 0.0)
        dz = max(z_race_bot - self.cavity_depth_mm, 0.0)
        return math.hypot(dy, dz)

    @property
    def top_web_mm(self) -> float:
        if self.kind == "solid":
            return self.height_mm
        return max(self.height_mm - self.cavity_depth_mm, 0.0)

    @property
    def foot_width_mm(self) -> float:
        if self.kind == "solid":
            return self.width_mm
        return 0.5 * (self.width_mm - self.cavity_width_mm)


RAIL_SPECS: dict[str, RailGeomSpec] = {
    "mgn9_rail": RailGeomSpec(
        profile_id="mgn9_rail",
        base_rail_id="mgn9_rail",
        display_name="MGN9H Solid Linear Rail (9x6.5mm)",
        kind="solid",
        width_mm=9.0,
        height_mm=6.5,
        pitch_mm=20.0,
        thru_d_mm=3.5,
        cbore_d_mm=6.0,
        cbore_depth_mm=3.5,
        race_z_from_bottom_mm=3.9,
        race_groove_h_mm=1.85,
        race_groove_d_mm=0.85,
        ball_d_mm=1.588,
        carriage_l1_mm=28.9,
        chamfer_mm=0.25,
        bake_length_mm=360.0,
        bare_fea_length_mm=200.0,
        target_mass_g_per_m=380.0,
    ),
    "mgn9_pocketed_rail": RailGeomSpec(
        profile_id="mgn9_pocketed_rail",
        base_rail_id="mgn9_rail",
        display_name="Pocketed MGN9H Linear Rail (9x6.5mm, Periodic Underside Pockets)",
        kind="pocketed",
        width_mm=9.0,
        height_mm=6.5,
        pitch_mm=20.0,
        thru_d_mm=3.5,
        cbore_d_mm=6.0,
        cbore_depth_mm=3.5,
        race_z_from_bottom_mm=3.9,
        race_groove_h_mm=1.85,
        race_groove_d_mm=0.85,
        ball_d_mm=1.588,
        carriage_l1_mm=28.9,
        cavity_width_mm=4.6,
        cavity_depth_mm=4.3,
        bridge_len_mm=8.0,
        pocket_corner_r_mm=1.5,
        pocket_roof_r_mm=0.6,
        chamfer_mm=0.25,
        bake_length_mm=360.0,
        bare_fea_length_mm=200.0,
    ),
    "mgn12_rail": RailGeomSpec(
        profile_id="mgn12_rail",
        base_rail_id="mgn12_rail",
        display_name="MGN12H Solid Linear Rail (12x8mm)",
        kind="solid",
        width_mm=12.0,
        height_mm=8.0,
        pitch_mm=25.0,
        thru_d_mm=3.5,
        cbore_d_mm=6.0,
        cbore_depth_mm=4.5,
        race_z_from_bottom_mm=4.8,
        race_groove_h_mm=2.65,
        race_groove_d_mm=1.25,
        ball_d_mm=2.381,
        carriage_l1_mm=32.4,
        chamfer_mm=0.30,
        bake_length_mm=375.0,
        bare_fea_length_mm=200.0,
        target_mass_g_per_m=650.0,
    ),
    "mgn12_hollow_rail": RailGeomSpec(
        profile_id="mgn12_hollow_rail",
        base_rail_id="mgn12_rail",
        display_name="Hollow MGN12H Linear Rail (12x8mm, Monolith/ULS C-Channel)",
        kind="continuous_hollow",
        width_mm=12.0,
        height_mm=8.0,
        pitch_mm=25.0,
        thru_d_mm=3.5,
        cbore_d_mm=6.0,
        # Low-head M3 counterbore (DIN 7984) leaving 1.52 mm steel seating roof over continuous 4.98 mm channel -> ~440 g/m
        cbore_depth_mm=1.50,
        race_z_from_bottom_mm=4.8,
        race_groove_h_mm=2.65,
        race_groove_d_mm=1.25,
        ball_d_mm=2.381,
        carriage_l1_mm=32.4,
        cavity_width_mm=6.6,
        cavity_depth_mm=4.98,
        pocket_roof_r_mm=0.75,
        chamfer_mm=0.30,
        bake_length_mm=375.0,
        bare_fea_length_mm=200.0,
        target_mass_g_per_m=440.0,
    ),
    "mgn12_pocketed_rail": RailGeomSpec(
        profile_id="mgn12_pocketed_rail",
        base_rail_id="mgn12_rail",
        display_name="Pocketed MGN12H Linear Rail (12x8mm, Periodic Underside Pockets)",
        kind="pocketed",
        width_mm=12.0,
        height_mm=8.0,
        pitch_mm=25.0,
        thru_d_mm=3.5,
        cbore_d_mm=6.0,
        cbore_depth_mm=4.5,
        race_z_from_bottom_mm=4.8,
        race_groove_h_mm=2.65,
        race_groove_d_mm=1.25,
        ball_d_mm=2.381,
        carriage_l1_mm=32.4,
        cavity_width_mm=6.6,
        cavity_depth_mm=5.5,
        bridge_len_mm=9.0,
        pocket_corner_r_mm=2.0,
        pocket_roof_r_mm=0.8,
        chamfer_mm=0.30,
        bake_length_mm=375.0,
        bare_fea_length_mm=200.0,
    ),
    "mgn15_rail": RailGeomSpec(
        profile_id="mgn15_rail",
        base_rail_id="mgn15_rail",
        display_name="MGN15H Solid Linear Rail (15x10mm)",
        kind="solid",
        width_mm=15.0,
        height_mm=10.0,
        pitch_mm=40.0,
        thru_d_mm=3.5,
        cbore_d_mm=6.0,
        cbore_depth_mm=4.5,
        race_z_from_bottom_mm=6.0,
        race_groove_h_mm=3.50,
        race_groove_d_mm=1.65,
        ball_d_mm=3.175,
        carriage_l1_mm=43.4,
        chamfer_mm=0.35,
        bake_length_mm=360.0,
        bare_fea_length_mm=240.0,
        target_mass_g_per_m=1060.0,
    ),
    "mgn15_pocketed_rail": RailGeomSpec(
        profile_id="mgn15_pocketed_rail",
        base_rail_id="mgn15_rail",
        display_name="Pocketed MGN15H Linear Rail (15x10mm, Periodic Underside Pockets)",
        kind="pocketed",
        width_mm=15.0,
        height_mm=10.0,
        pitch_mm=40.0,
        thru_d_mm=3.5,
        cbore_d_mm=6.0,
        cbore_depth_mm=4.5,
        race_z_from_bottom_mm=6.0,
        race_groove_h_mm=3.50,
        race_groove_d_mm=1.65,
        ball_d_mm=3.175,
        carriage_l1_mm=43.4,
        cavity_width_mm=8.4,
        cavity_depth_mm=7.0,
        bridge_len_mm=10.0,
        pocket_corner_r_mm=2.5,
        pocket_roof_r_mm=1.0,
        chamfer_mm=0.35,
        bake_length_mm=360.0,
        bare_fea_length_mm=240.0,
    ),
}


# ---------------------------------------------------------------------------
# 2. Parametric build123d B-Rep Builder (Watertight Solid in mm)
# ---------------------------------------------------------------------------


def bolt_positions_mm(length_mm: float, pitch_mm: float, center_on_pocket: bool = True) -> list[float]:
    """Symmetric M3 bolt X-coordinates along [-L/2, +L/2].

    When ``center_on_pocket=True``, bolts sit at ``+-(k + 0.5) * pitch`` so ``x = 0`` is centered
    on a milled pocket (for even pitch count). When ``center_on_pocket=False``, a bolt sits at
    ``x = 0`` (centered on a bolt bridge).
    """
    half = 0.5 * length_mm
    margin = 0.35 * pitch_mm
    xs: list[float] = []
    if center_on_pocket:
        k = 0
        while True:
            x = (k + 0.5) * pitch_mm
            if x > half - margin + 1e-6:
                break
            xs.extend([-x, x])
            k += 1
    else:
        xs.append(0.0)
        k = 1
        while True:
            x = k * pitch_mm
            if x > half - margin + 1e-6:
                break
            xs.extend([-x, x])
            k += 1
    return sorted(xs)


def pocket_centers_mm(length_mm: float, pitch_mm: float, center_on_pocket: bool = True) -> list[float]:
    """Return pocket X-centers strictly between consecutive bolts (and optional half-end pockets)."""
    bolts = bolt_positions_mm(length_mm, pitch_mm, center_on_pocket=center_on_pocket)
    return [0.5 * (bolts[i] + bolts[i + 1]) for i in range(len(bolts) - 1)]


def _make_rounded_rect_face_yz(
    y_center: float, z_center: float, dy: float, dz: float, r: float
) -> bd.Face:
    """Create a Face on the YZ plane (normal +X) for extrusion along X."""
    with bd.BuildSketch(bd.Plane.YZ) as sk:
        with bd.Locations((y_center, z_center)):
            bd.RectangleRounded(dy, dz, r) if r > 1e-4 else bd.Rectangle(dy, dz)
    return sk.sketch.faces()[0]


def build_rail_part_mm(
    spec: RailGeomSpec,
    length_mm: float,
    center_on_pocket: bool = True,
) -> bd.Solid:
    """Build exact watertight 3D B-Rep linear rail solid in mm.

    Coordinate convention (matches PrintSim ``x_beams`` GLBs & ``ProfileSpec``):
    - X in [-L/2, +L/2]: longitudinal rail axis
    - Y in [-W_R/2, +W_R/2]: rail width (side Gothic-arch raceways at y = +-W_R/2)
    - Z in [-H_R/2, +H_R/2]: rail height (bottom mounting face z = -H_R/2, top face z = +H_R/2)
    """
    w = spec.width_mm
    h = spec.height_mm
    ch = spec.chamfer_mm
    z_race = -0.5 * h + spec.race_z_from_bottom_mm
    gh = spec.race_groove_h_mm
    gd = spec.race_groove_d_mm

    # 1. Build 2D rail cross-section in YZ plane (Y horizontal, Z vertical in Plane.YZ)
    # Note: In build123d Plane.YZ: local X = global Y, local Y = global Z, normal = global +X.
    with bd.BuildSketch(bd.Plane.YZ) as sk:
        # Chamfered outer rectangle
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

        # Subtract Gothic-arch side raceway grooves on y = +-W/2 (modeled as curved-root
        # 45-deg trapezoidal/arc profile preserving exact groove depth gd and opening gh)
        r_arc = 0.5 * gh
        for s in (-1.0, 1.0):
            # Stadium/arc groove cut from outer edge inward by gd
            y_root = s * (0.5 * w - gd)
            y_out = s * (0.5 * w + 0.5)
            # Build clean curved raceway cutter: rounded tip at y_root + 45-deg lips to outer face
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

        # For continuous hollow C-channel rail, subtract the underside slot directly in 2D!
        if spec.kind == "continuous_hollow" and spec.cavity_width_mm > 0.0:
            cw = spec.cavity_width_mm
            cd = spec.cavity_depth_mm
            rr = min(spec.pocket_roof_r_mm, 0.45 * cw, 0.45 * cd)
            z_bot = -0.5 * h - 0.5
            z_top = -0.5 * h + cd
            with bd.BuildLine():
                if rr > 1e-3:
                    # Underside open U-channel with filleted internal roof corners
                    q0 = (-0.5 * cw, z_bot)
                    q1 = (0.5 * cw, z_bot)
                    q2 = (0.5 * cw, z_top - rr)
                    q3 = (0.5 * cw - rr, z_top)
                    q4 = (-0.5 * cw + rr, z_top)
                    q5 = (-0.5 * cw, z_top - rr)
                    bd.Line(q0, q1)
                    bd.Line(q1, q2)
                    bd.RadiusArc(q2, q3, rr)
                    bd.Line(q3, q4)
                    bd.RadiusArc(q4, q5, rr)
                    bd.Line(q5, q0)
                else:
                    bd.Polyline(
                        (-0.5 * cw, z_bot),
                        (0.5 * cw, z_bot),
                        (0.5 * cw, z_top),
                        (-0.5 * cw, z_top),
                        close=True,
                    )
            bd.make_face(mode=bd.Mode.SUBTRACT)

    face_sec = sk.sketch.faces()[0]
    rail: bd.Part | bd.Solid = bd.extrude(face_sec, amount=length_mm, both=False)
    rail = rail.translate(bd.Vector(-0.5 * length_mm, 0.0, 0.0))

    # 2. Subtract M3 counterbored bolt holes along X (from top z = +H/2 downward)
    bolts = bolt_positions_mm(length_mm, spec.pitch_mm, center_on_pocket=center_on_pocket)
    cutters: list[bd.Solid] = []
    z_top_face = 0.5 * h
    for xb in bolts:
        # Through clearance hole
        thru_h = h + 1.0
        c_thru = bd.Cylinder(
            radius=0.5 * spec.thru_d_mm,
            height=thru_h,
            align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.CENTER),
        ).translate(bd.Vector(xb, 0.0, 0.0))
        cutters.append(c_thru)
        # Top counterbore
        cb_h = spec.cbore_depth_mm + 0.5
        c_cb = bd.Cylinder(
            radius=0.5 * spec.cbore_d_mm,
            height=cb_h,
            align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MAX),
        ).translate(bd.Vector(xb, 0.0, z_top_face + 0.5))
        cutters.append(c_cb)

    # 3. For periodic pocketed rails, subtract underside pockets between every bolt bridge
    if spec.kind == "pocketed" and spec.pocket_len_mm > 1.0:
        lp = spec.pocket_len_mm
        wp = spec.cavity_width_mm
        dp = spec.cavity_depth_mm
        r_xy = min(spec.pocket_corner_r_mm, 0.45 * wp, 0.45 * lp)
        r_z = min(spec.pocket_roof_r_mm, 0.35 * wp, 0.35 * dp)
        pcenters = pocket_centers_mm(length_mm, spec.pitch_mm, center_on_pocket=center_on_pocket)
        # Also include partial end pockets if rail extends past outer bolt by > bridge/2 + 2 mm
        if bolts:
            first_b, last_b = bolts[0], bolts[-1]
            half_l = 0.5 * length_mm
            if first_b - (-half_l) > 0.5 * spec.bridge_len_mm + 2.0:
                pcenters.insert(0, first_b - 0.5 * spec.pitch_mm)
            if half_l - last_b > 0.5 * spec.bridge_len_mm + 2.0:
                pcenters.append(last_b + 0.5 * spec.pitch_mm)

        # Build one reusable filleted pocket solid at origin (z from -H/2 - 0.5 to -H/2 + dp)
        with bd.BuildSketch(bd.Plane.XY) as sk_p:
            bd.RectangleRounded(lp, wp, r_xy)
        p_face = sk_p.sketch.faces()[0]
        p_solid = bd.extrude(p_face, amount=dp + 0.5)
        # Fillet the pocket roof edges (at top z = dp + 0.5) for bull-nose endmill relief
        if r_z > 0.05:
            top_edges = p_solid.edges().filter_by_position(bd.Axis.Z, dp + 0.45, dp + 0.55)
            if len(top_edges) > 0:
                p_solid = bd.fillet(top_edges, radius=r_z)
        p_solid = p_solid.translate(bd.Vector(0.0, 0.0, -0.5 * h - 0.5))

        for xp in pcenters:
            cutters.append(p_solid.translate(bd.Vector(xp, 0.0, 0.0)))

    if cutters:
        rail = rail - bd.Compound(cutters)

    solids = rail.solids()
    if len(solids) != 1:
        raise RuntimeError(f"Expected 1 connected rail solid for {spec.profile_id}, got {len(solids)}")
    return solids[0]


# ---------------------------------------------------------------------------
# 3. 3D Elasticity Tensors, Gmsh Meshing & skfem Tet10 Assembly
# ---------------------------------------------------------------------------


def iso_constitutive_3d(e_pa: float, nu: float) -> np.ndarray:
    """SPD 6x6 3D isotropic elasticity tensor D (Voigt [xx, yy, zz, xy, yz, xz])."""
    c = e_pa / ((1.0 + nu) * (1.0 - 2.0 * nu))
    d_mat = np.zeros((6, 6), dtype=np.float64)
    d_mat[0:3, 0:3] = c * np.array(
        [[1.0 - nu, nu, nu], [nu, 1.0 - nu, nu], [nu, nu, 1.0 - nu]],
        dtype=np.float64,
    )
    g = e_pa / (2.0 * (1.0 + nu))
    d_mat[3, 3] = d_mat[4, 4] = d_mat[5, 5] = g
    return d_mat


def ortho_constitutive_3d(
    e_l_pa: float,
    e_t_pa: float,
    g_lt_pa: float,
    g_tt_pa: float,
    nu_lt: float = 0.28,
    nu_tt: float = 0.35,
) -> np.ndarray:
    """SPD 6x6 3D orthotropic elasticity tensor D (fibers along X)."""
    s_mat = np.array(
        [
            [1.0 / e_l_pa, -nu_lt / e_l_pa, -nu_lt / e_l_pa, 0.0, 0.0, 0.0],
            [-nu_lt / e_l_pa, 1.0 / e_t_pa, -nu_tt / e_t_pa, 0.0, 0.0, 0.0],
            [-nu_lt / e_l_pa, -nu_tt / e_t_pa, 1.0 / e_t_pa, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0 / g_lt_pa, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0 / g_tt_pa, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / g_lt_pa],
        ],
        dtype=np.float64,
    )
    d_mat = np.linalg.inv(s_mat)
    return 0.5 * (d_mat + d_mat.T)


def mindlin_pad_constitutive_3d(g_shear_gpa: float, e_norm_gpa: float = 80.0) -> np.ndarray:
    """Mindlin interfacial clamping pad: stiff normal contact (along Y), finite X/Z shear stiffness."""
    e_t = max(g_shear_gpa * 2.64, 1.0e-4) * 1e9
    g = max(g_shear_gpa, 1.0e-4) * 1e9
    s_mat = np.diag([1.0 / e_t, 1.0 / (e_norm_gpa * 1e9), 1.0 / e_t, 1.0 / g, 1.0 / g, 1.0 / e_t])
    return np.linalg.inv(s_mat)


def _fragment_and_tag(group_lists: list[tuple[int, list[tuple[int, int]]]]) -> None:
    """Fragment ordered priority volume groups and assign Gmsh 3D Physical Groups."""
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
    """Mesh current Gmsh OCC model and assemble multi-material skfem Tet10/Tet4 stiffness K."""
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


def _compute_peak_von_mises_mpa(
    basis: skfem.CellBasis,
    u: np.ndarray,
    d_mat: np.ndarray,
    x_exclude_m: float,
    x_half_len_m: float,
    exclude_load_x_m: tuple[float, ...] = (),
    load_band_m: float = 0.005,
) -> tuple[float, float]:
    """Evaluate centroidal von Mises stress (MPa) across interior elements (excluding BCs/loads).

    Returns ``(max_vm_mpa, p995_vm_mpa)`` over the periodic interior region.
    """
    u_interp = basis.interpolate(u)
    grad = u_interp.grad  # (3, 3, nelems, nqp)
    exx = grad[0, 0].mean(axis=1)
    eyy = grad[1, 1].mean(axis=1)
    ezz = grad[2, 2].mean(axis=1)
    gxy = (grad[0, 1] + grad[1, 0]).mean(axis=1)
    gyz = (grad[1, 2] + grad[2, 1]).mean(axis=1)
    gxz = (grad[0, 2] + grad[2, 0]).mean(axis=1)
    strain = np.vstack([exx, eyy, ezz, gxy, gyz, gxz])  # (6, nelems)
    stress = d_mat @ strain  # (6, nelems)
    sxx, syy, szz, txy, tyz, txz = stress
    vm = np.sqrt(
        np.maximum(
            0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
            + 3.0 * (txy**2 + tyz**2 + txz**2),
            0.0,
        )
    ) * 1e-6

    # Filter out end BC zones and load application patches
    xc = basis.mesh.p[0, basis.mesh.t].mean(axis=0)
    mask = np.abs(xc) <= (x_half_len_m - x_exclude_m)
    for xl in exclude_load_x_m:
        mask &= np.abs(xc - xl) > load_band_m
    if not np.any(mask):
        return float(np.max(vm)), float(np.percentile(vm, 99.5))
    vals = vm[mask]
    return float(np.max(vals)), float(np.percentile(vals, 99.5))


def _import_solid_to_gmsh(solid: bd.Solid) -> list[tuple[int, int]]:
    """Export build123d Solid to a temp STEP and import into current Gmsh OCC model."""
    with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as tf:
        tmp_path = Path(tf.name)
    try:
        bd.export_step(solid, str(tmp_path))
        ents = gmsh.model.occ.importShapes(str(tmp_path))
        return [e for e in ents if e[0] == 3]
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 4. Phase 1: Bare Rail Tet10 Calibration (Mass, EI_y, EI_z, GJ, GA_y, GA_z, Stress)
# ---------------------------------------------------------------------------


def run_bare_rail_case(
    profile_id: str,
    h_bulk_mm: float = 2.1,
    h_min_mm: float = 0.45,
    order: int = 2,
) -> dict[str, Any]:
    """Run full 3D Tet10 bare-rail calibration: mass, EI_y, EI_z, GA_y, GA_z, GJ, peak stress."""
    t0 = time.perf_counter()
    spec = RAIL_SPECS[profile_id]
    l_mm = spec.bare_fea_length_mm
    l_m = l_mm * 1e-3

    solid = build_rail_part_mm(spec, length_mm=l_mm, center_on_pocket=True)
    vol_mm3 = float(solid.volume)
    area_eff_mm2 = vol_mm3 / l_mm
    mass_g_per_m = area_eff_mm2 * 1e-6 * RHO_STEEL * 1e3

    gmsh.initialize(["gmsh"], interruptible=False)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(f"bare_{profile_id}")
    vols = _import_solid_to_gmsh(solid)
    _fragment_and_tag([(1, vols)])

    d_steel = iso_constitutive_3d(E_STEEL, NU_STEEL)
    mesh, basis, k_mat, _ = _mesh_and_assemble(
        {1: d_steel}, h_bulk_mm=h_bulk_mm, h_min_mm=h_min_mm, order=order, curvature_pts=6
    )

    dl = basis.doflocs
    tol = 1.5e-4
    half_l = 0.5 * l_m
    comp = np.arange(basis.N) % 3

    # Simply supported bending BCs at x = +-L/2 (u_y = u_z = 0 on end faces; fix u_x at midplane center)
    fix_ends_yz = np.nonzero((np.abs(np.abs(dl[0]) - half_l) < tol) & (comp != 0))[0]
    mid_ux_candidates = np.nonzero(
        (np.abs(dl[0]) < 0.004) & (np.abs(dl[1]) < 0.002) & (comp == 0)
    )[0]
    fix_bend = np.unique(np.concatenate([fix_ends_yz, mid_ux_candidates[:2]]))

    # Load application bands on outer side/top surfaces; align 4-pt loads and sagitta to exact
    # integer pitch multiples so periodic pocket curvature ripple cancels identically!
    f_val = 20.0
    pitch_m = spec.pitch_mm * 1e-3
    n_p_half = max(2, int(round(half_l / pitch_m)))
    a_pitches = max(2, n_p_half // 2)
    q_pitches = max(1, a_pitches - 1)
    a_4pt = a_pitches * pitch_m
    x_q = q_pitches * pitch_m
    z_top_m = 0.5 * spec.height_mm * 1e-3
    y_side_m = 0.5 * spec.width_mm * 1e-3

    # 3-pt Z bending (F_z = -20 N at x=0 on top face outer lands)
    d_3z = np.nonzero(
        (np.abs(dl[0]) <= 0.003) & (dl[2] >= z_top_m - 0.0012) & (comp == 2)
    )[0]
    f_3z = basis.zeros()
    f_3z[d_3z] = -f_val / len(d_3z)

    # 4-pt Z bending (F_z = -10 N each at x = +-a_4pt on top face)
    f_4z = basis.zeros()
    for xs in (-a_4pt, a_4pt):
        d4 = np.nonzero((np.abs(dl[0] - xs) <= 0.003) & (dl[2] >= z_top_m - 0.0012) & (comp == 2))[0]
        f_4z[d4] = -(0.5 * f_val) / len(d4)

    # 3-pt Y bending (F_y = -20 N at x=0 on both side faces near bottom/top lands)
    d_3y = np.nonzero(
        (np.abs(dl[0]) <= 0.003) & (np.abs(dl[1]) >= y_side_m - 0.0012) & (comp == 1)
    )[0]
    f_3y = basis.zeros()
    f_3y[d_3y] = -f_val / len(d_3y)

    # 4-pt Y bending (F_y = -10 N each at x = +-a_4pt on side faces)
    f_4y = basis.zeros()
    for xs in (-a_4pt, a_4pt):
        d4 = np.nonzero(
            (np.abs(dl[0] - xs) <= 0.003) & (np.abs(dl[1]) >= y_side_m - 0.0012) & (comp == 1)
        )[0]
        f_4y[d4] = -(0.5 * f_val) / len(d4)

    u_3z, u_4z, u_3y, u_4y = _solve_factored(k_mat, fix_bend, [f_3z, f_4z, f_3y, f_4y])

    # Extract cross-section averaged deflections at x = 0 and x = +-x_q (exact integer pitches apart)
    v_mid = np.nonzero(np.abs(mesh.p[0]) <= 0.003)[0]
    v_q = np.nonzero(np.abs(np.abs(mesh.p[0]) - x_q) <= 0.003)[0]

    # Pure 4-pt bending moment M0 = (F/2) * (L/2 - a_4pt)
    m0_4pt = (0.5 * f_val) * (half_l - a_4pt)
    sag_z = abs(
        float(np.mean(u_4z[basis.nodal_dofs[2, v_mid]]))
        - float(np.mean(u_4z[basis.nodal_dofs[2, v_q]]))
    )
    sag_y = abs(
        float(np.mean(u_4y[basis.nodal_dofs[1, v_mid]]))
        - float(np.mean(u_4y[basis.nodal_dofs[1, v_q]]))
    )
    ei_y_4pt = m0_4pt * (x_q**2) / (2.0 * max(sag_z, 1e-15))
    ei_z_4pt = m0_4pt * (x_q**2) / (2.0 * max(sag_y, 1e-15))

    # 3-pt equivalent EI (for sanity check vs 4-pt)
    dz_3pt = abs(float(np.mean(u_3z[basis.nodal_dofs[2, v_mid]])))
    dy_3pt = abs(float(np.mean(u_3y[basis.nodal_dofs[1, v_mid]])))
    ei_y_3pt = f_val * (l_m**3) / (48.0 * dz_3pt)
    ei_z_3pt = f_val * (l_m**3) / (48.0 * dy_3pt)

    # Dedicated short-span guided-shear stub (L_sh = 1 pitch or 2 pitches, L/H ~ 3-4) where
    # Timoshenko shear is 25-60% of total deflection: delta = F*L^3/(12*EI) + F*L/GA!
    # Built on the same exact periodic rail geometry (clamped x=-L_sh/2, u_x=0 at x=+L_sh/2
    # prevents end rotation -> exact double-cantilever shear!).
    l_sh_mm = 2.0 * spec.pitch_mm
    l_sh_m = l_sh_mm * 1e-3
    sh_solid = build_rail_part_mm(spec, length_mm=l_sh_mm, center_on_pocket=True)
    gmsh.initialize(["gmsh"], interruptible=False)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(f"shear_{profile_id}")
    sh_vols = _import_solid_to_gmsh(sh_solid)
    _fragment_and_tag([(1, sh_vols)])
    mesh_s, basis_s, k_s_mat, _ = _mesh_and_assemble(
        {1: d_steel}, h_bulk_mm=min(h_bulk_mm, 1.8), h_min_mm=h_min_mm, order=order, curvature_pts=6
    )
    dls = basis_s.doflocs
    comps = np.arange(basis_s.N) % 3
    half_sh = 0.5 * l_sh_m
    # Left face x=-L_sh/2 completely clamped (u_x=u_y=u_z=0); right face x=+L_sh/2 has u_x=0 (zero rotation!)
    fix_sh = np.unique(
        np.concatenate(
            [
                np.nonzero(np.abs(dls[0] + half_sh) < tol)[0],
                np.nonzero((np.abs(dls[0] - half_sh) < tol) & (comps == 0))[0],
            ]
        )
    )
    f_sh_r = np.nonzero(np.all(np.abs(mesh_s.p[0, mesh_s.facets] - half_sh) < tol, axis=0))[0]
    fb_sh_r = skfem.FacetBasis(mesh_s, skfem.ElementVector(basis_s.elem.elem), facets=f_sh_r, intorder=2)
    area_sh_r = float(np.sum(fb_sh_r.dx))

    @skfem.LinearForm
    def form_sy(v, w):
        return (100.0 / area_sh_r) * v[1]

    @skfem.LinearForm
    def form_sz(v, w):
        return (100.0 / area_sh_r) * v[2]

    u_sy, u_sz = _solve_factored(k_s_mat, fix_sh, [form_sy.assemble(fb_sh_r), form_sz.assemble(fb_sh_r)])
    vr_sh = np.nonzero(np.abs(mesh_s.p[0] - half_sh) < tol)[0]
    dy_sh_tot = abs(float(np.mean(u_sy[basis_s.nodal_dofs[1, vr_sh]])))
    dz_sh_tot = abs(float(np.mean(u_sz[basis_s.nodal_dofs[2, vr_sh]])))
    # Subtract exact double-cantilever bending F*L^3 / (12 * EI)
    dy_sh_pure = max(dy_sh_tot - 100.0 * (l_sh_m**3) / (12.0 * ei_z_4pt), 1e-15)
    dz_sh_pure = max(dz_sh_tot - 100.0 * (l_sh_m**3) / (12.0 * ei_y_4pt), 1e-15)
    ga_y = 100.0 * l_sh_m / dy_sh_pure
    ga_z = 100.0 * l_sh_m / dz_sh_pure
    area_m2 = area_eff_mm2 * 1e-6
    kappa_y = ga_y / (G_STEEL * area_m2)
    kappa_z = ga_z / (G_STEEL * area_m2)

    # Free-warping St. Venant torsion: fix u_y=u_z=0 at x=-L/2, one single u_x=0 node -> free warping!
    fix_tor_yz = np.nonzero((np.abs(dl[0] + half_l) < tol) & (comp != 0))[0]
    fix_tor_x1 = np.nonzero((np.abs(dl[0] + half_l) < tol) & (comp == 0))[0][:1]
    fix_tor_free = np.unique(np.concatenate([fix_tor_yz, fix_tor_x1]))
    # Clamped-warping torsion at x=-L/2 (models rigidly bolted end / XY-joint restraint)
    fix_tor_clamp = np.nonzero(np.abs(dl[0] + half_l) < tol)[0]

    # Apply pure torque T_x = 2.0 N*m on right end face x = +L/2 (about section centroid y_c=0, z_c)
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

    # Measure twist rate across central periodic span [-L/4, +L/4] (free of end warping boundary layers)
    x_span_tor = 0.5 * l_m
    dtheta_free = _twist_at_x(u_tor_free, +0.25 * l_m) - _twist_at_x(u_tor_free, -0.25 * l_m)
    gj_free = t_x * x_span_tor / max(abs(dtheta_free), 1e-15)

    theta_end_clamp = _twist_at_x(u_tor_clamp, half_l)
    gj_clamp_end = t_x * l_m / max(abs(theta_end_clamp), 1e-15)

    # Peak pocket-fillet von Mises stress under:
    # (a) 4-pt pure bending (normalized per 1.0 N*m bending moment)
    # (b) Torsion (T_x = 2.0 N*m, also normalized per 1.0 N*m)
    vm_4z_max, vm_4z_p995 = _compute_peak_von_mises_mpa(
        basis, u_4z, d_steel, x_exclude_m=0.025, x_half_len_m=half_l, exclude_load_x_m=(-a_4pt, a_4pt)
    )
    vm_4y_max, vm_4y_p995 = _compute_peak_von_mises_mpa(
        basis, u_4y, d_steel, x_exclude_m=0.025, x_half_len_m=half_l, exclude_load_x_m=(-a_4pt, a_4pt)
    )
    vm_tor_max, vm_tor_p995 = _compute_peak_von_mises_mpa(
        basis, u_tor_free, d_steel, x_exclude_m=0.030, x_half_len_m=half_l
    )

    iy_mm4 = (ei_y_4pt / E_STEEL) * 1e12
    iz_mm4 = (ei_z_4pt / E_STEEL) * 1e12
    j_free_mm4 = (gj_free / G_STEEL) * 1e12
    j_clamp_mm4 = (gj_clamp_end / G_STEEL) * 1e12

    return {
        "profile_id": profile_id,
        "kind": spec.kind,
        "length_mm": l_mm,
        "order": order,
        "h_bulk_mm": h_bulk_mm,
        "tets": int(mesh.nelements),
        "dofs": int(basis.N),
        "area_eff_mm2": round(area_eff_mm2, 3),
        "mass_g_per_m": round(mass_g_per_m, 2),
        "min_raceway_wall_mm": round(spec.min_raceway_wall_mm, 3),
        "top_web_mm": round(spec.top_web_mm, 3),
        "foot_width_mm": round(spec.foot_width_mm, 3),
        "ei_y_nm2": round(ei_y_4pt, 3),
        "ei_z_nm2": round(ei_z_4pt, 3),
        "ei_y_3pt_nm2": round(ei_y_3pt, 3),
        "ei_z_3pt_nm2": round(ei_z_3pt, 3),
        "gj_free_nm2": round(gj_free, 3),
        "gj_clamp_nm2": round(gj_clamp_end, 3),
        "iy_mm4": round(iy_mm4, 2),
        "iz_mm4": round(iz_mm4, 2),
        "j_free_mm4": round(j_free_mm4, 2),
        "j_clamp_mm4": round(j_clamp_mm4, 2),
        "ga_z_mn": round(ga_z * 1e-6, 3),
        "ga_y_mn": round(ga_y * 1e-6, 3),
        "kappa_z": round(min(kappa_z, 1.5), 3),
        "kappa_y": round(min(kappa_y, 1.5), 3),
        "vm_bend_z_mpa_per_nm": round(vm_4z_max / m0_4pt, 2),
        "vm_bend_y_mpa_per_nm": round(vm_4y_max / m0_4pt, 2),
        "vm_tor_mpa_per_nm": round(vm_tor_max / t_x, 2),
        "vm_tor_p995_mpa_per_nm": round(vm_tor_p995 / t_x, 2),
        "sf_at_2nm_tor": round(SIGMA_Y_STEEL_MPA / max(vm_tor_max, 1e-6), 2),
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }


# ---------------------------------------------------------------------------
# 5. Phase 2: Direct Gothic-Arch Raceway Backing Compliance (Mid-Pocket vs Over-Bridge)
# ---------------------------------------------------------------------------


def _raceway_strip_constitutive(e_lat_gpa: float, g_vert_gpa: float) -> np.ndarray:
    """Orthotropic Hertzian ball-raceway contact strip constitutive tensor (Y = lateral, Z = vertical)."""
    ey = e_lat_gpa * 1e9
    gz = g_vert_gpa * 1e9
    # Keep X longitudinal soft so carriage doesn't falsely bridge rail axial strains
    ex = 0.05 * ey
    s_mat = np.diag(
        [
            1.0 / ex,
            1.0 / ey,
            1.0 / max(ey * 0.9, 1e6),
            1.0 / max(0.05 * ey, 1e6),
            1.0 / gz,
            1.0 / max(0.05 * gz, 1e6),
        ]
    )
    return np.linalg.inv(s_mat)


def _solve_raceway_seat_model(
    spec: RailGeomSpec,
    center_on_pocket: bool,
    e_lat_gpa: float,
    g_vert_gpa: float,
    rigid_rail: bool = False,
    h_bulk_mm: float = 1.8,
) -> dict[str, float]:
    """Solve 3D Tet10 carriage-on-rail contact for lateral (k_yy), vertical (k_zz), and roll (k_rx).

    Rail is seated on a rigid substrate:
    - Bottom land steel (actual remaining rail feet at z = -H/2) has normal support u_z = 0,
      plus clamped M3 bolt seat footprints (r <= 3.0 mm around each bolt hole on bottom face
      for solid/pocketed rails, or side-foot clamping zones flanking each bolt hole for hollow
      C-channel rails) with u_x = u_y = u_z = 0.
    - Rigid steel carriage wraps the rail with a 0.5 mm air clearance everywhere EXCEPT the
      two Gothic-arch raceway contact strips of length L1 along the ball-track grooves.
    """
    l_seg = max(3.0 * spec.pitch_mm, spec.carriage_l1_mm + 24.0)
    # Round l_seg to exact integer multiple of pitch_mm for clean periodic symmetry
    n_pitches = max(3, int(math.ceil(l_seg / spec.pitch_mm)))
    if center_on_pocket and (n_pitches % 2 == 1):
        n_pitches += 1  # even pitches -> x=0 is mid-pocket
    elif (not center_on_pocket) and (n_pitches % 2 == 0):
        n_pitches += 1  # odd pitches -> x=0 is over bolt bridge
    l_seg = n_pitches * spec.pitch_mm

    rail_solid = build_rail_part_mm(spec, length_mm=l_seg, center_on_pocket=center_on_pocket)
    bolts = bolt_positions_mm(l_seg, spec.pitch_mm, center_on_pocket=center_on_pocket)

    gmsh.initialize(["gmsh"], interruptible=False)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(f"race_{spec.profile_id}_{'mid' if center_on_pocket else 'bridge'}")
    occ = gmsh.model.occ

    rail_vols = _import_solid_to_gmsh(rail_solid)

    w = spec.width_mm
    h = spec.height_mm
    l1 = spec.carriage_l1_mm
    z_race = -0.5 * h + spec.race_z_from_bottom_mm
    gh = spec.race_groove_h_mm
    gd = spec.race_groove_d_mm
    clr = 0.55

    # Two raceway contact strips overlapping slightly into groove & carriage; single-pass fragment
    # trims them cleanly against the Gothic-arch groove and carriage inner walls with zero coplanar artifacts!
    strip_h = 0.80 * gh
    race1 = occ.addBox(-0.5 * l1, -0.5 * w - clr - 0.2, z_race - 0.5 * strip_h, l1, gd + clr + 0.4, strip_h)
    race2 = occ.addBox(-0.5 * l1, 0.5 * w - gd - 0.2, z_race - 0.5 * strip_h, l1, gd + clr + 0.4, strip_h)

    # Rigid carriage block surrounding the rail with clr clearance
    blk_w = w + 10.0
    blk_h = 8.0
    z_blk_bot = z_race - 0.5 * strip_h - 1.2
    blk = occ.addBox(-0.5 * l1, -0.5 * blk_w, z_blk_bot, l1, blk_w, blk_h)
    u_cut = occ.addBox(-0.5 * l1 - 1.0, -0.5 * w - clr, z_blk_bot - 1.0, l1 + 2.0, w + 2.0 * clr, (0.5 * h + clr) - (z_blk_bot - 1.0))
    carr_vols, _ = occ.cut([(3, blk)], [(3, u_cut)])

    # Priority: carriage > rail > raceway strip
    _fragment_and_tag([(6, carr_vols), (1, rail_vols), (10, [(3, race1), (3, race2)])])

    e_r = 20000.0e9 if rigid_rail else E_STEEL
    mats = {
        1: iso_constitutive_3d(e_r, NU_STEEL),
        6: iso_constitutive_3d(20000.0e9, 0.30),  # rigid carriage platen
        10: _raceway_strip_constitutive(e_lat_gpa, g_vert_gpa),
    }
    mesh, basis, k_mat, elem_gid = _mesh_and_assemble(
        mats, h_bulk_mm=h_bulk_mm, h_min_mm=0.35, order=2, curvature_pts=6
    )

    dl = basis.doflocs
    tol = 1.2e-4
    comp = np.arange(basis.N) % 3
    z_bot_m = -0.5 * h * 1e-3

    rail_dofs = np.unique(basis.element_dofs[:, elem_gid == 1])
    is_rail = np.zeros(basis.N, dtype=bool)
    is_rail[rail_dofs] = True

    # Bottom substrate seating BCs on actual rail bottom face z = -H/2:
    # (a) Bilateral seating (compression/lateral): u_z = 0 across all bottom land steel + clamped
    #     M3 bolt footprints (u_x=u_y=u_z=0 within +-4.0 mm of each bolt).
    # (b) Unilateral roll seating (M_x lifts +Y foot, presses -Y foot into rigid substrate):
    #     u_z = 0 on compression half (y <= 0) + clamped M3 bolt footprints only (tension +Y foot
    #     between bolts is free to lift/cantilever over the open channel or pocket!).
    on_bottom = is_rail & (np.abs(dl[2] - z_bot_m) < tol)
    bolt_pad_list = []
    for xb_mm in bolts:
        xb_m = xb_mm * 1e-3
        in_bolt_pad = on_bottom & (np.abs(dl[0] - xb_m) <= 0.0040)
        bolt_pad_list.append(np.nonzero(in_bolt_pad)[0])
    bolt_pad_dofs = np.unique(np.concatenate(bolt_pad_list))
    fix_dofs = np.unique(np.concatenate([np.nonzero(on_bottom & (comp == 2))[0], bolt_pad_dofs]))
    fix_dofs_unilat = np.unique(
        np.concatenate([np.nonzero(on_bottom & (dl[1] <= 1e-5) & (comp == 2))[0], bolt_pad_dofs])
    )

    # Load carriage top face: F_y = 50 N, F_z = 50 N, M_x = 5.0 N*m about raceway centerline (y=0, z=z_race)
    carr_dofs = np.unique(basis.element_dofs[:, elem_gid == 6])
    is_carr = np.zeros(basis.N, dtype=bool)
    is_carr[carr_dofs] = True
    z_carr_top = float(np.max(dl[2, carr_dofs]))
    top_mask = is_carr & (np.abs(dl[2] - z_carr_top) < tol)

    d_fy = np.nonzero(top_mask & (comp == 1))[0]
    d_fz = np.nonzero(top_mask & (comp == 2))[0]

    # Lateral F_y applied at raceway height (subtract induced roll couple M_x = -F_y * dz_arm so pure lateral!)
    dz_arm = z_carr_top - (z_race * 1e-3)
    f_lat = basis.zeros()
    f_lat[d_fy] = 50.0 / len(d_fy)
    # Compensate lever arm so resultant line of action passes exactly through raceway center z = z_race
    y_top = dl[1, d_fz]
    f_lat[d_fz] = (50.0 * dz_arm / np.sum(y_top**2)) * y_top

    # Pure vertical F_z = -50 N (pushing into substrate) & +50 N (lift-off against bolts)
    f_vert = basis.zeros()
    f_vert[d_fz] = 50.0 / len(d_fz)

    # Pure roll torque M_x = 5.0 N*m about X axis (+Y lifted, -Y pressed into substrate)
    f_roll = basis.zeros()
    f_roll[d_fz] = (5.0 / np.sum(y_top**2)) * y_top

    u_lat, u_vert_push = _solve_factored(k_mat, fix_dofs, [f_lat, f_vert])
    (u_roll, u_vert_lift) = _solve_factored(k_mat, fix_dofs_unilat, [f_roll, f_vert])

    carr_verts = np.unique(mesh.t[:, elem_gid == 6])
    top_v = carr_verts[np.abs(mesh.p[2, carr_verts] - z_carr_top) < tol]
    yt = mesh.p[1, top_v]

    # Measure rigid-body carriage motion projected back to raceway center z = z_race
    th_lat = float(np.sum(yt * u_lat[basis.nodal_dofs[2, top_v]]) / np.sum(yt**2))
    uy_top_lat = float(np.mean(u_lat[basis.nodal_dofs[1, top_v]]))
    uy_race_lat = uy_top_lat + th_lat * dz_arm

    # Vertical stiffness: harmonic mean of push-in (compression) and lift-off (bolted tension)
    uz_push = abs(float(np.mean(u_vert_push[basis.nodal_dofs[2, top_v]])))
    uz_lift = abs(float(np.mean(u_vert_lift[basis.nodal_dofs[2, top_v]])))
    uz_vert = 0.5 * (uz_push + uz_lift)
    th_roll = float(np.sum(yt * u_roll[basis.nodal_dofs[2, top_v]]) / np.sum(yt**2))

    k_yy = 50.0 / max(abs(uy_race_lat), 1e-15)
    k_zz = 50.0 / max(abs(uz_vert), 1e-15)
    k_rx = 5.0 / max(abs(th_roll), 1e-15)

    return {
        "k_yy_n_m": k_yy,
        "k_zz_n_m": k_zz,
        "k_rx_nm_rad": k_rx,
        "tets": int(mesh.nelements),
        "dofs": int(basis.N),
    }


def run_raceway_series_calibration(series_base_id: str, milled_ids: list[str]) -> list[dict[str, Any]]:
    """Calibrate Hertzian strip on solid rail to HIWIN Z1, then measure milled rails mid-pocket & over-bridge."""
    t0 = time.perf_counter()
    base_spec = RAIL_SPECS[series_base_id]
    cat = hertzian_rail_carriage_stiffness(series_base_id, preload="Z1", block_type="H")
    k_yy_tgt = cat["k_yy_n_m"]
    k_zz_tgt = cat["k_zz_n_m"]

    # Step 1: 2-step calibration of Hertzian strip (e_lat, g_vert) on RIGID solid rail to hit HIWIN Z1 exactly
    e_lat, g_vert = 6.0, 2.5
    for _ in range(2):
        r0 = _solve_raceway_seat_model(
            base_spec, center_on_pocket=False, e_lat_gpa=e_lat, g_vert_gpa=g_vert, rigid_rail=True, h_bulk_mm=2.1
        )
        e_lat *= k_yy_tgt / r0["k_yy_n_m"]
        g_vert *= k_zz_tgt / r0["k_zz_n_m"]

    # Step 2: Real steel solid rail reference (over-bridge and mid-span)
    ref_bridge = _solve_raceway_seat_model(
        base_spec, center_on_pocket=False, e_lat_gpa=e_lat, g_vert_gpa=g_vert, rigid_rail=False, h_bulk_mm=1.8
    )
    ref_mid = _solve_raceway_seat_model(
        base_spec, center_on_pocket=True, e_lat_gpa=e_lat, g_vert_gpa=g_vert, rigid_rail=False, h_bulk_mm=1.8
    )
    ref_yy = 0.5 * (ref_bridge["k_yy_n_m"] + ref_mid["k_yy_n_m"])
    ref_zz = 0.5 * (ref_bridge["k_zz_n_m"] + ref_mid["k_zz_n_m"])
    ref_rx = 0.5 * (ref_bridge["k_rx_nm_rad"] + ref_mid["k_rx_nm_rad"])

    results: list[dict[str, Any]] = [
        {
            "profile_id": series_base_id,
            "position": "solid_ref",
            "k_yy_n_um": round(ref_yy * 1e-6, 2),
            "k_zz_n_um": round(ref_zz * 1e-6, 2),
            "k_rx_nm_rad": round(ref_rx, 1),
            "eta_yy": 1.0,
            "eta_zz": 1.0,
            "eta_rx": 1.0,
            "strip_e_gpa": round(e_lat, 3),
            "strip_g_gpa": round(g_vert, 3),
        }
    ]

    for mid_id in milled_ids:
        mspec = RAIL_SPECS[mid_id]
        positions = [("mid_pocket", True), ("over_bridge", False)]
        pos_data: dict[str, dict[str, float]] = {}
        for pos_label, is_mid in positions:
            res = _solve_raceway_seat_model(
                mspec, center_on_pocket=is_mid, e_lat_gpa=e_lat, g_vert_gpa=g_vert, rigid_rail=False, h_bulk_mm=1.8
            )
            pos_data[pos_label] = res
            eta_yy = res["k_yy_n_m"] / ref_yy
            eta_zz = res["k_zz_n_m"] / ref_zz
            eta_rx = res["k_rx_nm_rad"] / ref_rx
            # Series differential backing compliance delta_c = 1/k_milled - 1/k_solid (preload-independent!)
            dc_yy = max(1.0 / res["k_yy_n_m"] - 1.0 / ref_yy, 0.0)
            dc_zz = max(1.0 / res["k_zz_n_m"] - 1.0 / ref_zz, 0.0)
            dc_rx = max(1.0 / res["k_rx_nm_rad"] - 1.0 / ref_rx, 0.0)
            results.append(
                {
                    "profile_id": mid_id,
                    "position": pos_label,
                    "k_yy_n_um": round(res["k_yy_n_m"] * 1e-6, 2),
                    "k_zz_n_um": round(res["k_zz_n_m"] * 1e-6, 2),
                    "k_rx_nm_rad": round(res["k_rx_nm_rad"], 1),
                    "eta_yy": round(eta_yy, 4),
                    "eta_zz": round(eta_zz, 4),
                    "eta_rx": round(eta_rx, 4),
                    "k_back_delta_yy_mn_m": round((1.0 / dc_yy) * 1e-6, 1) if dc_yy > 1e-14 else None,
                    "k_back_delta_zz_mn_m": round((1.0 / dc_zz) * 1e-6, 1) if dc_zz > 1e-14 else None,
                    "k_back_delta_rx_nm_rad": round(1.0 / dc_rx, 1) if dc_rx > 1e-14 else None,
                    "tets": res["tets"],
                }
            )
        # Travel-averaged summary (harmonic compliance average 50/50 mid-pocket & over-bridge)
        avg_eta_yy = 0.5 * (pos_data["mid_pocket"]["k_yy_n_m"] + pos_data["over_bridge"]["k_yy_n_m"]) / ref_yy
        avg_eta_zz = 0.5 * (pos_data["mid_pocket"]["k_zz_n_m"] + pos_data["over_bridge"]["k_zz_n_m"]) / ref_zz
        avg_eta_rx = 0.5 * (pos_data["mid_pocket"]["k_rx_nm_rad"] + pos_data["over_bridge"]["k_rx_nm_rad"]) / ref_rx
        results.append(
            {
                "profile_id": mid_id,
                "position": "travel_avg",
                "eta_yy": round(avg_eta_yy, 4),
                "eta_zz": round(avg_eta_zz, 4),
                "eta_rx": round(avg_eta_rx, 4),
                "elapsed_s": round(time.perf_counter() - t0, 1),
            }
        )
    return results


# ---------------------------------------------------------------------------
# 6. Phase 3: Bolted Composite Newmark Coupling (2020 Al, Carbonara CF, CNC Al)
# ---------------------------------------------------------------------------


def _make_rounded_rect_face_occ(cx: float, cy: float, w: float, h: float, r: float) -> int:
    occ = gmsh.model.occ
    hw, hh = 0.5 * w, 0.5 * h
    if r <= 1e-6:
        return occ.addRectangle(cx - hw, cy - hh, 0.0, w, h)
    r1 = occ.addRectangle(cx - hw + r, cy - hh, 0.0, w - 2.0 * r, h)
    r2 = occ.addRectangle(cx - hw, cy - hh + r, 0.0, w, h - 2.0 * r)
    pts = [
        (cx - hw + r, cy - hh + r),
        (cx + hw - r, cy - hh + r),
        (cx + hw - r, cy + hh - r),
        (cx - hw + r, cy + hh - r),
    ]
    objs = [(2, r1), (2, r2)] + [(2, occ.addDisk(px, py, 0.0, r, r)) for px, py in pts]
    fused, _ = occ.fuse([objs[0]], objs[1:])
    return fused[0][1]


def _make_2020_tslot_face_occ() -> int:
    occ = gmsh.model.occ
    outer = _make_rounded_rect_face_occ(0.0, 0.0, 20.0, 20.0, 0.8)
    bore = occ.addDisk(0.0, 0.0, 0.0, 2.5, 2.5)
    cuts = [(2, bore)]
    for s in (1, -1):
        cuts.append((2, occ.addRectangle(8.35 if s > 0 else -10.5, -3.1, 0.0, 2.15, 6.2)))
        cuts.append((2, occ.addRectangle(5.55 if s > 0 else -8.35, -5.4, 0.0, 2.8, 10.8)))
        cuts.append((2, occ.addRectangle(4.1 if s > 0 else -5.55, -3.1, 0.0, 1.45, 6.2)))
        cuts.append((2, occ.addRectangle(-3.1, 8.35 if s > 0 else -10.5, 0.0, 6.2, 2.15)))
        cuts.append((2, occ.addRectangle(-5.4, 5.55 if s > 0 else -8.35, 0.0, 10.8, 2.8)))
        cuts.append((2, occ.addRectangle(-3.1, 4.1 if s > 0 else -5.55, 0.0, 6.2, 1.45)))
    res, _ = occ.cut([(2, outer)], cuts)
    return res[0][1]


def _extrude_occ_x(face_tag: int, x0: float, x1: float) -> list[tuple[int, int]]:
    occ = gmsh.model.occ
    ext = occ.extrude([(2, face_tag)], 0.0, 0.0, x1 - x0)
    vols = [e for e in ext if e[0] == 3]
    occ.rotate(vols, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, math.pi / 2.0)
    occ.translate(vols, x0, 0.0, 0.0)
    return vols


def run_composite_beam_rail_case(
    substrate: str,
    rail_profile_id: str,
    l_mm: float = 250.0,
) -> dict[str, Any]:
    """3D Tet10 bolted composite beam+rail test (2020 Al, Carbonara CF 2-wall, or CNC solid-flange Al).

    Rail is mounted on the front (-Y) face of the 20x20 X-beam with a real 0.25 mm air gap
    between discrete M3 bolt clamping pads, capturing exact partial-interaction Newmark gamma,
    fastener k_s, and composite EI_y / GJ for solid vs continuous-hollow vs pocketed rails.
    """
    t0 = time.perf_counter()
    spec = RAIL_SPECS[rail_profile_id]
    l_m = l_mm * 1e-3
    gap = 0.25
    rail_h = spec.height_mm
    rail_w = spec.width_mm

    # Build exact B-Rep rail oriented on front (-Y) face:
    # Base rail has bottom at z = -H/2, top at z = +H/2, width along Y.
    # Rotate +90 deg around X -> bottom faces +Y (toward beam at y=-10-gap), top faces -Y (front), width along Z!
    rail_solid = build_rail_part_mm(spec, length_mm=l_mm, center_on_pocket=True)
    rail_solid = rail_solid.rotate(bd.Axis.X, 90.0)
    # After +90 deg around X: old +Z -> new -Y, old -Z -> new +Y, old +Y -> new +Z.
    # So old bottom z=-H/2 is now at y=+H/2; translate by -(10 + gap + H/2) along Y so bottom sits at y = -10 - gap!
    rail_solid = rail_solid.translate(bd.Vector(0.0, -(10.0 + gap + 0.5 * rail_h), 0.0))

    gmsh.initialize(["gmsh"], interruptible=False)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(f"comp_{substrate}_{rail_profile_id}")
    occ = gmsh.model.occ

    # 1. X-beam substrate
    if substrate == "al_2020":
        f_b = _make_2020_tslot_face_occ()
        beam_vols = _extrude_occ_x(f_b, -0.5 * l_mm, 0.5 * l_mm)
        d_beam = iso_constitutive_3d(68.9e9, 0.33)
        e_l, g_lt = 68.9e9, 68.9e9 / 2.66
        beam_prof = get_profile("extrusion_2020")
        eff_pad_gpa = 0.50
    elif substrate == "cf_carbonara":
        f_out = _make_rounded_rect_face_occ(0.0, 0.0, 20.0, 20.0, 2.5)
        f_in = _make_rounded_rect_face_occ(0.0, 0.0, 16.4, 16.4, 0.7)
        sec, _ = occ.cut([(2, f_out)], [(2, f_in)])
        beam_vols = _extrude_occ_x(sec[0][1], -0.5 * l_mm, 0.5 * l_mm)
        cf_mat = get_material("carbon_fiber_torsional_45")
        e_l = cf_mat.youngs_modulus_pa
        g_lt = cf_mat.effective_shear_modulus_pa
        d_beam = ortho_constitutive_3d(e_l, 22.0e9, g_lt, 5.5e9)
        beam_prof = get_profile("cf_tube_20x20_carbonara")
        eff_pad_gpa = 0.50
    elif substrate == "cnc_al_solid_flange":
        # Solid-front-flange 20x20x2.2mm Al 6061 hollow box (representative of CNC monocoque front wall)
        f_out = _make_rounded_rect_face_occ(0.0, 0.0, 20.0, 20.0, 1.2)
        f_in = _make_rounded_rect_face_occ(0.0, 0.8, 15.6, 14.0, 0.8)  # thicker 3.0 mm front wall
        sec, _ = occ.cut([(2, f_out)], [(2, f_in)])
        beam_vols = _extrude_occ_x(sec[0][1], -0.5 * l_mm, 0.5 * l_mm)
        d_beam = iso_constitutive_3d(68.9e9, 0.33)
        e_l, g_lt = 68.9e9, 68.9e9 / 2.66
        beam_prof = get_profile("al_tube_20x20x2")  # reference; we measure bare beam directly or via section
        eff_pad_gpa = 0.75  # tapped solid Al front flange -> stiffer Mindlin pad
    else:
        raise ValueError(substrate)

    rail_vols = _import_solid_to_gmsh(rail_solid)

    # 2. Discrete M3 bolts & Mindlin clamping pads (matching actual rail bottom footprint!)
    bolts = bolt_positions_mm(l_mm, spec.pitch_mm, center_on_pocket=True)
    bolt_vols: list[tuple[int, int]] = []
    pad_vols: list[tuple[int, int]] = []
    insert_vols: list[tuple[int, int]] = []

    y_rail_front = -10.0 - gap - rail_h
    y_head_seat = y_rail_front + spec.cbore_depth_mm
    z_foot_out = 0.5 * rail_w - spec.chamfer_mm - 0.15  # stay strictly inside bottom chamfer land

    for xb in bolts:
        # Clean 0.25 mm radial clearance inside Ø6.0 counterbore (zero sliver facets)
        head = occ.addCylinder(xb, y_head_seat - 1.3, 0.0, 0.0, 1.3, 0.0, 2.75)
        if substrate == "al_2020":
            shaft = occ.addCylinder(xb, y_head_seat, 0.0, 0.0, (-7.0) - y_head_seat, 0.0, 1.5)
            tnut = occ.addBox(xb - 3.5, -8.35, -5.3, 7.0, 2.5, 10.6)
            bf, _ = occ.fuse([(3, head)], [(3, shaft), (3, tnut)])
            bolt_vols.extend(bf)
            # Contact pads on 2020 shoulders inside flat rail bottom land
            z_in = max(3.2, 0.5 * spec.cavity_width_mm + 0.15) if spec.kind == "continuous_hollow" else 3.2
            if z_foot_out > z_in + 0.2:
                pad_vols.append((3, occ.addBox(xb - 3.5, -10.0 - gap, -z_foot_out, 7.0, gap, z_foot_out - z_in)))
                pad_vols.append((3, occ.addBox(xb - 3.5, -10.0 - gap, z_in, 7.0, gap, z_foot_out - z_in)))
        elif substrate == "cf_carbonara":
            shaft = occ.addCylinder(xb, y_head_seat, 0.0, 0.0, 10.0 - y_head_seat, 0.0, 1.5)
            nut = occ.addCylinder(xb, 10.0, 0.0, 0.0, 1.8, 0.0, 3.5)
            bf, _ = occ.fuse([(3, head)], [(3, shaft), (3, nut)])
            bolt_vols.extend(bf)
            # PET-CF internal insert block
            h_in = 8.2
            b1 = occ.addBox(xb - 3.5, -h_in, -(h_in - 1.0), 7.0, 2.0 * h_in, 2.0 * (h_in - 1.0))
            b2 = occ.addBox(xb - 3.5, -(h_in - 1.0), -h_in, 7.0, 2.0 * (h_in - 1.0), 2.0 * h_in)
            ins_f, _ = occ.fuse([(3, b1)], [(3, b2)])
            insert_vols.extend(ins_f)
            if spec.kind == "continuous_hollow":
                # Continuous hollow rail only contacts CF face on its two side feet!
                z_in = 0.5 * spec.cavity_width_mm + 0.15
                pad_vols.append((3, occ.addBox(xb - 3.5, -10.0 - gap, -z_foot_out, 7.0, gap, z_foot_out - z_in)))
                pad_vols.append((3, occ.addBox(xb - 3.5, -10.0 - gap, z_in, 7.0, gap, z_foot_out - z_in)))
            else:
                pad_vols.append((3, occ.addCylinder(xb, -10.0 - gap, 0.0, 0.0, gap, 0.0, 3.8)))
        else:
            # Tapped solid front wall (y from -10 to -6.2)
            shaft = occ.addCylinder(xb, y_head_seat, 0.0, 0.0, (-6.2) - y_head_seat, 0.0, 1.5)
            bf, _ = occ.fuse([(3, head)], [(3, shaft)])
            bolt_vols.extend(bf)
            if spec.kind == "continuous_hollow":
                z_in = 0.5 * spec.cavity_width_mm + 0.15
                pad_vols.append((3, occ.addBox(xb - 3.5, -10.0 - gap, -z_foot_out, 7.0, gap, z_foot_out - z_in)))
                pad_vols.append((3, occ.addBox(xb - 3.5, -10.0 - gap, z_in, 7.0, gap, z_foot_out - z_in)))
            else:
                pad_vols.append((3, occ.addCylinder(xb, -10.0 - gap, 0.0, 0.0, gap, 0.0, 4.0)))

    _fragment_and_tag(
        [
            (3, bolt_vols),
            (4, pad_vols),
            (2, rail_vols),
            (5, insert_vols),
            (1, beam_vols),
        ]
    )

    mats = {
        1: d_beam,
        2: iso_constitutive_3d(E_STEEL, NU_STEEL),
        3: iso_constitutive_3d(114.0e9 if substrate == "cf_carbonara" else 205.0e9, 0.30),
        4: mindlin_pad_constitutive_3d(eff_pad_gpa / 2.64, e_norm_gpa=80.0),
        5: iso_constitutive_3d(5.5e9, 0.35),
    }
    mesh, basis, k_mat, elem_gid = _mesh_and_assemble(
        mats, h_bulk_mm=4.5, h_min_mm=0.75, order=2, curvature_pts=0
    )

    dl = basis.doflocs
    tol = 1.5e-4
    half_l = 0.5 * l_m
    comp = np.arange(basis.N) % 3
    beam_dofs = np.unique(basis.element_dofs[:, elem_gid == 1])
    is_beam = np.zeros(basis.N, dtype=bool)
    is_beam[beam_dofs] = True

    # Simply-supported beam ends (rail axial ends free -> pure bolt shear lag!)
    fix_ends = np.nonzero(is_beam & (np.abs(np.abs(dl[0]) - half_l) < tol) & (comp != 0))[0]
    fix_ux = np.nonzero(is_beam & (np.abs(dl[0]) < 0.006) & (np.abs(dl[1]) < 0.002) & (comp == 0))[0][:2]
    fix_bend = np.unique(np.concatenate([fix_ends, fix_ux]))

    # 3-pt Y bending + 4-pt pure Y bending on beam top/bottom webs
    d_3y = np.nonzero(is_beam & (np.abs(dl[0]) <= 0.004) & (np.abs(np.abs(dl[2]) - 0.010) < tol) & (comp == 1))[0]
    f_3y = basis.zeros()
    f_3y[d_3y] = -20.0 / len(d_3y)

    a_4 = 0.0625
    x_q_c = 0.050
    f_4y = basis.zeros()
    for xs in (-a_4, a_4):
        d4 = np.nonzero(is_beam & (np.abs(dl[0] - xs) <= 0.004) & (np.abs(np.abs(dl[2]) - 0.010) < tol) & (comp == 1))[0]
        f_4y[d4] = -10.0 / len(d4)

    u_3y, u_4y = _solve_factored(k_mat, fix_bend, [f_3y, f_4y])

    tube_v = np.unique(mesh.t[:, elem_gid == 1])
    v_mid = tube_v[(np.abs(mesh.p[0, tube_v]) <= 0.005) & (np.abs(np.abs(mesh.p[2, tube_v]) - 0.010) < 0.002)]
    v_q = tube_v[(np.abs(np.abs(mesh.p[0, tube_v]) - x_q_c) <= 0.004) & (np.abs(np.abs(mesh.p[2, tube_v]) - 0.010) < 0.002)]

    dy_3pt = abs(float(np.mean(u_3y[basis.nodal_dofs[1, v_mid]])))
    sag_4 = abs(
        float(np.mean(u_4y[basis.nodal_dofs[1, v_mid]]))
        - float(np.mean(u_4y[basis.nodal_dofs[1, v_q]]))
    )
    m0 = 10.0 * (half_l - a_4)
    ei_y_4pt = m0 * (x_q_c**2) / (2.0 * max(sag_4, 1e-15))

    # Global torsion GJ (clamped at x = -L/2, torque T_x = 10 N*m on right face)
    fix_tor = np.nonzero(np.abs(dl[0] + half_l) < tol)[0]
    f_right = np.nonzero(np.all(np.abs(mesh.p[0, mesh.facets] - half_l) < tol, axis=0))[0]
    fb_r = skfem.FacetBasis(mesh, skfem.ElementVector(basis.elem.elem), facets=f_right, intorder=2)

    @skfem.Functional
    def ip_f(w):
        return w.x[1] ** 2 + w.x[2] ** 2

    ip_e = float(ip_f.assemble(fb_r))

    @skfem.LinearForm
    def t_f(v, w):
        return (10.0 / ip_e) * (-w.x[2] * v[1] + w.x[1] * v[2])

    (u_tor,) = _solve_factored(k_mat, fix_tor, [t_f.assemble(fb_r)])
    rv = np.nonzero(np.abs(mesh.p[0] - half_l) < tol)[0]
    yr, zr = mesh.p[1, rv], mesh.p[2, rv]
    uyr, uzr = u_tor[basis.nodal_dofs[1, rv]], u_tor[basis.nodal_dofs[2, rv]]
    theta_r = float(np.sum(yr * uzr - zr * uyr) / np.sum(yr**2 + zr**2))
    gj_comp = 10.0 * l_m / max(abs(theta_r), 1e-15)

    # Extract Newmark gamma & k_s using actual rail solid volume/area
    r_area_m2 = (float(rail_solid.volume) / l_mm) * 1e-6
    r_prof_cat = get_profile(rail_profile_id) if rail_profile_id in ("mgn12_rail", "mgn12_hollow_rail", "mgn9_rail", "mgn15_rail") else None
    r_iy = r_prof_cat.iy_m4 if r_prof_cat is not None else 4.0e-10
    gamma_out: float | None = None
    ks_out: float | None = None
    if substrate in ("al_2020", "cf_carbonara"):
        i_y_comp = ei_y_4pt / e_l
        n_e = E_STEEL / e_l
        d_c = 0.5 * beam_prof.width_m + gap * 1e-3 + 0.5 * (rail_h * 1e-3)
        i_par_max = ((beam_prof.area_m2 * (n_e * r_area_m2)) / (beam_prof.area_m2 + n_e * r_area_m2)) * (d_c**2)
        gamma = (i_y_comp - beam_prof.iz_m4 - n_e * r_iy) / max(i_par_max, 1e-15)
        ea_star = (e_l * beam_prof.area_m2 * E_STEEL * r_area_m2) / (e_l * beam_prof.area_m2 + E_STEEL * r_area_m2)
        gamma_clamped = max(min(gamma, 0.999), 1e-4)
        k_s = (math.pi**2 * ea_star * (spec.pitch_mm * 1e-3)) / ((l_m**2) * (1.0 / gamma_clamped - 1.0))
        gamma_out = round(gamma, 4)
        ks_out = round(k_s * 1e-6, 2)

    return {
        "substrate": substrate,
        "rail_profile_id": rail_profile_id,
        "tets": int(mesh.nelements),
        "dofs": int(basis.N),
        "dy_3pt_um": round(dy_3pt * 1e6, 3),
        "ei_y_4pt_nm2": round(ei_y_4pt, 1),
        "gj_nm2": round(gj_comp, 2),
        "gamma_3d": gamma_out,
        "k_s_mn_m": ks_out,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }


# ---------------------------------------------------------------------------
# 7. Phase 5: Bake Parametric visual.glb + manifest.json for 4 Milled Rails
# ---------------------------------------------------------------------------


def bake_milled_rail_assets(bare_results: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Bake PBR visual.glb + manifest.json under src/printsim/assets/hardware/x_beams/<id>/."""
    from printsim.cad_assets.baker import VisualPart, build_visual_glb

    milled_ids = [
        "mgn12_hollow_rail",
        "mgn12_pocketed_rail",
        "mgn9_pocketed_rail",
        "mgn15_pocketed_rail",
    ]
    summaries: list[dict[str, Any]] = []
    for pid in milled_ids:
        spec = RAIL_SPECS[pid]
        out_dir = XBEAM_ASSETS_DIR / pid
        out_dir.mkdir(parents=True, exist_ok=True)

        solid = build_rail_part_mm(spec, length_mm=spec.bake_length_mm, center_on_pocket=True)
        vol_cm3 = float(solid.volume) * 1e-3
        mass_g = vol_cm3 * (RHO_STEEL * 1e-3)
        linear_mass_g_m = mass_g / (spec.bake_length_mm * 1e-3)

        vp = VisualPart(
            shape_or_mesh=solid,
            label=pid,
            color_hex="#9ba4ae",  # precision-ground hardened bearing steel
            metallic=0.88,
            roughness=0.28,
        )
        glb_path = out_dir / "visual.glb"
        tris, nbytes = build_visual_glb([vp], glb_path, max_total_triangles=16000)

        prof = get_profile(pid)
        fea_meta: dict[str, Any] = {
            "name": pid,
            "kind": spec.kind,
            "length_mm": spec.bake_length_mm,
            "width_mm": spec.width_mm,
            "height_mm": spec.height_mm,
            "pitch_mm": spec.pitch_mm,
            "cavity_width_mm": spec.cavity_width_mm,
            "cavity_depth_mm": spec.cavity_depth_mm,
            "bridge_len_mm": spec.bridge_len_mm,
            "pocket_len_mm": spec.pocket_len_mm,
            "min_raceway_wall_mm": round(spec.min_raceway_wall_mm, 3),
            "volume_cm3": round(vol_cm3, 3),
            "mass_steel_g": round(mass_g, 2),
            "linear_mass_steel_g_m": round(linear_mass_g_m, 2),
            "area_m2": prof.area_m2,
            "iy_m4": prof.iy_m4,
            "iz_m4": prof.iz_m4,
            "j_m4": prof.j_m4,
            "shear_kappa_y": prof.shear_kappa_y,
            "shear_kappa_z": prof.shear_kappa_z,
        }
        if bare_results and pid in bare_results:
            br = bare_results[pid]
            fea_meta.update(
                {
                    "area_m2": br["area_eff_mm2"] * 1e-6,
                    "iy_m4": br["iy_mm4"] * 1e-12,
                    "iz_m4": br["iz_mm4"] * 1e-12,
                    "j_m4": br["j_free_mm4"] * 1e-12,
                    "j_clamp_m4": br["j_clamp_mm4"] * 1e-12,
                    "kappa_y": br["kappa_y"],
                    "kappa_z": br["kappa_z"],
                    "n_tets": br["tets"],
                    "n_dofs": br["dofs"],
                }
            )

        manifest = {
            "asset_id": pid,
            "category": "x_beams",
            "name": spec.display_name,
            "source": (
                f"Parametric build123d {spec.width_mm:.0f}x{spec.height_mm:.1f}x{spec.bake_length_mm:.0f}mm "
                f"{spec.kind.replace('_', ' ')} linear rail (P={spec.pitch_mm:.0f}mm, Tet10 P2 calibrated)"
            ),
            "visual_glb": "visual.glb",
            "has_superelement": False,
            "length_mm": spec.bake_length_mm,
            "width_mm": spec.width_mm,
            "height_mm": spec.height_mm,
            "visual_triangles": tris,
            "visual_bytes": nbytes,
            "fea_section_properties": fea_meta,
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        summaries.append(
            {
                "asset_id": pid,
                "linear_mass_g_m": round(linear_mass_g_m, 2),
                "triangles": tris,
                "bytes": nbytes,
                "glb": str(glb_path.relative_to(REPO_ROOT)),
            }
        )
    return summaries


# ---------------------------------------------------------------------------
# 8. CLI Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="3D Tet10 FEA calibration & CAD bake for milled/pocketed rails.")
    parser.add_argument(
        "--only",
        nargs="+",
        default=["all"],
        choices=["all", "bare", "raceway", "composite", "convergence", "bake"],
    )
    parser.add_argument("--json-out", type=str, default="", help="Optional path to dump JSON results.")
    args = parser.parse_args()
    sel = set(args.only)
    run_all = "all" in sel

    all_results: dict[str, Any] = {}

    if run_all or "bare" in sel:
        print("\n=== [1] Bare Rail Tet10 (P2) Section, Shear, Torsion & Fillet Stress ===", flush=True)
        order_ids = [
            "mgn9_rail",
            "mgn9_pocketed_rail",
            "mgn12_rail",
            "mgn12_hollow_rail",
            "mgn12_pocketed_rail",
            "mgn15_rail",
            "mgn15_pocketed_rail",
        ]
        bare_map: dict[str, dict[str, Any]] = {}
        with ProcessPoolExecutor(max_workers=4) as ex:
            futs = [ex.submit(run_bare_rail_case, pid) for pid in order_ids]
            for pid, fut in zip(order_ids, futs):
                r = fut.result()
                bare_map[pid] = r
                print(r, flush=True)
        all_results["bare"] = bare_map

    if run_all or "raceway" in sel:
        print("\n=== [2] Direct Gothic-Arch Raceway Backing Compliance (Mid-Pocket vs Over-Bridge) ===", flush=True)
        series_jobs = [
            ("mgn9_rail", ["mgn9_pocketed_rail"]),
            ("mgn12_rail", ["mgn12_hollow_rail", "mgn12_pocketed_rail"]),
            ("mgn15_rail", ["mgn15_pocketed_rail"]),
        ]
        race_all: list[dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=3) as ex:
            futs = [ex.submit(run_raceway_series_calibration, base_id, mids) for base_id, mids in series_jobs]
            for fut in futs:
                rows = fut.result()
                for row in rows:
                    print(row, flush=True)
                race_all.extend(rows)
        all_results["raceway"] = race_all

    if run_all or "composite" in sel:
        print("\n=== [3] Bolted Composite Newmark Coupling (2020 Al, Carbonara CF, CNC Al) ===", flush=True)
        comp_jobs = [
            ("al_2020", "mgn12_rail"),
            ("al_2020", "mgn12_hollow_rail"),
            ("al_2020", "mgn12_pocketed_rail"),
            ("cf_carbonara", "mgn12_rail"),
            ("cf_carbonara", "mgn12_hollow_rail"),
            ("cf_carbonara", "mgn12_pocketed_rail"),
            ("cnc_al_solid_flange", "mgn12_rail"),
            ("cnc_al_solid_flange", "mgn12_hollow_rail"),
            ("cnc_al_solid_flange", "mgn12_pocketed_rail"),
        ]
        comp_rows: list[dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=2) as ex:
            futs = [ex.submit(run_composite_beam_rail_case, sub, rid) for sub, rid in comp_jobs]
            for fut in futs:
                r = fut.result()
                print(r, flush=True)
                comp_rows.append(r)
        all_results["composite"] = comp_rows

    if run_all or "convergence" in sel:
        print("\n=== [4] Mesh & Element Convergence (Tet4 vs Tet10 Coarse/Med/Fine on mgn12_pocketed_rail) ===", flush=True)
        conv_jobs = [
            ("mgn12_pocketed_rail", 2.1, 0.45, 1),
            ("mgn12_pocketed_rail", 2.8, 0.60, 2),
            ("mgn12_pocketed_rail", 2.1, 0.45, 2),
            ("mgn12_pocketed_rail", 1.5, 0.35, 2),
        ]
        conv_rows: list[dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=4) as ex:
            futs = [ex.submit(run_bare_rail_case, pid, hb, hm, o) for pid, hb, hm, o in conv_jobs]
            for fut in futs:
                r = fut.result()
                print(r, flush=True)
                conv_rows.append(r)
        all_results["convergence"] = conv_rows

    if run_all or "bake" in sel:
        print("\n=== [5] Baking Parametric visual.glb + manifest.json ===", flush=True)
        baked = bake_milled_rail_assets(all_results.get("bare"))
        for b in baked:
            print(b, flush=True)
        all_results["baked"] = baked

    if args.json_out:
        out_p = Path(args.json_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(all_results, indent=2) + "\n")
        print(f"\nWrote JSON results to {out_p}", flush=True)


if __name__ == "__main__":
    main()
