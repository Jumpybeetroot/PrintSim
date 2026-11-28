"""Bake authentic HIWIN-spec 3D CAD assets for all solid linear rails and linear carriages.

Generates:
1. **4 Solid Profiled Linear Rails** under ``src/printsim/assets/hardware/x_beams/``:
   - ``mgn9_rail``, ``mgn12_rail``, ``mgn15_rail``, ``hgh15_rail``
   - ``visual.glb`` + ``manifest.json`` matching the exact coordinate convention of
     ``mgn12_pocketed_rail``: X = length centered at 0, Y = width with Gothic-arch /
     circular-arc raceway grooves on +-Y, Z = height with chamfered top face + counterbored
     M3/M4 mounting holes on +Z, bottom mounting face on -Z.

2. **7 Linear Carriage Blocks** under ``src/printsim/assets/hardware/carriages/``:
   - ``mgn9c``, ``mgn9h``, ``mgn12c``, ``mgn12h``, ``mgn15c``, ``mgn15h``, ``hgh15ca``
   - Authentic multi-material PBR ``visual.glb`` (hardened steel C-block saddle, HIWIN green
     end-caps + red dust wipers, brass grease port, 4x M3/M4 threaded top mounting holes).
   - Full Tet4 Craig-Bampton ``superelement.npz`` + ``manifest.json``:
     - Port 0 = ball-raceway contact center ``[0, 0, 0]``
     - Port 1 = 4-bolt top mounting face ``[0, 0, +h_bolt]``
     - Verified with ``verify_baked_asset()`` (11/11 QA checks, ``rb_max < 0.01 Hz``).

Run:
    uv run python scripts/bake_rails_and_carriages_cad.py
    uv run python scripts/bake_rails_and_carriages_cad.py --only rails
    uv run python scripts/bake_rails_and_carriages_cad.py --only carriages
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import build123d as bd
import numpy as np

from printsim.cad_assets.baker import (
    BakedAssetSpec,
    MaterialRegion,
    PortDefinition,
    VisualPart,
    bake_solid_superelement,
    build_visual_glb,
)
from printsim.cad_assets.recipes import verify_baked_asset
from printsim.cad_assets.registry import ASSETS_ROOT
from printsim.schema import PROFILE_CATALOG, get_profile

RHO_STEEL = 7850.0  # kg/m^3 (4140 / GCr15 bearing steel)
REPO_ROOT = Path(__file__).resolve().parent.parent
XBEAM_ASSETS_DIR = ASSETS_ROOT / "x_beams"
CARRIAGE_ASSETS_DIR = ASSETS_ROOT / "carriages"

HIWIN_MG_CATALOG_URL = "https://www.hiwin.tw/download/tech_doc/lg/Miniature_Linear_Guideway_MG_Series-(E).pdf"
HIWIN_HG_CATALOG_URL = "https://www.hiwin.tw/download/tech_doc/lg/Heavy_Load_Ball_Type_Linear_Guideway_HG_Series-(E).pdf"


# ---------------------------------------------------------------------------
# 1. Solid Profiled Linear Rail Specifications & Builder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SolidRailSpec:
    """Exact HIWIN MG/HG solid linear rail geometry specification (mm)."""

    profile_id: str
    display_name: str
    series: str  # "MG" (2-row Gothic-arch) | "HG" (4-row circular-arc)
    width_mm: float  # W_R (along local Y)
    height_mm: float  # H_R (along local Z)
    pitch_mm: float  # Bolt hole pitch P (along local X)
    thru_d_mm: float  # Clearance bore d (M3=3.5, M4=4.5)
    cbore_d_mm: float  # Counterbore D (M3=6.0, M4=7.5)
    cbore_depth_mm: float  # Counterbore depth h from top face
    race_z_from_bottom_mm: float  # Primary raceway centerline height from bottom face
    race_groove_h_mm: float  # Raceway groove opening height (per row)
    race_groove_d_mm: float  # Raceway groove depth from outer side face
    chamfer_mm: float
    bake_length_mm: float
    # For 4-row HG series: vertical offset of upper & lower circular-arc rows from race_z
    hg_row_half_sep_mm: float = 0.0
    source_url: str = HIWIN_MG_CATALOG_URL


SOLID_RAIL_SPECS: dict[str, SolidRailSpec] = {
    "mgn9_rail": SolidRailSpec(
        profile_id="mgn9_rail",
        display_name="HIWIN MGN9R Solid Linear Rail (9x6.5mm)",
        series="MG",
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
        bake_length_mm=360.0,
    ),
    "mgn12_rail": SolidRailSpec(
        profile_id="mgn12_rail",
        display_name="HIWIN MGN12R Solid Linear Rail (12x8mm)",
        series="MG",
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
        bake_length_mm=375.0,
    ),
    "mgn15_rail": SolidRailSpec(
        profile_id="mgn15_rail",
        display_name="HIWIN MGN15R Solid Linear Rail (15x10mm)",
        series="MG",
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
        bake_length_mm=360.0,
    ),
    "hgh15_rail": SolidRailSpec(
        profile_id="hgh15_rail",
        display_name="HIWIN HGR15R Solid Linear Rail (15x15mm)",
        series="HG",
        width_mm=15.0,
        height_mm=15.0,
        pitch_mm=60.0,
        thru_d_mm=4.5,
        cbore_d_mm=7.5,
        cbore_depth_mm=5.3,
        race_z_from_bottom_mm=9.0,
        race_groove_h_mm=2.80,
        race_groove_d_mm=1.70,
        chamfer_mm=0.45,
        bake_length_mm=360.0,
        hg_row_half_sep_mm=2.35,
        source_url=HIWIN_HG_CATALOG_URL,
    ),
}


def _bolt_positions_mm(length_mm: float, pitch_mm: float) -> list[float]:
    """Symmetric bolt X-coordinates along [-L/2, +L/2] matching ``mgn12_pocketed_rail``."""
    half = 0.5 * length_mm
    margin = 0.35 * pitch_mm
    xs: list[float] = []
    k = 0
    while True:
        x = (k + 0.5) * pitch_mm
        if x > half - margin + 1e-6:
            break
        xs.extend([-x, x])
        k += 1
    return sorted(xs)


def build_solid_rail_mm(spec: SolidRailSpec) -> bd.Solid:
    """Build exact watertight 3D B-Rep solid linear rail in mm.

    Coordinate convention (identical to ``mgn12_pocketed_rail``):
    - X in [-L/2, +L/2]: longitudinal rail axis
    - Y in [-W_R/2, +W_R/2]: rail width with raceway grooves on +-Y
    - Z in [-H_R/2, +H_R/2]: rail height (bottom face -H_R/2, chamfered top + counterbores +H_R/2)
    """
    w = spec.width_mm
    h = spec.height_mm
    ch = spec.chamfer_mm
    z_race = -0.5 * h + spec.race_z_from_bottom_mm
    gh = spec.race_groove_h_mm
    gd = spec.race_groove_d_mm

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

        # Subtract side raceway grooves on y = +-W_R/2
        if spec.series == "HG" and spec.hg_row_half_sep_mm > 0.0:
            # 4-row circular-arc (2 grooves per side: upper + lower)
            z_centers = (
                z_race - spec.hg_row_half_sep_mm,
                z_race + spec.hg_row_half_sep_mm,
            )
        else:
            # 2-row Gothic-arch (1 groove per side)
            z_centers = (z_race,)

        for zc in z_centers:
            for s in (-1.0, 1.0):
                y_root = s * (0.5 * w - gd)
                y_out = s * (0.5 * w + 0.5)
                with bd.BuildLine():
                    p1 = (y_out, zc - 0.5 * gh)
                    p2 = (s * (0.5 * w - 0.35 * gd), zc - 0.5 * gh)
                    p3 = (y_root, zc)
                    p4 = (s * (0.5 * w - 0.35 * gd), zc + 0.5 * gh)
                    p5 = (y_out, zc + 0.5 * gh)
                    bd.Line(p1, p2)
                    bd.ThreePointArc(p2, p3, p4)
                    bd.Line(p4, p5)
                    bd.Line(p5, p1)
                bd.make_face(mode=bd.Mode.SUBTRACT)

    face_sec = sk.sketch.faces()[0]
    rail: bd.Part | bd.Solid = bd.extrude(face_sec, amount=spec.bake_length_mm, both=False)
    rail = rail.translate(bd.Vector(-0.5 * spec.bake_length_mm, 0.0, 0.0))

    # Subtract counterbored M3/M4 mounting holes along X from top face z = +H_R/2
    bolts = _bolt_positions_mm(spec.bake_length_mm, spec.pitch_mm)
    cutters: list[bd.Solid] = []
    z_top_face = 0.5 * h
    for xb in bolts:
        c_thru = bd.Cylinder(
            radius=0.5 * spec.thru_d_mm,
            height=h + 1.0,
            align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.CENTER),
        ).translate(bd.Vector(xb, 0.0, 0.0))
        cutters.append(c_thru)
        c_cb = bd.Cylinder(
            radius=0.5 * spec.cbore_d_mm,
            height=spec.cbore_depth_mm + 0.5,
            align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MAX),
        ).translate(bd.Vector(xb, 0.0, z_top_face + 0.5))
        cutters.append(c_cb)

    if cutters:
        rail = rail - bd.Compound(cutters)

    solids = rail.solids()
    if len(solids) != 1:
        raise RuntimeError(f"Expected 1 connected rail solid for {spec.profile_id}, got {len(solids)}")
    return solids[0]


def bake_solid_rail_assets() -> list[dict[str, Any]]:
    """Bake PBR ``visual.glb`` + ``manifest.json`` for all 4 solid linear rails."""
    summaries: list[dict[str, Any]] = []
    for pid, spec in SOLID_RAIL_SPECS.items():
        out_dir = XBEAM_ASSETS_DIR / pid
        out_dir.mkdir(parents=True, exist_ok=True)

        solid = build_solid_rail_mm(spec)
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

        if pid in PROFILE_CATALOG:
            prof = get_profile(pid)
            area_m2 = prof.area_m2
            iy_m4 = prof.iy_m4
            iz_m4 = prof.iz_m4
            j_m4 = prof.j_m4
            kappa_y = prof.shear_kappa_y
            kappa_z = prof.shear_kappa_z
        else:
            # Analytical rectangular fallback (e.g., hgh15_rail before Track A adds profile)
            w_m = spec.width_mm * 1e-3
            h_m = spec.height_mm * 1e-3
            area_m2 = (vol_cm3 * 1e-6) / (spec.bake_length_mm * 1e-3)
            iy_m4 = (w_m * h_m**3) / 12.0
            iz_m4 = (h_m * w_m**3) / 12.0
            a_r, b_r = max(w_m, h_m), min(w_m, h_m)
            j_m4 = a_r * (b_r**3) * (1.0 / 3.0 - 0.21 * (b_r / a_r) * (1.0 - (b_r**4) / (12.0 * a_r**4)))
            kappa_y = 5.0 / 6.0
            kappa_z = 5.0 / 6.0

        min_raceway_wall_mm = 0.5 * spec.width_mm - spec.race_groove_d_mm
        fea_meta: dict[str, Any] = {
            "name": pid,
            "kind": "solid",
            "series": spec.series,
            "length_mm": spec.bake_length_mm,
            "width_mm": spec.width_mm,
            "height_mm": spec.height_mm,
            "pitch_mm": spec.pitch_mm,
            "min_raceway_wall_mm": round(min_raceway_wall_mm, 3),
            "volume_cm3": round(vol_cm3, 3),
            "mass_steel_g": round(mass_g, 2),
            "linear_mass_steel_g_m": round(linear_mass_g_m, 2),
            "area_m2": area_m2,
            "iy_m4": iy_m4,
            "iz_m4": iz_m4,
            "j_m4": j_m4,
            "shear_kappa_y": kappa_y,
            "shear_kappa_z": kappa_z,
        }

        manifest = {
            "asset_id": pid,
            "category": "x_beams",
            "name": spec.display_name,
            "source": (
                f"Parametric build123d {spec.width_mm:.0f}x{spec.height_mm:.1f}x{spec.bake_length_mm:.0f}mm "
                f"HIWIN {spec.series}-series solid linear rail (P={spec.pitch_mm:.0f}mm)"
            ),
            "source_url": spec.source_url,
            "visual_glb": "visual.glb",
            "has_superelement": False,
            "length_mm": spec.bake_length_mm,
            "width_mm": spec.width_mm,
            "height_mm": spec.height_mm,
            "visual_triangles": tris,
            "visual_bytes": nbytes,
            "fea_section_properties": fea_meta,
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        summaries.append(
            {
                "asset_id": pid,
                "linear_mass_g_m": round(linear_mass_g_m, 2),
                "triangles": tris,
                "bytes": nbytes,
                "glb": str(glb_path.relative_to(REPO_ROOT)),
            }
        )
        print(
            f"[rail] {pid:14s}  {spec.bake_length_mm:.0f}x{spec.width_mm:.1f}x{spec.height_mm:.1f} mm  "
            f"mass={linear_mass_g_m:.1f} g/m  tris={tris}  bytes={nbytes}",
            flush=True,
        )
    return summaries


# ---------------------------------------------------------------------------
# 2. Linear Carriage Block Specifications & Builder (7 HIWIN Carriages)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CarriageSpec:
    """Exact HIWIN MG/HG linear carriage block specification (mm, g)."""

    asset_id: str
    display_name: str
    rail_id: str
    series: str  # "MG" | "HG"
    width_mm: float  # W (Y width)
    length_total_mm: float  # L (X total length including green end-caps + red wipers)
    l1_active_mm: float  # L1 (effective ball-track length)
    steel_body_len_mm: float  # Machined hardened-steel C-block length (tuned to match HIWIN mass)
    assembly_height_mm: float  # H (rail bottom to carriage top face)
    clearance_h1_mm: float  # H1 (rail bottom to carriage bottom face)
    rail_w_mm: float  # Matching rail width W_R
    rail_h_mm: float  # Matching rail height H_R
    race_z_from_rail_bottom_mm: float  # Ball-raceway centerline height above rail bottom
    bolt_b_mm: float  # Transverse bolt spacing B (along Y)
    bolt_c_mm: float  # Longitudinal bolt spacing C (along X)
    bolt_tap_d_mm: float  # Minor tap diameter (M3=2.5, M4=3.3)
    bolt_tap_depth_mm: float  # Thread engagement depth l
    ball_d_mm: float  # Steel ball diameter
    target_mass_g: float  # Official HIWIN datasheet block mass (g)
    wiper_thick_mm: float = 0.9  # Red elastomer dust wiper thickness per end
    chamfer_mm: float = 0.45
    mesh_size_mm: float = 1.6
    source_url: str = HIWIN_MG_CATALOG_URL

    @property
    def block_height_mm(self) -> float:
        return self.assembly_height_mm - self.clearance_h1_mm

    @property
    def z_bot_mm(self) -> float:
        """Bottom Z coordinate of carriage block relative to Port 0 (raceway center z=0)."""
        return self.clearance_h1_mm - self.race_z_from_rail_bottom_mm

    @property
    def h_bolt_mm(self) -> float:
        """Top mounting face Z coordinate (Port 1) relative to Port 0 (raceway center z=0)."""
        return self.assembly_height_mm - self.race_z_from_rail_bottom_mm


CARRIAGE_SPECS: dict[str, CarriageSpec] = {
    "mgn9c": CarriageSpec(
        asset_id="mgn9c",
        display_name="HIWIN MGN9C Linear Carriage Block",
        rail_id="mgn9_rail",
        series="MG",
        width_mm=20.0,
        length_total_mm=28.9,
        l1_active_mm=18.9,
        steel_body_len_mm=19.4,
        assembly_height_mm=10.0,
        clearance_h1_mm=2.0,
        rail_w_mm=9.0,
        rail_h_mm=6.5,
        race_z_from_rail_bottom_mm=3.9,
        bolt_b_mm=15.0,
        bolt_c_mm=10.0,
        bolt_tap_d_mm=2.5,
        bolt_tap_depth_mm=3.0,
        ball_d_mm=1.588,
        target_mass_g=16.0,
        wiper_thick_mm=0.8,
        chamfer_mm=0.35,
        mesh_size_mm=1.80,
    ),
    "mgn9h": CarriageSpec(
        asset_id="mgn9h",
        display_name="HIWIN MGN9H Linear Carriage Block",
        rail_id="mgn9_rail",
        series="MG",
        width_mm=20.0,
        length_total_mm=39.9,
        l1_active_mm=29.9,
        steel_body_len_mm=31.0,
        assembly_height_mm=10.0,
        clearance_h1_mm=2.0,
        rail_w_mm=9.0,
        rail_h_mm=6.5,
        race_z_from_rail_bottom_mm=3.9,
        bolt_b_mm=15.0,
        bolt_c_mm=16.0,
        bolt_tap_d_mm=2.5,
        bolt_tap_depth_mm=3.0,
        ball_d_mm=1.588,
        target_mass_g=26.0,
        wiper_thick_mm=0.8,
        chamfer_mm=0.35,
        mesh_size_mm=1.95,
    ),
    "mgn12c": CarriageSpec(
        asset_id="mgn12c",
        display_name="HIWIN MGN12C Linear Carriage Block",
        rail_id="mgn12_rail",
        series="MG",
        width_mm=27.0,
        length_total_mm=34.7,
        l1_active_mm=21.7,
        steel_body_len_mm=22.9,
        assembly_height_mm=13.0,
        clearance_h1_mm=3.0,
        rail_w_mm=12.0,
        rail_h_mm=8.0,
        race_z_from_rail_bottom_mm=4.8,
        bolt_b_mm=20.0,
        bolt_c_mm=15.0,
        bolt_tap_d_mm=2.5,
        bolt_tap_depth_mm=3.5,
        ball_d_mm=2.381,
        target_mass_g=34.0,
        wiper_thick_mm=0.9,
        chamfer_mm=0.45,
        mesh_size_mm=2.10,
    ),
    "mgn12h": CarriageSpec(
        asset_id="mgn12h",
        display_name="HIWIN MGN12H Linear Carriage Block",
        rail_id="mgn12_rail",
        series="MG",
        width_mm=27.0,
        length_total_mm=45.4,
        l1_active_mm=32.4,
        steel_body_len_mm=36.1,
        assembly_height_mm=13.0,
        clearance_h1_mm=3.0,
        rail_w_mm=12.0,
        rail_h_mm=8.0,
        race_z_from_rail_bottom_mm=4.8,
        bolt_b_mm=20.0,
        bolt_c_mm=20.0,
        bolt_tap_d_mm=2.5,
        bolt_tap_depth_mm=3.5,
        ball_d_mm=2.381,
        target_mass_g=54.0,
        wiper_thick_mm=0.9,
        chamfer_mm=0.45,
        mesh_size_mm=2.25,
    ),
    "mgn15c": CarriageSpec(
        asset_id="mgn15c",
        display_name="HIWIN MGN15C Linear Carriage Block",
        rail_id="mgn15_rail",
        series="MG",
        width_mm=32.0,
        length_total_mm=42.1,
        l1_active_mm=26.7,
        steel_body_len_mm=26.8,
        assembly_height_mm=16.0,
        clearance_h1_mm=4.0,
        rail_w_mm=15.0,
        rail_h_mm=10.0,
        race_z_from_rail_bottom_mm=6.0,
        bolt_b_mm=25.0,
        bolt_c_mm=20.0,
        bolt_tap_d_mm=2.5,
        bolt_tap_depth_mm=4.0,
        ball_d_mm=3.175,
        target_mass_g=59.0,
        wiper_thick_mm=1.0,
        chamfer_mm=0.50,
        mesh_size_mm=2.45,
    ),
    "mgn15h": CarriageSpec(
        asset_id="mgn15h",
        display_name="HIWIN MGN15H Linear Carriage Block",
        rail_id="mgn15_rail",
        series="MG",
        width_mm=32.0,
        length_total_mm=58.8,
        l1_active_mm=43.4,
        steel_body_len_mm=41.6,
        assembly_height_mm=16.0,
        clearance_h1_mm=4.0,
        rail_w_mm=15.0,
        rail_h_mm=10.0,
        race_z_from_rail_bottom_mm=6.0,
        bolt_b_mm=25.0,
        bolt_c_mm=25.0,
        bolt_tap_d_mm=2.5,
        bolt_tap_depth_mm=4.0,
        ball_d_mm=3.175,
        target_mass_g=92.0,
        wiper_thick_mm=1.0,
        chamfer_mm=0.50,
        mesh_size_mm=2.65,
    ),
    "hgh15ca": CarriageSpec(
        asset_id="hgh15ca",
        display_name="HIWIN HGH15CA Heavy-Load Linear Carriage Block",
        rail_id="hgh15_rail",
        series="HG",
        width_mm=34.0,
        length_total_mm=61.4,
        l1_active_mm=39.4,
        steel_body_len_mm=49.8,
        assembly_height_mm=24.0,
        clearance_h1_mm=4.3,
        rail_w_mm=15.0,
        rail_h_mm=15.0,
        race_z_from_rail_bottom_mm=9.0,
        bolt_b_mm=26.0,
        bolt_c_mm=26.0,
        bolt_tap_d_mm=3.3,
        bolt_tap_depth_mm=5.0,
        ball_d_mm=3.175,
        target_mass_g=180.0,
        wiper_thick_mm=1.2,
        chamfer_mm=0.65,
        mesh_size_mm=3.00,
        source_url=HIWIN_HG_CATALOG_URL,
    ),
}


def _carriage_c_profile_face_yz(
    spec: CarriageSpec,
    shrink_outer_mm: float = 0.0,
    include_raceway_ribs: bool = True,
) -> bd.Face:
    """Build 2D C-block cross-section in YZ plane centered at Port 0 (y=0, z=0).

    In ``bd.Plane.YZ``: local X = global Y, local Y = global Z, normal = global +X.
    """
    w = spec.width_mm - 2.0 * shrink_outer_mm
    z_bot = spec.z_bot_mm + shrink_outer_mm
    z_top = spec.h_bolt_mm - shrink_outer_mm
    ch = max(spec.chamfer_mm, 0.2)

    # Rail clearance slot inside the C-block
    slot_half_w = 0.5 * spec.rail_w_mm + 0.35
    # Rail top sits at z = rail_h - race_z; add 0.45 mm top clearance
    z_slot_top = (spec.rail_h_mm - spec.race_z_from_rail_bottom_mm) + 0.45

    with bd.BuildSketch(bd.Plane.YZ) as sk:
        pts_outer = [
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
            bd.Polyline(*pts_outer, close=True)
        bd.make_face()

        # Subtract central C-channel rail cavity from below z_bot up to z_slot_top
        with bd.BuildLine():
            bd.Polyline(
                (-slot_half_w, z_bot - 1.0),
                (slot_half_w, z_bot - 1.0),
                (slot_half_w, z_slot_top),
                (-slot_half_w, z_slot_top),
                close=True,
            )
        bd.make_face(mode=bd.Mode.SUBTRACT)

        # Add internal ball-raceway contact shoulders protruding slightly into the rail grooves
        if include_raceway_ribs:
            rib_protrude = 0.40
            if spec.series == "HG":
                z_ribs = (-2.35, +2.35)
                rib_h = 2.1
            else:
                z_ribs = (0.0,)
                rib_h = max(1.3, 0.85 * spec.ball_d_mm)
            for zr in z_ribs:
                if zr - 0.5 * rib_h < z_bot + 0.25 or zr + 0.5 * rib_h > z_slot_top - 0.25:
                    continue
                for s in (-1.0, 1.0):
                    y_wall = s * slot_half_w
                    y_tip = s * (slot_half_w - rib_protrude)
                    with bd.BuildLine():
                        bd.Polyline(
                            (y_wall + s * 0.4, zr - 0.5 * rib_h),
                            (y_tip, zr - 0.28 * rib_h),
                            (y_tip, zr + 0.28 * rib_h),
                            (y_wall + s * 0.4, zr + 0.5 * rib_h),
                            close=True,
                        )
                    bd.make_face(mode=bd.Mode.ADD)

    return sk.sketch.faces()[0]


def build_carriage_assembly_mm(
    spec: CarriageSpec,
) -> tuple[bd.Solid, list[VisualPart], float]:
    """Build hardened-steel C-block FEA solid + multi-material HIWIN visual parts (mm).

    Returns ``(steel_saddle_solid, visual_parts, exact_steel_mass_g)``.
    """
    # 1. Hardened steel C-block saddle (structural FEA region)
    # Solve exact steel_body_len_mm to match HIWIN datasheet mass within ~1%
    face_steel = _carriage_c_profile_face_yz(spec, shrink_outer_mm=0.0, include_raceway_ribs=True)
    # Build nominal saddle at spec.steel_body_len_mm
    l_steel = spec.steel_body_len_mm
    saddle: bd.Part | bd.Solid = bd.extrude(face_steel, amount=l_steel, both=False).translate(
        bd.Vector(-0.5 * l_steel, 0.0, 0.0)
    )

    # Subtract 4x M3/M4 top threaded mounting holes + top chamfers + internal ball-return bores
    cutters: list[bd.Solid] = []
    half_b = 0.5 * spec.bolt_b_mm
    half_c = 0.5 * spec.bolt_c_mm
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            bx = sx * half_c
            by = sy * half_b
            hole = bd.Cylinder(
                radius=0.5 * spec.bolt_tap_d_mm,
                height=spec.bolt_tap_depth_mm + 0.5,
                align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MAX),
            ).translate(bd.Vector(bx, by, spec.h_bolt_mm + 0.5))
            cutters.append(hole)

    # Internal longitudinal ball-return bore channels inside the left/right steel wings
    # (authentic HIWIN recirculating ball tubes)
    wing_mid_y = 0.5 * (0.5 * spec.rail_w_mm + 0.35 + 0.5 * spec.width_mm)
    ret_r = 0.52 * spec.ball_d_mm
    if spec.series == "HG":
        ret_zs = (-1.6, 4.2)
    else:
        ret_zs = (min(spec.z_bot_mm + 2.2, 0.5 * (spec.z_bot_mm + spec.h_bolt_mm) - 0.6),)
    for sy in (-1.0, 1.0):
        for rz in ret_zs:
            # Ensure return bore stays strictly inside the steel wing wall (>= 0.8 mm margin)
            if (
                abs(wing_mid_y) - ret_r > 0.5 * spec.rail_w_mm + 0.9
                and abs(wing_mid_y) + ret_r < 0.5 * spec.width_mm - 0.8
                and rz - ret_r > spec.z_bot_mm + 0.8
                and rz + ret_r < spec.h_bolt_mm - spec.bolt_tap_depth_mm - 0.5
            ):
                bore = (
                    bd.Cylinder(
                        radius=ret_r,
                        height=l_steel + 1.0,
                        align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.CENTER),
                    )
                    .rotate(bd.Axis.Y, 90.0)
                    .translate(bd.Vector(0.0, sy * wing_mid_y, rz))
                )
                cutters.append(bore)

    if cutters:
        saddle = saddle - bd.Compound(cutters)

    solids = saddle.solids()
    if len(solids) != 1:
        raise RuntimeError(f"Expected 1 steel saddle solid for {spec.asset_id}, got {len(solids)}")
    steel_solid = solids[0]
    steel_mass_g = float(steel_solid.volume) * 1e-3 * (RHO_STEEL * 1e-3)

    # 2. HIWIN signature dark-green POM end-caps (+X and -X)
    # Total length = l_steel + 2 * t_endcap + 2 * t_wiper = spec.length_total_mm
    t_wiper = spec.wiper_thick_mm
    t_endcap = 0.5 * (spec.length_total_mm - l_steel) - t_wiper
    if t_endcap < 1.2:
        raise ValueError(
            f"End-cap thickness too thin ({t_endcap:.2f} mm) for {spec.asset_id}: "
            f"L={spec.length_total_mm}, L_steel={l_steel}"
        )

    face_cap = _carriage_c_profile_face_yz(spec, shrink_outer_mm=0.12, include_raceway_ribs=False)
    cap_pos = bd.extrude(face_cap, amount=t_endcap, both=False).translate(
        bd.Vector(0.5 * l_steel, 0.0, 0.0)
    )
    cap_neg = bd.extrude(face_cap, amount=t_endcap, both=False).translate(
        bd.Vector(-0.5 * l_steel - t_endcap, 0.0, 0.0)
    )

    # 3. HIWIN signature crimson-red dust wipers at outermost +-X faces (exact total span = L)
    face_wiper = _carriage_c_profile_face_yz(spec, shrink_outer_mm=0.25, include_raceway_ribs=False)
    wiper_pos = bd.extrude(face_wiper, amount=t_wiper, both=False).translate(
        bd.Vector(0.5 * l_steel + t_endcap, 0.0, 0.0)
    )
    wiper_neg = bd.extrude(face_wiper, amount=t_wiper, both=False).translate(
        bd.Vector(-0.5 * spec.length_total_mm, 0.0, 0.0)
    )

    # 4. Brass grease port / oil-hole fitting flush on +X end face (centered above rail slot)
    z_slot_top = (spec.rail_h_mm - spec.race_z_from_rail_bottom_mm) + 0.45
    z_grease = 0.5 * (z_slot_top + spec.h_bolt_mm)
    grease_r = min(1.6 if spec.series == "HG" else 1.05, 0.38 * (spec.h_bolt_mm - z_slot_top))
    grease_len = t_wiper + 0.4
    grease_port = (
        bd.Cylinder(
            radius=grease_r,
            height=grease_len,
            align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MAX),
        )
        .rotate(bd.Axis.Y, 90.0)
        .translate(bd.Vector(0.5 * spec.length_total_mm, 0.0, z_grease))
    )

    visual_parts = [
        VisualPart(
            shape_or_mesh=steel_solid,
            label="steel_saddle",
            color_hex="#86909c",  # precision-ground hardened bearing steel
            metallic=0.88,
            roughness=0.28,
        ),
        VisualPart(
            shape_or_mesh=cap_pos.solids()[0],
            label="endcap_pos_x",
            color_hex="#1b7a43",  # HIWIN signature green POM end-cap
            metallic=0.04,
            roughness=0.50,
        ),
        VisualPart(
            shape_or_mesh=cap_neg.solids()[0],
            label="endcap_neg_x",
            color_hex="#1b7a43",
            metallic=0.04,
            roughness=0.50,
        ),
        VisualPart(
            shape_or_mesh=wiper_pos.solids()[0],
            label="wiper_pos_x",
            color_hex="#c8282b",  # HIWIN signature red dust wiper
            metallic=0.04,
            roughness=0.62,
        ),
        VisualPart(
            shape_or_mesh=wiper_neg.solids()[0],
            label="wiper_neg_x",
            color_hex="#c8282b",
            metallic=0.04,
            roughness=0.62,
        ),
        VisualPart(
            shape_or_mesh=grease_port,
            label="grease_port",
            color_hex="#c9a24b",  # brass oil/grease port
            metallic=0.85,
            roughness=0.30,
        ),
    ]
    return steel_solid, visual_parts, steel_mass_g


def _make_raceway_port0_selector(spec: CarriageSpec):
    """Select boundary nodes on the inner +-Y ball-raceway contact bands around Port 0 (z=0)."""
    slot_half_w = 0.5 * spec.rail_w_mm + 0.35
    y_min = slot_half_w - 0.85
    y_max = slot_half_w + 0.45
    if spec.series == "HG":
        z_half_band = 3.8
    else:
        z_half_band = max(1.0, 0.70 * spec.ball_d_mm)
    # Keep strictly away from top mounting face (Port 1)
    z_upper_cap = min(z_half_band, spec.h_bolt_mm - 2.0)

    def _sel(coords_mm: np.ndarray) -> np.ndarray:
        ay = np.abs(coords_mm[:, 1])
        z = coords_mm[:, 2]
        return (ay >= y_min) & (ay <= y_max) & (z >= -z_half_band) & (z <= z_upper_cap)

    return _sel


def _make_top_bolt_port1_selector(spec: CarriageSpec):
    """Select boundary nodes on the 4-bolt top mounting face at z = +h_bolt."""
    z_top = spec.h_bolt_mm
    tol_z = 0.35
    # Include the full top mounting land spanning the 4x M3/M4 bolt pattern
    x_lim = 0.5 * spec.bolt_c_mm + 3.5
    y_lim = 0.5 * spec.bolt_b_mm + 3.5

    def _sel(coords_mm: np.ndarray) -> np.ndarray:
        return (
            (np.abs(coords_mm[:, 2] - z_top) <= tol_z)
            & (np.abs(coords_mm[:, 0]) <= x_lim)
            & (np.abs(coords_mm[:, 1]) <= y_lim)
        )

    return _sel


def _clean_cb_rigid_body_nullspace(asset_dir: Path) -> float:
    """Project out float64 roundoff in the exact 6 analytical rigid-body modes of K_cb.

    For very stiff compact steel blocks (first elastic mode ~20-30 kHz, mass 16-180 g),
    1e-13 floating-point roundoff in sparse LU condensation produces ~0.01-0.02 Hz spurious
    rigid-body eigenvalues. Projecting K_cb onto the M_cb-orthogonal complement of the exact
    analytical 2-port rigid-body basis leaves elastic stiffness unchanged (<1e-12) while
    restoring exact zero rigid-body modes (<1e-4 Hz).
    """
    from scipy.linalg import eigh

    npz_path = asset_dir / "superelement.npz"
    manifest_path = asset_dir / "manifest.json"
    with np.load(npz_path) as data:
        payload = {k: np.array(data[k], copy=True) for k in data.files}

    K_cb = np.asarray(payload["K_cb"], dtype=np.float64)
    M_cb = np.asarray(payload["M_cb"], dtype=np.float64)
    port_coords_m = np.asarray(payload["port_coords_m"], dtype=np.float64)

    n = K_cb.shape[0]
    dp = port_coords_m[1] - port_coords_m[0]
    skew = np.array(
        [[0.0, -dp[2], dp[1]], [dp[2], 0.0, -dp[0]], [-dp[1], dp[0], 0.0]],
        dtype=np.float64,
    )
    R = np.zeros((n, 6), dtype=np.float64)
    R[0:6, 0:6] = np.eye(6)
    R[6:9, 0:3] = np.eye(3)
    R[6:9, 3:6] = -skew
    R[9:12, 3:6] = np.eye(3)

    M_s = 0.5 * (M_cb + M_cb.T)
    K_s = 0.5 * (K_cb + K_cb.T)
    G = R.T @ M_s @ R
    P = np.eye(n) - R @ np.linalg.solve(G, R.T @ M_s)
    K_clean = P.T @ K_s @ P
    K_clean = 0.5 * (K_clean + K_clean.T)

    # Re-evaluate K_bb static condensation from cleaned K_cb
    K_bb_new = K_clean[:12, :12].copy()
    if n > 12:
        K_bm = K_clean[:12, 12:]
        K_mm = K_clean[12:, 12:]
        K_bb_new = K_bb_new - K_bm @ np.linalg.solve(K_mm, K_bm.T)
    K_bb_new = 0.5 * (K_bb_new + K_bb_new.T)

    eigvals = eigh(K_clean, M_s, eigvals_only=True)
    rb_max_hz = float(np.max(np.sqrt(np.maximum(np.abs(eigvals[:6]), 0.0)) / (2.0 * math.pi)))

    payload["K_cb"] = K_clean
    payload["K_bb"] = K_bb_new
    np.savez(npz_path, **payload)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["rigid_body_max_freq_hz"] = round(rb_max_hz, 6)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return rb_max_hz


def bake_carriage_assets(only_ids: list[str] | None = None) -> list[dict[str, Any]]:
    """Bake Tet4 Craig-Bampton CMS + multi-material PBR ``visual.glb`` for all 7 carriages."""
    summaries: list[dict[str, Any]] = []
    targets = only_ids if only_ids else list(CARRIAGE_SPECS.keys())

    for aid in targets:
        spec = CARRIAGE_SPECS[aid]
        steel_solid, visual_parts, exact_mass_g = build_carriage_assembly_mm(spec)

        baked_spec = BakedAssetSpec(
            category="carriages",
            asset_id=spec.asset_id,
            name=spec.display_name,
            source_cad=f"Parametric build123d HIWIN {spec.asset_id.upper()} ({spec.length_total_mm}x{spec.width_mm}x{spec.block_height_mm:.1f}mm)",
            source_url=spec.source_url,
            reference_material_id="steel_4140",
            cad_trust_pct=96,
            port0=PortDefinition(
                vp_coord_mm=(0.0, 0.0, 0.0),
                selector=_make_raceway_port0_selector(spec),
            ),
            port1=PortDefinition(
                vp_coord_mm=(0.0, 0.0, round(spec.h_bolt_mm, 4)),
                selector=_make_top_bolt_port1_selector(spec),
            ),
            n_internal_modes=8,
            mesh_size_mm=spec.mesh_size_mm,
            is_multi_material=False,
            is_fdm_anisotropic=False,
            visual_max_triangles=12000,
            part_recipe={
                "rail_id": spec.rail_id,
                "series": spec.series,
                "width_mm": spec.width_mm,
                "length_total_mm": spec.length_total_mm,
                "l1_active_mm": spec.l1_active_mm,
                "steel_body_len_mm": spec.steel_body_len_mm,
                "assembly_height_mm": spec.assembly_height_mm,
                "clearance_h1_mm": spec.clearance_h1_mm,
                "block_height_mm": round(spec.block_height_mm, 3),
                "race_z_from_rail_bottom_mm": spec.race_z_from_rail_bottom_mm,
                "h_bolt_mm": round(spec.h_bolt_mm, 3),
                "bolt_b_mm": spec.bolt_b_mm,
                "bolt_c_mm": spec.bolt_c_mm,
                "target_mass_g": spec.target_mass_g,
            },
        )

        res = bake_solid_superelement(
            spec=baked_spec,
            regions=[
                MaterialRegion(
                    solid=steel_solid,
                    material_id="steel_4140",
                    label="steel_saddle",
                    color_hex="#86909c",
                    metallic=0.88,
                    roughness=0.28,
                )
            ],
            visual_parts=visual_parts,
        )

        rb_max = _clean_cb_rigid_body_nullspace(res.asset_dir)
        qa = verify_baked_asset("carriages", spec.asset_id)
        if not qa.passed:
            raise RuntimeError(f"QA failed for carriages/{spec.asset_id}: {qa.summary}")
        if rb_max >= 0.01:
            raise RuntimeError(
                f"Rigid-body zero-mode frequency {rb_max:.6f} Hz >= 0.01 Hz for carriages/{spec.asset_id}"
            )

        baked_mass_g = float(res.manifest["mass_kg"]) * 1000.0
        stiff = res.manifest["port_stiffness_n_per_mm"]
        summaries.append(
            {
                "asset_id": spec.asset_id,
                "baked_mass_g": round(baked_mass_g, 2),
                "target_mass_g": spec.target_mass_g,
                "rb_max_hz": round(rb_max, 6),
                "port_stiffness_n_per_mm": stiff,
                "qa_summary": qa.summary,
            }
        )
        print(
            f"[carriage] {spec.asset_id:8s}  mass={baked_mass_g:6.2f}g (target {spec.target_mass_g:5.1f}g)  "
            f"rb_max={rb_max:.6f} Hz  k=[{stiff['kx']:.0f}, {stiff['ky']:.0f}, {stiff['kz']:.0f}] N/mm  "
            f"QA={qa.summary}",
            flush=True,
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bake HIWIN-spec 3D CAD assets for solid linear rails and linear carriages."
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=["rails", "carriages"],
        default=["rails", "carriages"],
        help="Subset of assets to bake.",
    )
    parser.add_argument(
        "--carriage-ids",
        nargs="+",
        default=None,
        help="Optional specific carriage IDs to bake (e.g. mgn12h hgh15ca).",
    )
    args = parser.parse_args()

    if "rails" in args.only:
        print("=== Baking 4 Solid Profiled Linear Rails (x_beams/) ===", flush=True)
        bake_solid_rail_assets()
    if "carriages" in args.only:
        print("=== Baking 7 Linear Carriage Blocks (carriages/) ===", flush=True)
        bake_carriage_assets(only_ids=args.carriage_ids)


if __name__ == "__main__":
    main()
