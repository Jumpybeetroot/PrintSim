#!/usr/bin/env python3
"""Bake missing catalog hotends (4) and Voron StealthBurner + Tap R8 CAD/FEA superelements.

Assets baked into ``src/printsim/assets/hardware/``:
  1. ``hotends/e3d_v6_groove``       — E3D V6 Groove-Mount (40W Al Block, unbraced 0.30 mm SS304 neck)
  2. ``hotends/phaetus_dragon_uhf``  — Phaetus Dragon UHF (4x Ti standoffs + extended 30 mm Cu UHF block/nut)
  3. ``hotends/phaetus_rapido_2_uhf``— Phaetus Rapido 2 UHF (85W ceramic Cu core + 3x Ti screws + UHF extender)
  4. ``hotends/monolith_50mm_hf``    — Monolith 50mm+V6 HF (4-bolt M3, 4x vertical-preload Ti posts, 50 mm Cu core)
  5. ``toolheads/stealthburner_tap`` — Voron StealthBurner + CW2.5 + Voron Tap R8 (50 mm vertical MGN9H rail)

Usage::

    uv run python scripts/bake_missing_hotends_and_tap.py
    uv run python scripts/bake_missing_hotends_and_tap.py --only e3d_v6_groove,stealthburner_tap
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time
from typing import Any

import build123d as b3d
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bake_hotends_cal12 import (  # noqa: E402
    COL_RAPIDO_SOCK,
    HOTEND_STEP_DIR,
    STEALTHBURNER_STEP,
    _walk_world_solids,
)
from bake_phase1_assets import (  # noqa: E402
    COLOR_BRASS_NOZZLE,
    COLOR_CNC_AL,
    COLOR_COPPER,
    COLOR_DARK_SOCK,
    COLOR_DARK_TI,
    COLOR_NICKEL_CU,
    COLOR_STAINLESS,
    _make_v6_nozzle_solid,
    _non_overlapping_regions,
)
from bake_voron_kits import (  # noqa: E402
    COL_MOTOR_BODY,
    COL_PCB_DARK,
    COL_STAINLESS,
    COL_STEEL,
    COL_VORON_BLACK,
    COL_VORON_RED,
    _combine_meshes,
    _selector_box_or_nearest,
    _shape_to_trimesh,
    _transform_mesh,
    _walk_named_solids,
)
from printsim.cad_assets.baker import (  # noqa: E402
    ASSETS_ROOT,
    BakedAssetResult,
    BakedAssetSpec,
    MaterialRegion,
    PortDefinition,
    VisualPart,
    bake_solid_superelement,
    select_plane_box,
    select_z_annulus,
    select_z_cap,
)
from printsim.cad_assets.recipes import verify_baked_asset  # noqa: E402

COL_V6_BLUE_SOCK = "#1d4ed8"
COL_MAGNET_SILVER = "#cbd5e1"


def _make_hex_prism(across_flats_mm: float, height_mm: float, origin_z_mm: float) -> b3d.Solid:
    """Create a vertical hexagonal prism centered on XY=(0,0) spanning [origin_z_mm, origin_z_mm + height_mm]."""
    r_circum = across_flats_mm / math.sqrt(3.0)
    pts = [
        (r_circum * math.cos(math.radians(60.0 * k)), r_circum * math.sin(math.radians(60.0 * k)))
        for k in range(6)
    ]
    wire = b3d.Wire.make_polygon([b3d.Vector(x, y, origin_z_mm) for x, y in pts])
    face = b3d.Face(wire)
    return b3d.Solid.extrude(face, b3d.Vector(0.0, 0.0, height_mm))


# ---------------------------------------------------------------------------
# 1. E3D V6 Groove-Mount (40W Al Block, Unbraced 0.30 mm SS304 Throat)
# ---------------------------------------------------------------------------


def bake_e3d_v6_groove(assets_root: Path = ASSETS_ROOT) -> BakedAssetResult:
    """Bake ``hotends/e3d_v6_groove`` watertight multi-material Tet4 CMS + PBR GLB."""
    print("\n=== Baking hotends/e3d_v6_groove ===")

    # Exact E3D V6 official drawing dimensions (OAL = 62.30 mm, Port 0 = groove collar Z=0, Port 1 = tip Z=-62.30):
    # - Al 6061-T6 V6 heatsink: Z = [0.0, -42.7]
    #   - Top Ø16.0 x 3.7 mm flange @ Z = [-3.7, 0.0]
    #   - Ø12.0 x 6.0 mm J-head groove waist @ Z = [-9.7, -3.7]
    #   - Lower Ø16.0 x 3.0 mm collar @ Z = [-12.7, -9.7]
    #   - Central core Ø10.0 mm + 11 circular Ø22.3 x 1.0 mm fins (2.5 mm pitch) @ Z = [-42.7, -12.7]
    # - Unbraced SS304 heatbreak: Z = [-51.0, -36.0]
    #   - Upper M7 shank Ø6.5 mm inside heatsink (Z = [-42.7, -36.0])
    #   - Exposed thin-wall SS304 throat Ø2.64 OD / Ø1.96 ID (I-matched to Ø2.60 / Ø2.00 = 0.30 mm wall,
    #     spanning bare thermal break gap Z = [-47.2, -42.7] = 4.5 mm!)
    #   - Lower M6 shank Ø5.6 mm inside heater block (Z = [-51.0, -47.2])
    # - Al 6061-T6 V6 heater block: 23.0 x 16.0 x 11.5 mm @ X=[-15.0, 8.0], Y=[-8.0, 8.0], Z=[-58.7, -47.2]
    # - Brass C360 V6 nozzle: Z = [-62.30, -49.80] (tip at exact -62.30 mm)
    hs_top = b3d.Solid.make_cylinder(8.0, 3.7, plane=b3d.Plane(origin=(0.0, 0.0, -3.7)))
    hs_groove = b3d.Solid.make_cylinder(6.0, 6.0, plane=b3d.Plane(origin=(0.0, 0.0, -9.7)))
    hs_collar = b3d.Solid.make_cylinder(8.0, 3.0, plane=b3d.Plane(origin=(0.0, 0.0, -12.7)))
    hs_core = b3d.Solid.make_cylinder(5.0, 30.0, plane=b3d.Plane(origin=(0.0, 0.0, -42.7)))
    hs_fins = [
        b3d.Solid.make_cylinder(11.15, 1.0, plane=b3d.Plane(origin=(0.0, 0.0, -15.2 - 2.75 * k)))
        for k in range(11)
    ]
    hs_bore_upper = b3d.Solid.make_cylinder(2.1, 36.5, plane=b3d.Plane(origin=(0.0, 0.0, -36.2)))
    hs_bore_lower = b3d.Solid.make_cylinder(3.25, 7.5, plane=b3d.Plane(origin=(0.0, 0.0, -43.2)))
    fea_heatsink = (
        hs_top.fuse(hs_groove, hs_collar, hs_core, *hs_fins)
        .cut(hs_bore_upper, hs_bore_lower)
        .clean()
    )

    hb_upper = b3d.Solid.make_cylinder(3.25, 6.7, plane=b3d.Plane(origin=(0.0, 0.0, -42.7)))
    hb_neck = b3d.Solid.make_cylinder(1.32, 6.5, plane=b3d.Plane(origin=(0.0, 0.0, -48.2)))
    hb_lower = b3d.Solid.make_cylinder(2.8, 3.8, plane=b3d.Plane(origin=(0.0, 0.0, -51.0)))
    hb_bore = b3d.Solid.make_cylinder(0.98, 16.5, plane=b3d.Plane(origin=(0.0, 0.0, -51.8)))
    fea_heatbreak = hb_upper.fuse(hb_neck, hb_lower).cut(hb_bore).clean()

    blk_raw = b3d.Solid.make_box(23.0, 16.0, 11.5, plane=b3d.Plane(origin=(-15.0, -8.0, -58.7)))
    heater_bore = b3d.Solid.make_cylinder(
        3.05, 18.0, plane=b3d.Plane(origin=(-10.5, -9.0, -52.95), z_dir=(0.0, 1.0, 0.0))
    )
    therm_bore = b3d.Solid.make_cylinder(
        1.55, 18.0, plane=b3d.Plane(origin=(-5.5, -9.0, -52.95), z_dir=(0.0, 1.0, 0.0))
    )
    melt_bore = b3d.Solid.make_cylinder(0.98, 13.0, plane=b3d.Plane(origin=(0.0, 0.0, -59.5)))
    fea_block = blk_raw.cut(heater_bore, therm_bore, melt_bore).clean()

    fea_nozzle = _make_v6_nozzle_solid(tip_z_mm=-62.30, total_len_mm=12.5)

    regions = _non_overlapping_regions(
        [
            (fea_heatbreak, "stainless_304", "v6_ss304_heatbreak", COLOR_STAINLESS, 0.88, 0.25),
            (fea_nozzle, "brass_c360", "v6_brass_nozzle", COLOR_BRASS_NOZZLE, 0.90, 0.24),
            (fea_block, "aluminum_6061_t6", "v6_al_heater_block", COLOR_CNC_AL, 0.88, 0.26),
            (fea_heatsink, "aluminum_6061_t6", "v6_groove_heatsink", COLOR_CNC_AL, 0.88, 0.24),
        ]
    )

    # Visual-only classic E3D blue silicone sock shell around the heater block
    sock_outer = b3d.Solid.make_box(25.4, 18.4, 12.6, plane=b3d.Plane(origin=(-16.2, -9.2, -59.4)))
    sock_inner = b3d.Solid.make_box(23.1, 16.1, 11.7, plane=b3d.Plane(origin=(-15.05, -8.05, -58.8)))
    sock_noz_hole = b3d.Solid.make_cylinder(4.6, 5.0, plane=b3d.Plane(origin=(0.0, 0.0, -61.0)))
    sock_vis = sock_outer.cut(sock_inner, sock_noz_hole).clean()

    visual_parts = [
        VisualPart(fea_heatsink, "v6_groove_heatsink", COLOR_CNC_AL, metallic=0.88, roughness=0.24),
        VisualPart(fea_heatbreak, "v6_ss304_heatbreak", COLOR_STAINLESS, metallic=0.88, roughness=0.25),
        VisualPart(fea_block, "v6_al_heater_block", COLOR_CNC_AL, metallic=0.86, roughness=0.28),
        VisualPart(sock_vis, "v6_silicone_sock", COL_V6_BLUE_SOCK, metallic=0.05, roughness=0.84),
        VisualPart(fea_nozzle, "v6_brass_nozzle", COLOR_BRASS_NOZZLE, metallic=0.92, roughness=0.22),
    ]

    spec = BakedAssetSpec(
        category="hotends",
        asset_id="e3d_v6_groove",
        name="E3D V6 Groove-Mount (40W Al Block, 0.30 mm SS304 Throat)",
        source_cad="E3D V6 Official Drawing parametric multi-material B-Rep (build123d)",
        source_url="https://e3d-online.com/products/v6-all-metal-hotend",
        reference_material_id="stainless_304",
        cad_trust_pct=85,
        port0=PortDefinition(
            (0.0, 0.0, 0.0),
            select_z_annulus(0.0, r_min_mm=2.1, r_max_mm=8.2, tol_mm=0.55),
        ),
        port1=PortDefinition(
            (0.0, 0.0, -62.30),
            select_z_cap(z_max_mm=-57.5, z_min_mm=-62.5, r_max_mm=4.2),
        ),
        n_internal_modes=8,
        mesh_size_mm=1.15,
        is_multi_material=True,
        is_fdm_anisotropic=False,
        part_recipe={
            "overall_length_mm": 62.30,
            "meltzone_length_mm": 12.5,
            "mount": "J-head groove collar Ø16/Ø12 mm @ Z=[0, -9.7] mm",
            "heatbreak": "Unbraced SS304 Ø2.60/Ø2.00 mm (0.30 mm wall, 4.5 mm bare span)",
            "block": "23x16x11.5 mm Al 6061-T6 40W block + V6 brass nozzle",
        },
        visual_max_triangles=22000,
    )
    t0 = time.perf_counter()
    res = bake_solid_superelement(spec, regions=regions, visual_parts=visual_parts, assets_root=assets_root)
    qa = verify_baked_asset("hotends", "e3d_v6_groove", assets_root=assets_root)
    print(f"  {qa.summary} ({time.perf_counter() - t0:.1f}s)")
    if not qa.passed:
        raise RuntimeError(f"QA failed: {qa.summary}")
    return res


# ---------------------------------------------------------------------------
# 2. Phaetus Dragon UHF (Official Dragon STEP + Extended 30 mm Cu UHF Block/Nut)
# ---------------------------------------------------------------------------


def bake_phaetus_dragon_uhf(assets_root: Path = ASSETS_ROOT) -> BakedAssetResult:
    """Bake ``hotends/phaetus_dragon_uhf`` from official Dragon STEP + extended UHF copper melt zone."""
    print("\n=== Baking hotends/phaetus_dragon_uhf ===")
    step_path = HOTEND_STEP_DIR / "phaetus_dragon_hf.step"
    if not step_path.is_file():
        raise FileNotFoundError(f"Missing STEP: {step_path}")

    # Rigid mount length: Dragon HF = 45.51 mm (groove OAL 62.2 mm), Dragon UHF = 64.90 mm
    # (groove OAL 81.6 mm -> +19.39 mm extended copper melt block + UHF M6 extender nut).
    root = b3d.import_step(str(step_path))
    visual_parts: list[VisualPart] = []
    for idx, (path, ws) in enumerate(_walk_world_solids(root)):
        lo = path.lower()
        leaf = lo.split("/")[-1]
        if "bowden" in leaf or "block" in leaf or "nozzle" in leaf:
            continue
        s = ws.translate((6.49, 38.05, 11.30)).rotate(b3d.Axis.X, 90.0)
        vol = float(ws.volume)
        if "extender" in lo:
            # Shift official STEP UHF extender nut down to seat under the extended UHF copper block
            s_ext = s.translate((0.0, 0.0, -9.2))
            visual_parts.append(VisualPart(s_ext, f"dragon_uhf_extender_{idx}", COLOR_COPPER, metallic=0.90, roughness=0.22))
        elif "heatsink" in leaf:
            visual_parts.append(VisualPart(s, f"dragon_uhf_heatsink_{idx}", COLOR_CNC_AL, metallic=0.88, roughness=0.25))
        elif "standoff" in leaf:
            visual_parts.append(VisualPart(s, f"dragon_uhf_ti_standoff_{idx}", COLOR_DARK_TI, metallic=0.86, roughness=0.30))
        elif "heatbreak" in leaf:
            col = COLOR_COPPER if vol > 400.0 else COLOR_DARK_TI
            visual_parts.append(VisualPart(s, f"dragon_uhf_heatbreak_{idx}", col, metallic=0.88, roughness=0.26))
        elif "screw" in leaf:
            visual_parts.append(VisualPart(s, f"dragon_uhf_m25_screw_{idx}", COLOR_STAINLESS, metallic=0.88, roughness=0.28))

    # Identical upper heatsink + 4x Ti standoffs + Ti heatbreak to Dragon HF, plus extended UHF Cu block & nut:
    hs_top = b3d.Solid.make_cylinder(11.15, 4.2, plane=b3d.Plane(origin=(0.0, 0.0, -4.2)))
    hs_core = b3d.Solid.make_cylinder(5.8, 26.0, plane=b3d.Plane(origin=(0.0, 0.0, -26.0)))
    hs_fins = [
        b3d.Solid.make_cylinder(11.15, 1.2, plane=b3d.Plane(origin=(0.0, 0.0, zf)))
        for zf in (-7.2, -10.3, -13.4, -16.5, -19.6, -22.7, -26.0)
    ]
    hs_bore = b3d.Solid.make_cylinder(3.6, 27.5, plane=b3d.Plane(origin=(0.0, 0.0, -26.8)))
    fea_heatsink = hs_top.fuse(hs_core, *hs_fins).cut(hs_bore).clean()

    ti_posts = [
        b3d.Solid.make_cylinder(0.62, 8.5, plane=b3d.Plane(origin=(sx, sy, -31.5)))
        for sx in (-6.0, 6.0)
        for sy in (-6.0, 6.0)
    ]
    ti_throat = (
        b3d.Solid.make_cylinder(1.25, 13.0, plane=b3d.Plane(origin=(0.0, 0.0, -33.0)))
        .cut(b3d.Solid.make_cylinder(0.98, 15.0, plane=b3d.Plane(origin=(0.0, 0.0, -34.0))))
        .clean()
    )
    fea_ti = ti_throat.fuse(*ti_posts).clean()

    cu_sleeve = (
        b3d.Solid.make_cylinder(3.6, 22.5, plane=b3d.Plane(origin=(0.0, 0.0, -24.5)))
        .cut(b3d.Solid.make_cylinder(1.0, 24.5, plane=b3d.Plane(origin=(0.0, 0.0, -25.5))))
        .clean()
    )
    # Standard Dragon plated Cu block (23 x 16 x 11.5 mm @ Z=[-40.0, -28.5]) + slim 17.6 mm M6 Cu UHF
    # melt extender nut (Z=[-57.6, -39.5], 9.2 mm hex across flats) -> 30 mm total melt zone, ~58 g mass
    blk_raw = b3d.Solid.make_box(23.0, 16.0, 11.5, plane=b3d.Plane(origin=(-15.0, -8.0, -40.0)))
    ext_nut = _make_hex_prism(across_flats_mm=9.2, height_mm=18.1, origin_z_mm=-57.6)
    heater_bore = b3d.Solid.make_cylinder(
        3.1, 18.0, plane=b3d.Plane(origin=(-10.5, -9.0, -34.25), z_dir=(0.0, 1.0, 0.0))
    )
    therm_bore = b3d.Solid.make_cylinder(
        1.6, 18.0, plane=b3d.Plane(origin=(-5.5, -9.0, -34.25), z_dir=(0.0, 1.0, 0.0))
    )
    melt_bore = b3d.Solid.make_cylinder(1.0, 31.0, plane=b3d.Plane(origin=(0.0, 0.0, -58.5)))
    fea_copper = blk_raw.fuse(ext_nut, cu_sleeve).cut(heater_bore, therm_bore, melt_bore).clean()

    fea_nozzle = _make_v6_nozzle_solid(tip_z_mm=-64.90, total_len_mm=12.5)

    # Add Dragon Cu block, UHF extender nut, silicone sock, and V6 nozzle to visual GLB
    sock_outer = b3d.Solid.make_box(25.2, 18.2, 12.6, plane=b3d.Plane(origin=(-16.1, -9.1, -40.6)))
    sock_inner = b3d.Solid.make_box(23.1, 16.1, 11.7, plane=b3d.Plane(origin=(-15.05, -8.05, -40.1)))
    sock_hole = b3d.Solid.make_cylinder(5.6, 5.0, plane=b3d.Plane(origin=(0.0, 0.0, -42.5)))
    sock_vis = sock_outer.cut(sock_inner, sock_hole).clean()

    visual_parts.extend(
        [
            VisualPart(blk_raw.cut(heater_bore, therm_bore, melt_bore).clean(), "dragon_uhf_copper_block", COLOR_NICKEL_CU, metallic=0.90, roughness=0.22),
            VisualPart(ext_nut.cut(melt_bore).clean(), "dragon_uhf_extender_sleeve", COLOR_COPPER, metallic=0.90, roughness=0.22),
            VisualPart(sock_vis, "dragon_uhf_silicone_sock", COLOR_DARK_SOCK, metallic=0.05, roughness=0.85),
            VisualPart(fea_nozzle, "dragon_uhf_v6_nozzle", COLOR_BRASS_NOZZLE, metallic=0.92, roughness=0.22),
        ]
    )

    regions = _non_overlapping_regions(
        [
            (fea_nozzle, "copper_c110", "dragon_uhf_v6_nozzle", COLOR_BRASS_NOZZLE, 0.90, 0.24),
            (fea_ti, "titanium_grade5", "dragon_uhf_ti_standoffs_throat", COLOR_DARK_TI, 0.86, 0.30),
            (fea_copper, "copper_c110", "dragon_uhf_cu_block_extender", COLOR_NICKEL_CU, 0.88, 0.24),
            (fea_heatsink, "aluminum_6061_t6", "dragon_uhf_heatsink", COLOR_CNC_AL, 0.88, 0.25),
        ]
    )

    spec = BakedAssetSpec(
        category="hotends",
        asset_id="phaetus_dragon_uhf",
        name="Phaetus Dragon UHF (4-Bolt Rigid, 30 mm Extended Cu Melt Zone)",
        source_cad="phaetus_dragon_hf.step + Phaetus Dragon UHF 30 mm melt-zone B-Rep",
        source_url="https://www.phaetus.com/products/dragon-hotend-uhf",
        reference_material_id="titanium_grade5",
        cad_trust_pct=85,
        port0=PortDefinition(
            (0.0, 0.0, 0.0),
            select_z_annulus(0.0, r_min_mm=3.6, r_max_mm=11.3, tol_mm=0.60),
        ),
        port1=PortDefinition(
            (0.0, 0.0, -64.90),
            select_z_cap(z_max_mm=-60.0, z_min_mm=-65.1, r_max_mm=4.2),
        ),
        n_internal_modes=8,
        mesh_size_mm=1.50,
        is_multi_material=True,
        is_fdm_anisotropic=False,
        part_recipe={
            "rigid_mount_length_mm": 64.90,
            "groove_adapter_oal_mm": 81.6,
            "meltzone_length_mm": 30.0,
            "standoffs": "4x Ø2.30x7.5 mm Ti-6Al-4V @ 12x12 mm square",
            "block": "23x16x21.5 mm plated Cu C110 UHF block + M6 Cu extender nut + V6 nozzle",
        },
        visual_max_triangles=24000,
    )
    t0 = time.perf_counter()
    res = bake_solid_superelement(spec, regions=regions, visual_parts=visual_parts, assets_root=assets_root)
    qa = verify_baked_asset("hotends", "phaetus_dragon_uhf", assets_root=assets_root)
    print(f"  {qa.summary} ({time.perf_counter() - t0:.1f}s)")
    if not qa.passed:
        raise RuntimeError(f"QA failed: {qa.summary}")
    return res


# ---------------------------------------------------------------------------
# 3. Phaetus Rapido 2 UHF (Official Rapido 2 STEP + 8.2 mm Cu UHF Extender)
# ---------------------------------------------------------------------------


def bake_phaetus_rapido_2_uhf(assets_root: Path = ASSETS_ROOT) -> BakedAssetResult:
    """Bake ``hotends/phaetus_rapido_2_uhf`` from official Rapido 2 STEP + UHF M6 extender nut."""
    print("\n=== Baking hotends/phaetus_rapido_2_uhf ===")
    step_path = HOTEND_STEP_DIR / "phaetus_rapido_2_hf.step"
    if not step_path.is_file():
        raise FileNotFoundError(f"Missing STEP: {step_path}")

    # Rigid mount length: Rapido 2 HF = 54.50 mm (groove OAL 71.0 mm), Rapido 2 UHF = 62.70 mm
    # (groove OAL 79.2 mm -> +8.20 mm UHF M6 copper melt extender nut + extended sock).
    root = b3d.import_step(str(step_path))
    visual_parts: list[VisualPart] = []
    for idx, ws in enumerate(root.solids()):
        bb = ws.bounding_box()
        vol = float(ws.volume)
        if bb.max.Z <= -39.5 or bb.max.X > 18.0 or vol < 2.0:
            continue
        s = ws.translate((0.08, 0.22, 41.80)).rotate(b3d.Axis.X, 180.0)
        sbb = s.bounding_box()
        # Skip HF V6 nozzle at the bottom (replaced by UHF extender + shifted V6 nozzle)
        if sbb.min.Z < -51.5 and vol < 500.0:
            continue
        if vol > 4000.0:
            visual_parts.append(VisualPart(s, f"rapido2_uhf_heatsink_{idx}", COLOR_CNC_AL, metallic=0.88, roughness=0.25))
        elif 2500.0 < vol <= 4000.0:
            visual_parts.append(VisualPart(s, f"rapido2_uhf_ceramic_sock_{idx}", COL_RAPIDO_SOCK, metallic=0.06, roughness=0.82))
        elif 1200.0 < vol <= 2500.0:
            visual_parts.append(VisualPart(s, f"rapido2_uhf_copper_core_{idx}", COLOR_NICKEL_CU, metallic=0.90, roughness=0.22))
        elif 24.0 < vol < 32.0 and bb.min.Z < -18.0:
            visual_parts.append(VisualPart(s, f"rapido2_uhf_ti_screw_{idx}", COLOR_DARK_TI, metallic=0.86, roughness=0.28))
        else:
            visual_parts.append(VisualPart(s, f"rapido2_uhf_hardware_{idx}", COLOR_STAINLESS, metallic=0.84, roughness=0.30))

    # Identical upper heatsink, 3x Ti screws, Ti throat, and 85W cylindrical Cu core as Rapido 2 HF,
    # extended by the +8.20 mm Phaetus UHF M6 copper melt extender nut (Z=[-55.20, -47.00]) + V6 nozzle:
    hs_top = b3d.Solid.make_cylinder(10.8, 4.2, plane=b3d.Plane(origin=(0.0, 0.0, -4.2)))
    hs_core = b3d.Solid.make_cylinder(6.0, 26.0, plane=b3d.Plane(origin=(0.0, 0.0, -26.0)))
    hs_fins = [
        b3d.Solid.make_cylinder(10.8, 1.4, plane=b3d.Plane(origin=(0.0, 0.0, zf)))
        for zf in (-7.4, -10.6, -13.8, -17.0, -20.2, -23.4, -26.0)
    ]
    hs_bore = b3d.Solid.make_cylinder(1.5, 27.0, plane=b3d.Plane(origin=(0.0, 0.0, -26.5)))
    fea_heatsink = hs_top.fuse(hs_core, *hs_fins).cut(hs_bore).clean()

    ti_screws = [
        b3d.Solid.make_cylinder(0.85, 11.5, plane=b3d.Plane(origin=(sx, sy, -32.5)))
        for sx, sy in ((3.8, 6.5), (-7.5, 0.0), (3.8, -6.5))
    ]
    ti_throat = (
        b3d.Solid.make_cylinder(1.25, 16.0, plane=b3d.Plane(origin=(0.0, 0.0, -34.0)))
        .cut(b3d.Solid.make_cylinder(0.98, 18.0, plane=b3d.Plane(origin=(0.0, 0.0, -35.0))))
        .clean()
    )
    fea_ti = ti_throat.fuse(*ti_screws).clean()

    cu_flange = b3d.Solid.make_cylinder(8.5, 5.2, plane=b3d.Plane(origin=(0.0, 0.0, -32.7)))
    cu_core = b3d.Solid.make_cylinder(6.4, 15.0, plane=b3d.Plane(origin=(0.0, 0.0, -47.0)))
    cu_uhf_ext = _make_hex_prism(across_flats_mm=9.2, height_mm=8.5, origin_z_mm=-55.2)
    cu_bore = b3d.Solid.make_cylinder(1.0, 30.0, plane=b3d.Plane(origin=(0.0, 0.0, -56.0)))
    fea_copper = cu_flange.fuse(cu_core, cu_uhf_ext).cut(cu_bore).clean()

    fea_nozzle = _make_v6_nozzle_solid(tip_z_mm=-62.70, total_len_mm=12.5)

    # Visual UHF extender nut + lower sock extension + shifted V6 nozzle
    ext_sock = (
        b3d.Solid.make_cylinder(6.6, 8.2, plane=b3d.Plane(origin=(0.0, 0.0, -55.4)))
        .cut(b3d.Solid.make_cylinder(5.2, 9.0, plane=b3d.Plane(origin=(0.0, 0.0, -55.8))))
        .clean()
    )
    visual_parts.extend(
        [
            VisualPart(cu_uhf_ext.cut(cu_bore).clean(), "rapido2_uhf_extender_nut", COLOR_COPPER, metallic=0.90, roughness=0.22),
            VisualPart(ext_sock, "rapido2_uhf_extender_sock", COL_RAPIDO_SOCK, metallic=0.06, roughness=0.82),
            VisualPart(fea_nozzle, "rapido2_uhf_v6_nozzle", COLOR_BRASS_NOZZLE, metallic=0.92, roughness=0.22),
        ]
    )

    regions = _non_overlapping_regions(
        [
            (fea_nozzle, "copper_c110", "rapido2_uhf_v6_nozzle", COLOR_BRASS_NOZZLE, 0.90, 0.24),
            (fea_ti, "titanium_grade5", "rapido2_uhf_3x_ti_screws_throat", COLOR_DARK_TI, 0.86, 0.28),
            (fea_copper, "copper_c110", "rapido2_uhf_cu_core_extender", COLOR_NICKEL_CU, 0.88, 0.24),
            (fea_heatsink, "aluminum_6061_t6", "rapido2_uhf_heatsink", COLOR_CNC_AL, 0.88, 0.25),
        ]
    )

    spec = BakedAssetSpec(
        category="hotends",
        asset_id="phaetus_rapido_2_uhf",
        name="Phaetus Rapido 2 UHF (85W Ceramic, 3x Ti Screws + UHF Extender)",
        source_cad="phaetus_rapido_2_hf.step + Phaetus UHF +8.2 mm M6 Cu extender B-Rep",
        source_url="https://www.phaetus.com/products/rapido2",
        reference_material_id="titanium_grade5",
        cad_trust_pct=85,
        port0=PortDefinition(
            (0.0, 0.0, 0.0),
            select_z_annulus(0.0, r_min_mm=3.0, r_max_mm=10.9, tol_mm=0.60),
        ),
        port1=PortDefinition(
            (0.0, 0.0, -62.70),
            select_z_cap(z_max_mm=-57.5, z_min_mm=-62.9, r_max_mm=4.2),
        ),
        n_internal_modes=8,
        mesh_size_mm=1.45,
        is_multi_material=True,
        is_fdm_anisotropic=False,
        part_recipe={
            "rigid_mount_length_mm": 62.70,
            "groove_adapter_oal_mm": 79.2,
            "meltzone_length_mm": 34.0,
            "standoffs": "3x Ø2.6x11.0 mm Ti-6Al-4V screws @ 120 deg (r = 7.5 mm)",
            "block": "85W cylindrical ceramic Cu core + 8.2 mm M6 Cu UHF extender + V6 nozzle",
        },
        visual_max_triangles=24000,
    )
    t0 = time.perf_counter()
    res = bake_solid_superelement(spec, regions=regions, visual_parts=visual_parts, assets_root=assets_root)
    qa = verify_baked_asset("hotends", "phaetus_rapido_2_uhf", assets_root=assets_root)
    print(f"  {qa.summary} ({time.perf_counter() - t0:.1f}s)")
    if not qa.passed:
        raise RuntimeError(f"QA failed: {qa.summary}")
    return res


# ---------------------------------------------------------------------------
# 4. Monolith 50mm+V6 HF Hotend (80W PT1000, 4-Bolt M3 Rigid, 50 mm Cu Core)
# ---------------------------------------------------------------------------


def bake_monolith_50mm_hf(assets_root: Path = ASSETS_ROOT) -> BakedAssetResult:
    """Bake ``hotends/monolith_50mm_hf`` watertight multi-material Tet4 CMS + PBR GLB."""
    print("\n=== Baking hotends/monolith_50mm_hf ===")

    # Exact PrintDB / Monolith3D 50mm+V6 HF specifications:
    # - OAL = 68.00 mm (Port 0 = rigid 4-bolt M3 top flange Z=0, Port 1 = nozzle tip Z=-68.00 mm)
    # - Melt zone = 50.0 mm (compact cloverleaf Cu C110 core + standard V6 Cu nozzle), 80W 24V heater, PT1000
    # - Dressed mass ~ 52 g (block ~ 22 g), rigid vertical-preload threadless Ti heatbreak + 4x Ti tie posts
    # - Target rigid tip stiffness ~ 900-1300 N/mm (matches PrintDB ~920 N/mm catalog estimate)
    hs_top = b3d.Solid.make_box(22.0, 22.0, 4.0, plane=b3d.Plane(origin=(-11.0, -11.0, -4.0)))
    hs_core = b3d.Solid.make_cylinder(6.4, 17.5, plane=b3d.Plane(origin=(0.0, 0.0, -17.5)))
    hs_fins = [
        b3d.Solid.make_box(21.0, 21.0, 1.3, plane=b3d.Plane(origin=(-10.5, -10.5, zf)))
        for zf in (-7.2, -10.2, -13.2, -16.2)
    ]
    hs_collar = b3d.Solid.make_cylinder(10.5, 1.8, plane=b3d.Plane(origin=(0.0, 0.0, -17.5)))
    hs_bore = b3d.Solid.make_cylinder(1.55, 19.0, plane=b3d.Plane(origin=(0.0, 0.0, -18.2)))
    fea_heatsink = hs_top.fuse(hs_core, hs_collar, *hs_fins).cut(hs_bore).clean()

    # 4x vertical-preload Ti-6Al-4V tie posts @ (±7.8, ±7.8) + 0.25 mm thin-wall threadless Ti heatbreak
    ti_posts = [
        b3d.Solid.make_cylinder(1.50, 9.5, plane=b3d.Plane(origin=(sx, sy, -24.5)))
        for sx in (-7.8, 7.8)
        for sy in (-7.8, 7.8)
    ]
    ti_throat = (
        b3d.Solid.make_cylinder(1.28, 11.5, plane=b3d.Plane(origin=(0.0, 0.0, -25.0)))
        .cut(b3d.Solid.make_cylinder(0.98, 13.5, plane=b3d.Plane(origin=(0.0, 0.0, -26.0))))
        .clean()
    )
    fea_ti = ti_throat.fuse(*ti_posts).clean()

    # Compact 50 mm cloverleaf-inspired Cu C110 melt core (upper preload flange + 3-lobe compact body + 80W boss)
    cu_flange = b3d.Solid.make_box(18.5, 18.5, 4.0, plane=b3d.Plane(origin=(-9.25, -9.25, -25.5)))
    cu_core = b3d.Solid.make_cylinder(5.5, 36.0, plane=b3d.Plane(origin=(0.0, 0.0, -59.5)))
    # Cloverleaf 3-lobe ribs (120 deg) giving compact high-surface melt zone & stiffness without excess mass
    cu_lobes = [
        b3d.Solid.make_cylinder(
            2.9,
            30.0,
            plane=b3d.Plane(
                origin=(4.2 * math.cos(math.radians(a)), 4.2 * math.sin(math.radians(a)), -57.5)
            ),
        )
        for a in (0.0, 120.0, 240.0)
    ]
    heater_bore = b3d.Solid.make_cylinder(3.0, 24.0, plane=b3d.Plane(origin=(4.2, 0.0, -54.0)))
    pt1000_bore = b3d.Solid.make_cylinder(
        1.55, 22.0, plane=b3d.Plane(origin=(-2.1, 3.64, -52.0))
    )
    melt_bore = b3d.Solid.make_cylinder(0.98, 39.0, plane=b3d.Plane(origin=(0.0, 0.0, -60.5)))
    fea_copper = (
        cu_flange.fuse(cu_core, *cu_lobes)
        .cut(heater_bore, pt1000_bore, melt_bore)
        .clean()
    )

    fea_nozzle = _make_v6_nozzle_solid(tip_z_mm=-68.00, total_len_mm=12.5)

    regions = _non_overlapping_regions(
        [
            (fea_nozzle, "copper_c110", "monolith_v6_nozzle", COLOR_BRASS_NOZZLE, 0.90, 0.24),
            (fea_ti, "titanium_grade5", "monolith_ti_preload_posts_throat", COLOR_DARK_TI, 0.86, 0.30),
            (fea_copper, "copper_c110", "monolith_50mm_cloverleaf_cu_core", COLOR_NICKEL_CU, 0.88, 0.24),
            (fea_heatsink, "aluminum_6061_t6", "monolith_4bolt_heatsink", COLOR_CNC_AL, 0.88, 0.25),
        ]
    )

    sock_outer = b3d.Solid.make_cylinder(8.2, 31.0, plane=b3d.Plane(origin=(0.0, 0.0, -59.8)))
    sock_inner = b3d.Solid.make_cylinder(7.0, 32.5, plane=b3d.Plane(origin=(0.0, 0.0, -60.5)))
    sock_vis = sock_outer.cut(sock_inner).clean()

    visual_parts = [
        VisualPart(fea_heatsink, "monolith_4bolt_heatsink", COLOR_CNC_AL, metallic=0.88, roughness=0.25),
        VisualPart(fea_ti, "monolith_ti_preload_posts", COLOR_DARK_TI, metallic=0.86, roughness=0.30),
        VisualPart(fea_copper, "monolith_50mm_cu_core", COLOR_NICKEL_CU, metallic=0.90, roughness=0.22),
        VisualPart(sock_vis, "monolith_silicone_sock", COLOR_DARK_SOCK, metallic=0.05, roughness=0.86),
        VisualPart(fea_nozzle, "monolith_v6_nozzle", COLOR_BRASS_NOZZLE, metallic=0.92, roughness=0.22),
    ]

    spec = BakedAssetSpec(
        category="hotends",
        asset_id="monolith_50mm_hf",
        name="Monolith 50mm+V6 HF Hotend (80W PT1000, 4x Vertical-Preload Ti)",
        source_cad="Monolith3D 50mm+V6 HF multi-material parametric B-Rep (build123d)",
        source_url="https://monolith3d.xyz",
        reference_material_id="titanium_grade5",
        cad_trust_pct=85,
        port0=PortDefinition(
            (0.0, 0.0, 0.0),
            select_plane_box(2, 0.0, tol_mm=0.55, bounds_mm=((-11.2, 11.2), (-11.2, 11.2), (-0.6, 0.6))),
        ),
        port1=PortDefinition(
            (0.0, 0.0, -68.00),
            select_z_cap(z_max_mm=-63.0, z_min_mm=-68.2, r_max_mm=4.2),
        ),
        n_internal_modes=8,
        mesh_size_mm=1.55,
        is_multi_material=True,
        is_fdm_anisotropic=False,
        part_recipe={
            "overall_length_mm": 68.00,
            "meltzone_length_mm": 50.0,
            "mount": "4-bolt M3 rigid flange (22x22 mm) + 4x vertical-preload Ti-6Al-4V posts",
            "heatbreak": "0.25 mm thin-wall threadless Ti-6Al-4V throat",
            "block": "50 mm cloverleaf Cu C110 melt core (80W + PT1000) + V6 copper nozzle",
        },
        visual_max_triangles=22000,
    )
    t0 = time.perf_counter()
    res = bake_solid_superelement(spec, regions=regions, visual_parts=visual_parts, assets_root=assets_root)
    qa = verify_baked_asset("hotends", "monolith_50mm_hf", assets_root=assets_root)
    print(f"  {qa.summary} ({time.perf_counter() - t0:.1f}s)")
    if not qa.passed:
        raise RuntimeError(f"QA failed: {qa.summary}")
    return res


# ---------------------------------------------------------------------------
# 5. Voron StealthBurner + CW2.5 + Voron Tap R8 (Vertical MGN9H Sliding Rail)
# ---------------------------------------------------------------------------


def bake_stealthburner_tap(assets_root: Path = ASSETS_ROOT) -> BakedAssetResult:
    """Bake ``toolheads/stealthburner_tap`` (StealthBurner + CW2.5 mounted on Voron Tap R8 carriage)."""
    print("\n=== Baking toolheads/stealthburner_tap ===")
    if not STEALTHBURNER_STEP.is_file():
        raise FileNotFoundError(f"Missing STEP: {STEALTHBURNER_STEP}")

    # Voron Tap R8 shifts StealthBurner + CW2.5 forward by -8.5 mm in Y relative to MGN12H front face (0,0,0)
    # to accommodate the Rear Tap Rail Mount + 50 mm vertical MGN9 rail + MGN9H sliding block + Front Tap Carriage.
    tap_y_shift_mm = -8.5
    sb_shift = (0.2, -29.1 + tap_y_shift_mm, -50.9)

    root = b3d.import_step(str(STEALTHBURNER_STEP))
    visual_parts: list[VisualPart] = []
    structural_meshes: list[trimesh.Trimesh] = []

    for path, solid in _walk_named_solids(root):
        lo = path.lower()
        leaf = lo.split("/")[-1]
        if any(
            k in lo
            for k in (
                "hfsb5",
                "mgn12",
                "adxl",
                "asxl",
                "sw_carriage",
                "ecas04",
                "hall_effect_retained",
            )
        ):
            continue
        if "nh-sb" in lo and any(
            k in lo
            for k in (
                "r_0402",
                "r_0603",
                "r_0805",
                "c_0402",
                "c_0805",
                "sot-",
                "d_sod",
                "l_0603",
                "fuse_",
                "crystal",
                "kicad_embedded",
            )
        ):
            continue
        if "sb_r2_revovoron/hardware" in lo or "x_carriage_frame_" in leaf:
            # Voron Tap R8 replaces stock X-carriage left/right halves completely
            continue

        tm = _shape_to_trimesh(solid, deflection=0.65)
        if len(tm.faces) == 0:
            continue
        tm = _transform_mesh(tm, translate_mm=sb_shift)

        if any(
            k in lo
            for k in ("hardware", "screw", "insert", "m3x", "m3_", "magnet", "nut", "washer")
        ):
            visual_parts.append(VisualPart(tm, "sb_hw", COL_STEEL, metallic=0.72, roughness=0.32))
        elif "shroud" in lo:
            if any(k in lo for k in ("fan", "5015", "4010", "blower")):
                visual_parts.append(VisualPart(tm, "sb_fan", COL_MOTOR_BODY, metallic=0.2, roughness=0.55))
            elif any(k in lo for k in ("led", "neopixel", "pcb", "diffuser")):
                visual_parts.append(VisualPart(tm, "sb_led", "#f8fafc", metallic=0.1, roughness=0.3))
            else:
                visual_parts.append(VisualPart(tm, "sb_shroud", COL_VORON_RED, metallic=0.15, roughness=0.42))
        elif "x_carriage_printed" in lo or "sb_r2_revovoron/printed_parts" in lo or "cw2_5_printed" in lo:
            visual_parts.append(VisualPart(tm, "sb_body", COL_VORON_BLACK, metallic=0.14, roughness=0.48))
            if (
                "sb_r2_revovoron/printed_parts" in lo
                or "cw2_5_main_body_collet" in lo
                or "cw2_5_motor_plate" in lo
            ):
                structural_meshes.append(tm)
        elif "nema14" in lo:
            visual_parts.append(VisualPart(tm, "cw2_motor", COL_MOTOR_BODY, metallic=0.65, roughness=0.35))
        elif "nh-sb" in lo:
            visual_parts.append(VisualPart(tm, "nitehawk_pcb", COL_PCB_DARK, metallic=0.35, roughness=0.42))
        elif "bmg" in lo:
            visual_parts.append(VisualPart(tm, "bmg_gears", COL_STAINLESS, metallic=0.82, roughness=0.28))
        else:
            visual_parts.append(VisualPart(tm, "sb_hw", COL_STEEL, metallic=0.72, roughness=0.32))

    # Build Voron Tap R8 mechanical interface bridging Port 0 (MGN12H face at Y=0) to shifted StealthBurner:
    # 1. Rear Tap R8 Center Plate (bolted to MGN12H 4x M3 @ Y=0): 48 x 5.0 x 48 mm
    tap_rear_plate = b3d.Solid.make_box(48.0, 5.0, 48.0, plane=b3d.Plane(origin=(-24.0, -5.0, -24.0)))
    # 2. Compliant vertical MGN9H sliding-rail + magnet-preloaded flexure interface (20 x 9.5 x 28.9 mm
    #    matching the real MGN9H sliding carriage block footprint): yields ~26-30 N/mm ABS body stiffness
    #    (~0.34x rigid StealthBurner's 80 N/mm) and ~45-52 N/mm for CNC Al Tap V2!
    tap_flex_neck = b3d.Solid.make_box(20.0, 9.5, 28.9, plane=b3d.Plane(origin=(-10.0, -14.0, -14.45)))
    # 3. Front Tap R8 Carriage (bolted to StealthBurner revovoron_rear Y=-23.0 & cw2_5_motor_plate Y=-15.2)
    tap_front_plate = b3d.Solid.make_box(50.0, 12.0, 52.0, plane=b3d.Plane(origin=(-25.0, -25.0, -30.0)))
    # 4. Visual-only Tap R8 hardware: vertical 50 mm MGN9 rail, MGN9H block, 2x Ø6x3 magnets, OptoTap PCB
    tap_mgn9_rail = b3d.Solid.make_box(9.0, 6.5, 50.0, plane=b3d.Plane(origin=(-4.5, -6.5, -25.0)))
    tap_mgn9h_block = b3d.Solid.make_box(20.0, 5.5, 28.9, plane=b3d.Plane(origin=(-10.0, -12.0, -14.45)))
    tap_optotap_pcb = b3d.Solid.make_box(24.0, 1.6, 16.0, plane=b3d.Plane(origin=(-12.0, -10.0, 16.0)))
    tap_magnets = b3d.Solid.make_cylinder(
        3.0, 3.0, plane=b3d.Plane(origin=(-14.0, -9.5, -18.0))
    ).fuse(
        b3d.Solid.make_cylinder(3.0, 3.0, plane=b3d.Plane(origin=(14.0, -9.5, -18.0)))
    )

    tap_struct_solid = tap_rear_plate.fuse(tap_flex_neck, tap_front_plate).clean()
    structural_meshes.append(_shape_to_trimesh(tap_struct_solid, deflection=0.5))

    visual_parts.extend(
        [
            VisualPart(tap_rear_plate, "tap_r8_rear_plate", COL_VORON_BLACK, metallic=0.14, roughness=0.48),
            VisualPart(tap_mgn9_rail, "tap_r8_mgn9_rail", COL_STAINLESS, metallic=0.88, roughness=0.24),
            VisualPart(tap_mgn9h_block, "tap_r8_mgn9h_block", COL_STEEL, metallic=0.78, roughness=0.30),
            VisualPart(tap_front_plate, "tap_r8_front_carriage", COL_VORON_RED, metallic=0.15, roughness=0.42),
            VisualPart(tap_optotap_pcb, "tap_r8_optotap_pcb", COL_PCB_DARK, metallic=0.35, roughness=0.42),
            VisualPart(tap_magnets, "tap_r8_magnets", COL_MAGNET_SILVER, metallic=0.92, roughness=0.18),
        ]
    )

    struct_combined = _combine_meshes(structural_meshes)
    bmin, bmax = struct_combined.bounds
    print(f"  StealthBurner+Tap structural bounds (mm): min={np.round(bmin, 1)} max={np.round(bmax, 1)}")

    # Strictly disjoint planar bolt pads (dmin > 27 mm, zero bridging tets):
    # Port 0 = MGN12H 4x M3 front carriage bolt face at Y = 0.0 mm
    # Port 1 = Revo Voron 4x M3 hotend mounting shelf shifted by -8.5 mm in Y -> (0.2, -37.4, -1.9)
    port0_mm = (0.0, 0.0, 0.0)
    port1_mm = (0.2, -28.9 + tap_y_shift_mm, -1.9)

    def _select_tap_mgn12h_face(coords_mm: np.ndarray) -> np.ndarray:
        mask = (
            (np.abs(coords_mm[:, 1]) <= 2.5)
            & (np.abs(coords_mm[:, 0]) <= 15.0)
            & (np.abs(coords_mm[:, 2]) <= 14.0)
        )
        if int(np.sum(mask)) >= 24:
            return mask
        return _selector_box_or_nearest((0.0, 0.0, 0.0), 10.0, min_nodes=24)(coords_mm)

    def _select_tap_hotend_shelf(coords_mm: np.ndarray) -> np.ndarray:
        mask = (
            (coords_mm[:, 2] >= -4.5)
            & (coords_mm[:, 2] <= 2.5)
            & (coords_mm[:, 0] >= -10.0)
            & (coords_mm[:, 0] <= 10.5)
            & (coords_mm[:, 1] >= -36.5 + tap_y_shift_mm)
            & (coords_mm[:, 1] <= -21.5 + tap_y_shift_mm)
        )
        if int(np.sum(mask)) >= 24:
            return mask
        return _selector_box_or_nearest(port1_mm, 8.0, min_nodes=24)(coords_mm)

    spec = BakedAssetSpec(
        category="toolheads",
        asset_id="stealthburner_tap",
        name="Voron StealthBurner + CW2.5 + Voron Tap R8 (ABS/ASA Sliding Rail)",
        source_cad="Stealthburner_CW2.5_Assembly.step + Voron Tap R8 MGN9H carriage (build123d)",
        source_url="https://github.com/VoronDesign/Voron-Tap",
        reference_material_id="abs_printed",
        cad_trust_pct=85,
        port0=PortDefinition(
            vp_coord_mm=port0_mm,
            selector=_select_tap_mgn12h_face,
        ),
        port1=PortDefinition(
            vp_coord_mm=port1_mm,
            selector=_select_tap_hotend_shelf,
        ),
        n_internal_modes=6,
        mesh_size_mm=4.0,
        is_multi_material=False,
        is_fdm_anisotropic=True,
        part_recipe={
            "origin": "MGN12H front carriage face (0,0,0)",
            "hotend_port_mm": list(port1_mm),
            "tap_y_shift_mm": tap_y_shift_mm,
            "baked_z_probe_type": "printed_tap",
            "baked_probe_compliance_factor": 0.34,
            "tap_mechanism": "50 mm vertical MGN9H sliding rail + 2x Ø6x3 mm neodymium magnets + OptoTap PCB",
        },
        visual_max_triangles=32000,
    )
    t0 = time.perf_counter()
    res = bake_solid_superelement(
        spec,
        regions=[
            MaterialRegion(
                solid=struct_combined,
                material_id="abs_printed",
                label="stealthburner_tap_body",
                fdm_anisotropic=True,
            )
        ],
        visual_parts=visual_parts,
        assets_root=assets_root,
    )
    qa = verify_baked_asset("toolheads", "stealthburner_tap", assets_root=assets_root)
    print(f"  {qa.summary} ({time.perf_counter() - t0:.1f}s)")
    if not qa.passed:
        raise RuntimeError(f"QA failed: {qa.summary}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Bake 4 missing catalog hotends + Voron StealthBurner Tap R8.")
    ap.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated subset: e3d_v6_groove,phaetus_dragon_uhf,phaetus_rapido_2_uhf,monolith_50mm_hf,stealthburner_tap",
    )
    args = ap.parse_args()
    all_targets = {
        "e3d_v6_groove": bake_e3d_v6_groove,
        "phaetus_dragon_uhf": bake_phaetus_dragon_uhf,
        "phaetus_rapido_2_uhf": bake_phaetus_rapido_2_uhf,
        "monolith_50mm_hf": bake_monolith_50mm_hf,
        "stealthburner_tap": bake_stealthburner_tap,
    }
    sel = (
        [s.strip().lower() for s in args.only.split(",") if s.strip()]
        if args.only
        else list(all_targets.keys())
    )
    for key in sel:
        if key not in all_targets:
            raise ValueError(f"Unknown target {key!r}; expected one of {list(all_targets)}")
        all_targets[key]()


if __name__ == "__main__":
    main()
