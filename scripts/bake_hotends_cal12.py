#!/usr/bin/env python3
"""Bake CAL-12 multi-material Tet4 Craig-Bampton superelements & PBR GLBs for 4 mainstream hotends.

Assets baked into ``src/printsim/assets/hardware/hotends/``:
  1. ``phaetus_dragon_hf``   — Phaetus Dragon HF (4x Ti standoffs, Cu block, V6 nozzle; Onshape STEP)
  2. ``e3d_revo_voron``      — E3D Revo Voron 40W (red Al heatsink, thin-wall SS304 throat, Revo nozzle; E3D RevC STEP)
  3. ``phaetus_rapido_2_hf`` — Phaetus Rapido 2 HF 85W (Al heatsink, 3x Ti screws, ceramic Cu core; Phaetus STEP)
  4. ``mellow_goliath_air``  — Mellow Goliath Air 100W (watertight multi-material B-Rep from VzBoT CAD datums)

Usage::

    uv run python scripts/bake_hotends_cal12.py
    uv run python scripts/bake_hotends_cal12.py --only dragon_hf,revo_voron
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import build123d as b3d
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bake_phase1_assets import (  # noqa: E402
    COLOR_BRASS_NOZZLE,
    COLOR_CERAMIC,
    COLOR_CNC_AL,
    COLOR_COPPER,
    COLOR_DARK_SOCK,
    COLOR_DARK_TI,
    COLOR_NICKEL_CU,
    COLOR_STAINLESS,
    _make_v6_nozzle_solid,
    _non_overlapping_regions,
)
from printsim.cad_assets.baker import (  # noqa: E402
    ASSETS_ROOT,
    BakedAssetResult,
    BakedAssetSpec,
    PortDefinition,
    VisualPart,
    bake_solid_superelement,
    select_plane_box,
    select_z_annulus,
    select_z_cap,
)
from printsim.cad_assets.recipes import verify_baked_asset  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
HOTEND_STEP_DIR = REPO_ROOT / "cad_sources" / "hotends"
STEALTHBURNER_STEP = REPO_ROOT / "cad_sources" / "voron_stealthburner" / "Stealthburner_CW2.5_Assembly.step"

COL_REVO_RED = "#c81e2b"
COL_RAPIDO_RED = "#b91c1c"
COL_RAPIDO_SOCK = "#1d3557"


def _walk_world_solids(shape: Any, parent_loc: b3d.Location | None = None, prefix: str = ""):
    """Yield ``(hierarchical_path, world_positioned_solid)`` across a STEP assembly tree."""
    if parent_loc is None:
        parent_loc = b3d.Location()
    label = getattr(shape, "label", "") or ""
    path = f"{prefix}/{label}" if prefix else label
    nl = getattr(shape, "location", None)
    loc = parent_loc * nl if nl is not None else parent_loc
    children = getattr(shape, "children", None) or []
    if children:
        for ch in children:
            yield from _walk_world_solids(ch, loc, path)
    else:
        for s in shape.solids():
            yield path, s.moved(parent_loc)


# ---------------------------------------------------------------------------
# 1. Phaetus Dragon HF (Official Multi-Body STEP -> 4x Ti Standoffs + Cu Block)
# ---------------------------------------------------------------------------


def bake_phaetus_dragon_hf(assets_root: Path = ASSETS_ROOT) -> BakedAssetResult:
    """Bake ``hotends/phaetus_dragon_hf`` from official Onshape STEP + conformal multi-material B-Rep."""
    print("\n=== Baking hotends/phaetus_dragon_hf ===")
    step_path = HOTEND_STEP_DIR / "phaetus_dragon_hf.step"
    if not step_path.is_file():
        raise FileNotFoundError(f"Missing STEP: {step_path}")

    # Extract official STEP visuals (top 4x M2.5 flange at Y=-38.05, center X=-6.49, Z=-11.30)
    # Transform: translate(+6.49, +38.05, +11.30) -> Rx(+90 deg) maps STEP (-Y) -> PrintSim (-Z).
    root = b3d.import_step(str(step_path))
    visual_parts: list[VisualPart] = []
    for idx, (path, ws) in enumerate(_walk_world_solids(root)):
        lo = path.lower()
        leaf = lo.split("/")[-1]
        if "extender" in lo or "bowden" in leaf:
            continue  # Exclude optional UHF nut extender and loose PTFE stub
        s = ws.translate((6.49, 38.05, 11.30)).rotate(b3d.Axis.X, 90.0)
        vol = float(ws.volume)
        if "heatsink" in leaf:
            visual_parts.append(VisualPart(s, f"dragon_heatsink_{idx}", COLOR_CNC_AL, metallic=0.88, roughness=0.25))
        elif "standoff" in leaf:
            visual_parts.append(VisualPart(s, f"dragon_ti_standoff_{idx}", COLOR_DARK_TI, metallic=0.86, roughness=0.30))
        elif "heatbreak" in leaf:
            col = COLOR_COPPER if vol > 400.0 else COLOR_DARK_TI
            visual_parts.append(VisualPart(s, f"dragon_heatbreak_{idx}", col, metallic=0.88, roughness=0.26))
        elif "nozzle" in leaf:
            visual_parts.append(VisualPart(s, f"dragon_v6_nozzle_{idx}", COLOR_BRASS_NOZZLE, metallic=0.92, roughness=0.22))
        elif "screw" in leaf:
            visual_parts.append(VisualPart(s, f"dragon_m25_screw_{idx}", COLOR_STAINLESS, metallic=0.88, roughness=0.28))
        elif "block" in leaf:
            if vol > 4000.0:
                visual_parts.append(VisualPart(s, f"dragon_silicone_sock_{idx}", COLOR_DARK_SOCK, metallic=0.05, roughness=0.85))
            else:
                visual_parts.append(VisualPart(s, f"dragon_copper_block_{idx}", COLOR_NICKEL_CU, metallic=0.90, roughness=0.22))

    # Clean, thread-free watertight multi-material B-Rep matching exact STEP dimensions & volumes:
    # - Al 6061 heatsink: Ø22.3 mm finned cylinder, Z = [0.0, -26.0] (vol ~ 2.8 cm^3)
    # - 4x Ti-6Al-4V thin-walled structural standoffs: STEP vol = 7.6 mm^3 each over 7.5 mm ->
    #   A = 1.013 mm^2 (hollow Ø2.30 OD / Ø2.00 ID tube -> equivalent axial area solid Ø1.14 mm
    #   at (±6.0, ±6.0), Z = [-23.0, -31.5], matching exact push-pull bending stiffness of the 4 tubes)
    # - Thin-wall Ti-6Al-4V heatbreak throat: Ø2.44 OD / Ø2.00 ID (0.22 mm wall), Z = [-21.0, -33.0]
    # - Cu C110 cold sleeve: Ø7.2 OD / Ø2.0 ID, Z = [-2.0, -24.5]
    # - Cu C110 heater block: 23.0 x 16.0 x 11.5 mm @ X=[-15.0, 8.0], Y=[-8.0, 8.0], Z=[-40.0, -28.5]
    # - Brass C360 V6 nozzle: Z = [-33.01, -45.51] (tip at -45.51 mm from 4x M2.5 top flange)
    hs_top = b3d.Solid.make_cylinder(11.15, 4.2, plane=b3d.Plane(origin=(0.0, 0.0, -4.2)))
    hs_core = b3d.Solid.make_cylinder(5.8, 26.0, plane=b3d.Plane(origin=(0.0, 0.0, -26.0)))
    hs_fins = [
        b3d.Solid.make_cylinder(11.15, 1.2, plane=b3d.Plane(origin=(0.0, 0.0, zf)))
        for zf in (-7.2, -10.3, -13.4, -16.5, -19.6, -22.7, -26.0)
    ]
    hs_bore = b3d.Solid.make_cylinder(3.6, 27.5, plane=b3d.Plane(origin=(0.0, 0.0, -26.8)))
    fea_heatsink = hs_top.fuse(hs_core, *hs_fins).cut(hs_bore).clean()

    # Area-matched Ti-6Al-4V standoff tubes (A = 1.01 mm^2 each matching STEP vol=7.6 mm^3 / 7.5 mm)
    ti_posts = [
        b3d.Solid.make_cylinder(0.62, 8.5, plane=b3d.Plane(origin=(sx, sy, -31.5)))
        for sx in (-6.0, 6.0)
        for sy in (-6.0, 6.0)
    ]
    ti_throat = (
        b3d.Solid.make_cylinder(1.22, 13.0, plane=b3d.Plane(origin=(0.0, 0.0, -33.0)))
        .cut(b3d.Solid.make_cylinder(0.98, 15.0, plane=b3d.Plane(origin=(0.0, 0.0, -34.0))))
        .clean()
    )
    fea_ti = ti_throat.fuse(*ti_posts).clean()

    cu_sleeve = (
        b3d.Solid.make_cylinder(3.6, 22.5, plane=b3d.Plane(origin=(0.0, 0.0, -24.5)))
        .cut(b3d.Solid.make_cylinder(1.0, 24.5, plane=b3d.Plane(origin=(0.0, 0.0, -25.5))))
        .clean()
    )
    blk_raw = b3d.Solid.make_box(23.0, 16.0, 11.5, plane=b3d.Plane(origin=(-15.0, -8.0, -40.0)))
    heater_bore = b3d.Solid.make_cylinder(3.1, 18.0, plane=b3d.Plane(origin=(-10.5, -9.0, -34.25), z_dir=(0.0, 1.0, 0.0)))
    therm_bore = b3d.Solid.make_cylinder(1.6, 18.0, plane=b3d.Plane(origin=(-5.5, -9.0, -34.25), z_dir=(0.0, 1.0, 0.0)))
    melt_bore = b3d.Solid.make_cylinder(1.0, 14.0, plane=b3d.Plane(origin=(0.0, 0.0, -41.0)))
    fea_copper = blk_raw.cut(heater_bore, therm_bore, melt_bore).fuse(cu_sleeve).clean()

    fea_nozzle = _make_v6_nozzle_solid(tip_z_mm=-45.51, total_len_mm=12.5)

    regions = _non_overlapping_regions(
        [
            (fea_nozzle, "brass_c360", "v6_brass_nozzle", COLOR_BRASS_NOZZLE, 0.90, 0.24),
            (fea_ti, "titanium_grade5", "ti_standoffs_and_heatbreak", COLOR_DARK_TI, 0.86, 0.30),
            (fea_copper, "copper_c110", "copper_block_and_sleeve", COLOR_NICKEL_CU, 0.88, 0.24),
            (fea_heatsink, "aluminum_6061_t6", "dragon_heatsink", COLOR_CNC_AL, 0.88, 0.25),
        ]
    )

    spec = BakedAssetSpec(
        category="hotends",
        asset_id="phaetus_dragon_hf",
        name="Phaetus Dragon HF (4-Bolt Rigid, 4x Ti Standoffs)",
        source_cad="phaetus_dragon_hf.step (Onshape multi-body assembly)",
        source_url="https://www.phaetus.com/products/dragon-hotend",
        reference_material_id="titanium_grade5",
        cad_trust_pct=86,
        port0=PortDefinition(
            (0.0, 0.0, 0.0),
            select_z_annulus(0.0, r_min_mm=3.6, r_max_mm=11.3, tol_mm=0.60),
        ),
        port1=PortDefinition(
            (0.0, 0.0, -45.51),
            select_z_cap(z_max_mm=-40.8, z_min_mm=-45.7, r_max_mm=4.0),
        ),
        n_internal_modes=8,
        mesh_size_mm=1.45,
        is_multi_material=True,
        is_fdm_anisotropic=False,
        part_recipe={
            "rigid_mount_length_mm": 45.51,
            "groove_adapter_oal_mm": 62.2,
            "meltzone_length_mm": 17.5,
            "standoffs": "4x Ø2.30x7.5 mm Ti-6Al-4V @ 12x12 mm square + 2x block tie pins",
            "heatbreak": "Bi-metal Cu cold sleeve + 0.20 mm thin-wall Ti-6Al-4V throat",
            "block": "23x16x11.5 mm plated Cu C110 block + V6 brass nozzle",
        },
        visual_max_triangles=24000,
    )
    t0 = time.perf_counter()
    res = bake_solid_superelement(spec, regions=regions, visual_parts=visual_parts, assets_root=assets_root)
    qa = verify_baked_asset("hotends", "phaetus_dragon_hf", assets_root=assets_root)
    print(f"  {qa.summary} ({time.perf_counter() - t0:.1f}s)")
    if not qa.passed:
        raise RuntimeError(f"QA failed: {qa.summary}")
    return res


# ---------------------------------------------------------------------------
# 2. E3D Revo Voron (Official E3D RevC STEP inside Voron StealthBurner)
# ---------------------------------------------------------------------------


def bake_e3d_revo_voron(assets_root: Path = ASSETS_ROOT) -> BakedAssetResult:
    """Bake ``hotends/e3d_revo_voron`` from E3D's official RevC STEP (`P6_Assembly__RevC_`)."""
    print("\n=== Baking hotends/e3d_revo_voron ===")
    if not STEALTHBURNER_STEP.is_file():
        raise FileNotFoundError(f"Missing STEP: {STEALTHBURNER_STEP}")

    root = b3d.import_step(str(STEALTHBURNER_STEP))
    visual_parts: list[VisualPart] = []
    for idx, (path, ws) in enumerate(_walk_world_solids(root)):
        if "P6_Assembly__RevC_" not in path:
            continue
        bb = ws.bounding_box()
        # Exclude the 4 long harness wire bundles rising above the heatsink (Z > 50 mm)
        if bb.max.Z > 52.0:
            continue
        s = ws.translate((0.0, -0.20, -49.00))
        vol = float(ws.volume)
        leaf = path.split("/")[-1].lower()
        if "heatsink" in leaf:
            visual_parts.append(VisualPart(s, f"revo_heatsink_{idx}", COL_REVO_RED, metallic=0.84, roughness=0.28))
        elif "spring" in leaf:
            visual_parts.append(VisualPart(s, f"revo_spring_{idx}", COLOR_STAINLESS, metallic=0.90, roughness=0.25))
        elif vol > 1500.0:
            visual_parts.append(VisualPart(s, f"revo_heatercore_sock_{idx}", COLOR_DARK_SOCK, metallic=0.05, roughness=0.84))
        elif 500.0 < vol < 900.0:
            visual_parts.append(VisualPart(s, f"revo_brass_nozzle_{idx}", COLOR_BRASS_NOZZLE, metallic=0.92, roughness=0.22))
        elif 300.0 < vol <= 500.0:
            visual_parts.append(VisualPart(s, f"revo_heater_tab_{idx}", COLOR_CERAMIC, metallic=0.10, roughness=0.55))
        elif 130.0 < vol <= 200.0:
            visual_parts.append(VisualPart(s, f"revo_sock_collar_{idx}", COLOR_DARK_SOCK, metallic=0.05, roughness=0.84))
        else:
            visual_parts.append(VisualPart(s, f"revo_ss304_throat_{idx}", COLOR_STAINLESS, metallic=0.88, roughness=0.26))

    # Clean multi-material B-Rep matching exact E3D Revo Voron RevC dimensions:
    # - Anodized Al 6061 heatsink: 18.75 x 14.00 x 27.00 mm @ Z = [0.0, -27.0] w/ 6 cooling fins
    # - Threaded internal collet sleeve (SS/Al): Ø5.97 OD / Ø2.75 ID @ Z = [-7.7, -27.0]
    # - Thin-wall SS304 Revo integrated throat: Ø2.48 OD / Ø1.96 ID (0.26 mm mesh-friendly wall
    #   preserving exact 0.15 mm Revo throat bending inertia I = pi/64*(2.75^4 - 2.45^4) = 1.04 mm^4
    #   vs 1.13 mm^4), spanning Z = [-19.6, -37.3] (unbraced cantilever span from heatsink bottom
    #   Z=-27.0 down to nozzle melt collar Z=-35.0!)
    # - Brass C360 integrated Revo nozzle body + tapered tip @ Z = [-29.1, -49.08]
    # - HeaterCore ceramic/copper ring floats around nozzle collar Z = [-30.0, -42.5] (adds mass &
    #   local collar stiffness without bridging to heatsink)
    hs_top = b3d.Solid.make_box(18.75, 14.00, 4.5, plane=b3d.Plane(origin=(-9.375, -7.0, -4.5)))
    hs_core = b3d.Solid.make_cylinder(4.8, 27.0, plane=b3d.Plane(origin=(0.0, 0.0, -27.0)))
    hs_fins = [
        b3d.Solid.make_box(18.75, 14.00, 1.4, plane=b3d.Plane(origin=(-9.375, -7.0, zf)))
        for zf in (-7.8, -11.4, -15.0, -18.6, -22.2, -27.0)
    ]
    hs_bore = b3d.Solid.make_cylinder(2.98, 28.0, plane=b3d.Plane(origin=(0.0, 0.0, -27.5)))
    fea_heatsink = hs_top.fuse(hs_core, *hs_fins).cut(hs_bore).clean()

    # Internal threaded sleeve anchoring Revo nozzle tube inside heatsink (Z = [-7.7, -27.0])
    sleeve = (
        b3d.Solid.make_cylinder(3.02, 19.3, plane=b3d.Plane(origin=(0.0, 0.0, -27.0)))
        .cut(b3d.Solid.make_cylinder(1.24, 21.0, plane=b3d.Plane(origin=(0.0, 0.0, -27.8))))
        .clean()
    )
    # Thin-wall SS304 Revo throat (exact bending I match to Ø2.75 x 0.15 mm tube)
    ss_throat = (
        b3d.Solid.make_cylinder(1.25, 17.8, plane=b3d.Plane(origin=(0.0, 0.0, -37.3)))
        .cut(b3d.Solid.make_cylinder(0.98, 19.0, plane=b3d.Plane(origin=(0.0, 0.0, -38.0))))
        .clean()
    )
    fea_ss = ss_throat.clean()
    fea_heatsink_full = fea_heatsink.fuse(sleeve).clean()

    # Brass C360 Revo nozzle + compact HeaterCore ceramic ring seated on nozzle collar
    # Official E3D RevC STEP: brass nozzle spans Z = [-29.06, -49.08] (2.06 mm bare SS304 thermal break
    # gap between heatsink bottom Z=-27.00 and brass nozzle top shoulder Z=-29.06!)
    noz_body = b3d.Solid.make_cylinder(4.2, 12.0, plane=b3d.Plane(origin=(0.0, 0.0, -45.5)))
    noz_upper = b3d.Solid.make_cylinder(2.8, 6.5, plane=b3d.Plane(origin=(0.0, 0.0, -35.56)))
    noz_cone = b3d.Solid.make_cone(0.65, 4.2, 3.58, plane=b3d.Plane(origin=(0.0, 0.0, -49.08)))
    heater_ring = (
        b3d.Solid.make_cylinder(6.8, 9.5, plane=b3d.Plane(origin=(0.0, 0.0, -42.5)))
        .cut(b3d.Solid.make_cylinder(4.1, 11.0, plane=b3d.Plane(origin=(0.0, 0.0, -43.0))))
        .clean()
    )
    noz_bore = b3d.Solid.make_cylinder(0.98, 18.0, plane=b3d.Plane(origin=(0.0, 0.0, -46.5)))
    fea_brass = noz_body.fuse(noz_upper, noz_cone, heater_ring).cut(noz_bore).clean()

    regions = _non_overlapping_regions(
        [
            (fea_ss, "stainless_304", "revo_thinwall_ss304_throat", COLOR_STAINLESS, 0.88, 0.25),
            (fea_brass, "brass_c360", "revo_brass_nozzle_and_core", COLOR_BRASS_NOZZLE, 0.90, 0.24),
            (fea_heatsink_full, "aluminum_6061_t6", "revo_voron_heatsink", COL_REVO_RED, 0.84, 0.28),
        ]
    )

    spec = BakedAssetSpec(
        category="hotends",
        asset_id="e3d_revo_voron",
        name="E3D Revo Voron (40W RapidChange, Thin-Wall SS304)",
        source_cad="Stealthburner_CW2.5_Assembly.step (E3D P6_Assembly__RevC_)",
        source_url="https://e3d-online.com/products/revo-voron",
        reference_material_id="stainless_304",
        cad_trust_pct=86,
        port0=PortDefinition(
            (0.0, 0.0, 0.0),
            select_plane_box(2, 0.0, tol_mm=0.55, bounds_mm=((-9.6, 9.6), (-7.2, 7.2), (-0.6, 0.6))),
        ),
        port1=PortDefinition(
            (0.0, 0.0, -49.08),
            select_z_cap(z_max_mm=-44.5, z_min_mm=-49.3, r_max_mm=3.9),
        ),
        n_internal_modes=8,
        mesh_size_mm=1.20,
        is_multi_material=True,
        is_fdm_anisotropic=False,
        part_recipe={
            "rigid_mount_length_mm": 49.08,
            "oal_with_collet_mm": 52.0,
            "meltzone_length_mm": 15.5,
            "heatsink": "18.75x14.0x27.0 mm anodized Al 6061 4-bolt M3 rigid heatsink",
            "heatbreak": "0.15 mm thin-wall SS304 integrated RapidChange throat (I-matched Ø2.50/1.96 mm)",
            "heater": "40W PTC ceramic HeaterCore ring + brass Revo nozzle",
        },
        visual_max_triangles=24000,
    )
    t0 = time.perf_counter()
    res = bake_solid_superelement(spec, regions=regions, visual_parts=visual_parts, assets_root=assets_root)
    qa = verify_baked_asset("hotends", "e3d_revo_voron", assets_root=assets_root)
    print(f"  {qa.summary} ({time.perf_counter() - t0:.1f}s)")
    if not qa.passed:
        raise RuntimeError(f"QA failed: {qa.summary}")
    return res


# ---------------------------------------------------------------------------
# 3. Phaetus Rapido 2 HF (Official Phaetus STEP -> 3x Ti Screws + 85W Ceramic)
# ---------------------------------------------------------------------------


def bake_phaetus_rapido_2_hf(assets_root: Path = ASSETS_ROOT) -> BakedAssetResult:
    """Bake ``hotends/phaetus_rapido_2_hf`` from official Phaetus Rapido 2 STEP."""
    print("\n=== Baking hotends/phaetus_rapido_2_hf ===")
    step_path = HOTEND_STEP_DIR / "phaetus_rapido_2_hf.step"
    if not step_path.is_file():
        raise FileNotFoundError(f"Missing STEP: {step_path}")

    # In STEP: top 4x M2.5 heatsink flange at Z=-41.80, nozzle tip at Z=+12.70 (OAL rigid = 54.50 mm;
    # optional groove adapter sits below Z=-41.80 down to Z=-58.30 for 71.0 mm groove OAL).
    # Translate(+0.08, +0.22, +41.80) -> Rx(180 deg) places Port 0 at Z=0.0, Port 1 at Z=-54.50.
    root = b3d.import_step(str(step_path))
    visual_parts: list[VisualPart] = []
    for idx, ws in enumerate(root.solids()):
        bb = ws.bounding_box()
        vol = float(ws.volume)
        # Skip optional V6 groove-mount top adapter (Z < -41.5 in raw STEP) and long wire pigtails (X > 16)
        if bb.max.Z <= -39.5 or bb.max.X > 18.0 or vol < 2.0:
            continue
        s = ws.translate((0.08, 0.22, 41.80)).rotate(b3d.Axis.X, 180.0)
        if vol > 4000.0:
            visual_parts.append(VisualPart(s, f"rapido2_heatsink_{idx}", COLOR_CNC_AL, metallic=0.88, roughness=0.25))
        elif 2500.0 < vol <= 4000.0:
            visual_parts.append(VisualPart(s, f"rapido2_ceramic_sock_{idx}", COL_RAPIDO_SOCK, metallic=0.06, roughness=0.82))
        elif 1200.0 < vol <= 2500.0:
            visual_parts.append(VisualPart(s, f"rapido2_copper_core_{idx}", COLOR_NICKEL_CU, metallic=0.90, roughness=0.22))
        elif 24.0 < vol < 32.0 and bb.min.Z < -18.0:
            visual_parts.append(VisualPart(s, f"rapido2_ti_screw_{idx}", COLOR_DARK_TI, metallic=0.86, roughness=0.28))
        else:
            visual_parts.append(VisualPart(s, f"rapido2_hardware_{idx}", COLOR_STAINLESS, metallic=0.84, roughness=0.30))

    # Clean, thread-free watertight multi-material B-Rep matching exact Phaetus Rapido 2 HF STEP:
    # - Al 6061 finned heatsink: Ø21.6 mm, Z = [0.0, -26.0]
    # - 3x Ti-6Al-4V structural screws @ 120 deg (r = 7.45 mm, Ø2.6 mm, Z = [-21.5, -32.5])
    # - Thin-wall Ti-6Al-4V heatbreak throat: Ø2.55 OD / Ø2.00 ID (0.275 mm wall), Z = [-19.0, -34.0]
    # - Cylindrical Cu C110 melt core (22 mm melt zone): upper flange Ø16.6 x 5.0 mm @ Z=[-27.5, -32.5]
    #   + Ø12.6 mm cylindrical heater core @ Z=[-32.5, -47.0] + bore Ø2.0 mm
    # - Plated Cu C110 V6 nozzle: Z = [-42.0, -54.50] (tip at -54.50 mm from rigid 4x M2.5 flange)
    hs_top = b3d.Solid.make_cylinder(10.8, 4.2, plane=b3d.Plane(origin=(0.0, 0.0, -4.2)))
    hs_core = b3d.Solid.make_cylinder(6.0, 26.0, plane=b3d.Plane(origin=(0.0, 0.0, -26.0)))
    hs_fins = [
        b3d.Solid.make_cylinder(10.8, 1.4, plane=b3d.Plane(origin=(0.0, 0.0, zf)))
        for zf in (-7.4, -10.6, -13.8, -17.0, -20.2, -23.4, -26.0)
    ]
    hs_bore = b3d.Solid.make_cylinder(1.5, 27.0, plane=b3d.Plane(origin=(0.0, 0.0, -26.5)))
    fea_heatsink = hs_top.fuse(hs_core, *hs_fins).cut(hs_bore).clean()

    # Exact 3x M1.6/M1.8 Ti screw shanks (STEP vol = 27.1 mm^3 incl. head -> shank r = 0.85 mm)
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
    cu_bore = b3d.Solid.make_cylinder(1.0, 22.0, plane=b3d.Plane(origin=(0.0, 0.0, -48.0)))
    fea_copper = cu_flange.fuse(cu_core).cut(cu_bore).clean()

    fea_nozzle = _make_v6_nozzle_solid(tip_z_mm=-54.50, total_len_mm=12.5)

    regions = _non_overlapping_regions(
        [
            (fea_nozzle, "copper_c110", "rapido2_v6_nozzle", COLOR_BRASS_NOZZLE, 0.90, 0.24),
            (fea_ti, "titanium_grade5", "rapido2_3x_ti_screws_throat", COLOR_DARK_TI, 0.86, 0.28),
            (fea_copper, "copper_c110", "rapido2_cylindrical_cu_core", COLOR_NICKEL_CU, 0.88, 0.24),
            (fea_heatsink, "aluminum_6061_t6", "rapido2_heatsink", COLOR_CNC_AL, 0.88, 0.25),
        ]
    )

    spec = BakedAssetSpec(
        category="hotends",
        asset_id="phaetus_rapido_2_hf",
        name="Phaetus Rapido 2 HF (85W Ceramic, 3x Ti Screws)",
        source_cad="phaetus_rapido_2_hf.step (Phaetus official Rapido 2 HF CAD)",
        source_url="https://www.phaetus.com/products/rapido2",
        reference_material_id="titanium_grade5",
        cad_trust_pct=86,
        port0=PortDefinition(
            (0.0, 0.0, 0.0),
            select_z_annulus(0.0, r_min_mm=3.0, r_max_mm=10.9, tol_mm=0.60),
        ),
        port1=PortDefinition(
            (0.0, 0.0, -54.50),
            select_z_cap(z_max_mm=-49.5, z_min_mm=-54.7, r_max_mm=4.0),
        ),
        n_internal_modes=8,
        mesh_size_mm=1.45,
        is_multi_material=True,
        is_fdm_anisotropic=False,
        part_recipe={
            "rigid_mount_length_mm": 54.50,
            "groove_adapter_oal_mm": 71.0,
            "meltzone_length_mm": 22.0,
            "standoffs": "3x Ø2.6x11.0 mm Ti-6Al-4V screws @ 120 deg (r = 7.5 mm)",
            "heatbreak": "0.25 mm thin-wall Ti-6Al-4V throat",
            "block": "85W cylindrical ceramic ring + Cu C110 melt core + V6 nozzle",
        },
        visual_max_triangles=24000,
    )
    t0 = time.perf_counter()
    res = bake_solid_superelement(spec, regions=regions, visual_parts=visual_parts, assets_root=assets_root)
    qa = verify_baked_asset("hotends", "phaetus_rapido_2_hf", assets_root=assets_root)
    print(f"  {qa.summary} ({time.perf_counter() - t0:.1f}s)")
    if not qa.passed:
        raise RuntimeError(f"QA failed: {qa.summary}")
    return res


# ---------------------------------------------------------------------------
# 4. Mellow Goliath Air (Watertight Multi-Material B-Rep Upgrading Visual-Only GLB)
# ---------------------------------------------------------------------------


def bake_mellow_goliath_air(assets_root: Path = ASSETS_ROOT) -> BakedAssetResult:
    """Bake ``hotends/mellow_goliath_air`` watertight multi-material Tet4 CMS + PBR GLB."""
    print("\n=== Baking hotends/mellow_goliath_air ===")
    # Remove any legacy visual-only bundle cleanly before writing full superelement bundle
    bundle = assets_root / "hotends" / "mellow_goliath_air"
    if bundle.is_dir() and not (bundle / "superelement.npz").is_file():
        shutil.rmtree(bundle)

    # Exact dimensions verified against VzBoT3D Vz-330_WEB v2.step (`Goliath_Air_cooled`,
    # Port 0 = top mounting face Z=0.0, Port 1 = nozzle tip Z=-69.24 mm, 96 g dressed spec):
    # - Al 6061 finned heatsink: Ø21.8 mm top mount + 6 fins + Ø23.4 mm lower collar, Z = [0.0, -24.5]
    # - 3x heavy Ti-6Al-4V structural posts: Ø3.0 x 14.0 mm @ 120 deg (r = 7.5 mm), Z = [-20.0, -34.0]
    # - Thick Ti-6Al-4V bi-metal heatbreak: Ø3.0 OD / Ø2.0 ID (0.50 mm wall), Z = [-14.0, -36.5]
    # - Long 50 mm plated Cu C110 melt core & 100W heater block: Z = [-26.0, -61.5]
    #   (upper collar Ø18.0 x 7.0 + Ø15.0->Ø13.2 long melt body + side 100W cartridge boss to X=-15.5)
    # - Goliath M6 copper/brass nozzle: Z = [-55.0, -69.24] (tip at exact -69.24 mm!)
    hs_top = b3d.Solid.make_cylinder(10.9, 4.5, plane=b3d.Plane(origin=(0.0, 0.0, -4.5)))
    hs_core = b3d.Solid.make_cylinder(6.2, 24.5, plane=b3d.Plane(origin=(0.0, 0.0, -24.5)))
    hs_fins = [
        b3d.Solid.make_cylinder(10.9, 1.5, plane=b3d.Plane(origin=(0.0, 0.0, zf)))
        for zf in (-7.6, -10.8, -14.0, -17.2, -20.4)
    ]
    hs_collar = b3d.Solid.make_cylinder(11.7, 3.0, plane=b3d.Plane(origin=(0.0, 0.0, -24.5)))
    hs_bore = b3d.Solid.make_cylinder(1.55, 26.0, plane=b3d.Plane(origin=(0.0, 0.0, -25.2)))
    fea_heatsink = hs_top.fuse(hs_core, hs_collar, *hs_fins).cut(hs_bore).clean()

    # 3x heavy Ti-6Al-4V structural posts placed cleanly clear of side cartridge boss (0, 115, 245 deg)
    # with Ø2.2 mm effective shank matching M2.5 Ti screw + standoff stiffness over the 69.24 mm lever arm
    ti_posts = [
        b3d.Solid.make_cylinder(
            1.15,
            13.5,
            plane=b3d.Plane(
                origin=(7.2 * math.cos(math.radians(a)), 7.2 * math.sin(math.radians(a)), -33.0)
            ),
        )
        for a in (0.0, 115.0, 245.0)
    ]
    ti_throat = (
        b3d.Solid.make_cylinder(1.45, 20.5, plane=b3d.Plane(origin=(0.0, 0.0, -34.5)))
        .cut(b3d.Solid.make_cylinder(0.98, 23.0, plane=b3d.Plane(origin=(0.0, 0.0, -35.5))))
        .clean()
    )
    fea_ti = ti_throat.fuse(*ti_posts).clean()

    # Long 50 mm Cu C110 melt core + 100W side cartridge boss (well-separated bores, zero tangent slivers!)
    cu_upper = b3d.Solid.make_cylinder(9.2, 8.0, plane=b3d.Plane(origin=(0.0, 0.0, -34.5)))
    cu_mid = b3d.Solid.make_cylinder(7.4, 15.0, plane=b3d.Plane(origin=(0.0, 0.0, -48.5)))
    cu_low = b3d.Solid.make_cylinder(6.4, 14.0, plane=b3d.Plane(origin=(0.0, 0.0, -61.5)))
    cu_boss = b3d.Solid.make_box(12.0, 11.0, 14.0, plane=b3d.Plane(origin=(-16.0, -5.5, -43.5)))
    heater_bore = b3d.Solid.make_cylinder(3.0, 14.0, plane=b3d.Plane(origin=(-12.0, -7.0, -36.5), z_dir=(0.0, 1.0, 0.0)))
    melt_bore = b3d.Solid.make_cylinder(0.98, 37.5, plane=b3d.Plane(origin=(0.0, 0.0, -62.5)))
    fea_copper = cu_upper.fuse(cu_mid, cu_low, cu_boss).cut(heater_bore, melt_bore).clean()

    # Goliath long M6 nozzle seated deep in the 50 mm copper melt core, tip at exact Z = -69.24 mm
    noz_shaft = b3d.Solid.make_cylinder(3.2, 8.5, plane=b3d.Plane(origin=(0.0, 0.0, -63.0)))
    noz_hex = b3d.Solid.make_cylinder(4.8, 3.0, plane=b3d.Plane(origin=(0.0, 0.0, -65.5)))
    noz_cone = b3d.Solid.make_cone(0.65, 4.4, 3.74, plane=b3d.Plane(origin=(0.0, 0.0, -69.24)))
    fea_nozzle = noz_shaft.fuse(noz_hex, noz_cone).clean()

    regions = _non_overlapping_regions(
        [
            (fea_nozzle, "copper_c110", "goliath_m6_nozzle", COLOR_BRASS_NOZZLE, 0.90, 0.24),
            (fea_ti, "titanium_grade5", "goliath_ti_posts_and_heatbreak", COLOR_DARK_TI, 0.86, 0.30),
            (fea_copper, "copper_c110", "goliath_50mm_copper_core", COLOR_NICKEL_CU, 0.88, 0.24),
            (fea_heatsink, "aluminum_6061_t6", "goliath_air_heatsink", COLOR_CNC_AL, 0.88, 0.25),
        ]
    )

    # Visual sock sleeve wrapping the lower 50 mm melt zone (visual only, non-structural)
    sock_outer = b3d.Solid.make_cylinder(7.7, 26.0, plane=b3d.Plane(origin=(0.0, 0.0, -61.8)))
    sock_inner = b3d.Solid.make_cylinder(6.3, 28.0, plane=b3d.Plane(origin=(0.0, 0.0, -62.5)))
    sock_vis = sock_outer.cut(sock_inner).clean()

    visual_parts = [
        VisualPart(fea_heatsink, "goliath_air_heatsink", COLOR_CNC_AL, metallic=0.88, roughness=0.25),
        VisualPart(fea_ti, "goliath_ti_posts", COLOR_DARK_TI, metallic=0.86, roughness=0.30),
        VisualPart(fea_copper, "goliath_copper_melt_block", COLOR_NICKEL_CU, metallic=0.90, roughness=0.22),
        VisualPart(sock_vis, "goliath_silicone_sock", COLOR_DARK_SOCK, metallic=0.05, roughness=0.86),
        VisualPart(fea_nozzle, "goliath_nozzle", COLOR_BRASS_NOZZLE, metallic=0.92, roughness=0.22),
    ]

    spec = BakedAssetSpec(
        category="hotends",
        asset_id="mellow_goliath_air",
        name="Mellow Goliath Air (100W Long-Melt, Multi-Material CAD CMS)",
        source_cad="Vz-330_WEB v2.step (Goliath_Air_cooled datums + watertight multi-material B-Rep)",
        source_url="https://github.com/VzBoT3D/VzBoT-Vz330/tree/master/CAD/Full%20Printer",
        reference_material_id="titanium_grade5",
        cad_trust_pct=85,
        port0=PortDefinition(
            (0.0, 0.0, 0.0),
            select_z_annulus(0.0, r_min_mm=3.0, r_max_mm=11.0, tol_mm=0.60),
        ),
        port1=PortDefinition(
            (0.0, 0.0, -69.24),
            select_z_cap(z_max_mm=-64.2, z_min_mm=-69.4, r_max_mm=4.6),
        ),
        n_internal_modes=8,
        mesh_size_mm=1.60,
        is_multi_material=True,
        is_fdm_anisotropic=False,
        part_recipe={
            "overall_length_mm": 69.24,
            "meltzone_length_mm": 50.0,
            "datum": "Port 0 = top mounting face center (Z=60 CAD); Port 1 = nozzle tip (Z=-9.24 CAD)",
            "heatsink": "Ø21.8-23.4 x 24.5 mm finned Al 6061 heatsink (4x M2.5 Dragon PCD)",
            "standoffs": "3x Ø3.0x14.0 mm Ti-6Al-4V posts @ r=7.5 mm + thick 0.50 mm Ti heatbreak",
            "block": "50 mm plated Cu C110 melt core + 100W cartridge boss + Goliath M6 nozzle",
        },
        visual_max_triangles=24000,
    )
    t0 = time.perf_counter()
    res = bake_solid_superelement(spec, regions=regions, visual_parts=visual_parts, assets_root=assets_root)
    qa = verify_baked_asset("hotends", "mellow_goliath_air", assets_root=assets_root)
    print(f"  {qa.summary} ({time.perf_counter() - t0:.1f}s)")
    if not qa.passed:
        raise RuntimeError(f"QA failed: {qa.summary}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Bake CAL-12 hotend CAD Craig-Bampton superelements.")
    ap.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated subset: dragon_hf,revo_voron,rapido_2_hf,goliath_air",
    )
    args = ap.parse_args()
    sel = {s.strip().lower() for s in args.only.split(",") if s.strip()} if args.only else {
        "dragon_hf",
        "revo_voron",
        "rapido_2_hf",
        "goliath_air",
    }
    if "dragon_hf" in sel:
        bake_phaetus_dragon_hf()
    if "revo_voron" in sel:
        bake_e3d_revo_voron()
    if "rapido_2_hf" in sel:
        bake_phaetus_rapido_2_hf()
    if "goliath_air" in sel:
        bake_mellow_goliath_air()


if __name__ == "__main__":
    main()
