"""Bake all Phase 1 CAD-backed hardware superelement assets into ``src/printsim/assets/hardware/``.

Phase 1 Hardware Assets:
1. ``hotends/chube_compact``       (Luke's Lab Chube Compact — 47.2 mm OAL, Ø22.23 mm Ti monocoque)
2. ``hotends/chube_air``           (Luke's Lab Chube Air — 66.0 mm OAL, Ø25.4 mm Ti shell w/ side airflow windows)
3. ``hotends/chube_conduction``    (Luke's Lab Chube Conduction — 66.0 mm OAL, Ø25.4 mm solid-top Ti shell)
4. ``hotends/tricorn_short``       (Okano Tricorn Short — 54.0 mm OAL, multi-body CAD STEP)
5. ``hotends/tricorn_long``        (Okano Tricorn Long — 66.0 mm OAL, multi-body CAD STEP)
6. ``toolheads/microwaved_bowden`` (Microwaved Bowden Toolhead — SLM Ti-6Al-4V MBTCFD chassis)
7. ``assemblies/microwaved_bowden__tricorn_long`` (Coupled Tricorn Long + Microwaved Ti-6Al-4V bottom brace)
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import build123d as b3d
import numpy as np
from rich.console import Console
from rich.table import Table

from printsim.cad_assets import (
    ASSETS_ROOT,
    BakedAssetResult,
    BakedAssetSpec,
    MaterialRegion,
    PortDefinition,
    VisualPart,
    bake_solid_superelement,
    build_visual_glb,
    clear_asset_cache,
    select_plane_box,
    select_z_annulus,
    select_z_cap,
)
from printsim.cad_assets.baker import (
    _assemble_tet4_km,
    _compute_mass_com_inertia,
    _mesh_regions_tet4,
    _verify_and_extract_tip_stiffness,
)
from printsim.fea.superelement import condense_solid_patch_to_vp, virtual_point_transform
from printsim.schema import get_material

DEFAULT_CAD_DIR = Path(
    "/Users/jumpybeetroot/.gemini/antigravity/brain/6dd189c6-6c5d-4f1b-940e-62807fb9d046/scratch/cad"
)

# Authentic PBR material palette
COLOR_SATIN_TI = "#b8c2ce"
COLOR_CHUBE_TI = "#757c86"
COLOR_DARK_TI = "#8e99a8"
COLOR_CNC_AL = "#dfe6ef"
COLOR_COPPER = "#d97742"
COLOR_NICKEL_CU = "#cdc6ba"
COLOR_BRASS_NOZZLE = "#eab308"
COLOR_DARK_SOCK = "#232a36"
COLOR_TI_SHIELD = "#64748b"
COLOR_STAINLESS = "#d8e0ea"
COLOR_CERAMIC = "#e5e7eb"
COLOR_PCB_DARK = "#1e293b"


def _bake_or_refresh_visuals(
    spec: BakedAssetSpec,
    regions: list[MaterialRegion],
    visual_parts: list[VisualPart],
    assets_root: Path | None = None,
    visuals_only: bool = False,
) -> BakedAssetResult:
    """Run full bake or fast ``--visuals-only`` GLB + manifest refresh."""
    if not visuals_only:
        return bake_solid_superelement(
            spec, regions, visual_parts=visual_parts, assets_root=assets_root
        )

    root = assets_root if assets_root is not None else ASSETS_ROOT
    asset_dir = root / spec.category / spec.asset_id
    manifest_path = asset_dir / "manifest.json"
    npz_path = asset_dir / "superelement.npz"
    if not manifest_path.is_file() or not npz_path.is_file():
        raise FileNotFoundError(
            f"--visuals-only requires existing manifest.json and superelement.npz in {asset_dir}"
        )

    vis_tris, vis_bytes = build_visual_glb(
        visual_parts,
        asset_dir / "visual.glb",
        max_total_triangles=spec.visual_max_triangles,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["visual_triangles"] = int(vis_tris)
    manifest["visual_bytes"] = int(vis_bytes)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    clear_asset_cache()

    with np.load(npz_path) as data:
        return BakedAssetResult(
            category=spec.category,
            asset_id=spec.asset_id,
            asset_dir=asset_dir,
            manifest=manifest,
            K_cb=np.asarray(data["K_cb"], dtype=np.float64),
            M_cb=np.asarray(data["M_cb"], dtype=np.float64),
            K_bb=np.asarray(data["K_bb"], dtype=np.float64),
            port_coords_m=np.asarray(data["port_coords_m"], dtype=np.float64),
            com_m=np.asarray(data["com_m"], dtype=np.float64),
            inertia_kg_m2=np.asarray(data["inertia_kg_m2"], dtype=np.float64),
        )


def _non_overlapping_regions(
    ordered_parts: list[tuple[Any, str, str, str, float, float]],
) -> list[MaterialRegion]:
    """Cut higher-priority solids out of lower-priority ones so Gmsh fragment has 0 overlap."""
    regions: list[MaterialRegion] = []
    occupied_union: Any = None
    for solid, mat_id, label, color_hex, metallic, roughness in ordered_parts:
        cur = solid
        if occupied_union is not None:
            try:
                cur = cur.cut(occupied_union)
            except Exception:
                pass
        solids = cur.solids() if hasattr(cur, "solids") else [cur]
        valid_solids = [s for s in solids if s.is_valid and float(s.volume) > 1e-4]
        if not valid_solids:
            continue
        clean_shape = valid_solids[0] if len(valid_solids) == 1 else b3d.Compound(children=valid_solids)
        regions.append(
            MaterialRegion(
                solid=clean_shape,
                material_id=mat_id,
                label=label,
                fdm_anisotropic=False,
                color_hex=color_hex,
                metallic=metallic,
                roughness=roughness,
            )
        )
        for vs in valid_solids:
            occupied_union = vs if occupied_union is None else occupied_union.fuse(vs)
    return regions


def _make_v6_nozzle_solid(tip_z_mm: float, total_len_mm: float = 12.5) -> Any:
    """Build a standard E3D V6 brass/copper nozzle solid pointing downward to ``tip_z_mm``."""
    # Shaft seated in block (top half), 7 mm hex collar (2.5 mm), tapered cone (3.5 mm)
    shaft_len = max(total_len_mm - 6.0, 4.0)
    z_collar_top = tip_z_mm + 6.0
    shaft = b3d.Solid.make_cylinder(
        3.0, shaft_len, plane=b3d.Plane(origin=(0.0, 0.0, z_collar_top))
    )
    collar = b3d.Solid.make_cylinder(
        3.75, 2.5, plane=b3d.Plane(origin=(0.0, 0.0, tip_z_mm + 3.5))
    )
    cone = b3d.Solid.make_cone(
        0.65, 3.2, 3.5, plane=b3d.Plane(origin=(0.0, 0.0, tip_z_mm))
    )
    return shaft.fuse(collar, cone).clean()


def _make_m3_pin(
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    radius_mm: float = 1.6,
    head_radius_mm: float | None = None,
    head_len_mm: float = 2.4,
) -> Any:
    """Build a solid cylindrical fastener bridge (with optional head collar at ``p0``) between ``p0`` and ``p1``."""
    v = np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)
    length = float(np.linalg.norm(v))
    d = v / length
    ref = np.array([1.0, 0.0, 0.0]) if abs(d[2]) > 0.99 else np.array([0.0, 0.0, 1.0])
    x_dir = np.cross(ref, d)
    x_dir /= np.linalg.norm(x_dir)
    plane = b3d.Plane(
        origin=tuple(float(c) for c in p0),
        x_dir=tuple(float(c) for c in x_dir),
        z_dir=tuple(float(c) for c in d),
    )
    shaft = b3d.Solid.make_cylinder(radius_mm, length, plane=plane)
    if head_radius_mm is not None and head_radius_mm > radius_mm:
        head = b3d.Solid.make_cylinder(head_radius_mm, min(head_len_mm, length), plane=plane)
        return shaft.fuse(head).clean()
    return shaft


# ---------------------------------------------------------------------------
# 1. Chube Compact
# ---------------------------------------------------------------------------


def bake_chube_compact(
    cad_dir: Path,
    assets_root: Path | None = None,
    visuals_only: bool = False,
) -> BakedAssetResult:
    """Bake Luke's Lab Chube Compact (hollow Ti monocoque FEA + official STEP visual.glb)."""
    step_path = cad_dir / "chube_official/extracted/chube_compact.step"
    asm = b3d.import_step(str(step_path))
    solids = asm.solids()

    regions: list[MaterialRegion] = []
    if not visuals_only:
        # True hollow Ti-6Al-4V monocoque structural shell (Ø22.23 mm OD x 0.89 mm wall, 47.2 mm OAL)
        ro = 11.115
        ri = 10.225  # 0.89 mm wall
        body_h = 41.68

        outer_tube = b3d.Solid.make_cylinder(ro, body_h, plane=b3d.Plane(origin=(0.0, 0.0, -body_h)))
        inner_bore = b3d.Solid.make_cylinder(ri, body_h - 6.8, plane=b3d.Plane(origin=(0.0, 0.0, -body_h + 3.6)))
        shell = outer_tube.cut(inner_bore)

        # Central filament/heatbreak bore (Ø5.5 mm) in top flange & bottom nozzle bore (Ø6.0 mm)
        top_bore = b3d.Solid.make_cylinder(2.75, 8.0, plane=b3d.Plane(origin=(0.0, 0.0, -5.0)))
        bot_bore = b3d.Solid.make_cylinder(3.0, 8.0, plane=b3d.Plane(origin=(0.0, 0.0, -body_h - 2.0)))
        shell = shell.cut(top_bore, bot_bore)

        # 4x M3 Dragon PCD Ø16 mm mounting holes in top flange (45, 135, 225, 315 deg)
        m3_holes = [
            b3d.Solid.make_cylinder(
                1.45,
                8.0,
                plane=b3d.Plane(
                    origin=(8.0 * math.cos(math.radians(a)), 8.0 * math.sin(math.radians(a)), -5.5)
                ),
            )
            for a in (45.0, 135.0, 225.0, 315.0)
        ]
        shell = shell.cut(*m3_holes)

        # 4 cooling cutout windows in upper cold section (z in [-22.5, -7.5]) along 45/135/225/315 deg
        win_solids = []
        for a in (45.0, 135.0, 225.0, 315.0):
            ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
            pln = b3d.Plane(origin=(10.0 * ca, 10.0 * sa, -15.0), x_dir=(ca, sa, 0.0), z_dir=(0.0, 0.0, 1.0))
            box = b3d.Solid.make_box(8.0, 9.0, 14.5, plane=pln).translate((-4.0 * ca + 4.5 * sa, -4.0 * sa - 4.5 * ca, -7.25))
            win_solids.append(box)
        shell = shell.cut(*win_solids)

        # Lower heater/wire access slot
        wire_slot = b3d.Solid.make_box(12.0, 14.0, 10.0, plane=b3d.Plane(origin=(-6.0, -14.0, -34.0)))
        shell = shell.cut(wire_slot)

        # Internal Chalice Ti heatbreak connecting top flange to melt core
        chalice = b3d.Solid.make_cylinder(
            1.75, 11.5, plane=b3d.Plane(origin=(0.0, 0.0, -14.0))
        ).cut(b3d.Solid.make_cylinder(1.0, 14.0, plane=b3d.Plane(origin=(0.0, 0.0, -15.0))))
        ti_shell = shell.fuse(chalice).clean()

        # Internal Copper C110 melt core (33 mm meltzone, seated inside bottom collar)
        melt_upper = b3d.Solid.make_cylinder(4.8, 25.5, plane=b3d.Plane(origin=(0.0, 0.0, -38.5)))
        melt_lower = b3d.Solid.make_cylinder(7.4, 16.5, plane=b3d.Plane(origin=(0.0, 0.0, -40.0)))
        copper_core = melt_upper.fuse(melt_lower).cut(
            b3d.Solid.make_cylinder(1.0, 32.0, plane=b3d.Plane(origin=(0.0, 0.0, -41.0)))
        ).clean()

        nozzle = _make_v6_nozzle_solid(tip_z_mm=-47.20, total_len_mm=12.5)

        regions = _non_overlapping_regions(
            [
                (nozzle, "copper_c110", "v6_nozzle", COLOR_BRASS_NOZZLE, 0.90, 0.25),
                (ti_shell, "titanium_grade5", "ti_monocoque_shell", COLOR_SATIN_TI, 0.85, 0.32),
                (copper_core, "copper_c110", "copper_melt_core", COLOR_COPPER, 0.88, 0.28),
            ]
        )

    # Photo-accurate Chube Compact visual assembly (gunmetal Ti shell w/ 4 round cooling windows,
    # 2-fin bright CNC Al heatsink, Chalice Ti break, nickel-plated Cu melt core + official STEP sock/nozzle/screws)
    vis_ro, vis_ri, vis_h = 11.115, 10.225, 41.68
    vis_shell = b3d.Solid.make_cylinder(vis_ro, vis_h, plane=b3d.Plane(origin=(0.0, 0.0, -vis_h))).cut(
        b3d.Solid.make_cylinder(vis_ri, vis_h + 2.0, plane=b3d.Plane(origin=(0.0, 0.0, -vis_h - 1.0)))
    )
    # 4 crisp round cooling windows at 0/90/180/270 deg in upper heatsink zone (z = -12.5 mm)
    comp_wins = []
    for a in (0.0, 90.0, 180.0, 270.0):
        ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
        pln = b3d.Plane(origin=(0.0, 0.0, -12.5), z_dir=(ca, sa, 0.0))
        comp_wins.append(b3d.Solid.make_cylinder(5.2, 16.0, plane=pln))
    vis_shell = vis_shell.cut(*comp_wins).clean()

    # Bright CNC aluminum 2-fin upper heatsink w/ 4x M3 Dragon PCD Ø16 mm top face
    comp_m3 = [
        b3d.Solid.make_cylinder(
            1.45,
            6.0,
            plane=b3d.Plane(
                origin=(8.0 * math.cos(math.radians(a)), 8.0 * math.sin(math.radians(a)), -5.0)
            ),
        )
        for a in (45.0, 135.0, 225.0, 315.0)
    ]
    comp_hs_top = b3d.Solid.make_cylinder(vis_ri - 0.05, 3.6, plane=b3d.Plane(origin=(0.0, 0.0, -3.6)))
    comp_hs_core = b3d.Solid.make_cylinder(5.0, 18.5, plane=b3d.Plane(origin=(0.0, 0.0, -18.5)))
    comp_fin1 = b3d.Solid.make_cylinder(vis_ri - 0.1, 2.0, plane=b3d.Plane(origin=(0.0, 0.0, -9.8)))
    comp_fin2 = b3d.Solid.make_cylinder(vis_ri - 0.1, 2.0, plane=b3d.Plane(origin=(0.0, 0.0, -15.2)))
    comp_heatsink = (
        comp_hs_top.fuse(comp_hs_core, comp_fin1, comp_fin2)
        .cut(b3d.Solid.make_cylinder(2.75, 22.0, plane=b3d.Plane(origin=(0.0, 0.0, -20.0))), *comp_m3)
        .clean()
    )
    comp_chalice = b3d.Solid.make_cylinder(2.4, 8.0, plane=b3d.Plane(origin=(0.0, 0.0, -23.0))).cut(
        b3d.Solid.make_cylinder(1.0, 10.0, plane=b3d.Plane(origin=(0.0, 0.0, -24.0)))
    )
    comp_melt = b3d.Solid.make_cylinder(7.6, 17.5, plane=b3d.Plane(origin=(0.0, 0.0, -39.5))).cut(
        b3d.Solid.make_cylinder(1.0, 20.0, plane=b3d.Plane(origin=(0.0, 0.0, -40.5)))
    )

    visual_parts = [
        VisualPart(vis_shell, "chube_body", COLOR_CHUBE_TI, metallic=0.84, roughness=0.32),
        VisualPart(comp_heatsink, "compact_heatsink_fins", COLOR_CNC_AL, metallic=0.88, roughness=0.22),
        VisualPart(comp_chalice, "chalice_heatbreak", COLOR_DARK_TI, metallic=0.86, roughness=0.28),
        VisualPart(comp_melt, "nickel_melt_core", COLOR_NICKEL_CU, metallic=0.88, roughness=0.24),
        VisualPart(solids[0], "throat_bushing", COLOR_DARK_TI, metallic=0.85, roughness=0.32),
        VisualPart(solids[1], "silicone_sock", COLOR_DARK_SOCK, metallic=0.05, roughness=0.85),
        VisualPart(solids[3], "v6_nozzle", COLOR_BRASS_NOZZLE, metallic=0.92, roughness=0.22),
        VisualPart(solids[4], "thermistor_cartridge", COLOR_STAINLESS, metallic=0.88, roughness=0.26),
        VisualPart(solids[5], "heater_cartridge", COLOR_COPPER, metallic=0.88, roughness=0.24),
    ]
    for i, sc in enumerate(solids[6:]):
        visual_parts.append(VisualPart(sc, f"torx_screw_{i}", COLOR_STAINLESS, metallic=0.92, roughness=0.24))

    spec = BakedAssetSpec(
        category="hotends",
        asset_id="chube_compact",
        name="Luke's Lab Chube Compact (80W Ti Shell)",
        source_cad="chube_compact.step",
        source_url="https://www.chubehotends.com/",
        reference_material_id="titanium_grade5",
        cad_trust_pct=86,
        port0=PortDefinition(
            (0.0, 0.0, 0.0),
            select_z_annulus(0.0, r_min_mm=4.5, r_max_mm=11.2, tol_mm=0.45),
        ),
        port1=PortDefinition(
            (0.0, 0.0, -47.2),
            select_z_cap(z_max_mm=-43.2, z_min_mm=-47.3, r_max_mm=4.2),
        ),
        n_internal_modes=8,
        mesh_size_mm=1.25,
        is_multi_material=True,
        is_fdm_anisotropic=False,
        part_recipe={
            "overall_length_mm": 47.2,
            "outer_diameter_mm": 22.23,
            "wall_thickness_mm": 0.89,
            "mount_pcd_mm": 16.0,
            "meltzone_length_mm": 33.0,
        },
        visual_max_triangles=24000,
    )
    return _bake_or_refresh_visuals(
        spec, regions, visual_parts=visual_parts, assets_root=assets_root, visuals_only=visuals_only
    )


# ---------------------------------------------------------------------------
# 2 & 3. Chube Air & Chube Conduction (Rev A)
# ---------------------------------------------------------------------------


def _build_chube_reva_geometry(
    cad_dir: Path,
    variant: str,
    visuals_only: bool = False,
) -> tuple[list[MaterialRegion], list[VisualPart]]:
    """Build watertight Rev A structural shell (Air vs Conduction) + official STEP hardware visuals."""
    step_path = cad_dir / "chube_official/extracted/CHUBE_INTEGRATION_REVA.step"
    asm = b3d.import_step(str(step_path))

    # Align official STEP origin: axis center (-0.2634, -33.7764), top face z=282.2752 -> (0,0,0)
    shift = (0.2634, 33.7764, -282.2752)

    common = [c for c in asm.children if c.label == "Common_Features"][0]
    step_outer_shell = [c for c in common.children if c.label == "Outer_Shell"][0].translate(shift)
    step_nozzle = [c for c in common.children if "Nozzle" in c.label][0].translate(shift)
    step_shield = [c for c in common.children if "Sheild" in c.label or "Shield" in c.label][0].translate(shift)
    step_fasteners = [c for c in common.children if c.label == "Fasteners"][0].translate(shift)

    is_air = variant == "air"
    regions: list[MaterialRegion] = []

    if not visuals_only:
        ro = 12.70  # Ø25.4 mm OD
        ri = 11.80  # Ø23.6 mm ID (0.90 mm wall)
        body_h = 60.00  # bare shell length (66.0 mm OAL w/ 6 mm V6 nozzle)

        outer = b3d.Solid.make_cylinder(ro, body_h, plane=b3d.Plane(origin=(0.0, 0.0, -body_h)))
        # Air has 3.2 mm top flange; Conduction has 8.0 mm thick solid top conduction heatsink ("8mm BOTTOM OF HEATSINK")
        top_flange_t = 3.2 if is_air else 8.0
        bot_collar_t = 4.2
        cavity = b3d.Solid.make_cylinder(
            ri,
            body_h - top_flange_t - bot_collar_t,
            plane=b3d.Plane(origin=(0.0, 0.0, -body_h + bot_collar_t)),
        )
        shell = outer.cut(cavity)

        # Top center Chalice bore (Ø7.0 mm) & bottom nozzle clearance bore (Ø6.4 mm)
        top_bore = b3d.Solid.make_cylinder(3.5, top_flange_t + 2.0, plane=b3d.Plane(origin=(0.0, 0.0, -top_flange_t - 1.0)))
        bot_bore = b3d.Solid.make_cylinder(3.2, bot_collar_t + 2.0, plane=b3d.Plane(origin=(0.0, 0.0, -body_h - 1.0)))
        shell = shell.cut(top_bore, bot_bore)

        # 6x M3 mounting holes on Ø18.0 mm PCD (R = 9.0 mm)
        m3_holes = [
            b3d.Solid.make_cylinder(
                1.5,
                top_flange_t + 2.0,
                plane=b3d.Plane(
                    origin=(9.0 * math.cos(math.radians(a)), 9.0 * math.sin(math.radians(a)), -top_flange_t - 1.0)
                ),
            )
            for a in (0.0, 60.0, 120.0, 180.0, 240.0, 300.0)
        ]
        shell = shell.cut(*m3_holes)

        if is_air:
            air_wins = []
            for a in (30.0, 150.0, 210.0, 330.0):
                ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
                pln = b3d.Plane(
                    origin=(10.5 * ca, 10.5 * sa, -14.75),
                    x_dir=(ca, sa, 0.0),
                    z_dir=(0.0, 0.0, 1.0),
                )
                box = b3d.Solid.make_box(8.0, 8.5, 13.0, plane=pln).translate(
                    (-4.0 * ca + 4.25 * sa, -4.0 * sa - 4.25 * ca, -6.5)
                )
                air_wins.append(box)
            wire_cut = b3d.Solid.make_box(9.0, 12.0, 12.0, plane=b3d.Plane(origin=(-4.5, -14.5, -54.0)))
            shell = shell.cut(*air_wins, wire_cut)

            baffle = b3d.Solid.make_cylinder(
                ri + 0.1, 2.0, plane=b3d.Plane(origin=(0.0, 0.0, -22.5))
            ).cut(b3d.Solid.make_cylinder(4.8, 4.0, plane=b3d.Plane(origin=(0.0, 0.0, -23.5))))
            shell = shell.fuse(baffle)

            chalice = b3d.Solid.make_cylinder(
                1.75, 15.0, plane=b3d.Plane(origin=(0.0, 0.0, -18.2))
            ).cut(b3d.Solid.make_cylinder(1.0, 18.0, plane=b3d.Plane(origin=(0.0, 0.0, -19.5))))
            ti_shell = shell.fuse(chalice).clean()

            melt_top = b3d.Solid.make_cylinder(4.5, 22.0, plane=b3d.Plane(origin=(0.0, 0.0, -40.0)))
            melt_bot = b3d.Solid.make_cylinder(7.4, 22.0, plane=b3d.Plane(origin=(0.0, 0.0, -60.0)))
            heater_bore = b3d.Solid.make_cylinder(3.0, 22.0, plane=b3d.Plane(origin=(-4.6, 0.0, -60.5)))
            therm_bore = b3d.Solid.make_cylinder(1.5, 16.0, plane=b3d.Plane(origin=(4.6, 0.0, -60.5)))
            pin1 = _make_m3_pin((-12.6, 0.0, -52.0), (12.6, 0.0, -52.0), radius_mm=1.6)
            pin2 = _make_m3_pin((0.0, -12.6, -52.0), (0.0, 12.6, -52.0), radius_mm=1.6)
            copper_core = (
                melt_top.fuse(melt_bot, pin1, pin2)
                .cut(
                    b3d.Solid.make_cylinder(1.0, 46.0, plane=b3d.Plane(origin=(0.0, 0.0, -61.0))),
                    heater_bore,
                    therm_bore,
                )
                .clean()
            )
        else:
            wire_cut = b3d.Solid.make_box(7.0, 10.0, 10.0, plane=b3d.Plane(origin=(-3.5, -14.5, -54.0)))
            shell = shell.cut(wire_cut)

            chalice = b3d.Solid.make_cylinder(
                1.8, 8.5, plane=b3d.Plane(origin=(0.0, 0.0, -12.0))
            ).cut(b3d.Solid.make_cylinder(1.0, 12.0, plane=b3d.Plane(origin=(0.0, 0.0, -13.5))))
            ti_shell = shell.fuse(chalice).clean()

            melt_top = b3d.Solid.make_cylinder(4.8, 26.0, plane=b3d.Plane(origin=(0.0, 0.0, -34.0)))
            melt_bot = b3d.Solid.make_cylinder(7.3, 28.0, plane=b3d.Plane(origin=(0.0, 0.0, -60.0)))
            heater_bore = b3d.Solid.make_cylinder(3.0, 48.0, plane=b3d.Plane(origin=(-4.6, 0.0, -60.5)))
            therm_bore = b3d.Solid.make_cylinder(1.5, 16.0, plane=b3d.Plane(origin=(4.6, 0.0, -60.5)))
            pin1 = _make_m3_pin((-12.6, 0.0, -52.0), (12.6, 0.0, -52.0), radius_mm=1.6)
            pin2 = _make_m3_pin((0.0, -12.6, -52.0), (0.0, 12.6, -52.0), radius_mm=1.6)
            copper_core = (
                melt_top.fuse(melt_bot, pin1, pin2)
                .cut(
                    b3d.Solid.make_cylinder(1.0, 56.0, plane=b3d.Plane(origin=(0.0, 0.0, -61.0))),
                    heater_bore,
                    therm_bore,
                )
                .clean()
            )

        nozzle = _make_v6_nozzle_solid(tip_z_mm=-66.00, total_len_mm=12.5)
        regions = _non_overlapping_regions(
            [
                (nozzle, "copper_c110", "v6_nozzle", COLOR_BRASS_NOZZLE, 0.90, 0.25),
                (ti_shell, "titanium_grade5", "ti_monocoque_shell", COLOR_SATIN_TI, 0.85, 0.32),
                (copper_core, "copper_c110", "copper_melt_core", COLOR_COPPER, 0.88, 0.28),
            ]
        )

    # Photo-accurate real Chube visual.glb matching official Luke's Lab hardware photos & Rev A dimensions:
    # Thick-walled (0.90 mm) dark gunmetal-titanium tube w/ iconic circular/oval side cutouts,
    # bright CNC aluminum 3-fin heatsink (Air) or solid 8 mm conduction block (Conduction),
    # electroless-nickel-plated copper melt core + side heater clamp boss, Chalice Ti heatbreak,
    # official curved brushed stainless heat shield (SM000092), V6 nozzle, and 9x T10 Torx screws.
    v_ro, v_ri, v_h = 12.70, 11.80, 60.00
    vis_tube = b3d.Solid.make_cylinder(v_ro, v_h, plane=b3d.Plane(origin=(0.0, 0.0, -v_h))).cut(
        b3d.Solid.make_cylinder(v_ri, v_h + 2.0, plane=b3d.Plane(origin=(0.0, 0.0, -v_h - 1.0)))
    )

    def _radial_oval_cut(angle_deg: float, z_center: float, width_mm: float, height_mm: float) -> Any:
        ca, sa = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
        r = 0.5 * width_mm
        span = max(0.0, 0.5 * height_mm - r)
        p_top = b3d.Plane(origin=(0.0, 0.0, z_center + span), z_dir=(ca, sa, 0.0))
        p_bot = b3d.Plane(origin=(0.0, 0.0, z_center - span), z_dir=(ca, sa, 0.0))
        c1 = b3d.Solid.make_cylinder(r, 18.0, plane=p_top)
        if span <= 0.1:
            return c1
        c2 = b3d.Solid.make_cylinder(r, 18.0, plane=p_bot)
        pln_mid = b3d.Plane(origin=(0.0, 0.0, z_center), x_dir=(ca, sa, 0.0), z_dir=(0.0, 0.0, 1.0))
        mid_box = b3d.Solid.make_box(18.0, width_mm, 2.0 * span, plane=pln_mid).translate(
            (0.0 + r * sa, 0.0 - r * ca, -span)
        )
        return c1.fuse(c2, mid_box).clean()

    # Middle 3 large oval cutouts at 60/180/300 deg (rotated 60 deg from upper cooling windows, matching official photos)
    mid_cuts = [
        _radial_oval_cut(a, -33.0 if is_air else -34.5, width_mm=15.0, height_mm=17.5)
        for a in (60.0, 180.0, 300.0)
    ]
    if is_air:
        # Upper 3 large circular airflow cutouts at 0/120/240 deg exposing the 3-fin CNC Al heatsink
        top_cuts = [
            _radial_oval_cut(a, -12.5, width_mm=14.4, height_mm=15.2)
            for a in (0.0, 120.0, 240.0)
        ]
        vis_tube = vis_tube.cut(*top_cuts, *mid_cuts).clean()
    else:
        # Conduction: solid upper titanium wrap over the 8 mm CNC Al conduction block, 3 large middle oval windows
        vis_tube = vis_tube.cut(*mid_cuts).clean()

    m3_top_holes = [
        b3d.Solid.make_cylinder(
            1.5,
            8.5,
            plane=b3d.Plane(
                origin=(9.0 * math.cos(math.radians(a)), 9.0 * math.sin(math.radians(a)), -8.25)
            ),
        )
        for a in (0.0, 60.0, 120.0, 180.0, 240.0, 300.0)
    ]

    visual_parts: list[VisualPart] = [
        VisualPart(vis_tube, "ti_monocoque_shell", COLOR_CHUBE_TI, metallic=0.84, roughness=0.34),
        VisualPart(step_nozzle, "e3d_v6_nozzle", COLOR_BRASS_NOZZLE, metallic=0.92, roughness=0.22),
        VisualPart(step_shield, "heat_shield", COLOR_STAINLESS, metallic=0.90, roughness=0.22),
    ]

    if is_air:
        # Real 3-fin bright machined CNC aluminum heatsink (z = 0..-22.5 mm) visible through the 3 circular windows
        hs_top = b3d.Solid.make_cylinder(v_ri - 0.05, 4.0, plane=b3d.Plane(origin=(0.0, 0.0, -4.0)))
        hs_core = b3d.Solid.make_cylinder(5.8, 22.0, plane=b3d.Plane(origin=(0.0, 0.0, -22.2)))
        hs_fin1 = b3d.Solid.make_cylinder(v_ri - 0.1, 2.2, plane=b3d.Plane(origin=(0.0, 0.0, -9.2)))
        hs_fin2 = b3d.Solid.make_cylinder(v_ri - 0.1, 2.2, plane=b3d.Plane(origin=(0.0, 0.0, -14.6)))
        hs_fin3 = b3d.Solid.make_cylinder(v_ri - 0.1, 2.4, plane=b3d.Plane(origin=(0.0, 0.0, -20.2)))
        vis_air_heatsink = (
            hs_top.fuse(hs_core, hs_fin1, hs_fin2, hs_fin3)
            .cut(b3d.Solid.make_cylinder(2.75, 24.0, plane=b3d.Plane(origin=(0.0, 0.0, -23.0))), *m3_top_holes)
            .clean()
        )
        vis_chalice = (
            b3d.Solid.make_cylinder(2.75, 4.0, plane=b3d.Plane(origin=(0.0, 0.0, -18.5)))
            .fuse(b3d.Solid.make_cylinder(2.1, 9.0, plane=b3d.Plane(origin=(0.0, 0.0, -27.5))))
            .cut(b3d.Solid.make_cylinder(1.0, 18.0, plane=b3d.Plane(origin=(0.0, 0.0, -28.5))))
            .clean()
        )
        # Electroless-nickel-plated copper melt core + side heater clamp boss visible inside middle oval windows
        melt_stem = b3d.Solid.make_cylinder(4.8, 20.0, plane=b3d.Plane(origin=(0.0, 0.0, -42.0)))
        melt_lower = b3d.Solid.make_cylinder(7.8, 22.5, plane=b3d.Plane(origin=(0.0, 0.0, -59.5)))
        heater_boss = b3d.Solid.make_box(7.8, 10.5, 21.0, plane=b3d.Plane(origin=(-3.9, -11.3, -58.5)))
        vis_melt = (
            melt_stem.fuse(melt_lower, heater_boss)
            .cut(b3d.Solid.make_cylinder(1.0, 44.0, plane=b3d.Plane(origin=(0.0, 0.0, -60.5))))
            .clean()
        )
        vis_heater = b3d.Solid.make_cylinder(2.95, 21.0, plane=b3d.Plane(origin=(0.0, -7.2, -59.5)))
        vis_therm = b3d.Solid.make_cylinder(1.48, 16.0, plane=b3d.Plane(origin=(0.0, 6.8, -59.5)))
        visual_parts.extend(
            [
                VisualPart(vis_air_heatsink, "air_heatsink_fins", COLOR_CNC_AL, metallic=0.88, roughness=0.22),
                VisualPart(vis_chalice, "chalice_heatbreak", COLOR_DARK_TI, metallic=0.86, roughness=0.28),
                VisualPart(vis_melt, "copper_melt_core", COLOR_NICKEL_CU, metallic=0.88, roughness=0.24),
                VisualPart(vis_heater, "heater_cartridge", COLOR_STAINLESS, metallic=0.88, roughness=0.26),
                VisualPart(vis_therm, "thermistor_cartridge", COLOR_DARK_TI, metallic=0.85, roughness=0.30),
            ]
        )
    else:
        # Conduction: solid CNC aluminum upper heatsink (0..-8 mm w/ 6x M3 holes), Chalice hex,
        # and full 52 mm electroless-nickel-plated copper melt core + tall Supervolcano heater clamp boss
        vis_cond_heatsink = (
            b3d.Solid.make_cylinder(v_ri - 0.05, 8.0, plane=b3d.Plane(origin=(0.0, 0.0, -8.0)))
            .cut(
                b3d.Solid.make_cylinder(3.5, 9.0, plane=b3d.Plane(origin=(0.0, 0.0, -8.5))),
                *m3_top_holes,
            )
            .clean()
        )
        hex_pts = [
            (3.18 * math.cos(math.radians(60.0 * k)), 3.18 * math.sin(math.radians(60.0 * k)), -9.88)
            for k in range(6)
        ]
        hex_wire = b3d.Wire.make_polygon([b3d.Vector(*p) for p in hex_pts] + [b3d.Vector(*hex_pts[0])])
        chalice_hex = b3d.Solid.extrude(b3d.Face(hex_wire), (0.0, 0.0, 2.43))
        vis_chalice = (
            b3d.Solid.make_cylinder(2.75, 3.65, plane=b3d.Plane(origin=(0.0, 0.0, -3.70)))
            .fuse(
                b3d.Solid.make_cylinder(1.95, 4.0, plane=b3d.Plane(origin=(0.0, 0.0, -7.5))),
                chalice_hex,
            )
            .cut(b3d.Solid.make_cylinder(1.0, 11.0, plane=b3d.Plane(origin=(0.0, 0.0, -10.5))))
            .clean()
        )
        melt_stem = b3d.Solid.make_cylinder(5.5, 28.0, plane=b3d.Plane(origin=(0.0, 0.0, -36.5)))
        melt_lower = b3d.Solid.make_cylinder(7.8, 26.0, plane=b3d.Plane(origin=(0.0, 0.0, -59.5)))
        heater_boss = b3d.Solid.make_box(7.8, 10.5, 45.0, plane=b3d.Plane(origin=(-3.9, -11.3, -58.5)))
        vis_melt = (
            melt_stem.fuse(melt_lower, heater_boss)
            .cut(b3d.Solid.make_cylinder(1.0, 56.0, plane=b3d.Plane(origin=(0.0, 0.0, -60.5))))
            .clean()
        )
        vis_heater = b3d.Solid.make_cylinder(2.95, 46.5, plane=b3d.Plane(origin=(0.0, -7.2, -59.5)))
        vis_therm = b3d.Solid.make_cylinder(1.48, 16.0, plane=b3d.Plane(origin=(0.0, 6.8, -59.5)))
        visual_parts.extend(
            [
                VisualPart(vis_cond_heatsink, "conduction_heatsink", COLOR_CNC_AL, metallic=0.88, roughness=0.22),
                VisualPart(vis_chalice, "chalice_heatbreak", COLOR_DARK_TI, metallic=0.86, roughness=0.28),
                VisualPart(vis_melt, "copper_melt_core", COLOR_NICKEL_CU, metallic=0.88, roughness=0.24),
                VisualPart(vis_heater, "supervolcano_heater", COLOR_STAINLESS, metallic=0.88, roughness=0.26),
                VisualPart(vis_therm, "thermistor_cartridge", COLOR_DARK_TI, metallic=0.85, roughness=0.30),
            ]
        )

    for i, f_solid in enumerate(step_fasteners.solids()):
        visual_parts.append(VisualPart(f_solid, f"torx_screw_{i}", COLOR_STAINLESS, metallic=0.92, roughness=0.24))

    return regions, visual_parts


def bake_chube_air(
    cad_dir: Path,
    assets_root: Path | None = None,
    visuals_only: bool = False,
) -> BakedAssetResult:
    regions, visual_parts = _build_chube_reva_geometry(cad_dir, "air", visuals_only=visuals_only)
    spec = BakedAssetSpec(
        category="hotends",
        asset_id="chube_air",
        name="Luke's Lab Chube Air (100W Ti Shell)",
        source_cad="CHUBE_INTEGRATION_REVA.step",
        source_url="https://www.chubehotends.com/",
        reference_material_id="titanium_grade5",
        cad_trust_pct=86,
        port0=PortDefinition(
            (0.0, 0.0, 0.0),
            select_z_annulus(0.0, r_min_mm=6.2, r_max_mm=11.0, tol_mm=0.45),
        ),
        port1=PortDefinition(
            (0.0, 0.0, -66.0),
            select_z_cap(z_max_mm=-61.8, z_min_mm=-66.1, r_max_mm=4.2),
        ),
        n_internal_modes=8,
        mesh_size_mm=1.35,
        is_multi_material=True,
        is_fdm_anisotropic=False,
        part_recipe={
            "variant": "air",
            "overall_length_mm": 66.0,
            "outer_diameter_mm": 25.4,
            "wall_thickness_mm": 0.90,
            "mount_pcd_mm": 18.0,
            "mount_bolts": "6x M3",
            "meltzone_length_mm": 42.0,
        },
        visual_max_triangles=24000,
    )
    return _bake_or_refresh_visuals(
        spec, regions, visual_parts=visual_parts, assets_root=assets_root, visuals_only=visuals_only
    )


def bake_chube_conduction(
    cad_dir: Path,
    assets_root: Path | None = None,
    visuals_only: bool = False,
) -> BakedAssetResult:
    regions, visual_parts = _build_chube_reva_geometry(
        cad_dir, "conduction", visuals_only=visuals_only
    )
    spec = BakedAssetSpec(
        category="hotends",
        asset_id="chube_conduction",
        name="Luke's Lab Chube Conduction (120W Ti Shell)",
        source_cad="CHUBE_INTEGRATION_REVA.step",
        source_url="https://www.chubehotends.com/",
        reference_material_id="titanium_grade5",
        cad_trust_pct=86,
        port0=PortDefinition(
            (0.0, 0.0, 0.0),
            select_z_annulus(0.0, r_min_mm=6.2, r_max_mm=11.0, tol_mm=0.45),
        ),
        port1=PortDefinition(
            (0.0, 0.0, -66.0),
            select_z_cap(z_max_mm=-61.8, z_min_mm=-66.1, r_max_mm=4.2),
        ),
        n_internal_modes=8,
        mesh_size_mm=1.35,
        is_multi_material=True,
        is_fdm_anisotropic=False,
        part_recipe={
            "variant": "conduction",
            "overall_length_mm": 66.0,
            "outer_diameter_mm": 25.4,
            "wall_thickness_mm": 0.90,
            "mount_pcd_mm": 18.0,
            "mount_bolts": "6x M3",
            "meltzone_length_mm": 52.0,
        },
        visual_max_triangles=24000,
    )
    return _bake_or_refresh_visuals(
        spec, regions, visual_parts=visual_parts, assets_root=assets_root, visuals_only=visuals_only
    )


# ---------------------------------------------------------------------------
# 4 & 5. Tricorn Short & Tricorn Long (Official STEP Multi-Body CAD)
# ---------------------------------------------------------------------------


def _bake_tricorn_variant(
    cad_dir: Path,
    variant: str,
    assets_root: Path | None = None,
    visuals_only: bool = False,
) -> BakedAssetResult:
    step_path = cad_dir / "tricorn_hotends.step"
    th = b3d.import_step(str(step_path))

    if variant == "short":
        short_asm = [c for c in th.children if c.label == "Short_Assembly_<1>"][0]
        children = list(short_asm.children)
        shift = (-27.0430, -0.0942, -61.0145)
        oal_mm = 54.0
        meltzone_mm = 41.0
        asset_id = "tricorn_short"
        name = "Tricorn Short (120W 3x40W Inverted Ti)"
        trust = 87
    else:
        children = [
            c for c in th.children
            if -13.0 <= 0.5 * (c.bounding_box().min.X + c.bounding_box().max.X) <= 13.0
        ]
        shift = (0.0, 0.0, -61.0000)
        oal_mm = 66.0
        meltzone_mm = 53.5
        asset_id = "tricorn_long"
        name = "Tricorn Long (120W 3x40W Inverted Ti)"
        trust = 88

    by_lbl: dict[str, list[Any]] = {}
    for ch in children:
        by_lbl.setdefault(ch.label, []).append(ch.translate(shift))

    heatsink = by_lbl["heatsink"][0]
    crown = by_lbl["crown"][0]
    heatbreak = by_lbl["heatbreak"][0]
    hotblock = by_lbl["hotblock"][0]
    nozzle = by_lbl["nozzle"][0]
    bucket = by_lbl["bucket"][0]
    heaters = by_lbl.get("heater", [])
    screws = by_lbl.get("Hex_socket_countersunk_head_screw_M3x0_5_x_6", [])
    set_screws = by_lbl.get("Cup_point_hex_socket_set_screw_M4x0_7_x_4", [])

    regions: list[MaterialRegion] = []
    if not visuals_only:
        crown_screws = []
        for sc in screws:
            bb = sc.bounding_box()
            cz = 0.5 * (bb.min.Z + bb.max.Z)
            if cz < -32.0:  # exclude bottom bucket screw
                continue
            c = bb.center()
            s_sc = sc.translate((-c.X, -c.Y, -c.Z)).scale(1.06).translate((c.X, c.Y, c.Z))
            crown_screws.append(s_sc)

        ti_struct = b3d.Compound(children=[crown, heatbreak])
        screw_comp = b3d.Compound(children=crown_screws)

        # First-region-wins priority in Gmsh occ.fragment: nozzle > Ti crown/heatbreak > M3 screws > heatsink > hotblock
        regions = [
            MaterialRegion(nozzle, "copper_c110", "v6_nozzle", fdm_anisotropic=False, color_hex=COLOR_BRASS_NOZZLE),
            MaterialRegion(ti_struct, "titanium_grade5", "ti_crown_heatbreak", fdm_anisotropic=False, color_hex=COLOR_SATIN_TI),
            MaterialRegion(screw_comp, "stainless_304", "m3_crown_screws", fdm_anisotropic=False, color_hex=COLOR_STAINLESS),
            MaterialRegion(heatsink, "aluminum_6061_t6", "heatsink", fdm_anisotropic=False, color_hex=COLOR_CNC_AL),
            MaterialRegion(hotblock, "copper_c110", "cucrzr_hotblock", fdm_anisotropic=False, color_hex=COLOR_COPPER),
        ]

    # Authentic multi-part visual.glb directly from official tricorn_hotends.step solids
    visual_parts: list[VisualPart] = [
        VisualPart(heatsink, "heatsink", COLOR_CNC_AL, metallic=0.88, roughness=0.28),
        VisualPart(crown, "slm_ti_crown", COLOR_SATIN_TI, metallic=0.85, roughness=0.34),
        VisualPart(heatbreak, "ti_heatbreak", COLOR_DARK_TI, metallic=0.86, roughness=0.28),
        VisualPart(hotblock, "cucrzr_hotblock", COLOR_COPPER, metallic=0.88, roughness=0.24),
        VisualPart(nozzle, "nozzle", COLOR_BRASS_NOZZLE, metallic=0.92, roughness=0.22),
        VisualPart(bucket, "ti_shield_bucket", COLOR_TI_SHIELD, metallic=0.80, roughness=0.38),
    ]
    for i, h_sol in enumerate(heaters):
        visual_parts.append(VisualPart(h_sol, f"heater_{i}", COLOR_CERAMIC, metallic=0.20, roughness=0.60))
    for i, sc in enumerate(screws + set_screws):
        visual_parts.append(VisualPart(sc, f"screw_{i}", COLOR_STAINLESS, metallic=0.92, roughness=0.24))

    spec = BakedAssetSpec(
        category="hotends",
        asset_id=asset_id,
        name=name,
        source_cad="tricorn_hotends.step",
        source_url="https://tricornhotend.com/",
        reference_material_id="titanium_grade5",
        cad_trust_pct=trust,
        port0=PortDefinition(
            (0.0, 0.0, 0.0),
            select_z_annulus(0.0, r_min_mm=2.5, r_max_mm=11.0, tol_mm=0.45),
        ),
        port1=PortDefinition(
            (0.0, 0.0, -oal_mm),
            select_z_cap(z_max_mm=-oal_mm + 4.2, z_min_mm=-oal_mm - 0.1, r_max_mm=4.2),
        ),
        n_internal_modes=8,
        mesh_size_mm=1.8,
        is_multi_material=True,
        is_fdm_anisotropic=False,
        part_recipe={
            "variant": variant,
            "overall_length_mm": oal_mm,
            "meltzone_length_mm": meltzone_mm,
            "mount_style": "rigid_4bolt_m3",
            "heaters": "3x 40W (4x40 mm)",
        },
        visual_max_triangles=24000,
    )
    return _bake_or_refresh_visuals(
        spec, regions, visual_parts=visual_parts, assets_root=assets_root, visuals_only=visuals_only
    )


# ---------------------------------------------------------------------------
# 6. Microwaved Bowden Toolhead (SLM Ti-6Al-4V MBTCFD Chassis)
# ---------------------------------------------------------------------------


def bake_microwaved_bowden(
    cad_dir: Path,
    assets_root: Path | None = None,
    visuals_only: bool = False,
) -> BakedAssetResult:
    """Bake Microwaved Bowden Toolhead SLM chassis (excluding Tricorn hotend & MGN12 carriage)."""
    step_path = cad_dir / "mbt_tricorn_v2.step"
    asm = b3d.import_step(str(step_path))

    def _to_printsim(shape: Any) -> Any:
        # CAD frame: +X right, +Y up, +Z front (carriage rear bolt face at z=7.24)
        # PrintSim canonical frame: +X right, +Y back (-z_cad), +Z up (+y_cad), Port 0 at [0,0,0]
        return shape.translate((0.0, 0.0, -7.24)).rotate(b3d.Axis.X, 90.0)

    mbtcfd_raw = [c for c in asm.children if c.label == "MBTCFD"][0]
    mbtcfd = _to_printsim(mbtcfd_raw)

    def _port0_mgn12_bolts(coords_mm: np.ndarray) -> np.ndarray:
        on_face = np.abs(coords_mm[:, 1]) <= 0.45
        near_bolt = np.zeros(len(coords_mm), dtype=bool)
        for bx in (-10.0, 10.0):
            for bz in (-10.0, 10.0):
                near_bolt |= np.hypot(coords_mm[:, 0] - bx, coords_mm[:, 2] - bz) <= 4.0
        return on_face & near_bolt

    def _port1_crown_bolts(coords_mm: np.ndarray) -> np.ndarray:
        on_crown = np.abs(coords_mm[:, 2] - 30.50) <= 0.55
        near_bolt = np.zeros(len(coords_mm), dtype=bool)
        for bx, by in ((-7.73, -11.83), (2.07, -17.49), (7.73, -7.69), (-2.07, -2.03)):
            near_bolt |= np.hypot(coords_mm[:, 0] - bx, coords_mm[:, 1] - by) <= 3.6
        return on_crown & near_bolt

    port0_sel = _port0_mgn12_bolts
    port1_sel = _port1_crown_bolts

    # Build clean multi-part visual.glb excluding Tricorn hotend & MGN12H carriage
    visual_parts: list[VisualPart] = [
        VisualPart(mbtcfd, "mbt_slm_chassis", COLOR_SATIN_TI, metallic=0.85, roughness=0.34),
    ]
    for ch in asm.children:
        lbl = ch.label
        if lbl in ("MBTCFD", "Tricorn_Hotends", "SLM_Monolith_universal_belt_clamp_assembly"):
            continue
        t_ch = _to_printsim(ch)
        if "ECAS" in lbl:
            visual_parts.append(VisualPart(t_ch, "ecas_fitting", COLOR_COPPER, metallic=0.88, roughness=0.26))
        elif "Beacon" in lbl:
            visual_parts.append(VisualPart(t_ch, "beacon_probe", COLOR_PCB_DARK, metallic=0.20, roughness=0.55))
        elif "Hotend_Brace" in lbl:
            visual_parts.append(VisualPart(t_ch, "hotend_brace", COLOR_DARK_TI, metallic=0.86, roughness=0.30))
        elif "MBTShield" in lbl:
            visual_parts.append(VisualPart(t_ch, "mbt_shield", COLOR_TI_SHIELD, metallic=0.80, roughness=0.38))
        elif "M3" in lbl or "screw" in lbl.lower():
            visual_parts.append(VisualPart(t_ch, f"fastener_{len(visual_parts)}", COLOR_STAINLESS, metallic=0.92, roughness=0.24))

    spec = BakedAssetSpec(
        category="toolheads",
        asset_id="microwaved_bowden",
        name="Microwaved Bowden Toolhead (SLM Ti-6Al-4V)",
        source_cad="mbt_tricorn_v2.step",
        source_url="https://github.com/micreathonn/Microwaved-Bowden-Toolhead",
        reference_material_id="titanium_grade5",
        cad_trust_pct=85,
        port0=PortDefinition((0.0, 0.0, 0.0), port0_sel),
        port1=PortDefinition((0.0, -9.76, 30.50), port1_sel),
        n_internal_modes=8,
        mesh_size_mm=1.8,
        is_multi_material=False,  # Scalable across Ti-6Al-4V / Al 6061 / AlSi10Mg / polymers!
        is_fdm_anisotropic=False,
        part_recipe={
            "chassis_part": "MBTCFD",
            "carriage_bolt_pattern": "4x M3 (20x20 mm MGN12H)",
            "hotend_mount_interface": "Tricorn / Dragon 4-bolt crown",
            "port1_offset_mm": [0.0, -9.76, 30.50],
        },
        visual_max_triangles=24000,
        fea_surface_max_triangles=60000,
    )
    regions = (
        []
        if visuals_only
        else [MaterialRegion(mbtcfd, "titanium_grade5", "mbt_slm_chassis", fdm_anisotropic=False, color_hex=COLOR_SATIN_TI)]
    )
    return _bake_or_refresh_visuals(
        spec,
        regions,
        visual_parts=visual_parts,
        assets_root=assets_root,
        visuals_only=visuals_only,
    )


# ---------------------------------------------------------------------------
# 7. Coupled Microwaved Bowden + Tricorn Long Assembly (Parallel Bottom Brace)
# ---------------------------------------------------------------------------


def _bake_coupled_multi_mount_superelement(
    spec: BakedAssetSpec,
    regions: list[MaterialRegion],
    extra_port0_selectors: list[Any],
    visual_parts: list[VisualPart],
    assets_root: Path | None = None,
) -> BakedAssetResult:
    """Bake a coupled assembly whose Port 0 ties multiple separated RBE3 mounting patches.

    When Port 0 grounds multiple spatially separated mounting interfaces onto the same
    parent frame (e.g., top Dragon mount at Z=0 and bottom Ti brace lugs at Z=-48 mm),
    throwing all nodes into a single RBE3 nullspace would allow rigid drift between mounts.
    Instead, each mounting patch receives its own independent RBE3 filter around the
    shared Port 0 virtual point ``[0, 0, 0]``, and their 6-DOF virtual points are tied
    via exact congruence ``Q`` while preserving local bolt-hole warping relaxation.
    """
    root = assets_root if assets_root is not None else ASSETS_ROOT
    asset_dir = root / spec.category / spec.asset_id
    asset_dir.mkdir(parents=True, exist_ok=True)

    pts_mm, tets, region_ids = _mesh_regions_tet4(
        regions=regions,
        mesh_size_mm=spec.mesh_size_mm,
        fea_surface_max_triangles=spec.fea_surface_max_triangles,
    )
    pts_m = pts_mm * 1e-3

    mesh, K, M, coords_m = _assemble_tet4_km(
        points_m=pts_m,
        tets=tets,
        region_ids=region_ids,
        regions=regions,
        default_fdm_anisotropic=spec.is_fdm_anisotropic,
    )
    coords_mm = coords_m * 1000.0
    mass_kg, com_m, inertia_kg_m2 = _compute_mass_com_inertia(M, coords_m)

    bnd_nodes = np.asarray(mesh.boundary_nodes(), dtype=np.int64)
    bnd_coords_mm = coords_mm[bnd_nodes]

    p0_top = bnd_nodes[np.asarray(spec.port0.selector(bnd_coords_mm), dtype=bool)]
    p1_nodes = np.setdiff1d(
        bnd_nodes[np.asarray(spec.port1.selector(bnd_coords_mm), dtype=bool)],
        p0_top,
    )
    used_nodes = np.union1d(p0_top, p1_nodes)

    lug_node_sets: list[np.ndarray] = []
    for sel in extra_port0_selectors:
        lug_nodes = np.setdiff1d(bnd_nodes[np.asarray(sel(bnd_coords_mm), dtype=bool)], used_nodes)
        if len(lug_nodes) < 4:
            raise ValueError(f"Coupled Port 0 lug patch matched only {len(lug_nodes)} nodes (need >= 4)")
        lug_node_sets.append(lug_nodes)
        used_nodes = np.union1d(used_nodes, lug_nodes)

    if len(p0_top) < 4 or len(p1_nodes) < 4:
        raise ValueError(
            f"Port node count too small for '{spec.asset_id}': top={len(p0_top)}, tip={len(p1_nodes)}"
        )

    p0_vp_m = np.asarray(spec.port0.vp_coord_mm, dtype=np.float64) * 1e-3
    p1_vp_m = np.asarray(spec.port1.vp_coord_mm, dtype=np.float64) * 1e-3
    for label, p_nodes, vp_m in (
        [("Port 0 top", p0_top, p0_vp_m), ("Port 1 tip", p1_nodes, p1_vp_m)]
        + [(f"Port 0 lug {i}", ln, p0_vp_m) for i, ln in enumerate(lug_node_sets)]
    ):
        R_test = virtual_point_transform(coords_m[p_nodes], vp_m)
        if np.linalg.matrix_rank(R_test, tol=1e-9) != 6:
            raise ValueError(f"{label} patch for '{spec.asset_id}' is rank-deficient (<6)")

    port_coords_m = np.vstack([p0_vp_m, p1_vp_m]).astype(np.float64)
    hash_payload = {
        "schema_version": 1,
        "category": spec.category,
        "asset_id": spec.asset_id,
        "reference_material_id": spec.reference_material_id,
        "is_multi_material": spec.is_multi_material,
        "is_fdm_anisotropic": spec.is_fdm_anisotropic,
        "mesh_size_mm": spec.mesh_size_mm,
        "n_internal_modes": spec.n_internal_modes,
        "num_vertices": int(mesh.nvertices),
        "num_elements": int(mesh.nelements),
        "port0_mm": list(spec.port0.vp_coord_mm),
        "port1_mm": list(spec.port1.vp_coord_mm),
        "recipe": spec.part_recipe,
    }
    source_hash = hashlib.sha256(json.dumps(hash_payload, sort_keys=True).encode("utf-8")).hexdigest()

    port_node_sets = [
        (p0_top, coords_m[p0_top], p0_vp_m),
        (p1_nodes, coords_m[p1_nodes], p1_vp_m),
    ] + [(ln, coords_m[ln], p0_vp_m) for ln in lug_node_sets]

    se_multi = condense_solid_patch_to_vp(
        K,
        M,
        port_node_sets=port_node_sets,
        n_modes=spec.n_internal_modes,
        name=spec.asset_id,
        source_hash=source_hash,
        interface_mode="rbe3",
        mass_kg=mass_kg,
    )

    n_ports = 2 + len(lug_node_sets)
    n_m = se_multi.n_internal_modes
    Q = np.zeros((6 * n_ports + n_m, 12 + n_m), dtype=np.float64)
    Q[0:6, 0:6] = np.eye(6)
    Q[6:12, 6:12] = np.eye(6)
    for k in range(len(lug_node_sets)):
        r0 = 12 + 6 * k
        Q[r0 : r0 + 6, 0:6] = np.eye(6)
    if n_m > 0:
        Q[6 * n_ports :, 12:] = np.eye(n_m)

    K_cb = 0.5 * (Q.T @ se_multi.K_cb @ Q + (Q.T @ se_multi.K_cb @ Q).T)
    M_cb = 0.5 * (Q.T @ se_multi.M_cb @ Q + (Q.T @ se_multi.M_cb @ Q).T)

    rigid_max_hz, K_bb, port_stiff_n_mm, port_rot_nm_rad = _verify_and_extract_tip_stiffness(
        K_cb, M_cb
    )

    glb_path = asset_dir / "visual.glb"
    vis_tris, vis_bytes = build_visual_glb(
        visual_parts, glb_path, max_total_triangles=spec.visual_max_triangles
    )

    ref_mat = get_material(spec.reference_material_id)
    recipe_out = {
        "schema_version": 1,
        "category": spec.category,
        "asset_id": spec.asset_id,
        "source_cad": spec.source_cad,
        "source_url": spec.source_url,
        "regions": [
            {
                "label": r.label or f"region_{i}",
                "material_id": r.material_id,
                "fdm_anisotropic": r.fdm_anisotropic,
            }
            for i, r in enumerate(regions)
        ],
        **spec.part_recipe,
    }
    (asset_dir / "part_recipe.json").write_text(json.dumps(recipe_out, indent=2), encoding="utf-8")

    np.savez(
        asset_dir / "superelement.npz",
        K_cb=np.asarray(K_cb, dtype=np.float64),
        M_cb=np.asarray(M_cb, dtype=np.float64),
        K_bb=np.asarray(K_bb, dtype=np.float64),
        port_coords_m=port_coords_m,
        fixed_interface_freqs_hz=np.asarray(se_multi.fixed_interface_freqs_hz, dtype=np.float64),
        com_m=com_m,
        inertia_kg_m2=inertia_kg_m2,
    )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "category": spec.category,
        "asset_id": spec.asset_id,
        "name": spec.name,
        "source_cad": spec.source_cad,
        "source_url": spec.source_url,
        "baked_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "reference_material_id": spec.reference_material_id,
        "reference_youngs_modulus_pa": float(ref_mat.youngs_modulus_pa),
        "reference_density_kg_m3": float(ref_mat.density_kg_m3),
        "is_multi_material": bool(spec.is_multi_material),
        "is_fdm_anisotropic": bool(spec.is_fdm_anisotropic),
        "element_type": "Tet4",
        "num_vertices": int(mesh.nvertices),
        "num_elements": int(mesh.nelements),
        "num_dofs": int(3 * mesh.nvertices),
        "n_boundary_dofs": 12,
        "n_internal_modes": int(se_multi.n_internal_modes),
        "rigid_body_max_freq_hz": round(float(rigid_max_hz), 6),
        "mass_kg": round(float(mass_kg), 6),
        "com_mm": [round(float(v * 1000.0), 4) for v in com_m],
        "port0_mm": [round(float(v), 4) for v in spec.port0.vp_coord_mm],
        "port1_mm": [round(float(v), 4) for v in spec.port1.vp_coord_mm],
        "port_stiffness_n_per_mm": port_stiff_n_mm,
        "port_rot_stiffness_nm_per_rad": port_rot_nm_rad,
        "fixed_interface_freqs_hz": [round(float(f), 2) for f in se_multi.fixed_interface_freqs_hz],
        "visual_glb": "visual.glb",
        "visual_triangles": int(vis_tris),
        "visual_bytes": int(vis_bytes),
        "cad_trust_pct": int(spec.cad_trust_pct),
        "source_hash": source_hash,
    }
    (asset_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    clear_asset_cache()

    return BakedAssetResult(
        category=spec.category,
        asset_id=spec.asset_id,
        asset_dir=asset_dir,
        manifest=manifest,
        K_cb=K_cb,
        M_cb=M_cb,
        K_bb=K_bb,
        port_coords_m=port_coords_m,
        com_m=com_m,
        inertia_kg_m2=inertia_kg_m2,
    )


def bake_microwaved_bowden_tricorn_long(
    cad_dir: Path,
    assets_root: Path | None = None,
    visuals_only: bool = False,
) -> BakedAssetResult:
    """Bake coupled ``assemblies/microwaved_bowden__tricorn_long`` (Tricorn Long + Ti Hotend_Brace)."""
    mbt_step_path = cad_dir / "mbt_tricorn_v2.step"
    asm = b3d.import_step(str(mbt_step_path))

    def _to_printsim(shape: Any) -> Any:
        return shape.translate((0.0, 0.0, -7.24)).rotate(b3d.Axis.X, 90.0)

    # Verify Tricorn_Hotends canonical center (cx, cy) and top face Z inside mbt_tricorn_v2.step
    th_asm = [c for c in asm.children if c.label == "Tricorn_Hotends"][0]
    mbt_nozzle_raw = [c for c in th_asm.children if c.label == "nozzle"][0].moved(th_asm.location)
    mbt_hb_raw = [c for c in th_asm.children if c.label == "heatbreak"][0].moved(th_asm.location)
    mbt_hotblock_raw = [c for c in th_asm.children if c.label == "hotblock-5"][0].moved(th_asm.location)

    nbb = _to_printsim(mbt_nozzle_raw).bounding_box()
    hbb = _to_printsim(mbt_hb_raw).bounding_box()
    cx = 0.5 * (float(nbb.min.X) + float(nbb.max.X))
    cy = 0.5 * (float(nbb.min.Y) + float(nbb.max.Y))
    z_top = float(hbb.max.Z)
    z_tip = float(nbb.min.Z)
    if abs(z_top - 30.50) > 1e-3 or abs(z_tip - (-35.50)) > 1e-3:
        raise ValueError(f"Unexpected Tricorn datum in mbt_tricorn_v2.step: top={z_top}, tip={z_tip}")

    shift = (-cx, -cy, -z_top)

    def _to_coupled_frame(shape: Any) -> Any:
        return _to_printsim(shape).translate(shift)

    brace = _to_coupled_frame(
        [c for c in asm.children if c.label == "Hotend_Brace^Microwaved_Bowden_Toolhead"][0]
    )
    fhcs_screws = [_to_coupled_frame(c) for c in asm.children if c.label == "M3x0_5_FHCS"]
    if len(fhcs_screws) != 3:
        raise ValueError(f"Expected 3 M3x0_5_FHCS screws in mbt_tricorn_v2.step, found {len(fhcs_screws)}")

    # Locate the 2 outer brace mounting bolt centers (around X = +-11.0 mm, Z = -48.0 mm)
    outer_bolt_centers = sorted(
        [
            (float(sc.bounding_box().center().X), float(sc.bounding_box().center().Z))
            for sc in fhcs_screws
            if abs(float(sc.bounding_box().center().X)) > 5.0
        ],
        key=lambda p: p[0],
    )
    if len(outer_bolt_centers) != 2:
        raise ValueError(f"Expected 2 outer brace bolt centers, found {outer_bolt_centers}")

    # Load clean Tricorn Long B-Reps from tricorn_hotends.step rotated 180 deg around Z to match
    # mbt_tricorn_v2.step pose identically (avoids mbt_tricorn_v2's 62 split cosmetic nozzle thread faces).
    th_step = b3d.import_step(str(cad_dir / "tricorn_hotends.step"))
    th_children = [
        c for c in th_step.children
        if -13.0 <= 0.5 * (c.bounding_box().min.X + c.bounding_box().max.X) <= 13.0
    ]
    by_lbl: dict[str, list[Any]] = {}
    for ch in th_children:
        by_lbl.setdefault(ch.label, []).append(
            ch.translate((0.0, 0.0, -61.0)).rotate(b3d.Axis.Z, 180.0)
        )

    heatsink = by_lbl["heatsink"][0]
    crown = by_lbl["crown"][0]
    heatbreak = by_lbl["heatbreak"][0]
    hotblock = by_lbl["hotblock"][0]
    nozzle = by_lbl["nozzle"][0]
    heaters = by_lbl.get("heater", [])
    screws = by_lbl.get("Hex_socket_countersunk_head_screw_M3x0_5_x_6", [])
    set_screws = by_lbl.get("Cup_point_hex_socket_set_screw_M4x0_7_x_4", [])

    # Verify exact pose alignment against mbt_tricorn_v2.step's hotblock-5
    mbt_hb_shifted = _to_coupled_frame(mbt_hotblock_raw).bounding_box()
    th_hb_bb = hotblock.bounding_box()
    if (
        abs(float(mbt_hb_shifted.min.Y) - float(th_hb_bb.min.Y)) > 1e-3
        or abs(float(mbt_hb_shifted.max.Y) - float(th_hb_bb.max.Y)) > 1e-3
        or abs(float(mbt_hb_shifted.min.Z) - float(th_hb_bb.min.Z)) > 1e-3
    ):
        raise ValueError("Tricorn rotated pose mismatch against mbt_tricorn_v2.step hotblock-5")

    crown_screws = []
    for sc in screws:
        bb = sc.bounding_box()
        if 0.5 * (bb.min.Z + bb.max.Z) < -32.0:
            # Skip standalone Tricorn bucket screw; Microwaved's center M3x0_5_FHCS occupies this hole
            continue
        c = bb.center()
        crown_screws.append(sc.translate((-c.X, -c.Y, -c.Z)).scale(1.06).translate((c.X, c.Y, c.Z)))

    # Keep M3x0_5_FHCS unscaled: center screw already overlaps hotblock by 4.51 mm^3 across the
    # 0.0016 mm brace-to-hotblock gap, and unscaled 90-deg countersunk cones match Hotend_Brace's
    # seats with 0.0000 mm coincident surfaces (conformal single connected component in occ.fragment).
    regions: list[MaterialRegion] = []
    if not visuals_only:
        ti_struct = b3d.Compound(children=[crown, heatbreak, brace])
        screw_comp = b3d.Compound(children=crown_screws + fhcs_screws)
        regions = [
            MaterialRegion(nozzle, "copper_c110", "v6_nozzle", fdm_anisotropic=False, color_hex=COLOR_BRASS_NOZZLE),
            MaterialRegion(ti_struct, "titanium_grade5", "ti_crown_heatbreak_brace", fdm_anisotropic=False, color_hex=COLOR_SATIN_TI),
            MaterialRegion(screw_comp, "stainless_304", "m3_crown_and_brace_screws", fdm_anisotropic=False, color_hex=COLOR_STAINLESS),
            MaterialRegion(heatsink, "aluminum_6061_t6", "heatsink", fdm_anisotropic=False, color_hex=COLOR_CNC_AL),
            MaterialRegion(hotblock, "copper_c110", "cucrzr_hotblock", fdm_anisotropic=False, color_hex=COLOR_COPPER),
        ]

    visual_parts: list[VisualPart] = [
        VisualPart(heatsink, "heatsink", COLOR_CNC_AL, metallic=0.88, roughness=0.28),
        VisualPart(crown, "slm_ti_crown", COLOR_SATIN_TI, metallic=0.85, roughness=0.34),
        VisualPart(heatbreak, "ti_heatbreak", COLOR_DARK_TI, metallic=0.86, roughness=0.28),
        VisualPart(hotblock, "cucrzr_hotblock", COLOR_COPPER, metallic=0.88, roughness=0.24),
        VisualPart(nozzle, "nozzle", COLOR_BRASS_NOZZLE, metallic=0.92, roughness=0.22),
        VisualPart(brace, "hotend_brace", COLOR_DARK_TI, metallic=0.86, roughness=0.30),
    ]
    for i, h_sol in enumerate(heaters):
        visual_parts.append(VisualPart(h_sol, f"heater_{i}", COLOR_CERAMIC, metallic=0.20, roughness=0.60))
    vis_crown_screws = [sc for sc in screws if 0.5 * (sc.bounding_box().min.Z + sc.bounding_box().max.Z) >= -32.0]
    for i, sc in enumerate(vis_crown_screws + set_screws + fhcs_screws):
        visual_parts.append(VisualPart(sc, f"screw_{i}", COLOR_STAINLESS, metallic=0.92, roughness=0.24))

    def _make_lug_selector(bx: float, bz: float, is_left: bool):
        def _sel(coords_mm: np.ndarray) -> np.ndarray:
            x_side = coords_mm[:, 0] <= -7.6 if is_left else coords_mm[:, 0] >= 7.6
            on_face = np.abs(coords_mm[:, 1] - 9.76) <= 0.45
            near_bolt = np.hypot(coords_mm[:, 0] - bx, coords_mm[:, 2] - bz) <= 3.6
            return x_side & on_face & near_bolt

        return _sel

    lug_selectors = [
        _make_lug_selector(outer_bolt_centers[0][0], outer_bolt_centers[0][1], is_left=True),
        _make_lug_selector(outer_bolt_centers[1][0], outer_bolt_centers[1][1], is_left=False),
    ]

    spec = BakedAssetSpec(
        category="assemblies",
        asset_id="microwaved_bowden__tricorn_long",
        name="Tricorn Long + Microwaved Ti Bottom Brace",
        source_cad="mbt_tricorn_v2.step",
        source_url="https://github.com/micreathonn/Microwaved-Bowden-Toolhead",
        reference_material_id="titanium_grade5",
        cad_trust_pct=89,
        port0=PortDefinition(
            (0.0, 0.0, 0.0),
            select_z_annulus(0.0, r_min_mm=2.5, r_max_mm=11.0, tol_mm=0.45),
        ),
        port1=PortDefinition(
            (0.0, 0.0, -66.0),
            select_z_cap(z_max_mm=-61.8, z_min_mm=-66.1, r_max_mm=4.2),
        ),
        n_internal_modes=8,
        mesh_size_mm=1.8,
        is_multi_material=True,
        is_fdm_anisotropic=False,
        part_recipe={
            "toolhead_id": "microwaved_bowden",
            "hotend_id": "tricorn_long",
            "brace_part": "Hotend_Brace (Ti-6Al-4V, 0.32 g)",
            "overall_length_mm": 66.0,
            "meltzone_length_mm": 53.5,
            "brace_outer_bolt_centers_xz_mm": [
                [round(outer_bolt_centers[0][0], 3), round(outer_bolt_centers[0][1], 3)],
                [round(outer_bolt_centers[1][0], 3), round(outer_bolt_centers[1][1], 3)],
            ],
        },
        visual_max_triangles=24000,
    )
    if visuals_only:
        return _bake_or_refresh_visuals(
            spec, [], visual_parts=visual_parts, assets_root=assets_root, visuals_only=True
        )
    return _bake_coupled_multi_mount_superelement(
        spec,
        regions,
        extra_port0_selectors=lug_selectors,
        visual_parts=visual_parts,
        assets_root=assets_root,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Bake all Phase 1 CAD hardware superelement assets.")
    parser.add_argument("--cad-dir", type=Path, default=DEFAULT_CAD_DIR, help="Directory with downloaded STEP files")
    parser.add_argument("--out-dir", type=Path, default=ASSETS_ROOT, help="Output hardware assets root")
    parser.add_argument("--only", type=str, default="", help="Comma-separated asset IDs to bake (default: all)")
    parser.add_argument(
        "--visuals-only",
        action="store_true",
        help="Regenerate only visual.glb and update visual_triangles/visual_bytes in existing manifest.json",
    )
    args = parser.parse_args()

    only_set = {x.strip() for x in args.only.split(",") if x.strip()}
    vo = bool(args.visuals_only)
    builders = [
        ("chube_compact", lambda: bake_chube_compact(args.cad_dir, args.out_dir, visuals_only=vo)),
        ("chube_air", lambda: bake_chube_air(args.cad_dir, args.out_dir, visuals_only=vo)),
        ("chube_conduction", lambda: bake_chube_conduction(args.cad_dir, args.out_dir, visuals_only=vo)),
        ("tricorn_short", lambda: _bake_tricorn_variant(args.cad_dir, "short", args.out_dir, visuals_only=vo)),
        ("tricorn_long", lambda: _bake_tricorn_variant(args.cad_dir, "long", args.out_dir, visuals_only=vo)),
        ("microwaved_bowden", lambda: bake_microwaved_bowden(args.cad_dir, args.out_dir, visuals_only=vo)),
        (
            "microwaved_bowden__tricorn_long",
            lambda: bake_microwaved_bowden_tricorn_long(args.cad_dir, args.out_dir, visuals_only=vo),
        ),
    ]

    console = Console()
    results: list[tuple[BakedAssetResult, float]] = []
    for aid, fn in builders:
        if only_set and aid not in only_set:
            continue
        t0 = time.perf_counter()
        mode_lbl = "Refreshing visuals for" if vo else "Baking"
        console.print(f"[cyan]{mode_lbl}[/cyan] {aid}...")
        res = fn()
        dt = time.perf_counter() - t0
        results.append((res, dt))

    tbl = Table(title="Phase 1 Baked CAD Hardware Superelements", show_header=True, header_style="bold magenta")
    tbl.add_column("Category / ID")
    tbl.add_column("Mat / Multi")
    tbl.add_column("Verts / Tets", justify="right")
    tbl.add_column("Mass (g)", justify="right")
    tbl.add_column("Tip Stiff kx/ky/kz (N/mm)", justify="right")
    tbl.add_column("f_1 (Hz)", justify="right")
    tbl.add_column("RB max (Hz)", justify="right")
    tbl.add_column("GLB", justify="right")
    tbl.add_column("Time", justify="right")

    for res, dt in results:
        m = res.manifest
        ps = m["port_stiffness_n_per_mm"]
        f1 = m["fixed_interface_freqs_hz"][0] if m["fixed_interface_freqs_hz"] else 0.0
        tbl.add_row(
            f"{m['category']}/{m['asset_id']}",
            f"{m['reference_material_id']} ({'multi' if m['is_multi_material'] else 'mono'})",
            f"{m['num_vertices']:,} / {m['num_elements']:,}",
            f"{m['mass_kg'] * 1000.0:.1f}",
            f"{ps['kx']:.0f} / {ps['ky']:.0f} / {ps['kz']:.0f}",
            f"{f1:.0f}",
            f"{m['rigid_body_max_freq_hz']:.4f}",
            f"{m['visual_triangles']:,}t / {m['visual_bytes'] / 1024:.1f}KB",
            f"{dt:.1f}s",
        )
    console.print(tbl)


if __name__ == "__main__":
    main()
