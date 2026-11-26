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
COLOR_CHUBE_TI = "#6f7782"
COLOR_DARK_TI = "#8e99a8"
COLOR_CNC_AL = "#dfe6ef"
COLOR_SLM_AL = "#cfd8e3"
COLOR_COPPER = "#d97742"
COLOR_NICKEL_CU = "#cdc6ba"
COLOR_BRASS_NOZZLE = "#eab308"
COLOR_DARK_SOCK = "#232a36"
COLOR_TI_SHIELD = "#64748b"
COLOR_STAINLESS = "#d8e0ea"
COLOR_SCREW_SOCKET = "#374151"
COLOR_HEATER_LEAD = "#c2410c"
COLOR_THERM_LEAD = "#e2e8f0"
COLOR_CERAMIC = "#e5e7eb"
COLOR_PCB_DARK = "#1e293b"


def _resolve_cad_file(cad_dir: Path, *candidates: str) -> Path:
    """Resolve the first existing STEP path under ``cad_dir`` or ``cad_sources/``."""
    search_roots = [cad_dir, Path("cad_sources"), DEFAULT_CAD_DIR]
    for root in search_roots:
        for rel in candidates:
            p = root / rel
            if p.is_file():
                return p
    raise FileNotFoundError(
        f"None of {candidates} found under {[str(r) for r in search_roots]}"
    )


def _find_child_by_substr(parent: Any, *substrings: str) -> Any:
    """Find first child whose label contains any of ``substrings`` (case-insensitive)."""
    lows = [s.lower() for s in substrings]
    for ch in getattr(parent, "children", []):
        lbl = (getattr(ch, "label", "") or "").lower()
        if any(s in lbl for s in lows):
            return ch
    raise KeyError(f"No child matching {substrings} in {getattr(parent, 'label', parent)}")


def _make_flush_torx_screw(
    origin_mm: tuple[float, float, float],
    outward_dir: tuple[float, float, float],
    head_diam_mm: float = 5.2,
    head_height_mm: float = 1.4,
    shank_diam_mm: float = 3.0,
    shank_len_mm: float = 2.6,
) -> Any:
    """Build a crisp, flush-seated M3 countersunk flat-head Torx screw (no QEM-mangled spikes).

    ``origin_mm`` is the outer center of the flat screw face; ``outward_dir`` points outward
    from the mounting surface. The conical head extends inward along ``-outward_dir``.
    """
    d = np.asarray(outward_dir, dtype=float)
    d /= max(float(np.linalg.norm(d)), 1e-12)
    ref = np.array([1.0, 0.0, 0.0]) if abs(d[2]) > 0.95 else np.array([0.0, 0.0, 1.0])
    x_dir = np.cross(ref, d)
    x_dir /= np.linalg.norm(x_dir)
    o = np.asarray(origin_mm, dtype=float)

    # Slight 0.20 mm cylindrical rim + 90-deg countersunk taper + short shank inward
    rim_h = 0.22
    r_head = 0.5 * head_diam_mm
    r_shank = 0.5 * shank_diam_mm
    p_rim = b3d.Plane(origin=tuple(o - rim_h * d), x_dir=tuple(x_dir), z_dir=tuple(d))
    rim = b3d.Solid.make_cylinder(r_head, rim_h, plane=p_rim)

    cone_h = max(head_height_mm - rim_h, 0.6)
    p_cone = b3d.Plane(origin=tuple(o - (rim_h + cone_h) * d), x_dir=tuple(x_dir), z_dir=tuple(d))
    cone = b3d.Solid.make_cone(r_shank, r_head, cone_h, plane=p_cone)

    p_shank = b3d.Plane(
        origin=tuple(o - (rim_h + cone_h + shank_len_mm) * d),
        x_dir=tuple(x_dir),
        z_dir=tuple(d),
    )
    shank = b3d.Solid.make_cylinder(r_shank, shank_len_mm, plane=p_shank)

    # Clean T10 hex/Torx recess pocket in the top face for authentic visual detail
    p_sock = b3d.Plane(origin=tuple(o - 0.85 * d), x_dir=tuple(x_dir), z_dir=tuple(d))
    socket = b3d.Solid.make_cylinder(1.32, 1.10, plane=p_sock)
    return rim.fuse(cone, shank).cut(socket).clean()


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
# 1. Chube Compact (Official SolidWorks STEP Geometry)
# ---------------------------------------------------------------------------


def bake_chube_compact(
    cad_dir: Path,
    assets_root: Path | None = None,
    visuals_only: bool = False,
) -> BakedAssetResult:
    """Bake Luke's Lab Chube Compact (exact Ø22.23 mm Ti monocoque cutouts + official STEP parts)."""
    step_path = _resolve_cad_file(
        cad_dir,
        "chube_compact_integration.step",
        "chube_official/extracted/chube_compact.step",
    )
    asm = b3d.import_step(str(step_path))

    # Extract official STEP parts by label (robust across Onshape & SolidWorks exports)
    step_nozzle = _find_child_by_substr(asm, "nozzle")
    compact_sub = _find_child_by_substr(asm, "chube_compact_integration_cad")
    step_sock = _find_child_by_substr(compact_sub, "sock")
    step_chalice = _find_child_by_substr(compact_sub, "chalice")

    # Exact Ø22.23 mm OD x 0.89 mm wall Ti-6Al-4V monocoque outer shell (41.68 mm bare, 47.20 mm OAL)
    ro = 11.115
    ri = 10.225
    body_h = 41.68

    outer_tube = b3d.Solid.make_cylinder(ro, body_h, plane=b3d.Plane(origin=(0.0, 0.0, -body_h)))
    # Subtle stepped top flange register matching official STEP (z = 0..-1.30 mm)
    top_step_ring = b3d.Solid.make_cylinder(ro + 0.5, 1.30, plane=b3d.Plane(origin=(0.0, 0.0, -1.30))).cut(
        b3d.Solid.make_cylinder(ro - 0.18, 2.0, plane=b3d.Plane(origin=(0.0, 0.0, -1.60)))
    )
    inner_bore = b3d.Solid.make_cylinder(ri, body_h + 2.0, plane=b3d.Plane(origin=(0.0, 0.0, -body_h - 1.0)))
    ti_tube_open = outer_tube.cut(inner_bore, top_step_ring)

    # Exact 4x Ø12.50 mm circular cooling cutouts at z = -12.90 mm along 45/135/225/315 deg
    # (extracted directly from faces 54, 56, 58, 60 of Chube_Compact_integration_assembly.step —
    # staggered 45 deg cleanly away from the 4 upper radial screws at 0/90/180/270 deg!)
    cooling_cuts = []
    for a in (45.0, 135.0, 225.0, 315.0):
        ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
        pln = b3d.Plane(origin=(0.0, 0.0, -12.90), z_dir=(ca, sa, 0.0))
        cooling_cuts.append(b3d.Solid.make_cylinder(6.25, 16.0, plane=pln))

    # Exact horizontal 6 mm heater & 3 mm thermistor bores through shell at z = -27.50 mm
    heater_cut = b3d.Solid.make_cylinder(
        3.15, 28.0, plane=b3d.Plane(origin=(-5.00, -14.0, -27.50), z_dir=(0.0, 1.0, 0.0))
    )
    therm_cut = b3d.Solid.make_cylinder(
        1.65, 28.0, plane=b3d.Plane(origin=(5.70, -14.0, -27.50), z_dir=(0.0, 1.0, 0.0))
    )
    ti_tube_vis = ti_tube_open.cut(*cooling_cuts, heater_cut, therm_cut).clean()

    # Exact 4x M3 Dragon PCD Ø16 mm top mounting holes (45/135/225/315 deg, depth 7.50 mm)
    comp_m3 = [
        b3d.Solid.make_cylinder(
            1.48,
            8.0,
            plane=b3d.Plane(
                origin=(8.0 * math.cos(math.radians(a)), 8.0 * math.sin(math.radians(a)), -7.5)
            ),
        )
        for a in (45.0, 135.0, 225.0, 315.0)
    ]

    # Bright CNC aluminum 3-fin upper heatsink (z = 0..-20.0 mm) visible through the 4x Ø12.50 mm cutouts
    comp_hs_top = b3d.Solid.make_cylinder(ri - 0.03, 4.2, plane=b3d.Plane(origin=(0.0, 0.0, -4.2)))
    comp_hs_core = b3d.Solid.make_cylinder(4.8, 19.6, plane=b3d.Plane(origin=(0.0, 0.0, -19.6)))
    comp_fin1 = b3d.Solid.make_cylinder(ri - 0.06, 1.8, plane=b3d.Plane(origin=(0.0, 0.0, -8.8)))
    comp_fin2 = b3d.Solid.make_cylinder(ri - 0.06, 1.8, plane=b3d.Plane(origin=(0.0, 0.0, -13.6)))
    comp_fin3 = b3d.Solid.make_cylinder(ri - 0.06, 1.8, plane=b3d.Plane(origin=(0.0, 0.0, -18.4)))
    comp_heatsink = (
        comp_hs_top.fuse(comp_hs_core, comp_fin1, comp_fin2, comp_fin3)
        .cut(b3d.Solid.make_cylinder(2.75, 22.0, plane=b3d.Plane(origin=(0.0, 0.0, -20.5))), *comp_m3)
        .clean()
    )

    # Chalice Ti heatbreak throat connecting top Chalice cup to 33 mm copper melt core
    comp_chalice_throat = (
        b3d.Solid.make_cylinder(2.2, 18.5, plane=b3d.Plane(origin=(0.0, 0.0, -22.0)))
        .cut(b3d.Solid.make_cylinder(1.0, 22.0, plane=b3d.Plane(origin=(0.0, 0.0, -23.0))))
        .clean()
    )

    # Electroless-nickel-plated copper 33 mm melt core (z = -21.0..-41.60 mm) w/ horizontal cartridge bores
    comp_melt_upper = b3d.Solid.make_cylinder(5.2, 10.0, plane=b3d.Plane(origin=(0.0, 0.0, -31.0)))
    comp_melt_lower = b3d.Solid.make_cylinder(ri - 0.12, 19.2, plane=b3d.Plane(origin=(0.0, 0.0, -41.55)))
    comp_melt = (
        comp_melt_upper.fuse(comp_melt_lower)
        .cut(
            b3d.Solid.make_cylinder(1.0, 24.0, plane=b3d.Plane(origin=(0.0, 0.0, -42.5))),
            heater_cut,
            therm_cut,
        )
        .clean()
    )

    regions: list[MaterialRegion] = []
    if not visuals_only:
        # Conformal multi-material Tet4 FEA built directly from the exact cutout Ti shell + Al heatsink + Cu core
        fea_ti_shell = (
            ti_tube_vis.fuse(
                b3d.Solid.make_cylinder(ro, 3.6, plane=b3d.Plane(origin=(0.0, 0.0, -3.6))).cut(
                    b3d.Solid.make_cylinder(2.75, 5.0, plane=b3d.Plane(origin=(0.0, 0.0, -4.5))),
                    *comp_m3,
                ),
                b3d.Solid.make_cylinder(ro, 3.4, plane=b3d.Plane(origin=(0.0, 0.0, -body_h))).cut(
                    b3d.Solid.make_cylinder(3.0, 5.0, plane=b3d.Plane(origin=(0.0, 0.0, -body_h - 1.0)))
                ),
                comp_chalice_throat,
            ).clean()
        )
        # 2x lower M3 lateral pins coupling Ti tube to copper melt core at z = -36.88 mm
        pin_x = _make_m3_pin((-ro + 0.3, 0.0, -36.88), (ro - 0.3, 0.0, -36.88), radius_mm=1.55)
        fea_copper = comp_melt.fuse(pin_x).clean()
        fea_nozzle = _make_v6_nozzle_solid(tip_z_mm=-47.20, total_len_mm=12.5)

        regions = _non_overlapping_regions(
            [
                (fea_nozzle, "copper_c110", "v6_nozzle", COLOR_BRASS_NOZZLE, 0.90, 0.25),
                (fea_ti_shell, "titanium_grade5", "ti_monocoque_shell", COLOR_CHUBE_TI, 0.85, 0.32),
                (comp_heatsink, "aluminum_6061_t6", "compact_heatsink", COLOR_CNC_AL, 0.88, 0.22),
                (fea_copper, "copper_c110", "copper_melt_core", COLOR_NICKEL_CU, 0.88, 0.24),
            ]
        )

    # Stainless 6x16 mm horizontal heater + 3x15 mm thermistor + tidy short insulated lead stubs
    vis_heater_cart = b3d.Solid.make_cylinder(
        2.95, 16.0, plane=b3d.Plane(origin=(-5.00, -10.2, -27.50), z_dir=(0.0, 1.0, 0.0))
    )
    vis_heater_leads = b3d.Solid.make_cylinder(
        0.95, 4.5, plane=b3d.Plane(origin=(-5.95, -14.2, -27.50), z_dir=(0.0, 1.0, 0.0))
    ).fuse(
        b3d.Solid.make_cylinder(
            0.95, 4.5, plane=b3d.Plane(origin=(-4.05, -14.2, -27.50), z_dir=(0.0, 1.0, 0.0))
        )
    )
    vis_therm_cart = b3d.Solid.make_cylinder(
        1.48, 15.0, plane=b3d.Plane(origin=(5.70, -9.8, -27.50), z_dir=(0.0, 1.0, 0.0))
    )
    vis_therm_leads = b3d.Solid.make_cylinder(
        0.60, 4.2, plane=b3d.Plane(origin=(5.25, -13.6, -27.50), z_dir=(0.0, 1.0, 0.0))
    ).fuse(
        b3d.Solid.make_cylinder(
            0.60, 4.2, plane=b3d.Plane(origin=(6.15, -13.6, -27.50), z_dir=(0.0, 1.0, 0.0))
        )
    )

    # Flush-seated M3 countersunk Torx screws (4 upper radial @ z=-5.0, 2 bottom vertical @ z=-41.68)
    screw_specs = [
        ((ro + 0.16, 0.0, -5.00), (1.0, 0.0, 0.0)),
        ((-ro - 0.16, 0.0, -5.00), (-1.0, 0.0, 0.0)),
        ((0.0, ro + 0.16, -5.00), (0.0, 1.0, 0.0)),
        ((0.0, -ro - 0.16, -5.00), (0.0, -1.0, 0.0)),
        ((7.50, 0.0, -body_h - 0.16), (0.0, 0.0, -1.0)),
        ((-7.50, 0.0, -body_h - 0.16), (0.0, 0.0, -1.0)),
    ]

    visual_parts: list[VisualPart] = [
        VisualPart(ti_tube_vis, "chube_body", COLOR_CHUBE_TI, metallic=0.84, roughness=0.32),
        VisualPart(comp_heatsink, "compact_heatsink_fins", COLOR_CNC_AL, metallic=0.88, roughness=0.22),
        VisualPart(step_chalice, "chalice_top_cup", COLOR_DARK_TI, metallic=0.86, roughness=0.28),
        VisualPart(comp_chalice_throat, "chalice_heatbreak", COLOR_DARK_TI, metallic=0.86, roughness=0.28),
        VisualPart(comp_melt, "nickel_melt_core", COLOR_NICKEL_CU, metallic=0.88, roughness=0.24),
        VisualPart(step_sock, "silicone_sock", COLOR_DARK_SOCK, metallic=0.05, roughness=0.85),
        VisualPart(step_nozzle, "v6_nozzle", COLOR_BRASS_NOZZLE, metallic=0.92, roughness=0.22),
        VisualPart(vis_heater_cart, "heater_cartridge", COLOR_STAINLESS, metallic=0.88, roughness=0.24),
        VisualPart(vis_heater_leads, "heater_leads", COLOR_HEATER_LEAD, metallic=0.10, roughness=0.72),
        VisualPart(vis_therm_cart, "thermistor_cartridge", COLOR_STAINLESS, metallic=0.88, roughness=0.24),
        VisualPart(vis_therm_leads, "thermistor_leads", COLOR_THERM_LEAD, metallic=0.10, roughness=0.72),
    ]
    for i, (orig, nrm) in enumerate(screw_specs):
        sc = _make_flush_torx_screw(orig, nrm, head_diam_mm=4.8, head_height_mm=1.3)
        visual_parts.append(VisualPart(sc, f"torx_screw_{i}", COLOR_STAINLESS, metallic=0.92, roughness=0.24))

    spec = BakedAssetSpec(
        category="hotends",
        asset_id="chube_compact",
        name="Luke's Lab Chube Compact (80W Ti Shell)",
        source_cad="Chube_Compact_integration_assembly.step",
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
            "cutouts": "4x Ø12.50 mm @ z=-12.90 mm (45/135/225/315 deg)",
        },
        visual_max_triangles=24000,
    )
    return _bake_or_refresh_visuals(
        spec, regions, visual_parts=visual_parts, assets_root=assets_root, visuals_only=visuals_only
    )


# ---------------------------------------------------------------------------
# 2 & 3. Chube Air & Chube Conduction (Official Rev A STEP Geometry)
# ---------------------------------------------------------------------------


def _build_chube_reva_geometry(
    cad_dir: Path,
    variant: str,
    visuals_only: bool = False,
) -> tuple[list[MaterialRegion], list[VisualPart]]:
    """Build exact Rev A structural shell (Air vs Conduction) + photo-accurate STEP hardware visuals."""
    step_path = _resolve_cad_file(
        cad_dir,
        "chube_integration_reva.step",
        "chube_official/extracted/CHUBE_INTEGRATION_REVA.step",
    )
    asm = b3d.import_step(str(step_path))

    # Align official STEP origin: axis center (-0.2634, -33.7764), top face z=282.2752 -> (0,0,0)
    shift = (0.2634, 33.7764, -282.2752)

    common = _find_child_by_substr(asm, "common_features")
    step_nozzle = _find_child_by_substr(common, "nozzle").translate(shift)
    step_shield_raw = _find_child_by_substr(common, "sheild", "shield").translate(shift)
    # Offset curved stainless heat shield +0.12 mm radially outward from R=12.70 mm Ti tube to prevent Z-fighting
    shield_center_z = -53.93
    step_shield = (
        step_shield_raw.translate((0.0, 0.0, -shield_center_z))
        .scale(12.82 / 12.70)
        .translate((0.0, 0.0, shield_center_z))
    )

    is_air = variant == "air"

    # Exact official Luke's Lab Rev A Ø25.40 x 0.90 mm Titanium monocoque tube (60.00 mm bare)
    ro = 12.70
    ri = 11.80
    body_h = 60.00

    base_tube = b3d.Solid.make_cylinder(ro, body_h, plane=b3d.Plane(origin=(0.0, 0.0, -body_h))).cut(
        b3d.Solid.make_cylinder(ri, body_h + 2.0, plane=b3d.Plane(origin=(0.0, 0.0, -body_h - 1.0)))
    )

    # Exact official Outer_Shell cutouts measured from CHUBE_INTEGRATION_REVA.step:
    # - 3 upper Ø19.0 mm circular cutouts at z = -12.75 mm, angles 90 / -30 / -150 deg
    # - 3 lower Ø19.0 mm circular cutouts at z = -32.75 mm, angles 30 / 150 / -90 deg
    # - 2 Ø1.90 mm alignment pin holes at z = -51.00 mm, angles +90 / -90 deg
    def _radial_hole(angle_deg: float, z_center: float, radius_mm: float) -> Any:
        ca, sa = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
        pln = b3d.Plane(origin=(0.0, 0.0, z_center), z_dir=(ca, sa, 0.0))
        return b3d.Solid.make_cylinder(radius_mm, 18.0, plane=pln)

    upper_cuts = [_radial_hole(a, -12.75, 9.50) for a in (90.0, -30.0, -150.0)]
    lower_cuts = [_radial_hole(a, -32.75, 9.50) for a in (30.0, 150.0, -90.0)]
    pin_holes = [_radial_hole(a, -51.00, 0.95) for a in (90.0, -90.0)]
    ti_tube_vis = base_tube.cut(*upper_cuts, *lower_cuts, *pin_holes).clean()

    # Exact 6x M3 Ø18.0 mm PCD top mounting holes (0/60/120/180/240/300 deg)
    m3_top_holes = [
        b3d.Solid.make_cylinder(
            1.5,
            9.0,
            plane=b3d.Plane(
                origin=(9.0 * math.cos(math.radians(a)), 9.0 * math.sin(math.radians(a)), -8.5)
            ),
        )
        for a in (0.0, 60.0, 120.0, 180.0, 240.0, 300.0)
    ]

    # Helper for authentic Chube racetrack/stadium lower meltblock (13.6 mm wide in X, spanning ±Y
    # between the heater & thermistor bores to meet the lower M3 side fasteners at z=-46/-56 mm;
    # keeps exact ~69 g Air / ~74 g Conduction bare mass instead of over-filling the Ø23.6 bore!)
    def _make_racetrack_block(z_bot: float, height: float, half_x: float = 6.6, r_clip: float = 11.74) -> Any:
        cyl = b3d.Solid.make_cylinder(r_clip, height, plane=b3d.Plane(origin=(0.0, 0.0, z_bot)))
        slab = b3d.Solid.make_box(
            2.0 * half_x, 2.0 * (r_clip + 1.0), height,
            plane=b3d.Plane(origin=(-half_x, -(r_clip + 1.0), z_bot)),
        )
        return cyl.intersect(slab).solids()[0].clean()

    if is_air:
        # Chube Air: Bright machined CNC aluminum 4-fin heatsink (z = 0..-22.50 mm, exact STEP Air_Heatsink_Lower_Surface)
        hs_top = b3d.Solid.make_cylinder(ri - 0.02, 3.8, plane=b3d.Plane(origin=(0.0, 0.0, -3.8)))
        hs_core = b3d.Solid.make_cylinder(5.0, 22.4, plane=b3d.Plane(origin=(0.0, 0.0, -22.5)))
        hs_fins = [
            b3d.Solid.make_cylinder(ri - 0.05, 1.8, plane=b3d.Plane(origin=(0.0, 0.0, z_f)))
            for z_f in (-8.2, -12.9, -17.6, -22.5)
        ]
        heatsink_solid = (
            hs_top.fuse(hs_core, *hs_fins)
            .cut(b3d.Solid.make_cylinder(2.75, 24.0, plane=b3d.Plane(origin=(0.0, 0.0, -23.0))), *m3_top_holes)
            .clean()
        )

        # Chalice - Air Ti heatbreak (exact STEP z = -11.30..-14.80 cup + throat to -18.50)
        chalice_solid = (
            b3d.Solid.make_cylinder(2.75, 3.5, plane=b3d.Plane(origin=(0.0, 0.0, -14.80)))
            .fuse(b3d.Solid.make_cylinder(1.85, 7.5, plane=b3d.Plane(origin=(0.0, 0.0, -21.50))))
            .cut(b3d.Solid.make_cylinder(1.0, 14.0, plane=b3d.Plane(origin=(0.0, 0.0, -22.50))))
            .clean()
        )

        # Electroless-nickel-plated copper 42 mm melt core (z = -18.0..-59.8 mm, Ø10.4 mm upper melt column
        # + slim racetrack lower heater/thermistor block aligned along ±Y -> exact ~69 g bare mass!)
        melt_upper = b3d.Solid.make_cylinder(5.0, 21.5, plane=b3d.Plane(origin=(0.0, 0.0, -39.5)))
        melt_lower = _make_racetrack_block(-59.8, 20.8, half_x=4.7, r_clip=ri - 0.06)
        heater_bore = b3d.Solid.make_cylinder(3.05, 22.0, plane=b3d.Plane(origin=(0.0, -6.80, -60.5)))
        therm_bore = b3d.Solid.make_cylinder(1.55, 17.0, plane=b3d.Plane(origin=(0.0, 6.80, -60.5)))
        melt_core = (
            melt_upper.fuse(melt_lower)
            .cut(
                b3d.Solid.make_cylinder(1.0, 44.0, plane=b3d.Plane(origin=(0.0, 0.0, -61.0))),
                heater_bore,
                therm_bore,
            )
            .clean()
        )
        vis_heater = b3d.Solid.make_cylinder(2.95, 21.0, plane=b3d.Plane(origin=(0.0, -6.80, -60.0)))
        vis_therm = b3d.Solid.make_cylinder(1.48, 16.0, plane=b3d.Plane(origin=(0.0, 6.80, -60.0)))
        hs_label = "air_heatsink_fins"
        heater_label = "heater_cartridge"
    else:
        # Chube Conduction: Solid 8.0 mm CNC aluminum conduction heatsink (z = 0..-8.0 mm, exact STEP Lower_Surface_of_Heatsink)
        heatsink_solid = (
            b3d.Solid.make_cylinder(ri - 0.02, 8.0, plane=b3d.Plane(origin=(0.0, 0.0, -8.0)))
            .cut(
                b3d.Solid.make_cylinder(2.75, 9.5, plane=b3d.Plane(origin=(0.0, 0.0, -8.8))),
                *m3_top_holes,
            )
            .clean()
        )

        # Chalice - Conduction upper cup (0..-3.70) + Chalice_Hex_Conduction (-7.45..-9.88)
        hex_pts = [
            (3.18 * math.cos(math.radians(60.0 * k)), 3.18 * math.sin(math.radians(60.0 * k)), -9.88)
            for k in range(6)
        ]
        hex_wire = b3d.Wire.make_polygon([b3d.Vector(*p) for p in hex_pts] + [b3d.Vector(*hex_pts[0])])
        chalice_hex = b3d.Solid.extrude(b3d.Face(hex_wire), (0.0, 0.0, 2.43))
        chalice_solid = (
            b3d.Solid.make_cylinder(2.75, 3.70, plane=b3d.Plane(origin=(0.0, 0.0, -3.70)))
            .fuse(
                b3d.Solid.make_cylinder(2.00, 7.0, plane=b3d.Plane(origin=(0.0, 0.0, -10.50))),
                chalice_hex,
            )
            .cut(b3d.Solid.make_cylinder(1.0, 12.0, plane=b3d.Plane(origin=(0.0, 0.0, -11.20))))
            .clean()
        )

        # Massive electroless-nickel-plated copper 52 mm melt core (z = -9.88..-59.8 mm) + Supervolcano heater lobe (~74 g bare!)
        melt_upper = b3d.Solid.make_cylinder(5.0, 30.0, plane=b3d.Plane(origin=(0.0, 0.0, -39.9)))
        heater_lobe = b3d.Solid.make_cylinder(3.95, 46.5, plane=b3d.Plane(origin=(0.0, 6.80, -59.8)))
        melt_lower = _make_racetrack_block(-59.8, 20.8, half_x=4.7, r_clip=ri - 0.06)
        heater_bore = b3d.Solid.make_cylinder(3.05, 47.0, plane=b3d.Plane(origin=(0.0, 6.80, -60.5)))
        therm_bore = b3d.Solid.make_cylinder(1.55, 17.0, plane=b3d.Plane(origin=(0.0, -6.80, -60.5)))
        melt_core = (
            melt_upper.fuse(heater_lobe, melt_lower)
            .cut(
                b3d.Solid.make_cylinder(1.0, 52.0, plane=b3d.Plane(origin=(0.0, 0.0, -61.0))),
                heater_bore,
                therm_bore,
            )
            .clean()
        )
        vis_heater = b3d.Solid.make_cylinder(2.95, 46.5, plane=b3d.Plane(origin=(0.0, 6.80, -60.0)))
        vis_therm = b3d.Solid.make_cylinder(1.48, 16.0, plane=b3d.Plane(origin=(0.0, -6.80, -60.0)))
        hs_label = "conduction_heatsink"
        heater_label = "supervolcano_heater"

    regions: list[MaterialRegion] = []
    if not visuals_only:
        # Conformal multi-material Tet4 FEA using the EXACT official Rev A cutout shell!
        top_pins = [
            _make_m3_pin(
                (ro * math.cos(math.radians(a)), ro * math.sin(math.radians(a)), -5.0),
                (7.0 * math.cos(math.radians(a)), 7.0 * math.sin(math.radians(a)), -5.0),
                radius_mm=1.60,
            )
            for a in (30.0, 150.0, -90.0)
        ]
        bot_pins = [
            _make_m3_pin((0.0, -ro, z_p), (0.0, ro, z_p), radius_mm=1.60)
            for z_p in (-46.0, -56.0)
        ]
        bot_collar = b3d.Solid.make_cylinder(ro, 2.6, plane=b3d.Plane(origin=(0.0, 0.0, -body_h))).cut(
            b3d.Solid.make_cylinder(3.1, 4.0, plane=b3d.Plane(origin=(0.0, 0.0, -body_h - 0.5)))
        )

        if is_air:
            fea_hs = (
                b3d.Solid.make_cylinder(ri + 0.15, 6.2, plane=b3d.Plane(origin=(0.0, 0.0, -6.2)))
                .fuse(b3d.Solid.make_cylinder(4.8, 22.0, plane=b3d.Plane(origin=(0.0, 0.0, -22.2))))
                .cut(
                    b3d.Solid.make_cylinder(2.75, 24.0, plane=b3d.Plane(origin=(0.0, 0.0, -23.0))),
                    *m3_top_holes,
                )
                .clean()
            )
            fea_melt = (
                b3d.Solid.make_cylinder(5.0, 21.5, plane=b3d.Plane(origin=(0.0, 0.0, -39.5)))
                .fuse(_make_racetrack_block(-59.5, 19.5, half_x=4.7, r_clip=ri + 0.15))
                .cut(
                    b3d.Solid.make_cylinder(1.0, 44.0, plane=b3d.Plane(origin=(0.0, 0.0, -61.0))),
                    b3d.Solid.make_cylinder(3.0, 21.5, plane=b3d.Plane(origin=(0.0, -6.80, -60.2))),
                    b3d.Solid.make_cylinder(1.5, 16.5, plane=b3d.Plane(origin=(0.0, 6.80, -60.2))),
                )
                .clean()
            )
            fea_ti_shell = ti_tube_vis.fuse(bot_collar, chalice_solid, *top_pins, *bot_pins).clean()
        else:
            fea_hs = (
                b3d.Solid.make_cylinder(ri + 0.15, 8.0, plane=b3d.Plane(origin=(0.0, 0.0, -8.0)))
                .cut(
                    b3d.Solid.make_cylinder(2.75, 9.5, plane=b3d.Plane(origin=(0.0, 0.0, -8.8))),
                    *m3_top_holes,
                )
                .clean()
            )
            fea_melt = (
                b3d.Solid.make_cylinder(5.0, 30.0, plane=b3d.Plane(origin=(0.0, 0.0, -39.9)))
                .fuse(
                    b3d.Solid.make_cylinder(3.95, 46.5, plane=b3d.Plane(origin=(0.0, 6.80, -59.8))),
                    _make_racetrack_block(-59.5, 19.5, half_x=4.7, r_clip=ri + 0.15),
                )
                .cut(
                    b3d.Solid.make_cylinder(1.0, 52.0, plane=b3d.Plane(origin=(0.0, 0.0, -61.0))),
                    b3d.Solid.make_cylinder(3.0, 46.8, plane=b3d.Plane(origin=(0.0, 6.80, -60.2))),
                    b3d.Solid.make_cylinder(1.5, 16.5, plane=b3d.Plane(origin=(0.0, -6.80, -60.2))),
                )
                .clean()
            )
            cond_top_ring = b3d.Solid.make_cylinder(ro, 8.0, plane=b3d.Plane(origin=(0.0, 0.0, -8.0))).cut(
                b3d.Solid.make_cylinder(ri, 9.0, plane=b3d.Plane(origin=(0.0, 0.0, -8.5)))
            )
            fea_ti_shell = ti_tube_vis.fuse(cond_top_ring, bot_collar, chalice_solid, *top_pins, *bot_pins).clean()

        fea_nozzle = _make_v6_nozzle_solid(tip_z_mm=-66.00, total_len_mm=12.5)
        regions = _non_overlapping_regions(
            [
                (fea_nozzle, "copper_c110", "v6_nozzle", COLOR_BRASS_NOZZLE, 0.90, 0.25),
                (fea_ti_shell, "titanium_grade5", "ti_monocoque_shell", COLOR_CHUBE_TI, 0.85, 0.32),
                (fea_hs, "aluminum_6061_t6", hs_label, COLOR_CNC_AL, 0.88, 0.22),
                (fea_melt, "copper_c110", "copper_melt_core", COLOR_NICKEL_CU, 0.88, 0.24),
            ]
        )

    # Flush-seated M3 T10 Torx countersunk screws at exact CHUBE_INTEGRATION_REVA.step positions:
    # 3 upper radial @ z=-5.0 (30/150/-90 deg), 4 lower radial @ z=-46/-56 (±90 deg), 2 bottom @ z=-60 (0, ±6.80)
    r_screw_top = ro + 0.16
    r_screw_bot = ro + 0.36  # sits flush on outer face of 0.24 mm stainless heat shield!
    screw_specs = [
        (
            (r_screw_top * math.cos(math.radians(a)), r_screw_top * math.sin(math.radians(a)), -5.00),
            (math.cos(math.radians(a)), math.sin(math.radians(a)), 0.0),
        )
        for a in (30.0, 150.0, -90.0)
    ] + [
        ((0.0, r_screw_bot, -46.00), (0.0, 1.0, 0.0)),
        ((0.0, -r_screw_bot, -46.00), (0.0, -1.0, 0.0)),
        ((0.0, r_screw_bot, -56.00), (0.0, 1.0, 0.0)),
        ((0.0, -r_screw_bot, -56.00), (0.0, -1.0, 0.0)),
        ((0.0, 6.80, -60.76 - 0.16), (0.0, 0.0, -1.0)),
        ((0.0, -6.80, -60.76 - 0.16), (0.0, 0.0, -1.0)),
    ]

    visual_parts: list[VisualPart] = [
        VisualPart(ti_tube_vis, "ti_monocoque_shell", COLOR_CHUBE_TI, metallic=0.84, roughness=0.32),
        VisualPart(heatsink_solid, hs_label, COLOR_CNC_AL, metallic=0.88, roughness=0.22),
        VisualPart(chalice_solid, "chalice_heatbreak", COLOR_DARK_TI, metallic=0.86, roughness=0.28),
        VisualPart(melt_core, "copper_melt_core", COLOR_NICKEL_CU, metallic=0.88, roughness=0.24),
        VisualPart(vis_heater, heater_label, COLOR_STAINLESS, metallic=0.88, roughness=0.24),
        VisualPart(vis_therm, "thermistor_cartridge", COLOR_DARK_TI, metallic=0.85, roughness=0.30),
        VisualPart(step_nozzle, "e3d_v6_nozzle", COLOR_BRASS_NOZZLE, metallic=0.92, roughness=0.22),
        VisualPart(step_shield, "heat_shield", COLOR_STAINLESS, metallic=0.90, roughness=0.22),
    ]
    for i, (orig, nrm) in enumerate(screw_specs):
        sc = _make_flush_torx_screw(orig, nrm, head_diam_mm=5.2, head_height_mm=1.35)
        visual_parts.append(VisualPart(sc, f"torx_screw_{i}", COLOR_STAINLESS, metallic=0.92, roughness=0.24))

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
            select_z_annulus(0.0, r_min_mm=5.5, r_max_mm=12.75, tol_mm=0.45),
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
            select_z_annulus(0.0, r_min_mm=5.5, r_max_mm=12.75, tol_mm=0.45),
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
# 6B. Microwaved Bowden Toolhead — Chube Edition (SLM AlSi10Mg / Ti-6Al-4V)
# ---------------------------------------------------------------------------


def bake_microwaved_bowden_chube(
    cad_dir: Path,
    assets_root: Path | None = None,
    visuals_only: bool = False,
) -> BakedAssetResult:
    """Bake Microwaved Bowden Toolhead — Chube V1 SLM chassis from MBT_Chube_V1.STEP.

    Excludes MGN12H carriage / Monolith belt clamp (drawn by viewport) and excludes the
    embedded Chube integration reference (viewport attaches the live selected Chube Air /
    Conduction hotend flush at Port 1 ``[0.0, -12.76, +27.50]`` mm).
    """
    step_path = _resolve_cad_file(cad_dir, "mbt_chube_v1.step", "MBT_Chube_V1.STEP")
    asm = b3d.import_step(str(step_path))

    def _to_printsim(shape: Any) -> Any:
        # CAD frame: +X right, +Y up, +Z front (carriage rear bolt face at z=7.24)
        # PrintSim canonical frame: +X right, +Y back (-z_cad), +Z up (+y_cad), Port 0 at [0,0,0]
        return shape.translate((0.0, 0.0, -7.24)).rotate(b3d.Axis.X, 90.0)

    main_body_raw = _find_child_by_substr(asm, "main_body")
    main_body = _to_printsim(main_body_raw)

    def _port0_mgn12_bolts(coords_mm: np.ndarray) -> np.ndarray:
        on_face = np.abs(coords_mm[:, 1]) <= 0.65
        near_bolt = np.zeros(len(coords_mm), dtype=bool)
        for bx in (-10.0, 10.0):
            for bz in (-10.0, 10.0):
                near_bolt |= np.hypot(coords_mm[:, 0] - bx, coords_mm[:, 2] - bz) <= 4.2
        return on_face & near_bolt

    # Exact 6x M3 Ø18.0 mm PCD Chube mounting face at Z = +27.50 mm, centered at (0.0, -12.76)
    chube_bolt_xy = [
        (9.0 * math.cos(math.radians(a)), -12.76 + 9.0 * math.sin(math.radians(a)))
        for a in (0.0, 60.0, 120.0, 180.0, 240.0, 300.0)
    ]

    def _port1_chube_6bolts(coords_mm: np.ndarray) -> np.ndarray:
        on_flange = np.abs(coords_mm[:, 2] - 27.50) <= 0.65
        near_bolt = np.zeros(len(coords_mm), dtype=bool)
        for bx, by in chube_bolt_xy:
            near_bolt |= np.hypot(coords_mm[:, 0] - bx, coords_mm[:, 1] - by) <= 4.0
        return on_flange & near_bolt

    # Build authentic multi-part visual.glb excluding MGN12H carriage, Monolith belt clamp,
    # embedded Chube integration reference, and Beacon's Optimal_Spacing keep-out block.
    visual_parts: list[VisualPart] = [
        VisualPart(main_body, "mbt_chube_slm_chassis", COLOR_SLM_AL, metallic=0.86, roughness=0.30),
    ]

    def _collect_beacon_parts(node: Any) -> None:
        lbl = getattr(node, "label", "") or ""
        low = lbl.lower()
        if "optimal_spacing" in low:
            return
        children = getattr(node, "children", None) or []
        if children:
            for sub in children:
                _collect_beacon_parts(sub)
        elif node.solids():
            t_node = _to_printsim(node)
            if "pcb" in low:
                visual_parts.append(VisualPart(t_node, "beacon_pcb", COLOR_PCB_DARK, metallic=0.18, roughness=0.55))
            elif "electronics" in low:
                visual_parts.append(VisualPart(t_node, "beacon_electronics", "#334155", metallic=0.30, roughness=0.50))
            else:
                visual_parts.append(VisualPart(t_node, f"beacon_{len(visual_parts)}", "#475569", metallic=0.25, roughness=0.50))

    for ch in asm.children:
        lbl = getattr(ch, "label", "") or ""
        low = lbl.lower()
        if (
            "main_body" in low
            or "chube_integration" in low
            or "slm_monolith" in low
            or "mgn12" in low
        ):
            continue
        if "beacon" in low:
            _collect_beacon_parts(ch)
            continue
        t_ch = _to_printsim(ch)
        if "ecas" in low:
            visual_parts.append(VisualPart(t_ch, "ecas_fitting", COLOR_COPPER, metallic=0.88, roughness=0.26))
        elif "m3" in low or "screw" in low or "fhcs" in low or "button_head" in low:
            visual_parts.append(VisualPart(t_ch, f"fastener_{len(visual_parts)}", COLOR_STAINLESS, metallic=0.92, roughness=0.24))

    spec = BakedAssetSpec(
        category="toolheads",
        asset_id="microwaved_bowden_chube",
        name="Microwaved Bowden Chube V1 (SLM AlSi10Mg / Ti)",
        source_cad="MBT_Chube_V1.STEP",
        source_url="https://github.com/micreathonn/Microwaved-Bowden-Toolhead",
        reference_material_id="aluminum_6061_t6",
        cad_trust_pct=86,
        port0=PortDefinition((0.0, 0.0, 0.0), _port0_mgn12_bolts),
        port1=PortDefinition((0.0, -12.76, 27.50), _port1_chube_6bolts),
        n_internal_modes=8,
        mesh_size_mm=1.8,
        is_multi_material=False,  # Scales across Al 6061 / AlSi10Mg / Ti-6Al-4V / polymers!
        is_fdm_anisotropic=False,
        part_recipe={
            "chassis_part": "Main_Body^Microwaved_Bowden_Toolhead_-_Chube",
            "carriage_bolt_pattern": "4x M3 (20x20 mm MGN12H)",
            "hotend_mount_interface": "Chube 6-bolt M3 Ø18 mm PCD conduction boss",
            "port1_offset_mm": [0.0, -12.76, 27.50],
        },
        visual_max_triangles=24000,
        fea_surface_max_triangles=60000,
    )
    regions = (
        []
        if visuals_only
        else [
            MaterialRegion(
                main_body,
                "aluminum_6061_t6",
                "mbt_chube_slm_chassis",
                fdm_anisotropic=False,
                color_hex=COLOR_SLM_AL,
            )
        ]
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
            "microwaved_bowden_chube",
            lambda: bake_microwaved_bowden_chube(args.cad_dir, args.out_dir, visuals_only=vo),
        ),
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
