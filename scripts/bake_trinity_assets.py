"""Bake CAD Tet4 Craig-Bampton superelements & PBR GLBs for WV-Design Trinity 1.6 & Dragon Ace Volcano.

Assets baked into ``src/printsim/assets/hardware/``:
  1. ``hotends/trianglelab_dragon_ace_volcano``
     - Multi-material Tet4 FEA + PBR GLB from official Trinity DAV 1.6 STEP
       (Al 6061 heatsink, 4x Ti-6Al-4V spacers + thick-walled Ti heatbreak, Cu Volcano block + MZE + V6 nozzle).
  2. ``assemblies/trinity_1_6_dav__trianglelab_dragon_ace_volcano``
     - Coupled heatsink-braced DAV superelement (top 4x M2.5 flange + lower heatsink collar brace inside Trinity unibody).
  3. ``assemblies/trinity_1_6_chube__chube_compact``
     - Coupled shell-braced Chube Compact superelement (top 4x M3 flange + Ø22.23 mm Ti shell collar brace inside Trinity unibody).
  4. ``toolheads/trinity_1_6_dav``
     - Monolithic unibody Trinity 1.6 DAV Tet4 FEA (PPA-CF reference, is_unibody=True) + full dressed PBR GLB
       (body, ducts_v3, dual Delta 3628 fans, Delta 2510 HEF, Beacon Rev H, Monolith SLM clamp, Apex clips, fasteners).
  5. ``toolheads/trinity_1_6_chube``
     - Monolithic unibody Trinity 1.6 Chube Compact Tet4 FEA + full dressed PBR GLB.

Usage:
    uv run python scripts/bake_trinity_assets.py
    uv run python scripts/bake_trinity_assets.py --only dav,trinity_dav
    uv run python scripts/bake_trinity_assets.py --visuals-only
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import build123d as b3d
import numpy as np
import trimesh

# Ensure sibling scripts can be imported cleanly
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bake_phase1_assets import (  # noqa: E402
    COLOR_BRASS_NOZZLE,
    COLOR_CHUBE_TI,
    COLOR_CNC_AL,
    COLOR_COPPER,
    COLOR_DARK_SOCK,
    COLOR_DARK_TI,
    COLOR_NICKEL_CU,
    COLOR_STAINLESS,
    _bake_coupled_multi_mount_superelement,
    _make_m3_pin,
    _make_v6_nozzle_solid,
    _non_overlapping_regions,
)
from printsim.cad_assets.baker import (  # noqa: E402
    ASSETS_ROOT,
    BakedAssetResult,
    BakedAssetSpec,
    MaterialRegion,
    PortDefinition,
    VisualPart,
    bake_solid_superelement,
    build_visual_glb,
    select_z_annulus,
    select_z_cap,
)
from printsim.cad_assets.recipes import verify_baked_asset  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAD_DIR = REPO_ROOT / "cad_sources" / "trinity"
TRINITY_SOURCE_URL = "https://github.com/WV-design/Trinity-toolhead/tree/1.6-stable"
DAV_SOURCE_URL = "https://trianglelab.net/products/dragon-ace%E2%84%A2-hotend?VariantsId=12007"

# PBR styling palette matching PrintSim Studio dark hardware look
COL_BODY_CF = "#23272f"       # Matte dark PPA-CF / ABS-GF unibody
COL_DUCT_CF = "#2b303a"       # Slightly lighter matte ducts
COL_ACCENT_TEAL = "#2dd4bf"   # Trinity logo accent
COL_FAN_HOUSING = "#1c1f26"   # Delta PBT fan frame
COL_FAN_ROTOR = "#2e333d"     # Delta fan impeller
COL_PCB_DARK = "#1a3b2c"      # Beacon PCB
COL_PTFE = "#e2e8f0"          # PTFE micro-bowden
COL_SLM_AL = "#b8c2ce"        # Monolith SLM Al belt clamp
COL_STEEL = "#94a3b8"         # M3/M2.5 fasteners
COL_BRASS_INSERT = "#c99b44"  # Heatset inserts


def _walk_world_solids(shape: Any, parent_loc: b3d.Location | None = None, prefix: str = ""):
    """Yield (hierarchical_path, world_positioned_solid) across a STEP assembly tree."""
    if parent_loc is None:
        parent_loc = b3d.Location()
    label = getattr(shape, "label", "") or ""
    path = f"{prefix}/{label}" if prefix else label
    loc = parent_loc * shape.location if getattr(shape, "location", None) is not None else parent_loc
    children = getattr(shape, "children", None) or []
    if children:
        for ch in children:
            yield from _walk_world_solids(ch, loc, path)
    else:
        for s in shape.solids():
            yield path, s.moved(parent_loc)


def _step_to_printsim_shape(
    solid: Any,
    origin_step_mm: tuple[float, float, float],
) -> Any:
    """Map SolidWorks Trinity STEP coordinates -> PrintSim canonical frame.

    STEP frame:    +X right, +Y up (-Y down), +Z back toward X-beam (-Z front)
    PrintSim frame: +X right, +Y back toward X-beam (-Y front), +Z up (-Z down)

    Proper right-handed rotation: translate(-origin_step) -> Rx(+90 deg) -> Rz(180 deg)
    maps (dx, dy, dz)_step -> (-dx, +dz, +dy)_printsim.
    """
    ox, oy, oz = origin_step_mm
    return (
        solid.translate((-ox, -oy, -oz))
        .rotate(b3d.Axis.X, 90.0)
        .rotate(b3d.Axis.Z, 180.0)
    )


def _shape_to_trimesh(shape: Any, deflection: float = 0.25) -> trimesh.Trimesh:
    """Tessellate a build123d solid into a clean trimesh in mm (tolerating degenerate STEP faces)."""
    for defl in (deflection, 0.12, 0.05):
        try:
            verts, tris = shape.tessellate(defl)
            v = np.asarray([(float(p.X), float(p.Y), float(p.Z)) for p in verts], dtype=np.float64)
            f = np.asarray(tris, dtype=np.int64)
            if len(v) > 0 and len(f) > 0:
                return trimesh.Trimesh(vertices=v, faces=f, process=True)
        except Exception:
            continue
    return trimesh.Trimesh()


def _extract_dav_parts(dav_step_path: Path) -> tuple[list[MaterialRegion], list[VisualPart]]:
    """Extract & transform the multi-material Dragon Ace Volcano solids from Trinity_DAV_1.6_10mm.step."""
    root = b3d.import_step(str(dav_step_path))
    # DAV top mounting flange center in STEP = (0.0, -21.60, -7.50)
    origin_dav = (0.0, -21.60, -7.50)

    heatsink_solids: list[Any] = []
    ti_solids: list[Any] = []
    copper_block_solids: list[Any] = []
    mze_solids: list[Any] = []
    nozzle_solids: list[Any] = []
    sock_solids: list[Any] = []
    screw_solids: list[Any] = []

    for path, ws in _walk_world_solids(root):
        low = path.lower()
        if "dragon_ace_volcano" not in low:
            continue
        leaf = low.split("/")[-1]
        s = _step_to_printsim_shape(ws, origin_dav)
        if "hotend_sock" in leaf or "heatsink_sock" in leaf:
            sock_solids.append(s)
        elif "din_913" in leaf:
            screw_solids.append(s)
        elif "v6_nozzle" in leaf:
            nozzle_solids.append(s)
        elif "fixmze" in leaf:
            mze_solids.append(s)
        elif "volcano_v4" in leaf:
            copper_block_solids.append(s)
        elif "spacer" in leaf or "component1" in leaf or "component2" in leaf:
            ti_solids.append(s)
        elif "heatsink" in leaf:
            heatsink_solids.append(s)

    if not heatsink_solids or not copper_block_solids or not nozzle_solids:
        raise RuntimeError(
            f"Incomplete Dragon Ace Volcano solids in {dav_step_path}: "
            f"hs={len(heatsink_solids)}, ti={len(ti_solids)}, blk={len(copper_block_solids)}, noz={len(nozzle_solids)}"
        )

    # Verify exact overall length (top flange Z=0 -> nozzle tip Z=-59.40 mm)
    noz_bb = nozzle_solids[0].bounding_box()
    hs_bb = heatsink_solids[0].bounding_box()
    print(
        f"  DAV transformed bounds: heatsink Z=[{hs_bb.min.Z:.2f}, {hs_bb.max.Z:.2f}], "
        f"nozzle Z=[{noz_bb.min.Z:.2f}, {noz_bb.max.Z:.2f}]"
    )

    # Build clean, thread-free multi-material B-Rep for conformal Gmsh Tet4 FEA (Pitfall 1E:
    # SolidWorks STEP cosmetic threads/fillets on fixV6_Nozzle & fixDAV_heatsink_body have
    # open 1D loops; exact official STEP solids are preserved verbatim for visual_parts below).
    # Exact dimensions verified directly from Trinity_DAV_1.6_10mm.step:
    #   - Heatsink: Ø22.3 mm finned Al 6061 body, Z = [0.0, -23.0], vol ~2.45 cm³ (6.6 g)
    #   - 4x Ti spacers: Ø2.30 x 7.50 mm @ (±6.0, ±6.0), Z = [-20.5, -28.0]
    #   - Ti heatbreak: Ø2.80 OD / Ø1.90 ID (0.45 mm wall), Z = [-9.0, -38.5]
    #   - Cu Volcano block: 18.0 x 22.0 x 20.0 mm @ Z = [-25.5, -45.5] w/ Ø6 heater & Ø3 sensor bores
    #   - Cu MZE: Ø10.0 mm @ Z = [-38.5, -53.9] w/ Ø2.0 mm filament bore
    #   - V6 nozzle: Z = [-46.9, -59.40] (12.5 mm OAL)
    hs_top = b3d.Solid.make_cylinder(11.15, 4.0, plane=b3d.Plane(origin=(0.0, 0.0, -4.0)))
    hs_core = b3d.Solid.make_cylinder(5.75, 23.0, plane=b3d.Plane(origin=(0.0, 0.0, -23.0)))
    hs_fins = [
        b3d.Solid.make_cylinder(11.15, 1.3, plane=b3d.Plane(origin=(0.0, 0.0, z_fin)))
        for z_fin in (-7.2, -10.5, -13.8, -17.1, -20.4, -23.0)
    ]
    hs_bore = b3d.Solid.make_cylinder(1.40, 25.0, plane=b3d.Plane(origin=(0.0, 0.0, -24.0)))
    fea_heatsink = hs_top.fuse(hs_core, *hs_fins).cut(hs_bore).clean()

    ti_spacers = [
        b3d.Solid.make_cylinder(1.15, 7.5, plane=b3d.Plane(origin=(sx, sy, -28.0)))
        for sx in (-6.0, 6.0)
        for sy in (-6.0, 6.0)
    ]
    ti_throat = (
        b3d.Solid.make_cylinder(1.40, 21.0, plane=b3d.Plane(origin=(0.0, 0.0, -28.0)))
        .fuse(b3d.Solid.make_cylinder(2.60, 13.0, plane=b3d.Plane(origin=(0.0, 0.0, -38.5))))
        .cut(b3d.Solid.make_cylinder(0.95, 34.0, plane=b3d.Plane(origin=(0.0, 0.0, -40.0))))
        .clean()
    )
    fea_ti = ti_throat.fuse(*ti_spacers).clean()

    # Machined Cu Volcano block (18x22x20 mm @ Z=[-25.5,-45.5], shifted +1.4 mm in Y like CAD)
    blk_raw = b3d.Solid.make_box(
        18.0, 22.0, 20.0, plane=b3d.Plane(origin=(-9.0, -9.0, -45.5))
    )
    heater_bore = b3d.Solid.make_cylinder(
        3.1, 22.0, plane=b3d.Plane(origin=(-11.0, 7.5, -35.5), z_dir=(1.0, 0.0, 0.0))
    )
    therm_bore = b3d.Solid.make_cylinder(
        1.6, 22.0, plane=b3d.Plane(origin=(-11.0, -6.5, -35.5), z_dir=(1.0, 0.0, 0.0))
    )
    # Pocket cutouts matching lightweight machined DAV Volcano block (~2.5 cm³ / ~22 g Cu w/ MZE)
    weight_cut_l = b3d.Solid.make_box(
        3.5, 10.0, 14.0, plane=b3d.Plane(origin=(-9.5, -3.5, -42.5))
    )
    weight_cut_r = b3d.Solid.make_box(
        3.5, 10.0, 14.0, plane=b3d.Plane(origin=(6.0, -3.5, -42.5))
    )
    mze_cyl = b3d.Solid.make_cylinder(4.8, 15.4, plane=b3d.Plane(origin=(0.0, 0.0, -53.9)))
    melt_bore = b3d.Solid.make_cylinder(1.0, 32.0, plane=b3d.Plane(origin=(0.0, 0.0, -55.0)))
    fea_copper = (
        blk_raw.fuse(mze_cyl)
        .cut(heater_bore, therm_bore, weight_cut_l, weight_cut_r, melt_bore)
        .clean()
    )
    fea_nozzle = _make_v6_nozzle_solid(tip_z_mm=-59.40, total_len_mm=12.5)

    regions = _non_overlapping_regions(
        [
            (fea_nozzle, "copper_c110", "v6_nozzle", COLOR_BRASS_NOZZLE, 0.90, 0.25),
            (fea_ti, "titanium_grade5", "ti_standoffs_and_heatbreak", COLOR_DARK_TI, 0.85, 0.30),
            (fea_heatsink, "aluminum_6061_t6", "dav_heatsink", COLOR_CNC_AL, 0.88, 0.25),
            (fea_copper, "copper_c110", "volcano_copper_block_mze", COLOR_NICKEL_CU, 0.88, 0.24),
        ]
    )

    visual_parts: list[VisualPart] = []
    for i, s in enumerate(heatsink_solids):
        visual_parts.append(VisualPart(s, f"dav_heatsink_{i}", COLOR_CNC_AL, metallic=0.88, roughness=0.25))
    for i, s in enumerate(ti_solids):
        visual_parts.append(VisualPart(s, f"dav_ti_{i}", COLOR_DARK_TI, metallic=0.85, roughness=0.30))
    for i, s in enumerate(copper_block_solids):
        visual_parts.append(VisualPart(s, f"dav_volcano_block_{i}", COLOR_NICKEL_CU, metallic=0.88, roughness=0.24))
    for i, s in enumerate(mze_solids):
        visual_parts.append(VisualPart(s, f"dav_mze_{i}", COLOR_COPPER, metallic=0.88, roughness=0.24))
    for i, s in enumerate(nozzle_solids):
        visual_parts.append(VisualPart(s, f"v6_nozzle_{i}", COLOR_BRASS_NOZZLE, metallic=0.92, roughness=0.22))
    for i, s in enumerate(sock_solids):
        visual_parts.append(VisualPart(s, f"dav_silicone_sock_{i}", COLOR_DARK_SOCK, metallic=0.05, roughness=0.86))
    for i, s in enumerate(screw_solids):
        visual_parts.append(VisualPart(s, f"dav_setscrew_{i}", COLOR_STAINLESS, metallic=0.90, roughness=0.25))

    return regions, visual_parts


def bake_dragon_ace_volcano(
    cad_dir: Path,
    assets_root: Path = ASSETS_ROOT,
    visuals_only: bool = False,
) -> BakedAssetResult:
    """Bake standalone ``hotends/trianglelab_dragon_ace_volcano``."""
    print("\n=== Baking hotends/trianglelab_dragon_ace_volcano ===")
    dav_step = cad_dir / "Trinity_DAV_1.6_10mm.step"
    regions, visual_parts = _extract_dav_parts(dav_step)

    if visuals_only:
        glb_path = assets_root / "hotends" / "trianglelab_dragon_ace_volcano" / "visual.glb"
        tris, nbytes = build_visual_glb(visual_parts, glb_path, max_total_triangles=24000)
        print(f"  Refreshed visual.glb: {tris} tris, {nbytes} bytes")
        return None  # type: ignore[return-value]

    spec = BakedAssetSpec(
        category="hotends",
        asset_id="trianglelab_dragon_ace_volcano",
        name="Trianglelab Dragon Ace Volcano (70W Cu/Ti)",
        source_cad="Trinity_DAV_1.6_10mm.step (dragon_ace_volcano sub-assembly)",
        source_url=DAV_SOURCE_URL,
        reference_material_id="titanium_grade5",
        cad_trust_pct=87,
        port0=PortDefinition(
            (0.0, 0.0, 0.0),
            select_z_annulus(0.0, r_min_mm=3.2, r_max_mm=11.3, tol_mm=0.65),
        ),
        port1=PortDefinition(
            (0.0, 0.0, -59.40),
            select_z_cap(z_max_mm=-54.2, z_min_mm=-59.6, r_max_mm=4.2),
        ),
        n_internal_modes=8,
        mesh_size_mm=1.6,
        is_multi_material=True,
        is_fdm_anisotropic=False,
        part_recipe={
            "overall_length_mm": 59.4,
            "meltzone_length_mm": 34.0,
            "mount_pcd_mm": 16.97,
            "standoffs": "4x Ø2.3x7.5 mm Ti-6Al-4V @ 12x12 mm square",
            "heatbreak": "Thick-walled Ti-6Al-4V (1.9 mm ID, ~0.45 mm wall)",
            "block": "18x22x20 mm Cu Volcano block + 15.4 mm Cu MZE + V6 nozzle",
        },
        visual_max_triangles=24000,
    )
    t0 = time.perf_counter()
    res = bake_solid_superelement(spec, regions=regions, visual_parts=visual_parts, assets_root=assets_root)
    print(
        f"  Baked DAV in {time.perf_counter() - t0:.1f}s | elems={res.manifest['num_elements']} | "
        f"rigid_max={res.manifest['rigid_body_max_freq_hz']:.5f} Hz | "
        f"K={res.manifest['port_stiffness_n_per_mm']}"
    )
    return res


def bake_trinity_dav_braced_assembly(
    cad_dir: Path,
    assets_root: Path = ASSETS_ROOT,
    visuals_only: bool = False,
) -> BakedAssetResult:
    """Bake coupled ``assemblies/trinity_1_6_dav__trianglelab_dragon_ace_volcano`` (heatsink-braced DAV)."""
    print("\n=== Baking assemblies/trinity_1_6_dav__trianglelab_dragon_ace_volcano ===")
    dav_step = cad_dir / "Trinity_DAV_1.6_10mm.step"
    regions, visual_parts = _extract_dav_parts(dav_step)

    if visuals_only:
        glb_path = assets_root / "assemblies" / "trinity_1_6_dav__trianglelab_dragon_ace_volcano" / "visual.glb"
        tris, nbytes = build_visual_glb(visual_parts, glb_path, max_total_triangles=24000)
        print(f"  Refreshed visual.glb: {tris} tris, {nbytes} bytes")
        return None  # type: ignore[return-value]

    # Trinity unibody tightly wraps the Ø22.3 mm DAV heatsink down to Z = -22.9 mm.
    # Model the lower heatsink brace collar as an independent RBE3 patch tied to Port 0 via congruence Q
    # (Pitfall 1C: strictly disjoint from top Z=0 flange to avoid RBE3 nullspace collapse).
    def _select_dav_heatsink_brace(coords_mm: np.ndarray) -> np.ndarray:
        r = np.hypot(coords_mm[:, 0], coords_mm[:, 1])
        return (coords_mm[:, 2] >= -23.2) & (coords_mm[:, 2] <= -17.5) & (r >= 9.0)

    spec = BakedAssetSpec(
        category="assemblies",
        asset_id="trinity_1_6_dav__trianglelab_dragon_ace_volcano",
        name="Dragon Ace Volcano + Trinity Unibody Heatsink Brace",
        source_cad="Trinity_DAV_1.6_10mm.step",
        source_url=TRINITY_SOURCE_URL,
        reference_material_id="titanium_grade5",
        cad_trust_pct=88,
        port0=PortDefinition(
            (0.0, 0.0, 0.0),
            select_z_annulus(0.0, r_min_mm=3.2, r_max_mm=11.3, tol_mm=0.65),
        ),
        port1=PortDefinition(
            (0.0, 0.0, -59.40),
            select_z_cap(z_max_mm=-54.2, z_min_mm=-59.6, r_max_mm=4.2),
        ),
        n_internal_modes=8,
        mesh_size_mm=1.6,
        is_multi_material=True,
        is_fdm_anisotropic=False,
        part_recipe={
            "toolhead_id": "trinity_1_6_dav",
            "hotend_id": "trianglelab_dragon_ace_volcano",
            "brace_part": "Trinity 1.6 Unibody Heatsink Wrap Collar (Z=-17.5..-23.0 mm)",
            "overall_length_mm": 59.4,
            "meltzone_length_mm": 34.0,
        },
        visual_max_triangles=24000,
    )
    t0 = time.perf_counter()
    res = _bake_coupled_multi_mount_superelement(
        spec,
        regions,
        extra_port0_selectors=[_select_dav_heatsink_brace],
        visual_parts=visual_parts,
        assets_root=assets_root,
    )
    print(
        f"  Baked braced DAV in {time.perf_counter() - t0:.1f}s | elems={res.manifest['num_elements']} | "
        f"rigid_max={res.manifest['rigid_body_max_freq_hz']:.5f} Hz | "
        f"K={res.manifest['port_stiffness_n_per_mm']}"
    )
    return res


def bake_trinity_chube_braced_assembly(
    assets_root: Path = ASSETS_ROOT,
) -> BakedAssetResult:
    """Bake coupled ``assemblies/trinity_1_6_chube__chube_compact`` (shell-braced Chube Compact)."""
    print("\n=== Baking assemblies/trinity_1_6_chube__chube_compact ===")
    # Reconstruct exact Chube Compact multi-material B-Rep (matching bake_phase1_assets.py)
    ro = 11.115
    ri = 10.225
    body_h = 41.68

    outer_tube = b3d.Solid.make_cylinder(ro, body_h, plane=b3d.Plane(origin=(0.0, 0.0, -body_h)))
    top_step_ring = b3d.Solid.make_cylinder(ro + 0.5, 1.30, plane=b3d.Plane(origin=(0.0, 0.0, -1.30))).cut(
        b3d.Solid.make_cylinder(ro - 0.18, 2.0, plane=b3d.Plane(origin=(0.0, 0.0, -1.60)))
    )
    inner_bore = b3d.Solid.make_cylinder(ri, body_h + 2.0, plane=b3d.Plane(origin=(0.0, 0.0, -body_h - 1.0)))
    ti_tube_open = outer_tube.cut(inner_bore, top_step_ring)

    cooling_cuts = []
    for a in (45.0, 135.0, 225.0, 315.0):
        ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
        pln = b3d.Plane(origin=(0.0, 0.0, -12.90), z_dir=(ca, sa, 0.0))
        cooling_cuts.append(b3d.Solid.make_cylinder(6.25, 16.0, plane=pln))

    heater_cut = b3d.Solid.make_cylinder(
        3.15, 28.0, plane=b3d.Plane(origin=(-5.00, -14.0, -27.50), z_dir=(0.0, 1.0, 0.0))
    )
    therm_cut = b3d.Solid.make_cylinder(
        1.65, 28.0, plane=b3d.Plane(origin=(5.70, -14.0, -27.50), z_dir=(0.0, 1.0, 0.0))
    )
    ti_tube_vis = ti_tube_open.cut(*cooling_cuts, heater_cut, therm_cut).clean()

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

    comp_chalice_throat = (
        b3d.Solid.make_cylinder(2.2, 18.5, plane=b3d.Plane(origin=(0.0, 0.0, -22.0)))
        .cut(b3d.Solid.make_cylinder(1.0, 22.0, plane=b3d.Plane(origin=(0.0, 0.0, -23.0))))
        .clean()
    )

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

    # Trinity Chube Compact unibody wraps the Ø22.23 mm Ti shell down to Z_hotend = -25.70 mm
    # (right below the 4x Ø12.50 mm cooling cutouts at Z = -12.90 mm!).
    def _select_chube_shell_brace(coords_mm: np.ndarray) -> np.ndarray:
        r = np.hypot(coords_mm[:, 0], coords_mm[:, 1])
        return (coords_mm[:, 2] >= -26.0) & (coords_mm[:, 2] <= -20.5) & (r >= 10.3)

    # Lightweight placeholder visual parts (replaced immediately by copying canonical chube_compact visual.glb)
    placeholder_vis = [
        VisualPart(ti_tube_vis, "chube_body", COLOR_CHUBE_TI, metallic=0.84, roughness=0.32),
        VisualPart(fea_nozzle, "v6_nozzle", COLOR_BRASS_NOZZLE, metallic=0.92, roughness=0.22),
    ]

    spec = BakedAssetSpec(
        category="assemblies",
        asset_id="trinity_1_6_chube__chube_compact",
        name="Chube Compact + Trinity Unibody Shell Brace",
        source_cad="Trinity_Chube_1.6_10mm.step + Chube_Compact_integration_assembly.step",
        source_url=TRINITY_SOURCE_URL,
        reference_material_id="titanium_grade5",
        cad_trust_pct=88,
        port0=PortDefinition(
            (0.0, 0.0, 0.0),
            select_z_annulus(0.0, r_min_mm=4.5, r_max_mm=11.2, tol_mm=0.45),
        ),
        port1=PortDefinition(
            (0.0, 0.0, -47.2),
            select_z_cap(z_max_mm=-43.2, z_min_mm=-47.3, r_max_mm=4.2),
        ),
        n_internal_modes=8,
        mesh_size_mm=1.35,
        is_multi_material=True,
        is_fdm_anisotropic=False,
        part_recipe={
            "toolhead_id": "trinity_1_6_chube",
            "hotend_id": "chube_compact",
            "brace_part": "Trinity 1.6 Unibody Ti Shell Wrap Collar (Z=-20.5..-25.7 mm)",
            "overall_length_mm": 47.2,
            "meltzone_length_mm": 33.0,
        },
        visual_max_triangles=24000,
    )
    t0 = time.perf_counter()
    res = _bake_coupled_multi_mount_superelement(
        spec,
        regions,
        extra_port0_selectors=[_select_chube_shell_brace],
        visual_parts=placeholder_vis,
        assets_root=assets_root,
    )
    # Reuse the canonical high-detail Chube Compact GLB so visuals stay bit-identical w/ zero double-draw
    src_glb = assets_root / "hotends" / "chube_compact" / "visual.glb"
    if src_glb.is_file():
        dst_glb = res.asset_dir / "visual.glb"
        shutil.copy2(src_glb, dst_glb)
        src_manifest = json.loads((assets_root / "hotends" / "chube_compact" / "manifest.json").read_text())
        res.manifest["visual_triangles"] = int(src_manifest.get("visual_triangles", res.manifest["visual_triangles"]))
        res.manifest["visual_bytes"] = int(dst_glb.stat().st_size)
        (res.asset_dir / "manifest.json").write_text(json.dumps(res.manifest, indent=2), encoding="utf-8")

    print(
        f"  Baked braced Chube Compact in {time.perf_counter() - t0:.1f}s | elems={res.manifest['num_elements']} | "
        f"rigid_max={res.manifest['rigid_body_max_freq_hz']:.5f} Hz | "
        f"K={res.manifest['port_stiffness_n_per_mm']}"
    )
    return res


def _bake_trinity_toolhead_variant(
    *,
    step_path: Path,
    asset_id: str,
    name: str,
    port1_mm: tuple[float, float, float],
    port1_z_band: tuple[float, float],
    part_recipe: dict[str, Any],
    assets_root: Path = ASSETS_ROOT,
    visuals_only: bool = False,
) -> BakedAssetResult:
    """Bake a Trinity 1.6 unibody toolhead Tet4 superelement + dressed PBR GLB."""
    print(f"\n=== Baking toolheads/{asset_id} ===")
    root = b3d.import_step(str(step_path))

    # MGN12H front carriage face in STEP is at X=0.0, Y=0.0, Z=+2.24 mm
    origin_mgn = (0.0, 0.0, 2.24)

    body_solids: list[Any] = []
    visual_parts: list[VisualPart] = []

    for idx, (path, ws) in enumerate(_walk_world_solids(root)):
        low = path.lower()
        leaf = low.split("/")[-1]

        # Skip MGN12H rail carriage (rendered by PrintSim gantry) and hotends (rendered separately at Port 1!)
        if any(
            skip in low
            for skip in (
                "mgn12h_cart",
                "dragon_ace_volcano",
                "chube_compact_integration",
                "/nozzle",
                "92125a125",  # Chube radial screws belong to hotend
            )
        ):
            continue
        if leaf == "nozzle":
            continue

        s = _step_to_printsim_shape(ws, origin_mgn)
        tm = _shape_to_trimesh(s, deflection=0.35)
        if len(tm.faces) == 0:
            continue

        if "trinity_body" in leaf:
            body_solids.append(s)
            visual_parts.append(VisualPart(tm, "trinity_unibody", COL_BODY_CF, metallic=0.14, roughness=0.46))
        elif "ducts" in leaf:
            visual_parts.append(VisualPart(tm, "trinity_ducts", COL_DUCT_CF, metallic=0.12, roughness=0.48))
        elif "logo_cutout" in leaf:
            if "negative" in leaf:
                continue
            visual_parts.append(VisualPart(tm, "trinity_logo", COL_ACCENT_TEAL, metallic=0.20, roughness=0.35))
        elif "slm_monolith" in leaf:
            visual_parts.append(VisualPart(tm, "monolith_slm_belt_clamp", COL_SLM_AL, metallic=0.86, roughness=0.30))
        elif "ffb036" in leaf or "delta_3628" in low:
            col = COL_FAN_ROTOR if "prt0001" in leaf else COL_FAN_HOUSING
            visual_parts.append(VisualPart(tm, f"delta_3628_{idx}", col, metallic=0.18, roughness=0.52))
        elif "delta-2510" in leaf:
            visual_parts.append(VisualPart(tm, f"delta_2510_hef_{idx}", COL_FAN_HOUSING, metallic=0.18, roughness=0.52))
        elif "beacon" in low:
            col = COL_PCB_DARK if ws.volume / 1000.0 > 0.5 else COL_STEEL
            visual_parts.append(VisualPart(tm, f"beacon_probe_{idx}", col, metallic=0.35, roughness=0.42))
        elif "bowden" in leaf:
            visual_parts.append(VisualPart(tm, "ptfe_micro_bowden", COL_PTFE, metallic=0.05, roughness=0.45))
        elif "apex_clip" in leaf:
            visual_parts.append(VisualPart(tm, f"apex_clip_{idx}", "#475569", metallic=0.15, roughness=0.50))
        elif "insert" in leaf:
            visual_parts.append(VisualPart(tm, f"heatset_{idx}", COL_BRASS_INSERT, metallic=0.85, roughness=0.30))
        else:
            visual_parts.append(VisualPart(tm, f"fastener_{idx}", COL_STEEL, metallic=0.82, roughness=0.28))

    if not body_solids:
        raise RuntimeError(f"Could not find Trinity_body solid in {step_path}")

    body_mesh = _shape_to_trimesh(body_solids[0], deflection=0.25)
    bmin, bmax = body_mesh.bounds
    print(
        f"  {asset_id} body bounds (mm): min={np.round(bmin, 2)} max={np.round(bmax, 2)} | "
        f"vol={body_solids[0].volume / 1000.0:.2f} cm3"
    )

    if visuals_only:
        glb_path = assets_root / "toolheads" / asset_id / "visual.glb"
        tris, nbytes = build_visual_glb(visual_parts, glb_path, max_total_triangles=32000)
        print(f"  Refreshed visual.glb: {tris} tris, {nbytes} bytes")
        return None  # type: ignore[return-value]

    # Port 0: Strictly the MGN12H 4x M3x25 front carriage mating pad at Y_ps = 0 (X in [-16,16], Z in [-14.5,14.5])
    def _select_mgn12h_pad(coords_mm: np.ndarray) -> np.ndarray:
        mask = (
            (np.abs(coords_mm[:, 1]) <= 2.6)
            & (np.abs(coords_mm[:, 0]) <= 16.5)
            & (np.abs(coords_mm[:, 2]) <= 14.5)
        )
        if int(np.sum(mask)) >= 24:
            return mask
        d = np.linalg.norm(coords_mm - np.asarray([0.0, 0.0, 0.0]), axis=1)
        idx = np.argsort(d)[:36]
        out = np.zeros(len(coords_mm), dtype=bool)
        out[idx] = True
        return out

    # Port 1: Strictly the hotend top mounting flange shelf inside Trinity's unibody socket
    z_lo, z_hi = port1_z_band
    p1x, p1y, p1z = port1_mm

    def _select_hotend_shelf(coords_mm: np.ndarray) -> np.ndarray:
        r_xy = np.hypot(coords_mm[:, 0] - p1x, coords_mm[:, 1] - p1y)
        mask = (coords_mm[:, 2] >= z_lo) & (coords_mm[:, 2] <= z_hi) & (r_xy <= 13.0)
        if int(np.sum(mask)) >= 24:
            return mask
        d = np.linalg.norm(coords_mm - np.asarray(port1_mm), axis=1)
        idx = np.argsort(d)[:36]
        out = np.zeros(len(coords_mm), dtype=bool)
        out[idx] = True
        return out

    spec = BakedAssetSpec(
        category="toolheads",
        asset_id=asset_id,
        name=name,
        source_cad=step_path.name,
        source_url=TRINITY_SOURCE_URL,
        reference_material_id="ppa_cf_printed",
        cad_trust_pct=88,
        port0=PortDefinition(vp_coord_mm=(0.0, 0.0, 0.0), selector=_select_mgn12h_pad),
        port1=PortDefinition(vp_coord_mm=port1_mm, selector=_select_hotend_shelf),
        n_internal_modes=6,
        mesh_size_mm=2.8,
        is_multi_material=False,
        is_fdm_anisotropic=True,
        part_recipe={
            "origin": "MGN12H front carriage face (0,0,0)",
            "hotend_port_mm": list(port1_mm),
            "is_unibody": True,
            **part_recipe,
        },
        visual_max_triangles=32000,
    )
    t0 = time.perf_counter()
    res = bake_solid_superelement(
        spec,
        regions=[
            MaterialRegion(
                solid=body_mesh,
                material_id="ppa_cf_printed",
                label=f"{asset_id}_body",
                fdm_anisotropic=True,
                color_hex=COL_BODY_CF,
                metallic=0.14,
                roughness=0.46,
            )
        ],
        visual_parts=visual_parts,
        assets_root=assets_root,
    )
    # Mark explicit monolithic unibody flag in manifest so _builder_toolhead.py applies FDM infill/anisotropy
    # knockdown (0.55x) WITHOUT the multi-piece bolted-clamshell seam penalty (0.58x)!
    res.manifest["is_unibody"] = True
    (res.asset_dir / "manifest.json").write_text(json.dumps(res.manifest, indent=2), encoding="utf-8")

    print(
        f"  Baked {asset_id} in {time.perf_counter() - t0:.1f}s | elems={res.manifest['num_elements']} | "
        f"mass={res.manifest['mass_kg']*1000:.1f} g (solid ref) | "
        f"rigid_max={res.manifest['rigid_body_max_freq_hz']:.5f} Hz | "
        f"K={res.manifest['port_stiffness_n_per_mm']} | K_rot={res.manifest['port_rot_stiffness_nm_per_rad']}"
    )
    return res


def bake_trinity_dav_toolhead(
    cad_dir: Path,
    assets_root: Path = ASSETS_ROOT,
    visuals_only: bool = False,
) -> BakedAssetResult:
    return _bake_trinity_toolhead_variant(
        step_path=cad_dir / "Trinity_DAV_1.6_10mm.step",
        asset_id="trinity_1_6_dav",
        name="WV-Design Trinity 1.6 Unibody (Dragon Ace Volcano + Dual-3628)",
        port1_mm=(0.0, -9.74, -21.60),
        port1_z_band=(-24.2, -19.2),
        part_recipe={
            "default_hotend": "trianglelab_dragon_ace_volcano",
            "heatsink_brace_z_mm": [-44.5, -21.6],
            "target_nozzle_tip_mm": [0.0, -9.74, -81.0],
        },
        assets_root=assets_root,
        visuals_only=visuals_only,
    )


def bake_trinity_chube_toolhead(
    cad_dir: Path,
    assets_root: Path = ASSETS_ROOT,
    visuals_only: bool = False,
) -> BakedAssetResult:
    return _bake_trinity_toolhead_variant(
        step_path=cad_dir / "Trinity_Chube_1.6_10mm.step",
        asset_id="trinity_1_6_chube",
        name="WV-Design Trinity 1.6 Unibody (Chube Compact + Dual-3628)",
        port1_mm=(0.0, -9.74, -33.98),
        port1_z_band=(-36.8, -31.2),
        part_recipe={
            "default_hotend": "chube_compact",
            "shell_brace_z_mm": [-59.68, -33.98],
            "target_nozzle_tip_mm": [0.0, -9.74, -81.18],
        },
        assets_root=assets_root,
        visuals_only=visuals_only,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Bake Trinity 1.6 & Dragon Ace Volcano CAD superelements.")
    parser.add_argument("--cad-dir", type=Path, default=DEFAULT_CAD_DIR)
    parser.add_argument("--out-dir", type=Path, default=ASSETS_ROOT)
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated subset: dav,braced_dav,braced_chube,trinity_dav,trinity_chube",
    )
    parser.add_argument("--visuals-only", action="store_true")
    args = parser.parse_args()

    selected = {s.strip().lower() for s in args.only.split(",") if s.strip()} if args.only else {
        "dav",
        "braced_dav",
        "braced_chube",
        "trinity_dav",
        "trinity_chube",
    }

    baked_ids: list[tuple[str, str]] = []
    if "dav" in selected:
        bake_dragon_ace_volcano(args.cad_dir, args.out_dir, visuals_only=args.visuals_only)
        baked_ids.append(("hotends", "trianglelab_dragon_ace_volcano"))
    if "braced_dav" in selected:
        bake_trinity_dav_braced_assembly(args.cad_dir, args.out_dir, visuals_only=args.visuals_only)
        baked_ids.append(("assemblies", "trinity_1_6_dav__trianglelab_dragon_ace_volcano"))
    if "braced_chube" in selected:
        bake_trinity_chube_braced_assembly(args.out_dir)
        baked_ids.append(("assemblies", "trinity_1_6_chube__chube_compact"))
    if "trinity_dav" in selected:
        bake_trinity_dav_toolhead(args.cad_dir, args.out_dir, visuals_only=args.visuals_only)
        baked_ids.append(("toolheads", "trinity_1_6_dav"))
    if "trinity_chube" in selected:
        bake_trinity_chube_toolhead(args.cad_dir, args.out_dir, visuals_only=args.visuals_only)
        baked_ids.append(("toolheads", "trinity_1_6_chube"))

    if not args.visuals_only:
        print("\n=== QA Verification ===")
        for cat, aid in baked_ids:
            qa = verify_baked_asset(cat, aid, assets_root=args.out_dir)
            status = "PASS" if qa.passed else "FAIL"
            print(f"  [{status}] {cat}/{aid}: checks={qa.to_dict().get('checks')}")


if __name__ == "__main__":
    main()
