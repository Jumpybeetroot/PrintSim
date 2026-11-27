#!/usr/bin/env python3
"""CAL-02 Batch Baker: Tet4 Craig-Bampton CMS + smooth-normal PBR GLBs for 4 official toolheads.

Targets baked into ``src/printsim/assets/hardware/toolheads/``:
  1. ``xol_compact``    : XOL 2 Compact Dual-4010 + Sherpa Mini (PA12-CF reference)
  2. ``a4t_beacon``     : A4T Cowling + Dragon HE Duct + Xol Carriage (PPA-CF reference)
  3. ``dragon_burner``  : Dragon Burner V8 + Orbiter 2 Mount + MGN12H Carriage (PA12-CF reference)
  4. ``archetype_cpap`` : Archetype Breakneck v24 CPAP + Sherpa Micro + MGN12H (PPA-CF reference)

Usage:
    uv run python cad_recipes/bake_toolheads_cal02.py
    uv run python cad_recipes/bake_toolheads_cal02.py --only xol_compact,a4t_beacon
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import time
from typing import Any

import build123d as b3d
import numpy as np
import trimesh

from printsim.cad_assets import (
    BakedAssetSpec,
    MaterialRegion,
    PortDefinition,
    VisualPart,
    bake_solid_superelement,
    clear_asset_cache,
)
from printsim.cad_assets.recipes import verify_baked_asset

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = REPO_ROOT / "cad_sources" / "toolhead_audit"

# Authentic PBR material palette matching PrintSim Studio dark hardware theme
COL_CHARCOAL = "#22262e"
COL_CF_DARK = "#262c36"
COL_DUCT_BLACK = "#1c2028"
COL_ACCENT_RED = "#dc2626"
COL_FAN_BODY = "#1b202a"
COL_MOTOR_BODY = "#1a1f29"
COL_PCB_DARK = "#162922"
COL_STEEL = "#9ca8b8"
COL_STAINLESS = "#d8e0ea"
COL_BRASS = "#c99b44"
COL_LED_WHITE = "#f8fafc"


def _shape_to_trimesh(shape: Any, deflection: float = 0.55) -> trimesh.Trimesh:
    """Tessellate a build123d Shape/Solid into a trimesh.Trimesh (mm), skipping degenerate faces."""
    try:
        verts, tris = shape.tessellate(deflection)
        if verts and tris:
            v_arr = np.asarray([[v.X, v.Y, v.Z] for v in verts], dtype=np.float64)
            f_arr = np.asarray(tris, dtype=np.int64)
            return trimesh.Trimesh(vertices=v_arr, faces=f_arr, process=False)
    except Exception:
        pass
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    wrapped = shape.wrapped if hasattr(shape, "wrapped") else shape
    BRepMesh_IncrementalMesh(wrapped, float(deflection), False, 0.5, True)
    all_verts: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    v_offset = 0
    exp = TopExp_Explorer(wrapped, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face(exp.Current())
        loc = TopLoc_Location()
        poly = BRep_Tool.Triangulation_s(face, loc)
        if poly is not None and poly.NbNodes() >= 3 and poly.NbTriangles() >= 1:
            trsf = loc.Transformation()
            nb_n = poly.NbNodes()
            fv = np.empty((nb_n, 3), dtype=np.float64)
            for i in range(1, nb_n + 1):
                pt = poly.Node(i).Transformed(trsf)
                fv[i - 1, 0] = pt.X()
                fv[i - 1, 1] = pt.Y()
                fv[i - 1, 2] = pt.Z()
            nb_t = poly.NbTriangles()
            ff = np.empty((nb_t, 3), dtype=np.int64)
            is_rev = face.Orientation() == TopAbs_REVERSED
            for j in range(1, nb_t + 1):
                n1, n2, n3 = poly.Triangle(j).Get()
                if is_rev:
                    ff[j - 1] = (v_offset + n1 - 1, v_offset + n3 - 1, v_offset + n2 - 1)
                else:
                    ff[j - 1] = (v_offset + n1 - 1, v_offset + n2 - 1, v_offset + n3 - 1)
            all_verts.append(fv)
            all_faces.append(ff)
            v_offset += nb_n
        exp.Next()
    if not all_verts:
        return trimesh.Trimesh()
    return trimesh.Trimesh(
        vertices=np.vstack(all_verts),
        faces=np.vstack(all_faces),
        process=False,
    )


def _walk_named_solids(
    node: Any,
    prefix: str = "",
    parent_loc: Any = None,
) -> list[tuple[str, Any]]:
    """Collect ``(full_label, world_solid)`` pairs accumulating nested assembly Locations."""
    lbl = (getattr(node, "label", "") or "").strip()
    path = f"{prefix}/{lbl}" if (prefix and lbl) else (lbl or prefix)
    if parent_loc is None:
        parent_loc = b3d.Location()
    node_loc = getattr(node, "location", None)
    children = getattr(node, "children", None) or []
    if children:
        next_loc = parent_loc * node_loc if node_loc is not None else parent_loc
        out: list[tuple[str, Any]] = []
        for ch in children:
            out.extend(_walk_named_solids(ch, path, next_loc))
        return out
    solids = node.solids() if hasattr(node, "solids") else []
    if solids:
        return [(path, s.moved(parent_loc)) for s in solids]
    if isinstance(node, b3d.Solid):
        return [(path, node.moved(parent_loc))]
    return []


def _transform_mesh(
    mesh: trimesh.Trimesh,
    *,
    rx_deg: float = 0.0,
    ry_deg: float = 0.0,
    rz_deg: float = 0.0,
    translate_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> trimesh.Trimesh:
    m = mesh.copy()
    if rx_deg:
        m.apply_transform(trimesh.transformations.rotation_matrix(math.radians(rx_deg), [1, 0, 0]))
    if ry_deg:
        m.apply_transform(trimesh.transformations.rotation_matrix(math.radians(ry_deg), [0, 1, 0]))
    if rz_deg:
        m.apply_transform(trimesh.transformations.rotation_matrix(math.radians(rz_deg), [0, 0, 1]))
    tx, ty, tz = translate_mm
    if tx or ty or tz:
        m.apply_translation([tx, ty, tz])
    return m


def _combine_meshes(meshes: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    valid = [m for m in meshes if len(m.vertices) > 0 and len(m.faces) > 0]
    if not valid:
        raise ValueError("No non-empty meshes to combine")
    if len(valid) == 1:
        return valid[0].copy()
    return trimesh.util.concatenate(valid)


def _selector_box(
    x_rng: tuple[float, float],
    y_rng: tuple[float, float],
    z_rng: tuple[float, float],
    fallback_center: tuple[float, float, float],
    min_nodes: int = 24,
):
    def _sel(coords_mm: np.ndarray) -> np.ndarray:
        mask = (
            (coords_mm[:, 0] >= x_rng[0])
            & (coords_mm[:, 0] <= x_rng[1])
            & (coords_mm[:, 1] >= y_rng[0])
            & (coords_mm[:, 1] <= y_rng[1])
            & (coords_mm[:, 2] >= z_rng[0])
            & (coords_mm[:, 2] <= z_rng[1])
        )
        if int(np.sum(mask)) >= min_nodes:
            return mask
        # Expand around fallback center if coarse mesh falls slightly short
        d = np.linalg.norm(coords_mm - np.asarray(fallback_center, dtype=np.float64)[None, :], axis=1)
        order = np.argsort(d)
        out = np.zeros(len(coords_mm), dtype=bool)
        out[order[: max(min_nodes, 24)]] = True
        return out

    return _sel


def _is_fastener_leaf(leaf: str) -> bool:
    return any(
        k in leaf
        for k in (
            "screw",
            "shcs",
            "bhcs",
            "fhcs",
            "insert",
            "hexnut",
            "nut",
            "washer",
            "m3x",
            "m2_",
            "m2x",
            "m2.5",
            "pin",
            "magnet",
            "bolt",
        )
    )


def _print_result(asset_id: str, res: Any, dt: float) -> None:
    m = res.manifest
    ps = m["port_stiffness_n_per_mm"]
    pr = m["port_rot_stiffness_nm_per_rad"]
    qa = verify_baked_asset("toolheads", asset_id)
    print(
        f"  [CMS] toolheads/{asset_id}: {m['num_elements']:,} tets, "
        f"mass={m['mass_kg']*1000:.1f}g, rb={m['rigid_body_max_freq_hz']:.5f}Hz, "
        f"p1={m['port1_mm']}, "
        f"k=({ps['kx']:.1f},{ps['ky']:.1f},{ps['kz']:.1f}) N/mm, "
        f"krot=({pr['krx']:.1f},{pr['kry']:.1f},{pr['krz']:.1f}) Nm/rad, "
        f"tris={m['visual_triangles']}, glb={m['visual_bytes']/1024:.0f}KiB, "
        f"qa={'PASS' if qa.passed else 'FAIL'} ({dt:.1f}s)"
    )
    if not qa.passed:
        for ch in qa.checks:
            if not ch["passed"]:
                print(f"    [QA FAIL] {ch['name']}: {ch['value']} (expected {ch['threshold']})")


# =============================================================================
# 1. XOL 2 Compact Dual-4010 + Sherpa Mini
# =============================================================================
def _load_xol_items() -> list[tuple[str, Any]]:
    step_path = AUDIT_DIR / "xol" / "Xol Toolhead Assembly Guide 20240615.step"
    print(f"  Loading {step_path.name}...")
    root = b3d.import_step(str(step_path))
    return _walk_named_solids(root)


def bake_xol_compact(xol_items: list[tuple[str, Any]] | None = None) -> tuple[Any, list[tuple[str, Any]]]:
    print("\n=== Baking toolheads/xol_compact ===")
    if xol_items is None:
        xol_items = _load_xol_items()

    # MGN12H front carriage face in XOL STEP is at X=0.0, Y=-0.4, Z=0.0
    shift = (0.0, 0.4, 0.0)
    visual_parts: list[VisualPart] = []
    structural_meshes: list[trimesh.Trimesh] = []

    for path, solid in xol_items:
        lo = path.lower()
        leaf = lo.split("/")[-1]

        # Skip X-extrusion, MGN12 rail/carriage, TAP alternate, XolPAP alternate, Klicky probe,
        # long 480 mm rendered GT2 belts, and duplicate 9 mm belt clamps
        if any(
            k in lo
            for k in (
                "mgn12_x-rail_assembly",
                "tap-sherap-mini",
                "xolpap_rapido_assembly",
                "klicky_pcb_probe",
                "x-endstop_assembly",
                "render-belt",
                "belts_and_clips_-_9mm",
            )
        ):
            continue

        # Select ONLY the standard Rapido HF hotend mount (skip Dragon UHF, NF-Crazy, Heatset mod, and hotend hardware)
        if "hotend_mounts_assembly" in lo:
            if "hotend_mount_assembly_-_rapido_hf_<1>/" not in lo:
                continue
            if "rapido_hotend_hf" in lo:
                continue

        # Skip tiny SMD components on SHT-36v2 PCB (keep main PCB board)
        if "sht-36v2" in lo and "open_cascade_step_translator" not in lo:
            continue
        if "led_wires" in lo:
            continue

        tm = _shape_to_trimesh(solid, deflection=0.60)
        if len(tm.faces) == 0:
            continue
        tm = _transform_mesh(tm, translate_mm=shift)

        # Structural load-bearing printed chassis (Xol Carriage + Rapido HF duct body + Sherpa Mini mount)
        if leaf == "xol_carriage_mgn12h_-_arch-clips":
            structural_meshes.append(tm)
            visual_parts.append(VisualPart(tm, "xol_carriage", COL_CHARCOAL, metallic=0.14, roughness=0.48))
        elif "hotend_mount_assembly_-_rapido_hf_<1>/hf" in lo and leaf == "hf":
            structural_meshes.append(tm)
            visual_parts.append(VisualPart(tm, "xol_body", COL_CF_DARK, metallic=0.16, roughness=0.46))
        elif lo.endswith("extruder_mount_assembly_-_sherpa-mini_<1>/sherpa-mini"):
            structural_meshes.append(tm)
            visual_parts.append(VisualPart(tm, "xol_extruder_mount", COL_CHARCOAL, metallic=0.14, roughness=0.48))
        elif _is_fastener_leaf(leaf):
            col = COL_BRASS if "insert" in leaf else COL_STEEL
            visual_parts.append(VisualPart(tm, "xol_fastener", col, metallic=0.75, roughness=0.30))
        elif "faceplate" in leaf:
            visual_parts.append(VisualPart(tm, "xol_faceplate", COL_ACCENT_RED, metallic=0.15, roughness=0.42))
        elif any(k in leaf for k in ("4010_", "2510_")):
            visual_parts.append(VisualPart(tm, "xol_fan", COL_FAN_BODY, metallic=0.20, roughness=0.55))
        elif "neopixel" in leaf:
            visual_parts.append(VisualPart(tm, "xol_led", COL_LED_WHITE, metallic=0.10, roughness=0.28))
        elif "nema14" in leaf:
            visual_parts.append(VisualPart(tm, "xol_motor", COL_MOTOR_BODY, metallic=0.65, roughness=0.35))
        elif "gear" in leaf or "shaft" in leaf:
            visual_parts.append(VisualPart(tm, "xol_gear", COL_STAINLESS, metallic=0.82, roughness=0.28))
        elif "sht-36v2" in lo:
            visual_parts.append(VisualPart(tm, "xol_pcb", COL_PCB_DARK, metallic=0.35, roughness=0.42))
        elif "sherpa_mini" in lo or "ebb36_mount" in leaf or "belts_and_clips" in lo:
            visual_parts.append(VisualPart(tm, "xol_extruder_housing", COL_DUCT_BLACK, metallic=0.15, roughness=0.50))
        else:
            visual_parts.append(VisualPart(tm, "xol_fastener", COL_STEEL, metallic=0.72, roughness=0.32))

    struct_combined = _combine_meshes(structural_meshes)
    bmin, bmax = struct_combined.bounds
    print(f"  XOL structural bounds (mm): min={np.round(bmin, 2)} max={np.round(bmax, 2)} ({len(structural_meshes)} solids)")

    # Port 0: MGN12H 4x M3 front carriage face (Y=0.0, X=+-10, Z=+-10)
    # Port 1: Rapido HF 4-bolt top mounting flange shelf (-0.1, -24.35, +4.68)
    port0_mm = (0.0, 0.0, 0.0)
    port1_mm = (-0.1, -24.35, 4.68)

    spec = BakedAssetSpec(
        category="toolheads",
        asset_id="xol_compact",
        name="XOL 2 Compact Dual-4010 (PA12-CF / ASA)",
        source_cad="Xol Toolhead Assembly Guide 20240615.step",
        source_url="https://github.com/Armchair-Heavy-Industries/Xol-Toolhead",
        reference_material_id="pa12_cf_printed",
        cad_trust_pct=86,
        port0=PortDefinition(
            vp_coord_mm=port0_mm,
            selector=_selector_box((-15.0, 15.0), (-2.5, 2.5), (-14.0, 14.0), port0_mm),
        ),
        port1=PortDefinition(
            vp_coord_mm=port1_mm,
            selector=_selector_box((-12.0, 12.0), (-36.0, -13.0), (2.2, 7.2), port1_mm),
        ),
        n_internal_modes=6,
        mesh_size_mm=3.5,
        is_multi_material=False,
        is_fdm_anisotropic=True,
        part_recipe={
            "origin": "MGN12H front carriage face (0,0,0)",
            "hotend_port_mm": list(port1_mm),
            "multi_piece_bolted": True,
            "duct_bottom_z_mm": -47.88,
            "target_nozzle_tip_z_mm": -49.5,
        },
        visual_max_triangles=30000,
    )
    t0 = time.perf_counter()
    res = bake_solid_superelement(
        spec,
        regions=[
            MaterialRegion(
                solid=struct_combined,
                material_id="pa12_cf_printed",
                label="xol_compact_body",
                fdm_anisotropic=True,
            )
        ],
        visual_parts=visual_parts,
    )
    _print_result("xol_compact", res, time.perf_counter() - t0)
    return res, xol_items


# =============================================================================
# 2. A4T + Dragon HE Duct + Xol MGN12H Carriage
# =============================================================================
def bake_a4t_beacon(xol_items: list[tuple[str, Any]] | None = None) -> tuple[Any, list[tuple[str, Any]]]:
    print("\n=== Baking toolheads/a4t_beacon ===")
    if xol_items is None:
        xol_items = _load_xol_items()

    visual_parts: list[VisualPart] = []
    structural_meshes: list[trimesh.Trimesh] = []

    # 1. Extract official Xol MGN12H Carriage + belt clips (shifted +0.4 mm in Y so Port 0 = (0,0,0))
    xol_shift = (0.0, 0.4, 0.0)
    for path, solid in xol_items:
        lo = path.lower()
        leaf = lo.split("/")[-1]
        if leaf == "xol_carriage_mgn12h_-_arch-clips":
            tm = _transform_mesh(_shape_to_trimesh(solid, deflection=0.55), translate_mm=xol_shift)
            structural_meshes.append(tm)
            visual_parts.append(VisualPart(tm, "a4t_xol_carriage", COL_CHARCOAL, metallic=0.14, roughness=0.48))
        elif "mgn12h_carriage_and_hardware" in lo and "belts_and_clips" in lo:
            if "render-belt" in lo or "belts_and_clips_-_9mm" in lo:
                continue
            tm = _transform_mesh(_shape_to_trimesh(solid, deflection=0.65), translate_mm=xol_shift)
            if len(tm.faces) > 0:
                visual_parts.append(VisualPart(tm, "a4t_belt_clip", COL_DUCT_BLACK, metallic=0.16, roughness=0.48))
        elif "mgn12h_carriage_and_hardware" in lo and "mgn12_x-rail" not in lo and "klicky" not in lo and "endstop" not in lo:
            if _is_fastener_leaf(leaf):
                tm = _transform_mesh(_shape_to_trimesh(solid, deflection=0.70), translate_mm=xol_shift)
                if len(tm.faces) > 0:
                    col = COL_BRASS if "insert" in leaf else COL_STEEL
                    visual_parts.append(VisualPart(tm, "a4t_fastener", col, metallic=0.75, roughness=0.30))

    # 2. Load official A4T Cowling (skip sacrificial Support_* solids!)
    # Print-oriented (Z=0 is front face) -> rx=-90 deg puts front at -Y, rear mating face at +46.0 mm.
    # Translate (0.0, -56.20, -5.62) mates A4T rear face flush onto Xol Carriage front face (Y=-10.20)
    # and aligns bottom M3x50 holes (+-16.25, Z=-32.12) & upper M3 holes (+-15.50, Z=+11.18) to 0.05 mm!
    a4t_shift = (0.0, -56.20, -5.62)
    cowl_root = b3d.import_step(str(AUDIT_DIR / "a4t_cowling_dragon_xol.step"))
    for path, solid in _walk_named_solids(cowl_root):
        lo = path.lower()
        if "support" in lo or abs(solid.volume) < 10000.0:
            continue
        tm_vis = _transform_mesh(_shape_to_trimesh(solid, deflection=0.45), rx_deg=-90.0, translate_mm=a4t_shift)
        # Overlap 0.8 mm across the bolted Y=-10.20 mating face so 3.0 mm voxel Tet4 avoids seam pinching
        tm_fea = _transform_mesh(tm_vis, translate_mm=(0.0, 0.8, 0.0))
        structural_meshes.append(tm_fea)
        visual_parts.append(VisualPart(tm_vis, "a4t_cowling", COL_CF_DARK, metallic=0.16, roughness=0.44))

    # 3. Load official A4T Dragon HE Duct (same CAD origin as A4T Cowling -> same rx=-90 + a4t_shift!)
    he_root = b3d.import_step(str(AUDIT_DIR / "a4t_he_duct_dragon.step"))
    for path, solid in _walk_named_solids(he_root):
        tm_vis = _transform_mesh(_shape_to_trimesh(solid, deflection=0.45), rx_deg=-90.0, translate_mm=a4t_shift)
        tm_fea = _transform_mesh(tm_vis, translate_mm=(0.0, 0.8, 0.0))
        structural_meshes.append(tm_fea)
        visual_parts.append(VisualPart(tm_vis, "a4t_he_duct", COL_ACCENT_RED, metallic=0.15, roughness=0.42))

    # 4. Compact static Beacon RevH probe silhouette seated under Xol Carriage rear probe pad (visual only)
    beacon_pcb = b3d.Box(34.0, 14.0, 2.2).translate((0.0, -3.5, -40.3))
    visual_parts.append(
        VisualPart(_shape_to_trimesh(beacon_pcb, deflection=0.5), "a4t_beacon_pcb", COL_PCB_DARK, metallic=0.35, roughness=0.40)
    )

    struct_combined = _combine_meshes(structural_meshes)
    bmin, bmax = struct_combined.bounds
    print(f"  A4T structural bounds (mm): min={np.round(bmin, 2)} max={np.round(bmax, 2)} ({len(structural_meshes)} solids)")

    # Port 0: MGN12H 4x M3 front carriage face (Y=0.0, X=+-10, Z=+-10)
    # Port 1: Dragon/Rapido 4-bolt mounting shelf inside A4T (0.0, -24.35, +4.88)
    port0_mm = (0.0, 0.0, 0.0)
    port1_mm = (0.0, -24.35, 4.88)

    spec = BakedAssetSpec(
        category="toolheads",
        asset_id="a4t_beacon",
        name="A4T + Dragon HE Duct + Xol Carriage (PPA-CF / ASA)",
        source_cad="a4t_cowling_dragon_xol.step + a4t_he_duct_dragon.step + Xol_Carriage_MGN12H",
        source_url="https://github.com/DW-Tas/A4T",
        reference_material_id="ppa_cf_printed",
        cad_trust_pct=84,
        port0=PortDefinition(
            vp_coord_mm=port0_mm,
            selector=_selector_box((-15.0, 15.0), (-2.5, 2.5), (-14.0, 14.0), port0_mm),
        ),
        port1=PortDefinition(
            vp_coord_mm=port1_mm,
            selector=_selector_box((-12.0, 12.0), (-33.0, -14.0), (2.2, 7.6), port1_mm),
        ),
        n_internal_modes=6,
        mesh_size_mm=3.0,
        is_multi_material=False,
        is_fdm_anisotropic=True,
        part_recipe={
            "origin": "MGN12H front carriage face (0,0,0)",
            "hotend_port_mm": list(port1_mm),
            "multi_piece_bolted": True,
            "duct_bottom_z_mm": -47.14,
            "target_nozzle_tip_z_mm": -50.0,
        },
        visual_max_triangles=30000,
    )
    t0 = time.perf_counter()
    res = bake_solid_superelement(
        spec,
        regions=[
            MaterialRegion(
                solid=struct_combined,
                material_id="ppa_cf_printed",
                label="a4t_beacon_body",
                fdm_anisotropic=True,
            )
        ],
        visual_parts=visual_parts,
    )
    _print_result("a4t_beacon", res, time.perf_counter() - t0)
    return res, xol_items


# =============================================================================
# 3. Dragon Burner V8 + Orbiter 2 Mount + MGN12H X-Carriage Plate
# =============================================================================
def _build_dragon_burner_mgn12h_carriage() -> b3d.Solid:
    """Parametric Voron-style MGN12H X-carriage adapter mating Dragon Burner's rear face (Y=-8.65 mm).

    Front carriage bolt face sits at Y=0.0 mm (MGN12H 4x M3 20x20 mm pattern), body spans Y=[-8.85, 0.0]
    (with 0.2 mm overlap into Dragon Burner rear face at Y=-8.65 for clean voxel Tet4 bonding) plus
    side belt-clamp tabs extending to Y=+4.0 mm behind the MGN12H block sides (|X| in [16.5, 24.0]).
    Volume ~15.8 cm3 matching community MGN12H printed carriage adapters (~15-18 cm3).
    """
    # Keep carriage focused on MGN12H 4x M3 bolt pad + belt wings (no floating lower spine)
    main_plate = b3d.Box(48.0, 8.85, 44.0).translate((0.0, -4.425, 0.0))
    left_wing = b3d.Box(7.5, 6.0, 26.0).translate((-20.25, 3.0, 0.0))
    right_wing = b3d.Box(7.5, 6.0, 26.0).translate((20.25, 3.0, 0.0))
    plate = main_plate + left_wing + right_wing

    # Cut standard MGN12H 4x M3 clearance holes at (+-10, 0, +-10)
    for sx in (-10.0, 10.0):
        for sz in (-10.0, 10.0):
            hole = b3d.Cylinder(radius=1.7, height=14.0).rotate(b3d.Axis.X, 90.0).translate((sx, -4.4, sz))
            plate = plate - hole
    # Central weight-relief slot behind hotend area
    relief = b3d.Box(12.0, 12.0, 12.0).translate((0.0, -4.4, -22.0))
    plate = plate - relief
    solids = plate.solids()
    return solids[0] if solids else plate


def bake_dragon_burner() -> Any:
    print("\n=== Baking toolheads/dragon_burner ===")
    step_path = AUDIT_DIR / "dragon_burner_v8.step"
    root = b3d.import_step(str(step_path))
    items = _walk_named_solids(root)

    # Shift Dragon Burner assembly so Dragon 4-bolt hotend centerline (147.70, -130.78) sits at
    # X=0.0, Y=-20.00 mm (matching TOOLHEAD_PRESETS["dragon_burner"] nozzle_offset_mm = [0, -20, -50])
    # and Dragon_Mount bottom shelf Z=94.26 sits at Z=-3.54 mm (duct bottom Z=-47.50 mm, nozzle -50.0 mm).
    shift = (-147.70, 110.78, -97.80)

    visual_parts: list[VisualPart] = []
    structural_meshes: list[trimesh.Trimesh] = []

    # Add Voron MGN12H carriage plate bridging Port 0 (Y=0) to Dragon Burner rear face (Y=-8.65)
    carriage_solid = _build_dragon_burner_mgn12h_carriage()
    tm_car = _shape_to_trimesh(carriage_solid, deflection=0.45)
    structural_meshes.append(tm_car)
    visual_parts.append(VisualPart(tm_car, "db_mgn12h_carriage", COL_CHARCOAL, metallic=0.14, roughness=0.48))

    # Select ONE clean canonical configuration: load-bearing chassis (MainBody + Orbiter2 Dragon_Mount
    # + Orbiter2_Mount + MGN12H carriage) in structural FEA; cosmetic lower cooling ducts, rear brace,
    # and Neopixel LED carrier/diffuser in visual_parts only (prevents floppy thin-duct modes <100 Hz).
    exact_structural = {
        "cowl/mainbody": ("db_cowl", COL_ACCENT_RED, 0.15, 0.42),
        "orbiter2_mounts/dragon_mount__3_": ("db_dragon_mount", COL_CF_DARK, 0.16, 0.46),
        "orbiter2_mounts/orbiter2_mount__4_": ("db_orbiter_mount", COL_CHARCOAL, 0.14, 0.48),
    }
    exact_visual_only = {
        "cowl/ducts/duct_left/standard__1_": ("db_duct_left", COL_DUCT_BLACK, 0.14, 0.48),
        "cowl/ducts/duct_right/standard": ("db_duct_right", COL_DUCT_BLACK, 0.14, 0.48),
        "brace_straight": ("db_brace", COL_CHARCOAL, 0.14, 0.48),
        "led_holder_v41/[o]_carrier_neopixel": ("db_led_carrier", COL_DUCT_BLACK, 0.15, 0.48),
        "led_holder_v41/[c]_voron_diffuser": ("db_led_diffuser", COL_LED_WHITE, 0.10, 0.28),
    }

    for path, solid in items:
        sub = path.split("/", 1)[1].lower() if "/" in path else path.lower()
        if sub in exact_structural:
            lbl, col, met, rgh = exact_structural[sub]
            tm = _transform_mesh(_shape_to_trimesh(solid, deflection=0.45), translate_mm=shift)
            structural_meshes.append(tm)
            visual_parts.append(VisualPart(tm, lbl, col, metallic=met, roughness=rgh))
        elif sub in exact_visual_only:
            lbl, col, met, rgh = exact_visual_only[sub]
            tm = _transform_mesh(_shape_to_trimesh(solid, deflection=0.50), translate_mm=shift)
            visual_parts.append(VisualPart(tm, lbl, col, metallic=met, roughness=rgh))

    struct_combined = _combine_meshes(structural_meshes)
    bmin, bmax = struct_combined.bounds
    print(f"  Dragon Burner structural bounds (mm): min={np.round(bmin, 2)} max={np.round(bmax, 2)} ({len(structural_meshes)} solids)")

    # Port 0: MGN12H 4x M3 front carriage face (Y=0.0, X=+-10, Z=+-10)
    # Port 1: Dragon 4x M2.5 bottom mounting plane on Dragon_Mount__3_ (0.0, -20.0, -3.54)
    port0_mm = (0.0, 0.0, 0.0)
    port1_mm = (0.0, -20.0, -3.54)

    spec = BakedAssetSpec(
        category="toolheads",
        asset_id="dragon_burner",
        name="Dragon Burner V8 + Orbiter 2 Mount (PA12-CF / ABS)",
        source_cad="dragon_burner_v8.step",
        source_url="https://github.com/chirpy2605/voron/tree/main/V0/Dragon_Burner",
        reference_material_id="pa12_cf_printed",
        cad_trust_pct=83,
        port0=PortDefinition(
            vp_coord_mm=port0_mm,
            selector=_selector_box((-15.0, 15.0), (-2.5, 2.5), (-14.0, 14.0), port0_mm),
        ),
        port1=PortDefinition(
            vp_coord_mm=port1_mm,
            selector=_selector_box((-10.5, 10.5), (-28.5, -11.5), (-5.8, -1.0), port1_mm),
        ),
        n_internal_modes=6,
        mesh_size_mm=3.2,
        is_multi_material=False,
        is_fdm_anisotropic=True,
        part_recipe={
            "origin": "MGN12H front carriage face (0,0,0)",
            "hotend_port_mm": list(port1_mm),
            "multi_piece_bolted": True,
            "duct_bottom_z_mm": -47.50,
            "target_nozzle_tip_z_mm": -50.0,
        },
        visual_max_triangles=28000,
    )
    t0 = time.perf_counter()
    res = bake_solid_superelement(
        spec,
        regions=[
            MaterialRegion(
                solid=struct_combined,
                material_id="pa12_cf_printed",
                label="dragon_burner_body",
                fdm_anisotropic=True,
            )
        ],
        visual_parts=visual_parts,
    )
    _print_result("dragon_burner", res, time.perf_counter() - t0)
    return res


# =============================================================================
# 4. Archetype Breakneck v24 CPAP + Sherpa Micro + MGN12H
# =============================================================================
def bake_archetype_cpap() -> Any:
    print("\n=== Baking toolheads/archetype_cpap ===")
    step_path = AUDIT_DIR / "archetype" / "Archetype Public CAD STEP.step"
    print(f"  Loading {step_path.name}...")
    root = b3d.import_step(str(step_path))
    items = _walk_named_solids(root)

    # MGN12H front carriage face in Archetype STEP is at Y=+6.2, X=0.0, Z=0.0
    shift = (0.0, -6.2, 0.0)

    visual_parts: list[VisualPart] = []
    structural_meshes: list[trimesh.Trimesh] = []

    for path, solid in items:
        lo = path.lower()
        leaf = lo.split("/")[-1]

        # Include ONLY Breakneck_v24 + Core/Rail_Carriage/MGN12H_v45 + Rapido_Mount_Parametric_v23
        # + Breakneck_Sherpa_Micro_v29 + Beacon_Long (skip Atrocity/Blackbird/Mjolnir/Zephyr/MGN9H/other hotends/CPAP_Tube)
        in_mgn12 = "core/rail_carriage/mgn12h_v45/" in lo
        in_rapido_mount = "core/hotend_mount/rapido_mount_parametric_v23/" in lo
        in_breakneck = "archetypes/breakneck_v24/" in lo
        in_sherpa = "core/extruder_block/breakneck_blocks/breakneck_sherpa_micro_v29/" in lo
        in_beacon = "core/probes/probe_mounts/beacon_mounts_v12/beacon_long/" in lo

        if not (in_mgn12 or in_rapido_mount or in_breakneck or in_sherpa or in_beacon):
            continue

        # Skip steel MGN12H carriage block (PrintSim draws it), hotend solids (drawn separately at Port 1),
        # giant 464mm CPAP hose, alternate Breakneck lengths, LED soldering tools & tiny wire clips
        if any(
            k in lo
            for k in (
                "mgn12h_carriage/",
                "rapido_hf_v1/",
                "rapido_uhf_v2/",
                "cpap_tube",
                "breakneck_medium",
                "breakneck_short",
                "overmatch_large",
                "breakneck_fan_cover_led",
                "led's_v7/",
                "twist_wire",
                "thinwall_tube",
                "housing_front^",  # keep housing_front_k only (avoid duplicate front cover!)
                "filed_down_neopixel",
            )
        ):
            continue

        tm = _shape_to_trimesh(solid, deflection=0.60)
        if len(tm.faces) == 0:
            continue
        tm = _transform_mesh(tm, translate_mm=shift)

        # Structural load-bearing printed solids
        if in_mgn12 and "mgn12_carriage_mount/" in lo:
            structural_meshes.append(tm)
            visual_parts.append(VisualPart(tm, "arch_mgn12_mount", COL_CHARCOAL, metallic=0.14, roughness=0.48))
        elif in_mgn12 and "arch-xol_clips" in leaf:
            # Structural belt-lock clips bridge the carriage mount sides into the core
            structural_meshes.append(tm)
            visual_parts.append(VisualPart(tm, "arch_belt_clips", COL_DUCT_BLACK, metallic=0.15, roughness=0.48))
        elif in_rapido_mount and leaf == "rapido_mount":
            # Overlap +1.4 mm across the Y=-6.2 bolted mating plane into mgn12_left/right so the
            # 3.0 mm voxel grid does not pinch the carriage-to-core bolted interface (still 4.8 mm ahead of Port 0)
            structural_meshes.append(_transform_mesh(tm, translate_mm=(0.0, 1.4, 0.0)))
            visual_parts.append(VisualPart(tm, "arch_hotend_core", COL_CF_DARK, metallic=0.16, roughness=0.46))
        elif in_breakneck and leaf == "breakneck_long":
            structural_meshes.append(_transform_mesh(tm, translate_mm=(0.0, 1.4, 0.0)))
            visual_parts.append(VisualPart(tm, "arch_breakneck_duct", COL_DUCT_BLACK, metallic=0.15, roughness=0.46))
        elif in_breakneck and leaf == "removable_hose_attachment":
            structural_meshes.append(_transform_mesh(tm, translate_mm=(0.0, 1.4, 0.0)))
            visual_parts.append(VisualPart(tm, "arch_cpap_inlet", COL_CHARCOAL, metallic=0.15, roughness=0.46))
        elif (in_sherpa and leaf == "solid" and "bar_test" in lo) or (in_sherpa and leaf == "[a]_microbowden_micro-sherpa"):
            structural_meshes.append(_transform_mesh(tm, translate_mm=(0.0, 1.4, 0.0)))
            visual_parts.append(VisualPart(tm, "arch_extruder_block", COL_CF_DARK, metallic=0.16, roughness=0.46))
        # Visual-only dressed components (fasteners, fans, fan cover, Sherpa Micro extruder, Beacon)
        elif _is_fastener_leaf(leaf):
            col = COL_BRASS if "insert" in leaf else COL_STEEL
            visual_parts.append(VisualPart(tm, "arch_fastener", col, metallic=0.75, roughness=0.30))
        elif in_breakneck and "[a]_breakneck_fan_cover" in lo:
            if "diffuser" in leaf:
                visual_parts.append(VisualPart(tm, "arch_led_diffuser", COL_LED_WHITE, metallic=0.10, roughness=0.28))
            else:
                visual_parts.append(VisualPart(tm, "arch_fan_cover", COL_ACCENT_RED, metallic=0.15, roughness=0.42))
        elif in_rapido_mount and "2510_axial_fan" in lo:
            visual_parts.append(VisualPart(tm, "arch_2510_fan", COL_FAN_BODY, metallic=0.20, roughness=0.55))
        elif in_sherpa and "nema14" in leaf:
            visual_parts.append(VisualPart(tm, "arch_motor", COL_MOTOR_BODY, metallic=0.65, roughness=0.35))
        elif in_sherpa and ("gear" in leaf or "shaft" in leaf):
            visual_parts.append(VisualPart(tm, "arch_gear", COL_STAINLESS, metallic=0.82, roughness=0.28))
        elif in_sherpa and "ptfe" in leaf:
            visual_parts.append(VisualPart(tm, "arch_ptfe", "#e2e8f0", metallic=0.05, roughness=0.45))
        elif in_sherpa:
            visual_parts.append(VisualPart(tm, "arch_sherpa_housing", COL_DUCT_BLACK, metallic=0.15, roughness=0.50))
        elif in_beacon and "beacon_revd" in lo:
            visual_parts.append(VisualPart(tm, "arch_beacon_pcb", COL_PCB_DARK, metallic=0.35, roughness=0.40))
        elif in_beacon:
            visual_parts.append(VisualPart(tm, "arch_beacon_mount", COL_CHARCOAL, metallic=0.14, roughness=0.48))
        elif in_mgn12 and "arch-xol_clips" in leaf:
            visual_parts.append(VisualPart(tm, "arch_belt_clips", COL_DUCT_BLACK, metallic=0.15, roughness=0.48))
        else:
            visual_parts.append(VisualPart(tm, "arch_fastener", COL_STEEL, metallic=0.72, roughness=0.32))

    struct_combined = _combine_meshes(structural_meshes)
    bmin, bmax = struct_combined.bounds
    print(f"  Archetype structural bounds (mm): min={np.round(bmin, 2)} max={np.round(bmax, 2)} ({len(structural_meshes)} solids)")

    # Port 0: MGN12H 4x M3 front carriage face (Y=0.0, X=+-10, Z=+-10)
    # Port 1: Rapido/UHF hotend top mounting flange shelf inside rapido_mount (0.0, -21.7, +13.5)
    port0_mm = (0.0, 0.0, 0.0)
    port1_mm = (0.0, -21.7, 13.5)

    spec = BakedAssetSpec(
        category="toolheads",
        asset_id="archetype_cpap",
        name="Archetype Breakneck v24 CPAP (PPA-CF / ASA)",
        source_cad="Archetype Public CAD STEP.step",
        source_url="https://github.com/Armchair-Heavy-Industries/Archetype",
        reference_material_id="ppa_cf_printed",
        cad_trust_pct=82,
        port0=PortDefinition(
            vp_coord_mm=port0_mm,
            selector=_selector_box((-16.0, 16.0), (-2.5, 2.5), (-14.0, 14.0), port0_mm),
        ),
        port1=PortDefinition(
            vp_coord_mm=port1_mm,
            selector=_selector_box((-12.0, 12.0), (-32.0, -11.5), (11.0, 16.5), port1_mm),
        ),
        n_internal_modes=6,
        mesh_size_mm=3.0,
        is_multi_material=False,
        is_fdm_anisotropic=True,
        part_recipe={
            "origin": "MGN12H front carriage face (0,0,0)",
            "hotend_port_mm": list(port1_mm),
            "multi_piece_bolted": True,
            "duct_bottom_z_mm": -53.0,
            "target_nozzle_tip_z_mm": -64.0,
        },
        visual_max_triangles=30000,
    )
    t0 = time.perf_counter()
    res = bake_solid_superelement(
        spec,
        regions=[
            MaterialRegion(
                solid=struct_combined,
                material_id="ppa_cf_printed",
                label="archetype_cpap_body",
                fdm_anisotropic=True,
            )
        ],
        visual_parts=visual_parts,
    )
    _print_result("archetype_cpap", res, time.perf_counter() - t0)
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="Bake CAL-02 official toolhead CAD superelements.")
    parser.add_argument(
        "--only",
        type=str,
        default="xol_compact,a4t_beacon,dragon_burner,archetype_cpap",
        help="Comma-separated subset of toolhead IDs to bake.",
    )
    args = parser.parse_args()
    targets = [t.strip() for t in args.only.split(",") if t.strip()]

    clear_asset_cache()
    xol_items: list[tuple[str, Any]] | None = None

    if "xol_compact" in targets:
        _, xol_items = bake_xol_compact(xol_items)
    if "a4t_beacon" in targets:
        _, xol_items = bake_a4t_beacon(xol_items)
    if "dragon_burner" in targets:
        bake_dragon_burner()
    if "archetype_cpap" in targets:
        bake_archetype_cpap()

    clear_asset_cache()
    print("\n=== CAL-02 Bake Complete ===")


if __name__ == "__main__":
    main()
