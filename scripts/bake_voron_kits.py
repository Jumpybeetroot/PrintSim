#!/usr/bin/env python3
"""Bake Voron, LDO, Siboor, and Monolith CAD hardware assets into ``src/printsim/assets/hardware/``.

Groups (can be run independently in parallel via ``--group <name>``):
  - ``stealthburner`` : Voron StealthBurner + Clockwork 2.5 + LDO Nitehawk-SB v2.0 (Tet4 CMS + PBR GLB)
  - ``monolith``      : User Onshape LDO Monolith CNC AWD 9mm + LDO 2WD + Milled V2.4 Z-joint
                        + Structural Side Panel VT300 (Tet4 CMS + PBR GLBs)
  - ``siboor``        : Siboor / GKD CNC AWD & 2WD Gantry brackets (Tet4 CMS + PBR GLBs)
  - ``voron_stock``   : Official Voron 2.4r2 & Trident printed gantry/Z brackets (Tet4 CMS + PBR GLBs)
                        + Standard Voron printed visual parts (skirts, exhaust, Z covers, clips, handles)
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any

import build123d as b3d
import numpy as np
import trimesh

from printsim.cad_assets import (
    ASSETS_ROOT,
    BakedAssetSpec,
    MaterialRegion,
    PortDefinition,
    VisualPart,
    bake_solid_superelement,
    build_visual_glb,
    clear_asset_cache,
)

# Authentic PBR color palette
COL_VORON_BLACK = "#22262e"
COL_VORON_RED = "#dc2626"
COL_CNC_SILVER = "#d5dde8"
COL_CNC_DARK = "#2d3440"
COL_SIBOOR_RED = "#c81e2b"
COL_SIBOOR_BLACK = "#1f242d"
COL_STEEL = "#9ca8b8"
COL_STAINLESS = "#d8e0ea"
COL_PULLEY_AL = "#cfd7e2"
COL_BRASS = "#d4a63a"
COL_MOTOR_BODY = "#1b202a"
COL_MOTOR_CAP = "#9aa6b5"
COL_PCB_DARK = "#16202a"
COL_GASKET_GRAY = "#3a414d"
COL_PANEL_AL = "#bfc9d6"


def _shape_to_trimesh(shape: Any, deflection: float = 0.25) -> trimesh.Trimesh:
    """Tessellate a build123d Shape/Compound/Solid into a trimesh.Trimesh (in mm), skipping degenerate faces."""
    try:
        verts, tris = shape.tessellate(deflection)
        if verts and tris:
            v_arr = np.asarray([[v.X, v.Y, v.Z] for v in verts], dtype=np.float64)
            f_arr = np.asarray(tris, dtype=np.int64)
            return trimesh.Trimesh(vertices=v_arr, faces=f_arr, process=False)
    except Exception:
        pass
    # Robust OCCT face-by-face fallback that ignores degenerate/unmeshable micro-faces
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
    mirror_x: bool = False,
) -> trimesh.Trimesh:
    """Apply rotation -> optional X mirror (preserving outward winding) -> translation."""
    m = mesh.copy()
    if rx_deg:
        m.apply_transform(trimesh.transformations.rotation_matrix(math.radians(rx_deg), [1, 0, 0]))
    if ry_deg:
        m.apply_transform(trimesh.transformations.rotation_matrix(math.radians(ry_deg), [0, 1, 0]))
    if rz_deg:
        m.apply_transform(trimesh.transformations.rotation_matrix(math.radians(rz_deg), [0, 0, 1]))
    if mirror_x:
        m.vertices[:, 0] *= -1.0
        m.faces = m.faces[:, ::-1]
        m.fix_normals()
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


def _selector_nearest_k_plane(
    axis: int,
    target_mm: float,
    tol_mm: float = 3.0,
    min_nodes: int = 12,
):
    """Robust boundary vertex selector: vertices within ``tol_mm`` of plane (expands to >= min_nodes)."""

    def _select(coords_mm: np.ndarray) -> np.ndarray:
        dist = np.abs(coords_mm[:, axis] - target_mm)
        mask = dist <= tol_mm
        if int(np.sum(mask)) >= min_nodes:
            return mask
        # Expand smoothly until we have enough non-collinear nodes
        order = np.argsort(dist)
        out = np.zeros(len(coords_mm), dtype=bool)
        out[order[: max(min_nodes, min(24, len(coords_mm) // 8))]] = True
        return out

    return _select


def _selector_box_or_nearest(
    center_mm: tuple[float, float, float],
    radius_mm: float,
    min_nodes: int = 16,
    other_center_mm: tuple[float, float, float] | None = None,
):
    """Select boundary vertices within ``radius_mm`` of ``center_mm``, falling back to nearest k.

    When ``other_center_mm`` is provided, excludes vertices closer to the opposite port
    (Voronoi half-space with 0.5 mm gap) so Port 0 and Port 1 stay strictly disjoint.
    """
    c = np.asarray(center_mm, dtype=np.float64)
    c_other = np.asarray(other_center_mm, dtype=np.float64) if other_center_mm is not None else None

    def _select(coords_mm: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(coords_mm - c[None, :], axis=1)
        if c_other is not None:
            d_other = np.linalg.norm(coords_mm - c_other[None, :], axis=1)
            eligible = d + 0.5 < d_other
        else:
            eligible = np.ones(len(coords_mm), dtype=bool)
        mask = (d <= radius_mm) & eligible
        if int(np.sum(mask)) >= min_nodes:
            return mask
        # Fall back to nearest k among eligible vertices
        elig_idx = np.flatnonzero(eligible)
        if len(elig_idx) == 0:
            elig_idx = np.arange(len(coords_mm))
        order = elig_idx[np.argsort(d[elig_idx])]
        out = np.zeros(len(coords_mm), dtype=bool)
        k = min(len(order), max(min_nodes, min(28, len(coords_mm) // 8)))
        out[order[:k]] = True
        return out

    return _select


SKIP_EXISTING: bool = False


def _already_baked(category: str, asset_id: str, require_npz: bool = True) -> bool:
    if not SKIP_EXISTING:
        return False
    d = ASSETS_ROOT / category / asset_id
    if not ((d / "manifest.json").is_file() and (d / "visual.glb").is_file()):
        return False
    if require_npz:
        npz_path = d / "superelement.npz"
        if not npz_path.is_file():
            return False
        with np.load(npz_path) as data:
            if "field_points_m" not in data.files:
                return False
    print(f"  [skip] {category}/{asset_id} (already baked)")
    return True


def _write_visual_only_asset(
    *,
    category: str,
    asset_id: str,
    name: str,
    source_cad: str,
    source_url: str,
    visual_parts: list[VisualPart],
    max_triangles: int = 16000,
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    """Write a lightweight visual-only GLB + manifest.json (no heavy superelement.npz)."""
    asset_dir = ASSETS_ROOT / category / asset_id
    if _already_baked(category, asset_id, require_npz=False):
        return asset_dir
    asset_dir.mkdir(parents=True, exist_ok=True)
    glb_path = asset_dir / "visual.glb"
    vis_tris, vis_bytes = build_visual_glb(
        visual_parts, glb_path, max_total_triangles=max_triangles
    )

    # Compute overall bounds from visual parts (in mm)
    bmins = []
    bmaxs = []
    for vp in visual_parts:
        m = vp.shape_or_mesh
        if isinstance(m, trimesh.Trimesh) and len(m.vertices) > 0:
            bmins.append(m.bounds[0])
            bmaxs.append(m.bounds[1])
    if bmins:
        bmin = np.min(np.vstack(bmins), axis=0)
        bmax = np.max(np.vstack(bmaxs), axis=0)
        ref_size_mm = [round(float(v), 2) for v in (bmax - bmin)]
    else:
        ref_size_mm = [0.0, 0.0, 0.0]

    manifest = {
        "schema_version": 1,
        "category": category,
        "asset_id": asset_id,
        "name": name,
        "source_cad": source_cad,
        "source_url": source_url,
        "baked_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "has_superelement": False,
        "visual_glb": "visual.glb",
        "visual_triangles": int(vis_tris),
        "visual_bytes": int(vis_bytes),
        "ref_size_mm": ref_size_mm,
        **(extra_meta or {}),
    }
    (asset_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    clear_asset_cache()
    print(f"  [visual] {category}/{asset_id}: {vis_tris} tris, {vis_bytes/1024:.1f} KiB, size={ref_size_mm} mm")
    return asset_dir


def _bake_bracket_from_meshes(
    *,
    asset_id: str,
    name: str,
    source_cad: str,
    source_url: str,
    material_id: str,
    structural_mesh: trimesh.Trimesh,
    visual_parts: list[VisualPart],
    port0_mm: tuple[float, float, float],
    port1_mm: tuple[float, float, float],
    port0_radius_mm: float = 16.0,
    port1_radius_mm: float = 16.0,
    mesh_size_mm: float = 4.6,
    visual_max_triangles: int = 18000,
    is_fdm_anisotropic: bool = False,
    cad_trust_pct: int = 88,
    extra_recipe: dict[str, Any] | None = None,
) -> None:
    """Bake a single gantry bracket Tet4 CMS superelement + multi-part PBR visual GLB."""
    if _already_baked("gantry_brackets", asset_id, require_npz=True):
        return
    t0 = time.perf_counter()
    spec = BakedAssetSpec(
        category="gantry_brackets",
        asset_id=asset_id,
        name=name,
        source_cad=source_cad,
        source_url=source_url,
        reference_material_id=material_id,
        cad_trust_pct=cad_trust_pct,
        port0=PortDefinition(
            vp_coord_mm=port0_mm,
            selector=_selector_box_or_nearest(
                port0_mm, port0_radius_mm, min_nodes=16, other_center_mm=port1_mm
            ),
        ),
        port1=PortDefinition(
            vp_coord_mm=port1_mm,
            selector=_selector_box_or_nearest(
                port1_mm, port1_radius_mm, min_nodes=16, other_center_mm=port0_mm
            ),
        ),
        n_internal_modes=6,
        mesh_size_mm=mesh_size_mm,
        is_multi_material=False,
        is_fdm_anisotropic=is_fdm_anisotropic,
        part_recipe=extra_recipe or {},
        visual_max_triangles=visual_max_triangles,
    )
    res = bake_solid_superelement(
        spec,
        regions=[
            MaterialRegion(
                solid=structural_mesh,
                material_id=material_id,
                label=asset_id,
                fdm_anisotropic=is_fdm_anisotropic,
            )
        ],
        visual_parts=visual_parts,
    )
    dt = time.perf_counter() - t0
    ps = res.manifest["port_stiffness_n_per_mm"]
    print(
        f"  [CMS] gantry_brackets/{asset_id}: {res.manifest['num_elements']} tets, "
        f"mass={res.manifest['mass_kg']*1000:.1f}g, rb={res.manifest['rigid_body_max_freq_hz']:.4f}Hz, "
        f"k=({ps['kx']:.0f},{ps['ky']:.0f},{ps['kz']:.0f}) N/mm, "
        f"glb={res.manifest['visual_bytes']/1024:.0f}KiB ({dt:.1f}s)"
    )


def _classify_cnc_solid_style(
    label: str,
    primary_color: str,
    accent_color: str,
    *,
    is_cnc: bool = True,
) -> tuple[str, float, float, bool]:
    """Return ``(color_hex, metallic, roughness, is_structural)`` from STEP solid name."""
    lo = label.lower()
    # Skip loose T-nuts / tiny shims / set screws from heavy structural FEA
    if any(k in lo for k in ("stepper", "nema", "4248", "电机-")):
        if "shaft" in lo or "轴" in lo:
            return (COL_STAINLESS, 0.88, 0.22, False)
        return (COL_MOTOR_BODY, 0.55, 0.42, False)
    if any(k in lo for k in ("pulley", "gt2", "同步轮", "flange", "toothed")):
        return (COL_PULLEY_AL, 0.85, 0.26, False)
    if any(k in lo for k in ("bearing", "f695", "695", "轴承", "shaft", "pin", "spacer", "shim", "disc_spring", "standoff")):
        return (COL_STAINLESS, 0.86, 0.24, False)
    if any(k in lo for k in ("mgn", "rail", "滑块", "导轨")):
        return (COL_STEEL, 0.82, 0.28, False)
    if any(k in lo for k in ("fhcs", "shcs", "bhcs", "t-nut", "screw", "杯头", "圆头", "垫圈", "nut", "微动")):
        return (COL_CNC_DARK, 0.72, 0.35, False)
    if any(k in lo for k in ("rta", "rtb", "滑块", "tension", "盖")):
        return (accent_color, 0.78 if is_cnc else 0.18, 0.28 if is_cnc else 0.48, True)
    # Main bracket body / plates
    return (primary_color, 0.82 if is_cnc else 0.16, 0.26 if is_cnc else 0.48, True)


# =============================================================================
# GROUP 1: Voron StealthBurner + Clockwork 2.5 + LDO Nitehawk-SB v2.0
# =============================================================================
def bake_stealthburner() -> None:
    print("\n=== Baking Group: stealthburner ===")
    step_path = Path("cad_sources/voron_stealthburner/Stealthburner_CW2.5_Assembly.step")
    root = b3d.import_step(str(step_path))

    # MGN12H front carriage face in CAD is at X=-0.2, Y=29.1, Z=50.9
    shift = (0.2, -29.1, -50.9)

    visual_parts: list[VisualPart] = []
    structural_meshes: list[trimesh.Trimesh] = []

    for path, solid in _walk_named_solids(root):
        lo = path.lower()
        leaf = lo.split("/")[-1]
        # Skip X-axis extrusion, MGN12 rail/block (PrintSim draws them), ADXL mount,
        # and duplicate alternate STEP variants (ECAS04 body & retained hall-effect bracket sit
        # identically on top of Collet body / standard bracket and cancel out in voxel parity).
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
        # Skip tiny SMD components on NH-SB PCB to keep GLB crisp & lean (keep PCB board itself)
        if "nh-sb" in lo and any(
            k in lo for k in ("r_0402", "r_0603", "r_0805", "c_0402", "c_0805", "sot-", "d_sod", "l_0603", "fuse_", "crystal", "kicad_embedded")
        ):
            continue
        # Skip Revo Voron hotend hardware itself (PrintSim renders hotend separately at Port 1!)
        if "sb_r2_revovoron/hardware" in lo:
            continue

        tm = _shape_to_trimesh(solid, deflection=0.65)
        if len(tm.faces) == 0:
            continue
        tm = _transform_mesh(tm, translate_mm=shift)

        # Classify PBR styling & structural membership (hardware/fasteners BEFORE folder checks!)
        if any(
            k in lo
            for k in (
                "hardware",
                "screw",
                "insert",
                "m3x",
                "m3_",
                "magnet",
                "nut",
                "washer",
            )
        ):
            visual_parts.append(VisualPart(tm, "sb_hw", COL_STEEL, metallic=0.72, roughness=0.32))
        elif "shroud" in lo:
            if any(k in lo for k in ("fan", "5015", "4010", "blower")):
                visual_parts.append(VisualPart(tm, "sb_fan", COL_MOTOR_BODY, metallic=0.2, roughness=0.55))
            elif any(k in lo for k in ("led", "neopixel", "pcb", "diffuser")):
                visual_parts.append(VisualPart(tm, "sb_led", "#f8fafc", metallic=0.1, roughness=0.3))
            else:
                # Outer red 5015 cooling shroud is cosmetic/non-load-bearing (only 2 lower + 2 upper M3s)
                visual_parts.append(VisualPart(tm, "sb_shroud", COL_VORON_RED, metallic=0.15, roughness=0.42))
        elif "x_carriage_printed" in lo or "sb_r2_revovoron/printed_parts" in lo or "cw2_5_printed" in lo:
            visual_parts.append(VisualPart(tm, "sb_body", COL_VORON_BLACK, metallic=0.14, roughness=0.48))
            # Include ONLY the true load-bearing chassis from MGN12H face to Revo Voron hotend shelf & CW2 core
            # (exclude non-structural cable covers, umbilical anchor, spring latch/guidler/shuttle, probe bracket)
            if (
                "x_carriage_frame_" in leaf
                or "sb_r2_revovoron/printed_parts" in lo
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

    struct_combined = _combine_meshes(structural_meshes)
    bmin, bmax = struct_combined.bounds
    print(f"  StealthBurner structural bounds (mm): min={np.round(bmin,1)} max={np.round(bmax,1)}")

    # Port 0 = Exact MGN12H 4x M3x8 front carriage bolt face (Y = 0.0 mm, X = +-10 mm, Z = +-10 mm)
    # Port 1 = Exact 4x M3x8 Revo Voron hotend mounting shelf inside Stealthburner_CW2.5_Assembly.step:
    #   Heatsink X=[-9.18, 9.57] -> +0.20 mm, Y=[-35.90, -21.90] -> -28.90 mm, Z_top=-1.90 mm
    #   (49.08 mm Revo Voron nozzle tip lands at Z=-50.98 mm, 2.45 mm below red shroud bottom at Z=-48.53 mm;
    #    CW2_5_Main_Body bottom sits at Z=+10.60 mm so 62.2 mm Dragon HF / V6 top-mounts at Z=+10.60 mm -> tip Z=-51.6 mm).
    # Keep Port 0 and Port 1 strictly disjoint planar bolt pads (dmin ~ 19.8 mm, zero bridging tets).
    port0_mm = (0.0, 0.0, 0.0)
    port1_mm = (0.2, -28.9, -1.9)

    def _select_sb_mgn12h_face(coords_mm: np.ndarray) -> np.ndarray:
        mask = (
            (np.abs(coords_mm[:, 1]) <= 2.5)
            & (np.abs(coords_mm[:, 0]) <= 15.0)
            & (np.abs(coords_mm[:, 2]) <= 14.0)
        )
        if int(np.sum(mask)) >= 24:
            return mask
        return _selector_box_or_nearest((0.0, 0.0, 0.0), 10.0, min_nodes=24)(coords_mm)

    def _select_sb_hotend_shelf(coords_mm: np.ndarray) -> np.ndarray:
        mask = (
            (coords_mm[:, 2] >= -4.5)
            & (coords_mm[:, 2] <= 2.5)
            & (coords_mm[:, 0] >= -10.0)
            & (coords_mm[:, 0] <= 10.5)
            & (coords_mm[:, 1] >= -36.5)
            & (coords_mm[:, 1] <= -21.5)
        )
        if int(np.sum(mask)) >= 24:
            return mask
        return _selector_box_or_nearest(port1_mm, 8.0, min_nodes=24)(coords_mm)

    spec = BakedAssetSpec(
        category="toolheads",
        asset_id="stealthburner",
        name="Voron StealthBurner + Clockwork 2.5 (ABS/ASA)",
        source_cad="Stealthburner_CW2.5_Assembly.step",
        source_url="https://github.com/VoronDesign/Voron-Stealthburner",
        reference_material_id="abs_printed",
        cad_trust_pct=86,
        port0=PortDefinition(
            vp_coord_mm=port0_mm,
            selector=_select_sb_mgn12h_face,
        ),
        port1=PortDefinition(
            vp_coord_mm=port1_mm,
            selector=_select_sb_hotend_shelf,
        ),
        n_internal_modes=6,
        mesh_size_mm=4.0,
        is_multi_material=False,
        is_fdm_anisotropic=True,
        part_recipe={
            "origin": "MGN12H front carriage face (0,0,0)",
            "hotend_port_mm": list(port1_mm),
            "cw2_bottom_z_mm": 10.6,
            "shroud_bottom_z_mm": -48.53,
            "target_nozzle_tip_z_mm": -51.0,
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
                label="stealthburner_body",
                fdm_anisotropic=True,
            )
        ],
        visual_parts=visual_parts,
    )
    dt = time.perf_counter() - t0
    ps = res.manifest["port_stiffness_n_per_mm"]
    print(
        f"  [CMS] toolheads/stealthburner: {res.manifest['num_elements']} tets, "
        f"mass={res.manifest['mass_kg']*1000:.1f}g, rb={res.manifest['rigid_body_max_freq_hz']:.4f}Hz, "
        f"k=({ps['kx']:.0f},{ps['ky']:.0f},{ps['kz']:.0f}) N/mm, "
        f"glb={res.manifest['visual_bytes']/1024:.0f}KiB ({dt:.1f}s)"
    )


# =============================================================================
# GROUP 2: Monolith LDO CNC (User Onshape AWD + Official 2WD + Z-joint)
# =============================================================================
def _bake_monolith_step_station(
    step_path: Path,
    asset_id: str,
    name: str,
    source_url: str,
    shift_mm: tuple[float, float, float],
    port0_mm: tuple[float, float, float],
    port1_mm: tuple[float, float, float],
    *,
    primary_color: str = COL_CNC_SILVER,
    accent_color: str = COL_CNC_DARK,
    rx_deg: float = 0.0,
    rz_deg: float = 0.0,
    skip_substrings: tuple[str, ...] = ("t-nut", "mgn9h", "mgn12h"),
) -> None:
    if _already_baked("gantry_brackets", asset_id, require_npz=True):
        return
    root = b3d.import_step(str(step_path))
    vis_parts: list[VisualPart] = []
    struct_meshes: list[trimesh.Trimesh] = []
    tie_meshes: list[trimesh.Trimesh] = []
    for path, solid in _walk_named_solids(root):
        lo = path.lower()
        leaf = lo.split("/")[-1]
        if any(s in lo for s in skip_substrings):
            continue
        tm = _shape_to_trimesh(solid, deflection=0.85)
        if len(tm.faces) == 0:
            continue
        tm = _transform_mesh(tm, rx_deg=rx_deg, rz_deg=rz_deg, translate_mm=shift_mm)
        col, met, rough, is_struct = _classify_cnc_solid_style(path, primary_color, accent_color, is_cnc=True)
        # Never treat a stepper motor body or pulley as structural bracket metal
        if any(k in lo for k in ("stepper", "nema", "pulley", "bearing", "toothed")):
            is_struct = False
        vis_parts.append(VisualPart(tm, path.split("/")[-1][:28], col, metallic=met, roughness=rough))
        if is_struct:
            struct_meshes.append(tm)
        elif (
            any(k in leaf for k in ("standoff", "spacer", "pin", "m3x40", "m3x25", "m5x40"))
            and "stepper" not in lo
            and "nema" not in lo
        ):
            # Tie upper & lower sandwich plates in structural FEA via actual standoffs/spacers/pins
            # (check leaf name only so "..._with_45mm_long_shaft/Stepper_stator" is NEVER matched!)
            tie_meshes.append(tm)

    if len(struct_meshes) >= 2 and tie_meshes:
        struct_meshes.extend(tie_meshes)

    struct_combined = _combine_meshes(struct_meshes)
    _bake_bracket_from_meshes(
        asset_id=asset_id,
        name=name,
        source_cad=step_path.name,
        source_url=source_url,
        material_id="aluminum_6061_t6",
        structural_mesh=struct_combined,
        visual_parts=vis_parts,
        port0_mm=port0_mm,
        port1_mm=port1_mm,
        port0_radius_mm=18.0,
        port1_radius_mm=18.0,
        mesh_size_mm=4.6,
        visual_max_triangles=22000,
        is_fdm_anisotropic=False,
        cad_trust_pct=90,
        extra_recipe={"datum": "corner_or_carriage_centerline_0_0_0", "shift_mm": list(shift_mm)},
    )


def bake_monolith() -> None:
    print("\n=== Baking Group: monolith ===")
    onshape_url = "https://cad.onshape.com/documents/739b76283701789caa9999c7/w/434ca8ef5e541eb4a503de27"
    ldo_url = "https://github.com/Monolith3D/LDO_Monolith_Gantry_Kits"

    # User's Onshape 300mm Monolith AWD (460x460 outer -> extrusion centerlines at +-220 mm, Z=0)
    # 1. Rear Left & Rear Right (A/B motors + tensioners)
    _bake_monolith_step_station(
        Path("cad_sources/monolith_onshape/rear_left.step"),
        "monolith_awd_rear_left",
        "LDO Monolith CNC 6061 Rear-Left Drive (AWD 9mm)",
        onshape_url,
        shift_mm=(220.0, -220.0, 0.0),
        port0_mm=(0.0, 0.0, 0.0),      # 2020 corner extrusion mount
        port1_mm=(32.5, -8.0, 0.0),    # Stepper/pulley shaft center
    )
    _bake_monolith_step_station(
        Path("cad_sources/monolith_onshape/rear_right.step"),
        "monolith_awd_rear_right",
        "LDO Monolith CNC 6061 Rear-Right Drive (AWD 9mm)",
        onshape_url,
        shift_mm=(-220.0, -220.0, 0.0),
        port0_mm=(0.0, 0.0, 0.0),
        port1_mm=(-32.5, -8.0, 0.0),
    )

    # 2. Front Left & Front Right (AWD Front Motor Drives)
    _bake_monolith_step_station(
        Path("cad_sources/monolith_onshape/front_left.step"),
        "monolith_awd_front_left",
        "LDO Monolith CNC 6061 Front-Left Drive (AWD 9mm)",
        onshape_url,
        shift_mm=(220.0, 220.0, 0.0),
        port0_mm=(0.0, 0.0, 0.0),
        port1_mm=(32.5, 7.5, 0.0),
    )
    _bake_monolith_step_station(
        Path("cad_sources/monolith_onshape/front_right.step"),
        "monolith_awd_front_right",
        "LDO Monolith CNC 6061 Front-Right Drive (AWD 9mm)",
        onshape_url,
        shift_mm=(-220.0, 220.0, 0.0),
        port0_mm=(0.0, 0.0, 0.0),
        port1_mm=(-32.5, 7.5, 0.0),
    )

    # 3. XY Left & XY Right (AWD 9mm XY Joints)
    # Verified from VC9-JLA M5 bolt holes: Y-extrusion centerline at X=+-220 mm, X-beam centerline at Y=+23.50 mm!
    _bake_monolith_step_station(
        Path("cad_sources/monolith_onshape/xy_left.step"),
        "monolith_awd_xy_left",
        "LDO Monolith CNC 6061 XY-Joint Left (AWD 9mm)",
        onshape_url,
        shift_mm=(220.0, -23.5, 0.0),
        port0_mm=(18.0, -18.5, 0.0),   # Y-rail carriage face (VC9-JSC spine)
        port1_mm=(45.0, 0.0, 0.0),     # X-beam clamp end (exact M5 centerline Y=0)
    )
    _bake_monolith_step_station(
        Path("cad_sources/monolith_onshape/xy_right.step"),
        "monolith_awd_xy_right",
        "LDO Monolith CNC 6061 XY-Joint Right (AWD 9mm)",
        onshape_url,
        shift_mm=(-220.0, -23.5, 0.0),
        port0_mm=(-18.0, -18.5, 0.0),
        port1_mm=(-45.0, 0.0, 0.0),
    )

    # 4. Official LDO Monolith 2WD Front Idlers & 2WD XY Joints (250mm frame -> centerlines at +-195 mm)
    _bake_monolith_step_station(
        Path("cad_sources/ldo_monolith/repo/CAD/Front_left/LDO_2WD_VT_9mm.step"),
        "monolith_2wd_front_left",
        "LDO Monolith CNC 6061 Front-Left Idler (2WD 9mm)",
        ldo_url,
        shift_mm=(195.0, 195.0, 0.0),
        port0_mm=(0.0, 18.0, 0.0),
        port1_mm=(23.5, 4.5, 0.0),
    )
    _bake_monolith_step_station(
        Path("cad_sources/ldo_monolith/repo/CAD/Front_right/LDO_2WD_VT_9mm.step"),
        "monolith_2wd_front_right",
        "LDO Monolith CNC 6061 Front-Right Idler (2WD 9mm)",
        ldo_url,
        shift_mm=(-195.0, 195.0, 0.0),
        port0_mm=(0.0, 18.0, 0.0),
        port1_mm=(-23.5, 4.5, 0.0),
    )
    # 2WD XY joints: exact X=+-195.0 (250mm frame Y-extrusion centerline), Y=-23.5 (exact M5 X-beam centerline)
    for side, sgn in (("left", 1.0), ("right", -1.0)):
        stp = Path(f"cad_sources/ldo_monolith/repo/CAD/XY_{side}/LDO_2WD_9mm.step")
        _bake_monolith_step_station(
            stp,
            f"monolith_2wd_xy_{side}",
            f"LDO Monolith CNC 6061 XY-Joint {side.title()} (2WD 9mm)",
            ldo_url,
            shift_mm=(195.0 * sgn, -23.5, 0.0),
            port0_mm=(18.0 * sgn, -18.5, 0.0),
            port1_mm=(45.0 * sgn, 0.0, 0.0),
        )

    # 5. Official LDO Monolith Milled V2.4 Z-Joint
    # In LDO_Milled_V2_9mm_S_FT.step (250mm frame front-right corner):
    #   Column centerline = (+195, -195), gantry extrusion Z centerline = 0.0.
    #   MGN9H carriage at Y=[-183,-175] (+Y face of front column -> Y=[+12,+20] after shift),
    #   V29-ZSA arm at Y=[-175,-156] (-> Y=[+20,+39]), Z=[+19,+55] bolting onto top of gantry extrusion.
    zj_path = Path("cad_sources/ldo_monolith/repo/CAD/Z_joint/LDO_Milled_V2_9mm_S_FT.step")
    _bake_monolith_step_station(
        zj_path,
        "monolith_v24_z_joint",
        "LDO Monolith Milled CNC 6061 V2.4 Z-Joint",
        ldo_url,
        shift_mm=(-195.0, 195.0, 0.0),
        port0_mm=(0.0, 20.0, 37.0),    # MGN9H carriage bolt face on front column +Y Z-rail
        port1_mm=(0.0, 30.0, 20.0),    # Lower M5x40 gantry extrusion mount face
        skip_substrings=("t-nut", "mgn9h", "m5x10", "m5x40", "m5x4"),
    )


# =============================================================================
# GROUP 3: Siboor / GKD CNC Gantry (AWD & Dedicated 2WD)
# =============================================================================
def bake_siboor() -> None:
    print("\n=== Baking Group: siboor ===")
    step_path = Path("cad_sources/siboor/gkd_cnc/Voron-Trident-CNC-Gantry v1.stp")
    url = "https://github.com/GKD-Team/Voron-Trident-CNC-Gantry"
    root = b3d.import_step(str(step_path))

    # SolidWorks Y-up -> PrintSim Z-up: Rx(+90 deg) maps (X, Y_up, Z_front) -> (X, -Z_front, Y_up)
    # Verified from Gantry_Extrusions (250mm Trident = 410x410 outer, 390x390 center-to-center):
    #   - Left Y extrusion centerline:  X = 0.0   ([-10, +10])
    #   - Right Y extrusion centerline: X = 390.0 ([380, 400])
    #   - Front frame 2020 centerline:  Z_sw = +10.0  -> after Rx(+90): Y_ps = -10.0  (shift +10.0)
    #   - Rear frame 2020 centerline:   Z_sw = -380.0 -> after Rx(+90): Y_ps = +380.0 (shift -380.0)
    #   - X-beam 2020 centerline:       Z_sw = -183.6 -> after Rx(+90): Y_ps = +183.6 (shift -183.6)
    #   - Gantry extrusion Z centerline: Y_sw = 0.0   -> after Rx(+90): Z_ps = 0.0
    by_top: dict[str, Any] = {getattr(ch, "label", ""): ch for ch in (root.children or [])}

    def _extract_side_meshes(
        comp: Any,
        *,
        x_filter_fn,
        shift_mm: tuple[float, float, float],
        is_xy_station: bool = False,
        skip_keywords: tuple[str, ...] = ("hfs5", "导轨", "mgn"),
    ) -> tuple[list[VisualPart], list[trimesh.Trimesh]]:
        vis: list[VisualPart] = []
        structs: list[trimesh.Trimesh] = []
        ties: list[trimesh.Trimesh] = []
        for idx, (path, solid) in enumerate(_walk_named_solids(comp)):
            lo = path.lower()
            if any(k in lo for k in skip_keywords):
                continue
            tm = _shape_to_trimesh(solid, deflection=0.90)
            if len(tm.faces) == 0:
                continue
            tm_rot = _transform_mesh(tm, rx_deg=90.0)
            b0, b1 = tm_rot.bounds
            sz = b1 - b0  # (sx, sy, sz) in PrintSim Z-up mm
            cx = float(0.5 * (b0[0] + b1[0]))

            # Geometry-based exclusions for unnamed STEP compounds (AWD-X, AWD-AB):
            # 1) Skip 330mm X-extrusion & 300mm MGN12 rail & X-rail bolt row in XY stations
            if is_xy_station:
                if sz[0] > 95.0:
                    continue
                # Skip MGN9H carriage block (20 x 30..41 x 8 mm sitting at Z in [-21, -11])
                if (
                    abs(sz[0] - 20.0) < 1.2
                    and abs(sz[2] - 8.0) < 1.2
                    and 25.0 <= sz[1] <= 44.0
                    and b1[2] < -10.0
                ):
                    continue
                # Keep only parts belonging to Left (X < 70) or Right (X > 320) XY joint (skip X-rail screws)
                if 70.0 <= cx <= 320.0:
                    continue

            if not x_filter_fn(cx):
                continue

            tm_final = _transform_mesh(tm_rot, translate_mm=shift_mm)

            # Detect NEMA17 stepper motor body (+ shaft) even when STEP solid has empty label
            is_motor_geom = (
                (38.0 <= sz[0] <= 54.0 and 38.0 <= sz[1] <= 54.0 and sz[2] >= 44.0)
                or any(k in lo for k in ("stepper", "nema", "4248", "电机-"))
            )
            is_small_hw = max(sz[0], sz[1]) <= 18.0

            if is_motor_geom:
                col, met, rough, is_struct = COL_MOTOR_BODY, 0.55, 0.42, False
                part_name = f"nema17_motor_{idx}"
            elif is_small_hw:
                # Bearings, pulleys, shafts, bolts, standoffs
                is_pulley_or_brg = max(sz[0], sz[1]) >= 11.0
                col = COL_PULLEY_AL if is_pulley_or_brg else COL_STAINLESS
                met, rough, is_struct = 0.84, 0.26, False
                part_name = (path.split("/")[-1][:24] or f"hw_{idx}")
                # Tie vertical shafts/standoffs connecting upper & lower CNC plates in FEA
                if sz[2] >= 26.0 and max(sz[0], sz[1]) <= 9.0:
                    ties.append(tm_final)
            else:
                col, met, rough, is_struct = _classify_cnc_solid_style(
                    path, COL_SIBOOR_BLACK, COL_SIBOOR_RED, is_cnc=True
                )
                # Signature Siboor black + red anodized styling (even for unnamed AWD solids)
                if any(k in lo for k in ("滑块", "盖", "惰轮", "tension")) or (
                    not path.split("/")[-1] and sz[2] <= 12.0 and max(sz[0], sz[1]) <= 55.0
                ):
                    col = COL_SIBOOR_RED
                else:
                    col = COL_SIBOOR_BLACK
                part_name = (path.split("/")[-1][:24] or f"cnc_body_{idx}")

            vis.append(VisualPart(tm_final, part_name, col, metallic=met, roughness=rough))
            if is_struct:
                structs.append(tm_final)

        if len(structs) >= 2 and ties:
            structs.extend(ties)
        return vis, structs

    # 1. Siboor AWD Rear Left & Rear Right
    awd_rear = by_top["AWD-AB电机座-装配"]
    for side, xfn, sx, p1x in (
        ("left", lambda x: x < 195.0, 0.0, 55.0),
        ("right", lambda x: x >= 195.0, -390.0, -55.0),
    ):
        vis, structs = _extract_side_meshes(awd_rear, x_filter_fn=xfn, shift_mm=(sx, -380.0, 0.0))
        _bake_bracket_from_meshes(
            asset_id=f"siboor_awd_rear_{side}",
            name=f"Siboor CNC 6061 Rear-{side.title()} Drive (AWD)",
            source_cad=step_path.name,
            source_url=url,
            material_id="aluminum_6061_t6",
            structural_mesh=_combine_meshes(structs),
            visual_parts=vis,
            port0_mm=(0.0, 0.0, 0.0),
            port1_mm=(p1x, -16.0, 0.0),
            port0_radius_mm=22.0,
            port1_radius_mm=22.0,
            mesh_size_mm=5.0,
            visual_max_triangles=24000,
        )

    # 2. Siboor AWD Front Left & Front Right (Front corner centerline at Y_ps = -10.0 -> shift +10.0)
    awd_front = by_top["AWD前电机座【B版本】-装配"]
    for side, xfn, sx, p1x in (
        ("left", lambda x: x < 195.0, 0.0, 34.0),
        ("right", lambda x: x >= 195.0, -390.0, -34.0),
    ):
        vis, structs = _extract_side_meshes(awd_front, x_filter_fn=xfn, shift_mm=(sx, 10.0, 0.0))
        _bake_bracket_from_meshes(
            asset_id=f"siboor_awd_front_{side}",
            name=f"Siboor CNC 6061 Front-{side.title()} Drive (AWD)",
            source_cad=step_path.name,
            source_url=url,
            material_id="aluminum_6061_t6",
            structural_mesh=_combine_meshes(structs),
            visual_parts=vis,
            port0_mm=(0.0, 0.0, 0.0),
            port1_mm=(p1x, 18.0, 0.0),
            port0_radius_mm=20.0,
            port1_radius_mm=20.0,
            mesh_size_mm=5.0,
            visual_max_triangles=24000,
        )

    # 3. Siboor AWD XY Left & XY Right (X-beam centerline at Y_ps = +183.6 -> shift -183.6)
    awd_x = by_top["AWD-X组件-装配"]
    for side, xfn, sx, p1x in (
        ("left", lambda x: x < 195.0, 0.0, 32.0),
        ("right", lambda x: x >= 195.0, -390.0, -32.0),
    ):
        vis, structs = _extract_side_meshes(
            awd_x,
            x_filter_fn=xfn,
            shift_mm=(sx, -183.6, 0.0),
            is_xy_station=True,
            skip_keywords=("hfs5", "导轨", "mgn12", "m3x8-杯头"),
        )
        _bake_bracket_from_meshes(
            asset_id=f"siboor_awd_xy_{side}",
            name=f"Siboor CNC 6061 XY-Joint {side.title()} (AWD)",
            source_cad=step_path.name,
            source_url=url,
            material_id="aluminum_6061_t6",
            structural_mesh=_combine_meshes(structs),
            visual_parts=vis,
            port0_mm=(0.0, -14.0, -14.0),
            port1_mm=(p1x, 0.0, 0.0),
            port0_radius_mm=18.0,
            port1_radius_mm=18.0,
            mesh_size_mm=4.6,
            visual_max_triangles=22000,
        )

    # 4. Siboor 2WD Rear A/B Motor Mounts (Dedicated downward-hanging 2WD mounts!)
    rear_2wd = by_top["2WD-AB电机座-装配"]
    for side, xfn, sx, p1x in (
        ("left", lambda x: x < 195.0, 0.0, 55.0),
        ("right", lambda x: x >= 195.0, -390.0, -55.0),
    ):
        vis, structs = _extract_side_meshes(rear_2wd, x_filter_fn=xfn, shift_mm=(sx, -380.0, 0.0))
        _bake_bracket_from_meshes(
            asset_id=f"siboor_2wd_rear_{side}",
            name=f"Siboor CNC 6061 Rear-{side.title()} Drive (2WD)",
            source_cad=step_path.name,
            source_url=url,
            material_id="aluminum_6061_t6",
            structural_mesh=_combine_meshes(structs),
            visual_parts=vis,
            port0_mm=(0.0, 0.0, 0.0),
            port1_mm=(p1x, -16.0, 0.0),
            port0_radius_mm=22.0,
            port1_radius_mm=22.0,
            mesh_size_mm=5.0,
            visual_max_triangles=24000,
        )

    # 5. Siboor 2WD Front Idlers (Left & Right, Front corner centerline at Y_ps = -10.0 -> shift +10.0)
    f2wd = by_top["2WD前惰轮-装配"]
    for side, xfn, sx, p1x in (
        ("left", lambda x: x < 195.0, 0.0, 18.0),
        ("right", lambda x: x >= 195.0, -390.0, -18.0),
    ):
        vis, structs = _extract_side_meshes(f2wd, x_filter_fn=xfn, shift_mm=(sx, 10.0, 0.0))
        _bake_bracket_from_meshes(
            asset_id=f"siboor_2wd_front_{side}",
            name=f"Siboor CNC 6061 Front-{side.title()} Idler (2WD)",
            source_cad=step_path.name,
            source_url=url,
            material_id="aluminum_6061_t6",
            structural_mesh=_combine_meshes(structs),
            visual_parts=vis,
            port0_mm=(0.0, 18.0, 0.0),
            port1_mm=(p1x, 12.0, 0.0),
            port0_radius_mm=18.0,
            port1_radius_mm=18.0,
            mesh_size_mm=4.6,
            visual_max_triangles=22000,
        )

    # 6. Siboor 2WD XY Joints (Left & Right, X-beam centerline at Y_ps = +183.6 -> shift -183.6)
    x_2wd = by_top["2WD-X组件-装配"]
    for side, xfn, sx, p1x in (
        ("left", lambda x: x < 195.0, 0.0, 32.0),
        ("right", lambda x: x >= 195.0, -390.0, -32.0),
    ):
        vis, structs = _extract_side_meshes(
            x_2wd,
            x_filter_fn=xfn,
            shift_mm=(sx, -183.6, 0.0),
            is_xy_station=True,
            skip_keywords=("hfs5", "导轨", "mgn12", "m3x8-杯头"),
        )
        _bake_bracket_from_meshes(
            asset_id=f"siboor_2wd_xy_{side}",
            name=f"Siboor CNC 6061 XY-Joint {side.title()} (2WD)",
            source_cad=step_path.name,
            source_url=url,
            material_id="aluminum_6061_t6",
            structural_mesh=_combine_meshes(structs),
            visual_parts=vis,
            port0_mm=(0.0, -14.0, -14.0),
            port1_mm=(p1x, 0.0, 0.0),
            port0_radius_mm=18.0,
            port1_radius_mm=18.0,
            mesh_size_mm=4.6,
            visual_max_triangles=22000,
        )


# =============================================================================
# GROUP 4: Official Assembled Voron 2.4r2 & Trident Stock Hardware (STEP)
# =============================================================================
def _find_step_node(root: Any, path_parts: list[str]) -> tuple[Any, Any]:
    """Return ``(node, parent_world_location)`` for a subassembly path inside an imported STEP."""
    cur = root
    loc = b3d.Location()
    for part in path_parts:
        nl = getattr(cur, "location", None)
        if nl is not None:
            loc = loc * nl
        cur = next(ch for ch in (cur.children or []) if (getattr(ch, "label", "") or "") == part)
    return cur, loc


def _classify_voron_stock_solid(path: str) -> tuple[str, float, float, bool]:
    """Return ``(color_hex, metallic, roughness, is_structural_abs)`` for official Voron STEP parts."""
    lo = path.lower()
    leaf = lo.split("/")[-1]
    if any(k in lo for k in ("nema17", "motor_body", "stepper_motor", "nema17_stepper")):
        if "shaft" in leaf:
            return (COL_STAINLESS, 0.88, 0.22, False)
        return (COL_MOTOR_BODY, 0.58, 0.40, False)
    if any(k in lo for k in ("gt2_20t", "gt2_80t", "gt2_16t", "pulley")):
        return (COL_PULLEY_AL, 0.85, 0.26, False)
    if any(k in lo for k in ("f695", "625", "ge5c", "bearing_stack", "shaft", "5x60")):
        return (COL_STAINLESS, 0.86, 0.24, False)
    if any(k in lo for k in ("belt_loop", "188mm_belt", "rubber_foot")):
        return ("#181c24", 0.08, 0.82, False)
    if "leadscrew_nut" in lo:
        return ("#1e2430", 0.14, 0.65, False)
    if any(k in lo for k in ("m3x", "m5x", "shcs", "bhcs", "fhcs", "nut", "washer", "ptfe", "circlip", "hardware/")):
        return (COL_CNC_DARK, 0.72, 0.35, False)
    # Voron red accent printed parts
    if (
        "[a]_" in leaf
        or any(
            k in leaf
            for k in (
                "tensioner",
                "belt_tensioner",
                "cover_bearing",
                "cover_logo",
                "idler_front",
                "idler_carrier",
                "z_cover",
                "z_bed_",
                "z_belt_clamp_lower",
            )
        )
    ):
        return (COL_VORON_RED, 0.16, 0.44, True)
    # Primary black ABS/ASA printed structural housings
    return (COL_VORON_BLACK, 0.14, 0.48, True)


def _bake_voron_step_subassembly(
    root: Any,
    path_parts: list[str],
    *,
    asset_id: str,
    name: str,
    source_cad: str,
    source_url: str,
    shift_mm: tuple[float, float, float],
    port0_mm: tuple[float, float, float],
    port1_mm: tuple[float, float, float],
    port0_radius_mm: float = 20.0,
    port1_radius_mm: float = 20.0,
    mesh_size_mm: float = 4.6,
    visual_max_triangles: int = 24000,
    skip_substrings: tuple[str, ...] = ("mgn9h", "mgn_carriage"),
    skip_leaf_exact: tuple[str, ...] = (),
    z_range_mm: tuple[float, float] | None = None,
    tie_leaf_substrings: tuple[str, ...] = ("m5x40", "m5x30", "m3x40", "m3x30", "m5x16"),
    deflection: float = 0.75,
) -> None:
    """Extract an assembled subassembly directly in exact world coordinates from Voron STEP."""
    if _already_baked("gantry_brackets", asset_id, require_npz=True):
        return
    node, parent_loc = _find_step_node(root, path_parts)
    vis_parts: list[VisualPart] = []
    struct_meshes: list[trimesh.Trimesh] = []
    tie_meshes: list[trimesh.Trimesh] = []

    for path, solid in _walk_named_solids(node, "", parent_loc):
        lo = path.lower()
        leaf = lo.split("/")[-1]
        if leaf in skip_leaf_exact:
            continue
        if any(s in lo for s in skip_substrings):
            continue
        tm = _shape_to_trimesh(solid, deflection=deflection)
        if len(tm.faces) == 0:
            continue
        tm = _transform_mesh(tm, translate_mm=shift_mm)
        if z_range_mm is not None:
            cz = float(0.5 * (tm.bounds[0][2] + tm.bounds[1][2]))
            if cz < z_range_mm[0] or cz > z_range_mm[1]:
                continue
        col, met, rough, is_struct = _classify_voron_stock_solid(path)
        vis_parts.append(VisualPart(tm, leaf[:28] or "voron_part", col, metallic=met, roughness=rough))
        if is_struct:
            struct_meshes.append(tm)
        elif any(k in leaf for k in tie_leaf_substrings):
            tie_meshes.append(tm)

    if len(struct_meshes) >= 2 and tie_meshes:
        struct_meshes.extend(tie_meshes)

    struct_combined = _combine_meshes(struct_meshes)
    _bake_bracket_from_meshes(
        asset_id=asset_id,
        name=name,
        source_cad=source_cad,
        source_url=source_url,
        material_id="abs_printed",
        structural_mesh=struct_combined,
        visual_parts=vis_parts,
        port0_mm=port0_mm,
        port1_mm=port1_mm,
        port0_radius_mm=port0_radius_mm,
        port1_radius_mm=port1_radius_mm,
        mesh_size_mm=mesh_size_mm,
        visual_max_triangles=visual_max_triangles,
        is_fdm_anisotropic=True,
        cad_trust_pct=88,
        extra_recipe={
            "datum": "exact_official_step_assembled_coords",
            "step_subassembly": "/".join(path_parts),
            "shift_mm": list(shift_mm),
        },
    )


def bake_voron_stock_and_printed() -> None:
    print("\n=== Baking Group: voron_stock (Official Assembled STEPs) ===")
    v2_url = "https://github.com/VoronDesign/Voron-2"
    vt_url = "https://github.com/VoronDesign/Voron-Trident"
    vt_step = Path("cad_sources/voron_trident/extracted/trident_assembly.step")
    v2_step = Path("cad_sources/voron_24/extracted/Voron_2.4r2_Assembly.step")

    # ---- PART A: Official Voron Trident Assembly (trident_assembly.step) ----
    # Exact frame centerlines (250mm Trident = 410x410 outer, 390x390 center-to-center):
    #   X_L = -390.0, X_R = 0.0, X_MID = -195.0
    #   Y_F = 0.0,    Y_R = +390.0, Y_X_BEAM = +211.0
    #   Z_B = -510.0, Z_SUBFRAME = -243.3, Z_G = -160.0
    print(f"  Loading {vt_step} ...")
    rt = b3d.import_step(str(vt_step))

    # 1. Rear A/B Drive Mounts (with NEMA17 motors, 20T pulleys, F695 stacks, red bearing covers)
    _bake_voron_step_subassembly(
        rt,
        ["Gantry", "B/A_Drives", "B_Drive"],
        asset_id="voron_stock_rear_left",
        name="Voron Stock Printed B-Drive Mount (ABS/ASA)",
        source_cad=vt_step.name,
        source_url=vt_url,
        shift_mm=(390.0, -390.0, 160.0),
        port0_mm=(0.0, 0.0, 0.0),
        port1_mm=(49.0, -14.0, 0.0),
        port0_radius_mm=24.0,
        port1_radius_mm=24.0,
        mesh_size_mm=4.6,
        visual_max_triangles=22000,
        skip_substrings=("cover_logo", "y_endstop", "y_bumper"),
    )
    _bake_voron_step_subassembly(
        rt,
        ["Gantry", "B/A_Drives", "A_Drive"],
        asset_id="voron_stock_rear_right",
        name="Voron Stock Printed A-Drive Mount (ABS/ASA)",
        source_cad=vt_step.name,
        source_url=vt_url,
        shift_mm=(0.0, -390.0, 160.0),
        port0_mm=(0.0, 0.0, 0.0),
        port1_mm=(-49.0, -14.0, 0.0),
        port0_radius_mm=24.0,
        port1_radius_mm=24.0,
        mesh_size_mm=4.6,
        visual_max_triangles=22000,
        skip_substrings=("cover_logo", "y_endstop", "y_bumper"),
    )

    # 2. Front Idlers (with red tensioner front/carrier, F695 bearing stack, M3/M5 bolts)
    _bake_voron_step_subassembly(
        rt,
        ["Gantry", "Front_Idler_B"],
        asset_id="voron_stock_front_left",
        name="Voron Stock Printed Front-Left Idler (ABS/ASA)",
        source_cad=vt_step.name,
        source_url=vt_url,
        shift_mm=(390.0, 0.0, 160.0),
        port0_mm=(0.0, 18.0, 0.0),
        port1_mm=(19.0, 24.0, 0.0),
        port0_radius_mm=18.0,
        port1_radius_mm=18.0,
        mesh_size_mm=4.4,
        visual_max_triangles=20000,
        skip_substrings=("idler_carrier_b",),
    )
    _bake_voron_step_subassembly(
        rt,
        ["Gantry", "Front_Idler_A"],
        asset_id="voron_stock_front_right",
        name="Voron Stock Printed Front-Right Idler (ABS/ASA)",
        source_cad=vt_step.name,
        source_url=vt_url,
        shift_mm=(0.0, 0.0, 160.0),
        port0_mm=(0.0, 18.0, 0.0),
        port1_mm=(-19.0, 24.0, 0.0),
        port0_radius_mm=18.0,
        port1_radius_mm=18.0,
        mesh_size_mm=4.4,
        visual_max_triangles=20000,
        skip_substrings=("idler_carrier_b",),
    )

    # 3. Stock Voron XY Joints (MGN12 X / MGN9 bottom Y, exact X-beam centerline Y_X=+211.0)
    _bake_voron_step_subassembly(
        rt,
        ["Gantry", "X_Axis", "XY_Joint_-_Left"],
        asset_id="voron_stock_xy_left",
        name="Voron Stock Printed XY-Joint Left MGN12 (ABS/ASA)",
        source_cad=vt_step.name,
        source_url=vt_url,
        shift_mm=(390.0, -211.0, 160.0),
        port0_mm=(0.0, -14.0, -20.0),
        port1_mm=(45.0, 0.0, 0.0),
        port0_radius_mm=20.0,
        port1_radius_mm=20.0,
        mesh_size_mm=4.5,
        visual_max_triangles=22000,
        skip_substrings=("mgn9h", "cable_bridge", "d2f", "hall_effect", "endstop"),
    )
    _bake_voron_step_subassembly(
        rt,
        ["Gantry", "X_Axis", "XY_Joint_-_Right"],
        asset_id="voron_stock_xy_right",
        name="Voron Stock Printed XY-Joint Right MGN12 (ABS/ASA)",
        source_cad=vt_step.name,
        source_url=vt_url,
        shift_mm=(0.0, -211.0, 160.0),
        port0_mm=(0.0, -14.0, -20.0),
        port1_mm=(-45.0, 0.0, 0.0),
        port0_radius_mm=20.0,
        port1_radius_mm=20.0,
        mesh_size_mm=4.5,
        visual_max_triangles=22000,
        skip_substrings=("mgn9h", "cable_bridge", "d2f", "hall_effect", "endstop"),
        z_range_mm=(-33.0, 22.0),
    )

    # 4. Voron Trident Z Stepper Mounts (3 distinct stations: Left, Right, Rear — skip 333mm leadscrew rod only!)
    trident_z_mounts = (
        ("voron_trident_z_mount_left", "Z_Stepper_Left", "Left", (390.0, 0.0, 510.0), (15.0, 15.0, 0.0), (32.0, 36.0, 0.0)),
        ("voron_trident_z_mount_right", "Z_Stepper_Right", "Right", (0.0, 0.0, 510.0), (-15.0, 15.0, 0.0), (-32.0, 36.0, 0.0)),
        ("voron_trident_z_mount_rear", "Z_Stepper_Rear", "Rear", (195.0, -390.0, 510.0), (0.0, -10.0, 0.0), (0.0, -32.0, 0.0)),
        # Legacy alias (left) kept for backward compatibility
        ("voron_trident_z_mount", "Z_Stepper_Left", "Left", (390.0, 0.0, 510.0), (15.0, 15.0, 0.0), (32.0, 36.0, 0.0)),
    )
    for aid, sub_name, label_side, shift_mm, p0, p1 in trident_z_mounts:
        _bake_voron_step_subassembly(
            rt,
            ["Z_Assembly", "Z_Motors", sub_name],
            asset_id=aid,
            name=f"Voron Trident Z Leadscrew Motor Mount {label_side} (ABS/ASA)",
            source_cad=vt_step.name,
            source_url=vt_url,
            shift_mm=shift_mm,
            port0_mm=p0,
            port1_mm=p1,
            port0_radius_mm=22.0,
            port1_radius_mm=22.0,
            mesh_size_mm=4.8,
            visual_max_triangles=20000,
            skip_substrings=(),
            skip_leaf_exact=("leadscrew",),
        )

    # 5. Voron Trident Z Carriages & GE5C Bed Support Arms (3 distinct stations: Left, Right, Rear)
    # Aligned to z_bed_* Z_SUBFRAME = -243.3 mm!
    trident_bed_arms = (
        ("voron_trident_bed_arm_left", "Z_Left", "Left", (390.0, 0.0, 243.3), (0.0, 20.0, 18.0), (26.0, 68.0, 16.0)),
        ("voron_trident_bed_arm_right", "Z_Right", "Right", (0.0, 0.0, 243.3), (0.0, 20.0, 18.0), (-26.0, 68.0, 16.0)),
        ("voron_trident_bed_arm_rear", "Z_Rear", "Rear", (195.0, -390.0, 243.3), (-20.0, 0.0, 18.0), (0.0, -90.0, 13.0)),
        # Legacy alias (left) kept for backward compatibility
        ("voron_trident_bed_arm", "Z_Left", "Left", (390.0, 0.0, 243.3), (0.0, 20.0, 18.0), (26.0, 68.0, 16.0)),
    )
    for aid, sub_name, label_side, shift_mm, p0, p1 in trident_bed_arms:
        _bake_voron_step_subassembly(
            rt,
            ["Z_Assembly", sub_name],
            asset_id=aid,
            name=f"Voron Trident Z Bed Carriage & GE5C Arm {label_side} (ABS/ASA)",
            source_cad=vt_step.name,
            source_url=vt_url,
            shift_mm=shift_mm,
            port0_mm=p0,
            port1_mm=p1,
            port0_radius_mm=20.0,
            port1_radius_mm=20.0,
            mesh_size_mm=4.8,
            visual_max_triangles=22000,
            skip_substrings=("mgn9h", "mgn_carriage", "z_rear_insert_2hole"),
        )

    del rt

    # ---- PART B: Official Voron 2.4r2 Assembly (Voron_2.4r2_Assembly.step) ----
    # Exact frame centerlines (250mm Voron 2.4r2 = 410x410 outer, 390x390 center-to-center):
    #   X_L = -68.0, X_R = +322.0, Y_F = -28.0, Y_R = +362.0, Z_B = -48.1, Z_G = +78.0
    print(f"  Loading {v2_step} ...")
    r24 = b3d.import_step(str(v2_step))

    # 6. Voron 2.4 Belted Quad-Z Drive Pods (Left A__1_ at FL/RR, Right B__1_ at FR/RL)
    _bake_voron_step_subassembly(
        r24,
        ["Z_Assembly", "Z_Belt_Drive_A__1_"],
        asset_id="voron_24_z_drive_left",
        name="Voron 2.4r2 Belted Z-Drive Pod A (ABS/ASA)",
        source_cad=v2_step.name,
        source_url=v2_url,
        shift_mm=(68.0, 28.0, 48.1),
        port0_mm=(0.0, 10.0, -15.0),   # Bolts up into bottom 2020 frame extrusion
        port1_mm=(12.0, 24.0, -40.0),  # Main Z-drive shaft housing bearing boss
        port0_radius_mm=24.0,
        port1_radius_mm=24.0,
        mesh_size_mm=5.4,
        visual_max_triangles=24000,
        skip_substrings=(),
    )
    _bake_voron_step_subassembly(
        r24,
        ["Z_Assembly", "Z_Belt_Drive_B__1_"],
        asset_id="voron_24_z_drive_right",
        name="Voron 2.4r2 Belted Z-Drive Pod B (ABS/ASA)",
        source_cad=v2_step.name,
        source_url=v2_url,
        shift_mm=(-322.0, 28.0, 48.1),
        port0_mm=(0.0, 10.0, -15.0),
        port1_mm=(-12.0, 24.0, -40.0),
        port0_radius_mm=24.0,
        port1_radius_mm=24.0,
        mesh_size_mm=5.4,
        visual_max_triangles=24000,
        skip_substrings=(),
    )

    # 7. Voron 2.4 Flying Gantry Z-Joint (Front-Left Z_Joint__1_ -> rotY=0 at FL/FR, rotY=PI at RL/RR)
    _bake_voron_step_subassembly(
        r24,
        ["Gantry", "Z_Joints", "Z_Joint__1_"],
        asset_id="voron_24_z_joint",
        name="Voron 2.4r2 Flying Gantry Z-Joint (ABS/ASA)",
        source_cad=v2_step.name,
        source_url=v2_url,
        shift_mm=(68.0, 28.0, -78.0),
        port0_mm=(0.0, 20.0, -52.0),   # MGN9H carriage bolt face on front column +Y Z-rail
        port1_mm=(0.0, 32.0, -18.0),   # Top gantry extrusion bolt / Z-belt clamp face
        port0_radius_mm=16.0,
        port1_radius_mm=16.0,
        mesh_size_mm=4.4,
        visual_max_triangles=20000,
        skip_substrings=("mgn9h",),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Bake Voron / LDO / Siboor / Monolith CAD assets")
    parser.add_argument(
        "--group",
        choices=["all", "stealthburner", "monolith", "siboor", "voron_stock"],
        default="all",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip assets that already have manifest.json + visual.glb (+ superelement.npz)",
    )
    args = parser.parse_args()
    global SKIP_EXISTING
    SKIP_EXISTING = bool(args.skip_existing)
    t0 = time.perf_counter()
    if args.group in ("all", "stealthburner"):
        bake_stealthburner()
    if args.group in ("all", "monolith"):
        bake_monolith()
    if args.group in ("all", "siboor"):
        bake_siboor()
    if args.group in ("all", "voron_stock"):
        bake_voron_stock_and_printed()
    print(f"\nCompleted group '{args.group}' in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
