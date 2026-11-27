#!/usr/bin/env python3
"""Bake RatRig EVA 3 (PET-CF / ASA) Tet4 Craig-Bampton CMS superelement + PBR GLB.

Source: Official EVA 3 Onshape Assembly (5af1985e8b5d6fb20ee622b9) exported to
``cad_sources/onshape/eva3_orbiter2_rapido.step``.
"""

from __future__ import annotations

from pathlib import Path
import time

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
STEP_PATH = REPO_ROOT / "cad_sources" / "onshape" / "eva3_orbiter2_rapido.step"

COL_CHARCOAL = "#22262e"
COL_RATRIG_GREEN = "#84cc16"
COL_DUCT_BLACK = "#1c2028"
COL_FAN_BODY = "#1b202a"
COL_MOTOR_BODY = "#1a1f29"
COL_STEEL = "#9ca8b8"

# Structural printed EVA 3 core solids (front_universal, back_core_xy, top_endstop_mgn12c,
# hotend_mosquito, drive_lgx, shroud_chonky, bottom_horns, 40mm_fan_inlet, belt_grabbers, probe_mount)
STRUCTURAL_INDICES = {0, 2, 9, 18, 24, 27, 41, 52, 53, 54, 76, 117, 126}
ACCENT_INDICES = {0, 2, 18, 24, 52, 53}
FAN_INDICES = {31, 82}
EXTRUDER_INDICES = {134, 138, 144, 146, 147, 150}


def _shape_to_trimesh(shape: b3d.Shape, deflection: float = 0.50) -> trimesh.Trimesh:
    verts, tris = shape.tessellate(deflection)
    v_arr = np.asarray([[v.X, v.Y, v.Z] for v in verts], dtype=np.float64)
    f_arr = np.asarray(tris, dtype=np.int64)
    return trimesh.Trimesh(vertices=v_arr, faces=f_arr, process=False)


def _selector_box(
    x_lim: tuple[float, float],
    y_lim: tuple[float, float],
    z_lim: tuple[float, float],
    vp_mm: tuple[float, float, float],
):
    def _fn(pts_mm: np.ndarray) -> np.ndarray:
        mask = (
            (pts_mm[:, 0] >= x_lim[0])
            & (pts_mm[:, 0] <= x_lim[1])
            & (pts_mm[:, 1] >= y_lim[0])
            & (pts_mm[:, 1] <= y_lim[1])
            & (pts_mm[:, 2] >= z_lim[0])
            & (pts_mm[:, 2] <= z_lim[1])
        )
        if int(np.sum(mask)) >= 6:
            return mask
        d2 = np.sum((pts_mm - np.asarray(vp_mm, dtype=float)) ** 2, axis=1)
        idx = np.argsort(d2)[: max(8, min(20, len(pts_mm)))]
        out = np.zeros(len(pts_mm), dtype=bool)
        out[idx] = True
        return out

    return _fn


def main() -> None:
    if not STEP_PATH.exists():
        raise FileNotFoundError(f"Missing EVA 3 STEP: {STEP_PATH}")

    t0 = time.perf_counter()
    root = b3d.import_step(str(STEP_PATH))
    solids = root.solids()

    # Align Port 0 = bottom face center of top_endstop_mgn12c (MGN12C top carriage mount) at (0,0,0)
    bb_top = solids[117].bounding_box()
    bb_he = solids[76].bounding_box()
    p0_raw = (0.5 * (bb_top.min.X + bb_top.max.X), 0.5 * (bb_top.min.Y + bb_top.max.Y), bb_top.min.Z)
    p1_raw = (0.5 * (bb_he.min.X + bb_he.max.X), 0.5 * (bb_he.min.Y + bb_he.max.Y), bb_he.min.Z)
    port0_mm = (0.0, 0.0, 0.0)
    port1_mm = (round(p1_raw[0] - p0_raw[0], 2), round(p1_raw[1] - p0_raw[1], 2), round(p1_raw[2] - p0_raw[2], 2))

    structural_meshes: list[trimesh.Trimesh] = []
    visual_parts: list[VisualPart] = []

    for idx, solid in enumerate(solids):
        if solid.volume < 400.0:  # skip tiny M3 nuts/screws (< 0.4 cm3)
            continue
        moved = solid.moved(b3d.Location((-p0_raw[0], -p0_raw[1], -p0_raw[2])))
        tm = _shape_to_trimesh(moved, deflection=0.50)
        if len(tm.faces) == 0:
            continue

        if idx in STRUCTURAL_INDICES:
            structural_meshes.append(tm)
            if idx in ACCENT_INDICES:
                visual_parts.append(VisualPart(tm, f"eva3_accent_{idx}", COL_RATRIG_GREEN, metallic=0.14, roughness=0.42))
            elif idx == 27:
                visual_parts.append(VisualPart(tm, f"eva3_duct_{idx}", COL_DUCT_BLACK, metallic=0.14, roughness=0.45))
            else:
                visual_parts.append(VisualPart(tm, f"eva3_core_{idx}", COL_CHARCOAL, metallic=0.16, roughness=0.44))
        elif idx in FAN_INDICES:
            visual_parts.append(VisualPart(tm, f"eva3_fan_{idx}", COL_FAN_BODY, metallic=0.18, roughness=0.48))
        elif idx in EXTRUDER_INDICES:
            visual_parts.append(VisualPart(tm, f"eva3_extruder_{idx}", COL_MOTOR_BODY, metallic=0.35, roughness=0.40))
        elif idx == 42:
            visual_parts.append(VisualPart(tm, "eva3_probe", COL_STEEL, metallic=0.75, roughness=0.30))

    # Add thin 3 mm internal structural tie-block bridging top_endstop_mgn12c -> front_universal -> back_core_xy
    # (represents the 4x M3/M5 steel through-bolts clamping the EVA 3 core onto the MGN12C carriage)
    bridge = b3d.Box(36.0, 34.0, 14.0).translate((0.0, -14.0, -4.0))
    structural_meshes.append(_shape_to_trimesh(bridge, deflection=0.50))

    struct_combined = trimesh.util.concatenate(structural_meshes)
    bmin, bmax = struct_combined.bounds
    print(f"EVA 3 structural bounds (mm): min={np.round(bmin, 2)} max={np.round(bmax, 2)} | port1={port1_mm}")

    spec = BakedAssetSpec(
        category="toolheads",
        asset_id="eva3",
        name="RatRig EVA 3 (PET-CF / ASA)",
        source_cad="EVA 3 Onshape Assembly (5af1985e8b5d6fb20ee622b9)",
        source_url="https://cad.onshape.com/documents/5af1985e8b5d6fb20ee622b9/w/64ab471748be7627dcc5d993/e/154cc3182cd0a0ee76e3be5e",
        reference_material_id="pet_cf_printed",
        cad_trust_pct=84,
        port0=PortDefinition(
            vp_coord_mm=port0_mm,
            selector=_selector_box((-20.0, 20.0), (-14.0, 14.0), (-2.5, 3.0), port0_mm),
        ),
        port1=PortDefinition(
            vp_coord_mm=port1_mm,
            selector=_selector_box(
                (port1_mm[0] - 16.0, port1_mm[0] + 16.0),
                (port1_mm[1] - 14.0, port1_mm[1] + 14.0),
                (port1_mm[2] - 3.0, port1_mm[2] + 3.5),
                port1_mm,
            ),
        ),
        n_internal_modes=6,
        mesh_size_mm=3.5,
        is_multi_material=False,
        is_fdm_anisotropic=True,
        part_recipe={
            "origin": "MGN12C top carriage mount face (0,0,0)",
            "hotend_port_mm": list(port1_mm),
            "x_rail_mount": "top",
            "multi_piece_bolted": True,
        },
        visual_max_triangles=30000,
    )

    res = bake_solid_superelement(
        spec,
        regions=[
            MaterialRegion(
                solid=struct_combined,
                material_id="pet_cf_printed",
                label="eva3_body",
                fdm_anisotropic=True,
            )
        ],
        visual_parts=visual_parts,
    )
    clear_asset_cache()
    qa = verify_baked_asset("toolheads", "eva3")
    print(f"Baked EVA 3 in {time.perf_counter() - t0:.1f}s | QA passed={qa.passed} | checks={qa.to_dict().get('checks')}")
    m = res.manifest
    print(f"  k_t={m['port_stiffness_n_per_mm']}")
    print(f"  k_r={m['port_rot_stiffness_nm_per_rad']}")
    print(f"  mass={m['mass_kg']*1000:.1f} g | rigid_max={m['rigid_body_max_freq_hz']:.4f} Hz")


if __name__ == "__main__":
    main()
