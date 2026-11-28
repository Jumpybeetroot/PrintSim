#!/usr/bin/env python3
"""Bake 3D Tet4 Craig-Bampton superelements & intra-part 3D fields for CNC milled X-beams.

Bakes the 3 user-selectable CNC milled lightweight X-beams under
``src/printsim/assets/hardware/x_beams/``:
  1. ``cnc_al_fysetc_v0_15x15`` — FYSETC Hollow CNC 15×15×200 mm (Voron 0)
  2. ``cnc_al_monocoque_20x20`` — PrintSim CNC Monocoque Truss 20×20×380 mm
  3. ``cnc_al_truss_voron24_lw`` — Voron 2.4 Lightweight Milled 20×20×380 mm (Emil Wallberg)

Each STEP solid is aligned to the exact canonical frame of its existing PBR
``visual.glb`` (beam axis along X in ``[-L/2, +L/2]``, centered in Y/Z),
meshed into a 3D Tet4 continuum, condensed via RBE3 Virtual-Point-Transform +
Craig-Bampton CMS with Port 0 at ``(-L/2, 0, 0)`` and Port 1 at ``(+L/2, 0, 0)``,
and populated with exact intra-part ``field_points_m``, ``field_disp_op``,
``field_stress_op``, and ``subpart_*`` recovery operators.

Existing high-detail PBR ``visual.glb`` files and FEA section-property metadata
in ``manifest.json`` are preserved byte-for-byte.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import build123d as b3d
import numpy as np

from printsim.cad_assets.baker import (
    BakedAssetSpec,
    MaterialRegion,
    PortDefinition,
    bake_solid_superelement,
)
from printsim.cad_assets.registry import ASSETS_ROOT

REPO_ROOT = Path(__file__).resolve().parents[1]
CAD_DIR = REPO_ROOT / "cad_sources" / "milled_x_beams"


def _load_and_align_solid(
    step_path: Path,
    *,
    rot_y_deg: float = 0.0,
    rot_x_deg: float = 0.0,
    center_xyz: bool = True,
) -> b3d.Solid:
    """Load the primary solid from a STEP file and align it to the canonical GLB frame."""
    shape = b3d.import_step(str(step_path))
    solids = shape.solids()
    if not solids:
        raise ValueError(f"No solids found in {step_path}")
    solid = max(solids, key=lambda s: float(s.volume))

    if center_xyz:
        bb = solid.bounding_box()
        cx = 0.5 * (bb.min.X + bb.max.X)
        cy = 0.5 * (bb.min.Y + bb.max.Y)
        cz = 0.5 * (bb.min.Z + bb.max.Z)
        solid = solid.moved(b3d.Location((-cx, -cy, -cz)))

    if abs(rot_y_deg) > 1e-9:
        solid = solid.rotate(b3d.Axis.Y, rot_y_deg)
    if abs(rot_x_deg) > 1e-9:
        solid = solid.rotate(b3d.Axis.X, rot_x_deg)

    # Re-center after rotation to eliminate floating-point drift
    if center_xyz:
        bb2 = solid.bounding_box()
        cx2 = 0.5 * (bb2.min.X + bb2.max.X)
        cy2 = 0.5 * (bb2.min.Y + bb2.max.Y)
        cz2 = 0.5 * (bb2.min.Z + bb2.max.Z)
        if abs(cx2) > 1e-6 or abs(cy2) > 1e-6 or abs(cz2) > 1e-6:
            solid = solid.moved(b3d.Location((-cx2, -cy2, -cz2)))

    return solid


def bake_all_milled_xbeams() -> None:
    """Bake 3D Tet4 CB superelements + 3D intra-part fields for all 3 CNC milled X-beams."""
    configs: list[dict[str, Any]] = [
        {
            "asset_id": "cnc_al_fysetc_v0_15x15",
            "name": "FYSETC Hollow CNC 15x15 X-Beam (Voron 0)",
            "step_path": CAD_DIR / "fysetc_v0" / "Fysetc X-Beam [Voron0 200mm].step",
            "source_cad": "GrabCAD Nicotina52 FYSETC X-Beam Voron0 200mm (Fysetc X-Beam [Voron0 200mm].step, 15x15x200mm)",
            "source_url": "https://grabcad.com/library/fysetc-x-beam-voron0-200mm-1",
            "material_id": "aluminum_6061_t6",
            "rot_y_deg": 0.0,
            "rot_x_deg": 0.0,
            "mesh_size_mm": 3.0,
            "port_band_mm": 5.0,
            "cad_trust_pct": 93,
        },
        {
            "asset_id": "cnc_al_monocoque_20x20",
            "name": "PrintSim CNC Monocoque Truss 20x20 X-Beam",
            "step_path": CAD_DIR / "printsim_monocoque_truss_20x20.step",
            "source_cad": "Parametric build123d 20x20x380mm semi-monocoque Warren-truss beam (solid MGN12H front flange + 3-face CNC pockets)",
            "source_url": "https://github.com/Jumpybeetroot/PrintSim",
            "material_id": "aluminum_6061_t6",
            "rot_y_deg": 0.0,
            "rot_x_deg": 0.0,
            "mesh_size_mm": 3.8,
            "port_band_mm": 6.0,
            "cad_trust_pct": 94,
        },
        {
            "asset_id": "cnc_al_truss_voron24_lw",
            "name": "Voron 2.4 Lightweight Milled 20x20 X-Beam (Emil Wallberg)",
            "step_path": CAD_DIR / "voron24_lw" / "Voron 2.4 LW X-Axis" / "step" / "Voron_XAxis_LW_300.STEP",
            "source_cad": "GrabCAD Emil Wallberg Voron 2.4 Lightweight X-Axis 300mm (Voron_XAxis_LW_300.STEP, 19.6x19.6x380mm)",
            "source_url": "https://grabcad.com/library/voron-2-4-lightweight-x-axis-1",
            "material_id": "aluminum_6061_t6",
            "rot_y_deg": 90.0,
            "rot_x_deg": 0.0,
            "mesh_size_mm": 3.8,
            "port_band_mm": 6.0,
            "cad_trust_pct": 93,
        },
    ]

    for cfg in configs:
        asset_id = cfg["asset_id"]
        asset_dir = ASSETS_ROOT / "x_beams" / asset_id
        glb_path = asset_dir / "visual.glb"
        manifest_path = asset_dir / "manifest.json"

        # Preserve existing high-detail PBR visual.glb and FEA section properties
        existing_glb_bytes = glb_path.read_bytes() if glb_path.is_file() else None
        existing_manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file()
            else {}
        )

        solid = _load_and_align_solid(
            cfg["step_path"],
            rot_y_deg=cfg["rot_y_deg"],
            rot_x_deg=cfg["rot_x_deg"],
            center_xyz=True,
        )
        bb = solid.bounding_box()
        x_min, x_max = float(bb.min.X), float(bb.max.X)
        length_mm = x_max - x_min
        half_len = 0.5 * length_mm
        band = float(cfg["port_band_mm"])

        def _sel_left(coords: np.ndarray, xm: float = x_min, b: float = band) -> np.ndarray:
            return coords[:, 0] <= (xm + b)

        def _sel_right(coords: np.ndarray, xp: float = x_max, b: float = band) -> np.ndarray:
            return coords[:, 0] >= (xp - b)

        spec = BakedAssetSpec(
            category="x_beams",
            asset_id=asset_id,
            name=cfg["name"],
            source_cad=cfg["source_cad"],
            source_url=cfg["source_url"],
            reference_material_id=cfg["material_id"],
            cad_trust_pct=cfg["cad_trust_pct"],
            port0=PortDefinition(vp_coord_mm=(-half_len, 0.0, 0.0), selector=_sel_left),
            port1=PortDefinition(vp_coord_mm=(+half_len, 0.0, 0.0), selector=_sel_right),
            n_internal_modes=6,
            mesh_size_mm=cfg["mesh_size_mm"],
            visual_max_triangles=16000,
            part_recipe={
                "kind": "cnc_milled_x_beam",
                "step_source": str(cfg["step_path"].relative_to(REPO_ROOT)),
                "length_mm": round(length_mm, 3),
                "width_mm": round(float(bb.size.Y), 3),
                "height_mm": round(float(bb.size.Z), 3),
            },
        )

        print(
            f"=== Baking {asset_id} (L={length_mm:.1f}mm, "
            f"W={bb.size.Y:.2f}mm, H={bb.size.Z:.2f}mm, h={cfg['mesh_size_mm']}mm) ===",
            flush=True,
        )
        res = bake_solid_superelement(
            spec,
            regions=[
                MaterialRegion(
                    solid=solid,
                    material_id=cfg["material_id"],
                    label="cnc_xbeam_body",
                )
            ],
            assets_root=ASSETS_ROOT,
        )

        # Restore preserved PBR visual.glb and merge preserved manifest metadata
        if existing_glb_bytes is not None:
            glb_path.write_bytes(existing_glb_bytes)

        merged_manifest = dict(res.manifest)
        merged_manifest["has_superelement"] = True
        merged_manifest["length_mm"] = float(existing_manifest.get("length_mm", round(length_mm, 2)))
        merged_manifest["width_mm"] = float(existing_manifest.get("width_mm", round(float(bb.size.Y), 2)))
        merged_manifest["height_mm"] = float(existing_manifest.get("height_mm", round(float(bb.size.Z), 2)))
        if "source" in existing_manifest:
            merged_manifest["source"] = existing_manifest["source"]
        if "fea_section_properties" in existing_manifest:
            merged_manifest["fea_section_properties"] = existing_manifest["fea_section_properties"]
        if existing_glb_bytes is not None:
            merged_manifest["visual_triangles"] = int(
                existing_manifest.get("visual_triangles", merged_manifest.get("visual_triangles", 16000))
            )
            merged_manifest["visual_bytes"] = len(existing_glb_bytes)

        manifest_path.write_text(json.dumps(merged_manifest, indent=2), encoding="utf-8")

        with np.load(asset_dir / "superelement.npz") as npz:
            n_fpts = int(npz["field_points_m"].shape[0]) if "field_points_m" in npz else 0

        print(
            f"[OK] {asset_id}: mass={merged_manifest['mass_kg']*1000:.2f}g, "
            f"elems={merged_manifest['num_elements']}, "
            f"rb_max={merged_manifest['rigid_body_max_freq_hz']:.5f}Hz, "
            f"field_pts={n_fpts}, "
            f"k_port={merged_manifest['port_stiffness_n_per_mm']}",
            flush=True,
        )


if __name__ == "__main__":
    bake_all_milled_xbeams()
