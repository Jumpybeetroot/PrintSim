#!/usr/bin/env python3
"""Bake 3D CAD Craig-Bampton superelements & PBR GLBs for PrintSim frame corner braces.

Generates real-world dimensioned 2020 / 3030 / 4040 corner bracing hardware under
``src/printsim/assets/hardware/frame_braces/``:

1. Die-cast aluminum gusseted 90° L-corner brackets (Misumi HBLFSN5 / HBLFSN6 / HBLFSN8 style):
   - ``cast_corner_bracket_2020`` (28 × 28 × 20 mm)
   - ``cast_corner_bracket_3030`` (38 × 38 × 28 mm)
   - ``cast_corner_bracket_4040`` (48 × 48 × 38 mm)
   Canonical local frame:
   - Origin ``(0, 0, 0)`` is the inner corner vertex where the two extrusion faces meet.
   - Leg 0 extends along ``+X`` with mating face on ``y = 0`` (Port 0 at ``(bolt_offset, 0, 0)``).
   - Leg 1 extends along ``+Y`` with mating face on ``x = 0`` (Port 1 at ``(0, bolt_offset, 0)``).
   - Width is centered symmetrically on ``Z in [-width/2, +width/2]``.

2. CNC 6061-T6 / 5052 aluminum 5-hole L-corner gusset plates (RatRig / Monolith / DoomCube style):
   - ``cnc_corner_gusset_2020`` (60 × 60 × 5.0 mm)
   - ``cnc_corner_gusset_3030`` (90 × 90 × 6.0 mm)
   - ``cnc_corner_gusset_4040`` (120 × 120 × 6.0 mm)
   Canonical local frame:
   - Origin ``(0, 0, 0)`` is the outer corner of the two mating extrusions on the back mating face ``z = 0``.
   - Plate extends along ``+Z`` in ``[0, thickness]``.
   - Extrusion 0 centerline is at ``y = ext_w / 2`` along ``+X``; Extrusion 1 centerline is at ``x = ext_w / 2`` along ``+Y``.
   - Port 0 at ``(port_offset, ext_w / 2, 0)``, Port 1 at ``(ext_w / 2, port_offset, 0)``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import build123d as b3d
import numpy as np

from printsim.cad_assets.baker import (
    BakedAssetSpec,
    MaterialRegion,
    PortDefinition,
    VisualPart,
    bake_solid_superelement,
)
from printsim.cad_assets.registry import ASSETS_ROOT


def _make_shcs_bolt(
    center_xyz: tuple[float, float, float],
    axis: str,
    head_r: float,
    head_h: float,
    socket_r: float,
) -> b3d.Solid:
    """Create a lightweight visual socket-head / button-head cap screw."""
    cx, cy, cz = center_xyz
    if axis == "+Z":
        plane = b3d.Plane.XY.offset(cz)
        with b3d.BuildPart(plane) as bp:
            with b3d.Locations((cx, cy)):
                b3d.Cylinder(radius=head_r, height=head_h, align=(b3d.Align.CENTER, b3d.Align.CENTER, b3d.Align.MIN))
            with b3d.Locations((cx, cy, head_h)):
                b3d.Cylinder(
                    radius=socket_r,
                    height=head_h * 0.55,
                    align=(b3d.Align.CENTER, b3d.Align.CENTER, b3d.Align.MAX),
                    mode=b3d.Mode.SUBTRACT,
                )
        return bp.part.solids()[0]
    if axis == "+Y":
        plane = b3d.Plane.XZ.offset(-cy)
        with b3d.BuildPart(plane) as bp:
            with b3d.Locations((cx, cz)):
                b3d.Cylinder(radius=head_r, height=head_h, align=(b3d.Align.CENTER, b3d.Align.CENTER, b3d.Align.MIN))
        return bp.part.solids()[0]
    # "+X"
    plane = b3d.Plane.YZ.offset(cx)
    with b3d.BuildPart(plane) as bp:
        with b3d.Locations((cy, cz)):
            b3d.Cylinder(radius=head_r, height=head_h, align=(b3d.Align.CENTER, b3d.Align.CENTER, b3d.Align.MIN))
    return bp.part.solids()[0]


def _build_cast_corner_bracket(
    *,
    arm_mm: float,
    width_mm: float,
    wall_mm: float,
    rib_t_mm: float,
    bolt_offset_mm: float,
    slot_w_mm: float,
    slot_l_mm: float,
) -> tuple[b3d.Solid, list[VisualPart]]:
    """Build a realistic Misumi HBLFSN-style die-cast gusseted 90° L-corner bracket."""
    with b3d.BuildPart() as bp:
        # 1. X-leg flange (x in [0, arm], y in [0, wall], z in [-W/2, +W/2])
        with b3d.Locations((arm_mm * 0.5, wall_mm * 0.5, 0.0)):
            b3d.Box(arm_mm, wall_mm, width_mm)
        # 2. Y-leg flange (x in [0, wall], y in [0, arm], z in [-W/2, +W/2])
        with b3d.Locations((wall_mm * 0.5, arm_mm * 0.5, 0.0)):
            b3d.Box(wall_mm, arm_mm, width_mm)

        # 3. Dual outer triangular gusset cheeks (classic Misumi HBLFSN5 pocketed casting)
        cheek_t = rib_t_mm
        cheek_z_offsets = (-(width_mm * 0.5 - cheek_t * 0.5), +(width_mm * 0.5 - cheek_t * 0.5))
        for z_off in cheek_z_offsets:
            with b3d.BuildSketch(b3d.Plane.XY.offset(z_off)) as sk:
                with b3d.BuildLine():
                    b3d.Polyline(
                        (wall_mm - 0.5, wall_mm - 0.5),
                        (arm_mm - 1.5, wall_mm - 0.5),
                        (arm_mm - 1.5, wall_mm + 2.5),
                        (wall_mm + 2.5, arm_mm - 1.5),
                        (wall_mm - 0.5, arm_mm - 1.5),
                        close=True,
                    )
                b3d.make_face()
            b3d.extrude(amount=cheek_t * 0.5, both=True)

        # 4. Elongated T-slot mounting holes on both legs
        with b3d.Locations((bolt_offset_mm, wall_mm * 0.5, 0.0)):
            b3d.Box(slot_l_mm, wall_mm + 2.0, slot_w_mm, mode=b3d.Mode.SUBTRACT)
        with b3d.Locations((wall_mm * 0.5, bolt_offset_mm, 0.0)):
            b3d.Box(wall_mm + 2.0, slot_l_mm, slot_w_mm, mode=b3d.Mode.SUBTRACT)

    solid = bp.part.solids()[0]

    # Visual-only fasteners seated on each leg
    bolt_r = slot_w_mm * 0.80
    bolt_h = max(3.0, wall_mm * 0.72)
    b0 = _make_shcs_bolt((bolt_offset_mm, wall_mm, 0.0), "+Y", bolt_r, bolt_h, bolt_r * 0.45)
    b1 = _make_shcs_bolt((wall_mm, bolt_offset_mm, 0.0), "+X", bolt_r, bolt_h, bolt_r * 0.45)
    visuals = [
        VisualPart(
            shape_or_mesh=solid,
            label="cast_bracket_body",
            color_hex="#B8C0C8",
            metallic=0.82,
            roughness=0.40,
        ),
        VisualPart(
            shape_or_mesh=b0,
            label="fastener_x_leg",
            color_hex="#2B3038",
            metallic=0.85,
            roughness=0.32,
        ),
        VisualPart(
            shape_or_mesh=b1,
            label="fastener_y_leg",
            color_hex="#2B3038",
            metallic=0.85,
            roughness=0.32,
        ),
    ]
    return solid, visuals


def _build_cnc_corner_gusset(
    *,
    ext_w_mm: float,
    leg_len_mm: float,
    thickness_mm: float,
    bolt_d_mm: float,
) -> tuple[b3d.Solid, list[VisualPart]]:
    """Build a RatRig / Monolith / DoomCube style 5-hole CNC aluminum L-gusset plate."""
    # Classic 5-hole triangular-hypotenuse corner plate:
    # (0, 0) -> (leg_len, 0) -> (leg_len, ext_w) -> (ext_w, leg_len) -> (0, leg_len)
    with b3d.BuildPart() as bp:
        with b3d.BuildSketch(b3d.Plane.XY) as sk:
            with b3d.BuildLine():
                b3d.Polyline(
                    (0.0, 0.0),
                    (leg_len_mm, 0.0),
                    (leg_len_mm, ext_w_mm),
                    (ext_w_mm, leg_len_mm),
                    (0.0, leg_len_mm),
                    close=True,
                )
            b3d.make_face()
        b3d.extrude(amount=thickness_mm)

        # 5 standard T-slot mounting holes along the two extrusion centerlines
        half_w = ext_w_mm * 0.5
        hole_pts = [
            (half_w, half_w),
            (half_w + ext_w_mm, half_w),
            (half_w + 2.0 * ext_w_mm, half_w),
            (half_w, half_w + ext_w_mm),
            (half_w, half_w + 2.0 * ext_w_mm),
        ]
        for hx, hy in hole_pts:
            if hx < leg_len_mm - bolt_d_mm * 0.6 and hy < leg_len_mm - bolt_d_mm * 0.6:
                with b3d.Locations((hx, hy, 0.0)):
                    b3d.Cylinder(
                        radius=bolt_d_mm * 0.5,
                        height=thickness_mm + 2.0,
                        align=(b3d.Align.CENTER, b3d.Align.CENTER, b3d.Align.MIN),
                        mode=b3d.Mode.SUBTRACT,
                    )

    solid = bp.part.solids()[0]

    visuals = [
        VisualPart(
            shape_or_mesh=solid,
            label="cnc_gusset_plate",
            color_hex="#3A424E",
            metallic=0.86,
            roughness=0.28,
        )
    ]
    half_w = ext_w_mm * 0.5
    hole_pts = [
        (half_w, half_w),
        (half_w + ext_w_mm, half_w),
        (half_w + 2.0 * ext_w_mm, half_w),
        (half_w, half_w + ext_w_mm),
        (half_w, half_w + 2.0 * ext_w_mm),
    ]
    head_r = bolt_d_mm * 0.85
    head_h = max(2.2, thickness_mm * 0.48)
    for idx, (hx, hy) in enumerate(hole_pts):
        bolt = _make_shcs_bolt((hx, hy, thickness_mm), "+Z", head_r, head_h, head_r * 0.45)
        visuals.append(
            VisualPart(
                shape_or_mesh=bolt,
                label=f"fastener_{idx}",
                color_hex="#CFD6DF",
                metallic=0.90,
                roughness=0.24,
            )
        )
    return solid, visuals


def bake_all_frame_braces() -> None:
    """Bake all 6 cast corner bracket & CNC corner gusset CAD superelements."""
    cast_configs = [
        {
            "asset_id": "cast_corner_bracket_2020",
            "name": "Cast Al 2020 Corner Bracket (HBLFSN5 28×28×20 mm)",
            "profile_series": "2020",
            "ext_w_mm": 20.0,
            "arm_mm": 28.0,
            "width_mm": 20.0,
            "wall_mm": 4.5,
            "rib_t_mm": 3.2,
            "bolt_offset_mm": 14.0,
            "slot_w_mm": 5.5,
            "slot_l_mm": 9.0,
            "mesh_size_mm": 2.4,
            "unit_cost_usd": 1.40,
        },
        {
            "asset_id": "cast_corner_bracket_3030",
            "name": "Cast Al 3030 Corner Bracket (HBLFSN6 38×38×28 mm)",
            "profile_series": "3030",
            "ext_w_mm": 30.0,
            "arm_mm": 38.0,
            "width_mm": 28.0,
            "wall_mm": 5.5,
            "rib_t_mm": 4.0,
            "bolt_offset_mm": 19.0,
            "slot_w_mm": 6.5,
            "slot_l_mm": 11.0,
            "mesh_size_mm": 3.0,
            "unit_cost_usd": 2.20,
        },
        {
            "asset_id": "cast_corner_bracket_4040",
            "name": "Cast Al 4040 Corner Bracket (HBLFSN8 48×48×38 mm)",
            "profile_series": "4040",
            "ext_w_mm": 40.0,
            "arm_mm": 48.0,
            "width_mm": 38.0,
            "wall_mm": 6.5,
            "rib_t_mm": 4.8,
            "bolt_offset_mm": 24.0,
            "slot_w_mm": 8.5,
            "slot_l_mm": 13.0,
            "mesh_size_mm": 3.6,
            "unit_cost_usd": 3.40,
        },
    ]

    for cfg in cast_configs:
        solid, visuals = _build_cast_corner_bracket(
            arm_mm=cfg["arm_mm"],
            width_mm=cfg["width_mm"],
            wall_mm=cfg["wall_mm"],
            rib_t_mm=cfg["rib_t_mm"],
            bolt_offset_mm=cfg["bolt_offset_mm"],
            slot_w_mm=cfg["slot_w_mm"],
            slot_l_mm=cfg["slot_l_mm"],
        )
        bolt_off = cfg["bolt_offset_mm"]
        pad_min = max(cfg["wall_mm"] + 2.0, bolt_off - 7.5)
        pad_max = cfg["arm_mm"] + 1.0
        tol_face = 1.6

        def _sel_p0(coords: np.ndarray, pmin: float = pad_min, pmax: float = pad_max, tf: float = tol_face) -> np.ndarray:
            return (coords[:, 1] <= tf) & (coords[:, 0] >= pmin) & (coords[:, 0] <= pmax)

        def _sel_p1(coords: np.ndarray, pmin: float = pad_min, pmax: float = pad_max, tf: float = tol_face) -> np.ndarray:
            return (coords[:, 0] <= tf) & (coords[:, 1] >= pmin) & (coords[:, 1] <= pmax)

        spec = BakedAssetSpec(
            category="frame_braces",
            asset_id=cfg["asset_id"],
            name=cfg["name"],
            source_cad=f"Parametric build123d CAD (Misumi HBLFSN {cfg['profile_series']} datasheet)",
            source_url="https://us.misumi-ec.com/vona2/detail/110300465900/",
            reference_material_id="aluminum_6061_t6",
            cad_trust_pct=88,
            port0=PortDefinition(vp_coord_mm=(bolt_off, 0.0, 0.0), selector=_sel_p0),
            port1=PortDefinition(vp_coord_mm=(0.0, bolt_off, 0.0), selector=_sel_p1),
            n_internal_modes=4,
            mesh_size_mm=cfg["mesh_size_mm"],
            visual_max_triangles=2200,
            part_recipe={
                "kind": "cast_corner_bracket",
                "profile_series": cfg["profile_series"],
                "ext_w_mm": cfg["ext_w_mm"],
                "arm_mm": cfg["arm_mm"],
                "width_mm": cfg["width_mm"],
                "bolt_offset_mm": bolt_off,
                "unit_cost_usd": cfg["unit_cost_usd"],
            },
        )
        res = bake_solid_superelement(
            spec,
            regions=[MaterialRegion(solid=solid, material_id="aluminum_6061_t6", label="cast_bracket_body")],
            visual_parts=visuals,
            assets_root=ASSETS_ROOT,
        )
        print(
            f"[OK] {cfg['asset_id']}: mass={res.manifest['mass_kg']*1000:.1f}g, "
            f"elems={res.manifest['num_elements']}, "
            f"rb_max={res.manifest['rigid_body_max_freq_hz']:.4f}Hz, "
            f"k_port={res.manifest['port_stiffness_n_per_mm']}"
        )

    cnc_configs = [
        {
            "asset_id": "cnc_corner_gusset_2020",
            "name": "CNC Al 2020 5-Hole Corner Gusset Plate (60×60×5 mm)",
            "profile_series": "2020",
            "ext_w_mm": 20.0,
            "leg_len_mm": 60.0,
            "thickness_mm": 5.0,
            "bolt_d_mm": 5.5,
            "mesh_size_mm": 3.0,
            "unit_cost_usd": 3.80,
        },
        {
            "asset_id": "cnc_corner_gusset_3030",
            "name": "CNC Al 3030 5-Hole Corner Gusset Plate (90×90×6 mm)",
            "profile_series": "3030",
            "ext_w_mm": 30.0,
            "leg_len_mm": 90.0,
            "thickness_mm": 6.0,
            "bolt_d_mm": 6.5,
            "mesh_size_mm": 4.0,
            "unit_cost_usd": 5.60,
        },
        {
            "asset_id": "cnc_corner_gusset_4040",
            "name": "CNC Al 4040 5-Hole Corner Gusset Plate (120×120×6 mm)",
            "profile_series": "4040",
            "ext_w_mm": 40.0,
            "leg_len_mm": 120.0,
            "thickness_mm": 6.0,
            "bolt_d_mm": 8.5,
            "mesh_size_mm": 5.0,
            "unit_cost_usd": 7.80,
        },
    ]

    for cfg in cnc_configs:
        solid, visuals = _build_cnc_corner_gusset(
            ext_w_mm=cfg["ext_w_mm"],
            leg_len_mm=cfg["leg_len_mm"],
            thickness_mm=cfg["thickness_mm"],
            bolt_d_mm=cfg["bolt_d_mm"],
        )
        ew = cfg["ext_w_mm"]
        half_w = ew * 0.5
        port_off = 2.0 * ew  # midpoint between 2nd (1.5*ew) and 3rd (2.5*ew) bolt on each leg
        pad_start = ew * 1.15  # strictly disjoint from corner square [0, ew] x [0, ew]
        tol_z = 1.6

        def _sel_g0(coords: np.ndarray, ps: float = pad_start, w: float = ew, tz: float = tol_z) -> np.ndarray:
            return (coords[:, 2] <= tz) & (coords[:, 0] >= ps) & (coords[:, 1] <= w + 1.0)

        def _sel_g1(coords: np.ndarray, ps: float = pad_start, w: float = ew, tz: float = tol_z) -> np.ndarray:
            return (coords[:, 2] <= tz) & (coords[:, 1] >= ps) & (coords[:, 0] <= w + 1.0)

        spec = BakedAssetSpec(
            category="frame_braces",
            asset_id=cfg["asset_id"],
            name=cfg["name"],
            source_cad=f"Parametric build123d CAD ({int(cfg['leg_len_mm'])}×{int(cfg['leg_len_mm'])}×{cfg['thickness_mm']:.0f}mm 5-hole Al 6061-T6 plate)",
            source_url="https://github.com/Jumpybeetroot/PrintSim",
            reference_material_id="aluminum_6061_t6",
            cad_trust_pct=90,
            port0=PortDefinition(vp_coord_mm=(port_off, half_w, 0.0), selector=_sel_g0),
            port1=PortDefinition(vp_coord_mm=(half_w, port_off, 0.0), selector=_sel_g1),
            n_internal_modes=4,
            mesh_size_mm=cfg["mesh_size_mm"],
            visual_max_triangles=2200,
            part_recipe={
                "kind": "cnc_corner_gusset",
                "profile_series": cfg["profile_series"],
                "ext_w_mm": ew,
                "leg_len_mm": cfg["leg_len_mm"],
                "thickness_mm": cfg["thickness_mm"],
                "port_offset_mm": port_off,
                "unit_cost_usd": cfg["unit_cost_usd"],
            },
        )
        res = bake_solid_superelement(
            spec,
            regions=[MaterialRegion(solid=solid, material_id="aluminum_6061_t6", label="cnc_gusset_plate")],
            visual_parts=visuals,
            assets_root=ASSETS_ROOT,
        )
        print(
            f"[OK] {cfg['asset_id']}: mass={res.manifest['mass_kg']*1000:.1f}g, "
            f"elems={res.manifest['num_elements']}, "
            f"rb_max={res.manifest['rigid_body_max_freq_hz']:.4f}Hz, "
            f"k_port={res.manifest['port_stiffness_n_per_mm']}"
        )


if __name__ == "__main__":
    bake_all_frame_braces()
