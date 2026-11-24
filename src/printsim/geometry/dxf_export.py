"""2D CNC / Laser-cut DXF panel and structural plate exporter powered by ``ezdxf``."""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ezdxf

from printsim.geometry.builder import trident_backbone_dimensions_mm
from printsim.schema import PrinterSpec, get_material, get_profile

DXF_SPACING_MM = 30.0

VALID_DXF_PARTS: tuple[str, ...] = (
    "side_panel",
    "rear_panel",
    "top_panel",
    "trident_backbone",
    "corner_gusset",
    "all_plates",
)

_ContourPt = tuple[float, float] | tuple[float, float, float]


@dataclass
class _Plate2D:
    """Internal 2D CNC/laser plate representation in millimeters."""

    part_id: str
    title: str
    width_mm: float
    height_mm: float
    thickness_mm: float
    material_label: str
    contours: list[list[_ContourPt]]
    holes: list[tuple[float, float, float]]  # (cx_mm, cy_mm, diameter_mm)
    description: str
    hole_contours: list[list[_ContourPt]] = field(default_factory=list)


def _safe_material_name(material_id: str) -> str:
    try:
        return get_material(material_id).name
    except KeyError:
        return material_id


def _ext_width_mm(spec: PrinterSpec) -> float:
    try:
        return float(get_profile(spec.frame.profile).width_m) * 1000.0
    except KeyError:
        return 20.0


def _panel_outer_contour(
    width_mm: float,
    height_mm: float,
    clip_notches: bool,
    notch_mm: float,
) -> list[tuple[float, float]]:
    if not clip_notches:
        return [
            (0.0, 0.0),
            (width_mm, 0.0),
            (width_mm, height_mm),
            (0.0, height_mm),
        ]
    n = min(notch_mm, 0.15 * min(width_mm, height_mm))
    return [
        (n, 0.0),
        (width_mm - n, 0.0),
        (width_mm - n, n),
        (width_mm, n),
        (width_mm, height_mm - n),
        (width_mm - n, height_mm - n),
        (width_mm - n, height_mm),
        (n, height_mm),
        (n, height_mm - n),
        (0.0, height_mm - n),
        (0.0, n),
        (n, n),
    ]


def _perimeter_holes(
    width_mm: float,
    height_mm: float,
    inset_mm: float,
    diameter_mm: float,
    target_pitch_mm: float = 75.0,
) -> list[tuple[float, float, float]]:
    x0 = inset_mm
    x1 = max(inset_mm + 10.0, width_mm - inset_mm)
    y0 = inset_mm
    y1 = max(inset_mm + 10.0, height_mm - inset_mm)
    nx = max(2, int(round((x1 - x0) / target_pitch_mm)) + 1)
    ny = max(2, int(round((y1 - y0) / target_pitch_mm)) + 1)
    xs = [x0 + (x1 - x0) * i / (nx - 1) for i in range(nx)]
    ys = [y0 + (y1 - y0) * j / (ny - 1) for j in range(ny)]
    holes: list[tuple[float, float, float]] = []
    for x in xs:
        holes.append((x, y0, diameter_mm))
        holes.append((x, y1, diameter_mm))
    for y in ys[1:-1]:
        holes.append((x0, y, diameter_mm))
        holes.append((x1, y, diameter_mm))
    return holes


def _clip_locator_holes(
    width_mm: float,
    height_mm: float,
    inset_mm: float,
    diameter_mm: float = 3.4,
) -> list[tuple[float, float, float]]:
    return [
        (0.5 * width_mm, inset_mm, diameter_mm),
        (0.5 * width_mm, height_mm - inset_mm, diameter_mm),
        (inset_mm, 0.5 * height_mm, diameter_mm),
        (width_mm - inset_mm, 0.5 * height_mm, diameter_mm),
    ]


def _panel_mounting_geometry(
    spec: PrinterSpec,
    width_mm: float,
    height_mm: float,
    ext_w_mm: float,
) -> tuple[list[tuple[float, float]], list[tuple[float, float, float]]]:
    inset_mm = max(5.0, min(0.5 * ext_w_mm - 3.0, 0.45 * ext_w_mm))
    is_bolted = spec.frame.panel_mounting == "bolted_structural"
    if is_bolted:
        bolt_d = 5.5 if spec.frame.panel_thickness_mm >= 3.5 else 3.4
        contour = _panel_outer_contour(width_mm, height_mm, clip_notches=False, notch_mm=0.0)
        holes = _perimeter_holes(width_mm, height_mm, inset_mm, bolt_d, target_pitch_mm=75.0)
        return contour, holes
    notch_mm = min(14.0, 0.65 * ext_w_mm)
    contour = _panel_outer_contour(width_mm, height_mm, clip_notches=True, notch_mm=notch_mm)
    holes = _clip_locator_holes(width_mm, height_mm, inset_mm, diameter_mm=3.4)
    return contour, holes


def _build_side_panel(spec: PrinterSpec, title: str = "Side Enclosure Panel") -> _Plate2D:
    ext_w = _ext_width_mm(spec)
    lip_total = min(12.0, max(6.0, 0.30 * ext_w))
    w = max(80.0, float(spec.frame.depth_mm) - lip_total)
    h = max(80.0, float(spec.frame.height_mm) - lip_total)
    contour, holes = _panel_mounting_geometry(spec, w, h, ext_w)
    return _Plate2D(
        part_id="side_panel",
        title=title,
        width_mm=w,
        height_mm=h,
        thickness_mm=float(spec.frame.panel_thickness_mm),
        material_label=_safe_material_name(spec.frame.panel_material),
        contours=[contour],
        holes=holes,
        description=(
            f"Left/Right enclosure panel ({spec.frame.panel_mounting}), "
            f"{len(holes)} mounting holes"
        ),
    )


def _build_rear_panel(spec: PrinterSpec) -> _Plate2D:
    ext_w = _ext_width_mm(spec)
    lip_total = min(12.0, max(6.0, 0.30 * ext_w))
    w = max(80.0, float(spec.frame.width_mm) - lip_total)
    h = max(80.0, float(spec.frame.height_mm) - lip_total)
    contour, holes = _panel_mounting_geometry(spec, w, h, ext_w)
    inset_mm = max(5.0, min(0.5 * ext_w - 3.0, 0.45 * ext_w))

    # Lower cable & electronics pass-through cutout
    cut_w = min(max(45.0, 0.24 * w), 130.0)
    cut_h = min(max(25.0, 0.10 * h), 55.0)
    cx = 0.5 * w
    y_bot = max(inset_mm + 16.0, 0.06 * h)
    contours = [contour]
    if cx - 0.5 * cut_w > inset_mm + 10.0 and y_bot + cut_h < h - inset_mm - 10.0:
        contours.append(
            [
                (cx - 0.5 * cut_w, y_bot),
                (cx + 0.5 * cut_w, y_bot),
                (cx + 0.5 * cut_w, y_bot + cut_h),
                (cx - 0.5 * cut_w, y_bot + cut_h),
            ]
        )
        # 4x M3 cable gland / pass-through bezel holes
        for hx in (cx - 0.5 * cut_w - 6.0, cx + 0.5 * cut_w + 6.0):
            for hy in (y_bot - 5.0, y_bot + cut_h + 5.0):
                holes.append((hx, hy, 3.4))

    return _Plate2D(
        part_id="rear_panel",
        title="Rear Enclosure Panel",
        width_mm=w,
        height_mm=h,
        thickness_mm=float(spec.frame.panel_thickness_mm),
        material_label=_safe_material_name(spec.frame.panel_material),
        contours=contours,
        holes=holes,
        description=(
            f"Rear panel with lower cable/electronics pass-through ({cut_w:.0f}x{cut_h:.0f}mm)"
        ),
    )


def _build_top_panel(spec: PrinterSpec) -> _Plate2D:
    ext_w = _ext_width_mm(spec)
    lip_total = min(12.0, max(6.0, 0.30 * ext_w))
    w = max(80.0, float(spec.frame.width_mm) - lip_total)
    d = max(80.0, float(spec.frame.depth_mm) - lip_total)
    contour, holes = _panel_mounting_geometry(spec, w, d, ext_w)
    return _Plate2D(
        part_id="top_panel",
        title="Top Enclosure Panel",
        width_mm=w,
        height_mm=d,
        thickness_mm=float(spec.frame.panel_thickness_mm),
        material_label=_safe_material_name(spec.frame.panel_material),
        contours=[contour],
        holes=holes,
        description=f"Top enclosure panel ({spec.frame.panel_mounting}), {len(holes)} holes",
    )


def _rounded_rect_contour(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    radius_mm: float,
) -> list[_ContourPt]:
    """Build a closed CCW rectangular contour with corner fillets via DXF arc bulges."""
    r = min(max(float(radius_mm), 0.0), 0.45 * min(x1 - x0, y1 - y0))
    if r <= 1e-6:
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    b = math.tan(math.pi / 8.0)
    return [
        (x0 + r, y0, 0.0),
        (x1 - r, y0, b),
        (x1, y0 + r, 0.0),
        (x1, y1 - r, b),
        (x1 - r, y1, 0.0),
        (x0 + r, y1, b),
        (x0, y1 - r, 0.0),
        (x0, y0 + r, b),
    ]


def _obround_slot_contour(
    cx: float,
    cy: float,
    length_mm: float,
    width_mm: float,
    *,
    horizontal: bool,
) -> list[_ContourPt]:
    """Build a closed CCW stadium/obround slot contour using 180-deg semicircular DXF bulges."""
    r = 0.5 * width_mm
    half_straight = max(0.0, 0.5 * length_mm - r)
    if horizontal:
        return [
            (cx - half_straight, cy - r, 0.0),
            (cx + half_straight, cy - r, 1.0),
            (cx + half_straight, cy + r, 0.0),
            (cx - half_straight, cy + r, 1.0),
        ]
    return [
        (cx + r, cy - half_straight, 0.0),
        (cx + r, cy + half_straight, 1.0),
        (cx - r, cy + half_straight, 0.0),
        (cx - r, cy - half_straight, 1.0),
    ]


def _build_trident_backbone(spec: PrinterSpec) -> _Plate2D:
    ext_w = _ext_width_mm(spec)
    dims = trident_backbone_dimensions_mm(spec)
    w = float(spec.frame.width_mm)
    d = float(spec.frame.depth_mm)
    flange = dims["flange_width_mm"]
    cutout_w = dims["cutout_w_mm"]
    cutout_d = dims["cutout_d_mm"]
    recess = dims["rear_recess_mm"]

    # Inner open cutout bounds with R10 mm corner fillets
    x0 = 0.5 * (w - cutout_w)
    x1 = 0.5 * (w + cutout_w)
    y0 = min(flange, 0.5 * (d - cutout_d))
    y1 = y0 + cutout_d

    # Outer contour features:
    # - 2x front 20x20 mm vertical-post reliefs
    # - Rear edge recessed 20 mm inward on left/right wings with 86.1 x 20 mm protruding Z-motor tab
    # - 20.4 x 20.1 mm U-shaped rear Z-rail clearance notch centered in the rear tab
    # - 25 x 8 mm rear Z-chain wiring notch left of the rear tab
    corner_cut = min(20.0, max(10.0, 0.25 * flange))
    y_wing = d - recess
    tab_w = min(86.1, max(44.0, 0.35 * cutout_w))
    tab_x0 = 0.5 * w - 0.5 * tab_w
    tab_x1 = 0.5 * w + 0.5 * tab_w
    rail_notch_w = min(20.4, 0.30 * tab_w)
    rail_notch_d = min(20.1, recess + 0.1) if recess > 0.0 else 12.0
    rail_x0 = 0.5 * w - 0.5 * rail_notch_w
    rail_x1 = 0.5 * w + 0.5 * rail_notch_w
    y_rail_bot = d - rail_notch_d

    wire_notch_w = min(25.0, max(12.0, 0.25 * max(tab_x0 - x0, 20.0)))
    wire_notch_d = min(8.0, 0.15 * flange)
    wire_gap = min(12.0, max(4.0, 0.12 * max(tab_x0 - x0, 20.0)))
    wire_x1 = max(x0 + wire_notch_w + 4.0, tab_x0 - wire_gap)
    wire_x0 = wire_x1 - wire_notch_w
    y_wire_bot = y_wing - wire_notch_d

    outer: list[_ContourPt] = [
        (corner_cut, 0.0),
        (w - corner_cut, 0.0),
        (w - corner_cut, corner_cut),
        (w, corner_cut),
        (w, y_wing),
        (tab_x1, y_wing),
        (tab_x1, d),
        (rail_x1, d),
        (rail_x1, y_rail_bot),
        (rail_x0, y_rail_bot),
        (rail_x0, d),
        (tab_x0, d),
        (tab_x0, y_wing),
        (wire_x1, y_wing),
        (wire_x1, y_wire_bot),
        (wire_x0, y_wire_bot),
        (wire_x0, y_wing),
        (0.0, y_wing),
        (0.0, corner_cut),
        (corner_cut, corner_cut),
    ]
    inner_cutout = _rounded_rect_contour(x0, y0, x1, y1, radius_mm=min(10.0, 0.2 * min(cutout_w, cutout_d)))
    contours: list[list[_ContourPt]] = [outer, inner_cutout]

    # 3x Z-motor mounts: FL (41.8, 41.8), FR (w - 41.8, 41.8), Rear-Center (w/2, d - 49.9)
    # Each has 1x Ø26.0 mm pilot/leadscrew bore + 4x Ø3.5 mm M3 holes on a 31.0 x 31.0 mm square
    front_off = 41.8 if flange >= 72.0 else max(22.0, 0.52 * flange)
    rear_flange_span = d - y1 - recess
    rear_off = 49.9 if rear_flange_span >= 72.0 else max(26.0, recess + 0.37 * max(rear_flange_span, 30.0))
    motor_centers = [
        (front_off, front_off),
        (w - front_off, front_off),
        (0.5 * w, d - rear_off),
    ]
    holes: list[tuple[float, float, float]] = []
    for mx, my in motor_centers:
        holes.append((mx, my, 26.0))
        for dx in (-15.5, 15.5):
            for dy in (-15.5, 15.5):
                holes.append((mx + dx, my + dy, 3.5))

    # 10x Ø5.4 mm M5 frame-mounting holes: 4 FL, 4 FR, 2 Rear-Center tab
    t_cl = min(10.0, 0.5 * ext_w)
    m5_a = corner_cut + 15.0
    m5_b = min(flange - 10.0, corner_cut + 50.0)
    if m5_b <= m5_a + 12.0:
        m5_a = corner_cut + 10.0
        m5_b = max(m5_a + 14.0, flange - 8.0)
    tab_m5_dx = 0.5 * (0.5 * rail_notch_w + 0.5 * tab_w)
    tab_m5_y = d - min(10.0, 0.5 * max(recess, 12.0))
    m5_coords = [
        # 4x FL
        (t_cl, m5_a),
        (t_cl, m5_b),
        (m5_a, t_cl),
        (m5_b, t_cl),
        # 4x FR
        (w - t_cl, m5_a),
        (w - t_cl, m5_b),
        (w - m5_a, t_cl),
        (w - m5_b, t_cl),
        # 2x Rear-Center tab
        (0.5 * w - tab_m5_dx, tab_m5_y),
        (0.5 * w + tab_m5_dx, tab_m5_y),
    ]
    for hx, hy in m5_coords:
        holes.append((hx, hy, 5.4))

    # 8x 3.6 mm narrow DIN/accessory slots:
    # - 4x 50 mm vertical on left/right flanges
    # - 4x 40 mm horizontal on front/rear flanges
    slot_w = 3.6
    slot_v_len = min(50.0, max(20.0, 0.40 * cutout_d))
    slot_h_len = min(40.0, max(16.0, 0.30 * cutout_w))
    x_side_slot = 0.5 * (corner_cut + x0)
    y_side_1 = y0 + 0.25 * cutout_d
    y_side_2 = y0 + 0.75 * cutout_d
    y_front_slot = 0.5 * (corner_cut + y0)
    x_front_1 = x0 + 0.25 * cutout_w
    x_front_2 = x0 + 0.75 * cutout_w
    y_rear_slot = 0.5 * (y1 + y_wire_bot)
    x_rear_1 = 0.5 * (x0 + (0.5 * w - 20.0))
    x_rear_2 = w - x_rear_1

    hole_contours: list[list[_ContourPt]] = [
        _obround_slot_contour(x_side_slot, y_side_1, slot_v_len, slot_w, horizontal=False),
        _obround_slot_contour(x_side_slot, y_side_2, slot_v_len, slot_w, horizontal=False),
        _obround_slot_contour(w - x_side_slot, y_side_1, slot_v_len, slot_w, horizontal=False),
        _obround_slot_contour(w - x_side_slot, y_side_2, slot_v_len, slot_w, horizontal=False),
        _obround_slot_contour(x_front_1, y_front_slot, slot_h_len, slot_w, horizontal=True),
        _obround_slot_contour(x_front_2, y_front_slot, slot_h_len, slot_w, horizontal=True),
        _obround_slot_contour(x_rear_1, y_rear_slot, slot_h_len, slot_w, horizontal=True),
        _obround_slot_contour(x_rear_2, y_rear_slot, slot_h_len, slot_w, horizontal=True),
    ]

    return _Plate2D(
        part_id="trident_backbone",
        title="Trident Backbone Plate",
        width_mm=w,
        height_mm=d,
        thickness_mm=3.18,
        material_label='3.18mm (1/8") Mild Steel (Powder-Coated)',
        contours=contours,
        holes=holes,
        hole_contours=hole_contours,
        description=(
            f'Clee Trident Backbone 3.18mm (1/8") mild-steel open perimeter frame '
            f'({cutout_w:.0f}x{cutout_d:.0f}mm R10 cutout, {flange:.0f}mm flanges): '
            "3x Z-motor mounts (Ø26mm pilot + 31x31mm Ø3.5mm M3), 10x Ø5.4mm M5 frame holes, "
            "8x 3.6mm DIN slots, rear Z-motor tab + Z-rail/wiring notches"
        ),
    )


def _build_corner_gusset(spec: PrinterSpec, idx: int | None = None) -> _Plate2D:
    ext_w = _ext_width_mm(spec)
    arm_mm = max(40.0, 2.0 * ext_w)
    thickness_mm = max(4.0, round(0.2 * ext_w, 1))
    contour = [
        (0.0, 0.0),
        (arm_mm, 0.0),
        (arm_mm, ext_w),
        (ext_w, arm_mm),
        (0.0, arm_mm),
    ]
    holes = [
        (0.5 * ext_w, 0.5 * ext_w, 5.5),
        (arm_mm - 0.35 * ext_w, 0.5 * ext_w, 5.5),
        (0.5 * ext_w, arm_mm - 0.35 * ext_w, 5.5),
        (1.1 * ext_w, 1.1 * ext_w, 5.5),
    ]
    title = f"Corner Gusset #{idx}" if idx is not None else "Corner Gusset Plate"
    return _Plate2D(
        part_id="corner_gusset",
        title=title,
        width_mm=arm_mm,
        height_mm=arm_mm,
        thickness_mm=thickness_mm,
        material_label="Aluminum 6061-T6",
        contours=[contour],
        holes=holes,
        description=f"Triangular {arm_mm:.0f}x{arm_mm:.0f}mm L-gusset plate with 4x M5 holes",
    )


def _select_kit_plates(spec: PrinterSpec) -> list[_Plate2D]:
    """Select all applicable CNC/laser plates for a complete printer kit."""
    plates: list[_Plate2D] = []
    if spec.frame.enclosed_panels and spec.frame.panel_mounting != "none":
        plates.append(_build_side_panel(spec, title="Left Side Panel"))
        plates.append(_build_side_panel(spec, title="Right Side Panel"))
        plates.append(_build_rear_panel(spec))
        if spec.frame.top_panel:
            plates.append(_build_top_panel(spec))
    if spec.frame.trident_backbone and not spec.is_flying_gantry:
        plates.append(_build_trident_backbone(spec))
    if spec.frame.bracing == "corner_gussets" or spec.frame.corner_joint_type in (
        "gusset_plate",
        "l_bracket",
    ):
        for idx in range(1, 5):
            plates.append(_build_corner_gusset(spec, idx=idx))
    if not plates:
        for idx in range(1, 5):
            plates.append(_build_corner_gusset(spec, idx=idx))
    return plates


def _build_plates_for_part(spec: PrinterSpec, part: str) -> list[_Plate2D]:
    key = str(part).strip().lower()
    if key not in VALID_DXF_PARTS:
        valid = ", ".join(VALID_DXF_PARTS)
        raise ValueError(f"Unknown DXF part '{part}'. Available: {valid}")
    if key == "side_panel":
        return [_build_side_panel(spec)]
    if key == "rear_panel":
        return [_build_rear_panel(spec)]
    if key == "top_panel":
        return [_build_top_panel(spec)]
    if key == "trident_backbone":
        return [_build_trident_backbone(spec)]
    if key == "corner_gusset":
        return [_build_corner_gusset(spec)]
    return _select_kit_plates(spec)


def list_dxf_exportable_parts(spec: PrinterSpec) -> list[dict[str, Any]]:
    """Return metadata describing all CNC/laser DXF parts available for ``spec``."""
    panels_active = bool(spec.frame.enclosed_panels and spec.frame.panel_mounting != "none")
    bb_active = bool(spec.frame.trident_backbone and not spec.is_flying_gantry)
    gussets_active = bool(
        spec.frame.bracing == "corner_gussets"
        or spec.frame.corner_joint_type in ("gusset_plate", "l_bracket")
    )

    side = _build_side_panel(spec)
    rear = _build_rear_panel(spec)
    top = _build_top_panel(spec)
    backbone = _build_trident_backbone(spec)
    gusset = _build_corner_gusset(spec)
    kit = _select_kit_plates(spec)

    items = [
        ("side_panel", side, 2, panels_active),
        ("rear_panel", rear, 1, panels_active),
        ("top_panel", top, 1, panels_active and bool(spec.frame.top_panel)),
        ("trident_backbone", backbone, 1, bb_active),
        ("corner_gusset", gusset, 4, gussets_active),
    ]
    catalog: list[dict[str, Any]] = []
    for part_id, plate, qty, applicable in items:
        catalog.append(
            {
                "part": part_id,
                "label": plate.title,
                "width_mm": round(plate.width_mm, 1),
                "height_mm": round(plate.height_mm, 1),
                "thickness_mm": round(plate.thickness_mm, 2),
                "material": plate.material_label,
                "quantity": qty,
                "hole_count": len(plate.holes) + len(plate.hole_contours),
                "applicable": applicable,
                "description": plate.description,
            }
        )

    total_w = sum(p.width_mm for p in kit) + DXF_SPACING_MM * max(len(kit) - 1, 0)
    max_h = max((p.height_mm for p in kit), default=0.0)
    catalog.append(
        {
            "part": "all_plates",
            "label": "All Applicable Plates (Nested Kit)",
            "width_mm": round(total_w, 1),
            "height_mm": round(max_h, 1),
            "thickness_mm": round(float(spec.frame.panel_thickness_mm), 2),
            "material": "Mixed Kit",
            "quantity": len(kit),
            "hole_count": sum(len(p.holes) + len(p.hole_contours) for p in kit),
            "applicable": True,
            "description": f"2D nested kit layout ({len(kit)} plates, {DXF_SPACING_MM:.0f}mm spacing)",
        }
    )
    return catalog


def _emit_lwpolyline(
    msp: Any,
    ox: float,
    oy: float,
    pts: list[_ContourPt],
    layer: str,
) -> None:
    has_bulge = any(len(p) > 2 and abs(float(p[2])) > 1e-9 for p in pts)
    if has_bulge:
        xyb = [
            (ox + float(p[0]), oy + float(p[1]), float(p[2]) if len(p) > 2 else 0.0)
            for p in pts
        ]
        msp.add_lwpolyline(xyb, format="xyb", close=True, dxfattribs={"layer": layer})
    else:
        xy = [(ox + float(p[0]), oy + float(p[1])) for p in pts]
        msp.add_lwpolyline(xy, close=True, dxfattribs={"layer": layer})


def export_panels_dxf_bytes(spec: PrinterSpec, part: str = "all_plates") -> bytes:
    """Generate a 2D CNC/Laser DXF R2010 (mm) drawing for a plate or nested kit."""
    plates = _build_plates_for_part(spec, part)

    doc = ezdxf.new("R2010")
    doc.units = ezdxf.units.MM
    doc.layers.add("CUT_OUTER", color=7)
    doc.layers.add("HOLES", color=1)
    doc.layers.add("ANNOTATIONS", color=3)
    msp = doc.modelspace()

    x_cursor = 0.0
    for plate in plates:
        ox = x_cursor
        oy = 0.0

        for contour in plate.contours:
            _emit_lwpolyline(msp, ox, oy, contour, layer="CUT_OUTER")

        for slot_contour in plate.hole_contours:
            _emit_lwpolyline(msp, ox, oy, slot_contour, layer="HOLES")

        for cx, cy, diam in plate.holes:
            msp.add_circle(
                center=(ox + cx, oy + cy),
                radius=0.5 * diam,
                dxfattribs={"layer": "HOLES"},
            )

        text_h = max(2.2, min(8.0, plate.width_mm * 0.022))
        t_str = (
            f"{plate.thickness_mm:.2f}"
            if abs(plate.thickness_mm - round(plate.thickness_mm, 1)) > 1e-4
            else f"{plate.thickness_mm:.1f}"
        )
        slot_suffix = f", {len(plate.hole_contours)} slots" if plate.hole_contours else ""
        line1 = f"{plate.title} ({plate.width_mm:.1f} x {plate.height_mm:.1f} mm)"
        line2 = (
            f"{plate.material_label} | t={t_str} mm | "
            f"{len(plate.holes)} holes{slot_suffix}"
        )
        if plate.width_mm < 100.0:
            tx = ox
            ty1 = oy + plate.height_mm + text_h * 2.2
            ty2 = oy + plate.height_mm + text_h * 0.6
        else:
            tx = ox + 0.06 * plate.width_mm
            ty1 = oy + 0.54 * plate.height_mm
            ty2 = ty1 - text_h * 1.6

        t1 = msp.add_text(line1, height=text_h, dxfattribs={"layer": "ANNOTATIONS"})
        t1.set_placement((tx, ty1))
        t2 = msp.add_text(line2, height=text_h * 0.85, dxfattribs={"layer": "ANNOTATIONS"})
        t2.set_placement((tx, ty2))

        x_cursor += plate.width_mm + DXF_SPACING_MM

    stream = io.StringIO()
    doc.write(stream)
    return stream.getvalue().encode("utf-8")


def export_panels_dxf(
    spec: PrinterSpec,
    out_path: str | Path,
    part: str = "all_plates",
) -> Path:
    """Write a 2D CNC/Laser DXF R2010 (mm) file to ``out_path`` and return its Path."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(export_panels_dxf_bytes(spec, part=part))
    return out
