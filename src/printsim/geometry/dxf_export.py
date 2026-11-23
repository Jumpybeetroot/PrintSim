"""2D CNC / Laser-cut DXF panel and structural plate exporter powered by ``ezdxf``."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ezdxf

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


@dataclass
class _Plate2D:
    """Internal 2D CNC/laser plate representation in millimeters."""

    part_id: str
    title: str
    width_mm: float
    height_mm: float
    thickness_mm: float
    material_label: str
    contours: list[list[tuple[float, float]]]
    holes: list[tuple[float, float, float]]  # (cx_mm, cy_mm, diameter_mm)
    description: str


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


def _build_trident_backbone(spec: PrinterSpec) -> _Plate2D:
    ext_w = _ext_width_mm(spec)
    w = float(spec.frame.width_mm)
    d = float(spec.frame.depth_mm)
    outer = [(0.0, 0.0), (w, 0.0), (w, d), (0.0, d)]
    contours: list[list[tuple[float, float]]] = [outer]

    # 3x NEMA17 Z-motor mounts: front-left, front-right, rear-center
    off = min(0.22 * w, 0.22 * d, ext_w + 24.0)
    motor_centers = [
        (off, off),
        (w - off, off),
        (0.5 * w, d - off),
    ]
    holes: list[tuple[float, float, float]] = []
    for mx, my in motor_centers:
        # NEMA17 Ø22.5 mm pilot bore + TR8 leadscrew clearance
        holes.append((mx, my, 22.5))
        # 4x NEMA17 M3 holes on 31 x 31 mm square
        for dx in (-15.5, 15.5):
            for dy in (-15.5, 15.5):
                holes.append((mx + dx, my + dy, 3.4))

    # Perimeter M5 frame mounting holes along bottom extrusion centerlines
    holes.extend(_perimeter_holes(w, d, inset_mm=0.5 * ext_w, diameter_mm=5.5, target_pitch_mm=90.0))

    # Large central chamfered weight-relief cutout clearing the 3 motor pads
    margin = off + 30.0
    x0, x1 = margin, w - margin
    y0, y1 = margin, d - margin
    if (x1 - x0) >= 30.0 and (y1 - y0) >= 30.0:
        ch = min(15.0, 0.15 * min(x1 - x0, y1 - y0))
        contours.append(
            [
                (x0 + ch, y0),
                (x1 - ch, y0),
                (x1, y0 + ch),
                (x1, y1 - ch),
                (x1 - ch, y1),
                (x0 + ch, y1),
                (x0, y1 - ch),
                (x0, y0 + ch),
            ]
        )

    return _Plate2D(
        part_id="trident_backbone",
        title="Trident Backbone Plate",
        width_mm=w,
        height_mm=d,
        thickness_mm=3.0,
        material_label="3mm Mild Steel (Powder-Coated)",
        contours=contours,
        holes=holes,
        description=(
            "Clee-style 3mm steel bottom stiffener plate with 3x NEMA17 Z-motor patterns "
            "(31x31mm M3 + Ø22.5mm pilot bores) and central weight-relief cutout"
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
                "thickness_mm": round(plate.thickness_mm, 1),
                "material": plate.material_label,
                "quantity": qty,
                "hole_count": len(plate.holes),
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
            "thickness_mm": round(float(spec.frame.panel_thickness_mm), 1),
            "material": "Mixed Kit",
            "quantity": len(kit),
            "hole_count": sum(len(p.holes) for p in kit),
            "applicable": True,
            "description": f"2D nested kit layout ({len(kit)} plates, {DXF_SPACING_MM:.0f}mm spacing)",
        }
    )
    return catalog


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
            pts = [(ox + x, oy + y) for x, y in contour]
            msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "CUT_OUTER"})

        for cx, cy, diam in plate.holes:
            msp.add_circle(
                center=(ox + cx, oy + cy),
                radius=0.5 * diam,
                dxfattribs={"layer": "HOLES"},
            )

        text_h = max(2.2, min(8.0, plate.width_mm * 0.022))
        line1 = f"{plate.title} ({plate.width_mm:.1f} x {plate.height_mm:.1f} mm)"
        line2 = (
            f"{plate.material_label} | t={plate.thickness_mm:.1f} mm | "
            f"{len(plate.holes)} holes"
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
