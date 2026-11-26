"""Structural cross-section profile catalog with real area and moments of inertia."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field


class ProfileSpec(BaseModel):
    """Beam / extrusion / rail cross-section properties in SI units."""

    id: str
    name: str
    shape: Literal["extrusion", "tube", "solid_bar", "rail", "rod"] = "extrusion"
    width_m: float = Field(..., gt=0, description="Outer width along local y (m)")
    height_m: float = Field(..., gt=0, description="Outer height along local z (m)")
    area_m2: float = Field(..., gt=0, description="Cross-sectional area A (m^2)")
    iy_m4: float = Field(..., gt=0, description="Second moment of area about local y axis (m^4)")
    iz_m4: float = Field(..., gt=0, description="Second moment of area about local z axis (m^4)")
    j_m4: float = Field(..., gt=0, description="Torsional constant J (m^4)")
    wall_thickness_m: float | None = Field(
        None, gt=0, description="Nominal wall thickness for hollow closed-tube profiles (m)"
    )
    corner_radius_m: float = Field(
        0.0, ge=0, description="Outer corner fillet radius for roll-wrapped/formed hollow tubes (m)"
    )

    @property
    def max_c_y(self) -> float:
        return self.width_m * 0.5

    @property
    def max_c_z(self) -> float:
        return self.height_m * 0.5


def _solid_rounded_rect_i(w: float, h: float, r: float) -> float:
    """Exact second moment of area of a solid rounded rectangle about the axis parallel to ``w``."""
    if r <= 0.0:
        return w * (h**3) / 12.0
    i_center = (w - 2.0 * r) * (h**3) / 12.0
    i_sides = (2.0 * r) * ((h - 2.0 * r) ** 3) / 12.0
    a_q = 0.25 * math.pi * (r**2)
    y_bar = 4.0 * r / (3.0 * math.pi)
    i_q_c = (math.pi / 16.0 - 4.0 / (9.0 * math.pi)) * (r**4)
    y_q = (0.5 * h - r) + y_bar
    return i_center + i_sides + 4.0 * (i_q_c + a_q * (y_q**2))


def _hollow_rect_tube(
    id_: str,
    name: str,
    width_mm: float,
    height_mm: float,
    wall_mm: float,
    corner_radius_mm: float = 0.0,
) -> ProfileSpec:
    """Exact hollow rectangular/square closed tube with optional outer corner fillet radius.

    Computes exact area ``A``, bending inertias ``Iy`` (local y / global Z-bending) and ``Iz``
    (local z / global Y-bending), and Bredt-Batho closed-section torsion ``J = 4 * A_m^2 * t / s_m``
    using the rounded midline contour ``r_m = max(r_out - t/2, 0)``.
    """
    w = width_mm * 1e-3
    h = height_mm * 1e-3
    t = wall_mm * 1e-3
    ro = min(max(corner_radius_mm * 1e-3, 0.0), 0.5 * min(w, h))
    ri = max(ro - t, 0.0)
    wi = w - 2.0 * t
    hi = h - 2.0 * t

    area_out = w * h - (4.0 - math.pi) * (ro**2)
    area_in = wi * hi - (4.0 - math.pi) * (ri**2)
    area = area_out - area_in

    iy = _solid_rounded_rect_i(w, h, ro) - _solid_rounded_rect_i(wi, hi, ri)
    iz = _solid_rounded_rect_i(h, w, ro) - _solid_rounded_rect_i(hi, wi, ri)

    # Bredt-Batho closed thin-wall torsion with rounded midline contour
    wm = w - t
    hm = h - t
    rm = max(ro - 0.5 * t, 0.0)
    a_m = wm * hm - (4.0 - math.pi) * (rm**2)
    s_m = 2.0 * (wm + hm) - (8.0 - 2.0 * math.pi) * rm
    j_val = 4.0 * (a_m**2) * t / s_m

    return ProfileSpec(
        id=id_,
        name=name,
        shape="tube",
        width_m=w,
        height_m=h,
        area_m2=area,
        iy_m4=iy,
        iz_m4=iz,
        j_m4=j_val,
        wall_thickness_m=t,
        corner_radius_m=ro,
    )


def _hollow_square_tube(
    id_: str, name: str, outer_mm: float, wall_mm: float, corner_radius_mm: float = 0.0
) -> ProfileSpec:
    return _hollow_rect_tube(id_, name, outer_mm, outer_mm, wall_mm, corner_radius_mm=corner_radius_mm)


def _round_rod(id_: str, name: str, diameter_mm: float) -> ProfileSpec:
    r = (diameter_mm * 1e-3) * 0.5
    area = math.pi * r**2
    i_val = 0.25 * math.pi * r**4
    j_val = 0.5 * math.pi * r**4
    return ProfileSpec(
        id=id_,
        name=name,
        shape="rod",
        width_m=2.0 * r,
        height_m=2.0 * r,
        area_m2=area,
        iy_m4=i_val,
        iz_m4=i_val,
        j_m4=j_val,
    )


def _solid_rect(
    id_: str, name: str, width_mm: float, height_mm: float, shape: Literal["solid_bar", "rail"]
) -> ProfileSpec:
    w = width_mm * 1e-3
    h = height_mm * 1e-3
    area = w * h
    iy = w * (h**3) / 12.0
    iz = h * (w**3) / 12.0
    a, b = max(w, h), min(w, h)
    # Roark's torsional constant for rectangular bar
    j_val = a * (b**3) * (1.0 / 3.0 - 0.21 * (b / a) * (1.0 - (b**4) / (12.0 * a**4)))
    return ProfileSpec(
        id=id_,
        name=name,
        shape=shape,
        width_m=w,
        height_m=h,
        area_m2=area,
        iy_m4=iy,
        iz_m4=iz,
        j_m4=j_val,
    )


# Standard Misumi / Bosch Rexroth T-slot / V-slot extrusion properties (real datasheet values)
PROFILE_CATALOG: dict[str, ProfileSpec] = {
    "extrusion_2020": ProfileSpec(
        id="extrusion_2020",
        name="2020 T-Slot Extrusion (20x20mm)",
        shape="extrusion",
        width_m=0.020,
        height_m=0.020,
        area_m2=1.83e-4,  # 183 mm^2 = 0.494 kg/m (D-04: Misumi HFS5-2020 datasheet)
        iy_m4=0.742e-8,  # 0.742 cm^4 (Misumi HFS5-2020 datasheet)
        iz_m4=0.742e-8,
        j_m4=0.11e-8,  # 0.11 cm^4 open T-slot St. Venant torsion constant I_t (Bosch/Item datasheet)
    ),
    "extrusion_2040": ProfileSpec(
        id="extrusion_2040",
        name="2040 T-Slot Extrusion (20x40mm)",
        shape="extrusion",
        width_m=0.020,
        height_m=0.040,
        area_m2=2.98e-4,  # 298 mm^2 (Misumi HFS5-2040)
        iy_m4=4.55e-8,  # 4.55 cm^4 (strong axis)
        iz_m4=1.24e-8,  # 1.24 cm^4 (weak axis)
        j_m4=0.36e-8,  # 0.36 cm^4 open T-slot torsion constant I_t (Item/Misumi datasheet)
    ),
    "extrusion_3030": ProfileSpec(
        id="extrusion_3030",
        name="3030 T-Slot Extrusion (30x30mm)",
        shape="extrusion",
        width_m=0.030,
        height_m=0.030,
        area_m2=3.48e-4,  # 348 mm^2 (Misumi HFS6-3030)
        iy_m4=2.95e-8,  # 2.95 cm^4
        iz_m4=2.95e-8,
        j_m4=0.45e-8,  # 0.45 cm^4 open T-slot torsion constant I_t (Bosch/Item datasheet)
    ),
    "extrusion_4040": ProfileSpec(
        id="extrusion_4040",
        name="4040 T-Slot Extrusion (40x40mm)",
        shape="extrusion",
        width_m=0.040,
        height_m=0.040,
        area_m2=6.15e-4,  # 615 mm^2 (Misumi HFS8-4040)
        iy_m4=9.30e-8,  # 9.30 cm^4
        iz_m4=9.30e-8,
        j_m4=1.30e-8,  # 1.30 cm^4 open T-slot torsion constant I_t (Bosch Rexroth 40x40L datasheet)
    ),
    "extrusion_3060": ProfileSpec(
        id="extrusion_3060",
        name="3060 Heavy T-Slot Extrusion (30x60mm)",
        shape="extrusion",
        width_m=0.030,
        height_m=0.060,
        area_m2=6.80e-4,
        iy_m4=21.2e-8,
        iz_m4=5.80e-8,
        j_m4=2.10e-8,  # 2.10 cm^4 open T-slot torsion constant I_t (Bosch/Item 30x60 datasheet)
    ),
    "cnc_al_beam_20x20": ProfileSpec(
        id="cnc_al_beam_20x20",
        name="CNC Pocketed 7075/6061 Aluminum X-Beam (20x20mm)",
        shape="tube",
        width_m=0.020,
        height_m=0.020,
        area_m2=1.18e-4,  # 27% lighter than 2020 extrusion (1.62e-4)
        iy_m4=0.64e-8,
        iz_m4=0.64e-8,
        j_m4=0.68e-8,  # closed box torsional stiffness > open T-slot
    ),
    # Roll-wrapped / molded Carbon Fiber X-gantry square & rectangular tubes with realistic
    # mandrel corner radii (R1.5-R2.5 mm; fibers cannot wrap 0 mm sharp corners without voids).
    "cf_tube_20x20x1": _hollow_square_tube(
        "cf_tube_20x20x1",
        "Carbon Fiber Tube 20x20x1.0mm (Ultralight)",
        20.0,
        1.0,
        corner_radius_mm=1.5,
    ),
    "cf_tube_20x20x1_5": _hollow_square_tube(
        "cf_tube_20x20x1_5",
        "Carbon Fiber Tube 20x20x1.5mm",
        20.0,
        1.5,
        corner_radius_mm=2.0,
    ),
    # Scale-calibrated to Carbonara (Monolith / Brazuka / Fabreeko, 0.01 g scale):
    # 64.7 g @ 330 mm, 74.5 g @ 380 mm, 84.3 g @ 430 mm (196.1 g/m; R2.5 mm rounded corners, 1.8 mm eff. wall)
    "cf_tube_20x20_carbonara": _hollow_square_tube(
        "cf_tube_20x20_carbonara",
        "Carbonara 20x20mm CF X-Beam (196 g/m)",
        20.0,
        1.8,
        corner_radius_mm=2.5,
    ),
    "cf_tube_20x20x2": _hollow_square_tube(
        "cf_tube_20x20x2",
        "Carbon Fiber Tube 20x20x2.0mm",
        20.0,
        2.0,
        corner_radius_mm=2.5,
    ),
    # Wide-Y rectangular CF gantry beam: 30 mm front-to-back (local y -> global Y) x 20 mm tall (local z -> global Z)
    "cf_tube_30x20x2": _hollow_rect_tube(
        "cf_tube_30x20x2",
        "Carbon Fiber Wide-Y Tube 30x20x2.0mm",
        30.0,
        20.0,
        2.0,
        corner_radius_mm=2.5,
    ),
    "cf_tube_25x25x2": _hollow_square_tube(
        "cf_tube_25x25x2",
        "Carbon Fiber Tube 25x25x2.0mm",
        25.0,
        2.0,
        corner_radius_mm=2.5,
    ),
    "ti_tube_20x20x1_2": _hollow_square_tube(
        "ti_tube_20x20x1_2", "Titanium Square Tube 20x20x1.2mm", 20.0, 1.2
    ),
    "al_tube_20x20x2": _hollow_square_tube(
        "al_tube_20x20x2", "Aluminum Square Tube 20x20x2.0mm", 20.0, 2.0
    ),
    "al_tube_30x30x3": _hollow_square_tube(
        "al_tube_30x30x3", "Aluminum Square Tube 30x30x3.0mm", 30.0, 3.0
    ),
    "steel_tube_25x25x2": _hollow_square_tube(
        "steel_tube_25x25x2", "Steel Square Tube 25x25x2.0mm", 25.0, 2.0
    ),
    # HIWIN MG Series linear guideways (D-03: exact HIWIN catalog linear mass @ 7850 kg/m^3 steel_4140:
    # MGN9R = 0.38 kg/m, MGN12R = 0.65 kg/m, MGN15R = 1.06 kg/m; cross-section includes
    # bolt counterbores and ball-raceway grooves vs nominal outer W x H bounding box).
    "mgn9_rail": _solid_rect("mgn9_rail", "MGN9H Linear Rail (9x6.5mm)", 9.0, 6.5, "rail").model_copy(
        update={"area_m2": 0.380 / 7850.0}
    ),
    "mgn12_rail": _solid_rect("mgn12_rail", "MGN12H Linear Rail (12x8mm)", 12.0, 8.0, "rail").model_copy(
        update={"area_m2": 0.650 / 7850.0}
    ),
    "mgn15_rail": _solid_rect("mgn15_rail", "MGN15H Linear Rail (15x10mm)", 15.0, 10.0, "rail").model_copy(
        update={"area_m2": 1.060 / 7850.0}
    ),
    "steel_rod_8mm": _round_rod("steel_rod_8mm", "Hardened Linear Rod 8mm", 8.0),
    "steel_rod_10mm": _round_rod("steel_rod_10mm", "Hardened Linear Rod 10mm", 10.0),
    "steel_rod_12mm": _round_rod("steel_rod_12mm", "Hardened Linear Rod 12mm", 12.0),
    "brace_rod_10mm": _round_rod("brace_rod_10mm", "Diagonal Brace Rod / Strut 10mm", 10.0),
    # D-05: Roark rectangular torsion constant via _solid_rect (9.104e-10 m^4, not thin-strip b*t^3/3)
    "al_flat_bar_25x5": _solid_rect(
        "al_flat_bar_25x5", "25x5mm Aluminum Flat Bar (6061-T6)", 25.0, 5.0, "solid_bar"
    ),
}


def get_profile(profile_id: str) -> ProfileSpec:
    """Look up a structural profile by ID."""
    if profile_id not in PROFILE_CATALOG:
        valid = ", ".join(sorted(PROFILE_CATALOG))
        raise KeyError(f"Unknown profile '{profile_id}'. Available: {valid}")
    return PROFILE_CATALOG[profile_id]
