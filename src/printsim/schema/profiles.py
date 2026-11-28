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
    shear_kappa_y: float | None = Field(
        None, gt=0, description="Optional Tet10-calibrated Timoshenko shear factor kappa_y (local Y)"
    )
    shear_kappa_z: float | None = Field(
        None, gt=0, description="Optional Tet10-calibrated Timoshenko shear factor kappa_z (local Z)"
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
    "extrusion_1515": ProfileSpec(
        id="extrusion_1515",
        name="1515 T-Slot Extrusion (15x15mm, 1.1mm Wall)",
        shape="extrusion",
        width_m=0.015,
        height_m=0.015,
        area_m2=8.50e-5,  # 85 mm^2 = 0.230 kg/m (MakerBeam / LDO Voron V0 & Monolith Zero 1515)
        iy_m4=0.198e-8,  # 0.198 cm^4
        iz_m4=0.198e-8,
        j_m4=0.032e-8,  # 0.032 cm^4 open T-slot St. Venant torsion constant
    ),
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
    # 3D Tet4 continuum FEA-calibrated CNC-milled lightweight aluminum X-beams (exact CAD volume + pocket shear-lag):
    # PrintSim parametric build123d 20x20x380mm semi-monocoque Warren-truss (solid 2mm MGN12H front flange + 3-face pockets,
    # 90.46 g @ 380 mm in Al 6061-T6 = 238.0 g/m; 141,182-tet cantilever bending/torsion FEA).
    "cnc_al_monocoque_20x20": ProfileSpec(
        id="cnc_al_monocoque_20x20",
        name="PrintSim CNC Monocoque Truss 20x20 Al X-Beam (238 g/m)",
        shape="tube",
        width_m=0.020,
        height_m=0.020,
        area_m2=0.8816e-4,
        iy_m4=0.344e-8,
        iz_m4=0.409e-8,
        j_m4=0.188e-8,
        wall_thickness_m=0.002,
    ),
    # GrabCAD Emil Wallberg Voron 2.4 Lightweight X-Axis (Voron_XAxis_LW_300.STEP, 19.6x19.6x380mm,
    # 4-face skeletonized truss; 64.75 g @ 380 mm in Al 6061-T6 = 170.4 g/m; 95,649-tet FEA).
    "cnc_al_truss_voron24_lw": ProfileSpec(
        id="cnc_al_truss_voron24_lw",
        name="Voron 2.4 LW Milled Truss 20x20 Al X-Beam (170 g/m)",
        shape="tube",
        width_m=0.0196,
        height_m=0.0196,
        area_m2=0.6311e-4,
        iy_m4=0.179e-8,
        iz_m4=0.187e-8,
        j_m4=0.150e-8,
    ),
    # GrabCAD Nicotina52 FYSETC Hollow CNC X-Beam Voron 0 (15.0x15.13x200mm Warren-truss hollow beam;
    # 17.43 g @ 200 mm in Al 6061-T6 = 87.1 g/m; 93,440-tet FEA).
    "cnc_al_fysetc_v0_15x15": ProfileSpec(
        id="cnc_al_fysetc_v0_15x15",
        name="FYSETC Hollow CNC 15x15 Al X-Beam (87 g/m, Voron 0)",
        shape="tube",
        width_m=0.0150,
        height_m=0.01513,
        area_m2=0.3228e-4,
        iy_m4=0.072e-8,
        iz_m4=0.073e-8,
        j_m4=0.037e-8,
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
    # Theoretical thick-walled Carbonara (PrintSim wall-sweep optimum: 2.5 mm wall, R2.5 mm corners,
    # 15.0 mm inner bore clearing two-wall clamped PET-CF inserts + M5 standoffs; +26% I, +23% J, ~0 Hz Y-shaper penalty)
    "cf_tube_20x20_carbonara_2_5": _hollow_square_tube(
        "cf_tube_20x20_carbonara_2_5",
        "Carbonara 20x20x2.5mm CF X-Beam (Theoretical, 264 g/m)",
        20.0,
        2.5,
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
    "ti_tube_20x20x2": _hollow_square_tube(
        "ti_tube_20x20x2", "Titanium Square Tube 20x20x2.0mm", 20.0, 2.0
    ),
    "al_tube_12x12x1": _hollow_square_tube(
        "al_tube_12x12x1", "Aluminum Square Tube 12x12x1.0mm (Monolith Zero)", 12.0, 1.0
    ),
    "al_tube_20x20x2": _hollow_square_tube(
        "al_tube_20x20x2", "Aluminum Square Tube 20x20x2.0mm", 20.0, 2.0
    ),
    "al_tube_20x20x3": _hollow_square_tube(
        "al_tube_20x20x3", "Aluminum Square Tube 20x20x3.0mm", 20.0, 3.0
    ),
    "al_tube_30x30x3": _hollow_square_tube(
        "al_tube_30x30x3", "Aluminum Square Tube 30x30x3.0mm", 30.0, 3.0
    ),
    "steel_tube_25x25x2": _hollow_square_tube(
        "steel_tube_25x25x2", "Steel Square Tube 25x25x2.0mm", 25.0, 2.0
    ),
    # HIWIN MG / HG Series linear guideways — 3D Tet10 (P2) continuum FEA calibrated cross-sections
    # (scripts/calibrate_all_rails_and_carriages_fea.py): exact Gothic-arch (MGN) / waisted 4-row
    # circular-arc (HGR15) raceway grooves + M3/M4 counterbored holes on catalog pitch (20/25/40/60 mm),
    # with exact HIWIN catalog linear mass @ 7850 kg/m^3 steel_4140 (0.38 / 0.65 / 1.06 / 1.45 kg/m).
    "mgn9_rail": _solid_rect("mgn9_rail", "MGN9H Linear Rail (9x6.5mm)", 9.0, 6.5, "rail").model_copy(
        update={
            "area_m2": 0.380 / 7850.0,
            "iy_m4": 1.480e-10,
            "iz_m4": 3.365e-10,
            "j_m4": 2.872e-10,
            "shear_kappa_y": 0.508,
            "shear_kappa_z": 0.842,
        }
    ),
    # 3D Tet10 (P2) FEA-calibrated periodic bottom-pocketed MGN9H rail (12x4.6x4.3 mm pockets between
    # 8 mm solid M3 bolt bridges, 1.35 mm min raceway wall; 299.0 g/m, -22.9% mass vs solid MGN9R).
    "mgn9_pocketed_rail": _solid_rect(
        "mgn9_pocketed_rail",
        "Pocketed MGN9H Linear Rail (9x6.5mm, 299 g/m)",
        9.0,
        6.5,
        "rail",
    ).model_copy(
        update={
            "area_m2": 38.09e-6,
            "iy_m4": 1.149e-10,
            "iz_m4": 3.105e-10,
            "j_m4": 1.239e-10,
            "shear_kappa_y": 0.329,
            "shear_kappa_z": 0.85,
        }
    ),
    "mgn12_rail": _solid_rect("mgn12_rail", "MGN12H Linear Rail (12x8mm)", 12.0, 8.0, "rail").model_copy(
        update={
            "area_m2": 0.650 / 7850.0,
            "iy_m4": 4.119e-10,
            "iz_m4": 9.623e-10,
            "j_m4": 8.562e-10,
            "shear_kappa_y": 0.646,
            "shear_kappa_z": 0.812,
        }
    ),
    # 3D Tet10 (P2) FEA-calibrated Monolith/ULS continuous bottom-channel hollow MGN12H rail
    # (6.6x4.98 mm open C-channel, 1.45 mm min raceway wall, 440.9 g/m, -33.1% mass vs solid MGN12R;
    # captures true open-section C-channel torsion loss J = 130.1 mm^4 and shear kappa_y = 0.179).
    "mgn12_hollow_rail": _solid_rect(
        "mgn12_hollow_rail", "Hollow MGN12H Linear Rail (12x8mm, 441 g/m)", 12.0, 8.0, "rail"
    ).model_copy(
        update={
            "area_m2": 56.16e-6,
            "iy_m4": 2.975e-10,
            "iz_m4": 8.882e-10,
            "j_m4": 1.301e-10,
            "shear_kappa_y": 0.179,
            "shear_kappa_z": 0.769,
        }
    ),
    # 3D Tet10 (P2) FEA-calibrated periodic bottom-pocketed MGN12H rail (16x6.6x5.5 mm pockets between
    # 9 mm solid M3 bolt bridges, 1.45 mm min raceway wall; 484.6 g/m, -26.5% mass, 2.26x bare GJ vs hollow).
    "mgn12_pocketed_rail": _solid_rect(
        "mgn12_pocketed_rail",
        "Pocketed MGN12H Linear Rail (12x8mm, 485 g/m)",
        12.0,
        8.0,
        "rail",
    ).model_copy(
        update={
            "area_m2": 61.74e-6,
            "iy_m4": 3.090e-10,
            "iz_m4": 8.964e-10,
            "j_m4": 2.942e-10,
            "shear_kappa_y": 0.329,
            "shear_kappa_z": 0.690,
        }
    ),
    # Sentinel for rail-only unsupported X-gantries (e.g. Monolith Zero bare MGN12H X-rail;
    # builder_gantry constructs the X member directly from x_rail_profile + optional rail backer).
    "unsupported_rail": _solid_rect(
        "unsupported_rail", "Unsupported Bare Linear Rail (No X-Beam)", 12.0, 8.0, "rail"
    ).model_copy(
        update={
            "area_m2": 0.650 / 7850.0,
            "iy_m4": 4.119e-10,
            "iz_m4": 9.623e-10,
            "j_m4": 8.562e-10,
            "shear_kappa_y": 0.646,
            "shear_kappa_z": 0.812,
        }
    ),
    "mgn15_rail": _solid_rect("mgn15_rail", "MGN15H Linear Rail (15x10mm)", 15.0, 10.0, "rail").model_copy(
        update={
            "area_m2": 1.060 / 7850.0,
            "iy_m4": 11.290e-10,
            "iz_m4": 24.134e-10,
            "j_m4": 22.329e-10,
            "shear_kappa_y": 0.777,
            "shear_kappa_z": 0.796,
        }
    ),
    # 3D Tet10 (P2) FEA-calibrated periodic bottom-pocketed MGN15H rail (30x8.4x7.0 mm pockets between
    # 10 mm solid M3 bolt bridges, 1.65 mm min raceway wall; 730.6 g/m, -31.5% mass vs solid MGN15R).
    "mgn15_pocketed_rail": _solid_rect(
        "mgn15_pocketed_rail",
        "Pocketed MGN15H Linear Rail (15x10mm, 731 g/m)",
        15.0,
        10.0,
        "rail",
    ).model_copy(
        update={
            "area_m2": 93.07e-6,
            "iy_m4": 7.756e-10,
            "iz_m4": 21.452e-10,
            "j_m4": 5.272e-10,
            "shear_kappa_y": 0.237,
            "shear_kappa_z": 0.557,
        }
    ),
    # 3D Tet10 (P2) FEA-calibrated HIWIN HGR15R heavy 4-row waisted linear rail (15x15 mm,
    # P = 60 mm M4 counterbores, twin circular-arc ball tracks per side; 1.450 kg/m @ 7850 kg/m^3).
    "hgh15_rail": _solid_rect("hgh15_rail", "HGR15R Heavy Linear Rail (15x15mm)", 15.0, 15.0, "rail").model_copy(
        update={
            "area_m2": 1.450 / 7850.0,
            "iy_m4": 34.509e-10,
            "iz_m4": 27.743e-10,
            "j_m4": 38.583e-10,
            "shear_kappa_y": 0.782,
            "shear_kappa_z": 0.769,
        }
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

from printsim.schema.printdb_bridge import hydrate_profile_catalog as _hydrate_profile_catalog

_hydrate_profile_catalog(PROFILE_CATALOG)


def get_profile(profile_id: str) -> ProfileSpec:
    """Look up a structural profile by ID."""
    if profile_id not in PROFILE_CATALOG:
        valid = ", ".join(sorted(PROFILE_CATALOG))
        raise KeyError(f"Unknown profile '{profile_id}'. Available: {valid}")
    return PROFILE_CATALOG[profile_id]
