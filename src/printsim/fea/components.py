"""High-fidelity component mechanics and thermal-structural coupling models (strict SI).

References
----------
- HIWIN Corporation, *MG Series / HG Series Linear Guideway Technical Catalog* (ball dimensions,
  dynamic ratings C, preload classes ZF/Z0/Z1/ZA/ZB).
- Hung, J.-P. et al. (2011), "Load effect on the vibration characteristics of a stage with rolling
  guides," *Int. J. Mach. Tools Manuf.*, 51(10–11), carriage-skirt series compliance (~15%).
- Newmark, N. M., Siess, C. P., & Viest, I. M. (1951), "Tests and analysis of composite beams with
  incomplete interaction," *Proc. Soc. Exp. Stress Anal.*, 9(1), 75–92.
- Huth, H. (1986), "Influence of fastener flexibility on the prediction of load transfer and fatigue
  life for multiple-row joints," *ASTM STP 927*, 221–250.
- Gerbert, B. G. et al. (1972) / Childs, P. R. N. (2014), *Mechanical Design Engineering Handbook*,
  toothed-belt discrete spring mesh model.
- Harris, T. A. & Kotzalas, M. N. (2006), *Rolling Bearing Analysis*, Hertzian radial stiffness &
  ISO 5753 radial internal clearance classes (C2, CN, C3).
- Timoshenko, S. (1925), "Analysis of bi-metal thermostats," *J. Opt. Soc. Am.*, 11(3), 233–255.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from printsim.schema import get_material, get_profile
from printsim.schema.overrides import resolve_bed_mount, resolve_rail_quality
from printsim.schema.printdb_bridge import (
    get_reference_constant as _get_printdb_const,
    hydrate_rail_catalog as _hydrate_rail_catalog,
)
from printsim.schema.printer import (
    BEARING_RADIAL_PLAY_UM,
    CORNER_JOINT_BASE_2020,
    CORNER_JOINT_IWAN_DAMPING,
    RAIL_PRELOAD_FRACTIONS,
    RAIL_QUALITY_CATALOG,
    BeltSpec,
)

if TYPE_CHECKING:
    from printsim.geometry.builder import AssemblyModel
    from printsim.schema.printer import PrinterSpec

_G_ACC = _get_printdb_const(
    "references/fluid_cooling/fan-pq-operating-point", "mmh2o_to_pa", 9.80665
)

# HIWIN MGN / HGH catalog parameters: (ball_d_m, row_half_pitch_m, n_rows, L1_C_m, C_C_n, L1_H_m, C_H_n, seal_drag_n)
# row_half_pitch_m = W_R / 2 (actual ball-track lateral half-span matching HIWIN static roll moment M_R / C_0:
# MGN9: 4.5 mm, MGN12: 6.0 mm, MGN15: 7.5 mm, HGH15: 7.5 mm). Vertical stiffness k_zz already includes the
# 45 deg contact-angle projection, so multiplying by r_y**2 must NOT project by sin(45 deg) a second time.
_RAIL_CARRIAGE_CATALOG: dict[str, tuple[float, float, int, float, float, float, float, float]] = {
    "mgn9_rail": (1.588e-3, 0.0045, 2, 0.0189, 1860.0, 0.0289, 2550.0, 0.35),
    "mgn12_rail": (2.381e-3, 0.0060, 2, 0.0217, 2840.0, 0.0324, 3920.0, 0.50),
    "mgn15_rail": (3.175e-3, 0.0075, 2, 0.0267, 4610.0, 0.0434, 6370.0, 0.75),
    "hgh15_rail": (3.175e-3, 0.0075, 4, 0.0394, 11380.0, 0.0568, 14700.0, 1.20),
    # Round linear ball bushings (LM8UU/LM10UU/LM12UU & LUU long equivalent baseline)
    "steel_rod_8mm": (1.588e-3, 0.0040, 4, 0.0170, 265.0, 0.0240, 420.0, 0.40),
    "steel_rod_10mm": (1.984e-3, 0.0050, 4, 0.0200, 372.0, 0.0290, 588.0, 0.45),
    "steel_rod_12mm": (2.381e-3, 0.0060, 4, 0.0240, 412.0, 0.0360, 650.0, 0.50),
}

# Official HIWIN MG/HG & Misumi/THK LMxUU carriage block masses (g): (C_standard_g, H_long_g)
_RAIL_BLOCK_MASS_G: dict[str, tuple[float, float]] = {
    "mgn9_rail": (16.0, 26.0),
    "mgn12_rail": (34.0, 54.0),
    "mgn15_rail": (59.0, 92.0),
    "hgh15_rail": (180.0, 180.0),
    "steel_rod_8mm": (16.0, 31.0),
    "steel_rod_10mm": (31.0, 56.0),
    "steel_rod_12mm": (41.0, 76.0),
}
_hydrate_rail_catalog(_RAIL_CARRIAGE_CATALOG, _RAIL_BLOCK_MASS_G)

_PRELOAD_FRACTION: dict[str, float] = RAIL_PRELOAD_FRACTIONS

# Mindlin/Huth bolted M3 fastener interface shear stiffness k_s (N/m) calibrated to 3D Tet10 continuum FEA:
# - extrusion (HFS5-2020 T-nut + Mindlin preload): 26.0 MN/m (3D FEA: 27.1-29.7 MGN12, 22.9-23.2 MGN9)
# - carbon_tube (1-wall bare bolt bearing): 8.4 MN/m at 2.0 mm wall (3D FEA: 6.7/7.5/8.8 MN/m across 1.0/1.8/2.0 mm)
_BOLT_SHEAR_STIFFNESS_N_M: dict[str, float] = {
    "extrusion": 26.0e6,
    "carbon_tube": 8.4e6,
    "solid_bar": 42.0e6,
}

# Corner joint base stiffness for 2020 Al extrusion: (k_t_n_m, k_r_inplane_nm_rad, k_r_torsion_nm_rad)
_CORNER_JOINT_BASE_2020: dict[str, tuple[float, float, float]] = CORNER_JOINT_BASE_2020

_IWAN_DAMPING_DELTA: dict[str, float] = CORNER_JOINT_IWAN_DAMPING

_ISO_5753_RADIAL_PLAY_UM: dict[str, float] = BEARING_RADIAL_PLAY_UM


def hertzian_rail_carriage_stiffness(
    rail_profile_id: str,
    preload: str = "Z1",
    block_type: str = "H",
    rail_quality: str = "hiwin_genuine",
    preload_fraction_override: float | None = None,
    stiffness_mult_override: float | None = None,
    drag_mult_override: float | None = None,
) -> dict[str, float]:
    """Compute 6-DOF Hertzian ball-raceway contact stiffness & drag for a linear rail carriage.

    Includes 15% Hung et al. (2011) carriage-skirt structural compliance in series with the
    Hertzian elliptical point-contact stiffness (conformity f = 0.525, 45 deg contact angle),
    scaled by ``rail_quality`` raceway grinding & ball-sorting load-sharing multipliers.
    """
    key = rail_profile_id.lower().strip()
    if key in ("mgn9", "mgn12", "mgn15", "hgh15"):
        key = f"{key}_rail"
    if key not in _RAIL_CARRIAGE_CATALOG:
        key = "mgn12_rail"

    d_b, r_y, n_rows, l1_c, c_c, l1_h, c_h, seal_drag = _RAIL_CARRIAGE_CATALOG[key]
    mass_c_g, mass_h_g = _RAIL_BLOCK_MASS_G[key]
    is_h = block_type.upper() == "H"
    l1 = l1_h if is_h else l1_c
    c_dyn = c_h if is_h else c_c
    block_mass_g = float(mass_h_g if is_h else mass_c_g)

    pre_key = preload.upper()
    f_pre = (
        float(preload_fraction_override)
        if preload_fraction_override is not None
        else _PRELOAD_FRACTION.get(pre_key, 0.030)
    )
    f_pre_z1 = _PRELOAD_FRACTION["Z1"]

    qual_key = str(rail_quality).lower().strip() if rail_quality else "hiwin_genuine"
    qual_cfg = RAIL_QUALITY_CATALOG.get(qual_key, RAIL_QUALITY_CATALOG["hiwin_genuine"])
    stiffness_mult = (
        float(stiffness_mult_override)
        if stiffness_mult_override is not None
        else float(qual_cfg["stiffness_mult"])
    )
    drag_mult = (
        float(drag_mult_override)
        if drag_mult_override is not None
        else float(qual_cfg["drag_mult"])
    )

    # Active balls in the loaded raceway zone along L1 (~1.05 * d_b pitch)
    n_b = max(4, int(round(l1 / (1.05 * d_b))))
    sin_alpha = math.sin(math.radians(45.0))

    # Hardened bearing steel (E=207 GPa, nu=0.29)
    e_steel = 207.0e9
    nu_steel = 0.29
    e_star = e_steel / (2.0 * (1.0 - nu_steel**2))
    is_rod_bushing = key.startswith("steel_rod_")
    if is_rod_bushing:
        # B-12: Round linear bushing (LMxUU) runs balls on a convex cylindrical shaft + concave sleeve
        d_shaft = 2.0 * r_y
        r_e_in = 0.5 * d_b * math.sqrt(d_shaft / (d_b + d_shaft))
        r_e_out = 0.5 * d_b * math.sqrt((d_shaft + 2.0 * d_b) / (d_shaft + d_b))
        c_in = (9.0 / (16.0 * r_e_in * (e_star**2))) ** (1.0 / 3.0)
        c_out = (9.0 / (16.0 * r_e_out * (e_star**2))) ** (1.0 / 3.0)
        # Series compliance (c_in + c_out) represented via single-contact c_n = 0.5*(c_in+c_out)
        c_hertz = 0.5 * (c_in + c_out)
    else:
        # Profiled rail elliptical groove contact (f=0.525 -> R_e=d_b*sqrt(5.25), F2=0.68)
        r_e = d_b * math.sqrt(5.25)
        f2_ellip = 0.68
        c_hertz = ((9.0 / (16.0 * r_e * (e_star**2))) ** (1.0 / 3.0)) * f2_ellip

    def _raw_hertz_radial(f_frac: float) -> float:
        q_ball = max((f_frac * c_dyn) / (n_rows * n_b * sin_alpha), 0.15)
        k_n = (1.5 / c_hertz) * (q_ball ** (1.0 / 3.0))
        # B-12: Each ball has two contacts in series (k_n/2).
        # 2-row Gothic arch (mgn*, n_rows=2): 4-pt contact has 2 active diagonals -> 0.5 * n_rows = 1.0
        # 4-row circular-arc (hgh15, steel_rod_*, n_rows=4): 2-pt contact per row -> 0.25 * n_rows = 1.0
        row_factor = 0.25 * n_rows if n_rows >= 4 else 0.5 * n_rows
        return row_factor * n_b * k_n

    k_hertz_raw = _raw_hertz_radial(f_pre)
    # Slight clearance ZF unloads opposing ball row under initial load -> ~55% contact stiffness
    if preload_fraction_override is None and pre_key == "ZF":
        k_hertz_raw *= 0.55
    # Ball-tolerance load sharing and raceway grinding conformity scale Hertzian contact stiffness
    k_hertz_raw *= stiffness_mult

    # 15% Hung carriage-skirt structural compliance referenced to Z1 nominal block stiffness
    k_skirt = _raw_hertz_radial(f_pre_z1) / 0.15
    k_eff_radial = (k_hertz_raw * k_skirt) / max(k_hertz_raw + k_skirt, 1e-9)

    k_yy = k_eff_radial
    k_zz = k_eff_radial * (1.05 if (n_rows == 4 and not is_rod_bushing) else 1.0)

    # Roll stiffness about rail X axis (two side tracks at +-r_y with 1.25 C-block skirt opening factor;
    # k_zz already includes 45 deg contact-angle projection and Hung 15% carriage-skirt series compliance)
    k_rx = k_zz * (r_y**2) / 1.25
    # Pitch (Y) and yaw (Z) stiffness from ball distribution along raceway length L1
    k_ry = k_zz * ((l1**2) / 12.0)
    k_rz = k_yy * ((l1**2) / 12.0)

    q_total_preload = f_pre * c_dyn
    mu_roll = 0.0025
    rolling_friction = mu_roll * q_total_preload * drag_mult
    seal_drag_eff = seal_drag * drag_mult
    breakaway_friction = 1.6 * rolling_friction + 1.35 * seal_drag_eff

    return {
        "k_yy_n_m": k_yy,
        "k_zz_n_m": k_zz,
        "k_rx_nm_rad": k_rx,
        "k_ry_nm_rad": k_ry,
        "k_rz_nm_rad": k_rz,
        "rolling_friction_n": rolling_friction,
        "seal_drag_n": seal_drag_eff,
        "breakaway_friction_n": breakaway_friction,
        "block_mass_g": block_mass_g,
    }


def newmark_rail_composite_gamma(
    e_rail_pa: float,
    a_rail_m2: float,
    e_beam_pa: float,
    a_beam_m2: float,
    bolt_pitch_m: float,
    l_eff_m: float,
    substrate_kind: str = "extrusion",
    wall_thickness_m: float | None = None,
    matrix_shear_retention: float = 1.0,
    cf_rail_mounting: str = "single_wall_bare",
) -> float:
    """Newmark (1951) partial-interaction shear-lag composite factor gamma in (0, 1).

    Computes ``gamma = 1 / (1 + pi^2 * EA* * s / (L_eff^2 * k_s))`` using fastener interface
    shear stiffness ``k_s`` calibrated directly against 3D orthotropic Tet10 continuum FEA
    (``scripts/calibrate_xbeam_3d_fea.py``): ~26.0 MN/m for T-nut bolted aluminum extrusion,
    ~8.4 MN/m baseline (2.0 mm wall) for single-wall bare CF bolt bearing, ~27.3 MN/m (3.25x)
    for two-wall clamped PET-CF internal inserts with Shigley/Mindlin thermal preload retention,
    and ~42 MN/m for tapped solid metal bar.
    """
    ea_rail = max(float(e_rail_pa) * float(a_rail_m2), 1.0)
    ea_beam = max(float(e_beam_pa) * float(a_beam_m2), 1.0)
    ea_star = (ea_rail * ea_beam) / (ea_rail + ea_beam)

    s = max(float(bolt_pitch_m), 0.005)
    l_eff = max(float(l_eff_m), 0.05)

    sub_key = substrate_kind.lower().strip()
    if sub_key.startswith("cf") or "carbon" in sub_key:
        mount_key = str(cf_rail_mounting).lower().strip()
        if mount_key == "two_wall_clamped_inserts":
            wall_scale = (
                min(1.10, max(0.85, (float(wall_thickness_m) / 0.0018) ** 0.25))
                if wall_thickness_m is not None and wall_thickness_m > 0.0
                else 1.0
            )
            mount_scale = 3.25
            shear_ret_eff = 0.20 + 0.80 * max(0.25, float(matrix_shear_retention))
        else:
            wall_scale = (
                min(1.25, max(0.55, (float(wall_thickness_m) / 0.002) ** 0.72))
                if wall_thickness_m is not None and wall_thickness_m > 0.0
                else 1.0
            )
            mount_scale = 1.0
            shear_ret_eff = max(0.25, float(matrix_shear_retention))
        k_s = (
            _BOLT_SHEAR_STIFFNESS_N_M["carbon_tube"]
            * wall_scale
            * mount_scale
            * shear_ret_eff
        )
    elif "solid" in sub_key:
        k_s = _BOLT_SHEAR_STIFFNESS_N_M["solid_bar"]
    else:
        k_s = _BOLT_SHEAR_STIFFNESS_N_M["extrusion"]

    return 1.0 / (1.0 + (math.pi**2) * ea_star * s / ((l_eff**2) * k_s))


def dual_rail_steiner_efficiency(
    x_beam_profile_id: str,
    cf_rail_mounting: str = "single_wall_bare",
) -> float:
    """3D Tet10-calibrated Steiner parallel-axis roll coupling factor for dual_top_bottom rails.

    Calibrated with real steel MGN12H carriage blocks + 5 mm Al 6061-T6 wrap shuttle
    (``scripts/calibrate_xbeam_3d_fea.py``):
    - Extruded/solid metal beams (diagonal internal webs resist diamond distortion): ``0.45``
      (3D FEA measured 0.430 at 5 mm Al front bridge to 0.509 at MGN12H steel face).
    - Hollow CF square tube with 2-wall clamped PET-CF inserts every 25 mm: ``0.07``
      (opposing +-Y lateral rail loads induce tube diamond shear between discrete inserts;
      still ~3.0x single MGN12H at ~19.6 kNm/rad).
    - Hollow CF tube without internal inserts (single_wall_bare): ``0.03``.
    """
    beam_key = str(x_beam_profile_id).lower().strip()
    if beam_key.startswith("cf_") or "carbon" in beam_key:
        return 0.07 if str(cf_rail_mounting).lower().strip() == "two_wall_clamped_inserts" else 0.03
    return 0.45


def corner_joint_stiffness(
    profile_id: str,
    joint_type: str,
    k_t_override_n_m: float | None = None,
    k_r_inplane_override_nm_rad: float | None = None,
    k_r_torsion_override_nm_rad: float | None = None,
) -> dict[str, float]:
    """Compute semi-rigid frame corner joint translational & rotational stiffnesses (SI)."""
    jkey = joint_type.lower().strip()
    if jkey not in _CORNER_JOINT_BASE_2020:
        jkey = "blind_joint"

    k_t_base, k_r_base, k_tor_base = _CORNER_JOINT_BASE_2020[jkey]
    if jkey == "rigid":
        return {
            "k_t_n_m": float(k_t_override_n_m) if k_t_override_n_m is not None else k_t_base,
            "k_r_inplane_nm_rad": (
                float(k_r_inplane_override_nm_rad)
                if k_r_inplane_override_nm_rad is not None
                else k_r_base
            ),
            "k_r_torsion_nm_rad": (
                float(k_r_torsion_override_nm_rad)
                if k_r_torsion_override_nm_rad is not None
                else k_tor_base
            ),
        }

    prof = get_profile(profile_id)
    i_avg = 0.5 * (prof.iy_m4 + prof.iz_m4)
    scale_t = max(prof.width_m / 0.020, 0.4)
    scale_r = max((i_avg / 0.68e-8) ** 0.65, 0.25)

    return {
        "k_t_n_m": float(k_t_override_n_m) if k_t_override_n_m is not None else k_t_base * scale_t,
        "k_r_inplane_nm_rad": (
            float(k_r_inplane_override_nm_rad)
            if k_r_inplane_override_nm_rad is not None
            else k_r_base * scale_r
        ),
        "k_r_torsion_nm_rad": (
            float(k_r_torsion_override_nm_rad)
            if k_r_torsion_override_nm_rad is not None
            else k_tor_base * scale_r
        ),
    }


def iwan_joint_damping_delta(
    joint_type: str, damping_override: float | None = None
) -> float:
    """Small physical Iwan micro-slip damping increment delta_zeta by frame corner joint type."""
    if damping_override is not None:
        return float(damping_override)
    return _IWAN_DAMPING_DELTA.get(joint_type.lower().strip(), 0.003)


def belt_flexural_rigidity_nm2(
    belt: BeltSpec,
    chamber_temp_c: float = 25.0,
) -> float:
    """Effective composite belt bending stiffness EI_b (N·m²) for pulley wrap-fillet compliance.

    Calibrated to multi-filament twisted fiberglass/aramid/steel cords embedded in EPDM/TPU
    backing (e.g., 6 mm Gates 2GT fiberglass EI ≈ 9.0 N·mm² -> ~7 mm self-weight droop over
    50 mm cantilever; thicker 3GT / steel-cord TPU / AT5 scale with h_back^3 and wire EI).
    """
    w_m = max(float(belt.width_mm), 1.0) * 1e-3
    if belt.pitch_mm >= 4.5:
        base_ei_per_m = 0.036
    elif belt.pitch_mm >= 2.8:
        base_ei_per_m = 0.0145 if belt.core == "fiberglass" else 0.0195
    else:
        base_ei_per_m = 0.0015
        if belt.core == "aramid":
            base_ei_per_m *= 0.85
        elif belt.core == "steel":
            base_ei_per_m *= 2.40
    # Matrix elastomer contributes ~35% of bending stiffness and softens with chamber temperature
    ret = belt.tooth_modulus_retention(chamber_temp_c)
    return base_ei_per_m * w_m * (0.65 + 0.35 * ret)


def belt_damping_ratio(
    belt: BeltSpec,
    chamber_temp_c: float = 25.0,
    slack_margin_ratio: float = 1.0,
) -> float:
    """Effective belt-subsystem damping ratio zeta_belt vs core, elastomer, temp, and pretension.

    Captures base elastomer/cord hysteresis (EPDM fiberglass ~0.044, aramid ~0.054, TPU/steel
    ~0.036 rising as TPU approaches softening temp), low-tension wrap/tooth micro-slip hysteresis,
    and bilinear slack-unloading PSD broadening when dynamic load approaches pretension.
    """
    t0 = max(float(belt.pretension_n), 0.25)
    q0 = t0 / max(float(belt.width_mm), 1.0)
    # Calibrated to measured CoreXY ADXL damping (HVFDesigns X=0.034-0.041, Y=0.044-0.063)
    if belt.core == "aramid":
        z_base = 0.052
    elif belt.body_material == "tpu":
        z_base = 0.036
    else:
        z_base = 0.044

    # Thermal viscoelastic loss rise as matrix softens (notably TPU > 45 C)
    ret = belt.tooth_modulus_retention(chamber_temp_c)
    z_temp = 0.040 * max(0.0, 1.0 - ret)

    # Low-tension cord-bedding / wrap-unwrapping / tooth-seating hysteresis
    z_low_t = 0.036 * math.exp(-q0 / 1.7)

    # Nonlinear bilinear peak broadening when dynamic slack margin drops below 45%
    z_slack = 0.0
    if slack_margin_ratio < 0.45:
        u_slack = max(0.0, min(1.0, (0.45 - float(slack_margin_ratio)) / 0.85))
        z_slack = 0.050 * u_slack

    return float(min(0.18, z_base + z_temp + z_low_t + z_slack))


def belt_effective_ea_n(
    belt: BeltSpec,
    span_m: float,
    ea_override_n: float | None = None,
    linear_mass_override_kg_m: float | None = None,
    n_pulley_tangents: float = 4.0,
    chamber_temp_c: float = 25.0,
) -> float:
    """Effective belt tensile modulus EA (N) vs pretension, width, pulley wrap bending, & sag.

    Combines four physical compliance mechanisms in series:
      1. Width-normalized cord crimp / elastomer toe straightening (q0 = T0 / width_mm,
         Wang et al. 2018; Perneder & Osborne 2012) + preload-engaged dynamic stiffening.
      2. Euler-Bernoulli pulley/idler wrap flexural-fillet seating compliance around pulley
         radius R (Gerbert 1981 / Kong & Parker 2005): c_wrap = n_tan * 3/8 * EI^1.5 / (R^2 * T_reg^2.5),
         regularized smoothly by Euler wrap tension T_wrap = EI / R^2.
      3. Ernst (1965) gravity catenary sag axial compliance: EA_cat = 12 * T0^3 / ((mu * g * L)^2).
    """
    t0 = max(float(belt.pretension_n), 0.25)
    mu = max(
        float(
            linear_mass_override_kg_m
            if linear_mass_override_kg_m is not None
            else belt.linear_mass_kg_m
        ),
        1e-4,
    )
    span = max(float(span_m), 0.05)
    w_mm = max(float(belt.width_mm), 1.0)
    q0 = t0 / w_mm  # line tension (N per mm of belt width)

    # 1. Width-normalized cord crimp straightening + elastomer toe modulus (Wang et al. 2018; Perneder 2012)
    # Single unified preload-engagement curve keyed to line tension q0 = T0 / width_mm so belts are
    # compliant in the low-tension toe regime (q0 < 2 N/mm) while reaching >92% of rated dynamic EA
    # at normal working tensions (q0 ~ 3.5-5 N/mm).
    if belt.core == "steel":
        a_crimp, q_crimp = 0.30, 1.5
    elif belt.core == "aramid":
        a_crimp, q_crimp = 0.52, 2.2
    else:
        a_crimp, q_crimp = 0.52, 2.3
    crimp_factor = 1.0 - a_crimp * math.exp(-q0 / q_crimp)

    ea_raw = float(ea_override_n) if ea_override_n is not None else belt.stiffness_ea_n
    ea_cord = max(ea_raw * crimp_factor, 1.0)

    # 2. Pulley/idler wrap flexural-fillet seating compliance
    n_tan = max(float(n_pulley_tangents), 0.0)
    if n_tan > 0.0:
        ei_b = belt_flexural_rigidity_nm2(belt, chamber_temp_c=chamber_temp_c)
        r_p = max(float(belt.pitch_radius_m), 0.003)
        t_wrap = ei_b / (r_p**2)
        t_reg = t0 + 0.85 * t_wrap
        c_wrap = n_tan * 0.375 * (ei_b**1.5) / ((r_p**2) * (t_reg**2.5))
        inv_ea_wrap = c_wrap / span
    else:
        inv_ea_wrap = 0.0

    # 3. Ernst (1965) catenary sag axial stiffness: EA_cat = 12*T0^3 / ((mu*g*L)^2)
    ea_cat = 12.0 * (t0**3) / (((mu * _G_ACC) ** 2) * (span**2))
    inv_ea_cat = 1.0 / max(ea_cat, 1e-9)

    return 1.0 / max((1.0 / ea_cord) + inv_ea_wrap + inv_ea_cat, 1e-18)


def gerbert_tooth_mesh_stiffness_n_m(
    belt: BeltSpec,
    ea_eff_n: float,
    chamber_temp_c: float = 25.0,
    tooth_shear_override: float | None = None,
) -> float:
    """Gerbert (1972) discrete spring-chain tooth-mesh shear stiffness (N/m) with T(chamber) & radial seating."""
    w_mm = max(float(belt.width_mm), 1.0)
    q0 = max(float(belt.pretension_n), 0.25) / w_mm
    # Radial belt pressure T0/(w*R) seats belt tooth flanks into pulley grooves (Koyama 1979; Gerbert 1978)
    seating_factor = 0.22 + 0.78 * (1.0 - math.exp(-q0 / 2.2))
    k_t_base = (
        float(tooth_shear_override)
        if tooth_shear_override is not None
        else float(belt.tooth_shear_k_n_per_m_mm)
    )
    k_t = k_t_base * belt.tooth_modulus_retention(chamber_temp_c) * seating_factor
    p_m = belt.pitch_mm * 1e-3
    z_e = max(float(belt.pulley_teeth) * (float(belt.wrap_angle_deg) / 360.0), 1.0)
    ea_safe = max(float(ea_eff_n), 100.0)

    lam = math.acosh(1.0 + (k_t * w_mm * p_m) / (2.0 * ea_safe))
    return k_t * w_mm * math.tanh(lam * z_e) / max(math.tanh(lam), 1e-9)


def idler_shaft_and_bearing_stiffness(
    belt: BeltSpec,
    double_shear: bool,
    bracket_e_pa: float,
    radial_play_um_override: float | None = None,
) -> dict[str, float]:
    """Compute steel idler/motor shaft bending, polymer root flex, F695/625 bearing stiffness, & cam-out slip limit."""
    d_m = max(float(belt.idler_shaft_diameter_mm), 2.0) * 1e-3
    e_steel = 205.0e9
    i_shaft = math.pi * (d_m**4) / 64.0
    ei_shaft = e_steel * i_shaft

    w_m = float(belt.hardware_width_mm) * 1e-3
    e_bkt = max(float(bracket_e_pa), 1.0e8)

    if double_shear:
        l_span = 0.008 + 0.7 * w_m
        k_beam = 48.0 * ei_shaft / (l_span**3)
        k_root = 12.0 * e_bkt * d_m
    else:
        l_span = 0.009 + 0.5 * w_m
        k_beam = 3.0 * ei_shaft / (l_span**3)
        k_root = 1.2 * e_bkt * d_m

    k_shaft = (k_beam * k_root) / max(k_beam + k_root, 1e-9)

    wrap_rad = math.radians(float(belt.wrap_angle_deg))
    f_pulley = 2.0 * float(belt.pretension_n) * math.sin(0.5 * wrap_rad)
    shaft_deflection_um = (f_pulley / max(k_shaft, 1.0)) * 1e6

    # Paired F695 (5mm) / 626 (6mm) deep-groove ball bearing radial stiffness (Harris Hertzian)
    z_balls = 8.0
    d_ball_mm = 2.0 if belt.idler_shaft_diameter_mm <= 5.2 else 2.38
    f_brg = max(0.5 * f_pulley, 1.0)
    q_max = 5.0 * f_brg / z_balls
    delta_r_m = 0.00044e-3 * (q_max ** (2.0 / 3.0)) / (d_ball_mm ** (1.0 / 3.0))
    k_brg_single = 1.5 * f_brg / max(delta_r_m, 1e-9)
    k_bearing = 2.0 * k_brg_single

    k_idler = (k_shaft * k_bearing) / max(k_shaft + k_bearing, 1e-9)
    bearing_play_um = (
        float(radial_play_um_override)
        if radial_play_um_override is not None
        else _ISO_5753_RADIAL_PLAY_UM.get(belt.bearing_clearance_class, 7.5)
    )

    # Euler-Eytelwein belt wrap friction + curvilinear tooth cam-out minimum slack tension
    mu_eff = 0.35
    z_e = max(float(belt.pulley_teeth) * (float(belt.wrap_angle_deg) / 360.0), 1.5)
    min_slack_tension_n = max(
        1.8,
        (2.0 * float(belt.pretension_n))
        / (math.exp(mu_eff * wrap_rad) * (1.0 + 0.45 * min(z_e, 10.0)))
        + 6.0 / z_e,
    )

    return {
        "k_shaft_n_m": k_shaft,
        "k_bearing_n_m": k_bearing,
        "k_idler_n_m": k_idler,
        "idler_shaft_deflection_um": shaft_deflection_um,
        "bearing_play_um": bearing_play_um,
        "f_pulley_n": f_pulley,
        "min_slack_tension_n": min_slack_tension_n,
    }


@dataclass
class ThermalStructuralReport:
    """Coupled thermal-structural distortion, bimetallic bow, bed buckling, & Tg softening telemetry."""

    chamber_temp_c: float
    delta_t_k: float
    gantry_bimetallic_bow_um: float
    gantry_backer_reduction_pct: float
    bed_target_temp_c: float
    bed_mount_type: str
    bed_gradient_bow_um: float
    bed_buckling_taco_um: float
    frame_z_drift_um: float
    bracket_modulus_retention_pct: float
    bracket_tg_warning: bool
    total_thermal_z_error_um: float
    rail_newmark_gamma: float
    rail_carriage_roll_nm_per_rad: float
    rail_carriage_radial_n_per_um: float
    x_beam_modulus_retention_pct: float = 100.0
    x_beam_shear_retention_pct: float = 100.0
    x_beam_tg_warning: bool = False
    x_beam_has_tg: bool = False
    gantry_bow_direction: str = "neutral"

    def to_dict(self) -> dict:
        return {
            "chamber_temp_c": round(self.chamber_temp_c, 1),
            "delta_t_k": round(self.delta_t_k, 1),
            "gantry_bimetallic_bow_um": round(self.gantry_bimetallic_bow_um, 2),
            "gantry_backer_reduction_pct": round(self.gantry_backer_reduction_pct, 1),
            "bed_target_temp_c": round(self.bed_target_temp_c, 1),
            "bed_mount_type": self.bed_mount_type,
            "bed_gradient_bow_um": round(self.bed_gradient_bow_um, 2),
            "bed_buckling_taco_um": round(self.bed_buckling_taco_um, 2),
            "frame_z_drift_um": round(self.frame_z_drift_um, 2),
            "bracket_modulus_retention_pct": round(self.bracket_modulus_retention_pct, 1),
            "bracket_tg_warning": bool(self.bracket_tg_warning),
            "total_thermal_z_error_um": round(self.total_thermal_z_error_um, 2),
            "rail_newmark_gamma": round(self.rail_newmark_gamma, 4),
            "rail_carriage_roll_nm_per_rad": round(self.rail_carriage_roll_nm_per_rad, 1),
            "rail_carriage_radial_n_per_um": round(self.rail_carriage_radial_n_per_um, 1),
            "x_beam_modulus_retention_pct": round(self.x_beam_modulus_retention_pct, 1),
            "x_beam_shear_retention_pct": round(self.x_beam_shear_retention_pct, 1),
            "x_beam_tg_warning": bool(self.x_beam_tg_warning),
            "x_beam_has_tg": bool(self.x_beam_has_tg),
            "gantry_bow_direction": self.gantry_bow_direction,
        }


def compute_bed_thermal_bow_and_taco_um(spec: PrinterSpec) -> tuple[float, float]:
    """Compute heated bed vertical thermal gradient bow and mount-restraint taco buckling (um).

    Shared single source of truth used by both ``compute_thermal_structural_report`` and
    ``printsim.fea.bed_plate_fem`` so ``bed_buckling_taco_um == taco_buckling_um`` identically.
    """
    bed_temp = float(spec.bed.target_temp_c)
    chamber_c = float(spec.frame.chamber_temp_c)
    bed_mat = get_material(spec.bed.plate_material)
    frame_mat = get_material(spec.frame.material)
    l_bed = max(float(spec.bed.size_mm[0]), float(spec.bed.size_mm[1]), 50.0) * 1e-3
    t_bed = max(float(spec.bed.plate_thickness_mm) * 1e-3, 1e-4)

    # Steady upward heat loss q_top ~ h_eff * (T_bed - T_chamber) -> z_bow = L^2 * alpha * q_top / (8 * k)
    h_eff_bed = 12.0  # W/(m^2*K) horizontal upward convection + radiation
    q_top = h_eff_bed * max(0.0, bed_temp - chamber_c)
    bed_grad_bow_um = (
        (l_bed**2)
        * bed_mat.thermal_expansion_1_k
        * q_top
        / (8.0 * max(bed_mat.thermal_conductivity_w_mk, 0.1))
        * 1e6
    )

    # Bed mounting thermal expansion & post-buckling taco bow using true mount diagonal/span
    mount_cfg = resolve_bed_mount(spec)
    mount_inset_ratio = float(mount_cfg.get("mount_inset_ratio", 0.38))
    l_mount = 2.0 * mount_inset_ratio * l_bed
    dl_thermal = (
        bed_mat.thermal_expansion_1_k * max(0.0, bed_temp - 25.0)
        - frame_mat.thermal_expansion_1_k * max(0.0, chamber_c - 25.0)
    ) * l_mount
    dl_clearance = 0.00015  # 0.15 mm typical M4 mounting hole clearance
    dl_excess = max(0.0, dl_thermal - dl_clearance)

    k_restraint = float(mount_cfg["k_restraint_thermal_n_per_m"])
    if k_restraint <= 0.0 or dl_excess <= 0.0:
        bed_buckle_um = 0.0
    else:
        # B-13: Self-consistent Timoshenko beam-column + post-buckling elastica with Euler P_cr
        d_plate = (
            bed_mat.youngs_modulus_pa
            * (t_bed**3)
            / (12.0 * (1.0 - bed_mat.poisson_ratio**2))
        )
        p_cr = (math.pi**2) * d_plate * l_bed / max(l_mount**2, 1e-9)
        k_plate_axial = bed_mat.youngs_modulus_pa * l_bed * t_bed / max(l_mount, 1e-6)
        k_ser = (k_restraint * k_plate_axial) / max(k_restraint + k_plate_axial, 1e-9)
        # Eccentric standoff/imperfection amplitude scale: (pi^2 / 8) * (0.25 * t_bed)
        w_scale = (math.pi**2 / 8.0) * (0.25 * t_bed)
        w_lo = 0.0
        w_hi = (2.0 / math.pi) * math.sqrt(l_mount * dl_excess) + 1e-6
        geom_coeff = (math.pi**2) / (4.0 * max(l_mount, 1e-9))
        for _ in range(56):
            w_mid = 0.5 * (w_lo + w_hi)
            dl_rem = max(0.0, dl_excess - geom_coeff * (w_mid**2))
            r_load = (k_ser * dl_rem) / max(p_cr, 1e-9)
            g_val = (w_mid + w_scale) * (1.0 - r_load) - w_scale
            if g_val < 0.0:
                w_lo = w_mid
            else:
                w_hi = w_mid
        bed_buckle_um = 0.5 * (w_lo + w_hi) * 1e6

    return float(bed_grad_bow_um), float(bed_buckle_um)


def compute_thermal_structural_report(assembly: AssemblyModel) -> ThermalStructuralReport:
    """Compute Timoshenko bimetallic gantry bow, bed gradient/taco buckling, Z drift, & E(T) retention."""
    spec = assembly.spec
    chamber_c = float(spec.frame.chamber_temp_c)
    delta_t = chamber_c - 25.0

    # 1. X-rail Newmark shear-lag gamma & 6-DOF Hertzian carriage stiffness
    x_beam_prof = get_profile(spec.gantry.x_beam_profile)
    x_beam_mat = get_material(spec.gantry.x_beam_material)
    x_rail_prof = get_profile(spec.gantry.x_rail_profile)
    rail_mat = get_material(spec.gantry.rail_material)

    l_x = max(spec.frame.width_mm * 1e-3, 0.15)
    bolt_pitch_m = spec.gantry.rail_bolt_pitch_mm * 1e-3
    if x_beam_prof.id.startswith("cf_") or x_beam_mat.id.startswith("carbon_fiber"):
        sub_kind = "carbon_tube"
    elif x_beam_prof.shape == "solid_bar":
        sub_kind = "solid_bar"
    else:
        sub_kind = "extrusion"

    x_beam_mod_ret = x_beam_mat.modulus_retention_at_temp(chamber_c)
    x_beam_shear_ret = x_beam_mat.shear_modulus_retention_at_temp(chamber_c)
    eb_eff = x_beam_mat.effective_modulus_pa(chamber_c)
    x_beam_has_tg = x_beam_mat.tg_c is not None
    x_beam_tg_warn = bool(
        x_beam_has_tg and (chamber_c >= float(x_beam_mat.tg_c) - 15.0 or x_beam_shear_ret < 0.85)
    )

    cf_mount = spec.gantry.effective_cf_rail_mounting
    gamma_rail = newmark_rail_composite_gamma(
        e_rail_pa=rail_mat.youngs_modulus_pa,
        a_rail_m2=x_rail_prof.area_m2,
        e_beam_pa=eb_eff,
        a_beam_m2=x_beam_prof.area_m2,
        bolt_pitch_m=bolt_pitch_m,
        l_eff_m=l_x,
        substrate_kind=sub_kind,
        wall_thickness_m=x_beam_prof.wall_thickness_m,
        matrix_shear_retention=x_beam_shear_ret,
        cf_rail_mounting=cf_mount,
    )

    rail_qual = resolve_rail_quality(spec)
    carriage = hertzian_rail_carriage_stiffness(
        spec.gantry.x_rail_profile,
        preload=spec.gantry.rail_preload,
        block_type=spec.gantry.rail_block_type,
        rail_quality=spec.gantry.rail_quality,
        preload_fraction_override=spec.model_overrides.rail_preload.preload_fraction,
        stiffness_mult_override=rail_qual["stiffness_mult"],
        drag_mult_override=rail_qual["drag_mult"],
    )
    if spec.gantry.effective_dual_x_rails:
        z_track = 0.5 * (x_beam_prof.height_m + x_rail_prof.height_m)
        eta_steiner = dual_rail_steiner_efficiency(
            spec.gantry.x_beam_profile,
            spec.gantry.effective_cf_rail_mounting,
        )
        roll_nm_rad = 2.0 * carriage["k_rx_nm_rad"] + eta_steiner * 2.0 * carriage["k_yy_n_m"] * (z_track**2)
        radial_n_um = 2.0 * carriage["k_zz_n_m"] * 1e-6
    else:
        roll_nm_rad = carriage["k_rx_nm_rad"]
        radial_n_um = carriage["k_zz_n_m"] * 1e-6

    # 2. Timoshenko (1925) partial-interaction bimetallic bow on X gantry
    x_mount = spec.gantry.effective_x_rail_mount
    eb, ab, ib, alphab = (
        eb_eff,
        x_beam_prof.area_m2,
        x_beam_prof.iy_m4 if x_mount in ("top", "bottom") else x_beam_prof.iz_m4,
        x_beam_mat.thermal_expansion_1_k,
    )
    er, ar, ir, alphar = (
        rail_mat.youngs_modulus_pa,
        x_rail_prof.area_m2,
        x_rail_prof.iy_m4,
        rail_mat.thermal_expansion_1_k,
    )
    beam_dim = x_beam_prof.height_m if x_mount in ("top", "bottom") else x_beam_prof.width_m
    c_rail = 0.5 * beam_dim + 0.5 * x_rail_prof.height_m

    ei_sum = eb * ib + er * ir
    ea_inv = 1.0 / max(eb * ab, 1.0) + 1.0 / max(er * ar, 1.0)

    def _bimetallic_bow_signed(dt_val: float, use_backer: bool) -> tuple[float, float]:
        if x_mount == "dual_top_bottom" or spec.architecture == "delta":
            return 0.0, 0.0
        kappa_0 = (alphab - alphar) * dt_val * c_rail / (c_rail**2 + ei_sum * ea_inv)
        kappa_signed_0 = gamma_rail * kappa_0
        bow_0 = abs(kappa_signed_0) * (l_x**2) / 8.0 * 1e6
        if not use_backer or spec.gantry.rail_backer_material == "none":
            return bow_0, float(kappa_signed_0)

        bk_mat = get_material(spec.gantry.rail_backer_material)
        w_bk = min(beam_dim, max(x_rail_prof.width_m, 0.012))
        t_bk = spec.gantry.rail_backer_thickness_mm * 1e-3
        ak = w_bk * t_bk
        ik = w_bk * (t_bk**3) / 12.0
        ek = bk_mat.youngs_modulus_pa
        alphak = bk_mat.thermal_expansion_1_k
        ck = 0.5 * beam_dim + 0.5 * t_bk
        ei_tot = ei_sum + ek * ik
        gamma_k = newmark_rail_composite_gamma(
            e_rail_pa=ek,
            a_rail_m2=ak,
            e_beam_pa=eb,
            a_beam_m2=ab,
            bolt_pitch_m=bolt_pitch_m,
            l_eff_m=l_x,
            substrate_kind=sub_kind,
            wall_thickness_m=x_beam_prof.wall_thickness_m,
            matrix_shear_retention=x_beam_shear_ret,
            cf_rail_mounting=cf_mount,
        )
        m_mat = np.array(
            [
                [1.0 / (eb * ab) + 1.0 / (er * ar) + (c_rail**2) / ei_tot, 1.0 / (eb * ab) - (c_rail * ck) / ei_tot],
                [1.0 / (eb * ab) - (c_rail * ck) / ei_tot, 1.0 / (eb * ab) + 1.0 / (ek * ak) + (ck**2) / ei_tot],
            ],
            dtype=float,
        )
        rhs = np.array([(alphab - alphar) * dt_val, (alphab - alphak) * dt_val], dtype=float)
        p_r, p_k = np.linalg.solve(m_mat, rhs)
        kappa_bk = float((gamma_rail * p_r * c_rail - gamma_k * p_k * ck) / ei_tot)
        return abs(kappa_bk) * (l_x**2) / 8.0 * 1e6, kappa_bk

    def _bimetallic_bow_um(dt_val: float, use_backer: bool) -> float:
        return _bimetallic_bow_signed(dt_val, use_backer)[0]

    bow_no_backer_um = _bimetallic_bow_um(delta_t, use_backer=False)
    bow_actual_um, signed_kappa_actual = _bimetallic_bow_signed(delta_t, use_backer=True)

    # Positive signed_kappa means beam expands more than rail (beam face convex -> center bows away from rail);
    # negative means rail expands more (CF beam vs steel rail -> steel rail convex -> center bows toward rail).
    if bow_actual_um <= 0.5:
        bow_direction = "neutral"
    elif signed_kappa_actual > 1e-12:
        bow_direction = "away_from_rail"
    elif signed_kappa_actual < -1e-12:
        bow_direction = "toward_rail"
    else:
        bow_direction = "neutral"

    if spec.architecture == "delta":
        backer_reduction_pct = 0.0
    elif x_mount == "dual_top_bottom":
        backer_reduction_pct = 100.0
    elif spec.gantry.rail_backer_material != "none":
        ref_0 = _bimetallic_bow_um(1.0, use_backer=False)
        ref_bk = _bimetallic_bow_um(1.0, use_backer=True)
        backer_reduction_pct = max(0.0, min(100.0, 100.0 * (1.0 - ref_bk / max(ref_0, 1e-12))))
    else:
        backer_reduction_pct = 0.0

    # 3. Heated bed thermal gradient bow + rigid 4-pt thermal taco buckling
    bed_temp = float(spec.bed.target_temp_c)
    bed_mount = spec.bed.mount_type
    frame_mat = get_material(spec.frame.material)
    bed_grad_bow_um, bed_buckle_um = compute_bed_thermal_bow_and_taco_um(spec)

    # 4. Vertical frame Z-drift (column CTE - Z-drive CTE) * H_active * delta_T
    if spec.bed.z_drive == "belted_quad_z":
        alpha_z_drive = {"fiberglass": 5.5e-6, "aramid": -2.0e-6, "steel": 12.2e-6}.get(
            spec.gantry.belt.core, 5.5e-6
        )
    else:
        # Leadscrews (hardened/stainless steel)
        alpha_z_drive = 12.2e-6
    h_active = max(min(spec.kinematics.build_volume_mm[2], spec.frame.height_mm - 80.0) * 1e-3, 0.15)
    frame_z_drift_um = (frame_mat.thermal_expansion_1_k - alpha_z_drive) * h_active * delta_t * 1e6

    # 5. Bracket polymer modulus retention E(T) & Tg warning
    bkt_mat = get_material(spec.gantry.bracket_material)
    ret_factor = bkt_mat.modulus_retention_at_temp(chamber_c)
    retention_pct = ret_factor * 100.0
    tg_warn = False
    if bkt_mat.tg_c is not None:
        if chamber_c >= (bkt_mat.tg_c - 5.0) or ret_factor < 0.80:
            tg_warn = True
    elif ret_factor < 0.80:
        tg_warn = True

    # Aggregate Z-axis thermal error
    z_bow_factor = 1.0 if x_mount in ("top", "bottom") else 0.25
    total_z_err_um = (
        z_bow_factor * bow_actual_um
        + bed_grad_bow_um
        + bed_buckle_um
        + abs(frame_z_drift_um)
    )

    return ThermalStructuralReport(
        chamber_temp_c=chamber_c,
        delta_t_k=delta_t,
        gantry_bimetallic_bow_um=bow_actual_um,
        gantry_backer_reduction_pct=backer_reduction_pct,
        bed_target_temp_c=bed_temp,
        bed_mount_type=bed_mount,
        bed_gradient_bow_um=bed_grad_bow_um,
        bed_buckling_taco_um=bed_buckle_um,
        frame_z_drift_um=frame_z_drift_um,
        bracket_modulus_retention_pct=retention_pct,
        bracket_tg_warning=tg_warn,
        total_thermal_z_error_um=total_z_err_um,
        rail_newmark_gamma=gamma_rail,
        rail_carriage_roll_nm_per_rad=roll_nm_rad,
        rail_carriage_radial_n_per_um=radial_n_um,
        x_beam_modulus_retention_pct=x_beam_mod_ret * 100.0,
        x_beam_shear_retention_pct=x_beam_shear_ret * 100.0,
        x_beam_tg_warning=x_beam_tg_warn,
        x_beam_has_tg=x_beam_has_tg,
        gantry_bow_direction=bow_direction,
    )
