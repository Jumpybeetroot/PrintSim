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
from printsim.schema.printer import BED_MOUNT_CATALOG, RAIL_QUALITY_CATALOG, BeltSpec

if TYPE_CHECKING:
    from printsim.geometry.builder import AssemblyModel
    from printsim.schema.printer import PrinterSpec

_G_ACC = 9.80665

# HIWIN MGN / HGH catalog parameters: (ball_d_m, row_half_pitch_m, n_rows, L1_C_m, C_C_n, L1_H_m, C_H_n, seal_drag_n)
_RAIL_CARRIAGE_CATALOG: dict[str, tuple[float, float, int, float, float, float, float, float]] = {
    "mgn9_rail": (1.588e-3, 0.0033, 2, 0.0189, 1860.0, 0.0289, 2550.0, 0.35),
    "mgn12_rail": (2.381e-3, 0.0042, 2, 0.0217, 2840.0, 0.0324, 3920.0, 0.50),
    "mgn15_rail": (3.175e-3, 0.0054, 2, 0.0267, 4610.0, 0.0434, 6370.0, 0.75),
    "hgh15_rail": (3.175e-3, 0.0062, 4, 0.0394, 11380.0, 0.0568, 14700.0, 1.20),
    # Round linear ball bushings (LM8UU/LM10UU/LM12UU equivalent baseline)
    "steel_rod_8mm": (1.588e-3, 0.0040, 4, 0.0170, 265.0, 0.0240, 420.0, 0.40),
    "steel_rod_10mm": (1.984e-3, 0.0050, 4, 0.0200, 372.0, 0.0290, 588.0, 0.45),
    "steel_rod_12mm": (2.381e-3, 0.0060, 4, 0.0220, 412.0, 0.0300, 650.0, 0.50),
}

_PRELOAD_FRACTION: dict[str, float] = {
    "ZF": 0.002,
    "Z0": 0.010,
    "Z1": 0.030,
    "ZA": 0.060,
    "ZB": 0.100,
}

# Huth-Schwarmann M3 fastener shear stiffness k_s (N/m) per substrate type
_BOLT_SHEAR_STIFFNESS_N_M: dict[str, float] = {
    "extrusion": 18.0e6,
    "carbon_tube": 10.0e6,
    "solid_bar": 42.0e6,
}

# Corner joint base stiffness for 2020 Al extrusion: (k_t_n_m, k_r_inplane_nm_rad, k_r_torsion_nm_rad)
_CORNER_JOINT_BASE_2020: dict[str, tuple[float, float, float]] = {
    "l_bracket": (3.5e6, 4500.0, 1800.0),
    "blind_joint": (1.2e7, 22000.0, 7500.0),
    "gusset_plate": (2.2e7, 42000.0, 14000.0),
    "corner_cube": (3.8e7, 85000.0, 28000.0),
    "rigid": (1.0e10, 1.0e9, 1.0e9),
}

_IWAN_DAMPING_DELTA: dict[str, float] = {
    "rigid": 0.0,
    "corner_cube": 0.001,
    "gusset_plate": 0.002,
    "blind_joint": 0.003,
    "l_bracket": 0.005,
}

_ISO_5753_RADIAL_PLAY_UM: dict[str, float] = {
    "C2": 3.5,
    "CN": 7.5,
    "C3": 15.5,
}


def hertzian_rail_carriage_stiffness(
    rail_profile_id: str,
    preload: str = "Z1",
    block_type: str = "H",
    rail_quality: str = "hiwin_genuine",
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
    is_h = block_type.upper() == "H"
    l1 = l1_h if is_h else l1_c
    c_dyn = c_h if is_h else c_c

    pre_key = preload.upper()
    f_pre = _PRELOAD_FRACTION.get(pre_key, 0.030)
    f_pre_z1 = _PRELOAD_FRACTION["Z1"]

    qual_key = str(rail_quality).lower().strip() if rail_quality else "hiwin_genuine"
    qual_cfg = RAIL_QUALITY_CATALOG.get(qual_key, RAIL_QUALITY_CATALOG["hiwin_genuine"])
    stiffness_mult = float(qual_cfg["stiffness_mult"])
    drag_mult = float(qual_cfg["drag_mult"])

    # Active balls in the loaded raceway zone along L1 (~1.05 * d_b pitch)
    n_b = max(4, int(round(l1 / (1.05 * d_b))))
    sin_alpha = math.sin(math.radians(45.0))

    # Hardened bearing steel (E=207 GPa, nu=0.29) elliptical contact (f=0.525 -> R_e=d_b*sqrt(5.25), F2=0.68)
    e_steel = 207.0e9
    nu_steel = 0.29
    e_star = e_steel / (2.0 * (1.0 - nu_steel**2))
    r_e = d_b * math.sqrt(5.25)
    f2_ellip = 0.68
    c_hertz = ((9.0 / (16.0 * r_e * (e_star**2))) ** (1.0 / 3.0)) * f2_ellip

    def _raw_hertz_radial(f_frac: float) -> float:
        q_ball = max((f_frac * c_dyn) / (n_rows * n_b * sin_alpha), 0.15)
        k_n = (1.5 / c_hertz) * (q_ball ** (1.0 / 3.0))
        # 2-row Gothic arch (n_rows=2): 2*sin^2(45)=1.0; 4-row HGH (n_rows=4): 4*0.5=2.0
        row_factor = 0.5 * n_rows
        return row_factor * n_b * k_n

    k_hertz_raw = _raw_hertz_radial(f_pre)
    # Slight clearance ZF unloads opposing ball row under initial load -> ~55% contact stiffness
    if pre_key == "ZF":
        k_hertz_raw *= 0.55
    # Ball-tolerance load sharing and raceway grinding conformity scale Hertzian contact stiffness
    k_hertz_raw *= stiffness_mult

    # 15% Hung carriage-skirt structural compliance referenced to Z1 nominal block stiffness
    k_skirt = _raw_hertz_radial(f_pre_z1) / 0.15
    k_eff_radial = (k_hertz_raw * k_skirt) / max(k_hertz_raw + k_skirt, 1e-9)

    k_yy = k_eff_radial
    k_zz = k_eff_radial * (1.05 if n_rows == 4 else 1.0)

    # Roll stiffness about rail X axis (two side tracks at +-r_y with wing flex factor 1.5)
    k_rx = k_zz * (r_y**2) / 1.5
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
    }


def newmark_rail_composite_gamma(
    e_rail_pa: float,
    a_rail_m2: float,
    e_beam_pa: float,
    a_beam_m2: float,
    bolt_pitch_m: float,
    l_eff_m: float,
    substrate_kind: str = "extrusion",
) -> float:
    """Newmark (1951) partial-interaction shear-lag composite factor gamma in (0, 1).

    Computes ``gamma = 1 / (1 + pi^2 * EA* * s / (L_eff^2 * k_s))`` using Huth-Schwarmann M3
    fastener shear stiffness ``k_s`` for aluminum extrusion (~18 MN/m), carbon fiber tube
    (~10 MN/m), or tapped solid metal bar (~42 MN/m).
    """
    ea_rail = max(float(e_rail_pa) * float(a_rail_m2), 1.0)
    ea_beam = max(float(e_beam_pa) * float(a_beam_m2), 1.0)
    ea_star = (ea_rail * ea_beam) / (ea_rail + ea_beam)

    s = max(float(bolt_pitch_m), 0.005)
    l_eff = max(float(l_eff_m), 0.05)

    sub_key = substrate_kind.lower().strip()
    if sub_key.startswith("cf") or "carbon" in sub_key:
        k_s = _BOLT_SHEAR_STIFFNESS_N_M["carbon_tube"]
    elif "solid" in sub_key:
        k_s = _BOLT_SHEAR_STIFFNESS_N_M["solid_bar"]
    else:
        k_s = _BOLT_SHEAR_STIFFNESS_N_M["extrusion"]

    return 1.0 / (1.0 + (math.pi**2) * ea_star * s / ((l_eff**2) * k_s))


def corner_joint_stiffness(profile_id: str, joint_type: str) -> dict[str, float]:
    """Compute semi-rigid frame corner joint translational & rotational stiffnesses (SI)."""
    jkey = joint_type.lower().strip()
    if jkey not in _CORNER_JOINT_BASE_2020:
        jkey = "blind_joint"

    k_t_base, k_r_base, k_tor_base = _CORNER_JOINT_BASE_2020[jkey]
    if jkey == "rigid":
        return {
            "k_t_n_m": k_t_base,
            "k_r_inplane_nm_rad": k_r_base,
            "k_r_torsion_nm_rad": k_tor_base,
        }

    prof = get_profile(profile_id)
    i_avg = 0.5 * (prof.iy_m4 + prof.iz_m4)
    scale_t = max(prof.width_m / 0.020, 0.4)
    scale_r = max((i_avg / 0.68e-8) ** 0.65, 0.25)

    return {
        "k_t_n_m": k_t_base * scale_t,
        "k_r_inplane_nm_rad": k_r_base * scale_r,
        "k_r_torsion_nm_rad": k_tor_base * scale_r,
    }


def iwan_joint_damping_delta(joint_type: str) -> float:
    """Small physical Iwan micro-slip damping increment delta_zeta by frame corner joint type."""
    return _IWAN_DAMPING_DELTA.get(joint_type.lower().strip(), 0.003)


def belt_effective_ea_n(belt: BeltSpec, span_m: float) -> float:
    """Effective belt tensile modulus EA (N) from cord crimp straightening & Ernst catenary sag."""
    t0 = max(float(belt.pretension_n), 0.5)
    mu = max(float(belt.linear_mass_kg_m), 1e-4)
    span = max(float(span_m), 0.05)

    # 1. Cord crimp straightening under tension T0
    crimp_factor = 1.0 - 0.32 * math.exp(-t0 / 15.0)
    ea_crimp = belt.stiffness_ea_n * crimp_factor

    # 2. Ernst (1965) catenary sag axial stiffness k_cat = 12*T0^3 / ((mu*g)^2 * L^3) -> EA_cat = k_cat * L
    k_cat = 12.0 * (t0**3) / (((mu * _G_ACC) ** 2) * (span**3))
    ea_cat = k_cat * span

    return (ea_crimp * ea_cat) / max(ea_crimp + ea_cat, 1e-9)


def gerbert_tooth_mesh_stiffness_n_m(
    belt: BeltSpec,
    ea_eff_n: float,
    chamber_temp_c: float = 25.0,
) -> float:
    """Gerbert (1972) discrete spring-chain tooth-mesh shear stiffness (N/m) with elastomer T(chamber) retention."""
    w_mm = float(belt.width_mm)
    k_t = float(belt.tooth_shear_k_n_per_m_mm) * belt.tooth_modulus_retention(chamber_temp_c)
    p_m = belt.pitch_mm * 1e-3
    z_e = max(float(belt.pulley_teeth) * (float(belt.wrap_angle_deg) / 360.0), 1.0)
    ea_safe = max(float(ea_eff_n), 100.0)

    lam = math.acosh(1.0 + (k_t * w_mm * p_m) / (2.0 * ea_safe))
    return k_t * w_mm * math.tanh(lam * z_e) / max(math.tanh(lam), 1e-9)


def idler_shaft_and_bearing_stiffness(
    belt: BeltSpec,
    double_shear: bool,
    bracket_e_pa: float,
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
    bearing_play_um = _ISO_5753_RADIAL_PLAY_UM.get(belt.bearing_clearance_class, 7.5)

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
    mount_cfg = BED_MOUNT_CATALOG.get(spec.bed.mount_type, BED_MOUNT_CATALOG["kinematic_3pt"])
    mount_inset_ratio = float(mount_cfg.get("mount_inset_ratio", 0.38))
    l_mount = 2.0 * mount_inset_ratio * l_bed
    dl_thermal = (
        bed_mat.thermal_expansion_1_k * max(0.0, bed_temp - 25.0)
        - frame_mat.thermal_expansion_1_k * max(0.0, chamber_c - 25.0)
    ) * l_mount
    dl_clearance = 0.00015  # 0.15 mm typical M4 mounting hole clearance
    dl_excess = max(0.0, dl_thermal - dl_clearance)

    k_restraint = float(mount_cfg["k_restraint_thermal"])
    if k_restraint <= 0.0 or dl_excess <= 0.0:
        bed_buckle_um = 0.0
    else:
        # Elastica arch amplitude attenuated by mounting standoff flexibility & plate flexural rigidity D
        d_plate = (
            bed_mat.youngs_modulus_pa
            * (t_bed**3)
            / (12.0 * (1.0 - bed_mat.poisson_ratio**2))
        )
        k_plate_taco = 48.0 * d_plate * l_bed / max(l_mount**3, 1e-9)
        w_elastica = (2.0 / math.pi) * math.sqrt(l_mount * dl_excess)
        # Fraction of excess shortening absorbed into out-of-plane taco arch vs standoff/frame flex
        arch_fraction = k_restraint / (k_restraint + k_plate_taco)
        bed_buckle_um = w_elastica * math.sqrt(arch_fraction) * 1e6

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
    if x_beam_prof.id.startswith("cf_"):
        sub_kind = "carbon_tube"
    elif x_beam_prof.shape == "solid_bar":
        sub_kind = "solid_bar"
    else:
        sub_kind = "extrusion"

    gamma_rail = newmark_rail_composite_gamma(
        e_rail_pa=rail_mat.youngs_modulus_pa,
        a_rail_m2=x_rail_prof.area_m2,
        e_beam_pa=x_beam_mat.youngs_modulus_pa,
        a_beam_m2=x_beam_prof.area_m2,
        bolt_pitch_m=bolt_pitch_m,
        l_eff_m=l_x,
        substrate_kind=sub_kind,
    )

    carriage = hertzian_rail_carriage_stiffness(
        spec.gantry.x_rail_profile,
        preload=spec.gantry.rail_preload,
        block_type=spec.gantry.rail_block_type,
        rail_quality=spec.gantry.rail_quality,
    )
    car_mult = 2.0 if spec.gantry.effective_dual_x_rails else 1.0
    roll_nm_rad = carriage["k_rx_nm_rad"] * car_mult
    radial_n_um = carriage["k_zz_n_m"] * car_mult * 1e-6

    # 2. Timoshenko (1925) partial-interaction bimetallic bow on X gantry
    x_mount = spec.gantry.effective_x_rail_mount
    eb, ab, ib, alphab = (
        x_beam_mat.youngs_modulus_pa,
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

    def _bimetallic_bow_um(dt_val: float, use_backer: bool) -> float:
        if x_mount == "dual_top_bottom" or spec.architecture == "delta":
            return 0.0
        kappa_0 = (alphab - alphar) * dt_val * c_rail / (c_rail**2 + ei_sum * ea_inv)
        bow_0 = abs(gamma_rail * kappa_0) * (l_x**2) / 8.0 * 1e6
        if not use_backer or spec.gantry.rail_backer_material == "none":
            return bow_0

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
        kappa_bk = (gamma_rail * p_r * c_rail - gamma_k * p_k * ck) / ei_tot
        return abs(float(kappa_bk)) * (l_x**2) / 8.0 * 1e6

    bow_no_backer_um = _bimetallic_bow_um(delta_t, use_backer=False)
    bow_actual_um = _bimetallic_bow_um(delta_t, use_backer=True)

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
    )
