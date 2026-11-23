"""Kinematics (CoreXY, Cartesian, Delta), motor torque curves, and belt dynamics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from printsim.geometry.builder import AssemblyModel
from printsim.schema import (
    BELT_ROUTING_CATALOG,
    GANTRY_BRACKET_CATALOG,
    PrinterSpec,
    StepperSpec,
    get_material,
)


@dataclass
class KinematicsReport:
    architecture: str
    build_volume_liters: float
    frame_volume_liters: float
    packaging_efficiency_pct: float
    max_motor_rpm: float
    available_torque_at_vmax_nm: float
    required_torque_x_nm: float
    required_torque_y_nm: float
    torque_safety_margin: float
    max_safe_accel_motor_mm_s2: float
    step_skip_risk: bool
    belt_peak_tension_n: float
    belt_slip_risk: bool
    delta_workspace_valid: bool = True
    drive_mode: str = "2wd"
    xy_motor_count: int = 2
    belt_pluck_freq_150mm_hz: float = 0.0
    belt_max_working_tension_n: float = 58.0
    belt_overtension_risk: bool = False
    shaft_bending_risk: bool = False
    frame_belt_preload_n: float = 0.0
    motor_run_current_a: float = 1.2
    motor_stiffness_n_per_mm: float = 0.0
    belt_stiffness_n_per_mm: float = 0.0
    bracket_material: str = "abs_printed"
    bracket_stiffness_n_per_mm: float = 0.0
    bracket_creep_risk: bool = False
    drive_stiffness_n_per_mm: float = 0.0
    motor_voltage_efficiency_pct: float = 100.0
    motor_heat_w_each: float = 0.0
    motor_overcurrent_risk: bool = False
    tooth_mesh_stiffness_n_per_mm: float = 0.0
    idler_shaft_stiffness_n_per_mm: float = 0.0
    idler_shaft_deflection_um: float = 0.0
    bearing_play_um: float = 7.5
    bracket_modulus_retention_pct: float = 100.0
    belt_routing: str = "stock_corexy"
    steel_core_fatigue_risk: bool = False
    belt_bend_fatigue_risk: bool = False
    belt_creep_risk: bool = False
    belt_min_pulley_teeth: int = 16
    belt_min_back_idler_mm: float = 12.0
    belt_back_idler_mm: float = 12.7
    belt_body_material: str = "epdm"
    belt_thermal_retention_pct: float = 100.0
    belt_label: str = "Gates 2GT 6mm (fiberglass)"
    belt_validation_tier: str = "proven"
    belt_theoretical_summary: str = ""
    belt_practical_issues: list[str] = field(default_factory=list)
    pulley_circumference_mm: float = 40.0
    vfa_belt_tooth_speed_range_mm_s: list[float] = field(default_factory=lambda: [60.0, 110.0])
    custom_bracket_active: bool = False
    custom_bracket_label: str = ""

    def to_dict(self) -> dict:
        return {
            "architecture": self.architecture,
            "build_volume_liters": round(self.build_volume_liters, 2),
            "frame_volume_liters": round(self.frame_volume_liters, 2),
            "packaging_efficiency_pct": round(self.packaging_efficiency_pct, 1),
            "max_motor_rpm": round(self.max_motor_rpm, 1),
            "available_torque_at_vmax_nm": round(self.available_torque_at_vmax_nm, 3),
            "required_torque_x_nm": round(self.required_torque_x_nm, 3),
            "required_torque_y_nm": round(self.required_torque_y_nm, 3),
            "torque_safety_margin": round(self.torque_safety_margin, 2),
            "max_safe_accel_motor_mm_s2": round(self.max_safe_accel_motor_mm_s2, 0),
            "step_skip_risk": self.step_skip_risk,
            "belt_peak_tension_n": round(self.belt_peak_tension_n, 2),
            "belt_slip_risk": self.belt_slip_risk,
            "delta_workspace_valid": self.delta_workspace_valid,
            "drive_mode": self.drive_mode,
            "belt_routing": self.belt_routing,
            "xy_motor_count": self.xy_motor_count,
            "belt_pluck_freq_150mm_hz": round(self.belt_pluck_freq_150mm_hz, 2),
            "belt_max_working_tension_n": round(self.belt_max_working_tension_n, 2),
            "belt_overtension_risk": self.belt_overtension_risk,
            "steel_core_fatigue_risk": self.steel_core_fatigue_risk,
            "belt_bend_fatigue_risk": self.belt_bend_fatigue_risk,
            "belt_creep_risk": self.belt_creep_risk,
            "belt_min_pulley_teeth": self.belt_min_pulley_teeth,
            "belt_min_back_idler_mm": round(self.belt_min_back_idler_mm, 1),
            "belt_back_idler_mm": round(self.belt_back_idler_mm, 1),
            "belt_body_material": self.belt_body_material,
            "belt_thermal_retention_pct": round(self.belt_thermal_retention_pct, 1),
            "belt_label": self.belt_label,
            "belt_validation_tier": self.belt_validation_tier,
            "belt_theoretical_summary": self.belt_theoretical_summary,
            "belt_practical_issues": list(self.belt_practical_issues),
            "pulley_circumference_mm": round(self.pulley_circumference_mm, 1),
            "vfa_belt_tooth_speed_range_mm_s": [
                round(float(v), 1) for v in self.vfa_belt_tooth_speed_range_mm_s
            ],
            "shaft_bending_risk": self.shaft_bending_risk,
            "frame_belt_preload_n": round(self.frame_belt_preload_n, 2),
            "motor_run_current_a": round(self.motor_run_current_a, 2),
            "motor_stiffness_n_per_mm": round(self.motor_stiffness_n_per_mm, 1),
            "belt_stiffness_n_per_mm": round(self.belt_stiffness_n_per_mm, 1),
            "bracket_material": self.bracket_material,
            "bracket_stiffness_n_per_mm": round(self.bracket_stiffness_n_per_mm, 1),
            "bracket_creep_risk": self.bracket_creep_risk,
            "drive_stiffness_n_per_mm": round(self.drive_stiffness_n_per_mm, 1),
            "motor_voltage_efficiency_pct": round(self.motor_voltage_efficiency_pct, 1),
            "motor_heat_w_each": round(self.motor_heat_w_each, 2),
            "motor_overcurrent_risk": self.motor_overcurrent_risk,
            "tooth_mesh_stiffness_n_per_mm": round(self.tooth_mesh_stiffness_n_per_mm, 1),
            "idler_shaft_stiffness_n_per_mm": round(self.idler_shaft_stiffness_n_per_mm, 1),
            "idler_shaft_deflection_um": round(self.idler_shaft_deflection_um, 2),
            "bearing_play_um": round(self.bearing_play_um, 1),
            "bracket_modulus_retention_pct": round(self.bracket_modulus_retention_pct, 1),
            "custom_bracket_active": bool(self.custom_bracket_active),
            "custom_bracket_label": self.custom_bracket_label,
        }


def _delta_towers_xy(spec: PrinterSpec) -> np.ndarray:
    r = spec.kinematics.delta_radius_mm
    angles = [math.radians(90.0), math.radians(210.0), math.radians(330.0)]
    return np.array([[r * math.cos(a), r * math.sin(a)] for a in angles], dtype=float)


def inverse_kinematics(spec: PrinterSpec, xyz_mm: tuple[float, float, float]) -> tuple[float, float, float]:
    """Map Cartesian nozzle coordinates (mm) to actuator positions (A, B, C) in mm."""
    x, y, z = xyz_mm
    if spec.architecture == "corexy":
        return (x + y, x - y, z)
    if spec.architecture == "cartesian_bedslinger":
        return (x, y, z)
    if spec.architecture == "delta":
        # Center delta coordinates around build center
        bx, by, _ = spec.kinematics.build_volume_mm
        cx, cy = x - bx * 0.5, y - by * 0.5
        arm = spec.kinematics.delta_arm_length_mm
        towers = _delta_towers_xy(spec)
        zs = []
        for tx, ty in towers:
            d2 = (cx - tx) ** 2 + (cy - ty) ** 2
            if d2 >= arm**2:
                raise ValueError(f"Delta position {xyz_mm} outside arm reach (L={arm}mm)")
            zs.append(z + math.sqrt(arm**2 - d2))
        return (zs[0], zs[1], zs[2])
    raise ValueError(f"Unsupported architecture: {spec.architecture}")


def forward_kinematics(spec: PrinterSpec, abc_mm: tuple[float, float, float]) -> tuple[float, float, float]:
    """Map actuator positions (A, B, C) in mm back to Cartesian nozzle coordinates (mm)."""
    a, b, c = abc_mm
    if spec.architecture == "corexy":
        return (0.5 * (a + b), 0.5 * (a - b), c)
    if spec.architecture == "cartesian_bedslinger":
        return (a, b, c)
    if spec.architecture == "delta":
        # Exact trilateration of 3 spheres centered at (tx_i, ty_i, z_i) with radius L
        bx, by, _ = spec.kinematics.build_volume_mm
        arm = spec.kinematics.delta_arm_length_mm
        towers = _delta_towers_xy(spec)
        p1 = np.array([towers[0, 0], towers[0, 1], a])
        p2 = np.array([towers[1, 0], towers[1, 1], b])
        p3 = np.array([towers[2, 0], towers[2, 1], c])

        ex = p2 - p1
        d = float(np.linalg.norm(ex))
        ex /= d
        i_val = float(np.dot(ex, p3 - p1))
        ey = p3 - p1 - i_val * ex
        ey /= float(np.linalg.norm(ey))
        ez = np.cross(ex, ey)
        j_val = float(np.dot(ey, p3 - p1))

        x_loc = d / 2.0
        y_loc = (i_val**2 + j_val**2) / (2.0 * j_val) - (i_val / j_val) * x_loc
        z_sq = arm**2 - x_loc**2 - y_loc**2
        if z_sq < 0:
            raise ValueError("Invalid delta carriage heights; spheres do not intersect")
        z_loc = math.sqrt(z_sq)

        # Pick lower intersection (effector hangs below carriages)
        cand1 = p1 + x_loc * ex + y_loc * ey + z_loc * ez
        cand2 = p1 + x_loc * ex + y_loc * ey - z_loc * ez
        pt = cand1 if cand1[2] < cand2[2] else cand2
        return (float(pt[0] + bx * 0.5), float(pt[1] + by * 0.5), float(pt[2]))
    raise ValueError(f"Unsupported architecture: {spec.architecture}")


def stepper_available_torque_nm(stepper: StepperSpec, rpm: float) -> float:
    """Electromagnetic pull-out torque model including current saturation and L/R voltage roll-off."""
    # Dynamic holding-to-pull-out factor (~0.72 at low speed) scaled by current and inductive efficiency
    return stepper.effective_holding_torque_nm * 0.72 * stepper.dynamic_voltage_efficiency(rpm)


def compute_bracket_stiffness_n_per_m(assembly: AssemblyModel) -> float:
    """Compute effective gantry bracket / motor-mount stiffness in series (N/m), scaled by E(T)."""
    spec = assembly.spec
    override = spec.gantry.custom_bracket_override
    if override is not None:
        return float(override.stiffness_n_per_mm) * 1e3
    bkt_mat = get_material(spec.gantry.bracket_material)
    e_bkt = bkt_mat.effective_modulus_pa(spec.frame.chamber_temp_c)
    k_ref = 5.0e5  # Base 2WD bolted ABS motor+idler+XY-joint chain (500 N/mm at 2.1 GPa; vs 1842 N/mm solid ABS mount FEA)
    ds_mult = 1.45 if spec.gantry.double_shear_pulleys else 1.0
    awd_mult = 1.65 if (spec.architecture == "corexy" and spec.gantry.drive_mode == "awd") else 1.0
    k_mat = k_ref * ds_mult * awd_mult * (e_bkt / 2.1e9)
    k_ceil = 8.0e6  # Bolted fastener / bearing compliance ceiling (8,000 N/mm)
    return (k_mat * k_ceil) / max(k_mat + k_ceil, 1e-9)


def compute_drive_stiffness_details_n_per_m(
    assembly: AssemblyModel, axis: str = "X"
) -> dict[str, object]:
    """Compute detailed series drive-train stiffness components (cord, tooth mesh, idler, motor, bracket)."""
    from printsim.fea.components import (
        belt_effective_ea_n,
        gerbert_tooth_mesh_stiffness_n_m,
        idler_shaft_and_bearing_stiffness,
    )

    spec = assembly.spec
    belt = spec.gantry.belt
    kin = spec.kinematics
    w = spec.frame.width_mm * 1e-3
    d = spec.frame.depth_mm * 1e-3
    bkt_mat = get_material(spec.gantry.bracket_material)
    e_bkt_eff = bkt_mat.effective_modulus_pa(spec.frame.chamber_temp_c)

    routing_cfg = BELT_ROUTING_CATALOG.get(
        spec.gantry.belt_routing, BELT_ROUTING_CATALOG["stock_corexy"]
    )
    span_factor = float(routing_cfg["span_factor"])
    idler_factor = float(routing_cfg["idler_stiffness_factor"])
    idler_stiffness_mult = idler_factor if spec.architecture == "corexy" else 1.0

    if spec.architecture == "corexy":
        if spec.gantry.drive_mode == "awd":
            # 4-motor AWD divides belt loops into 4 independent motor-to-carriage spans
            span_len = max(0.50 * (w + d), 0.16) * span_factor
            ea_eff = belt_effective_ea_n(belt, span_len)
            k_belt_cord = 4.0 * ea_eff / span_len
            drive_branches = 4.0
        else:
            # 2WD CoreXY: 2 parallel belts, each with short motor branch & long return loop in parallel
            span_short = max(0.45 * (w + d), 0.18) * span_factor
            span_long = max(1.35 * (w + d), 0.48) * span_factor
            ea_short = belt_effective_ea_n(belt, span_short)
            ea_long = belt_effective_ea_n(belt, span_long)
            k_belt_cord = 2.0 * (ea_short / span_short + ea_long / span_long)
            ea_eff = 0.5 * (ea_short + ea_long)
            drive_branches = 2.0
    elif spec.architecture == "delta":
        span_len = max(spec.frame.height_mm * 1e-3, 0.45)
        ea_eff = belt_effective_ea_n(belt, span_len)
        k_belt_cord = 2.5 * ea_eff / span_len
        drive_branches = 2.5
    else:
        span_len = max(w if axis.upper() == "X" else d, 0.25)
        ea_eff = belt_effective_ea_n(belt, span_len)
        k_belt_cord = 2.0 * ea_eff / span_len
        drive_branches = 1.0

    k_tooth_mesh = drive_branches * gerbert_tooth_mesh_stiffness_n_m(
        belt, ea_eff, spec.frame.chamber_temp_c
    )
    idler_info = idler_shaft_and_bearing_stiffness(
        belt, double_shear=spec.gantry.double_shear_pulleys, bracket_e_pa=e_bkt_eff
    )
    k_idler_shaft = drive_branches * idler_info["k_idler_n_m"] * idler_stiffness_mult
    if idler_stiffness_mult != 1.0:
        idler_info = dict(idler_info)
        idler_info["idler_shaft_deflection_um"] = (
            idler_info["idler_shaft_deflection_um"] / idler_stiffness_mult
        )

    k_belt_subsystem = 1.0 / (
        1.0 / max(k_belt_cord, 1e-9)
        + 1.0 / max(k_tooth_mesh, 1e-9)
        + 1.0 / max(k_idler_shaft, 1e-9)
    )

    r_p = belt.pitch_radius_m
    motor_lin_v = (kin.max_velocity_mm_s * 1e-3) * (
        1.4142 if spec.architecture == "corexy" else 1.0
    )
    max_rpm = (motor_lin_v / (2.0 * math.pi * r_p)) * 60.0
    rpm_op = max(120.0, 0.35 * max_rpm)
    k_theta = kin.xy_stepper.torsional_stiffness_nm_per_rad(rpm=rpm_op)
    k_motor_single = k_theta / (r_p**2)

    if spec.architecture == "corexy":
        motor_factor = 4.0 if spec.gantry.drive_mode == "awd" else 2.0
    elif spec.architecture == "delta":
        motor_factor = 2.5
    else:
        motor_factor = 1.0

    k_motor_axis = motor_factor * k_motor_single
    override = spec.gantry.custom_bracket_override
    if override is not None:
        k_bracket = float(override.stiffness_n_per_mm) * 1e3
        bracket_retention_pct = 100.0
    else:
        k_bracket = compute_bracket_stiffness_n_per_m(assembly)
        bracket_retention_pct = bkt_mat.modulus_retention_at_temp(spec.frame.chamber_temp_c) * 100.0
    k_drive = 1.0 / (
        1.0 / max(k_belt_subsystem, 1e-9)
        + 1.0 / max(k_motor_axis, 1e-9)
        + 1.0 / max(k_bracket, 1e-9)
    )
    return {
        "k_belt_cord_n_m": k_belt_cord,
        "k_tooth_mesh_n_m": k_tooth_mesh,
        "k_idler_shaft_n_m": k_idler_shaft,
        "k_belt_subsystem_n_m": k_belt_subsystem,
        "k_motor_axis_n_m": k_motor_axis,
        "k_bracket_n_m": k_bracket,
        "k_drive_n_m": k_drive,
        "idler_info": idler_info,
        "bracket_retention_pct": bracket_retention_pct,
    }


def compute_drive_stiffness_n_per_m(
    assembly: AssemblyModel, axis: str = "X"
) -> tuple[float, float, float]:
    """Compute (k_belt_n_m, k_motor_axis_n_m, k_drive_n_m) coupling belt, tooth mesh, idler, motor, & bracket."""
    details = compute_drive_stiffness_details_n_per_m(assembly, axis)
    return (
        float(details["k_belt_subsystem_n_m"]),
        float(details["k_motor_axis_n_m"]),
        float(details["k_drive_n_m"]),
    )


def analyze_kinematics(assembly: AssemblyModel) -> KinematicsReport:
    """Evaluate packaging efficiency, stepper torque margin, belt slip, and workspace."""
    spec = assembly.spec
    kin = spec.kinematics
    belt = spec.gantry.belt

    bx, by, bz = kin.build_volume_mm
    fw, fd, fh = spec.frame.width_mm, spec.frame.depth_mm, spec.frame.height_mm
    build_l = (bx * by * bz) * 1e-6
    frame_l = (fw * fd * fh) * 1e-6
    pkg_eff = 100.0 * build_l / max(frame_l, 1e-6)

    # Pulley kinematics
    r_p = belt.pitch_radius_m
    v_max_m_s = kin.max_velocity_mm_s * 1e-3
    a_max_m_s2 = kin.max_accel_mm_s2 * 1e-3

    # CoreXY diagonal moves run a single motor at sqrt(2)*v, pure X/Y run both motors at v
    motor_lin_v = v_max_m_s * (1.4142 if spec.architecture == "corexy" else 1.0)
    max_rpm = (motor_lin_v / (2.0 * math.pi * r_p)) * 60.0

    # Available torque at 70% peak speed (typical accel-to-cruise knee)
    t_avail = stepper_available_torque_nm(kin.xy_stepper, max_rpm * 0.7)

    is_awd = spec.architecture == "corexy" and spec.gantry.drive_mode == "awd"
    drive_mode = "awd" if is_awd else "2wd"
    xy_motor_count = 4 if is_awd else (3 if spec.architecture == "delta" else 2)

    # Linear friction estimate (extra AWD idlers/motors add modest drag: 3.6 N vs 2.5 N)
    f_fric = 3.6 if is_awd else 2.5
    j_rot = kin.xy_stepper.rotor_inertia_kg_m2
    # CoreXY 45-degree diagonal move accelerates one belt loop & rotor at sqrt(2)*a_max
    alpha_max = ((math.sqrt(2.0) * a_max_m_s2) if spec.architecture == "corexy" else a_max_m_s2) / r_p

    if spec.architecture == "corexy":
        # Worst-case 45-degree diagonal move: one motor carries 1/sqrt(2) of load (2WD) or 1/(2*sqrt(2)) (AWD)
        f_x = assembly.x_moving_mass_kg * a_max_m_s2 + f_fric
        f_y = assembly.y_moving_mass_kg * a_max_m_s2 + f_fric
        share = (1.0 / (2.0 * math.sqrt(2.0))) if is_awd else (1.0 / math.sqrt(2.0))
        t_req_x = (share * f_x) * r_p + j_rot * alpha_max
        t_req_y = (share * f_y) * r_p + j_rot * alpha_max
        peak_belt_force = max(f_x, f_y) * share
    elif spec.architecture == "delta":
        # 3 motors share XY load (~0.65 peak factor for worst-case arm angle)
        f_eff = assembly.x_moving_mass_kg * a_max_m_s2 + f_fric
        t_req_x = 0.65 * f_eff * r_p + j_rot * alpha_max
        t_req_y = t_req_x
        peak_belt_force = 0.65 * f_eff
    else:
        # Cartesian bedslinger: 1 motor for X, 1 motor for heavy Y bed
        f_x = assembly.x_moving_mass_kg * a_max_m_s2 + f_fric
        f_y = assembly.y_moving_mass_kg * a_max_m_s2 + f_fric * 1.4
        t_req_x = f_x * r_p + j_rot * alpha_max
        t_req_y = f_y * r_p + j_rot * alpha_max
        peak_belt_force = max(f_x, f_y)

    t_req_max = max(t_req_x, t_req_y, 1e-6)
    margin = t_avail / t_req_max
    # Safe motor-limited acceleration with 1.35x dynamic safety factor
    max_safe_a = kin.max_accel_mm_s2 * (margin / 1.35)

    stepper = kin.xy_stepper
    details = compute_drive_stiffness_details_n_per_m(assembly, "X")
    k_belt_n_m = float(details["k_belt_subsystem_n_m"])
    k_tooth_n_m = float(details["k_tooth_mesh_n_m"])
    k_idler_n_m = float(details["k_idler_shaft_n_m"])
    k_motor_n_m = float(details["k_motor_axis_n_m"])
    k_bracket_n_m = float(details["k_bracket_n_m"])
    k_drive_n_m = float(details["k_drive_n_m"])
    idler_info: dict[str, float] = details["idler_info"]  # type: ignore[assignment]
    bracket_ret_pct = float(details["bracket_retention_pct"])

    # Belt dynamic tension: tight side = T0 + F/2, slack side = T0 - F/2 (Euler-Eytelwein + cam-out slip check)
    peak_tension = belt.pretension_n + 0.5 * peak_belt_force
    slack_tension = belt.pretension_n - 0.5 * peak_belt_force
    belt_slip = slack_tension < idler_info["min_slack_tension_n"]

    belt_pluck_150 = belt.pluck_freq_hz(150.0)
    belt_max_t = belt.max_working_tension_n
    belt_overtension = peak_tension > belt_max_t
    shaft_bending = (
        idler_info["idler_shaft_deflection_um"] > 12.0
        or ((belt.pretension_n > 45.0) and (not spec.gantry.double_shear_pulleys))
    )
    frame_preload = 2.0 * belt.pretension_n * (2.0 if is_awd else 1.0)

    # Validate Delta reachability at perimeter corners if Delta
    delta_ok = True
    if spec.architecture == "delta":
        for px in (0.0, bx):
            for py in (0.0, by):
                try:
                    inverse_kinematics(spec, (px, py, 0.0))
                except ValueError:
                    delta_ok = False

    bkt_id = spec.gantry.bracket_material
    bkt_info = GANTRY_BRACKET_CATALOG.get(bkt_id, GANTRY_BRACKET_CATALOG["abs_printed"])
    bracket_creep = belt.pretension_n > float(bkt_info["creep_tension_limit_n"])
    motor_volt_eff_pct = stepper.dynamic_voltage_efficiency(max_rpm * 0.7) * 100.0
    motor_heat_w = stepper.power_dissipation_w
    motor_overcurrent = stepper.run_current_a > stepper.rated_current_a * 1.05

    min_pulley_t = belt.min_pulley_teeth
    min_back_idler = belt.min_back_idler_diameter_mm
    eff_back_idler = belt.effective_back_idler_diameter_mm
    belt_thermal_ret_pct = belt.tooth_modulus_retention(spec.frame.chamber_temp_c) * 100.0
    # Reverse-bend fatigue check: CoreXY uses smooth back-side idlers (contraflexure)
    has_reverse_bend = spec.architecture == "corexy"
    pulley_too_small = belt.pulley_teeth < min_pulley_t
    back_idler_too_small = has_reverse_bend and (eff_back_idler + 0.25 < min_back_idler)
    steel_core_fatigue = belt.core == "steel" and (
        (belt.type.startswith("gt2_") and belt.pulley_teeth <= 20)
        or pulley_too_small
        or back_idler_too_small
    )
    belt_bend_fatigue = pulley_too_small or back_idler_too_small or steel_core_fatigue
    # Creep check: aramid viscoelastic relaxation under preload OR TPU body softening > 55 C
    belt_creep_risk = (belt.core == "aramid" and belt.pretension_n > 25.0) or (
        belt.body_material == "tpu"
        and spec.frame.chamber_temp_c > 55.0
        and belt.pretension_n > 30.0
    )
    pulley_circ_mm = float(belt.pulley_teeth) * float(belt.pitch_mm)
    m_x_eff = max(assembly.x_moving_mass_kg, 0.05)
    m_y_eff = max(assembly.y_moving_mass_kg, 0.05)
    k_drive_y_n_m = (
        k_drive_n_m
        if spec.architecture != "cartesian_bedslinger"
        else float(compute_drive_stiffness_details_n_per_m(assembly, "Y")["k_drive_n_m"])
    )
    f_drive_x = math.sqrt(max(k_drive_n_m, 1.0) / m_x_eff) / (2.0 * math.pi)
    f_drive_y = math.sqrt(max(k_drive_y_n_m, 1.0) / m_y_eff) / (2.0 * math.pi)
    f_drive_ref = min(f_drive_x, f_drive_y)
    v_vfa_center = min(max(float(belt.pitch_mm) * f_drive_ref, 35.0), 304.0)
    vfa_range = [
        round(max(30.0, 0.85 * v_vfa_center), 1),
        round(min(350.0, 1.15 * v_vfa_center), 1),
    ]

    override = spec.gantry.custom_bracket_override
    return KinematicsReport(
        architecture=spec.architecture,
        build_volume_liters=build_l,
        frame_volume_liters=frame_l,
        packaging_efficiency_pct=pkg_eff,
        max_motor_rpm=max_rpm,
        available_torque_at_vmax_nm=t_avail,
        required_torque_x_nm=t_req_x,
        required_torque_y_nm=t_req_y,
        torque_safety_margin=margin,
        max_safe_accel_motor_mm_s2=max_safe_a,
        step_skip_risk=margin < 1.15,
        belt_peak_tension_n=peak_tension,
        belt_slip_risk=belt_slip,
        delta_workspace_valid=delta_ok,
        drive_mode=drive_mode,
        belt_routing=spec.gantry.belt_routing,
        xy_motor_count=xy_motor_count,
        belt_pluck_freq_150mm_hz=belt_pluck_150,
        belt_max_working_tension_n=belt_max_t,
        belt_overtension_risk=belt_overtension,
        steel_core_fatigue_risk=steel_core_fatigue,
        belt_bend_fatigue_risk=belt_bend_fatigue,
        belt_creep_risk=belt_creep_risk,
        belt_min_pulley_teeth=min_pulley_t,
        belt_min_back_idler_mm=min_back_idler,
        belt_back_idler_mm=eff_back_idler,
        belt_body_material=belt.body_material,
        belt_thermal_retention_pct=belt_thermal_ret_pct,
        belt_label=belt.label,
        belt_validation_tier=belt.validation_tier,
        belt_theoretical_summary=belt.theoretical_summary,
        belt_practical_issues=belt.practical_issues,
        pulley_circumference_mm=pulley_circ_mm,
        vfa_belt_tooth_speed_range_mm_s=vfa_range,
        shaft_bending_risk=shaft_bending,
        frame_belt_preload_n=frame_preload,
        motor_run_current_a=stepper.run_current_a,
        motor_stiffness_n_per_mm=k_motor_n_m * 1e-3,
        belt_stiffness_n_per_mm=k_belt_n_m * 1e-3,
        bracket_material=bkt_id,
        bracket_stiffness_n_per_mm=k_bracket_n_m * 1e-3,
        bracket_creep_risk=bracket_creep,
        drive_stiffness_n_per_mm=k_drive_n_m * 1e-3,
        motor_voltage_efficiency_pct=motor_volt_eff_pct,
        motor_heat_w_each=motor_heat_w,
        motor_overcurrent_risk=motor_overcurrent,
        tooth_mesh_stiffness_n_per_mm=k_tooth_n_m * 1e-3,
        idler_shaft_stiffness_n_per_mm=k_idler_n_m * 1e-3,
        idler_shaft_deflection_um=idler_info["idler_shaft_deflection_um"],
        bearing_play_um=idler_info["bearing_play_um"],
        bracket_modulus_retention_pct=bracket_ret_pct,
        custom_bracket_active=override is not None,
        custom_bracket_label=override.label if override is not None else "",
    )
