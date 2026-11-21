"""Kinematics (CoreXY, Cartesian, Delta), motor torque curves, and belt dynamics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from printsim.geometry.builder import AssemblyModel
from printsim.schema import GANTRY_BRACKET_CATALOG, PrinterSpec, StepperSpec, get_material


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
            "xy_motor_count": self.xy_motor_count,
            "belt_pluck_freq_150mm_hz": round(self.belt_pluck_freq_150mm_hz, 2),
            "belt_max_working_tension_n": round(self.belt_max_working_tension_n, 2),
            "belt_overtension_risk": self.belt_overtension_risk,
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
    """Compute effective gantry bracket / motor-mount / XY-joint stiffness in series (N/m)."""
    spec = assembly.spec
    e_bkt = get_material(spec.gantry.bracket_material).youngs_modulus_pa
    k_ref = 2.6e5  # Base 2WD single-shear ABS bracket stiffness (260 N/mm) at 2.1 GPa
    ds_mult = 1.45 if spec.gantry.double_shear_pulleys else 1.0
    awd_mult = 1.65 if (spec.architecture == "corexy" and spec.gantry.drive_mode == "awd") else 1.0
    k_mat = k_ref * ds_mult * awd_mult * (e_bkt / 2.1e9)
    k_ceil = 8.0e6  # Bolted fastener / bearing compliance ceiling (8,000 N/mm)
    return (k_mat * k_ceil) / max(k_mat + k_ceil, 1e-9)


def compute_drive_stiffness_n_per_m(
    assembly: AssemblyModel, axis: str = "X"
) -> tuple[float, float, float]:
    """Compute (k_belt_n_m, k_motor_axis_n_m, k_drive_n_m) coupling belt, motor EM, & bracket springs in series."""
    spec = assembly.spec
    belt = spec.gantry.belt
    kin = spec.kinematics
    w = spec.frame.width_mm * 1e-3
    d = spec.frame.depth_mm * 1e-3

    if spec.architecture == "corexy":
        if spec.gantry.drive_mode == "awd":
            # 4-motor AWD halves active belt span driving the toolhead with 4 parallel driven branches
            span_len = max(0.52 * (w + d), 0.22)
            k_belt = 4.0 * belt.stiffness_ea_n / span_len
        else:
            # CoreXY belt loop spans roughly 2*W + 2*D; two parallel spans act on X or Y
            span_len = max(w + d, 0.35)
            k_belt = 2.0 * belt.stiffness_ea_n / span_len
    elif spec.architecture == "delta":
        span_len = max(spec.frame.height_mm * 1e-3, 0.45)
        k_belt = 2.5 * belt.stiffness_ea_n / span_len
    else:
        span_len = w if axis.upper() == "X" else d
        k_belt = 2.0 * belt.stiffness_ea_n / max(span_len, 0.25)

    # Belt pretension stiffness coupling (Gates belt backlash/slack compliance factor)
    tension_eff = min(1.08, 0.72 + 0.36 * min(belt.pretension_n / 45.0, 1.0))
    k_belt *= tension_eff

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
    k_bracket = compute_bracket_stiffness_n_per_m(assembly)
    k_drive = 1.0 / (
        1.0 / max(k_belt, 1e-9)
        + 1.0 / max(k_motor_axis, 1e-9)
        + 1.0 / max(k_bracket, 1e-9)
    )
    return (k_belt, k_motor_axis, k_drive)


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
    alpha_max = a_max_m_s2 / r_p

    if spec.architecture == "corexy":
        # 2WD shares pure X/Y load across 2 motors (0.5 each); AWD shares across 4 motors (0.25 each)
        f_x = assembly.x_moving_mass_kg * a_max_m_s2 + f_fric
        f_y = assembly.y_moving_mass_kg * a_max_m_s2 + f_fric
        share = 0.25 if is_awd else 0.5
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

    # Belt dynamic tension: tight side = T0 + F/2, slack side = T0 - F/2 (slip if slack <= 0)
    peak_tension = belt.pretension_n + 0.5 * peak_belt_force
    slack_tension = belt.pretension_n - 0.5 * peak_belt_force
    belt_slip = slack_tension < 2.0

    belt_pluck_150 = belt.pluck_freq_hz(150.0)
    belt_max_t = belt.max_working_tension_n
    belt_overtension = peak_tension > belt_max_t
    shaft_bending = (belt.pretension_n > 45.0) and (not spec.gantry.double_shear_pulleys)
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

    stepper = kin.xy_stepper
    k_belt_n_m, k_motor_n_m, k_drive_n_m = compute_drive_stiffness_n_per_m(assembly, "X")
    bkt_id = spec.gantry.bracket_material
    bkt_info = GANTRY_BRACKET_CATALOG.get(bkt_id, GANTRY_BRACKET_CATALOG["abs_printed"])
    k_bracket_n_m = compute_bracket_stiffness_n_per_m(assembly)
    bracket_creep = belt.pretension_n > float(bkt_info["creep_tension_limit_n"])
    motor_volt_eff_pct = stepper.dynamic_voltage_efficiency(max_rpm * 0.7) * 100.0
    motor_heat_w = stepper.power_dissipation_w
    motor_overcurrent = stepper.run_current_a > stepper.rated_current_a * 1.05

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
        xy_motor_count=xy_motor_count,
        belt_pluck_freq_150mm_hz=belt_pluck_150,
        belt_max_working_tension_n=belt_max_t,
        belt_overtension_risk=belt_overtension,
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
    )
