"""Kinematics (CoreXY, Cartesian, Delta), motor torque curves, and belt dynamics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from printsim.geometry.builder import AssemblyModel
from printsim.schema import PrinterSpec, StepperSpec


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
    """Electromagnetic pull-out torque model including L/R corner speed roll-off."""
    # Dynamic holding-to-pull-out factor (~0.72 at low speed)
    t0 = stepper.holding_torque_nm * 0.72
    # Corner frequency where inductive reactance equals supply voltage drive headroom:
    # f_corner_rps = V / (2 * pi * L * I * steps_per_rev/4)
    l_h = stepper.inductance_mh * 1e-3
    pole_pairs = stepper.steps_per_rev / 4.0
    omega_corner_rad_s = stepper.supply_voltage_v / max(pole_pairs * l_h * stepper.rated_current_a, 1e-6)
    omega_rad_s = max(rpm, 0.0) * (2.0 * math.pi / 60.0)
    rolloff = 1.0 / math.sqrt(1.0 + (omega_rad_s / max(omega_corner_rad_s, 1.0)) ** 2)
    return t0 * rolloff


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

    # Linear friction estimate (linear rails + belt idlers ~ 2.5 N)
    f_fric = 2.5
    j_rot = kin.xy_stepper.rotor_inertia_kg_m2
    alpha_max = a_max_m_s2 / r_p

    if spec.architecture == "corexy":
        # Two motors share pure X or pure Y force (1/2 each), plus rotor inertia
        f_x = assembly.x_moving_mass_kg * a_max_m_s2 + f_fric
        f_y = assembly.y_moving_mass_kg * a_max_m_s2 + f_fric
        t_req_x = (0.5 * f_x) * r_p + j_rot * alpha_max
        t_req_y = (0.5 * f_y) * r_p + j_rot * alpha_max
        peak_belt_force = max(f_x, f_y) * 0.5
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

    # Validate Delta reachability at perimeter corners if Delta
    delta_ok = True
    if spec.architecture == "delta":
        for px in (0.0, bx):
            for py in (0.0, by):
                try:
                    inverse_kinematics(spec, (px, py, 0.0))
                except ValueError:
                    delta_ok = False

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
    )
