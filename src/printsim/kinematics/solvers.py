"""Kinematics (CoreXY, Cartesian, Delta), motor torque curves, and belt dynamics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from printsim.geometry.builder import AssemblyModel
from printsim.schema import (
    PrinterSpec,
    StepperSpec,
    get_material,
    resolve_bearing_clearance,
    resolve_belt,
    resolve_belt_routing,
    resolve_gantry_bracket,
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
    motor_est_temp_c: float = 50.0
    motor_max_winding_temp_c: float = 130.0
    motor_thermal_risk: bool = False
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
    motor_max_accel_x_mm_s2: float = 0.0
    motor_max_accel_y_mm_s2: float = 0.0
    motor_max_accel_mm_s2: float = 0.0
    belt_max_accel_mm_s2: float = 0.0
    motor_max_velocity_mm_s: float = 0.0
    shaper_l1_accel_factor: float = 1.0

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
            "motor_est_temp_c": round(self.motor_est_temp_c, 1),
            "motor_max_winding_temp_c": round(self.motor_max_winding_temp_c, 0),
            "motor_thermal_risk": self.motor_thermal_risk,
            "motor_overcurrent_risk": self.motor_overcurrent_risk,
            "tooth_mesh_stiffness_n_per_mm": round(self.tooth_mesh_stiffness_n_per_mm, 1),
            "idler_shaft_stiffness_n_per_mm": round(self.idler_shaft_stiffness_n_per_mm, 1),
            "idler_shaft_deflection_um": round(self.idler_shaft_deflection_um, 2),
            "bearing_play_um": round(self.bearing_play_um, 1),
            "bracket_modulus_retention_pct": round(self.bracket_modulus_retention_pct, 1),
            "custom_bracket_active": bool(self.custom_bracket_active),
            "custom_bracket_label": self.custom_bracket_label,
            "motor_max_accel_x_mm_s2": round(self.motor_max_accel_x_mm_s2, 0),
            "motor_max_accel_y_mm_s2": round(self.motor_max_accel_y_mm_s2, 0),
            "motor_max_accel_mm_s2": round(self.motor_max_accel_mm_s2, 0),
            "belt_max_accel_mm_s2": round(self.belt_max_accel_mm_s2, 0),
            "motor_max_velocity_mm_s": round(self.motor_max_velocity_mm_s, 0),
            "shaper_l1_accel_factor": round(self.shaper_l1_accel_factor, 3),
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


def _delta_jacobian(spec: PrinterSpec, cx_m: float, cy_m: float) -> np.ndarray:
    """Exact 3x3 analytic Delta Jacobian J = d(z_0, z_1, z_2) / d(cx, cy, z) in SI units."""
    arm_m = max(spec.kinematics.delta_arm_length_mm * 1e-3, 1e-4)
    towers_m = _delta_towers_xy(spec) * 1e-3
    rows = []
    for tx, ty in towers_m:
        dx = cx_m - float(tx)
        dy = cy_m - float(ty)
        h_i = math.sqrt(max(arm_m * arm_m - dx * dx - dy * dy, 1e-6))
        rows.append([-dx / h_i, -dy / h_i, 1.0])
    return np.asarray(rows, dtype=float)


def _delta_center_j_norm(spec: PrinterSpec) -> float:
    """Center Jacobian projection factor mean(diag(J_xy^T J_xy)) = 1.5 * (R / h_0)^2."""
    j_xy = _delta_jacobian(spec, 0.0, 0.0)[:, :2]
    return float(np.mean(np.diag(j_xy.T @ j_xy)))


def _delta_workspace_envelope(
    spec: PrinterSpec,
    m_e_kg: float,
    m_c_kg: float,
    r_p_m: float,
    j_rot_kg_m2: float,
    f_fric_car_n: float,
) -> dict[str, object]:
    """Sweep inscribed circular Delta build disk (center, half-radius, perimeter) x 24 directions.

    Returns exact speed ratio max|J*u| and helpers for worst-case motor torque, belt force,
    and analytical closed-form max acceleration across all (point, direction, tower) triples.
    """
    bx_m, by_m, _ = [v * 1e-3 for v in spec.kinematics.build_volume_mm]
    r_bed_m = 0.5 * min(bx_m, by_m)
    g_acc = 9.80665
    g_vec = np.array([0.0, 0.0, g_acc], dtype=float)

    pts_xy: list[tuple[float, float]] = [(0.0, 0.0)]
    for rho in (0.5 * r_bed_m, r_bed_m):
        for k in range(24):
            ang = 2.0 * math.pi * k / 24.0
            pts_xy.append((rho * math.cos(ang), rho * math.sin(ang)))

    # 24 evenly spaced directions include exact pure-X (0, 180 deg) and pure-Y (90, 270 deg)
    u_vecs = np.array(
        [
            [math.cos(2.0 * math.pi * k / 24.0), math.sin(2.0 * math.pi * k / 24.0), 0.0]
            for k in range(24)
        ],
        dtype=float,
    )
    x_dir_mask = np.array([k in (0, 12) for k in range(24)], dtype=bool)
    y_dir_mask = np.array([k in (6, 18) for k in range(24)], dtype=bool)

    n_pts = len(pts_xy)
    n_dirs = len(u_vecs)
    tau_grav = np.zeros((n_pts, 3), dtype=float)
    f_grav = np.zeros((n_pts, 3), dtype=float)
    s_tau = np.zeros((n_pts, n_dirs, 3), dtype=float)
    s_belt = np.zeros((n_pts, n_dirs, 3), dtype=float)
    max_speed_ratio = 0.0

    for idx_p, (cx_m, cy_m) in enumerate(pts_xy):
        j_mat = _delta_jacobian(spec, cx_m, cy_m)
        jt_inv = np.linalg.inv(j_mat.T)
        f_g = jt_inv @ (m_e_kg * g_vec) + m_c_kg * g_acc
        f_grav[idx_p] = f_g
        tau_grav[idx_p] = r_p_m * f_g

        # Vectorized over all 24 directions (3 x 24 -> transpose to 24 x 3)
        j_u = (j_mat @ u_vecs.T).T
        max_speed_ratio = max(max_speed_ratio, float(np.max(np.abs(j_u))))
        f_eff_u = (jt_inv @ (m_e_kg * u_vecs.T)).T
        s_b = f_eff_u + m_c_kg * j_u
        s_belt[idx_p] = s_b
        s_tau[idx_p] = r_p_m * s_b + (j_rot_kg_m2 / max(r_p_m, 1e-6)) * j_u

    tau_fric = r_p_m * f_fric_car_n

    def _mask_slice(dir_filter: str | None) -> slice | np.ndarray:
        if dir_filter == "x":
            return x_dir_mask
        if dir_filter == "y":
            return y_dir_mask
        return slice(None)

    def max_torque_nm(a_m_s2: float, dir_filter: str | None = None) -> float:
        d_sel = _mask_slice(dir_filter)
        tau_dyn = tau_grav[:, None, :] + a_m_s2 * s_tau[:, d_sel, :]
        return float(np.max(np.abs(tau_dyn) + tau_fric))

    def max_belt_force_n(a_m_s2: float) -> float:
        f_dyn = f_grav[:, None, :] + a_m_s2 * s_belt
        return float(np.max(np.abs(f_dyn) + f_fric_car_n))

    def _solve_max_a(base_2d: np.ndarray, slope_3d: np.ndarray, net_budget: float) -> float:
        if net_budget <= 0.0:
            return 0.0
        base = base_2d[:, None, :]
        if float(np.max(np.abs(base))) >= net_budget:
            return 0.0
        # For each (pt, dir, tower): |base + a * slope| <= net_budget with a >= 0
        pos = slope_3d > 1e-12
        neg = slope_3d < -1e-12
        a_lim = math.inf
        if np.any(pos):
            a_pos = np.min((net_budget - np.broadcast_to(base, slope_3d.shape)[pos]) / slope_3d[pos])
            a_lim = min(a_lim, float(a_pos))
        if np.any(neg):
            a_neg = np.min((-net_budget - np.broadcast_to(base, slope_3d.shape)[neg]) / slope_3d[neg])
            a_lim = min(a_lim, float(a_neg))
        return max(0.0, a_lim)

    def max_accel_torque_m_s2(t_budget_nm: float, dir_filter: str | None = None) -> float:
        d_sel = _mask_slice(dir_filter)
        return _solve_max_a(tau_grav, s_tau[:, d_sel, :], t_budget_nm - tau_fric)

    def max_accel_belt_m_s2(f_budget_n: float) -> float:
        return _solve_max_a(f_grav, s_belt, f_budget_n - f_fric_car_n)

    return {
        "max_speed_ratio": max(max_speed_ratio, 1e-6),
        "max_torque_nm": max_torque_nm,
        "max_belt_force_n": max_belt_force_n,
        "max_accel_torque_m_s2": max_accel_torque_m_s2,
        "max_accel_belt_m_s2": max_accel_belt_m_s2,
    }


def _solve_worst_dir_corexy_accel(
    m_x: float,
    m_y: float,
    fx_fric: float,
    fy_fric: float,
    force_budget_n: float,
) -> float:
    """Solve sqrt((m_x*a + fx_fric)^2 + (m_y*a + fy_fric)^2) <= force_budget_n for a >= 0."""
    if force_budget_n <= 0.0:
        return 0.0
    a_q = m_x * m_x + m_y * m_y
    b_q = 2.0 * (m_x * fx_fric + m_y * fy_fric)
    c_q = fx_fric * fx_fric + fy_fric * fy_fric - force_budget_n * force_budget_n
    if c_q >= 0.0:
        return 0.0
    if a_q <= 1e-12:
        return 0.0
    return (-b_q + math.sqrt(max(b_q * b_q - 4.0 * a_q * c_q, 0.0))) / (2.0 * a_q)


def stepper_available_torque_nm(
    stepper: StepperSpec,
    rpm: float,
    drive_type: str = "open_loop_stepper",
) -> float:
    """Electromagnetic pull-out torque model using drive utilization & piecewise phasor voltage limit.

    Evaluates iron saturation at the actual voltage-limited quadrature RMS current I_q / sqrt(2),
    so high-speed torque remains monotonically non-decreasing even if run_current_a > rated_current_a.
    """
    # Utilization & dual-H-bridge bus efficiency (2-phase steppers have no 3-phase SVPWM boost)
    if drive_type == "foc_servo":
        util, bus_util = 0.95, 0.98
    elif drive_type == "closed_loop_stepper":
        util, bus_util = 0.85, 0.92
    else:
        util, bus_util = 0.72, 0.90
    eta_pullout = stepper.pullout_torque_efficiency(rpm, bus_utilization=bus_util)
    i_q_rms = stepper.run_current_a * eta_pullout
    return stepper.torque_at_rms_current_nm(i_q_rms) * util


def _compute_motor_max_velocity_mm_s(
    stepper: StepperSpec,
    r_p: float,
    drive_type: str,
    sf: float,
    vel_scale: float,
    t_req_reserve: float,
) -> float:
    """Solve torque-balance max Cartesian velocity (mm/s) along the piecewise pull-out curve."""

    def _feasible(v_m_s: float) -> bool:
        omega_m = (v_m_s * vel_scale) / max(r_p, 1e-6)
        rpm = (omega_m * 60.0) / (2.0 * math.pi)
        t_avail_v = stepper_available_torque_nm(stepper, rpm, drive_type=drive_type)
        return (t_avail_v / sf) >= t_req_reserve

    if not _feasible(0.02):
        return 0.0

    lo, hi = 0.02, 5.0
    while _feasible(hi) and hi < 25.0:
        lo = hi
        hi *= 2.0
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        if _feasible(mid):
            lo = mid
        else:
            hi = mid
    return lo * 1000.0


def compute_bracket_stiffness_n_per_m(assembly: AssemblyModel) -> float:
    """Compute effective gantry bracket / motor-mount stiffness in series (N/m), scaled by E(T)."""
    return float(resolve_gantry_bracket(assembly.spec)["stiffness_n_m"])


def compute_drive_stiffness_details_n_per_m(
    assembly: AssemblyModel,
    axis: str = "X",
    u_rel: float = 0.5,
    v_rel: float = 0.5,
) -> dict[str, object]:
    """Compute detailed series drive-train stiffness components using exact pulley-to-pulley free spans."""
    from printsim.fea.components import (
        belt_effective_ea_n,
        compute_bearing_cte_clearance_um,
        compute_stepper_mount_temp_c,
        gerbert_tooth_mesh_stiffness_n_m,
        idler_shaft_and_bearing_stiffness,
    )
    from printsim.schema import get_profile

    spec = assembly.spec
    belt = spec.gantry.belt
    kin = spec.kinematics
    w = spec.frame.width_mm * 1e-3
    d = spec.frame.depth_mm * 1e-3
    bx, by, _ = [v * 1e-3 for v in kin.build_volume_mm]
    b_ext = get_profile(spec.frame.profile).width_m
    # Exact inner pulley-to-pulley free-span envelope (excludes frame extrusion width + 25 mm corner mount inset per side)
    wp = max(w - 2.0 * (b_ext + 0.025), bx + 0.04)
    dp = max(d - 2.0 * (b_ext + 0.025), by + 0.04)
    u = float(min(max(u_rel, 0.05), 0.95))
    v = float(min(max(v_rel, 0.05), 0.95))
    t_ch = float(spec.frame.chamber_temp_c)
    motor_mount_temp_c, _ = compute_stepper_mount_temp_c(spec)
    bkt_mat = get_material(spec.gantry.bracket_material)
    e_bkt_eff = bkt_mat.effective_modulus_pa(motor_mount_temp_c)

    belt_cfg = resolve_belt(spec)
    belt_ea_n = float(belt_cfg["stiffness_ea_n"])
    belt_tooth_k = float(belt_cfg["tooth_shear_k_n_per_m_mm"])
    belt_mu_kg_m = float(belt_cfg["linear_mass_kg_m"])

    routing_cfg = resolve_belt_routing(spec)
    span_factor = float(routing_cfg["span_factor"])
    idler_factor = float(routing_cfg["idler_stiffness_factor"])
    idler_stiffness_mult = idler_factor if spec.architecture == "corexy" else 1.0
    bc_cfg = resolve_bearing_clearance(spec)
    cte_clearance_um, _ = compute_bearing_cte_clearance_um(spec, t_ch)
    idler_info = idler_shaft_and_bearing_stiffness(
        belt,
        double_shear=spec.gantry.double_shear_pulleys,
        bracket_e_pa=e_bkt_eff,
        radial_play_um_override=float(bc_cfg["radial_play_um"]),
        thermal_clearance_um=cte_clearance_um,
    )
    k_idler_single = float(idler_info["k_idler_n_m"]) * idler_stiffness_mult
    if idler_stiffness_mult != 1.0:
        idler_info = dict(idler_info)
        idler_info["idler_shaft_deflection_um"] = (
            idler_info["idler_shaft_deflection_um"] / idler_stiffness_mult
        )

    if spec.architecture == "corexy":
        if spec.gantry.drive_mode == "awd":
            # 4-motor AWD: exact toolhead-to-motor free spans at (u, v) on Belt A and Belt B
            # Each carriage-to-motor branch wraps ~2 pulleys/idlers (4 tangent points)
            n_tan_awd = 4.0
            l_a1 = max(((1.0 - u) * wp + (1.0 - v) * dp) * span_factor, 0.12)
            l_a2 = max((u * wp + v * dp) * span_factor, 0.12)
            l_b1 = max((u * wp + (1.0 - v) * dp) * span_factor, 0.12)
            l_b2 = max(((1.0 - u) * wp + v * dp) * span_factor, 0.12)
            ea_a1 = belt_effective_ea_n(
                belt, l_a1, ea_override_n=belt_ea_n, linear_mass_override_kg_m=belt_mu_kg_m,
                n_pulley_tangents=n_tan_awd, chamber_temp_c=t_ch,
            )
            ea_a2 = belt_effective_ea_n(
                belt, l_a2, ea_override_n=belt_ea_n, linear_mass_override_kg_m=belt_mu_kg_m,
                n_pulley_tangents=n_tan_awd, chamber_temp_c=t_ch,
            )
            ea_b1 = belt_effective_ea_n(
                belt, l_b1, ea_override_n=belt_ea_n, linear_mass_override_kg_m=belt_mu_kg_m,
                n_pulley_tangents=n_tan_awd, chamber_temp_c=t_ch,
            )
            ea_b2 = belt_effective_ea_n(
                belt, l_b2, ea_override_n=belt_ea_n, linear_mass_override_kg_m=belt_mu_kg_m,
                n_pulley_tangents=n_tan_awd, chamber_temp_c=t_ch,
            )
            kc_a1, kc_a2 = ea_a1 / l_a1, ea_a2 / l_a2
            kc_b1, kc_b2 = ea_b1 / l_b1, ea_b2 / l_b2
            k_belt_cord = (kc_a1 + kc_a2) + (kc_b1 + kc_b2)
            ea_eff = 0.25 * (ea_a1 + ea_a2 + ea_b1 + ea_b2)
            drive_branches = 4.0
            k_mesh_1 = gerbert_tooth_mesh_stiffness_n_m(
                belt, ea_eff, t_ch, tooth_shear_override=belt_tooth_k
            )
            kb_a1 = 1.0 / (1.0 / kc_a1 + 1.0 / k_mesh_1 + 1.0 / k_idler_single)
            kb_a2 = 1.0 / (1.0 / kc_a2 + 1.0 / k_mesh_1 + 1.0 / k_idler_single)
            kb_b1 = 1.0 / (1.0 / kc_b1 + 1.0 / k_mesh_1 + 1.0 / k_idler_single)
            kb_b2 = 1.0 / (1.0 / kc_b2 + 1.0 / k_mesh_1 + 1.0 / k_idler_single)
            k_a_loop = kb_a1 + kb_a2
            k_b_loop = kb_b1 + kb_b2
            branch_stiffnesses = (kb_a1, kb_a2, kb_b1, kb_b2)
            # Rayleigh moving-belt inertia across the 4 carriage-to-motor spans: (1/3)*mu*sum(L)
            belt_moving_mass_kg = (1.0 / 3.0) * belt_mu_kg_m * (l_a1 + l_a2 + l_b1 + l_b2)
            longest_free_span_m = max(wp, dp) * span_factor
            # Equal carriage-to-motor branches at center -> 50/50 tight/slack split
            slack_load_share = max(
                max(kc_a1, kc_a2) / max(kc_a1 + kc_a2, 1e-9),
                max(kc_b1, kc_b2) / max(kc_b1 + kc_b2, 1e-9),
            )
        else:
            # 2WD CoreXY: exact short motor branch (~4 tangents) & long 3-corner return branch (~10 tangents)
            n_tan_short = 4.0
            n_tan_long = 10.0
            l_a1 = max(((1.0 - u) * wp + (1.0 - v) * dp) * span_factor, 0.14)
            l_a2 = max((u * wp + v * dp + wp + dp) * span_factor, 0.40)
            l_b1 = max((u * wp + (1.0 - v) * dp) * span_factor, 0.14)
            l_b2 = max(((1.0 - u) * wp + v * dp + wp + dp) * span_factor, 0.40)
            ea_a1 = belt_effective_ea_n(
                belt, l_a1, ea_override_n=belt_ea_n, linear_mass_override_kg_m=belt_mu_kg_m,
                n_pulley_tangents=n_tan_short, chamber_temp_c=t_ch,
            )
            ea_a2 = belt_effective_ea_n(
                belt, l_a2, ea_override_n=belt_ea_n, linear_mass_override_kg_m=belt_mu_kg_m,
                n_pulley_tangents=n_tan_long, chamber_temp_c=t_ch,
            )
            ea_b1 = belt_effective_ea_n(
                belt, l_b1, ea_override_n=belt_ea_n, linear_mass_override_kg_m=belt_mu_kg_m,
                n_pulley_tangents=n_tan_short, chamber_temp_c=t_ch,
            )
            ea_b2 = belt_effective_ea_n(
                belt, l_b2, ea_override_n=belt_ea_n, linear_mass_override_kg_m=belt_mu_kg_m,
                n_pulley_tangents=n_tan_long, chamber_temp_c=t_ch,
            )
            kc_a1, kc_a2 = ea_a1 / l_a1, ea_a2 / l_a2
            kc_b1, kc_b2 = ea_b1 / l_b1, ea_b2 / l_b2
            k_belt_cord = (kc_a1 + kc_a2) + (kc_b1 + kc_b2)
            ea_eff = 0.25 * (ea_a1 + ea_a2 + ea_b1 + ea_b2)
            drive_branches = 2.0
            k_mesh_a = gerbert_tooth_mesh_stiffness_n_m(
                belt, 0.5 * (ea_a1 + ea_a2), t_ch, tooth_shear_override=belt_tooth_k
            )
            k_mesh_b = gerbert_tooth_mesh_stiffness_n_m(
                belt, 0.5 * (ea_b1 + ea_b2), t_ch, tooth_shear_override=belt_tooth_k
            )
            k_a_loop = 1.0 / (1.0 / max(kc_a1 + kc_a2, 1e-9) + 1.0 / k_mesh_a + 1.0 / k_idler_single)
            k_b_loop = 1.0 / (1.0 / max(kc_b1 + kc_b2, 1e-9) + 1.0 / k_mesh_b + 1.0 / k_idler_single)
            branch_stiffnesses = (k_a_loop, k_b_loop)
            belt_moving_mass_kg = (1.0 / 3.0) * belt_mu_kg_m * (l_a1 + l_a2 + l_b1 + l_b2)
            longest_free_span_m = max(wp, dp) * span_factor
            # Compatibility split on unequal 2WD branches: short stiff span carries k1/(k1+k2) (~0.72-0.75)
            slack_load_share = max(
                max(kc_a1, kc_a2) / max(kc_a1 + kc_a2, 1e-9),
                max(kc_b1, kc_b2) / max(kc_b1 + kc_b2, 1e-9),
            )
    elif spec.architecture == "delta":
        # Project per-tower parallel upper/lower belt loop stiffness onto Cartesian XY via J_xy^T J_xy.
        # Active pulley-to-pulley span excludes top/bottom housing clearance (~25% of outer height;
        # FLSUN V400: 920 mm frame -> ~690 mm center distance, ~1380-1458 mm single-belt loop).
        j_norm = _delta_center_j_norm(spec)
        span_len = max(0.75 * spec.frame.height_mm * 1e-3, 0.38)
        span_short = 0.5 * span_len
        span_long = 1.5 * span_len
        ea_short = belt_effective_ea_n(
            belt, span_short, ea_override_n=belt_ea_n, linear_mass_override_kg_m=belt_mu_kg_m,
            n_pulley_tangents=2.0, chamber_temp_c=t_ch,
        )
        ea_long = belt_effective_ea_n(
            belt, span_long, ea_override_n=belt_ea_n, linear_mass_override_kg_m=belt_mu_kg_m,
            n_pulley_tangents=2.0, chamber_temp_c=t_ch,
        )
        kc_s, kc_l = ea_short / span_short, ea_long / span_long
        k_belt_cord = j_norm * (kc_s + kc_l)
        ea_eff = 0.5 * (ea_short + ea_long)
        drive_branches = j_norm
        k_a_loop = 0.0
        k_b_loop = 0.0
        branch_stiffnesses = ()
        belt_moving_mass_kg = j_norm * (1.0 / 3.0) * belt_mu_kg_m * (span_short + span_long)
        longest_free_span_m = span_len
        slack_load_share = max(kc_s, kc_l) / max(kc_s + kc_l, 1e-9)
    else:
        pos_rel = u if axis.upper() == "X" else v
        span_total = max(wp if axis.upper() == "X" else dp, 0.22)
        l1 = max(pos_rel * span_total, 0.05)
        l2 = max((1.0 - pos_rel) * span_total, 0.05)
        ea1 = belt_effective_ea_n(
            belt, l1, ea_override_n=belt_ea_n, linear_mass_override_kg_m=belt_mu_kg_m,
            n_pulley_tangents=2.0, chamber_temp_c=t_ch,
        )
        ea2 = belt_effective_ea_n(
            belt, l2, ea_override_n=belt_ea_n, linear_mass_override_kg_m=belt_mu_kg_m,
            n_pulley_tangents=2.0, chamber_temp_c=t_ch,
        )
        kc1, kc2 = ea1 / l1, ea2 / l2
        k_belt_cord = kc1 + kc2
        ea_eff = 0.5 * (ea1 + ea2)
        drive_branches = 1.0
        k_a_loop = 0.0
        k_b_loop = 0.0
        branch_stiffnesses = ()
        belt_moving_mass_kg = (1.0 / 3.0) * belt_mu_kg_m * (l1 + l2)
        longest_free_span_m = span_total
        slack_load_share = max(kc1, kc2) / max(kc1 + kc2, 1e-9)

    k_tooth_mesh = drive_branches * gerbert_tooth_mesh_stiffness_n_m(
        belt, ea_eff, t_ch, tooth_shear_override=belt_tooth_k
    )
    k_idler_shaft = drive_branches * k_idler_single

    k_belt_subsystem = 1.0 / (
        1.0 / max(k_belt_cord, 1e-9)
        + 1.0 / max(k_tooth_mesh, 1e-9)
        + 1.0 / max(k_idler_shaft, 1e-9)
    )
    if spec.architecture != "corexy":
        k_a_loop = k_belt_subsystem
        k_b_loop = k_belt_subsystem
        branch_stiffnesses = (k_belt_subsystem,)

    r_p = belt.pitch_radius_m
    if spec.architecture == "corexy":
        vel_scale = 1.4142
    elif spec.architecture == "delta":
        # Drive stiffness is evaluated at bed center (where ADXL resonance testing occurs):
        # center carriage-to-effector velocity ratio is R / h0 (not extreme perimeter 2.5x).
        j_xy_center = _delta_jacobian(spec, 0.0, 0.0)[:, :2]
        vel_scale = max(float(np.max(np.linalg.norm(j_xy_center, axis=1))), 1e-6)
    else:
        vel_scale = 1.0
    motor_lin_v = (kin.max_velocity_mm_s * 1e-3) * vel_scale
    max_rpm = (motor_lin_v / (2.0 * math.pi * r_p)) * 60.0
    rpm_op = max(120.0, 0.35 * max_rpm)
    k_theta = kin.xy_stepper.torsional_stiffness_nm_per_rad(rpm=rpm_op)
    k_motor_single = k_theta / (r_p**2)

    if spec.architecture == "corexy":
        motor_factor = 4.0 if spec.gantry.drive_mode == "awd" else 2.0
    elif spec.architecture == "delta":
        motor_factor = j_norm
    else:
        motor_factor = 1.0

    k_motor_axis = motor_factor * k_motor_single
    override = spec.gantry.custom_bracket_override
    k_bracket = compute_bracket_stiffness_n_per_m(assembly)
    if override is not None or spec.model_overrides.gantry_bracket.stiffness_n_per_mm is not None:
        bracket_retention_pct = 100.0
    else:
        bracket_retention_pct = bkt_mat.modulus_retention_at_temp(motor_mount_temp_c) * 100.0
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
        "k_A_n_m": k_a_loop,
        "k_B_n_m": k_b_loop,
        "branch_stiffnesses_n_m": branch_stiffnesses,
        "belt_moving_mass_kg": belt_moving_mass_kg,
        "longest_free_span_m": longest_free_span_m,
        "slack_load_share": slack_load_share,
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

    is_awd = spec.architecture == "corexy" and spec.gantry.drive_mode == "awd"
    drive_mode = "awd" if is_awd else "2wd"
    xy_motor_count = 4 if is_awd else (3 if spec.architecture == "delta" else 2)

    # Linear friction estimate (extra AWD idlers/motors add modest drag: 3.6 N vs 2.5 N).
    # Belt/idler rolling hysteresis (~60% of gantry drag) scales by belt.rolling_drag_mult
    # (1.0 for standard 2GT EPDM, 1.18 for softer-elastomer GT3 2MGT).
    f_fric_base = 3.6 if is_awd else 2.5
    drag_mult = max(float(belt.rolling_drag_mult), 0.5)
    f_fric = f_fric_base * (0.40 + 0.60 * drag_mult)
    j_rot = kin.xy_stepper.rotor_inertia_kg_m2
    m_x = assembly.x_moving_mass_kg
    m_y = assembly.y_moving_mass_kg

    delta_env: dict[str, object] | None = None
    if spec.architecture == "corexy":
        vel_scale = math.sqrt(2.0)
    elif spec.architecture == "delta":
        m_c_car = (spec.gantry.carriage_mass_g / 3.0) * 1e-3
        delta_env = _delta_workspace_envelope(
            spec=spec,
            m_e_kg=m_x,
            m_c_kg=m_c_car,
            r_p_m=r_p,
            j_rot_kg_m2=j_rot,
            f_fric_car_n=f_fric / 3.0,
        )
        vel_scale = float(delta_env["max_speed_ratio"])
    else:
        vel_scale = 1.0

    # CoreXY diagonal moves run a single motor at sqrt(2)*v; Delta runs at max|J*u|*v
    motor_lin_v = v_max_m_s * vel_scale
    max_rpm = (motor_lin_v / (2.0 * math.pi * r_p)) * 60.0

    drive_type = kin.drive_type
    op_rpm = max_rpm * 0.7
    # Available torque at 70% peak speed (typical accel-to-cruise knee)
    t_avail = stepper_available_torque_nm(kin.xy_stepper, op_rpm, drive_type=drive_type)
    sf = 1.15 if drive_type == "foc_servo" else (1.20 if drive_type == "closed_loop_stepper" else 1.35)
    shaper_l1_factor = 1.169 if kin.input_shaper == "smooth_flatness" else 1.0
    a_reserve_m_s2 = 0.5  # 500 mm/s^2 cruise accel reserve

    if spec.architecture == "corexy":
        k_mot = 4.0 if is_awd else 2.0
        # CoreXY always has 2 belt loops (A and B) transmitting carriage force; AWD halves per-motor
        # torque (k_mot = 4) and balances the 2WD 3:1 unequal branch split to 50/50, rather than doubling belts.
        k_belts_xy = 2.0
        fx_fric, fy_fric = f_fric, f_fric
        m_x_mot = m_x + k_mot * j_rot / (r_p**2)
        m_y_mot = m_y + k_mot * j_rot / (r_p**2)
        # Pure X & pure Y motor torque demands (single-axis move) + exact worst-direction torque
        t_req_x = ((m_x_mot * a_max_m_s2 + fx_fric) / k_mot) * r_p
        t_req_y = ((m_y_mot * a_max_m_s2 + fy_fric) / k_mot) * r_p
        t_req_max = max(
            (math.hypot(m_x_mot * a_max_m_s2 + fx_fric, m_y_mot * a_max_m_s2 + fy_fric) / k_mot) * r_p,
            1e-6,
        )
        peak_belt_force = math.hypot(m_x * a_max_m_s2 + fx_fric, m_y * a_max_m_s2 + fy_fric) / k_belts_xy

        f_mot_budget = (t_avail / sf) / max(r_p, 1e-6) * k_mot
        a_x_m_s2 = max(f_mot_budget - fx_fric, 0.0) / max(m_x_mot, 1e-9)
        a_y_m_s2 = max(f_mot_budget - fy_fric, 0.0) / max(m_y_mot, 1e-9)
        a_worst_m_s2 = _solve_worst_dir_corexy_accel(m_x_mot, m_y_mot, fx_fric, fy_fric, f_mot_budget)
        motor_max_accel_x_mm_s2 = (a_x_m_s2 / shaper_l1_factor) * 1000.0
        motor_max_accel_y_mm_s2 = (a_y_m_s2 / shaper_l1_factor) * 1000.0
        motor_max_accel_mm_s2 = (a_worst_m_s2 / shaper_l1_factor) * 1000.0
        t_req_reserve = (
            math.hypot(m_x_mot * a_reserve_m_s2 + fx_fric, m_y_mot * a_reserve_m_s2 + fy_fric) / k_mot
        ) * r_p
    elif spec.architecture == "delta" and delta_env is not None:
        max_torque_fn = delta_env["max_torque_nm"]  # type: ignore[assignment]
        max_belt_fn = delta_env["max_belt_force_n"]  # type: ignore[assignment]
        max_a_torque_fn = delta_env["max_accel_torque_m_s2"]  # type: ignore[assignment]
        t_req_x = float(max_torque_fn(a_max_m_s2, "x"))  # type: ignore[operator]
        t_req_y = float(max_torque_fn(a_max_m_s2, "y"))  # type: ignore[operator]
        t_req_max = max(float(max_torque_fn(a_max_m_s2, None)), 1e-6)  # type: ignore[operator]
        peak_belt_force = float(max_belt_fn(a_max_m_s2))  # type: ignore[operator]

        t_budget = t_avail / sf
        a_x_m_s2 = float(max_a_torque_fn(t_budget, "x"))  # type: ignore[operator]
        a_y_m_s2 = float(max_a_torque_fn(t_budget, "y"))  # type: ignore[operator]
        a_worst_m_s2 = float(max_a_torque_fn(t_budget, None))  # type: ignore[operator]
        motor_max_accel_x_mm_s2 = (a_x_m_s2 / shaper_l1_factor) * 1000.0
        motor_max_accel_y_mm_s2 = (a_y_m_s2 / shaper_l1_factor) * 1000.0
        motor_max_accel_mm_s2 = (a_worst_m_s2 / shaper_l1_factor) * 1000.0
        t_req_reserve = float(max_torque_fn(a_reserve_m_s2, None))  # type: ignore[operator]
    else:
        # Cartesian bedslinger: 1 motor for X, 1 motor for heavy Y bed
        fx_fric, fy_fric = f_fric, f_fric * 1.4
        f_x = m_x * a_max_m_s2 + fx_fric
        f_y = m_y * a_max_m_s2 + fy_fric
        alpha_max = (1.0 / r_p) * a_max_m_s2
        t_req_x = f_x * r_p + j_rot * alpha_max
        t_req_y = f_y * r_p + j_rot * alpha_max
        t_req_max = max(t_req_x, t_req_y, 1e-6)
        peak_belt_force = max(f_x, f_y)

        t_budget_x = max(t_avail / sf - fx_fric * r_p, 0.0)
        t_budget_y = max(t_avail / sf - fy_fric * r_p, 0.0)
        a_x_m_s2 = t_budget_x / max(m_x * r_p + j_rot / r_p, 1e-9)
        a_y_m_s2 = t_budget_y / max(m_y * r_p + j_rot / r_p, 1e-9)
        motor_max_accel_x_mm_s2 = (a_x_m_s2 / shaper_l1_factor) * 1000.0
        motor_max_accel_y_mm_s2 = (a_y_m_s2 / shaper_l1_factor) * 1000.0
        motor_max_accel_mm_s2 = min(motor_max_accel_x_mm_s2, motor_max_accel_y_mm_s2)
        t_req_reserve = max(
            (m_x * a_reserve_m_s2 + fx_fric) * r_p + j_rot * (a_reserve_m_s2 / r_p),
            (m_y * a_reserve_m_s2 + fy_fric) * r_p + j_rot * (a_reserve_m_s2 / r_p),
        )

    margin = t_avail / t_req_max
    max_safe_a = motor_max_accel_mm_s2

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

    # Belt dynamic tension: elastic compatibility split on tight/slack branches:
    # dT_max = alpha_split * F_belt (alpha = k_short / (k_short + k_long) ~ 0.75 in 2WD CoreXY, 0.50 in AWD/symmetric)
    slack_share = float(details.get("slack_load_share", 0.5))
    peak_tension = belt.pretension_n + slack_share * peak_belt_force
    slack_tension = belt.pretension_n - slack_share * peak_belt_force
    belt_slip = slack_tension < idler_info["min_slack_tension_n"]

    # Exact Belt-Slip-Limited Max Accel (consistent with worst-direction peak_belt_force & compatibility split)
    f_belt_avail = max(
        (belt.pretension_n - idler_info["min_slack_tension_n"]) / max(slack_share, 0.5),
        0.0,
    )
    if spec.architecture == "corexy":
        a_belt_worst = _solve_worst_dir_corexy_accel(m_x, m_y, fx_fric, fy_fric, 2.0 * f_belt_avail)
        belt_max_accel_mm_s2 = (a_belt_worst / shaper_l1_factor) * 1000.0
    elif spec.architecture == "delta" and delta_env is not None:
        max_a_belt_fn = delta_env["max_accel_belt_m_s2"]  # type: ignore[assignment]
        a_belt_worst = float(max_a_belt_fn(f_belt_avail))  # type: ignore[operator]
        belt_max_accel_mm_s2 = (a_belt_worst / shaper_l1_factor) * 1000.0
    else:
        a_belt_x = max(f_belt_avail - fx_fric, 0.0) / max(m_x, 1e-6)
        a_belt_y = max(f_belt_avail - fy_fric, 0.0) / max(m_y, 1e-6)
        belt_max_accel_mm_s2 = (min(a_belt_x, a_belt_y) / shaper_l1_factor) * 1000.0

    # Exact Torque-Balance Max Velocity along piecewise voltage-limited pull-out curve
    motor_max_velocity_mm_s = _compute_motor_max_velocity_mm_s(
        stepper=stepper,
        r_p=r_p,
        drive_type=drive_type,
        sf=sf,
        vel_scale=vel_scale,
        t_req_reserve=t_req_reserve,
    )

    belt_cfg = resolve_belt(spec)
    belt_pluck_150 = belt.pluck_freq_hz(
        150.0, linear_mass_kg_m=float(belt_cfg["linear_mass_kg_m"])
    )
    belt_max_t = float(belt_cfg["max_working_tension_n"])
    belt_overtension = peak_tension > belt_max_t
    # Single-shear motor shaft bending limit scales with shaft diameter (Ø5mm NEMA 17: 45 N; Ø6.35mm NEMA 23: 80 N)
    single_shear_shaft_limit_n = 80.0 if stepper.shaft_diameter_mm >= 6.0 else 45.0
    shaft_bending = (
        idler_info["idler_shaft_deflection_um"] > 12.0
        or ((belt.pretension_n > single_shear_shaft_limit_n) and (not spec.gantry.double_shear_pulleys))
    )
    # Total tension preload reacted into the frame: 2 belt spans per belt loop
    if spec.architecture == "corexy":
        frame_preload = (8.0 if is_awd else 4.0) * belt.pretension_n
    elif spec.architecture == "delta":
        frame_preload = 6.0 * belt.pretension_n
    else:
        frame_preload = 4.0 * belt.pretension_n

    # Validate Delta reachability & minimum arm elevation angle (>= 20 deg) across inscribed circular bed
    delta_ok = True
    if spec.architecture == "delta":
        r_bed = 0.5 * min(bx, by)
        cx_bed, cy_bed = 0.5 * bx, 0.5 * by
        arm_len = spec.kinematics.delta_arm_length_mm
        sin_min_angle = math.sin(math.radians(20.0))
        test_pts_xy = [(cx_bed, cy_bed)] + [
            (
                cx_bed + r_bed * math.cos(2.0 * math.pi * k / 24.0),
                cy_bed + r_bed * math.sin(2.0 * math.pi * k / 24.0),
            )
            for k in range(24)
        ]
        for px, py in test_pts_xy:
            for pz in (0.0, bz):
                try:
                    car_zs = inverse_kinematics(spec, (px, py, pz))
                except ValueError:
                    delta_ok = False
                    break
                if any(((z_c - pz) / max(arm_len, 1e-6)) < sin_min_angle for z_c in car_zs):
                    delta_ok = False
                    break
            if not delta_ok:
                break

    bkt_id = spec.gantry.bracket_material
    bkt_resolved = resolve_gantry_bracket(spec)
    bracket_creep = belt.pretension_n > float(bkt_resolved["creep_tension_limit_n"])
    motor_volt_eff_pct = stepper.dynamic_voltage_efficiency(max_rpm * 0.7) * 100.0
    motor_heat_w = stepper.power_dissipation_w
    motor_est_temp_c = stepper.estimated_winding_temp_c(spec.frame.chamber_temp_c)
    motor_max_temp_c = stepper.max_winding_temp_c
    motor_thermal_risk = motor_est_temp_c > (motor_max_temp_c - 10.0)
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
        motor_est_temp_c=motor_est_temp_c,
        motor_max_winding_temp_c=motor_max_temp_c,
        motor_thermal_risk=motor_thermal_risk,
        motor_overcurrent_risk=motor_overcurrent,
        tooth_mesh_stiffness_n_per_mm=k_tooth_n_m * 1e-3,
        idler_shaft_stiffness_n_per_mm=k_idler_n_m * 1e-3,
        idler_shaft_deflection_um=idler_info["idler_shaft_deflection_um"],
        bearing_play_um=idler_info["bearing_play_um"],
        bracket_modulus_retention_pct=bracket_ret_pct,
        custom_bracket_active=override is not None,
        custom_bracket_label=override.label if override is not None else "",
        motor_max_accel_x_mm_s2=motor_max_accel_x_mm_s2,
        motor_max_accel_y_mm_s2=motor_max_accel_y_mm_s2,
        motor_max_accel_mm_s2=motor_max_accel_mm_s2,
        belt_max_accel_mm_s2=belt_max_accel_mm_s2,
        motor_max_velocity_mm_s=motor_max_velocity_mm_s,
        shaper_l1_accel_factor=shaper_l1_factor,
    )
