"""Time-domain forced response, 90-deg corner ringing, and virtual ISO 230-4 circular ballbar metrology.

Uses exact zero-order-hold (ZOH) matrix exponential (`scipy.linalg.expm`) discretization at
`dt = 0.0005 s` over the coupled electro-mechanical LPV state-space from `printsim.fea.mechatronics`,
including Klipper SCV corner planning, input-shaper convolution, Stribeck rail stiction, and
ISO 5753 bearing radial clearance backlash (`C2`, `CN`, `C3`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import expm

from printsim.fea.components import (
    hertzian_rail_carriage_stiffness,
    idler_shaft_and_bearing_stiffness,
)
from printsim.fea.klipper_shaper import apply_shaper_to_trajectory, get_shaper_pulses
from printsim.fea.mechatronics import (
    DriveControlConfig,
    build_coupled_state_space,
    corexy_belt_span_stiffness_matrix,
)
from printsim.schema import get_material
from printsim.schema.overrides import resolve_rail_quality

if TYPE_CHECKING:
    from printsim.fea.modal import ModalResult
    from printsim.geometry.builder import AssemblyModel

_DT_S: float = 0.0005


@dataclass
class TimeDomainMetrologyResult:
    """Combined Benchmark A (90° corner ringing) & Benchmark B (ISO 230-4 ballbar) results."""

    # Benchmark A: 90° High-Speed Corner Ringing
    corner_rounding_um: float
    corner_overshoot_um: float
    ringing_peak_to_peak_um: float
    ringing_unshaped_um: float
    ringing_reduction_pct: float
    ringing_decay_mm: float
    corner_speed_mm_s: float
    corner_accel_mm_s2: float
    shaper_used: str
    shaper_freq_hz: float
    corner_trace: dict[str, list[float]]

    # Benchmark B: Virtual ISO 230-4 Circular Ballbar
    ballbar_radius_mm: float
    ballbar_speed_mm_s: float
    circularity_gxy_um: float
    mean_radial_lag_um: float
    quadrant_glitch_um: float
    ballbar_angles_deg: list[float]
    ballbar_radial_error_um: list[float]

    def to_dict(self) -> dict:
        return {
            "corner_rounding_um": round(float(self.corner_rounding_um), 2),
            "corner_overshoot_um": round(float(self.corner_overshoot_um), 2),
            "ringing_peak_to_peak_um": round(float(self.ringing_peak_to_peak_um), 2),
            "ringing_unshaped_um": round(float(self.ringing_unshaped_um), 2),
            "ringing_reduction_pct": round(float(self.ringing_reduction_pct), 2),
            "ringing_decay_mm": round(float(self.ringing_decay_mm), 2),
            "corner_speed_mm_s": round(float(self.corner_speed_mm_s), 1),
            "corner_accel_mm_s2": round(float(self.corner_accel_mm_s2), 0),
            "shaper_used": self.shaper_used,
            "shaper_freq_hz": round(float(self.shaper_freq_hz), 2),
            "corner_trace": self.corner_trace,
            "ballbar_radius_mm": round(float(self.ballbar_radius_mm), 2),
            "ballbar_speed_mm_s": round(float(self.ballbar_speed_mm_s), 1),
            "circularity_gxy_um": round(float(self.circularity_gxy_um), 2),
            "mean_radial_lag_um": round(float(self.mean_radial_lag_um), 2),
            "quadrant_glitch_um": round(float(self.quadrant_glitch_um), 2),
            "ballbar_angles_deg": [round(float(a), 1) for a in self.ballbar_angles_deg],
            "ballbar_radial_error_um": [round(float(e), 2) for e in self.ballbar_radial_error_um],
        }


def _trapezoid_params(
    dist_m: float,
    v0: float,
    v_cruise: float,
    v1: float,
    accel: float,
) -> tuple[float, float, float, float, float, float, float, float, float]:
    """Compute exact analytical 1D trapezoidal velocity profile parameters."""
    v_tri = math.sqrt(max(0.5 * (v0 * v0 + v1 * v1) + accel * dist_m, 0.0))
    v_peak = min(v_cruise, v_tri)
    v_peak = max(v_peak, min(v0, v1))

    d_acc = max(0.0, (v_peak * v_peak - v0 * v0) / (2.0 * accel))
    d_dec = max(0.0, (v_peak * v_peak - v1 * v1) / (2.0 * accel))
    d_flat = max(0.0, dist_m - d_acc - d_dec)

    t_acc = abs(v_peak - v0) / accel
    t_flat = d_flat / max(v_peak, 1e-6)
    t_dec = abs(v_peak - v1) / accel
    t_tot = t_acc + t_flat + t_dec
    return dist_m, v0, v_peak, accel, d_acc, d_flat, t_acc, t_flat, t_tot


def _eval_trapezoid(
    t_local: np.ndarray,
    params: tuple[float, float, float, float, float, float, float, float, float],
) -> np.ndarray:
    """Evaluate exact analytical position `s(t_local)` clamped to `[0, dist_m]`."""
    dist_m, v0, v_peak, accel, d_acc, d_flat, t_acc, t_flat, t_tot = params
    s_arr = np.empty_like(t_local, dtype=float)

    m0 = t_local <= 0.0
    m4 = t_local >= t_tot
    m1 = (~m0) & (t_local <= t_acc)
    m2 = (~m0) & (~m1) & (t_local <= t_acc + t_flat)
    m3 = (~m0) & (~m1) & (~m2) & (~m4)

    s_arr[m0] = 0.0
    s_arr[m4] = dist_m
    s_arr[m1] = v0 * t_local[m1] + 0.5 * accel * (t_local[m1] ** 2)
    tau2 = t_local[m2] - t_acc
    s_arr[m2] = d_acc + v_peak * tau2
    tau3 = t_local[m3] - (t_acc + t_flat)
    s_arr[m3] = d_acc + d_flat + v_peak * tau3 - 0.5 * accel * (tau3**2)
    return s_arr


def _trapezoidal_1d(
    dist_m: float,
    v0: float,
    v_cruise: float,
    v1: float,
    accel: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample exact 1D trapezoidal velocity profile `(t_local, s_local)` from `0` to `dist_m`."""
    params = _trapezoid_params(dist_m, v0, v_cruise, v1, accel)
    t_tot = params[-1]
    t_arr = np.arange(0.0, t_tot + 0.5 * dt, dt, dtype=float)
    return t_arr, _eval_trapezoid(t_arr, params)


def _discretize_zoh(
    A: np.ndarray,
    B: np.ndarray,
    B_dist: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute exact ZOH `(A_d, B_d, B_dist_d)` in a single augmented matrix exponential."""
    n = A.shape[0]
    m_u = B.shape[1]
    m_f = B_dist.shape[1]
    aug = np.zeros((n + m_u + m_f, n + m_u + m_f), dtype=float)
    aug[:n, :n] = A * dt
    aug[:n, n : n + m_u] = B * dt
    aug[:n, n + m_u :] = B_dist * dt
    em = expm(aug)
    return em[:n, :n], em[:n, n : n + m_u], em[:n, n + m_u :]


def _run_zoh_lti(
    A_d: np.ndarray,
    B_d: np.ndarray,
    C_noz: np.ndarray,
    D: np.ndarray,
    u_cmd: np.ndarray,
) -> np.ndarray:
    """Fast ZOH time-domain integration starting from steady equilibrium at `u_cmd[0]`."""
    n_steps = u_cmd.shape[0]
    n_state = A_d.shape[0]
    u0 = u_cmd[0]
    # Shift coordinates relative to initial rest point u0 so initial state is exactly 0
    du = u_cmd - u0[None, :]
    x = np.zeros(n_state, dtype=float)
    y_out = np.empty((n_steps, 2), dtype=float)
    c_t = C_noz.T
    a_t = A_d.T
    b_t = B_d.T
    for k in range(n_steps):
        y_out[k] = x @ c_t + du[k] @ D.T + u0
        x = x @ a_t + du[k] @ b_t
    return y_out


def _resolve_shaper(
    assembly: AssemblyModel,
    modal_result: ModalResult,
    coupled_fx: float,
    coupled_fy: float,
    coupled_zx: float,
    coupled_zy: float,
    shaper_name: str | None,
    shaper_freq_hz: float | None,
) -> tuple[str, tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], float]:
    spec = assembly.spec
    if shaper_name is not None:
        chosen = shaper_name.strip().upper()
    elif spec.kinematics.input_shaper.strip().lower() == "none":
        chosen = "NONE"
    elif modal_result.shaper_x is not None and modal_result.shaper_x.recommended_shaper:
        chosen = modal_result.shaper_x.recommended_shaper.strip().upper()
    else:
        chosen = "EI"

    fx = float(shaper_freq_hz) if shaper_freq_hz is not None else max(coupled_fx, 8.0)
    fy = float(shaper_freq_hz) if shaper_freq_hz is not None else max(coupled_fy, 8.0)
    zx = float(np.clip(coupled_zx, 0.02, 0.25))
    zy = float(np.clip(coupled_zy, 0.02, 0.25))

    if chosen == "NONE":
        sh_x = get_shaper_pulses("NONE", fx)
        sh_y = get_shaper_pulses("NONE", fy)
        return "NONE", sh_x, sh_y, 0.0

    sh_x = get_shaper_pulses(chosen, fx, damping_ratio=zx)
    sh_y = get_shaper_pulses(chosen, fy, damping_ratio=zy)
    # Klipper-style phase alignment: shift faster axis by (t_center_max - t_center_axis)
    # so X and Y share identical shaper group delay and introduce zero elliptic skew.
    ts_x = float(np.sum(sh_x[0] * sh_x[1]) / max(float(np.sum(sh_x[0])), 1e-12))
    ts_y = float(np.sum(sh_y[0] * sh_y[1]) / max(float(np.sum(sh_y[0])), 1e-12))
    ts_max = max(ts_x, ts_y)
    if ts_x < ts_max - 1e-9:
        sh_x = (sh_x[0], sh_x[1] + (ts_max - ts_x))
    if ts_y < ts_max - 1e-9:
        sh_y = (sh_y[0], sh_y[1] + (ts_max - ts_y))
    return chosen, sh_x, sh_y, 0.5 * (fx + fy)


def simulate_time_domain_metrology(
    assembly: AssemblyModel,
    modal_result: ModalResult,
    shaper_name: str | None = None,
    shaper_freq_hz: float | None = None,
    drive_cfg: DriveControlConfig | None = None,
    ballbar_speed_mm_s: float | None = None,
) -> TimeDomainMetrologyResult:
    """Run Benchmark A (90° corner ringing) and Benchmark B (ISO 230-4 ballbar) via ZOH `expm`."""
    cfg = drive_cfg or DriveControlConfig()
    spec = assembly.spec
    dt = _DT_S

    # Build coupled state-space at bed center
    coupled = build_coupled_state_space(assembly, modal_result, u_rel=0.5, v_rel=0.5, drive_cfg=cfg)
    n_state = coupled.state_dim
    n_dof = coupled.n_mot + coupled.m_struct
    idx_cx = 2 + coupled.n_mot
    idx_cy = 3 + coupled.n_mot
    idx_vx = n_dof + idx_cx
    idx_vy = n_dof + idx_cy

    m_cx = max(float(assembly.x_moving_mass_kg), 0.05)
    m_cy = max(float(assembly.y_moving_mass_kg), 0.05)
    B_dist = np.zeros((n_state, 2), dtype=float)
    B_dist[idx_vx, 0] = 1.0 / m_cx
    B_dist[idx_vy, 1] = 1.0 / m_cy

    A_d, B_d, B_dist_d = _discretize_zoh(coupled.A, coupled.B, B_dist, dt)

    sh_used, shaper_x, shaper_y, sh_freq_rep = _resolve_shaper(
        assembly,
        modal_result,
        coupled.coupled_freq_x_hz,
        coupled.coupled_freq_y_hz,
        coupled.coupled_damping_x,
        coupled.coupled_damping_y,
        shaper_name,
        shaper_freq_hz,
    )

    # =========================================================================
    # Benchmark A: 90° High-Speed Corner Ringing (40 mm +X -> 90° -> 40 mm +Y)
    # =========================================================================
    v_c_mm = min(float(spec.kinematics.max_velocity_mm_s), 250.0)
    a_c_mm = min(float(spec.kinematics.max_accel_mm_s2), 15000.0)
    v_c = v_c_mm * 1e-3
    a_c = a_c_mm * 1e-3
    v_scv = min(v_c, max(float(spec.kinematics.square_corner_velocity_mm_s) * 1e-3, 0.001))

    leg_len = 0.040
    leg1 = _trapezoid_params(leg_len, 0.0, v_c, v_scv, a_c)
    leg2 = _trapezoid_params(leg_len, v_scv, v_c, 0.0, a_c)
    t_corner = float(leg1[-1])
    t_tot_2 = float(leg2[-1])
    t_shaper_tail = max(float(shaper_x[1][-1]), float(shaper_y[1][-1]))
    t_dwell = max(0.06, t_shaper_tail + 0.04)

    t_a = np.arange(0.0, t_corner + t_tot_2 + t_dwell + 0.5 * dt, dt, dtype=float)
    x_ideal_a = _eval_trapezoid(t_a, leg1) - leg_len
    y_ideal_a = _eval_trapezoid(t_a - t_corner, leg2)
    r_ideal_a = np.column_stack([x_ideal_a, y_ideal_a])

    if sh_used == "NONE":
        r_shaped_a = r_ideal_a
    else:
        r_shaped_a = np.column_stack(
            [
                apply_shaper_to_trajectory(t_a, x_ideal_a, shaper_x),
                apply_shaper_to_trajectory(t_a, y_ideal_a, shaper_y),
            ]
        )

    y_unshaped = _run_zoh_lti(A_d, B_d, coupled.C_noz, coupled.D, r_ideal_a)
    y_shaped = y_unshaped if sh_used == "NONE" else _run_zoh_lti(A_d, B_d, coupled.C_noz, coupled.D, r_shaped_a)

    x_sh, y_sh = y_shaped[:, 0], y_shaped[:, 1]
    x_un, y_un = y_unshaped[:, 0], y_unshaped[:, 1]

    # Corner rounding: max inward diagonal excursion into (x <= 0, y >= 0) near vertex (0, 0)
    mask_in = (x_sh <= 0.0) & (y_sh >= 0.0) & (np.hypot(x_sh, y_sh) < 0.015)
    corner_rounding_um = float(np.max(np.minimum(-x_sh[mask_in], y_sh[mask_in]))) * 1e6 if np.any(mask_in) else 0.0

    # Corner overshoot: max outward +X excursion past x = 0 after entering +Y wall
    mask_post_all = (t_a >= t_corner) & (y_sh < 0.036)
    corner_overshoot_um = (
        max(0.0, float(np.max(x_sh[mask_post_all]))) * 1e6 if np.any(mask_post_all) else 0.0
    )

    # Unshaped vs shaped residual post-corner wall ringing
    mask_wall_un = (t_a >= t_corner) & (y_un <= 0.036)
    ringing_unshaped_um = (
        float(np.max(x_un[mask_wall_un]) - np.min(x_un[mask_wall_un])) * 1e6
        if np.any(mask_wall_un)
        else 0.0
    )

    mask_wall_sh = (t_a >= t_corner + t_shaper_tail) & (y_sh <= 0.036)
    if np.any(mask_wall_sh):
        x_w = x_sh[mask_wall_sh]
        ringing_peak_to_peak_um = float(np.max(x_w) - np.min(x_w)) * 1e6
    else:
        ringing_peak_to_peak_um = 0.0

    if sh_used == "NONE":
        ringing_peak_to_peak_um = ringing_unshaped_um
        ringing_reduction_pct = 0.0
    else:
        ringing_reduction_pct = 100.0 * max(
            0.0, 1.0 - ringing_peak_to_peak_um / max(ringing_unshaped_um, 1e-6)
        )

    # Decay distance along +Y wall until |x| < 5 um
    mask_decay = (t_a >= t_corner + 0.5 * t_shaper_tail) & (y_sh <= 0.036)
    if np.any(mask_decay):
        idx_decay = np.flatnonzero(mask_decay)
        over_5um = np.flatnonzero(np.abs(x_sh[idx_decay]) >= 5.0e-6)
        if over_5um.size > 0:
            last_bad = idx_decay[int(over_5um[-1])]
            ringing_decay_mm = max(0.0, float(y_sh[last_bad]) * 1e3)
        else:
            ringing_decay_mm = 0.0
    else:
        ringing_decay_mm = 0.0

    # Downsampled corner trace (~90 points centered around the 90° vertex)
    i_lo = max(0, int(round((t_corner - 0.055) / dt)))
    i_hi = min(len(t_a) - 1, int(round((t_corner + 0.115 + t_shaper_tail) / dt)))
    trace_idx = np.unique(np.linspace(i_lo, i_hi, 90, dtype=int))
    corner_trace = {
        "ideal_x_mm": [round(float(x_ideal_a[i]) * 1e3, 4) for i in trace_idx],
        "ideal_y_mm": [round(float(y_ideal_a[i]) * 1e3, 4) for i in trace_idx],
        "actual_x_mm": [round(float(x_sh[i]) * 1e3, 4) for i in trace_idx],
        "actual_y_mm": [round(float(y_sh[i]) * 1e3, 4) for i in trace_idx],
        "unshaped_x_mm": [round(float(x_un[i]) * 1e3, 4) for i in trace_idx],
        "unshaped_y_mm": [round(float(y_un[i]) * 1e3, 4) for i in trace_idx],
    }

    # =========================================================================
    # Benchmark B: Virtual ISO 230-4 Circular Ballbar (R=50mm, 360° CW)
    # =========================================================================
    r_ball_mm = 50.0
    r_ball = r_ball_mm * 1e-3
    v_ball_mm = (
        float(ballbar_speed_mm_s)
        if ballbar_speed_mm_s is not None
        else min(float(spec.kinematics.max_velocity_mm_s) * 0.45, 120.0)
    )
    v_ball_mm = max(5.0, v_ball_mm)
    v_ball = v_ball_mm * 1e-3
    omega_b = v_ball / r_ball
    t_rev = (2.0 * math.pi) / omega_b
    t_lead = 0.25 * t_rev
    t_end_b = t_lead + t_rev
    t_b = np.arange(0.0, t_end_b + 0.5 * dt, dt, dtype=float)
    phi_b = omega_b * (t_b - t_lead)  # -pi/2 -> 0 -> 2*pi

    x_ball_ideal = r_ball * np.cos(phi_b)
    y_ball_ideal = -r_ball * np.sin(phi_b)
    if sh_used == "NONE":
        u_ball_sh = np.column_stack([x_ball_ideal, y_ball_ideal])
    else:
        u_ball_sh = np.column_stack(
            [
                apply_shaper_to_trajectory(t_b, x_ball_ideal, shaper_x),
                apply_shaper_to_trajectory(t_b, y_ball_ideal, shaper_y),
            ]
        )

    # Effective contouring tracking bandwidth (servo position loop / stepper L/R + interpolator lag).
    # Note: foc_servo already carries exact plant lag b_em / k_em = 1 / omega_p inside (A, B),
    # so we skip the redundant 1st-order pre-filter on foc_servo to avoid double counting.
    is_foc = cfg.drive_type in ("foc_servo", "closed_loop_stepper")
    stepper = spec.kinematics.xy_stepper
    tau_vff_foc = 0.0
    if cfg.drive_type == "foc_servo":
        omega_p = 2.0 * math.pi * max(float(cfg.servo_pos_bw_hz), 2.0)
        alpha_trk = 1.0
        if cfg.mode_space_torque_ff:
            tau_vff_foc = (1.0 - 1.0 / 1.45) / omega_p
    elif cfg.drive_type == "closed_loop_stepper":
        omega_track = 2.0 * math.pi * max(float(cfg.servo_pos_bw_hz) * 1.25, 22.0)
        if cfg.mode_space_torque_ff:
            omega_track *= 1.45
        tau_target = 1.0 / omega_track
        tau_plant = coupled.b_em_per_motor_ns_m / max(coupled.k_em_per_motor_n_m, 1e-6)
        if tau_target > tau_plant:
            tau_pre = math.sqrt(tau_target * tau_target - tau_plant * tau_plant)
            alpha_trk = 1.0 - math.exp(-dt / max(tau_pre, 1e-6))
        else:
            alpha_trk = 1.0
    else:
        # Open-loop stepper: keep only electrical winding current lag tau_e = L / R (no artificial
        # servo following-error constant); shaper smoothing + plant compliance produce radial lag.
        tau_e = (stepper.inductance_mh * 1e-3) / max(stepper.phase_resistance_ohm, 0.05)
        tau_eff = max(tau_e, 1e-6)
        omega_track = 1.0 / tau_eff
        alpha_trk = 1.0 - math.exp(-omega_track * dt)

    # Precompute CoreXY LPV delta_K(theta) on 72 angular sectors for fast loop lookup
    bx_m = max(float(spec.kinematics.build_volume_mm[0]) * 1e-3, 0.12)
    by_m = max(float(spec.kinematics.build_volume_mm[1]) * 1e-3, 0.12)
    k_cart_center = coupled.k_belt_cart
    dk_sectors = np.zeros((72, 2, 2), dtype=float)
    if spec.architecture == "corexy":
        for sec in range(72):
            ang = math.radians(5.0 * sec)
            u_sec = float(np.clip(0.5 + (r_ball * math.cos(ang)) / bx_m, 0.05, 0.95))
            v_sec = float(np.clip(0.5 - (r_ball * math.sin(ang)) / by_m, 0.05, 0.95))
            k_sec, _, _ = corexy_belt_span_stiffness_matrix(assembly, u_sec, v_sec)
            dk_sectors[sec] = k_sec - k_cart_center
        if is_foc and cfg.mode_space_torque_ff:
            dk_sectors *= 0.65

    # Stribeck rail friction + ISO 5753 bearing radial clearance play
    rail_qual = resolve_rail_quality(spec)
    rail_info = hertzian_rail_carriage_stiffness(
        spec.gantry.x_rail_profile,
        preload=spec.gantry.rail_preload,
        block_type=spec.gantry.rail_block_type,
        rail_quality=spec.gantry.rail_quality,
        preload_fraction_override=spec.model_overrides.rail_preload.preload_fraction,
        stiffness_mult_override=rail_qual["stiffness_mult"],
        drag_mult_override=rail_qual["drag_mult"],
    )
    f_coulomb = float(rail_info["rolling_friction_n"] + rail_info["seal_drag_n"])
    f_break = float(rail_info["breakaway_friction_n"])
    v_str = 0.008
    v_eps = 0.0015

    bkt_mat = get_material(spec.gantry.bracket_material)
    e_bkt_eff = bkt_mat.effective_modulus_pa(spec.frame.chamber_temp_c)
    idler_info = idler_shaft_and_bearing_stiffness(
        spec.gantry.belt,
        double_shear=spec.gantry.double_shear_pulleys,
        bracket_e_pa=e_bkt_eff,
    )
    half_play_m = 0.5 * float(idler_info["bearing_play_um"]) * 1e-6

    if is_foc and cfg.mode_space_torque_ff:
        # Mode-space torque FF + AWD direction_split compensates quadrant reversal stiction & play
        f_coulomb *= 0.55
        f_break *= 0.55
        half_play_m *= 0.70

    # Initialize state cleanly at phi = -pi/2: pos=(0, +R), vel=(+v_ball, 0)
    x_state = np.zeros(n_state, dtype=float)
    u_start = u_ball_sh[0]
    if coupled.T_cart is not None:
        x_state[2 : 2 + coupled.n_mot] = coupled.T_cart @ u_start
    elif coupled.n_mot == 2 and spec.architecture == "corexy":
        x_state[2] = u_start[0] + u_start[1]
        x_state[3] = u_start[0] - u_start[1]
    elif coupled.n_mot == 4 and spec.architecture == "corexy":
        x_state[2] = u_start[0] + u_start[1]
        x_state[3] = u_start[0] + u_start[1]
        x_state[4] = u_start[0] - u_start[1]
        x_state[5] = u_start[0] - u_start[1]
    elif coupled.n_mot == 2:
        x_state[2] = u_start[0]
        x_state[3] = u_start[1]
    x_state[idx_cx] = u_start[0]
    x_state[idx_cy] = u_start[1]
    x_state[idx_vx] = v_ball
    x_state[idx_vy] = 0.0

    # Precompute contouring lag + ISO 5753 bearing radial play command sequence
    u_trk_x = float(u_start[0])
    u_trk_y = float(u_start[1])
    bx_play = u_trk_x
    by_play = u_trk_y - half_play_m

    a_t = A_d.T
    b_t = B_d.T
    bf_t = B_dist_d.T
    c_t = coupled.C_noz.T

    n_b = len(t_b)
    inv_5deg = 72.0 / (2.0 * math.pi)
    two_pi = 2.0 * math.pi
    u_sh_x = u_ball_sh[:, 0]
    u_sh_y = u_ball_sh[:, 1]
    u_eff_arr = np.empty((n_b, 2), dtype=float)
    one_m_alpha = 1.0 - alpha_trk
    prev_sh_x = float(u_sh_x[0])
    prev_sh_y = float(u_sh_y[0])

    for k in range(n_b):
        cur_sh_x = float(u_sh_x[k])
        cur_sh_y = float(u_sh_y[k])
        u_trk_x = one_m_alpha * u_trk_x + alpha_trk * cur_sh_x
        u_trk_y = one_m_alpha * u_trk_y + alpha_trk * cur_sh_y
        if u_trk_x - bx_play > half_play_m:
            bx_play = u_trk_x - half_play_m
        elif bx_play - u_trk_x > half_play_m:
            bx_play = u_trk_x + half_play_m
        if u_trk_y - by_play > half_play_m:
            by_play = u_trk_y - half_play_m
        elif by_play - u_trk_y > half_play_m:
            by_play = u_trk_y + half_play_m
        if tau_vff_foc > 0.0:
            vx_ff = (cur_sh_x - prev_sh_x) / dt if k > 0 else v_ball
            vy_ff = (cur_sh_y - prev_sh_y) / dt if k > 0 else 0.0
            u_eff_arr[k, 0] = bx_play + tau_vff_foc * vx_ff
            u_eff_arr[k, 1] = by_play + tau_vff_foc * vy_ff
        else:
            u_eff_arr[k, 0] = bx_play
            u_eff_arr[k, 1] = by_play
        prev_sh_x = cur_sh_x
        prev_sh_y = cur_sh_y

    u_drive_all = u_eff_arr @ b_t
    sec_all = ((phi_b % two_pi) * inv_5deg).astype(int) % 72

    xy_actual = np.empty((n_b, 2), dtype=float)
    f_dist_vec = np.zeros(2, dtype=float)
    is_corexy = spec.architecture == "corexy"
    is_awd = is_corexy and coupled.n_mot == 4
    f_delta = f_break - f_coulomb
    m_exp = math.exp
    m_tanh = math.tanh
    inv_v_str = 1.0 / v_str
    inv_v_eps = 1.0 / v_eps

    for k in range(n_b):
        xy_actual[k] = x_state @ c_t

        vx = float(x_state[idx_vx])
        vy = float(x_state[idx_vy])
        fx_fric = -(f_coulomb + f_delta * m_exp(-((vx * inv_v_str) ** 2))) * m_tanh(vx * inv_v_eps)
        fy_fric = -(f_coulomb + f_delta * m_exp(-((vy * inv_v_str) ** 2))) * m_tanh(vy * inv_v_eps)

        if is_corexy:
            if is_awd:
                qm_a = 0.5 * (float(x_state[2]) + float(x_state[3]))
                qm_b = 0.5 * (float(x_state[4]) + float(x_state[5]))
            else:
                qm_a = float(x_state[2])
                qm_b = float(x_state[3])
            qm_x = 0.5 * (qm_a + qm_b)
            qm_y = 0.5 * (qm_a - qm_b)
            dx_c = float(x_state[idx_cx]) - float(x_state[0]) - qm_x
            dy_c = float(x_state[idx_cy]) - float(x_state[1]) - qm_y
            dk = dk_sectors[int(sec_all[k])]
            f_dist_vec[0] = fx_fric - (float(dk[0, 0]) * dx_c + float(dk[0, 1]) * dy_c)
            f_dist_vec[1] = fy_fric - (float(dk[1, 0]) * dx_c + float(dk[1, 1]) * dy_c)
        else:
            f_dist_vec[0] = fx_fric
            f_dist_vec[1] = fy_fric

        x_state = x_state @ a_t + u_drive_all[k] + f_dist_vec @ bf_t

    # Extract final full 360° steady-state revolution (phi in [0, 2*pi))
    rev_mask = phi_b >= -1e-9
    phi_rev = np.degrees(phi_b[rev_mask]) % 360.0
    xy_rev = xy_actual[rev_mask]
    e_r_um = (np.hypot(xy_rev[:, 0], xy_rev[:, 1]) - r_ball) * 1e6

    # Bin onto 72 polar angles (0, 5, ..., 355 deg) preserving quadrant spike extrema
    angles_deg = [5.0 * i for i in range(72)]
    radial_72: list[float] = []
    global_med = float(np.median(e_r_um))
    for i in range(72):
        center_deg = 5.0 * i
        diff = ((phi_rev - center_deg + 180.0) % 360.0) - 180.0
        in_bin = np.abs(diff) <= 2.5
        if np.any(in_bin):
            vals = e_r_um[in_bin]
            idx_ext = int(np.argmax(np.abs(vals - global_med)))
            radial_72.append(float(vals[idx_ext]))
        else:
            radial_72.append(float(e_r_um[int(np.argmin(np.abs(diff)))]))

    arr_72 = np.asarray(radial_72, dtype=float)
    circularity_gxy_um = float(np.max(arr_72) - np.min(arr_72))
    mean_radial_lag_um = float(np.mean(e_r_um))

    # Measure quadrant glitch spike height above local arc baseline at 0°, 90°, 180°, 270°
    quad_glitches: list[float] = []
    for q_deg in (0.0, 90.0, 180.0, 270.0):
        d_q = ((phi_rev - q_deg + 180.0) % 360.0) - 180.0
        win_spike = (d_q >= -8.0) & (d_q <= 16.0)
        win_base = ((d_q >= -28.0) & (d_q < -10.0)) | ((d_q > 18.0) & (d_q <= 35.0))
        if np.any(win_spike) and np.any(win_base):
            base_val = float(np.median(e_r_um[win_base]))
            spike_val = float(np.max(np.abs(e_r_um[win_spike] - base_val)))
            quad_glitches.append(spike_val)
    quadrant_glitch_um = max(quad_glitches) if quad_glitches else 0.0

    return TimeDomainMetrologyResult(
        corner_rounding_um=corner_rounding_um,
        corner_overshoot_um=corner_overshoot_um,
        ringing_peak_to_peak_um=ringing_peak_to_peak_um,
        ringing_unshaped_um=ringing_unshaped_um,
        ringing_reduction_pct=ringing_reduction_pct,
        ringing_decay_mm=ringing_decay_mm,
        corner_speed_mm_s=v_c_mm,
        corner_accel_mm_s2=a_c_mm,
        shaper_used=sh_used,
        shaper_freq_hz=sh_freq_rep,
        corner_trace=corner_trace,
        ballbar_radius_mm=r_ball_mm,
        ballbar_speed_mm_s=v_ball_mm,
        circularity_gxy_um=circularity_gxy_um,
        mean_radial_lag_um=mean_radial_lag_um,
        quadrant_glitch_um=quadrant_glitch_um,
        ballbar_angles_deg=angles_deg,
        ballbar_radial_error_um=radial_72,
    )
