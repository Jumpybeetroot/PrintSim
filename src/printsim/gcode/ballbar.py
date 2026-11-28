"""Time-domain forced response, 90-deg corner ringing, and virtual ISO 230-4 circular ballbar metrology.

Uses exact zero-order-hold (ZOH) matrix exponential (`scipy.linalg.expm`) discretization at
`dt = 0.0005 s` over the coupled electro-mechanical LPV state-space from `printsim.fea.mechatronics`,
including Klipper SCV corner planning, input-shaper convolution, Stribeck rail stiction, and
ISO 5753 bearing radial clearance backlash (`C2`, `CN`, `C3`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import expm

from printsim.fea.components import (
    compute_bearing_cte_clearance_um,
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
    from printsim.schema.printer import PrinterSpec

_DT_S: float = 0.0005


def _belt_tooth_backlash_um(spec: PrinterSpec) -> float:
    """Estimate belt tooth-flank reversal backlash (µm) by tooth profile, pretension, and AWD."""
    belt = spec.gantry.belt
    btype = str(belt.type).lower()
    if "htd" in btype:
        base_um = 18.0  # Semicircular HTD profile exhibits high tooth-flank clearance
    elif btype.startswith("at"):
        base_um = 2.0  # Trapezoidal AT3/AT5 precision profile
    elif btype.startswith("3gt"):
        base_um = 5.0  # 3 mm pitch curvilinear 3GT
    else:
        base_um = 3.0  # Gates PowerGrip 2GT / GT3 modified curvilinear profile
    t0 = max(float(belt.pretension_n), 1.0)
    tension_scale = 1.0 / (1.0 + t0 / 35.0)
    awd_scale = 0.55 if (spec.architecture == "corexy" and spec.gantry.drive_mode == "awd") else 1.0
    return float(base_um * tension_scale * awd_scale)


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

    # Benchmark A separated geometric vs. tracking-lag error metrics
    printed_path_error_um: float = 0.0
    shaper_path_smoothing_um: float = 0.0
    cross_track_error_um: float = 0.0
    along_track_lag_um: float = 0.0

    # Benchmark C: Multi-Segment Zig-Zag S-Curve
    zigzag_path_error_um: float = 0.0
    zigzag_shaper_smoothing_um: float = 0.0
    zigzag_residual_ringing_um: float = 0.0
    zigzag_unshaped_ringing_um: float = 0.0
    zigzag_trace: dict[str, list[float]] = field(default_factory=dict)

    # ISO 230-4 reversal backlash & per-axis stiction diagnostics
    diagonal_glitch_45deg_um: float = 0.0
    reversal_backlash_um: float = 0.0
    rail_stiction_x_n: float = 0.0
    rail_stiction_y_n: float = 0.0

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
            "printed_path_error_um": round(float(self.printed_path_error_um), 2),
            "shaper_path_smoothing_um": round(float(self.shaper_path_smoothing_um), 2),
            "cross_track_error_um": round(float(self.cross_track_error_um), 2),
            "along_track_lag_um": round(float(self.along_track_lag_um), 2),
            "corner_trace": self.corner_trace,
            "ballbar_radius_mm": round(float(self.ballbar_radius_mm), 2),
            "ballbar_speed_mm_s": round(float(self.ballbar_speed_mm_s), 1),
            "circularity_gxy_um": round(float(self.circularity_gxy_um), 2),
            "mean_radial_lag_um": round(float(self.mean_radial_lag_um), 2),
            "quadrant_glitch_um": round(float(self.quadrant_glitch_um), 2),
            "diagonal_glitch_45deg_um": round(float(self.diagonal_glitch_45deg_um), 2),
            "reversal_backlash_um": round(float(self.reversal_backlash_um), 2),
            "rail_stiction_x_n": round(float(self.rail_stiction_x_n), 3),
            "rail_stiction_y_n": round(float(self.rail_stiction_y_n), 3),
            "ballbar_angles_deg": [round(float(a), 1) for a in self.ballbar_angles_deg],
            "ballbar_radial_error_um": [round(float(e), 2) for e in self.ballbar_radial_error_um],
            "zigzag_path_error_um": round(float(self.zigzag_path_error_um), 2),
            "zigzag_shaper_smoothing_um": round(float(self.zigzag_shaper_smoothing_um), 2),
            "zigzag_residual_ringing_um": round(float(self.zigzag_residual_ringing_um), 2),
            "zigzag_unshaped_ringing_um": round(float(self.zigzag_unshaped_ringing_um), 2),
            "zigzag_trace": self.zigzag_trace,
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


def _dist_to_polyline(pts: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Exact minimum Euclidean distance from each 2D point in `pts` to the polyline `vertices`."""
    min_d2 = np.full(pts.shape[0], np.inf, dtype=float)
    for i in range(len(vertices) - 1):
        a = vertices[i]
        b = vertices[i + 1]
        ab = b - a
        ab_len2 = float(ab[0] * ab[0] + ab[1] * ab[1])
        if ab_len2 <= 1e-24:
            diff = pts - a[None, :]
        else:
            t = np.clip(((pts - a[None, :]) @ ab) / ab_len2, 0.0, 1.0)
            diff = pts - (a[None, :] + t[:, None] * ab[None, :])
        d2 = diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1]
        min_d2 = np.minimum(min_d2, d2)
    return np.sqrt(np.maximum(min_d2, 0.0))


def _sample_wall_dev_um(
    y_m: np.ndarray,
    x_m: np.ndarray,
    i_start: int,
    wall_d_mm: np.ndarray,
) -> list[float]:
    """Interpolate lateral wall deviation `x * 1e6` (um) vs. +Y wall distance `y * 1e3` (mm)."""
    if i_start >= len(y_m) - 1:
        return [0.0] * len(wall_d_mm)
    i_peak = i_start + int(np.argmax(y_m[i_start:]))
    y_seg = y_m[i_start : i_peak + 1] * 1e3
    x_seg = x_m[i_start : i_peak + 1] * 1e6
    if y_seg.size < 2:
        return [0.0] * len(wall_d_mm)
    order = np.argsort(y_seg, kind="mergesort")
    vals = np.interp(wall_d_mm, y_seg[order], x_seg[order])
    return [round(float(v), 2) for v in vals]


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

    # Separated geometric printed path error vs. shaper smoothing vs. cross-track / along-track lag
    corner_poly = np.array([[-leg_len, 0.0], [0.0, 0.0], [0.0, leg_len]], dtype=float)
    printed_path_error_um = float(np.max(_dist_to_polyline(y_shaped, corner_poly))) * 1e6
    shaper_path_smoothing_um = (
        0.0
        if sh_used == "NONE"
        else float(np.max(_dist_to_polyline(r_shaped_a, corner_poly))) * 1e6
    )

    err_a = y_shaped - r_shaped_a
    dr_a = np.gradient(r_shaped_a, dt, axis=0)
    spd_a = np.hypot(dr_a[:, 0], dr_a[:, 1])
    moving_a = spd_a > 1e-3
    if np.any(moving_a):
        t_hat = dr_a[moving_a] / spd_a[moving_a, None]
        n_hat = np.column_stack([-t_hat[:, 1], t_hat[:, 0]])
        err_m = err_a[moving_a]
        cross_track_error_um = float(np.max(np.abs(np.sum(err_m * n_hat, axis=1)))) * 1e6
        along_track_lag_um = float(np.max(np.abs(np.sum(err_m * t_hat, axis=1)))) * 1e6
    else:
        cross_track_error_um = 0.0
        along_track_lag_um = 0.0

    # Downsampled corner trace (~90 points centered around the 90° vertex) + +Y wall profile
    i_lo = max(0, int(round((t_corner - 0.055) / dt)))
    i_hi = min(len(t_a) - 1, int(round((t_corner + 0.115 + t_shaper_tail) / dt)))
    trace_idx = np.unique(np.linspace(i_lo, i_hi, 90, dtype=int))
    wall_d_mm = np.linspace(0.0, 25.0, 120, dtype=float)
    i_wall_start = int(np.searchsorted(t_a, t_corner))
    corner_trace = {
        "ideal_x_mm": [round(float(x_ideal_a[i]) * 1e3, 4) for i in trace_idx],
        "ideal_y_mm": [round(float(y_ideal_a[i]) * 1e3, 4) for i in trace_idx],
        "shaped_cmd_x_mm": [round(float(r_shaped_a[i, 0]) * 1e3, 4) for i in trace_idx],
        "shaped_cmd_y_mm": [round(float(r_shaped_a[i, 1]) * 1e3, 4) for i in trace_idx],
        "actual_x_mm": [round(float(x_sh[i]) * 1e3, 4) for i in trace_idx],
        "actual_y_mm": [round(float(y_sh[i]) * 1e3, 4) for i in trace_idx],
        "unshaped_x_mm": [round(float(x_un[i]) * 1e3, 4) for i in trace_idx],
        "unshaped_y_mm": [round(float(y_un[i]) * 1e3, 4) for i in trace_idx],
        "wall_dist_mm": [round(float(d), 3) for d in wall_d_mm],
        "wall_dev_unshaped_um": _sample_wall_dev_um(y_un, x_un, i_wall_start, wall_d_mm),
        "wall_dev_shaped_um": _sample_wall_dev_um(y_sh, x_sh, i_wall_start, wall_d_mm),
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

    # Per-axis Stribeck rail friction + ISO 5753 bearing radial clearance + CTE & belt tooth backlash
    rail_qual = resolve_rail_quality(spec)
    rail_info_x = hertzian_rail_carriage_stiffness(
        spec.gantry.x_rail_profile,
        preload=spec.gantry.rail_preload,
        block_type=spec.gantry.rail_block_type,
        rail_quality=spec.gantry.rail_quality,
        preload_fraction_override=spec.model_overrides.rail_preload.preload_fraction,
        stiffness_mult_override=rail_qual["stiffness_mult"],
        drag_mult_override=rail_qual["drag_mult"],
    )
    rail_info_y = hertzian_rail_carriage_stiffness(
        spec.gantry.y_rail_profile,
        preload=spec.gantry.rail_preload,
        block_type=spec.gantry.rail_block_type,
        rail_quality=spec.gantry.rail_quality,
        preload_fraction_override=spec.model_overrides.rail_preload.preload_fraction,
        stiffness_mult_override=rail_qual["stiffness_mult"],
        drag_mult_override=rail_qual["drag_mult"],
    )
    n_x_blocks = 2.0 if (spec.gantry.dual_x_rails or spec.gantry.x_rail_mount == "dual_top_bottom") else 1.0
    n_y_blocks = 2.0 if spec.architecture in ("corexy", "cartesian_bedslinger") else 1.0
    f_coulomb_x = n_x_blocks * float(rail_info_x["rolling_friction_n"] + rail_info_x["seal_drag_n"])
    f_break_x = n_x_blocks * float(rail_info_x["breakaway_friction_n"])
    f_coulomb_y = n_y_blocks * float(rail_info_y["rolling_friction_n"] + rail_info_y["seal_drag_n"])
    f_break_y = n_y_blocks * float(rail_info_y["breakaway_friction_n"])
    v_str = 0.008
    v_eps = 0.0015

    bkt_mat = get_material(spec.gantry.bracket_material)
    e_bkt_eff = bkt_mat.effective_modulus_pa(spec.frame.chamber_temp_c)
    cte_play_um, _ = compute_bearing_cte_clearance_um(spec, spec.frame.chamber_temp_c)
    idler_info = idler_shaft_and_bearing_stiffness(
        spec.gantry.belt,
        double_shear=spec.gantry.double_shear_pulleys,
        bracket_e_pa=e_bkt_eff,
        radial_play_um_override=spec.model_overrides.bearing_clearance.radial_play_um,
        thermal_clearance_um=cte_play_um,
    )
    bearing_play_um = float(idler_info["bearing_play_um"])
    tooth_backlash_um = _belt_tooth_backlash_um(spec)
    reversal_backlash_um = bearing_play_um + tooth_backlash_um
    half_play_xy_m = 0.5 * bearing_play_um * 1e-6
    half_play_ab_m = (
        0.5 * (0.35 * bearing_play_um + tooth_backlash_um) * 1e-6
        if spec.architecture == "corexy"
        else 0.0
    )
    if spec.architecture != "corexy":
        half_play_xy_m += 0.5 * tooth_backlash_um * 1e-6

    if is_foc and cfg.mode_space_torque_ff:
        # Mode-space torque FF + AWD direction_split compensates quadrant reversal stiction & play
        f_coulomb_x *= 0.55
        f_break_x *= 0.55
        f_coulomb_y *= 0.55
        f_break_y *= 0.55
        half_play_xy_m *= 0.70
        half_play_ab_m *= 0.70

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

    # Precompute contouring lag + ISO 5753 bearing play (Cartesian X/Y + CoreXY A/B belt loops)
    u_trk_x = float(u_start[0])
    u_trk_y = float(u_start[1])
    bx_play = u_trk_x
    by_play = u_trk_y - half_play_xy_m
    ba_play = bx_play + by_play
    bb_play = bx_play - by_play

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
    use_ab_play = half_play_ab_m > 0.0

    for k in range(n_b):
        cur_sh_x = float(u_sh_x[k])
        cur_sh_y = float(u_sh_y[k])
        u_trk_x = one_m_alpha * u_trk_x + alpha_trk * cur_sh_x
        u_trk_y = one_m_alpha * u_trk_y + alpha_trk * cur_sh_y
        if u_trk_x - bx_play > half_play_xy_m:
            bx_play = u_trk_x - half_play_xy_m
        elif bx_play - u_trk_x > half_play_xy_m:
            bx_play = u_trk_x + half_play_xy_m
        if u_trk_y - by_play > half_play_xy_m:
            by_play = u_trk_y - half_play_xy_m
        elif by_play - u_trk_y > half_play_xy_m:
            by_play = u_trk_y + half_play_xy_m
        if use_ab_play:
            # CoreXY A/B belt-loop reversal play (reverses at 45°/135°/225°/315°)
            ua_in = bx_play + by_play
            ub_in = bx_play - by_play
            if ua_in - ba_play > half_play_ab_m:
                ba_play = ua_in - half_play_ab_m
            elif ba_play - ua_in > half_play_ab_m:
                ba_play = ua_in + half_play_ab_m
            if ub_in - bb_play > half_play_ab_m:
                bb_play = ub_in - half_play_ab_m
            elif bb_play - ub_in > half_play_ab_m:
                bb_play = ub_in + half_play_ab_m
            px_out = 0.5 * (ba_play + bb_play)
            py_out = 0.5 * (ba_play - bb_play)
        else:
            px_out = bx_play
            py_out = by_play
        if tau_vff_foc > 0.0:
            vx_ff = (cur_sh_x - prev_sh_x) / dt if k > 0 else v_ball
            vy_ff = (cur_sh_y - prev_sh_y) / dt if k > 0 else 0.0
            u_eff_arr[k, 0] = px_out + tau_vff_foc * vx_ff
            u_eff_arr[k, 1] = py_out + tau_vff_foc * vy_ff
        else:
            u_eff_arr[k, 0] = px_out
            u_eff_arr[k, 1] = py_out
        prev_sh_x = cur_sh_x
        prev_sh_y = cur_sh_y

    u_drive_all = u_eff_arr @ b_t
    sec_all = ((phi_b % two_pi) * inv_5deg).astype(int) % 72

    xy_actual = np.empty((n_b, 2), dtype=float)
    f_dist_vec = np.zeros(2, dtype=float)
    is_corexy = spec.architecture == "corexy"
    is_awd = is_corexy and coupled.n_mot == 4
    f_delta_x = f_break_x - f_coulomb_x
    f_delta_y = f_break_y - f_coulomb_y
    m_exp = math.exp
    m_tanh = math.tanh
    inv_v_str = 1.0 / v_str
    inv_v_eps = 1.0 / v_eps

    for k in range(n_b):
        xy_actual[k] = x_state @ c_t

        vx = float(x_state[idx_vx])
        vy = float(x_state[idx_vy])
        fx_fric = -(f_coulomb_x + f_delta_x * m_exp(-((vx * inv_v_str) ** 2))) * m_tanh(vx * inv_v_eps)
        fy_fric = -(f_coulomb_y + f_delta_y * m_exp(-((vy * inv_v_str) ** 2))) * m_tanh(vy * inv_v_eps)

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

    # Measure 45° diagonal reversal glitch (CoreXY A/B belt & idler reversals at 45°, 135°, 225°, 315°)
    diag_glitches: list[float] = []
    for d_deg in (45.0, 135.0, 225.0, 315.0):
        d_q = ((phi_rev - d_deg + 180.0) % 360.0) - 180.0
        win_spike = (d_q >= -8.0) & (d_q <= 16.0)
        win_base = ((d_q >= -24.0) & (d_q < -10.0)) | ((d_q > 18.0) & (d_q <= 28.0))
        if np.any(win_spike) and np.any(win_base):
            base_val = float(np.median(e_r_um[win_base]))
            spike_val = float(np.max(np.abs(e_r_um[win_spike] - base_val)))
            diag_glitches.append(spike_val)
    diagonal_glitch_45deg_um = max(diag_glitches) if diag_glitches else 0.0

    # =========================================================================
    # Benchmark C: Multi-Segment Zig-Zag S-Curve (Klipper SCV Junction Deviation)
    # =========================================================================
    zz_pts = (
        np.array(
            [
                [-15.0, 0.0],
                [0.0, 0.0],
                [6.0, 1.2],
                [12.0, -1.2],
                [18.0, 1.2],
                [24.0, -0.8],
                [30.0, 0.0],
                [50.0, 0.0],
            ],
            dtype=float,
        )
        * 1e-3
    )
    n_seg = len(zz_pts) - 1
    seg_vecs = zz_pts[1:] - zz_pts[:-1]
    seg_lens = np.hypot(seg_vecs[:, 0], seg_vecs[:, 1])
    seg_dirs = seg_vecs / np.maximum(seg_lens[:, None], 1e-12)

    # Klipper SCV junction deviation: R = jd * sin(theta/2) / (1 - sin(theta/2)),
    # where jd = v_scv^2 * (sqrt(2) - 1) / a_c recovers v_j = v_scv at a 90° corner.
    jd = (v_scv * v_scv) * (math.sqrt(2.0) - 1.0) / max(a_c, 1e-6)
    v_node = np.zeros(len(zz_pts), dtype=float)
    for idx_v in range(1, n_seg):
        cos_theta = float(np.clip(-np.dot(seg_dirs[idx_v - 1], seg_dirs[idx_v]), -1.0, 1.0))
        sin_half = math.sqrt(max(0.0, 0.5 * (1.0 - cos_theta)))
        if sin_half >= 0.999999:
            v_node[idx_v] = v_c
        elif sin_half <= 1e-9:
            v_node[idx_v] = 0.0
        else:
            r_arc = jd * sin_half / (1.0 - sin_half)
            v_node[idx_v] = min(v_c, math.sqrt(max(0.0, a_c * r_arc)))

    # Forward & backward lookahead velocity pass
    for idx_s in range(n_seg):
        v_max_reach = math.sqrt(v_node[idx_s] ** 2 + 2.0 * a_c * float(seg_lens[idx_s]))
        v_node[idx_s + 1] = min(v_node[idx_s + 1], v_max_reach)
    for idx_s in range(n_seg - 1, -1, -1):
        v_max_reach = math.sqrt(v_node[idx_s + 1] ** 2 + 2.0 * a_c * float(seg_lens[idx_s]))
        v_node[idx_s] = min(v_node[idx_s], v_max_reach)

    seg_params = [
        _trapezoid_params(
            float(seg_lens[s]),
            float(v_node[s]),
            v_c,
            float(v_node[s + 1]),
            a_c,
        )
        for s in range(n_seg)
    ]
    t_seg_starts = np.zeros(n_seg, dtype=float)
    t_seg_ends = np.zeros(n_seg, dtype=float)
    t_cur = 0.0
    for s in range(n_seg):
        t_seg_starts[s] = t_cur
        t_cur += float(seg_params[s][-1])
        t_seg_ends[s] = t_cur
    t_motion_c = t_cur
    t_dwell_c = max(0.06, t_shaper_tail + 0.04)
    dt_c = 2.0 * dt  # 1 kHz exact ZOH step doubling (A_dc = A_d^2, B_dc = (A_d + I) B_d)
    t_c = np.arange(0.0, t_motion_c + t_dwell_c + 0.5 * dt_c, dt_c, dtype=float)

    x_ideal_c = np.empty_like(t_c)
    y_ideal_c = np.empty_like(t_c)
    x_ideal_c.fill(float(zz_pts[-1, 0]))
    y_ideal_c.fill(float(zz_pts[-1, 1]))
    for s in range(n_seg):
        t0_s = float(t_seg_starts[s])
        t1_s = float(t_seg_ends[s])
        mask_s = (t_c >= t0_s) & (t_c < t1_s) if s < n_seg - 1 else (t_c >= t0_s) & (t_c <= t1_s)
        if np.any(mask_s):
            frac = _eval_trapezoid(t_c[mask_s] - t0_s, seg_params[s]) / max(float(seg_lens[s]), 1e-12)
            x_ideal_c[mask_s] = float(zz_pts[s, 0]) + frac * float(seg_vecs[s, 0])
            y_ideal_c[mask_s] = float(zz_pts[s, 1]) + frac * float(seg_vecs[s, 1])

    r_ideal_c = np.column_stack([x_ideal_c, y_ideal_c])
    if sh_used == "NONE":
        r_shaped_c = r_ideal_c
    else:
        r_shaped_c = np.column_stack(
            [
                apply_shaper_to_trajectory(t_c, x_ideal_c, shaper_x),
                apply_shaper_to_trajectory(t_c, y_ideal_c, shaper_y),
            ]
        )

    A_dc = A_d @ A_d
    B_dc = (A_d + np.eye(n_state, dtype=float)) @ B_d
    y_unshaped_c = _run_zoh_lti(A_dc, B_dc, coupled.C_noz, coupled.D, r_ideal_c)
    y_shaped_c = (
        y_unshaped_c
        if sh_used == "NONE"
        else _run_zoh_lti(A_dc, B_dc, coupled.C_noz, coupled.D, r_shaped_c)
    )

    dist_actual_c = _dist_to_polyline(y_shaped_c, zz_pts)
    zigzag_path_error_um = float(np.max(dist_actual_c)) * 1e6
    zigzag_shaper_smoothing_um = (
        0.0
        if sh_used == "NONE"
        else float(np.max(_dist_to_polyline(r_shaped_c, zz_pts))) * 1e6
    )

    t_exit_c = float(t_seg_ends[-2])
    mask_zz_un = (t_c >= t_exit_c) & (y_unshaped_c[:, 0] <= 0.048)
    zigzag_unshaped_ringing_um = (
        float(np.max(y_unshaped_c[mask_zz_un, 1]) - np.min(y_unshaped_c[mask_zz_un, 1])) * 1e6
        if np.any(mask_zz_un)
        else 0.0
    )
    mask_zz_sh = (t_c >= t_exit_c + t_shaper_tail) & (y_shaped_c[:, 0] <= 0.048)
    if sh_used == "NONE":
        zigzag_residual_ringing_um = zigzag_unshaped_ringing_um
    elif np.any(mask_zz_sh):
        zigzag_residual_ringing_um = (
            float(np.max(y_shaped_c[mask_zz_sh, 1]) - np.min(y_shaped_c[mask_zz_sh, 1])) * 1e6
        )
    else:
        zigzag_residual_ringing_um = 0.0

    in_zz_win = np.flatnonzero((x_ideal_c >= -0.004) & (x_ideal_c <= 0.042))
    if in_zz_win.size > 0:
        zz_idx = np.unique(in_zz_win[np.linspace(0, in_zz_win.size - 1, min(110, in_zz_win.size), dtype=int)])
    else:
        zz_idx = np.unique(np.linspace(0, len(t_c) - 1, min(110, len(t_c)), dtype=int))
    zigzag_trace = {
        "ideal_x_mm": [round(float(x_ideal_c[i]) * 1e3, 4) for i in zz_idx],
        "ideal_y_mm": [round(float(y_ideal_c[i]) * 1e3, 4) for i in zz_idx],
        "actual_x_mm": [round(float(y_shaped_c[i, 0]) * 1e3, 4) for i in zz_idx],
        "actual_y_mm": [round(float(y_shaped_c[i, 1]) * 1e3, 4) for i in zz_idx],
        "unshaped_x_mm": [round(float(y_unshaped_c[i, 0]) * 1e3, 4) for i in zz_idx],
        "unshaped_y_mm": [round(float(y_unshaped_c[i, 1]) * 1e3, 4) for i in zz_idx],
        "perp_error_um": [round(float(dist_actual_c[i]) * 1e6, 2) for i in zz_idx],
    }

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
        printed_path_error_um=printed_path_error_um,
        shaper_path_smoothing_um=shaper_path_smoothing_um,
        cross_track_error_um=cross_track_error_um,
        along_track_lag_um=along_track_lag_um,
        zigzag_path_error_um=zigzag_path_error_um,
        zigzag_shaper_smoothing_um=zigzag_shaper_smoothing_um,
        zigzag_residual_ringing_um=zigzag_residual_ringing_um,
        zigzag_unshaped_ringing_um=zigzag_unshaped_ringing_um,
        zigzag_trace=zigzag_trace,
        diagonal_glitch_45deg_um=diagonal_glitch_45deg_um,
        reversal_backlash_um=reversal_backlash_um,
        rail_stiction_x_n=f_break_x,
        rail_stiction_y_n=f_break_y,
    )
