"""Lookahead motion planner & multiphysics virtual print quality simulator."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from printsim.fea.klipper_shaper import (
    DEFAULT_SHAPER_DAMPING_RATIO,
    DEFAULT_TARGET_SMOOTHING,
    estimate_shaper,
    get_flatness_toolhead_bell,
    get_shaper_pulses,
    get_shaper_smoothing,
)
from printsim.fea.modal import ModalResult
from printsim.gcode.parser import GCodeMove, parse_gcode
from printsim.geometry.builder import AssemblyModel
from printsim.kinematics.collision import check_trajectory_bounds
from printsim.kinematics.solvers import KinematicsReport
from printsim.thermal.filaments import FILAMENT_CATALOG, get_filament
from printsim.thermal.hotend import HotendThermalReport

RINGING_VISIBLE_AMPLITUDE_UM: float = 15.0  # Physical onset of visible ghosting (~15-20 um)
RINGING_SEVERE_AMPLITUDE_UM: float = 50.0  # Severe ringing reference scale (severity = 1.0)
CORNER_SMOOTHING_LIMIT_UM: float = DEFAULT_TARGET_SMOOTHING * 1e3  # 120 um Klipper smoothing limit


@dataclass
class ToolpathSegmentResult:
    index: int
    layer_index: int
    start_mm: list[float]
    end_mm: list[float]
    is_extrusion: bool
    duration_s: float
    peak_speed_mm_s: float
    volumetric_flow_mm3_s: float
    ringing_severity: float  # physical amplitude / RINGING_SEVERE_AMPLITUDE_UM
    defect: str  # "ok", "travel", "ringing", "smoothing", "underextrusion", "step_skip", "belt_slip", "cooling"
    ringing_amplitude_um: float = 0.0
    corner_smoothing_um: float = 0.0


@dataclass
class VirtualPrintReport:
    total_time_s: float
    total_moves: int
    extrusion_moves: int
    layer_count: int
    filament_used_mm: float
    filament_used_g: float
    avg_print_speed_mm_s: float
    peak_print_speed_mm_s: float
    peak_volumetric_flow_mm3_s: float
    max_ringing_severity: float
    underextrusion_segments: int
    ringing_segments: int
    step_skip_segments: int
    cooling_warning_layers: int
    out_of_bounds_issues: list[str]
    print_quality_score: float  # 0..100
    segments: list[ToolpathSegmentResult] = field(default_factory=list)
    max_ringing_amplitude_um: float = 0.0
    max_corner_smoothing_um: float = 0.0
    smoothing_segments: int = 0
    belt_slip_segments: int = 0

    def to_dict(self, max_segments: int = 1500) -> dict:
        # Downsample very large G-code toolpaths for responsive JSON API responses
        segs = self.segments
        if max_segments <= 0:
            segs = []
        elif len(segs) > max_segments:
            stride = max(1, len(segs) // max_segments)
            segs = segs[::stride]

        return {
            "total_time_s": round(self.total_time_s, 1),
            "total_moves": self.total_moves,
            "extrusion_moves": self.extrusion_moves,
            "layer_count": self.layer_count,
            "filament_used_mm": round(self.filament_used_mm, 1),
            "filament_used_g": round(self.filament_used_g, 2),
            "avg_print_speed_mm_s": round(self.avg_print_speed_mm_s, 1),
            "peak_print_speed_mm_s": round(self.peak_print_speed_mm_s, 1),
            "peak_volumetric_flow_mm3_s": round(self.peak_volumetric_flow_mm3_s, 2),
            "max_ringing_severity": round(self.max_ringing_severity, 3),
            "max_ringing_amplitude_um": round(self.max_ringing_amplitude_um, 2),
            "max_corner_smoothing_um": round(self.max_corner_smoothing_um, 2),
            "underextrusion_segments": self.underextrusion_segments,
            "ringing_segments": self.ringing_segments,
            "smoothing_segments": self.smoothing_segments,
            "step_skip_segments": self.step_skip_segments,
            "belt_slip_segments": self.belt_slip_segments,
            "cooling_warning_layers": self.cooling_warning_layers,
            "out_of_bounds_issues": self.out_of_bounds_issues,
            "print_quality_score": round(self.print_quality_score, 1),
            "segments": [
                {
                    "index": s.index,
                    "layer_index": s.layer_index,
                    "start_mm": [round(v, 2) for v in s.start_mm],
                    "end_mm": [round(v, 2) for v in s.end_mm],
                    "is_extrusion": s.is_extrusion,
                    "peak_speed_mm_s": round(s.peak_speed_mm_s, 1),
                    "volumetric_flow_mm3_s": round(s.volumetric_flow_mm3_s, 2),
                    "ringing_severity": round(s.ringing_severity, 3),
                    "ringing_amplitude_um": round(s.ringing_amplitude_um, 2),
                    "corner_smoothing_um": round(s.corner_smoothing_um, 2),
                    "defect": s.defect,
                }
                for s in segs
            ],
        }


def _trapezoidal_segment_time(
    dist: float,
    v_entry: float,
    v_cruise: float,
    v_exit: float,
    accel: float,
    minimum_cruise_ratio: float = 0.0,
) -> tuple[float, float]:
    """Compute exact trapezoidal/triangular segment duration and peak velocity.

    Supports Klipper's ``minimum_cruise_ratio`` (0.0 = pure trapezoidal, 0.5 = Klipper default),
    limiting peak velocity on short segments so at least ``minimum_cruise_ratio`` of the
    acceleration-change distance is spent cruising.
    """
    if dist <= 1e-6 or accel <= 1e-6:
        return 0.0, v_cruise

    mcr = float(np.clip(minimum_cruise_ratio, 0.0, 0.95))
    # Peak velocity achievable given entry/exit speeds and segment distance
    eff_accel = accel * (1.0 - mcr) if mcr > 0.0 else accel
    v_tri_sq = 0.5 * (v_entry**2 + v_exit**2) + max(eff_accel, 1e-6) * dist
    v_peak = min(v_cruise, math.sqrt(max(v_tri_sq, 0.0)))
    v_peak = max(v_peak, min(v_entry, v_exit))

    d_acc = max(0.0, (v_peak**2 - v_entry**2) / (2.0 * accel))
    d_dec = max(0.0, (v_peak**2 - v_exit**2) / (2.0 * accel))
    d_flat = max(0.0, dist - d_acc - d_dec)

    t_acc = abs(v_peak - v_entry) / accel
    t_dec = abs(v_peak - v_exit) / accel
    t_flat = d_flat / max(v_peak, 1e-6)
    return t_acc + t_flat + t_dec, v_peak


def _klipper_junction_speed_mm_s(
    d_prev: np.ndarray,
    d_next: np.ndarray,
    len_prev: float,
    len_next: float,
    v_prev: float,
    v_next: float,
    accel_j: float,
    scv: float,
) -> float:
    """Compute exact Klipper toolhead.py centripetal junction_deviation corner speed.

    Let theta be the interior angle between incoming (-d_prev) and outgoing (+d_next) moves,
    so cos(theta) = -dot(d_prev, d_next) (theta=pi straight, pi/2 for 90 deg, 0 for 180 deg reversal).
    """
    if len_prev <= 1e-6 or len_next <= 1e-6:
        return 0.0

    jcos_theta = float(np.clip(-np.dot(d_prev, d_next), -1.0, 1.0))
    sin_theta_d2 = math.sqrt(max(0.5 * (1.0 - jcos_theta), 0.0))
    cos_theta_d2 = math.sqrt(max(0.5 * (1.0 + jcos_theta), 0.0))

    if sin_theta_d2 >= 1.0 - 1e-6:
        return min(v_prev, v_next)
    if sin_theta_d2 <= 1e-9:
        return 0.0

    a_j = max(accel_j, 1e-6)
    junction_dev = (scv**2) * (math.sqrt(2.0) - 1.0) / a_j
    v_j_sq = a_j * junction_dev * (sin_theta_d2 / (1.0 - sin_theta_d2))
    if cos_theta_d2 > 1e-6:
        tan_theta_d2 = sin_theta_d2 / cos_theta_d2
        v_j_sq = min(
            v_j_sq,
            0.5 * len_prev * tan_theta_d2 * a_j,
            0.5 * len_next * tan_theta_d2 * a_j,
        )
    return min(v_prev, v_next, math.sqrt(max(v_j_sq, 0.0)))


def _resolve_axis_shaper_response(
    axis_rec: object | None,
    axis_graph: object | None,
    shaper_choice: str,
    fallback_freq_hz: float,
    fallback_zeta: float,
) -> tuple[float, float, float, tuple[np.ndarray, np.ndarray] | None]:
    """Return `(f_n_hz, zeta, V_shaper(f_n, zeta), smoothing_shaper)` for one XY axis."""
    eff_f = getattr(axis_rec, "effective_freq_hz", 0.0) if axis_rec is not None else 0.0
    f_n = float(eff_f) if eff_f and eff_f > 0.0 else max(float(fallback_freq_hz), 5.0)

    graph_zeta = getattr(axis_graph, "damping_ratio", 0.0) if axis_graph is not None else 0.0
    zeta_raw = float(graph_zeta) if graph_zeta and graph_zeta > 0.0 else float(fallback_zeta)
    zeta = min(0.25, max(0.02, zeta_raw))

    choice_norm = shaper_choice.strip().upper()
    rec_shaper = getattr(axis_rec, "recommended_shaper", "") if axis_rec is not None else ""
    if choice_norm == "NONE":
        sh_name = "NONE"
    elif rec_shaper:
        sh_name = str(rec_shaper).strip().upper()
    elif choice_norm == "AUTO":
        sh_name = "MZV"
    else:
        sh_name = choice_norm

    if sh_name == "NONE":
        return f_n, zeta, 1.0, None

    rec_freq = getattr(axis_graph, "recommended_freq_hz", 0.0) if axis_graph is not None else 0.0
    sh_freq = float(rec_freq) if rec_freq and rec_freq > 0.0 else f_n
    sh_pulses = get_shaper_pulses(sh_name, sh_freq, damping_ratio=DEFAULT_SHAPER_DAMPING_RATIO)
    v_shaper = float(estimate_shaper(sh_pulses, [f_n], test_damping_ratio=zeta)[0])
    smooth_shaper = get_flatness_toolhead_bell(sh_freq) if sh_name == "SMOOTH_FLATNESS" else sh_pulses
    return f_n, zeta, v_shaper, smooth_shaper


def simulate_gcode(
    gcode_text: str,
    assembly: AssemblyModel,
    modal: ModalResult,
    kinematics: KinematicsReport,
    thermal: HotendThermalReport,
    minimum_cruise_ratio: float = 0.0,
) -> VirtualPrintReport:
    """Execute a virtual print of G-code through lookahead motion & multiphysics defect models."""
    moves = parse_gcode(gcode_text)
    if not moves:
        return VirtualPrintReport(
            total_time_s=0.0,
            total_moves=0,
            extrusion_moves=0,
            layer_count=0,
            filament_used_mm=0.0,
            filament_used_g=0.0,
            avg_print_speed_mm_s=0.0,
            peak_print_speed_mm_s=0.0,
            peak_volumetric_flow_mm3_s=0.0,
            max_ringing_severity=0.0,
            underextrusion_segments=0,
            ringing_segments=0,
            step_skip_segments=0,
            cooling_warning_layers=0,
            out_of_bounds_issues=[],
            print_quality_score=100.0,
            segments=[],
            max_ringing_amplitude_um=0.0,
            max_corner_smoothing_um=0.0,
            smoothing_segments=0,
            belt_slip_segments=0,
        )

    spec = assembly.spec
    kin = spec.kinematics
    fil_area = math.pi * (spec.toolhead.extruder.filament_diameter_mm * 0.5) ** 2

    # Check bounds on all move endpoints
    pts = np.array([m.end_mm for m in moves], dtype=float)
    oob_issues = check_trajectory_bounds(assembly, pts)

    # Precompute unit direction vectors and junction corner speeds (Klipper square_corner_velocity)
    n = len(moves)
    dirs = np.zeros((n, 3), dtype=float)
    lengths = np.zeros(n, dtype=float)
    v_targets = np.zeros(n, dtype=float)
    accels = np.zeros(n, dtype=float)

    for i, m in enumerate(moves):
        d_vec = np.array(m.end_mm) - np.array(m.start_mm)
        dist = float(np.linalg.norm(d_vec))
        lengths[i] = dist
        if dist > 1e-6:
            dirs[i] = d_vec / dist
            # Clamp requested feedrate to machine max velocity (Z-only moves clamped to max_z_vel)
            is_z_only = abs(dirs[i, 2]) > 0.9
            v_cap = kin.max_z_velocity_mm_s if is_z_only else kin.max_velocity_mm_s
            v_targets[i] = min(m.feedrate_mm_s, v_cap)
            a_base = m.accel_override_mm_s2 if m.accel_override_mm_s2 else kin.max_accel_mm_s2
            accels[i] = min(a_base, kin.max_z_accel_mm_s2 if is_z_only else kin.max_accel_mm_s2)
        else:
            # Zero-length E-only move (retract / unretract) stops XY motion at both ends
            v_targets[i] = 0.0
            accels[i] = kin.max_accel_mm_s2

    # Junction speed calculation between consecutive moves
    scv = kin.square_corner_velocity_mm_s
    junction_v = np.zeros(n + 1, dtype=float)
    corner_angles_deg = np.zeros(n, dtype=float)
    for i in range(1, n):
        if lengths[i - 1] <= 1e-6 or lengths[i] <= 1e-6:
            corner_angles_deg[i] = 0.0
            junction_v[i] = 0.0
            continue
        cos_turn = float(np.clip(np.dot(dirs[i - 1], dirs[i]), -1.0, 1.0))
        corner_angles_deg[i] = math.degrees(math.acos(cos_turn))
        junction_v[i] = _klipper_junction_speed_mm_s(
            d_prev=dirs[i - 1],
            d_next=dirs[i],
            len_prev=lengths[i - 1],
            len_next=lengths[i],
            v_prev=v_targets[i - 1],
            v_next=v_targets[i],
            accel_j=min(accels[i - 1], accels[i]),
            scv=scv,
        )

    # Backward + forward lookahead pass so segments can decelerate/accelerate smoothly
    for i in range(n - 1, 0, -1):
        if lengths[i] <= 1e-6:
            junction_v[i] = 0.0
            junction_v[i + 1] = 0.0
            continue
        v_max_entry = math.sqrt(junction_v[i + 1] ** 2 + 2.0 * accels[i] * lengths[i])
        junction_v[i] = min(junction_v[i], v_max_entry)
    for i in range(n):
        if lengths[i] <= 1e-6:
            junction_v[i] = 0.0
            junction_v[i + 1] = 0.0
            continue
        v_max_exit = math.sqrt(junction_v[i] ** 2 + 2.0 * accels[i] * lengths[i])
        junction_v[i + 1] = min(junction_v[i + 1], v_max_exit)

    # Precompute segment durations and actual peak velocities before junction impulse evaluation
    durations = np.zeros(n, dtype=float)
    v_peaks = np.zeros(n, dtype=float)
    for i, m in enumerate(moves):
        if lengths[i] <= 1e-6:
            durations[i] = (
                abs(m.extrude_mm) / max(m.feedrate_mm_s, 1e-6)
                if abs(m.extrude_mm) > 1e-6
                else 0.0
            )
            v_peaks[i] = 0.0
        else:
            durations[i], v_peaks[i] = _trapezoidal_segment_time(
                lengths[i],
                junction_v[i],
                v_targets[i],
                junction_v[i + 1],
                accels[i],
                minimum_cruise_ratio=minimum_cruise_ratio,
            )

    # Physical modal & input-shaper response per axis for junction ringing + corner smoothing
    fallback_f = max(float(modal.fundamental_freq_hz), 5.0)
    fallback_zeta = max(float(modal.frame_damping_ratio), 0.02)
    f_nx, zeta_x, v_sh_x, smooth_sh_x = _resolve_axis_shaper_response(
        modal.shaper_x, modal.shaper_graph_x, kin.input_shaper, fallback_f, fallback_zeta
    )
    f_ny, zeta_y, v_sh_y, smooth_sh_y = _resolve_axis_shaper_response(
        modal.shaper_y, modal.shaper_graph_y, kin.input_shaper, fallback_f, fallback_zeta
    )
    # CoreXY A/B belt tension mismatch couples X and Y modes; the primary shaper tuned to f_nx
    # imperfectly cancels cross-coupled excitation at f_ny (and vice versa).
    mismatch_pct = (
        max(0.0, float(spec.gantry.belt.tension_mismatch_pct))
        if spec.architecture == "corexy"
        else 0.0
    )
    eta_xy_cross = min(0.45, 0.008 * mismatch_pct)
    v_cross_x_to_y = (
        float(estimate_shaper(smooth_sh_x, [f_ny], test_damping_ratio=zeta_y)[0])
        if (eta_xy_cross > 0.0 and smooth_sh_x is not None)
        else 1.0
    )
    v_cross_y_to_x = (
        float(estimate_shaper(smooth_sh_y, [f_nx], test_damping_ratio=zeta_x)[0])
        if (eta_xy_cross > 0.0 and smooth_sh_y is not None)
        else 1.0
    )

    corner_smoothing_um_by_accel: dict[float, tuple[float, float]] = {}

    max_motor_accel = kinematics.max_safe_accel_motor_mm_s2
    belt_max_accel = kinematics.belt_max_accel_mm_s2
    max_flow = thermal.max_volumetric_flow_mm3_s

    segments: list[ToolpathSegmentResult] = []
    total_time = 0.0
    total_fil_mm = 0.0
    ext_dist = 0.0
    ext_time = 0.0
    peak_speed = 0.0
    peak_flow = 0.0
    max_ringing = 0.0
    max_ringing_um = 0.0
    max_smoothing_um = 0.0
    underext_count = 0
    ringing_count = 0
    smoothing_count = 0
    step_skip_count = 0
    belt_slip_count = 0
    motion_fault_count = 0
    layer_times: dict[int, float] = {}

    for i, m in enumerate(moves):
        dt = float(durations[i])
        v_peak = float(v_peaks[i])
        total_time += dt
        layer_times[m.layer_index] = layer_times.get(m.layer_index, 0.0) + dt

        # Volumetric flow (mm^3/s) & junction ringing / smoothing (skip zero-length E-only moves)
        is_xy_ext = m.is_extrusion and lengths[i] > 1e-6
        vol_flow = (m.extrude_mm * fil_area) / max(dt, 1e-4) if is_xy_ext else 0.0

        amp_um = 0.0
        ringing_sev = 0.0
        corner_smooth_um = 0.0
        if is_xy_ext:
            ext_dist += lengths[i]
            ext_time += dt
            total_fil_mm += m.extrude_mm
            peak_speed = max(peak_speed, v_peak)
            peak_flow = max(peak_flow, vol_flow)

            v_in_xy = (
                v_peaks[i - 1] * dirs[i - 1, :2]
                if i > 0 and lengths[i - 1] > 1e-6
                else np.zeros(2, dtype=float)
            )
            v_out_xy = v_peak * dirs[i, :2]
            dv_xy = v_out_xy - v_in_xy
            dv_mm_s = float(np.linalg.norm(dv_xy))

            if dv_mm_s > 1e-6:
                wx = abs(float(dv_xy[0]))
                wy = abs(float(dv_xy[1]))
                if wx + wy <= 1e-9:
                    wx = abs(float(dirs[i, 0]))
                    wy = abs(float(dirs[i, 1]))
                w_sum = max(wx + wy, 1e-9)
                f_n = (wx * f_nx + wy * f_ny) / w_sum
                v_shaper = (wx * v_sh_x + wy * v_sh_y) / w_sum
                omega_n = 2.0 * math.pi * f_n
                dv_m_s = dv_mm_s * 1e-3
                a_m_s2 = float(accels[i]) * 1e-3
                amp_direct_um = min(dv_m_s / omega_n, a_m_s2 / (omega_n * omega_n)) * v_shaper * 1e6
                if eta_xy_cross > 0.0:
                    # Cross-axis CoreXY A/B mismatch beat-frequency ringing (RSS with direct mode)
                    omega_nx = 2.0 * math.pi * f_nx
                    omega_ny = 2.0 * math.pi * f_ny
                    dv_x_m = abs(float(dv_xy[0])) * 1e-3
                    dv_y_m = abs(float(dv_xy[1])) * 1e-3
                    amp_cxy = eta_xy_cross * min(dv_x_m / omega_ny, a_m_s2 / (omega_ny * omega_ny)) * v_cross_x_to_y * 1e6
                    amp_cyx = eta_xy_cross * min(dv_y_m / omega_nx, a_m_s2 / (omega_nx * omega_nx)) * v_cross_y_to_x * 1e6
                    amp_um = math.sqrt(amp_direct_um**2 + amp_cxy**2 + amp_cyx**2)
                else:
                    amp_um = amp_direct_um
                ringing_sev = amp_um / RINGING_SEVERE_AMPLITUDE_UM
                max_ringing_um = max(max_ringing_um, amp_um)
                max_ringing = max(max_ringing, ringing_sev)

                if corner_angles_deg[i] >= 15.0:
                    # Corner smoothing depends only on (axis shaper, accel); G-code uses a handful
                    # of distinct accelerations, so memoize instead of re-evaluating per corner.
                    accel_i = float(accels[i])
                    sm_pair = corner_smoothing_um_by_accel.get(accel_i)
                    if sm_pair is None:
                        sm_pair = (
                            get_shaper_smoothing(smooth_sh_x, accel=accel_i, scv=scv) * 1e3
                            if smooth_sh_x is not None
                            else 0.0,
                            get_shaper_smoothing(smooth_sh_y, accel=accel_i, scv=scv) * 1e3
                            if smooth_sh_y is not None
                            else 0.0,
                        )
                        corner_smoothing_um_by_accel[accel_i] = sm_pair
                    sm_x_um, sm_y_um = sm_pair
                    # Geometric turn-angle scaling: 1.0 at 90 deg, smaller for shallow bends, up to sqrt(2) at 180 deg
                    angle_scale = math.sin(0.5 * math.radians(float(corner_angles_deg[i]))) / math.sin(0.25 * math.pi)
                    corner_smooth_um = ((wx * sm_x_um + wy * sm_y_um) / w_sum) * angle_scale
                    max_smoothing_um = max(max_smoothing_um, corner_smooth_um)

        # Classify primary defect for segment heatmap
        defect = "travel" if not is_xy_ext else "ok"
        if is_xy_ext:
            exceeds_motor = max_motor_accel > 0.0 and accels[i] > max_motor_accel * 1.15
            exceeds_belt = (belt_max_accel > 0.0 or kinematics.belt_slip_risk) and accels[i] > belt_max_accel
            if exceeds_motor:
                step_skip_count += 1
            if exceeds_belt:
                belt_slip_count += 1
            if exceeds_motor or exceeds_belt:
                motion_fault_count += 1

            if exceeds_motor:
                defect = "step_skip"
            elif exceeds_belt:
                defect = "belt_slip"
            elif vol_flow > max_flow * 1.03:
                defect = "underextrusion"
                underext_count += 1
            elif amp_um >= RINGING_VISIBLE_AMPLITUDE_UM:
                defect = "ringing"
                ringing_count += 1
            elif corner_smooth_um > CORNER_SMOOTHING_LIMIT_UM:
                defect = "smoothing"
                smoothing_count += 1

        segments.append(
            ToolpathSegmentResult(
                index=m.index,
                layer_index=m.layer_index,
                start_mm=list(m.start_mm),
                end_mm=list(m.end_mm),
                is_extrusion=m.is_extrusion,
                duration_s=dt,
                peak_speed_mm_s=v_peak,
                volumetric_flow_mm3_s=vol_flow,
                ringing_severity=ringing_sev,
                defect=defect,
                ringing_amplitude_um=amp_um,
                corner_smoothing_um=corner_smooth_um,
            )
        )

    # Part cooling check: min layer time threshold improves with higher cooling fan CFM
    min_safe_layer_s = max(2.0, 12.0 - 0.75 * spec.toolhead.hotend.part_cooling_cfm)
    cooling_warn_layers = sum(1 for t in layer_times.values() if t < min_safe_layer_s)

    n_ext = max(1, sum(1 for m in moves if m.is_extrusion))
    underext_frac = underext_count / n_ext
    ringing_frac = ringing_count / n_ext
    smooth_frac = smoothing_count / n_ext
    motion_fault_frac = motion_fault_count / n_ext
    cool_frac = cooling_warn_layers / max(1, len(layer_times))

    # Composite virtual print quality score (0..100)
    penalty = (
        65.0 * motion_fault_frac
        + 45.0 * underext_frac
        + 30.0 * ringing_frac
        + 12.0 * smooth_frac
        + 15.0 * cool_frac
        + (25.0 if oob_issues else 0.0)
        + (20.0 if thermal.heat_creep_risk else 0.0)
    )
    quality = max(0.0, min(100.0, 100.0 - penalty))

    # Filament mass using selected filament density from FILAMENT_CATALOG
    fil_spec = FILAMENT_CATALOG.get(thermal.filament_id) or get_filament("pla")
    fil_density_g_cm3 = fil_spec.density_kg_m3 * 1e-3
    fil_g = (total_fil_mm * fil_area * 1e-3) * fil_density_g_cm3

    return VirtualPrintReport(
        total_time_s=total_time,
        total_moves=len(moves),
        extrusion_moves=sum(1 for m in moves if m.is_extrusion),
        layer_count=len(layer_times),
        filament_used_mm=total_fil_mm,
        filament_used_g=fil_g,
        avg_print_speed_mm_s=ext_dist / max(ext_time, 1e-6),
        peak_print_speed_mm_s=peak_speed,
        peak_volumetric_flow_mm3_s=peak_flow,
        max_ringing_severity=max_ringing,
        underextrusion_segments=underext_count,
        ringing_segments=ringing_count,
        step_skip_segments=step_skip_count,
        cooling_warning_layers=cooling_warn_layers,
        out_of_bounds_issues=oob_issues,
        print_quality_score=quality,
        segments=segments,
        max_ringing_amplitude_um=max_ringing_um,
        max_corner_smoothing_um=max_smoothing_um,
        smoothing_segments=smoothing_count,
        belt_slip_segments=belt_slip_count,
    )
