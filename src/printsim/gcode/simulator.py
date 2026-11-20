"""Lookahead motion planner & multiphysics virtual print quality simulator."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from printsim.fea.modal import ModalResult
from printsim.gcode.parser import GCodeMove, parse_gcode
from printsim.geometry.builder import AssemblyModel
from printsim.kinematics.collision import check_trajectory_bounds
from printsim.kinematics.solvers import KinematicsReport
from printsim.thermal.hotend import HotendThermalReport


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
    ringing_severity: float  # 0..1+
    defect: str  # "ok", "travel", "ringing", "underextrusion", "step_skip", "cooling"


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
            "underextrusion_segments": self.underextrusion_segments,
            "ringing_segments": self.ringing_segments,
            "step_skip_segments": self.step_skip_segments,
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
                    "defect": s.defect,
                }
                for s in segs
            ],
        }


def _trapezoidal_segment_time(
    dist: float, v_entry: float, v_cruise: float, v_exit: float, accel: float
) -> tuple[float, float]:
    """Compute exact trapezoidal/triangular segment duration and peak velocity."""
    if dist <= 1e-6 or accel <= 1e-6:
        return 0.0, v_cruise

    # Peak velocity achievable given entry/exit speeds and segment distance
    v_tri_sq = 0.5 * (v_entry**2 + v_exit**2) + accel * dist
    v_peak = min(v_cruise, math.sqrt(max(v_tri_sq, 0.0)))
    v_peak = max(v_peak, min(v_entry, v_exit))

    d_acc = max(0.0, (v_peak**2 - v_entry**2) / (2.0 * accel))
    d_dec = max(0.0, (v_peak**2 - v_exit**2) / (2.0 * accel))
    d_flat = max(0.0, dist - d_acc - d_dec)

    t_acc = abs(v_peak - v_entry) / accel
    t_dec = abs(v_peak - v_exit) / accel
    t_flat = d_flat / max(v_peak, 1e-6)
    return t_acc + t_flat + t_dec, v_peak


def simulate_gcode(
    gcode_text: str,
    assembly: AssemblyModel,
    modal: ModalResult,
    kinematics: KinematicsReport,
    thermal: HotendThermalReport,
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

    # Junction speed calculation between consecutive moves
    scv = kin.square_corner_velocity_mm_s
    junction_v = np.zeros(n + 1, dtype=float)
    corner_angles_deg = np.zeros(n, dtype=float)
    for i in range(1, n):
        cos_theta = float(np.clip(np.dot(dirs[i - 1], dirs[i]), -1.0, 1.0))
        angle_rad = math.acos(cos_theta)
        corner_angles_deg[i] = math.degrees(angle_rad)
        sin_half = math.sin(angle_rad * 0.5)
        if sin_half < 1e-4:
            v_j = min(v_targets[i - 1], v_targets[i])
        else:
            # Klipper junction speed approximation
            v_j = min(
                v_targets[i - 1],
                v_targets[i],
                scv * math.sqrt(max(1.0 - sin_half, 0.05) / sin_half),
            )
        junction_v[i] = v_j

    # Backward + forward lookahead pass so segments can decelerate/accelerate smoothly
    for i in range(n - 1, 0, -1):
        v_max_entry = math.sqrt(junction_v[i + 1] ** 2 + 2.0 * accels[i] * lengths[i])
        junction_v[i] = min(junction_v[i], v_max_entry)
    for i in range(n):
        v_max_exit = math.sqrt(junction_v[i] ** 2 + 2.0 * accels[i] * lengths[i])
        junction_v[i + 1] = min(junction_v[i + 1], v_max_exit)

    # Effective ringing thresholds from modal FEA
    rec_accel_x = modal.shaper_x.max_accel_mm_s2 if modal.shaper_x else 4000.0
    rec_accel_y = modal.shaper_y.max_accel_mm_s2 if modal.shaper_y else 4000.0
    max_motor_accel = kinematics.max_safe_accel_motor_mm_s2
    max_flow = thermal.max_volumetric_flow_mm3_s

    segments: list[ToolpathSegmentResult] = []
    total_time = 0.0
    total_fil_mm = 0.0
    ext_dist = 0.0
    ext_time = 0.0
    peak_speed = 0.0
    peak_flow = 0.0
    max_ringing = 0.0
    underext_count = 0
    ringing_count = 0
    step_skip_count = 0
    layer_times: dict[int, float] = {}

    for i, m in enumerate(moves):
        dt, v_peak = _trapezoidal_segment_time(
            lengths[i], junction_v[i], v_targets[i], junction_v[i + 1], accels[i]
        )
        total_time += dt
        layer_times[m.layer_index] = layer_times.get(m.layer_index, 0.0) + dt

        # Volumetric flow (mm^3/s)
        vol_flow = (m.extrude_mm * fil_area) / max(dt, 1e-4) if m.is_extrusion else 0.0

        # Directional ringing impulse after corners
        ringing_sev = 0.0
        if m.is_extrusion:
            ext_dist += lengths[i]
            ext_time += dt
            total_fil_mm += m.extrude_mm
            peak_speed = max(peak_speed, v_peak)
            peak_flow = max(peak_flow, vol_flow)

            # Directional allowable accel along this move
            ax_weight = abs(dirs[i, 0])
            ay_weight = abs(dirs[i, 1])
            denom = max(ax_weight + ay_weight, 1e-6)
            allow_a = (ax_weight * rec_accel_x + ay_weight * rec_accel_y) / denom
            corner_factor = min(1.0, corner_angles_deg[i] / 75.0)
            # Short moves after sharp corners show strongest ghosting
            ringing_sev = max(0.0, (accels[i] / max(allow_a, 200.0) - 0.85)) * (0.45 + 0.55 * corner_factor)
            max_ringing = max(max_ringing, ringing_sev)

        # Classify primary defect for segment heatmap
        defect = "travel" if not m.is_extrusion else "ok"
        if m.is_extrusion:
            if accels[i] > max_motor_accel * 1.15:
                defect = "step_skip"
                step_skip_count += 1
            elif vol_flow > max_flow * 1.03:
                defect = "underextrusion"
                underext_count += 1
            elif ringing_sev > 0.35:
                defect = "ringing"
                ringing_count += 1

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
            )
        )

    # Part cooling check: min layer time threshold improves with higher cooling fan CFM
    min_safe_layer_s = max(2.0, 12.0 - 0.75 * spec.toolhead.hotend.part_cooling_cfm)
    cooling_warn_layers = sum(1 for t in layer_times.values() if t < min_safe_layer_s)

    n_ext = max(1, sum(1 for m in moves if m.is_extrusion))
    underext_frac = underext_count / n_ext
    ringing_frac = ringing_count / n_ext
    skip_frac = step_skip_count / n_ext
    cool_frac = cooling_warn_layers / max(1, len(layer_times))

    # Composite virtual print quality score (0..100)
    penalty = (
        65.0 * skip_frac
        + 45.0 * underext_frac
        + 30.0 * ringing_frac
        + 15.0 * cool_frac
        + (25.0 if oob_issues else 0.0)
        + (20.0 if thermal.heat_creep_risk else 0.0)
    )
    quality = max(0.0, min(100.0, 100.0 - penalty))

    # Filament mass (PLA 1.24 g/cm^3)
    fil_g = (total_fil_mm * fil_area * 1e-3) * 1.24

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
    )
