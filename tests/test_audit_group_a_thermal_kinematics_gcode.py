"""Tests for Group A audit fixes: A-03, A-04, A-05, A-06, and A-09."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from printsim.engine import evaluate_printer
from printsim.fea import solve_modal_fea
from printsim.gcode import parse_gcode, simulate_gcode
from printsim.geometry import build_assembly
from printsim.kinematics import (
    analyze_kinematics,
    check_frame_clearance,
    check_trajectory_bounds,
)
from printsim.schema import PrinterSpec
from printsim.thermal.filaments import get_filament
from printsim.thermal.hotend import arrhenius_viscosity_shift, simulate_hotend_thermal

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


# ---------------------------------------------------------------------------
# A-03: Delta circular workspace check & cylindrical trajectory bounds
# ---------------------------------------------------------------------------


def test_a03_delta_circular_workspace_and_cylindrical_bounds() -> None:
    """Delta examples validate inscribed circular workspace with >= 20 deg arms & cylindrical bounds."""
    for preset_name in ("delta_racer.yaml", "flsun_v400_delta.yaml"):
        spec = PrinterSpec.from_yaml(EXAMPLES_DIR / preset_name)
        asm = build_assembly(spec)
        kin = analyze_kinematics(asm)
        assert kin.delta_workspace_valid is True

    # Full evaluation on delta_racer should emit no delta workspace geometry warning
    rep = evaluate_printer(
        PrinterSpec.from_yaml(EXAMPLES_DIR / "delta_racer.yaml"),
        include_advanced_analysis=False,
    )
    assert not any("workspace geometry limit" in d.lower() for d in rep.diagnostics)

    # Shortening arm length so angle drops below 20 deg (< arccos(285/295) = 14.96 deg) fails check
    spec_shallow = PrinterSpec.from_yaml(EXAMPLES_DIR / "delta_racer.yaml")
    spec_shallow.kinematics.delta_arm_length_mm = 295.0
    assert analyze_kinematics(build_assembly(spec_shallow)).delta_workspace_valid is False

    # Shortening arm length inside reach limit (< 285 mm) also fails check
    spec_short = PrinterSpec.from_yaml(EXAMPLES_DIR / "delta_racer.yaml")
    spec_short.kinematics.delta_arm_length_mm = 270.0
    assert analyze_kinematics(build_assembly(spec_short)).delta_workspace_valid is False

    # Cylindrical trajectory bounds check for delta
    spec_delta = PrinterSpec.from_yaml(EXAMPLES_DIR / "delta_racer.yaml")
    asm_delta = build_assembly(spec_delta)
    bx, by, bz = spec_delta.kinematics.build_volume_mm
    cx, cy = 0.5 * bx, 0.5 * by
    r_bed = 0.5 * min(bx, by)

    # Inscribed circle boundary points at z=0 and z=bz must pass
    valid_pts = np.array(
        [
            [cx, cy, 0.0],
            [cx + r_bed, cy, 10.0],
            [cx - r_bed, cy, bz],
            [cx, cy + r_bed, 0.5 * bz],
            [cx, cy - r_bed, 0.5 * bz],
        ]
    )
    assert check_trajectory_bounds(asm_delta, valid_pts) == []

    # Square bounding-box corner (0, 0) has radial reach 169.7 mm > 120 mm -> flagged out of bounds
    corner_pts = np.array([[0.0, 0.0, 10.0], [bx, by, 10.0]])
    issues = check_trajectory_bounds(asm_delta, corner_pts)
    assert any("radial" in issue.lower() for issue in issues)


# ---------------------------------------------------------------------------
# A-04: Delta tower carriage height check in check_frame_clearance
# ---------------------------------------------------------------------------


def test_a04_delta_tower_height_clearance() -> None:
    """Delta frame clearance accounts for max tower carriage height bz + sqrt(L^2 - d_min^2)."""
    racer = PrinterSpec.from_yaml(EXAMPLES_DIR / "delta_racer.yaml")
    col_racer = check_frame_clearance(build_assembly(racer))
    assert col_racer.has_collision is False
    assert 20.0 <= col_racer.clearance_z_mm <= 90.0

    flsun = PrinterSpec.from_yaml(EXAMPLES_DIR / "flsun_v400_delta.yaml")
    col_flsun = check_frame_clearance(build_assembly(flsun))
    assert col_flsun.has_collision is False
    assert 20.0 <= col_flsun.clearance_z_mm <= 90.0

    # Cutting delta_racer frame height to 600 mm cannot accommodate 626.7 mm carriage travel
    racer_short = racer.model_copy(deep=True)
    racer_short.frame.height_mm = 600.0
    col_short = check_frame_clearance(build_assembly(racer_short))
    assert col_short.has_collision is True
    assert col_short.clearance_z_mm < 0.0
    assert any("z interference" in issue.lower() for issue in col_short.issues)


# ---------------------------------------------------------------------------
# A-05: Chamber temperature reaches hotend thermal model
# ---------------------------------------------------------------------------


def test_a05_chamber_temperature_reaches_hotend_thermal() -> None:
    """Heated chamber air temp raises hotend cold-side temp, FVM heatsink profile, and PLA creep risk."""
    spec_25 = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    spec_25.toolhead.hotend.ambient_temp_c = 25.0
    spec_25.frame.chamber_temp_c = 25.0

    spec_65 = spec_25.model_copy(deep=True)
    spec_65.frame.chamber_temp_c = 65.0

    th25_pla = simulate_hotend_thermal(spec_25, filament_id="pla")
    th65_pla = simulate_hotend_thermal(spec_65, filament_id="pla")
    th65_abs = simulate_hotend_thermal(spec_65, filament_id="abs")

    assert th25_pla.heat_creep_risk is False
    assert th65_pla.cold_side_temp_c > th25_pla.cold_side_temp_c + 30.0
    assert th65_pla.heat_creep_risk is True
    assert th65_abs.cold_side_temp_c > 65.0
    assert th65_abs.heat_creep_risk is False
    assert th65_pla.steady_heat_loss_w < th25_pla.steady_heat_loss_w

    # 1D FVM axial profile heatsink cells also reflect the 65 C enclosure air
    hs_25 = [p["temp_c"] for p in th25_pla.axial_profile if p["zone"] == "heatsink"]
    hs_65 = [p["temp_c"] for p in th65_pla.axial_profile if p["zone"] == "heatsink"]
    assert min(hs_65) >= 65.0
    assert min(hs_65) > max(hs_25)


# ---------------------------------------------------------------------------
# A-06: Arrhenius TTS shift K(T) = K_ref * a_T^n
# ---------------------------------------------------------------------------


def test_a06_arrhenius_shift_power_law_exponent() -> None:
    """Pressure-limited flow scales as Q(T)/Q(T_ref) = 1 / a_T(T) from K(T) = K_ref * a_T^n."""
    spec = PrinterSpec()
    fil = get_filament("pla")
    t_ref = fil.t_ref_c

    q_ref = simulate_hotend_thermal(
        spec, target_temp_c=t_ref, filament_id="pla"
    ).extruder_pressure_flow_limit_mm3_s
    q_p20 = simulate_hotend_thermal(
        spec, target_temp_c=t_ref + 20.0, filament_id="pla"
    ).extruder_pressure_flow_limit_mm3_s
    q_m20 = simulate_hotend_thermal(
        spec, target_temp_c=t_ref - 20.0, filament_id="pla"
    ).extruder_pressure_flow_limit_mm3_s

    a_p20 = arrhenius_viscosity_shift(t_ref + 20.0, t_ref, fil.arrhenius_ea_j_mol)
    a_m20 = arrhenius_viscosity_shift(t_ref - 20.0, t_ref, fil.arrhenius_ea_j_mol)

    assert (q_p20 / q_ref) == pytest.approx(1.0 / a_p20, rel=1e-4)
    assert (q_m20 / q_ref) == pytest.approx(1.0 / a_m20, rel=1e-4)
    # Guard against old K * a_T bug which produced ~8.6x at +20 C instead of ~2.17x
    assert 1.8 < (q_p20 / q_ref) < 2.6


# ---------------------------------------------------------------------------
# A-09: G-code parser G2/G3 arcs, E-only moves, and simulator planning
# ---------------------------------------------------------------------------


def test_a09_gcode_parser_arcs_and_e_only_moves() -> None:
    """Parser tessellates G2/G3 (I/J and R forms), conserves E, and keeps E-only retracts."""
    # 1. CW quarter circle (I/J) in M83 relative E, followed by G1 and full 360 CCW circle (M82)
    gcode_ij = "\n".join(
        [
            "G90",
            "M83",
            "G0 X20 Y0 Z0.2 F6000",
            "G2 X30 Y10 I10 J0 E1.2 F3000",
            "G1 X40 Y10 E0.4 F3000",
            "M82",
            "G92 E0",
            "G3 X40 Y10 I-10 J0 E2.5 F3000",
        ]
    )
    moves = parse_gcode(gcode_ij)

    # First move is G0 to (20, 0, 0.2)
    assert moves[0].end_mm == pytest.approx((20.0, 0.0, 0.2))

    # Find G2 segments between (20,0) and (30,10), and the following G1 to (40,10)
    g1_idx = next(i for i, m in enumerate(moves) if m.end_mm == pytest.approx((40.0, 10.0, 0.2)))
    arc1_moves = moves[1:g1_idx]
    assert len(arc1_moves) >= 15
    assert arc1_moves[0].start_mm == pytest.approx((20.0, 0.0, 0.2))
    assert arc1_moves[-1].end_mm == pytest.approx((30.0, 10.0, 0.2))
    # CW from (20,0) around center (30,0) to (30,10) goes through (20..30, +Y) -> midpoint ~ (22.93, 7.07)
    mid_pt = arc1_moves[len(arc1_moves) // 2].start_mm
    assert mid_pt[1] > 5.0
    arc1_len = sum(m.length_mm for m in arc1_moves)
    assert arc1_len == pytest.approx(0.5 * math.pi * 10.0, rel=2e-3)
    assert sum(m.extrude_mm for m in arc1_moves) == pytest.approx(1.2, rel=1e-9)

    # G1 starts exactly from arc end (30, 10, 0.2), NOT (20, 0, 0.2)
    assert moves[g1_idx].start_mm == pytest.approx((30.0, 10.0, 0.2))
    assert moves[g1_idx].extrude_mm == pytest.approx(0.4, rel=1e-9)

    # Full 360 deg CCW circle around (30, 10) with R=10 in M82 mode
    circle_moves = moves[g1_idx + 1 :]
    assert len(circle_moves) >= 60
    assert sum(m.length_mm for m in circle_moves) == pytest.approx(2.0 * math.pi * 10.0, rel=2e-3)
    assert sum(m.extrude_mm for m in circle_moves) == pytest.approx(2.5, rel=1e-9)

    # 2. G2/G3 R-form minor (R > 0 -> 90 deg) vs major (R < 0 -> 270 deg)
    gcode_r = "\n".join(
        [
            "G90",
            "M83",
            "G0 X100 Y100 F6000",
            "G3 X110 Y110 R10 E0.5 F3000",
            "G0 X100 Y100 F6000",
            "G2 X110 Y110 R-10 E1.5 F3000",
        ]
    )
    moves_r = parse_gcode(gcode_r)
    travel2_idx = next(i for i in range(1, len(moves_r)) if not moves_r[i].is_extrusion)
    minor_arc = moves_r[1:travel2_idx]
    major_arc = moves_r[travel2_idx + 1 :]
    assert sum(m.length_mm for m in minor_arc) == pytest.approx(0.5 * math.pi * 10.0, rel=2e-3)
    assert sum(m.length_mm for m in major_arc) == pytest.approx(1.5 * math.pi * 10.0, rel=2e-3)
    assert sum(m.extrude_mm for m in minor_arc) == pytest.approx(0.5, rel=1e-9)
    assert sum(m.extrude_mm for m in major_arc) == pytest.approx(1.5, rel=1e-9)

    # 3. E-only moves (M83 & M82) and M204 P/T + SET_VELOCITY_LIMIT ACCEL=
    gcode_retract = "\n".join(
        [
            "G90",
            "M83",
            "M204 P1500 T4500",
            "G0 X50 Y50 F12000",
            "G1 X90 Y50 E1.0 F3000",
            "G1 E-1.5 F1800",
            "G0 X120 Y50 F12000",
            "SET_VELOCITY_LIMIT ACCEL=6000",
            "G1 E1.5 F1800",
            "G1 X160 Y50 E1.0 F3000",
        ]
    )
    rmoves = parse_gcode(gcode_retract)
    assert len(rmoves) == 6
    assert rmoves[0].accel_override_mm_s2 == pytest.approx(4500.0)
    assert rmoves[1].accel_override_mm_s2 == pytest.approx(1500.0)
    # Retract move
    assert rmoves[2].length_mm == pytest.approx(0.0)
    assert rmoves[2].extrude_mm == pytest.approx(-1.5)
    assert rmoves[2].is_extrusion is False
    # Unretract after SET_VELOCITY_LIMIT ACCEL=6000
    assert rmoves[4].length_mm == pytest.approx(0.0)
    assert rmoves[4].extrude_mm == pytest.approx(1.5)
    assert rmoves[4].is_extrusion is False
    assert rmoves[5].accel_override_mm_s2 == pytest.approx(6000.0)


def test_a09_simulator_retract_duration_and_junction_stop() -> None:
    """E-only moves force junction speed to zero and add exact |dE|/F duration without false defects."""
    spec = PrinterSpec()
    spec.kinematics.max_accel_mm_s2 = 4000.0
    spec.kinematics.max_velocity_mm_s = 400.0
    asm = build_assembly(spec)
    mod = solve_modal_fea(asm)
    kin = analyze_kinematics(asm)
    th = simulate_hotend_thermal(spec)

    # Baseline: two collinear extrusion legs (total 80 mm at 200 mm/s)
    g_base = "\n".join(
        [
            "G90",
            "M83",
            "G0 X50 Y50 F12000",
            "G1 X90 Y50 E1.5 F12000",
            "G1 X130 Y50 E1.5 F12000",
        ]
    )
    # Same legs with an E-only retract + unretract inserted at (90, 50) at 30 mm/s (F1800)
    g_with_ret = "\n".join(
        [
            "G90",
            "M83",
            "G0 X50 Y50 F12000",
            "G1 X90 Y50 E1.5 F12000",
            "G1 E-1.5 F1800",
            "G1 E1.5 F1800",
            "G1 X130 Y50 E1.5 F12000",
        ]
    )

    vp_base = simulate_gcode(g_base, asm, mod, kin, th)
    vp_ret = simulate_gcode(g_with_ret, asm, mod, kin, th)

    assert len(vp_ret.segments) == 5
    # Retract and unretract segments: exact duration 1.5 / 30 = 0.05 s each, 0 XY speed, no defects
    for idx in (2, 3):
        seg = vp_ret.segments[idx]
        assert seg.duration_s == pytest.approx(0.05, rel=1e-6)
        assert seg.peak_speed_mm_s == pytest.approx(0.0, abs=1e-9)
        assert seg.volumetric_flow_mm3_s == pytest.approx(0.0, abs=1e-9)
        assert seg.ringing_severity == pytest.approx(0.0, abs=1e-9)
        assert seg.defect == "travel"

    # Stopping at junction adds exact decel + accel penalty on top of 0.10 s E-only retract/prime time
    assert vp_ret.total_time_s > vp_base.total_time_s + 0.10
    assert vp_ret.filament_used_mm == pytest.approx(vp_base.filament_used_mm, rel=1e-9)
    assert vp_ret.underextrusion_segments == 0
    assert vp_ret.ringing_segments == 0
