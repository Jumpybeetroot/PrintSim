"""Tests for kinematics (CoreXY, Cartesian, Delta), collisions, thermal, and G-code."""

from __future__ import annotations

import pytest

from printsim.fea import solve_modal_fea
from printsim.gcode import generate_benchmark_gcode, simulate_gcode
from printsim.geometry import build_assembly
from printsim.kinematics import (
    analyze_kinematics,
    check_frame_clearance,
    forward_kinematics,
    inverse_kinematics,
)
from printsim.schema import PrinterSpec
from printsim.thermal import simulate_hotend_thermal


@pytest.mark.parametrize("arch", ["corexy", "cartesian_bedslinger", "delta"])
def test_fk_ik_roundtrip(arch: str) -> None:
    spec = PrinterSpec(architecture=arch)  # type: ignore[arg-type]
    if arch == "delta":
        spec.kinematics.build_volume_mm = (200.0, 200.0, 250.0)
        spec.kinematics.delta_arm_length_mm = 300.0
        spec.kinematics.delta_radius_mm = 155.0

    target = (120.0, 95.0, 42.5)
    act = inverse_kinematics(spec, target)
    recovered = forward_kinematics(spec, act)
    assert recovered == pytest.approx(target, abs=1e-4)


def test_frame_collision_detected_when_frame_too_narrow() -> None:
    spec = PrinterSpec(architecture="corexy")
    spec.frame.width_mm = 310.0  # Too narrow for 300mm build + 68mm toolhead!
    asm = build_assembly(spec)
    col = check_frame_clearance(asm)
    assert col.has_collision is True
    assert col.clearance_x_mm < 0


def test_titanium_heatbreak_reduces_heat_creep_and_uhf_boosts_flow() -> None:
    # Thick stainless heatbreak with weak heatsink -> higher cold-side temp
    spec_std = PrinterSpec()
    spec_std.toolhead.hotend.heatbreak_material = "stainless_304"
    spec_std.toolhead.hotend.heatbreak_wall_mm = 0.65
    spec_std.toolhead.hotend.heatsink_thermal_resistance_k_w = 5.5
    spec_std.toolhead.hotend.melt_zone_length_mm = 11.5
    spec_std.toolhead.hotend.heater_power_w = 40.0

    spec_uhf = spec_std.model_copy(deep=True)
    spec_uhf.toolhead.hotend.heatbreak_material = "titanium_grade5"
    spec_uhf.toolhead.hotend.heatbreak_wall_mm = 0.22
    spec_uhf.toolhead.hotend.heatsink_thermal_resistance_k_w = 2.8
    spec_uhf.toolhead.hotend.melt_zone_length_mm = 24.0
    spec_uhf.toolhead.hotend.heater_power_w = 100.0
    spec_uhf.toolhead.extruder.max_push_force_n = 75.0

    th_std = simulate_hotend_thermal(spec_std)
    th_uhf = simulate_hotend_thermal(spec_uhf)

    assert th_uhf.cold_side_temp_c < th_std.cold_side_temp_c
    assert th_uhf.max_volumetric_flow_mm3_s > th_std.max_volumetric_flow_mm3_s * 1.5


def test_gcode_simulator_detects_underextrusion_on_low_flow_hotend() -> None:
    spec = PrinterSpec()
    spec.toolhead.hotend.melt_zone_length_mm = 9.0
    spec.toolhead.hotend.heater_power_w = 30.0
    asm = build_assembly(spec)
    mod = solve_modal_fea(asm)
    kin = analyze_kinematics(asm)
    th = simulate_hotend_thermal(spec)

    gcode = generate_benchmark_gcode(print_speed_mm_s=350.0, infill_speed_mm_s=450.0, height_mm=4.0)
    vp = simulate_gcode(gcode, asm, mod, kin, th)
    assert vp.total_moves > 20
    assert vp.underextrusion_segments > 0
    assert vp.total_time_s > 0
