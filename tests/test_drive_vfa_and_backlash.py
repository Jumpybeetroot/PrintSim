"""Physics verification tests for Ballbar backlash/CTE/45-deg play and MIMO VFA / FOC / mid-band sweeps."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from printsim.fea.mechatronics import (
    DriveControlConfig,
    analyze_midband_resonance_sweep,
    build_coupled_state_space,
    compute_vfa_speed_sweep,
)
from printsim.fea.modal import solve_modal_fea
from printsim.gcode.ballbar import simulate_time_domain_metrology
from printsim.geometry.builder import build_assembly
from printsim.schema.printer import PrinterSpec

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


# ---------------------------------------------------------------------------
# Track #2: Ballbar Reversal Backlash, Per-Axis Stribeck, CTE Play & 45° Play
# ---------------------------------------------------------------------------


def test_ballbar_per_axis_stiction_and_corexy_45deg_diagonal_play() -> None:
    """Verify CoreXY Y (2-block) > X (1-block) stiction, 45° A/B reversal glitch, and belt tooth backlash."""
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    asm = build_assembly(spec)
    mod = solve_modal_fea(asm)
    res = simulate_time_domain_metrology(asm, mod)

    # CoreXY carries 2× Y-rail blocks vs 1× X-rail block
    assert res.rail_stiction_x_n > 0.0
    assert res.rail_stiction_y_n > res.rail_stiction_x_n
    assert res.reversal_backlash_um > 0.0
    assert res.diagonal_glitch_45deg_um > 0.0

    # Dual top/bottom X-rail doubles X breakaway stiction
    spec_dual = spec.model_copy(deep=True)
    spec_dual.gantry.x_rail_mount = "dual_top_bottom"
    asm_dual = build_assembly(spec_dual)
    res_dual = simulate_time_domain_metrology(asm_dual, solve_modal_fea(asm_dual))
    assert res_dual.rail_stiction_x_n == pytest.approx(2.0 * res.rail_stiction_x_n, rel=1e-3)

    # HTD 3M semicircular belt profile carries much higher tooth-flank backlash than Gates 2GT
    spec_htd = spec.model_copy(deep=True)
    spec_htd.gantry.belt.type = "htd3m_15mm"
    asm_htd = build_assembly(spec_htd)
    res_htd = simulate_time_domain_metrology(asm_htd, solve_modal_fea(asm_htd))
    assert res_htd.reversal_backlash_um > res.reversal_backlash_um + 5.0

    # Serialized dict exposes all new ISO 230-4 backlash & stiction diagnostics
    payload = res.to_dict()
    for key in (
        "diagonal_glitch_45deg_um",
        "reversal_backlash_um",
        "rail_stiction_x_n",
        "rail_stiction_y_n",
    ):
        assert key in payload


def test_ballbar_cte_press_fit_clearance_and_bearing_override() -> None:
    """Heated CNC 6061 bracket loosens F695 press-fit (>66 °C); disc_spring_m5 & Ti64 eliminate it."""
    spec_cold = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_monolith_awd.yaml")
    spec_cold.gantry.bracket_material = "aluminum_6061_t6"
    spec_cold.gantry.idler_axial_preload = "rigid_stack"
    spec_cold.frame.chamber_temp_c = 25.0

    spec_hot = spec_cold.model_copy(deep=True)
    spec_hot.frame.chamber_temp_c = 85.0

    spec_disc = spec_hot.model_copy(deep=True)
    spec_disc.gantry.idler_axial_preload = "disc_spring_m5"

    asm_cold = build_assembly(spec_cold)
    asm_hot = build_assembly(spec_hot)
    asm_disc = build_assembly(spec_disc)

    r_cold = simulate_time_domain_metrology(asm_cold, solve_modal_fea(asm_cold))
    r_hot = simulate_time_domain_metrology(asm_hot, solve_modal_fea(asm_hot))
    r_disc = simulate_time_domain_metrology(asm_disc, solve_modal_fea(asm_disc))

    assert r_hot.reversal_backlash_um > r_cold.reversal_backlash_um + 1.4
    assert r_disc.reversal_backlash_um == pytest.approx(r_cold.reversal_backlash_um, rel=0.05)

    # Explicit bearing_clearance override flows directly into reversal_backlash_um
    spec_ov = spec_cold.model_copy(deep=True)
    spec_ov.model_overrides.bearing_clearance.radial_play_um = 24.0
    asm_ov = build_assembly(spec_ov)
    r_ov = simulate_time_domain_metrology(asm_ov, solve_modal_fea(asm_ov))
    assert r_ov.reversal_backlash_um > r_cold.reversal_backlash_um + 10.0


# ---------------------------------------------------------------------------
# Track #3: Speed-Dependent Mid-Band Sweep, FOC Phase Lag & MIMO VFA Sweep
# ---------------------------------------------------------------------------


def test_midband_resonance_sweep_voltage_and_closed_loop_mitigation() -> None:
    """Verify 48 V and closed-loop FOC raise Acarnley mid-band electrical damping vs 24 V open-loop."""
    spec_24 = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    spec_24.kinematics.xy_stepper.supply_voltage_v = 24.0
    spec_24.kinematics.max_velocity_mm_s = 400.0

    spec_48 = spec_24.model_copy(deep=True)
    spec_48.kinematics.xy_stepper.supply_voltage_v = 48.0

    asm_24 = build_assembly(spec_24)
    asm_48 = build_assembly(spec_48)

    sw_24 = analyze_midband_resonance_sweep(asm_24)
    sw_48 = analyze_midband_resonance_sweep(asm_48)
    sw_cl = analyze_midband_resonance_sweep(
        asm_24, drive_cfg=DriveControlConfig(drive_type="closed_loop_stepper")
    )
    sw_foc = analyze_midband_resonance_sweep(
        asm_24, drive_cfg=DriveControlConfig(drive_type="foc_servo")
    )

    assert len(sw_24.speeds_mm_s) == 36
    assert sw_48.min_damping_ratio > sw_24.min_damping_ratio
    assert sw_cl.midband_instability_risk is False
    assert sw_foc.midband_instability_risk is False
    assert sw_cl.min_damping_ratio > sw_24.min_damping_ratio
    assert sw_foc.min_damping_ratio > sw_24.min_damping_ratio


def test_foc_current_loop_phase_lag_reduces_velocity_damping() -> None:
    """Lower FOC current-loop bandwidth degrades velocity-loop phase margin and coupled damping."""
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_monolith_awd.yaml")
    asm = build_assembly(spec)
    mod = solve_modal_fea(asm)

    mdl_fast_curr = build_coupled_state_space(
        asm,
        mod,
        drive_cfg=DriveControlConfig(
            drive_type="foc_servo",
            servo_vel_bw_hz=150.0,
            servo_pos_bw_hz=30.0,
            servo_curr_bw_hz=1500.0,
        ),
    )
    mdl_slow_curr = build_coupled_state_space(
        asm,
        mod,
        drive_cfg=DriveControlConfig(
            drive_type="foc_servo",
            servo_vel_bw_hz=150.0,
            servo_pos_bw_hz=30.0,
            servo_curr_bw_hz=220.0,
        ),
    )

    assert mdl_slow_curr.b_em_per_motor_ns_m < mdl_fast_curr.b_em_per_motor_ns_m * 0.85


def test_mimo_vfa_speed_sweep_resonance_09deg_foc_and_3gt() -> None:
    """Verify MIMO VFA speed sweep captures belt/motor resonance, 0.9° & FOC cuts, and 3GT penalty."""
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    asm = build_assembly(spec)
    mod = solve_modal_fea(asm)

    t0 = time.perf_counter()
    vfa_18 = compute_vfa_speed_sweep(asm, mod, axis="X", n_points=40)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 80.0, f"VFA speed sweep took {elapsed_ms:.1f} ms"

    assert len(vfa_18.speeds_mm_s) == 40
    assert vfa_18.peak_vfa_um > 0.0
    assert vfa_18.recommended_clean_speed_mm_s >= 20.0
    payload = vfa_18.to_dict()
    assert payload["axis"] == "X"
    assert "midband_sweep" in payload

    # 0.9° stepper (400 steps/rev) cuts motor-side VFA ripple compared to 1.8° stepper (200 steps/rev)
    spec_09 = spec.model_copy(deep=True)
    spec_09.kinematics.xy_stepper.steps_per_rev = 400
    asm_09 = build_assembly(spec_09)
    vfa_09 = compute_vfa_speed_sweep(asm_09, solve_modal_fea(asm_09), axis="X", n_points=40)
    assert max(vfa_09.motor_elec_ripple_um) < max(vfa_18.motor_elec_ripple_um) * 0.75

    # Closed-loop FOC dramatically suppresses motor-side electrical/detent VFA ripple
    vfa_foc = compute_vfa_speed_sweep(
        asm, mod, drive_cfg=DriveControlConfig(drive_type="foc_servo"), axis="X", n_points=40
    )
    assert max(vfa_foc.motor_elec_ripple_um) < max(vfa_18.motor_elec_ripple_um) * 0.25

    # 3GT (3 mm pitch) generates stronger belt-mesh VFA than 2GT (2 mm pitch)
    spec_3gt = spec.model_copy(deep=True)
    spec_3gt.gantry.belt.type = "3gt_9mm"
    spec_3gt.gantry.belt.pitch_mm = 3.0
    asm_3gt = build_assembly(spec_3gt)
    vfa_3gt = compute_vfa_speed_sweep(asm_3gt, solve_modal_fea(asm_3gt), axis="X", n_points=40)
    assert max(vfa_3gt.belt_mesh_ripple_um) > max(vfa_18.belt_mesh_ripple_um) * 1.4


def test_gcode_cross_axis_mismatch_ringing_corner_angle_and_mcr() -> None:
    """Verify CoreXY tension-mismatch cross-axis ringing, angle-scaled smoothing, and minimum_cruise_ratio."""
    from printsim.gcode.simulator import simulate_gcode
    from printsim.kinematics.solvers import analyze_kinematics
    from printsim.thermal.hotend import simulate_hotend_thermal

    spec_bal = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    spec_mis = spec_bal.model_copy(deep=True)
    spec_mis.gantry.belt.tension_mismatch_pct = 30.0

    asm_bal = build_assembly(spec_bal)
    asm_mis = build_assembly(spec_mis)
    mod_bal = solve_modal_fea(asm_bal)
    mod_mis = solve_modal_fea(asm_mis)
    kin_bal = analyze_kinematics(asm_bal)
    kin_mis = analyze_kinematics(asm_mis)
    th_bal = simulate_hotend_thermal(spec_bal)

    # Sharp 90° corner G-code
    gcode_90 = "G90\nM83\nG1 X50 Y50 Z0.2 F6000\nG1 X110 Y50 E2.5 F12000\nG1 X110 Y110 E2.5 F12000\n"
    # Shallow 25° bend G-code
    gcode_25 = "G90\nM83\nG1 X50 Y50 Z0.2 F6000\nG1 X110 Y50 E2.5 F12000\nG1 X164.4 Y75.4 E2.5 F12000\n"

    rep_bal = simulate_gcode(gcode_90, asm_bal, mod_bal, kin_bal, th_bal)
    rep_mis = simulate_gcode(gcode_90, asm_mis, mod_mis, kin_mis, th_bal)
    assert rep_mis.max_ringing_amplitude_um > rep_bal.max_ringing_amplitude_um

    # Shallow 25° bend produces less corner smoothing than a 90° corner
    rep_25 = simulate_gcode(gcode_25, asm_bal, mod_bal, kin_bal, th_bal)
    assert rep_25.max_corner_smoothing_um < rep_bal.max_corner_smoothing_um * 0.55

    # Klipper minimum_cruise_ratio=0.5 increases short-segment duration by capping triangular peak speed
    gcode_short = "G90\nM83\nG1 X50 Y50 Z0.2 F6000\nG1 X55 Y50 E0.2 F18000\nG1 X55 Y55 E0.2 F18000\n"
    rep_mcr0 = simulate_gcode(gcode_short, asm_bal, mod_bal, kin_bal, th_bal, minimum_cruise_ratio=0.0)
    rep_mcr5 = simulate_gcode(gcode_short, asm_bal, mod_bal, kin_bal, th_bal, minimum_cruise_ratio=0.5)
    assert rep_mcr5.total_time_s > rep_mcr0.total_time_s


def test_vfa_sweep_wasm_and_fastapi_endpoints() -> None:
    """Verify /api/vfa-sweep works identically in both wasm_dispatch and FastAPI."""
    import json
    from fastapi.testclient import TestClient

    from printsim.api.server import app
    from printsim.wasm_dispatch import dispatch_api_request

    yaml_text = (EXAMPLES_DIR / "corexy_voron_style.yaml").read_text(encoding="utf-8")
    body = {"yaml_text": yaml_text, "axis": "X", "drive_type": "open_loop_stepper", "n_points": 20}

    wasm_raw = dispatch_api_request("POST", "/api/vfa-sweep", json.dumps(body))
    assert wasm_raw["status"] == 200
    wasm_data = json.loads(wasm_raw["body_text"])

    client = TestClient(app)
    resp = client.post("/api/vfa-sweep", json=body)
    assert resp.status_code == 200
    api_data = resp.json()

    assert api_data["peak_vfa_um"] == pytest.approx(wasm_data["peak_vfa_um"], rel=1e-6)
    assert len(api_data["speeds_mm_s"]) == 20
    assert "midband_sweep" in api_data

