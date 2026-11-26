"""Verification suite for Kalico / AWD Closed-Loop FOC Beta Software Features."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from printsim.api.server import create_app
from printsim.engine import evaluate_printer
from printsim.fea.klipper_shaper import (
    ADVANCED_SHAPER_NAMES,
    SHAPER_NAMES,
    biquad_notch_params_from_db_q,
    calibrate_axis_shapers,
    estimate_shaper,
    find_shaper_max_accel,
    get_flatness_toolhead_bell,
    get_shaper_pulses,
    get_shaper_smoothing,
)
from printsim.fea.mechatronics import (
    DriveControlConfig,
    build_coupled_state_space,
    corexy_belt_span_stiffness_matrix,
)
from printsim.fea.modal import InputShaperRecommendation, ModalResult, solve_modal_fea
from printsim.gcode.ballbar import simulate_time_domain_metrology
from printsim.geometry import build_assembly
from printsim.kinematics.solvers import analyze_kinematics, stepper_available_torque_nm
from printsim.mcp_server import dispatch_tool
from printsim.schema import PrinterSpec, StepperSpec

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def test_smooth_flatness_kernel_cancellation_l1_and_toolhead_bell_smoothing() -> None:
    """Verify SMOOTH_FLATNESS exact inverse-plant cancellation, L1 norm, -10% sensitivity, and MZV comparison."""
    assert "SMOOTH_FLATNESS" in ADVANCED_SHAPER_NAMES
    assert "SMOOTH_FLATNESS" not in SHAPER_NAMES

    for f_n in (48.0, 75.0, 130.0):
        for zeta in (0.055, 0.10):
            amps, times = get_shaper_pulses("SMOOTH_FLATNESS", f_n, damping_ratio=zeta)
            assert len(amps) == 65
            assert len(times) == 65
            assert float(np.sum(amps)) == pytest.approx(1.0, rel=1e-9)
            assert times[0] == pytest.approx(0.0, abs=1e-12)
            assert times[-1] == pytest.approx(0.70 / f_n, rel=1e-9)

            # 1. Exact pole-zero cancellation at f = f_n (< 0.5% residual vibration)
            v_exact = float(estimate_shaper((amps, times), [f_n], test_damping_ratio=zeta)[0])
            assert v_exact < 0.005, f"Residual at f_n={f_n}, zeta={zeta} was {v_exact:.5f}"

            # 2. ZV-class -10% frequency mismatch sensitivity (~10-16% residual)
            v_m10 = float(estimate_shaper((amps, times), [0.9 * f_n], test_damping_ratio=zeta)[0])
            assert 0.08 < v_m10 < 0.20, f"Expected ZV-class -10% residual in (0.08, 0.20), got {v_m10:.4f}"

            # 3. L1 acceleration norm matches Kalico's ~1.169x peak (1.15..1.20)
            l1_norm = float(np.sum(np.abs(amps)))
            assert 1.15 <= l1_norm <= 1.20, f"Unexpected L1 norm {l1_norm:.4f}"

    # 4. Toolhead septic bell w0 gives lower corner smoothing & higher safe accel than MZV
    f_test = 72.0
    bell = get_flatness_toolhead_bell(f_test)
    mzv = get_shaper_pulses("MZV", f_test, damping_ratio=0.10)
    sm_bell = get_shaper_smoothing(bell, accel=5000.0, scv=5.0)
    sm_mzv = get_shaper_smoothing(mzv, accel=5000.0, scv=5.0)
    assert sm_bell < sm_mzv * 0.65
    assert find_shaper_max_accel(bell, scv=5.0) > find_shaper_max_accel(mzv, scv=5.0)

    g_flat = calibrate_axis_shapers("X", f_test, scv=5.0, shaper_choice="smooth_flatness")
    g_mzv = calibrate_axis_shapers("X", f_test, scv=5.0, shaper_choice="mzv")
    assert g_flat.recommended_shaper == "SMOOTH_FLATNESS"
    assert g_flat.recommended_max_accel_mm_s2 > g_mzv.recommended_max_accel_mm_s2


def test_biquad_notch_db_q_phase_lag_and_psd_carving() -> None:
    """Verify biquad_notch_params_from_db_q phase lag, monotonicity, and Klipper PSD notch carving."""
    zz, zp, lag_deg = biquad_notch_params_from_db_q(12.0, 1.75)
    assert zp == pytest.approx(1.0 / (2.0 * 1.75), rel=1e-9)
    assert zz == pytest.approx(zp * (10.0 ** (-12.0 / 20.0)), rel=1e-9)
    assert 30.0 <= lag_deg <= 45.0

    # Verify exact numerical max phase lag over r in (0, 1)
    r = np.linspace(1e-4, 1.0 - 1e-4, 10000)
    h = ((1.0 - r * r) + 1j * 2.0 * zz * r) / ((1.0 - r * r) + 1j * 2.0 * zp * r)
    lag_num = math.degrees(float(np.max(-np.angle(h))))
    assert lag_deg == pytest.approx(lag_num, abs=0.05)

    # Monotonic phase lag increase with notch depth
    _, _, lag_10 = biquad_notch_params_from_db_q(10.0, 1.75)
    _, _, lag_18 = biquad_notch_params_from_db_q(18.0, 1.75)
    assert lag_10 < lag_deg < lag_18
    assert lag_18 > 48.0

    # Visible PSD notch carving in solve_modal_fea when notch_filter_hz is enabled
    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_monolith_awd.yaml")
    base.kinematics.drive_type = "foc_servo"
    mod_base = solve_modal_fea(build_assembly(base))
    assert mod_base.shaper_graph_x is not None
    f_notch = float(mod_base.shaper_graph_x.peak_freq_hz)

    notched = base.model_copy(deep=True)
    notched.kinematics.notch_filter_hz = f_notch
    notched.kinematics.notch_depth_db = 12.0
    notched.kinematics.notch_q = 1.75
    mod_notched = solve_modal_fea(build_assembly(notched))
    assert mod_notched.shaper_graph_x is not None

    freqs = mod_notched.shaper_graph_x.freqs_hz
    val_base_at_notch = float(np.interp(f_notch, freqs, mod_base.shaper_graph_x.raw_psd))
    val_notched_at_notch = float(np.interp(f_notch, freqs, mod_notched.shaper_graph_x.raw_psd))
    assert val_notched_at_notch < val_base_at_notch * 0.12

    # Mechatronics state-space FRF matches exact depth attenuation (-12 dB -> ~0.2512)
    asm = build_assembly(notched)
    mdl_no = build_coupled_state_space(asm, mod_base, 0.5, 0.5, DriveControlConfig(drive_type="foc_servo"))
    mdl_db = build_coupled_state_space(
        asm,
        mod_base,
        0.5,
        0.5,
        DriveControlConfig(drive_type="foc_servo", notch_filter_hz=f_notch, notch_depth_db=12.0, notch_q=1.75),
    )
    h_no = abs(mdl_no.evaluate_frf(np.array([f_notch]))[0, 0, 0])
    h_db = abs(mdl_db.evaluate_frf(np.array([f_notch]))[0, 0, 0])
    assert (h_db / h_no) == pytest.approx(10.0 ** (-12.0 / 20.0), rel=1e-3)


def test_foc_beta_physics_pin_rotor_active_damping_awd_sync_mode_space_and_lpv() -> None:
    """Verify pin_rotor_ff, toolhead_imu_active_damping, awd_sync_mode, mode_space_torque_ff, and LPV."""
    spec_trident = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    asm_tri = build_assembly(spec_trident)
    mod_tri = solve_modal_fea(asm_tri)

    # 1. pin_rotor_ff multiplies k_em by 3.5x and moves coupled frequency toward locked-rotor f_b
    rigid_mod = ModalResult(
        fundamental_freq_hz=20000.0,
        shaper_x=InputShaperRecommendation("X", 20000.0, 20000.0, 100.0, 100.0, 100.0, "ZV", 10000.0, 0.5),
        shaper_y=InputShaperRecommendation("Y", 20000.0, 20000.0, 100.0, 100.0, 100.0, "ZV", 10000.0, 0.5),
        frame_damping_ratio=0.04,
    )
    k_cart_tri, _, _ = corexy_belt_span_stiffness_matrix(asm_tri, 0.5, 0.5)
    f_b_x = math.sqrt(float(k_cart_tri[0, 0]) / asm_tri.x_moving_mass_kg) / (2.0 * math.pi)

    for dt in ("closed_loop_stepper", "foc_servo"):
        mdl_base = build_coupled_state_space(
            asm_tri, rigid_mod, 0.5, 0.5, DriveControlConfig(drive_type=dt, pin_rotor_ff=False)
        )
        mdl_pin = build_coupled_state_space(
            asm_tri, rigid_mod, 0.5, 0.5, DriveControlConfig(drive_type=dt, pin_rotor_ff=True)
        )
        assert mdl_pin.k_em_per_motor_n_m == pytest.approx(3.5 * mdl_base.k_em_per_motor_n_m, rel=1e-9)
        assert mdl_pin.coupled_freq_x_hz > mdl_base.coupled_freq_x_hz
        assert mdl_pin.coupled_freq_y_hz > mdl_base.coupled_freq_y_hz
        assert abs(f_b_x - mdl_pin.coupled_freq_x_hz) < abs(f_b_x - mdl_base.coupled_freq_x_hz)

    # Also improves motor_x / motor_y resonance in modal.py
    spec_pin = spec_trident.model_copy(deep=True)
    spec_pin.kinematics.drive_type = "foc_servo"
    mod_no_pin = solve_modal_fea(build_assembly(spec_pin))
    spec_pin.kinematics.pin_rotor_ff = True
    mod_yes_pin = solve_modal_fea(build_assembly(spec_pin))
    assert mod_yes_pin.shaper_x is not None and mod_no_pin.shaper_x is not None
    assert mod_yes_pin.shaper_y is not None and mod_no_pin.shaper_y is not None
    assert mod_yes_pin.shaper_x.motor_freq_hz == pytest.approx(
        mod_no_pin.shaper_x.motor_freq_hz * math.sqrt(3.5), rel=1e-3
    )
    assert mod_yes_pin.shaper_x.effective_freq_hz > mod_no_pin.shaper_x.effective_freq_hz

    # 2. toolhead_imu_active_damping increases damping, flattens PSD peak, and raises safe accel
    mdl_no_imu = build_coupled_state_space(
        asm_tri, mod_tri, 0.5, 0.5, DriveControlConfig(drive_type="foc_servo", toolhead_imu_active_damping=False)
    )
    mdl_imu = build_coupled_state_space(
        asm_tri, mod_tri, 0.5, 0.5, DriveControlConfig(drive_type="foc_servo", toolhead_imu_active_damping=True)
    )
    assert mdl_imu.coupled_damping_x > mdl_no_imu.coupled_damping_x + 0.02
    assert mdl_imu.coupled_damping_y > mdl_no_imu.coupled_damping_y + 0.05

    spec_imu = spec_trident.model_copy(deep=True)
    spec_imu.kinematics.drive_type = "foc_servo"
    mod_imu_off = solve_modal_fea(build_assembly(spec_imu))
    spec_imu.kinematics.toolhead_imu_active_damping = True
    mod_imu_on = solve_modal_fea(build_assembly(spec_imu))
    assert mod_imu_on.shaper_graph_x is not None and mod_imu_off.shaper_graph_x is not None
    assert mod_imu_on.shaper_graph_x.damping_ratio > mod_imu_off.shaper_graph_x.damping_ratio + 0.05
    assert float(np.max(mod_imu_on.shaper_graph_x.raw_psd)) < float(np.max(mod_imu_off.shaper_graph_x.raw_psd)) * 0.5
    assert mod_imu_on.recommended_max_accel_mm_s2 > mod_imu_off.recommended_max_accel_mm_s2

    # 3. AWD sync modes (position_equalizer +12% / +0.015 vs torque_mirror +6% / +0.008 vs none)
    spec_awd = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_monolith_awd.yaml")
    spec_awd.kinematics.drive_type = "closed_loop_stepper"
    asm_awd = build_assembly(spec_awd)
    mod_awd = solve_modal_fea(asm_awd)
    mdl_awd_none = build_coupled_state_space(
        asm_awd, mod_awd, 0.5, 0.5, DriveControlConfig(drive_type="closed_loop_stepper", awd_sync_mode="none")
    )
    mdl_awd_tm = build_coupled_state_space(
        asm_awd, mod_awd, 0.5, 0.5, DriveControlConfig(drive_type="closed_loop_stepper", awd_sync_mode="torque_mirror")
    )
    mdl_awd_eq = build_coupled_state_space(
        asm_awd,
        mod_awd,
        0.5,
        0.5,
        DriveControlConfig(drive_type="closed_loop_stepper", awd_sync_mode="position_equalizer", awd_equalizer_bw_hz=2.0),
    )
    mdl_awd_eq_hi = build_coupled_state_space(
        asm_awd,
        mod_awd,
        0.5,
        0.5,
        DriveControlConfig(drive_type="closed_loop_stepper", awd_sync_mode="position_equalizer", awd_equalizer_bw_hz=10.0),
    )
    assert mdl_awd_tm.k_em_per_motor_n_m == pytest.approx(mdl_awd_none.k_em_per_motor_n_m * 1.06, rel=1e-9)
    assert mdl_awd_eq.k_em_per_motor_n_m == pytest.approx(mdl_awd_none.k_em_per_motor_n_m * 1.12, rel=1e-9)
    assert mdl_awd_eq.coupled_freq_x_hz > mdl_awd_tm.coupled_freq_x_hz > mdl_awd_none.coupled_freq_x_hz
    assert mdl_awd_eq.coupled_damping_y > mdl_awd_tm.coupled_damping_y > mdl_awd_none.coupled_damping_y
    # High equalizer BW (> 5 Hz) clamps/degrades damping slightly due to runout lag
    assert mdl_awd_eq_hi.coupled_damping_y < mdl_awd_eq.coupled_damping_y

    # 4. mode_space_torque_ff cuts corner XY cross-coupling ~35% and ballbar circularity ~30%
    mdl_cc_off = build_coupled_state_space(
        asm_tri, mod_tri, 0.05, 0.95, DriveControlConfig(drive_type="foc_servo", mode_space_torque_ff=False)
    )
    mdl_cc_on = build_coupled_state_space(
        asm_tri, mod_tri, 0.05, 0.95, DriveControlConfig(drive_type="foc_servo", mode_space_torque_ff=True)
    )
    assert mdl_cc_on.xy_cross_coupling_pct < mdl_cc_off.xy_cross_coupling_pct * 0.72

    bb_off = simulate_time_domain_metrology(
        asm_tri, mod_tri, drive_cfg=DriveControlConfig(drive_type="foc_servo", mode_space_torque_ff=False)
    )
    bb_on = simulate_time_domain_metrology(
        asm_tri, mod_tri, drive_cfg=DriveControlConfig(drive_type="foc_servo", mode_space_torque_ff=True)
    )
    assert bb_on.circularity_gxy_um < bb_off.circularity_gxy_um * 0.78

    # 5. LPV 3x3 Workspace Scheduling eliminates corner ringing spread on SMOOTH_FLATNESS
    spec_flat_no_lpv = spec_trident.model_copy(deep=True)
    spec_flat_no_lpv.kinematics.drive_type = "foc_servo"
    spec_flat_no_lpv.kinematics.input_shaper = "smooth_flatness"
    spec_flat_no_lpv.kinematics.lpv_workspace_scheduling = False
    mod_no_lpv = solve_modal_fea(build_assembly(spec_flat_no_lpv))

    spec_flat_lpv = spec_flat_no_lpv.model_copy(deep=True)
    spec_flat_lpv.kinematics.lpv_workspace_scheduling = True
    mod_yes_lpv = solve_modal_fea(build_assembly(spec_flat_lpv))

    assert mod_no_lpv.shaper_x is not None and mod_yes_lpv.shaper_x is not None
    assert mod_no_lpv.shaper_graph_x is not None and mod_yes_lpv.shaper_graph_x is not None
    assert float(np.sum(mod_yes_lpv.shaper_graph_x.shaped_psd)) <= float(
        np.sum(mod_no_lpv.shaper_graph_x.shaped_psd)
    )
    assert mod_yes_lpv.recommended_max_accel_mm_s2 > mod_no_lpv.recommended_max_accel_mm_s2

    # 6. Time-domain corner rounding of SMOOTH_FLATNESS vs MZV at matched frequency (~35% less rounding)
    bb_mzv = simulate_time_domain_metrology(
        asm_tri, mod_tri, shaper_name="MZV", drive_cfg=DriveControlConfig(drive_type="closed_loop_stepper")
    )
    bb_flat = simulate_time_domain_metrology(
        asm_tri, mod_tri, shaper_name="SMOOTH_FLATNESS", drive_cfg=DriveControlConfig(drive_type="closed_loop_stepper")
    )
    assert bb_flat.corner_rounding_um < bb_mzv.corner_rounding_um * 0.75


def test_open_loop_stepper_ignores_foc_features_and_emits_diagnostic() -> None:
    """FOC-only features gracefully no-op on open_loop_stepper and emit the required diagnostic warning."""
    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_monolith_awd.yaml")
    base.kinematics.drive_type = "open_loop_stepper"
    rep_base = evaluate_printer(base, panel_mesh_n=2)

    ol_with_foc_flags = base.model_copy(deep=True)
    ol_with_foc_flags.kinematics.pin_rotor_ff = True
    ol_with_foc_flags.kinematics.toolhead_imu_active_damping = True
    ol_with_foc_flags.kinematics.awd_sync_mode = "position_equalizer"
    ol_with_foc_flags.kinematics.mode_space_torque_ff = True
    rep_ol_flags = evaluate_printer(ol_with_foc_flags, panel_mesh_n=2)

    assert rep_ol_flags.modal_fea.recommended_max_accel_mm_s2 == pytest.approx(
        rep_base.modal_fea.recommended_max_accel_mm_s2, rel=1e-9
    )
    assert rep_ol_flags.workspace_frf is not None and rep_base.workspace_frf is not None
    assert rep_ol_flags.workspace_frf.center_freq_hz == pytest.approx(
        rep_base.workspace_frf.center_freq_hz, rel=1e-9
    )
    assert rep_ol_flags.time_domain_metrology is not None and rep_base.time_domain_metrology is not None
    assert rep_ol_flags.time_domain_metrology.circularity_gxy_um == pytest.approx(
        rep_base.time_domain_metrology.circularity_gxy_um, rel=1e-9
    )
    assert any(
        ("closed-loop foc" in d.lower() and "0x65" in d.lower())
        for d in rep_ol_flags.diagnostics
    )


def test_all_five_honest_engineering_diagnostics_and_api_catalog() -> None:
    """Verify all 5 Kalico/FOC diagnostic warnings trigger cleanly and API/MCP expose smooth_flatness."""
    # 1. smooth_flatness on 2WD CoreXY without lpv_workspace_scheduling
    s1 = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    s1.kinematics.drive_type = "foc_servo"
    s1.kinematics.input_shaper = "smooth_flatness"
    s1.kinematics.lpv_workspace_scheduling = False
    r1 = evaluate_printer(s1, panel_mesh_n=2)
    assert any("lpv" in d.lower() and "corner" in d.lower() for d in r1.diagnostics)

    # 2. smooth_flatness with torque_safety_margin < 1.25
    # (With B-04 consistent peak/RMS pull-out curve & B-06 exact CoreXY vector torque, margin at 35k is ~1.34;
    #  42,000 mm/s^2 triggers the < 1.25x L1 margin diagnostic cleanly.)
    s2 = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    s2.kinematics.drive_type = "foc_servo"
    s2.kinematics.input_shaper = "smooth_flatness"
    s2.kinematics.lpv_workspace_scheduling = True
    s2.kinematics.max_accel_mm_s2 = 42000.0
    s2.kinematics.max_velocity_mm_s = 700.0
    r2 = evaluate_printer(s2, panel_mesh_n=2)
    assert r2.kinematics.torque_safety_margin < 1.25
    assert any("1.169" in d and "l1" in d.lower() for d in r2.diagnostics)

    # 3. awd_sync_mode == "position_equalizer" with awd_equalizer_bw_hz > 4.0
    s3 = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_monolith_awd.yaml")
    s3.kinematics.drive_type = "foc_servo"
    s3.kinematics.awd_sync_mode = "position_equalizer"
    s3.kinematics.awd_equalizer_bw_hz = 7.5
    r3 = evaluate_printer(s3, panel_mesh_n=2)
    assert any("runout" in d.lower() and "1.5" in d for d in r3.diagnostics)

    # 4. notch_filter_hz set with notch_depth_db > 15.0
    s4 = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_monolith_awd.yaml")
    s4.kinematics.drive_type = "foc_servo"
    s4.kinematics.notch_filter_hz = 120.0
    s4.kinematics.notch_depth_db = 22.0
    r4 = evaluate_printer(s4, panel_mesh_n=2)
    assert any("phase lag" in d.lower() and "50" in d for d in r4.diagnostics)

    # 5. FOC-only feature with open_loop_stepper
    s5 = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_monolith_awd.yaml")
    s5.kinematics.drive_type = "open_loop_stepper"
    s5.kinematics.mode_space_torque_ff = True
    r5 = evaluate_printer(s5, panel_mesh_n=2)
    assert any("0x65" in d.lower() and "open_loop_stepper" in d.lower() for d in r5.diagnostics)

    # API /catalog & /simulate + MCP list_catalog expose smooth_flatness
    client = TestClient(create_app())
    cat_api = client.get("/api/catalog").json()
    assert "smooth_flatness" in {item["id"] for item in cat_api["input_shapers"]}

    cat_mcp = dispatch_tool("list_catalog", {})
    assert "smooth_flatness" in cat_mcp["input_shapers"]

    sim_resp = client.post(
        "/api/simulate",
        json={"spec": s1.model_dump(mode="json"), "include_toolpath_segments": False},
    )
    assert sim_resp.status_code == 200
    assert sim_resp.json()["modal_fea"]["shaper_x"]["recommended_shaper"] == "SMOOTH_FLATNESS"


def test_motor_voltage_limited_velocity_kalico_report1_benchmark() -> None:
    """Verify motor back-EMF + w_e*L*i bus-voltage velocity ceiling against Kalico Report 1 benchmarks."""
    motor_r1 = StepperSpec(
        holding_torque_nm=0.55,
        rated_current_a=2.5,
        run_current_a=2.5,
        phase_resistance_ohm=1.2,
        inductance_mh=1.6,
        supply_voltage_v=48.0,
        steps_per_rev=200,
    )

    spec_48_20t = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    spec_48_20t.gantry.belt.pulley_teeth = 20
    spec_48_20t.kinematics.xy_stepper = motor_r1.model_copy(deep=True)
    kin_48_20t = analyze_kinematics(build_assembly(spec_48_20t))

    spec_48_24t = spec_48_20t.model_copy(deep=True)
    spec_48_24t.gantry.belt.pulley_teeth = 24
    kin_48_24t = analyze_kinematics(build_assembly(spec_48_24t))

    spec_24_20t = spec_48_20t.model_copy(deep=True)
    spec_24_20t.kinematics.xy_stepper.supply_voltage_v = 24.0
    kin_24_20t = analyze_kinematics(build_assembly(spec_24_20t))

    spec_foc_48 = spec_48_20t.model_copy(deep=True)
    spec_foc_48.kinematics.drive_type = "foc_servo"
    kin_foc_48 = analyze_kinematics(build_assembly(spec_foc_48))

    # B-04/B-05 upgraded piecewise pull-out torque balance above corner speed with CoreXY sqrt(2)
    # diagonal factor and dual-H-bridge 0.98*V_bus FOC (without unphysical 3-phase 1.12 SVPWM boost).
    assert 1150.0 <= kin_48_20t.motor_max_velocity_mm_s <= 1350.0
    assert 1380.0 <= kin_48_24t.motor_max_velocity_mm_s <= 1600.0
    assert 550.0 <= kin_24_20t.motor_max_velocity_mm_s <= 680.0

    # Monotonicity: 48V > 24V, 24T ~20% > 20T, and FOC dual-H-bridge > open-loop chopper
    assert kin_48_20t.motor_max_velocity_mm_s > kin_24_20t.motor_max_velocity_mm_s * 1.8
    assert kin_48_24t.motor_max_velocity_mm_s == pytest.approx(kin_48_20t.motor_max_velocity_mm_s * 1.20, rel=0.03)
    assert kin_foc_48.motor_max_velocity_mm_s > kin_48_20t.motor_max_velocity_mm_s * 1.06


def test_motor_torque_utilization_3way_accel_and_smooth_flatness_l1() -> None:
    """Verify FOC vs closed-loop vs open-loop torque utilization, AWD > 2WD, and SMOOTH_FLATNESS 1.169x L1."""
    st = StepperSpec(
        holding_torque_nm=0.65,
        rated_current_a=2.8,
        run_current_a=2.0,
        phase_resistance_ohm=0.7,
        inductance_mh=0.6,
        supply_voltage_v=48.0,
    )
    t_ol = stepper_available_torque_nm(st, 600.0)
    t_cl = stepper_available_torque_nm(st, 600.0, drive_type="closed_loop_stepper")
    t_foc = stepper_available_torque_nm(st, 600.0, drive_type="foc_servo")
    assert t_cl == pytest.approx(t_ol * (0.85 / 0.72), rel=1e-9)
    assert t_foc == pytest.approx(t_ol * (0.95 / 0.72), rel=1e-9)

    base_2wd = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    base_2wd.kinematics.drive_type = "open_loop_stepper"
    kin_ol = analyze_kinematics(build_assembly(base_2wd))

    spec_cl = base_2wd.model_copy(deep=True)
    spec_cl.kinematics.drive_type = "closed_loop_stepper"
    kin_cl = analyze_kinematics(build_assembly(spec_cl))

    spec_foc = base_2wd.model_copy(deep=True)
    spec_foc.kinematics.drive_type = "foc_servo"
    kin_foc = analyze_kinematics(build_assembly(spec_foc))

    assert kin_foc.motor_max_accel_mm_s2 > kin_cl.motor_max_accel_mm_s2 > kin_ol.motor_max_accel_mm_s2
    # B-06: motor_max_accel_mm_s2 is exact worst-direction CoreXY limit sqrt(M_x^2 + M_y^2),
    # which is strictly tighter than pure-axis min(ax, ay) when both X and Y moving masses are positive.
    assert kin_foc.motor_max_accel_mm_s2 < min(
        kin_foc.motor_max_accel_x_mm_s2, kin_foc.motor_max_accel_y_mm_s2
    )

    # AWD doubles motor count and halves per-motor diagonal load share -> higher motor_max_accel_mm_s2
    spec_awd = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_monolith_awd.yaml")
    spec_awd.kinematics.drive_type = "foc_servo"
    spec_awd_2wd = spec_awd.model_copy(deep=True)
    spec_awd_2wd.gantry.drive_mode = "2wd"
    kin_awd = analyze_kinematics(build_assembly(spec_awd))
    kin_2wd = analyze_kinematics(build_assembly(spec_awd_2wd))
    assert kin_awd.motor_max_accel_mm_s2 > kin_2wd.motor_max_accel_mm_s2 * 1.48

    # SMOOTH_FLATNESS applies exact 1.169x L1 factor to motor_max_accel and belt_max_accel
    spec_flat = spec_awd.model_copy(deep=True)
    spec_flat.kinematics.input_shaper = "smooth_flatness"
    kin_flat = analyze_kinematics(build_assembly(spec_flat))
    assert kin_awd.shaper_l1_accel_factor == pytest.approx(1.0, rel=1e-9)
    assert kin_flat.shaper_l1_accel_factor == pytest.approx(1.169, rel=1e-9)
    assert kin_flat.motor_max_accel_mm_s2 == pytest.approx(kin_awd.motor_max_accel_mm_s2 / 1.169, rel=1e-6)
    assert kin_flat.belt_max_accel_mm_s2 == pytest.approx(kin_awd.belt_max_accel_mm_s2 / 1.169, rel=1e-6)


def test_motion_limits_evaluator_api_and_motor_ceiling_diagnostics() -> None:
    """Verify motion_limits 3-way limiting factor switch to 'motor', API payload, and velocity ceiling WARN."""
    # Stiff AWD FOC + smooth_flatness + IMU active damping drives shaper accel above motor torque ceiling
    spec_motor_lim = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_monolith_awd.yaml")
    spec_motor_lim.kinematics.drive_type = "foc_servo"
    spec_motor_lim.kinematics.input_shaper = "smooth_flatness"
    spec_motor_lim.kinematics.toolhead_imu_active_damping = True
    spec_motor_lim.kinematics.lpv_workspace_scheduling = True
    rep_lim = evaluate_printer(spec_motor_lim, panel_mesh_n=2)

    ml = rep_lim.motion_limits
    assert ml is not None
    expected_keys = {
        "shaper_max_accel_mm_s2",
        "motor_max_accel_mm_s2",
        "belt_max_accel_mm_s2",
        "achievable_max_accel_mm_s2",
        "accel_limiting_factor",
        "motor_max_velocity_mm_s",
        "flow_max_velocity_mm_s",
        "configured_max_velocity_mm_s",
        "configured_max_accel_mm_s2",
        "speed_limiting_factor",
        "shaper_l1_accel_factor",
    }
    assert expected_keys.issubset(ml.keys())
    assert ml["shaper_max_accel_mm_s2"] > ml["motor_max_accel_mm_s2"]
    assert ml["accel_limiting_factor"] == "motor"
    assert ml["achievable_max_accel_mm_s2"] == ml["motor_max_accel_mm_s2"]
    assert ml["shaper_l1_accel_factor"] == pytest.approx(1.169, rel=1e-6)
    assert any("motor-torque limited" in d.lower() for d in rep_lim.diagnostics)

    # Speed limiting switch + velocity ceiling WARN when configured max_velocity > motor voltage ceiling
    spec_vwarn = spec_motor_lim.model_copy(deep=True)
    spec_vwarn.kinematics.xy_stepper.supply_voltage_v = 24.0
    spec_vwarn.kinematics.max_velocity_mm_s = 900.0
    rep_vwarn = evaluate_printer(spec_vwarn, panel_mesh_n=2)
    assert rep_vwarn.motion_limits is not None
    assert rep_vwarn.motion_limits["speed_limiting_factor"] == "motor_voltage"
    assert any(
        ("warn" in d.lower() and "voltage ceiling" in d.lower())
        for d in rep_vwarn.diagnostics
    )

    # /api/simulate exposes motion_limits cleanly
    client = TestClient(create_app())
    sim_resp = client.post(
        "/api/simulate",
        json={"spec": spec_motor_lim.model_dump(mode="json"), "include_toolpath_segments": False},
    )
    assert sim_resp.status_code == 200
    payload = sim_resp.json()
    assert payload["motion_limits"]["accel_limiting_factor"] == "motor"
    assert payload["kinematics"]["motor_max_velocity_mm_s"] > 0.0
    assert payload["kinematics"]["belt_max_accel_mm_s2"] > 0.0
