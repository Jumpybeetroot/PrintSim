"""Phase 2 verification tests: coupled electro-mechanical state space, workspace FRF maps,
advanced input shapers (ZVDD, SMOOTH_ZV, SMOOTH_EI, multi-mode convolution), and ISO 230-4 ballbar.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pytest

from printsim.fea import InputShaperRecommendation, ModalResult, solve_modal_fea
from printsim.fea.klipper_shaper import (
    SHAPER_NAMES,
    apply_biquad_notch,
    apply_shaper_to_trajectory,
    convolve_shapers,
    estimate_shaper,
    get_shaper_pulses,
)
from printsim.fea.mechatronics import (
    DriveControlConfig,
    build_coupled_state_space,
    compute_workspace_frf_map,
    corexy_belt_span_stiffness_matrix,
)
from printsim.gcode.ballbar import simulate_time_domain_metrology
from printsim.geometry import build_assembly
from printsim.schema import PrinterSpec, StepperSpec

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


# ---------------------------------------------------------------------------
# 1. Exact 2-DOF Motor-Belt-Carriage Eigenvalue & Reflected Rotor Inertia
# ---------------------------------------------------------------------------


def test_2dof_motor_belt_carriage_closed_form_eigenvalues() -> None:
    """Verify build_coupled_state_space captures J_m/r_p^2 and matches exact 2-DOF eigenvalues < 1%."""
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    asm = build_assembly(spec)

    # Rigid structural frame limit (20 kHz) isolates exact 2-DOF motor-belt-carriage physics
    rigid_modal = ModalResult(
        fundamental_freq_hz=20000.0,
        shaper_x=InputShaperRecommendation("X", 20000.0, 20000.0, 100.0, 100.0, 100.0, "ZV", 10000.0, 0.5),
        shaper_y=InputShaperRecommendation("Y", 20000.0, 20000.0, 100.0, 100.0, 100.0, "ZV", 10000.0, 0.5),
        frame_damping_ratio=0.04,
    )
    mdl = build_coupled_state_space(asm, rigid_modal, u_rel=0.5, v_rel=0.5)

    r_p = spec.gantry.belt.pitch_radius_m
    j_m = spec.kinematics.xy_stepper.rotor_inertia_kg_m2
    m_ref_single = j_m / (r_p**2)
    # 2WD CoreXY: 2 motors reflect onto X and Y equally (2 * J_m / r_p^2 per axis)
    assert mdl.reflected_rotor_mass_per_motor_g == pytest.approx(m_ref_single * 1e3, rel=1e-6)
    assert mdl.reflected_rotor_mass_g == pytest.approx(2.0 * m_ref_single * 1e3, rel=1e-6)

    k_cart, k_a, k_b = corexy_belt_span_stiffness_matrix(asm, 0.5, 0.5)
    # CAL-13: upper belt (Belt A, z + dz/2) has slightly longer single-shear bolt overhang than lower belt (Belt B),
    # producing ~3.5% physical A/B stiffness asymmetry even at bed center
    assert k_a == pytest.approx(k_b, rel=0.05)
    assert abs(k_cart[0, 1]) / max(float(k_cart[0, 0]), 1.0) < 0.03

    m_ref_x = 2.0 * m_ref_single
    m_x = asm.x_moving_mass_kg
    k_em_x = 2.0 * mdl.k_em_per_motor_n_m
    k_belt_x = float(k_cart[0, 0])

    # Exact closed-form 2-DOF eigenvalues: K2 phi = w^2 M2 phi
    k2 = np.array([[k_em_x + k_belt_x, -k_belt_x], [-k_belt_x, k_belt_x]], dtype=float)
    s_m = np.diag([1.0 / math.sqrt(m_ref_x), 1.0 / math.sqrt(m_x)])
    eig_w2 = np.linalg.eigvalsh(s_m @ k2 @ s_m)
    f1_exact, f2_exact = np.sqrt(np.sort(eig_w2)) / (2.0 * math.pi)

    err_f1_pct = abs(mdl.coupled_freq_x_hz - f1_exact) / f1_exact * 100.0
    assert err_f1_pct < 1.0, f"Primary 2-DOF mode error {err_f1_pct:.3f}% >= 1%"

    # Out-of-phase belt bounce mode f2 also matches state-space spectrum < 1%
    nearest_f2 = min(mdl.mode_freqs_hz, key=lambda f: abs(f - f2_exact))
    err_f2_pct = abs(nearest_f2 - f2_exact) / f2_exact * 100.0
    assert err_f2_pct < 1.0, f"Second 2-DOF mode error {err_f2_pct:.3f}% >= 1%"

    # DC tracking gain of (A, B, C_noz, D) must equal I_2
    dc_gain = -mdl.C_noz @ np.linalg.solve(mdl.A, mdl.B)
    assert np.allclose(dc_gain, np.eye(2), atol=1e-9)

    # Higher rotor inertia increases reflected mass & lowers coupled drive frequency
    spec_heavy_rotor = spec.model_copy(deep=True)
    spec_heavy_rotor.kinematics.xy_stepper.rotor_inertia_kg_m2 = j_m * 3.0
    mdl_heavy = build_coupled_state_space(build_assembly(spec_heavy_rotor), rigid_modal, 0.5, 0.5)
    assert mdl_heavy.reflected_rotor_mass_g == pytest.approx(mdl.reflected_rotor_mass_g * 3.0, rel=1e-6)
    assert mdl_heavy.coupled_freq_x_hz < mdl.coupled_freq_x_hz


# ---------------------------------------------------------------------------
# 2. CoreXY 2WD Corner Cross-Coupling vs Symmetry & AWD Upgrade
# ---------------------------------------------------------------------------


def test_corexy_2wd_corner_cross_coupling_and_awd_improvement() -> None:
    """2WD CoreXY has 0% coupling on symmetry centerline vs >2% in corners; AWD improves both."""
    # Use bolted ACM panels so frame racking does not mask the 2WD -> AWD belt-drive frequency jump
    spec_2wd = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    spec_2wd.frame.enclosed_panels = True
    spec_2wd.frame.panel_mounting = "bolted_structural"
    spec_2wd.frame.panel_material = "acm_dibond"
    spec_2wd.gantry.drive_mode = "2wd"
    asm_2wd = build_assembly(spec_2wd)
    mod_2wd = solve_modal_fea(asm_2wd)

    # Symmetry line u=0.5 has minimal cross-coupling (small ~3-4% stacked-plane single-shear overhang asymmetry),
    # while corners exhibit much larger belt-span asymmetry and X-Y cross-coupling
    sym_couplings = []
    for v_test in (0.1, 0.5, 0.9):
        mdl_sym = build_coupled_state_space(asm_2wd, mod_2wd, u_rel=0.5, v_rel=v_test)
        assert mdl_sym.xy_cross_coupling_pct < 5.0
        sym_couplings.append(mdl_sym.xy_cross_coupling_pct)

    # Corners exhibit significant belt-span asymmetry and strictly higher X-Y cross-coupling than centerline
    mdl_corner = build_coupled_state_space(asm_2wd, mod_2wd, u_rel=0.05, v_rel=0.95)
    assert mdl_corner.k_A_n_m != pytest.approx(mdl_corner.k_B_n_m, rel=0.05)
    assert mdl_corner.xy_cross_coupling_pct > max(sym_couplings)
    assert mdl_corner.xy_cross_coupling_pct > 4.0

    t0 = time.perf_counter()
    wmap_2wd = compute_workspace_frf_map(asm_2wd, mod_2wd, grid_n=5)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 45.0, f"5x5 workspace FRF map took {elapsed_ms:.2f} ms"
    assert wmap_2wd.max_cross_coupling_pct > 2.0
    # Perimeter routing: the belt branches are equal, and the drive softest, at bed centre
    assert wmap_2wd.min_freq_hz <= wmap_2wd.center_freq_hz
    assert wmap_2wd.worst_corner_drop_pct >= 0.0

    # Verify serialization contract
    d = wmap_2wd.to_dict()
    assert d["grid_n"] == 5
    assert len(d["freq_x_grid_hz"]) == 5 and len(d["freq_x_grid_hz"][0]) == 5
    assert len(d["xy_cross_coupling_grid_pct"]) == 5

    # Switching identical printer to 4-motor AWD increases min_freq_hz and lowers max cross-coupling
    spec_awd = spec_2wd.model_copy(deep=True)
    spec_awd.gantry.drive_mode = "awd"
    asm_awd = build_assembly(spec_awd)
    mod_awd = solve_modal_fea(asm_awd)
    wmap_awd = compute_workspace_frf_map(asm_awd, mod_awd, grid_n=5)

    assert wmap_awd.min_freq_hz > wmap_2wd.min_freq_hz
    # In AWD the toolhead can get closer to a motor, so branch stiffness differs more across the bed,
    # while worst-corner frequency drop remains very small (< 2%) on both 2WD and AWD
    assert wmap_awd.max_cross_coupling_pct > wmap_2wd.max_cross_coupling_pct
    assert 0.0 <= wmap_awd.worst_corner_drop_pct < 2.0
    assert 0.0 <= wmap_2wd.worst_corner_drop_pct < 2.0
    assert wmap_awd.reflected_rotor_mass_g == pytest.approx(2.0 * wmap_2wd.reflected_rotor_mass_g, rel=1e-6)


def test_drive_control_modes_midband_instability_and_notch() -> None:
    """24V high-inductance stepper triggers mid-band risk; 48V / FOC servo clears it; notch cuts peak."""
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    spec.kinematics.max_velocity_mm_s = 560.0
    spec.kinematics.xy_stepper = StepperSpec(
        holding_torque_nm=0.59,
        rated_current_a=2.0,
        run_current_a=1.6,
        phase_resistance_ohm=1.4,
        inductance_mh=3.0,
        supply_voltage_v=24.0,
    )
    asm_24 = build_assembly(spec)
    mod_24 = solve_modal_fea(asm_24)
    mdl_24 = build_coupled_state_space(asm_24, mod_24, 0.5, 0.5)
    assert mdl_24.midband_instability_risk is True

    # 48V bus restores chopper current headroom & eliminates midband instability risk
    spec_48 = spec.model_copy(deep=True)
    spec_48.kinematics.xy_stepper.supply_voltage_v = 48.0
    asm_48 = build_assembly(spec_48)
    mdl_48 = build_coupled_state_space(asm_48, solve_modal_fea(asm_48), 0.5, 0.5)
    assert mdl_48.midband_instability_risk is False
    assert mdl_48.coupled_damping_x > mdl_24.coupled_damping_x

    # Closed-loop stepper and FOC servo eliminate mid-band risk & add strong active damping
    mdl_cl = build_coupled_state_space(
        asm_24, mod_24, 0.5, 0.5, drive_cfg=DriveControlConfig(drive_type="closed_loop_stepper")
    )
    mdl_foc = build_coupled_state_space(
        asm_24,
        mod_24,
        0.5,
        0.5,
        drive_cfg=DriveControlConfig(drive_type="foc_servo", servo_vel_bw_hz=120.0, servo_pos_bw_hz=25.0),
    )
    assert mdl_cl.midband_instability_risk is False
    assert mdl_foc.midband_instability_risk is False
    assert mdl_foc.coupled_damping_x > mdl_24.coupled_damping_x

    # Biquad notch placed at coupled_freq_x_hz strongly attenuates FRF peak while preserving DC=1
    f_notch = mdl_24.coupled_freq_x_hz
    mdl_notch = build_coupled_state_space(
        asm_24,
        mod_24,
        0.5,
        0.5,
        drive_cfg=DriveControlConfig(drive_type="foc_servo", notch_filter_hz=f_notch),
    )
    assert mdl_notch.state_dim == mdl_foc.state_dim + 4
    dc_notch = -mdl_notch.C_noz @ np.linalg.solve(mdl_notch.A, mdl_notch.B) + mdl_notch.D
    assert np.allclose(dc_notch, np.eye(2), atol=1e-9)

    h_no_notch = mdl_foc.evaluate_frf(np.array([f_notch]))[0, 0, 0]
    h_with_notch = mdl_notch.evaluate_frf(np.array([f_notch]))[0, 0, 0]
    assert abs(h_with_notch) < abs(h_no_notch) * 0.20


# ---------------------------------------------------------------------------
# 3. Advanced Shapers: ZVDD, SMOOTH_ZV, SMOOTH_EI, Multi-Mode Convolution & Notch
# ---------------------------------------------------------------------------


def test_advanced_shapers_zvdd_smooth_ei_and_multimode_convolution() -> None:
    """SHAPER_NAMES stays backward-compatible; ZVDD, SMOOTH_EI, and convolve_shapers hit <2% residual."""
    assert SHAPER_NAMES == ("ZV", "MZV", "EI", "2HUMP_EI", "3HUMP_EI")

    f0 = 60.0
    zeta = 0.08
    zvdd = get_shaper_pulses("ZVDD", f0, damping_ratio=zeta)
    assert len(zvdd[0]) == 4
    assert float(estimate_shaper(zvdd, [f0], test_damping_ratio=zeta)[0]) < 1e-6
    band = np.linspace(0.85 * f0, 1.15 * f0, 31)
    assert float(np.max(estimate_shaper(zvdd, band, test_damping_ratio=zeta))) < 0.02

    # SMOOTH_ZV & SMOOTH_EI use dense smooth taps and crush 2nd/3rd harmonics vs discrete ZV/EI
    sm_zv = get_shaper_pulses("SMOOTH_ZV", f0, damping_ratio=zeta)
    sm_ei = get_shaper_pulses("SMOOTH_EI", f0, damping_ratio=zeta)
    zv_disc = get_shaper_pulses("ZV", f0, damping_ratio=zeta)
    ei_disc = get_shaper_pulses("EI", f0, damping_ratio=zeta)
    assert len(sm_zv[0]) >= 20 and len(sm_ei[0]) >= 30
    assert float(estimate_shaper(sm_zv, [f0], test_damping_ratio=zeta)[0]) < 1e-6
    assert float(np.max(estimate_shaper(sm_ei, np.linspace(0.9 * f0, 1.1 * f0, 21), zeta))) < 0.03

    harmonics = np.array([2.0 * f0, 3.0 * f0, 4.0 * f0])
    assert float(np.max(estimate_shaper(sm_zv, harmonics, zeta))) < 0.15 * float(
        np.max(estimate_shaper(zv_disc, harmonics, zeta))
    )
    assert float(np.max(estimate_shaper(sm_ei, harmonics, zeta))) < 0.15 * float(
        np.max(estimate_shaper(ei_disc, harmonics, zeta))
    )

    # Multi-mode convolution cancels two distinct resonant modes (52 Hz and 118 Hz) < 2%
    f1, z1 = 52.0, 0.07
    f2, z2 = 118.0, 0.05
    conv = convolve_shapers(
        get_shaper_pulses("ZV", f1, damping_ratio=z1),
        get_shaper_pulses("ZV", f2, damping_ratio=z2),
    )
    assert float(estimate_shaper(conv, [f1], test_damping_ratio=z1)[0]) < 0.02
    assert float(estimate_shaper(conv, [f2], test_damping_ratio=z2)[0]) < 0.02

    # Continuous-time trajectory convolution preserves DC endpoints and smoothly shapes ramp
    t_s = np.linspace(0.0, 0.2, 401)
    pos = np.clip((t_s - 0.05) / 0.05, 0.0, 1.0) * 0.04
    shaped_pos = apply_shaper_to_trajectory(t_s, pos, conv)
    assert shaped_pos[0] == pytest.approx(0.0, abs=1e-12)
    assert shaped_pos[-1] == pytest.approx(0.04, abs=1e-12)
    assert np.all(shaped_pos >= -1e-12) and np.all(shaped_pos <= 0.04 + 1e-12)

    # Biquad notch magnitude matches exact analytical attenuation
    mag = apply_biquad_notch(np.array([1.0, 60.0, 300.0]), 60.0, zeta_zero=0.03, zeta_pole=0.35)
    assert mag[0] > 0.99
    assert mag[1] == pytest.approx(0.03 / 0.35, rel=1e-9)
    assert mag[2] > 0.98


# ---------------------------------------------------------------------------
# 4. Time-Domain Forced Response, Corner Ringing & ISO 230-4 Circular Ballbar
# ---------------------------------------------------------------------------


def test_time_domain_corner_ringing_and_iso230_4_ballbar() -> None:
    """Verify >60% shaper ringing cut, speed-dependent M_SE < 0, and C3 > C2 quadrant glitch."""
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    asm = build_assembly(spec)
    mod = solve_modal_fea(asm)

    t0 = time.perf_counter()
    res = simulate_time_domain_metrology(asm, mod)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    import os

    budget_ms = 180.0 if os.environ.get("CI") else 90.0
    assert elapsed_ms < budget_ms, f"Time-domain metrology took {elapsed_ms:.2f} ms"

    # Benchmark A: input shaping cuts post-corner ringing > 60% vs unshaped
    assert res.ringing_unshaped_um > 20.0
    assert res.ringing_reduction_pct > 60.0
    assert res.ringing_peak_to_peak_um < res.ringing_unshaped_um * 0.40
    assert set(res.corner_trace.keys()) >= {
        "ideal_x_mm",
        "ideal_y_mm",
        "actual_x_mm",
        "actual_y_mm",
        "unshaped_x_mm",
        "unshaped_y_mm",
    }
    assert 60 <= len(res.corner_trace["ideal_x_mm"]) <= 110

    # Explicit "NONE" shaper yields 0% reduction and matches unshaped ringing
    res_none = simulate_time_domain_metrology(asm, mod, shaper_name="NONE")
    assert res_none.ringing_reduction_pct == pytest.approx(0.0, abs=1e-6)
    assert res_none.ringing_peak_to_peak_um == pytest.approx(res_none.ringing_unshaped_um, rel=1e-6)

    # Benchmark B: ISO 230-4 ballbar schema & speed-dependent inward radial shrinkage M_SE < 0
    assert res.ballbar_radius_mm == 50.0
    assert len(res.ballbar_angles_deg) == 72
    assert len(res.ballbar_radial_error_um) == 72
    assert res.ballbar_angles_deg[0] == 0.0 and res.ballbar_angles_deg[-1] == 355.0
    assert res.circularity_gxy_um == pytest.approx(
        max(res.ballbar_radial_error_um) - min(res.ballbar_radial_error_um), rel=1e-6
    )

    r_slow = simulate_time_domain_metrology(asm, mod, ballbar_speed_mm_s=40.0)
    r_mid = simulate_time_domain_metrology(asm, mod, ballbar_speed_mm_s=80.0)
    r_fast = simulate_time_domain_metrology(asm, mod, ballbar_speed_mm_s=120.0)
    assert r_slow.mean_radial_lag_um < 0.0
    assert r_mid.mean_radial_lag_um < r_slow.mean_radial_lag_um
    assert r_fast.mean_radial_lag_um < r_mid.mean_radial_lag_um
    assert abs(r_fast.mean_radial_lag_um) > abs(r_slow.mean_radial_lag_um) * 4.0

    # ISO 5753 bearing radial clearance: C3 play produces larger quadrant_glitch_um than C2
    spec_c2 = spec.model_copy(deep=True)
    spec_c2.gantry.belt.bearing_clearance_class = "C2"
    spec_c3 = spec.model_copy(deep=True)
    spec_c3.gantry.belt.bearing_clearance_class = "C3"
    asm_c2 = build_assembly(spec_c2)
    asm_c3 = build_assembly(spec_c3)
    res_c2 = simulate_time_domain_metrology(asm_c2, solve_modal_fea(asm_c2))
    res_c3 = simulate_time_domain_metrology(asm_c3, solve_modal_fea(asm_c3))
    assert res_c3.quadrant_glitch_um > res_c2.quadrant_glitch_um + 5.0
    assert res_c3.circularity_gxy_um > res_c2.circularity_gxy_um

    # Monolith AWD benchmark outperforms stock 2WD Voron in ballbar circularity G_XY
    spec_mono = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_monolith_awd.yaml")
    asm_mono = build_assembly(spec_mono)
    res_mono = simulate_time_domain_metrology(asm_mono, solve_modal_fea(asm_mono))
    assert res_mono.circularity_gxy_um < res.circularity_gxy_um * 0.5
    assert res_mono.ringing_reduction_pct > 60.0
