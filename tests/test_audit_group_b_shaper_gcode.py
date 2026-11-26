"""Verification and regression tests for Audit Group B (B-15, B-16, B-17, B-18) & Group E fixes
across klipper_shaper.py, ballbar.py, and simulator.py.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from printsim.fea import (
    ADVANCED_SHAPER_NAMES,
    SHAPER_NAMES,
    InputShaperRecommendation,
    ModalResult,
    calibrate_axis_shapers,
    estimate_shaper,
    find_shaper_max_accel,
    get_shaper_pulses,
    solve_modal_fea,
    synthesize_axis_psd,
)
from printsim.fea.klipper_shaper import (
    NONE_SHAPER_ALLOWABLE_RINGING_MM,
    SHAPER_REGISTRY,
    TEST_DAMPING_RATIOS,
    ShaperCandidate,
    _fit_single_shaper,
    _select_klipper_best_shaper,
    _shaper_candidate_freqs,
    get_zvd_shaper,
)
from printsim.fea.modal import AxisShaperGraph
from printsim.gcode import simulate_gcode
from printsim.gcode.ballbar import simulate_time_domain_metrology
from printsim.geometry import build_assembly
from printsim.kinematics.solvers import KinematicsReport, analyze_kinematics
from printsim.schema import PrinterSpec
from printsim.thermal.filaments import FILAMENT_CATALOG
from printsim.thermal.hotend import simulate_hotend_thermal

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


# ---------------------------------------------------------------------------
# B-17 & Group E: Exact Klipper shaper_calibrate.py rules & ZVD in klipper_shaper.py
# ---------------------------------------------------------------------------


def test_b17_zvd_shaper_definition_and_registry() -> None:
    """ZVD is defined via get_zvd_shaper, registered in SHAPER_REGISTRY & ADVANCED_SHAPER_NAMES."""
    assert "ZVD" in ADVANCED_SHAPER_NAMES
    assert "ZVD" not in SHAPER_NAMES  # preserve 5-shaper backward compatibility
    assert "ZVD" in SHAPER_REGISTRY

    f0 = 65.0
    zeta = 0.1
    amps, times = get_zvd_shaper(f0, damping_ratio=zeta)
    df = math.sqrt(1.0 - zeta * zeta)
    k = math.exp(-zeta * math.pi / df)
    t_d = 1.0 / (f0 * df)

    assert len(amps) == 3
    assert np.allclose(amps, [1.0, 2.0 * k, k * k], rtol=1e-12)
    assert np.allclose(times, [0.0, 0.5 * t_d, t_d], rtol=1e-12)

    # Zero vibration AND zero derivative at design frequency f0 (wider notch than ZV)
    amps_p, times_p = get_shaper_pulses("ZVD", f0, damping_ratio=zeta)
    assert np.allclose(amps_p, amps)
    assert np.allclose(times_p, times)

    v_f0 = float(estimate_shaper((amps, times), [f0], test_damping_ratio=zeta)[0])
    assert v_f0 < 1e-6

    band = np.linspace(0.92 * f0, 1.08 * f0, 17)
    zv = get_shaper_pulses("ZV", f0, damping_ratio=zeta)
    assert float(np.max(estimate_shaper((amps, times), band, zeta))) < 0.2 * float(
        np.max(estimate_shaper(zv, band, zeta))
    )

    g_zvd = calibrate_axis_shapers("X", f0, scv=5.0, shaper_choice="zvd")
    assert g_zvd.recommended_shaper == "ZVD"
    assert g_zvd.recommended_max_accel_mm_s2 > 0.0


def test_b17_multi_damping_pessimisation_and_within_cross_selection() -> None:
    """Pessimises over TEST_DAMPING_RATIOS=(0.075,0.1,0.15) and enforces Klipper selection rules."""
    assert TEST_DAMPING_RATIOS == (0.075, 0.1, 0.15)

    # 1. Single-peak 60 Hz PSD: ZV with damping pessimisation reports non-zero residual (> 0.2%)
    #    unlike single-damping evaluation where ZV at 60 Hz had ~0.00% residual.
    g_single = calibrate_axis_shapers("X", 60.0, scv=5.0, shaper_choice="zv")
    zv_pess = g_single.candidates[0]
    assert zv_pess.name == "ZV"
    assert zv_pess.vibrations_pct > 0.2

    # Comparing against single test_damping_ratio=(0.1,) confirms pessimisation increases reported vibration
    g_single_dr = calibrate_axis_shapers(
        "X", 60.0, scv=5.0, shaper_choice="zv", test_damping_ratios=(0.1,)
    )
    assert zv_pess.vibrations_pct > g_single_dr.candidates[0].vibrations_pct

    # 2. Within-shaper frequency selection rule: pick lowest-score within 1.1*min_v + 0.0005
    #    (not strictly min_v if a slightly higher frequency achieves lower score within tolerance)
    freqs_w, psd_w = synthesize_axis_psd("X", 60.0, secondary_modes=[(72.0, 0.8)])
    v_thr = 100.0 / 20.0
    all_v = float(np.sum(np.maximum(psd_w - v_thr, 0.0)))
    c_freqs = _shaper_candidate_freqs("ZV", 60.0)
    cand_zv = _fit_single_shaper("ZV", freqs_w, psd_w, c_freqs, v_thr, all_v, scv=5.0, damping_ratio=0.1)

    # Evaluate dense grid independently to confirm cand_zv respects min_v * 1.1 + 0.0005 and lowers score
    min_v_ref = min(
        max(
            float(np.sum(np.maximum(psd_w * estimate_shaper(get_shaper_pulses("ZV", float(fc), 0.1), freqs_w, dr) - v_thr, 0.0))) / all_v
            for dr in TEST_DAMPING_RATIOS
        )
        for fc in np.arange(float(c_freqs[0]), float(c_freqs[-1]) + 0.125, 0.25)
    )
    assert cand_zv.vibrations_pct / 100.0 <= min_v_ref * 1.1 + 0.0006

    # 3. Cross-shaper rule: 1.2x score improvement OR (1.05x score and 1.1x smoothing)
    dummy_att = np.ones(10, dtype=float)
    zv_c = ShaperCandidate("ZV", 60.0, 2.0, 0.060, 10000.0, 0.00100, dummy_att)
    # Only 12% better score (0.00100 vs 0.00089) and MORE smoothing -> Klipper keeps ZV
    mzv_marginal = ShaperCandidate("MZV", 61.0, 0.2, 0.070, 8200.0, 0.00089, dummy_att)
    assert _select_klipper_best_shaper([zv_c, mzv_marginal]).name == "ZV"

    # >20% better score (0.00080 * 1.2 = 0.00096 < 0.00100) -> Klipper upgrades to MZV
    mzv_clear = ShaperCandidate("MZV", 61.0, 0.1, 0.070, 8200.0, 0.00080, dummy_att)
    assert _select_klipper_best_shaper([zv_c, mzv_clear]).name == "MZV"

    # >5% better score AND >10% less smoothing -> upgrades
    better_sm = ShaperCandidate("EI", 80.0, 0.1, 0.050, 12000.0, 0.00094, dummy_att)
    assert _select_klipper_best_shaper([zv_c, better_sm]).name == "EI"


def test_b17_zero_max_accel_on_unattainable_smoothing_and_none_constant() -> None:
    """find_shaper_max_accel returns 0.0 when 100 mm/s^2 exceeds target_smoothing; NONE uses named constant."""
    # Very low frequency 3HUMP_EI cannot satisfy 0.12 mm smoothing even at 100 mm/s^2
    slow_3hump = get_shaper_pulses("3HUMP_EI", 12.0, damping_ratio=0.1)
    assert find_shaper_max_accel(slow_3hump, scv=5.0, target_smoothing=0.12) == 0.0

    # Group E: NONE shaper uses NONE_SHAPER_ALLOWABLE_RINGING_MM * (2*pi*f)^2
    assert NONE_SHAPER_ALLOWABLE_RINGING_MM == pytest.approx(0.0228, abs=1e-6)
    f_test = 65.0
    g_none = calibrate_axis_shapers("X", f_test, scv=5.0, shaper_choice="none")
    expected_none_accel = NONE_SHAPER_ALLOWABLE_RINGING_MM * ((2.0 * math.pi * f_test) ** 2)
    assert g_none.recommended_max_accel_mm_s2 == pytest.approx(expected_none_accel, rel=1e-9)


# ---------------------------------------------------------------------------
# B-18: Synthetic PSD secondary-mode weighting (1/f-normalized Klipper sweep)
# ---------------------------------------------------------------------------


def test_b18_synthetic_psd_secondary_mode_scales_with_frequency() -> None:
    """Secondary modes within primary drive passband scale with f (not 1/f^2 displacement atten)."""
    f_p = 50.0
    p = 0.5
    freqs = np.linspace(5.0, 220.0, 431, dtype=float)  # 0.5 Hz bins for exact peak reading

    _, psd_0 = synthesize_axis_psd("X", f_p, secondary_modes=[], freqs_hz=freqs)
    _, psd_40 = synthesize_axis_psd("X", f_p, secondary_modes=[(40.0, p)], freqs_hz=freqs)
    _, psd_60 = synthesize_axis_psd("X", f_p, secondary_modes=[(60.0, p)], freqs_hz=freqs)
    _, psd_75 = synthesize_axis_psd("X", f_p, secondary_modes=[(75.0, p)], freqs_hz=freqs)

    def _isolated_secondary_peak(psd_arr: np.ndarray, f_target: float) -> float:
        mask = np.abs(freqs - f_target) <= 1.0
        return float(np.max(psd_arr[mask] - psd_0[mask]))

    h_40 = _isolated_secondary_peak(psd_40, 40.0)
    h_60 = _isolated_secondary_peak(psd_60, 60.0)
    h_75 = _isolated_secondary_peak(psd_75, 75.0)

    # Under 1/f-normalized acceleration sweep, higher modes in passband are amplified, not suppressed by 1/f^2
    assert h_75 > h_60 > h_40
    # Old 1/f^2 weighting gave 0.25*0.45*(50/75)^2 = 5.0% at 75 Hz; Klipper sweep gives > 10%
    assert h_75 > 10.0
    # Primary peak stays dominant at 100.0
    assert float(np.max(psd_75[np.abs(freqs - f_p) <= 1.5])) == pytest.approx(100.0, abs=0.5)


# ---------------------------------------------------------------------------
# B-15: Ballbar open-loop stepper radial lag without hardcoded 11 ms fudge
# ---------------------------------------------------------------------------


def test_b15_open_loop_stepper_ballbar_depends_on_machine_and_shaper() -> None:
    """Dropping 11 ms constant leaves realistic machine/shaper/L-R-driven M_SE < 0."""
    spec_v24 = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    asm_v24 = build_assembly(spec_v24)
    mod_v24 = solve_modal_fea(asm_v24)

    # 1. With NONE shaper and near-zero inductance, open-loop stepper has NO ~ -21 um fake lag
    spec_no_ind = spec_v24.model_copy(deep=True)
    spec_no_ind.kinematics.xy_stepper.inductance_mh = 0.05
    asm_no_ind = build_assembly(spec_no_ind)
    res_none = simulate_time_domain_metrology(
        asm_no_ind, mod_v24, shaper_name="NONE", ballbar_speed_mm_s=120.0
    )
    # Without shaper or 11 ms fudge, compliant belt gives small outward centripetal stretch (not -21 um lag)
    assert -1.0 < res_none.mean_radial_lag_um < 5.0

    # 2. Shaper choice drives radial lag directly: |EI| > |MZV| > |ZV|
    res_zv = simulate_time_domain_metrology(asm_v24, mod_v24, shaper_name="ZV", ballbar_speed_mm_s=120.0)
    res_mzv = simulate_time_domain_metrology(asm_v24, mod_v24, shaper_name="MZV", ballbar_speed_mm_s=120.0)
    res_ei = simulate_time_domain_metrology(asm_v24, mod_v24, shaper_name="EI", ballbar_speed_mm_s=120.0)
    assert res_ei.mean_radial_lag_um < res_mzv.mean_radial_lag_um < res_zv.mean_radial_lag_um < 0.0
    assert -8.0 < res_mzv.mean_radial_lag_um < -1.0

    # 3. Stiff high-frequency machine (Monolith AWD) has far smaller inward lag than 2WD Voron
    spec_mono = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_monolith_awd.yaml")
    asm_mono = build_assembly(spec_mono)
    mod_mono = solve_modal_fea(asm_mono)
    res_mono = simulate_time_domain_metrology(asm_mono, mod_mono, ballbar_speed_mm_s=120.0)
    assert res_mono.mean_radial_lag_um < 0.0
    assert abs(res_mono.mean_radial_lag_um) < abs(res_mzv.mean_radial_lag_um) * 0.35

    # 4. Electrical lag tau_e = L / R still contributes physically when inductance increases
    spec_hi_l = spec_v24.model_copy(deep=True)
    spec_hi_l.kinematics.xy_stepper.inductance_mh = 8.0
    asm_hi_l = build_assembly(spec_hi_l)
    res_hi_l = simulate_time_domain_metrology(
        asm_hi_l, mod_v24, shaper_name="MZV", ballbar_speed_mm_s=120.0
    )
    assert res_hi_l.mean_radial_lag_um < res_mzv.mean_radial_lag_um - 0.5


# ---------------------------------------------------------------------------
# B-16 & Group E: Physical junction ringing, corner smoothing, filament density, belt slip
# ---------------------------------------------------------------------------


def _make_mock_modal_and_kinematics(
    spec: PrinterSpec,
    freq_hz: float = 50.0,
    zeta: float = 0.05,
    shaper_name: str = "NONE",
    belt_max_accel: float = 100000.0,
    motor_safe_accel: float = 100000.0,
) -> tuple[ModalResult, KinematicsReport]:
    rec_x = InputShaperRecommendation("X", freq_hz, freq_hz, freq_hz, freq_hz, freq_hz, shaper_name, 8000.0, 0.2)
    rec_y = InputShaperRecommendation("Y", freq_hz, freq_hz, freq_hz, freq_hz, freq_hz, shaper_name, 8000.0, 0.2)
    freqs = np.linspace(5.0, 220.0, 108)
    psd = np.ones_like(freqs) * 100.0
    cand = ShaperCandidate(shaper_name, freq_hz, 1.0, 0.05, 8000.0, 0.001, np.ones_like(freqs))
    gx = AxisShaperGraph("X", freqs, psd, psd, freq_hz, zeta, shaper_name, freq_hz, 8000.0, [cand])
    gy = AxisShaperGraph("Y", freqs, psd, psd, freq_hz, zeta, shaper_name, freq_hz, 8000.0, [cand])
    mod = ModalResult(
        fundamental_freq_hz=freq_hz,
        shaper_x=rec_x,
        shaper_y=rec_y,
        shaper_graph_x=gx,
        shaper_graph_y=gy,
        frame_damping_ratio=zeta,
    )
    kin = KinematicsReport(
        architecture=spec.architecture,
        build_volume_liters=27.0,
        frame_volume_liters=64.0,
        packaging_efficiency_pct=42.0,
        max_motor_rpm=500.0,
        available_torque_at_vmax_nm=0.4,
        required_torque_x_nm=0.1,
        required_torque_y_nm=0.1,
        torque_safety_margin=4.0,
        max_safe_accel_motor_mm_s2=motor_safe_accel,
        step_skip_risk=False,
        belt_peak_tension_n=35.0,
        belt_slip_risk=False,
        delta_workspace_valid=True,
        belt_max_accel_mm_s2=belt_max_accel,
    )
    return mod, kin


def test_b16_physical_ringing_scaling_dv_omega_n_shaper_and_smoothing() -> None:
    """Ringing scales with min(dv/omega_n, a/omega_n^2) * V_shaper; smoothing is tracked separately."""
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    spec.kinematics.max_accel_mm_s2 = 10000.0
    spec.kinematics.max_velocity_mm_s = 300.0
    spec.kinematics.square_corner_velocity_mm_s = 5.0
    asm = build_assembly(spec, panel_mesh_n=2)
    th = simulate_hotend_thermal(spec, filament_id="pla")

    # Sharp 90-degree corner G-code
    gcode_90 = "\n".join(
        [
            "G90",
            "M83",
            "G0 X50 Y50 F12000",
            "G1 X90 Y50 E1.5 F12000",
            "G1 X90 Y90 E1.5 F12000",
        ]
    )

    # 1. NONE shaper on 50 Hz frame at 10,000 mm/s^2: amp ~ a / omega_n^2 = 10 / (2*pi*50)^2 = 101.3 um
    spec_none = spec.model_copy(deep=True)
    spec_none.kinematics.input_shaper = "none"
    mod_none, kin_ok = _make_mock_modal_and_kinematics(spec_none, freq_hz=50.0, zeta=0.05, shaper_name="NONE")
    vp_none = simulate_gcode(gcode_90, asm, mod_none, kin_ok, th)
    expected_unshaped_um = (10.0 / ((2.0 * math.pi * 50.0) ** 2)) * 1e6
    assert vp_none.max_ringing_amplitude_um == pytest.approx(expected_unshaped_um, rel=1e-4)
    assert vp_none.ringing_segments >= 1
    assert vp_none.segments[2].defect == "ringing"
    assert vp_none.max_corner_smoothing_um == pytest.approx(0.0, abs=1e-9)

    # 2. Doubling modal frequency 50 Hz -> 100 Hz cuts ringing amplitude by 4x (1 / omega_n^2)
    mod_100, _ = _make_mock_modal_and_kinematics(spec_none, freq_hz=100.0, zeta=0.05, shaper_name="NONE")
    vp_100 = simulate_gcode(gcode_90, asm, mod_100, kin_ok, th)
    assert vp_100.max_ringing_amplitude_um == pytest.approx(expected_unshaped_um / 4.0, rel=1e-4)

    # 3. Active MZV shaper crushes ringing amplitude while separately tracking corner smoothing
    spec_mzv = spec.model_copy(deep=True)
    spec_mzv.kinematics.input_shaper = "mzv"
    mod_mzv, _ = _make_mock_modal_and_kinematics(spec_mzv, freq_hz=50.0, zeta=0.05, shaper_name="MZV")
    vp_mzv = simulate_gcode(gcode_90, asm, mod_mzv, kin_ok, th)
    assert vp_mzv.max_ringing_amplitude_um < vp_none.max_ringing_amplitude_um * 0.15
    assert vp_mzv.ringing_segments == 0
    assert vp_mzv.max_corner_smoothing_um > 120.0  # 10,000 mm/s^2 at 50 Hz MZV exceeds 0.12 mm smoothing
    assert vp_mzv.smoothing_segments >= 1
    assert vp_mzv.segments[2].defect == "smoothing"

    # 4. Small velocity change (dv/omega_n < a/omega_n^2) scales linearly with dv, NOT a/omega_n^2
    gcode_5deg = "\n".join(
        [
            "G90",
            "M83",
            "G0 X50 Y50 F6000",
            "G1 X90 Y50 E1.5 F6000",  # 100 mm/s +X
            f"G1 X{90 + 40*math.cos(math.radians(5.0)):.4f} Y{50 + 40*math.sin(math.radians(5.0)):.4f} E1.5 F6000",
        ]
    )
    vp_5deg = simulate_gcode(gcode_5deg, asm, mod_none, kin_ok, th)
    # 5-degree turn at 100 mm/s -> dv ~ 2 * 100 * sin(2.5 deg) = 8.72 mm/s -> dv/omega_n ~ 27.8 um < 101.3 um
    assert 20.0 < vp_5deg.segments[2].ringing_amplitude_um < 35.0

    # 5. Collinear constant-speed continuation has zero velocity jump -> zero ringing
    gcode_line = "\n".join(
        [
            "G90",
            "M83",
            "G0 X50 Y50 F6000",
            "G1 X90 Y50 E1.5 F6000",
            "G1 X130 Y50 E1.5 F6000",
        ]
    )
    vp_line = simulate_gcode(gcode_line, asm, mod_none, kin_ok, th)
    assert vp_line.segments[2].ringing_amplitude_um == pytest.approx(0.0, abs=1e-9)
    assert vp_line.segments[2].ringing_severity == pytest.approx(0.0, abs=1e-9)


def test_group_e_simulator_filament_density_and_belt_slip_detection() -> None:
    """simulator.py uses FILAMENT_CATALOG density and flags belt slip against belt_max_accel_mm_s2."""
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    spec.kinematics.max_accel_mm_s2 = 12000.0
    asm = build_assembly(spec, panel_mesh_n=2)
    mod, kin_ok = _make_mock_modal_and_kinematics(
        spec, freq_hz=80.0, zeta=0.05, shaper_name="ZV", belt_max_accel=50000.0, motor_safe_accel=50000.0
    )

    gcode = "\n".join(
        [
            "G90",
            "M83",
            "G0 X50 Y50 F12000",
            "G1 X90 Y50 E2.0 F12000",
            "G1 X90 Y90 E2.0 F12000",
        ]
    )

    # 1. Filament mass scales strictly with FILAMENT_CATALOG[filament_id].density_kg_m3
    vp_pla = simulate_gcode(gcode, asm, mod, kin_ok, simulate_hotend_thermal(spec, filament_id="pla"))
    vp_abs = simulate_gcode(gcode, asm, mod, kin_ok, simulate_hotend_thermal(spec, filament_id="abs"))
    vp_petg = simulate_gcode(gcode, asm, mod, kin_ok, simulate_hotend_thermal(spec, filament_id="petg"))

    assert (vp_abs.filament_used_g / vp_pla.filament_used_g) == pytest.approx(
        FILAMENT_CATALOG["abs"].density_kg_m3 / FILAMENT_CATALOG["pla"].density_kg_m3,
        rel=1e-6,
    )
    assert (vp_petg.filament_used_g / vp_pla.filament_used_g) == pytest.approx(
        FILAMENT_CATALOG["petg"].density_kg_m3 / FILAMENT_CATALOG["pla"].density_kg_m3,
        rel=1e-6,
    )

    # 2. Belt slip detected when move accel > belt_max_accel_mm_s2 even if motor accel margin is fine
    _, kin_belt_lim = _make_mock_modal_and_kinematics(
        spec, freq_hz=80.0, zeta=0.05, shaper_name="ZV", belt_max_accel=9000.0, motor_safe_accel=50000.0
    )
    vp_belt_slip = simulate_gcode(
        gcode, asm, mod, kin_belt_lim, simulate_hotend_thermal(spec, filament_id="pla")
    )
    assert vp_belt_slip.step_skip_segments == 0
    assert vp_belt_slip.belt_slip_segments == 2
    assert vp_belt_slip.segments[1].defect == "belt_slip"
    assert vp_belt_slip.print_quality_score < vp_pla.print_quality_score
