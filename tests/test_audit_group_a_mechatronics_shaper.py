"""Regression and physics verification tests for Group A audit fixes (A-08, A-13, A-14, A-16, A-17, A-20, A-26)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import welch

from printsim.fea import solve_modal_fea
from printsim.fea.adxl_calibration import _compute_welch_nperseg, parse_adxl_csv
from printsim.fea.klipper_shaper import MIN_FREQ_HZ, calibrate_axis_shapers
from printsim.fea.mechatronics import DriveControlConfig, build_coupled_state_space
from printsim.gcode.ballbar import (
    _eval_trapezoid,
    _trapezoid_params,
    simulate_time_domain_metrology,
)
from printsim.geometry import build_assembly
from printsim.schema import PrinterSpec

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def test_a08_multihump_shaper_frequency_search_unpinned_and_grid_density() -> None:
    """A-08: 2HUMP_EI (~1.55x) and 3HUMP_EI (~1.85x+) are not pinned at 1.35x on 80 Hz & dual-peak PSDs."""
    g80 = calibrate_axis_shapers("X", 80.0, scv=5.0)
    by_name = {c.name: c for c in g80.candidates}

    c2 = by_name["2HUMP_EI"]
    c3 = by_name["3HUMP_EI"]

    # Previously both were pinned at 1.35 * 79.35 = 107.1 Hz (12,800 / 8,400 mm/s^2)
    assert c2.freq_hz > 118.0
    assert 120.0 <= c2.freq_hz <= 132.0
    assert c2.max_accel_mm_s2 >= 16500.0

    assert c3.freq_hz > 140.0
    assert 144.0 <= c3.freq_hz <= 160.0
    assert c3.max_accel_mm_s2 >= 15500.0
    assert c3.vibrations_pct < 1.0

    # Explicit multi-hump selection also uses widened coarse grid
    g_explicit = calibrate_axis_shapers("X", 80.0, scv=5.0, shaper_choice="3hump_ei")
    assert g_explicit.recommended_shaper == "3HUMP_EI"
    assert g_explicit.recommended_freq_hz == pytest.approx(c3.freq_hz, rel=1e-6)
    assert g_explicit.recommended_max_accel_mm_s2 == pytest.approx(c3.max_accel_mm_s2, rel=1e-6)

    # Check 45 Hz and dual-peak 52+66 Hz are also unpinned above 1.35x
    for f0, sec in ((45.0, []), (52.0, [(66.0, 1.0)])):
        g = calibrate_axis_shapers("X", f0, secondary_modes=sec, scv=5.0)
        cb = {c.name: c for c in g.candidates}
        assert cb["2HUMP_EI"].freq_hz > 1.45 * g.peak_freq_hz
        assert cb["3HUMP_EI"].freq_hz > 1.70 * g.peak_freq_hz


def test_a20_raw_adxl_sums_axes_and_applies_klipper_normalization() -> None:
    """A-20: Raw CSV path sums available accel axes, divides by (f + 0.1), and tapers < 2*MIN_FREQ."""
    fs_hz = 1600.0
    dur_s = 4.0
    t = np.arange(0.0, dur_s, 1.0 / fs_hz, dtype=float)
    f_start, f_end = 5.0, 133.0
    k_rate = (f_end - f_start) / dur_s
    phase = 2.0 * math.pi * (f_start * t + 0.5 * k_rate * t * t)
    f_inst = f_start + k_rate * t

    def _lorentz_amp(f_arr: np.ndarray, f0: float, zeta: float = 0.055) -> np.ndarray:
        r = f_arr / f0
        return (r / np.sqrt((1.0 - r * r) ** 2 + (2.0 * zeta * r) ** 2)) * (2.0 * zeta)

    # Raw excitation with 48 Hz primary, 110 Hz secondary, and 3 Hz low-frequency bench wobble
    sig_x = (
        _lorentz_amp(f_inst, 48.0) + 0.78 * _lorentz_amp(f_inst, 110.0)
    ) * np.sin(phase) + 1.8 * np.sin(2.0 * math.pi * 3.0 * t)
    sig_y = 0.02 * np.sin(2.0 * math.pi * 110.0 * t)
    sig_z = 9.81 + 0.01 * np.cos(2.0 * math.pi * 48.0 * t)

    # Compare raw unnormalized sum-Welch vs parse_adxl_csv Klipper-normalized output
    nperseg = _compute_welch_nperseg(len(t), fs_hz)
    f_w, px = welch(sig_x, fs=fs_hz, window="hann", nperseg=nperseg, detrend="constant", scaling="density")
    _, py = welch(sig_y, fs=fs_hz, window="hann", nperseg=nperseg, detrend="constant", scaling="density")
    _, pz = welch(sig_z, fs=fs_hz, window="hann", nperseg=nperseg, detrend="constant", scaling="density")
    raw_sum = np.maximum(px, 0.0) + np.maximum(py, 0.0) + np.maximum(pz, 0.0)

    lines = ["time,accel_x,accel_y,accel_z"]
    for ti, xi, yi, zi in zip(t, sig_x, sig_y, sig_z):
        lines.append(f"{ti:.6f},{xi:.6f},{yi:.6f},{zi:.6f}")
    parsed_3 = parse_adxl_csv("\n".join(lines), axis="X")

    m48 = (f_w >= 42.0) & (f_w <= 54.0)
    m110 = (f_w >= 102.0) & (f_w <= 118.0)
    m3 = (f_w >= 1.5) & (f_w <= 4.5)

    raw_ratio_110_48 = float(np.max(raw_sum[m110]) / np.max(raw_sum[m48]))
    norm_ratio_110_48 = float(np.max(parsed_3.psd[m110]) / np.max(parsed_3.psd[m48]))
    # Klipper 1/(f+0.1) scaling reduces 110/48 Hz peak ratio by ~48/110 = 0.436x
    assert norm_ratio_110_48 == pytest.approx(raw_ratio_110_48 * (48.0 / 110.0), rel=0.08)

    # Low-frequency < 2*MIN_FREQ_HZ (3 Hz wobble) is heavily suppressed by exponential taper
    taper_ratios = parsed_3.psd[m3] / np.maximum(raw_sum[m3], 1e-18)
    expected_taper = (1.0 / (f_w[m3] + 0.1)) * np.exp(
        -((2.0 * MIN_FREQ_HZ / (f_w[m3] + 0.1)) ** 2) + 1.0
    )
    assert np.allclose(taper_ratios, expected_taper, rtol=1e-6)
    assert float(taper_ratios[0]) < 1e-6

    # Rotated sensor: noise on accel_x, 64 Hz mode purely on accel_z -> still recovered via psd_x+y+z
    rng = np.random.default_rng(7)
    noise_x = rng.normal(0.0, 0.03, size=t.size)
    noise_y = rng.normal(0.0, 0.03, size=t.size)
    rot_z = 9.81 + 2.5 * np.sin(2.0 * math.pi * 64.0 * t)
    rot_lines = ["time,accel_x,accel_y,accel_z"]
    for ti, xi, yi, zi in zip(t, noise_x, noise_y, rot_z):
        rot_lines.append(f"{ti:.6f},{xi:.6f},{yi:.6f},{zi:.6f}")
    parsed_rot = parse_adxl_csv("\n".join(rot_lines), axis="X")
    band = (parsed_rot.freqs_hz >= MIN_FREQ_HZ) & (parsed_rot.freqs_hz <= 200.0)
    peak_rot = float(parsed_rot.freqs_hz[band][int(np.argmax(parsed_rot.psd[band]))])
    assert peak_rot == pytest.approx(64.0, abs=1.0)

    # 1-column and 2-column raw CSV formats also parse cleanly
    csv_1col = "\n".join(["time,accel"] + [f"{ti:.6f},{xi:.6f}" for ti, xi in zip(t, sig_x)])
    csv_2col = "\n".join(
        ["time,accel_x,accel_y"] + [f"{ti:.6f},{xi:.6f},{yi:.6f}" for ti, xi, yi in zip(t, sig_x, sig_y)]
    )
    p1 = parse_adxl_csv(csv_1col, axis="X")
    p2 = parse_adxl_csv(csv_2col, axis="X")
    assert float(p1.freqs_hz[band][int(np.argmax(p1.psd[band]))]) == pytest.approx(48.0, abs=2.0)
    assert float(p2.freqs_hz[band][int(np.argmax(p2.psd[band]))]) == pytest.approx(48.0, abs=2.0)


def test_a13_foc_gain_sizing_true_per_motor_inertia_corexy_and_cartesian() -> None:
    """A-13: CoreXY uses (m_cx+m_cy)/(2*n_mot) and Cartesian sizes X and Y motors independently."""
    omega_v = 2.0 * math.pi * 120.0
    omega_p = 2.0 * math.pi * 25.0
    foc_cfg = DriveControlConfig(drive_type="foc_servo", servo_vel_bw_hz=120.0, servo_pos_bw_hz=25.0)

    # 1. CoreXY 2WD
    spec_2wd = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    asm_2wd = build_assembly(spec_2wd)
    mod_2wd = solve_modal_fea(asm_2wd)
    mdl_2wd = build_coupled_state_space(asm_2wd, mod_2wd, 0.5, 0.5, drive_cfg=foc_cfg)
    m_ref_2wd = mdl_2wd.reflected_rotor_mass_per_motor_g * 1e-3
    m_car_2wd = (asm_2wd.x_moving_mass_kg + asm_2wd.y_moving_mass_kg) / 4.0
    expected_kem_2wd = (m_ref_2wd + m_car_2wd) * omega_v * omega_p
    assert mdl_2wd.k_em_per_motor_n_m == pytest.approx(expected_kem_2wd, rel=1e-6)

    # 2. CoreXY AWD
    spec_awd = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_monolith_awd.yaml")
    asm_awd = build_assembly(spec_awd)
    mod_awd = solve_modal_fea(asm_awd)
    mdl_awd = build_coupled_state_space(asm_awd, mod_awd, 0.5, 0.5, drive_cfg=foc_cfg)
    m_ref_awd = mdl_awd.reflected_rotor_mass_per_motor_g * 1e-3
    m_car_awd = (asm_awd.x_moving_mass_kg + asm_awd.y_moving_mass_kg) / 8.0
    expected_kem_awd = (m_ref_awd + m_car_awd) * omega_v * omega_p
    assert mdl_awd.k_em_per_motor_n_m == pytest.approx(expected_kem_awd, rel=1e-6)

    # 3. Cartesian bedslinger: X motor sized to m_cx, Y motor sized to m_cy (not averaged)
    spec_bs = PrinterSpec.from_yaml(EXAMPLES_DIR / "cartesian_bedslinger.yaml")
    asm_bs = build_assembly(spec_bs)
    mod_bs = solve_modal_fea(asm_bs)
    mdl_bs = build_coupled_state_space(asm_bs, mod_bs, 0.5, 0.5, drive_cfg=foc_cfg)
    assert mdl_bs.k_em_per_motor_arr is not None
    assert mdl_bs.b_em_per_motor_arr is not None
    m_ref_bs = mdl_bs.reflected_rotor_mass_per_motor_g * 1e-3
    exp_kx = (m_ref_bs + asm_bs.x_moving_mass_kg) * omega_v * omega_p
    exp_ky = (m_ref_bs + asm_bs.y_moving_mass_kg) * omega_v * omega_p
    assert mdl_bs.k_em_per_motor_arr[0] == pytest.approx(exp_kx, rel=1e-6)
    assert mdl_bs.k_em_per_motor_arr[1] == pytest.approx(exp_ky, rel=1e-6)
    assert mdl_bs.k_em_per_motor_arr[1] > mdl_bs.k_em_per_motor_arr[0] * 2.0

    # DC gain remains exact I_2 with asymmetric Cartesian per-motor gains
    dc_gain = -mdl_bs.C_noz @ np.linalg.solve(mdl_bs.A, mdl_bs.B)
    assert np.allclose(dc_gain, np.eye(2), atol=1e-9)


def test_a14_foc_ballbar_lag_not_double_counted_and_scales_quadratically() -> None:
    """A-14: foc_servo skips duplicate 1st-order tracking pre-filter while preserving M_SE < 0."""
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    asm = build_assembly(spec)
    mod = solve_modal_fea(asm)
    cfg = DriveControlConfig(drive_type="foc_servo", servo_vel_bw_hz=120.0, servo_pos_bw_hz=25.0)

    r_slow = simulate_time_domain_metrology(asm, mod, drive_cfg=cfg, ballbar_speed_mm_s=40.0)
    r_mid = simulate_time_domain_metrology(asm, mod, drive_cfg=cfg, ballbar_speed_mm_s=80.0)
    r_fast = simulate_time_domain_metrology(asm, mod, drive_cfg=cfg, ballbar_speed_mm_s=120.0)

    # Previously double-counted (-13.29 um at 120 mm/s); single plant+shaper lag is ~ -6.7 um
    assert -9.5 < r_fast.mean_radial_lag_um < -4.0
    assert r_slow.mean_radial_lag_um < 0.0
    assert r_mid.mean_radial_lag_um < r_slow.mean_radial_lag_um
    assert r_fast.mean_radial_lag_um < r_mid.mean_radial_lag_um
    assert abs(r_fast.mean_radial_lag_um) > abs(r_slow.mean_radial_lag_um) * 4.0

    # Monolith AWD FOC lag drops from -10.31 um (double-counted) to ~ -3.7 um
    spec_mono = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_monolith_awd.yaml")
    asm_mono = build_assembly(spec_mono)
    mod_mono = solve_modal_fea(asm_mono)
    r_mono = simulate_time_domain_metrology(asm_mono, mod_mono, drive_cfg=cfg, ballbar_speed_mm_s=120.0)
    assert -6.0 < r_mono.mean_radial_lag_um < -2.0


def test_a16_delta_reflected_rotor_mass_and_active_damping_use_jacobian_norm() -> None:
    """A-16: Delta reflected_rotor_mass_g and active damping scale by j_norm = mean(diag(T_cart^T T_cart))."""
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "delta_racer.yaml")
    asm = build_assembly(spec)
    mod = solve_modal_fea(asm)
    mdl = build_coupled_state_space(
        asm, mod, 0.5, 0.5, drive_cfg=DriveControlConfig(drive_type="foc_servo")
    )
    assert mdl.T_cart is not None
    assert mdl.T_cart.shape == (3, 2)

    j_norm = float(np.mean(np.diag(mdl.T_cart.T @ mdl.T_cart)))
    assert 0.4 < j_norm < 0.8
    assert mdl.reflected_rotor_mass_g == pytest.approx(
        j_norm * mdl.reflected_rotor_mass_per_motor_g, rel=1e-6
    )

    # Toolhead IMU active damping works cleanly on Delta with n_mot_axis = j_norm
    mdl_imu = build_coupled_state_space(
        asm,
        mod,
        0.5,
        0.5,
        drive_cfg=DriveControlConfig(drive_type="foc_servo", toolhead_imu_active_damping=True),
    )
    assert mdl_imu.coupled_damping_x > mdl.coupled_damping_x
    assert np.allclose(-mdl_imu.C_noz @ np.linalg.solve(mdl_imu.A, mdl_imu.B), np.eye(2), atol=1e-9)


def test_a17_corner_trapezoid_continuous_time_base_without_snap() -> None:
    """A-17: 90-deg corner benchmark samples both legs on one continuous analytical time base."""
    leg_len = 0.040
    v_c = 0.250
    v_scv = 0.005
    a_c = 8.0
    dt = 0.0005

    p1 = _trapezoid_params(leg_len, 0.0, v_c, v_scv, a_c)
    p2 = _trapezoid_params(leg_len, v_scv, v_c, 0.0, a_c)
    t_corner = float(p1[-1])
    # Ensure t_corner does not land on an exact dt sample for this non-trivial test
    assert abs((t_corner / dt) - round(t_corner / dt)) > 0.05

    t_a = np.arange(0.0, t_corner + float(p2[-1]) + 0.08 + 0.5 * dt, dt, dtype=float)
    x_ideal = _eval_trapezoid(t_a, p1) - leg_len
    y_ideal = _eval_trapezoid(t_a - t_corner, p2)

    # L1 path speed (vx + vy) must remain smooth across the corner junction (no 9.5 mm/s snap)
    v_path = (np.diff(x_ideal) + np.diff(y_ideal)) / dt
    idx_c = int(np.searchsorted(t_a, t_corner))
    assert v_path[idx_c - 1] <= (v_scv + a_c * dt) * 1.05
    assert v_path[idx_c] <= (v_scv + 1.5 * a_c * dt) * 1.05
    assert float(np.min(v_path[idx_c - 2 : idx_c + 2])) >= v_scv * 0.95


def test_a26_delta_ballbar_initializes_3_motor_states_cleanly() -> None:
    """A-26: Delta (n_mot == 3) ballbar starts cleanly from T_cart @ [0, R] with 4-fold periodic steady state."""
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "delta_racer.yaml")
    asm = build_assembly(spec)
    mod = solve_modal_fea(asm)

    # Equilibrium check: q_m = T_cart @ [0, R], q_c = [0, R] has zero net elastic force at rest
    mdl = build_coupled_state_space(asm, mod, 0.5, 0.5, drive_cfg=DriveControlConfig(drive_type="foc_servo"))
    assert mdl.T_cart is not None and mdl.n_mot == 3
    u_start = np.array([0.0, 0.050], dtype=float)
    x_eq = np.zeros(mdl.state_dim, dtype=float)
    x_eq[2 : 2 + mdl.n_mot] = mdl.T_cart @ u_start
    x_eq[2 + mdl.n_mot] = u_start[0]
    x_eq[3 + mdl.n_mot] = u_start[1]
    assert np.allclose(mdl.A @ x_eq + mdl.B @ u_start, 0.0, atol=1e-9)

    for drive in ("open_loop_stepper", "foc_servo"):
        cfg = DriveControlConfig(drive_type=drive)  # type: ignore[arg-type]
        res = simulate_time_domain_metrology(asm, mod, drive_cfg=cfg, ballbar_speed_mm_s=120.0)
        assert res.mean_radial_lag_um < 0.0
        assert res.circularity_gxy_um < 120.0
        # Axisymmetric Delta ballbar exhibits periodic 90-degree quadrant symmetry
        assert res.ballbar_radial_error_um[0] == pytest.approx(res.ballbar_radial_error_um[18], abs=0.5)
        assert res.ballbar_radial_error_um[18] == pytest.approx(res.ballbar_radial_error_um[36], abs=0.5)
        assert res.ballbar_radial_error_um[36] == pytest.approx(res.ballbar_radial_error_um[54], abs=0.5)
