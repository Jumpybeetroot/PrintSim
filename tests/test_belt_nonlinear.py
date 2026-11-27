"""Physics benchmarks, ODE-vs-HB cross-validation, and runtime tests for `belt_nonlinear.py`."""

from __future__ import annotations

import math
import subprocess
import sys
import time

import numpy as np
import pytest

from printsim.fea.belt_nonlinear import (
    BeltBranchSpec,
    NonlinearBeltDriveSpec,
    _CompiledBeltDrive,
    _build_describing_function_table,
    simulate_swept_sine_time_domain,
    solve_linear_coupled_drive_2dof,
    solve_nonlinear_belt_shaker_response,
    transverse_span_natural_freq_hz,
)
from printsim.fea.modal import _solve_coupled_drive_2dof


def test_linear_high_tension_limit_matches_exact_2dof_eigenproblem() -> None:
    """High-tension limit (T0 >> dT_shaker) matches exact 2-DOF eigenproblem to < 0.5%."""
    k_belt = 110_000.0
    k_motor = 240_000.0
    k_bkt = 950_000.0
    m_car = 0.55
    m_rot = 0.28

    # 1. Exact algebraic agreement with modal._solve_coupled_drive_2dof
    ref = _solve_coupled_drive_2dof(k_belt, k_motor, k_bkt, m_car, m_rot)
    ours = solve_linear_coupled_drive_2dof(k_belt, k_motor, k_bkt, m_car, m_rot)
    f_exact_hz = math.sqrt(ref["w1_sq"]) / (2.0 * math.pi)
    assert math.isclose(ours["f1_hz"], f_exact_hz, rel_tol=1e-12)
    assert math.isclose(ours["c_drive_n_inv"], ref["c_drive_n_inv"], rel_tol=1e-12)
    assert math.isclose(ours["u_belt_frac"], ref["u_belt_frac"], rel_tol=1e-12)

    # 2. Fast HB solver in high-tension linear limit (with rotor inertia)
    drive = NonlinearBeltDriveSpec(
        m_car_eff_kg=m_car,
        m_rot_kg=m_rot,
        k_motor_raw_n_m=k_motor,
        k_bracket_n_m=k_bkt,
        pretension_n=250.0,
        k_belt_linear_n_m=k_belt,
        n_loops=2.0,
        slack_load_share=0.5,
        longest_free_span_m=0.35,
        belt_linear_mass_kg_m=0.0085,
        belt_ei_nm2=9.0e-6,
        zeta_belt_base=0.045,
        zeta_motor=0.050,
        zeta_car=0.010,
    )
    hb = solve_nonlinear_belt_shaker_response(drive)
    rel_err_hb = abs(hb.effective_freq_hz - f_exact_hz) / f_exact_hz
    assert rel_err_hb < 0.005, f"HB linear-limit error {rel_err_hb:.4%} >= 0.5%"
    assert hb.stiffness_ratio > 0.995
    assert hb.slack_duty_cycle == 0.0
    assert hb.min_branch_tension_n > 200.0
    assert hb.whip_participation < 0.05

    # 3. Massless-rotor 1-DOF linear limit
    drive_1dof = NonlinearBeltDriveSpec(
        m_car_eff_kg=0.60,
        m_rot_kg=0.0,
        k_motor_raw_n_m=220_000.0,
        k_bracket_n_m=1_200_000.0,
        pretension_n=250.0,
        k_belt_linear_n_m=90_000.0,
        n_loops=1.0,
    )
    ref_1dof = _solve_coupled_drive_2dof(90_000.0, 220_000.0, 1_200_000.0, 0.60, 0.0)
    f_1dof_exact = math.sqrt(ref_1dof["w1_sq"]) / (2.0 * math.pi)
    hb_1dof = solve_nonlinear_belt_shaker_response(drive_1dof)
    assert abs(hb_1dof.effective_freq_hz - f_1dof_exact) / f_1dof_exact < 0.005

    # 4. Full time-domain ODE reference also matches exact 2-DOF frequency closely
    td = simulate_swept_sine_time_domain(
        drive,
        f_start_hz=15.0,
        f_end_hz=115.0,
        sweep_rate_hz_per_s=12.0,
        sample_rate_hz=2000.0,
    )
    rel_err_td = abs(td.peak_freq_hz - f_exact_hz) / f_exact_hz
    assert rel_err_td < 0.03, f"Time-domain linear-limit error {rel_err_td:.4%} >= 3%"
    assert td.min_branch_tension_n > 200.0
    assert td.slack_time_fraction == 0.0


def test_closed_form_symmetric_bilinear_describing_function() -> None:
    """Cycle quadrature matches exact analytical symmetric bilinear spring describing function."""
    k_belt = 80_000.0
    t0 = 10.0
    # Very stiff bracket so delta_b ~ delta
    drive = NonlinearBeltDriveSpec(
        m_car_eff_kg=0.5,
        m_rot_kg=0.0,
        k_motor_raw_n_m=1e9,
        k_bracket_n_m=1e11,
        pretension_n=t0,
        k_belt_linear_n_m=k_belt,
        n_loops=2.0,
        slack_load_share=0.5,
        include_transverse_coupling=False,
    )
    model = _CompiledBeltDrive(drive)
    # Single branch stiffness k_branch = k_belt / (2 * n_loops) = 20_000 N/m -> delta_s = t0 / k_branch = 0.5 mm
    delta_s = t0 / (k_belt / 4.0)
    table = _build_describing_function_table(model, a_max=5.0 * delta_s, n_amp=80, n_theta=96)

    for ratio in (0.5, 1.5, 2.0, 4.0):
        amp = ratio * delta_s
        k_num = float(np.interp(amp, table["a_grid"], table["k_bb_eq"]))
        if ratio <= 1.0:
            k_exact = k_belt
        else:
            x = 1.0 / ratio
            k_exact = 0.5 * k_belt + 0.5 * k_belt * (2.0 / math.pi) * (
                math.asin(x) + x * math.sqrt(1.0 - x * x)
            )
        rel_err = abs(k_num - k_exact) / k_exact
        assert rel_err < 0.005, f"ratio={ratio}: k_num={k_num:.1f} vs k_exact={k_exact:.1f} ({rel_err:.3%})"


def test_low_tension_bilinear_slackening_matches_time_domain_ode() -> None:
    """Low-tension regime (T0 ~ dT_shaker) exhibits bilinear frequency drop & peak broadening matching ODE."""
    common_kwargs = dict(
        m_car_eff_kg=0.65,
        m_rot_kg=0.20,
        k_motor_raw_n_m=280_000.0,
        k_bracket_n_m=1_500_000.0,
        k_belt_linear_n_m=100_000.0,
        n_loops=2.0,
        slack_load_share=0.55,
        zeta_belt_base=0.040,
        zeta_motor=0.045,
        zeta_car=0.010,
        include_transverse_coupling=False,  # isolate pure longitudinal bilinear slackening first
    )

    drive_hi = NonlinearBeltDriveSpec(pretension_n=120.0, **common_kwargs)
    drive_lo = NonlinearBeltDriveSpec(pretension_n=4.0, **common_kwargs)

    hb_hi = solve_nonlinear_belt_shaker_response(drive_hi)
    hb_lo = solve_nonlinear_belt_shaker_response(drive_lo)

    td_hi = simulate_swept_sine_time_domain(
        drive_hi, f_start_hz=15.0, f_end_hz=110.0, sweep_rate_hz_per_s=8.0, sample_rate_hz=2000.0
    )
    td_lo = simulate_swept_sine_time_domain(
        drive_lo, f_start_hz=15.0, f_end_hz=110.0, sweep_rate_hz_per_s=8.0, sample_rate_hz=2000.0
    )

    # Both HB and ODE show clear bilinear slackening frequency drop at low tension
    assert hb_lo.min_branch_tension_n <= 1e-6
    assert hb_lo.slack_duty_cycle > 0.15
    assert hb_lo.stiffness_ratio < 0.88
    assert hb_lo.effective_freq_hz < 0.96 * hb_hi.effective_freq_hz
    assert hb_lo.bilinear_softening_shift_hz > 2.5
    assert hb_lo.softening_shoulder_hz is not None
    assert hb_lo.softening_shoulder_hz < hb_lo.effective_freq_hz
    assert hb_lo.softening_shoulder_participation > 0.15

    assert td_lo.min_branch_tension_n <= 1e-6
    assert td_lo.slack_time_fraction > 0.04
    assert td_lo.peak_freq_hz < 0.97 * td_hi.peak_freq_hz

    # Both HB and ODE show strong nonlinear peak broadening (higher apparent damping ratio at low tension)
    assert hb_lo.zeta_nl > 1.35 * hb_hi.zeta_nl
    assert td_lo.apparent_damping_ratio > 1.30 * td_hi.apparent_damping_ratio

    # Quantitative agreement between fast HB and full time-domain ODE reference
    freq_agreement_err = abs(hb_lo.effective_freq_hz - td_lo.peak_freq_hz) / td_lo.peak_freq_hz
    assert freq_agreement_err < 0.05, (
        f"HB low-tension peak {hb_lo.effective_freq_hz:.2f} Hz vs ODE {td_lo.peak_freq_hz:.2f} Hz "
        f"(err={freq_agreement_err:.2%})"
    )

    # Monotonic softening and broadening as pretension drops
    tensions = (100.0, 12.0, 5.0, 2.0)
    results = [
        solve_nonlinear_belt_shaker_response(NonlinearBeltDriveSpec(pretension_n=t, **common_kwargs))
        for t in tensions
    ]
    freqs = [r.effective_freq_hz for r in results]
    ks = [r.k_belt_eff_n_m for r in results]
    zetas = [r.zeta_nl for r in results]
    assert freqs[0] >= freqs[1] >= freqs[2] > freqs[3]
    assert ks[0] >= ks[1] >= ks[2] > ks[3]
    assert zetas[3] > zetas[2] > zetas[0]


def test_nonlinear_tangent_ea_fn_and_asymmetric_branches() -> None:
    """Tension-dependent tangent ea_fn(T) (cord-crimp toe) integrates cleanly in both HB and ODE."""
    ea_nom = 42_000.0

    def crimp_ea_fn(tension_n: float) -> float:
        q = max(float(tension_n), 0.1) / 6.0  # 6 mm belt line tension (N/mm)
        return ea_nom * (1.0 - 0.52 * math.exp(-q / 2.3))

    t0 = 8.0
    branches_const = (
        BeltBranchSpec(length_m=0.30, pretension_n=t0, ea_n=crimp_ea_fn(t0), sign=1.0, multiplicity=2.0),
        BeltBranchSpec(length_m=0.60, pretension_n=t0, ea_n=crimp_ea_fn(t0), sign=-1.0, multiplicity=2.0),
    )
    branches_nl = (
        BeltBranchSpec(
            length_m=0.30, pretension_n=t0, ea_n=crimp_ea_fn(t0), sign=1.0, multiplicity=2.0, ea_fn=crimp_ea_fn
        ),
        BeltBranchSpec(
            length_m=0.60, pretension_n=t0, ea_n=crimp_ea_fn(t0), sign=-1.0, multiplicity=2.0, ea_fn=crimp_ea_fn
        ),
    )

    drive_const = NonlinearBeltDriveSpec(
        m_car_eff_kg=0.55,
        m_rot_kg=0.18,
        k_motor_raw_n_m=260_000.0,
        k_bracket_n_m=1_200_000.0,
        branches=branches_const,
        include_transverse_coupling=False,
    )
    drive_nl = NonlinearBeltDriveSpec(
        m_car_eff_kg=0.55,
        m_rot_kg=0.18,
        k_motor_raw_n_m=260_000.0,
        k_bracket_n_m=1_200_000.0,
        branches=branches_nl,
        include_transverse_coupling=False,
    )

    hb_const = solve_nonlinear_belt_shaker_response(drive_const)
    hb_nl = solve_nonlinear_belt_shaker_response(drive_nl)
    # Identical linear-limit tangent stiffness at T0
    assert math.isclose(hb_const.linear_freq_hz, hb_nl.linear_freq_hz, rel_tol=1e-9)
    assert math.isclose(hb_const.k_belt_linear_n_m, hb_nl.k_belt_linear_n_m, rel_tol=1e-9)
    # Nonlinear crimp toe produces finite softening & matches time-domain ODE closely
    td_nl = simulate_swept_sine_time_domain(
        drive_nl, f_start_hz=20.0, f_end_hz=150.0, sweep_rate_hz_per_s=15.0, sample_rate_hz=2400.0
    )
    assert abs(hb_nl.effective_freq_hz - td_nl.peak_freq_hz) / td_nl.peak_freq_hz < 0.08


def test_transverse_1to1_resonance_near_f_eff_couples_into_longitudinal() -> None:
    """Transverse string mode near f_eff excites 1:1 geometric coupling in both HB and ODE."""
    # Set drive so f_eff ~ 65 Hz, and tune longest_free_span_m so f_w ~ 65 Hz (1:1 internal resonance)
    t0 = 18.0
    mu = 0.0085
    ei = 9.0e-6
    base_kwargs = dict(
        m_car_eff_kg=0.55,
        m_rot_kg=0.15,
        k_motor_raw_n_m=280_000.0,
        k_bracket_n_m=1_500_000.0,
        pretension_n=t0,
        k_belt_linear_n_m=120_000.0,
        n_loops=2.0,
        belt_linear_mass_kg_m=mu,
        belt_ei_nm2=ei,
        transverse_damping_ratio=0.025,
        pulley_eccentricity_m=2.0e-4,  # 0.2 mm realistic tooth/pulley runout + gravity sag
        zeta_belt_base=0.040,
        zeta_motor=0.045,
        zeta_car=0.010,
    )
    lin_ref = solve_linear_coupled_drive_2dof(120_000.0, 280_000.0, 1_500_000.0, 0.55, 0.15)
    f_eff = lin_ref["f1_hz"]
    # Solve L_f = (1 / (2*f_eff)) * sqrt(T0 / mu) for 1:1 tuning
    l_tuned = math.sqrt(t0 / mu) / (2.0 * f_eff)
    f_w_tuned = transverse_span_natural_freq_hz(t0, l_tuned, mu, ei)
    assert abs(f_w_tuned - f_eff) / f_eff < 0.05

    drive_tuned = NonlinearBeltDriveSpec(longest_free_span_m=l_tuned, include_transverse_coupling=True, **base_kwargs)
    drive_detuned = NonlinearBeltDriveSpec(longest_free_span_m=0.12, include_transverse_coupling=True, **base_kwargs)
    drive_uncoupled = NonlinearBeltDriveSpec(longest_free_span_m=l_tuned, include_transverse_coupling=False, **base_kwargs)

    hb_tuned = solve_nonlinear_belt_shaker_response(drive_tuned)
    hb_detuned = solve_nonlinear_belt_shaker_response(drive_detuned)

    assert hb_tuned.whip_coupling_regime in ("direct_1to1", "combined")
    assert hb_tuned.whip_participation > 0.20
    assert hb_tuned.whip_participation > 4.0 * hb_detuned.whip_participation
    assert hb_tuned.transverse_amp_mm > 0.25
    assert hb_tuned.transverse_amp_mm > 5.0 * hb_detuned.transverse_amp_mm

    # Verify in full time-domain ODE as well
    td_tuned = simulate_swept_sine_time_domain(
        drive_tuned, f_start_hz=25.0, f_end_hz=110.0, sweep_rate_hz_per_s=10.0, sample_rate_hz=2000.0
    )
    td_uncoupled = simulate_swept_sine_time_domain(
        drive_uncoupled, f_start_hz=25.0, f_end_hz=110.0, sweep_rate_hz_per_s=10.0, sample_rate_hz=2000.0
    )
    assert td_tuned.max_transverse_amp_mm > 0.25
    # Transverse mode visibly perturbs the longitudinal acceleration PSD
    psd_l2_diff = float(np.sqrt(np.mean((td_tuned.psd_norm - td_uncoupled.psd_norm) ** 2)))
    assert psd_l2_diff > 1.0, f"Expected noticeable 1:1 longitudinal PSD perturbation, got L2={psd_l2_diff:.3f}"


def test_transverse_2to1_parametric_mathieu_resonance_near_half_f_eff() -> None:
    """Transverse string mode near f_eff / 2 excites 2:1 parametric Mathieu-Duffing coupling."""
    # Pure parametric test: q0 = 0 (no 1:1 sag/eccentricity!), so ONLY Mathieu 2:1 instability can excite q(t)!
    t0 = 14.0
    mu = 0.0085
    ei = 9.0e-6
    lin_ref = solve_linear_coupled_drive_2dof(160_000.0, 360_000.0, 2_000_000.0, 0.50, 0.12)
    f_eff = lin_ref["f1_hz"]
    f_w_target = 0.5 * f_eff
    l_param = math.sqrt(t0 / mu) / (2.0 * f_w_target)
    f_w_actual = transverse_span_natural_freq_hz(t0, l_param, mu, ei)
    assert abs(f_w_actual - f_w_target) / f_w_target < 0.05

    common = dict(
        m_car_eff_kg=0.50,
        m_rot_kg=0.12,
        k_motor_raw_n_m=360_000.0,
        k_bracket_n_m=2_000_000.0,
        pretension_n=t0,
        k_belt_linear_n_m=160_000.0,
        n_loops=2.0,
        longest_free_span_m=l_param,
        belt_linear_mass_kg_m=mu,
        belt_ei_nm2=ei,
        transverse_damping_ratio=0.020,
        transverse_initial_offset_m=0.0,  # purely parametric (zero 1:1 linear coupling!)
        zeta_belt_base=0.035,
        zeta_motor=0.040,
        zeta_car=0.008,
        accel_per_hz_mm_s2=75.0,
    )
    drive_param = NonlinearBeltDriveSpec(include_transverse_coupling=True, **common)
    drive_uncoupled = NonlinearBeltDriveSpec(include_transverse_coupling=False, **common)

    hb_param = solve_nonlinear_belt_shaker_response(drive_param)
    hb_uncoupled = solve_nonlinear_belt_shaker_response(drive_uncoupled)

    assert hb_param.whip_coupling_regime == "parametric_2to1"
    assert hb_param.transverse_amp_mm > 0.5
    assert hb_param.whip_participation > 0.05
    # Parametric energy sink into transverse string broadens / damps longitudinal resonance
    assert hb_param.zeta_nl > hb_uncoupled.zeta_nl

    # Full time-domain ODE: tiny 10 um seed grows exponentially via Mathieu instability near f_eff = 2*f_w!
    seed_m = 1.0e-5
    td_param = simulate_swept_sine_time_domain(
        drive_param,
        f_start_hz=max(20.0, f_eff - 30.0),
        f_end_hz=f_eff + 25.0,
        sweep_rate_hz_per_s=4.0,
        sample_rate_hz=2400.0,
        transverse_seed_m=seed_m,
    )
    td_uncoupled = simulate_swept_sine_time_domain(
        drive_uncoupled,
        f_start_hz=max(20.0, f_eff - 30.0),
        f_end_hz=f_eff + 25.0,
        sweep_rate_hz_per_s=4.0,
        sample_rate_hz=2400.0,
        transverse_seed_m=seed_m,
    )
    # Must amplify far above the 0.01 mm initial seed (Mathieu exponential growth + Duffing saturation)
    assert td_param.max_transverse_amp_mm > 20.0 * (seed_m * 1e3), (
        f"Expected Mathieu amplification > 0.2 mm, got {td_param.max_transverse_amp_mm:.4f} mm"
    )
    # Geometric shortening feeds back into the longitudinal carriage PSD
    psd_l2_diff = float(np.sqrt(np.mean((td_param.psd_norm - td_uncoupled.psd_norm) ** 2)))
    assert psd_l2_diff > 0.5, f"Expected 2:1 longitudinal PSD feedback, got L2={psd_l2_diff:.3f}"


def test_fast_solver_runtime_budget_and_pyodide_lean_imports() -> None:
    """Fast HB solver stays well under 25 ms/axis and never eagerly imports heavy SciPy modules."""

    def crimp_ea(t: float) -> float:
        return 42_000.0 * (1.0 - 0.52 * math.exp(-max(t, 0.1) / 13.8))

    drive = NonlinearBeltDriveSpec(
        m_car_eff_kg=0.55,
        m_rot_kg=0.22,
        k_motor_raw_n_m=240_000.0,
        k_bracket_n_m=1_100_000.0,
        pretension_n=10.0,
        k_belt_linear_n_m=95_000.0,
        n_loops=2.0,
        slack_load_share=0.58,
        ea_fn=crimp_ea,
        longest_free_span_m=0.42,
        pulley_eccentricity_m=1.0e-4,
    )

    # Warm-up call
    _ = solve_nonlinear_belt_shaker_response(drive)

    times_ms: list[float] = []
    for _ in range(5):
        t0 = time.perf_counter()
        res = solve_nonlinear_belt_shaker_response(drive)
        times_ms.append((time.perf_counter() - t0) * 1e3)
        assert res.runtime_ms > 0.0

    median_ms = float(np.median(times_ms))
    assert median_ms < 25.0, f"Fast HB solver median runtime {median_ms:.2f} ms exceeded 25 ms budget"

    # Pyodide WASM boot hygiene: importing & running fast solver must NOT load scipy.integrate/signal/etc.
    probe = (
        "import sys\n"
        "from printsim.fea.belt_nonlinear import NonlinearBeltDriveSpec, solve_nonlinear_belt_shaker_response\n"
        "res = solve_nonlinear_belt_shaker_response(NonlinearBeltDriveSpec(m_car_eff_kg=0.5, pretension_n=12.0))\n"
        "assert res.effective_freq_hz > 10.0\n"
        "heavy = ('scipy.integrate', 'scipy.signal', 'scipy.optimize', 'scipy.stats', 'scipy.spatial')\n"
        "loaded = [m for m in heavy if m in sys.modules]\n"
        "assert not loaded, f'Heavy SciPy modules loaded by fast HB solver: {loaded}'\n"
    )
    proc = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"Pyodide lean-import probe failed:\n{proc.stderr}"
