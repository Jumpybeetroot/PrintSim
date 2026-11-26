"""Regression & analytical verification tests for Group B (B-03..B-07, D-06) and Group E cleanups."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from printsim.fea.modal import solve_modal_fea
from printsim.fea.mechatronics import DriveControlConfig, build_coupled_state_space
from printsim.geometry import build_assembly
from printsim.kinematics.collision import check_frame_clearance
from printsim.kinematics.solvers import (
    _delta_center_j_norm,
    _delta_jacobian,
    analyze_kinematics,
    compute_drive_stiffness_details_n_per_m,
    inverse_kinematics,
    stepper_available_torque_nm,
)
from printsim.schema import PrinterSpec, StepperSpec
from printsim.thermal.filaments import FILAMENT_CATALOG, get_filament
from printsim.thermal.hotend import simulate_hotend_thermal

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


# ---------------------------------------------------------------------------
# B-03: Full Delta Jacobian motor torque, speed, and drive stiffness model
# ---------------------------------------------------------------------------


def test_b03_delta_analytic_jacobian_matches_finite_difference_ik() -> None:
    """Analytic Delta Jacobian matches central finite-difference inverse_kinematics to < 1e-9."""
    for yaml_name in ("delta_racer.yaml", "flsun_v400_delta.yaml"):
        spec = PrinterSpec.from_yaml(EXAMPLES_DIR / yaml_name)
        bx, by, bz = spec.kinematics.build_volume_mm
        r_bed_mm = 0.5 * min(bx, by)
        eps_mm = 1e-4

        pts_mm = [(0.5 * bx, 0.5 * by, 0.5 * bz)]
        for rho in (0.5 * r_bed_mm, r_bed_mm):
            for ang_deg in (0.0, 45.0, 90.0, 150.0, 210.0, 285.0):
                ang = math.radians(ang_deg)
                pts_mm.append((0.5 * bx + rho * math.cos(ang), 0.5 * by + rho * math.sin(ang), 0.5 * bz))

        for px, py, pz in pts_mm:
            cx_m = (px - 0.5 * bx) * 1e-3
            cy_m = (py - 0.5 * by) * 1e-3
            j_analytic = _delta_jacobian(spec, cx_m, cy_m)

            j_fd = np.zeros((3, 3), dtype=float)
            for col, (dx, dy, dz) in enumerate(((eps_mm, 0.0, 0.0), (0.0, eps_mm, 0.0), (0.0, 0.0, eps_mm))):
                q_p = np.asarray(inverse_kinematics(spec, (px + dx, py + dy, pz + dz)), dtype=float)
                q_m = np.asarray(inverse_kinematics(spec, (px - dx, py - dy, pz - dz)), dtype=float)
                j_fd[:, col] = (q_p - q_m) / (2.0 * eps_mm)

            assert np.allclose(j_analytic, j_fd, atol=1e-9, rtol=1e-9)


def test_b03_delta_perimeter_torque_rpm_gravity_and_stiffness_j_norm() -> None:
    """Delta uses perimeter Jacobian speed ratio, m_c = carriage_mass/3 + gravity, and j_norm stiffness."""
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "delta_racer.yaml")
    asm = build_assembly(spec)
    kin = analyze_kinematics(asm)

    # 1. Peak motor RPM reflects perimeter Jacobian speed ratio (~2.337 on delta_racer -> ~2103 rpm)
    r_p = spec.gantry.belt.pitch_radius_m
    v_max_m_s = spec.kinematics.max_velocity_mm_s * 1e-3
    bx, by, _ = spec.kinematics.build_volume_mm
    r_bed_m = 0.5 * min(bx, by) * 1e-3
    r_del_m = spec.kinematics.delta_radius_mm * 1e-3
    l_arm_m = spec.kinematics.delta_arm_length_mm * 1e-3
    d_max = r_del_m + r_bed_m
    h_min = math.sqrt(l_arm_m**2 - d_max**2)
    expected_ratio = d_max / h_min
    expected_rpm = (v_max_m_s * expected_ratio / (2.0 * math.pi * r_p)) * 60.0
    assert expected_ratio > 2.2
    assert kin.max_motor_rpm == pytest.approx(expected_rpm, rel=1e-4)

    # 2. Required torque strictly exceeds old optimistic 0.65 * m * a approximation
    a_m_s2 = spec.kinematics.max_accel_mm_s2 * 1e-3
    t_old_optimistic = (0.65 * (asm.x_moving_mass_kg * a_m_s2 + 2.5)) * r_p + spec.kinematics.xy_stepper.rotor_inertia_kg_m2 * (a_m_s2 / r_p)
    assert max(kin.required_torque_x_nm, kin.required_torque_y_nm) > t_old_optimistic * 1.15
    assert kin.torque_safety_margin < 3.0  # replaces non-credible 6.0x margin

    # 3. Gravity hold torque is nonzero at a = 0 and carriage_mass_g increases torque demand
    s_zero_a = spec.model_copy(deep=True)
    s_zero_a.kinematics.max_accel_mm_s2 = 1.0  # near-zero horizontal accel
    kin_zero_a = analyze_kinematics(build_assembly(s_zero_a))
    # Static gravity hold at center: (m_e / 3 + m_c) * g * r_p + f_fric_car * r_p
    m_c_single = (spec.gantry.carriage_mass_g / 3.0) * 1e-3
    t_hold_center = ((asm.x_moving_mass_kg / 3.0 + m_c_single) * 9.80665 + 2.5 / 3.0) * r_p
    assert kin_zero_a.required_torque_y_nm >= t_hold_center * 0.99

    s_heavy_car = spec.model_copy(deep=True)
    s_heavy_car.gantry.carriage_mass_g = spec.gantry.carriage_mass_g + 300.0
    kin_heavy_car = analyze_kinematics(build_assembly(s_heavy_car))
    assert kin_heavy_car.required_torque_y_nm > kin.required_torque_y_nm
    assert kin_heavy_car.motor_max_accel_mm_s2 < kin.motor_max_accel_mm_s2

    # 4. Drive stiffness uses true center Jacobian projection j_norm = 1.5 * (R/h0)^2 instead of ad-hoc 2.5
    details = compute_drive_stiffness_details_n_per_m(asm, "X")
    j_norm = _delta_center_j_norm(spec)
    h0 = math.sqrt(l_arm_m**2 - r_del_m**2)
    assert j_norm == pytest.approx(1.5 * (r_del_m / h0) ** 2, rel=1e-9)

    center_rpm = (v_max_m_s * (r_del_m / h0) / (2.0 * math.pi * r_p)) * 60.0
    rpm_op = max(120.0, 0.35 * center_rpm)
    k_mot_expected = j_norm * spec.kinematics.xy_stepper.torsional_stiffness_nm_per_rad(rpm=rpm_op) / (r_p**2)
    assert details["k_motor_axis_n_m"] == pytest.approx(k_mot_expected, rel=1e-4)

    # Larger delta radius raises (R/h0)^2 -> higher XY Jacobian stiffness
    s_wide = spec.model_copy(deep=True)
    s_wide.kinematics.delta_radius_mm = spec.kinematics.delta_radius_mm + 20.0
    s_wide.kinematics.delta_arm_length_mm = spec.kinematics.delta_arm_length_mm + 20.0
    det_wide = compute_drive_stiffness_details_n_per_m(build_assembly(s_wide), "X")
    assert _delta_center_j_norm(s_wide) > j_norm
    assert det_wide["k_belt_subsystem_n_m"] > details["k_belt_subsystem_n_m"]


# ---------------------------------------------------------------------------
# B-04 & B-05: Two-phase stepper Kt_phase / I_pk & torque-balance max velocity
# ---------------------------------------------------------------------------


def test_b04_stepper_peak_rms_constants_and_piecewise_pullout_curve() -> None:
    """Kt_phase = T_hold/(sqrt(2)*I_rated), piecewise constant-then-voltage-limited pullout, & b_bemf_rot."""
    st = StepperSpec(
        holding_torque_nm=0.55,
        rated_current_a=2.5,
        run_current_a=1.2,
        phase_resistance_ohm=1.2,
        inductance_mh=1.5,
        supply_voltage_v=24.0,
        steps_per_rev=200,
    )
    k_t, i_pk = st._phase_constants()
    assert k_t == pytest.approx(0.55 / (math.sqrt(2.0) * 2.5), rel=1e-12)
    assert i_pk == pytest.approx(math.sqrt(2.0) * 1.2, rel=1e-12)
    # Peak holding torque consistency: Kt_phase * sqrt(2) * I_rated == holding_torque_nm
    assert k_t * math.sqrt(2.0) * st.rated_current_a == pytest.approx(st.holding_torque_nm, rel=1e-12)

    # Piecewise pull-out curve: strictly 1.0 up to corner speed, strictly decreasing above, 0 at E >= V_eff
    v_eff = 0.90 * st.supply_voltage_v
    assert st.pullout_torque_efficiency(10.0, bus_utilization=0.90) == pytest.approx(1.0, rel=1e-12)
    assert st.pullout_torque_efficiency(400.0, bus_utilization=0.90) == pytest.approx(1.0, rel=1e-12)
    assert st.pullout_torque_efficiency(800.0, bus_utilization=0.90) == pytest.approx(1.0, rel=1e-12)

    # Above corner speed (~890 rpm for 24V, 0.90*V_bus): rolls off along voltage ellipse
    eta_1000 = st.pullout_torque_efficiency(1000.0, bus_utilization=0.90)
    eta_1200 = st.pullout_torque_efficiency(1200.0, bus_utilization=0.90)
    assert 0.0 < eta_1200 < eta_1000 < 1.0

    # At no-load back-EMF speed (k_t * omega_m == v_eff), pull-out torque reaches zero
    rpm_nl = (v_eff / k_t) * 60.0 / (2.0 * math.pi)
    assert st.pullout_torque_efficiency(rpm_nl, bus_utilization=0.90) == pytest.approx(0.0, abs=1e-9)
    assert st.pullout_torque_efficiency(rpm_nl + 100.0, bus_utilization=0.90) == pytest.approx(0.0, abs=1e-9)

    # No unphysical 1.12x 3-phase SVPWM boost on dual-H-bridge foc_servo:
    # at high speed above FOC corner, foc_servo tracks 0.98 * V_bus (not 1.12 * V_bus)
    t_foc_1100 = stepper_available_torque_nm(st, 1100.0, drive_type="foc_servo")
    expected_foc_1100 = st.effective_holding_torque_nm * 0.95 * st.pullout_torque_efficiency(1100.0, bus_utilization=0.98)
    assert t_foc_1100 == pytest.approx(expected_foc_1100, rel=1e-12)
    t_if_boosted = st.effective_holding_torque_nm * 0.95 * st.pullout_torque_efficiency(1100.0, bus_utilization=0.98 * 1.12)
    assert t_foc_1100 < t_if_boosted

    # Propagate Kt_phase to mechatronics midband check: 24V high-L still flags risk, 48V clears it
    spec_mb = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    spec_mb.kinematics.max_velocity_mm_s = 560.0
    spec_mb.kinematics.xy_stepper = StepperSpec(
        holding_torque_nm=0.59,
        rated_current_a=2.0,
        run_current_a=1.6,
        phase_resistance_ohm=1.4,
        inductance_mh=3.0,
        supply_voltage_v=24.0,
    )
    asm_24 = build_assembly(spec_mb)
    mod_24 = solve_modal_fea(asm_24)
    mdl_24 = build_coupled_state_space(asm_24, mod_24, 0.5, 0.5, DriveControlConfig(drive_type="open_loop_stepper"))
    assert mdl_24.midband_instability_risk is True


def test_b05_motor_max_velocity_solves_torque_balance_beyond_corner_speed() -> None:
    """motor_max_velocity_mm_s solves T_avail(omega)/sf >= T_req above corner speed with CoreXY sqrt(2)."""
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    asm = build_assembly(spec)
    kin = analyze_kinematics(asm)

    st = spec.kinematics.xy_stepper
    r_p = spec.gantry.belt.pitch_radius_m
    k_t, i_pk = st._phase_constants()

    # Compute corner speed in Cartesian mm/s using CoreXY diagonal sqrt(2) factor:
    # hypot(R*I_pk + k_t*w, p*L*I_pk*w) = 0.90 * V_bus
    v_eff = 0.90 * st.supply_voltage_v
    l_h = st.inductance_mh * 1e-3
    lo_w, hi_w = 0.0, 500.0
    for _ in range(60):
        mid_w = 0.5 * (lo_w + hi_w)
        v_req = math.hypot(st.phase_resistance_ohm * i_pk + k_t * mid_w, st.pole_pairs * l_h * i_pk * mid_w)
        if v_req <= v_eff:
            lo_w = mid_w
        else:
            hi_w = mid_w
    corner_vel_corexy_mm_s = (lo_w * r_p / math.sqrt(2.0)) * 1000.0

    # B-05: real torque-balance max velocity strictly exceeds the corner speed (no hard cutoff at corner)
    assert kin.motor_max_velocity_mm_s > corner_vel_corexy_mm_s + 100.0

    # At the reported motor_max_velocity_mm_s, CoreXY diagonal RPM gives T_avail / sf == T_req_reserve
    v_sol_m_s = kin.motor_max_velocity_mm_s * 1e-3
    rpm_sol = (v_sol_m_s * math.sqrt(2.0) / (2.0 * math.pi * r_p)) * 60.0
    t_avail_sol = stepper_available_torque_nm(st, rpm_sol, drive_type=spec.kinematics.drive_type)
    m_x_mot = asm.x_moving_mass_kg + 2.0 * st.rotor_inertia_kg_m2 / (r_p**2)
    m_y_mot = asm.y_moving_mass_kg + 2.0 * st.rotor_inertia_kg_m2 / (r_p**2)
    t_res = 0.5 * math.hypot(m_x_mot * 0.5 + 2.5, m_y_mot * 0.5 + 2.5) * r_p
    assert (t_avail_sol / 1.35) == pytest.approx(t_res, rel=1e-3, abs=2e-4)

    # Cartesian (vel_scale=1.0) achieves ~sqrt(2)x higher linear velocity than 2WD CoreXY for the same motor
    spec_cart = spec.model_copy(deep=True)
    spec_cart.architecture = "cartesian_bedslinger"
    kin_cart = analyze_kinematics(build_assembly(spec_cart))
    assert kin_cart.motor_max_velocity_mm_s > kin.motor_max_velocity_mm_s * 1.30


# ---------------------------------------------------------------------------
# B-06 & Group E: CoreXY pure-X, pure-Y, exact worst-direction accel & belt preload
# ---------------------------------------------------------------------------


def test_b06_corexy_per_axis_and_worst_direction_accel_2wd_and_awd() -> None:
    """CoreXY reports pure-X, pure-Y, and exact worst-dir accel for 2WD & AWD; consistent belt accel."""
    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    for mode, k_mot, f_fric in (("2wd", 2.0, 2.5), ("awd", 4.0, 3.6)):
        spec = base.model_copy(deep=True)
        spec.gantry.drive_mode = mode  # type: ignore[assignment]
        asm = build_assembly(spec)
        kin = analyze_kinematics(asm)

        r_p = spec.gantry.belt.pitch_radius_m
        j_rot = spec.kinematics.xy_stepper.rotor_inertia_kg_m2
        m_x = asm.x_moving_mass_kg
        m_y = asm.y_moving_mass_kg
        mx_mot = m_x + k_mot * j_rot / (r_p**2)
        my_mot = m_y + k_mot * j_rot / (r_p**2)

        t_avail = kin.available_torque_at_vmax_nm
        f_budget = (t_avail / 1.35) / r_p * k_mot
        expected_ax = (f_budget - f_fric) / mx_mot * 1000.0
        expected_ay = (f_budget - f_fric) / my_mot * 1000.0
        assert kin.motor_max_accel_x_mm_s2 == pytest.approx(expected_ax, rel=1e-6)
        assert kin.motor_max_accel_y_mm_s2 == pytest.approx(expected_ay, rel=1e-6)

        # Exact worst direction over 360 deg numerical sweep matches motor_max_accel_mm_s2
        angles = np.linspace(0.0, 2.0 * math.pi, 720, endpoint=False)
        a_dir_proj = min(
            (f_budget - f_fric * (abs(math.cos(phi)) + abs(math.sin(phi))))
            / (mx_mot * abs(math.cos(phi)) + my_mot * abs(math.sin(phi)))
            for phi in angles
        ) * 1000.0
        assert kin.motor_max_accel_mm_s2 == pytest.approx(a_dir_proj, rel=1e-4)
        assert kin.motor_max_accel_mm_s2 < min(kin.motor_max_accel_x_mm_s2, kin.motor_max_accel_y_mm_s2)

        # Closed-form check at reported motor_max_accel_mm_s2:
        # F_mot_max = hypot(M_x*a + f_fric, M_y*a + f_fric) / k_mot == (t_avail / 1.35) / r_p
        a_rep_m_s2 = kin.motor_max_accel_mm_s2 * 1e-3
        assert math.hypot(mx_mot * a_rep_m_s2 + f_fric, my_mot * a_rep_m_s2 + f_fric) == pytest.approx(
            f_budget, rel=1e-6
        )

        # Evaluating kinematics at motor_max_accel_mm_s2 hits torque_safety_margin == 1.35 (no step skip)
        s_at_mot = spec.model_copy(deep=True)
        s_at_mot.kinematics.max_accel_mm_s2 = kin.motor_max_accel_mm_s2 * (1.0 - 1e-6)
        k_at_mot = analyze_kinematics(build_assembly(s_at_mot))
        assert k_at_mot.torque_safety_margin == pytest.approx(1.35, rel=1e-3)
        assert k_at_mot.step_skip_risk is False

        # Evaluating kinematics at belt_max_accel_mm_s2 avoids belt slip right at threshold
        s_at_belt = spec.model_copy(deep=True)
        s_at_belt.kinematics.max_accel_mm_s2 = kin.belt_max_accel_mm_s2 * (1.0 - 1e-6)
        k_at_belt = analyze_kinematics(build_assembly(s_at_belt))
        assert k_at_belt.belt_slip_risk is False

        # Group E: frame_belt_preload_n counts 2 spans per belt loop (4*T0 for 2WD, 8*T0 for AWD)
        t0 = spec.gantry.belt.pretension_n
        expected_preload = (8.0 if mode == "awd" else 4.0) * t0
        assert kin.frame_belt_preload_n == pytest.approx(expected_preload, rel=1e-9)

    # Group E: collision check uses consistent 85 + 0.5*ez Z overhead for CoreXY
    asm_c = build_assembly(base)
    col_c = check_frame_clearance(asm_c)
    expected_cz = base.frame.height_mm - (base.kinematics.build_volume_mm[2] + 85.0 + 0.5 * base.toolhead.envelope_mm[2])
    assert col_c.clearance_z_mm == pytest.approx(expected_cz, rel=1e-6)


# ---------------------------------------------------------------------------
# B-07, D-06 & Group E Thermal: DSC cp, zero amorphous latent heat, & flow ranking
# ---------------------------------------------------------------------------


def test_b07_d06_melt_dsc_cp_zero_amorphous_latent_heat_and_flow_ranking() -> None:
    """Unified melt-averaged cp, zero amorphous latent heat, & realistic cross-filament flow ranking."""
    # 1. All catalog filaments use consistent melt-range averaged DSC cp in [1850, 2200] J/(kg*K)
    for fid, fil in FILAMENT_CATALOG.items():
        assert 1850.0 <= fil.specific_heat_j_kgk <= 2200.0, f"{fid} cp={fil.specific_heat_j_kgk} out of band"

    # 2. D-06: strictly amorphous polymers have latent_heat_j_kg == 0.0; semi-crystalline > 30 kJ/kg
    for amorph_id in ("petg", "abs", "asa", "tpu_95a"):
        assert get_filament(amorph_id).latent_heat_j_kg == pytest.approx(0.0, abs=1e-9)
    for semi_cryst_id in ("pla", "pa12_cf"):
        assert get_filament(semi_cryst_id).latent_heat_j_kg >= 30000.0

    # 3. Cross-filament volumetric flow ranking on default 16 mm hotend
    spec_def = PrinterSpec()
    q_pla = simulate_hotend_thermal(spec_def, filament_id="pla").max_volumetric_flow_mm3_s
    q_abs = simulate_hotend_thermal(spec_def, filament_id="abs").max_volumetric_flow_mm3_s
    q_asa = simulate_hotend_thermal(spec_def, filament_id="asa").max_volumetric_flow_mm3_s
    q_petg = simulate_hotend_thermal(spec_def, filament_id="petg").max_volumetric_flow_mm3_s
    q_pa12 = simulate_hotend_thermal(spec_def, filament_id="pa12_cf").max_volumetric_flow_mm3_s
    q_tpu = simulate_hotend_thermal(spec_def, filament_id="tpu_95a").max_volumetric_flow_mm3_s

    assert 21.0 <= q_pla <= 24.0
    assert 21.0 <= q_abs <= 24.0
    assert abs(q_abs - q_pla) <= 2.0
    assert 18.0 <= q_petg <= 22.0
    assert q_petg <= q_pla
    assert q_pa12 < q_pla
    assert q_tpu < q_pa12
    assert 21.0 <= q_asa <= 24.0

    # Touchstone: corexy_voron_style.yaml PLA flow stays in ~22-24 mm^3/s
    spec_v24 = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    th_v24 = simulate_hotend_thermal(spec_v24, filament_id="pla")
    assert 22.0 <= th_v24.max_volumetric_flow_mm3_s <= 24.5


def test_group_e_hotend_loss_bed_target_and_02x04_linear_speed() -> None:
    """Reconciled 0.05 W/K block+nozzle loss, active bed_target in bed heatup, & 0.20*0.40 speed."""
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    th_pla = simulate_hotend_thermal(spec, filament_id="pla")
    th_petg = simulate_hotend_thermal(spec, filament_id="petg")
    th_abs = simulate_hotend_thermal(spec, filament_id="abs")

    # 1. Exact 0.20 * 0.40 mm^2 bead cross-section for max_linear_speed_02x04_mm_s
    assert th_pla.max_linear_speed_02x04_mm_s == pytest.approx(
        th_pla.max_volumetric_flow_mm3_s / (0.20 * 0.40), rel=1e-9
    )

    # 2. Steady heat loss matches R_hb+R_hs conduction + exact 0.050 W/K FVM block+nozzle loss
    he = spec.toolhead.hotend
    r_in = (he.heatbreak_id_mm * 1e-3) * 0.5
    r_out = r_in + he.heatbreak_wall_mm * 1e-3
    area_hb = math.pi * (r_out**2 - r_in**2)
    r_hb = th_pla.heatbreak_thermal_resistance_k_w
    assert area_hb > 0.0
    dt_pla = th_pla.target_temp_c - max(he.ambient_temp_c, spec.frame.chamber_temp_c)
    expected_loss = dt_pla / (r_hb + he.heatsink_thermal_resistance_k_w) + 0.050 * dt_pla
    assert th_pla.steady_heat_loss_w == pytest.approx(expected_loss, rel=1e-9)

    # 3. Bed heatup time actually uses filament bed_target (PLA 60C < PETG 80C < ABS 105C)
    assert th_pla.bed_heatup_time_100c_s < th_petg.bed_heatup_time_100c_s < th_abs.bed_heatup_time_100c_s
