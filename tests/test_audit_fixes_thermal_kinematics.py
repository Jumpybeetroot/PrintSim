"""Regression tests for thermal, rheology, profile, kinematics, and G-code planner audit fixes."""

from __future__ import annotations

import math
import subprocess
import sys

import numpy as np
import pytest

from printsim.fea import solve_modal_fea
from printsim.gcode.simulator import _klipper_junction_speed_mm_s, simulate_gcode
from printsim.geometry import build_assembly
from printsim.kinematics import analyze_kinematics
from printsim.schema import PROFILE_CATALOG, BeltSpec, PrinterSpec, StepperSpec, get_material
from printsim.thermal.filaments import FILAMENT_CATALOG, FilamentSpec, get_filament
from printsim.thermal.hotend import (
    analyze_hotend_thermal,
    nozzle_total_pressure_drop_pa,
    rabinowitsch_pipe_pressure_drop_pa,
    simulate_hotend_thermal,
    solve_1d_fvm_conduction,
    solve_pressure_limited_flow_m3_s,
)


# ---------------------------------------------------------------------------
# T1: Open T-slot extrusion torsional constants J / (Iy + Iz) < 0.15
# ---------------------------------------------------------------------------


def test_t1_open_tslot_extrusion_torsional_constants() -> None:
    """Verify all open T-slot extrusions use realistic datasheet I_t with J / (Iy + Iz) < 0.15."""
    extrusions = [p for p in PROFILE_CATALOG.values() if p.shape == "extrusion"]
    assert len(extrusions) >= 5

    expected_j = {
        "extrusion_2020": 0.11e-8,
        "extrusion_2040": 0.36e-8,
        "extrusion_3030": 0.45e-8,
        "extrusion_4040": 1.30e-8,
        "extrusion_3060": 2.10e-8,
    }
    for prof in extrusions:
        ratio = prof.j_m4 / (prof.iy_m4 + prof.iz_m4)
        assert 0.05 <= ratio < 0.15, f"{prof.id} has unrealistic J/(Iy+Iz) = {ratio:.3f}"
        if prof.id in expected_j:
            assert prof.j_m4 == pytest.approx(expected_j[prof.id], rel=1e-6)

    # Closed 20x20 tube must retain substantially higher torsional constant than open 2020 T-slot
    assert PROFILE_CATALOG["al_tube_20x20x2"].j_m4 > PROFILE_CATALOG["extrusion_2020"].j_m4 * 8.0


# ---------------------------------------------------------------------------
# T8 & T9: GT2 belt linear mass density & stepper linear-below-rated torque
# ---------------------------------------------------------------------------


def test_t9_gt2_belt_linear_mass_and_voron_pluck_spec() -> None:
    """Gates GT2-6mm fiberglass (8.3 g/m) maps 110 Hz @ 150 mm span to ~9 N."""
    belt_6 = BeltSpec(type="gt2_6mm", core="fiberglass", pretension_n=9.04)
    belt_9 = BeltSpec(type="gt2_9mm", core="fiberglass", pretension_n=9.04)
    belt_12 = BeltSpec(type="gt2_12mm", core="fiberglass", pretension_n=9.04)

    assert belt_6.linear_mass_kg_m == pytest.approx(0.0083, rel=1e-6)
    assert belt_9.linear_mass_kg_m == pytest.approx(0.0083 * 1.5, rel=1e-6)
    assert belt_12.linear_mass_kg_m == pytest.approx(0.0083 * 2.0, rel=1e-6)

    # Inverse check: T = mu * (2 * L * f)^2 for 110 Hz on 150 mm free span
    tension_at_110hz = belt_6.linear_mass_kg_m * ((2.0 * 0.150 * 110.0) ** 2)
    assert tension_at_110hz == pytest.approx(8.9, abs=0.5)
    assert belt_6.pluck_freq_hz(150.0) == pytest.approx(110.0, abs=0.5)


def test_t8_stepper_torque_linear_slope_below_rated_and_saturation_above() -> None:
    """Stepper holding torque has slope 1 below I_rated, exact T_h at I_rated, and soft sat above."""
    t_hold = 0.55
    i_rated = 2.5

    # Exact linear slope = 1.0 from near-zero up to rated current
    st_low = StepperSpec.model_construct(
        holding_torque_nm=t_hold,
        rated_current_a=i_rated,
        run_current_a=0.05,
        phase_resistance_ohm=1.2,
        inductance_mh=1.5,
        supply_voltage_v=24.0,
        rotor_inertia_kg_m2=6.8e-6,
        motor_mass_g=340.0,
        steps_per_rev=200,
    )
    assert st_low.effective_holding_torque_nm == pytest.approx(t_hold * (0.05 / i_rated), rel=1e-9)

    for i_run in (0.5, 1.2, 1.8, 2.5):
        st = StepperSpec(holding_torque_nm=t_hold, rated_current_a=i_rated, run_current_a=i_run)
        assert st.effective_holding_torque_nm == pytest.approx(t_hold * (i_run / i_rated), rel=1e-9)

    # Exact equality at run_current == rated_current
    st_rated = StepperSpec(holding_torque_nm=t_hold, rated_current_a=i_rated, run_current_a=i_rated)
    assert st_rated.effective_holding_torque_nm == pytest.approx(t_hold, rel=1e-9)

    # Above rated current: strictly greater than T_h, strictly increasing, and sub-linear soft saturation
    st_over = StepperSpec(holding_torque_nm=t_hold, rated_current_a=i_rated, run_current_a=3.5)
    assert t_hold < st_over.effective_holding_torque_nm < t_hold * (3.5 / i_rated)


# ---------------------------------------------------------------------------
# T4 & T5: Filament catalog thermal conductivity, melt temps, and Tg / creep
# ---------------------------------------------------------------------------


def test_t4_t5_filament_catalog_tg_conductivity_and_backward_compatibility() -> None:
    """Verify corrected PA12-CF/TPU Tg, realistic thermal conductivities, and FilamentSpec defaults."""
    pa12 = get_filament("pa12_cf")
    tpu = get_filament("tpu_95a")
    pa12_mat = get_material("pa12_cf_printed")

    assert pa12.glass_transition_c == pytest.approx(pa12_mat.tg_c, abs=1e-6)
    assert pa12.glass_transition_c == pytest.approx(56.0, abs=1e-6)
    assert tpu.glass_transition_c == pytest.approx(-35.0, abs=1e-6)
    assert tpu.effective_heat_creep_temp_c == pytest.approx(55.0, abs=1e-6)

    expected_k = {
        "pla": 0.13,
        "petg": 0.20,
        "abs": 0.18,
        "asa": 0.18,
        "pa12_cf": 0.35,
        "tpu_95a": 0.19,
    }
    for fid, k_val in expected_k.items():
        fil = FILAMENT_CATALOG[fid]
        assert fil.thermal_conductivity_w_mk == pytest.approx(k_val, rel=1e-6)
        assert fil.extrudable_temp_c > fil.melt_temp_c
        assert fil.print_temp_c > fil.extrudable_temp_c

    # TPU on standard Ti hotend must NOT falsely trigger creep risk despite Tg = -35 C
    spec = PrinterSpec()
    th_tpu_ok = simulate_hotend_thermal(spec, filament_id="tpu_95a")
    assert th_tpu_ok.glass_transition_c == pytest.approx(-35.0, abs=1e-6)
    assert th_tpu_ok.cold_side_temp_c < 45.0
    assert th_tpu_ok.heat_creep_risk is False

    # Poorly cooled thick stainless hotend pushes cold-side > 49 C and triggers TPU softening warning
    spec_hot_throat = spec.model_copy(deep=True)
    spec_hot_throat.toolhead.hotend.heatbreak_material = "stainless_304"
    spec_hot_throat.toolhead.hotend.heatbreak_wall_mm = 0.65
    spec_hot_throat.toolhead.hotend.heatsink_thermal_resistance_k_w = 5.8
    assert simulate_hotend_thermal(spec_hot_throat, filament_id="tpu_95a").heat_creep_risk is True

    # Backward-compatible construction without new optional fields
    custom = FilamentSpec(
        id="custom_poly",
        name="Custom",
        print_temp_c=220.0,
        bed_temp_c=60.0,
        glass_transition_c=60.0,
        density_kg_m3=1200.0,
        specific_heat_j_kgk=1600.0,
        latent_heat_j_kg=30000.0,
    )
    assert custom.thermal_conductivity_w_mk > 0.0
    assert custom.effective_heat_creep_temp_c == pytest.approx(60.0)


# ---------------------------------------------------------------------------
# T7: Patankar harmonic mean FVM exact 1D series conduction benchmark
# ---------------------------------------------------------------------------


def test_t7_patankar_harmonic_fvm_exact_three_layer_conduction() -> None:
    """Zero-convection 3-material FVM rod reproduces exact R = sum(L_i / (k_i * A_i)) within 0.1%."""
    # Layer 1: Copper (L=10 mm, k=385, A=100 mm^2)
    # Layer 2: Ti-6Al-4V throat (L=4 mm, k=6.7, A=2 mm^2)  -- ~1000x jump in k*A!
    # Layer 3: Al 6061 heatsink (L=15 mm, k=167, A=80 mm^2)
    layers = [
        (0.010, 385.0 * 100e-6, 6),
        (0.004, 6.7 * 2e-6, 8),
        (0.015, 167.0 * 80e-6, 10),
    ]
    dz_list = []
    kA_list = []
    r_exact = 0.0
    for length_m, kA_val, n_cells in layers:
        r_exact += length_m / kA_val
        dz_list.extend([length_m / n_cells] * n_cells)
        kA_list.extend([kA_val] * n_cells)

    dz = np.array(dz_list, dtype=float)
    kA = np.array(kA_list, dtype=float)
    t_left, t_right = 215.0, 25.0
    temps = solve_1d_fvm_conduction(
        cell_lengths_m=dz,
        cell_kA=kA,
        g_amb_w_k=None,
        left_dirichlet_temp_c=t_left,
        right_dirichlet_temp_c=t_right,
    )

    q_left = (2.0 * kA[0] / dz[0]) * (t_left - temps[0])
    q_right = (2.0 * kA[-1] / dz[-1]) * (temps[-1] - t_right)
    r_fvm = (t_left - t_right) / q_left

    assert q_left == pytest.approx(q_right, rel=1e-9)
    assert r_fvm == pytest.approx(r_exact, rel=1e-3)

    # Uniform cell grid across a sharp 2-material interface also reproduces exact series resistance
    n1, n2 = 5, 5
    dz_u = np.full(n1 + n2, 0.002, dtype=float)
    kA_u = np.array([0.05] * n1 + [5e-5] * n2, dtype=float)
    r_exact_u = (n1 * 0.002) / 0.05 + (n2 * 0.002) / 5e-5
    temps_u = solve_1d_fvm_conduction(
        cell_lengths_m=dz_u,
        cell_kA=kA_u,
        left_dirichlet_temp_c=200.0,
        right_dirichlet_temp_c=20.0,
    )
    q_u = (2.0 * kA_u[0] / dz_u[0]) * (200.0 - temps_u[0])
    assert (180.0 / q_u) == pytest.approx(r_exact_u, rel=1e-9)


# ---------------------------------------------------------------------------
# T2 & T3: Graetz melt-zone temperature monotonicity & Rabinowitsch rheology
# ---------------------------------------------------------------------------


def test_t2_graetz_melt_flow_strictly_increases_with_nozzle_temp_and_benchmarks() -> None:
    """Melt-limited flow strictly increases 200 -> 225 -> 250 C for PLA & PETG and hits benchmarks."""
    assert analyze_hotend_thermal is simulate_hotend_thermal
    spec = PrinterSpec()

    for fid in ("pla", "petg"):
        reports = [simulate_hotend_thermal(spec, target_temp_c=t, filament_id=fid) for t in (200.0, 225.0, 250.0)]
        assert reports[0].melt_zone_flow_limit_mm3_s < reports[1].melt_zone_flow_limit_mm3_s < reports[2].melt_zone_flow_limit_mm3_s
        assert reports[0].max_volumetric_flow_mm3_s < reports[1].max_volumetric_flow_mm3_s < reports[2].max_volumetric_flow_mm3_s

    # Graceful positive monotonic behavior when target_temp_c <= extrudable_temp_c
    r_cold = simulate_hotend_thermal(spec, target_temp_c=160.0, filament_id="pla")
    r_edge = simulate_hotend_thermal(spec, target_temp_c=175.0, filament_id="pla")
    r_warm = simulate_hotend_thermal(spec, target_temp_c=190.0, filament_id="pla")
    assert 0.0 < r_cold.melt_zone_flow_limit_mm3_s < r_edge.melt_zone_flow_limit_mm3_s < r_warm.melt_zone_flow_limit_mm3_s

    # Linear proportionality in melt_zone_length_mm
    s_short = spec.model_copy(deep=True)
    s_short.toolhead.hotend.melt_zone_length_mm = 12.0
    s_long = spec.model_copy(deep=True)
    s_long.toolhead.hotend.melt_zone_length_mm = 24.0
    q_short = simulate_hotend_thermal(s_short, filament_id="pla").melt_zone_flow_limit_mm3_s
    q_long = simulate_hotend_thermal(s_long, filament_id="pla").melt_zone_flow_limit_mm3_s
    assert q_long == pytest.approx(2.0 * q_short, rel=1e-6)

    # Real-world V6 -> Rapido HF -> UHF PLA volumetric flow ranges (~14-45 mm^3/s)
    s_v6 = PrinterSpec()
    s_v6.toolhead.hotend.block_material = "aluminum_6061_t6"
    s_v6.toolhead.hotend.melt_zone_length_mm = 12.0
    s_v6.toolhead.hotend.heater_power_w = 40.0

    s_rapido = PrinterSpec()
    s_rapido.toolhead.hotend.melt_zone_length_mm = 18.0
    s_rapido.toolhead.hotend.heater_power_w = 70.0

    s_uhf = PrinterSpec()
    s_uhf.toolhead.hotend.nozzle_material = "copper_c110"
    s_uhf.toolhead.hotend.melt_zone_length_mm = 26.0
    s_uhf.toolhead.hotend.heater_power_w = 110.0

    q_v6 = simulate_hotend_thermal(s_v6, filament_id="pla").max_volumetric_flow_mm3_s
    q_rap = simulate_hotend_thermal(s_rapido, filament_id="pla").max_volumetric_flow_mm3_s
    q_uhf = simulate_hotend_thermal(s_uhf, filament_id="pla").max_volumetric_flow_mm3_s
    assert 13.0 <= q_v6 <= 18.0
    assert 22.0 <= q_rap <= 28.0
    assert 35.0 <= q_uhf <= 45.0


def test_t3_rabinowitsch_power_law_and_arrhenius_pressure_flow() -> None:
    """Verify exact Rabinowitsch pipe solution, P^(1/n) * d^(3+1/n) scaling, and Arrhenius T-shift."""
    n = 0.36
    k_pa_sn = 20000.0
    radius_m = 0.2e-3  # 0.4 mm nozzle
    length_m = 1.0e-3
    q_m3_s = 20.0e-9

    dp_pipe = rabinowitsch_pipe_pressure_drop_pa(q_m3_s, radius_m, length_m, k_pa_sn, n)
    # Closed-form Rabinowitsch pipe inversion: Q = (pi*n*R^3/(3n+1)) * ((R*dP)/(2*K*L))^(1/n)
    q_exact = (math.pi * n * (radius_m**3) / (3.0 * n + 1.0)) * (
        (radius_m * dp_pipe) / (2.0 * k_pa_sn * length_m)
    ) ** (1.0 / n)
    assert q_exact == pytest.approx(q_m3_s, rel=1e-9)

    # In the orifice-dominated pipe limit, Q ~ dP^(1/n) * d^(3 + 1/n)
    # Check doubling pressure scales Q by 2^(1/n)
    q_p1 = solve_pressure_limited_flow_m3_s(10e6, 0.4e-3, 0.0, 2.0e-3, k_pa_sn, n, orifice_length_m=1.0e-3)
    q_p2 = solve_pressure_limited_flow_m3_s(20e6, 0.4e-3, 0.0, 2.0e-3, k_pa_sn, n, orifice_length_m=1.0e-3)
    # Wait: total_pressure_drop includes cone + Bagley; in pure orifice scaling Q scales by 2^(1/n) as well
    # because chamber, cone, orifice, and Bagley all scale identically as Q^n!
    assert (q_p2 / q_p1) == pytest.approx(2.0 ** (1.0 / n), rel=1e-4)

    # Pure capillary pipe scaling with diameter: d^(3 + 1/n) when orifice dominates (r_chamber large, tiny cone/bagley)
    dp1 = nozzle_total_pressure_drop_pa(
        20e-9, 0.4e-3, 0.0, 100.0e-3, k_pa_sn, n, orifice_length_m=0.05, cone_half_angle_deg=89.0, bagley_factor=0.0
    )
    dp2 = nozzle_total_pressure_drop_pa(
        20e-9 * ((0.6 / 0.4) ** (3.0 + 1.0 / n)),
        0.6e-3,
        0.0,
        100.0e-3,
        k_pa_sn,
        n,
        orifice_length_m=0.05,
        cone_half_angle_deg=89.0,
        bagley_factor=0.0,
    )
    assert dp2 == pytest.approx(dp1, rel=1e-3)

    # End-to-end simulate_hotend_thermal: hotter nozzle increases pressure-limited flow via Arrhenius shift
    spec = PrinterSpec()
    q_press_200 = simulate_hotend_thermal(spec, target_temp_c=200.0, filament_id="pla").extruder_pressure_flow_limit_mm3_s
    q_press_225 = simulate_hotend_thermal(spec, target_temp_c=225.0, filament_id="pla").extruder_pressure_flow_limit_mm3_s
    q_press_250 = simulate_hotend_thermal(spec, target_temp_c=250.0, filament_id="pla").extruder_pressure_flow_limit_mm3_s
    assert q_press_200 < q_press_225 < q_press_250

    # Doubling push force scales extruder_pressure_flow_limit_mm3_s by 2^(1/n)
    s_double_f = spec.model_copy(deep=True)
    s_double_f.toolhead.extruder.max_push_force_n = spec.toolhead.extruder.max_push_force_n * 2.0
    q_f1 = simulate_hotend_thermal(spec, filament_id="pla").extruder_pressure_flow_limit_mm3_s
    q_f2 = simulate_hotend_thermal(s_double_f, filament_id="pla").extruder_pressure_flow_limit_mm3_s
    assert (q_f2 / q_f1) == pytest.approx(2.0 ** (1.0 / get_filament("pla").power_law_n), rel=1e-3)


# ---------------------------------------------------------------------------
# M1 & M3: Standalone import (no circular ImportError) & CoreXY 45-deg torque
# ---------------------------------------------------------------------------


def test_m3_standalone_kinematics_import_no_circular_error() -> None:
    """Importing printsim.kinematics in a clean Python interpreter must succeed without ImportError."""
    for stmt in (
        "import printsim.kinematics",
        "from printsim.kinematics.solvers import analyze_kinematics",
        "import printsim.fea; import printsim.kinematics",
    ):
        proc = subprocess.run(
            [sys.executable, "-c", stmt],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, f"Import '{stmt}' failed:\n{proc.stderr}"


def test_m1_corexy_diagonal_45deg_torque_and_rotor_inertia() -> None:
    """CoreXY torque demand uses 45-deg worst-case 1/sqrt(2) (2WD), 1/(2*sqrt(2)) (AWD), and sqrt(2)*a."""
    spec_2wd = PrinterSpec(architecture="corexy")
    spec_2wd.gantry.drive_mode = "2wd"
    asm_2wd = build_assembly(spec_2wd)
    kin_2wd = analyze_kinematics(asm_2wd)

    r_p = spec_2wd.gantry.belt.pitch_radius_m
    a_m_s2 = spec_2wd.kinematics.max_accel_mm_s2 * 1e-3
    j_rot = spec_2wd.kinematics.xy_stepper.rotor_inertia_kg_m2
    alpha_diag = math.sqrt(2.0) * a_m_s2 / r_p

    expected_ty_2wd = ((asm_2wd.y_moving_mass_kg * a_m_s2 + 2.5) / math.sqrt(2.0)) * r_p + j_rot * alpha_diag
    assert kin_2wd.required_torque_y_nm == pytest.approx(expected_ty_2wd, rel=1e-6)

    spec_awd = spec_2wd.model_copy(deep=True)
    spec_awd.gantry.drive_mode = "awd"
    asm_awd = build_assembly(spec_awd)
    kin_awd = analyze_kinematics(asm_awd)
    expected_ty_awd = (
        (asm_awd.y_moving_mass_kg * a_m_s2 + 3.6) / (2.0 * math.sqrt(2.0))
    ) * r_p + j_rot * alpha_diag
    assert kin_awd.required_torque_y_nm == pytest.approx(expected_ty_awd, rel=1e-6)


# ---------------------------------------------------------------------------
# C1-Mech: Exact Klipper toolhead.py centripetal junction deviation speed
# ---------------------------------------------------------------------------


def test_c1_klipper_junction_speed_90deg_scv_and_180deg_zero() -> None:
    """Verify Klipper junction speed equals SCV at 90 deg, 0 at 180 deg reversal, and matches toolhead.py."""
    scv = 5.0
    accel = 8000.0
    v_cruise = 300.0
    d_x = np.array([1.0, 0.0, 0.0])
    d_y = np.array([0.0, 1.0, 0.0])
    d_neg_x = np.array([-1.0, 0.0, 0.0])
    d_30 = np.array([math.cos(math.radians(30.0)), math.sin(math.radians(30.0)), 0.0])
    d_120 = np.array([math.cos(math.radians(120.0)), math.sin(math.radians(120.0)), 0.0])

    # 90-degree corner: exact SCV within < 0.01% (well inside 1%)
    v_90 = _klipper_junction_speed_mm_s(d_x, d_y, 20.0, 20.0, v_cruise, v_cruise, accel, scv)
    assert v_90 == pytest.approx(scv, rel=1e-4)

    # 180-degree reversal: exact 0.0
    v_180 = _klipper_junction_speed_mm_s(d_x, d_neg_x, 20.0, 20.0, v_cruise, v_cruise, accel, scv)
    assert v_180 == pytest.approx(0.0, abs=1e-9)

    # Straight move (0 deg turn): preserves cruise velocity
    v_0 = _klipper_junction_speed_mm_s(d_x, d_x, 20.0, 20.0, v_cruise, 250.0, accel, scv)
    assert v_0 == pytest.approx(250.0, rel=1e-9)

    # Reference Klipper toolhead.py values for 30 deg (17.13 mm/s) and 120 deg (3.22 mm/s)
    v_30 = _klipper_junction_speed_mm_s(d_x, d_30, 50.0, 50.0, v_cruise, v_cruise, accel, scv)
    v_120 = _klipper_junction_speed_mm_s(d_x, d_120, 50.0, 50.0, v_cruise, v_cruise, accel, scv)
    assert v_30 == pytest.approx(17.13, abs=0.02)
    assert v_120 == pytest.approx(3.22, abs=0.02)

    # End-to-end simulate_gcode check:
    # 1) 180-deg reversal (Move 1 -> Move 2) forces exact v_j = 0.0 so Move 2 takes 2*sqrt(40/4000) = 0.200s
    # 2) 90-deg corner (Move 1 -> Move 2 in gcode_90) uses exact v_j = SCV
    spec = PrinterSpec()
    spec.kinematics.max_accel_mm_s2 = 4000.0
    spec.kinematics.max_velocity_mm_s = 500.0
    spec.kinematics.square_corner_velocity_mm_s = 10.0
    asm = build_assembly(spec)
    mod = solve_modal_fea(asm)
    kin = analyze_kinematics(asm)
    th = simulate_hotend_thermal(spec)

    gcode_rev = "\n".join(
        [
            "G90",
            "G0 X10 Y10 F30000",
            "G1 X50 Y10 E1.0 F30000",
            "G1 X10 Y10 E2.0 F30000",
        ]
    )
    vp_rev = simulate_gcode(gcode_rev, asm, mod, kin, th)
    assert vp_rev.segments[2].duration_s == pytest.approx(2.0 * math.sqrt(40.0 / 4000.0), rel=1e-6)

    # 90-deg corner: start at (10,10), go +X 40mm, then +Y 40mm
    # Move 2 enters at v_j = SCV (10 mm/s) and exits at 0 -> exact triangular time:
    # v_peak = sqrt(0.5*10^2 + 4000*40) = sqrt(160050)
    gcode_90 = "\n".join(
        [
            "G90",
            "G0 X10 Y10 F30000",
            "G1 X50 Y10 E1.0 F30000",
            "G1 X50 Y50 E2.0 F30000",
        ]
    )
    vp_90 = simulate_gcode(gcode_90, asm, mod, kin, th)
    v_peak_expected = math.sqrt(0.5 * (10.0**2) + 4000.0 * 40.0)
    dt_90_leg2 = (v_peak_expected - 10.0) / 4000.0 + v_peak_expected / 4000.0
    assert vp_90.segments[2].duration_s == pytest.approx(dt_90_leg2, rel=1e-6)
    assert vp_90.segments[2].peak_speed_mm_s == pytest.approx(v_peak_expected, rel=1e-6)
