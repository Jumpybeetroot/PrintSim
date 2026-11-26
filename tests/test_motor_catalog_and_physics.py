"""Comprehensive physics, datasheet, thermal, and integration tests for STEPPER_CATALOG & StepperSpec."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from printsim.engine import evaluate_printer
from printsim.fea import solve_modal_fea
from printsim.fea.mechatronics import DriveControlConfig, build_coupled_state_space
from printsim.geometry import build_assembly, generate_bom
from printsim.kinematics.solvers import (
    _compute_motor_max_velocity_mm_s,
    analyze_kinematics,
    stepper_available_torque_nm,
)
from printsim.schema import STEPPER_CATALOG, PrinterSpec, StepperSpec

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def test_all_catalog_motors_evaluate_cleanly_across_physics_and_bom() -> None:
    """Every motor in STEPPER_CATALOG must pass Pydantic validation, FEA/kinematics/LPV/BOM with finite outputs."""
    assert len(STEPPER_CATALOG) >= 16
    for required_id in (
        "excit3d_max",
        "isik_ouroboros",
        "ldo_2504ac",
        "ldo_2504ah",
        "ldo_2804ah",
        "kraken_motor_v2",
        "ldo_60_3004",
        "ldo_60_2004ah",
        "ldo_09deg_2504",
        "leadshine_42cm06",
        "moons_420i",
        "stepperonline_2504s",
        "stepperonline_2004",
        "stepperonline_09deg_2004",
        "ldo_40_1684ac",
        "ldo_57_2804mac",
        "stepperonline_23hs22_2804",
    ):
        assert required_id in STEPPER_CATALOG

    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    for pid, entry in STEPPER_CATALOG.items():
        stepper = StepperSpec(**{**entry, "supply_voltage_v": 48.0})
        assert stepper.preset_id == pid
        assert stepper.holding_torque_nm > 0.0
        assert stepper.rated_current_a > 0.0
        assert stepper.run_current_a <= stepper.rated_current_a * 1.05
        assert stepper.phase_resistance_ohm > 0.05
        assert stepper.inductance_mh > 0.0
        assert stepper.rotor_inertia_kg_m2 > 0.0
        assert stepper.motor_mass_g > 0.0
        assert stepper.steps_per_rev in (200, 400)
        assert stepper.nema_size in (14, 17, 23)
        assert stepper.insulation_class in ("B", "F", "H")

        spec = base.model_copy(deep=True)
        spec.kinematics.xy_stepper = stepper
        spec.kinematics.drive_type = "foc_servo" if stepper.has_encoder else "open_loop_stepper"
        rep = evaluate_printer(spec)

        assert math.isfinite(rep.kinematics.motor_max_velocity_mm_s) and rep.kinematics.motor_max_velocity_mm_s > 200.0
        assert math.isfinite(rep.kinematics.motor_max_accel_mm_s2) and rep.kinematics.motor_max_accel_mm_s2 > 500.0
        assert math.isfinite(rep.kinematics.motor_stiffness_n_per_mm) and rep.kinematics.motor_stiffness_n_per_mm > 10.0
        assert math.isfinite(rep.kinematics.motor_heat_w_each) and rep.kinematics.motor_heat_w_each > 0.5
        assert rep.kinematics.motor_est_temp_c > spec.frame.chamber_temp_c
        assert rep.kinematics.motor_overcurrent_risk is False

        asm = build_assembly(spec)
        mod = solve_modal_fea(asm)
        mdl = build_coupled_state_space(
            asm,
            mod,
            0.5,
            0.5,
            DriveControlConfig(drive_type=spec.kinematics.drive_type),
        )
        assert mdl.k_em_per_motor_n_m > 0.0
        assert mdl.b_em_per_motor_ns_m > 0.0

        bom = generate_bom(spec)
        xy_item = next(item for item in bom.items if item.sku == "MTR-XY")
        assert xy_item.unit_cost_usd > 0.0
        assert xy_item.total_mass_g == pytest.approx(stepper.motor_mass_g * xy_item.quantity)


def test_excit3d_max_motor_datasheet_and_48v_high_speed_physics() -> None:
    """Verify Excit3D Max Motor (EX3D-MAX-OL) datasheet values, 3.5A-4.0A copper loss, Class H margin, and 48V speed."""
    st = StepperSpec(**{**STEPPER_CATALOG["excit3d_max"], "supply_voltage_v": 48.0})
    assert st.holding_torque_nm == pytest.approx(0.56)
    assert st.rated_current_a == pytest.approx(4.0)
    assert st.run_current_a == pytest.approx(3.5)
    assert st.phase_resistance_ohm == pytest.approx(0.45)
    assert st.inductance_mh == pytest.approx(0.6)
    assert st.rotor_inertia_kg_m2 == pytest.approx(8.2e-6)
    assert st.steps_per_rev == 200
    assert st.insulation_class == "H"
    assert st.max_winding_temp_c == pytest.approx(180.0)

    # Low winding resistance (0.45 Ohm) keeps copper loss moderate (~11.0 W) even at 3.5 A RMS
    assert st.power_dissipation_w == pytest.approx(2.0 * (3.5**2) * 0.45, rel=1e-9)
    assert st.power_dissipation_w < 11.5

    # Full 4.0 A rated operation is accepted by schema and stays safely under Class H 180 C in a 60 C chamber
    st_4a = st.model_copy(update={"run_current_a": 4.0})
    assert st_4a.effective_holding_torque_nm == pytest.approx(0.56, rel=1e-9)
    assert st_4a.estimated_winding_temp_c(60.0) < 170.0

    # At 48V, Excit3D Max (0.56 Nm, 4.0A, 0.6 mH) beats both LDO 2504AC (1.5 mH) and LDO 2804AH (0.42 Nm) at high speed
    ldo_2504 = StepperSpec(**{**STEPPER_CATALOG["ldo_2504ac"], "supply_voltage_v": 48.0, "run_current_a": 2.0})
    ldo_2804 = StepperSpec(**{**STEPPER_CATALOG["ldo_2804ah"], "supply_voltage_v": 48.0, "run_current_a": 2.4})

    assert stepper_available_torque_nm(st, 1500.0) > stepper_available_torque_nm(ldo_2504, 1500.0)
    assert stepper_available_torque_nm(st, 1500.0) > stepper_available_torque_nm(ldo_2804, 1500.0)

    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_monolith_awd.yaml")
    s_ex = base.model_copy(deep=True)
    s_ex.kinematics.xy_stepper = st
    s_2504 = base.model_copy(deep=True)
    s_2504.kinematics.xy_stepper = ldo_2504

    k_ex = analyze_kinematics(build_assembly(s_ex))
    k_2504 = analyze_kinematics(build_assembly(s_2504))
    assert k_ex.motor_max_velocity_mm_s > k_2504.motor_max_velocity_mm_s * 1.35
    assert k_ex.motor_max_accel_mm_s2 > k_2504.motor_max_accel_mm_s2
    assert k_ex.motor_thermal_risk is False


def test_isik_ouroboros_stepper_datasheet_and_foc_closed_loop_physics() -> None:
    """Verify Isik's Tech Ouroboros (ENCSTP) datasheet specs, peak/RMS consistency, and TMC4671 FOC boost."""
    st = StepperSpec(**{**STEPPER_CATALOG["isik_ouroboros"], "supply_voltage_v": 48.0})
    assert st.holding_torque_nm == pytest.approx(0.55)
    assert st.rated_current_a == pytest.approx(2.5)
    assert st.phase_resistance_ohm == pytest.approx(1.2)
    assert st.inductance_mh == pytest.approx(1.6)
    assert st.rotor_inertia_kg_m2 == pytest.approx(8.2e-6)
    assert st.steps_per_rev == 200
    assert st.insulation_class == "B"
    assert st.has_encoder is True

    # Matches Isik's Tech TMC4671 docs: rated_current = 2.5 A RMS -> peak = 2.5 * sqrt(2) = 3.5355 A
    st_rated = st.model_copy(update={"run_current_a": 2.5})
    k_t, i_pk = st_rated._phase_constants()
    assert i_pk == pytest.approx(2.5 * math.sqrt(2.0), rel=1e-9)
    assert k_t * i_pk == pytest.approx(0.55, rel=1e-9)

    # Paired with FOC Servo (Ouroboros TMC4671), torque utilization and max velocity exceed open-loop stepper
    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    base.kinematics.xy_stepper = st
    base.kinematics.drive_type = "open_loop_stepper"
    k_ol = analyze_kinematics(build_assembly(base))

    foc = base.model_copy(deep=True)
    foc.kinematics.drive_type = "foc_servo"
    k_foc = analyze_kinematics(build_assembly(foc))

    assert k_foc.motor_max_accel_mm_s2 > k_ol.motor_max_accel_mm_s2 * 1.35
    assert k_foc.motor_max_velocity_mm_s > k_ol.motor_max_velocity_mm_s * 1.06
    assert k_foc.motor_overcurrent_risk is False


def test_overcurrent_high_speed_torque_stays_monotonically_non_decreasing() -> None:
    """Above corner speed, raising run_current_a above rated_current_a must not artificially decrease torque."""
    st_rated = StepperSpec(
        holding_torque_nm=0.55,
        rated_current_a=2.5,
        run_current_a=2.5,
        phase_resistance_ohm=1.2,
        inductance_mh=1.5,
        supply_voltage_v=24.0,
        steps_per_rev=200,
    )
    st_over = st_rated.model_copy(update={"run_current_a": 3.5})

    # Low speed (below corner): overcurrent yields strictly more (softly saturated) torque
    t_low_rated = stepper_available_torque_nm(st_rated, 200.0)
    t_low_over = stepper_available_torque_nm(st_over, 200.0)
    assert t_low_over > t_low_rated

    # High speed (deep above corner, voltage-limited): actual I_q is voltage-clamped below I_rated,
    # so torque must stay equal (monotonically non-decreasing), never artificially dip below rated!
    for rpm in (900.0, 1200.0, 1500.0):
        t_r = stepper_available_torque_nm(st_rated, rpm)
        t_o = stepper_available_torque_nm(st_over, rpm)
        assert t_o >= t_r - 1e-12


def test_winding_thermal_equilibrium_and_class_b_vs_class_h_in_heated_chamber() -> None:
    """Verify exact closed-form hot-copper TCR equilibrium and Class B (130C) vs Class H (180C) chamber safety."""
    alpha_cu = 0.003854
    st_b = StepperSpec(**{**STEPPER_CATALOG["ldo_2504ac"], "run_current_a": 2.3})
    st_h = StepperSpec(**{**STEPPER_CATALOG["ldo_2504ah"], "run_current_a": 2.3})

    t_amb = 65.0
    t_w = st_b.estimated_winding_temp_c(t_amb)
    # Exact self-consistent balance check: T_w - T_amb == R_th * P_25 * (1 + alpha_cu * (T_w - 25))
    lhs = t_w - t_amb
    rhs = st_b.thermal_resistance_k_per_w * st_b.power_dissipation_w * (1.0 + alpha_cu * (t_w - 25.0))
    assert lhs == pytest.approx(rhs, rel=1e-9)

    # Same current & chamber: Class B (130 C) flags thermal risk while Class H (180 C) remains safe
    spec_b = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    spec_b.frame.chamber_temp_c = 65.0
    spec_b.kinematics.xy_stepper = st_b
    rep_b = evaluate_printer(spec_b)

    spec_h = spec_b.model_copy(deep=True)
    spec_h.kinematics.xy_stepper = st_h
    rep_h = evaluate_printer(spec_h)

    assert rep_b.kinematics.motor_est_temp_c == pytest.approx(rep_h.kinematics.motor_est_temp_c, rel=1e-6)
    assert rep_b.kinematics.motor_thermal_risk is True
    assert rep_h.kinematics.motor_thermal_risk is False
    assert any("winding thermal risk" in d.lower() for d in rep_b.diagnostics)
    assert not any("winding thermal risk" in d.lower() for d in rep_h.diagnostics)


def test_nema23_rotor_inertia_pulley_tradeoff_shaft_limit_and_unclipped_velocity() -> None:
    """Verify NEMA 23 reflected inertia penalty on 20T vs 36T pulleys, 6.35mm shaft limit, and >5000 mm/s bisection."""
    n23 = StepperSpec(**{**STEPPER_CATALOG["stepperonline_23hs22_2804"], "supply_voltage_v": 48.0})
    assert n23.nema_size == 23
    assert n23.shaft_diameter_mm == pytest.approx(6.35)
    assert n23.thermal_resistance_k_per_w < StepperSpec(**STEPPER_CATALOG["ldo_2504ac"]).thermal_resistance_k_per_w

    # On 60 N single-shear belt tension with CNC mounts & 6mm idler pins: NEMA 17 (5mm shaft) flags shaft_bending_risk, NEMA 23 (6.35mm) does not
    spec_17 = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    spec_17.gantry.bracket_material = "aluminum_6061_t6"
    spec_17.gantry.double_shear_pulleys = False
    spec_17.gantry.belt.type = "gt2_9mm"
    spec_17.gantry.belt.idler_shaft_diameter_mm = 6.0
    spec_17.gantry.belt.pretension_n = 60.0
    k_17 = analyze_kinematics(build_assembly(spec_17))
    assert k_17.shaft_bending_risk is True

    spec_23 = spec_17.model_copy(deep=True)
    spec_23.kinematics.xy_stepper = n23
    k_23 = analyze_kinematics(build_assembly(spec_23))
    assert k_23.shaft_bending_risk is False

    # Dynamic upper-bound expansion in _compute_motor_max_velocity_mm_s does not clip at 5000 mm/s
    st_fast = StepperSpec(**{**STEPPER_CATALOG["excit3d_max"], "supply_voltage_v": 60.0})
    r_p_36t = (36 * 2.0 * 1e-3) / (2.0 * math.pi)
    v_max_fast = _compute_motor_max_velocity_mm_s(
        stepper=st_fast,
        r_p=r_p_36t,
        drive_type="foc_servo",
        sf=1.15,
        vel_scale=1.0,
        t_req_reserve=0.05,
    )
    assert v_max_fast > 5000.0
