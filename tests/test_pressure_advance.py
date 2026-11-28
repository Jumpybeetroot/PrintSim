"""Pressure-advance, filament-path compliance, and extrusion transient tests.

Closed-form checks: a linear nozzle ``F = R v`` behind a spring of compliance ``C`` is a
first-order lag with time constant ``tau = R C``; linear advance with ``K = tau`` cancels it
exactly; the volume oozed after a hard stop equals the stored compression ``C F A_fil``.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from printsim.schema import PrinterSpec
from printsim.thermal.filaments import FILAMENT_CATALOG, get_filament
from printsim.thermal.hotend import simulate_hotend_thermal
from printsim.thermal.pressure_advance import (
    BOWDEN_TUBE_OD_M,
    PTFE_TUBE_MODULUS_PA,
    TAIT_ABS_B3M_PA,
    TAIT_ABS_B4M_PER_K,
    TAIT_ABS_B5_K,
    build_push_force_table,
    compute_filament_path_compliance,
    depressurization_time_s,
    simulate_extrusion_transient,
    simulate_pressure_advance,
    tait_melt_bulk_modulus_pa,
)

_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "corexy_voron_style.yaml"
_BASE = yaml.safe_load(_EXAMPLE.read_text())


def _spec(**extruder_overrides) -> PrinterSpec:
    data = copy.deepcopy(_BASE)
    if extruder_overrides:
        data.setdefault("toolhead", {}).setdefault("extruder", {}).update(extruder_overrides)
    return PrinterSpec.model_validate(data)


def _linear_tables(r_ns_per_m: float, v_max: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    v = np.linspace(0.0, v_max, 6)
    return v, r_ns_per_m * v


# --------------------------------------------------------------------------- Tait melt compressibility


def test_tait_bulk_modulus_matches_printdb_abs_worked_example():
    # PrintDB references/thermal/thermal-properties-vs-temperature-pvt worked example:
    # ABS at 230 C: beta_T = 4.03e-10 1/Pa (2.48 GPa) at 0 Pa and 3.73e-10 1/Pa at 20 MPa.
    assert 1.0 / tait_melt_bulk_modulus_pa(230.0, 0.0) == pytest.approx(4.03e-10, rel=5e-3)
    assert 1.0 / tait_melt_bulk_modulus_pa(230.0, 20e6) == pytest.approx(3.73e-10, rel=5e-3)


def test_tait_bulk_modulus_softens_with_temperature_and_stiffens_with_pressure():
    assert tait_melt_bulk_modulus_pa(260.0) < tait_melt_bulk_modulus_pa(230.0) < tait_melt_bulk_modulus_pa(200.0)
    assert tait_melt_bulk_modulus_pa(230.0, 30e6) > tait_melt_bulk_modulus_pa(230.0, 0.0)


# --------------------------------------------------------------------------- compliance closed forms


def test_direct_drive_compliance_matches_closed_form():
    spec = _spec(type="direct_drive", filament_path_length_mm=40.0)
    fil = get_filament("pla")
    he = spec.toolhead.hotend
    comp = compute_filament_path_compliance(spec, fil, target_temp_c=215.0, cold_side_temp_c=45.0)

    a_fil = math.pi * (0.5 * 1.75e-3) ** 2
    assert comp.filament_column_m_per_n == pytest.approx(0.040 / (fil.youngs_modulus_pa * a_fil), rel=1e-12)
    assert comp.bowden_tube_m_per_n == 0.0
    assert comp.bowden_slack_m == 0.0

    hot_fraction = (215.0 - fil.glass_transition_c) / (215.0 - 45.0)
    volume = 0.25 * math.pi * (he.heatbreak_id_mm * 1e-3) ** 2 * (
        he.melt_zone_length_mm * 1e-3 + hot_fraction * he.heatbreak_length_mm * 1e-3
    )
    assert comp.compressible_volume_m3 == pytest.approx(volume, rel=1e-12)
    assert comp.melt_m_per_n == pytest.approx(
        volume / (tait_melt_bulk_modulus_pa(215.0) * a_fil**2), rel=1e-12
    )
    assert comp.total_m_per_n == pytest.approx(comp.filament_column_m_per_n + comp.melt_m_per_n)
    # Euler fixed-fixed load of the 40 mm run: pi^2 E I / (0.5 L)^2
    i_fil = math.pi * (1.75e-3) ** 4 / 64.0
    assert comp.column_buckling_load_n == pytest.approx(
        math.pi**2 * fil.youngs_modulus_pa * i_fil / 0.020**2, rel=1e-12
    )


def test_bowden_adds_tube_stretch_column_length_and_bend_slack():
    fil = get_filament("pla")
    dd = compute_filament_path_compliance(_spec(type="direct_drive"), fil, 215.0, 45.0)
    spec = _spec(type="bowden", bowden_tube_length_mm=600.0, bowden_tube_id_mm=2.0)
    bw = compute_filament_path_compliance(spec, fil, 215.0, 45.0, bowden_bend_angle_deg=90.0)

    a_fil = math.pi * (0.5 * 1.75e-3) ** 2
    a_tube = 0.25 * math.pi * (BOWDEN_TUBE_OD_M**2 - 2.0e-3**2)
    assert bw.bowden_tube_m_per_n == pytest.approx(0.600 / (PTFE_TUBE_MODULUS_PA * a_tube), rel=1e-12)
    assert bw.filament_column_m_per_n - dd.filament_column_m_per_n == pytest.approx(
        0.600 / (fil.youngs_modulus_pa * a_fil), rel=1e-9
    )
    assert bw.melt_m_per_n == pytest.approx(dd.melt_m_per_n)
    # Bend-clearance dead band: swept angle x diametral clearance
    assert bw.bowden_slack_m == pytest.approx(0.5 * math.pi * 0.25e-3, rel=1e-12)
    assert bw.total_m_per_n > 5.0 * dd.total_m_per_n


# --------------------------------------------------------------------------- linear-nozzle transients


def test_step_response_of_linear_nozzle_is_first_order_lag():
    r, c = 2000.0, 1.0e-5  # N*s/m, m/N -> tau = 20 ms
    tau = r * c
    v_tab, f_tab = _linear_tables(r)
    t = np.arange(0.0, 0.2, 1e-4)
    v0 = 0.005
    res = simulate_extrusion_transient(t, np.full_like(t, v0), c, v_tab, f_tab)
    expected = v0 * (1.0 - np.exp(-t / tau))
    assert np.max(np.abs(res.outflow_speed_m_s - expected)) < 1e-6 * v0
    # Under-extruded volume during the start-up equals the stored compression C*F*A = tau*v0*A
    a_fil = math.pi * (0.5 * 1.75e-3) ** 2
    assert -res.volume_error_m3(a_fil)[-1] == pytest.approx(tau * v0 * a_fil, rel=2e-3)


def test_linear_advance_equal_to_time_constant_cancels_the_lag_on_a_ramp():
    r, c = 2000.0, 1.0e-5
    tau = r * c
    v_tab, f_tab = _linear_tables(r)
    t = np.arange(0.0, 0.1, 1e-4)
    accel = 0.05  # m/s^2 of filament
    v_nom = accel * t

    uncompensated = simulate_extrusion_transient(t, v_nom, c, v_tab, f_tab)
    compensated = simulate_extrusion_transient(t, v_nom, c, v_tab, f_tab, pressure_advance_s=tau)

    # Without advance the outflow trails the ramp by tau * accel once the start-up decays
    assert (v_nom - uncompensated.outflow_speed_m_s)[-1] == pytest.approx(
        tau * accel * (1.0 - math.exp(-t[-1] / tau)), rel=1e-3
    )
    # With K = tau the outflow tracks the commanded ramp
    assert np.max(np.abs(compensated.outflow_speed_m_s - v_nom)) < 2e-3 * v_nom[-1]


def test_ooze_volume_after_hard_stop_equals_stored_compression():
    r, c = 2000.0, 1.0e-5
    v_tab, f_tab = _linear_tables(r)
    v0 = 0.005
    x0 = c * r * v0
    t = np.arange(0.0, 0.4, 1e-4)
    res = simulate_extrusion_transient(t, np.zeros_like(t), c, v_tab, f_tab, initial_compression_m=x0)
    oozed_m = float(np.sum(0.5 * (res.outflow_speed_m_s[1:] + res.outflow_speed_m_s[:-1]) * np.diff(t)))
    assert oozed_m == pytest.approx(x0, rel=1e-3)
    assert res.compression_m[-1] < 1e-6 * x0
    assert np.all(res.compression_m >= 0.0)


def test_depressurization_time_matches_linear_closed_form():
    r, c = 2000.0, 1.0e-5
    v_tab, f_tab = _linear_tables(r)
    t95 = depressurization_time_s(c, v_tab, f_tab, start_speed_m_s=0.005, end_fraction=0.05)
    assert t95 == pytest.approx(r * c * math.log(20.0), rel=1e-3)


def test_transient_rejects_non_uniform_time_grid():
    v_tab, f_tab = _linear_tables(2000.0)
    with pytest.raises(ValueError):
        simulate_extrusion_transient(np.array([0.0, 0.1, 0.3]), np.zeros(3), 1e-5, v_tab, f_tab)


# --------------------------------------------------------------------------- full report on a real spec


def test_report_is_self_consistent_and_in_documented_klipper_band():
    spec = _spec(type="direct_drive")
    rep = simulate_pressure_advance(spec, filament_id="pla")
    a_fil_mm2 = math.pi * 0.875**2

    assert rep.reference_flow_mm3_s == pytest.approx(12.0)
    assert rep.reference_filament_speed_mm_s == pytest.approx(12.0 / a_fil_mm2, rel=1e-9)
    # Advance = K * v = C * F, and the oozed volume is that advance times the filament area
    assert rep.stored_filament_compression_mm == pytest.approx(
        rep.pressure_advance_s * rep.reference_filament_speed_mm_s, rel=1e-9
    )
    assert rep.stored_filament_compression_mm == pytest.approx(
        rep.system_compliance_um_per_n * 1e-3 * rep.push_force_at_reference_n, rel=1e-9
    )
    assert rep.ooze_volume_mm3 == pytest.approx(rep.stored_filament_compression_mm * a_fil_mm2, rel=1e-9)
    assert rep.system_stiffness_n_per_mm == pytest.approx(1e3 / rep.system_compliance_um_per_n, rel=1e-9)
    assert rep.straight_column_valid and not rep.warnings

    # Sanity band, not a calibration: Klipper documents 0.050-1.000 s as typical with the high
    # end for Bowden; the unmodelled compliance terms make this prediction a lower bound.
    assert 0.01 < rep.pressure_advance_s < 0.10

    bowden = simulate_pressure_advance(_spec(type="bowden"), filament_id="pla")
    assert 0.2 < bowden.pressure_advance_s < 1.0
    assert bowden.pressure_advance_s > 8.0 * rep.pressure_advance_s
    assert bowden.min_retraction_mm == pytest.approx(
        bowden.stored_filament_compression_mm + bowden.bowden_slack_mm, rel=1e-9
    )
    assert bowden.pa_extruder_velocity_spike_mm_s > 8.0 * rep.pa_extruder_velocity_spike_mm_s


def test_equivalent_constant_falls_with_flow_while_shear_thinning_dominates():
    rep = simulate_pressure_advance(_spec(), filament_id="pla")
    ks = [pt["pressure_advance_s"] for pt in rep.advance_curve]
    # 5 %..50 % of the flow limit sits in the shear-thinning regime, before the melt plug term
    assert ks[0] > ks[1] > ks[2] > ks[3] > ks[4]
    # The advance itself always grows with flow
    adv = [pt["advance_mm"] for pt in rep.advance_curve]
    assert all(b > a for a, b in zip(adv, adv[1:]))


def test_physical_monotonic_trends():
    base = simulate_pressure_advance(_spec(type="bowden", bowden_tube_length_mm=400.0), "pla")
    longer = simulate_pressure_advance(_spec(type="bowden", bowden_tube_length_mm=800.0), "pla")
    assert longer.pressure_advance_s > base.pressure_advance_s

    tight = simulate_pressure_advance(
        _spec(type="bowden", bowden_tube_length_mm=400.0, bowden_tube_id_mm=1.9), "pla"
    )
    assert tight.bowden_slack_mm < base.bowden_slack_mm

    spec = _spec()
    cool = simulate_pressure_advance(spec, "pla", target_temp_c=200.0, reference_flow_mm3_s=6.0)
    hot = simulate_pressure_advance(spec, "pla", target_temp_c=235.0, reference_flow_mm3_s=6.0)
    assert hot.push_force_at_reference_n < cool.push_force_at_reference_n
    assert hot.pressure_advance_s < cool.pressure_advance_s

    data = copy.deepcopy(_BASE)
    data.setdefault("toolhead", {}).setdefault("hotend", {})["nozzle_diameter_mm"] = 0.6
    wide = simulate_pressure_advance(PrinterSpec.model_validate(data), "pla", reference_flow_mm3_s=6.0)
    narrow = simulate_pressure_advance(spec, "pla", reference_flow_mm3_s=6.0)
    assert wide.pressure_advance_s < narrow.pressure_advance_s


def test_flexible_filament_is_flagged_outside_straight_column_validity():
    rep = simulate_pressure_advance(_spec(), filament_id="tpu_95a")
    assert rep.push_force_at_reference_n > rep.column_buckling_load_n
    assert not rep.straight_column_valid
    assert rep.warnings and "buckling" in rep.warnings[0]


def test_predicted_advance_removes_most_of_the_acceleration_flow_error():
    spec = _spec()
    fil = get_filament("pla")
    thermal = simulate_hotend_thermal(spec, filament_id="pla")
    rep = simulate_pressure_advance(spec, "pla", thermal=thermal)
    v_tab, f_tab = build_push_force_table(
        spec, fil, thermal.target_temp_c, thermal.nozzle_type, thermal.max_volumetric_flow_mm3_s * 1e-9
    )
    c = rep.system_compliance_um_per_n * 1e-6
    v_ref = rep.reference_filament_speed_mm_s * 1e-3
    dt = 2e-4
    t = np.arange(0.0, 0.5, dt)
    # 20 ms ramp from 20 % to 100 % of the reference speed, started at steady state
    v_nom = np.clip(0.2 * v_ref + (0.8 * v_ref / 0.02) * (t - 0.05), 0.2 * v_ref, v_ref)
    x0 = c * float(np.interp(0.2 * v_ref, v_tab, f_tab))

    raw = simulate_extrusion_transient(t, v_nom, c, v_tab, f_tab, initial_compression_m=x0)
    chord_k = (c * float(np.interp(v_ref, v_tab, f_tab)) - x0) / (0.8 * v_ref)
    comp = simulate_extrusion_transient(
        t, v_nom, c, v_tab, f_tab, pressure_advance_s=chord_k, initial_compression_m=x0
    )
    a_fil = math.pi * (0.5 * spec.toolhead.extruder.filament_diameter_mm * 1e-3) ** 2
    raw_deficit = -raw.volume_error_m3(a_fil)[-1]
    comp_deficit = abs(comp.volume_error_m3(a_fil)[-1])
    # Uncompensated deficit is the extra stored compression between the two speeds
    assert raw_deficit == pytest.approx((c * float(np.interp(v_ref, v_tab, f_tab)) - x0) * a_fil, rel=0.02)
    assert comp_deficit < 0.05 * raw_deficit


def test_thermal_report_reuse_matches_standalone_call():
    spec = _spec()
    thermal = simulate_hotend_thermal(spec, filament_id="petg")
    a = simulate_pressure_advance(spec, "petg", thermal=thermal)
    b = simulate_pressure_advance(spec, "petg")
    assert a.to_dict() == b.to_dict()


# --------------------------------------------------------------------------- PrintDB single source of truth


def test_constants_match_printdb_records_when_available():
    from printsim.schema.printdb_bridge import FILAMENT_PRINTDB_MAP, get_printdb, get_record_specs

    if get_printdb() is None:
        pytest.skip("PrintDB not installed")
    abs_si = get_record_specs("materials/generic-abs", si=True)
    tait = abs_si["tait_pvt"]
    assert tait["b3m_pa"] == pytest.approx(TAIT_ABS_B3M_PA)
    assert tait["b4m_per_k"] == pytest.approx(TAIT_ABS_B4M_PER_K)
    assert tait["b5_k"] == pytest.approx(TAIT_ABS_B5_K)
    ptfe = get_record_specs("materials/ptfe-gasket-sheet", si=True)
    assert ptfe["tensile_modulus_pa"] == pytest.approx(PTFE_TUBE_MODULUS_PA)
    for key, rid in FILAMENT_PRINTDB_MAP.items():
        si = get_record_specs(rid, si=True)
        assert FILAMENT_CATALOG[key].youngs_modulus_pa == pytest.approx(si["tensile_modulus_pa"]), key
