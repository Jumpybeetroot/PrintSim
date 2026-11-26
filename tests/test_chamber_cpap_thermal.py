"""Unit tests for enclosed chamber PTC heating, PIR insulation, CPAP blower/hose & Kalico shared duct."""

from __future__ import annotations

import pytest

from printsim.schema import PrinterSpec
from printsim.thermal import (
    ChamberAndCpapReport,
    simulate_chamber_and_cpap,
    simulate_hotend_thermal,
)


def _enclosed_abs_spec(
    heater_id: str = "none",
    insulation_mm: float = 0.0,
    cpap_blower_id: str = "none",
    cpap_hose_id_mm: float = 15.0,
    cpap_pwm_pct: float = 100.0,
    shared_duct: bool = False,
) -> PrinterSpec:
    spec = PrinterSpec()
    spec.frame.enclosed_panels = True
    spec.frame.top_panel = True
    spec.frame.panel_material = "acm_dibond"
    spec.frame.panel_thickness_mm = 3.0
    spec.frame.chamber_heater_id = heater_id  # type: ignore[assignment]
    spec.frame.chamber_insulation_mm = insulation_mm
    spec.bed.target_temp_c = 110.0
    spec.toolhead.cpap_blower_id = cpap_blower_id  # type: ignore[assignment]
    spec.toolhead.cpap_hose_id_mm = cpap_hose_id_mm
    spec.toolhead.cpap_pwm_pct = cpap_pwm_pct
    spec.toolhead.shared_heatsink_cpap_duct = shared_duct
    return spec


def test_ws7040_exhaust_heating_scales_with_pwm_power() -> None:
    """WS7040 exhaust heating is +5.5 C at 100% PWM and scales as (pwm/100)^1.5."""
    spec_100 = PrinterSpec()
    spec_100.toolhead.cpap_blower_id = "ws7040_24v"
    spec_100.toolhead.cpap_pwm_pct = 100.0

    spec_50 = spec_100.model_copy(deep=True)
    spec_50.toolhead.cpap_pwm_pct = 50.0

    rep_100 = simulate_chamber_and_cpap(spec_100, room_temp_c=22.0)
    rep_50 = simulate_chamber_and_cpap(spec_50, room_temp_c=22.0)

    assert isinstance(rep_100, ChamberAndCpapReport)
    assert rep_100.cpap_exhaust_temp_rise_c == pytest.approx(5.5, abs=1e-6)
    assert rep_50.cpap_exhaust_temp_rise_c == pytest.approx(5.5 * (0.5**1.5), rel=1e-6)
    assert rep_50.cpap_exhaust_temp_rise_c < rep_100.cpap_exhaust_temp_rise_c
    assert rep_100.cpap_outlet_air_temp_c == pytest.approx(27.5, abs=1e-6)
    assert rep_50.cpap_outlet_air_temp_c < rep_100.cpap_outlet_air_temp_c


def test_cpap_blower_airflow_ordering_wm7575_gt_wm9290_gt_ws7040() -> None:
    """Delivered airflow strictly follows WM7575 > WM9290 > WS7040 for both 15 mm and 18 mm hoses."""
    for hose_mm in (15.0, 18.0):
        reports = {}
        for bid in ("ws7040_24v", "wm9290_24v", "wm7575_220vac"):
            spec = PrinterSpec()
            spec.toolhead.cpap_blower_id = bid  # type: ignore[assignment]
            spec.toolhead.cpap_hose_id_mm = hose_mm
            spec.toolhead.cpap_pwm_pct = 100.0
            reports[bid] = simulate_chamber_and_cpap(spec)

        assert (
            reports["wm7575_220vac"].cpap_delivered_dm3_s
            > reports["wm9290_24v"].cpap_delivered_dm3_s
            > reports["ws7040_24v"].cpap_delivered_dm3_s
            > 0.0
        )
        assert (
            reports["wm7575_220vac"].cpap_delivered_cfm
            > reports["wm9290_24v"].cpap_delivered_cfm
            > reports["ws7040_24v"].cpap_delivered_cfm
            > 0.0
        )

    # At 18 mm reference hose & 100% PWM, delivered dm3/s matches Monolith catalog benchmarks
    s_ref = PrinterSpec()
    s_ref.toolhead.cpap_blower_id = "ws7040_24v"
    s_ref.toolhead.cpap_hose_id_mm = 18.0
    s_ref.toolhead.cpap_pwm_pct = 100.0
    r_ref = simulate_chamber_and_cpap(s_ref)
    assert r_ref.cpap_delivered_dm3_s == pytest.approx(5.8, rel=1e-3)
    assert r_ref.cpap_delivered_cfm == pytest.approx(12.29, rel=2e-3)


def test_cpap_15mm_vs_18mm_hose_flow_drop_and_moving_mass() -> None:
    """15 mm HT hose delivers ~15-22% less flow than 18 mm hose and cuts moving mass (~18 g vs ~32 g)."""
    spec_18 = PrinterSpec()
    spec_18.toolhead.cpap_blower_id = "ws7040_24v"
    spec_18.toolhead.cpap_hose_id_mm = 18.0
    spec_18.toolhead.cpap_pwm_pct = 100.0

    spec_15 = spec_18.model_copy(deep=True)
    spec_15.toolhead.cpap_hose_id_mm = 15.0

    spec_none = PrinterSpec()
    spec_none.toolhead.cpap_blower_id = "none"

    rep_18 = simulate_chamber_and_cpap(spec_18)
    rep_15 = simulate_chamber_and_cpap(spec_15)
    rep_none = simulate_chamber_and_cpap(spec_none)

    ratio = rep_15.cpap_delivered_dm3_s / rep_18.cpap_delivered_dm3_s
    assert rep_18.cpap_hose_flow_factor == pytest.approx(1.0, abs=1e-6)
    assert 0.78 <= ratio <= 0.85
    assert 0.82 <= rep_15.cpap_hose_flow_factor <= 0.85

    assert rep_15.cpap_hose_moving_mass_g == pytest.approx(18.0, abs=1.0)
    assert rep_18.cpap_hose_moving_mass_g == pytest.approx(32.0, abs=1.5)
    assert rep_15.cpap_hose_moving_mass_g < rep_18.cpap_hose_moving_mass_g
    assert rep_none.cpap_hose_moving_mass_g == 0.0
    assert rep_none.cpap_delivered_cfm == 0.0


def test_chamber_heater_progression_and_pir_insulation_effects() -> None:
    """500W PTC > 100W PTC > passive steady temp & heatup speed; 20 mm PIR cuts UA & heatup time."""
    rep_none = simulate_chamber_and_cpap(_enclosed_abs_spec(heater_id="none", insulation_mm=0.0))
    rep_100 = simulate_chamber_and_cpap(
        _enclosed_abs_spec(heater_id="monolith_100w_ptc", insulation_mm=0.0)
    )
    rep_500 = simulate_chamber_and_cpap(
        _enclosed_abs_spec(heater_id="monolith_500w_ptc", insulation_mm=0.0)
    )

    assert rep_500.chamber_steady_temp_c > rep_100.chamber_steady_temp_c > rep_none.chamber_steady_temp_c
    assert rep_none.chamber_heatup_time_60c_s is None
    assert rep_100.chamber_heatup_time_60c_s is not None
    assert rep_500.chamber_heatup_time_60c_s is not None
    assert rep_500.chamber_heatup_time_60c_s < rep_100.chamber_heatup_time_60c_s

    # 20 mm PIR insulation vs 0 mm PIR (for both 100W and 500W heaters)
    rep_100_pir = simulate_chamber_and_cpap(
        _enclosed_abs_spec(heater_id="monolith_100w_ptc", insulation_mm=20.0)
    )
    rep_500_pir = simulate_chamber_and_cpap(
        _enclosed_abs_spec(heater_id="monolith_500w_ptc", insulation_mm=20.0)
    )

    assert rep_100_pir.enclosure_ua_w_per_k < rep_100.enclosure_ua_w_per_k * 0.5
    assert rep_100_pir.chamber_steady_temp_c > rep_100.chamber_steady_temp_c + 15.0
    assert rep_100_pir.chamber_heatup_time_60c_s is not None
    assert rep_100_pir.chamber_heatup_time_60c_s < rep_100.chamber_heatup_time_60c_s

    assert rep_500_pir.chamber_steady_temp_c > rep_500.chamber_steady_temp_c
    assert rep_500_pir.chamber_steady_temp_c < 135.0  # Thermal bridges + PTC Curie cap prevent runaway
    assert rep_500_pir.chamber_heatup_time_60c_s is not None
    assert rep_500_pir.chamber_heatup_time_60c_s < rep_500.chamber_heatup_time_60c_s

    # Open frame without panels stays near room temperature and never reaches 60 C
    open_spec = PrinterSpec()
    open_spec.frame.enclosed_panels = False
    open_spec.frame.chamber_heater_id = "monolith_100w_ptc"
    open_rep = simulate_chamber_and_cpap(open_spec, room_temp_c=22.0)
    assert open_rep.enclosure_ua_w_per_k > rep_100.enclosure_ua_w_per_k * 5.0
    assert open_rep.chamber_steady_temp_c < 35.0
    assert open_rep.chamber_heatup_time_60c_s is None


def test_kalico_shared_heatsink_cpap_duct_enforces_minimum_pwm() -> None:
    """Kalico [heater_fan] shared duct enforces >=25% CPAP PWM at hotend >50 C even when PWM=0%."""
    spec_shared = PrinterSpec()
    spec_shared.toolhead.cpap_blower_id = "ws7040_24v"
    spec_shared.toolhead.cpap_pwm_pct = 0.0
    spec_shared.toolhead.shared_heatsink_cpap_duct = True

    rep_hot = simulate_chamber_and_cpap(spec_shared, hotend_target_temp_c=250.0)
    assert rep_hot.kalico_shared_duct_active is True
    assert rep_hot.cpap_pwm_pct == 0.0
    assert rep_hot.cpap_effective_pwm_pct >= 25.0
    assert rep_hot.cpap_delivered_dm3_s > 0.0
    assert any("kalico" in note.lower() for note in rep_hot.cpap_cooling_notes)

    # Below 50 C hotend threshold, heater_fan floor is inactive
    rep_cold = simulate_chamber_and_cpap(spec_shared, hotend_target_temp_c=40.0)
    assert rep_cold.kalico_shared_duct_active is False
    assert rep_cold.cpap_effective_pwm_pct == 0.0
    assert rep_cold.cpap_delivered_dm3_s == 0.0

    # Shared duct active airflow also improves hotend cold-side cooling vs non-shared 0% PWM
    spec_unshared = spec_shared.model_copy(deep=True)
    spec_unshared.toolhead.shared_heatsink_cpap_duct = False
    th_shared = simulate_hotend_thermal(spec_shared, target_temp_c=260.0, filament_id="abs")
    th_unshared = simulate_hotend_thermal(spec_unshared, target_temp_c=260.0, filament_id="abs")
    assert th_shared.cold_side_temp_c < th_unshared.cold_side_temp_c
    assert th_unshared.chamber_cpap is not None
    assert any("0%" in n for n in th_unshared.chamber_cpap["cpap_cooling_notes"])


def test_monolith_50mm_hf_high_volumetric_flow_abs_benchmark() -> None:
    """Monolith 50mm HF preset delivers realistic high ABS volumetric flow (~83 mm3/s bench class)."""
    spec_mono = _enclosed_abs_spec(
        heater_id="monolith_500w_ptc",
        insulation_mm=20.0,
        cpap_blower_id="ws7040_24v",
        cpap_hose_id_mm=15.0,
        cpap_pwm_pct=40.0,
        shared_duct=True,
    )
    spec_mono.frame.chamber_temp_c = 85.0
    spec_mono.toolhead.hotend.preset_id = "monolith_50mm_hf"

    spec_rapido = spec_mono.model_copy(deep=True)
    spec_rapido.toolhead.hotend.preset_id = "phaetus_rapido_2_hf"

    th_mono_255 = simulate_hotend_thermal(spec_mono, target_temp_c=255.0, filament_id="abs")
    th_mono_300 = simulate_hotend_thermal(spec_mono, target_temp_c=300.0, filament_id="abs")
    th_rapido_300 = simulate_hotend_thermal(spec_rapido, target_temp_c=300.0, filament_id="abs")

    # Standard ABS @ 255 C / 85 C chamber sits right in Monolith's 60-95 mm3/s high-flow regime (~87 mm3/s)
    assert 60.0 <= th_mono_255.max_volumetric_flow_mm3_s <= 95.0
    # At 300 C / 85 C chamber, Monolith 50mm HF pushes well past Rapido 2 HF and hits the 80W heater ceiling
    assert th_mono_300.max_volumetric_flow_mm3_s > th_rapido_300.max_volumetric_flow_mm3_s * 1.5
    assert th_mono_300.max_volumetric_flow_mm3_s >= 80.0
    assert th_mono_300.flow_limiting_factor == "heater_power"


def test_hotend_integration_cpap_overhang_and_legacy_backward_compat() -> None:
    """CPAP airflow drives overhang/bridging when enabled; legacy presets stay 100% identical when none."""
    base = PrinterSpec()
    assert base.toolhead.cpap_blower_id == "none"
    th_base = simulate_hotend_thermal(base, filament_id="pla")
    payload = th_base.to_dict()
    assert "chamber_cpap" in payload
    assert payload["chamber_cpap"] is not None
    assert payload["chamber_cpap"]["cpap_blower_id"] == "none"

    # Stronger CPAP blower improves PLA overhang angle and reduces bridge sag
    s_ws = base.model_copy(deep=True)
    s_ws.toolhead.cpap_blower_id = "ws7040_24v"
    s_ws.toolhead.cpap_hose_id_mm = 15.0
    s_ws.toolhead.cpap_pwm_pct = 80.0

    s_wm = s_ws.model_copy(deep=True)
    s_wm.toolhead.cpap_blower_id = "wm7575_220vac"

    th_ws = simulate_hotend_thermal(s_ws, filament_id="pla")
    th_wm = simulate_hotend_thermal(s_wm, filament_id="pla")
    assert th_wm.max_overhang_deg > th_ws.max_overhang_deg
    assert th_wm.bridge_sag_40mm_mm < th_ws.bridge_sag_40mm_mm

    # Verify ChamberAndCpapReport.to_dict() exposes all required contract keys
    expected_keys = {
        "chamber_heater_id",
        "chamber_heater_power_w",
        "chamber_insulation_mm",
        "enclosure_ua_w_per_k",
        "chamber_steady_temp_c",
        "chamber_heatup_time_60c_s",
        "cpap_blower_id",
        "cpap_hose_id_mm",
        "cpap_pwm_pct",
        "cpap_effective_pwm_pct",
        "cpap_hose_flow_factor",
        "cpap_delivered_dm3_s",
        "cpap_delivered_cfm",
        "cpap_exhaust_temp_rise_c",
        "cpap_outlet_air_temp_c",
        "cpap_hose_moving_mass_g",
        "kalico_shared_duct_active",
        "cpap_cooling_notes",
    }
    assert expected_keys.issubset(payload["chamber_cpap"].keys())
