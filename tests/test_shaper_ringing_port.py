"""Unit tests for shaper-sim & ringing-sim physics ports into PrintSim.

Covers:
1. Exact 2nd-order step response & Klipper shaper impulse convolution (`klipper_shaper.py`).
2. Speed-dependent VFA belt-mesh & stepper electrical forcing harmonics (`klipper_shaper.py`).
3. Carriage ADXL vs. nozzle-tip PSD divergence diagnostic (`modal.py`).
4. Separated 90° corner geometric error vs. tracking lag, +Y wall profile, and
   Multi-Segment Zig-Zag S-Curve Benchmark C (`ballbar.py`).
5. Reference ADXL CSV fixtures (`tests/fixtures/resonances_{x,y}.csv`) via `parse_adxl_csv`.
6. Datasheet-verified 0.9° stepper motors (`ldo_09deg_2004mah`, `moons_09deg_ha2p4200`).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from printsim.fea.adxl_calibration import parse_adxl_csv
from printsim.fea.klipper_shaper import (
    calibrate_axis_shapers,
    compute_step_response,
    compute_vfa_forcing_harmonics,
)
from printsim.fea.modal import solve_modal_fea
from printsim.gcode.ballbar import simulate_time_domain_metrology
from printsim.geometry.builder import build_assembly
from printsim.schema import PrinterSpec
from printsim.schema.printer import STEPPER_CATALOG, StepperSpec

FIXTURES_DIR = Path(__file__).parent / "fixtures"
VORON_YAML = Path(__file__).resolve().parents[1] / "examples" / "corexy_voron_style.yaml"


# ---------------------------------------------------------------------------
# 1. Analytical Step Response & Shaper Impulse Convolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("zeta", [0.05, 0.10, 0.15, 0.25])
def test_step_response_matches_exact_analytical_overshoot(zeta: float) -> None:
    """Unshaped 2nd-order step response must match M_p = exp(-pi*zeta / sqrt(1 - zeta^2))."""
    freq_hz = 50.0
    exact_mp_pct = 100.0 * math.exp(-math.pi * zeta / math.sqrt(1.0 - zeta * zeta))

    # Fine grid resolves exact peak time t_p = 1 / (2 * f_n * sqrt(1 - zeta^2))
    res = compute_step_response(
        freq_hz=freq_hz,
        damping_ratio=zeta,
        shaper_name="NONE",
        shaper_freq_hz=freq_hz,
        t_max_s=0.16,
        n_pts=801,
    )
    assert res["overshoot_unshaped_pct"] == pytest.approx(exact_mp_pct, abs=0.15)
    assert res["overshoot_shaped_pct"] == pytest.approx(res["overshoot_unshaped_pct"], abs=1e-6)
    assert res["settle_time_shaped_ms"] == pytest.approx(res["settle_time_unshaped_ms"], abs=1e-6)
    assert len(res["time_ms"]) == 801
    assert res["unshaped"][0] == pytest.approx(0.0, abs=1e-6)
    assert res["shaped"] == res["unshaped"]


def test_matched_zv_shaper_achieves_zero_overshoot_and_cancellation() -> None:
    """Exact matched ZV shaper eliminates overshoot and all post-T2 residual oscillation."""
    f_n = 50.0
    zeta = 0.08
    res = compute_step_response(
        freq_hz=f_n,
        damping_ratio=zeta,
        shaper_name="ZV",
        shaper_freq_hz=f_n,
        shaper_damping_ratio=zeta,
        t_max_s=0.16,
        n_pts=321,
    )
    exact_mp_pct = 100.0 * math.exp(-math.pi * zeta / math.sqrt(1.0 - zeta * zeta))
    assert res["overshoot_unshaped_pct"] == pytest.approx(exact_mp_pct, abs=0.25)
    assert res["overshoot_shaped_pct"] < 0.05

    t_half_cycle_ms = 500.0 / (f_n * math.sqrt(1.0 - zeta * zeta))
    t_arr = np.asarray(res["time_ms"], dtype=float)
    y_sh = np.asarray(res["shaped"], dtype=float)
    post = t_arr >= t_half_cycle_ms + 0.5
    assert np.max(np.abs(y_sh[post] - 1.0)) < 1e-4
    assert res["settle_time_shaped_ms"] < res["settle_time_unshaped_ms"] * 0.25


@pytest.mark.parametrize(
    "shaper_name",
    ["NONE", "ZV", "ZVD", "MZV", "EI", "2HUMP_EI", "3HUMP_EI", "SMOOTH_FLATNESS"],
)
def test_step_response_supports_all_shaper_types_and_graph_serialization(shaper_name: str) -> None:
    """Every supported shaper convolves cleanly and reaches steady-state ~1.0."""
    res = compute_step_response(
        freq_hz=55.0,
        damping_ratio=0.07,
        shaper_name=shaper_name,
        shaper_freq_hz=55.0,
        t_max_s=0.16,
        n_pts=161,
    )
    assert len(res["time_ms"]) == 161
    assert len(res["unshaped"]) == 161
    assert len(res["shaped"]) == 161
    assert res["shaped"][-1] == pytest.approx(1.0, abs=0.05)
    if shaper_name != "NONE":
        assert res["overshoot_shaped_pct"] < res["overshoot_unshaped_pct"]

    graph = calibrate_axis_shapers("X", primary_freq_hz=55.0, shaper_choice=shaper_name)
    assert graph.step_response is not None
    payload = graph.to_dict()
    assert "step_response" in payload
    assert payload["step_response"]["overshoot_unshaped_pct"] > 0.0
    assert len(payload["step_response"]["time_ms"]) == 161


# ---------------------------------------------------------------------------
# 2. VFA Forcing Harmonics (Speed-Dependent Diagnostic)
# ---------------------------------------------------------------------------


def test_vfa_forcing_harmonics_gt2_18deg_vs_09deg() -> None:
    """Verify exact belt-mesh and stepper electrical/full-step harmonics for 1.8° vs. 0.9°."""
    v = 120.0
    h_18 = compute_vfa_forcing_harmonics(
        speed_mm_s=v, pulley_teeth=20, belt_pitch_mm=2.0, steps_per_rev=200
    )
    h_09 = compute_vfa_forcing_harmonics(
        speed_mm_s=v, pulley_teeth=20, belt_pitch_mm=2.0, steps_per_rev=400
    )

    # GT2 2 mm belt mesh
    assert h_18["belt_mesh_hz"] == pytest.approx(60.0, rel=1e-6)
    assert h_18["belt_mesh_wavelength_mm"] == pytest.approx(2.0, rel=1e-6)
    assert h_09["belt_mesh_hz"] == pytest.approx(60.0, rel=1e-6)

    # 20T × 2 mm = 40 mm/rev => 120 / 40 = 3.0 rev/s
    assert h_18["motor_rps_hz"] == pytest.approx(3.0, rel=1e-6)
    assert h_18["rotor_teeth"] == pytest.approx(50.0, rel=1e-6)
    assert h_18["motor_elec_hz"] == pytest.approx(150.0, rel=1e-6)
    assert h_18["vfa_step_hz"] == pytest.approx(600.0, rel=1e-6)
    assert h_18["motor_full_step_wavelength_mm"] == pytest.approx(0.2, rel=1e-6)
    assert h_18["motor_elec_wavelength_mm"] == pytest.approx(0.8, rel=1e-6)

    # 0.9° motor doubles rotor teeth & forcing frequencies and halves spatial wavelengths
    assert h_09["rotor_teeth"] == pytest.approx(100.0, rel=1e-6)
    assert h_09["motor_elec_hz"] == pytest.approx(2.0 * h_18["motor_elec_hz"], rel=1e-6)
    assert h_09["vfa_step_hz"] == pytest.approx(2.0 * h_18["vfa_step_hz"], rel=1e-6)
    assert h_09["motor_full_step_wavelength_mm"] == pytest.approx(0.1, rel=1e-6)
    assert h_09["motor_elec_wavelength_mm"] == pytest.approx(0.4, rel=1e-6)


# ---------------------------------------------------------------------------
# 3. Separated Corner Metrics, Wall Profile & Zig-Zag S-Curve (ballbar.py)
# ---------------------------------------------------------------------------


def test_separated_corner_metrics_wall_profile_and_zigzag_benchmark() -> None:
    """Verify separated corner geometric vs. lag metrics, wall profile, and Zig-Zag Benchmark C."""
    spec = PrinterSpec.from_yaml(VORON_YAML)
    asm = build_assembly(spec)
    modal = solve_modal_fea(asm)

    res_sh = simulate_time_domain_metrology(asm, modal)
    res_none = simulate_time_domain_metrology(asm, modal, shaper_name="NONE")

    # NONE has zero shaper path smoothing
    assert res_none.shaper_path_smoothing_um == pytest.approx(0.0, abs=1e-9)
    assert res_none.zigzag_shaper_smoothing_um == pytest.approx(0.0, abs=1e-9)
    assert res_none.printed_path_error_um > 0.0

    # Shaped corner: geometric path error < along-track tracking lag
    assert res_sh.shaper_path_smoothing_um > 0.0
    assert res_sh.printed_path_error_um > 0.0
    assert res_sh.cross_track_error_um > 0.0
    assert res_sh.along_track_lag_um > 0.0
    assert res_sh.printed_path_error_um < res_sh.along_track_lag_um

    # Enriched corner_trace arrays
    tr = res_sh.corner_trace
    n_tr = len(tr["ideal_x_mm"])
    assert len(tr["shaped_cmd_x_mm"]) == n_tr
    assert len(tr["shaped_cmd_y_mm"]) == n_tr
    assert len(tr["wall_dist_mm"]) == 120
    assert tr["wall_dist_mm"][0] == pytest.approx(0.0, abs=1e-6)
    assert tr["wall_dist_mm"][-1] == pytest.approx(25.0, abs=1e-6)
    assert len(tr["wall_dev_unshaped_um"]) == 120
    assert len(tr["wall_dev_shaped_um"]) == 120

    # Shaped wall surface beyond initial corner transition has lower peak-to-peak ripple
    wall_d = np.asarray(tr["wall_dist_mm"], dtype=float)
    post_mask = wall_d >= 5.0
    un_ptp = float(np.ptp(np.asarray(tr["wall_dev_unshaped_um"])[post_mask]))
    sh_ptp = float(np.ptp(np.asarray(tr["wall_dev_shaped_um"])[post_mask]))
    assert sh_ptp < un_ptp

    # Benchmark C: Multi-Segment Zig-Zag S-Curve
    assert res_sh.zigzag_path_error_um > 0.0
    assert res_sh.zigzag_shaper_smoothing_um > 0.0
    assert res_sh.zigzag_unshaped_ringing_um > 0.0
    assert res_sh.zigzag_residual_ringing_um < res_sh.zigzag_unshaped_ringing_um

    zz = res_sh.zigzag_trace
    required_zz_keys = (
        "ideal_x_mm",
        "ideal_y_mm",
        "actual_x_mm",
        "actual_y_mm",
        "unshaped_x_mm",
        "unshaped_y_mm",
        "perp_error_um",
    )
    for key in required_zz_keys:
        assert key in zz
    n_zz = len(zz["ideal_x_mm"])
    assert 80 <= n_zz <= 120
    for key in required_zz_keys:
        assert len(zz[key]) == n_zz
    assert min(zz["ideal_x_mm"]) >= -4.1
    assert max(zz["ideal_x_mm"]) <= 42.1
    assert max(zz["perp_error_um"]) > 0.0

    # Serialization includes all new fields
    d = res_sh.to_dict()
    for field_name in (
        "printed_path_error_um",
        "shaper_path_smoothing_um",
        "cross_track_error_um",
        "along_track_lag_um",
        "zigzag_path_error_um",
        "zigzag_shaper_smoothing_um",
        "zigzag_residual_ringing_um",
        "zigzag_unshaped_ringing_um",
        "zigzag_trace",
    ):
        assert field_name in d


# ---------------------------------------------------------------------------
# 4. Carriage ADXL vs. Nozzle-Tip PSD Diagnostic (modal.py)
# ---------------------------------------------------------------------------


def test_adxl_carriage_vs_nozzle_diagnostic_flexible_vs_stiff_toolhead() -> None:
    """Flexible long-drop toolhead diverges strongly between carriage ADXL and nozzle tip."""
    # Flexible / long-drop printed toolhead
    spec_flex = PrinterSpec.from_yaml(VORON_YAML)
    spec_flex.toolhead.nozzle_offset_mm = (0.0, -30.0, -78.0)
    spec_flex.toolhead.cog_offset_mm = (0.0, -24.0, -32.0)
    spec_flex.toolhead.mount_material = "abs_printed"
    spec_flex.toolhead.hotend.mount_style = "groove_mount"
    modal_flex = solve_modal_fea(build_assembly(spec_flex))

    # Stiff short-drop CNC aluminum toolhead
    spec_stiff = PrinterSpec.from_yaml(VORON_YAML)
    spec_stiff.toolhead.nozzle_offset_mm = (0.0, -6.0, -24.0)
    spec_stiff.toolhead.cog_offset_mm = (0.0, -5.0, -10.0)
    spec_stiff.toolhead.mount_material = "aluminum_6061_t6"
    spec_stiff.toolhead.hotend.mount_style = "chube_6bolt_m3_18pcd"
    spec_stiff.toolhead.carriage_plate_thickness_mm = 10.0
    spec_stiff.toolhead.body_wall_thickness_mm = 8.0
    spec_stiff.toolhead.hotend_mount_thickness_mm = 12.0
    modal_stiff = solve_modal_fea(build_assembly(spec_stiff))

    diag_flex = modal_flex.adxl_vs_nozzle_diagnostic
    diag_stiff = modal_stiff.adxl_vs_nozzle_diagnostic
    assert diag_flex is not None and diag_stiff is not None
    assert modal_flex.to_dict()["adxl_vs_nozzle_diagnostic"] == diag_flex

    for axis in ("x", "y"):
        entry = diag_flex[axis]
        for key in (
            "carriage_peak_hz",
            "nozzle_peak_hz",
            "nozzle_to_carriage_ratio",
            "carriage_recommended_shaper",
            "carriage_recommended_freq_hz",
            "carriage_max_accel_mm_s2",
            "nozzle_recommended_shaper",
            "nozzle_recommended_freq_hz",
            "nozzle_max_accel_mm_s2",
            "carriage_raw_psd",
            "has_significant_divergence",
        ):
            assert key in entry
        graph = modal_flex.shaper_graph_x if axis == "x" else modal_flex.shaper_graph_y
        assert graph is not None
        assert len(entry["carriage_raw_psd"]) == len(graph.freqs_hz)

    # Flexible toolhead has larger nozzle-to-carriage amplification & flags divergence on Y
    assert diag_flex["y"]["has_significant_divergence"] is True
    assert diag_flex["y"]["nozzle_to_carriage_ratio"] > diag_stiff["y"]["nozzle_to_carriage_ratio"]
    assert diag_flex["y"]["carriage_peak_hz"] > diag_flex["y"]["nozzle_peak_hz"]
    assert diag_flex["y"]["carriage_peak_hz"] - diag_flex["y"]["nozzle_peak_hz"] > (
        diag_stiff["y"]["carriage_peak_hz"] - diag_stiff["y"]["nozzle_peak_hz"]
    )


# ---------------------------------------------------------------------------
# 5. Reference ADXL CSV Fixtures
# ---------------------------------------------------------------------------


def test_parse_adxl_csv_reference_fixtures() -> None:
    """Reference shaper-sim ADXL CSV fixtures parse cleanly with peaks at ~48.5 Hz (X) & ~38.2 Hz (Y)."""
    csv_x = (FIXTURES_DIR / "resonances_x.csv").read_text(encoding="utf-8")
    csv_y = (FIXTURES_DIR / "resonances_y.csv").read_text(encoding="utf-8")

    parsed_x = parse_adxl_csv(csv_x, axis="X")
    parsed_y = parse_adxl_csv(csv_y, axis="Y")

    peak_x = float(parsed_x.freqs_hz[int(np.argmax(parsed_x.psd))])
    peak_y = float(parsed_y.freqs_hz[int(np.argmax(parsed_y.psd))])
    assert peak_x == pytest.approx(48.5, abs=0.2)
    assert peak_y == pytest.approx(38.2, abs=0.2)

    graph_x = calibrate_axis_shapers("X", primary_freq_hz=peak_x, freqs_hz=parsed_x.freqs_hz, raw_psd=parsed_x.psd)
    graph_y = calibrate_axis_shapers("Y", primary_freq_hz=peak_y, freqs_hz=parsed_y.freqs_hz, raw_psd=parsed_y.psd)
    assert graph_x.step_response is not None
    assert graph_y.step_response is not None
    assert graph_x.recommended_max_accel_mm_s2 > 0.0
    assert graph_y.recommended_max_accel_mm_s2 > 0.0


# ---------------------------------------------------------------------------
# 6. Datasheet-Verified 0.9° Stepper Motors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("motor_id", "expected_torque", "expected_r", "expected_l", "expected_j", "expected_mass", "expected_class"),
    [
        ("ldo_09deg_2004mah", 0.40, 1.45, 2.0, 7.1e-6, 350.0, "H"),
        ("moons_09deg_ha2p4200", 0.39, 1.05, 2.8, 5.7e-6, 280.0, "B"),
    ],
)
def test_new_09deg_motors_instantiate_and_have_2x_magnetic_stiffness(
    motor_id: str,
    expected_torque: float,
    expected_r: float,
    expected_l: float,
    expected_j: float,
    expected_mass: float,
    expected_class: str,
) -> None:
    """New 0.9° datasheet motors instantiate cleanly and show 2× magnetic stiffness factor."""
    assert motor_id in STEPPER_CATALOG
    raw = STEPPER_CATALOG[motor_id]
    motor = StepperSpec(**{k: v for k, v in raw.items() if k in StepperSpec.model_fields})

    assert motor.steps_per_rev == 400
    assert motor.pole_pairs == pytest.approx(100.0)
    assert motor.holding_torque_nm == pytest.approx(expected_torque, rel=1e-4)
    assert motor.phase_resistance_ohm == pytest.approx(expected_r, rel=1e-4)
    assert motor.inductance_mh == pytest.approx(expected_l, rel=1e-4)
    assert motor.rotor_inertia_kg_m2 == pytest.approx(expected_j, rel=1e-4)
    assert motor.motor_mass_g == pytest.approx(expected_mass, rel=1e-4)
    assert motor.insulation_class == expected_class

    # Compare torsional stiffness at near-standstill (10 RPM) against identical 1.8° twin
    motor_18_twin = motor.model_copy(update={"steps_per_rev": 200})
    assert motor_18_twin.pole_pairs == pytest.approx(50.0)
    ratio = motor.torsional_stiffness_nm_per_rad(rpm=10.0) / motor_18_twin.torsional_stiffness_nm_per_rad(rpm=10.0)
    assert ratio == pytest.approx(2.0, rel=0.02)

    # Full printer assembly + modal solve executes cleanly with the new motor
    spec = PrinterSpec.from_yaml(VORON_YAML)
    spec.kinematics.xy_stepper = motor
    modal = solve_modal_fea(build_assembly(spec))
    assert modal.fundamental_freq_hz > 20.0
    assert modal.recommended_max_accel_mm_s2 > 1000.0
