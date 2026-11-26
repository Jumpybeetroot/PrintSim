"""Analytical and integration tests for Klipper input-shaper physics and ADXL345 PSD calibration."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from printsim.engine import evaluate_printer
from printsim.fea import (
    SHAPER_NAMES,
    calibrate_axis_shapers,
    estimate_shaper,
    find_shaper_max_accel,
    get_shaper_pulses,
    get_shaper_smoothing,
    synthesize_axis_psd,
)
from printsim.schema import PrinterSpec

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def test_klipper_shaper_zero_vibration_and_envelope() -> None:
    """ZV, MZV, and 2HUMP_EI produce < 1e-6 residual vibration at exact design frequency."""
    f0 = 62.5
    zeta = 0.1
    freqs = np.linspace(10.0, 220.0, 211)

    for name in ("ZV", "MZV", "2HUMP_EI"):
        shaper = get_shaper_pulses(name, f0, damping_ratio=zeta)
        v_exact = float(estimate_shaper(shaper, np.array([f0]), test_damping_ratio=zeta)[0])
        assert v_exact < 1e-6, f"{name} residual vibration at design freq was {v_exact}"

    for name in SHAPER_NAMES:
        shaper = get_shaper_pulses(name, f0, damping_ratio=zeta)
        v_curve = estimate_shaper(shaper, freqs, test_damping_ratio=zeta)
        assert np.all(v_curve >= 0.0)
        assert np.all(v_curve <= 1.0 + 1e-12)

    # EI & 2HUMP_EI maintain <= v_tol (0.05) across a wide notch around f0
    band = np.linspace(0.9 * f0, 1.1 * f0, 25)
    for name in ("EI", "2HUMP_EI", "3HUMP_EI"):
        shaper = get_shaper_pulses(name, f0, damping_ratio=zeta)
        v_band = estimate_shaper(shaper, band, test_damping_ratio=zeta)
        assert float(np.max(v_band)) <= 0.055


def test_klipper_smoothing_and_max_accel_monotonicity() -> None:
    """At identical frequency, ZV > MZV > EI > 2HUMP_EI > 3HUMP_EI in max_accel (and inverse in smoothing)."""
    for f0 in (45.0, 85.0, 140.0):
        smoothings = []
        accels = []
        for name in SHAPER_NAMES:
            shaper = get_shaper_pulses(name, f0, damping_ratio=0.1)
            sm = get_shaper_smoothing(shaper, accel=5000.0, scv=5.0)
            ma = find_shaper_max_accel(shaper, scv=5.0, target_smoothing=0.12)
            assert ma % 100.0 == 0.0
            assert get_shaper_smoothing(shaper, accel=max(ma - 50.0, 100.0), scv=5.0) <= 0.12 + 1e-6
            smoothings.append(sm)
            accels.append(ma)

        for idx in range(len(SHAPER_NAMES) - 1):
            assert smoothings[idx] < smoothings[idx + 1]
            assert accels[idx] > accels[idx + 1]


def test_klipper_quadratic_frequency_scaling() -> None:
    """Doubling resonance frequency 50 Hz -> 100 Hz quadruples (4.0x) Klipper max_accel."""
    for name in ("ZV", "MZV", "EI", "2HUMP_EI"):
        a_50 = find_shaper_max_accel(get_shaper_pulses(name, 50.0), scv=5.0)
        a_100 = find_shaper_max_accel(get_shaper_pulses(name, 100.0), scv=5.0)
        ratio = a_100 / a_50
        assert ratio == pytest.approx(4.0, rel=0.03), f"{name}: 50Hz={a_50}, 100Hz={a_100}, ratio={ratio}"


def test_multipeak_psd_robustness_ei_mzv_beat_zv() -> None:
    """On a close dual-peak PSD (55 Hz + 68 Hz), MZV and EI achieve lower vibration than ZV."""
    freqs, psd = synthesize_axis_psd(
        axis="X",
        primary_freq_hz=55.0,
        secondary_modes=[(68.0, 1.0)],
    )
    assert len(freqs) == 108
    assert float(np.max(psd)) == pytest.approx(100.0, rel=1e-9)

    graph = calibrate_axis_shapers(
        axis="X",
        primary_freq_hz=55.0,
        secondary_modes=[(68.0, 1.0)],
        scv=5.0,
    )
    by_name = {c.name: c for c in graph.candidates}
    assert set(by_name.keys()) == set(SHAPER_NAMES)
    assert by_name["MZV"].vibrations_pct < by_name["ZV"].vibrations_pct
    assert by_name["EI"].vibrations_pct < by_name["ZV"].vibrations_pct
    assert graph.recommended_shaper in ("MZV", "EI", "2HUMP_EI")


def test_monolith_high_freq_above_150hz_and_explicit_choice() -> None:
    """220 Hz PSD span calibrates 175 Hz AWD resonances without clipping at 150 Hz."""
    graph_auto = calibrate_axis_shapers(
        axis="Y",
        primary_freq_hz=175.0,
        scv=8.0,
        shaper_choice="auto",
    )
    assert float(graph_auto.freqs_hz[-1]) == pytest.approx(220.0, rel=1e-6)
    assert graph_auto.peak_freq_hz > 165.0
    assert graph_auto.recommended_freq_hz > 160.0
    assert graph_auto.recommended_max_accel_mm_s2 > 50000.0

    for choice in ("zv", "mzv", "ei", "2hump_ei"):
        g_exp = calibrate_axis_shapers(
            axis="X",
            primary_freq_hz=72.0,
            scv=5.0,
            shaper_choice=choice,
        )
        assert g_exp.recommended_shaper == choice.upper()
        matched = next(c for c in g_exp.candidates if c.name == choice.upper())
        assert g_exp.recommended_max_accel_mm_s2 == matched.max_accel_mm_s2

    g_none = calibrate_axis_shapers(axis="X", primary_freq_hz=72.0, scv=5.0, shaper_choice="none")
    assert g_none.recommended_shaper == "NONE"
    assert np.allclose(g_none.shaped_psd, g_none.raw_psd)


def test_evaluate_printer_shaper_graphs_schema_and_speed() -> None:
    """evaluate_printer emits locked shaper_graphs schema with 108 bins up to 220 Hz and <35ms runtime."""
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_monolith_awd.yaml")
    rep = evaluate_printer(spec)
    modal_dict = rep.modal_fea.to_dict()

    assert "shaper_graphs" in modal_dict
    graphs = modal_dict["shaper_graphs"]
    assert set(graphs.keys()) == {"x", "y"}

    expected_graph_keys = {
        "axis",
        "freqs_hz",
        "raw_psd",
        "shaped_psd",
        "peak_freq_hz",
        "damping_ratio",
        "recommended_shaper",
        "recommended_freq_hz",
        "recommended_max_accel_mm_s2",
        "candidates",
        "step_response",
    }
    expected_cand_keys = {
        "name",
        "freq_hz",
        "vibrations_pct",
        "smoothing",
        "max_accel_mm_s2",
        "score",
        "attenuation_curve",
    }

    for axis_key, expected_axis in (("x", "X"), ("y", "Y")):
        g = graphs[axis_key]
        assert g is not None
        assert set(g.keys()) == expected_graph_keys
        assert g["axis"] == expected_axis
        assert len(g["freqs_hz"]) == 108
        assert g["freqs_hz"][0] == pytest.approx(5.0, abs=1e-6)
        assert g["freqs_hz"][-1] == pytest.approx(220.0, abs=1e-6)
        assert len(g["raw_psd"]) == 108
        assert len(g["shaped_psd"]) == 108
        assert max(g["raw_psd"]) == pytest.approx(100.0, abs=0.01)
        assert 0.01 < g["damping_ratio"] < 0.30
        assert len(g["candidates"]) == 5
        assert [c["name"] for c in g["candidates"]] == list(SHAPER_NAMES)
        for cand in g["candidates"]:
            assert set(cand.keys()) == expected_cand_keys
            assert len(cand["attenuation_curve"]) == 108
            assert 0.0 <= cand["vibrations_pct"] <= 100.0
            assert cand["max_accel_mm_s2"] >= 100.0

    # Calibration speed check (< 35 ms for both X and Y combined)
    t0 = time.perf_counter()
    calibrate_axis_shapers("X", 95.7, secondary_modes=[(209.5, 0.8)], scv=8.0)
    calibrate_axis_shapers("Y", 123.2, secondary_modes=[(180.0, 0.4)], scv=8.0)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    import os

    budget_ms = 110.0 if os.environ.get("CI") else 50.0
    assert elapsed_ms < budget_ms, f"Calibration took {elapsed_ms:.2f} ms (budget < {budget_ms:.0f} ms)"
