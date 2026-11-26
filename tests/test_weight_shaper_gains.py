"""Unit tests for coupled limiting-shaper weight-reduction gains (`weight_shaper_gains`)."""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import time

import pytest

from printsim.engine.evaluator import compute_weight_shaper_gains, evaluate_printer
from printsim.schema import PrinterSpec

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"

_REQUIRED_TOP_KEYS = {
    "limiting_axis",
    "limiting_shaper_hz",
    "limiting_max_accel_mm_s2",
    "limiting_moving_mass_g",
    "total_potential_cut_g",
    "total_potential_hz_gain",
    "total_potential_accel_gain_mm_s2",
    "items",
}

_REQUIRED_ITEM_KEYS = {
    "id",
    "label",
    "current_mass_g",
    "example_cut_g",
    "hz_per_100g",
    "est_hz_gain",
    "est_accel_gain_mm_s2",
    "affects_axis",
    "note",
}


@pytest.mark.parametrize(
    "preset_name,expected_ids",
    [
        (
            "corexy_voron_style.yaml",
            {"hotend", "toolhead", "x_beam", "x_rail_backer", "y_carriages", "y_gantry_total"},
        ),
        (
            "cartesian_bedslinger.yaml",
            {"hotend", "toolhead", "moving_bed", "x_beam"},
        ),
        (
            "delta_racer.yaml",
            {"hotend", "toolhead", "y_carriages", "y_gantry_total"},
        ),
    ],
)
def test_weight_shaper_gains_schema_and_serialization(
    preset_name: str, expected_ids: set[str]
) -> None:
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / preset_name)
    report = evaluate_printer(spec)
    wsg = report.weight_shaper_gains
    assert isinstance(wsg, dict)
    assert set(wsg.keys()) == _REQUIRED_TOP_KEYS
    assert report.to_dict(include_toolpath_segments=False)["weight_shaper_gains"] == wsg

    assert wsg["limiting_axis"] in ("X", "Y")
    for k in (
        "limiting_shaper_hz",
        "limiting_max_accel_mm_s2",
        "limiting_moving_mass_g",
        "total_potential_cut_g",
        "total_potential_hz_gain",
        "total_potential_accel_gain_mm_s2",
    ):
        val = wsg[k]
        assert isinstance(val, (int, float))
        assert math.isfinite(val)
        assert val >= 0.0

    assert wsg["total_potential_cut_g"] == pytest.approx(
        round(0.25 * wsg["limiting_moving_mass_g"], 1), abs=0.2
    )
    assert wsg["total_potential_hz_gain"] > 0.0
    assert wsg["total_potential_accel_gain_mm_s2"] > 0.0

    items = wsg["items"]
    assert isinstance(items, list)
    assert {item["id"] for item in items} == expected_ids

    for item in items:
        assert set(item.keys()) == _REQUIRED_ITEM_KEYS
        assert isinstance(item["id"], str) and item["id"]
        assert isinstance(item["label"], str) and item["label"]
        assert item["affects_axis"] in ("X", "Y", "XY")
        assert isinstance(item["note"], str) and len(item["note"]) > 10
        for nk in (
            "current_mass_g",
            "example_cut_g",
            "hz_per_100g",
            "est_hz_gain",
            "est_accel_gain_mm_s2",
        ):
            v = item[nk]
            assert isinstance(v, (int, float))
            assert math.isfinite(v)
            assert v >= 0.0
        assert item["example_cut_g"] <= 0.41 * item["current_mass_g"] + 0.1


def test_corexy_y_limiting_gains_positive_and_monotonic_with_cut() -> None:
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    report = evaluate_printer(spec)
    wsg = report.weight_shaper_gains
    assert wsg is not None
    assert wsg["limiting_axis"] == "Y"

    by_id = {item["id"]: item for item in wsg["items"]}
    assert by_id["hotend"]["current_mass_g"] == pytest.approx(
        round(spec.toolhead.hotend.mass_g, 1), abs=0.1
    )
    for item_id in ("y_gantry_total", "toolhead", "x_beam", "x_rail_backer", "y_carriages", "hotend"):
        assert by_id[item_id]["est_hz_gain"] > 0.0
        assert by_id[item_id]["est_accel_gain_mm_s2"] > 0.0
        assert by_id[item_id]["hz_per_100g"] > 0.0

    # Total potential matches 25% cut of Y moving gantry
    assert wsg["total_potential_cut_g"] == by_id["y_gantry_total"]["example_cut_g"]
    assert wsg["total_potential_hz_gain"] == by_id["y_gantry_total"]["est_hz_gain"]
    assert wsg["total_potential_accel_gain_mm_s2"] == by_id["y_gantry_total"]["est_accel_gain_mm_s2"]

    # Larger cuts across moving Y components yield strictly larger frequency and accel gains
    ordered = [
        by_id["hotend"],
        by_id["y_carriages"],
        by_id["x_beam"],
        by_id["toolhead"],
        by_id["y_gantry_total"],
    ]
    for smaller, larger in zip(ordered, ordered[1:]):
        assert larger["example_cut_g"] > smaller["example_cut_g"]
        assert larger["est_hz_gain"] > smaller["est_hz_gain"]
        assert larger["est_accel_gain_mm_s2"] > smaller["est_accel_gain_mm_s2"]


def test_coupled_gain_matches_analytical_series_spring_formula() -> None:
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    report = evaluate_printer(spec)
    wsg = report.weight_shaper_gains
    assert wsg is not None
    lim = report.modal_fea.shaper_y
    assert lim is not None

    m_lim_g = report.assembly.y_moving_mass_kg * 1000.0
    cut_g = round(0.25 * m_lim_g, 1)
    mass_boost = math.sqrt(m_lim_g / (m_lim_g - cut_g))
    f_drive_new = lim.drive_freq_hz * mass_boost
    # B-01 consistent moving-mass compliance split: 1/f_eff^2 = 1/f_struct_m^2 + 1/f_drive^2
    inv_f_str_m2 = max(1.0 / (lim.effective_freq_hz**2) - 1.0 / (lim.drive_freq_hz**2), 1e-12)
    f_struct_m = 1.0 / math.sqrt(inv_f_str_m2)
    f_eff_drive_only = 1.0 / math.sqrt(1.0 / (f_struct_m**2) + 1.0 / (f_drive_new**2))
    f_eff_full_cap = lim.effective_freq_hz * mass_boost
    df_min = f_eff_drive_only - lim.effective_freq_hz
    df_max = f_eff_full_cap - lim.effective_freq_hz
    assert df_min - 0.02 <= wsg["total_potential_hz_gain"] <= df_max + 0.02


def test_bedslinger_y_limiting_moving_bed_dominates() -> None:
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "cartesian_bedslinger.yaml")
    report = evaluate_printer(spec)
    wsg = report.weight_shaper_gains
    assert wsg is not None
    assert wsg["limiting_axis"] == "Y"

    by_id = {item["id"]: item for item in wsg["items"]}
    bed = by_id["moving_bed"]
    assert bed["affects_axis"] == "Y"
    assert bed["est_hz_gain"] > 1.0
    assert bed["est_accel_gain_mm_s2"] > 100.0

    for other_id in ("hotend", "toolhead", "x_beam"):
        assert bed["est_hz_gain"] > by_id[other_id]["est_hz_gain"]
        assert bed["est_accel_gain_mm_s2"] > by_id[other_id]["est_accel_gain_mm_s2"]
        # Stationary in Y on a bedslinger -> 0 Y-shaper gain when Y is limiting
        assert by_id[other_id]["est_hz_gain"] == 0.0


def test_x_limiting_tie_break_and_no_sensitivity_fallback() -> None:
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    report = evaluate_printer(spec, include_advanced_analysis=False)
    assert report.sensitivity is None
    assert report.weight_shaper_gains is not None
    assert report.weight_shaper_gains["total_potential_hz_gain"] > 0.0

    # Force X to be limiting via tie-break on max_accel_mm_s2 with consistently lower effective_freq_hz
    sx = report.modal_fea.shaper_x
    sy = report.modal_fea.shaper_y
    assert sx is not None and sy is not None
    scale = 0.55
    sx_tied = replace(
        sx,
        max_accel_mm_s2=sy.max_accel_mm_s2,
        structural_freq_hz=sx.structural_freq_hz * scale,
        drive_freq_hz=sx.drive_freq_hz * scale,
        effective_freq_hz=sx.effective_freq_hz * scale,
    )
    modal_x_lim = replace(report.modal_fea, shaper_x=sx_tied)
    wsg_x = compute_weight_shaper_gains(report.assembly, modal_x_lim, None)
    assert wsg_x["limiting_axis"] == "X"
    assert wsg_x["limiting_moving_mass_g"] == pytest.approx(
        round(report.assembly.x_moving_mass_kg * 1000.0, 1), abs=0.1
    )
    by_id_x = {item["id"]: item for item in wsg_x["items"]}
    assert by_id_x["toolhead"]["est_hz_gain"] > 0.0
    assert by_id_x["y_gantry_total"]["est_hz_gain"] == 0.0

    # When shaper_x is None, defaults cleanly to "Y"
    modal_no_x = replace(report.modal_fea, shaper_x=None)
    wsg_default_y = compute_weight_shaper_gains(report.assembly, modal_no_x, None)
    assert wsg_default_y["limiting_axis"] == "Y"


@pytest.mark.parametrize(
    "preset_name",
    ["corexy_voron_style.yaml", "cartesian_bedslinger.yaml", "delta_racer.yaml"],
)
def test_evaluate_printer_runtime_stays_fast(preset_name: str) -> None:
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / preset_name)
    t0 = time.perf_counter()
    report = evaluate_printer(spec)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert report.weight_shaper_gains is not None
    import os

    budget_ms = 1050.0 if os.environ.get("CI") else 450.0
    assert elapsed_ms < budget_ms


def test_weight_shaper_gains_uses_exact_hotend_mass_g() -> None:
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    spec.toolhead.hotend.mass_g = 96.0
    report = evaluate_printer(spec, include_advanced_analysis=False)
    wsg = report.weight_shaper_gains
    assert wsg is not None
    by_id = {item["id"]: item for item in wsg["items"]}
    assert by_id["hotend"]["current_mass_g"] == pytest.approx(96.0, abs=0.1)
    assert by_id["hotend"]["example_cut_g"] == pytest.approx(25.0, abs=0.1)
