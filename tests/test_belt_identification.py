"""Checks for the belt identification method: string physics and separating belt from machine."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from printsim.fea import solve_modal_fea
from printsim.fea.belt_identification import (
    axis_compliance_m_per_n,
    belt_share_from_swap,
    belt_stiffness_ratio_from_tension_sweep,
    cord_geometry_factor_m,
    identify_belt_from_position_sweep,
    identify_belt_from_swap,
    pluck_frequency_hz,
    tension_from_pluck_n,
)
from printsim.geometry import build_assembly
from printsim.kinematics.solvers import compute_drive_stiffness_details_n_per_m
from printsim.schema import PrinterSpec

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def _spec(name: str = "voron_trident_300.yaml") -> PrinterSpec:
    return PrinterSpec.from_yaml(EXAMPLES_DIR / name)


# Monolith matched-tension 10 mm belt shootout: four belts set to ~20 lbf (89 N), plucked on a
# 150 mm span. Scale-weighed linear mass (g/m) and the measured pluck frequency (Hz).
_SHOOTOUT = (
    ("gates_gt2_epdm_fiberglass", 12.36, 283.0),
    ("gates_gt3_2mgt", 14.99, 257.0),
    ("rubber_aramid", 13.99, 266.0),
    ("tpu_kevlar", 12.88, 277.0),
)


@pytest.mark.parametrize(("name", "mass_g_per_m", "pluck_hz"), _SHOOTOUT)
def test_string_physics_reproduces_measured_pluck_at_matched_tension(
    name: str, mass_g_per_m: float, pluck_hz: float
) -> None:
    """f = (1 / 2L) sqrt(T / mu): four belts of different mass at one tension, within 0.5 Hz."""
    mu = mass_g_per_m * 1e-3
    assert pluck_frequency_hz(89.0, 0.150, mu) == pytest.approx(pluck_hz, abs=0.5)
    assert tension_from_pluck_n(pluck_hz, 0.150, mu) == pytest.approx(89.0, rel=0.005)


def test_pluck_tension_round_trip_with_bending_stiffness() -> None:
    mu, ei = 8.4e-3, 9.0e-6
    for tension in (5.0, 9.15, 28.0):
        for span in (0.08, 0.15, 0.30):
            f = pluck_frequency_hz(tension, span, mu, ei)
            assert tension_from_pluck_n(f, span, mu, ei) == pytest.approx(tension, rel=1e-9)
            # Bending stiffness makes a short span read high, never low
            assert f >= pluck_frequency_hz(tension, span, mu)
    # Voron guidance: 110 Hz on a 150 mm span of 6 mm 2GT is about 2 lb
    assert tension_from_pluck_n(110.0, 0.150, mu) == pytest.approx(9.15, rel=0.01)


def test_belt_swap_share_closed_form() -> None:
    """Synthetic two-spring chain: C = C0 + G / EA; doubling EA must return the planted share."""
    c0, belt_c, m = 4.0e-6, 6.0e-6, 0.5
    f_a = 1.0 / (2.0 * math.pi * math.sqrt(m * (c0 + belt_c)))
    f_b = 1.0 / (2.0 * math.pi * math.sqrt(m * (c0 + belt_c / 2.0)))
    assert belt_share_from_swap(f_a, f_b, 2.0) == pytest.approx(belt_c / (c0 + belt_c), rel=1e-9)
    assert axis_compliance_m_per_n(f_a, m) == pytest.approx(c0 + belt_c, rel=1e-9)
    # Published 6 mm -> 12 mm swap on one machine: 40.9 Hz -> 47.2 Hz puts half the compliance in the belt
    assert belt_share_from_swap(40.9, 47.2, 2.0) == pytest.approx(0.50, abs=0.01)
    with pytest.raises(ValueError):
        belt_share_from_swap(40.0, 40.0, 1.0)


def test_tension_sweep_ratio_closed_form() -> None:
    c0, g_over_ea_hi, m = 5.0e-6, 5.0e-6, 0.5
    ratio = 0.8  # EA at low tension is 80 % of EA at high tension
    f_hi = 1.0 / (2.0 * math.pi * math.sqrt(m * (c0 + g_over_ea_hi)))
    f_lo = 1.0 / (2.0 * math.pi * math.sqrt(m * (c0 + g_over_ea_hi / ratio)))
    share_hi = g_over_ea_hi / (c0 + g_over_ea_hi)
    assert belt_stiffness_ratio_from_tension_sweep(f_lo, f_hi, share_hi) == pytest.approx(ratio, rel=1e-9)


def test_position_sweep_recovers_planted_belt_stiffness_exactly() -> None:
    """Frequencies generated from C0 + G(u, v) / EA return that EA and C0."""
    asm = build_assembly(_spec())
    m = asm.x_moving_mass_kg
    ea_true, c0_true = 21000.0, 5.5e-6
    pts = []
    for u, v in ((0.5, 0.5), (0.5, 0.1), (0.5, 0.9), (0.2, 0.3), (0.8, 0.7)):
        g = cord_geometry_factor_m(asm, "X", u, v)
        f = 1.0 / (2.0 * math.pi * math.sqrt(m * (c0_true + g / ea_true)))
        pts.append((u, v, f))
    res = identify_belt_from_position_sweep(asm, "X", pts)
    assert res.ea_in_place_n == pytest.approx(ea_true, rel=1e-6)
    assert res.other_compliance_um_per_n == pytest.approx(c0_true * 1e6, rel=1e-6)
    assert res.r_squared == pytest.approx(1.0, abs=1e-9)
    assert res.ea_over_datasheet_static == pytest.approx(ea_true / 16900.0, rel=1e-6)
    with pytest.raises(ValueError):
        identify_belt_from_position_sweep(asm, "X", pts[:2])

    swap = identify_belt_from_swap(asm, "X", pts[0][2], pts[0][2] * 1.05, 2.0)
    assert 0.0 < swap.belt_share_pct < 100.0


def test_geometry_factor_follows_perimeter_routing_and_feeds_the_forward_model() -> None:
    """Both layouts are softest at bed centre; AWD halves the belt path and varies more with position."""
    spec = _spec()  # stock Trident, 2WD
    asm_c = build_assembly(spec)
    assert asm_c.toolhead_rel_pos == (0.5, 0.5, 0.5)
    g_mid = cord_geometry_factor_m(asm_c, "X", 0.5, 0.5)
    for u, v in ((0.5, 0.1), (0.5, 0.9), (0.1, 0.1), (0.9, 0.9)):
        assert cord_geometry_factor_m(asm_c, "X", u, v) <= g_mid * (1.0 + 1e-9)
    # 2WD branches stay long everywhere, so the factor changes little front to back
    assert cord_geometry_factor_m(asm_c, "X", 0.5, 0.9) > 0.85 * g_mid

    awd = _spec()
    awd.gantry.drive_mode = "awd"
    asm_awd = build_assembly(awd)
    g_awd_mid = cord_geometry_factor_m(asm_awd, "X", 0.5, 0.5)
    assert g_mid == pytest.approx(2.0 * g_awd_mid, rel=0.02)
    assert cord_geometry_factor_m(asm_awd, "X", 0.1, 0.9) < 0.85 * g_awd_mid

    # Cord compliance equals G / EA_eff with one effective EA at every position (within 4 %)
    ea_vals = []
    for u, v in ((0.5, 0.5), (0.5, 0.1), (0.5, 0.9)):
        d = compute_drive_stiffness_details_n_per_m(asm_c, "X", u_rel=u, v_rel=v)
        ea_vals.append(float(d["k_belt_cord_n_m"]) * float(d["cord_geometry_factor_m"]))
    assert max(ea_vals) / min(ea_vals) < 1.04
    assert 16900.0 < ea_vals[0] < 16900.0 * 1.85

    # The assembly's own position is the default for belt geometry
    asm_back = build_assembly(spec, toolhead_rel_pos=(0.5, 0.9, 0.5))
    assert asm_back.toolhead_rel_pos == (0.5, 0.9, 0.5)
    assert cord_geometry_factor_m(asm_back, "X") == pytest.approx(
        cord_geometry_factor_m(asm_c, "X", 0.5, 0.9), rel=1e-12
    )
    f_mid = solve_modal_fea(asm_awd).shaper_x.effective_freq_hz
    f_corner = solve_modal_fea(build_assembly(awd, toolhead_rel_pos=(0.1, 0.9, 0.5))).shaper_x.effective_freq_hz
    assert f_corner > f_mid * 1.02


def test_identification_on_forward_model_sweep_lands_near_model_stiffness() -> None:
    """An AWD sweep simulated by the full model (frame compliance also moves) identifies EA within 25 %."""
    spec = _spec("voron_monolith_awd.yaml")
    pts = []
    for u, v in ((0.5, 0.5), (0.1, 0.1), (0.9, 0.9), (0.5, 0.1), (0.5, 0.9), (0.1, 0.5), (0.9, 0.5)):
        m = solve_modal_fea(build_assembly(spec, toolhead_rel_pos=(u, v, 0.5)), include_adxl_diagnostic=False)
        pts.append((u, v, m.shaper_y.effective_freq_hz))
    asm = build_assembly(spec)
    d = compute_drive_stiffness_details_n_per_m(asm, "Y")
    ea_model = float(d["k_belt_cord_n_m"]) * float(d["cord_geometry_factor_m"])
    res = identify_belt_from_position_sweep(asm, "Y", pts)
    assert res.ea_in_place_n == pytest.approx(ea_model, rel=0.32)
    assert res.r_squared > 0.85
    assert 15.0 < res.belt_share_pct < 90.0


# ---------------------------------------------------------------------------
# Controlled same-machine measurements (CloakedWayne 300 mm testbed, Monolith Discord #updates)
# 10 mm Gates EPDM belts at ~20 lbf (89 N), LDO 42STH48-2504AH at 36 V, Beacon accelerometer, room temp
# ---------------------------------------------------------------------------

_CW_AWD_ZV_HZ = {"X": 178.6, "Y": 114.4}  # 2025-07-10, LDO CNC kit in AWD, 2.5 A
_CW_2WD_ZV_HZ = {"X": 129.0, "Y": 83.2}  # 2025-07-10, same machine in 2WD, 2.5 A
_CW_CURRENT_SWEEP_Y_ZV_HZ = {1.0: 97.4, 1.5: 104.0, 2.0: 107.6, 2.5: 110.4, 2.8: 111.6}  # 2025-08-12, AWD


def _cw_zv(**mods: object) -> dict[str, float]:
    spec = _spec("cw_fixed_gantry_300.yaml")
    for dotted, value in mods.items():
        obj = spec
        parts = dotted.split("__")
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
    spec = PrinterSpec.model_validate(spec.model_dump())
    modal = solve_modal_fea(build_assembly(spec), include_adxl_diagnostic=False)
    out = {}
    for ax, graph in (("X", modal.shaper_graph_x), ("Y", modal.shaper_graph_y)):
        out[ax] = {c["name"]: c["freq_hz"] for c in graph.to_dict()["candidates"]}["ZV"]
    return out


def test_model_reproduces_controlled_awd_vs_2wd_measurement() -> None:
    """Same machine, same day, AWD vs 2WD: measured frequency ratio 1.38 on both axes."""
    awd = _cw_zv(kinematics__xy_stepper__run_current_a=2.5)
    two = _cw_zv(kinematics__xy_stepper__run_current_a=2.5, gantry__drive_mode="2wd")
    for ax in ("X", "Y"):
        assert awd[ax] == pytest.approx(_CW_AWD_ZV_HZ[ax], rel=0.10)
        assert two[ax] == pytest.approx(_CW_2WD_ZV_HZ[ax], rel=0.10)
        measured_ratio = _CW_AWD_ZV_HZ[ax] / _CW_2WD_ZV_HZ[ax]
        assert awd[ax] / two[ax] == pytest.approx(measured_ratio, rel=0.07)
        # Belt geometry alone gives sqrt(2); the old short-plus-long-branch routing gave only ~1.2
        assert awd[ax] / two[ax] > 1.27


def test_model_reproduces_controlled_motor_current_sweep() -> None:
    """Y resonance vs stepper current follows a series-spring law; motor share ~19 % at 2.5 A."""
    # Measured: 1 / f^2 is linear in 1 / I
    cur = sorted(_CW_CURRENT_SWEEP_Y_ZV_HZ)
    inv_i = [1.0 / c for c in cur]
    inv_f2 = [1.0 / _CW_CURRENT_SWEEP_Y_ZV_HZ[c] ** 2 for c in cur]
    n = len(cur)
    mx, my = sum(inv_i) / n, sum(inv_f2) / n
    slope = sum((a - mx) * (b - my) for a, b in zip(inv_i, inv_f2)) / sum((a - mx) ** 2 for a in inv_i)
    intercept = my - slope * mx
    fit = [intercept + slope * a for a in inv_i]
    assert max(abs(f - b) / b for f, b in zip(fit, inv_f2)) < 0.006
    motor_share_at_2p5 = slope / 2.5 / (intercept + slope / 2.5)
    assert motor_share_at_2p5 == pytest.approx(0.19, abs=0.02)

    model = {c: _cw_zv(kinematics__xy_stepper__run_current_a=c)["Y"] for c in cur}
    for c in cur:
        assert model[c] == pytest.approx(_CW_CURRENT_SWEEP_Y_ZV_HZ[c], rel=0.04)
        measured_rel = _CW_CURRENT_SWEEP_Y_ZV_HZ[c] / _CW_CURRENT_SWEEP_Y_ZV_HZ[2.5]
        assert model[c] / model[2.5] == pytest.approx(measured_rel, abs=0.025)


# ---------------------------------------------------------------------------
# Real stock Voron Trident measurements at known belt tension (VORONDesign Discord resonance forum)
# ---------------------------------------------------------------------------

_LBF = 4.44822


def _trident_mzv(tension_n: float, tap: bool = False, chamber_c: float = 25.0) -> dict[str, float]:
    import yaml

    from printsim.schema import TOOLHEAD_PRESETS

    raw = yaml.safe_load((EXAMPLES_DIR / "voron_trident_300.yaml").read_text())
    raw["gantry"]["belt"]["pretension_n"] = tension_n
    raw["frame"]["chamber_temp_c"] = chamber_c
    if tap:
        preset = TOOLHEAD_PRESETS["stealthburner_tap"]
        raw["toolhead"]["preset_id"] = "stealthburner_tap"
        for key in ("total_mass_g", "cog_offset_mm", "nozzle_offset_mm", "z_probe_type"):
            if key in preset:
                raw["toolhead"][key] = preset[key]
    modal = solve_modal_fea(build_assembly(PrinterSpec.model_validate(raw)), include_adxl_diagnostic=False)
    out = {"X_peak": modal.shaper_x.effective_freq_hz, "Y_peak": modal.shaper_y.effective_freq_hz}
    for ax, graph in (("X", modal.shaper_graph_x), ("Y", modal.shaper_graph_y)):
        out[ax] = {c["name"]: c["freq_hz"] for c in graph.to_dict()["candidates"]}["MZV"]
    return out


def test_model_matches_real_stock_trident_at_measured_belt_tension() -> None:
    """Stock Trident 300 measured hot at 2.2 lb belt tension: X MZV 61.4-64.0 Hz, Y MZV 40.8-42.2 Hz."""
    hot = _trident_mzv(2.2 * _LBF, chamber_c=50.0)
    assert hot["X"] == pytest.approx(62.7, rel=0.10)
    assert hot["Y"] == pytest.approx(41.5, rel=0.12)
    # The same machine cold at 1.4 lb read X 55.4 / Y 34.4 Hz (Y with unusually high damping)
    cold = _trident_mzv(1.4 * _LBF)
    assert cold["X"] == pytest.approx(55.4, rel=0.12)
    assert cold["Y"] == pytest.approx(34.4, rel=0.22)
    # Tension alone moves both axes up by 8-25 % between those two readings
    for ax in ("X", "Y"):
        assert 1.08 <= hot[ax] / cold[ax] <= 1.25
    # X / Y follows the moving-mass ratio (about 1.3-1.6), not the 1.15-1.2 of the curated rows
    assert 1.3 <= hot["X"] / hot["Y"] <= 1.6


def test_model_matches_real_trident_tap_accelerometer_file() -> None:
    """Formbot Trident R2 + Tap raw CSVs in data/touchstones: peaks X 50.3 Hz, Y 36.1 Hz."""
    sim = _trident_mzv(9.15, tap=True)  # Voron-documented 110 Hz pluck on a 150 mm span
    assert sim["X_peak"] == pytest.approx(50.3, rel=0.16)
    assert sim["Y_peak"] == pytest.approx(36.1, rel=0.16)
    assert sim["X_peak"] / sim["Y_peak"] == pytest.approx(50.3 / 36.1, rel=0.05)
