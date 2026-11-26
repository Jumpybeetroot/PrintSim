"""Analytical checks for suspended mounting, frame ballast, mount-mode coupling, and belt refinements."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from printsim.engine import evaluate_printer
from printsim.fea import solve_modal_fea
from printsim.fea.components import (
    belt_operating_pretension_n,
    belt_thermal_tension_delta_n,
    operating_belt,
)
from printsim.fea.modal import (
    mount_decoupling_factor,
    solve_corexy_cross_axis_coupling,
    solve_mount_coupled_axis,
)
from printsim.geometry import build_assembly
from printsim.schema import (
    BELT_CATALOG,
    FRAME_FEET_CATALOG,
    SUSPENSION_LINE_CATALOG,
    BeltSpec,
    PrinterSpec,
    is_printdb_available,
    resolve_frame_feet,
)
from printsim.schema.printdb_bridge import get_record_specs

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
G_ACC = 9.80665


def _spec(name: str = "voron_trident_300.yaml") -> PrinterSpec:
    return PrinterSpec.from_yaml(EXAMPLES_DIR / name)


def _suspended(name: str, line: str, length_mm: float, **frame_kw) -> PrinterSpec:
    s = _spec(name)
    s.frame.feet_type = "suspended_lines"
    s.frame.suspension_line = line  # type: ignore[assignment]
    s.frame.suspension_length_mm = length_mm
    for k, v in frame_kw.items():
        setattr(s.frame, k, v)
    return s


def _rb(mod, kind: str) -> list[float]:
    return [m["frequency_hz"] for m in mod.mounting_dynamics["rigid_body_modes"] if m["kind"] == kind]


# ---------------------------------------------------------------------------
# Suspended mounting: pendulum sway, vertical bounce, yaw
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("length_mm", [400.0, 1000.0, 2000.0])
def test_suspended_sway_matches_simple_pendulum(length_mm: float) -> None:
    """Parallel lines: sway f = (1/2pi) sqrt(g/L), independent of the printer's mass."""
    mod = solve_modal_fea(build_assembly(_suspended("voron_trident_300.yaml", "steel_cable_1_5mm", length_mm)))
    f_pend = math.sqrt(G_ACC / (length_mm * 1e-3)) / (2.0 * math.pi)
    for kind in ("sway_x", "sway_y"):
        (f_fe,) = _rb(mod, kind)
        assert f_fe == pytest.approx(f_pend, rel=0.015)
    assert mod.mounting_dynamics["suspension"]["pendulum_hz_analytic"] == pytest.approx(f_pend, abs=1e-3)


def test_suspended_sway_is_mass_invariant_and_tension_tracks_weight() -> None:
    """Ballast raises line tension T = M g / n but leaves the pendulum frequency unchanged."""
    light = build_assembly(_suspended("voron_trident_300.yaml", "uhmwpe_braid_200lb", 800.0))
    heavy = build_assembly(
        _suspended("voron_trident_300.yaml", "uhmwpe_braid_200lb", 800.0, ballast_mass_kg=20.0)
    )
    assert heavy.total_mass_kg == pytest.approx(light.total_mass_kg + 20.0, rel=1e-9)
    for asm in (light, heavy):
        assert asm.suspension_line_tension_n == pytest.approx(asm.total_mass_kg * G_ACC / 4.0, rel=1e-9)
        for sp in (sp for sp in asm.fastener_springs if sp.bay == "feet"):
            assert sp.k_shear_n_m == pytest.approx(asm.suspension_line_tension_n / 0.8, rel=1e-9)
            assert sp.k_rot_nm_rad == 0.0
    f_light = _rb(solve_modal_fea(light), "sway_x")[0]
    f_heavy = _rb(solve_modal_fea(heavy), "sway_x")[0]
    assert f_heavy == pytest.approx(f_light, rel=0.01)


@pytest.mark.parametrize("line", ["uhmwpe_braid_200lb", "steel_cable_1_5mm", "bungee_cord_6mm"])
def test_suspended_bounce_matches_line_axial_stiffness(line: str) -> None:
    """Vertical bounce f = (1/2pi) sqrt(n EA / (L M))."""
    length_mm = 600.0
    asm = build_assembly(_suspended("voron_trident_300.yaml", line, length_mm))
    ea = SUSPENSION_LINE_CATALOG[line]["stiffness_ea_n"]
    f_bounce = math.sqrt(4.0 * ea / (length_mm * 1e-3) / asm.total_mass_kg) / (2.0 * math.pi)
    (f_fe,) = _rb(solve_modal_fea(asm), "bounce_z")
    assert f_fe == pytest.approx(f_bounce, rel=0.03)


def test_suspended_yaw_matches_quadrifilar_pendulum() -> None:
    """Yaw f = f_sway * (attachment radius / yaw radius of gyration); lines attach at the top corners."""
    asm = build_assembly(_suspended("voron_trident_300.yaml", "steel_cable_1_5mm", 1000.0))
    lines = [sp for sp in asm.fastener_springs if sp.bay == "feet"]
    nmap = asm.node_map
    assert len(lines) == 4
    z_top = max(n.position_m[2] for n in asm.nodes if n.role in ("frame", "foot"))
    for sp in lines:
        attach, anchor = nmap[sp.frame_node_id], nmap[sp.shell_node_id]
        assert attach.position_m[2] == pytest.approx(z_top, abs=1e-6)
        assert anchor.position_m[2] - attach.position_m[2] == pytest.approx(1.0, abs=1e-9)
        assert anchor.fixed
    mod = solve_modal_fea(asm)
    ratio = _rb(mod, "yaw")[0] / _rb(mod, "sway_x")[0]
    # Attachment radius of a 300-class cube is larger than its yaw gyradius, so yaw sits above sway
    assert 1.05 < ratio < 1.8


def test_suspended_modes_stay_out_of_structural_mode_list_and_bolted_is_unchanged() -> None:
    """Whole-printer swing modes are reported under mounting_dynamics, never as the fundamental."""
    mod = solve_modal_fea(build_assembly(_suspended("voron_trident_300.yaml", "paracord_550", 600.0)))
    assert mod.fundamental_freq_hz > 30.0
    assert all(m.frequency_hz > 30.0 for m in mod.modes)
    md = mod.mounting_dynamics
    assert md["lowest_sway_hz"] < 1.0
    assert md["suspension"]["line_count"] == 4
    assert md["suspension"]["safety_factor"] > md["suspension"]["design_factor"]
    assert solve_modal_fea(build_assembly(_spec())).mounting_dynamics is None


@pytest.mark.parametrize(
    "yaml_name", ["voron_trident_300.yaml", "cartesian_bedslinger.yaml", "delta_racer.yaml"]
)
def test_suspended_evaluates_on_every_architecture(yaml_name: str) -> None:
    s = _suspended(yaml_name, "paracord_550", 600.0)
    rep = evaluate_printer(s)
    rigid = evaluate_printer(_spec(yaml_name))
    assert math.isfinite(rep.scores.composite)
    for ax in ("shaper_x", "shaper_y"):
        ratio = getattr(rep.modal_fea, ax).effective_freq_hz / getattr(rigid.modal_fea, ax).effective_freq_hz
        assert 0.90 <= ratio <= 1.25
    assert rep.modal_fea.mounting_dynamics["lowest_sway_hz"] == pytest.approx(0.643, abs=0.08)
    geo = build_assembly(s).to_geometry_dict()
    assert geo["feet_type"] == "suspended_lines"
    assert len(geo["suspension"]["lines"]) >= 3
    assert geo["suspension"]["line_tension_n"] > 0.0
    assert build_assembly(_spec(yaml_name)).to_geometry_dict()["suspension"] is None


def test_overloaded_line_is_flagged() -> None:
    """100 lb monofilament under a ballasted printer drops below its design factor."""
    s = _suspended("voron_trident_300.yaml", "nylon_mono_100lb", 600.0, ballast_mass_kg=40.0)
    md = solve_modal_fea(build_assembly(s)).mounting_dynamics
    assert md["suspension"]["safety_factor"] < md["suspension"]["design_factor"]
    assert any("break strength" in n for n in md["notes"])


def test_suspension_resolver_and_catalog() -> None:
    s = _suspended("voron_trident_300.yaml", "uhmwpe_braid_200lb", 500.0)
    cfg = resolve_frame_feet(s)
    assert cfg["is_suspended"] and cfg["is_compliant"] and cfg["lateral_from_tension"]
    assert cfg["k_axial_n_m"] == pytest.approx(26000.0 / 0.5)
    assert "suspended_lines" in FRAME_FEET_CATALOG
    for entry in SUSPENSION_LINE_CATALOG.values():
        assert entry["stiffness_ea_n"] > 0 and entry["break_strength_n"] > 0
        assert entry["design_factor"] >= 5.0
    # Default spec is unaffected by the new fields
    d = _spec()
    assert d.frame.feet_type != "suspended_lines" and d.frame.ballast_mass_kg == 0.0


# ---------------------------------------------------------------------------
# Frame ballast vs mounting type
# ---------------------------------------------------------------------------


def test_base_ballast_is_inert_on_a_rigidly_bolted_frame() -> None:
    """Weight bolted to grounded feet cannot change the dynamics; top ballast lowers frame modes."""
    base = solve_modal_fea(build_assembly(_spec()))
    s_base = _spec()
    s_base.frame.ballast_mass_kg = 15.0
    asm_b = build_assembly(s_base)
    mod_b = solve_modal_fea(asm_b)
    assert asm_b.ballast_mass_kg == 15.0 and len(asm_b.ballast_node_ids) == 4
    assert mod_b.fundamental_freq_hz == pytest.approx(base.fundamental_freq_hz, rel=1e-9)
    assert mod_b.shaper_x.effective_freq_hz == pytest.approx(base.shaper_x.effective_freq_hz, rel=1e-9)

    s_top = _spec()
    s_top.frame.ballast_mass_kg = 15.0
    s_top.frame.ballast_location = "top"
    mod_t = solve_modal_fea(build_assembly(s_top))
    assert mod_t.fundamental_freq_hz < base.fundamental_freq_hz * 0.9


def test_base_ballast_on_rubber_feet_lowers_mount_modes_like_sqrt_mass_and_cuts_recoil() -> None:
    """Rigid-body sway on feet scales ~ sqrt(k / M); a heavier frame recoils less."""
    s0 = _spec()
    s0.frame.feet_type = "rubber_feet"
    s1 = s0.model_copy(deep=True)
    s1.frame.ballast_mass_kg = 12.0
    a0, a1 = build_assembly(s0), build_assembly(s1)
    m0, m1 = solve_modal_fea(a0), solve_modal_fea(a1)
    (b0,) = _rb(m0, "bounce_z")
    (b1,) = _rb(m1, "bounce_z")
    assert b1 / b0 == pytest.approx(math.sqrt(a0.total_mass_kg / a1.total_mass_kg), rel=0.05)
    for ax in ("x", "y"):
        assert (
            m1.mounting_dynamics[ax]["frame_recoil_ratio"]
            < 0.75 * m0.mounting_dynamics[ax]["frame_recoil_ratio"]
        )
    # A heavier frame converges toward the rigidly bolted result
    rigid = solve_modal_fea(build_assembly(_spec()))
    d0 = abs(m0.shaper_y.effective_freq_hz - rigid.shaper_y.effective_freq_hz)
    s2 = s0.model_copy(deep=True)
    s2.frame.ballast_mass_kg = 150.0
    d2 = abs(solve_modal_fea(build_assembly(s2)).shaper_y.effective_freq_hz - rigid.shaper_y.effective_freq_hz)
    assert d2 < d0


# ---------------------------------------------------------------------------
# Mount-mode coupling closed forms
# ---------------------------------------------------------------------------


def test_mount_decoupling_factor_limits() -> None:
    assert mount_decoupling_factor(0.0) == 0.0
    assert mount_decoupling_factor(1.0) == pytest.approx(0.5)
    assert mount_decoupling_factor(0.1) < 1e-3
    assert mount_decoupling_factor(10.0) > 0.999
    vals = [mount_decoupling_factor(r) for r in np.linspace(0.0, 6.0, 40)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))


def test_free_frame_reduced_mass_matches_two_body_closed_form() -> None:
    """Rigid frame M_f + moving mass m on a spring k: f = f_fixed * sqrt(1 + m / M_f)."""
    m, m_frame, k = 0.5, 12.0, 2.0e5
    c_drive = 1.0 / k
    f_fixed = math.sqrt(k / m) / (2.0 * math.pi)
    # One rigid-body sway mode far below the drive resonance: A = 1 / (M_f + m), zero relative residue
    mode = {"freq_hz": 0.6, "r_x": 0.0, "a_x": 1.0 / (m_frame + m)}
    sol = solve_mount_coupled_axis([mode], "X", 1e-12, c_drive, m)
    assert sol["freq_hz"] == pytest.approx(f_fixed * math.sqrt(1.0 + m / m_frame), rel=1e-3)
    assert sol["m_reduced_kg"] == pytest.approx(m * m_frame / (m + m_frame), rel=1e-3)
    # A mount mode far above the drive resonance leaves the fixed-base result
    stiff = {"freq_hz": 5000.0, "r_x": 0.0, "a_x": 1.0 / (m_frame + m)}
    sol_hi = solve_mount_coupled_axis([stiff], "X", 1e-12, c_drive, m)
    assert sol_hi["freq_hz"] == pytest.approx(f_fixed, rel=1e-4)
    # No mount modes: exactly the static chain
    sol_none = solve_mount_coupled_axis([], "X", 2.0e-6, c_drive, m)
    assert sol_none["freq_hz"] == sol_none["freq_static_hz"]
    assert sol_none["c_dyn_m_per_n"] == 2.0e-6 and sol_none["m_reduced_kg"] == m


def test_inertia_relief_drops_low_mount_mode_residue() -> None:
    """A mount mode far below the axis resonance stops contributing its static residue."""
    c_static, r_mount = 3.0e-6, 1.0e-6
    low = {"freq_hz": 1.0, "r_x": r_mount, "a_x": 0.0}
    sol = solve_mount_coupled_axis([low], "X", c_static, 2.0e-6, 0.5)
    assert sol["c_dyn_m_per_n"] == pytest.approx(c_static - r_mount, rel=1e-3)
    assert sol["retained_mount_compliance_m_per_n"] < 1e-3 * r_mount


# ---------------------------------------------------------------------------
# Belt refinements
# ---------------------------------------------------------------------------


def test_corexy_cross_axis_coupling_matches_belt_only_closed_form() -> None:
    """K = [[kA+kB, kA-kB],[kA-kB, kA+kB]], M = diag(mx, my): compare against numpy eigh."""
    k_a, k_b, mx, my = 1.3e5, 0.9e5, 0.45, 1.1
    kxx = k_a + k_b
    kappa = (k_a - k_b) / (k_a + k_b)
    sol = solve_corexy_cross_axis_coupling(kxx, kxx, mx, my, kappa)
    k_mat = np.array([[k_a + k_b, k_a - k_b], [k_a - k_b, k_a + k_b]])
    m_inv_sqrt = np.diag([1.0 / math.sqrt(mx), 1.0 / math.sqrt(my)])
    lam = np.linalg.eigvalsh(m_inv_sqrt @ k_mat @ m_inv_sqrt)
    f_ref = np.sqrt(lam) / (2.0 * math.pi)
    assert sol["f_y_hz"] == pytest.approx(f_ref[0], rel=1e-9)
    assert sol["f_x_hz"] == pytest.approx(f_ref[1], rel=1e-9)
    assert 0.0 < sol["cross_peak_in_x"] < 1.0 and 0.0 < sol["cross_peak_in_y"] < 1.0

    matched = solve_corexy_cross_axis_coupling(kxx, kxx, mx, my, 0.0)
    assert matched["f_x_hz"] == pytest.approx(math.sqrt(kxx / mx) / (2.0 * math.pi), rel=1e-12)
    assert matched["f_y_hz"] == pytest.approx(math.sqrt(kxx / my) / (2.0 * math.pi), rel=1e-12)
    assert matched["cross_peak_in_x"] == pytest.approx(0.0, abs=1e-9)
    assert matched["cross_peak_in_y"] == pytest.approx(0.0, abs=1e-9)


def test_belt_tension_mismatch_couples_axes_monotonically() -> None:
    """More A/B mismatch at a toe-region tension -> larger coupling and cross peak; matched = none."""
    base = _spec()
    base.gantry.belt.pretension_n = 10.0
    peaks = []
    for pct in (0.0, 20.0, 40.0, 60.0):
        s = base.model_copy(deep=True)
        s.gantry.belt.tension_mismatch_pct = pct
        bd = solve_modal_fea(build_assembly(s)).belt_dynamics
        if pct == 0.0:
            assert bd["coupling_kappa"] == 0.0 and "cross_peak_in_x" not in bd
            peaks.append(0.0)
        else:
            assert bd["tension_a_n"] == pytest.approx(10.0 * (1 + pct / 200.0), abs=0.01)
            assert bd["tension_b_n"] == pytest.approx(10.0 * (1 - pct / 200.0), abs=0.01)
            assert bd["k_belt_a_n_per_mm"] > bd["k_belt_b_n_per_mm"]
            peaks.append(bd["cross_peak_in_x"])
    assert all(b > a for a, b in zip(peaks, peaks[1:]))


def test_cold_tensioned_belt_tightens_in_hot_chamber_like_voron_measurement() -> None:
    """6 mm 2GT on aluminium: ~7.6 N at 21 C plucks ~100 Hz and reaches ~145 Hz above 50 C."""
    s = _spec()
    s.gantry.belt.type = "gt2_6mm"
    s.gantry.belt.core = "fiberglass"
    s.gantry.belt.pretension_n = 7.6
    s.gantry.belt.pretension_set_temp_c = 21.0
    s.frame.chamber_temp_c = 53.0
    t_hot = belt_operating_pretension_n(s)
    assert t_hot == pytest.approx(15.9, rel=0.25)
    assert t_hot - 7.6 == pytest.approx(belt_thermal_tension_delta_n(s), rel=1e-9)
    bd = solve_modal_fea(build_assembly(s)).belt_dynamics
    assert bd["pluck_hz_150mm_installed"] == pytest.approx(100.0, rel=0.03)
    assert bd["pluck_hz_150mm_operating"] == pytest.approx(145.0, rel=0.12)

    # Opt-in: without a set temperature the configured pretension is the operating tension
    s.gantry.belt.pretension_set_temp_c = None
    assert belt_thermal_tension_delta_n(s) == 0.0
    assert operating_belt(s) is s.gantry.belt

    # Steel cord expands with the frame less differently; aramid (negative CTE) tightens the most
    deltas = {}
    for core in ("steel", "fiberglass", "aramid"):
        c = _spec()
        c.gantry.belt.core = core  # type: ignore[assignment]
        c.gantry.belt.pretension_set_temp_c = 22.0
        c.frame.chamber_temp_c = 60.0
        deltas[core] = belt_thermal_tension_delta_n(c) / c.gantry.belt.static_stiffness_ea_n
    assert 0.0 < deltas["steel"] < deltas["fiberglass"] < deltas["aramid"]


def test_hot_chamber_tension_rise_raises_drive_frequency() -> None:
    cold = _spec()
    cold.gantry.belt.pretension_n = 9.0
    cold.frame.chamber_temp_c = 55.0
    hot = cold.model_copy(deep=True)
    hot.gantry.belt.pretension_set_temp_c = 22.0
    f_cold = solve_modal_fea(build_assembly(cold)).shaper_x.effective_freq_hz
    f_hot = solve_modal_fea(build_assembly(hot)).shaper_x.effective_freq_hz
    assert f_hot > f_cold


# ---------------------------------------------------------------------------
# Gates static EA (datasheet) and recalibrated dynamic ratio
# ---------------------------------------------------------------------------

_GATES_FIBERGLASS = (
    "gt2_6mm", "gt2_9mm", "gt2_10mm", "gt3_10mm", "gt2_12mm", "gt2_15mm", "gt2_dual_9mm",
    "3gt_9mm", "3gt_12mm", "3gt_15mm",
)


def test_gates_static_ea_follows_datasheet_with_one_measured_dynamic_ratio() -> None:
    """Table 6: 3.8 / 5.7 / 8.9 / 16.5 lb at 0.1 % elongation; one family ratio from controlled tests."""
    lb = 4.44822
    assert BeltSpec(type="gt2_6mm").static_stiffness_ea_n == pytest.approx(3.8 * lb / 0.001, rel=0.002)
    assert BeltSpec(type="gt2_9mm").static_stiffness_ea_n == pytest.approx(5.7 * lb / 0.001, rel=0.007)
    assert BeltSpec(type="3gt_9mm").static_stiffness_ea_n == pytest.approx(8.9 * lb / 0.001, rel=0.002)
    assert BeltSpec(type="3gt_15mm").static_stiffness_ea_n == pytest.approx(16.5 * lb / 0.001, rel=0.002)
    # Narrow-belt rule from the 2M unit-strain modulus (18,000 lb/in per inch of width)
    nominal_per_mm = 18000.0 * lb / 25.4
    assert BeltSpec(type="gt2_12mm").static_stiffness_ea_n == pytest.approx(nominal_per_mm * 12 * 0.90, rel=0.002)
    assert BeltSpec(type="gt2_15mm").static_stiffness_ea_n == pytest.approx(nominal_per_mm * 15, rel=0.002)

    # One dynamic/static ratio for the 2GT / 3GT neoprene-fiberglass family (GT3 2MGT 3 % softer)
    ratios = {k: BeltSpec(type=k).dynamic_stiffening_factor for k in _GATES_FIBERGLASS}
    family = {v for k, v in ratios.items() if k != "gt3_10mm"}
    assert family == {1.85}
    assert ratios["gt3_10mm"] == pytest.approx(1.85 * 0.97, rel=1e-4)
    for k in _GATES_FIBERGLASS:
        b = BeltSpec(type=k)
        assert b.stiffness_ea_n == pytest.approx(b.static_stiffness_ea_n * b.dynamic_stiffening_factor)
        assert BELT_CATALOG[k]["ea_default_core_n"] == pytest.approx(b.stiffness_ea_n)

    # The ratio sits inside the range the controlled testbed measurements support (1.6 - 2.0):
    # stiffer in place than the datasheet figure, softer than the earlier fitted 2.23
    assert 1.6 <= BeltSpec(type="gt2_10mm").dynamic_stiffening_factor <= 2.0
    # Stiffness per mm of width still rises monotonically with width inside each pitch family
    two_gt = [BeltSpec(type=k).static_stiffness_ea_n for k in ("gt2_6mm", "gt2_9mm", "gt2_10mm", "gt2_12mm", "gt2_15mm")]
    three_gt = [BeltSpec(type=k).static_stiffness_ea_n for k in ("3gt_9mm", "3gt_12mm", "3gt_15mm")]
    assert two_gt == sorted(two_gt) and three_gt == sorted(three_gt)


@pytest.mark.skipif(not is_printdb_available(), reason="PrintDB not installed")
def test_gates_static_ea_matches_printdb_records() -> None:
    """PrintDB is the single source of truth for datasheet tensile stiffness."""
    for key in _GATES_FIBERGLASS:
        rid = BELT_CATALOG[key].get("printdb_id")
        assert rid, key
        raw = get_record_specs(rid, si=False)
        assert raw and raw.get("tensile_stiffness_n_per_strain") is not None, rid
        assert BELT_CATALOG[key]["ea_static_n"] == pytest.approx(
            float(raw["tensile_stiffness_n_per_strain"]), rel=1e-9
        ), key
