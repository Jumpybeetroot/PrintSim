"""Tests for CoreXY top panel toggle, expanded bracing, gantry top clearance, and Clee Trident Backbone."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import eigvalsh

from printsim.engine import evaluate_printer
from printsim.fea import assemble_global_matrices, solve_modal_fea, solve_static_fea
from printsim.fea.solver import _build_factorized_solver
from printsim.geometry import build_assembly
from printsim.schema import PrinterSpec, get_material

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def test_new_frame_defaults_match_baseline_exactly() -> None:
    """Default top_panel=True, gantry_top_clearance_mm=55.0, trident_backbone=False preserve baseline."""
    spec_default = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    assert spec_default.frame.top_panel is True
    assert spec_default.frame.gantry_top_clearance_mm == pytest.approx(55.0)
    assert spec_default.frame.trident_backbone is False

    spec_explicit = spec_default.model_copy(deep=True)
    spec_explicit.frame.top_panel = True
    spec_explicit.frame.gantry_top_clearance_mm = 55.0
    spec_explicit.frame.trident_backbone = False

    asm_def = build_assembly(spec_default)
    asm_exp = build_assembly(spec_explicit)
    assert asm_def.to_geometry_dict() == asm_exp.to_geometry_dict()
    assert asm_def.total_mass_kg == pytest.approx(asm_exp.total_mass_kg, rel=1e-12)
    assert asm_def.estimated_material_cost_usd == pytest.approx(
        asm_exp.estimated_material_cost_usd, rel=1e-12
    )


def test_top_panel_toggle_removes_top_shells_reduces_mass_and_lowers_freq() -> None:
    """Disabling top_panel removes 'top' MITC4 shells, reduces mass, and drops open-top torsional freq."""
    spec_on = PrinterSpec(architecture="corexy")
    spec_on.frame.bracing = "none"
    spec_on.frame.enclosed_panels = True
    spec_on.frame.panel_mounting = "bolted_structural"
    spec_on.frame.panel_material = "aluminum_5052_sheet"
    spec_on.frame.panel_thickness_mm = 4.0
    spec_on.frame.top_panel = True

    spec_off = spec_on.model_copy(deep=True)
    spec_off.frame.top_panel = False

    asm_on = build_assembly(spec_on)
    asm_off = build_assembly(spec_off)
    gd_on = asm_on.to_geometry_dict()
    gd_off = asm_off.to_geometry_dict()

    assert any(s.bay == "top" for s in asm_on.shell_elements)
    assert not any(s.bay == "top" for s in asm_off.shell_elements)
    assert len(asm_on.shell_elements) == 16
    assert len(asm_off.shell_elements) == 12
    assert gd_on["panels"]["top_panel"] is True
    assert gd_off["panels"]["top_panel"] is False

    al5052_rho = get_material("aluminum_5052_sheet").density_kg_m3
    expected_top_mass = (
        (spec_on.frame.width_mm * 1e-3)
        * (spec_on.frame.depth_mm * 1e-3)
        * (al5052_rho * 0.004)
    )
    assert (asm_on.panel_mass_kg - asm_off.panel_mass_kg) == pytest.approx(
        expected_top_mass, rel=1e-6
    )
    assert gd_off["panels"]["total_panel_mass_kg"] < gd_on["panels"]["total_panel_mass_kg"]

    mo_on = solve_modal_fea(asm_on)
    mo_off = solve_modal_fea(asm_off)
    st_on = solve_static_fea(asm_on)
    st_off = solve_static_fea(asm_off)
    assert mo_off.fundamental_freq_hz < mo_on.fundamental_freq_hz
    assert st_off.toolhead_stiffness_x_n_per_mm < st_on.toolhead_stiffness_x_n_per_mm


def test_expanded_bracing_modes_and_open_top_cross_stiffening() -> None:
    """New bracing modes add expected truss rods and stiffen open-top and bare CoreXY frames."""
    expected_braces = {
        "none": set(),
        "corner_gussets": set(),
        "top_cross": {"brace_top_1", "brace_top_2"},
        "rear_cross": {"brace_rear_1", "brace_rear_2", "brace_top_1"},
        "side_cross": {"brace_left_1", "brace_left_2", "brace_right_1", "brace_right_2"},
        "rear_and_sides": {
            "brace_rear_1",
            "brace_rear_2",
            "brace_left_1",
            "brace_left_2",
            "brace_right_1",
            "brace_right_2",
        },
        "full_truss": {
            "brace_rear_1",
            "brace_rear_2",
            "brace_left_1",
            "brace_left_2",
            "brace_right_1",
            "brace_right_2",
            "brace_top_1",
            "brace_top_2",
        },
    }

    open_results = {}
    for mode, expected_ids in expected_braces.items():
        spec = PrinterSpec(architecture="corexy")
        spec.frame.enclosed_panels = False
        spec.frame.bracing = mode  # type: ignore[assignment]
        asm = build_assembly(spec)
        brace_ids = {m.id for m in asm.members if m.role == "brace"}
        assert brace_ids == expected_ids
        st = solve_static_fea(asm)
        mo = solve_modal_fea(asm)
        open_results[mode] = (st, mo)

    # All new modes increase directional static stiffness vs bare 'none'
    st_none, mo_none = open_results["none"]
    assert open_results["top_cross"][0].toolhead_stiffness_x_n_per_mm > st_none.toolhead_stiffness_x_n_per_mm
    assert open_results["side_cross"][0].toolhead_stiffness_y_n_per_mm > st_none.toolhead_stiffness_y_n_per_mm
    assert open_results["side_cross"][1].shaper_y.effective_freq_hz > mo_none.shaper_y.effective_freq_hz
    assert open_results["side_cross"][1].fundamental_freq_hz > mo_none.fundamental_freq_hz
    assert open_results["rear_and_sides"][0].toolhead_stiffness_x_n_per_mm > st_none.toolhead_stiffness_x_n_per_mm
    assert open_results["rear_and_sides"][0].toolhead_stiffness_y_n_per_mm > st_none.toolhead_stiffness_y_n_per_mm
    assert open_results["rear_and_sides"][1].fundamental_freq_hz > mo_none.fundamental_freq_hz * 1.5
    assert open_results["full_truss"][1].fundamental_freq_hz > open_results["rear_and_sides"][1].fundamental_freq_hz

    # Open-top bolted enclosure (top_panel=False) + top_cross restores torsional & structural resonance
    spec_open_top = PrinterSpec(architecture="corexy")
    spec_open_top.frame.bracing = "none"
    spec_open_top.frame.enclosed_panels = True
    spec_open_top.frame.panel_mounting = "bolted_structural"
    spec_open_top.frame.panel_material = "aluminum_5052_sheet"
    spec_open_top.frame.panel_thickness_mm = 4.0
    spec_open_top.frame.top_panel = False

    spec_top_cross = spec_open_top.model_copy(deep=True)
    spec_top_cross.frame.bracing = "top_cross"

    asm_open_top = build_assembly(spec_open_top)
    asm_top_cross = build_assembly(spec_top_cross)
    mo_open_top = solve_modal_fea(asm_open_top)
    mo_top_cross = solve_modal_fea(asm_top_cross)
    st_open_top = solve_static_fea(asm_open_top)
    st_top_cross = solve_static_fea(asm_top_cross)

    assert mo_top_cross.fundamental_freq_hz > mo_open_top.fundamental_freq_hz + 10.0
    assert st_top_cross.toolhead_stiffness_x_n_per_mm > st_open_top.toolhead_stiffness_x_n_per_mm


def test_gantry_top_clearance_shifts_elevation_and_evaluates_cleanly() -> None:
    """Changing gantry_top_clearance_mm from 55 to 150 mm shifts g_* Z down by 95 mm."""
    spec_55 = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    spec_55.frame.gantry_top_clearance_mm = 55.0

    spec_150 = spec_55.model_copy(deep=True)
    spec_150.frame.gantry_top_clearance_mm = 150.0

    rep_55 = evaluate_printer(spec_55, include_advanced_analysis=False)
    rep_150 = evaluate_printer(spec_150, include_advanced_analysis=False)

    gd_55 = rep_55.assembly.to_geometry_dict()
    gd_150 = rep_150.assembly.to_geometry_dict()
    assert gd_55["gantry_top_clearance_mm"] == pytest.approx(55.0)
    assert gd_150["gantry_top_clearance_mm"] == pytest.approx(150.0)

    nmap_55 = rep_55.assembly.node_map
    nmap_150 = rep_150.assembly.node_map
    for nid in ("g_fl", "g_fr", "g_rl", "g_rr", "g_rear_mid", "y_car_l", "y_car_r"):
        z_55_mm = nmap_55[nid].position_m[2] * 1000.0
        z_150_mm = nmap_150[nid].position_m[2] * 1000.0
        assert (z_55_mm - z_150_mm) == pytest.approx(95.0, abs=1e-6)

    assert 0.0 < rep_150.scores.composite <= 100.0
    assert rep_150.modal_fea.fundamental_freq_hz > 0.0


def test_trident_backbone_physics_mass_cost_stiffness_and_flying_guard() -> None:
    """Clee Trident Backbone stiffens Z/bed/lower-frame, adds ~4.1 kg & $44.99-$54.99, and stays inactive on V2.4."""
    base_spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    base_spec.frame.trident_backbone = False
    bb_spec = base_spec.model_copy(deep=True)
    bb_spec.frame.trident_backbone = True

    for enclosed in (False, True):
        s0 = base_spec.model_copy(deep=True)
        s0.frame.enclosed_panels = enclosed
        s1 = bb_spec.model_copy(deep=True)
        s1.frame.enclosed_panels = enclosed

        rep0 = evaluate_printer(s0, include_advanced_analysis=True)
        rep1 = evaluate_printer(s1, include_advanced_analysis=True)
        gd0 = rep0.assembly.to_geometry_dict()
        gd1 = rep1.assembly.to_geometry_dict()

        assert gd0["trident_backbone"]["active"] is False
        assert gd0["trident_backbone"]["cost_usd"] == 0.0
        assert gd0["trident_backbone"]["mass_kg"] == 0.0

        assert gd1["trident_backbone"]["enabled"] is True
        assert gd1["trident_backbone"]["active"] is True
        assert gd1["trident_backbone"]["cost_usd"] == pytest.approx(49.99)
        assert gd1["trident_backbone"]["z_thrust_boost_factor"] == pytest.approx(2.4)
        assert 3.5 <= gd1["trident_backbone"]["mass_kg"] <= 4.8
        assert any(s.bay == "backbone" for s in rep1.assembly.shell_elements)
        assert any(
            sp.bay == "backbone" and sp.frame_node_id == "b_rear_mid"
            for sp in rep1.assembly.fastener_springs
        )

        mass_delta = rep1.assembly.total_mass_kg - rep0.assembly.total_mass_kg
        cost_delta = rep1.assembly.estimated_material_cost_usd - rep0.assembly.estimated_material_cost_usd
        assert 3.5 <= mass_delta <= 4.8
        assert mass_delta == pytest.approx(rep1.assembly.backbone_mass_kg, rel=1e-4)
        assert cost_delta == pytest.approx(49.99, abs=0.05)

        # Lower vertical columns receive base-plate corner-joint stiffening; upper columns keep upper-joint compliance
        m0_map = {m.id: m for m in rep0.assembly.members}
        m1_map = {m.id: m for m in rep1.assembly.members}
        for lower_id in ("col_fl_bot", "col_fr_bot", "col_rl_low", "col_rr_low"):
            assert m1_map[lower_id].profile.iy_m4 > m0_map[lower_id].profile.iy_m4
            assert m1_map[lower_id].profile.j_m4 > m0_map[lower_id].profile.j_m4
        for upper_id in ("col_fl_top", "col_rl_top"):
            assert m1_map[upper_id].profile.iy_m4 == pytest.approx(m0_map[upper_id].profile.iy_m4, rel=1e-9)
            assert m1_map[upper_id].profile.j_m4 == pytest.approx(m0_map[upper_id].profile.j_m4, rel=1e-9)

        # Matrix symmetry and strict positive definiteness on free DOFs
        K, M, _, nidx, free = assemble_global_matrices(rep1.assembly)
        K_ff = K[np.ix_(free, free)]
        M_ff = M[np.ix_(free, free)]
        assert np.allclose(K_ff, K_ff.T, atol=1e-8)
        assert np.allclose(M_ff, M_ff.T, atol=1e-12)
        assert float(np.min(eigvalsh(K_ff))) > 0.0
        assert float(np.min(eigvalsh(M_ff))) > 0.0

        # Measurable gains across f1, input shaper XY, toolhead Kxyz, bed deflection, and b_rear_mid sag
        assert rep1.static_fea.toolhead_stiffness_x_n_per_mm > rep0.static_fea.toolhead_stiffness_x_n_per_mm
        assert rep1.static_fea.toolhead_stiffness_y_n_per_mm > rep0.static_fea.toolhead_stiffness_y_n_per_mm
        assert rep1.static_fea.toolhead_stiffness_z_n_per_mm > rep0.static_fea.toolhead_stiffness_z_n_per_mm
        assert rep1.static_fea.bed_deflection_um < rep0.static_fea.bed_deflection_um * 0.90
        assert rep1.modal_fea.fundamental_freq_hz > rep0.modal_fea.fundamental_freq_hz + 0.8
        assert rep1.modal_fea.shaper_x.effective_freq_hz > rep0.modal_fea.shaper_x.effective_freq_hz
        assert rep1.modal_fea.shaper_y.effective_freq_hz > rep0.modal_fea.shaper_y.effective_freq_hz

        K0, _, _, nidx0, free0 = assemble_global_matrices(rep0.assembly)
        slv0 = _build_factorized_solver(K0, free0)
        slv1 = _build_factorized_solver(K, free)
        f0 = np.zeros(K0.shape[0])
        f0[6 * nidx0["bed_surface"] + 2] = -10.0
        f1 = np.zeros(K.shape[0])
        f1[6 * nidx["bed_surface"] + 2] = -10.0
        u0 = slv0(f0)
        u1 = slv1(f1)
        assert abs(u1[6 * nidx["bed_surface"] + 2]) < abs(u0[6 * nidx0["bed_surface"] + 2]) * 0.90
        assert abs(u1[6 * nidx["b_rear_mid"] + 2]) < abs(u0[6 * nidx0["b_rear_mid"] + 2]) * 0.65

    # Tiered West3D retail pricing by Trident build size (250 / 300 / 350 mm)
    for size_mm, expected_price in ((250.0, 44.99), (300.0, 49.99), (350.0, 54.99)):
        s_sz = bb_spec.model_copy(deep=True)
        s_sz.kinematics.build_volume_mm = [size_mm, size_mm, 250.0]
        asm_sz = build_assembly(s_sz)
        assert asm_sz.backbone_cost_usd == pytest.approx(expected_price)
        assert asm_sz.to_geometry_dict()["trident_backbone"]["cost_usd"] == pytest.approx(expected_price)

    # On rubber_feet, Trident Backbone also increases frame XY & Z stiffness and fundamental frequency
    s_rf0 = base_spec.model_copy(deep=True)
    s_rf0.frame.feet_type = "rubber_feet"
    s_rf0.frame.enclosed_panels = False
    s_rf1 = s_rf0.model_copy(deep=True)
    s_rf1.frame.trident_backbone = True

    asm_rf0 = build_assembly(s_rf0)
    asm_rf1 = build_assembly(s_rf1)
    st_rf0 = solve_static_fea(asm_rf0)
    st_rf1 = solve_static_fea(asm_rf1)
    mo_rf0 = solve_modal_fea(asm_rf0)
    mo_rf1 = solve_modal_fea(asm_rf1)

    assert st_rf1.toolhead_stiffness_x_n_per_mm > st_rf0.toolhead_stiffness_x_n_per_mm
    assert st_rf1.toolhead_stiffness_y_n_per_mm > st_rf0.toolhead_stiffness_y_n_per_mm
    assert st_rf1.toolhead_stiffness_z_n_per_mm > st_rf0.toolhead_stiffness_z_n_per_mm
    assert mo_rf1.fundamental_freq_hz > mo_rf0.fundamental_freq_hz

    # Flying gantry CoreXY (Voron 2.4) safely keeps trident_backbone inactive
    v24_off = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    v24_on = v24_off.model_copy(deep=True)
    v24_on.frame.trident_backbone = True

    asm_v24_off = build_assembly(v24_off)
    asm_v24_on = build_assembly(v24_on)
    gd_v24 = asm_v24_on.to_geometry_dict()
    assert gd_v24["trident_backbone"]["enabled"] is True
    assert gd_v24["trident_backbone"]["active"] is False
    assert gd_v24["trident_backbone"]["mass_kg"] == 0.0
    assert gd_v24["trident_backbone"]["cost_usd"] == 0.0
    assert asm_v24_on.total_mass_kg == pytest.approx(asm_v24_off.total_mass_kg, rel=1e-12)
    assert asm_v24_on.estimated_material_cost_usd == pytest.approx(
        asm_v24_off.estimated_material_cost_usd, rel=1e-12
    )
    assert not any(s.bay == "backbone" for s in asm_v24_on.shell_elements)
