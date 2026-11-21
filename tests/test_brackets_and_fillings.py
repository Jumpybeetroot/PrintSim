"""Physics, BOM, diagnostics, and API verification for gantry bracket materials and frame extrusion cavity fillings."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from printsim.api.server import create_app
from printsim.engine import evaluate_printer
from printsim.fea import calibrate_axis_shapers, solve_modal_fea, solve_static_fea
from printsim.geometry import build_assembly, generate_bom
from printsim.kinematics import analyze_kinematics
from printsim.kinematics.solvers import (
    compute_bracket_stiffness_n_per_m,
    compute_drive_stiffness_n_per_m,
)
from printsim.schema import (
    EXTRUSION_FILLINGS,
    GANTRY_BRACKET_CATALOG,
    PrinterSpec,
    get_material,
    get_profile,
)

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def _base_corexy_spec(
    *,
    drive_mode: str = "2wd",
    double_shear: bool = False,
    pretension_n: float = 28.0,
    bracket_material: str = "abs_printed",
    extrusion_filling: str = "none",
    enclosed_panels: bool = False,
    panel_mounting: str = "none",
) -> PrinterSpec:
    spec = PrinterSpec(architecture="corexy")
    spec.frame.bracing = "corner_gussets"
    spec.frame.enclosed_panels = enclosed_panels
    spec.frame.panel_mounting = panel_mounting  # type: ignore[assignment]
    spec.frame.extrusion_filling = extrusion_filling  # type: ignore[assignment]
    spec.gantry.drive_mode = drive_mode  # type: ignore[assignment]
    spec.gantry.double_shear_pulleys = double_shear
    spec.gantry.bracket_material = bracket_material  # type: ignore[assignment]
    spec.gantry.belt.pretension_n = pretension_n
    return spec


# ---------------------------------------------------------------------------
# 1. Gantry Bracket Materials: Stiffness, Mass, Creep, and Monolith AWD
# ---------------------------------------------------------------------------


def test_all_seven_bracket_materials_stiffness_order_and_series_formula() -> None:
    """Verify exact series bracket compliance across all 7 bracket materials."""
    ordered_materials = [
        "abs_printed",
        "pet_cf_printed",
        "pa12_cf_printed",
        "pps_cf_printed",
        "ppa_cf_printed",
        "aluminum_6061_t6",
        "aluminum_7075_t6",
    ]
    assert set(ordered_materials) == set(GANTRY_BRACKET_CATALOG.keys())

    k_bkts: list[float] = []
    k_drvs: list[float] = []
    for mat_id in ordered_materials:
        spec = _base_corexy_spec(
            drive_mode="awd",
            double_shear=True,
            pretension_n=60.0,
            bracket_material=mat_id,
        )
        asm = build_assembly(spec)
        kin = analyze_kinematics(asm)

        # Analytical series formula check
        e_bkt = get_material(mat_id).youngs_modulus_pa
        k_mat = 2.6e5 * 1.45 * 1.65 * (e_bkt / 2.1e9)
        expected_k_bkt = (k_mat * 8.0e6) / (k_mat + 8.0e6)
        k_bkt = compute_bracket_stiffness_n_per_m(asm)
        assert k_bkt == pytest.approx(expected_k_bkt, rel=1e-9)

        k_belt, k_motor, k_drive = compute_drive_stiffness_n_per_m(asm, "X")
        expected_k_drive = 1.0 / (1.0 / k_belt + 1.0 / k_motor + 1.0 / k_bkt)
        assert k_drive == pytest.approx(expected_k_drive, rel=1e-9)

        assert kin.bracket_material == mat_id
        assert kin.bracket_stiffness_n_per_mm == pytest.approx(k_bkt * 1e-3, rel=1e-6)
        assert kin.drive_stiffness_n_per_mm == pytest.approx(k_drive * 1e-3, rel=1e-6)

        k_bkts.append(k_bkt)
        k_drvs.append(k_drive)

        gdict = asm.to_geometry_dict()
        bkt_dict = gdict["gantry_brackets"]
        info = GANTRY_BRACKET_CATALOG[mat_id]
        assert bkt_dict == {
            "material_id": mat_id,
            "name": info["name"],
            "short_name": info["short_name"],
            "youngs_modulus_gpa": info["youngs_modulus_gpa"],
            "hdt_c": info["hdt_c"],
            "creep_tension_limit_n": info["creep_tension_limit_n"],
            "cnc_machined": info["cnc_machined"],
            "kit_cost_usd": info["kit_cost_usd"],
            "carriage_mass_delta_g": info["carriage_mass_delta_g"],
            "color_hex": info["color_hex"],
        }

    for i in range(len(ordered_materials) - 1):
        assert k_bkts[i + 1] > k_bkts[i]
        assert k_drvs[i + 1] > k_drvs[i]


def test_bracket_carriage_mass_delta_across_architectures() -> None:
    """CNC metal & CF polymers add exact carriage_mass_delta_g to Y moving mass & carriage nodes."""
    # CoreXY: splits 0.5*delta onto y_car_l & y_car_r, adds full delta to y_moving_mass_kg
    asm_abs = build_assembly(_base_corexy_spec(bracket_material="abs_printed"))
    asm_ppa = build_assembly(_base_corexy_spec(bracket_material="ppa_cf_printed"))
    asm_cnc = build_assembly(_base_corexy_spec(bracket_material="aluminum_6061_t6"))

    assert GANTRY_BRACKET_CATALOG["abs_printed"]["carriage_mass_delta_g"] == 0.0
    delta_ppa_kg = GANTRY_BRACKET_CATALOG["ppa_cf_printed"]["carriage_mass_delta_g"] * 1e-3
    delta_cnc_kg = GANTRY_BRACKET_CATALOG["aluminum_6061_t6"]["carriage_mass_delta_g"] * 1e-3

    assert (asm_ppa.y_moving_mass_kg - asm_abs.y_moving_mass_kg) == pytest.approx(delta_ppa_kg, rel=1e-9)
    assert (asm_cnc.y_moving_mass_kg - asm_abs.y_moving_mass_kg) == pytest.approx(delta_cnc_kg, rel=1e-9)
    assert (asm_cnc.total_mass_kg - asm_abs.total_mass_kg) == pytest.approx(delta_cnc_kg, rel=1e-9)

    for nid in ("y_car_l", "y_car_r"):
        assert (
            asm_cnc.node_map[nid].lumped_mass_kg - asm_abs.node_map[nid].lumped_mass_kg
        ) == pytest.approx(0.5 * delta_cnc_kg, rel=1e-9)

    # Cost delta matches kit_cost_usd delta
    expected_cost_delta = (
        GANTRY_BRACKET_CATALOG["aluminum_6061_t6"]["kit_cost_usd"]
        - GANTRY_BRACKET_CATALOG["abs_printed"]["kit_cost_usd"]
    )
    assert (asm_cnc.estimated_material_cost_usd - asm_abs.estimated_material_cost_usd) == pytest.approx(
        expected_cost_delta, rel=1e-9
    )

    # Cartesian & Delta also reflect carriage_mass_delta_g on y_moving_mass_kg & total_mass_kg
    for arch in ("cartesian_bedslinger", "delta"):
        s_a = PrinterSpec(architecture=arch)  # type: ignore[arg-type]
        if arch == "delta":
            s_a.kinematics.build_volume_mm = (200.0, 200.0, 250.0)
            s_a.kinematics.delta_arm_length_mm = 300.0
            s_a.kinematics.delta_radius_mm = 155.0
        s_a.gantry.bracket_material = "abs_printed"
        s_c = s_a.model_copy(deep=True)
        s_c.gantry.bracket_material = "aluminum_6061_t6"
        a_abs = build_assembly(s_a)
        a_cnc = build_assembly(s_c)
        assert (a_cnc.y_moving_mass_kg - a_abs.y_moving_mass_kg) == pytest.approx(delta_cnc_kg, rel=1e-9)
        assert (a_cnc.total_mass_kg - a_abs.total_mass_kg) == pytest.approx(delta_cnc_kg, rel=1e-9)


def test_bracket_creep_risk_flags_diagnostics_and_penalty() -> None:
    """Sustained pretension above bracket creep limit triggers flag, diagnostic, and -5 dynamics penalty."""
    # At 70 N pretension: ABS (45N) & PA12-CF (65N) creep; PET-CF15 (75N), PPA-CF (105N), PPS-CF (115N), 6061 (250N) pass
    spec_abs = _base_corexy_spec(
        drive_mode="awd",
        double_shear=True,
        pretension_n=70.0,
        bracket_material="abs_printed",
        enclosed_panels=True,
        panel_mounting="bolted_structural",
    )
    spec_abs.gantry.belt.type = "gt2_9mm"
    spec_abs.gantry.belt.core = "aramid"
    spec_abs.toolhead.mount_material = "abs_printed"

    rep_abs = evaluate_printer(spec_abs)
    assert rep_abs.kinematics.bracket_creep_risk is True
    assert rep_abs.kinematics.to_dict()["bracket_creep_risk"] is True
    creep_diags = [d for d in rep_abs.diagnostics if "creep" in d.lower()]
    # Only one creep warning when both bracket & toolhead mount are ABS (no duplicate ABS warning)
    assert len(creep_diags) == 1
    assert "abs_printed" in creep_diags[0]
    assert "70" in creep_diags[0]
    assert "45" in creep_diags[0]
    assert "ppa_cf_printed" in creep_diags[0]
    assert "pps_cf_printed" in creep_diags[0]
    assert "aluminum_6061_t6" in creep_diags[0]

    # PA12-CF creeps at 70 N (> 65 N limit), while PET-CF15 (75 N limit) clears bracket creep
    spec_pa12 = spec_abs.model_copy(deep=True)
    spec_pa12.gantry.bracket_material = "pa12_cf_printed"
    spec_pa12.toolhead.mount_material = "pa12_cf_printed"
    rep_pa12 = evaluate_printer(spec_pa12)
    assert rep_pa12.kinematics.bracket_creep_risk is True

    spec_pet = spec_abs.model_copy(deep=True)
    spec_pet.gantry.bracket_material = "pet_cf_printed"
    spec_pet.toolhead.mount_material = "pet_cf_printed"
    rep_pet = evaluate_printer(spec_pet)
    assert rep_pet.kinematics.bracket_creep_risk is False

    # Switching gantry bracket to Al 6061 while leaving toolhead mount as ABS keeps toolhead ABS warning
    spec_mixed = spec_abs.model_copy(deep=True)
    spec_mixed.gantry.bracket_material = "aluminum_6061_t6"
    spec_mixed.toolhead.mount_material = "abs_printed"
    rep_mixed = evaluate_printer(spec_mixed)
    assert rep_mixed.kinematics.bracket_creep_risk is False
    assert any("abs toolhead mount" in d.lower() for d in rep_mixed.diagnostics)
    assert rep_mixed.scores.dynamics > rep_abs.scores.dynamics + 5.0

    # Verify -5 dynamics penalty directly when comparing borderline creep vs non-creep (same stiffness bracket)
    spec_ok = spec_pa12.model_copy(deep=True)
    spec_ok.gantry.belt.pretension_n = 64.0  # just below 65 N
    spec_creep = spec_pa12.model_copy(deep=True)
    spec_creep.gantry.belt.pretension_n = 66.0  # just above 65 N
    rep_ok = evaluate_printer(spec_ok)
    rep_creep = evaluate_printer(spec_creep)
    assert rep_ok.kinematics.bracket_creep_risk is False
    assert rep_creep.kinematics.bracket_creep_risk is True
    assert rep_ok.scores.dynamics - rep_creep.scores.dynamics >= 4.5


def test_monolith_awd_preset_and_bracket_upgrade_jump() -> None:
    """Voron Monolith AWD uses aluminum_6061_t6 and jumps drive stiffness & shaper freq over ABS."""
    monolith = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_monolith_awd.yaml")
    assert monolith.gantry.bracket_material == "aluminum_6061_t6"

    s_abs = monolith.model_copy(deep=True)
    s_abs.gantry.bracket_material = "abs_printed"

    s_ppa = monolith.model_copy(deep=True)
    s_ppa.gantry.bracket_material = "ppa_cf_printed"

    rep_abs = evaluate_printer(s_abs)
    rep_ppa = evaluate_printer(s_ppa)
    rep_cnc = evaluate_printer(monolith)

    assert rep_abs.kinematics.bracket_creep_risk is True
    assert rep_ppa.kinematics.bracket_creep_risk is False
    assert rep_cnc.kinematics.bracket_creep_risk is False

    # Drive stiffness & effective frequency jump significantly over ABS
    assert rep_ppa.kinematics.drive_stiffness_n_per_mm > rep_abs.kinematics.drive_stiffness_n_per_mm * 1.6
    assert rep_cnc.kinematics.drive_stiffness_n_per_mm > rep_abs.kinematics.drive_stiffness_n_per_mm * 1.9
    assert rep_ppa.modal_fea.shaper_x is not None and rep_abs.modal_fea.shaper_x is not None
    assert rep_cnc.modal_fea.shaper_x is not None and rep_cnc.modal_fea.shaper_y is not None
    assert rep_ppa.modal_fea.shaper_x.effective_freq_hz > rep_abs.modal_fea.shaper_x.effective_freq_hz + 12.0
    assert rep_cnc.modal_fea.shaper_x.effective_freq_hz > rep_abs.modal_fea.shaper_x.effective_freq_hz + 15.0
    assert rep_cnc.modal_fea.shaper_y.effective_freq_hz > rep_abs.modal_fea.shaper_y.effective_freq_hz + 25.0
    assert rep_cnc.modal_fea.recommended_max_accel_mm_s2 > rep_abs.modal_fea.recommended_max_accel_mm_s2 * 1.6


# ---------------------------------------------------------------------------
# 2. Frame Extrusion Cavity Fillings: Mass, Stiffness, Damping, & PSD
# ---------------------------------------------------------------------------


def test_extrusion_filling_cavity_mass_cost_and_moving_gantry_exclusion() -> None:
    """Verify exact A_cav*L*rho mass, zero moving X-beam fill, and exact filler cost without double-charge."""
    spec_none = _base_corexy_spec(extrusion_filling="none")
    asm_none = build_assembly(spec_none)
    nmap_none = asm_none.node_map

    # Verify exact 2020 cavity geometry: A_cav = 0.58 * (20*20 - 162) mm^2 = 138.04 mm^2
    prof_2020 = get_profile("extrusion_2020")
    a_cav_2020 = 0.58 * (prof_2020.width_m * prof_2020.height_m - prof_2020.area_m2)
    assert a_cav_2020 * 1e6 == pytest.approx(138.04, rel=1e-4)

    # Stationary extrusions on CoreXY (frame + rail); moving X-beam (role="gantry") + braces must NOT be filled
    stationary_extr_len_m = sum(
        m.length_m(nmap_none)
        for m in asm_none.members
        if m.profile.shape == "extrusion" and m.role in ("frame", "rail", "tower")
    )
    assert stationary_extr_len_m == pytest.approx(
        4 * 0.46 + 4 * 0.50 + 4 * 0.46 + 2 * 0.46 + 0.46, rel=1e-6
    )

    for fill_id, fill_spec in EXTRUSION_FILLINGS.items():
        spec = _base_corexy_spec(extrusion_filling=fill_id)
        asm = build_assembly(spec)
        nmap = asm.node_map

        expected_mass_kg = float(fill_spec["density_kg_m3"]) * a_cav_2020 * stationary_extr_len_m
        assert asm.extrusion_fill_mass_kg == pytest.approx(expected_mass_kg, rel=1e-6)
        assert (asm.frame_mass_kg - asm_none.frame_mass_kg) == pytest.approx(expected_mass_kg, rel=1e-6)
        assert (asm.total_mass_kg - asm_none.total_mass_kg) == pytest.approx(expected_mass_kg, rel=1e-6)

        # Moving X-beam (role == "gantry") is NEVER filled, so x/y moving mass stays identical
        assert asm.x_moving_mass_kg == pytest.approx(asm_none.x_moving_mass_kg, rel=1e-9)
        assert asm.y_moving_mass_kg == pytest.approx(asm_none.y_moving_mass_kg, rel=1e-9)
        for m_g, m_g0 in zip(
            [m for m in asm.members if m.role == "gantry"],
            [m for m in asm_none.members if m.role == "gantry"],
            strict=True,
        ):
            assert m_g.fill_mass_kg == 0.0
            assert m_g.mass_kg(nmap) == pytest.approx(m_g0.mass_kg(nmap_none), rel=1e-9)
            assert m_g.material.density_kg_m3 == pytest.approx(m_g0.material.density_kg_m3, rel=1e-9)
            assert m_g.profile.iy_m4 == pytest.approx(m_g0.profile.iy_m4, rel=1e-12)

        # Exact filler cost (no double-charging aluminum $/kg on boosted density)
        expected_cost_delta = expected_mass_kg * float(fill_spec["cost_per_kg_usd"])
        assert (asm.estimated_material_cost_usd - asm_none.estimated_material_cost_usd) == pytest.approx(
            expected_cost_delta, rel=1e-6
        )

    # Delta towers (role="tower") are stationary extrusions and ARE filled
    s_delta_0 = PrinterSpec(architecture="delta")
    s_delta_0.kinematics.build_volume_mm = (200.0, 200.0, 250.0)
    s_delta_0.kinematics.delta_arm_length_mm = 300.0
    s_delta_0.kinematics.delta_radius_mm = 155.0
    s_delta_fill = s_delta_0.model_copy(deep=True)
    s_delta_fill.frame.extrusion_filling = "epoxy_granite"
    a_d0 = build_assembly(s_delta_0)
    a_df = build_assembly(s_delta_fill)
    delta_stat_len = sum(
        m.length_m(a_d0.node_map)
        for m in a_d0.members
        if m.profile.shape == "extrusion" and m.role in ("frame", "rail", "tower")
    )
    assert a_df.extrusion_fill_mass_kg == pytest.approx(2250.0 * a_cav_2020 * delta_stat_len, rel=1e-6)


def test_bonded_vs_unbonded_stiffness_and_monotonic_damping() -> None:
    """Bonded fills boost bending EI & static stiffness; unbonded add zero stiffness; damping & PSD increase monotonically."""
    damping_order = ["none", "pu_foam", "silica_sand", "uhpc_grout", "epoxy_granite", "lead_shot"]
    assert set(damping_order) == set(EXTRUSION_FILLINGS.keys())

    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    asms = {}
    statics = {}
    modals = {}
    for fid in damping_order:
        s = base.model_copy(deep=True)
        s.frame.extrusion_filling = fid  # type: ignore[assignment]
        asm = build_assembly(s)
        asms[fid] = asm
        statics[fid] = solve_static_fea(asm)
        modals[fid] = solve_modal_fea(asm)

    # 1. Non-bonded fills add ZERO bending/torsional stiffness to frame profiles or static toolhead stiffness
    prof_ref = next(m for m in asms["none"].members if m.id == "col_fl_bot").profile
    for unbonded_id in ("pu_foam", "silica_sand", "lead_shot"):
        m_unb = next(m for m in asms[unbonded_id].members if m.id == "col_fl_bot")
        assert m_unb.profile.iy_m4 == pytest.approx(prof_ref.iy_m4, rel=1e-12)
        assert m_unb.profile.iz_m4 == pytest.approx(prof_ref.iz_m4, rel=1e-12)
        assert m_unb.profile.j_m4 == pytest.approx(prof_ref.j_m4, rel=1e-12)
        assert statics[unbonded_id].toolhead_stiffness_x_n_per_mm == pytest.approx(
            statics["none"].toolhead_stiffness_x_n_per_mm, rel=1e-6
        )
        assert statics[unbonded_id].toolhead_stiffness_y_n_per_mm == pytest.approx(
            statics["none"].toolhead_stiffness_y_n_per_mm, rel=1e-6
        )

    # 2. Bonded fills (epoxy_granite 30 GPa < uhpc_grout 40 GPa) strictly increase static stiffness & cut deflection
    for bonded_id in ("epoxy_granite", "uhpc_grout"):
        m_bnd = next(m for m in asms[bonded_id].members if m.id == "col_fl_bot")
        assert m_bnd.profile.iy_m4 > prof_ref.iy_m4 * 1.10
        assert m_bnd.profile.iz_m4 > prof_ref.iz_m4 * 1.10
        assert m_bnd.profile.j_m4 > prof_ref.j_m4 * 1.10
        assert statics[bonded_id].toolhead_stiffness_x_n_per_mm > statics["none"].toolhead_stiffness_x_n_per_mm * 1.05
        # In Y on Trident, ~72% of structural-loop compliance lies in the unfilled moving X-gantry + toolhead
        assert statics[bonded_id].toolhead_stiffness_y_n_per_mm > statics["none"].toolhead_stiffness_y_n_per_mm * 1.015
        assert statics[bonded_id].combined_max_deflection_um < statics["none"].combined_max_deflection_um

    assert statics["uhpc_grout"].toolhead_stiffness_x_n_per_mm > statics["epoxy_granite"].toolhead_stiffness_x_n_per_mm
    assert statics["uhpc_grout"].combined_max_deflection_um < statics["epoxy_granite"].combined_max_deflection_um

    # Bonded cavity filling also stacks cleanly with bolted structural panels (doesn't clamp or wipe out)
    s_panel_base = _base_corexy_spec(enclosed_panels=True, panel_mounting="bolted_structural", extrusion_filling="none")
    s_panel_uhpc = _base_corexy_spec(enclosed_panels=True, panel_mounting="bolted_structural", extrusion_filling="uhpc_grout")
    st_pb = solve_static_fea(build_assembly(s_panel_base))
    st_pu = solve_static_fea(build_assembly(s_panel_uhpc))
    assert st_pu.toolhead_stiffness_x_n_per_mm > st_pb.toolhead_stiffness_x_n_per_mm

    # 3. Structural damping ratio (frame_damping_ratio & Klipper PSD measured damping_ratio) strictly monotonic
    for i in range(len(damping_order) - 1):
        curr_id = damping_order[i]
        next_id = damping_order[i + 1]
        assert asms[next_id].frame_damping_ratio > asms[curr_id].frame_damping_ratio
        assert modals[next_id].frame_damping_ratio > modals[curr_id].frame_damping_ratio
        assert modals[next_id].shaper_graph_x is not None and modals[curr_id].shaper_graph_x is not None
        assert modals[next_id].shaper_graph_y is not None and modals[curr_id].shaper_graph_y is not None
        # Allow minor ~0.003 discrete 2.0 Hz PSD bin-grid quantization variance on adjacent 0.002 zeta steps
        assert modals[next_id].shaper_graph_x.damping_ratio >= modals[curr_id].shaper_graph_x.damping_ratio - 0.003
        assert modals[next_id].shaper_graph_y.damping_ratio >= modals[curr_id].shaper_graph_y.damping_ratio - 0.003
    assert modals["lead_shot"].shaper_graph_x.damping_ratio > modals["epoxy_granite"].shaper_graph_x.damping_ratio > modals["silica_sand"].shaper_graph_x.damping_ratio > modals["pu_foam"].shaper_graph_x.damping_ratio > modals["none"].shaper_graph_x.damping_ratio

    # CNC machined brackets reduce structural damping vs printed polymer brackets (-0.003 vs +0.004)
    asm_cnc_none = build_assembly(_base_corexy_spec(bracket_material="aluminum_6061_t6", extrusion_filling="none"))
    assert asm_cnc_none.frame_damping_ratio == pytest.approx(0.039, abs=1e-6)
    assert asms["none"].frame_damping_ratio == pytest.approx(0.046, abs=1e-6)

    # Geometry dict and ModalResult dict expose exact damping & filling metadata
    gdict = asms["epoxy_granite"].to_geometry_dict()
    assert gdict["extrusion_filling"]["id"] == "epoxy_granite"
    assert gdict["extrusion_filling"]["bonded"] is True
    assert gdict["extrusion_filling"]["added_mass_kg"] == pytest.approx(asms["epoxy_granite"].extrusion_fill_mass_kg, abs=1e-3)
    assert gdict["extrusion_filling"]["damping_ratio"] == pytest.approx(0.082, abs=1e-4)
    assert modals["epoxy_granite"].to_dict()["frame_damping_ratio"] == pytest.approx(0.082, abs=1e-4)

    # Backward compatibility: calibrate_axis_shapers without psd_damping_ratio defaults to 0.055
    g_default = calibrate_axis_shapers("X", 85.0)
    g_explicit = calibrate_axis_shapers("X", 85.0, psd_damping_ratio=0.055)
    assert g_default.damping_ratio == pytest.approx(g_explicit.damping_ratio, rel=1e-9)


# ---------------------------------------------------------------------------
# 3. BOM & FastAPI Integration
# ---------------------------------------------------------------------------


def test_bom_includes_extrusion_filling_and_gantry_brackets() -> None:
    """BOM emits EXT-FILL-* only when filled and GNT-BKT-* for gantry bracket kit with accurate mass/cost."""
    spec_base = _base_corexy_spec(bracket_material="abs_printed", extrusion_filling="none")
    bom_base = generate_bom(spec_base)
    assert not any(i.sku.startswith("EXT-FILL-") for i in bom_base.items)

    bkt_base = next(i for i in bom_base.items if i.sku == "GNT-BKT-ABS-PRINTED")
    assert bkt_base.category.lower() == "gantry"
    assert bkt_base.total_cost_usd == pytest.approx(28.0, abs=0.01)
    assert bkt_base.total_mass_g > 100.0

    # With silica_sand + ppa_cf_printed
    spec_up = _base_corexy_spec(bracket_material="ppa_cf_printed", extrusion_filling="epoxy_granite")
    asm_up = build_assembly(spec_up)
    bom_up = generate_bom(asm_up)

    fill_item = next(i for i in bom_up.items if i.sku == "EXT-FILL-EPOXY-GRANITE")
    assert fill_item.category.lower() == "frame"
    assert fill_item.unit == "kg"
    assert fill_item.quantity == pytest.approx(asm_up.extrusion_fill_mass_kg, abs=1e-3)
    assert fill_item.total_mass_g == pytest.approx(asm_up.extrusion_fill_mass_kg * 1000.0, abs=1.0)
    assert fill_item.total_cost_usd == pytest.approx(
        asm_up.extrusion_fill_mass_kg * EXTRUSION_FILLINGS["epoxy_granite"]["cost_per_kg_usd"],
        abs=0.05,
    )

    bkt_up = next(i for i in bom_up.items if i.sku == "GNT-BKT-PPA-CF-PRINTED")
    assert bkt_up.category.lower() == "gantry"
    assert "PPA-CF" in bkt_up.name
    assert bkt_up.total_cost_usd == pytest.approx(GANTRY_BRACKET_CATALOG["ppa_cf_printed"]["kit_cost_usd"], abs=0.01)
    assert bkt_up.total_mass_g > bkt_base.total_mass_g

    assert sum(bom_up.category_costs_usd.values()) == pytest.approx(bom_up.total_cost_usd, abs=0.1)
    assert bom_up.total_mass_kg > bom_base.total_mass_kg + asm_up.extrusion_fill_mass_kg - 0.01
    assert bom_up.total_cost_usd > bom_base.total_cost_usd + 70.0


def test_api_catalog_and_simulate_expose_fillings_and_brackets() -> None:
    """FastAPI /api/catalog and /api/simulate expose extrusion_fillings & gantry_brackets cleanly."""
    client = TestClient(create_app())

    r_cat = client.get("/api/catalog")
    assert r_cat.status_code == 200
    cat = r_cat.json()
    assert len(cat["extrusion_fillings"]) == 6
    assert len(cat["gantry_brackets"]) == 7

    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_monolith_awd.yaml")
    spec.frame.extrusion_filling = "uhpc_grout"
    r_sim = client.post("/api/simulate", json={"spec": spec.model_dump(mode="json")})
    assert r_sim.status_code == 200
    sim = r_sim.json()

    assert sim["geometry"]["extrusion_filling"]["id"] == "uhpc_grout"
    assert sim["geometry"]["extrusion_filling"]["bonded"] is True
    assert sim["geometry"]["gantry_brackets"]["material_id"] == "aluminum_6061_t6"
    assert sim["modal_fea"]["frame_damping_ratio"] == pytest.approx(0.069, abs=1e-4)
    assert sim["kinematics"]["bracket_material"] == "aluminum_6061_t6"
    assert sim["kinematics"]["bracket_stiffness_n_per_mm"] > 5000.0
    assert sim["kinematics"]["bracket_creep_risk"] is False
