"""Tests for SLM/LPBF printed metal material data, PrintDB hydration, and SLM vs CNC physics."""

from __future__ import annotations

import numpy as np
import pytest

from printsim.engine import evaluate_printer
from printsim.fea.components import compute_bearing_cte_clearance_um
from printsim.fea.solid_fem import (
    constitutive_matrix_3d,
    generate_parametric_part_grid,
    solve_solid_voxel_fea,
)
from printsim.geometry import build_assembly
from printsim.geometry._builder_toolhead import compute_toolhead_link_props
from printsim.kinematics.solvers import compute_bracket_stiffness_n_per_m
from printsim.schema import (
    GANTRY_BRACKET_CATALOG,
    MATERIAL_CATALOG,
    PrinterSpec,
    get_material,
)
from printsim.schema.overrides import resolve_gantry_bracket
from printsim.schema.printdb_bridge import get_record_specs, is_printdb_available


SLM_MATERIAL_IDS = (
    "slm_aluminum",
    "slm_ti6al4v",
    "slm_316l_steel",
    "slm_17_4ph_steel",
    "slm_maraging_steel",
    "slm_cucrzr",
)


def test_slm_materials_exist_with_physical_metadata() -> None:
    """All 6 SLM printed metals carry LPBF roughness, porosity, Z anisotropy, and bolted contact factors."""
    for mid in SLM_MATERIAL_IDS:
        mat = get_material(mid)
        assert mat.is_slm_metal is True
        assert mat.is_printed_polymer is False
        assert 0.90 <= mat.e3_ratio < 1.0
        assert 0.88 <= mat.z_strength_ratio < 1.0
        assert 0.90 <= mat.bolted_joint_stiffness_factor < 1.0
        assert mat.surface_roughness_ra_um is not None and mat.surface_roughness_ra_um >= 6.0
        assert mat.porosity_pct > 0.0

    # CNC wrought 6061-T6 remains isotropic with precision machined finish
    cnc = get_material("aluminum_6061_t6")
    assert cnc.is_slm_metal is False
    assert cnc.e3_ratio == pytest.approx(1.0)
    assert cnc.z_strength_ratio == pytest.approx(1.0)
    assert cnc.bolted_joint_stiffness_factor == pytest.approx(1.0)
    assert cnc.surface_roughness_ra_um is not None and cnc.surface_roughness_ra_um < 2.0
    assert cnc.porosity_pct == pytest.approx(0.0)


def test_slm_alsi10mg_differs_from_cnc_6061_t6_physically() -> None:
    """SLM AlSi10Mg differs from CNC 6061-T6 in density, modulus, yield, k, CTE, roughness, and damping."""
    slm = get_material("slm_aluminum")
    cnc = get_material("aluminum_6061_t6")

    assert slm.density_kg_m3 < cnc.density_kg_m3  # 2670 vs 2700 kg/m^3
    assert slm.youngs_modulus_pa < cnc.youngs_modulus_pa  # 68.0 vs 68.9 GPa
    assert slm.yield_strength_pa < cnc.yield_strength_pa  # 240 vs 276 MPa
    assert slm.thermal_conductivity_w_mk < cnc.thermal_conductivity_w_mk  # 140 vs 167 W/(m*K)
    assert slm.thermal_expansion_1_k < cnc.thermal_expansion_1_k  # 20.5 vs 23.6 ppm/K
    assert slm.damping_ratio > cnc.damping_ratio  # 0.015 vs 0.012
    assert slm.bolted_joint_stiffness_factor < cnc.bolted_joint_stiffness_factor


def test_printdb_hydrates_all_slm_metals_and_reference() -> None:
    """When PrintDB is available, all 6 SLM materials and the SLM vs CNC reference resolve cleanly."""
    if not is_printdb_available():
        pytest.skip("PrintDB not installed in this environment")

    mapping = {
        "slm_aluminum": "materials/slm-alsi10mg-aluminum",
        "slm_ti6al4v": "materials/slm-ti6al4v-titanium",
        "slm_316l_steel": "materials/slm-316l-stainless",
        "slm_17_4ph_steel": "materials/slm-17-4ph-stainless",
        "slm_maraging_steel": "materials/slm-maraging-steel-ms1",
        "slm_cucrzr": "materials/slm-cucrzr-copper",
    }
    for key, rid in mapping.items():
        si = get_record_specs(rid, si=True)
        assert si is not None, f"Missing PrintDB record {rid}"
        mat = MATERIAL_CATALOG[key]
        assert mat.youngs_modulus_pa == pytest.approx(float(si["tensile_modulus_pa"]), rel=1e-6)
        assert mat.density_kg_m3 == pytest.approx(float(si["density_kg_per_m3"]), rel=1e-6)
        assert mat.yield_strength_pa == pytest.approx(float(si["yield_strength_pa"]), rel=1e-6)
        assert mat.thermal_expansion_1_k == pytest.approx(float(si["cte_per_k"]), rel=1e-6)
        assert mat.thermal_conductivity_w_mk == pytest.approx(
            float(si["thermal_conductivity_w_per_mk"]), rel=1e-6
        )

    # Verify the SLM vs CNC physics reference record exists in PrintDB
    from printsim.schema.printdb_bridge import get_printdb

    db = get_printdb()
    assert db is not None
    ref = db.get("references/materials_mechanics/slm-lpbf-metals-vs-cnc-billet")
    assert ref.record_type == "reference"


def test_slm_vs_cnc_bracket_stiffness_mass_and_damping() -> None:
    """SLM AlSi10Mg brackets have lower bolted contact stiffness, lighter mass, and slightly higher damping than CNC 6061."""
    spec_slm = PrinterSpec(architecture="corexy")
    spec_slm.gantry.bracket_material = "slm_aluminum"
    spec_slm.gantry.drive_mode = "awd"

    spec_cnc = spec_slm.model_copy(deep=True)
    spec_cnc.gantry.bracket_material = "aluminum_6061_t6"

    asm_slm = build_assembly(spec_slm)
    asm_cnc = build_assembly(spec_cnc)

    k_slm = compute_bracket_stiffness_n_per_m(asm_slm)
    k_cnc = compute_bracket_stiffness_n_per_m(asm_cnc)
    assert k_slm < k_cnc  # 68 * 0.92 = 62.56 GPa eff vs 68.9 GPa CNC

    # SLM hollow topology saves both moving carriage mass and stationary motor-mount mass
    assert asm_slm.y_moving_mass_kg < asm_cnc.y_moving_mass_kg
    assert asm_slm.total_mass_kg < asm_cnc.total_mass_kg

    # SLM damping sits between CNC billet (-0.003) and polymer (+0.004)
    res_slm = resolve_gantry_bracket(spec_slm)
    res_cnc = resolve_gantry_bracket(spec_cnc)
    assert res_slm["damping_delta"] == pytest.approx(-0.0015)
    assert res_cnc["damping_delta"] == pytest.approx(-0.003)
    assert res_slm["damping_delta"] > res_cnc["damping_delta"]


def test_slm_bearing_pressfit_cte_advantage_over_cnc_6061() -> None:
    """SLM AlSi10Mg (20.5 ppm/K) resists F695 press-fit loosening deeper into chamber heating than CNC 6061 (23.6 ppm/K)."""
    spec_cnc = PrinterSpec(architecture="corexy")
    spec_cnc.gantry.bracket_material = "aluminum_6061_t6"
    spec_cnc.gantry.idler_axial_preload = "none"

    spec_slm_al = spec_cnc.model_copy(deep=True)
    spec_slm_al.gantry.bracket_material = "slm_aluminum"

    # At 75 C: CNC 6061 (>66 C) already has radial clearance loss, while SLM AlSi10Mg (~81.2 C threshold) does NOT
    clr_cnc_75, warn_cnc_75 = compute_bearing_cte_clearance_um(spec_cnc, 75.0)
    clr_slm_75, warn_slm_75 = compute_bearing_cte_clearance_um(spec_slm_al, 75.0)
    assert warn_cnc_75 is True and clr_cnc_75 > 0.5
    assert warn_slm_75 is False and clr_slm_75 == pytest.approx(0.0)

    # At 95 C: both loosen without disc springs, but SLM AlSi10Mg opens significantly less than CNC 6061
    clr_cnc_95, warn_cnc_95 = compute_bearing_cte_clearance_um(spec_cnc, 95.0)
    clr_slm_95, warn_slm_95 = compute_bearing_cte_clearance_um(spec_slm_al, 95.0)
    assert warn_cnc_95 is True and warn_slm_95 is True
    assert 0.0 < clr_slm_95 < clr_cnc_95 * 0.5

    # Low-CTE SLM metals (Ti-6Al-4V, 17-4PH, MS1, 316L) have zero press-fit loss even at 100 C
    for mat_id in ("slm_ti6al4v", "slm_17_4ph_steel", "slm_maraging_steel", "slm_316l_steel"):
        s = spec_cnc.model_copy(deep=True)
        s.gantry.bracket_material = mat_id  # type: ignore[assignment]
        clr, warn = compute_bearing_cte_clearance_um(s, 100.0)
        assert warn is False and clr == pytest.approx(0.0)


def test_slm_solid_fea_mild_z_anisotropy_and_yield() -> None:
    """Solid Hex8 FEA applies mild LPBF Z-axis anisotropy & z_strength_ratio for SLM metals vs isotropic CNC."""
    slm_mat = get_material("slm_aluminum")
    D_slm = constitutive_matrix_3d(slm_mat, fdm_anisotropic=True)
    assert np.all(np.linalg.eigvalsh(D_slm) > 1e8)
    assert D_slm[2, 2] < D_slm[0, 0]  # Z build axis slightly softer than XY
    assert D_slm[2, 2] / D_slm[0, 0] > 0.90  # Mild LPBF anisotropy (unlike FDM polymers)

    grid, h, fixed_face, load_face, _ = generate_parametric_part_grid("cantilever_bar")
    res_slm = solve_solid_voxel_fea(
        grid, h, material_id="slm_aluminum", fixed_face=fixed_face, load_face=load_face, force_n=(0.0, 0.0, -500.0)
    )
    res_cnc = solve_solid_voxel_fea(
        grid, h, material_id="aluminum_6061_t6", fixed_face=fixed_face, load_face=load_face, force_n=(0.0, 0.0, -500.0)
    )

    assert res_slm.is_fdm_anisotropic is True
    assert res_cnc.is_fdm_anisotropic is False
    assert res_slm.z_layer_strength_mpa == pytest.approx(240.0 * 0.90, rel=1e-4)
    assert res_cnc.z_layer_strength_mpa == pytest.approx(276.0, rel=1e-4)
    assert res_slm.mass_g < res_cnc.mass_g  # 2670 vs 2700 kg/m^3
    assert res_slm.stiffness_n_per_mm < res_cnc.stiffness_n_per_mm


def test_slm_toolhead_bolted_interface_vs_cnc_and_full_eval() -> None:
    """SLM toolhead carriage/clamp bolted interfaces show mild contact compliance vs CNC and evaluate cleanly."""
    spec_slm = PrinterSpec(architecture="corexy")
    spec_slm.gantry.bracket_material = "slm_ti6al4v"
    spec_slm.toolhead.mount_material = "slm_aluminum"

    spec_cnc = spec_slm.model_copy(deep=True)
    spec_cnc.toolhead.mount_material = "aluminum_6061_t6"

    props_slm = compute_toolhead_link_props(spec_slm, 0.025, 0.040)
    props_cnc = compute_toolhead_link_props(spec_cnc, 0.025, 0.040)
    assert props_slm["k_backplate_only_n_per_mm"] < props_cnc["k_backplate_only_n_per_mm"]
    assert props_slm["k_hotend_clamp_n_per_mm"] < props_cnc["k_hotend_clamp_n_per_mm"]

    # Full multiphysics evaluation runs cleanly for all new SLM bracket materials
    for mat_id in ("slm_aluminum", "slm_ti6al4v", "slm_316l_steel", "slm_17_4ph_steel", "slm_maraging_steel"):
        assert mat_id in GANTRY_BRACKET_CATALOG
        s = PrinterSpec(architecture="corexy")
        s.gantry.bracket_material = mat_id  # type: ignore[assignment]
        rep = evaluate_printer(s)
        assert rep.modal_fea.shaper_x.effective_freq_hz > 25.0
        assert rep.kinematics.bracket_creep_risk is False
