"""Analytical & 3D Tet10 FEA parity tests for milled/pocketed linear rails."""

from __future__ import annotations

from pathlib import Path

import pytest

from printsim.cad_assets.registry import (
    get_hardware_asset_manifest,
    get_visual_glb_path,
    list_hardware_assets,
)
from printsim.engine import evaluate_printer
from printsim.fea.beam3d import beam_shear_coefficients
from printsim.fea.components import (
    hertzian_rail_carriage_stiffness,
    newmark_rail_composite_gamma,
)
from printsim.geometry.bom import generate_bom
from printsim.geometry.builder import build_assembly
from printsim.schema import PrinterSpec, get_material, get_profile

MILLED_RAIL_IDS = (
    "mgn12_hollow_rail",
    "mgn12_pocketed_rail",
    "mgn9_pocketed_rail",
    "mgn15_pocketed_rail",
)

RHO_STEEL = 7850.0


# ---------------------------------------------------------------------------
# 1. Profile Catalog, Tet10 Section Properties & Visual GLB Asset Bundles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("profile_id", MILLED_RAIL_IDS)
def test_milled_rail_profile_and_visual_glb_present(profile_id: str) -> None:
    prof = get_profile(profile_id)
    assert prof.shape == "rail"
    assert prof.area_m2 > 0.0
    assert prof.iy_m4 > 0.0
    assert prof.iz_m4 > 0.0
    assert prof.j_m4 > 0.0
    assert prof.shear_kappa_y is not None and 0.10 < prof.shear_kappa_y < 0.85
    assert prof.shear_kappa_z is not None and 0.30 < prof.shear_kappa_z <= 0.90

    glb = get_visual_glb_path("x_beams", profile_id)
    assert glb is not None and glb.is_file()
    assert glb.read_bytes()[:4] == b"glTF"

    man = get_hardware_asset_manifest("x_beams", profile_id)
    assert man is not None
    assert man["visual_url"] == f"/api/assets/hardware/x_beams/{profile_id}/visual.glb"
    assert float(man["length_mm"]) >= 300.0
    assert int(man["visual_triangles"]) > 1000
    fea_sec = man["fea_section_properties"]
    assert float(fea_sec["min_raceway_wall_mm"]) >= 1.25

    catalog = list_hardware_assets()
    assert "x_beams" in catalog
    assert profile_id in catalog["x_beams"]


def test_milled_rail_mass_and_stiffness_hierarchy() -> None:
    """Verify Tet10-calibrated mass, bending, open-section torsion, and shear monotonicity."""
    mgn9_s = get_profile("mgn9_rail")
    mgn9_p = get_profile("mgn9_pocketed_rail")
    mgn12_s = get_profile("mgn12_rail")
    mgn12_h = get_profile("mgn12_hollow_rail")
    mgn12_p = get_profile("mgn12_pocketed_rail")
    mgn15_s = get_profile("mgn15_rail")
    mgn15_p = get_profile("mgn15_pocketed_rail")

    # Mass targets: Monolith/ULS hollow MGN12 ~440 g/m; pocketed variants save 20-35% mass vs solid
    assert mgn12_h.area_m2 * RHO_STEEL * 1e3 == pytest.approx(440.9, abs=3.0)
    assert 0.70 < (mgn9_p.area_m2 / mgn9_s.area_m2) < 0.85
    assert 0.70 < (mgn12_p.area_m2 / mgn12_s.area_m2) < 0.80
    assert 0.65 < (mgn15_p.area_m2 / mgn15_s.area_m2) < 0.75
    assert mgn12_h.area_m2 < mgn12_p.area_m2 < mgn12_s.area_m2

    # Bending monotonicity: milled rails preserve outer raceway envelope but lose core Iy/Iz
    for milled, solid in ((mgn9_p, mgn9_s), (mgn12_h, mgn12_s), (mgn12_p, mgn12_s), (mgn15_p, mgn15_s)):
        assert milled.iy_m4 < solid.iy_m4
        assert milled.iz_m4 < solid.iz_m4
        assert milled.j_m4 < solid.j_m4

    # Open-section C-channel torsion physics: continuous hollow MGN12 loses ~85% J vs solid,
    # whereas periodic bolt bridges in mgn12_pocketed_rail deliver >2x higher bare torsional J
    # and >1.7x higher lateral shear factor kappa_y.
    assert mgn12_p.j_m4 > 2.0 * mgn12_h.j_m4
    assert mgn12_h.j_m4 < 0.20 * mgn12_s.j_m4
    assert mgn12_p.shear_kappa_y is not None and mgn12_h.shear_kappa_y is not None
    assert mgn12_p.shear_kappa_y > 1.7 * mgn12_h.shear_kappa_y


# ---------------------------------------------------------------------------
# 2. Direct Gothic-Arch Raceway Backing Compliance (Mid-Pocket vs Bridge & Preload Scaling)
# ---------------------------------------------------------------------------


def test_tet10_raceway_backing_compliance_anisotropy_and_positions() -> None:
    """Anisotropic series backing compliance replaces flat 0.92x scalar across all 4 milled rails."""
    c12_s = hertzian_rail_carriage_stiffness("mgn12_rail", preload="Z1", block_type="H")
    c12_h = hertzian_rail_carriage_stiffness("mgn12_hollow_rail", preload="Z1", block_type="H")
    c12_p = hertzian_rail_carriage_stiffness("mgn12_pocketed_rail", preload="Z1", block_type="H")

    # Block mass unchanged
    assert c12_h["block_mass_g"] == pytest.approx(c12_s["block_mass_g"])
    assert c12_p["block_mass_g"] == pytest.approx(c12_s["block_mass_g"])

    # Continuous C-channel hollow MGN12 suffers strong lateral portal-frame sway (eta_yy ~ 0.79-0.86)
    # much softer than vertical k_zz (~0.96-0.98) and roll k_rx (~0.96-0.98)
    assert c12_h["raceway_backing_eta_yy"] < c12_h["raceway_backing_eta_zz"]
    assert c12_h["raceway_backing_eta_yy"] < c12_h["raceway_backing_eta_rx"]

    # Periodic pocketed MGN12 bolt bridges keep lateral raceway stiffness substantially stiffer than hollow
    assert c12_p["k_yy_n_m"] > c12_h["k_yy_n_m"] * 1.10
    assert c12_p["k_rx_nm_rad"] > c12_h["k_rx_nm_rad"]
    assert c12_p["k_yy_n_m"] < c12_s["k_yy_n_m"]

    # Mid-pocket vs over-bridge position query works and captures MGN15 long-pocket ripple
    c15_mid = hertzian_rail_carriage_stiffness(
        "mgn15_pocketed_rail", preload="Z1", block_type="H", raceway_position="mid_pocket"
    )
    c15_brg = hertzian_rail_carriage_stiffness(
        "mgn15_pocketed_rail", preload="Z1", block_type="H", raceway_position="over_bridge"
    )
    c15_avg = hertzian_rail_carriage_stiffness(
        "mgn15_pocketed_rail", preload="Z1", block_type="H", raceway_position="travel_avg"
    )
    assert c15_mid["k_yy_n_m"] < c15_avg["k_yy_n_m"] < c15_brg["k_yy_n_m"]
    assert c15_mid["k_rx_nm_rad"] < c15_avg["k_rx_nm_rad"] < c15_brg["k_rx_nm_rad"]

    # Series physics: heavier preload (ZB) has stiffer Hertzian contact, so fixed steel backing
    # compliance causes a larger fractional knockdown eta than light preload (ZF).
    c12_h_zf = hertzian_rail_carriage_stiffness("mgn12_hollow_rail", preload="ZF", block_type="H")
    c12_h_zb = hertzian_rail_carriage_stiffness("mgn12_hollow_rail", preload="ZB", block_type="H")
    assert c12_h_zb["raceway_backing_eta_yy"] < c12_h_zf["raceway_backing_eta_yy"]


# ---------------------------------------------------------------------------
# 3. Bolted Composite Newmark Coupling vs Bare Unsupported Rail FEA
# ---------------------------------------------------------------------------


def test_newmark_composite_coupling_hollow_vs_pocketed_vs_solid() -> None:
    """Bolted composite gamma captures hollow bolt-pad k_s reduction vs preserved pocketed bridges."""
    e_rail = get_material("steel_4140").youngs_modulus_pa
    cf_mat = get_material("carbon_fiber_torsional_45")
    cf_prof = get_profile("cf_tube_20x20_carbonara")

    p_sol = get_profile("mgn12_rail")
    p_hol = get_profile("mgn12_hollow_rail")
    p_poc = get_profile("mgn12_pocketed_rail")

    g_sol = newmark_rail_composite_gamma(
        e_rail, p_sol.area_m2, cf_mat.youngs_modulus_pa, cf_prof.area_m2,
        0.025, 0.35, "carbon_tube", cf_prof.wall_thickness_m, 1.0,
        "two_wall_clamped_inserts", rail_profile_id="mgn12_rail",
    )
    g_poc = newmark_rail_composite_gamma(
        e_rail, p_poc.area_m2, cf_mat.youngs_modulus_pa, cf_prof.area_m2,
        0.025, 0.35, "carbon_tube", cf_prof.wall_thickness_m, 1.0,
        "two_wall_clamped_inserts", rail_profile_id="mgn12_pocketed_rail",
    )
    # Same area comparison isolating hollow's 0.90x bolt-pad k_s knockdown
    g_hol_same_a = newmark_rail_composite_gamma(
        e_rail, p_poc.area_m2, cf_mat.youngs_modulus_pa, cf_prof.area_m2,
        0.025, 0.35, "carbon_tube", cf_prof.wall_thickness_m, 1.0,
        "two_wall_clamped_inserts", rail_profile_id="mgn12_hollow_rail",
    )
    assert g_poc > g_sol  # lower rail EA* with full bolt bridges increases composite shear-lag gamma
    assert g_hol_same_a < g_poc  # open channel straddling M3 bolts lowers k_s by 10%
    assert 0.5 < g_sol < 0.99 and 0.5 < g_poc < 0.99


def test_bare_unsupported_rail_vs_bolted_xbeam_end_to_end() -> None:
    """Bare unsupported hollow vs pocketed rail exposes open-channel torsion/shear penalty; bolted X-beam retains stiffness."""
    base_path = Path("examples/corexy_voron_style.yaml")
    spec_2020_sol = PrinterSpec.from_yaml(base_path)
    spec_2020_sol.gantry.x_rail_profile = "mgn12_rail"

    spec_2020_poc = spec_2020_sol.model_copy(deep=True)
    spec_2020_poc.gantry.x_rail_profile = "mgn12_pocketed_rail"

    spec_2020_hol = spec_2020_sol.model_copy(deep=True)
    spec_2020_hol.gantry.x_rail_profile = "mgn12_hollow_rail"

    ev_sol = evaluate_printer(spec_2020_sol)
    ev_poc = evaluate_printer(spec_2020_poc)
    ev_hol = evaluate_printer(spec_2020_hol)

    # Bolted on 2020 extrusion: lighter moving X-rail mass for pocketed & hollow, clean positive frequencies
    assert ev_hol.assembly.x_rail_mass_kg < ev_poc.assembly.x_rail_mass_kg < ev_sol.assembly.x_rail_mass_kg
    assert ev_poc.modal_fea.fundamental_freq_hz > 20.0
    assert ev_hol.modal_fea.fundamental_freq_hz > 20.0

    # Unsupported bare rail: pocketed MGN12 must have higher torsional stiffness (and lower static
    # roll/nozzle deflection) than continuous open C-channel hollow MGN12, and beam_shear_coefficients
    # must activate Tet10 kappa on bare milled rails!
    spec_bare_poc = spec_2020_poc.model_copy(deep=True)
    spec_bare_poc.gantry.x_beam_profile = "unsupported_rail"
    spec_bare_poc.gantry.x_rail_mount = "front"
    spec_bare_poc.gantry.rail_backer_material = "none"

    spec_bare_hol = spec_bare_poc.model_copy(deep=True)
    spec_bare_hol.gantry.x_rail_profile = "mgn12_hollow_rail"

    asm_bare_poc = build_assembly(spec_bare_poc)
    asm_bare_hol = build_assembly(spec_bare_hol)
    m_poc = next(m for m in asm_bare_poc.members if m.id == "x_gantry_l")
    m_hol = next(m for m in asm_bare_hol.members if m.id == "x_gantry_l")

    ky_p, kz_p = beam_shear_coefficients(m_poc)
    ky_h, kz_h = beam_shear_coefficients(m_hol)
    assert ky_p > 0.0 and kz_p > 0.0
    assert ky_h > 0.0 and kz_h > 0.0
    # Front-mounted rail rotates local y/z -> kz holds original kappa_y (pocketed > hollow)
    assert kz_p > kz_h
    assert m_poc.profile.j_m4 > 2.0 * m_hol.profile.j_m4

    ev_bare_poc = evaluate_printer(spec_bare_poc)
    ev_bare_hol = evaluate_printer(spec_bare_hol)
    assert ev_bare_poc.static_fea.combined_max_deflection_um < ev_bare_hol.static_fea.combined_max_deflection_um
    assert ev_bare_poc.modal_fea.fundamental_freq_hz > 10.0
    assert ev_bare_hol.modal_fea.fundamental_freq_hz > 10.0


@pytest.mark.parametrize("rail_id", MILLED_RAIL_IDS)
def test_bom_generates_cleanly_for_all_milled_rails(rail_id: str) -> None:
    spec = PrinterSpec.from_yaml(Path("examples/corexy_voron_style.yaml"))
    spec.gantry.x_rail_profile = rail_id
    bom = generate_bom(spec)
    assert bom.total_cost_usd > 0.0
    assert any(
        ("Hollow" in it.name or "Pocketed" in it.name) and it.category == "motion"
        for it in bom.items
    )
