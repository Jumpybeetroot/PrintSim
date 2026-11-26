"""Comprehensive tests for community hardware realism (Monolith3D, Voron, DoomCube, VzBot, Annex)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from printsim.api.server import create_app
from printsim.engine import evaluate_printer
from printsim.fea import (
    assemble_global_matrices,
    compute_thermal_structural_report,
    solve_modal_fea,
    solve_static_fea,
)
from printsim.geometry import build_assembly
from printsim.schema import (
    BED_MOUNT_CATALOG,
    BELT_ROUTING_CATALOG,
    FRAME_FEET_CATALOG,
    MATERIAL_CATALOG,
    PROFILE_CATALOG,
    TOOLHEAD_PRESETS,
    Z_PROBE_CATALOG,
    BeltRoutingType,
    PrinterSpec,
    ZProbeType,
)

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def _paneled_trident() -> PrinterSpec:
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    spec.frame.enclosed_panels = True
    spec.frame.panel_mounting = "bolted_structural"
    spec.frame.panel_material = "aluminum_5052_sheet"
    spec.frame.panel_thickness_mm = 4.0
    return spec


def test_community_materials_profiles_and_schema_exports() -> None:
    """Verify new community materials, profiles, catalogs, and schema exports."""
    assert ZProbeType is not None
    assert BeltRoutingType is not None

    assert set(Z_PROBE_CATALOG.keys()) == {
        "static_eddy",
        "klicky_inductive",
        "cnc_tap",
        "printed_tap",
    }
    assert set(BELT_ROUTING_CATALOG.keys()) == {"stock_corexy", "monolith_short"}
    assert "whopping_pochard_kinematic" in BED_MOUNT_CATALOG
    assert "paver_on_foam" in FRAME_FEET_CATALOG

    # Toolhead presets
    for pid in ("stealthburner", "stealthburner_tap", "xol_compact", "a4t_beacon", "dragon_burner", "archetype_cpap", "heavy_uhf_long"):
        assert pid in TOOLHEAD_PRESETS
        assert "z_probe_type" in TOOLHEAD_PRESETS[pid]
    assert TOOLHEAD_PRESETS["stealthburner_tap"]["z_probe_type"] == "printed_tap"
    assert TOOLHEAD_PRESETS["a4t_beacon"]["z_probe_type"] == "static_eddy"

    # Materials
    atp5 = MATERIAL_CATALOG["atp5_aluminum"]
    assert atp5.density_kg_m3 == pytest.approx(2660.0)
    assert atp5.youngs_modulus_pa == pytest.approx(70.3e9)
    assert atp5.yield_strength_pa == pytest.approx(145.0e6)

    cf_std = MATERIAL_CATALOG["carbon_fiber_tube"]
    cf_hi = MATERIAL_CATALOG["carbon_fiber_high_tg"]
    assert cf_hi.youngs_modulus_pa == pytest.approx(cf_std.youngs_modulus_pa)
    assert cf_hi.effective_modulus_pa(25.0) == pytest.approx(cf_std.effective_modulus_pa(25.0))
    assert cf_hi.tg_c == pytest.approx(125.0)
    assert cf_hi.cost_per_kg_usd > cf_std.cost_per_kg_usd

    # Profiles
    cnc_x = PROFILE_CATALOG["cnc_al_beam_20x20"]
    ext_2020 = PROFILE_CATALOG["extrusion_2020"]
    assert cnc_x.area_m2 < ext_2020.area_m2
    assert cnc_x.j_m4 > ext_2020.j_m4

    ti_tube = PROFILE_CATALOG["ti_tube_20x20x1_2"]
    st_tube = PROFILE_CATALOG["steel_tube_25x25x2"]
    ext_3060 = PROFILE_CATALOG["extrusion_3060"]
    assert ti_tube.width_m == pytest.approx(0.020)
    assert st_tube.width_m == pytest.approx(0.025)
    assert ext_3060.height_m == pytest.approx(0.060)
    assert ext_3060.iy_m4 == pytest.approx(21.2e-8)

    # API catalog exposes z_probes and belt_routings
    client = TestClient(create_app())
    r_cat = client.get("/api/catalog")
    assert r_cat.status_code == 200
    cat = r_cat.json()
    assert {z["id"] for z in cat["z_probes"]} == set(Z_PROBE_CATALOG.keys())
    assert {b["id"] for b in cat["belt_routings"]} == set(BELT_ROUTING_CATALOG.keys())


def test_z_probe_monotonic_stiffness_pitch_freq_and_tap_compliance() -> None:
    """printed_tap < cnc_tap < static_eddy monotonically for toolhead stiffness, pitch freq, and resonance."""
    base = _paneled_trident()
    probes = ("printed_tap", "cnc_tap", "klicky_inductive", "static_eddy")
    reports = {}
    for zp in probes:
        s = base.model_copy(deep=True)
        s.toolhead.z_probe_type = zp  # type: ignore[assignment]
        reports[zp] = evaluate_printer(s)

    # Strictly monotonic: printed_tap < cnc_tap < static_eddy
    for a, b in (("printed_tap", "cnc_tap"), ("cnc_tap", "static_eddy")):
        assert (
            reports[a].static_fea.toolhead_stiffness_x_n_per_mm
            < reports[b].static_fea.toolhead_stiffness_x_n_per_mm
        )
        assert (
            reports[a].static_fea.toolhead_stiffness_y_n_per_mm
            < reports[b].static_fea.toolhead_stiffness_y_n_per_mm
        )
        assert (
            reports[a].modal_fea.nozzle_pitch_twist_freq_hz
            < reports[b].modal_fea.nozzle_pitch_twist_freq_hz
        )
        assert (
            reports[a].modal_fea.fundamental_freq_hz
            < reports[b].modal_fea.fundamental_freq_hz
        )

    # printed_tap measurably increases toolhead compliance & lowers accel vs static_eddy without injecting hidden mass
    r_tap = reports["printed_tap"]
    r_eddy = reports["static_eddy"]
    assert r_tap.modal_fea.modes[0].subsystem_share_pct["toolhead"] > r_eddy.modal_fea.modes[0].subsystem_share_pct["toolhead"]
    assert r_tap.static_fea.nozzle_twist_amplification_pct > r_eddy.static_fea.nozzle_twist_amplification_pct
    assert r_tap.modal_fea.recommended_max_accel_mm_s2 < r_eddy.modal_fea.recommended_max_accel_mm_s2
    assert r_tap.assembly.x_moving_mass_kg == pytest.approx(r_eddy.assembly.x_moving_mass_kg, rel=1e-12)
    assert r_tap.assembly.y_moving_mass_kg == pytest.approx(r_eddy.assembly.y_moving_mass_kg, rel=1e-12)
    assert r_tap.assembly.total_mass_kg == pytest.approx(r_eddy.assembly.total_mass_kg, abs=1e-8)
    assert any("Voron Tap" in d or "static_eddy" in d for d in r_tap.diagnostics)
    assert not any("Voron Tap" in d for d in r_eddy.diagnostics)


def test_belt_routing_monolith_short_increases_stiffness_and_accel() -> None:
    """monolith_short raises belt/drive stiffness and recommended accel over stock_corexy."""
    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    s_stock = base.model_copy(deep=True)
    s_stock.gantry.belt_routing = "stock_corexy"
    s_mono = base.model_copy(deep=True)
    s_mono.gantry.belt_routing = "monolith_short"

    r_stock = evaluate_printer(s_stock)
    r_mono = evaluate_printer(s_mono)

    assert r_mono.kinematics.belt_routing == "monolith_short"
    assert r_stock.kinematics.belt_routing == "stock_corexy"
    assert r_mono.kinematics.belt_stiffness_n_per_mm > r_stock.kinematics.belt_stiffness_n_per_mm
    assert r_mono.kinematics.drive_stiffness_n_per_mm > r_stock.kinematics.drive_stiffness_n_per_mm
    assert r_mono.kinematics.idler_shaft_stiffness_n_per_mm > r_stock.kinematics.idler_shaft_stiffness_n_per_mm
    assert r_mono.kinematics.idler_shaft_deflection_um < r_stock.kinematics.idler_shaft_deflection_um
    assert r_mono.modal_fea.shaper_x is not None and r_stock.modal_fea.shaper_x is not None
    assert r_mono.modal_fea.shaper_x.effective_freq_hz > r_stock.modal_fea.shaper_x.effective_freq_hz

    # On a stiffened paneled CoreXY (where 27 Hz frame+toolhead doesn't cap Y accel), monolith_short raises safe accel
    paneled = _paneled_trident()
    paneled.toolhead.mount_material = "aluminum_6061_t6"
    paneled.toolhead.internal_stiffness_n_per_mm = 1850.0
    p_stock = paneled.model_copy(deep=True)
    p_stock.gantry.belt_routing = "stock_corexy"
    p_mono = paneled.model_copy(deep=True)
    p_mono.gantry.belt_routing = "monolith_short"
    assert evaluate_printer(p_mono).modal_fea.recommended_max_accel_mm_s2 > evaluate_printer(p_stock).modal_fea.recommended_max_accel_mm_s2

    # Pulley circumference & VFA speed band sanity
    assert r_mono.kinematics.pulley_circumference_mm == pytest.approx(40.0)
    vfa = r_mono.kinematics.vfa_belt_tooth_speed_range_mm_s
    assert len(vfa) == 2
    assert 30.0 <= vfa[0] < vfa[1] <= 350.0


def test_steel_core_20t_fatigue_risk_and_diagnostic() -> None:
    """Steel-core GT2 on <=20T pulley sets steel_core_fatigue_risk=True and emits diagnostic; aramid does not."""
    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")

    s_steel = base.model_copy(deep=True)
    s_steel.gantry.belt.core = "steel"
    s_steel.gantry.belt.pulley_teeth = 20
    r_steel = evaluate_printer(s_steel)

    s_aramid = base.model_copy(deep=True)
    s_aramid.gantry.belt.core = "aramid"
    s_aramid.gantry.belt.pulley_teeth = 20
    r_aramid = evaluate_printer(s_aramid)

    assert r_steel.kinematics.steel_core_fatigue_risk is True
    assert any("steel-core" in d.lower() or "fatigue" in d.lower() for d in r_steel.diagnostics)

    assert r_aramid.kinematics.steel_core_fatigue_risk is False
    assert not any("steel-core" in d.lower() for d in r_aramid.diagnostics)


def test_carbon_fiber_chamber_epoxy_softening_diagnostic() -> None:
    """carbon_fiber_tube at 70 C chamber emits epoxy softening warning; carbon_fiber_high_tg does not."""
    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    base.frame.enclosed_panels = True
    base.frame.chamber_temp_c = 70.0
    base.gantry.x_beam_profile = "cf_tube_20x20x2"

    s_std = base.model_copy(deep=True)
    s_std.gantry.x_beam_material = "carbon_fiber_tube"
    r_std = evaluate_printer(s_std)

    s_hi = base.model_copy(deep=True)
    s_hi.gantry.x_beam_material = "carbon_fiber_high_tg"
    r_hi = evaluate_printer(s_hi)

    assert any("carbon_fiber_high_tg" in d or "soften" in d.lower() for d in r_std.diagnostics)
    assert not any("carbon_fiber_high_tg" in d for d in r_hi.diagnostics)


def test_whopping_pochard_and_paver_on_foam_physics() -> None:
    """whopping_pochard_kinematic has zero thermal taco & high stiffness; paver_on_foam adds 18kg and high damping."""
    base = _paneled_trident()
    base.bed.target_temp_c = 110.0

    # 1. Whopping Pochard kinematic leaf mount
    s_wp = base.model_copy(deep=True)
    s_wp.bed.mount_type = "whopping_pochard_kinematic"
    s_spr = base.model_copy(deep=True)
    s_spr.bed.mount_type = "springs_4pt"
    s_rig = base.model_copy(deep=True)
    s_rig.bed.mount_type = "rigid_4pt"

    asm_wp = build_assembly(s_wp)
    st_wp = solve_static_fea(asm_wp)
    st_spr = solve_static_fea(build_assembly(s_spr))
    st_rig = solve_static_fea(build_assembly(s_rig))
    ts_wp = compute_thermal_structural_report(asm_wp)

    assert ts_wp.bed_buckling_taco_um == pytest.approx(0.0, abs=1e-9)
    assert st_spr.toolhead_stiffness_z_n_per_mm < st_wp.toolhead_stiffness_z_n_per_mm <= st_rig.toolhead_stiffness_z_n_per_mm

    # 2. Paver on foam
    asm_rigid = build_assembly(base)
    s_paver = base.model_copy(deep=True)
    s_paver.frame.feet_type = "paver_on_foam"
    asm_paver = build_assembly(s_paver)

    assert (asm_paver.total_mass_kg - asm_rigid.total_mass_kg) == pytest.approx(18.0, abs=1e-6)

    s_rubber = base.model_copy(deep=True)
    s_rubber.frame.feet_type = "rubber_feet"
    asm_rubber = build_assembly(s_rubber)
    assert asm_paver.frame_damping_ratio > asm_rubber.frame_damping_ratio > asm_rigid.frame_damping_ratio

    # SPD & Modal check
    K, M, _, _, free_dofs = assemble_global_matrices(asm_paver)
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    M_ff = M[np.ix_(free_dofs, free_dofs)]
    assert np.allclose(K_ff, K_ff.T, rtol=1e-12, atol=1e-6)
    assert np.allclose(M_ff, M_ff.T, rtol=1e-12, atol=1e-10)
    assert float(np.linalg.eigvalsh(0.5 * (K_ff + K_ff.T))[0]) > 0.0

    mo_paver = solve_modal_fea(asm_paver)
    assert math.isfinite(mo_paver.fundamental_freq_hz) and mo_paver.fundamental_freq_hz > 5.0


@pytest.mark.parametrize(
    ("preset_file", "min_composite"),
    [("doomcube_4040.yaml", 85.0), ("vzbot_330_awd.yaml", 72.0)],
)
def test_new_community_presets_load_build_and_evaluate_cleanly(
    preset_file: str, min_composite: float
) -> None:
    """DoomCube 4040 and VzBot 330 AWD evaluate with clean scores and zero collisions."""
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / preset_file)
    report = evaluate_printer(spec)
    assert not report.collision.has_collision
    assert report.scores.composite > min_composite
    if preset_file == "vzbot_330_awd.yaml":
        # Real VzBot: stock CoreXY belt path on printed rubber-insert feet, so the global
        # fundamental is the ~20 Hz whole-printer rocking mode on the feet, not a frame mode.
        assert report.kinematics.belt_routing == "stock_corexy"
        assert 12.0 < report.modal_fea.fundamental_freq_hz < 30.0
        assert report.modal_fea.shaper_y.effective_freq_hz > 47.0
    else:
        assert report.modal_fea.fundamental_freq_hz > 40.0
        assert report.kinematics.belt_routing == "monolith_short"
