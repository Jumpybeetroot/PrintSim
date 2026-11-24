"""Physics and schema verification tests for Z-axis rails/leadscrews, 3-stage toolhead, and hotends."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
import pytest

from printsim.api.server import app
from printsim.fea import solve_modal_fea, solve_static_fea
from printsim.fea.components import hertzian_rail_carriage_stiffness
from printsim.geometry import build_assembly
from printsim.geometry.builder import compute_toolhead_link_props
from printsim.mcp_server import dispatch_tool
from printsim.schema import (
    GANTRY_BRACKET_CATALOG,
    GANTRY_WEIGHT_CLASSES,
    HOTEND_CATALOG,
    RAIL_QUALITY_CATALOG,
    TOOLHEAD_CONSTRUCTION_CATALOG,
    TOOLHEAD_PRESETS,
    Z_DRIVE_MECHANISM_CATALOG,
    Z_RAIL_PROFILES,
    BedSpec,
    GantrySpec,
    HotendSpec,
    PrinterSpec,
    get_material,
    get_profile,
)

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


# ---------------------------------------------------------------------------
# 1. Schema Catalogs & Hotend Stiffness Spectrum
# ---------------------------------------------------------------------------


def test_hotend_catalog_presets_and_monotonic_stiffness_spectrum() -> None:
    """All 10 HOTEND_CATALOG entries validate and produce monotonic stiffness from V6 to Chube."""
    ordered_ids = [
        "e3d_v6_groove",
        "e3d_revo_voron",
        "phaetus_rapido_2_uhf",
        "phaetus_dragon_uhf",
        "phaetus_rapido_2_hf",
        "phaetus_dragon_hf",
        "mellow_goliath_air",
        "chube_compact",
        "chube_air",
        "chube_conduction",
    ]
    assert set(HOTEND_CATALOG.keys()) == set(ordered_ids)

    specs = [HotendSpec(preset_id=hid) for hid in ordered_ids]
    for hid, hs in zip(ordered_ids, specs, strict=True):
        assert hs.preset_id == hid
        assert hs.structural_stiffness_n_per_mm == pytest.approx(
            float(HOTEND_CATALOG[hid]["structural_stiffness_n_per_mm"]), rel=1e-9
        )
        assert hs.mount_style == HOTEND_CATALOG[hid]["mount_style"]
    # Verify Chube Conduction 52mm UHF melt zone passes validation cleanly
    assert specs[-1].melt_zone_length_mm == pytest.approx(52.0, rel=1e-9)

    for idx in range(len(specs) - 1):
        assert specs[idx + 1].structural_stiffness_n_per_mm > specs[idx].structural_stiffness_n_per_mm

    # Physics sweep on stiff CNC CoreXY so hotend flex signal is unmistakable
    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_monolith_awd.yaml")
    sweep_ids = [
        "e3d_v6_groove",
        "e3d_revo_voron",
        "phaetus_rapido_2_uhf",
        "phaetus_dragon_hf",
        "mellow_goliath_air",
        "chube_compact",
        "chube_air",
        "chube_conduction",
    ]
    statics = []
    modals = []
    for hid in sweep_ids:
        s = base.model_copy(deep=True)
        s.toolhead.hotend = HotendSpec(preset_id=hid)
        asm = build_assembly(s)
        statics.append(solve_static_fea(asm))
        modals.append(solve_modal_fea(asm))

    for idx in range(len(sweep_ids) - 1):
        cur_s, nxt_s = statics[idx], statics[idx + 1]
        assert nxt_s.toolhead_stiffness_x_n_per_mm > cur_s.toolhead_stiffness_x_n_per_mm
        assert nxt_s.toolhead_stiffness_y_n_per_mm > cur_s.toolhead_stiffness_y_n_per_mm
        assert (
            nxt_s.toolhead_flex_breakdown["effective_tip_stiffness_n_per_mm"]
            > cur_s.toolhead_flex_breakdown["effective_tip_stiffness_n_per_mm"]
        )
        assert (
            nxt_s.toolhead_flex_breakdown["hotend_stiffness_n_per_mm"]
            > cur_s.toolhead_flex_breakdown["hotend_stiffness_n_per_mm"]
        )
        assert (
            nxt_s.toolhead_flex_breakdown["hotend_and_nozzle_pct"]
            < cur_s.toolhead_flex_breakdown["hotend_and_nozzle_pct"]
        )

    # Massive real-world difference between floppy groove-mount V6 and 6-bolt Chube Conduction
    v6_st, chube_st = statics[0], statics[-1]
    v6_mo, chube_mo = modals[0], modals[-1]
    assert v6_st.toolhead_flex_breakdown["dominant_flex_source"] == "hotend_nozzle"
    assert v6_st.toolhead_flex_breakdown["hotend_and_nozzle_pct"] > 70.0
    assert chube_st.toolhead_flex_breakdown["hotend_and_nozzle_pct"] < 20.0
    assert chube_st.nozzle_twist_amplification_pct < v6_st.nozzle_twist_amplification_pct
    assert chube_st.toolhead_stiffness_y_n_per_mm > v6_st.toolhead_stiffness_y_n_per_mm * 3.0
    assert chube_mo.nozzle_pitch_twist_freq_hz > v6_mo.nozzle_pitch_twist_freq_hz * 1.5


def test_custom_non_preset_hotend_scales_with_heatbreak_wall_and_material() -> None:
    """Custom non-preset HotendSpec scales nozzle stiffness with heatbreak wall & material."""
    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_monolith_awd.yaml")
    s_thin_ti = base.model_copy(deep=True)
    s_thin_ti.toolhead.hotend = HotendSpec(
        preset_id=None,
        mount_style="dragon_4bolt_m25",
        structural_stiffness_n_per_mm=450.0,
        heatbreak_material="titanium_grade5",
        heatbreak_wall_mm=0.15,
    )
    s_thick_ss = base.model_copy(deep=True)
    s_thick_ss.toolhead.hotend = HotendSpec(
        preset_id=None,
        mount_style="dragon_4bolt_m25",
        structural_stiffness_n_per_mm=450.0,
        heatbreak_material="stainless_304",
        heatbreak_wall_mm=0.45,
    )
    st_thin = solve_static_fea(build_assembly(s_thin_ti))
    st_thick = solve_static_fea(build_assembly(s_thick_ss))
    assert (
        st_thick.toolhead_flex_breakdown["hotend_stiffness_n_per_mm"]
        > st_thin.toolhead_flex_breakdown["hotend_stiffness_n_per_mm"] * 1.3
    )
    assert st_thick.toolhead_stiffness_y_n_per_mm > st_thin.toolhead_stiffness_y_n_per_mm


# ---------------------------------------------------------------------------
# 2. Multi-Part 3-Stage Toolhead & Axisymmetric Delta Symmetry
# ---------------------------------------------------------------------------


def test_independent_3stage_toolhead_material_upgrades() -> None:
    """Carriage backplate, body shroud, and hotend clamp materials each independently stiffen toolhead."""
    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    base.frame.enclosed_panels = True
    base.frame.panel_mounting = "bolted_structural"
    base.frame.panel_material = "aluminum_5052_sheet"
    base.frame.panel_thickness_mm = 4.0
    base.toolhead.mount_material = "abs_printed"
    base.toolhead.carriage_mount_material = None
    base.toolhead.body_material = None
    base.toolhead.hotend_mount_material = None
    base.toolhead.hotend = HotendSpec(preset_id="chube_conduction")

    assert base.toolhead.effective_carriage_mount_material == "abs_printed"
    assert base.toolhead.effective_body_material == "abs_printed"
    assert base.toolhead.effective_hotend_mount_material == "abs_printed"

    st_abs = solve_static_fea(build_assembly(base))

    # Stage 1 upgrade only: CNC aluminum carriage backplate
    s_car = base.model_copy(deep=True)
    s_car.toolhead.carriage_mount_material = "aluminum_6061_t6"
    st_car = solve_static_fea(build_assembly(s_car))
    assert (
        st_car.toolhead_flex_breakdown["carriage_mount_stiffness_n_per_mm"]
        > st_abs.toolhead_flex_breakdown["carriage_mount_stiffness_n_per_mm"] * 3.0
    )
    assert st_car.toolhead_stiffness_y_n_per_mm > st_abs.toolhead_stiffness_y_n_per_mm * 1.15

    # Stage 2 upgrade only: PA12-CF -> CNC aluminum toolhead body
    s_body_pa = base.model_copy(deep=True)
    s_body_pa.toolhead.body_material = "pa12_cf_printed"
    s_body_al = base.model_copy(deep=True)
    s_body_al.toolhead.body_material = "aluminum_6061_t6"
    st_body_pa = solve_static_fea(build_assembly(s_body_pa))
    st_body_al = solve_static_fea(build_assembly(s_body_al))
    assert st_body_al.toolhead_stiffness_y_n_per_mm > st_body_pa.toolhead_stiffness_y_n_per_mm
    assert st_body_pa.toolhead_stiffness_y_n_per_mm > st_abs.toolhead_stiffness_y_n_per_mm

    # Stage 3 upgrade only: CNC aluminum hotend mount clamp with stiff Chube
    s_hm = base.model_copy(deep=True)
    s_hm.toolhead.hotend_mount_material = "aluminum_6061_t6"
    st_hm = solve_static_fea(build_assembly(s_hm))
    assert (
        st_hm.toolhead_flex_breakdown["hotend_stiffness_n_per_mm"]
        > st_abs.toolhead_flex_breakdown["hotend_stiffness_n_per_mm"] * 1.25
    )
    assert st_hm.toolhead_stiffness_y_n_per_mm > st_abs.toolhead_stiffness_y_n_per_mm


def test_delta_axisymmetric_toolhead_xy_symmetry_and_bracket_mass() -> None:
    """Axisymmetric Delta effector has iy == iz, Kx ~= Ky, equal moving masses, and C3 tower modes."""
    delta = PrinterSpec.from_yaml(EXAMPLES_DIR / "delta_racer.yaml")
    delta.gantry.bracket_material = "aluminum_6061_t6"
    asm = build_assembly(delta)
    th_mem = next(m for m in asm.members if m.id == "toolhead_mount")
    assert th_mem.profile.iy_m4 == pytest.approx(th_mem.profile.iz_m4, rel=1e-9)
    assert asm.x_moving_mass_kg == pytest.approx(asm.y_moving_mass_kg, rel=1e-9)

    st = solve_static_fea(asm)
    assert st.toolhead_stiffness_x_n_per_mm == pytest.approx(st.toolhead_stiffness_y_n_per_mm, rel=5e-3)
    delta_no_z = delta.model_copy(deep=True)
    delta_no_z.bed.z_rail_profile = "none"
    st_no_z = solve_static_fea(build_assembly(delta_no_z))
    assert st_no_z.toolhead_stiffness_x_n_per_mm == pytest.approx(
        st_no_z.toolhead_stiffness_y_n_per_mm, rel=1e-6
    )

    mo = solve_modal_fea(asm)
    assert abs(mo.modes[1].frequency_hz - mo.modes[2].frequency_hz) < 1.0


# ---------------------------------------------------------------------------
# 3. Z-Axis Linear Rails, Smooth Rods, Z-Leadscrews & Z-Belts
# ---------------------------------------------------------------------------


def test_z_rail_profiles_stiffness_and_bedslinger_geometry() -> None:
    """Verify none < mgn9 < mgn12, rod_8 < rod_10 < rod_12 < mgn12, and Bedslinger Y-rail height."""
    assert len(Z_RAIL_PROFILES) == 6
    trident = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")

    rail_profiles = ["none", "mgn9_rail", "mgn12_rail"]
    kx_rails = [
        solve_static_fea(
            build_assembly(trident.model_copy(update={"bed": trident.bed.model_copy(update={"z_rail_profile": zp})}))
        ).toolhead_stiffness_x_n_per_mm
        for zp in rail_profiles
    ]
    assert kx_rails[0] < kx_rails[1] < kx_rails[2]

    rod_profiles = ["steel_rod_8mm", "steel_rod_10mm", "steel_rod_12mm", "mgn12_rail"]
    kx_rods = [
        solve_static_fea(
            build_assembly(trident.model_copy(update={"bed": trident.bed.model_copy(update={"z_rail_profile": zp})}))
        ).toolhead_stiffness_x_n_per_mm
        for zp in rod_profiles
    ]
    assert kx_rods[0] < kx_rods[1] < kx_rods[2] < kx_rods[3]

    # Bedslinger Z-rails & base-height Y-rails (z = 15 mm, bkt_delta_kg on X moving mass)
    bs = PrinterSpec.from_yaml(EXAMPLES_DIR / "cartesian_bedslinger.yaml")
    bs_cnc = bs.model_copy(deep=True)
    bs_cnc.gantry.bracket_material = "aluminum_6061_t6"
    bs_cnc.bed.z_rail_profile = "mgn12_rail"

    asm_bs = build_assembly(bs)
    asm_bs_cnc = build_assembly(bs_cnc)
    assert asm_bs.node_map["bed_rail_l"].position_m[2] == pytest.approx(0.015, abs=1e-6)
    assert asm_bs_cnc.x_moving_mass_kg > asm_bs.x_moving_mass_kg
    assert asm_bs_cnc.y_moving_mass_kg > asm_bs.y_moving_mass_kg
    assert solve_static_fea(asm_bs_cnc).toolhead_stiffness_x_n_per_mm > solve_static_fea(asm_bs).toolhead_stiffness_x_n_per_mm


def test_z_leadscrew_and_z_belt_mechanisms_monotonic_bed_stiffness() -> None:
    """Verify tr8x8_pom < tr8x4_pom < tr8x8_brass < sfu1204_bk10 and gt2_6mm < gt2_9mm Z-belt."""
    assert len(Z_DRIVE_MECHANISM_CATALOG) == 6
    assert BedSpec(z_drive="belted_quad_z").effective_z_leadscrew_type == "gt2_6mm_z_belt"
    assert BedSpec(z_drive="leadscrew_triple").effective_z_leadscrew_type == "tr8x4_pom"
    assert BedSpec(z_drive="leadscrew_dual").effective_z_leadscrew_type == "tr8x8_pom"
    assert BedSpec(z_drive="leadscrew_single").effective_z_leadscrew_type == "tr8x8_pom"

    trident = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    trident.bed.mount_type = "fixed_4pt"  # stiff mount so Z-screw axial compliance stands out clearly
    screws = ["tr8x8_pom", "tr8x4_pom", "tr8x8_brass", "sfu1204_bk10"]
    bed_droops = []
    bed_kzs = []
    for zls in screws:
        s = trident.model_copy(deep=True)
        s.bed.z_leadscrew_type = zls  # type: ignore[assignment]
        asm = build_assembly(s)
        st = solve_static_fea(asm)
        bed_droops.append(abs(float(st.nodal_displacements_mm["bed_surface"][2])))
        bed_kzs.append(st.toolhead_stiffness_z_n_per_mm)

    for idx in range(len(screws) - 1):
        assert bed_droops[idx + 1] < bed_droops[idx]
        assert bed_kzs[idx + 1] > bed_kzs[idx]

    # Voron 2.4 Z-belt upgrade: gt2_6mm_z_belt -> gt2_9mm_z_belt (+50% z_joint normal stiffness)
    v24 = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    v24_9 = v24.model_copy(deep=True)
    v24_9.bed.z_leadscrew_type = "gt2_9mm_z_belt"
    asm_6 = build_assembly(v24)
    asm_9 = build_assembly(v24_9)
    sp6 = next(sp for sp in asm_6.fastener_springs if sp.id == "z_joint_fl")
    sp9 = next(sp for sp in asm_9.fastener_springs if sp.id == "z_joint_fl")
    assert sp6.k_normal_n_m == pytest.approx(1.8e5, rel=1e-6)
    assert sp9.k_normal_n_m == pytest.approx(2.7e5, rel=1e-6)
    assert solve_static_fea(asm_9).toolhead_stiffness_z_n_per_mm > solve_static_fea(asm_6).toolhead_stiffness_z_n_per_mm


# ---------------------------------------------------------------------------
# 4. Flex Breakdown, Geometry Serialization & API / MCP Catalogs
# ---------------------------------------------------------------------------


def test_flex_breakdown_sums_to_100_and_geometry_dict_z_bed_metadata() -> None:
    """Verify toolhead_flex_breakdown sums to 100%, dominant source shifts, and geometry dict Z fields."""
    v24 = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    asm = build_assembly(v24)
    st = solve_static_fea(asm)
    brk = st.toolhead_flex_breakdown
    pct_sum = (
        float(brk["rail_and_xbeam_pct"])
        + float(brk["carriage_and_body_pct"])
        + float(brk["hotend_and_nozzle_pct"])
    )
    assert pct_sum == pytest.approx(100.0, abs=0.2)
    assert brk["dominant_flex_source"] in ("rail_xbeam", "carriage_body", "hotend_nozzle")
    assert "toolhead_flex_breakdown" in st.to_dict()

    # Geometry dict Z-rails, Z-leadscrews, bed_mount_points_mm inside bed, and toolhead_structure
    gdict = asm.to_geometry_dict()
    assert gdict["z_rail_profile"] == "mgn12_rail"
    assert gdict["effective_z_leadscrew_type"] == "gt2_6mm_z_belt"
    assert len(gdict["z_rails"]) == 4
    assert len(gdict["z_leadscrews"]) == 4
    assert gdict["z_leadscrews"][0]["kind"] == "belt"
    assert gdict["toolhead_structure"]["hotend_preset_id"] == "phaetus_dragon_hf"

    bc_x, bc_y, _ = gdict["bed_center_mm"]
    bx, by = gdict["bed_size_mm"]
    assert len(gdict["bed_mount_points_mm"]) == 3  # kinematic_3pt default
    for mx, my, _ in gdict["bed_mount_points_mm"]:
        assert bc_x - 0.5 * bx <= mx <= bc_x + 0.5 * bx
        assert bc_y - 0.5 * by <= my <= bc_y + 0.5 * by
    assert len(gdict["bed_support_points_mm"]) == 4

    # None z_rail_profile returns empty z_rails
    v24_no_zr = v24.model_copy(deep=True)
    v24_no_zr.bed.z_rail_profile = "none"
    assert build_assembly(v24_no_zr).to_geometry_dict()["z_rails"] == []


def test_api_and_mcp_expose_new_catalogs_and_hotend_overrides() -> None:
    """Verify /api/catalog and MCP list_catalog expose hotends/Z-drives/Z-rails & dot overrides work."""
    client = TestClient(app)
    resp = client.get("/api/catalog")
    assert resp.status_code == 200
    payload = resp.json()
    assert len(payload["hotend_presets"]) == 10
    assert len(payload["z_drive_mechanisms"]) == 6
    assert len(payload["z_rail_profiles"]) == 6

    mcp_cat = dispatch_tool("list_catalog", {})
    assert len(mcp_cat["hotend_presets"]) == 10
    assert len(mcp_cat["z_drive_mechanisms"]) == 6
    assert len(mcp_cat["z_rail_profiles"]) == 6

    trident_yaml = dispatch_tool("get_preset_yaml", {"preset_id": "voron_trident_300"})["yaml_text"]
    cmp_res = dispatch_tool(
        "mutate_and_compare_printer",
        {
            "yaml_text": trident_yaml,
            "candidates": [
                {
                    "name": "chube_ballscrew",
                    "overrides": {
                        "toolhead.hotend.preset_id": "chube_conduction",
                        "bed.z_leadscrew_type": "sfu1204_bk10",
                    },
                }
            ],
        },
    )
    assert (
        cmp_res["ranked_candidates"][0]["toolhead_stiffness_y_n_per_mm"]
        > cmp_res["baseline"]["toolhead_stiffness_y_n_per_mm"]
    )
    eval_best = dispatch_tool("evaluate_printer_yaml", {"yaml_text": cmp_res["best_yaml"]})
    assert math.isfinite(
        eval_best["static_fea"]["toolhead_flex_breakdown"]["hotend_stiffness_n_per_mm"]
    )


# ---------------------------------------------------------------------------
# 5. Per-Part Toolhead Thicknesses & Modal Nodal Rotations
# ---------------------------------------------------------------------------


def test_toolhead_default_thicknesses_produce_zero_mass_delta() -> None:
    """Default thicknesses (5.0, 4.0, 8.0) mm yield strictly 0.0 mass delta across materials."""
    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    assert base.toolhead.carriage_plate_thickness_mm == 5.0
    assert base.toolhead.body_wall_thickness_mm == 4.0
    assert base.toolhead.hotend_mount_thickness_mm == 8.0

    asm_default = build_assembly(base)
    th_struct = asm_default.to_geometry_dict()["toolhead_structure"]
    assert th_struct["carriage_plate_thickness_mm"] == 5.0
    assert th_struct["body_wall_thickness_mm"] == 4.0
    assert th_struct["hotend_mount_thickness_mm"] == 8.0
    assert th_struct["thickness_mass_delta_g"] == 0.0

    # Even with CNC aluminum overrides, default reference thicknesses add zero mass delta
    s_alu = base.model_copy(deep=True)
    s_alu.toolhead.carriage_mount_material = "aluminum_6061_t6"
    s_alu.toolhead.body_material = "aluminum_6061_t6"
    s_alu.toolhead.hotend_mount_material = "aluminum_6061_t6"
    assert build_assembly(s_alu).to_geometry_dict()["toolhead_structure"]["thickness_mass_delta_g"] == 0.0


def test_carriage_plate_thickness_monotonic_stiffness_and_invariant_mass() -> None:
    """Thicker carriage plate increases carriage_mount_n_per_mm while keeping user-entered mass strictly invariant."""
    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    base.toolhead.hotend = HotendSpec(preset_id="chube_conduction")

    thicknesses = [2.5, 5.0, 8.0, 12.0]
    car_ks: list[float] = []
    cog_masses: list[float] = []
    x_masses: list[float] = []
    y_masses: list[float] = []
    tot_masses: list[float] = []
    deltas_g: list[float] = []

    for t_mm in thicknesses:
        s = base.model_copy(deep=True)
        s.toolhead.carriage_plate_thickness_mm = t_mm
        asm = build_assembly(s)
        gdict = asm.to_geometry_dict()
        car_ks.append(asm.toolhead_stage_stiffnesses_n_per_mm["carriage_mount_n_per_mm"])
        cog_masses.append(asm.node_map["toolhead_cog"].lumped_mass_kg)
        x_masses.append(asm.x_moving_mass_kg)
        y_masses.append(asm.y_moving_mass_kg)
        tot_masses.append(asm.total_mass_kg)
        deltas_g.append(gdict["toolhead_structure"]["thickness_mass_delta_g"])

    for idx in range(len(thicknesses) - 1):
        assert car_ks[idx + 1] > car_ks[idx]
        assert cog_masses[idx + 1] == pytest.approx(cog_masses[0], rel=1e-12)
        assert x_masses[idx + 1] == pytest.approx(x_masses[0], rel=1e-12)
        assert y_masses[idx + 1] == pytest.approx(y_masses[0], rel=1e-12)
        assert tot_masses[idx + 1] == pytest.approx(tot_masses[0], abs=1e-8)
    assert all(d == 0.0 for d in deltas_g)

    # Material swap (ABS vs Al 6061-T6 backplate) stiffens carriage mount without injecting hidden mass
    s_abs = base.model_copy(deep=True)
    s_abs.toolhead.carriage_mount_material = "abs_printed"
    s_abs.toolhead.carriage_plate_thickness_mm = 10.0

    s_al = base.model_copy(deep=True)
    s_al.toolhead.carriage_mount_material = "aluminum_6061_t6"
    s_al.toolhead.carriage_plate_thickness_mm = 10.0

    asm_abs = build_assembly(s_abs)
    asm_al = build_assembly(s_al)
    assert (
        asm_al.toolhead_stage_stiffnesses_n_per_mm["carriage_mount_n_per_mm"]
        > asm_abs.toolhead_stage_stiffnesses_n_per_mm["carriage_mount_n_per_mm"]
    )
    assert asm_abs.to_geometry_dict()["toolhead_structure"]["thickness_mass_delta_g"] == 0.0
    assert asm_al.to_geometry_dict()["toolhead_structure"]["thickness_mass_delta_g"] == 0.0
    assert asm_al.x_moving_mass_kg == pytest.approx(asm_abs.x_moving_mass_kg, rel=1e-12)
    assert asm_al.y_moving_mass_kg == pytest.approx(asm_abs.y_moving_mass_kg, rel=1e-12)
    assert asm_al.total_mass_kg == pytest.approx(asm_abs.total_mass_kg, abs=1e-8)


def test_body_wall_thickness_monotonic_stiffness_and_invariant_mass() -> None:
    """Increasing body_wall_thickness_mm monotonically increases body_n_per_mm while keeping mass invariant."""
    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    base.toolhead.hotend = HotendSpec(preset_id="chube_conduction")

    thicknesses = [2.0, 4.0, 7.0, 11.0]
    body_ks: list[float] = []
    cog_masses: list[float] = []
    x_masses: list[float] = []
    y_masses: list[float] = []
    tot_masses: list[float] = []

    for t_mm in thicknesses:
        s = base.model_copy(deep=True)
        s.toolhead.body_wall_thickness_mm = t_mm
        asm = build_assembly(s)
        body_ks.append(asm.toolhead_stage_stiffnesses_n_per_mm["body_n_per_mm"])
        cog_masses.append(asm.node_map["toolhead_cog"].lumped_mass_kg)
        x_masses.append(asm.x_moving_mass_kg)
        y_masses.append(asm.y_moving_mass_kg)
        tot_masses.append(asm.total_mass_kg)

    for idx in range(len(thicknesses) - 1):
        assert body_ks[idx + 1] > body_ks[idx]
        assert cog_masses[idx + 1] == pytest.approx(cog_masses[0], rel=1e-12)
        assert x_masses[idx + 1] == pytest.approx(x_masses[0], rel=1e-12)
        assert y_masses[idx + 1] == pytest.approx(y_masses[0], rel=1e-12)
        assert tot_masses[idx + 1] == pytest.approx(tot_masses[0], abs=1e-8)


def test_hotend_mount_thickness_monotonic_stiffness_and_invariant_mass() -> None:
    """Increasing hotend_mount_thickness_mm stiffens clamp and total hotend while keeping mass invariant."""
    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_monolith_awd.yaml")
    base.toolhead.hotend = HotendSpec(preset_id="chube_conduction")

    thicknesses = [4.0, 8.0, 12.0, 18.0]
    clamp_ks: list[float] = []
    total_ks: list[float] = []
    cog_masses: list[float] = []
    x_masses: list[float] = []
    y_masses: list[float] = []
    tot_masses: list[float] = []

    for t_mm in thicknesses:
        s = base.model_copy(deep=True)
        s.toolhead.hotend_mount_thickness_mm = t_mm
        asm = build_assembly(s)
        clamp_ks.append(asm.toolhead_stage_stiffnesses_n_per_mm["hotend_clamp_n_per_mm"])
        total_ks.append(asm.toolhead_stage_stiffnesses_n_per_mm["hotend_total_n_per_mm"])
        cog_masses.append(asm.node_map["toolhead_cog"].lumped_mass_kg)
        x_masses.append(asm.x_moving_mass_kg)
        y_masses.append(asm.y_moving_mass_kg)
        tot_masses.append(asm.total_mass_kg)

    for idx in range(len(thicknesses) - 1):
        assert clamp_ks[idx + 1] > clamp_ks[idx]
        assert total_ks[idx + 1] > total_ks[idx]
        assert cog_masses[idx + 1] == pytest.approx(cog_masses[0], rel=1e-12)
        assert x_masses[idx + 1] == pytest.approx(x_masses[0], rel=1e-12)
        assert y_masses[idx + 1] == pytest.approx(y_masses[0], rel=1e-12)
        assert tot_masses[idx + 1] == pytest.approx(tot_masses[0], abs=1e-8)

    # Noticeable total hotend stiffness gain on Chube from 4mm -> 18mm clamp
    assert total_ks[-1] > total_ks[0] * 1.15


def test_modal_nodal_rotations_populated_and_v6_vs_chube_hotend_bending() -> None:
    """ModeShape.nodal_rotations serializes cleanly and captures V6 >> Chube hotend bending."""
    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_monolith_awd.yaml")
    s_v6 = base.model_copy(deep=True)
    s_v6.toolhead.hotend = HotendSpec(preset_id="e3d_v6_groove")
    s_chube = base.model_copy(deep=True)
    s_chube.toolhead.hotend = HotendSpec(preset_id="chube_conduction")

    asm_v6 = build_assembly(s_v6)
    asm_chube = build_assembly(s_chube)
    mo_v6 = solve_modal_fea(asm_v6, num_modes=8)
    mo_chube = solve_modal_fea(asm_chube, num_modes=8)

    for mo in (mo_v6, mo_chube):
        d = mo.to_dict()
        assert len(d["modes"]) == len(mo.modes)
        for m, md in zip(mo.modes, d["modes"], strict=True):
            assert "nodal_rotations" in md
            for nid in ("toolhead", "toolhead_cog", "nozzle_tip", "bed_surface"):
                assert nid in m.nodal_rotations
                assert nid in md["nodal_rotations"]
                assert len(m.nodal_rotations[nid]) == 3
                assert all(math.isfinite(val) for val in m.nodal_rotations[nid])
                assert md["nodal_rotations"][nid] == m.nodal_rotations[nid]

    def _max_rel_hotend_bending(asm, mo) -> tuple[float, float]:
        r_stem = asm.node_map["nozzle_tip"].position_m - asm.node_map["toolhead_cog"].position_m
        max_drot = 0.0
        max_dv_elastic = 0.0
        for m in mo.modes:
            rn = np.asarray(m.nodal_rotations["nozzle_tip"], dtype=float)
            rc = np.asarray(m.nodal_rotations["toolhead_cog"], dtype=float)
            vn = np.asarray(m.nodal_vectors["nozzle_tip"], dtype=float)
            vc = np.asarray(m.nodal_vectors["toolhead_cog"], dtype=float)
            drot = float(np.linalg.norm(rn - rc))
            # Subtract rigid-body rotation of toolhead_cog to isolate true elastic hotend bend
            dv_elastic = float(np.linalg.norm((vn - vc) - np.cross(rc, r_stem)))
            max_drot = max(max_drot, drot)
            max_dv_elastic = max(max_dv_elastic, dv_elastic)
        return max_drot, max_dv_elastic

    v6_drot, v6_dv = _max_rel_hotend_bending(asm_v6, mo_v6)
    chube_drot, chube_dv = _max_rel_hotend_bending(asm_chube, mo_chube)
    assert v6_drot > chube_drot * 10.0
    assert v6_dv > chube_dv * 10.0


# ---------------------------------------------------------------------------
# 6. First-Principles Toolhead Construction Architectures
# ---------------------------------------------------------------------------


def test_toolhead_construction_catalog_and_api_exposure() -> None:
    """All 6 TOOLHEAD_CONSTRUCTION_CATALOG styles and preset mappings are exposed via /api/catalog."""
    expected_ids = {
        "multi_piece_clamshell",
        "compact_cage",
        "plate_sandwich_cf",
        "extruder_as_chassis",
        "cnc_monoblock",
        "cantilever_bowden",
    }
    assert set(TOOLHEAD_CONSTRUCTION_CATALOG.keys()) == expected_ids

    client = TestClient(app)
    resp = client.get("/api/catalog")
    assert resp.status_code == 200
    payload = resp.json()

    constructions = payload["toolhead_constructions"]
    assert len(constructions) == 6
    assert {c["id"] for c in constructions} == expected_ids
    for entry in constructions:
        for field in (
            "id",
            "name",
            "short_name",
            "body_stiffness_base_n_per_mm",
            "torsion_mult",
            "mass_delta_g",
            "notes",
        ):
            assert field in entry

    # Every toolhead preset automatically specifies a valid body_construction
    for pid, preset in TOOLHEAD_PRESETS.items():
        assert preset.get("body_construction") in expected_ids, f"Missing/invalid body_construction on {pid}"
    assert TOOLHEAD_PRESETS["stealthburner"]["body_construction"] == "multi_piece_clamshell"
    assert TOOLHEAD_PRESETS["stealthburner_tap"]["body_construction"] == "multi_piece_clamshell"
    assert TOOLHEAD_PRESETS["xol_compact"]["body_construction"] == "compact_cage"
    assert TOOLHEAD_PRESETS["a4t_beacon"]["body_construction"] == "compact_cage"
    assert TOOLHEAD_PRESETS["dragon_burner"]["body_construction"] == "compact_cage"
    assert TOOLHEAD_PRESETS["archetype_cpap"]["body_construction"] == "plate_sandwich_cf"
    assert TOOLHEAD_PRESETS["heavy_uhf_long"]["body_construction"] == "multi_piece_clamshell"


def test_body_construction_monotonic_ranking_and_no_double_counting() -> None:
    """Stage 2 body_n_per_mm ranks constructions monotonically and ignores internal_stiffness_n_per_mm."""
    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_monolith_awd.yaml")
    base.toolhead.hotend = HotendSpec(preset_id="chube_conduction")

    ordered = [
        "multi_piece_clamshell",
        "compact_cage",
        "plate_sandwich_cf",
        "extruder_as_chassis",
        "cnc_monoblock",
    ]
    body_ks: list[float] = []
    torsion_js: list[float] = []

    for bc in ordered:
        s = base.model_copy(deep=True)
        s.toolhead.body_construction = bc  # type: ignore[assignment]
        asm = build_assembly(s)
        body_ks.append(asm.toolhead_stage_stiffnesses_n_per_mm["body_n_per_mm"])
        props = compute_toolhead_link_props(s, l_mount_m=0.025, l_stem_m=0.040)
        torsion_js.append(props["j_mount"] * props["g_mount"])

        # Immune to internal_stiffness_n_per_mm double-counting
        s_low = s.model_copy(deep=True)
        s_low.toolhead.internal_stiffness_n_per_mm = 150.0
        s_high = s.model_copy(deep=True)
        s_high.toolhead.internal_stiffness_n_per_mm = 3000.0

        asm_low = build_assembly(s_low)
        asm_high = build_assembly(s_high)
        assert asm_low.toolhead_stage_stiffnesses_n_per_mm["body_n_per_mm"] == pytest.approx(
            asm_high.toolhead_stage_stiffnesses_n_per_mm["body_n_per_mm"], rel=1e-12
        )
        assert asm_low.toolhead_stage_stiffnesses_n_per_mm[
            "combined_toolhead_n_per_mm"
        ] == pytest.approx(
            asm_high.toolhead_stage_stiffnesses_n_per_mm["combined_toolhead_n_per_mm"], rel=1e-12
        )
        st_low = solve_static_fea(asm_low)
        st_high = solve_static_fea(asm_high)
        assert st_low.toolhead_stiffness_y_n_per_mm == pytest.approx(
            st_high.toolhead_stiffness_y_n_per_mm, rel=1e-12
        )

    for idx in range(len(ordered) - 1):
        assert body_ks[idx + 1] > body_ks[idx]
        assert torsion_js[idx + 1] > torsion_js[idx]

    # body_material and body_wall_thickness_mm still scale body_n_per_mm cleanly when body_construction is set
    s_abs_thin = base.model_copy(deep=True)
    s_abs_thin.toolhead.body_construction = "compact_cage"
    s_abs_thin.toolhead.body_material = "abs_printed"
    s_abs_thin.toolhead.body_wall_thickness_mm = 3.0

    s_cf_thick = s_abs_thin.model_copy(deep=True)
    s_cf_thick.toolhead.body_material = "ppa_cf_printed"
    s_cf_thick.toolhead.body_wall_thickness_mm = 6.0

    assert (
        build_assembly(s_cf_thick).toolhead_stage_stiffnesses_n_per_mm["body_n_per_mm"]
        > build_assembly(s_abs_thin).toolhead_stage_stiffnesses_n_per_mm["body_n_per_mm"] * 2.0
    )


def test_body_construction_mass_deltas_and_geometry_export() -> None:
    """All body_construction styles, stage materials, and z_probe_type keep user-entered mass strictly invariant."""
    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    base_asm = build_assembly(base)

    asms = {}
    for bc, cfg in TOOLHEAD_CONSTRUCTION_CATALOG.items():
        assert cfg["mass_delta_g"] == 0.0
        s = base.model_copy(deep=True)
        s.toolhead.body_construction = bc  # type: ignore[assignment]
        asms[bc] = build_assembly(s)

    for bc in TOOLHEAD_CONSTRUCTION_CATALOG:
        asm = asms[bc]
        assert asm.x_moving_mass_kg == pytest.approx(base_asm.x_moving_mass_kg, rel=1e-12)
        assert asm.y_moving_mass_kg == pytest.approx(base_asm.y_moving_mass_kg, rel=1e-12)
        assert asm.total_mass_kg == pytest.approx(base_asm.total_mass_kg, abs=1e-8)
        assert asm.node_map["toolhead_cog"].lumped_mass_kg == pytest.approx(
            base_asm.node_map["toolhead_cog"].lumped_mass_kg, rel=1e-12
        )

        gdict = asm.to_geometry_dict()
        th_struct = gdict["toolhead_structure"]
        assert th_struct["body_construction"] == bc
        assert th_struct["construction_mass_delta_g"] == 0.0
        assert th_struct["thickness_mass_delta_g"] == 0.0
        assert th_struct["toolhead_body_mass_g"] == pytest.approx(base.toolhead.total_mass_g, abs=1e-9)
        assert th_struct["hotend_mass_g"] == pytest.approx(base.toolhead.hotend.mass_g, abs=1e-9)
        assert th_struct["assembled_toolhead_mass_g"] == pytest.approx(
            base.toolhead.total_mass_g + base.toolhead.hotend.mass_g, abs=1e-9
        )

    # Changing body_material, hotend_mount_material, or z_probe_type also leaves masses invariant
    s_mat = base.model_copy(deep=True)
    s_mat.toolhead.carriage_mount_material = "aluminum_6061_t6"
    s_mat.toolhead.body_material = "aluminum_6061_t6"
    s_mat.toolhead.hotend_mount_material = "titanium_grade5"
    s_mat.toolhead.z_probe_type = "printed_tap"
    asm_mat = build_assembly(s_mat)
    assert asm_mat.x_moving_mass_kg == pytest.approx(base_asm.x_moving_mass_kg, rel=1e-12)
    assert asm_mat.y_moving_mass_kg == pytest.approx(base_asm.y_moving_mass_kg, rel=1e-12)
    assert asm_mat.total_mass_kg == pytest.approx(base_asm.total_mass_kg, abs=1e-8)


def test_legacy_spec_without_body_construction_backward_compatible() -> None:
    """Legacy specs with body_construction=None preserve exact zero mass delta & internal_stiffness behavior."""
    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    assert base.toolhead.body_construction is None

    asm_base = build_assembly(base)
    th_struct = asm_base.to_geometry_dict()["toolhead_structure"]
    assert th_struct["body_construction"] is None
    assert th_struct["construction_mass_delta_g"] == 0.0

    # Matches compact_cage (0.0 g delta) moving mass identically
    s_cage = base.model_copy(deep=True)
    s_cage.toolhead.body_construction = "compact_cage"
    assert asm_base.x_moving_mass_kg == pytest.approx(
        build_assembly(s_cage).x_moving_mass_kg, rel=1e-12
    )

    # Legacy internal_stiffness_n_per_mm continues to drive body_n_per_mm when body_construction is None
    s_low = base.model_copy(deep=True)
    s_low.toolhead.internal_stiffness_n_per_mm = 150.0
    s_high = base.model_copy(deep=True)
    s_high.toolhead.internal_stiffness_n_per_mm = 3000.0

    k_low = build_assembly(s_low).toolhead_stage_stiffnesses_n_per_mm["body_n_per_mm"]
    k_high = build_assembly(s_high).toolhead_stage_stiffnesses_n_per_mm["body_n_per_mm"]
    assert k_high > k_low * 5.0


def test_decoupled_rail_carriage_vs_backplate_stiffness_and_abs_calibration() -> None:
    """Verify exact series split of rail_carriage vs backplate_only stiffness and 2.1 GPa ABS calibration."""
    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    base.frame.chamber_temp_c = 25.0
    base.toolhead.z_probe_type = "static_eddy"

    expected_abs_bases = {
        "multi_piece_clamshell": 110.0,
        "compact_cage": 185.0,
        "cantilever_bowden": 230.0,
        "plate_sandwich_cf": 270.0,
        "extruder_as_chassis": 320.0,
        "cnc_monoblock": 480.0,
    }
    for bc, expected_k in expected_abs_bases.items():
        assert TOOLHEAD_CONSTRUCTION_CATALOG[bc]["body_stiffness_base_n_per_mm"] == pytest.approx(
            expected_k, rel=1e-12
        )

    # 5 mm Al 6061-T6 backplate + 4 mm printed ABS body (extruder_as_chassis & compact_cage)
    for bc in ("extruder_as_chassis", "compact_cage"):
        s_al_plate = base.model_copy(deep=True)
        s_al_plate.toolhead.carriage_mount_material = "aluminum_6061_t6"
        s_al_plate.toolhead.carriage_plate_thickness_mm = 5.0
        s_al_plate.toolhead.body_material = "abs_printed"
        s_al_plate.toolhead.body_wall_thickness_mm = 4.0
        s_al_plate.toolhead.body_construction = bc  # type: ignore[assignment]

        asm = build_assembly(s_al_plate)
        stiff = asm.toolhead_stage_stiffnesses_n_per_mm
        gdict_stiff = asm.to_geometry_dict()["toolhead_structure"]["stage_stiffnesses_n_per_mm"]

        for key in ("rail_carriage_n_per_mm", "backplate_only_n_per_mm", "carriage_mount_n_per_mm"):
            assert key in stiff
            assert key in gdict_stiff
            assert gdict_stiff[key] == pytest.approx(round(stiff[key], 2), abs=1e-9)

        assert stiff["body_n_per_mm"] == pytest.approx(expected_abs_bases[bc], rel=1e-9)
        assert stiff["backplate_only_n_per_mm"] > stiff["body_n_per_mm"] * 10.0

        # Exact series decomposition: 1/k_rail + 1/k_backplate == 1/k_carriage_mount
        inv_sum = (1.0 / stiff["rail_carriage_n_per_mm"]) + (
            1.0 / stiff["backplate_only_n_per_mm"]
        )
        assert inv_sum == pytest.approx(1.0 / stiff["carriage_mount_n_per_mm"], rel=1e-12)

    # Reverse dominance: 3 mm printed ABS backplate + 6 mm Al 6061-T6 CNC monoblock body
    s_al_body = base.model_copy(deep=True)
    s_al_body.toolhead.carriage_mount_material = "abs_printed"
    s_al_body.toolhead.carriage_plate_thickness_mm = 3.0
    s_al_body.toolhead.body_material = "aluminum_6061_t6"
    s_al_body.toolhead.body_wall_thickness_mm = 6.0
    s_al_body.toolhead.body_construction = "cnc_monoblock"

    asm_rev = build_assembly(s_al_body)
    stiff_rev = asm_rev.toolhead_stage_stiffnesses_n_per_mm
    assert stiff_rev["body_n_per_mm"] > stiff_rev["backplate_only_n_per_mm"] * 10.0
    inv_sum_rev = (1.0 / stiff_rev["rail_carriage_n_per_mm"]) + (
        1.0 / stiff_rev["backplate_only_n_per_mm"]
    )
    assert inv_sum_rev == pytest.approx(1.0 / stiff_rev["carriage_mount_n_per_mm"], rel=1e-12)


def test_rail_quality_and_preload_stiffness_drag_and_catalog() -> None:
    """Verify rail_quality catalog exposure, monotonic Hertzian stiffness & drag, and invariant backplate."""
    ordered_qualities = [
        "budget_generic",
        "midgrade_matched",
        "hiwin_genuine",
        "precision_thk_nsk",
    ]
    ordered_preloads = ["ZF", "Z0", "Z1", "ZA", "ZB"]

    assert list(RAIL_QUALITY_CATALOG.keys()) == ordered_qualities

    client = TestClient(app)
    resp = client.get("/api/catalog")
    assert resp.status_code == 200
    payload = resp.json()
    assert "rail_qualities" in payload
    assert [q["id"] for q in payload["rail_qualities"]] == ordered_qualities
    for entry in payload["rail_qualities"]:
        for field in (
            "id",
            "name",
            "short_name",
            "stiffness_mult",
            "drag_mult",
            "cost_mult",
            "notes",
        ):
            assert field in entry

    # Direct Hertzian component checks across rail_quality and rail_preload
    car_by_q = [
        hertzian_rail_carriage_stiffness("mgn12_rail", preload="Z1", block_type="H", rail_quality=q)
        for q in ordered_qualities
    ]
    for idx in range(len(ordered_qualities) - 1):
        for k_key in ("k_yy_n_m", "k_zz_n_m", "k_rx_nm_rad", "k_ry_nm_rad", "k_rz_nm_rad"):
            assert car_by_q[idx + 1][k_key] > car_by_q[idx][k_key]
        assert car_by_q[idx]["rolling_friction_n"] > car_by_q[idx + 1]["rolling_friction_n"]
        assert car_by_q[idx]["seal_drag_n"] > car_by_q[idx + 1]["seal_drag_n"]
        assert car_by_q[idx]["breakaway_friction_n"] > car_by_q[idx + 1]["breakaway_friction_n"]

    assert car_by_q[0]["breakaway_friction_n"] > car_by_q[-1]["breakaway_friction_n"]
    car_zf = hertzian_rail_carriage_stiffness("mgn12_rail", preload="ZF", block_type="H", rail_quality="hiwin_genuine")
    car_zb = hertzian_rail_carriage_stiffness("mgn12_rail", preload="ZB", block_type="H", rail_quality="hiwin_genuine")
    assert car_zf["breakaway_friction_n"] < car_zb["breakaway_friction_n"]

    # Assembly-level check: rail_carriage_n_per_mm rises monotonically with quality & preload, backplate stays invariant
    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    assert base.gantry.rail_quality == "hiwin_genuine"
    base.toolhead.carriage_mount_material = "aluminum_6061_t6"

    base_backplate_k = build_assembly(base).toolhead_stage_stiffnesses_n_per_mm["backplate_only_n_per_mm"]

    q_carriage_ks: list[float] = []
    for q in ordered_qualities:
        s = base.model_copy(deep=True)
        s.gantry.rail_quality = q  # type: ignore[assignment]
        asm = build_assembly(s)
        stiff = asm.toolhead_stage_stiffnesses_n_per_mm
        q_carriage_ks.append(stiff["rail_carriage_n_per_mm"])
        assert stiff["backplate_only_n_per_mm"] == pytest.approx(base_backplate_k, rel=1e-12)
        lr = asm.to_geometry_dict()["linear_rails"]
        assert lr["rail_quality"] == q
        assert lr["rail_preload"] == s.gantry.rail_preload

    for idx in range(len(ordered_qualities) - 1):
        assert q_carriage_ks[idx + 1] > q_carriage_ks[idx]

    p_carriage_ks: list[float] = []
    for p in ordered_preloads:
        s = base.model_copy(deep=True)
        s.gantry.rail_preload = p  # type: ignore[assignment]
        asm = build_assembly(s)
        stiff = asm.toolhead_stage_stiffnesses_n_per_mm
        p_carriage_ks.append(stiff["rail_carriage_n_per_mm"])
        assert stiff["backplate_only_n_per_mm"] == pytest.approx(base_backplate_k, rel=1e-12)
        assert asm.to_geometry_dict()["linear_rails"]["rail_preload"] == p

    for idx in range(len(ordered_preloads) - 1):
        assert p_carriage_ks[idx + 1] > p_carriage_ks[idx]


# ---------------------------------------------------------------------------
# 6. Gantry Weight Classes & Exact Y-Gantry Moving Mass Breakdown
# ---------------------------------------------------------------------------


def test_gantry_weight_classes_catalog_and_validation() -> None:
    """/api/catalog exposes all 5 gantry_weight_classes and GantrySpec enforces carriage_mass_g bounds."""
    expected_classes = [
        ("ultralight_printed", 85.0, "abs_printed"),
        ("standard_printed", 140.0, "abs_printed"),
        ("stiffness_opt_printed", 185.0, "pet_cf_printed"),
        ("cnc_aluminum_light", 210.0, "aluminum_6061_t6"),
        ("monolith_cnc_awd", 214.0, "aluminum_6061_t6"),
        ("cnc_aluminum_hd", 280.0, "aluminum_6061_t6"),
    ]
    assert list(GANTRY_WEIGHT_CLASSES.keys()) == [cid for cid, _, _ in expected_classes]

    client = TestClient(app)
    resp = client.get("/api/catalog")
    assert resp.status_code == 200
    payload = resp.json()
    assert "gantry_weight_classes" in payload
    cat_classes = payload["gantry_weight_classes"]
    assert [c["id"] for c in cat_classes] == [cid for cid, _, _ in expected_classes]

    for entry, (cid, mass_g, bracket_mat) in zip(cat_classes, expected_classes, strict=True):
        assert entry["id"] == cid
        assert entry["carriage_mass_g"] == pytest.approx(mass_g, abs=1e-9)
        assert entry["suggested_bracket_material"] == bracket_mat
        for field in ("name", "short_name", "notes"):
            assert isinstance(entry[field], str) and len(entry[field]) > 0

    # Sensible validation bounds on GantrySpec.carriage_mass_g
    assert GantrySpec(carriage_mass_g=20.0).carriage_mass_g == 20.0
    assert GantrySpec(carriage_mass_g=1200.0).carriage_mass_g == 1200.0
    with pytest.raises(Exception):
        GantrySpec(carriage_mass_g=5.0)
    with pytest.raises(Exception):
        GantrySpec(carriage_mass_g=2500.0)


def test_exact_x_beam_mass_and_y_gantry_breakdown_zero_double_counting() -> None:
    """Verify exact X-beam mass = area * span * rho across profiles/materials with rail & backer."""
    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    span_m = base.frame.width_mm * 1e-3
    rail_prof = get_profile("mgn12_rail")
    rail_mat = get_material("steel_4140")
    expected_rail_g = rail_prof.area_m2 * span_m * rail_mat.density_kg_m3 * 1000.0

    combos = [
        ("extrusion_2020", "aluminum_6063_t5", "none"),
        ("cf_tube_20x20x2", "carbon_fiber_tube", "none"),
        ("ti_tube_20x20x1_2", "titanium_grade5", "none"),
        ("extrusion_2040", "aluminum_6063_t5", "none"),
        ("cf_tube_20x20x2", "carbon_fiber_tube", "titanium_grade5"),
        ("extrusion_2020", "aluminum_6063_t5", "steel_4140"),
    ]
    for prof_id, mat_id, backer_id in combos:
        s = base.model_copy(deep=True)
        s.gantry.x_beam_profile = prof_id
        s.gantry.x_beam_material = mat_id
        s.gantry.x_rail_profile = "mgn12_rail"
        s.gantry.x_rail_mount = "front"
        s.gantry.rail_backer_material = backer_id  # type: ignore[assignment]
        s.gantry.rail_backer_thickness_mm = 3.0

        asm = build_assembly(s)
        nmap = asm.node_map
        prof = get_profile(prof_id)
        mat = get_material(mat_id)
        expected_beam_kg = prof.area_m2 * span_m * mat.density_kg_m3
        expected_beam_g = expected_beam_kg * 1000.0

        # Check raw member mass stripped of rail & backer matches bare beam formula to 1e-6
        stripped_beam_kg = sum(
            m.mass_kg(nmap) - m.rail_mass_kg - m.backer_mass_kg
            for m in asm.members
            if m.role == "gantry"
        )
        assert stripped_beam_kg == pytest.approx(expected_beam_kg, rel=1e-9, abs=1e-9)
        assert asm.x_beam_mass_kg == pytest.approx(expected_beam_kg, rel=1e-9, abs=1e-9)
        assert asm.x_rail_mass_kg * 1000.0 == pytest.approx(expected_rail_g, rel=1e-9, abs=1e-9)

        ms = asm.to_geometry_dict()["mass_summary"]
        assert ms["x_beam_mass_g"] == pytest.approx(round(expected_beam_g, 1), abs=0.05)
        assert ms["x_rail_mass_g"] == pytest.approx(round(expected_rail_g, 1), abs=0.05)
        if backer_id == "none":
            assert ms["x_backer_mass_g"] == 0.0
        else:
            assert ms["x_backer_mass_g"] > 0.0
            assert asm.x_backer_mass_kg * 1000.0 == pytest.approx(
                sum(m.backer_mass_kg for m in asm.members if m.role == "gantry") * 1000.0,
                rel=1e-9,
            )

        # Full Y moving mass sum identity (exact in kg, rounded consistently in mass_summary)
        sum_kg = (
            asm.toolhead_moving_mass_kg
            + asm.x_beam_mass_kg
            + asm.x_rail_mass_kg
            + asm.x_backer_mass_kg
            + asm.y_carriage_joints_mass_kg
        )
        assert sum_kg == pytest.approx(asm.y_moving_mass_kg, rel=1e-12, abs=1e-12)
        assert ms["toolhead_moving_mass_g"] + ms["x_beam_mass_g"] + ms["x_rail_mass_g"] + ms[
            "x_backer_mass_g"
        ] + ms["y_carriage_joints_mass_g"] == pytest.approx(ms["y_moving_mass_g"], abs=0.3)

    # Also verify Bedslinger and Delta populate breakdown cleanly
    bs = PrinterSpec.from_yaml(EXAMPLES_DIR / "cartesian_bedslinger.yaml")
    asm_bs = build_assembly(bs)
    ms_bs = asm_bs.to_geometry_dict()["mass_summary"]
    assert ms_bs["x_beam_mass_g"] > 0.0
    assert ms_bs["x_rail_mass_g"] > 0.0
    assert ms_bs["y_carriage_joints_mass_g"] == pytest.approx(bs.gantry.carriage_mass_g, abs=0.1)
    assert ms_bs["toolhead_moving_mass_g"] > 0.0

    delta = PrinterSpec.from_yaml(EXAMPLES_DIR / "delta_racer.yaml")
    asm_dl = build_assembly(delta)
    ms_dl = asm_dl.to_geometry_dict()["mass_summary"]
    assert ms_dl["x_beam_mass_g"] == 0.0
    assert ms_dl["x_rail_mass_g"] == 0.0
    assert ms_dl["y_carriage_joints_mass_g"] > 0.0
    assert ms_dl["toolhead_moving_mass_g"] > 0.0


def test_gantry_weight_class_carriage_mass_delta_and_monotonic_y_resonance() -> None:
    """Sweeping Ultralight Printed (85g) -> CNC Al HD (280g) adds exact +195g and lowers Y resonance."""
    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    base.gantry.bracket_material = "abs_printed"

    ordered_ids = [
        "ultralight_printed",
        "standard_printed",
        "stiffness_opt_printed",
        "cnc_aluminum_light",
        "cnc_aluminum_hd",
    ]
    asms = []
    modals = []
    for cid in ordered_ids:
        s = base.model_copy(deep=True)
        s.gantry.carriage_mass_g = float(GANTRY_WEIGHT_CLASSES[cid]["carriage_mass_g"])
        asm = build_assembly(s)
        asms.append(asm)
        modals.append(solve_modal_fea(asm))

    ms_ultra = asms[0].to_geometry_dict()["mass_summary"]
    ms_hd = asms[-1].to_geometry_dict()["mass_summary"]
    assert (ms_hd["y_carriage_joints_mass_g"] - ms_ultra["y_carriage_joints_mass_g"]) == pytest.approx(
        195.0, abs=1e-6
    )
    assert (ms_hd["y_moving_mass_g"] - ms_ultra["y_moving_mass_g"]) == pytest.approx(195.0, abs=0.2)
    assert (asms[-1].y_moving_mass_kg - asms[0].y_moving_mass_kg) * 1000.0 == pytest.approx(
        195.0, abs=1e-6
    )

    for idx in range(len(ordered_ids) - 1):
        assert asms[idx + 1].y_carriage_joints_mass_kg > asms[idx].y_carriage_joints_mass_kg
        assert asms[idx + 1].y_moving_mass_kg > asms[idx].y_moving_mass_kg
        sh_cur = modals[idx].shaper_y
        sh_nxt = modals[idx + 1].shaper_y
        assert sh_cur is not None and sh_nxt is not None
        assert sh_nxt.belt_freq_hz < sh_cur.belt_freq_hz
        assert sh_nxt.drive_freq_hz < sh_cur.drive_freq_hz
        assert sh_nxt.effective_freq_hz < sh_cur.effective_freq_hz
        assert modals[idx + 1].fundamental_freq_hz < modals[idx].fundamental_freq_hz


def test_hotend_mass_g_presets_nozzle_cantilever_resonance_and_mass_conservation() -> None:
    """All HOTEND_CATALOG entries expose datasheet mass_g; heavier hotend increases nozzle_tip lump & lowers cantilever resonance."""
    expected_masses_g = {
        "e3d_v6_groove": 42.0,
        "e3d_revo_voron": 30.0,
        "phaetus_dragon_hf": 46.0,
        "phaetus_dragon_uhf": 56.0,
        "phaetus_rapido_2_hf": 50.0,
        "phaetus_rapido_2_uhf": 62.0,
        "mellow_goliath_air": 96.0,
        "chube_compact": 62.0,
        "chube_air": 86.0,
        "chube_conduction": 82.0,
    }
    assert set(HOTEND_CATALOG.keys()) == set(expected_masses_g.keys())
    assert HotendSpec().mass_g == pytest.approx(45.0, abs=1e-9)

    for hid, exp_mass in expected_masses_g.items():
        assert HOTEND_CATALOG[hid]["mass_g"] == pytest.approx(exp_mass, abs=1e-9)
        hs = HotendSpec(preset_id=hid)
        assert hs.mass_g == pytest.approx(exp_mass, abs=1e-9)

    # Pure entered-weight assembled mass & nozzle cantilever sweep on compliant E3D V6 hotend
    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_monolith_awd.yaml")
    masses_g = [30.0, 60.0, 120.0]
    asms = []
    modals = []
    cantilever_hzs: list[float] = []

    for m_g in masses_g:
        s = base.model_copy(deep=True)
        s.toolhead.hotend = HotendSpec(preset_id="e3d_v6_groove", mass_g=m_g)
        asm = build_assembly(s)
        mo = solve_modal_fea(asm, num_modes=12)
        asms.append(asm)
        modals.append(mo)

        expected_th_kg = (s.toolhead.total_mass_g + m_g) * 1e-3
        assert asm.toolhead_moving_mass_kg == pytest.approx(expected_th_kg, rel=1e-12)
        assert asm.x_moving_mass_kg == pytest.approx(expected_th_kg + 0.04, rel=1e-12)

        # Exact mass conservation across toolhead carriage, CoG, and nozzle_tip nodes
        nmap = asm.node_map
        th_node_sum_kg = (
            nmap["toolhead"].lumped_mass_kg
            + nmap["toolhead_cog"].lumped_mass_kg
            + nmap["nozzle_tip"].lumped_mass_kg
        )
        assert th_node_sum_kg == pytest.approx(expected_th_kg, rel=1e-12)
        assert nmap["nozzle_tip"].lumped_mass_kg == pytest.approx(
            max(0.010, 0.28 * m_g * 1e-3), rel=1e-12
        )

        th_struct = asm.to_geometry_dict()["toolhead_structure"]
        assert th_struct["toolhead_body_mass_g"] == pytest.approx(s.toolhead.total_mass_g, abs=1e-9)
        assert th_struct["hotend_mass_g"] == pytest.approx(m_g, abs=1e-9)
        assert th_struct["assembled_toolhead_mass_g"] == pytest.approx(
            s.toolhead.total_mass_g + m_g, abs=1e-9
        )

        whip_modes = [
            md.frequency_hz
            for md in mo.modes
            if md.subsystem_share_pct.get("toolhead", 0.0) > 80.0 and md.nozzle_whip_ratio > 1.5
        ]
        assert len(whip_modes) >= 1
        cantilever_hzs.append(min(whip_modes))

    for idx in range(len(masses_g) - 1):
        dm_kg = (masses_g[idx + 1] - masses_g[idx]) * 1e-3
        assert (asms[idx + 1].x_moving_mass_kg - asms[idx].x_moving_mass_kg) == pytest.approx(
            dm_kg, abs=1e-12
        )
        assert (asms[idx + 1].y_moving_mass_kg - asms[idx].y_moving_mass_kg) == pytest.approx(
            dm_kg, abs=1e-12
        )
        assert (asms[idx + 1].total_mass_kg - asms[idx].total_mass_kg) == pytest.approx(
            dm_kg, abs=1e-9
        )
        assert (
            asms[idx + 1].node_map["nozzle_tip"].lumped_mass_kg
            > asms[idx].node_map["nozzle_tip"].lumped_mass_kg
        )
        assert cantilever_hzs[idx + 1] < cantilever_hzs[idx]
        assert modals[idx + 1].nozzle_pitch_twist_freq_hz < modals[idx].nozzle_pitch_twist_freq_hz






