"""Tests for Monolith3D structural, thermal-mount, bearing-CTE, belt-elastomer, Z-joint,
unsupported-rail, and spatial input-shaper physics features.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from printsim.engine import evaluate_printer
from printsim.fea.components import (
    belt_effective_ea_n,
    compute_bearing_cte_clearance_um,
    compute_stepper_mount_temp_c,
    gerbert_tooth_mesh_stiffness_n_m,
    hertzian_rail_carriage_stiffness,
)
from printsim.geometry import build_assembly
from printsim.schema import PrinterSpec, get_item_confidence

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def _load_preset(filename: str) -> PrinterSpec:
    return PrinterSpec.from_yaml(EXAMPLES_DIR / filename)


# ---------------------------------------------------------------------------
# 1. Monolith Presets End-to-End Smoke & Sanity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "preset_file",
    ["monolith_zero_m0.yaml", "monolith_sheet_metal_v2.yaml"],
)
def test_monolith_presets_evaluate_cleanly(preset_file: str) -> None:
    """Monolith Zero M0 (1515 + unsupported rail) and Sheet Metal v1.5 evaluate cleanly."""
    spec = _load_preset(preset_file)
    report = evaluate_printer(spec)
    d = report.to_dict()

    assert report.static_fea.combined_max_deflection_um > 0.0
    assert report.modal_fea.fundamental_freq_hz > 25.0
    assert report.modal_fea.shaper_x is not None
    assert report.modal_fea.shaper_y is not None
    assert report.modal_fea.shaper_x.effective_freq_hz > 30.0
    assert report.modal_fea.shaper_y.effective_freq_hz > 30.0
    assert report.workspace_frf is not None
    assert "is_spatial_spread_pct" in d["workspace_frf"]
    assert d["workspace_frf"]["is_spatial_spread_pct"] >= 0.0

    ts = d["thermal_structural"]
    for key in (
        "motor_mount_temp_c",
        "bearing_temp_c",
        "bearing_cte_clearance_um",
        "bearing_cte_pressfit_warning",
        "toolhead_belt_cog_z_offset_mm",
        "toolhead_nozzle_y_lever_mm",
        "toolhead_x_roll_warning",
    ):
        assert key in ts
    # Monolith Toolhead is belt-coplanar (dz=-1.5 mm, belt_plane=-1.5 mm, nozzle_y=-14 mm)
    assert report.thermal_structural.toolhead_belt_cog_z_offset_mm == pytest.approx(0.0, abs=1e-6)
    assert report.thermal_structural.toolhead_x_roll_warning is False


# ---------------------------------------------------------------------------
# 2. Unsupported Bare Rail X-Gantry, Backer, Hollow MGN12 & Compact 1515/12x12
# ---------------------------------------------------------------------------


def test_unsupported_bare_rail_vs_5mm_steel_backer_and_no_double_count() -> None:
    """Bare unsupported rail has zero phantom beam mass, zero bimetallic bow, and stiffens with 5mm backer."""
    spec_bare = _load_preset("monolith_zero_m0.yaml")
    assert spec_bare.gantry.x_beam_profile == "unsupported_rail"
    assert spec_bare.gantry.rail_backer_material == "none"

    asm_bare = build_assembly(spec_bare)
    # No phantom X-beam mass double-counting
    assert asm_bare.x_beam_mass_kg == pytest.approx(0.0, abs=1e-9)
    assert asm_bare.x_rail_mass_kg > 0.05
    assert asm_bare.x_backer_mass_kg == pytest.approx(0.0, abs=1e-9)

    ev_bare = evaluate_printer(spec_bare)
    # Bare monolithic steel rail has zero bimetallic thermal bow
    assert ev_bare.thermal_structural.gantry_bimetallic_bow_um == pytest.approx(0.0, abs=1e-9)
    assert ev_bare.thermal_structural.gantry_bow_direction == "neutral"

    # Add Monolith Zero 5 mm steel opposite-face backer directly onto bare MGN12 rail
    spec_backed = spec_bare.model_copy(deep=True)
    spec_backed.gantry.rail_backer_material = "steel_4140"
    spec_backed.gantry.rail_backer_thickness_mm = 5.0

    asm_backed = build_assembly(spec_backed)
    ev_backed = evaluate_printer(spec_backed)

    assert asm_backed.x_beam_mass_kg == pytest.approx(0.0, abs=1e-9)
    assert asm_backed.x_backer_mass_kg > 0.03
    assert asm_backed.x_rail_ei_y_boost_pct > 100.0  # 5 mm backer >2.4x bare MGN12 Y-bending inertia
    # Monotonic Y-direction structural stiffness increase (lower Y deflection / higher Y stiffness)
    assert ev_backed.static_fea.accel_deflection_y_um < ev_bare.static_fea.accel_deflection_y_um
    assert ev_backed.static_fea.toolhead_stiffness_y_n_per_mm > ev_bare.static_fea.toolhead_stiffness_y_n_per_mm
    # Steel-on-steel backer has matched CTE -> still zero bimetallic bow
    assert ev_backed.thermal_structural.gantry_bimetallic_bow_um == pytest.approx(0.0, abs=1e-6)


def test_al_tube_12x12x1_on_1515_frame_evaluates_cleanly() -> None:
    """Compact 1515 frame with al_tube_12x12x1 X-beam builds and solves without 2020 assumptions."""
    spec = _load_preset("monolith_zero_m0.yaml")
    spec.gantry.x_beam_profile = "al_tube_12x12x1"
    spec.gantry.x_beam_material = "aluminum_6063_t5"
    ev = evaluate_printer(spec)
    assert ev.assembly.x_beam_mass_kg > 0.0
    assert ev.modal_fea.fundamental_freq_hz > 25.0
    assert ev.static_fea.combined_max_deflection_um > 0.0


def test_hollow_mgn12_lighter_and_slightly_softer_raceway() -> None:
    """Hollow MGN12H rail is lighter than solid MGN12H with ~0.92x Hertzian raceway stiffness."""
    car_solid = hertzian_rail_carriage_stiffness("mgn12_rail", preload="Z1", block_type="H")
    car_hollow = hertzian_rail_carriage_stiffness("mgn12_hollow_rail", preload="Z1", block_type="H")

    assert car_hollow["block_mass_g"] == pytest.approx(car_solid["block_mass_g"], rel=1e-9)
    assert car_hollow["k_yy_n_m"] == pytest.approx(0.92 * car_solid["k_yy_n_m"], rel=1e-6)
    assert car_hollow["k_rx_nm_rad"] == pytest.approx(0.92 * car_solid["k_rx_nm_rad"], rel=1e-6)

    spec_solid = _load_preset("monolith_sheet_metal_v2.yaml")
    spec_solid.gantry.x_rail_profile = "mgn12_rail"
    spec_hollow = spec_solid.model_copy(deep=True)
    spec_hollow.gantry.x_rail_profile = "mgn12_hollow_rail"

    asm_solid = build_assembly(spec_solid)
    asm_hollow = build_assembly(spec_hollow)
    assert asm_hollow.x_rail_mass_kg < asm_solid.x_rail_mass_kg * 0.72
    assert asm_hollow.y_moving_mass_kg < asm_solid.y_moving_mass_kg


# ---------------------------------------------------------------------------
# 3. Stepper Self-Heating & 2 mm PTFE Mount Thermal Gasket
# ---------------------------------------------------------------------------


def test_ptfe_2mm_gasket_reduces_mount_temp_14c_and_recovers_polymer_stiffness() -> None:
    """Calibrated to Monolith 25 W @ 80 °C study: ptfe_2mm drops mount temp ~14 °C and raises retention."""
    spec_bare = _load_preset("corexy_voron_style.yaml")
    spec_bare.frame.chamber_temp_c = 80.0
    spec_bare.gantry.bracket_material = "abs_printed"
    spec_bare.gantry.stepper_thermal_gasket = "none"
    # Set exact 25 W copper dissipation: P = 2 * I^2 * R = 2 * 2.5^2 * 2.0 = 25.0 W
    spec_bare.kinematics.xy_stepper.run_current_a = 2.5
    spec_bare.kinematics.xy_stepper.phase_resistance_ohm = 2.0

    spec_ptfe = spec_bare.model_copy(deep=True)
    spec_ptfe.gantry.stepper_thermal_gasket = "ptfe_2mm"

    t_bare, rise_bare = compute_stepper_mount_temp_c(spec_bare)
    t_ptfe, rise_ptfe = compute_stepper_mount_temp_c(spec_ptfe)

    assert t_bare == pytest.approx(106.5, abs=0.2)
    assert t_ptfe == pytest.approx(92.5, abs=0.2)
    assert 12.0 <= (t_bare - t_ptfe) <= 15.0
    assert rise_bare > rise_ptfe > 0.0

    ev_bare = evaluate_printer(spec_bare)
    ev_ptfe = evaluate_printer(spec_ptfe)

    assert ev_ptfe.thermal_structural.motor_mount_temp_c < ev_bare.thermal_structural.motor_mount_temp_c - 12.0
    assert ev_ptfe.thermal_structural.bracket_modulus_retention_pct > ev_bare.thermal_structural.bracket_modulus_retention_pct
    assert ev_ptfe.kinematics.bracket_stiffness_n_per_mm > ev_bare.kinematics.bracket_stiffness_n_per_mm


# ---------------------------------------------------------------------------
# 4. Bearing Press-Fit CTE Thermal Clearance (F695 in Al > 66 °C)
# ---------------------------------------------------------------------------


def test_f695_bearing_cte_clearance_warning_and_disc_spring_mitigation() -> None:
    """Al housing > 66 °C without disc springs opens F695 CTE radial play; disc_spring_m5 clears it."""
    spec_al_none = _load_preset("voron_monolith_awd.yaml")
    spec_al_none.frame.chamber_temp_c = 80.0
    spec_al_none.gantry.bracket_material = "aluminum_6061_t6"
    spec_al_none.gantry.idler_axial_preload = "none"

    clr_80, warn_80 = compute_bearing_cte_clearance_um(spec_al_none, 80.0)
    expected_clr = 0.5 * 13_000.0 * 12.1e-6 * (80.0 - 66.0)
    assert warn_80 is True
    assert clr_80 == pytest.approx(expected_clr, rel=1e-6)

    # Below 66 °C threshold -> no clearance loss
    clr_60, warn_60 = compute_bearing_cte_clearance_um(spec_al_none, 60.0)
    assert warn_60 is False
    assert clr_60 == pytest.approx(0.0, abs=1e-9)

    # With M5 Belleville disc-spring axial preload -> zero radial play & no warning
    spec_al_disc = spec_al_none.model_copy(deep=True)
    spec_al_disc.gantry.idler_axial_preload = "disc_spring_m5"
    clr_disc, warn_disc = compute_bearing_cte_clearance_um(spec_al_disc, 80.0)
    assert warn_disc is False
    assert clr_disc == pytest.approx(0.0, abs=1e-9)

    # Full evaluate_printer propagation into ThermalStructuralReport and KinematicsReport.bearing_play_um
    ev_none = evaluate_printer(spec_al_none)
    ev_disc = evaluate_printer(spec_al_disc)
    assert ev_none.thermal_structural.bearing_cte_pressfit_warning is True
    assert ev_none.thermal_structural.bearing_cte_clearance_um == pytest.approx(expected_clr, rel=1e-3)
    assert ev_disc.thermal_structural.bearing_cte_pressfit_warning is False
    assert ev_disc.thermal_structural.bearing_cte_clearance_um == pytest.approx(0.0, abs=1e-6)
    assert ev_none.kinematics.bearing_play_um == pytest.approx(
        ev_disc.kinematics.bearing_play_um + expected_clr, rel=1e-3
    )

    # Steel sheet-metal housing has matched CTE with steel bearing -> no CTE warning even without disc springs
    spec_steel = spec_al_none.model_copy(deep=True)
    spec_steel.gantry.bracket_material = "sheet_metal_steel"
    clr_steel, warn_steel = compute_bearing_cte_clearance_um(spec_steel, 80.0)
    assert warn_steel is False
    assert clr_steel == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 5. Belt Base-Elastomer Seating & Drag (GT2 10 mm EPDM vs GT3 10 mm 2MGT)
# ---------------------------------------------------------------------------


def test_gt2_10mm_outperforms_gt3_10mm_at_matched_89n_tension() -> None:
    """Matched 89 N shootout: GT2 10 mm has stiffer EA/tooth mesh, higher ZV freq, and lower drag than GT3 10 mm."""
    spec_gt2 = _load_preset("monolith_sheet_metal_v2.yaml")
    spec_gt2.gantry.belt.type = "gt2_10mm"
    spec_gt2.gantry.belt.pretension_n = 89.0

    spec_gt3 = spec_gt2.model_copy(deep=True)
    spec_gt3.gantry.belt.type = "gt3_10mm"
    spec_gt3.gantry.belt.pretension_n = 89.0

    ea_gt2 = belt_effective_ea_n(spec_gt2.gantry.belt, span_m=0.45)
    ea_gt3 = belt_effective_ea_n(spec_gt3.gantry.belt, span_m=0.45)
    assert ea_gt2 > ea_gt3

    kt_gt2 = gerbert_tooth_mesh_stiffness_n_m(spec_gt2.gantry.belt, ea_gt2)
    kt_gt3 = gerbert_tooth_mesh_stiffness_n_m(spec_gt3.gantry.belt, ea_gt3)
    assert kt_gt2 > kt_gt3

    ev_gt2 = evaluate_printer(spec_gt2)
    ev_gt3 = evaluate_printer(spec_gt3)

    assert ev_gt2.kinematics.belt_stiffness_n_per_mm > ev_gt3.kinematics.belt_stiffness_n_per_mm
    assert ev_gt2.modal_fea.shaper_x is not None and ev_gt3.modal_fea.shaper_x is not None
    assert ev_gt2.modal_fea.shaper_y is not None and ev_gt3.modal_fea.shaper_y is not None
    assert ev_gt2.modal_fea.shaper_x.effective_freq_hz > ev_gt3.modal_fea.shaper_x.effective_freq_hz
    assert ev_gt2.modal_fea.shaper_y.effective_freq_hz > ev_gt3.modal_fea.shaper_y.effective_freq_hz
    # Higher rolling drag on GT3 slightly increases required motor torque
    assert ev_gt3.kinematics.required_torque_x_nm > ev_gt2.kinematics.required_torque_x_nm


# ---------------------------------------------------------------------------
# 6. Flying-Gantry V2 Single Printed vs Double SLM Z-Joints
# ---------------------------------------------------------------------------


def test_flying_gantry_double_slm_z_joints_stiffer_than_single_printed() -> None:
    """Double SLM V2 Z-joints significantly stiffen flying gantry vs Single Printed Z-joints."""
    spec_single = _load_preset("monolith_sheet_metal_v2.yaml")
    assert spec_single.is_flying_gantry
    spec_single.gantry.z_joint_type = "single"
    spec_single.gantry.z_joint_material = "printed"

    spec_double_slm = spec_single.model_copy(deep=True)
    spec_double_slm.gantry.z_joint_type = "double"
    spec_double_slm.gantry.z_joint_material = "slm"

    asm_single = build_assembly(spec_single)
    asm_double = build_assembly(spec_double_slm)

    sp_single = next(sp for sp in asm_single.fastener_springs if sp.id == "z_joint_fl")
    sp_double = next(sp for sp in asm_double.fastener_springs if sp.id == "z_joint_fl")
    assert sp_double.k_shear_n_m == pytest.approx(sp_single.k_shear_n_m * 1.65 * 1.55, rel=1e-6)
    assert sp_double.k_rot_nm_rad > sp_single.k_rot_nm_rad * 2.0

    ev_single = evaluate_printer(spec_single)
    ev_double = evaluate_printer(spec_double_slm)
    assert ev_double.static_fea.combined_max_deflection_um < ev_single.static_fea.combined_max_deflection_um


# ---------------------------------------------------------------------------
# 7. Toolhead Belt-Plane CoG Alignment & X-Roll Warning
# ---------------------------------------------------------------------------


def test_toolhead_belt_plane_cog_warning_stealthburner_vs_monolith() -> None:
    """StealthBurner (dz=-28 mm) triggers X-roll warning; belt-coplanar Monolith Toolhead clears it."""
    spec_sb = _load_preset("corexy_voron_style.yaml")
    spec_sb.toolhead.cog_offset_mm = (0.0, -22.0, -28.0)
    spec_sb.toolhead.nozzle_offset_mm = (0.0, -26.0, -66.0)
    spec_sb.toolhead.belt_plane_z_offset_mm = 0.0
    ev_sb = evaluate_printer(spec_sb)

    assert ev_sb.thermal_structural.toolhead_belt_cog_z_offset_mm == pytest.approx(28.0, abs=1e-6)
    assert ev_sb.thermal_structural.toolhead_nozzle_y_lever_mm == pytest.approx(26.0, abs=1e-6)
    assert ev_sb.thermal_structural.toolhead_x_roll_warning is True

    spec_mono = _load_preset("monolith_zero_m0.yaml")
    ev_mono = evaluate_printer(spec_mono)
    assert ev_mono.thermal_structural.toolhead_belt_cog_z_offset_mm == pytest.approx(0.0, abs=1e-6)
    assert ev_mono.thermal_structural.toolhead_nozzle_y_lever_mm == pytest.approx(14.0, abs=1e-6)
    assert ev_mono.thermal_structural.toolhead_x_roll_warning is False


# ---------------------------------------------------------------------------
# 8. Rear Bottom Extrusion & Spatial IS Spread
# ---------------------------------------------------------------------------


def test_rear_bottom_extrusion_omission_increases_spatial_is_spread() -> None:
    """Omitting bottom rear extrusion removes base_rear member and widens spatial IS spread."""
    spec_with = _load_preset("monolith_sheet_metal_v2.yaml")
    spec_with.frame.rear_bottom_extrusion = True
    spec_without = spec_with.model_copy(deep=True)
    spec_without.frame.rear_bottom_extrusion = False

    asm_with = build_assembly(spec_with)
    asm_without = build_assembly(spec_without)
    assert any(m.id.startswith("base_rear") for m in asm_with.members)
    assert not any(m.id.startswith("base_rear") for m in asm_without.members)

    ev_with = evaluate_printer(spec_with)
    ev_without = evaluate_printer(spec_without)
    assert ev_with.workspace_frf is not None and ev_without.workspace_frf is not None
    assert ev_without.workspace_frf.is_spatial_spread_pct > ev_with.workspace_frf.is_spatial_spread_pct


# ---------------------------------------------------------------------------
# 9. Confidence Registry Completeness for New Monolith Hardware
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("category", "item_id"),
    [
        ("belts", "gt2_10mm"),
        ("belts", "gt3_10mm"),
        ("bed_mounts", "craz_kinematic_ge5c"),
        ("gantry_brackets", "slm_aluminum"),
        ("gantry_brackets", "sheet_metal_steel"),
        ("gantry_brackets", "pc_pbt_gf_printed"),
        ("hotends", "monolith_50mm_hf"),
        ("toolhead_presets", "monolith_toolhead_cpap"),
        ("profiles", "extrusion_1515"),
        ("profiles", "al_tube_12x12x1"),
        ("profiles", "mgn12_hollow_rail"),
        ("profiles", "unsupported_rail"),
    ],
)
def test_confidence_registry_has_monolith_entries(category: str, item_id: str) -> None:
    """All new Monolith hardware items return non-fallback confidence metadata."""
    meta = get_item_confidence(category, item_id)
    assert 50 <= meta["trust_pct"] <= 95
    assert "Lumped engineering parameter estimate" not in meta["basis"]
    assert "Unverified catalog option" not in meta["caveat"]
