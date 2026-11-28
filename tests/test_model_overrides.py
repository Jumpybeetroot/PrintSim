"""Comprehensive tests for lumped-model parameter inspection and explicit overrides."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from printsim.api.server import create_app
from printsim.engine import evaluate_printer
from printsim.fea.components import hertzian_rail_carriage_stiffness
from printsim.schema import (
    BED_MOUNT_CATALOG,
    BELT_CATALOG,
    BELT_ROUTING_CATALOG,
    CORNER_JOINT_BASE_2020,
    EXTRUSION_FILLINGS,
    FRAME_FEET_CATALOG,
    GANTRY_BRACKET_CATALOG,
    HOTEND_CATALOG,
    HOTEND_MOUNT_CLAMP_BASE_N_PER_MM,
    ITEM_CONFIDENCE_REGISTRY,
    RAIL_QUALITY_CATALOG,
    TOOLHEAD_CONSTRUCTION_CATALOG,
    TOOLHEAD_PRESETS,
    Z_DRIVE_MECHANISM_CATALOG,
    Z_PROBE_CATALOG,
    CustomBracketOverride,
    ModelOverridesSpec,
    PrinterSpec,
    build_effective_model_params,
    build_model_confidence_audit,
)

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"

ALL_14_GROUPS = (
    "corner_joint",
    "extrusion_filling",
    "frame_feet",
    "belt",
    "gantry_bracket",
    "rail_preload",
    "rail_quality",
    "belt_routing",
    "bearing_clearance",
    "z_probe",
    "toolhead_construction",
    "hotend_clamp",
    "bed_mount",
    "z_drive_mechanism",
)


def _load_preset(name: str = "corexy_voron_style.yaml") -> PrinterSpec:
    return PrinterSpec.from_yaml(EXAMPLES_DIR / name)


def test_default_empty_overrides_match_baseline_exactly() -> None:
    """Empty overrides & setting overrides equal to baseline yield identical simulation outputs."""
    for yaml_name in (
        "corexy_voron_style.yaml",
        "voron_trident_300.yaml",
        "cartesian_bedslinger.yaml",
        "delta_racer.yaml",
    ):
        spec_base = _load_preset(yaml_name)
        spec_null = PrinterSpec.model_validate({**spec_base.model_dump(mode="json"), "model_overrides": None})
        assert not spec_base.model_overrides.has_any_override()
        assert not spec_null.model_overrides.has_any_override()

        rep_base = evaluate_printer(spec_base, include_advanced_analysis=True)
        rep_null = evaluate_printer(spec_null, include_advanced_analysis=True)

        assert rep_base.modal_fea.fundamental_freq_hz == rep_null.modal_fea.fundamental_freq_hz
        assert rep_base.modal_fea.recommended_max_accel_mm_s2 == rep_null.modal_fea.recommended_max_accel_mm_s2
        assert rep_base.static_fea.combined_max_deflection_um == rep_null.static_fea.combined_max_deflection_um
        assert rep_base.kinematics.drive_stiffness_n_per_mm == rep_null.kinematics.drive_stiffness_n_per_mm
        assert rep_base.bed_plate_fea is not None and rep_null.bed_plate_fea is not None
        assert rep_base.bed_plate_fea.plate_first_mode_hz == rep_null.bed_plate_fea.plate_first_mode_hz
        assert rep_base.effective_model_params == rep_null.effective_model_params


def test_setting_overrides_equal_to_baseline_preserves_physics() -> None:
    """Explicitly setting overrides to the inspected baseline values preserves simulation outputs."""
    from printsim.schema import (
        resolve_bearing_clearance,
        resolve_bed_mount,
        resolve_belt,
        resolve_belt_routing,
        resolve_corner_joint,
        resolve_extrusion_filling,
        resolve_frame_feet,
        resolve_gantry_bracket,
        resolve_hotend_clamp,
        resolve_rail_preload,
        resolve_rail_quality,
        resolve_toolhead_construction,
        resolve_z_drive_mechanism,
        resolve_z_probe,
    )

    spec_base = _load_preset("voron_trident_300.yaml")
    spec_base.frame.feet_type = "rubber_feet"
    rep_base = evaluate_printer(spec_base, include_advanced_analysis=True)
    params = build_effective_model_params(spec_base)

    # 1. Cleanly rounded inspected baseline payload matches within display rounding tolerance (1e-4)
    override_payload = {
        group: {k: v for k, v in info["baseline"].items() if v is not None}
        for group, info in params.items()
    }
    spec_pinned = spec_base.model_copy(
        update={"model_overrides": ModelOverridesSpec.model_validate(override_payload)}
    )
    rep_pinned = evaluate_printer(spec_pinned, include_advanced_analysis=True)

    assert rep_pinned.modal_fea.fundamental_freq_hz == pytest.approx(
        rep_base.modal_fea.fundamental_freq_hz, rel=1e-4
    )
    assert rep_pinned.modal_fea.recommended_max_accel_mm_s2 == pytest.approx(
        rep_base.modal_fea.recommended_max_accel_mm_s2, rel=1e-4
    )
    assert rep_pinned.static_fea.combined_max_deflection_um == pytest.approx(
        rep_base.static_fea.combined_max_deflection_um, rel=1e-4
    )
    assert rep_pinned.kinematics.drive_stiffness_n_per_mm == pytest.approx(
        rep_base.kinematics.drive_stiffness_n_per_mm, rel=1e-4
    )
    assert rep_pinned.bed_plate_fea is not None and rep_base.bed_plate_fea is not None
    assert rep_pinned.bed_plate_fea.plate_first_mode_hz == pytest.approx(
        rep_base.bed_plate_fea.plate_first_mode_hz, rel=1e-4
    )

    # 2. Exact unrounded resolver baselines match baseline outputs to floating-point precision
    cj = resolve_corner_joint(spec_base)
    ef = resolve_extrusion_filling(spec_base)
    ff = resolve_frame_feet(spec_base)
    bl = resolve_belt(spec_base)
    gb = resolve_gantry_bracket(spec_base)
    rp = resolve_rail_preload(spec_base)
    rq = resolve_rail_quality(spec_base)
    br = resolve_belt_routing(spec_base)
    bc = resolve_bearing_clearance(spec_base)
    zp = resolve_z_probe(spec_base)
    tc = resolve_toolhead_construction(spec_base)
    hc = resolve_hotend_clamp(spec_base)
    bm = resolve_bed_mount(spec_base)
    zd = resolve_z_drive_mechanism(spec_base)

    exact_overrides = ModelOverridesSpec.model_validate(
        {
            "corner_joint": {
                "k_t_n_per_mm": cj["baseline_k_t_n_m"] * 1e-3,
                "k_r_inplane_nm_rad": cj["baseline_k_r_inplane_nm_rad"],
                "k_r_torsion_nm_rad": cj["baseline_k_r_torsion_nm_rad"],
                "damping_delta": cj["baseline_damping_delta"],
            },
            "extrusion_filling": {
                "density_kg_m3": ef["baseline_density_kg_m3"],
                "youngs_modulus_gpa": ef["baseline_youngs_modulus_pa"] * 1e-9,
                "bonded": ef["baseline_bonded"],
                "damping_boost": ef["baseline_damping_boost"],
            },
            "frame_feet": {
                "k_axial_n_per_mm": ff["baseline_k_axial_n_per_mm"],
                "k_lateral_n_per_mm": ff["baseline_k_lateral_n_per_mm"],
                "damping_delta": ff["baseline_damping_delta"],
                "base_mass_kg": ff["baseline_base_mass_kg"],
            },
            "belt": {
                "stiffness_ea_n": bl["baseline_stiffness_ea_n"],
                "tooth_shear_k_n_per_m_mm": bl["baseline_tooth_shear_k_n_per_m_mm"],
                "linear_mass_g_per_m": bl["baseline_linear_mass_kg_m"] * 1e3,
                "max_working_tension_n": bl["baseline_max_working_tension_n"],
            },
            "gantry_bracket": {
                "stiffness_n_per_mm": gb["baseline_stiffness_n_m"] * 1e-3,
                "carriage_mass_delta_g": gb["baseline_carriage_mass_delta_g"],
                "creep_tension_limit_n": gb["baseline_creep_tension_limit_n"],
                "damping_delta": gb["baseline_damping_delta"],
            },
            "rail_preload": {"preload_fraction": rp["baseline_preload_fraction"]},
            "rail_quality": {
                "stiffness_mult": rq["baseline_stiffness_mult"],
                "drag_mult": rq["baseline_drag_mult"],
            },
            "belt_routing": {
                "span_factor": br["baseline_span_factor"],
                "idler_stiffness_factor": br["baseline_idler_stiffness_factor"],
            },
            "bearing_clearance": {"radial_play_um": bc["baseline_radial_play_um"]},
            "z_probe": {
                "compliance_factor": zp["baseline_compliance_factor"],
                "cog_y_shift_mm": zp["baseline_cog_y_shift_mm"],
            },
            "toolhead_construction": {
                "body_stiffness_base_n_per_mm": tc["baseline_body_stiffness_base_n_per_mm"],
                "torsion_mult": tc["baseline_torsion_mult"],
            },
            "hotend_clamp": {"clamp_stiffness_base_n_per_mm": hc["baseline_clamp_stiffness_base_n_per_mm"]},
            "bed_mount": {
                "k_axial_n_per_mm": bm["baseline_k_axial_n_per_mm"],
                "k_lateral_n_per_mm": bm["baseline_k_lateral_n_per_mm"],
                "k_restraint_thermal_n_per_mm": bm["baseline_k_restraint_thermal_n_per_mm"],
                "damping_delta": bm["baseline_damping_delta"],
                "standoff_height_mm": bm["baseline_standoff_height_mm"],
            },
            "z_drive_mechanism": {
                "k_nut_n_per_um": zd["baseline_k_nut_n_per_um"],
                "k_thrust_n_per_um": zd["baseline_k_thrust_n_per_um"],
                "backlash_um": zd["baseline_backlash_um"],
            },
        }
    )
    rep_exact = evaluate_printer(
        spec_base.model_copy(update={"model_overrides": exact_overrides}),
        include_advanced_analysis=True,
    )
    assert rep_exact.modal_fea.fundamental_freq_hz == pytest.approx(
        rep_base.modal_fea.fundamental_freq_hz, rel=1e-12
    )
    assert rep_exact.kinematics.drive_stiffness_n_per_mm == pytest.approx(
        rep_base.kinematics.drive_stiffness_n_per_mm, rel=1e-12
    )


def test_yaml_omits_empty_overrides_and_roundtrips_nonempty() -> None:
    """Empty model_overrides must be omitted from YAML; non-empty overrides (including 0/False) round-trip."""
    spec = _load_preset("corexy_voron_style.yaml")
    yaml_clean = spec.to_yaml()
    parsed_clean = yaml.safe_load(yaml_clean)
    assert "model_overrides" not in parsed_clean

    # Partial JSON null sub-group validation
    partial = PrinterSpec.model_validate(
        {
            **spec.model_dump(mode="json"),
            "model_overrides": {
                "z_probe": None,
                "bearing_clearance": {"radial_play_um": 0.0},
                "extrusion_filling": {"bonded": False, "damping_boost": 0.025},
                "belt": {"stiffness_ea_n": 95000.0},
            },
        }
    )
    assert partial.model_overrides.has_any_override()
    assert partial.model_overrides.z_probe.compliance_factor is None
    assert partial.model_overrides.bearing_clearance.radial_play_um == 0.0
    assert partial.model_overrides.extrusion_filling.bonded is False

    yaml_ov = partial.to_yaml()
    parsed_ov = yaml.safe_load(yaml_ov)
    assert "model_overrides" in parsed_ov
    assert "z_probe" not in parsed_ov["model_overrides"]
    assert parsed_ov["model_overrides"]["bearing_clearance"] == {"radial_play_um": 0.0}
    assert parsed_ov["model_overrides"]["extrusion_filling"] == {
        "bonded": False,
        "damping_boost": 0.025,
    }
    assert parsed_ov["model_overrides"]["belt"] == {"stiffness_ea_n": 95000.0}

    roundtripped = PrinterSpec.model_validate(parsed_ov)
    assert roundtripped.model_overrides == partial.model_overrides


def test_corner_joint_override_changes_unpaneled_frame_and_upright_baseline() -> None:
    """Corner joint overrides change unpaneled frame stiffness, damping, and rigid-joint baseline."""
    doom_spec = _load_preset("doomcube_4040.yaml")
    doom_params = build_effective_model_params(doom_spec)
    # upright_profile (4040) != profile (2020) uses geometric mean > 2020 baseline (22000 Nm/rad)
    assert doom_params["corner_joint"]["baseline"]["k_r_inplane_nm_rad"] > 22000.0

    spec_soft = _load_preset("corexy_voron_style.yaml")
    spec_soft.frame.enclosed_panels = False
    spec_soft.frame.bracing = "none"
    spec_soft.frame.corner_joint_type = "rigid"
    spec_soft.model_overrides.corner_joint.k_r_inplane_nm_rad = 1500.0
    spec_soft.model_overrides.corner_joint.damping_delta = 0.015

    spec_stiff = spec_soft.model_copy(deep=True)
    spec_stiff.model_overrides.corner_joint.k_r_inplane_nm_rad = 250000.0
    spec_stiff.model_overrides.corner_joint.damping_delta = 0.001

    rep_soft = evaluate_printer(spec_soft, include_advanced_analysis=False)
    rep_stiff = evaluate_printer(spec_stiff, include_advanced_analysis=False)

    assert rep_stiff.modal_fea.fundamental_freq_hz > rep_soft.modal_fea.fundamental_freq_hz
    assert rep_soft.modal_fea.frame_damping_ratio > rep_stiff.modal_fea.frame_damping_ratio


def test_extrusion_filling_override_changes_mass_stiffness_and_damping() -> None:
    """Extrusion filling overrides work even when catalog selection is 'none'."""
    spec_base = _load_preset("corexy_voron_style.yaml")
    spec_base.frame.extrusion_filling = "none"
    rep_base = evaluate_printer(spec_base, include_advanced_analysis=False)

    spec_heavy = spec_base.model_copy(deep=True)
    spec_heavy.model_overrides.extrusion_filling.density_kg_m3 = 5000.0
    spec_heavy.model_overrides.extrusion_filling.damping_boost = 0.04
    rep_heavy = evaluate_printer(spec_heavy, include_advanced_analysis=False)

    assert rep_heavy.assembly.extrusion_fill_mass_kg > 1.0
    assert rep_heavy.assembly.total_mass_kg > rep_base.assembly.total_mass_kg
    assert rep_heavy.modal_fea.frame_damping_ratio > rep_base.modal_fea.frame_damping_ratio

    spec_bonded = spec_base.model_copy(deep=True)
    spec_bonded.frame.enclosed_panels = False
    spec_bonded.model_overrides.extrusion_filling.density_kg_m3 = 100.0
    spec_bonded.model_overrides.extrusion_filling.youngs_modulus_gpa = 120.0
    spec_bonded.model_overrides.extrusion_filling.bonded = True
    rep_unbonded_base = evaluate_printer(
        spec_base.model_copy(update={"frame": spec_base.frame.model_copy(update={"enclosed_panels": False})}),
        include_advanced_analysis=False,
    )
    rep_bonded = evaluate_printer(spec_bonded, include_advanced_analysis=False)
    assert rep_bonded.modal_fea.fundamental_freq_hz > rep_unbonded_base.modal_fea.fundamental_freq_hz


def test_frame_feet_override_changes_rocking_and_damping() -> None:
    """Frame feet overrides affect rocking frequency & damping, including from rigid_bolted."""
    spec_soft = _load_preset("corexy_voron_style.yaml")
    spec_soft.frame.feet_type = "rigid_bolted"
    spec_soft.model_overrides.frame_feet.k_lateral_n_per_mm = 15.0
    spec_soft.model_overrides.frame_feet.k_axial_n_per_mm = 40.0
    spec_soft.model_overrides.frame_feet.damping_delta = 0.025
    spec_soft.model_overrides.frame_feet.base_mass_kg = 12.0

    spec_stiff = spec_soft.model_copy(deep=True)
    spec_stiff.model_overrides.frame_feet.k_lateral_n_per_mm = 2000.0
    spec_stiff.model_overrides.frame_feet.k_axial_n_per_mm = 5000.0
    spec_stiff.model_overrides.frame_feet.damping_delta = 0.002

    rep_soft = evaluate_printer(spec_soft, include_advanced_analysis=False)
    rep_stiff = evaluate_printer(spec_stiff, include_advanced_analysis=False)

    assert rep_stiff.modal_fea.fundamental_freq_hz > rep_soft.modal_fea.fundamental_freq_hz
    assert rep_soft.modal_fea.frame_damping_ratio > rep_stiff.modal_fea.frame_damping_ratio
    assert rep_soft.assembly.total_mass_kg > evaluate_printer(_load_preset("corexy_voron_style.yaml"), include_advanced_analysis=False).assembly.total_mass_kg


def test_belt_and_routing_and_bearing_clearance_overrides() -> None:
    """Belt, belt_routing, and bearing_clearance overrides propagate into stiffness, pluck, and play."""
    spec_lo = _load_preset("corexy_voron_style.yaml")
    spec_lo.model_overrides.belt.stiffness_ea_n = 20000.0
    spec_lo.model_overrides.belt.tooth_shear_k_n_per_m_mm = 20000.0
    spec_lo.model_overrides.belt.linear_mass_g_per_m = 25.0
    spec_lo.model_overrides.belt.max_working_tension_n = 25.0
    spec_lo.model_overrides.belt_routing.span_factor = 1.4
    spec_lo.model_overrides.belt_routing.idler_stiffness_factor = 0.5
    spec_lo.model_overrides.bearing_clearance.radial_play_um = 24.5
    spec_lo.gantry.belt.pretension_n = 28.0  # above the 25 N override limit, whatever the preset ships

    spec_hi = _load_preset("corexy_voron_style.yaml")
    spec_hi.gantry.belt.pretension_n = 28.0
    spec_hi.model_overrides.belt.stiffness_ea_n = 180000.0
    spec_hi.model_overrides.belt.tooth_shear_k_n_per_m_mm = 150000.0
    spec_hi.model_overrides.belt.linear_mass_g_per_m = 6.0
    spec_hi.model_overrides.belt.max_working_tension_n = 400.0
    spec_hi.model_overrides.belt_routing.span_factor = 0.65
    spec_hi.model_overrides.belt_routing.idler_stiffness_factor = 2.5
    spec_hi.model_overrides.bearing_clearance.radial_play_um = 1.5

    rep_lo = evaluate_printer(spec_lo, include_advanced_analysis=False)
    rep_hi = evaluate_printer(spec_hi, include_advanced_analysis=False)

    assert rep_hi.kinematics.belt_stiffness_n_per_mm > rep_lo.kinematics.belt_stiffness_n_per_mm
    assert rep_hi.kinematics.tooth_mesh_stiffness_n_per_mm > rep_lo.kinematics.tooth_mesh_stiffness_n_per_mm
    assert rep_hi.kinematics.idler_shaft_stiffness_n_per_mm > rep_lo.kinematics.idler_shaft_stiffness_n_per_mm
    assert rep_hi.modal_fea.shaper_x is not None and rep_lo.modal_fea.shaper_x is not None
    assert rep_hi.modal_fea.shaper_x.belt_freq_hz > rep_lo.modal_fea.shaper_x.belt_freq_hz
    assert rep_hi.kinematics.belt_pluck_freq_150mm_hz > rep_lo.kinematics.belt_pluck_freq_150mm_hz
    assert rep_lo.kinematics.belt_max_working_tension_n == 25.0
    assert rep_lo.kinematics.belt_overtension_risk is True
    assert rep_hi.kinematics.belt_overtension_risk is False
    assert rep_lo.kinematics.bearing_play_um == 24.5
    assert rep_hi.kinematics.bearing_play_um == 1.5


def test_gantry_bracket_overrides() -> None:
    """Gantry bracket overrides affect bracket stiffness, moving mass, creep limit, and damping."""
    spec_a = _load_preset("corexy_voron_style.yaml")
    spec_a.gantry.belt.pretension_n = 60.0
    spec_a.model_overrides.gantry_bracket.stiffness_n_per_mm = 250.0
    spec_a.model_overrides.gantry_bracket.carriage_mass_delta_g = -40.0
    spec_a.model_overrides.gantry_bracket.creep_tension_limit_n = 40.0
    spec_a.model_overrides.gantry_bracket.damping_delta = 0.012

    spec_b = _load_preset("corexy_voron_style.yaml")
    spec_b.gantry.belt.pretension_n = 60.0
    spec_b.model_overrides.gantry_bracket.stiffness_n_per_mm = 6000.0
    spec_b.model_overrides.gantry_bracket.carriage_mass_delta_g = 90.0
    spec_b.model_overrides.gantry_bracket.creep_tension_limit_n = 250.0
    spec_b.model_overrides.gantry_bracket.damping_delta = -0.005

    rep_a = evaluate_printer(spec_a, include_advanced_analysis=False)
    rep_b = evaluate_printer(spec_b, include_advanced_analysis=False)

    assert rep_a.kinematics.bracket_stiffness_n_per_mm == pytest.approx(250.0, rel=1e-6)
    assert rep_b.kinematics.bracket_stiffness_n_per_mm == pytest.approx(6000.0, rel=1e-6)
    assert rep_b.kinematics.drive_stiffness_n_per_mm > rep_a.kinematics.drive_stiffness_n_per_mm
    assert rep_b.assembly.y_moving_mass_kg > rep_a.assembly.y_moving_mass_kg + 0.1
    assert rep_a.kinematics.bracket_creep_risk is True
    assert rep_b.kinematics.bracket_creep_risk is False
    assert rep_a.modal_fea.frame_damping_ratio > rep_b.modal_fea.frame_damping_ratio


def test_rail_preload_and_rail_quality_overrides() -> None:
    """Rail preload and rail quality overrides scale carriage stiffness and drag monotonically."""
    spec_lo = _load_preset("corexy_voron_style.yaml")
    spec_lo.model_overrides.rail_preload.preload_fraction = 0.002
    spec_lo.model_overrides.rail_quality.stiffness_mult = 0.4
    spec_lo.model_overrides.rail_quality.drag_mult = 2.0

    spec_hi = _load_preset("corexy_voron_style.yaml")
    spec_hi.model_overrides.rail_preload.preload_fraction = 0.15
    spec_hi.model_overrides.rail_quality.stiffness_mult = 2.0
    spec_hi.model_overrides.rail_quality.drag_mult = 0.5

    rep_lo = evaluate_printer(spec_lo, include_advanced_analysis=False)
    rep_hi = evaluate_printer(spec_hi, include_advanced_analysis=False)

    assert (
        rep_hi.thermal_structural.rail_carriage_radial_n_per_um
        > rep_lo.thermal_structural.rail_carriage_radial_n_per_um
    )
    assert (
        rep_hi.thermal_structural.rail_carriage_roll_nm_per_rad
        > rep_lo.thermal_structural.rail_carriage_roll_nm_per_rad
    )
    assert (
        rep_hi.assembly.toolhead_stage_stiffnesses_n_per_mm["rail_carriage_n_per_mm"]
        > rep_lo.assembly.toolhead_stage_stiffnesses_n_per_mm["rail_carriage_n_per_mm"]
    )
    car_lo = hertzian_rail_carriage_stiffness(
        "mgn12_rail", preload_fraction_override=0.002, stiffness_mult_override=0.4, drag_mult_override=2.0
    )
    car_hi = hertzian_rail_carriage_stiffness(
        "mgn12_rail", preload_fraction_override=0.15, stiffness_mult_override=2.0, drag_mult_override=0.5
    )
    assert car_lo["seal_drag_n"] > car_hi["seal_drag_n"]


def test_toolhead_zprobe_and_hotend_clamp_overrides() -> None:
    """z_probe, toolhead_construction (both modes), and hotend_clamp overrides scale toolhead stages."""
    spec_base = _load_preset("corexy_voron_style.yaml")

    # 1. z_probe.compliance_factor & cog_y_shift_mm
    spec_probe_soft = spec_base.model_copy(deep=True)
    spec_probe_soft.model_overrides.z_probe.compliance_factor = 0.25
    spec_probe_soft.model_overrides.z_probe.cog_y_shift_mm = -25.0
    rep_base = evaluate_printer(spec_base, include_advanced_analysis=False)
    rep_probe = evaluate_printer(spec_probe_soft, include_advanced_analysis=False)
    assert (
        rep_probe.assembly.toolhead_stage_stiffnesses_n_per_mm["combined_toolhead_n_per_mm"]
        < rep_base.assembly.toolhead_stage_stiffnesses_n_per_mm["combined_toolhead_n_per_mm"]
    )
    assert rep_probe.static_fea.accel_deflection_y_um > rep_base.static_fea.accel_deflection_y_um

    # 2. toolhead_construction with body_construction set and with body_construction=None (lumped mode)
    for bc in (spec_base.toolhead.body_construction, None):
        spec_tc_lo = spec_base.model_copy(deep=True)
        spec_tc_lo.use_cad_superelements = False
        spec_tc_lo.toolhead.body_construction = bc
        spec_tc_lo.model_overrides.toolhead_construction.body_stiffness_base_n_per_mm = 60.0
        spec_tc_lo.model_overrides.toolhead_construction.torsion_mult = 0.4

        spec_tc_hi = spec_tc_lo.model_copy(deep=True)
        spec_tc_hi.model_overrides.toolhead_construction.body_stiffness_base_n_per_mm = 1200.0
        spec_tc_hi.model_overrides.toolhead_construction.torsion_mult = 3.0

        rep_tc_lo = evaluate_printer(spec_tc_lo, include_advanced_analysis=False)
        rep_tc_hi = evaluate_printer(spec_tc_hi, include_advanced_analysis=False)
        assert (
            rep_tc_hi.assembly.toolhead_stage_stiffnesses_n_per_mm["body_n_per_mm"]
            > rep_tc_lo.assembly.toolhead_stage_stiffnesses_n_per_mm["body_n_per_mm"]
        )
        assert (
            rep_tc_hi.static_fea.toolhead_stiffness_x_n_per_mm
            > rep_tc_lo.static_fea.toolhead_stiffness_x_n_per_mm
        )

    # 3. hotend_clamp.clamp_stiffness_base_n_per_mm
    spec_hc_lo = spec_base.model_copy(deep=True)
    spec_hc_lo.model_overrides.hotend_clamp.clamp_stiffness_base_n_per_mm = 300.0
    spec_hc_hi = spec_base.model_copy(deep=True)
    spec_hc_hi.model_overrides.hotend_clamp.clamp_stiffness_base_n_per_mm = 12000.0

    rep_hc_lo = evaluate_printer(spec_hc_lo, include_advanced_analysis=False)
    rep_hc_hi = evaluate_printer(spec_hc_hi, include_advanced_analysis=False)
    assert (
        rep_hc_hi.assembly.toolhead_stage_stiffnesses_n_per_mm["hotend_clamp_n_per_mm"]
        > rep_hc_lo.assembly.toolhead_stage_stiffnesses_n_per_mm["hotend_clamp_n_per_mm"]
    )


def test_bed_mount_and_z_drive_mechanism_overrides() -> None:
    """bed_mount and z_drive_mechanism overrides change bed FEM Hz, taco buckling, and Z stiffness."""
    # Compliant bed mount -> stiffer k_axial increases bed plate modal frequency
    spec_bed_lo = _load_preset("voron_trident_300.yaml")
    spec_bed_lo.bed.mount_type = "springs_4pt"
    spec_bed_lo.bed.target_temp_c = 110.0
    spec_bed_lo.model_overrides.bed_mount.k_axial_n_per_mm = 8.0
    spec_bed_lo.model_overrides.bed_mount.k_lateral_n_per_mm = 4.0
    spec_bed_lo.model_overrides.bed_mount.k_restraint_thermal_n_per_mm = 0.0
    spec_bed_lo.model_overrides.bed_mount.damping_delta = 0.02

    spec_bed_hi = spec_bed_lo.model_copy(deep=True)
    spec_bed_hi.model_overrides.bed_mount.k_axial_n_per_mm = 40000.0
    spec_bed_hi.model_overrides.bed_mount.k_lateral_n_per_mm = 20000.0
    spec_bed_hi.model_overrides.bed_mount.k_restraint_thermal_n_per_mm = 60000.0
    spec_bed_hi.model_overrides.bed_mount.damping_delta = 0.0
    spec_bed_hi.model_overrides.bed_mount.standoff_height_mm = 30.0

    rep_bed_lo = evaluate_printer(spec_bed_lo, include_advanced_analysis=True)
    rep_bed_hi = evaluate_printer(spec_bed_hi, include_advanced_analysis=True)

    assert rep_bed_lo.bed_plate_fea is not None and rep_bed_hi.bed_plate_fea is not None
    assert rep_bed_hi.bed_plate_fea.plate_first_mode_hz > rep_bed_lo.bed_plate_fea.plate_first_mode_hz
    assert rep_bed_lo.bed_plate_fea.taco_buckling_um == 0.0
    assert rep_bed_hi.bed_plate_fea.taco_buckling_um > 30.0
    assert rep_bed_hi.thermal_structural.bed_buckling_taco_um == pytest.approx(
        rep_bed_hi.bed_plate_fea.taco_buckling_um, abs=0.05
    )
    assert rep_bed_lo.modal_fea.frame_damping_ratio > rep_bed_hi.modal_fea.frame_damping_ratio
    assert (
        rep_bed_hi.assembly.node_map["bed_frame"].position_m[2]
        < rep_bed_lo.assembly.node_map["bed_frame"].position_m[2]
    )

    # Leadscrew z_drive_mechanism k_nut_n_per_um & k_thrust_n_per_um
    spec_z_lo = _load_preset("voron_trident_300.yaml")
    spec_z_lo.model_overrides.z_drive_mechanism.k_nut_n_per_um = 1.0
    spec_z_lo.model_overrides.z_drive_mechanism.k_thrust_n_per_um = 1.0
    spec_z_hi = _load_preset("voron_trident_300.yaml")
    spec_z_hi.model_overrides.z_drive_mechanism.k_nut_n_per_um = 200.0
    spec_z_hi.model_overrides.z_drive_mechanism.k_thrust_n_per_um = 250.0

    rep_z_lo = evaluate_printer(spec_z_lo, include_advanced_analysis=True)
    rep_z_hi = evaluate_printer(spec_z_hi, include_advanced_analysis=True)
    assert rep_z_lo.bed_plate_fea is not None and rep_z_hi.bed_plate_fea is not None
    assert rep_z_hi.bed_plate_fea.mount_settlement_um < rep_z_lo.bed_plate_fea.mount_settlement_um
    assert rep_z_hi.bed_plate_fea.plate_first_mode_hz > rep_z_lo.bed_plate_fea.plate_first_mode_hz

    # Belted quad-Z k_drive_n_per_um_per_corner affects flying-gantry z_joint springs
    spec_v24_lo = _load_preset("corexy_voron_style.yaml")
    spec_v24_lo.model_overrides.z_drive_mechanism.k_drive_n_per_um_per_corner = 0.05
    spec_v24_hi = _load_preset("corexy_voron_style.yaml")
    spec_v24_hi.model_overrides.z_drive_mechanism.k_drive_n_per_um_per_corner = 5.0
    rep_v24_lo = evaluate_printer(spec_v24_lo, include_advanced_analysis=False)
    rep_v24_hi = evaluate_printer(spec_v24_hi, include_advanced_analysis=False)
    z_joint_lo = next(s for s in rep_v24_lo.assembly.fastener_springs if s.id == "z_joint_fl")
    z_joint_hi = next(s for s in rep_v24_hi.assembly.fastener_springs if s.id == "z_joint_fl")
    assert z_joint_hi.k_normal_n_m > z_joint_lo.k_normal_n_m


def test_api_simulate_and_catalog_metadata() -> None:
    """/api/simulate exposes effective_model_params and /api/catalog returns enriched metadata."""
    client = TestClient(create_app())

    r_cat = client.get("/api/catalog")
    assert r_cat.status_code == 200
    cat = r_cat.json()

    assert len(cat["corner_joints"]) == 5
    for cj in cat["corner_joints"]:
        assert {"k_t_n_per_mm", "k_r_inplane_nm_rad", "k_r_torsion_nm_rad", "damping_delta"} <= set(cj)

    assert len(cat["rail_preloads"]) == 5
    for rp in cat["rail_preloads"]:
        assert "preload_fraction" in rp and rp["preload_fraction"] > 0.0

    assert len(cat["bearing_clearances"]) == 3
    for bc in cat["bearing_clearances"]:
        assert "radial_play_um" in bc and bc["radial_play_um"] > 0.0

    assert "hotend_mount_styles" in cat and len(cat["hotend_mount_styles"]) == 4
    for hms in cat["hotend_mount_styles"]:
        assert "clamp_stiffness_base_n_per_mm" in hms and hms["clamp_stiffness_base_n_per_mm"] > 0.0

    spec = _load_preset("corexy_voron_style.yaml")
    spec_dict = spec.model_dump(mode="json")
    spec_dict["model_overrides"] = {
        "belt": {"stiffness_ea_n": 72000.0},
        "z_probe": None,
        "bearing_clearance": {"radial_play_um": 4.2},
    }
    r_sim = client.post("/api/simulate", json={"spec": spec_dict})
    assert r_sim.status_code == 200
    sim = r_sim.json()

    eff = sim["effective_model_params"]
    assert set(eff.keys()) == set(ALL_14_GROUPS)
    for group_key, entry in eff.items():
        assert set(entry.keys()) == {"baseline", "effective", "overridden_fields"}
        assert set(entry["baseline"].keys()) == set(entry["effective"].keys())
        sub_cls = type(getattr(spec.model_overrides, group_key))
        assert set(entry["baseline"].keys()) == set(sub_cls.model_fields.keys())

    assert eff["belt"]["overridden_fields"] == ["stiffness_ea_n"]
    assert eff["belt"]["effective"]["stiffness_ea_n"] == 72000.0
    assert eff["bearing_clearance"]["overridden_fields"] == ["radial_play_um"]
    assert eff["bearing_clearance"]["effective"]["radial_play_um"] == 4.2
    assert eff["z_probe"]["overridden_fields"] == []

    mc = sim["model_confidence"]
    assert mc == build_model_confidence_audit(PrinterSpec.model_validate(spec_dict))
    by_id = {entry["id"]: entry for entry in mc["entries"]}
    assert by_id["belt"]["overridden"] is True
    assert "72,000 N" in by_id["belt"]["active_values"]
    assert by_id["bearing_clearance"]["overridden"] is True
    assert "4.2 µm" in by_id["bearing_clearance"]["active_values"]
    assert by_id["z_probe"]["overridden"] is False


CATALOG_BACKED_ENTRY_IDS = (
    "corner_joint",
    "extrusion_filling",
    "frame_feet",
    "rail_quality",
    "belt",
    "belt_routing",
    "gantry_bracket",
    "toolhead_preset",
    "toolhead_construction",
    "z_probe",
    "hotend_clamp",
    "bed_mount",
    "z_drive_mechanism",
)


def test_model_confidence_audit_contract_across_all_presets() -> None:
    """build_model_confidence_audit returns valid 23-entry Unicode audit on every preset."""
    expected_subsystems = {
        "frame": "Frame",
        "gantry": "Gantry",
        "toolhead": "Toolhead",
        "hotend": "Hotend",
        "bed": "Bed / Z",
        "solver": "Solver & Shaper",
    }
    expected_entry_keys = {
        "id",
        "subsystem",
        "subsystem_label",
        "name",
        "active_selection",
        "active_values",
        "trust_pct",
        "tier",
        "override_group",
        "overridden",
        "basis",
        "caveat",
        "options_breakdown",
        "cad_superelement_active",
        "cad_superelement_id",
        "cad_trust_pct",
        "effective_trust_pct",
    }
    expected_breakdown_keys = {
        "id",
        "label",
        "trust_pct",
        "tier",
        "active",
        "summary_values",
        "basis",
        "caveat",
    }
    raw_snake_re = re.compile(r"\b[a-z][a-z0-9]*_[a-z0-9_]+\b")

    yamls = sorted(EXAMPLES_DIR.glob("*.yaml"))
    assert len(yamls) >= 8
    for ypath in yamls:
        spec = PrinterSpec.from_yaml(ypath)
        audit = build_model_confidence_audit(spec)

        assert audit["disclaimer_title"] == "Work in Progress — Physics Model Confidence & Assumptions"
        assert "ADXL345" in audit["disclaimer_body"]
        assert isinstance(audit["overall_trust_pct"], int)
        assert 0 <= audit["overall_trust_pct"] <= 100

        entries = audit["entries"]
        assert len(entries) == 23
        assert audit["overall_trust_pct"] == round(sum(e["trust_pct"] for e in entries) / 23)

        seen_ids: set[str] = set()
        seen_override_groups: set[str] = set()
        for entry in entries:
            assert set(entry.keys()) == expected_entry_keys
            assert entry["id"] not in seen_ids
            seen_ids.add(entry["id"])

            sub = entry["subsystem"]
            assert sub in expected_subsystems
            assert entry["subsystem_label"] == expected_subsystems[sub]

            pct = entry["trust_pct"]
            assert isinstance(pct, int) and 0 <= pct <= 100
            expected_tier = (
                "exact"
                if pct >= 90
                else ("calibrated" if pct >= 70 else ("estimate" if pct >= 50 else "heuristic"))
            )
            assert entry["tier"] == expected_tier
            assert isinstance(entry["overridden"], bool)
            assert entry["overridden"] is False
            assert entry["basis"] and entry["caveat"]
            assert entry["active_selection"]
            assert not raw_snake_re.search(entry["active_selection"]), entry["active_selection"]
            assert entry["active_values"]

            bd = entry["options_breakdown"]
            assert isinstance(bd, list)
            if entry["id"] in CATALOG_BACKED_ENTRY_IDS:
                assert len(bd) >= 2
                for item in bd:
                    assert set(item.keys()) == expected_breakdown_keys
                    assert isinstance(item["trust_pct"], int) and 0 <= item["trust_pct"] <= 100
                    item_tier = (
                        "exact"
                        if item["trust_pct"] >= 90
                        else (
                            "calibrated"
                            if item["trust_pct"] >= 70
                            else ("estimate" if item["trust_pct"] >= 50 else "heuristic")
                        )
                    )
                    assert item["tier"] == item_tier
                    assert isinstance(item["active"], bool)
                    assert item["label"]
                    assert not raw_snake_re.search(item["label"]), item["label"]
                    assert item["summary_values"] and item["basis"] and item["caveat"]
            else:
                assert bd == []

            if entry["override_group"] is not None:
                assert entry["override_group"] in ALL_14_GROUPS
                seen_override_groups.add(entry["override_group"])

        assert seen_override_groups == set(ALL_14_GROUPS)

        # Verify Unicode units across the full audit payload
        joined_values = " | ".join(e["active_values"] for e in entries)
        for token in ("mm²", "cm⁴", "GPa", "N/mm", "N·m/rad", "µm", "×", "kg/m³", "°C", "W"):
            assert token in joined_values, f"Missing Unicode unit '{token}' in {ypath.name}"


def test_item_confidence_registry_completeness_all_14_catalogs() -> None:
    """Every single ID across all 14 PrintSim catalogs (95 options) has valid registry metadata."""
    catalog_map = {
        "toolhead_presets": TOOLHEAD_PRESETS,
        "toolhead_constructions": TOOLHEAD_CONSTRUCTION_CATALOG,
        "z_probes": Z_PROBE_CATALOG,
        "hotends": HOTEND_CATALOG,
        "hotend_mount_styles": HOTEND_MOUNT_CLAMP_BASE_N_PER_MM,
        "belts": BELT_CATALOG,
        "belt_routings": BELT_ROUTING_CATALOG,
        "gantry_brackets": GANTRY_BRACKET_CATALOG,
        "rail_qualities": RAIL_QUALITY_CATALOG,
        "corner_joints": CORNER_JOINT_BASE_2020,
        "extrusion_fillings": EXTRUSION_FILLINGS,
        "frame_feet": FRAME_FEET_CATALOG,
        "bed_mounts": BED_MOUNT_CATALOG,
        "z_drive_mechanisms": Z_DRIVE_MECHANISM_CATALOG,
    }
    assert set(ITEM_CONFIDENCE_REGISTRY.keys()) == set(catalog_map.keys())
    total_options = 0
    for cat_key, catalog in catalog_map.items():
        reg = ITEM_CONFIDENCE_REGISTRY[cat_key]
        assert set(reg.keys()) == set(catalog.keys()), f"Mismatch in {cat_key}"
        for item_id, meta in reg.items():
            assert set(meta.keys()) == {"trust_pct", "basis", "caveat"}, f"{cat_key}.{item_id}"
            assert isinstance(meta["trust_pct"], int) and 0 <= meta["trust_pct"] <= 100
            assert isinstance(meta["basis"], str) and len(meta["basis"].strip()) > 10
            assert isinstance(meta["caveat"], str) and len(meta["caveat"].strip()) > 10
            total_options += 1
    assert total_options == 120


def test_per_item_confidence_differentiation_and_options_breakdown() -> None:
    """Per-item trust scores differentiate within catalogs and match active options_breakdown items."""

    def _by_id(s: PrinterSpec) -> dict[str, dict]:
        return {e["id"]: e for e in build_model_confidence_audit(s)["entries"]}

    # Every catalog-backed entry on catalog-complete specs (heuristic mode) has exactly 1 active item matching trust_pct
    catalog_complete_specs = [
        _load_preset("doomcube_4040.yaml").model_copy(update={"use_cad_superelements": False}),
        _load_preset("voron_monolith_awd.yaml").model_copy(update={"use_cad_superelements": False}),
        _load_preset("vzbot_330_awd.yaml").model_copy(update={"use_cad_superelements": False}),
        _load_preset("corexy_voron_style.yaml").model_copy(
            update={
                "use_cad_superelements": False,
                "toolhead": _load_preset("corexy_voron_style.yaml").toolhead.model_copy(
                    update={"body_construction": "multi_piece_clamshell"}
                ),
            }
        ),
    ]
    expected_breakdown_counts = {
        "corner_joint": 5,
        "extrusion_filling": 6,
        "frame_feet": 5,
        "rail_quality": 4,
        "belt": 15,
        "belt_routing": 2,
        "gantry_bracket": 14,
        "toolhead_preset": 16,
        "toolhead_construction": 6,
        "z_probe": 4,
        "hotend_clamp": 14,
        "bed_mount": 13,
        "z_drive_mechanism": 12,
    }
    for spec_cat in catalog_complete_specs:
        entries_by_id = _by_id(spec_cat)
        for eid in CATALOG_BACKED_ENTRY_IDS:
            entry = entries_by_id[eid]
            bd = entry["options_breakdown"]
            assert len(bd) == expected_breakdown_counts[eid], f"{spec_cat.name}:{eid}"
            active_items = [item for item in bd if item["active"]]
            assert len(active_items) == 1, f"{spec_cat.name}:{eid} active count={len(active_items)}"
            assert active_items[0]["trust_pct"] == entry["trust_pct"], f"{spec_cat.name}:{eid}"

    # Legacy specs with body_construction=None report 0 active construction items and fallback 45% (when CAD SEs off)
    for legacy_yaml in ("corexy_voron_style.yaml", "voron_trident_300.yaml"):
        s_leg = _load_preset(legacy_yaml).model_copy(update={"use_cad_superelements": False})
        assert s_leg.toolhead.body_construction is None
        leg_entries = _by_id(s_leg)
        tc_leg = leg_entries["toolhead_construction"]
        assert tc_leg["trust_pct"] == 45
        assert [item for item in tc_leg["options_breakdown"] if item["active"]] == []
        for eid in CATALOG_BACKED_ENTRY_IDS:
            if eid == "toolhead_construction":
                continue
            active_items = [item for item in leg_entries[eid]["options_breakdown"] if item["active"]]
            assert len(active_items) == 1
            assert active_items[0]["trust_pct"] == leg_entries[eid]["trust_pct"]

    base = _load_preset("corexy_voron_style.yaml")

    # 1. toolhead_presets differentiation + custom fallback (65%)
    th_scores = {}
    for pid in (
        "stealthburner",
        "microwaved_bowden",
        "microwaved_bowden_chube",
        "el_fuego",
        "xol_compact",
        "archetype_cpap",
        "heavy_uhf_long",
    ):
        s = base.model_copy(deep=True)
        s.toolhead.preset_id = pid
        e = _by_id(s)["toolhead_preset"]
        th_scores[pid] = e["trust_pct"]
        assert len(e["options_breakdown"]) == len(TOOLHEAD_PRESETS)
        assert [x["id"] for x in e["options_breakdown"] if x["active"]] == [pid]
    assert th_scores == {
        "stealthburner": 88,
        "microwaved_bowden": 85,  # elevated from 84 -> 85 by baked 75k-elem Tet4 CAD CMS superelement
        "microwaved_bowden_chube": 86,  # elevated from 84 -> 86 by baked 75.3k-elem Tet4 CAD CMS superelement
        "el_fuego": 85,  # elevated from 83 -> 85 by baked 241.9k-elem Tet4 CAD CMS superelement
        "xol_compact": 86,
        "archetype_cpap": 82,
        "heavy_uhf_long": 56,
    }
    # Disabling CAD superelements reverts microwaved_bowden to its catalog baseline (84%)
    s_mb_lumped = base.model_copy(deep=True)
    s_mb_lumped.toolhead.preset_id = "microwaved_bowden"
    s_mb_lumped.use_cad_superelements = False
    assert _by_id(s_mb_lumped)["toolhead_preset"]["trust_pct"] == 84
    assert _by_id(s_mb_lumped)["toolhead_preset"]["cad_superelement_active"] is False
    s_custom_th = base.model_copy(deep=True)
    s_custom_th.toolhead.preset_id = "custom"
    e_custom_th = _by_id(s_custom_th)["toolhead_preset"]
    assert e_custom_th["trust_pct"] == 65
    assert "User-entered custom mass/CoG" in e_custom_th["basis"]
    assert [x for x in e_custom_th["options_breakdown"] if x["active"]] == []

    # 2. toolhead_constructions differentiation (lumped heuristic mode)
    tc_scores = {}
    for bc in ("multi_piece_clamshell", "compact_cage", "plate_sandwich_cf", "cnc_monoblock", "extruder_as_chassis"):
        s = base.model_copy(deep=True)
        s.use_cad_superelements = False
        s.toolhead.body_construction = bc
        tc_scores[bc] = _by_id(s)["toolhead_construction"]["trust_pct"]
    assert tc_scores == {
        "multi_piece_clamshell": 50,
        "compact_cage": 48,
        "plate_sandwich_cf": 44,
        "cnc_monoblock": 42,
        "extruder_as_chassis": 40,
    }
    # When a real CAD toolhead superelement (microwaved_bowden) is active, entry #16 reports
    # clean CAD Superelement selection, real Tet4 directional stiffnesses, CAD trust (85%),
    # and no generic lumped construction marked active; disabling CAD reverts to cnc_monoblock (42%).
    s_mb_cad = base.model_copy(deep=True)
    s_mb_cad.toolhead.preset_id = "microwaved_bowden"
    s_mb_cad.toolhead.body_construction = "cnc_monoblock"
    s_mb_cad.toolhead.mount_material = "aluminum_6061_t6"
    e_mb_tc = _by_id(s_mb_cad)["toolhead_construction"]
    assert e_mb_tc["cad_superelement_active"] is True
    assert e_mb_tc["trust_pct"] == 85
    assert e_mb_tc["active_selection"] == "CAD Superelement (Microwaved Bowden Toolhead (SLM Ti-6Al-4V))"
    assert "75,638-elem Tet4" in e_mb_tc["active_values"]
    assert "Kx =" in e_mb_tc["active_values"] and "Ky =" in e_mb_tc["active_values"]
    assert "Body wall" not in e_mb_tc["active_values"]
    assert [x for x in e_mb_tc["options_breakdown"] if x["active"]] == []
    s_mb_lumped_tc = s_mb_cad.model_copy(deep=True)
    s_mb_lumped_tc.use_cad_superelements = False
    e_mb_tc_off = _by_id(s_mb_lumped_tc)["toolhead_construction"]
    assert e_mb_tc_off["cad_superelement_active"] is False
    assert e_mb_tc_off["trust_pct"] == 42
    assert [x["id"] for x in e_mb_tc_off["options_breakdown"] if x["active"]] == ["cnc_monoblock"]

    # 3. belts differentiation across proven / experimental / theoretical
    belt_scores = {}
    for bid in ("gt2_6mm", "gt2_9mm", "gt2_12mm", "gt2_15mm", "3gt_9mm", "at5_16mm"):
        s = base.model_copy(deep=True)
        s.gantry.belt = s.gantry.belt.model_copy(update={"type": bid})
        belt_scores[bid] = _by_id(s)["belt"]["trust_pct"]
    assert belt_scores == {
        "gt2_6mm": 87,
        "gt2_9mm": 85,
        "gt2_12mm": 80,
        "gt2_15mm": 62,
        "3gt_9mm": 52,
        "at5_16mm": 40,
    }

    # 4. rail_qualities differentiation
    rq_scores = {}
    for qid in ("hiwin_genuine", "midgrade_matched", "precision_thk_nsk", "budget_generic"):
        s = base.model_copy(deep=True)
        s.gantry.rail_quality = qid
        rq_scores[qid] = _by_id(s)["rail_quality"]["trust_pct"]
    assert rq_scores == {
        "hiwin_genuine": 88,
        "midgrade_matched": 82,
        "precision_thk_nsk": 84,
        "budget_generic": 80,
    }

    # 5. hotends + hotend_mount_styles fallback differentiation
    he_scores = {}
    for hid in ("tricorn_long", "tricorn_short", "chube_conduction", "phaetus_dragon_hf", "e3d_v6_groove", "mellow_goliath_air"):
        s = base.model_copy(deep=True)
        s.toolhead.hotend.preset_id = hid
        e = _by_id(s)["hotend_clamp"]
        he_scores[hid] = e["trust_pct"]
        assert len(e["options_breakdown"]) == 14
        assert [x["id"] for x in e["options_breakdown"] if x["active"]] == [hid]
    assert he_scores == {
        "tricorn_long": 88,       # elevated from 78 -> 88 by baked 116k-elem Tet4 CAD CMS superelement
        "tricorn_short": 87,      # elevated from 76 -> 87 by baked 112k-elem Tet4 CAD CMS superelement
        "chube_conduction": 86,   # elevated from 78 -> 86 by baked 80.2k-elem Tet4 CAD CMS superelement
        "phaetus_dragon_hf": 86,  # CAL-12 baked 46.9k-elem Tet4 CAD CMS superelement
        "e3d_v6_groove": 85,      # baked 47.5k-elem Tet4 CAD CMS superelement
        "mellow_goliath_air": 85, # CAL-12 baked 51.8k-elem Tet4 CAD CMS superelement
    }
    # Disabling CAD superelements reverts baked hotends to their catalog baselines
    for hid, base_pct in (
        ("tricorn_long", 78),
        ("tricorn_short", 76),
        ("chube_conduction", 78),
        ("e3d_v6_groove", 68),
    ):
        s_off = base.model_copy(deep=True)
        s_off.toolhead.hotend.preset_id = hid
        s_off.use_cad_superelements = False
        e_off = _by_id(s_off)["hotend_clamp"]
        assert e_off["trust_pct"] == base_pct
        assert e_off["cad_superelement_active"] is False
    s_custom_he = _load_preset("flsun_v400_delta.yaml")
    assert s_custom_he.toolhead.hotend.preset_id is None
    e_custom_he = _by_id(s_custom_he)["hotend_clamp"]
    assert e_custom_he["trust_pct"] == 58  # rigid_4bolt_m3 fallback
    assert [x for x in e_custom_he["options_breakdown"] if x["active"]] == []

    # 6. z_probes differentiation (including cnc_tap vs printed_tap)
    zp_scores = {}
    for zid in ("static_eddy", "klicky_inductive", "cnc_tap", "printed_tap"):
        s = base.model_copy(deep=True)
        s.toolhead.z_probe_type = zid
        zp_scores[zid] = _by_id(s)["z_probe"]["trust_pct"]
    assert zp_scores == {
        "static_eddy": 94,
        "klicky_inductive": 86,
        "cnc_tap": 74,
        "printed_tap": 70,
    }


def test_model_confidence_audit_reflects_hardware_and_overrides() -> None:
    """Trust scores, tiers, active values, and overridden flags update dynamically."""
    spec = _load_preset("corexy_voron_style.yaml")
    rep = evaluate_printer(spec, include_advanced_analysis=False)
    assert rep.model_confidence == build_model_confidence_audit(spec)

    def _by_id(s: PrinterSpec) -> dict[str, dict]:
        return {e["id"]: e for e in build_model_confidence_audit(s)["entries"]}

    # Corner joint: rigid -> 95% exact; blind_joint -> 64% estimate; corner_cube -> 54% estimate
    s_rigid = spec.model_copy(deep=True)
    s_rigid.frame.corner_joint_type = "rigid"
    assert _by_id(s_rigid)["corner_joint"]["trust_pct"] == 95
    assert _by_id(s_rigid)["corner_joint"]["tier"] == "exact"

    s_bolted = spec.model_copy(deep=True)
    s_bolted.frame.corner_joint_type = "blind_joint"
    assert _by_id(s_bolted)["corner_joint"]["trust_pct"] == 84
    assert _by_id(s_bolted)["corner_joint"]["tier"] == "calibrated"

    s_cube = spec.model_copy(deep=True)
    s_cube.frame.corner_joint_type = "corner_cube"
    assert _by_id(s_cube)["corner_joint"]["trust_pct"] == 82

    # Enclosure panels: none -> 95%, bolted -> 86%, clip -> 80%
    s_open = spec.model_copy(deep=True)
    s_open.frame.enclosed_panels = False
    assert _by_id(s_open)["enclosure_panels"]["trust_pct"] == 95
    s_panel_bolt = spec.model_copy(deep=True)
    s_panel_bolt.frame.enclosed_panels = True
    s_panel_bolt.frame.panel_mounting = "bolted_structural"
    assert _by_id(s_panel_bolt)["enclosure_panels"]["trust_pct"] == 86
    s_panel_clip = spec.model_copy(deep=True)
    s_panel_clip.frame.enclosed_panels = True
    s_panel_clip.frame.panel_mounting = "clip_enclosure"
    assert _by_id(s_panel_clip)["enclosure_panels"]["trust_pct"] == 80

    # Extrusion filling: none (unoverridden) -> 96%, epoxy_granite -> 82%, silica_sand -> 80%, overridden -> 62%
    assert _by_id(spec)["extrusion_filling"]["trust_pct"] == 96
    s_sand = spec.model_copy(deep=True)
    s_sand.frame.extrusion_filling = "silica_sand"
    assert _by_id(s_sand)["extrusion_filling"]["trust_pct"] == 80
    s_fill = spec.model_copy(deep=True)
    s_fill.frame.extrusion_filling = "epoxy_granite"
    assert _by_id(s_fill)["extrusion_filling"]["trust_pct"] == 82
    s_fill_ov = spec.model_copy(deep=True)
    s_fill_ov.model_overrides.extrusion_filling.damping_boost = 0.02
    assert _by_id(s_fill_ov)["extrusion_filling"]["trust_pct"] == 62
    assert _by_id(s_fill_ov)["extrusion_filling"]["overridden"] is True

    # Frame feet: rigid_bolted (unoverridden) -> 95%, rubber_feet -> 79%, anti_vib_pads -> 78%, overridden -> 58%
    assert _by_id(spec)["frame_feet"]["trust_pct"] == 95
    s_feet = spec.model_copy(deep=True)
    s_feet.frame.feet_type = "rubber_feet"
    assert _by_id(s_feet)["frame_feet"]["trust_pct"] == 79
    s_pads = spec.model_copy(deep=True)
    s_pads.frame.feet_type = "anti_vib_pads"
    assert _by_id(s_pads)["frame_feet"]["trust_pct"] == 78
    s_feet_ov = spec.model_copy(deep=True)
    s_feet_ov.model_overrides.frame_feet.k_axial_n_per_mm = 400.0
    assert _by_id(s_feet_ov)["frame_feet"]["trust_pct"] == 58
    assert _by_id(s_feet_ov)["frame_feet"]["overridden"] is True

    # X-gantry beam: Aluminum -> 96%, Carbonara 3D Tet10 -> 90%, generic CF 3D Tet10 -> 88%
    assert _by_id(spec)["x_gantry_beam"]["trust_pct"] == 96
    s_cf = spec.model_copy(deep=True)
    s_cf.gantry.x_beam_profile = "cf_tube_20x20_carbonara"
    s_cf.gantry.x_beam_material = "carbon_fiber_torsional_45"
    assert _by_id(s_cf)["x_gantry_beam"]["trust_pct"] == 90
    assert _by_id(s_cf)["x_gantry_beam"]["tier"] == "exact"
    s_cf_gen = spec.model_copy(deep=True)
    s_cf_gen.gantry.x_beam_profile = "cf_tube_20x20x2"
    s_cf_gen.gantry.x_beam_material = "carbon_fiber_tube"
    assert _by_id(s_cf_gen)["x_gantry_beam"]["trust_pct"] == 88
    assert _by_id(s_cf_gen)["x_gantry_beam"]["tier"] == "calibrated"

    # Belt per-item confidence: gt2_6mm -> 87%, gt2_15mm -> 62%, at3_16mm -> 46%
    assert _by_id(spec)["belt"]["trust_pct"] == 87
    s_exp = spec.model_copy(deep=True)
    s_exp.gantry.belt = spec.gantry.belt.model_copy(update={"type": "gt2_15mm"})
    assert _by_id(s_exp)["belt"]["trust_pct"] == 62
    s_theo = spec.model_copy(deep=True)
    s_theo.gantry.belt = spec.gantry.belt.model_copy(update={"type": "at3_16mm"})
    assert _by_id(s_theo)["belt"]["trust_pct"] == 46
    assert _by_id(s_theo)["belt"]["tier"] == "heuristic"

    # Belt routing: stock_corexy -> 87%, monolith_short -> 86%
    assert _by_id(spec)["belt_routing"]["trust_pct"] == 87
    s_mono = spec.model_copy(deep=True)
    s_mono.gantry.belt_routing = "monolith_short"
    assert _by_id(s_mono)["belt_routing"]["trust_pct"] == 86

    # Gantry bracket: CAD-backed default >= 85%, PrintDB polymer fallback abs_printed -> 82%, aluminum_6061_t6 -> 66%, custom -> 85%
    assert _by_id(spec)["gantry_bracket"]["trust_pct"] >= 85
    assert _by_id(spec)["gantry_bracket"]["cad_superelement_active"] is True
    s_heur_bkt = spec.model_copy(deep=True)
    s_heur_bkt.use_cad_superelements = False
    assert _by_id(s_heur_bkt)["gantry_bracket"]["trust_pct"] == 82
    assert _by_id(s_heur_bkt)["gantry_bracket"]["overridden"] is False
    s_al_bkt = s_heur_bkt.model_copy(deep=True)
    s_al_bkt.gantry.bracket_material = "aluminum_6061_t6"
    assert _by_id(s_al_bkt)["gantry_bracket"]["trust_pct"] == 66
    s_cbo = spec.model_copy(deep=True)
    s_cbo.gantry.custom_bracket_override = CustomBracketOverride(
        label="CNC 7075 Motor Mount",
        stiffness_n_per_mm=9200.0,
        mass_delta_g=14.0,
        source="tet10_fea",
    )
    cbo_entry = _by_id(s_cbo)["gantry_bracket"]
    assert cbo_entry["trust_pct"] == 85
    assert cbo_entry["tier"] == "calibrated"
    assert cbo_entry["overridden"] is True
    assert "9,200 N/mm" in cbo_entry["active_values"]
    assert [x for x in cbo_entry["options_breakdown"] if x["active"]] == []

    # Z-probe: static_eddy -> 94%, klicky_inductive -> 86%, cnc_tap -> 74%, printed_tap -> 70%
    assert _by_id(spec)["z_probe"]["trust_pct"] == 94
    s_klicky = spec.model_copy(deep=True)
    s_klicky.toolhead.z_probe_type = "klicky_inductive"
    assert _by_id(s_klicky)["z_probe"]["trust_pct"] == 86
    s_cnc_tap = spec.model_copy(deep=True)
    s_cnc_tap.toolhead.z_probe_type = "cnc_tap"
    assert _by_id(s_cnc_tap)["z_probe"]["trust_pct"] == 74
    s_tap = spec.model_copy(deep=True)
    s_tap.toolhead.z_probe_type = "printed_tap"
    assert _by_id(s_tap)["z_probe"]["trust_pct"] == 70

    # Hotend clamp: phaetus_dragon_hf -> 86% (CAL-12 CAD CMS), e3d_v6_groove -> 85% (CAD CMS, 68% w/o CAD)
    assert _by_id(spec)["hotend_clamp"]["trust_pct"] == 86
    s_v6 = spec.model_copy(deep=True)
    s_v6.toolhead.hotend.preset_id = "e3d_v6_groove"
    assert _by_id(s_v6)["hotend_clamp"]["trust_pct"] == 85
    s_v6.use_cad_superelements = False
    assert _by_id(s_v6)["hotend_clamp"]["trust_pct"] == 68

    # Bed mount: kinematic_3pt -> 85%, rigid_4pt -> 86%, springs_4pt -> 80%
    assert _by_id(spec)["bed_mount"]["trust_pct"] == 85
    s_r4 = spec.model_copy(deep=True)
    s_r4.bed.mount_type = "rigid_4pt"
    assert _by_id(s_r4)["bed_mount"]["trust_pct"] == 86
    s_spr = spec.model_copy(deep=True)
    s_spr.bed.mount_type = "springs_4pt"
    assert _by_id(s_spr)["bed_mount"]["trust_pct"] == 80

    # Z-drive mechanism: gt2_6mm_z_belt -> 85%, tr8x4_wobblex -> 83%, gt2_9mm_worm_z -> 80%
    assert _by_id(spec)["z_drive_mechanism"]["trust_pct"] == 85
    s_wx = spec.model_copy(deep=True)
    s_wx.bed.z_leadscrew_type = "tr8x4_wobblex"
    assert _by_id(s_wx)["z_drive_mechanism"]["trust_pct"] == 83
    s_worm = spec.model_copy(deep=True)
    s_worm.bed.z_leadscrew_type = "gt2_9mm_worm_z"
    assert _by_id(s_worm)["z_drive_mechanism"]["trust_pct"] == 80

    # Setting overrides on all 14 groups flips overridden=True and updates active_values live
    s_all_ov = _load_preset("voron_trident_300.yaml")
    s_all_ov.model_overrides = ModelOverridesSpec.model_validate(
        {
            "corner_joint": {"k_t_n_per_mm": 18500.0},
            "extrusion_filling": {"density_kg_m3": 1950.0},
            "frame_feet": {"k_axial_n_per_mm": 420.0},
            "belt": {"stiffness_ea_n": 64000.0},
            "gantry_bracket": {"stiffness_n_per_mm": 3100.0},
            "rail_preload": {"preload_fraction": 0.085},
            "rail_quality": {"stiffness_mult": 1.35},
            "belt_routing": {"span_factor": 0.78},
            "bearing_clearance": {"radial_play_um": 3.2},
            "z_probe": {"compliance_factor": 0.72},
            "toolhead_construction": {"body_stiffness_base_n_per_mm": 640.0},
            "hotend_clamp": {"clamp_stiffness_base_n_per_mm": 4800.0},
            "bed_mount": {"k_axial_n_per_mm": 21500.0},
            "z_drive_mechanism": {"k_nut_n_per_um": 44.0},
        }
    )
    ov_by_id = _by_id(s_all_ov)
    for grp in ALL_14_GROUPS:
        assert ov_by_id[grp]["overridden"] is True
    assert "18,500 N/mm" in ov_by_id["corner_joint"]["active_values"]
    assert "1,950 kg/m³" in ov_by_id["extrusion_filling"]["active_values"]
    assert "420 N/mm" in ov_by_id["frame_feet"]["active_values"]
    assert "64,000 N" in ov_by_id["belt"]["active_values"]
    assert "3,100 N/mm" in ov_by_id["gantry_bracket"]["active_values"]
    assert "0.085×C" in ov_by_id["rail_preload"]["active_values"]
    assert "1.35×" in ov_by_id["rail_quality"]["active_values"]
    assert "0.78×" in ov_by_id["belt_routing"]["active_values"]
    assert "3.2 µm" in ov_by_id["bearing_clearance"]["active_values"]
    assert "0.72×" in ov_by_id["z_probe"]["active_values"]
    assert "640 N/mm" in ov_by_id["toolhead_construction"]["active_values"]
    assert "4,800 N/mm" in ov_by_id["hotend_clamp"]["active_values"]
    assert "21,500 N/mm" in ov_by_id["bed_mount"]["active_values"]
    assert "44.0 N/µm" in ov_by_id["z_drive_mechanism"]["active_values"]


def test_rcsa_ballbar_rail_overrides_and_entry15_accessories() -> None:
    """Verify rcsa.py and ballbar.py obey rail_preload/rail_quality overrides and entry #15 shows accessory mass/CoG."""
    import numpy as np

    from printsim.fea.modal import solve_modal_fea
    from printsim.fea.rcsa import analyze_toolhead_rcsa
    from printsim.gcode.ballbar import simulate_time_domain_metrology
    from printsim.geometry.builder import build_assembly
    from printsim.schema.printer import HotendSpec, TOOLHEAD_PRESETS

    s_lo = _load_preset("corexy_voron_style.yaml")
    s_lo.toolhead.hotend = HotendSpec(preset_id="chube_conduction")
    s_lo.model_overrides.rail_preload.preload_fraction = 0.005
    s_lo.model_overrides.rail_quality.stiffness_mult = 0.45
    s_lo.model_overrides.rail_quality.drag_mult = 0.50

    s_hi = s_lo.model_copy(deep=True)
    s_hi.model_overrides.rail_preload.preload_fraction = 0.14
    s_hi.model_overrides.rail_quality.stiffness_mult = 1.60
    s_hi.model_overrides.rail_quality.drag_mult = 2.80

    a_lo, a_hi = build_assembly(s_lo), build_assembly(s_hi)
    m_lo, m_hi = solve_modal_fea(a_lo), solve_modal_fea(a_hi)
    r_lo = analyze_toolhead_rcsa(a_lo, m_lo, mount_stiffness_scale=10.0)
    r_hi = analyze_toolhead_rcsa(a_hi, m_hi, mount_stiffness_scale=10.0)
    b_lo = simulate_time_domain_metrology(a_lo, m_lo)
    b_hi = simulate_time_domain_metrology(a_hi, m_hi)

    assert r_lo.nozzle_to_sensor_amp_ratio > r_hi.nozzle_to_sensor_amp_ratio
    assert not np.allclose(r_lo.nozzle_frf_mag, r_hi.nozzle_frf_mag)
    assert b_hi.quadrant_glitch_um > b_lo.quadrant_glitch_um * 3.0
    assert b_hi.circularity_gxy_um > b_lo.circularity_gxy_um * 3.0

    # Verify Model Confidence entry #15 reflects effective accessory mass & CoG deltas
    th_p = TOOLHEAD_PRESETS["microwaved_bowden"]
    s_mb = _load_preset("voron_monolith_awd.yaml")
    s_mb.toolhead.preset_id = "microwaved_bowden"
    s_mb.toolhead.total_mass_g = float(th_p["total_mass_g"])
    s_mb.toolhead.cog_offset_mm = tuple(float(v) for v in th_p["cog_offset_mm"])
    s_mb.toolhead.nozzle_offset_mm = tuple(float(v) for v in th_p["nozzle_offset_mm"])
    s_mb.toolhead.hotend = HotendSpec(preset_id="tricorn_long")
    s_mb.toolhead.use_hotend_bottom_brace = True
    s_mb.toolhead.nozzle_shield_mode = "microwaved"
    e15_on = {e["id"]: e for e in build_model_confidence_audit(s_mb)["entries"]}["toolhead_preset"]
    assert "130.0 g (+59.3 g hotend)" in e15_on["active_values"]

    s_mb_off = s_mb.model_copy(deep=True)
    s_mb_off.toolhead.use_hotend_bottom_brace = False
    s_mb_off.toolhead.nozzle_shield_mode = "none"
    e15_off = {e["id"]: e for e in build_model_confidence_audit(s_mb_off)["entries"]}["toolhead_preset"]
    assert "127.1 g (+59.3 g hotend)" in e15_off["active_values"]
    assert e15_off["active_values"] != e15_on["active_values"]
