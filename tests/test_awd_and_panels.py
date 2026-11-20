"""Physics verification tests for CoreXY AWD, high-tension belts, and structural shear panels."""

from __future__ import annotations

import math

import pytest

from printsim.engine import evaluate_printer
from printsim.fea import solve_modal_fea, solve_static_fea
from printsim.geometry import build_assembly, generate_bom
from printsim.kinematics import analyze_kinematics
from printsim.schema import PrinterSpec, get_material


def _panel_corexy_spec(
    enclosed: bool,
    mounting: str = "bolted_structural",
    material: str = "acm_dibond",
    thickness_mm: float = 3.0,
) -> PrinterSpec:
    spec = PrinterSpec(architecture="corexy")
    spec.frame.bracing = "none"
    spec.frame.enclosed_panels = enclosed
    spec.frame.panel_mounting = mounting  # type: ignore[assignment]
    spec.frame.panel_material = material
    spec.frame.panel_thickness_mm = thickness_mm
    return spec


def test_equivalent_shear_panel_formula_and_geometry_dict() -> None:
    """Verify exact A_eq = eta * G * t * L_d^3 / (2 * E * b * h), shape='plate', and metadata."""
    spec = _panel_corexy_spec(True, "bolted_structural", "aluminum_5052_sheet", 4.0)
    spec.gantry.drive_mode = "awd"
    spec.gantry.double_shear_pulleys = True

    asm = build_assembly(spec)
    pmat = get_material("aluminum_5052_sheet")

    rear_1 = next(m for m in asm.members if m.id == "panel_shear_rear_1")
    assert rear_1.is_truss is True
    assert rear_1.profile.id == "panel_shear_rear"
    assert rear_1.profile.shape == "plate"
    assert rear_1.material.id == "aluminum_5052_sheet"

    b = spec.frame.width_mm * 1e-3
    h = spec.frame.height_mm * 1e-3
    t = spec.frame.panel_thickness_mm * 1e-3
    ld = math.hypot(b, h)
    expected_aeq = 0.65 * (pmat.effective_shear_modulus_pa * t * (ld**3)) / (
        2.0 * pmat.youngs_modulus_pa * b * h
    )
    assert rear_1.profile.area_m2 == pytest.approx(expected_aeq, rel=1e-6)

    gdict = asm.to_geometry_dict()
    assert gdict["drive_mode"] == "awd"
    assert gdict["double_shear_pulleys"] is True
    assert gdict["panels"] == {
        "enabled": True,
        "mounting": "bolted_structural",
        "material_id": "aluminum_5052_sheet",
        "material_name": pmat.name,
        "thickness_mm": 4.0,
        "color_hex": pmat.color_hex,
    }


def test_structural_panel_monotonicity_frequency_stiffness_and_mass() -> None:
    """Verify open < clip < bolted acrylic < bolted 3mm Dibond < 6mm Dibond < 4mm Al5052."""
    configs = [
        ("open", _panel_corexy_spec(False, "none", "acm_dibond", 3.0)),
        ("clip_3mm_acrylic", _panel_corexy_spec(True, "clip_enclosure", "acrylic_pmma", 3.0)),
        ("bolt_3mm_acrylic", _panel_corexy_spec(True, "bolted_structural", "acrylic_pmma", 3.0)),
        ("bolt_3mm_dibond", _panel_corexy_spec(True, "bolted_structural", "acm_dibond", 3.0)),
        ("bolt_6mm_dibond", _panel_corexy_spec(True, "bolted_structural", "acm_dibond", 6.0)),
        ("bolt_4mm_al5052", _panel_corexy_spec(True, "bolted_structural", "aluminum_5052_sheet", 4.0)),
    ]

    results = []
    for label, spec in configs:
        asm = build_assembly(spec)
        st = solve_static_fea(asm)
        mo = solve_modal_fea(asm)
        results.append((label, mo.fundamental_freq_hz, st.toolhead_stiffness_x_n_per_mm, st.toolhead_stiffness_y_n_per_mm, asm.total_mass_kg))

    for i in range(len(results) - 1):
        curr_label, f_c, kx_c, ky_c, _ = results[i]
        next_label, f_n, kx_n, ky_n, _ = results[i + 1]
        assert f_n > f_c, f"Expected f1({next_label})={f_n:.3f} > f1({curr_label})={f_c:.3f}"
        assert kx_n > kx_c, f"Expected kx({next_label})={kx_n:.2f} > kx({curr_label})={kx_c:.2f}"
        assert ky_n > ky_c, f"Expected ky({next_label})={ky_n:.2f} > ky({curr_label})={ky_c:.2f}"

    # Thickness increase (3mm -> 6mm Dibond) increases both stiffness and total printer mass
    _, _, kx_3, ky_3, m_3 = results[3]
    _, _, kx_6, ky_6, m_6 = results[4]
    assert kx_6 > kx_3 and ky_6 > ky_3
    assert m_6 > m_3 + 2.5

    # Even in clip_enclosure mode, thickness (3mm -> 6mm) and heavier material (PC -> Al 5052) increase mass
    clip_pc_3 = build_assembly(_panel_corexy_spec(True, "clip_enclosure", "polycarbonate", 3.0))
    clip_pc_6 = build_assembly(_panel_corexy_spec(True, "clip_enclosure", "polycarbonate", 6.0))
    clip_al_3 = build_assembly(_panel_corexy_spec(True, "clip_enclosure", "aluminum_5052_sheet", 3.0))
    assert clip_pc_6.total_mass_kg > clip_pc_3.total_mass_kg + 2.0
    assert clip_al_3.total_mass_kg > clip_pc_3.total_mass_kg + 2.5

    # panel_mounting == "none" adds no shear struts even if enclosed_panels=True
    none_mount = build_assembly(_panel_corexy_spec(True, "none", "aluminum_5052_sheet", 4.0))
    open_asm = build_assembly(configs[0][1])
    assert not any(m.id.startswith("panel_shear_") for m in none_mount.members)
    assert none_mount.to_geometry_dict()["panels"]["enabled"] is False
    assert solve_modal_fea(none_mount).fundamental_freq_hz == pytest.approx(
        solve_modal_fea(open_asm).fundamental_freq_hz, rel=1e-6
    )


def test_delta_outer_bay_shear_panels_and_bom_exclusion() -> None:
    """Delta enclosed outer bays add shear stiffness and exclude virtual struts from BOM."""
    spec_open = PrinterSpec(architecture="delta")
    spec_open.kinematics.build_volume_mm = (200.0, 200.0, 250.0)
    spec_open.kinematics.delta_arm_length_mm = 300.0
    spec_open.kinematics.delta_radius_mm = 155.0
    spec_open.frame.bracing = "none"
    spec_open.frame.enclosed_panels = False

    spec_clip = spec_open.model_copy(deep=True)
    spec_clip.frame.enclosed_panels = True
    spec_clip.frame.panel_mounting = "clip_enclosure"
    spec_clip.frame.panel_material = "acrylic_pmma"

    spec_bolt = spec_clip.model_copy(deep=True)
    spec_bolt.frame.panel_mounting = "bolted_structural"
    spec_bolt.frame.panel_material = "acm_dibond"

    asm_open = build_assembly(spec_open)
    asm_clip = build_assembly(spec_clip)
    asm_bolt = build_assembly(spec_bolt)

    assert len([m for m in asm_bolt.members if m.id.startswith("panel_shear_")]) == 6
    assert solve_modal_fea(asm_bolt).fundamental_freq_hz > solve_modal_fea(asm_clip).fundamental_freq_hz > solve_modal_fea(asm_open).fundamental_freq_hz
    assert solve_static_fea(asm_bolt).toolhead_stiffness_x_n_per_mm > solve_static_fea(asm_clip).toolhead_stiffness_x_n_per_mm > solve_static_fea(asm_open).toolhead_stiffness_x_n_per_mm

    bom = generate_bom(spec_bolt)
    assert all(c.role != "brace" for c in bom.cut_list)


def test_belt_induced_frame_preload_and_panel_resistance() -> None:
    """High belt tension deforms unbraced 2020 frames; gravity sag stays invariant; panels/3030 resist."""
    spec_25 = _panel_corexy_spec(False, "none")
    spec_25.gantry.belt.pretension_n = 25.0

    spec_90 = spec_25.model_copy(deep=True)
    spec_90.gantry.belt.pretension_n = 90.0

    spec_90_bolted = spec_90.model_copy(deep=True)
    spec_90_bolted.frame.enclosed_panels = True
    spec_90_bolted.frame.panel_mounting = "bolted_structural"
    spec_90_bolted.frame.panel_material = "acm_dibond"

    spec_90_3030 = spec_90.model_copy(deep=True)
    spec_90_3030.frame.profile = "extrusion_3030"

    fea_25 = solve_static_fea(build_assembly(spec_25))
    fea_90 = solve_static_fea(build_assembly(spec_90))
    fea_90_bolted = solve_static_fea(build_assembly(spec_90_bolted))
    fea_90_3030 = solve_static_fea(build_assembly(spec_90_3030))

    assert fea_90.gravity_sag_toolhead_um == pytest.approx(fea_25.gravity_sag_toolhead_um, rel=1e-6)
    assert fea_90.combined_max_deflection_um > fea_25.combined_max_deflection_um * 1.4
    assert fea_90.max_von_mises_mpa > fea_25.max_von_mises_mpa * 1.3
    assert fea_90_bolted.combined_max_deflection_um < fea_90.combined_max_deflection_um * 0.5
    assert fea_90_3030.combined_max_deflection_um < fea_90.combined_max_deflection_um * 0.5
    assert fea_90_3030.max_von_mises_mpa < fea_90.max_von_mises_mpa


def test_corexy_awd_vs_2wd_at_high_accel() -> None:
    """Compare identical CoreXY at 20,000 mm/s^2 in 2WD vs AWD."""
    spec_2wd = _panel_corexy_spec(True, "bolted_structural", "acm_dibond", 4.0)
    spec_2wd.kinematics.max_accel_mm_s2 = 20000.0
    spec_2wd.gantry.belt.type = "gt2_9mm"
    spec_2wd.gantry.belt.core = "aramid"
    spec_2wd.gantry.belt.pretension_n = 55.0
    spec_2wd.gantry.double_shear_pulleys = True
    spec_2wd.gantry.drive_mode = "2wd"

    spec_awd = spec_2wd.model_copy(deep=True)
    spec_awd.gantry.drive_mode = "awd"

    asm_2wd = build_assembly(spec_2wd)
    asm_awd = build_assembly(spec_awd)

    # Front corners carry front XY motor + mount mass in AWD
    nmap_awd = asm_awd.node_map
    expected_front_kg = spec_awd.kinematics.xy_stepper.motor_mass_g * 1e-3 + 0.03
    for nid in ("g_fl", "g_fr"):
        assert nmap_awd[nid].role == "mount"
        assert nmap_awd[nid].lumped_mass_kg == pytest.approx(expected_front_kg, rel=1e-6)
    assert asm_awd.total_mass_kg > asm_2wd.total_mass_kg + 0.6

    kin_2wd = analyze_kinematics(asm_2wd)
    kin_awd = analyze_kinematics(asm_awd)
    mod_2wd = solve_modal_fea(asm_2wd)
    mod_awd = solve_modal_fea(asm_awd)

    assert kin_2wd.xy_motor_count == 2
    assert kin_awd.xy_motor_count == 4
    assert kin_awd.drive_mode == "awd"
    assert kin_awd.frame_belt_preload_n == pytest.approx(2.0 * kin_2wd.frame_belt_preload_n, rel=1e-6)
    assert kin_awd.torque_safety_margin > kin_2wd.torque_safety_margin * 1.45

    # Dynamic belt tension swing (peak - pretension) is roughly halved in AWD
    swing_2wd = kin_2wd.belt_peak_tension_n - spec_2wd.gantry.belt.pretension_n
    swing_awd = kin_awd.belt_peak_tension_n - spec_awd.gantry.belt.pretension_n
    assert swing_awd < swing_2wd * 0.6

    assert mod_awd.shaper_x is not None and mod_2wd.shaper_x is not None
    assert mod_awd.shaper_y is not None and mod_2wd.shaper_y is not None
    assert mod_awd.shaper_x.belt_freq_hz > mod_2wd.shaper_x.belt_freq_hz * 1.7
    assert mod_awd.shaper_y.belt_freq_hz > mod_2wd.shaper_y.belt_freq_hz * 1.7
    assert mod_awd.recommended_max_accel_mm_s2 > mod_2wd.recommended_max_accel_mm_s2 * 1.5


def test_high_tension_double_shear_and_pluck_frequency() -> None:
    """Verify overtension/shaft-bending flags, diagnostics, dynamics score, and 150mm pluck freq."""
    spec_risky = _panel_corexy_spec(True, "bolted_structural", "acm_dibond", 4.0)
    spec_risky.gantry.belt.type = "gt2_6mm"
    spec_risky.gantry.belt.core = "fiberglass"
    spec_risky.gantry.belt.pretension_n = 70.0
    spec_risky.gantry.double_shear_pulleys = False

    rep_risky = evaluate_printer(spec_risky)
    assert rep_risky.kinematics.shaft_bending_risk is True
    assert rep_risky.kinematics.belt_overtension_risk is True
    joined_diag = " ".join(rep_risky.diagnostics).lower()
    assert "overtension" in joined_diag
    assert "double-shear" in joined_diag

    # Enabling double-shear clears shaft bending risk and raises dynamics score
    spec_ds = spec_risky.model_copy(deep=True)
    spec_ds.gantry.double_shear_pulleys = True
    rep_ds = evaluate_printer(spec_ds)
    assert rep_ds.kinematics.shaft_bending_risk is False
    assert rep_ds.kinematics.belt_overtension_risk is True
    assert rep_ds.scores.dynamics > rep_risky.scores.dynamics

    # Upgrading to 9mm Aramid + double-shear clears both flags and boosts dynamics further
    spec_safe = spec_ds.model_copy(deep=True)
    spec_safe.gantry.belt.type = "gt2_9mm"
    spec_safe.gantry.belt.core = "aramid"
    rep_safe = evaluate_printer(spec_safe)
    assert rep_safe.kinematics.shaft_bending_risk is False
    assert rep_safe.kinematics.belt_overtension_risk is False
    assert rep_safe.scores.dynamics > rep_ds.scores.dynamics

    # Exact transverse belt pluck frequency check: f = (1 / 2L) * sqrt(T / mu) within < 0.1%
    mu = spec_safe.gantry.belt.linear_mass_kg_m
    expected_pluck = (1.0 / (2.0 * 0.150)) * math.sqrt(70.0 / mu)
    assert rep_safe.kinematics.belt_pluck_freq_150mm_hz == pytest.approx(expected_pluck, rel=1e-3)
    assert rep_safe.kinematics.to_dict()["belt_pluck_freq_150mm_hz"] == pytest.approx(expected_pluck, rel=1e-3)

    # Belt pretension stiffness coupling increases modal belt_freq_hz from 15 N -> 30 N -> 45 N
    mod_15 = solve_modal_fea(build_assembly(_with_pretension(spec_safe, 15.0)))
    mod_30 = solve_modal_fea(build_assembly(_with_pretension(spec_safe, 30.0)))
    mod_45 = solve_modal_fea(build_assembly(_with_pretension(spec_safe, 45.0)))
    assert mod_15.shaper_x is not None and mod_30.shaper_x is not None and mod_45.shaper_x is not None
    assert mod_15.shaper_x.belt_freq_hz < mod_30.shaper_x.belt_freq_hz < mod_45.shaper_x.belt_freq_hz


def _with_pretension(spec: PrinterSpec, pretension_n: float) -> PrinterSpec:
    out = spec.model_copy(deep=True)
    out.gantry.belt.pretension_n = pretension_n
    return out
