"""Tests for mixed frame profiles (upright_profile), RatRig V-Core 3.1 300 preset, and KB cleanup."""

from __future__ import annotations

from pathlib import Path

import pytest

from printsim.engine import evaluate_printer
from printsim.fea import solve_modal_fea, solve_static_fea
from printsim.geometry import build_assembly, generate_bom
from printsim.kb import (
    TouchstoneKB,
    extract_structured_records,
    infer_preset_hint,
    run_kb_calibration,
    seed_from_benchmark_docs,
)
from printsim.kb.extract import is_malformed_build_name
from printsim.schema import PrinterSpec

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


# ---------------------------------------------------------------------------
# 1. Mixed Frame Profiles (FrameSpec.upright_profile)
# ---------------------------------------------------------------------------


def test_mixed_frame_profile_corexy_mass_stiffness_and_bom() -> None:
    """4040 uprights + 2020 horizontals sits between all-2020 and all-4040 mass, stiffens columns, and splits BOM."""
    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    base.frame.enclosed_panels = False
    base.frame.panel_mounting = "none"
    base.frame.corner_joint_type = "corner_cube"

    spec_2020 = base.model_copy(deep=True)
    spec_2020.frame.profile = "extrusion_2020"
    spec_2020.frame.upright_profile = None
    assert spec_2020.frame.effective_upright_profile == "extrusion_2020"

    spec_mixed = base.model_copy(deep=True)
    spec_mixed.frame.profile = "extrusion_2020"
    spec_mixed.frame.upright_profile = "extrusion_4040"
    assert spec_mixed.frame.effective_upright_profile == "extrusion_4040"

    spec_4040 = base.model_copy(deep=True)
    spec_4040.frame.profile = "extrusion_4040"
    spec_4040.frame.upright_profile = None

    asm_2020 = build_assembly(spec_2020)
    asm_mixed = build_assembly(spec_mixed)
    asm_4040 = build_assembly(spec_4040)

    # Verify member-level profile assignment on mixed CoreXY frame
    col_members = [m for m in asm_mixed.members if m.id.startswith("col_")]
    horiz_members = [
        m for m in asm_mixed.members if m.id.startswith(("base_", "top_", "y_rail_", "gantry_rear_bar"))
    ]
    assert len(col_members) >= 8
    assert len(horiz_members) >= 10
    assert all(m.profile.id == "extrusion_4040" for m in col_members)
    assert all(m.profile.id == "extrusion_2020" for m in horiz_members)

    # Mass ordering: all-2020 < mixed 4040/2020 < all-4040
    assert asm_2020.frame_mass_kg < asm_mixed.frame_mass_kg < asm_4040.frame_mass_kg

    # Column bending inertia & structural stiffness strictly stiffer than all-2020
    col_2020 = next(m for m in asm_2020.members if m.id == "col_rl_low")
    col_mixed = next(m for m in asm_mixed.members if m.id == "col_rl_low")
    assert col_mixed.profile.iy_m4 > col_2020.profile.iy_m4 * 4.0
    assert col_mixed.profile.iz_m4 > col_2020.profile.iz_m4 * 4.0

    st_2020 = solve_static_fea(asm_2020)
    st_mixed = solve_static_fea(asm_mixed)
    mo_2020 = solve_modal_fea(asm_2020)
    mo_mixed = solve_modal_fea(asm_mixed)
    assert st_mixed.toolhead_stiffness_x_n_per_mm > st_2020.toolhead_stiffness_x_n_per_mm
    assert st_mixed.toolhead_stiffness_y_n_per_mm > st_2020.toolhead_stiffness_y_n_per_mm
    assert mo_mixed.fundamental_freq_hz > mo_2020.fundamental_freq_hz

    # BOM cut list and 1D stock nesting separate 4040 uprights and 2020 horizontals
    bom = generate_bom(spec_mixed)
    nest_profiles = {s.profile_id for s in bom.stock_nesting}
    assert "extrusion_4040" in nest_profiles
    assert "extrusion_2020" in nest_profiles

    col_cuts = [c for c in bom.cut_list if c.profile_id == "extrusion_4040"]
    assert len(col_cuts) == 1
    assert col_cuts[0].quantity == 4
    assert col_cuts[0].length_mm == pytest.approx(spec_mixed.frame.height_mm, abs=0.1)


def test_mixed_frame_profile_bedslinger_delta_and_doomcube_preset() -> None:
    """Verify upright_profile works across Cartesian Bedslinger, Delta, and DoomCube 4040 preset."""
    # Cartesian Bedslinger: portal uprights use upright_profile, base & portal top stay profile
    bs = PrinterSpec.from_yaml(EXAMPLES_DIR / "cartesian_bedslinger.yaml")
    bs.frame.profile = "extrusion_2020"
    bs.frame.upright_profile = "extrusion_2040"
    asm_bs = build_assembly(bs)
    assert all(m.profile.id == "extrusion_2040" for m in asm_bs.members if m.id.startswith("z_col_"))
    assert all(m.profile.id == "extrusion_2020" for m in asm_bs.members if m.id.startswith("base_"))
    assert next(m for m in asm_bs.members if m.id == "portal_top").profile.id == "extrusion_2020"

    # Delta: vertical towers use upright_profile, base & top rings stay profile
    delta = PrinterSpec.from_yaml(EXAMPLES_DIR / "delta_racer.yaml")
    delta.frame.profile = "extrusion_2020"
    delta.frame.upright_profile = "extrusion_3030"
    asm_d = build_assembly(delta)
    assert all(m.profile.id == "extrusion_3030" for m in asm_d.members if m.id.startswith("tower_"))
    assert all(m.profile.id == "extrusion_2020" for m in asm_d.members if m.id.startswith(("base_", "top_")))

    # DoomCube preset: authentic 4040 uprights + 2020 horizontals
    doom = PrinterSpec.from_yaml(EXAMPLES_DIR / "doomcube_4040.yaml")
    assert doom.frame.profile == "extrusion_2020"
    assert doom.frame.upright_profile == "extrusion_4040"
    bom_doom = generate_bom(doom)
    nest_profs = {s.profile_id for s in bom_doom.stock_nesting}
    assert {"extrusion_4040", "extrusion_2020"}.issubset(nest_profs)
    rep_doom = evaluate_printer(doom, include_advanced_analysis=False)
    assert not rep_doom.collision.has_collision
    assert rep_doom.scores.composite > 85.0


# ---------------------------------------------------------------------------
# 2. Dedicated RatRig V-Core 3.1 300 Preset
# ---------------------------------------------------------------------------


def test_ratrig_vcore31_300_preset_evaluates_near_community_touchstone() -> None:
    """RatRig V-Core 3.1 300 preset evaluates cleanly within ~25% of measured 74 Hz X / 54 Hz Y."""
    preset_path = EXAMPLES_DIR / "ratrig_vcore31_300.yaml"
    assert preset_path.exists()

    spec = PrinterSpec.from_yaml(preset_path)
    assert spec.architecture == "corexy"
    assert spec.frame.profile == "extrusion_3030"
    assert spec.gantry.drive_mode == "2wd"
    assert spec.gantry.belt.type == "gt2_9mm"
    assert spec.gantry.x_rail_mount == "top"
    assert spec.bed.z_drive == "leadscrew_triple"
    assert spec.bed.mount_type == "kinematic_3pt"

    rep = evaluate_printer(spec, include_advanced_analysis=False)
    assert not rep.collision.has_collision
    assert rep.modal_fea.shaper_x is not None
    assert rep.modal_fea.shaper_y is not None

    pred_x = rep.modal_fea.shaper_x.effective_freq_hz
    pred_y = rep.modal_fea.shaper_y.effective_freq_hz
    assert pred_x > pred_y
    # Within 25% of community touchstone (74.0 Hz X / 54.0 Hz Y)
    assert abs(pred_x - 74.0) / 74.0 <= 0.25, f"RatRig X={pred_x:.1f} Hz outside ±25% of 74 Hz"
    assert abs(pred_y - 54.0) / 54.0 <= 0.25, f"RatRig Y={pred_y:.1f} Hz outside ±25% of 54 Hz"


# ---------------------------------------------------------------------------
# 3. KB Mapping & Error-Log Noise Filtering
# ---------------------------------------------------------------------------


def test_kb_noise_filtering_and_ratrig_mapping(tmp_path: Path) -> None:
    """KB rejects malformed error-log build names and maps RatRig V-Core builds to ratrig_vcore31_300."""
    # Preset hint mapping
    assert infer_preset_hint(printer_model="RatRig V-Core 3.1 300", frame_profile="3030", belt_width_mm=9.0) == "ratrig_vcore31_300"
    assert infer_preset_hint(printer_model="RatRig V-Core 3.1 400") == "ratrig_vcore31_300"
    assert infer_preset_hint(printer_model="VCore 3.1") == "ratrig_vcore31_300"

    # Malformed build name detection
    for bad in (
        "Failed to configure shaper(s) shaper_z with given parameters",
        "Error: MCU 'mcu' shutdown",
        "Traceback (most recent call last):",
        "shaper_z config failed",
        "   ",
        "12345",
    ):
        assert is_malformed_build_name(bad) is True, f"Expected malformed: {bad!r}"

    for good in (
        "RatRig V-Core 3.1 300",
        "Stock Voron Trident 300 (Baseline)",
        "DoomCube 4040 AWD",
        "Hanarius (VT 250)",
    ):
        assert is_malformed_build_name(good) is False, f"Unexpectedly rejected: {good!r}"

    # Deterministic extractor drops error-log posts even when body has shaper Hz
    err_extracted = extract_structured_records(
        text="Printer Model: voron trident\nRecommended shaper_type_y = mzv, shaper_freq_y = 47.2 Hz",
        title="Failed to configure shaper(s) shaper_z with given parameters",
    )
    assert err_extracted["builds"] == []

    # Calibration filters injected noise rows and resolves legacy RatRig hints to ratrig_vcore31_300.yaml
    db_path = tmp_path / "kb_clean.db"
    with TouchstoneKB(db_path) as kb:
        seed_from_benchmark_docs(kb)
        # Inject the exact real-world Discourse error row + an identity-less row
        kb.add_build(
            build_name="Failed to configure shaper(s) shaper_z with given parameters",
            architecture="corexy",
            printer_model="voron trident",
            y_shaper_type="mzv",
            y_shaper_hz=47.2,
            preset_hint="voron_trident_300",
        )
        kb.add_build(
            build_name="Raw resonance data needed",
            architecture="corexy",
            printer_model="",
            x_shaper_hz=55.0,
            preset_hint="corexy_voron_style",
        )
        # Also inject a RatRig build with legacy preset_hint="hyper_corexy_carbon"
        kb.add_build(
            build_name="Legacy Seeded RatRig V-Core 3.1 300",
            architecture="corexy",
            printer_model="RatRig V-Core 3.1 300",
            frame_profile="3030",
            drive_mode="2wd",
            x_shaper_hz=74.0,
            y_shaper_hz=54.0,
            preset_hint="hyper_corexy_carbon",
        )
        cal = run_kb_calibration(kb)

    names = [r["build_name"] for r in cal["results"]]
    assert not any("Failed to configure" in n for n in names)
    assert not any("Raw resonance data needed" in n for n in names)
    assert cal["builds_skipped_unmapped"] >= 2

    ratrig_rows = [r for r in cal["results"] if "RatRig" in r["build_name"]]
    assert len(ratrig_rows) == 3  # 2 seeded + 1 legacy-hinted
    for row in ratrig_rows:
        assert row["preset_file"] == "ratrig_vcore31_300.yaml"
    rr300 = next(r for r in ratrig_rows if r["build_name"] == "RatRig V-Core 3.1 300")
    assert abs(rr300["x_error_pct"]) < 22.0
    assert abs(rr300["y_error_pct"]) < 15.0


# ---------------------------------------------------------------------------
# 4. VzBot 4x10mm Belt-Synced Dual-Z, Mid-Y Continuity & Z-Motor Mass Counts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("z_drive", ["leadscrew_dual", "leadscrew_dual_belt_sync"])
def test_dual_z_mid_y_modal_continuity_without_carriage_weld(z_drive: str) -> None:
    """Modal frequencies vary smoothly across mid-Y (0.45..0.55) with zero carriage-to-frame weld spike."""
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "vzbot_330_awd.yaml")
    spec.bed.z_drive = z_drive  # type: ignore[assignment]

    v_positions = (0.45, 0.49, 0.50, 0.51, 0.55)
    f1_vals: list[float] = []
    sx_vals: list[float] = []
    sy_vals: list[float] = []

    for v_rel in v_positions:
        asm = build_assembly(spec, toolhead_rel_pos=(0.5, v_rel, 0.5))
        # Ensure the unphysical 1e9 mid-Y carriage weld spring is completely absent
        assert not any(sp.id.startswith("z_rail_top_tie_") for sp in asm.fastener_springs)
        mo = solve_modal_fea(asm)
        assert mo.shaper_x is not None and mo.shaper_y is not None
        f1_vals.append(mo.fundamental_freq_hz)
        sx_vals.append(mo.shaper_x.effective_freq_hz)
        sy_vals.append(mo.shaper_y.effective_freq_hz)

    # Assert smooth continuity across mid-Y: no step > 3 Hz in f1, X shaper, or Y shaper
    for i in range(len(v_positions) - 1):
        assert abs(f1_vals[i + 1] - f1_vals[i]) < 3.0, f"{z_drive} f1 jump: {f1_vals}"
        assert abs(sx_vals[i + 1] - sx_vals[i]) < 3.0, f"{z_drive} shaper_x jump: {sx_vals}"
        assert abs(sy_vals[i + 1] - sy_vals[i]) < 3.0, f"{z_drive} shaper_y jump: {sy_vals}"


def test_vzbot_330_awd_belt_sync_4x10mm_rods_bom_and_evaluation() -> None:
    """VzBot 330 AWD uses leadscrew_dual_belt_sync (1 Z motor, 4x10mm smooth rods, LMK10LUU, sync belt) and evaluates cleanly."""
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "vzbot_330_awd.yaml")
    assert spec.bed.z_drive == "leadscrew_dual_belt_sync"
    assert spec.bed.z_rail_profile == "steel_rod_10mm"
    assert spec.is_flying_gantry is False
    assert spec.panels_bolted_to_gantry_effective is True

    # Structural assembly builds 4x vertical smooth rods + 4x LMK bearing springs + 2x leadscrews
    asm = build_assembly(spec)
    rod_members = [m for m in asm.members if m.id.startswith("bed_rod_")]
    assert len(rod_members) == 8  # 4 rods x (bot + top)
    assert all(m.profile.id == "steel_rod_10mm" for m in rod_members)
    brg_springs = [sp for sp in asm.fastener_springs if sp.id.startswith("z_rod_brg_")]
    assert len(brg_springs) == 4

    gdict = asm.to_geometry_dict()
    assert gdict["bed_z_drive"] == "leadscrew_dual_belt_sync"
    assert len(gdict["z_rails"]) == 4
    assert all(r["profile_id"] == "steel_rod_10mm" for r in gdict["z_rails"])
    assert len(gdict["z_leadscrews"]) == 2

    # BOM reports 1 Z motor, 4x smooth rods + 4x LMK10LUU bearings, 2x TR8 leadscrews, 1x Z-sync belt, 3x 20T pulleys
    bom = generate_bom(spec)
    items_by_sku = {item.sku: item for item in bom.items}
    assert items_by_sku["MTR-Z"].quantity == 1
    assert items_by_sku["MOT-Z-ROD"].quantity == 4
    assert "10mm" in items_by_sku["MOT-Z-ROD"].specification
    assert items_by_sku["MOT-Z-BRG"].quantity == 4
    assert "LMK10LUU" in items_by_sku["MOT-Z-BRG"].name
    assert items_by_sku["MOT-Z-LS"].quantity == 2
    assert items_by_sku["MOT-Z-NUT"].quantity == 2
    assert items_by_sku["MOT-Z-SYNC-BELT"].quantity == 1
    assert items_by_sku["MOT-Z-SYNC-PULLEY"].quantity == 3
    assert "MOT-Z-RAIL" not in items_by_sku
    assert all(c.role != "bed" for c in bom.cut_list)

    # Full multiphysics evaluation: zero collisions and realistic community AWD frequencies
    rep = evaluate_printer(spec, include_advanced_analysis=False)
    assert not rep.collision.has_collision
    assert rep.modal_fea.fundamental_freq_hz > 95.0
    assert rep.modal_fea.shaper_x is not None and rep.modal_fea.shaper_y is not None
    assert 100.0 <= rep.modal_fea.shaper_x.effective_freq_hz <= 155.0
    assert 70.0 <= rep.modal_fea.shaper_y.effective_freq_hz <= 120.0
    assert rep.modal_fea.shaper_x.effective_freq_hz > rep.modal_fea.shaper_y.effective_freq_hz


def test_corexy_z_motor_mass_lump_counts_across_z_drives() -> None:
    """Z-motor lumped mass and BOM counts scale accurately: quad (4) > triple (3) > dual (2) > belt_sync (1) == single (1)."""
    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    base.frame.enclosed_panels = False
    base.frame.panel_mounting = "none"
    base.kinematics.z_stepper.motor_mass_g = 300.0

    expected_counts = {
        "belted_quad_z": 4,
        "leadscrew_triple": 3,
        "leadscrew_dual": 2,
        "leadscrew_dual_belt_sync": 1,
        "leadscrew_single": 1,
    }
    z_motor_total_kg: dict[str, float] = {}

    for zd, expected_n in expected_counts.items():
        s_lo = base.model_copy(deep=True)
        s_lo.bed.z_drive = zd  # type: ignore[assignment]
        s_lo.kinematics.z_stepper.motor_mass_g = 300.0

        s_hi = s_lo.model_copy(deep=True)
        s_hi.kinematics.z_stepper.motor_mass_g = 1300.0  # +1.0 kg per Z stepper

        asm_lo = build_assembly(s_lo)
        asm_hi = build_assembly(s_hi)
        delta_kg = asm_hi.total_mass_kg - asm_lo.total_mass_kg
        assert delta_kg == pytest.approx(float(expected_n), abs=1e-6), (
            f"{zd}: expected +{expected_n} kg delta for +1kg/motor, got {delta_kg:.4f} kg"
        )
        z_motor_total_kg[zd] = expected_n * 0.300

        # Also verify BOM motor count matches physical Z-motor count
        bom = generate_bom(s_lo)
        z_mtr = next(i for i in bom.items if i.sku == "MTR-Z")
        assert z_mtr.quantity == expected_n

    assert (
        z_motor_total_kg["belted_quad_z"]
        > z_motor_total_kg["leadscrew_triple"]
        > z_motor_total_kg["leadscrew_dual"]
        > z_motor_total_kg["leadscrew_dual_belt_sync"]
    )
    assert z_motor_total_kg["leadscrew_dual_belt_sync"] == z_motor_total_kg["leadscrew_single"]


# ---------------------------------------------------------------------------
# 5. FLSUN V400 Direct-Drive Delta Preset, KB Mapping, Modular Builders & CMS Mass
# ---------------------------------------------------------------------------


def test_flsun_v400_delta_preset_evaluates_near_community_touchstone() -> None:
    """FLSUN V400 delta preset evaluates cleanly with zero collisions and lands within ±20% of 68/67 Hz."""
    preset_path = EXAMPLES_DIR / "flsun_v400_delta.yaml"
    assert preset_path.exists()

    spec = PrinterSpec.from_yaml(preset_path)
    assert spec.architecture == "delta"
    assert spec.frame.width_mm == pytest.approx(460.0)
    assert spec.frame.height_mm == pytest.approx(920.0)
    assert spec.kinematics.build_volume_mm == (300.0, 300.0, 410.0)
    assert spec.kinematics.delta_arm_length_mm == pytest.approx(350.0)
    assert spec.kinematics.delta_radius_mm == pytest.approx(175.0)
    assert spec.gantry.belt.type == "gt2_9mm"
    assert spec.toolhead.extruder.type == "direct_drive"
    assert 320.0 <= (spec.toolhead.total_mass_g + spec.toolhead.hotend.mass_g) <= 360.0
    assert spec.kinematics.xy_stepper.supply_voltage_v == pytest.approx(24.0)
    assert spec.kinematics.xy_stepper.steps_per_rev == 200
    assert spec.kinematics.xy_stepper.run_current_a == pytest.approx(1.2)

    rep = evaluate_printer(spec, include_advanced_analysis=False)
    assert not rep.collision.has_collision
    assert rep.modal_fea.shaper_x is not None
    assert rep.modal_fea.shaper_y is not None

    pred_x = rep.modal_fea.shaper_x.effective_freq_hz
    pred_y = rep.modal_fea.shaper_y.effective_freq_hz
    assert abs(pred_x - 68.0) / 68.0 <= 0.20, f"FLSUN V400 X={pred_x:.1f} Hz outside ±20% of 68 Hz"
    assert abs(pred_y - 67.0) / 67.0 <= 0.20, f"FLSUN V400 Y={pred_y:.1f} Hz outside ±20% of 67 Hz"


def test_kb_flsun_v400_and_doron_velta_delta_mapping(tmp_path: Path) -> None:
    """infer_preset_hint and run_kb_calibration map FLSUN V400/SR to flsun_v400_delta and Doron Velta to delta_racer."""
    assert infer_preset_hint(architecture="delta", printer_model="FLSUN V400") == "flsun_v400_delta"
    assert infer_preset_hint(printer_model="FLSUN V400 Klipper") == "flsun_v400_delta"
    assert infer_preset_hint(architecture="delta", printer_model="FLSUN SR") == "flsun_v400_delta"
    assert infer_preset_hint(architecture="delta", printer_model="Custom Heavy Direct-Drive Delta", belt_width_mm=10.0) == "flsun_v400_delta"

    assert infer_preset_hint(architecture="delta", printer_model="Doron Velta", belt_width_mm=6.0) == "delta_racer"
    assert infer_preset_hint(printer_model="Doron Velta") == "delta_racer"
    assert infer_preset_hint(architecture="delta", printer_model="Kossel Mini") == "delta_racer"

    db_path = tmp_path / "kb_delta.db"
    with TouchstoneKB(db_path) as kb:
        seed_from_benchmark_docs(kb)
        # Inject a legacy-hinted FLSUN V400 row to verify _resolve_preset_yaml remapping
        kb.add_build(
            build_name="Legacy Seeded FLSUN V400",
            architecture="delta",
            printer_model="FLSUN V400",
            belt_width_mm=10.0,
            x_shaper_hz=68.0,
            y_shaper_hz=67.0,
            preset_hint="delta_racer",
        )
        cal = run_kb_calibration(kb)

    flsun_rows = [r for r in cal["results"] if "FLSUN" in r["build_name"]]
    assert len(flsun_rows) >= 2
    for row in flsun_rows:
        assert row["preset_file"] == "flsun_v400_delta.yaml"
        assert abs(row["x_error_pct"]) < 15.0
        assert abs(row["y_error_pct"]) < 15.0

    doron_row = next(r for r in cal["results"] if "Doron Velta" in r["build_name"])
    assert doron_row["preset_file"] == "delta_racer.yaml"


def test_builder_modular_delegation_and_custom_bracket_mass_delta() -> None:
    """builder.py delegates cleanly to _builder_bedslinger/_builder_delta and applies custom_bracket_override.mass_delta_g."""
    from printsim.geometry import _builder_bedslinger, _builder_delta, builder
    from printsim.schema import CustomBracketOverride

    # Clean modular delegation & backwards-compatible re-exports
    assert builder._build_cartesian_bedslinger is _builder_bedslinger._build_cartesian_bedslinger
    assert builder._build_delta is _builder_delta._build_delta

    bs_spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "cartesian_bedslinger.yaml")
    bs_asm = build_assembly(bs_spec)
    assert bs_asm.spec.architecture == "cartesian_bedslinger"
    assert any(m.id.startswith("z_col_") for m in bs_asm.members)

    delta_spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "flsun_v400_delta.yaml")
    delta_asm = build_assembly(delta_spec)
    assert delta_asm.spec.architecture == "delta"
    assert any(m.id.startswith("tower_") for m in delta_asm.members)

    # Custom CAD bracket mass_delta_g flows into CoreXY carriage nodes, y_moving_mass_kg, and modal dynamics
    base_spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    base_spec.gantry.custom_bracket_override = None
    asm_base = build_assembly(base_spec)
    mo_base = solve_modal_fea(asm_base)

    heavy_spec = base_spec.model_copy(deep=True)
    heavy_spec.gantry.custom_bracket_override = CustomBracketOverride(
        label="Heavy Billet XY-Joints",
        stiffness_n_per_mm=2400.0,
        mass_delta_g=120.0,
        source="cms_superelement",
    )
    asm_heavy = build_assembly(heavy_spec)
    mo_heavy = solve_modal_fea(asm_heavy)

    light_spec = base_spec.model_copy(deep=True)
    light_spec.gantry.custom_bracket_override = CustomBracketOverride(
        label="Topo-Opt CF XY-Joints",
        stiffness_n_per_mm=2400.0,
        mass_delta_g=-40.0,
        source="tet10_fea",
    )
    asm_light = build_assembly(light_spec)
    mo_light = solve_modal_fea(asm_light)

    assert asm_heavy.y_moving_mass_kg - asm_base.y_moving_mass_kg == pytest.approx(0.120, abs=1e-6)
    assert asm_heavy.y_carriage_joints_mass_kg - asm_base.y_carriage_joints_mass_kg == pytest.approx(0.120, abs=1e-6)
    assert asm_base.y_moving_mass_kg - asm_light.y_moving_mass_kg == pytest.approx(0.040, abs=1e-6)

    # Carriage nodes receive half of the mass delta each, lowering Y modal frequency when heavier at equal stiffness
    y_car_l_base = asm_base.node_map["y_car_l"].lumped_mass_kg
    y_car_l_heavy = asm_heavy.node_map["y_car_l"].lumped_mass_kg
    assert y_car_l_heavy - y_car_l_base == pytest.approx(0.060, abs=1e-6)
    assert mo_heavy.shaper_y is not None and mo_light.shaper_y is not None and mo_base.shaper_y is not None
    assert mo_heavy.shaper_y.effective_freq_hz < mo_light.shaper_y.effective_freq_hz
    assert mo_heavy.shaper_y.structural_freq_hz < mo_base.shaper_y.structural_freq_hz


# ---------------------------------------------------------------------------
# 6. Stock Trident 3-Point Bakelite Standoffs & Calibrated Monolith CNC AWD Weights
# ---------------------------------------------------------------------------


def test_trident_3pt_standoffs_geometry_and_thermal_warp_ordering() -> None:
    """trident_3pt_standoffs yields 3 mount points and thermally warps strictly between kinematic_3pt and rigid_4pt at 100 C."""
    from printsim.fea import compute_thermal_structural_report, solve_bed_plate_fem
    from printsim.schema import BED_MOUNT_CATALOG

    cfg = BED_MOUNT_CATALOG["trident_3pt_standoffs"]
    assert cfg["mount_points"] == 3
    assert cfg["k_axial_n_per_mm"] == pytest.approx(18000.0)
    assert cfg["k_lateral_n_per_mm"] == pytest.approx(9500.0)
    assert cfg["k_restraint_thermal"] == pytest.approx(22.0)
    assert cfg["damping_delta"] == pytest.approx(0.003)
    assert cfg["cost_usd"] == pytest.approx(16.0)

    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    base.bed.target_temp_c = 100.0

    spec_kin = base.model_copy(deep=True)
    spec_kin.bed.mount_type = "kinematic_3pt"

    spec_tri = base.model_copy(deep=True)
    spec_tri.bed.mount_type = "trident_3pt_standoffs"

    spec_rig = base.model_copy(deep=True)
    spec_rig.bed.mount_type = "rigid_4pt"

    asm_kin = build_assembly(spec_kin)
    asm_tri = build_assembly(spec_tri)
    asm_rig = build_assembly(spec_rig)

    # Geometry dict and bed-plate FEM both expose 3-point triangle layout
    gdict_tri = asm_tri.to_geometry_dict()
    assert gdict_tri["bed_mount_type"] == "trident_3pt_standoffs"
    assert len(gdict_tri["bed_mount_points_mm"]) == 3
    assert gdict_tri["bed_mount_points_mm"] == asm_kin.to_geometry_dict()["bed_mount_points_mm"]

    plate_kin = solve_bed_plate_fem(spec_kin)
    plate_tri = solve_bed_plate_fem(spec_tri)
    plate_rig = solve_bed_plate_fem(spec_rig)

    assert len(plate_tri.mount_points_rel) == 3
    assert plate_tri.mount_points_rel == plate_kin.mount_points_rel

    # Thermal taco buckling & thermal PTV strictly between kinematic_3pt (lowest) and rigid_4pt (highest)
    assert plate_kin.taco_buckling_um == pytest.approx(0.0, abs=1e-9)
    assert plate_kin.taco_buckling_um < plate_tri.taco_buckling_um < plate_rig.taco_buckling_um
    assert plate_kin.thermal_peak_to_valley_um < plate_tri.thermal_peak_to_valley_um < plate_rig.thermal_peak_to_valley_um

    ts_kin = compute_thermal_structural_report(asm_kin)
    ts_tri = compute_thermal_structural_report(asm_tri)
    ts_rig = compute_thermal_structural_report(asm_rig)
    assert ts_kin.bed_buckling_taco_um < ts_tri.bed_buckling_taco_um < ts_rig.bed_buckling_taco_um


def test_monolith_cnc_awd_weight_class_and_stationary_motor_mount_mass() -> None:
    """monolith_cnc_awd + aluminum_6061_t6 yields exact 256 g moving Y-carriages (CloakedWayne scale-weighed) and +0.44 kg stationary AWD billet mounts."""
    from printsim.schema import GANTRY_WEIGHT_CLASSES

    wc = GANTRY_WEIGHT_CLASSES["monolith_cnc_awd"]
    assert wc["carriage_mass_g"] == pytest.approx(214.0)
    assert wc["suggested_bracket_material"] == "aluminum_6061_t6"

    spec_mono = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_monolith_awd.yaml")
    assert spec_mono.gantry.carriage_mass_g == pytest.approx(214.0)
    assert spec_mono.gantry.bracket_material == "aluminum_6061_t6"

    asm_mono = build_assembly(spec_mono)
    assert asm_mono.y_carriage_joints_mass_kg == pytest.approx(0.256, abs=1e-9)
    assert asm_mono.to_geometry_dict()["mass_summary"]["y_carriage_joints_mass_g"] == pytest.approx(
        256.0, abs=1e-6
    )

    # Stationary CNC billet motor mounts add +0.11 kg per XY motor corner (+0.44 kg total on AWD)
    # without inflating moving Y mass beyond the +42 g carriage bracket delta
    spec_abs = spec_mono.model_copy(deep=True)
    spec_abs.gantry.bracket_material = "abs_printed"
    asm_abs = build_assembly(spec_abs)

    assert (asm_mono.y_moving_mass_kg - asm_abs.y_moving_mass_kg) == pytest.approx(0.042, abs=1e-9)
    for nid in ("g_fl", "g_fr", "g_rl", "g_rr"):
        assert (
            asm_mono.node_map[nid].lumped_mass_kg - asm_abs.node_map[nid].lumped_mass_kg
        ) == pytest.approx(0.110, abs=1e-9)
    assert (asm_mono.total_mass_kg - asm_abs.total_mass_kg) == pytest.approx(0.042 + 0.440, abs=1e-6)

    rep = evaluate_printer(spec_mono, include_advanced_analysis=False)
    assert not rep.collision.has_collision
    assert rep.modal_fea.shaper_x is not None and rep.modal_fea.shaper_y is not None
    assert rep.modal_fea.shaper_x.effective_freq_hz > rep.modal_fea.shaper_y.effective_freq_hz
