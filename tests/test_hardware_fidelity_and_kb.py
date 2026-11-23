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
    assert abs(rr300["x_error_pct"]) < 15.0
    assert abs(rr300["y_error_pct"]) < 15.0
