"""Regression & physics verification tests for WV-Design Trinity 1.6 toolhead & Dragon Ace Volcano hotend."""

from __future__ import annotations

from pathlib import Path

import pytest

from printsim.cad_assets.recipes import verify_baked_asset
from printsim.engine import evaluate_printer
from printsim.geometry._builder_toolhead import resolve_toolhead_hardware_accessories
from printsim.schema import HOTEND_CATALOG, TOOLHEAD_PRESETS, PrinterSpec, get_item_confidence

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def _make_trinity_spec(
    preset_id: str,
    hotend_id: str,
    *,
    use_brace: bool = True,
    body_material: str = "ppa_cf_printed",
) -> PrinterSpec:
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_awd_300.yaml")
    tp = TOOLHEAD_PRESETS[preset_id]
    spec.toolhead.preset_id = preset_id
    spec.toolhead.name = tp["name"]
    spec.toolhead.total_mass_g = float(tp["total_mass_g"])
    spec.toolhead.cog_offset_mm = tuple(tp["cog_offset_mm"])
    spec.toolhead.nozzle_offset_mm = tuple(tp["nozzle_offset_mm"])
    spec.toolhead.mount_material = body_material
    spec.toolhead.body_material = body_material
    spec.toolhead.carriage_mount_material = body_material
    spec.toolhead.hotend_mount_material = body_material
    spec.toolhead.body_construction = tp["body_construction"]
    spec.toolhead.belt_plane_z_offset_mm = float(tp["belt_plane_z_offset_mm"])
    spec.toolhead.cpap_blower_id = "none"
    spec.toolhead.use_hotend_bottom_brace = use_brace
    spec.toolhead.extruder.type = "direct_drive"
    spec.toolhead.extruder.gear_ratio = 6.25
    spec.toolhead.hotend.preset_id = hotend_id
    spec.toolhead.hotend.part_cooling_cfm = 20.0
    spec.use_cad_superelements = True
    return spec


def test_trinity_and_dav_catalog_and_confidence_entries() -> None:
    assert "trianglelab_dragon_ace_volcano" in HOTEND_CATALOG
    dav = HOTEND_CATALOG["trianglelab_dragon_ace_volcano"]
    assert dav["overall_length_mm"] == pytest.approx(59.4, abs=0.2)
    assert dav["melt_zone_length_mm"] == pytest.approx(34.0, abs=1.0)
    assert dav["mass_g"] == pytest.approx(45.0, abs=1.0)
    assert dav["heater_power_w"] == pytest.approx(70.0, abs=1.0)
    assert dav["structural_stiffness_n_per_mm"] == pytest.approx(842.0, abs=5.0)

    assert "trinity_1_6_dav" in TOOLHEAD_PRESETS
    assert "trinity_1_6_chube" in TOOLHEAD_PRESETS
    assert TOOLHEAD_PRESETS["trinity_1_6_dav"]["default_hotend_preset_id"] == "trianglelab_dragon_ace_volcano"
    assert TOOLHEAD_PRESETS["trinity_1_6_chube"]["default_hotend_preset_id"] == "chube_compact"
    assert TOOLHEAD_PRESETS["trinity_1_6_dav"]["extruder_gear_ratio"] == pytest.approx(6.25)
    assert TOOLHEAD_PRESETS["trinity_1_6_dav"]["part_cooling_cfm"] == pytest.approx(20.0)

    assert get_item_confidence("hotends", "trianglelab_dragon_ace_volcano")["trust_pct"] >= 85
    assert get_item_confidence("toolhead_presets", "trinity_1_6_dav")["trust_pct"] >= 85
    assert get_item_confidence("toolhead_presets", "trinity_1_6_chube")["trust_pct"] >= 85


@pytest.mark.parametrize(
    ("category", "asset_id"),
    [
        ("hotends", "trianglelab_dragon_ace_volcano"),
        ("assemblies", "trinity_1_6_dav__trianglelab_dragon_ace_volcano"),
        ("assemblies", "trinity_1_6_chube__chube_compact"),
        ("toolheads", "trinity_1_6_dav"),
        ("toolheads", "trinity_1_6_chube"),
    ],
)
def test_trinity_and_dav_baked_cad_qa(category: str, asset_id: str) -> None:
    qa = verify_baked_asset(category, asset_id)
    assert qa.passed, f"QA failed for {category}/{asset_id}: {qa.to_dict()}"


@pytest.mark.parametrize(
    ("preset_id", "hotend_id", "expected_assembly"),
    [
        ("trinity_1_6_dav", "trianglelab_dragon_ace_volcano", "trinity_1_6_dav__trianglelab_dragon_ace_volcano"),
        ("trinity_1_6_chube", "chube_compact", "trinity_1_6_chube__chube_compact"),
    ],
)
def test_trinity_unibody_brace_is_monotonic_and_zero_mass_delta(
    preset_id: str,
    hotend_id: str,
    expected_assembly: str,
) -> None:
    spec_braced = _make_trinity_spec(preset_id, hotend_id, use_brace=True)
    spec_unbraced = _make_trinity_spec(preset_id, hotend_id, use_brace=False)

    # Integral unibody brace has zero accessory mass delta (unlike Microwaved's bolt-on 1.67 g Ti brace)
    acc_on = resolve_toolhead_hardware_accessories(spec_braced)
    acc_off = resolve_toolhead_hardware_accessories(spec_unbraced)
    assert acc_on["use_hotend_bottom_brace"] is True
    assert acc_off["use_hotend_bottom_brace"] is False
    assert acc_on["accessory_mass_delta_g"] == 0.0
    assert acc_off["accessory_mass_delta_g"] == 0.0

    rep_braced = evaluate_printer(spec_braced)
    rep_unbraced = evaluate_printer(spec_unbraced)

    th_braced = rep_braced.assembly.to_geometry_dict()["toolhead_structure"]
    th_unbraced = rep_unbraced.assembly.to_geometry_dict()["toolhead_structure"]

    assert th_braced["cad_hotend_brace_available"] is True
    assert th_braced["cad_hotend_braced"] is True
    assert th_unbraced["cad_hotend_brace_available"] is True
    assert th_unbraced["cad_hotend_braced"] is False

    se_ids_braced = {se["asset_id"] for se in th_braced["cad_superelements_active"]}
    assert preset_id in se_ids_braced
    assert expected_assembly in se_ids_braced

    # Bracing the hotend inside Trinity's unibody must strictly stiffen hotend & lower static nozzle deflection
    assert th_braced["stage_stiffnesses_n_per_mm"]["hotend_total_n_per_mm"] > (
        th_unbraced["stage_stiffnesses_n_per_mm"]["hotend_total_n_per_mm"] * 1.15
    )
    assert rep_braced.static_fea.toolhead_stiffness_x_n_per_mm > (
        rep_unbraced.static_fea.toolhead_stiffness_x_n_per_mm
    )
    assert rep_braced.static_fea.toolhead_stiffness_y_n_per_mm > (
        rep_unbraced.static_fea.toolhead_stiffness_y_n_per_mm
    )
    assert rep_braced.static_fea.accel_deflection_x_um < rep_unbraced.static_fea.accel_deflection_x_um
    assert rep_braced.static_fea.accel_deflection_y_um < rep_unbraced.static_fea.accel_deflection_y_um
    assert rep_braced.modal_fea.nozzle_pitch_twist_freq_hz >= (
        rep_unbraced.modal_fea.nozzle_pitch_twist_freq_hz - 1e-3
    )


def test_trinity_dav_thermal_and_unibody_material_scaling() -> None:
    spec_ppa = _make_trinity_spec("trinity_1_6_dav", "trianglelab_dragon_ace_volcano", body_material="ppa_cf_printed")
    spec_slm = _make_trinity_spec("trinity_1_6_dav", "trianglelab_dragon_ace_volcano", body_material="slm_aluminum")

    rep_ppa = evaluate_printer(spec_ppa)
    rep_slm = evaluate_printer(spec_slm)

    # Dragon Ace Volcano 34 mm melt zone + 70 W heater should deliver realistic high-flow ABS/PLA rates (>25 mm^3/s)
    assert rep_ppa.thermal.max_volumetric_flow_mm3_s >= 25.0

    # Monolithic unibody SLM metal upgrade (1.0x vs 0.55x FDM * E_ratio) must be substantially stiffer
    k_body_ppa = rep_ppa.assembly.to_geometry_dict()["toolhead_structure"]["stage_stiffnesses_n_per_mm"]["body_n_per_mm"]
    k_body_slm = rep_slm.assembly.to_geometry_dict()["toolhead_structure"]["stage_stiffnesses_n_per_mm"]["body_n_per_mm"]
    assert k_body_slm > k_body_ppa * 4.0
