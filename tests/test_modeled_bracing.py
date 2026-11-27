"""Physics, CAD superelement, and schema verification for modeled frame bracing options."""

from __future__ import annotations

import pytest

from printsim.cad_assets.registry import get_hardware_asset_manifest, has_hardware_asset, has_visual_asset
from printsim.engine import evaluate_printer
from printsim.geometry import build_assembly, generate_bom, list_dxf_exportable_parts
from printsim.schema import FRAME_BRACING_CATALOG, PrinterSpec

ALL_MODELED_BRACINGS = (
    "none",
    "cast_corner_brackets",
    "cnc_corner_gussets",
    "rear_flat_bar_x",
    "side_flat_bar_x",
    "rear_and_side_flat_bars",
    "gussets_and_flat_bars",
)

EXPECTED_CAD_ASSETS = (
    "cast_corner_bracket_2020",
    "cast_corner_bracket_3030",
    "cast_corner_bracket_4040",
    "cnc_corner_gusset_2020",
    "cnc_corner_gusset_3030",
    "cnc_corner_gusset_4040",
)


def test_baked_frame_brace_cad_assets_valid() -> None:
    """All 6 2020/3030/4040 cast corner brackets & CNC gussets must have valid superelements and GLBs."""
    for asset_id in EXPECTED_CAD_ASSETS:
        assert has_hardware_asset("frame_braces", asset_id), f"Missing superelement: {asset_id}"
        assert has_visual_asset("frame_braces", asset_id), f"Missing visual.glb: {asset_id}"
        manifest = get_hardware_asset_manifest("frame_braces", asset_id)
        assert manifest is not None
        assert float(manifest["mass_kg"]) > 0.01
        assert float(manifest["rigid_body_max_freq_hz"]) < 0.05
        assert int(manifest["num_elements"]) >= 1000


def test_legacy_bracing_aliases_migrate_cleanly() -> None:
    """Old YAML/JSON bracing strings migrate to modeled equivalents without validation errors."""
    expected = {
        "corner_gussets": "cnc_corner_gussets",
        "top_cross": "cnc_corner_gussets",
        "rear_cross": "rear_flat_bar_x",
        "side_cross": "side_flat_bar_x",
        "rear_and_sides": "rear_and_side_flat_bars",
        "full_truss": "gussets_and_flat_bars",
    }
    for legacy, target in expected.items():
        spec = PrinterSpec(frame={"bracing": legacy})
        assert spec.frame.bracing == target


def test_frame_bracing_catalog_complete() -> None:
    for br in ALL_MODELED_BRACINGS:
        assert br in FRAME_BRACING_CATALOG


def test_corexy_modeled_bracing_monotonicity_and_cad_attachment() -> None:
    """Modeled CAD corner brackets/gussets and bolted flat bars stiffen an open CoreXY frame monotonically."""
    base = PrinterSpec.from_yaml("examples/corexy_voron_style.yaml")
    base.frame.enclosed_panels = False
    base.frame.panel_mounting = "none"

    results = {}
    for br in ALL_MODELED_BRACINGS:
        spec = base.model_copy(deep=True)
        spec.frame.bracing = br  # type: ignore[assignment]
        results[br] = evaluate_printer(spec)

    # CAD superelement attachment & mass accounting
    r_none = results["none"]
    r_cast = results["cast_corner_brackets"]
    r_cnc = results["cnc_corner_gussets"]
    r_rear = results["rear_flat_bar_x"]
    r_side = results["side_flat_bar_x"]
    r_rs = results["rear_and_side_flat_bars"]
    r_full = results["gussets_and_flat_bars"]

    assert r_none.assembly.frame_brace_cad_count == 0
    assert r_cast.assembly.frame_brace_cad_count == 24
    assert r_cast.assembly.frame_brace_cad_asset_id == "cast_corner_bracket_2020"
    assert r_cnc.assembly.frame_brace_cad_count == 16
    assert r_cnc.assembly.frame_brace_cad_asset_id == "cnc_corner_gusset_2020"
    assert r_full.assembly.frame_brace_cad_count == 16
    assert r_full.assembly.frame_brace_flat_bar_mass_kg > 0.5

    # Physical stiffness monotonicity on open CoreXY frame
    assert r_cast.static_fea.toolhead_stiffness_x_n_per_mm > r_none.static_fea.toolhead_stiffness_x_n_per_mm
    assert r_cnc.static_fea.toolhead_stiffness_x_n_per_mm > r_cast.static_fea.toolhead_stiffness_x_n_per_mm
    assert r_rear.static_fea.toolhead_stiffness_x_n_per_mm > r_none.static_fea.toolhead_stiffness_x_n_per_mm
    assert r_side.static_fea.toolhead_stiffness_y_n_per_mm > r_none.static_fea.toolhead_stiffness_y_n_per_mm
    assert r_rs.modal_fea.fundamental_freq_hz > r_none.modal_fea.fundamental_freq_hz + 15.0
    assert r_full.static_fea.toolhead_stiffness_x_n_per_mm >= r_rs.static_fea.toolhead_stiffness_x_n_per_mm
    assert r_full.static_fea.toolhead_stiffness_y_n_per_mm >= r_rs.static_fea.toolhead_stiffness_y_n_per_mm

    # Mass increases with added hardware
    assert r_cast.assembly.total_mass_kg > r_none.assembly.total_mass_kg
    assert r_cnc.assembly.total_mass_kg > r_cast.assembly.total_mass_kg
    assert r_full.assembly.total_mass_kg > r_rs.assembly.total_mass_kg


@pytest.mark.parametrize(
    "yaml_path",
    [
        "examples/cartesian_bedslinger.yaml",
        "examples/delta_racer.yaml",
    ],
)
def test_non_corexy_architectures_support_modeled_bracing(yaml_path: str) -> None:
    """Bedslinger and Delta architectures support CAD corner braces and bolted flat bars."""
    base = PrinterSpec.from_yaml(yaml_path)
    base.frame.enclosed_panels = False
    base.frame.panel_mounting = "none"
    for br in ("none", "cast_corner_brackets", "cnc_corner_gussets", "rear_flat_bar_x", "gussets_and_flat_bars"):
        s = base.model_copy(deep=True)
        s.frame.bracing = br  # type: ignore[assignment]
        report = evaluate_printer(s)
        assert report.modal_fea.fundamental_freq_hz > 10.0
        if br in ("cast_corner_brackets", "cnc_corner_gussets", "gussets_and_flat_bars"):
            assert report.assembly.frame_brace_cad_count > 0


def test_bom_and_dxf_reflect_modeled_bracing() -> None:
    spec = PrinterSpec.from_yaml("examples/corexy_voron_style.yaml")
    spec.frame.bracing = "cnc_corner_gussets"
    asm = build_assembly(spec)
    bom = generate_bom(spec)
    assert any(item.sku.startswith("FRM-BRACE-") for item in bom.items)
    dxf_parts = list_dxf_exportable_parts(spec)
    gusset_item = next(p for p in dxf_parts if p["part"] == "corner_gusset")
    assert gusset_item["applicable"] is True
