"""Unit tests for PrintSim v0.3 Manufacturable Cut-List & BOM Generator."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest

from printsim.geometry import (
    BOMLineItem,
    CutListItem,
    PrinterBOM,
    StockNestingSummary,
    build_assembly,
    generate_bom,
)
from printsim.schema import PrinterSpec

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def _load_example(filename: str) -> PrinterSpec:
    return PrinterSpec.from_yaml(EXAMPLES_DIR / filename)


class TestCoreXYBOM:
    """Verify cut-list merging, enclosure panels, hardware, and pricing on CoreXY."""

    def test_collinear_merging_and_no_virtual_panel_shear(self) -> None:
        spec = _load_example("corexy_voron_style.yaml")
        bom = generate_bom(spec)

        assert isinstance(bom, PrinterBOM)
        assert all(isinstance(c, CutListItem) for c in bom.cut_list)
        assert all(isinstance(i, BOMLineItem) for i in bom.items)
        assert bom.architecture == "corexy"
        assert bom.total_cost_usd > 0
        assert bom.total_mass_kg > 0
        assert bom.total_extrusion_length_m > 0
        assert bom.total_fasteners > 0

        # 4 vertical columns merged to exact frame.height_mm (500.0 mm), NOT 8 split halves
        col_cuts = [
            c
            for c in bom.cut_list
            if c.role == "frame"
            and c.profile_id == spec.frame.profile
            and c.length_mm == pytest.approx(spec.frame.height_mm, abs=0.1)
        ]
        assert len(col_cuts) == 1
        assert col_cuts[0].quantity == 4

        # Ensure no split column halves (445 mm or 55 mm) appear in cut list
        lengths = [c.length_mm for c in bom.cut_list]
        assert all(abs(L - 445.0) > 1.0 and abs(L - 55.0) > 1.0 for L in lengths)

        # X gantry merged from x_gantry_l + x_gantry_r to exact frame.width_mm (460.0 mm)
        gantry_cuts = [c for c in bom.cut_list if c.role == "gantry"]
        assert len(gantry_cuts) == 1
        assert gantry_cuts[0].quantity == 1
        assert gantry_cuts[0].length_mm == pytest.approx(spec.frame.width_mm, abs=0.1)

        # Y rail support extrusions merged to exact frame.depth_mm (460.0 mm, qty=2)
        rail_cuts = [c for c in bom.cut_list if c.role == "rail"]
        assert len(rail_cuts) == 1
        assert rail_cuts[0].quantity == 2
        assert rail_cuts[0].length_mm == pytest.approx(spec.frame.depth_mm, abs=0.1)

        # No virtual panel_shear_* beams leak into cut list:
        # corner_gussets has only 1 physical brace (b_rl -> g_rr, ~640.0 mm),
        # whereas panel_shear_* diagonals would be sqrt(460^2 + 500^2) = 679.4 mm
        brace_cuts = [c for c in bom.cut_list if c.role == "brace"]
        assert sum(c.quantity for c in brace_cuts) == 1
        assert all(abs(c.length_mm - 679.4) > 2.0 for c in brace_cuts)

    def test_enclosure_panels_toggle(self) -> None:
        spec = _load_example("corexy_voron_style.yaml")
        assert spec.frame.enclosed_panels is True
        bom_enclosed = generate_bom(spec)

        enc_items = [i for i in bom_enclosed.items if i.category == "enclosure"]
        assert len(enc_items) == 5
        assert sum(i.quantity for i in enc_items) == 5
        assert "enclosure" in bom_enclosed.category_costs_usd
        names_joined = " ".join(i.name.lower() for i in enc_items)
        for panel_label in ("left", "right", "rear", "top", "front"):
            assert panel_label in names_joined

        # Disable enclosed panels -> enclosure category disappears
        spec_open = spec.model_copy(deep=True)
        spec_open.frame.enclosed_panels = False
        bom_open = generate_bom(spec_open)
        assert [i for i in bom_open.items if i.category == "enclosure"] == []
        assert "enclosure" not in bom_open.category_costs_usd
        assert bom_enclosed.total_cost_usd > bom_open.total_cost_usd

    def test_corexy_belt_motors_and_dual_x_rails(self) -> None:
        spec = _load_example("corexy_voron_style.yaml")
        bom = generate_bom(spec)

        # CoreXY XY belt length formula: 2 * (2*(W + D) + 400) / 1000
        expected_xy_belt_m = 2.0 * (2.0 * (spec.frame.width_mm + spec.frame.depth_mm) + 400.0) / 1000.0
        xy_belt_item = next(i for i in bom.items if i.sku == "MOT-BELT-XY")
        assert xy_belt_item.quantity == pytest.approx(expected_xy_belt_m, abs=0.01)

        # 2 XY + 4 Z (belted_quad_z) + 1 extruder = 7 motors
        motor_qty = sum(i.quantity for i in bom.items if i.category == "motors")
        assert motor_qty == 7

        # Dual X rails adds 2nd X linear rail to motion BOM without polluting cut list
        spec_dual = spec.model_copy(deep=True)
        spec_dual.gantry.dual_x_rails = True
        bom_dual = generate_bom(spec_dual)
        x_rail_item = next(i for i in bom_dual.items if i.sku == "MOT-X-RAIL")
        assert x_rail_item.quantity == 2
        assert all(c.profile_id != spec_dual.gantry.x_rail_profile for c in bom_dual.cut_list)


class TestCartesianAndDeltaBOM:
    """Verify Cartesian bedslinger and Delta architectures."""

    def test_cartesian_bedslinger_bom(self) -> None:
        spec = _load_example("cartesian_bedslinger.yaml")
        bom = generate_bom(spec)

        assert bom.architecture == "cartesian_bedslinger"
        assert bom.total_cost_usd > 0
        assert bom.total_mass_kg > 0
        assert len(bom.cut_list) >= 3

        # Z uprights merged to frame.height_mm (420 mm, qty=2)
        z_uprights = [
            c
            for c in bom.cut_list
            if c.role == "frame" and c.length_mm == pytest.approx(spec.frame.height_mm, abs=0.1)
        ]
        assert len(z_uprights) == 1
        assert z_uprights[0].quantity == 2

        # Base Y rails merged to frame.depth_mm (380 mm, qty=2)
        y_base = [
            c
            for c in bom.cut_list
            if c.role == "frame" and c.length_mm == pytest.approx(spec.frame.depth_mm, abs=0.1)
        ]
        assert len(y_base) == 1
        assert y_base[0].quantity == 2

        # X gantry merged to frame.width_mm (360 mm, qty=1)
        gantry = [c for c in bom.cut_list if c.role == "gantry"]
        assert len(gantry) == 1
        assert gantry[0].quantity == 1
        assert gantry[0].length_mm == pytest.approx(spec.frame.width_mm, abs=0.1)

        # Belt length formula: (2*W + 150 + 2*D + 150) / 1000
        expected_belt_m = (2.0 * spec.frame.width_mm + 150.0 + 2.0 * spec.frame.depth_mm + 150.0) / 1000.0
        assert bom.total_belt_length_m == pytest.approx(expected_belt_m, abs=0.01)

        # Motors: 2 XY + 2 Z (leadscrew_dual) + 1 extruder = 5
        motor_qty = sum(i.quantity for i in bom.items if i.category == "motors")
        assert motor_qty == 5

    def test_delta_racer_bom(self) -> None:
        spec = _load_example("delta_racer.yaml")
        bom = generate_bom(spec)

        assert bom.architecture == "delta"
        assert bom.total_cost_usd > 0
        assert bom.total_mass_kg > 0

        # 3 vertical towers merged to frame.height_mm (780.0 mm)
        towers = [c for c in bom.cut_list if c.role == "tower"]
        assert len(towers) == 1
        assert towers[0].quantity == 3
        assert towers[0].length_mm == pytest.approx(spec.frame.height_mm, abs=0.1)

        # Belt length formula: 3 * (2*H + 120) / 1000
        expected_belt_m = 3.0 * (2.0 * spec.frame.height_mm + 120.0) / 1000.0
        assert bom.total_belt_length_m == pytest.approx(expected_belt_m, abs=0.01)

        # 6x carbon delta rods
        delta_rods = next(i for i in bom.items if i.sku == "MOT-DELTA-RODS")
        assert delta_rods.quantity == 6
        assert f"{spec.kinematics.delta_arm_length_mm:.1f}" in delta_rods.specification

        # Motors: 3 tower + 1 extruder = 4
        motor_qty = sum(i.quantity for i in bom.items if i.category == "motors")
        assert motor_qty == 4


class TestStockNestingMath:
    """Verify 1D First-Fit Decreasing stock bar nesting invariants."""

    @pytest.mark.parametrize(
        "yaml_file",
        [
            "corexy_voron_style.yaml",
            "cartesian_bedslinger.yaml",
            "delta_racer.yaml",
        ],
    )
    def test_nesting_invariants_and_monotonicity(self, yaml_file: str) -> None:
        spec = _load_example(yaml_file)
        bom_1000 = generate_bom(spec, stock_length_mm=1000.0, kerf_mm=3.0)
        bom_1500 = generate_bom(spec, stock_length_mm=1500.0, kerf_mm=3.0)
        bom_2000 = generate_bom(spec, stock_length_mm=2000.0, kerf_mm=3.0)

        for bom in (bom_1000, bom_1500, bom_2000):
            assert len(bom.stock_nesting) > 0
            for nest in bom.stock_nesting:
                assert isinstance(nest, StockNestingSummary)
                assert nest.bars_required == len(nest.bar_cuts_mm)
                assert nest.bars_required >= 1
                assert nest.bars_required * nest.stock_length_mm >= nest.used_length_mm - 1e-6
                assert 0.0 <= nest.waste_pct < 100.0
                assert nest.used_length_mm + nest.waste_length_mm == pytest.approx(
                    nest.bars_required * nest.stock_length_mm, abs=0.5
                )

        # Increasing stock length never increases bars_required for identical cuts
        bars_1000 = {(s.profile_id, s.material_id): s.bars_required for s in bom_1000.stock_nesting}
        bars_1500 = {(s.profile_id, s.material_id): s.bars_required for s in bom_1500.stock_nesting}
        bars_2000 = {(s.profile_id, s.material_id): s.bars_required for s in bom_2000.stock_nesting}

        for key in bars_1000:
            assert bars_2000[key] <= bars_1500[key] <= bars_1000[key]

    def test_oversized_cut_bumps_stock_length(self) -> None:
        spec = _load_example("corexy_voron_style.yaml")
        spec.frame.height_mm = 1350.0  # exceeds 1000 mm default stock
        bom = generate_bom(spec, stock_length_mm=1000.0, kerf_mm=3.0)

        ext_nest = next(s for s in bom.stock_nesting if s.profile_id == spec.frame.profile)
        assert ext_nest.stock_length_mm == 2000.0
        assert ext_nest.bars_required * ext_nest.stock_length_mm >= ext_nest.used_length_mm

    def test_kerf_effect_on_exact_half_cuts(self) -> None:
        spec = _load_example("corexy_voron_style.yaml")
        # Make all frame dimensions exactly 500 mm: with 3 mm kerf, two 500 mm cuts
        # require 1003 mm (> 1000 mm stock), whereas with 0 mm kerf they fit in 1 bar.
        spec.frame.width_mm = 500.0
        spec.frame.depth_mm = 500.0
        spec.frame.height_mm = 500.0
        bom_kerf = generate_bom(spec, stock_length_mm=1000.0, kerf_mm=3.0)
        bom_nokerf = generate_bom(spec, stock_length_mm=1000.0, kerf_mm=0.0)

        nest_kerf = next(s for s in bom_kerf.stock_nesting if s.profile_id == "extrusion_2020")
        nest_nokerf = next(s for s in bom_nokerf.stock_nesting if s.profile_id == "extrusion_2020")
        assert nest_nokerf.bars_required < nest_kerf.bars_required
        assert nest_nokerf.waste_pct == pytest.approx(0.0, abs=1e-6)


class TestSerializationAndConsistency:
    """Verify to_dict, to_csv, to_markdown, cost sums, and AssemblyModel input."""

    @pytest.mark.parametrize(
        "yaml_file",
        [
            "corexy_voron_style.yaml",
            "cartesian_bedslinger.yaml",
            "delta_racer.yaml",
        ],
    )
    def test_exports_and_cost_consistency(self, yaml_file: str) -> None:
        spec = _load_example(yaml_file)
        bom = generate_bom(spec)

        # Category cost sum matches total cost
        assert sum(bom.category_costs_usd.values()) == pytest.approx(bom.total_cost_usd, abs=0.1)

        # AssemblyModel input matches PrinterSpec input
        bom_from_asm = generate_bom(build_assembly(spec))
        assert bom_from_asm.to_dict() == bom.to_dict()

        # to_dict() is strictly JSON serializable
        data = bom.to_dict()
        encoded = json.dumps(data)
        decoded = json.loads(encoded)
        assert decoded["printer_name"] == spec.name
        assert decoded["architecture"] == spec.architecture
        assert len(decoded["cut_list"]) == len(bom.cut_list)
        assert len(decoded["stock_nesting"]) == len(bom.stock_nesting)
        assert len(decoded["items"]) == len(bom.items)

        # to_csv() is valid RFC-4180 CSV with both CUT_LIST and BOM sections
        csv_text = bom.to_csv()
        rows = list(csv.DictReader(io.StringIO(csv_text)))
        assert len(rows) == len(bom.cut_list) + len(bom.items)
        sections = {r["section"] for r in rows}
        assert sections == {"CUT_LIST", "BOM"}

        # to_markdown() produces valid structured GFM report
        md_text = bom.to_markdown()
        assert f"# Bill of Materials & Cut List — {spec.name}" in md_text
        assert "## Structural Cut List" in md_text
        assert "## 1D Stock Bar Nesting" in md_text
        assert "## Complete Hardware BOM" in md_text
