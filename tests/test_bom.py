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

        # 4 vertical columns merged to exact outer frame.height_mm (480.0 mm on V2.4 300), NOT 8 split halves
        col_cuts = [
            c
            for c in bom.cut_list
            if c.role == "frame"
            and c.profile_id == spec.frame.profile
            and c.length_mm == pytest.approx(spec.frame.height_mm, abs=0.1)
        ]
        assert len(col_cuts) == 1
        assert col_cuts[0].quantity == 4

        # Ensure no split column halves appear in cut list
        lengths = [c.length_mm for c in bom.cut_list]
        assert all(abs(L - 445.0) > 1.0 and abs(L - 55.0) > 1.0 for L in lengths)

        # X gantry merged from x_gantry_l + x_gantry_r to exact structural span between CAD XY-joint
        # ports (350 mm on Voron Stock +/-45 mm ports) & net 380 mm official Voron 300 BOM cut length
        from printsim.geometry.builder import build_assembly, corexy_frame_centerlines_m

        asm_v = build_assembly(spec)
        expected_x_span_mm = (
            sum(
                m.length_m(asm_v.node_map)
                for m in asm_v.members
                if m.id in ("x_gantry_l", "x_gantry_r")
            )
            * 1000.0
        )
        w_cl_mm, d_cl_mm, _ = [v * 1000.0 for v in corexy_frame_centerlines_m(spec)]
        gantry_cuts = [c for c in bom.cut_list if c.role == "gantry"]
        assert len(gantry_cuts) == 1
        assert gantry_cuts[0].quantity == 1
        assert gantry_cuts[0].length_mm == pytest.approx(expected_x_span_mm, abs=0.1)
        assert gantry_cuts[0].length_mm < w_cl_mm
        assert gantry_cuts[0].net_length_mm == pytest.approx(380.0, abs=0.1)

        # Y rail support extrusions merged to centerline span (440 mm, qty=2) & net 420 mm Voron 300 cut
        rail_cuts = [c for c in bom.cut_list if c.role == "rail"]
        assert len(rail_cuts) == 1
        assert rail_cuts[0].quantity == 2
        assert rail_cuts[0].length_mm == pytest.approx(d_cl_mm, abs=0.1)
        assert rail_cuts[0].net_length_mm == pytest.approx(420.0, abs=0.1)

        # No virtual panel_shear_* beams leak into cut list:
        # corner_gussets uses corner joint stiffening (0 diagonal rods), while rear_cross adds 2 rear + 1 top diagonal
        brace_cuts = [c for c in bom.cut_list if c.role == "brace"]
        assert sum(c.quantity for c in brace_cuts) == 0
        spec_rc = spec.model_copy(deep=True)
        spec_rc.frame.bracing = "rear_cross"
        rc_brace_cuts = [c for c in generate_bom(spec_rc).cut_list if c.role == "brace"]
        assert sum(c.quantity for c in rc_brace_cuts) == 3

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


class TestAWDAndStructuralPanelsBOM:
    """Verify v0.5 CoreXY AWD, double-shear pulley kits, and structural panel BOM."""

    def test_awd_corexy_monolith_vs_2wd_trident(self) -> None:
        trident_spec = _load_example("voron_trident_300.yaml")
        monolith_spec = _load_example("voron_monolith_awd.yaml")

        bom_trident = generate_bom(trident_spec)
        bom_monolith = generate_bom(monolith_spec)

        # Stock Voron Trident 300 is 2WD without double-shear kit, clip-in acrylic panels
        trident_xy_motors = next(i for i in bom_trident.items if i.sku == "MTR-XY")
        trident_pulleys = next(i for i in bom_trident.items if i.sku == "MOT-PULLEY-DRV")
        trident_idlers = next(i for i in bom_trident.items if i.sku == "MOT-IDLER")
        assert trident_xy_motors.quantity == 2
        assert trident_pulleys.quantity == 2
        assert trident_idlers.quantity == 8
        assert all(i.sku != "MOT-DBL-SHEAR" for i in bom_trident.items)

        # Voron Monolith AWD 300 has 4 XY steppers, 4 drive pulleys, 12 idlers, 4 double-shear sets
        monolith_xy_motors = next(i for i in bom_monolith.items if i.sku == "MTR-XY")
        monolith_pulleys = next(i for i in bom_monolith.items if i.sku == "MOT-PULLEY-DRV")
        monolith_idlers = next(i for i in bom_monolith.items if i.sku == "MOT-IDLER")
        monolith_ds = next(i for i in bom_monolith.items if i.sku == "MOT-DBL-SHEAR")
        monolith_belt = next(i for i in bom_monolith.items if i.sku == "MOT-BELT-XY")

        assert monolith_xy_motors.quantity == 4
        assert monolith_pulleys.quantity == 4
        assert monolith_idlers.quantity == 12
        assert monolith_ds.category == "motion"
        assert monolith_ds.quantity == 4
        assert monolith_ds.total_cost_usd > 0
        assert monolith_ds.total_mass_g > 0

        expected_awd_belt_m = (
            2.0 * (2.0 * (monolith_spec.frame.width_mm + monolith_spec.frame.depth_mm) + 500.0)
            / 1000.0
        )
        assert monolith_belt.quantity == pytest.approx(expected_awd_belt_m, abs=0.01)
        assert "AWD" in monolith_belt.specification
        assert "GT2_9MM" in monolith_belt.name and "fiberglass" in monolith_belt.name

        # 4 XY + 3 Z (leadscrew_triple) + 1 extruder = 8 motors
        monolith_motor_qty = sum(i.quantity for i in bom_monolith.items if i.category == "motors")
        assert monolith_motor_qty == 8

        # Monolith AWD with 48V motors, 9mm fiberglass belts, double-shear, CF beam & Al panels costs more
        assert bom_monolith.total_cost_usd > bom_trident.total_cost_usd
        assert bom_monolith.category_costs_usd["motors"] > bom_trident.category_costs_usd["motors"]
        assert bom_monolith.category_costs_usd["motion"] > bom_trident.category_costs_usd["motion"]

        # Enabling double-shear on 2WD Trident adds 2 sets
        trident_ds_spec = trident_spec.model_copy(deep=True)
        trident_ds_spec.gantry.double_shear_pulleys = True
        bom_trident_ds = generate_bom(trident_ds_spec)
        ds_2wd = next(i for i in bom_trident_ds.items if i.sku == "MOT-DBL-SHEAR")
        assert ds_2wd.quantity == 2
        assert bom_trident_ds.total_cost_usd > bom_trident.total_cost_usd

    def test_structural_panel_materials_thickness_and_fasteners(self) -> None:
        base = _load_example("voron_trident_300.yaml")

        # 1) 3mm Cast Acrylic (clip_enclosure)
        spec_acrylic = base.model_copy(deep=True)
        spec_acrylic.frame.enclosed_panels = True
        spec_acrylic.frame.panel_mounting = "clip_enclosure"
        spec_acrylic.frame.panel_material = "acrylic_pmma"
        spec_acrylic.frame.panel_thickness_mm = 3.0

        # 2) 4mm ACM Dibond (bolted_structural)
        spec_acm = base.model_copy(deep=True)
        spec_acm.frame.enclosed_panels = True
        spec_acm.frame.panel_mounting = "bolted_structural"
        spec_acm.frame.panel_material = "acm_dibond"
        spec_acm.frame.panel_thickness_mm = 4.0

        # 3) 4mm Carbon Fiber 3K Sheet (bolted_structural)
        spec_cf = base.model_copy(deep=True)
        spec_cf.frame.enclosed_panels = True
        spec_cf.frame.panel_mounting = "bolted_structural"
        spec_cf.frame.panel_material = "carbon_fiber_plate"
        spec_cf.frame.panel_thickness_mm = 4.0

        bom_acrylic = generate_bom(spec_acrylic)
        bom_acm = generate_bom(spec_acm)
        bom_cf = generate_bom(spec_cf)

        def _enc_totals(bom: PrinterBOM) -> tuple[float, float]:
            enc = [i for i in bom.items if i.category == "enclosure"]
            return sum(i.total_cost_usd for i in enc), sum(i.total_mass_g for i in enc)

        cost_acrylic, mass_acrylic = _enc_totals(bom_acrylic)
        cost_acm, mass_acm = _enc_totals(bom_acm)
        cost_cf, mass_cf = _enc_totals(bom_cf)

        # Monotonic increase in panel cost and mass: 3mm acrylic < 4mm ACM < 4mm CF sheet
        assert cost_acrylic < cost_acm < cost_cf
        assert mass_acrylic < mass_acm < mass_cf

        # Bolted structural panels on fixed-gantry Trident add +48 perimeter + 24 gantry bolts vs +16 clip (+56 delta)
        assert bom_acm.total_fasteners == bom_acrylic.total_fasteners + 56
        assert any(i.sku == "FST-PANEL-STRUCT" and i.quantity == 48 for i in bom_acm.items)
        gantry_fasteners = next(i for i in bom_acm.items if i.sku == "FST-PANEL-GANTRY")
        assert gantry_fasteners.category == "fasteners"
        assert gantry_fasteners.quantity == 24
        assert gantry_fasteners.total_cost_usd == pytest.approx(24 * 0.22, abs=0.01)
        assert gantry_fasteners.total_mass_g == pytest.approx(24 * 4.5, abs=0.1)
        assert any(i.sku == "FST-PANEL-CLIP" and i.quantity == 16 for i in bom_acrylic.items)

        # Verify Dibond sandwich areal mass (~4.75 kg/m2 for 4mm ACM) is used for ENC-LEFT
        left_acm = next(i for i in bom_acm.items if i.sku == "ENC-LEFT")
        expected_left_acm_g = 0.46 * 0.50 * 4.748 * 1000.0
        assert left_acm.total_mass_g == pytest.approx(expected_left_acm_g, rel=1e-2)

        # Disabling bolt_panels_to_gantry OR switching to flying gantry (belted_quad_z) omits FST-PANEL-GANTRY
        spec_acm_nobolt = spec_acm.model_copy(deep=True)
        spec_acm_nobolt.frame.bolt_panels_to_gantry = False
        bom_acm_nobolt = generate_bom(spec_acm_nobolt)
        assert all(i.sku != "FST-PANEL-GANTRY" for i in bom_acm_nobolt.items)
        assert bom_acm_nobolt.total_fasteners == bom_acrylic.total_fasteners + 32

        spec_acm_flying = spec_acm.model_copy(deep=True)
        spec_acm_flying.bed.z_drive = "belted_quad_z"
        spec_acm_flying.frame.bolt_panels_to_gantry = True
        bom_acm_flying = generate_bom(spec_acm_flying)
        assert all(i.sku != "FST-PANEL-GANTRY" for i in bom_acm_flying.items)
        spec_acrylic_flying = spec_acrylic.model_copy(deep=True)
        spec_acrylic_flying.bed.z_drive = "belted_quad_z"
        bom_acrylic_flying = generate_bom(spec_acrylic_flying)
        assert bom_acm_flying.total_fasteners == bom_acrylic_flying.total_fasteners + 32

        # Specifications explicitly state material name, thickness, mounting type, and cut dimensions
        left_acrylic = next(i for i in bom_acrylic.items if i.sku == "ENC-LEFT")
        left_cf = next(i for i in bom_cf.items if i.sku == "ENC-LEFT")

        assert "Cast Acrylic Sheet (PMMA)" in left_acrylic.specification
        assert "3.0 mm" in left_acrylic.specification
        assert "Clip-In Enclosure" in left_acrylic.specification
        assert "460x500 mm" in left_acrylic.specification

        assert "ACM / Dibond" in left_acm.specification
        assert "4.0 mm" in left_acm.specification
        assert "Bolted Structural" in left_acm.specification
        assert "460x500 mm" in left_acm.specification

        assert "Carbon Fiber 3K Structural Sheet" in left_cf.specification
        assert "4.0 mm" in left_cf.specification
        assert "Bolted Structural" in left_cf.specification

        # panel_mounting == "none" disables enclosure panels even if enclosed_panels is True
        spec_none = base.model_copy(deep=True)
        spec_none.frame.enclosed_panels = True
        spec_none.frame.panel_mounting = "none"
        bom_none = generate_bom(spec_none)
        assert [i for i in bom_none.items if i.category == "enclosure"] == []
        assert "enclosure" not in bom_none.category_costs_usd
        assert bom_none.total_fasteners < bom_acrylic.total_fasteners


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

        # Base Y rails merged to centerline depth (frame.depth_mm - profile_width, qty=2)
        from printsim.schema import get_profile

        fr_w_mm = get_profile(spec.frame.profile).width_m * 1000.0
        up_w_mm = get_profile(spec.frame.effective_upright_profile).width_m * 1000.0
        y_base = [
            c
            for c in bom.cut_list
            if c.role == "frame" and c.length_mm == pytest.approx(spec.frame.depth_mm - fr_w_mm, abs=0.1)
        ]
        assert len(y_base) == 1
        assert y_base[0].quantity == 2

        # X gantry merged to centerline width (frame.width_mm - upright_width, qty=1)
        gantry = [c for c in bom.cut_list if c.role == "gantry"]
        assert len(gantry) == 1
        assert gantry[0].quantity == 1
        assert gantry[0].length_mm == pytest.approx(spec.frame.width_mm - up_w_mm, abs=0.1)

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
            "voron_trident_300.yaml",
            "voron_monolith_awd.yaml",
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
        # Make all 2020 cut lengths exactly 500 mm (520 mm outer W/D -> 500 mm centerlines, 500 mm outer H):
        # with 3 mm kerf, two 500 mm cuts require 1003 mm (> 1000 mm stock), whereas with 0 mm kerf they fit in 1 bar.
        spec.frame.width_mm = 520.0
        spec.frame.depth_mm = 520.0
        spec.frame.height_mm = 500.0
        bom_kerf = generate_bom(spec, stock_length_mm=1000.0, kerf_mm=3.0)
        bom_nokerf = generate_bom(spec, stock_length_mm=1000.0, kerf_mm=0.0)

        nest_kerf = next(s for s in bom_kerf.stock_nesting if s.profile_id == "extrusion_2020")
        nest_nokerf = next(s for s in bom_nokerf.stock_nesting if s.profile_id == "extrusion_2020")
        # 17 cuts of 500 mm: with 3 mm kerf needs 17 bars (1 per bar); with 0 mm kerf fits in 9 bars (2 per bar, 1 half-bar left)
        assert nest_nokerf.bars_required < nest_kerf.bars_required
        assert nest_nokerf.bars_required == (len(nest_kerf.bar_cuts_mm) + 1) // 2
        assert nest_nokerf.waste_pct < nest_kerf.waste_pct


class TestSerializationAndConsistency:
    """Verify to_dict, to_csv, to_markdown, cost sums, and AssemblyModel input."""

    @pytest.mark.parametrize(
        "yaml_file",
        [
            "corexy_voron_style.yaml",
            "voron_trident_300.yaml",
            "voron_monolith_awd.yaml",
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

    def test_outer_frame_edge_to_edge_matches_real_voron_cut_lists(self) -> None:
        """Outer frame.width/depth/height map cleanly to inset beam centerlines & real Voron/DoomCube BOM cuts."""
        cases = [
            # (yaml_file, outer_w_d_h, centerline_w_d_h, vert_cut_mm, horiz_net_mm, xbeam_net_mm)
            ("corexy_voron_style.yaml", [460.0, 460.0, 480.0], [440.0, 440.0, 460.0], 480.0, 420.0, 380.0),
            ("voron_trident_300.yaml", [460.0, 460.0, 500.0], [440.0, 440.0, 480.0], 500.0, 420.0, 380.0),
            ("doomcube_4040.yaml", [500.0, 500.0, 500.0], [460.0, 460.0, 480.0], 500.0, 420.0, 380.0),
        ]
        for yaml_file, exp_outer, exp_cl, exp_vert, exp_horiz, exp_xbeam in cases:
            spec = _load_example(yaml_file)
            asm = build_assembly(spec)
            gd = asm.to_geometry_dict()
            assert gd["frame_outer_dims_mm"] == pytest.approx(exp_outer, abs=0.1)
            assert gd["frame_dims_mm"] == pytest.approx(exp_cl, abs=0.1)

            bom = generate_bom(spec, clip_corner_overlaps=True)
            vert_cuts = [
                c for c in bom.cut_list
                if c.role == "frame" and c.profile_id == spec.frame.effective_upright_profile
                and c.length_mm == pytest.approx(exp_vert, abs=0.1)
            ]
            assert len(vert_cuts) == 1 and vert_cuts[0].quantity == 4

            rail_cuts = [c for c in bom.cut_list if c.role == "rail"]
            assert len(rail_cuts) == 1
            assert rail_cuts[0].length_mm == pytest.approx(exp_horiz, abs=0.1)

            gantry_cuts = [c for c in bom.cut_list if c.role == "gantry"]
            assert len(gantry_cuts) == 1
            assert gantry_cuts[0].length_mm == pytest.approx(exp_xbeam, abs=0.1)

        # to_markdown() produces valid structured GFM report
        md_text = bom.to_markdown()
        assert f"# Bill of Materials & Cut List — {spec.name}" in md_text
        assert "## Structural Cut List" in md_text
        assert "## 1D Stock Bar Nesting" in md_text
        assert "## Complete Hardware BOM" in md_text
