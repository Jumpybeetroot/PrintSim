"""Tests for upgraded optimizer mutations, 12-gene Pareto frontier, and MCP server tools."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from printsim.engine import optimize_design, optimize_pareto_front
from printsim.engine import optimizer as optimizer_mod
from printsim.engine import pareto as pareto_mod
from printsim.mcp_server import handle_jsonrpc_message
from printsim.schema import BED_MOUNT_CATALOG, FRAME_FEET_CATALOG, PrinterSpec

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def _make_weak_corexy_spec() -> PrinterSpec:
    """Create a deliberately weak CoreXY printer (ABS mount, die springs, single-Z, open frame)."""
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    spec.name = "Weak CoreXY Test Rig"
    spec.frame.enclosed_panels = False
    spec.frame.panel_mounting = "none"
    spec.frame.bracing = "none"
    spec.gantry.bracket_material = "abs_printed"
    spec.toolhead.mount_material = "abs_printed"
    spec.toolhead.internal_stiffness_n_per_mm = 380.0
    spec.bed.z_drive = "leadscrew_single"
    spec.bed.mount_type = "springs_4pt"
    return spec


def _call_mcp(tool_name: str, arguments: dict | None = None, req_id: int = 1) -> dict:
    resp = handle_jsonrpc_message(
        {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments or {}},
        }
    )
    assert resp is not None
    assert resp["result"]["isError"] is False, resp["result"]["content"][0]["text"]
    return json.loads(resp["result"]["content"][0]["text"])


def test_optimizer_explores_new_levers_and_improves_weak_printer() -> None:
    """optimize_design on a weak printer explores new levers and improves objective score."""
    weak = _make_weak_corexy_spec()

    # 1. Verify _mutate_spec reaches all new CoreXY mutation dimensions with valid stiffness coupling
    rng = random.Random(123)
    seen_prefixes: set[str] = set()
    for _ in range(90):
        cand, changes = optimizer_mod._mutate_spec(weak, rng)
        assert len(changes) >= 1
        assert cand.toolhead.internal_stiffness_n_per_mm == pytest.approx(
            optimizer_mod.TOOLHEAD_MOUNT_STIFFNESS_N_PER_MM[cand.toolhead.mount_material],
            rel=1e-6,
        )
        if cand.gantry.drive_mode == "awd":
            assert cand.gantry.double_shear_pulleys is True
        if cand.is_flying_gantry and cand.frame.panel_mounting == "bolted_structural":
            assert cand.frame.bolt_panels_to_gantry is False
        for ch in changes:
            seen_prefixes.add(ch.split("=")[0])

    expected_prefixes = {
        "toolhead.mount",
        "bed.mount_type",
        "frame.panels",
        "gantry.bracket_material",
        "frame.extrusion_filling",
        "gantry.drive_mode",
        "bed.z_drive",
    }
    assert expected_prefixes <= seen_prefixes

    # Non-CoreXY printers must never mutate drive_mode or bed_z_drive
    cart = PrinterSpec.from_yaml(EXAMPLES_DIR / "cartesian_bedslinger.yaml")
    rng_cart = random.Random(99)
    for _ in range(40):
        c_cart, ch_cart = optimizer_mod._mutate_spec(cart, rng_cart)
        assert c_cart.gantry.drive_mode == "2wd"
        assert c_cart.bed.z_drive == "leadscrew_dual"
        assert not any(c.startswith(("gantry.drive_mode=", "bed.z_drive=")) for c in ch_cart)

    # 2. Full optimize_design loop improves weak printer & stays deterministic
    res1 = optimize_design(weak, iterations=20, objective="balanced", seed=42)
    res2 = optimize_design(weak, iterations=20, objective="balanced", seed=42)
    assert res1.best_objective > res1.baseline_objective
    assert res1.improvement_delta > 0.0
    assert res1.best_objective == pytest.approx(res2.best_objective)
    assert res1.best_report.scores.composite > res1.history[0].composite_score

    explored_all = " | ".join(ch for c in res1.history for ch in c.changes)
    assert "toolhead.mount=" in explored_all
    assert "bed.mount_type=" in explored_all
    assert "gantry.bracket_material=" in explored_all
    assert "frame.panels=" in explored_all


def test_pareto_12_gene_encode_decode_and_nondominated_front() -> None:
    """Pareto encodes/decodes all 12 genes (including mounts & brackets) and finds valid front."""
    assert pareto_mod._N_VARS == 12
    assert len(pareto_mod._XL) == 12
    assert len(pareto_mod._XU) == 12

    base = _make_weak_corexy_spec()
    base_vec = pareto_mod._encode_baseline_vector(base)
    assert len(base_vec) == 12
    # Weak spec starts at abs_printed (0), springs_4pt (0), abs_printed (0)
    assert base_vec[9] == 0
    assert base_vec[10] == 0
    assert base_vec[11] == 0

    # Round-trip decode with upgraded mounts & brackets emits descriptive change labels
    upgraded_vec = list(base_vec)
    upgraded_vec[9] = pareto_mod.TOOLHEAD_MOUNT_OPTIONS.index("aluminum_6061_t6")
    upgraded_vec[10] = pareto_mod.BED_MOUNT_OPTIONS.index("kinematic_3pt")
    upgraded_vec[11] = pareto_mod.BRACKET_OPTIONS.index("ppa_cf_printed")

    dec_spec, dec_changes = pareto_mod._decode_candidate(base, tuple(upgraded_vec))
    assert dec_spec.toolhead.mount_material == "aluminum_6061_t6"
    assert dec_spec.toolhead.internal_stiffness_n_per_mm == pytest.approx(1850.0)
    assert dec_spec.bed.mount_type == "kinematic_3pt"
    assert dec_spec.gantry.bracket_material == "ppa_cf_printed"
    assert "toolhead mount -> aluminum_6061_t6" in dec_changes
    assert "bed mount -> kinematic_3pt" in dec_changes
    assert "brackets -> ppa_cf_printed" in dec_changes

    # Intermediate materials outside the 3 discrete options still encode cleanly within bounds
    mid_spec = base.model_copy(deep=True)
    mid_spec.toolhead.mount_material = "pa12_cf_printed"
    mid_spec.gantry.bracket_material = "aluminum_7075_t6"
    mid_vec = pareto_mod._encode_baseline_vector(mid_spec)
    assert mid_vec[9] == 1
    assert mid_vec[11] == 2
    assert all(0 <= mid_vec[i] <= int(pareto_mod._XU[i]) for i in range(pareto_mod._N_VARS))

    # Full NSGA-II Pareto run on weak spec produces non-dominated upgrades
    res = optimize_pareto_front(base, population_size=14, generations=4, seed=42)
    front = res.pareto_front
    assert len(front) >= 3
    assert [p.total_cost_usd for p in front] == sorted(p.total_cost_usd for p in front)

    for i, a in enumerate(front):
        assert a.primary_problem_subsystem in {"toolhead", "gantry", "frame", "bed"}
        for j, b in enumerate(front):
            if i == j:
                continue
            a_dom_b = (
                a.total_cost_usd <= b.total_cost_usd
                and a.safe_max_accel_mm_s2 >= b.safe_max_accel_mm_s2
                and a.nozzle_deflection_um <= b.nozzle_deflection_um
                and (
                    a.total_cost_usd < b.total_cost_usd
                    or a.safe_max_accel_mm_s2 > b.safe_max_accel_mm_s2
                    or a.nozzle_deflection_um < b.nozzle_deflection_um
                )
            )
            assert not a_dom_b

    all_changes = [chg for pt in front for chg in pt.changes]
    assert any(c.startswith("toolhead mount -> ") for c in all_changes)
    assert any(c.startswith("bed mount -> ") for c in all_changes)
    assert any(c.startswith("brackets -> ") for c in all_changes)

    payload = res.to_dict(include_specs=False)
    assert payload["baseline"]["primary_problem_subsystem"] in {"toolhead", "gantry", "frame", "bed"}


def test_mcp_catalog_evaluate_and_sensitivity_updates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify MCP list_catalog, evaluate_printer_yaml options/summaries, and analyze_sensitivity_hotspots."""
    # 1. list_catalog exposes bed_mounts and frame_feet matching schema catalogs
    cat = _call_mcp("list_catalog")
    assert len(cat["bed_mounts"]) == len(BED_MOUNT_CATALOG)
    assert {m["id"] for m in cat["bed_mounts"]} == set(BED_MOUNT_CATALOG.keys())
    assert len(cat["frame_feet"]) == len(FRAME_FEET_CATALOG)
    assert {f["id"] for f in cat["frame_feet"]} == set(FRAME_FEET_CATALOG.keys())

    # 2. evaluate_printer_yaml accepts panel_mesh_n, toolhead_rel_pos, accel_override_mm_s2
    trident = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    trident.frame.enclosed_panels = True
    trident.frame.panel_mounting = "bolted_structural"
    trident_yaml = trident.to_yaml()

    eval_m2 = _call_mcp(
        "evaluate_printer_yaml",
        {
            "yaml_text": trident_yaml,
            "panel_mesh_n": 2,
            "toolhead_rel_pos": [0.5, 0.5, 0.5],
            "accel_override_mm_s2": 6000.0,
        },
    )
    eval_m3 = _call_mcp(
        "evaluate_printer_yaml",
        {
            "yaml_text": trident_yaml,
            "panel_mesh_n": 3,
            "toolhead_rel_pos": [0.25, 0.75, 0.3],
            "accel_override_mm_s2": 12000.0,
        },
    )
    assert eval_m2["geometry"]["panel_mesh_n"] == 2
    assert eval_m3["geometry"]["panel_mesh_n"] == 3
    assert len(eval_m3["geometry"]["shell_panels"]) > len(eval_m2["geometry"]["shell_panels"])
    assert eval_m3["static_fea"]["peak_inertial_force_x_n"] == pytest.approx(
        2.0 * eval_m2["static_fea"]["peak_inertial_force_x_n"], rel=1e-4
    )

    # Modal & Sensitivity clarity fields surfaced cleanly
    assert eval_m2["modal_fea"]["worst_print_mode_number"] >= 1
    top_mode = eval_m2["modal_fea"]["modes"][0]
    for key in (
        "print_impact_score",
        "print_impact_level",
        "nozzle_whip_ratio",
        "toolhead_flop_pct",
        "subsystem_share_pct",
    ):
        assert key in top_mode

    sub_bd = eval_m2["sensitivity"]["subsystem_breakdown"]
    assert set(sub_bd.keys()) == {"toolhead", "gantry", "frame", "bed"}
    assert eval_m2["sensitivity"]["primary_problem_subsystem"] in {"toolhead", "gantry", "frame", "bed"}

    summary = eval_m2["summary"]
    assert summary["worst_print_mode_number"] == eval_m2["modal_fea"]["worst_print_mode_number"]
    assert summary["primary_problem_subsystem"] == eval_m2["sensitivity"]["primary_problem_subsystem"]
    assert summary["subsystem_breakdown"] == sub_bd
    assert len(summary["top_modes"]) >= 1
    assert "print_impact_score" in summary["top_modes"][0]

    # Verify bed_plate_fea summary is surfaced cleanly when present, and handled defensively when None
    if isinstance(eval_m2.get("bed_plate_fea"), dict):
        bp_sum = summary["bed_plate_fea"]
        assert bp_sum["total_flatness_error_um"] == eval_m2["bed_plate_fea"]["total_flatness_error_um"]
        assert bp_sum["taco_buckling_um"] == eval_m2["bed_plate_fea"]["taco_buckling_um"]
        assert bp_sum["plate_first_mode_hz"] == eval_m2["bed_plate_fea"]["plate_first_mode_hz"]
        assert bp_sum["first_layer_risk"] == eval_m2["bed_plate_fea"]["first_layer_risk"]

    from printsim.engine.evaluator import SimulationReport

    orig_to_dict = SimulationReport.to_dict

    def _without_bed_plate(self: SimulationReport, include_toolpath_segments: bool = True) -> dict:
        out = orig_to_dict(self, include_toolpath_segments=include_toolpath_segments)
        out.pop("bed_plate_fea", None)
        return out

    monkeypatch.setattr(SimulationReport, "to_dict", _without_bed_plate)
    eval_no_bp = _call_mcp("evaluate_printer_yaml", {"yaml_text": trident_yaml})
    assert "bed_plate_fea" not in eval_no_bp["summary"]

    # 3. analyze_sensitivity_hotspots surfaces subsystem_breakdown, primary_problem_subsystem & worst_print_mode_number
    hotspots = _call_mcp(
        "analyze_sensitivity_hotspots",
        {"yaml_text": trident_yaml, "toolhead_rel_pos": [0.5, 0.5, 0.5]},
    )
    assert hotspots["primary_problem_subsystem"] in {"toolhead", "gantry", "frame", "bed"}
    assert set(hotspots["subsystem_breakdown"].keys()) == {"toolhead", "gantry", "frame", "bed"}
    assert hotspots["worst_print_mode_number"] >= 1
    assert hotspots["worst_print_mode"]["mode_number"] == hotspots["worst_print_mode_number"]
    assert "print_impact_score" in hotspots["worst_print_mode"]
    assert len(hotspots["top_stiffen_targets"]) >= 1
    assert len(hotspots["top_lighten_targets"]) >= 1
