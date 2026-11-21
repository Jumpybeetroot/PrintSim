"""Verification tests for PrintSim v0.8 AI Agent Discovery Pack (MCP tools, bracket designer & discovery demo)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import trimesh
import yaml

from printsim.fea.bracket_designer import design_and_evaluate_bracket
from printsim.mcp_server import dispatch_tool, handle_jsonrpc_message
from printsim.schema import PrinterSpec

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "examples"


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


def test_mcp_server_metadata_and_enriched_list_catalog() -> None:
    """Verify v0.9.0 MCP metadata, 14-tool schema, and enriched list_catalog fields."""
    init_resp = handle_jsonrpc_message({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert init_resp is not None
    assert init_resp["result"]["serverInfo"]["name"] == "printsim-mcp"
    assert init_resp["result"]["serverInfo"]["version"] == "0.9.0"

    tools_resp = handle_jsonrpc_message({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert tools_resp is not None
    tools = tools_resp["result"]["tools"]
    assert len(tools) == 14
    names = {t["name"] for t in tools}
    assert {
        "list_catalog",
        "get_preset_yaml",
        "evaluate_printer_yaml",
        "mutate_and_compare_printer",
        "design_and_test_bracket",
        "optimize_printer_yaml",
        "optimize_printer_pareto",
        "generate_printer_bom",
        "run_solid_part_fea",
        "run_tet_part_fea",
        "optimize_part_topology",
        "export_printer_stl",
        "export_printer_step",
        "run_deep_frame_verify",
    } == names

    cat = _call_mcp("list_catalog")
    expected_keys = {
        "materials",
        "profiles",
        "filaments",
        "steppers",
        "toolhead_presets",
        "extrusion_fillings",
        "gantry_brackets",
        "x_rail_mounts",
        "y_rail_mounts",
        "part_presets",
        "presets",
    }
    assert expected_keys <= set(cat.keys())
    assert "ppa_cf_printed" in cat["materials"]
    assert "extrusion_3030" in cat["profiles"]
    assert "abs" in cat["filaments"]
    assert "ldo_2504ac" in cat["steppers"]
    assert "archetype_cpap" in cat["toolhead_presets"]

    filling_map = {f["id"]: f for f in cat["extrusion_fillings"]}
    assert "epoxy_granite" in filling_map and len(filling_map["epoxy_granite"]["notes"]) > 5

    bkt_map = {b["id"]: b for b in cat["gantry_brackets"]}
    assert "ppa_cf_printed" in bkt_map
    assert bkt_map["ppa_cf_printed"]["youngs_modulus_gpa"] > bkt_map["abs_printed"]["youngs_modulus_gpa"]
    assert bkt_map["ppa_cf_printed"]["creep_tension_limit_n"] > bkt_map["abs_printed"]["creep_tension_limit_n"]

    assert set(cat["x_rail_mounts"]) == {"front", "top", "bottom", "dual_top_bottom"}
    assert set(cat["y_rail_mounts"]) == {"top", "inner", "bottom"}
    expected_presets = {p.stem for p in EXAMPLES_DIR.glob("*.yaml")}
    assert expected_presets <= set(cat["presets"])


def test_get_preset_yaml_and_toolhead_rel_pos_probe() -> None:
    """Verify get_preset_yaml and optional toolhead_rel_pos probing in evaluate_printer_yaml."""
    preset = _call_mcp("get_preset_yaml", {"preset_id": "voron_trident_300"})
    assert preset["preset_id"] == "voron_trident_300"
    assert preset["architecture"] == "corexy"
    assert "Voron Trident 300" in preset["name"]
    spec = PrinterSpec.model_validate(yaml.safe_load(preset["yaml_text"]))
    assert spec.architecture == "corexy"

    # Unknown preset returns JSON-RPC tool error cleanly
    err_resp = handle_jsonrpc_message(
        {
            "jsonrpc": "2.0",
            "id": 9,
            "method": "tools/call",
            "params": {"name": "get_preset_yaml", "arguments": {"preset_id": "nonexistent_printer"}},
        }
    )
    assert err_resp is not None
    assert err_resp["result"]["isError"] is True

    rep_low_z = _call_mcp(
        "evaluate_printer_yaml",
        {"yaml_text": preset["yaml_text"], "toolhead_rel_pos": [0.5, 0.5, 0.05]},
    )
    rep_high_z = _call_mcp(
        "evaluate_printer_yaml",
        {"yaml_text": preset["yaml_text"], "toolhead_rel_pos": [0.5, 0.5, 0.95]},
    )
    assert rep_low_z["scores"]["composite"] > 0
    assert rep_high_z["scores"]["composite"] > 0
    assert (
        rep_low_z["static_fea"]["combined_max_deflection_um"]
        != rep_high_z["static_fea"]["combined_max_deflection_um"]
    )


def test_mutate_and_compare_printer_overrides_ranking_and_deltas() -> None:
    """Verify dot-path overrides (including custom_members), ranking modes, deltas, and best_yaml."""
    base_yaml = (EXAMPLES_DIR / "voron_trident_300.yaml").read_text(encoding="utf-8")

    candidates = [
        {
            "name": "Epoxy Granite + PPA-CF",
            "overrides": {
                "frame.extrusion_filling": "epoxy_granite",
                "gantry.bracket_material": "ppa_cf_printed",
            },
        },
        {
            "name": "Bottom X-Rail + Low CoG",
            "overrides": {
                "gantry.x_rail_mount": "bottom",
                "toolhead.cog_offset_mm": [0.0, -10.0, -12.0],
            },
        },
        {
            "name": "4mm Bolted Al5052 Panels",
            "overrides": {
                "frame.enclosed_panels": True,
                "frame.panel_mounting": "bolted_structural",
                "frame.panel_material": "aluminum_5052_sheet",
                "frame.panel_thickness_mm": 4.0,
            },
        },
        {
            "name": "Custom Rear Diagonal Brace",
            "overrides": {
                "frame.custom_nodes": [
                    {"id": "c_mid_l", "position_mm": [0.0, 460.0, 250.0]},
                    {"id": "c_mid_r", "position_mm": [460.0, 460.0, 250.0]},
                ],
                "frame.custom_members": [
                    {
                        "id": "custom_x_tie",
                        "start_node": "c_mid_l",
                        "end_node": "c_mid_r",
                        "profile": "extrusion_2020",
                        "material": "aluminum_6063_t5",
                        "role": "brace",
                    }
                ],
            },
        },
    ]

    res = _call_mcp(
        "mutate_and_compare_printer",
        {
            "yaml_text": base_yaml,
            "candidates": candidates,
            "rank_by": "composite",
            "filament_id": "abs",
            "toolhead_rel_pos": [0.5, 0.5, 0.5],
        },
    )

    base = res["baseline"]
    required_summary_keys = {
        "composite_score",
        "fundamental_freq_hz",
        "recommended_max_accel_mm_s2",
        "nozzle_deflection_um",
        "toolhead_stiffness_x_n_per_mm",
        "toolhead_stiffness_y_n_per_mm",
        "total_mass_kg",
        "estimated_cost_usd",
        "diagnostics_count",
    }
    assert required_summary_keys <= set(base.keys())

    ranked = res["ranked_candidates"]
    assert len(ranked) == 4
    assert [c["rank"] for c in ranked] == [1, 2, 3, 4]
    assert res["best_candidate_name"] == ranked[0]["name"]

    # Ranked descending by composite_score
    scores = [c["composite_score"] for c in ranked]
    assert scores == sorted(scores, reverse=True)

    # Exact delta accounting against baseline
    for c in ranked:
        assert c["delta_score"] == pytest.approx(c["composite_score"] - base["composite_score"], abs=0.05)
        assert c["delta_freq_hz"] == pytest.approx(c["fundamental_freq_hz"] - base["fundamental_freq_hz"], abs=0.05)
        assert c["delta_accel_mm_s2"] == pytest.approx(
            c["recommended_max_accel_mm_s2"] - base["recommended_max_accel_mm_s2"], abs=0.2
        )
        assert c["delta_deflection_um"] == pytest.approx(
            c["nozzle_deflection_um"] - base["nozzle_deflection_um"], abs=0.05
        )
        assert c["delta_cost_usd"] == pytest.approx(c["estimated_cost_usd"] - base["estimated_cost_usd"], abs=0.05)

    # best_yaml must validate cleanly back into PrinterSpec
    best_spec = PrinterSpec.model_validate(yaml.safe_load(res["best_yaml"]))
    assert best_spec.architecture == "corexy"

    # Budget ranking puts lowest-cost candidate #1
    res_budget = dispatch_tool(
        "mutate_and_compare_printer",
        {"yaml_text": base_yaml, "candidates": candidates[:3], "rank_by": "budget"},
    )
    costs = [c["estimated_cost_usd"] for c in res_budget["ranked_candidates"]]
    assert costs == sorted(costs)

    # Stiffness ranking puts lowest deflection candidate #1
    res_stiff = dispatch_tool(
        "mutate_and_compare_printer",
        {"yaml_text": base_yaml, "candidates": candidates[:3], "rank_by": "stiffness"},
    )
    defls = [c["nozzle_deflection_um"] for c in res_stiff["ranked_candidates"]]
    assert defls == sorted(defls)


@pytest.mark.parametrize("bracket_type", ["corner_gusset", "motor_mount", "xy_joint_l_bracket"])
def test_design_and_test_bracket_monotonicity_and_materials(bracket_type: str) -> None:
    """Verify wall_thickness, rib_thickness, and PPA-CF > ABS physical monotonicity across all brackets."""
    # 1. Thicker wall strictly increases stiffness & reduces displacement
    r_thin = _call_mcp(
        "design_and_test_bracket",
        {"bracket_type": bracket_type, "wall_thickness_mm": 3.5, "rib_thickness_mm": 4.0},
    )
    r_med = _call_mcp(
        "design_and_test_bracket",
        {"bracket_type": bracket_type, "wall_thickness_mm": 6.0, "rib_thickness_mm": 4.0},
    )
    r_thick = _call_mcp(
        "design_and_test_bracket",
        {"bracket_type": bracket_type, "wall_thickness_mm": 10.0, "rib_thickness_mm": 4.0},
    )
    assert r_thin["stiffness_n_per_mm"] < r_med["stiffness_n_per_mm"] < r_thick["stiffness_n_per_mm"]
    assert r_thin["max_displacement_um"] > r_med["max_displacement_um"] > r_thick["max_displacement_um"]

    # 2. Adding a gusset rib (rib > 0 vs 0) strictly increases stiffness & reduces displacement
    r_no_rib = _call_mcp(
        "design_and_test_bracket",
        {"bracket_type": bracket_type, "wall_thickness_mm": 6.0, "rib_thickness_mm": 0.0},
    )
    r_rib_4 = _call_mcp(
        "design_and_test_bracket",
        {"bracket_type": bracket_type, "wall_thickness_mm": 6.0, "rib_thickness_mm": 4.0},
    )
    r_rib_8 = _call_mcp(
        "design_and_test_bracket",
        {"bracket_type": bracket_type, "wall_thickness_mm": 6.0, "rib_thickness_mm": 8.0},
    )
    assert r_no_rib["stiffness_n_per_mm"] < r_rib_4["stiffness_n_per_mm"] < r_rib_8["stiffness_n_per_mm"]
    assert r_no_rib["max_displacement_um"] > r_rib_4["max_displacement_um"] > r_rib_8["max_displacement_um"]

    # 3. Material hierarchy: ppa_cf_printed > abs_printed
    r_abs = design_and_evaluate_bracket(bracket_type=bracket_type, material_id="abs_printed")  # type: ignore[arg-type]
    r_ppa = design_and_evaluate_bracket(bracket_type=bracket_type, material_id="ppa_cf_printed")  # type: ignore[arg-type]
    assert r_ppa["stiffness_n_per_mm"] > r_abs["stiffness_n_per_mm"] * 4.0
    assert r_ppa["specific_stiffness_n_per_mm_per_g"] > r_abs["specific_stiffness_n_per_mm_per_g"] * 3.0
    assert r_ppa["fdm_anisotropic"] is True


def test_design_and_test_bracket_exports_stl_and_step_and_tet10(tmp_path: Path) -> None:
    """Verify Hex8 STL export (watertight), Tet10 Gmsh/build123d engine, and STEP CAD export."""
    stl_hex = tmp_path / "gusset_hex8.stl"
    res_hex = _call_mcp(
        "design_and_test_bracket",
        {
            "bracket_type": "corner_gusset",
            "arm_length_mm": 45.0,
            "height_mm": 22.0,
            "wall_thickness_mm": 6.0,
            "rib_thickness_mm": 6.0,
            "material_id": "ppa_cf_printed",
            "engine": "hex8",
            "output_stl_path": str(stl_hex),
        },
    )
    assert res_hex["engine"] == "hex8"
    assert res_hex["exported_stl_path"] == str(stl_hex.resolve())
    assert res_hex["exported_step_path"] is None
    assert stl_hex.exists() and stl_hex.stat().st_size > 200
    mesh_hex = trimesh.load(str(stl_hex), force="mesh")
    assert mesh_hex.is_watertight

    stl_tet = tmp_path / "gusset_tet10.stl"
    step_tet = tmp_path / "gusset_tet10.step"
    res_tet = _call_mcp(
        "design_and_test_bracket",
        {
            "bracket_type": "corner_gusset",
            "arm_length_mm": 45.0,
            "height_mm": 22.0,
            "wall_thickness_mm": 6.0,
            "rib_thickness_mm": 6.0,
            "material_id": "ppa_cf_printed",
            "engine": "tet10",
            "output_stl_path": str(stl_tet),
            "output_step_path": str(step_tet),
        },
    )
    assert res_tet["engine"] == "tet10"
    assert res_tet["exported_stl_path"] == str(stl_tet.resolve())
    assert res_tet["exported_step_path"] == str(step_tet.resolve())
    assert step_tet.exists() and b"ISO-10303-21" in step_tet.read_bytes()
    assert stl_tet.exists() and stl_tet.stat().st_size > 200
    assert res_tet["stiffness_n_per_mm"] > 100.0
    assert res_tet["safety_factor"] > 1.0

    # Validation check on out-of-range parameter
    with pytest.raises(ValueError, match="arm_length_mm"):
        design_and_evaluate_bracket(arm_length_mm=5.0)


def test_agent_discovery_loop_script_runs_cleanly() -> None:
    """Verify examples/agent_discovery_loop.py executes end-to-end with zero errors."""
    script_path = PROJECT_ROOT / "examples" / "agent_discovery_loop.py"
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Stage 1: Printer Architecture Trade Study" in proc.stdout
    assert "Stage 2: Parametric Corner-Gusset Bracket FEA Sweep" in proc.stdout
    assert "Winning Specific Stiffness" in proc.stdout
