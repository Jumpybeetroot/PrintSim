"""Native stdio MCP (Model Context Protocol) JSON-RPC 2.0 server for AI design agents."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Literal

import yaml

from printsim.api.session import (
    enqueue_command,
    get_live_session,
    request_snapshot_capture,
)
from printsim.engine import evaluate_printer, optimize_design, optimize_pareto_front
from printsim.engine.evaluator import verify_printer_deep
from printsim.fea.adxl_calibration import calibrate_printer_against_adxl, fit_printer_to_adxl
from printsim.fea.bracket_designer import design_and_evaluate_bracket
from printsim.fea.solid_fem import solve_preset_part_fea, solve_solid_voxel_fea, voxelize_stl_file
from printsim.fea.superelement import condense_parametric_bracket
from printsim.fea.tet_fem import solve_preset_tet_fea, solve_step_tet_fea
from printsim.fea.topology import export_topology_stl, optimize_preset_topology
from printsim.geometry import build_assembly, export_assembly_step, export_assembly_stl, generate_bom
from printsim.schema import (
    BED_MOUNT_CATALOG,
    EXTRUSION_FILLINGS,
    FRAME_FEET_CATALOG,
    GANTRY_BRACKET_CATALOG,
    HOTEND_CATALOG,
    MATERIAL_CATALOG,
    PROFILE_CATALOG,
    STEPPER_CATALOG,
    TOOLHEAD_PRESETS,
    Z_DRIVE_MECHANISM_CATALOG,
    Z_RAIL_PROFILES,
    PrinterSpec,
)
from printsim.thermal import FILAMENT_CATALOG

SERVER_INFO = {"name": "printsim-mcp", "version": "0.9.0"}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = PROJECT_ROOT / "examples"

_VALID_RANK_BY = ("composite", "safe_accel", "stiffness", "budget")


def _list_preset_ids() -> list[str]:
    if not EXAMPLES_DIR.exists():
        return []
    return sorted(p.stem for p in EXAMPLES_DIR.glob("*.yaml"))


def _apply_dot_overrides(base_data: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy of ``base_data`` with dot-path ``overrides`` applied."""
    out = copy.deepcopy(base_data)
    for raw_key, value in overrides.items():
        key = str(raw_key).strip()
        parts = [p for p in key.split(".") if p]
        if not parts:
            raise ValueError(f"Invalid override key: {raw_key!r}")

        # Convenience preset expansion when an agent sets stepper or toolhead by catalog ID
        if key == "kinematics.xy_stepper" and isinstance(value, str):
            if value not in STEPPER_CATALOG:
                raise ValueError(f"Unknown stepper preset '{value}'.")
            cur_st = dict(out.get("kinematics", {}).get("xy_stepper", {}))
            cur_st.update(STEPPER_CATALOG[value])
            out.setdefault("kinematics", {})["xy_stepper"] = cur_st
            continue
        if key == "toolhead" and isinstance(value, str):
            if value not in TOOLHEAD_PRESETS:
                raise ValueError(f"Unknown toolhead preset '{value}'.")
            cur_th = dict(out.get("toolhead", {}))
            cur_th.update(TOOLHEAD_PRESETS[value])
            out["toolhead"] = cur_th
            continue
        if key in ("toolhead.hotend", "toolhead.hotend.preset_id") and isinstance(value, str):
            if value not in HOTEND_CATALOG:
                raise ValueError(f"Unknown hotend preset '{value}'.")
            cur_th = dict(out.get("toolhead", {}))
            cur_he = dict(cur_th.get("hotend", {}))
            cur_he.update(HOTEND_CATALOG[value])
            cur_he["preset_id"] = value
            cur_th["hotend"] = cur_he
            out["toolhead"] = cur_th
            continue

        cur: Any = out
        for part in parts[:-1]:
            if not isinstance(cur, dict):
                raise ValueError(f"Cannot traverse into non-dict segment '{part}' in '{raw_key}'")
            if part not in cur or cur[part] is None:
                cur[part] = {}
            cur = cur[part]
        if not isinstance(cur, dict):
            raise ValueError(f"Cannot set key '{parts[-1]}' on non-dict target in '{raw_key}'")
        cur[parts[-1]] = copy.deepcopy(value)
    return out


def _mode_impact_summary(mode: dict[str, Any]) -> dict[str, Any]:
    """Return a compact print-impact summary dictionary for a serialized mode shape."""
    return {
        "mode_number": mode.get("mode_number"),
        "frequency_hz": mode.get("frequency_hz"),
        "mode_label": mode.get("mode_label"),
        "print_impact_score": mode.get("print_impact_score"),
        "print_impact_level": mode.get("print_impact_level"),
        "nozzle_whip_ratio": mode.get("nozzle_whip_ratio"),
        "toolhead_flop_pct": mode.get("toolhead_flop_pct"),
        "subsystem_share_pct": mode.get("subsystem_share_pct"),
    }


def _summarize_printer(
    spec: PrinterSpec,
    filament_id: str = "pla",
    toolhead_rel_pos: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> dict[str, Any]:
    report = evaluate_printer(
        spec,
        toolhead_rel_pos=toolhead_rel_pos,
        filament_id=filament_id,
    )
    bom = generate_bom(report.assembly)
    real_diags = [
        d
        for d in report.diagnostics
        if not d.startswith("Design passes all structural, kinematic, thermal")
    ]
    return {
        "composite_score": round(float(report.scores.composite), 2),
        "fundamental_freq_hz": round(float(report.modal_fea.fundamental_freq_hz), 2),
        "recommended_max_accel_mm_s2": round(float(report.modal_fea.recommended_max_accel_mm_s2), 1),
        "nozzle_deflection_um": round(float(report.static_fea.combined_max_deflection_um), 2),
        "toolhead_stiffness_x_n_per_mm": round(float(report.static_fea.toolhead_stiffness_x_n_per_mm), 2),
        "toolhead_stiffness_y_n_per_mm": round(float(report.static_fea.toolhead_stiffness_y_n_per_mm), 2),
        "total_mass_kg": round(float(bom.total_mass_kg), 3),
        "estimated_cost_usd": round(float(bom.total_cost_usd), 2),
        "diagnostics_count": len(real_diags),
    }


TOOLS_SCHEMA = [
    {
        "name": "list_catalog",
        "description": (
            "List all available engineering materials, structural profiles, filaments, steppers, "
            "toolhead presets, extrusion fillings, gantry bracket materials, bed mounts, frame feet, "
            "rail mount orientations, solid part presets, and built-in YAML printer presets in PrintSim."
        ),
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_preset_yaml",
        "description": "Load the full declarative YAML text and metadata for a built-in printer preset ID from examples/.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "preset_id": {
                    "type": "string",
                    "description": "Preset stem from list_catalog().presets (e.g. 'voron_trident_300', 'voron_monolith_awd')",
                }
            },
            "required": ["preset_id"],
        },
    },
    {
        "name": "evaluate_printer_yaml",
        "description": (
            "Run full multiphysics 3D printer simulation (static 3D frame FEA, modal resonance & "
            "print-impact ranking, Klipper input shaper, kinematics/belt/torque, hotend FVM thermal, "
            "adjoint ROI subsystem breakdown, bed plate thermal/modal FEA, and virtual G-code print) "
            "on a YAML printer specification."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "yaml_text": {"type": "string", "description": "Declarative PrintSim YAML specification"},
                "filament_id": {
                    "type": "string",
                    "default": "pla",
                    "description": "Filament material ID (pla, petg, abs, asa, pa12_cf, tpu_95a)",
                },
                "gcode_text": {"type": "string", "description": "Optional custom G-code string to simulate"},
                "toolhead_rel_pos": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                    "default": [0.5, 0.5, 0.5],
                    "description": "Normalized [rx, ry, rz] toolhead/bed probe position in 0..1 inside the build volume",
                },
                "panel_mesh_n": {
                    "type": "integer",
                    "minimum": 2,
                    "maximum": 10,
                    "default": 3,
                    "description": "MITC4 shell panel refinement grid per bay (2..10; default 3 for <300ms balanced accuracy)",
                },
                "bed_mesh_n": {
                    "type": "integer",
                    "minimum": 4,
                    "maximum": 16,
                    "default": 8,
                    "description": "2D MITC4 heated bed plate mesh grid (4..16; default 8)",
                },
                "accel_override_mm_s2": {
                    "type": "number",
                    "minimum": 500.0,
                    "maximum": 100000.0,
                    "description": "Optional inertial acceleration override (mm/s^2) for static FEA probe",
                },
            },
            "required": ["yaml_text"],
        },
    },
    {
        "name": "mutate_and_compare_printer",
        "description": (
            "Batch-evaluate and rank multiple hardware/layout mutations of a base printer YAML using "
            "dot-path overrides (e.g. 'frame.extrusion_filling', 'gantry.bracket_material', "
            "'gantry.x_rail_mount', 'toolhead.cog_offset_mm', 'frame.custom_members') and compute deltas vs baseline."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "yaml_text": {"type": "string", "description": "Base printer YAML specification"},
                "candidates": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Candidate variant label"},
                            "overrides": {
                                "type": "object",
                                "description": "Mapping of dot-path keys to replacement values",
                            },
                        },
                        "required": ["name", "overrides"],
                    },
                },
                "rank_by": {
                    "type": "string",
                    "enum": ["composite", "safe_accel", "stiffness", "budget"],
                    "default": "composite",
                },
                "filament_id": {"type": "string", "default": "pla"},
                "toolhead_rel_pos": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                    "default": [0.5, 0.5, 0.5],
                },
            },
            "required": ["yaml_text", "candidates"],
        },
    },
    {
        "name": "optimize_printer_yaml",
        "description": "Run an evolutionary design optimization loop over frame profiles, bracing, gantry beams, belts, hotends, and accelerations.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "yaml_text": {"type": "string"},
                "objective": {
                    "type": "string",
                    "enum": ["balanced", "speed", "rigidity", "budget"],
                    "default": "balanced",
                },
                "iterations": {"type": "integer", "default": 20, "minimum": 3, "maximum": 100},
            },
            "required": ["yaml_text"],
        },
    },
    {
        "name": "run_solid_part_fea",
        "description": "Run 3D solid continuum Hex8 Finite Element Analysis (including FDM Z-layer anisotropy) on a built-in bracket/mount preset or a local STL/OBJ mesh file.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "preset": {
                    "type": "string",
                    "enum": ["corner_gusset", "xy_motor_mount", "toolhead_carriage", "cantilever_bar"],
                },
                "stl_path": {"type": "string", "description": "Optional path to a local STL/OBJ file (overrides preset)"},
                "material_id": {"type": "string", "default": "abs_printed"},
                "fixed_face": {"type": "string", "default": "min_z"},
                "load_face": {"type": "string", "default": "max_z"},
                "force_n": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                    "description": "Applied 3D force vector [Fx, Fy, Fz] in Newtons",
                },
                "fdm_anisotropic": {"type": "boolean"},
            },
            "required": [],
        },
    },
    {
        "name": "design_and_test_bracket",
        "description": (
            "Parametrically generate a 3D printer structural bracket (corner_gusset, motor_mount, "
            "or xy_joint_l_bracket), run 3D Hex8 (~0.1s) or conformal build123d+Gmsh Tet10 FEA, "
            "and optionally export watertight STL and/or OpenCASCADE STEP CAD."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "bracket_type": {
                    "type": "string",
                    "enum": ["corner_gusset", "motor_mount", "xy_joint_l_bracket"],
                    "default": "corner_gusset",
                },
                "arm_length_mm": {"type": "number", "default": 40.0, "minimum": 25.0, "maximum": 90.0},
                "height_mm": {"type": "number", "default": 20.0, "minimum": 10.0, "maximum": 60.0},
                "wall_thickness_mm": {"type": "number", "default": 6.0, "minimum": 2.5, "maximum": 18.0},
                "rib_thickness_mm": {"type": "number", "default": 6.0, "minimum": 0.0, "maximum": 18.0},
                "bolt_hole_mm": {"type": "number", "default": 5.5, "minimum": 0.0, "maximum": 10.0},
                "fillet_mm": {"type": "number", "default": 2.0, "minimum": 0.0, "maximum": 6.0},
                "material_id": {"type": "string", "default": "ppa_cf_printed"},
                "force_n": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                    "default": [60.0, 0.0, 0.0],
                },
                "fdm_anisotropic": {"type": "boolean"},
                "engine": {"type": "string", "enum": ["hex8", "tet10"], "default": "hex8"},
                "output_stl_path": {"type": "string", "description": "Optional STL output path"},
                "output_step_path": {"type": "string", "description": "Optional OpenCASCADE STEP output path"},
            },
            "required": [],
        },
    },
    {
        "name": "export_printer_stl",
        "description": "Generate and save a 3D solid STL/OBJ assembly mesh from a PrintSim YAML spec.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "yaml_text": {"type": "string"},
                "output_path": {"type": "string"},
            },
            "required": ["yaml_text", "output_path"],
        },
    },
    {
        "name": "optimize_part_topology",
        "description": "Run 3D SIMP topology optimization on a printer bracket/mount preset to reduce mass while maximizing stiffness, and optionally export the resulting watertight STL mesh.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "preset": {
                    "type": "string",
                    "enum": ["corner_gusset", "xy_motor_mount", "toolhead_carriage", "cantilever_bar"],
                    "default": "corner_gusset",
                },
                "material_id": {"type": "string", "default": "pa12_cf_printed"},
                "target_volume_fraction": {"type": "number", "default": 0.45, "minimum": 0.2, "maximum": 0.85},
                "iterations": {"type": "integer", "default": 18, "minimum": 5, "maximum": 40},
                "force_n": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "preserve_boundaries": {"type": "boolean", "default": True},
                "output_stl_path": {"type": "string", "description": "Optional path to save optimized STL"},
            },
            "required": [],
        },
    },
    {
        "name": "generate_printer_bom",
        "description": "Generate a manufacturable extrusion cut-list, 1D stock bar nesting plan, hardware BOM, and cost breakdown from a PrintSim YAML specification.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "yaml_text": {"type": "string", "description": "Declarative PrintSim YAML specification"},
                "stock_length_mm": {"type": "number", "default": 1000.0},
            },
            "required": ["yaml_text"],
        },
    },
    {
        "name": "export_printer_step",
        "description": "Export an exact OpenCASCADE B-Rep STEP CAD file (via build123d) of the full 3D printer assembly from a PrintSim YAML spec.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "yaml_text": {"type": "string"},
                "output_path": {"type": "string"},
            },
            "required": ["yaml_text", "output_path"],
        },
    },
    {
        "name": "run_tet_part_fea",
        "description": "Run high-accuracy conformal 3D tetrahedral continuum FEA (Gmsh + scikit-fem Tet10/Tet4) on a parametric build123d B-Rep bracket or a local STEP CAD file.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "preset": {
                    "type": "string",
                    "enum": ["corner_gusset", "xy_motor_mount", "toolhead_carriage", "cantilever_bar"],
                },
                "step_path": {"type": "string", "description": "Optional path to a local STEP CAD file (overrides preset)"},
                "material_id": {"type": "string", "default": "pa12_cf_printed"},
                "order": {"type": "integer", "enum": [1, 2], "default": 2},
                "fixed_face": {"type": "string", "default": "min_z"},
                "load_face": {"type": "string", "default": "max_z"},
                "force_n": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "fdm_anisotropic": {"type": "boolean"},
            },
            "required": [],
        },
    },
    {
        "name": "optimize_printer_pareto",
        "description": "Run pymoo NSGA-II multi-objective Pareto optimization trading off Build Cost ($), Safe Max Acceleration (mm/s^2), and Nozzle-Tip Stiffness (um).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "yaml_text": {"type": "string"},
                "population_size": {"type": "integer", "default": 16, "minimum": 8, "maximum": 40},
                "generations": {"type": "integer", "default": 6, "minimum": 2, "maximum": 20},
                "filament_id": {"type": "string", "default": "pla"},
            },
            "required": ["yaml_text"],
        },
    },
    {
        "name": "run_deep_frame_verify",
        "description": (
            "Run deep offline 6x6 MITC4 shell + Huth CBUSH bolted-fastener verification vs fast 2x2 "
            "simulation on a PrintSim YAML printer specification, returning mesh convergence deltas, "
            "panel drum resonance frequency, and shell panel von Mises stresses."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "yaml_text": {"type": "string", "description": "Declarative PrintSim YAML specification"},
                "filament_id": {"type": "string", "default": "pla"},
                "deep_mesh_n": {"type": "integer", "default": 8, "minimum": 4, "maximum": 10},
                "deep_bed_mesh_n": {"type": "integer", "default": 16, "minimum": 8, "maximum": 20},
                "toolhead_rel_pos": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                    "default": [0.5, 0.5, 0.5],
                },
            },
            "required": ["yaml_text"],
        },
    },
    {
        "name": "analyze_sensitivity_hotspots",
        "description": (
            "Run analytical structural adjoint and Fox-Kapoor modal sensitivity analysis (with "
            "subsystem_breakdown, primary_problem_subsystem, top stiffen/lighten targets, and "
            "worst_print_mode_number) alongside coupled workspace FRF mapping, time-domain "
            "corner/ballbar metrology, and RCSA toolhead whip diagnostics on a PrintSim YAML spec."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "yaml_text": {"type": "string", "description": "Declarative PrintSim YAML specification"},
                "filament_id": {"type": "string", "default": "pla"},
                "toolhead_rel_pos": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                    "default": [0.5, 0.5, 0.5],
                },
            },
            "required": ["yaml_text"],
        },
    },
    {
        "name": "condense_bracket_superelement",
        "description": (
            "Run Craig-Bampton Component Mode Synthesis (CMS) and RBE3 Virtual Point Transformation (VPT) "
            "condensation on a parametric 3D structural bracket, returning 6-DOF port stiffnesses, mass, "
            "fixed-interface modal frequencies, and SHA256 cache hash."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "bracket_type": {
                    "type": "string",
                    "enum": ["corner_gusset", "motor_mount", "xy_joint_l_bracket"],
                    "default": "corner_gusset",
                },
                "material_id": {"type": "string", "default": "ppa_cf_printed"},
                "arm_length_mm": {"type": "number", "default": 40.0, "minimum": 25.0, "maximum": 90.0},
                "height_mm": {"type": "number", "default": 20.0, "minimum": 10.0, "maximum": 60.0},
                "wall_thickness_mm": {"type": "number", "default": 6.0, "minimum": 2.5, "maximum": 18.0},
                "rib_thickness_mm": {"type": "number", "default": 5.0, "minimum": 0.0, "maximum": 18.0},
                "rib_count": {"type": "integer", "minimum": 0, "maximum": 6},
                "bolt_boss_thickness_mm": {"type": "number", "minimum": 0.0, "maximum": 18.0},
                "bolt_hole_mm": {"type": "number", "default": 5.2, "minimum": 0.0, "maximum": 10.0},
                "fillet_mm": {"type": "number", "default": 2.0, "minimum": 0.0, "maximum": 6.0},
                "n_modes": {"type": "integer", "default": 6, "minimum": 0, "maximum": 24},
                "fdm_anisotropic": {"type": "boolean", "default": False},
                "interface_mode": {"type": "string", "enum": ["rbe3", "rbe2"], "default": "rbe3"},
            },
            "required": [],
        },
    },
    {
        "name": "calibrate_adxl_csv",
        "description": (
            "Import real Klipper TEST_RESONANCES / Shake&Tune PSD CSVs or raw ADXL345 time-series CSVs, "
            "compare measured vs simulated PrintSim modal PSDs, compute Klipper shaper/damping deltas "
            "and spectral correlation, and return physical root-cause diagnostics (or auto-fit digital twin "
            "parameters with fit_parameters=true)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "yaml_spec": {"type": "string", "description": "Declarative PrintSim YAML specification"},
                "axis_x_csv": {
                    "type": "string",
                    "description": "Optional Klipper resonances_x.csv or raw ADXL345 time-series CSV text for X axis",
                },
                "axis_y_csv": {
                    "type": "string",
                    "description": "Optional Klipper resonances_y.csv or raw ADXL345 time-series CSV text for Y axis",
                },
                "max_freq_hz": {
                    "type": "number",
                    "default": 200.0,
                    "minimum": 20.0,
                    "maximum": 500.0,
                    "description": "Maximum frequency span in Hz (default 200.0)",
                },
                "fit_parameters": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "When true, run inverse digital-twin auto-fit calibration (fit_printer_to_adxl) "
                        "to tune belt tension and rail/corner stiffness parameters and return before/after error, "
                        "changes, and fitted_spec."
                    ),
                },
            },
            "required": ["yaml_spec"],
        },
    },
    {
        "name": "get_live_studio_state",
        "description": (
            "Inspect the user's live PrintSim Web Studio session: active printer YAML/spec, "
            "UI view mode, selected modal resonance mode detail, static/modal/thermal/sensitivity "
            "telemetry summary, browser connection status, and latest viewport snapshot metadata."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "include_yaml": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include the active PrinterSpec yaml_text in the response (default True)",
                }
            },
            "required": [],
        },
    },
    {
        "name": "capture_studio_viewport",
        "description": (
            "Request live 3D WebGL viewport PNG screenshot capture(s) from the connected "
            "PrintSim Studio browser (1 frame for static/geometry, 2 frames across modal phase "
            "for vibrating mode shapes) and wait until saved to disk."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "frames": {
                    "type": "integer",
                    "enum": [1, 2],
                    "default": 2,
                    "description": "Number of PNG frames to capture (1 or 2; default 2)",
                },
                "delay_ms": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 5000,
                    "default": 420,
                    "description": "Delay in ms between frames when capturing 2 frames (default 420)",
                },
                "timeout_s": {
                    "type": "number",
                    "minimum": 0.2,
                    "maximum": 30.0,
                    "default": 5.0,
                    "description": "Max seconds to wait for the live browser to upload snapshots (default 5.0)",
                },
            },
            "required": [],
        },
    },
    {
        "name": "update_live_studio",
        "description": (
            "Push a new YAML spec, dot-path spec overrides applied onto the live session's "
            "current spec, or UI view/load-case/mode changes directly into the live PrintSim Studio browser."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "yaml_text": {
                    "type": "string",
                    "description": "Optional replacement PrintSim YAML specification",
                },
                "overrides": {
                    "type": "object",
                    "description": "Optional dot-path overrides applied to the active live spec (or yaml_text)",
                },
                "view_mode": {
                    "type": "string",
                    "description": "Optional Studio viewport mode (e.g. 'geometry', 'static', 'modal', 'sensitivity', 'toolpath')",
                },
                "selected_mode_idx": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Optional 0-based modal resonance mode index to inspect",
                },
                "fea_load_case": {
                    "type": "string",
                    "description": "Optional static FEA load case ('combined', 'x', 'y', 'xy')",
                },
                "fea_scale": {
                    "type": "number",
                    "minimum": 0.0,
                    "description": "Optional FEA deflection visualization scale multiplier",
                },
                "toolhead_rel_pos": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                    "description": "Optional normalized [rx, ry, rz] toolhead position in 0..1",
                },
                "filament_id": {
                    "type": "string",
                    "description": "Optional filament material ID (e.g. 'pla', 'abs', 'pa12_cf')",
                },
            },
            "required": [],
        },
    },
]


def dispatch_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    if name == "list_catalog":
        return {
            "materials": list(MATERIAL_CATALOG.keys()),
            "profiles": list(PROFILE_CATALOG.keys()),
            "filaments": list(FILAMENT_CATALOG.keys()),
            "steppers": list(STEPPER_CATALOG.keys()),
            "toolhead_presets": list(TOOLHEAD_PRESETS.keys()),
            "extrusion_fillings": [
                {
                    "id": k,
                    "name": v["name"],
                    "bonded": v["bonded"],
                    "notes": v["notes"],
                }
                for k, v in EXTRUSION_FILLINGS.items()
            ],
            "gantry_brackets": [
                {
                    "id": k,
                    "name": v["name"],
                    "youngs_modulus_gpa": v["youngs_modulus_gpa"],
                    "hdt_c": v["hdt_c"],
                    "creep_tension_limit_n": v["creep_tension_limit_n"],
                    "kit_cost_usd": v["kit_cost_usd"],
                }
                for k, v in GANTRY_BRACKET_CATALOG.items()
            ],
            "bed_mounts": list(BED_MOUNT_CATALOG.values()),
            "frame_feet": list(FRAME_FEET_CATALOG.values()),
            "hotend_presets": list(HOTEND_CATALOG.values()),
            "z_drive_mechanisms": list(Z_DRIVE_MECHANISM_CATALOG.values()),
            "z_rail_profiles": Z_RAIL_PROFILES,
            "x_rail_mounts": ["front", "top", "bottom", "dual_top_bottom"],
            "y_rail_mounts": ["top", "inner", "bottom"],
            "drive_types": ["open_loop_stepper", "closed_loop_stepper", "foc_servo"],
            "input_shapers": [
                "auto",
                "none",
                "zv",
                "mzv",
                "ei",
                "2hump_ei",
                "3hump_ei",
                "zvdd",
                "smooth_zv",
                "smooth_ei",
            ],
            "part_presets": ["corner_gusset", "xy_motor_mount", "toolhead_carriage", "cantilever_bar"],
            "presets": _list_preset_ids(),
        }

    if name == "get_preset_yaml":
        preset_id = str(args["preset_id"]).strip()
        available = _list_preset_ids()
        if preset_id not in available:
            raise ValueError(f"Unknown preset '{preset_id}'. Available: {', '.join(available)}")
        yaml_text = (EXAMPLES_DIR / f"{preset_id}.yaml").read_text(encoding="utf-8")
        spec = PrinterSpec.model_validate(yaml.safe_load(yaml_text))
        return {
            "preset_id": preset_id,
            "name": spec.name,
            "architecture": spec.architecture,
            "yaml_text": yaml_text,
        }

    if name == "evaluate_printer_yaml":
        spec = PrinterSpec.model_validate(yaml.safe_load(args["yaml_text"]))
        raw_pos = args.get("toolhead_rel_pos")
        rel_pos: tuple[float, float, float] = (0.5, 0.5, 0.5)
        if raw_pos is not None:
            if len(raw_pos) != 3:
                raise ValueError("toolhead_rel_pos must be a 3-element [rx, ry, rz] list")
            rel_pos = (float(raw_pos[0]), float(raw_pos[1]), float(raw_pos[2]))
        panel_mesh_n = int(args.get("panel_mesh_n", 3))
        if panel_mesh_n < 2 or panel_mesh_n > 10:
            raise ValueError("panel_mesh_n must be an integer between 2 and 10")
        bed_mesh_n = int(args.get("bed_mesh_n", 8))
        if bed_mesh_n < 4 or bed_mesh_n > 16:
            raise ValueError("bed_mesh_n must be an integer between 4 and 16")
        raw_accel = args.get("accel_override_mm_s2")
        accel_override = float(raw_accel) if raw_accel is not None else None
        report = evaluate_printer(
            spec,
            gcode_text=args.get("gcode_text"),
            toolhead_rel_pos=rel_pos,
            filament_id=args.get("filament_id", "pla"),
            panel_mesh_n=panel_mesh_n,
            bed_mesh_n=bed_mesh_n,
            accel_override_mm_s2=accel_override,
        )
        payload = report.to_dict(include_toolpath_segments=False)
        modal_dict = payload.get("modal_fea") or {}
        sens_dict = payload.get("sensitivity") or {}
        bed_plate = payload.get("bed_plate_fea")
        top_modes = [_mode_impact_summary(m) for m in (modal_dict.get("modes") or [])[:4]]
        summary: dict[str, Any] = {
            "composite_score": (payload.get("scores") or {}).get("composite"),
            "worst_print_mode_number": modal_dict.get("worst_print_mode_number"),
            "top_modes": top_modes,
            "primary_problem_subsystem": sens_dict.get("primary_problem_subsystem"),
            "subsystem_breakdown": sens_dict.get("subsystem_breakdown"),
        }
        if isinstance(bed_plate, dict):
            summary["bed_plate_fea"] = {
                "total_flatness_error_um": bed_plate.get("total_flatness_error_um"),
                "taco_buckling_um": bed_plate.get("taco_buckling_um"),
                "plate_first_mode_hz": bed_plate.get("plate_first_mode_hz"),
                "first_layer_risk": bed_plate.get("first_layer_risk"),
            }
        payload["summary"] = summary
        return payload

    if name == "mutate_and_compare_printer":
        base_spec = PrinterSpec.model_validate(yaml.safe_load(args["yaml_text"]))
        candidates_in = args.get("candidates") or []
        if not candidates_in:
            raise ValueError("candidates must be a non-empty list of {name, overrides} objects")

        rank_by: Literal["composite", "safe_accel", "stiffness", "budget"] = args.get(
            "rank_by", "composite"
        )
        if rank_by not in _VALID_RANK_BY:
            raise ValueError(f"Unknown rank_by '{rank_by}'. Expected one of {_VALID_RANK_BY}.")

        filament_id = str(args.get("filament_id", "pla"))
        raw_pos = args.get("toolhead_rel_pos", [0.5, 0.5, 0.5])
        if len(raw_pos) != 3:
            raise ValueError("toolhead_rel_pos must be a 3-element [rx, ry, rz] list")
        rel_pos = (float(raw_pos[0]), float(raw_pos[1]), float(raw_pos[2]))

        baseline_summary = _summarize_printer(
            base_spec, filament_id=filament_id, toolhead_rel_pos=rel_pos
        )
        base_dict = base_spec.model_dump(mode="json")

        evaluated: list[tuple[dict[str, Any], PrinterSpec]] = []
        for cand in candidates_in:
            cname = str(cand["name"])
            overrides = cand.get("overrides") or {}
            cand_dict = _apply_dot_overrides(base_dict, overrides)
            cand_spec = PrinterSpec.model_validate(cand_dict)
            csum = _summarize_printer(
                cand_spec, filament_id=filament_id, toolhead_rel_pos=rel_pos
            )
            entry = {
                "name": cname,
                "overrides": overrides,
                **csum,
                "delta_score": round(csum["composite_score"] - baseline_summary["composite_score"], 2),
                "delta_freq_hz": round(
                    csum["fundamental_freq_hz"] - baseline_summary["fundamental_freq_hz"], 2
                ),
                "delta_accel_mm_s2": round(
                    csum["recommended_max_accel_mm_s2"]
                    - baseline_summary["recommended_max_accel_mm_s2"],
                    1,
                ),
                "delta_deflection_um": round(
                    csum["nozzle_deflection_um"] - baseline_summary["nozzle_deflection_um"], 2
                ),
                "delta_cost_usd": round(
                    csum["estimated_cost_usd"] - baseline_summary["estimated_cost_usd"], 2
                ),
            }
            evaluated.append((entry, cand_spec))

        def _sort_key(item: tuple[dict[str, Any], PrinterSpec]) -> tuple[float, ...]:
            c = item[0]
            if rank_by == "safe_accel":
                return (
                    -c["recommended_max_accel_mm_s2"],
                    -c["composite_score"],
                    c["nozzle_deflection_um"],
                )
            if rank_by == "stiffness":
                return (
                    c["nozzle_deflection_um"],
                    -(c["toolhead_stiffness_x_n_per_mm"] + c["toolhead_stiffness_y_n_per_mm"]),
                    -c["composite_score"],
                )
            if rank_by == "budget":
                return (c["estimated_cost_usd"], -c["composite_score"])
            return (-c["composite_score"], c["nozzle_deflection_um"], c["estimated_cost_usd"])

        evaluated.sort(key=_sort_key)
        ranked_candidates: list[dict[str, Any]] = []
        for idx, (entry, _) in enumerate(evaluated, start=1):
            ranked_candidates.append({"rank": idx, **entry})

        best_entry, best_spec = evaluated[0]
        return {
            "rank_by": rank_by,
            "filament_id": filament_id,
            "toolhead_rel_pos": [rel_pos[0], rel_pos[1], rel_pos[2]],
            "baseline": baseline_summary,
            "ranked_candidates": ranked_candidates,
            "best_candidate_name": best_entry["name"],
            "best_yaml": best_spec.to_yaml(),
        }

    if name == "optimize_printer_yaml":
        spec = PrinterSpec.model_validate(yaml.safe_load(args["yaml_text"]))
        res = optimize_design(
            spec,
            iterations=int(args.get("iterations", 20)),
            objective=args.get("objective", "balanced"),
        )
        out = res.to_dict()
        out["best_yaml"] = res.best_spec.to_yaml()
        return out

    if name == "run_solid_part_fea":
        force = tuple(args["force_n"]) if "force_n" in args and args["force_n"] else None
        if args.get("stl_path"):
            grid, h_mm = voxelize_stl_file(args["stl_path"])
            res = solve_solid_voxel_fea(
                grid=grid,
                voxel_size_mm=h_mm,
                material_id=args.get("material_id", "pa12_cf_printed"),
                fixed_face=args.get("fixed_face", "min_z"),
                load_face=args.get("load_face", "max_z"),
                force_n=force or (50.0, 0.0, -25.0),  # type: ignore[arg-type]
                part_name=Path(args["stl_path"]).name,
                fdm_anisotropic=args.get("fdm_anisotropic"),
            )
        else:
            res = solve_preset_part_fea(
                preset=args.get("preset", "corner_gusset"),
                material_id=args.get("material_id", "abs_printed"),
                force_n=force,  # type: ignore[arg-type]
                fdm_anisotropic=args.get("fdm_anisotropic"),
            )
        d = res.to_dict()
        d.pop("voxels", None)  # keep MCP response compact for LLM context
        return d

    if name == "design_and_test_bracket":
        raw_force = args.get("force_n")
        force_vec: tuple[float, float, float] = (
            (float(raw_force[0]), float(raw_force[1]), float(raw_force[2]))
            if raw_force is not None
            else (60.0, 0.0, 0.0)
        )
        return design_and_evaluate_bracket(
            bracket_type=args.get("bracket_type", "corner_gusset"),
            arm_length_mm=float(args.get("arm_length_mm", 40.0)),
            height_mm=float(args.get("height_mm", 20.0)),
            wall_thickness_mm=float(args.get("wall_thickness_mm", 6.0)),
            rib_thickness_mm=float(args.get("rib_thickness_mm", 6.0)),
            bolt_hole_mm=float(args.get("bolt_hole_mm", 5.5)),
            fillet_mm=float(args.get("fillet_mm", 2.0)),
            material_id=args.get("material_id", "ppa_cf_printed"),
            force_n=force_vec,
            fdm_anisotropic=args.get("fdm_anisotropic"),
            engine=args.get("engine", "hex8"),
            output_stl_path=args.get("output_stl_path"),
            output_step_path=args.get("output_step_path"),
        )

    if name == "export_printer_stl":
        spec = PrinterSpec.model_validate(yaml.safe_load(args["yaml_text"]))
        asm = build_assembly(spec)
        saved = export_assembly_stl(asm, args["output_path"])
        return {"saved_path": str(saved.resolve()), "bytes": saved.stat().st_size}

    if name == "optimize_part_topology":
        force = tuple(args["force_n"]) if "force_n" in args and args["force_n"] else None
        res = optimize_preset_topology(
            preset=args.get("preset", "corner_gusset"),
            material_id=args.get("material_id", "pa12_cf_printed"),
            force_n=force,  # type: ignore[arg-type]
            target_volume_fraction=float(args.get("target_volume_fraction", 0.45)),
            iterations=int(args.get("iterations", 18)),
            preserve_boundaries=bool(args.get("preserve_boundaries", True)),
        )
        out = res.to_dict(include_voxels=False)
        if args.get("output_stl_path"):
            saved = export_topology_stl(res, args["output_stl_path"])
            out["saved_stl_path"] = str(saved.resolve())
            out["stl_bytes"] = saved.stat().st_size
        return out

    if name == "generate_printer_bom":
        spec = PrinterSpec.model_validate(yaml.safe_load(args["yaml_text"]))
        bom = generate_bom(spec, stock_length_mm=float(args.get("stock_length_mm", 1000.0)))
        return bom.to_dict()

    if name == "export_printer_step":
        spec = PrinterSpec.model_validate(yaml.safe_load(args["yaml_text"]))
        asm = build_assembly(spec)
        saved = export_assembly_step(asm, args["output_path"])
        return {"saved_path": str(saved.resolve()), "bytes": saved.stat().st_size}

    if name == "run_tet_part_fea":
        force = tuple(args["force_n"]) if "force_n" in args and args["force_n"] else None
        order = 2 if int(args.get("order", 2)) >= 2 else 1
        if args.get("step_path"):
            res = solve_step_tet_fea(
                step_path=args["step_path"],
                material_id=args.get("material_id", "pa12_cf_printed"),
                fixed_face=args.get("fixed_face", "min_z"),
                load_face=args.get("load_face", "max_z"),
                force_n=force or (50.0, 0.0, -25.0),  # type: ignore[arg-type]
                order=order,  # type: ignore[arg-type]
                fdm_anisotropic=args.get("fdm_anisotropic"),
            )
        else:
            res = solve_preset_tet_fea(
                preset=args.get("preset", "corner_gusset"),
                material_id=args.get("material_id", "pa12_cf_printed"),
                force_n=force,  # type: ignore[arg-type]
                order=order,  # type: ignore[arg-type]
                fdm_anisotropic=args.get("fdm_anisotropic"),
            )
        return res.to_dict(include_surface_mesh=False)

    if name == "optimize_printer_pareto":
        spec = PrinterSpec.model_validate(yaml.safe_load(args["yaml_text"]))
        res = optimize_pareto_front(
            spec,
            population_size=int(args.get("population_size", 16)),
            generations=int(args.get("generations", 6)),
            filament_id=args.get("filament_id", "pla"),
        )
        return res.to_dict(include_specs=False)

    if name == "run_deep_frame_verify":
        spec = PrinterSpec.model_validate(yaml.safe_load(args["yaml_text"]))
        raw_pos = args.get("toolhead_rel_pos")
        rel_pos: tuple[float, float, float] = (0.5, 0.5, 0.5)
        if raw_pos is not None:
            if len(raw_pos) != 3:
                raise ValueError("toolhead_rel_pos must be a 3-element [rx, ry, rz] list")
            rel_pos = (float(raw_pos[0]), float(raw_pos[1]), float(raw_pos[2]))
        return verify_printer_deep(
            spec,
            toolhead_rel_pos=rel_pos,
            filament_id=str(args.get("filament_id", "pla")),
            deep_mesh_n=int(args.get("deep_mesh_n", 8)),
            deep_bed_mesh_n=int(args.get("deep_bed_mesh_n", 16)),
        )

    if name == "analyze_sensitivity_hotspots":
        spec = PrinterSpec.model_validate(yaml.safe_load(args["yaml_text"]))
        raw_pos = args.get("toolhead_rel_pos")
        rel_pos: tuple[float, float, float] = (0.5, 0.5, 0.5)
        if raw_pos is not None:
            if len(raw_pos) != 3:
                raise ValueError("toolhead_rel_pos must be a 3-element [rx, ry, rz] list")
            rel_pos = (float(raw_pos[0]), float(raw_pos[1]), float(raw_pos[2]))
        report = evaluate_printer(
            spec,
            toolhead_rel_pos=rel_pos,
            filament_id=str(args.get("filament_id", "pla")),
        )
        sens = report.sensitivity.to_dict() if report.sensitivity is not None else {}
        modal_dict = report.modal_fea.to_dict()
        worst_mode_num = int(modal_dict.get("worst_print_mode_number", 1))
        worst_mode = next(
            (m for m in modal_dict.get("modes", []) if m.get("mode_number") == worst_mode_num),
            None,
        )
        wfrf = report.workspace_frf.to_dict() if report.workspace_frf is not None else {}
        tdm = (
            report.time_domain_metrology.to_dict()
            if report.time_domain_metrology is not None
            else {}
        )
        rcsa = report.rcsa.to_dict() if report.rcsa is not None else {}
        return {
            "printer_name": report.spec.name,
            "fundamental_freq_hz": round(float(report.modal_fea.fundamental_freq_hz), 2),
            "worst_print_mode_number": worst_mode_num,
            "worst_print_mode": _mode_impact_summary(worst_mode) if worst_mode is not None else None,
            "primary_problem_subsystem": sens.get("primary_problem_subsystem"),
            "subsystem_breakdown": sens.get("subsystem_breakdown", {}),
            "summary_diagnosis": sens.get("summary_diagnosis", ""),
            "top_stiffen_targets": sens.get("top_stiffen_targets", []),
            "top_lighten_targets": sens.get("top_lighten_targets", []),
            "panel_bay_sensitivities": sens.get("panel_bay_sensitivities", {}),
            "workspace_frf": {
                "grid_n": wfrf.get("grid_n"),
                "center_freq_hz": wfrf.get("center_freq_hz"),
                "min_freq_hz": wfrf.get("min_freq_hz"),
                "max_freq_hz": wfrf.get("max_freq_hz"),
                "worst_corner_drop_pct": wfrf.get("worst_corner_drop_pct"),
                "max_cross_coupling_pct": wfrf.get("max_cross_coupling_pct"),
                "reflected_rotor_mass_g": wfrf.get("reflected_rotor_mass_g"),
                "drive_type": wfrf.get("drive_type"),
                "midband_instability_risk": wfrf.get("midband_instability_risk"),
            },
            "time_domain_metrology": {
                "corner_rounding_um": tdm.get("corner_rounding_um"),
                "corner_overshoot_um": tdm.get("corner_overshoot_um"),
                "ringing_peak_to_peak_um": tdm.get("ringing_peak_to_peak_um"),
                "ringing_unshaped_um": tdm.get("ringing_unshaped_um"),
                "ringing_reduction_pct": tdm.get("ringing_reduction_pct"),
                "ringing_decay_mm": tdm.get("ringing_decay_mm"),
                "shaper_used": tdm.get("shaper_used"),
                "shaper_freq_hz": tdm.get("shaper_freq_hz"),
                "ballbar_radius_mm": tdm.get("ballbar_radius_mm"),
                "ballbar_speed_mm_s": tdm.get("ballbar_speed_mm_s"),
                "circularity_gxy_um": tdm.get("circularity_gxy_um"),
                "mean_radial_lag_um": tdm.get("mean_radial_lag_um"),
                "quadrant_glitch_um": tdm.get("quadrant_glitch_um"),
            },
            "rcsa": {
                "sensor_peak_freq_hz": rcsa.get("sensor_peak_freq_hz"),
                "nozzle_tip_peak_freq_hz": rcsa.get("nozzle_tip_peak_freq_hz"),
                "nozzle_to_sensor_amp_ratio": rcsa.get("nozzle_to_sensor_amp_ratio"),
                "hotend_bending_stiffness_n_per_mm": rcsa.get("hotend_bending_stiffness_n_per_mm"),
                "whip_risk": rcsa.get("whip_risk"),
                "warning_message": rcsa.get("warning_message"),
            },
        }

    if name == "condense_bracket_superelement":
        rib_count = args.get("rib_count")
        if args.get("rib_thickness_mm") is not None:
            resolved_rib = float(args["rib_thickness_mm"])
        elif rib_count is not None:
            rc = int(rib_count)
            resolved_rib = 0.0 if rc == 0 else min(18.0, 3.5 + 1.5 * rc)
        else:
            resolved_rib = 5.0

        wall_base = float(args.get("wall_thickness_mm", 6.0))
        boss_t = args.get("bolt_boss_thickness_mm")
        resolved_wall = max(wall_base, float(boss_t)) if boss_t is not None else wall_base

        bracket_params = {
            "bracket_type": str(args.get("bracket_type", "corner_gusset")),
            "arm_length_mm": float(args.get("arm_length_mm", 40.0)),
            "height_mm": float(args.get("height_mm", 20.0)),
            "wall_thickness_mm": resolved_wall,
            "rib_thickness_mm": resolved_rib,
            "bolt_hole_mm": float(args.get("bolt_hole_mm", 5.2)),
            "fillet_mm": float(args.get("fillet_mm", 2.0)),
            "n_modes": int(args.get("n_modes", 6)),
            "fdm_anisotropic": bool(args.get("fdm_anisotropic", False)),
            "interface_mode": str(args.get("interface_mode", "rbe3")),
        }
        se = condense_parametric_bracket(
            bracket_params,
            material_id=str(args.get("material_id", "ppa_cf_printed")),
        )
        return se.to_dict()

    if name == "calibrate_adxl_csv":
        spec = PrinterSpec.model_validate(yaml.safe_load(args["yaml_spec"]))
        axis_x_csv = args.get("axis_x_csv")
        axis_y_csv = args.get("axis_y_csv")
        max_freq_hz = float(args.get("max_freq_hz", 200.0))
        if bool(args.get("fit_parameters", False)):
            fit_report = fit_printer_to_adxl(
                spec,
                axis_x_csv=axis_x_csv,
                axis_y_csv=axis_y_csv,
                max_freq_hz=max_freq_hz,
            )
            return fit_report.to_dict()
        report = calibrate_printer_against_adxl(
            spec,
            axis_x_csv=axis_x_csv,
            axis_y_csv=axis_y_csv,
            max_freq_hz=max_freq_hz,
        )
        return report.to_dict()

    if name == "get_live_studio_state":
        include_yaml = bool(args.get("include_yaml", True))
        session = get_live_session()
        has_session = bool(
            session.get("spec")
            or session.get("simulation_summary")
            or session.get("last_heartbeat_at")
            or session.get("updated_at")
        )
        if not include_yaml:
            session.pop("yaml_text", None)
        session["has_session"] = has_session
        if not has_session:
            session["message"] = (
                "No active or recorded PrintSim Studio session found yet. "
                "Open the Studio browser UI or run POST /api/simulate first."
            )
        return session

    if name == "capture_studio_viewport":
        frames = int(args.get("frames", 2))
        delay_ms = int(args.get("delay_ms", 420))
        timeout_s = float(args.get("timeout_s", 5.0))
        return request_snapshot_capture(
            frames=frames,
            delay_ms=delay_ms,
            timeout_s=timeout_s,
        )

    if name == "update_live_studio":
        validated_spec: PrinterSpec | None = None
        base_dict: dict[str, Any] | None = None

        raw_yaml = args.get("yaml_text")
        raw_overrides = args.get("overrides")

        if raw_yaml is not None and str(raw_yaml).strip():
            parsed = yaml.safe_load(str(raw_yaml))
            validated_spec = PrinterSpec.model_validate(parsed)
            base_dict = validated_spec.model_dump(mode="json")
        elif raw_overrides is not None:
            live = get_live_session()
            live_spec = live.get("spec")
            if isinstance(live_spec, dict):
                base_dict = live_spec
            elif live.get("yaml_text"):
                base_dict = yaml.safe_load(str(live["yaml_text"]))
            if not isinstance(base_dict, dict):
                raise ValueError(
                    "No live Studio spec found to apply overrides to. "
                    "Provide yaml_text or simulate a printer in Studio first."
                )

        if raw_overrides is not None:
            if not isinstance(raw_overrides, dict):
                raise ValueError("overrides must be a mapping of dot-path keys to values")
            assert base_dict is not None
            updated_dict = _apply_dot_overrides(base_dict, raw_overrides)
            validated_spec = PrinterSpec.model_validate(updated_dict)

        cmd_payload: dict[str, Any] = {}
        if validated_spec is not None:
            cmd_payload["spec"] = validated_spec.model_dump(mode="json")
            cmd_payload["yaml_text"] = validated_spec.to_yaml()

        if args.get("view_mode") is not None:
            cmd_payload["view_mode"] = str(args["view_mode"])
        if args.get("selected_mode_idx") is not None:
            cmd_payload["selected_mode_idx"] = int(args["selected_mode_idx"])
        if args.get("fea_load_case") is not None:
            cmd_payload["fea_load_case"] = str(args["fea_load_case"])
        if args.get("fea_scale") is not None:
            cmd_payload["fea_scale"] = float(args["fea_scale"])
        if args.get("filament_id") is not None:
            cmd_payload["filament_id"] = str(args["filament_id"])
        if args.get("toolhead_rel_pos") is not None:
            raw_pos = args["toolhead_rel_pos"]
            if len(raw_pos) != 3:
                raise ValueError("toolhead_rel_pos must be a 3-element [rx, ry, rz] list")
            cmd_payload["toolhead_rel_pos"] = [
                float(raw_pos[0]),
                float(raw_pos[1]),
                float(raw_pos[2]),
            ]

        if not cmd_payload:
            raise ValueError(
                "Provide at least one update field (yaml_text, overrides, view_mode, "
                "selected_mode_idx, fea_load_case, fea_scale, toolhead_rel_pos, or filament_id)."
            )

        cmd = enqueue_command("update_state", cmd_payload)
        live_after = get_live_session()
        response: dict[str, Any] = {
            "ok": True,
            "enqueued": True,
            "command": cmd,
            "connected": bool(live_after.get("connected")),
            "updated_fields": sorted(cmd_payload.keys()),
        }
        if validated_spec is not None:
            response["spec_summary"] = {
                "name": validated_spec.name,
                "architecture": validated_spec.architecture,
                "build_volume_mm": list(validated_spec.kinematics.build_volume_mm),
                "frame_profile": validated_spec.frame.profile,
                "x_beam_profile": validated_spec.gantry.x_beam_profile,
                "bracket_material": validated_spec.gantry.bracket_material,
                "max_accel_mm_s2": validated_spec.kinematics.max_accel_mm_s2,
            }
        return response

    raise ValueError(f"Unknown MCP tool: {name}")


def handle_jsonrpc_message(msg: dict[str, Any]) -> dict[str, Any] | None:
    req_id = msg.get("id")
    method = msg.get("method", "")
    params = msg.get("params") or {}

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": SERVER_INFO,
            },
        }
    if method == "notifications/initialized":
        return None
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS_SCHEMA}}
    if method == "tools/call":
        tname = params.get("name", "")
        targs = params.get("arguments") or {}
        try:
            result = dispatch_tool(tname, targs)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result, indent=2)}],
                    "isError": False,
                },
            }
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": f"Error: {exc}"}],
                    "isError": True,
                },
            }

    if req_id is None:
        return None
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def run_stdio_server() -> None:
    """Run line-delimited JSON-RPC 2.0 MCP server on stdin/stdout."""
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = handle_jsonrpc_message(msg)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
