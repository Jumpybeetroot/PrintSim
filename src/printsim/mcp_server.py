"""Native stdio MCP (Model Context Protocol) JSON-RPC 2.0 server for AI design agents."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml

from printsim.engine import evaluate_printer, optimize_design
from printsim.fea.solid_fem import solve_preset_part_fea, solve_solid_voxel_fea, voxelize_stl_file
from printsim.geometry import build_assembly, export_assembly_stl
from printsim.schema import MATERIAL_CATALOG, PROFILE_CATALOG, PrinterSpec
from printsim.thermal import FILAMENT_CATALOG

SERVER_INFO = {"name": "printsim-mcp", "version": "0.2.0"}

TOOLS_SCHEMA = [
    {
        "name": "list_catalog",
        "description": "List all available engineering materials, structural profiles, filaments, and built-in solid part presets in PrintSim.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "evaluate_printer_yaml",
        "description": "Run full multiphysics 3D printer simulation (static 3D frame FEA, modal resonance & Klipper input shaper, kinematics/belt/torque, hotend FVM thermal & volumetric flow, and virtual G-code print) on a YAML printer specification.",
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
            },
            "required": ["yaml_text"],
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
]


def dispatch_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    if name == "list_catalog":
        return {
            "materials": list(MATERIAL_CATALOG.keys()),
            "profiles": list(PROFILE_CATALOG.keys()),
            "filaments": list(FILAMENT_CATALOG.keys()),
            "part_presets": ["corner_gusset", "xy_motor_mount", "toolhead_carriage", "cantilever_bar"],
        }

    if name == "evaluate_printer_yaml":
        spec = PrinterSpec.model_validate(yaml.safe_load(args["yaml_text"]))
        report = evaluate_printer(
            spec,
            gcode_text=args.get("gcode_text"),
            filament_id=args.get("filament_id", "pla"),
        )
        return report.to_dict(include_toolpath_segments=False)

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

    if name == "export_printer_stl":
        spec = PrinterSpec.model_validate(yaml.safe_load(args["yaml_text"]))
        asm = build_assembly(spec)
        saved = export_assembly_stl(asm, args["output_path"])
        return {"saved_path": str(saved.resolve()), "bytes": saved.stat().st_size}

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
