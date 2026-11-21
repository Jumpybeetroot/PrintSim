# PrintSim AI Agent Integration Guide (v0.8)

PrintSim exposes a zero-setup **Model Context Protocol (MCP)** stdio server (`printsim mcp`), a **FastAPI REST API** (`printsim serve`), and a **Python SDK** so external AI agents (**Claude Desktop**, **Cursor**, **Antigravity / Windsurf**, **Gemini CLI**, **OpenAI / LangGraph**, and autonomous Python loops) can inspect catalogs, mutate printer YAML specs, batch-compare upgrades, design parametric brackets, run Hex8/Tet10 FEA, and export printable STL/STEP CAD.

---

## 1. Plug-and-Play MCP Server Setup

All MCP clients talk to PrintSim over line-delimited JSON-RPC 2.0 on `stdin`/`stdout` via:

```bash
uv --directory /absolute/path/to/printsim run printsim mcp
```

Replace `/absolute/path/to/printsim` with your local clone path (for example, `/Users/you/Projects/printsim`).

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "printsim": {
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/printsim", "run", "printsim", "mcp"]
    }
  }
}
```

### Cursor

Add to `.cursor/mcp.json` (project-level) or `~/.cursor/mcp.json` (global):

```json
{
  "mcpServers": {
    "printsim": {
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/printsim", "run", "printsim", "mcp"]
    }
  }
}
```

### Antigravity / Windsurf

Add to `~/.gemini/antigravity/mcp_config.json` (Antigravity) or `~/.codeium/windsurf/mcp_config.json` (Windsurf):

```json
{
  "mcpServers": {
    "printsim": {
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/printsim", "run", "printsim", "mcp"]
    }
  }
}
```

### Gemini CLI

Add to `~/.gemini/settings.json`:

```json
{
  "mcpServers": {
    "printsim": {
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/printsim", "run", "printsim", "mcp"],
      "trust": true
    }
  }
}
```

---

## 2. Complete MCP Tool Reference (13 Tools)

| # | Tool Name | Key Inputs | When to Use |
| :--- | :--- | :--- | :--- |
| 1 | `list_catalog` | `{}` | First step in any agent session: lists all `materials`, `profiles`, `filaments`, `steppers`, `toolhead_presets`, `extrusion_fillings` (with notes), `gantry_brackets` (with GPa & creep limits), `x_rail_mounts`, `y_rail_mounts`, `part_presets`, and `presets`. |
| 2 | `get_preset_yaml` | `preset_id` | Fetch a ready-to-simulate printer YAML baseline (e.g., `voron_trident_300`, `voron_monolith_awd`, `corexy_voron_style`, `cartesian_bedslinger`, `delta_racer`, `hyper_corexy_carbon`). |
| 3 | `evaluate_printer_yaml` | `yaml_text`, `filament_id?`, `gcode_text?`, `toolhead_rel_pos?` | Full multiphysics simulation (3D frame FEA + nozzle-tip lever twist, modal resonance + Klipper ADXL345 shaper, belt/stepper dynamics, FVM hotend thermal, virtual G-code print). Use `toolhead_rel_pos` (e.g., `[0.5, 0.5, 0.05]` vs `[0.5, 0.5, 0.95]`) to probe bed/gantry Z-height sensitivity. |
| 4 | `mutate_and_compare_printer` | `yaml_text`, `candidates`, `rank_by?`, `filament_id?`, `toolhead_rel_pos?` | Batch-evaluate multiple design hypotheses in one call using dot-path overrides (e.g., `frame.extrusion_filling`, `gantry.bracket_material`, `gantry.x_rail_mount`, `toolhead.cog_offset_mm`, `frame.custom_members`). Returns baseline, ranked candidates with exact deltas (`delta_score`, `delta_freq_hz`, `delta_accel_mm_s2`, `delta_deflection_um`, `delta_cost_usd`), and `best_yaml`. |
| 5 | `optimize_printer_yaml` | `yaml_text`, `objective?`, `iterations?` | Run single-objective evolutionary search (`balanced`, `speed`, `rigidity`, `budget`) across frame profiles, bracing, gantry beams, belts, and hotend specs. |
| 6 | `optimize_printer_pareto` | `yaml_text`, `population_size?`, `generations?`, `filament_id?` | Run `pymoo` NSGA-II multi-objective optimization trading off Build Cost (`$`), Safe Max Acceleration (`mm/s^2`), and Nozzle-Tip Deflection (`um`). |
| 7 | `generate_printer_bom` | `yaml_text`, `stock_length_mm?` | Generate manufacturable extrusion cut-lists, 1D stock bar nesting plans, fastener/belt/motor/panel BOM, and total build cost (`$`). |
| 8 | `design_and_test_bracket` | `bracket_type?`, `arm_length_mm?`, `height_mm?`, `wall_thickness_mm?`, `rib_thickness_mm?`, `bolt_hole_mm?`, `fillet_mm?`, `material_id?`, `force_n?`, `engine?`, `output_stl_path?`, `output_step_path?` | Parametric bracket designer (`corner_gusset`, `motor_mount`, `xy_joint_l_bracket`) with fast ~0.1s Hex8 voxel FEA or conformal `build123d` + Gmsh `tet10` FEA, plus direct watertight STL and OpenCASCADE STEP CAD export. |
| 9 | `run_solid_part_fea` | `preset?` or `stl_path?`, `material_id?`, `fixed_face?`, `load_face?`, `force_n?`, `fdm_anisotropic?` | Run 3D Hex8 continuum FEA (with optional FDM Z-layer anisotropy) on a built-in part preset or an existing local `.stl`/`.obj` file. |
| 10 | `run_tet_part_fea` | `preset?` or `step_path?`, `material_id?`, `order?`, `fixed_face?`, `load_face?`, `force_n?`, `fdm_anisotropic?` | Run high-accuracy curved `Tet10` (`order=2`) or `Tet4` (`order=1`) FEA via Gmsh + `scikit-fem` on a built-in B-Rep preset or a local `.step` CAD file. |
| 11 | `optimize_part_topology` | `preset?`, `material_id?`, `target_volume_fraction?`, `iterations?`, `force_n?`, `preserve_boundaries?`, `output_stl_path?` | Run 3D SIMP topology optimization to carve weight out of a bracket/mount while preserving stiffness and boundary faces, with optional watertight STL export. |
| 12 | `export_printer_stl` | `yaml_text`, `output_path` | Export a 3D solid STL/OBJ/GLB assembly mesh of the full printer. |
| 13 | `export_printer_step` | `yaml_text`, `output_path` | Export an exact OpenCASCADE ISO-10303-21 B-Rep STEP CAD assembly (via `build123d`) for FreeCAD/Fusion 360/SolidWorks. |

---

## 3. Ready-to-Use AI Agent Workflow Prompts

### Prompt A: Discover the Stiffest Sub-\$1,200 300mm CoreXY
> "Use `list_catalog` and `get_preset_yaml` (`voron_trident_300`). Run `mutate_and_compare_printer` with `rank_by='stiffness'` testing 5 targeted upgrades: (1) `bolted_structural` 4mm `aluminum_5052_sheet` panels, (2) `epoxy_granite` extrusion filling, (3) `ppa_cf_printed` gantry brackets, (4) `bottom` X-rail + `inner` Y-rail mounts, and (5) `extrusion_3030` frame. Then run `generate_printer_bom` on the best sub-\$1,200 combination and report the exact cost, frequency, and nozzle deflection."

### Prompt B: Design and FEA-Verify a Lightweight PPA-CF Corner Gusset Bracket
> "Using `design_and_test_bracket`, sweep `corner_gusset` geometries in `ppa_cf_printed` under `force_n=[60, 0, 0]` varying `wall_thickness_mm` (`3.5` to `8.0`) and `rib_thickness_mm` (`0` to `8.0`) with the fast `hex8` engine to maximize `specific_stiffness_n_per_mm_per_g` while keeping `safety_factor >= 5.0`. Then verify the winning dimensions with `engine='tet10'` and export both `.stl` and `.step` files."

### Prompt C: Compare Front vs Bottom X-Rail Mounting & Extrusion Fillings
> "Load `corexy_voron_style` via `get_preset_yaml`. Use `mutate_and_compare_printer` to compare `gantry.x_rail_mount` (`front`, `top`, `bottom`, `dual_top_bottom`) combined with `frame.extrusion_filling` (`none`, `silica_sand`, `epoxy_granite`, `uhpc_grout`). Also run `evaluate_printer_yaml` at `toolhead_rel_pos=[0.5, 0.5, 0.05]` vs `[0.5, 0.5, 0.95]` on the top candidate."

---

## 4. HTTP REST API (FastAPI Server)

Start the local HTTP server (also serves the Three.js Web Studio):

```bash
uv run printsim serve --host 127.0.0.1 --port 8000
```

Key endpoints for programmatic agents:

| Method & Path | Purpose |
| :--- | :--- |
| `GET /api/catalog` | Full materials, profiles, steppers, toolheads, fillings, brackets, rail mounts & presets |
| `GET /api/presets/{preset_id}` | Preset JSON spec + `yaml_text` |
| `POST /api/simulate` | Full multiphysics simulation (`yaml_text` or `spec`, `filament_id`, `toolhead_rel_pos`) |
| `POST /api/optimize` | Single-objective evolutionary optimizer |
| `POST /api/optimize/pareto` | `pymoo` NSGA-II Cost vs. Safe Accel vs. Nozzle Deflection Pareto frontier |
| `POST /api/bom` | JSON cut-list, stock bar nesting, fasteners, and cost BOM (`POST /api/bom/csv` for CSV) |
| `POST /api/fea/solid-part` | 3D Hex8 FEA on built-in bracket presets |
| `POST /api/fea/solid-stl` | 3D Hex8 FEA on base64-encoded `.stl`/`.obj` uploads |
| `POST /api/fea/tet-part` | Gmsh + `scikit-fem` curved `Tet10`/`Tet4` FEA on built-in B-Rep presets |
| `POST /api/fea/tet-step` | Gmsh + `scikit-fem` curved `Tet10`/`Tet4` FEA on base64-encoded `.step` CAD uploads |
| `POST /api/fea/topology-optimize` | 3D SIMP topology optimization (`POST /api/fea/topology-optimize/stl` for binary STL) |
| `POST /api/export/step` | Full printer assembly ISO-10303-21 STEP export |
| `POST /api/export/stl` | Full printer assembly STL mesh export |

---

## 5. Python SDK, LangGraph & OpenAI Function Calling

### Direct Python Discovery Loop

Run the included two-stage autonomous discovery script:

```bash
uv run python examples/agent_discovery_loop.py
```

Or embed PrintSim's MCP tool dispatcher directly in any **OpenAI Agents SDK** or **LangGraph** tool wrapper:

```python
from printsim.mcp_server import TOOLS_SCHEMA, dispatch_tool

# 1. Expose PrintSim tools directly to OpenAI / LangGraph / Anthropic function-calling
openai_tools = [
    {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["inputSchema"],
        },
    }
    for t in TOOLS_SCHEMA
]

# 2. Execute any model tool call in-process with zero subprocess overhead
preset = dispatch_tool("get_preset_yaml", {"preset_id": "voron_trident_300"})
comparison = dispatch_tool(
    "mutate_and_compare_printer",
    {
        "yaml_text": preset["yaml_text"],
        "rank_by": "stiffness",
        "candidates": [
            {"name": "Bottom X-Rail", "overrides": {"gantry.x_rail_mount": "bottom"}},
            {
                "name": "PPA-CF + Epoxy Granite",
                "overrides": {
                    "frame.extrusion_filling": "epoxy_granite",
                    "gantry.bracket_material": "ppa_cf_printed",
                },
            },
        ],
    },
)
print(comparison["best_candidate_name"])
```

### Direct Python Physics API

```python
from printsim.engine import evaluate_printer, optimize_pareto_front
from printsim.fea.bracket_designer import design_and_evaluate_bracket
from printsim.schema import PrinterSpec

spec = PrinterSpec.from_yaml("examples/voron_trident_300.yaml")
report = evaluate_printer(spec, filament_id="abs", toolhead_rel_pos=(0.5, 0.5, 0.5))
print(report.scores.to_dict())

bracket = design_and_evaluate_bracket(
    bracket_type="corner_gusset",
    arm_length_mm=40.0,
    wall_thickness_mm=5.0,
    rib_thickness_mm=6.0,
    material_id="ppa_cf_printed",
    engine="hex8",
)
print(bracket["stiffness_n_per_mm"], bracket["specific_stiffness_n_per_mm_per_g"])
```
