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

---

### Mounting, Ballast & Belt Tension Fields

| Field | Values | Effect |
| :--- | :--- | :--- |
| `frame.feet_type` | `rigid_bolted`, `rubber_feet`, `anti_vib_pads`, `paver_on_foam`, `suspended_lines` | Mount under (or above) the frame |
| `frame.suspension_line`, `frame.suspension_length_mm` | catalog `suspension_lines`; 50–5000 mm | Line material and free length when suspended |
| `frame.ballast_mass_kg`, `frame.ballast_location` | 0–200 kg; `base`, `top`, `distributed` | Dead weight on the frame corners |
| `gantry.belt.pretension_set_temp_c` | °C or null | Temperature at which `pretension_n` was set; null = already the operating tension |
| `gantry.belt.tension_mismatch_pct` | 0–80 % | CoreXY A/B belt tension difference |

### Frame Topology & Panel Fields (VzBot-style builds)

| Field | Values | Effect |
| :--- | :--- | :--- |
| `frame.gantry_on_top_frame` | `true` / `false` (fixed-gantry CoreXY only) | Y rails bolt to the top-ring Y extrusions: the gantry plane is the top ring, no separate gantry ring or stub columns; `gantry_top_clearance_mm` is ignored and Z clearance uses the explicit bed/nozzle stack |
| `frame.side_panel_material` | material id or null | Left/right panel sheet when it differs from `panel_material` (e.g. 3 mm Al rear + polycarbonate sides) |
| `gantry.hardware_kit` | `vzbot_330_awd`, `vzbot_330_2wd` (+ Voron/Siboor/Monolith kits) | CAD-baked VzBot rear motor mounts, front AWD/2WD mounts and XY joints, plus kit settings (MGN9H top X, inner MGN12H Y, 6 mm GT2 at 22 N) |
| `toolhead.offset_datum_rail_mount` | `front` (default), `top` | X-rail mounting the CoG / nozzle offsets were measured on; the builder's top/bottom-mount datum shift is skipped when it matches `gantry.x_rail_mount` |
| `toolhead.preset_id` | `vz_printhead_cnc` | CAD-measured Vz-Printhead CNC + HextrudORT Low + Goliath (343 g dressed, nozzle −54.2 mm below the tube axis) |

Reference build: `examples/vzbot_330_awd.yaml` (official VzBoT3D BOM + `Vz-330_v1.2` STEP), also a Studio base machine (`/api/catalog` preset `vzbot_330_awd`, DIY / Self-Sourced Build group).

`modal_fea.mounting_dynamics` (null on a rigidly bolted frame) lists the whole-printer rigid-body
modes, frame recoil per axis and, when suspended, line tension and safety factor.
`modal_fea.belt_dynamics` reports installed and operating tension, pluck frequency on a 150 mm
span, and the A/B coupling and cross-axis peaks.

`printsim belt-identify <spec> --axis X|Y [--point u,v,freq_hz ...] [--json]` separates the belt's
in-place stiffness from the rest of the drive chain using the axis resonance measured at three or
more toolhead positions (`fea/belt_identification.py`); without `--point` it returns the
measurement plan and the model's predicted frequency at each position. Belt branch lengths in
`evaluate_printer` follow `toolhead_rel_pos`.

## 6. Hands-Off Discord & Touchstone KB Research Workflow

To gather empirical 3D-printer hardware specs (toolhead/hotend weights, gantry masses, polymer tests, and Klipper/Shake&Tune ADXL resonance data) from public 3D-printing Discord communities without manual babysitting, PrintSim combines:

- **Workspace Skill & Server Directory**: [`.agents/skills/discord-research/SKILL.md`](../.agents/skills/discord-research/SKILL.md) + [`.agents/skills/discord-research/servers.yaml`](../.agents/skills/discord-research/servers.yaml) (pre-mapped channel IDs for Monolith, VzBoT, Klipper, Luke's Laboratory, Isik's Tech, ANNEX Engineering, Armchair Heavy Industries, Kalico, and VORONDesign).
- **Zero-LLM Query Router + KB Cache Check**:
  ```bash
  uv run python .agents/skills/discord-research/scripts/route_query.py "vzbot 330 adxl" --topic adxl
  ```
  Checks PrintSim's local SQLite+FTS5 Touchstone KB (`data/touchstones/printsim_kb.db`) first, then emits direct Discord channel URLs, `[CHECK PINS FIRST]` hints, and native Discord search queries (`in:#channel`, `has:file`, `has:image`).
- **Automatic Persistence**: Captured thread findings (`data/touchstones/discord_captures/*.json`) and downloaded Klipper/Shake&Tune CSVs (`data/touchstones/adxl_csvs/*.csv`) are ingested directly into the local KB via `uv run printsim kb ingest <path> --auto-approve` so future lookups return in <1 second without re-opening Discord.

---

## 7. Hybrid Onshape CAD + Local `build123d` Workflow

PrintSim agents can pair the **Onshape MCP server** (full cloud CAD REST access when connected) with local **`build123d` + `gmsh` + `scikit-fem`** (`printsim cad`) for an end-to-end CAD-to-FEA pipeline.

### What Agents Can Do with Onshape MCP
When the `onshape` MCP server is connected and authenticated (verify with `onshape_auth_status`):
- **Discover & triage CAD documents:** Search the user's library and public community Onshape documents (`onshape_list_documents`, `onshape_search_documents`), inspect every workspace/tab (`onshape_get_document_summary`), render server-side shaded previews (`onshape_shaded_view`), and query mass properties, BOMs, measurements, and clearances (`onshape_get_bom`, `onshape_assembly_mass_properties`, `onshape_measure`, `onshape_clearance`).
- **Configure & export STEP/STL/3MF:** Flip parametric Onshape Variables and Configurations (`onshape_get_variables`, `onshape_set_variable`, `onshape_get_configuration`, `onshape_encode_configuration`) and export both **Part Studios** and **Assemblies** straight into `cad_sources/` (`onshape_export` with `kind: "partstudios"` or `kind: "assemblies"`, `format: "STEP"`).
- **Author & modify human-editable CAD:** Create or edit Part Studios, Feature Studios / FeatureScript, sketches, extrudes, revolves, fillets, booleans, assemblies, mates, and drawings directly in the user's Onshape workspace when the user wants a live interactive CAD document they can orbit, tweak, and share in their browser.

### When to Use Onshape vs. Local `build123d`

| Task | Best Tool | Why |
| :--- | :--- | :--- |
| Finding, configuring & exporting user/community assemblies (Monolith, Voron, toolheads, hotends) | **Onshape MCP** (`onshape_export` / `onshape_set_variable`) or `printsim cad fetch-onshape --kind assemblies` | Native Onshape Configurations/Variables, BOM/mass/clearance checks, and visual triage (`onshape_shaded_view`) before downloading large STEPs. |
| Designing human-editable brackets/mounts the user wants to inspect or tweak in CAD | **Onshape MCP** (FeatureScript / Variables / Part Studios) | Produces a live browser-editable, versioned Onshape document instead of a script-only part. |
| Preparing OEM "integration" STEPs for FEA (thickening surface-only shells, hollowing keep-out blocks, cleaning micro-overlaps) | **Local `build123d`** (`cad_recipes/*.yaml` + `printsim cad bake`) | Runs in ~1–2 s locally, stays version-controlled, and feeds straight into conformal `gmsh` Tet4/Tet10 + `scikit-fem` Craig–Bampton condensation. |
| Automated parametric geometry/FEA optimization sweeps (10–100+ variants) | **Local `build123d` + `printsim sweep`** | Orders of magnitude faster than cloud API round-trips, with SHA-256 mesh caching in `.cache/cad_fea/`. |

### Practical Onshape MCP & CLI Notes
- **`onshape_list_documents`:** Keep `limit <= 20` (Onshape's API rejects `limit > 20` with HTTP 400).
- **`onshape_shaded_view`:** Always pass an explicit writable `out` path (e.g., `/tmp/onshape_preview.png`), because the server's default output path can hit a read-only root directory.
- **Assemblies vs. Part Studios:** Pass `kind: "assemblies"` on `onshape_export` / `onshape_shaded_view` (or `--kind assemblies` on `printsim cad fetch-onshape`) when targeting an Assembly tab.
- **Units:** Onshape MCP tool parameters use **inches and degrees**; exported STEP files preserve exact physical geometry and are read in **mm** by `build123d` / `printsim cad`.
- **Sandboxed agent sessions vs. host CLI:** Inside a sandboxed shell, `printsim cad fetch-onshape` cannot read `~/.onshape-mcp` or macOS Keychain unless `ONSHAPE_ACCESS_KEY`/`ONSHAPE_SECRET_KEY` are exported — use the MCP tool `onshape_export` directly instead. Outside the sandbox (user terminal), `printsim cad fetch-onshape` automatically resolves env vars, `~/.onshape-mcp/credentials.json`, or macOS Keychain.

---

## 8. Zero-LLM Local CAD & FEA Batch CLI Workflow (Token-Efficient Mode)

To avoid burning LLM turns on interactive STEP geometry probing, Gmsh re-meshing, or multi-case 3D FEA calibration loops, offload heavy CAD and sweep work to PrintSim's local cached CLIs and feed only the compact Markdown/JSON summaries back to the agent:

```bash
# 1. Inspect any STEP/STL in 1 command (solids, bbox, faces, bolt holes, overlaps, port hints)
uv run printsim cad inspect cad_sources/my_hotend.step

# 2. Bake a declarative YAML recipe with SHA-256 mesh caching + 2x2 preview PNG + QA gate
uv run printsim cad bake cad_recipes/my_hotend.yaml --fast
uv run printsim cad verify

# 3. Export STEP directly from an existing Onshape Part Studio or Assembly
#    (uses ONSHAPE_ACCESS_KEY/SECRET_KEY env vars or ~/.onshape-mcp / macOS Keychain)
uv run printsim cad fetch-onshape "https://cad.onshape.com/documents/<did>/w/<wid>/e/<eid>" --out cad_sources/part.step
uv run printsim cad fetch-onshape "https://cad.onshape.com/documents/<did>/w/<wid>/e/<eid>" --kind assemblies --out cad_sources/asm.step

# 4. Run compact physics diagnostics, A/B override deltas, and parallel SQLite-cached sweeps
uv run printsim diagnose examples/corexy_voron_style.yaml
uv run printsim compare examples/corexy_voron_style.yaml -O frame.bracing=full_truss
uv run printsim sweep sweeps/example_parametric_sweep.yaml --workers 4
```
