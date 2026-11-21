# PrintSim — Full-Featured Multiphysics 3D Printer Simulator

**PrintSim** is an engineering-grade 3D printer design, simulation, and AI optimization studio built for both **human mechanical engineers** and **AI design agents**. Design and test complete 3D printers — frames, trusses, bolted structural panels, extrusion fillings, composite linear rails, gantry brackets, belts, steppers, hotends, and custom CAD parts — before cutting a single aluminum extrusion or printing a bracket.

## What's New in v0.8

- **Plug-and-Play AI Agent MCP Server (13 Tools) & Parametric Bracket Designer (`printsim.mcp_server`, `printsim.fea.bracket_designer`)**: Zero-setup stdio JSON-RPC 2.0 MCP server (`printsim mcp`) for **Claude Desktop**, **Cursor**, **Antigravity / Windsurf**, and **Gemini CLI**, featuring `list_catalog`, `get_preset_yaml`, `mutate_and_compare_printer` (batch dot-path comparisons with exact deltas), and `design_and_test_bracket` (parametric `corner_gusset`, `motor_mount`, and `xy_joint_l_bracket` with ~0.1s Hex8 or conformal `Tet10` FEA and STL/STEP export). See **[docs/AI_AGENT_INTEGRATION.md](docs/AI_AGENT_INTEGRATION.md)** and [`examples/agent_discovery_loop.py`](examples/agent_discovery_loop.py).
- **Gantry Bracket Materials & Creep Physics (`GANTRY_BRACKET_CATALOG`)**: Models 7 bracket materials (`abs_printed`, `pet_cf_printed`, `pa12_cf_printed`, `ppa_cf_printed`, `pps_cf_printed`, `aluminum_6061_t6`, `aluminum_7075_t6`) in series with belt + motor electromagnetic stiffness, plus sustained belt-tension creep warnings, 2WD vs. 4-motor AWD CoreXY drive, and double-shear pulley supports.
- **Frame Extrusion Cavity Fillings (`EXTRUSION_FILLINGS`)**: Models 6 cavity fillings (`none`, `pu_foam`, `silica_sand`, `lead_shot`, `epoxy_granite`, `uhpc_grout`) capturing bonded mineral-cast bending EI gains, granular friction/resin damping boosts, mass changes, and BOM cost.
- **Composite Linear Rails & Mounting Orientation**: Transformed-section bolted steel linear rail mechanics (`mgn9_rail`, `mgn12_rail`, `mgn15_rail`) across X-rail orientations (`front`, `top`, `bottom`, `dual_top_bottom`) and Y-rail orientations (`top`, `inner`, `bottom`), updating directional $EI_y/EI_z$, rail-backer flange stiffness, and nozzle-drop lever arm.
- **Nozzle-to-Bed Structural Loop & 6-DOF CoG Vibration**: Evaluates the complete frame → bed → Z-axis ↔ gantry → toolhead CoG → nozzle tip compliance loop at arbitrary `[rx, ry, rz]` print coordinates (`toolhead_rel_pos`), capturing pitch/twist amplification at the nozzle tip.
- **Klipper ADXL345 PSD Shaper Calibration (`klipper_shaper.py`)**: Synthesizes realistic axis frequency-response PSDs up to 220 Hz and runs Klipper-exact `ZV`, `MZV`, `EI`, `2HUMP_EI`, and `3HUMP_EI` smoothing/residual-vibration optimization.
- **`pymoo` NSGA-II Multi-Objective Pareto Optimizer (`pareto.py`)**: Explores the 3-objective frontier trading off **Build Cost (\$)**, **Safe Max Acceleration (`mm/s^2`)**, and **Nozzle-Tip Flex (`um`)** via `printsim pareto`, `/api/optimize/pareto`, and `optimize_printer_pareto`.

## Core Capabilities

- **Parametric Assembly DSL & OpenCASCADE STEP CAD (`printsim.schema`, `printsim.geometry.cad_b3d`)**: Declarative YAML/JSON/Python schema covering CoreXY, Cartesian Bedslinger, Delta, and arbitrary custom node-and-member trusses, backed by real datasheet materials and profiles — plus **`build123d` (OpenCASCADE B-Rep) STEP export** for full printer assemblies and parametric brackets with real fillets, M5/M3 bolt holes, and NEMA17 mounting patterns.
- **3D Frame FEA, Curved Tet10 Continuum FEA, Solid Hex8 FEA & SIMP Topology Optimization (`printsim.fea`)**:
  - Exact 12-DOF 3D Euler-Bernoulli frame FEM solver (`K` stiffness & consistent `M` mass matrices, verified against textbook analytical cantilever deflection & frequency benchmarks).
  - **Conformal 3D Tetrahedral Continuum FEA (`tet_fem.py`)** powered by **Gmsh + scikit-fem** (`Tet10 P2` / `Tet4 P1`) directly on OpenCASCADE STEP files or parametric brackets, supporting isotropic metals & transversely isotropic FDM Z-layer delamination, smooth Three.js surface stress rendering, and ParaView `.vtu` export.
  - **3D Solid Hex8 Continuum FEM (`solid_fem.py`)** and **3D SIMP Topology Optimization (`topology.py`)** with sensitivity filtering, boundary preservation, and watertight printable STL export.
  - Generalized eigenvalue modal analysis ($K\phi = \omega^2 M\phi$) → 3D animated vibration modes, dominant X/Y resonance frequencies, coupled belt+motor+bracket+structure resonance, and Klipper Input Shaper recommendations.
- **Manufacturable Cut-List, Stock Nesting & BOM (`printsim.geometry.bom`)**: Auto-merges split structural members into physical cut lengths, runs 1D stock bar nesting (`1000mm`/`2000mm` bars + saw kerf & scrap `%`), calculates exact belt/rail/motor/bracket/filling/panel/fastener counts and cost breakdowns, and exports to **JSON, CSV, or Markdown**.
- **Kinematics & Clearance (`printsim.kinematics`)**: CoreXY, Cartesian, and Delta exact FK/IK, stepper electromagnetic torque roll-off at high RPM (`24V`/`48V` + inductance), belt dynamic slack-side slip & overtension checks, packaging efficiency, and frame/toolhead collision detection.
- **Hotend FVM Thermal, Multi-Filament Rheology & Cooling (`printsim.thermal`)**: 40-node 1D Finite-Volume axial temperature solver $T(z)$ (Nozzle → Heater Block → Heatbreak → Heatsink), multi-filament database (`PLA`, `PETG`, `ABS`, `ASA`, `PA12-CF`, `TPU 95A`) with shear-thinning viscosity & $T_g$ heat-creep thresholds, transient heat-up curves, and overhang/bridging cooling predictions.
- **G-code Lookahead & Virtual Print (`printsim.gcode`)**: Parses standard Marlin/Klipper/OrcaSlicer G-code (or runs the built-in high-speed calibration cube benchmark), computes lookahead trapezoidal motion timings, and tags every toolpath segment with physical defects (`ok`, `ringing`, `underextrusion`, `step_skip`).
- **Interactive 3D Web Studio (`web/`)**: React + Three.js studio with live 3D modes (Assembly, Frame FEA heatmap, Animated Modal Vibration, Virtual Print toolpath scrubber, **Curved Tet10 / Voxel Hex8 Solid Part FEA** with `.step`/`.stl` upload, and **3D SIMP Topology Optimization**), **Klipper PSD & Pareto Trade-Off Frontier** charts, **BOM & Cut-List inspector**, and one-click STEP/STL/CSV downloads.

## Quickstart

### 1. CLI (Headless & AI Agent Workflows)

```bash
# Run full multiphysics evaluation on a preset (optionally specifying filament)
uv run printsim evaluate examples/voron_trident_300.yaml --filament abs

# Run pymoo NSGA-II multi-objective Pareto optimization (Cost vs Safe Accel vs Nozzle Flex)
uv run printsim pareto examples/voron_trident_300.yaml --pop 16 --gen 6

# Run native 13-tool stdio MCP server for Claude Desktop, Cursor, Antigravity & Gemini CLI
uv run printsim mcp

# Run autonomous 2-stage AI agent discovery & parametric bracket FEA demo
uv run python examples/agent_discovery_loop.py

# Run 3D Solid Hex8 FEA on a bracket preset or custom STL file
uv run printsim solid-fea --preset xy_motor_mount --material pa12_cf_printed
uv run printsim solid-fea --stl my_bracket.stl --material abs_printed --fx 80 --fz -40

# Run 3D SIMP topology optimization on a bracket & export watertight STL
uv run printsim topo-opt --preset corner_gusset --volume-fraction 0.45 --material pa12_cf_printed --out opt_gusset.stl

# Generate manufacturable cut-list, stock bar nesting & full hardware BOM
uv run printsim bom examples/voron_trident_300.yaml --csv trident_bom.csv

# Machine-readable JSON output for AI agents
uv run printsim evaluate examples/corexy_voron_style.yaml --json --no-segments

# Run evolutionary AI design optimization
uv run printsim optimize examples/cartesian_bedslinger.yaml --iterations 30 --objective rigidity --out optimized.yaml

# Export a 3D STL mesh or exact OpenCASCADE STEP CAD assembly
uv run printsim export-stl examples/hyper_corexy_carbon.yaml --out hyper_corexy.stl
uv run printsim export-step examples/voron_monolith_awd.yaml --out monolith_awd.step
uv run printsim export-step --preset xy_motor_mount --out xy_motor_mount.step

# Run conformal 3D Tet10 FEA (Gmsh + scikit-fem) on a B-Rep bracket or STEP file
uv run printsim tet-fea --preset xy_motor_mount --material pa12_cf_printed --vtu motor_mount.vtu
```

### 2. AI Agent & MCP Plug-and-Play Setup

Connect PrintSim directly to **Claude Desktop**, **Cursor**, **Antigravity / Windsurf**, **Gemini CLI**, or **LangGraph / OpenAI Agents** — see **[docs/AI_AGENT_INTEGRATION.md](docs/AI_AGENT_INTEGRATION.md)** for copy-paste config JSONs, all 13 MCP tool schemas, REST endpoints, and example engineering prompts.

### 3. Launch the Interactive 3D Web Studio

```bash
# Build frontend once (if not already built)
cd web && npm install && npm run build && cd ..

# Start FastAPI backend + Web 3D Studio on http://127.0.0.1:8000
uv run printsim serve --port 8000
```

For frontend hot-reload development, run `uv run printsim serve --port 8000` in one terminal and `cd web && npm run dev` in another (proxies `/api` to port 8000).

### 4. Run the Physics Verification Suite

```bash
uv run pytest -v
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for architecture details and [docs/monolith_discord_benchmarks.md](docs/monolith_discord_benchmarks.md) for empirical CoreXY input-shaper calibration benchmarks.
