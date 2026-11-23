# PrintSim — Full-Featured Multiphysics 3D Printer Simulator

**PrintSim** is an engineering-grade 3D printer design, simulation, and AI optimization studio built for both **human mechanical engineers** and **AI design agents**. Design and test complete 3D printers — frames, trusses, bolted structural panels, extrusion fillings, composite linear rails, gantry brackets, belts, steppers, hotends, and custom CAD parts — before cutting a single aluminum extrusion or printing a bracket.

## What's New in v0.9.0

- **21-Tool AI Agent MCP Server & Live Studio Bridge (`printsim.mcp_server`)**: Zero-setup stdio JSON-RPC 2.0 MCP server (`printsim mcp`) for **Claude Desktop**, **Cursor**, **Antigravity / Windsurf**, and **Gemini CLI**, expanded to **21 engineering tools** including `calibrate_adxl_csv`, `fit_printer_adxl`, `analyze_sensitivity_hotspots`, `condense_bracket_superelement`, `run_deep_frame_verify`, `export_printer_dxf`, `mutate_and_compare_printer`, `design_and_test_bracket`, and live 3D Studio inspection/update tools (`get_live_studio_state`, `update_live_studio`, `capture_studio_viewport`). See **[docs/AI_AGENT_INTEGRATION.md](docs/AI_AGENT_INTEGRATION.md)** and [`examples/agent_discovery_loop.py`](examples/agent_discovery_loop.py).
- **Klipper ADXL345 CSV Measured-vs-Simulated Calibration & Digital-Twin Auto-Fit (`printsim calibrate-adxl --fit`)**: Upload or pass real Klipper `resonances_*.csv` / raw ADXL345 accelerometer logs to overlay measured vs. FEA PSD curves, quantify peak-frequency & damping deltas, and automatically fit belt pretension and rail/joint stiffness parameters to match your physical machine.
- **Industrial Belt Catalog with Theoretical/Experimental Caveats (`BELT_CATALOG`)**: Expanded belt database covering `GT2` (6–15 mm & dual 9 mm), `3GT`, `AT3` / `AT3 Gen3`, `HTD 3M`, and `AT5` with datasheet tensile stiffness, minimum pulley-tooth & back-idler bend-fatigue checks, chamber-temperature modulus retention, and explicit **Validated / Theoretical / Experimental** engineering caveats.
- **Mixed Frame Upright Profiles (`FrameSpec.upright_profile`) & VzBot 4×Ø10 mm Belt-Synced Z**: Model authentic asymmetric frames (e.g., DoomCube `4040` vertical uprights + `2020` horizontals) with separate BOM stock-bar nesting, plus VzBot 330/235 single-motor belt-synced dual-Z (`leadscrew_dual_belt_sync`) with 4×Ø10 mm hardened smooth rods and `LMK10LUU` flanged linear bearings.
- **New Community-Calibrated Presets — RatRig V-Core 3.1 300 & FLSUN V400 Delta**: Dedicated `examples/ratrig_vcore31_300.yaml` and `examples/flsun_v400_delta.yaml` builds calibrated directly against community Klipper input-shaper touchstones, alongside automatic KB preset mapping (`printsim kb calibrate`).
- **2D CNC/Laser DXF Export, One-Click Klipper `printer.cfg` & `#spec=` Shareable URLs**: Export manufacturable R2010 2D DXF panel/plate kits (`printsim export-dxf`, including `side_panel`, `rear_panel`, `top_panel`, `trident_backbone`, and `corner_gusset`), generate ready-to-paste Klipper `printer.cfg` `[input_shaper]` / `[printer]` / stepper config snippets in one click, and share any live printer build via compressed `#spec=` URLs in the 3D Web Studio.
- **Full Multiphysics Hardware Fidelity**: Bolted MITC4 shell enclosure panels, `GANTRY_BRACKET_CATALOG` creep physics, `EXTRUSION_FILLINGS` mineral-cast/granular damping, transformed-section composite linear rails, 6-DOF nozzle-to-bed structural loop, Klipper-exact PSD input shaper calibration, and `pymoo` NSGA-II 3-objective Pareto optimization (`printsim pareto`).

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
- **Interactive 3D Web Studio (`web/`)**: React + Three.js studio with live 3D modes (Assembly, Frame FEA heatmap, Animated Modal Vibration, Virtual Print toolpath scrubber, **Curved Tet10 / Voxel Hex8 Solid Part FEA** with `.step`/`.stl` upload, and **3D SIMP Topology Optimization**), **Klipper ADXL345 CSV Overlay & Auto-Fit**, **Klipper PSD & Pareto Trade-Off Frontier** charts, **BOM & Cut-List inspector**, **one-click Klipper `printer.cfg` export**, **`#spec=` shareable build URLs**, and STEP/STL/DXF/CSV downloads.

## Quickstart

### 1. CLI (Headless & AI Agent Workflows)

```bash
# Run full multiphysics evaluation on a preset (optionally specifying filament)
uv run printsim evaluate examples/voron_trident_300.yaml --filament abs
uv run printsim evaluate examples/flsun_v400_delta.yaml

# Compare & auto-fit a printer spec against real Klipper ADXL345 CSV resonance measurements
uv run printsim calibrate-adxl examples/voron_trident_300.yaml --x-csv resonances_x.csv --y-csv resonances_y.csv --fit

# Run pymoo NSGA-II multi-objective Pareto optimization (Cost vs Safe Accel vs Nozzle Flex)
uv run printsim pareto examples/voron_trident_300.yaml --pop 16 --gen 6

# Run native 21-tool stdio MCP server for Claude Desktop, Cursor, Antigravity & Gemini CLI
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

# Export a 3D STL mesh, OpenCASCADE STEP CAD assembly, or 2D CNC/laser DXF plate kit
uv run printsim export-stl examples/hyper_corexy_carbon.yaml --out hyper_corexy.stl
uv run printsim export-step examples/voron_monolith_awd.yaml --out monolith_awd.step
uv run printsim export-step --preset xy_motor_mount --out xy_motor_mount.step
uv run printsim export-dxf examples/voron_trident_300.yaml --part all_plates --out trident_plates.dxf

# Run conformal 3D Tet10 FEA (Gmsh + scikit-fem) on a B-Rep bracket or STEP file
uv run printsim tet-fea --preset xy_motor_mount --material pa12_cf_printed --vtu motor_mount.vtu
```

### 2. AI Agent & MCP Plug-and-Play Setup

Connect PrintSim directly to **Claude Desktop**, **Cursor**, **Antigravity / Windsurf**, **Gemini CLI**, or **LangGraph / OpenAI Agents** — see **[docs/AI_AGENT_INTEGRATION.md](docs/AI_AGENT_INTEGRATION.md)** for copy-paste config JSONs, all 21 MCP tool schemas, REST endpoints, and example engineering prompts.

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
