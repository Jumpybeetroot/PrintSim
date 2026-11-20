# PrintSim — Full-Featured Multiphysics 3D Printer Simulator

**PrintSim** is an engineering-grade 3D printer design, simulation, and AI optimization studio built for both **human mechanical engineers** and **AI design agents**. Design and test complete 3D printers — frames, trusses, gantries, belts, steppers, hotends, and custom bodies — before cutting a single aluminum extrusion or printing a bracket.

## What's Included (v0.2)

- **Parametric Assembly DSL (`printsim.schema`)**: Declarative YAML/JSON/Python schema covering CoreXY, Cartesian Bedslinger, Delta, and arbitrary custom node-and-member trusses, backed by real datasheet materials (Al 6063/6061, MIC-6, Carbon Fiber Tube, 4140 Steel, Ti-6Al-4V, Brass, Plated Copper, ABS, PA12-CF) and structural cross-sections (2020/2040/3030/4040 extrusions, CF square tubes, MGN9/12/15 rails, linear rods).
- **3D Frame & Solid Continuum FEA (`printsim.fea`)**:
  - Exact 12-DOF 3D Euler-Bernoulli frame FEM solver (`K` stiffness & consistent `M` mass matrices, verified against textbook analytical cantilever deflection & frequency benchmarks).
  - **3D Solid Hex8 Continuum FEM (`solid_fem.py`)** with SciPy sparse direct solver & transversely isotropic FDM Z-layer delamination model — runs on parametric brackets/mounts (`corner_gusset`, `xy_motor_mount`, `toolhead_carriage`) or **any uploaded STL/OBJ mesh**.
  - Generalized eigenvalue modal analysis ($K\phi = \omega^2 M\phi$) → 3D animated vibration modes, dominant X/Y resonance frequencies, coupled belt+structure resonance, and Klipper Input Shaper (`ZV`, `MZV`, `EI`) max acceleration recommendations.
- **Kinematics & Clearance (`printsim.kinematics`)**: CoreXY, Cartesian, and Delta exact FK/IK, stepper electromagnetic torque roll-off at high RPM, belt dynamic slack-side slip checks, packaging efficiency, and frame/toolhead collision detection.
- **Hotend FVM Thermal, Multi-Filament Rheology & Cooling (`printsim.thermal`)**: 40-node 1D Finite-Volume axial temperature solver $T(z)$ (Nozzle → Heater Block → Heatbreak → Heatsink), multi-filament database (`PLA`, `PETG`, `ABS`, `ASA`, `PA12-CF`, `TPU 95A`) with shear-thinning viscosity & $T_g$ heat-creep thresholds, transient heat-up curves, and overhang/bridging cooling predictions.
- **G-code Lookahead & Virtual Print (`printsim.gcode`)**: Parses standard Marlin/Klipper/OrcaSlicer G-code (or runs the built-in high-speed calibration cube benchmark), computes lookahead trapezoidal motion timings, and tags every toolpath segment with physical defects (`ok`, `ringing`, `underextrusion`, `step_skip`).
- **Headless AI Harness, CLI & Native MCP Server (`printsim.engine`, `printsim.cli`, `printsim.mcp_server`)**: Single-call JSON multiphysics evaluator, evolutionary design optimizer (`balanced`, `speed`, `rigidity`, `budget`), STL mesh export, and a stdio JSON-RPC 2.0 MCP server (`printsim mcp`) pre-wired in `.agents/plugins/printsim-tools/`.
- **Interactive 3D Web Studio (`web/`)**: React + Three.js studio with 5 live 3D modes (Assembly, Frame FEA heatmap, Animated Modal Vibration, Virtual Print toolpath scrubber, and **3D Solid Hex8 Part/STL FEA**), axial FVM hotend chart, filament switcher, YAML/G-code editors, and one-click AI design optimization.

## Quickstart

### 1. CLI (Headless & AI Agent Workflows)

```bash
# Run full multiphysics evaluation on a preset (optionally specifying filament)
uv run printsim evaluate examples/corexy_voron_style.yaml --filament abs

# Run 3D Solid Hex8 FEA on a bracket preset or custom STL file
uv run printsim solid-fea --preset xy_motor_mount --material pa12_cf_printed
uv run printsim solid-fea --stl my_bracket.stl --material abs_printed --fx 80 --fz -40

# Run native MCP stdio server for AI agents
uv run printsim mcp

# Machine-readable JSON output for AI agents
uv run printsim evaluate examples/corexy_voron_style.yaml --json --no-segments

# Run evolutionary AI design optimization
uv run printsim optimize examples/cartesian_bedslinger.yaml --iterations 30 --objective rigidity --out optimized.yaml

# Export a 3D STL mesh of the printer assembly
uv run printsim export-stl examples/hyper_corexy_carbon.yaml --out hyper_corexy.stl
```

### 2. Launch the Interactive 3D Web Studio

```bash
# Build frontend once (if not already built)
cd web && npm install && npm run build && cd ..

# Start FastAPI backend + Web 3D Studio on http://127.0.0.1:8000
uv run printsim serve --port 8000
```

For frontend hot-reload development, run `uv run printsim serve --port 8000` in one terminal and `cd web && npm run dev` in another (proxies `/api` to port 8000).

### 3. Run the Physics Verification Suite

```bash
uv run pytest -v
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full roadmap toward solid 3D tetrahedral FEA (Gmsh/CalculiX/FEniCSx), STEP CAD generation, CFD part cooling, and Klipper firmware-in-the-loop simulation.
