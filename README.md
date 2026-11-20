# PrintSim — Full-Featured Multiphysics 3D Printer Simulator

**PrintSim** is an engineering-grade 3D printer design, simulation, and AI optimization studio built for both **human mechanical engineers** and **AI design agents**. Design and test complete 3D printers — frames, trusses, gantries, belts, steppers, hotends, and custom bodies — before cutting a single aluminum extrusion or printing a bracket.

## What's Included (v0.4)

- **Parametric Assembly DSL & OpenCASCADE STEP CAD (`printsim.schema`, `printsim.geometry.cad_b3d`)**: Declarative YAML/JSON/Python schema covering CoreXY, Cartesian Bedslinger, Delta, and arbitrary custom node-and-member trusses, backed by real datasheet materials and profiles — plus **`build123d` (OpenCASCADE B-Rep) STEP export** for full printer assemblies and parametric brackets with real fillets, M5/M3 bolt holes, and NEMA17 mounting patterns.
- **3D Frame FEA, Curved Tet10 Continuum FEA, Solid Hex8 FEA & SIMP Topology Optimization (`printsim.fea`)**:
  - Exact 12-DOF 3D Euler-Bernoulli frame FEM solver (`K` stiffness & consistent `M` mass matrices, verified against textbook analytical cantilever deflection & frequency benchmarks).
  - **Conformal 3D Tetrahedral Continuum FEA (`tet_fem.py`)** powered by **Gmsh + scikit-fem** (`Tet10 P2` / `Tet4 P1`) directly on OpenCASCADE STEP files or parametric brackets, supporting isotropic metals & transversely isotropic FDM Z-layer delamination, smooth Three.js surface stress rendering, and ParaView `.vtu` export.
  - **3D Solid Hex8 Continuum FEM (`solid_fem.py`)** and **3D SIMP Topology Optimization (`topology.py`)** with sensitivity filtering, boundary preservation, and watertight printable STL export.
  - Generalized eigenvalue modal analysis ($K\phi = \omega^2 M\phi$) → 3D animated vibration modes, dominant X/Y resonance frequencies, coupled belt+structure resonance, and Klipper Input Shaper (`ZV`, `MZV`, `EI`) max acceleration recommendations.
- **Manufacturable Cut-List, Stock Nesting & BOM (`printsim.geometry.bom`)**: Auto-merges split structural members into physical cut lengths, runs 1D stock bar nesting (`1000mm`/`2000mm` bars + saw kerf & scrap `%`), calculates exact belt/rail/motor/enclosure/fastener counts and cost breakdowns, and exports to **JSON, CSV, or Markdown**.
- **Kinematics & Clearance (`printsim.kinematics`)**: CoreXY, Cartesian, and Delta exact FK/IK, stepper electromagnetic torque roll-off at high RPM, belt dynamic slack-side slip checks, packaging efficiency, and frame/toolhead collision detection.
- **Hotend FVM Thermal, Multi-Filament Rheology & Cooling (`printsim.thermal`)**: 40-node 1D Finite-Volume axial temperature solver $T(z)$ (Nozzle → Heater Block → Heatbreak → Heatsink), multi-filament database (`PLA`, `PETG`, `ABS`, `ASA`, `PA12-CF`, `TPU 95A`) with shear-thinning viscosity & $T_g$ heat-creep thresholds, transient heat-up curves, and overhang/bridging cooling predictions.
- **G-code Lookahead & Virtual Print (`printsim.gcode`)**: Parses standard Marlin/Klipper/OrcaSlicer G-code (or runs the built-in high-speed calibration cube benchmark), computes lookahead trapezoidal motion timings, and tags every toolpath segment with physical defects (`ok`, `ringing`, `underextrusion`, `step_skip`).
- **Headless AI Harness, CLI & Native MCP Server (`printsim.engine`, `printsim.cli`, `printsim.mcp_server`)**: Single-call JSON multiphysics evaluator, evolutionary design optimizer (`balanced`, `speed`, `rigidity`, `budget`), STL/STEP CAD export, Tet10/Hex8 FEA, topology optimization & BOM CLI commands, and a 9-tool stdio JSON-RPC 2.0 MCP server (`printsim mcp`).
- **Interactive 3D Web Studio (`web/`)**: React + Three.js studio with 6 live 3D modes (Assembly, Frame FEA heatmap, Animated Modal Vibration, Virtual Print toolpath scrubber, **Curved Tet10 / Voxel Hex8 Solid Part FEA** with `.step`/`.stl` upload, and **3D SIMP Topology Optimization**), **BOM & Cut-List inspector**, and one-click STEP/STL/CSV downloads.

## Quickstart

### 1. CLI (Headless & AI Agent Workflows)

```bash
# Run full multiphysics evaluation on a preset (optionally specifying filament)
uv run printsim evaluate examples/corexy_voron_style.yaml --filament abs

# Run 3D Solid Hex8 FEA on a bracket preset or custom STL file
uv run printsim solid-fea --preset xy_motor_mount --material pa12_cf_printed
uv run printsim solid-fea --stl my_bracket.stl --material abs_printed --fx 80 --fz -40

# Run 3D SIMP topology optimization on a bracket & export watertight STL
uv run printsim topo-opt --preset corner_gusset --volume-fraction 0.45 --material pa12_cf_printed --out opt_gusset.stl

# Generate manufacturable cut-list, stock bar nesting & full hardware BOM
uv run printsim bom examples/corexy_voron_style.yaml --csv voron_bom.csv

# Run native MCP stdio server for AI agents
uv run printsim mcp

# Machine-readable JSON output for AI agents
uv run printsim evaluate examples/corexy_voron_style.yaml --json --no-segments

# Run evolutionary AI design optimization
uv run printsim optimize examples/cartesian_bedslinger.yaml --iterations 30 --objective rigidity --out optimized.yaml

# Export a 3D STL mesh or exact OpenCASCADE STEP CAD assembly
uv run printsim export-stl examples/hyper_corexy_carbon.yaml --out hyper_corexy.stl
uv run printsim export-step examples/corexy_voron_style.yaml --out voron_frame.step
uv run printsim export-step --preset xy_motor_mount --out xy_motor_mount.step

# Run conformal 3D Tet10 FEA (Gmsh + scikit-fem) on a B-Rep bracket or STEP file
uv run printsim tet-fea --preset xy_motor_mount --material pa12_cf_printed --vtu motor_mount.vtu
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
