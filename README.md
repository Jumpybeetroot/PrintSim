# PrintSim — Full-Featured Multiphysics 3D Printer Simulator

**PrintSim** is an engineering-grade 3D printer design, simulation, and AI optimization studio built for both **human mechanical engineers** and **AI design agents**. Design and test complete 3D printers — frames, trusses, gantries, belts, steppers, hotends, and custom bodies — before cutting a single aluminum extrusion or printing a bracket.

## What's Included in v0.1 MVP

- **Parametric Assembly DSL (`printsim.schema`)**: Declarative YAML/JSON/Python schema covering CoreXY, Cartesian Bedslinger, Delta, and arbitrary custom node-and-member trusses, backed by real datasheet materials (Al 6063/6061, MIC-6, Carbon Fiber Tube, 4140 Steel, Ti-6Al-4V, Brass, Plated Copper, ABS, PA12-CF) and structural cross-sections (2020/2040/3030/4040 extrusions, CF square tubes, MGN9/12/15 rails, linear rods).
- **3D Finite Element Analysis (`printsim.fea`)**:
  - Exact 12-DOF 3D Euler-Bernoulli frame FEM solver (`K` stiffness & consistent `M` mass matrices, verified against textbook analytical cantilever deflection & frequency benchmarks).
  - Static FEA under gravity sag and peak X/Y toolhead acceleration inertial loads → nodal displacements, per-member von Mises stress, safety factors, and directional toolhead stiffness (N/mm).
  - Generalized eigenvalue modal analysis ($K\phi = \omega^2 M\phi$) → 3D animated vibration modes, dominant X/Y resonance frequencies, coupled belt+structure resonance, and Klipper Input Shaper (`ZV`, `MZV`, `EI`) max acceleration recommendations.
- **Kinematics & Clearance (`printsim.kinematics`)**: CoreXY, Cartesian, and Delta exact FK/IK, stepper electromagnetic torque roll-off at high RPM, belt dynamic slack-side slip checks, packaging efficiency, and frame/toolhead collision detection.
- **Hotend Thermal & Extrusion (`printsim.thermal`)**: Multi-node thermal network (heater block, bi-metal/stainless heatbreak conduction, heatsink thermal resistance → cold-throat heat-creep risk), transient heat-up curves, and max volumetric flow rate ($\text{mm}^3/\text{s}$) across heater power, melt-zone conduction, and extruder pressure-drop limits.
- **G-code Lookahead & Virtual Print (`printsim.gcode`)**: Parses standard Marlin/Klipper/OrcaSlicer G-code (or runs the built-in high-speed calibration cube benchmark), computes lookahead trapezoidal motion timings, and tags every toolpath segment with physical defects (`ok`, `ringing`, `underextrusion`, `step_skip`).
- **Headless AI Harness & CLI (`printsim.engine`, `printsim.cli`)**: Single-call JSON multiphysics evaluator, evolutionary design optimizer (`balanced`, `speed`, `rigidity`, `budget`), and 3D STL mesh export.
- **Interactive 3D Web Studio (`web/`)**: React + Three.js studio with live Assembly, FEA Stress & Deflection heatmap, animated 3D Modal Vibration modes, Virtual Print toolpath defect scrubber, YAML/G-code editors, and one-click AI design optimization.

## Quickstart

### 1. CLI (Headless & AI Agent Workflows)

```bash
# Run full multiphysics evaluation on a preset
uv run printsim evaluate examples/corexy_voron_style.yaml

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
