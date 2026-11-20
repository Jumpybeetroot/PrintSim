# PrintSim Architecture & Full-Feature Roadmap

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Human UI & AI Agent Interfaces                      │
│  • Interactive 3D Web Studio (React + Three.js)                         │
│  • Headless CLI (`printsim evaluate / optimize / export-stl`)           │
│  • REST API (`/api/simulate`, `/api/optimize`, `/api/export/stl`)       │
│  • Python SDK (`evaluate_printer`, `optimize_design`)                   │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │ PrinterSpec (YAML / JSON / Python)
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  Parametric Geometry & CAD Kernel                       │
│  • Structural graph builder (CoreXY, Cartesian, Delta, Custom Trusses)  │
│  • Engineering material & beam cross-section catalogs                   │
│  • Mesh / STL / OBJ I/O (trimesh) + optional STEP/B-Rep (build123d/OCP) │
└───────┬──────────────────┬──────────────────┬───────────────────┬───────┘
        ▼                  ▼                  ▼                   ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐   ┌───────────────┐
│ Structural    │  │ Kinematics &  │  │ Thermal &     │   │ G-code &      │
│ FEA & Modal   │  │ Dynamics      │  │ Extrusion     │   │ Virtual Print │
│               │  │               │  │               │   │               │
│ • 12-DOF 3D   │  │ • FK/IK       │  │ • Multi-node  │   │ • Lookahead   │
│   Beam FEM    │  │ • Stepper L/R │  │   thermal net │   │   trapezoid   │
│ • Static sag  │  │   torque      │  │ • Heat creep  │   │ • Flow limit  │
│   & stress    │  │ • Belt slip   │  │ • Melt-zone   │   │ • Ringing &   │
│ • Eigenmodes  │  │ • Collisions  │  │   vol. flow   │   │   ghosting    │
│ • Input shaper│  │               │  │ • PID step    │   │ • Cooling     │
└───────┬───────┘  └───────┬───────┘  └───────┬───────┘   └───────┬───────┘
        └──────────────────┴──────────────────┴───────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│            Unified Evaluation Scorecard & AI Optimizer                  │
│  • Composite engineering fitness + actionable diagnostics               │
│  • Evolutionary / Bayesian / RL design optimization loop                │
└─────────────────────────────────────────────────────────────────────────┘
```

## Roadmap to Full Industrial Fidelity

### Phase 1 — Implemented (v0.1 MVP)
- Exact 12-DOF 3D Euler-Bernoulli frame FEA (static stress, gravity sag, inertial deflection, modal eigenvalues/eigenvectors, Klipper input shaper tuning).
- CoreXY, Cartesian Bedslinger, Delta, and custom node/member truss assemblies.
- Stepper torque-speed curve, belt dynamics, frame/envelope collision checks.
- Hotend thermal resistance network, heat-creep detection, melt-zone volumetric flow model.
- G-code lookahead motion planner & per-segment virtual print defect simulator.
- Interactive Three.js Web Studio + CLI + REST API + STL export + evolutionary AI optimizer.

### Phase 2 — Solid 3D Tetrahedral FEA & Custom Part Meshes
- Integrate **Gmsh + CalculiX / FEniCSx / SfePy** as a pluggable `SolidFEABackend` for 3D printed toolhead mounts, XY motor brackets, and CNC gantry plates imported from STEP/STL.
- Orthotropic material models for FDM-printed parts (weak Z inter-layer tensile strength vs XY strength, infill density).
- Contact stiffness modeling for bolted extrusion corner joints and linear rail preload.

### Phase 3 — Parametric Manufacturable CAD & Topology Optimization
- Full **build123d / OpenCASCADE (OCP)** export of cut-list BOM, drilled extrusion drawings, and ready-to-print STL/STEP brackets.
- SIMP / truss topology optimization driven directly by the FEA compliance & modal frequency gradients.

### Phase 4 — Coupled Thermal-Fluid (Hotend & Part Cooling CFD) & Polymer Viscoelasticity
- 2D/3D finite-volume heat transfer inside heater blocks, bi-metal heatbreaks, and CHT multi-channel nozzles.
- Simplified CFD / surrogate airflow models for part-cooling ducts (`5015` / `4010` / CPAP blowers) predicting overhang curling and bridging sag.
- Non-Newtonian Cross-WLF melt viscosity and pressure-advance / linear-advance transient oozing & stringing simulation.

### Phase 5 — Firmware-in-the-Loop & Multi-Agent AI Design Co-Pilot
- Direct Klipper `klippy` batch-mode integration to replay real step-pulse streams and resonance shaper filters.
- MCP (Model Context Protocol) tool server exposing `evaluate_printer`, `sweep_parameters`, `inspect_mode_shape`, and `mutate_frame` so external LLM agents can iteratively invent new printer architectures autonomously.
