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
│  • PrintDB bridge (`schema/printdb_bridge.py`) for hardware & physics   │
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

### Phase 2 — Implemented (v0.2–v0.9) & Remaining Extensions
- **Implemented**: Conformal 3D tetrahedral continuum FEA (**Gmsh + scikit-fem** `Tet10 P2` / `Tet4 P1` in `tet_fem.py` with direct OpenCASCADE STEP meshing, vertex surface stress recovery, transversely isotropic FDM Z-layer delamination, and ParaView `.vtu` export); Wilson–Taylor **QM6-3D** Hex8 FEA (`solid_fem.py`) with feature-aligned rectilinear bracket meshing (`bracket_designer.py`) & Craig–Bampton CMS superelements (`superelement.py`); **Zero-LLM Local Hardware CAD Pipeline (`cad_assets/inspect.py`, `cad_assets/recipes.py`, `cad_assets/baker.py`, `cad_assets/registry.py`, `cad_recipes/`, `assets/hardware/`)** providing 1-command STEP/STL inspection (`printsim cad inspect`), declarative YAML recipe baking with SHA-256 `.cache/cad_fea/` caching, `--fast` coarse preview + headless 2×2 orthographic contact-sheet PNGs, automated QA gating (`printsim cad verify`), and Onshape Part Studio & Assembly STEP REST export (`printsim cad fetch-onshape --kind partstudios|assemblies` with env/Keychain auth, paired with the live Onshape MCP workflow in `docs/AI_AGENT_INTEGRATION.md` §7), condensing 30k–125k-element STEP Tet meshes into `<1 ms` runtime Craig–Bampton superelements (`AssemblyModel.superelements`, exact rigid-arm port transform, series clamp/rail coupling, static/modal/adjoint attribution) paired with decimated `visual.glb` Studio meshes (`GET /api/assets/hardware`, Three.js `GLTFLoader` with live modal/stress deformation) for `chube_compact`, `chube_air`, `chube_conduction`, `tricorn_short`, `tricorn_long`, `microwaved_bowden`, `microwaved_bowden_chube`, and coupled multi-mount braced assemblies (`assemblies/microwaved_bowden__tricorn_long`, toggleable via `toolhead.use_hotend_bottom_brace` alongside CAD-weighed `toolhead.nozzle_shield_mode` mass/CoG/mesh switching between Microwaved SLM Ti shield, stock Tricorn bucket, and bare nozzle); **Zero-LLM Offline Diagnostics & Parallel Sweep Engine (`engine/diagnostics.py`, `engine/sweep.py`)** providing `printsim diagnose`, `printsim compare`, and SQLite-cached parallel `printsim sweep` (supporting both parametric `printer` grids and `xbeam_3d_fea` 3D Tet10 calibration batches); modularized structural builder (`geometry/builder.py` + `_builder_{panels,toolhead,gantry,bed,bedslinger,delta}.py`); QM6 MITC4 bolted shell panels (`shell_mitc4.py`), semi-rigid bolted corner joint compliance, Hertzian 2-row/4-row rail & rod-bushing contact stiffness with official HIWIN/Misumi X & Y carriage block masses and BOM weights, 268k-DOF 3D orthotropic Tet10 continuum FEA-calibrated X-beam & linear-rail mounting (`scripts/calibrate_xbeam_3d_fea.py`: Carbonara two-wall clamped internal inserts `cf_rail_mounting`, 2020/CF local rail-seat compliance, partial St. Venant rail torsion, opposite-face rail+backer neutral-axis sandwich, and `dual_top_bottom` Steiner roll coupling incl. hollow-CF diamond-lozenge shear), CoreXY Y-carriage roll compliance & live Studio telemetry, Euler critical-strain bed thermal taco buckling, and plane-removed 2D bed plate FEM (`bed_plate_fem.py`).
- **Next**: Phase 2 community toolhead/hotend recipes (StealthBurner, A4T, XOL, Dragon Burner, Archetype, Dragon/Rapido/Revo), and optional nonlinear CalculiX (`pygccx`) bolted-flange contact backend.

### Phase 3 — Implemented (v0.3–v0.9) & Remaining Extensions
- **Implemented**: OpenCASCADE B-Rep parametric CAD & STEP export via **`build123d`** (`cad_b3d.py`) for full printer assemblies and brackets with real fillets, M5/M3 holes, and NEMA17 patterns; 2D CNC/laser DXF panel/plate kit export (`export-dxf`); 3D SIMP compliance topology optimization (`topology.py`) with watertight printable STL export; manufacturable extrusion cut-list with collinear bar merging, net butt-joint lengths, 1D stock bar nesting, and reconciled hardware BOM generator (`bom.py`); decomposed static/dynamic belt `EA` with exact pulley-to-pulley free-span lengths, Rayleigh reflected stepper rotor inertia, and full Delta Jacobian torque/RPM/workspace envelopes.
- **Next**: Exact T-slot cross-section property derivation via `sectionproperties`.

### Phase 4 — Coupled Thermal-Fluid (Hotend & Part Cooling CFD) & Polymer Viscoelasticity
- **Implemented**: 40-node 1D FVM axial hotend thermal solver (`hotend.py`) honoring enclosure `chamber_temp_c`, melt-averaged DSC multi-filament catalog (`filaments.py`), Arrhenius $K \cdot a_T^n$ shear-thinning viscosity & Rabinowitsch nozzle pressure drop.
- **Next**: 2D/3D CFD / surrogate airflow models for part-cooling ducts (`5015` / `4010` / CPAP blowers) predicting overhang curling and bridging sag, and Cross-WLF melt viscosity with pressure-advance transient oozing/stringing simulation.

### Phase 5 — Implemented (v0.9) & Remaining Extensions
- **Implemented**: 21-tool stdio JSON-RPC 2.0 MCP server (`printsim mcp`) with live 3D Studio inspection/update/capture bridge, upstream Klipper multi-damping input shaper calibration (`ZV`, `ZVD`, `MZV`, `EI`, `2HUMP_EI`, `3HUMP_EI`) + exact 2nd-order step response (`±5 Hz` peak-width envelope), speed VFA forcing harmonics, Carriage ADXL vs. Nozzle-Tip FEA PSD divergence diagnostics, ADXL345 CSV overlay/auto-fit, coupled ZOH 90° corner + multi-segment Zig-Zag S-curve + ISO 230-4 ballbar metrology (`ballbar.py`) with interactive Printed Wall Surface Texture Preview (`SurfaceTexturePreview.tsx`), `G2`/`G3` arc + retract G-code lookahead simulator with physical ringing/smoothing/belt-slip defects, `pymoo` NSGA-II (plus pure-NumPy fallback) 3-objective Pareto optimization (`Cost` vs `Achievable Safe Accel` vs `Inertial Nozzle Flex @ 10k`), and zero-server **Hybrid GitHub Pages Web Studio** (`src/printsim/wasm_dispatch.py`, `scripts/build_gh_pages_data.py`, `web/src/workers/pyodideWorker.ts`, `web/src/utils/pagesApiBridge.ts`, `.github/workflows/pages.yml`) pairing instant pre-baked presets & static PBR GLBs with a background Pyodide WebAssembly solver.
- **Next**: Direct Klipper `klippy` batch-mode step-pulse replay integration.

## Engineering Reference Docs
- `docs/CODEBASE_MAP.md` — Auto-generated compact module, class, function, and line-range index (`uv run python scripts/gen_codebase_map.py`).
- `docs/PHYSICS_AND_MATH.md` — Multiphysics solver derivations and analytical validation benchmarks.
- `docs/MODEL_CONFIDENCE_AND_REFERENCES.md` — Per-model engineering confidence ratings, datasheet citations, and calibration history.
- `docs/CAD_AND_FEA_PITFALLS.md` — Permanent engineering playbook & pre-finish checklist covering CAD superelement baking, rigid-arm port shifting, lumped-mass inertial splits, Hertzian rail roll mechanics, 3D carriage/4-bolt modal animation, and sidebar CSS flex invariants.
- `docs/STUDIO_STYLE_GUIDE.md` — Web Studio visual tokens, component patterns, and pre-finish UI checklist.
