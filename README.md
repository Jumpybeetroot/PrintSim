# PrintSim — Full-Featured Multiphysics 3D Printer Simulator

[![Live Web Studio](https://img.shields.io/badge/Launch%203D%20Web%20Studio-0a78bd?style=for-the-badge&logo=githubpages&logoColor=white)](https://jumpybeetroot.github.io/PrintSim/)
[![Support on Ko-fi](https://img.shields.io/badge/Support%20on%20Ko--fi-FF5E5B?style=for-the-badge&logo=ko-fi&logoColor=white)](https://ko-fi.com/jumpybeetroot)

**PrintSim** is an engineering-grade 3D printer design, simulation, and AI optimization studio built for both **human mechanical engineers** and **AI design agents**. Try the interactive 3D Studio directly in your browser at **[jumpybeetroot.github.io/PrintSim](https://jumpybeetroot.github.io/PrintSim/)** (hybrid instant pre-baked presets + in-browser Pyodide WebAssembly FEA engine — zero install required), or run locally with full native Gmsh Tet10, OpenCASCADE STEP, and MCP agent support. Design and test complete 3D printers — frames, trusses, bolted structural panels, extrusion fillings, composite linear rails, gantry brackets, belts, steppers, hotends, and custom CAD parts — before cutting a single aluminum extrusion or printing a bracket.

## What's New in v0.9.0

- **21-Tool AI Agent MCP Server & Live Studio Bridge (`printsim.mcp_server`)**: Zero-setup stdio JSON-RPC 2.0 MCP server (`printsim mcp`) for **Claude Desktop**, **Cursor**, **Antigravity / Windsurf**, and **Gemini CLI**, expanded to **21 engineering tools** including `calibrate_adxl_csv`, `fit_printer_adxl`, `analyze_sensitivity_hotspots`, `condense_bracket_superelement`, `run_deep_frame_verify`, `export_printer_dxf`, `mutate_and_compare_printer`, `design_and_test_bracket`, and live 3D Studio inspection/update tools (`get_live_studio_state`, `update_live_studio`, `capture_studio_viewport`). See **[docs/AI_AGENT_INTEGRATION.md](docs/AI_AGENT_INTEGRATION.md)** and [`examples/agent_discovery_loop.py`](examples/agent_discovery_loop.py).
- **Klipper ADXL345 CSV Measured-vs-Simulated Calibration & Digital-Twin Auto-Fit (`printsim calibrate-adxl --fit`)**: Upload or pass real Klipper `resonances_*.csv` / raw ADXL345 accelerometer logs to overlay measured vs. FEA PSD curves, quantify peak-frequency & damping deltas, and automatically fit belt pretension and rail/joint stiffness parameters to match your physical machine.
- **Industrial Belt Catalog with Theoretical/Experimental Caveats (`BELT_CATALOG`)**: Expanded belt database covering `GT2` (6–15 mm & dual 9 mm), `3GT`, `AT3` / `AT3 Gen3`, `HTD 3M`, and `AT5` with datasheet tensile stiffness, minimum pulley-tooth & back-idler bend-fatigue checks, chamber-temperature modulus retention, and explicit **Validated / Theoretical / Experimental** engineering caveats.
- **Official-Spec VzBot 330 AWD Full Build (`examples/vzbot_330_awd.yaml`, `gantry.hardware_kit: vzbot_330_awd`)**: Rebuilt from the official VzBoT3D BOM, `Vz-330_v1.2` full-printer STEP and Klipper config — 530×500×570 mm frame with `2040` uprights, Y rails bolted to the inner faces of the top-ring Y extrusions (new `frame.gantry_on_top_frame` topology: no phantom gantry ring), 3 mm Al rear + polycarbonate sides (new `frame.side_panel_material`), CAD-baked Tet4 Craig–Bampton rear motor mounts, AWD/2WD front mounts and XY joints (`scripts/bake_vzbot_kit.py`), 20×20×1 CF tube with top MGN9H, 6 mm GT2 at ~171 Hz/150 mm (22 N), 4× LDO 2804AC @ 48 V, the CAD-measured `vz_printhead_cnc` toolhead (343 g dressed, nozzle −54.2 mm), 8 mm cast bed on bolted silicone spacers, and belt-synced dual TR8x8 Z. Simulates X ≈ 106 Hz / Y ≈ 65 Hz, inside the cited community Klipper band (X 75–118, Y 47–80 Hz). Selectable in the Studio **Base Machine** picker as *VzBot 330 AWD (Mellow CNC)*.
- **Mixed Frame Upright Profiles (`FrameSpec.upright_profile`) & VzBot 4×Ø10 mm Belt-Synced Z**: Model authentic asymmetric frames (e.g., DoomCube `4040` vertical uprights + `2020` horizontals) with separate BOM stock-bar nesting, plus VzBot 330/235 single-motor belt-synced dual-Z (`leadscrew_dual_belt_sync`) with 4×Ø10 mm hardened smooth rods and `LMK10LUU` flanged linear bearings.
- **Base Machine Kit Chooser & Coordinated Gantry Upgrade Sync (`src/printsim/presets.py`, `GANTRY_HARDWARE_KIT_CATALOG`, `apply_gantry_hardware_kit`, `BaseMachineModal.tsx`)**: On startup (and anytime from the left-sidebar **Base Machine** picker), users choose what their machine is based on — **Siboor Kit** (`Siboor Voron Trident AWD 300`, `Siboor Voron 2.4r2 CNC 300`), **LDO Kit** (`LDO Voron Trident Rev D 300`, `LDO Voron 2.4r2 Rev D 300`), or **DIY / Self-Sourced Build** (Voron Trident / 2.4r2 printed, Monolith AWD, Custom CoreXY, Bedslinger, Delta) — which configures the complete matching gantry kit, belt routing, X/Y rail sizes & orientations, Z drive/rails/bed mounts, and Trident `110 mm` vs. V2.4 `55 mm` top-to-gantry headroom. Switching **Gantry Kit / Upgrade** (e.g., upgrading from Siboor CNC AWD or LDO Voron Kit to **Monolith × LDO CNC AWD/2WD**) atomically updates every coupled physical parameter (drive mode, belt routing, bracket material, double-shear pulleys, rail size & orientation, belt width/tension, carriage mass, M5 disc-spring axial preload, and milled V2.4 Z-joints) while allowing individual tweaks afterward.
- **Community-Calibrated Test & Benchmark Fixtures**: Dedicated `examples/monolith_zero_m0.yaml` (`1515` frame, bare unsupported `MGN12H` X-rail, `WS7040` CPAP, `100W` PTC + `10 mm` PIR), `examples/monolith_sheet_metal_v2.yaml` (`3–5 mm` laser-cut steel plates, Hollow `MGN12H`, Double SLM V2 Z-joints, `2 mm` PTFE stepper gaskets, `M5` disc-spring preload, `500W` PTC + `20 mm` PIR, `WM9290` CPAP, `Cra-Z` kinematic bed mount), `examples/ratrig_vcore31_300.yaml`, and `examples/flsun_v400_delta.yaml` builds remain available for offline testing & touchstone calibration against PrintDB (`392` validated records, including 6 SLM/LPBF printed metal alloys with distinct roughness/contact compliance, mild build-Z anisotropy, and alloy-specific CTE press-fit physics vs CNC billet).
- **Chamber Thermal, CPAP Blower/Hose & Thermal-Mechanical Coupling (`printsim.thermal.chamber`, `fea/components.py`)**: Models `100W`/`500W` active PTC chamber heaters + `10/20 mm` PIR insulation heat-up & steady-state temps, `WS7040`/`WM9290`/`WM7575` CPAP blowers (`15/18 mm` Darcy-Weisbach hose loss, `+5.5 °C` exhaust heating, Kalico shared-duct minimum PWM), stepper self-heating + `2 mm` PTFE mount isolation (`-14 °C`), `F695`-in-Al CTE press-fit clearance warning (`>66 °C`), belt-plane CoG X-roll diagnostics, and `GT2 10mm` vs `GT3 10mm` elastomer seating/drag.
- **2D CNC/Laser DXF Export, One-Click Klipper `printer.cfg` & `#spec=` Shareable URLs**: Export manufacturable R2010 2D DXF panel/plate kits (`printsim export-dxf`, including `side_panel`, `rear_panel`, `top_panel`, `trident_backbone`, and `corner_gusset`), generate ready-to-paste Klipper `printer.cfg` `[input_shaper]` / `[printer]` / stepper config snippets in one click, and share any live printer build via compressed `#spec=` URLs in the 3D Web Studio.
- **CAD-Backed Craig–Bampton Hardware Superelements & Studio 3D GLB Assets (`printsim.cad_assets`, `use_cad_superelements`)**: Bake real vendor `.STEP` / `.STL` CAD once offline (`printsim bake-asset` / `printsim list-assets` / `scripts/bake_voron_kits.py`) via Gmsh/voxel Tet4 + `scikit-fem` + VPT/RBE3 Craig–Bampton CMS into compact `superelement.npz` matrices (~10 KB) + QEM-decimated `visual.glb` PBR meshes. At runtime, PrintSim injects exact 3D coupled bending/torsion/shear stiffness into global static, modal, and adjoint FEA solvers in `< 1 ms`, renders the real structural CAD geometry in Three.js with live modal/stress deformation (`/api/assets/hardware`), and elevates confidence with a `CAD CMS` badge. Covers **50 structural CMS bundles**: hotends (**Chube Compact / Air / Conduction**, **Tricorn Short / Long**), toolheads (**Voron StealthBurner + Clockwork 2.5**, **El Fuego SLM**, **Microwaved Bowden SLM & Chube V1**, plus braced `assemblies/microwaved_bowden__tricorn_long`), and **40 gantry & Z hardware stations (`gantry.hardware_kit`)** extracted directly in assembled world coordinates from official full-printer STEP assemblies for **LDO Monolith CNC AWD & 2WD** (incl. milled V2.4 Z-joint), **Siboor / GKD CNC AWD & dedicated 2WD** (rear/front/XY), and **Stock Printed Voron 2.4r2 & Trident / LDO Rev D Kit** (A/B drives, front idlers, XY joints, belted quad-Z drives & Z-joints, and per-station Left/Right/Rear Trident Z mounts & GE5C bed arms), alongside official kit reference YAMLs (`examples/ldo_voron_24_revd_300.yaml`, `examples/ldo_voron_trident_revd_300.yaml`, `examples/siboor_voron_trident_awd_300.yaml`, `examples/siboor_voron_24_cnc_300.yaml`).
- **Full Multiphysics Hardware Fidelity, Advanced Mode Lumped-Model Overrides & Model Confidence Audit (`model_overrides`, `model_confidence`)**: Bolted MITC4 shell enclosure panels, `GANTRY_BRACKET_CATALOG` creep physics, `EXTRUSION_FILLINGS` mineral-cast/granular damping, transformed-section composite linear rails, 6-DOF nozzle-to-bed structural loop, Klipper-exact PSD input shaper calibration, and `pymoo` NSGA-II 3-objective Pareto optimization (`printsim pareto`). Every lumped catalog option (14 physical models across Frame, Gantry, Toolhead, Hotend, and Bed/Z) exposes its exact baseline & effective parameters (`effective_model_params`) and supports direct per-parameter overrides via `PrinterSpec.model_overrides` (compactly omitted from YAML when unused) and the Studio **Advanced Mode** toggle, paired with a **WIP · Model Trust?** modal (`model_confidence`) listing every active parameter and its 0–100% engineering confidence rating.

## Core Capabilities

- **Parametric Assembly DSL & OpenCASCADE STEP CAD (`printsim.schema`, `printsim.geometry.cad_b3d`)**: Declarative YAML/JSON/Python schema covering CoreXY, Cartesian Bedslinger, Delta, and arbitrary custom node-and-member trusses, backed by real datasheet materials and profiles — plus **`build123d` (OpenCASCADE B-Rep) STEP export** for full printer assemblies and parametric brackets with real fillets, M5/M3 bolt holes, and NEMA17 mounting patterns.
- **3D Frame FEA, Curved Tet10 Continuum FEA, QM6-3D Hex8 FEA & SIMP Topology Optimization (`printsim.fea`)**:
  - Deterministic 12-DOF 3D Euler-Bernoulli frame FEM with consistent self-weight fixed-end moments, per-node plate/disk bed rotary inertia, and QM6 MITC4 bolted shell panels (`_ALPHA_DRILL = 1e-3`).
  - **Conformal 3D Tetrahedral Continuum FEA (`tet_fem.py`)** powered by **Gmsh + scikit-fem** (`Tet10 P2` / `Tet4 P1`) with surface vertex stress recovery, isotropic metals & transversely isotropic FDM Z-layer delamination, smooth Three.js rendering, and ParaView `.vtu` export.
  - **Wilson–Taylor QM6-3D Solid Hex8 Continuum FEM (`solid_fem.py`)**, feature-aligned rectilinear bracket meshing (`bracket_designer.py` / `superelement.py`), and **3D SIMP Topology Optimization (`topology.py`)** with watertight printable STL export.
  - Coupled modal + drive receptance analysis (moving-mass-consistent series compliance + Rayleigh reflected stepper rotor inertia + mount-mode receptance coupling for compliant feet, frame ballast, and a suspended-from-lines mount; hot-chamber belt tension rise and CoreXY A/B tension-mismatch coupling) and upstream Klipper multi-damping input shaper calibration (`ZV`, `ZVD`, `MZV`, `EI`, `2HUMP_EI`, `3HUMP_EI`).
- **Manufacturable Cut-List, Stock Nesting & BOM (`printsim.geometry.bom`)**: Auto-merges split structural members into physical cut lengths (including net butt-joint lengths), runs 1D stock bar nesting (`1000mm`/`2000mm` bars + saw kerf & scrap `%`), calculates exact belt/rail/smooth-rod/motor/bracket/filling/panel/fastener counts and reconciled cost breakdowns, and exports to **JSON, CSV, or Markdown**.
- **Kinematics & Clearance (`printsim.kinematics`)**: CoreXY, Cartesian, and Delta exact FK/IK + full Delta Jacobian torque/RPM/workspace & tower-height checks, **19-motor datasheet-verified NEMA 17/23 catalog** (Excit3D Max, Isik's Tech Ouroboros 1000 PPR encoder, LDO 1.8° & 0.9° incl. `42STH48-2004MAH(VRN)`, Kraken V2, MOONS' 1.8° & 0.9° `MS17HA2P4200`, Leadshine, STEPPERONLINE) with peak/RMS-consistent stepper pull-out torque-speed curves, torque-balance max velocity, hot-copper TCR winding thermal equilibrium (Class B/F/H), decomposed static/dynamic belt `EA` with exact pulley-to-pulley free-span lengths, exact 2WD/AWD worst-direction accel limits, and dynamic tight-side overtension / slack-side slip checks.
- **Hotend FVM Thermal, Multi-Filament Rheology & Cooling (`printsim.thermal`)**: 40-node 1D Finite-Volume axial temperature solver $T(z)$ honoring enclosure `chamber_temp_c`, melt-averaged DSC multi-filament database (`PLA`, `PETG`, `ABS`, `ASA`, `PA12-CF`, `TPU 95A`) with Arrhenius $K \cdot a_T^n$ shear-thinning viscosity & $T_g$ heat-creep thresholds, transient heat-up curves, and overhang/bridging cooling predictions.
- **G-code Lookahead, Time-Domain Metrology & Virtual Print (`printsim.gcode`)**: Parses Marlin/Klipper/OrcaSlicer G-code including `G2`/`G3` arcs, `M204 T`, `SET_VELOCITY_LIMIT`, and E-only retracts, computes lookahead trapezoidal motion timings, tags every toolpath segment with physical defects (`ok`, `ringing` in µm, `smoothing`, `underextrusion`, `step_skip`, `belt_slip`), and runs coupled ZOH time-domain metrology (`ballbar.py`) covering **90° corners** (with separated geometric path error, shaper smoothing, cross-track wobble, along-track timing lag, and 25 mm wall-deviation profiles), **multi-segment Zig-Zag S-curves** with perpendicular-error heatmaps, and **virtual ISO 230-4 circular ballbars**.
- **Interactive 3D Web Studio (`web/`)**: React + Three.js studio surfacing inertial nozzle flex vs. gravity sag, plane-removed bed plate flatness + tilt/settlement, dynamic belt tension breakdowns, live 3D modes (Assembly, Frame FEA heatmap, Animated Modal Vibration, Virtual Print toolpath scrubber, **Curved Tet10 / QM6-3D Hex8 Solid Part FEA** with `.step`/`.stl` upload, and **3D SIMP Topology Optimization**), **Klipper ADXL345 CSV Overlay & Auto-Fit**, **Klipper PSD & Step-Response (with `±5 Hz` peak-width envelope, Ghost Snapshot, Speed VFA harmonic markers, and Carriage ADXL vs. Nozzle-Tip comparison)**, **Interactive Printed Wall Surface Texture Preview** (draggable Unshaped vs. Shaped curtain with grazing light & VFA ripples), **90° Corner & Zig-Zag S-Curve Metrology**, **Pareto Trade-Off Frontier (`Flex @ 10k`)** charts, **BOM & Cut-List inspector**, **one-click Klipper `printer.cfg` export**, **`#spec=` shareable build URLs**, and STEP/STL/DXF/CSV downloads.

## Quickstart

### 1. CLI (Headless & AI Agent Workflows)

```bash
# Run full multiphysics evaluation on a preset (optionally specifying filament)
uv run printsim evaluate examples/voron_trident_300.yaml --filament abs
uv run printsim evaluate examples/flsun_v400_delta.yaml

# Compare & auto-fit a printer spec against real Klipper ADXL345 CSV resonance measurements
uv run printsim calibrate-adxl examples/voron_trident_300.yaml --x-csv resonances_x.csv --y-csv resonances_y.csv --fit

# Separate in-place belt stiffness from the rest of the drive: print the measurement plan, then fit
# the resonance measured at 3+ toolhead positions (u, v = fraction of X / Y travel)
uv run printsim belt-identify examples/voron_trident_300.yaml --axis X
uv run printsim belt-identify examples/voron_trident_300.yaml --axis X -p 0.5,0.1,66.0 -p 0.5,0.5,68.5 -p 0.5,0.9,74.0

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

# PrintDB Integration (optional install for shared hardware datasheets & physics constants)
uv pip install "git+https://github.com/Jumpybeetroot/PrintDB.git"

# Run evolutionary AI design optimization
uv run printsim optimize examples/cartesian_bedslinger.yaml --iterations 30 --objective rigidity --out optimized.yaml

# Export a 3D STL mesh, OpenCASCADE STEP CAD assembly, or 2D CNC/laser DXF plate kit
uv run printsim export-stl examples/hyper_corexy_carbon.yaml --out hyper_corexy.stl
uv run printsim export-step examples/voron_monolith_awd.yaml --out monolith_awd.step
uv run printsim export-step --preset xy_motor_mount --out xy_motor_mount.step
uv run printsim export-dxf examples/voron_trident_300.yaml --part all_plates --out trident_plates.dxf

# Run conformal 3D Tet10 FEA (Gmsh + scikit-fem) on a B-Rep bracket or STEP file
uv run printsim tet-fea --preset xy_motor_mount --material pa12_cf_printed --vtu motor_mount.vtu

# List pre-baked CAD Craig-Bampton hardware superelements & run the zero-LLM local CAD pipeline
uv run printsim list-assets
uv run printsim cad inspect path/to/part.step
uv run printsim cad bake cad_recipes/my_part.yaml --fast   # cached Tet4 + 2x2 preview PNG + QA gate
uv run printsim cad verify                                 # verify all baked assets against QA invariants
uv run printsim cad fetch-onshape "https://cad.onshape.com/documents/<did>/w/<wid>/e/<eid>" --out cad_sources/part.step
uv run printsim cad fetch-onshape "https://cad.onshape.com/documents/<did>/w/<wid>/e/<eid>" --kind assemblies --out cad_sources/asm.step

# Zero-LLM diagnostics, A/B comparisons & parallel SQLite-cached sweeps
uv run printsim diagnose examples/corexy_voron_style.yaml
uv run printsim compare examples/corexy_voron_style.yaml -O frame.bracing=full_truss
uv run printsim sweep sweeps/example_parametric_sweep.yaml --workers 4
```

### 2. AI Agent & MCP Plug-and-Play Setup

Connect PrintSim directly to **Claude Desktop**, **Cursor**, **Antigravity / Windsurf**, **Gemini CLI**, or **LangGraph / OpenAI Agents** — see **[docs/AI_AGENT_INTEGRATION.md](docs/AI_AGENT_INTEGRATION.md)** for copy-paste config JSONs, all 21 MCP tool schemas, REST endpoints, zero-LLM local CLI workflows, and example engineering prompts.

### 3. Launch the Interactive 3D Web Studio

- **Zero-install Browser Studio (GitHub Pages)**: Open **[https://jumpybeetroot.github.io/PrintSim/](https://jumpybeetroot.github.io/PrintSim/)**. Built-in presets, 3D hardware GLB meshes, and BOMs load instantly from static pre-baked bundles while PrintSim's Python multiphysics engine (`numpy`, `scipy`, `pydantic`, `pyyaml`) boots in a background Pyodide WebAssembly worker so every slider edit, ADXL calibration, Solid Hex8 FEA, SIMP topology optimization, and STL/DXF/CSV export runs live in your browser. Rapid edits cancel superseded solves and returning to an earlier design is served instantly from an in-memory result cache.
- **Local Studio (full native Gmsh Tet10, OpenCASCADE STEP & MCP Live Bridge)**:

```bash
# Build frontend once (if not already built)
cd web && npm install && npm run build && cd ..

# Start FastAPI backend + Web 3D Studio on http://127.0.0.1:8000 (with Python hot-reload)
uv run printsim serve --port 8000 --reload
```

For frontend hot-reload development, run `uv run printsim serve --port 8000 --reload` in one terminal and `cd web && npm run dev` in another (proxies `/api` to port 8000).

Changing the studio UI? Follow [docs/STUDIO_STYLE_GUIDE.md](docs/STUDIO_STYLE_GUIDE.md). `npm run build` runs a style check (`npm run check:style`) before compiling and fails on raw colours, off-scale sizes and unlabelled controls.

### 4. Run the Physics Verification Suite

```bash
uv run pytest -n auto -q --tb=short
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for architecture details, [docs/CODEBASE_MAP.md](docs/CODEBASE_MAP.md) for the auto-generated symbol/line-range map, [docs/MODEL_CONFIDENCE_AND_REFERENCES.md](docs/MODEL_CONFIDENCE_AND_REFERENCES.md) for the full 0–100% physics confidence ledger & engineering references, and [docs/monolith_discord_benchmarks.md](docs/monolith_discord_benchmarks.md) for empirical CoreXY input-shaper calibration benchmarks.
