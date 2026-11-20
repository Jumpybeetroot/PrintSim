# PrintSim Project Rules (`AGENTS.md`)

You are working on **PrintSim**, a full-featured multiphysics 3D printer simulator for both human engineers and AI design agents.

## 1. Repository Layout

- `src/printsim/schema/` — Declarative parametric printer DSL (`PrinterSpec`), material catalog, and beam/rail profile catalog.
- `src/printsim/geometry/` — 3D structural graph builder (`AssemblyModel`) and STL/OBJ/STEP CAD I/O.
- `src/printsim/fea/` — Structural static & modal Finite Element Analysis (12-DOF 3D beam FEM + future solid tetrahedral FEM backends).
- `src/printsim/kinematics/` — CoreXY, Cartesian, and Delta FK/IK, stepper torque-speed curves, belt dynamics, and collision/clearance checks.
- `src/printsim/thermal/` — Hotend thermal networks, heat-creep analysis, melt-zone volumetric flow, and extrusion pressure drop.
- `src/printsim/gcode/` — G-code parser, lookahead motion planner, and virtual print defect simulator.
- `src/printsim/engine/` — Unified multiphysics evaluator (`evaluate_printer`) and headless AI design optimizer (`optimize_design`).
- `src/printsim/api/` — FastAPI server (`/api/simulate`, `/api/optimize`, `/api/export/stl`).
- `src/printsim/cli.py` — `printsim` Typer CLI.
- `web/` — Interactive React + TypeScript + Three.js 3D Studio.
- `examples/` — Reference printer specs (YAML) and benchmark fixtures.
- `tests/` — Pytest suite including analytical physics benchmarks.

## 2. Engineering & Units Standards

- **SI Inside, Friendly Units at the Boundary**:
  - All physics solvers (`fea`, `thermal`, `kinematics` force/torque math) MUST operate in strict **SI units** (`m`, `kg`, `s`, `N`, `Pa`, `W`, `K`, `rad/s`).
  - User-facing DSL fields, API responses, and UI displays use explicit suffixed units (`_mm`, `_g`, `_um`, `_mpa`, `_hz`, `_mm_s`, `_mm_s2`, `_mm3_s`, `_c`, `_w`, `_n`). Never create unit-ambiguous numeric fields.
- **Verify Physics Against Ground Truth**:
  - Any new structural, modal, kinematic, or thermal model MUST include a unit test in `tests/` verifying either a closed-form analytical solution (e.g., Euler-Bernoulli deflection/frequency, FK/IK round-trip, steady-state thermal resistance) or a monotonic physical sanity check.
- **Real Datasheet Parameters**:
  - New materials and profiles in `schema/materials.py` and `schema/profiles.py` must cite realistic engineering values (e.g., Misumi extrusion datasheets, Gates PowerGrip belt stiffness, MatWeb material constants).

## 3. Verification Commands

Always run verification before committing or declaring a task complete:
- Python physics & API tests: `uv run pytest -v`
- Frontend TypeScript & bundle build (when `web/` changes): `npm run build --prefix web`
- CLI smoke check: `uv run printsim evaluate examples/corexy_voron_style.yaml`

## 4. Version Control Rules

- Make small, focused commits with clear prefixes (`feat:`, `fix:`, `phys:`, `ui:`, `test:`, `docs:`, `chore:`).
- Never commit `.venv/`, `web/node_modules/`, `web/dist/`, or generated `.stl`/`.msh` outputs.
- Keep example YAML presets valid and passing `tests/test_engine_api.py`.

## 5. Specialized Subagents

Delegate domain-heavy tasks to the workspace subagents in `.agents/agents/`:
- `fea-engineer`: 3D beam/solid FEA, modal analysis, topology optimization, input shaper modeling.
- `thermal-extrusion-engineer`: Hotend thermal networks, melt flow, heat creep, part-cooling CFD, and rheology.
- `printer-designer`: Creating, mutating, benchmarking, and optimizing printer YAML designs via the CLI/SDK.
- `studio-frontend`: Three.js viewport, FEA/modal/toolpath shaders and visualizations, React UI panels.
