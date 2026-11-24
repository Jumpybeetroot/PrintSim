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
- `web/` — Interactive React + TypeScript + Three.js 3D Studio. Visual system and UI rules: `docs/STUDIO_STYLE_GUIDE.md`.
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
- **Studio UI Follows the Style Guide** (`docs/STUDIO_STYLE_GUIDE.md`, `.agents/rules/web-studio.md`):
  - Any change under `web/src/` must use the design tokens and documented patterns (cards, pills, fields, labels with `Name (unit)`), keep every control accessibly named, and pass the guide's pre-finish checklist. Background on the system: `docs/STUDIO_UI_OVERHAUL.md`.
- **Explore Existing Work & Research First** (`.agents/rules/research-and-reuse.md`):
  - Before building custom solvers, CAD tools, or optimizers from scratch, research proven open-source libraries (`gmsh`, `scikit-fem`, `build123d`, `sectionproperties`, `pymoo`, `optuna`, Klipper `shaper_defs`/`shaper_calibrate`, `fullcontrol`, `ht`) and published engineering models; integrate existing work via clean adapters/extras whenever it improves accuracy or saves reinventing the wheel.

## 3. Verification Commands

Always run verification before committing or declaring a task complete:
- Python physics & API tests: `uv run pytest -v`
- Frontend style check, TypeScript & bundle build (when `web/` changes): `npm run build --prefix web` (runs `web/scripts/check-style.mjs` first; fix violations per `docs/STUDIO_STYLE_GUIDE.md`, do not bypass the check)
- CLI smoke check: `uv run printsim evaluate examples/corexy_voron_style.yaml`

## 4. Version Control Rules

- Make small, focused commits with clear prefixes (`feat:`, `fix:`, `phys:`, `ui:`, `test:`, `docs:`, `chore:`).
- Never commit `.venv/`, `web/node_modules/`, `web/dist/`, or generated `.stl`/`.msh` outputs.
- Keep example YAML presets valid and passing `tests/test_engine_api.py`.

## 5. Specialized Subagents, Parallel Splitting & Model Routing

Follow `.agents/rules/orchestration-and-models.md` automatically on every task:

- **Split jobs across parallel subagents whenever possible**: If a task breaks into 2+ independent workstreams (e.g., FEA/thermal backend + `web/` frontend + YAML benchmarks + tests/research), assign non-overlapping file ownership and launch subagents concurrently in the same turn (or via `run_workflow` / `swarm`) to maximize speed.
- **Specialized workspace subagents** (`.agents/agents/`):
  - `fea-engineer` (`pro` / `gemini-4-neon-high`, escalate to `gemini-4-neon-xhigh` only for hard FEM/eigen math): 3D beam/solid FEA, modal analysis, topology optimization, input shaper modeling.
  - `thermal-extrusion-engineer` (`pro` / `gemini-4-neon-high`, escalate to `gemini-4-neon-xhigh` only for tough rheology/PDE math): Hotend thermal networks, melt flow, heat creep, part-cooling CFD, and rheology.
  - `studio-frontend` (`inherit` / `gemini-4-neon-medium`, or `gemini-4-neon-high` for complex Three.js shaders): Three.js viewport, FEA/modal/toolpath shaders and visualizations, React UI panels.
  - `printer-designer` (`inherit` / `gemini-4-hydrogen-high` or `gemini-4-neon-medium`): Creating, mutating, benchmarking, and optimizing printer YAML designs via the CLI/SDK.
- **Orchestrator & fallback routing**: Run the orchestrator on `Orchestrator (Neon)` or `gemini-4-neon-high` (`gemini-4-neon-medium` for pure Swarm routing). Use `claude-opus-5-5-high` for cross-family verification or separate-pool fallback. Never assign `gemini-3.8-flash-*`, `gemini-4-hydrogen-xhigh`, `gemini-4-hydrogen-medium`/`low` (for coding), `claude-sonnet-5-5-*` (for coding), `gemini-4-argon-*` (unless Neon quota is out), or `gpt-oss-120b-medium`.
