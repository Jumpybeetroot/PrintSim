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
- **CAD Superelement, Rail & Viewport Invariants** (`docs/CAD_AND_FEA_PITFALLS.md`):
  - Before adding new CAD hardware assets, modifying structural/modal excitation or Hertzian rail stiffness, or editing 3D toolhead modal/FEA deformation and sidebar flex CSS, consult `docs/CAD_AND_FEA_PITFALLS.md` and run its pre-finish verification checklist (rigid-arm port shifts, `cog`+`nozzle_tip` inertial splits, rail roll pivot & 4-bolt pad clamping, no `flex: 1 0 100%` in column containers).
- **Explore Existing Work & Research First** (`.agents/rules/research-and-reuse.md`):
  - Before building custom solvers, CAD tools, or optimizers from scratch, research proven open-source libraries (`gmsh`, `scikit-fem`, `build123d`, `sectionproperties`, `pymoo`, `optuna`, Klipper `shaper_defs`/`shaper_calibrate`, `fullcontrol`, `ht`) and published engineering models; integrate existing work via clean adapters/extras whenever it improves accuracy or saves reinventing the wheel.
- **Keep Project Files & Docs Synchronized**:
  - Whenever you change physics solvers, catalog constants, schemas, API/MCP payloads, presets, CLI commands, or Studio UI behavior, ALWAYS update the corresponding project docs and reference files in the same task (`README.md`, `ARCHITECTURE.md`, `docs/PHYSICS_AND_MATH.md`, `docs/MODEL_CONFIDENCE_AND_REFERENCES.md`, `docs/PHYSICS_AUDIT_2026-11.md`, `docs/physics/*.md`, `docs/STUDIO_STYLE_GUIDE.md`, and `docs/AI_AGENT_INTEGRATION.md` as applicable). Never leave docs out of sync with the implementation.
  - Whenever you calibrate a lumped model, add a datasheet/bench measurement, or change confidence scores, update **both** `docs/MODEL_CONFIDENCE_AND_REFERENCES.md` (including its Calibration History table) and `build_model_confidence_audit()` in `src/printsim/schema/overrides.py` together.

## 3. Verification Commands & Fast Dev-Loop

Always run verification before committing or declaring a task complete:
- Targeted iteration tests: `uv run pytest tests/test_<area>.py -q --tb=short`
- Full Python physics & API tests (final verification): `uv run pytest -q --tb=short` (or `-n auto -q --tb=short` when `pytest-xdist` is installed)
- Frontend style check, TypeScript & bundle build (when `web/` changes): `npm run build --prefix web` (runs `web/scripts/check-style.mjs` first; fix violations per `docs/STUDIO_STYLE_GUIDE.md`, do not bypass the check)
- CLI smoke check: `uv run printsim evaluate examples/corexy_voron_style.yaml` (or `uv run printsim diagnose examples/corexy_voron_style.yaml`)
- Live Studio handoff: Run `uv run printsim serve --reload` once per session so backend Python edits hot-reload automatically without manual kills/restarts. Rebuild `web/dist` once at handoff (only restart `printsim serve` if not already running with `--reload`), using `BypassSandbox: true` if needed, and remind the user to hard-refresh after frontend rebuilds.
- Codebase navigation: Consult `docs/CODEBASE_MAP.md` first (regenerate via `uv run python scripts/gen_codebase_map.py` after structural refactors) and use `grep -rn` (not `rg`) so searches succeed on the first try.
- Zero-LLM CAD & FEA CLIs: Use `uv run printsim cad inspect|bake|verify|preview|fetch-onshape`, `uv run printsim sweep`, and `uv run printsim diagnose|compare` instead of ad-hoc `python -c` scripts or polling loops.
- Milestone doc syncs: Batch updates to `README.md`, `ARCHITECTURE.md`, and `docs/*.md` once per completed milestone/commit rather than after every intermediate tweak.

## 4. Version Control Rules

- Make small, focused commits with clear prefixes (`feat:`, `fix:`, `phys:`, `ui:`, `test:`, `docs:`, `chore:`).
- Never commit `.venv/`, `web/node_modules/`, `web/dist/`, `.cache/`, `cad_sources/`, or generated `.stl`/`.msh` outputs (baked hardware `src/printsim/assets/hardware/**/*.glb` ARE tracked).
- Keep example YAML presets valid and passing `tests/test_engine_api.py`.

## 5. Specialized Subagents, Parallel Splitting & Model Routing

Follow `.agents/rules/orchestration-and-models.md` automatically on every task:

- **Split jobs across parallel subagents whenever possible**: If a task breaks into 2+ independent workstreams (e.g., FEA/thermal backend + `web/` frontend + YAML benchmarks + tests/research), assign non-overlapping file ownership and launch subagents concurrently in the same turn (or via `run_workflow` / `swarm`) to maximize speed.
- **Specialized workspace subagents** (`.agents/agents/`):
  - `fea-engineer` (`Model="inherit"` for `invoke_subagent`; `gemini-4-neon-high` in Swarm — **never pass `Model="pro"`**, which resolves to banned `hydrogen-xhigh`): 3D beam/solid FEA, modal analysis, topology optimization, input shaper modeling.
  - `thermal-extrusion-engineer` (`Model="inherit"` for `invoke_subagent`; `gemini-4-neon-high` in Swarm): Hotend thermal networks, melt flow, heat creep, part-cooling CFD, and rheology.
  - `studio-frontend` (`Model="inherit"` for `invoke_subagent`; `gemini-4-neon-medium` / `gemini-4-neon-high` in Swarm): Three.js viewport, FEA/modal/toolpath shaders and visualizations, React UI panels.
  - `printer-designer` (`Model="inherit"` for `invoke_subagent`; `gemini-4-hydrogen-high` or `gemini-4-neon-medium` in Swarm): Creating, mutating, benchmarking, and optimizing printer YAML designs via the CLI/SDK.
- **Orchestrator & fallback routing**: Run the orchestrator on `Gemini 4 Neon (Medium)` for routine UI/wiring/asset tasks and `Gemini 4 Neon (High)` for complex physics/architecture (`Orchestrator (Neon)` also supported). Use `claude-opus-5-5-high` for cross-family verification or separate-pool fallback. Never assign `gemini-3.8-flash-*`, `gemini-4-hydrogen-xhigh`, `gemini-4-hydrogen-medium`/`low` (for coding), `claude-sonnet-5-5-*` (for coding), `gemini-4-argon-*` (unless Neon quota is out), or `gpt-oss-120b-medium`.
