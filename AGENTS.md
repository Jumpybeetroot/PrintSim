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
- **Real Datasheet Parameters & PrintDB Single Source of Truth**:
  - New materials, hardware specs, and physics constants should use **PrintDB** (`/Users/jumpybeetroot/Projects/PrintDB`, bridged via `src/printsim/schema/printdb_bridge.py` and the `printdb` MCP server) and cite realistic engineering values (e.g., Misumi extrusion datasheets, Gates PowerGrip belt stiffness, MatWeb material constants).
  - **Always persist hard references & verified data directly to tracked `PrintDB/data/`**: Whenever you discover or verify a new hardware datasheet, scale-weighed component/toolhead mass, performance curve, material spec, or physics/math reference, add it **directly to tracked `PrintDB/data/`** using the `printdb_add_record(..., target="data")` MCP tool (avoid leaving verified data in gitignored `local/`; if any record was added to `local/`, run `printdb_promote(all_local=True)` before finishing). Always include explicit unit suffixes, `verification` tier, cited `sources`, and `related_ids` (on references, so `printdb lint` stays clean), then verify `printdb validate --community-only`, `printdb build`, and `tests/test_printdb_integration.py` (which enforces 100% catalog parity and zero stranded `local/` records).
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
- Real-machine shaper regression check (after physics/FEA/catalog edits): `uv run printsim shaper-benchmark --arch corexy --compare data/touchstones/shaper_baseline_corexy.json` (or omit `--arch corexy` with `--compare data/touchstones/shaper_baseline.json`). The canonical agent-editable list of real measured machines (currently 33 CoreXY + 3 bedslinger/delta + 16 controlled same-machine A/B sweeps) lives in `data/touchstones/shaper_benchmarks.yaml`; any agent may append new cited machines/sweeps there, verify with `uv run pytest tests/test_shaper_benchmarks.py -q`, and refresh baselines with `--save`.
- Live Studio handoff: Run `uv run printsim serve --reload` once per session so backend Python edits hot-reload automatically without manual kills/restarts. Rebuild `web/dist` once at handoff (only restart `printsim serve` if not already running with `--reload`), using `BypassSandbox: true` if needed, and remind the user to hard-refresh after frontend rebuilds.
- Codebase navigation: Consult `docs/CODEBASE_MAP.md` first (regenerate via `uv run python scripts/gen_codebase_map.py` after structural refactors), use `grep -rn` (not `rg`) + tight line ranges instead of re-reading huge files (`printer.py`, `PrinterViewport.tsx`, `builder.py`), and `tail -n 40` `.log` files instead of calling `read_file` on them.
- Zero-LLM CAD & FEA CLIs: Use `uv run printsim cad inspect|bake|verify|preview|fetch-onshape`, `uv run printsim sweep`, and `uv run printsim diagnose|compare` instead of ad-hoc `python -c` scripts or polling loops.
- Hybrid Onshape CAD + Local `build123d` Workflow (`docs/AI_AGENT_INTEGRATION.md` §6): The `onshape` MCP server is available for discovering/inspecting user & public Onshape documents (`onshape_list_documents` with `limit<=20`, `onshape_search_documents`, `onshape_get_document_summary`, `onshape_shaded_view` with writable `out="/tmp/..."`, `onshape_get_bom`, `onshape_clearance`), flipping Variables/Configurations, authoring human-editable CAD (FeatureScript/Part Studios/Assemblies), and exporting Part Studio or Assembly STEPs (`onshape_export` with `kind: "partstudios"|"assemblies"`, or host CLI `uv run printsim cad fetch-onshape --kind assemblies`). Use local `build123d` for fast FEA prep (thickening surface shells, hollowing OEM envelopes, Boolean cleanup) and multi-variant parametric sweeps.
- Milestone doc syncs: Batch updates to `README.md`, `ARCHITECTURE.md`, and `docs/*.md` once per completed milestone/commit rather than after every intermediate tweak.
- Pyodide WASM & GitHub Pages Safety (`src/printsim/wasm_dispatch.py`, `web/src/workers/pyodideWorker.ts`): Modules reachable from `wasm_dispatch.py` (`schema`, `geometry`, `fea`, `kinematics`, `thermal`, `gcode`, `engine`, `cad_assets.registry`) MUST NOT eagerly import optional native/web deps (`trimesh`, `ezdxf`, `httpx`, `fastapi`, `build123d`, `gmsh`, `pymoo`) or Pyodide-stripped stdlib modules (`_multiprocessing` / `ProcessPoolExecutor`, `sqlite3`, `fcntl`, `termios`). Use PEP 562 lazy `__getattr__` or in-function imports, keep Pyodide worker requests serialized via direct Python callables (never shared `py.globals`), and verify with `uv run pytest tests/test_wasm_dispatch.py -q`.
- Evaluation Hot-Path Performance (`ARCHITECTURE.md` § Evaluation Pipeline Performance): New solvers that need global `K`/`M` or element matrices MUST accept the shared `system=` (`assemble_global_system`, `fea/beam3d.py`) from `evaluate_printer` instead of re-assembling; treat everything on a `GlobalSystem` as read-only. Keep `scipy.signal` / `scipy.optimize` / `scipy.spatial` imports in-function for modules reachable from `wasm_dispatch.py` (browser boot cost), and keep hot-loop rewrites bit-identical (`uv run pytest tests/test_shared_fe_system.py tests/test_wasm_dispatch.py -q`).
- CI Portability (`pyproject.toml`, `.github/workflows/ci.yml`): Keep every unconditional runtime import declared in `pyproject.toml` + `uv.lock`; guard tests requiring optional extras (`--extra cad|tetfea|opt`) with `is_cad_available()` / `is_tet_fea_available()` / `is_pymoo_available()`, and scale wall-clock timing budgets when `os.environ.get("CI")` is set.

## 4. Version Control Rules

- Make small, focused commits with clear prefixes (`feat:`, `fix:`, `phys:`, `ui:`, `test:`, `docs:`, `chore:`).
- Never commit `.venv/`, `web/node_modules/`, `web/dist/`, `.cache/`, `cad_sources/`, or generated `.stl`/`.msh` outputs (baked hardware `src/printsim/assets/hardware/**/*.glb` ARE tracked).
- In `data/touchstones/`, only the canonical inputs are tracked: `shaper_benchmarks.yaml`, `shaper_baseline.json`, `shaper_baseline_corexy.json`, and `belt_pitch_cell_fea.json` (tests and the regression check read them). The KB database, raw captures, and downloaded accelerometer CSVs stay local; to track another file there, add a `!` line for it in `.gitignore`.
- Keep example YAML presets valid and passing `tests/test_engine_api.py`.

## 5. Specialized Subagents, Parallel Splitting & Model Routing

Follow `.agents/rules/orchestration-and-models.md` automatically on every task:

- **Auto-split big requests & topic pivots into fresh cost-routed threads** (global `split-work` skill at `~/.gemini/config/skills/split-work/SKILL.md`): When a prompt spans 2+ independent tracks, pivots topics mid-thread, or hits the global cost/context guardrail (`150k ctx / $3.50`, enforced by `~/.gemini/config/hooks.json`), auto-spawn fresh sidebar threads via `python3 /Users/jumpybeetroot/.gemini/config/skills/split-work/scripts/spawn_thread.py --role physics|ui|research|quick --title "..." --prompt-file /tmp/antigravity_handoffs/<slug>.md` with non-overlapping file ownership (use `--staged` for dependent phases, or `/swarm` with `workspace: "share"` when tight coordination is needed).
- **Split in-thread helper jobs across parallel subagents**: For tightly scoped parallel helpers inside one milestone, assign non-overlapping file ownership and launch subagents concurrently in the same turn.
- **Specialized workspace subagents** (`.agents/agents/`):
  - `fea-engineer` (`Model="inherit"` for `invoke_subagent`; `gemini-4-neon-high` in Swarm — **never pass `Model="pro"`**, auto-rewritten by global `cost_guard.py`): 3D beam/solid FEA, modal analysis, topology optimization, input shaper modeling.
  - `thermal-extrusion-engineer` (`Model="inherit"` for `invoke_subagent`; `gemini-4-neon-high` in Swarm): Hotend thermal networks, melt flow, heat creep, part-cooling CFD, and rheology.
  - `studio-frontend` (`Model="inherit"` for `invoke_subagent`; `gemini-4-neon-medium` / `gemini-4-neon-high` in Swarm): Three.js viewport, FEA/modal/toolpath shaders and visualizations, React UI panels.
  - `printer-designer` (`Model="inherit"` for `invoke_subagent`; `gemini-4-hydrogen-high` or `gemini-4-neon-medium` in Swarm): Creating, mutating, benchmarking, and optimizing printer YAML designs via the CLI/SDK.
- **Orchestrator & cost-first routing**: Run routine UI/wiring/GitHub/PrintDB threads on `Gemini 4 Neon (Medium)` (\$1.90/task, Intel 55) and complex physics/FEA on `Gemini 4 Neon (High)` (\$3.74/task, Intel 60). Use `claude-opus-5-5-high` for cross-family verification or separate-pool fallback. Never assign Pareto-dominated models (`gemini-4-carbon-*`, `gemini-4-hydrogen-xhigh`, `gemini-4-hydrogen-medium`/`low` for coding, `gemini-3.8-flash-*`, `claude-sonnet-5-5-*` for coding, `gemini-4-argon-*` as primary, or `gpt-oss-120b-medium`). Check daily dollar spend via `python3 /Users/jumpybeetroot/.gemini/config/skills/split-work/scripts/cost_report.py`.
