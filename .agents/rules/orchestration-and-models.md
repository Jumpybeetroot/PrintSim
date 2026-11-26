---
trigger: always_on
---

# Orchestration, Parallel Subagents & Model Routing Rules

These are **explicit user-approved project defaults** for PrintSim. Apply them automatically on every session (including `invoke_subagent`, `run_workflow`, and `swarm` / `/teamwork` sessions) unless the user overrides them in-chat.

## 1. Parallel-First Job Splitting (Speed Optimization)

Whenever a user request can be decomposed into **2 or more independent workstreams**, split and run them **concurrently** across multiple subagents instead of doing them sequentially in one thread:

1. **Dispatch in the same turn**: Launch independent `invoke_subagent` calls together in a single turn (or use `run_workflow` `parallel(...)` / `swarm` for 3+ parallel tracks, batch sweeps, or author-verifier pipelines).
2. **Natural PrintSim split boundaries**:
   - Structural / FEM solver work (`src/printsim/fea/`, `tests/test_fea.py`) → `fea-engineer`
   - Thermal / extrusion / rheology work (`src/printsim/thermal/`, `tests/test_kinematics_thermal.py`) → `thermal-extrusion-engineer`
   - 3D Web Studio UI / Three.js (`web/`) → `studio-frontend`
   - Printer YAML specs, trade studies & CLI benchmarks (`examples/`, `printsim evaluate` / `optimize`) → `printer-designer`
   - Independent codebase investigation / docs lookup → `research` (multiple parallel `research` calls for unrelated modules)
3. **Strict file-ownership partitioning**:
   - Assign explicit, non-overlapping file/directory ownership in every subagent brief so two writers never edit the same file simultaneously.
   - If shared interfaces (e.g., `src/printsim/api/server.py` ↔ `web/src/types.ts` or `schema/`) must change, define the contract upfront in the orchestrator (or sequentially first), then fan out backend, frontend, and test subagents in parallel.
   - Use `workspace: "share"` (or `"branch"`) when parallel writers run concurrent builds/tests that could contend on git or build artifacts; keep readers, researchers, and verifiers on `workspace: "inherit"`.
4. **When NOT to split**:
   - Do not spawn subagents for tiny single-file tweaks, quick one-command checks, or tightly coupled single-function refactors where coordination overhead exceeds execution time.
5. **Integrate & verify**:
   - After parallel workers finish, verify integration (`uv run pytest -v`, `npm run build --prefix web` if `web/` changed, and CLI smoke test) — optionally via a dedicated verifier subagent.

## 2. Automatic Model & Reasoning-Level Routing

Based on Artificial Analysis Intelligence Index v5.0.1 and Coding Agent Index v2.1 benchmarks for models exposed in Antigravity:

### A. Orchestrator / Main Agent / Swarm Coordinator
- **Solo / standard orchestration**: **`Orchestrator (Neon)`** (`gemini-pro-orchestrator-neon`) or **`Gemini 4 Neon (High)`** (`gemini-4-neon-high`).
- **Pure Swarm coordinator** (delegating all edits to workers): **`Gemini 4 Neon (Medium)`** (`gemini-4-neon-medium`) for fast routing turns, or **`Gemini 4 Neon (High)`** (`gemini-4-neon-high`) for complex multi-physics architecture.
- **Quota fallback order** (if daily Neon/Hydrogen pool runs out): **`Claude Opus 5.5 (High)`** (`claude-opus-5-5-high`) first → **`Orchestrator (Argon)`** (`gemini-pro-orchestrator-g1-pro`) last.
- Do **not** keep an orchestrator on `gemini-4-neon-xhigh` for routine routing turns (+72% cost & 56s TTFT for +1 Intelligence point).

### B. Per-Role Subagent & Swarm Worker Mapping
In a **Swarm**, explicitly pass the `model` ID below on `swarm(action="add")` (user-approved override of the default coordinator-inheritance behavior).

> **CRITICAL `invoke_subagent` ROUTING RULE (Verified from Transcript Telemetry):**
> In `invoke_subagent`, **`Model="pro"` currently resolves to `gemini-4-hydrogen-tiered-xhigh`** (which is on the Do-Not-Use list and historically caused 52% of all PrintSim token burn via 80–293 turn thrashing!). Therefore:
> - **NEVER pass `Model="pro"` to `invoke_subagent`** and **NEVER set `model: pro` in `.agents/agents/*.md`**.
> - Use **`Model="inherit"`** for `fea-engineer`, `thermal-extrusion-engineer`, `studio-frontend`, and `printer-designer` so they run on the parent's `Gemini 4 Neon (High)` or `Gemini 4 Neon (Medium)`!
> - Use **`Model="flash"`** only for `research` and `browser` subagents (maps to `gemini-4-hydrogen-tiered-high`).

| Role / Agent | Swarm Exact Model ID | Separate-Pool Fallback | `invoke_subagent` Tier | Notes |
| :--- | :--- | :--- | :---: | :--- |
| `fea-engineer`, `thermal-extrusion-engineer` (routine solver/tests) | `gemini-4-neon-high` | `claude-opus-5-5-high` | `inherit` | Inherits parent Neon High/Medium. Never pass `pro` (routes to banned `hydrogen-xhigh`). |
| `fea-engineer`, `thermal-extrusion-engineer` (hard FEM derivations, eigensolver/nonlinear bugs) | `gemini-4-neon-xhigh` | `claude-opus-5-5-high` | `inherit` | Escalate via Swarm `model` or parent model switch **only** for hard physics/math. |
| `studio-frontend` (day-to-day React/TS/UI) | `gemini-4-neon-medium` | `claude-opus-5-5-high` | `inherit` | Beats Claude Opus/Sonnet High on Engineering (61) & FrontierCode (36.9%). |
| `studio-frontend` (complex Three.js shaders/state) | `gemini-4-neon-high` | `claude-opus-5-5-high` | `inherit` | Use Neon High parent or Swarm for intricate 3D geometry/WebGL refactors. |
| `printer-designer` (YAML sweeps, `printsim evaluate`/`optimize`) | `gemini-4-hydrogen-high` or `gemini-4-neon-medium` | `claude-opus-5-5-medium` | `inherit` | Prefer running `printsim sweep`/`diagnose` locally at 0 tokens first. |
| Verifier / Code Reviewer | `gemini-4-neon-high` or `claude-opus-5-5-high` | `claude-opus-5-5-high` | `inherit` | `claude-opus-5-5-high` adds cross-family review & separate quota; require cited files & test output. |
| `research` / codebase search | `gemini-4-hydrogen-high` (agentic) / `gemini-4-neon-low` (fast Q&A) | `claude-sonnet-5-5-high` | `flash` | Fast exploration without burning flagship quota. |
| `browser` | `gemini-4-hydrogen-high` | — | `flash` | Keep default `flash` tier per browser guidelines. |
| Docs / specs / architecture prose | `claude-opus-5-5-high` or `claude-sonnet-5-5-high` | `gemini-4-neon-medium` | `inherit` | Top AA-Briefcase Elo (1682 / 1634). |
| Trivial zero-tool mechanical edits / extraction | `gemini-4-hydrogen-low` | — | `flash_lite` | Only safe for non-agentic mechanical text tasks. |

## 3. Token & Quota Hygiene Rules (Mandatory)
1. **Zero Polling Loops:** NEVER call `manage_task(action="status")` repeatedly while waiting on Gmsh, `scikit-fem`, `pytest`, or builds. Set `NotificationTimeoutSeconds` and end the turn to wait reactively — or for sweeps taking >60s, hand the user the exact CLI command (`uv run printsim sweep ...`) to run locally at 0 LLM tokens.
2. **Keep Context Lean (< 150k):** Aim for **one milestone per thread**; recommend `/compact` or a fresh thread after completing a feature. Scope every subagent to a crisp deliverable that completes in **<= 35 turns**.
3. **Zero-LLM CAD & FEA First:**
   - Use `uv run printsim cad inspect <part.step>` instead of ad-hoc `python -c` STEP probing.
   - Use declarative `cad_recipes/*.yaml` + `uv run printsim cad bake` (with `--fast` preview contact-sheet PNG and `.cache/cad_fea/` mesh caching) instead of writing one-off baker scripts or taking repeated browser screenshots.
   - Use `build123d` scripts locally for AI-authored CAD; use Onshape MCP only for exporting existing user assemblies (`printsim cad fetch-onshape`) or flipping Onshape Variables — never build complex parts via dozens of turn-by-turn Onshape MCP sketch/extrude calls.
   - Use `uv run printsim sweep`, `printsim diagnose`, and `printsim compare` instead of ad-hoc `python -c` diagnostic scripts.
4. **Fast Dev-Loop Defaults:**
   - Read `docs/CODEBASE_MAP.md` first to jump to exact line ranges instead of reading entire large files.
   - Use `grep -rn` (not `rg`, unless `rg` is verified installed).
   - Run `uv run printsim serve --reload` once per session instead of repeatedly killing/restarting the server.
   - Run targeted `uv run pytest tests/test_<area>.py -q --tb=short` during iteration and `uv run pytest -n auto -q --tb=short` (or `-q --tb=short`) once at final verification.

### C. Explicit Do-Not-Use Models (Guardrails)
Never select or pin these models for PrintSim coding/orchestration roles:
- **`gemini-3.8-flash-*`**: Deprecated; low agentic accuracy (20% Terminal-Bench) and high cost-per-task thrashing ($7.77/task). Use `gemini-4-hydrogen-high` instead.
- **`gemini-4-hydrogen-xhigh`**: Dominated by `gemini-4-neon-high` (Neon High is faster and +6–16pp stronger for +18% cost).
- **`gemini-4-hydrogen-medium` / `gemini-4-hydrogen-low` for coding/terminal work**: Severe Terminal-Bench cliff (54.0% → 29.3% → 6.6%).
- **`claude-sonnet-5-5-*` for coding subagents** while capped at `High` in Antigravity: Dominated on coding by `gemini-4-hydrogen-high` and `gemini-4-neon-medium`.
- **`gemini-4-argon-*` / `Orchestrator (Argon)` as primary**: Slowest throughput (69 tok/s, 28m/task) and lower Coding Agent score (45 vs Neon's 59); reserve strictly as weekly-quota backup.
- **`gpt-oss-120b-medium`**: Unsuitable for software engineering or physics tasks (0% FrontierCode / Terminal-Bench).
