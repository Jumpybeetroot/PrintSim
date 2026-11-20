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
In a **Swarm**, explicitly pass the `model` ID below on `swarm(action="add")` (user-approved override of the default coordinator-inheritance behavior). In **`invoke_subagent`** or **`run_workflow`**, use the listed tier.

| Role / Agent | Swarm Exact Model ID | Separate-Pool Fallback | `invoke_subagent` Tier | Notes |
| :--- | :--- | :--- | :---: | :--- |
| `fea-engineer`, `thermal-extrusion-engineer` (routine solver/tests) | `gemini-4-neon-high` | `claude-opus-5-5-high` | `pro` | Best cost/intelligence Pareto knee (60 Intel, 65.7% TB-Science). |
| `fea-engineer`, `thermal-extrusion-engineer` (hard FEM derivations, eigensolver/nonlinear bugs) | `gemini-4-neon-xhigh` | `claude-opus-5-5-high` | `pro` | Escalate to `xhigh` **only** for hard physics/math (70.0% TB-Science, 71.3% HLE-Diamond). |
| `studio-frontend` (day-to-day React/TS/UI) | `gemini-4-neon-medium` | `claude-opus-5-5-high` | `inherit` | Beats Claude Opus/Sonnet High on Engineering (61) & FrontierCode (36.9%). |
| `studio-frontend` (complex Three.js shaders/state) | `gemini-4-neon-high` | `claude-opus-5-5-high` | `pro` | Use `high` for intricate 3D geometry/WebGL refactors. |
| `printer-designer` (YAML sweeps, `printsim evaluate`/`optimize`) | `gemini-4-hydrogen-high` or `gemini-4-neon-medium` | `claude-opus-5-5-medium` | `inherit` | Fast CLI loops (161 tok/s, 54.0% TB, 91.3% non-hallucination). |
| Verifier / Code Reviewer | `gemini-4-neon-high` or `claude-opus-5-5-high` | `claude-opus-5-5-high` | `pro` | `claude-opus-5-5-high` adds cross-family review & separate quota; require cited files & test output (lower non-hallucination). |
| `research` / codebase search | `gemini-4-hydrogen-high` (agentic) / `gemini-4-neon-low` (fast Q&A) | `claude-sonnet-5-5-high` | `flash` | Fast exploration without burning flagship quota. |
| `browser` | `gemini-4-hydrogen-high` | — | `flash` | Keep default `flash` tier per browser guidelines. |
| Docs / specs / architecture prose | `claude-opus-5-5-high` or `claude-sonnet-5-5-high` | `gemini-4-neon-medium` | `inherit` | Top AA-Briefcase Elo (1682 / 1634). |
| Trivial zero-tool mechanical edits / extraction | `gemini-4-hydrogen-low` | — | `flash_lite` | Only safe for non-agentic mechanical text tasks. |

### C. Explicit Do-Not-Use Models (Guardrails)
Never select or pin these models for PrintSim coding/orchestration roles:
- **`gemini-3.8-flash-*`**: Deprecated; low agentic accuracy (20% Terminal-Bench) and high cost-per-task thrashing ($7.77/task). Use `gemini-4-hydrogen-high` instead.
- **`gemini-4-hydrogen-xhigh`**: Dominated by `gemini-4-neon-high` (Neon High is faster and +6–16pp stronger for +18% cost).
- **`gemini-4-hydrogen-medium` / `gemini-4-hydrogen-low` for coding/terminal work**: Severe Terminal-Bench cliff (54.0% → 29.3% → 6.6%).
- **`claude-sonnet-5-5-*` for coding subagents** while capped at `High` in Antigravity: Dominated on coding by `gemini-4-hydrogen-high` and `gemini-4-neon-medium`.
- **`gemini-4-argon-*` / `Orchestrator (Argon)` as primary**: Slowest throughput (69 tok/s, 28m/task) and lower Coding Agent score (45 vs Neon's 59); reserve strictly as weekly-quota backup.
- **`gpt-oss-120b-medium`**: Unsuitable for software engineering or physics tasks (0% FrontierCode / Terminal-Bench).
