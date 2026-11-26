---
trigger: always_on
---

# Orchestration, Parallel Subagents & Model Routing Rules

These are **explicit user-approved project defaults** for PrintSim. Apply them automatically on every session (including `invoke_subagent`, `run_workflow`, and `swarm` / `/teamwork` sessions) unless the user overrides them in-chat.

## 1. Auto-Splitting & Parallel Job Routing (Cost + Speed Optimization)

Whenever a user request spans **2+ independent subsystems/milestones**, pivots to a new feature mid-thread, or approaches **150k context / 40 turns / \$3.50**:

1. **Prefer Auto-Splitting Into Fresh Threads (global `split-work` skill at `~/.gemini/config/skills/split-work/SKILL.md`)**:
   - The user has pre-approved **auto-spawning** separate top-level threads via `python3 /Users/jumpybeetroot/.gemini/config/skills/split-work/scripts/spawn_thread.py --role <role> --title "<title>" --prompt-file /tmp/antigravity_handoffs/<slug>.md` so each track starts at fresh `~25k` context on its exact Pareto-optimal thinking level (`physics` → `neon-high`, `ui`/`wiring`/`cad`/`github`/`printdb` → `neon-medium`, `research`/`browser` → `hydrogen-high`, `quick`/`docs` → `neon-low`).
   - Independent non-overlapping file scopes launch **active in parallel**; dependent/overlapping phases launch Phase 1 active + Phase 2 `--staged` (or use `/swarm` with `workspace: "share"` if tight real-time coordination is needed).
2. **In-Thread Parallel Subagents (for tightly scoped helper tasks within one milestone)**:
   - Dispatch independent `invoke_subagent` calls together in the same turn (or `run_workflow` / `swarm` for 3+ tracks) with strict non-overlapping file ownership:
     - Structural / FEM solver work (`src/printsim/fea/`, `tests/test_fea.py`) → `fea-engineer`
     - Thermal / extrusion / rheology work (`src/printsim/thermal/`, `tests/test_kinematics_thermal.py`) → `thermal-extrusion-engineer`
     - 3D Web Studio UI / Three.js (`web/`) → `studio-frontend`
     - Printer YAML specs, trade studies & CLI benchmarks (`examples/`, `printsim evaluate` / `optimize`) → `printer-designer`
     - Independent codebase investigation / docs lookup → `research` (`Model="flash"`)
   - Use `workspace: "share"` (or `"branch"`) when parallel writers run concurrent builds/tests; keep readers/verifiers on `workspace: "inherit"`.
3. **When NOT to split**:
   - Do not split tiny 1–2 file tweaks, quick one-command checks, or single-function fixes that finish in <10 turns.
4. **Integrate & verify**:
   - Verify targeted tests (`uv run pytest tests/test_<area>.py -q --tb=short`, `npm run build --prefix web` if `web/` changed, and `uv run printsim diagnose` / `evaluate`).

## 2. Automatic Model & Thinking-Level Routing (Cost-First)

Based on live **Artificial Analysis v5.0.2** Intelligence Index, Coding Agent Index v2.1, and Cost per Task benchmarks:

| Pareto Tier | Exact Model ID | Intel | Cost / Task | Input / Cache / Output (per 1M) | Role |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Default Coding** | `gemini-4-neon-medium` | 55 | **\$1.90** | \$3.00 / \$0.15 / \$15.00 | Everyday UI, CAD wiring, GitHub/CI, PrintDB, presets (49% cheaper than High; beats Hydrogen xHigh on both score & cost) |
| **Complex Physics** | `gemini-4-neon-high` | 60 | **\$3.74** | \$3.00 / \$0.15 / \$15.00 | FEA/modal/solid FEM, belt/thermal physics, deep architecture (matches Carbon High at half the cost) |
| **Research / Browser** | `gemini-4-hydrogen-high` | 51 | **\$1.87** | \$1.50 / \$0.075 / \$7.50 | Input-heavy Discord/web/codebase research (`Model="flash"` — 2× cheaper input/cache, 177 tok/s) |
| **Quick / Docs** | `gemini-4-neon-low` | 44 | **\$0.88** | \$3.00 / \$0.15 / \$15.00 | Quick Q&A, triage, mechanical doc syncs (beats Hydrogen Medium 44 vs 41 for ~same cost) |
| **Rare Escalation** | `gemini-4-neon-xhigh` | 62 | **\$6.44** | \$3.00 / \$0.15 / \$15.00 | Only if Neon High gets stuck on a frontier derivation (+72% cost & 2× time for +2 Intel) |

### A. Orchestrator / Main Agent / Swarm Coordinator
- **Default for routine work** (UI/React, GitHub/CI, PrintDB ingestion, preset cleanup, CAD wiring): **`Gemini 4 Neon (Medium)`** (`gemini-4-neon-medium`, \$1.90/task). Because coding subagents use `Model="inherit"`, keeping routine threads on Neon Medium cuts both parent and subagent cost by ~49%.
- **Complex physics / architecture** (FEA/modal/solid FEM derivations, deep solver debugging): **`Gemini 4 Neon (High)`** (`gemini-4-neon-high`, \$3.74/task) or **`Orchestrator (Neon)`** (`gemini-pro-orchestrator-neon`).
- **Quota fallback order** (if daily Neon/Hydrogen pool runs out): **`Claude Opus 5.5 (High)`** (`claude-opus-5-5-high`) first → **`Orchestrator (Argon)`** (`gemini-pro-orchestrator-g1-pro`) last.
- Do **not** keep an orchestrator on `gemini-4-neon-xhigh` for routine routing turns (+72% cost & 61s TTFT for +2 Intelligence points).

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
2. **Keep Context Lean (< 150k) & Split Topic Pivots:** Aim for **one milestone per thread**; proactively recommend `/compact` or a fresh thread whenever a thread pivots from a quick question into a large multi-file build or crosses ~150k context / ~40 turns. Scope coding subagents to **<= 35 turns** and `browser` subagents to **<= 25 turns** (`<= 10 KB` inline JS extraction, `<= 8` scrolls, `<= 5` image reads — never loop `read_file` on spilled `outputs/step_*.json`, and kill abandoned running browser subagents with `manage_subagents`).
3. **Zero-LLM CAD & FEA First:**
   - Use `uv run printsim cad inspect <part.step>` instead of ad-hoc `python -c` STEP probing.
   - Use declarative `cad_recipes/*.yaml` + `uv run printsim cad bake` (with `--fast` preview contact-sheet PNG and `.cache/cad_fea/` mesh caching) instead of writing one-off baker scripts or taking repeated browser screenshots.
   - Use `build123d` for fast local parametric B-Rep generation and FEA meshing, and use **Onshape MCP** freely whenever inspecting, thickening, modifying, visually rendering (`onshape_shaded_view`), flipping Variables/FeatureScript, or exporting user/public Onshape Part Studios & Assemblies (`onshape_export` / `printsim cad fetch-onshape --kind assemblies`); avoid redundant turn-by-turn primitive sketching when a single FeatureScript or `build123d` script is faster and cleaner.
   - Use `uv run printsim sweep`, `printsim diagnose`, and `printsim compare` instead of ad-hoc `python -c` diagnostic scripts.
4. **Fast Dev-Loop & Lean File-Read Defaults:**
   - Read `docs/CODEBASE_MAP.md` first and use `grep -rn` + tight `StartLine`/`EndLine` slices instead of re-reading huge monolithic files (`printer.py`, `PrinterViewport.tsx`, `builder.py`, `modal.py`) multiple times per thread.
   - Never call `read_file` on `.log` build/bake outputs — use `tail -n 40` via `run_command`.
   - Batch independent reads, edits, and checks in parallel so single-tool turns stay minimal.
   - Run `uv run printsim serve --reload` once per session instead of repeatedly killing/restarting the server.
   - Run targeted `uv run pytest tests/test_<area>.py -q --tb=short` during iteration and `uv run pytest -n auto -q --tb=short` (or `-q --tb=short`) once at final verification.

### C. Explicit Do-Not-Use Models (Pareto-Dominated Guardrails)
Never select or pin these models for PrintSim coding/orchestration roles (enforced globally by `~/.gemini/config/hooks.json` and `spawn_thread.py`):
- **All `gemini-4-carbon-*` (`low` / `medium` / `high`)**: Strictly Pareto-dominated by Neon everywhere — Neon High matches Carbon High (60 Intel) at **half the cost** (\$3.74 vs \$7.76) and 2.7× faster; Neon Medium beats Carbon Low (55 vs 54 Intel) at **3.8× lower cost** (\$1.90 vs \$7.19).
- **`gemini-4-hydrogen-xhigh`**: Strictly dominated by `gemini-4-neon-medium` (55 vs 54 Intel, \$1.90 vs \$3.16/task, 8.8m vs 14.7m).
- **`gemini-4-hydrogen-medium` / `gemini-4-hydrogen-low` for coding/terminal work**: `gemini-4-neon-low` beats Hydrogen Medium (44 vs 41 Intel for \$0.88 vs \$0.80); severe Terminal-Bench cliff on Hydrogen Medium/Low.
- **`gemini-3.8-flash-*`**: Deprecated; low agentic accuracy (20% Terminal-Bench) and high cost-per-task thrashing (\$7.77/task). Use `gemini-4-hydrogen-high` instead.
- **`claude-sonnet-5-5-*` for coding subagents** while capped at `High` in Antigravity: Dominated on coding by `gemini-4-neon-medium` and `gemini-4-hydrogen-high`.
- **`gemini-4-argon-*` / `Orchestrator (Argon)` as primary**: Slowest throughput (69 tok/s, 28m/task) and lower Coding Agent score (45 vs Neon's 59); reserve strictly as weekly-quota backup.
- **`gpt-oss-120b-medium`**: Unsuitable for software engineering or physics tasks (0% FrontierCode / Terminal-Bench).

### D. Automated Enforcement & Daily Cost Tracking (Global)
- **Lifecycle hook**: `~/.gemini/config/hooks.json` (`skills/split-work/scripts/cost_guard.py`) automatically rewrites `Model="pro"` → `inherit`/`flash`, blocks `.log` reads and >2 browser `step_*.json` reads, emits zero-token user-only cost/context banners at `150k ctx / 40t / $3.50`, triggers a split circuit breaker at `250k ctx / $7.00`, and logs each completed thread's dollar cost to `~/.gemini/config/cache/agent_cost_log.jsonl`.
- **Daily spend summary**: Run `python3 /Users/jumpybeetroot/.gemini/config/skills/split-work/scripts/cost_report.py [--days 7] [--workspace printsim] [--scan-transcripts]` for an instant dollar breakdown by project, model, and thread.
