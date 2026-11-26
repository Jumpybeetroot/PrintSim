---
name: discord-research
description: Hands-off, read-only engineering research pipeline for 3D-printing Discord communities (Monolith, VzBoT, Klipper, Luke's Laboratory, Isik's Tech, ANNEX Engineering, Armchair Heavy Industries, Kalico, VORONDesign). Use whenever searching for buried thread specs, toolhead/component weights, gantry/hotend measurements, or Klipper/Shake&Tune ADXL resonance data.
---

# Hands-Off Discord 3D-Printing Engineering Research

Use this workflow whenever you need real-world 3D printer hardware specs, scale-weighed component/toolhead masses, belt/polymer tests, or Klipper/Shake&Tune ADXL input-shaper data that lives in public 3D-printing Discord servers.

## 1. Check Local Knowledge Base & Route Query First (0 Browser Turns)

Never open Discord cold. Always run the zero-LLM router first:

```bash
uv run python .agents/skills/discord-research/scripts/route_query.py "<query>" [--topic adxl|weights|toolheads|hotends|gantry|belts|frame|steppers|rails|cooling|probes|polymers] [--server monolith|vzbot|klipper|lukes_lab|isiks_tech|annex|ahi|kalico|voron]
```

This does two things in <1 second:
1. Searches PrintSim's local SQLite+FTS5 Touchstone KB (`data/touchstones/printsim_kb.db`) so you never re-search data already captured.
2. Ranks the exact Discord channel URLs (`https://discord.com/channels/<guild_id>/<channel_id>`) from [`servers.yaml`](servers.yaml) and generates ready-to-use Discord search queries (`in:<channel> ...`, `has:file`, `has:image`) plus `[CHECK PINS FIRST]` hints.

If the question may also be documented on GitHub repos, VoronDB, Klipper Discourse, Answer Overflow, or Discord's public **Discoverable Server Web Pages** (`site:discord.com/servers`), run `search_web` in parallel before or alongside Discord lookup. Note that Discoverable Server Web Pages expose anonymized summaries/snippets without raw ADXL `.csv` attachments or full pinned tables, so still use Step 2 (read-only Pins + native search) for verified measurements and raw CSVs.

## 2. Dispatch a Fast, Read-Only Browser Subagent

When the local KB does not yet contain the answer, invoke the `browser` subagent (`Model="flash"`) with a tightly scoped, hands-off brief using the exact URLs and search strings from Step 1.

**Mandatory Speed & Reliability Playbook (include in every browser prompt):**
1. **Jump straight to direct channel URLs**: Navigate directly to `https://discord.com/channels/<guild_id>/<channel_id>`. Never hunt through server sidebar icons.
2. **Check Pinned Messages first** on channels flagged `[CHECK PINS FIRST]`:
   - Click `[aria-label*="Pinned Messages"]` and extract all pinned message texts, authors, timestamps, and attachment links in a single DOM read. Most curated weight tables, BOMs, and benchmark links live in pins.
3. **Use Discord Native Search Operators** instead of scrolling chat history:
   - Focus the search bar (`[role="combobox"][aria-label*="Search"]` or `Cmd/Ctrl+F`) and run the exact `in:<channel>` queries from `route_query.py`:
     - **ADXL / Shaper data**: `in:<channel> has:image shaper`, `in:<channel> has:file csv`, `in:<channel> MZV`, `in:<channel> shaper_freq`
     - **Toolhead & Hardware weights**: `in:<channel> weight`, `in:<channel> grams`, `in:<channel> has:image scale`
     - **Forum channels** (`#monolith-builds`, `#user-projects`, `#build-log`, `#mods`, `#print-support`): Use the forum's built-in search/filter input (`Search posts`) to jump straight to matching threads.
4. **Batch DOM Extraction (1 script call per search page)**:
   - Do **not** scroll message-by-message taking screenshots. Run a single DOM script over the search results pane (`[id^="search-results"]` / `[class*="searchResult"]` / `li[id^="chat-messages-"]`) to collect `{author, timestamp, content, attachment_urls, message_id}` for the top 15–25 hits at once.
   - Click **Jump** (or open the thread) **only** on the top 2–5 highest-signal hits to read surrounding reply context (usually ±10 messages around the jumped message).
5. **Download Raw ADXL CSVs & Key Spec Images**:
   - When a message attaches a Klipper/Shake&Tune `.csv` (`resonances_x_*.csv`, `resonances_y_*.csv`, `calibration_data_*.csv`) or a clear shaper plot / scale photo, download it so the parent agent can inspect it or run `calibrate_adxl_csv`.
6. **Strict Read-Only & Safety Guardrails**:
   - Never send messages, never add reactions, never click join/accept prompts unprompted, never read or exfiltrate user auth tokens, and never attempt to bypass login/CAPTCHA gates (pause and ask the user if signed out).
   - If searching 2+ completely separate servers in one request, prefer sequential visits in one browser subagent (Discord shares one authenticated browser tab cleanly without session collisions).

## 3. Auto-Persist Every Verified Finding to PrintSim KB

Never let a Discord lookup evaporate at the end of a session! After the browser subagent returns verified findings:

1. **Save structured capture JSON** under `data/touchstones/discord_captures/<YYYYMMDD>_<topic_slug>.json` (this directory is gitignored under `data/touchstones/`) using the `DiscordChatExporter`-compatible format natively parsed by `printsim kb ingest`:
   ```json
   {
     "guild": {"id": "<guild_id>", "name": "<server_name>"},
     "channel": {"id": "<channel_id>", "name": "<channel_name>"},
     "messages": [
       {
         "id": "<message_id>",
         "timestamp": "2026-09-08T14:22:00Z",
         "author": {"username": "<author>"},
         "url": "https://discord.com/channels/<guild_id>/<channel_id>/<message_id>",
         "content": "<verbatim or structured engineering finding with exact numbers, units, and config>",
         "attachments": [{"filename": "resonances_x.csv", "url": "..."}]
       }
     ]
   }
   ```
2. **Ingest into SQLite + FTS5 immediately**:
   ```bash
   uv run printsim kb ingest data/touchstones/discord_captures/<YYYYMMDD>_<topic_slug>.json --auto-approve
   ```
   (Or run `uv run python .agents/skills/discord-research/scripts/route_query.py --ingest <file>`.)
3. **If ADXL CSV files were downloaded**:
   - Copy them to `data/touchstones/adxl_csvs/`, run `uv run printsim kb ingest data/touchstones/adxl_csvs/<file>.csv --auto-approve`, and optionally run PrintSim's `calibrate_adxl_csv` MCP tool to extract exact $\omega_0$, $\zeta$, and Klipper shaper recommendations.
4. **Promote major reference touchstones into tracked docs & PrintDB**:
   - For high-confidence calibration data (e.g., scale-weighed toolhead/hotend masses, controlled before/after ADXL benchmarks, or polymer TDS), also append a cited row to `docs/community_touchstones.md` or `docs/monolith_discord_benchmarks.md`, run `uv run printsim kb seed`, and **add the verified hardware/material/reference record to PrintDB** (`/Users/jumpybeetroot/Projects/PrintDB`, via `printdb_add_local` MCP tool or `PrintDB/data/` + `printdb validate && printdb build`).

## 4. Adding More Discord Servers Later

Whenever the user shares a new channel link (`https://discord.com/channels/<guild_id>/<channel_id>`):
1. Dispatch one read-only browser pass to extract its server name and engineering channel list.
2. Append the server + high-signal channels to [`.agents/skills/discord-research/servers.yaml`](servers.yaml).
