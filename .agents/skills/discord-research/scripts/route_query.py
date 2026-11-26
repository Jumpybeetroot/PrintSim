#!/usr/bin/env python3
"""Zero-LLM Discord engineering research router + PrintSim Touchstone KB bridge.

Usage:
  # 1. Check local KB first + get exact Discord channel URLs & search queries:
  uv run python .agents/skills/discord-research/scripts/route_query.py "chube air weight"
  uv run python .agents/skills/discord-research/scripts/route_query.py "vzbot 330 adxl" --topic adxl --server vzbot

  # 2. Save a captured Discord JSON batch & ingest it directly into printsim_kb.db:
  uv run python .agents/skills/discord-research/scripts/route_query.py --ingest data/touchstones/discord_captures/foo.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

SKILL_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SKILL_DIR.parents[2]
SERVERS_YAML = SKILL_DIR / "servers.yaml"

# Topic keyword expansions for channel scoring & Discord native search queries
TOPIC_SYNONYMS: dict[str, set[str]] = {
    "adxl": {"adxl", "shaper", "mzv", "zv", "zvd", "ei", "smooth_zv", "smooth_mzv", "resonance", "shaketune", "frequency", "hz", "graph", "psd", "damping", "input"},
    "weights": {"weight", "weights", "mass", "gram", "grams", "scale", "weigh", "weighed", "lightweight", "ultralight"},
    "toolheads": {"toolhead", "toolheads", "printhead", "xol", "a4t", "archetype", "stealthburner", "burner", "calamity", "takeoff", "vz-printhead", "carriage", "shuttle", "ahi", "armchair"},
    "hotends": {"hotend", "hotends", "chube", "pika", "goliath", "rapido", "dragon", "revo", "flow", "volumetric", "melt", "nozzle", "heater", "thermal", "mpc"},
    "gantry": {"gantry", "xy", "joint", "joints", "beam", "xbeam", "carbonara", "carbon", "cf", "awd", "2wd", "monolith", "sheet", "cnc", "slm", "idler", "k3", "crossed"},
    "belts": {"belt", "belts", "gt2", "2gt", "gt3", "at3", "pluck", "tension", "kevlar", "aramid", "epdm", "tpu", "gates"},
    "frame": {"frame", "panel", "panels", "extrusion", "extrusions", "2020", "3030", "4040", "5052", "acm", "brace", "stiffness", "trident", "voron", "k3"},
    "steppers": {"stepper", "steppers", "motor", "motors", "servo", "servos", "ldo", "kraken", "tmc", "5160", "48v", "torque", "driver", "drivers"},
    "rails": {"rail", "rails", "mgn", "mgn9", "mgn12", "mgn15", "hiwin", "preload", "carriage", "linear"},
    "cooling": {"cooling", "cpap", "roborock", "fan", "fans", "5015", "4010", "duct", "watercooling", "archetype"},
    "probes": {"probe", "probes", "beacon", "cartographer", "euclid", "tap", "eddy", "klicky", "quickdraw", "leveling"},
    "polymers": {"polymer", "polymers", "plastic", "ppa", "pps", "pet-cf", "pa12", "abs", "asa", "creep", "hdt", "anneal", "annealing", "modulus"},
    "extruders": {"extruder", "extruders", "hextrudort", "sherpa", "orbiter", "galileo", "cw2", "wristwatch", "wwg2", "gear"},
}

TOPIC_SEARCH_MODIFIERS: dict[str, list[str]] = {
    "adxl": ["has:image shaper", "has:file csv", "MZV OR ZV", "shaper_freq"],
    "weights": ["grams OR weight", "has:image scale", "g"],
    "hotends": ["weight OR flow", "mm3/s OR grams"],
    "gantry": ["weight OR shaper", "has:image"],
    "belts": ["pluck OR tension", "Hz"],
    "frame": ["shaper OR panel", "5052 OR 4mm"],
    "steppers": ["48v OR torque", "weight OR amps"],
    "rails": ["preload OR weight", "MGN9 OR MGN12"],
    "probes": ["weight OR grams"],
    "polymers": ["creep OR HDT", "anneal"],
}


def _tokenize(text: str) -> set[str]:
    raw = (text or "").lower()
    parts = set(re.findall(r"[a-z0-9]+", raw))
    # Also retain hyphenated compounds like pet-cf or pa12-cf
    parts |= set(re.findall(r"[a-z0-9]+-[a-z0-9]+", raw))
    return parts


def load_servers(path: Path = SERVERS_YAML) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data.get("servers") or {}


def infer_topics(query: str, explicit_topic: str | None = None) -> list[str]:
    tokens = _tokenize(query)
    scored: list[tuple[int, str]] = []
    for topic, syns in TOPIC_SYNONYMS.items():
        hits = len(tokens & syns)
        if explicit_topic and topic == explicit_topic.lower():
            hits += 5
        if hits > 0:
            scored.append((hits, topic))
    scored.sort(reverse=True)
    return [t for _, t in scored]


def rank_channels(
    servers: dict[str, Any],
    query: str,
    *,
    server_filter: str | None = None,
    topic: str | None = None,
    top_k: int = 8,
) -> list[dict[str, Any]]:
    tokens = _tokenize(query)
    topics = infer_topics(query, explicit_topic=topic)
    topic_set = set(topics)

    # Extract concise core search terms (exclude stop/generic words)
    stop = {"what", "is", "the", "of", "for", "a", "an", "in", "on", "to", "how", "much", "does", "find", "get", "data", "specs", "spec", "discord"}
    core_terms = [t for t in re.findall(r"[A-Za-z0-9_.-]+", query) if t.lower() not in stop]
    core_str = " ".join(core_terms[:4]) or query.strip()

    candidates: list[dict[str, Any]] = []
    for slug, srv in servers.items():
        if server_filter and slug.lower() != server_filter.lower() and server_filter.lower() not in str(srv.get("name", "")).lower():
            continue

        srv_name = str(srv.get("name") or slug)
        guild_id = str(srv.get("guild_id") or "")
        srv_focus = set(t.lower() for t in (srv.get("focus") or []))
        srv_tokens = _tokenize(f"{slug} {srv_name} {srv.get('notes', '')}") | srv_focus

        srv_match = len(tokens & srv_tokens)
        srv_name_hit = 1 if (slug.lower() in tokens or any(w in tokens for w in _tokenize(srv_name) if len(w) > 2)) else 0

        for ch in srv.get("channels") or []:
            ch_id = str(ch["id"])
            ch_name = str(ch["name"])
            ch_type = str(ch.get("type") or "text")
            ch_topics = set(t.lower() for t in (ch.get("topics") or []))
            ch_tokens = _tokenize(f"{ch_name} {ch.get('notes', '')}") | ch_topics

            score = 0.0
            score += 4.5 * srv_name_hit
            score += 1.0 * min(srv_match, 3)
            score += 3.5 * len(tokens & ch_tokens)
            score += 2.5 * len(topic_set & ch_topics)
            if ch.get("pins_priority"):
                score += 0.6
            if ch_type == "forum" and ({"adxl", "builds", "shaper", "weights"} & (topic_set | tokens)):
                score += 0.8

            if score <= 0.6 and not server_filter:
                continue

            # Build ready-to-use Discord search bar queries.
            # Discord search ANDs every token on a single message, so strip tokens
            # already implied by the server or channel name to avoid over-filtering.
            redundant = _tokenize(f"{slug} {srv_name} {ch_name}")
            filtered_terms = [t for t in core_terms if t.lower() not in redundant]
            if not filtered_terms:
                filtered_terms = core_terms[:2]
            ch_query = " ".join(filtered_terms[:3]) or core_str

            queries: list[str] = [f"in:{ch_name} {ch_query}".strip()]
            for tp in topics[:2]:
                for mod in TOPIC_SEARCH_MODIFIERS.get(tp, [])[:1]:
                    first_mod = mod.split(" OR ")[0].strip()
                    if first_mod.lower() not in ch_query.lower():
                        queries.append(f"in:{ch_name} {first_mod}")
            if "adxl" in topic_set:
                queries.append(f"in:{ch_name} has:file")
            elif "weights" in topic_set:
                queries.append(f"in:{ch_name} grams")

            candidates.append(
                {
                    "score": round(score, 2),
                    "server_slug": slug,
                    "server_name": srv_name,
                    "guild_id": guild_id,
                    "channel_id": ch_id,
                    "channel_name": ch_name,
                    "channel_type": ch_type,
                    "pins_priority": bool(ch.get("pins_priority")),
                    "url": f"https://discord.com/channels/{guild_id}/{ch_id}",
                    "notes": str(ch.get("notes") or ""),
                    "search_queries": list(dict.fromkeys(queries))[:3],
                }
            )

    candidates.sort(key=lambda x: (-x["score"], x["server_slug"], x["channel_name"]))
    return candidates[:top_k]


def query_local_kb(query: str, limit: int = 8) -> list[dict[str, Any]]:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    try:
        from printsim.kb import TouchstoneKB
    except Exception:
        return []

    with TouchstoneKB() as kb:
        hits = kb.search(query, limit=limit)
    out = []
    for h in hits:
        rec = h.get("record") or {}
        summary: dict[str, Any] = {
            "item_type": h["item_type"],
            "ref_id": h["ref_id"],
            "title": h["title"],
            "snippet": h["snippet"],
        }
        if h["item_type"] == "component_weight":
            summary["mass_g"] = rec.get("mass_g")
            summary["category"] = rec.get("category")
        elif h["item_type"] == "build":
            summary["x_shaper_hz"] = rec.get("x_shaper_hz")
            summary["y_shaper_hz"] = rec.get("y_shaper_hz")
            summary["x_max_accel"] = rec.get("x_max_accel")
            summary["y_max_accel"] = rec.get("y_max_accel")
        out.append(summary)
    return out


def ingest_capture(path: Path, auto_approve: bool = True, min_score: float = 0.15) -> dict[str, Any]:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from printsim.kb import TouchstoneKB, ingest_local_path

    with TouchstoneKB() as kb:
        res = ingest_local_path(kb, path, min_score=min_score, auto_approve=auto_approve)
        return {"db_path": str(kb.db_path), **res}


def main() -> int:
    parser = argparse.ArgumentParser(description="Route a 3D-printing query to local KB hits + exact Discord channels.")
    parser.add_argument("query", nargs="?", default="", help="Engineering question or keywords (e.g. 'chube compact weight')")
    parser.add_argument("--server", "-s", default=None, help="Filter to one server slug/name (monolith, vzbot, klipper, lukes_lab, isiks_tech, annex, ahi, kalico)")
    parser.add_argument("--topic", "-t", default=None, help="Optional topic boost (adxl, weights, toolheads, hotends, gantry, belts, frame, steppers, rails, cooling, probes, polymers)")
    parser.add_argument("--top", "-n", type=int, default=6, help="Number of Discord channels to return")
    parser.add_argument("--ingest", type=Path, default=None, help="Ingest a Discord capture JSON/CSV into data/touchstones/printsim_kb.db")
    parser.add_argument("--no-auto-approve", action="store_true", help="Keep ingested items in pending review status")
    parser.add_argument("--json", action="store_true", dest="json_out", help="Emit JSON output")
    args = parser.parse_args()

    if args.ingest:
        res = ingest_capture(args.ingest, auto_approve=not args.no_auto_approve)
        print(json.dumps(res, indent=2))
        return 0

    if not args.query.strip():
        parser.error("Provide a search query or --ingest <path>")

    servers = load_servers()
    kb_hits = query_local_kb(args.query)
    topics = infer_topics(args.query, explicit_topic=args.topic)
    channels = rank_channels(servers, args.query, server_filter=args.server, topic=args.topic, top_k=args.top)

    payload = {
        "query": args.query,
        "inferred_topics": topics,
        "local_kb_hits": kb_hits,
        "discord_targets": channels,
    }

    if args.json_out:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"Query: {args.query}")
    print(f"Inferred topics: {', '.join(topics) or 'general'}")
    print()
    print(f"=== 1. Local Touchstone KB Hits ({len(kb_hits)}) ===")
    if not kb_hits:
        print("  (none — proceed to public web / Discord targets below)")
    else:
        for h in kb_hits:
            extra = ""
            if h["item_type"] == "component_weight" and h.get("mass_g") is not None:
                extra = f" [{h['mass_g']} g]"
            elif h["item_type"] == "build" and (h.get("x_shaper_hz") or h.get("y_shaper_hz")):
                extra = f" [X:{h.get('x_shaper_hz')}Hz Y:{h.get('y_shaper_hz')}Hz]"
            print(f"  - [{h['item_type']} #{h['ref_id']}] {h['title']}{extra}")
            print(f"      {h['snippet'][:140]}")

    print()
    print(f"=== 2. Top Discord Targets ({len(channels)}) ===")
    for idx, ch in enumerate(channels, start=1):
        pin_flag = " [CHECK PINS FIRST]" if ch["pins_priority"] else ""
        print(f"  {idx}. {ch['server_name']} / #{ch['channel_name']} ({ch['channel_type']}){pin_flag} (score={ch['score']})")
        print(f"     URL: {ch['url']}")
        if ch["notes"]:
            print(f"     Note: {ch['notes']}")
        print(f"     Discord Search: {' | '.join(ch['search_queries'])}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
