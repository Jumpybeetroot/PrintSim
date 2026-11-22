"""Seed importer, local archive batch ingester, and polite public API harvesters (Discourse & GitHub)."""

from __future__ import annotations

import csv
import html
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from printsim.kb.extract import (
    _infer_architecture,
    _infer_belt_core,
    _infer_belt_pluck_hz,
    _infer_belt_width_mm,
    _infer_drive_mode,
    _infer_frame_profile,
    _infer_frame_size_mm,
    _infer_hotend,
    _infer_motors,
    _infer_panel_spec,
    _infer_printer_model,
    _infer_toolhead,
    _infer_voltage,
    _parse_accel_number,
    extract_structured_records,
    infer_preset_hint,
    parse_klipper_shaper_csv,
    score_engineering_signal,
)
from printsim.kb.store import TouchstoneKB

USER_AGENT = "PrintSim-KB/1.0"

DEFAULT_PUBLIC_DISCOURSE_TARGETS: list[tuple[str, list[str]]] = [
    (
        "https://klipper.discourse.group",
        ["SHAPER_CALIBRATE corexy", "input shaper resonance belt tension"],
    ),
    (
        "https://forum.vorondesign.com",
        ["input shaper trident", "AWD monolith resonance"],
    ),
]

DEFAULT_PUBLIC_GITHUB_REPOS: list[tuple[str, list[str]]] = [
    ("Frix-x/klippain-shaketune", ["resonance shaper", "belt tension"]),
    ("VoronDesign/Voron-Trident", ["resonance", "input shaper"]),
]


def _default_docs_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "docs"


def _strip_md(text: str) -> str:
    s = re.sub(r"[`*]+", "", text or "")
    return re.sub(r"\s+", " ", s).strip()


def _strip_html(raw_html: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", raw_html or "", flags=re.IGNORECASE)
    text = re.sub(r"</p\s*>", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"[ \t]+", " ", text).strip()


def _parse_shaper_cell(cell: str) -> tuple[str, float | None, float | None]:
    """Parse a Markdown table shaper cell such as ``MZV @ **86.8 Hz** (≤22k)``."""
    clean = _strip_md(cell)
    shaper_type = ""
    freq_hz: float | None = None
    max_accel: float | None = None

    m_type = re.search(
        r"\b(smooth_mzv|smooth_zv|2hump_ei|3hump_ei|mzv|zvd|zv|ei|raw)\b",
        clean,
        re.IGNORECASE,
    )
    if m_type:
        shaper_type = m_type.group(1).lower()

    m_hz = re.search(r"~?\s*(\d+(?:\.\d+)?)\s*hz\b", clean, re.IGNORECASE)
    if m_hz:
        freq_hz = float(m_hz.group(1))

    m_acc = re.search(r"\(\s*(?:<=|≤|~)?\s*(\d+(?:\.\d+)?)\s*(k|mm/s\^?2|mm/s²)?\s*\)", clean, re.IGNORECASE)
    if m_acc:
        max_accel = _parse_accel_number(m_acc.group(1), m_acc.group(2))

    return (shaper_type, freq_hz, max_accel)


# ---------------------------------------------------------------------------
# 1. Seed from docs/monolith_discord_benchmarks.md (and other benchmark docs)
# ---------------------------------------------------------------------------

def seed_from_benchmark_docs(
    kb: TouchstoneKB,
    docs_dir: Path | None = None,
) -> dict[str, int]:
    """Seed the KB from ``docs/monolith_discord_benchmarks.md`` (builds, belt shootout, panel/creep cases)."""
    root = Path(docs_dir) if docs_dir is not None else _default_docs_dir()
    summary = {
        "files_processed": 0,
        "raw_items_added": 0,
        "builds_added": 0,
        "material_tests_added": 0,
        "troubleshooting_added": 0,
    }
    if not root.exists():
        return summary

    doc_files = sorted(root.glob("*benchmark*.md"))
    if not doc_files and (root / "monolith_discord_benchmarks.md").exists():
        doc_files = [root / "monolith_discord_benchmarks.md"]

    for doc_path in doc_files:
        text = doc_path.read_text(encoding="utf-8")
        summary["files_processed"] += 1
        score, tags = score_engineering_signal(text)
        raw_id, inserted = kb.upsert_raw_item(
            source_type="benchmark_doc",
            source_name=doc_path.name,
            title=doc_path.stem.replace("_", " ").title(),
            raw_text=text,
            signal_score=max(score, 0.95),
            signal_tags=sorted(set(tags + ["shaper_data", "belt_data", "material_data"])),
            review_status="approved",
        )
        if inserted:
            summary["raw_items_added"] += 1

        # A. Parse Markdown benchmark table rows (Build | Frame | Drive & Belts | Gantry / Panels | Measured X | Measured Y)
        lines = text.splitlines()
        in_table = False
        header_cols: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("|") and stripped.endswith("|"):
                cells = [c.strip() for c in stripped.strip("|").split("|")]
                if not in_table:
                    lower_cells = [c.lower() for c in cells]
                    if any("build" in c for c in lower_cells) and any("shaper" in c for c in lower_cells):
                        header_cols = lower_cells
                        in_table = True
                    continue
                if all(re.fullmatch(r"[:\-\s]+", c) for c in cells):
                    continue
                if len(cells) < 6:
                    continue

                build_cell = _strip_md(cells[0])
                frame_cell = _strip_md(cells[1])
                drive_cell = _strip_md(cells[2])
                gantry_cell = _strip_md(cells[3])
                x_cell = cells[4]
                y_cell = cells[5]

                row_combined = f"{build_cell} | {frame_cell} | {drive_cell} | {gantry_cell}"
                x_type, x_hz, x_accel = _parse_shaper_cell(x_cell)
                y_type, y_hz, y_accel = _parse_shaper_cell(y_cell)

                arch = _infer_architecture(row_combined)
                model = _infer_printer_model(row_combined) or build_cell
                frame_profile = _infer_frame_profile(frame_cell) or _infer_frame_profile(row_combined)
                frame_size = _infer_frame_size_mm(f"{build_cell} {frame_cell}")
                drive_mode = _infer_drive_mode(drive_cell) or _infer_drive_mode(row_combined)
                belt_width = _infer_belt_width_mm(drive_cell)
                if belt_width is None:
                    bm = re.search(r"(\d+)(?:\s*[-–]\s*(\d+))?\s*mm", drive_cell)
                    if bm:
                        v1 = float(bm.group(1))
                        v2 = float(bm.group(2)) if bm.group(2) else v1
                        belt_width = round((v1 + v2) / 2.0, 1)
                belt_pluck = _infer_belt_pluck_hz(drive_cell)
                belt_core = _infer_belt_core(drive_cell) or "fiberglass"
                voltage = _infer_voltage(drive_cell)
                motors = _infer_motors(drive_cell)
                panel_spec = _infer_panel_spec(gantry_cell) or gantry_cell
                preset = infer_preset_hint(
                    architecture=arch,
                    printer_model=model,
                    frame_profile=frame_profile,
                    drive_mode=drive_mode,
                    belt_width_mm=belt_width,
                    panel_spec=panel_spec,
                )

                before = kb.conn.execute("SELECT COUNT(*) FROM builds").fetchone()[0]
                kb.add_build(
                    raw_item_id=raw_id,
                    build_name=build_cell,
                    architecture=arch,
                    printer_model=model,
                    frame_profile=frame_profile,
                    frame_size_mm=frame_size,
                    panel_spec=panel_spec,
                    drive_mode=drive_mode,
                    belt_width_mm=belt_width,
                    belt_core=belt_core,
                    belt_pluck_hz=belt_pluck,
                    motors=motors,
                    voltage_v=voltage,
                    toolhead=_infer_toolhead(row_combined),
                    hotend=_infer_hotend(row_combined),
                    x_shaper_type=x_type,
                    x_shaper_hz=x_hz,
                    x_max_accel=x_accel,
                    y_shaper_type=y_type,
                    y_shaper_hz=y_hz,
                    y_max_accel=y_accel,
                    preset_hint=preset,
                    notes=f"Frame: {frame_cell}; Drive: {drive_cell}; Gantry/Panels: {gantry_cell}",
                )
                after = kb.conn.execute("SELECT COUNT(*) FROM builds").fetchone()[0]
                if after > before:
                    summary["builds_added"] += 1
            else:
                if in_table and stripped:
                    in_table = False

        # B. Parse Matched-Tension Belt Core Shootout & Panel Material Comparisons
        for raw_line in lines:
            clean = _strip_md(raw_line).lstrip("-*• ").strip()
            if not clean:
                continue

            # Belt shootout bullet: "Gates GT2 EPDM (Fiberglass): 113.8 Hz ZV (@ 283 Hz pluck)"
            bm = re.match(
                r"^([^:]{4,60}):\s*(\d+(?:\.\d+)?)\s*Hz\s*(?:ZV|MZV|EI|smooth_zv|smooth_mzv)\s*\(@\s*(\d+(?:\.\d+)?)\s*Hz\s*pluck\)(.*)$",
                clean,
                re.IGNORECASE,
            )
            if bm:
                mat_name = bm.group(1).strip()
                shaper_hz = float(bm.group(2))
                pluck_hz = float(bm.group(3))
                extra_note = bm.group(4).strip(" ,;-")
                hdt_m = re.search(r"at\s*(\d+(?:\.\d+)?)(?:\s*[-–]\s*\d+)?\s*°?C", extra_note, re.IGNORECASE)
                before = kb.conn.execute("SELECT COUNT(*) FROM material_tests").fetchone()[0]
                kb.add_material_test(
                    raw_item_id=raw_id,
                    material_name=mat_name,
                    hdt_c=float(hdt_m.group(1)) if hdt_m else None,
                    creep_limit_n=89.0,  # ~20 lbf matched tension shootout
                    belt_pluck_hz=pluck_hz,
                    shaper_hz=shaper_hz,
                    test_notes=f"Matched-tension 10mm belt core shootout (~20 lbf): {clean}",
                )
                after = kb.conn.execute("SELECT COUNT(*) FROM material_tests").fetchone()[0]
                if after > before:
                    summary["material_tests_added"] += 1
                continue

            # Panel material bullet notes
            if "Al 5052-H32" in clean and "Steel" in clean:
                before = kb.conn.execute("SELECT COUNT(*) FROM material_tests").fetchone()[0]
                kb.add_material_test(
                    raw_item_id=raw_id,
                    material_name="Aluminum 5052-H32 (4-5mm Structural Panel)",
                    modulus_gpa=70.3,
                    test_notes=clean,
                )
                after = kb.conn.execute("SELECT COUNT(*) FROM material_tests").fetchone()[0]
                if after > before:
                    summary["material_tests_added"] += 1
            elif clean.startswith("Thin ACM"):
                before = kb.conn.execute("SELECT COUNT(*) FROM material_tests").fetchone()[0]
                kb.add_material_test(
                    raw_item_id=raw_id,
                    material_name="ACM / Dibond (3mm Panel)",
                    test_notes=clean,
                )
                after = kb.conn.execute("SELECT COUNT(*) FROM material_tests").fetchone()[0]
                if after > before:
                    summary["material_tests_added"] += 1
            elif clean.startswith("Polycarbonate / Acrylic"):
                before = kb.conn.execute("SELECT COUNT(*) FROM material_tests").fetchone()[0]
                kb.add_material_test(
                    raw_item_id=raw_id,
                    material_name="Polycarbonate / Acrylic Enclosure Panel",
                    hdt_c=65.0,
                    test_notes=clean,
                )
                after = kb.conn.execute("SELECT COUNT(*) FROM material_tests").fetchone()[0]
                if after > before:
                    summary["material_tests_added"] += 1

        # C. Parse recurring failure modes / troubleshooting rules documented in the benchmark doc
        doc_trouble_rules: list[tuple[str, dict[str, str]]] = [
            (
                "compressible foam gasket",
                {
                    "printer_class": "CoreXY Enclosed Frame",
                    "symptom": "Structural panel shear decoupling / lower frame stiffness",
                    "measured_signature": "Missing +16% to +19% side-panel IS frequency gain",
                    "root_cause": "Compressible foam gasket between structural panels and extrusion T-slots decouples shear transfer.",
                    "verified_fix": "Bolt 4-5 mm Aluminum 5052-H32 panels rigidly directly to extrusion T-slots with no foam gasket.",
                },
            ),
            (
                "loses tension at 60",
                {
                    "printer_class": "High-Temp Enclosed CoreXY AWD",
                    "symptom": "Belt tension drop in heated chamber",
                    "measured_signature": "TPU + Kevlar 2GT drops tension at 60-70 C chamber temps",
                    "root_cause": "Thermoplastic polyurethane (TPU) belt body softens and creeps in enclosed chambers at 60-70 C.",
                    "verified_fix": "Switch to EPDM rubber + Aramid (Kevlar) or Gates GT2 EPDM fiberglass belts.",
                },
            ),
            (
                "Steel-Core Belts",
                {
                    "printer_class": "CoreXY (20T pulleys/idlers)",
                    "symptom": "Premature belt core fracture",
                    "measured_signature": "Rapid fatigue fracture around 20T pulleys",
                    "root_cause": "Steel-core belts exceed bending fatigue limit around tight 20T pulley/idler bend radii.",
                    "verified_fix": "Use fiberglass or aramid (Kevlar) core EPDM 2GT belts instead of steel-core belts.",
                },
            ),
            (
                "Printed Mount Creep",
                {
                    "printer_class": "High-Tension AWD CoreXY",
                    "symptom": "Motor/idler mount creep and loss of belt tension",
                    "measured_signature": "ABS/ASA and even annealed PA-CF creep under ~20 lbf (89 N) AWD tension",
                    "root_cause": "Polymer viscoelastic creep under sustained high belt preload at elevated chamber temperature.",
                    "verified_fix": "Upgrade mounts to annealed PPA-CF/GF, PPS-CF/GF, PET-CF, or CNC/SLM aluminum.",
                },
            ),
            (
                "peak / sqrt(2)",
                {
                    "printer_class": "Klipper / Shake&Tune Diagnostics",
                    "symptom": "Inflated/inconsistent Shake&Tune damping ratio calculation",
                    "measured_signature": "PSD half-power bandwidth quirk (peak / sqrt(2) vs peak / 2)",
                    "root_cause": "Shake&Tune historically used peak/sqrt(2) on PSD instead of power half-point peak/2 (amplitude peak/sqrt(2)).",
                    "verified_fix": "Account for PSD vs amplitude half-power definition and use >150 Hz scans on stiff AWD builds.",
                },
            ),
        ]
        for needle, case in doc_trouble_rules:
            if needle.lower() in text.lower():
                before = kb.conn.execute("SELECT COUNT(*) FROM troubleshooting_cases").fetchone()[0]
                kb.add_troubleshooting_case(raw_item_id=raw_id, **case)
                after = kb.conn.execute("SELECT COUNT(*) FROM troubleshooting_cases").fetchone()[0]
                if after > before:
                    summary["troubleshooting_added"] += 1

    return summary


# ---------------------------------------------------------------------------
# 2. Local File / Directory Batch Importer
# ---------------------------------------------------------------------------

def _normalize_discord_attachments(raw_atts: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_atts, list):
        return []
    out: list[dict[str, Any]] = []
    for att in raw_atts:
        if not isinstance(att, dict):
            continue
        out.append(
            {
                "filename": str(att.get("filename") or att.get("fileName") or att.get("name") or ""),
                "url": str(att.get("url") or ""),
                "content_type": str(att.get("content_type") or att.get("contentType") or ""),
            }
        )
    return out


def _extract_author_name(author_field: Any) -> str:
    if isinstance(author_field, dict):
        return str(
            author_field.get("nickname")
            or author_field.get("global_name")
            or author_field.get("name")
            or author_field.get("username")
            or ""
        )
    return str(author_field or "")


def _is_klipper_shaper_csv(path: Path) -> bool:
    if path.suffix.lower() != ".csv":
        return False
    try:
        head = path.read_text(encoding="utf-8", errors="replace").splitlines()[:15]
    except OSError:
        return False
    for line in head:
        low = line.strip().lstrip("#").lower()
        if "freq" in low and "psd" in low:
            return True
    return False


def _ingest_single_record(
    kb: TouchstoneKB,
    *,
    source_type: str,
    source_name: str,
    channel: str,
    author: str,
    url: str,
    published_at: str,
    title: str,
    raw_text: str,
    attachments: list[dict[str, Any]] | None,
    min_score: float,
    auto_approve: bool,
    stats: dict[str, int],
    forced_score: float | None = None,
    forced_tags: list[str] | None = None,
    extra_extracted: dict[str, list[dict[str, Any]]] | None = None,
) -> None:
    stats["items_seen"] += 1
    score, tags = score_engineering_signal(f"{title}\n{raw_text}", attachments=attachments)
    if forced_score is not None:
        score = max(score, forced_score)
    if forced_tags:
        tags = sorted(set(tags + forced_tags))

    if score < min_score:
        stats["items_skipped_low_signal"] += 1
        return

    review_status = "approved" if auto_approve else "pending"
    item_id, inserted = kb.upsert_raw_item(
        source_type=source_type,
        source_name=source_name,
        channel=channel,
        author=author,
        url=url,
        published_at=published_at,
        title=title,
        raw_text=raw_text,
        attachments=attachments or [],
        signal_score=score,
        signal_tags=tags,
        review_status=review_status,
    )
    if not inserted:
        stats["items_deduplicated"] += 1
        return

    stats["items_kept"] += 1
    if auto_approve:
        extracted = extra_extracted or extract_structured_records(
            raw_text,
            title=title,
            meta={"author": author, "channel": channel, "source_name": source_name},
        )
        ins = kb.insert_extracted_records(extracted, raw_item_id=item_id)
        for k, v in ins.items():
            stats[k] += v


def ingest_local_path(
    kb: TouchstoneKB,
    path: Path,
    min_score: float = 0.20,
    auto_approve: bool = False,
) -> dict[str, int]:
    """Recursively ingest Discord export JSON, JSON/JSONL, Markdown/TXT, and Klipper shaper CSV files."""
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")

    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    stats: dict[str, int] = {
        "files_scanned": 0,
        "items_seen": 0,
        "items_kept": 0,
        "items_skipped_low_signal": 0,
        "items_deduplicated": 0,
        "builds_added": 0,
        "weights_added": 0,
        "material_tests_added": 0,
        "troubleshooting_added": 0,
    }

    supported_exts = {".json", ".jsonl", ".md", ".txt", ".csv"}
    for fpath in files:
        ext = fpath.suffix.lower()
        if ext not in supported_exts:
            continue
        stats["files_scanned"] += 1

        # 1. Klipper / Shake&Tune shaper CSV
        if ext == ".csv" and _is_klipper_shaper_csv(fpath):
            parsed = parse_klipper_shaper_csv(fpath)
            axis = parsed["axis"]
            peak_hz = parsed["peak_freq_hz"]
            fitted = parsed["fitted_shapers"]
            shaper_type = next(iter(fitted.keys()), "mzv") if fitted else ""
            shaper_hz = next(iter(fitted.values()), peak_hz) if fitted else peak_hz
            summary_text = (
                f"Klipper shaper CSV {fpath.name}: {axis}: {shaper_type or 'raw'} @ {shaper_hz} Hz "
                f"(peak PSD {parsed['peak_psd']} at {peak_hz} Hz, {parsed['sample_count']} samples)"
            )
            extra = {
                "builds": [
                    {
                        "build_name": fpath.stem,
                        "architecture": "corexy",
                        "x_shaper_type": shaper_type if axis == "X" else "",
                        "x_shaper_hz": shaper_hz if axis == "X" else None,
                        "y_shaper_type": shaper_type if axis == "Y" else "",
                        "y_shaper_hz": shaper_hz if axis == "Y" else None,
                        "preset_hint": "corexy_voron_style",
                        "notes": summary_text,
                    }
                ],
                "component_weights": [],
                "material_tests": [],
                "troubleshooting_cases": [],
            }
            _ingest_single_record(
                kb,
                source_type="shaketune_csv",
                source_name=fpath.name,
                channel="",
                author="",
                url=str(fpath.resolve()),
                published_at="",
                title=f"Shaper CSV ({axis} peak {peak_hz} Hz) - {fpath.name}",
                raw_text=summary_text,
                attachments=[{"filename": fpath.name, "url": str(fpath.resolve())}],
                min_score=min_score,
                auto_approve=auto_approve,
                stats=stats,
                forced_score=0.95,
                forced_tags=["shaper_data", "shaper_attachment"],
                extra_extracted=extra,
            )
            continue

        # 2. Generic CSV
        if ext == ".csv":
            with fpath.open("r", encoding="utf-8", errors="replace", newline="") as fh:
                reader = csv.DictReader(fh)
                for idx, row in enumerate(reader, start=1):
                    row_text = " | ".join(f"{k}: {v}" for k, v in row.items() if v)
                    _ingest_single_record(
                        kb,
                        source_type="local_export",
                        source_name=fpath.name,
                        channel="",
                        author=str(row.get("author") or ""),
                        url="",
                        published_at=str(row.get("timestamp") or row.get("date") or ""),
                        title=str(row.get("title") or row.get("build_name") or f"{fpath.stem} row {idx}"),
                        raw_text=row_text,
                        attachments=[],
                        min_score=min_score,
                        auto_approve=auto_approve,
                        stats=stats,
                    )
            continue

        # 3. JSON (Discord export format or generic JSON)
        if ext == ".json":
            try:
                data = json.loads(fpath.read_text(encoding="utf-8", errors="replace"))
            except json.JSONDecodeError:
                continue

            # DiscordChatExporter format: {"guild": {...}, "channel": {...}, "messages": [...]}
            if isinstance(data, dict) and isinstance(data.get("messages"), list):
                guild = data.get("guild") or {}
                channel_obj = data.get("channel") or {}
                guild_name = str(guild.get("name") or fpath.stem) if isinstance(guild, dict) else fpath.stem
                chan_name = str(channel_obj.get("name") or channel_obj) if isinstance(channel_obj, dict) else str(channel_obj)
                for msg in data["messages"]:
                    if not isinstance(msg, dict):
                        continue
                    content = str(msg.get("content") or "")
                    atts = _normalize_discord_attachments(msg.get("attachments"))
                    if not content.strip() and not atts:
                        continue
                    author = _extract_author_name(msg.get("author"))
                    ts = str(msg.get("timestamp") or "")
                    msg_id = str(msg.get("id") or "")
                    url = str(msg.get("url") or (f"discord://{guild_name}/{chan_name}/{msg_id}" if msg_id else ""))
                    _ingest_single_record(
                        kb,
                        source_type="local_export",
                        source_name=guild_name,
                        channel=chan_name,
                        author=author,
                        url=url,
                        published_at=ts,
                        title=f"{author} in #{chan_name}" if chan_name else author or fpath.stem,
                        raw_text=content,
                        attachments=atts,
                        min_score=min_score,
                        auto_approve=auto_approve,
                        stats=stats,
                    )
                continue

            # List of messages or records
            records = data if isinstance(data, list) else [data]
            for idx, rec in enumerate(records, start=1):
                if not isinstance(rec, dict):
                    continue
                content = str(rec.get("content") or rec.get("raw_text") or rec.get("body") or rec.get("text") or "")
                atts = _normalize_discord_attachments(rec.get("attachments"))
                if not content.strip() and not atts:
                    continue
                author = _extract_author_name(rec.get("author"))
                chan = str(rec.get("channel") or "")
                title = str(rec.get("title") or (f"{author} #{chan}".strip() if author else f"{fpath.stem} #{idx}"))
                _ingest_single_record(
                    kb,
                    source_type="local_export",
                    source_name=str(rec.get("source_name") or fpath.name),
                    channel=chan,
                    author=author,
                    url=str(rec.get("url") or ""),
                    published_at=str(rec.get("timestamp") or rec.get("published_at") or ""),
                    title=title,
                    raw_text=content,
                    attachments=atts,
                    min_score=min_score,
                    auto_approve=auto_approve,
                    stats=stats,
                )
            continue

        # 4. JSONL
        if ext == ".jsonl":
            for idx, line in enumerate(fpath.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(rec, dict):
                    continue
                content = str(rec.get("content") or rec.get("raw_text") or rec.get("body") or rec.get("text") or "")
                atts = _normalize_discord_attachments(rec.get("attachments"))
                _ingest_single_record(
                    kb,
                    source_type="local_export",
                    source_name=str(rec.get("source_name") or fpath.name),
                    channel=str(rec.get("channel") or ""),
                    author=_extract_author_name(rec.get("author")),
                    url=str(rec.get("url") or ""),
                    published_at=str(rec.get("timestamp") or rec.get("published_at") or ""),
                    title=str(rec.get("title") or f"{fpath.stem} line {idx}"),
                    raw_text=content,
                    attachments=atts,
                    min_score=min_score,
                    auto_approve=auto_approve,
                    stats=stats,
                )
            continue

        # 5. Markdown / plain text
        text = fpath.read_text(encoding="utf-8", errors="replace")
        first_heading = ""
        for ln in text.splitlines():
            if ln.strip().startswith("#"):
                first_heading = ln.strip().lstrip("#").strip()
                break
        _ingest_single_record(
            kb,
            source_type="local_export",
            source_name=fpath.name,
            channel="",
            author="",
            url=str(fpath.resolve()),
            published_at="",
            title=first_heading or fpath.stem,
            raw_text=text,
            attachments=[],
            min_score=min_score,
            auto_approve=auto_approve,
            stats=stats,
        )

    return stats


# ---------------------------------------------------------------------------
# 3. Public Discourse JSON Harvester
# ---------------------------------------------------------------------------

def _http_get_json(url: str, headers: dict[str, str], timeout_s: float = 20.0) -> Any:
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def harvest_discourse(
    kb: TouchstoneKB,
    base_url: str,
    queries: list[str],
    max_topics: int = 20,
    auto_approve: bool = False,
    delay_s: float = 0.8,
    min_score: float = 0.20,
) -> dict[str, Any]:
    """Politely harvest public Discourse forum topics via ``/search.json`` and ``/t/{id}.json``."""
    base = base_url.rstrip("/")
    source_name = urllib.parse.urlparse(base).netloc or base
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}

    stats: dict[str, Any] = {
        "source": base,
        "topics_fetched": 0,
        "items_seen": 0,
        "items_kept": 0,
        "items_skipped_low_signal": 0,
        "items_deduplicated": 0,
        "builds_added": 0,
        "weights_added": 0,
        "material_tests_added": 0,
        "troubleshooting_added": 0,
        "errors": [],
    }

    seen_topic_ids: list[int] = []
    seen_set: set[int] = set()
    req_count = 0

    for q in queries:
        if len(seen_topic_ids) >= max_topics:
            break
        if req_count > 0 and delay_s > 0:
            time.sleep(delay_s)
        req_count += 1
        search_url = f"{base}/search.json?q={urllib.parse.quote_plus(q)}"
        try:
            payload = _http_get_json(search_url, headers=headers)
        except Exception as exc:  # noqa: BLE001
            stats["errors"].append(f"search '{q}': {exc}")
            continue

        topics = payload.get("topics") or []
        for t in topics:
            tid = t.get("id")
            if isinstance(tid, int) and tid not in seen_set:
                seen_set.add(tid)
                seen_topic_ids.append(tid)
                if len(seen_topic_ids) >= max_topics:
                    break

    for tid in seen_topic_ids[:max_topics]:
        if req_count > 0 and delay_s > 0:
            time.sleep(delay_s)
        req_count += 1
        topic_url = f"{base}/t/{tid}.json"
        try:
            topic_data = _http_get_json(topic_url, headers=headers)
        except Exception as exc:  # noqa: BLE001
            stats["errors"].append(f"topic {tid}: {exc}")
            continue

        stats["topics_fetched"] += 1
        title = str(topic_data.get("title") or f"Topic {tid}")
        slug = str(topic_data.get("slug") or "")
        web_url = f"{base}/t/{slug}/{tid}" if slug else f"{base}/t/{tid}"
        posts = (topic_data.get("post_stream") or {}).get("posts") or []
        post_texts: list[str] = []
        first_author = ""
        published_at = str(topic_data.get("created_at") or "")
        for p in posts[:6]:
            if not first_author:
                first_author = str(p.get("username") or p.get("name") or "")
            if not published_at:
                published_at = str(p.get("created_at") or "")
            cooked = str(p.get("cooked") or p.get("raw") or "")
            cleaned = _strip_html(cooked)
            if cleaned:
                post_texts.append(cleaned)

        combined_text = "\n\n---\n\n".join(post_texts)
        _ingest_single_record(
            kb,
            source_type="discourse",
            source_name=source_name,
            channel=str(topic_data.get("category_id") or ""),
            author=first_author,
            url=web_url,
            published_at=published_at,
            title=title,
            raw_text=combined_text,
            attachments=[],
            min_score=min_score,
            auto_approve=auto_approve,
            stats=stats,
        )

    return stats


# ---------------------------------------------------------------------------
# 4. Public GitHub Issues / Discussions Harvester
# ---------------------------------------------------------------------------

def harvest_github_issues(
    kb: TouchstoneKB,
    repo: str,
    queries: list[str],
    max_items: int = 20,
    auto_approve: bool = False,
    token: str | None = None,
    delay_s: float = 0.8,
    min_score: float = 0.20,
) -> dict[str, Any]:
    """Politely harvest public GitHub issues via ``/search/issues``."""
    auth_token = token or os.environ.get("GITHUB_TOKEN", "").strip() or None
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/vnd.github+json",
    }
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    stats: dict[str, Any] = {
        "repo": repo,
        "issues_fetched": 0,
        "items_seen": 0,
        "items_kept": 0,
        "items_skipped_low_signal": 0,
        "items_deduplicated": 0,
        "builds_added": 0,
        "weights_added": 0,
        "material_tests_added": 0,
        "troubleshooting_added": 0,
        "errors": [],
    }

    seen_urls: set[str] = set()
    req_count = 0

    for q in queries:
        if stats["issues_fetched"] >= max_items:
            break
        if req_count > 0 and delay_s > 0:
            time.sleep(delay_s)
        req_count += 1

        remaining = max(1, max_items - stats["issues_fetched"])
        per_page = min(30, remaining)
        full_q = f"repo:{repo} {q}".strip()
        url = (
            "https://api.github.com/search/issues"
            f"?q={urllib.parse.quote_plus(full_q)}&per_page={per_page}"
        )
        try:
            payload = _http_get_json(url, headers=headers)
        except Exception as exc:  # noqa: BLE001
            stats["errors"].append(f"query '{q}': {exc}")
            continue

        for issue in payload.get("items") or []:
            if stats["issues_fetched"] >= max_items:
                break
            html_url = str(issue.get("html_url") or "")
            if html_url and html_url in seen_urls:
                continue
            if html_url:
                seen_urls.add(html_url)
            stats["issues_fetched"] += 1

            title = str(issue.get("title") or "")
            body = str(issue.get("body") or "")
            user = issue.get("user") or {}
            author = str(user.get("login") or "") if isinstance(user, dict) else ""
            created_at = str(issue.get("created_at") or "")

            _ingest_single_record(
                kb,
                source_type="github",
                source_name=repo,
                channel="issues",
                author=author,
                url=html_url,
                published_at=created_at,
                title=title,
                raw_text=body,
                attachments=[],
                min_score=min_score,
                auto_approve=auto_approve,
                stats=stats,
            )

    return stats
