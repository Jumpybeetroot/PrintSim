"""SQLite + FTS5 local 3D printer engineering touchstone knowledge base."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

from printsim.kb.extract import extract_structured_records

DEFAULT_KB_REL_PATH = Path("data/touchstones/printsim_kb.db")


def default_kb_path() -> Path:
    """Resolve the default PrintSim KB SQLite path (respecting ``PRINTSIM_KB_PATH``)."""
    env_val = os.environ.get("PRINTSIM_KB_PATH", "").strip()
    if env_val:
        return Path(env_val)
    return DEFAULT_KB_REL_PATH


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def compute_content_hash(
    source_type: str,
    source_name: str,
    url: str,
    title: str,
    raw_text: str,
) -> str:
    """Deterministic SHA-256 hash used to deduplicate ``raw_items``."""
    norm_text = re.sub(r"\s+", " ", (raw_text or "").strip())
    payload = "\x1f".join(
        [
            (source_type or "").strip().lower(),
            (source_name or "").strip().lower(),
            (url or "").strip(),
            (title or "").strip(),
            norm_text,
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


_TABLE_TO_ITEM_TYPE: dict[str, str] = {
    "raw_items": "raw_item",
    "raw_item": "raw_item",
    "builds": "build",
    "build": "build",
    "component_weights": "component_weight",
    "component_weight": "component_weight",
    "material_tests": "material_test",
    "material_test": "material_test",
    "troubleshooting_cases": "troubleshooting_case",
    "troubleshooting_case": "troubleshooting_case",
}

_ITEM_TYPE_TO_TABLE: dict[str, str] = {
    "raw_item": "raw_items",
    "build": "builds",
    "component_weight": "component_weights",
    "material_test": "material_tests",
    "troubleshooting_case": "troubleshooting_cases",
}


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS raw_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content_hash TEXT UNIQUE NOT NULL,
    source_type TEXT NOT NULL,
    source_name TEXT NOT NULL DEFAULT '',
    channel TEXT NOT NULL DEFAULT '',
    author TEXT NOT NULL DEFAULT '',
    url TEXT NOT NULL DEFAULT '',
    published_at TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    raw_text TEXT NOT NULL DEFAULT '',
    attachments_json TEXT NOT NULL DEFAULT '[]',
    signal_score REAL NOT NULL DEFAULT 0.0,
    signal_tags_json TEXT NOT NULL DEFAULT '[]',
    review_status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS builds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_item_id INTEGER REFERENCES raw_items(id) ON DELETE SET NULL,
    build_name TEXT NOT NULL,
    architecture TEXT NOT NULL DEFAULT 'corexy',
    printer_model TEXT NOT NULL DEFAULT '',
    frame_profile TEXT NOT NULL DEFAULT '',
    frame_size_mm REAL,
    panel_spec TEXT NOT NULL DEFAULT '',
    drive_mode TEXT NOT NULL DEFAULT '',
    belt_width_mm REAL,
    belt_core TEXT NOT NULL DEFAULT '',
    belt_pluck_hz REAL,
    motors TEXT NOT NULL DEFAULT '',
    voltage_v REAL,
    toolhead TEXT NOT NULL DEFAULT '',
    hotend TEXT NOT NULL DEFAULT '',
    x_shaper_type TEXT NOT NULL DEFAULT '',
    x_shaper_hz REAL,
    x_max_accel REAL,
    y_shaper_type TEXT NOT NULL DEFAULT '',
    y_shaper_hz REAL,
    y_max_accel REAL,
    x_moving_mass_g REAL,
    y_moving_mass_g REAL,
    preset_hint TEXT NOT NULL DEFAULT '',
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS component_weights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_item_id INTEGER REFERENCES raw_items(id) ON DELETE SET NULL,
    category TEXT NOT NULL,
    component_name TEXT NOT NULL,
    mass_g REAL NOT NULL,
    details TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS material_tests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_item_id INTEGER REFERENCES raw_items(id) ON DELETE SET NULL,
    material_name TEXT NOT NULL,
    modulus_gpa REAL,
    hdt_c REAL,
    creep_limit_n REAL,
    belt_pluck_hz REAL,
    shaper_hz REAL,
    test_notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS troubleshooting_cases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_item_id INTEGER REFERENCES raw_items(id) ON DELETE SET NULL,
    printer_class TEXT NOT NULL DEFAULT '',
    symptom TEXT NOT NULL,
    measured_signature TEXT NOT NULL DEFAULT '',
    root_cause TEXT NOT NULL DEFAULT '',
    verified_fix TEXT NOT NULL DEFAULT ''
);

CREATE VIRTUAL TABLE IF NOT EXISTS kb_fts USING fts5(
    item_type,
    ref_id UNINDEXED,
    title,
    body,
    tags
);

CREATE INDEX IF NOT EXISTS idx_raw_items_status_score
    ON raw_items(review_status, signal_score DESC);
CREATE INDEX IF NOT EXISTS idx_builds_arch_drive
    ON builds(architecture, drive_mode);
"""


class TouchstoneKB:
    """Zero-dependency SQLite + FTS5 knowledge base for real-world 3D printer benchmarks."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else default_kb_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self.conn.execute("PRAGMA journal_mode = WAL;")
        self.conn.executescript(_SCHEMA_SQL)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> TouchstoneKB:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # FTS5 synchronization
    # ------------------------------------------------------------------

    def _sync_fts(
        self,
        item_type: str,
        ref_id: int,
        title: str,
        body: str,
        tags: str = "",
    ) -> None:
        self.conn.execute(
            "DELETE FROM kb_fts WHERE item_type = ? AND ref_id = ?",
            (item_type, str(ref_id)),
        )
        self.conn.execute(
            "INSERT INTO kb_fts(item_type, ref_id, title, body, tags) VALUES (?, ?, ?, ?, ?)",
            (item_type, str(ref_id), title or "", body or "", tags or ""),
        )

    # ------------------------------------------------------------------
    # Raw items
    # ------------------------------------------------------------------

    def upsert_raw_item(
        self,
        *,
        source_type: str,
        raw_text: str,
        source_name: str = "",
        channel: str = "",
        author: str = "",
        url: str = "",
        published_at: str = "",
        title: str = "",
        attachments: list[dict[str, Any]] | None = None,
        signal_score: float = 0.0,
        signal_tags: list[str] | None = None,
        review_status: str = "pending",
        content_hash: str | None = None,
    ) -> tuple[int, bool]:
        """Insert a raw engineering item if new, deduplicating by ``content_hash``.

        Returns ``(item_id, inserted)``.
        """
        c_hash = content_hash or compute_content_hash(
            source_type=source_type,
            source_name=source_name,
            url=url,
            title=title,
            raw_text=raw_text,
        )
        existing = self.conn.execute(
            "SELECT id FROM raw_items WHERE content_hash = ?",
            (c_hash,),
        ).fetchone()
        if existing is not None:
            return (int(existing["id"]), False)

        att_json = json.dumps(attachments or [], ensure_ascii=False)
        tags_list = list(signal_tags or [])
        tags_json = json.dumps(tags_list, ensure_ascii=False)
        created_at = _utc_now_iso()

        cur = self.conn.execute(
            """
            INSERT INTO raw_items (
                content_hash, source_type, source_name, channel, author, url,
                published_at, title, raw_text, attachments_json,
                signal_score, signal_tags_json, review_status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                c_hash,
                source_type,
                source_name,
                channel,
                author,
                url,
                published_at,
                title,
                raw_text,
                att_json,
                float(signal_score),
                tags_json,
                review_status,
                created_at,
            ),
        )
        item_id = int(cur.lastrowid)
        fts_tags = " ".join([source_type, source_name, channel, author, *tags_list]).strip()
        self._sync_fts(
            "raw_item",
            item_id,
            title or f"{source_name} {channel}".strip(),
            raw_text,
            fts_tags,
        )
        self.conn.commit()
        return (item_id, True)

    def get_raw_item(self, item_id: int) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM raw_items WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["attachments"] = json.loads(d.get("attachments_json") or "[]")
        d["signal_tags"] = json.loads(d.get("signal_tags_json") or "[]")
        return d

    # ------------------------------------------------------------------
    # Structured tables
    # ------------------------------------------------------------------

    def add_build(
        self,
        *,
        build_name: str,
        raw_item_id: int | None = None,
        architecture: str = "corexy",
        printer_model: str = "",
        frame_profile: str = "",
        frame_size_mm: float | None = None,
        panel_spec: str = "",
        drive_mode: str = "",
        belt_width_mm: float | None = None,
        belt_core: str = "",
        belt_pluck_hz: float | None = None,
        motors: str = "",
        voltage_v: float | None = None,
        toolhead: str = "",
        hotend: str = "",
        x_shaper_type: str = "",
        x_shaper_hz: float | None = None,
        x_max_accel: float | None = None,
        y_shaper_type: str = "",
        y_shaper_hz: float | None = None,
        y_max_accel: float | None = None,
        x_moving_mass_g: float | None = None,
        y_moving_mass_g: float | None = None,
        preset_hint: str = "",
        notes: str = "",
        dedup: bool = True,
    ) -> int:
        if dedup:
            existing = self.conn.execute(
                """
                SELECT id FROM builds
                WHERE build_name = ?
                  AND COALESCE(raw_item_id, -1) = COALESCE(?, -1)
                  AND COALESCE(x_shaper_hz, -1) = COALESCE(?, -1)
                  AND COALESCE(y_shaper_hz, -1) = COALESCE(?, -1)
                """,
                (build_name, raw_item_id, x_shaper_hz, y_shaper_hz),
            ).fetchone()
            if existing is not None:
                return int(existing["id"])

        cur = self.conn.execute(
            """
            INSERT INTO builds (
                raw_item_id, build_name, architecture, printer_model, frame_profile,
                frame_size_mm, panel_spec, drive_mode, belt_width_mm, belt_core,
                belt_pluck_hz, motors, voltage_v, toolhead, hotend,
                x_shaper_type, x_shaper_hz, x_max_accel,
                y_shaper_type, y_shaper_hz, y_max_accel,
                x_moving_mass_g, y_moving_mass_g, preset_hint, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                raw_item_id,
                build_name,
                architecture,
                printer_model,
                frame_profile,
                frame_size_mm,
                panel_spec,
                drive_mode,
                belt_width_mm,
                belt_core,
                belt_pluck_hz,
                motors,
                voltage_v,
                toolhead,
                hotend,
                x_shaper_type,
                x_shaper_hz,
                x_max_accel,
                y_shaper_type,
                y_shaper_hz,
                y_max_accel,
                x_moving_mass_g,
                y_moving_mass_g,
                preset_hint,
                notes,
            ),
        )
        build_id = int(cur.lastrowid)
        body_parts = [
            printer_model,
            frame_profile,
            f"{frame_size_mm}mm" if frame_size_mm else "",
            panel_spec,
            drive_mode,
            f"{belt_width_mm}mm" if belt_width_mm else "",
            belt_core,
            motors,
            toolhead,
            hotend,
            f"X {x_shaper_type} {x_shaper_hz}Hz" if x_shaper_hz else "",
            f"Y {y_shaper_type} {y_shaper_hz}Hz" if y_shaper_hz else "",
            notes,
        ]
        tags = " ".join(
            t for t in [architecture, drive_mode, frame_profile, preset_hint, x_shaper_type, y_shaper_type] if t
        )
        self._sync_fts(
            "build",
            build_id,
            build_name,
            " ".join(p for p in body_parts if p),
            tags,
        )
        self.conn.commit()
        return build_id

    def add_component_weight(
        self,
        *,
        category: str,
        component_name: str,
        mass_g: float,
        raw_item_id: int | None = None,
        details: str = "",
        dedup: bool = True,
    ) -> int:
        if dedup:
            existing = self.conn.execute(
                """
                SELECT id FROM component_weights
                WHERE LOWER(component_name) = LOWER(?)
                  AND ABS(mass_g - ?) < 0.05
                  AND COALESCE(raw_item_id, -1) = COALESCE(?, -1)
                """,
                (component_name, float(mass_g), raw_item_id),
            ).fetchone()
            if existing is not None:
                return int(existing["id"])

        cur = self.conn.execute(
            """
            INSERT INTO component_weights (raw_item_id, category, component_name, mass_g, details)
            VALUES (?, ?, ?, ?, ?)
            """,
            (raw_item_id, category, component_name, float(mass_g), details),
        )
        wid = int(cur.lastrowid)
        self._sync_fts(
            "component_weight",
            wid,
            component_name,
            f"{component_name} {mass_g}g {details}",
            f"weight {category}",
        )
        self.conn.commit()
        return wid

    def add_material_test(
        self,
        *,
        material_name: str,
        raw_item_id: int | None = None,
        modulus_gpa: float | None = None,
        hdt_c: float | None = None,
        creep_limit_n: float | None = None,
        belt_pluck_hz: float | None = None,
        shaper_hz: float | None = None,
        test_notes: str = "",
        dedup: bool = True,
    ) -> int:
        if dedup:
            existing = self.conn.execute(
                """
                SELECT id FROM material_tests
                WHERE LOWER(material_name) = LOWER(?)
                  AND COALESCE(raw_item_id, -1) = COALESCE(?, -1)
                  AND COALESCE(shaper_hz, -1) = COALESCE(?, -1)
                """,
                (material_name, raw_item_id, shaper_hz),
            ).fetchone()
            if existing is not None:
                return int(existing["id"])

        cur = self.conn.execute(
            """
            INSERT INTO material_tests (
                raw_item_id, material_name, modulus_gpa, hdt_c, creep_limit_n,
                belt_pluck_hz, shaper_hz, test_notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                raw_item_id,
                material_name,
                modulus_gpa,
                hdt_c,
                creep_limit_n,
                belt_pluck_hz,
                shaper_hz,
                test_notes,
            ),
        )
        mid = int(cur.lastrowid)
        body = " ".join(
            p
            for p in [
                material_name,
                f"{modulus_gpa} GPa" if modulus_gpa is not None else "",
                f"HDT {hdt_c} C" if hdt_c is not None else "",
                f"pluck {belt_pluck_hz} Hz" if belt_pluck_hz is not None else "",
                f"shaper {shaper_hz} Hz" if shaper_hz is not None else "",
                test_notes,
            ]
            if p
        )
        self._sync_fts("material_test", mid, material_name, body, "material test")
        self.conn.commit()
        return mid

    def add_troubleshooting_case(
        self,
        *,
        symptom: str,
        raw_item_id: int | None = None,
        printer_class: str = "",
        measured_signature: str = "",
        root_cause: str = "",
        verified_fix: str = "",
        dedup: bool = True,
    ) -> int:
        if dedup:
            existing = self.conn.execute(
                """
                SELECT id FROM troubleshooting_cases
                WHERE LOWER(symptom) = LOWER(?)
                  AND COALESCE(raw_item_id, -1) = COALESCE(?, -1)
                  AND LOWER(root_cause) = LOWER(?)
                """,
                (symptom, raw_item_id, root_cause),
            ).fetchone()
            if existing is not None:
                return int(existing["id"])

        cur = self.conn.execute(
            """
            INSERT INTO troubleshooting_cases (
                raw_item_id, printer_class, symptom, measured_signature, root_cause, verified_fix
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (raw_item_id, printer_class, symptom, measured_signature, root_cause, verified_fix),
        )
        tid = int(cur.lastrowid)
        body = " ".join(p for p in [printer_class, symptom, measured_signature, root_cause, verified_fix] if p)
        self._sync_fts(
            "troubleshooting_case",
            tid,
            f"{symptom} ({printer_class})" if printer_class else symptom,
            body,
            f"troubleshooting {printer_class}".strip(),
        )
        self.conn.commit()
        return tid

    def insert_extracted_records(
        self,
        extracted: dict[str, list[dict[str, Any]]],
        raw_item_id: int | None = None,
    ) -> dict[str, int]:
        """Insert candidate records returned by ``extract_structured_records``."""
        counts = {
            "builds_added": 0,
            "weights_added": 0,
            "material_tests_added": 0,
            "troubleshooting_added": 0,
        }
        for b in extracted.get("builds", []):
            if not b.get("build_name"):
                continue
            before = self.conn.execute("SELECT COUNT(*) FROM builds").fetchone()[0]
            self.add_build(
                raw_item_id=raw_item_id,
                build_name=str(b["build_name"]),
                architecture=str(b.get("architecture") or "corexy"),
                printer_model=str(b.get("printer_model") or ""),
                frame_profile=str(b.get("frame_profile") or ""),
                frame_size_mm=b.get("frame_size_mm"),
                panel_spec=str(b.get("panel_spec") or ""),
                drive_mode=str(b.get("drive_mode") or ""),
                belt_width_mm=b.get("belt_width_mm"),
                belt_core=str(b.get("belt_core") or ""),
                belt_pluck_hz=b.get("belt_pluck_hz"),
                motors=str(b.get("motors") or ""),
                voltage_v=b.get("voltage_v"),
                toolhead=str(b.get("toolhead") or ""),
                hotend=str(b.get("hotend") or ""),
                x_shaper_type=str(b.get("x_shaper_type") or ""),
                x_shaper_hz=b.get("x_shaper_hz"),
                x_max_accel=b.get("x_max_accel"),
                y_shaper_type=str(b.get("y_shaper_type") or ""),
                y_shaper_hz=b.get("y_shaper_hz"),
                y_max_accel=b.get("y_max_accel"),
                x_moving_mass_g=b.get("x_moving_mass_g"),
                y_moving_mass_g=b.get("y_moving_mass_g"),
                preset_hint=str(b.get("preset_hint") or ""),
                notes=str(b.get("notes") or ""),
            )
            after = self.conn.execute("SELECT COUNT(*) FROM builds").fetchone()[0]
            if after > before:
                counts["builds_added"] += 1

        for w in extracted.get("component_weights", []):
            if not w.get("component_name") or w.get("mass_g") is None:
                continue
            before = self.conn.execute("SELECT COUNT(*) FROM component_weights").fetchone()[0]
            self.add_component_weight(
                raw_item_id=raw_item_id,
                category=str(w.get("category") or "toolhead"),
                component_name=str(w["component_name"]),
                mass_g=float(w["mass_g"]),
                details=str(w.get("details") or ""),
            )
            after = self.conn.execute("SELECT COUNT(*) FROM component_weights").fetchone()[0]
            if after > before:
                counts["weights_added"] += 1

        for mt in extracted.get("material_tests", []):
            if not mt.get("material_name"):
                continue
            before = self.conn.execute("SELECT COUNT(*) FROM material_tests").fetchone()[0]
            self.add_material_test(
                raw_item_id=raw_item_id,
                material_name=str(mt["material_name"]),
                modulus_gpa=mt.get("modulus_gpa"),
                hdt_c=mt.get("hdt_c"),
                creep_limit_n=mt.get("creep_limit_n"),
                belt_pluck_hz=mt.get("belt_pluck_hz"),
                shaper_hz=mt.get("shaper_hz"),
                test_notes=str(mt.get("test_notes") or ""),
            )
            after = self.conn.execute("SELECT COUNT(*) FROM material_tests").fetchone()[0]
            if after > before:
                counts["material_tests_added"] += 1

        for tc in extracted.get("troubleshooting_cases", []):
            if not tc.get("symptom"):
                continue
            before = self.conn.execute("SELECT COUNT(*) FROM troubleshooting_cases").fetchone()[0]
            self.add_troubleshooting_case(
                raw_item_id=raw_item_id,
                printer_class=str(tc.get("printer_class") or ""),
                symptom=str(tc["symptom"]),
                measured_signature=str(tc.get("measured_signature") or ""),
                root_cause=str(tc.get("root_cause") or ""),
                verified_fix=str(tc.get("verified_fix") or ""),
            )
            after = self.conn.execute("SELECT COUNT(*) FROM troubleshooting_cases").fetchone()[0]
            if after > before:
                counts["troubleshooting_added"] += 1

        return counts

    # ------------------------------------------------------------------
    # Query & Search
    # ------------------------------------------------------------------

    @staticmethod
    def _build_safe_fts_query(query: str) -> str:
        tokens = re.findall(r"[A-Za-z0-9_]+", query or "")
        if not tokens:
            return ""
        # Quote each token safely so FTS5 operators (AND/OR/NOT/-) never throw syntax errors
        return " AND ".join(f'"{tok}"*' for tok in tokens)

    def _fetch_backing_record(self, item_type: str, ref_id: int) -> dict[str, Any] | None:
        table_name = _ITEM_TYPE_TO_TABLE.get(item_type)
        if not table_name:
            return None
        row = self.conn.execute(f"SELECT * FROM {table_name} WHERE id = ?", (ref_id,)).fetchone()
        return dict(row) if row is not None else None

    def search(
        self,
        query: str,
        table: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Instant FTS5 search across all KB tables with safe query escaping & LIKE fallback."""
        q_clean = (query or "").strip()
        if not q_clean:
            return []

        item_type_filter: str | None = None
        if table:
            item_type_filter = _TABLE_TO_ITEM_TYPE.get(table.strip().lower(), table.strip().lower())

        fts_expr = self._build_safe_fts_query(q_clean)
        rows: list[sqlite3.Row] = []

        if fts_expr:
            try:
                if item_type_filter:
                    sql = """
                        SELECT item_type, ref_id, title, body, tags, bm25(kb_fts) AS rank
                        FROM kb_fts
                        WHERE kb_fts MATCH ? AND item_type = ?
                        ORDER BY rank ASC
                        LIMIT ?
                    """
                    rows = self.conn.execute(sql, (fts_expr, item_type_filter, limit)).fetchall()
                else:
                    sql = """
                        SELECT item_type, ref_id, title, body, tags, bm25(kb_fts) AS rank
                        FROM kb_fts
                        WHERE kb_fts MATCH ?
                        ORDER BY rank ASC
                        LIMIT ?
                    """
                    rows = self.conn.execute(sql, (fts_expr, limit)).fetchall()
            except sqlite3.OperationalError:
                rows = []

        # Fallback LIKE search if FTS5 returned nothing (e.g. punctuation/decimal/substring)
        if not rows:
            like_pat = f"%{q_clean}%"
            if item_type_filter:
                sql = """
                    SELECT item_type, ref_id, title, body, tags, 0.0 AS rank
                    FROM kb_fts
                    WHERE item_type = ? AND (title LIKE ? OR body LIKE ? OR tags LIKE ?)
                    LIMIT ?
                """
                rows = self.conn.execute(
                    sql, (item_type_filter, like_pat, like_pat, like_pat, limit)
                ).fetchall()
            else:
                sql = """
                    SELECT item_type, ref_id, title, body, tags, 0.0 AS rank
                    FROM kb_fts
                    WHERE title LIKE ? OR body LIKE ? OR tags LIKE ?
                    LIMIT ?
                """
                rows = self.conn.execute(sql, (like_pat, like_pat, like_pat, limit)).fetchall()

        results: list[dict[str, Any]] = []
        for r in rows:
            ref_id = int(r["ref_id"])
            item_type = str(r["item_type"])
            body = str(r["body"] or "")
            results.append(
                {
                    "item_type": item_type,
                    "ref_id": ref_id,
                    "title": str(r["title"] or ""),
                    "snippet": body[:240],
                    "tags": str(r["tags"] or ""),
                    "rank": float(r["rank"]),
                    "record": self._fetch_backing_record(item_type, ref_id),
                }
            )
        return results

    def list_builds(
        self,
        architecture: str | None = None,
        drive_mode: str | None = None,
        min_y_hz: float | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if architecture:
            clauses.append("LOWER(architecture) = LOWER(?)")
            params.append(architecture)
        if drive_mode:
            clauses.append("LOWER(drive_mode) = LOWER(?)")
            params.append(drive_mode)
        if min_y_hz is not None:
            clauses.append("y_shaper_hz IS NOT NULL AND y_shaper_hz >= ?")
            params.append(float(min_y_hz))

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.conn.execute(
            f"SELECT * FROM builds {where_sql} ORDER BY id ASC",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict[str, Any]:
        def _count(table: str) -> int:
            return int(self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

        status_rows = self.conn.execute(
            "SELECT review_status, COUNT(*) AS cnt FROM raw_items GROUP BY review_status"
        ).fetchall()
        source_rows = self.conn.execute(
            "SELECT source_type, COUNT(*) AS cnt FROM raw_items GROUP BY source_type"
        ).fetchall()

        return {
            "db_path": str(self.db_path),
            "raw_items": _count("raw_items"),
            "raw_by_status": {str(r["review_status"]): int(r["cnt"]) for r in status_rows},
            "raw_by_source": {str(r["source_type"]): int(r["cnt"]) for r in source_rows},
            "builds": _count("builds"),
            "component_weights": _count("component_weights"),
            "material_tests": _count("material_tests"),
            "troubleshooting_cases": _count("troubleshooting_cases"),
            "fts_entries": _count("kb_fts"),
        }

    # ------------------------------------------------------------------
    # Review Queue Export & Approval Workflow
    # ------------------------------------------------------------------

    def export_review_queue(
        self,
        out_dir: Path,
        min_score: float = 0.25,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Export pending high-signal items to ``review_queue.json`` and ``review_queue.md``."""
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        rows = self.conn.execute(
            """
            SELECT * FROM raw_items
            WHERE review_status = 'pending' AND signal_score >= ?
            ORDER BY signal_score DESC, id DESC
            LIMIT ?
            """,
            (float(min_score), int(limit)),
        ).fetchall()

        items_payload: list[dict[str, Any]] = []
        for r in rows:
            attachments = json.loads(r["attachments_json"] or "[]")
            tags = json.loads(r["signal_tags_json"] or "[]")
            extracted = extract_structured_records(
                r["raw_text"],
                title=r["title"],
                meta={
                    "author": r["author"],
                    "channel": r["channel"],
                    "source_name": r["source_name"],
                    "url": r["url"],
                },
            )
            items_payload.append(
                {
                    "raw_item_id": int(r["id"]),
                    "decision": "pending",
                    "signal_score": round(float(r["signal_score"]), 3),
                    "signal_tags": tags,
                    "source_type": r["source_type"],
                    "source_name": r["source_name"],
                    "channel": r["channel"],
                    "author": r["author"],
                    "title": r["title"],
                    "url": r["url"],
                    "published_at": r["published_at"],
                    "attachments": attachments,
                    "raw_text": r["raw_text"],
                    "extracted": extracted,
                }
            )

        json_file = out_path / "review_queue.json"
        md_file = out_path / "review_queue.md"

        payload = {
            "generated_at": _utc_now_iso(),
            "db_path": str(self.db_path),
            "min_score": float(min_score),
            "count": len(items_payload),
            "items": items_payload,
        }
        json_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        md_lines: list[str] = [
            "# PrintSim Knowledge Base — Review Queue",
            "",
            f"- **Generated:** `{payload['generated_at']}`",
            f"- **Database:** `{self.db_path}`",
            f"- **Pending items (score >= {min_score:.2f}):** **{len(items_payload)}**",
            "",
            "> Edit `\"decision\"` in `review_queue.json` to `\"approve\"` or `\"reject\"` for each item, then run:",
            f"> `printsim kb approve {json_file}`",
            "",
        ]
        if not items_payload:
            md_lines.append("_No pending items matched the minimum signal score._")
        else:
            for idx, it in enumerate(items_payload, start=1):
                heading = it["title"] or f"Item #{it['raw_item_id']}"
                tag_str = ", ".join(it["signal_tags"]) or "none"
                md_lines.extend(
                    [
                        f"## {idx}. [ID {it['raw_item_id']}] {heading}",
                        f"- **Score:** `{it['signal_score']:.2f}` | **Tags:** `{tag_str}` | "
                        f"**Source:** `{it['source_type']}` ({it['source_name'] or '-'} / {it['channel'] or '-'})",
                        f"- **Author:** `{it['author'] or '-'}` | **Published:** `{it['published_at'] or '-'}`",
                    ]
                )
                if it["url"]:
                    md_lines.append(f"- **URL:** {it['url']}")
                if it["attachments"]:
                    att_names = [
                        str(a.get("filename") or a.get("fileName") or a.get("url") or "attachment")
                        for a in it["attachments"]
                    ]
                    md_lines.append(f"- **Attachments:** {', '.join(att_names)}")

                excerpt = (it["raw_text"] or "").strip()
                if len(excerpt) > 600:
                    excerpt = excerpt[:597] + "..."
                md_lines.extend(["", "```text", excerpt, "```"])

                ext = it["extracted"]
                cand_summary: list[str] = []
                for b in ext.get("builds", []):
                    cand_summary.append(
                        f"Build `{b['build_name']}` (X={b.get('x_shaper_hz')} Hz, Y={b.get('y_shaper_hz')} Hz, "
                        f"drive={b.get('drive_mode') or '-'}, preset={b.get('preset_hint') or '-'})"
                    )
                for w in ext.get("component_weights", []):
                    cand_summary.append(
                        f"Weight `{w['component_name']}` = **{w['mass_g']} g** ({w['category']})"
                    )
                for mt in ext.get("material_tests", []):
                    cand_summary.append(
                        f"Material `{mt['material_name']}` (shaper={mt.get('shaper_hz')} Hz, "
                        f"GPa={mt.get('modulus_gpa')}, HDT={mt.get('hdt_c')} C)"
                    )
                for tc in ext.get("troubleshooting_cases", []):
                    cand_summary.append(
                        f"Troubleshooting `{tc['symptom']}` -> {tc.get('verified_fix', '')[:100]}"
                    )
                if cand_summary:
                    md_lines.append("")
                    md_lines.append("**Extracted Candidates:**")
                    for line in cand_summary:
                        md_lines.append(f"- {line}")
                md_lines.append("")

        md_file.write_text("\n".join(md_lines), encoding="utf-8")

        return {
            "count": len(items_payload),
            "json_path": str(json_file),
            "markdown_path": str(md_file),
        }

    def apply_review_decisions(self, review_json_path: Path) -> dict[str, int]:
        """Apply human approve/reject decisions from an edited ``review_queue.json``."""
        data = json.loads(Path(review_json_path).read_text(encoding="utf-8"))
        items = data.get("items", []) if isinstance(data, dict) else data

        summary = {
            "approved": 0,
            "rejected": 0,
            "pending": 0,
            "builds_added": 0,
            "weights_added": 0,
            "material_tests_added": 0,
            "troubleshooting_added": 0,
        }

        for entry in items:
            raw_item_id = int(entry["raw_item_id"])
            decision = str(entry.get("decision", "pending")).strip().lower()
            if decision in ("approve", "approved"):
                self.conn.execute(
                    "UPDATE raw_items SET review_status = 'approved' WHERE id = ?",
                    (raw_item_id,),
                )
                self.conn.commit()
                summary["approved"] += 1

                extracted = entry.get("extracted")
                if not isinstance(extracted, dict):
                    raw_row = self.get_raw_item(raw_item_id)
                    if raw_row:
                        extracted = extract_structured_records(
                            raw_row["raw_text"],
                            title=raw_row["title"],
                            meta={"author": raw_row["author"], "channel": raw_row["channel"]},
                        )
                    else:
                        extracted = {}

                ins = self.insert_extracted_records(extracted, raw_item_id=raw_item_id)
                for k, v in ins.items():
                    summary[k] += v
            elif decision in ("reject", "rejected"):
                self.conn.execute(
                    "UPDATE raw_items SET review_status = 'rejected' WHERE id = ?",
                    (raw_item_id,),
                )
                self.conn.commit()
                summary["rejected"] += 1
            else:
                summary["pending"] += 1

        return summary
