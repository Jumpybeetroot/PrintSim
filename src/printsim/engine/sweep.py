"""Config-driven parallel batch sweep engine with SQLite caching."""

from __future__ import annotations

import ast
from concurrent.futures import ProcessPoolExecutor
import copy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import time
from typing import Any

import yaml

from printsim.engine.diagnostics import (
    PrinterDiagnosis,
    apply_dot_overrides,
    extract_diagnosis_from_report,
    parse_overrides_input,
    _run_evaluate_printer,
)
from printsim.mcp_server import EXAMPLES_DIR, PROJECT_ROOT
from printsim.schema import PrinterSpec

CACHE_SCHEMA_VERSION = "1.0"
DEFAULT_CACHE_DB = PROJECT_ROOT / ".cache" / "sweeps" / "sweep_cache.sqlite"
DEFAULT_SWEEP_OUT_ROOT = PROJECT_ROOT / "sweeps" / "out"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip()).strip("_").lower()
    return slug or "sweep"


def _resolve_base_spec_path(raw_path: str | Path, config_dir: Path) -> Path:
    p = Path(raw_path)
    candidates = [
        config_dir / p,
        PROJECT_ROOT / p,
        p,
        EXAMPLES_DIR / f"{raw_path}.yaml",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(
        f"Base printer spec YAML not found: {raw_path!r} "
        f"(checked {config_dir}, {PROJECT_ROOT}, and {EXAMPLES_DIR})"
    )


def _resolve_out_dir(
    config_data: dict[str, Any],
    sweep_name: str,
    config_dir: Path,
    out_dir_arg: Path | str | None,
) -> Path:
    if out_dir_arg is not None:
        return Path(out_dir_arg)
    raw_yaml_out = config_data.get("out_dir")
    if raw_yaml_out:
        p = Path(str(raw_yaml_out))
        return p if p.is_absolute() else (config_dir / p)
    return DEFAULT_SWEEP_OUT_ROOT / _slugify(sweep_name)


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _printer_cache_key(spec_dict: dict[str, Any], include_gcode: bool) -> str:
    blob = _canonical_json(
        {
            "spec": spec_dict,
            "include_gcode": bool(include_gcode),
            "schema_version": CACHE_SCHEMA_VERSION,
        }
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _xbeam_cache_key(script_sha256: str, only_args: list[str]) -> str:
    blob = _canonical_json(
        {
            "script_sha256": script_sha256,
            "only_args": list(only_args),
            "schema_version": CACHE_SCHEMA_VERSION,
        }
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _init_cache_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sweep_cache (
            cache_key TEXT PRIMARY KEY,
            mode TEXT NOT NULL,
            label TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def _cache_get(conn: sqlite3.Connection, cache_key: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT payload_json FROM sweep_cache WHERE cache_key = ?", (cache_key,)
    ).fetchone()
    if row is None:
        return None
    return json.loads(row[0])


def _cache_put(
    conn: sqlite3.Connection, cache_key: str, mode: str, label: str, payload: dict[str, Any]
) -> None:
    now_iso = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT OR REPLACE INTO sweep_cache (cache_key, mode, label, payload_json, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (cache_key, mode, label, json.dumps(payload, ensure_ascii=False), now_iso),
    )


def _printer_metrics_from_diagnosis(diag: PrinterDiagnosis) -> dict[str, Any]:
    return {
        "overall_score": diag.overall_score,
        "fundamental_freq_hz": diag.fundamental_freq_hz,
        "shaper_x_hz": diag.shaper_x["freq_hz"],
        "shaper_y_hz": diag.shaper_y["freq_hz"],
        "shaper_x_type": diag.shaper_x["type"],
        "shaper_y_type": diag.shaper_y["type"],
        "rec_max_accel_mm_s2": diag.recommended_max_accel_mm_s2,
        "static_stiffness_x_n_per_mm": diag.static_stiffness_n_per_mm["x"],
        "static_stiffness_y_n_per_mm": diag.static_stiffness_n_per_mm["y"],
        "worst_deflection_um": diag.static_tip_deflection_um["worst"],
        "moving_mass_x_g": diag.moving_mass_x_g,
        "moving_mass_y_g": diag.moving_mass_y_g,
        "total_mass_kg": diag.total_mass_kg,
        "max_volumetric_flow_mm3_s": diag.max_volumetric_flow_mm3_s,
        "bom_cost_usd": diag.bom_cost_usd,
        "primary_bottleneck_x": diag.shaper_x["limiting_cause"],
        "primary_bottleneck_y": diag.shaper_y["limiting_cause"],
    }


def _eval_printer_case_worker(task: tuple[int, str, dict[str, Any], dict[str, Any], bool]) -> tuple[int, dict[str, Any]]:
    """Top-level pickleable worker for ProcessPoolExecutor printer sweeps."""
    idx, label, spec_dict, overrides, include_gcode = task
    t0 = time.perf_counter()
    try:
        spec = PrinterSpec.model_validate(spec_dict)
        report = _run_evaluate_printer(spec, include_gcode=include_gcode)
        diag = extract_diagnosis_from_report(report, include_gcode=include_gcode)
        elapsed_s = round(time.perf_counter() - t0, 4)
        payload = {
            "status": "ok",
            "error": None,
            "metrics": _printer_metrics_from_diagnosis(diag),
            "diagnosis": diag.to_dict(),
            "elapsed_s": elapsed_s,
            "overrides": overrides,
            "label": label,
        }
    except Exception as exc:  # pragma: no cover - error boundary for batch safety
        payload = {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "metrics": {},
            "diagnosis": None,
            "elapsed_s": round(time.perf_counter() - t0, 4),
            "overrides": overrides,
            "label": label,
        }
    return idx, payload


@dataclass
class SweepCaseResult:
    """Single evaluated or cached case inside a ``SweepReport``."""

    label: str
    overrides: dict[str, Any] = field(default_factory=dict)
    cached: bool = False
    elapsed_s: float = 0.0
    status: str = "ok"
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    diagnosis: dict[str, Any] | None = None
    cache_key: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SweepReport:
    """Aggregate report for a completed parametric or 3D calibration sweep."""

    name: str
    mode: str
    config_path: str
    base_spec: str | None
    include_gcode: bool
    total_cases: int
    cache_hits: int
    cache_misses: int
    elapsed_s: float
    out_dir: str
    cases: list[SweepCaseResult] = field(default_factory=list)
    results_json_path: str = ""
    report_md_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode,
            "config_path": self.config_path,
            "base_spec": self.base_spec,
            "include_gcode": self.include_gcode,
            "total_cases": self.total_cases,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "elapsed_s": round(self.elapsed_s, 3),
            "out_dir": self.out_dir,
            "results_json_path": self.results_json_path,
            "report_md_path": self.report_md_path,
            "cases": [c.to_dict() for c in self.cases],
        }

    def to_markdown(self) -> str:
        lines: list[str] = [
            f"# Sweep Report: {self.name}",
            (
                f"Mode: `{self.mode}` | Cases: **{self.total_cases}** "
                f"(cache hits: {self.cache_hits}, misses: {self.cache_misses}) | "
                f"Elapsed: {self.elapsed_s:.2f}s"
            ),
        ]
        if self.base_spec:
            lines.append(f"Base spec: `{self.base_spec}` | include_gcode: `{self.include_gcode}`")
        lines.append("")

        if self.mode == "printer":
            lines.extend(
                [
                    "| Case | Score | f0 (Hz) | Shaper X/Y (Hz) | Rec Accel (mm/s²) | k_X / k_Y (N/mm) | X/Y Mass (g) | Defl (µm) | Cost ($) | Cache |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
                ]
            )
            ok_cases = [c for c in self.cases if c.status == "ok"]
            for c in self.cases:
                if c.status != "ok":
                    lines.append(
                        f"| {c.label} | ERR | — | — | — | — | — | — | — | {'hit' if c.cached else 'miss'} |"
                    )
                    continue
                m = c.metrics
                lines.append(
                    f"| {c.label} | {m['overall_score']:.1f} | {m['fundamental_freq_hz']:.2f} | "
                    f"{m['shaper_x_hz']:.1f} / {m['shaper_y_hz']:.1f} | {m['rec_max_accel_mm_s2']:.0f} | "
                    f"{m['static_stiffness_x_n_per_mm']:.1f} / {m['static_stiffness_y_n_per_mm']:.1f} | "
                    f"{m['moving_mass_x_g']:.0f} / {m['moving_mass_y_g']:.0f} | "
                    f"{m['worst_deflection_um']:.1f} | ${m['bom_cost_usd']:.0f} | "
                    f"{'hit' if c.cached else 'miss'} |"
                )
            if ok_cases:
                best_score = max(ok_cases, key=lambda c: float(c.metrics["overall_score"]))
                best_accel = max(ok_cases, key=lambda c: float(c.metrics["rec_max_accel_mm_s2"]))
                lines.extend(
                    [
                        "",
                        f"- Best overall score: **{best_score.label}** ({best_score.metrics['overall_score']:.1f}/100)",
                        f"- Highest shaper accel: **{best_accel.label}** ({best_accel.metrics['rec_max_accel_mm_s2']:.0f} mm/s²)",
                    ]
                )
        else:
            lines.extend(
                [
                    "| Case | Status | Records | Elapsed (s) | Cache |",
                    "|---|---|---:|---:|:---:|",
                ]
            )
            for c in self.cases:
                rec_count = int(c.metrics.get("record_count", 0))
                lines.append(
                    f"| {c.label} | {c.status} | {rec_count} | {c.elapsed_s:.2f} | "
                    f"{'hit' if c.cached else 'miss'} |"
                )
            for c in self.cases:
                records = c.metrics.get("records") or []
                if records:
                    lines.extend(["", f"## {c.label} Records"])
                    for rec in records[:24]:
                        lines.append(f"- `{rec}`")
                if c.error:
                    lines.extend(["", f"## {c.label} Error", f"```\n{c.error}\n```"])

        return "\n".join(lines)


def _expand_printer_cases(config_data: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    expanded: list[tuple[str, dict[str, Any]]] = []

    raw_cases = config_data.get("cases") or []
    for idx, entry in enumerate(raw_cases):
        if not isinstance(entry, dict):
            raise ValueError(f"Sweep case #{idx + 1} must be a mapping with 'label'/'overrides'")
        overrides = parse_overrides_input(entry.get("overrides"))
        label = str(entry.get("label") or f"case_{idx + 1}").strip()
        expanded.append((label, overrides))

    raw_grid = config_data.get("grid")
    if raw_grid:
        if not isinstance(raw_grid, dict):
            raise ValueError("'grid' must be a mapping of dot.path -> list of values")
        keys: list[str] = []
        value_lists: list[list[Any]] = []
        for k, vals in raw_grid.items():
            if not isinstance(vals, list) or not vals:
                raise ValueError(f"Grid axis '{k}' must be a non-empty list of values")
            keys.append(str(k).strip())
            value_lists.append(vals)

        for combo in itertools.product(*value_lists):
            ov_dict = {k: v for k, v in zip(keys, combo)}
            label_parts = [f"{k.split('.')[-1]}={v}" for k, v in ov_dict.items()]
            expanded.append((", ".join(label_parts), parse_overrides_input(ov_dict)))

    if not expanded:
        expanded.append(("baseline", {}))
    return expanded


def _run_printer_sweep(
    *,
    sweep_name: str,
    config_path: Path,
    config_data: dict[str, Any],
    workers: int | None,
    use_cache: bool,
    cache_db: Path,
    out_dir: Path,
) -> SweepReport:
    t_start = time.perf_counter()
    raw_base = config_data.get("base_spec")
    if not raw_base:
        raise ValueError("Printer sweep config requires 'base_spec'")
    base_spec_path = _resolve_base_spec_path(str(raw_base), config_path.parent)
    base_spec = PrinterSpec.from_yaml(base_spec_path)
    include_gcode = bool(config_data.get("include_gcode", False))

    raw_cases = _expand_printer_cases(config_data)

    # Pre-resolve validated specs & cache keys in main process
    prepared: list[tuple[int, str, dict[str, Any], dict[str, Any], str]] = []
    for idx, (label, overrides) in enumerate(raw_cases):
        spec = apply_dot_overrides(base_spec, overrides)
        spec_dict = spec.model_dump(mode="json")
        ckey = _printer_cache_key(spec_dict, include_gcode)
        prepared.append((idx, label, spec_dict, overrides, ckey))

    results_by_idx: dict[int, SweepCaseResult] = {}
    uncached_tasks: list[tuple[int, str, dict[str, Any], dict[str, Any], bool]] = []
    cache_hits = 0

    conn: sqlite3.Connection | None = _init_cache_db(cache_db) if use_cache else None
    try:
        for idx, label, spec_dict, overrides, ckey in prepared:
            cached_payload = _cache_get(conn, ckey) if conn is not None else None
            if cached_payload is not None:
                cache_hits += 1
                results_by_idx[idx] = SweepCaseResult(
                    label=label,
                    overrides=overrides,
                    cached=True,
                    elapsed_s=float(cached_payload.get("elapsed_s", 0.0)),
                    status=str(cached_payload.get("status", "ok")),
                    error=cached_payload.get("error"),
                    metrics=dict(cached_payload.get("metrics") or {}),
                    diagnosis=cached_payload.get("diagnosis"),
                    cache_key=ckey,
                )
            else:
                uncached_tasks.append((idx, label, spec_dict, overrides, include_gcode))

        key_by_idx = {idx: ckey for idx, _, _, _, ckey in prepared}
        if uncached_tasks:
            use_parallel = workers != 1 and len(raw_cases) > 2 and len(uncached_tasks) > 1
            if use_parallel:
                max_workers = (
                    max(1, int(workers))
                    if workers is not None
                    else min(len(uncached_tasks), os.cpu_count() or 1, 8)
                )
                with ProcessPoolExecutor(max_workers=max_workers) as pool:
                    worker_outputs = list(pool.map(_eval_printer_case_worker, uncached_tasks))
            else:
                worker_outputs = [_eval_printer_case_worker(t) for t in uncached_tasks]

            for idx, payload in worker_outputs:
                ckey = key_by_idx[idx]
                label = str(payload["label"])
                overrides = dict(payload["overrides"])
                if conn is not None and payload.get("status") == "ok":
                    _cache_put(conn, ckey, "printer", label, payload)
                results_by_idx[idx] = SweepCaseResult(
                    label=label,
                    overrides=overrides,
                    cached=False,
                    elapsed_s=float(payload.get("elapsed_s", 0.0)),
                    status=str(payload.get("status", "ok")),
                    error=payload.get("error"),
                    metrics=dict(payload.get("metrics") or {}),
                    diagnosis=payload.get("diagnosis"),
                    cache_key=ckey,
                )
            if conn is not None:
                conn.commit()
    finally:
        if conn is not None:
            conn.close()

    try:
        base_spec_disp = str(base_spec_path.relative_to(PROJECT_ROOT))
    except ValueError:
        base_spec_disp = str(base_spec_path)

    ordered_cases = [results_by_idx[i] for i in range(len(prepared))]
    elapsed_s = time.perf_counter() - t_start
    return SweepReport(
        name=sweep_name,
        mode="printer",
        config_path=str(config_path),
        base_spec=base_spec_disp,
        include_gcode=include_gcode,
        total_cases=len(ordered_cases),
        cache_hits=cache_hits,
        cache_misses=len(ordered_cases) - cache_hits,
        elapsed_s=elapsed_s,
        out_dir=str(out_dir),
        cases=ordered_cases,
    )


def _parse_xbeam_stdout(stdout_text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw_line in stdout_text.splitlines():
        line = raw_line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                parsed = ast.literal_eval(line)
                if isinstance(parsed, dict):
                    records.append(parsed)
            except (SyntaxError, ValueError):
                continue
    return records


def _run_xbeam_sweep(
    *,
    sweep_name: str,
    config_path: Path,
    config_data: dict[str, Any],
    use_cache: bool,
    cache_db: Path,
    out_dir: Path,
) -> SweepReport:
    t_start = time.perf_counter()
    script_path = PROJECT_ROOT / "scripts" / "calibrate_xbeam_3d_fea.py"
    if not script_path.is_file():
        raise FileNotFoundError(f"Calibration script not found: {script_path}")
    script_sha256 = hashlib.sha256(script_path.read_bytes()).hexdigest()

    raw_cases = config_data.get("cases")
    case_specs: list[tuple[str, list[str]]] = []
    if isinstance(raw_cases, list) and raw_cases:
        for idx, entry in enumerate(raw_cases):
            raw_only = entry.get("only") if isinstance(entry, dict) else entry
            if isinstance(raw_only, str):
                only_list = [raw_only]
            elif isinstance(raw_only, list):
                only_list = [str(x) for x in raw_only]
            else:
                only_list = ["bare"]
            lbl = (
                str(entry.get("label"))
                if isinstance(entry, dict) and entry.get("label")
                else ",".join(only_list)
            )
            case_specs.append((lbl or f"xbeam_{idx + 1}", only_list))
    else:
        raw_only = config_data.get("only", ["bare"])
        only_list = [raw_only] if isinstance(raw_only, str) else [str(x) for x in raw_only]
        case_specs.append((",".join(only_list), only_list))

    cases_out: list[SweepCaseResult] = []
    cache_hits = 0
    conn: sqlite3.Connection | None = _init_cache_db(cache_db) if use_cache else None
    try:
        for label, only_args in case_specs:
            ckey = _xbeam_cache_key(script_sha256, only_args)
            cached_payload = _cache_get(conn, ckey) if conn is not None else None
            if cached_payload is not None:
                cache_hits += 1
                cases_out.append(
                    SweepCaseResult(
                        label=label,
                        overrides={"only": only_args},
                        cached=True,
                        elapsed_s=float(cached_payload.get("elapsed_s", 0.0)),
                        status=str(cached_payload.get("status", "ok")),
                        error=cached_payload.get("error"),
                        metrics=dict(cached_payload.get("metrics") or {}),
                        cache_key=ckey,
                    )
                )
                continue

            c_t0 = time.perf_counter()
            proc = subprocess.run(
                [sys.executable, str(script_path), "--only", *only_args],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
            c_elapsed = round(time.perf_counter() - c_t0, 4)
            records = _parse_xbeam_stdout(proc.stdout)
            status = "ok" if proc.returncode == 0 else "error"
            err_msg = proc.stderr.strip() if proc.returncode != 0 else None
            metrics = {
                "returncode": proc.returncode,
                "script_sha256": script_sha256,
                "only_args": only_args,
                "record_count": len(records),
                "records": records,
                "stdout_tail": "\n".join(proc.stdout.splitlines()[-20:]),
            }
            payload = {
                "status": status,
                "error": err_msg,
                "metrics": metrics,
                "elapsed_s": c_elapsed,
            }
            if conn is not None and status == "ok":
                _cache_put(conn, ckey, "xbeam_3d_fea", label, payload)
                conn.commit()
            cases_out.append(
                SweepCaseResult(
                    label=label,
                    overrides={"only": only_args},
                    cached=False,
                    elapsed_s=c_elapsed,
                    status=status,
                    error=err_msg,
                    metrics=metrics,
                    cache_key=ckey,
                )
            )
    finally:
        if conn is not None:
            conn.close()

    return SweepReport(
        name=sweep_name,
        mode="xbeam_3d_fea",
        config_path=str(config_path),
        base_spec=None,
        include_gcode=False,
        total_cases=len(cases_out),
        cache_hits=cache_hits,
        cache_misses=len(cases_out) - cache_hits,
        elapsed_s=time.perf_counter() - t_start,
        out_dir=str(out_dir),
        cases=cases_out,
    )


def run_sweep(
    config_path: Path | str,
    *,
    workers: int | None = None,
    use_cache: bool = True,
    cache_db: Path | None = None,
    out_dir: Path | str | None = None,
) -> SweepReport:
    """Run a YAML-configured parametric printer or 3D calibration sweep with SQLite caching."""
    cfg_path = Path(config_path)
    if not cfg_path.is_file():
        alt = PROJECT_ROOT / cfg_path
        if alt.is_file():
            cfg_path = alt
        else:
            raise FileNotFoundError(f"Sweep config YAML not found: {config_path!r}")
    cfg_path = cfg_path.resolve()

    config_data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(config_data, dict):
        raise ValueError(f"Sweep config must be a YAML mapping: {cfg_path}")

    sweep_name = str(config_data.get("name") or cfg_path.stem).strip()
    mode = str(config_data.get("mode") or "printer").strip()
    if mode not in ("printer", "xbeam_3d_fea"):
        raise ValueError(f"Unsupported sweep mode {mode!r}; expected 'printer' or 'xbeam_3d_fea'")

    resolved_cache_db = Path(cache_db) if cache_db is not None else DEFAULT_CACHE_DB
    resolved_out_dir = _resolve_out_dir(config_data, sweep_name, cfg_path.parent, out_dir)
    resolved_out_dir.mkdir(parents=True, exist_ok=True)

    if mode == "printer":
        report = _run_printer_sweep(
            sweep_name=sweep_name,
            config_path=cfg_path,
            config_data=config_data,
            workers=workers,
            use_cache=use_cache,
            cache_db=resolved_cache_db,
            out_dir=resolved_out_dir,
        )
    else:
        report = _run_xbeam_sweep(
            sweep_name=sweep_name,
            config_path=cfg_path,
            config_data=config_data,
            use_cache=use_cache,
            cache_db=resolved_cache_db,
            out_dir=resolved_out_dir,
        )

    results_json_path = resolved_out_dir / "results.json"
    report_md_path = resolved_out_dir / "report.md"
    report.results_json_path = str(results_json_path)
    report.report_md_path = str(report_md_path)

    results_json_path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report_md_path.write_text(report.to_markdown() + "\n", encoding="utf-8")
    return report
