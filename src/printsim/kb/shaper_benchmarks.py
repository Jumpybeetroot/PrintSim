"""Curated real-world machine input-shaper benchmark runner and baseline diff engine.

Evaluates PrintSim against the curated manifest of real measured machines and
controlled same-machine A/B sweeps in ``data/touchstones/shaper_benchmarks.yaml``.
Each benchmark entry specifies an exact base preset plus validated ``PrinterSpec``
overrides so predictions are compared apples-to-apples against the real machine's
hardware, belt tension, drive mode, motor current, and Klipper shaper type.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import yaml

from printsim.engine import evaluate_printer
from printsim.schema import PrinterSpec


def default_manifest_path() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "touchstones" / "shaper_benchmarks.yaml"


def default_examples_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "examples"


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def load_shaper_benchmarks(manifest_path: Path | None = None) -> dict[str, Any]:
    """Load the curated real-machine shaper benchmark YAML manifest."""
    path = Path(manifest_path) if manifest_path is not None else default_manifest_path()
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict) or "machines" not in data:
        raise ValueError(f"Invalid shaper benchmark manifest: {path}")
    return data


def build_benchmark_spec(
    entry: dict[str, Any],
    examples_dir: Path | None = None,
) -> PrinterSpec:
    """Build and schema-validate the exact PrinterSpec for a benchmark machine entry."""
    ex_dir = Path(examples_dir) if examples_dir is not None else default_examples_dir()
    base_file = ex_dir / str(entry["base_preset"])
    if not base_file.exists():
        raise FileNotFoundError(
            f"Benchmark '{entry['id']}' references missing preset: {base_file}"
        )
    base_spec = PrinterSpec.from_yaml(base_file)
    overrides = entry.get("overrides") or {}
    if not overrides:
        return base_spec
    merged = _deep_merge(base_spec.model_dump(mode="python"), overrides)
    return PrinterSpec.model_validate(merged)


def _match_shaper_freq_hz(
    axis_shaper: Any,
    axis_graph: Any,
    shaper_type: str | None,
) -> tuple[float, str]:
    """Return PrintSim's predicted frequency matching the measured Klipper shaper type.

    When ``shaper_type`` is ``"RAW"`` or ``None``, returns ``axis_shaper.effective_freq_hz``.
    Otherwise looks up the matching fitted candidate (e.g. ``MZV``, ``ZV``, ``EI``) in
    ``axis_graph.candidates``, falling back to ``effective_freq_hz`` if unavailable.
    """
    eff_hz = float(axis_shaper.effective_freq_hz)
    if not shaper_type or shaper_type.upper() == "RAW" or axis_graph is None:
        return eff_hz, "RAW"

    target = shaper_type.strip().upper()
    if target.startswith("SMOOTH_"):
        target = target.removeprefix("SMOOTH_")

    for cand in getattr(axis_graph, "candidates", ()) or ():
        if str(cand.name).strip().upper() == target:
            return float(cand.freq_hz), target
    return eff_hz, "RAW"


def _signed_error_pct(predicted: float | None, measured: float | None) -> float | None:
    if predicted is None or measured is None or measured <= 0.0:
        return None
    return round(((predicted - measured) / measured) * 100.0, 2)


def _summarize_errors(rows: list[dict[str, Any]]) -> dict[str, Any]:
    x_signed = [r["x_error_pct"] for r in rows if r["x_error_pct"] is not None]
    y_signed = [r["y_error_pct"] for r in rows if r["y_error_pct"] is not None]
    x_abs = [abs(v) for v in x_signed]
    y_abs = [abs(v) for v in y_signed]
    all_abs = x_abs + y_abs
    n_axes = len(all_abs)

    return {
        "machines_evaluated": len(rows),
        "axes_evaluated": n_axes,
        "x_mape_pct": round(sum(x_abs) / len(x_abs), 2) if x_abs else None,
        "y_mape_pct": round(sum(y_abs) / len(y_abs), 2) if y_abs else None,
        "overall_mape_pct": round(sum(all_abs) / n_axes, 2) if n_axes else None,
        "x_mean_signed_error_pct": round(sum(x_signed) / len(x_signed), 2) if x_signed else None,
        "y_mean_signed_error_pct": round(sum(y_signed) / len(y_signed), 2) if y_signed else None,
        "max_abs_error_pct": round(max(all_abs), 2) if all_abs else None,
        "within_10pct": sum(1 for v in all_abs if v <= 10.0),
        "within_15pct": sum(1 for v in all_abs if v <= 15.0),
        "within_20pct": sum(1 for v in all_abs if v <= 20.0),
    }


def run_shaper_benchmarks(
    manifest_path: Path | None = None,
    examples_dir: Path | None = None,
    tiers: list[int] | tuple[int, ...] | None = None,
    architectures: list[str] | tuple[str, ...] | None = None,
    machine_ids: list[str] | tuple[str, ...] | None = None,
    include_sweeps: bool = True,
) -> dict[str, Any]:
    """Evaluate all selected real-world machine configs and controlled A/B sweeps."""
    manifest = load_shaper_benchmarks(manifest_path)
    tier_set = set(tiers) if tiers else None
    arch_set = {a.strip().lower() for a in architectures} if architectures else None
    id_set = set(machine_ids) if machine_ids else None

    machines_raw: list[dict[str, Any]] = list(manifest.get("machines", []))
    selected: list[dict[str, Any]] = []
    for entry in machines_raw:
        if tier_set is not None and int(entry.get("tier", 2)) not in tier_set:
            continue
        if arch_set is not None and str(entry.get("architecture", "")).strip().lower() not in arch_set:
            continue
        if id_set is not None and str(entry["id"]) not in id_set:
            continue
        selected.append(entry)

    results: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}

    for entry in selected:
        spec = build_benchmark_spec(entry, examples_dir=examples_dir)
        report = evaluate_printer(spec, include_advanced_analysis=False)
        modal = report.modal_fea
        meas = entry.get("measured") or {}

        meas_x = float(meas["x_freq_hz"]) if meas.get("x_freq_hz") is not None else None
        meas_y = float(meas["y_freq_hz"]) if meas.get("y_freq_hz") is not None else None
        x_shaper = meas.get("x_shaper")
        y_shaper = meas.get("y_shaper")

        pred_x_matched, matched_x_type = _match_shaper_freq_hz(
            modal.shaper_x, modal.shaper_graph_x, x_shaper
        )
        pred_y_matched, matched_y_type = _match_shaper_freq_hz(
            modal.shaper_y, modal.shaper_graph_y, y_shaper
        )
        pred_x_eff = float(modal.shaper_x.effective_freq_hz)
        pred_y_eff = float(modal.shaper_y.effective_freq_hz)

        x_err = _signed_error_pct(pred_x_matched, meas_x)
        y_err = _signed_error_pct(pred_y_matched, meas_y)

        meas_ratio = round(meas_x / meas_y, 3) if (meas_x and meas_y and meas_y > 0) else None
        pred_ratio = round(pred_x_matched / pred_y_matched, 3) if pred_y_matched > 0 else None

        row = {
            "id": str(entry["id"]),
            "name": str(entry["name"]),
            "tier": int(entry.get("tier", 2)),
            "architecture": str(entry.get("architecture", spec.architecture)),
            "base_preset": str(entry["base_preset"]),
            "source": str(entry.get("source", "")),
            "x_shaper": x_shaper,
            "matched_x_shaper": matched_x_type,
            "measured_x_hz": round(meas_x, 2) if meas_x is not None else None,
            "predicted_x_hz": round(pred_x_matched, 2),
            "predicted_x_eff_hz": round(pred_x_eff, 2),
            "x_error_pct": x_err,
            "y_shaper": y_shaper,
            "matched_y_shaper": matched_y_type,
            "measured_y_hz": round(meas_y, 2) if meas_y is not None else None,
            "predicted_y_hz": round(pred_y_matched, 2),
            "predicted_y_eff_hz": round(pred_y_eff, 2),
            "y_error_pct": y_err,
            "measured_xy_ratio": meas_ratio,
            "predicted_xy_ratio": pred_ratio,
            "predicted_rec_x": (
                f"{modal.shaper_graph_x.recommended_shaper}@{modal.shaper_graph_x.recommended_freq_hz:.1f}"
                if modal.shaper_graph_x
                else None
            ),
            "predicted_rec_y": (
                f"{modal.shaper_graph_y.recommended_shaper}@{modal.shaper_graph_y.recommended_freq_hz:.1f}"
                if modal.shaper_graph_y
                else None
            ),
            "predicted_max_accel_mm_s2": round(float(modal.recommended_max_accel_mm_s2), 0),
        }
        results.append(row)
        by_id[row["id"]] = row

    sweep_results: list[dict[str, Any]] = []
    if include_sweeps:
        for sw in manifest.get("controlled_sweeps", []) or []:
            base_id = str(sw["baseline_machine"])
            var_id = str(sw["variant_machine"])
            if base_id not in by_id or var_id not in by_id:
                continue
            axis = str(sw["axis"]).upper()
            key = "predicted_x_hz" if axis == "X" else "predicted_y_hz"
            base_pred = float(by_id[base_id][key])
            var_pred = float(by_id[var_id][key])
            pred_ratio = round(var_pred / base_pred, 3) if base_pred > 0 else 0.0
            meas_ratio = float(sw["measured_ratio"])
            ratio_err_pct = _signed_error_pct(pred_ratio, meas_ratio)
            direction_match = (pred_ratio >= 1.0) == (meas_ratio >= 1.0)
            sweep_results.append(
                {
                    "id": str(sw["id"]),
                    "name": str(sw["name"]),
                    "axis": axis,
                    "baseline_machine": base_id,
                    "variant_machine": var_id,
                    "measured_ratio": round(meas_ratio, 3),
                    "predicted_ratio": pred_ratio,
                    "ratio_error_pct": ratio_err_pct,
                    "direction_match": direction_match,
                    "source": str(sw.get("source", "")),
                }
            )

    tier1_rows = [r for r in results if r["tier"] == 1]
    tier2_rows = [r for r in results if r["tier"] == 2]
    sweep_abs_errs = [
        abs(s["ratio_error_pct"]) for s in sweep_results if s["ratio_error_pct"] is not None
    ]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "manifest_version": manifest.get("version", 1),
        "summary": {
            "all": _summarize_errors(results),
            "tier1": _summarize_errors(tier1_rows),
            "tier2": _summarize_errors(tier2_rows),
            "sweeps_evaluated": len(sweep_results),
            "sweeps_direction_match": sum(1 for s in sweep_results if s["direction_match"]),
            "sweeps_ratio_mape_pct": (
                round(sum(sweep_abs_errs) / len(sweep_abs_errs), 2) if sweep_abs_errs else None
            ),
        },
        "machines": results,
        "controlled_sweeps": sweep_results,
    }


def save_benchmark_report(report: dict[str, Any], out_path: Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return out


def compare_benchmark_reports(
    baseline: dict[str, Any],
    current: dict[str, Any],
    eps_pct: float = 0.25,
) -> dict[str, Any]:
    """Diff a current shaper benchmark report against a saved baseline snapshot."""
    base_map = {m["id"]: m for m in baseline.get("machines", [])}
    curr_map = {m["id"]: m for m in current.get("machines", [])}

    machine_diffs: list[dict[str, Any]] = []
    improved_axes = 0
    regressed_axes = 0
    unchanged_axes = 0

    for mid, cm in curr_map.items():
        bm = base_map.get(mid)
        if bm is None:
            continue

        row: dict[str, Any] = {
            "id": mid,
            "name": cm["name"],
            "tier": cm["tier"],
        }
        for ax in ("x", "y"):
            b_pred = bm.get(f"predicted_{ax}_hz")
            c_pred = cm.get(f"predicted_{ax}_hz")
            b_err = bm.get(f"{ax}_error_pct")
            c_err = cm.get(f"{ax}_error_pct")
            row[f"baseline_{ax}_hz"] = b_pred
            row[f"current_{ax}_hz"] = c_pred
            row[f"delta_{ax}_hz"] = (
                round(c_pred - b_pred, 2) if (b_pred is not None and c_pred is not None) else None
            )
            row[f"baseline_{ax}_err_pct"] = b_err
            row[f"current_{ax}_err_pct"] = c_err
            if b_err is not None and c_err is not None:
                delta_abs = round(abs(c_err) - abs(b_err), 2)
                row[f"delta_{ax}_abs_err_pct"] = delta_abs
                if delta_abs < -eps_pct:
                    improved_axes += 1
                    row[f"{ax}_status"] = "improved"
                elif delta_abs > eps_pct:
                    regressed_axes += 1
                    row[f"{ax}_status"] = "regressed"
                else:
                    unchanged_axes += 1
                    row[f"{ax}_status"] = "unchanged"
            else:
                row[f"delta_{ax}_abs_err_pct"] = None
                row[f"{ax}_status"] = None
        machine_diffs.append(row)

    base_all = baseline.get("summary", {}).get("all", {})
    curr_all = current.get("summary", {}).get("all", {})
    base_t1 = baseline.get("summary", {}).get("tier1", {})
    curr_t1 = current.get("summary", {}).get("tier1", {})

    def _diff_opt(c: float | None, b: float | None) -> float | None:
        if c is None or b is None:
            return None
        return round(c - b, 2)

    return {
        "baseline_generated_at": baseline.get("generated_at"),
        "current_generated_at": current.get("generated_at"),
        "overall_mape_delta_pct": _diff_opt(
            curr_all.get("overall_mape_pct"), base_all.get("overall_mape_pct")
        ),
        "tier1_mape_delta_pct": _diff_opt(
            curr_t1.get("overall_mape_pct"), base_t1.get("overall_mape_pct")
        ),
        "improved_axes": improved_axes,
        "regressed_axes": regressed_axes,
        "unchanged_axes": unchanged_axes,
        "machines": machine_diffs,
    }
