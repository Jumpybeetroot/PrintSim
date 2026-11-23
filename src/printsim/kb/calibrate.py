"""Calibration comparator benchmarking PrintSim modal/shaper predictions against real KB builds."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from printsim.engine import evaluate_printer
from printsim.kb.extract import (
    MAX_SHAPER_FREQ_HZ,
    MIN_SHAPER_FREQ_HZ,
    infer_preset_hint,
    is_malformed_build_name,
)
from printsim.kb.store import TouchstoneKB
from printsim.schema import PrinterSpec

_UNMAPPED_OUTLIER_RE = re.compile(
    r"\b(?:ender\s*[567]|creality\s*k1|k1\s*max|k1c|bambu(?:\s*lab)?|x1c|p1[ps]|a1\s*mini|"
    r"qidi|sovol\s*sv0[78]|tronxy|sapphire|anycubic|flashforge|elegoo\s*centauri|"
    r"snapmaker|biqu\s*hurakan|flying\s*bear|kingroon|artillery)\b",
    re.IGNORECASE,
)
_CONFIDENT_VORON_COREXY_RE = re.compile(
    r"\b(?:voron|v2\.4|2\.4|trident|vt\s*\d|monolith|doomcube|doomish|hevort|annex\s*k3|v0(?:\.\d)?)\b",
    re.IGNORECASE,
)
_KNOWN_PRINTER_IDENTITY_RE = re.compile(
    r"\b(?:voron|v2\.4|2\.4|trident|vt\s*\d|monolith|doomcube|doomish|hevort|annex\s*k3|v0(?:\.\d)?|"
    r"vzbot|vz\s*330|vz\s*235|ratrig|v-?core|ender\s*3|prusa|mk[234]\w*|switchwire|"
    r"flsun|doron\s*velta|kossel|delta|bedslinger|corexy)\b",
    re.IGNORECASE,
)


def _valid_shaper_hz(val: Any) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if MIN_SHAPER_FREQ_HZ <= f <= MAX_SHAPER_FREQ_HZ:
        return f
    return None


def _has_confident_preset_match(build: dict[str, Any], preset_path: Path) -> bool:
    """Return True only when a build maps confidently to a supported PrintSim reference preset."""
    build_name = str(build.get("build_name") or "").strip()
    if is_malformed_build_name(build_name):
        return False

    has_identity = bool(
        str(build.get("printer_model") or "").strip()
        or str(build.get("frame_profile") or "").strip()
        or str(build.get("drive_mode") or "").strip()
        or _KNOWN_PRINTER_IDENTITY_RE.search(build_name)
    )
    if not has_identity:
        return False

    identity = " ".join(
        str(build.get(k) or "") for k in ("build_name", "printer_model")
    ).strip()
    full_text = " ".join(
        str(build.get(k) or "")
        for k in ("build_name", "printer_model", "frame_profile", "panel_spec", "notes")
    ).strip()

    # Exclude obvious unmapped non-Voron community outliers (e.g. Ender 7, Bambu X1C, Creality K1)
    if _UNMAPPED_OUTLIER_RE.search(identity) and not _CONFIDENT_VORON_COREXY_RE.search(identity):
        return False

    if preset_path.stem == "corexy_voron_style":
        if _UNMAPPED_OUTLIER_RE.search(full_text) and not _CONFIDENT_VORON_COREXY_RE.search(full_text):
            return False
        if _CONFIDENT_VORON_COREXY_RE.search(full_text):
            return True
        if build.get("frame_profile") and build.get("drive_mode"):
            return True
        return False

    return True


def _default_examples_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "examples"


def _resolve_preset_yaml(build: dict[str, Any], examples_dir: Path) -> Path | None:
    """Resolve the closest PrintSim example YAML file for a real KB build record."""
    identity = " ".join(
        str(build.get(k) or "") for k in ("printer_model", "build_name")
    ).lower()
    hint = str(build.get("preset_hint") or "").strip()

    # Re-map RatRig / V-Core records (including previously seeded rows hinted as hyper_corexy_carbon)
    # to the dedicated RatRig V-Core 3.1 300 preset when present.
    if any(tok in identity for tok in ("ratrig", "v-core", "vcore")) and hint in (
        "",
        "hyper_corexy_carbon",
        "hyper_corexy_carbon.yaml",
    ):
        ratrig_cand = examples_dir / "ratrig_vcore31_300.yaml"
        if ratrig_cand.exists():
            return ratrig_cand

    if hint:
        cand = examples_dir / (hint if hint.endswith(".yaml") else f"{hint}.yaml")
        if cand.exists():
            return cand

    inferred = infer_preset_hint(
        architecture=str(build.get("architecture") or "corexy"),
        printer_model=str(build.get("printer_model") or build.get("build_name") or ""),
        frame_profile=str(build.get("frame_profile") or ""),
        drive_mode=str(build.get("drive_mode") or ""),
        belt_width_mm=build.get("belt_width_mm"),
        panel_spec=str(build.get("panel_spec") or ""),
    )
    cand = examples_dir / f"{inferred}.yaml"
    if cand.exists():
        return cand

    fallback = examples_dir / "corexy_voron_style.yaml"
    return fallback if fallback.exists() else None


def _signed_error_pct(predicted: float, measured: float | None) -> float | None:
    if measured is None or measured <= 0:
        return None
    return round(((predicted - measured) / measured) * 100.0, 2)


def run_kb_calibration(
    kb: TouchstoneKB,
    examples_dir: Path | None = None,
) -> dict[str, Any]:
    """Compare PrintSim's modal/input-shaper predictions against real measured builds in SQLite."""
    ex_dir = Path(examples_dir) if examples_dir is not None else _default_examples_dir()
    builds = [
        b
        for b in kb.list_builds()
        if b.get("x_shaper_hz") is not None or b.get("y_shaper_hz") is not None
    ]

    eval_cache: dict[Path, Any] = {}
    results: list[dict[str, Any]] = []
    x_abs_errs: list[float] = []
    y_abs_errs: list[float] = []
    x_signed_errs: list[float] = []
    y_signed_errs: list[float] = []
    skipped_invalid_hz = 0
    skipped_unmapped = 0

    for b in builds:
        meas_x = _valid_shaper_hz(b.get("x_shaper_hz"))
        meas_y = _valid_shaper_hz(b.get("y_shaper_hz"))
        if meas_x is None and meas_y is None:
            skipped_invalid_hz += 1
            continue

        preset_path = _resolve_preset_yaml(b, ex_dir)
        if preset_path is None or not _has_confident_preset_match(b, preset_path):
            skipped_unmapped += 1
            continue

        if preset_path not in eval_cache:
            spec = PrinterSpec.from_yaml(preset_path)
            eval_cache[preset_path] = evaluate_printer(spec, include_advanced_analysis=False)

        report = eval_cache[preset_path]
        pred_x = round(float(report.modal_fea.shaper_x.effective_freq_hz), 2)
        pred_y = round(float(report.modal_fea.shaper_y.effective_freq_hz), 2)

        x_err = _signed_error_pct(pred_x, meas_x)
        y_err = _signed_error_pct(pred_y, meas_y)

        if x_err is not None:
            x_signed_errs.append(x_err)
            x_abs_errs.append(abs(x_err))
        if y_err is not None:
            y_signed_errs.append(y_err)
            y_abs_errs.append(abs(y_err))

        meas_ratio = round(meas_x / meas_y, 2) if (meas_x and meas_y and meas_y > 0) else None
        pred_ratio = round(pred_x / pred_y, 2) if pred_y > 0 else None

        results.append(
            {
                "build_id": b["id"],
                "build_name": b["build_name"],
                "drive_mode": b.get("drive_mode") or "",
                "frame_profile": b.get("frame_profile") or "",
                "preset_file": preset_path.name,
                "preset_name": report.spec.name,
                "measured_x_hz": meas_x,
                "predicted_x_hz": pred_x,
                "x_error_pct": x_err,
                "measured_y_hz": meas_y,
                "predicted_y_hz": pred_y,
                "y_error_pct": y_err,
                "measured_xy_ratio": meas_ratio,
                "predicted_xy_ratio": pred_ratio,
                "predicted_max_accel_mm_s2": round(float(report.modal_fea.recommended_max_accel_mm_s2), 0),
                "measured_x_max_accel": b.get("x_max_accel"),
                "measured_y_max_accel": b.get("y_max_accel"),
            }
        )

    all_abs_errs = x_abs_errs + y_abs_errs
    return {
        "builds_evaluated": len(results),
        "builds_skipped_unmapped": skipped_unmapped,
        "builds_skipped_invalid_hz": skipped_invalid_hz,
        "presets_evaluated": len(eval_cache),
        "x_mape_pct": round(sum(x_abs_errs) / len(x_abs_errs), 2) if x_abs_errs else None,
        "y_mape_pct": round(sum(y_abs_errs) / len(y_abs_errs), 2) if y_abs_errs else None,
        "overall_mape_pct": round(sum(all_abs_errs) / len(all_abs_errs), 2) if all_abs_errs else None,
        "x_mean_signed_error_pct": round(sum(x_signed_errs) / len(x_signed_errs), 2) if x_signed_errs else None,
        "y_mean_signed_error_pct": round(sum(y_signed_errs) / len(y_signed_errs), 2) if y_signed_errs else None,
        "results": results,
    }


_TARGET_BUILD_BUCKETS: list[tuple[str, str, int, str]] = [
    ("voron_trident_2wd_6mm", "Stock Voron Trident (2020, 2WD 6mm)", 3, "Voron Discord #voron_trident_questions / shaper graphs"),
    ("voron_24_flying_2wd", "Stock Voron 2.4 Flying Gantry (2020, 2WD 6mm)", 3, "Voron Discord #voron_2_questions / shaper graphs"),
    ("corexy_9mm_2wd", "Stiff 2WD CoreXY (9mm belt, 3030/4040 or paneled)", 2, "Monolith / RatRig / VzBot build logs"),
    ("corexy_awd_9_12mm", "High-Speed AWD CoreXY (9–12mm belt, 48V)", 3, "Monolith #monolith-builds / VzBot AWD shaper graphs"),
    ("vzbot_awd", "VzBot 235/330 AWD (CF/Al gantry)", 2, "VzBot Discord / GitHub shaper benchmarks"),
    ("bedslinger", "Cartesian Bedslinger (Ender 3 / Prusa MK4 / Switchwire)", 2, "Klipper Discourse / r/klippers shaper logs"),
    ("delta", "Delta Printer (Flsun / Kossel / Doron Velta)", 2, "Delta / Klipper shaper benchmarks"),
]

_TARGET_WEIGHT_CATEGORIES: list[tuple[str, str, int, str]] = [
    ("toolhead", "Complete Dressed Toolheads (StealthBurner, XOL, A4T, Dragon Burner, CPAP)", 5, "Scale-weighed toolhead photos/tables"),
    ("carriage", "XY-Joints & Carriages (Printed ABS/CF vs CNC 6061/7075 billet)", 3, "Monolith #printed-gantry / CNC kit weights"),
    ("gantry", "Moving X-Gantries (2020 Al vs CF tube vs CNC Al beam + MGN rail)", 3, "Scale-weighed gantry beam assemblies"),
    ("hotend", "Assembled Hotends (Dragon, Rapido, Revo, Goliath, Chube)", 3, "Manufacturer datasheets / scale checks"),
]

_TARGET_MATERIAL_CHECKS: list[tuple[str, str, str]] = [
    ("ppa_cf", "PPA-CF modulus, annealing & sustained belt creep", "MyTechFun / Monolith #printed-gantry"),
    ("pps_cf", "PPS-CF modulus, HDT & creep vs PPA-CF", "MyTechFun / Bambu / Siraya TDS"),
    ("pet_cf", "PET-CF15 modulus & creep limit", "MyTechFun / Monolith #printed-gantry"),
    ("panel_al_vs_acm", "4-5mm Al 5052 vs 3mm ACM vs Acrylic before/after IS delta", "Monolith #panels-and-frame"),
    ("belt_cores", "GT2 Fiberglass vs Aramid/Kevlar vs GT3 matched-tension shootout", "Monolith #updates / CloakedWayne tests"),
]


def _classify_build_bucket(b: dict[str, Any]) -> str:
    arch = str(b.get("architecture") or "corexy").lower()
    text = " ".join(
        str(b.get(k) or "")
        for k in ("build_name", "printer_model", "frame_profile", "panel_spec", "preset_hint", "notes")
    ).lower()
    drive = str(b.get("drive_mode") or "").lower()
    belt_w = float(b.get("belt_width_mm") or 0.0)

    if arch == "cartesian_bedslinger" or any(
        k in text for k in ("bedslinger", "ender 3", "ender-3", "ender3", "mk4", "switchwire")
    ):
        return "bedslinger"
    if arch == "delta" or "delta" in text:
        return "delta"
    if "vzbot" in text or "vz330" in text or "vz235" in text:
        return "vzbot_awd"
    if drive == "awd" or "awd" in text:
        return "corexy_awd_9_12mm"
    if "2.4" in text or "v2.4" in text or "flying" in text:
        return "voron_24_flying_2wd"
    if belt_w >= 9.0 or "3030" in text or "4040" in text or "ratrig" in text or "v-core" in text:
        return "corexy_9mm_2wd"
    return "voron_trident_2wd_6mm"


def analyze_coverage_gaps(kb: TouchstoneKB) -> dict[str, Any]:
    """Identify which real-world build classes, component weights, and material tests are under-covered."""
    builds = kb.list_builds()
    bucket_counts: dict[str, int] = {k: 0 for k, _, _, _ in _TARGET_BUILD_BUCKETS}
    for b in builds:
        if _valid_shaper_hz(b.get("x_shaper_hz")) is None and _valid_shaper_hz(b.get("y_shaper_hz")) is None:
            continue
        bkt = _classify_build_bucket(b)
        bucket_counts[bkt] = bucket_counts.get(bkt, 0) + 1

    build_gaps: list[dict[str, Any]] = []
    for key, label, target, where in _TARGET_BUILD_BUCKETS:
        have = bucket_counts.get(key, 0)
        build_gaps.append(
            {
                "bucket": key,
                "label": label,
                "have": have,
                "target": target,
                "missing": max(0, target - have),
                "status": "covered" if have >= target else ("thin" if have > 0 else "missing"),
                "suggested_source": where,
            }
        )

    cur = kb.conn.cursor()
    weight_rows = cur.execute(
        "SELECT lower(category), COUNT(*) FROM component_weights GROUP BY lower(category)"
    ).fetchall()
    w_counts = {str(r[0]): int(r[1]) for r in weight_rows}
    weight_gaps: list[dict[str, Any]] = []
    for key, label, target, where in _TARGET_WEIGHT_CATEGORIES:
        have = w_counts.get(key, 0)
        weight_gaps.append(
            {
                "category": key,
                "label": label,
                "have": have,
                "target": target,
                "missing": max(0, target - have),
                "status": "covered" if have >= target else ("thin" if have > 0 else "missing"),
                "suggested_source": where,
            }
        )

    mat_rows = cur.execute("SELECT material_name, test_notes FROM material_tests").fetchall()
    mat_corpus = " ".join(f"{r[0] or ''} {r[1] or ''}" for r in mat_rows).lower()
    mat_matchers = {
        "ppa_cf": ("ppa-cf", "ppa_cf", "ppa"),
        "pps_cf": ("pps-cf", "pps_cf", "pps"),
        "pet_cf": ("pet-cf", "pet_cf"),
        "panel_al_vs_acm": ("5052", "acm", "dibond"),
        "belt_cores": ("aramid", "kevlar", "epdm", "gt3"),
    }
    material_gaps: list[dict[str, Any]] = []
    for key, label, where in _TARGET_MATERIAL_CHECKS:
        covered = any(tok in mat_corpus for tok in mat_matchers[key])
        material_gaps.append(
            {
                "topic": key,
                "label": label,
                "status": "covered" if covered else "missing",
                "suggested_source": where,
            }
        )

    priority_queries: list[str] = []
    for g in build_gaps:
        if g["status"] != "covered":
            priority_queries.append(f"[Build Shaper] {g['label']} ({g['have']}/{g['target']}) — check {g['suggested_source']}")
    for g in weight_gaps:
        if g["status"] != "covered":
            priority_queries.append(f"[Weight] {g['label']} ({g['have']}/{g['target']}) — check {g['suggested_source']}")
    for g in material_gaps:
        if g["status"] != "covered":
            priority_queries.append(f"[Material] {g['label']} — check {g['suggested_source']}")

    return {
        "stats": kb.stats(),
        "build_gaps": build_gaps,
        "weight_gaps": weight_gaps,
        "material_gaps": material_gaps,
        "priority_missing": priority_queries,
    }
