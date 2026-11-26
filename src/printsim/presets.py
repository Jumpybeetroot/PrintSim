"""Curated Studio default printer presets vs. test/benchmark reference builds.

This module is pure stdlib + Pydantic (Pyodide WASM safe) and serves as the single
source of truth for:
  1. ``STUDIO_DEFAULT_PRESETS`` — Clean, generic, architecture-grouped starter
     printers exposed in the Studio UI (``/api/catalog``) and pre-baked for
     GitHub Pages. None of these reference specific commercial or community
     printer builds.
  2. Test / benchmark reference YAMLs in ``examples/`` — Retained exclusively for
     automated physics tests, touchstone calibration, CLI benchmarks, and direct
     ``/api/presets/{id}`` or MCP ``get_preset_yaml`` lookups.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from printsim.schema import PrinterSpec

# Display order of printer type categories in Studio UI
PRINTER_TYPE_ORDER: tuple[str, ...] = (
    "corexy",
    "cartesian_bedslinger",
    "delta",
)

PRINTER_TYPE_LABELS: dict[str, str] = {
    "corexy": "CoreXY",
    "cartesian_bedslinger": "Cartesian Bedslinger",
    "delta": "Delta",
}

DEFAULT_STARTUP_PRESET_ID = "corexy_fixed_gantry_300"

# Ordered list of clean, generic default starter printers shown in the Studio UI.
# Specific community/commercial printer builds in examples/ are excluded here and
# reserved strictly for testing, calibration, and benchmarking.
STUDIO_DEFAULT_PRESET_META: tuple[dict[str, str], ...] = (
    # CoreXY
    {
        "id": "corexy_fixed_gantry_300",
        "short_name": "Fixed Gantry 300 (2WD)",
        "category": "corexy",
        "category_label": "CoreXY",
        "summary": "300 mm · Triple Z · 2WD",
    },
    {
        "id": "corexy_flying_gantry_300",
        "short_name": "Flying Gantry 300 (2WD)",
        "category": "corexy",
        "category_label": "CoreXY",
        "summary": "300 mm · Quad-Z Belted · 2WD",
    },
    {
        "id": "corexy_awd_300",
        "short_name": "High-Speed AWD 300",
        "category": "corexy",
        "category_label": "CoreXY",
        "summary": "300 mm · 4-Motor AWD · Structural Panels",
    },
    {
        "id": "hyper_corexy_carbon",
        "short_name": "Carbon Truss 300 (2WD)",
        "category": "corexy",
        "category_label": "CoreXY",
        "summary": "300 mm · 3030 Truss · Carbon X",
    },
    # Cartesian Bedslinger
    {
        "id": "cartesian_bedslinger",
        "short_name": "Standard Bedslinger 220",
        "category": "cartesian_bedslinger",
        "category_label": "Cartesian Bedslinger",
        "summary": "220 mm · Open Portal · Moving Y",
    },
    {
        "id": "cartesian_bedslinger_braced_300",
        "short_name": "Braced Bedslinger 300",
        "category": "cartesian_bedslinger",
        "category_label": "Cartesian Bedslinger",
        "summary": "300 mm · 3030 Braced · Dual Z",
    },
    # Delta
    {
        "id": "delta_racer",
        "short_name": "3-Tower Carbon Delta 240",
        "category": "delta",
        "category_label": "Delta",
        "summary": "240 mm · Bowden · Carbon Arms",
    },
    {
        "id": "delta_direct_drive_300",
        "short_name": "Tall Direct-Drive Delta 300",
        "category": "delta",
        "category_label": "Delta",
        "summary": "300 mm · Direct Drive · 3030 Frame",
    },
)

STUDIO_DEFAULT_PRESET_IDS: tuple[str, ...] = tuple(
    item["id"] for item in STUDIO_DEFAULT_PRESET_META
)
_STUDIO_META_BY_ID: dict[str, dict[str, str]] = {
    item["id"]: item for item in STUDIO_DEFAULT_PRESET_META
}


def find_examples_dir() -> Path:
    """Locate the ``examples/`` directory in repo or Pyodide WASM virtual FS."""
    here = Path(__file__).resolve()
    for parent in (here.parents[2], here.parents[1], Path("/printsim")):
        cand = parent / "examples"
        if cand.is_dir():
            return cand
    return here.parents[2] / "examples"


def is_studio_default_preset(preset_id: str) -> bool:
    """Return True if ``preset_id`` is a generic Studio starter preset."""
    return preset_id in _STUDIO_META_BY_ID


def load_presets(*, include_test_builds: bool = False) -> dict[str, PrinterSpec]:
    """Load printer presets from ``examples/``.

    Args:
        include_test_builds: When ``False`` (default, used for ``/api/catalog``
            and GitHub Pages), returns only the curated generic Studio starter
            presets in display order. When ``True`` (used for direct
            ``/api/presets/{id}`` lookups and test suites), loads all YAML
            presets in ``examples/`` including test/benchmark reference builds.
    """
    examples_dir = find_examples_dir()
    if not examples_dir.exists():
        return {}

    presets: dict[str, PrinterSpec] = {}
    if not include_test_builds:
        for pid in STUDIO_DEFAULT_PRESET_IDS:
            p = examples_dir / f"{pid}.yaml"
            if not p.is_file():
                continue
            try:
                presets[pid] = PrinterSpec.from_yaml(p)
            except Exception:
                continue
        return presets

    # Load curated defaults first, then remaining test/benchmark builds sorted by stem
    for pid in STUDIO_DEFAULT_PRESET_IDS:
        p = examples_dir / f"{pid}.yaml"
        if p.is_file():
            try:
                presets[pid] = PrinterSpec.from_yaml(p)
            except Exception:
                continue
    for p in sorted(examples_dir.glob("*.yaml")):
        if p.stem in presets:
            continue
        try:
            presets[p.stem] = PrinterSpec.from_yaml(p)
        except Exception:
            continue
    return presets


def build_catalog_presets_payload() -> list[dict[str, Any]]:
    """Return the curated, printer-type-categorized default presets for ``/api/catalog``."""
    presets = load_presets(include_test_builds=False)
    items: list[dict[str, Any]] = []
    for pid, spec in presets.items():
        meta = _STUDIO_META_BY_ID.get(pid, {})
        arch = spec.architecture
        items.append(
            {
                "id": pid,
                "name": spec.name,
                "short_name": meta.get("short_name", spec.name),
                "architecture": arch,
                "category": meta.get("category", arch),
                "category_label": meta.get(
                    "category_label", PRINTER_TYPE_LABELS.get(arch, arch)
                ),
                "summary": meta.get("summary", ""),
                "description": spec.description,
            }
        )
    return items
