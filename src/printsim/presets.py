"""Curated Studio base-machine presets grouped by what the user's machine is based off.

This module is pure stdlib + Pydantic (Pyodide WASM safe) and serves as the single
source of truth for:
  1. ``STUDIO_DEFAULT_PRESETS`` — Base machine kits & DIY starters exposed in the
     Studio UI (``/api/catalog``) and pre-baked for GitHub Pages, grouped by kit
     ecosystem: **Siboor Kit**, **LDO Kit**, and **DIY / Self-Sourced Build**.
  2. Additional benchmark/touchstone YAMLs in ``examples/`` — Accessible via
     ``load_presets(include_test_builds=True)``, direct ``/api/presets/{id}``,
     and MCP ``get_preset_yaml`` lookups.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from printsim.schema import PrinterSpec

# Display order of base machine categories in Studio UI
PRINTER_TYPE_ORDER: tuple[str, ...] = (
    "siboor_kit",
    "ldo_kit",
    "diy_build",
)

PRINTER_TYPE_LABELS: dict[str, str] = {
    "siboor_kit": "Siboor Kit",
    "ldo_kit": "LDO Kit",
    "diy_build": "DIY / Self-Sourced Build",
    # Legacy architecture fallbacks
    "corexy": "CoreXY",
    "cartesian_bedslinger": "Cartesian Bedslinger",
    "delta": "Delta",
}

DEFAULT_STARTUP_PRESET_ID = "ldo_voron_trident_revd_300"

# Ordered list of Studio base machine presets grouped by kit ecosystem
# (Siboor Kit, LDO Kit, DIY Build). Selecting a base machine configures the full
# gantry kit, belt routing, rail sizes & orientations, Z drive/rails/bed mounts,
# and frame/top-to-gantry clearance for that machine.
STUDIO_DEFAULT_PRESET_META: tuple[dict[str, Any], ...] = (
    # ---- Siboor Kits ----
    {
        "id": "siboor_voron_trident_awd_300",
        "short_name": "Siboor Voron Trident AWD (300)",
        "base_kit": "siboor",
        "category": "siboor_kit",
        "category_label": "Siboor Kit",
        "summary": "Siboor CNC AWD · 9 mm Stock · MGN12F/MGN9B · Triple Z · 110 mm Top",
        "highlights": [
            "Siboor CNC AWD (6061)",
            "9 mm Stock CoreXY",
            "MGN12H Front X · MGN9H Bottom Y",
            "Triple TR8×4 Z · MGN9 Z · 110 mm Top",
        ],
    },
    {
        "id": "siboor_voron_24_cnc_300",
        "short_name": "Siboor Voron 2.4r2 CNC (300)",
        "base_kit": "siboor",
        "category": "siboor_kit",
        "category_label": "Siboor Kit",
        "summary": "Siboor CNC 2WD · 6 mm XY / 9 mm Quad Z · MGN12F/MGN9B",
        "highlights": [
            "Siboor CNC 2WD (6061)",
            "6 mm Stock CoreXY",
            "MGN12H Front X · MGN9H Bottom Y",
            "Quad Belted Z (9 mm) · MGN9 Z",
        ],
    },
    # ---- LDO Kits ----
    {
        "id": "ldo_voron_trident_revd_300",
        "short_name": "LDO Voron Trident Rev D (300)",
        "base_kit": "ldo",
        "category": "ldo_kit",
        "category_label": "LDO Kit",
        "summary": "LDO Kit 2WD · 6 mm Stock · MGN12F/MGN9B · Triple Z · 110 mm Top",
        "highlights": [
            "LDO Printed + Matched Hardware",
            "6 mm Stock CoreXY",
            "MGN12H Front X · MGN9H Bottom Y",
            "Triple TR8×4 Z · MGN9 Z · 110 mm Top",
        ],
    },
    {
        "id": "ldo_voron_24_revd_300",
        "short_name": "LDO Voron 2.4r2 Rev D (300)",
        "base_kit": "ldo",
        "category": "ldo_kit",
        "category_label": "LDO Kit",
        "summary": "LDO Kit 2WD · 6 mm Stock · Quad Belted Z (6 mm) · MGN9 Z",
        "highlights": [
            "LDO Printed + Matched Hardware",
            "6 mm Stock CoreXY",
            "MGN12H Front X · MGN9H Bottom Y",
            "Quad Belted Z (6 mm) · MGN9 Z",
        ],
    },
    # ---- DIY / Self-Sourced Builds ----
    {
        "id": "voron_trident_300",
        "short_name": "Voron Trident 300 (DIY Printed)",
        "base_kit": "diy",
        "category": "diy_build",
        "category_label": "DIY / Self-Sourced Build",
        "summary": "Stock Printed 2WD · 6 mm · Triple TR8×4 Z",
        "highlights": [
            "Voron Stock Printed ABS",
            "6 mm Stock CoreXY",
            "MGN12H Front X · MGN9H Bottom Y",
            "Triple TR8×4 Z",
        ],
    },
    {
        "id": "corexy_voron_style",
        "short_name": "Voron 2.4r2 300 (DIY Printed)",
        "base_kit": "diy",
        "category": "diy_build",
        "category_label": "DIY / Self-Sourced Build",
        "summary": "Stock Printed 2WD · 6 mm · Flying Quad Belted Z",
        "highlights": [
            "Voron Stock Printed ABS",
            "6 mm Stock CoreXY",
            "MGN12H Front X · MGN9H Bottom Y",
            "Quad Belted Z (6 mm)",
        ],
    },
    {
        "id": "voron_monolith_awd",
        "short_name": "Monolith × LDO AWD 300 (DIY)",
        "base_kit": "diy",
        "category": "diy_build",
        "category_label": "DIY / Self-Sourced Build",
        "summary": "Monolith CNC AWD · 9 mm Short-Path · Inner MGN12 Y · CF X",
        "highlights": [
            "Monolith × LDO CNC AWD",
            "9 mm Monolith Short-Path",
            "MGN12H Front X · Inner MGN12H Y",
            "Triple TR8×4 Z · 4 mm Bolted Al",
        ],
    },
    {
        "id": "vzbot_330_awd",
        "short_name": "VzBot 330 AWD (Mellow CNC)",
        "base_kit": "diy",
        "category": "diy_build",
        "category_label": "DIY / Self-Sourced Build",
        "summary": "VzBot CNC AWD · 6 mm · MGN9H on CF Tube · Belt-Synced Dual Z",
        "highlights": [
            "VzBot CNC AWD (6061) · 48 V 2804AC",
            "6 mm Stock CoreXY · Y Rails on Top Ring",
            "Top MGN9H X on 20×20 CF · Inner MGN12H Y",
            "Belt-Synced 2× TR8×8 Z · 4× Ø10 mm Rods",
        ],
    },
    {
        "id": "corexy_fixed_gantry_300",
        "short_name": "Custom CoreXY Fixed Gantry 300",
        "base_kit": "diy",
        "category": "diy_build",
        "category_label": "DIY / Self-Sourced Build",
        "summary": "300 mm · Fixed Gantry · Triple Z · 2WD Starter",
        "highlights": [
            "Generic Fixed-Gantry CoreXY",
            "6 mm Stock CoreXY",
            "Triple TR8×4 Z · 110 mm Top",
        ],
    },
    {
        "id": "corexy_flying_gantry_300",
        "short_name": "Custom CoreXY Flying Gantry 300",
        "base_kit": "diy",
        "category": "diy_build",
        "category_label": "DIY / Self-Sourced Build",
        "summary": "300 mm · Flying Gantry · Quad Belted Z · 2WD Starter",
        "highlights": [
            "Generic Flying-Gantry CoreXY",
            "6 mm Stock CoreXY",
            "Quad Belted Z (6 mm)",
        ],
    },
    {
        "id": "hyper_corexy_carbon",
        "short_name": "Carbon Truss CoreXY 300 (DIY)",
        "base_kit": "diy",
        "category": "diy_build",
        "category_label": "DIY / Self-Sourced Build",
        "summary": "300 mm · 3030 Truss · Carbon X-Beam · 2WD",
        "highlights": [
            "3030 Full Truss Frame",
            "Carbon X-Beam",
            "DIY High-Stiffness CoreXY",
        ],
    },
    {
        "id": "cartesian_bedslinger",
        "short_name": "DIY Bedslinger 220",
        "base_kit": "diy",
        "category": "diy_build",
        "category_label": "DIY / Self-Sourced Build",
        "summary": "220 mm · Open Portal · Moving Y Bed",
        "highlights": [
            "Cartesian Bedslinger",
            "220 mm Bed",
            "Open Portal Frame",
        ],
    },
    {
        "id": "cartesian_bedslinger_braced_300",
        "short_name": "Braced Bedslinger 300 (DIY)",
        "base_kit": "diy",
        "category": "diy_build",
        "category_label": "DIY / Self-Sourced Build",
        "summary": "300 mm · 3030 Braced · Dual Z",
        "highlights": [
            "Cartesian Bedslinger",
            "3030 Braced Frame",
            "Dual Leadscrew Z",
        ],
    },
    {
        "id": "delta_racer",
        "short_name": "Carbon Delta 240 (DIY)",
        "base_kit": "diy",
        "category": "diy_build",
        "category_label": "DIY / Self-Sourced Build",
        "summary": "240 mm · 3-Tower Bowden · Carbon Arms",
        "highlights": [
            "3-Tower Delta",
            "Bowden Effector",
            "Carbon Arms",
        ],
    },
    {
        "id": "delta_direct_drive_300",
        "short_name": "Direct-Drive Delta 300 (DIY)",
        "base_kit": "diy",
        "category": "diy_build",
        "category_label": "DIY / Self-Sourced Build",
        "summary": "300 mm · 3030 Towers · Direct Drive",
        "highlights": [
            "3-Tower Delta",
            "3030 Tall Frame",
            "Direct Drive",
        ],
    },
)

STUDIO_DEFAULT_PRESET_IDS: tuple[str, ...] = tuple(
    item["id"] for item in STUDIO_DEFAULT_PRESET_META
)
_STUDIO_META_BY_ID: dict[str, dict[str, Any]] = {
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
    """Return True if ``preset_id`` is a Studio base-machine preset."""
    return preset_id in _STUDIO_META_BY_ID


def load_presets(*, include_test_builds: bool = False) -> dict[str, PrinterSpec]:
    """Load printer presets from ``examples/``.

    Args:
        include_test_builds: When ``False`` (default, used for ``/api/catalog``
            and GitHub Pages), returns only the curated Studio base-machine
            presets in display order. When ``True`` (used for direct
            ``/api/presets/{id}`` lookups and test suites), loads all YAML
            presets in ``examples/`` including benchmark reference builds.
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

    # Load curated Studio base machines first, then remaining benchmark builds sorted by stem
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
    """Return the curated, base-kit-categorized presets for ``/api/catalog``."""
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
                "base_kit": meta.get("base_kit", "diy"),
                "category": meta.get("category", "diy_build"),
                "category_label": meta.get(
                    "category_label", PRINTER_TYPE_LABELS.get(arch, arch)
                ),
                "hardware_kit": spec.resolve_effective_gantry_hardware_kit(),
                "summary": meta.get("summary", ""),
                "highlights": list(meta.get("highlights", [])),
                "description": spec.description,
            }
        )
    return items
