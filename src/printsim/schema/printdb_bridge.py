"""Bridge to PrintDB for hardware component specs, material properties, and physics constants.

PrintDB (https://github.com/Jumpybeetroot/PrintDB) is an optional dependency
providing version-controlled hardware datasheets, material properties, and physics
reference constants with lossless SI conversion (``.to_si()``).

When ``printdb`` is installed and its repository root is accessible, PrintSim
hydrates overlapping catalog entries and reference physics constants from PrintDB
at import time. When ``printdb`` is unavailable (e.g. restricted sandbox or minimal
install), all callers transparently fall back to PrintSim's built-in defaults.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _resolve_printdb_root() -> Path | None:
    """Locate the PrintDB repository root without confusing it with PrintSim's own ``data/``."""
    env_root = os.environ.get("PRINTDB_ROOT")
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        if (candidate / "data").is_dir():
            return candidate

    try:
        import printdb  # type: ignore[import-not-found]
    except ImportError:
        return None

    pkg_file = getattr(printdb, "__file__", None)
    if pkg_file:
        pkg_root = Path(pkg_file).resolve().parents[2]
        if (pkg_root / "data").is_dir() and (pkg_root / "schemas").is_dir():
            return pkg_root

    return None


@lru_cache(maxsize=1)
def get_printdb() -> Any | None:
    """Return an opened ``printdb.PrintDB`` instance, or ``None`` if unavailable."""
    root = _resolve_printdb_root()
    if root is None:
        return None
    try:
        import printdb  # type: ignore[import-not-found]

        return printdb.open(root=root)
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.debug("PrintDB unavailable (%s); using built-in catalogs.", exc)
        return None


def is_printdb_available() -> bool:
    """Return True if PrintDB is installed and its catalog is readable."""
    return get_printdb() is not None


def get_printdb_status() -> dict[str, Any]:
    """Return diagnostic metadata about the active PrintDB connection."""
    db = get_printdb()
    if db is None:
        return {"available": False, "root": None}
    return {
        "available": True,
        "root": str(db.paths.root),
    }


def get_record_specs(record_id: str, *, si: bool = False) -> dict[str, Any] | None:
    """Fetch a PrintDB record's ``specs`` dict (optionally converted to SI), or ``None``."""
    db = get_printdb()
    if db is None:
        return None
    try:
        rec = db.get(record_id)
        return rec.to_si() if si else dict(rec.specs)
    except Exception:
        return None


def get_reference_constant(record_id: str, key: str, default: float) -> float:
    """Fetch a scalar physics constant from a PrintDB ``references/`` record."""
    specs = get_record_specs(record_id, si=False)
    if not specs or key not in specs or specs[key] is None:
        return float(default)
    try:
        return float(specs[key])
    except (TypeError, ValueError):
        return float(default)


def hydrate_stepper_catalog(catalog: dict[str, dict[str, Any]]) -> None:
    """Overlay PrintDB motor datasheet specs onto matching ``STEPPER_CATALOG`` entries."""
    mapping: dict[str, str] = {
        "ldo_2504ac": "components/motors/ldo-42sth48-2504ac",
        "ldo_2804ah": "components/motors/ldo-42sth48-2804ah",
        "moons_420i": "components/motors/moons-ms17hd6p420i-04",
    }
    for key, rid in mapping.items():
        if key not in catalog:
            continue
        si = get_record_specs(rid, si=True)
        raw = get_record_specs(rid, si=False)
        if not si or not raw:
            continue
        entry = catalog[key]
        if si.get("holding_torque_nm") is not None:
            entry["holding_torque_nm"] = float(si["holding_torque_nm"])
        if si.get("rated_current_a") is not None:
            entry["rated_current_a"] = float(si["rated_current_a"])
        if si.get("phase_resistance_ohm") is not None:
            entry["phase_resistance_ohm"] = float(si["phase_resistance_ohm"])
        if raw.get("phase_inductance_mh") is not None:
            entry["inductance_mh"] = float(raw["phase_inductance_mh"])
        if si.get("rotor_inertia_kg_m2") is not None:
            entry["rotor_inertia_kg_m2"] = float(si["rotor_inertia_kg_m2"])
        if raw.get("mass_g") is not None:
            entry["motor_mass_g"] = float(raw["mass_g"])
        if raw.get("body_length_mm") is not None:
            entry["body_length_mm"] = float(raw["body_length_mm"])
        if raw.get("step_angle_deg"):
            entry["steps_per_rev"] = int(round(360.0 / float(raw["step_angle_deg"])))
        entry["printdb_id"] = rid


def hydrate_belt_catalog(catalog: dict[str, dict[str, Any]]) -> None:
    """Overlay PrintDB belt datasheet masses & geometry onto matching ``BELT_CATALOG`` entries."""
    mapping: dict[str, str] = {
        "gt2_6mm": "components/belts/gates-2gt-6mm-ll",
        "gt2_9mm": "components/belts/gates-2gt-9mm-ll",
    }
    for key, rid in mapping.items():
        if key not in catalog:
            continue
        si = get_record_specs(rid, si=True)
        raw = get_record_specs(rid, si=False)
        if not si or not raw:
            continue
        entry = catalog[key]
        if raw.get("pitch_mm") is not None:
            entry["pitch_mm"] = float(raw["pitch_mm"])
        if raw.get("width_mm") is not None:
            entry["width_mm"] = float(raw["width_mm"])
        # Note: keep PrintSim's calibrated linear_mass_kg_m (0.0083 kg/m for 6mm) when
        # PrintDB's rounded 8.4 g/m differs only by ~1%, preserving Voron pluck-Hz calibration.
        entry["printdb_id"] = rid


def hydrate_hotend_catalog(catalog: dict[str, dict[str, Any]]) -> None:
    """Overlay PrintDB hotend datasheet specs onto matching ``HOTEND_CATALOG`` entries."""
    mapping: dict[str, str] = {
        "e3d_revo_voron": "components/hotends/e3d-revo-voron",
    }
    for key, rid in mapping.items():
        if key not in catalog:
            continue
        raw = get_record_specs(rid, si=False)
        if not raw:
            continue
        entry = catalog[key]
        if raw.get("heater_power_w") is not None:
            entry["heater_power_w"] = float(raw["heater_power_w"])
        if raw.get("max_temp_c") is not None:
            entry["max_temp_c"] = float(raw["max_temp_c"])
        # Note: PrintDB records bare Revo Voron mass (25 g); PrintSim's catalog mass_g (30 g)
        # includes heater/thermistor/sock assembly used by BOM & toolhead modal calibrations.
        entry["printdb_id"] = rid


def hydrate_rail_catalog(
    carriage_catalog: dict[str, tuple[float, float, int, float, float, float, float, float]],
    block_mass_g: dict[str, tuple[float, float]],
) -> None:
    """Overlay PrintDB linear rail specs onto ``_RAIL_CARRIAGE_CATALOG`` & ``_RAIL_BLOCK_MASS_G``."""
    mapping: dict[str, str] = {
        "mgn9_rail": "components/linear_rails/hiwin-mgn9h",
        "mgn12_rail": "components/linear_rails/hiwin-mgn12h",
    }
    for key, rid in mapping.items():
        raw = get_record_specs(rid, si=False)
        if not raw:
            continue
        if key in block_mass_g and raw.get("carriage_mass_g") is not None:
            c_mass, _h_mass = block_mass_g[key]
            block_mass_g[key] = (c_mass, float(raw["carriage_mass_g"]))
        if key in carriage_catalog and raw.get("dynamic_load_n") is not None:
            d_b, r_y, n_rows, l1_c, c_c, l1_h, _c_h, seal_drag = carriage_catalog[key]
            carriage_catalog[key] = (
                d_b,
                r_y,
                n_rows,
                l1_c,
                c_c,
                l1_h,
                float(raw["dynamic_load_n"]),
                seal_drag,
            )


def hydrate_material_catalog(catalog: dict[str, Any]) -> None:
    """Overlay PrintDB structural material SI properties onto ``MATERIAL_CATALOG``."""
    mapping: dict[str, str] = {
        "aluminum_6061_t6": "materials/aluminum-6061-t6",
    }
    for key, rid in mapping.items():
        if key not in catalog:
            continue
        si = get_record_specs(rid, si=True)
        if not si:
            continue
        mat = catalog[key]
        if si.get("tensile_modulus_pa") is not None:
            mat.youngs_modulus_pa = float(si["tensile_modulus_pa"])
        if si.get("density_kg_per_m3") is not None:
            mat.density_kg_m3 = float(si["density_kg_per_m3"])
        if si.get("yield_strength_pa") is not None:
            mat.yield_strength_pa = float(si["yield_strength_pa"])
        if si.get("thermal_conductivity_w_per_mk") is not None:
            mat.thermal_conductivity_w_mk = float(si["thermal_conductivity_w_per_mk"])
        if si.get("specific_heat_j_per_kgk") is not None:
            mat.specific_heat_j_kgk = float(si["specific_heat_j_per_kgk"])
        if si.get("cte_per_k") is not None:
            mat.thermal_expansion_1_k = float(si["cte_per_k"])


def hydrate_filament_catalog(catalog: dict[str, Any]) -> None:
    """Overlay PrintDB polymer density & thermal conductivity onto ``FILAMENT_CATALOG``."""
    mapping: dict[str, str] = {
        "pla": "materials/generic-pla",
        "petg": "materials/generic-petg",
        "abs": "materials/generic-abs",
        "asa": "materials/generic-asa",
    }
    for key, rid in mapping.items():
        if key not in catalog:
            continue
        si = get_record_specs(rid, si=True)
        raw = get_record_specs(rid, si=False)
        if not si or not raw:
            continue
        fil = catalog[key]
        if si.get("density_kg_per_m3") is not None:
            fil.density_kg_m3 = float(si["density_kg_per_m3"])
        # Note: keep PrintSim's calibrated melt-zone thermal_conductivity_w_mk and Tg
        # (e.g. PLA Tg=58 C, ABS/ASA k=0.18 W/mK) used by Graetz flow benchmarks.
