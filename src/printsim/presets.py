"""Curated Studio base-machine presets and 3-tier model maturity classification.

This module is pure stdlib + Pydantic (Pyodide WASM safe) and serves as the single
source of truth for classifying PrintSim machines and hardware catalogs into three
maturity tiers:

  1. ``complete`` — **Complete Models**: Named real machines/components with
     authentic geometry, PrintDB-backed specs, dedicated baked 3D CAD assets
     (``.glb`` + Tet4 Craig-Bampton superelements), and benchmark/datasheet validation.
  2. ``wip_specific`` — **WIP Specific Models**: Named real machines/components that
     are partially modeled (e.g., borrowing another kit's gantry brackets or awaiting
     dedicated baked CAD/calibration), each carrying a concise ``wip_notes`` summary.
  3. ``generic_standin`` — **Generic Stand-Ins**: Parametric, procedural, or archetype
     templates not tied to a specific vendor/community CAD kit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from printsim.schema import PrinterSpec

ModelStatus = Literal["complete", "wip_specific", "generic_standin"]

# Display order of 3-tier maturity categories in Studio UI
MODEL_STATUS_ORDER: tuple[ModelStatus, ...] = (
    "complete",
    "wip_specific",
    "generic_standin",
)

MODEL_STATUS_LABELS: dict[str, str] = {
    "complete": "Complete Models",
    "wip_specific": "WIP Specific Models",
    "generic_standin": "Generic Stand-Ins",
}

PRINTER_TYPE_ORDER: tuple[str, ...] = MODEL_STATUS_ORDER

PRINTER_TYPE_LABELS: dict[str, str] = {
    **MODEL_STATUS_LABELS,
    # Legacy architecture / kit fallbacks
    "siboor_kit": "Siboor Kit",
    "ldo_kit": "LDO Kit",
    "diy_build": "DIY / Self-Sourced Build",
    "corexy": "CoreXY",
    "cartesian_bedslinger": "Cartesian Bedslinger",
    "delta": "Delta",
}

DEFAULT_STARTUP_PRESET_ID = "ldo_voron_trident_revd_300"

# Ordered list of all Studio presets grouped by maturity tier
# (Complete Models, WIP Specific Models, Generic Stand-Ins).
STUDIO_DEFAULT_PRESET_META: tuple[dict[str, Any], ...] = (
    # =========================================================================
    # Complete Models — Full CAD-backed, PrintDB-verified real machines
    # =========================================================================
    {
        "id": "siboor_voron_trident_awd_300",
        "short_name": "Siboor Voron Trident AWD (300)",
        "base_kit": "siboor",
        "category": "complete",
        "category_label": "Complete Models",
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
        "category": "complete",
        "category_label": "Complete Models",
        "summary": "Siboor CNC 2WD · 6 mm XY / 9 mm Quad Z · MGN12F/MGN9B",
        "highlights": [
            "Siboor CNC 2WD (6061)",
            "6 mm Stock CoreXY",
            "MGN12H Front X · MGN9H Bottom Y",
            "Quad Belted Z (9 mm) · MGN9 Z",
        ],
    },
    {
        "id": "ldo_voron_trident_revd_300",
        "short_name": "LDO Voron Trident Rev D (300)",
        "base_kit": "ldo",
        "category": "complete",
        "category_label": "Complete Models",
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
        "category": "complete",
        "category_label": "Complete Models",
        "summary": "LDO Kit 2WD · 6 mm Stock · Quad Belted Z (6 mm) · MGN9 Z",
        "highlights": [
            "LDO Printed + Matched Hardware",
            "6 mm Stock CoreXY",
            "MGN12H Front X · MGN9H Bottom Y",
            "Quad Belted Z (6 mm) · MGN9 Z",
        ],
    },
    {
        "id": "voron_trident_300",
        "short_name": "Voron Trident 300 (DIY Printed)",
        "base_kit": "diy",
        "category": "complete",
        "category_label": "Complete Models",
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
        "category": "complete",
        "category_label": "Complete Models",
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
        "category": "complete",
        "category_label": "Complete Models",
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
        "category": "complete",
        "category_label": "Complete Models",
        "summary": "VzBot CNC AWD · 6 mm · MGN9H on CF Tube · Belt-Synced Dual Z",
        "highlights": [
            "VzBot CNC AWD (6061) · 48 V 2804AC",
            "6 mm Stock CoreXY · Y Rails on Top Ring",
            "Top MGN9H X on 20×20 CF · Inner MGN12H Y",
            "Belt-Synced 2× TR8×8 Z · 4× Ø10 mm Rods",
        ],
    },
    # =========================================================================
    # WIP Specific Models — Named real machines awaiting full dedicated CAD
    # =========================================================================
    {
        "id": "ratrig_vcore31_300",
        "short_name": "RatRig V-Core 3.1 (300)",
        "base_kit": "diy",
        "category": "wip_specific",
        "category_label": "WIP Specific Models",
        "summary": "3030 Frame · Top MGN12H X · EVA 3 Toolhead · 3-Point Kinematic Z",
        "wip_notes": "Has EVA 3 CAD & shaper benchmarks; currently borrows Voron stock gantry brackets until RatRig V-Core 3.1 CAD kit is baked.",
        "highlights": [
            "3030 Extrusion Frame",
            "EVA 3 Baked CAD Toolhead",
            "Top MGN12H X · 3-Pt Kinematic Bed",
            "WIP: Borrowed Gantry Brackets",
        ],
    },
    {
        "id": "monolith_zero_m0",
        "short_name": "Monolith Zero (M0)",
        "base_kit": "diy",
        "category": "wip_specific",
        "category_label": "WIP Specific Models",
        "summary": "Compact Monolith AWD · 9 mm Short-Path · CPAP Toolhead",
        "wip_notes": "Uses Monolith AWD gantry CAD; dedicated Monolith CPAP toolhead CAD & M0 compact frame brackets pending.",
        "highlights": [
            "Monolith AWD Short-Path",
            "Compact High-Accel Frame",
            "WIP: Lumped CPAP Toolhead",
        ],
    },
    {
        "id": "monolith_sheet_metal_v2",
        "short_name": "Monolith Sheet-Metal v2",
        "base_kit": "diy",
        "category": "wip_specific",
        "category_label": "WIP Specific Models",
        "summary": "Monolith AWD · Sheet-Metal Enclosure · CPAP Toolhead",
        "wip_notes": "Sheet-metal monocoque frame approximated via panels; dedicated Monolith CPAP toolhead CAD pending.",
        "highlights": [
            "Monolith AWD Gantry",
            "Sheet-Metal Enclosure Approximation",
            "WIP: Lumped CPAP Toolhead",
        ],
    },
    {
        "id": "cw_fixed_gantry_300",
        "short_name": "CW Fixed Gantry 300",
        "base_kit": "diy",
        "category": "wip_specific",
        "category_label": "WIP Specific Models",
        "summary": "High-Speed Fixed-Gantry CoreXY · Monolith AWD · 50 mm HF Hotend",
        "wip_notes": "Benchmark reference build; borrows Monolith AWD gantry CAD and uses lumped Monolith CPAP toolhead.",
        "highlights": [
            "High-Accel Fixed-Gantry CoreXY",
            "Monolith 50 mm HF Baked CAD",
            "WIP: Lumped CPAP Toolhead",
        ],
    },
    {
        "id": "doomcube_4040",
        "short_name": "DoomCube 4040 (300)",
        "base_kit": "diy",
        "category": "wip_specific",
        "category_label": "WIP Specific Models",
        "summary": "4040 Dual-Wall Thermal Frame · Monolith AWD · A4T Beacon",
        "wip_notes": "Has A4T CAD & 4040 frame mechanics; borrows Monolith AWD gantry kit until dedicated DoomCube corner/panel CAD is baked.",
        "highlights": [
            "4040 Frame + Chamber Thermal Mass",
            "A4T Beacon Baked CAD Toolhead",
            "WIP: Borrowed Gantry Brackets",
        ],
    },
    {
        "id": "flsun_v400_delta",
        "short_name": "FLSUN V400 Delta",
        "base_kit": "diy",
        "category": "wip_specific",
        "category_label": "WIP Specific Models",
        "summary": "300 mm Direct-Drive Delta · Carbon Arms · Shaper Touchstone",
        "wip_notes": "Uses procedural Delta tower/arm/effector model; dedicated FLSUN V400 effector & tower CAD pending.",
        "highlights": [
            "3-Tower Direct-Drive Delta",
            "Shaper Benchmark Touchstone",
            "WIP: Procedural Effector & Arms",
        ],
    },
    # =========================================================================
    # Generic Stand-Ins — Parametric / procedural / archetype templates
    # =========================================================================
    {
        "id": "corexy_fixed_gantry_300",
        "short_name": "Generic CoreXY Fixed Gantry 300",
        "base_kit": "generic",
        "category": "generic_standin",
        "category_label": "Generic Stand-Ins",
        "summary": "300 mm · Fixed Gantry · Triple Z · 2WD Parametric Starter",
        "highlights": [
            "Generic Fixed-Gantry CoreXY",
            "6 mm Stock CoreXY",
            "Triple TR8×4 Z · 110 mm Top",
        ],
    },
    {
        "id": "corexy_flying_gantry_300",
        "short_name": "Generic CoreXY Flying Gantry 300",
        "base_kit": "generic",
        "category": "generic_standin",
        "category_label": "Generic Stand-Ins",
        "summary": "300 mm · Flying Gantry · Quad Belted Z · 2WD Parametric Starter",
        "highlights": [
            "Generic Flying-Gantry CoreXY",
            "6 mm Stock CoreXY",
            "Quad Belted Z (6 mm)",
        ],
    },
    {
        "id": "corexy_awd_300",
        "short_name": "Generic CoreXY AWD 300",
        "base_kit": "generic",
        "category": "generic_standin",
        "category_label": "Generic Stand-Ins",
        "summary": "300 mm · Fixed Gantry · 4-Motor AWD Parametric Starter",
        "highlights": [
            "Generic AWD CoreXY",
            "4-Motor Belt Drive",
            "Parametric Toolhead Stand-In",
        ],
    },
    {
        "id": "hyper_corexy_carbon",
        "short_name": "Generic Carbon Truss CoreXY 300",
        "base_kit": "generic",
        "category": "generic_standin",
        "category_label": "Generic Stand-Ins",
        "summary": "300 mm · 3030 Truss · Carbon X-Beam · 2WD Archetype",
        "highlights": [
            "3030 Full Truss Frame",
            "Carbon X-Beam",
            "High-Stiffness CoreXY Archetype",
        ],
    },
    {
        "id": "cartesian_bedslinger",
        "short_name": "Generic Bedslinger 220",
        "base_kit": "generic",
        "category": "generic_standin",
        "category_label": "Generic Stand-Ins",
        "summary": "220 mm · Open Portal · Moving Y Bed",
        "highlights": [
            "Cartesian Bedslinger",
            "220 mm Bed",
            "Open Portal Frame",
        ],
    },
    {
        "id": "cartesian_bedslinger_braced_300",
        "short_name": "Generic Braced Bedslinger 300",
        "base_kit": "generic",
        "category": "generic_standin",
        "category_label": "Generic Stand-Ins",
        "summary": "300 mm · 3030 Braced · Dual Leadscrew Z",
        "highlights": [
            "Cartesian Bedslinger",
            "3030 Braced Frame",
            "Dual Leadscrew Z",
        ],
    },
    {
        "id": "delta_racer",
        "short_name": "Generic Carbon Delta 240",
        "base_kit": "generic",
        "category": "generic_standin",
        "category_label": "Generic Stand-Ins",
        "summary": "240 mm · 3-Tower Bowden · Carbon Arms",
        "highlights": [
            "3-Tower Delta",
            "Bowden Effector",
            "Carbon Arms",
        ],
    },
    {
        "id": "delta_direct_drive_300",
        "short_name": "Generic Direct-Drive Delta 300",
        "base_kit": "generic",
        "category": "generic_standin",
        "category_label": "Generic Stand-Ins",
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

# =============================================================================
# Component-level 3-tier maturity classification (toolheads, gantry kits, hotends)
# =============================================================================

TOOLHEAD_MODEL_STATUS: dict[str, tuple[ModelStatus, str | None]] = {
    "monolith_toolhead_cpap": (
        "wip_specific",
        "Lumped mass & stiffness; dedicated baked Tet4 CAD pending.",
    ),
    "heavy_uhf_long": (
        "generic_standin",
        "Generic heavy UHF cantilever stand-in (no dedicated CAD).",
    ),
    "custom": (
        "generic_standin",
        "Parametric user-defined toolhead stand-in.",
    ),
}

GANTRY_KIT_MODEL_STATUS: dict[str, tuple[ModelStatus, str | None]] = {
    "auto": ("generic_standin", "Auto-selects gantry kit from architecture & drive mode."),
    "procedural": ("generic_standin", "Parametric lumped gantry brackets without vendor CAD."),
}

HOTEND_MODEL_STATUS: dict[str, tuple[ModelStatus, str | None]] = {
    # All 14 catalog hotends have baked Tet4 CAD CMS + thermal specs -> default "complete".
}

PROFILE_MODEL_STATUS: dict[str, tuple[ModelStatus, str | None]] = {
    # Complete: baked 3D Tet4 CAD milled X-beams & linear rails
    "cnc_al_monocoque_20x20": ("complete", None),
    "cnc_al_truss_voron24_lw": ("complete", None),
    "cnc_al_fysetc_v0_15x15": ("complete", None),
    "mgn9_rail": ("complete", None),
    "mgn9_pocketed_rail": ("complete", None),
    "mgn12_rail": ("complete", None),
    "mgn12_hollow_rail": ("complete", None),
    "mgn12_pocketed_rail": ("complete", None),
    "mgn15_rail": ("complete", None),
    "mgn15_pocketed_rail": ("complete", None),
    "hgh15_rail": ("complete", None),
    # WIP Specific: named vendor/community X-beams calibrated via section/FEA, CAD mesh pending
    "cnc_al_beam_20x20": (
        "wip_specific",
        "Calibrated pocketed CNC Al X-beam section; dedicated vendor CAD mesh pending.",
    ),
    "cf_tube_20x20_carbonara": (
        "wip_specific",
        "Carbonara 20×20 CF layup calibrated via 3D FEA; dedicated CAD mesh pending.",
    ),
    "cf_tube_20x20_carbonara_2_5": (
        "wip_specific",
        "Theoretical 2.5 mm wall Carbonara CF X-beam variant; section-calibrated.",
    ),
    "al_tube_12x12x1": (
        "wip_specific",
        "Monolith Zero 12×12×1.0 Al X-beam section; dedicated M0 CAD pending.",
    ),
    # All standard extrusions, generic tubes, rods, flat bar, and bare rail default to generic_standin
}

Z_DRIVE_MODEL_STATUS: dict[str, tuple[ModelStatus, str | None]] = {
    # Complete: backed by baked CAD Z-mounts / Z-drives & validated hardware specs
    "tr8x4_pom": ("complete", None),
    "gt2_6mm_z_belt": ("complete", None),
    "gt2_9mm_z_belt": ("complete", None),
    # WIP Specific: named real Z hardware with calibrated analytical stiffness; dedicated CAD pending
    "tr8x4_wobblex": (
        "wip_specific",
        "WobbleX decoupler compliance calibrated analytically; dedicated CAD pending.",
    ),
    "tr8x8_pom": (
        "wip_specific",
        "High-lead TR8×8 POM nut stiffness modeled analytically; dedicated VzBot Z-drive CAD pending.",
    ),
    "tr8x8_brass": (
        "wip_specific",
        "High-lead TR8×8 brass nut stiffness modeled analytically; dedicated CAD pending.",
    ),
    "sfu1204_bk10": (
        "wip_specific",
        "SFU1204 + BK10 datasheet axial stiffness; dedicated ballscrew/pillow-block CAD pending.",
    ),
    "sfu1204_wobblex": (
        "wip_specific",
        "SFU1204 + WobbleX analytical stiffness; dedicated CAD pending.",
    ),
    "sfu1605_fk12": (
        "wip_specific",
        "SFU1605 + FK12 datasheet axial stiffness; dedicated CAD pending.",
    ),
    "kevinakasam_belted_z": (
        "wip_specific",
        "KevinAkaSam 80:16 + Keybak belted-Z analytical model; dedicated CAD pending.",
    ),
    # Generic Stand-Ins: parametric belted-Z gearbox archetypes
    "gt2_9mm_planetary_z": (
        "generic_standin",
        "Parametric 5:1 planetary belted-Z stand-in.",
    ),
    "gt2_9mm_worm_z": (
        "generic_standin",
        "Parametric self-locking worm belted-Z stand-in.",
    ),
}

Z_RAIL_MODEL_STATUS: dict[str, tuple[ModelStatus, str | None]] = {
    "mgn12_rail": ("complete", None),
    "mgn9_rail": ("complete", None),
    "steel_rod_8mm": ("generic_standin", "Parametric Ø8 mm linear rod + LM8UU stand-in."),
    "steel_rod_10mm": ("generic_standin", "Parametric Ø10 mm linear rod + LM10UU stand-in."),
    "steel_rod_12mm": ("generic_standin", "Parametric Ø12 mm linear rod + LM12UU stand-in."),
    "none": ("generic_standin", "Bare frame without Z guide rails."),
}

BED_MOUNT_MODEL_STATUS: dict[str, tuple[ModelStatus, str | None]] = {
    "trident_3pt_standoffs": ("complete", None),
    "whopping_pochard_kinematic": (
        "wip_specific",
        "Whopping Pochard leaf-spring stiffness calibrated analytically; dedicated CAD pending.",
    ),
    "cnc_kelvin_magnetic": (
        "wip_specific",
        "MRW / ChaoticLab N48SH magnetic Kelvin analytical stiffness; dedicated CAD pending.",
    ),
    "captive_belleville_kinematic": (
        "wip_specific",
        "Captive Belleville Tap-rated kinematic analytical stiffness; dedicated CAD pending.",
    ),
    "dual_leaf_ti_kinematic": (
        "wip_specific",
        "Annex / Heavy Pochard dual-blade leaf-spring analytical model; dedicated CAD pending.",
    ),
    "spherical_3pt_ratrig": (
        "wip_specific",
        "RatRig V-Core 3/4 3-pt spherical pivot analytical model; dedicated CAD pending.",
    ),
    "craz_kinematic_ge5c": (
        "wip_specific",
        "Monolith Cra-Z 2+2+2 GE5C analytical model; dedicated CAD pending.",
    ),
    "silicone_4pt_clamped": (
        "wip_specific",
        "VzBot torqued through-bolt silicone 4-pt analytical model; dedicated CAD pending.",
    ),
    # Generic 3-pt / 4-pt mounting archetypes default to generic_standin
}

Z_PROBE_MODEL_STATUS: dict[str, tuple[ModelStatus, str | None]] = {
    "static_eddy": ("complete", None),
    "printed_tap": ("complete", None),
    "cnc_tap": (
        "wip_specific",
        "ChaoticLab / CNC Al Tap analytical stiffness scaling; dedicated CNC Tap CAD pending.",
    ),
    "klicky_inductive": (
        "wip_specific",
        "Klicky / Omron dock lumped mass & rigid mount interface.",
    ),
}


def annotate_toolhead_presets(
    toolhead_presets: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return toolhead preset dicts annotated with 3-tier ``model_status``."""
    out: list[dict[str, Any]] = []
    for pid, entry in toolhead_presets.items():
        status, wip_notes = TOOLHEAD_MODEL_STATUS.get(pid, ("complete", None))
        item = dict(entry)
        item["model_status"] = status
        item["model_status_label"] = MODEL_STATUS_LABELS[status]
        if wip_notes:
            item["wip_notes"] = wip_notes
        out.append(item)
    return out


def annotate_gantry_hardware_kits(
    gantry_kits: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return gantry hardware kit dicts annotated with 3-tier ``model_status``."""
    out: list[dict[str, Any]] = []
    for kid, entry in gantry_kits.items():
        status, wip_notes = GANTRY_KIT_MODEL_STATUS.get(kid, ("complete", None))
        item = dict(entry)
        item["model_status"] = status
        item["model_status_label"] = MODEL_STATUS_LABELS[status]
        if wip_notes:
            item["wip_notes"] = wip_notes
        out.append(item)
    return out


def annotate_hotend_presets(
    hotend_catalog: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return hotend preset dicts annotated with 3-tier ``model_status``."""
    out: list[dict[str, Any]] = []
    for hid, entry in hotend_catalog.items():
        status, wip_notes = HOTEND_MODEL_STATUS.get(hid, ("complete", None))
        item = dict(entry)
        item["model_status"] = status
        item["model_status_label"] = MODEL_STATUS_LABELS[status]
        if wip_notes:
            item["wip_notes"] = wip_notes
        out.append(item)
    return out


def annotate_profiles(
    profile_catalog: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return serialized profile dicts annotated with 3-tier ``model_status``."""
    out: list[dict[str, Any]] = []
    for pid, prof in profile_catalog.items():
        status, wip_notes = PROFILE_MODEL_STATUS.get(pid, ("generic_standin", None))
        item = prof.model_dump(mode="json") if hasattr(prof, "model_dump") else dict(prof)
        item["model_status"] = status
        item["model_status_label"] = MODEL_STATUS_LABELS[status]
        if wip_notes:
            item["wip_notes"] = wip_notes
        out.append(item)
    return out


def annotate_z_drive_mechanisms(
    z_drive_catalog: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return Z-drive mechanism dicts annotated with 3-tier ``model_status``."""
    out: list[dict[str, Any]] = []
    for zid, entry in z_drive_catalog.items():
        status, wip_notes = Z_DRIVE_MODEL_STATUS.get(zid, ("generic_standin", None))
        item = dict(entry)
        item["model_status"] = status
        item["model_status_label"] = MODEL_STATUS_LABELS[status]
        if wip_notes:
            item["wip_notes"] = wip_notes
        out.append(item)
    return out


def annotate_z_rail_profiles(
    z_rail_catalog: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return Z-rail profile dicts annotated with 3-tier ``model_status``."""
    out: list[dict[str, Any]] = []
    for rid, entry in z_rail_catalog.items():
        status, wip_notes = Z_RAIL_MODEL_STATUS.get(rid, ("generic_standin", None))
        item = dict(entry)
        item["model_status"] = status
        item["model_status_label"] = MODEL_STATUS_LABELS[status]
        if wip_notes:
            item["wip_notes"] = wip_notes
        out.append(item)
    return out


def annotate_bed_mounts(
    bed_mount_catalog: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return bed mount dicts annotated with 3-tier ``model_status``."""
    out: list[dict[str, Any]] = []
    for mid, entry in bed_mount_catalog.items():
        status, wip_notes = BED_MOUNT_MODEL_STATUS.get(mid, ("generic_standin", None))
        item = dict(entry)
        item["model_status"] = status
        item["model_status_label"] = MODEL_STATUS_LABELS[status]
        if wip_notes:
            item["wip_notes"] = wip_notes
        out.append(item)
    return out


def annotate_z_probes(
    z_probe_catalog: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return Z-probe dicts annotated with 3-tier ``model_status``."""
    out: list[dict[str, Any]] = []
    for pid, entry in z_probe_catalog.items():
        status, wip_notes = Z_PROBE_MODEL_STATUS.get(pid, ("generic_standin", None))
        item = dict(entry)
        item["model_status"] = status
        item["model_status_label"] = MODEL_STATUS_LABELS[status]
        if wip_notes:
            item["wip_notes"] = wip_notes
        out.append(item)
    return out


def find_examples_dir() -> Path:
    """Locate the ``examples/`` directory in repo or Pyodide WASM virtual FS."""
    here = Path(__file__).resolve()
    for parent in (here.parents[2], here.parents[1], Path("/printsim")):
        cand = parent / "examples"
        if cand.is_dir():
            return cand
    return here.parents[2] / "examples"


def is_studio_default_preset(preset_id: str) -> bool:
    """Return True if ``preset_id`` is registered in Studio preset metadata."""
    return preset_id in _STUDIO_META_BY_ID


def get_preset_model_status(preset_id: str) -> ModelStatus:
    """Return the 3-tier maturity status for ``preset_id``."""
    meta = _STUDIO_META_BY_ID.get(preset_id)
    if meta and meta.get("category") in MODEL_STATUS_LABELS:
        return meta["category"]
    return "wip_specific"


def load_presets(*, include_test_builds: bool = False) -> dict[str, PrinterSpec]:
    """Load printer presets from ``examples/``.

    Args:
        include_test_builds: When ``False`` (default, used for ``/api/catalog``
            and GitHub Pages), returns only the curated Studio presets in display
            order. When ``True`` (used for direct ``/api/presets/{id}`` lookups
            and test suites), loads all YAML presets in ``examples/``.
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

    # Load curated Studio presets first, then any remaining YAML files sorted by stem
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
    """Return the 3-tier categorized presets for ``/api/catalog``."""
    presets = load_presets(include_test_builds=False)
    items: list[dict[str, Any]] = []
    for pid, spec in presets.items():
        meta = _STUDIO_META_BY_ID.get(pid, {})
        arch = spec.architecture
        status = meta.get("category", "generic_standin")
        item: dict[str, Any] = {
            "id": pid,
            "name": spec.name,
            "short_name": meta.get("short_name", spec.name),
            "architecture": arch,
            "base_kit": meta.get("base_kit", "generic"),
            "category": status,
            "category_label": meta.get(
                "category_label", PRINTER_TYPE_LABELS.get(status, status)
            ),
            "model_status": status,
            "model_status_label": MODEL_STATUS_LABELS.get(status, status),
            "hardware_kit": spec.resolve_effective_gantry_hardware_kit(),
            "summary": meta.get("summary", ""),
            "highlights": list(meta.get("highlights", [])),
            "description": spec.description,
        }
        if meta.get("wip_notes"):
            item["wip_notes"] = meta["wip_notes"]
        items.append(item)
    return items
