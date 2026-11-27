"""Bridge to PrintDB for hardware component specs, material properties, and physics constants.

PrintDB (https://github.com/Jumpybeetroot/PrintDB) is an optional dependency
providing version-controlled hardware datasheets, material properties, structural
profiles, Z-drives, bed mounts, assemblies, and physics reference constants with
lossless SI conversion (``.to_si()``).

When ``printdb`` is installed and its repository root is accessible, PrintSim
hydrates all matching catalog entries and reference physics constants from PrintDB
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
    """Overlay PrintDB motor datasheet specs onto all matching ``STEPPER_CATALOG`` entries."""
    mapping: dict[str, str] = {
        "excit3d_max": "components/motors/excit3d-max-ol",
        "isik_ouroboros": "components/motors/isik-tech-ouroboros-encstp",
        "ldo_2504ac": "components/motors/ldo-42sth48-2504ac",
        "ldo_2504ah": "components/motors/ldo-42sth48-2504ah",
        "ldo_2504ah_s45r": "components/motors/ldo-42sth48-2504ah-s45r",
        "ldo_2504ah_s80": "components/motors/ldo-42sth48-2504ah-s80",
        "ldo_2804ah": "components/motors/ldo-42sth48-2804ah",
        "ldo_2804ac": "components/motors/ldo-42sth48-2804ac",
        "kraken_motor_v2": "components/motors/kraken-motor-v2-60mm",
        "ldo_60_3004": "components/motors/ldo-42sth60-3004ac",
        "ldo_60_2004ah": "components/motors/ldo-42sth60-2004ah",
        "ldo_09deg_2504": "components/motors/ldo-42sth48-2504mac",
        "ldo_09deg_2004mah": "components/motors/ldo-42sth48-2004mah-vrn",
        "moons_09deg_ha2p4200": "components/motors/moons-ms17ha2p4200",
        "leadshine_42cm06": "components/motors/leadshine-42cm06",
        "moons_420i": "components/motors/moons-ms17hd6p420i-04",
        "stepperonline_2504s": "components/motors/stepperonline-17hs19-2504s",
        "stepperonline_2004": "components/motors/stepperonline-17hs19-2004s1",
        "stepperonline_09deg_2004": "components/motors/stepperonline-17hm19-2004s",
        "ldo_40_1684ac": "components/motors/ldo-42sth40-1684ac",
        "ldo_57_2804mac": "components/motors/ldo-57sth41-2804mac",
        "stepperonline_23hs22_2804": "components/motors/stepperonline-23hs22-2804s",
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
    """Overlay PrintDB belt datasheet geometry onto all matching ``BELT_CATALOG`` entries."""
    mapping: dict[str, str] = {
        "gt2_6mm": "components/belts/gates-2gt-6mm-ll",
        "gt2_9mm": "components/belts/gates-2gt-9mm-ll",
        "gt2_10mm": "components/belts/gates-2gt-10mm-ll",
        "gt3_10mm": "components/belts/gates-gt3-2mgt-10mm",
        "gt2_12mm": "components/belts/gates-2gt-12mm-ll",
        "gt2_15mm": "components/belts/gates-2gt-15mm-ll",
        "gt2_dual_9mm": "components/belts/gates-2gt-dual-9mm",
        "3gt_9mm": "components/belts/gates-3gt-9mm",
        "3gt_12mm": "components/belts/gates-3gt-12mm",
        "3gt_15mm": "components/belts/gates-3gt-15mm",
        "at3_10mm": "components/belts/elatech-at3-10mm",
        "at3_16mm": "components/belts/elatech-at3-16mm",
        "at3_gen3_16mm": "components/belts/contitech-at3-gen3-16mm",
        "htd3m_15mm": "components/belts/elatech-htd3m-15mm",
        "at5_16mm": "components/belts/elatech-at5-16mm",
    }
    for key, rid in mapping.items():
        if key not in catalog:
            continue
        raw = get_record_specs(rid, si=False)
        if not raw:
            continue
        entry = catalog[key]
        if raw.get("pitch_mm") is not None:
            entry["pitch_mm"] = float(raw["pitch_mm"])
        if raw.get("width_mm") is not None:
            entry["width_mm"] = float(raw["width_mm"])
        # Note: keep PrintSim's calibrated linear_mass_kg_m on 6/9 mm 2GT to preserve
        # Voron pluck-Hz calibration.
        entry["printdb_id"] = rid


def hydrate_hotend_catalog(catalog: dict[str, dict[str, Any]]) -> None:
    """Overlay PrintDB hotend datasheet/CAD specs onto all matching ``HOTEND_CATALOG`` entries."""
    mapping: dict[str, str] = {
        "e3d_v6_groove": "components/hotends/e3d-v6-groove",
        "e3d_revo_voron": "components/hotends/e3d-revo-voron",
        "phaetus_dragon_hf": "components/hotends/phaetus-dragon-hf",
        "phaetus_dragon_uhf": "components/hotends/phaetus-dragon-uhf",
        "phaetus_rapido_2_hf": "components/hotends/phaetus-rapido-2-hf",
        "phaetus_rapido_2_uhf": "components/hotends/phaetus-rapido-2-uhf",
        "mellow_goliath_air": "components/hotends/mellow-goliath-air",
        "chube_compact": "components/hotends/lukes-lab-chube-compact",
        "chube_air": "components/hotends/lukes-lab-chube-air",
        "chube_conduction": "components/hotends/lukes-lab-chube-conduction",
        "tricorn_short": "components/hotends/tricorn-short",
        "tricorn_long": "components/hotends/tricorn-long",
        "monolith_50mm_hf": "components/hotends/monolith-50mm-hf-hotend",
        "trianglelab_dragon_ace_volcano": "components/hotends/trianglelab-dragon-ace-volcano",
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
        if raw.get("overall_length_mm") is not None:
            entry["overall_length_mm"] = float(raw["overall_length_mm"])
        if raw.get("meltzone_length_mm") is not None:
            entry["melt_zone_length_mm"] = float(raw["meltzone_length_mm"])
        if raw.get("tip_stiffness_n_per_mm") is not None:
            entry["structural_stiffness_n_per_mm"] = float(raw["tip_stiffness_n_per_mm"])
        # Note: keep PrintSim's dressed mass_g on e3d_revo_voron (30 g w/ heater/sock vs 25 g bare).
        if key != "e3d_revo_voron" and raw.get("mass_g") is not None:
            entry["mass_g"] = float(raw["mass_g"])
        entry["printdb_id"] = rid


def hydrate_rail_catalog(
    carriage_catalog: dict[str, tuple[float, float, int, float, float, float, float, float]],
    block_mass_g: dict[str, tuple[float, float]],
) -> None:
    """Overlay PrintDB linear rail specs onto ``_RAIL_CARRIAGE_CATALOG`` & ``_RAIL_BLOCK_MASS_G``."""
    # Map each rail profile key -> (short_c_id, long_h_id)
    mapping: dict[str, tuple[str | None, str | None]] = {
        "mgn9_rail": ("components/linear_rails/hiwin-mgn9c", "components/linear_rails/hiwin-mgn9h"),
        "mgn12_rail": ("components/linear_rails/hiwin-mgn12c", "components/linear_rails/hiwin-mgn12h"),
        "mgn15_rail": ("components/linear_rails/hiwin-mgn15c", "components/linear_rails/hiwin-mgn15h"),
        "hgh15_rail": ("components/linear_rails/hiwin-hgh15ca", None),
        "steel_rod_8mm": ("components/linear_rails/misumi-lm8uu-8mm-rod", None),
        "steel_rod_10mm": ("components/linear_rails/misumi-lm10uu-10mm-rod", None),
        "steel_rod_12mm": ("components/linear_rails/misumi-lm12uu-12mm-rod", None),
    }
    for key, (cid_c, cid_h) in mapping.items():
        raw_c = get_record_specs(cid_c, si=False) if cid_c else None
        raw_h = get_record_specs(cid_h, si=False) if cid_h else None
        if not raw_c and not raw_h:
            continue
        if key in block_mass_g:
            c_mass, h_mass = block_mass_g[key]
            if raw_c and raw_c.get("carriage_mass_g") is not None:
                c_mass = float(raw_c["carriage_mass_g"])
            if raw_h and raw_h.get("carriage_mass_g") is not None:
                h_mass = float(raw_h["carriage_mass_g"])
            block_mass_g[key] = (c_mass, h_mass)
        if key in carriage_catalog:
            d_b, r_y, n_rows, l1_c, c_c, l1_h, c_h, seal_drag = carriage_catalog[key]
            if raw_c and raw_c.get("dynamic_load_n") is not None:
                c_c = float(raw_c["dynamic_load_n"])
            if raw_h and raw_h.get("dynamic_load_n") is not None:
                c_h = float(raw_h["dynamic_load_n"])
            carriage_catalog[key] = (d_b, r_y, n_rows, l1_c, c_c, l1_h, c_h, seal_drag)


def hydrate_material_catalog(catalog: dict[str, Any]) -> None:
    """Overlay PrintDB structural material SI properties onto all matching ``MATERIAL_CATALOG`` entries."""
    mapping: dict[str, str] = {
        "aluminum_6063_t5": "materials/aluminum-6063-t5",
        "aluminum_6061_t6": "materials/aluminum-6061-t6",
        "aluminum_7075_t6": "materials/aluminum-7075-t6",
        "mic6_aluminum": "materials/alcoa-mic6-cast-aluminum",
        "atp5_aluminum": "materials/vista-atp5-cast-aluminum",
        "carbon_fiber_3k_twill": "materials/carbon-fiber-3k-twill-tube",
        "carbon_fiber_tube": "materials/carbon-fiber-ud-std-epoxy",
        "carbon_fiber_high_tg": "materials/carbon-fiber-ud-high-tg",
        "carbon_fiber_torsional_45": "materials/carbon-fiber-torsional-45-high-tg",
        "carbon_fiber_m40j_hm": "materials/carbon-fiber-m40j-high-modulus",
        "steel_4140": "materials/steel-4140",
        "stainless_304": "materials/stainless-steel-304",
        "titanium_grade5": "materials/titanium-grade5-ti6al4v",
        "brass_c360": "materials/brass-c360",
        "copper_c110": "materials/copper-c110",
        "hardened_steel": "materials/hardened-tool-steel",
        "abs_printed": "materials/generic-abs",
        "pa12_cf_printed": "materials/generic-pa12-cf15",
        "pet_cf_printed": "materials/generic-pet-cf15-annealed",
        "ppa_cf_printed": "materials/generic-ppa-cf-annealed",
        "pps_cf_printed": "materials/generic-pps-cf-annealed",
        "pc_pbt_gf_printed": "materials/treed-pc-pbt-gf",
        "slm_aluminum": "materials/slm-alsi10mg-aluminum",
        "slm_ti6al4v": "materials/slm-ti6al4v-titanium",
        "slm_316l_steel": "materials/slm-316l-stainless",
        "slm_17_4ph_steel": "materials/slm-17-4ph-stainless",
        "slm_maraging_steel": "materials/slm-maraging-steel-ms1",
        "slm_cucrzr": "materials/slm-cucrzr-copper",
        "acrylic_pmma": "materials/acrylic-pmma-sheet",
        "polycarbonate": "materials/polycarbonate-sheet",
        "acm_dibond": "materials/acm-dibond-panel",
        "aluminum_5052_sheet": "materials/aluminum-5052-h32",
        "carbon_fiber_plate": "materials/carbon-fiber-3k-plate",
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
        if si.get("thermal_conductivity_w_per_mk") is not None and key != "abs_printed":
            mat.thermal_conductivity_w_mk = float(si["thermal_conductivity_w_per_mk"])
        if si.get("specific_heat_j_per_kgk") is not None:
            mat.specific_heat_j_kgk = float(si["specific_heat_j_per_kgk"])
        if si.get("cte_per_k") is not None and key != "abs_printed":
            mat.thermal_expansion_1_k = float(si["cte_per_k"])
        if getattr(mat, "is_slm_metal", False):
            if si.get("shear_modulus_pa") is not None:
                mat.shear_modulus_pa = float(si["shear_modulus_pa"])
            if si.get("damping_ratio") is not None:
                mat.damping_ratio = float(si["damping_ratio"])


def hydrate_filament_catalog(catalog: dict[str, Any]) -> None:
    """Overlay PrintDB polymer density onto all matching ``FILAMENT_CATALOG`` entries."""
    mapping: dict[str, str] = {
        "pla": "materials/generic-pla",
        "petg": "materials/generic-petg",
        "abs": "materials/generic-abs",
        "asa": "materials/generic-asa",
        "pa12_cf": "materials/generic-pa12-cf15",
        "tpu_95a": "materials/generic-tpu-95a",
    }
    for key, rid in mapping.items():
        if key not in catalog:
            continue
        si = get_record_specs(rid, si=True)
        if not si:
            continue
        fil = catalog[key]
        if si.get("density_kg_per_m3") is not None:
            fil.density_kg_m3 = float(si["density_kg_per_m3"])
        # Note: keep PrintSim's calibrated melt-zone thermal_conductivity_w_mk and Tg
        # (e.g. PLA Tg=58 C, ABS/ASA k=0.18 W/mK) used by Graetz flow benchmarks.


def hydrate_profile_catalog(catalog: dict[str, Any]) -> None:
    """Overlay PrintDB cross-section SI geometry onto all matching ``PROFILE_CATALOG`` entries."""
    mapping: dict[str, str] = {
        "extrusion_1515": "components/profiles/makerbeam-1515-extrusion",
        "extrusion_2020": "components/profiles/misumi-hfs5-2020",
        "extrusion_2040": "components/profiles/misumi-hfs5-2040",
        "extrusion_3030": "components/profiles/misumi-hfs6-3030",
        "extrusion_4040": "components/profiles/misumi-hfs8-4040",
        "extrusion_3060": "components/profiles/misumi-hfs6-3060",
        "cnc_al_beam_20x20": "components/profiles/cnc-al-beam-20x20",
        "cnc_al_monocoque_20x20": "components/profiles/cnc-al-monocoque-20x20",
        "cnc_al_truss_voron24_lw": "components/profiles/cnc-al-truss-voron24-lw",
        "cnc_al_fysetc_v0_15x15": "components/profiles/cnc-al-fysetc-v0-15x15",
        "cf_tube_20x20x1": "components/profiles/cf-tube-20x20x1",
        "cf_tube_20x20x1_5": "components/profiles/cf-tube-20x20x1-5",
        "cf_tube_20x20_carbonara": "components/profiles/carbonara-20x20-cf-xbeam",
        "cf_tube_20x20_carbonara_2_5": "components/profiles/carbonara-20x20x2-5-cf-xbeam",
        "cf_tube_20x20x2": "components/profiles/cf-tube-20x20x2",
        "cf_tube_30x20x2": "components/profiles/cf-tube-30x20x2",
        "cf_tube_25x25x2": "components/profiles/cf-tube-25x25x2",
        "ti_tube_20x20x1_2": "components/profiles/ti-tube-20x20x1-2",
        "ti_tube_20x20x2": "components/profiles/ti-tube-20x20x2",
        "al_tube_12x12x1": "components/profiles/al-tube-12x12x1",
        "al_tube_20x20x2": "components/profiles/al-tube-20x20x2",
        "al_tube_20x20x3": "components/profiles/al-tube-20x20x3",
        "al_tube_30x30x3": "components/profiles/al-tube-30x30x3",
        "steel_tube_25x25x2": "components/profiles/steel-tube-25x25x2",
        "mgn9_rail": "components/profiles/profile-mgn9-rail",
        "mgn12_rail": "components/profiles/profile-mgn12-rail",
        "mgn12_hollow_rail": "components/profiles/profile-hollow-mgn12-rail",
        "mgn15_rail": "components/profiles/profile-mgn15-rail",
        "steel_rod_8mm": "components/profiles/profile-steel-rod-8mm",
        "steel_rod_10mm": "components/profiles/profile-steel-rod-10mm",
        "steel_rod_12mm": "components/profiles/profile-steel-rod-12mm",
        "brace_rod_10mm": "components/profiles/profile-brace-rod-10mm",
        "al_flat_bar_25x5": "components/profiles/al-flat-bar-25x5",
    }
    for key, rid in mapping.items():
        if key not in catalog:
            continue
        si = get_record_specs(rid, si=True)
        if not si:
            continue
        prof = catalog[key]
        if si.get("area_m2") is not None:
            prof.area_m2 = float(si["area_m2"])
        if si.get("iy_m4") is not None:
            prof.iy_m4 = float(si["iy_m4"])
        if si.get("iz_m4") is not None:
            prof.iz_m4 = float(si["iz_m4"])
        if si.get("j_m4") is not None:
            prof.j_m4 = float(si["j_m4"])


def hydrate_z_drive_catalog(catalog: dict[str, dict[str, Any]]) -> None:
    """Attach canonical PrintDB IDs and specs onto ``Z_DRIVE_MECHANISM_CATALOG``."""
    mapping: dict[str, str] = {
        "tr8x4_pom": "components/z_drives/generic-tr8x4-pom",
        "tr8x4_wobblex": "components/z_drives/miragec-tr8x4-wobblex",
        "tr8x8_pom": "components/z_drives/generic-tr8x8-pom",
        "tr8x8_brass": "components/z_drives/generic-tr8x8-brass",
        "sfu1204_bk10": "components/z_drives/hiwin-sfu1204-bk10",
        "sfu1204_wobblex": "components/z_drives/hiwin-sfu1204-bk10-wobblex",
        "sfu1605_fk12": "components/z_drives/hiwin-sfu1605-fk12",
        "gt2_6mm_z_belt": "components/z_drives/voron-gt2-6mm-belted-z",
        "gt2_9mm_z_belt": "components/z_drives/voron-gt2-9mm-belted-z",
        "gt2_9mm_planetary_z": "components/z_drives/generic-gt2-9mm-planetary-z",
        "gt2_9mm_worm_z": "components/z_drives/generic-gt2-9mm-worm-z",
        "kevinakasam_belted_z": "components/z_drives/kevinakasam-belted-z",
    }
    for key, rid in mapping.items():
        if key not in catalog:
            continue
        raw = get_record_specs(rid, si=False)
        if not raw:
            continue
        entry = catalog[key]
        if raw.get("lead_mm") is not None:
            entry["lead_mm"] = float(raw["lead_mm"])
        if raw.get("backlash_um") is not None:
            entry["backlash_um"] = float(raw["backlash_um"])
        entry["printdb_id"] = rid


def hydrate_bed_mount_catalog(catalog: dict[str, dict[str, Any]]) -> None:
    """Attach canonical PrintDB IDs and specs onto ``BED_MOUNT_CATALOG``."""
    mapping: dict[str, str] = {
        "kinematic_3pt": "components/bed_mounts/generic-kinematic-3pt-maxwell",
        "whopping_pochard_kinematic": "components/bed_mounts/whopping-pochard-leaf-kinematic",
        "trident_3pt_standoffs": "components/bed_mounts/voron-trident-3pt-bakelite-ge5c",
        "cnc_kelvin_magnetic": "components/bed_mounts/chaoticlab-cnc-kelvin-magnetic",
        "captive_belleville_kinematic": "components/bed_mounts/generic-captive-belleville-kinematic",
        "dual_leaf_ti_kinematic": "components/bed_mounts/annex-dual-leaf-ti-kinematic",
        "spherical_3pt_ratrig": "components/bed_mounts/ratrig-spherical-3pt-kinematic",
        "craz_kinematic_ge5c": "components/bed_mounts/monolith-craz-kinematic-ge5c",
        "rigid_4pt": "components/bed_mounts/generic-rigid-4pt-standoffs",
        "floating_4pt": "components/bed_mounts/generic-floating-4pt-tabs",
        "silicone_4pt": "components/bed_mounts/generic-silicone-4pt-70a",
        "silicone_4pt_clamped": "components/bed_mounts/vzbot-silicone-4pt-m3-clamped",
        "springs_4pt": "components/bed_mounts/generic-springs-4pt-yellow",
    }
    for key, rid in mapping.items():
        if key not in catalog:
            continue
        raw = get_record_specs(rid, si=False)
        if not raw:
            continue
        entry = catalog[key]
        if raw.get("axial_stiffness_n_per_mm") is not None:
            entry["k_axial_n_per_mm"] = float(raw["axial_stiffness_n_per_mm"])
        if raw.get("lateral_stiffness_n_per_mm") is not None:
            entry["k_lateral_n_per_mm"] = float(raw["lateral_stiffness_n_per_mm"])
        if raw.get("standoff_height_mm") is not None:
            entry["standoff_height_mm"] = float(raw["standoff_height_mm"])
        entry["printdb_id"] = rid
