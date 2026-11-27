"""Runtime hardware superelement registry & cached loader (strict SI internals)."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from printsim.schema import get_material

ASSETS_ROOT: Path = Path(__file__).resolve().parent.parent / "assets" / "hardware"

_DEFAULT_CATEGORIES: tuple[str, ...] = (
    "hotends",
    "toolheads",
    "assemblies",
    "gantry_brackets",
    "frame_braces",
    "x_beams",
)


@dataclass
class LoadedHardwareAsset:
    """In-memory Craig-Bampton superelement bundle for a CAD-backed hardware component."""

    category: str
    asset_id: str
    manifest: dict[str, Any]
    K_cb: np.ndarray
    M_cb: np.ndarray
    K_bb: np.ndarray
    port_coords_m: np.ndarray
    fixed_interface_freqs_hz: list[float]
    com_m: np.ndarray
    inertia_kg_m2: np.ndarray
    visual_glb_path: Path | None
    stiffness_scale_applied: float = 1.0
    mass_scale_applied: float = 1.0


@dataclass(frozen=True)
class _RawCachedAsset:
    """Immutable on-disk bundle cached in memory for <0.05 ms instantiation."""

    manifest: dict[str, Any]
    K_cb: np.ndarray
    M_cb: np.ndarray
    K_bb: np.ndarray
    port_coords_m: np.ndarray
    fixed_interface_freqs_hz: tuple[float, ...]
    com_m: np.ndarray
    inertia_kg_m2: np.ndarray
    visual_glb_path: Path | None


_ASSET_CACHE: dict[tuple[str, str], _RawCachedAsset] = {}
_VISUAL_MANIFEST_CACHE: dict[tuple[str, str], tuple[dict[str, Any], Path]] = {}

# PrintDB-calibrated printed-polymer gantry-bracket creep, HDT & moisture knockdown defaults.
# Hydrated lazily from PrintDB records (``materials/generic-abs``, ``materials/bambu-pet-cf``,
# ``materials/generic-pa12-cf15``, ``materials/treed-pc-pbt-gf``, ``materials/bambu-ppa-cf``,
# ``materials/bambu-pps-cf``) with WASM-safe built-in fallbacks.
_POLYMER_PRINTDB_IDS: dict[str, str] = {
    "abs_printed": "materials/generic-abs",
    "pet_cf_printed": "materials/bambu-pet-cf",
    "pa12_cf_printed": "materials/generic-pa12-cf15",
    "pc_pbt_gf_printed": "materials/treed-pc-pbt-gf",
    "ppa_cf_printed": "materials/bambu-ppa-cf",
    "pps_cf_printed": "materials/bambu-pps-cf",
}

_POLYMER_CREEP_FALLBACKS: dict[str, dict[str, float]] = {
    "abs_printed": {
        "hdt_c": 95.0,
        "tg_c": 105.0,
        "creep_limit_n": 45.0,
        "water_absorption_pct": 0.25,
        "infill_shell_factor": 0.78,
    },
    "pet_cf_printed": {
        "hdt_c": 182.0,
        "tg_c": 75.0,
        "creep_limit_n": 85.0,
        "water_absorption_pct": 0.37,
        "infill_shell_factor": 0.78,
    },
    "pa12_cf_printed": {
        "hdt_c": 135.0,
        "tg_c": 56.0,
        "creep_limit_n": 65.0,
        "water_absorption_pct": 1.60,
        "infill_shell_factor": 0.78,
    },
    "pc_pbt_gf_printed": {
        "hdt_c": 120.0,
        "tg_c": 138.0,
        "creep_limit_n": 72.0,
        "water_absorption_pct": 0.30,
        "infill_shell_factor": 0.78,
    },
    "ppa_cf_printed": {
        "hdt_c": 196.0,
        "tg_c": 85.0,
        "creep_limit_n": 120.0,
        "water_absorption_pct": 0.65,  # annealed conditioned effective uptake (1.30% raw TDS)
        "infill_shell_factor": 0.78,
    },
    "pps_cf_printed": {
        "hdt_c": 235.0,
        "tg_c": 100.0,
        "creep_limit_n": 140.0,
        "water_absorption_pct": 0.05,
        "infill_shell_factor": 0.78,
    },
}

_POLYMER_CREEP_CACHE: dict[str, dict[str, float]] = {}


def _get_polymer_creep_params(material_id: str) -> dict[str, float]:
    """Return PrintDB-hydrated polymer HDT/creep/moisture calibration parameters (WASM-safe)."""
    cached = _POLYMER_CREEP_CACHE.get(material_id)
    if cached is not None:
        return cached

    base = dict(
        _POLYMER_CREEP_FALLBACKS.get(
            material_id,
            {
                "hdt_c": 110.0,
                "tg_c": 90.0,
                "creep_limit_n": 60.0,
                "water_absorption_pct": 0.50,
                "infill_shell_factor": 0.78,
            },
        )
    )
    rid = _POLYMER_PRINTDB_IDS.get(material_id)
    if rid is not None:
        try:
            from printsim.schema.printdb_bridge import get_record_specs

            raw = get_record_specs(rid, si=False)
            if raw:
                if raw.get("hd_temp_c") is not None:
                    base["hdt_c"] = float(raw["hd_temp_c"])
                if raw.get("glass_transition_c") is not None:
                    base["tg_c"] = float(raw["glass_transition_c"])
                if raw.get("creep_limit_n") is not None:
                    base["creep_limit_n"] = float(raw["creep_limit_n"])
                if raw.get("water_absorption_pct") is not None:
                    # Annealed PPA-CF halves equilibrium moisture plasticization vs raw unannealed TDS
                    wa = float(raw["water_absorption_pct"])
                    base["water_absorption_pct"] = 0.5 * wa if material_id == "ppa_cf_printed" else wa
        except Exception:
            pass

    _POLYMER_CREEP_CACHE[material_id] = base
    return base


def _orthotropic_effective_ratio(material_id: str) -> float:
    """Mean 3D structural modulus factor from FDM orthotropic ratios (1, e2_ratio, e3_ratio)."""
    mat = get_material(material_id)
    return float((1.0 + 0.65 * float(mat.e2_ratio) + 0.35 * float(mat.e3_ratio)) / 2.0)


def gantry_bracket_polymer_knockdown(
    material_id: str,
    chamber_temp_c: float = 25.0,
    *,
    reference_is_fdm: bool = False,
    reference_material_id: str | None = None,
    belt_tension_n: float | None = None,
) -> dict[str, float]:
    """Compute PrintDB-calibrated FDM anisotropy, moisture & HDT/creep stiffness knockdown for printed brackets.

    Calibrated against PrintDB polymer TDS & community creep records (Bambu PPA-CF/PPS-CF/PET-CF,
    Polymaker PA12-CF15, TreeD PC-PBT-GF, generic ABS). Returns exact factor decomposition and
    temperature-derated creep tension limit (N). For non-printed metals, returns neutral ``1.0`` factors.
    """
    mat = get_material(material_id)
    if not bool(mat.is_printed_polymer):
        return {
            "fdm_anisotropy_infill_factor": 1.0,
            "moisture_conditioning_factor": 1.0,
            "hdt_creep_factor": 1.0,
            "tension_creep_factor": 1.0,
            "total_stiffness_factor": 1.0,
            "hdt_c": float(mat.hdt_c or 300.0),
            "base_creep_limit_n": 250.0,
            "effective_creep_limit_n": 250.0,
        }

    params = _get_polymer_creep_params(material_id)
    aniso_cur = _orthotropic_effective_ratio(material_id)
    if reference_is_fdm and reference_material_id:
        aniso_ref = max(_orthotropic_effective_ratio(reference_material_id), 0.25)
        fdm_factor = float(np.clip(aniso_cur / aniso_ref, 0.55, 1.35))
    else:
        fdm_factor = float(params["infill_shell_factor"]) * aniso_cur

    # Conditioned ambient moisture plasticization (strongest for hygroscopic PA12-CF, ~0 for PPS-CF)
    water_pct = max(0.0, float(params["water_absorption_pct"]))
    moisture_factor = 1.0 / (1.0 + 0.11 * water_pct)

    # Sustained-load thermal creep secant-modulus knockdown vs PrintDB HDT @ 1.8 MPa
    hdt_c = max(float(params["hdt_c"]), 55.0)
    tau = max(0.0, (float(chamber_temp_c) - 25.0) / max(hdt_c - 25.0, 20.0))
    hdt_creep_factor = 1.0 / (1.0 + 0.18 * (tau**2.2))

    # Optional sustained belt-tension creep knockdown above 50% of PrintDB creep tension ceiling
    base_creep_n = max(float(params["creep_limit_n"]), 20.0)
    eff_creep_n = base_creep_n * hdt_creep_factor * moisture_factor
    if belt_tension_n is not None and float(belt_tension_n) > 0.0:
        load_ratio = float(belt_tension_n) / max(eff_creep_n, 1.0)
        tension_creep_factor = 1.0 / (1.0 + 0.22 * (max(load_ratio - 0.5, 0.0) ** 2.0))
    else:
        tension_creep_factor = 1.0

    total = float(np.clip(fdm_factor * moisture_factor * hdt_creep_factor * tension_creep_factor, 0.25, 1.10))
    return {
        "fdm_anisotropy_infill_factor": round(fdm_factor, 4),
        "moisture_conditioning_factor": round(moisture_factor, 4),
        "hdt_creep_factor": round(hdt_creep_factor, 4),
        "tension_creep_factor": round(tension_creep_factor, 4),
        "total_stiffness_factor": round(total, 4),
        "hdt_c": round(hdt_c, 1),
        "base_creep_limit_n": round(base_creep_n, 1),
        "effective_creep_limit_n": round(eff_creep_n, 1),
    }


def clear_asset_cache() -> None:
    """Clear the in-memory hardware superelement & visual manifest caches (useful after re-baking)."""
    _ASSET_CACHE.clear()
    _VISUAL_MANIFEST_CACHE.clear()
    _POLYMER_CREEP_CACHE.clear()


def _asset_dir(category: str, asset_id: str | None) -> Path | None:
    if not category or not asset_id:
        return None
    cat = str(category).strip()
    aid = str(asset_id).strip()
    if not cat or not aid or "/" in cat or "/" in aid or "\\" in cat or "\\" in aid:
        return None
    if cat in (".", "..") or aid in (".", ".."):
        return None
    return ASSETS_ROOT / cat / aid


def _manifest_with_visual_url(
    manifest: dict[str, Any],
    category: str,
    asset_id: str,
    *,
    has_superelement: bool | None = None,
) -> dict[str, Any]:
    out = dict(manifest)
    out["visual_url"] = f"/api/assets/hardware/{category}/{asset_id}/visual.glb"
    if has_superelement is not None:
        out["has_superelement"] = bool(has_superelement)
    return out


def _read_visual_only_manifest(
    category: str, asset_id: str
) -> tuple[dict[str, Any], Path] | None:
    """Read a lightweight visual-only bundle (``manifest.json`` + ``visual.glb`` without ``superelement.npz``)."""
    key = (category, asset_id)
    cached = _VISUAL_MANIFEST_CACHE.get(key)
    if cached is not None:
        return cached

    adir = _asset_dir(category, asset_id)
    if adir is None:
        return None
    manifest_path = adir / "manifest.json"
    if not manifest_path.is_file():
        return None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    glb_name = str(manifest.get("visual_glb") or "visual.glb")
    glb_path = adir / glb_name
    if not glb_path.is_file():
        return None

    entry = (manifest, glb_path)
    _VISUAL_MANIFEST_CACHE[key] = entry
    return entry


def _read_raw_bundle(category: str, asset_id: str) -> _RawCachedAsset | None:
    key = (category, asset_id)
    cached = _ASSET_CACHE.get(key)
    if cached is not None:
        return cached

    adir = _asset_dir(category, asset_id)
    if adir is None:
        return None
    manifest_path = adir / "manifest.json"
    npz_path = adir / "superelement.npz"
    if not (manifest_path.is_file() and npz_path.is_file()):
        return None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with np.load(npz_path) as data:
        K_cb = np.array(data["K_cb"], dtype=np.float64)
        M_cb = np.array(data["M_cb"], dtype=np.float64)
        K_bb = np.array(data["K_bb"], dtype=np.float64)
        port_coords_m = np.array(data["port_coords_m"], dtype=np.float64)
        freqs = tuple(float(x) for x in np.asarray(data["fixed_interface_freqs_hz"], dtype=np.float64).ravel())
        com_m = np.array(data["com_m"], dtype=np.float64).reshape(3)
        inertia_kg_m2 = np.array(data["inertia_kg_m2"], dtype=np.float64).reshape(3, 3)

    K_cb.setflags(write=False)
    M_cb.setflags(write=False)
    K_bb.setflags(write=False)
    port_coords_m.setflags(write=False)
    com_m.setflags(write=False)
    inertia_kg_m2.setflags(write=False)

    glb_name = str(manifest.get("visual_glb") or "visual.glb")
    glb_path = adir / glb_name
    visual_path = glb_path if glb_path.is_file() else None

    raw = _RawCachedAsset(
        manifest=manifest,
        K_cb=K_cb,
        M_cb=M_cb,
        K_bb=K_bb,
        port_coords_m=port_coords_m,
        fixed_interface_freqs_hz=freqs,
        com_m=com_m,
        inertia_kg_m2=inertia_kg_m2,
        visual_glb_path=visual_path,
    )
    _ASSET_CACHE[key] = raw
    return raw


def has_hardware_asset(category: str, asset_id: str | None) -> bool:
    """Return True if a baked superelement bundle exists for ``(category, asset_id)``."""
    adir = _asset_dir(category, asset_id)
    if adir is None:
        return False
    return (adir / "manifest.json").is_file() and (adir / "superelement.npz").is_file()


def has_visual_asset(category: str, asset_id: str | None) -> bool:
    """Return True if a baked ``visual.glb`` exists for ``(category, asset_id)``."""
    adir = _asset_dir(category, asset_id)
    if adir is None:
        return False
    if (adir / "visual.glb").is_file():
        return True
    return get_visual_glb_path(category, asset_id) is not None


def get_hardware_asset_manifest(category: str, asset_id: str | None) -> dict[str, Any] | None:
    """Return a copy of the baked ``manifest.json`` (with ``visual_url`` and ``has_superelement``) or ``None``."""
    if not asset_id:
        return None
    aid = str(asset_id)
    raw = _read_raw_bundle(category, aid)
    if raw is not None:
        return _manifest_with_visual_url(raw.manifest, category, aid, has_superelement=True)
    vis = _read_visual_only_manifest(category, aid)
    if vis is not None:
        return _manifest_with_visual_url(vis[0], category, aid, has_superelement=False)
    return None


def get_visual_glb_path(category: str, asset_id: str | None) -> Path | None:
    """Return the path to ``visual.glb`` for ``(category, asset_id)`` if present (supports visual-only bundles)."""
    if not asset_id:
        return None
    aid = str(asset_id)
    adir = _asset_dir(category, aid)
    if adir is None:
        return None
    cached_raw = _ASSET_CACHE.get((category, aid))
    if cached_raw is not None:
        return cached_raw.visual_glb_path
    vis = _read_visual_only_manifest(category, aid)
    if vis is not None:
        return vis[1]
    default_glb = adir / "visual.glb"
    return default_glb if default_glb.is_file() else None


def list_hardware_assets() -> dict[str, dict[str, dict[str, Any]]]:
    """Discover all baked CAD hardware assets (superelements and visual-only bundles) grouped by category."""
    catalog: dict[str, dict[str, dict[str, Any]]] = {cat: {} for cat in _DEFAULT_CATEGORIES}
    if not ASSETS_ROOT.is_dir():
        return catalog

    for cat_dir in sorted(p for p in ASSETS_ROOT.iterdir() if p.is_dir()):
        cat_name = cat_dir.name
        bucket = catalog.setdefault(cat_name, {})
        for asset_dir in sorted(p for p in cat_dir.iterdir() if p.is_dir()):
            aid = asset_dir.name
            raw = _read_raw_bundle(cat_name, aid)
            if raw is not None:
                bucket[aid] = _manifest_with_visual_url(
                    raw.manifest, cat_name, aid, has_superelement=True
                )
            else:
                vis = _read_visual_only_manifest(cat_name, aid)
                if vis is not None:
                    bucket[aid] = _manifest_with_visual_url(
                        vis[0], cat_name, aid, has_superelement=False
                    )
    return catalog


def load_superelement(
    category: str,
    asset_id: str,
    material_id: str | None = None,
    chamber_temp_c: float = 25.0,
    stiffness_mult: float = 1.0,
) -> LoadedHardwareAsset | None:
    """Load a baked CAD superelement with optional material/temperature/stiffness scaling.

    For single-material assets (``not manifest["is_multi_material"]``), providing
    ``material_id`` scales ``K_cb`` and ``K_bb`` by ``E_eff(material_id, chamber_temp_c) / E_ref``,
    scales ``M_cb`` and ``inertia_kg_m2`` by ``rho / rho_ref``, and updates
    ``fixed_interface_freqs_hz`` by ``sqrt(scale_k / scale_m)``.
    For multi-material assets, only ``stiffness_mult`` is applied to ``K_cb`` and ``K_bb``.
    """
    raw = _read_raw_bundle(category, asset_id)
    if raw is None:
        return None

    manifest = _manifest_with_visual_url(raw.manifest, category, asset_id, has_superelement=True)
    scale_k = float(stiffness_mult)
    scale_m = 1.0

    is_multi = bool(manifest.get("is_multi_material", False))
    if material_id is not None and not is_multi:
        mat = get_material(material_id)
        e_ref = float(manifest["reference_youngs_modulus_pa"])
        rho_ref = float(manifest["reference_density_kg_m3"])
        e_eff = float(mat.effective_modulus_pa(float(chamber_temp_c)))
        scale_k *= e_eff / max(e_ref, 1e-12)
        scale_m *= float(mat.density_kg_m3) / max(rho_ref, 1e-12)
        if category == "gantry_brackets" and bool(mat.is_printed_polymer):
            kd = gantry_bracket_polymer_knockdown(
                material_id,
                chamber_temp_c=float(chamber_temp_c),
                reference_is_fdm=bool(manifest.get("is_fdm_anisotropic", False)),
                reference_material_id=str(manifest.get("reference_material_id") or "") or None,
            )
            scale_k *= float(kd["total_stiffness_factor"])
            manifest["polymer_knockdown_applied"] = kd

    K_cb = raw.K_cb * scale_k
    K_bb = raw.K_bb * scale_k
    M_cb = raw.M_cb * scale_m
    inertia = raw.inertia_kg_m2 * scale_m
    freq_scale = math.sqrt(max(scale_k, 0.0) / max(scale_m, 1e-18))
    freqs = [f * freq_scale for f in raw.fixed_interface_freqs_hz]

    return LoadedHardwareAsset(
        category=category,
        asset_id=asset_id,
        manifest=manifest,
        K_cb=K_cb,
        M_cb=M_cb,
        K_bb=K_bb,
        port_coords_m=raw.port_coords_m.copy(),
        fixed_interface_freqs_hz=freqs,
        com_m=raw.com_m.copy(),
        inertia_kg_m2=inertia,
        visual_glb_path=raw.visual_glb_path,
        stiffness_scale_applied=scale_k,
        mass_scale_applied=scale_m,
    )
