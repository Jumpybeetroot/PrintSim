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

_DEFAULT_CATEGORIES: tuple[str, ...] = ("hotends", "toolheads", "assemblies", "gantry_brackets")


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


def clear_asset_cache() -> None:
    """Clear the in-memory hardware superelement cache (useful after re-baking assets)."""
    _ASSET_CACHE.clear()


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
) -> dict[str, Any]:
    out = dict(manifest)
    out["visual_url"] = f"/api/assets/hardware/{category}/{asset_id}/visual.glb"
    return out


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


def get_hardware_asset_manifest(category: str, asset_id: str | None) -> dict[str, Any] | None:
    """Return a copy of the baked ``manifest.json`` (with ``visual_url``) or ``None``."""
    if not asset_id:
        return None
    raw = _read_raw_bundle(category, str(asset_id))
    if raw is None:
        return None
    return _manifest_with_visual_url(raw.manifest, category, str(asset_id))


def get_visual_glb_path(category: str, asset_id: str | None) -> Path | None:
    """Return the path to ``visual.glb`` for ``(category, asset_id)`` if present."""
    if not asset_id:
        return None
    raw = _read_raw_bundle(category, str(asset_id))
    if raw is None:
        return None
    return raw.visual_glb_path


def list_hardware_assets() -> dict[str, dict[str, dict[str, Any]]]:
    """Discover all baked CAD hardware assets grouped by category (always includes hotends, toolheads, assemblies, gantry_brackets)."""
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
                bucket[aid] = _manifest_with_visual_url(raw.manifest, cat_name, aid)
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

    manifest = _manifest_with_visual_url(raw.manifest, category, asset_id)
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
