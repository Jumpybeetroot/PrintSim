"""Offline CAD-backed hardware superelement registry, inspector, recipe baker & QA.

Runtime superelement loading (``.registry``) is imported eagerly with only NumPy
and core schema dependencies; offline CAD baking, inspection, and Onshape recipe
helpers are lazy-loaded via PEP 562 ``__getattr__`` so minimal & Pyodide WASM
environments never require ``trimesh`` or ``httpx`` at import time.
"""

from __future__ import annotations

import importlib
from typing import Any

from .registry import (
    ASSETS_ROOT,
    LoadedHardwareAsset,
    clear_asset_cache,
    get_hardware_asset_manifest,
    get_visual_glb_path,
    has_hardware_asset,
    list_hardware_assets,
    load_superelement,
)

_LAZY_EXPORTS: dict[str, str] = {
    # .baker
    "BakedAssetResult": "baker",
    "BakedAssetSpec": "baker",
    "MaterialRegion": "baker",
    "PortDefinition": "baker",
    "VisualPart": "baker",
    "bake_solid_superelement": "baker",
    "bake_step_file": "baker",
    "build_visual_glb": "baker",
    "decimate_trimesh_qem": "baker",
    "select_plane_box": "baker",
    "select_z_annulus": "baker",
    "select_z_cap": "baker",
    # .inspect
    "CylindricalFeature": "inspect",
    "PlanarFaceGroup": "inspect",
    "SolidInspectInfo": "inspect",
    "StepInspectReport": "inspect",
    "inspect_step_file": "inspect",
    # .recipes
    "AssetQAReport": "recipes",
    "CadAssetRecipe": "recipes",
    "RecipeOnshapeSpec": "recipes",
    "RecipePortSpec": "recipes",
    "RecipeRegionSpec": "recipes",
    "RecipeTransform": "recipes",
    "bake_from_recipe": "recipes",
    "fetch_onshape_step": "recipes",
    "parse_onshape_reference": "recipes",
    "render_contact_sheet": "recipes",
    "verify_baked_asset": "recipes",
}


def __getattr__(name: str) -> Any:
    mod_name = _LAZY_EXPORTS.get(name)
    if mod_name is not None:
        mod = importlib.import_module(f".{mod_name}", __name__)
        val = getattr(mod, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ASSETS_ROOT",
    "AssetQAReport",
    "BakedAssetResult",
    "BakedAssetSpec",
    "CadAssetRecipe",
    "CylindricalFeature",
    "LoadedHardwareAsset",
    "MaterialRegion",
    "PlanarFaceGroup",
    "PortDefinition",
    "RecipeOnshapeSpec",
    "RecipePortSpec",
    "RecipeRegionSpec",
    "RecipeTransform",
    "SolidInspectInfo",
    "StepInspectReport",
    "VisualPart",
    "bake_from_recipe",
    "bake_solid_superelement",
    "bake_step_file",
    "build_visual_glb",
    "clear_asset_cache",
    "decimate_trimesh_qem",
    "fetch_onshape_step",
    "get_hardware_asset_manifest",
    "get_visual_glb_path",
    "has_hardware_asset",
    "inspect_step_file",
    "list_hardware_assets",
    "load_superelement",
    "parse_onshape_reference",
    "render_contact_sheet",
    "select_plane_box",
    "select_z_annulus",
    "select_z_cap",
    "verify_baked_asset",
]
