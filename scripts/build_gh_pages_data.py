#!/usr/bin/env python3
"""Pre-bake static JSON presets, hardware GLB assets, and a lean Pyodide WASM bundle for GitHub Pages.

Writes to ``web/public/pages-data/`` and ``web/public/api/assets/hardware/`` (both gitignored)
so the GitHub Pages Vite build serves:
  1. Instant (<1 s) pre-baked catalog, presets, simulations, BOMs, default Hex8 solid FEA,
     and PBR ``visual.glb`` hardware models straight from static CDN files.
  2. A compact ``printsim_wasm.zip`` loaded by a background Pyodide Web Worker so every
     interactive design edit, ADXL calibration, optimizer, and export runs live in-browser.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

from printsim.cad_assets.registry import ASSETS_ROOT, list_hardware_assets
from printsim.wasm_dispatch import build_catalog_payload, dispatch_api_request, load_presets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_PUBLIC = PROJECT_ROOT / "web" / "public"
PAGES_DATA_DIR = WEB_PUBLIC / "pages-data"
STATIC_HARDWARE_DIR = WEB_PUBLIC / "api" / "assets" / "hardware"

# Directories/files inside src/printsim excluded from the lean browser WASM zip
_ZIP_EXCLUDED_PARTS = frozenset({"__pycache__", "api", "kb"})
_ZIP_EXCLUDED_FILES = frozenset({"cli.py", "mcp_server.py", "visual.glb"})


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False), encoding="utf-8")


def _build_wasm_zip(out_zip: Path) -> tuple[int, int]:
    """Bundle pure-Python printsim + baked .npz superelements + example YAMLs into a compact zip."""
    src_pkg = PROJECT_ROOT / "src" / "printsim"
    examples_dir = PROJECT_ROOT / "examples"
    file_count = 0

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for file_path in sorted(src_pkg.rglob("*")):
            if not file_path.is_file():
                continue
            rel = file_path.relative_to(src_pkg)
            if any(part in _ZIP_EXCLUDED_PARTS for part in rel.parts):
                continue
            if file_path.name in _ZIP_EXCLUDED_FILES:
                continue
            if file_path.suffix in (".pyc", ".pyo", ".stl", ".step", ".stp", ".vtu", ".msh", ".png"):
                continue
            arcname = str(Path("printsim") / rel)
            zf.write(file_path, arcname)
            file_count += 1

        for yaml_path in sorted(examples_dir.glob("*.yaml")):
            zf.write(yaml_path, str(Path("examples") / yaml_path.name))
            file_count += 1

    return file_count, out_zip.stat().st_size


def _copy_hardware_glbs() -> int:
    if STATIC_HARDWARE_DIR.exists():
        shutil.rmtree(STATIC_HARDWARE_DIR)
    copied = 0
    if not ASSETS_ROOT.is_dir():
        return 0
    for glb in sorted(ASSETS_ROOT.rglob("visual.glb")):
        rel = glb.relative_to(ASSETS_ROOT)
        dest = STATIC_HARDWARE_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(glb, dest)
        copied += 1
    return copied


def main() -> None:
    if PAGES_DATA_DIR.exists():
        shutil.rmtree(PAGES_DATA_DIR)
    PAGES_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Catalog & hardware asset index
    catalog = build_catalog_payload()
    _write_json(PAGES_DATA_DIR / "catalog.json", catalog)
    _write_json(PAGES_DATA_DIR / "hardware_assets.json", list_hardware_assets())

    # 2. Copy baked visual.glb meshes to static /api/assets/hardware/...
    glb_count = _copy_hardware_glbs()

    # 3. Pre-bake all example presets (spec, simulation, BOM) for instant <1s switching
    presets = load_presets()
    preset_index: list[str] = []
    for preset_id, spec in presets.items():
        preset_index.append(preset_id)
        spec_dict = spec.model_dump(mode="json")
        _write_json(
            PAGES_DATA_DIR / "presets" / f"{preset_id}.json",
            {"id": preset_id, "spec": spec_dict, "yaml_text": spec.to_yaml()},
        )

        sim_env = dispatch_api_request(
            "POST",
            "/api/simulate",
            json.dumps(
                {
                    "spec": spec_dict,
                    "filament_id": "pla",
                    "toolhead_rel_pos": [0.5, 0.5, 0.5],
                    "include_toolpath_segments": True,
                }
            ),
        )
        if sim_env["status"] != 200:
            raise RuntimeError(f"Failed pre-baking simulation for {preset_id}: {sim_env}")
        (PAGES_DATA_DIR / "simulations").mkdir(parents=True, exist_ok=True)
        (PAGES_DATA_DIR / "simulations" / f"{preset_id}.json").write_text(
            sim_env["body_text"], encoding="utf-8"
        )

        bom_env = dispatch_api_request(
            "POST",
            "/api/bom",
            json.dumps({"spec": spec_dict}),
        )
        if bom_env["status"] != 200:
            raise RuntimeError(f"Failed pre-baking BOM for {preset_id}: {bom_env}")
        (PAGES_DATA_DIR / "boms").mkdir(parents=True, exist_ok=True)
        (PAGES_DATA_DIR / "boms" / f"{preset_id}.json").write_text(
            bom_env["body_text"], encoding="utf-8"
        )

    # 4. Pre-bake default Solid Part Hex8 FEA so initial Studio hydration never waits for WASM
    solid_env = dispatch_api_request(
        "POST",
        "/api/fea/solid-part",
        json.dumps({"preset": "corner_gusset", "material_id": "abs_printed"}),
    )
    if solid_env["status"] == 200:
        (PAGES_DATA_DIR / "solid_part_default.json").write_text(
            solid_env["body_text"], encoding="utf-8"
        )

    # 5. Build lean Pyodide WASM code + superelement archive
    wasm_zip = PAGES_DATA_DIR / "printsim_wasm.zip"
    n_files, zip_bytes = _build_wasm_zip(wasm_zip)

    _write_json(
        PAGES_DATA_DIR / "manifest.json",
        {
            "presets": preset_index,
            "glb_count": glb_count,
            "wasm_zip_files": n_files,
            "wasm_zip_bytes": zip_bytes,
        },
    )
    print(
        f"Baked GitHub Pages bundle: {len(preset_index)} presets, {glb_count} GLBs, "
        f"printsim_wasm.zip ({n_files} files, {zip_bytes / 1024:.1f} KiB)"
    )


if __name__ == "__main__":
    main()
