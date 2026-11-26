"""Tests for the zero-FastAPI Pyodide WASM & GitHub Pages request dispatcher."""

from __future__ import annotations

import base64
import json

from printsim.wasm_dispatch import dispatch_api_request


def test_wasm_dispatch_catalog_presets_and_hardware() -> None:
    health = dispatch_api_request("GET", "/api/health")
    assert health["status"] == 200
    assert json.loads(health["body_text"])["status"] == "ok"

    cat_env = dispatch_api_request("GET", "/api/catalog")
    assert cat_env["status"] == 200
    cat = json.loads(cat_env["body_text"])
    assert len(cat["presets"]) >= 5
    assert len(cat["materials"]) >= 10
    assert len(cat["steppers"]) >= 10

    preset_env = dispatch_api_request("GET", "/api/presets/voron_trident_300")
    assert preset_env["status"] == 200
    preset = json.loads(preset_env["body_text"])
    assert preset["id"] == "voron_trident_300"
    assert "spec" in preset and "yaml_text" in preset

    hw_env = dispatch_api_request("GET", "/api/assets/hardware")
    assert hw_env["status"] == 200
    hw = json.loads(hw_env["body_text"])
    assert "hotends" in hw


def test_wasm_dispatch_simulate_bom_solid_and_exports() -> None:
    preset = json.loads(dispatch_api_request("GET", "/api/presets/voron_trident_300")["body_text"])
    spec = preset["spec"]

    sim_env = dispatch_api_request(
        "POST",
        "/api/simulate",
        json.dumps({"spec": spec, "include_toolpath_segments": False}),
    )
    assert sim_env["status"] == 200
    sim = json.loads(sim_env["body_text"])
    assert sim["modal_fea"]["fundamental_freq_hz"] > 20.0
    assert sim["static_fea"]["toolhead_stiffness_x_n_per_mm"] > 1.0

    bom_env = dispatch_api_request("POST", "/api/bom", json.dumps({"spec": spec}))
    assert bom_env["status"] == 200
    bom = json.loads(bom_env["body_text"])
    assert bom["total_cost_usd"] > 100.0

    csv_env = dispatch_api_request("POST", "/api/bom/csv", json.dumps({"spec": spec}))
    assert csv_env["status"] == 200
    assert csv_env["content_type"] == "text/csv"
    assert "section,code,category" in csv_env["body_text"]

    solid_env = dispatch_api_request(
        "POST",
        "/api/fea/solid-part",
        json.dumps({"preset": "corner_gusset", "material_id": "abs_printed"}),
    )
    assert solid_env["status"] == 200
    solid = json.loads(solid_env["body_text"])
    assert solid["max_von_mises_mpa"] > 0.0

    dxf_env = dispatch_api_request(
        "POST",
        "/api/export/dxf",
        json.dumps({"spec": spec, "part": "side_panel"}),
    )
    assert dxf_env["status"] == 200
    dxf_raw = base64.b64decode(dxf_env["body_base64"])
    assert b"SECTION" in dxf_raw

    # Native-only endpoints return clean 400 guidance so frontend falls back gracefully
    tet_env = dispatch_api_request("POST", "/api/fea/tet-part", json.dumps({"preset": "corner_gusset"}))
    assert tet_env["status"] == 400
    assert "Hex8" in json.loads(tet_env["body_text"])["detail"]

    step_env = dispatch_api_request("POST", "/api/export/step", json.dumps(spec))
    assert step_env["status"] == 400
    assert "build123d" in json.loads(step_env["body_text"])["detail"]


def test_wasm_dispatch_runs_without_optional_native_or_web_packages() -> None:
    """Verify Pyodide-like minimal runtime (only numpy/scipy/pydantic/pyyaml) can simulate."""
    import subprocess
    import sys

    probe = (
        "import json, sys\n"
        "for pkg in ('trimesh', 'ezdxf', 'httpx', 'fastapi', 'uvicorn', 'typer', 'rich', 'gmsh', 'build123d'):\n"
        "    sys.modules[pkg] = None\n"
        "from printsim.wasm_dispatch import dispatch_api_request\n"
        "p = json.loads(dispatch_api_request('GET', '/api/presets/voron_monolith_awd')['body_text'])\n"
        "s = dispatch_api_request('POST', '/api/simulate', json.dumps({'spec': p['spec'], 'include_toolpath_segments': False}))\n"
        "assert s['status'] == 200, s\n"
        "sim = json.loads(s['body_text'])\n"
        "assert sim['modal_fea']['fundamental_freq_hz'] > 40.0\n"
        "assert dispatch_api_request('GET', '/api/assets/hardware')['status'] == 200\n"
    )
    proc = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"Pyodide-minimal probe failed:\n{proc.stderr}"
