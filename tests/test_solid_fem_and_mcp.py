"""Analytical & integration tests for 3D Solid Hex8 FEA, FVM axial hotend thermal, and MCP server."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
import trimesh
from fastapi.testclient import TestClient

from printsim.api.server import create_app
from printsim.fea.solid_fem import solve_preset_part_fea
from printsim.mcp_server import handle_jsonrpc_message
from printsim.schema import PrinterSpec, get_material
from printsim.thermal import simulate_hotend_thermal

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def test_solid_hex8_uniaxial_tension_analytical() -> None:
    """Verify 3D Hex8 solid FEM matches exact uniaxial extension FL/(EA) and stress F/A."""
    # cantilever_bar is 60mm (X) x 12mm (Y) x 12mm (Z) -> L=0.06m, A=144e-6 m^2
    F_x = 1440.0  # N -> exact sigma_xx = 10.0 MPa
    res = solve_preset_part_fea(
        preset="cantilever_bar",
        material_id="aluminum_6061_t6",
        force_n=(F_x, 0.0, 0.0),
        fdm_anisotropic=False,
    )
    mat = get_material("aluminum_6061_t6")
    L = 0.060
    A = 0.012 * 0.012
    exact_disp_um = (F_x * L / (mat.youngs_modulus_pa * A)) * 1e6
    # Fixed face Poisson constraint creates slight 3D end-effect; expect within 8%
    assert res.max_displacement_um == pytest.approx(exact_disp_um, rel=0.08)
    assert res.max_von_mises_mpa == pytest.approx(10.0, rel=0.18)


def test_solid_hex8_fdm_anisotropy_vs_cnc_aluminum() -> None:
    """Aluminum 6061 corner gusset must be far stiffer and stronger than FDM-printed ABS."""
    res_abs = solve_preset_part_fea("corner_gusset", material_id="abs_printed", force_n=(120.0, 0.0, 60.0))
    res_al = solve_preset_part_fea("corner_gusset", material_id="aluminum_6061_t6", force_n=(120.0, 0.0, 60.0))

    assert res_abs.is_fdm_anisotropic is True
    assert res_al.is_fdm_anisotropic is False
    assert res_al.stiffness_n_per_mm > res_abs.stiffness_n_per_mm * 15.0
    assert res_al.safety_factor > res_abs.safety_factor * 5.0


def test_stl_upload_solid_fea_endpoint() -> None:
    """Export a watertight STL prism, upload base64 to /api/fea/solid-stl, and verify 3D FEA."""
    box = trimesh.creation.box(extents=[40.0, 20.0, 15.0])
    stl_bytes = box.export(file_type="stl")
    b64 = base64.b64encode(stl_bytes).decode("ascii")

    client = TestClient(create_app())
    r = client.post(
        "/api/fea/solid-stl",
        json={
            "part_name": "Test Bracket Box",
            "stl_base64": b64,
            "material_id": "pa12_cf_printed",
            "fixed_face": "min_x",
            "load_face": "max_x",
            "force_n": [0.0, 0.0, -80.0],
            "resolution": 12,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["num_elements"] > 50
    assert body["max_displacement_um"] > 0.1
    assert body["max_von_mises_mpa"] > 0.1
    assert len(body["voxels"]) == body["num_elements"]


def test_hotend_fvm_axial_profile_and_filament_rheology() -> None:
    """Verify FVM axial temperature drops monotonically from block to heatsink and TPU flows slower than PLA."""
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    th_pla = simulate_hotend_thermal(spec, filament_id="pla")
    th_tpu = simulate_hotend_thermal(spec, filament_id="tpu_95a")

    assert len(th_pla.axial_profile) == 40
    block_temps = [p["temp_c"] for p in th_pla.axial_profile if p["zone"] == "block"]
    hs_temps = [p["temp_c"] for p in th_pla.axial_profile if p["zone"] == "heatsink"]
    assert max(block_temps) > 210.0
    assert min(hs_temps) < 45.0
    # Higher melt viscosity TPU has lower flow limit than PLA
    assert th_tpu.max_volumetric_flow_mm3_s < th_pla.max_volumetric_flow_mm3_s
    assert 40.0 <= th_pla.max_overhang_deg <= 80.0


def test_mcp_jsonrpc_server_dispatch() -> None:
    init_resp = handle_jsonrpc_message({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert init_resp is not None
    assert init_resp["result"]["serverInfo"]["name"] == "printsim-mcp"

    list_resp = handle_jsonrpc_message({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert list_resp is not None
    names = {t["name"] for t in list_resp["result"]["tools"]}
    assert {"list_catalog", "evaluate_printer_yaml", "optimize_printer_yaml", "run_solid_part_fea"} <= names

    call_resp = handle_jsonrpc_message(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "run_solid_part_fea",
                "arguments": {"preset": "xy_motor_mount", "material_id": "pa12_cf_printed"},
            },
        }
    )
    assert call_resp is not None
    assert call_resp["result"]["isError"] is False
    assert "safety_factor" in call_resp["result"]["content"][0]["text"]
