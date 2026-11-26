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


def test_v03_topology_and_bom_api_and_mcp(tmp_path: Path) -> None:
    """End-to-end check for v0.3 topology optimization, STL export, BOM API, and MCP tools."""
    client = TestClient(create_app())

    # 1. Topology optimization HTTP API
    r_topo = client.post(
        "/api/fea/topology-optimize",
        json={
            "preset": "cantilever_bar",
            "material_id": "pa12_cf_printed",
            "target_volume_fraction": 0.5,
            "iterations": 10,
        },
    )
    assert r_topo.status_code == 200
    topo = r_topo.json()
    assert topo["optimized_mass_g"] < topo["baseline_mass_g"]
    assert topo["mass_reduction_pct"] > 20.0
    assert len(topo["history"]) >= 5
    assert len(topo["voxels"]) == topo["baseline_elements"]

    # 2. Topology STL download HTTP API
    r_stl = client.post(
        "/api/fea/topology-optimize/stl",
        json={
            "preset": "cantilever_bar",
            "material_id": "pa12_cf_printed",
            "target_volume_fraction": 0.5,
            "iterations": 8,
        },
    )
    assert r_stl.status_code == 200
    assert len(r_stl.content) > 500

    # 3. BOM JSON & CSV HTTP API
    yaml_text = (EXAMPLES_DIR / "corexy_voron_style.yaml").read_text(encoding="utf-8")
    r_bom = client.post("/api/bom", json={"yaml_text": yaml_text, "stock_length_mm": 1000.0})
    assert r_bom.status_code == 200
    bom = r_bom.json()
    assert bom["total_cost_usd"] > 100.0
    assert len(bom["cut_list"]) >= 3
    assert len(bom["stock_nesting"]) >= 1

    r_csv = client.post("/api/bom/csv", json={"yaml_text": yaml_text, "stock_length_mm": 1000.0})
    assert r_csv.status_code == 200
    assert "text/csv" in r_csv.headers["content-type"]
    assert "2020" in r_csv.text or "extrusion" in r_csv.text.lower()

    # 4. MCP tools: optimize_part_topology + generate_printer_bom
    out_stl = tmp_path / "mcp_opt.stl"
    r_mcp_topo = handle_jsonrpc_message(
        {
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/call",
            "params": {
                "name": "optimize_part_topology",
                "arguments": {
                    "preset": "cantilever_bar",
                    "target_volume_fraction": 0.5,
                    "iterations": 8,
                    "output_stl_path": str(out_stl),
                },
            },
        }
    )
    assert r_mcp_topo is not None
    assert r_mcp_topo["result"]["isError"] is False
    assert out_stl.exists() and out_stl.stat().st_size > 500

    r_mcp_bom = handle_jsonrpc_message(
        {
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tools/call",
            "params": {
                "name": "generate_printer_bom",
                "arguments": {"yaml_text": yaml_text, "stock_length_mm": 1000.0},
            },
        }
    )
    assert r_mcp_bom is not None
    assert r_mcp_bom["result"]["isError"] is False
    assert "total_cost_usd" in r_mcp_bom["result"]["content"][0]["text"]


def test_v04_build123d_step_and_gmsh_skfem_api_and_mcp(tmp_path: Path) -> None:
    """End-to-end check for v0.4 build123d STEP export and Gmsh + scikit-fem Tet10 FEA API & MCP."""
    from printsim.fea.tet_fem import is_tet_fea_available
    from printsim.geometry.cad_b3d import is_cad_available

    if not (is_cad_available() and is_tet_fea_available()):
        pytest.skip("Requires optional CAD + TetFEA extras (`uv sync --extra cad --extra tetfea`)")
    client = TestClient(create_app())

    # 1. Bracket STEP download
    r_part_step = client.post("/api/export/part-step", json={"preset": "corner_gusset"})
    assert r_part_step.status_code == 200
    assert b"ISO-10303-21" in r_part_step.content

    # 2. Full printer assembly STEP download
    r_preset = client.get("/api/presets/corexy_voron_style")
    assert r_preset.status_code == 200
    spec_json = r_preset.json()["spec"]
    r_asm_step = client.post("/api/export/step", json=spec_json)
    assert r_asm_step.status_code == 200
    assert b"ISO-10303-21" in r_asm_step.content

    # 3. Curved Tet10 FEA on preset bracket
    r_tet = client.post(
        "/api/fea/tet-part",
        json={"preset": "corner_gusset", "material_id": "pa12_cf_printed", "order": 2},
    )
    assert r_tet.status_code == 200
    tet = r_tet.json()
    assert tet["backend"] == "gmsh+scikit-fem"
    assert tet["num_elements"] > 50
    assert tet["safety_factor"] > 1.0
    assert len(tet["surface_mesh"]["triangles"]) > 50

    # 4. Upload STEP bytes directly to /api/fea/tet-step
    b64_step = base64.b64encode(r_part_step.content).decode("ascii")
    r_tet_step = client.post(
        "/api/fea/tet-step",
        json={
            "part_name": "uploaded_gusset.step",
            "step_base64": b64_step,
            "material_id": "aluminum_6061_t6",
            "fixed_face": "min_y",
            "load_face": "max_y",
            "force_n": [60.0, 0.0, 0.0],
            "order": 2,
        },
    )
    assert r_tet_step.status_code == 200
    assert r_tet_step.json()["stiffness_n_per_mm"] > tet["stiffness_n_per_mm"] * 5.0

    # 5. MCP tools: export_printer_step + run_tet_part_fea
    yaml_text = (EXAMPLES_DIR / "corexy_voron_style.yaml").read_text(encoding="utf-8")
    out_step = tmp_path / "mcp_corexy.step"
    r_mcp_step = handle_jsonrpc_message(
        {
            "jsonrpc": "2.0",
            "id": 20,
            "method": "tools/call",
            "params": {
                "name": "export_printer_step",
                "arguments": {"yaml_text": yaml_text, "output_path": str(out_step)},
            },
        }
    )
    assert r_mcp_step is not None
    assert r_mcp_step["result"]["isError"] is False
    assert out_step.exists() and b"ISO-10303-21" in out_step.read_bytes()

    r_mcp_tet = handle_jsonrpc_message(
        {
            "jsonrpc": "2.0",
            "id": 21,
            "method": "tools/call",
            "params": {
                "name": "run_tet_part_fea",
                "arguments": {"preset": "xy_motor_mount", "material_id": "pa12_cf_printed", "order": 2},
            },
        }
    )
    assert r_mcp_tet is not None
    assert r_mcp_tet["result"]["isError"] is False
    assert "gmsh+scikit-fem" in r_mcp_tet["result"]["content"][0]["text"]


def test_v07_klipper_psd_and_pymoo_pareto_api_and_mcp() -> None:
    """End-to-end check for v0.7 Klipper ADXL345 shaper PSD graphs and pymoo Pareto API & MCP."""
    client = TestClient(create_app())
    yaml_text = (EXAMPLES_DIR / "voron_trident_300.yaml").read_text(encoding="utf-8")

    # 1. Simulate returns full Klipper shaper_graphs for X and Y
    r_sim = client.post("/api/simulate", json={"yaml_text": yaml_text, "include_toolpath_segments": False})
    assert r_sim.status_code == 200
    sim = r_sim.json()
    graphs = sim["modal_fea"]["shaper_graphs"]
    assert len(graphs["x"]["freqs_hz"]) >= 80
    assert max(graphs["x"]["freqs_hz"]) >= 215.0
    assert len(graphs["x"]["candidates"]) == 5
    assert len(graphs["y"]["candidates"]) == 5

    # 2. Pareto HTTP endpoint returns non-dominated frontier with specs
    r_par = client.post(
        "/api/optimize/pareto",
        json={"yaml_text": yaml_text, "population_size": 12, "generations": 4},
    )
    assert r_par.status_code == 200
    par = r_par.json()
    assert "NSGA-II" in par["backend"]
    assert len(par["pareto_front"]) >= 3
    assert "spec" in par["pareto_front"][0]

    # 3. MCP optimize_printer_pareto tool
    r_mcp_par = handle_jsonrpc_message(
        {
            "jsonrpc": "2.0",
            "id": 30,
            "method": "tools/call",
            "params": {
                "name": "optimize_printer_pareto",
                "arguments": {"yaml_text": yaml_text, "population_size": 10, "generations": 3},
            },
        }
    )
    assert r_mcp_par is not None
    assert r_mcp_par["result"]["isError"] is False
    assert "pareto_front" in r_mcp_par["result"]["content"][0]["text"]
