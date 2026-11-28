"""End-to-end tests for evaluator, AI optimizer, STL export, presets, and FastAPI server."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from printsim.api.server import create_app
from printsim.engine import evaluate_printer, optimize_design
from printsim.geometry import build_assembly, export_assembly_stl
from printsim.schema import PrinterSpec

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def test_all_example_presets_evaluate_cleanly() -> None:
    yamls = sorted(EXAMPLES_DIR.glob("*.yaml"))
    assert len(yamls) >= 4
    for y in yamls:
        spec = PrinterSpec.from_yaml(y)
        rep = evaluate_printer(spec)
        assert 0.0 <= rep.scores.composite <= 100.0
        assert rep.modal_fea.fundamental_freq_hz > 10.0
        assert not rep.collision.has_collision


def test_ai_optimizer_improves_baseline_bedslinger() -> None:
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "cartesian_bedslinger.yaml")
    res = optimize_design(spec, iterations=12, objective="rigidity", seed=7)
    assert res.best_objective >= res.baseline_objective
    assert len(res.history) == 13


def test_stl_export_creates_valid_binary_mesh(tmp_path: Path) -> None:
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    asm = build_assembly(spec)
    out = export_assembly_stl(asm, tmp_path / "corexy.stl")
    assert out.exists()
    assert out.stat().st_size > 1000


def test_fastapi_endpoints() -> None:
    client = TestClient(create_app())

    r_health = client.get("/api/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"

    r_cat = client.get("/api/catalog")
    assert r_cat.status_code == 200
    data = r_cat.json()
    assert len(data["materials"]) >= 10
    assert len(data["profiles"]) >= 10
    assert len(data["presets"]) >= 4

    r_preset = client.get("/api/presets/corexy_voron_style")
    assert r_preset.status_code == 200
    spec_json = r_preset.json()["spec"]

    r_sim = client.post("/api/simulate", json={"spec": spec_json})
    assert r_sim.status_code == 200
    sim = r_sim.json()
    assert sim["scores"]["composite"] > 40
    assert len(sim["modal_fea"]["modes"]) >= 4
    assert len(sim["virtual_print"]["segments"]) > 10

    r_opt = client.post(
        "/api/optimize",
        json={"spec": spec_json, "iterations": 6, "objective": "speed"},
    )
    assert r_opt.status_code == 200
    assert r_opt.json()["best_objective"] >= r_opt.json()["baseline_objective"]


def test_studio_catalog_presets_are_grouped_by_model_status() -> None:
    """Studio /api/catalog groups presets & component catalogs into Complete, WIP, and Generic."""
    from pathlib import Path

    client = TestClient(create_app())
    r_cat = client.get("/api/catalog")
    assert r_cat.status_code == 200
    payload = r_cat.json()
    presets = payload["presets"]
    assert len(presets) >= 20

    categories = {p["category"] for p in presets}
    assert categories == {"complete", "wip_specific", "generic_standin"}

    by_id = {p["id"]: p for p in presets}
    # Every examples/*.yaml preset must be explicitly classified
    yaml_ids = {p.stem for p in Path("examples").glob("*.yaml")}
    assert yaml_ids == set(by_id.keys())

    assert by_id["siboor_voron_trident_awd_300"]["category"] == "complete"
    assert by_id["siboor_voron_trident_awd_300"]["base_kit"] == "siboor"
    assert by_id["siboor_voron_trident_awd_300"]["hardware_kit"] == "siboor_cnc_awd"
    assert by_id["ldo_voron_trident_revd_300"]["category"] == "complete"
    assert by_id["ldo_voron_trident_revd_300"]["base_kit"] == "ldo"
    assert by_id["ldo_voron_trident_revd_300"]["hardware_kit"] == "ldo_voron_kit"
    assert by_id["voron_trident_300"]["category"] == "complete"
    assert by_id["voron_monolith_awd"]["category"] == "complete"
    assert by_id["vzbot_330_awd"]["category"] == "complete"
    assert by_id["vzbot_330_awd"]["hardware_kit"] == "vzbot_330_awd"

    assert by_id["ratrig_vcore31_300"]["category"] == "wip_specific"
    assert by_id["ratrig_vcore31_300"].get("wip_notes")
    assert by_id["monolith_zero_m0"]["category"] == "wip_specific"
    assert by_id["corexy_fixed_gantry_300"]["category"] == "generic_standin"
    assert by_id["cartesian_bedslinger"]["category"] == "generic_standin"
    assert by_id["delta_racer"]["category"] == "generic_standin"

    # Component catalogs carry 3-tier model_status tags
    th_by_id = {t["preset_id"]: t for t in payload["toolhead_presets"]}
    assert th_by_id["stealthburner"]["model_status"] == "complete"
    assert th_by_id["monolith_toolhead_cpap"]["model_status"] == "wip_specific"
    assert th_by_id["heavy_uhf_long"]["model_status"] == "generic_standin"

    kit_by_id = {k["id"]: k for k in payload["gantry_hardware_kits"]}
    assert kit_by_id["monolith_ldo_cnc_awd"]["model_status"] == "complete"
    assert kit_by_id["procedural"]["model_status"] == "generic_standin"

    he_by_id = {h["id"]: h for h in payload["hotend_presets"]}
    assert he_by_id["chube_conduction"]["model_status"] == "complete"

    prof_by_id = {p["id"]: p for p in payload["profiles"]}
    assert prof_by_id["cnc_al_monocoque_20x20"]["model_status"] == "complete"
    assert prof_by_id["cf_tube_20x20_carbonara"]["model_status"] == "wip_specific"
    assert prof_by_id["extrusion_2020"]["model_status"] == "generic_standin"

    z_drive_by_id = {z["id"]: z for z in payload["z_drive_mechanisms"]}
    assert z_drive_by_id["tr8x4_pom"]["model_status"] == "complete"
    assert z_drive_by_id["tr8x4_wobblex"]["model_status"] == "wip_specific"
    assert z_drive_by_id["gt2_9mm_planetary_z"]["model_status"] == "generic_standin"

    z_rail_by_id = {r["id"]: r for r in payload["z_rail_profiles"]}
    assert z_rail_by_id["mgn12_rail"]["model_status"] == "complete"
    assert z_rail_by_id["steel_rod_10mm"]["model_status"] == "generic_standin"

    bed_mount_by_id = {m["id"]: m for m in payload["bed_mounts"]}
    assert bed_mount_by_id["trident_3pt_standoffs"]["model_status"] == "complete"
    assert bed_mount_by_id["whopping_pochard_kinematic"]["model_status"] == "wip_specific"
    assert bed_mount_by_id["rigid_4pt"]["model_status"] == "generic_standin"

    z_probe_by_id = {z["id"]: z for z in payload["z_probes"]}
    assert z_probe_by_id["printed_tap"]["model_status"] == "complete"
    assert z_probe_by_id["cnc_tap"]["model_status"] == "wip_specific"

    for p in presets:
        assert p.get("short_name")
        assert p.get("category_label")
        assert p.get("model_status") in {"complete", "wip_specific", "generic_standin"}
        assert p.get("summary")
        assert p.get("highlights")
        r_get = client.get(f"/api/presets/{p['id']}")
        assert r_get.status_code == 200
        assert r_get.json()["id"] == p["id"]

    # WIP specific builds are surfaced in /api/catalog under wip_specific with notes
    for wip_build_id in (
        "doomcube_4040",
        "ratrig_vcore31_300",
        "flsun_v400_delta",
        "monolith_sheet_metal_v2",
        "monolith_zero_m0",
        "cw_fixed_gantry_300",
    ):
        assert wip_build_id in by_id
        assert by_id[wip_build_id]["category"] == "wip_specific"
        assert by_id[wip_build_id].get("wip_notes")
        r_test = client.get(f"/api/presets/{wip_build_id}")
        assert r_test.status_code == 200
