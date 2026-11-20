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
