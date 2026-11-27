"""Regression tests for the curated real-machine input-shaper benchmark suite."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from printsim.cli import app
from printsim.kb.shaper_benchmarks import (
    build_benchmark_spec,
    compare_benchmark_reports,
    load_shaper_benchmarks,
    run_shaper_benchmarks,
)


def test_shaper_benchmark_manifest_valid_and_complete() -> None:
    """Every machine entry builds a valid PrinterSpec and every sweep references valid IDs."""
    manifest = load_shaper_benchmarks()
    machines = manifest["machines"]
    assert len(machines) >= 20

    ids = set()
    for entry in machines:
        mid = entry["id"]
        assert mid not in ids, f"Duplicate benchmark id: {mid}"
        ids.add(mid)
        assert entry["tier"] in (1, 2)
        spec = build_benchmark_spec(entry)
        assert spec.name

    for sw in manifest.get("controlled_sweeps", []):
        assert sw["baseline_machine"] in ids
        assert sw["variant_machine"] in ids
        assert sw["axis"].upper() in ("X", "Y")
        assert sw["measured_ratio"] > 0.0


def test_shaper_benchmarks_accuracy_and_controlled_sweeps() -> None:
    """Full curated real-machine benchmark suite meets accuracy & physical-trend guardrails."""
    rep = run_shaper_benchmarks()
    s_all = rep["summary"]["all"]
    s_t1 = rep["summary"]["tier1"]

    assert s_all["machines_evaluated"] >= 20
    assert s_t1["machines_evaluated"] >= 15
    assert s_all["overall_mape_pct"] is not None
    assert s_t1["overall_mape_pct"] is not None

    # Guardrail against major physics regressions (tighten as models improve)
    assert s_all["overall_mape_pct"] < 15.0
    assert s_t1["overall_mape_pct"] < 12.0

    # CoreXY-only architecture filter
    corexy_rep = run_shaper_benchmarks(architectures=["corexy"])
    assert corexy_rep["summary"]["all"]["machines_evaluated"] >= 20
    assert all(m["architecture"] == "corexy" for m in corexy_rep["machines"])
    assert corexy_rep["summary"]["all"]["overall_mape_pct"] < 12.0

    # Controlled same-machine A/B sweeps must preserve strong directional and ratio accuracy
    sweeps = rep["controlled_sweeps"]
    assert len(sweeps) >= 10
    dir_matches = sum(1 for sw in sweeps if sw["direction_match"])
    assert dir_matches >= len(sweeps) - 1, (
        f"Too many controlled sweeps flipped direction: {dir_matches}/{len(sweeps)}"
    )
    assert rep["summary"]["sweeps_ratio_mape_pct"] < 8.0


def test_compare_benchmark_reports_tracks_improvements_and_regressions() -> None:
    """Baseline diff engine detects improved, regressed, and unchanged axes accurately."""
    baseline = {
        "generated_at": "2026-11-27T00:00:00+00:00",
        "summary": {
            "all": {"overall_mape_pct": 12.0},
            "tier1": {"overall_mape_pct": 11.0},
        },
        "machines": [
            {
                "id": "m1",
                "name": "Machine 1",
                "tier": 1,
                "predicted_x_hz": 55.0,
                "x_error_pct": -10.0,
                "predicted_y_hz": 44.0,
                "y_error_pct": +10.0,
            },
        ],
    }
    current = {
        "generated_at": "2026-11-27T01:00:00+00:00",
        "summary": {
            "all": {"overall_mape_pct": 9.5},
            "tier1": {"overall_mape_pct": 8.5},
        },
        "machines": [
            {
                "id": "m1",
                "name": "Machine 1",
                "tier": 1,
                "predicted_x_hz": 58.0,
                "x_error_pct": -5.0,  # improved by 5%
                "predicted_y_hz": 46.0,
                "y_error_pct": +15.0,  # regressed by 5%
            },
        ],
    }
    diff = compare_benchmark_reports(baseline, current)
    assert diff["overall_mape_delta_pct"] == -2.5
    assert diff["tier1_mape_delta_pct"] == -2.5
    assert diff["improved_axes"] == 1
    assert diff["regressed_axes"] == 1
    assert diff["unchanged_axes"] == 0
    assert diff["machines"][0]["x_status"] == "improved"
    assert diff["machines"][0]["y_status"] == "regressed"


def test_cli_shaper_benchmark_smoke_and_compare(tmp_path) -> None:
    """CLI `printsim shaper-benchmark` supports filtering, --save, --compare, and --json."""
    runner = CliRunner()
    snap = tmp_path / "baseline.json"

    res1 = runner.invoke(
        app,
        [
            "shaper-benchmark",
            "--machine",
            "voron_trident_300_jeffk_hot",
            "--no-sweeps",
            "--save",
            str(snap),
            "--json",
        ],
    )
    assert res1.exit_code == 0, res1.output
    data1 = json.loads(res1.output)
    assert data1["summary"]["all"]["machines_evaluated"] == 1
    assert snap.exists()

    res2 = runner.invoke(
        app,
        [
            "shaper-benchmark",
            "--machine",
            "voron_trident_300_jeffk_hot",
            "--no-sweeps",
            "--compare",
            str(snap),
            "--json",
        ],
    )
    assert res2.exit_code == 0, res2.output
    data2 = json.loads(res2.output)
    assert data2["diff"]["overall_mape_delta_pct"] == 0.0
    assert data2["diff"]["regressed_axes"] == 0
