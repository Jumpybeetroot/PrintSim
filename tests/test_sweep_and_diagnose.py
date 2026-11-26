"""Tests for offline diagnostics, A/B comparison, and SQLite-cached batch sweeps."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from printsim.engine.diagnostics import (
    apply_dot_overrides,
    compare_printers,
    diagnose_printer,
)
from printsim.engine.sweep import _parse_xbeam_stdout, run_sweep
from printsim.schema import PrinterSpec

EXAMPLES_COREXY = Path(__file__).resolve().parents[1] / "examples" / "corexy_voron_style.yaml"
EXAMPLE_SWEEP = Path(__file__).resolve().parents[1] / "sweeps" / "example_parametric_sweep.yaml"


def test_apply_dot_overrides_dict_and_cli_parsing() -> None:
    base = PrinterSpec.from_yaml(EXAMPLES_COREXY)

    # Dict overrides + nested path
    updated_dict = apply_dot_overrides(
        base,
        {
            "gantry.x_beam_profile": "cf_tube_20x20x2",
            "gantry.belt.pretension_n": 42.5,
            "frame.bracing": "full_truss",
        },
    )
    assert updated_dict.gantry.x_beam_profile == "cf_tube_20x20x2"
    assert updated_dict.gantry.belt.pretension_n == pytest.approx(42.5)
    assert updated_dict.frame.bracing == "full_truss"
    # Base spec must remain untouched
    assert base.gantry.x_beam_profile == "extrusion_2020"
    assert base.frame.bracing == "corner_gussets"

    # CLI string list overrides: bool, int, float, str, list, nested
    updated_cli = apply_dot_overrides(
        base,
        [
            "gantry.x_beam_profile=cf_tube_20x20x2",
            "toolhead.hotend_brace=false",
            "gantry.belt.pulley_teeth=16",
            "frame.panel_thickness_mm=4.5",
            "kinematics.build_volume_mm=[350, 350, 320]",
            "frame.bracing=none",
        ],
    )
    assert updated_cli.gantry.x_beam_profile == "cf_tube_20x20x2"
    assert updated_cli.toolhead.use_hotend_bottom_brace is False
    assert updated_cli.gantry.belt.pulley_teeth == 16
    assert isinstance(updated_cli.gantry.belt.pulley_teeth, int)
    assert updated_cli.frame.panel_thickness_mm == pytest.approx(4.5)
    assert updated_cli.kinematics.build_volume_mm == (350.0, 350.0, 320.0)
    assert updated_cli.frame.bracing == "none"

    with pytest.raises(ValueError):
        apply_dot_overrides(base, ["gantry.nonexistent_field=123"])
    with pytest.raises(ValueError):
        apply_dot_overrides(base, ["missing_equals"])


def test_diagnose_printer_dict_and_compact_markdown() -> None:
    diag = diagnose_printer(EXAMPLES_COREXY)
    payload = diag.to_dict()

    for key in (
        "name",
        "architecture",
        "overall_score",
        "total_mass_kg",
        "moving_mass_x_g",
        "moving_mass_y_g",
        "bom_cost_usd",
        "fundamental_freq_hz",
        "shaper_x",
        "shaper_y",
        "recommended_max_accel_mm_s2",
        "static_tip_deflection_um",
        "static_stiffness_n_per_mm",
        "toolhead_breakdown",
        "top_modes",
        "top_adjoint_hotspots",
        "diagnostics",
    ):
        assert key in payload

    assert payload["architecture"] == "corexy"
    assert payload["overall_score"] > 0.0
    assert payload["fundamental_freq_hz"] > 10.0
    assert set(payload["shaper_x"].keys()) >= {"type", "freq_hz", "max_accel_mm_s2", "limiting_cause"}
    assert set(payload["shaper_y"].keys()) >= {"type", "freq_hz", "max_accel_mm_s2", "limiting_cause"}
    assert set(payload["static_tip_deflection_um"].keys()) == {"x", "y", "z", "worst"}
    assert set(payload["static_stiffness_n_per_mm"].keys()) == {"x", "y", "z"}
    assert 1 <= len(payload["top_modes"]) <= 6
    assert set(payload["top_modes"][0].keys()) == {
        "mode_num",
        "freq_hz",
        "dominant_axis",
        "description",
        "root_cause",
    }
    assert len(payload["top_adjoint_hotspots"]) <= 5
    if payload["top_adjoint_hotspots"]:
        assert set(payload["top_adjoint_hotspots"][0].keys()) == {
            "group",
            "compliance_share_pct",
            "hint",
        }
    assert payload["toolhead_breakdown"]["effective_mass_g"] > 0.0

    md = diag.to_markdown()
    md_lines = md.splitlines()
    assert len(md_lines) <= 45
    assert "## Top Modes" in md
    assert "## Adjoint Hotspots" in md

    diag_gcode = diagnose_printer(EXAMPLES_COREXY, include_gcode=True)
    assert diag_gcode.virtual_print is not None
    assert "print_quality_score" in diag_gcode.virtual_print
    assert len(diag_gcode.to_markdown().splitlines()) <= 45


def test_compare_printers_full_truss_deltas() -> None:
    cmp = compare_printers(
        EXAMPLES_COREXY,
        overrides=["frame.bracing=full_truss"],
    )
    d = cmp.to_dict()
    expected_metrics = {
        "overall_score",
        "fundamental_freq_hz",
        "shaper_x_hz",
        "shaper_y_hz",
        "rec_max_accel_mm_s2",
        "static_stiffness_x_n_per_mm",
        "static_stiffness_y_n_per_mm",
        "moving_mass_x_g",
        "moving_mass_y_g",
        "total_mass_kg",
        "max_volumetric_flow_mm3_s",
        "bom_cost_usd",
    }
    assert set(d["metrics"].keys()) == expected_metrics

    # Full truss bracing must raise fundamental frequency, total frame mass, and BOM cost
    assert d["metrics"]["fundamental_freq_hz"]["delta"] > 0.0
    assert d["metrics"]["total_mass_kg"]["delta"] > 0.0
    assert d["metrics"]["bom_cost_usd"]["delta"] > 0.0
    for row in d["metrics"].values():
        assert row["delta"] == pytest.approx(row["variant"] - row["base"], abs=1e-3)

    md = cmp.to_markdown()
    assert "# Comparison:" in md
    assert "Fundamental Freq" in md


def test_run_sweep_grid_parallel_and_sqlite_cache(tmp_path: Path) -> None:
    cfg_path = tmp_path / "grid_sweep.yaml"
    cache_db = tmp_path / "cache" / "sweep_cache.sqlite"
    out_dir = tmp_path / "out"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "name": "xbeam_grid_test",
                "mode": "printer",
                "base_spec": str(EXAMPLES_COREXY),
                "include_gcode": False,
                "grid": {
                    "gantry.x_beam_profile": [
                        "extrusion_2020",
                        "cf_tube_20x20x2",
                        "cf_tube_20x20_carbonara",
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    # First run: 3 uncached cases (exercises ProcessPoolExecutor path since len(cases) == 3)
    report1 = run_sweep(cfg_path, workers=2, use_cache=True, cache_db=cache_db, out_dir=out_dir)
    assert report1.total_cases == 3
    assert report1.cache_hits == 0
    assert report1.cache_misses == 3
    assert all(c.status == "ok" and not c.cached for c in report1.cases)

    results_json = out_dir / "results.json"
    report_md = out_dir / "report.md"
    assert results_json.is_file()
    assert report_md.is_file()
    parsed_json = json.loads(results_json.read_text(encoding="utf-8"))
    assert parsed_json["total_cases"] == 3
    assert len(parsed_json["cases"]) == 3
    assert "# Sweep Report: xbeam_grid_test" in report_md.read_text(encoding="utf-8")

    # Second run: 100% SQLite cache hits
    report2 = run_sweep(cfg_path, workers=2, use_cache=True, cache_db=cache_db, out_dir=out_dir)
    assert report2.total_cases == 3
    assert report2.cache_hits == 3
    assert report2.cache_misses == 0
    assert all(c.cached for c in report2.cases)
    for c1, c2 in zip(report1.cases, report2.cases):
        assert c1.metrics["overall_score"] == c2.metrics["overall_score"]
        assert c1.metrics["moving_mass_x_g"] == c2.metrics["moving_mass_x_g"]


def test_example_sweep_yaml_valid_and_xbeam_mode_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Ensure committed example sweep YAML resolves its base_spec and overrides cleanly
    example_cfg = yaml.safe_load(EXAMPLE_SWEEP.read_text(encoding="utf-8"))
    assert example_cfg["mode"] == "printer"
    assert len(example_cfg["cases"]) == 4
    base = PrinterSpec.from_yaml(EXAMPLES_COREXY)
    for case in example_cfg["cases"]:
        apply_dot_overrides(base, case["overrides"])

    # Verify xbeam_3d_fea mode execution + script SHA-256 SQLite caching
    calls: list[list[str]] = []

    def _fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "=== [1] Bare Validation ===\n"
                "{'label': 'cf_carbonara', 'err_pct': 0.8}\n"
                "{'label': 'al_2020', 'err_pct': 1.1}\n"
            ),
            stderr="",
        )

    monkeypatch.setattr("printsim.engine.sweep.subprocess.run", _fake_run)
    xb_cfg = tmp_path / "xbeam_sweep.yaml"
    xb_cfg.write_text(
        yaml.safe_dump({"name": "xbeam_smoke", "mode": "xbeam_3d_fea", "only": ["bare"]}),
        encoding="utf-8",
    )
    cache_db = tmp_path / "xb_cache.sqlite"
    out_dir = tmp_path / "xb_out"

    r1 = run_sweep(xb_cfg, cache_db=cache_db, out_dir=out_dir)
    assert r1.total_cases == 1
    assert r1.cache_hits == 0
    assert r1.cases[0].metrics["record_count"] == 2
    assert len(calls) == 1

    r2 = run_sweep(xb_cfg, cache_db=cache_db, out_dir=out_dir)
    assert r2.cache_hits == 1
    assert len(calls) == 1
    assert _parse_xbeam_stdout("{'a': 1}\nnot a dict\n") == [{"a": 1}]
