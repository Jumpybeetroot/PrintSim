"""Tests for Sign-Off Verification (fine-mesh convergence + true workspace 3D FEA + 90% CI)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

import printsim.engine.signoff as signoff_module
from printsim.api.server import create_app
from printsim.cli import app as cli_app
from printsim.engine.evaluator import evaluate_printer
from printsim.engine.signoff import (
    _accel_sigma_from_freq,
    _apply_spec_field_scale,
    _build_readiness_checks,
    _extract_core_metrics,
    _run_uncertainty_analysis,
    _run_workspace_fea_sweep,
    _trust_to_sigma_rel,
    run_signoff_verification,
)
from printsim.schema.printer import PrinterSpec
from printsim.mcp_server import handle_jsonrpc_message
from printsim.wasm_dispatch import dispatch_api_request


def test_trust_to_sigma_rel_monotonic_and_override_tightens() -> None:
    s_high = _trust_to_sigma_rel(95.0)
    s_mid = _trust_to_sigma_rel(75.0)
    s_low = _trust_to_sigma_rel(45.0)
    assert 0.02 <= s_high < s_mid < s_low <= 0.32
    assert _trust_to_sigma_rel(60.0, overridden=True) < _trust_to_sigma_rel(60.0, overridden=False)


def test_signoff_verification_physics_and_structure() -> None:
    spec = PrinterSpec.from_yaml("examples/corexy_voron_style.yaml")
    res = run_signoff_verification(spec, deep_mesh_n=4, deep_bed_mesh_n=8, workspace_grid_n=2)

    assert res["verdict"] in ("pass", "caution", "fail")
    assert len(res["checks"]) == 5
    assert res["total_fea_solves"] >= 6

    ws = res["workspace_fea"]
    assert ws["grid_n"] == 2
    assert len(ws["min_axis_grid_hz"]) == 2
    assert len(ws["min_axis_grid_hz"][0]) == 2
    assert len(ws["z_sweep"]) == 3
    # Worst-case across the real 3D workspace must be <= center
    assert ws["worst_case_min_axis_hz"] <= ws["center_min_axis_hz"] + 1e-6
    assert ws["worst_case_safe_accel_mm_s2"] <= ws["center_safe_accel_mm_s2"] + 1e-6

    unc = res["uncertainty"]
    bands = unc["bands"]
    for key in ("shaper_x_hz", "shaper_y_hz", "min_axis_hz", "recommended_max_accel_mm_s2", "combined_max_deflection_um"):
        b = bands[key]
        assert b["p05"] <= b["nominal"] <= b["p95"]
        assert b["ci90_half_width"] >= 0.0

    drivers = unc["drivers"]
    assert len(drivers) >= 1
    total_share = sum(d["variance_share_pct"] for d in drivers)
    assert 99.0 <= total_share <= 101.0


def test_signoff_higher_uncertainty_for_uncalibrated_build() -> None:
    """Adding low-trust components (printed Tap + generic rails + rubber feet) widens 90% CI."""
    base = PrinterSpec.from_yaml("examples/corexy_voron_style.yaml")
    uncertain = base.model_copy(deep=True)
    uncertain.frame.feet_type = "anti_vib_pads"
    uncertain.gantry.rail_quality = "budget_generic"
    uncertain.toolhead.z_probe_type = "printed_tap"

    r_base = run_signoff_verification(
        base, deep_mesh_n=4, deep_bed_mesh_n=8, workspace_grid_n=2, include_z_extremes=False
    )
    r_unc = run_signoff_verification(
        uncertain, deep_mesh_n=4, deep_bed_mesh_n=8, workspace_grid_n=2, include_z_extremes=False
    )

    assert r_unc["uncertainty"]["overall_trust_pct"] < r_base["uncertainty"]["overall_trust_pct"]
    assert (
        r_unc["uncertainty"]["bands"]["combined_max_deflection_um"]["ci90_pct"]
        > r_base["uncertainty"]["bands"]["combined_max_deflection_um"]["ci90_pct"]
    )


def test_signoff_api_and_wasm_dispatch() -> None:
    spec = PrinterSpec.from_yaml("examples/corexy_voron_style.yaml")
    payload = {
        "spec": spec.model_dump(mode="json"),
        "deep_mesh_n": 4,
        "deep_bed_mesh_n": 8,
        "workspace_grid_n": 2,
        "include_z_extremes": False,
        "max_uncertainty_groups": 3,
    }

    # FastAPI endpoint
    client = TestClient(create_app())
    resp = client.post("/api/verify/signoff", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["verdict"] in ("pass", "caution", "fail")
    assert "workspace_fea" in body and "uncertainty" in body and "deep_simulation" in body

    # Pyodide WASM dispatcher parity
    wasm_raw = dispatch_api_request("POST", "/api/verify/signoff", json.dumps(payload))
    assert wasm_raw["status"] == 200
    wasm_body = json.loads(wasm_raw["body_text"])
    assert wasm_body["verdict"] == body["verdict"]
    assert wasm_body["uncertainty"]["bands"]["min_axis_hz"]["nominal"] == body["uncertainty"]["bands"]["min_axis_hz"]["nominal"]


def test_override_never_raises_sigma() -> None:
    """An override tightens or keeps sigma at every trust level, including above the 3% floor."""
    for trust in range(0, 101):
        assert _trust_to_sigma_rel(trust, overridden=True) <= _trust_to_sigma_rel(trust)


def test_workspace_sweep_even_grid_includes_center() -> None:
    """Even grids have no centre cell; the centre must still be solved, counted, and compete."""
    spec = PrinterSpec.from_yaml("examples/corexy_voron_style.yaml")
    ws = _run_workspace_fea_sweep(spec, "pla", grid_n=2, include_z_extremes=False)
    assert ws["xy_position_modeled"] is True
    assert ws["num_fea_solves"] == 5  # 2x2 grid + centre
    assert ws["worst_case_min_axis_hz"] <= ws["center_min_axis_hz"] + 1e-6
    assert ws["worst_case_safe_accel_mm_s2"] <= ws["center_safe_accel_mm_s2"] + 1e-6


def test_workspace_sweep_delta_skips_unmodeled_xy() -> None:
    """The delta builder ignores XY: solve the centre once and sweep only the Z it really uses."""
    spec = PrinterSpec.from_yaml("examples/delta_racer.yaml")
    ws = _run_workspace_fea_sweep(spec, "pla", grid_n=3)
    assert ws["xy_position_modeled"] is False
    assert ws["num_fea_solves"] == 3  # centre + two Z stations
    assert [z["z_rel"] for z in ws["z_sweep"]] == [0.1, 0.5, 0.8]
    flat = [hz for row in ws["min_axis_grid_hz"] for hz in row]
    assert len(flat) == 9 and len(set(flat)) == 1


_NOMINAL = {
    "shaper_x_hz": 54.0,
    "shaper_y_hz": 37.0,
    "min_axis_hz": 37.0,
    "fundamental_freq_hz": 42.0,
    "recommended_max_accel_mm_s2": 5400.0,
    "combined_max_deflection_um": 79.0,
}


def test_uncertainty_perturbs_every_group_unless_capped() -> None:
    """Trust does not rank influence: the default must not drop the dominant (belt) driver."""
    spec = PrinterSpec.from_yaml("examples/corexy_voron_style.yaml")
    full = _run_uncertainty_analysis(spec, _NOMINAL, "pla")
    assert full["skipped_groups"] == []
    driver_ids = {d["group_id"] for d in full["drivers"]}
    assert {"belt", "toolhead_preset"} <= driver_ids

    capped = _run_uncertainty_analysis(spec, _NOMINAL, "pla", max_groups=1)
    assert len(capped["drivers"]) == 1
    assert len(capped["skipped_groups"]) == len(full["drivers"]) - 1
    assert capped["bands"]["min_axis_hz"]["ci90_pct"] < full["bands"]["min_axis_hz"]["ci90_pct"]


def test_accel_sigma_follows_frequency_squared_law() -> None:
    """Closed form: with one limiting axis, 0.5 * ((1 + r)^2 - (1 - r)^2) = 2r exactly."""
    # Y limits (5400 << 9300): only the Y frequency response matters.
    assert _accel_sigma_from_freq(5400.0, 9300.0, 5400.0, 0.20, 0.03) == pytest.approx(
        2.0 * 0.03 * 5400.0, rel=1e-12
    )
    # The sign of the response does not matter, and a derated nominal scales the result.
    assert _accel_sigma_from_freq(5000.0, 9300.0, 5400.0, 0.0, -0.03) == pytest.approx(
        2.0 * 0.03 * 5000.0, rel=1e-12
    )
    # Equal axes: the softer response on each side sets the limit (min of the two laws).
    sigma = _accel_sigma_from_freq(1000.0, 1000.0, 1000.0, 0.10, 0.0)
    assert sigma == pytest.approx(0.5 * (1.0 - 0.9**2) * 1000.0, rel=1e-12)
    # Degenerate inputs never divide by zero.
    assert _accel_sigma_from_freq(0.0, 0.0, 0.0, 0.1, 0.1) == 0.0


def test_accel_band_is_twice_the_limiting_frequency_band() -> None:
    """End to end: Y limits the Voron-style safe accel, so its band is 2x the shaper-Y band."""
    spec = PrinterSpec.from_yaml("examples/corexy_voron_style.yaml")
    nominal = {**_NOMINAL, "accel_x_mm_s2": 9300.0, "accel_y_mm_s2": 5400.0}
    bands = _run_uncertainty_analysis(spec, nominal, "pla")["bands"]
    accel_pct = bands["recommended_max_accel_mm_s2"]["ci90_pct"]
    assert accel_pct == pytest.approx(2.0 * bands["shaper_y_hz"]["ci90_pct"], rel=0.1)


def test_toolhead_mass_perturbation_lowers_resonance() -> None:
    """Physical sanity: a heavier toolhead on the same structure resonates lower (f ~ 1/sqrt(m))."""
    spec = PrinterSpec.from_yaml("examples/corexy_voron_style.yaml")
    fields = [("toolhead.total_mass_g", 20.5, 1.0e5)]
    light = _apply_spec_field_scale(spec, fields, 0.85)
    heavy = _apply_spec_field_scale(spec, fields, 1.15)
    assert light is not None and heavy is not None
    assert heavy.toolhead.total_mass_g == pytest.approx(1.15 * spec.toolhead.total_mass_g)
    assert spec.toolhead.total_mass_g == pytest.approx(light.toolhead.total_mass_g / 0.85)

    def shaper_x_hz(s: PrinterSpec) -> float:
        report = evaluate_printer(s, panel_mesh_n=2, bed_mesh_n=8, include_advanced_analysis=False)
        return _extract_core_metrics(report)["shaper_x_hz"]

    assert shaper_x_hz(heavy) < shaper_x_hz(light)
    # A missing or non-positive field is not perturbable.
    assert _apply_spec_field_scale(spec, [("toolhead.no_such_field", 0.0, 1.0)], 1.1) is None


def _readiness_inputs(
    *,
    freq_drop: float = 1.0,
    accel_drop: float = 1.0,
    belt_peak_n: float = 20.0,
    belt_max_n: float = 60.0,
    skipped: list[str] | None = None,
) -> tuple[dict, dict, dict]:
    deep_verify = {
        "convergence": {"freq_diff_pct": 0.5, "deflection_diff_pct": 0.5},
        "deep_summary": {"bed_flatness_error_um": 20.0},
        "deep_simulation": {
            "collision": {"has_collision": False},
            "static_fea": {"min_safety_factor": 50.0},
            "kinematics": {
                "belt_peak_tension_n": belt_peak_n,
                "belt_max_working_tension_n": belt_max_n,
                "belt_overtension_risk": False,
                "bracket_creep_risk": False,
            },
        },
    }
    workspace = {
        "worst_freq_drop_pct": freq_drop,
        "worst_accel_drop_pct": accel_drop,
        "worst_freq_pos": [0.15, 0.85, 0.5],
        "worst_accel_pos": [0.5, 0.5, 0.9],
        "worst_case_safe_accel_mm_s2": 4500.0,
        "xy_position_modeled": True,
    }
    uncertainty = {
        "overall_trust_pct": 85,
        "bands": {"min_axis_hz": {"ci90_pct": 3.0}},
        "drivers": [{"label": "XY Drive Belt Modulus"}],
        "skipped_groups": skipped or [],
    }
    return deep_verify, workspace, uncertainty


def _check(checks: list[dict], check_id: str) -> dict:
    return next(c for c in checks if c["id"] == check_id)


def test_readiness_all_pass_baseline() -> None:
    verdict, checks = _build_readiness_checks(*_readiness_inputs())
    assert verdict == "pass"
    assert {c["status"] for c in checks} == {"pass"}


def test_readiness_workspace_gate_uses_accel_drop_and_its_position() -> None:
    """A safe-accel drop trips the gate even when resonance is flat, at its own position."""
    verdict, checks = _build_readiness_checks(*_readiness_inputs(freq_drop=0.0, accel_drop=17.4))
    ws = _check(checks, "workspace_uniformity")
    assert verdict == "caution" and ws["status"] == "caution"
    assert "safe accel -17.4% at (u=0.50, v=0.50, z=0.90)" in ws["detail"]
    assert "resonance -0.0% at (u=0.15, v=0.85, z=0.50)" in ws["detail"]

    _, checks = _build_readiness_checks(*_readiness_inputs(freq_drop=0.0, accel_drop=25.0))
    assert _check(checks, "workspace_uniformity")["status"] == "fail"


def test_readiness_capped_uncertainty_cannot_pass() -> None:
    _, checks = _build_readiness_checks(*_readiness_inputs(skipped=["XY Drive Belt Modulus"]))
    unc = _check(checks, "prediction_uncertainty")
    assert unc["status"] == "caution"
    assert "XY Drive Belt Modulus" in unc["detail"]


def test_readiness_belt_margin_limits_and_text() -> None:
    """Fail only below the rated working tension (SF < 1.0); 1.0-1.25 is caution; text truncates."""
    # SF 0.999: over the rated working tension. Must not round up to a passing-looking 1.00.
    verdict, checks = _build_readiness_checks(*_readiness_inputs(belt_peak_n=58.06, belt_max_n=58.0))
    mech = _check(checks, "mechanical_margins")
    assert verdict == "fail" and mech["status"] == "fail"
    assert "Belt 0.99×" in mech["value"]
    assert "belt safety factor 0.99× is below 1.00× (58.1 N peak vs 58.0 N working limit)" in mech["detail"]
    assert "bed flatness" not in mech["detail"]

    # SF 1.099 (monolith_zero_m0 class): thin margin, not a failure.
    _, checks = _build_readiness_checks(*_readiness_inputs(belt_peak_n=52.78, belt_max_n=58.0))
    mech = _check(checks, "mechanical_margins")
    assert mech["status"] == "caution" and "Belt 1.09×" in mech["value"]

    # SF 1.249 is still caution; 1.25 passes.
    _, checks = _build_readiness_checks(*_readiness_inputs(belt_peak_n=80.06, belt_max_n=100.0))
    assert _check(checks, "mechanical_margins")["status"] == "caution"
    _, checks = _build_readiness_checks(*_readiness_inputs(belt_peak_n=80.0, belt_max_n=100.0))
    assert _check(checks, "mechanical_margins")["status"] == "pass"


def test_readiness_uncertainty_band_limits() -> None:
    """Band on the limiting shaper frequency: pass <= 5%, caution <= 10%, fail above."""
    for band_pct, expected in ((5.0, "pass"), (5.1, "caution"), (10.0, "caution"), (10.1, "fail")):
        deep_verify, workspace, uncertainty = _readiness_inputs()
        uncertainty["bands"]["min_axis_hz"]["ci90_pct"] = band_pct
        _, checks = _build_readiness_checks(deep_verify, workspace, uncertainty)
        assert _check(checks, "prediction_uncertainty")["status"] == expected, band_pct


def test_workspace_sweep_resolves_softest_cell_at_z_extremes() -> None:
    """The softest XY cell is re-solved at bottom/top Z unless it is the centre (already swept)."""
    spec = PrinterSpec.from_yaml("examples/monolith_zero_m0.yaml")
    ws = _run_workspace_fea_sweep(spec, "pla", grid_n=3)
    extra = ws["worst_xy_z_sweep"]
    assert len(extra) in (0, 2)
    # centre + 8 other grid cells + 2 centre-column Z stations + the softest-cell Z stations
    assert ws["num_fea_solves"] == 1 + 8 + 2 + len(extra)

    cells = [
        (hz, ws["coords"][c], ws["coords"][r])
        for r, row in enumerate(ws["min_axis_grid_hz"])
        for c, hz in enumerate(row)
    ]
    _, u_soft, v_soft = min(cells)
    softest_is_center = abs(u_soft - 0.5) < 1e-6 and abs(v_soft - 0.5) < 1e-6
    assert bool(extra) == (not softest_is_center)
    for point in extra:
        assert (point["u_rel"], point["v_rel"]) == (u_soft, v_soft)
        assert point["z_rel"] in (0.1, 0.9)
        assert ws["worst_case_min_axis_hz"] <= point["min_axis_hz"] + 1e-6


def test_uncertainty_drivers_report_both_variance_shares() -> None:
    """Drivers carry resonance and flex shares (each summing to 100%) and rank by the larger."""
    spec = PrinterSpec.from_yaml("examples/corexy_voron_style.yaml")
    drivers = _run_uncertainty_analysis(spec, _NOMINAL, "pla")["drivers"]
    assert sum(d["variance_share_pct"] for d in drivers) == pytest.approx(100.0, abs=1.0)
    assert sum(d["deflection_variance_share_pct"] for d in drivers) == pytest.approx(100.0, abs=1.0)
    ranks = [max(d["variance_share_pct"], d["deflection_variance_share_pct"]) for d in drivers]
    assert ranks == sorted(ranks, reverse=True)


def test_signoff_clamps_mesh_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every entry point is bounded: out-of-range mesh densities are clamped before solving."""
    spec = PrinterSpec.from_yaml("examples/corexy_voron_style.yaml")
    seen: dict[str, int] = {}

    class _Stop(Exception):
        pass

    def _record(_spec: PrinterSpec, **kwargs: object) -> dict:
        seen.update({k: v for k, v in kwargs.items() if k.startswith("deep_")})  # type: ignore[misc]
        raise _Stop

    monkeypatch.setattr(signoff_module, "verify_printer_deep", _record)
    with pytest.raises(_Stop):
        run_signoff_verification(spec, deep_mesh_n=500, deep_bed_mesh_n=500)
    assert seen == {"deep_mesh_n": 10, "deep_bed_mesh_n": 20}
    with pytest.raises(_Stop):
        run_signoff_verification(spec, deep_mesh_n=0, deep_bed_mesh_n=0)
    assert seen == {"deep_mesh_n": 4, "deep_bed_mesh_n": 8}


def test_signoff_cli_json_is_compact_and_honours_flags() -> None:
    result = CliRunner().invoke(
        cli_app,
        [
            "signoff",
            "examples/corexy_voron_style.yaml",
            "--json",
            "--mesh-n",
            "4",
            "--bed-mesh-n",
            "8",
            "--grid-n",
            "2",
            "--no-z-extremes",
        ],
    )
    assert result.exit_code == 0, result.output
    body = json.loads(result.output)
    assert body["verdict"] in ("pass", "caution", "fail")
    assert "deep_simulation" not in body
    assert body["workspace_fea"]["z_sweep"] == []
    assert body["workspace_fea"]["worst_xy_z_sweep"] == []


def test_signoff_mcp_tool_returns_compact_report() -> None:
    spec = PrinterSpec.from_yaml("examples/corexy_voron_style.yaml")
    resp = handle_jsonrpc_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "run_signoff_verification",
                "arguments": {
                    "yaml_text": spec.to_yaml(),
                    "deep_mesh_n": 4,
                    "deep_bed_mesh_n": 8,
                    "workspace_grid_n": 2,
                    "include_z_extremes": False,
                },
            },
        }
    )
    assert resp is not None
    assert resp["result"]["isError"] is False, resp["result"]["content"][0]["text"]
    body = json.loads(resp["result"]["content"][0]["text"])
    assert body["verdict"] in ("pass", "caution", "fail")
    assert len(body["checks"]) == 5
    assert "deep_simulation" not in body
    assert body["uncertainty"]["skipped_groups"] == []
