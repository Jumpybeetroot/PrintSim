"""Tests for Klipper ADXL345 / Shake&Tune CSV import and measured-vs-simulated PSD calibration."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from printsim.api.server import create_app
from printsim.cli import app as cli_app
from printsim.fea import (
    calibrate_printer_against_adxl,
    parse_adxl_csv,
    solve_modal_fea,
)
from printsim.geometry import build_assembly
from printsim.mcp_server import handle_jsonrpc_message
from printsim.schema import PrinterSpec

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def _make_klipper_lorentzian_csv(
    peak_hz: float,
    damping_ratio: float = 0.055,
    freqs_hz: np.ndarray | None = None,
    axis_col: str = "psd_x",
    include_all_cols: bool = True,
    noise_floor: float = 0.002,
) -> str:
    """Synthesize a Klipper TEST_RESONANCES / calibrate_shaper PSD CSV with an exact 2nd-order Lorentzian."""
    freqs = (
        np.arange(10.0, 200.5, 0.5, dtype=float)
        if freqs_hz is None
        else np.asarray(freqs_hz, dtype=float)
    )
    r = freqs / peak_hz
    r2 = r * r
    psd = (r2 / np.maximum((1.0 - r2) ** 2 + (2.0 * damping_ratio * r) ** 2, 1e-12)) * (
        (2.0 * damping_ratio) ** 2
    ) + noise_floor

    lines = [
        "# Klipper TEST_RESONANCES shaper_calibrate dump",
        f"# synthetic target peak={peak_hz:.2f} Hz zeta={damping_ratio:.4f}",
    ]
    if include_all_cols:
        lines.append(f"freq,psd_x,psd_y,psd_z,psd_xyz,mzv({peak_hz:.1f}),zv({peak_hz:.1f})")
        for f, p in zip(freqs, psd):
            px = p if axis_col == "psd_x" else 0.04 * p
            py = p if axis_col == "psd_y" else 0.04 * p
            pz = 0.02 * p
            pxyz = px + py + pz
            lines.append(f"{f:.2f},{px:.6f},{py:.6f},{pz:.6f},{pxyz:.6f},0.100,0.120")
    else:
        lines.append(f"freq,{axis_col}")
        for f, p in zip(freqs, psd):
            lines.append(f"{f:.2f},{p:.6f}")
    return "\n".join(lines)


def _make_raw_sine_csv(
    freq_hz: float,
    fs_hz: float = 1600.0,
    duration_s: float = 3.0,
    axis: str = "X",
    shorthand_cols: bool = False,
) -> str:
    """Synthesize a raw ADXL345 time-series CSV with a known sinusoid + gravity + sensor noise."""
    rng = np.random.default_rng(101)
    t = np.arange(0.0, duration_s, 1.0 / fs_hz, dtype=float)
    sig = 2.5 * np.sin(2.0 * math.pi * freq_hz * t) + rng.normal(0.0, 0.04, size=t.size)
    other = rng.normal(0.0, 0.02, size=t.size)
    z_grav = 9.81 + rng.normal(0.0, 0.02, size=t.size)

    ax = sig if axis.upper() == "X" else other
    ay = sig if axis.upper() == "Y" else other

    if shorthand_cols:
        lines = ["t,x,y,z"]
    else:
        lines = ["# Klipper raw accelerometer dump", "#time,accel_x,accel_y,accel_z"]
    for ti, xi, yi, zi in zip(t, ax, ay, z_grav):
        lines.append(f"{ti:.6f},{xi:.5f},{yi:.5f},{zi:.5f}")
    return "\n".join(lines)


def _make_damped_oscillator_csv(
    f0_hz: float,
    zeta: float = 0.05,
    fs_hz: float = 3200.0,
    duration_s: float = 2.0,
) -> str:
    """Emit raw ADXL CSV for an exact impulse-excited damped 2nd-order oscillator ringdown."""
    n = int(round(fs_hz * duration_s))
    dt = 1.0 / fs_hz
    t = np.arange(n, dtype=float) * dt
    w0 = 2.0 * math.pi * f0_hz
    wd = w0 * math.sqrt(max(1.0 - zeta * zeta, 1e-6))
    t_rel = np.maximum(t - 0.5, 0.0)
    sig = np.where(t >= 0.5, 5.0 * np.exp(-zeta * w0 * t_rel) * np.sin(wd * t_rel), 0.0)

    lines = ["time,accel_x,accel_y,accel_z"]
    for ti, vi in zip(t, sig):
        lines.append(f"{ti:.6f},{vi:.6f},0.001,9.81")
    return "\n".join(lines)


def test_parse_klipper_psd_csv_exact_peak_and_damping() -> None:
    """Klipper resonances_x.csv parsing recovers exact synthetic Lorentzian peak & damping."""
    csv_text = _make_klipper_lorentzian_csv(peak_hz=54.0, damping_ratio=0.065, noise_floor=0.0)
    parsed = parse_adxl_csv(csv_text, axis="X")

    assert parsed.source_format == "klipper_psd_csv"
    assert parsed.selected_column == "psd_x"
    assert parsed.sample_count == 381

    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    report = calibrate_printer_against_adxl(spec, axis_x_csv=csv_text)
    ax_x = report.axes["X"]

    assert ax_x.measured_peak_hz == pytest.approx(54.0, abs=0.1)
    assert ax_x.measured_damping_ratio == pytest.approx(0.065, abs=0.003)

    # Fallback to psd_xyz when psd_x is absent
    xyz_only_csv = _make_klipper_lorentzian_csv(
        peak_hz=72.5, damping_ratio=0.05, axis_col="psd_xyz", include_all_cols=False
    )
    parsed_xyz = parse_adxl_csv(xyz_only_csv, axis="X")
    assert parsed_xyz.selected_column == "psd_xyz"
    assert float(parsed_xyz.freqs_hz[int(np.argmax(parsed_xyz.psd))]) == pytest.approx(72.5, abs=0.1)


def test_parse_raw_adxl_csv_welch_recovers_sine_and_damped_oscillator() -> None:
    """Raw time-series ADXL CSV Welch PSD recovers known sine and damped oscillator frequencies."""
    # 1. Klipper #time,accel_x,accel_y,accel_z format with 68 Hz sine
    sine_csv = _make_raw_sine_csv(freq_hz=68.0, fs_hz=1600.0, duration_s=3.0, axis="X")
    parsed_sine = parse_adxl_csv(sine_csv, axis="X")
    assert parsed_sine.source_format == "raw_accel_csv"
    assert parsed_sine.selected_column == "accel_x"
    assert parsed_sine.sample_rate_hz == pytest.approx(1600.0, rel=1e-3)
    band = (parsed_sine.freqs_hz >= 5.0) & (parsed_sine.freqs_hz <= 200.0)
    peak_sine_hz = float(parsed_sine.freqs_hz[band][int(np.argmax(parsed_sine.psd[band]))])
    assert peak_sine_hz == pytest.approx(68.0, abs=1.0)

    # 2. Shorthand t,x,y,z format on Y axis with 92 Hz sine
    y_csv = _make_raw_sine_csv(
        freq_hz=92.0, fs_hz=1600.0, duration_s=3.0, axis="Y", shorthand_cols=True
    )
    parsed_y = parse_adxl_csv(y_csv, axis="Y")
    assert parsed_y.source_format == "raw_accel_csv"
    assert parsed_y.selected_column == "y"
    band_y = (parsed_y.freqs_hz >= 5.0) & (parsed_y.freqs_hz <= 200.0)
    peak_y_hz = float(parsed_y.freqs_hz[band_y][int(np.argmax(parsed_y.psd[band_y]))])
    assert peak_y_hz == pytest.approx(92.0, abs=1.0)

    # 3. Impulse-excited damped 2nd-order oscillator at 75 Hz, zeta=0.05
    osc_csv = _make_damped_oscillator_csv(f0_hz=75.0, zeta=0.05, fs_hz=3200.0, duration_s=2.0)
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    osc_rep = calibrate_printer_against_adxl(spec, axis_x_csv=osc_csv)
    ax_osc = osc_rep.axes["X"]
    assert ax_osc.source_format == "raw_accel_csv"
    assert ax_osc.measured_peak_hz == pytest.approx(75.0, abs=1.0)
    assert ax_osc.measured_damping_ratio == pytest.approx(0.05, abs=0.01)


def test_calibrate_printer_against_adxl_contract_and_diagnostics() -> None:
    """End-to-end calibration validates exact JSON contract and stiffer/softer/damping diagnostics."""
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    modal = solve_modal_fea(build_assembly(spec))
    assert modal.shaper_graph_x is not None
    assert modal.shaper_graph_y is not None

    sim_x_peak = float(modal.shaper_graph_x.peak_freq_hz)
    sim_x_zeta = float(modal.shaper_graph_x.damping_ratio)
    sim_y_peak = float(modal.shaper_graph_y.peak_freq_hz)

    # Case 1: Close match on both X and Y -> high spectral correlation, small error
    csv_match_x = _make_klipper_lorentzian_csv(
        peak_hz=sim_x_peak, damping_ratio=sim_x_zeta, axis_col="psd_x"
    )
    csv_match_y = _make_klipper_lorentzian_csv(
        peak_hz=sim_y_peak, damping_ratio=sim_x_zeta, axis_col="psd_y"
    )
    rep_match = calibrate_printer_against_adxl(spec, axis_x_csv=csv_match_x, axis_y_csv=csv_match_y)
    d = rep_match.to_dict()

    assert set(d.keys()) == {"printer_name", "axes", "overall_summary", "recommended_adjustments"}
    assert d["printer_name"] == spec.name
    assert set(d["axes"].keys()) == {"X", "Y"}

    expected_axis_keys = {
        "axis",
        "source_format",
        "freqs_hz",
        "measured_psd_norm",
        "simulated_psd_norm",
        "measured_peak_hz",
        "simulated_peak_hz",
        "peak_delta_hz",
        "peak_error_pct",
        "measured_damping_ratio",
        "simulated_damping_ratio",
        "measured_shaper",
        "measured_shaper_freq_hz",
        "measured_max_accel_mm_s2",
        "simulated_shaper",
        "simulated_shaper_freq_hz",
        "simulated_max_accel_mm_s2",
        "spectral_correlation",
        "sample_count",
        "diagnostics",
    }
    ax_x = d["axes"]["X"]
    assert set(ax_x.keys()) == expected_axis_keys
    assert ax_x["axis"] == "X"
    assert ax_x["source_format"] == "klipper_psd_csv"
    assert len(ax_x["freqs_hz"]) == len(ax_x["measured_psd_norm"]) == len(ax_x["simulated_psd_norm"])
    assert ax_x["freqs_hz"][-1] <= 200.0 + 1e-6
    assert max(ax_x["measured_psd_norm"]) == pytest.approx(1.0, abs=1e-4)
    assert max(ax_x["simulated_psd_norm"]) == pytest.approx(1.0, abs=1e-4)
    assert min(ax_x["measured_psd_norm"]) >= 0.0
    assert ax_x["spectral_correlation"] > 0.85
    assert abs(ax_x["peak_error_pct"]) < 2.0

    # Case 2: Softer real frame/belts (-30% peak) + higher damping (+0.04 zeta)
    soft_peak = round(sim_x_peak * 0.70, 1)
    csv_soft = _make_klipper_lorentzian_csv(
        peak_hz=soft_peak, damping_ratio=sim_x_zeta + 0.04, axis_col="psd_x"
    )
    rep_soft = calibrate_printer_against_adxl(spec, axis_x_csv=csv_soft)
    sx = rep_soft.axes["X"]
    assert sx.peak_delta_hz < -5.0
    assert sx.peak_error_pct < -10.0
    joined_soft = " ".join(sx.diagnostics).lower()
    assert "belt" in joined_soft
    assert "preload" in joined_soft or "fastener" in joined_soft
    assert "damping" in joined_soft
    assert "feet" in joined_soft or "seal" in joined_soft

    # Case 3: Stiffer real build (+35% peak)
    stiff_peak = round(sim_x_peak * 1.35, 1)
    csv_stiff = _make_klipper_lorentzian_csv(
        peak_hz=stiff_peak, damping_ratio=sim_x_zeta, axis_col="psd_x"
    )
    rep_stiff = calibrate_printer_against_adxl(spec, axis_x_csv=csv_stiff)
    st = rep_stiff.axes["X"]
    assert st.peak_delta_hz > 5.0
    assert st.peak_error_pct > 10.0
    joined_stiff = " ".join(st.diagnostics).lower()
    assert "stiffer" in joined_stiff or "tighter belt" in joined_stiff


def test_fastapi_adxl_calibration_endpoint() -> None:
    """FastAPI POST /api/calibration/adxl returns 200 OK on valid CSVs and 400 on missing/malformed inputs."""
    client = TestClient(create_app())
    preset = client.get("/api/presets/corexy_voron_style").json()
    spec_json = preset["spec"]

    csv_x = _make_klipper_lorentzian_csv(peak_hz=54.0, damping_ratio=0.06, axis_col="psd_x")
    raw_y = _make_raw_sine_csv(freq_hz=45.0, fs_hz=1200.0, duration_s=2.0, axis="Y")

    r_ok = client.post(
        "/api/calibration/adxl",
        json={
            "spec": spec_json,
            "axis_x_csv": csv_x,
            "axis_y_csv": raw_y,
            "max_freq_hz": 200.0,
        },
    )
    assert r_ok.status_code == 200
    payload = r_ok.json()
    assert payload["printer_name"] == spec_json["name"]
    assert payload["axes"]["X"]["source_format"] == "klipper_psd_csv"
    assert payload["axes"]["Y"]["source_format"] == "raw_accel_csv"
    assert payload["axes"]["X"]["measured_peak_hz"] == pytest.approx(54.0, abs=0.2)
    assert payload["axes"]["Y"]["measured_peak_hz"] == pytest.approx(45.0, abs=1.0)

    # Error 400 when neither CSV is provided
    r_missing = client.post(
        "/api/calibration/adxl",
        json={"spec": spec_json},
    )
    assert r_missing.status_code == 400

    # Error 400 when CSV is malformed
    r_bad = client.post(
        "/api/calibration/adxl",
        json={"spec": spec_json, "axis_x_csv": "foo,bar\n1,2\n"},
    )
    assert r_bad.status_code == 400


def test_cli_and_mcp_adxl_calibration(tmp_path: Path) -> None:
    """CLI `printsim calibrate-adxl` and MCP `calibrate_adxl_csv` work end-to-end."""
    spec_path = EXAMPLES_DIR / "corexy_voron_style.yaml"
    x_path = tmp_path / "resonances_x.csv"
    x_path.write_text(_make_klipper_lorentzian_csv(peak_hz=52.0, damping_ratio=0.055), encoding="utf-8")

    runner = CliRunner()
    res_json = runner.invoke(cli_app, ["calibrate-adxl", str(spec_path), "--x-csv", str(x_path), "--json"])
    assert res_json.exit_code == 0, res_json.output
    data = json.loads(res_json.output)
    assert "X" in data["axes"]
    assert data["axes"]["X"]["measured_peak_hz"] == pytest.approx(52.0, abs=0.1)

    res_table = runner.invoke(cli_app, ["calibrate-adxl", str(spec_path), "--x-csv", str(x_path)])
    assert res_table.exit_code == 0, res_table.output
    assert "PrintSim ADXL345" in res_table.output

    # MCP tool discovery and call
    list_resp = handle_jsonrpc_message({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
    assert list_resp is not None
    tool_names = [t["name"] for t in list_resp["result"]["tools"]]
    assert "calibrate_adxl_csv" in tool_names

    call_resp = handle_jsonrpc_message(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "calibrate_adxl_csv",
                "arguments": {
                    "yaml_spec": spec_path.read_text(encoding="utf-8"),
                    "axis_x_csv": x_path.read_text(encoding="utf-8"),
                },
            },
        }
    )
    assert call_resp is not None
    assert not call_resp["result"]["isError"], call_resp["result"]["content"][0]["text"]
    mcp_payload = json.loads(call_resp["result"]["content"][0]["text"])
    assert mcp_payload["axes"]["X"]["measured_peak_hz"] == pytest.approx(52.0, abs=0.1)
