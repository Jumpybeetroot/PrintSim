"""Tests for the PrintSim Live Studio Session Bridge (session store, FastAPI, MCP, and CLI)."""

from __future__ import annotations

import base64
import json
import struct
import threading
import time
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from printsim.api import session as session_mod
from printsim.api.server import create_app
from printsim.cli import app as cli_app
from printsim.mcp_server import dispatch_tool, handle_jsonrpc_message
from printsim.schema import PrinterSpec

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def _make_png_bytes(r: int = 32, g: int = 180, b: int = 255) -> bytes:
    """Create a minimal valid 1x1 RGB PNG byte payload."""

    def _chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00" + bytes([r, g, b]))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", idat)
        + _chunk(b"IEND", b"")
    )


def test_simulate_populates_studio_state_and_ui_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PRINTSIM_SESSION_DIR", str(tmp_path))
    client = TestClient(create_app())

    # Empty session before simulation or heartbeat
    r_empty = client.get("/api/studio/state")
    assert r_empty.status_code == 200
    empty_state = r_empty.json()
    assert empty_state["connected"] is False
    assert empty_state["spec"] is None
    assert empty_state["simulation_summary"] is None

    # Load CoreXY preset and run POST /api/simulate
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    r_sim = client.post(
        "/api/simulate",
        json={
            "spec": spec.model_dump(mode="json"),
            "filament_id": "abs",
            "toolhead_rel_pos": [0.4, 0.6, 0.5],
            "include_toolpath_segments": False,
        },
    )
    assert r_sim.status_code == 200

    # Simulate auto-records spec, yaml_text, and rich simulation_summary without marking connected=True
    r_state = client.get("/api/studio/state")
    assert r_state.status_code == 200
    state = r_state.json()
    assert state["connected"] is False
    assert state["printer_name"] == spec.name
    assert state["filament_id"] == "abs"
    assert state["toolhead_rel_pos"] == [0.4, 0.6, 0.5]
    assert state["yaml_text"]
    parsed_yaml_spec = PrinterSpec.model_validate(yaml.safe_load(state["yaml_text"]))
    assert parsed_yaml_spec.name == spec.name

    summary = state["simulation_summary"]
    assert summary["scores"]["composite"] > 0
    assert isinstance(summary["diagnostics"], list)
    assert summary["collision"]["has_collision"] is False

    # Static FEA scalars + X-beam key nodal displacements (in um) across all 4 load cases
    static_fea = summary["static_fea"]
    for key in (
        "gravity_sag_toolhead_um",
        "accel_deflection_x_um",
        "accel_deflection_y_um",
        "combined_max_deflection_um",
        "bed_deflection_um",
        "toolhead_stiffness_x_n_per_mm",
        "toolhead_stiffness_y_n_per_mm",
        "toolhead_stiffness_z_n_per_mm",
        "max_von_mises_mpa",
    ):
        assert static_fea[key] is not None

    key_disp = static_fea["key_nodal_displacements_um"]
    assert set(key_disp.keys()) == {"combined", "x", "y", "xy"}
    expected_nodes = (
        "col_g_fl",
        "col_g_fr",
        "col_g_rl",
        "col_g_rr",
        "g_fl",
        "g_fr",
        "g_rl",
        "g_rr",
        "y_car_l",
        "y_car_r",
        "toolhead",
        "toolhead_cog",
        "nozzle_tip",
        "bed_surface",
    )
    for load_case in ("combined", "x", "y", "xy"):
        for node_id in expected_nodes:
            assert node_id in key_disp[load_case]
            assert len(key_disp[load_case][node_id]) == 3

    # Verify mm -> um conversion against raw simulate response
    raw_toolhead_mm = r_sim.json()["static_fea"]["nodal_displacements_mm"]["toolhead"]
    assert key_disp["combined"]["toolhead"][0] == pytest.approx(raw_toolhead_mm[0] * 1000.0, abs=1e-2)

    # Modal FEA summary & X-beam key nodal vectors
    modal_fea = summary["modal_fea"]
    assert modal_fea["fundamental_freq_hz"] > 0
    assert modal_fea["worst_print_mode_number"] >= 1
    assert modal_fea["shaper_x"] is not None
    assert modal_fea["shaper_y"] is not None
    assert "panel_drum_freq_hz" in modal_fea
    assert 1 <= len(modal_fea["modes"]) <= 8
    first_mode = modal_fea["modes"][0]
    for node_id in expected_nodes:
        assert node_id in first_mode["key_nodal_vectors"]
        assert len(first_mode["key_nodal_vectors"][node_id]) == 3

    # Thermal & sensitivity summary
    assert summary["thermal"]["max_volumetric_flow_mm3_s"] > 0
    assert summary["thermal"]["limiting_factor"]
    assert summary["thermal"]["cold_throat_temp_c"] > 0
    assert summary["sensitivity"]["primary_problem_subsystem"]
    assert len(summary["sensitivity"]["top_stiffen_targets"]) <= 5
    assert len(summary["sensitivity"]["top_lighten_targets"]) <= 3

    # POST /api/studio/state merges UI fields & selects mode index 3 (Mode #4)
    r_ui = client.post(
        "/api/studio/state",
        json={
            "client_id": "browser-1",
            "active_preset_id": "corexy_voron_style",
            "view_mode": "modal",
            "selected_mode_idx": 3,
            "fea_load_case": "xy",
            "fea_scale": 150.0,
            "sensitivity_lens": "stiffness",
            "modal_color_mode": "strain_energy",
            "inspected_part_id": "x_gantry_l",
        },
    )
    assert r_ui.status_code == 200
    assert r_ui.json()["ok"] is True

    after_ui = client.get("/api/studio/state").json()
    assert after_ui["connected"] is True
    assert after_ui["client_id"] == "browser-1"
    assert after_ui["view_mode"] == "modal"
    assert after_ui["selected_mode_idx"] == 3
    assert after_ui["fea_load_case"] == "xy"
    assert after_ui["fea_scale"] == 150.0
    assert after_ui["active_mode_detail"] is not None
    assert after_ui["active_mode_detail"]["mode_number"] == 4
    # Preserves simulation summary
    assert after_ui["simulation_summary"]["scores"]["composite"] == summary["scores"]["composite"]

    # Partial UI update with spec dict (without yaml_text) auto-serializes yaml_text
    spec_dict = spec.model_dump(mode="json")
    spec_dict["name"] = "Renamed Live Printer"
    r_ui2 = client.post("/api/studio/state", json={"spec": spec_dict})
    assert r_ui2.status_code == 200
    after_ui2 = client.get("/api/studio/state").json()
    assert after_ui2["printer_name"] == "Renamed Live Printer"
    assert "Renamed Live Printer" in after_ui2["yaml_text"]
    assert after_ui2["view_mode"] == "modal"


def test_command_round_trip_and_snapshot_persistence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PRINTSIM_SESSION_DIR", str(tmp_path))
    client = TestClient(create_app())

    # Invalid spec on POST /api/studio/command returns 422
    r_bad = client.post(
        "/api/studio/command",
        json={"type": "update_state", "spec": {"kinematics": {"build_volume_mm": [-10, 100, 100]}}},
    )
    assert r_bad.status_code == 422

    # Enqueue capture_snapshot command
    r_cmd = client.post(
        "/api/studio/command",
        json={"type": "capture_snapshot", "frames": 2, "delay_ms": 250},
    )
    assert r_cmd.status_code == 200
    cmd = r_cmd.json()["command"]
    assert cmd["type"] == "capture_snapshot"
    cmd_id = cmd["id"]

    # First poll delivers command and updates heartbeat -> connected=True
    r_poll1 = client.get("/api/studio/commands", params={"client_id": "studio-tab-a"})
    assert r_poll1.status_code == 200
    cmds = r_poll1.json()["commands"]
    assert len(cmds) == 1
    assert cmds[0]["id"] == cmd_id

    # Second poll returns empty (delivered once)
    r_poll2 = client.get("/api/studio/commands", params={"client_id": "studio-tab-a"})
    assert r_poll2.status_code == 200
    assert r_poll2.json()["commands"] == []

    state = client.get("/api/studio/state").json()
    assert state["connected"] is True
    assert state["client_id"] == "studio-tab-a"

    # Upload 2 PNG frames (one with data URL prefix, one raw base64)
    png1 = _make_png_bytes(255, 40, 40)
    png2 = _make_png_bytes(40, 220, 120)
    b64_1 = "data:image/png;base64," + base64.b64encode(png1).decode("ascii")
    b64_2 = base64.b64encode(png2).decode("ascii")

    r_snap = client.post(
        "/api/studio/snapshot",
        json={
            "command_id": cmd_id,
            "client_id": "studio-tab-a",
            "view_mode": "modal",
            "selected_mode_idx": 2,
            "frames_base64": [b64_1, b64_2],
        },
    )
    assert r_snap.status_code == 200
    snap_resp = r_snap.json()
    assert snap_resp["ok"] is True
    assert len(snap_resp["paths"]) == 2

    # Verify actual PNG files written to disk
    for idx, expected_bytes in ((1, png1), (2, png2)):
        assert (tmp_path / f"snapshot_frame_{idx}.png").read_bytes() == expected_bytes
        assert (tmp_path / f"latest_snapshot_{idx}.png").read_bytes() == expected_bytes
        assert (tmp_path / f"snapshot_{cmd_id}_{idx}.png").read_bytes() == expected_bytes

    after_snap = client.get("/api/studio/state").json()
    latest = after_snap["latest_snapshots"]
    assert latest["command_id"] == cmd_id
    assert latest["view_mode"] == "modal"
    assert latest["selected_mode_idx"] == 2
    assert len(latest["paths"]) == 2
    for p in latest["paths"]:
        assert Path(p).exists()
        assert Path(p).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_command_expiration_and_heartbeat_disconnect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PRINTSIM_SESSION_DIR", str(tmp_path))
    expired_cmd = {
        "id": "expired-1",
        "type": "capture_snapshot",
        "payload": {"frames": 1},
        "created_at": (datetime.now(timezone.utc) - timedelta(seconds=90)).isoformat(),
    }
    (tmp_path / session_mod.COMMANDS_FILE_NAME).write_text(
        json.dumps([expired_cmd]), encoding="utf-8"
    )
    (tmp_path / session_mod.SESSION_FILE_NAME).write_text(
        json.dumps(
            {
                "last_heartbeat_at": (
                    datetime.now(timezone.utc) - timedelta(seconds=25)
                ).isoformat()
            }
        ),
        encoding="utf-8",
    )
    assert session_mod.get_live_session()["connected"] is False
    assert session_mod.poll_commands("tab-1") == []
    assert session_mod.get_live_session()["connected"] is True


def test_mcp_live_studio_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PRINTSIM_SESSION_DIR", str(tmp_path))
    client = TestClient(create_app())

    # 1. get_live_studio_state when empty
    empty_mcp = dispatch_tool("get_live_studio_state", {})
    assert empty_mcp["connected"] is False
    assert empty_mcp["has_session"] is False
    assert "No active or recorded" in empty_mcp["message"]

    # 2. Populate live session via /api/simulate
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    r_sim = client.post("/api/simulate", json={"spec": spec.model_dump(mode="json")})
    assert r_sim.status_code == 200

    live_no_yaml = dispatch_tool("get_live_studio_state", {"include_yaml": False})
    assert live_no_yaml["has_session"] is True
    assert "yaml_text" not in live_no_yaml
    assert live_no_yaml["simulation_summary"]["scores"]["composite"] > 0

    live_with_yaml = dispatch_tool("get_live_studio_state", {"include_yaml": True})
    assert "yaml_text" in live_with_yaml

    # 3. update_live_studio with dot-path overrides applied to live session's spec
    upd = dispatch_tool(
        "update_live_studio",
        {
            "overrides": {
                "gantry.x_beam_profile": "extrusion_3030",
                "kinematics.max_accel_mm_s2": 12000,
            },
            "view_mode": "modal",
            "selected_mode_idx": 1,
            "fea_load_case": "xy",
            "fea_scale": 120.0,
            "toolhead_rel_pos": [0.25, 0.75, 0.5],
            "filament_id": "petg",
        },
    )
    assert upd["ok"] is True
    assert upd["enqueued"] is True
    assert upd["spec_summary"]["x_beam_profile"] == "extrusion_3030"
    assert upd["spec_summary"]["max_accel_mm_s2"] == 12000

    polled = client.get("/api/studio/commands", params={"client_id": "mcp-sim-tab"}).json()["commands"]
    assert len(polled) == 1
    assert polled[0]["type"] == "update_state"
    assert polled[0]["payload"]["spec"]["gantry"]["x_beam_profile"] == "extrusion_3030"
    assert polled[0]["payload"]["view_mode"] == "modal"
    assert polled[0]["payload"]["selected_mode_idx"] == 1
    assert polled[0]["payload"]["toolhead_rel_pos"] == [0.25, 0.75, 0.5]

    # 4. capture_studio_viewport with background thread simulating browser poll + snapshot POST
    stop_event = threading.Event()
    png1 = base64.b64encode(_make_png_bytes(10, 200, 50)).decode("ascii")
    png2 = base64.b64encode(_make_png_bytes(200, 50, 10)).decode("ascii")

    def _mock_browser_worker() -> None:
        deadline = time.monotonic() + 4.0
        while not stop_event.is_set() and time.monotonic() < deadline:
            r = client.get("/api/studio/commands", params={"client_id": "mock-browser"})
            for cmd in r.json().get("commands", []):
                if cmd.get("type") == "capture_snapshot":
                    client.post(
                        "/api/studio/snapshot",
                        json={
                            "command_id": cmd["id"],
                            "client_id": "mock-browser",
                            "view_mode": "modal",
                            "selected_mode_idx": 1,
                            "frames_base64": [png1, png2],
                        },
                    )
                    return
            time.sleep(0.05)

    worker = threading.Thread(target=_mock_browser_worker, daemon=True)
    worker.start()
    try:
        cap_resp = handle_jsonrpc_message(
            {
                "jsonrpc": "2.0",
                "id": 99,
                "method": "tools/call",
                "params": {
                    "name": "capture_studio_viewport",
                    "arguments": {"frames": 2, "delay_ms": 100, "timeout_s": 3.0},
                },
            }
        )
    finally:
        stop_event.set()
        worker.join(timeout=2.0)

    assert cap_resp is not None
    assert cap_resp["result"]["isError"] is False
    cap_data = json.loads(cap_resp["result"]["content"][0]["text"])
    assert cap_data["captured"] is True
    assert len(cap_data["snapshot_paths"]) == 2
    assert all(Path(p).exists() for p in cap_data["snapshot_paths"])
    assert cap_data["view_mode"] == "modal"
    assert cap_data["selected_mode_idx"] == 1
    assert cap_data["active_mode_detail"] is not None

    # 5. Timeout path when no browser responds
    timeout_cap = dispatch_tool(
        "capture_studio_viewport",
        {"frames": 1, "delay_ms": 0, "timeout_s": 0.25},
    )
    assert timeout_cap["captured"] is False
    assert timeout_cap["snapshot_paths"] == []


def test_cli_studio_state_json_and_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PRINTSIM_SESSION_DIR", str(tmp_path))
    client = TestClient(create_app())
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    r_sim = client.post("/api/simulate", json={"spec": spec.model_dump(mode="json")})
    assert r_sim.status_code == 200

    client.post(
        "/api/studio/state",
        json={"client_id": "cli-browser", "view_mode": "static", "selected_mode_idx": 0},
    )

    runner = CliRunner()
    res_json = runner.invoke(cli_app, ["studio-state", "--json"])
    assert res_json.exit_code == 0, res_json.output
    parsed = json.loads(res_json.output)
    assert parsed["connected"] is True
    assert parsed["printer_name"] == spec.name
    assert parsed["view_mode"] == "static"
    assert parsed["simulation_summary"]["scores"]["composite"] > 0
    assert parsed["active_mode_detail"]["mode_number"] == 1

    res_yaml = runner.invoke(cli_app, ["studio-state", "--yaml"])
    assert res_yaml.exit_code == 0, res_yaml.output
    parsed_yaml = yaml.safe_load(res_yaml.output)
    assert parsed_yaml["name"] == spec.name

    # Human-readable Rich summary
    res_human = runner.invoke(cli_app, ["studio-state"])
    assert res_human.exit_code == 0, res_human.output
    assert "PrintSim Live Studio Session" in res_human.output

    # CLI --capture --frames 1 --json with background browser responder
    stop_event = threading.Event()
    png1 = base64.b64encode(_make_png_bytes(80, 90, 240)).decode("ascii")

    def _cli_browser_worker() -> None:
        deadline = time.monotonic() + 4.0
        while not stop_event.is_set() and time.monotonic() < deadline:
            r = client.get("/api/studio/commands", params={"client_id": "cli-mock-browser"})
            for cmd in r.json().get("commands", []):
                if cmd.get("type") == "capture_snapshot":
                    client.post(
                        "/api/studio/snapshot",
                        json={
                            "command_id": cmd["id"],
                            "client_id": "cli-mock-browser",
                            "view_mode": "static",
                            "selected_mode_idx": 0,
                            "frames_base64": [png1],
                        },
                    )
                    return
            time.sleep(0.05)

    worker = threading.Thread(target=_cli_browser_worker, daemon=True)
    worker.start()
    try:
        res_cap = runner.invoke(
            cli_app,
            ["studio-state", "--capture", "--frames", "1", "--delay-ms", "0", "--timeout", "3.0", "--json"],
        )
    finally:
        stop_event.set()
        worker.join(timeout=2.0)

    assert res_cap.exit_code == 0, res_cap.output
    cap_json = json.loads(res_cap.output)
    assert cap_json["capture"]["captured"] is True
    assert len(cap_json["capture"]["snapshot_paths"]) == 1
    assert Path(cap_json["capture"]["snapshot_paths"][0]).exists()


def test_session_dir_fallback_when_unwritable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tempfile

    monkeypatch.delenv("PRINTSIM_SESSION_DIR", raising=False)
    fallback_dir = tmp_path / "temp_fallback"
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(fallback_dir))
    monkeypatch.setattr(session_mod, "PROJECT_ROOT", tmp_path / "ro_root")

    def _mock_writable(p: Path) -> bool:
        if p == (tmp_path / "ro_root" / ".printsim_session"):
            return False
        p.mkdir(parents=True, exist_ok=True)
        return True

    monkeypatch.setattr(session_mod, "_is_dir_writable", _mock_writable)
    resolved = session_mod.get_session_dir()
    assert resolved == fallback_dir / "printsim_session"
    assert resolved.exists()
