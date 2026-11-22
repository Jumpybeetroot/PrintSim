"""Persistent file-backed session bridge between PrintSim Web Studio, FastAPI, MCP, and CLI."""

from __future__ import annotations

import base64
import json
import os
import re
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import yaml

from printsim.schema import PrinterSpec

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SESSION_FILE_NAME = "live_studio_session.json"
COMMANDS_FILE_NAME = "studio_commands.json"
LOCK_FILE_NAME = ".session.lock"

HEARTBEAT_TIMEOUT_S = 10.0
COMMAND_TTL_S = 60.0

KEY_NODES: tuple[str, ...] = (
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

_UI_FIELDS: tuple[str, ...] = (
    "client_id",
    "active_preset_id",
    "view_mode",
    "selected_mode_idx",
    "fea_load_case",
    "fea_scale",
    "sensitivity_lens",
    "modal_color_mode",
    "toolhead_rel_pos",
    "filament_id",
    "inspected_part_id",
)
_NULLABLE_UI_FIELDS = {"active_preset_id", "inspected_part_id"}
_RESERVED_SESSION_FIELDS = {
    "connected",
    "heartbeat_age_s",
    "active_mode_detail",
    "simulation_summary",
    "latest_snapshots",
    "session_dir",
    "last_heartbeat_at",
    "last_simulated_at",
    "updated_at",
}

_THREAD_LOCK = threading.RLock()


def _now_dt() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now_dt().isoformat()


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts or not isinstance(ts, str):
        return None
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def _is_dir_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path, prefix=".probe_", delete=True):
            pass
        return True
    except OSError:
        return False


def get_session_dir() -> Path:
    """Return writable session directory from PRINTSIM_SESSION_DIR, project root, or temp fallback."""
    env_dir = os.environ.get("PRINTSIM_SESSION_DIR", "").strip()
    candidates: list[Path] = []
    if env_dir:
        candidates.append(Path(env_dir).expanduser())
    candidates.append(PROJECT_ROOT / ".printsim_session")
    candidates.append(Path(tempfile.gettempdir()) / "printsim_session")

    for candidate in candidates:
        if _is_dir_writable(candidate):
            return candidate
    raise RuntimeError("Unable to find or create a writable PrintSim session directory")


@contextmanager
def _session_lock(session_dir: Path) -> Iterator[None]:
    """Serialize read-modify-write access across threads and processes (FastAPI + MCP stdio)."""
    with _THREAD_LOCK:
        lock_path = session_dir / LOCK_FILE_NAME
        lock_file = open(lock_path, "a+", encoding="utf-8")
        try:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            if fcntl is not None:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
            lock_file.close()


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _write_json_atomic(path: Path, data: Any) -> None:
    encoded = (json.dumps(data, indent=2, default=str) + "\n").encode("utf-8")
    _write_bytes_atomic(path, encoded)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        raw = path.read_text(encoding="utf-8")
        if not raw.strip():
            return default
        return json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return default


def _load_session_unlocked(session_dir: Path) -> dict[str, Any]:
    data = _read_json(session_dir / SESSION_FILE_NAME, {})
    return data if isinstance(data, dict) else {}


def _save_session_unlocked(session_dir: Path, data: dict[str, Any]) -> None:
    _write_json_atomic(session_dir / SESSION_FILE_NAME, data)


def _load_commands_unlocked(session_dir: Path) -> list[dict[str, Any]]:
    data = _read_json(session_dir / COMMANDS_FILE_NAME, [])
    return [c for c in data if isinstance(c, dict)] if isinstance(data, list) else []


def _save_commands_unlocked(session_dir: Path, commands: list[dict[str, Any]]) -> None:
    _write_json_atomic(session_dir / COMMANDS_FILE_NAME, commands)


def _extract_key_nodal_um(disp_map: Any) -> dict[str, list[float]]:
    if not isinstance(disp_map, dict):
        return {}
    out: dict[str, list[float]] = {}
    for nid in KEY_NODES:
        vec = disp_map.get(nid)
        if isinstance(vec, (list, tuple)) and len(vec) >= 3:
            out[nid] = [
                round(float(vec[0]) * 1000.0, 3),
                round(float(vec[1]) * 1000.0, 3),
                round(float(vec[2]) * 1000.0, 3),
            ]
    return out


def _extract_key_mode_vectors(nodal_vectors: Any) -> dict[str, list[float]]:
    if not isinstance(nodal_vectors, dict):
        return {}
    out: dict[str, list[float]] = {}
    for nid in KEY_NODES:
        vec = nodal_vectors.get(nid)
        if isinstance(vec, (list, tuple)) and len(vec) >= 3:
            out[nid] = [
                round(float(vec[0]), 4),
                round(float(vec[1]), 4),
                round(float(vec[2]), 4),
            ]
    return out


def _build_simulation_summary(report_dict: dict[str, Any]) -> dict[str, Any]:
    static_raw = report_dict.get("static_fea") or {}
    modal_raw = report_dict.get("modal_fea") or {}
    thermal_raw = report_dict.get("thermal") or {}
    sens_raw = report_dict.get("sensitivity") or {}

    modes_summary: list[dict[str, Any]] = []
    for mode in (modal_raw.get("modes") or [])[:8]:
        if not isinstance(mode, dict):
            continue
        modes_summary.append(
            {
                "mode_number": mode.get("mode_number"),
                "frequency_hz": mode.get("frequency_hz"),
                "dominant_axis": mode.get("dominant_axis"),
                "mode_label": mode.get("mode_label"),
                "dominant_problem_area": mode.get("dominant_problem_area"),
                "print_impact_score": mode.get("print_impact_score"),
                "print_impact_level": mode.get("print_impact_level"),
                "nozzle_whip_ratio": mode.get("nozzle_whip_ratio"),
                "toolhead_flop_pct": mode.get("toolhead_flop_pct"),
                "subsystem_share_pct": mode.get("subsystem_share_pct"),
                "what_is_shaking": mode.get("what_is_shaking"),
                "how_to_fix": mode.get("how_to_fix"),
                "key_nodal_vectors": _extract_key_mode_vectors(mode.get("nodal_vectors")),
            }
        )

    return {
        "scores": report_dict.get("scores") or {},
        "diagnostics": list(report_dict.get("diagnostics") or []),
        "collision": report_dict.get("collision") or {},
        "static_fea": {
            "gravity_sag_toolhead_um": static_raw.get("gravity_sag_toolhead_um"),
            "accel_deflection_x_um": static_raw.get("accel_deflection_x_um"),
            "accel_deflection_y_um": static_raw.get("accel_deflection_y_um"),
            "combined_max_deflection_um": static_raw.get("combined_max_deflection_um"),
            "bed_deflection_um": static_raw.get("bed_deflection_um"),
            "toolhead_stiffness_x_n_per_mm": static_raw.get("toolhead_stiffness_x_n_per_mm"),
            "toolhead_stiffness_y_n_per_mm": static_raw.get("toolhead_stiffness_y_n_per_mm"),
            "toolhead_stiffness_z_n_per_mm": static_raw.get("toolhead_stiffness_z_n_per_mm"),
            "max_von_mises_mpa": static_raw.get("max_von_mises_mpa"),
            "min_safety_factor": static_raw.get("min_safety_factor"),
            "nozzle_twist_amplification_pct": static_raw.get("nozzle_twist_amplification_pct"),
            "key_nodal_displacements_um": {
                "combined": _extract_key_nodal_um(static_raw.get("nodal_displacements_mm")),
                "x": _extract_key_nodal_um(static_raw.get("nodal_displacements_x_mm")),
                "y": _extract_key_nodal_um(static_raw.get("nodal_displacements_y_mm")),
                "xy": _extract_key_nodal_um(static_raw.get("nodal_displacements_xy_mm")),
            },
        },
        "modal_fea": {
            "fundamental_freq_hz": modal_raw.get("fundamental_freq_hz"),
            "worst_print_mode_number": modal_raw.get("worst_print_mode_number"),
            "recommended_max_accel_mm_s2": modal_raw.get("recommended_max_accel_mm_s2"),
            "shaper_x": modal_raw.get("shaper_x"),
            "shaper_y": modal_raw.get("shaper_y"),
            "panel_drum_freq_hz": modal_raw.get("panel_drum_freq_hz"),
            "modes": modes_summary,
        },
        "thermal": {
            "max_volumetric_flow_mm3_s": thermal_raw.get("max_volumetric_flow_mm3_s"),
            "limiting_factor": thermal_raw.get("flow_limiting_factor"),
            "cold_throat_temp_c": thermal_raw.get("cold_side_temp_c"),
            "heat_creep_risk": thermal_raw.get("heat_creep_risk"),
        },
        "sensitivity": {
            "primary_problem_subsystem": sens_raw.get("primary_problem_subsystem"),
            "subsystem_breakdown": sens_raw.get("subsystem_breakdown") or {},
            "top_stiffen_targets": list(sens_raw.get("top_stiffen_targets") or [])[:5],
            "top_lighten_targets": list(sens_raw.get("top_lighten_targets") or [])[:3],
        },
    }


def _spec_to_yaml(spec_dict: dict[str, Any]) -> str:
    try:
        validated = PrinterSpec.model_validate(spec_dict)
        return validated.to_yaml()
    except Exception:
        return yaml.safe_dump(spec_dict, sort_keys=False)


def record_ui_state(payload: dict[str, Any]) -> dict[str, Any]:
    """Merge Studio UI state fields and update heartbeat timestamps."""
    session_dir = get_session_dir()
    now_iso = _now_iso()

    with _session_lock(session_dir):
        session = _load_session_unlocked(session_dir)

        for key in _UI_FIELDS:
            if key not in payload:
                continue
            val = payload[key]
            if val is None and key not in _NULLABLE_UI_FIELDS:
                continue
            if key == "toolhead_rel_pos" and isinstance(val, (list, tuple)) and len(val) == 3:
                session[key] = [float(val[0]), float(val[1]), float(val[2])]
            else:
                session[key] = val

        spec_in = payload.get("spec")
        yaml_in = payload.get("yaml_text")
        if isinstance(spec_in, dict):
            session["spec"] = spec_in
            session["printer_name"] = spec_in.get("name", session.get("printer_name"))
            session["architecture"] = spec_in.get("architecture", session.get("architecture"))
            if isinstance(yaml_in, str) and yaml_in.strip():
                session["yaml_text"] = yaml_in
            else:
                session["yaml_text"] = _spec_to_yaml(spec_in)
        elif isinstance(yaml_in, str) and yaml_in.strip():
            session["yaml_text"] = yaml_in
            try:
                parsed = yaml.safe_load(yaml_in)
                validated = PrinterSpec.model_validate(parsed)
                session["spec"] = validated.model_dump(mode="json")
                session["printer_name"] = validated.name
                session["architecture"] = validated.architecture
            except Exception:
                pass

        for key, val in payload.items():
            if key in _UI_FIELDS or key in ("spec", "yaml_text") or key in _RESERVED_SESSION_FIELDS:
                continue
            session[key] = val

        session["updated_at"] = now_iso
        session["last_heartbeat_at"] = now_iso
        _save_session_unlocked(session_dir, session)

    return get_live_session()


def record_simulation(
    spec: PrinterSpec,
    report_dict: dict[str, Any],
    toolhead_rel_pos: tuple[float, float, float],
    filament_id: str,
) -> dict[str, Any]:
    """Persist the latest simulated spec and structured telemetry summary."""
    session_dir = get_session_dir()
    now_iso = _now_iso()
    spec_dict = spec.model_dump(mode="json")
    yaml_text = spec.to_yaml()
    summary = _build_simulation_summary(report_dict)

    with _session_lock(session_dir):
        session = _load_session_unlocked(session_dir)
        session["spec"] = spec_dict
        session["yaml_text"] = yaml_text
        session["printer_name"] = spec.name
        session["architecture"] = spec.architecture
        session["toolhead_rel_pos"] = [
            float(toolhead_rel_pos[0]),
            float(toolhead_rel_pos[1]),
            float(toolhead_rel_pos[2]),
        ]
        session["filament_id"] = filament_id
        session["simulation_summary"] = summary
        session["last_simulated_at"] = now_iso
        session["updated_at"] = now_iso
        _save_session_unlocked(session_dir, session)

    return get_live_session()


def get_live_session() -> dict[str, Any]:
    """Return merged Studio session state with live connectivity and active mode details."""
    session_dir = get_session_dir()
    with _session_lock(session_dir):
        session = _load_session_unlocked(session_dir)

    now_dt = _now_dt()
    hb_dt = _parse_iso(session.get("last_heartbeat_at"))
    heartbeat_age_s: float | None = None
    connected = False
    if hb_dt is not None:
        age = (now_dt - hb_dt).total_seconds()
        heartbeat_age_s = round(max(0.0, age), 2)
        connected = age <= HEARTBEAT_TIMEOUT_S

    selected_idx = session.get("selected_mode_idx")
    if selected_idx is None:
        selected_idx = 0

    modes = (
        ((session.get("simulation_summary") or {}).get("modal_fea") or {}).get("modes")
        or []
    )
    active_mode_detail: dict[str, Any] | None = None
    if isinstance(modes, list) and isinstance(selected_idx, int) and 0 <= selected_idx < len(modes):
        active_mode_detail = modes[selected_idx]

    result: dict[str, Any] = {
        "client_id": session.get("client_id"),
        "active_preset_id": session.get("active_preset_id"),
        "view_mode": session.get("view_mode"),
        "selected_mode_idx": selected_idx if "selected_mode_idx" in session or modes else session.get("selected_mode_idx"),
        "fea_load_case": session.get("fea_load_case"),
        "fea_scale": session.get("fea_scale"),
        "sensitivity_lens": session.get("sensitivity_lens"),
        "modal_color_mode": session.get("modal_color_mode"),
        "toolhead_rel_pos": session.get("toolhead_rel_pos"),
        "filament_id": session.get("filament_id"),
        "inspected_part_id": session.get("inspected_part_id"),
        "printer_name": session.get("printer_name"),
        "architecture": session.get("architecture"),
        "spec": session.get("spec"),
        "yaml_text": session.get("yaml_text"),
        "simulation_summary": session.get("simulation_summary"),
        "latest_snapshots": session.get("latest_snapshots"),
        "updated_at": session.get("updated_at"),
        "last_heartbeat_at": session.get("last_heartbeat_at"),
        "last_simulated_at": session.get("last_simulated_at"),
        "connected": connected,
        "heartbeat_age_s": heartbeat_age_s,
        "active_mode_detail": active_mode_detail,
        "session_dir": str(session_dir),
    }
    for key, val in session.items():
        if key not in result:
            result[key] = val
    return result


def enqueue_command(cmd_type: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Append a command with UUID and timestamp to studio_commands.json."""
    session_dir = get_session_dir()
    cmd = {
        "id": str(uuid.uuid4()),
        "type": cmd_type,
        "payload": dict(payload or {}),
        "created_at": _now_iso(),
    }
    with _session_lock(session_dir):
        commands = _load_commands_unlocked(session_dir)
        now_dt = _now_dt()
        valid_commands: list[dict[str, Any]] = []
        for existing in commands:
            c_dt = _parse_iso(existing.get("created_at"))
            if c_dt is None or (now_dt - c_dt).total_seconds() <= COMMAND_TTL_S:
                valid_commands.append(existing)
        valid_commands.append(cmd)
        _save_commands_unlocked(session_dir, valid_commands)
    return cmd


def poll_commands(client_id: str | None = None) -> list[dict[str, Any]]:
    """Update Studio heartbeat and pop all non-expired pending commands."""
    session_dir = get_session_dir()
    now_dt = _now_dt()
    now_iso = now_dt.isoformat()

    with _session_lock(session_dir):
        session = _load_session_unlocked(session_dir)
        if client_id is not None and str(client_id).strip():
            session["client_id"] = str(client_id).strip()
        session["last_heartbeat_at"] = now_iso
        _save_session_unlocked(session_dir, session)

        commands = _load_commands_unlocked(session_dir)
        pending: list[dict[str, Any]] = []
        for cmd in commands:
            c_dt = _parse_iso(cmd.get("created_at"))
            if c_dt is not None and (now_dt - c_dt).total_seconds() <= COMMAND_TTL_S:
                pending.append(cmd)
        if commands:
            _save_commands_unlocked(session_dir, [])
    return pending


def _decode_png_base64(frame_b64: str) -> bytes:
    raw = str(frame_b64).strip()
    if raw.startswith("data:") and "," in raw:
        raw = raw.split(",", 1)[1]
    raw = "".join(raw.split())
    if not raw:
        raise ValueError("Empty base64 snapshot frame")
    try:
        data = base64.b64decode(raw, validate=True)
    except Exception as exc:
        raise ValueError(f"Invalid base64 PNG frame: {exc}") from exc
    if not data:
        raise ValueError("Decoded snapshot frame is empty")
    return data


def save_snapshots(
    frames_base64: list[str],
    command_id: str | None = None,
    client_id: str | None = None,
    view_mode: str | None = None,
    selected_mode_idx: int | None = None,
) -> dict[str, Any]:
    """Decode base64 PNG frames, save to session_dir, and record latest_snapshots metadata."""
    if not frames_base64:
        raise ValueError("frames_base64 must contain at least one frame")

    decoded_frames = [_decode_png_base64(frame) for frame in frames_base64]
    session_dir = get_session_dir()
    now_iso = _now_iso()
    safe_cmd_id = (
        re.sub(r"[^A-Za-z0-9_.-]", "_", command_id.strip())
        if command_id and command_id.strip()
        else None
    )

    frame_paths: list[str] = []
    cmd_paths: list[str] = []
    for idx, png_bytes in enumerate(decoded_frames, start=1):
        frame_path = session_dir / f"snapshot_frame_{idx}.png"
        latest_path = session_dir / f"latest_snapshot_{idx}.png"
        _write_bytes_atomic(frame_path, png_bytes)
        _write_bytes_atomic(latest_path, png_bytes)
        frame_paths.append(str(frame_path))
        if safe_cmd_id:
            cmd_path = session_dir / f"snapshot_{safe_cmd_id}_{idx}.png"
            _write_bytes_atomic(cmd_path, png_bytes)
            cmd_paths.append(str(cmd_path))

    primary_paths = cmd_paths if safe_cmd_id else frame_paths

    with _session_lock(session_dir):
        session = _load_session_unlocked(session_dir)
        if client_id is not None and str(client_id).strip():
            session["client_id"] = str(client_id).strip()
        if view_mode is not None:
            session["view_mode"] = view_mode
        if selected_mode_idx is not None:
            session["selected_mode_idx"] = selected_mode_idx

        latest_snapshots = {
            "command_id": command_id,
            "captured_at": now_iso,
            "view_mode": view_mode if view_mode is not None else session.get("view_mode"),
            "selected_mode_idx": (
                selected_mode_idx
                if selected_mode_idx is not None
                else session.get("selected_mode_idx")
            ),
            "paths": primary_paths,
            "frame_paths": frame_paths,
        }
        session["latest_snapshots"] = latest_snapshots
        session["last_heartbeat_at"] = now_iso
        session["updated_at"] = now_iso
        _save_session_unlocked(session_dir, session)

    return latest_snapshots


def request_snapshot_capture(
    frames: int = 2,
    delay_ms: int = 420,
    timeout_s: float = 5.0,
    poll_interval_s: float = 0.1,
) -> dict[str, Any]:
    """Enqueue capture_snapshot and wait until matching snapshot arrives or timeout expires."""
    if frames < 1 or frames > 2:
        raise ValueError("frames must be 1 or 2")
    if delay_ms < 0:
        raise ValueError("delay_ms must be non-negative")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")

    cmd = enqueue_command(
        "capture_snapshot",
        {"frames": int(frames), "delay_ms": int(delay_ms)},
    )
    deadline = time.monotonic() + float(timeout_s)
    session = get_live_session()

    while True:
        snaps = session.get("latest_snapshots") or {}
        if isinstance(snaps, dict) and snaps.get("command_id") == cmd["id"]:
            return {
                "captured": True,
                "command_id": cmd["id"],
                "snapshot_paths": list(snaps.get("paths") or []),
                "frame_paths": list(snaps.get("frame_paths") or []),
                "view_mode": snaps.get("view_mode", session.get("view_mode")),
                "selected_mode_idx": snaps.get(
                    "selected_mode_idx", session.get("selected_mode_idx")
                ),
                "active_mode_detail": session.get("active_mode_detail"),
                "captured_at": snaps.get("captured_at"),
                "connected": bool(session.get("connected")),
            }

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(poll_interval_s, remaining))
        session = get_live_session()

    connected = bool(session.get("connected"))
    message = (
        f"Timed out after {timeout_s:.1f}s waiting for the connected Studio browser to upload snapshot."
        if connected
        else "No live PrintSim Studio browser connected (no heartbeat within the last 10s)."
    )
    return {
        "captured": False,
        "command_id": cmd["id"],
        "snapshot_paths": [],
        "view_mode": session.get("view_mode"),
        "selected_mode_idx": session.get("selected_mode_idx"),
        "active_mode_detail": session.get("active_mode_detail"),
        "connected": connected,
        "message": message,
    }
