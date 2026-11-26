"""Zero-FastAPI request dispatcher for in-browser Pyodide WASM and static Pages pre-baking.

Mirrors the JSON/binary contracts of ``printsim.api.server`` using only pure-Python
core dependencies (``numpy``, ``scipy``, ``pydantic``, ``pyyaml``), with lazy on-demand
imports for ``trimesh`` (STL export/import) and ``ezdxf`` (2D DXF export).
"""

from __future__ import annotations

import base64
import json
import tempfile
from pathlib import Path
from typing import Any

import yaml

from printsim.engine import evaluate_printer, optimize_design, optimize_pareto_front
from printsim.engine.evaluator import verify_printer_deep
from printsim.fea.adxl_calibration import calibrate_printer_against_adxl, fit_printer_to_adxl
from printsim.fea.solid_fem import (
    solve_preset_part_fea,
    solve_solid_voxel_fea,
    voxelize_stl_bytes,
)
from printsim.fea.superelement import condense_parametric_bracket
from printsim.fea.topology import (
    optimize_preset_topology,
    topology_result_to_stl_bytes,
)
from printsim.geometry import (
    build_assembly,
    export_assembly_stl,
    export_panels_dxf_bytes,
    generate_bom,
)
from printsim.presets import (
    build_catalog_presets_payload,
    find_examples_dir as _find_examples_dir,
    load_presets,
)
from printsim.schema import (
    BEARING_RADIAL_PLAY_UM,
    BED_MOUNT_CATALOG,
    BELT_CATALOG,
    BELT_ROUTING_CATALOG,
    CHAMBER_HEATER_CATALOG,
    CORNER_JOINT_BASE_2020,
    CORNER_JOINT_IWAN_DAMPING,
    CPAP_BLOWER_CATALOG,
    EXTRUSION_FILLINGS,
    FRAME_FEET_CATALOG,
    SUSPENSION_LINE_CATALOG,
    FRONT_BRACE_CATALOG,
    GANTRY_BRACKET_CATALOG,
    GANTRY_HARDWARE_KIT_CATALOG,
    GANTRY_WEIGHT_CLASSES,
    HOTEND_CATALOG,
    HOTEND_MOUNT_CLAMP_BASE_N_PER_MM,
    MATERIAL_CATALOG,
    PROFILE_CATALOG,
    RAIL_PRELOAD_FRACTIONS,
    RAIL_QUALITY_CATALOG,
    STEPPER_CATALOG,
    TOOLHEAD_CONSTRUCTION_CATALOG,
    TOOLHEAD_PRESETS,
    TOP_BRACE_CATALOG,
    Z_DRIVE_MECHANISM_CATALOG,
    Z_PROBE_CATALOG,
    Z_RAIL_PROFILES,
    CustomBracketOverride,
    PrinterSpec,
)
from printsim.thermal import FILAMENT_CATALOG

_BRACKET_TYPE_TO_CMS: dict[str, str] = {
    "xy_motor_mount": "xy_motor_mount",
    "corner_gusset": "corner_gusset",
    "toolhead_carriage": "toolhead_carriage",
    "motor_mount": "xy_motor_mount",
    "idler_block": "corner_gusset",
    "xy_joint": "toolhead_carriage",
}


def build_catalog_payload() -> dict[str, Any]:
    return {
        "materials": [m.model_dump(mode="json") for m in MATERIAL_CATALOG.values()],
        "profiles": [p.model_dump(mode="json") for p in PROFILE_CATALOG.values()],
        "filaments": [f.model_dump(mode="json") for f in FILAMENT_CATALOG.values()],
        "steppers": list(STEPPER_CATALOG.values()),
        "toolhead_presets": list(TOOLHEAD_PRESETS.values()),
        "toolhead_constructions": list(TOOLHEAD_CONSTRUCTION_CATALOG.values()),
        "extrusion_fillings": list(EXTRUSION_FILLINGS.values()),
        "gantry_brackets": list(GANTRY_BRACKET_CATALOG.values()),
        "gantry_hardware_kits": list(GANTRY_HARDWARE_KIT_CATALOG.values()),
        "gantry_weight_classes": list(GANTRY_WEIGHT_CLASSES.values()),
        "x_rail_mounts": [
            {"id": "front", "name": "Front Face (Voron Standard)", "short_name": "Front", "notes": "Bolted to front (-Y) face: max Y-accel bending stiffness; standard forward CoG"},
            {"id": "top", "name": "Top Face (RatRig / Annex)", "short_name": "Top", "notes": "Bolted to top (+Z) face: max Z stiffness & shorter Y overhang; slightly taller nozzle drop"},
            {"id": "bottom", "name": "Bottom Face (Under-Slung)", "short_name": "Bottom", "notes": "Bolted to bottom (-Z) face: max Z stiffness & ~12 mm shorter nozzle Z lever arm (less pitch twist)"},
            {"id": "dual_top_bottom", "name": "Dual Top + Bottom (I-Beam)", "short_name": "Dual Top+Bot", "notes": "Symmetric top & bottom rails: highest Z/Y stiffness & zero bimetallic warp, +2× rail mass"},
        ],
        "y_rail_mounts": [
            {"id": "top", "name": "Top Face (Voron Standard)", "short_name": "Top", "notes": "Bolted to top (+Z) face of Y extrusions: max Z stiffness + wide face resists X"},
            {"id": "inner", "name": "Inner Side Face (Monolith / VzBot)", "short_name": "Inner Side", "notes": "Bolted to inner (±X) face of Y extrusions: max X-bending stiffness against gantry push"},
            {"id": "bottom", "name": "Bottom Face (Inverted)", "short_name": "Bottom", "notes": "Bolted to bottom (-Z) face of Y extrusions: high Z stiffness & debris protection"},
        ],
        "corner_joints": [
            {
                "id": jid,
                "name": jname,
                "short_name": jshort,
                "k_t_n_per_mm": CORNER_JOINT_BASE_2020[jid][0] * 1e-3,
                "k_r_inplane_nm_rad": CORNER_JOINT_BASE_2020[jid][1],
                "k_r_torsion_nm_rad": CORNER_JOINT_BASE_2020[jid][2],
                "damping_delta": CORNER_JOINT_IWAN_DAMPING[jid],
            }
            for jid, jname, jshort in (
                ("l_bracket", "Internal/Stamped L-Bracket", "L-Bracket"),
                ("blind_joint", "Blind Joint (Voron Standard)", "Blind Joint"),
                ("gusset_plate", "External CNC Gusset Plate", "Gusset Plate"),
                ("corner_cube", "3-Way Cast/CNC Corner Cube", "Corner Cube"),
                ("rigid", "Idealized Rigid Welded Corner", "Rigid"),
            )
        ],
        "rail_preloads": [
            {
                "id": pid,
                "name": pname,
                "short_name": pid,
                "preload_fraction": RAIL_PRELOAD_FRACTIONS[pid],
            }
            for pid, pname in (
                ("ZF", "ZF (Slight Clearance)"),
                ("Z0", "Z0 (Zero Clearance / Light)"),
                ("Z1", "Z1 (Light Preload — Voron Recommended)"),
                ("ZA", "ZA (Medium Preload)"),
                ("ZB", "ZB (Heavy Preload)"),
            )
        ],
        "rail_qualities": list(RAIL_QUALITY_CATALOG.values()),
        "rail_block_types": [
            {"id": "C", "name": "C Block (Standard Length)", "short_name": "C"},
            {"id": "H", "name": "H Block (Long Heavy-Load)", "short_name": "H"},
        ],
        "rail_backers": [
            {"id": "none", "name": "None (Bare Extrusion Opposite Face)", "short_name": "None"},
            {"id": "titanium_grade5", "name": "Ti-6Al-4V Grade 5 Backer", "short_name": "Titanium"},
            {"id": "steel_4140", "name": "4140 Steel Backer", "short_name": "Steel 4140"},
        ],
        "bed_mounts": list(BED_MOUNT_CATALOG.values()),
        "frame_feet": list(FRAME_FEET_CATALOG.values()),
        "suspension_lines": list(SUSPENSION_LINE_CATALOG.values()),
        "z_probes": list(Z_PROBE_CATALOG.values()),
        "belts": list(BELT_CATALOG.values()),
        "belt_routings": list(BELT_ROUTING_CATALOG.values()),
        "cpap_blowers": list(CPAP_BLOWER_CATALOG.values()),
        "chamber_heaters": list(CHAMBER_HEATER_CATALOG.values()),
        "hotend_presets": list(HOTEND_CATALOG.values()),
        "hotend_mount_styles": [
            {
                "id": msid,
                "name": msname,
                "short_name": msshort,
                "clamp_stiffness_base_n_per_mm": HOTEND_MOUNT_CLAMP_BASE_N_PER_MM[msid],
            }
            for msid, msname, msshort in (
                ("groove_mount", "J-Head Groove-Mount Collar (E3D V6)", "Groove Mount"),
                ("rigid_4bolt_m3", "4-Bolt M3 Rigid Flange (Revo Voron)", "4-Bolt M3"),
                ("dragon_4bolt_m25", "4-Bolt M2.5 Rigid Pattern (Dragon / Rapido / Goliath)", "4-Bolt M2.5"),
                ("chube_6bolt_m3_18pcd", "6-Bolt M3 18mm PCD Monocoque Flange (Chube)", "6-Bolt M3 18PCD"),
            )
        ],
        "z_drive_mechanisms": list(Z_DRIVE_MECHANISM_CATALOG.values()),
        "z_rail_profiles": list(Z_RAIL_PROFILES.values()),
        "top_braces": list(TOP_BRACE_CATALOG.values()),
        "front_braces": list(FRONT_BRACE_CATALOG.values()),
        "bearing_clearances": [
            {
                "id": cid,
                "name": cname,
                "short_name": cid,
                "radial_play_um": BEARING_RADIAL_PLAY_UM[cid],
            }
            for cid, cname in (
                ("C2", "ISO C2 Tight Clearance (3.5 um)"),
                ("CN", "ISO CN Normal Clearance (7.5 um)"),
                ("C3", "ISO C3 Loose Clearance (15.5 um)"),
            )
        ],
        "drive_types": [
            {
                "id": "open_loop_stepper",
                "name": "Open-Loop Stepper (Standard Step/Dir)",
                "short_name": "Open-Loop Stepper",
                "notes": "Standard TMC2209/5160 step/dir drive; subject to Acarnley L/R mid-band negative damping at high stepping rates",
            },
            {
                "id": "closed_loop_stepper",
                "name": "Closed-Loop Stepper (Encoder FOC)",
                "short_name": "Closed-Loop Stepper",
                "notes": "Encoder-locked FOC microstepping suppresses mid-band resonance and adds active velocity damping",
            },
            {
                "id": "foc_servo",
                "name": "FOC AC/BLDC Servo (Velocity + Position Cascade)",
                "short_name": "FOC Servo",
                "notes": "High-bandwidth cascaded velocity/position FOC loop with active damping and optional biquad notch filter",
            },
        ],
        "input_shapers": [
            {"id": "auto", "name": "Auto (Klipper Optimal Score)", "short_name": "Auto"},
            {"id": "none", "name": "None (Unshaped)", "short_name": "None"},
            {"id": "zv", "name": "ZV (Zero Vibration)", "short_name": "ZV"},
            {"id": "mzv", "name": "MZV (Modified Zero Vibration)", "short_name": "MZV"},
            {"id": "ei", "name": "EI (Extra Insensitive)", "short_name": "EI"},
            {"id": "2hump_ei", "name": "2HUMP_EI (Two-Hump EI)", "short_name": "2HUMP_EI"},
            {"id": "3hump_ei", "name": "3HUMP_EI (Three-Hump EI)", "short_name": "3HUMP_EI"},
            {"id": "zvdd", "name": "ZVDD (Zero Vibration Double-Derivative)", "short_name": "ZVDD"},
            {"id": "smooth_zv", "name": "SMOOTH_ZV (Kalico C1/C2 Smooth ZV)", "short_name": "SMOOTH_ZV"},
            {"id": "smooth_ei", "name": "SMOOTH_EI (Kalico C1/C2 Smooth EI)", "short_name": "SMOOTH_EI"},
            {"id": "smooth_flatness", "name": "SMOOTH_FLATNESS (Kalico Exact Flatness Bell)", "short_name": "SMOOTH_FLATNESS"},
        ],
        "part_presets": [
            {"id": "corner_gusset", "name": "40mm L-Corner Gusset Bracket"},
            {"id": "xy_motor_mount", "name": "CoreXY NEMA17 Motor Mount Plate"},
            {"id": "toolhead_carriage", "name": "Direct-Drive X-Carriage Plate"},
            {"id": "cantilever_bar", "name": "60mm Solid Test Cantilever Bar"},
        ],
        "presets": build_catalog_presets_payload(),
    }


def _resolve_spec_payload(payload: dict[str, Any]) -> PrinterSpec:
    if payload.get("spec") is not None:
        return PrinterSpec.model_validate(payload["spec"])
    yaml_text = payload.get("yaml_text")
    if yaml_text:
        parsed = yaml.safe_load(yaml_text)
        return PrinterSpec.model_validate(parsed)
    # Endpoints like /api/export/stl accept PrinterSpec directly as the root body
    if "architecture" in payload or "frame" in payload:
        return PrinterSpec.model_validate(payload)
    raise ValueError("Provide either 'spec' or 'yaml_text'")


def _ok_json(data: Any, status: int = 200) -> dict[str, Any]:
    return {
        "status": status,
        "content_type": "application/json",
        "body_text": json.dumps(data, allow_nan=False),
    }


def _err_json(detail: str, status: int = 400) -> dict[str, Any]:
    return _ok_json({"detail": detail}, status=status)


def _ok_bytes(
    raw: bytes,
    content_type: str,
    filename: str | None = None,
) -> dict[str, Any]:
    headers: dict[str, str] = {}
    if filename:
        headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    return {
        "status": 200,
        "content_type": content_type,
        "headers": headers,
        "body_base64": base64.b64encode(raw).decode("ascii"),
    }


def dispatch_api_request(
    method: str,
    path: str,
    body_json: str | None = None,
) -> dict[str, Any]:
    """Execute a PrintSim ``/api/...`` request synchronously and return a serializable envelope."""
    clean_path = path.split("?", 1)[0].rstrip("/")
    verb = method.upper()
    payload: dict[str, Any] = {}
    if body_json:
        try:
            parsed = json.loads(body_json)
            if isinstance(parsed, dict):
                payload = parsed
        except Exception as exc:
            return _err_json(f"Invalid JSON request body: {exc}", status=422)

    try:
        if verb == "GET" and clean_path == "/api/health":
            return _ok_json({"status": "ok", "version": "1.1.0", "runtime": "pyodide-wasm"})

        if verb == "GET" and clean_path == "/api/catalog":
            return _ok_json(build_catalog_payload())

        if verb == "GET" and clean_path == "/api/assets/hardware":
            try:
                from printsim.cad_assets.registry import list_hardware_assets

                return _ok_json(list_hardware_assets())
            except ImportError:
                return _ok_json(
                    {
                        "hotends": {},
                        "toolheads": {},
                        "assemblies": {},
                        "gantry_brackets": {},
                    }
                )

        if verb == "GET" and clean_path.startswith("/api/presets/"):
            preset_id = clean_path.split("/api/presets/", 1)[1]
            presets = load_presets(include_test_builds=True)
            if preset_id not in presets:
                return _err_json(f"Preset '{preset_id}' not found", status=404)
            spec = presets[preset_id]
            return _ok_json(
                {
                    "id": preset_id,
                    "spec": spec.model_dump(mode="json"),
                    "yaml_text": spec.to_yaml(),
                }
            )

        if verb == "POST" and clean_path == "/api/simulate":
            spec = _resolve_spec_payload(payload)
            pos = tuple(float(v) for v in payload.get("toolhead_rel_pos", (0.5, 0.5, 0.5)))
            report = evaluate_printer(
                spec,
                gcode_text=payload.get("gcode_text"),
                toolhead_rel_pos=(pos[0], pos[1], pos[2]),
                filament_id=str(payload.get("filament_id", "pla")),
                panel_mesh_n=int(payload.get("panel_mesh_n", 3)),
                bed_mesh_n=int(payload.get("bed_mesh_n", 8)),
                accel_override_mm_s2=payload.get("accel_override_mm_s2"),
            )
            out = report.to_dict(
                include_toolpath_segments=bool(payload.get("include_toolpath_segments", True))
            )
            out["spec"] = spec.model_dump(mode="json")
            out["yaml_text"] = spec.to_yaml()
            return _ok_json(out)

        if verb == "POST" and clean_path == "/api/verify/deep":
            spec = _resolve_spec_payload(payload)
            pos = tuple(float(v) for v in payload.get("toolhead_rel_pos", (0.5, 0.5, 0.5)))
            res = verify_printer_deep(
                spec,
                gcode_text=payload.get("gcode_text"),
                toolhead_rel_pos=(pos[0], pos[1], pos[2]),
                filament_id=str(payload.get("filament_id", "pla")),
                deep_mesh_n=int(payload.get("deep_mesh_n", 8)),
                deep_bed_mesh_n=int(payload.get("deep_bed_mesh_n", 16)),
            )
            deep_sim = res.get("deep_simulation")
            if isinstance(deep_sim, dict):
                deep_sim.setdefault("spec", spec.model_dump(mode="json"))
                deep_sim.setdefault("yaml_text", spec.to_yaml())
            return _ok_json(res)

        if verb == "POST" and clean_path == "/api/bom":
            spec = _resolve_spec_payload(payload)
            bom = generate_bom(
                spec,
                stock_length_mm=float(payload.get("stock_length_mm", 1000.0)),
                kerf_mm=float(payload.get("kerf_mm", 3.0)),
            )
            return _ok_json(bom.to_dict())

        if verb == "POST" and clean_path == "/api/bom/csv":
            spec = _resolve_spec_payload(payload)
            bom = generate_bom(
                spec,
                stock_length_mm=float(payload.get("stock_length_mm", 1000.0)),
                kerf_mm=float(payload.get("kerf_mm", 3.0)),
            )
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in spec.name.lower())
            return {
                "status": 200,
                "content_type": "text/csv",
                "headers": {
                    "Content-Disposition": f'attachment; filename="{safe_name or "printer"}_bom.csv"'
                },
                "body_text": bom.to_csv(),
            }

        if verb == "POST" and clean_path == "/api/optimize":
            spec = _resolve_spec_payload(payload)
            res = optimize_design(
                spec,
                iterations=int(payload.get("iterations", 20)),
                objective=payload.get("objective", "balanced"),
                seed=int(payload.get("seed", 42)),
            )
            data = res.to_dict()
            data["best_yaml"] = res.best_spec.to_yaml()
            return _ok_json(data)

        if verb == "POST" and clean_path == "/api/optimize/pareto":
            spec = _resolve_spec_payload(payload)
            res = optimize_pareto_front(
                spec,
                population_size=int(payload.get("population_size", 16)),
                generations=int(payload.get("generations", 6)),
                seed=int(payload.get("seed", 42)),
                filament_id=str(payload.get("filament_id", "pla")),
            )
            return _ok_json(res.to_dict(include_specs=True))

        if verb == "POST" and clean_path in ("/api/calibration/adxl", "/api/calibration/adxl/fit"):
            spec = _resolve_spec_payload(payload)
            axis_x = payload.get("axis_x_csv")
            axis_y = payload.get("axis_y_csv")
            if not (axis_x and str(axis_x).strip()) and not (axis_y and str(axis_y).strip()):
                return _err_json(
                    "Provide at least one measured CSV ('axis_x_csv' and/or 'axis_y_csv').",
                    status=400,
                )
            max_freq = float(payload.get("max_freq_hz", 200.0))
            if clean_path.endswith("/fit"):
                fit_rep = fit_printer_to_adxl(
                    spec, axis_x_csv=axis_x, axis_y_csv=axis_y, max_freq_hz=max_freq
                )
                return _ok_json(fit_rep.to_dict())
            rep = calibrate_printer_against_adxl(
                spec, axis_x_csv=axis_x, axis_y_csv=axis_y, max_freq_hz=max_freq
            )
            return _ok_json(rep.to_dict())

        if verb == "POST" and clean_path == "/api/fea/solid-part":
            force_raw = payload.get("force_n")
            force_n = tuple(float(v) for v in force_raw) if force_raw is not None else None
            res = solve_preset_part_fea(
                preset=payload.get("preset", "corner_gusset"),
                material_id=str(payload.get("material_id", "abs_printed")),
                force_n=force_n,  # type: ignore[arg-type]
                fdm_anisotropic=payload.get("fdm_anisotropic"),
            )
            return _ok_json(res.to_dict())

        if verb == "POST" and clean_path == "/api/fea/solid-stl":
            raw = base64.b64decode(str(payload["stl_base64"]))
            grid, h_mm = voxelize_stl_bytes(
                raw,
                file_type=str(payload.get("file_type", "stl")),
                target_max_dim_voxels=int(payload.get("resolution", 16)),
            )
            f_raw = payload.get("force_n", (50.0, 0.0, -25.0))
            res = solve_solid_voxel_fea(
                grid=grid,
                voxel_size_mm=h_mm,
                material_id=str(payload.get("material_id", "pa12_cf_printed")),
                fixed_face=payload.get("fixed_face", "min_z"),
                load_face=payload.get("load_face", "max_z"),
                force_n=(float(f_raw[0]), float(f_raw[1]), float(f_raw[2])),
                part_name=str(payload.get("part_name", "Uploaded STL Part")),
                fdm_anisotropic=payload.get("fdm_anisotropic"),
            )
            return _ok_json(res.to_dict())

        if verb == "POST" and clean_path == "/api/fea/topology-optimize":
            force_raw = payload.get("force_n")
            force_n = tuple(float(v) for v in force_raw) if force_raw is not None else None
            res = optimize_preset_topology(
                preset=payload.get("preset", "corner_gusset"),
                material_id=str(payload.get("material_id", "pa12_cf_printed")),
                force_n=force_n,  # type: ignore[arg-type]
                target_volume_fraction=float(payload.get("target_volume_fraction", 0.45)),
                iterations=int(payload.get("iterations", 18)),
                penalization=float(payload.get("penalization", 3.0)),
                filter_radius_voxels=float(payload.get("filter_radius_voxels", 1.5)),
                preserve_boundaries=bool(payload.get("preserve_boundaries", True)),
                fdm_anisotropic=payload.get("fdm_anisotropic"),
            )
            return _ok_json(res.to_dict(include_voxels=True))

        if verb == "POST" and clean_path == "/api/fea/topology-optimize/stl":
            force_raw = payload.get("force_n")
            force_n = tuple(float(v) for v in force_raw) if force_raw is not None else None
            res = optimize_preset_topology(
                preset=payload.get("preset", "corner_gusset"),
                material_id=str(payload.get("material_id", "pa12_cf_printed")),
                force_n=force_n,  # type: ignore[arg-type]
                target_volume_fraction=float(payload.get("target_volume_fraction", 0.45)),
                iterations=int(payload.get("iterations", 18)),
                penalization=float(payload.get("penalization", 3.0)),
                filter_radius_voxels=float(payload.get("filter_radius_voxels", 1.5)),
                preserve_boundaries=bool(payload.get("preserve_boundaries", True)),
                fdm_anisotropic=payload.get("fdm_anisotropic"),
            )
            stl_bytes = topology_result_to_stl_bytes(
                res, smooth_iterations=int(payload.get("smooth_iterations", 0))
            )
            fname = f"{payload.get('preset', 'corner_gusset')}_topo_opt_{int(round(res.achieved_volume_fraction * 100))}pct.stl"
            return _ok_bytes(stl_bytes, "model/stl", fname)

        if verb == "POST" and clean_path == "/api/superelement/bracket":
            rib_t = payload.get("rib_thickness_mm")
            rib_c = payload.get("rib_count")
            if rib_t is not None:
                resolved_rib = float(rib_t)
            elif rib_c is not None:
                resolved_rib = 0.0 if int(rib_c) == 0 else min(18.0, 3.5 + 1.5 * int(rib_c))
            else:
                resolved_rib = 5.0
            wall = float(payload.get("wall_thickness_mm", 6.0))
            boss = payload.get("bolt_boss_thickness_mm")
            resolved_wall = max(wall, float(boss)) if boss is not None else wall
            bracket_params = {
                "bracket_type": payload.get("bracket_type", "corner_gusset"),
                "arm_length_mm": float(payload.get("arm_length_mm", 40.0)),
                "height_mm": float(payload.get("height_mm", 40.0)),
                "wall_thickness_mm": resolved_wall,
                "rib_thickness_mm": resolved_rib,
                "bolt_hole_mm": float(payload.get("bolt_hole_mm", 5.2)),
                "fillet_mm": float(payload.get("fillet_mm", 3.0)),
                "n_modes": int(payload.get("n_modes", 6)),
                "fdm_anisotropic": payload.get("fdm_anisotropic"),
                "interface_mode": payload.get("interface_mode", "rbe3_vpt"),
            }
            se = condense_parametric_bracket(
                bracket_params, material_id=str(payload.get("material_id", "pa12_cf_printed"))
            )
            return _ok_json(se.to_dict())

        if verb == "POST" and clean_path == "/api/brackets/inject-evaluate":
            base_spec = _resolve_spec_payload(payload)
            btype = str(payload.get("bracket_type", "motor_mount"))
            mat = str(payload.get("material", "pa12_cf_printed"))
            custom_k = payload.get("custom_stiffness_n_per_mm")
            condensed_mass_g: float | None = None
            if custom_k is None:
                cms_type = _BRACKET_TYPE_TO_CMS.get(btype)
                if cms_type is None:
                    raise ValueError(f"Unknown bracket_type '{btype}'")
                se = condense_parametric_bracket({"bracket_type": cms_type}, material_id=mat)
                diag = se.port_stiffness_diag
                k_eff = 2.0 / (
                    1.0 / max(float(diag["k_x_n_m"]), 1e-6)
                    + 1.0 / max(float(diag["k_y_n_m"]), 1e-6)
                )
                k_eff_n_mm = min(500000.0, max(1.1, k_eff * 1e-3))
                condensed_mass_g = round(float(se.mass_kg) * 1000.0, 2)
                source = "cms_superelement"
                default_label = f"CMS {btype} ({mat})"
            else:
                k_eff_n_mm = float(custom_k)
                source = "custom_cad"
                default_label = f"Custom {btype} ({k_eff_n_mm:.0f} N/mm)"

            override = CustomBracketOverride(
                label=payload.get("label") or default_label,
                stiffness_n_per_mm=k_eff_n_mm,
                mass_delta_g=float(payload.get("mass_delta_g", 0.0)),
                source=source,
            )
            injected_spec = base_spec.model_copy(
                update={
                    "gantry": base_spec.gantry.model_copy(
                        update={"custom_bracket_override": override}
                    )
                }
            )
            base_rep = evaluate_printer(base_spec, include_advanced_analysis=False)
            inj_rep = evaluate_printer(injected_spec, include_advanced_analysis=False)
            base_x = float(base_rep.modal_fea.shaper_x.effective_freq_hz if base_rep.modal_fea.shaper_x else base_rep.modal_fea.fundamental_freq_hz)
            base_y = float(base_rep.modal_fea.shaper_y.effective_freq_hz if base_rep.modal_fea.shaper_y else base_rep.modal_fea.fundamental_freq_hz)
            inj_x = float(inj_rep.modal_fea.shaper_x.effective_freq_hz if inj_rep.modal_fea.shaper_x else inj_rep.modal_fea.fundamental_freq_hz)
            inj_y = float(inj_rep.modal_fea.shaper_y.effective_freq_hz if inj_rep.modal_fea.shaper_y else inj_rep.modal_fea.fundamental_freq_hz)
            base_acc = float(base_rep.modal_fea.recommended_max_accel_mm_s2)
            inj_acc = float(inj_rep.modal_fea.recommended_max_accel_mm_s2)
            return _ok_json(
                {
                    "baseline_x_hz": round(base_x, 2),
                    "baseline_y_hz": round(base_y, 2),
                    "injected_x_hz": round(inj_x, 2),
                    "injected_y_hz": round(inj_y, 2),
                    "delta_x_hz": round(inj_x - base_x, 2),
                    "delta_y_hz": round(inj_y - base_y, 2),
                    "baseline_accel_mm_s2": round(base_acc, 0),
                    "injected_accel_mm_s2": round(inj_acc, 0),
                    "delta_accel_mm_s2": round(inj_acc - base_acc, 0),
                    "injected_stiffness_n_per_mm": round(k_eff_n_mm, 1),
                    "condensed_mass_g": condensed_mass_g,
                    "custom_bracket_override": override.model_dump(mode="json"),
                    "injected_spec": injected_spec.model_dump(mode="json"),
                    "injected_yaml": injected_spec.to_yaml(),
                }
            )

        if verb == "POST" and clean_path == "/api/export/stl":
            spec = _resolve_spec_payload(payload)
            assembly = build_assembly(spec)
            with tempfile.NamedTemporaryFile(suffix=".stl", delete=True) as tmp:
                export_assembly_stl(assembly, tmp.name)
                raw = Path(tmp.name).read_bytes()
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in spec.name.lower())
            return _ok_bytes(raw, "model/stl", f"{safe_name or 'printer'}.stl")

        if verb == "POST" and clean_path == "/api/export/dxf":
            spec = _resolve_spec_payload(payload)
            part = str(payload.get("part", "all_plates"))
            dxf_bytes = export_panels_dxf_bytes(spec, part=part)
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in spec.name.lower())
            safe_part = "".join(c if c.isalnum() or c in "-_" else "_" for c in part.lower())
            return _ok_bytes(dxf_bytes, "application/dxf", f"{safe_name or 'printer'}_{safe_part or 'all_plates'}.dxf")

        if clean_path in ("/api/fea/tet-part", "/api/fea/tet-step"):
            return _err_json(
                "Curved Tet10 FEA (Gmsh + scikit-fem) requires the local native backend (`printsim serve`); falling back to in-browser QM6-3D Hex8 FEA.",
                status=400,
            )

        if clean_path in ("/api/export/step", "/api/export/part-step"):
            return _err_json(
                "OpenCASCADE B-Rep STEP export (`build123d`) requires a local PrintSim install (`uv run printsim serve`). STL, DXF, and BOM CSV exports work directly in the browser.",
                status=400,
            )

        return _err_json(f"Unsupported endpoint in browser engine: {verb} {clean_path}", status=404)
    except Exception as exc:
        return _err_json(str(exc), status=400)
