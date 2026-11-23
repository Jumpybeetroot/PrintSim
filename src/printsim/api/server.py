"""FastAPI backend for PrintSim 3D Studio & AI Agent HTTP API."""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path
from typing import Any, Literal

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from printsim.api.session import (
    enqueue_command,
    get_live_session,
    poll_commands,
    record_simulation,
    record_ui_state,
    save_snapshots,
)

from printsim.engine import evaluate_printer, optimize_design, optimize_pareto_front
from printsim.engine.evaluator import verify_printer_deep
from printsim.fea.adxl_calibration import calibrate_printer_against_adxl, fit_printer_to_adxl
from printsim.fea.solid_fem import (
    FaceName,
    PartPreset,
    solve_preset_part_fea,
    solve_solid_voxel_fea,
    voxelize_stl_bytes,
)
from printsim.fea.superelement import condense_parametric_bracket
from printsim.fea.tet_fem import (
    solve_preset_tet_fea,
    solve_step_bytes_tet_fea,
)
from printsim.fea.topology import (
    optimize_preset_topology,
    topology_result_to_stl_bytes,
)
from printsim.geometry import (
    build_assembly,
    export_assembly_step_bytes,
    export_assembly_stl,
    export_part_step_bytes,
    generate_bom,
)
from printsim.schema import (
    BED_MOUNT_CATALOG,
    BELT_CATALOG,
    BELT_ROUTING_CATALOG,
    EXTRUSION_FILLINGS,
    FRAME_FEET_CATALOG,
    GANTRY_BRACKET_CATALOG,
    GANTRY_WEIGHT_CLASSES,
    HOTEND_CATALOG,
    MATERIAL_CATALOG,
    PROFILE_CATALOG,
    RAIL_QUALITY_CATALOG,
    STEPPER_CATALOG,
    TOOLHEAD_CONSTRUCTION_CATALOG,
    TOOLHEAD_PRESETS,
    Z_DRIVE_MECHANISM_CATALOG,
    Z_PROBE_CATALOG,
    Z_RAIL_PROFILES,
    PrinterSpec,
)
from printsim.thermal import FILAMENT_CATALOG

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_DIR = PROJECT_ROOT / "examples"


class SimulateRequest(BaseModel):
    spec: PrinterSpec | None = None
    yaml_text: str | None = None
    gcode_text: str | None = None
    filament_id: str = "pla"
    toolhead_rel_pos: tuple[float, float, float] = (0.5, 0.5, 0.5)
    include_toolpath_segments: bool = True
    panel_mesh_n: int = Field(3, ge=2, le=10)
    bed_mesh_n: int = Field(8, ge=4, le=16)
    accel_override_mm_s2: float | None = Field(None, ge=500.0, le=100000.0)


class DeepVerifyRequest(BaseModel):
    spec: PrinterSpec | None = None
    yaml_text: str | None = None
    gcode_text: str | None = None
    filament_id: str = "pla"
    toolhead_rel_pos: tuple[float, float, float] = (0.5, 0.5, 0.5)
    deep_mesh_n: int = Field(8, ge=4, le=10)
    deep_bed_mesh_n: int = Field(16, ge=8, le=20)


class AdxlCalibrationRequest(BaseModel):
    spec: dict[str, Any] | None = None
    yaml_text: str | None = None
    axis_x_csv: str | None = None
    axis_y_csv: str | None = None
    max_freq_hz: float = Field(200.0, ge=20.0, le=500.0)


class OptimizeRequest(BaseModel):
    spec: PrinterSpec
    iterations: int = Field(20, ge=3, le=100)
    objective: Literal["balanced", "speed", "rigidity", "budget"] = "balanced"
    seed: int = 42


class ParetoOptimizeRequest(BaseModel):
    spec: PrinterSpec | None = None
    yaml_text: str | None = None
    population_size: int = Field(16, ge=8, le=40)
    generations: int = Field(6, ge=2, le=20)
    seed: int = 42
    filament_id: str = "pla"


class SolidPartFEARequest(BaseModel):
    preset: PartPreset = "corner_gusset"
    material_id: str = "abs_printed"
    force_n: tuple[float, float, float] | None = None
    fdm_anisotropic: bool | None = None


class SolidStlFEARequest(BaseModel):
    part_name: str = "Uploaded STL Part"
    stl_base64: str
    file_type: str = "stl"
    material_id: str = "pa12_cf_printed"
    fixed_face: FaceName = "min_z"
    load_face: FaceName = "max_z"
    force_n: tuple[float, float, float] = (50.0, 0.0, -25.0)
    resolution: int = Field(16, ge=8, le=22)
    fdm_anisotropic: bool | None = None


class TopologyOptRequest(BaseModel):
    preset: PartPreset = "corner_gusset"
    material_id: str = "pa12_cf_printed"
    force_n: tuple[float, float, float] | None = None
    target_volume_fraction: float = Field(0.45, ge=0.2, le=0.85)
    iterations: int = Field(18, ge=5, le=40)
    penalization: float = Field(3.0, ge=1.5, le=5.0)
    filter_radius_voxels: float = Field(1.5, ge=1.0, le=3.5)
    preserve_boundaries: bool = True
    fdm_anisotropic: bool | None = None
    smooth_iterations: int = Field(0, ge=0, le=10)


class BOMRequest(BaseModel):
    spec: PrinterSpec | None = None
    yaml_text: str | None = None
    stock_length_mm: float = Field(1000.0, ge=500.0, le=4000.0)
    kerf_mm: float = Field(3.0, ge=0.0, le=10.0)


class TetPartFEARequest(BaseModel):
    preset: PartPreset = "corner_gusset"
    material_id: str = "pa12_cf_printed"
    force_n: tuple[float, float, float] | None = None
    mesh_size_mm: float | None = Field(None, ge=1.0, le=12.0)
    order: Literal[1, 2] = 2
    fdm_anisotropic: bool | None = None


class TetStepFEARequest(BaseModel):
    part_name: str = "Uploaded STEP Part"
    step_base64: str
    material_id: str = "pa12_cf_printed"
    fixed_face: FaceName = "min_z"
    load_face: FaceName = "max_z"
    force_n: tuple[float, float, float] = (50.0, 0.0, -25.0)
    mesh_size_mm: float | None = Field(None, ge=1.0, le=15.0)
    order: Literal[1, 2] = 2
    fdm_anisotropic: bool | None = None


class PartStepExportRequest(BaseModel):
    preset: PartPreset = "corner_gusset"


class BracketSuperelementRequest(BaseModel):
    bracket_type: Literal["corner_gusset", "motor_mount", "xy_joint_l_bracket"] = "corner_gusset"
    material_id: str = "ppa_cf_printed"
    arm_length_mm: float = Field(40.0, ge=25.0, le=90.0)
    height_mm: float = Field(20.0, ge=10.0, le=60.0)
    wall_thickness_mm: float = Field(6.0, ge=2.5, le=18.0)
    rib_thickness_mm: float | None = Field(None, ge=0.0, le=18.0)
    rib_count: int | None = Field(None, ge=0, le=6)
    bolt_boss_thickness_mm: float | None = Field(None, ge=0.0, le=18.0)
    bolt_hole_mm: float = Field(5.2, ge=0.0, le=10.0)
    fillet_mm: float = Field(2.0, ge=0.0, le=6.0)
    n_modes: int = Field(6, ge=0, le=24)
    fdm_anisotropic: bool = False
    interface_mode: Literal["rbe3", "rbe2"] = "rbe3"


class StudioStateUpdate(BaseModel):
    model_config = ConfigDict(extra="allow")

    client_id: str | None = None
    active_preset_id: str | None = None
    view_mode: str | None = None
    selected_mode_idx: int | None = Field(None, ge=0)
    fea_load_case: str | None = None
    fea_scale: float | None = Field(None, ge=0.0)
    sensitivity_lens: str | None = None
    modal_color_mode: str | None = None
    toolhead_rel_pos: tuple[float, float, float] | None = None
    filament_id: str | None = None
    inspected_part_id: str | None = None
    spec: dict[str, Any] | None = None
    yaml_text: str | None = None


class StudioCommandRequest(BaseModel):
    type: Literal["capture_snapshot", "update_state"]
    frames: int | None = Field(None, ge=1, le=2)
    delay_ms: int | None = Field(None, ge=0, le=5000)
    spec: dict[str, Any] | None = None
    yaml_text: str | None = None
    view_mode: str | None = None
    selected_mode_idx: int | None = Field(None, ge=0)
    fea_load_case: str | None = None
    fea_scale: float | None = Field(None, ge=0.0)
    sensitivity_lens: str | None = None
    modal_color_mode: str | None = None
    toolhead_rel_pos: tuple[float, float, float] | None = None
    filament_id: str | None = None
    active_preset_id: str | None = None
    inspected_part_id: str | None = None
    payload: dict[str, Any] | None = None


class StudioSnapshotRequest(BaseModel):
    command_id: str | None = None
    client_id: str | None = None
    view_mode: str | None = None
    selected_mode_idx: int | None = Field(None, ge=0)
    frames_base64: list[str] = Field(..., min_length=1)


def _resolve_spec(spec: PrinterSpec | None, yaml_text: str | None) -> PrinterSpec:
    if spec is not None:
        return spec
    if yaml_text:
        try:
            parsed = yaml.safe_load(yaml_text)
            return PrinterSpec.model_validate(parsed)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Invalid YAML spec: {exc}") from exc
    raise HTTPException(status_code=422, detail="Provide either 'spec' or 'yaml_text'")


def _load_presets() -> dict[str, PrinterSpec]:
    presets: dict[str, PrinterSpec] = {}
    if EXAMPLES_DIR.exists():
        for p in sorted(EXAMPLES_DIR.glob("*.yaml")):
            try:
                presets[p.stem] = PrinterSpec.from_yaml(p)
            except Exception:
                continue
    return presets


def create_app() -> FastAPI:
    api = FastAPI(
        title="PrintSim Multiphysics 3D Printer Simulator API",
        version="1.1.0",
        description="Frame, MITC4 Shell Panel & solid Hex8/Tet10 FEA (Gmsh + scikit-fem), Craig-Bampton VPT superelements, Huth CBUSH bolted fasteners, Hertzian rail carriages, Newmark composite rails, Timoshenko thermal-structural coupling, coupled LPV mechatronics & workspace FRF maps, ZOH time-domain corner/ballbar metrology, RCSA toolhead whip, exact adjoint & Fox-Kapoor sensitivities, Klipper ADXL345 shaper PSD, pymoo NSGA-II Pareto optimization, SIMP topology optimization, manufacturable BOM, hotend FVM thermal, and virtual G-code API.",
    )

    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @api.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "version": "1.1.0"}

    @api.get("/api/catalog")
    def catalog() -> dict:
        presets = _load_presets()
        return {
            "materials": [m.model_dump(mode="json") for m in MATERIAL_CATALOG.values()],
            "profiles": [p.model_dump(mode="json") for p in PROFILE_CATALOG.values()],
            "filaments": [f.model_dump(mode="json") for f in FILAMENT_CATALOG.values()],
            "steppers": list(STEPPER_CATALOG.values()),
            "toolhead_presets": list(TOOLHEAD_PRESETS.values()),
            "toolhead_constructions": list(TOOLHEAD_CONSTRUCTION_CATALOG.values()),
            "extrusion_fillings": list(EXTRUSION_FILLINGS.values()),
            "gantry_brackets": list(GANTRY_BRACKET_CATALOG.values()),
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
                {"id": "l_bracket", "name": "Internal/Stamped L-Bracket", "short_name": "L-Bracket"},
                {"id": "blind_joint", "name": "Blind Joint (Voron Standard)", "short_name": "Blind Joint"},
                {"id": "gusset_plate", "name": "External CNC Gusset Plate", "short_name": "Gusset Plate"},
                {"id": "corner_cube", "name": "3-Way Cast/CNC Corner Cube", "short_name": "Corner Cube"},
                {"id": "rigid", "name": "Idealized Rigid Welded Corner", "short_name": "Rigid"},
            ],
            "rail_preloads": [
                {"id": "ZF", "name": "ZF (Slight Clearance)", "short_name": "ZF"},
                {"id": "Z0", "name": "Z0 (Zero Clearance / Light)", "short_name": "Z0"},
                {"id": "Z1", "name": "Z1 (Light Preload — Voron Recommended)", "short_name": "Z1"},
                {"id": "ZA", "name": "ZA (Medium Preload)", "short_name": "ZA"},
                {"id": "ZB", "name": "ZB (Heavy Preload)", "short_name": "ZB"},
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
            "z_probes": list(Z_PROBE_CATALOG.values()),
            "belts": list(BELT_CATALOG.values()),
            "belt_routings": list(BELT_ROUTING_CATALOG.values()),
            "hotend_presets": list(HOTEND_CATALOG.values()),
            "z_drive_mechanisms": list(Z_DRIVE_MECHANISM_CATALOG.values()),
            "z_rail_profiles": Z_RAIL_PROFILES,
            "bearing_clearances": [
                {"id": "C2", "name": "ISO C2 Tight Clearance (3.5 um)", "short_name": "C2"},
                {"id": "CN", "name": "ISO CN Normal Clearance (7.5 um)", "short_name": "CN"},
                {"id": "C3", "name": "ISO C3 Loose Clearance (15.5 um)", "short_name": "C3"},
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
            ],
            "part_presets": [
                {"id": "corner_gusset", "name": "40mm L-Corner Gusset Bracket"},
                {"id": "xy_motor_mount", "name": "CoreXY NEMA17 Motor Mount Plate"},
                {"id": "toolhead_carriage", "name": "Direct-Drive X-Carriage Plate"},
                {"id": "cantilever_bar", "name": "60mm Solid Test Cantilever Bar"},
            ],
            "presets": [
                {
                    "id": pid,
                    "name": spec.name,
                    "architecture": spec.architecture,
                    "description": spec.description,
                }
                for pid, spec in presets.items()
            ],
        }

    @api.get("/api/presets/{preset_id}")
    def get_preset(preset_id: str) -> dict:
        presets = _load_presets()
        if preset_id not in presets:
            raise HTTPException(status_code=404, detail=f"Preset '{preset_id}' not found")
        spec = presets[preset_id]
        return {
            "id": preset_id,
            "spec": spec.model_dump(mode="json"),
            "yaml_text": spec.to_yaml(),
        }

    @api.post("/api/simulate")
    def run_simulation(req: SimulateRequest) -> dict:
        if req.spec is not None:
            spec = req.spec
        elif req.yaml_text:
            try:
                parsed = yaml.safe_load(req.yaml_text)
                spec = PrinterSpec.model_validate(parsed)
            except Exception as exc:
                raise HTTPException(status_code=422, detail=f"Invalid YAML spec: {exc}") from exc
        else:
            raise HTTPException(status_code=422, detail="Provide either 'spec' or 'yaml_text'")

        try:
            report = evaluate_printer(
                spec,
                gcode_text=req.gcode_text,
                toolhead_rel_pos=req.toolhead_rel_pos,
                filament_id=req.filament_id,
                panel_mesh_n=req.panel_mesh_n,
                bed_mesh_n=req.bed_mesh_n,
                accel_override_mm_s2=req.accel_override_mm_s2,
            )
            payload = report.to_dict(include_toolpath_segments=req.include_toolpath_segments)
            payload["spec"] = spec.model_dump(mode="json")
            payload["yaml_text"] = spec.to_yaml()
            try:
                record_simulation(spec, payload, req.toolhead_rel_pos, req.filament_id)
            except Exception:
                pass
            return payload
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @api.post("/api/verify/deep")
    def run_deep_verify(req: DeepVerifyRequest) -> dict:
        spec = _resolve_spec(req.spec, req.yaml_text)
        try:
            result = verify_printer_deep(
                spec,
                gcode_text=req.gcode_text,
                toolhead_rel_pos=req.toolhead_rel_pos,
                filament_id=req.filament_id,
                deep_mesh_n=req.deep_mesh_n,
                deep_bed_mesh_n=req.deep_bed_mesh_n,
            )
            deep_sim = result.get("deep_simulation")
            if isinstance(deep_sim, dict):
                deep_sim.setdefault("spec", spec.model_dump(mode="json"))
                deep_sim.setdefault("yaml_text", spec.to_yaml())
            return result
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def _resolve_adxl_request(req: AdxlCalibrationRequest) -> PrinterSpec:
        if req.spec is not None:
            try:
                spec = PrinterSpec.model_validate(req.spec)
            except Exception as exc:
                raise HTTPException(status_code=422, detail=f"Invalid PrinterSpec: {exc}") from exc
        else:
            spec = _resolve_spec(None, req.yaml_text)

        if not (req.axis_x_csv and req.axis_x_csv.strip()) and not (
            req.axis_y_csv and req.axis_y_csv.strip()
        ):
            raise HTTPException(
                status_code=400,
                detail="Provide at least one measured CSV ('axis_x_csv' and/or 'axis_y_csv').",
            )
        return spec

    @api.post("/api/calibration/adxl")
    def run_adxl_calibration(req: AdxlCalibrationRequest) -> dict:
        spec = _resolve_adxl_request(req)
        try:
            report = calibrate_printer_against_adxl(
                spec,
                axis_x_csv=req.axis_x_csv,
                axis_y_csv=req.axis_y_csv,
                max_freq_hz=req.max_freq_hz,
            )
            return report.to_dict()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @api.post("/api/calibration/adxl/fit")
    def run_adxl_calibration_fit(req: AdxlCalibrationRequest) -> dict:
        spec = _resolve_adxl_request(req)
        try:
            fit_report = fit_printer_to_adxl(
                spec,
                axis_x_csv=req.axis_x_csv,
                axis_y_csv=req.axis_y_csv,
                max_freq_hz=req.max_freq_hz,
            )
            return fit_report.to_dict()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @api.post("/api/fea/solid-part")
    def run_solid_part_fea(req: SolidPartFEARequest) -> dict:
        try:
            res = solve_preset_part_fea(
                preset=req.preset,
                material_id=req.material_id,
                force_n=req.force_n,
                fdm_anisotropic=req.fdm_anisotropic,
            )
            return res.to_dict()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @api.post("/api/superelement/bracket")
    def run_bracket_superelement(req: BracketSuperelementRequest) -> dict:
        try:
            if req.rib_thickness_mm is not None:
                resolved_rib = req.rib_thickness_mm
            elif req.rib_count is not None:
                resolved_rib = 0.0 if req.rib_count == 0 else min(18.0, 3.5 + 1.5 * req.rib_count)
            else:
                resolved_rib = 5.0

            resolved_wall = (
                max(req.wall_thickness_mm, req.bolt_boss_thickness_mm)
                if req.bolt_boss_thickness_mm is not None
                else req.wall_thickness_mm
            )
            bracket_params = {
                "bracket_type": req.bracket_type,
                "arm_length_mm": req.arm_length_mm,
                "height_mm": req.height_mm,
                "wall_thickness_mm": resolved_wall,
                "rib_thickness_mm": resolved_rib,
                "bolt_hole_mm": req.bolt_hole_mm,
                "fillet_mm": req.fillet_mm,
                "n_modes": req.n_modes,
                "fdm_anisotropic": req.fdm_anisotropic,
                "interface_mode": req.interface_mode,
            }
            se = condense_parametric_bracket(bracket_params, material_id=req.material_id)
            return se.to_dict()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @api.post("/api/fea/solid-stl")
    def run_solid_stl_fea(req: SolidStlFEARequest) -> dict:
        try:
            raw = base64.b64decode(req.stl_base64)
            grid, h_mm = voxelize_stl_bytes(
                raw, file_type=req.file_type, target_max_dim_voxels=req.resolution
            )
            res = solve_solid_voxel_fea(
                grid=grid,
                voxel_size_mm=h_mm,
                material_id=req.material_id,
                fixed_face=req.fixed_face,
                load_face=req.load_face,
                force_n=req.force_n,
                part_name=req.part_name,
                fdm_anisotropic=req.fdm_anisotropic,
            )
            return res.to_dict()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @api.post("/api/fea/topology-optimize")
    def run_topology_optimize(req: TopologyOptRequest) -> dict:
        try:
            res = optimize_preset_topology(
                preset=req.preset,
                material_id=req.material_id,
                force_n=req.force_n,
                target_volume_fraction=req.target_volume_fraction,
                iterations=req.iterations,
                penalization=req.penalization,
                filter_radius_voxels=req.filter_radius_voxels,
                preserve_boundaries=req.preserve_boundaries,
                fdm_anisotropic=req.fdm_anisotropic,
            )
            return res.to_dict(include_voxels=True)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @api.post("/api/fea/topology-optimize/stl")
    def export_topology_opt_stl(req: TopologyOptRequest) -> Response:
        try:
            res = optimize_preset_topology(
                preset=req.preset,
                material_id=req.material_id,
                force_n=req.force_n,
                target_volume_fraction=req.target_volume_fraction,
                iterations=req.iterations,
                penalization=req.penalization,
                filter_radius_voxels=req.filter_radius_voxels,
                preserve_boundaries=req.preserve_boundaries,
                fdm_anisotropic=req.fdm_anisotropic,
            )
            stl_bytes = topology_result_to_stl_bytes(res, smooth_iterations=req.smooth_iterations)
            fname = f"{req.preset}_topo_opt_{int(round(res.achieved_volume_fraction * 100))}pct.stl"
            return Response(
                content=stl_bytes,
                media_type="model/stl",
                headers={"Content-Disposition": f'attachment; filename="{fname}"'},
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @api.post("/api/bom")
    def compute_bom(req: BOMRequest) -> dict:
        spec = _resolve_spec(req.spec, req.yaml_text)
        try:
            bom = generate_bom(spec, stock_length_mm=req.stock_length_mm, kerf_mm=req.kerf_mm)
            return bom.to_dict()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @api.post("/api/bom/csv")
    def export_bom_csv(req: BOMRequest) -> PlainTextResponse:
        spec = _resolve_spec(req.spec, req.yaml_text)
        try:
            bom = generate_bom(spec, stock_length_mm=req.stock_length_mm, kerf_mm=req.kerf_mm)
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in spec.name.lower())
            return PlainTextResponse(
                content=bom.to_csv(),
                media_type="text/csv",
                headers={"Content-Disposition": f'attachment; filename="{safe_name or "printer"}_bom.csv"'},
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @api.post("/api/optimize")
    def run_optimization(req: OptimizeRequest) -> dict:
        try:
            res = optimize_design(
                req.spec,
                iterations=req.iterations,
                objective=req.objective,
                seed=req.seed,
            )
            data = res.to_dict()
            data["best_yaml"] = res.best_spec.to_yaml()
            return data
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @api.post("/api/optimize/pareto")
    def run_pareto_optimization(req: ParetoOptimizeRequest) -> dict:
        spec = _resolve_spec(req.spec, req.yaml_text)
        try:
            res = optimize_pareto_front(
                spec,
                population_size=req.population_size,
                generations=req.generations,
                seed=req.seed,
                filament_id=req.filament_id,
            )
            return res.to_dict(include_specs=True)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @api.post("/api/fea/tet-part")
    def run_tet_part_fea(req: TetPartFEARequest) -> dict:
        try:
            res = solve_preset_tet_fea(
                preset=req.preset,
                material_id=req.material_id,
                force_n=req.force_n,
                mesh_size_mm=req.mesh_size_mm,
                order=req.order,
                fdm_anisotropic=req.fdm_anisotropic,
            )
            return res.to_dict(include_surface_mesh=True)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @api.post("/api/fea/tet-step")
    def run_tet_step_fea(req: TetStepFEARequest) -> dict:
        try:
            raw = base64.b64decode(req.step_base64)
            res = solve_step_bytes_tet_fea(
                step_bytes=raw,
                material_id=req.material_id,
                fixed_face=req.fixed_face,
                load_face=req.load_face,
                force_n=req.force_n,
                mesh_size_mm=req.mesh_size_mm,
                order=req.order,
                part_name=req.part_name,
                fdm_anisotropic=req.fdm_anisotropic,
            )
            return res.to_dict(include_surface_mesh=True)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @api.post("/api/export/step")
    def export_step(spec: PrinterSpec) -> Response:
        try:
            assembly = build_assembly(spec)
            step_bytes = export_assembly_step_bytes(assembly)
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in spec.name.lower())
            return Response(
                content=step_bytes,
                media_type="application/step",
                headers={"Content-Disposition": f'attachment; filename="{safe_name or "printer"}.step"'},
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @api.post("/api/export/part-step")
    def export_part_step_endpoint(req: PartStepExportRequest) -> Response:
        try:
            step_bytes = export_part_step_bytes(req.preset)
            return Response(
                content=step_bytes,
                media_type="application/step",
                headers={"Content-Disposition": f'attachment; filename="{req.preset}.step"'},
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @api.post("/api/export/stl")
    def export_stl(spec: PrinterSpec) -> FileResponse:
        assembly = build_assembly(spec)
        tmp = tempfile.NamedTemporaryFile(suffix=".stl", delete=False)
        tmp.close()
        out_path = export_assembly_stl(assembly, tmp.name)
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in spec.name.lower())
        return FileResponse(
            path=str(out_path),
            media_type="model/stl",
            filename=f"{safe_name or 'printer'}.stl",
        )

    @api.get("/api/studio/state")
    def get_studio_state() -> dict:
        return get_live_session()

    @api.post("/api/studio/state")
    def post_studio_state(req: StudioStateUpdate) -> dict:
        payload = req.model_dump(exclude_unset=True)
        updated = record_ui_state(payload)
        return {"ok": True, "updated_at": updated.get("updated_at")}

    @api.get("/api/studio/commands")
    def get_studio_commands(client_id: str | None = None) -> dict:
        commands = poll_commands(client_id=client_id)
        return {"commands": commands}

    @api.post("/api/studio/command")
    def post_studio_command(req: StudioCommandRequest) -> dict:
        cmd_payload: dict[str, Any] = dict(req.payload or {})
        optional_fields = (
            "frames",
            "delay_ms",
            "spec",
            "yaml_text",
            "view_mode",
            "selected_mode_idx",
            "fea_load_case",
            "fea_scale",
            "sensitivity_lens",
            "modal_color_mode",
            "toolhead_rel_pos",
            "filament_id",
            "active_preset_id",
            "inspected_part_id",
        )
        for field_name in optional_fields:
            val = getattr(req, field_name)
            if val is not None:
                if field_name == "toolhead_rel_pos":
                    cmd_payload[field_name] = [float(val[0]), float(val[1]), float(val[2])]
                else:
                    cmd_payload[field_name] = val

        if req.type == "capture_snapshot":
            cmd_payload.setdefault("frames", 2)
            cmd_payload.setdefault("delay_ms", 420)

        if cmd_payload.get("spec") is not None:
            try:
                validated = PrinterSpec.model_validate(cmd_payload["spec"])
            except Exception as exc:
                raise HTTPException(status_code=422, detail=f"Invalid spec: {exc}") from exc
            cmd_payload["spec"] = validated.model_dump(mode="json")
            if not cmd_payload.get("yaml_text"):
                cmd_payload["yaml_text"] = validated.to_yaml()
        elif cmd_payload.get("yaml_text"):
            try:
                parsed = yaml.safe_load(str(cmd_payload["yaml_text"]))
                validated = PrinterSpec.model_validate(parsed)
            except Exception as exc:
                raise HTTPException(status_code=422, detail=f"Invalid yaml_text: {exc}") from exc
            cmd_payload["spec"] = validated.model_dump(mode="json")

        cmd = enqueue_command(req.type, cmd_payload)
        return {"ok": True, "command": cmd}

    @api.post("/api/studio/snapshot")
    def post_studio_snapshot(req: StudioSnapshotRequest) -> dict:
        try:
            saved = save_snapshots(
                frames_base64=req.frames_base64,
                command_id=req.command_id,
                client_id=req.client_id,
                view_mode=req.view_mode,
                selected_mode_idx=req.selected_mode_idx,
            )
            return {
                "ok": True,
                "paths": saved["paths"],
                "latest_snapshots": saved,
            }
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Serve built Vite frontend if present
    web_dist = PROJECT_ROOT / "web" / "dist"
    if web_dist.exists():
        api.mount("/", StaticFiles(directory=str(web_dist), html=True), name="web")

    return api


app = create_app()
