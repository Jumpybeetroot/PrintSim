"""FastAPI backend for PrintSim 3D Studio & AI Agent HTTP API."""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path
from typing import Literal

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from printsim.engine import evaluate_printer, optimize_design
from printsim.fea.solid_fem import (
    FaceName,
    PartPreset,
    solve_preset_part_fea,
    solve_solid_voxel_fea,
    voxelize_stl_bytes,
)
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
from printsim.schema import MATERIAL_CATALOG, PROFILE_CATALOG, PrinterSpec
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


class OptimizeRequest(BaseModel):
    spec: PrinterSpec
    iterations: int = Field(20, ge=3, le=100)
    objective: Literal["balanced", "speed", "rigidity", "budget"] = "balanced"
    seed: int = 42


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
        version="0.5.0",
        description="Frame & solid Hex8/Tet10 FEA (Gmsh + scikit-fem), build123d STEP CAD, CoreXY AWD & structural panels, SIMP topology optimization, manufacturable BOM, modal analysis, kinematics, hotend FVM thermal, and virtual G-code API.",
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
        return {"status": "ok", "version": "0.5.0"}

    @api.get("/api/catalog")
    def catalog() -> dict:
        presets = _load_presets()
        return {
            "materials": [m.model_dump(mode="json") for m in MATERIAL_CATALOG.values()],
            "profiles": [p.model_dump(mode="json") for p in PROFILE_CATALOG.values()],
            "filaments": [f.model_dump(mode="json") for f in FILAMENT_CATALOG.values()],
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
            )
            payload = report.to_dict(include_toolpath_segments=req.include_toolpath_segments)
            payload["spec"] = spec.model_dump(mode="json")
            payload["yaml_text"] = spec.to_yaml()
            return payload
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

    # Serve built Vite frontend if present
    web_dist = PROJECT_ROOT / "web" / "dist"
    if web_dist.exists():
        api.mount("/", StaticFiles(directory=str(web_dist), html=True), name="web")

    return api


app = create_app()
