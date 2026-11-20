"""FastAPI backend for PrintSim 3D Studio & AI Agent HTTP API."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Literal

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from printsim.engine import evaluate_printer, optimize_design
from printsim.geometry import build_assembly, export_assembly_stl
from printsim.schema import MATERIAL_CATALOG, PROFILE_CATALOG, PrinterSpec

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_DIR = PROJECT_ROOT / "examples"


class SimulateRequest(BaseModel):
    spec: PrinterSpec | None = None
    yaml_text: str | None = None
    gcode_text: str | None = None
    toolhead_rel_pos: tuple[float, float, float] = (0.5, 0.5, 0.5)
    include_toolpath_segments: bool = True


class OptimizeRequest(BaseModel):
    spec: PrinterSpec
    iterations: int = Field(20, ge=3, le=100)
    objective: Literal["balanced", "speed", "rigidity", "budget"] = "balanced"
    seed: int = 42


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
        version="0.1.0",
        description="FEA, modal analysis, kinematics, hotend thermal, and virtual G-code simulation API.",
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
        return {"status": "ok", "version": "0.1.0"}

    @api.get("/api/catalog")
    def catalog() -> dict:
        presets = _load_presets()
        return {
            "materials": [m.model_dump(mode="json") for m in MATERIAL_CATALOG.values()],
            "profiles": [p.model_dump(mode="json") for p in PROFILE_CATALOG.values()],
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
            )
            payload = report.to_dict(include_toolpath_segments=req.include_toolpath_segments)
            payload["spec"] = spec.model_dump(mode="json")
            payload["yaml_text"] = spec.to_yaml()
            return payload
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
