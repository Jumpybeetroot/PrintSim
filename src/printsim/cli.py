"""Headless CLI for humans and AI agents (`printsim`)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
import uvicorn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from printsim.engine import evaluate_printer, optimize_design
from printsim.geometry import build_assembly, export_assembly_stl
from printsim.schema import PrinterSpec

app = typer.Typer(
    name="printsim",
    help="Multiphysics 3D Printer Simulator CLI (FEA, modal, kinematics, hotend thermal, G-code).",
    add_completion=False,
)
console = Console()


@app.command()
def evaluate(
    spec_path: Path = typer.Argument(..., exists=True, help="Path to printer YAML spec"),
    gcode: Optional[Path] = typer.Option(None, "--gcode", "-g", exists=True, help="Optional G-code file"),
    filament: str = typer.Option("pla", "--filament", "-f", help="Filament ID (pla, petg, abs, asa, pa12_cf, tpu_95a)"),
    json_out: bool = typer.Option(False, "--json", help="Emit full machine-readable JSON report"),
    no_segments: bool = typer.Option(False, "--no-segments", help="Omit per-segment toolpath array in JSON"),
) -> None:
    """Run full multiphysics evaluation (FEA, modal, kinematics, thermal, virtual print)."""
    spec = PrinterSpec.from_yaml(spec_path)
    gcode_text = gcode.read_text(encoding="utf-8") if gcode else None
    report = evaluate_printer(spec, gcode_text=gcode_text, filament_id=filament)

    if json_out:
        print(json.dumps(report.to_dict(include_toolpath_segments=not no_segments), indent=2))
        return

    s = report.scores
    console.print(
        Panel.fit(
            f"[bold cyan]{spec.name}[/bold cyan] ({spec.architecture})\n"
            f"Composite Score: [bold green]{s.composite:.1f} / 100[/bold green]  |  "
            f"Structural: {s.structural:.1f}  Dynamics: {s.dynamics:.1f}  "
            f"Thermal: {s.thermal:.1f}  Print: {s.print_quality:.1f}  Efficiency: {s.efficiency:.1f}",
            title="PrintSim Multiphysics Report",
        )
    )

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Subsystem")
    table.add_column("Key Metrics")

    fea = report.static_fea
    mod = report.modal_fea
    kin = report.kinematics
    th = report.thermal
    vp = report.virtual_print

    table.add_row(
        "Static FEA",
        f"Gravity sag: {fea.gravity_sag_toolhead_um:.1f} um | "
        f"Peak accel defl: {fea.combined_max_deflection_um:.1f} um | "
        f"Max von Mises: {fea.max_von_mises_mpa:.2f} MPa (SF {fea.min_safety_factor:.1f}x)",
    )
    table.add_row(
        "Modal & Shaper",
        f"f1: {mod.fundamental_freq_hz:.1f} Hz | "
        f"X: {mod.shaper_x.effective_freq_hz:.1f} Hz ({mod.shaper_x.recommended_shaper}) | "
        f"Y: {mod.shaper_y.effective_freq_hz:.1f} Hz ({mod.shaper_y.recommended_shaper}) | "
        f"Rec Max Accel: {mod.recommended_max_accel_mm_s2:.0f} mm/s^2",
    )
    table.add_row(
        "Kinematics",
        f"Torque margin: {kin.torque_safety_margin:.2f}x | "
        f"Belt peak: {kin.belt_peak_tension_n:.1f} N (slip={'YES' if kin.belt_slip_risk else 'no'}) | "
        f"Packaging: {kin.packaging_efficiency_pct:.1f}%",
    )
    table.add_row(
        "Hotend Thermal",
        f"Max flow: {th.max_volumetric_flow_mm3_s:.1f} mm^3/s ({th.flow_limiting_factor}) | "
        f"Heatup(215C): {th.heatup_time_215c_s:.1f}s | "
        f"Cold throat: {th.cold_side_temp_c:.1f}C (creep={'RISK' if th.heat_creep_risk else 'ok'})",
    )
    table.add_row(
        "Virtual Print",
        f"Time: {vp.total_time_s:.1f}s | Quality: {vp.print_quality_score:.1f}/100 | "
        f"Peak flow: {vp.peak_volumetric_flow_mm3_s:.1f} mm^3/s | "
        f"Defects: {vp.underextrusion_segments} flow, {vp.ringing_segments} ringing, {vp.step_skip_segments} skip",
    )
    console.print(table)

    console.print("\n[bold]Diagnostics:[/bold]")
    for d in report.diagnostics:
        console.print(f"  • {d}")


@app.command()
def optimize(
    spec_path: Path = typer.Argument(..., exists=True, help="Baseline printer YAML spec"),
    iterations: int = typer.Option(25, "--iterations", "-n", help="Number of candidate evaluations"),
    objective: str = typer.Option("balanced", "--objective", "-o", help="balanced | speed | rigidity | budget"),
    out: Optional[Path] = typer.Option(None, "--out", help="Optional path to save optimized YAML"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON result"),
) -> None:
    """Run headless evolutionary optimization to improve frame, gantry, and hotend specs."""
    spec = PrinterSpec.from_yaml(spec_path)
    res = optimize_design(spec, iterations=iterations, objective=objective)  # type: ignore[arg-type]

    if out:
        res.best_spec.save_yaml(out)

    if json_out:
        print(json.dumps(res.to_dict(), indent=2))
        return

    console.print(
        Panel.fit(
            f"Objective: [bold]{res.objective}[/bold] ({res.iterations} evals)\n"
            f"Baseline: {res.baseline_objective:.2f}  ->  Best: [bold green]{res.best_objective:.2f}[/bold green] "
            f"([cyan]+{res.improvement_delta:.2f}[/cyan])",
            title="PrintSim AI Design Optimizer",
        )
    )
    if out:
        console.print(f"Saved optimized spec to [green]{out}[/green]")


@app.command("export-stl")
def export_stl(
    spec_path: Path = typer.Argument(..., exists=True, help="Printer YAML spec"),
    out: Path = typer.Option(Path("printer_assembly.stl"), "--out", "-o", help="Output STL/OBJ/GLB path"),
) -> None:
    """Generate and export a 3D solid mesh assembly (STL/OBJ/GLB) of the printer."""
    spec = PrinterSpec.from_yaml(spec_path)
    assembly = build_assembly(spec)
    saved = export_assembly_stl(assembly, out)
    console.print(f"Exported 3D assembly mesh to [bold green]{saved}[/bold green]")


@app.command("solid-fea")
def solid_fea(
    preset: str = typer.Option("corner_gusset", "--preset", "-p", help="corner_gusset | xy_motor_mount | toolhead_carriage | cantilever_bar"),
    stl: Optional[Path] = typer.Option(None, "--stl", exists=True, help="Optional STL/OBJ mesh file to voxelize & analyze"),
    material: str = typer.Option("abs_printed", "--material", "-m", help="Material ID"),
    fx: float = typer.Option(50.0, "--fx", help="Applied Force X (N)"),
    fy: float = typer.Option(0.0, "--fy", help="Applied Force Y (N)"),
    fz: float = typer.Option(-25.0, "--fz", help="Applied Force Z (N)"),
    fixed_face: str = typer.Option("min_z", "--fixed-face", help="min_x|max_x|min_y|max_y|min_z|max_z (STL mode)"),
    load_face: str = typer.Option("max_z", "--load-face", help="min_x|max_x|min_y|max_y|min_z|max_z (STL mode)"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON output"),
) -> None:
    """Run 3D solid continuum Hex8 FEA on a printer bracket/mount preset or custom STL mesh."""
    from printsim.fea.solid_fem import solve_preset_part_fea, solve_solid_voxel_fea, voxelize_stl_file

    if stl is not None:
        grid, h_mm = voxelize_stl_file(stl)
        res = solve_solid_voxel_fea(
            grid=grid,
            voxel_size_mm=h_mm,
            material_id=material,
            fixed_face=fixed_face,  # type: ignore[arg-type]
            load_face=load_face,  # type: ignore[arg-type]
            force_n=(fx, fy, fz),
            part_name=stl.name,
        )
    else:
        res = solve_preset_part_fea(preset=preset, material_id=material, force_n=(fx, fy, fz))  # type: ignore[arg-type]

    if json_out:
        d = res.to_dict()
        d.pop("voxels", None)
        print(json.dumps(d, indent=2))
        return

    console.print(
        Panel.fit(
            f"[bold cyan]{res.part_name}[/bold cyan] ({res.material_id}, FDM anisotropic={res.is_fdm_anisotropic})\n"
            f"Elements: {res.num_elements} Hex8 ({res.num_dofs} DOFs) | Mass: {res.mass_g:.1f} g\n"
            f"Max Displacement: [bold]{res.max_displacement_um:.1f} um[/bold] | Stiffness: {res.stiffness_n_per_mm:.1f} N/mm\n"
            f"Max von Mises: {res.max_von_mises_mpa:.2f} MPa | Max Z-Layer Tension: {res.max_z_tensile_mpa:.2f} MPa\n"
            f"Safety Factor: [bold green]{res.safety_factor:.2f}x[/bold green] ({res.failure_mode})",
            title="PrintSim 3D Solid Hex8 FEA",
        )
    )


@app.command()
def mcp() -> None:
    """Run the stdio MCP (Model Context Protocol) JSON-RPC server for AI design agents."""
    from printsim.mcp_server import run_stdio_server

    run_stdio_server()


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host"),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
) -> None:
    """Launch the PrintSim FastAPI server (backend + built Web 3D Studio)."""
    uvicorn.run("printsim.api.server:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
