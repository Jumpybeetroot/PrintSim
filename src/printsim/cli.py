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

from printsim.engine import evaluate_printer, optimize_design, optimize_pareto_front
from printsim.geometry import (
    build_assembly,
    export_assembly_step,
    export_assembly_stl,
    export_part_step,
    generate_bom,
)
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


@app.command("pareto")
def pareto_cmd(
    spec_path: Path = typer.Argument(..., exists=True, help="Baseline printer YAML spec"),
    pop: int = typer.Option(16, "--pop", help="NSGA-II population size"),
    gen: int = typer.Option(6, "--gen", help="NSGA-II generations"),
    filament: str = typer.Option("pla", "--filament", "-f", help="Filament ID"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON Pareto frontier"),
) -> None:
    """Run pymoo NSGA-II multi-objective Pareto optimization (Cost vs. Safe Accel vs. Nozzle Stiffness)."""
    spec = PrinterSpec.from_yaml(spec_path)
    res = optimize_pareto_front(spec, population_size=pop, generations=gen, filament_id=filament)

    if json_out:
        print(json.dumps(res.to_dict(include_specs=False), indent=2))
        return

    console.print(
        Panel.fit(
            f"[bold cyan]{spec.name}[/bold cyan] — {res.backend} ({res.evaluations} unique evals)\n"
            f"Baseline: ${res.baseline.total_cost_usd:.0f} | {res.baseline.safe_max_accel_mm_s2:.0f} mm/s² | "
            f"{res.baseline.nozzle_deflection_um:.1f} um flex | Score {res.baseline.composite_score:.1f}",
            title="PrintSim Multi-Goal Pareto Frontier",
        )
    )
    tbl = Table(show_header=True, header_style="bold magenta")
    tbl.add_column("Label")
    tbl.add_column("Cost ($)", justify="right")
    tbl.add_column("Safe Accel", justify="right")
    tbl.add_column("Nozzle Flex", justify="right")
    tbl.add_column("Score", justify="right")
    tbl.add_column("Key Upgrades")
    for p in res.pareto_front:
        tbl.add_row(
            p.label,
            f"${p.total_cost_usd:.0f}",
            f"{p.safe_max_accel_mm_s2:.0f} mm/s²",
            f"{p.nozzle_deflection_um:.1f} um",
            f"{p.composite_score:.1f}",
            ", ".join(p.changes[:3]) or "Stock",
        )
    console.print(tbl)


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


@app.command("export-step")
def export_step_cmd(
    spec_path: Optional[Path] = typer.Argument(None, help="Printer YAML spec (omit when using --preset)"),
    preset: Optional[str] = typer.Option(None, "--preset", "-p", help="Export a single parametric B-Rep bracket preset instead of full printer (corner_gusset | xy_motor_mount | toolhead_carriage | cantilever_bar)"),
    out: Path = typer.Option(Path("printer_assembly.step"), "--out", "-o", help="Output OpenCASCADE STEP path"),
) -> None:
    """Export exact OpenCASCADE B-Rep STEP CAD (build123d) for a full printer assembly or parametric bracket."""
    if preset is not None:
        saved = export_part_step(preset, out)  # type: ignore[arg-type]
        console.print(f"Exported parametric B-Rep bracket STEP to [bold green]{saved}[/bold green]")
        return
    if spec_path is None or not spec_path.exists():
        raise typer.BadParameter("Provide a valid printer YAML spec path or --preset.")
    spec = PrinterSpec.from_yaml(spec_path)
    assembly = build_assembly(spec)
    saved = export_assembly_step(assembly, out)
    console.print(f"Exported full printer B-Rep STEP assembly to [bold green]{saved}[/bold green]")


@app.command("tet-fea")
def tet_fea_cmd(
    preset: str = typer.Option("corner_gusset", "--preset", "-p", help="corner_gusset | xy_motor_mount | toolhead_carriage | cantilever_bar"),
    step: Optional[Path] = typer.Option(None, "--step", exists=True, help="Optional STEP CAD file to mesh with Gmsh & solve with scikit-fem"),
    material: str = typer.Option("pa12_cf_printed", "--material", "-m", help="Material ID"),
    order: int = typer.Option(2, "--order", help="Element order: 1 (Tet4 P1) or 2 (Tet10 P2)"),
    mesh_size: Optional[float] = typer.Option(None, "--mesh-size", help="Target tetrahedral edge size (mm)"),
    fx: Optional[float] = typer.Option(None, "--fx", help="Override Force X (N)"),
    fy: Optional[float] = typer.Option(None, "--fy", help="Override Force Y (N)"),
    fz: Optional[float] = typer.Option(None, "--fz", help="Override Force Z (N)"),
    fixed_face: str = typer.Option("min_z", "--fixed-face", help="Fixed face for --step mode"),
    load_face: str = typer.Option("max_z", "--load-face", help="Loaded face for --step mode"),
    vtu_out: Optional[Path] = typer.Option(None, "--vtu", help="Optional ParaView .vtu/.vtk output path"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON output"),
) -> None:
    """Run conformal 3D tetrahedral continuum FEA (Gmsh + scikit-fem) on a B-Rep bracket or STEP file."""
    from printsim.fea.tet_fem import solve_preset_tet_fea, solve_step_tet_fea

    force_override = None
    if fx is not None or fy is not None or fz is not None:
        force_override = (fx or 0.0, fy or 0.0, fz or 0.0)

    if step is not None:
        res = solve_step_tet_fea(
            step_path=step,
            material_id=material,
            fixed_face=fixed_face,  # type: ignore[arg-type]
            load_face=load_face,  # type: ignore[arg-type]
            force_n=force_override or (50.0, 0.0, -25.0),
            mesh_size_mm=mesh_size,
            order=2 if order >= 2 else 1,
            export_vtu_path=vtu_out,
        )
    else:
        res = solve_preset_tet_fea(
            preset=preset,  # type: ignore[arg-type]
            material_id=material,
            force_n=force_override,
            mesh_size_mm=mesh_size,
            order=2 if order >= 2 else 1,
        )

    if json_out:
        print(json.dumps(res.to_dict(include_surface_mesh=False), indent=2))
        return

    console.print(
        Panel.fit(
            f"[bold cyan]{res.part_name}[/bold cyan] ({res.backend} • {res.element_type})\n"
            f"Material: {res.material_id} (FDM anisotropic={res.is_fdm_anisotropic})\n"
            f"Mesh: {res.num_elements} tets ({res.num_dofs} DOFs, h≈{res.mesh_size_mm:.2f} mm) | Mass: {res.mass_g:.2f} g\n"
            f"Max Displacement: [bold]{res.max_displacement_um:.2f} um[/bold] | Stiffness: {res.stiffness_n_per_mm:.1f} N/mm\n"
            f"Max von Mises: {res.max_von_mises_mpa:.2f} MPa | Max Z-Layer Tension: {res.max_z_tensile_mpa:.2f} MPa\n"
            f"Safety Factor: [bold green]{res.safety_factor:.2f}x[/bold green] ({res.failure_mode})",
            title="PrintSim Conformal 3D Tet FEA (Gmsh + scikit-fem)",
        )
    )


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


@app.command("topo-opt")
def topo_opt(
    preset: str = typer.Option("corner_gusset", "--preset", "-p", help="corner_gusset | xy_motor_mount | toolhead_carriage | cantilever_bar"),
    material: str = typer.Option("pa12_cf_printed", "--material", "-m", help="Material ID"),
    volume_fraction: float = typer.Option(0.45, "--volume-fraction", "-v", help="Target retained volume fraction (0.2..0.85)"),
    iterations: int = typer.Option(18, "--iterations", "-n", help="SIMP optimization iterations"),
    fx: Optional[float] = typer.Option(None, "--fx", help="Optional override Force X (N)"),
    fy: Optional[float] = typer.Option(None, "--fy", help="Optional override Force Y (N)"),
    fz: Optional[float] = typer.Option(None, "--fz", help="Optional override Force Z (N)"),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Optional output STL file path for the optimized part"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON output"),
) -> None:
    """Run 3D SIMP topology optimization on a printer bracket/mount and optionally export printable STL."""
    from printsim.fea.topology import export_topology_stl, optimize_preset_topology

    force_override = None
    if fx is not None or fy is not None or fz is not None:
        force_override = (fx or 0.0, fy or 0.0, fz or 0.0)

    res = optimize_preset_topology(
        preset=preset,  # type: ignore[arg-type]
        material_id=material,
        force_n=force_override,
        target_volume_fraction=volume_fraction,
        iterations=iterations,
    )
    saved_path = export_topology_stl(res, out) if out is not None else None

    if json_out:
        d = res.to_dict(include_voxels=False)
        if saved_path is not None:
            d["exported_stl"] = str(saved_path.resolve())
        print(json.dumps(d, indent=2))
        return

    console.print(
        Panel.fit(
            f"[bold cyan]{res.part_name}[/bold cyan] ({res.material_id}, {res.iterations_run} SIMP iters)\n"
            f"Mass: {res.baseline_mass_g:.1f} g -> [bold green]{res.optimized_mass_g:.1f} g[/bold green] "
            f"([cyan]-{res.mass_reduction_pct:.1f}%[/cyan]) | VolFrac: {res.achieved_volume_fraction:.2f}\n"
            f"Stiffness: {res.baseline_stiffness_n_per_mm:.1f} -> {res.optimized_stiffness_n_per_mm:.1f} N/mm "
            f"(Specific Stiffness: [bold]{res.specific_stiffness_gain_pct:+.1f}%[/bold])\n"
            f"Max von Mises: {res.max_von_mises_mpa:.2f} MPa | Safety Factor: [bold green]{res.safety_factor:.2f}x[/bold green] ({res.failure_mode})",
            title="PrintSim 3D SIMP Topology Optimization",
        )
    )
    if saved_path is not None:
        console.print(f"Exported optimized bracket STL to [bold green]{saved_path}[/bold green]")


@app.command("bom")
def bom_cmd(
    spec_path: Path = typer.Argument(..., exists=True, help="Printer YAML spec"),
    stock_length: float = typer.Option(1000.0, "--stock-length", "-s", help="Standard extrusion stock bar length (mm)"),
    csv_out: Optional[Path] = typer.Option(None, "--csv", help="Optional path to export BOM & cut-list as CSV"),
    md_out: Optional[Path] = typer.Option(None, "--markdown", help="Optional path to export BOM as Markdown"),
    json_out: bool = typer.Option(False, "--json", help="Emit machine-readable JSON BOM"),
) -> None:
    """Generate a manufacturable extrusion cut-list, stock bar nesting plan, and full hardware BOM."""
    spec = PrinterSpec.from_yaml(spec_path)
    bom = generate_bom(spec, stock_length_mm=stock_length)

    if csv_out is not None:
        csv_out.write_text(bom.to_csv(), encoding="utf-8")
    if md_out is not None:
        md_out.write_text(bom.to_markdown(), encoding="utf-8")

    if json_out:
        print(json.dumps(bom.to_dict(), indent=2))
        return

    console.print(
        Panel.fit(
            f"[bold cyan]{bom.printer_name}[/bold cyan] ({bom.architecture})\n"
            f"Estimated Total Cost: [bold green]${bom.total_cost_usd:.2f}[/bold green] | "
            f"Total Mass: {bom.total_mass_kg:.2f} kg | "
            f"Extrusions: {bom.total_extrusion_length_m:.2f} m | "
            f"Belts: {bom.total_belt_length_m:.2f} m | "
            f"Fasteners: {bom.total_fasteners} pcs",
            title="PrintSim Manufacturable BOM & Cut-List",
        )
    )

    cut_table = Table(title="Extrusion & Structural Cut-List", show_header=True, header_style="bold magenta")
    cut_table.add_column("Code")
    cut_table.add_column("Profile")
    cut_table.add_column("Role")
    cut_table.add_column("Length (mm)", justify="right")
    cut_table.add_column("Qty", justify="right")
    cut_table.add_column("Mass (g)", justify="right")
    cut_table.add_column("Cost ($)", justify="right")
    for c in bom.cut_list:
        cut_table.add_row(
            c.part_code,
            c.profile_id,
            c.role,
            f"{c.length_mm:.1f}",
            str(c.quantity),
            f"{c.total_mass_g:.0f}",
            f"${c.total_cost_usd:.2f}",
        )
    console.print(cut_table)

    if bom.stock_nesting:
        nest_table = Table(title="Stock Bar Nesting", show_header=True, header_style="bold cyan")
        nest_table.add_column("Profile")
        nest_table.add_column("Stock (mm)", justify="right")
        nest_table.add_column("Bars", justify="right")
        nest_table.add_column("Waste", justify="right")
        for n in bom.stock_nesting:
            nest_table.add_row(
                n.profile_id,
                f"{n.stock_length_mm:.0f}",
                str(n.bars_required),
                f"{n.waste_pct:.1f}% ({n.waste_length_mm:.0f} mm)",
            )
        console.print(nest_table)

    if csv_out is not None:
        console.print(f"Saved CSV BOM to [green]{csv_out}[/green]")
    if md_out is not None:
        console.print(f"Saved Markdown BOM to [green]{md_out}[/green]")


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
