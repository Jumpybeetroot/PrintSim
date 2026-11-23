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

from printsim.api.session import get_live_session, request_snapshot_capture
from printsim.engine import evaluate_printer, optimize_design, optimize_pareto_front
from printsim.engine.evaluator import verify_printer_deep
from printsim.fea.adxl_calibration import calibrate_printer_against_adxl, fit_printer_to_adxl
from printsim.geometry import (
    build_assembly,
    export_assembly_step,
    export_assembly_stl,
    export_panels_dxf,
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
def verify(
    spec_path: Path = typer.Argument(..., exists=True, help="Path to printer YAML spec"),
    filament: str = typer.Option("pla", "--filament", "-f", help="Filament ID"),
    mesh_n: int = typer.Option(8, "--mesh-n", "-m", min=4, max=10, help="Deep MITC4 shell grid density (n x n per bay)"),
    bed_mesh_n: int = typer.Option(16, "--bed-mesh-n", min=8, max=20, help="Deep 2D MITC4 bed plate mesh grid"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON deep verification report"),
) -> None:
    """Run maximum-accuracy 8x8 shell + 16x16 bed MITC4 verification vs balanced 3x3/8x8 default."""
    spec = PrinterSpec.from_yaml(spec_path)
    res = verify_printer_deep(spec, filament_id=filament, deep_mesh_n=mesh_n, deep_bed_mesh_n=bed_mesh_n)

    if json_out:
        print(json.dumps(res, indent=2))
        return

    fast = res["fast_summary"]
    deep = res["deep_summary"]
    conv = res["convergence"]
    status = "[bold green]CONVERGED[/bold green]" if conv["converged"] else "[bold yellow]SHIFTED[/bold yellow]"
    console.print(
        Panel.fit(
            f"[bold cyan]{spec.name}[/bold cyan] ({spec.architecture})\n"
            f"Mesh Refinement: {res['fast_mesh_n']}x{res['fast_mesh_n']} ({fast['num_shell_elements']} shells, {fast['solve_time_ms']:.1f} ms) "
            f"-> {res['deep_mesh_n']}x{res['deep_mesh_n']} ({deep['num_shell_elements']} shells, {deep['solve_time_ms']:.1f} ms)\n"
            f"Status: {status} | Freq delta: {conv['freq_diff_pct']:.2f}% | "
            f"Defl delta: {conv['deflection_diff_pct']:.2f}% | Accel delta: {conv['accel_diff_pct']:.2f}%",
            title="PrintSim Deep MITC4 Shell Verification",
        )
    )
    tbl = Table(show_header=True, header_style="bold magenta")
    tbl.add_column("Metric")
    tbl.add_column(f"Fast ({res['fast_mesh_n']}x{res['fast_mesh_n']})", justify="right")
    tbl.add_column(f"Deep ({res['deep_mesh_n']}x{res['deep_mesh_n']})", justify="right")
    tbl.add_row("Fundamental Freq (Hz)", f"{fast['fundamental_freq_hz']:.2f}", f"{deep['fundamental_freq_hz']:.2f}")
    tbl.add_row("Panel Drum Freq (Hz)", f"{fast['panel_drum_freq_hz']:.2f}", f"{deep['panel_drum_freq_hz']:.2f}")
    tbl.add_row("Combined Deflection (um)", f"{fast['combined_max_deflection_um']:.2f}", f"{deep['combined_max_deflection_um']:.2f}")
    tbl.add_row("Safe Max Accel (mm/s^2)", f"{fast['recommended_max_accel_mm_s2']:.0f}", f"{deep['recommended_max_accel_mm_s2']:.0f}")
    tbl.add_row("Panel Max von Mises (MPa)", f"{fast['panel_max_von_mises_mpa']:.3f}", f"{deep['panel_max_von_mises_mpa']:.3f}")
    console.print(tbl)


@app.command("calibrate-adxl")
def calibrate_adxl_cmd(
    spec_path: Path = typer.Argument(..., exists=True, help="Path to printer YAML spec"),
    x_csv: Optional[Path] = typer.Option(None, "--x-csv", "-x", exists=True, help="Klipper PSD or raw ADXL345 CSV for X axis"),
    y_csv: Optional[Path] = typer.Option(None, "--y-csv", "-y", exists=True, help="Klipper PSD or raw ADXL345 CSV for Y axis"),
    max_freq_hz: float = typer.Option(200.0, "--max-freq", min=20.0, max=500.0, help="Maximum frequency span (Hz)"),
    fit: bool = typer.Option(False, "--fit", help="Auto-fit belt pretension and rail/joint stiffness knobs to match measured peaks"),
    json_out: bool = typer.Option(False, "--json", help="Emit machine-readable JSON calibration report"),
) -> None:
    """Compare real Klipper ADXL345 / Shake&Tune CSV measurements against PrintSim modal FEA."""
    if x_csv is None and y_csv is None:
        raise typer.BadParameter("Provide at least one measured CSV (--x-csv and/or --y-csv).")

    spec = PrinterSpec.from_yaml(spec_path)
    axis_x_text = x_csv.read_text(encoding="utf-8", errors="replace") if x_csv is not None else None
    axis_y_text = y_csv.read_text(encoding="utf-8", errors="replace") if y_csv is not None else None

    if fit:
        fit_report = fit_printer_to_adxl(
            spec,
            axis_x_csv=axis_x_text,
            axis_y_csv=axis_y_text,
            max_freq_hz=max_freq_hz,
        )
        if json_out:
            print(json.dumps(fit_report.to_dict(), indent=2))
            return

        console.print(
            Panel.fit(
                f"[bold cyan]{fit_report.printer_name}[/bold cyan] ({spec.architecture})\n"
                f"Mean |Peak Error|: {fit_report.before_mean_abs_error_pct:.2f}% -> "
                f"[bold green]{fit_report.after_mean_abs_error_pct:.2f}%[/bold green]\n"
                f"{fit_report.summary}",
                title="PrintSim ADXL345 Digital-Twin Auto-Fit",
            )
        )

        ax_tbl = Table(show_header=True, header_style="bold magenta")
        ax_tbl.add_column("Axis")
        ax_tbl.add_column("Meas Peak", justify="right")
        ax_tbl.add_column("Before Sim (Err%)", justify="right")
        ax_tbl.add_column("After Sim (Err%)", justify="right")
        for ax_key in fit_report.after_axes:
            b_ax = fit_report.before_axes[ax_key]
            a_ax = fit_report.after_axes[ax_key]
            ax_tbl.add_row(
                ax_key,
                f"{a_ax.measured_peak_hz:.1f} Hz",
                f"{b_ax.simulated_peak_hz:.1f} Hz ({b_ax.peak_error_pct:+.1f}%)",
                f"{a_ax.simulated_peak_hz:.1f} Hz ({a_ax.peak_error_pct:+.1f}%)",
            )
        console.print(ax_tbl)

        if fit_report.changes:
            chg_tbl = Table(title="Fitted Parameter Changes", show_header=True, header_style="bold cyan")
            chg_tbl.add_column("Field")
            chg_tbl.add_column("Parameter")
            chg_tbl.add_column("Before", justify="right")
            chg_tbl.add_column("After", justify="right")
            chg_tbl.add_column("Reason")
            for chg in fit_report.changes:
                chg_tbl.add_row(chg.field, chg.label, str(chg.before), str(chg.after), chg.reason)
            console.print(chg_tbl)
        else:
            console.print("\n[dim]No parameter changes applied.[/dim]")
        return

    report = calibrate_printer_against_adxl(
        spec,
        axis_x_csv=axis_x_text,
        axis_y_csv=axis_y_text,
        max_freq_hz=max_freq_hz,
    )

    if json_out:
        print(json.dumps(report.to_dict(), indent=2))
        return

    console.print(
        Panel.fit(
            f"[bold cyan]{report.printer_name}[/bold cyan] ({spec.architecture})\n"
            f"{report.overall_summary}",
            title="PrintSim ADXL345 Measured-vs-Simulated Calibration",
        )
    )

    tbl = Table(show_header=True, header_style="bold magenta")
    tbl.add_column("Axis")
    tbl.add_column("Source Format")
    tbl.add_column("Meas Peak", justify="right")
    tbl.add_column("Sim Peak", justify="right")
    tbl.add_column("Delta / Err%", justify="right")
    tbl.add_column("Damping (Meas / Sim)", justify="right")
    tbl.add_column("Shaper (Meas / Sim)")
    tbl.add_column("Max Accel (Meas / Sim)", justify="right")
    tbl.add_column("Corr (r)", justify="right")

    for ax_key, ax_res in report.axes.items():
        err_color = "green" if abs(ax_res.peak_error_pct) <= 5.0 else ("yellow" if abs(ax_res.peak_error_pct) <= 10.0 else "red")
        tbl.add_row(
            ax_key,
            ax_res.source_format,
            f"{ax_res.measured_peak_hz:.1f} Hz",
            f"{ax_res.simulated_peak_hz:.1f} Hz",
            f"[{err_color}]{ax_res.peak_delta_hz:+.1f} Hz ({ax_res.peak_error_pct:+.1f}%)[/{err_color}]",
            f"{ax_res.measured_damping_ratio:.4f} / {ax_res.simulated_damping_ratio:.4f}",
            f"{ax_res.measured_shaper}@{ax_res.measured_shaper_freq_hz:.1f} / {ax_res.simulated_shaper}@{ax_res.simulated_shaper_freq_hz:.1f}",
            f"{ax_res.measured_max_accel_mm_s2:.0f} / {ax_res.simulated_max_accel_mm_s2:.0f}",
            f"{ax_res.spectral_correlation:.3f}",
        )
    console.print(tbl)

    console.print("\n[bold]Diagnostics:[/bold]")
    for ax_res in report.axes.values():
        for diag in ax_res.diagnostics:
            console.print(f"  • {diag}")

    console.print("\n[bold]Recommended Adjustments:[/bold]")
    for adj in report.recommended_adjustments:
        console.print(f"  • {adj}")


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


@app.command("export-dxf")
def export_dxf_cmd(
    spec_path: Path = typer.Argument(..., exists=True, help="Printer YAML spec"),
    out: Path = typer.Option(Path("printer_plates.dxf"), "--out", "-o", help="Output 2D CNC/laser DXF path"),
    part: str = typer.Option(
        "all_plates",
        "--part",
        "-p",
        help="Plate/kit part: all_plates | side_panel | rear_panel | top_panel | trident_backbone | corner_gusset",
    ),
) -> None:
    """Export 2D CNC/laser-cut DXF panels and structural plates (R2010, mm)."""
    spec = PrinterSpec.from_yaml(spec_path)
    saved = export_panels_dxf(spec, out, part=part)
    console.print(f"Exported 2D CNC/laser DXF ({part}) to [bold green]{saved}[/bold green]")


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


@app.command("studio-state")
def studio_state_cmd(
    json_out: bool = typer.Option(False, "--json", help="Emit machine-readable JSON live studio session state"),
    yaml_out: bool = typer.Option(False, "--yaml", help="Emit only the active live YAML specification"),
    capture: bool = typer.Option(False, "--capture", help="Request and wait for a live 3D viewport PNG capture"),
    frames: int = typer.Option(2, "--frames", min=1, max=2, help="Number of snapshot frames to capture (1 or 2)"),
    delay_ms: int = typer.Option(420, "--delay-ms", min=0, max=5000, help="Delay between 2-frame modal captures (ms)"),
    timeout: float = typer.Option(5.0, "--timeout", min=0.2, max=30.0, help="Timeout waiting for browser snapshot upload (s)"),
) -> None:
    """Inspect the live PrintSim Web Studio session and optionally capture viewport snapshots."""
    capture_res: dict | None = None
    if capture:
        capture_res = request_snapshot_capture(
            frames=frames,
            delay_ms=delay_ms,
            timeout_s=timeout,
        )

    session = get_live_session()

    if yaml_out:
        yaml_text = session.get("yaml_text")
        if not yaml_text:
            console.print("[yellow]No YAML specification recorded in the live Studio session yet.[/yellow]")
            raise typer.Exit(code=1)
        print(str(yaml_text).rstrip())
        return

    if json_out:
        out = dict(session)
        if capture_res is not None:
            out["capture"] = capture_res
        print(json.dumps(out, indent=2))
        return

    conn_badge = "[bold green]CONNECTED[/bold green]" if session.get("connected") else "[bold yellow]OFFLINE[/bold yellow]"
    printer_name = session.get("printer_name") or "(none)"
    arch = session.get("architecture") or "-"
    console.print(
        Panel.fit(
            f"[bold cyan]{printer_name}[/bold cyan] ({arch})  |  Browser: {conn_badge}\n"
            f"View: {session.get('view_mode') or '-'}  |  Mode idx: {session.get('selected_mode_idx')}  |  "
            f"Load case: {session.get('fea_load_case') or '-'}  |  Filament: {session.get('filament_id') or '-'}\n"
            f"Preset: {session.get('active_preset_id') or '-'}  |  Session dir: {session.get('session_dir')}",
            title="PrintSim Live Studio Session",
        )
    )

    sim = session.get("simulation_summary")
    if isinstance(sim, dict):
        tbl = Table(show_header=True, header_style="bold magenta")
        tbl.add_column("Subsystem")
        tbl.add_column("Live Telemetry")
        scores = sim.get("scores") or {}
        st = sim.get("static_fea") or {}
        mod = sim.get("modal_fea") or {}
        th = sim.get("thermal") or {}
        sens = sim.get("sensitivity") or {}
        tbl.add_row(
            "Scores",
            f"Composite: {scores.get('composite')} | Struct: {scores.get('structural')} | "
            f"Dyn: {scores.get('dynamics')} | Therm: {scores.get('thermal')}",
        )
        tbl.add_row(
            "Static FEA",
            f"Peak defl: {st.get('combined_max_deflection_um')} um | "
            f"Gravity sag: {st.get('gravity_sag_toolhead_um')} um | "
            f"Max vM: {st.get('max_von_mises_mpa')} MPa",
        )
        tbl.add_row(
            "Modal & Shaper",
            f"f1: {mod.get('fundamental_freq_hz')} Hz | "
            f"Worst mode: #{mod.get('worst_print_mode_number')} | "
            f"Rec max accel: {mod.get('recommended_max_accel_mm_s2')} mm/s^2",
        )
        tbl.add_row(
            "Thermal",
            f"Max flow: {th.get('max_volumetric_flow_mm3_s')} mm^3/s ({th.get('limiting_factor')}) | "
            f"Cold throat: {th.get('cold_throat_temp_c')} C",
        )
        tbl.add_row(
            "Sensitivity",
            f"Primary hotspot: {sens.get('primary_problem_subsystem')}",
        )
        console.print(tbl)

    active_mode = session.get("active_mode_detail")
    if isinstance(active_mode, dict):
        console.print(
            f"\n[bold]Selected Mode #{active_mode.get('mode_number')}[/bold]: "
            f"{active_mode.get('frequency_hz')} Hz — {active_mode.get('mode_label')} "
            f"(impact={active_mode.get('print_impact_score')}, area={active_mode.get('dominant_problem_area')})"
        )

    if capture_res is not None:
        if capture_res.get("captured"):
            console.print(f"\n[bold green]Captured viewport PNGs:[/bold green] {capture_res.get('snapshot_paths')}")
        else:
            console.print(f"\n[yellow]Capture incomplete:[/yellow] {capture_res.get('message')}")
    elif isinstance(session.get("latest_snapshots"), dict):
        snaps = session["latest_snapshots"]
        console.print(f"\n[bold]Latest Snapshots[/bold] ({snaps.get('captured_at')}): {snaps.get('paths')}")


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


# ---------------------------------------------------------------------------
# Knowledge Base (`printsim kb ...`) subcommand group
# ---------------------------------------------------------------------------

kb_app = typer.Typer(
    help="Local 3D printer engineering touchstone database & calibration",
    add_completion=False,
)
app.add_typer(kb_app, name="kb")


@kb_app.command("seed")
def kb_seed_cmd(
    db: Optional[Path] = typer.Option(None, "--db", help="Path to SQLite KB file"),
    docs_dir: Optional[Path] = typer.Option(None, "--docs-dir", help="Optional benchmark docs directory"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON summary"),
) -> None:
    """Seed the local KB from built-in empirical benchmark docs (e.g. monolith_discord_benchmarks.md)."""
    from printsim.kb import TouchstoneKB, seed_from_benchmark_docs

    with TouchstoneKB(db) as kb:
        res = seed_from_benchmark_docs(kb, docs_dir=docs_dir)
        out = {"db_path": str(kb.db_path), **res}

    if json_out:
        print(json.dumps(out, indent=2))
        return

    console.print(
        Panel.fit(
            f"Database: [bold cyan]{out['db_path']}[/bold cyan]\n"
            f"Docs processed: {out['files_processed']} | Raw items added: {out['raw_items_added']}\n"
            f"Builds: [bold green]+{out['builds_added']}[/bold green] | "
            f"Weights: +{out.get('weights_added', 0)} | "
            f"Material tests: +{out['material_tests_added']} | "
            f"Troubleshooting cases: +{out['troubleshooting_added']}",
            title="PrintSim KB Seed",
        )
    )


@kb_app.command("ingest")
def kb_ingest_cmd(
    path: Path = typer.Argument(..., exists=True, help="Local file or directory (Discord JSON, JSONL, MD/TXT, Klipper CSV)"),
    db: Optional[Path] = typer.Option(None, "--db", help="Path to SQLite KB file"),
    min_score: float = typer.Option(0.20, "--min-score", min=0.0, max=1.0, help="Minimum engineering signal score"),
    auto_approve: bool = typer.Option(False, "--auto-approve", help="Approve kept items and insert structured records immediately"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON summary"),
) -> None:
    """Batch-ingest local Discord exports, Klipper shaper CSVs, JSON/JSONL, or Markdown notes."""
    from printsim.kb import TouchstoneKB, ingest_local_path

    with TouchstoneKB(db) as kb:
        res = ingest_local_path(kb, path, min_score=min_score, auto_approve=auto_approve)
        out = {"db_path": str(kb.db_path), **res}

    if json_out:
        print(json.dumps(out, indent=2))
        return

    console.print(
        Panel.fit(
            f"Database: [bold cyan]{out['db_path']}[/bold cyan]\n"
            f"Files scanned: {out['files_scanned']} | Items seen: {out['items_seen']}\n"
            f"Kept: [bold green]{out['items_kept']}[/bold green] | "
            f"Skipped (low signal): {out['items_skipped_low_signal']} | "
            f"Deduplicated: {out['items_deduplicated']}\n"
            f"Structured added — Builds: {out['builds_added']}, Weights: {out['weights_added']}, "
            f"Materials: {out['material_tests_added']}, Troubleshooting: {out['troubleshooting_added']}",
            title="PrintSim KB Local Ingest",
        )
    )


@kb_app.command("sync-public")
def kb_sync_public_cmd(
    db: Optional[Path] = typer.Option(None, "--db", help="Path to SQLite KB file"),
    source: str = typer.Option("all", "--source", help="all | discourse | github | answer_overflow"),
    max_items: int = typer.Option(15, "--max-items", min=1, max=100, help="Max topics/issues/pages per source"),
    include_answer_overflow: bool = typer.Option(
        True,
        "--include-answer-overflow/--no-include-answer-overflow",
        help="Include public Answer Overflow Discord-mirror threads when --source=all",
    ),
    auto_approve: bool = typer.Option(False, "--auto-approve", help="Auto-approve harvested items"),
    delay_s: float = typer.Option(0.8, "--delay", min=0.0, help="Polite delay between HTTP requests (seconds)"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON summary"),
) -> None:
    """Politely harvest public 3D printer engineering posts from Discourse, GitHub, and Answer Overflow."""
    from printsim.kb import (
        DEFAULT_PUBLIC_DISCOURSE_TARGETS,
        DEFAULT_PUBLIC_GITHUB_REPOS,
        TouchstoneKB,
        harvest_answer_overflow,
        harvest_discourse,
        harvest_github_issues,
    )

    src = source.strip().lower()
    if src not in ("all", "discourse", "github", "answer_overflow"):
        raise typer.BadParameter("--source must be one of: all, discourse, github, answer_overflow")

    reports: list[dict] = []
    with TouchstoneKB(db) as kb:
        if src in ("all", "discourse"):
            for base_url, queries in DEFAULT_PUBLIC_DISCOURSE_TARGETS:
                reports.append(
                    harvest_discourse(
                        kb,
                        base_url=base_url,
                        queries=queries,
                        max_topics=max_items,
                        auto_approve=auto_approve,
                        delay_s=delay_s,
                    )
                )
        if src in ("all", "github"):
            for repo, queries in DEFAULT_PUBLIC_GITHUB_REPOS:
                reports.append(
                    harvest_github_issues(
                        kb,
                        repo=repo,
                        queries=queries,
                        max_items=max_items,
                        auto_approve=auto_approve,
                        delay_s=delay_s,
                    )
                )
        if src == "answer_overflow" or (src == "all" and include_answer_overflow):
            reports.append(
                harvest_answer_overflow(
                    kb,
                    max_pages=max_items,
                    auto_approve=auto_approve,
                    delay_s=delay_s,
                )
            )
        out = {"db_path": str(kb.db_path), "source": src, "reports": reports}

    if json_out:
        print(json.dumps(out, indent=2))
        return

    tbl = Table(title="PrintSim KB Public Sync", show_header=True, header_style="bold magenta")
    tbl.add_column("Source")
    tbl.add_column("Seen", justify="right")
    tbl.add_column("Kept", justify="right")
    tbl.add_column("Low Signal", justify="right")
    tbl.add_column("Errors", justify="right")
    for rep in reports:
        tbl.add_row(
            str(rep.get("source") or rep.get("repo") or "-"),
            str(rep.get("items_seen", 0)),
            str(rep.get("items_kept", 0)),
            str(rep.get("items_skipped_low_signal", 0)),
            str(len(rep.get("errors") or [])),
        )
    console.print(tbl)


@kb_app.command("discord-scan")
def kb_discord_scan_cmd(
    channel: str = typer.Option(..., "--channel", "-c", help="Discord channel ID to scan with an official Bot token"),
    limit: int = typer.Option(500, "--limit", min=1, max=10000, help="Max messages to scan"),
    out: Path = typer.Option(Path("review_queue"), "--out", "-o", help="Directory for review_queue.json & review_queue.md"),
    db: Optional[Path] = typer.Option(None, "--db", help="Path to SQLite KB file"),
    bot_token: Optional[str] = typer.Option(
        None,
        "--bot-token",
        envvar="DISCORD_BOT_TOKEN",
        help="Official Discord Bot token (never use a user self-bot token)",
    ),
    min_score: float = typer.Option(0.25, "--min-score", min=0.0, max=1.0, help="Minimum signal score"),
    download_attachments: Optional[Path] = typer.Option(
        None,
        "--download-attachments",
        help="Optional directory to save matching shaper PNG/CSV attachments",
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON summary"),
) -> None:
    """Scan a Discord channel using an official Discord Bot token and export a human review queue."""
    from printsim.kb import TouchstoneKB, scan_discord_channel

    with TouchstoneKB(db) as kb:
        res = scan_discord_channel(
            kb,
            channel_id=channel,
            bot_token=bot_token,
            limit=limit,
            min_score=min_score,
            download_attachments_dir=download_attachments,
            review_out_dir=out,
        )
        payload = {"db_path": str(kb.db_path), **res}

    if json_out:
        print(json.dumps(payload, indent=2))
        return

    rev = payload.get("review_export") or {}
    console.print(
        Panel.fit(
            f"Channel: [bold cyan]{payload['channel_id']}[/bold cyan] | Pages: {payload['pages_fetched']}\n"
            f"Messages scanned: {payload['messages_scanned']} | Kept: [bold green]{payload['items_kept']}[/bold green] | "
            f"Deduplicated: {payload['items_deduplicated']} | Attachments saved: {payload['attachments_downloaded']}\n"
            f"Review JSON: {rev.get('json_path', '-')}\n"
            f"Review Markdown: {rev.get('markdown_path', '-')}",
            title="PrintSim KB Discord Bot Scan",
        )
    )


@kb_app.command("review")
def kb_review_cmd(
    out: Path = typer.Option(Path("review_queue"), "--out", "-o", help="Output directory for review_queue.json & review_queue.md"),
    min_score: float = typer.Option(0.25, "--min-score", min=0.0, max=1.0, help="Minimum signal score"),
    limit: int = typer.Option(100, "--limit", min=1, max=1000, help="Maximum pending items to export"),
    db: Optional[Path] = typer.Option(None, "--db", help="Path to SQLite KB file"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON summary"),
) -> None:
    """Export pending high-signal KB items to review_queue.json and review_queue.md."""
    from printsim.kb import TouchstoneKB

    with TouchstoneKB(db) as kb:
        res = kb.export_review_queue(out, min_score=min_score, limit=limit)
        payload = {"db_path": str(kb.db_path), **res}

    if json_out:
        print(json.dumps(payload, indent=2))
        return

    console.print(
        Panel.fit(
            f"Exported [bold green]{payload['count']}[/bold green] pending items (min_score >= {min_score:.2f})\n"
            f"JSON: {payload['json_path']}\n"
            f"Markdown: {payload['markdown_path']}",
            title="PrintSim KB Review Queue",
        )
    )


@kb_app.command("approve")
def kb_approve_cmd(
    review_json: Path = typer.Argument(..., exists=True, help="Path to edited review_queue.json"),
    db: Optional[Path] = typer.Option(None, "--db", help="Path to SQLite KB file"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON summary"),
) -> None:
    """Apply approve/reject decisions from review_queue.json and insert approved structured records."""
    from printsim.kb import TouchstoneKB

    with TouchstoneKB(db) as kb:
        res = kb.apply_review_decisions(review_json)
        payload = {"db_path": str(kb.db_path), **res}

    if json_out:
        print(json.dumps(payload, indent=2))
        return

    console.print(
        Panel.fit(
            f"Approved: [bold green]{payload['approved']}[/bold green] | "
            f"Rejected: [bold red]{payload['rejected']}[/bold red] | "
            f"Still pending: {payload['pending']}\n"
            f"Inserted — Builds: {payload['builds_added']}, Weights: {payload['weights_added']}, "
            f"Materials: {payload['material_tests_added']}, Troubleshooting: {payload['troubleshooting_added']}",
            title="PrintSim KB Review Applied",
        )
    )


@kb_app.command("search")
def kb_search_cmd(
    query: str = typer.Argument(..., help="Full-text search query"),
    db: Optional[Path] = typer.Option(None, "--db", help="Path to SQLite KB file"),
    table: Optional[str] = typer.Option(None, "--table", "-t", help="Optional filter: builds|component_weights|material_tests|troubleshooting_cases|raw_items"),
    limit: int = typer.Option(15, "--limit", "-n", min=1, max=200, help="Maximum hits to show"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON results"),
) -> None:
    """Instant FTS5 search across builds, weights, materials, troubleshooting cases, and raw items."""
    from printsim.kb import TouchstoneKB

    with TouchstoneKB(db) as kb:
        hits = kb.search(query, table=table, limit=limit)

    if json_out:
        print(json.dumps(hits, indent=2))
        return

    if not hits:
        console.print(f"[yellow]No KB matches found for '{query}'.[/yellow]")
        return

    tbl = Table(title=f"KB Search: {query} ({len(hits)} hits)", show_header=True, header_style="bold magenta")
    tbl.add_column("Type")
    tbl.add_column("ID", justify="right")
    tbl.add_column("Title")
    tbl.add_column("Snippet")
    for h in hits:
        tbl.add_row(
            str(h["item_type"]),
            str(h["ref_id"]),
            str(h["title"])[:42],
            str(h["snippet"])[:90],
        )
    console.print(tbl)


@kb_app.command("stats")
def kb_stats_cmd(
    db: Optional[Path] = typer.Option(None, "--db", help="Path to SQLite KB file"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON stats"),
) -> None:
    """Show record counts and source/review breakdowns for the local KB."""
    from printsim.kb import TouchstoneKB

    with TouchstoneKB(db) as kb:
        st = kb.stats()

    if json_out:
        print(json.dumps(st, indent=2))
        return

    console.print(
        Panel.fit(
            f"Database: [bold cyan]{st['db_path']}[/bold cyan]\n"
            f"Raw Items: {st['raw_items']} ({st['raw_by_status']})\n"
            f"Builds: [bold green]{st['builds']}[/bold green] | "
            f"Weights: {st['component_weights']} | "
            f"Material Tests: {st['material_tests']} | "
            f"Troubleshooting: {st['troubleshooting_cases']} | "
            f"FTS5 Entries: {st['fts_entries']}",
            title="PrintSim Touchstone KB Stats",
        )
    )


@kb_app.command("calibrate")
def kb_calibrate_cmd(
    db: Optional[Path] = typer.Option(None, "--db", help="Path to SQLite KB file"),
    examples_dir: Optional[Path] = typer.Option(None, "--examples-dir", help="Optional directory containing PrintSim YAML presets"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON calibration report"),
) -> None:
    """Compare PrintSim modal/shaper predictions against real measured builds in the local KB."""
    from printsim.kb import TouchstoneKB, run_kb_calibration

    with TouchstoneKB(db) as kb:
        rep = run_kb_calibration(kb, examples_dir=examples_dir)

    if json_out:
        print(json.dumps(rep, indent=2))
        return

    console.print(
        Panel.fit(
            f"Builds evaluated: [bold cyan]{rep['builds_evaluated']}[/bold cyan] "
            f"(across {rep['presets_evaluated']} reference presets)\n"
            f"X MAPE: [bold]{rep['x_mape_pct']}%[/bold] (mean signed: {rep['x_mean_signed_error_pct']}%) | "
            f"Y MAPE: [bold]{rep['y_mape_pct']}%[/bold] (mean signed: {rep['y_mean_signed_error_pct']}%) | "
            f"Overall MAPE: [bold green]{rep['overall_mape_pct']}%[/bold green]",
            title="PrintSim vs. Real-World KB Calibration",
        )
    )
    tbl = Table(show_header=True, header_style="bold magenta")
    tbl.add_column("Real Build")
    tbl.add_column("Matched Preset")
    tbl.add_column("Meas X", justify="right")
    tbl.add_column("Pred X", justify="right")
    tbl.add_column("X Err%", justify="right")
    tbl.add_column("Meas Y", justify="right")
    tbl.add_column("Pred Y", justify="right")
    tbl.add_column("Y Err%", justify="right")
    for r in rep["results"]:
        tbl.add_row(
            str(r["build_name"]),
            str(r["preset_file"]),
            f"{r['measured_x_hz']:.1f} Hz" if r["measured_x_hz"] is not None else "-",
            f"{r['predicted_x_hz']:.1f} Hz",
            f"{r['x_error_pct']:+.1f}%" if r["x_error_pct"] is not None else "-",
            f"{r['measured_y_hz']:.1f} Hz" if r["measured_y_hz"] is not None else "-",
            f"{r['predicted_y_hz']:.1f} Hz",
            f"{r['y_error_pct']:+.1f}%" if r["y_error_pct"] is not None else "-",
        )
    console.print(tbl)


@kb_app.command("gaps")
def kb_gaps_cmd(
    db: Optional[Path] = typer.Option(None, "--db", help="Path to SQLite KB file"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON coverage gap report"),
) -> None:
    """Report which real-world build classes, weights, and material touchstones are missing from the KB."""
    from printsim.kb import TouchstoneKB, analyze_coverage_gaps

    with TouchstoneKB(db) as kb:
        rep = analyze_coverage_gaps(kb)

    if json_out:
        print(json.dumps(rep, indent=2))
        return

    tbl = Table(title="KB Coverage Gaps — Real Shaper Builds", show_header=True, header_style="bold magenta")
    tbl.add_column("Printer Class")
    tbl.add_column("Have / Target", justify="right")
    tbl.add_column("Status")
    tbl.add_column("Where to Look")
    for g in rep["build_gaps"]:
        color = "green" if g["status"] == "covered" else ("yellow" if g["status"] == "thin" else "red")
        tbl.add_row(g["label"], f"{g['have']} / {g['target']}", f"[{color}]{g['status'].upper()}[/{color}]", g["suggested_source"])
    console.print(tbl)

    wtbl = Table(title="KB Coverage Gaps — Measured Weights & Materials", show_header=True, header_style="bold cyan")
    wtbl.add_column("Topic")
    wtbl.add_column("Coverage", justify="right")
    wtbl.add_column("Status")
    wtbl.add_column("Where to Look")
    for g in rep["weight_gaps"]:
        color = "green" if g["status"] == "covered" else ("yellow" if g["status"] == "thin" else "red")
        wtbl.add_row(g["label"], f"{g['have']} / {g['target']}", f"[{color}]{g['status'].upper()}[/{color}]", g["suggested_source"])
    for g in rep["material_gaps"]:
        color = "green" if g["status"] == "covered" else "red"
        wtbl.add_row(g["label"], "1 / 1" if g["status"] == "covered" else "0 / 1", f"[{color}]{g['status'].upper()}[/{color}]", g["suggested_source"])
    console.print(wtbl)


if __name__ == "__main__":
    app()
