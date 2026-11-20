"""Unified multiphysics simulation pipeline & structured scorecard for humans and AI agents."""

from __future__ import annotations

from dataclasses import dataclass, field

from printsim.fea.modal import ModalResult, solve_modal_fea
from printsim.fea.solver import StaticFEAResult, solve_static_fea
from printsim.gcode.parser import generate_benchmark_gcode
from printsim.gcode.simulator import VirtualPrintReport, simulate_gcode
from printsim.geometry.builder import AssemblyModel, build_assembly
from printsim.kinematics.collision import CollisionReport, check_frame_clearance
from printsim.kinematics.solvers import KinematicsReport, analyze_kinematics
from printsim.schema import PrinterSpec
from printsim.thermal.hotend import HotendThermalReport, simulate_hotend_thermal


@dataclass
class ScoreBreakdown:
    structural: float  # 0..100 (stiffness, deflection, resonance, safety factor)
    dynamics: float  # 0..100 (accel headroom, torque margin, belt integrity)
    thermal: float  # 0..100 (volumetric flow, heatup speed, zero heat creep)
    print_quality: float  # 0..100 (virtual G-code defect-free execution)
    efficiency: float  # 0..100 (build-to-frame packaging, mass-to-stiffness, cost)
    composite: float  # weighted overall 0..100

    def to_dict(self) -> dict:
        return {
            "structural": round(self.structural, 1),
            "dynamics": round(self.dynamics, 1),
            "thermal": round(self.thermal, 1),
            "print_quality": round(self.print_quality, 1),
            "efficiency": round(self.efficiency, 1),
            "composite": round(self.composite, 1),
        }


@dataclass
class SimulationReport:
    spec: PrinterSpec
    assembly: AssemblyModel
    collision: CollisionReport
    static_fea: StaticFEAResult
    modal_fea: ModalResult
    kinematics: KinematicsReport
    thermal: HotendThermalReport
    virtual_print: VirtualPrintReport
    scores: ScoreBreakdown
    diagnostics: list[str] = field(default_factory=list)

    def to_dict(self, include_toolpath_segments: bool = True) -> dict:
        vp = self.virtual_print.to_dict(max_segments=1500 if include_toolpath_segments else 0)
        return {
            "printer_name": self.spec.name,
            "architecture": self.spec.architecture,
            "scores": self.scores.to_dict(),
            "diagnostics": self.diagnostics,
            "geometry": self.assembly.to_geometry_dict(),
            "collision": self.collision.to_dict(),
            "static_fea": self.static_fea.to_dict(),
            "modal_fea": self.modal_fea.to_dict(),
            "kinematics": self.kinematics.to_dict(),
            "thermal": self.thermal.to_dict(),
            "virtual_print": vp,
        }


def _compute_scores_and_diagnostics(
    spec: PrinterSpec,
    assembly: AssemblyModel,
    collision: CollisionReport,
    static_fea: StaticFEAResult,
    modal_fea: ModalResult,
    kinematics: KinematicsReport,
    thermal: HotendThermalReport,
    vprint: VirtualPrintReport,
) -> tuple[ScoreBreakdown, list[str]]:
    diag: list[str] = []

    # 1. Structural score (deflection < 15um ideal, fundamental freq > 60Hz ideal, SF >= 5)
    defl_pen = min(45.0, static_fea.combined_max_deflection_um * 0.65)
    freq_bonus = min(45.0, modal_fea.fundamental_freq_hz * 0.68)
    sf_pen = 0.0 if static_fea.min_safety_factor >= 4.0 else (4.0 - static_fea.min_safety_factor) * 10.0
    structural = max(0.0, min(100.0, 55.0 + freq_bonus - defl_pen - sf_pen))

    if modal_fea.fundamental_freq_hz < 40.0:
        diag.append(
            f"Low frame natural frequency ({modal_fea.fundamental_freq_hz:.1f} Hz): "
            "add diagonal bracing ('rear_cross' or 'full_truss') or step up frame profile (3030)."
        )
    if static_fea.combined_max_deflection_um > 40.0:
        diag.append(
            f"High peak deflection ({static_fea.combined_max_deflection_um:.1f} um) under acceleration: "
            "stiffen gantry beam or reduce moving toolhead mass."
        )

    # 2. Dynamics score (effective resonance accel, motor torque margin, belt tension)
    rec_a = modal_fea.recommended_max_accel_mm_s2
    accel_score = min(55.0, (rec_a / 12000.0) * 55.0)
    torque_score = min(30.0, kinematics.torque_safety_margin * 15.0)
    belt_score = 0.0 if kinematics.belt_slip_risk else 15.0
    dynamics = max(0.0, min(100.0, accel_score + torque_score + belt_score))
    if kinematics.step_skip_risk:
        diag.append(
            f"Stepper torque margin is low ({kinematics.torque_safety_margin:.2f}x): "
            f"limit accel to <= {kinematics.max_safe_accel_motor_mm_s2:.0f} mm/s^2 or use stronger/48V steppers."
        )
    if kinematics.belt_slip_risk:
        diag.append(
            f"Belt slack-side slip risk at {spec.kinematics.max_accel_mm_s2:.0f} mm/s^2: "
            "increase belt pretension or upgrade to 9mm/12mm GT2 belt."
        )
    if not kinematics.delta_workspace_valid:
        diag.append("Delta arm length cannot reach full build volume corners; increase delta_arm_length_mm.")

    # 3. Thermal score (volumetric flow, heatup speed, heat creep safety)
    flow_score = min(55.0, (thermal.max_volumetric_flow_mm3_s / 35.0) * 55.0)
    heatup_score = max(0.0, min(25.0, 25.0 * (1.0 - max(0.0, thermal.heatup_time_215c_s - 25.0) / 140.0)))
    creep_score = 0.0 if thermal.heat_creep_risk else 20.0
    thermal_score = max(0.0, min(100.0, flow_score + heatup_score + creep_score))
    if thermal.heat_creep_risk:
        diag.append(
            f"Heat creep risk (cold-side throat {thermal.cold_side_temp_c:.1f} C > 52 C): "
            "use a titanium bi-metal heatbreak or improve heatsink cooling."
        )
    if thermal.max_volumetric_flow_mm3_s < 18.0:
        diag.append(
            f"Hotend volumetric flow capped at {thermal.max_volumetric_flow_mm3_s:.1f} mm^3/s "
            f"(limited by {thermal.flow_limiting_factor}): lengthen melt zone or increase heater wattage."
        )

    # 4. Print quality score (directly from virtual G-code print)
    print_quality = vprint.print_quality_score
    if vprint.underextrusion_segments > 0:
        diag.append(
            f"Virtual print detected {vprint.underextrusion_segments} under-extruded moves "
            f"(peak {vprint.peak_volumetric_flow_mm3_s:.1f} vs hotend max {thermal.max_volumetric_flow_mm3_s:.1f} mm^3/s)."
        )
    if vprint.ringing_segments > 0:
        diag.append(
            f"Virtual print detected {vprint.ringing_segments} moves with visible ghosting/ringing "
            f"(recommended input shaper max accel: {modal_fea.recommended_max_accel_mm_s2:.0f} mm/s^2)."
        )

    # 5. Packaging / mass / cost efficiency
    pkg_score = min(45.0, kinematics.packaging_efficiency_pct * 1.4)
    mass_score = max(0.0, min(30.0, 30.0 * (1.0 - max(0.0, assembly.total_mass_kg - 8.0) / 28.0)))
    cost_score = max(0.0, min(25.0, 25.0 * (1.0 - max(0.0, assembly.estimated_material_cost_usd - 220.0) / 800.0)))
    efficiency = max(0.0, min(100.0, pkg_score + mass_score + cost_score))

    # Hard collision penalty
    collision_pen = 30.0 if collision.has_collision else 0.0
    for issue in collision.issues:
        diag.append(f"COLLISION: {issue}")

    composite = max(
        0.0,
        min(
            100.0,
            0.26 * structural
            + 0.24 * dynamics
            + 0.18 * thermal_score
            + 0.22 * print_quality
            + 0.10 * efficiency
            - collision_pen,
        ),
    )

    if not diag:
        diag.append("Design passes all structural, kinematic, thermal, and virtual print checks cleanly.")

    return (
        ScoreBreakdown(
            structural=structural,
            dynamics=dynamics,
            thermal=thermal_score,
            print_quality=print_quality,
            efficiency=efficiency,
            composite=composite,
        ),
        diag,
    )


def evaluate_printer(
    spec: PrinterSpec,
    gcode_text: str | None = None,
    toolhead_rel_pos: tuple[float, float, float] = (0.5, 0.5, 0.5),
    filament_id: str = "pla",
) -> SimulationReport:
    """Run the full multiphysics simulation suite on a PrinterSpec."""
    assembly = build_assembly(spec, toolhead_rel_pos=toolhead_rel_pos)
    collision = check_frame_clearance(assembly)
    static_fea = solve_static_fea(assembly)
    modal_fea = solve_modal_fea(assembly)
    kinematics = analyze_kinematics(assembly)
    thermal = simulate_hotend_thermal(spec, filament_id=filament_id)

    if gcode_text is None:
        bx, by, _ = spec.kinematics.build_volume_mm
        cube_size = min(40.0, bx * 0.25, by * 0.25)
        gcode_text = generate_benchmark_gcode(
            build_center_xy=(bx * 0.5, by * 0.5),
            size_mm=cube_size,
            height_mm=15.0,
            print_speed_mm_s=min(220.0, spec.kinematics.max_velocity_mm_s * 0.65),
            infill_speed_mm_s=min(320.0, spec.kinematics.max_velocity_mm_s * 0.85),
            travel_speed_mm_s=spec.kinematics.max_velocity_mm_s,
        )

    vprint = simulate_gcode(gcode_text, assembly, modal_fea, kinematics, thermal)
    scores, diagnostics = _compute_scores_and_diagnostics(
        spec, assembly, collision, static_fea, modal_fea, kinematics, thermal, vprint
    )

    return SimulationReport(
        spec=spec,
        assembly=assembly,
        collision=collision,
        static_fea=static_fea,
        modal_fea=modal_fea,
        kinematics=kinematics,
        thermal=thermal,
        virtual_print=vprint,
        scores=scores,
        diagnostics=diagnostics,
    )
