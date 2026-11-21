"""Unified multiphysics simulation pipeline & structured scorecard for humans and AI agents."""

from __future__ import annotations

from dataclasses import dataclass, field
import time

from printsim.fea.bed_plate_fem import BedPlateFEAResult, solve_bed_plate_fem
from printsim.fea.components import ThermalStructuralReport, compute_thermal_structural_report
from printsim.fea.mechatronics import (
    DriveControlConfig,
    WorkspaceFRFMap,
    compute_workspace_frf_map,
)
from printsim.fea.modal import ModalResult, solve_modal_fea
from printsim.fea.rcsa import RCSAResult, analyze_toolhead_rcsa
from printsim.fea.sensitivity import SensitivityReport, compute_structural_sensitivities
from printsim.fea.solver import StaticFEAResult, solve_static_fea
from printsim.gcode.ballbar import TimeDomainMetrologyResult, simulate_time_domain_metrology
from printsim.gcode.parser import generate_benchmark_gcode
from printsim.gcode.simulator import VirtualPrintReport, simulate_gcode
from printsim.geometry.builder import AssemblyModel, build_assembly
from printsim.kinematics.collision import CollisionReport, check_frame_clearance
from printsim.kinematics.solvers import KinematicsReport, analyze_kinematics
from printsim.schema import GANTRY_BRACKET_CATALOG, PrinterSpec, get_material
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
    thermal_structural: ThermalStructuralReport
    diagnostics: list[str] = field(default_factory=list)
    workspace_frf: WorkspaceFRFMap | None = None
    time_domain_metrology: TimeDomainMetrologyResult | None = None
    rcsa: RCSAResult | None = None
    sensitivity: SensitivityReport | None = None
    bed_plate_fea: BedPlateFEAResult | None = None

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
            "thermal_structural": self.thermal_structural.to_dict(),
            "virtual_print": vp,
            "workspace_frf": self.workspace_frf.to_dict() if self.workspace_frf is not None else None,
            "time_domain_metrology": (
                self.time_domain_metrology.to_dict()
                if self.time_domain_metrology is not None
                else None
            ),
            "rcsa": self.rcsa.to_dict() if self.rcsa is not None else None,
            "sensitivity": self.sensitivity.to_dict() if self.sensitivity is not None else None,
            "bed_plate_fea": (
                self.bed_plate_fea.to_dict() if self.bed_plate_fea is not None else None
            ),
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
    thermal_structural: ThermalStructuralReport | None = None,
    workspace_frf: WorkspaceFRFMap | None = None,
    rcsa: RCSAResult | None = None,
    bed_plate_fea: BedPlateFEAResult | None = None,
) -> tuple[ScoreBreakdown, list[str]]:
    diag: list[str] = []

    # 1. Structural score (deflection < 15um ideal, high fundamental freq rewarded up to 200 Hz, SF >= 4)
    defl_pen = min(45.0, static_fea.combined_max_deflection_um * 0.65)
    f0 = modal_fea.fundamental_freq_hz
    freq_bonus = min(36.0, f0 * 0.60) + min(12.0, max(0.0, f0 - 60.0) / 12.0)
    sf_pen = 0.0 if static_fea.min_safety_factor >= 4.0 else (4.0 - static_fea.min_safety_factor) * 10.0
    structural = max(0.0, min(100.0, 52.0 + freq_bonus - defl_pen - sf_pen))

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
    if static_fea.nozzle_twist_amplification_pct > 45.0:
        diag.append(
            f"High nozzle-tip lever-arm twist amplification (+{static_fea.nozzle_twist_amplification_pct:.1f}% vs X-beam center, "
            f"{static_fea.nozzle_twist_angle_mrad:.2f} mrad): lower toolhead CoG, shorten nozzle drop, "
            f"or upgrade carriage mount from '{spec.toolhead.mount_material}' to 'pa12_cf_printed' or 'aluminum_6061_t6'."
        )
    if spec.toolhead.z_probe_type in ("printed_tap", "cnc_tap"):
        max_th_share = max(
            (float(m.subsystem_share_pct.get("toolhead", 0.0)) for m in modal_fea.modes),
            default=0.0,
        )
        max_flop = max((float(m.toolhead_flop_pct) for m in modal_fea.modes), default=0.0)
        if (
            max_th_share > 25.0
            or max_flop > 25.0
            or static_fea.nozzle_twist_amplification_pct > 25.0
            or (rcsa is not None and rcsa.whip_risk)
        ):
            diag.append(
                f"Voron Tap probe ('{spec.toolhead.z_probe_type}') adds moving mass and sliding-rail pitch compliance "
                f"(toolhead compliance {max_th_share:.1f}%, nozzle whip +{static_fea.nozzle_twist_amplification_pct:.1f}%): "
                "switching to a static eddy probe ('static_eddy', Beacon/Cartographer) raises X/Y resonance significantly."
            )
    if (
        spec.architecture == "corexy"
        and spec.frame.enclosed_panels
        and spec.frame.panel_mounting == "bolted_structural"
        and 0.0 < modal_fea.panel_drum_freq_hz < 65.0
    ):
        diag.append(
            f"Panel oil-canning / drum resonance risk ({modal_fea.panel_drum_freq_hz:.1f} Hz < 65 Hz): "
            f"{spec.frame.panel_thickness_mm:.1f} mm {spec.frame.panel_material} panels vibrate out-of-plane "
            "within print excitation bandwidth; step up to 4-5 mm Aluminum 5052 or >= 4 mm ACM Dibond."
        )
    if (
        structural < 80.0
        and spec.is_flying_gantry
        and spec.frame.enclosed_panels
        and spec.frame.panel_mounting == "bolted_structural"
        and spec.frame.bolt_panels_to_gantry
    ):
        diag.append(
            "Flying-gantry CoreXY (belted_quad_z): structural panels bolt only to the outer frame perimeter "
            "and cannot tie into the moving Z gantry beams; switch to a fixed-gantry Z drive "
            "(e.g., leadscrew_triple) if direct gantry-to-panel bolting is desired."
        )

    # 2. Dynamics score (effective resonance accel up to high-speed AWD range, motor torque margin, belt integrity)
    rec_a = modal_fea.recommended_max_accel_mm_s2
    accel_score = min(45.0, (rec_a / 12000.0) * 45.0) + min(
        10.0, max(0.0, rec_a - 12000.0) / 6000.0
    )
    torque_score = min(30.0, kinematics.torque_safety_margin * 15.0)
    belt_score = 0.0 if kinematics.belt_slip_risk else 15.0
    belt_overtension_pen = 8.0 if kinematics.belt_overtension_risk else 0.0
    shaft_bend_pen = 6.0 if kinematics.shaft_bending_risk else 0.0
    bracket_creep_pen = 5.0 if kinematics.bracket_creep_risk else 0.0
    dynamics = max(
        0.0,
        min(
            100.0,
            accel_score
            + torque_score
            + belt_score
            - belt_overtension_pen
            - shaft_bend_pen
            - bracket_creep_pen,
        ),
    )
    if kinematics.step_skip_risk:
        diag.append(
            f"Stepper torque margin is low ({kinematics.torque_safety_margin:.2f}x): "
            f"limit accel to <= {kinematics.max_safe_accel_motor_mm_s2:.0f} mm/s^2 or use stronger/48V steppers."
        )
    if kinematics.motor_overcurrent_risk:
        diag.append(
            f"Stepper overcurrent risk (run {kinematics.motor_run_current_a:.2f} A > rated "
            f"{spec.kinematics.xy_stepper.rated_current_a:.2f} A * 1.05, ~{kinematics.motor_heat_w_each:.1f} W/motor): "
            "risk of coil overheating or magnet demagnetization; reduce run_current_a or use higher-current motors."
        )
    if kinematics.belt_slip_risk:
        diag.append(
            f"Belt slack-side slip risk at {spec.kinematics.max_accel_mm_s2:.0f} mm/s^2: "
            "increase belt pretension or upgrade to 9mm/12mm GT2 belt."
        )
    if kinematics.belt_overtension_risk:
        diag.append(
            f"Belt overtension risk (peak {kinematics.belt_peak_tension_n:.1f} N > rated "
            f"{kinematics.belt_max_working_tension_n:.1f} N): upgrade to 9mm/12mm EPDM Fiberglass or Aramid (Kevlar) belts."
        )
    if kinematics.steel_core_fatigue_risk or (
        spec.gantry.belt.core == "steel" and spec.gantry.belt.pulley_teeth <= 24
    ):
        diag.append(
            f"Steel-core belt fatigue fracture risk: steel-core GT2 belts suffer rapid bend-fatigue cracking on "
            f"<=20T/{spec.gantry.belt.pulley_teeth}T pulleys — use Gates PowerGrip 2GT fiberglass or aramid (EPDM) instead."
        )
    if kinematics.bracket_creep_risk:
        bkt_info = GANTRY_BRACKET_CATALOG.get(
            spec.gantry.bracket_material, GANTRY_BRACKET_CATALOG["abs_printed"]
        )
        creep_limit_n = float(bkt_info["creep_tension_limit_n"])
        diag.append(
            f"Gantry bracket creep risk ({spec.gantry.belt.pretension_n:.0f} N belt pretension > "
            f"{creep_limit_n:.0f} N sustained limit for '{spec.gantry.bracket_material}'): "
            "upgrade gantry brackets to 'ppa_cf_printed', 'pps_cf_printed', or 'aluminum_6061_t6'."
        )
    if (
        spec.gantry.belt.pretension_n > 50.0
        and spec.toolhead.mount_material == "abs_printed"
        and not (kinematics.bracket_creep_risk and spec.gantry.bracket_material == "abs_printed")
    ):
        diag.append(
            f"High-tension printed mount creep warning ({spec.gantry.belt.pretension_n:.0f} N with ABS toolhead mount): "
            "unfilled ABS creeps under sustained high belt tension — upgrade to PA12-CF / PPA-CF or CNC Aluminum mounts."
        )
    if kinematics.shaft_bending_risk:
        diag.append(
            f"Stepper shaft bending risk at {spec.gantry.belt.pretension_n:.0f} N belt pretension (> 45 N) "
            "on cantilevered single-shear motor shafts: enable Double-Shear Pulley Supports."
        )
    if not kinematics.delta_workspace_valid:
        diag.append("Delta arm length cannot reach full build volume corners; increase delta_arm_length_mm.")
    if workspace_frf is not None:
        if workspace_frf.worst_corner_drop_pct > 12.0:
            diag.append(
                f"CoreXY corner belt-span asymmetry drops workspace frequency by "
                f"{workspace_frf.worst_corner_drop_pct:.1f}% (center {workspace_frf.center_freq_hz:.1f} Hz -> "
                f"corner {workspace_frf.min_freq_hz:.1f} Hz, max X-Y cross-coupling {workspace_frf.max_cross_coupling_pct:.1f}%): "
                "upgrade to 4-motor AWD or use a wide-notch 'ei' / 'smooth_ei' input shaper."
            )
        if workspace_frf.midband_instability_risk:
            diag.append(
                "Open-loop stepper mid-band resonance risk (negative electrical L/R damping at speed): "
                "upgrade stepper supply to 48V, switch to lower-inductance motors, or enable closed-loop stepper / FOC servo drive."
            )
    if rcsa is not None and rcsa.whip_risk and rcsa.warning_message:
        diag.append(rcsa.warning_message)

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
    if spec.gantry.x_beam_material == "carbon_fiber_tube" and spec.frame.chamber_temp_c >= 65.0:
        diag.append(
            f"Standard epoxy carbon fiber X-beam softening/creep risk at {spec.frame.chamber_temp_c:.0f} C chamber "
            "(>= 65 C): standard roll-wrapped CF tubes can soften and creep in heated enclosures — switch to "
            "'carbon_fiber_high_tg' (Tg 125 C) or 'cnc_al_beam_20x20'."
        )
    if thermal_structural is not None:
        if thermal_structural.gantry_bimetallic_bow_um > 40.0:
            diag.append(
                f"Bimetallic X-gantry thermal bow ({thermal_structural.gantry_bimetallic_bow_um:.1f} um at "
                f"{thermal_structural.chamber_temp_c:.0f} C chamber): add a 'titanium_grade5' or 'steel_4140' "
                "rail backer or switch to 'dual_top_bottom' X rails."
            )
        if (
            thermal_structural.bed_mount_type == "rigid_4pt"
            and thermal_structural.bed_buckling_taco_um > 30.0
        ):
            diag.append(
                f"Rigid 4-point bed thermal taco buckling ({thermal_structural.bed_buckling_taco_um:.1f} um at "
                f"{thermal_structural.bed_target_temp_c:.0f} C): switch bed.mount_type to a 3-point Maxwell "
                "'kinematic_3pt' mount."
            )
        if thermal_structural.bracket_tg_warning:
            diag.append(
                f"Polymer bracket thermal softening near Tg ({spec.gantry.bracket_material} retains "
                f"{thermal_structural.bracket_modulus_retention_pct:.1f}% modulus at "
                f"{thermal_structural.chamber_temp_c:.0f} C chamber): upgrade to 'ppa_cf_printed', "
                "'pps_cf_printed', or CNC Aluminum ('aluminum_6061_t6')."
            )
        elif spec.toolhead.mount_material != spec.gantry.bracket_material:
            th_mat = get_material(spec.toolhead.mount_material)
            th_ret = th_mat.modulus_retention_at_temp(thermal_structural.chamber_temp_c)
            if th_mat.tg_c is not None and (
                thermal_structural.chamber_temp_c >= th_mat.tg_c - 5.0 or th_ret < 0.80
            ):
                diag.append(
                    f"Toolhead mount polymer thermal softening near Tg ({spec.toolhead.mount_material} retains "
                    f"{th_ret * 100.0:.1f}% modulus at {thermal_structural.chamber_temp_c:.0f} C): "
                    "upgrade toolhead.mount_material to 'ppa_cf_printed', 'pps_cf_printed', or CNC Aluminum."
                )

    if bed_plate_fea is not None:
        if bed_plate_fea.total_flatness_error_um >= 120.0:
            if spec.architecture == "corexy" and spec.bed.z_drive == "leadscrew_single":
                diag.append(
                    f"Bed plate flatness error is {bed_plate_fea.total_flatness_error_um:.1f} um "
                    f"({bed_plate_fea.first_layer_risk} first-layer risk, gravity sag {bed_plate_fea.gravity_sag_um:.1f} um): "
                    "single rear-cantilever Z leadscrew droops at the front edge; upgrade to 'leadscrew_triple' or 'belted_quad_z'."
                )
            elif bed_plate_fea.taco_buckling_um >= 30.0:
                diag.append(
                    f"Bed plate flatness error is {bed_plate_fea.total_flatness_error_um:.1f} um "
                    f"({bed_plate_fea.first_layer_risk} first-layer risk): over-constrained thermal bow "
                    f"({bed_plate_fea.taco_buckling_um:.1f} um); switch to 'kinematic_3pt' or 'floating_4pt' mounts."
                )
            else:
                diag.append(
                    f"Bed plate flatness error is {bed_plate_fea.total_flatness_error_um:.1f} um "
                    f"({bed_plate_fea.first_layer_risk} first-layer risk): step up bed plate thickness "
                    "to 6-8 mm MIC-6 cast aluminum or enable mesh bed leveling."
                )
        if bed_plate_fea.plate_first_mode_hz < 60.0:
            if bed_plate_fea.plate_first_mode_type == "cantilever_flap":
                bed_mode_fix = "upgrade Z-drive from 'leadscrew_single' to 'leadscrew_triple' or 'belted_quad_z'"
            elif bed_plate_fea.plate_first_mode_type == "mount_bounce":
                bed_mode_fix = "replace soft bed springs/silicone spacers with 'kinematic_3pt' or 'floating_4pt' mounts"
            else:
                bed_mode_fix = "increase bed plate thickness (6-8 mm MIC-6) or add multi-point bed support"
            diag.append(
                f"Low bed plate natural frequency ({bed_plate_fea.plate_first_mode_hz:.1f} Hz, "
                f"{bed_plate_fea.plate_first_mode_type}): {bed_mode_fix}."
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

    # 5. Packaging / moving-mass / cost efficiency
    pkg_score = min(45.0, kinematics.packaging_efficiency_pct * 1.4)
    moving_mass_score = max(0.0, min(20.0, 20.0 * (1.0 - max(0.0, assembly.y_moving_mass_kg - 0.35) / 1.5)))
    frame_mass_score = max(0.0, min(10.0, 10.0 * (1.0 - max(0.0, assembly.total_mass_kg - 15.0) / 60.0)))
    mass_score = moving_mass_score + frame_mass_score
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
    panel_mesh_n: int = 2,
    include_advanced_analysis: bool = True,
    accel_override_mm_s2: float | None = None,
) -> SimulationReport:
    """Run the full multiphysics simulation suite on a PrinterSpec."""
    assembly = build_assembly(
        spec,
        toolhead_rel_pos=toolhead_rel_pos,
        panel_mesh_n=panel_mesh_n,
    )
    collision = check_frame_clearance(assembly)
    static_fea = solve_static_fea(assembly, accel_mm_s2=accel_override_mm_s2)
    modal_fea = solve_modal_fea(assembly)
    kinematics = analyze_kinematics(assembly)
    thermal = simulate_hotend_thermal(spec, filament_id=filament_id)
    thermal_structural = compute_thermal_structural_report(assembly)

    workspace_frf: WorkspaceFRFMap | None = None
    time_domain_metrology: TimeDomainMetrologyResult | None = None
    rcsa: RCSAResult | None = None
    sensitivity: SensitivityReport | None = None
    bed_plate_fea: BedPlateFEAResult | None = None
    if include_advanced_analysis:
        drive_cfg = DriveControlConfig(
            drive_type=spec.kinematics.drive_type,
            servo_vel_bw_hz=spec.kinematics.servo_vel_bw_hz,
            servo_pos_bw_hz=spec.kinematics.servo_pos_bw_hz,
            notch_filter_hz=spec.kinematics.notch_filter_hz,
        )
        workspace_frf = compute_workspace_frf_map(
            assembly, modal_fea, grid_n=5, drive_cfg=drive_cfg
        )
        time_domain_metrology = simulate_time_domain_metrology(
            assembly, modal_fea, drive_cfg=drive_cfg
        )
        rcsa = analyze_toolhead_rcsa(assembly, modal_fea)
        sensitivity = compute_structural_sensitivities(assembly, static_fea, modal_fea)
        bed_plate_fea = solve_bed_plate_fem(spec)

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
        spec,
        assembly,
        collision,
        static_fea,
        modal_fea,
        kinematics,
        thermal,
        vprint,
        thermal_structural=thermal_structural,
        workspace_frf=workspace_frf,
        rcsa=rcsa,
        bed_plate_fea=bed_plate_fea,
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
        thermal_structural=thermal_structural,
        diagnostics=diagnostics,
        workspace_frf=workspace_frf,
        time_domain_metrology=time_domain_metrology,
        rcsa=rcsa,
        sensitivity=sensitivity,
        bed_plate_fea=bed_plate_fea,
    )


def _extract_verify_summary(report: SimulationReport, solve_time_ms: float) -> dict:
    return {
        "fundamental_freq_hz": round(float(report.modal_fea.fundamental_freq_hz), 2),
        "recommended_max_accel_mm_s2": round(float(report.modal_fea.recommended_max_accel_mm_s2), 0),
        "combined_max_deflection_um": round(float(report.static_fea.combined_max_deflection_um), 2),
        "panel_drum_freq_hz": round(float(report.modal_fea.panel_drum_freq_hz), 2),
        "panel_max_von_mises_mpa": round(float(report.static_fea.panel_max_von_mises_mpa), 3),
        "num_shell_elements": len(report.assembly.shell_elements),
        "solve_time_ms": round(float(solve_time_ms), 1),
    }


def verify_printer_deep(
    spec: PrinterSpec,
    gcode_text: str | None = None,
    toolhead_rel_pos: tuple[float, float, float] = (0.5, 0.5, 0.5),
    filament_id: str = "pla",
    deep_mesh_n: int = 6,
) -> dict:
    """Compare fast 2x2 shell FEA vs deep nxn mesh verification and report convergence."""
    fast_n = 2
    deep_n = max(4, int(deep_mesh_n))

    t0 = time.perf_counter()
    fast_report = evaluate_printer(
        spec,
        gcode_text=gcode_text,
        toolhead_rel_pos=toolhead_rel_pos,
        filament_id=filament_id,
        panel_mesh_n=fast_n,
    )
    fast_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    deep_report = evaluate_printer(
        spec,
        gcode_text=gcode_text,
        toolhead_rel_pos=toolhead_rel_pos,
        filament_id=filament_id,
        panel_mesh_n=deep_n,
    )
    deep_ms = (time.perf_counter() - t1) * 1000.0

    fast_summary = _extract_verify_summary(fast_report, fast_ms)
    deep_summary = _extract_verify_summary(deep_report, deep_ms)

    f_fast = fast_summary["fundamental_freq_hz"]
    f_deep = deep_summary["fundamental_freq_hz"]
    d_fast = fast_summary["combined_max_deflection_um"]
    d_deep = deep_summary["combined_max_deflection_um"]
    a_fast = fast_summary["recommended_max_accel_mm_s2"]
    a_deep = deep_summary["recommended_max_accel_mm_s2"]

    freq_diff_pct = round(abs(f_deep - f_fast) / max(f_deep, 1e-6) * 100.0, 2)
    deflection_diff_pct = round(abs(d_deep - d_fast) / max(d_deep, 1e-6) * 100.0, 2)
    accel_diff_pct = round(abs(a_deep - a_fast) / max(a_deep, 1e-6) * 100.0, 2)
    converged = bool(freq_diff_pct <= 15.0 and deflection_diff_pct <= 15.0)

    return {
        "fast_mesh_n": fast_n,
        "deep_mesh_n": deep_n,
        "fast_summary": fast_summary,
        "deep_summary": deep_summary,
        "convergence": {
            "freq_diff_pct": freq_diff_pct,
            "deflection_diff_pct": deflection_diff_pct,
            "accel_diff_pct": accel_diff_pct,
            "converged": converged,
        },
        "deep_simulation": deep_report.to_dict(include_toolpath_segments=False),
    }
