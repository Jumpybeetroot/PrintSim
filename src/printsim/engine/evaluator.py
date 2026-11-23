"""Unified multiphysics simulation pipeline & structured scorecard for humans and AI agents."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any

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
    weight_shaper_gains: dict[str, Any] | None = None

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
            "weight_shaper_gains": self.weight_shaper_gains,
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
            "main limit is outer frame racking and bending stiffness relative to moving/frame mass."
        )
    if static_fea.combined_max_deflection_um > 40.0:
        diag.append(
            f"High peak deflection ({static_fea.combined_max_deflection_um:.1f} um) under acceleration: "
            "main limit is gantry/toolhead structural stiffness relative to moving inertial load."
        )
    if static_fea.nozzle_twist_amplification_pct > 45.0:
        diag.append(
            f"High nozzle-tip lever-arm twist amplification (+{static_fea.nozzle_twist_amplification_pct:.1f}% vs X-beam center, "
            f"{static_fea.nozzle_twist_angle_mrad:.2f} mrad): main limit is toolhead mount compliance "
            f"('{spec.toolhead.mount_material}') and CoG/nozzle lever arm overhang."
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
                f"Voron Tap probe ('{spec.toolhead.z_probe_type}') adds moving carriage mass and sliding-rail pitch compliance "
                f"(toolhead compliance {max_th_share:.1f}%, nozzle whip +{static_fea.nozzle_twist_amplification_pct:.1f}%): "
                "main limit is Z-probe carriage pitch compliance and overhanging mass."
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
            "within print excitation bandwidth (main limit: panel flexural rigidity)."
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
            "and cannot couple into the moving Z gantry beams (main limit: unbraced gantry Y-rail load path)."
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
            f"motor pull-out torque headroom limits acceleration to <= {kinematics.max_safe_accel_motor_mm_s2:.0f} mm/s^2 at target speed/voltage."
        )
    if kinematics.motor_overcurrent_risk:
        diag.append(
            f"Stepper overcurrent risk (run {kinematics.motor_run_current_a:.2f} A > rated "
            f"{spec.kinematics.xy_stepper.rated_current_a:.2f} A * 1.05, ~{kinematics.motor_heat_w_each:.1f} W/motor): "
            "main limit is continuous coil thermal dissipation and magnet demagnetization margin."
        )
    if kinematics.belt_slip_risk:
        diag.append(
            f"Belt slack-side slip risk at {spec.kinematics.max_accel_mm_s2:.0f} mm/s^2: "
            f"main limit is belt pretension ({spec.gantry.belt.pretension_n:.1f} N) relative to peak dynamic drive tension."
        )
    if kinematics.belt_overtension_risk:
        diag.append(
            f"Belt overtension risk (peak {kinematics.belt_peak_tension_n:.1f} N > rated "
            f"{kinematics.belt_max_working_tension_n:.1f} N): peak dynamic load exceeds the "
            f"{spec.gantry.belt.type} ({spec.gantry.belt.core}) working tension limit."
        )
    if kinematics.steel_core_fatigue_risk or (
        spec.gantry.belt.type.startswith("gt2_")
        and spec.gantry.belt.core == "steel"
        and spec.gantry.belt.pulley_teeth <= 24
    ):
        diag.append(
            f"Steel-core belt fatigue fracture risk ({kinematics.belt_label}): steel cords experience severe "
            f"reverse-bend fatigue on {spec.gantry.belt.pulley_teeth}T pulleys / "
            f"{kinematics.belt_back_idler_mm:.1f} mm back-idlers "
            f"(min {kinematics.belt_min_pulley_teeth}T, back-idler >= {kinematics.belt_min_back_idler_mm:.0f} mm)."
        )
    elif kinematics.belt_bend_fatigue_risk:
        diag.append(
            f"Belt minimum bend-radius risk ({kinematics.belt_label}): "
            f"{spec.gantry.belt.pulley_teeth}T pulley / {kinematics.belt_back_idler_mm:.1f} mm back-idler "
            f"is below datasheet limit ({kinematics.belt_min_pulley_teeth}T, "
            f"back-idler >= {kinematics.belt_min_back_idler_mm:.0f} mm)."
        )
    if kinematics.belt_creep_risk:
        if spec.gantry.belt.core == "aramid":
            diag.append(
                f"Aramid belt cord viscoelastic relaxation risk ({spec.gantry.belt.pretension_n:.0f} N sustained preload): "
                "aramid/Kevlar cords exhibit tension relaxation and reverse-bend kink fatigue under high preload."
            )
        else:
            diag.append(
                f"TPU belt matrix thermal softening & relaxation risk ({kinematics.belt_label} at "
                f"{spec.frame.chamber_temp_c:.0f} C chamber, {kinematics.belt_thermal_retention_pct:.0f}% "
                "tooth stiffness retention): 92 ShA polyurethane softens above 55 C causing tooth/clamp cold-flow."
            )
    if kinematics.belt_validation_tier in ("theoretical", "experimental"):
        tier_label = (
            "Theoretical / unproven on 3D printers"
            if kinematics.belt_validation_tier == "theoretical"
            else "Experimental custom-hardware belt"
        )
        lead_issue = (
            f" Key real-world caveat: {kinematics.belt_practical_issues[0]}"
            if kinematics.belt_practical_issues
            else ""
        )
        diag.append(
            f"{tier_label} ({kinematics.belt_label}): {kinematics.belt_theoretical_summary}{lead_issue}"
        )
    if kinematics.bracket_creep_risk:
        bkt_info = GANTRY_BRACKET_CATALOG.get(
            spec.gantry.bracket_material, GANTRY_BRACKET_CATALOG["abs_printed"]
        )
        creep_limit_n = float(bkt_info["creep_tension_limit_n"])
        diag.append(
            f"Gantry bracket creep risk ({spec.gantry.belt.pretension_n:.0f} N belt pretension > "
            f"{creep_limit_n:.0f} N sustained limit for '{spec.gantry.bracket_material}'): "
            "main limit is sustained viscoelastic creep strength of the bracket material."
        )
    if (
        spec.gantry.belt.pretension_n > 50.0
        and spec.toolhead.mount_material == "abs_printed"
        and not (kinematics.bracket_creep_risk and spec.gantry.bracket_material == "abs_printed")
    ):
        diag.append(
            f"High-tension printed mount creep warning ({spec.gantry.belt.pretension_n:.0f} N with "
            f"'{spec.toolhead.mount_material}' toolhead mount): sustained belt preload exceeds long-term "
            "viscoelastic creep margin of unfilled polymer."
        )
    if kinematics.shaft_bending_risk:
        diag.append(
            f"Stepper shaft bending risk at {spec.gantry.belt.pretension_n:.0f} N belt pretension (> 45 N): "
            "main limit is cantilevered single-shear motor shaft bending without double-shear pulley support."
        )
    if not kinematics.delta_workspace_valid:
        diag.append(
            f"Delta arm workspace geometry limit: arm length ({spec.kinematics.delta_arm_length_mm:.0f} mm) "
            "cannot reach full build volume perimeter within rod-angle limits."
        )
    if workspace_frf is not None:
        if workspace_frf.worst_corner_drop_pct > 12.0:
            diag.append(
                f"CoreXY corner belt-span asymmetry drops workspace frequency by "
                f"{workspace_frf.worst_corner_drop_pct:.1f}% (center {workspace_frf.center_freq_hz:.1f} Hz -> "
                f"corner {workspace_frf.min_freq_hz:.1f} Hz, max X-Y cross-coupling {workspace_frf.max_cross_coupling_pct:.1f}%): "
                "main limit is asymmetric free belt-span stiffness across the XY workspace."
            )
        if workspace_frf.midband_instability_risk:
            diag.append(
                "Open-loop stepper mid-band resonance risk (negative electrical L/R damping at speed): "
                "main limit is winding inductance-to-voltage time constant limiting high-speed phase margin."
            )
    if rcsa is not None and rcsa.whip_risk:
        diag.append(
            f"Nozzle-to-sensor whip risk ({rcsa.nozzle_to_sensor_amp_ratio:.2f}x > 1.35x at "
            f"{rcsa.nozzle_tip_peak_freq_hz:.1f} Hz): main limit is toolhead mount bending stiffness "
            f"('{spec.toolhead.mount_material}') and CoG/nozzle lever arm."
        )

    # 3. Thermal score (volumetric flow, heatup speed, heat creep safety)
    flow_score = min(55.0, (thermal.max_volumetric_flow_mm3_s / 35.0) * 55.0)
    heatup_score = max(0.0, min(25.0, 25.0 * (1.0 - max(0.0, thermal.heatup_time_215c_s - 25.0) / 140.0)))
    creep_score = 0.0 if thermal.heat_creep_risk else 20.0
    thermal_score = max(0.0, min(100.0, flow_score + heatup_score + creep_score))
    if thermal.heat_creep_risk:
        diag.append(
            f"Heat creep risk (cold-side throat {thermal.cold_side_temp_c:.1f} C > 52 C): "
            "main limit is heatbreak axial thermal conductance relative to heatsink convective cooling capacity."
        )
    if thermal.max_volumetric_flow_mm3_s < 18.0:
        diag.append(
            f"Hotend volumetric flow capped at {thermal.max_volumetric_flow_mm3_s:.1f} mm^3/s: "
            f"primary thermal/extrusion bottleneck is {thermal.flow_limiting_factor}."
        )
    if spec.gantry.x_beam_material == "carbon_fiber_tube" and spec.frame.chamber_temp_c >= 65.0:
        diag.append(
            f"Standard epoxy carbon fiber X-beam softening/creep risk at {spec.frame.chamber_temp_c:.0f} C chamber "
            "(>= 65 C): main limit is epoxy matrix glass-transition temperature (Tg), reducing flexural modulus and creep resistance."
        )
    if thermal_structural is not None:
        if thermal_structural.gantry_bimetallic_bow_um > 40.0:
            diag.append(
                f"Bimetallic X-gantry thermal bow ({thermal_structural.gantry_bimetallic_bow_um:.1f} um at "
                f"{thermal_structural.chamber_temp_c:.0f} C chamber): main limit is CTE mismatch between the "
                "X-gantry beam and steel linear rail causing asymmetric thermal bending."
            )
        if (
            thermal_structural.bed_mount_type == "rigid_4pt"
            and thermal_structural.bed_buckling_taco_um > 30.0
        ):
            diag.append(
                f"Rigid 4-point bed thermal taco buckling ({thermal_structural.bed_buckling_taco_um:.1f} um at "
                f"{thermal_structural.bed_target_temp_c:.0f} C): main limit is over-constrained lateral mounting "
                "blocking radial thermal expansion."
            )
        if thermal_structural.bracket_tg_warning:
            diag.append(
                f"Polymer bracket thermal softening near Tg ({spec.gantry.bracket_material} retains "
                f"{thermal_structural.bracket_modulus_retention_pct:.1f}% modulus at "
                f"{thermal_structural.chamber_temp_c:.0f} C chamber): main limit is bracket polymer Tg stiffness retention."
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
                    "main limit is mount polymer Tg stiffness retention."
                )

    if bed_plate_fea is not None:
        if bed_plate_fea.total_flatness_error_um >= 120.0:
            if spec.architecture == "corexy" and spec.bed.z_drive == "leadscrew_single":
                diag.append(
                    f"Bed plate flatness error is {bed_plate_fea.total_flatness_error_um:.1f} um "
                    f"({bed_plate_fea.first_layer_risk} first-layer risk, gravity sag {bed_plate_fea.gravity_sag_um:.1f} um): "
                    "main limit is cantilevered single-Z bed support compliance causing front-edge droop."
                )
            elif bed_plate_fea.taco_buckling_um >= 30.0:
                diag.append(
                    f"Bed plate flatness error is {bed_plate_fea.total_flatness_error_um:.1f} um "
                    f"({bed_plate_fea.first_layer_risk} first-layer risk): main limit is over-constrained "
                    f"thermal bow ({bed_plate_fea.taco_buckling_um:.1f} um) from rigid lateral mounting."
                )
            else:
                diag.append(
                    f"Bed plate flatness error is {bed_plate_fea.total_flatness_error_um:.1f} um "
                    f"({bed_plate_fea.first_layer_risk} first-layer risk): main limit is bed plate flexural "
                    "rigidity against gravity sag and thermal gradient bow."
                )
        if bed_plate_fea.plate_first_mode_hz < 60.0:
            if bed_plate_fea.plate_first_mode_type == "cantilever_flap":
                bed_mode_limit = "cantilevered Z-support pitch compliance"
            elif bed_plate_fea.plate_first_mode_type == "mount_bounce":
                bed_mode_limit = "compliant bed-mount vertical spring stiffness"
            else:
                bed_mode_limit = "bed plate flexural stiffness relative to bed mass and support span"
            diag.append(
                f"Low bed plate natural frequency ({bed_plate_fea.plate_first_mode_hz:.1f} Hz, "
                f"{bed_plate_fea.plate_first_mode_type}): main limit is {bed_mode_limit}."
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


def _extract_struct_mass_slope(
    sensitivity: SensitivityReport | None,
    node_ids: tuple[str, ...] | list[str],
    member_ids: tuple[str, ...] | list[str],
    ms_key: str,
) -> float:
    """Return positive structural frequency slope ``df / d(-100 g)`` in ``Hz / -100g``."""
    if sensitivity is None:
        return 0.0
    lumped = getattr(sensitivity, "lumped_mass_sensitivities", None) or {}
    members = getattr(sensitivity, "member_sensitivities", None) or {}
    ms_tot = 0.0
    mass_tot = 0.0
    fallback_hz_weighted = 0.0
    has_ms = False
    for coll, ids in ((lumped, node_ids), (members, member_ids)):
        for key in ids:
            d = coll.get(key)
            if not isinstance(d, dict):
                continue
            mk = float(d.get("mass_kg", 0.0) or 0.0)
            if mk <= 1e-8:
                continue
            mass_tot += mk
            if ms_key in d or "mass_sens_hz" in d:
                has_ms = True
                ms_tot += float(d.get(ms_key, d.get("mass_sens_hz", 0.0)) or 0.0)
            else:
                fallback_hz_weighted += float(d.get("hz_per_100g", 0.0) or 0.0) * mk
    if mass_tot <= 1e-8:
        return 0.0
    if has_ms:
        return max(0.0, -0.1 * ms_tot / mass_tot)
    return max(0.0, -fallback_hz_weighted / mass_tot)


def compute_weight_shaper_gains(
    assembly: AssemblyModel,
    modal_fea: ModalResult,
    sensitivity: SensitivityReport | None = None,
) -> dict[str, Any]:
    """Compute coupled drive + structural input-shaper gains from component weight cuts."""
    sx = modal_fea.shaper_x
    sy = modal_fea.shaper_y
    if sx is None or sy is None:
        limiting_axis = "Y"
        lim = sy if sy is not None else sx
    elif sx.max_accel_mm_s2 < sy.max_accel_mm_s2:
        limiting_axis = "X"
        lim = sx
    elif sy.max_accel_mm_s2 < sx.max_accel_mm_s2:
        limiting_axis = "Y"
        lim = sy
    elif sx.effective_freq_hz < sy.effective_freq_hz:
        limiting_axis = "X"
        lim = sx
    else:
        limiting_axis = "Y"
        lim = sy

    if lim is not None:
        f_eff = float(lim.effective_freq_hz)
        f_struct = float(lim.structural_freq_hz)
        f_drive = float(lim.drive_freq_hz)
        a_max = float(lim.max_accel_mm_s2)
    else:
        f_eff = max(float(modal_fea.fundamental_freq_hz), 5.0)
        f_struct = f_eff
        f_drive = f_eff
        a_max = max(float(modal_fea.recommended_max_accel_mm_s2), 0.0)

    m_lim_g = float(
        assembly.x_moving_mass_kg if limiting_axis == "X" else assembly.y_moving_mass_kg
    ) * 1000.0
    ms_key = "mass_sens_x_hz" if limiting_axis == "X" else "mass_sens_y_hz"

    def _coupled_gain(
        cut_g: float,
        moves_on_limiting: bool,
        struct_hz_per_100g: float,
    ) -> tuple[float, float]:
        if m_lim_g <= 1.0 or f_eff <= 0.0:
            return 0.0, 0.0
        cut_safe = max(0.0, min(float(cut_g), 0.75 * m_lim_g))
        if cut_safe <= 0.0:
            return 0.0, 0.0
        f_drive_new = (
            f_drive * math.sqrt(m_lim_g / max(m_lim_g - cut_safe, 1.0))
            if moves_on_limiting
            else f_drive
        )
        struct_cap = (
            f_struct * math.sqrt(m_lim_g / max(m_lim_g - cut_safe, 1.0))
            if moves_on_limiting
            else f_struct * 1.35
        )
        f_struct_new = min(
            f_struct + max(0.0, float(struct_hz_per_100g)) * (cut_safe / 100.0),
            struct_cap,
        )
        f_eff_new = 1.0 / math.sqrt(
            1.0 / (max(f_struct_new, 5.0) ** 2) + 1.0 / (max(f_drive_new, 5.0) ** 2)
        )
        df_hz = max(0.0, f_eff_new - f_eff)
        da = max(0.0, a_max * ((f_eff_new / max(f_eff, 1.0)) ** 2 - 1.0))
        return df_hz, da

    spec = assembly.spec
    arch = spec.architecture
    nmap = assembly.node_map
    gantry_members = [m for m in assembly.members if m.role == "gantry"]

    hotend_mass_g = float(spec.toolhead.hotend.mass_g)
    toolhead_mass_g = float(assembly.x_moving_mass_kg) * 1000.0
    if toolhead_mass_g <= 0.0:
        toolhead_mass_g = float(spec.toolhead.total_mass_g) + hotend_mass_g

    hotend_slope = _extract_struct_mass_slope(sensitivity, ("nozzle_tip",), (), ms_key)
    toolhead_slope = _extract_struct_mass_slope(
        sensitivity, ("toolhead_cog", "toolhead", "nozzle_tip"), (), ms_key
    )

    x_beam_mass_g = float(getattr(assembly, "x_beam_mass_kg", 0.0) or 0.0) * 1000.0
    if x_beam_mass_g <= 0.0:
        x_beam_mass_g = (
            sum(
                max(m.mass_kg(nmap) - m.rail_mass_kg - m.backer_mass_kg - m.fill_mass_kg, 0.0)
                for m in gantry_members
                if m.start_id in nmap and m.end_id in nmap
            )
            * 1000.0
        )

    x_rail_mass_g = float(getattr(assembly, "x_rail_mass_kg", 0.0) or 0.0) * 1000.0
    x_backer_mass_g = float(getattr(assembly, "x_backer_mass_kg", 0.0) or 0.0) * 1000.0
    x_rail_backer_g = x_rail_mass_g + x_backer_mass_g
    if x_rail_backer_g <= 0.0:
        x_rail_backer_g = (
            sum(m.rail_mass_kg + m.backer_mass_kg for m in gantry_members) * 1000.0
        )

    y_carriages_g = float(getattr(assembly, "y_carriage_joints_mass_kg", 0.0) or 0.0) * 1000.0
    if y_carriages_g <= 0.0:
        y_carriages_g = float(spec.gantry.carriage_mass_g)

    member_sens = (getattr(sensitivity, "member_sensitivities", None) or {}) if sensitivity else {}
    x_gantry_mids = ("x_gantry_l", "x_gantry_r")
    x_gantry_slope = _extract_struct_mass_slope(sensitivity, (), x_gantry_mids, ms_key)

    raw_items: list[tuple[str, str, float, float, str, bool, float, str]] = []
    if arch == "cartesian_bedslinger":
        raw_items.append(
            (
                "hotend",
                "Hotend & Nozzle",
                hotend_mass_g,
                round(min(25.0, 0.35 * hotend_mass_g), 1),
                "X",
                limiting_axis == "X",
                hotend_slope,
                "Bare hotend, heatbreak, and heatsink assembly mass; translates on the X carriage and loads the nozzle cantilever.",
            )
        )
        raw_items.append(
            (
                "toolhead",
                "Toolhead Total",
                toolhead_mass_g,
                round(min(100.0, 0.30 * toolhead_mass_g), 1),
                "X",
                limiting_axis == "X",
                toolhead_slope,
                "Total X-moving toolhead, carriage block, extruder, and probe mass; accelerates along X only and is stationary during Y bed moves.",
            )
        )
        moving_bed_g = float(assembly.y_moving_mass_kg) * 1000.0
        if moving_bed_g <= 0.0:
            moving_bed_g = float(spec.bed.mass_g)
        bed_slope = _extract_struct_mass_slope(sensitivity, ("bed_surface",), (), ms_key)
        raw_items.append(
            (
                "moving_bed",
                "Moving Y-Bed Assembly",
                moving_bed_g,
                round(0.25 * moving_bed_g, 1),
                "Y",
                limiting_axis == "Y",
                bed_slope,
                "Total Y-moving heated bed plate, magnetic sheet, heater, and Y-carriage mass; carries full Y-axis inertial and belt-drive load.",
            )
        )
        if x_rail_mass_g > 0.0:
            bs_x_beam_g = x_beam_mass_g + x_rail_mass_g
        else:
            bs_x_beam_g = (
                sum(
                    max(m.mass_kg(nmap) - m.backer_mass_kg - m.fill_mass_kg, 0.0)
                    for m in gantry_members
                    if m.start_id in nmap and m.end_id in nmap
                )
                * 1000.0
            )
        raw_items.append(
            (
                "x_beam",
                "X-Beam & Rail",
                bs_x_beam_g,
                round(min(100.0, 0.35 * bs_x_beam_g), 1),
                "X",
                False,
                x_gantry_slope,
                "X-axis portal crossbeam and linear rail mass; stationary during X and Y print moves while contributing to X/Z portal structural modes.",
            )
        )
        total_struct_slope = bed_slope if limiting_axis == "Y" else toolhead_slope
    elif arch == "delta":
        arm_mids = [mid for mid in member_sens if mid.startswith("arm_")]
        delta_car_nodes = ("c_0", "c_1", "c_2")
        delta_car_slope = _extract_struct_mass_slope(sensitivity, delta_car_nodes, arm_mids, ms_key)
        delta_total_slope = _extract_struct_mass_slope(
            sensitivity,
            ("nozzle_tip", "toolhead_cog", "toolhead", "c_0", "c_1", "c_2"),
            arm_mids,
            ms_key,
        )
        y_gantry_total_g = float(assembly.y_moving_mass_kg) * 1000.0
        if y_gantry_total_g <= 0.0:
            y_gantry_total_g = toolhead_mass_g
        raw_items.extend(
            [
                (
                    "hotend",
                    "Hotend & Nozzle",
                    hotend_mass_g,
                    round(min(25.0, 0.35 * hotend_mass_g), 1),
                    "XY",
                    True,
                    hotend_slope,
                    "Bare hotend, heatbreak, and heatsink assembly mass; moves with the effector in X, Y, and Z at the nozzle cantilever tip.",
                ),
                (
                    "toolhead",
                    "Effector & Toolhead Total",
                    toolhead_mass_g,
                    round(min(100.0, 0.30 * toolhead_mass_g), 1),
                    "XY",
                    True,
                    toolhead_slope,
                    "Total moving effector, toolhead, probe, and bracket mass; accelerates in all XY directions across the parallel delta linkage.",
                ),
                (
                    "y_carriages",
                    "Arms & Carriages",
                    y_carriages_g,
                    round(min(80.0, 0.35 * y_carriages_g), 1),
                    "XY",
                    True,
                    delta_car_slope,
                    "Tower linear carriages and delta rod linkage mass coupled into 3-axis effector acceleration.",
                ),
                (
                    "y_gantry_total",
                    "Total Moving Effector + Arms",
                    y_gantry_total_g,
                    round(0.25 * y_gantry_total_g, 1),
                    "XY",
                    True,
                    delta_total_slope,
                    "Total equivalent moving effector and delta arm inertia reflected to the XY/Z parallel drive.",
                ),
            ]
        )
        total_struct_slope = delta_total_slope
    else:
        y_car_slope = _extract_struct_mass_slope(sensitivity, ("y_car_l", "y_car_r"), (), ms_key)
        y_gantry_total_g = float(assembly.y_moving_mass_kg) * 1000.0
        if y_gantry_total_g <= 0.0:
            y_gantry_total_g = toolhead_mass_g + x_beam_mass_g + x_rail_backer_g + y_carriages_g
        y_gantry_slope = _extract_struct_mass_slope(
            sensitivity,
            ("nozzle_tip", "toolhead_cog", "toolhead", "y_car_l", "y_car_r"),
            x_gantry_mids,
            ms_key,
        )
        raw_items.extend(
            [
                (
                    "hotend",
                    "Hotend & Nozzle",
                    hotend_mass_g,
                    round(min(25.0, 0.35 * hotend_mass_g), 1),
                    "XY",
                    True,
                    hotend_slope,
                    "Bare hotend, heatbreak, and heatsink assembly mass; moves on both X and Y axes and sits at the nozzle cantilever tip.",
                ),
                (
                    "toolhead",
                    "Toolhead Total",
                    toolhead_mass_g,
                    round(min(100.0, 0.30 * toolhead_mass_g), 1),
                    "XY",
                    True,
                    toolhead_slope,
                    "Total X-moving toolhead, carriage block, extruder, probe, and mount mass; accelerates on every X and Y print move.",
                ),
                (
                    "x_beam",
                    "X-Beam (Bare)",
                    x_beam_mass_g,
                    round(min(100.0, 0.40 * x_beam_mass_g), 1),
                    "Y",
                    limiting_axis == "Y",
                    x_gantry_slope,
                    "Bare X-gantry beam structural mass excluding linear rail and backer; accelerates on every Y move and governs X-gantry span inertia.",
                ),
                (
                    "x_rail_backer",
                    "X-Rail & Backer",
                    x_rail_backer_g,
                    round(min(100.0, 0.35 * x_rail_backer_g), 1),
                    "Y",
                    limiting_axis == "Y",
                    x_gantry_slope,
                    "X-axis linear guide rail and backer bar mass mounted along the X gantry; rides along on every Y acceleration.",
                ),
                (
                    "y_carriages",
                    "XY-Joints & Y-Carriages",
                    y_carriages_g,
                    round(min(80.0, 0.35 * y_carriages_g), 1),
                    "Y",
                    limiting_axis == "Y",
                    y_car_slope,
                    "Combined left/right Y-rail carriages and XY-joint bracket hardware; accelerates on Y moves and carries gantry belt tension.",
                ),
                (
                    "y_gantry_total",
                    "Total Moving Y-Gantry",
                    y_gantry_total_g,
                    round(0.25 * y_gantry_total_g, 1),
                    "Y",
                    limiting_axis == "Y",
                    y_gantry_slope,
                    "Combined Y-axis moving assembly (toolhead, X-beam, X-rail, backer, and XY-joints/carriages) accelerated by the Y drive path.",
                ),
            ]
        )
        total_struct_slope = y_gantry_slope if limiting_axis == "Y" else toolhead_slope

    delta_cut = min(10.0, 0.25 * m_lim_g) if m_lim_g > 1.0 else 0.0
    slope_scale = (100.0 / delta_cut) if delta_cut > 0.0 else 0.0

    items: list[dict[str, Any]] = []
    for item_id, label, mass_g, cut_g, affects_axis, moves_on_lim, struct_slope, note in raw_items:
        df_hz, da = _coupled_gain(cut_g, moves_on_lim, struct_slope)
        hz_100 = (
            _coupled_gain(delta_cut, moves_on_lim, struct_slope)[0] * slope_scale
            if delta_cut > 0.0
            else 0.0
        )
        items.append(
            {
                "id": item_id,
                "label": label,
                "current_mass_g": round(float(mass_g), 1),
                "example_cut_g": round(float(cut_g), 1),
                "hz_per_100g": round(float(hz_100), 2),
                "est_hz_gain": round(float(df_hz), 2),
                "est_accel_gain_mm_s2": round(float(da), 0),
                "affects_axis": affects_axis,
                "note": note,
            }
        )

    total_potential_cut_g = round(0.25 * m_lim_g, 1)
    tot_df_hz, tot_da = _coupled_gain(total_potential_cut_g, True, total_struct_slope)

    return {
        "limiting_axis": limiting_axis,
        "limiting_shaper_hz": round(float(f_eff), 2),
        "limiting_max_accel_mm_s2": round(float(a_max), 0),
        "limiting_moving_mass_g": round(float(m_lim_g), 1),
        "total_potential_cut_g": total_potential_cut_g,
        "total_potential_hz_gain": round(float(tot_df_hz), 2),
        "total_potential_accel_gain_mm_s2": round(float(tot_da), 0),
        "items": items,
    }


def evaluate_printer(
    spec: PrinterSpec,
    gcode_text: str | None = None,
    toolhead_rel_pos: tuple[float, float, float] = (0.5, 0.5, 0.5),
    filament_id: str = "pla",
    panel_mesh_n: int = 3,
    bed_mesh_n: int = 8,
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
        bed_plate_fea = solve_bed_plate_fem(spec, mesh_n=max(4, int(bed_mesh_n)))

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

    weight_shaper_gains = compute_weight_shaper_gains(assembly, modal_fea, sensitivity)

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
        weight_shaper_gains=weight_shaper_gains,
    )


def _extract_verify_summary(report: SimulationReport, solve_time_ms: float) -> dict:
    bp = report.bed_plate_fea
    return {
        "fundamental_freq_hz": round(float(report.modal_fea.fundamental_freq_hz), 2),
        "recommended_max_accel_mm_s2": round(float(report.modal_fea.recommended_max_accel_mm_s2), 0),
        "combined_max_deflection_um": round(float(report.static_fea.combined_max_deflection_um), 2),
        "panel_drum_freq_hz": round(float(report.modal_fea.panel_drum_freq_hz), 2),
        "panel_max_von_mises_mpa": round(float(report.static_fea.panel_max_von_mises_mpa), 3),
        "num_shell_elements": len(report.assembly.shell_elements),
        "bed_flatness_error_um": round(float(bp.total_flatness_error_um), 2) if bp else 0.0,
        "bed_first_mode_hz": round(float(bp.plate_first_mode_hz), 2) if bp else 0.0,
        "solve_time_ms": round(float(solve_time_ms), 1),
    }


def verify_printer_deep(
    spec: PrinterSpec,
    gcode_text: str | None = None,
    toolhead_rel_pos: tuple[float, float, float] = (0.5, 0.5, 0.5),
    filament_id: str = "pla",
    deep_mesh_n: int = 8,
    deep_bed_mesh_n: int = 16,
) -> dict:
    """Compare balanced interactive 3x3/8x8 FEA vs maximum-accuracy 8x8/16x16 verification."""
    fast_n = 3
    fast_bed_n = 8
    deep_n = max(4, int(deep_mesh_n))
    deep_bed_n = max(8, int(deep_bed_mesh_n))

    t0 = time.perf_counter()
    fast_report = evaluate_printer(
        spec,
        gcode_text=gcode_text,
        toolhead_rel_pos=toolhead_rel_pos,
        filament_id=filament_id,
        panel_mesh_n=fast_n,
        bed_mesh_n=fast_bed_n,
    )
    fast_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    deep_report = evaluate_printer(
        spec,
        gcode_text=gcode_text,
        toolhead_rel_pos=toolhead_rel_pos,
        filament_id=filament_id,
        panel_mesh_n=deep_n,
        bed_mesh_n=deep_bed_n,
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
        "fast_bed_mesh_n": fast_bed_n,
        "deep_bed_mesh_n": deep_bed_n,
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
