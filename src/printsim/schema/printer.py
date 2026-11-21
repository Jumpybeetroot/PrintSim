"""Parametric 3D printer assembly DSL (YAML / JSON / Python)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

Architecture = Literal["corexy", "cartesian_bedslinger", "delta"]
BracingType = Literal["none", "corner_gussets", "rear_cross", "full_truss"]
PanelMountingType = Literal["none", "clip_enclosure", "bolted_structural"]
ExtrusionFillingType = Literal[
    "none",
    "pu_foam",
    "silica_sand",
    "lead_shot",
    "epoxy_granite",
    "uhpc_grout",
]
BracketMaterialType = Literal[
    "abs_printed",
    "pet_cf_printed",
    "pa12_cf_printed",
    "ppa_cf_printed",
    "pps_cf_printed",
    "aluminum_6061_t6",
    "aluminum_7075_t6",
]
DriveModeType = Literal["2wd", "awd"]
XRailMountType = Literal["front", "top", "bottom", "dual_top_bottom"]
YRailMountType = Literal["top", "inner", "bottom"]
InputShaperType = Literal["auto", "none", "zv", "mzv", "ei", "2hump_ei"]

# Engineering catalog for frame extrusion cavity fillings (SI units; literature / CNC casting data)
EXTRUSION_FILLINGS: dict[str, dict] = {
    "none": {
        "id": "none",
        "name": "Hollow (Unfilled)",
        "short_name": "Hollow",
        "density_kg_m3": 0.0,
        "youngs_modulus_pa": 0.0,
        "bonded": False,
        "damping_boost": 0.0,
        "cost_per_kg_usd": 0.0,
        "color_hex": "#94a3b8",
        "notes": "Standard hollow aluminum extrusion (low structural damping)",
    },
    "pu_foam": {
        "id": "pu_foam",
        "name": "Expanding PU Acoustic Foam",
        "short_name": "PU Foam",
        "density_kg_m3": 45.0,
        "youngs_modulus_pa": 0.02e9,
        "bonded": False,
        "damping_boost": 0.010,
        "cost_per_kg_usd": 18.0,
        "color_hex": "#fde68a",
        "notes": "Ultra-light acoustic deadening; negligible added mass or stiffness",
    },
    "silica_sand": {
        "id": "silica_sand",
        "name": "Kiln-Dried Silica Sand",
        "short_name": "Silica Sand",
        "density_kg_m3": 1550.0,
        "youngs_modulus_pa": 0.0,
        "bonded": False,
        "damping_boost": 0.028,
        "cost_per_kg_usd": 1.5,
        "color_hex": "#d6b88d",
        "notes": "Low-cost granular Coulomb friction damping; zero bonded bending stiffness",
    },
    "lead_shot": {
        "id": "lead_shot",
        "name": "Fine Lead Shot (#8–#9)",
        "short_name": "Lead Shot",
        "density_kg_m3": 6800.0,
        "youngs_modulus_pa": 0.0,
        "bonded": False,
        "damping_boost": 0.042,
        "cost_per_kg_usd": 11.0,
        "color_hex": "#475569",
        "notes": "Maximum granular impact damping & base inertia; heavy mass lowers raw Hz",
    },
    "epoxy_granite": {
        "id": "epoxy_granite",
        "name": "Epoxy Granite (85% Quartz + Epoxy)",
        "short_name": "Epoxy Granite",
        "density_kg_m3": 2250.0,
        "youngs_modulus_pa": 30.0e9,
        "bonded": True,
        "damping_boost": 0.036,
        "cost_per_kg_usd": 9.5,
        "color_hex": "#64748b",
        "notes": "Mineral casting bonds to cavity walls: +10-16% bending stiffness & strong resin damping",
    },
    "uhpc_grout": {
        "id": "uhpc_grout",
        "name": "Non-Shrink UHPC / CSA Grout",
        "short_name": "UHPC Grout",
        "density_kg_m3": 2350.0,
        "youngs_modulus_pa": 40.0e9,
        "bonded": True,
        "damping_boost": 0.030,
        "cost_per_kg_usd": 4.5,
        "color_hex": "#9ca3af",
        "notes": "High-modulus bonded cementitious grout: highest core EI boost + solid damping",
    },
}

# Gantry bracket / motor-mount / XY-joint material catalog (ISO 527 / ISO 75 TDS + Monolith data)
GANTRY_BRACKET_CATALOG: dict[str, dict] = {
    "abs_printed": {
        "id": "abs_printed",
        "name": "3D-Printed ABS / ASA (2.1 GPa)",
        "short_name": "Printed ABS",
        "youngs_modulus_gpa": 2.1,
        "hdt_c": 95.0,
        "creep_tension_limit_n": 45.0,
        "cnc_machined": False,
        "kit_cost_usd": 28.0,
        "carriage_mass_delta_g": 0.0,
        "color_hex": "#e63946",
    },
    "pet_cf_printed": {
        "id": "pet_cf_printed",
        "name": "3D-Printed PET-CF15 Annealed (5.2 GPa)",
        "short_name": "Printed PET-CF15",
        "youngs_modulus_gpa": 5.2,
        "hdt_c": 182.0,
        "creep_tension_limit_n": 75.0,
        "cnc_machined": False,
        "kit_cost_usd": 52.0,
        "carriage_mass_delta_g": 10.0,
        "color_hex": "#3b4252",
    },
    "pa12_cf_printed": {
        "id": "pa12_cf_printed",
        "name": "3D-Printed PA12-CF15 (6.2 GPa)",
        "short_name": "Printed PA12-CF",
        "youngs_modulus_gpa": 6.2,
        "hdt_c": 155.0,
        "creep_tension_limit_n": 65.0,
        "cnc_machined": False,
        "kit_cost_usd": 72.0,
        "carriage_mass_delta_g": 6.0,
        "color_hex": "#333840",
    },
    "ppa_cf_printed": {
        "id": "ppa_cf_printed",
        "name": "3D-Printed PPA-CF Annealed (11.8 GPa)",
        "short_name": "Printed PPA-CF",
        "youngs_modulus_gpa": 11.8,
        "hdt_c": 196.0,
        "creep_tension_limit_n": 105.0,
        "cnc_machined": False,
        "kit_cost_usd": 98.0,
        "carriage_mass_delta_g": 9.0,
        "color_hex": "#2d333b",
    },
    "pps_cf_printed": {
        "id": "pps_cf_printed",
        "name": "3D-Printed PPS-CF Annealed (8.5 GPa, 235°C HDT)",
        "short_name": "Printed PPS-CF",
        "youngs_modulus_gpa": 8.5,
        "hdt_c": 235.0,
        "creep_tension_limit_n": 115.0,
        "cnc_machined": False,
        "kit_cost_usd": 120.0,
        "carriage_mass_delta_g": 12.0,
        "color_hex": "#23272e",
    },
    "aluminum_6061_t6": {
        "id": "aluminum_6061_t6",
        "name": "CNC Aluminum 6061-T6 (68.9 GPa)",
        "short_name": "CNC Al 6061",
        "youngs_modulus_gpa": 68.9,
        "hdt_c": 300.0,
        "creep_tension_limit_n": 250.0,
        "cnc_machined": True,
        "kit_cost_usd": 210.0,
        "carriage_mass_delta_g": 42.0,
        "color_hex": "#c6d0db",
    },
    "aluminum_7075_t6": {
        "id": "aluminum_7075_t6",
        "name": "CNC Aluminum 7075-T6 Billet (71.7 GPa)",
        "short_name": "CNC Al 7075",
        "youngs_modulus_gpa": 71.7,
        "hdt_c": 300.0,
        "creep_tension_limit_n": 280.0,
        "cnc_machined": True,
        "kit_cost_usd": 285.0,
        "carriage_mass_delta_g": 46.0,
        "color_hex": "#cbd5e1",
    },
}


class FrameNodeSpec(BaseModel):
    """Explicit 3D structural joint node (coordinates in mm)."""

    id: str
    position_mm: tuple[float, float, float]
    fixed: bool = False
    lumped_mass_g: float = Field(0.0, ge=0)
    role: Literal["frame", "gantry", "foot", "mount", "toolhead"] = "frame"


class FrameMemberSpec(BaseModel):
    """Explicit structural beam/bar/rail connecting two nodes."""

    id: str
    start_node: str
    end_node: str
    profile: str = "extrusion_2020"
    material: str = "aluminum_6063_t5"
    role: Literal["frame", "brace", "gantry", "rail", "tower"] = "frame"


class CustomMeshSpec(BaseModel):
    """Reference to an external STL / OBJ / STEP part attached to the printer."""

    id: str
    name: str
    file_path: str
    attach_to: Literal["frame", "gantry", "toolhead", "bed"] = "frame"
    position_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    material: str = "abs_printed"
    override_mass_g: float | None = Field(None, gt=0)


class FrameSpec(BaseModel):
    """Parametric frame specification with optional custom nodes/members override."""

    width_mm: float = Field(460.0, gt=100, description="Outer frame X span (mm)")
    depth_mm: float = Field(460.0, gt=100, description="Outer frame Y span (mm)")
    height_mm: float = Field(500.0, gt=100, description="Outer frame Z height (mm)")
    profile: str = Field("extrusion_2020", description="Primary structural frame profile ID")
    material: str = Field("aluminum_6063_t5", description="Primary structural frame material ID")
    bracing: BracingType = Field("corner_gussets", description="Structural triangulation strategy")
    brace_profile: str = Field("brace_rod_10mm", description="Profile for diagonal braces")
    brace_material: str = Field("aluminum_6063_t5", description="Material for diagonal braces")
    enclosed_panels: bool = Field(
        False, description="Whether side/rear/top panels are installed on the frame"
    )
    panel_mounting: PanelMountingType = Field(
        "bolted_structural",
        description="Panel attachment stiffness: none, clip_enclosure (loose), or bolted_structural (shear diaphragm)",
    )
    panel_material: str = Field(
        "acm_dibond",
        description="Material ID for enclosure/structural panels (acrylic_pmma, polycarbonate, acm_dibond, aluminum_5052_sheet, carbon_fiber_plate)",
    )
    panel_thickness_mm: float = Field(3.0, ge=1.0, le=12.0, description="Panel sheet thickness (mm)")
    bolt_panels_to_gantry: bool = Field(
        True,
        description="On non-flying-gantry CoreXY machines, bolt structural panels directly to fixed Y-rail & rear gantry beams",
    )
    extrusion_filling: ExtrusionFillingType = Field(
        "none",
        description="Frame extrusion cavity filling for damping & bonded core stiffness (none, pu_foam, silica_sand, lead_shot, epoxy_granite, uhpc_grout)",
    )
    custom_nodes: list[FrameNodeSpec] = Field(default_factory=list)
    custom_members: list[FrameMemberSpec] = Field(default_factory=list)


class BeltSpec(BaseModel):
    """Drive belt specification calibrated to Gates PowerGrip 2GT datasheets."""

    type: Literal["gt2_6mm", "gt2_9mm", "gt2_12mm"] = "gt2_6mm"
    core: Literal["fiberglass", "steel", "aramid"] = "fiberglass"
    pulley_teeth: int = Field(20, ge=12, le=40)
    pitch_mm: float = Field(2.0, gt=0)
    pretension_n: float = Field(25.0, gt=0, le=200.0)

    @property
    def stiffness_ea_n(self) -> float:
        """Effective belt tensile modulus EA (Newtons) calibrated to Gates PowerGrip 2GT datasheets."""
        width_scale = {"gt2_6mm": 1.0, "gt2_9mm": 1.5, "gt2_12mm": 2.0}[self.type]
        core_base = {"fiberglass": 52000.0, "aramid": 74000.0, "steel": 110000.0}[self.core]
        return core_base * width_scale

    @property
    def linear_mass_kg_m(self) -> float:
        """Belt linear mass density mu (kg/m) for transverse pluck frequency f = (1/2L)*sqrt(T/mu)."""
        width_scale = {"gt2_6mm": 1.0, "gt2_9mm": 1.5, "gt2_12mm": 2.0}[self.type]
        core_density_scale = {"fiberglass": 1.0, "aramid": 0.94, "steel": 1.35}[self.core]
        return 0.0115 * width_scale * core_density_scale

    @property
    def max_working_tension_n(self) -> float:
        """Safe continuous peak working tension (N) per Gates 2GT engineering ratings."""
        width_scale = {"gt2_6mm": 1.0, "gt2_9mm": 1.6, "gt2_12mm": 2.2}[self.type]
        core_limit = {"fiberglass": 58.0, "aramid": 85.0, "steel": 105.0}[self.core]
        return core_limit * width_scale

    def pluck_freq_hz(self, span_mm: float = 150.0) -> float:
        """Transverse acoustic belt pluck frequency at given free span (standard Voron/VzBot 150 mm)."""
        span_m = max(span_mm * 1e-3, 0.05)
        return (1.0 / (2.0 * span_m)) * ((self.pretension_n / self.linear_mass_kg_m) ** 0.5)

    @property
    def pitch_radius_m(self) -> float:
        return (self.pulley_teeth * self.pitch_mm * 1e-3) / (2.0 * 3.141592653589793)


class GantrySpec(BaseModel):
    """X/Y gantry beam, rails, drive architecture (2WD/AWD), and belt parameters."""

    drive_mode: DriveModeType = Field(
        "2wd", description="XY drive layout: 2wd (2-motor CoreXY) or awd (4-motor All-Wheel Drive)"
    )
    double_shear_pulleys: bool = Field(
        False, description="Double-shear bearing supports on XY motor shafts & idlers for high belt tension"
    )
    bracket_material: BracketMaterialType = Field(
        "abs_printed",
        description="Material for XY motor mounts, front idlers, and XY-joint gantry brackets",
    )
    x_beam_profile: str = "extrusion_2020"
    x_beam_material: str = "aluminum_6063_t5"
    x_rail_profile: str = "mgn12_rail"
    y_rail_profile: str = "mgn12_rail"
    rail_material: str = "steel_4140"
    x_rail_mount: XRailMountType = Field(
        "front",
        description="X-axis linear rail mounting face on the X gantry beam: front (Voron standard, max Y stiffness), top (RatRig/Annex, max Z stiffness), bottom (under-slung, shorter nozzle drop), or dual_top_bottom (symmetric dual-rail I-beam)",
    )
    y_rail_mount: YRailMountType = Field(
        "top",
        description="Y-axis linear rail mounting face on the left/right Y extrusions: top (Voron standard), inner (side-mounted Monolith/VzBot, max X stiffness), or bottom (inverted)",
    )
    dual_x_rails: bool = False
    carriage_mass_g: float = Field(140.0, gt=0, description="Combined X/Y gantry carriages & joints (g)")
    belt: BeltSpec = Field(default_factory=BeltSpec)

    @property
    def effective_x_rail_mount(self) -> XRailMountType:
        """Resolve effective X rail mount, honoring legacy ``dual_x_rails=True``."""
        if self.dual_x_rails or self.x_rail_mount == "dual_top_bottom":
            return "dual_top_bottom"
        return self.x_rail_mount

    @property
    def effective_dual_x_rails(self) -> bool:
        """True when dual top+bottom X rails are active."""
        return self.dual_x_rails or self.x_rail_mount == "dual_top_bottom"


class StepperSpec(BaseModel):
    """Stepper motor electromechanical model calibrated to manufacturer datasheets."""

    preset_id: str = "ldo_2504ac"
    name: str = "NEMA17-48mm (LDO 42STH48-2504AC)"
    holding_torque_nm: float = Field(0.55, gt=0)
    rated_current_a: float = Field(2.5, gt=0, description="Motor datasheet rated RMS current per phase (A)")
    run_current_a: float = Field(
        1.2, gt=0.1, le=5.0, description="Configured driver RMS run current per phase (A)"
    )
    phase_resistance_ohm: float = Field(1.2, gt=0.05, description="Winding phase resistance R (Ohms)")
    inductance_mh: float = Field(1.5, gt=0, description="Winding phase inductance L (mH)")
    supply_voltage_v: float = Field(24.0, gt=0, le=96.0, description="Stepper driver bus voltage (V)")
    rotor_inertia_kg_m2: float = Field(6.8e-6, gt=0)
    motor_mass_g: float = Field(340.0, gt=0)
    steps_per_rev: int = Field(200, ge=100, description="200 for 1.8 deg, 400 for 0.9 deg")

    @property
    def pole_pairs(self) -> float:
        """Electrical pole pairs p = steps_per_rev / 4 (50 for 1.8°, 100 for 0.9°)."""
        return self.steps_per_rev / 4.0

    @property
    def effective_holding_torque_nm(self) -> float:
        """Current-scaled static holding torque with magnetic core soft saturation above rated current."""
        import math

        ratio = max(self.run_current_a, 0.05) / max(self.rated_current_a, 0.1)
        return self.holding_torque_nm * (math.tanh(1.1 * ratio) / math.tanh(1.1))

    def dynamic_voltage_efficiency(self, rpm: float = 250.0) -> float:
        """Inductive current-tracking efficiency eta_V in (0, 1] at speed/vibration operating point."""
        import math

        l_h = self.inductance_mh * 1e-3
        omega_mech = max(rpm, 10.0) * (2.0 * math.pi / 60.0)
        v_inductive = omega_mech * self.pole_pairs * l_h * self.run_current_a
        v_res = self.run_current_a * self.phase_resistance_ohm
        v_headroom = max(self.supply_voltage_v - v_res, 1.0)
        return 1.0 / math.sqrt(1.0 + (v_inductive / v_headroom) ** 2)

    def torsional_stiffness_nm_per_rad(self, rpm: float = 250.0) -> float:
        """Electromagnetic torsional stiffness k_theta = p * T_hold(I) * eta_V (N·m/rad)."""
        return self.pole_pairs * self.effective_holding_torque_nm * self.dynamic_voltage_efficiency(rpm)

    @property
    def power_dissipation_w(self) -> float:
        """Approximate steady-state 2-phase copper heat dissipation P = 2 * I_rms^2 * R (W)."""
        return 2.0 * (self.run_current_a**2) * self.phase_resistance_ohm


STEPPER_CATALOG: dict[str, dict] = {
    "ldo_2504ac": {
        "preset_id": "ldo_2504ac",
        "name": "LDO 42STH48-2504AC (48mm 1.8° Speedy Power)",
        "holding_torque_nm": 0.55,
        "rated_current_a": 2.5,
        "run_current_a": 1.2,
        "phase_resistance_ohm": 1.2,
        "inductance_mh": 1.5,
        "rotor_inertia_kg_m2": 6.8e-6,
        "motor_mass_g": 340.0,
        "steps_per_rev": 200,
    },
    "ldo_2804ah": {
        "preset_id": "ldo_2804ah",
        "name": "LDO 42STH48-2804AH (48mm 1.8° Ultra-Low 0.6mH)",
        "holding_torque_nm": 0.42,
        "rated_current_a": 2.8,
        "run_current_a": 2.0,
        "phase_resistance_ohm": 0.7,
        "inductance_mh": 0.6,
        "rotor_inertia_kg_m2": 6.2e-6,
        "motor_mass_g": 350.0,
        "steps_per_rev": 200,
    },
    "ldo_60_3004": {
        "preset_id": "ldo_60_3004",
        "name": "LDO 42STH60-3004MAC (60mm 1.8° Kraken High-Torque)",
        "holding_torque_nm": 0.80,
        "rated_current_a": 3.0,
        "run_current_a": 2.2,
        "phase_resistance_ohm": 0.9,
        "inductance_mh": 1.8,
        "rotor_inertia_kg_m2": 8.6e-6,
        "motor_mass_g": 520.0,
        "steps_per_rev": 200,
    },
    "ldo_09deg_2504": {
        "preset_id": "ldo_09deg_2504",
        "name": "LDO 42STH48-2504MAC (48mm 0.9° 2× Magnetic Stiffness)",
        "holding_torque_nm": 0.50,
        "rated_current_a": 2.5,
        "run_current_a": 1.6,
        "phase_resistance_ohm": 1.2,
        "inductance_mh": 2.2,
        "rotor_inertia_kg_m2": 6.8e-6,
        "motor_mass_g": 340.0,
        "steps_per_rev": 400,
    },
    "moons_420i": {
        "preset_id": "moons_420i",
        "name": "MOONS' MS17HD6P420I (48mm 1.8° Voron Standard)",
        "holding_torque_nm": 0.65,
        "rated_current_a": 2.0,
        "run_current_a": 1.2,
        "phase_resistance_ohm": 1.3,
        "inductance_mh": 2.8,
        "rotor_inertia_kg_m2": 7.2e-6,
        "motor_mass_g": 360.0,
        "steps_per_rev": 200,
    },
    "stepperonline_2004": {
        "preset_id": "stepperonline_2004",
        "name": "STEPPERONLINE 17HS19-2004S1 (48mm 1.8° Budget)",
        "holding_torque_nm": 0.59,
        "rated_current_a": 2.0,
        "run_current_a": 1.0,
        "phase_resistance_ohm": 1.4,
        "inductance_mh": 3.0,
        "rotor_inertia_kg_m2": 6.8e-6,
        "motor_mass_g": 390.0,
        "steps_per_rev": 200,
    },
}


class KinematicsSpec(BaseModel):
    """Motion limits, build volume, motors, and input shaping."""

    build_volume_mm: tuple[float, float, float] = (300.0, 300.0, 300.0)
    max_velocity_mm_s: float = Field(450.0, gt=0)
    max_accel_mm_s2: float = Field(8000.0, gt=0)
    max_z_velocity_mm_s: float = Field(25.0, gt=0)
    max_z_accel_mm_s2: float = Field(300.0, gt=0)
    square_corner_velocity_mm_s: float = Field(5.0, gt=0)
    input_shaper: InputShaperType = "auto"
    xy_stepper: StepperSpec = Field(default_factory=StepperSpec)
    z_stepper: StepperSpec = Field(
        default_factory=lambda: StepperSpec(
            name="NEMA17-40mm Z", holding_torque_nm=0.45, motor_mass_g=280.0
        )
    )
    # Delta-specific parameters (ignored for CoreXY/Cartesian)
    delta_arm_length_mm: float = Field(280.0, gt=50)
    delta_radius_mm: float = Field(145.0, gt=30)


class HotendSpec(BaseModel):
    """Thermomechanical hotend & melt-zone specification."""

    name: str = "High-Flow Bi-Metal Hotend"
    heater_power_w: float = Field(60.0, gt=5)
    block_material: str = "copper_c110"
    block_mass_g: float = Field(14.0, gt=1)
    nozzle_material: str = "brass_c360"
    nozzle_diameter_mm: float = Field(0.4, gt=0.1, le=2.0)
    melt_zone_length_mm: float = Field(16.0, gt=4.0, le=50.0)
    heatbreak_material: str = "titanium_grade5"
    heatbreak_length_mm: float = Field(12.0, gt=3.0)
    heatbreak_wall_mm: float = Field(0.25, gt=0.05)
    heatbreak_id_mm: float = Field(2.0, gt=1.5)
    heatsink_thermal_resistance_k_w: float = Field(
        3.2, gt=0.2, description="Heatsink-to-ambient thermal resistance with fan running (K/W)"
    )
    part_cooling_cfm: float = Field(8.5, ge=0.0, description="Part cooling fan airflow (CFM)")
    max_temp_c: float = Field(300.0, gt=180.0, le=550.0)
    ambient_temp_c: float = Field(25.0, ge=0.0, le=100.0)


class ExtruderSpec(BaseModel):
    """Filament drive extruder specification."""

    type: Literal["direct_drive", "bowden"] = "direct_drive"
    gear_ratio: float = Field(5.0, gt=0.5)
    max_push_force_n: float = Field(55.0, gt=5.0)
    filament_diameter_mm: float = Field(1.75, gt=1.0)


class ToolheadSpec(BaseModel):
    """Complete toolhead assembly (extruder, hotend, shroud, fans, carriage, CoG & nozzle lever arm)."""

    preset_id: str = "stealthburner"
    name: str = "Stealth-Style Direct Toolhead"
    total_mass_g: float = Field(380.0, gt=20.0)
    cog_offset_mm: tuple[float, float, float] = Field(
        (0.0, -18.0, -25.0),
        description="Toolhead center-of-mass offset [dx, dy, dz] from X-beam centerline (mm)",
    )
    nozzle_offset_mm: tuple[float, float, float] = Field(
        (0.0, -22.0, -62.0),
        description="Nozzle tip offset [dx, dy, dz] from X-beam centerline (mm)",
    )
    mount_material: BracketMaterialType = Field(
        "abs_printed",
        description="Toolhead carriage & XY-joint mount material affecting nozzle pitch/twist stiffness",
    )
    envelope_mm: tuple[float, float, float] = (68.0, 74.0, 95.0)
    hotend: HotendSpec = Field(default_factory=HotendSpec)
    extruder: ExtruderSpec = Field(default_factory=ExtruderSpec)


TOOLHEAD_PRESETS: dict[str, dict] = {
    "stealthburner": {
        "preset_id": "stealthburner",
        "name": "Voron StealthBurner + CW2",
        "total_mass_g": 380.0,
        "cog_offset_mm": [0.0, -22.0, -28.0],
        "nozzle_offset_mm": [0.0, -26.0, -66.0],
        "mount_material": "abs_printed",
    },
    "xol_compact": {
        "preset_id": "xol_compact",
        "name": "XOL / A4T Compact Dual-5015",
        "total_mass_g": 285.0,
        "cog_offset_mm": [0.0, -15.0, -18.0],
        "nozzle_offset_mm": [0.0, -18.0, -54.0],
        "mount_material": "pa12_cf_printed",
    },
    "archetype_cpap": {
        "preset_id": "archetype_cpap",
        "name": "Archetype / Monolith CPAP Ultra-Low CoG",
        "total_mass_g": 235.0,
        "cog_offset_mm": [0.0, -11.0, -12.0],
        "nozzle_offset_mm": [0.0, -14.0, -48.0],
        "mount_material": "aluminum_6061_t6",
    },
    "heavy_uhf_long": {
        "preset_id": "heavy_uhf_long",
        "name": "Heavy Long-Nozzle UHF Toolhead",
        "total_mass_g": 460.0,
        "cog_offset_mm": [0.0, -26.0, -34.0],
        "nozzle_offset_mm": [0.0, -28.0, -78.0],
        "mount_material": "abs_printed",
    },
}


class BedSpec(BaseModel):
    """Heated print bed & Z-axis mechanism."""

    size_mm: tuple[float, float] = (300.0, 300.0)
    plate_material: str = "mic6_aluminum"
    plate_thickness_mm: float = Field(8.0, gt=1.0)
    heater_power_w: float = Field(600.0, gt=50.0)
    mass_g: float = Field(2400.0, gt=100.0)
    z_drive: Literal[
        "leadscrew_single", "leadscrew_dual", "leadscrew_triple", "belted_quad_z"
    ] = "belted_quad_z"


class PrinterSpec(BaseModel):
    """Root declarative specification for a complete 3D printer design."""

    name: str = "Custom 3D Printer"
    description: str = ""
    architecture: Architecture = "corexy"
    frame: FrameSpec = Field(default_factory=FrameSpec)
    gantry: GantrySpec = Field(default_factory=GantrySpec)
    kinematics: KinematicsSpec = Field(default_factory=KinematicsSpec)
    toolhead: ToolheadSpec = Field(default_factory=ToolheadSpec)
    bed: BedSpec = Field(default_factory=BedSpec)
    custom_meshes: list[CustomMeshSpec] = Field(default_factory=list)

    @property
    def is_flying_gantry(self) -> bool:
        """True for flying-gantry CoreXY machines (e.g. Voron 2.4 quad-belted Z) where the gantry moves in Z."""
        return self.architecture == "corexy" and self.bed.z_drive == "belted_quad_z"

    @property
    def panels_bolted_to_gantry_effective(self) -> bool:
        """True when bolted structural panels can and do tie directly into fixed gantry beams."""
        return (
            self.architecture == "corexy"
            and self.frame.enclosed_panels
            and self.frame.panel_mounting == "bolted_structural"
            and self.frame.bolt_panels_to_gantry
            and not self.is_flying_gantry
        )

    @model_validator(mode="after")
    def _validate_build_fits_frame(self) -> "PrinterSpec":
        # Warn/enforce only absurd negative clearances; collision engine handles fine clearances
        bx, by, bz = self.kinematics.build_volume_mm
        if bx <= 0 or by <= 0 or bz <= 0:
            raise ValueError("build_volume_mm dimensions must be positive")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PrinterSpec":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)

    def save_yaml(self, path: str | Path) -> None:
        Path(path).write_text(self.to_yaml(), encoding="utf-8")
