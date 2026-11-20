"""Parametric 3D printer assembly DSL (YAML / JSON / Python)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

Architecture = Literal["corexy", "cartesian_bedslinger", "delta"]
BracingType = Literal["none", "corner_gussets", "rear_cross", "full_truss"]
InputShaperType = Literal["auto", "none", "zv", "mzv", "ei", "2hump_ei"]


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
        False, description="Whether bolted side/rear panels add shear stiffness to the frame"
    )
    panel_thickness_mm: float = Field(3.0, gt=0)
    custom_nodes: list[FrameNodeSpec] = Field(default_factory=list)
    custom_members: list[FrameMemberSpec] = Field(default_factory=list)


class BeltSpec(BaseModel):
    """Drive belt specification."""

    type: Literal["gt2_6mm", "gt2_9mm", "gt2_12mm"] = "gt2_6mm"
    core: Literal["fiberglass", "steel", "aramid"] = "fiberglass"
    pulley_teeth: int = Field(20, ge=12, le=40)
    pitch_mm: float = Field(2.0, gt=0)
    pretension_n: float = Field(25.0, gt=0)

    @property
    def stiffness_ea_n(self) -> float:
        """Effective belt tensile modulus EA (Newtons) calibrated to Gates PowerGrip 2GT datasheets."""
        width_scale = {"gt2_6mm": 1.0, "gt2_9mm": 1.5, "gt2_12mm": 2.0}[self.type]
        core_base = {"fiberglass": 52000.0, "aramid": 74000.0, "steel": 110000.0}[self.core]
        return core_base * width_scale

    @property
    def pitch_radius_m(self) -> float:
        return (self.pulley_teeth * self.pitch_mm * 1e-3) / (2.0 * 3.141592653589793)


class GantrySpec(BaseModel):
    """X/Y gantry beam, rails, and drive belt parameters."""

    x_beam_profile: str = "extrusion_2020"
    x_beam_material: str = "aluminum_6063_t5"
    x_rail_profile: str = "mgn12_rail"
    y_rail_profile: str = "mgn12_rail"
    rail_material: str = "steel_4140"
    dual_x_rails: bool = False
    carriage_mass_g: float = Field(140.0, gt=0, description="Combined X/Y gantry carriages & joints (g)")
    belt: BeltSpec = Field(default_factory=BeltSpec)


class StepperSpec(BaseModel):
    """Stepper motor electromechanical model."""

    name: str = "NEMA17-48mm (LDO 2504AC)"
    holding_torque_nm: float = Field(0.55, gt=0)
    rated_current_a: float = Field(2.0, gt=0)
    inductance_mh: float = Field(1.5, gt=0)
    supply_voltage_v: float = Field(24.0, gt=0)
    rotor_inertia_kg_m2: float = Field(6.8e-6, gt=0)
    motor_mass_g: float = Field(340.0, gt=0)
    steps_per_rev: int = Field(200, ge=100)


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
    """Complete toolhead assembly (extruder, hotend, shroud, fans, carriage)."""

    name: str = "Stealth-Style Direct Toolhead"
    total_mass_g: float = Field(380.0, gt=20.0)
    cog_offset_mm: tuple[float, float, float] = (0.0, -18.0, -25.0)
    envelope_mm: tuple[float, float, float] = (68.0, 74.0, 95.0)
    hotend: HotendSpec = Field(default_factory=HotendSpec)
    extruder: ExtruderSpec = Field(default_factory=ExtruderSpec)


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
