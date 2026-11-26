"""Pydantic schemas, centralized resolvers, and inspection payloads for lumped-model overrides."""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field, model_validator

from printsim.schema.confidence import (
    ITEM_CONFIDENCE_REGISTRY,
    confidence_tier,
    get_item_confidence,
)

if TYPE_CHECKING:
    from printsim.schema.printer import PrinterSpec

_G_ACC = 9.80665  # standard gravity (m/s^2)
# Reference suspended mass for catalog/baseline display of the pendulum geometric stiffness T / L.
SUSPENSION_REFERENCE_MASS_KG = 15.0


class CornerJointOverride(BaseModel):
    """Explicit overrides for frame corner joint stiffnesses and Iwan micro-slip damping."""

    k_t_n_per_mm: float | None = Field(None, ge=100.0, le=1e8)
    k_r_inplane_nm_rad: float | None = Field(None, ge=100.0, le=1e9)
    k_r_torsion_nm_rad: float | None = Field(None, ge=100.0, le=1e9)
    damping_delta: float | None = Field(None, ge=0.0, le=0.05)


class ExtrusionFillingOverride(BaseModel):
    """Explicit overrides for stationary frame extrusion cavity filling properties."""

    density_kg_m3: float | None = Field(None, ge=0.0, le=15000.0)
    youngs_modulus_gpa: float | None = Field(None, ge=0.0, le=200.0)
    bonded: bool | None = None
    damping_boost: float | None = Field(None, ge=0.0, le=0.10)


class FrameFeetOverride(BaseModel):
    """Explicit overrides for benchtop frame feet stiffness, damping, and paver base mass."""

    k_axial_n_per_mm: float | None = Field(None, ge=0.01, le=1e6)
    k_lateral_n_per_mm: float | None = Field(None, ge=0.01, le=1e6)
    damping_delta: float | None = Field(None, ge=0.0, le=0.08)
    base_mass_kg: float | None = Field(None, ge=0.0, le=200.0)


class BeltOverride(BaseModel):
    """Explicit overrides for XY drive belt axial EA, tooth shear, linear mass, and tension rating."""

    stiffness_ea_n: float | None = Field(None, ge=1000.0, le=2e6)
    tooth_shear_k_n_per_m_mm: float | None = Field(None, ge=5000.0, le=1e6)
    linear_mass_g_per_m: float | None = Field(None, ge=1.0, le=300.0)
    max_working_tension_n: float | None = Field(None, ge=10.0, le=3000.0)


class GantryBracketOverride(BaseModel):
    """Explicit overrides for XY gantry bracket stiffness, mass delta, creep limit, and damping."""

    stiffness_n_per_mm: float | None = Field(None, ge=20.0, le=200000.0)
    carriage_mass_delta_g: float | None = Field(None, ge=-200.0, le=500.0)
    creep_tension_limit_n: float | None = Field(None, ge=5.0, le=1000.0)
    damping_delta: float | None = Field(None, ge=-0.03, le=0.05)


class RailPreloadOverride(BaseModel):
    """Explicit override for linear rail carriage Hertzian preload fraction (Q / C)."""

    preload_fraction: float | None = Field(None, ge=0.0005, le=0.25)


class RailQualityOverride(BaseModel):
    """Explicit overrides for linear rail raceway/ball-sorting stiffness and drag multipliers."""

    stiffness_mult: float | None = Field(None, ge=0.1, le=5.0)
    drag_mult: float | None = Field(None, ge=0.1, le=5.0)


class BeltRoutingOverride(BaseModel):
    """Explicit overrides for CoreXY belt routing free-span and idler stiffness factors."""

    span_factor: float | None = Field(None, ge=0.3, le=2.0)
    idler_stiffness_factor: float | None = Field(None, ge=0.2, le=5.0)


class BearingClearanceOverride(BaseModel):
    """Explicit override for ISO 5753 idler bearing radial internal clearance play (um)."""

    radial_play_um: float | None = Field(None, ge=0.0, le=100.0)


class ZProbeOverride(BaseModel):
    """Explicit overrides for Z-probe carriage pitch compliance factor and CoG Y shift."""

    compliance_factor: float | None = Field(None, ge=0.05, le=2.0)
    cog_y_shift_mm: float | None = Field(None, ge=-60.0, le=60.0)


class ToolheadConstructionOverride(BaseModel):
    """Explicit overrides for Stage 2 toolhead chassis base stiffness and torsion multiplier."""

    body_stiffness_base_n_per_mm: float | None = Field(None, ge=25.0, le=50000.0)
    torsion_mult: float | None = Field(None, ge=0.1, le=10.0)


class HotendClampOverride(BaseModel):
    """Explicit override for Stage 3 hotend mount clamp base stiffness (N/mm)."""

    clamp_stiffness_base_n_per_mm: float | None = Field(None, ge=50.0, le=50000.0)


class BedMountOverride(BaseModel):
    """Explicit overrides for bed mount axial/lateral/thermal stiffness, damping, and standoff height."""

    k_axial_n_per_mm: float | None = Field(None, ge=1.0, le=500000.0)
    k_lateral_n_per_mm: float | None = Field(None, ge=1.0, le=500000.0)
    k_restraint_thermal_n_per_mm: float | None = Field(None, ge=0.0, le=500000.0)
    damping_delta: float | None = Field(None, ge=-0.02, le=0.05)
    standoff_height_mm: float | None = Field(None, ge=2.0, le=60.0)


class ZDriveMechanismOverride(BaseModel):
    """Explicit overrides for Z-axis leadscrew/ballscrew/belt nut, thrust, drive stiffness, and backlash."""

    k_nut_n_per_um: float | None = Field(None, ge=0.5, le=500.0)
    k_thrust_n_per_um: float | None = Field(None, ge=0.5, le=500.0)
    k_drive_n_per_um_per_corner: float | None = Field(None, ge=0.02, le=50.0)
    backlash_um: float | None = Field(None, ge=0.0, le=200.0)


class ModelOverridesSpec(BaseModel):
    """Optional parameter-level overrides for all 14 lumped physical models."""

    corner_joint: CornerJointOverride = Field(default_factory=CornerJointOverride)
    extrusion_filling: ExtrusionFillingOverride = Field(default_factory=ExtrusionFillingOverride)
    frame_feet: FrameFeetOverride = Field(default_factory=FrameFeetOverride)
    belt: BeltOverride = Field(default_factory=BeltOverride)
    gantry_bracket: GantryBracketOverride = Field(default_factory=GantryBracketOverride)
    rail_preload: RailPreloadOverride = Field(default_factory=RailPreloadOverride)
    rail_quality: RailQualityOverride = Field(default_factory=RailQualityOverride)
    belt_routing: BeltRoutingOverride = Field(default_factory=BeltRoutingOverride)
    bearing_clearance: BearingClearanceOverride = Field(default_factory=BearingClearanceOverride)
    z_probe: ZProbeOverride = Field(default_factory=ZProbeOverride)
    toolhead_construction: ToolheadConstructionOverride = Field(
        default_factory=ToolheadConstructionOverride
    )
    hotend_clamp: HotendClampOverride = Field(default_factory=HotendClampOverride)
    bed_mount: BedMountOverride = Field(default_factory=BedMountOverride)
    z_drive_mechanism: ZDriveMechanismOverride = Field(default_factory=ZDriveMechanismOverride)

    @model_validator(mode="before")
    @classmethod
    def _coerce_none_subgroups(cls, data: Any) -> Any:
        if data is None:
            return {}
        if isinstance(data, dict):
            cleaned = dict(data)
            for field_name in cls.model_fields:
                if field_name in cleaned and cleaned[field_name] is None:
                    cleaned[field_name] = {}
            return cleaned
        return data

    def to_compact_dict(self) -> dict[str, dict[str, Any]]:
        """Serialize only sub-groups and fields with explicit non-None overrides."""
        out: dict[str, dict[str, Any]] = {}
        for group_name in type(self).model_fields:
            sub = getattr(self, group_name)
            sub_dict = {
                k: v for k, v in sub.model_dump(mode="json").items() if v is not None
            }
            if sub_dict:
                out[group_name] = sub_dict
        return out

    def has_any_override(self) -> bool:
        """Return True if at least one override parameter is explicitly set."""
        return bool(self.to_compact_dict())


def _base_corner_joint_for_profile(profile_id: str, joint_type: str) -> tuple[float, float, float]:
    from printsim.schema import printer as _p
    from printsim.schema.profiles import get_profile

    jkey = joint_type.lower().strip()
    if jkey not in _p.CORNER_JOINT_BASE_2020:
        jkey = "blind_joint"

    k_t_base, k_r_base, k_tor_base = _p.CORNER_JOINT_BASE_2020[jkey]
    if jkey == "rigid":
        return k_t_base, k_r_base, k_tor_base

    prof = get_profile(profile_id)
    i_avg = 0.5 * (prof.iy_m4 + prof.iz_m4)
    scale_t = max(prof.width_m / 0.020, 0.4)
    scale_r = max((i_avg / 0.68e-8) ** 0.65, 0.25)
    return k_t_base * scale_t, k_r_base * scale_r, k_tor_base * scale_r


def resolve_corner_joint(spec: PrinterSpec) -> dict[str, Any]:
    """Resolve profile-scaled corner joint stiffnesses (SI) and Iwan damping."""
    from printsim.schema import printer as _p

    jtype = spec.frame.corner_joint_type
    kt_h, kr_h, ktor_h = _base_corner_joint_for_profile(spec.frame.profile, jtype)
    up_prof_id = spec.frame.effective_upright_profile
    if up_prof_id != spec.frame.profile:
        kt_u, kr_u, ktor_u = _base_corner_joint_for_profile(up_prof_id, jtype)
        base_k_t_n_m = kt_h if kt_h == kt_u else math.sqrt(kt_h * kt_u)
        base_k_r_in = kr_h if kr_h == kr_u else math.sqrt(kr_h * kr_u)
        base_k_r_tor = ktor_h if ktor_h == ktor_u else math.sqrt(ktor_h * ktor_u)
    else:
        base_k_t_n_m, base_k_r_in, base_k_r_tor = kt_h, kr_h, ktor_h

    base_damping = float(_p.CORNER_JOINT_IWAN_DAMPING.get(jtype.lower().strip(), 0.003))
    ov = spec.model_overrides.corner_joint

    k_t_n_m = float(ov.k_t_n_per_mm) * 1e3 if ov.k_t_n_per_mm is not None else base_k_t_n_m
    k_r_in = (
        float(ov.k_r_inplane_nm_rad) if ov.k_r_inplane_nm_rad is not None else base_k_r_in
    )
    k_r_tor = (
        float(ov.k_r_torsion_nm_rad) if ov.k_r_torsion_nm_rad is not None else base_k_r_tor
    )
    damping_delta = float(ov.damping_delta) if ov.damping_delta is not None else base_damping

    return {
        "k_t_n_m": k_t_n_m,
        "k_r_inplane_nm_rad": k_r_in,
        "k_r_torsion_nm_rad": k_r_tor,
        "damping_delta": damping_delta,
        "baseline_k_t_n_m": base_k_t_n_m,
        "baseline_k_r_inplane_nm_rad": base_k_r_in,
        "baseline_k_r_torsion_nm_rad": base_k_r_tor,
        "baseline_damping_delta": base_damping,
    }


def resolve_extrusion_filling(spec: PrinterSpec) -> dict[str, Any]:
    """Resolve extrusion cavity filling density, modulus (Pa), bonding, and damping boost."""
    from printsim.schema import printer as _p

    fill_id = spec.frame.extrusion_filling
    base_entry = _p.EXTRUSION_FILLINGS.get(fill_id, _p.EXTRUSION_FILLINGS["none"])
    base_rho = float(base_entry["density_kg_m3"])
    base_e_pa = float(base_entry["youngs_modulus_pa"])
    base_bonded = bool(base_entry["bonded"])
    base_damping = float(base_entry["damping_boost"])

    ov = spec.model_overrides.extrusion_filling
    rho = float(ov.density_kg_m3) if ov.density_kg_m3 is not None else base_rho
    e_pa = float(ov.youngs_modulus_gpa) * 1e9 if ov.youngs_modulus_gpa is not None else base_e_pa
    bonded = bool(ov.bonded) if ov.bonded is not None else base_bonded
    damping_boost = float(ov.damping_boost) if ov.damping_boost is not None else base_damping
    is_active = (fill_id != "none") or (rho > 0.0) or (e_pa > 0.0) or (damping_boost > 0.0)

    return {
        "density_kg_m3": rho,
        "youngs_modulus_pa": e_pa,
        "bonded": bonded,
        "damping_boost": damping_boost,
        "cost_per_kg_usd": float(base_entry["cost_per_kg_usd"]),
        "is_active": is_active,
        "baseline_density_kg_m3": base_rho,
        "baseline_youngs_modulus_pa": base_e_pa,
        "baseline_bonded": base_bonded,
        "baseline_damping_boost": base_damping,
    }


def resolve_frame_feet(spec: PrinterSpec) -> dict[str, Any]:
    """Resolve mount stiffness (SI), damping delta, and base paver mass for feet or suspension lines.

    For ``feet_type = "suspended_lines"`` the per-line axial stiffness is ``EA / L`` from
    ``SUSPENSION_LINE_CATALOG`` and the lateral stiffness is the pendulum geometric stiffness
    ``T / L``. The value returned here uses a 15 kg reference printer on 4 lines; the assembly
    builder replaces it with the exact ``(M_total g / n_lines) / L`` once the mass is known.
    """
    from printsim.schema import printer as _p

    feet_type = spec.frame.feet_type
    base_entry = _p.FRAME_FEET_CATALOG.get(feet_type, _p.FRAME_FEET_CATALOG["rigid_bolted"])
    base_k_axial_n_mm = float(base_entry["k_axial_n_per_mm"])
    base_k_lat_n_mm = float(base_entry["k_lateral_n_per_mm"])
    base_damping = float(base_entry["damping_delta"])
    base_mass_kg = float(base_entry.get("base_mass_kg", 0.0))

    is_suspended = feet_type == "suspended_lines"
    line_id = ""
    line_ea_n = 0.0
    line_length_m = 0.0
    line_break_n = 0.0
    line_loss_factor = 0.0
    line_design_factor = 5.0
    mount_damping_ratio = float(base_entry.get("mount_damping_ratio", 0.05))
    if is_suspended:
        line_id = str(spec.frame.suspension_line)
        line_entry = _p.SUSPENSION_LINE_CATALOG.get(
            line_id, _p.SUSPENSION_LINE_CATALOG["paracord_550"]
        )
        line_ea_n = float(line_entry["stiffness_ea_n"])
        line_break_n = float(line_entry["break_strength_n"])
        line_loss_factor = float(line_entry["loss_factor"])
        line_design_factor = float(line_entry.get("design_factor", 5.0))
        # Hysteretic loss factor eta -> equivalent viscous damping ratio eta / 2 at resonance
        mount_damping_ratio = max(0.5 * line_loss_factor, 0.005)
        line_length_m = max(float(spec.frame.suspension_length_mm), 50.0) * 1e-3
        base_k_axial_n_mm = line_ea_n / line_length_m * 1e-3
        base_k_lat_n_mm = (SUSPENSION_REFERENCE_MASS_KG * _G_ACC / 4.0) / line_length_m * 1e-3

    ov = spec.model_overrides.frame_feet
    k_axial_n_mm = (
        float(ov.k_axial_n_per_mm) if ov.k_axial_n_per_mm is not None else base_k_axial_n_mm
    )
    k_lat_n_mm = (
        float(ov.k_lateral_n_per_mm) if ov.k_lateral_n_per_mm is not None else base_k_lat_n_mm
    )
    damping_delta = float(ov.damping_delta) if ov.damping_delta is not None else base_damping
    mass_kg = float(ov.base_mass_kg) if ov.base_mass_kg is not None else base_mass_kg

    is_compliant = (
        (feet_type != "rigid_bolted")
        or (ov.k_axial_n_per_mm is not None and ov.k_axial_n_per_mm > 0.0)
        or (ov.k_lateral_n_per_mm is not None and ov.k_lateral_n_per_mm > 0.0)
    )

    return {
        "k_axial_n_m": k_axial_n_mm * 1e3,
        "k_lateral_n_m": k_lat_n_mm * 1e3,
        "damping_delta": damping_delta,
        "base_mass_kg": mass_kg,
        "is_compliant": is_compliant,
        "is_suspended": is_suspended,
        "lateral_from_tension": is_suspended and ov.k_lateral_n_per_mm is None,
        "suspension_line": line_id,
        "line_ea_n": line_ea_n,
        "line_length_m": line_length_m,
        "line_break_strength_n": line_break_n,
        "line_loss_factor": line_loss_factor,
        "line_design_factor": line_design_factor,
        "mount_damping_ratio": mount_damping_ratio,
        "baseline_k_axial_n_per_mm": base_k_axial_n_mm,
        "baseline_k_lateral_n_per_mm": base_k_lat_n_mm,
        "baseline_damping_delta": base_damping,
        "baseline_base_mass_kg": base_mass_kg,
    }


def resolve_belt(spec: PrinterSpec) -> dict[str, Any]:
    """Resolve XY drive belt EA (N), tooth shear modulus, linear mass (kg/m), and tension rating."""
    belt = spec.gantry.belt
    base_ea = float(belt.stiffness_ea_n)
    base_tooth_k = float(belt.tooth_shear_k_n_per_m_mm)
    base_mu_kg_m = float(belt.linear_mass_kg_m)
    base_max_t = float(belt.max_working_tension_n)

    ov = spec.model_overrides.belt
    ea_n = float(ov.stiffness_ea_n) if ov.stiffness_ea_n is not None else base_ea
    tooth_k = (
        float(ov.tooth_shear_k_n_per_m_mm)
        if ov.tooth_shear_k_n_per_m_mm is not None
        else base_tooth_k
    )
    mu_kg_m = (
        float(ov.linear_mass_g_per_m) * 1e-3
        if ov.linear_mass_g_per_m is not None
        else base_mu_kg_m
    )
    max_t = (
        float(ov.max_working_tension_n) if ov.max_working_tension_n is not None else base_max_t
    )

    return {
        "stiffness_ea_n": ea_n,
        "tooth_shear_k_n_per_m_mm": tooth_k,
        "linear_mass_kg_m": mu_kg_m,
        "max_working_tension_n": max_t,
        "baseline_stiffness_ea_n": base_ea,
        "baseline_tooth_shear_k_n_per_m_mm": base_tooth_k,
        "baseline_linear_mass_kg_m": base_mu_kg_m,
        "baseline_max_working_tension_n": base_max_t,
    }


def _material_bracket_stiffness_n_m(spec: PrinterSpec) -> float:
    from printsim.fea.components import compute_stepper_mount_temp_c
    from printsim.schema.materials import get_material

    bkt_mat = get_material(spec.gantry.bracket_material)
    motor_mount_temp_c, _ = compute_stepper_mount_temp_c(spec)
    e_bkt = bkt_mat.effective_modulus_pa(motor_mount_temp_c) * float(
        bkt_mat.bolted_joint_stiffness_factor
    )
    k_ref = 5.0e5
    ds_mult = 1.45 if spec.gantry.double_shear_pulleys else 1.0
    awd_mult = 1.65 if (spec.architecture == "corexy" and spec.gantry.drive_mode == "awd") else 1.0
    k_mat = k_ref * ds_mult * awd_mult * (e_bkt / 2.1e9)
    k_ceil = 8.0e6
    return (k_mat * k_ceil) / max(k_mat + k_ceil, 1e-9)


def _cad_gantry_bracket_stiffness_n_m(
    spec: PrinterSpec, kit_id: str
) -> tuple[float | None, dict[str, float]]:
    """Compute series-parallel CAD gantry bracket stiffness (N/m) from baked superelement stations."""
    from printsim.fea.components import compute_stepper_mount_temp_c
    from printsim.schema import printer as _p
    from printsim.schema.materials import get_material

    if spec.architecture != "corexy" or kit_id == "procedural":
        return None, {}

    kit_meta = _p.GANTRY_HARDWARE_KIT_CATALOG.get(kit_id)
    if not kit_meta:
        return None, {}
    stations = kit_meta.get("stations") or {}
    if not stations:
        return None, {}

    try:
        from printsim.cad_assets.registry import load_superelement
    except ImportError:
        return None, {}

    bkt_mat = get_material(spec.gantry.bracket_material)
    motor_mount_temp_c, _ = compute_stepper_mount_temp_c(spec)

    loaded_k: dict[str, float] = {}
    station_stiffs_n_mm: dict[str, float] = {}
    for st_key in ("rear_left", "front_left", "xy_left"):
        aid = stations.get(st_key)
        if not aid:
            continue
        se = load_superelement(
            "gantry_brackets",
            aid,
            material_id=spec.gantry.bracket_material,
            chamber_temp_c=motor_mount_temp_c,
        )
        if se is None:
            continue
        ps = se.manifest.get("port_stiffness_n_per_mm")
        if isinstance(ps, dict) and "kx" in ps and "ky" in ps:
            scale_k = float(getattr(se, "stiffness_scale_applied", 1.0))
            kx_n_m = max(float(ps["kx"]) * scale_k * 1e3, 1.0)
            ky_n_m = max(float(ps["ky"]) * scale_k * 1e3, 1.0)
        else:
            import numpy as np

            k_11 = 0.5 * (se.K_bb[6:12, 6:12] + se.K_bb[6:12, 6:12].T)
            try:
                c_11 = np.linalg.inv(k_11)
            except np.linalg.LinAlgError:
                c_11 = np.linalg.pinv(k_11, rcond=1e-10)
            kx_n_m = 1.0 / max(float(c_11[0, 0]), 1e-12)
            ky_n_m = 1.0 / max(float(c_11[1, 1]), 1e-12)
        k_planar = 2.0 / (1.0 / kx_n_m + 1.0 / ky_n_m)
        loaded_k[st_key] = k_planar
        station_stiffs_n_mm[st_key] = round(k_planar * 1e-3, 2)

    if not loaded_k:
        return None, {}

    is_awd = spec.gantry.drive_mode == "awd"
    k_rear = loaded_k.get("rear_left")
    k_front = loaded_k.get("front_left")
    k_xy = loaded_k.get("xy_left")

    # CoreXY symmetric L+R pairs:
    # - In AWD with a native AWD kit (front motor mounts), all 4 corners are driven motor mounts
    #   acting in parallel (2*k_rear + 2*k_front), in series with the L+R moving XY-joint brackets (2*k_xy).
    # - In AWD on a 2WD-only kit (compact front idlers), do NOT plug tiny 2WD front idlers in as
    #   front motor mounts; model 4 symmetric motor mounts (4*k_rear) in series with 2*k_xy.
    # - In 2WD, rear motor mounts, free-spinning front idlers, and XY-joints carry the belt loop
    #   in series (so a stiff compact front idler cannot short-circuit a flexible rear motor mount).
    kit_has_awd_front_mounts = kit_id in ("monolith_ldo_cnc_awd", "siboor_cnc_awd", "vzbot_330_awd")
    k_corners: float | None = None
    if is_awd:
        if kit_has_awd_front_mounts and k_rear is not None and k_front is not None:
            k_corners = 2.0 * k_rear + 2.0 * k_front
        elif k_rear is not None:
            k_corners = 4.0 * k_rear
        elif k_front is not None:
            k_corners = 4.0 * k_front
    else:
        if k_rear is not None and k_front is not None:
            k_corners = 1.0 / (1.0 / max(2.0 * k_rear, 1e-9) + 1.0 / max(2.0 * k_front, 1e-9))
        elif k_rear is not None:
            k_corners = 2.0 * k_rear
        elif k_front is not None:
            k_corners = 2.0 * k_front

    if k_corners is not None and k_xy is not None:
        k_cad_raw = 1.0 / (1.0 / max(k_corners, 1e-9) + 1.0 / max(2.0 * k_xy, 1e-9))
    elif k_xy is not None:
        k_cad_raw = 2.0 * k_xy
    else:
        assert k_corners is not None
        k_cad_raw = k_corners

    bolt_factor = float(bkt_mat.bolted_joint_stiffness_factor)
    cad_is_double_shear = kit_id in (
        "monolith_ldo_cnc_awd",
        "monolith_ldo_cnc_2wd",
        "siboor_cnc_awd",
        "siboor_cnc_2wd",
        "vzbot_330_awd",
        "vzbot_330_2wd",
    )
    ds_mult = 1.45 if (spec.gantry.double_shear_pulleys and not cad_is_double_shear) else 1.0
    k_cad_scaled = k_cad_raw * bolt_factor * ds_mult
    k_ceil = (12.0e6 if is_awd else 8.0e6) * bolt_factor
    k_eff = (k_cad_scaled * k_ceil) / max(k_cad_scaled + k_ceil, 1e-9)
    return float(k_eff), station_stiffs_n_mm


def resolve_gantry_bracket(spec: PrinterSpec) -> dict[str, Any]:
    """Resolve XY gantry bracket stiffness (N/m), carriage mass delta, creep limit, and damping."""
    from printsim.schema import printer as _p

    bkt_info = _p.GANTRY_BRACKET_CATALOG.get(
        spec.gantry.bracket_material, _p.GANTRY_BRACKET_CATALOG["abs_printed"]
    )
    cbo = spec.gantry.custom_bracket_override
    ov = spec.model_overrides.gantry_bracket
    kit_id = spec.resolve_effective_gantry_hardware_kit()
    cad_backed = False
    cad_station_stiffs: dict[str, float] = {}

    if cbo is not None:
        base_stiff_n_m = float(cbo.stiffness_n_per_mm) * 1e3
    else:
        base_stiff_n_m = _material_bracket_stiffness_n_m(spec)
        if bool(getattr(spec, "use_cad_superelements", True)) and ov.stiffness_n_per_mm is None:
            cad_k, cad_station_stiffs = _cad_gantry_bracket_stiffness_n_m(spec, kit_id)
            if cad_k is not None:
                base_stiff_n_m = cad_k
                cad_backed = True

    base_mass_delta_g = float(bkt_info["carriage_mass_delta_g"])
    base_creep_n = float(bkt_info["creep_tension_limit_n"])
    if bkt_info.get("manufacturing_process") == "slm":
        # As-sintered rough bolted interfaces & cellular microstructure damp slightly more than CNC billet
        base_damping = -0.0015
    else:
        base_damping = -0.003 if bool(bkt_info["cnc_machined"]) else 0.004

    stiff_n_m = (
        float(ov.stiffness_n_per_mm) * 1e3
        if ov.stiffness_n_per_mm is not None
        else base_stiff_n_m
    )
    mass_delta_g = (
        float(ov.carriage_mass_delta_g)
        if ov.carriage_mass_delta_g is not None
        else base_mass_delta_g
    )
    creep_n = (
        float(ov.creep_tension_limit_n)
        if ov.creep_tension_limit_n is not None
        else base_creep_n
    )
    damping_delta = float(ov.damping_delta) if ov.damping_delta is not None else base_damping

    return {
        "stiffness_n_m": stiff_n_m,
        "carriage_mass_delta_g": mass_delta_g,
        "creep_tension_limit_n": creep_n,
        "damping_delta": damping_delta,
        "baseline_stiffness_n_m": base_stiff_n_m,
        "baseline_carriage_mass_delta_g": base_mass_delta_g,
        "baseline_creep_tension_limit_n": base_creep_n,
        "baseline_damping_delta": base_damping,
        "cad_backed": cad_backed,
        "cad_kit_id": kit_id,
        "cad_station_stiffnesses_n_per_mm": cad_station_stiffs,
    }


def resolve_rail_preload(spec: PrinterSpec) -> dict[str, Any]:
    """Resolve linear rail Hertzian ball preload fraction (Q / C)."""
    from printsim.schema import printer as _p

    pre_key = spec.gantry.rail_preload.upper()
    base_frac = float(_p.RAIL_PRELOAD_FRACTIONS.get(pre_key, 0.030))
    ov = spec.model_overrides.rail_preload
    eff_frac = float(ov.preload_fraction) if ov.preload_fraction is not None else base_frac
    return {
        "preload_fraction": eff_frac,
        "baseline_preload_fraction": base_frac,
    }


def resolve_rail_quality(spec: PrinterSpec) -> dict[str, Any]:
    """Resolve linear rail grinding/sorting stiffness and drag multipliers."""
    from printsim.schema import printer as _p

    qual_key = str(spec.gantry.rail_quality).lower().strip()
    qual_cfg = _p.RAIL_QUALITY_CATALOG.get(qual_key, _p.RAIL_QUALITY_CATALOG["hiwin_genuine"])
    base_stiff = float(qual_cfg["stiffness_mult"])
    base_drag = float(qual_cfg["drag_mult"])

    ov = spec.model_overrides.rail_quality
    eff_stiff = float(ov.stiffness_mult) if ov.stiffness_mult is not None else base_stiff
    eff_drag = float(ov.drag_mult) if ov.drag_mult is not None else base_drag
    return {
        "stiffness_mult": eff_stiff,
        "drag_mult": eff_drag,
        "baseline_stiffness_mult": base_stiff,
        "baseline_drag_mult": base_drag,
    }


def resolve_belt_routing(spec: PrinterSpec) -> dict[str, Any]:
    """Resolve CoreXY belt routing free-span and idler stiffness factors."""
    from printsim.schema import printer as _p

    cfg = _p.BELT_ROUTING_CATALOG.get(
        spec.gantry.belt_routing, _p.BELT_ROUTING_CATALOG["stock_corexy"]
    )
    base_span = float(cfg["span_factor"])
    base_idler = float(cfg["idler_stiffness_factor"])

    ov = spec.model_overrides.belt_routing
    eff_span = float(ov.span_factor) if ov.span_factor is not None else base_span
    eff_idler = (
        float(ov.idler_stiffness_factor) if ov.idler_stiffness_factor is not None else base_idler
    )
    return {
        "span_factor": eff_span,
        "idler_stiffness_factor": eff_idler,
        "baseline_span_factor": base_span,
        "baseline_idler_stiffness_factor": base_idler,
    }


def resolve_bearing_clearance(spec: PrinterSpec) -> dict[str, Any]:
    """Resolve ISO 5753 idler bearing radial play (um)."""
    from printsim.schema import printer as _p

    base_play = float(
        _p.BEARING_RADIAL_PLAY_UM.get(spec.gantry.belt.bearing_clearance_class, 7.5)
    )
    ov = spec.model_overrides.bearing_clearance
    eff_play = float(ov.radial_play_um) if ov.radial_play_um is not None else base_play
    return {
        "radial_play_um": eff_play,
        "baseline_radial_play_um": base_play,
    }


def resolve_z_probe(spec: PrinterSpec) -> dict[str, Any]:
    """Resolve Z-probe carriage compliance knockdown factor and forward CoG Y shift (mm)."""
    from printsim.schema import printer as _p

    probe_cfg = _p.Z_PROBE_CATALOG.get(
        spec.toolhead.z_probe_type, _p.Z_PROBE_CATALOG["static_eddy"]
    )
    base_comp = float(probe_cfg["compliance_factor"])
    base_shift = float(probe_cfg["cog_y_shift_mm"])

    ov = spec.model_overrides.z_probe
    eff_comp = float(ov.compliance_factor) if ov.compliance_factor is not None else base_comp
    eff_shift = float(ov.cog_y_shift_mm) if ov.cog_y_shift_mm is not None else base_shift
    return {
        "compliance_factor": eff_comp,
        "cog_y_shift_mm": eff_shift,
        "baseline_compliance_factor": base_comp,
        "baseline_cog_y_shift_mm": base_shift,
    }


def resolve_toolhead_construction(spec: PrinterSpec) -> dict[str, Any]:
    """Resolve Stage 2 toolhead chassis base stiffness (N/mm) and torsion multiplier."""
    from printsim.schema import printer as _p

    bc = spec.toolhead.body_construction
    if bc is not None and bc in _p.TOOLHEAD_CONSTRUCTION_CATALOG:
        cfg = _p.TOOLHEAD_CONSTRUCTION_CATALOG[bc]
        base_stiff = float(cfg["body_stiffness_base_n_per_mm"])
        base_tor = float(cfg["torsion_mult"])
    else:
        base_stiff = float(spec.toolhead.internal_stiffness_n_per_mm)
        base_tor = 1.0

    ov = spec.model_overrides.toolhead_construction
    eff_stiff = (
        float(ov.body_stiffness_base_n_per_mm)
        if ov.body_stiffness_base_n_per_mm is not None
        else base_stiff
    )
    eff_tor = float(ov.torsion_mult) if ov.torsion_mult is not None else base_tor
    return {
        "body_stiffness_base_n_per_mm": eff_stiff,
        "torsion_mult": eff_tor,
        "baseline_body_stiffness_base_n_per_mm": base_stiff,
        "baseline_torsion_mult": base_tor,
    }


def resolve_hotend_clamp(spec: PrinterSpec) -> dict[str, Any]:
    """Resolve Stage 3 hotend mounting clamp base stiffness (N/mm)."""
    from printsim.schema import printer as _p

    mount_style = spec.toolhead.hotend.mount_style
    base_stiff = float(_p.HOTEND_MOUNT_CLAMP_BASE_N_PER_MM.get(mount_style, 3200.0))

    ov = spec.model_overrides.hotend_clamp
    eff_stiff = (
        float(ov.clamp_stiffness_base_n_per_mm)
        if ov.clamp_stiffness_base_n_per_mm is not None
        else base_stiff
    )
    return {
        "clamp_stiffness_base_n_per_mm": eff_stiff,
        "baseline_clamp_stiffness_base_n_per_mm": base_stiff,
    }


def resolve_bed_mount(spec: PrinterSpec) -> dict[str, Any]:
    """Resolve bed mount axial/lateral/thermal stiffnesses, damping, standoff height, and topology."""
    from printsim.schema import printer as _p

    cfg = _p.BED_MOUNT_CATALOG.get(spec.bed.mount_type, _p.BED_MOUNT_CATALOG["kinematic_3pt"])
    base_k_ax = float(cfg["k_axial_n_per_mm"])
    base_k_lat = float(cfg["k_lateral_n_per_mm"])
    base_k_therm_n_m = float(
        cfg.get(
            "k_restraint_thermal_n_per_m",
            float(cfg.get("k_restraint_thermal", 0.0)) * 1e3,
        )
    )
    base_k_therm_n_mm = base_k_therm_n_m * 1e-3
    base_damping = float(cfg["damping_delta"])
    base_standoff = float(cfg.get("standoff_height_mm", 14.0))

    ov = spec.model_overrides.bed_mount
    eff_k_ax = float(ov.k_axial_n_per_mm) if ov.k_axial_n_per_mm is not None else base_k_ax
    eff_k_lat = float(ov.k_lateral_n_per_mm) if ov.k_lateral_n_per_mm is not None else base_k_lat
    if ov.k_restraint_thermal_n_per_mm is not None:
        eff_k_therm_n_mm = float(ov.k_restraint_thermal_n_per_mm)
        eff_k_therm_n_m = eff_k_therm_n_mm * 1e3
    else:
        eff_k_therm_n_mm = base_k_therm_n_mm
        eff_k_therm_n_m = base_k_therm_n_m
    eff_damping = float(ov.damping_delta) if ov.damping_delta is not None else base_damping
    eff_standoff = float(ov.standoff_height_mm) if ov.standoff_height_mm is not None else base_standoff

    return {
        "mount_points": int(cfg["mount_points"]),
        "k_axial_n_per_mm": eff_k_ax,
        "k_lateral_n_per_mm": eff_k_lat,
        "k_restraint_thermal_n_per_m": eff_k_therm_n_m,
        "k_restraint_thermal_n_per_mm": eff_k_therm_n_mm,
        "damping_delta": eff_damping,
        "standoff_height_mm": eff_standoff,
        "radial_dof_mode": str(cfg.get("radial_dof_mode", "maxwell_vgroove")),
        "tap_rated": bool(cfg.get("tap_rated", False)),
        "mount_inset_ratio": float(cfg.get("mount_inset_ratio", 0.38)),
        "baseline_k_axial_n_per_mm": base_k_ax,
        "baseline_k_lateral_n_per_mm": base_k_lat,
        "baseline_k_restraint_thermal_n_per_mm": base_k_therm_n_mm,
        "baseline_damping_delta": base_damping,
        "baseline_standoff_height_mm": base_standoff,
    }


def resolve_z_drive_mechanism(spec: PrinterSpec) -> dict[str, Any]:
    """Resolve Z-axis drive mechanism stiffnesses (N/um), root area, and backlash (um)."""
    from printsim.schema import printer as _p

    mech_id = spec.bed.effective_z_leadscrew_type
    z_mech = _p.Z_DRIVE_MECHANISM_CATALOG.get(mech_id, _p.Z_DRIVE_MECHANISM_CATALOG["tr8x4_pom"])

    base_k_nut = float(z_mech["k_nut_n_per_um"])
    base_k_thrust = float(z_mech["k_thrust_n_per_um"])
    raw_k_drive = z_mech.get("k_drive_n_per_um_per_corner")
    base_k_drive = float(raw_k_drive) if raw_k_drive is not None else None
    base_backlash = float(z_mech["backlash_um"])

    ov = spec.model_overrides.z_drive_mechanism
    eff_k_nut = float(ov.k_nut_n_per_um) if ov.k_nut_n_per_um is not None else base_k_nut
    eff_k_thrust = (
        float(ov.k_thrust_n_per_um) if ov.k_thrust_n_per_um is not None else base_k_thrust
    )
    eff_k_drive = (
        float(ov.k_drive_n_per_um_per_corner)
        if ov.k_drive_n_per_um_per_corner is not None
        else base_k_drive
    )
    eff_backlash = float(ov.backlash_um) if ov.backlash_um is not None else base_backlash

    return {
        "id": str(z_mech["id"]),
        "kind": str(z_mech["kind"]),
        "root_area_m2": float(z_mech["root_area_m2"]),
        "diameter_mm": float(z_mech["diameter_mm"]),
        "k_nut_n_per_um": eff_k_nut,
        "k_thrust_n_per_um": eff_k_thrust,
        "k_drive_n_per_um_per_corner": eff_k_drive,
        "backlash_um": eff_backlash,
        "baseline_k_nut_n_per_um": base_k_nut,
        "baseline_k_thrust_n_per_um": base_k_thrust,
        "baseline_k_drive_n_per_um_per_corner": base_k_drive,
        "baseline_backlash_um": base_backlash,
    }


def _overridden_fields(sub_model: BaseModel) -> list[str]:
    return [
        field_name
        for field_name in type(sub_model).model_fields
        if getattr(sub_model, field_name) is not None
    ]


def build_effective_model_params(spec: PrinterSpec) -> dict[str, dict[str, Any]]:
    """Build complete baseline vs effective lumped-model inspection payload for all 14 groups."""
    mo = spec.model_overrides

    cj = resolve_corner_joint(spec)
    ef = resolve_extrusion_filling(spec)
    ff = resolve_frame_feet(spec)
    bl = resolve_belt(spec)
    gb = resolve_gantry_bracket(spec)
    rp = resolve_rail_preload(spec)
    rq = resolve_rail_quality(spec)
    br = resolve_belt_routing(spec)
    bc = resolve_bearing_clearance(spec)
    zp = resolve_z_probe(spec)
    tc = resolve_toolhead_construction(spec)
    hc = resolve_hotend_clamp(spec)
    bm = resolve_bed_mount(spec)
    zd = resolve_z_drive_mechanism(spec)

    return {
        "corner_joint": {
            "baseline": {
                "k_t_n_per_mm": round(float(cj["baseline_k_t_n_m"]) * 1e-3, 2),
                "k_r_inplane_nm_rad": round(float(cj["baseline_k_r_inplane_nm_rad"]), 2),
                "k_r_torsion_nm_rad": round(float(cj["baseline_k_r_torsion_nm_rad"]), 2),
                "damping_delta": round(float(cj["baseline_damping_delta"]), 6),
            },
            "effective": {
                "k_t_n_per_mm": round(float(cj["k_t_n_m"]) * 1e-3, 2),
                "k_r_inplane_nm_rad": round(float(cj["k_r_inplane_nm_rad"]), 2),
                "k_r_torsion_nm_rad": round(float(cj["k_r_torsion_nm_rad"]), 2),
                "damping_delta": round(float(cj["damping_delta"]), 6),
            },
            "overridden_fields": _overridden_fields(mo.corner_joint),
        },
        "extrusion_filling": {
            "baseline": {
                "density_kg_m3": round(float(ef["baseline_density_kg_m3"]), 2),
                "youngs_modulus_gpa": round(float(ef["baseline_youngs_modulus_pa"]) * 1e-9, 4),
                "bonded": bool(ef["baseline_bonded"]),
                "damping_boost": round(float(ef["baseline_damping_boost"]), 6),
            },
            "effective": {
                "density_kg_m3": round(float(ef["density_kg_m3"]), 2),
                "youngs_modulus_gpa": round(float(ef["youngs_modulus_pa"]) * 1e-9, 4),
                "bonded": bool(ef["bonded"]),
                "damping_boost": round(float(ef["damping_boost"]), 6),
            },
            "overridden_fields": _overridden_fields(mo.extrusion_filling),
        },
        "frame_feet": {
            "baseline": {
                "k_axial_n_per_mm": round(float(ff["baseline_k_axial_n_per_mm"]), 2),
                "k_lateral_n_per_mm": round(float(ff["baseline_k_lateral_n_per_mm"]), 2),
                "damping_delta": round(float(ff["baseline_damping_delta"]), 6),
                "base_mass_kg": round(float(ff["baseline_base_mass_kg"]), 3),
            },
            "effective": {
                "k_axial_n_per_mm": round(float(ff["k_axial_n_m"]) * 1e-3, 2),
                "k_lateral_n_per_mm": round(float(ff["k_lateral_n_m"]) * 1e-3, 2),
                "damping_delta": round(float(ff["damping_delta"]), 6),
                "base_mass_kg": round(float(ff["base_mass_kg"]), 3),
            },
            "overridden_fields": _overridden_fields(mo.frame_feet),
        },
        "belt": {
            "baseline": {
                "stiffness_ea_n": round(float(bl["baseline_stiffness_ea_n"]), 2),
                "tooth_shear_k_n_per_m_mm": round(
                    float(bl["baseline_tooth_shear_k_n_per_m_mm"]), 2
                ),
                "linear_mass_g_per_m": round(float(bl["baseline_linear_mass_kg_m"]) * 1e3, 3),
                "max_working_tension_n": round(float(bl["baseline_max_working_tension_n"]), 2),
            },
            "effective": {
                "stiffness_ea_n": round(float(bl["stiffness_ea_n"]), 2),
                "tooth_shear_k_n_per_m_mm": round(float(bl["tooth_shear_k_n_per_m_mm"]), 2),
                "linear_mass_g_per_m": round(float(bl["linear_mass_kg_m"]) * 1e3, 3),
                "max_working_tension_n": round(float(bl["max_working_tension_n"]), 2),
            },
            "overridden_fields": _overridden_fields(mo.belt),
        },
        "gantry_bracket": {
            "baseline": {
                "stiffness_n_per_mm": round(float(gb["baseline_stiffness_n_m"]) * 1e-3, 2),
                "carriage_mass_delta_g": round(float(gb["baseline_carriage_mass_delta_g"]), 2),
                "creep_tension_limit_n": round(float(gb["baseline_creep_tension_limit_n"]), 2),
                "damping_delta": round(float(gb["baseline_damping_delta"]), 6),
            },
            "effective": {
                "stiffness_n_per_mm": round(float(gb["stiffness_n_m"]) * 1e-3, 2),
                "carriage_mass_delta_g": round(float(gb["carriage_mass_delta_g"]), 2),
                "creep_tension_limit_n": round(float(gb["creep_tension_limit_n"]), 2),
                "damping_delta": round(float(gb["damping_delta"]), 6),
            },
            "overridden_fields": _overridden_fields(mo.gantry_bracket),
        },
        "rail_preload": {
            "baseline": {
                "preload_fraction": round(float(rp["baseline_preload_fraction"]), 6),
            },
            "effective": {
                "preload_fraction": round(float(rp["preload_fraction"]), 6),
            },
            "overridden_fields": _overridden_fields(mo.rail_preload),
        },
        "rail_quality": {
            "baseline": {
                "stiffness_mult": round(float(rq["baseline_stiffness_mult"]), 4),
                "drag_mult": round(float(rq["baseline_drag_mult"]), 4),
            },
            "effective": {
                "stiffness_mult": round(float(rq["stiffness_mult"]), 4),
                "drag_mult": round(float(rq["drag_mult"]), 4),
            },
            "overridden_fields": _overridden_fields(mo.rail_quality),
        },
        "belt_routing": {
            "baseline": {
                "span_factor": round(float(br["baseline_span_factor"]), 4),
                "idler_stiffness_factor": round(
                    float(br["baseline_idler_stiffness_factor"]), 4
                ),
            },
            "effective": {
                "span_factor": round(float(br["span_factor"]), 4),
                "idler_stiffness_factor": round(float(br["idler_stiffness_factor"]), 4),
            },
            "overridden_fields": _overridden_fields(mo.belt_routing),
        },
        "bearing_clearance": {
            "baseline": {
                "radial_play_um": round(float(bc["baseline_radial_play_um"]), 3),
            },
            "effective": {
                "radial_play_um": round(float(bc["radial_play_um"]), 3),
            },
            "overridden_fields": _overridden_fields(mo.bearing_clearance),
        },
        "z_probe": {
            "baseline": {
                "compliance_factor": round(float(zp["baseline_compliance_factor"]), 4),
                "cog_y_shift_mm": round(float(zp["baseline_cog_y_shift_mm"]), 3),
            },
            "effective": {
                "compliance_factor": round(float(zp["compliance_factor"]), 4),
                "cog_y_shift_mm": round(float(zp["cog_y_shift_mm"]), 3),
            },
            "overridden_fields": _overridden_fields(mo.z_probe),
        },
        "toolhead_construction": {
            "baseline": {
                "body_stiffness_base_n_per_mm": round(
                    float(tc["baseline_body_stiffness_base_n_per_mm"]), 2
                ),
                "torsion_mult": round(float(tc["baseline_torsion_mult"]), 4),
            },
            "effective": {
                "body_stiffness_base_n_per_mm": round(
                    float(tc["body_stiffness_base_n_per_mm"]), 2
                ),
                "torsion_mult": round(float(tc["torsion_mult"]), 4),
            },
            "overridden_fields": _overridden_fields(mo.toolhead_construction),
        },
        "hotend_clamp": {
            "baseline": {
                "clamp_stiffness_base_n_per_mm": round(
                    float(hc["baseline_clamp_stiffness_base_n_per_mm"]), 2
                ),
            },
            "effective": {
                "clamp_stiffness_base_n_per_mm": round(
                    float(hc["clamp_stiffness_base_n_per_mm"]), 2
                ),
            },
            "overridden_fields": _overridden_fields(mo.hotend_clamp),
        },
        "bed_mount": {
            "baseline": {
                "k_axial_n_per_mm": round(float(bm["baseline_k_axial_n_per_mm"]), 2),
                "k_lateral_n_per_mm": round(float(bm["baseline_k_lateral_n_per_mm"]), 2),
                "k_restraint_thermal_n_per_mm": round(
                    float(bm["baseline_k_restraint_thermal_n_per_mm"]), 2
                ),
                "damping_delta": round(float(bm["baseline_damping_delta"]), 6),
                "standoff_height_mm": round(float(bm["baseline_standoff_height_mm"]), 2),
            },
            "effective": {
                "k_axial_n_per_mm": round(float(bm["k_axial_n_per_mm"]), 2),
                "k_lateral_n_per_mm": round(float(bm["k_lateral_n_per_mm"]), 2),
                "k_restraint_thermal_n_per_mm": round(
                    float(bm["k_restraint_thermal_n_per_mm"]), 2
                ),
                "damping_delta": round(float(bm["damping_delta"]), 6),
                "standoff_height_mm": round(float(bm["standoff_height_mm"]), 2),
            },
            "overridden_fields": _overridden_fields(mo.bed_mount),
        },
        "z_drive_mechanism": {
            "baseline": {
                "k_nut_n_per_um": round(float(zd["baseline_k_nut_n_per_um"]), 3),
                "k_thrust_n_per_um": round(float(zd["baseline_k_thrust_n_per_um"]), 3),
                "k_drive_n_per_um_per_corner": (
                    round(float(zd["baseline_k_drive_n_per_um_per_corner"]), 4)
                    if zd["baseline_k_drive_n_per_um_per_corner"] is not None
                    else None
                ),
                "backlash_um": round(float(zd["baseline_backlash_um"]), 2),
            },
            "effective": {
                "k_nut_n_per_um": round(float(zd["k_nut_n_per_um"]), 3),
                "k_thrust_n_per_um": round(float(zd["k_thrust_n_per_um"]), 3),
                "k_drive_n_per_um_per_corner": (
                    round(float(zd["k_drive_n_per_um_per_corner"]), 4)
                    if zd["k_drive_n_per_um_per_corner"] is not None
                    else None
                ),
                "backlash_um": round(float(zd["backlash_um"]), 2),
            },
            "overridden_fields": _overridden_fields(mo.z_drive_mechanism),
        },
    }


_SUBSYSTEM_LABELS: dict[str, str] = {
    "frame": "Frame",
    "gantry": "Gantry",
    "toolhead": "Toolhead",
    "hotend": "Hotend",
    "bed": "Bed / Z",
    "solver": "Solver & Shaper",
}

_CORNER_JOINT_LABELS: dict[str, str] = {
    "l_bracket": "Internal / Stamped L-Bracket",
    "blind_joint": "Blind Joint (Voron Standard)",
    "gusset_plate": "External CNC Gusset Plate",
    "corner_cube": "3-Way Cast / CNC Corner Cube",
    "rigid": "Idealized Rigid Welded Corner",
}

_RAIL_PRELOAD_LABELS: dict[str, str] = {
    "ZF": "ZF (Slight Clearance)",
    "Z0": "Z0 (Zero Clearance / Light)",
    "Z1": "Z1 (Light Preload)",
    "ZA": "ZA (Medium Preload)",
    "ZB": "ZB (Heavy Preload)",
}

_RAIL_BLOCK_LABELS: dict[str, str] = {
    "C": "C-Block (Standard)",
    "H": "H-Block (Long)",
}

_BEARING_CLEARANCE_LABELS: dict[str, str] = {
    "C2": "ISO 5753 Class C2 (Tight)",
    "CN": "ISO 5753 Class CN (Normal)",
    "C3": "ISO 5753 Class C3 (Loose)",
}

_BELT_CORE_LABELS: dict[str, str] = {
    "fiberglass": "Fiberglass",
    "steel": "Steel Cord",
    "aramid": "Aramid",
}

_BRACKET_SOURCE_LABELS: dict[str, str] = {
    "cms_superelement": "CMS Superelement",
    "tet10_fea": "Tet10 FEA",
    "hex8_fea": "Hex8 FEA",
    "custom_cad": "Custom CAD",
}

_DRIVE_TYPE_LABELS: dict[str, str] = {
    "open_loop_stepper": "Open-Loop Stepper",
    "closed_loop_stepper": "Closed-Loop Stepper",
    "foc_servo": "FOC AC/BLDC Servo",
}

_HOTEND_MOUNT_STYLE_LABELS: dict[str, str] = {
    "groove_mount": "J-Head Groove-Mount Collar",
    "rigid_4bolt_m3": "4-Bolt M3 Rigid Flange",
    "dragon_4bolt_m25": "4-Bolt M2.5 Rigid Pattern",
    "chube_6bolt_m3_18pcd": "6-Bolt M3 18mm PCD Monocoque Flange",
}

_Z_DRIVE_LABELS: dict[str, str] = {
    "belted_quad_z": "Belted Quad-Z (Flying Gantry)",
    "leadscrew_triple": "Triple Leadscrew Z",
    "leadscrew_dual_belt_sync": "Dual Belt-Synced Leadscrew Z",
    "leadscrew_dual": "Dual Leadscrew Z",
    "leadscrew_single": "Single Leadscrew Z",
}

_INPUT_SHAPER_LABELS: dict[str, str] = {
    "auto": "Auto (Klipper Optimal Score)",
    "none": "None (Unshaped)",
    "zv": "ZV (Zero Vibration)",
    "mzv": "MZV (Modified Zero Vibration)",
    "ei": "EI (Extra Insensitive)",
    "2hump_ei": "2-Hump EI (Two-Hump EI)",
    "3hump_ei": "3-Hump EI (Three-Hump EI)",
    "zvdd": "ZVDD (Zero Vibration Double-Derivative)",
    "smooth_zv": "Smooth ZV (Kalico)",
    "smooth_ei": "Smooth EI (Kalico)",
    "smooth_flatness": "Smooth Flatness (Kalico)",
}

_SUPERSCRIPT_TRANS = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
_SNAKE_ID_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$")


def _humanize_label(raw: str) -> str:
    s = str(raw).strip()
    if _SNAKE_ID_RE.match(s):
        return s.replace("_", " ").title()
    return s


def _fmt_cm4(val_cm4: float) -> str:
    av = abs(val_cm4)
    if av >= 100.0:
        return f"{val_cm4:.1f}"
    if av >= 10.0:
        return f"{val_cm4:.2f}".rstrip("0").rstrip(".")
    if av >= 1.0:
        return f"{val_cm4:.2f}"
    if av > 0.0:
        return f"{val_cm4:.3f}".rstrip("0").rstrip(".")
    return "0"


def _fmt_num_compact(val: float) -> str:
    av = abs(val)
    if av >= 1.0e6:
        exp = int(math.floor(math.log10(av)))
        mant = val / (10.0**exp)
        exp_str = str(exp).translate(_SUPERSCRIPT_TRANS)
        mant_str = f"{mant:.1f}".rstrip("0").rstrip(".")
        return f"{mant_str}×10{exp_str}"
    if av >= 100.0:
        return f"{val:,.0f}"
    if av >= 10.0:
        return f"{val:.1f}"
    if av >= 1.0:
        return f"{val:.2f}".rstrip("0").rstrip(".")
    if av > 0.0:
        return f"{val:.3f}".rstrip("0").rstrip(".")
    return "0"


def _get_cad_asset_manifest(category: str, asset_id: str | None) -> dict[str, Any] | None:
    if not asset_id:
        return None
    try:
        from printsim.cad_assets.registry import get_hardware_asset_manifest, has_hardware_asset
    except ImportError:
        return None
    if not has_hardware_asset(category, asset_id):
        return None
    return get_hardware_asset_manifest(category, asset_id)


def _make_option_breakdown_item(
    *,
    opt_id: str,
    label: str,
    category: str,
    active: bool,
    summary_values: str,
    cad_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    conf = get_item_confidence(category, opt_id)
    base_trust = max(0, min(100, int(round(conf["trust_pct"]))))
    cad_active = cad_manifest is not None
    cad_trust = (
        max(0, min(100, int(round(float(cad_manifest.get("cad_trust_pct", base_trust))))))
        if cad_manifest is not None
        else None
    )
    trust_pct = max(base_trust, cad_trust) if cad_trust is not None else base_trust
    basis = conf["basis"]
    if cad_manifest is not None:
        cad_name = _humanize_label(str(cad_manifest.get("name") or opt_id))
        basis = f"3D CAD Craig-Bampton superelement ({cad_name}) condensed from solid STEP/Tet mesh. {basis}"
    return {
        "id": opt_id,
        "label": _humanize_label(label),
        "trust_pct": trust_pct,
        "tier": confidence_tier(trust_pct),
        "active": bool(active),
        "summary_values": summary_values,
        "basis": basis,
        "caveat": conf["caveat"],
    }


def _has_group_override(spec: PrinterSpec, group_name: str | None) -> bool:
    if group_name is None:
        return False
    sub = getattr(spec.model_overrides, group_name, None)
    if sub is not None and bool(_overridden_fields(sub)):
        return True
    if group_name == "gantry_bracket" and spec.gantry.custom_bracket_override is not None:
        return True
    return False


def _make_confidence_entry(
    spec: PrinterSpec,
    *,
    entry_id: str,
    subsystem: Literal["frame", "gantry", "toolhead", "hotend", "bed", "solver"],
    name: str,
    active_selection: str,
    active_values: str,
    trust_pct: int,
    override_group: str | None,
    basis: str,
    caveat: str,
    options_breakdown: list[dict[str, Any]] | None = None,
    cad_superelement_active: bool = False,
    cad_superelement_id: str | None = None,
    cad_trust_pct: int | None = None,
) -> dict[str, Any]:
    clamped_trust = max(0, min(100, int(round(trust_pct))))
    clamped_cad = (
        max(0, min(100, int(round(cad_trust_pct))))
        if cad_trust_pct is not None
        else None
    )
    effective_trust = (
        max(clamped_trust, clamped_cad)
        if (cad_superelement_active and clamped_cad is not None)
        else clamped_trust
    )
    return {
        "id": entry_id,
        "subsystem": subsystem,
        "subsystem_label": _SUBSYSTEM_LABELS[subsystem],
        "name": name,
        "active_selection": _humanize_label(active_selection),
        "active_values": active_values,
        "trust_pct": effective_trust,
        "tier": confidence_tier(effective_trust),
        "override_group": override_group,
        "overridden": _has_group_override(spec, override_group),
        "basis": basis,
        "caveat": caveat,
        "options_breakdown": list(options_breakdown) if options_breakdown is not None else [],
        "cad_superelement_active": bool(cad_superelement_active),
        "cad_superelement_id": cad_superelement_id if cad_superelement_active else None,
        "cad_trust_pct": clamped_cad,
        "effective_trust_pct": effective_trust,
    }


def _fmt_corner_joint_values(
    k_t_n_m: float, k_r_in: float, k_r_tor: float, damping_delta: float
) -> str:
    return (
        f"k_t = {_fmt_num_compact(k_t_n_m * 1e-3)} N/mm, "
        f"k_r = {_fmt_num_compact(k_r_in)} N·m/rad, "
        f"k_tor = {_fmt_num_compact(k_r_tor)} N·m/rad, "
        f"Δζ = {damping_delta:.3f}"
    )


def _fmt_extrusion_filling_values(
    density_kg_m3: float, youngs_modulus_pa: float, bonded: bool, damping_boost: float
) -> str:
    return (
        f"ρ = {density_kg_m3:,.0f} kg/m³, "
        f"E = {youngs_modulus_pa * 1e-9:.2f} GPa, "
        f"{'Bonded' if bonded else 'Unbonded'}, "
        f"Δζ = {damping_boost:.3f}"
    )


def _fmt_frame_feet_values(
    *,
    is_rigid: bool,
    k_axial_n_mm: float,
    k_lat_n_mm: float,
    damping_delta: float,
    base_mass_kg: float,
) -> str:
    if is_rigid:
        return (
            f"k_axial = Fixed BC, k_lat = Fixed BC, "
            f"Δζ = {damping_delta:.3f}, Base = {base_mass_kg:.1f} kg"
        )
    return (
        f"k_axial = {_fmt_num_compact(k_axial_n_mm)} N/mm, "
        f"k_lat = {_fmt_num_compact(k_lat_n_mm)} N/mm, "
        f"Δζ = {damping_delta:.3f}, Base = {base_mass_kg:.1f} kg"
    )


def _fmt_belt_values(
    ea_n: float, tooth_k: float, linear_mass_kg_m: float, max_tension_n: float
) -> str:
    mass_str = f"{linear_mass_kg_m * 1e3:.2f}".rstrip("0").rstrip(".")
    return (
        f"EA = {ea_n:,.0f} N, "
        f"Tooth k = {tooth_k:,.0f} N/(m·mm), "
        f"Mass = {mass_str} g/m, "
        f"Max T = {max_tension_n:.0f} N"
    )


def _fmt_bed_mount_values(
    k_axial: float, k_lat: float, k_therm_n_mm: float, damping_delta: float, standoff_mm: float
) -> str:
    return (
        f"k_axial = {_fmt_num_compact(k_axial)} N/mm, "
        f"k_lat = {_fmt_num_compact(k_lat)} N/mm, "
        f"k_therm = {_fmt_num_compact(k_therm_n_mm)} N/mm, "
        f"Δζ = {damping_delta:+.3f}, H = {standoff_mm:.1f} mm"
    )


def _fmt_z_drive_values(
    *,
    kind: str,
    k_drive: float | None,
    k_nut: float,
    k_thrust: float,
    backlash_um: float,
) -> str:
    if kind == "belt":
        drv = float(k_drive or 0.18)
        return (
            f"k_drive = {drv:.2f} N/µm ({drv * 1000.0:.0f} N/mm)/corner, "
            f"Backlash = {backlash_um:.1f} µm"
        )
    return (
        f"k_nut = {k_nut:.1f} N/µm, "
        f"k_thrust = {k_thrust:.1f} N/µm, "
        f"Backlash = {backlash_um:.1f} µm"
    )


def build_model_confidence_audit(spec: PrinterSpec) -> dict[str, Any]:
    """Build a live, honest physics model confidence & WIP disclaimer audit for a PrinterSpec."""
    from printsim.fea.components import (
        dual_rail_steiner_efficiency,
        hertzian_rail_carriage_stiffness,
    )
    from printsim.geometry.builder import resolve_toolhead_hardware_accessories
    from printsim.schema import printer as _p
    from printsim.schema.materials import get_material
    from printsim.schema.profiles import get_profile

    chamber_c = float(spec.frame.chamber_temp_c)

    # 1. Frame Extrusion Profile
    horiz_prof = get_profile(spec.frame.profile)
    up_prof_id = spec.frame.effective_upright_profile
    up_prof = get_profile(up_prof_id)
    if up_prof_id != spec.frame.profile:
        frame_prof_sel = f"{horiz_prof.name} (Horiz.) + {up_prof.name} (Uprights)"
        frame_prof_vals = (
            f"Horiz. A = {horiz_prof.area_m2 * 1e6:.0f} mm², "
            f"Iy = {_fmt_cm4(horiz_prof.iy_m4 * 1e8)} cm⁴, "
            f"Iz = {_fmt_cm4(horiz_prof.iz_m4 * 1e8)} cm⁴, "
            f"J = {_fmt_cm4(horiz_prof.j_m4 * 1e8)} cm⁴ · "
            f"Uprights A = {up_prof.area_m2 * 1e6:.0f} mm², "
            f"Iy = {_fmt_cm4(up_prof.iy_m4 * 1e8)} cm⁴, "
            f"J = {_fmt_cm4(up_prof.j_m4 * 1e8)} cm⁴"
        )
    else:
        frame_prof_sel = horiz_prof.name
        frame_prof_vals = (
            f"A = {horiz_prof.area_m2 * 1e6:.0f} mm², "
            f"Iy = {_fmt_cm4(horiz_prof.iy_m4 * 1e8)} cm⁴, "
            f"Iz = {_fmt_cm4(horiz_prof.iz_m4 * 1e8)} cm⁴, "
            f"J = {_fmt_cm4(horiz_prof.j_m4 * 1e8)} cm⁴"
        )

    # 2. Frame Material Constants
    frame_mat = get_material(spec.frame.material)
    frame_e_gpa = frame_mat.effective_modulus_pa(chamber_c) * 1e-9
    frame_alpha_ppm = frame_mat.thermal_expansion_1_k * 1e6

    # 3. Frame Corner Joints
    cj = resolve_corner_joint(spec)
    cj_overridden = _has_group_override(spec, "corner_joint")
    cj_key = spec.frame.corner_joint_type.lower().strip()
    cj_sel = _CORNER_JOINT_LABELS.get(cj_key, _humanize_label(spec.frame.corner_joint_type))
    cj_vals = _fmt_corner_joint_values(
        cj["k_t_n_m"],
        cj["k_r_inplane_nm_rad"],
        cj["k_r_torsion_nm_rad"],
        cj["damping_delta"],
    )
    if cj_key == "rigid" and cj_overridden:
        cj_trust = 64
        cj_basis = "User-overridden corner joint stiffness and Iwan micro-slip damping."
        cj_caveat = "Accuracy reflects your entered/measured corner joint stiffness and damping parameters."
    else:
        cj_conf = get_item_confidence("corner_joints", cj_key)
        cj_trust = cj_conf["trust_pct"]
        cj_basis = cj_conf["basis"]
        cj_caveat = cj_conf["caveat"]
    cj_breakdown: list[dict[str, Any]] = []
    for jid in _p.CORNER_JOINT_BASE_2020:
        kt_h, kr_h, ktor_h = _base_corner_joint_for_profile(spec.frame.profile, jid)
        if up_prof_id != spec.frame.profile:
            kt_u, kr_u, ktor_u = _base_corner_joint_for_profile(up_prof_id, jid)
            opt_kt = kt_h if kt_h == kt_u else math.sqrt(kt_h * kt_u)
            opt_kr = kr_h if kr_h == kr_u else math.sqrt(kr_h * kr_u)
            opt_ktor = ktor_h if ktor_h == ktor_u else math.sqrt(ktor_h * ktor_u)
        else:
            opt_kt, opt_kr, opt_ktor = kt_h, kr_h, ktor_h
        opt_damp = float(_p.CORNER_JOINT_IWAN_DAMPING.get(jid, 0.003))
        cj_breakdown.append(
            _make_option_breakdown_item(
                opt_id=jid,
                label=_CORNER_JOINT_LABELS.get(jid, jid),
                category="corner_joints",
                active=(jid == cj_key),
                summary_values=_fmt_corner_joint_values(opt_kt, opt_kr, opt_ktor, opt_damp),
            )
        )

    # 4. Enclosure Panels & Fasteners
    if not spec.frame.enclosed_panels or spec.frame.panel_mounting == "none":
        panel_sel = "Open Frame (No Structural Panels)"
        panel_vals = "0 panels · t = 0.0 mm (unenclosed frame)"
        panel_trust = 95
        panel_basis = "Open unpaneled beam-truss FEM (no shear diaphragm coupling)."
        panel_caveat = "Assumes any optional loose doors or dust covers carry zero structural shear load."
    else:
        p_mat = get_material(spec.frame.panel_material)
        p_e_gpa = p_mat.effective_modulus_pa(chamber_c) * 1e-9
        p_g_gpa = p_mat.effective_shear_modulus_at_temp_pa(chamber_c) * 1e-9
        if spec.frame.panel_mounting == "bolted_structural":
            panel_sel = f"Bolted Structural · {p_mat.name}"
            gantry_note = (
                "gantry-coupled"
                if spec.panels_bolted_to_gantry_effective
                else "outer-frame only"
            )
            panel_vals = (
                f"t = {spec.frame.panel_thickness_mm:.1f} mm, "
                f"E = {p_e_gpa:.1f} GPa, G = {p_g_gpa:.2f} GPa ({gantry_note})"
            )
            panel_trust = 86
            panel_basis = (
                "MITC4 Mindlin-Reissner shell plate FEM coupled via Huth-Schwarmann M3 bolt-bearing compliance."
            )
            panel_caveat = (
                "Huth fastener compliance (~70–85% certainty) varies with bolt spacing, washer clamping, and hole clearance."
            )
        else:
            panel_sel = f"Clip-In Enclosure · {p_mat.name}"
            panel_vals = (
                f"t = {spec.frame.panel_thickness_mm:.1f} mm, "
                f"E = {p_e_gpa:.1f} GPa, ρ = {p_mat.density_kg_m3:,.0f} kg/m³"
            )
            panel_trust = 68
            panel_basis = "MITC4 shell panel mass with nominal low-stiffness clip-mount edge springs."
            panel_caveat = (
                "Plastic quarter-turn clips and foam VHB tape decouple unpredictably, adding rattle and edge friction."
            )

    # 5. Extrusion Cavity Filling
    ef = resolve_extrusion_filling(spec)
    ef_overridden = _has_group_override(spec, "extrusion_filling")
    ef_key = spec.frame.extrusion_filling
    ef_entry = _p.EXTRUSION_FILLINGS.get(ef_key, _p.EXTRUSION_FILLINGS["none"])
    ef_unfilled = (
        ef_key == "none"
        and not ef_overridden
        and not bool(ef["is_active"])
    )
    ef_vals = _fmt_extrusion_filling_values(
        float(ef["density_kg_m3"]),
        float(ef["youngs_modulus_pa"]),
        bool(ef["bonded"]),
        float(ef["damping_boost"]),
    )
    if ef_key == "none" and not ef_unfilled:
        ef_trust = 62
        ef_basis = "User-overridden extrusion cavity filling density, modulus, and damping."
        ef_caveat = "Accuracy reflects your entered composite cavity fill properties."
    else:
        ef_conf = get_item_confidence("extrusion_fillings", ef_key)
        ef_trust = ef_conf["trust_pct"]
        ef_basis = ef_conf["basis"]
        ef_caveat = ef_conf["caveat"]
    ef_breakdown = [
        _make_option_breakdown_item(
            opt_id=fid,
            label=str(fentry["name"]),
            category="extrusion_fillings",
            active=(fid == ef_key),
            summary_values=_fmt_extrusion_filling_values(
                float(fentry["density_kg_m3"]),
                float(fentry["youngs_modulus_pa"]),
                bool(fentry["bonded"]),
                float(fentry["damping_boost"]),
            ),
        )
        for fid, fentry in _p.EXTRUSION_FILLINGS.items()
    ]

    # 6. Benchtop Feet / Isolators
    ff = resolve_frame_feet(spec)
    ff_overridden = _has_group_override(spec, "frame_feet")
    ff_key = spec.frame.feet_type
    ff_entry = _p.FRAME_FEET_CATALOG.get(ff_key, _p.FRAME_FEET_CATALOG["rigid_bolted"])
    ff_rigid = (
        ff_key == "rigid_bolted"
        and not ff_overridden
        and not bool(ff["is_compliant"])
    )
    ff_vals = _fmt_frame_feet_values(
        is_rigid=ff_rigid,
        k_axial_n_mm=float(ff["k_axial_n_m"]) * 1e-3,
        k_lat_n_mm=float(ff["k_lateral_n_m"]) * 1e-3,
        damping_delta=float(ff["damping_delta"]),
        base_mass_kg=float(ff["base_mass_kg"]),
    )
    if ff_key == "rigid_bolted" and not ff_rigid:
        ff_trust = 58
        ff_basis = "User-overridden benchtop foot spring-damper impedance and base paver mass."
        ff_caveat = "Accuracy reflects your entered isolator stiffness, damping, and base mass parameters."
    else:
        ff_conf = get_item_confidence("frame_feet", ff_key)
        ff_trust = ff_conf["trust_pct"]
        ff_basis = ff_conf["basis"]
        ff_caveat = ff_conf["caveat"]
    ff_breakdown = [
        _make_option_breakdown_item(
            opt_id=fid,
            label=str(fentry["name"]),
            category="frame_feet",
            active=(fid == ff_key),
            summary_values=_fmt_frame_feet_values(
                is_rigid=(fid == "rigid_bolted"),
                k_axial_n_mm=float(fentry["k_axial_n_per_mm"]),
                k_lat_n_mm=float(fentry["k_lateral_n_per_mm"]),
                damping_delta=float(fentry["damping_delta"]),
                base_mass_kg=float(fentry.get("base_mass_kg", 0.0)),
            ),
        )
        for fid, fentry in _p.FRAME_FEET_CATALOG.items()
    ]

    # 7. X-Gantry Beam Profile & Material
    xb_prof = get_profile(spec.gantry.x_beam_profile)
    xb_mat = get_material(spec.gantry.x_beam_material)
    xb_e_gpa = xb_mat.effective_modulus_pa(chamber_c) * 1e-9
    xb_g_gpa = xb_mat.effective_shear_modulus_at_temp_pa(chamber_c) * 1e-9
    xb_is_cf = (
        spec.gantry.x_beam_profile.startswith("cf_")
        or spec.gantry.x_beam_material.startswith("carbon_fiber")
        or "carbon" in xb_mat.name.lower()
    )
    cf_mount = spec.gantry.effective_cf_rail_mounting
    xb_vals = (
        f"E = {xb_e_gpa:.1f} GPa, G = {xb_g_gpa:.2f} GPa, "
        f"Iy = {_fmt_cm4(xb_prof.iy_m4 * 1e8)} cm⁴, Iz = {_fmt_cm4(xb_prof.iz_m4 * 1e8)} cm⁴"
    )
    if cf_mount == "two_wall_clamped_inserts":
        xb_vals += " · 2-Wall Clamped Inserts"
    elif cf_mount == "single_wall_bare":
        xb_vals += " · Single-Wall Bare"

    # 8. Linear Rails & Hertzian Carriage
    rp = resolve_rail_preload(spec)
    rq = resolve_rail_quality(spec)
    x_rail_prof = get_profile(spec.gantry.x_rail_profile)
    y_rail_prof = get_profile(spec.gantry.y_rail_profile)
    pre_key = spec.gantry.rail_preload.upper()
    pre_label = _RAIL_PRELOAD_LABELS.get(pre_key, pre_key)
    blk_label = _RAIL_BLOCK_LABELS.get(
        spec.gantry.rail_block_type.upper(), spec.gantry.rail_block_type.upper()
    )
    dual_prefix = "Dual " if spec.gantry.effective_dual_x_rails else ""
    if spec.gantry.x_rail_profile == spec.gantry.y_rail_profile:
        rail_sel = f"{dual_prefix}{x_rail_prof.name} · {blk_label} · {pre_label}"
    else:
        rail_sel = (
            f"X: {dual_prefix}{x_rail_prof.name} / Y: {y_rail_prof.name} · {blk_label} · {pre_label}"
        )
    car_x = hertzian_rail_carriage_stiffness(
        spec.gantry.x_rail_profile,
        preload=spec.gantry.rail_preload,
        block_type=spec.gantry.rail_block_type,
        rail_quality=spec.gantry.rail_quality,
        preload_fraction_override=spec.model_overrides.rail_preload.preload_fraction,
        stiffness_mult_override=rq["stiffness_mult"],
        drag_mult_override=rq["drag_mult"],
    )
    car_mult = 2.0 if spec.gantry.effective_dual_x_rails else 1.0
    if spec.gantry.effective_dual_x_rails:
        z_track = 0.5 * (xb_prof.height_m + x_rail_prof.height_m)
        eta_steiner = dual_rail_steiner_efficiency(spec.gantry.x_beam_profile, cf_mount)
        roll_nm_rad = (
            2.0 * float(car_x["k_rx_nm_rad"])
            + eta_steiner * 2.0 * float(car_x["k_yy_n_m"]) * (z_track**2)
        )
    else:
        roll_nm_rad = float(car_x["k_rx_nm_rad"])
    radial_n_um = float(car_x["k_zz_n_m"]) * car_mult * 1e-6
    block_mass_g = float(car_x.get("block_mass_g", 54.0)) * car_mult
    rail_vals = (
        f"Preload = {rp['preload_fraction']:.3f}×C, "
        f"Block = {block_mass_g:.0f} g, "
        f"k_roll = {roll_nm_rad:,.0f} N·m/rad, "
        f"k_radial = {radial_n_um:.1f} N/µm"
    )

    # 9. Rail Quality / Grade Multiplier
    rq_key = str(spec.gantry.rail_quality).lower().strip()
    rq_entry = _p.RAIL_QUALITY_CATALOG.get(rq_key, _p.RAIL_QUALITY_CATALOG["hiwin_genuine"])
    rq_vals = f"Stiffness = {rq['stiffness_mult']:.2f}×, Drag = {rq['drag_mult']:.2f}×"
    rq_conf = get_item_confidence("rail_qualities", rq_key)
    rq_breakdown = [
        _make_option_breakdown_item(
            opt_id=qid,
            label=str(qentry["name"]),
            category="rail_qualities",
            active=(qid == rq_key),
            summary_values=(
                f"Stiffness = {float(qentry['stiffness_mult']):.2f}×, "
                f"Drag = {float(qentry['drag_mult']):.2f}×"
            ),
        )
        for qid, qentry in _p.RAIL_QUALITY_CATALOG.items()
    ]

    # 10. XY Drive Belt Stiffness
    bl = resolve_belt(spec)
    belt = spec.gantry.belt
    belt_core_label = _BELT_CORE_LABELS.get(belt.core, belt.core.title())
    belt_sel = f"{belt.catalog_entry['label']} ({belt_core_label})"
    belt_vals = _fmt_belt_values(
        float(bl["stiffness_ea_n"]),
        float(bl["tooth_shear_k_n_per_m_mm"]),
        float(bl["linear_mass_kg_m"]),
        float(bl["max_working_tension_n"]),
    )
    belt_conf = get_item_confidence("belts", belt.type)
    belt_breakdown = [
        _make_option_breakdown_item(
            opt_id=bid,
            label=str(bentry["label"]),
            category="belts",
            active=(bid == belt.type),
            summary_values=_fmt_belt_values(
                float(bentry["ea_default_core_n"]),
                float(bentry["tooth_shear_k_n_per_m_mm"]),
                float(bentry["linear_mass_kg_m"]),
                float(bentry["max_working_tension_n"]),
            ),
        )
        for bid, bentry in _p.BELT_CATALOG.items()
    ]

    # 11. Belt Path & Idler Routing
    br = resolve_belt_routing(spec)
    br_key = spec.gantry.belt_routing
    br_entry = _p.BELT_ROUTING_CATALOG.get(br_key, _p.BELT_ROUTING_CATALOG["stock_corexy"])
    br_vals = (
        f"Span = {br['span_factor']:.2f}×, "
        f"Idler stiffness = {br['idler_stiffness_factor']:.2f}×"
    )
    br_conf = get_item_confidence("belt_routings", br_key)
    br_breakdown = [
        _make_option_breakdown_item(
            opt_id=brid,
            label=str(brentry["name"]),
            category="belt_routings",
            active=(brid == br_key),
            summary_values=(
                f"Span = {float(brentry['span_factor']):.2f}×, "
                f"Idler stiffness = {float(brentry['idler_stiffness_factor']):.2f}×"
            ),
        )
        for brid, brentry in _p.BELT_ROUTING_CATALOG.items()
    ]

    # 12. Idler Bearing Play
    bc = resolve_bearing_clearance(spec)
    c_cls = belt.bearing_clearance_class
    bc_sel = _BEARING_CLEARANCE_LABELS.get(c_cls, f"ISO 5753 Class {c_cls}")
    bc_vals = f"Radial play = {bc['radial_play_um']:.1f} µm"

    # 13. Gantry Brackets & Motor Mounts
    gb = resolve_gantry_bracket(spec)
    bkt_key = spec.gantry.bracket_material
    bkt_entry = _p.GANTRY_BRACKET_CATALOG.get(bkt_key, _p.GANTRY_BRACKET_CATALOG["abs_printed"])
    cbo = spec.gantry.custom_bracket_override
    bkt_cad_active = bool(gb.get("cad_backed", False))
    bkt_cad_kit_id = str(gb.get("cad_kit_id") or "")
    bkt_cad_trust: int | None = None
    if cbo is not None:
        src_label = _BRACKET_SOURCE_LABELS.get(cbo.source, _humanize_label(cbo.source))
        bkt_sel = f"{_humanize_label(cbo.label)} ({src_label})"
        bkt_trust = 85
        bkt_basis = "3D continuum solid Tet10/Hex8 FEA & Craig-Bampton CMS superelement condensation."
        bkt_caveat = "Accurate for analyzed bracket geometry; bolt-boss preload and FDM layer anisotropy add ±10–15% spread."
    elif bkt_cad_active:
        kit_meta = _p.GANTRY_HARDWARE_KIT_CATALOG.get(bkt_cad_kit_id, {})
        kit_name = str(kit_meta.get("name") or _humanize_label(bkt_cad_kit_id))
        kit_short = str(kit_meta.get("short_name") or kit_name)
        kit_stations = kit_meta.get("stations") or {}
        trust_vals: list[float] = []
        for st_key in ("rear_left", "front_left", "xy_left"):
            aid = kit_stations.get(st_key)
            man = _get_cad_asset_manifest("gantry_brackets", aid) if aid else None
            if man is not None:
                trust_vals.append(float(man.get("cad_trust_pct", 86)))
        bkt_cad_trust = int(round(sum(trust_vals) / len(trust_vals))) if trust_vals else 86
        bkt_sel = f"{bkt_entry['name']} · CAD {kit_short}"
        bkt_conf = get_item_confidence("gantry_brackets", bkt_key)
        bkt_trust = max(int(bkt_conf["trust_pct"]), bkt_cad_trust)
        bkt_basis = (
            f"3D CAD Craig-Bampton gantry bracket superelements ({kit_name}: rear, front, and XY-joint stations) "
            f"scaled by {bkt_entry['short_name']} temperature-dependent modulus in series with bolted-joint contact compliance."
        )
        bkt_caveat = (
            "Bolted-joint preload scatter, T-nut micro-slip, and polymer creep/print-orientation anisotropy add ±10–15% spread."
        )
    else:
        bkt_sel = str(bkt_entry["name"])
        bkt_conf = get_item_confidence("gantry_brackets", bkt_key)
        bkt_trust = bkt_conf["trust_pct"]
        bkt_basis = bkt_conf["basis"]
        bkt_caveat = bkt_conf["caveat"]
    eff_mass_delta_g = float(gb["carriage_mass_delta_g"]) + (
        float(cbo.mass_delta_g) if cbo is not None else 0.0
    )
    bkt_vals = (
        f"k = {_fmt_num_compact(gb['stiffness_n_m'] * 1e-3)} N/mm, "
        f"Δm = {eff_mass_delta_g:+.0f} g, "
        f"Creep limit = {gb['creep_tension_limit_n']:.0f} N, "
        f"Δζ = {gb['damping_delta']:+.3f}"
    )
    bkt_breakdown = [
        _make_option_breakdown_item(
            opt_id=gid,
            label=str(gentry["name"]),
            category="gantry_brackets",
            active=(cbo is None and gid == bkt_key),
            summary_values=(
                f"E = {float(gentry['youngs_modulus_gpa']):.1f} GPa, "
                f"HDT = {float(gentry['hdt_c']):.0f} °C, "
                f"Creep limit = {float(gentry['creep_tension_limit_n']):.0f} N, "
                f"Δm = {float(gentry['carriage_mass_delta_g']):+.0f} g"
            ),
        )
        for gid, gentry in _p.GANTRY_BRACKET_CATALOG.items()
    ]

    # 14. Stepper Motor Electromechanics
    mot = spec.kinematics.xy_stepper
    drive_label = _DRIVE_TYPE_LABELS.get(
        spec.kinematics.drive_type, _humanize_label(spec.kinematics.drive_type)
    )
    mot_sel = f"{_humanize_label(mot.name)} · {drive_label}"
    mot_vals = (
        f"T_hold = {mot.holding_torque_nm:.2f} N·m (eff. {mot.effective_holding_torque_nm:.2f} N·m), "
        f"R = {mot.phase_resistance_ohm:.2f} Ω, L = {mot.inductance_mh:.2f} mH, "
        f"V = {mot.supply_voltage_v:.0f} V, I_run = {mot.run_current_a:.2f} A, "
        f"k_EM = {mot.torsional_stiffness_nm_per_rad():.1f} N·m/rad, "
        f"Class {mot.insulation_class} ({mot.max_winding_temp_c:.0f}°C)"
    )

    # 15. Toolhead Preset (Mass & CoG)
    use_cad_se = bool(getattr(spec, "use_cad_superelements", True))
    th = spec.toolhead
    th_pid = (th.preset_id or "").lower().strip()
    th_cad_id = "stealthburner" if th_pid == "stealthburner_tap" else th_pid
    th_cad_manifest = _get_cad_asset_manifest("toolheads", th_cad_id) if use_cad_se else None
    th_cad_active = th_cad_manifest is not None
    th_cad_trust = (
        int(round(float(th_cad_manifest.get("cad_trust_pct", 90))))
        if th_cad_manifest is not None
        else None
    )
    if th_pid in _p.TOOLHEAD_PRESETS:
        th_preset = _p.TOOLHEAD_PRESETS[th_pid]
        th_sel = str(th_preset["name"])
        th_conf = get_item_confidence("toolhead_presets", th_pid)
        th_trust = th_conf["trust_pct"]
        th_basis = th_conf["basis"]
        th_caveat = th_conf["caveat"]
    else:
        th_sel = f"Custom · {_humanize_label(th.name)}"
        th_trust = 65
        th_basis = "User-entered custom mass/CoG — accuracy reflects your scale/CAD measurements."
        th_caveat = (
            "Custom toolhead dressed mass, CoG offset, and nozzle Z-drop are not verified against a preset scale benchmark."
        )
    if th_cad_manifest is not None:
        th_cad_name = _humanize_label(str(th_cad_manifest.get("name") or th_pid))
        th_basis = (
            f"3D CAD Craig-Bampton superelement ({th_cad_name}) active for structural coupling. "
            f"{th_basis}"
        )
    th_acc = resolve_toolhead_hardware_accessories(spec)
    eff_body_g = float(th_acc["effective_body_mass_g"])
    eff_hotend_g = float(th_acc["effective_hotend_mass_g"])
    eff_cog = th_acc["effective_cog_offset_mm"]
    th_vals = (
        f"m_body = {eff_body_g:.1f} g (+{eff_hotend_g:.1f} g hotend), "
        f"CoG = ({eff_cog[0]:+.1f}, {eff_cog[1]:+.1f}, {eff_cog[2]:+.1f}) mm, "
        f"Nozzle drop = {abs(th.nozzle_offset_mm[2]):.1f} mm"
    )
    th_breakdown = [
        _make_option_breakdown_item(
            opt_id=tid,
            label=str(tentry["name"]),
            category="toolhead_presets",
            active=(tid == th_pid),
            summary_values=(
                f"m_body = {float(tentry['total_mass_g']):.0f} g, "
                f"CoG = ({float(tentry['cog_offset_mm'][0]):+.0f}, "
                f"{float(tentry['cog_offset_mm'][1]):+.0f}, "
                f"{float(tentry['cog_offset_mm'][2]):+.0f}) mm, "
                f"Nozzle drop = {abs(float(tentry['nozzle_offset_mm'][2])):.0f} mm"
            ),
            cad_manifest=(
                _get_cad_asset_manifest(
                    "toolheads", "stealthburner" if tid == "stealthburner_tap" else tid
                )
                if use_cad_se
                else None
            ),
        )
        for tid, tentry in _p.TOOLHEAD_PRESETS.items()
    ]

    # 16. Toolhead Body Construction
    zp = resolve_z_probe(spec)
    tc = resolve_toolhead_construction(spec)
    bc_key = spec.toolhead.body_construction
    body_mat = get_material(spec.toolhead.effective_body_material)
    if bc_key is not None and bc_key in _p.TOOLHEAD_CONSTRUCTION_CATALOG:
        tc_sel = str(_p.TOOLHEAD_CONSTRUCTION_CATALOG[bc_key]["name"])
        tc_conf = get_item_confidence("toolhead_constructions", bc_key)
        tc_trust = tc_conf["trust_pct"]
        tc_basis = tc_conf["basis"]
        tc_caveat = tc_conf["caveat"]
    else:
        tc_sel = f"Direct Chassis Stiffness · {body_mat.name}"
        tc_trust = 45
        tc_basis = "Lumped Stage 2 heuristic for unmodeled multi-part plastic/metal toolhead chassis."
        tc_caveat = "Real stiffness depends heavily on internal ribs, fan duct seams, heat-set inserts, and print orientation."
    if th_cad_manifest is not None:
        th_cad_name = _humanize_label(str(th_cad_manifest.get("name") or th_pid))
        tc_sel = f"CAD Superelement ({th_cad_name})"
        if th_cad_trust is not None:
            tc_trust = th_cad_trust
        tc_basis = (
            f"3D CAD Craig-Bampton superelement ({th_cad_name}) condensed from solid STEP/Tet mesh "
            "in series with Hertzian MGN linear rail carriage compliance (backplate integrated into 3D CAD solid)."
        )
        tc_caveat = (
            "Material modulus scaling and bolted-seam micro-slip are approximated relative to the "
            "reference CAD solid mesh."
        )
        port_k_th = th_cad_manifest.get("port_stiffness_n_per_mm")
        tc_ov = spec.model_overrides.toolhead_construction
        if isinstance(port_k_th, dict) and {"kx", "ky", "kz"}.issubset(port_k_th):
            mat_scale = 1.0
            if (
                not bool(th_cad_manifest.get("is_multi_material", False))
                and "reference_youngs_modulus_pa" in th_cad_manifest
            ):
                e_ref = float(th_cad_manifest["reference_youngs_modulus_pa"])
                e_eff = float(body_mat.effective_modulus_pa(chamber_c))
                mat_scale = e_eff / max(e_ref, 1.0)
            ov_scale = 1.0
            if tc_ov.body_stiffness_base_n_per_mm is not None:
                ov_scale = float(tc_ov.body_stiffness_base_n_per_mm) / max(
                    float(tc["baseline_body_stiffness_base_n_per_mm"]), 1e-6
                )
            tot_scale = mat_scale * ov_scale * float(zp["compliance_factor"])
            kx_th = float(port_k_th["kx"]) * tot_scale
            ky_th = float(port_k_th["ky"]) * tot_scale
            kz_th = float(port_k_th["kz"]) * tot_scale
            n_el_th = th_cad_manifest.get("num_elements")
            el_ty_th = str(th_cad_manifest.get("element_type") or "Tet4")
            mesh_tag = f"{int(n_el_th):,}-elem {el_ty_th}, " if n_el_th else ""
            tc_vals = (
                f"Kx = {_fmt_num_compact(kx_th)} / Ky = {_fmt_num_compact(ky_th)} / "
                f"Kz = {_fmt_num_compact(kz_th)} N/mm ({mesh_tag}{body_mat.name})"
            )
            if tc_ov.body_stiffness_base_n_per_mm is not None:
                tc_vals += (
                    f", Override k_base = {_fmt_num_compact(tc['body_stiffness_base_n_per_mm'])} N/mm"
                )
        else:
            tc_vals = (
                f"3D CAD CMS Superelement ({body_mat.name}), "
                f"k_base = {_fmt_num_compact(tc['body_stiffness_base_n_per_mm'])} N/mm"
            )
    else:
        tc_vals = (
            f"k_base = {_fmt_num_compact(tc['body_stiffness_base_n_per_mm'])} N/mm, "
            f"Torsion = {tc['torsion_mult']:.2f}×, "
            f"Body wall = {spec.toolhead.body_wall_thickness_mm:.1f} mm, "
            f"Carriage plate = {spec.toolhead.carriage_plate_thickness_mm:.1f} mm"
        )
    tc_breakdown = [
        _make_option_breakdown_item(
            opt_id=cid,
            label=str(centry["name"]),
            category="toolhead_constructions",
            active=(cid == bc_key and not th_cad_active),
            summary_values=(
                f"k_base = {_fmt_num_compact(float(centry['body_stiffness_base_n_per_mm']))} N/mm, "
                f"Torsion = {float(centry['torsion_mult']):.2f}×"
            ),
            cad_manifest=None,
        )
        for cid, centry in _p.TOOLHEAD_CONSTRUCTION_CATALOG.items()
    ]

    # 17. Z-Probe / Carriage Interface
    zp_key = spec.toolhead.z_probe_type
    zp_entry = _p.Z_PROBE_CATALOG.get(zp_key, _p.Z_PROBE_CATALOG["static_eddy"])
    zp_vals = (
        f"Compliance = {zp['compliance_factor']:.2f}×, "
        f"CoG ΔY = {zp['cog_y_shift_mm']:+.1f} mm"
    )
    zp_conf = get_item_confidence("z_probes", zp_key)
    zp_breakdown = [
        _make_option_breakdown_item(
            opt_id=zid,
            label=str(zentry["name"]),
            category="z_probes",
            active=(zid == zp_key),
            summary_values=(
                f"Compliance = {float(zentry['compliance_factor']):.2f}×, "
                f"Added mass = {float(zentry['added_mass_g']):.0f} g, "
                f"CoG ΔY = {float(zentry['cog_y_shift_mm']):+.1f} mm"
            ),
        )
        for zid, zentry in _p.Z_PROBE_CATALOG.items()
    ]

    # 18. Hotend Structural Clamp & Neck
    hc = resolve_hotend_clamp(spec)
    he = spec.toolhead.hotend
    hm_label = _HOTEND_MOUNT_STYLE_LABELS.get(
        he.mount_style, _humanize_label(he.mount_style)
    )
    he_pid = (he.preset_id or "").lower().strip()
    he_combo_id = f"{th_pid}__{he_pid}" if (th_pid and he_pid) else ""
    use_bottom_brace = bool(getattr(spec.toolhead, "use_hotend_bottom_brace", True))
    he_cad_is_braced = False
    he_cad_manifest = None
    if use_cad_se:
        if he_combo_id and use_bottom_brace:
            he_cad_manifest = _get_cad_asset_manifest("assemblies", he_combo_id)
            if he_cad_manifest is not None:
                he_cad_is_braced = True
        if he_cad_manifest is None and he_pid:
            he_cad_manifest = _get_cad_asset_manifest("hotends", he_pid)
    he_cad_active = he_cad_manifest is not None
    he_cad_id = (he_combo_id if he_cad_is_braced else he_pid) if he_cad_active else None
    he_cad_trust = (
        int(round(float(he_cad_manifest.get("cad_trust_pct", 92))))
        if he_cad_manifest is not None
        else None
    )
    hc_sel = (
        f"{_humanize_label(he.name)} + Microwaved Ti Bottom Brace · {hm_label}"
        if he_cad_is_braced
        else f"{_humanize_label(he.name)} · {hm_label}"
    )
    hc_vals = (
        f"Clamp base = {_fmt_num_compact(hc['clamp_stiffness_base_n_per_mm'])} N/mm, "
        f"Hotend k = {_fmt_num_compact(he.structural_stiffness_n_per_mm)} N/mm, "
        f"Collar = {spec.toolhead.hotend_mount_thickness_mm:.1f} mm"
    )
    if he_cad_manifest is not None:
        port_k_he = he_cad_manifest.get("port_stiffness_n_per_mm")
        if isinstance(port_k_he, dict) and {"kx", "ky"}.issubset(port_k_he):
            he_k_scale = 1.0
            cat_he = _p.HOTEND_CATALOG.get(he_pid)
            legacy_ks_he = {
                "tricorn_long": (1100.0,),
                "tricorn_short": (1650.0,),
                "chube_air": (2600.0, 2250.0),
                "chube_compact": (2200.0, 2600.0),
                "chube_conduction": (3600.0, 5670.0),
            }.get(he_pid, ())
            if cat_he is not None:
                cat_k_he = float(cat_he["structural_stiffness_n_per_mm"])
                he_k_val = float(he.structural_stiffness_n_per_mm)
                is_def_he = (
                    cat_k_he > 0.0 and abs(he_k_val - cat_k_he) / cat_k_he <= 1e-4
                ) or any(abs(he_k_val - lk) / lk <= 1e-4 for lk in legacy_ks_he)
                if cat_k_he > 0.0 and not is_def_he:
                    he_k_scale = max(he_k_val / cat_k_he, 1e-3)
            kx_he = float(port_k_he["kx"]) * he_k_scale
            ky_he = float(port_k_he["ky"]) * he_k_scale
            n_el_he = he_cad_manifest.get("num_elements")
            el_ty_he = str(he_cad_manifest.get("element_type") or "Tet4")
            mesh_he = f" ({int(n_el_he):,}-elem {el_ty_he})" if n_el_he else ""
            cad_he_tag = "CAD Braced Hotend" if he_cad_is_braced else "CAD Hotend"
            hc_vals = (
                f"Clamp base = {_fmt_num_compact(hc['clamp_stiffness_base_n_per_mm'])} N/mm, "
                f"{cad_he_tag} Kx/Ky = {_fmt_num_compact(kx_he)}/{_fmt_num_compact(ky_he)} N/mm{mesh_he}, "
                f"Collar = {spec.toolhead.hotend_mount_thickness_mm:.1f} mm"
            )
    ms_conf = get_item_confidence("hotend_mount_styles", he.mount_style)
    if he_pid in ITEM_CONFIDENCE_REGISTRY["hotends"]:
        he_conf = get_item_confidence("hotends", he_pid)
        hc_trust = he_conf["trust_pct"]
        hc_basis = f"{he_conf['basis']} Clamp style ({hm_label}): {ms_conf['basis']}"
        hc_caveat = f"{he_conf['caveat']} {ms_conf['caveat']}"
    else:
        hc_trust = ms_conf["trust_pct"]
        hc_basis = ms_conf["basis"]
        hc_caveat = ms_conf["caveat"]
    if he_cad_manifest is not None:
        he_cad_name = _humanize_label(str(he_cad_manifest.get("name") or (he_cad_id or he_pid)))
        if he_cad_is_braced:
            hc_basis = (
                f"Coupled 3D CAD Craig-Bampton superelement ({he_cad_name}) capturing both the top "
                f"{hm_label} mount and lower Ti bottom brace parallel load path, condensed from solid STEP/Tet mesh. {hc_basis}"
            )
        else:
            hc_basis = (
                f"3D CAD Craig-Bampton superelement ({he_cad_name}) condensed from solid STEP/Tet mesh "
                f"in series with {hm_label} clamp compliance. {hc_basis}"
            )
    hc_breakdown = [
        _make_option_breakdown_item(
            opt_id=hid,
            label=str(hentry["name"]),
            category="hotends",
            active=(hid == he_pid),
            summary_values=(
                f"Hotend k = {_fmt_num_compact(float(hentry['structural_stiffness_n_per_mm']))} N/mm, "
                f"Clamp base = {_fmt_num_compact(float(_p.HOTEND_MOUNT_CLAMP_BASE_N_PER_MM.get(hentry['mount_style'], 3200.0)))} N/mm, "
                f"Mass = {float(hentry['mass_g']):.0f} g, "
                f"L = {float(hentry['overall_length_mm']):.1f} mm"
            ),
            cad_manifest=(
                he_cad_manifest
                if hid == he_pid
                else (
                    (
                        (
                            _get_cad_asset_manifest("assemblies", f"{th_pid}__{hid}")
                            if (th_pid and use_bottom_brace)
                            else None
                        )
                        or _get_cad_asset_manifest("hotends", hid)
                    )
                    if use_cad_se
                    else None
                )
            ),
        )
        for hid, hentry in _p.HOTEND_CATALOG.items()
    ]

    # 19. Hotend Thermal & Melt Flow
    hb_mat = get_material(he.heatbreak_material)
    ht_sel = f"{_humanize_label(he.name)} · {hb_mat.name}"
    ht_vals = (
        f"{he.heater_power_w:.0f} W, Melt = {he.melt_zone_length_mm:.1f} mm, "
        f"Heatbreak = {he.heatbreak_wall_mm:.2f} mm wall, Max = {he.max_temp_c:.0f} °C"
    )

    # 20. Heated Bed Plate Mechanics
    bed = spec.bed
    bed_mat = get_material(bed.plate_material)
    bed_e_gpa = bed_mat.effective_modulus_pa(bed.target_temp_c) * 1e-9
    bed_alpha_ppm = bed_mat.thermal_expansion_1_k * 1e6
    bed_dim_str = (
        f"{bed.size_mm[0]:.0f}×{bed.size_mm[1]:.0f}×{bed.plate_thickness_mm:.1f} mm"
    )
    bed_sel = f"{bed_mat.name} ({bed_dim_str})"
    bed_vals = (
        f"{bed_dim_str}, E = {bed_e_gpa:.1f} GPa, "
        f"α = {bed_alpha_ppm:.1f} ppm/K, Target = {bed.target_temp_c:.0f} °C"
    )

    # 21. Bed Mount Hardware
    bm = resolve_bed_mount(spec)
    bm_key = bed.mount_type
    bm_entry = _p.BED_MOUNT_CATALOG.get(bm_key, _p.BED_MOUNT_CATALOG["kinematic_3pt"])
    bm_vals = _fmt_bed_mount_values(
        float(bm["k_axial_n_per_mm"]),
        float(bm["k_lateral_n_per_mm"]),
        float(bm["k_restraint_thermal_n_per_mm"]),
        float(bm["damping_delta"]),
        float(bm["standoff_height_mm"]),
    )
    bm_conf = get_item_confidence("bed_mounts", bm_key)
    bm_breakdown = [
        _make_option_breakdown_item(
            opt_id=bmid,
            label=str(bmentry["name"]),
            category="bed_mounts",
            active=(bmid == bm_key),
            summary_values=_fmt_bed_mount_values(
                float(bmentry["k_axial_n_per_mm"]),
                float(bmentry["k_lateral_n_per_mm"]),
                float(bmentry.get("k_restraint_thermal_n_per_m", 0.0)) * 1e-3,
                float(bmentry["damping_delta"]),
                float(bmentry.get("standoff_height_mm", 14.0)),
            ),
        )
        for bmid, bmentry in _p.BED_MOUNT_CATALOG.items()
    ]

    # 22. Z-Drive Mechanism
    zd = resolve_z_drive_mechanism(spec)
    zd_id = bed.effective_z_leadscrew_type
    zd_entry = _p.Z_DRIVE_MECHANISM_CATALOG.get(
        zd_id, _p.Z_DRIVE_MECHANISM_CATALOG["tr8x4_pom"]
    )
    z_drive_label = _Z_DRIVE_LABELS.get(bed.z_drive, _humanize_label(bed.z_drive))
    zd_sel = f"{zd_entry['name']} · {z_drive_label}"
    zd_vals = _fmt_z_drive_values(
        kind=str(zd["kind"]),
        k_drive=zd["k_drive_n_per_um_per_corner"],
        k_nut=float(zd["k_nut_n_per_um"]),
        k_thrust=float(zd["k_thrust_n_per_um"]),
        backlash_um=float(zd["backlash_um"]),
    )
    zd_conf = get_item_confidence("z_drive_mechanisms", zd_id)
    zd_breakdown = [
        _make_option_breakdown_item(
            opt_id=zid,
            label=str(zentry["name"]),
            category="z_drive_mechanisms",
            active=(zid == zd_id),
            summary_values=_fmt_z_drive_values(
                kind=str(zentry["kind"]),
                k_drive=zentry.get("k_drive_n_per_um_per_corner"),
                k_nut=float(zentry["k_nut_n_per_um"]),
                k_thrust=float(zentry["k_thrust_n_per_um"]),
                backlash_um=float(zentry["backlash_um"]),
            ),
        )
        for zid, zentry in _p.Z_DRIVE_MECHANISM_CATALOG.items()
    ]

    # 23. Klipper Input Shaper & Modal Solver
    shaper_key = spec.kinematics.input_shaper
    shaper_label = _INPUT_SHAPER_LABELS.get(shaper_key, _humanize_label(shaper_key))
    shaper_short = shaper_label.split(" (")[0]
    solver_sel = f"{shaper_label} · 12-DOF FEM Eigensolver"
    solver_vals = (
        f"Shaper = {shaper_short}, 5–220 Hz PSD, "
        f"Config accel = {spec.kinematics.max_accel_mm_s2:,.0f} mm/s², "
        f"SCV = {spec.kinematics.square_corner_velocity_mm_s:.1f} mm/s"
    )

    entries: list[dict[str, Any]] = [
        _make_confidence_entry(
            spec,
            entry_id="frame_profile",
            subsystem="frame",
            name="Frame Extrusion Profile",
            active_selection=frame_prof_sel,
            active_values=frame_prof_vals,
            trust_pct=96,
            override_group=None,
            basis="Misumi/Bosch extrusion CAD cross-section datasheets & 12-DOF Euler-Bernoulli beam FEM.",
            caveat="Minor ±3% variance from extrusion die tolerance, T-slot chamfers, and blind-joint access bores.",
        ),
        _make_confidence_entry(
            spec,
            entry_id="frame_material",
            subsystem="frame",
            name="Frame Material Constants",
            active_selection=frame_mat.name,
            active_values=(
                f"E = {frame_e_gpa:.1f} GPa, "
                f"ρ = {frame_mat.density_kg_m3:,.0f} kg/m³, "
                f"α = {frame_alpha_ppm:.1f} ppm/K"
            ),
            trust_pct=98,
            override_group=None,
            basis="MatWeb / ASM International structural alloy datasheets.",
            caveat="Real extruded alloy modulus and density vary <2% across mill lots.",
        ),
        _make_confidence_entry(
            spec,
            entry_id="corner_joint",
            subsystem="frame",
            name="Frame Corner Joints",
            active_selection=cj_sel,
            active_values=cj_vals,
            trust_pct=cj_trust,
            override_group="corner_joint",
            basis=cj_basis,
            caveat=cj_caveat,
            options_breakdown=cj_breakdown,
        ),
        _make_confidence_entry(
            spec,
            entry_id="enclosure_panels",
            subsystem="frame",
            name="Enclosure Panels & Fasteners",
            active_selection=panel_sel,
            active_values=panel_vals,
            trust_pct=panel_trust,
            override_group=None,
            basis=panel_basis,
            caveat=panel_caveat,
        ),
        _make_confidence_entry(
            spec,
            entry_id="extrusion_filling",
            subsystem="frame",
            name="Extrusion Cavity Filling",
            active_selection=str(ef_entry["name"]),
            active_values=ef_vals,
            trust_pct=ef_trust,
            override_group="extrusion_filling",
            basis=ef_basis,
            caveat=ef_caveat,
            options_breakdown=ef_breakdown,
        ),
        _make_confidence_entry(
            spec,
            entry_id="frame_feet",
            subsystem="frame",
            name="Benchtop Feet / Isolators",
            active_selection=str(ff_entry["name"]),
            active_values=ff_vals,
            trust_pct=ff_trust,
            override_group="frame_feet",
            basis=ff_basis,
            caveat=ff_caveat,
            options_breakdown=ff_breakdown,
        ),
        _make_confidence_entry(
            spec,
            entry_id="x_gantry_beam",
            subsystem="gantry",
            name="X-Gantry Beam Profile & Material",
            active_selection=f"{xb_prof.name} · {xb_mat.name}",
            active_values=xb_vals,
            trust_pct=(
                (90 if spec.gantry.x_beam_profile == "cf_tube_20x20_carbonara" else 88)
                if xb_is_cf
                else 96
            ),
            override_group=None,
            basis=(
                "3D orthotropic Tet10 continuum FEA calibration (2-wall clamped PET-CF inserts, "
                "M5 end standoffs, Mindlin bolted shear-lag, local thin-wall seat & diamond torsion), "
                "Classical Lamination Theory, Mahieux-Reifsnider Tg retention, "
                "Carbonara 125 °C onset Tg thermal bench test, and HVFDesigns Klippain ADXL before/after data "
                "(Y 61.2 -> 65.8 Hz, zeta 0.044 -> 0.063; X 84.2 -> 88.8 Hz)."
                if xb_is_cf
                else "Manufacturer cross-section geometry / CAD datasheet, 12-DOF beam FEM, and 3D Tet10 bolted T-slot FEA."
            ),
            caveat=(
                "Carbonara's exact proprietary vendor ply layup is inferred via CLT from published E/G targets; "
                "fiber volume fraction, resin Tg, and bolt preload vary across commercial CF tubes, while "
                "single-wall bare bolting suffers thin-wall oil-canning and end crush near Tg "
                "compared to 2-wall clamped internal inserts."
                if xb_is_cf
                else "Minor ±3% tolerance from extrusion/tube wall thickness and Newmark rail bolt shear lag."
            ),
        ),
        _make_confidence_entry(
            spec,
            entry_id="rail_preload",
            subsystem="gantry",
            name="Linear Rails & Hertzian Carriage",
            active_selection=rail_sel,
            active_values=rail_vals,
            trust_pct=90,
            override_group="rail_preload",
            basis=(
                "HIWIN Gothic-arch Hertzian elliptical contact + Hung et al. (2011) 15% carriage-skirt compliance, "
                "official HIWIN MGN/HGW block catalog masses, and 3D Tet10 continuum FEA calibration of "
                "2020 T-slot/CF rail-seat compliance and dual-rail Steiner roll coupling."
            ),
            caveat="Depends on actual factory ball-preload interference tolerance, rail mounting flatness, and grease state.",
        ),
        _make_confidence_entry(
            spec,
            entry_id="rail_quality",
            subsystem="gantry",
            name="Rail Quality / Grade Multiplier",
            active_selection=str(rq_entry["name"]),
            active_values=rq_vals,
            trust_pct=rq_conf["trust_pct"],
            override_group="rail_quality",
            basis=rq_conf["basis"],
            caveat=rq_conf["caveat"],
            options_breakdown=rq_breakdown,
        ),
        _make_confidence_entry(
            spec,
            entry_id="belt",
            subsystem="gantry",
            name="XY Drive Belt Stiffness",
            active_selection=belt_sel,
            active_values=belt_vals,
            trust_pct=belt_conf["trust_pct"],
            override_group="belt",
            basis=belt_conf["basis"],
            caveat=belt_conf["caveat"],
            options_breakdown=belt_breakdown,
        ),
        _make_confidence_entry(
            spec,
            entry_id="belt_routing",
            subsystem="gantry",
            name="Belt Path & Idler Routing",
            active_selection=str(br_entry["name"]),
            active_values=br_vals,
            trust_pct=br_conf["trust_pct"],
            override_group="belt_routing",
            basis=br_conf["basis"],
            caveat=br_conf["caveat"],
            options_breakdown=br_breakdown,
        ),
        _make_confidence_entry(
            spec,
            entry_id="bearing_clearance",
            subsystem="gantry",
            name="Idler Bearing Play",
            active_selection=bc_sel,
            active_values=bc_vals,
            trust_pct=88,
            override_group="bearing_clearance",
            basis="ISO 5753 radial internal clearance ranges (C2 / CN / C3) for miniature ball bearings.",
            caveat="Bearing lot scatter, flange axial preload, and shaft press-fit interference shift real radial play.",
        ),
        _make_confidence_entry(
            spec,
            entry_id="gantry_bracket",
            subsystem="gantry",
            name="Gantry Brackets & Motor Mounts",
            active_selection=bkt_sel,
            active_values=bkt_vals,
            trust_pct=bkt_trust,
            override_group="gantry_bracket",
            basis=bkt_basis,
            caveat=bkt_caveat,
            options_breakdown=bkt_breakdown,
            cad_superelement_active=bkt_cad_active,
            cad_superelement_id=bkt_cad_kit_id if bkt_cad_active else None,
            cad_trust_pct=bkt_cad_trust,
        ),
        _make_confidence_entry(
            spec,
            entry_id="stepper_electromechanics",
            subsystem="gantry",
            name="Stepper Motor Electromechanics",
            active_selection=mot_sel,
            active_values=mot_vals,
            trust_pct=88,
            override_group=None,
            basis="Datasheet two-phase winding L/R pull-out curve & pole-pair electromagnetic torsional spring.",
            caveat="Driver microstepping, PWM chopper current ripple, and mid-band resonance damping vary ~15%.",
        ),
        _make_confidence_entry(
            spec,
            entry_id="toolhead_preset",
            subsystem="toolhead",
            name="Toolhead Preset (Mass & CoG)",
            active_selection=th_sel,
            active_values=th_vals,
            trust_pct=th_trust,
            override_group=None,
            basis=th_basis,
            caveat=th_caveat,
            options_breakdown=th_breakdown,
            cad_superelement_active=th_cad_active,
            cad_superelement_id=th_pid if th_cad_active else None,
            cad_trust_pct=th_cad_trust,
        ),
        _make_confidence_entry(
            spec,
            entry_id="toolhead_construction",
            subsystem="toolhead",
            name="Toolhead Body Construction",
            active_selection=tc_sel,
            active_values=tc_vals,
            trust_pct=tc_trust,
            override_group="toolhead_construction",
            basis=tc_basis,
            caveat=tc_caveat,
            options_breakdown=tc_breakdown,
            cad_superelement_active=th_cad_active,
            cad_superelement_id=th_pid if th_cad_active else None,
            cad_trust_pct=th_cad_trust,
        ),
        _make_confidence_entry(
            spec,
            entry_id="z_probe",
            subsystem="toolhead",
            name="Z-Probe / Carriage Interface",
            active_selection=str(zp_entry["name"]),
            active_values=zp_vals,
            trust_pct=zp_conf["trust_pct"],
            override_group="z_probe",
            basis=zp_conf["basis"],
            caveat=zp_conf["caveat"],
            options_breakdown=zp_breakdown,
        ),
        _make_confidence_entry(
            spec,
            entry_id="hotend_clamp",
            subsystem="hotend",
            name="Hotend Structural Clamp & Neck",
            active_selection=hc_sel,
            active_values=hc_vals,
            trust_pct=hc_trust,
            override_group="hotend_clamp",
            basis=hc_basis,
            caveat=hc_caveat,
            options_breakdown=hc_breakdown,
            cad_superelement_active=he_cad_active,
            cad_superelement_id=he_cad_id,
            cad_trust_pct=he_cad_trust,
        ),
        _make_confidence_entry(
            spec,
            entry_id="hotend_thermal",
            subsystem="hotend",
            name="Hotend Thermal & Melt Flow",
            active_selection=ht_sel,
            active_values=ht_vals,
            trust_pct=76,
            override_group=None,
            basis="40-node 1D FVM axial heat solver + Graetz melt conduction & Arrhenius shear-thinning flow.",
            caveat="Real filament brand viscosity/additives and heater-block contact resistance vary ±15–20%.",
        ),
        _make_confidence_entry(
            spec,
            entry_id="bed_plate",
            subsystem="bed",
            name="Heated Bed Plate Mechanics",
            active_selection=bed_sel,
            active_values=bed_vals,
            trust_pct=93,
            override_group=None,
            basis="9×9 MITC4 Mindlin-Reissner plate FEM & MIC-6 / ATP-5 tooling plate material constants.",
            caveat="Silicone heater pad, magnetic sheet, and PEI spring-steel plate add minor composite damping.",
        ),
        _make_confidence_entry(
            spec,
            entry_id="bed_mount",
            subsystem="bed",
            name="Bed Mount Hardware",
            active_selection=str(bm_entry["name"]),
            active_values=bm_vals,
            trust_pct=bm_conf["trust_pct"],
            override_group="bed_mount",
            basis=bm_conf["basis"],
            caveat=bm_conf["caveat"],
            options_breakdown=bm_breakdown,
        ),
        _make_confidence_entry(
            spec,
            entry_id="z_drive_mechanism",
            subsystem="bed",
            name="Z-Drive Mechanism",
            active_selection=zd_sel,
            active_values=zd_vals,
            trust_pct=zd_conf["trust_pct"],
            override_group="z_drive_mechanism",
            basis=zd_conf["basis"],
            caveat=zd_conf["caveat"],
            options_breakdown=zd_breakdown,
        ),
        _make_confidence_entry(
            spec,
            entry_id="shaper_and_solver",
            subsystem="solver",
            name="Klipper Input Shaper & Modal Solver",
            active_selection=solver_sel,
            active_values=solver_vals,
            trust_pct=96,
            override_group=None,
            basis="Exact generalized matrix eigensolver (K·φ = ω²·M·φ) & upstream Klipper shaper_defs.py convolution math.",
            caveat="Solver & shaper math are exact for the assembled model; real print ringing inherits upstream structural/belt estimates.",
        ),
    ]

    overall_trust_pct = int(
        round(sum(entry["trust_pct"] for entry in entries) / max(len(entries), 1))
    )

    return {
        "disclaimer_title": "Work in Progress — Physics Model Confidence & Assumptions",
        "disclaimer_body": (
            "PrintSim is an active work in progress. Textbook beam/plate mechanics and manufacturer "
            "datasheets (90–100%) are high confidence, whereas unmodeled assemblies like toolhead bodies, "
            "printed brackets, Tap rails, and benchtop feet rely on lumped engineering estimates (40–65%). "
            "Use results for comparative engineering trends, validate critical builds with ADXL345 "
            "measurements, and override any lumped parameter directly in Advanced Mode."
        ),
        "overall_trust_pct": overall_trust_pct,
        "entries": entries,
    }
