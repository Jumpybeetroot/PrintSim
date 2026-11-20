"""Lumped-parameter hotend thermal network, heat-creep analysis, and volumetric flow model."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from printsim.schema import PrinterSpec, get_material


@dataclass
class HotendThermalReport:
    target_temp_c: float
    heatup_time_215c_s: float
    bed_heatup_time_100c_s: float
    steady_heat_loss_w: float
    heatbreak_thermal_resistance_k_w: float
    cold_side_temp_c: float
    heat_creep_risk: bool
    max_volumetric_flow_mm3_s: float
    flow_limiting_factor: str  # "heater_power", "melt_zone_conduction", "extruder_pressure"
    max_linear_speed_02x04_mm_s: float  # equivalent print speed at 0.2mm layer x 0.42mm width
    extruder_pressure_margin: float
    heatup_curve: list[dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "target_temp_c": round(self.target_temp_c, 1),
            "heatup_time_215c_s": round(self.heatup_time_215c_s, 1),
            "bed_heatup_time_100c_s": round(self.bed_heatup_time_100c_s, 1),
            "steady_heat_loss_w": round(self.steady_heat_loss_w, 2),
            "heatbreak_thermal_resistance_k_w": round(self.heatbreak_thermal_resistance_k_w, 1),
            "cold_side_temp_c": round(self.cold_side_temp_c, 1),
            "heat_creep_risk": self.heat_creep_risk,
            "max_volumetric_flow_mm3_s": round(self.max_volumetric_flow_mm3_s, 1),
            "flow_limiting_factor": self.flow_limiting_factor,
            "max_linear_speed_02x04_mm_s": round(self.max_linear_speed_02x04_mm_s, 0),
            "extruder_pressure_margin": round(self.extruder_pressure_margin, 2),
            "heatup_curve": self.heatup_curve,
        }


def simulate_hotend_thermal(
    spec: PrinterSpec,
    target_temp_c: float = 215.0,
    filament_density_kg_m3: float = 1240.0,  # PLA default
    filament_cp_j_kgk: float = 1800.0,
    filament_latent_j_kg: float = 45000.0,
) -> HotendThermalReport:
    """Evaluate hotend thermal dynamics, heat-creep margin, and max volumetric melt flow."""
    he = spec.toolhead.hotend
    ext = spec.toolhead.extruder
    block_mat = get_material(he.block_material)
    nozzle_mat = get_material(he.nozzle_material)
    hb_mat = get_material(he.heatbreak_material)

    t_amb = he.ambient_temp_c
    delta_t = max(target_temp_c - t_amb, 1.0)

    # 1. Heatbreak axial thermal resistance R_hb = L / (k * A)
    r_in = (he.heatbreak_id_mm * 1e-3) * 0.5
    r_out = r_in + he.heatbreak_wall_mm * 1e-3
    area_hb = math.pi * (r_out**2 - r_in**2)
    r_hb = (he.heatbreak_length_mm * 1e-3) / max(hb_mat.thermal_conductivity_w_mk * area_hb, 1e-9)

    # 2. Steady heat leak through heatbreak + parasitic radiation/filament conduction into heatsink
    r_hs = he.heatsink_thermal_resistance_k_w
    q_hb_w = delta_t / (r_hb + r_hs)
    q_parasitic_w = 0.016 * delta_t  # filament core conduction & block-to-heatsink radiation
    t_cold = t_amb + (q_hb_w + q_parasitic_w) * r_hs
    heat_creep_risk = t_cold > 52.0  # PLA softens around 55-60C inside cold throat

    # Block external convective + radiative loss (~0.045 W/K bare, ~0.028 W/K with silicone sock)
    q_conv_w = 0.032 * delta_t
    steady_loss_w = q_hb_w + q_conv_w

    # 3. Transient thermal capacitance & heat-up simulation (forward Euler 0.25s steps)
    # Include heater block + nozzle + ~3g steel/ceramic cartridge overhead
    c_block = (he.block_mass_g * 1e-3) * block_mat.specific_heat_j_kgk + 0.005 * nozzle_mat.specific_heat_j_kgk + 2.5
    t_cur = t_amb
    dt = 0.25
    t_sec = 0.0
    heatup_time = 300.0
    reached = False
    curve: list[dict[str, float]] = [{"time_s": 0.0, "temp_c": round(t_amb, 1)}]

    for step in range(1, 1200):
        t_sec = step * dt
        # Proportional-limited heater power near setpoint (simulating PID ramp)
        err = target_temp_c - t_cur
        duty = max(0.0, min(1.0, err / 8.0)) if reached else 1.0
        p_in = he.heater_power_w * duty
        p_loss = (t_cur - t_amb) * (1.0 / (r_hb + r_hs) + 0.032)
        t_cur += ((p_in - p_loss) / c_block) * dt
        if not reached and t_cur >= target_temp_c - 0.5:
            heatup_time = t_sec
            reached = True
        if step % 8 == 0 and t_sec <= max(heatup_time + 15.0, 45.0):
            curve.append({"time_s": round(t_sec, 1), "temp_c": round(t_cur, 1)})
        if reached and t_sec > heatup_time + 15.0:
            break

    # 4. Bed heat-up time (ambient to 100C)
    bed_mat = get_material(spec.bed.plate_material)
    c_bed = (spec.bed.mass_g * 1e-3) * bed_mat.specific_heat_j_kgk
    bed_area_m2 = (spec.bed.size_mm[0] * spec.bed.size_mm[1]) * 1e-6
    h_bed_loss = 14.0 * bed_area_m2  # ~14 W/(m^2*K) combined top/bottom convection
    t_bed = t_amb
    bed_time = 600.0
    for step in range(1, 2400):
        p_loss_b = (t_bed - t_amb) * h_bed_loss
        t_bed += ((spec.bed.heater_power_w - p_loss_b) / max(c_bed, 10.0)) * 0.5
        if t_bed >= 100.0:
            bed_time = step * 0.5
            break

    # 5. Volumetric flow rate limits (mm^3/s)
    # 5a. Heater power limit (enthalpy required per mm^3 of melted polymer)
    enthalpy_j_per_mm3 = (
        filament_density_kg_m3 * 1e-9 * (filament_cp_j_kgk * delta_t + filament_latent_j_kg)
    )
    avail_melt_power_w = max(he.heater_power_w * 0.88 - steady_loss_w, 1.0)
    flow_power = avail_melt_power_w / enthalpy_j_per_mm3

    # 5b. Melt-zone conductive heat transfer limit (calibrated to real V6 / Rapido / Volcano data)
    # Standard 11.5mm brass V6 -> ~13-15 mm^3/s; 16mm copper -> ~24-28 mm^3/s; 24mm UHF -> ~42+ mm^3/s
    k_eff = 2.0 * block_mat.thermal_conductivity_w_mk * nozzle_mat.thermal_conductivity_w_mk / (
        block_mat.thermal_conductivity_w_mk + nozzle_mat.thermal_conductivity_w_mk
    )
    # Polymer internal thermal resistance dominates, modulated weakly by nozzle/block wall conductivity
    k_factor = (k_eff / 115.0) ** 0.28
    # Nozzle orifice also improves flow slightly for wider nozzles
    nozzle_factor = (he.nozzle_diameter_mm / 0.4) ** 0.45
    flow_melt = 1.48 * he.melt_zone_length_mm * k_factor * nozzle_factor

    # 5c. Extruder push force / pressure drop limit (shear-thinning capillary approximation)
    # Force F = P * A_fil, where P ~ eta * L_orifice * Q / d_noz^3.2 (power-law n~0.4)
    fil_area_mm2 = math.pi * (ext.filament_diameter_mm * 0.5) ** 2
    max_pressure_mpa = ext.max_push_force_n / fil_area_mm2  # N / mm^2 = MPa
    # Calibrated so 50N on 0.4mm nozzle supports ~32 mm^3/s before extruder skipping
    flow_pressure = 1.55 * max_pressure_mpa * ((he.nozzle_diameter_mm / 0.4) ** 2.2)

    limits = {
        "heater_power": flow_power,
        "melt_zone_conduction": flow_melt,
        "extruder_pressure": flow_pressure,
    }
    limiting_factor = min(limits, key=limits.get)  # type: ignore[arg-type]
    max_flow = limits[limiting_factor]

    # Extruder pressure margin at nominal melt-limited flow
    pressure_margin = flow_pressure / max(min(flow_power, flow_melt), 1.0)

    # Equivalent max linear print speed for standard 0.20mm layer x 0.42mm extrusion width
    max_lin_speed = max_flow / (0.20 * 0.42)

    return HotendThermalReport(
        target_temp_c=target_temp_c,
        heatup_time_215c_s=heatup_time,
        bed_heatup_time_100c_s=bed_time,
        steady_heat_loss_w=steady_loss_w,
        heatbreak_thermal_resistance_k_w=r_hb,
        cold_side_temp_c=t_cold,
        heat_creep_risk=heat_creep_risk,
        max_volumetric_flow_mm3_s=max_flow,
        flow_limiting_factor=limiting_factor,
        max_linear_speed_02x04_mm_s=max_lin_speed,
        extruder_pressure_margin=pressure_margin,
        heatup_curve=curve,
    )
