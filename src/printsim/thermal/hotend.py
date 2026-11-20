"""Finite-volume axial hotend thermal solver, heat-creep analysis, rheology & part cooling."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solve

from printsim.schema import PrinterSpec, get_material
from printsim.thermal.filaments import get_filament


@dataclass
class HotendThermalReport:
    filament_id: str
    filament_name: str
    glass_transition_c: float
    target_temp_c: float
    heatup_time_215c_s: float
    bed_heatup_time_100c_s: float
    steady_heat_loss_w: float
    heatbreak_thermal_resistance_k_w: float
    cold_side_temp_c: float
    heat_creep_risk: bool
    max_volumetric_flow_mm3_s: float
    flow_limiting_factor: str  # "heater_power", "melt_zone_conduction", "extruder_pressure"
    max_linear_speed_02x04_mm_s: float
    extruder_pressure_margin: float
    max_overhang_deg: float
    bridge_sag_40mm_mm: float
    heatup_curve: list[dict[str, float]] = field(default_factory=list)
    axial_profile: list[dict] = field(default_factory=list)  # [{z_mm, temp_c, zone}]

    def to_dict(self) -> dict:
        return {
            "filament_id": self.filament_id,
            "filament_name": self.filament_name,
            "glass_transition_c": round(self.glass_transition_c, 1),
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
            "max_overhang_deg": round(self.max_overhang_deg, 1),
            "bridge_sag_40mm_mm": round(self.bridge_sag_40mm_mm, 2),
            "heatup_curve": self.heatup_curve,
            "axial_profile": self.axial_profile,
        }


def _solve_axial_hotend_fvm(
    spec: PrinterSpec,
    target_temp_c: float,
    k_block: float,
    k_nozzle: float,
    k_hb: float,
    area_hb: float,
    r_hs: float,
) -> list[dict]:
    """Solve 1D finite-volume steady heat conduction + fin convection along the hotend Z axis.

    Zones (from nozzle tip z=0 upward to heatsink top):
      0..5 mm: Nozzle tip & cone
      5..5+L_melt: Heater block (Dirichlet heated core)
      5+L_melt..5+L_melt+L_hb: Heatbreak throat (thin-walled thermal barrier)
      ..+22 mm: Finned heatsink (convective cooling to ambient)
    """
    he = spec.toolhead.hotend
    t_amb = he.ambient_temp_c
    z_noz = 5.0
    z_block_end = z_noz + he.melt_zone_length_mm
    z_hb_end = z_block_end + he.heatbreak_length_mm
    z_hs_len = 22.0
    z_total = z_hb_end + z_hs_len

    n = 40
    z_mm = np.linspace(0.0, z_total, n)
    dz = (z_total * 1e-3) / (n - 1)

    # Effective cross-section conductance k*A (W*m/K) and convective loss g_conv (W/K per node)
    kA = np.zeros(n, dtype=float)
    g_amb = np.zeros(n, dtype=float)
    zones: list[str] = []

    a_noz = math.pi * (0.0035**2 - 0.001**2)
    a_block = 0.016 * 0.016
    a_hs = math.pi * (0.008**2 - 0.001**2)
    k_hs = 170.0  # Aluminum heatsink

    hs_indices: list[int] = []
    block_indices: list[int] = []

    for i, z in enumerate(z_mm):
        if z < z_noz:
            zones.append("nozzle")
            kA[i] = k_nozzle * a_noz
            g_amb[i] = 0.004
        elif z <= z_block_end:
            zones.append("block")
            kA[i] = k_block * a_block
            g_amb[i] = 0.003
            block_indices.append(i)
        elif z <= z_hb_end:
            zones.append("heatbreak")
            # Include filament core (~0.25 W/mK) in parallel with metal wall
            kA[i] = k_hb * area_hb + 0.25 * (math.pi * (0.000875**2))
            g_amb[i] = 0.002
        else:
            zones.append("heatsink")
            kA[i] = k_hs * a_hs
            hs_indices.append(i)

    # Distribute total heatsink convective conductance 1/R_hs uniformly across heatsink nodes
    if hs_indices:
        per_hs_g = (1.0 / max(r_hs, 0.2)) / len(hs_indices)
        for idx in hs_indices:
            g_amb[idx] = per_hs_g

    # Assemble tridiagonal FVM system A T = b
    A = np.zeros((n, n), dtype=float)
    b = np.zeros(n, dtype=float)
    for i in range(n):
        if i > 0:
            g_left = 0.5 * (kA[i - 1] + kA[i]) / dz
            A[i, i] += g_left
            A[i, i - 1] -= g_left
        if i < n - 1:
            g_right = 0.5 * (kA[i] + kA[i + 1]) / dz
            A[i, i] += g_right
            A[i, i + 1] -= g_right
        A[i, i] += g_amb[i]
        b[i] += g_amb[i] * t_amb

    # Enforce heated block temperature via high-conductance penalty (heater cartridge control)
    heater_g = 1e4
    for idx in block_indices:
        A[idx, idx] += heater_g
        b[idx] += heater_g * target_temp_c

    # Add small radiant heat coupling from heater block into bottom of heatsink
    if hs_indices:
        q_rad = 0.016 * max(target_temp_c - t_amb, 0.0)
        b[hs_indices[0]] += q_rad * 0.65
        if len(hs_indices) > 1:
            b[hs_indices[1]] += q_rad * 0.35

    temps = solve(A, b)
    return [
        {"z_mm": round(float(z_mm[i]), 2), "temp_c": round(float(temps[i]), 1), "zone": zones[i]}
        for i in range(n)
    ]


def simulate_hotend_thermal(
    spec: PrinterSpec,
    target_temp_c: float | None = None,
    filament_id: str = "pla",
) -> HotendThermalReport:
    """Evaluate hotend FVM thermal profile, heat-creep margin, rheology melt flow, and cooling."""
    he = spec.toolhead.hotend
    ext = spec.toolhead.extruder
    fil = get_filament(filament_id)
    block_mat = get_material(he.block_material)
    nozzle_mat = get_material(he.nozzle_material)
    hb_mat = get_material(he.heatbreak_material)

    t_target = target_temp_c if target_temp_c is not None else fil.print_temp_c
    t_amb = he.ambient_temp_c
    delta_t = max(t_target - t_amb, 1.0)

    # 1. Heatbreak axial thermal resistance R_hb = L / (k * A)
    r_in = (he.heatbreak_id_mm * 1e-3) * 0.5
    r_out = r_in + he.heatbreak_wall_mm * 1e-3
    area_hb = math.pi * (r_out**2 - r_in**2)
    r_hb = (he.heatbreak_length_mm * 1e-3) / max(hb_mat.thermal_conductivity_w_mk * area_hb, 1e-9)

    # 2. Steady heat leak through heatbreak + parasitic radiation/filament conduction into heatsink
    r_hs = he.heatsink_thermal_resistance_k_w
    q_hb_w = delta_t / (r_hb + r_hs)
    q_parasitic_w = 0.016 * delta_t
    t_cold = t_amb + (q_hb_w + q_parasitic_w) * r_hs
    heat_creep_risk = t_cold > (fil.glass_transition_c - 6.0)

    # Solve 1D axial FVM temperature profile
    axial_profile = _solve_axial_hotend_fvm(
        spec=spec,
        target_temp_c=t_target,
        k_block=block_mat.thermal_conductivity_w_mk,
        k_nozzle=nozzle_mat.thermal_conductivity_w_mk,
        k_hb=hb_mat.thermal_conductivity_w_mk,
        area_hb=area_hb,
        r_hs=r_hs,
    )

    # Block external convective + radiative loss
    q_conv_w = 0.032 * delta_t
    steady_loss_w = q_hb_w + q_conv_w

    # 3. Transient thermal capacitance & heat-up simulation
    c_block = (he.block_mass_g * 1e-3) * block_mat.specific_heat_j_kgk + 0.005 * nozzle_mat.specific_heat_j_kgk + 2.5
    t_cur = t_amb
    dt = 0.25
    heatup_time = 300.0
    reached = False
    curve: list[dict[str, float]] = [{"time_s": 0.0, "temp_c": round(t_amb, 1)}]

    for step in range(1, 1200):
        t_sec = step * dt
        err = t_target - t_cur
        duty = max(0.0, min(1.0, err / 8.0)) if reached else 1.0
        p_in = he.heater_power_w * duty
        p_loss = (t_cur - t_amb) * (1.0 / (r_hb + r_hs) + 0.032)
        t_cur += ((p_in - p_loss) / c_block) * dt
        if not reached and t_cur >= t_target - 0.5:
            heatup_time = t_sec
            reached = True
        if step % 8 == 0 and t_sec <= max(heatup_time + 15.0, 45.0):
            curve.append({"time_s": round(t_sec, 1), "temp_c": round(t_cur, 1)})
        if reached and t_sec > heatup_time + 15.0:
            break

    # 4. Bed heat-up time (ambient to filament bed_temp_c or 100C benchmark)
    bed_target = max(fil.bed_temp_c, 60.0)
    bed_mat = get_material(spec.bed.plate_material)
    c_bed = (spec.bed.mass_g * 1e-3) * bed_mat.specific_heat_j_kgk
    bed_area_m2 = (spec.bed.size_mm[0] * spec.bed.size_mm[1]) * 1e-6
    h_bed_loss = 14.0 * bed_area_m2
    t_bed = t_amb
    bed_time = 600.0
    for step in range(1, 2400):
        p_loss_b = (t_bed - t_amb) * h_bed_loss
        t_bed += ((spec.bed.heater_power_w - p_loss_b) / max(c_bed, 10.0)) * 0.5
        if t_bed >= 100.0:
            bed_time = step * 0.5
            break

    # 5. Volumetric flow rate limits (mm^3/s) with polymer rheology
    enthalpy_j_per_mm3 = (
        fil.density_kg_m3 * 1e-9 * (fil.specific_heat_j_kgk * delta_t + fil.latent_heat_j_kg)
    )
    avail_melt_power_w = max(he.heater_power_w * 0.88 - steady_loss_w, 1.0)
    flow_power = avail_melt_power_w / enthalpy_j_per_mm3

    k_eff = 2.0 * block_mat.thermal_conductivity_w_mk * nozzle_mat.thermal_conductivity_w_mk / (
        block_mat.thermal_conductivity_w_mk + nozzle_mat.thermal_conductivity_w_mk
    )
    k_factor = (k_eff / 115.0) ** 0.28
    nozzle_factor = (he.nozzle_diameter_mm / 0.4) ** 0.45
    # Higher enthalpy polymers need slightly longer residence time in melt zone (PLA ref ~ 0.48 J/mm^3)
    pol_therm_factor = (0.48 / max(enthalpy_j_per_mm3, 0.05)) ** 0.35
    flow_melt = 1.48 * he.melt_zone_length_mm * k_factor * nozzle_factor * pol_therm_factor

    # Non-Newtonian shear-thinning capillary pressure limit (Q ~ (P / K_visc)^(1/n) * d^(3 + 1/n))
    fil_area_mm2 = math.pi * (ext.filament_diameter_mm * 0.5) ** 2
    max_pressure_mpa = ext.max_push_force_n / fil_area_mm2
    visc_scale = 1.0 / (fil.viscosity_factor ** (1.0 / max(fil.power_law_n * 2.2, 0.5)))
    flow_pressure = 1.55 * max_pressure_mpa * visc_scale * ((he.nozzle_diameter_mm / 0.4) ** 2.2)

    limits = {
        "heater_power": flow_power,
        "melt_zone_conduction": flow_melt,
        "extruder_pressure": flow_pressure,
    }
    limiting_factor = min(limits, key=limits.get)  # type: ignore[arg-type]
    max_flow = limits[limiting_factor]

    pressure_margin = flow_pressure / max(min(flow_power, flow_melt), 1.0)
    max_lin_speed = max_flow / (0.20 * 0.42)

    # 6. Part-cooling overhang & bridging capability
    eff_cooling = he.part_cooling_cfm / max(fil.cooling_demand, 0.2)
    max_overhang = min(78.0, max(38.0, 42.0 + 18.0 * math.log1p(eff_cooling * 0.55)))
    bridge_sag = max(0.05, 1.45 / (1.0 + 0.38 * eff_cooling))

    return HotendThermalReport(
        filament_id=fil.id,
        filament_name=fil.name,
        glass_transition_c=fil.glass_transition_c,
        target_temp_c=t_target,
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
        max_overhang_deg=max_overhang,
        bridge_sag_40mm_mm=bridge_sag,
        heatup_curve=curve,
        axial_profile=axial_profile,
    )
