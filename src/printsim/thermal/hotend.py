"""Finite-volume axial hotend thermal solver, heat-creep analysis, rheology & part cooling."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solve

from printsim.schema import PrinterSpec, get_material
from printsim.thermal.filaments import get_filament


_R_GAS_J_MOLK = 8.3144626
# Reconciled total convective + radiative conductance from nozzle tip (0.020 W/K) and heater block (0.030 W/K)
_NOZZLE_AMB_LOSS_W_K = 0.020
_BLOCK_AMB_LOSS_W_K = 0.030
_BLOCK_NOZZLE_LOSS_W_K = _NOZZLE_AMB_LOSS_W_K + _BLOCK_AMB_LOSS_W_K


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
    melt_zone_flow_limit_mm3_s: float = 0.0
    heater_power_flow_limit_mm3_s: float = 0.0
    extruder_pressure_flow_limit_mm3_s: float = 0.0
    required_push_force_n: float = 0.0
    nozzle_pressure_drop_mpa: float = 0.0

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
            "melt_zone_flow_limit_mm3_s": round(self.melt_zone_flow_limit_mm3_s, 1),
            "heater_power_flow_limit_mm3_s": round(self.heater_power_flow_limit_mm3_s, 1),
            "extruder_pressure_flow_limit_mm3_s": round(self.extruder_pressure_flow_limit_mm3_s, 1),
            "required_push_force_n": round(self.required_push_force_n, 2),
            "nozzle_pressure_drop_mpa": round(self.nozzle_pressure_drop_mpa, 2),
            "heatup_curve": self.heatup_curve,
            "axial_profile": self.axial_profile,
        }


def solve_1d_fvm_conduction(
    cell_lengths_m: np.ndarray,
    cell_kA: np.ndarray,
    g_amb_w_k: np.ndarray | None = None,
    t_amb_c: float = 25.0,
    left_dirichlet_temp_c: float | None = None,
    right_dirichlet_temp_c: float | None = None,
    penalty_nodes: dict[int, tuple[float, float]] | None = None,
    sources_w: np.ndarray | None = None,
) -> np.ndarray:
    """Solve steady 1D finite-volume conduction with Patankar harmonic face conductance.

    Face conductance between adjacent control volumes i and i+1 uses exact series
    half-cell thermal resistance (Patankar 1980, Sec 4.2.3):
        G_{i,i+1} = 1 / (dz_i / (2 * kA_i) + dz_{i+1} / (2 * kA_{i+1}))
    """
    n = len(cell_lengths_m)
    A = np.zeros((n, n), dtype=float)
    b = np.zeros(n, dtype=float)

    for i in range(n - 1):
        r_face = (0.5 * cell_lengths_m[i]) / max(float(cell_kA[i]), 1e-12) + (
            0.5 * cell_lengths_m[i + 1]
        ) / max(float(cell_kA[i + 1]), 1e-12)
        g_face = 1.0 / max(r_face, 1e-15)
        A[i, i] += g_face
        A[i, i + 1] -= g_face
        A[i + 1, i + 1] += g_face
        A[i + 1, i] -= g_face

    if g_amb_w_k is not None:
        for i in range(n):
            g_a = float(g_amb_w_k[i])
            if g_a > 0.0:
                A[i, i] += g_a
                b[i] += g_a * t_amb_c

    if left_dirichlet_temp_c is not None:
        g_left = (2.0 * max(float(cell_kA[0]), 1e-12)) / max(float(cell_lengths_m[0]), 1e-12)
        A[0, 0] += g_left
        b[0] += g_left * left_dirichlet_temp_c

    if right_dirichlet_temp_c is not None:
        g_right = (2.0 * max(float(cell_kA[-1]), 1e-12)) / max(float(cell_lengths_m[-1]), 1e-12)
        A[-1, -1] += g_right
        b[-1] += g_right * right_dirichlet_temp_c

    if penalty_nodes:
        for idx, (g_pen, t_set) in penalty_nodes.items():
            A[idx, idx] += g_pen
            b[idx] += g_pen * t_set

    if sources_w is not None:
        b += sources_w

    return solve(A, b)


def _solve_axial_hotend_fvm(
    spec: PrinterSpec,
    target_temp_c: float,
    k_block: float,
    k_nozzle: float,
    k_hb: float,
    area_hb: float,
    r_hs: float,
    k_filament: float = 0.18,
    fil_diameter_m: float = 0.00175,
) -> list[dict]:
    """Solve 1D finite-volume steady heat conduction + fin convection along the hotend Z axis.

    Zones (40 zone-aligned control volumes from nozzle tip z=0 upward to heatsink top):
      0..5 mm (5 cells): Nozzle tip & cone
      5..5+L_melt (10 cells): Heater block (Dirichlet heated core)
      5+L_melt..5+L_melt+L_hb (12 cells): Heatbreak throat (thin-walled thermal barrier)
      ..+22 mm (13 cells): Finned heatsink (convective cooling to ambient)
    """
    he = spec.toolhead.hotend
    t_amb = max(he.ambient_temp_c, spec.frame.chamber_temp_c)
    z_noz_mm = 5.0
    z_melt_mm = he.melt_zone_length_mm
    z_hb_mm = he.heatbreak_length_mm
    z_hs_mm = 22.0

    zone_counts = [
        ("nozzle", z_noz_mm, 5),
        ("block", z_melt_mm, 10),
        ("heatbreak", z_hb_mm, 12),
        ("heatsink", z_hs_mm, 13),
    ]
    n = sum(cnt for _, _, cnt in zone_counts)

    dz_m = np.zeros(n, dtype=float)
    z_mm = np.zeros(n, dtype=float)
    kA = np.zeros(n, dtype=float)
    g_amb = np.zeros(n, dtype=float)
    zones: list[str] = []

    a_noz = math.pi * (0.0035**2 - 0.001**2)
    a_block = 0.016 * 0.016
    a_hs = math.pi * (0.008**2 - 0.001**2)
    k_hs = 170.0  # Aluminum heatsink
    a_fil = math.pi * (0.5 * fil_diameter_m) ** 2

    hs_indices: list[int] = []
    block_indices: list[int] = []

    idx = 0
    z_cursor_mm = 0.0
    for zone_name, length_mm, count in zone_counts:
        cell_mm = length_mm / count
        cell_m = cell_mm * 1e-3
        for j in range(count):
            dz_m[idx] = cell_m
            z_mm[idx] = z_cursor_mm + (j + 0.5) * cell_mm
            zones.append(zone_name)
            if zone_name == "nozzle":
                kA[idx] = k_nozzle * a_noz
                g_amb[idx] = _NOZZLE_AMB_LOSS_W_K / count
            elif zone_name == "block":
                kA[idx] = k_block * a_block
                g_amb[idx] = _BLOCK_AMB_LOSS_W_K / count
                block_indices.append(idx)
            elif zone_name == "heatbreak":
                kA[idx] = k_hb * area_hb + k_filament * a_fil
                g_amb[idx] = 0.002
            else:
                kA[idx] = k_hs * a_hs
                hs_indices.append(idx)
            idx += 1
        z_cursor_mm += length_mm

    # Distribute total heatsink convective conductance 1/R_hs uniformly across heatsink nodes
    if hs_indices:
        per_hs_g = (1.0 / max(r_hs, 0.2)) / len(hs_indices)
        for hs_idx in hs_indices:
            g_amb[hs_idx] = per_hs_g

    heater_g = 1e4
    penalty_nodes = {b_idx: (heater_g, target_temp_c) for b_idx in block_indices}

    sources_w = np.zeros(n, dtype=float)
    if hs_indices:
        q_rad = 0.016 * max(target_temp_c - t_amb, 0.0)
        sources_w[hs_indices[0]] += q_rad * 0.65
        if len(hs_indices) > 1:
            sources_w[hs_indices[1]] += q_rad * 0.35

    temps = solve_1d_fvm_conduction(
        cell_lengths_m=dz_m,
        cell_kA=kA,
        g_amb_w_k=g_amb,
        t_amb_c=t_amb,
        penalty_nodes=penalty_nodes,
        sources_w=sources_w,
    )
    return [
        {"z_mm": round(float(z_mm[i]), 2), "temp_c": round(float(temps[i]), 1), "zone": zones[i]}
        for i in range(n)
    ]


def graetz_melt_zone_flow_m3_s(
    melt_zone_length_m: float,
    thermal_diffusivity_m2_s: float,
    t_nozzle_c: float,
    t_inlet_c: float,
    t_core_c: float,
    k_block_w_mk: float = 385.0,
    k_nozzle_w_mk: float = 115.0,
    melt_transfer_factor: float = 1.0,
) -> float:
    """Compute Graetz radial-conduction melt-zone volumetric flow limit (m^3/s).

    Q_melt = C_core * f_mat * f_wall * (pi * Nu * alpha * L) / ln((T_nozzle - T_in) / (T_nozzle - T_core))
    where Nu = 3.66 (circular tube constant wall temperature), C_core = 3.30 accounts for
    partially melted PLA core shear-film extrusion on a melt-averaged DSC cp basis
    (Go et al. 2017; Mackay 2018), and melt_transfer_factor (f_mat) scales per polymer.
    """
    nu = 3.66
    c_melt_core = 3.30 * max(melt_transfer_factor, 0.05)
    alpha = max(thermal_diffusivity_m2_s, 1e-9)
    k_eff = (2.0 * k_block_w_mk * k_nozzle_w_mk) / max(k_block_w_mk + k_nozzle_w_mk, 1e-9)
    wall_factor = (k_eff / 150.0) ** 0.12

    t_crit = t_core_c + 0.5
    if t_nozzle_c >= t_crit:
        dt_num = max(t_nozzle_c - t_inlet_c, 1.0)
        dt_den = t_nozzle_c - t_core_c
        log_term = max(math.log(dt_num / dt_den), 0.15)
        return c_melt_core * wall_factor * (math.pi * nu * alpha * melt_zone_length_m) / log_term

    # Smooth positive monotonic taper when nozzle is at or below core melt threshold
    dt_num_crit = max(t_crit - t_inlet_c, 1.0)
    log_crit = max(math.log(dt_num_crit / 0.5), 0.15)
    q_crit = c_melt_core * wall_factor * (math.pi * nu * alpha * melt_zone_length_m) / log_crit
    return q_crit * math.exp(0.15 * (t_nozzle_c - t_crit))


def arrhenius_viscosity_shift(temp_c: float, t_ref_c: float, ea_j_mol: float) -> float:
    """Arrhenius shift factor a_T = exp((Ea / R) * (1 / T_K - 1 / T_ref_K))."""
    t_k = max(temp_c + 273.15, 200.0)
    t_ref_k = max(t_ref_c + 273.15, 200.0)
    exponent = (ea_j_mol / _R_GAS_J_MOLK) * (1.0 / t_k - 1.0 / t_ref_k)
    return math.exp(max(-20.0, min(20.0, exponent)))


def rabinowitsch_wall_shear_stress_pa(
    q_m3_s: float, radius_m: float, k_pa_sn: float, n: float
) -> float:
    """Wall shear stress tau_w = K * (gamma_a * (3n + 1) / (4n))^n in a circular pipe (Pa)."""
    r = max(radius_m, 1e-6)
    gamma_a = (4.0 * max(q_m3_s, 0.0)) / (math.pi * (r**3))
    gamma_w = gamma_a * (3.0 * n + 1.0) / (4.0 * n)
    return k_pa_sn * (gamma_w**n)


def rabinowitsch_pipe_pressure_drop_pa(
    q_m3_s: float, radius_m: float, length_m: float, k_pa_sn: float, n: float
) -> float:
    """Exact Rabinowitsch-corrected power-law pipe pressure drop dP = 2 * L * tau_w / R (Pa)."""
    tau_w = rabinowitsch_wall_shear_stress_pa(q_m3_s, radius_m, k_pa_sn, n)
    return (2.0 * max(length_m, 0.0) * tau_w) / max(radius_m, 1e-6)


def convergent_cone_pressure_drop_pa(
    q_m3_s: float,
    r_in_m: float,
    r_out_m: float,
    k_pa_sn: float,
    n: float,
    half_angle_deg: float = 30.0,
) -> float:
    """Power-law lubrication pressure drop in a tapered nozzle cone (Bellini 2004; Osswald 2018)."""
    r_in = max(r_in_m, r_out_m + 1e-9)
    r_out = max(r_out_m, 1e-6)
    tan_b = math.tan(math.radians(half_angle_deg))
    return (
        (2.0 * k_pa_sn / (3.0 * n * tan_b))
        * ((((3.0 * n + 1.0) * max(q_m3_s, 0.0)) / (math.pi * n)) ** n)
        * (r_out ** (-3.0 * n) - r_in ** (-3.0 * n))
    )


def nozzle_total_pressure_drop_pa(
    q_m3_s: float,
    nozzle_diameter_m: float,
    melt_zone_length_m: float,
    chamber_diameter_m: float,
    k_pa_sn: float,
    n: float,
    orifice_length_m: float = 0.0010,
    cone_half_angle_deg: float = 30.0,
    bagley_factor: float = 3.0,
) -> float:
    """Total capillary pressure drop: chamber + convergent cone + orifice + Bagley entry (Pa)."""
    r_orifice = max(0.5 * nozzle_diameter_m, 1e-6)
    r_chamber = max(0.5 * chamber_diameter_m, r_orifice + 1e-6)
    dp_chamber = rabinowitsch_pipe_pressure_drop_pa(
        q_m3_s, r_chamber, melt_zone_length_m, k_pa_sn, n
    )
    dp_cone = convergent_cone_pressure_drop_pa(
        q_m3_s, r_chamber, r_orifice, k_pa_sn, n, cone_half_angle_deg
    )
    tau_w_orifice = rabinowitsch_wall_shear_stress_pa(q_m3_s, r_orifice, k_pa_sn, n)
    dp_orifice = (2.0 * orifice_length_m * tau_w_orifice) / r_orifice
    dp_entry = bagley_factor * tau_w_orifice
    return dp_chamber + dp_cone + dp_orifice + dp_entry


def solve_pressure_limited_flow_m3_s(
    max_pressure_pa: float,
    nozzle_diameter_m: float,
    melt_zone_length_m: float,
    chamber_diameter_m: float,
    k_pa_sn: float,
    n: float,
    orifice_length_m: float = 0.0010,
) -> float:
    """Solve dP_total(Q) = max_pressure_pa by monotonic bisection (m^3/s)."""
    if max_pressure_pa <= 0.0:
        return 0.0
    lo = 0.0
    hi = 1e-7  # 100 mm^3/s initial upper bracket
    while nozzle_total_pressure_drop_pa(
        hi, nozzle_diameter_m, melt_zone_length_m, chamber_diameter_m, k_pa_sn, n, orifice_length_m
    ) < max_pressure_pa:
        hi *= 2.0
        if hi > 1e-2:  # 10,000,000 mm^3/s safety cap
            break
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        dp_mid = nozzle_total_pressure_drop_pa(
            mid, nozzle_diameter_m, melt_zone_length_m, chamber_diameter_m, k_pa_sn, n, orifice_length_m
        )
        if dp_mid < max_pressure_pa:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


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
    t_amb = max(he.ambient_temp_c, spec.frame.chamber_temp_c)
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
    heat_creep_risk = t_cold > (fil.effective_heat_creep_temp_c - 6.0)

    # Solve 1D axial FVM temperature profile using Patankar harmonic conductances & filament k
    axial_profile = _solve_axial_hotend_fvm(
        spec=spec,
        target_temp_c=t_target,
        k_block=block_mat.thermal_conductivity_w_mk,
        k_nozzle=nozzle_mat.thermal_conductivity_w_mk,
        k_hb=hb_mat.thermal_conductivity_w_mk,
        area_hb=area_hb,
        r_hs=r_hs,
        k_filament=fil.thermal_conductivity_w_mk,
        fil_diameter_m=ext.filament_diameter_mm * 1e-3,
    )

    # Block + nozzle external convective & radiative loss (reconciled with 0.050 W/K FVM conductance)
    q_conv_w = _BLOCK_NOZZLE_LOSS_W_K * delta_t
    steady_loss_w = q_hb_w + q_conv_w

    # 3. Transient thermal capacitance & heat-up simulation to target_temp_c
    #    (stored in heatup_time_215c_s for API/UI schema compatibility)
    c_block = (he.block_mass_g * 1e-3) * block_mat.specific_heat_j_kgk + 0.005 * nozzle_mat.specific_heat_j_kgk + 2.5
    t_cur = t_amb
    dt = 0.25
    heatup_time = 0.0 if t_cur >= t_target - 0.5 else 300.0
    reached = t_cur >= t_target - 0.5
    curve: list[dict[str, float]] = [{"time_s": 0.0, "temp_c": round(t_amb, 1)}]

    for step in range(1, 1200):
        t_sec = step * dt
        err = t_target - t_cur
        duty = max(0.0, min(1.0, err / 8.0)) if reached else 1.0
        p_in = he.heater_power_w * duty
        p_loss = (t_cur - t_amb) * (1.0 / (r_hb + r_hs) + _BLOCK_NOZZLE_LOSS_W_K)
        t_cur += ((p_in - p_loss) / c_block) * dt
        if not reached and t_cur >= t_target - 0.5:
            heatup_time = t_sec
            reached = True
        if step % 8 == 0 and t_sec <= max(heatup_time + 15.0, 45.0):
            curve.append({"time_s": round(t_sec, 1), "temp_c": round(t_cur, 1)})
        if reached and t_sec > heatup_time + 15.0:
            break

    # 4. Bed heat-up time (ambient to filament bed_target = max(fil.bed_temp_c, 60.0))
    #    (stored in bed_heatup_time_100c_s for API/UI schema compatibility)
    bed_target = max(fil.bed_temp_c, 60.0)
    bed_mat = get_material(spec.bed.plate_material)
    c_bed = (spec.bed.mass_g * 1e-3) * bed_mat.specific_heat_j_kgk
    bed_area_m2 = (spec.bed.size_mm[0] * spec.bed.size_mm[1]) * 1e-6
    h_bed_loss = 14.0 * bed_area_m2
    t_bed = t_amb
    if t_bed >= bed_target:
        bed_time = 0.0
    else:
        bed_time = 1200.0
        for step in range(1, 2400):
            p_loss_b = (t_bed - t_amb) * h_bed_loss
            t_bed += ((spec.bed.heater_power_w - p_loss_b) / max(c_bed, 10.0)) * 0.5
            if t_bed >= bed_target:
                bed_time = step * 0.5
                break

    # 5. Volumetric flow rate limits (strict SI internally, mm^3/s at boundary)
    enthalpy_j_per_m3 = fil.density_kg_m3 * (fil.specific_heat_j_kgk * delta_t + fil.latent_heat_j_kg)
    avail_melt_power_w = max(he.heater_power_w * 0.88 - steady_loss_w, 1.0)
    flow_power_m3_s = avail_melt_power_w / max(enthalpy_j_per_m3, 1.0)

    flow_melt_m3_s = graetz_melt_zone_flow_m3_s(
        melt_zone_length_m=he.melt_zone_length_mm * 1e-3,
        thermal_diffusivity_m2_s=fil.thermal_diffusivity_m2_s,
        t_nozzle_c=t_target,
        t_inlet_c=t_amb,
        t_core_c=fil.extrudable_temp_c,
        k_block_w_mk=block_mat.thermal_conductivity_w_mk,
        k_nozzle_w_mk=nozzle_mat.thermal_conductivity_w_mk,
        melt_transfer_factor=fil.melt_transfer_factor,
    )

    # Non-Newtonian Rabinowitsch power-law capillary flow with Arrhenius T-shift
    fil_diameter_m = ext.filament_diameter_mm * 1e-3
    fil_area_m2 = math.pi * (0.5 * fil_diameter_m) ** 2
    max_pressure_pa = ext.max_push_force_n / fil_area_m2
    a_t = arrhenius_viscosity_shift(t_target, fil.t_ref_c, fil.arrhenius_ea_j_mol)
    k_melt_pa_sn = fil.power_law_k_pa_sn * (a_t ** fil.power_law_n)
    nozzle_d_m = he.nozzle_diameter_mm * 1e-3
    chamber_d_m = he.heatbreak_id_mm * 1e-3
    melt_l_m = he.melt_zone_length_mm * 1e-3

    flow_pressure_m3_s = solve_pressure_limited_flow_m3_s(
        max_pressure_pa=max_pressure_pa,
        nozzle_diameter_m=nozzle_d_m,
        melt_zone_length_m=melt_l_m,
        chamber_diameter_m=chamber_d_m,
        k_pa_sn=k_melt_pa_sn,
        n=fil.power_law_n,
    )

    flow_power = flow_power_m3_s * 1e9
    flow_melt = flow_melt_m3_s * 1e9
    flow_pressure = flow_pressure_m3_s * 1e9

    limits = {
        "heater_power": flow_power,
        "melt_zone_conduction": flow_melt,
        "extruder_pressure": flow_pressure,
    }
    limiting_factor = min(limits, key=limits.get)  # type: ignore[arg-type]
    max_flow = limits[limiting_factor]

    dp_at_max_pa = nozzle_total_pressure_drop_pa(
        q_m3_s=max_flow * 1e-9,
        nozzle_diameter_m=nozzle_d_m,
        melt_zone_length_m=melt_l_m,
        chamber_diameter_m=chamber_d_m,
        k_pa_sn=k_melt_pa_sn,
        n=fil.power_law_n,
    )
    req_push_force_n = dp_at_max_pa * fil_area_m2

    pressure_margin = flow_pressure / max(min(flow_power, flow_melt), 1.0)
    max_lin_speed = max_flow / (0.20 * 0.40)

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
        melt_zone_flow_limit_mm3_s=flow_melt,
        heater_power_flow_limit_mm3_s=flow_power,
        extruder_pressure_flow_limit_mm3_s=flow_pressure,
        required_push_force_n=req_push_force_n,
        nozzle_pressure_drop_mpa=dp_at_max_pa * 1e-6,
    )


analyze_hotend_thermal = simulate_hotend_thermal
