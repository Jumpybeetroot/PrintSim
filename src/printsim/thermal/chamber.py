"""Enclosed chamber thermodynamics, PIR insulation, CPAP blower/hose losses & Kalico shared duct."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from printsim.schema import (
    CHAMBER_HEATER_CATALOG,
    CPAP_BLOWER_CATALOG,
    PrinterSpec,
    get_material,
    get_profile,
)

# Strict SI thermodynamic & fluid-mechanics constants
_DM3_S_TO_CFM = 2.11888  # Exact: 1e-3 m^3/s * 35.3146667 ft^3/m^3 * 60 s/min
_AIR_RHO_KG_M3 = 1.184  # Dry air density near 25 C (kg/m^3)
_AIR_CP_J_KGK = 1005.0  # Dry air constant-pressure specific heat (J/(kg*K))

# Enclosure wall & PIR rigid foam constants (SI)
_K_PIR_W_MK = 0.023  # Closed-cell PIR rigid foam board thermal conductivity (W/(m*K))
_RHO_PIR_KG_M3 = 32.0  # PIR foam board density (kg/m^3)
_CP_PIR_J_KGK = 1400.0  # PIR foam specific heat capacity (J/(kg*K))
_H_IN_W_M2K = 8.0  # Internal enclosure convective + linearized radiative coefficient (W/(m^2*K))
_H_OUT_W_M2K = 7.0  # External room-side natural convection + radiation coefficient (W/(m^2*K))
_BRIDGE_AREA_FRAC = 0.22  # Frame extrusions, acrylic front door & door-seal parallel leakage fraction
_U_BRIDGE_W_M2K = 2.8  # Effective thermal-bridge conductance over extrusion/door perimeter (W/(m^2*K))
_U_OPEN_TOP_W_M2K = 35.0  # Open-top chimney convective exchange coefficient (W/(m^2*K))
_U_OPEN_FRAME_W_M2K = 45.0  # Unenclosed open-frame air exchange coefficient (W/(m^2*K))

# Corrugated CPAP hose Darcy-Weisbach hydraulic model constants (SI)
_CPAP_HOSE_LENGTH_M = 1.25  # Nominal routing length from electronics bay to toolhead (m)
_CPAP_HOSE_DARCY_F = 0.065  # Corrugated flexible tubing turbulent Darcy friction factor
_CPAP_HOSE_K_MINOR = 3.0  # Entry + two sweeping 90-deg bends + toolhead manifold elbow
_CPAP_REF_HOSE_ID_M = 0.018  # 18 mm reference standard CPAP hose internal diameter (m)
# Blower internal + toolhead nozzle exit duct impedance (Pa*s^2/m^6), calibrated so 15 mm hose
# delivers ~0.833 of 18 mm hose flow (matching Monolith's ~20% worst-case theoretical drop).
_CPAP_BASE_DUCT_IMPEDANCE_PA_S2_M6 = 1.38e8

# Kalico [heater_fan] shared heatsink CPAP duct enforcement
_KALICO_SHARED_DUCT_MIN_PWM_PCT = 25.0
_KALICO_HEATER_FAN_THRESHOLD_C = 50.0


@dataclass
class ChamberAndCpapReport:
    """Steady/transient enclosed chamber heat balance and remote CPAP blower/hose fluid-thermal report."""

    chamber_heater_id: str
    chamber_heater_power_w: float
    chamber_insulation_mm: float
    enclosure_ua_w_per_k: float
    chamber_steady_temp_c: float
    chamber_heatup_time_60c_s: float | None
    cpap_blower_id: str
    cpap_hose_id_mm: float
    cpap_pwm_pct: float
    cpap_effective_pwm_pct: float
    cpap_hose_flow_factor: float
    cpap_delivered_dm3_s: float
    cpap_delivered_cfm: float
    cpap_exhaust_temp_rise_c: float
    cpap_outlet_air_temp_c: float
    cpap_hose_moving_mass_g: float
    kalico_shared_duct_active: bool
    cpap_cooling_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chamber_heater_id": self.chamber_heater_id,
            "chamber_heater_power_w": round(self.chamber_heater_power_w, 1),
            "chamber_insulation_mm": round(self.chamber_insulation_mm, 1),
            "enclosure_ua_w_per_k": round(self.enclosure_ua_w_per_k, 3),
            "chamber_steady_temp_c": round(self.chamber_steady_temp_c, 1),
            "chamber_heatup_time_60c_s": (
                round(self.chamber_heatup_time_60c_s, 1)
                if self.chamber_heatup_time_60c_s is not None
                else None
            ),
            "cpap_blower_id": self.cpap_blower_id,
            "cpap_hose_id_mm": round(self.cpap_hose_id_mm, 1),
            "cpap_pwm_pct": round(self.cpap_pwm_pct, 1),
            "cpap_effective_pwm_pct": round(self.cpap_effective_pwm_pct, 1),
            "cpap_hose_flow_factor": round(self.cpap_hose_flow_factor, 3),
            "cpap_delivered_dm3_s": round(self.cpap_delivered_dm3_s, 2),
            "cpap_delivered_cfm": round(self.cpap_delivered_cfm, 2),
            "cpap_exhaust_temp_rise_c": round(self.cpap_exhaust_temp_rise_c, 2),
            "cpap_outlet_air_temp_c": round(self.cpap_outlet_air_temp_c, 1),
            "cpap_hose_moving_mass_g": round(self.cpap_hose_moving_mass_g, 1),
            "kalico_shared_duct_active": self.kalico_shared_duct_active,
            "cpap_cooling_notes": list(self.cpap_cooling_notes),
        }


def _cpap_hose_hydraulic_impedance(diameter_m: float) -> float:
    """Darcy-Weisbach major + minor hydraulic impedance R_hose = dP / Q^2 (Pa*s^2/m^6)."""
    d = max(diameter_m, 0.004)
    k_total = _CPAP_HOSE_DARCY_F * (_CPAP_HOSE_LENGTH_M / d) + _CPAP_HOSE_K_MINOR
    return k_total * (8.0 * _AIR_RHO_KG_M3) / ((math.pi**2) * (d**4))


def cpap_hose_flow_factor(hose_id_mm: float) -> float:
    """Relative volumetric flow factor vs 18 mm reference CPAP hose via Darcy-Weisbach + fan P-Q intersection.

    Returns 1.000 at 18 mm ID and ~0.833 at 15 mm ID (matching Monolith's ~20% worst-case drop).
    """
    r_ref = _CPAP_BASE_DUCT_IMPEDANCE_PA_S2_M6 + _cpap_hose_hydraulic_impedance(_CPAP_REF_HOSE_ID_M)
    r_cur = _CPAP_BASE_DUCT_IMPEDANCE_PA_S2_M6 + _cpap_hose_hydraulic_impedance(hose_id_mm * 1e-3)
    return math.sqrt(r_ref / max(r_cur, 1e-9))


def cpap_hose_moving_mass_g(hose_id_mm: float) -> float:
    """Estimated moving corrugated CPAP hose mass carried by the toolhead (g).

    Calibrated to Monolith/Voron scale data: ~18 g for 15 mm HT hose vs ~32 g for 18 mm standard hose.
    """
    d_mm = max(hose_id_mm, 5.0)
    return 18.0 * ((d_mm / 15.0) ** 3.15)


def _ptc_effective_power_w(nominal_power_w: float, chamber_temp_c: float) -> float:
    """Active PTC ceramic heater output derated smoothly near its Curie transition knee (~115 C)."""
    if nominal_power_w <= 0.0:
        return 0.0
    if chamber_temp_c <= 65.0:
        return nominal_power_w
    # Smooth self-regulating PTC resistance ramp centered around ~105-115 C air temperature
    curie_mid_c = 105.0
    scale_c = 10.0
    ref_65 = 1.0 / (1.0 + math.exp((65.0 - curie_mid_c) / scale_c))
    x = max(-20.0, min(20.0, (chamber_temp_c - curie_mid_c) / scale_c))
    cur = 1.0 / (1.0 + math.exp(x))
    return nominal_power_w * min(1.0, cur / ref_65)


def simulate_chamber_and_cpap(
    spec: PrinterSpec,
    hotend_target_temp_c: float = 250.0,
    room_temp_c: float = 22.0,
) -> ChamberAndCpapReport:
    """Evaluate enclosed chamber thermal equilibrium, 60 C heat-up transient, and CPAP fluid-thermal behavior."""
    frame = spec.frame
    th = spec.toolhead
    bed = spec.bed
    notes: list[str] = []

    # ------------------------------------------------------------------
    # 1. CPAP blower, Kalico shared heatsink duct & hose Darcy-Weisbach
    # ------------------------------------------------------------------
    blower_id = th.cpap_blower_id
    blower = CPAP_BLOWER_CATALOG.get(blower_id, CPAP_BLOWER_CATALOG["none"])
    hose_id_mm = float(th.cpap_hose_id_mm)
    pwm_cmd = max(0.0, min(100.0, float(th.cpap_pwm_pct)))

    if blower_id == "none":
        kalico_shared_active = False
        effective_pwm = 0.0
        hose_factor = 0.0
        delivered_dm3_s = 0.0
        delivered_cfm = 0.0
        exhaust_rise_c = 0.0
        moving_mass_g = 0.0
    else:
        kalico_shared_active = bool(
            th.shared_heatsink_cpap_duct and hotend_target_temp_c > _KALICO_HEATER_FAN_THRESHOLD_C
        )
        effective_pwm = (
            max(pwm_cmd, _KALICO_SHARED_DUCT_MIN_PWM_PCT) if kalico_shared_active else pwm_cmd
        )
        hose_factor = cpap_hose_flow_factor(hose_id_mm)
        delivered_dm3_s = float(blower["ducted_max_dm3_s"]) * (effective_pwm / 100.0) * hose_factor
        delivered_cfm = delivered_dm3_s * _DM3_S_TO_CFM
        exhaust_rise_c = float(blower["exhaust_heating_at_100pct_c"]) * ((effective_pwm / 100.0) ** 1.5)
        moving_mass_g = cpap_hose_moving_mass_g(hose_id_mm)

        if kalico_shared_active:
            notes.append(
                f"Kalico [heater_fan] shared heatsink CPAP duct active: enforcing >={_KALICO_SHARED_DUCT_MIN_PWM_PCT:.0f}% "
                f"minimum blower PWM while hotend ({hotend_target_temp_c:.0f} C) > {_KALICO_HEATER_FAN_THRESHOLD_C:.0f} C."
            )
        elif not th.shared_heatsink_cpap_duct and pwm_cmd <= 0.0:
            notes.append(
                "CPAP PWM is 0% (blower off); ensure an independent hotend heatsink fan is fitted "
                "or enable Kalico shared heatsink CPAP duct."
            )

        min_stable = float(blower.get("min_stable_pwm_pct", 0.0))
        if 0.0 < effective_pwm < min_stable:
            notes.append(
                f"Effective CPAP PWM ({effective_pwm:.0f}%) is below {blower['short_name']} "
                f"minimum stable threshold ({min_stable:.0f}%); airflow may stall or pulse."
            )

        if hose_id_mm <= 15.5:
            flow_drop_pct = max(0.0, (1.0 - hose_factor) * 100.0)
            mass_saved_g = max(0.0, cpap_hose_moving_mass_g(18.0) - moving_mass_g)
            notes.append(
                f"{hose_id_mm:.0f} mm HT CPAP hose reduces delivered flow by ~{flow_drop_pct:.0f}% "
                f"vs 18 mm hose while saving ~{mass_saved_g:.0f} g moving toolhead hose mass."
            )

        if blower.get("acoustic_profile") == "loud_hf":
            notes.append(
                "WM7575 AC blower exhibits loud high-frequency acoustic output; "
                "consider acoustic baffling or WM9290 24V for quieter operation."
            )

    # ------------------------------------------------------------------
    # 2. Enclosure thermal conductance UA (W/K) & PIR insulation
    # ------------------------------------------------------------------
    w_m = frame.width_mm * 1e-3
    d_m = frame.depth_mm * 1e-3
    h_m = frame.height_mm * 1e-3
    a_enc_m2 = 2.0 * (w_m * d_m + w_m * h_m + d_m * h_m)
    a_top_m2 = w_m * d_m
    vol_enc_m3 = w_m * d_m * h_m

    insulation_mm = max(0.0, float(frame.chamber_insulation_mm))
    t_pir_m = insulation_mm * 1e-3
    t_panel_m = frame.panel_thickness_mm * 1e-3

    try:
        panel_mat = get_material(frame.panel_material)
        k_panel = panel_mat.thermal_conductivity_w_mk
        rho_panel = panel_mat.density_kg_m3
        cp_panel = panel_mat.specific_heat_j_kgk
    except KeyError:
        k_panel = 0.20
        rho_panel = 1200.0
        cp_panel = 1200.0

    def _wall_u(mat_id: str, thick_mm: float) -> float:
        try:
            mat = get_material(mat_id)
            k_p = mat.thermal_conductivity_w_mk
        except KeyError:
            k_p = 0.20
        r_w = (
            (1.0 / _H_IN_W_M2K)
            + ((thick_mm * 1e-3) / max(k_p, 0.01))
            + (t_pir_m / _K_PIR_W_MK)
            + (1.0 / _H_OUT_W_M2K)
        )
        return 1.0 / r_w

    u_wall = _wall_u(frame.panel_material, frame.panel_thickness_mm)
    r_wall = 1.0 / u_wall

    if frame.enclosed_panels:
        if frame.has_mixed_panels:
            face_areas = [
                ("rear", w_m * h_m),
                ("left", d_m * h_m),
                ("right", d_m * h_m),
                ("front_bot", w_m * h_m + w_m * d_m),
            ]
            if frame.top_panel:
                face_areas.append(("top", a_top_m2))
            ua_paneled = 0.0
            a_paneled_m2 = 0.0
            for f_name, f_area in face_areas:
                if f_name == "front_bot":
                    u_f = u_wall
                else:
                    f_mat_id, f_thick_mm = frame.effective_panel_spec(f_name)
                    u_f = _wall_u(f_mat_id, f_thick_mm)
                u_eff_f = (1.0 - _BRIDGE_AREA_FRAC) * u_f + _BRIDGE_AREA_FRAC * _U_BRIDGE_W_M2K
                ua_paneled += f_area * u_eff_f
                a_paneled_m2 += f_area
            ua_w_per_k = ua_paneled if frame.top_panel else (ua_paneled + a_top_m2 * _U_OPEN_TOP_W_M2K)
        else:
            u_eff_paneled = (1.0 - _BRIDGE_AREA_FRAC) * u_wall + _BRIDGE_AREA_FRAC * _U_BRIDGE_W_M2K
            if frame.top_panel:
                ua_w_per_k = a_enc_m2 * u_eff_paneled
                a_paneled_m2 = a_enc_m2
            else:
                a_paneled_m2 = max(a_enc_m2 - a_top_m2, 0.0)
                ua_w_per_k = a_paneled_m2 * u_eff_paneled + a_top_m2 * _U_OPEN_TOP_W_M2K
    else:
        a_paneled_m2 = 0.0
        ua_w_per_k = a_enc_m2 * _U_OPEN_FRAME_W_M2K
        if insulation_mm > 0.0:
            notes.append(
                f"{insulation_mm:.0f} mm PIR insulation specified with enclosed_panels=False; "
                "install enclosure panels to retain chamber heat."
            )

    # ------------------------------------------------------------------
    # 3. Active PTC heater, bed convection & CPAP exhaust heat balance
    # ------------------------------------------------------------------
    heater_id = frame.chamber_heater_id
    heater = CHAMBER_HEATER_CATALOG.get(heater_id, CHAMBER_HEATER_CATALOG["none"])
    heater_nominal_w = float(heater["power_w"])

    a_bed_m2 = (bed.size_mm[0] * bed.size_mm[1]) * 1e-6
    p_bed_enclosed_w = min(
        bed.heater_power_w * 0.55,
        11.0 * a_bed_m2 * max(bed.target_temp_c - room_temp_c, 0.0),
    )
    p_bed_base_w = p_bed_enclosed_w if frame.enclosed_panels else (0.15 * p_bed_enclosed_w)

    # Recirculated CPAP blower compression/motor dissipation enters enclosed chamber air
    if frame.enclosed_panels and blower_id != "none" and delivered_dm3_s > 0.0:
        q_m3_s = delivered_dm3_s * 1e-3
        p_cpap_heat_w = _AIR_RHO_KG_M3 * q_m3_s * _AIR_CP_J_KGK * exhaust_rise_c
    else:
        p_cpap_heat_w = 0.0

    def _net_chamber_heat_w(temp_c: float) -> float:
        # Respect 2nd law when chamber rises above bed target temperature (smooth taper over 15 K band)
        if p_bed_base_w > 0.0:
            bed_taper = max(0.0, min(1.0, (bed.target_temp_c - temp_c + 15.0) / 15.0))
        else:
            bed_taper = 0.0
        p_bed_cur = p_bed_base_w * bed_taper
        p_ptc_cur = _ptc_effective_power_w(heater_nominal_w, temp_c)
        return p_bed_cur + p_ptc_cur + p_cpap_heat_w - ua_w_per_k * (temp_c - room_temp_c)

    # Unique equilibrium by monotonic bisection
    if _net_chamber_heat_w(room_temp_c) <= 0.0:
        t_steady_c = room_temp_c
    else:
        lo_c = room_temp_c
        hi_c = 260.0
        while _net_chamber_heat_w(hi_c) > 0.0 and hi_c < 500.0:
            hi_c += 100.0
        for _ in range(60):
            mid_c = 0.5 * (lo_c + hi_c)
            if _net_chamber_heat_w(mid_c) > 0.0:
                lo_c = mid_c
            else:
                hi_c = mid_c
        t_steady_c = 0.5 * (lo_c + hi_c)

    # ------------------------------------------------------------------
    # 4. Lumped thermal capacitance C_ch (J/K) & transient heat-up to 60 C
    # ------------------------------------------------------------------
    c_air_j_k = _AIR_RHO_KG_M3 * vol_enc_m3 * _AIR_CP_J_KGK

    try:
        frame_mat = get_material(frame.material)
        rho_frame = frame_mat.density_kg_m3
        cp_frame = frame_mat.specific_heat_j_kgk
    except KeyError:
        rho_frame = 2700.0
        cp_frame = 900.0

    try:
        horiz_prof = get_profile(frame.profile)
        upright_prof = get_profile(frame.effective_upright_profile)
        area_horiz = horiz_prof.area_m2
        area_upright = upright_prof.area_m2
    except KeyError:
        area_horiz = 1.83e-4
        area_upright = 1.83e-4

    m_frame_kg = rho_frame * (4.0 * (w_m + d_m) * area_horiz + 4.0 * h_m * area_upright)
    c_frame_coupled_j_k = 0.25 * m_frame_kg * cp_frame

    if frame.enclosed_panels and a_paneled_m2 > 0.0:
        # Internal PIR decouples outer panel thermal mass via exact steady mid-plane temperature fraction
        panel_temp_frac = ((1.0 / _H_OUT_W_M2K) + 0.5 * t_panel_m / max(k_panel, 0.01)) / r_wall
        m_panels_kg = a_paneled_m2 * t_panel_m * rho_panel
        c_panels_j_k = 0.65 * panel_temp_frac * m_panels_kg * cp_panel
        m_pir_kg = (1.0 - _BRIDGE_AREA_FRAC) * a_paneled_m2 * t_pir_m * _RHO_PIR_KG_M3
        c_pir_j_k = 0.50 * m_pir_kg * _CP_PIR_J_KGK
    else:
        c_panels_j_k = 0.0
        c_pir_j_k = 0.0

    # Internal rails/gantry/mounts coupled to chamber air (~900 J/K on 300 mm CoreXY, scaled by area)
    c_internal_j_k = 900.0 * ((max(vol_enc_m3, 1e-3) / 0.1058) ** (2.0 / 3.0))
    c_chamber_j_k = max(c_air_j_k + c_frame_coupled_j_k + c_panels_j_k + c_pir_j_k + c_internal_j_k, 100.0)

    target_60c = 60.0
    if room_temp_c >= target_60c:
        heatup_60c_s: float | None = 0.0
    elif t_steady_c < target_60c + 0.05:
        heatup_60c_s = None
    else:
        # RK2 (Heun midpoint) transient integration with exact linear crossing interpolation
        dt_s = 1.0
        t_cur = room_temp_c
        heatup_60c_s = None
        for step in range(14400):
            k1 = _net_chamber_heat_w(t_cur) / c_chamber_j_k
            t_mid = t_cur + 0.5 * dt_s * k1
            k2 = _net_chamber_heat_w(t_mid) / c_chamber_j_k
            t_next = t_cur + dt_s * k2
            if t_next >= target_60c:
                frac = (target_60c - t_cur) / max(t_next - t_cur, 1e-12)
                heatup_60c_s = (step + max(0.0, min(1.0, frac))) * dt_s
                break
            t_cur = t_next

    # ------------------------------------------------------------------
    # 5. CPAP outlet air temperature (intake air + exhaust compression heat)
    # ------------------------------------------------------------------
    intake_temp_c = t_steady_c if frame.enclosed_panels else room_temp_c
    outlet_air_temp_c = intake_temp_c + exhaust_rise_c

    if blower_id != "none" and exhaust_rise_c >= 3.0:
        notes.append(
            f"{blower['short_name']} adds +{exhaust_rise_c:.1f} C exhaust heating at "
            f"{effective_pwm:.0f}% PWM (outlet air {outlet_air_temp_c:.1f} C)."
        )

    return ChamberAndCpapReport(
        chamber_heater_id=heater_id,
        chamber_heater_power_w=heater_nominal_w,
        chamber_insulation_mm=insulation_mm,
        enclosure_ua_w_per_k=ua_w_per_k,
        chamber_steady_temp_c=t_steady_c,
        chamber_heatup_time_60c_s=heatup_60c_s,
        cpap_blower_id=blower_id,
        cpap_hose_id_mm=hose_id_mm,
        cpap_pwm_pct=pwm_cmd,
        cpap_effective_pwm_pct=effective_pwm,
        cpap_hose_flow_factor=hose_factor,
        cpap_delivered_dm3_s=delivered_dm3_s,
        cpap_delivered_cfm=delivered_cfm,
        cpap_exhaust_temp_rise_c=exhaust_rise_c,
        cpap_outlet_air_temp_c=outlet_air_temp_c,
        cpap_hose_moving_mass_g=moving_mass_g,
        kalico_shared_duct_active=kalico_shared_active,
        cpap_cooling_notes=notes,
    )
