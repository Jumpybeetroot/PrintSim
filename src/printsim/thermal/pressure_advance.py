"""Filament-path compliance, pressure-advance constant, and extrusion transient model.

The hotend is a viscous orifice fed through a spring: push force ``F(v)`` (coupled
Cross-WLF + Graetz nozzle model in ``hotend.py``) rises with filament speed ``v``, and the
filament path between the drive gear and the nozzle stores ``x = C * F`` of filament as
elastic compression. The state equation is

    dx/dt = v_in(t) - v_out(x / C)

where ``v_out`` is the inverse of ``F(v)``. At steady state ``x = C * F(v)``, which is the
exact (nonlinear) advance curve ``A(v)``; the equivalent linear Klipper constant at a given
speed is ``A(v) / v`` (PrintDB ``references/dynamics/nonlinear-pressure-advance-kalico``).

Compliance terms (series, all first-principles; PrintDB
``references/dynamics/pressure-advance-extruder-flow`` and
``references/fluid_cooling/filament-drive-buckling-bowden-friction``):

* solid filament column ``L / (E * A_fil)``
* PTFE Bowden tube stretch ``L_tube / (E_ptfe * A_tube)`` (the tube carries the reaction force)
* confined melt and above-Tg polymer ``V / (K_bulk * A_fil^2)`` with the two-domain Tait
  melt bulk modulus (PrintDB ``references/thermal/thermal-properties-vs-temperature-pvt``)

Not modelled: helical/sinusoidal buckling of the filament inside straight tube runs,
drive-gear tooth indentation, stepper torque-angle compliance, capstan friction hysteresis,
and softening of the filament column in a heated chamber. Each of these adds compliance, so
predictions are a lower bound on the tuned constant.

Strict SI internally; friendly suffixed units only in ``to_dict()``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from printsim.schema import PrinterSpec, get_material
from printsim.thermal.filaments import FilamentSpec, get_filament

if TYPE_CHECKING:  # pragma: no cover - typing only
    from printsim.thermal.hotend import HotendThermalReport

# Two-domain Tait melt coefficients for extruded ABS (PrintDB materials/generic-abs `tait_pvt`,
# Kazmer et al. 2021 as tabulated by Kazmer & Peterson). No Tait set was found for any other
# filament polymer, so the ABS melt set is used as a proxy for them (flagged in the report).
TAIT_UNIVERSAL_C = 0.0894
TAIT_ABS_B3M_PA = 3.036e8
TAIT_ABS_B4M_PER_K = 2.490e-3
TAIT_ABS_B5_K = 376.75

# PTFE Bowden tube: 4 mm OD pneumatic tubing, E from PrintDB materials/ptfe-gasket-sheet.
PTFE_TUBE_MODULUS_PA = 0.5e9
BOWDEN_TUBE_OD_M = 4.0e-3

# Klipper's documented tuning-tower condition for a 0.4 mm nozzle (100 mm/s, 0.3 mm layer,
# ~0.4 mm line) is about 12 mm^3/s; used as the default flow at which the linear constant
# is quoted.
DEFAULT_REFERENCE_FLOW_MM3_S = 12.0
# Reference bead cross-section for the extruder velocity spike: 1.125 x nozzle wide, 0.5 x nozzle tall.
_BEAD_WIDTH_RATIO = 1.125
_BEAD_HEIGHT_RATIO = 0.5
_FORCE_TABLE_POINTS = 32
_ADVANCE_CURVE_FRACTIONS: tuple[float, ...] = (0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 1.00)


def tait_melt_bulk_modulus_pa(
    temp_c: float,
    pressure_pa: float = 0.0,
    b3m_pa: float = TAIT_ABS_B3M_PA,
    b4m_per_k: float = TAIT_ABS_B4M_PER_K,
    b5_k: float = TAIT_ABS_B5_K,
) -> float:
    """Isothermal melt bulk modulus K = 1 / beta_T from the two-domain Tait equation (Pa).

    ``B(T) = b3m * exp(-b4m * (T - b5))`` and ``v(T, p) = v0(T) * [1 - C ln(1 + p / B)]`` give
    ``beta_T = -(1/v) dv/dp = C / ((B + p) * [1 - C ln(1 + p / B)])``.
    """
    t_k = temp_c + 273.15
    b_pa = b3m_pa * math.exp(-b4m_per_k * (t_k - b5_k))
    p_pa = max(pressure_pa, 0.0)
    return (b_pa + p_pa) * (1.0 - TAIT_UNIVERSAL_C * math.log1p(p_pa / b_pa)) / TAIT_UNIVERSAL_C


@dataclass(frozen=True)
class FilamentPathCompliance:
    """Series axial compliance of the filament path, referred to filament travel at the drive gear."""

    filament_column_m_per_n: float
    bowden_tube_m_per_n: float
    melt_m_per_n: float
    bowden_slack_m: float
    compressible_volume_m3: float
    melt_bulk_modulus_pa: float
    melt_bulk_modulus_source: str
    column_buckling_load_n: float

    @property
    def total_m_per_n(self) -> float:
        return self.filament_column_m_per_n + self.bowden_tube_m_per_n + self.melt_m_per_n


def compute_filament_path_compliance(
    spec: PrinterSpec,
    fil: FilamentSpec,
    target_temp_c: float,
    cold_side_temp_c: float,
    mean_pressure_pa: float = 0.0,
    bowden_bend_angle_deg: float = 180.0,
) -> FilamentPathCompliance:
    """Filament column, Bowden tube, and confined-melt compliance (m of filament per N of push force)."""
    he = spec.toolhead.hotend
    ext = spec.toolhead.extruder
    d_fil_m = ext.filament_diameter_mm * 1e-3
    a_fil_m2 = math.pi * (0.5 * d_fil_m) ** 2
    is_bowden = ext.type == "bowden"

    # 1. Solid filament column between drive-gear nip and heat break inlet
    column_len_m = ext.filament_path_length_mm * 1e-3
    if is_bowden:
        column_len_m += ext.bowden_tube_length_mm * 1e-3
    c_column = column_len_m / (fil.youngs_modulus_pa * a_fil_m2)
    # Euler load of the guided (fixed-fixed, K = 0.5) run between drive gear and heat break.
    # Above it the filament bows against its guide and the straight-column term no longer holds.
    i_fil_m4 = math.pi * d_fil_m**4 / 64.0
    p_cr_n = (
        math.pi**2 * fil.youngs_modulus_pa * i_fil_m4 / (0.5 * ext.filament_path_length_mm * 1e-3) ** 2
    )

    # 2. Bowden tube stretch (tube is in tension by the same force) and bend-clearance dead band
    c_tube = 0.0
    slack_m = 0.0
    if is_bowden:
        tube_id_m = ext.bowden_tube_id_mm * 1e-3
        tube_len_m = ext.bowden_tube_length_mm * 1e-3
        a_tube_m2 = 0.25 * math.pi * max(BOWDEN_TUBE_OD_M**2 - tube_id_m**2, 1e-9)
        c_tube = tube_len_m / (PTFE_TUBE_MODULUS_PA * a_tube_m2)
        slack_m = math.radians(max(bowden_bend_angle_deg, 0.0)) * max(tube_id_m - d_fil_m, 0.0)

    # 3. Confined polymer above Tg: melt zone bore plus the hot fraction of the heat break,
    #    taken from a linear axial gradient between the cold side and the block.
    bore_area_m2 = 0.25 * math.pi * (he.heatbreak_id_mm * 1e-3) ** 2
    span_c = max(target_temp_c - cold_side_temp_c, 1.0)
    hot_fraction = min(1.0, max(0.0, (target_temp_c - fil.glass_transition_c) / span_c))
    volume_m3 = bore_area_m2 * (
        he.melt_zone_length_mm * 1e-3 + hot_fraction * he.heatbreak_length_mm * 1e-3
    )
    k_bulk_pa = tait_melt_bulk_modulus_pa(target_temp_c, mean_pressure_pa)
    c_melt = volume_m3 / (k_bulk_pa * a_fil_m2**2)

    return FilamentPathCompliance(
        filament_column_m_per_n=c_column,
        bowden_tube_m_per_n=c_tube,
        melt_m_per_n=c_melt,
        bowden_slack_m=slack_m,
        compressible_volume_m3=volume_m3,
        melt_bulk_modulus_pa=k_bulk_pa,
        melt_bulk_modulus_source="tait_abs" if fil.id == "abs" else "tait_abs_proxy",
        column_buckling_load_n=p_cr_n,
    )


def build_push_force_table(
    spec: PrinterSpec,
    fil: FilamentSpec,
    target_temp_c: float,
    nozzle_type: str,
    max_flow_m3_s: float,
    n_points: int = _FORCE_TABLE_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Monotonic table of filament speed (m/s) vs push force (N) from the coupled nozzle model.

    Starts at (0, 0) and spans geometrically to ``max_flow_m3_s``.
    """
    # In-function import: hotend.py imports nothing from this module, but keep the edge one-way.
    from printsim.thermal.hotend import coupled_melt_pressure_drop_pa, graetz_characteristic_flow_m3_s

    he = spec.toolhead.hotend
    ext = spec.toolhead.extruder
    block_mat = get_material(he.block_material)
    nozzle_mat = get_material(he.nozzle_material)
    a_fil_m2 = math.pi * (0.5 * ext.filament_diameter_mm * 1e-3) ** 2
    nozzle_d_m = he.nozzle_diameter_mm * 1e-3
    chamber_d_m = he.heatbreak_id_mm * 1e-3
    melt_l_m = he.melt_zone_length_mm * 1e-3
    t_amb = max(he.ambient_temp_c, spec.frame.chamber_temp_c)
    q_th_char = graetz_characteristic_flow_m3_s(
        melt_zone_length_m=melt_l_m,
        thermal_diffusivity_m2_s=fil.thermal_diffusivity_m2_s,
        k_block_w_mk=block_mat.thermal_conductivity_w_mk,
        k_nozzle_w_mk=nozzle_mat.thermal_conductivity_w_mk,
        melt_transfer_factor=fil.melt_transfer_factor,
        nozzle_type=nozzle_type,
    )
    q_max = max(max_flow_m3_s, 1e-12)
    flows = np.concatenate(([0.0], np.geomspace(q_max * 2e-3, q_max, max(n_points - 1, 4))))
    forces = np.zeros_like(flows)
    for i in range(1, flows.size):
        res = coupled_melt_pressure_drop_pa(
            q_m3_s=float(flows[i]),
            nozzle_diameter_m=nozzle_d_m,
            melt_zone_length_m=melt_l_m,
            chamber_diameter_m=chamber_d_m,
            t_wall_c=target_temp_c,
            t_inlet_c=t_amb,
            q_th_char_m3_s=q_th_char,
            fil=fil,
            nozzle_type=nozzle_type,
        )
        forces[i] = res["dp_total_pa"] * a_fil_m2
    forces = np.maximum.accumulate(forces)
    # Strictly increasing so the table is invertible
    forces += np.arange(forces.size) * 1e-12
    return flows / a_fil_m2, forces


def depressurization_time_s(
    compliance_m_per_n: float,
    speed_table_m_s: np.ndarray,
    force_table_n: np.ndarray,
    start_speed_m_s: float,
    end_fraction: float = 0.05,
) -> float:
    """Time for nozzle outflow to decay from ``start_speed_m_s`` to ``end_fraction`` of it after a hard stop.

    With ``v_in = 0`` the state equation gives ``C dF = -v dt``, so ``t = C * integral(dF / v)``
    between the two forces. For a linear nozzle ``F = R v`` this is ``R C ln(1 / end_fraction)``.
    """
    v_hi = max(start_speed_m_s, 0.0)
    v_lo = v_hi * min(max(end_fraction, 1e-6), 1.0)
    if v_hi <= 0.0 or v_lo >= v_hi:
        return 0.0
    v = np.geomspace(v_lo, v_hi, 200)
    f = np.interp(v, speed_table_m_s, force_table_n)
    return float(compliance_m_per_n * np.sum(0.5 * (1.0 / v[1:] + 1.0 / v[:-1]) * np.diff(f)))


@dataclass
class ExtrusionTransientResult:
    """Time histories of a filament-path pressure transient (SI)."""

    time_s: np.ndarray
    nominal_speed_m_s: np.ndarray
    commanded_speed_m_s: np.ndarray
    outflow_speed_m_s: np.ndarray
    compression_m: np.ndarray

    def volume_error_m3(self, filament_area_m2: float) -> np.ndarray:
        """Cumulative extruded-minus-nominal volume (negative = under-extruded so far)."""
        dt = np.diff(self.time_s, prepend=self.time_s[0])
        return np.cumsum((self.outflow_speed_m_s - self.nominal_speed_m_s) * dt) * filament_area_m2


def _triangular_smooth(values: np.ndarray, dt_s: float, smooth_time_s: float) -> np.ndarray:
    """Klipper symmetric triangular kernel of full width ``smooth_time_s`` (edge-padded)."""
    half = int(round(0.5 * smooth_time_s / dt_s))
    if half < 1:
        return values
    kernel = (half + 1 - np.abs(np.arange(-half, half + 1))).astype(float)
    kernel /= kernel.sum()
    padded = np.pad(values, half, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def simulate_extrusion_transient(
    time_s: np.ndarray,
    filament_speed_m_s: np.ndarray,
    compliance_m_per_n: float,
    speed_table_m_s: np.ndarray,
    force_table_n: np.ndarray,
    pressure_advance_s: float = 0.0,
    smooth_time_s: float = 0.0,
    initial_compression_m: float = 0.0,
) -> ExtrusionTransientResult:
    """Integrate ``dx/dt = v_in - v_out(x / C)`` over a nominal filament-speed profile.

    ``v_in = v + pressure_advance_s * dv/dt`` (Klipper linear advance), optionally smoothed with
    Klipper's triangular kernel. Uses an exponential integrator on the locally linearised
    nozzle, which is exact for a linear nozzle with piecewise-constant input and
    unconditionally stable otherwise. ``time_s`` must be uniformly spaced.
    """
    t = np.asarray(time_s, dtype=float)
    v_nom = np.asarray(filament_speed_m_s, dtype=float)
    if t.size != v_nom.size or t.size < 2:
        raise ValueError("time_s and filament_speed_m_s must have the same length (>= 2)")
    dt = float(t[1] - t[0])
    if dt <= 0.0 or not np.allclose(np.diff(t), dt, rtol=1e-6, atol=1e-12):
        raise ValueError("time_s must be uniformly spaced and increasing")
    c = max(compliance_m_per_n, 1e-15)
    v_tab = np.asarray(speed_table_m_s, dtype=float)
    f_tab = np.asarray(force_table_n, dtype=float)

    v_cmd = v_nom.copy()
    if pressure_advance_s != 0.0:
        v_cmd = v_nom + pressure_advance_s * np.gradient(v_nom, dt)
        if smooth_time_s > 0.0:
            v_cmd = _triangular_smooth(v_cmd, dt, smooth_time_s)

    # Segment slopes dF/dv for the local time constant tau = C * dF/dv
    slopes = np.diff(f_tab) / np.maximum(np.diff(v_tab), 1e-18)
    last_slope = float(slopes[-1])

    def outflow_and_slope(x_m: float) -> tuple[float, float]:
        f_n = x_m / c
        if f_n >= f_tab[-1]:
            return float(v_tab[-1] + (f_n - f_tab[-1]) / last_slope), last_slope
        idx = int(np.searchsorted(f_tab, f_n, side="right")) - 1
        idx = min(max(idx, 0), slopes.size - 1)
        return float(v_tab[idx] + (f_n - f_tab[idx]) / slopes[idx]), float(slopes[idx])

    x = np.empty_like(t)
    v_out = np.empty_like(t)
    x[0] = max(initial_compression_m, 0.0)
    v_out[0], _ = outflow_and_slope(float(x[0]))
    for i in range(1, t.size):
        v_prev, slope = outflow_and_slope(float(x[i - 1]))
        tau = max(c * slope, 1e-12)
        v_in = 0.5 * (v_cmd[i - 1] + v_cmd[i])
        # The melt cannot be pulled back out of the nozzle: compression floors at zero.
        x[i] = max(x[i - 1] + (v_in - v_prev) * tau * (1.0 - math.exp(-dt / tau)), 0.0)
        v_out[i], _ = outflow_and_slope(float(x[i]))

    return ExtrusionTransientResult(
        time_s=t,
        nominal_speed_m_s=v_nom,
        commanded_speed_m_s=v_cmd,
        outflow_speed_m_s=v_out,
        compression_m=x,
    )


@dataclass
class PressureAdvanceReport:
    filament_id: str
    extruder_type: str
    reference_flow_mm3_s: float
    reference_filament_speed_mm_s: float
    push_force_at_reference_n: float
    pressure_advance_s: float  # secant A(v) / v at the reference flow (Klipper linear constant)
    pressure_advance_tangent_s: float  # local slope dA/dv = pressure time constant at the reference flow
    system_compliance_um_per_n: float
    filament_column_compliance_um_per_n: float
    bowden_tube_compliance_um_per_n: float
    melt_compliance_um_per_n: float
    system_stiffness_n_per_mm: float
    melt_bulk_modulus_mpa: float
    melt_bulk_modulus_source: str
    stored_filament_compression_mm: float
    ooze_volume_mm3: float
    depressurization_time_s: float
    bowden_slack_mm: float
    min_retraction_mm: float
    reference_accel_mm_s2: float
    pa_extruder_velocity_spike_mm_s: float
    column_buckling_load_n: float
    straight_column_valid: bool
    advance_curve: list[dict[str, float]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "filament_id": self.filament_id,
            "extruder_type": self.extruder_type,
            "reference_flow_mm3_s": round(self.reference_flow_mm3_s, 2),
            "reference_filament_speed_mm_s": round(self.reference_filament_speed_mm_s, 3),
            "push_force_at_reference_n": round(self.push_force_at_reference_n, 2),
            "pressure_advance_s": round(self.pressure_advance_s, 4),
            "pressure_advance_tangent_s": round(self.pressure_advance_tangent_s, 4),
            "system_compliance_um_per_n": round(self.system_compliance_um_per_n, 2),
            "filament_column_compliance_um_per_n": round(self.filament_column_compliance_um_per_n, 2),
            "bowden_tube_compliance_um_per_n": round(self.bowden_tube_compliance_um_per_n, 2),
            "melt_compliance_um_per_n": round(self.melt_compliance_um_per_n, 2),
            "system_stiffness_n_per_mm": round(self.system_stiffness_n_per_mm, 1),
            "melt_bulk_modulus_mpa": round(self.melt_bulk_modulus_mpa, 0),
            "melt_bulk_modulus_source": self.melt_bulk_modulus_source,
            "stored_filament_compression_mm": round(self.stored_filament_compression_mm, 4),
            "ooze_volume_mm3": round(self.ooze_volume_mm3, 3),
            "depressurization_time_s": round(self.depressurization_time_s, 4),
            "bowden_slack_mm": round(self.bowden_slack_mm, 3),
            "min_retraction_mm": round(self.min_retraction_mm, 3),
            "reference_accel_mm_s2": round(self.reference_accel_mm_s2, 0),
            "pa_extruder_velocity_spike_mm_s": round(self.pa_extruder_velocity_spike_mm_s, 2),
            "column_buckling_load_n": round(self.column_buckling_load_n, 2),
            "straight_column_valid": self.straight_column_valid,
            "advance_curve": self.advance_curve,
            "warnings": self.warnings,
        }


def simulate_pressure_advance(
    spec: PrinterSpec,
    filament_id: str = "pla",
    target_temp_c: float | None = None,
    nozzle_type: str | None = None,
    reference_flow_mm3_s: float | None = None,
    thermal: "HotendThermalReport | None" = None,
    accel_mm_s2: float | None = None,
    bowden_bend_angle_deg: float = 180.0,
) -> PressureAdvanceReport:
    """Predict the pressure-advance constant, advance curve, ooze volume, and retraction floor.

    Pass the already-computed ``thermal`` report from ``evaluate_printer`` to reuse its flow
    limit, nozzle type, and cold-side temperature instead of re-solving the hotend.
    """
    fil = get_filament(filament_id)
    if thermal is None:
        from printsim.thermal.hotend import simulate_hotend_thermal

        thermal = simulate_hotend_thermal(
            spec, target_temp_c=target_temp_c, filament_id=filament_id, nozzle_type=nozzle_type
        )
    t_target = target_temp_c if target_temp_c is not None else thermal.target_temp_c
    resolved_nozzle_type = (nozzle_type or thermal.nozzle_type or "standard").lower()
    if resolved_nozzle_type not in ("standard", "cht"):
        resolved_nozzle_type = "standard"

    he = spec.toolhead.hotend
    ext = spec.toolhead.extruder
    a_fil_m2 = math.pi * (0.5 * ext.filament_diameter_mm * 1e-3) ** 2
    q_max_m3_s = max(thermal.max_volumetric_flow_mm3_s, 0.5) * 1e-9
    v_tab, f_tab = build_push_force_table(spec, fil, t_target, resolved_nozzle_type, q_max_m3_s)

    q_ref_mm3 = (
        reference_flow_mm3_s
        if reference_flow_mm3_s is not None
        else min(DEFAULT_REFERENCE_FLOW_MM3_S, 0.9 * q_max_m3_s * 1e9)
    )
    q_ref_m3_s = min(max(q_ref_mm3, 1e-3) * 1e-9, q_max_m3_s)
    v_ref = q_ref_m3_s / a_fil_m2
    f_ref = float(np.interp(v_ref, v_tab, f_tab))

    comp = compute_filament_path_compliance(
        spec,
        fil,
        target_temp_c=t_target,
        cold_side_temp_c=thermal.cold_side_temp_c,
        mean_pressure_pa=0.5 * f_ref / a_fil_m2,
        bowden_bend_angle_deg=bowden_bend_angle_deg,
    )
    c_total = comp.total_m_per_n

    x_ref = c_total * f_ref
    k_secant = x_ref / v_ref
    dv = 0.01 * v_ref
    k_tangent = c_total * float(
        (np.interp(v_ref + dv, v_tab, f_tab) - np.interp(v_ref - dv, v_tab, f_tab)) / (2.0 * dv)
    )

    curve: list[dict[str, float]] = []
    for frac in _ADVANCE_CURVE_FRACTIONS:
        v_pt = frac * q_max_m3_s / a_fil_m2
        adv_m = c_total * float(np.interp(v_pt, v_tab, f_tab))
        curve.append(
            {
                "flow_mm3_s": round(frac * q_max_m3_s * 1e9, 2),
                "filament_speed_mm_s": round(v_pt * 1e3, 3),
                "advance_mm": round(adv_m * 1e3, 4),
                "pressure_advance_s": round(adv_m / v_pt, 4),
            }
        )

    # Peak extra filament speed the advance term demands while the toolhead accelerates
    accel_m_s2 = (accel_mm_s2 if accel_mm_s2 is not None else spec.kinematics.max_accel_mm_s2) * 1e-3
    nozzle_d_m = he.nozzle_diameter_mm * 1e-3
    bead_area_m2 = (_BEAD_WIDTH_RATIO * nozzle_d_m) * (_BEAD_HEIGHT_RATIO * nozzle_d_m)
    spike_m_s = k_secant * accel_m_s2 * bead_area_m2 / a_fil_m2

    warnings: list[str] = []
    straight_ok = f_ref <= comp.column_buckling_load_n
    if not straight_ok:
        warnings.append(
            f"Push force at the reference flow ({f_ref:.1f} N) exceeds the Euler buckling load of the "
            f"{ext.filament_path_length_mm:.0f} mm filament run ({comp.column_buckling_load_n:.1f} N): "
            "the filament bows against its guide, so the straight-column compliance and the predicted "
            "pressure advance are not reliable for this filament and path length."
        )

    return PressureAdvanceReport(
        filament_id=fil.id,
        extruder_type=ext.type,
        reference_flow_mm3_s=q_ref_m3_s * 1e9,
        reference_filament_speed_mm_s=v_ref * 1e3,
        push_force_at_reference_n=f_ref,
        pressure_advance_s=k_secant,
        pressure_advance_tangent_s=k_tangent,
        system_compliance_um_per_n=c_total * 1e6,
        filament_column_compliance_um_per_n=comp.filament_column_m_per_n * 1e6,
        bowden_tube_compliance_um_per_n=comp.bowden_tube_m_per_n * 1e6,
        melt_compliance_um_per_n=comp.melt_m_per_n * 1e6,
        system_stiffness_n_per_mm=1e-3 / c_total,
        melt_bulk_modulus_mpa=comp.melt_bulk_modulus_pa * 1e-6,
        melt_bulk_modulus_source=comp.melt_bulk_modulus_source,
        stored_filament_compression_mm=x_ref * 1e3,
        ooze_volume_mm3=x_ref * a_fil_m2 * 1e9,
        depressurization_time_s=depressurization_time_s(c_total, v_tab, f_tab, v_ref),
        bowden_slack_mm=comp.bowden_slack_m * 1e3,
        min_retraction_mm=(x_ref + comp.bowden_slack_m) * 1e3,
        reference_accel_mm_s2=accel_m_s2 * 1e3,
        pa_extruder_velocity_spike_mm_s=spike_m_s * 1e3,
        column_buckling_load_n=comp.column_buckling_load_n,
        straight_column_valid=straight_ok,
        advance_curve=curve,
        warnings=warnings,
    )
