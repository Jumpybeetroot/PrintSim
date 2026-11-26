"""Coupled electro-mechanical LPV state-space & workspace FRF maps (strict SI internally).

References
----------
- Acarnley, P. P. (2002), *Stepping Motors: A Guide to Theory and Practice* (4th ed., IET):
  electromagnetic stiffness `K_em = p * T_hold * eta_V` and mid-band negative electrical damping.
- Leenhouts, A. C. (1991), *The Art and Practice of Step Motor Control*: L/R current-lag damping
  roll-off at mid-band stepping frequencies.
- Gerbert, B. G. (1972) & Ernst, J. (1965) via `printsim.fea.components`: belt cord crimp/sag
  `belt_effective_ea_n`, tooth-mesh `gerbert_tooth_mesh_stiffness_n_m`, idler/bearing flex, and
  Gantry bracket compliance `compute_bracket_stiffness_n_per_m`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from printsim.fea.components import (
    belt_damping_ratio,
    belt_effective_ea_n,
    gerbert_tooth_mesh_stiffness_n_m,
    idler_shaft_and_bearing_stiffness,
)
from printsim.fea.klipper_shaper import biquad_notch_params_from_db_q
from printsim.kinematics.solvers import (
    compute_bracket_stiffness_n_per_m,
    compute_drive_stiffness_details_n_per_m,
)
from printsim.schema import get_material

if TYPE_CHECKING:
    from printsim.fea.modal import ModalResult
    from printsim.geometry.builder import AssemblyModel


_UNSET_CFG = object()


@dataclass(init=False)
class DriveControlConfig:
    """Motor drive topology & control-loop bandwidth configuration."""

    drive_type: Literal["open_loop_stepper", "closed_loop_stepper", "foc_servo"] = "open_loop_stepper"
    servo_vel_bw_hz: float = 120.0
    servo_pos_bw_hz: float = 25.0
    notch_filter_hz: float | None = None
    notch_zeta_zero: float = 0.03
    notch_zeta_pole: float = 0.35
    notch_depth_db: float = 12.0
    notch_q: float = 1.75
    awd_sync_mode: str = "none"
    awd_equalizer_bw_hz: float = 2.0
    pin_rotor_ff: bool = False
    toolhead_imu_active_damping: bool = False
    mode_space_torque_ff: bool = False

    def __init__(
        self,
        drive_type: Literal["open_loop_stepper", "closed_loop_stepper", "foc_servo"] = "open_loop_stepper",
        servo_vel_bw_hz: float = 120.0,
        servo_pos_bw_hz: float = 25.0,
        notch_filter_hz: float | None = None,
        notch_zeta_zero: float | object = _UNSET_CFG,
        notch_zeta_pole: float | object = _UNSET_CFG,
        notch_depth_db: float | object = _UNSET_CFG,
        notch_q: float | object = _UNSET_CFG,
        awd_sync_mode: str = "none",
        awd_equalizer_bw_hz: float = 2.0,
        pin_rotor_ff: bool = False,
        toolhead_imu_active_damping: bool = False,
        mode_space_torque_ff: bool = False,
    ) -> None:
        self.drive_type = drive_type
        self.servo_vel_bw_hz = float(servo_vel_bw_hz)
        self.servo_pos_bw_hz = float(servo_pos_bw_hz)
        self.notch_filter_hz = notch_filter_hz
        has_db_q = (notch_depth_db is not _UNSET_CFG) or (notch_q is not _UNSET_CFG)
        has_zeta = (notch_zeta_zero is not _UNSET_CFG) or (notch_zeta_pole is not _UNSET_CFG)
        self.notch_zeta_zero = 0.03 if notch_zeta_zero is _UNSET_CFG else float(notch_zeta_zero)  # type: ignore[arg-type]
        self.notch_zeta_pole = 0.35 if notch_zeta_pole is _UNSET_CFG else float(notch_zeta_pole)  # type: ignore[arg-type]
        self.notch_depth_db = 12.0 if notch_depth_db is _UNSET_CFG else float(notch_depth_db)  # type: ignore[arg-type]
        self.notch_q = 1.75 if notch_q is _UNSET_CFG else float(notch_q)  # type: ignore[arg-type]
        self._use_db_q_notch = bool(has_db_q and not has_zeta)
        self.awd_sync_mode = str(awd_sync_mode)
        self.awd_equalizer_bw_hz = float(awd_equalizer_bw_hz)
        self.pin_rotor_ff = bool(pin_rotor_ff)
        self.toolhead_imu_active_damping = bool(toolhead_imu_active_damping)
        self.mode_space_torque_ff = bool(mode_space_torque_ff)


@dataclass
class CoupledMechatronicsModel:
    """Coupled continuous-time LPV state-space `(A, B, C_noz, D)` and modal/FRF metrics."""

    A: np.ndarray
    B: np.ndarray
    C_noz: np.ndarray
    D: np.ndarray
    n_mot: int
    m_struct: int
    u_rel: float
    v_rel: float
    drive_cfg: DriveControlConfig
    k_belt_cart: np.ndarray
    k_A_n_m: float
    k_B_n_m: float
    k_em_per_motor_n_m: float
    b_em_per_motor_ns_m: float
    reflected_rotor_mass_per_motor_g: float
    reflected_rotor_mass_g: float
    coupled_freq_x_hz: float
    coupled_freq_y_hz: float
    coupled_damping_x: float
    coupled_damping_y: float
    mode_freqs_hz: list[float]
    mode_dampings: list[float]
    freqs_hz: np.ndarray
    H_frf: np.ndarray
    xy_cross_coupling_pct: float
    midband_instability_risk: bool
    T_cart: np.ndarray | None = None
    k_em_per_motor_arr: np.ndarray | None = None
    b_em_per_motor_arr: np.ndarray | None = None

    @property
    def state_dim(self) -> int:
        return int(self.A.shape[0])

    def evaluate_frf(self, freqs_hz: np.ndarray) -> np.ndarray:
        """Compute exact MIMO transfer matrix H(jw) = C (jwI - A)^(-1) B + D."""
        return _eval_mimo_frf(self.A, self.B, self.C_noz, self.D, freqs_hz)

    def to_dict(self) -> dict:
        return {
            "drive_type": self.drive_cfg.drive_type,
            "state_dim": self.state_dim,
            "n_mot": self.n_mot,
            "m_struct": self.m_struct,
            "u_rel": round(float(self.u_rel), 3),
            "v_rel": round(float(self.v_rel), 3),
            "k_A_n_per_mm": round(float(self.k_A_n_m) * 1e-3, 2),
            "k_B_n_per_mm": round(float(self.k_B_n_m) * 1e-3, 2),
            "k_em_per_motor_n_per_mm": round(float(self.k_em_per_motor_n_m) * 1e-3, 2),
            "b_em_per_motor_ns_m": round(float(self.b_em_per_motor_ns_m), 3),
            "reflected_rotor_mass_per_motor_g": round(float(self.reflected_rotor_mass_per_motor_g), 2),
            "reflected_rotor_mass_g": round(float(self.reflected_rotor_mass_g), 2),
            "coupled_freq_x_hz": round(float(self.coupled_freq_x_hz), 2),
            "coupled_freq_y_hz": round(float(self.coupled_freq_y_hz), 2),
            "coupled_damping_x": round(float(self.coupled_damping_x), 4),
            "coupled_damping_y": round(float(self.coupled_damping_y), 4),
            "mode_freqs_hz": [round(float(f), 2) for f in self.mode_freqs_hz],
            "xy_cross_coupling_pct": round(float(self.xy_cross_coupling_pct), 2),
            "midband_instability_risk": bool(self.midband_instability_risk),
        }


@dataclass
class WorkspaceFRFMap:
    """Position-dependent 2D workspace natural-frequency & cross-coupling map."""

    grid_n: int
    u_coords: list[float]
    v_coords: list[float]
    freq_x_grid_hz: list[list[float]]
    freq_y_grid_hz: list[list[float]]
    min_axis_freq_grid_hz: list[list[float]]
    xy_cross_coupling_grid_pct: list[list[float]]
    center_freq_hz: float
    min_freq_hz: float
    max_freq_hz: float
    worst_corner_drop_pct: float
    max_cross_coupling_pct: float
    reflected_rotor_mass_g: float
    drive_type: str
    midband_instability_risk: bool
    is_spatial_spread_pct: float = 0.0

    def to_dict(self) -> dict:
        return {
            "grid_n": int(self.grid_n),
            "u_coords": [round(float(u), 3) for u in self.u_coords],
            "v_coords": [round(float(v), 3) for v in self.v_coords],
            "freq_x_grid_hz": [[round(float(val), 2) for val in row] for row in self.freq_x_grid_hz],
            "freq_y_grid_hz": [[round(float(val), 2) for val in row] for row in self.freq_y_grid_hz],
            "min_axis_freq_grid_hz": [
                [round(float(val), 2) for val in row] for row in self.min_axis_freq_grid_hz
            ],
            "xy_cross_coupling_grid_pct": [
                [round(float(val), 2) for val in row] for row in self.xy_cross_coupling_grid_pct
            ],
            "center_freq_hz": round(float(self.center_freq_hz), 2),
            "min_freq_hz": round(float(self.min_freq_hz), 2),
            "max_freq_hz": round(float(self.max_freq_hz), 2),
            "worst_corner_drop_pct": round(float(self.worst_corner_drop_pct), 2),
            "is_spatial_spread_pct": round(float(self.is_spatial_spread_pct), 2),
            "max_cross_coupling_pct": round(float(self.max_cross_coupling_pct), 2),
            "reflected_rotor_mass_g": round(float(self.reflected_rotor_mass_g), 2),
            "drive_type": self.drive_type,
            "midband_instability_risk": bool(self.midband_instability_risk),
        }


def _xy_coordinates_m(assembly: AssemblyModel, u_rel: float, v_rel: float) -> tuple[float, float, float, float]:
    """Map normalized workspace `(u_rel, v_rel) in [0, 1]^2` to gantry frame `(x, y, w_g, d_g)` in meters."""
    spec = assembly.spec
    w_g = max(spec.frame.width_mm * 1e-3, 0.12)
    d_g = max(spec.frame.depth_mm * 1e-3, 0.12)
    bx = max(spec.kinematics.build_volume_mm[0] * 1e-3, 0.05)
    by = max(spec.kinematics.build_volume_mm[1] * 1e-3, 0.05)
    orig_x = max(0.5 * (w_g - bx), 0.03)
    orig_y = max(0.5 * (d_g - by), 0.03)
    u = float(np.clip(u_rel, 0.0, 1.0))
    v = float(np.clip(v_rel, 0.0, 1.0))
    x = min(max(orig_x + u * bx, 0.03), max(w_g - 0.03, 0.04))
    y = min(max(orig_y + v * by, 0.03), max(d_g - 0.03, 0.04))
    return x, y, w_g, d_g


def _corexy_branch_details(
    assembly: AssemblyModel,
    u_rel: float,
    v_rel: float,
) -> dict[str, object]:
    """Compute exact position-dependent belt branch stiffnesses unified with canonical solvers.py."""
    spec = assembly.spec
    det_x = compute_drive_stiffness_details_n_per_m(assembly, axis="X", u_rel=u_rel, v_rel=v_rel)
    k_bkt_total = float(det_x["k_bracket_n_m"])

    if spec.architecture == "corexy":
        is_awd = spec.gantry.drive_mode == "awd"
        k_bkt_loop = 0.5 * k_bkt_total
        if is_awd:
            kb_a1, kb_a2, kb_b1, kb_b2 = det_x["branch_stiffnesses_n_m"]  # type: ignore[misc]

            def _apply_bkt_awd(kb1: float, kb2: float) -> tuple[float, float, float]:
                k_sum = float(kb1) + float(kb2)
                k_loop = 1.0 / (1.0 / max(k_sum, 1e-9) + 1.0 / max(k_bkt_loop, 1e-9))
                frac1 = float(kb1) / max(k_sum, 1e-12)
                return k_loop * frac1, k_loop * (1.0 - frac1), k_loop

            k_a1, k_a2, k_a = _apply_bkt_awd(kb_a1, kb_a2)
            k_b1, k_b2, k_b = _apply_bkt_awd(kb_b1, kb_b2)
            k_cart = np.array([[k_a + k_b, k_a - k_b], [k_a - k_b, k_a + k_b]], dtype=float)
            return {
                "arch": "corexy_awd",
                "k_cart": k_cart,
                "k_A": k_a,
                "k_B": k_b,
                "branches": (k_a1, k_a2, k_b1, k_b2),
            }

        k_a_belt = float(det_x["k_A_n_m"])
        k_b_belt = float(det_x["k_B_n_m"])
        k_a = 1.0 / (1.0 / max(k_a_belt, 1e-9) + 1.0 / max(k_bkt_loop, 1e-9))
        k_b = 1.0 / (1.0 / max(k_b_belt, 1e-9) + 1.0 / max(k_bkt_loop, 1e-9))
        k_cart = np.array([[k_a + k_b, k_a - k_b], [k_a - k_b, k_a + k_b]], dtype=float)
        return {
            "arch": "corexy_2wd",
            "k_cart": k_cart,
            "k_A": k_a,
            "k_B": k_b,
            "branches": (k_a, k_b),
        }

    if spec.architecture == "cartesian_bedslinger":
        det_y = compute_drive_stiffness_details_n_per_m(assembly, axis="Y", u_rel=u_rel, v_rel=v_rel)
        k_x = 1.0 / (1.0 / max(float(det_x["k_belt_subsystem_n_m"]), 1e-9) + 1.0 / max(k_bkt_total, 1e-9))
        k_y = 1.0 / (1.0 / max(float(det_y["k_belt_subsystem_n_m"]), 1e-9) + 1.0 / max(k_bkt_total, 1e-9))
        return {
            "arch": "cartesian",
            "k_cart": np.diag([k_x, k_y]),
            "k_A": k_x,
            "k_B": k_y,
            "branches": (k_x, k_y),
        }

    # Delta: 3 vertical tower belts acting through effector Jacobian
    x, y, w_g, d_g = _xy_coordinates_m(assembly, u_rel, v_rel)
    k_belt_base = 1.0 / (
        1.0 / max(float(det_x["k_belt_subsystem_n_m"]), 1e-9)
        + 1.0 / max(k_bkt_total, 1e-9)
    )
    # Mild geometric modulation from effector radial offset from center
    r_eff = math.hypot(x - 0.5 * w_g, y - 0.5 * d_g)
    r_max = max(spec.kinematics.delta_radius_mm * 1e-3, 0.10)
    mod = 1.0 - 0.12 * min(1.0, (r_eff / r_max) ** 2)
    k_diag = k_belt_base * mod
    return {
        "arch": "delta",
        "k_cart": np.diag([k_diag, k_diag]),
        "k_A": k_diag,
        "k_B": k_diag,
        "branches": (k_diag, k_diag, k_diag),
    }


def corexy_belt_span_stiffness_matrix(
    assembly: AssemblyModel,
    u_rel: float = 0.5,
    v_rel: float = 0.5,
) -> tuple[np.ndarray, float, float]:
    """Return position-dependent Cartesian belt stiffness `K_cart(u, v)` and `(k_A, k_B)` (N/m)."""
    info = _corexy_branch_details(assembly, u_rel, v_rel)
    k_cart = info["k_cart"]
    return k_cart, float(info["k_A"]), float(info["k_B"])  # type: ignore[return-value]


def _estimate_structural_modal_params(
    assembly: AssemblyModel,
    modal_result: ModalResult,
    omega_x_override_hz: float | None = None,
    omega_y_override_hz: float | None = None,
) -> tuple[float, float, float, float, float, float, float]:
    """Return `(f_sx_hz, f_sy_hz, zeta_frame, m_fx_kg, m_fy_kg, m_cx_kg, m_cy_kg)`."""
    m_cx = max(float(assembly.x_moving_mass_kg), 0.05)
    m_cy = max(float(assembly.y_moving_mass_kg), 0.05)

    f_sx = float(omega_x_override_hz) if omega_x_override_hz is not None else (
        float(modal_result.shaper_x.structural_freq_hz)
        if modal_result.shaper_x and modal_result.shaper_x.structural_freq_hz > 0.0
        else max(float(modal_result.fundamental_freq_hz), 15.0)
    )
    f_sy = float(omega_y_override_hz) if omega_y_override_hz is not None else (
        float(modal_result.shaper_y.structural_freq_hz)
        if modal_result.shaper_y and modal_result.shaper_y.structural_freq_hz > 0.0
        else max(float(modal_result.fundamental_freq_hz), 15.0)
    )
    zeta_frame = float(
        getattr(modal_result, "frame_damping_ratio", assembly.frame_damping_ratio)
        or assembly.frame_damping_ratio
    )
    zeta_frame = float(np.clip(zeta_frame, 0.005, 0.25))

    # Effective frame-only participating mass m_f (excluding moving carriage mass m_c)
    frame_mass = max(float(assembly.frame_mass_kg), 0.5)
    m_fx = 0.22 * frame_mass + 0.15
    m_fy = 0.24 * frame_mass + 0.15

    if modal_result.modes:
        nmap = assembly.node_map
        moving_ids = {
            assembly.toolhead_node_id,
            assembly.toolhead_cog_node_id,
            assembly.nozzle_node_id,
        }
        noz_id = assembly.nozzle_node_id if assembly.nozzle_node_id in nmap else assembly.toolhead_node_id
        for axis_tag, target_f, set_x in (("X", f_sx, True), ("Y", f_sy, False)):
            best_mode = max(
                modal_result.modes,
                key=lambda md: (
                    md.x_axis_share_pct if axis_tag == "X" else md.y_axis_share_pct,
                    1.0 if md.excited_axis in (axis_tag, "XY") else 0.0,
                    -abs(md.frequency_hz - target_f),
                ),
            )
            if best_mode.nodal_vectors and noz_id in best_mode.nodal_vectors:
                v_noz = np.asarray(best_mode.nodal_vectors[noz_id], dtype=float)
                amp_noz_sq = max(float(np.dot(v_noz, v_noz)), 1e-6)
                m_part = 0.0
                for nid, vec in best_mode.nodal_vectors.items():
                    if nid in moving_ids or nid not in nmap:
                        continue
                    v_arr = np.asarray(vec, dtype=float)
                    m_part += nmap[nid].lumped_mass_kg * float(np.dot(v_arr, v_arr)) / amp_noz_sq
                # Add distributed frame beam inertia participation (~20% of frame_mass)
                m_est = max(0.20 * m_cx, min(2.0 * frame_mass, m_part + 0.18 * frame_mass))
                if set_x:
                    m_fx = m_est
                else:
                    m_fy = m_est

    return f_sx, f_sy, zeta_frame, m_fx, m_fy, m_cx, m_cy


def _stepper_em_params(
    assembly: AssemblyModel,
    m_ref_1: float,
    m_carriage_per_motor: float,
    drive_cfg: DriveControlConfig,
) -> tuple[float, float, bool]:
    """Compute `(k_em_per_motor_n_m, b_em_per_motor_ns_m, midband_instability_risk)`."""
    spec = assembly.spec
    stepper = spec.kinematics.xy_stepper
    belt = spec.gantry.belt
    r_p = belt.pitch_radius_m

    # Nominal mid-band operating speed (~35% peak speed, matches kinematics/solvers.py)
    motor_lin_v = (spec.kinematics.max_velocity_mm_s * 1e-3) * (
        1.4142 if spec.architecture == "corexy" else 1.0
    )
    max_rpm = (motor_lin_v / (2.0 * math.pi * r_p)) * 60.0
    rpm_op = max(120.0, 0.35 * max_rpm)
    omega_m0 = rpm_op * (2.0 * math.pi / 60.0)

    k_theta = stepper.torsional_stiffness_nm_per_rad(rpm=rpm_op)
    k_em_stepper = k_theta / (r_p**2)

    if drive_cfg.drive_type == "foc_servo":
        omega_v = 2.0 * math.pi * max(float(drive_cfg.servo_vel_bw_hz), 10.0)
        omega_p = 2.0 * math.pi * max(float(drive_cfg.servo_pos_bw_hz), 2.0)
        m_eff = m_ref_1 + m_carriage_per_motor
        b_em = m_eff * omega_v
        k_em = m_eff * omega_v * omega_p
        return k_em, b_em, False

    if drive_cfg.drive_type == "closed_loop_stepper":
        # Encoder-locked FOC microstepping suppresses mid-band negative damping & adds velocity damping
        omega_v = 2.0 * math.pi * max(float(drive_cfg.servo_vel_bw_hz), 10.0)
        k_em = k_em_stepper
        b_em = max(0.32 * 2.0 * math.sqrt(k_em * (m_ref_1 + m_carriage_per_motor)), 0.15 * (m_ref_1 + m_carriage_per_motor) * omega_v)
        return k_em, b_em, False

    # Open-loop stepper: Acarnley (2002) & Leenhouts (1991) speed-dependent electrical damping
    # tau_e = L / R, electrical commutation frequency omega_e = pole_pairs * omega_m0
    j_m = stepper.rotor_inertia_kg_m2
    r_ohm = max(float(stepper.phase_resistance_ohm), 0.05)
    l_h = max(float(stepper.inductance_mh) * 1e-3, 1e-5)
    tau_e = l_h / r_ohm
    omega_e = stepper.pole_pairs * omega_m0
    wt = omega_e * tau_e
    eta_v = stepper.dynamic_voltage_efficiency(rpm=rpm_op)
    k_e = stepper.holding_torque_nm / (math.sqrt(2.0) * max(stepper.rated_current_a, 0.1))

    # Mechanical + microstepping hysteresis baseline damping (~5.5% critical)
    b_mech_rot = 2.0 * 0.055 * math.sqrt(max(k_theta * j_m, 1e-12))

    # Chopper current-loop saturation (1 - eta_v^2) transitions driver from current-source
    # toward voltage-fed mode where Acarnley negative electrical damping emerges for wt > 1.
    chop_sat = max(0.0, 1.0 - eta_v * eta_v)
    b_bemf_rot = ((k_e**2) / r_ohm) * ((1.0 - wt * wt) / ((1.0 + wt * wt) ** 2)) * (0.40 * chop_sat)
    b_ind_lag_rot = (
        0.14
        * stepper.pole_pairs
        * stepper.effective_holding_torque_nm
        * tau_e
        * (wt / (1.0 + wt * wt))
        * chop_sat
    )
    b_rot_raw = b_mech_rot + b_bemf_rot - b_ind_lag_rot

    zeta_raw = b_rot_raw / max(2.0 * math.sqrt(k_theta * j_m), 1e-12)
    midband_risk = bool(zeta_raw < 0.015)

    # Enforce tiny positive LTI floor so linearized state-space stays numerically bounded in time domain
    b_rot_floor = 2.0 * 0.005 * math.sqrt(k_theta * j_m)
    b_rot_eff = max(b_rot_raw, b_rot_floor)
    b_em = b_rot_eff / (r_p**2)
    return k_em_stepper, b_em, midband_risk


def _eval_mimo_frf(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    freqs_hz: np.ndarray,
) -> np.ndarray:
    """Fast vectorized evaluation of H(jw) = C (jwI - A)^(-1) B + D across `freqs_hz`."""
    n = A.shape[0]
    n_f = len(freqs_hz)
    omega = 2.0 * math.pi * np.asarray(freqs_hz, dtype=float)
    eye_n = np.eye(n, dtype=complex)
    # Batch solve (jwI - A) X = B for all frequencies simultaneously
    mats = (1j * omega[:, None, None]) * eye_n[None, :, :] - A[None, :, :]
    rhs = np.broadcast_to(B[None, :, :].astype(complex), (n_f, n, B.shape[1]))
    sol = np.linalg.solve(mats, rhs)
    return np.einsum("ij,fjk->fik", C.astype(complex), sol) + D[None, :, :]


def _cascade_biquad_notch(
    A_p: np.ndarray,
    B_p: np.ndarray,
    C_p: np.ndarray,
    D_p: np.ndarray,
    f_notch_hz: float,
    zeta_zero: float,
    zeta_pole: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cascade 2-channel 2nd-order biquad notch filter onto inputs of plant `(A_p, B_p, C_p, D_p)`."""
    wn = 2.0 * math.pi * max(float(f_notch_hz), 1.0)
    # Per-channel realization: dz1=z2, dz2=-wn^2*z1 - 2*zp*wn*z2 + u, y = u + 2*(zz-zp)*wn*z2
    a_1d = np.array([[0.0, 1.0], [-(wn * wn), -2.0 * zeta_pole * wn]], dtype=float)
    b_1d = np.array([[0.0], [1.0]], dtype=float)
    c_1d = np.array([[0.0, 2.0 * (zeta_zero - zeta_pole) * wn]], dtype=float)
    d_1d = np.array([[1.0]], dtype=float)

    A_f = np.block([[a_1d, np.zeros((2, 2))], [np.zeros((2, 2)), a_1d]])
    B_f = np.block([[b_1d, np.zeros((2, 1))], [np.zeros((2, 1)), b_1d]])
    C_f = np.block([[c_1d, np.zeros((1, 2))], [np.zeros((1, 2)), c_1d]])
    D_f = np.block([[d_1d, np.zeros((1, 1))], [np.zeros((1, 1)), d_1d]])

    n_p = A_p.shape[0]
    A_aug = np.block([[A_p, B_p @ C_f], [np.zeros((4, n_p)), A_f]])
    B_aug = np.vstack([B_p @ D_f, B_f])
    C_aug = np.hstack([C_p, D_p @ C_f])
    D_aug = D_p @ D_f
    return A_aug, B_aug, C_aug, D_aug


def build_coupled_state_space(
    assembly: AssemblyModel,
    modal_result: ModalResult,
    u_rel: float = 0.5,
    v_rel: float = 0.5,
    drive_cfg: DriveControlConfig | None = None,
    omega_x_override_hz: float | None = None,
    omega_y_override_hz: float | None = None,
) -> CoupledMechatronicsModel:
    """Build coupled `(2(N_mot + m_struct))`-state electro-mechanical LPV state-space model."""
    cfg = drive_cfg or DriveControlConfig()
    spec = assembly.spec
    stepper = spec.kinematics.xy_stepper
    belt = spec.gantry.belt
    r_p = belt.pitch_radius_m

    m_ref_1 = stepper.rotor_inertia_kg_m2 / (r_p**2)
    f_sx, f_sy, zeta_frame, m_fx, m_fy, m_cx, m_cy = _estimate_structural_modal_params(
        assembly,
        modal_result,
        omega_x_override_hz=omega_x_override_hz,
        omega_y_override_hz=omega_y_override_hz,
    )

    branch_info = _corexy_branch_details(assembly, u_rel, v_rel)
    arch_kind = str(branch_info["arch"])
    k_cart = np.asarray(branch_info["k_cart"], dtype=float)
    k_a = float(branch_info["k_A"])
    k_b = float(branch_info["k_B"])
    branches = branch_info["branches"]

    # Belt coupling matrix T_cart (n_mot x 2) and branch stiffnesses first so Delta Jacobian
    # j_norm is available for motor load sizing, reflected rotor mass, and active damping.
    j_norm = 1.0
    if arch_kind == "corexy_awd":
        n_mot = 4
        T_cart = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, -1.0], [1.0, -1.0]], dtype=float)
        k_branches = np.array(branches, dtype=float)
        m_car_per_mot = (m_cx + m_cy) / (2.0 * n_mot)
        reflected_per_axis = 4.0 * m_ref_1
        k_em_1, b_em_1, midband_risk = _stepper_em_params(assembly, m_ref_1, m_car_per_mot, cfg)
        k_em_arr = np.full(n_mot, k_em_1, dtype=float)
        b_em_arr = np.full(n_mot, b_em_1, dtype=float)
    elif arch_kind == "corexy_2wd":
        n_mot = 2
        T_cart = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=float)
        k_branches = np.array(branches, dtype=float)
        m_car_per_mot = (m_cx + m_cy) / (2.0 * n_mot)
        reflected_per_axis = 2.0 * m_ref_1
        k_em_1, b_em_1, midband_risk = _stepper_em_params(assembly, m_ref_1, m_car_per_mot, cfg)
        k_em_arr = np.full(n_mot, k_em_1, dtype=float)
        b_em_arr = np.full(n_mot, b_em_1, dtype=float)
    elif arch_kind == "cartesian":
        n_mot = 2
        T_cart = np.eye(2, dtype=float)
        k_branches = np.array(branches, dtype=float)
        reflected_per_axis = 1.0 * m_ref_1
        k_em_x, b_em_x, risk_x = _stepper_em_params(assembly, m_ref_1, m_cx, cfg)
        k_em_y, b_em_y, risk_y = _stepper_em_params(assembly, m_ref_1, m_cy, cfg)
        midband_risk = risk_x or risk_y
        k_em_arr = np.array([k_em_x, k_em_y], dtype=float)
        b_em_arr = np.array([b_em_x, b_em_y], dtype=float)
    else:
        n_mot = 3
        # Delta tower Jacobian J_delta (3x2) evaluated within circular build disk (r <= 0.45 * min(bx, by))
        x_m, y_m, w_g, d_g = _xy_coordinates_m(assembly, u_rel, v_rel)
        cx, cy = x_m - 0.5 * w_g, y_m - 0.5 * d_g
        r_bed_max = 0.45 * min(
            max(spec.kinematics.build_volume_mm[0] * 1e-3, 0.05),
            max(spec.kinematics.build_volume_mm[1] * 1e-3, 0.05),
        )
        r_cur = math.hypot(cx, cy)
        if r_cur > r_bed_max and r_cur > 1e-9:
            cx *= r_bed_max / r_cur
            cy *= r_bed_max / r_cur
        r_del = max(spec.kinematics.delta_radius_mm * 1e-3, 0.10)
        l_arm = max(spec.kinematics.delta_arm_length_mm * 1e-3, r_del * 1.6)
        j_rows = []
        for ang_deg in (90.0, 210.0, 330.0):
            ang = math.radians(ang_deg)
            dx = cx - r_del * math.cos(ang)
            dy = cy - r_del * math.sin(ang)
            z_arm = math.sqrt(max(l_arm**2 - dx * dx - dy * dy, 0.02))
            j_rows.append([-dx / z_arm, -dy / z_arm])
        T_cart = np.asarray(j_rows, dtype=float)
        # Scale diagonal stiffness so T_cart^T diag(k_b) T_cart matches k_cart at center
        cx0, cy0 = 0.0, 0.0
        j0_rows = []
        for ang_deg in (90.0, 210.0, 330.0):
            ang = math.radians(ang_deg)
            dx0 = cx0 - r_del * math.cos(ang)
            dy0 = cy0 - r_del * math.sin(ang)
            z_arm0 = math.sqrt(max(l_arm**2 - dx0 * dx0 - dy0 * dy0, 0.02))
            j0_rows.append([-dx0 / z_arm0, -dy0 / z_arm0])
        j0 = np.asarray(j0_rows, dtype=float)
        j0_norm = float(np.mean(np.diag(j0.T @ j0)))
        k_diag_b = float(k_cart[0, 0]) / max(j0_norm, 1e-6)
        k_branches = np.full(3, k_diag_b, dtype=float)

        j_norm = float(np.mean(np.diag(T_cart.T @ T_cart)))
        m_car_per_mot = 0.5 * (m_cx + m_cy) / max(j_norm, 1e-6)
        reflected_per_axis = j_norm * m_ref_1
        k_em_1, b_em_1, midband_risk = _stepper_em_params(assembly, m_ref_1, m_car_per_mot, cfg)
        k_em_arr = np.full(n_mot, k_em_1, dtype=float)
        b_em_arr = np.full(n_mot, b_em_1, dtype=float)

    # Closed-loop / FOC beta control laws (gracefully no-op on open_loop_stepper)
    is_foc = cfg.drive_type in ("foc_servo", "closed_loop_stepper")
    delta_zeta_belt = 0.0
    if is_foc:
        k_em_mult = 1.0
        if cfg.pin_rotor_ff:
            k_em_mult *= 3.5
        if arch_kind == "corexy_awd":
            sync_mode = str(cfg.awd_sync_mode).strip().lower()
            if sync_mode == "position_equalizer":
                k_em_mult *= 1.12
                excess_bw = max(0.0, float(cfg.awd_equalizer_bw_hz) - 5.0)
                delta_zeta_belt = max(0.004, 0.015 - 0.0012 * excess_bw)
            elif sync_mode == "torque_mirror":
                k_em_mult *= 1.06
                delta_zeta_belt = 0.008
        k_em_arr *= k_em_mult
        b_em_arr *= k_em_mult

    # Mechanical coordinates q = [q_fx, q_fy, q_m(0..N_mot-1), q_cx, q_cy]
    # Total mechanical DOFs = n_mot + m_struct (m_struct = 4: 2 frame modal + 2 carriage XY)
    m_struct = 4
    n_dof = n_mot + m_struct
    idx_f = slice(0, 2)
    idx_m = slice(2, 2 + n_mot)
    idx_c = slice(2 + n_mot, n_dof)

    M = np.zeros((n_dof, n_dof), dtype=float)
    K = np.zeros((n_dof, n_dof), dtype=float)
    C = np.zeros((n_dof, n_dof), dtype=float)

    # Include Rayleigh moving-belt inertia on carriage DOFs (consistent with modal.py)
    det_c = compute_drive_stiffness_details_n_per_m(assembly, axis="X", u_rel=u_rel, v_rel=v_rel)
    m_belt_eff = float(det_c.get("belt_moving_mass_kg", 0.0))
    M[0, 0] = m_fx
    M[1, 1] = m_fy
    for i in range(n_mot):
        M[2 + i, 2 + i] = m_ref_1
    M[2 + n_mot, 2 + n_mot] = m_cx + m_belt_eff
    M[3 + n_mot, 3 + n_mot] = m_cy + m_belt_eff

    # Structural frame stiffness & Rayleigh/modal damping: k_struct = (m_f + m_c) * omega_struct^2
    omega_sx = 2.0 * math.pi * max(f_sx, 5.0)
    omega_sy = 2.0 * math.pi * max(f_sy, 5.0)
    k_sx = (m_fx + m_cx) * (omega_sx**2)
    k_sy = (m_fy + m_cy) * (omega_sy**2)
    c_sx = 2.0 * zeta_frame * math.sqrt(k_sx * (m_fx + m_cx))
    c_sy = 2.0 * zeta_frame * math.sqrt(k_sy * (m_fy + m_cy))
    K[0, 0] += k_sx
    K[1, 1] += k_sy
    C[0, 0] += c_sx
    C[1, 1] += c_sy

    # Motor electromagnetic stiffness & damping relative to motor housing on frame
    for i in range(n_mot):
        K[2 + i, 2 + i] += k_em_arr[i]
        C[2 + i, 2 + i] += b_em_arr[i]

    # Mode-space torque FF compensates CoreXY branch pulling/pushing asymmetry (~35% cross-coupling reduction)
    k_branches_eff = k_branches.copy()
    if is_foc and cfg.mode_space_torque_ff and arch_kind in ("corexy_awd", "corexy_2wd"):
        k_mean = float(np.mean(k_branches_eff))
        k_branches_eff = k_mean + 0.65 * (k_branches_eff - k_mean)

    # Assemble K_belt = E^T diag(k_branches) E where E = [-T_cart, -I_nmot, +T_cart]
    E_belt = np.zeros((n_mot, n_dof), dtype=float)
    E_belt[:, idx_f] = -T_cart
    E_belt[:, idx_m] = -np.eye(n_mot, dtype=float)
    E_belt[:, idx_c] = T_cart
    K_belt_full = E_belt.T @ np.diag(k_branches_eff) @ E_belt
    K += K_belt_full

    # Elastomer belt hysteresis damping proportional to K_belt (material/temp/tension dependent + AWD sync delta)
    k_drive_ref = max(float(np.mean(np.diag(k_cart))), 1e3)
    m_avg = 0.5 * (m_cx + m_cy)
    omega_drive_ref = math.sqrt(k_drive_ref / max(m_avg, 0.05))
    zeta_belt = belt_damping_ratio(belt, chamber_temp_c=spec.frame.chamber_temp_c) + delta_zeta_belt
    beta_belt = (2.0 * zeta_belt) / max(omega_drive_ref, 10.0)
    C += beta_belt * K_belt_full

    # Toolhead IMU skyhook velocity feedback damping on carriage/nozzle DOFs (idx_c)
    if is_foc and cfg.toolhead_imu_active_damping:
        if arch_kind == "cartesian":
            k_em_axis_x = max(float(k_em_arr[0]), 1.0)
            k_em_axis_y = max(float(k_em_arr[1]), 1.0)
        else:
            n_mot_axis = (
                4.0
                if arch_kind == "corexy_awd"
                else (2.0 if arch_kind == "corexy_2wd" else (j_norm if arch_kind == "delta" else 1.0))
            )
            k_em_axis_x = max(n_mot_axis * float(k_em_arr[0]), 1.0)
            k_em_axis_y = k_em_axis_x
        k_drv_x = 1.0 / (1.0 / max(float(k_cart[0, 0]), 1.0) + 1.0 / k_em_axis_x)
        k_drv_y = 1.0 / (1.0 / max(float(k_cart[1, 1]), 1.0) + 1.0 / k_em_axis_y)
        omega_dx = math.sqrt(k_drv_x / max(m_cx, 0.05))
        omega_dy = math.sqrt(k_drv_y / max(m_cy, 0.05))
        omega_eff_x = 1.0 / math.sqrt(1.0 / (omega_sx**2) + 1.0 / (omega_dx**2))
        omega_eff_y = 1.0 / math.sqrt(1.0 / (omega_sy**2) + 1.0 / (omega_dy**2))
        omega_sky_x = max(omega_eff_x, 0.95 * omega_dx)
        omega_sky_y = max(omega_eff_y, 0.95 * omega_dy)
        C[2 + n_mot, 2 + n_mot] += 2.0 * 0.07 * (m_cx + m_belt_eff) * omega_sky_x
        C[3 + n_mot, 3 + n_mot] += 2.0 * 0.07 * (m_cy + m_belt_eff) * omega_sky_y

    # Input matrix F_cmd (n_dof x 2): motor command u_cart = [u_x, u_y] -> T_cart @ u_cart
    F_cmd = np.zeros((n_dof, 2), dtype=float)
    F_cmd[idx_m, :] = k_em_arr[:, None] * T_cart

    M_inv = np.diag(1.0 / np.diag(M))
    A_mech = np.zeros((2 * n_dof, 2 * n_dof), dtype=float)
    A_mech[:n_dof, n_dof:] = np.eye(n_dof, dtype=float)
    A_mech[n_dof:, :n_dof] = -M_inv @ K
    A_mech[n_dof:, n_dof:] = -M_inv @ C

    B_mech = np.zeros((2 * n_dof, 2), dtype=float)
    B_mech[n_dof:, :] = M_inv @ F_cmd

    C_mech = np.zeros((2, 2 * n_dof), dtype=float)
    C_mech[0, 2 + n_mot] = 1.0
    C_mech[1, 3 + n_mot] = 1.0
    D_mech = np.zeros((2, 2), dtype=float)

    if cfg.notch_filter_hz is not None and cfg.notch_filter_hz > 0.0:
        use_db_q = getattr(cfg, "_use_db_q_notch", False) or (
            cfg.notch_zeta_zero == 0.03
            and cfg.notch_zeta_pole == 0.35
            and (cfg.notch_depth_db != 12.0 or cfg.notch_q != 1.75)
        )
        if use_db_q:
            zz, zp, _ = biquad_notch_params_from_db_q(cfg.notch_depth_db, cfg.notch_q)
        else:
            zz, zp = cfg.notch_zeta_zero, cfg.notch_zeta_pole
        A, B, C_noz, D = _cascade_biquad_notch(
            A_mech,
            B_mech,
            C_mech,
            D_mech,
            f_notch_hz=cfg.notch_filter_hz,
            zeta_zero=zz,
            zeta_pole=zp,
        )
    else:
        A, B, C_noz, D = A_mech, B_mech, C_mech, D_mech

    # Extract eigenvalues & identify dominant X and Y modes by nozzle receptance residue
    eigvals, eigvecs = np.linalg.eig(A_mech)
    v_inv_b = np.linalg.solve(eigvecs, B_mech)
    c_v = C_mech @ eigvecs

    osc_modes: list[tuple[float, float, float, float]] = []
    for k_idx, lam in enumerate(eigvals):
        if lam.imag <= 1.0:
            continue
        wn = abs(lam)
        freq_hz = wn / (2.0 * math.pi)
        zeta = max(0.0, min(0.999, -float(lam.real) / max(wn, 1e-12)))
        # Corner-ringing displacement-energy weight for velocity-step turn: |R| / (zeta * wn^3)
        denom_w = max(zeta, 0.005) * (max(wn, 1.0) ** 3)
        res_xx = abs(c_v[0, k_idx] * v_inv_b[k_idx, 0]) / denom_w
        res_yy = abs(c_v[1, k_idx] * v_inv_b[k_idx, 1]) / denom_w
        osc_modes.append((freq_hz, zeta, res_xx, res_yy))

    osc_modes.sort(key=lambda item: item[0])
    mode_freqs = [item[0] for item in osc_modes]
    mode_dampings = [item[1] for item in osc_modes]

    if osc_modes:
        best_x = max(osc_modes, key=lambda item: item[2])
        best_y = max(osc_modes, key=lambda item: item[3])
        freq_x, zeta_x = best_x[0], best_x[1]
        freq_y, zeta_y = best_y[0], best_y[1]
    else:
        freq_x, zeta_x = f_sx, zeta_frame
        freq_y, zeta_y = f_sy, zeta_frame

    # Evaluate MIMO FRF over [5, 300] Hz to measure X-Y cross coupling
    f_lo = max(5.0, 0.4 * min(freq_x, freq_y))
    f_hi = min(320.0, max(200.0, 2.2 * max(freq_x, freq_y)))
    freqs_hz = np.linspace(f_lo, f_hi, 72, dtype=float)
    h_frf = _eval_mimo_frf(A, B, C_noz, D, freqs_hz)
    h_xx_max = max(float(np.max(np.abs(h_frf[:, 0, 0]))), 1e-12)
    h_xy_max = float(np.max(np.abs(h_frf[:, 0, 1])))
    xy_cross_coupling_pct = 100.0 * h_xy_max / h_xx_max

    return CoupledMechatronicsModel(
        A=A,
        B=B,
        C_noz=C_noz,
        D=D,
        n_mot=n_mot,
        m_struct=m_struct,
        u_rel=u_rel,
        v_rel=v_rel,
        drive_cfg=cfg,
        k_belt_cart=k_cart,
        k_A_n_m=k_a,
        k_B_n_m=k_b,
        k_em_per_motor_n_m=float(np.mean(k_em_arr)),
        b_em_per_motor_ns_m=float(np.mean(b_em_arr)),
        reflected_rotor_mass_per_motor_g=m_ref_1 * 1e3,
        reflected_rotor_mass_g=reflected_per_axis * 1e3,
        coupled_freq_x_hz=freq_x,
        coupled_freq_y_hz=freq_y,
        coupled_damping_x=zeta_x,
        coupled_damping_y=zeta_y,
        mode_freqs_hz=mode_freqs,
        mode_dampings=mode_dampings,
        freqs_hz=freqs_hz,
        H_frf=h_frf,
        xy_cross_coupling_pct=xy_cross_coupling_pct,
        midband_instability_risk=midband_risk,
        T_cart=T_cart,
        k_em_per_motor_arr=k_em_arr,
        b_em_per_motor_arr=b_em_arr,
    )


def compute_workspace_frf_map(
    assembly: AssemblyModel,
    modal_result: ModalResult,
    grid_n: int = 5,
    drive_cfg: DriveControlConfig | None = None,
) -> WorkspaceFRFMap:
    """Compute position-dependent `(grid_n x grid_n)` workspace frequency & cross-coupling map."""
    n = max(2, int(grid_n))
    coords = np.linspace(0.05, 0.95, n, dtype=float)
    cfg = drive_cfg or DriveControlConfig()

    base_fx = (
        float(modal_result.shaper_x.structural_freq_hz)
        if modal_result.shaper_x and modal_result.shaper_x.structural_freq_hz > 0.0
        else max(float(modal_result.fundamental_freq_hz), 15.0)
    )
    base_fy = (
        float(modal_result.shaper_y.structural_freq_hz)
        if modal_result.shaper_y and modal_result.shaper_y.structural_freq_hz > 0.0
        else max(float(modal_result.fundamental_freq_hz), 15.0)
    )

    fx_grid: list[list[float]] = []
    fy_grid: list[list[float]] = []
    min_grid: list[list[float]] = []
    cc_grid: list[list[float]] = []

    reflected_g = 0.0
    midband_any = False

    # Note: row index = v_rel (Y), col index = u_rel (X)
    for v_val in coords:
        row_fx: list[float] = []
        row_fy: list[float] = []
        row_min: list[float] = []
        row_cc: list[float] = []
        for u_val in coords:
            # Beam bending span modulation:
            # - X gantry beam bends maximally in Y at midspan u=0.5 (4u(1-u)=1) and stiffens near carriages
            # - Y rails & open front portal soften slightly at mid-rail v=0.5 & front v->0.05
            span_u = 4.0 * float(u_val) * (1.0 - float(u_val))
            span_v = 4.0 * float(v_val) * (1.0 - float(v_val))
            comp_y_scale = 1.0 + 0.22 * (span_u - 1.0)
            comp_x_scale = 1.0 + 0.14 * (span_v - 1.0) + 0.08 * (0.5 - float(v_val))
            if (
                assembly.spec.architecture == "corexy"
                and not assembly.spec.frame.rear_bottom_extrusion
            ):
                # Omitting lower rear tie extrusion leaves the rear portal open to torsional splay
                # and racking, softening rear-quadrant structural modes and widening spatial IS spread.
                rear_weight = (float(v_val) ** 1.5) * (0.65 + 1.35 * abs(float(u_val) - 0.5))
                comp_x_scale += 1.05 * rear_weight
                comp_y_scale += 0.90 * rear_weight
            f_sx_uv = base_fx / math.sqrt(max(comp_x_scale, 0.6))
            f_sy_uv = base_fy / math.sqrt(max(comp_y_scale, 0.6))

            mdl = build_coupled_state_space(
                assembly,
                modal_result,
                u_rel=float(u_val),
                v_rel=float(v_val),
                drive_cfg=cfg,
                omega_x_override_hz=f_sx_uv,
                omega_y_override_hz=f_sy_uv,
            )
            reflected_g = mdl.reflected_rotor_mass_g
            midband_any = midband_any or mdl.midband_instability_risk
            row_fx.append(mdl.coupled_freq_x_hz)
            row_fy.append(mdl.coupled_freq_y_hz)
            row_min.append(min(mdl.coupled_freq_x_hz, mdl.coupled_freq_y_hz))
            row_cc.append(mdl.xy_cross_coupling_pct)
        fx_grid.append(row_fx)
        fy_grid.append(row_fy)
        min_grid.append(row_min)
        cc_grid.append(row_cc)

    center_mdl = build_coupled_state_space(
        assembly, modal_result, u_rel=0.5, v_rel=0.5, drive_cfg=cfg
    )
    center_freq = min(center_mdl.coupled_freq_x_hz, center_mdl.coupled_freq_y_hz)
    all_min = [val for row in min_grid for val in row]
    all_max = [max(fx_grid[r][c], fy_grid[r][c]) for r in range(n) for c in range(n)]
    all_cc = [val for row in cc_grid for val in row]

    min_freq = min(all_min)
    max_freq = max(all_max)
    worst_drop_pct = max(0.0, 100.0 * (center_freq - min_freq) / max(center_freq, 1e-6))
    is_spatial_spread_pct = max(
        0.0, 100.0 * (max(all_min) - min(all_min)) / max(center_freq, 1e-6)
    )

    return WorkspaceFRFMap(
        grid_n=n,
        u_coords=[float(u) for u in coords],
        v_coords=[float(v) for v in coords],
        freq_x_grid_hz=fx_grid,
        freq_y_grid_hz=fy_grid,
        min_axis_freq_grid_hz=min_grid,
        xy_cross_coupling_grid_pct=cc_grid,
        center_freq_hz=center_freq,
        min_freq_hz=min_freq,
        max_freq_hz=max_freq,
        worst_corner_drop_pct=worst_drop_pct,
        max_cross_coupling_pct=max(all_cc),
        reflected_rotor_mass_g=reflected_g,
        drive_type=cfg.drive_type,
        midband_instability_risk=midband_any,
        is_spatial_spread_pct=is_spatial_spread_pct,
    )
