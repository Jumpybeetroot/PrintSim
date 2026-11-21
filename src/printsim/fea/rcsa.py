"""Receptance Coupling Substructure Analysis (RCSA / LM-FBS) for Toolhead-Hotend-Nozzle.

Implements:
- Exact 4x4 analytical Timoshenko beam receptance matrix ``beam_tip_receptance_4x4``
  for flexural bending + transverse shear + rotary inertia over ``[u1, theta1, u2, theta2]``.
- Dual Lagrange-Multiplier Frequency-Based Substructuring (LM-FBS, de Klerk, Rixen &
  Voormeeren 2008) ``Y_coupled = Y - Y B^T (B Y B^T + Y_joint)^{-1} B Y``.
- Toolhead-to-nozzle whip & ADXL345 discrepancy analyzer ``analyze_toolhead_rcsa``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import block_diag, expm

from printsim.fea.components import hertzian_rail_carriage_stiffness
from printsim.fea.modal import ModalResult
from printsim.geometry.builder import AssemblyModel, compute_toolhead_link_props
from printsim.schema import get_material


@dataclass
class RCSAResult:
    """Frequency-domain toolhead/hotend/nozzle RCSA whip & stiffness report."""

    sensor_peak_freq_hz: float
    nozzle_tip_peak_freq_hz: float
    nozzle_to_sensor_amp_ratio: float
    hotend_bending_stiffness_n_per_mm: float
    whip_risk: bool
    warning_message: str
    freqs_hz: np.ndarray
    sensor_frf_mag: np.ndarray
    nozzle_frf_mag: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "sensor_peak_freq_hz": round(float(self.sensor_peak_freq_hz), 2),
            "nozzle_tip_peak_freq_hz": round(float(self.nozzle_tip_peak_freq_hz), 2),
            "nozzle_to_sensor_amp_ratio": round(float(self.nozzle_to_sensor_amp_ratio), 3),
            "hotend_bending_stiffness_n_per_mm": round(
                float(self.hotend_bending_stiffness_n_per_mm), 2
            ),
            "whip_risk": bool(self.whip_risk),
            "warning_message": self.warning_message,
            "freqs_hz": [round(float(f), 2) for f in self.freqs_hz],
            "sensor_frf_mag": [float(f"{v:.6e}") for v in self.sensor_frf_mag],
            "nozzle_frf_mag": [float(f"{v:.6e}") for v in self.nozzle_frf_mag],
        }


def _timoshenko_receptance_batch(
    L: float,
    Es: complex,
    I: float,
    Gs: complex,
    A: float,
    rho: float,
    w: np.ndarray,
    kappa: float,
) -> np.ndarray:
    """Evaluate exact 4x4 Timoshenko receptance across angular frequency vector ``w``."""
    nf = len(w)
    w2 = w * w
    Amat = np.zeros((nf, 4, 4), dtype=complex)
    Amat[:, 0, 1] = 1.0
    Amat[:, 0, 3] = 1.0 / (kappa * Gs * A)
    Amat[:, 1, 2] = 1.0 / (Es * I)
    Amat[:, 2, 1] = -rho * I * w2
    Amat[:, 2, 3] = -1.0
    Amat[:, 3, 0] = -rho * A * w2

    # Bound matrix exponential argument norm to avoid overflow for ultra-slender high-f beams
    beta_l = float(np.max(((rho * A * w2 / max(abs(Es) * I, 1e-18)) ** 0.25) * L))
    if beta_l <= 14.0:
        U = expm(Amat * L)
        U11, U12 = U[:, :2, :2], U[:, :2, 2:]
        U21, U22 = U[:, 2:, :2], U[:, 2:, 2:]
        J = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)

        lhs = np.zeros((nf, 4, 4), dtype=complex)
        lhs[:, :2, :2] = U11
        lhs[:, :2, 2:] = -np.eye(2, dtype=complex)
        lhs[:, 2:, :2] = U21

        rhs = np.zeros((nf, 4, 4), dtype=complex)
        rhs[:, :2, :2] = U12 @ J
        rhs[:, 2:, :2] = U22 @ J
        rhs[:, 2:, 2:] = J

        try:
            H = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            H = np.empty((nf, 4, 4), dtype=complex)
            for k in range(nf):
                H[k] = np.linalg.pinv(lhs[k]) @ rhs[k]
        return 0.5 * (H + np.swapaxes(H, 1, 2))

    # Subdivide into n_seg segments and couple recursively in receptance space via LM-FBS
    n_seg = int(math.ceil(beta_l / 10.0))
    H_seg = _timoshenko_receptance_batch(L / n_seg, Es, I, Gs, A, rho, w, kappa)
    H_curr = H_seg
    B_pair = np.zeros((2, 8), dtype=float)
    B_pair[0, 2], B_pair[0, 4] = 1.0, -1.0
    B_pair[1, 3], B_pair[1, 5] = 1.0, -1.0
    keep = [0, 1, 6, 7]
    for _ in range(n_seg - 1):
        H_cpl = lm_fbs_couple([H_curr, H_seg], B_pair)
        H_curr = H_cpl[:, np.ix_(keep, keep)[0], np.ix_(keep, keep)[1]]
    return 0.5 * (H_curr + np.swapaxes(H_curr, 1, 2))


def beam_tip_receptance_4x4(
    L: float,
    E: float,
    I: float,
    G: float,
    A: float,
    rho: float,
    freqs_hz: float | Sequence[float] | np.ndarray,
    zeta: float = 0.03,
    kappa: float = 5.0 / 6.0,
) -> np.ndarray:
    """Compute analytical 4x4 Timoshenko beam receptance matrix for ``[u1, theta1, u2, theta2]``.

    Uses complex hysteretic moduli ``E* = E (1 + 2j*zeta)`` and ``G* = G (1 + 2j*zeta)``.
    Returns shape ``(4, 4)`` if ``freqs_hz`` is scalar, or ``(n_f, 4, 4)`` for an array.
    """
    if L <= 0.0 or E <= 0.0 or I <= 0.0 or G <= 0.0 or A <= 0.0 or rho <= 0.0:
        raise ValueError("Beam geometric and material parameters (L, E, I, G, A, rho) must be > 0")

    is_scalar = np.ndim(freqs_hz) == 0
    f_arr = np.atleast_1d(np.asarray(freqs_hz, dtype=float))
    w = 2.0 * math.pi * np.maximum(f_arr, 1e-3)

    eta = 2.0 * max(float(zeta), 0.0)
    Es = complex(E, E * eta)
    Gs = complex(G, G * eta)
    H = _timoshenko_receptance_batch(float(L), Es, float(I), Gs, float(A), float(rho), w, float(kappa))
    return H[0] if is_scalar else H


def lm_fbs_couple(
    Y_list: Sequence[np.ndarray],
    B_compat: np.ndarray,
    Y_joint: np.ndarray | None = None,
) -> np.ndarray:
    """Exact dual Lagrange-Multiplier Frequency-Based Substructuring (de Klerk et al., 2008).

    Computes::

        Y_coupled = Y - Y @ B^T @ (B @ Y @ B^T + Y_joint)^{-1} @ B @ Y

    Supports both 2D single-frequency ``(n_s, n_s)`` arrays and batched 3D
    ``(n_f, n_s, n_s)`` receptance stacks.
    """
    if not Y_list:
        raise ValueError("Y_list must contain at least one substructure receptance matrix")
    B = np.asarray(B_compat, dtype=complex)

    first = np.asarray(Y_list[0], dtype=complex)
    if first.ndim == 2:
        Y = block_diag(*[np.asarray(y, dtype=complex) for y in Y_list])
        YB = Y @ B.T
        BYB = B @ YB
        if Y_joint is not None:
            BYB = BYB + np.asarray(Y_joint, dtype=complex)
        BY = B @ Y
        try:
            sol = np.linalg.solve(BYB, BY)
        except np.linalg.LinAlgError:
            sol = np.linalg.pinv(BYB) @ BY
        Yc = Y - YB @ sol
        return 0.5 * (Yc + Yc.T)

    if first.ndim == 3:
        nf = first.shape[0]
        N = sum(np.asarray(y).shape[1] for y in Y_list)
        Y = np.zeros((nf, N, N), dtype=complex)
        off = 0
        for y in Y_list:
            ya = np.asarray(y, dtype=complex)
            ns = ya.shape[1]
            Y[:, off : off + ns, off : off + ns] = ya
            off += ns

        if B.ndim == 2:
            YB = Y @ B.T
            BYB = B @ YB
            BY = B @ Y
        else:
            YB = Y @ np.swapaxes(B, 1, 2)
            BYB = B @ YB
            BY = B @ Y

        if Y_joint is not None:
            BYB = BYB + np.asarray(Y_joint, dtype=complex)

        try:
            sol = np.linalg.solve(BYB, BY)
        except np.linalg.LinAlgError:
            sol = np.empty_like(BY)
            for k in range(nf):
                sol[k] = np.linalg.pinv(BYB[k]) @ BY[k]

        Yc = Y - YB @ sol
        return 0.5 * (Yc + np.swapaxes(Yc, 1, 2))

    raise ValueError(f"Expected 2D or 3D receptance arrays in Y_list, got ndim={first.ndim}")


def _effective_member_stiffness_scale(assembly: AssemblyModel) -> float:
    """Detect direct stiffness/material mutations on toolhead_mount / nozzle_stem members."""
    spec = assembly.spec
    nmap = assembly.node_map
    base_node = nmap.get(assembly.toolhead_node_id)
    cog_node = nmap.get(assembly.toolhead_cog_node_id)
    noz_node = nmap.get(assembly.nozzle_node_id)
    l_mount = (
        float(np.linalg.norm(cog_node.position_m - base_node.position_m))
        if base_node is not None and cog_node is not None
        else float(np.linalg.norm(np.array(spec.toolhead.cog_offset_mm) * 1e-3))
    )
    l_stem = (
        float(np.linalg.norm(noz_node.position_m - cog_node.position_m))
        if noz_node is not None and cog_node is not None
        else 0.04
    )
    props = compute_toolhead_link_props(spec, l_mount_m=l_mount, l_stem_m=l_stem)

    mount_scale = 1.0
    mount_mem = next((m for m in assembly.members if m.id == "toolhead_mount"), None)
    if mount_mem is not None:
        nom_ei = props["e_mount_eff"] * (props["iy_mount"] + props["iz_mount"])
        actual_ei = mount_mem.material.youngs_modulus_pa * (
            mount_mem.profile.iy_m4 + mount_mem.profile.iz_m4
        )
        if nom_ei > 0.0 and actual_ei > 0.0:
            mount_scale = actual_ei / nom_ei

    stem_scale = 1.0
    stem_mem = next((m for m in assembly.members if m.id == "nozzle_stem"), None)
    if stem_mem is not None:
        nom_stem_ei = props["e_stem"] * (2.0 * props["i_stem"])
        actual_stem_ei = stem_mem.material.youngs_modulus_pa * (
            stem_mem.profile.iy_m4 + stem_mem.profile.iz_m4
        )
        if nom_stem_ei > 0.0 and actual_stem_ei > 0.0:
            stem_scale = actual_stem_ei / nom_stem_ei

    return float(max(mount_scale * stem_scale, 1e-3))


def analyze_toolhead_rcsa(
    assembly: AssemblyModel,
    modal_result: ModalResult,
    freqs_hz: np.ndarray | None = None,
    mount_stiffness_scale: float = 1.0,
) -> RCSAResult:
    """Couple carriage modal receptance with toolhead mount + heatbreak + block via LM-FBS.

    Compares the ADXL345 sensor peak frequency/amplitude on the toolhead carriage against
    the actual nozzle tip peak frequency/amplitude, flagging ``"Nozzle-to-ADXL345 whip risk"``
    whenever ``nozzle_to_sensor_amp_ratio > 1.35``.
    """
    if freqs_hz is None:
        f_grid = np.linspace(10.0, 450.0, 441)
    else:
        f_grid = np.asarray(freqs_hz, dtype=float)
    w = 2.0 * math.pi * np.maximum(f_grid, 1e-3)
    nf = len(f_grid)

    spec = assembly.spec
    th = spec.toolhead
    he = th.hotend
    chamber_c = spec.frame.chamber_temp_c

    # 1. Substructure parameters: Mount/Heatsink, Heatbreak, Heater Block + Nozzle
    member_scale = _effective_member_stiffness_scale(assembly)
    eff_scale = max(float(mount_stiffness_scale), 1e-4) * member_scale

    mmat = get_material(th.mount_material)
    e_mount = mmat.effective_modulus_pa(chamber_c) * eff_scale
    g_mount = e_mount / (2.0 * (1.0 + mmat.poisson_ratio))
    L_mount = max(abs(float(th.cog_offset_mm[2])) * 1e-3, 0.012)
    I_mount = 4.5e-9
    A_mount = 1.8e-4
    m_mount = max(th.total_mass_g * 1e-3 - he.block_mass_g * 1e-3 - 0.05, 0.05)
    rho_mount = m_mount / (A_mount * L_mount)
    Y_mount = beam_tip_receptance_4x4(
        L_mount, e_mount, I_mount, g_mount, A_mount, rho_mount, f_grid, zeta=mmat.damping_ratio
    )

    hb_mat = get_material(he.heatbreak_material)
    ri = 0.5 * he.heatbreak_id_mm * 1e-3
    ro = ri + he.heatbreak_wall_mm * 1e-3
    A_hb = math.pi * max(ro**2 - ri**2, 1e-8)
    I_hb_bare = 0.25 * math.pi * max(ro**4 - ri**4, 1e-14)
    # Parallel heatsink collar/standoff stiffening scaled by mount modulus
    I_hb = I_hb_bare + 8.0e-11 * min(e_mount / 2.1e9, 15.0)
    L_hb = max(he.heatbreak_length_mm * 1e-3, 0.004)
    m_hb = A_hb * L_hb * hb_mat.density_kg_m3
    Y_hb = beam_tip_receptance_4x4(
        L_hb,
        hb_mat.youngs_modulus_pa,
        I_hb,
        hb_mat.effective_shear_modulus_pa,
        A_hb,
        hb_mat.density_kg_m3,
        f_grid,
        zeta=0.025,
    )

    blk_mat = get_material(he.block_material)
    L_noz_drop = abs(float(th.nozzle_offset_mm[2])) * 1e-3
    L_blk = max(L_noz_drop - L_mount - L_hb, he.melt_zone_length_mm * 1e-3, 0.006)
    d_blk = 0.011
    A_blk = math.pi * (0.5 * d_blk) ** 2
    I_blk = 0.25 * math.pi * (0.5 * d_blk) ** 4
    m_blk = max(he.block_mass_g * 1e-3 + 0.008, 0.010)
    rho_blk = m_blk / (A_blk * L_blk)
    Y_blk = beam_tip_receptance_4x4(
        L_blk,
        blk_mat.youngs_modulus_pa,
        I_blk,
        blk_mat.effective_shear_modulus_pa,
        A_blk,
        rho_blk,
        f_grid,
        zeta=0.025,
    )

    # 2. Extract carriage interface receptance from modal_result (decoupling attached mass)
    m_attached = m_mount + m_hb + m_blk
    th_modes = [
        m
        for m in modal_result.modes
        if m.frequency_hz >= 5.0
        and float(np.linalg.norm(m.nodal_vectors.get(assembly.toolhead_node_id, [1.0, 0.0, 0.0]))) > 0.05
    ]
    dom_mode = th_modes[0] if th_modes else (modal_result.modes[0] if modal_result.modes else None)
    f0 = max(
        dom_mode.frequency_hz if dom_mode is not None else modal_result.fundamental_freq_hz,
        15.0,
    )
    w0 = 2.0 * math.pi * f0
    z_f = max(float(modal_result.frame_damping_ratio), 0.015)

    dom_axis = dom_mode.dominant_axis if dom_mode is not None else "Y"
    m_axis = (
        assembly.y_moving_mass_kg
        if dom_axis == "Y"
        else (assembly.x_moving_mass_kg + 0.25)
    )
    m_modal = max(m_axis, m_attached + 0.25)
    k_modal = m_modal * (w0**2)
    m_bare = max(m_modal - m_attached, 0.10)

    if spec.architecture != "delta":
        car = hertzian_rail_carriage_stiffness(
            spec.gantry.x_rail_profile,
            preload=spec.gantry.rail_preload,
            block_type=spec.gantry.rail_block_type,
        )
        car_mult = 2.0 if spec.gantry.effective_dual_x_rails else 1.0
        k_rot_car = car["k_rx_nm_rad"] * car_mult
    else:
        k_rot_car = 2500.0
    j_car = 0.5 * (0.022**2) * m_bare

    Y_car = np.zeros((nf, 2, 2), dtype=complex)
    Y_car[:, 0, 0] = 1.0 / (k_modal - (w**2) * m_bare + 2j * z_f * w0 * w * m_modal)
    Y_car[:, 1, 1] = 1.0 / (
        k_rot_car
        - (w**2) * j_car
        + 1j * 2.0 * z_f * math.sqrt(k_rot_car * j_car) * (w / max(w0, 1.0))
    )

    # 3. Bolted joint impedances Z_j = K_j + i*w*C_j (Carriage-Mount J1 & Mount-Hotend J2)
    e_ratio = e_mount / 2.1e9
    k_u1 = 6.0e6 * e_ratio / (1.0 + 0.08 * e_ratio)
    k_t1 = 150.0 * e_ratio / (1.0 + 0.04 * e_ratio)
    k_u2 = 8.0e6 * e_ratio / (1.0 + 0.08 * e_ratio)
    k_t2 = 170.0 * e_ratio / (1.0 + 0.04 * e_ratio)

    # Wire user's internal_stiffness_n_per_mm (N/m, T-modulated) in series with J1 carriage-mount joint
    k_int_n_m = (
        float(th.internal_stiffness_n_per_mm)
        * 1e3
        * mmat.modulus_retention_at_temp(chamber_c)
        * eff_scale
    )
    k_int_u = k_int_n_m * 2.0 * (max(e_ratio, 0.1) ** 0.35)
    k_int_rot = k_int_n_m * (0.045**2) * (max(e_ratio, 0.1) ** 0.65)
    k_u1 = (k_u1 * k_int_u) / max(k_u1 + k_int_u, 1e-9)
    k_t1 = (k_t1 * k_int_rot) / max(k_t1 + k_int_rot, 1e-9)

    zeta_j = 0.04
    c_u1 = 2.0 * zeta_j * math.sqrt(k_u1 * m_mount)
    c_t1 = 2.0 * zeta_j * math.sqrt(k_t1 * m_mount * (L_mount**2))
    c_u2 = 2.0 * zeta_j * math.sqrt(k_u2 * m_blk)
    c_t2 = 2.0 * zeta_j * math.sqrt(k_t2 * m_blk * ((L_hb + L_blk) ** 2))

    Y_joint = np.zeros((nf, 6, 6), dtype=complex)
    Y_joint[:, 0, 0] = 1.0 / (k_u1 + 1j * w * c_u1)
    Y_joint[:, 1, 1] = 1.0 / (k_t1 + 1j * w * c_t1)
    Y_joint[:, 2, 2] = 1.0 / (k_u2 + 1j * w * c_u2)
    Y_joint[:, 3, 3] = 1.0 / (k_t2 + 1j * w * c_t2)

    # Compatibility matrix B (6 constraints on 14 DOFs: car[0:2], mount[2:6], hb[6:10], blk[10:14])
    B = np.zeros((6, 14), dtype=float)
    B[0, 0], B[0, 2] = 1.0, -1.0
    B[1, 1], B[1, 3] = 1.0, -1.0
    B[2, 4], B[2, 6] = 1.0, -1.0
    B[3, 5], B[3, 7] = 1.0, -1.0
    B[4, 8], B[4, 10] = 1.0, -1.0
    B[5, 9], B[5, 11] = 1.0, -1.0

    Y_coupled = lm_fbs_couple([Y_car, Y_mount, Y_hb, Y_blk], B, Y_joint=Y_joint)

    sensor_frf_mag = np.abs(Y_coupled[:, 0, 0])
    nozzle_frf_mag = np.abs(Y_coupled[:, 12, 0])

    i_sen = int(np.argmax(sensor_frf_mag))
    i_noz = int(np.argmax(nozzle_frf_mag))
    sensor_peak_freq_hz = float(f_grid[i_sen])
    nozzle_tip_peak_freq_hz = float(f_grid[i_noz])

    peak_sen = max(float(sensor_frf_mag[i_sen]), 1e-18)
    peak_noz = float(nozzle_frf_mag[i_noz])
    whip_ratio = max(1.0, peak_noz / peak_sen)

    # 4. Quasi-static cantilever bending stiffness of [J1 + mount + J2 + heatbreak + block]
    f_stat = np.array([0.5], dtype=float)
    Y_m0 = beam_tip_receptance_4x4(L_mount, e_mount, I_mount, g_mount, A_mount, rho_mount, f_stat, zeta=0.0)
    Y_hb0 = beam_tip_receptance_4x4(
        L_hb,
        hb_mat.youngs_modulus_pa,
        I_hb,
        hb_mat.effective_shear_modulus_pa,
        A_hb,
        hb_mat.density_kg_m3,
        f_stat,
        zeta=0.0,
    )
    Y_b0 = beam_tip_receptance_4x4(
        L_blk,
        blk_mat.youngs_modulus_pa,
        I_blk,
        blk_mat.effective_shear_modulus_pa,
        A_blk,
        rho_blk,
        f_stat,
        zeta=0.0,
    )
    Y_j_hot = np.zeros((1, 4, 4), dtype=complex)
    Y_j_hot[0, 0, 0] = 1.0 / k_u2
    Y_j_hot[0, 1, 1] = 1.0 / k_t2
    Y_hot0 = lm_fbs_couple([Y_m0, Y_hb0, Y_b0], B[2:, 2:], Y_joint=Y_j_hot)[0]
    c_cant = Y_hot0[10, 10] - Y_hot0[10, :2] @ np.linalg.solve(Y_hot0[:2, :2], Y_hot0[:2, 10])
    L_tot = L_mount + L_hb + L_blk
    c_total_m_per_n = max(float(abs(c_cant.real)) + 1.0 / k_u1 + (L_tot**2) / k_t1, 1e-12)
    hotend_bending_stiffness_n_per_mm = 1.0 / (c_total_m_per_n * 1e3)

    whip_risk = whip_ratio > 1.35
    warning_msg = (
        f"Nozzle-to-ADXL345 whip risk ({whip_ratio:.2f}x > 1.35x at {nozzle_tip_peak_freq_hz:.1f} Hz): "
        f"upgrade toolhead mount from '{th.mount_material}' to 'ppa_cf_printed' or 'aluminum_6061_t6'."
        if whip_risk
        else ""
    )

    return RCSAResult(
        sensor_peak_freq_hz=sensor_peak_freq_hz,
        nozzle_tip_peak_freq_hz=nozzle_tip_peak_freq_hz,
        nozzle_to_sensor_amp_ratio=whip_ratio,
        hotend_bending_stiffness_n_per_mm=hotend_bending_stiffness_n_per_mm,
        whip_risk=whip_risk,
        warning_message=warning_msg,
        freqs_hz=f_grid,
        sensor_frf_mag=sensor_frf_mag,
        nozzle_frf_mag=nozzle_frf_mag,
    )
