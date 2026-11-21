"""3D Modal Finite Element Analysis (K phi = omega^2 M phi) & Input Shaper Tuning."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import LinearOperator, eigsh, splu

from printsim.fea.beam3d import assemble_global_matrices
from printsim.fea.klipper_shaper import AxisShaperGraph, calibrate_axis_shapers
from printsim.geometry.builder import AssemblyModel
from printsim.kinematics.solvers import (
    compute_bracket_stiffness_n_per_m,
    compute_drive_stiffness_n_per_m,
)


_BOTTLENECK_LABELS: dict[str, str] = {
    "frame": "Frame & Gantry Structure",
    "belt": "Belt Stretch",
    "motor": "Motor Magnetic Stiffness",
    "bracket": "Gantry Bracket Flex",
}

_BOTTLENECK_TIPS: dict[str, str] = {
    "frame": "Bolt 4-5mm Aluminum 5052 structural panels, upgrade frame bracing/extrusions, or mount X/Y rails for higher composite EI.",
    "belt": "Upgrade to wider 9mm/12mm GT2 Aramid belts, raise belt tension (with double-shear pulleys), or switch to 4-motor AWD.",
    "motor": "Raise stepper bus voltage (48V), increase run current within motor rating, use lower-inductance/0.9° steppers, or switch to AWD.",
    "bracket": "Upgrade motor mounts & XY-joint brackets from ABS to annealed PPA-CF / PPS-CF or CNC Aluminum 6061-T6 (and enable double-shear).",
}


@dataclass
class ModeShape:
    mode_number: int
    frequency_hz: float
    dominant_axis: str
    toolhead_participation_xyz: list[float]
    nodal_vectors: dict[str, list[float]]  # normalized 3D displacement per node for animation


@dataclass
class InputShaperRecommendation:
    axis: str
    effective_freq_hz: float
    structural_freq_hz: float
    belt_freq_hz: float
    motor_freq_hz: float
    drive_freq_hz: float
    recommended_shaper: str  # ZV, MZV, EI
    max_accel_mm_s2: float
    estimated_ringing_ratio: float  # residual vibration amplitude ratio at current max_accel
    bracket_freq_hz: float = 0.0
    compliance_share_pct: dict[str, float] = field(default_factory=dict)
    primary_bottleneck: str = "frame"


@dataclass
class ModalResult:
    fundamental_freq_hz: float
    modes: list[ModeShape] = field(default_factory=list)
    shaper_x: InputShaperRecommendation | None = None
    shaper_y: InputShaperRecommendation | None = None
    shaper_graph_x: AxisShaperGraph | None = None
    shaper_graph_y: AxisShaperGraph | None = None
    recommended_max_accel_mm_s2: float = 0.0
    nozzle_pitch_twist_freq_hz: float = 0.0
    panel_drum_freq_hz: float = 0.0
    frame_damping_ratio: float = 0.042
    accel_bottleneck: dict | None = None

    def to_dict(self) -> dict:
        return {
            "fundamental_freq_hz": round(self.fundamental_freq_hz, 2),
            "nozzle_pitch_twist_freq_hz": round(self.nozzle_pitch_twist_freq_hz, 2),
            "panel_drum_freq_hz": round(self.panel_drum_freq_hz, 2),
            "frame_damping_ratio": round(self.frame_damping_ratio, 4),
            "recommended_max_accel_mm_s2": round(self.recommended_max_accel_mm_s2, 0),
            "accel_bottleneck": self.accel_bottleneck,
            "shaper_x": _shaper_dict(self.shaper_x) if self.shaper_x else None,
            "shaper_y": _shaper_dict(self.shaper_y) if self.shaper_y else None,
            "shaper_graphs": {
                "x": self.shaper_graph_x.to_dict() if self.shaper_graph_x else None,
                "y": self.shaper_graph_y.to_dict() if self.shaper_graph_y else None,
            },
            "modes": [
                {
                    "mode_number": m.mode_number,
                    "frequency_hz": round(m.frequency_hz, 2),
                    "dominant_axis": m.dominant_axis,
                    "toolhead_participation_xyz": [round(v, 3) for v in m.toolhead_participation_xyz],
                    "nodal_vectors": m.nodal_vectors,
                }
                for m in self.modes
            ],
        }


def _shaper_dict(s: InputShaperRecommendation) -> dict:
    return {
        "axis": s.axis,
        "effective_freq_hz": round(s.effective_freq_hz, 2),
        "structural_freq_hz": round(s.structural_freq_hz, 2),
        "belt_freq_hz": round(s.belt_freq_hz, 2),
        "motor_freq_hz": round(s.motor_freq_hz, 2),
        "bracket_freq_hz": round(s.bracket_freq_hz, 2),
        "drive_freq_hz": round(s.drive_freq_hz, 2),
        "recommended_shaper": s.recommended_shaper,
        "max_accel_mm_s2": round(s.max_accel_mm_s2, 0),
        "estimated_ringing_ratio": round(s.estimated_ringing_ratio, 3),
        "primary_bottleneck": s.primary_bottleneck,
        "compliance_share_pct": s.compliance_share_pct,
    }


def _drive_axis_resonances_hz(assembly: AssemblyModel, axis: str) -> tuple[float, float, float, float]:
    """Compute (belt_freq_hz, motor_freq_hz, bracket_freq_hz, drive_freq_hz) for X or Y."""
    k_belt, k_motor_axis, k_drive = compute_drive_stiffness_n_per_m(assembly, axis)
    k_bracket = compute_bracket_stiffness_n_per_m(assembly)
    mass = max(assembly.x_moving_mass_kg if axis == "X" else assembly.y_moving_mass_kg, 0.05)
    c = 1.0 / (2.0 * math.pi)
    return (
        c * math.sqrt(k_belt / mass),
        c * math.sqrt(k_motor_axis / mass),
        c * math.sqrt(k_bracket / mass),
        c * math.sqrt(k_drive / mass),
    )


def _belt_axis_resonance_hz(assembly: AssemblyModel, axis: str) -> float:
    """Compute belt-driven axial spring-mass natural frequency f = (1/2pi) * sqrt(k_belt / m)."""
    return _drive_axis_resonances_hz(assembly, axis)[0]


def _compute_compliance_shares(
    structural_hz: float,
    belt_hz: float,
    motor_hz: float,
    bracket_hz: float,
) -> tuple[dict[str, float], str]:
    """Exact series-spring compliance shares (1/f_i^2) / sum(1/f_j^2) summing to 100.0%."""
    c_frame = 1.0 / max(structural_hz, 5.0) ** 2
    c_belt = 1.0 / max(belt_hz, 5.0) ** 2
    c_motor = 1.0 / max(motor_hz, 5.0) ** 2
    c_bracket = 1.0 / max(bracket_hz, 5.0) ** 2
    c_tot = max(c_frame + c_belt + c_motor + c_bracket, 1e-18)
    raw = {
        "frame": 100.0 * c_frame / c_tot,
        "belt": 100.0 * c_belt / c_tot,
        "motor": 100.0 * c_motor / c_tot,
        "bracket": 100.0 * c_bracket / c_tot,
    }
    primary = max(raw, key=lambda k: raw[k])
    rounded = {k: round(v, 1) for k, v in raw.items()}
    # Keep rounded percentages summing to exactly 100.0% by absorbing 0.1% rounding residue on primary
    diff = round(100.0 - sum(rounded.values()), 1)
    rounded[primary] = round(rounded[primary] + diff, 1)
    return rounded, primary


def _recommend_shaper(
    axis: str,
    structural_hz: float,
    belt_hz: float,
    motor_hz: float,
    bracket_hz: float,
    drive_hz: float,
    configured_accel: float,
    shaper_choice: str,
    scv_mm_s: float = 5.0,
    scanned_modes: list[tuple[float, float]] | None = None,
    psd_damping_ratio: float | None = None,
) -> tuple[InputShaperRecommendation, AxisShaperGraph]:
    # Effective coupled resonance of nozzle structure + drive (belt + motor + bracket in series)
    f_eff = 1.0 / math.sqrt(1.0 / max(structural_hz, 5.0) ** 2 + 1.0 / max(drive_hz, 5.0) ** 2)
    shares_pct, primary_bn = _compute_compliance_shares(structural_hz, belt_hz, motor_hz, bracket_hz)

    secondary_modes = [
        (mf, part_val)
        for mf, part_val in (scanned_modes or [])
        if abs(mf - structural_hz) >= 2.0 and abs(mf - f_eff) >= 2.0 and 10.0 <= mf <= 215.0 and part_val > 0.20
    ]

    graph = calibrate_axis_shapers(
        axis=axis,
        primary_freq_hz=f_eff,
        secondary_modes=secondary_modes,
        scv=scv_mm_s,
        shaper_choice=shaper_choice,
        psd_damping_ratio=psd_damping_ratio,
    )

    shaper = graph.recommended_shaper
    max_accel = graph.recommended_max_accel_mm_s2

    # Estimate residual ringing risk (0 = crisp, >1 = visible ghosting)
    ringing_ratio = configured_accel / max(max_accel, 100.0)
    rec = InputShaperRecommendation(
        axis=axis,
        effective_freq_hz=f_eff,
        structural_freq_hz=structural_hz,
        belt_freq_hz=belt_hz,
        motor_freq_hz=motor_hz,
        bracket_freq_hz=bracket_hz,
        drive_freq_hz=drive_hz,
        recommended_shaper=shaper,
        max_accel_mm_s2=max_accel,
        estimated_ringing_ratio=ringing_ratio,
        compliance_share_pct=shares_pct,
        primary_bottleneck=primary_bn,
    )
    return rec, graph


def solve_modal_fea(assembly: AssemblyModel, num_modes: int = 6) -> ModalResult:
    """Solve generalized structural eigenproblem K phi = omega^2 M phi."""
    K, M, _, node_idx, free_dofs = assemble_global_matrices(assembly)
    n_free = len(free_dofs)
    if n_free == 0:
        return ModalResult(fundamental_freq_hz=0.0)

    K_ff = 0.5 * (K[np.ix_(free_dofs, free_dofs)] + K[np.ix_(free_dofs, free_dofs)].T)
    M_ff = 0.5 * (M[np.ix_(free_dofs, free_dofs)] + M[np.ix_(free_dofs, free_dofs)].T)

    # Ensure strictly positive-definite mass diagonal floor (for zero-mass rotational DOFs if any)
    m_diag = np.diag(M_ff)
    m_floor = max(float(np.max(m_diag)) * 1e-12, 1e-12)
    M_ff = M_ff + np.diag(np.where(m_diag < m_floor, m_floor, 0.0))

    panel_indices = [i for i, n in enumerate(assembly.nodes) if n.role == "panel"]
    non_panel_indices = [i for i, n in enumerate(assembly.nodes) if n.role != "panel"]
    gantry_indices = [
        i
        for i, n in enumerate(assembly.nodes)
        if n.role in ("gantry", "toolhead", "nozzle", "cog")
    ]
    has_panels = len(panel_indices) > 0 and len(non_panel_indices) > 0

    k_ret = min(num_modes, n_free)
    target_scan = 56 if has_panels else max(num_modes, 16)
    k_scan = min(target_scan, n_free - 2 if n_free > 250 else n_free)

    used_sparse = False
    if n_free > 250 and 1 <= k_scan < n_free - 1:
        try:
            k_csc = csc_matrix(K_ff)
            m_csc = csc_matrix(M_ff)
            lu = splu(k_csc)
            op = LinearOperator(K_ff.shape, matvec=lu.solve, dtype=float)
            eigvals, eigvecs = eigsh(
                k_csc,
                k=k_scan,
                M=m_csc,
                sigma=0.0,
                which="LM",
                OPinv=op,
            )
            order = np.argsort(eigvals)
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]
            used_sparse = True
        except RuntimeError:
            used_sparse = False

    if not used_sparse:
        k_scan = min(target_scan, n_free)
        eigvals, eigvecs = eigh(K_ff, M_ff, subset_by_index=[0, k_scan - 1])

    th_idx = node_idx[assembly.toolhead_node_id]
    noz_id = (
        assembly.nozzle_node_id
        if assembly.nozzle_node_id in node_idx
        else assembly.toolhead_node_id
    )
    noz_idx = node_idx[noz_id]
    has_nozzle = noz_idx != th_idx
    bed_idx = node_idx.get(assembly.bed_node_id) if assembly.bed_node_id else None

    n_dof = K.shape[0]
    r_x = np.zeros(n_dof, dtype=float)
    r_y = np.zeros(n_dof, dtype=float)
    r_x[0::6] = 1.0
    r_y[1::6] = 1.0
    rx_f = r_x[free_dofs]
    ry_f = r_y[free_dofs]
    m_tot_x = max(float(rx_f @ M_ff @ rx_f), 1e-12)
    m_tot_y = max(float(ry_f @ M_ff @ ry_f), 1e-12)

    # Precompute panel-DOF mask in free_dofs space for fast modal kinetic energy ratio
    free_panel_mask = np.zeros(n_free, dtype=bool)
    if has_panels:
        panel_dof_set = {6 * pi + d for pi in panel_indices for d in range(6)}
        free_panel_mask = np.array([dof in panel_dof_set for dof in free_dofs], dtype=bool)

    modes: list[ModeShape] = []
    fallback_modes: list[ModeShape] = []
    scanned_x: list[tuple[float, float]] = []
    scanned_y: list[tuple[float, float]] = []
    best_x_freq: float | None = None
    best_y_freq: float | None = None
    nozzle_twist_freq: float | None = None
    panel_drum_freq: float | None = None
    panel_drum_fallback_freq: float | None = None

    for idx in range(len(eigvals)):
        lam = max(float(eigvals[idx]), 0.0)
        freq_hz = math.sqrt(lam) / (2.0 * math.pi)
        phi = eigvecs[:, idx]
        m_gen = max(float(phi @ M_ff @ phi), 1e-18)
        gamma_x = float(phi @ M_ff @ rx_f) / m_gen
        gamma_y = float(phi @ M_ff @ ry_f) / m_gen
        meff_ratio_x = math.sqrt(max((gamma_x * gamma_x * m_gen) / m_tot_x, 0.0))
        meff_ratio_y = math.sqrt(max((gamma_y * gamma_y * m_gen) / m_tot_y, 0.0))

        full_vec = np.zeros(n_dof, dtype=float)
        full_vec[free_dofs] = phi

        # Extract translational 3D displacements per node and normalize max translation to 1.0
        trans_raw = np.array([full_vec[6 * i : 6 * i + 3] for i in range(len(assembly.nodes))])
        max_amp = float(np.max(np.linalg.norm(trans_raw, axis=1)))
        trans = trans_raw / max_amp if max_amp > 1e-12 else trans_raw.copy()

        noz_rel_all = (
            np.abs(trans[noz_idx] - trans[bed_idx]) if bed_idx is not None else np.abs(trans[noz_idx])
        )
        noz_norm_all = float(np.linalg.norm(noz_rel_all))

        if has_panels:
            np_max = float(np.max(np.linalg.norm(trans[non_panel_indices], axis=1)))
            gantry_max = (
                float(np.max(np.linalg.norm(trans[gantry_indices], axis=1)))
                if gantry_indices
                else np_max
            )
            phi_p = np.where(free_panel_mask, phi, 0.0)
            ke_panel = float(phi_p @ M_ff @ phi_p) / m_gen
            if ke_panel > 0.50 and panel_drum_fallback_freq is None:
                panel_drum_fallback_freq = freq_hz
            is_isolated_panel_drum = (
                (ke_panel > 0.68 and gantry_max < 0.35 and noz_norm_all < 0.15)
                or (np_max < 0.25 and gantry_max < 0.20)
            )
            if is_isolated_panel_drum and panel_drum_freq is None:
                panel_drum_freq = freq_hz

            # Normalize structural amplitudes against frame+gantry+nozzle nodes so local panel
            # bulging never hides a true coupled frame/gantry resonance
            tr_struct = trans_raw[non_panel_indices]
            max_struct = max(float(np.max(np.linalg.norm(tr_struct, axis=1))), 1e-12)
            noz_rel = (
                np.abs(trans_raw[noz_idx] - trans_raw[bed_idx]) / max_struct
                if bed_idx is not None
                else np.abs(trans_raw[noz_idx]) / max_struct
            )
            th_rel = (
                np.abs(trans_raw[th_idx] - trans_raw[bed_idx]) / max_struct
                if bed_idx is not None
                else np.abs(trans_raw[th_idx]) / max_struct
            )
            bed_z_amp = abs(float(trans_raw[bed_idx][2] / max_struct)) if bed_idx is not None else 0.0
            axis_energy = np.sum(tr_struct**2, axis=0)
        else:
            is_isolated_panel_drum = False
            noz_rel = noz_rel_all
            th_rel = (
                np.abs(trans[th_idx] - trans[bed_idx]) if bed_idx is not None else np.abs(trans[th_idx])
            )
            bed_z_amp = abs(float(trans[bed_idx][2])) if bed_idx is not None else 0.0
            axis_energy = np.sum(trans**2, axis=0)

        noz_norm = float(np.linalg.norm(noz_rel))
        th_norm = float(np.linalg.norm(th_rel))
        part = (noz_rel / noz_norm).tolist() if noz_norm > 1e-9 else [0.0, 0.0, 0.0]

        dom_axis = ["X", "Y", "Z"][int(np.argmax(axis_energy))]
        is_bed_z_mode = (
            bed_idx is not None
            and bed_z_amp > 0.60
            and float(noz_rel[0]) < 0.20
            and float(noz_rel[1]) < 0.20
        )

        if not is_isolated_panel_drum:
            scanned_x.append((freq_hz, float(noz_rel[0]) * meff_ratio_x))
            scanned_y.append((freq_hz, float(noz_rel[1]) * meff_ratio_y))

            if (
                best_x_freq is None
                and not is_bed_z_mode
                and (
                    (
                        float(noz_rel[0]) > 0.20
                        and (dom_axis == "X" or (noz_norm > 0.25 and part[0] > 0.45))
                    )
                    or (
                        has_panels
                        and (
                            (dom_axis == "X" and meff_ratio_x > 0.25 and float(noz_rel[0]) > 0.05)
                            or (float(noz_rel[0]) > 0.25 and meff_ratio_x > 0.25)
                        )
                    )
                )
            ):
                best_x_freq = freq_hz
            if (
                best_y_freq is None
                and not is_bed_z_mode
                and (
                    (
                        float(noz_rel[1]) > 0.20
                        and (dom_axis == "Y" or (noz_norm > 0.25 and part[1] > 0.45))
                    )
                    or (
                        has_panels
                        and (
                            (dom_axis == "Y" and meff_ratio_y > 0.25 and float(noz_rel[1]) > 0.05)
                            or (float(noz_rel[1]) > 0.25 and meff_ratio_y > 0.25)
                        )
                    )
                )
            ):
                best_y_freq = freq_hz

            if (
                nozzle_twist_freq is None
                and has_nozzle
                and not is_bed_z_mode
                and noz_norm > 0.25
                and noz_norm > 1.25 * max(th_norm, 1e-6)
                and (part[0] > 0.50 or part[1] > 0.50)
            ):
                nozzle_twist_freq = freq_hz

            if len(modes) < k_ret:
                nodal_vectors = {
                    node.id: [round(float(v), 4) for v in trans[i]]
                    for i, node in enumerate(assembly.nodes)
                }
                modes.append(
                    ModeShape(
                        mode_number=len(modes) + 1,
                        frequency_hz=freq_hz,
                        dominant_axis=dom_axis,
                        toolhead_participation_xyz=part,
                        nodal_vectors=nodal_vectors,
                    )
                )
        elif len(fallback_modes) < k_ret:
            nodal_vectors = {
                node.id: [round(float(v), 4) for v in trans[i]]
                for i, node in enumerate(assembly.nodes)
            }
            fallback_modes.append(
                ModeShape(
                    mode_number=len(fallback_modes) + 1,
                    frequency_hz=freq_hz,
                    dominant_axis=dom_axis,
                    toolhead_participation_xyz=part,
                    nodal_vectors=nodal_vectors,
                )
            )

    if not modes:
        modes = fallback_modes
    fundamental = modes[0].frequency_hz if modes else 0.0
    resolved_panel_drum = (
        panel_drum_freq
        if panel_drum_freq is not None
        else (panel_drum_fallback_freq if panel_drum_fallback_freq is not None else 0.0)
    )
    struct_x = best_x_freq if best_x_freq is not None else fundamental
    struct_y = best_y_freq if best_y_freq is not None else fundamental
    if nozzle_twist_freq is None:
        nozzle_twist_freq = min(struct_x, struct_y)

    p0_x = max([px for mf, px in scanned_x if abs(mf - struct_x) < 2.0] or [1.0])
    p0_y = max([py for mf, py in scanned_y if abs(mf - struct_y) < 2.0] or [1.0])
    p0_x = max(p0_x, 0.05)
    p0_y = max(p0_y, 0.05)
    norm_scanned_x = [(mf, min(1.0, px / p0_x)) for mf, px in scanned_x]
    norm_scanned_y = [(mf, min(1.0, py / p0_y)) for mf, py in scanned_y]

    belt_x, motor_x, bracket_x, drive_x = _drive_axis_resonances_hz(assembly, "X")
    belt_y, motor_y, bracket_y, drive_y = _drive_axis_resonances_hz(assembly, "Y")

    scv_mm_s = assembly.spec.kinematics.square_corner_velocity_mm_s
    configured_accel = assembly.spec.kinematics.max_accel_mm_s2
    shaper_choice = assembly.spec.kinematics.input_shaper

    shaper_x, shaper_graph_x = _recommend_shaper(
        "X",
        struct_x,
        belt_x,
        motor_x,
        bracket_x,
        drive_x,
        configured_accel,
        shaper_choice,
        scv_mm_s=scv_mm_s,
        scanned_modes=norm_scanned_x,
        psd_damping_ratio=assembly.frame_damping_ratio,
    )
    shaper_y, shaper_graph_y = _recommend_shaper(
        "Y",
        struct_y,
        belt_y,
        motor_y,
        bracket_y,
        drive_y,
        configured_accel,
        shaper_choice,
        scv_mm_s=scv_mm_s,
        scanned_modes=norm_scanned_y,
        psd_damping_ratio=assembly.frame_damping_ratio,
    )

    limiting = shaper_y if shaper_y.max_accel_mm_s2 <= shaper_x.max_accel_mm_s2 else shaper_x
    primary_src = limiting.primary_bottleneck
    accel_bottleneck = {
        "limiting_axis": limiting.axis,
        "primary_source": primary_src,
        "primary_label": _BOTTLENECK_LABELS.get(primary_src, primary_src),
        "primary_share_pct": limiting.compliance_share_pct.get(primary_src, 0.0),
        "shares_pct": limiting.compliance_share_pct,
        "x_shares_pct": shaper_x.compliance_share_pct,
        "y_shares_pct": shaper_y.compliance_share_pct,
        "recommendation": _BOTTLENECK_TIPS.get(primary_src, ""),
    }

    return ModalResult(
        fundamental_freq_hz=fundamental,
        modes=modes,
        shaper_x=shaper_x,
        shaper_y=shaper_y,
        shaper_graph_x=shaper_graph_x,
        shaper_graph_y=shaper_graph_y,
        recommended_max_accel_mm_s2=min(shaper_x.max_accel_mm_s2, shaper_y.max_accel_mm_s2),
        nozzle_pitch_twist_freq_hz=nozzle_twist_freq,
        panel_drum_freq_hz=resolved_panel_drum,
        frame_damping_ratio=assembly.frame_damping_ratio,
        accel_bottleneck=accel_bottleneck,
    )
