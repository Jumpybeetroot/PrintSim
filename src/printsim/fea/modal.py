"""3D Modal Finite Element Analysis (K phi = omega^2 M phi) & Input Shaper Tuning."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eigh

from printsim.fea.beam3d import assemble_global_matrices
from printsim.fea.klipper_shaper import AxisShaperGraph, calibrate_axis_shapers
from printsim.geometry.builder import AssemblyModel
from printsim.kinematics.solvers import compute_drive_stiffness_n_per_m


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

    def to_dict(self) -> dict:
        return {
            "fundamental_freq_hz": round(self.fundamental_freq_hz, 2),
            "nozzle_pitch_twist_freq_hz": round(self.nozzle_pitch_twist_freq_hz, 2),
            "recommended_max_accel_mm_s2": round(self.recommended_max_accel_mm_s2, 0),
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
        "drive_freq_hz": round(s.drive_freq_hz, 2),
        "recommended_shaper": s.recommended_shaper,
        "max_accel_mm_s2": round(s.max_accel_mm_s2, 0),
        "estimated_ringing_ratio": round(s.estimated_ringing_ratio, 3),
    }


def _drive_axis_resonances_hz(assembly: AssemblyModel, axis: str) -> tuple[float, float, float]:
    """Compute (belt_freq_hz, motor_freq_hz, drive_freq_hz) for X or Y."""
    k_belt, k_motor_axis, k_drive = compute_drive_stiffness_n_per_m(assembly, axis)
    mass = max(assembly.x_moving_mass_kg if axis == "X" else assembly.y_moving_mass_kg, 0.05)
    c = 1.0 / (2.0 * math.pi)
    return (
        c * math.sqrt(k_belt / mass),
        c * math.sqrt(k_motor_axis / mass),
        c * math.sqrt(k_drive / mass),
    )


def _belt_axis_resonance_hz(assembly: AssemblyModel, axis: str) -> float:
    """Compute belt-driven axial spring-mass natural frequency f = (1/2pi) * sqrt(k_belt / m)."""
    return _drive_axis_resonances_hz(assembly, axis)[0]


def _recommend_shaper(
    axis: str,
    structural_hz: float,
    belt_hz: float,
    motor_hz: float,
    drive_hz: float,
    configured_accel: float,
    shaper_choice: str,
    scv_mm_s: float = 5.0,
    scanned_modes: list[tuple[float, float]] | None = None,
) -> tuple[InputShaperRecommendation, AxisShaperGraph]:
    # Effective coupled resonance of nozzle structure + drive (belt + motor in series)
    f_eff = 1.0 / math.sqrt(1.0 / max(structural_hz, 5.0) ** 2 + 1.0 / max(drive_hz, 5.0) ** 2)

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
        drive_freq_hz=drive_hz,
        recommended_shaper=shaper,
        max_accel_mm_s2=max_accel,
        estimated_ringing_ratio=ringing_ratio,
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

    k_ret = min(num_modes, n_free)
    k_scan = min(max(num_modes, 12), n_free)
    eigvals, eigvecs = eigh(K_ff, M_ff, subset_by_index=[0, k_scan - 1])

    th_idx = node_idx[assembly.toolhead_node_id]
    noz_id = (
        assembly.nozzle_node_id
        if assembly.nozzle_node_id in node_idx
        else assembly.toolhead_node_id
    )
    noz_idx = node_idx[noz_id]
    has_nozzle = noz_idx != th_idx

    n_dof = K.shape[0]
    r_x = np.zeros(n_dof, dtype=float)
    r_y = np.zeros(n_dof, dtype=float)
    r_x[0::6] = 1.0
    r_y[1::6] = 1.0
    rx_f = r_x[free_dofs]
    ry_f = r_y[free_dofs]
    m_tot_x = max(float(rx_f @ M_ff @ rx_f), 1e-12)
    m_tot_y = max(float(ry_f @ M_ff @ ry_f), 1e-12)

    modes: list[ModeShape] = []
    scanned_x: list[tuple[float, float]] = []
    scanned_y: list[tuple[float, float]] = []
    best_x_freq: float | None = None
    best_y_freq: float | None = None
    nozzle_twist_freq: float | None = None

    for idx in range(k_scan):
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
        trans = np.array([full_vec[6 * i : 6 * i + 3] for i in range(len(assembly.nodes))])
        max_amp = float(np.max(np.linalg.norm(trans, axis=1)))
        if max_amp > 1e-12:
            trans /= max_amp

        noz_disp = np.abs(trans[noz_idx])
        noz_norm = float(np.linalg.norm(noz_disp))
        th_norm = float(np.linalg.norm(trans[th_idx]))
        part = (noz_disp / noz_norm).tolist() if noz_norm > 1e-9 else [0.0, 0.0, 0.0]

        scanned_x.append((freq_hz, float(noz_disp[0]) * meff_ratio_x))
        scanned_y.append((freq_hz, float(noz_disp[1]) * meff_ratio_y))

        axis_energy = np.sum(trans**2, axis=0)
        dom_axis = ["X", "Y", "Z"][int(np.argmax(axis_energy))]

        if best_x_freq is None and (dom_axis == "X" or (noz_norm > 0.25 and part[0] > 0.45)):
            best_x_freq = freq_hz
        if best_y_freq is None and (dom_axis == "Y" or (noz_norm > 0.25 and part[1] > 0.45)):
            best_y_freq = freq_hz

        if (
            nozzle_twist_freq is None
            and has_nozzle
            and noz_norm > 0.25
            and noz_norm > 1.25 * max(th_norm, 1e-6)
            and (part[0] > 0.50 or part[1] > 0.50)
        ):
            nozzle_twist_freq = freq_hz

        if idx < k_ret:
            nodal_vectors = {
                node.id: [round(float(v), 4) for v in trans[i]]
                for i, node in enumerate(assembly.nodes)
            }
            modes.append(
                ModeShape(
                    mode_number=idx + 1,
                    frequency_hz=freq_hz,
                    dominant_axis=dom_axis,
                    toolhead_participation_xyz=part,
                    nodal_vectors=nodal_vectors,
                )
            )

    fundamental = modes[0].frequency_hz if modes else 0.0
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

    belt_x, motor_x, drive_x = _drive_axis_resonances_hz(assembly, "X")
    belt_y, motor_y, drive_y = _drive_axis_resonances_hz(assembly, "Y")

    scv_mm_s = assembly.spec.kinematics.square_corner_velocity_mm_s
    configured_accel = assembly.spec.kinematics.max_accel_mm_s2
    shaper_choice = assembly.spec.kinematics.input_shaper

    shaper_x, shaper_graph_x = _recommend_shaper(
        "X",
        struct_x,
        belt_x,
        motor_x,
        drive_x,
        configured_accel,
        shaper_choice,
        scv_mm_s=scv_mm_s,
        scanned_modes=norm_scanned_x,
    )
    shaper_y, shaper_graph_y = _recommend_shaper(
        "Y",
        struct_y,
        belt_y,
        motor_y,
        drive_y,
        configured_accel,
        shaper_choice,
        scv_mm_s=scv_mm_s,
        scanned_modes=norm_scanned_y,
    )

    return ModalResult(
        fundamental_freq_hz=fundamental,
        modes=modes,
        shaper_x=shaper_x,
        shaper_y=shaper_y,
        shaper_graph_x=shaper_graph_x,
        shaper_graph_y=shaper_graph_y,
        recommended_max_accel_mm_s2=min(shaper_x.max_accel_mm_s2, shaper_y.max_accel_mm_s2),
        nozzle_pitch_twist_freq_hz=nozzle_twist_freq,
    )
