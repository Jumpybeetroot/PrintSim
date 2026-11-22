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
    "frame": "Main limit: outer frame racking, rail-beam bending, and toolhead-mount structural stiffness along the load path.",
    "belt": "Main limit: belt tensile axial stiffness and free-span drive-path compliance.",
    "motor": "Main limit: stepper electromagnetic stiffness (holding torque headroom and winding L/R back-EMF margin) relative to moving inertia.",
    "bracket": "Main limit: XY-joint / motor-mount bracket compliance and cantilevered pulley shaft bending.",
}


@dataclass
class ModeShape:
    mode_number: int
    frequency_hz: float
    dominant_axis: str
    toolhead_participation_xyz: list[float]
    nodal_vectors: dict[str, list[float]]  # normalized 3D displacement per node for animation
    excited_axis: str = "XY"
    mode_label: str = ""
    what_is_shaking: str = ""
    how_to_fix: str = ""
    severity: str = "moderate"
    subsystem_share_pct: dict[str, float] = field(
        default_factory=lambda: {"toolhead": 0.0, "gantry": 0.0, "frame": 0.0, "bed": 0.0}
    )
    nozzle_whip_ratio: float = 1.0
    toolhead_flop_pct: float = 0.0
    relative_nozzle_bed_amp: float = 0.0
    print_impact_score: float = 0.0
    print_impact_level: str = "low"
    dominant_problem_area: str = "frame"


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
    worst_print_mode_number: int = 1

    def to_dict(self) -> dict:
        return {
            "fundamental_freq_hz": round(self.fundamental_freq_hz, 2),
            "nozzle_pitch_twist_freq_hz": round(self.nozzle_pitch_twist_freq_hz, 2),
            "panel_drum_freq_hz": round(self.panel_drum_freq_hz, 2),
            "frame_damping_ratio": round(self.frame_damping_ratio, 4),
            "recommended_max_accel_mm_s2": round(self.recommended_max_accel_mm_s2, 0),
            "accel_bottleneck": self.accel_bottleneck,
            "worst_print_mode_number": self.worst_print_mode_number,
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
                    "excited_axis": m.excited_axis,
                    "toolhead_participation_xyz": [round(v, 3) for v in m.toolhead_participation_xyz],
                    "nodal_vectors": m.nodal_vectors,
                    "mode_label": m.mode_label,
                    "what_is_shaking": m.what_is_shaking,
                    "how_to_fix": m.how_to_fix,
                    "severity": m.severity,
                    "subsystem_share_pct": m.subsystem_share_pct,
                    "nozzle_whip_ratio": m.nozzle_whip_ratio,
                    "toolhead_flop_pct": m.toolhead_flop_pct,
                    "relative_nozzle_bed_amp": m.relative_nozzle_bed_amp,
                    "print_impact_score": m.print_impact_score,
                    "print_impact_level": m.print_impact_level,
                    "dominant_problem_area": m.dominant_problem_area,
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
        if abs(mf - structural_hz) >= 3.0 and abs(mf - f_eff) >= 3.0 and 10.0 <= mf <= 215.0 and part_val > 0.15
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


def _mode_severity(freq_hz: float) -> str:
    if freq_hz < 65.0:
        return "critical"
    if freq_hz < 115.0:
        return "moderate"
    return "stiff"


def _frame_mode_fix_hint(assembly: AssemblyModel) -> str:
    spec = assembly.spec
    if spec.architecture == "corexy":
        if not spec.frame.enclosed_panels or spec.frame.panel_mounting != "bolted_structural":
            return (
                "Main limit: outer frame shear & racking stiffness "
                "(open or non-structural enclosure panels leave frame columns uncoupled in shear)."
            )
        if (
            spec.frame.panel_material not in ("aluminum_5052_sheet", "carbon_fiber_plate")
            or spec.frame.panel_thickness_mm < 4.0
        ):
            return (
                "Main limit: outer frame shear-panel in-plane stiffness, fastener shear stiffness, "
                "and perimeter racking rigidity."
            )
        return "Main limit: outer frame extrusion bending/torsional stiffness and corner-joint rigidity."
    if spec.architecture == "cartesian_bedslinger":
        return "Main limit: Z-portal lateral/fore-aft bending stiffness and diagonal portal racking rigidity."
    return "Main limit: vertical tower extrusion bending/torsional stiffness and perimeter triangulation rigidity."


def _toolhead_mode_fix_hint(assembly: AssemblyModel) -> str:
    spec = assembly.spec
    if spec.architecture == "delta":
        return (
            f"Main limit: effector mount compliance ('{spec.toolhead.mount_material}'), "
            "nozzle/CoG lever arm overhang, and effector moving mass."
        )
    return (
        f"Main limit: toolhead mount stiffness ('{spec.toolhead.mount_material}'), "
        "rail carriage pitch/torsional compliance, CoG/nozzle lever arm, and toolhead mass."
    )


def _gantry_mode_fix_hint(assembly: AssemblyModel) -> str:
    spec = assembly.spec
    if spec.architecture == "delta":
        return "Main limit: delta tower bending stiffness, perimeter bracing rigidity, and diagonal arm axial stiffness."
    return (
        "Main limit: X-gantry beam & rail bending/torsional stiffness, XY-joint bracket compliance, "
        "and moving gantry inertia."
    )


def _bed_mode_fix_hint(assembly: AssemblyModel) -> str:
    spec = assembly.spec
    if spec.architecture == "cartesian_bedslinger":
        return "Main limit: moving Y-bed carriage stiffness, Y-axis belt stiffness, and heated bed inertia."
    return "Main limit: Z-drive / bed support axial/cantilever stiffness and bed plate flexural rigidity."


def _classify_mode_diagnosis(
    assembly: AssemblyModel,
    freq_hz: float,
    dom_axis: str,
    excited_axis: str,
    frame_ke_share: float,
    gantry_ke_share: float,
    th_ke_share: float,
    bed_ke_share: float,
    frame_amp: float,
    gantry_amp: float,
    noz_amp: float,
    th_amp: float,
    is_torsional: bool,
) -> tuple[str, str, str, str]:
    """Return ``(mode_label, what_is_shaking, how_to_fix, severity)`` for a structural mode."""
    spec = assembly.spec
    severity = _mode_severity(freq_hz)
    axis_use = excited_axis if excited_axis in ("X", "Y") else (dom_axis if dom_axis in ("X", "Y") else "XY")

    if is_torsional and frame_amp > 0.40 and frame_ke_share > 0.25 and noz_amp < 0.75:
        return (
            "Frame Torsional Twist",
            "Opposite sides of the outer frame twist out of phase in torsion during rapid gantry cornering.",
            _frame_mode_fix_hint(assembly),
            severity,
        )

    if frame_ke_share > 0.42 or (
        frame_amp > 0.75 and noz_amp < 0.45 and bed_ke_share < 0.50 and gantry_ke_share < 0.35
    ):
        frame_desc = "Z-axis portal frame" if spec.architecture == "cartesian_bedslinger" else "outer extrusion frame"
        if axis_use == "Y":
            label = "Whole-Frame Y Sway"
            shaking = (
                f"The {frame_desc} rocks sideways as a parallelogram when the gantry accelerates in Y."
            )
        elif axis_use == "X":
            label = "Whole-Frame X Racking"
            shaking = (
                f"The {frame_desc} shears and racks side-to-side along X under rapid toolhead acceleration."
            )
        else:
            label = "Frame Torsional Twist"
            shaking = f"The {frame_desc} flexes diagonally across X and Y during rapid cornering moves."
        return label, shaking, _frame_mode_fix_hint(assembly), severity

    if (noz_amp > 1.25 * max(th_amp, 1e-6) and (th_ke_share > 0.35 or noz_amp > 0.50)) or th_ke_share > 0.55:
        anchor = "effector" if spec.architecture == "delta" else "linear carriage"
        return (
            "Toolhead & Nozzle Pitch Whip",
            f"The nozzle tip whips relative to the {anchor} due to toolhead CoG overhang and mount compliance.",
            _toolhead_mode_fix_hint(assembly),
            severity,
        )

    if bed_ke_share >= 0.45 and bed_ke_share > (gantry_ke_share + th_ke_share):
        if spec.architecture == "cartesian_bedslinger":
            shaking = (
                "The moving heated bed assembly flexes on its Y-belt and carriage supports during rapid Y acceleration."
            )
        else:
            shaking = "The heated bed assembly flexes relative to the nozzle and frame during rapid acceleration."
        return "Bed Support Vibration", shaking, _bed_mode_fix_hint(assembly), severity

    if gantry_ke_share >= 0.25 or gantry_amp >= 0.65:
        if spec.architecture == "delta":
            return (
                "Delta Tower & Arm Resonance",
                "The diagonal carbon arms and tower carriages vibrate against the vertical towers during rapid effector moves.",
                _gantry_mode_fix_hint(assembly),
                severity,
            )
        if axis_use == "Y":
            return (
                "Moving X-Gantry Flex",
                "The moving X-gantry beam bends and twists along Y under rapid toolhead acceleration.",
                _gantry_mode_fix_hint(assembly),
                severity,
            )
        return (
            "Gantry & Y-Rail Flex",
            "The moving X-gantry and Y-axis rail beams flex laterally under rapid acceleration.",
            _gantry_mode_fix_hint(assembly),
            severity,
        )

    if frame_ke_share >= max(gantry_ke_share, th_ke_share, bed_ke_share):
        fix_hint = _frame_mode_fix_hint(assembly)
    elif th_ke_share >= max(gantry_ke_share, bed_ke_share):
        fix_hint = _toolhead_mode_fix_hint(assembly)
    else:
        fix_hint = _gantry_mode_fix_hint(assembly)

    return (
        "Coupled Frame & Gantry Resonance",
        f"Coupled vibration of the frame, gantry, and toolhead along {excited_axis} during rapid print moves.",
        fix_hint,
        severity,
    )


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
        try:
            # Invert to M_ff phi = mu K_ff phi so eigh Cholesky-factorizes well-conditioned SPD K_ff
            mu, eigvecs = eigh(
                M_ff,
                K_ff,
                subset_by_index=[n_free - k_scan, n_free - 1],
            )
            mu = mu[::-1]
            eigvecs = eigvecs[:, ::-1].copy()
            eigvals = 1.0 / np.maximum(mu, 1e-30)
        except np.linalg.LinAlgError:
            # Clean fallback if K_ff is singular/semidefinite (e.g. unconstrained mechanism)
            eigvals, eigvecs = eigh(K_ff, M_ff, subset_by_index=[0, k_scan - 1])

    eigvals = np.maximum(eigvals, 0.0)
    for j in range(eigvecs.shape[1]):
        v = eigvecs[:, j]
        m_gen_j = float(v @ M_ff @ v)
        if m_gen_j > 1e-24:
            eigvecs[:, j] = v / math.sqrt(m_gen_j)

    th_idx = node_idx[assembly.toolhead_node_id]
    cog_id = (
        assembly.toolhead_cog_node_id
        if assembly.toolhead_cog_node_id in node_idx
        else assembly.toolhead_node_id
    )
    cog_idx = node_idx[cog_id]
    noz_id = (
        assembly.nozzle_node_id
        if assembly.nozzle_node_id in node_idx
        else assembly.toolhead_node_id
    )
    noz_idx = node_idx[noz_id]
    has_nozzle = noz_idx != th_idx
    bed_idx = node_idx.get(assembly.bed_node_id) if assembly.bed_node_id else None

    n_dof = K.shape[0]
    # Exact unit inertial excitation force patterns for X and Y acceleration (used for modal receptance)
    fx_exc = np.zeros(n_dof, dtype=float)
    fx_exc[6 * cog_idx + 0] = 1.0

    fy_exc = np.zeros(n_dof, dtype=float)
    if assembly.spec.architecture == "cartesian_bedslinger":
        if bed_idx is not None:
            fy_exc[6 * bed_idx + 1] = -1.0
        else:
            fy_exc[6 * cog_idx + 1] = 1.0
    else:
        th_share = assembly.spec.toolhead.total_mass_g * 1e-3 / max(assembly.y_moving_mass_kg, 1e-6)
        fy_exc[6 * cog_idx + 1] = th_share
        other_gantry = [
            gid
            for gid in assembly.gantry_node_ids
            if gid != assembly.toolhead_node_id and gid in node_idx
        ]
        if other_gantry:
            rem = (1.0 - th_share) / len(other_gantry)
            for gid in other_gantry:
                fy_exc[6 * node_idx[gid] + 1] += rem
        else:
            fy_exc[6 * cog_idx + 1] = 1.0

    # Precompute role-DOF masks in free_dofs space for fast modal kinetic energy partitioning
    free_panel_mask = np.zeros(n_free, dtype=bool)
    if has_panels:
        panel_dof_set = {6 * pi + d for pi in panel_indices for d in range(6)}
        free_panel_mask = np.array([dof in panel_dof_set for dof in free_dofs], dtype=bool)

    frame_node_indices = [
        i
        for i, n in enumerate(assembly.nodes)
        if n.role in ("frame", "mount", "foot", "tower", "brace")
    ]
    active_frame_indices = [
        i
        for i, n in enumerate(assembly.nodes)
        if n.role in ("frame", "mount", "tower", "foot") and not n.fixed
    ]
    gantry_only_indices = [i for i, n in enumerate(assembly.nodes) if n.role == "gantry"]
    th_all_indices = [
        i for i, n in enumerate(assembly.nodes) if n.role in ("toolhead", "cog", "nozzle")
    ]
    bed_all_indices = [i for i, n in enumerate(assembly.nodes) if n.role == "bed"]

    def _free_mask(node_list: list[int]) -> np.ndarray:
        if not node_list:
            return np.zeros(n_free, dtype=bool)
        dset = {6 * ni + d for ni in node_list for d in range(6)}
        return np.array([dof in dset for dof in free_dofs], dtype=bool)

    free_frame_mask = _free_mask(frame_node_indices)
    free_gantry_mask = _free_mask(gantry_only_indices)
    free_th_mask = _free_mask(th_all_indices)
    free_bed_mask = _free_mask(bed_all_indices)

    candidates: list[dict] = []
    receptance_x: list[tuple[float, float]] = []
    receptance_y: list[tuple[float, float]] = []
    car_mode_entries: list[tuple[float, float, float, float]] = []
    sum_comp_x_car = 0.0
    sum_comp_y_car = 0.0
    sum_comp_x_noz = 0.0
    sum_comp_y_noz = 0.0
    panel_drum_freq: float | None = None
    panel_drum_fallback_freq: float | None = None

    for idx in range(len(eigvals)):
        lam = max(float(eigvals[idx]), 0.0)
        freq_hz = math.sqrt(lam) / (2.0 * math.pi)
        phi = eigvecs[:, idx]
        m_phi = M_ff @ phi
        m_gen = max(float(phi @ m_phi), 1e-18)

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

        ke_panel = 0.0
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
                (ke_panel > 0.55 and noz_norm_all < 0.12)
                or (ke_panel > 0.58 and noz_norm_all < 0.25)
                or (ke_panel > 0.68 and gantry_max < 0.35 and noz_norm_all < 0.28)
                or (np_max < 0.25 and gantry_max < 0.20)
            )
            if is_isolated_panel_drum and panel_drum_freq is None:
                panel_drum_freq = freq_hz

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
            tr_norm = trans_raw / max_struct
        else:
            is_isolated_panel_drum = False
            noz_rel = noz_rel_all
            th_rel = (
                np.abs(trans[th_idx] - trans[bed_idx]) if bed_idx is not None else np.abs(trans[th_idx])
            )
            bed_z_amp = abs(float(trans[bed_idx][2])) if bed_idx is not None else 0.0
            axis_energy = np.sum(trans**2, axis=0)
            tr_norm = trans

        ke_struct = max(1.0 - max(0.0, ke_panel), 1e-6)
        frame_ke_share = max(0.0, float(phi[free_frame_mask] @ m_phi[free_frame_mask]) / m_gen) / ke_struct
        gantry_ke_share = max(0.0, float(phi[free_gantry_mask] @ m_phi[free_gantry_mask]) / m_gen) / ke_struct
        th_ke_share = max(0.0, float(phi[free_th_mask] @ m_phi[free_th_mask]) / m_gen) / ke_struct
        bed_ke_share = max(0.0, float(phi[free_bed_mask] @ m_phi[free_bed_mask]) / m_gen) / ke_struct

        frame_amp = (
            float(np.max(np.linalg.norm(tr_norm[active_frame_indices], axis=1)))
            if active_frame_indices
            else 0.0
        )
        gantry_amp = (
            float(np.max(np.linalg.norm(tr_norm[gantry_only_indices], axis=1)))
            if gantry_only_indices
            else 0.0
        )

        is_torsional = False
        if all(cid in node_idx for cid in ("t_fl", "t_fr", "t_rl", "t_rr")) and frame_amp > 1e-6:
            v_fl = tr_norm[node_idx["t_fl"]]
            v_fr = tr_norm[node_idx["t_fr"]]
            v_rl = tr_norm[node_idx["t_rl"]]
            opp_y = v_fl[1] * v_fr[1] < -0.02 and min(abs(v_fl[1]), abs(v_fr[1])) > 0.25 * frame_amp
            opp_x = v_fl[0] * v_rl[0] < -0.02 and min(abs(v_fl[0]), abs(v_rl[0])) > 0.25 * frame_amp
            is_torsional = bool(opp_y or opp_x)
        elif all(cid in node_idx for cid in ("p_tl", "p_tr")) and frame_amp > 1e-6:
            v_pl = tr_norm[node_idx["p_tl"]]
            v_pr = tr_norm[node_idx["p_tr"]]
            is_torsional = bool(
                v_pl[1] * v_pr[1] < -0.02 and min(abs(v_pl[1]), abs(v_pr[1])) > 0.25 * frame_amp
            )

        noz_norm = float(np.linalg.norm(noz_rel))
        th_norm = float(np.linalg.norm(th_rel))
        noz_th_diff_norm = (
            float(np.linalg.norm(tr_norm[noz_idx] - tr_norm[th_idx])) if has_nozzle else 0.0
        )
        part = (noz_rel / noz_norm).tolist() if noz_norm > 1e-9 else [0.0, 0.0, 0.0]

        dom_axis = ["X", "Y", "Z"][int(np.argmax(axis_energy))]
        is_bed_z_mode = (
            bed_idx is not None
            and bed_z_amp > 0.60
            and float(noz_rel[0]) < 0.20
            and float(noz_rel[1]) < 0.20
        )

        # Exact modal compliance / receptance residue at the nozzle-to-bed and carriage-to-bed interfaces:
        # R_{k,a} = |phi_{k,rel,a} * (phi_k^T F_a)| / (m_{gen,k} * omega_k^2)
        rec_x = 0.0
        rec_y = 0.0
        if lam > 1.0:
            pf_x = float(full_vec @ fx_exc)
            pf_y = float(full_vec @ fy_exc)
            rel_x_raw = (
                float(full_vec[6 * noz_idx + 0] - full_vec[6 * bed_idx + 0])
                if bed_idx is not None
                else float(full_vec[6 * noz_idx + 0])
            )
            rel_y_raw = (
                float(full_vec[6 * noz_idx + 1] - full_vec[6 * bed_idx + 1])
                if bed_idx is not None
                else float(full_vec[6 * noz_idx + 1])
            )
            rel_x_car = (
                float(full_vec[6 * th_idx + 0] - full_vec[6 * bed_idx + 0])
                if bed_idx is not None
                else float(full_vec[6 * th_idx + 0])
            )
            rel_y_car = (
                float(full_vec[6 * th_idx + 1] - full_vec[6 * bed_idx + 1])
                if bed_idx is not None
                else float(full_vec[6 * th_idx + 1])
            )
            comp_x_noz = (rel_x_raw * pf_x) / (m_gen * lam)
            comp_y_noz = (rel_y_raw * pf_y) / (m_gen * lam)
            comp_x_car = (rel_x_car * pf_x) / (m_gen * lam)
            comp_y_car = (rel_y_car * pf_y) / (m_gen * lam)

            rec_x = abs(comp_x_noz)
            rec_y = abs(comp_y_noz)
            if not is_isolated_panel_drum:
                receptance_x.append((freq_hz, rec_x))
                receptance_y.append((freq_hz, rec_y))
                car_mode_entries.append((freq_hz, comp_x_car, comp_y_car, max(0.0, ke_panel)))

            sum_comp_x_car += comp_x_car
            sum_comp_y_car += comp_y_car
            sum_comp_x_noz += comp_x_noz
            sum_comp_y_noz += comp_y_noz

        candidates.append(
            {
                "freq_hz": freq_hz,
                "dom_axis": dom_axis,
                "part": part,
                "trans": trans,
                "noz_rel": noz_rel,
                "noz_norm": noz_norm,
                "th_norm": th_norm,
                "noz_th_diff_norm": noz_th_diff_norm,
                "is_isolated_panel_drum": is_isolated_panel_drum,
                "is_bed_z_mode": is_bed_z_mode,
                "rec_x": rec_x,
                "rec_y": rec_y,
                "frame_ke_share": frame_ke_share,
                "gantry_ke_share": gantry_ke_share,
                "th_ke_share": th_ke_share,
                "bed_ke_share": bed_ke_share,
                "frame_amp": frame_amp,
                "gantry_amp": gantry_amp,
                "is_torsional": is_torsional,
            }
        )

    rx_max = max((c["rec_x"] for c in candidates), default=0.0)
    ry_max = max((c["rec_y"] for c in candidates), default=0.0)

    def _build_mode_shape(
        c: dict,
        mode_num: int,
        dom_axis: str,
        excited_axis: str,
        noz_norm: float,
        th_norm: float,
        part: list[float],
    ) -> tuple[ModeShape, float]:
        freq_hz = float(c["freq_hz"])
        frame_ke = max(0.0, float(c["frame_ke_share"]))
        gantry_ke = max(0.0, float(c["gantry_ke_share"]))
        th_ke = max(0.0, float(c["th_ke_share"]))
        bed_ke = max(0.0, float(c["bed_ke_share"]))

        raw_sub = {
            "toolhead": th_ke,
            "gantry": gantry_ke,
            "frame": frame_ke,
            "bed": bed_ke,
        }
        sub_tot = sum(raw_sub.values())
        if sub_tot <= 1e-12:
            raw_sub = {"toolhead": 0.0, "gantry": 0.0, "frame": 1.0, "bed": 0.0}
            sub_tot = 1.0
        sub_pct = {k: round(100.0 * v / sub_tot, 1) for k, v in raw_sub.items()}
        dom_sub = max(raw_sub, key=lambda k: raw_sub[k])
        diff_sub = round(100.0 - sum(sub_pct.values()), 1)
        sub_pct[dom_sub] = round(sub_pct[dom_sub] + diff_sub, 1)

        if not has_nozzle:
            whip_ratio = 1.0
            flop_pct = 0.0
        else:
            whip_ratio = round(min(10.0, max(0.0, noz_norm / max(th_norm, 1e-6))), 2)
            if noz_norm < 1e-6:
                flop_pct = 0.0
            else:
                flop_pct = round(
                    min(100.0, 100.0 * float(c["noz_th_diff_norm"]) / max(noz_norm, 1e-9)),
                    1,
                )

        rel_noz_bed = round(noz_norm, 3)

        trans = c["trans"]
        nodal_vectors = {
            node.id: [round(float(v), 4) for v in trans[i]]
            for i, node in enumerate(assembly.nodes)
        }
        m_lbl, m_shake, m_fix, m_sev = _classify_mode_diagnosis(
            assembly=assembly,
            freq_hz=freq_hz,
            dom_axis=dom_axis,
            excited_axis=excited_axis,
            frame_ke_share=frame_ke,
            gantry_ke_share=gantry_ke,
            th_ke_share=th_ke,
            bed_ke_share=bed_ke,
            frame_amp=float(c["frame_amp"]),
            gantry_amp=float(c["gantry_amp"]),
            noz_amp=noz_norm,
            th_amp=th_norm,
            is_torsional=bool(c["is_torsional"]),
        )

        if (
            flop_pct >= 35.0
            or (whip_ratio >= 1.30 and noz_norm >= 0.35)
            or th_ke > max(gantry_ke, frame_ke, bed_ke)
        ):
            dom_area = "toolhead"
        elif m_lbl in ("Whole-Frame Y Sway", "Whole-Frame X Racking", "Frame Torsional Twist"):
            dom_area = "frame"
        elif m_lbl == "Bed Support Vibration":
            dom_area = "bed"
        elif m_lbl in ("Moving X-Gantry Flex", "Gantry & Y-Rail Flex", "Delta Tower & Arm Resonance"):
            dom_area = "gantry"
        else:
            dom_area = max(
                (("gantry", gantry_ke), ("frame", frame_ke), ("bed", bed_ke)),
                key=lambda kv: kv[1],
            )[0]

        c_k = (float(c["rec_x"]) + float(c["rec_y"])) / max(freq_hz**2, 1.0)
        ms = ModeShape(
            mode_number=mode_num,
            frequency_hz=freq_hz,
            dominant_axis=dom_axis,
            toolhead_participation_xyz=part,
            nodal_vectors=nodal_vectors,
            excited_axis=excited_axis,
            mode_label=m_lbl,
            what_is_shaking=m_shake,
            how_to_fix=m_fix,
            severity=m_sev,
            subsystem_share_pct=sub_pct,
            nozzle_whip_ratio=whip_ratio,
            toolhead_flop_pct=flop_pct,
            relative_nozzle_bed_amp=rel_noz_bed,
            dominant_problem_area=dom_area,
        )
        return ms, c_k

    modes: list[ModeShape] = []
    mode_cks: list[float] = []
    fallback_modes: list[ModeShape] = []
    fallback_cks: list[float] = []
    nozzle_twist_freq: float | None = None

    for c in candidates:
        dom_axis = c["dom_axis"]
        noz_rel = c["noz_rel"]
        rec_x = float(c["rec_x"])
        rec_y = float(c["rec_y"])
        part = c["part"]
        freq_hz = float(c["freq_hz"])
        noz_norm = float(c["noz_norm"])
        th_norm = float(c["th_norm"])

        r_tot = rec_x + rec_y
        if r_tot <= 1e-18:
            excited_axis = dom_axis if dom_axis in ("X", "Y") else "XY"
        elif rec_x >= 3.0 * rec_y:
            excited_axis = "X"
        elif rec_y >= 3.0 * rec_x:
            excited_axis = "Y"
        else:
            excited_axis = "XY"

        is_pure_vertical_z = dom_axis == "Z" and (
            bool(c["is_bed_z_mode"])
            or float(max(noz_rel[0], noz_rel[1])) < 0.15
            or (
                rx_max > 1e-18
                and ry_max > 1e-18
                and rec_x < 0.03 * rx_max
                and rec_y < 0.03 * ry_max
            )
        )

        if not c["is_isolated_panel_drum"] and not is_pure_vertical_z:
            if (
                nozzle_twist_freq is None
                and has_nozzle
                and not c["is_bed_z_mode"]
                and noz_norm > 0.25
                and noz_norm > 1.25 * max(th_norm, 1e-6)
                and (part[0] > 0.50 or part[1] > 0.50)
            ):
                nozzle_twist_freq = freq_hz

            if len(modes) < k_ret:
                ms, c_k = _build_mode_shape(
                    c, len(modes) + 1, dom_axis, excited_axis, noz_norm, th_norm, part
                )
                modes.append(ms)
                mode_cks.append(c_k)
        elif len(fallback_modes) < k_ret:
            ms, c_k = _build_mode_shape(
                c, len(fallback_modes) + 1, dom_axis, excited_axis, noz_norm, th_norm, part
            )
            fallback_modes.append(ms)
            fallback_cks.append(c_k)

    if not modes:
        modes = fallback_modes
        mode_cks = fallback_cks

    c_max = max(mode_cks, default=0.0)
    for m, c_k in zip(modes, mode_cks):
        if c_max <= 1e-30 or m.relative_nozzle_bed_amp < 1e-6:
            score = 0.0
        else:
            c_norm = max(0.0, c_k / c_max)
            amp_factor = min(1.0, max(0.0, m.relative_nozzle_bed_amp) / 0.30)
            freq_urgency = min(1.0, (140.0 / max(m.frequency_hz, 20.0)) ** 0.6)
            whip_boost = (
                1.0 + 0.15 * min(1.0, max(0.0, (m.nozzle_whip_ratio - 1.35) / 1.5))
                if m.relative_nozzle_bed_amp >= 0.25
                else 1.0
            )
            raw_score = 100.0 * (c_norm**0.5) * (0.15 + 0.85 * amp_factor) * freq_urgency * whip_boost
            score = round(min(100.0, max(0.0, raw_score)), 1)
        m.print_impact_score = score
        if score >= 75.0:
            m.print_impact_level = "critical"
        elif score >= 45.0:
            m.print_impact_level = "high"
        elif score >= 20.0:
            m.print_impact_level = "moderate"
        else:
            m.print_impact_level = "low"

    worst_print_mode_number = (
        max(modes, key=lambda m: (m.print_impact_score, -m.mode_number)).mode_number
        if modes
        else 1
    )
    fundamental = modes[0].frequency_hz if modes else 0.0
    resolved_panel_drum = (
        panel_drum_freq
        if panel_drum_freq is not None
        else (panel_drum_fallback_freq if panel_drum_fallback_freq is not None else 0.0)
    )

    def _receptance_structural_freq_and_psd(
        rmodes: list[tuple[float, float]], fallback_hz: float
    ) -> tuple[float, list[tuple[float, float]]]:
        """Compute acceleration-weighted structural frequency & secondary PSD modes.

        Under Klipper input shaping, the max acceleration penalty of a resonant mode at
        frequency f_k with nozzle-to-bed modal compliance R_k scales as w_k = R_k / f_k^2.
        Weighting the compliance-domain frequency by w_k smoothly tracks both low-frequency
        frame sway (open frames) and high-frequency local gantry modes (stiff paneled frames)
        without discontinuous threshold cliffs across panel thicknesses.
        """
        weighted = [(f, r, r / (f * f)) for f, r in rmodes if f >= 5.0 and r > 0.0]
        if not weighted:
            return fallback_hz, []
        wmax = max(w for _, _, w in weighted)
        if wmax <= 1e-24:
            return fallback_hz, []
        sig = [(f, r, w) for f, r, w in weighted if w >= 0.01 * wmax]
        if not sig:
            return fallback_hz, []
        w_tot = sum(w for _, _, w in sig)
        inv_f2 = sum(w / (f * f) for f, _, w in sig) / max(w_tot, 1e-24)
        eff_hz = 1.0 / math.sqrt(max(inv_f2, 1e-12))

        # Exclude the primary dominant mode from secondary_modes so it is not double-counted in the PSD
        dom_f = max(sig, key=lambda item: item[2])[0]
        rmax = max(r for _, r, _ in sig)
        sec_modes = [
            (f, min(1.0, math.sqrt(r / max(rmax, 1e-18))))
            for f, r, _ in sig
            if abs(f - dom_f) >= 3.0 and abs(f - eff_hz) >= 3.0
        ]
        return eff_hz, sec_modes

    # Filter out minor negative-residue panel/bed modes from carriage receptance while preserving major bed modes
    max_pos_car_x = max((ccx for _, ccx, _, _ in car_mode_entries if ccx > 0.0), default=0.0)
    max_pos_car_y = max((ccy for _, _, ccy, _ in car_mode_entries if ccy > 0.0), default=0.0)
    receptance_x_car = [
        (f, abs(ccx))
        for f, ccx, _, kp in car_mode_entries
        if ccx > 0.0 or (kp < 0.50 and abs(ccx) > 0.5 * max_pos_car_x)
    ] or [(f, abs(ccx)) for f, ccx, _, _ in car_mode_entries]
    receptance_y_car = [
        (f, abs(ccy))
        for f, _, ccy, kp in car_mode_entries
        if ccy > 0.0 or (kp < 0.50 and abs(ccy) > 0.5 * max_pos_car_y)
    ] or [(f, abs(ccy)) for f, _, ccy, _ in car_mode_entries]

    struct_x_car, _ = _receptance_structural_freq_and_psd(receptance_x_car, fundamental)
    struct_y_car, _ = _receptance_structural_freq_and_psd(receptance_y_car, fundamental)
    _, norm_scanned_x = _receptance_structural_freq_and_psd(receptance_x, fundamental)
    _, norm_scanned_y = _receptance_structural_freq_and_psd(receptance_y, fundamental)

    # Exact signed Rayleigh nozzle-to-carriage compliance amplification ratio
    ratio_x = max(1.0, abs(sum_comp_x_noz) / max(abs(sum_comp_x_car), 1e-18)) if has_nozzle else 1.0
    ratio_y = max(1.0, abs(sum_comp_y_noz) / max(abs(sum_comp_y_car), 1e-18)) if has_nozzle else 1.0
    struct_x = struct_x_car / math.sqrt(ratio_x)
    struct_y = struct_y_car / math.sqrt(ratio_y)
    if nozzle_twist_freq is None:
        nozzle_twist_freq = min(struct_x, struct_y)

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
        worst_print_mode_number=worst_print_mode_number,
    )
