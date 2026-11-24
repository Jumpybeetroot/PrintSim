"""3D Modal Finite Element Analysis (K phi = omega^2 M phi) & Input Shaper Tuning."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import LinearOperator, eigsh, splu

from printsim.fea.beam3d import (
    assemble_global_matrices,
    beam_rotation_matrix,
    local_beam_mass,
    local_beam_stiffness,
    transformation_12x12,
)
from printsim.fea.klipper_shaper import AxisShaperGraph, calibrate_axis_shapers
from printsim.fea.shell_mitc4 import fastener_spring_matrices, mitc4_shell_matrices
from printsim.fea.solver import classify_part_group
from printsim.geometry.builder import AssemblyModel
from printsim.kinematics.solvers import (
    compute_bracket_stiffness_n_per_m,
    compute_drive_stiffness_details_n_per_m,
    compute_drive_stiffness_n_per_m,
)
from printsim.schema import FRAME_FEET_CATALOG, GANTRY_BRACKET_CATALOG


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

_LIMITING_KIND_LABELS: dict[str, str] = {
    "structural": "Frame & Gantry Structure",
    "belt": "Belt Stretch",
    "motor": "Stepper Motor Magnetic Stiffness",
    "bracket": "Gantry Bracket Flex",
}


@dataclass
class ModeShape:
    mode_number: int
    frequency_hz: float
    dominant_axis: str
    toolhead_participation_xyz: list[float]
    nodal_vectors: dict[str, list[float]]  # normalized 3D displacement per node for animation
    nodal_rotations: dict[str, list[float]] = field(default_factory=dict)
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
    top_causes: list[dict] = field(default_factory=list)
    limiting_axis_share_pct: float = 0.0
    x_axis_share_pct: float = 0.0
    y_axis_share_pct: float = 0.0
    is_limiting_axis_mode: bool = False
    is_primary_limiting_mode: bool = False


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
    limiting_modal_summary: dict | None = None

    def to_dict(self) -> dict:
        return {
            "fundamental_freq_hz": round(self.fundamental_freq_hz, 2),
            "nozzle_pitch_twist_freq_hz": round(self.nozzle_pitch_twist_freq_hz, 2),
            "panel_drum_freq_hz": round(self.panel_drum_freq_hz, 2),
            "frame_damping_ratio": round(self.frame_damping_ratio, 4),
            "recommended_max_accel_mm_s2": round(self.recommended_max_accel_mm_s2, 0),
            "accel_bottleneck": self.accel_bottleneck,
            "worst_print_mode_number": self.worst_print_mode_number,
            "limiting_modal_summary": self.limiting_modal_summary,
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
                    "nodal_rotations": m.nodal_rotations,
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
                    "top_causes": m.top_causes,
                    "limiting_axis_share_pct": m.limiting_axis_share_pct,
                    "x_axis_share_pct": m.x_axis_share_pct,
                    "y_axis_share_pct": m.y_axis_share_pct,
                    "is_limiting_axis_mode": m.is_limiting_axis_mode,
                    "is_primary_limiting_mode": m.is_primary_limiting_mode,
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
        if abs(mf - f_eff) >= 3.0 and 10.0 <= mf <= 215.0 and part_val > 0.15
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


def _canonical_bay_key(bay: str) -> str:
    if bay in ("top_brace", "front_brace", "backbone"):
        return bay
    return bay.split("_")[0]


def _classify_modal_group(member_id: str, role: str = "", bay: str = "") -> tuple[str, str]:
    """Map member/spring/bay IDs to canonical physical part groups (matching sensitivity.py)."""
    mid = member_id.strip()
    if mid.startswith("feet_") or role == "foot" or bay == "feet":
        return "frame_feet", "Machine Feet & Benchtop Pads"
    if bay == "z_joint" or mid.startswith("z_joint"):
        return "z_gantry_joints", "Z-Axis Gantry Joints"
    if bay == "top_brace" or mid.startswith(("bay_top_brace", "top_brace")):
        return "top_brace", "Top Open-Center Brace"
    if bay == "front_brace" or mid.startswith(("bay_front_brace", "front_brace")):
        return "front_brace", "Front Open-Door Brace"
    return classify_part_group(member_id, role)


def _role_to_subsystem(role: str) -> str:
    if role in ("toolhead", "cog", "nozzle"):
        return "toolhead"
    if role in ("gantry", "rail"):
        return "gantry"
    if role == "bed":
        return "bed"
    return "frame"


def _classify_lumped_mass_group(
    assembly: AssemblyModel, node_id: str, role: str
) -> tuple[str, str, str]:
    """Group non-fixed lumped masses into clean physical moving/structural inertia bodies."""
    th_ids = {
        assembly.toolhead_node_id,
        assembly.toolhead_cog_node_id,
        assembly.nozzle_node_id,
    }
    if node_id in th_ids or role in ("toolhead", "cog", "nozzle"):
        return "toolhead_assembly", "Toolhead Moving Mass", "toolhead"
    if node_id.startswith("y_car_"):
        return "y_carriages", "Y-Axis Gantry Carriages", "gantry"
    if node_id in ("z_l", "z_r"):
        return "z_carriages", "Z-Axis Carriages & X-Motor", "gantry"
    if node_id.startswith("c_") and role == "gantry":
        return "delta_carriages", "Delta Tower Carriages", "gantry"
    if role == "bed":
        if assembly.spec.architecture == "cartesian_bedslinger":
            return "moving_y_bed", "Moving Y-Bed Mass", "bed"
        return "bed_assembly", "Heated Bed Assembly", "bed"
    if role == "foot":
        return "frame_feet", "Machine Feet & Base Corners", "frame"
    if role == "gantry":
        return "gantry_motors_idlers", "Gantry Motors & Idler Corners", "gantry"
    if role in ("frame", "mount", "tower", "brace"):
        return "frame_motors_mounts", "Frame Motors & Corner Hardware", "frame"
    gid, label = _classify_modal_group(node_id, role)
    return gid, label, _role_to_subsystem(role)


def _precompute_modal_element_blocks(
    assembly: AssemblyModel,
    node_idx: dict[str, int],
) -> tuple[
    list[tuple[str, str, str, list[int], np.ndarray, np.ndarray | None]],
    list[tuple[str, str, str, int, np.ndarray]],
]:
    """Precompute element (Ke, Me) and lumped-mass blocks once per solve_modal_fea call."""
    elem_blocks: list[tuple[str, str, str, list[int], np.ndarray, np.ndarray | None]] = []

    for mem in assembly.members:
        i0 = node_idx[mem.start_id]
        i1 = node_idx[mem.end_id]
        p0 = assembly.nodes[i0].position_m
        p1 = assembly.nodes[i1].position_m
        r_mat, length = beam_rotation_matrix(p0, p1)
        t_mat = transformation_12x12(r_mat)
        bend_scale = 0.08 if mem.is_truss else 1.0
        k_loc = local_beam_stiffness(
            e_pa=mem.material.youngs_modulus_pa,
            g_pa=mem.material.effective_shear_modulus_pa,
            area=mem.profile.area_m2,
            iy=mem.profile.iy_m4,
            iz=mem.profile.iz_m4,
            j_torsion=mem.profile.j_m4,
            length=length,
            bending_scale=bend_scale,
        )
        m_loc = local_beam_mass(
            rho=mem.material.density_kg_m3,
            area=mem.profile.area_m2,
            iy=mem.profile.iy_m4,
            iz=mem.profile.iz_m4,
            length=length,
        )
        ke = t_mat.T @ k_loc @ t_mat
        me = t_mat.T @ m_loc @ t_mat
        dofs = [6 * i0 + d for d in range(6)] + [6 * i1 + d for d in range(6)]
        gid, label = _classify_modal_group(mem.id, mem.role)
        elem_blocks.append((gid, label, _role_to_subsystem(mem.role), dofs, ke, me))

    for sh in assembly.shell_elements:
        n_indices = [node_idx[nid] for nid in sh.node_ids]
        coords = np.array([assembly.nodes[ni].position_m for ni in n_indices], dtype=float)
        k_sh, m_sh, _ = mitc4_shell_matrices(
            coords_3d=coords,
            e_pa=sh.material.youngs_modulus_pa,
            nu=sh.material.poisson_ratio,
            thickness_m=sh.thickness_m,
            areal_mass_kg_m2=sh.areal_mass_kg_m2,
            d_bend_nm=sh.d_bend_nm,
            g_transverse_h_n_m=sh.g_transverse_h_n_m,
            e_mem_h_n_m=sh.e_mem_h_n_m,
        )
        dofs = [6 * ni + d for ni in n_indices for d in range(6)]
        bay_key = _canonical_bay_key(sh.bay)
        gid, label = _classify_modal_group(f"bay_{bay_key}", "panel", bay=sh.bay)
        elem_blocks.append((gid, label, "frame", dofs, k_sh, m_sh))

    for sp in assembly.fastener_springs:
        i_s = node_idx[sp.shell_node_id]
        i_f = node_idx[sp.frame_node_id]
        k_sp = fastener_spring_matrices(
            k_shear_n_m=sp.k_shear_n_m,
            k_normal_n_m=sp.k_normal_n_m,
            k_rot_nm_rad=sp.k_rot_nm_rad,
            normal_vec=np.asarray(sp.normal_vec, dtype=float),
            offset_vec=np.asarray(sp.offset_m, dtype=float),
        )
        dofs = [6 * i_s + d for d in range(6)] + [6 * i_f + d for d in range(6)]
        bay_raw = sp.bay
        bay_prefix = _canonical_bay_key(bay_raw)
        if bay_raw not in ("bed_mount", "toolhead", "feet", "z_joint") and bay_prefix in (
            "left",
            "right",
            "rear",
            "top",
            "backbone",
            "top_brace",
            "front_brace",
        ):
            gid, label = _classify_modal_group(f"bay_{bay_prefix}", "panel", bay=bay_raw)
            subsys = "frame"
        else:
            sp_role = (
                "bed"
                if bay_raw == "bed_mount"
                else (
                    "toolhead"
                    if bay_raw == "toolhead"
                    else ("gantry" if bay_raw == "z_joint" else "frame")
                )
            )
            gid, label = _classify_modal_group(sp.id, sp_role, bay_raw)
            subsys = _role_to_subsystem(sp_role)
        elem_blocks.append((gid, label, subsys, dofs, k_sp, None))

    lumped_blocks: list[tuple[str, str, str, int, np.ndarray]] = []
    for i, node in enumerate(assembly.nodes):
        if node.lumped_mass_kg <= 0.0 or node.fixed:
            continue
        m_val = float(node.lumped_mass_kg)
        r_gyr = 0.007 if node.role == "nozzle" else 0.025
        rot_i = m_val * (r_gyr**2)
        m_diag_6 = np.array([m_val, m_val, m_val, rot_i, rot_i, rot_i], dtype=float)
        gid, label, subsys = _classify_lumped_mass_group(assembly, node.id, node.role)
        lumped_blocks.append((gid, label, subsys, 6 * i, m_diag_6))

    return elem_blocks, lumped_blocks


def _fmt_hz_gain(gain: float) -> str:
    return f"{gain:.2f}" if 0.0 < gain < 0.5 else f"{gain:.1f}"


def _compute_mode_root_causes(
    full_vec: np.ndarray,
    lam: float,
    freq_hz: float,
    elem_blocks: list[tuple[str, str, str, list[int], np.ndarray, np.ndarray | None]],
    lumped_blocks: list[tuple[str, str, str, int, np.ndarray]],
) -> tuple[list[dict], dict[str, dict], float, dict[str, dict], float]:
    """Compute exact per-group strain & kinetic energy shares and Fox-Kapoor mode causes."""
    u_groups: dict[str, dict] = {}
    t_groups: dict[str, dict] = {}
    u_total = 0.0
    t_total = 0.0

    for gid, label, subsys, dofs, ke, me in elem_blocks:
        u_e = full_vec[dofs]
        strain_e = max(0.0, float(u_e @ ke @ u_e))
        u_total += strain_e
        ug = u_groups.get(gid)
        if ug is None:
            u_groups[gid] = {"id": gid, "label": label, "subsystem": subsys, "energy": strain_e}
        else:
            ug["energy"] += strain_e

        if me is not None:
            kin_e = max(0.0, float(u_e @ me @ u_e))
            t_total += kin_e
            tg = t_groups.get(gid)
            if tg is None:
                t_groups[gid] = {"id": gid, "label": label, "subsystem": subsys, "energy": kin_e}
            else:
                tg["energy"] += kin_e

    for gid, label, subsys, base_dof, m_diag_6 in lumped_blocks:
        u_nd = full_vec[base_dof : base_dof + 6]
        kin_nd = max(0.0, float(np.sum(m_diag_6 * (u_nd * u_nd))))
        t_total += kin_nd
        tg = t_groups.get(gid)
        if tg is None:
            t_groups[gid] = {"id": gid, "label": label, "subsystem": subsys, "energy": kin_nd}
        else:
            tg["energy"] += kin_nd

    denom_fk = 8.0 * (math.pi**2) * max(freq_hz, 1e-6)
    u_norm_denom = max(u_total, 1e-18)
    t_norm_denom = max(t_total, 1e-18)

    stiff_all: list[dict] = []
    for gid, g in u_groups.items():
        u_val = float(g["energy"])
        share_pct = round(100.0 * u_val / u_norm_denom, 1)
        est_hz = round(0.5 * u_val / denom_fk, 2)
        hz_s = _fmt_hz_gain(est_hz)
        if gid in (
            "side_panels",
            "rear_panel",
            "top_panel",
            "enclosure_panels",
            "trident_backbone",
            "top_brace",
            "front_brace",
        ):
            detail = f"{share_pct:.0f}% of mode shear/panel flex (+{hz_s} Hz if +50% stiffer)"
        elif gid == "frame_feet":
            detail = f"{share_pct:.0f}% of mode anchor flex (+{hz_s} Hz if +50% stiffer)"
        elif gid == "z_gantry_joints":
            detail = f"{share_pct:.0f}% of mode Z-joint flex (+{hz_s} Hz if +50% stiffer)"
        else:
            detail = f"{share_pct:.0f}% of mode bending flex (+{hz_s} Hz if +50% stiffer)"
        stiff_all.append(
            {
                "id": gid,
                "label": str(g["label"]),
                "subsystem": str(g["subsystem"]),
                "kind": "stiffness",
                "share_pct": float(share_pct),
                "est_hz_gain": float(est_hz),
                "detail": detail,
            }
        )

    stiff_eligible = [c for c in stiff_all if c["share_pct"] >= 5.0]
    stiff_eligible.sort(key=lambda c: (-c["est_hz_gain"], -c["share_pct"]))
    stiff_picked = stiff_eligible[:3]
    if not stiff_picked and u_total > 1e-18 and stiff_all:
        stiff_picked = [max(stiff_all, key=lambda c: (c["share_pct"], c["est_hz_gain"]))]

    mass_all: list[dict] = []
    for gid, g in t_groups.items():
        t_val = float(g["energy"])
        share_pct = round(100.0 * t_val / t_norm_denom, 1)
        est_hz = round(0.25 * max(lam, 0.0) * t_val / denom_fk, 2)
        hz_s = _fmt_hz_gain(est_hz)
        mass_all.append(
            {
                "id": gid,
                "label": str(g["label"]),
                "subsystem": str(g["subsystem"]),
                "kind": "mass",
                "share_pct": float(share_pct),
                "est_hz_gain": float(est_hz),
                "detail": f"{share_pct:.0f}% of mode swinging inertia (+{hz_s} Hz if 25% lighter)",
            }
        )

    mass_eligible = [c for c in mass_all if c["share_pct"] >= 12.0]
    mass_eligible.sort(key=lambda c: (-c["est_hz_gain"], -c["share_pct"]))
    mass_picked = mass_eligible[:2]

    top_causes = stiff_picked + mass_picked
    top_causes.sort(key=lambda c: (-c["est_hz_gain"], c["kind"] != "stiffness", -c["share_pct"]))
    return top_causes[:5], u_groups, u_total, t_groups, t_total


def _build_limiting_modal_summary(
    assembly: AssemblyModel,
    limiting: InputShaperRecommendation,
    other: InputShaperRecommendation,
    primary_mode: ModeShape | None,
    primary_u_groups: dict[str, dict],
    primary_u_total: float,
    primary_t_groups: dict[str, dict],
    primary_t_total: float,
    ratio_lim: float,
    feet_rock_frac: float = 0.0,
) -> dict:
    """Build unified limiting-resonance summary bridging 3D structural modes and drive train."""
    spec = assembly.spec
    ax = limiting.axis
    f_eff = float(limiting.effective_freq_hz)
    struct_hz = float(limiting.structural_freq_hz)
    belt_hz = float(limiting.belt_freq_hz)
    accel = float(limiting.max_accel_mm_s2)

    details = compute_drive_stiffness_details_n_per_m(assembly, ax)
    k_cord = float(details["k_belt_cord_n_m"])  # type: ignore[arg-type]
    k_mesh = float(details["k_tooth_mesh_n_m"])  # type: ignore[arg-type]
    k_idler = float(details["k_idler_shaft_n_m"])  # type: ignore[arg-type]
    k_sub = float(details["k_belt_subsystem_n_m"])  # type: ignore[arg-type]
    k_motor_axis = float(details["k_motor_axis_n_m"])  # type: ignore[arg-type]
    k_bracket = float(details["k_bracket_n_m"])  # type: ignore[arg-type]

    inv_cord = 1.0 / max(k_cord, 1e-9)
    inv_mesh = 1.0 / max(k_mesh, 1e-9)
    inv_idler = 1.0 / max(k_idler, 1e-9)
    inv_tot = max(inv_cord + inv_mesh + inv_idler, 1e-18)
    cord_frac = inv_cord / inv_tot
    mesh_frac = inv_mesh / inv_tot
    idler_frac = inv_idler / inv_tot

    raw_belt_shares = {
        "belt_cord_share_pct": 100.0 * cord_frac,
        "tooth_mesh_share_pct": 100.0 * mesh_frac,
        "idler_shaft_share_pct": 100.0 * idler_frac,
    }
    belt_shares_rounded = {k: round(v, 1) for k, v in raw_belt_shares.items()}
    top_belt_key = max(raw_belt_shares, key=lambda k: raw_belt_shares[k])
    belt_shares_rounded[top_belt_key] = round(
        belt_shares_rounded[top_belt_key] + round(100.0 - sum(belt_shares_rounded.values()), 1),
        1,
    )

    belt = spec.gantry.belt
    belt_label = belt.label

    belt_sub_breakdown = {
        "belt_label": belt_label,
        "k_belt_cord_n_per_mm": round(k_cord * 1e-3, 1),
        "k_tooth_mesh_n_per_mm": round(k_mesh * 1e-3, 1),
        "k_idler_shaft_n_per_mm": round(k_idler * 1e-3, 1),
        "k_belt_total_n_per_mm": round(k_sub * 1e-3, 1),
        **belt_shares_rounded,
    }

    s_struct = float(limiting.compliance_share_pct.get("frame", 0.0))
    s_belt = float(limiting.compliance_share_pct.get("belt", 0.0))
    s_motor = float(limiting.compliance_share_pct.get("motor", 0.0))
    s_bracket = float(limiting.compliance_share_pct.get("bracket", 0.0))

    bkt_info = GANTRY_BRACKET_CATALOG.get(
        spec.gantry.bracket_material, GANTRY_BRACKET_CATALOG["abs_printed"]
    )
    bkt_name = str(bkt_info.get("short_name", spec.gantry.bracket_material))

    def _overall_gain(share_val: float) -> tuple[float, float]:
        p = min(max(share_val, 0.0) / 100.0, 0.99)
        f_new = f_eff / math.sqrt(max(1.0 - p / 3.0, 0.05))
        hz_gain = round(max(0.0, f_new - f_eff), 2)
        accel_gain = round(max(0.0, accel * ((f_new / max(f_eff, 1.0)) ** 2 - 1.0)), 0)
        return float(hz_gain), float(accel_gain)

    raw_overall: list[dict] = []

    def _add_overall_item(
        item_id: str,
        label: str,
        category: str,
        subsystem: str,
        raw_share: float,
        detail_prefix: str,
    ) -> None:
        share_r = round(raw_share, 1)
        if share_r < 1.5:
            return
        hz_g, acc_g = _overall_gain(raw_share)
        hz_s = _fmt_hz_gain(hz_g)
        raw_overall.append(
            {
                "id": item_id,
                "label": label,
                "category": category,
                "subsystem": subsystem,
                "share_pct": float(share_r),
                "est_hz_gain": float(hz_g),
                "est_accel_gain_mm_s2": float(acc_g),
                "detail": f"{share_r:.0f}% of total {ax} compliance — {detail_prefix}; +50% stiffer adds +{hz_s} Hz",
            }
        )

    _add_overall_item(
        "belt_cord_stretch",
        f"Belt Cord Stretch ({belt_label})",
        "belt",
        "drive",
        s_belt * cord_frac,
        f"k_cord = {k_cord * 1e-3:.0f} N/mm",
    )
    _add_overall_item(
        "belt_tooth_mesh",
        "Pulley Tooth-Mesh Compliance",
        "belt",
        "drive",
        s_belt * mesh_frac,
        f"k_mesh = {k_mesh * 1e-3:.0f} N/mm",
    )
    _add_overall_item(
        "belt_idler_shaft",
        "Idler Shaft & Bearing Flex",
        "belt",
        "drive",
        s_belt * idler_frac,
        f"k_idler = {k_idler * 1e-3:.0f} N/mm",
    )
    _add_overall_item(
        "motor_stiffness",
        "Stepper Motor Magnetic Stiffness",
        "motor",
        "drive",
        s_motor,
        f"k_motor = {k_motor_axis * 1e-3:.0f} N/mm ({limiting.motor_freq_hz:.1f} Hz)",
    )
    _add_overall_item(
        "gantry_brackets",
        f"Gantry Brackets ({bkt_name})",
        "bracket",
        "drive",
        s_bracket,
        f"k_bracket = {k_bracket * 1e-3:.0f} N/mm ({limiting.bracket_freq_hz:.1f} Hz)",
    )

    pm_num = primary_mode.mode_number if primary_mode is not None else 1
    pm_label = primary_mode.mode_label if primary_mode is not None else "Primary Structural Mode"

    struct_parts: dict[str, dict] = {}
    feet_share = s_struct * min(max(feet_rock_frac, 0.0), 0.99) if s_struct > 0.0 else 0.0
    internal_struct_share = max(0.0, s_struct - feet_share)
    car_struct_share = internal_struct_share / max(ratio_lim, 1.0)
    if primary_u_total > 1e-18 and primary_u_groups:
        for gid, g in primary_u_groups.items():
            g_share = car_struct_share * (float(g["energy"]) / primary_u_total)
            struct_parts[gid] = {
                "id": gid,
                "label": str(g["label"]),
                "subsystem": str(g["subsystem"]),
                "share": g_share,
            }
    elif internal_struct_share > 0.0:
        struct_parts["frame_perimeter_beams"] = {
            "id": "frame_perimeter_beams",
            "label": "Frame & Gantry Structure",
            "subsystem": "frame",
            "share": car_struct_share,
        }

    if ratio_lim > 1.05 and internal_struct_share > 0.0:
        pitch_share = internal_struct_share * (1.0 - 1.0 / ratio_lim)
        if "toolhead_mount" in struct_parts:
            struct_parts["toolhead_mount"]["share"] += pitch_share
            struct_parts["toolhead_mount"]["label"] = "Toolhead & Nozzle Pitch Mount"
        else:
            struct_parts["toolhead_mount"] = {
                "id": "toolhead_mount",
                "label": "Toolhead & Nozzle Pitch Mount",
                "subsystem": "toolhead",
                "share": pitch_share,
            }

    if feet_share > 0.0:
        struct_parts["frame_feet"] = {
            "id": "frame_feet",
            "label": "Benchtop Feet / Isolation Pads",
            "subsystem": "frame",
            "share": feet_share,
        }

    for sp in struct_parts.values():
        _add_overall_item(
            str(sp["id"]),
            str(sp["label"]),
            "structural",
            str(sp["subsystem"]),
            float(sp["share"]),
            f"Mode M{pm_num}",
        )

    raw_overall.sort(key=lambda x: (-x["share_pct"], -x["est_hz_gain"]))
    top_overall_causes = raw_overall[:6]

    mass_lim_kg = float(assembly.x_moving_mass_kg if ax == "X" else assembly.y_moving_mass_kg)
    arch = spec.architecture
    if arch == "delta":
        mass_label = "Moving Effector + Arms"
        mov_groups = {"toolhead_assembly", "delta_carriages", "delta_arms"}
    elif arch == "cartesian_bedslinger":
        if ax == "Y":
            mass_label = "Moving Y Heated Bed"
            mov_groups = {"moving_y_bed"}
        else:
            mass_label = "Moving Toolhead"
            mov_groups = {"toolhead_assembly"}
    else:
        if ax == "Y":
            mass_label = "Moving Y-Gantry + Toolhead"
            mov_groups = {"toolhead_assembly", "x_gantry_beam", "y_carriages"}
        else:
            mass_label = "Moving Toolhead"
            mov_groups = {"toolhead_assembly"}

    t_mov = sum(
        float(primary_t_groups[g]["energy"]) for g in mov_groups if g in primary_t_groups
    )
    t_mov_share = min(1.0, max(0.0, t_mov / max(primary_t_total, 1e-18))) if primary_t_total > 1e-18 else 0.0
    f_drive_new = float(limiting.drive_freq_hz) / math.sqrt(0.75)
    f_struct_new = struct_hz / math.sqrt(max(1.0 - 0.25 * t_mov_share, 0.75))
    f_eff_new_mass = 1.0 / math.sqrt(
        1.0 / (max(f_struct_new, 5.0) ** 2) + 1.0 / (max(f_drive_new, 5.0) ** 2)
    )
    est_hz_gain_25 = round(max(0.0, f_eff_new_mass - f_eff), 2)
    est_accel_gain_25 = round(
        max(0.0, accel * ((f_eff_new_mass / max(f_eff, 1.0)) ** 2 - 1.0)), 0
    )

    moving_mass_info = {
        "axis": ax,
        "mass_g": round(mass_lim_kg * 1000.0, 1),
        "label": mass_label,
        "est_hz_gain_25pct_lighter": float(est_hz_gain_25),
        "est_accel_gain_25pct_lighter_mm_s2": float(est_accel_gain_25),
    }

    limiting_kind = (
        "structural" if limiting.primary_bottleneck == "frame" else limiting.primary_bottleneck
    )
    limiting_kind_label = _LIMITING_KIND_LABELS.get(limiting_kind, limiting_kind)

    if limiting_kind == "structural":
        headline = (
            f"{ax}-Axis @ {f_eff:.1f} Hz ({accel:,.0f} mm/s²) — Limited by Structural Mode "
            f"M{pm_num} ({pm_label}, {s_struct:.0f}% structural vs {s_belt:.0f}% belt)"
        )
    elif limiting_kind == "belt":
        headline = (
            f"{ax}-Axis @ {f_eff:.1f} Hz ({accel:,.0f} mm/s²) — Limited by {ax}-Belt Stretch "
            f"({s_belt:.0f}% belt vs {s_struct:.0f}% structural)"
        )
    elif limiting_kind == "motor":
        headline = (
            f"{ax}-Axis @ {f_eff:.1f} Hz ({accel:,.0f} mm/s²) — Limited by Stepper Magnetic "
            f"Stiffness ({s_motor:.0f}% motor vs {s_struct:.0f}% structural)"
        )
    else:
        headline = (
            f"{ax}-Axis @ {f_eff:.1f} Hz ({accel:,.0f} mm/s²) — Limited by Gantry Bracket Flex "
            f"({s_bracket:.0f}% bracket vs {s_struct:.0f}% structural)"
        )

    belt_included_note = (
        f"Belt stretch IS included in Input Shaper & Max Accel ({s_belt:.0f}% of {ax} compliance, "
        f"{belt_hz:.1f} Hz belt vs {struct_hz:.1f} Hz structural → {f_eff:.1f} Hz effective). "
        "The 3D mode animations below isolate pure structural frame/gantry vibration (locked belts)."
    )

    return {
        "limiting_axis": ax,
        "other_axis": other.axis,
        "effective_freq_hz": round(f_eff, 2),
        "structural_freq_hz": round(struct_hz, 2),
        "belt_freq_hz": round(belt_hz, 2),
        "motor_freq_hz": round(float(limiting.motor_freq_hz), 2),
        "bracket_freq_hz": round(float(limiting.bracket_freq_hz), 2),
        "drive_freq_hz": round(float(limiting.drive_freq_hz), 2),
        "recommended_shaper": limiting.recommended_shaper,
        "max_accel_mm_s2": round(accel, 0),
        "other_axis_effective_freq_hz": round(float(other.effective_freq_hz), 2),
        "other_axis_max_accel_mm_s2": round(float(other.max_accel_mm_s2), 0),
        "limiting_kind": limiting_kind,
        "limiting_kind_label": limiting_kind_label,
        "compliance_share_pct": dict(limiting.compliance_share_pct),
        "structural_share_pct": round(s_struct, 1),
        "belt_share_pct": round(s_belt, 1),
        "motor_share_pct": round(s_motor, 1),
        "bracket_share_pct": round(s_bracket, 1),
        "drive_share_pct": round(s_belt + s_motor + s_bracket, 1),
        "primary_structural_mode_number": int(pm_num),
        "primary_structural_mode_freq_hz": round(float(primary_mode.frequency_hz), 2)
        if primary_mode is not None
        else 0.0,
        "primary_structural_mode_label": pm_label,
        "primary_structural_mode_axis": primary_mode.excited_axis
        if primary_mode is not None
        else ax,
        "primary_structural_mode_share_pct": round(float(primary_mode.limiting_axis_share_pct), 1)
        if primary_mode is not None
        else 0.0,
        "nozzle_pitch_amplification_ratio": round(float(ratio_lim), 3),
        "nozzle_whip_amp_ratio": round(float(ratio_lim), 3),
        "belt_sub_breakdown": belt_sub_breakdown,
        "top_overall_causes": top_overall_causes,
        "moving_mass_info": moving_mass_info,
        "headline": headline,
        "belt_included_note": belt_included_note,
    }


def _solve_generalized_eigenproblem(
    K_ff: np.ndarray,
    M_ff: np.ndarray,
    target_scan: int,
) -> tuple[np.ndarray, np.ndarray, object | None]:
    """Solve K_ff phi = lam M_ff phi with sparse shift-invert or dense Cholesky eigh."""
    n_free = K_ff.shape[0]
    k_scan = min(target_scan, n_free - 2 if n_free > 250 else n_free)
    used_sparse = False
    lu_solver = None
    if n_free > 250 and 1 <= k_scan < n_free - 1:
        try:
            k_csc = csc_matrix(K_ff)
            m_csc = csc_matrix(M_ff)
            lu = splu(k_csc)
            lu_solver = lu
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
            lu_solver = None
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

    return eigvals, eigvecs, lu_solver


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

    # Exclude the primary dominant mode from secondary_modes and normalize by its residue
    # so higher-frequency local modes cannot wash out secondary PSD participation.
    dom_f, dom_r, _ = max(sig, key=lambda item: item[2])
    sec_modes = [
        (f, min(1.0, math.sqrt(r / max(dom_r, 1e-18))))
        for f, r, _ in sig
        if abs(f - dom_f) >= 3.0 and abs(f - eff_hz) >= 3.0
    ]
    return eff_hz, sec_modes


def _geometric_car_freq(
    entries: list[tuple[float, float, float, float]], fallback_hz: float
) -> float:
    """Smooth panel-attenuated geometric compliance-weighted carriage frequency."""
    log_sum = 0.0
    w_sum = 0.0
    for f_k, c_car, ke_p, g_m in entries:
        if f_k < 5.0 or c_car <= 0.0:
            continue
        p_att = (
            max(0.02, 1.0 - max(0.0, ke_p - 0.62) / 0.35)
            if (ke_p > 0.62 and g_m < 0.35)
            else 1.0
        )
        w_k = c_car * p_att
        log_sum += w_k * math.log(f_k)
        w_sum += w_k
    if w_sum <= 1e-24:
        for f_k, c_car, ke_p, g_m in entries:
            if f_k < 5.0:
                continue
            p_att = (
                max(0.02, 1.0 - max(0.0, ke_p - 0.62) / 0.35)
                if (ke_p > 0.62 and g_m < 0.35)
                else 1.0
            )
            w_k = abs(c_car) * p_att
            log_sum += w_k * math.log(f_k)
            w_sum += w_k
    if w_sum <= 1e-24:
        return fallback_hz
    return math.exp(log_sum / w_sum)


def _eval_panel_drum_status(
    phi: np.ndarray,
    m_phi: np.ndarray,
    m_gen: float,
    trans: np.ndarray,
    noz_norm_all: float,
    has_panels: bool,
    panel_indices: list[int],
    non_panel_indices: list[int],
    gantry_indices: list[int],
    free_panel_mask: np.ndarray,
    free_th_mask: np.ndarray,
) -> tuple[bool, float, float, bool]:
    """Classify isolated enclosure panel-drum modes cleanly across both free and pinned solves."""
    if not has_panels:
        return False, 0.0, 1.0, False
    np_max = float(np.max(np.linalg.norm(trans[non_panel_indices], axis=1)))
    gantry_max = (
        float(np.max(np.linalg.norm(trans[gantry_indices], axis=1)))
        if gantry_indices
        else np_max
    )
    phi_p = np.where(free_panel_mask, phi, 0.0)
    ke_panel = float(phi_p @ np.where(free_panel_mask, m_phi, 0.0)) / m_gen
    ke_th_total = max(0.0, float(phi[free_th_mask] @ m_phi[free_th_mask]) / m_gen)
    is_drum = (
        (ke_panel > 0.55 and noz_norm_all < 0.12 and ke_th_total < 0.04)
        or (ke_panel > 0.58 and noz_norm_all < 0.25 and ke_th_total < 0.04)
        or (ke_panel > 0.68 and gantry_max < 0.35 and noz_norm_all < 0.28)
        or (np_max < 0.25 and gantry_max < 0.20)
    )
    return is_drum, ke_panel, gantry_max, ke_panel > 0.50


def _dunkerley_overhang_struct(
    f_car: float,
    axis: int,
    architecture: str,
    th_idx: int,
    cog_idx: int,
    noz_idx: int,
    bed_idx: int | None,
    has_nozzle: bool,
    m_cog_eff: float,
    m_noz_eff: float,
    solve_unit_load_fn,
) -> tuple[float, float]:
    """Combine carriage frequency with exact Castigliano static overhang compliance."""
    if axis == 1 and architecture == "cartesian_bedslinger":
        return f_car, 1.0
    if cog_idx == th_idx and not has_nozzle:
        return f_car, 1.0
    u_th = solve_unit_load_fn(th_idx, axis)
    if u_th is None:
        return f_car, 1.0
    bed_dof = (6 * bed_idx + axis) if bed_idx is not None else -1
    c_th = float(u_th[6 * th_idx + axis] - (u_th[bed_dof] if bed_dof >= 0 else 0.0))

    dc_cog = 0.0
    if cog_idx != th_idx:
        u_cg = solve_unit_load_fn(cog_idx, axis)
        if u_cg is not None:
            c_cg = float(u_cg[6 * cog_idx + axis] - (u_cg[bed_dof] if bed_dof >= 0 else 0.0))
            dc_cog = max(0.0, c_cg - c_th)

    dc_noz = 0.0
    if has_nozzle:
        u_nz = solve_unit_load_fn(noz_idx, axis)
        if u_nz is not None:
            dc_noz = max(0.0, float(u_nz[6 * noz_idx + axis] - u_nz[6 * cog_idx + axis]))

    dunkerley_extra = (2.0 * math.pi) ** 2 * (
        (m_cog_eff + m_noz_eff) * dc_cog + m_noz_eff * dc_noz
    )
    inv_f2 = 1.0 / max(f_car * f_car, 1e-12) + max(0.0, dunkerley_extra)
    f_eff = 1.0 / math.sqrt(max(inv_f2, 1e-12))
    ratio = (f_car / max(f_eff, 1e-12)) ** 2
    return f_eff, ratio


def _compute_benchtop_rocking_penalty(
    assembly: AssemblyModel,
    M_ff: np.ndarray,
    free_dofs: np.ndarray,
    node_idx: dict[str, int],
    feet_springs: list,
    feet_cfg: dict,
) -> tuple[float, float]:
    """Compute physical series rocking+shear compliance ratio against feet springs & base mass."""
    base_mass_kg = float(feet_cfg.get("base_mass_kg", 0.0))
    feet_dd = float(feet_cfg.get("damping_delta", 0.0))
    feet_nids = sorted({sp.frame_node_id for sp in feet_springs if sp.frame_node_id in node_idx})
    if not feet_nids:
        return 0.0, 0.0

    n_free = len(free_dofs)
    v_x = np.zeros(n_free, dtype=float)
    z_vec = np.zeros(n_free, dtype=float)
    z_feet = sum(float(assembly.nodes[node_idx[nid]].position_m[2]) for nid in feet_nids) / len(
        feet_nids
    )
    for fi, gdof in enumerate(free_dofs):
        if int(gdof) % 6 == 0:
            ni = int(gdof) // 6
            v_x[fi] = 1.0
            z_vec[fi] = float(assembly.nodes[ni].position_m[2]) - z_feet

    M_tot = max(float(v_x @ M_ff @ v_x), 1e-6)
    m_printer = max(M_tot - base_mass_kg, 0.5)
    z_cog = max(0.02, float(v_x @ M_ff @ z_vec) / M_tot)

    coords = [assembly.nodes[node_idx[nid]].position_m for nid in feet_nids]
    x_c = sum(float(c[0]) for c in coords) / len(coords)
    y_c = sum(float(c[1]) for c in coords) / len(coords)

    k_lat = 0.0
    K_theta_x = 0.0
    K_theta_y = 0.0
    for sp in feet_springs:
        if sp.frame_node_id not in node_idx:
            continue
        pos = assembly.nodes[node_idx[sp.frame_node_id]].position_m
        dx = float(pos[0]) - x_c
        dy = float(pos[1]) - y_c
        k_lat += float(sp.k_shear_n_m)
        K_theta_x += float(sp.k_normal_n_m) * (dx * dx) + float(sp.k_rot_nm_rad)
        K_theta_y += float(sp.k_normal_n_m) * (dy * dy) + float(sp.k_rot_nm_rad)

    k_feet_x = 1.0 / (1.0 / max(k_lat, 1.0) + (z_cog * z_cog) / max(K_theta_x, 1e-3))
    k_feet_y = 1.0 / (1.0 / max(k_lat, 1.0) + (z_cog * z_cog) / max(K_theta_y, 1e-3))
    k_ref_inert = m_printer * ((2.0 * math.pi * 45.0) ** 2)
    mass_scale = (m_printer / M_tot) ** 1.5
    damp_scale = 0.045 / (0.045 + max(0.0, feet_dd))
    pen_x = 0.052 * mass_scale * (k_ref_inert / max(k_feet_x, 1.0)) * damp_scale
    pen_y = 0.052 * mass_scale * (k_ref_inert / max(k_feet_y, 1.0)) * damp_scale
    return pen_x, pen_y


def _solve_internal_pinned_struct(
    assembly: AssemblyModel,
    K_ff: np.ndarray,
    M_ff: np.ndarray,
    free_dofs: np.ndarray,
    dof_to_free: dict[int, int],
    node_idx: dict[str, int],
    m_floor: float,
    feet_springs: list,
    feet_cfg: dict,
    target_scan: int,
    has_panels: bool,
    panel_indices: list[int],
    non_panel_indices: list[int],
    gantry_indices: list[int],
    free_panel_mask: np.ndarray,
    free_th_mask: np.ndarray,
    th_idx: int,
    cog_idx: int,
    noz_idx: int,
    bed_idx: int | None,
    has_nozzle: bool,
    dr_noz: np.ndarray,
    dr_th: np.ndarray,
    fx_exc: np.ndarray,
    fy_exc: np.ndarray,
    m_cog_eff: float,
    m_noz_eff: float,
    fundamental: float,
) -> tuple[
    float,
    float,
    float,
    float,
    list[tuple[float, float]],
    list[tuple[float, float]],
    dict | None,
    dict | None,
]:
    """Solve internal frame+gantry modes with foot translations pinned & base mass excluded."""
    K_pin = K_ff.copy()
    M_pin = M_ff.copy()
    n_dof = len(assembly.nodes) * 6

    for sp in feet_springs:
        i_s = node_idx[sp.shell_node_id]
        i_f = node_idx[sp.frame_node_id]
        dofs = [6 * i_s + d for d in range(6)] + [6 * i_f + d for d in range(6)]
        pairs = [(ei, dof_to_free[g]) for ei, g in enumerate(dofs) if g in dof_to_free]
        if not pairs:
            continue
        k_old = fastener_spring_matrices(
            k_shear_n_m=sp.k_shear_n_m,
            k_normal_n_m=sp.k_normal_n_m,
            k_rot_nm_rad=sp.k_rot_nm_rad,
            normal_vec=np.asarray(sp.normal_vec, dtype=float),
            offset_vec=np.asarray(sp.offset_m, dtype=float),
        )
        k_new = fastener_spring_matrices(
            k_shear_n_m=1e9,
            k_normal_n_m=1e9,
            k_rot_nm_rad=1e9,
            normal_vec=np.asarray(sp.normal_vec, dtype=float),
            offset_vec=np.asarray(sp.offset_m, dtype=float),
        )
        e_idx = [ei for ei, _ in pairs]
        f_idx = [fi for _, fi in pairs]
        K_pin[np.ix_(f_idx, f_idx)] += k_new[np.ix_(e_idx, e_idx)] - k_old[np.ix_(e_idx, e_idx)]

    K_pin = 0.5 * (K_pin + K_pin.T)

    base_mass_kg = float(feet_cfg.get("base_mass_kg", 0.0))
    feet_nids = sorted({sp.frame_node_id for sp in feet_springs if sp.frame_node_id in node_idx})
    if base_mass_kg > 0.0 and feet_nids:
        dm = base_mass_kg / len(feet_nids)
        d_rot = dm * (0.025**2)
        for nid in feet_nids:
            ni = node_idx[nid]
            for d in range(3):
                fi = dof_to_free.get(6 * ni + d)
                if fi is not None:
                    M_pin[fi, fi] = max(m_floor, float(M_pin[fi, fi]) - dm)
            for d in range(3, 6):
                fi = dof_to_free.get(6 * ni + d)
                if fi is not None:
                    M_pin[fi, fi] = max(m_floor, float(M_pin[fi, fi]) - d_rot)
    M_pin = 0.5 * (M_pin + M_pin.T)

    eigvals_p, eigvecs_p, lu_pin = _solve_generalized_eigenproblem(K_pin, M_pin, target_scan)

    rec_x_pin: list[tuple[float, float]] = []
    rec_y_pin: list[tuple[float, float]] = []
    car_pin_entries: list[tuple[float, float, float, float]] = []
    best_x_w = -1.0
    best_y_w = -1.0
    best_pin_x: dict | None = None
    best_pin_y: dict | None = None

    for idx in range(len(eigvals_p)):
        lam = max(float(eigvals_p[idx]), 0.0)
        if lam <= 1.0:
            continue
        freq_hz = math.sqrt(lam) / (2.0 * math.pi)
        phi = eigvecs_p[:, idx]
        m_phi = M_pin @ phi
        m_gen = max(float(phi @ m_phi), 1e-18)

        full_vec = np.zeros(n_dof, dtype=float)
        full_vec[free_dofs] = phi
        trans_raw = np.array([full_vec[6 * i : 6 * i + 3] for i in range(len(assembly.nodes))])
        rot_raw = np.array([full_vec[6 * i + 3 : 6 * i + 6] for i in range(len(assembly.nodes))])
        max_amp = float(np.max(np.linalg.norm(trans_raw, axis=1)))
        trans = trans_raw / max_amp if max_amp > 1e-12 else trans_raw.copy()
        rot = rot_raw / max_amp if max_amp > 1e-12 else rot_raw.copy()

        if bed_idx is not None:
            noz_rel_all = np.abs(trans[noz_idx] - trans[bed_idx])
        else:
            noz_rel_all = np.abs(trans[noz_idx])
        noz_norm_all = float(np.linalg.norm(noz_rel_all))

        is_drum, ke_panel, gantry_max_mode, _ = _eval_panel_drum_status(
            phi=phi,
            m_phi=m_phi,
            m_gen=m_gen,
            trans=trans,
            noz_norm_all=noz_norm_all,
            has_panels=has_panels,
            panel_indices=panel_indices,
            non_panel_indices=non_panel_indices,
            gantry_indices=gantry_indices,
            free_panel_mask=free_panel_mask,
            free_th_mask=free_th_mask,
        )
        if is_drum:
            continue

        pf_x = float(full_vec @ fx_exc)
        pf_y = float(full_vec @ fy_exc)
        if bed_idx is not None:
            rel_x_raw = float(full_vec[6 * noz_idx + 0] - full_vec[6 * bed_idx + 0])
            rel_y_raw = float(full_vec[6 * noz_idx + 1] - full_vec[6 * bed_idx + 1])
            rel_x_car = float(full_vec[6 * th_idx + 0] - full_vec[6 * bed_idx + 0])
            rel_y_car = float(full_vec[6 * th_idx + 1] - full_vec[6 * bed_idx + 1])
        else:
            rel_x_raw = float(full_vec[6 * noz_idx + 0])
            rel_y_raw = float(full_vec[6 * noz_idx + 1])
            rel_x_car = float(full_vec[6 * th_idx + 0])
            rel_y_car = float(full_vec[6 * th_idx + 1])

        comp_x_noz = (rel_x_raw * pf_x) / (m_gen * lam)
        comp_y_noz = (rel_y_raw * pf_y) / (m_gen * lam)
        comp_x_car = (rel_x_car * pf_x) / (m_gen * lam)
        comp_y_car = (rel_y_car * pf_y) / (m_gen * lam)
        rx = abs(comp_x_noz)
        ry = abs(comp_y_noz)
        rec_x_pin.append((freq_hz, rx))
        rec_y_pin.append((freq_hz, ry))
        car_pin_entries.append((freq_hz, comp_x_car, comp_y_car, max(0.0, ke_panel), gantry_max_mode))

        if freq_hz >= 5.0:
            wx = rx / (freq_hz * freq_hz)
            wy = ry / (freq_hz * freq_hz)
            if wx > best_x_w:
                best_x_w = wx
                best_pin_x = {"full_vec": full_vec, "lam": lam, "freq_hz": freq_hz}
            if wy > best_y_w:
                best_y_w = wy
                best_pin_y = {"full_vec": full_vec, "lam": lam, "freq_hz": freq_hz}

    struct_x_car_int = _geometric_car_freq(
        [(f, ccx, kp, gm) for f, ccx, _, kp, gm in car_pin_entries], fundamental
    )
    struct_y_car_int = _geometric_car_freq(
        [(f, ccy, kp, gm) for f, _, ccy, kp, gm in car_pin_entries], fundamental
    )
    _, norm_scanned_x = _receptance_structural_freq_and_psd(rec_x_pin, fundamental)
    _, norm_scanned_y = _receptance_structural_freq_and_psd(rec_y_pin, fundamental)

    pin_cache: dict[tuple[int, int], np.ndarray] = {}

    def _solve_pin_unit(node_i: int, axis: int) -> np.ndarray | None:
        key = (node_i, axis)
        if key in pin_cache:
            return pin_cache[key]
        gdof = 6 * node_i + axis
        fi = dof_to_free.get(gdof)
        if fi is None:
            return None
        rhs = np.zeros(len(free_dofs), dtype=float)
        rhs[fi] = 1.0
        try:
            u_f = lu_pin.solve(rhs) if lu_pin is not None else np.linalg.solve(K_pin, rhs)
        except (np.linalg.LinAlgError, RuntimeError):
            return None
        u_full = np.zeros(n_dof, dtype=float)
        u_full[free_dofs] = u_f
        pin_cache[key] = u_full
        return u_full

    arch = assembly.spec.architecture
    struct_x_int, ratio_x = _dunkerley_overhang_struct(
        struct_x_car_int,
        0,
        arch,
        th_idx,
        cog_idx,
        noz_idx,
        bed_idx,
        has_nozzle,
        m_cog_eff,
        m_noz_eff,
        _solve_pin_unit,
    )
    struct_y_int, ratio_y = _dunkerley_overhang_struct(
        struct_y_car_int,
        1,
        arch,
        th_idx,
        cog_idx,
        noz_idx,
        bed_idx,
        has_nozzle,
        m_cog_eff,
        m_noz_eff,
        _solve_pin_unit,
    )
    return (
        struct_x_int,
        struct_y_int,
        ratio_x,
        ratio_y,
        norm_scanned_x,
        norm_scanned_y,
        best_pin_x,
        best_pin_y,
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

    feet_springs = [sp for sp in assembly.fastener_springs if sp.bay == "feet"]
    has_feet = bool(feet_springs)

    k_ret = min(num_modes, n_free)
    target_scan = 56 if has_panels else (28 if has_feet else max(num_modes, 16))
    eigvals, eigvecs, lu_solver = _solve_generalized_eigenproblem(K_ff, M_ff, target_scan)

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
    if bed_idx is not None:
        r_bed = assembly.nodes[bed_idx].position_m
        dr_noz = assembly.nodes[noz_idx].position_m - r_bed
        dr_th = assembly.nodes[th_idx].position_m - r_bed
    else:
        dr_noz = np.zeros(3, dtype=float)
        dr_th = np.zeros(3, dtype=float)

    feet_blocks: list[tuple[list[int], np.ndarray]] = []
    for sp in feet_springs:
        i_s = node_idx[sp.shell_node_id]
        i_f = node_idx[sp.frame_node_id]
        k_sp = fastener_spring_matrices(
            k_shear_n_m=sp.k_shear_n_m,
            k_normal_n_m=sp.k_normal_n_m,
            k_rot_nm_rad=sp.k_rot_nm_rad,
            normal_vec=np.asarray(sp.normal_vec, dtype=float),
            offset_vec=np.asarray(sp.offset_m, dtype=float),
        )
        dofs_sp = [6 * i_s + d for d in range(6)] + [6 * i_f + d for d in range(6)]
        feet_blocks.append((dofs_sp, k_sp))

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
        th_share = assembly.toolhead_moving_mass_kg / max(assembly.y_moving_mass_kg, 1e-6)
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
        if n.role in ("frame", "mount", "foot", "tower", "brace", "backbone")
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
    car_mode_entries: list[tuple[float, float, float, float, float]] = []
    noz_car_entries: list[tuple[float, float, float, float, float, float]] = []
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
        rot_raw = np.array([full_vec[6 * i + 3 : 6 * i + 6] for i in range(len(assembly.nodes))])
        max_amp = float(np.max(np.linalg.norm(trans_raw, axis=1)))
        trans = trans_raw / max_amp if max_amp > 1e-12 else trans_raw.copy()
        rot = rot_raw / max_amp if max_amp > 1e-12 else rot_raw.copy()

        if bed_idx is not None:
            bed_wp_noz = trans[bed_idx] + (np.cross(rot[bed_idx], dr_noz) if has_feet else 0.0)
            bed_wp_th = trans[bed_idx] + (np.cross(rot[bed_idx], dr_th) if has_feet else 0.0)
            noz_rel_all = np.abs(trans[noz_idx] - bed_wp_noz)
        else:
            noz_rel_all = np.abs(trans[noz_idx])
        noz_norm_all = float(np.linalg.norm(noz_rel_all))

        is_isolated_panel_drum, ke_panel, gantry_max_mode, above_drum_thresh = _eval_panel_drum_status(
            phi=phi,
            m_phi=m_phi,
            m_gen=m_gen,
            trans=trans,
            noz_norm_all=noz_norm_all,
            has_panels=has_panels,
            panel_indices=panel_indices,
            non_panel_indices=non_panel_indices,
            gantry_indices=gantry_indices,
            free_panel_mask=free_panel_mask,
            free_th_mask=free_th_mask,
        )
        if has_panels:
            if above_drum_thresh and panel_drum_fallback_freq is None:
                panel_drum_fallback_freq = freq_hz
            if is_isolated_panel_drum and panel_drum_freq is None:
                panel_drum_freq = freq_hz

            tr_struct = trans_raw[non_panel_indices]
            max_struct = max(float(np.max(np.linalg.norm(tr_struct, axis=1))), 1e-12)
            if bed_idx is not None:
                rot_bed_s = rot_raw[bed_idx] / max_struct
                bed_wp_noz_s = trans_raw[bed_idx] / max_struct + (
                    np.cross(rot_bed_s, dr_noz) if has_feet else 0.0
                )
                bed_wp_th_s = trans_raw[bed_idx] / max_struct + (
                    np.cross(rot_bed_s, dr_th) if has_feet else 0.0
                )
                noz_rel = np.abs(trans_raw[noz_idx] / max_struct - bed_wp_noz_s)
                th_rel = np.abs(trans_raw[th_idx] / max_struct - bed_wp_th_s)
                bed_z_amp = abs(float(trans_raw[bed_idx][2] / max_struct))
            else:
                noz_rel = np.abs(trans_raw[noz_idx]) / max_struct
                th_rel = np.abs(trans_raw[th_idx]) / max_struct
                bed_z_amp = 0.0
            axis_energy = np.sum(tr_struct**2, axis=0)
            tr_norm = trans_raw / max_struct
        else:
            noz_rel = noz_rel_all
            th_rel = (
                np.abs(trans[th_idx] - bed_wp_th) if bed_idx is not None else np.abs(trans[th_idx])
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
        u_feet_share = 0.0
        if lam > 1.0:
            if feet_blocks:
                u_feet_e = sum(
                    max(0.0, float(full_vec[dofs_sp] @ k_sp @ full_vec[dofs_sp]))
                    for dofs_sp, k_sp in feet_blocks
                )
                u_feet_share = min(1.0, max(0.0, u_feet_e / lam))

            pf_x = float(full_vec @ fx_exc)
            pf_y = float(full_vec @ fy_exc)
            if bed_idx is not None:
                u_b_fv = full_vec[6 * bed_idx : 6 * bed_idx + 3]
                th_b_fv = full_vec[6 * bed_idx + 3 : 6 * bed_idx + 6]
                u_wp_noz_fv = u_b_fv + (np.cross(th_b_fv, dr_noz) if has_feet else 0.0)
                u_wp_th_fv = u_b_fv + (np.cross(th_b_fv, dr_th) if has_feet else 0.0)
                rel_x_raw = float(full_vec[6 * noz_idx + 0] - u_wp_noz_fv[0])
                rel_y_raw = float(full_vec[6 * noz_idx + 1] - u_wp_noz_fv[1])
                rel_x_car = float(full_vec[6 * th_idx + 0] - u_wp_th_fv[0])
                rel_y_car = float(full_vec[6 * th_idx + 1] - u_wp_th_fv[1])
            else:
                rel_x_raw = float(full_vec[6 * noz_idx + 0])
                rel_y_raw = float(full_vec[6 * noz_idx + 1])
                rel_x_car = float(full_vec[6 * th_idx + 0])
                rel_y_car = float(full_vec[6 * th_idx + 1])

            comp_x_noz = (rel_x_raw * pf_x) / (m_gen * lam)
            comp_y_noz = (rel_y_raw * pf_y) / (m_gen * lam)
            comp_x_car = (rel_x_car * pf_x) / (m_gen * lam)
            comp_y_car = (rel_y_car * pf_y) / (m_gen * lam)

            feet_att = (1.0 - u_feet_share) ** 2 if has_feet else 1.0
            rec_x = abs(comp_x_noz) * feet_att
            rec_y = abs(comp_y_noz) * feet_att
            if not is_isolated_panel_drum:
                receptance_x.append((freq_hz, rec_x))
                receptance_y.append((freq_hz, rec_y))
                car_mode_entries.append((freq_hz, comp_x_car, comp_y_car, max(0.0, ke_panel), gantry_max_mode))
                noz_car_entries.append(
                    (freq_hz, comp_x_car, comp_x_noz, comp_y_car, comp_y_noz, th_ke_share)
                )

        candidates.append(
            {
                "lam": lam,
                "full_vec": full_vec,
                "freq_hz": freq_hz,
                "dom_axis": dom_axis,
                "part": part,
                "trans": trans,
                "rot": rot,
                "noz_rel": noz_rel,
                "noz_norm": noz_norm,
                "th_norm": th_norm,
                "noz_th_diff_norm": noz_th_diff_norm,
                "is_isolated_panel_drum": is_isolated_panel_drum,
                "is_bed_z_mode": is_bed_z_mode,
                "u_feet_share": u_feet_share,
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
    rot_node_ids: list[str] = []
    for nid in (
        assembly.toolhead_node_id,
        assembly.toolhead_cog_node_id,
        assembly.nozzle_node_id,
        assembly.bed_node_id,
        "toolhead",
        "toolhead_cog",
        "nozzle_tip",
        "bed_surface",
    ):
        if nid and nid in node_idx and nid not in rot_node_ids:
            rot_node_ids.append(nid)

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
        rot = c["rot"]
        nodal_rotations = {
            nid: [round(float(v), 4) for v in rot[node_idx[nid]]]
            for nid in rot_node_ids
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
            nodal_rotations=nodal_rotations,
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
    selected_candidates: list[dict] = []
    fallback_modes: list[ModeShape] = []
    fallback_cks: list[float] = []
    fallback_candidates: list[dict] = []
    nozzle_twist_freq: float | None = None
    nozzle_twist_fallback_freq: float | None = None

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

        is_extra_feet_mode = len(modes) >= 1 and float(c.get("u_feet_share", 0.0)) > 0.72
        if not c["is_isolated_panel_drum"] and not is_pure_vertical_z and not is_extra_feet_mode:
            is_whip_shape = (
                has_nozzle
                and not c["is_bed_z_mode"]
                and noz_norm > 0.25
                and noz_norm > 1.25 * max(th_norm, 1e-6)
                and (part[0] > 0.50 or part[1] > 0.50)
            )
            if is_whip_shape:
                if nozzle_twist_fallback_freq is None:
                    nozzle_twist_fallback_freq = freq_hz
                if (
                    nozzle_twist_freq is None
                    and float(c["th_ke_share"]) >= 0.15
                    and float(c["noz_th_diff_norm"]) >= 0.25
                    and (float(c["frame_ke_share"]) + float(c["bed_ke_share"])) < 0.55
                ):
                    nozzle_twist_freq = freq_hz

            if len(modes) < k_ret:
                ms, c_k = _build_mode_shape(
                    c, len(modes) + 1, dom_axis, excited_axis, noz_norm, th_norm, part
                )
                modes.append(ms)
                mode_cks.append(c_k)
                selected_candidates.append(c)
        elif len(fallback_modes) < k_ret:
            ms, c_k = _build_mode_shape(
                c, len(fallback_modes) + 1, dom_axis, excited_axis, noz_norm, th_norm, part
            )
            fallback_modes.append(ms)
            fallback_cks.append(c_k)
            fallback_candidates.append(c)

    if not modes:
        modes = fallback_modes
        mode_cks = fallback_cks
        selected_candidates = fallback_candidates

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

    # Exact Castigliano static excess overhang compliance from carriage -> CoG -> nozzle tip,
    # combined with carriage structural frequency via Dunkerley's series theorem (SI units).
    dof_to_free = {int(gdof): fi for fi, gdof in enumerate(free_dofs)}
    static_cache: dict[tuple[int, int], np.ndarray] = {}

    def _solve_unit_load(node_i: int, axis: int) -> np.ndarray | None:
        key = (node_i, axis)
        if key in static_cache:
            return static_cache[key]
        gdof = 6 * node_i + axis
        fi = dof_to_free.get(gdof)
        if fi is None:
            return None
        rhs = np.zeros(n_free, dtype=float)
        rhs[fi] = 1.0
        try:
            if lu_solver is not None:
                u_f = lu_solver.solve(rhs)
            else:
                u_f = np.linalg.solve(K_ff, rhs)
        except (np.linalg.LinAlgError, RuntimeError):
            return None
        u_full = np.zeros(n_dof, dtype=float)
        u_full[free_dofs] = u_f
        static_cache[key] = u_full
        return u_full

    m_cog_eff = float(assembly.nodes[cog_idx].lumped_mass_kg) if cog_idx != th_idx else 0.0
    m_noz_eff = float(assembly.nodes[noz_idx].lumped_mass_kg) if has_nozzle else 0.0
    arch = assembly.spec.architecture

    feet_type = getattr(assembly.spec.frame, "feet_type", "rubber_feet")
    feet_cfg = FRAME_FEET_CATALOG.get(feet_type, {})
    feet_dd = float(feet_cfg.get("damping_delta", 0.0)) if has_feet else 0.0
    pen_x = 0.0
    pen_y = 0.0
    pin_best_x: dict | None = None
    pin_best_y: dict | None = None

    if has_feet:
        (
            struct_x_int,
            struct_y_int,
            ratio_x,
            ratio_y,
            norm_scanned_x,
            norm_scanned_y,
            pin_best_x,
            pin_best_y,
        ) = _solve_internal_pinned_struct(
            assembly=assembly,
            K_ff=K_ff,
            M_ff=M_ff,
            free_dofs=free_dofs,
            dof_to_free=dof_to_free,
            node_idx=node_idx,
            m_floor=m_floor,
            feet_springs=feet_springs,
            feet_cfg=feet_cfg,
            target_scan=target_scan,
            has_panels=has_panels,
            panel_indices=panel_indices,
            non_panel_indices=non_panel_indices,
            gantry_indices=gantry_indices,
            free_panel_mask=free_panel_mask,
            free_th_mask=free_th_mask,
            th_idx=th_idx,
            cog_idx=cog_idx,
            noz_idx=noz_idx,
            bed_idx=bed_idx,
            has_nozzle=has_nozzle,
            dr_noz=dr_noz,
            dr_th=dr_th,
            fx_exc=fx_exc,
            fy_exc=fy_exc,
            m_cog_eff=m_cog_eff,
            m_noz_eff=m_noz_eff,
            fundamental=fundamental,
        )
        pen_x, pen_y = _compute_benchtop_rocking_penalty(
            assembly=assembly,
            M_ff=M_ff,
            free_dofs=free_dofs,
            node_idx=node_idx,
            feet_springs=feet_springs,
            feet_cfg=feet_cfg,
        )
        struct_x = struct_x_int / math.sqrt(1.0 + max(0.0, pen_x))
        struct_y = struct_y_int / math.sqrt(1.0 + max(0.0, pen_y))
        psd_zeta = max(0.025, assembly.frame_damping_ratio - feet_dd)
    else:
        struct_x_car = _geometric_car_freq(
            [(f, ccx, kp, gm) for f, ccx, _, kp, gm in car_mode_entries], fundamental
        )
        struct_y_car = _geometric_car_freq(
            [(f, ccy, kp, gm) for f, _, ccy, kp, gm in car_mode_entries], fundamental
        )
        _, norm_scanned_x = _receptance_structural_freq_and_psd(receptance_x, fundamental)
        _, norm_scanned_y = _receptance_structural_freq_and_psd(receptance_y, fundamental)
        struct_x, ratio_x = _dunkerley_overhang_struct(
            struct_x_car,
            0,
            arch,
            th_idx,
            cog_idx,
            noz_idx,
            bed_idx,
            has_nozzle,
            m_cog_eff,
            m_noz_eff,
            _solve_unit_load,
        )
        struct_y, ratio_y = _dunkerley_overhang_struct(
            struct_y_car,
            1,
            arch,
            th_idx,
            cog_idx,
            noz_idx,
            bed_idx,
            has_nozzle,
            m_cog_eff,
            m_noz_eff,
            _solve_unit_load,
        )
        psd_zeta = assembly.frame_damping_ratio

    if nozzle_twist_freq is None:
        nozzle_twist_freq = (
            nozzle_twist_fallback_freq
            if nozzle_twist_fallback_freq is not None
            else min(struct_x, struct_y)
        )

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
        psd_damping_ratio=psd_zeta,
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
        psd_damping_ratio=psd_zeta,
    )

    limiting = shaper_y if shaper_y.max_accel_mm_s2 <= shaper_x.max_accel_mm_s2 else shaper_x
    other = shaper_x if limiting.axis == "Y" else shaper_y
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

    # Per-mode exact root causes (Fox-Kapoor) & axis compliance shares
    tot_w_x = sum(
        float(c["rec_x"]) / max(float(c["freq_hz"]) ** 2, 1e-12)
        for c in candidates
        if not c["is_isolated_panel_drum"] and float(c["rec_x"]) > 0.0 and float(c["lam"]) > 1.0
    )
    tot_w_y = sum(
        float(c["rec_y"]) / max(float(c["freq_hz"]) ** 2, 1e-12)
        for c in candidates
        if not c["is_isolated_panel_drum"] and float(c["rec_y"]) > 0.0 and float(c["lam"]) > 1.0
    )

    elem_blocks, lumped_blocks = _precompute_modal_element_blocks(assembly, node_idx)
    mode_energy_data: list[tuple[dict[str, dict], float, dict[str, dict], float, float]] = []
    for m, c_m in zip(modes, selected_candidates):
        top_causes, u_groups, u_total, t_groups, t_total = _compute_mode_root_causes(
            full_vec=c_m["full_vec"],
            lam=float(c_m["lam"]),
            freq_hz=m.frequency_hz,
            elem_blocks=elem_blocks,
            lumped_blocks=lumped_blocks,
        )
        m.top_causes = top_causes

        if c_m["is_isolated_panel_drum"] or float(c_m["lam"]) <= 1.0:
            raw_x_pct = 0.0
            raw_y_pct = 0.0
        else:
            f2_safe = max(float(c_m["freq_hz"]) ** 2, 1e-12)
            wx_m = float(c_m["rec_x"]) / f2_safe
            wy_m = float(c_m["rec_y"]) / f2_safe
            raw_x_pct = 100.0 * wx_m / tot_w_x if tot_w_x > 1e-24 else 0.0
            raw_y_pct = 100.0 * wy_m / tot_w_y if tot_w_y > 1e-24 else 0.0

        m.x_axis_share_pct = round(raw_x_pct, 1)
        m.y_axis_share_pct = round(raw_y_pct, 1)
        raw_lim_pct = raw_x_pct if limiting.axis == "X" else raw_y_pct
        m.limiting_axis_share_pct = (
            m.x_axis_share_pct if limiting.axis == "X" else m.y_axis_share_pct
        )
        m.is_limiting_axis_mode = bool(
            m.limiting_axis_share_pct >= 5.0 or m.excited_axis in (limiting.axis, "XY")
        )
        mode_energy_data.append((u_groups, u_total, t_groups, t_total, raw_lim_pct))

    primary_mode: ModeShape | None = None
    primary_u_groups: dict[str, dict] = {}
    primary_u_total = 0.0
    primary_t_groups: dict[str, dict] = {}
    primary_t_total = 0.0
    if modes:
        max_lim_share = max(m.limiting_axis_share_pct for m in modes)
        if max_lim_share > 0.0:
            primary_idx = max(
                range(len(modes)),
                key=lambda idx: (
                    modes[idx].limiting_axis_share_pct,
                    mode_energy_data[idx][4],
                    -modes[idx].mode_number,
                ),
            )
        else:
            exact_matches = [
                idx for idx, m in enumerate(modes) if m.excited_axis == limiting.axis
            ]
            xy_matches = [idx for idx, m in enumerate(modes) if m.excited_axis == "XY"]
            if exact_matches:
                primary_idx = min(exact_matches, key=lambda idx: modes[idx].mode_number)
            elif xy_matches:
                primary_idx = min(xy_matches, key=lambda idx: modes[idx].mode_number)
            else:
                primary_idx = 0
        modes[primary_idx].is_primary_limiting_mode = True
        primary_mode = modes[primary_idx]
        primary_u_groups, primary_u_total, primary_t_groups, primary_t_total, _ = mode_energy_data[
            primary_idx
        ]

    ratio_lim = ratio_x if limiting.axis == "X" else ratio_y
    pen_lim = pen_x if limiting.axis == "X" else pen_y
    feet_rock_frac = (pen_lim / (1.0 + pen_lim)) if (has_feet and pen_lim > 0.0) else 0.0
    summary_u_groups = primary_u_groups
    summary_u_total = primary_u_total
    summary_t_groups = primary_t_groups
    summary_t_total = primary_t_total
    if has_feet:
        pin_best = pin_best_x if limiting.axis == "X" else pin_best_y
        if pin_best is not None:
            pin_elems = [b for b in elem_blocks if b[0] != "frame_feet"]
            base_mass_kg = float(feet_cfg.get("base_mass_kg", 0.0))
            feet_dof_set = {
                6 * node_idx[sp.frame_node_id] for sp in feet_springs if sp.frame_node_id in node_idx
            }
            if base_mass_kg > 0.0 and feet_dof_set:
                dm = base_mass_kg / len(feet_dof_set)
                d_rot = dm * (0.025**2)
                sub_vec = np.array([dm, dm, dm, d_rot, d_rot, d_rot], dtype=float)
                pin_lumped = []
                for gid, lbl, sub, bdof, m6 in lumped_blocks:
                    if bdof in feet_dof_set:
                        m_adj = np.maximum(0.0, m6 - sub_vec)
                        if float(np.max(m_adj)) > 1e-12:
                            pin_lumped.append((gid, lbl, sub, bdof, m_adj))
                    else:
                        pin_lumped.append((gid, lbl, sub, bdof, m6))
            else:
                pin_lumped = lumped_blocks
            _, summary_u_groups, summary_u_total, summary_t_groups, summary_t_total = (
                _compute_mode_root_causes(
                    full_vec=pin_best["full_vec"],
                    lam=float(pin_best["lam"]),
                    freq_hz=float(pin_best["freq_hz"]),
                    elem_blocks=pin_elems,
                    lumped_blocks=pin_lumped,
                )
            )

    limiting_modal_summary = _build_limiting_modal_summary(
        assembly=assembly,
        limiting=limiting,
        other=other,
        primary_mode=primary_mode,
        primary_u_groups=summary_u_groups,
        primary_u_total=summary_u_total,
        primary_t_groups=summary_t_groups,
        primary_t_total=summary_t_total,
        ratio_lim=ratio_lim,
        feet_rock_frac=feet_rock_frac,
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
        panel_drum_freq_hz=resolved_panel_drum,
        frame_damping_ratio=assembly.frame_damping_ratio,
        accel_bottleneck=accel_bottleneck,
        worst_print_mode_number=worst_print_mode_number,
        limiting_modal_summary=limiting_modal_summary,
    )
