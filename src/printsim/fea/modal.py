"""3D Modal Finite Element Analysis (K phi = omega^2 M phi) & Input Shaper Tuning."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import LinearOperator, eigsh, splu

from printsim.fea.beam3d import (
    GlobalSystem,
    assemble_global_system,
    beam_rotation_matrix,
    beam_shear_coefficients,
    local_beam_mass,
    local_beam_stiffness,
    node_rotational_inertia_xyz,
    superelement_element_matrices,
    transformation_12x12,
)
from printsim.fea.belt_micromechanics import cord_crimp_factor_micromechanics
from printsim.fea.belt_nonlinear import (
    NonlinearBeltDriveSpec,
    solve_nonlinear_belt_shaker_response,
)
from printsim.fea.components import (
    belt_damping_ratio,
    belt_flexural_rigidity_nm2,
    belt_operating_pretension_n,
    belt_thermal_tension_delta_n,
    operating_belt,
)
from printsim.fea.klipper_shaper import (
    DEFAULT_SHAPER_DAMPING_RATIO,
    MAX_FREQ_HZ,
    SHAPER_VIBRATION_REDUCTION,
    AxisShaperGraph,
    apply_biquad_notch,
    biquad_notch_params_from_db_q,
    calibrate_axis_shapers,
    estimate_psd_damping_ratio,
    estimate_shaper,
    get_shaper_pulses,
    synthesize_axis_psd,
)
from printsim.fea.shell_mitc4 import fastener_spring_matrices, mitc4_shell_matrices
from printsim.fea.solver import classify_part_group
from printsim.geometry.builder import AssemblyModel
from printsim.kinematics.solvers import (
    compute_bracket_stiffness_n_per_m,
    compute_drive_stiffness_details_n_per_m,
    compute_drive_stiffness_n_per_m,
)
from printsim.schema import (
    GANTRY_BRACKET_CATALOG,
    get_material,
    resolve_belt,
    resolve_frame_feet,
)

_G_ACC = 9.80665  # standard gravity (m/s^2)

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
    toolhead_part_flex: list[dict] = field(default_factory=list)
    toolhead_mode_fields: dict[str, dict] = field(default_factory=dict)


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
    adxl_vs_nozzle_diagnostic: dict | None = None
    mounting_dynamics: dict | None = None
    belt_dynamics: dict | None = None

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
            "adxl_vs_nozzle_diagnostic": self.adxl_vs_nozzle_diagnostic,
            "mounting_dynamics": self.mounting_dynamics,
            "belt_dynamics": self.belt_dynamics,
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
                    "toolhead_part_flex": m.toolhead_part_flex,
                    "toolhead_mode_fields": m.toolhead_mode_fields,
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


def _foc_motor_stiffness_multiplier(assembly: AssemblyModel) -> float:
    """Closed-loop FOC motor-loop stiffness multiplier from pin_rotor_ff and AWD sync."""
    spec = assembly.spec
    kin = spec.kinematics
    if kin.drive_type not in ("foc_servo", "closed_loop_stepper"):
        return 1.0
    mult = 3.5 if kin.pin_rotor_ff else 1.0
    if spec.architecture == "corexy" and spec.gantry.drive_mode == "awd":
        if kin.awd_sync_mode == "position_equalizer":
            mult *= 1.12
        elif kin.awd_sync_mode == "torque_mirror":
            mult *= 1.06
    return mult


def _awd_sync_damping_delta(assembly: AssemblyModel) -> float:
    """Belt damping delta from closed-loop/FOC AWD sync mode."""
    spec = assembly.spec
    kin = spec.kinematics
    if kin.drive_type not in ("foc_servo", "closed_loop_stepper"):
        return 0.0
    if spec.architecture != "corexy" or spec.gantry.drive_mode != "awd":
        return 0.0
    if kin.awd_sync_mode == "position_equalizer":
        excess_bw = max(0.0, float(kin.awd_equalizer_bw_hz) - 5.0)
        return max(0.004, 0.015 - 0.0012 * excess_bw)
    if kin.awd_sync_mode == "torque_mirror":
        return 0.008
    return 0.0


def _reflected_rotor_mass_kg(assembly: AssemblyModel) -> float:
    """Reflected stepper rotor inertia J_rot / r_p^2 projected onto one Cartesian axis (kg)."""
    spec = assembly.spec
    r_p = max(float(spec.gantry.belt.pitch_radius_m), 1e-6)
    m_rot_1 = max(float(spec.kinematics.xy_stepper.rotor_inertia_kg_m2), 0.0) / (r_p**2)
    if spec.architecture == "corexy":
        return (4.0 if spec.gantry.drive_mode == "awd" else 2.0) * m_rot_1
    if spec.architecture == "delta":
        from printsim.kinematics.solvers import _delta_center_j_norm

        return _delta_center_j_norm(spec) * m_rot_1
    return m_rot_1


def _solve_coupled_drive_2dof(
    k_belt_n_m: float,
    k_motor_raw_n_m: float,
    k_bracket_n_m: float,
    m_car_eff_kg: float,
    m_rot_kg: float,
) -> dict[str, float]:
    """Exact closed-form 2-DOF eigenproblem for `ground --k_motor--> [m_rot] --(k_belt||series k_bkt)--> [m_car]`.

    Replaces the 1-DOF Rayleigh rigid-body approximation (which underestimates drive frequency
    by up to 11.5% on light AWD carriages where m_rot > m_car) with exact eigenvalues (w1, w2)
    and exact mode-1 modal strain-energy compliance shares for belt, motor, and bracket.
    """
    kb = max(float(k_belt_n_m), 1.0)
    km = max(float(k_motor_raw_n_m), 1.0)
    kbkt = max(float(k_bracket_n_m), 1.0)
    mc = max(float(m_car_eff_kg), 0.02)
    mr = max(float(m_rot_kg), 0.0)
    k_bb = 1.0 / (1.0 / kb + 1.0 / kbkt)

    if mr <= 1e-12:
        c_drv = 1.0 / kb + 1.0 / km + 1.0 / kbkt
        w1_sq = 1.0 / (mc * c_drv)
        return {
            "w1_sq": w1_sq,
            "w2_sq": 1e18,
            "c_drive_n_inv": c_drv,
            "k_drive_eff_n_m": 1.0 / c_drv,
            "u_belt_frac": (1.0 / kb) / c_drv,
            "u_motor_frac": (1.0 / km) / c_drv,
            "u_bracket_frac": (1.0 / kbkt) / c_drv,
            "mode2_car_part": 0.0,
        }

    # det([[km + k_bb - w^2*mr, -k_bb], [-k_bb, k_bb - w^2*mc]]) = 0
    # => w^4 - b_q * w^2 + c_q = 0
    b_q = (km + k_bb) / mr + k_bb / mc
    c_q = (km * k_bb) / (mr * mc)
    disc = math.sqrt(max(b_q * b_q - 4.0 * c_q, 0.0))
    w1_sq = max(0.5 * (b_q - disc), 1e-6)
    w2_sq = max(0.5 * (b_q + disc), w1_sq + 1.0)

    c_drv = 1.0 / (mc * w1_sq)
    # Mode-1 eigenvector normalized to carriage amplitude = 1: phi_1 = [alpha_1, 1]^T
    alpha_1 = k_bb / max(km + k_bb - w1_sq * mr, 1e-9)
    alpha_1 = max(0.0, min(1.0, alpha_1))
    u_mot = 0.5 * km * (alpha_1**2)
    u_bb = 0.5 * k_bb * ((1.0 - alpha_1) ** 2)
    u_belt = u_bb * (k_bb / kb)
    u_bkt = u_bb * (k_bb / kbkt)
    u_tot = max(u_mot + u_belt + u_bkt, 1e-18)

    # Mode-2 relative carriage acceleration-receptance weight vs mode-1 (for upper rotor-belt PSD bump)
    # Force applied at carriage -> modal residue ratio R2/R1 = (m_gen_1 * w1^2) / (m_gen_2 * w2^2) * ...
    # With phi_k = [alpha_k, 1]^T, m_gen_k = mr * alpha_k^2 + mc:
    alpha_2 = k_bb / min(km + k_bb - w2_sq * mr, -1e-9)
    m_gen_1 = mr * (alpha_1**2) + mc
    m_gen_2 = mr * (alpha_2**2) + mc
    # Static receptance residue ratio = (1 / (m_gen_2 * w2^2)) / (1 / (m_gen_1 * w1^2))
    res_ratio = (m_gen_1 * w1_sq) / max(m_gen_2 * w2_sq, 1e-12)
    mode2_part = math.sqrt(max(0.0, min(1.0, res_ratio)))

    return {
        "w1_sq": w1_sq,
        "w2_sq": w2_sq,
        "c_drive_n_inv": c_drv,
        "k_drive_eff_n_m": 1.0 / c_drv,
        "u_belt_frac": u_belt / u_tot,
        "u_motor_frac": u_mot / u_tot,
        "u_bracket_frac": u_bkt / u_tot,
        "mode2_car_part": mode2_part,
    }


def _drive_axis_stiffness_n_per_m(
    assembly: AssemblyModel, axis: str
) -> tuple[float, float, float, float, float]:
    """Return ``(k_belt, k_motor_eff, k_bracket, k_drive, m_moving_eff_kg)`` in strict SI.

    Uses the exact 2-DOF `[m_rot, m_car + m_belt_eff]` coupled drive eigenproblem rather than
    the 1-DOF Rayleigh approximation, and includes Rayleigh moving-belt inertia `(1/3)*mu*L`.
    """
    details = compute_drive_stiffness_details_n_per_m(assembly, axis)
    k_belt = float(details["k_belt_subsystem_n_m"])
    k_motor_raw = float(details["k_motor_axis_n_m"])
    k_bracket = float(details["k_bracket_n_m"])
    k_mult = _foc_motor_stiffness_multiplier(assembly)
    if k_mult != 1.0:
        k_motor_raw = k_motor_raw * k_mult
    mass_base = max(assembly.x_moving_mass_kg if axis == "X" else assembly.y_moving_mass_kg, 0.05)
    m_belt_eff = float(details.get("belt_moving_mass_kg", 0.0))
    mass = mass_base + m_belt_eff
    m_rot = _reflected_rotor_mass_kg(assembly)
    sol = _solve_coupled_drive_2dof(k_belt, k_motor_raw, k_bracket, mass, m_rot)
    # Expose rigid-belt rotor-adjusted motor stiffness for standalone motor_freq_hz readout
    k_motor_eff = k_motor_raw * (mass / max(mass + m_rot, 1e-6))
    return k_belt, k_motor_eff, k_bracket, float(sol["k_drive_eff_n_m"]), mass


def _drive_axis_resonances_hz(assembly: AssemblyModel, axis: str) -> tuple[float, float, float, float]:
    """Compute (belt_freq_hz, motor_freq_hz, bracket_freq_hz, drive_freq_hz) for X or Y."""
    k_belt, k_motor_axis, k_bracket, k_drive, mass = _drive_axis_stiffness_n_per_m(assembly, axis)
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
    c_struct_m_per_n: float,
    k_belt_n_m: float,
    k_motor_n_m: float,
    k_bracket_n_m: float,
    c_drive_override: float | None = None,
    drive_energy_fracs: tuple[float, float, float] | None = None,
) -> tuple[dict[str, float], str]:
    """Exact modal strain-energy compliance shares summing to 100.0% (B-01 + exact 2-DOF)."""
    c_frame = max(c_struct_m_per_n, 0.0)
    if c_drive_override is not None and drive_energy_fracs is not None:
        c_drv = max(float(c_drive_override), 1e-18)
        fb, fm, fbkt = drive_energy_fracs
        c_belt = c_drv * max(fb, 0.0)
        c_motor = c_drv * max(fm, 0.0)
        c_bracket = c_drv * max(fbkt, 0.0)
    else:
        c_belt = 1.0 / max(k_belt_n_m, 1.0)
        c_motor = 1.0 / max(k_motor_n_m, 1.0)
        c_bracket = 1.0 / max(k_bracket_n_m, 1.0)
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
    c_struct_m_per_n: float,
    m_moving_kg: float,
    k_belt_n_m: float,
    k_motor_n_m: float,
    k_bracket_n_m: float,
    k_drive_n_m: float,
    configured_accel: float,
    shaper_choice: str,
    scv_mm_s: float = 5.0,
    scanned_modes: list[tuple[float, float]] | None = None,
    psd_damping_ratio: float | None = None,
    assembly: AssemblyModel | None = None,
    fast_diagnostic: bool = False,
) -> tuple[InputShaperRecommendation, AxisShaperGraph]:
    # Exact 2-DOF drive coupling + first-principles nonlinear belt shaker dynamics + strain-energy damping
    c_struct = max(c_struct_m_per_n, 0.0)
    k_belt_eff = max(float(k_belt_n_m), 1.0)
    k_bracket_eff = max(float(k_bracket_n_m), 1.0)
    m_eff = max(float(m_moving_kg), 0.05)
    nl_zeta_delta = 0.0
    extra_belt_modes: list[tuple[float, float]] = []

    zeta_frame = 0.055 if psd_damping_ratio is None else float(psd_damping_ratio)
    is_foc = (
        assembly is not None
        and assembly.spec.kinematics.drive_type in ("foc_servo", "closed_loop_stepper")
    )

    if assembly is not None:
        spec = assembly.spec
        belt = operating_belt(spec)
        t_ch = float(spec.frame.chamber_temp_c)
        belt_cfg = resolve_belt(spec)
        ea_override_n = float(belt_cfg["stiffness_ea_n"])
        belt_mu_kg_m = max(float(belt_cfg["linear_mass_kg_m"]), 1e-4)

        details = compute_drive_stiffness_details_n_per_m(assembly, axis)
        k_motor_raw = float(details["k_motor_axis_n_m"]) * _foc_motor_stiffness_multiplier(assembly)
        m_rot = _reflected_rotor_mass_kg(assembly)
        sol = _solve_coupled_drive_2dof(k_belt_eff, k_motor_raw, k_bracket_eff, m_eff, m_rot)
        c_drv = float(sol["c_drive_n_inv"])

        # First-principles coupled longitudinal-transverse nonlinear shaker dynamics
        # (unilateral tension-only branches with Costello tangent EA(T) + Beikmann-Perkins-Ulsoy
        # 1:1 sag/runout & 2:1 parametric Mathieu-Duffing transverse whip).
        if spec.architecture == "corexy":
            n_loops = 2.0
        elif spec.architecture == "delta":
            from printsim.kinematics.solvers import _delta_center_j_norm

            n_loops = max(float(_delta_center_j_norm(spec)), 1.0)
        else:
            n_loops = 1.0
        slack_share = float(details.get("slack_load_share", 0.5))
        t0 = max(float(belt.pretension_n), 0.25)
        l_free = max(float(details.get("longest_free_span_m", 0.35)), 0.10)
        ei_belt = belt_flexural_rigidity_nm2(belt, chamber_temp_c=t_ch)
        zeta_belt_lin = belt_damping_ratio(belt, chamber_temp_c=t_ch, slack_margin_ratio=1.0)
        # Pure viscoelastic material + thermal damping (Ling et al. 2020 DMA tan delta ~ 0.06-0.08)
        # without the empirical low-tension proxy, so NonlinearBeltDriveSpec captures low-tension
        # slackening and transverse-whip broadening from first principles without double-damping.
        ret_t = belt.tooth_modulus_retention(t_ch)
        z_belt_base_mat = (
            0.052 if belt.core == "aramid" else (0.036 if belt.body_material == "tpu" else 0.044)
        )
        zeta_belt_mat = z_belt_base_mat + 0.040 * max(0.0, 1.0 - ret_t)
        zeta_motor_lin = 0.080 if is_foc else 0.050

        profile_family = str(belt.catalog_entry.get("profile_family", "2gt"))
        width_mm = float(belt.width_mm)
        core = belt.core
        e_seat = max(float(belt.elastomer_seating_factor), 0.1)

        def _belt_ea_fn(tension_n: float) -> float:
            c_t = cord_crimp_factor_micromechanics(
                max(tension_n, 0.05),
                profile_family=profile_family,
                width_mm=width_mm,
                core=core,
                matrix_modulus_retention=ret_t,
                elastomer_modulus_scale=e_seat,
            )
            return max(ea_override_n * c_t * e_seat, 100.0)

        nl_drive = NonlinearBeltDriveSpec(
            m_car_eff_kg=m_eff,
            m_rot_kg=m_rot,
            k_motor_raw_n_m=k_motor_raw,
            k_bracket_n_m=k_bracket_eff,
            c_struct_m_per_n=c_struct,
            pretension_n=t0,
            k_belt_linear_n_m=k_belt_eff,
            n_loops=n_loops,
            slack_load_share=slack_share,
            ea_fn=_belt_ea_fn,
            longest_free_span_m=l_free,
            belt_linear_mass_kg_m=belt_mu_kg_m,
            belt_ei_nm2=ei_belt,
            transverse_damping_ratio=max(0.025, 0.75 * zeta_belt_mat),
            pulley_eccentricity_m=0.0,
            zeta_belt_base=zeta_belt_mat,
            zeta_motor=zeta_motor_lin,
            zeta_car=0.010,
        )
        nl_res = solve_nonlinear_belt_shaker_response(
            nl_drive,
            num_freq_bins=72 if fast_diagnostic else 108,
        )
        nl_zeta_delta = float(nl_res.zeta_nl_delta)
        if nl_res.slack_duty_cycle > 0.0 or nl_res.stiffness_ratio < 0.995:
            k_belt_eff = max(float(nl_res.k_belt_eff_n_m), 1.0)
            sol = _solve_coupled_drive_2dof(k_belt_eff, k_motor_raw, k_bracket_eff, m_eff, m_rot)
            c_drv = float(sol["c_drive_n_inv"])
            belt_hz = math.sqrt(k_belt_eff / m_eff) / (2.0 * math.pi)
            drive_hz = math.sqrt(float(sol["w1_sq"])) / (2.0 * math.pi)

        c_total = c_struct + c_drv
        f_eff = 1.0 / (2.0 * math.pi * math.sqrt(max(m_eff * c_total, 1e-15)))
        shares_pct, primary_bn = _compute_compliance_shares(
            c_struct,
            k_belt_eff,
            k_motor_n_m,
            k_bracket_eff,
            c_drive_override=c_drv,
            drive_energy_fracs=(
                float(sol["u_belt_frac"]),
                float(sol["u_motor_frac"]),
                float(sol["u_bracket_frac"]),
            ),
        )

        # Upper 2-DOF rotor-belt antiphase mode if inside the shaper band with meaningful carriage weight
        f2_hz = math.sqrt(float(sol["w2_sq"])) / (2.0 * math.pi)
        m2_part = float(sol["mode2_car_part"])
        if 10.0 <= f2_hz <= 215.0 and abs(f2_hz - f_eff) >= 3.0 and m2_part > 0.15:
            extra_belt_modes.append((f2_hz, m2_part))

        # Add first-principles nonlinear softening-shoulder and Beikmann-Perkins-Ulsoy whip sidebands
        for mf_nl, part_nl in nl_res.extra_modes:
            if 10.0 <= mf_nl <= 215.0 and abs(mf_nl - f_eff) >= 3.0 and part_nl > 0.15:
                extra_belt_modes.append((float(mf_nl), float(part_nl)))

        # During deep intermittent slack-branch unloading (slack_duty_cycle > 0.28 with meaningful
        # cycle-averaged stiffness loss), the belt spends duty fraction D in the unloaded branch
        # stiffness k_belt * (1 - slack_share), radiating low-side spectral power at the exact
        # slack-branch frequency f_slack (matching the time-domain ODE Welch PSD ZV -> MZV transition).
        duty = float(nl_res.slack_duty_cycle)
        s_belt_frac = shares_pct["belt"] / 100.0
        if duty > 0.28 and nl_res.stiffness_ratio < 0.97 and s_belt_frac >= 0.20:
            c_slack_ratio = s_belt_frac * (slack_share / max(1.0 - slack_share, 0.15))
            f_slack = max(10.0, f_eff / math.sqrt(1.0 + c_slack_ratio))
            # Convert duty-cycle PSD fraction into synthesize_axis_psd participation space (0.45 * p^2)
            p_slack = min(1.15, math.sqrt((1.5 * duty) / 0.45))
            if abs(f_slack - f_eff) >= 3.0 and p_slack > 0.15:
                extra_belt_modes.append((f_slack, p_slack))
    else:
        c_total = c_struct + 1.0 / max(k_drive_n_m, 1.0)
        f_eff = 1.0 / (2.0 * math.pi * math.sqrt(max(m_eff * c_total, 1e-15)))
        shares_pct, primary_bn = _compute_compliance_shares(
            c_struct, k_belt_eff, k_motor_n_m, k_bracket_eff
        )

    secondary_modes = [
        (mf, part_val)
        for mf, part_val in list(scanned_modes or []) + extra_belt_modes
        if abs(mf - f_eff) >= 3.0 and 10.0 <= mf <= 215.0 and part_val > 0.15
    ]

    if assembly is not None:
        # Ross-Ungar-Kerwin modal strain-energy weighted linear damping + first-principles nonlinear broadening
        bkt_mat = get_material(assembly.spec.gantry.bracket_material)
        if bkt_mat.is_printed_polymer:
            zeta_bracket = 0.044
        elif bkt_mat.is_slm_metal:
            zeta_bracket = 0.024
        else:
            zeta_bracket = 0.020
        zeta_lin = (
            (shares_pct["frame"] / 100.0) * zeta_frame
            + (shares_pct["belt"] / 100.0) * zeta_belt_lin
            + (shares_pct["motor"] / 100.0) * zeta_motor_lin
            + (shares_pct["bracket"] / 100.0) * zeta_bracket
        )
        zeta_base = float(min(0.22, max(0.025, zeta_lin + nl_zeta_delta)))
    else:
        zeta_base = zeta_frame

    delta_zeta_awd = _awd_sync_damping_delta(assembly) if assembly is not None else 0.0
    delta_zeta_active = (
        0.07
        if (is_foc and assembly is not None and assembly.spec.kinematics.toolhead_imu_active_damping)
        else 0.0
    )
    zeta_eff = float(min(0.25, zeta_base + delta_zeta_awd + delta_zeta_active))
    shaper_damping = float(min(0.25, DEFAULT_SHAPER_DAMPING_RATIO + delta_zeta_awd + delta_zeta_active))

    kin = assembly.spec.kinematics if assembly is not None else None
    notch_hz = kin.notch_filter_hz if (kin is not None and kin.notch_filter_hz is not None and kin.notch_filter_hz > 0.0) else None
    is_flatness_corexy = (
        assembly is not None
        and assembly.spec.architecture == "corexy"
        and shaper_choice.strip().lower() == "smooth_flatness"
    )
    # Only use custom PSD path when active skyhook damping, notch filtering, or CoreXY
    # workspace flatness blending requires raw-amplitude PSD scaling / unnormalized fit.
    # AWD sync delta_zeta_awd flows through zeta_eff and shaper_damping cleanly without
    # disabling normalize_psd (which would enable tie_w=0.02 and trigger a ZV->MZV cliff).
    need_custom_psd = (delta_zeta_active > 0.0) or (notch_hz is not None) or is_flatness_corexy

    corner_detune_freqs: tuple[float, float] | None = None
    if not need_custom_psd:
        graph = calibrate_axis_shapers(
            axis=axis,
            primary_freq_hz=f_eff,
            secondary_modes=secondary_modes,
            scv=scv_mm_s,
            shaper_choice=shaper_choice,
            damping_ratio=shaper_damping,
            psd_damping_ratio=zeta_eff,
            psd_damping_secondary=zeta_frame,
            fast_diagnostic=fast_diagnostic,
        )
    else:
        freqs, psd_base = synthesize_axis_psd(
            axis=axis,
            primary_freq_hz=f_eff,
            secondary_modes=secondary_modes,
            damping_primary=zeta_base,
            damping_secondary=zeta_base,
        )
        _, psd = synthesize_axis_psd(
            axis=axis,
            primary_freq_hz=f_eff,
            secondary_modes=secondary_modes,
            damping_primary=zeta_eff,
            damping_secondary=zeta_eff,
        )

        # Measure damping ratio on fine pre-notch grid so notch dip does not distort half-power BW
        f_p = max(float(f_eff), 5.0)
        fine_lo = max(2.0, f_p * 0.5)
        fine_hi = min(MAX_FREQ_HZ, max(f_p * 1.6, f_p + 12.0))
        fine_freqs = np.arange(fine_lo, fine_hi + 0.05, 0.15, dtype=float)
        _, fine_psd = synthesize_axis_psd(
            axis=axis,
            primary_freq_hz=f_eff,
            secondary_modes=secondary_modes,
            damping_primary=zeta_eff,
            damping_secondary=zeta_eff,
            freqs_hz=fine_freqs,
        )
        measured_zeta_pre_notch = estimate_psd_damping_ratio(fine_freqs, fine_psd)

        if is_flatness_corexy and assembly is not None and kin is not None:
            is_awd = assembly.spec.gantry.drive_mode == "awd"
            base_spread = 0.06 if is_awd else 0.18
            spread = base_spread * (0.15 if kin.lpv_workspace_scheduling else 1.0)
            f_lo = max(10.0, f_eff * (1.0 - 0.55 * spread))
            f_hi = min(215.0, f_eff * (1.0 + 0.45 * spread))
            corner_detune_freqs = (f_lo, f_hi)
            _, psd_lo = synthesize_axis_psd(
                axis=axis,
                primary_freq_hz=f_lo,
                secondary_modes=secondary_modes,
                damping_primary=zeta_eff,
                damping_secondary=zeta_eff,
            )
            _, psd_hi = synthesize_axis_psd(
                axis=axis,
                primary_freq_hz=f_hi,
                secondary_modes=secondary_modes,
                damping_primary=zeta_eff,
                damping_secondary=zeta_eff,
            )
            psd = 0.50 * psd + 0.25 * psd_lo + 0.25 * psd_hi
            pk_blend = float(np.max(psd))
            if pk_blend > 0.0:
                psd *= 100.0 / pk_blend

        # Physical resonance PSD flattening from active skyhook damping (|H|^2 ~ 1 / zeta^2)
        if delta_zeta_active > 0.0 and zeta_base > 0.0:
            psd *= (zeta_base / zeta_eff) ** 2

        # TMC4671 BIQUAD_T depth-limited notch carving out resonance peak on Klipper PSD graph
        if notch_hz is not None and kin is not None:
            zz, zp, _ = biquad_notch_params_from_db_q(kin.notch_depth_db, kin.notch_q)
            psd *= apply_biquad_notch(freqs, notch_hz, zeta_zero=zz, zeta_pole=zp) ** 2

        normalize_flag = not ((delta_zeta_active > 0.0) or (notch_hz is not None))
        ref_v_th = 100.0 / SHAPER_VIBRATION_REDUCTION
        ref_all_vib = max(float(np.sum(np.maximum(psd_base - ref_v_th, 0.0))), 1e-12)
        graph = calibrate_axis_shapers(
            axis=axis,
            primary_freq_hz=f_eff,
            secondary_modes=secondary_modes,
            scv=scv_mm_s,
            shaper_choice=shaper_choice,
            damping_ratio=shaper_damping,
            freqs_hz=freqs,
            raw_psd=psd,
            psd_damping_ratio=zeta_eff,
            normalize_psd=normalize_flag,
            ref_peak=100.0,
            ref_all_vibrations=ref_all_vib,
            measured_damping_override=measured_zeta_pre_notch,
            fast_diagnostic=fast_diagnostic,
        )

    shaper = graph.recommended_shaper
    max_accel = graph.recommended_max_accel_mm_s2

    if (
        is_flatness_corexy
        and kin is not None
        and not kin.lpv_workspace_scheduling
        and corner_detune_freqs is not None
        and shaper == "SMOOTH_FLATNESS"
    ):
        sh_flat = get_shaper_pulses("SMOOTH_FLATNESS", graph.recommended_freq_hz, damping_ratio=shaper_damping)
        v_corner_pct = float(
            np.max(estimate_shaper(sh_flat, list(corner_detune_freqs), test_damping_ratio=shaper_damping))
        ) * 100.0
        derate = max(0.60, 1.0 - max(0.0, v_corner_pct - 5.0) * 0.02)
        max_accel = float(round((max_accel * derate) / 100.0) * 100.0)
        graph.recommended_max_accel_mm_s2 = max_accel

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
    car_mat = spec.toolhead.effective_carriage_mount_material
    body_mat = spec.toolhead.effective_body_material
    mat_label = f"'{body_mat}'" if car_mat == body_mat else f"'{car_mat}' carriage / '{body_mat}' body"
    if spec.architecture == "delta":
        return (
            f"Main limit: effector mount compliance ({mat_label}), "
            "nozzle/CoG lever arm overhang, and effector moving mass."
        )
    return (
        f"Main limit: toolhead mount stiffness ({mat_label}), "
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
    if bay == "y_carriage" or mid.startswith("y_car_bearing_"):
        return "y_rail_carriages", "Y-Rail Carriage Bearings (Hertzian)"
    if bay in ("bed_z_drive", "z_guide") or mid.startswith(("z_bed_drive_", "z_rod_brg_")):
        return "bed_support", "Bed Support & Z Arms"
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
    system: GlobalSystem | None = None,
) -> tuple[
    list[tuple[str, str, str, list[int], np.ndarray, np.ndarray | None]],
    list[tuple[str, str, str, int, np.ndarray]],
]:
    """Precompute element (Ke, Me) and lumped-mass blocks once per solve_modal_fea call.

    With a shared ``system`` the element matrices already computed during global assembly are
    reused (identical values) instead of being re-derived per element.
    """
    elem_blocks: list[tuple[str, str, str, list[int], np.ndarray, np.ndarray | None]] = []

    for m_i, mem in enumerate(assembly.members):
        if system is not None:
            dofs, ke, me, _ = system.member_blocks[m_i]
            gid, label = _classify_modal_group(mem.id, mem.role)
            elem_blocks.append((gid, label, _role_to_subsystem(mem.role), dofs, ke, me))
            continue
        i0 = node_idx[mem.start_id]
        i1 = node_idx[mem.end_id]
        p0 = assembly.nodes[i0].position_m
        p1 = assembly.nodes[i1].position_m
        r_mat, length = beam_rotation_matrix(p0, p1)
        t_mat = transformation_12x12(r_mat)
        bend_scale = 0.08 if mem.is_truss else 1.0
        kappa_y, kappa_z = beam_shear_coefficients(mem)
        k_loc = local_beam_stiffness(
            e_pa=mem.material.youngs_modulus_pa,
            g_pa=mem.material.effective_shear_modulus_pa,
            area=mem.profile.area_m2,
            iy=mem.profile.iy_m4,
            iz=mem.profile.iz_m4,
            j_torsion=mem.profile.j_m4,
            length=length,
            bending_scale=bend_scale,
            kappa_shear_y=kappa_y,
            kappa_shear_z=kappa_z,
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

    for sh_i, sh in enumerate(assembly.shell_elements):
        if system is not None:
            dofs, k_sh, m_sh = system.shell_blocks[sh_i]
        else:
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

    for sp_i, sp in enumerate(assembly.fastener_springs):
        if system is not None:
            dofs, k_sp = system.spring_blocks[sp_i]
        else:
            i_s = node_idx[sp.shell_node_id]
            i_f = node_idx[sp.frame_node_id]
            k_sp = fastener_spring_matrices(
                k_shear_n_m=sp.k_shear_n_m,
                k_normal_n_m=sp.k_normal_n_m,
                k_rot_nm_rad=sp.k_rot_nm_rad,
                normal_vec=np.asarray(sp.normal_vec, dtype=float),
                offset_vec=np.asarray(sp.offset_m, dtype=float),
                k_diag_6dof=sp.k_diag_6dof,
            )
            dofs = [6 * i_s + d for d in range(6)] + [6 * i_f + d for d in range(6)]
        bay_raw = sp.bay
        bay_prefix = _canonical_bay_key(bay_raw)
        if bay_raw not in ("bed_mount", "bed_z_drive", "z_guide", "toolhead", "feet", "z_joint", "y_carriage") and bay_prefix in (
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
                if bay_raw in ("bed_mount", "bed_z_drive", "z_guide")
                else (
                    "toolhead"
                    if bay_raw == "toolhead"
                    else ("gantry" if bay_raw == "z_joint" else "frame")
                )
            )
            gid, label = _classify_modal_group(sp.id, sp_role, bay_raw)
            subsys = _role_to_subsystem(sp_role)
        elem_blocks.append((gid, label, subsys, dofs, k_sp, None))

    internal_dof_cursor = 6 * len(assembly.nodes)
    for se_i, se in enumerate(getattr(assembly, "superelements", ())):
        if system is not None:
            dofs_se, ke_se, me_se = system.superelement_blocks[se_i]
        else:
            dofs_se, ke_se, me_se = superelement_element_matrices(
                se, assembly, node_idx, internal_dof_cursor
            )
        internal_dof_cursor += max(0, int(getattr(se, "n_internal_modes", max(0, ke_se.shape[0] - 12))))
        gid = getattr(se, "part_group_id", "") or "toolhead_mount"
        label = getattr(se, "part_group_label", "") or "Toolhead & Nozzle Mount"
        subsys = _role_to_subsystem(getattr(se, "role", "toolhead"))
        elem_blocks.append((gid, label, subsys, dofs_se, ke_se, me_se))

    lumped_blocks: list[tuple[str, str, str, int, np.ndarray]] = []
    for i, node in enumerate(assembly.nodes):
        if node.lumped_mass_kg <= 0.0 or node.fixed:
            continue
        m_val = float(node.lumped_mass_kg)
        ixx, iyy, izz = node_rotational_inertia_xyz(node, assembly)
        m_diag_6 = np.array([m_val, m_val, m_val, ixx, iyy, izz], dtype=float)
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
        # C-08: Frozen-mode Rayleigh quotient frequency gain for +50% stiffness
        est_hz = round(
            max(0.0, freq_hz * (math.sqrt(1.0 + 0.5 * (u_val / u_norm_denom)) - 1.0)), 2
        )
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
    k_motor_axis = float(details["k_motor_axis_n_m"]) * _foc_motor_stiffness_multiplier(assembly)  # type: ignore[operator]
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
    # B-01: In compliance domain c_total = c_struct + 1/k_drive, reducing moving mass by 25% scales drive
    # compliance 100% and structural compliance by t_mov_share on the primary mode.
    s_struct_frac = min(1.0, max(0.0, s_struct / 100.0))
    s_drive_frac = max(0.0, 1.0 - s_struct_frac)
    f_eff_new_mass = f_eff / math.sqrt(
        max(1.0 - 0.25 * (s_drive_frac + s_struct_frac * t_mov_share), 0.75)
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
    system: GlobalSystem | None = None,
) -> tuple[np.ndarray, np.ndarray, object | None]:
    """Solve K_ff phi = lam M_ff phi with sparse shift-invert or dense Cholesky eigh.

    With a shared ``system`` the sparse ``K_ff`` factorization is reused from the static solve
    and the M-orthonormalized eigenpairs are published under ``("eigen", method, k)`` so the
    sensitivity solver can skip an identical second eigensolve.
    """
    n_free = K_ff.shape[0]
    k_scan = min(target_scan, n_free - 2 if n_free > 250 else n_free)
    used_sparse = False
    lu_solver = None
    if n_free > 250 and 1 <= k_scan < n_free - 1:
        try:
            k_csc = system.reduced_stiffness_csc() if system is not None else csc_matrix(K_ff)
            m_csc = csc_matrix(M_ff)
            lu = system.reduced_stiffness_lu() if system is not None else splu(k_csc)
            lu_solver = lu
            op = LinearOperator(K_ff.shape, matvec=lu.solve, dtype=float)
            v0 = np.random.default_rng(0).standard_normal(n_free)
            eigvals, eigvecs = eigsh(
                k_csc,
                k=k_scan,
                M=m_csc,
                sigma=0.0,
                which="LM",
                OPinv=op,
                v0=v0,
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
            v = v / math.sqrt(m_gen_j)
        max_abs = float(np.max(np.abs(v)))
        if max_abs > 0.0:
            lead_idx = int(np.flatnonzero(np.abs(v) >= 0.99 * max_abs)[0])
            if v[lead_idx] < 0.0:
                v = -v
        eigvecs[:, j] = v

    if system is not None:
        system.store(
            ("eigen", "sparse" if used_sparse else "dense", int(k_scan)),
            (eigvals.copy(), eigvecs.copy()),
        )
    return eigvals, eigvecs, lu_solver


def _receptance_structural_freq_and_psd(
    rmodes: list[tuple[float, float]], fallback_hz: float
) -> tuple[float, list[tuple[float, float]]]:
    """Compute acceleration-weighted structural frequency & secondary PSD modes (B-02).

    Under Klipper input shaping, the max acceleration penalty of a resonant mode at
    frequency f_k with signed nozzle-to-bed modal compliance residue R_k scales as
    w_k = R_k / f_k^2. Secondary PSD amplitudes are normalized by max(R_k) across
    significant modes so soft low-frequency feet-bounce modes with small residues
    cannot saturate internal structural modes to 1.0.
    """
    weighted = [(f, abs(r), abs(r) / (f * f)) for f, r in rmodes if f >= 5.0 and abs(r) > 1e-24]
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

    dom_f, _, _ = max(sig, key=lambda item: item[2])
    r_max = max(r for _, r, _ in sig)
    sec_modes = [
        (f, min(1.0, math.sqrt(r / max(r_max, 1e-18))))
        for f, r, _ in sig
        if abs(f - dom_f) >= 3.0 and abs(f - eff_hz) >= 3.0
    ]
    return eff_hz, sec_modes


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
        (ke_panel > 0.55 and noz_norm_all < 0.12 and ke_th_total < 0.055)
        or (ke_panel > 0.58 and noz_norm_all < 0.25 and ke_th_total < 0.04)
        or (ke_panel > 0.68 and gantry_max < 0.35 and noz_norm_all < 0.28)
        or (ke_panel > 0.80 and noz_norm_all < 0.28 and ke_th_total < 0.06)
        or (ke_panel > 0.85 and ke_th_total < 0.05)
        or (np_max < 0.25 and gantry_max < 0.20)
    )
    return is_drum, ke_panel, gantry_max, ke_panel > 0.50


# Mount (feet / suspension-line) rigid-body modes: a mode counts as a mount mode when at least
# this share of its strain energy sits in the mount springs.
MOUNT_MODE_STRAIN_SHARE = 0.50
# Mount modes below the Klipper resonance-test band are reported in `mounting_dynamics` only.
MOUNT_MODE_BAND_FLOOR_HZ = 5.0
# Cap on the reduced-mass correction m * A_dyn (keeps the reduced moving mass >= 45% of m).
_MOUNT_MASS_RATIO_CAP = 0.55


def mount_decoupling_factor(freq_ratio: float) -> float:
    """Inertial decoupling ``d(r) = r^4 / (1 + r^4)`` of a mount mode at ``r = f / f_mount``.

    A mount mode far above the frequency of interest (``r << 1``) acts as a quasi-static
    spring (``d -> 0``: its static residue is kept). Far below it (``r >> 1``) the frame is
    inertially free of the mount (``d -> 1``: the static residue is dropped and the frame
    recoils as a free mass). The quartic form is the regularised limit of the undamped modal
    receptance factor ``1 / (1 - r^2)`` with the resonance singularity at ``r = 1`` removed,
    because the mount mode itself is drawn as its own peak on the shaper graph.
    """
    r = max(float(freq_ratio), 0.0)
    r4 = r**4
    return r4 / (1.0 + r4)


def solve_mount_coupled_axis(
    mount_modes: list[dict],
    axis: str,
    c_static_m_per_n: float,
    c_drive_m_per_n: float,
    m_moving_kg: float,
    residue_key: str | None = None,
) -> dict[str, float]:
    """Couple one drive axis to the mount's rigid-body modes (modal receptance, fixed point).

    The FE static nozzle-to-bed compliance on compliant mounts is the complete modal sum
    ``c_static = sum_k R_k``. Mount modes below the axis resonance do not respond
    quasi-statically there, so their residues are removed in proportion to ``d(r_k)``
    (inertia relief), and the frame they release recoils against the moving mass:

        c_dyn  = c_static - sum_mount R_k d(r_k)
        A_dyn  = sum_mount A_k d(r_k)          A_k = (phi_k^T F)^2 / m_gen,k   [1/kg]
        m_red  = m (1 - m A_dyn)               (two-body reduced mass: 1 + m/M_f = 1/(1 - m A))
        f      = 1 / (2 pi sqrt(m_red (c_dyn + c_drive)))

    solved by fixed-point iteration on ``r_k = f / f_k``. With no mount modes this returns the
    static chain unchanged.
    """
    key_r = residue_key or f"r_{axis.lower()}"
    key_a = f"a_{axis.lower()}"
    c_static = max(float(c_static_m_per_n), 1e-12)
    c_drv = max(float(c_drive_m_per_n), 0.0)
    m = max(float(m_moving_kg), 1e-6)
    two_pi = 2.0 * math.pi
    f_static = 1.0 / (two_pi * math.sqrt(max(m * (c_static + c_drv), 1e-18)))
    if not mount_modes:
        return {
            "c_dyn_m_per_n": c_static,
            "m_reduced_kg": m,
            "freq_hz": f_static,
            "freq_static_hz": f_static,
            "mass_ratio": 0.0,
            "inertance_1_per_kg": 0.0,
            "retained_mount_compliance_m_per_n": 0.0,
        }

    f = f_static
    c_dyn = c_static
    m_red = m
    a_dyn = 0.0
    retained = 0.0
    for _ in range(8):
        removed = 0.0
        a_dyn = 0.0
        retained = 0.0
        for mm in mount_modes:
            f_k = max(float(mm["freq_hz"]), 1e-3)
            d = mount_decoupling_factor(f / f_k)
            r_k = float(mm.get(key_r, 0.0))
            removed += r_k * d
            retained += r_k * (1.0 - d)
            a_dyn += float(mm.get(key_a, 0.0)) * d
        # Inertia relief may raise or lower the deformation compliance; keep it physical.
        c_dyn = min(max(c_static - removed, 0.05 * c_static), 20.0 * c_static)
        mass_ratio = min(max(m * a_dyn, 0.0), _MOUNT_MASS_RATIO_CAP)
        m_red = m * (1.0 - mass_ratio)
        f_new = 1.0 / (two_pi * math.sqrt(max(m_red * (c_dyn + c_drv), 1e-18)))
        if abs(f_new - f) <= 1e-4 * max(f, 1.0):
            f = f_new
            break
        f = 0.5 * (f + f_new)
    return {
        "c_dyn_m_per_n": c_dyn,
        "m_reduced_kg": m_red,
        "freq_hz": f,
        "freq_static_hz": f_static,
        "mass_ratio": min(max(m * a_dyn, 0.0), _MOUNT_MASS_RATIO_CAP),
        "inertance_1_per_kg": a_dyn,
        "retained_mount_compliance_m_per_n": max(retained, 0.0),
    }


def solve_corexy_cross_axis_coupling(
    k_x_n_m: float,
    k_y_n_m: float,
    m_x_kg: float,
    m_y_kg: float,
    coupling: float,
) -> dict[str, float]:
    """Exact 2-DOF X/Y eigenproblem of a CoreXY whose A and B belt loops differ in stiffness.

    With the motors holding, belt A stretches by ``x + y`` and belt B by ``x - y``, so
    ``V = k_A (x + y)^2 / 2 + k_B (x - y)^2 / 2`` and

        K = [[k_A + k_B, k_A - k_B], [k_A - k_B, k_A + k_B]],   M = diag(m_x, m_y).

    Written per axis: ``K = [[k_x, c], [c, k_y]]`` with ``c = coupling * sqrt(k_x k_y)`` and
    ``coupling = (k_A - k_B) / (k_A + k_B)`` for a belt-only drive. Matched belts (c = 0) leave
    X and Y uncoupled; any mismatch puts each axis' resonance into the other axis' graph.
    Returns the X- and Y-dominant coupled frequencies and the cross peak of each graph as
    ``sqrt`` of its receptance-residue ratio to that graph's own main peak.
    """
    kx = max(float(k_x_n_m), 1e-6)
    ky = max(float(k_y_n_m), 1e-6)
    mx = max(float(m_x_kg), 1e-9)
    my = max(float(m_y_kg), 1e-9)
    c = max(-0.999, min(0.999, float(coupling))) * math.sqrt(kx * ky)
    two_pi = 2.0 * math.pi
    wx2, wy2 = kx / mx, ky / my
    # det([[kx - w mx, c], [c, ky - w my]]) = 0  ->  w^2 - (wx2 + wy2) w + wx2 wy2 - c^2/(mx my) = 0
    b_q = wx2 + wy2
    disc = math.sqrt(max((wx2 - wy2) ** 2 + 4.0 * c * c / (mx * my), 0.0))
    lam = (0.5 * (b_q - disc), 0.5 * (b_q + disc))
    modes = []
    for lam_j in lam:
        lam_j = max(lam_j, 1e-12)
        # Eigenvector from the better-conditioned row
        if abs(kx - lam_j * mx) >= abs(ky - lam_j * my):
            vec = (-c, kx - lam_j * mx)
        else:
            vec = (ky - lam_j * my, -c)
        if abs(vec[0]) + abs(vec[1]) < 1e-30:
            vec = (1.0, 0.0) if abs(wx2 - lam_j) <= abs(wy2 - lam_j) else (0.0, 1.0)
        m_gen = mx * vec[0] ** 2 + my * vec[1] ** 2
        modes.append(
            {
                "lam": lam_j,
                "res_x": vec[0] ** 2 / (m_gen * lam_j),
                "res_y": vec[1] ** 2 / (m_gen * lam_j),
                "x_share": mx * vec[0] ** 2 / m_gen,
            }
        )
    x_dom, y_dom = (modes[0], modes[1]) if modes[0]["x_share"] >= modes[1]["x_share"] else (modes[1], modes[0])
    return {
        "f_x_hz": math.sqrt(x_dom["lam"]) / two_pi,
        "f_y_hz": math.sqrt(y_dom["lam"]) / two_pi,
        "f_x_uncoupled_hz": math.sqrt(wx2) / two_pi,
        "f_y_uncoupled_hz": math.sqrt(wy2) / two_pi,
        # Y-dominant mode seen on the X graph, and X-dominant mode seen on the Y graph
        "cross_peak_in_x": math.sqrt(min(1.0, y_dom["res_x"] / max(x_dom["res_x"], 1e-30))),
        "cross_peak_in_y": math.sqrt(min(1.0, x_dom["res_y"] / max(y_dom["res_y"], 1e-30))),
    }


def _belt_tension_state(assembly: AssemblyModel) -> dict:
    """Operating belt tension, pluck frequencies, and CoreXY A/B loop stiffness mismatch."""
    spec = assembly.spec
    belt = spec.gantry.belt
    t_installed = float(belt.pretension_n)
    t_op = belt_operating_pretension_n(spec)
    mu = max(float(belt.linear_mass_kg_m), 1e-6)

    def _pluck_hz(tension_n: float) -> float:
        return (1.0 / (2.0 * 0.150)) * math.sqrt(max(tension_n, 0.0) / mu)

    out: dict = {
        "installed_pretension_n": round(t_installed, 2),
        "operating_pretension_n": round(t_op, 2),
        "thermal_tension_delta_n": round(belt_thermal_tension_delta_n(spec), 2),
        "pretension_set_temp_c": belt.pretension_set_temp_c,
        "pluck_hz_150mm_installed": round(_pluck_hz(t_installed), 1),
        "pluck_hz_150mm_operating": round(_pluck_hz(t_op), 1),
        "tension_mismatch_pct": round(float(belt.tension_mismatch_pct), 2),
        "coupling_kappa": 0.0,
    }
    p = float(belt.tension_mismatch_pct)
    if spec.architecture == "corexy" and p > 0.0:
        belt_a = operating_belt(spec, 1.0 + p / 200.0)
        belt_b = operating_belt(spec, 1.0 - p / 200.0)
        k_a = float(
            compute_drive_stiffness_details_n_per_m(assembly, "X", belt_override=belt_a)[
                "k_belt_subsystem_n_m"
            ]
        )
        k_b = float(
            compute_drive_stiffness_details_n_per_m(assembly, "X", belt_override=belt_b)[
                "k_belt_subsystem_n_m"
            ]
        )
        out.update(
            {
                "tension_a_n": round(float(belt_a.pretension_n), 2),
                "tension_b_n": round(float(belt_b.pretension_n), 2),
                "pluck_hz_150mm_a": round(_pluck_hz(float(belt_a.pretension_n)), 1),
                "pluck_hz_150mm_b": round(_pluck_hz(float(belt_b.pretension_n)), 1),
                "k_belt_a_n_per_mm": round(k_a * 1e-3, 2),
                "k_belt_b_n_per_mm": round(k_b * 1e-3, 2),
                "coupling_kappa": (k_a - k_b) / max(k_a + k_b, 1e-9),
            }
        )
    return out


def _merge_psd_modes(
    base: list[tuple[float, float]], extra: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Merge extra ``(freq_hz, relative_peak)`` PSD modes, keeping the taller of peaks within 3 Hz."""
    merged = list(base)
    for f_e, p_e in extra:
        for i, (f_b, p_b) in enumerate(merged):
            if abs(f_b - f_e) < 3.0:
                merged[i] = (f_b, max(p_b, p_e))
                break
        else:
            merged.append((f_e, p_e))
    return merged


def _classify_mount_mode(p_frac_xyz: tuple[float, float, float], rot_mean: np.ndarray) -> str:
    """Label a rigid-body mount mode from its translational effective-mass fractions."""
    px, py, pz = p_frac_xyz
    best = max(px, py, pz)
    if best >= 0.45:
        if best == pz:
            return "bounce_z"
        return "sway_x" if best == px else "sway_y"
    ax = int(np.argmax(np.abs(rot_mean))) if float(np.linalg.norm(rot_mean)) > 1e-12 else 2
    return ("rock_about_x", "rock_about_y", "yaw")[ax]


_MOUNT_MODE_LABELS = {
    "sway_x": "Whole-printer X sway",
    "sway_y": "Whole-printer Y sway",
    "bounce_z": "Whole-printer vertical bounce",
    "rock_about_x": "Whole-printer rocking (front-back tilt)",
    "rock_about_y": "Whole-printer rocking (left-right tilt)",
    "yaw": "Whole-printer yaw twist",
}


def _build_mounting_dynamics(
    assembly: AssemblyModel,
    feet_cfg: dict,
    mount_modes: list[dict],
    total_mass_kg: float,
    axis_x: dict[str, float],
    axis_y: dict[str, float],
    m_x_kg: float,
    m_y_kg: float,
) -> dict:
    """Assemble the user-facing mount report: rigid-body modes, recoil, and suspension safety."""
    spec = assembly.spec
    is_suspended = bool(feet_cfg.get("is_suspended", False))
    rb_modes = [
        {
            "frequency_hz": round(float(mm["freq_hz"]), 3),
            "kind": mm["kind"],
            "label": _MOUNT_MODE_LABELS.get(mm["kind"], mm["kind"]),
            "mount_strain_share_pct": round(100.0 * float(mm["u_feet_share"]), 1),
            "in_shaper_band": bool(float(mm["freq_hz"]) >= MOUNT_MODE_BAND_FLOOR_HZ),
            # Peak height a toolhead accelerometer reads for this whole-printer mode, relative
            # to the axis' main resonance peak (absolute motion; nozzle and bed move together).
            "adxl_peak_ratio_x": round(float(mm.get("adxl_x", 0.0)), 3),
            "adxl_peak_ratio_y": round(float(mm.get("adxl_y", 0.0)), 3),
        }
        for mm in sorted(mount_modes, key=lambda item: float(item["freq_hz"]))
    ]

    def _axis_block(sol: dict[str, float], m_kg: float) -> dict:
        ratio = float(sol["mass_ratio"])
        return {
            "moving_mass_kg": round(float(m_kg), 4),
            "reduced_moving_mass_kg": round(float(sol["m_reduced_kg"]), 4),
            # Frame recoil per unit of toolhead travel relative to the frame (momentum balance)
            "frame_recoil_ratio": round(ratio, 4),
            "frame_recoil_um_per_mm": round(1e3 * ratio, 1),
            # Drive-resonance rise from the frame recoiling as a free body: 1 / sqrt(1 - m A)
            "recoil_freq_gain_pct": round(100.0 * (1.0 / math.sqrt(max(1.0 - ratio, 1e-6)) - 1.0), 2),
            "retained_mount_compliance_um_per_n": round(
                float(sol["retained_mount_compliance_m_per_n"]) * 1e6, 4
            ),
        }

    sway = [float(mm["freq_hz"]) for mm in mount_modes if mm["kind"] in ("sway_x", "sway_y")]
    mount_zeta_val = float(feet_cfg.get("mount_damping_ratio", 0.05))
    mount_eta_val = float(feet_cfg.get("loss_factor", 2.0 * mount_zeta_val))
    out: dict = {
        "feet_type": spec.frame.feet_type,
        "mount_damping_ratio": round(mount_zeta_val, 4),
        "mount_loss_factor": round(mount_eta_val, 4),
        "total_mass_kg": round(float(total_mass_kg), 3),
        "ballast_mass_kg": round(float(assembly.ballast_mass_kg), 3),
        "ballast_location": spec.frame.ballast_location,
        "rigid_body_modes": rb_modes,
        "lowest_sway_hz": round(min(sway), 3) if sway else None,
        "x": _axis_block(axis_x, m_x_kg),
        "y": _axis_block(axis_y, m_y_kg),
        "notes": [],
    }
    notes: list[str] = out["notes"]
    in_band = [m for m in rb_modes if m["in_shaper_band"]]
    if in_band:
        lo = min(m["frequency_hz"] for m in in_band)
        hi = max(m["frequency_hz"] for m in in_band)
        span = f"{lo:.0f} Hz" if hi - lo < 1.0 else f"{lo:.0f}-{hi:.0f} Hz"
        peak = max(max(m["adxl_peak_ratio_x"], m["adxl_peak_ratio_y"]) for m in in_band)
        if peak >= 0.15:
            notes.append(
                f"Whole-printer mount modes at {span} sit inside the resonance-test band and reach "
                f"{100.0 * peak:.0f}% of the main peak on a toolhead accelerometer, although nozzle "
                "and bed move together in them."
            )
        else:
            notes.append(
                f"Whole-printer mount modes at {span} sit inside the resonance-test band but stay "
                f"below {max(100.0 * peak, 1.0):.0f}% of the main peak on a toolhead accelerometer."
            )
    if is_suspended:
        line_len_m = float(feet_cfg.get("line_length_m", 0.0))
        tension_n = float(assembly.suspension_line_tension_n)
        break_n = float(feet_cfg.get("line_break_strength_n", 0.0))
        f_pend = math.sqrt(_G_ACC / max(line_len_m, 1e-3)) / (2.0 * math.pi)
        v_max_mm_s = float(spec.kinematics.max_velocity_mm_s)
        out["suspension"] = {
            "line": feet_cfg.get("suspension_line", ""),
            "line_length_mm": round(line_len_m * 1e3, 1),
            "line_count": len([sp for sp in assembly.fastener_springs if sp.bay == "feet"]),
            "line_tension_n": round(tension_n, 2),
            "line_break_strength_n": round(break_n, 1),
            "safety_factor": round(break_n / max(tension_n, 1e-6), 1),
            "static_stretch_mm": round(
                1e3 * tension_n * line_len_m / max(float(feet_cfg.get("line_ea_n", 0.0)), 1e-6), 2
            ),
            "pendulum_hz_analytic": round(f_pend, 3),
            # Back-and-forth moves whose reversal rate v / (2 l) matches the pendulum pump the swing
            "swing_resonant_line_length_mm": round(v_max_mm_s / (2.0 * f_pend), 0),
        }
        design_factor = float(feet_cfg.get("line_design_factor", 5.0))
        out["suspension"]["design_factor"] = design_factor
        if break_n / max(tension_n, 1e-6) < design_factor:
            notes.append(
                f"Line tension {tension_n:.0f} N is above 1/{design_factor:.0f} of the "
                f"{break_n:.0f} N break strength: use a stronger line or more lines."
            )
        notes.append(
            f"Pendulum sway at {f_pend:.2f} Hz is far below the shaper band, so the frame reacts "
            "to toolhead moves as a free body; long straight moves reversing near that rate "
            f"(about {v_max_mm_s / (2.0 * f_pend):.0f} mm lines at {v_max_mm_s:.0f} mm/s) build up visible swing."
        )
    return out


def solve_modal_fea(
    assembly: AssemblyModel,
    num_modes: int = 6,
    include_adxl_diagnostic: bool = True,
    system: GlobalSystem | None = None,
) -> ModalResult:
    """Solve generalized structural eigenproblem K phi = omega^2 M phi.

    Pass a pre-assembled ``system`` (``assemble_global_system(assembly)``) to share the global
    matrices, element blocks, and ``K_ff`` factorization with the static and sensitivity solvers.
    """
    if system is None:
        system = assemble_global_system(assembly)
    K, M, _, node_idx, free_dofs = system.as_tuple()
    n_free = len(free_dofs)
    if n_free == 0:
        return ModalResult(fundamental_freq_hz=0.0)

    # Symmetrized free-free blocks; M_ff carries the strictly positive-definite diagonal floor
    # (for zero-mass rotational DOFs if any). Shared read-only via the GlobalSystem cache.
    K_ff = system.reduced_stiffness()
    M_ff = system.reduced_mass()

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
    target_scan = max(56 if has_panels else (32 if has_feet else 24), min(n_free, k_ret * 2))
    eigvals, eigvecs, lu_solver = _solve_generalized_eigenproblem(
        K_ff, M_ff, target_scan, system=system
    )

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
    # Exact unit inertial excitation force patterns for X and Y acceleration (used for modal receptance).
    # Split the toolhead inertial load between toolhead_cog and nozzle_tip in proportion to their
    # physical lumped masses (Newton's 2nd law F_i = m_i * a) so cantilevered hotend compliance
    # (e.g. V6 groove-mount vs rigid 4-bolt vs braced Tricorn) contributes physically to nozzle receptance.
    m_cog_lumped = max(float(assembly.nodes[cog_idx].lumped_mass_kg), 0.0)
    m_noz_lumped = max(float(assembly.nodes[noz_idx].lumped_mass_kg), 0.0) if noz_idx != cog_idx else 0.0
    if noz_idx != cog_idx and (m_cog_lumped + m_noz_lumped) > 1e-9:
        noz_inertia_frac = m_noz_lumped / (m_cog_lumped + m_noz_lumped)
        cog_inertia_frac = 1.0 - noz_inertia_frac
    else:
        noz_inertia_frac = 0.0
        cog_inertia_frac = 1.0

    fx_exc = np.zeros(n_dof, dtype=float)
    fx_exc[6 * cog_idx + 0] = cog_inertia_frac
    if noz_inertia_frac > 0.0:
        fx_exc[6 * noz_idx + 0] += noz_inertia_frac

    fy_exc = np.zeros(n_dof, dtype=float)
    if assembly.spec.architecture == "cartesian_bedslinger":
        if bed_idx is not None:
            fy_exc[6 * bed_idx + 1] = -1.0
        else:
            fy_exc[6 * cog_idx + 1] = cog_inertia_frac
            if noz_inertia_frac > 0.0:
                fy_exc[6 * noz_idx + 1] += noz_inertia_frac
    elif assembly.spec.architecture == "delta":
        fy_exc[6 * cog_idx + 1] = cog_inertia_frac
        if noz_inertia_frac > 0.0:
            fy_exc[6 * noz_idx + 1] += noz_inertia_frac
    else:
        th_share = min(
            1.0,
            max(0.0, assembly.toolhead_moving_mass_kg / max(assembly.y_moving_mass_kg, 1e-6)),
        )
        other_gantry = [
            gid
            for gid in assembly.gantry_node_ids
            if gid != assembly.toolhead_node_id and gid in node_idx
        ]
        if other_gantry:
            fy_exc[6 * cog_idx + 1] = th_share * cog_inertia_frac
            if noz_inertia_frac > 0.0:
                fy_exc[6 * noz_idx + 1] += th_share * noz_inertia_frac
            rem = (1.0 - th_share) / len(other_gantry)
            for gid in other_gantry:
                fy_exc[6 * node_idx[gid] + 1] += rem
        else:
            fy_exc[6 * cog_idx + 1] = cog_inertia_frac
            if noz_inertia_frac > 0.0:
                fy_exc[6 * noz_idx + 1] += noz_inertia_frac

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

    internal_dof_cursor = 6 * len(assembly.nodes)
    se_dofs_by_sub: dict[str, set[int]] = {"frame": set(), "gantry": set(), "toolhead": set(), "bed": set()}
    for se in getattr(assembly, "superelements", ()):
        n_m_se = max(0, int(getattr(se, "n_internal_modes", max(0, np.asarray(se.K_cb).shape[0] - 12))))
        if n_m_se > 0:
            sub = _role_to_subsystem(getattr(se, "role", "toolhead"))
            se_dofs_by_sub[sub].update(range(internal_dof_cursor, internal_dof_cursor + n_m_se))
            internal_dof_cursor += n_m_se

    def _free_mask(node_list: list[int], extra_dofs: set[int] | None = None) -> np.ndarray:
        if not node_list and not extra_dofs:
            return np.zeros(n_free, dtype=bool)
        dset = {6 * ni + d for ni in node_list for d in range(6)}
        if extra_dofs:
            dset.update(extra_dofs)
        return np.array([dof in dset for dof in free_dofs], dtype=bool)

    free_frame_mask = _free_mask(frame_node_indices, se_dofs_by_sub["frame"])
    free_gantry_mask = _free_mask(gantry_only_indices, se_dofs_by_sub["gantry"])
    free_th_mask = _free_mask(th_all_indices, se_dofs_by_sub["toolhead"])
    free_bed_mask = _free_mask(bed_all_indices, se_dofs_by_sub["bed"])

    feet_cfg = resolve_frame_feet(assembly.spec) if has_feet else {}
    feet_base_mass_kg = float(feet_cfg.get("base_mass_kg", 0.0)) if has_feet else 0.0
    mount_is_suspended = bool(feet_cfg.get("is_suspended", False))
    foot_node_ids = {sp.frame_node_id for sp in feet_springs if sp.frame_node_id in node_idx}

    # Exact workplane-compensated relative displacement extraction rows (dual to nozzle - (bed + theta_bed x dr))
    cx_rel = np.zeros(n_dof, dtype=float)
    cx_rel[6 * noz_idx + 0] = 1.0
    cy_rel = np.zeros(n_dof, dtype=float)
    cy_rel[6 * noz_idx + 1] = 1.0
    if bed_idx is not None:
        cx_rel[6 * bed_idx + 0] -= 1.0
        cx_rel[6 * bed_idx + 4] -= float(dr_noz[2])
        cx_rel[6 * bed_idx + 5] += float(dr_noz[1])
        cy_rel[6 * bed_idx + 1] -= 1.0
        cy_rel[6 * bed_idx + 5] -= float(dr_noz[0])
        cy_rel[6 * bed_idx + 3] += float(dr_noz[2])

    # Carriage-mounted ADXL relative displacement extraction rows (th_idx relative to bed_idx)
    cx_car_rel = np.zeros(n_dof, dtype=float)
    cx_car_rel[6 * th_idx + 0] = 1.0
    cy_car_rel = np.zeros(n_dof, dtype=float)
    cy_car_rel[6 * th_idx + 1] = 1.0
    if bed_idx is not None:
        cx_car_rel[6 * bed_idx + 0] -= 1.0
        cx_car_rel[6 * bed_idx + 4] -= float(dr_th[2])
        cx_car_rel[6 * bed_idx + 5] += float(dr_th[1])
        cy_car_rel[6 * bed_idx + 1] -= 1.0
        cy_car_rel[6 * bed_idx + 5] -= float(dr_th[0])
        cy_car_rel[6 * bed_idx + 3] += float(dr_th[2])

    # Exact unit-load static nozzle-to-bed structural compliance c_struct [m/N] (zero mode-truncation drift, B-01/B-02).
    # For top-heavy toolheads (cog_z > 0, noz_z < 0), out-of-phase toolhead pitch swings the nozzle
    # opposite to the carriage (c_noz < c_car). In dynamic vibration / input shaping, opposite-phase
    # pitch whip ADDS resonance compliance rather than stiffening the frame, so we form the
    # zero-truncation non-collocated compliance c_eff = c_car + w_th * |c_noz - c_car|.
    # Weighting the relative nozzle-to-carriage pitch whip by the toolhead moving-mass share
    # w_th = m_toolhead / m_axis (1.0 on X; ~0.35-0.50 on CoreXY Y) prevents a cantilevered nozzle
    # tip from attributing 100% of its local pitch whip to the X-beam and Y-carriage mass, while
    # staying exact and linear in mount-mode residues.
    w_th_x = 1.0
    if (
        assembly.spec.architecture == "corexy"
        and assembly.toolhead_moving_mass_kg > 0.0
        and assembly.y_moving_mass_kg > 0.0
    ):
        w_th_y = min(1.0, max(0.15, assembly.toolhead_moving_mass_kg / assembly.y_moving_mass_kg))
    else:
        w_th_y = 1.0
    try:
        u_x_stat = lu_solver.solve(fx_exc[free_dofs]) if lu_solver is not None else np.linalg.solve(K_ff, fx_exc[free_dofs])
        u_y_stat = lu_solver.solve(fy_exc[free_dofs]) if lu_solver is not None else np.linalg.solve(K_ff, fy_exc[free_dofs])
        c_noz_x_raw = float(cx_rel[free_dofs] @ u_x_stat)
        c_noz_y_raw = float(cy_rel[free_dofs] @ u_y_stat)
        c_car_x_raw = float(cx_car_rel[free_dofs] @ u_x_stat)
        c_car_y_raw = float(cy_car_rel[free_dofs] @ u_y_stat)
        flip_pitch_x = noz_idx != th_idx and c_noz_x_raw < c_car_x_raw
        flip_pitch_y = noz_idx != th_idx and c_noz_y_raw < c_car_y_raw
        c_struct_fe_x = max(
            1e-12,
            c_car_x_raw + w_th_x * ((c_car_x_raw - c_noz_x_raw) if flip_pitch_x else (c_noz_x_raw - c_car_x_raw)),
        )
        c_struct_fe_y = max(
            1e-12,
            c_car_y_raw + w_th_y * ((c_car_y_raw - c_noz_y_raw) if flip_pitch_y else (c_noz_y_raw - c_car_y_raw)),
        )
        c_struct_fe_car_x = max(1e-12, c_car_x_raw)
        c_struct_fe_car_y = max(1e-12, c_car_y_raw)
    except (np.linalg.LinAlgError, RuntimeError):
        flip_pitch_x = False
        flip_pitch_y = False
        c_struct_fe_x = 1e-6
        c_struct_fe_y = 1e-6
        c_struct_fe_car_x = 1e-6
        c_struct_fe_car_y = 1e-6

    candidates: list[dict] = []
    mount_modes: list[dict] = []
    n_nodal_dofs_all = 6 * len(assembly.nodes)
    mount_dir_vecs: list[np.ndarray] = []
    for d_ax in range(3):
        e_dir = np.zeros(n_free, dtype=float)
        if feet_blocks:
            for fi, gdof in enumerate(free_dofs):
                if int(gdof) < n_nodal_dofs_all and int(gdof) % 6 == d_ax:
                    e_dir[fi] = 1.0
        mount_dir_vecs.append(e_dir)
    mount_total_mass_kg = (
        max(float(mount_dir_vecs[0] @ M_ff @ mount_dir_vecs[0]), 1e-6) if feet_blocks else 1.0
    )
    frame_struct_idx = [
        i for i, n in enumerate(assembly.nodes) if n.role in ("frame", "foot", "tower", "mount")
    ]
    receptance_x: list[tuple[float, float]] = []
    receptance_y: list[tuple[float, float]] = []
    carriage_rec_x: list[tuple[float, float]] = []
    carriage_rec_y: list[tuple[float, float]] = []
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
        is_bed_z_mode = bed_idx is not None and (
            (bed_z_amp > 0.60 and float(noz_rel[0]) < 0.20 and float(noz_rel[1]) < 0.20)
            or (dom_axis == "Z" and bed_z_amp > 0.70 and float(max(noz_rel[0], noz_rel[1])) < 0.40)
            or (dom_axis == "Z" and bed_z_amp > 0.80)
        )

        # Exact signed modal compliance / receptance residue at the nozzle-to-bed interface (B-02):
        # R_{k,a} = (phi_{k,rel,a} * (phi_k^T F_a)) / (m_{gen,k} * omega_k^2)
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
            rel_x_raw = float(cx_rel @ full_vec)
            rel_y_raw = float(cy_rel @ full_vec)

            comp_x_noz = (rel_x_raw * pf_x) / (m_gen * lam)
            comp_y_noz = (rel_y_raw * pf_y) / (m_gen * lam)
            rec_x = abs(comp_x_noz)
            rec_y = abs(comp_y_noz)

            p_frac = (0.0, 0.0, 0.0)
            if feet_blocks:
                p_lin = [float(e_dir @ m_phi) for e_dir in mount_dir_vecs]
                p_frac = tuple(
                    min(1.0, (pv * pv) / (m_gen * mount_total_mass_kg)) for pv in p_lin
                )
            # When an enclosure panel drum frequency coincides with a whole-printer foot sway mode,
            # eigh() hybridizes the degenerate eigenspace; normalize foot strain share by non-panel
            # structural energy `ke_struct` so hybridized sway modes retain their mount classification.
            u_feet_struct = (
                min(1.0, u_feet_share / ke_struct)
                if (has_panels and not is_isolated_panel_drum and u_feet_share >= 0.08)
                else u_feet_share
            )
            is_mount_mode = bool(
                feet_blocks
                and (
                    u_feet_struct >= MOUNT_MODE_STRAIN_SHARE
                    or (u_feet_share >= 0.35 and max(p_frac) >= 0.65)
                )
            )
            if is_mount_mode:
                rot_mean = (
                    np.mean(rot_raw[frame_struct_idx], axis=0)
                    if frame_struct_idx
                    else np.zeros(3, dtype=float)
                )
                r_car_x_m = (float(cx_car_rel @ full_vec) * pf_x) / (m_gen * lam)
                r_car_y_m = (float(cy_car_rel @ full_vec) * pf_y) / (m_gen * lam)
                mount_modes.append(
                    {
                        "freq_hz": freq_hz,
                        "r_x": r_car_x_m + w_th_x * ((r_car_x_m - comp_x_noz) if flip_pitch_x else (comp_x_noz - r_car_x_m)),
                        "r_y": r_car_y_m + w_th_y * ((r_car_y_m - comp_y_noz) if flip_pitch_y else (comp_y_noz - r_car_y_m)),
                        "r_car_x": r_car_x_m,
                        "r_car_y": r_car_y_m,
                        "a_x": (pf_x * pf_x) / m_gen,
                        "a_y": (pf_y * pf_y) / m_gen,
                        "u_feet_share": u_feet_share,
                        "kind": _classify_mount_mode(p_frac, rot_mean),
                    }
                )

            # Exclude local corner-paver mass bounce modes above the whole-body rock band (>35 Hz)
            # created when paver base_mass_kg is lumped at frame corner nodes.
            is_local_paver_mode = False
            if feet_base_mass_kg > 0.0 and foot_node_ids and freq_hz > 35.0:
                ke_foot = (
                    sum(
                        assembly.nodes[node_idx[fid]].lumped_mass_kg
                        * float(np.sum(full_vec[6 * node_idx[fid] : 6 * node_idx[fid] + 3] ** 2))
                        for fid in foot_node_ids
                    )
                    / m_gen
                )
                is_local_paver_mode = ke_foot > 0.12

            if not is_isolated_panel_drum and not is_local_paver_mode:
                receptance_x.append((freq_hz, comp_x_noz))
                receptance_y.append((freq_hz, comp_y_noz))
                if noz_idx != th_idx:
                    rel_car_x_raw = float(cx_car_rel @ full_vec)
                    rel_car_y_raw = float(cy_car_rel @ full_vec)
                    carriage_rec_x.append((freq_hz, (rel_car_x_raw * pf_x) / (m_gen * lam)))
                    carriage_rec_y.append((freq_hz, (rel_car_y_raw * pf_y) / (m_gen * lam)))

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
        "y_rail_car_l",
        "y_rail_car_r",
        "y_car_l",
        "y_car_r",
        "x_beam_l",
        "x_beam_r",
        "g_fl",
        "g_fr",
        "g_rl",
        "g_rr",
        "pulley_fl",
        "pulley_fr",
        "pulley_rl",
        "pulley_rr",
        "col_g_fl",
        "col_g_fr",
        "col_g_rl",
        "col_g_rr",
        "bed_surface",
        "bed_frame",
        "bed_t_fl",
        "bed_t_fr",
        "bed_t_rear_mount",
        "z_bed_fl",
        "z_bed_fr",
        "z_bed_l",
        "z_bed_r",
        "z_arm_tip_fl",
        "z_arm_tip_fr",
        "z_arm_tip_r",
        "z_brg_lf",
        "z_brg_lr",
        "z_brg_rf",
        "z_brg_rr",
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

        from printsim.fea.toolhead_parts import (
            compute_modal_toolhead_parts,
            compute_toolhead_part_fields,
        )

        phi_norm = np.hstack([np.asarray(trans, dtype=np.float64), np.asarray(rot, dtype=np.float64)]).ravel()
        th_part_flex = compute_modal_toolhead_parts(
            assembly=assembly,
            node_idx=node_idx,
            phi_vec=phi_norm,
            max_node_disp=1.0,
        )
        th_mode_fields = compute_toolhead_part_fields(
            assembly=assembly,
            node_idx=node_idx,
            u_vec=phi_norm,
            normalized=True,
        )

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
            toolhead_part_flex=th_part_flex,
            toolhead_mode_fields=th_mode_fields,
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
                and rec_x < 0.05 * rx_max
                and rec_y < 0.05 * ry_max
            )
        )

        c_feet_share = float(c.get("u_feet_share", 0.0))
        # Sub-band rigid-body mount modes (e.g. sub-1 Hz pendulum sway on suspension lines) are
        # reported in `mounting_dynamics`, never as the structural fundamental.
        is_extra_feet_mode = (len(modes) >= 1 and c_feet_share > 0.72) or (
            c_feet_share >= MOUNT_MODE_STRAIN_SHARE
            and (mount_is_suspended or freq_hz < MOUNT_MODE_BAND_FLOOR_HZ)
        )
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

    # B-02 & B-14: Signed nozzle-to-bed modal receptance directly from the FE model
    # (no _geometric_car_freq truncation, no redundant Dunkerley overhang double-count).
    struct_x, norm_scanned_x = _receptance_structural_freq_and_psd(receptance_x, fundamental)
    struct_y, norm_scanned_y = _receptance_structural_freq_and_psd(receptance_y, fundamental)

    cf_insert_dd = (
        0.006
        if assembly.spec.gantry.effective_cf_rail_mounting == "two_wall_clamped_inserts"
        else 0.0
    )
    feet_dd = float(feet_cfg.get("damping_delta", 0.0)) if has_feet else 0.0
    if has_feet:
        psd_zeta = min(0.14, max(0.025, assembly.frame_damping_ratio - feet_dd) + cf_insert_dd)
    else:
        psd_zeta = min(0.14, max(0.025, assembly.frame_damping_ratio + cf_insert_dd))
    modal_frame_zeta = min(0.14, max(0.025, assembly.frame_damping_ratio + cf_insert_dd))

    if nozzle_twist_freq is None:
        nozzle_twist_freq = (
            nozzle_twist_fallback_freq
            if nozzle_twist_fallback_freq is not None
            else min(struct_x, struct_y)
        )

    k_belt_x, k_motor_x, k_bracket_x, k_drive_x, m_x_eff = _drive_axis_stiffness_n_per_m(assembly, "X")
    k_belt_y, k_motor_y, k_bracket_y, k_drive_y, m_y_eff = _drive_axis_stiffness_n_per_m(assembly, "Y")

    # B-14: mount (feet / suspension) rigid-body modes couple to each drive axis through their
    # FE modal receptance: inertia relief of the static compliance + free-frame reduced mass.
    mount_x = solve_mount_coupled_axis(
        mount_modes, "X", c_struct_fe_x, 1.0 / max(k_drive_x, 1.0), m_x_eff
    )
    mount_y = solve_mount_coupled_axis(
        mount_modes, "Y", c_struct_fe_y, 1.0 / max(k_drive_y, 1.0), m_y_eff
    )
    c_struct_x = float(mount_x["c_dyn_m_per_n"])
    c_struct_y = float(mount_y["c_dyn_m_per_n"])
    m_x_drive = float(mount_x["m_reduced_kg"])
    m_y_drive = float(mount_y["m_reduced_kg"])
    # CoreXY A/B belt tension mismatch: the two loops sit at different points of the belt's
    # low-tension stiffness toe, so k_A != k_B couples X and Y into a double peak.
    belt_dynamics = _belt_tension_state(assembly)
    cross_modes_x: list[tuple[float, float]] = []
    cross_modes_y: list[tuple[float, float]] = []
    kappa_belt = float(belt_dynamics["coupling_kappa"])
    if abs(kappa_belt) > 1e-9:
        c_tot_x = c_struct_x + 1.0 / max(k_drive_x, 1.0)
        c_tot_y = c_struct_y + 1.0 / max(k_drive_y, 1.0)
        # Only the belt's share of each axis' compliance takes part in the A/B coupling
        s_belt_x = min(1.0, (1.0 / max(k_belt_x, 1.0)) / max(c_tot_x, 1e-18))
        s_belt_y = min(1.0, (1.0 / max(k_belt_y, 1.0)) / max(c_tot_y, 1e-18))
        kappa_eff = kappa_belt * math.sqrt(s_belt_x * s_belt_y)
        cpl = solve_corexy_cross_axis_coupling(
            1.0 / c_tot_x, 1.0 / c_tot_y, m_x_drive, m_y_drive, kappa_eff
        )
        # Carry the coupled-mode frequency shift into each axis through its moving mass (f ~ m^-1/2)
        m_x_drive *= (cpl["f_x_uncoupled_hz"] / max(cpl["f_x_hz"], 1e-9)) ** 2
        m_y_drive *= (cpl["f_y_uncoupled_hz"] / max(cpl["f_y_hz"], 1e-9)) ** 2
        cross_modes_x.append((cpl["f_y_hz"], cpl["cross_peak_in_x"]))
        cross_modes_y.append((cpl["f_x_hz"], cpl["cross_peak_in_y"]))
        belt_dynamics.update(
            {
                "effective_coupling": round(kappa_eff, 4),
                "coupled_x_hz": round(cpl["f_x_hz"], 2),
                "coupled_y_hz": round(cpl["f_y_hz"], 2),
                "cross_peak_in_x": round(cpl["cross_peak_in_x"], 3),
                "cross_peak_in_y": round(cpl["cross_peak_in_y"], 3),
            }
        )
    belt_dynamics["coupling_kappa"] = round(kappa_belt, 4)

    # A toolhead accelerometer measures absolute motion, so it also reads the whole-printer mount
    # modes. Peak height relative to the axis' main resonance (acceleration response to the same
    # swept force): (A_k / (2 zeta_mount)) / ((1 / m) / (2 zeta_main)) = m A_k zeta_main / zeta_mount.
    mount_zeta = max(float(feet_cfg.get("mount_damping_ratio", 0.05)), 0.005) if has_feet else 0.05
    mount_adxl_x: list[tuple[float, float]] = []
    mount_adxl_y: list[tuple[float, float]] = []
    for mm in mount_modes:
        mm["adxl_x"] = min(1.0, m_x_eff * float(mm["a_x"]) * psd_zeta / mount_zeta)
        mm["adxl_y"] = min(1.0, m_y_eff * float(mm["a_y"]) * psd_zeta / mount_zeta)
        if float(mm["freq_hz"]) >= MOUNT_MODE_BAND_FLOOR_HZ:
            mount_adxl_x.append((float(mm["freq_hz"]), float(mm["adxl_x"])))
            mount_adxl_y.append((float(mm["freq_hz"]), float(mm["adxl_y"])))
    mounting_dynamics: dict | None = None
    if has_feet:
        mounting_dynamics = _build_mounting_dynamics(
            assembly=assembly,
            feet_cfg=feet_cfg,
            mount_modes=mount_modes,
            total_mass_kg=mount_total_mass_kg,
            axis_x=mount_x,
            axis_y=mount_y,
            m_x_kg=m_x_eff,
            m_y_kg=m_y_eff,
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
        c_struct_x,
        m_x_drive,
        k_belt_x,
        k_motor_x,
        k_bracket_x,
        k_drive_x,
        configured_accel,
        shaper_choice,
        scv_mm_s=scv_mm_s,
        scanned_modes=_merge_psd_modes(norm_scanned_x, cross_modes_x),
        psd_damping_ratio=psd_zeta,
        assembly=assembly,
    )
    shaper_y, shaper_graph_y = _recommend_shaper(
        "Y",
        struct_y,
        belt_y,
        motor_y,
        bracket_y,
        drive_y,
        c_struct_y,
        m_y_drive,
        k_belt_y,
        k_motor_y,
        k_bracket_y,
        k_drive_y,
        configured_accel,
        shaper_choice,
        scv_mm_s=scv_mm_s,
        scanned_modes=_merge_psd_modes(norm_scanned_y, cross_modes_y),
        psd_damping_ratio=psd_zeta,
        assembly=assembly,
    )

    adxl_vs_nozzle_diagnostic: dict | None = None
    if include_adxl_diagnostic and noz_idx != th_idx:
        struct_car_x, norm_scanned_car_x = _receptance_structural_freq_and_psd(
            carriage_rec_x, fundamental
        )
        struct_car_y, norm_scanned_car_y = _receptance_structural_freq_and_psd(
            carriage_rec_y, fundamental
        )
        c_struct_carriage_x = float(
            solve_mount_coupled_axis(
                mount_modes, "X", c_struct_fe_car_x, 1.0 / max(k_drive_x, 1.0), m_x_eff,
                residue_key="r_car_x",
            )["c_dyn_m_per_n"]
        )
        c_struct_carriage_y = float(
            solve_mount_coupled_axis(
                mount_modes, "Y", c_struct_fe_car_y, 1.0 / max(k_drive_y, 1.0), m_y_eff,
                residue_key="r_car_y",
            )["c_dyn_m_per_n"]
        )

        car_shaper_x, car_graph_x = _recommend_shaper(
            "X",
            struct_car_x,
            belt_x,
            motor_x,
            bracket_x,
            drive_x,
            c_struct_carriage_x,
            m_x_drive,
            k_belt_x,
            k_motor_x,
            k_bracket_x,
            k_drive_x,
            configured_accel,
            shaper_choice,
            scv_mm_s=scv_mm_s,
            scanned_modes=_merge_psd_modes(norm_scanned_car_x, mount_adxl_x + cross_modes_x),
            psd_damping_ratio=psd_zeta,
            assembly=assembly,
            fast_diagnostic=True,
        )
        car_shaper_y, car_graph_y = _recommend_shaper(
            "Y",
            struct_car_y,
            belt_y,
            motor_y,
            bracket_y,
            drive_y,
            c_struct_carriage_y,
            m_y_drive,
            k_belt_y,
            k_motor_y,
            k_bracket_y,
            k_drive_y,
            configured_accel,
            shaper_choice,
            scv_mm_s=scv_mm_s,
            scanned_modes=_merge_psd_modes(norm_scanned_car_y, mount_adxl_y + cross_modes_y),
            psd_damping_ratio=psd_zeta,
            assembly=assembly,
            fast_diagnostic=True,
        )

        def _build_adxl_diag_axis(
            car_sh: InputShaperRecommendation,
            car_gr: AxisShaperGraph,
            noz_sh: InputShaperRecommendation,
            noz_gr: AxisShaperGraph,
            c_noz_fe: float,
            c_car_fe: float,
        ) -> dict:
            if np.array_equal(car_gr.freqs_hz, noz_gr.freqs_hz):
                car_psd_aligned = car_gr.raw_psd
            else:
                car_psd_aligned = np.interp(noz_gr.freqs_hz, car_gr.freqs_hz, car_gr.raw_psd)
            car_peak = float(car_sh.effective_freq_hz)
            noz_peak = float(noz_sh.effective_freq_hz)
            ratio = float(c_noz_fe / max(c_car_fe, 1e-12))
            has_div = bool(
                car_gr.recommended_shaper != noz_gr.recommended_shaper
                or abs(car_peak - noz_peak) >= 3.0
                or ratio >= 1.25
            )
            return {
                "carriage_peak_hz": round(car_peak, 1),
                "nozzle_peak_hz": round(noz_peak, 1),
                "nozzle_to_carriage_ratio": round(ratio, 3),
                "carriage_recommended_shaper": car_gr.recommended_shaper,
                "carriage_recommended_freq_hz": round(float(car_gr.recommended_freq_hz), 1),
                "carriage_max_accel_mm_s2": round(float(car_sh.max_accel_mm_s2), 0),
                "nozzle_recommended_shaper": noz_gr.recommended_shaper,
                "nozzle_recommended_freq_hz": round(float(noz_gr.recommended_freq_hz), 1),
                "nozzle_max_accel_mm_s2": round(float(noz_sh.max_accel_mm_s2), 0),
                "carriage_raw_psd": [round(float(v), 3) for v in car_psd_aligned],
                "has_significant_divergence": has_div,
            }

        adxl_vs_nozzle_diagnostic = {
            "x": _build_adxl_diag_axis(
                car_shaper_x, car_graph_x, shaper_x, shaper_graph_x, c_struct_fe_x, c_struct_fe_car_x
            ),
            "y": _build_adxl_diag_axis(
                car_shaper_y, car_graph_y, shaper_y, shaper_graph_y, c_struct_fe_y, c_struct_fe_car_y
            ),
        }

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

    elem_blocks, lumped_blocks = _precompute_modal_element_blocks(
        assembly, node_idx, system=system
    )
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
        modes[primary_idx].is_limiting_axis_mode = True
        primary_mode = modes[primary_idx]
        primary_u_groups, primary_u_total, primary_t_groups, primary_t_total, _ = mode_energy_data[
            primary_idx
        ]

    # Share of the limiting axis' structural compliance still carried quasi-statically by the mount
    mount_lim = mount_x if limiting.axis == "X" else mount_y
    c_feet_lim = float(mount_lim["retained_mount_compliance_m_per_n"])
    c_struct_lim = c_struct_x if limiting.axis == "X" else c_struct_y
    feet_rock_frac = (
        min(0.99, c_feet_lim / max(c_struct_lim, 1e-18)) if (has_feet and c_feet_lim > 0.0) else 0.0
    )

    limiting_modal_summary = _build_limiting_modal_summary(
        assembly=assembly,
        limiting=limiting,
        other=other,
        primary_mode=primary_mode,
        primary_u_groups=primary_u_groups,
        primary_u_total=primary_u_total,
        primary_t_groups=primary_t_groups,
        primary_t_total=primary_t_total,
        ratio_lim=1.0,
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
        frame_damping_ratio=modal_frame_zeta,
        accel_bottleneck=accel_bottleneck,
        worst_print_mode_number=worst_print_mode_number,
        limiting_modal_summary=limiting_modal_summary,
        adxl_vs_nozzle_diagnostic=adxl_vs_nozzle_diagnostic,
        mounting_dynamics=mounting_dynamics,
        belt_dynamics=belt_dynamics,
    )
