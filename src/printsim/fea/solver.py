"""Static 3D Finite Element Analysis solver for gravity and peak acceleration loads."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu, spsolve

from printsim.fea.beam3d import assemble_global_matrices, element_internal_forces
from printsim.fea.shell_mitc4 import mitc4_shell_von_mises_mpa
from printsim.geometry.builder import AssemblyModel


def classify_part_group(member_id: str, role: str = "") -> tuple[str, str]:
    """Map internal split member/bay IDs to clean physical ``(group_id, label)`` pairs."""
    mid = member_id.strip()
    if mid.startswith(("col_fl", "col_fr")):
        return "front_frame_columns", "Front Frame Columns"
    if mid.startswith(("col_rl", "col_rr")):
        return "rear_frame_columns", "Rear Frame Columns"
    if mid.startswith("col_"):
        return "frame_columns", "Frame Vertical Columns"
    if mid.startswith("z_col_"):
        return "z_gantry_uprights", "Z-Axis Gantry Uprights"
    if mid.startswith("z_joint"):
        return "z_gantry_joints", "Z-Axis Gantry Joints"
    if mid.startswith("tower_") or role == "tower":
        return "delta_towers", "Delta Vertical Towers"
    if mid.startswith("arm_"):
        return "delta_arms", "Delta Diagonal Arms"
    if mid.startswith(("x_gantry", "x_beam")):
        return "x_gantry_beam", "Moving X-Gantry Beam"
    if mid.startswith(("y_rail", "gantry_y")):
        return "y_axis_rail_beams", "Y-Axis Rail Beams"
    if mid.startswith(("toolhead_mount", "nozzle_stem")):
        return "toolhead_mount", "Toolhead & Nozzle Mount"
    if mid.startswith(("bed_", "z_bed")):
        return "bed_support", "Bed Support & Z Arms"
    if mid.startswith("panel_shear_"):
        if "left" in mid or "right" in mid:
            return "side_panels", "Left & Right Enclosure Panels"
        if "rear" in mid:
            return "rear_panel", "Rear Enclosure Panel"
        if "top" in mid:
            return "top_panel", "Top Enclosure Panel"
        return "enclosure_panels", "Enclosure Shear Panels"
    if mid.startswith(("brace_", "diag_")):
        return "frame_bracing", "Frame Diagonal Bracing"
    if mid.startswith(("bay_left", "bay_right")):
        return "side_panels", "Left & Right Enclosure Panels"
    if mid.startswith("bay_rear"):
        return "rear_panel", "Rear Enclosure Panel"
    if mid.startswith(("bay_top_brace", "top_brace")):
        return "top_brace", "Top Open-Center Brace"
    if mid.startswith(("bay_front_brace", "front_brace")):
        return "front_brace", "Front Open-Door Brace"
    if mid.startswith("bay_top"):
        return "top_panel", "Top Enclosure Panel"
    if mid.startswith(("bay_backbone", "backbone")):
        return "trident_backbone", "Clee Trident Backbone Plate"
    if mid.startswith(
        ("top_", "bot_", "base_", "g_front", "g_rear", "g_left", "g_right", "gantry_rear_bar", "portal_top")
    ):
        return "frame_perimeter_beams", "Frame Perimeter Beams"

    if role == "rail":
        return "y_axis_rail_beams", "Y-Axis Rail Beams"
    if role == "gantry":
        return "x_gantry_beam", "Moving X-Gantry Beam"
    if role == "brace":
        return "frame_bracing", "Frame Diagonal Bracing"
    if role == "bed":
        return "bed_support", "Bed Support & Z Arms"
    if role == "toolhead":
        return "toolhead_mount", "Toolhead & Nozzle Mount"
    if role == "panel" or mid.startswith("bay_"):
        return "enclosure_panels", "Enclosure Panels"
    if role == "frame":
        return "frame_perimeter_beams", "Frame Perimeter Beams"
    return mid, mid.replace("_", " ").title()


@dataclass
class MemberStressResult:
    member_id: str
    role: str
    axial_force_n: float
    max_bending_moment_nm: float
    von_mises_mpa: float
    yield_mpa: float
    safety_factor: float
    utilization: float  # von_mises / yield


@dataclass
class ShellStressResult:
    id: str
    bay: str
    node_ids: tuple[str, str, str, str]
    von_mises_mpa: float


@dataclass
class StaticFEAResult:
    """Results from multi-case static FEA (gravity, peak X accel, peak Y accel, combined)."""

    gravity_sag_toolhead_um: float
    accel_deflection_x_um: float
    accel_deflection_y_um: float
    combined_max_deflection_um: float
    toolhead_stiffness_x_n_per_mm: float
    toolhead_stiffness_y_n_per_mm: float
    toolhead_stiffness_z_n_per_mm: float
    max_von_mises_mpa: float
    min_safety_factor: float
    peak_inertial_force_x_n: float
    peak_inertial_force_y_n: float
    beam_center_deflection_um: float = 0.0
    nozzle_twist_amplification_pct: float = 0.0
    nozzle_twist_angle_mrad: float = 0.0
    nozzle_absolute_deflection_um: float = 0.0
    bed_deflection_um: float = 0.0
    panel_max_von_mises_mpa: float = 0.0
    nodal_displacements_mm: dict[str, list[float]] = field(default_factory=dict)
    nodal_displacements_x_mm: dict[str, list[float]] = field(default_factory=dict)
    nodal_displacements_y_mm: dict[str, list[float]] = field(default_factory=dict)
    nodal_displacements_xy_mm: dict[str, list[float]] = field(default_factory=dict)
    member_stresses: list[MemberStressResult] = field(default_factory=list)
    shell_panel_stresses: list[ShellStressResult] = field(default_factory=list)
    peak_stress_part_label: str = ""
    top_flex_explanation: str = ""
    toolhead_flex_breakdown: dict[str, float | str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "gravity_sag_toolhead_um": round(self.gravity_sag_toolhead_um, 2),
            "accel_deflection_x_um": round(self.accel_deflection_x_um, 2),
            "accel_deflection_y_um": round(self.accel_deflection_y_um, 2),
            "combined_max_deflection_um": round(self.combined_max_deflection_um, 2),
            "nozzle_absolute_deflection_um": round(self.nozzle_absolute_deflection_um, 2),
            "bed_deflection_um": round(self.bed_deflection_um, 2),
            "beam_center_deflection_um": round(self.beam_center_deflection_um, 2),
            "nozzle_twist_amplification_pct": round(self.nozzle_twist_amplification_pct, 1),
            "nozzle_twist_angle_mrad": round(self.nozzle_twist_angle_mrad, 3),
            "toolhead_stiffness_x_n_per_mm": round(self.toolhead_stiffness_x_n_per_mm, 2),
            "toolhead_stiffness_y_n_per_mm": round(self.toolhead_stiffness_y_n_per_mm, 2),
            "toolhead_stiffness_z_n_per_mm": round(self.toolhead_stiffness_z_n_per_mm, 2),
            "max_von_mises_mpa": round(self.max_von_mises_mpa, 3),
            "panel_max_von_mises_mpa": round(self.panel_max_von_mises_mpa, 3),
            "min_safety_factor": round(min(self.min_safety_factor, 999.0), 2),
            "peak_inertial_force_x_n": round(self.peak_inertial_force_x_n, 2),
            "peak_inertial_force_y_n": round(self.peak_inertial_force_y_n, 2),
            "peak_stress_part_label": self.peak_stress_part_label,
            "top_flex_explanation": self.top_flex_explanation,
            "toolhead_flex_breakdown": self.toolhead_flex_breakdown,
            "nodal_displacements_mm": self.nodal_displacements_mm,
            "nodal_displacements_x_mm": self.nodal_displacements_x_mm,
            "nodal_displacements_y_mm": self.nodal_displacements_y_mm,
            "nodal_displacements_xy_mm": self.nodal_displacements_xy_mm,
            "member_stresses": [
                {
                    "member_id": ms.member_id,
                    "role": ms.role,
                    "axial_force_n": round(ms.axial_force_n, 2),
                    "max_bending_moment_nm": round(ms.max_bending_moment_nm, 3),
                    "von_mises_mpa": round(ms.von_mises_mpa, 3),
                    "yield_mpa": round(ms.yield_mpa, 1),
                    "safety_factor": round(min(ms.safety_factor, 999.0), 2),
                    "utilization": round(ms.utilization, 4),
                }
                for ms in self.member_stresses
            ],
            "shell_panel_stresses": [
                {
                    "id": s.id,
                    "bay": s.bay,
                    "node_ids": list(s.node_ids),
                    "von_mises_mpa": round(s.von_mises_mpa, 3),
                }
                for s in self.shell_panel_stresses
            ],
        }


def _build_factorized_solver(K: np.ndarray, free_dofs: list[int]):
    """Pre-factor K_ff once for fast multi-loadcase static solves."""
    n_dof = K.shape[0]
    if not free_dofs:
        return lambda F: np.zeros(n_dof, dtype=float)
    K_ff = 0.5 * (K[np.ix_(free_dofs, free_dofs)] + K[np.ix_(free_dofs, free_dofs)].T)

    if len(free_dofs) > 250:
        try:
            lu = splu(csc_matrix(K_ff))

            def _solve_sparse(F: np.ndarray) -> np.ndarray:
                u = np.zeros(n_dof, dtype=float)
                u[free_dofs] = lu.solve(F[free_dofs])
                return u

            return _solve_sparse
        except RuntimeError:
            reg = 1e-6 * max(float(np.max(np.diag(K_ff))), 1.0)
            lu = splu(csc_matrix(K_ff + reg * np.eye(len(free_dofs))))

            def _solve_sparse_reg(F: np.ndarray) -> np.ndarray:
                u = np.zeros(n_dof, dtype=float)
                u[free_dofs] = lu.solve(F[free_dofs])
                return u

            return _solve_sparse_reg

    try:
        c_fac = cho_factor(K_ff, check_finite=False)

        def _solve_chol(F: np.ndarray) -> np.ndarray:
            u = np.zeros(n_dof, dtype=float)
            u[free_dofs] = cho_solve(c_fac, F[free_dofs], check_finite=False)
            return u

        return _solve_chol
    except np.linalg.LinAlgError:
        reg = 1e-6 * max(float(np.max(np.diag(K_ff))), 1.0)
        K_reg = K_ff + reg * np.eye(len(free_dofs))

        def _solve_reg(F: np.ndarray) -> np.ndarray:
            u = np.zeros(n_dof, dtype=float)
            u[free_dofs] = solve(K_reg, F[free_dofs])
            return u

        return _solve_reg


def _solve_linear_system(K: np.ndarray, F: np.ndarray, free_dofs: list[int]) -> np.ndarray:
    u = np.zeros(K.shape[0], dtype=float)
    if not free_dofs:
        return u
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    F_f = F[free_dofs]
    if len(free_dofs) > 250:
        try:
            u[free_dofs] = spsolve(csc_matrix(K_ff), F_f)
            return u
        except RuntimeError:
            pass
    # Small Tikhonov regularization only if near-singular (e.g., unconstrained custom truss)
    try:
        u_f = solve(K_ff, F_f, assume_a="pos")
    except np.linalg.LinAlgError:
        reg = 1e-6 * np.max(np.diag(K_ff))
        u_f = solve(K_ff + reg * np.eye(len(free_dofs)), F_f)
    u[free_dofs] = u_f
    return u


def solve_static_fea(
    assembly: AssemblyModel,
    accel_mm_s2: float | None = None,
) -> StaticFEAResult:
    """Run static 3D frame FEA under gravity and peak toolhead acceleration loads."""
    K, _, F_grav, node_idx, free_dofs = assemble_global_matrices(assembly)
    n_dof = K.shape[0]
    th_idx = node_idx[assembly.toolhead_node_id]
    th_dof = 6 * th_idx
    cog_id = (
        assembly.toolhead_cog_node_id
        if assembly.toolhead_cog_node_id in node_idx
        else assembly.toolhead_node_id
    )
    noz_id = (
        assembly.nozzle_node_id
        if assembly.nozzle_node_id in node_idx
        else assembly.toolhead_node_id
    )
    cog_idx = node_idx[cog_id]
    cog_dof = 6 * cog_idx
    noz_idx = node_idx[noz_id]
    noz_dof = 6 * noz_idx
    has_explicit_cog = cog_idx != th_idx
    bed_idx = node_idx.get(assembly.bed_node_id) if assembly.bed_node_id else None
    has_bed = bed_idx is not None
    bed_dof = 6 * bed_idx if bed_idx is not None else -1
    feet_springs = [sp for sp in assembly.fastener_springs if sp.bay == "feet"]
    has_feet = bool(feet_springs)

    nmap_fea = assembly.node_map
    r_th = nmap_fea[assembly.toolhead_node_id].position_m
    r_cog = nmap_fea[cog_id].position_m
    r_noz = nmap_fea[noz_id].position_m
    r_bed = (
        nmap_fea[assembly.bed_node_id].position_m
        if has_bed and assembly.bed_node_id
        else np.zeros(3, dtype=float)
    )
    dr_noz_bed = r_noz - r_bed
    dr_th_bed = r_th - r_bed

    def _bed_wp_disp(u_vec: np.ndarray, dr: np.ndarray) -> np.ndarray:
        if not has_bed:
            return np.zeros(3, dtype=float)
        u_b = u_vec[bed_dof : bed_dof + 3]
        if has_feet:
            return u_b + np.cross(u_vec[bed_dof + 3 : bed_dof + 6], dr)
        return u_b

    def _rel_disp_vec(u_vec: np.ndarray) -> np.ndarray:
        u_noz = u_vec[noz_dof : noz_dof + 3]
        if has_bed:
            return u_noz - _bed_wp_disp(u_vec, dr_noz_bed)
        return u_noz

    accel_m_s2 = (
        accel_mm_s2 if accel_mm_s2 is not None else assembly.spec.kinematics.max_accel_mm_s2
    ) * 1e-3

    solve_rhs = _build_factorized_solver(K, free_dofs)

    # 1. Gravity sag case (nozzle relative to bed)
    u_grav = solve_rhs(F_grav)
    grav_sag_um = abs(float(_rel_disp_vec(u_grav)[2])) * 1e6

    # 2. Unit load compliance probing across structural loop (+1 N nozzle / -1 N bed in X, Y, Z)
    def _probe_stiffness(axis: int) -> tuple[float, tuple[float, float, float]]:
        f_unit = np.zeros(n_dof, dtype=float)
        f_unit[noz_dof + axis] = 1.0
        if has_bed:
            f_unit[bed_dof + axis] = -1.0
        u_unit = solve_rhs(f_unit)
        u_wp_noz = _bed_wp_disp(u_unit, dr_noz_bed)
        u_th = u_unit[th_dof : th_dof + 3]
        th_th = u_unit[th_dof + 3 : th_dof + 6]
        u_cog = u_unit[cog_dof : cog_dof + 3]
        th_cog = u_unit[cog_dof + 3 : cog_dof + 6]
        u_noz = u_unit[noz_dof : noz_dof + 3]

        p_th_at_noz = u_th + np.cross(th_th, r_noz - r_th)
        p_cog_at_noz = u_cog + np.cross(th_cog, r_noz - r_cog)

        d1 = max(0.0, float((p_th_at_noz - u_wp_noz)[axis]))
        d2 = max(0.0, float((p_cog_at_noz - p_th_at_noz)[axis]))
        d3 = max(0.0, float((u_noz - p_cog_at_noz)[axis]))
        rel_disp = float((u_noz - u_wp_noz)[axis])
        disp_mm = abs(rel_disp) * 1e3
        return 1.0 / max(disp_mm, 1e-9), (d1, d2, d3)

    stiff_x, (c1_x, c2_x, c3_x) = _probe_stiffness(0)
    stiff_y, (c1_y, c2_y, c3_y) = _probe_stiffness(1)
    stiff_z, _ = _probe_stiffness(2)

    c1_xy = c1_x + c1_y
    c2_xy = c2_x + c2_y
    c3_xy = c3_x + c3_y
    c_xy_tot = max(c1_xy + c2_xy + c3_xy, 1e-18)
    pct_rail = 100.0 * c1_xy / c_xy_tot
    pct_car = 100.0 * c2_xy / c_xy_tot
    pct_hot = 100.0 * c3_xy / c_xy_tot
    dom_flex_source = max(
        (("rail_xbeam", pct_rail), ("carriage_body", pct_car), ("hotend_nozzle", pct_hot)),
        key=lambda item: item[1],
    )[0]
    stage_stiff = assembly.toolhead_stage_stiffnesses_n_per_mm
    toolhead_flex_breakdown: dict[str, float | str] = {
        "rail_and_xbeam_pct": round(pct_rail, 2),
        "carriage_and_body_pct": round(pct_car, 2),
        "hotend_and_nozzle_pct": round(100.0 - round(pct_rail, 2) - round(pct_car, 2), 2),
        "effective_tip_stiffness_n_per_mm": round((2.0 / max(c_xy_tot, 1e-12)) * 1e-3, 2),
        "hotend_stiffness_n_per_mm": round(
            float(
                stage_stiff.get(
                    "hotend_total_n_per_mm",
                    assembly.spec.toolhead.hotend.structural_stiffness_n_per_mm,
                )
            ),
            2,
        ),
        "carriage_mount_stiffness_n_per_mm": round(
            float(stage_stiff.get("carriage_mount_n_per_mm", 0.0)), 2
        ),
        "dominant_flex_source": dom_flex_source,
    }

    # 3. Peak inertial loads applied at toolhead_cog (+ gantry / moving bed) during max X and max Y accel
    f_inertial_x = assembly.x_moving_mass_kg * accel_m_s2
    f_inertial_y = assembly.y_moving_mass_kg * accel_m_s2

    F_x = np.zeros(n_dof, dtype=float)
    F_x[cog_dof + 0] = f_inertial_x
    cog_z_m = assembly.spec.toolhead.cog_offset_mm[2] * 1e-3
    if not has_explicit_cog:
        F_x[th_dof + 4] = f_inertial_x * cog_z_m
    u_x = solve_rhs(F_x)
    defl_x_um = float(np.linalg.norm(_rel_disp_vec(u_x))) * 1e6

    F_y = np.zeros(n_dof, dtype=float)
    if assembly.spec.architecture == "cartesian_bedslinger":
        # Bedslinger Y acceleration moves only the heated bed + Y-carriage;
        # stationary Z-portal & toolhead see zero direct Y inertial force.
        if has_bed:
            F_y[bed_dof + 1] = -f_inertial_y
    else:
        # Toolhead + X gantry accelerate in Y -> load distributed on toolhead_cog and Y carriages
        th_share = assembly.spec.toolhead.total_mass_g * 1e-3 / max(assembly.y_moving_mass_kg, 1e-6)
        F_y[cog_dof + 1] = f_inertial_y * th_share
        if not has_explicit_cog:
            F_y[th_dof + 3] = -f_inertial_y * th_share * cog_z_m
        other_gantry = [gid for gid in assembly.gantry_node_ids if gid != assembly.toolhead_node_id]
        if other_gantry:
            rem = f_inertial_y * (1.0 - th_share) / len(other_gantry)
            for gid in other_gantry:
                F_y[6 * node_idx[gid] + 1] += rem
        else:
            F_y[cog_dof + 1] = f_inertial_y
    u_y = solve_rhs(F_y)
    defl_y_um = float(np.linalg.norm(_rel_disp_vec(u_y))) * 1e6

    # 4. Belt-induced frame preload (true CoreXY belt routing: self-equilibrating, sum F = 0, sum M = 0)
    # Neither 2WD nor AWD CoreXY runs a belt across the open front door (g_fl <-> g_fr).
    # Belt spans run along: rear bar (g_rl <-> g_rr), left/right Y rails (g_fl <-> y_car_l <-> g_rl),
    # and the moving X gantry beam (y_car_l <-> y_car_r), with equal forward/backward Y tensions
    # cancelling to 0 net Y force on y_car_l / y_car_r.
    F_belt = np.zeros(n_dof, dtype=float)
    if assembly.spec.architecture == "corexy":
        t0 = assembly.spec.gantry.belt.pretension_n
        belt_loads = [
            ("g_rl", +2.0 * t0, -2.0 * t0),
            ("g_rr", -2.0 * t0, -2.0 * t0),
            ("g_fl", 0.0, +2.0 * t0),
            ("g_fr", 0.0, +2.0 * t0),
            ("y_car_l", +2.0 * t0, 0.0),
            ("y_car_r", -2.0 * t0, 0.0),
        ]
        for nid, fx, fy in belt_loads:
            if nid in node_idx:
                bi = 6 * node_idx[nid]
                F_belt[bi + 0] += fx
                F_belt[bi + 1] += fy
    u_belt = solve_rhs(F_belt)

    # 5. Combined worst-case envelope (gravity + belt preload + peak unit-circle XY acceleration)
    s_diag = math.sqrt(0.5)
    accel_dirs = (
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (s_diag, s_diag),
        (s_diag, -s_diag),
        (-s_diag, s_diag),
        (-s_diag, -s_diag),
    )
    u_base = u_grav + u_belt
    u_comb = u_base + u_x
    best_peak = -1.0
    u_xy_comb = u_x.copy()
    best_xy_peak = -1.0
    for sx, sy in accel_dirs:
        cand_xy = sx * u_x + sy * u_y
        cand = u_base + cand_xy
        peak = float(np.linalg.norm(_rel_disp_vec(cand)))
        if peak > best_peak:
            best_peak = peak
            u_comb = cand
        xy_peak = float(np.linalg.norm(_rel_disp_vec(cand_xy)))
        if xy_peak > best_xy_peak:
            best_xy_peak = xy_peak
            u_xy_comb = cand_xy

    def _to_disp_map(u_vec: np.ndarray) -> dict[str, list[float]]:
        return {
            node.id: [round(float(v) * 1e3, 6) for v in u_vec[6 * i : 6 * i + 3]]
            for i, node in enumerate(assembly.nodes)
        }

    nodal_disp_mm = _to_disp_map(u_comb)
    nodal_disp_x_mm = _to_disp_map(u_x)
    nodal_disp_y_mm = _to_disp_map(u_y)
    nodal_disp_xy_mm = _to_disp_map(u_xy_comb)

    combined_nozzle_um = float(np.linalg.norm(_rel_disp_vec(u_comb))) * 1e6
    u_th_rel = u_comb[th_dof : th_dof + 3] - (
        _bed_wp_disp(u_comb, dr_th_bed) if has_bed else 0.0
    )
    beam_center_um = float(np.linalg.norm(u_th_rel)) * 1e6
    nozzle_twist_amp_pct = ((combined_nozzle_um / max(beam_center_um, 1e-6)) - 1.0) * 100.0
    nozzle_twist_angle_mrad = float(np.hypot(u_comb[noz_dof + 3], u_comb[noz_dof + 4])) * 1e3
    nozzle_abs_um = float(np.linalg.norm(u_comb[noz_dof : noz_dof + 3])) * 1e6
    if has_bed and has_feet:
        feet_idx_list = sorted(
            {node_idx[sp.frame_node_id] for sp in feet_springs if sp.frame_node_id in node_idx}
        )
        u_bed_raw = u_comb[bed_dof : bed_dof + 3]
        if len(feet_idx_list) >= 3:
            p_feet = np.array([assembly.nodes[fi].position_m for fi in feet_idx_list], dtype=float)
            p_mean = np.mean(p_feet, axis=0)
            u_feet = np.array([u_comb[6 * fi : 6 * fi + 3] for fi in feet_idx_list], dtype=float)
            a_rows = []
            b_rows = []
            for dp, uv in zip(p_feet - p_mean, u_feet):
                dx, dy, dz = float(dp[0]), float(dp[1]), float(dp[2])
                c_mat = np.array(
                    [[0.0, dz, -dy], [-dz, 0.0, dx], [dy, -dx, 0.0]],
                    dtype=float,
                )
                a_rows.append(np.hstack([np.eye(3, dtype=float), c_mat]))
                b_rows.append(uv)
            a_ls = np.vstack(a_rows)
            b_ls = np.concatenate(b_rows)
            rig_x, _, _, _ = np.linalg.lstsq(a_ls, b_ls, rcond=None)
            u_bed_rigid = rig_x[:3] + np.cross(rig_x[3:], r_bed - p_mean)
            bed_defl_um = float(np.linalg.norm(u_bed_raw - u_bed_rigid)) * 1e6
        elif feet_idx_list:
            u_feet_mean = np.mean(
                [u_comb[6 * fi : 6 * fi + 3] for fi in feet_idx_list], axis=0
            )
            bed_defl_um = float(np.linalg.norm(u_bed_raw - u_feet_mean)) * 1e6
        else:
            bed_defl_um = float(np.linalg.norm(u_bed_raw)) * 1e6
    else:
        bed_defl_um = float(np.linalg.norm(u_comb[bed_dof : bed_dof + 3])) * 1e6 if has_bed else 0.0

    # Element stress recovery
    member_stresses: list[MemberStressResult] = []
    max_vm = 0.0
    min_sf = float("inf")
    for member in assembly.members:
        i0 = node_idx[member.start_id]
        i1 = node_idx[member.end_id]
        p0 = assembly.nodes[i0].position_m
        p1 = assembly.nodes[i1].position_m
        u_elem = np.concatenate([u_comb[6 * i0 : 6 * i0 + 6], u_comb[6 * i1 : 6 * i1 + 6]])
        res = element_internal_forces(member, p0, p1, u_elem)
        yield_mpa = member.material.yield_strength_pa * 1e-6
        util = res["von_mises_mpa"] / max(yield_mpa, 1e-6)
        max_vm = max(max_vm, res["von_mises_mpa"])
        min_sf = min(min_sf, res["safety_factor"])
        member_stresses.append(
            MemberStressResult(
                member_id=member.id,
                role=member.role,
                axial_force_n=res["axial_force_n"],
                max_bending_moment_nm=res["max_bending_moment_nm"],
                von_mises_mpa=res["von_mises_mpa"],
                yield_mpa=yield_mpa,
                safety_factor=res["safety_factor"],
                utilization=util,
            )
        )

    # 2D MITC4 Shell panel stress recovery
    shell_panel_stresses: list[ShellStressResult] = []
    panel_max_vm = 0.0
    for shell in assembly.shell_elements:
        n_indices = [node_idx[nid] for nid in shell.node_ids]
        coords = np.array([assembly.nodes[ni].position_m for ni in n_indices], dtype=float)
        u_shell = np.concatenate([u_comb[6 * ni : 6 * ni + 6] for ni in n_indices])
        vm_mpa = mitc4_shell_von_mises_mpa(
            coords_3d=coords,
            u_elem_global=u_shell,
            e_pa=shell.material.youngs_modulus_pa,
            nu=shell.material.poisson_ratio,
            thickness_m=shell.thickness_m,
            d_bend_nm=shell.d_bend_nm,
            e_mem_h_n_m=shell.e_mem_h_n_m,
        )
        panel_max_vm = max(panel_max_vm, vm_mpa)
        yield_mpa = shell.material.yield_strength_pa * 1e-6
        max_vm = max(max_vm, vm_mpa)
        if vm_mpa > 1e-6:
            min_sf = min(min_sf, yield_mpa / vm_mpa)
        shell_panel_stresses.append(
            ShellStressResult(
                id=shell.id,
                bay=shell.bay,
                node_ids=shell.node_ids,
                von_mises_mpa=vm_mpa,
            )
        )

    resolved_sf = min_sf if np.isfinite(min_sf) else 999.0
    sf_disp = min(resolved_sf, 999.0)

    peak_member = max(member_stresses, key=lambda m: m.von_mises_mpa, default=None)
    peak_shell = max(shell_panel_stresses, key=lambda s: s.von_mises_mpa, default=None)
    if peak_shell is not None and (
        peak_member is None or peak_shell.von_mises_mpa > peak_member.von_mises_mpa
    ):
        bay_base = (
            peak_shell.bay
            if peak_shell.bay in ("top_brace", "front_brace", "backbone")
            else peak_shell.bay.split("_")[0]
        )
        _, peak_stress_label = classify_part_group(f"bay_{bay_base}", "panel")
    elif peak_member is not None:
        _, peak_stress_label = classify_part_group(peak_member.member_id, peak_member.role)
    else:
        peak_stress_label = "Frame Structure"

    if max(defl_x_um, defl_y_um) >= grav_sag_um:
        top_flex_explanation = (
            f"Under horizontal XY print acceleration, the nozzle flexes {defl_x_um:.1f} µm (X) / "
            f"{defl_y_um:.1f} µm (Y) relative to the bed vs {grav_sag_um:.1f} µm gravity sag "
            f"(peak stress in {peak_stress_label}: {max_vm:.1f} MPa, SF {sf_disp:.0f}×)."
        )
    else:
        top_flex_explanation = (
            f"Static gravity sag is {grav_sag_um:.1f} µm while XY print acceleration adds "
            f"{defl_x_um:.1f} µm (X) / {defl_y_um:.1f} µm (Y) nozzle-to-bed flex "
            f"(peak stress in {peak_stress_label}: {max_vm:.1f} MPa, SF {sf_disp:.0f}×)."
        )
    if nozzle_twist_amp_pct > 25.0:
        top_flex_explanation += (
            f" Toolhead pitch adds +{nozzle_twist_amp_pct:.0f}% lever-arm twist at the nozzle tip."
        )

    return StaticFEAResult(
        gravity_sag_toolhead_um=grav_sag_um,
        accel_deflection_x_um=defl_x_um,
        accel_deflection_y_um=defl_y_um,
        combined_max_deflection_um=combined_nozzle_um,
        beam_center_deflection_um=beam_center_um,
        nozzle_twist_amplification_pct=nozzle_twist_amp_pct,
        nozzle_twist_angle_mrad=nozzle_twist_angle_mrad,
        nozzle_absolute_deflection_um=nozzle_abs_um,
        bed_deflection_um=bed_defl_um,
        panel_max_von_mises_mpa=panel_max_vm,
        toolhead_stiffness_x_n_per_mm=stiff_x,
        toolhead_stiffness_y_n_per_mm=stiff_y,
        toolhead_stiffness_z_n_per_mm=stiff_z,
        max_von_mises_mpa=max_vm,
        min_safety_factor=resolved_sf,
        peak_inertial_force_x_n=f_inertial_x,
        peak_inertial_force_y_n=f_inertial_y,
        nodal_displacements_mm=nodal_disp_mm,
        nodal_displacements_x_mm=nodal_disp_x_mm,
        nodal_displacements_y_mm=nodal_disp_y_mm,
        nodal_displacements_xy_mm=nodal_disp_xy_mm,
        member_stresses=member_stresses,
        shell_panel_stresses=shell_panel_stresses,
        peak_stress_part_label=peak_stress_label,
        top_flex_explanation=top_flex_explanation,
        toolhead_flex_breakdown=toolhead_flex_breakdown,
    )
