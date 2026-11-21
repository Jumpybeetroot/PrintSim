"""Exact Analytical Adjoint Static & Fox-Kapoor Modal Sensitivity Engine.

Computes closed-form analytical derivatives (zero finite-difference loops) across all
beam members, MITC4 shell panels, CBUSH fastener springs, and lumped masses:
- Static adjoint compliance sensitivities via a single back-substitution of ``K_ff``
  for nozzle-to-bed relative displacement ``u_rel`` in X and Y.
- Fox-Kapoor (1968) modal frequency sensitivities ``df_k / d(ln E_e)`` and
  ``df_k / d(ln m_e)`` with Dailey (1989) repeated-root subspace protection.
- Uniform thickening ROI metrics ``hz_per_100g``, ``hz_per_usd``, ``compliance_share_pct``,
  normalized ``heat_score`` in ``[-1, +1]``, bay-grouped panel sensitivities, and top
  stiffening / weight-reduction hotspots.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

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
from printsim.fea.modal import ModalResult
from printsim.fea.shell_mitc4 import fastener_spring_matrices, mitc4_shell_matrices
from printsim.fea.solver import StaticFEAResult, _build_factorized_solver
from printsim.geometry.builder import AssemblyModel, StructuralMember
from printsim.schema import EXTRUSION_FILLINGS, get_material


@dataclass
class SensitivityReport:
    """Analytical structural sensitivity and ROI breakdown for an AssemblyModel."""

    fundamental_freq_hz: float
    primary_x_freq_hz: float
    primary_y_freq_hz: float
    static_rel_disp_x_um: float
    static_rel_disp_y_um: float
    member_sensitivities: dict[str, dict[str, Any]] = field(default_factory=dict)
    panel_bay_sensitivities: dict[str, dict[str, Any]] = field(default_factory=dict)
    lumped_mass_sensitivities: dict[str, dict[str, Any]] = field(default_factory=dict)
    top_stiffen_targets: list[dict[str, Any]] = field(default_factory=list)
    top_lighten_targets: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fundamental_freq_hz": round(float(self.fundamental_freq_hz), 2),
            "primary_x_freq_hz": round(float(self.primary_x_freq_hz), 2),
            "primary_y_freq_hz": round(float(self.primary_y_freq_hz), 2),
            "static_rel_disp_x_um": round(float(self.static_rel_disp_x_um), 3),
            "static_rel_disp_y_um": round(float(self.static_rel_disp_y_um), 3),
            "member_sensitivities": self.member_sensitivities,
            "panel_bay_sensitivities": self.panel_bay_sensitivities,
            "lumped_mass_sensitivities": self.lumped_mass_sensitivities,
            "top_stiffen_targets": self.top_stiffen_targets,
            "top_lighten_targets": self.top_lighten_targets,
        }


def _solve_eigen_modes(
    K_ff: np.ndarray,
    M_ff: np.ndarray,
    target_scan: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve generalized eigenproblem K_ff phi = lam M_ff phi with M-orthonormalization."""
    n_free = K_ff.shape[0]
    k_scan = min(target_scan, n_free)
    used_sparse = False
    if n_free > 500 and 1 <= k_scan < n_free - 1:
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
        eigvals, eigvecs = eigh(K_ff, M_ff, subset_by_index=[0, k_scan - 1])

    eigvals = np.maximum(eigvals, 0.0)
    for j in range(eigvecs.shape[1]):
        v = eigvecs[:, j]
        m_gen = float(v @ M_ff @ v)
        if m_gen > 1e-24:
            eigvecs[:, j] = v / math.sqrt(m_gen)
    return eigvals, eigvecs


def _dailey_subspace_vector(
    eigvals: np.ndarray,
    full_vecs: np.ndarray,
    target_idx: int,
    exc_vec: np.ndarray,
    rel_tol: float = 1e-4,
) -> np.ndarray:
    """Dailey (1989) repeated-root guard returning an M-normalized subspace mode vector."""
    f_all = np.sqrt(np.maximum(eigvals, 0.0)) / (2.0 * math.pi)
    f_k = float(f_all[target_idx])
    cluster = [
        j
        for j in range(len(f_all))
        if abs(float(f_all[j]) - f_k) / max(f_k, 1.0) < rel_tol
    ]
    if len(cluster) <= 1:
        return full_vecs[:, target_idx]

    Phi_R = full_vecs[:, cluster]
    w_proj = Phi_R.T @ exc_vec
    norm_w = float(np.linalg.norm(w_proj))
    if norm_w > 1e-12:
        w_proj = w_proj / norm_w
        return Phi_R @ w_proj
    return full_vecs[:, target_idx]


def _member_mass_and_cost(
    member: StructuralMember,
    length_m: float,
    assembly: AssemblyModel,
) -> tuple[float, float, float, float]:
    """Return ``(exact_mass_kg, ref_mass_kg, cost_usd, cost_per_kg_usd)`` for a member."""
    m_exact = member.profile.area_m2 * length_m * member.material.density_kg_m3
    spec = assembly.spec
    if member.material.density_kg_m3 >= 1.0:
        rail_mat = get_material(spec.gantry.rail_material)
        base_mk = max(
            m_exact - member.fill_mass_kg - member.rail_mass_kg - member.backer_mass_kg,
            0.0,
        )
        cost = base_mk * member.material.cost_per_kg_usd
        if member.rail_mass_kg > 0.0:
            cost += member.rail_mass_kg * rail_mat.cost_per_kg_usd
        if member.backer_mass_kg > 0.0 and member.backer_material_id:
            cost += member.backer_mass_kg * get_material(member.backer_material_id).cost_per_kg_usd
        if member.fill_mass_kg > 0.0:
            fill_spec = EXTRUSION_FILLINGS.get(
                spec.frame.extrusion_filling, EXTRUSION_FILLINGS["none"]
            )
            cost += member.fill_mass_kg * float(fill_spec["cost_per_kg_usd"])
        eff_cost_kg = cost / max(m_exact, 1e-6) if cost > 0.0 else member.material.cost_per_kg_usd
        return m_exact, max(m_exact, 1e-4), cost, max(eff_cost_kg, 0.5)

    # Virtual massless link (toolhead_mount, nozzle_stem, bed_arm_*, panel_shear_*)
    if member.id == "toolhead_mount":
        ref_mat = get_material(spec.toolhead.mount_material)
        m_ref = 0.06
    elif member.id == "nozzle_stem":
        ref_mat = get_material("aluminum_6061_t6")
        m_ref = 0.025
    elif member.role == "bed":
        ref_mat = get_material(spec.bed.plate_material)
        m_ref = max(spec.bed.mass_g * 1e-3 / 3.0, 0.25)
    else:
        ref_mat = get_material(spec.frame.panel_material)
        m_ref = 0.25
    cost_kg = max(ref_mat.cost_per_kg_usd, 1.0)
    return m_exact, m_ref, m_ref * cost_kg, cost_kg


def _stiffen_hint(entry_id: str, role: str) -> str:
    if role == "toolhead" or entry_id in ("toolhead_mount", "nozzle_stem"):
        return "Upgrade toolhead carriage mount from polymer to CNC Al 6061 / PPA-CF or shorten nozzle overhang."
    if role == "rail":
        return "Upgrade linear rail profile (e.g. MGN9 -> MGN12/15) or bolt structural side panels to Y-rail extrusion."
    if role == "panel" or entry_id.startswith("bay_"):
        return "Increase panel thickness (4-5mm) or switch to bolted aluminum_5052_sheet / carbon_fiber_plate."
    if role == "brace":
        return "Upgrade diagonal truss brace cross-section or switch to full_truss bracing."
    if role == "gantry":
        return "Step up X-gantry beam section (e.g. 2020 -> 2040/cf_tube) or add dual top+bottom X rails."
    return "Step up frame extrusion profile (2020 -> 3030), add diagonal bracing, or bolt structural panels."


def _lighten_hint(entry_id: str, role: str) -> str:
    if role in ("cog", "toolhead", "nozzle") or "toolhead" in entry_id:
        return "Reduce moving toolhead mass (compact direct-drive extruder, CPAP remote cooling, lighter hotend)."
    if role == "gantry" or "x_gantry" in entry_id:
        return "Switch moving X-gantry beam to roll-wrapped carbon_fiber_tube or hollow lighter extrusion."
    if role == "rail":
        return "Use hollow/lightened rail backing or match rail size (MGN9 vs MGN12) to carriage load."
    if role == "bed":
        return "Lighten moving heated bed plate (thinner cast Al / mag plate) to raise Y-axis resonance."
    return "Trim non-structural upper frame mass or substitute lower-density structural material."


def compute_structural_sensitivities(
    assembly: AssemblyModel,
    static_res: StaticFEAResult | None = None,
    modal_res: ModalResult | None = None,
) -> SensitivityReport:
    """Compute exact analytical static adjoint and Fox-Kapoor modal sensitivities."""
    _ = static_res  # static fields are solved exactly with one factorization of K_ff
    K, M, _, node_idx, free_dofs = assemble_global_matrices(assembly)
    n_dof = K.shape[0]
    n_free = len(free_dofs)

    if n_free == 0:
        return SensitivityReport(
            fundamental_freq_hz=0.0,
            primary_x_freq_hz=0.0,
            primary_y_freq_hz=0.0,
            static_rel_disp_x_um=0.0,
            static_rel_disp_y_um=0.0,
        )

    K_ff = 0.5 * (K[np.ix_(free_dofs, free_dofs)] + K[np.ix_(free_dofs, free_dofs)].T)
    M_ff = 0.5 * (M[np.ix_(free_dofs, free_dofs)] + M[np.ix_(free_dofs, free_dofs)].T)
    m_diag = np.diag(M_ff)
    m_floor = max(float(np.max(m_diag)) * 1e-12, 1e-12)
    M_ff = M_ff + np.diag(np.where(m_diag < m_floor, m_floor, 0.0))

    # 1. Static primary solves (X & Y peak acceleration loads) + single-backsub adjoints
    th_idx = node_idx[assembly.toolhead_node_id]
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
    noz_idx = node_idx[noz_id]
    has_explicit_cog = cog_idx != th_idx
    bed_idx = node_idx.get(assembly.bed_node_id) if assembly.bed_node_id else None
    has_bed = bed_idx is not None

    solve_rhs = _build_factorized_solver(K, free_dofs)
    accel_m_s2 = assembly.spec.kinematics.max_accel_mm_s2 * 1e-3
    cog_z_m = assembly.spec.toolhead.cog_offset_mm[2] * 1e-3

    F_x = np.zeros(n_dof, dtype=float)
    f_inertial_x = assembly.x_moving_mass_kg * accel_m_s2
    F_x[6 * cog_idx + 0] = f_inertial_x
    if not has_explicit_cog:
        F_x[6 * th_idx + 4] = f_inertial_x * cog_z_m

    F_y = np.zeros(n_dof, dtype=float)
    f_inertial_y = assembly.y_moving_mass_kg * accel_m_s2
    if assembly.spec.architecture == "cartesian_bedslinger":
        F_y[6 * cog_idx + 1] = assembly.x_moving_mass_kg * accel_m_s2
        if not has_explicit_cog:
            F_y[6 * th_idx + 3] = -(assembly.x_moving_mass_kg * accel_m_s2) * cog_z_m
        if has_bed and bed_idx is not None:
            F_y[6 * bed_idx + 1] = -(assembly.spec.bed.mass_g * 1e-3) * accel_m_s2
    else:
        th_share = assembly.spec.toolhead.total_mass_g * 1e-3 / max(assembly.y_moving_mass_kg, 1e-6)
        F_y[6 * cog_idx + 1] = f_inertial_y * th_share
        if not has_explicit_cog:
            F_y[6 * th_idx + 3] = -f_inertial_y * th_share * cog_z_m
        other_gantry = [
            gid
            for gid in assembly.gantry_node_ids
            if gid != assembly.toolhead_node_id and gid in node_idx
        ]
        if other_gantry:
            rem = f_inertial_y * (1.0 - th_share) / len(other_gantry)
            for gid in other_gantry:
                F_y[6 * node_idx[gid] + 1] += rem
        else:
            F_y[6 * cog_idx + 1] = f_inertial_y

    c_x = np.zeros(n_dof, dtype=float)
    c_x[6 * noz_idx + 0] = 1.0
    if has_bed and bed_idx is not None:
        c_x[6 * bed_idx + 0] = -1.0

    c_y = np.zeros(n_dof, dtype=float)
    c_y[6 * noz_idx + 1] = 1.0
    if has_bed and bed_idx is not None:
        c_y[6 * bed_idx + 1] = -1.0

    u_x = solve_rhs(F_x)
    u_y = solve_rhs(F_y)
    lam_x = solve_rhs(c_x)
    lam_y = solve_rhs(c_y)

    u_rel_x = float(c_x @ u_x)
    u_rel_y = float(c_y @ u_y)

    # 2. Modal eigensolve and primary X / Y / fundamental structural mode identification
    panel_indices = [i for i, n in enumerate(assembly.nodes) if n.role == "panel"]
    non_panel_indices = [i for i, n in enumerate(assembly.nodes) if n.role != "panel"]
    gantry_indices = [
        i
        for i, n in enumerate(assembly.nodes)
        if n.role in ("gantry", "toolhead", "nozzle", "cog")
    ]
    has_panels = len(panel_indices) > 0 and len(non_panel_indices) > 0
    target_scan = 56 if has_panels else 24

    eigvals, eigvecs = _solve_eigen_modes(K_ff, M_ff, target_scan=target_scan)
    n_eig = len(eigvals)
    full_vecs = np.zeros((n_dof, n_eig), dtype=float)
    full_vecs[free_dofs, :] = eigvecs

    fx_exc = np.zeros(n_dof, dtype=float)
    fx_exc[6 * cog_idx + 0] = 1.0
    fy_exc = np.zeros(n_dof, dtype=float)
    if assembly.spec.architecture == "cartesian_bedslinger":
        fy_exc[6 * cog_idx + 1] = 1.0
        if has_bed and bed_idx is not None:
            fy_exc[6 * bed_idx + 1] = -1.0
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

    free_panel_mask = np.zeros(n_free, dtype=bool)
    if has_panels:
        panel_dof_set = {6 * pi + d for pi in panel_indices for d in range(6)}
        free_panel_mask = np.array([dof in panel_dof_set for dof in free_dofs], dtype=bool)

    struct_candidates: list[dict[str, Any]] = []
    for idx in range(n_eig):
        lam = max(float(eigvals[idx]), 0.0)
        freq_hz = math.sqrt(lam) / (2.0 * math.pi)
        phi_f = eigvecs[:, idx]
        full_vec = full_vecs[:, idx]

        trans_raw = np.array([full_vec[6 * i : 6 * i + 3] for i in range(len(assembly.nodes))])
        max_amp = float(np.max(np.linalg.norm(trans_raw, axis=1)))
        trans = trans_raw / max_amp if max_amp > 1e-12 else trans_raw.copy()

        noz_rel_all = (
            np.abs(trans[noz_idx] - trans[bed_idx]) if bed_idx is not None else np.abs(trans[noz_idx])
        )
        noz_norm_all = float(np.linalg.norm(noz_rel_all))

        is_isolated_panel_drum = False
        if has_panels:
            np_max = float(np.max(np.linalg.norm(trans[non_panel_indices], axis=1)))
            gantry_max = (
                float(np.max(np.linalg.norm(trans[gantry_indices], axis=1)))
                if gantry_indices
                else np_max
            )
            phi_p = np.where(free_panel_mask, phi_f, 0.0)
            ke_panel = float(phi_p @ M_ff @ phi_p)
            is_isolated_panel_drum = (
                (ke_panel > 0.55 and noz_norm_all < 0.12)
                or (ke_panel > 0.68 and gantry_max < 0.35 and noz_norm_all < 0.15)
                or (np_max < 0.25 and gantry_max < 0.20)
            )
            tr_struct = trans_raw[non_panel_indices]
            max_struct = max(float(np.max(np.linalg.norm(tr_struct, axis=1))), 1e-12)
            noz_rel = (
                np.abs(trans_raw[noz_idx] - trans_raw[bed_idx]) / max_struct
                if bed_idx is not None
                else np.abs(trans_raw[noz_idx]) / max_struct
            )
            bed_z_amp = abs(float(trans_raw[bed_idx][2] / max_struct)) if bed_idx is not None else 0.0
            axis_energy = np.sum(tr_struct**2, axis=0)
        else:
            noz_rel = noz_rel_all
            bed_z_amp = abs(float(trans[bed_idx][2])) if bed_idx is not None else 0.0
            axis_energy = np.sum(trans**2, axis=0)

        dom_axis = ["X", "Y", "Z"][int(np.argmax(axis_energy))]
        is_bed_z_mode = (
            bed_idx is not None
            and bed_z_amp > 0.60
            and float(noz_rel[0]) < 0.20
            and float(noz_rel[1]) < 0.20
        )

        rec_x = 0.0
        rec_y = 0.0
        if lam > 1.0:
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
            rec_x = abs(rel_x_raw * float(full_vec @ fx_exc)) / lam
            rec_y = abs(rel_y_raw * float(full_vec @ fy_exc)) / lam

        struct_candidates.append(
            {
                "idx": idx,
                "freq_hz": freq_hz,
                "lam": lam,
                "dom_axis": dom_axis,
                "noz_rel": noz_rel,
                "is_isolated_panel_drum": is_isolated_panel_drum,
                "is_bed_z_mode": is_bed_z_mode,
                "rec_x": rec_x,
                "rec_y": rec_y,
            }
        )

    rx_max = max((c["rec_x"] for c in struct_candidates), default=0.0)
    ry_max = max((c["rec_y"] for c in struct_candidates), default=0.0)
    filtered = [
        c
        for c in struct_candidates
        if not c["is_isolated_panel_drum"]
        and not (
            c["dom_axis"] == "Z"
            and (
                bool(c["is_bed_z_mode"])
                or float(max(c["noz_rel"][0], c["noz_rel"][1])) < 0.15
                or (
                    rx_max > 1e-18
                    and ry_max > 1e-18
                    and c["rec_x"] < 0.03 * rx_max
                    and c["rec_y"] < 0.03 * ry_max
                )
            )
        )
    ]
    if not filtered:
        filtered = struct_candidates

    # Match modal_res.fundamental_freq_hz if supplied, else lowest filtered structural mode
    if modal_res is not None and modal_res.fundamental_freq_hz > 0.0:
        c_fund = min(filtered, key=lambda c: abs(c["freq_hz"] - modal_res.fundamental_freq_hz))
    else:
        c_fund = filtered[0]

    # Pick primary X and primary Y structural modes (max compliance-weighted residue)
    c_x_mode = max(
        filtered,
        key=lambda c: c["rec_x"] / max(c["freq_hz"] ** 2, 1.0),
    )
    c_y_mode = max(
        filtered,
        key=lambda c: c["rec_y"] / max(c["freq_hz"] ** 2, 1.0),
    )

    idx_fund = int(c_fund["idx"])
    idx_x = int(c_x_mode["idx"])
    idx_y = int(c_y_mode["idx"])

    f_fund = max(float(c_fund["freq_hz"]), 1e-6)
    f_x = max(float(c_x_mode["freq_hz"]), 1e-6)
    f_y = max(float(c_y_mode["freq_hz"]), 1e-6)
    eig_fund = float(c_fund["lam"])
    eig_x = float(c_x_mode["lam"])
    eig_y = float(c_y_mode["lam"])

    phi_fund = _dailey_subspace_vector(eigvals, full_vecs, idx_fund, fx_exc + fy_exc)
    phi_x = _dailey_subspace_vector(eigvals, full_vecs, idx_x, fx_exc)
    phi_y = _dailey_subspace_vector(eigvals, full_vecs, idx_y, fy_exc)

    inv_8pi2_fund = 1.0 / (8.0 * (math.pi**2) * f_fund)
    inv_8pi2_x = 1.0 / (8.0 * (math.pi**2) * f_x)
    inv_8pi2_y = 1.0 / (8.0 * (math.pi**2) * f_y)

    # 3. Element-level exact Fox-Kapoor & Adjoint evaluation
    nmap = assembly.node_map
    raw_members: list[dict[str, Any]] = []

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
        Ke = t_mat.T @ k_loc @ t_mat
        Me = t_mat.T @ m_loc @ t_mat

        dofs = [6 * i0 + d for d in range(6)] + [6 * i1 + d for d in range(6)]

        # Static adjoint derivatives
        dux = -float(lam_x[dofs] @ Ke @ u_x[dofs])
        duy = -float(lam_y[dofs] @ Ke @ u_y[dofs])
        comp_raw = max(0.0, -dux / max(abs(u_rel_x), 1e-12)) + max(
            0.0, -duy / max(abs(u_rel_y), 1e-12)
        )

        # Modal Fox-Kapoor derivatives
        pf = phi_fund[dofs]
        px = phi_x[dofs]
        py = phi_y[dofs]

        stiff_f = float(pf @ Ke @ pf) * inv_8pi2_fund
        mass_f = -eig_fund * float(pf @ Me @ pf) * inv_8pi2_fund
        stiff_x = float(px @ Ke @ px) * inv_8pi2_x
        mass_x = -eig_x * float(px @ Me @ px) * inv_8pi2_x
        stiff_y = float(py @ Ke @ py) * inv_8pi2_y
        mass_y = -eig_y * float(py @ Me @ py) * inv_8pi2_y

        m_exact, m_ref, cost_usd, cost_per_kg = _member_mass_and_cost(mem, length, assembly)
        net_f = stiff_f + mass_f
        df_dm = net_f / max(m_ref, 1e-6)

        raw_members.append(
            {
                "member_id": mem.id,
                "role": mem.role,
                "mass_kg": m_exact,
                "ref_mass_kg": m_ref,
                "cost_usd": cost_usd,
                "cost_per_kg": cost_per_kg,
                "stiffness_sens_hz": stiff_f,
                "mass_sens_hz": mass_f,
                "stiffness_sens_x_hz": stiff_x,
                "mass_sens_x_hz": mass_x,
                "stiffness_sens_y_hz": stiff_y,
                "mass_sens_y_hz": mass_y,
                "du_rel_x_dlnE_m": dux,
                "du_rel_y_dlnE_m": duy,
                "du_rel_dlnE_m": dux + duy,
                "hz_per_100g": 0.1 * df_dm,
                "hz_per_usd": df_dm / max(cost_per_kg, 0.5),
                "_comp_raw": comp_raw,
                "_net_sens": net_f,
            }
        )

    # Shell panel elements grouped by canonical bay ('left', 'right', 'rear', 'top')
    bay_acc: dict[str, dict[str, float]] = {
        bay: {
            "mass_kg": 0.0,
            "cost_usd": 0.0,
            "cost_per_kg": 0.0,
            "stiffness_sens_hz": 0.0,
            "mass_sens_hz": 0.0,
            "stiffness_sens_x_hz": 0.0,
            "mass_sens_x_hz": 0.0,
            "stiffness_sens_y_hz": 0.0,
            "mass_sens_y_hz": 0.0,
            "du_rel_x_dlnE_m": 0.0,
            "du_rel_y_dlnE_m": 0.0,
            "du_rel_dlnE_m": 0.0,
            "_comp_raw": 0.0,
            "n_elements": 0,
        }
        for bay in ("left", "right", "rear", "top")
    }

    for sh in assembly.shell_elements:
        bay_key = sh.bay.split("_")[0]
        if bay_key not in bay_acc:
            bay_acc[bay_key] = {k: 0.0 for k in bay_acc["left"]}
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
        dux = -float(lam_x[dofs] @ k_sh @ u_x[dofs])
        duy = -float(lam_y[dofs] @ k_sh @ u_y[dofs])
        comp_raw = max(0.0, -dux / max(abs(u_rel_x), 1e-12)) + max(
            0.0, -duy / max(abs(u_rel_y), 1e-12)
        )

        pf = phi_fund[dofs]
        px = phi_x[dofs]
        py = phi_y[dofs]
        b = bay_acc[bay_key]
        m_sh_kg = float(np.sum(np.diag(m_sh)[0::6]))
        b["mass_kg"] += m_sh_kg
        b["cost_usd"] += m_sh_kg * sh.material.cost_per_kg_usd
        b["cost_per_kg"] = max(sh.material.cost_per_kg_usd, 0.5)
        b["stiffness_sens_hz"] += float(pf @ k_sh @ pf) * inv_8pi2_fund
        b["mass_sens_hz"] += -eig_fund * float(pf @ m_sh @ pf) * inv_8pi2_fund
        b["stiffness_sens_x_hz"] += float(px @ k_sh @ px) * inv_8pi2_x
        b["mass_sens_x_hz"] += -eig_x * float(px @ m_sh @ px) * inv_8pi2_x
        b["stiffness_sens_y_hz"] += float(py @ k_sh @ py) * inv_8pi2_y
        b["mass_sens_y_hz"] += -eig_y * float(py @ m_sh @ py) * inv_8pi2_y
        b["du_rel_x_dlnE_m"] += dux
        b["du_rel_y_dlnE_m"] += duy
        b["du_rel_dlnE_m"] += dux + duy
        b["_comp_raw"] += comp_raw
        b["n_elements"] += 1

    # Fastener springs folded into their corresponding bay compliance & stiffness sens
    for sp in assembly.fastener_springs:
        bay_key = sp.bay.split("_")[0]
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
        dux = -float(lam_x[dofs] @ k_sp @ u_x[dofs])
        duy = -float(lam_y[dofs] @ k_sp @ u_y[dofs])
        comp_raw = max(0.0, -dux / max(abs(u_rel_x), 1e-12)) + max(
            0.0, -duy / max(abs(u_rel_y), 1e-12)
        )
        if bay_key in bay_acc:
            b = bay_acc[bay_key]
            b["du_rel_x_dlnE_m"] += dux
            b["du_rel_y_dlnE_m"] += duy
            b["du_rel_dlnE_m"] += dux + duy
            b["_comp_raw"] += comp_raw
            pf = phi_fund[dofs]
            b["stiffness_sens_hz"] += float(pf @ k_sp @ pf) * inv_8pi2_fund

    # Normalize compliance_share_pct across all structural elements (members + bays) to sum to 100.0%
    tot_comp_raw = sum(m["_comp_raw"] for m in raw_members) + sum(
        b["_comp_raw"] for b in bay_acc.values()
    )
    comp_scale = 100.0 / max(tot_comp_raw, 1e-18)

    # Compute normalized heat_score in [-1, +1]
    all_net = [m["_net_sens"] for m in raw_members] + [
        b["stiffness_sens_hz"] + b["mass_sens_hz"] for b in bay_acc.values() if b["n_elements"] > 0
    ]
    max_pos_net = max([v for v in all_net if v > 0.0], default=1e-9)
    max_neg_mag = max([-v for v in all_net if v < 0.0], default=1e-9)

    def _heat_score(net_val: float) -> float:
        if net_val > 0.0:
            return float(min(1.0, net_val / max(max_pos_net, 1e-12)))
        if net_val < 0.0:
            return float(max(-1.0, net_val / max(max_neg_mag, 1e-12)))
        return 0.0

    member_sensitivities: dict[str, dict[str, Any]] = {}
    for m in raw_members:
        comp_pct = m["_comp_raw"] * comp_scale
        hs = _heat_score(m["_net_sens"])
        member_sensitivities[m["member_id"]] = {
            "member_id": m["member_id"],
            "role": m["role"],
            "mass_kg": round(m["mass_kg"], 5),
            "cost_usd": round(m["cost_usd"], 3),
            "hz_per_100g": round(m["hz_per_100g"], 4),
            "hz_per_usd": round(m["hz_per_usd"], 4),
            "stiffness_sens_hz": float(m["stiffness_sens_hz"]),
            "mass_sens_hz": float(m["mass_sens_hz"]),
            "stiffness_sens_x_hz": float(m["stiffness_sens_x_hz"]),
            "mass_sens_x_hz": float(m["mass_sens_x_hz"]),
            "stiffness_sens_y_hz": float(m["stiffness_sens_y_hz"]),
            "mass_sens_y_hz": float(m["mass_sens_y_hz"]),
            "du_rel_x_dlnE_m": float(m["du_rel_x_dlnE_m"]),
            "du_rel_y_dlnE_m": float(m["du_rel_y_dlnE_m"]),
            "du_rel_dlnE_m": float(m["du_rel_dlnE_m"]),
            "compliance_share_pct": round(comp_pct, 3),
            "heat_score": round(hs, 4),
        }

    panel_bay_sensitivities: dict[str, dict[str, Any]] = {}
    for bay_key, b in bay_acc.items():
        if b["n_elements"] > 0 and b["mass_kg"] > 1e-6:
            net_b = b["stiffness_sens_hz"] + b["mass_sens_hz"]
            df_dm_b = net_b / b["mass_kg"]
            hz_100g_b = 0.1 * df_dm_b
            hz_usd_b = df_dm_b / max(b["cost_per_kg"], 0.5)
            hs_b = _heat_score(net_b)
        else:
            hz_100g_b = 0.0
            hz_usd_b = 0.0
            hs_b = 0.0
        comp_pct_b = b["_comp_raw"] * comp_scale
        panel_bay_sensitivities[bay_key] = {
            "bay": bay_key,
            "n_elements": int(b["n_elements"]),
            "mass_kg": round(b["mass_kg"], 4),
            "cost_usd": round(b["cost_usd"], 3),
            "hz_per_100g": round(hz_100g_b, 4),
            "hz_per_usd": round(hz_usd_b, 4),
            "stiffness_sens_hz": float(b["stiffness_sens_hz"]),
            "mass_sens_hz": float(b["mass_sens_hz"]),
            "stiffness_sens_x_hz": float(b["stiffness_sens_x_hz"]),
            "mass_sens_x_hz": float(b["mass_sens_x_hz"]),
            "stiffness_sens_y_hz": float(b["stiffness_sens_y_hz"]),
            "mass_sens_y_hz": float(b["mass_sens_y_hz"]),
            "du_rel_x_dlnE_m": float(b["du_rel_x_dlnE_m"]),
            "du_rel_y_dlnE_m": float(b["du_rel_y_dlnE_m"]),
            "du_rel_dlnE_m": float(b["du_rel_dlnE_m"]),
            "compliance_share_pct": round(comp_pct_b, 3),
            "heat_score": round(hs_b, 4),
        }

    # 4. Lumped nodal mass sensitivities (for toolhead, carriages, motors, bed)
    lumped_mass_sensitivities: dict[str, dict[str, Any]] = {}
    for i, node in enumerate(assembly.nodes):
        if node.lumped_mass_kg <= 0.0 or node.fixed:
            continue
        base = 6 * i
        rot_i = node.lumped_mass_kg * (0.025**2)
        m_diag_nd = np.array(
            [node.lumped_mass_kg, node.lumped_mass_kg, node.lumped_mass_kg, rot_i, rot_i, rot_i],
            dtype=float,
        )
        pf = phi_fund[base : base + 6]
        px = phi_x[base : base + 6]
        py = phi_y[base : base + 6]
        ms_f = -eig_fund * float(np.sum(m_diag_nd * (pf**2))) * inv_8pi2_fund
        ms_x = -eig_x * float(np.sum(m_diag_nd * (px**2))) * inv_8pi2_x
        ms_y = -eig_y * float(np.sum(m_diag_nd * (py**2))) * inv_8pi2_y
        lumped_mass_sensitivities[node.id] = {
            "node_id": node.id,
            "role": node.role,
            "mass_kg": round(node.lumped_mass_kg, 5),
            "mass_sens_hz": float(ms_f),
            "mass_sens_x_hz": float(ms_x),
            "mass_sens_y_hz": float(ms_y),
            "hz_per_100g": round(0.1 * ms_f / max(node.lumped_mass_kg, 1e-5), 4),
        }

    # 5. Top 5 stiffening ROI hotspots & Top 3 weight-reduction hotspots
    stiffen_pool: list[dict[str, Any]] = []
    for mid, d in member_sensitivities.items():
        if d["stiffness_sens_hz"] + d["mass_sens_hz"] > 0.0 or d["compliance_share_pct"] > 0.5:
            if d["hz_per_usd"] > 0.0 or d["compliance_share_pct"] > 1.0:
                stiffen_pool.append(
                    {
                        "id": mid,
                        "role": d["role"],
                        "hz_per_usd": round(d["hz_per_usd"], 4),
                        "compliance_share_pct": round(d["compliance_share_pct"], 2),
                        "action_hint": _stiffen_hint(mid, d["role"]),
                    }
                )
    for bay_key, b in panel_bay_sensitivities.items():
        if b["n_elements"] > 0 and (b["hz_per_usd"] > 0.0 or b["compliance_share_pct"] > 0.5):
            stiffen_pool.append(
                {
                    "id": f"bay_{bay_key}",
                    "role": "panel",
                    "hz_per_usd": round(b["hz_per_usd"], 4),
                    "compliance_share_pct": round(b["compliance_share_pct"], 2),
                    "action_hint": _stiffen_hint(f"bay_{bay_key}", "panel"),
                }
            )
    stiffen_pool.sort(
        key=lambda x: (x["hz_per_usd"], x["compliance_share_pct"]),
        reverse=True,
    )
    top_stiffen_targets = stiffen_pool[:5]

    lighten_pool: list[dict[str, Any]] = []
    for mid, d in member_sensitivities.items():
        if d["mass_sens_hz"] < -1e-4 and (d["stiffness_sens_hz"] + d["mass_sens_hz"]) < 0.0:
            lighten_pool.append(
                {
                    "id": mid,
                    "role": d["role"],
                    "mass_sens_hz": round(d["mass_sens_hz"], 4),
                    "hz_per_100g": round(d["hz_per_100g"], 4),
                    "action_hint": _lighten_hint(mid, d["role"]),
                }
            )
    # Include moving lumped-mass bodies (toolhead assembly + carriages, plus bed on bedslingers)
    th_group_ids = [
        nid
        for nid in (
            assembly.toolhead_node_id,
            assembly.toolhead_cog_node_id,
            assembly.nozzle_node_id,
        )
        if nid in lumped_mass_sensitivities
    ]
    th_unique = list(dict.fromkeys(th_group_ids))
    if th_unique:
        th_ms = sum(lumped_mass_sensitivities[nid]["mass_sens_hz"] for nid in th_unique)
        th_kg = sum(lumped_mass_sensitivities[nid]["mass_kg"] for nid in th_unique)
        if th_ms < -1e-4 and th_kg > 1e-5:
            lighten_pool.append(
                {
                    "id": "toolhead_assembly",
                    "role": "toolhead",
                    "mass_sens_hz": round(th_ms, 4),
                    "hz_per_100g": round(0.1 * th_ms / th_kg, 4),
                    "action_hint": _lighten_hint("toolhead_assembly", "toolhead"),
                }
            )
    for nid, ld in lumped_mass_sensitivities.items():
        if nid in th_unique:
            continue
        if ld["role"] == "bed" and assembly.spec.architecture != "cartesian_bedslinger":
            continue
        if ld["role"] in ("gantry", "bed") and ld["mass_sens_hz"] < -1e-4:
            lighten_pool.append(
                {
                    "id": nid,
                    "role": ld["role"],
                    "mass_sens_hz": round(ld["mass_sens_hz"], 4),
                    "hz_per_100g": round(ld["hz_per_100g"], 4),
                    "action_hint": _lighten_hint(nid, ld["role"]),
                }
            )

    lighten_pool.sort(key=lambda x: x["mass_sens_hz"])  # most negative first
    top_lighten_targets = lighten_pool[:3]

    return SensitivityReport(
        fundamental_freq_hz=f_fund,
        primary_x_freq_hz=f_x,
        primary_y_freq_hz=f_y,
        static_rel_disp_x_um=u_rel_x * 1e6,
        static_rel_disp_y_um=u_rel_y * 1e6,
        member_sensitivities=member_sensitivities,
        panel_bay_sensitivities=panel_bay_sensitivities,
        lumped_mass_sensitivities=lumped_mass_sensitivities,
        top_stiffen_targets=top_stiffen_targets,
        top_lighten_targets=top_lighten_targets,
    )
