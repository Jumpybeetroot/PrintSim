"""Exact 12-DOF 3D Euler-Bernoulli frame finite element matrices."""

from __future__ import annotations

import numpy as np

from printsim.geometry.builder import AssemblyModel, StructuralMember


def beam_rotation_matrix(p0: np.ndarray, p1: np.ndarray) -> tuple[np.ndarray, float]:
    """Compute 3x3 local-to-global rotation matrix R (rows = local axes ex, ey, ez) and length L."""
    vec = p1 - p0
    length = float(np.linalg.norm(vec))
    if length < 1e-9:
        raise ValueError("Zero-length structural member encountered")

    ex = vec / length
    # Choose reference up vector not collinear with ex
    global_z = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(ex, global_z))) > 0.99:
        ref = np.array([0.0, 1.0, 0.0])
    else:
        ref = global_z

    ey = np.cross(ref, ex)
    ey /= np.linalg.norm(ey)
    ez = np.cross(ex, ey)
    ez /= np.linalg.norm(ez)

    r_mat = np.vstack([ex, ey, ez])  # local = R @ global
    return r_mat, length


def transformation_12x12(r_mat: np.ndarray) -> np.ndarray:
    """Build 12x12 block-diagonal rotation matrix T mapping global DOFs to local DOFs."""
    t_mat = np.zeros((12, 12), dtype=float)
    for k in range(4):
        s = 3 * k
        t_mat[s : s + 3, s : s + 3] = r_mat
    return t_mat


def local_beam_stiffness(
    e_pa: float,
    g_pa: float,
    area: float,
    iy: float,
    iz: float,
    j_torsion: float,
    length: float,
    bending_scale: float = 1.0,
) -> np.ndarray:
    """12x12 local stiffness matrix for a 3D Euler-Bernoulli beam element.

    Local DOF ordering:
      [u_x1, u_y1, u_z1, th_x1, th_y1, th_z1, u_x2, u_y2, u_z2, th_x2, th_y2, th_z2]
    """
    L = length
    L2 = L * L
    L3 = L2 * L
    k = np.zeros((12, 12), dtype=float)

    # Axial (u_x1, u_x2)
    ea_l = e_pa * area / L
    k[0, 0] = k[6, 6] = ea_l
    k[0, 6] = k[6, 0] = -ea_l

    # Torsion (th_x1, th_x2)
    gj_l = g_pa * j_torsion * bending_scale / L
    k[3, 3] = k[9, 9] = gj_l
    k[3, 9] = k[9, 3] = -gj_l

    # Bending in local x-y plane: (u_y1, th_z1, u_y2, th_z2), v'(x) = +th_z
    c_z = e_pa * iz * bending_scale / L3
    idx_xy = [1, 5, 7, 11]
    k_xy = c_z * np.array(
        [
            [12.0, 6.0 * L, -12.0, 6.0 * L],
            [6.0 * L, 4.0 * L2, -6.0 * L, 2.0 * L2],
            [-12.0, -6.0 * L, 12.0, -6.0 * L],
            [6.0 * L, 2.0 * L2, -6.0 * L, 4.0 * L2],
        ]
    )
    k[np.ix_(idx_xy, idx_xy)] += k_xy

    # Bending in local x-z plane: (u_z1, th_y1, u_z2, th_y2), w'(x) = -th_y
    c_y = e_pa * iy * bending_scale / L3
    idx_xz = [2, 4, 8, 10]
    k_xz = c_y * np.array(
        [
            [12.0, -6.0 * L, -12.0, -6.0 * L],
            [-6.0 * L, 4.0 * L2, 6.0 * L, 2.0 * L2],
            [-12.0, 6.0 * L, 12.0, 6.0 * L],
            [-6.0 * L, 2.0 * L2, 6.0 * L, 4.0 * L2],
        ]
    )
    k[np.ix_(idx_xz, idx_xz)] += k_xz

    return k


def local_beam_mass(
    rho: float,
    area: float,
    iy: float,
    iz: float,
    length: float,
) -> np.ndarray:
    """12x12 consistent mass matrix for a 3D Euler-Bernoulli beam element."""
    L = length
    L2 = L * L
    m_tot = rho * area * L
    m = np.zeros((12, 12), dtype=float)

    # Axial consistent mass
    m[0, 0] = m[6, 6] = m_tot / 3.0
    m[0, 6] = m[6, 0] = m_tot / 6.0

    # Torsional consistent mass (polar moment I_p = I_y + I_z)
    rx2 = (iy + iz) / area
    m[3, 3] = m[9, 9] = m_tot * rx2 / 3.0
    m[3, 9] = m[9, 3] = m_tot * rx2 / 6.0

    # Transverse bending in x-y plane: (u_y1, th_z1, u_y2, th_z2)
    c = m_tot / 420.0
    idx_xy = [1, 5, 7, 11]
    m_xy = c * np.array(
        [
            [156.0, 22.0 * L, 54.0, -13.0 * L],
            [22.0 * L, 4.0 * L2, 13.0 * L, -3.0 * L2],
            [54.0, 13.0 * L, 156.0, -22.0 * L],
            [-13.0 * L, -3.0 * L2, -22.0 * L, 4.0 * L2],
        ]
    )
    m[np.ix_(idx_xy, idx_xy)] += m_xy

    # Transverse bending in x-z plane: (u_z1, th_y1, u_z2, th_y2), w'(x) = -th_y
    idx_xz = [2, 4, 8, 10]
    m_xz = c * np.array(
        [
            [156.0, -22.0 * L, 54.0, 13.0 * L],
            [-22.0 * L, 4.0 * L2, -13.0 * L, -3.0 * L2],
            [54.0, -13.0 * L, 156.0, 22.0 * L],
            [13.0 * L, -3.0 * L2, 22.0 * L, 4.0 * L2],
        ]
    )
    m[np.ix_(idx_xz, idx_xz)] += m_xz

    return m


def assemble_global_matrices(
    assembly: AssemblyModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int], list[int]]:
    """Assemble global K, M, gravity load vector F_g, node index map, and free DOF list."""
    n_nodes = len(assembly.nodes)
    n_dof = 6 * n_nodes
    node_idx = {node.id: i for i, node in enumerate(assembly.nodes)}

    K = np.zeros((n_dof, n_dof), dtype=float)
    M = np.zeros((n_dof, n_dof), dtype=float)
    F_grav = np.zeros(n_dof, dtype=float)
    g_acc = 9.80665

    # Assemble beam elements
    for member in assembly.members:
        i0 = node_idx[member.start_id]
        i1 = node_idx[member.end_id]
        p0 = assembly.nodes[i0].position_m
        p1 = assembly.nodes[i1].position_m
        r_mat, length = beam_rotation_matrix(p0, p1)
        t_mat = transformation_12x12(r_mat)

        bend_scale = 0.08 if member.is_truss else 1.0
        k_loc = local_beam_stiffness(
            e_pa=member.material.youngs_modulus_pa,
            g_pa=member.material.effective_shear_modulus_pa,
            area=member.profile.area_m2,
            iy=member.profile.iy_m4,
            iz=member.profile.iz_m4,
            j_torsion=member.profile.j_m4,
            length=length,
            bending_scale=bend_scale,
        )
        m_loc = local_beam_mass(
            rho=member.material.density_kg_m3,
            area=member.profile.area_m2,
            iy=member.profile.iy_m4,
            iz=member.profile.iz_m4,
            length=length,
        )

        k_glob = t_mat.T @ k_loc @ t_mat
        m_glob = t_mat.T @ m_loc @ t_mat

        dofs = list(range(6 * i0, 6 * i0 + 6)) + list(range(6 * i1, 6 * i1 + 6))
        K[np.ix_(dofs, dofs)] += k_glob
        M[np.ix_(dofs, dofs)] += m_glob

        # Equivalent nodal gravity load (-Z)
        elem_weight = member.profile.area_m2 * length * member.material.density_kg_m3 * g_acc
        F_grav[6 * i0 + 2] -= 0.5 * elem_weight
        F_grav[6 * i1 + 2] -= 0.5 * elem_weight

    # Add lumped nodal masses (motors, toolhead, gantry carriages) + tiny rotational inertia
    for i, node in enumerate(assembly.nodes):
        if node.lumped_mass_kg > 0.0:
            base = 6 * i
            M[base, base] += node.lumped_mass_kg
            M[base + 1, base + 1] += node.lumped_mass_kg
            M[base + 2, base + 2] += node.lumped_mass_kg
            # Small rotational inertia for toolhead/motor body (approx radius ~25mm)
            rot_i = node.lumped_mass_kg * (0.025**2)
            M[base + 3, base + 3] += rot_i
            M[base + 4, base + 4] += rot_i
            M[base + 5, base + 5] += rot_i
            F_grav[base + 2] -= node.lumped_mass_kg * g_acc

    # Boundary conditions
    fixed_dofs: set[int] = set()
    for i, node in enumerate(assembly.nodes):
        if node.fixed:
            for d in range(6):
                fixed_dofs.add(6 * i + d)
        elif node.pinned:
            for d in range(3):
                fixed_dofs.add(6 * i + d)

    free_dofs = [d for d in range(n_dof) if d not in fixed_dofs]
    return K, M, F_grav, node_idx, free_dofs


def element_internal_forces(
    member: StructuralMember,
    p0: np.ndarray,
    p1: np.ndarray,
    u_elem_global: np.ndarray,
) -> dict[str, float]:
    """Compute local internal forces and peak combined axial + bending stress (Pa)."""
    r_mat, length = beam_rotation_matrix(p0, p1)
    t_mat = transformation_12x12(r_mat)
    bend_scale = 0.08 if member.is_truss else 1.0
    k_loc = local_beam_stiffness(
        e_pa=member.material.youngs_modulus_pa,
        g_pa=member.material.effective_shear_modulus_pa,
        area=member.profile.area_m2,
        iy=member.profile.iy_m4,
        iz=member.profile.iz_m4,
        j_torsion=member.profile.j_m4,
        length=length,
        bending_scale=bend_scale,
    )
    u_loc = t_mat @ u_elem_global
    f_loc = k_loc @ u_loc

    # f_loc = [Fx1, Fy1, Fz1, Mx1, My1, Mz1, Fx2, Fy2, Fz2, Mx2, My2, Mz2]
    axial_n = max(abs(float(f_loc[0])), abs(float(f_loc[6])))
    shear_y_n = max(abs(float(f_loc[1])), abs(float(f_loc[7])))
    shear_z_n = max(abs(float(f_loc[2])), abs(float(f_loc[8])))
    torque_nm = max(abs(float(f_loc[3])), abs(float(f_loc[9])))
    moment_y_nm = max(abs(float(f_loc[4])), abs(float(f_loc[10])))
    moment_z_nm = max(abs(float(f_loc[5])), abs(float(f_loc[11])))

    sigma_axial = axial_n / member.profile.area_m2
    sigma_bend_y = moment_y_nm * member.profile.max_c_z / member.profile.iy_m4
    sigma_bend_z = moment_z_nm * member.profile.max_c_y / member.profile.iz_m4
    sigma_normal = sigma_axial + sigma_bend_y + sigma_bend_z

    # Shear + torsional shear estimate
    tau_shear = 1.5 * np.hypot(shear_y_n, shear_z_n) / member.profile.area_m2
    r_eff = max(member.profile.max_c_y, member.profile.max_c_z)
    tau_torsion = torque_nm * r_eff / member.profile.j_m4
    tau_total = tau_shear + tau_torsion

    von_mises_pa = float(np.sqrt(sigma_normal**2 + 3.0 * tau_total**2))
    return {
        "axial_force_n": axial_n,
        "max_bending_moment_nm": float(np.hypot(moment_y_nm, moment_z_nm)),
        "torque_nm": torque_nm,
        "von_mises_mpa": von_mises_pa * 1e-6,
        "safety_factor": float(member.material.yield_strength_pa / max(von_mises_pa, 1.0)),
    }
