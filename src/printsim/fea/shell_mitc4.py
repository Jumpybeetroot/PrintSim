"""24-DOF Bathe & Dvorkin (1984) MITC4 flat shell element + 12-DOF Huth fastener spring."""

from __future__ import annotations

import math

import numpy as np

_GP_COORDS: tuple[float, float] = (-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0))
_NODE_XI: tuple[float, ...] = (-1.0, 1.0, 1.0, -1.0)
_NODE_ETA: tuple[float, ...] = (-1.0, -1.0, 1.0, 1.0)
# Hughes-Brezzi / MacNeal-Harder frame-invariant drilling stabilization factor
_ALPHA_DRILL: float = 1e-6


def mitc4_local_frame(coords_3d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute orthonormal local frame R (rows e1, e2, e3), 2D planar coords, and centroid.

    Parameters
    ----------
    coords_3d:
        Array of shape ``(4, 3)`` containing quad corner coordinates ``[p0, p1, p2, p3]``
        ordered counter-clockwise about the local shell normal ``e3``.
    """
    pts = np.asarray(coords_3d, dtype=float)
    if pts.shape != (4, 3):
        raise ValueError(f"Expected coords_3d of shape (4, 3), got {pts.shape}")

    pc = 0.25 * np.sum(pts, axis=0)
    d02 = pts[2] - pts[0]
    d13 = pts[3] - pts[1]
    normal = np.cross(d02, d13)
    norm_len = float(np.linalg.norm(normal))
    if norm_len < 1e-12:
        raise ValueError("Degenerate zero-area quad element in mitc4_local_frame")
    e3 = normal / norm_len

    v01 = pts[1] - pts[0]
    v01_proj = v01 - float(np.dot(v01, e3)) * e3
    e1_len = float(np.linalg.norm(v01_proj))
    if e1_len < 1e-12:
        raise ValueError("Degenerate edge 0-1 in mitc4_local_frame")
    e1 = v01_proj / e1_len
    e2 = np.cross(e3, e1)
    e2 /= max(float(np.linalg.norm(e2)), 1e-12)

    r_mat = np.vstack([e1, e2, e3])
    xy = (pts - pc) @ r_mat[:2].T
    return r_mat, xy, pc


def _shape_and_derivs(xi: float, eta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_vals = np.array(
        [0.25 * (1.0 + _NODE_XI[k] * xi) * (1.0 + _NODE_ETA[k] * eta) for k in range(4)],
        dtype=float,
    )
    dn_dxi = np.array(
        [0.25 * _NODE_XI[k] * (1.0 + _NODE_ETA[k] * eta) for k in range(4)],
        dtype=float,
    )
    dn_deta = np.array(
        [0.25 * _NODE_ETA[k] * (1.0 + _NODE_XI[k] * xi) for k in range(4)],
        dtype=float,
    )
    return n_vals, dn_dxi, dn_deta


def _transformation_24x24(r_mat: np.ndarray) -> np.ndarray:
    t_mat = np.zeros((24, 24), dtype=float)
    for blk in range(8):
        s = 3 * blk
        t_mat[s : s + 3, s : s + 3] = r_mat
    return t_mat


def _plane_stress_constitutive(
    e_pa: float,
    nu: float,
    thickness_m: float,
    d_bend_nm: float | None = None,
    g_transverse_h_n_m: float | None = None,
    e_mem_h_n_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    c_iso = np.array(
        [
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, 0.5 * (1.0 - nu)],
        ],
        dtype=float,
    )
    e_mem_h = float(e_mem_h_n_m) if e_mem_h_n_m is not None else float(e_pa * thickness_m)
    d_bend = (
        float(d_bend_nm)
        if d_bend_nm is not None
        else float(e_pa * (thickness_m**3) / (12.0 * (1.0 - nu**2)))
    )
    if g_transverse_h_n_m is not None:
        ds_val = float(g_transverse_h_n_m)
    else:
        g_pa = e_pa / (2.0 * (1.0 + nu))
        ds_val = (5.0 / 6.0) * g_pa * thickness_m

    d_m = (e_mem_h / (1.0 - nu**2)) * c_iso
    d_b = d_bend * c_iso
    d_s = ds_val * np.eye(2, dtype=float)
    return d_m, d_b, d_s, e_mem_h


def mitc4_shell_matrices(
    coords_3d: np.ndarray,
    e_pa: float,
    nu: float,
    thickness_m: float,
    areal_mass_kg_m2: float,
    d_bend_nm: float | None = None,
    g_transverse_h_n_m: float | None = None,
    e_mem_h_n_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute exact 24x24 global stiffness, HRZ mass, and gravity load for a 4-node MITC4 shell.

    Local DOF ordering per node ``k = 0..3``:
      ``[u_k, v_k, w_k, theta_xk, theta_yk, theta_zk]``
    Right-hand-rule Mindlin-Reissner sign convention matching ``beam3d.py``:
      ``beta_x = +theta_y``, ``beta_y = -theta_x``,
      ``kappa_x = d(theta_y)/dx``, ``kappa_y = -d(theta_x)/dy``,
      ``kappa_xy = d(theta_y)/dy - d(theta_x)/dx``,
      ``gamma_xz = dw/dx + theta_y``, ``gamma_yz = dw/dy - theta_x``.
    """
    if thickness_m <= 0.0:
        raise ValueError("thickness_m must be positive")
    if e_pa <= 0.0:
        raise ValueError("e_pa must be positive")
    if not (-0.5 < nu < 0.5):
        raise ValueError(f"Poisson ratio nu={nu} out of valid range (-0.5, 0.5)")

    r_mat, xy, _ = mitc4_local_frame(coords_3d)
    x = xy[:, 0]
    y = xy[:, 1]
    t = float(thickness_m)

    d_m, d_b, d_s, e_mem_h = _plane_stress_constitutive(
        e_pa=e_pa,
        nu=nu,
        thickness_m=t,
        d_bend_nm=d_bend_nm,
        g_transverse_h_n_m=g_transverse_h_n_m,
        e_mem_h_n_m=e_mem_h_n_m,
    )

    # Bathe & Dvorkin (1984) MITC4 covariant transverse shear tying rows on [w_k, thx_k, thy_k] (12 DOFs)
    # Tying points: A=(0,-1) on edge 0->1, B=(1,0) on edge 1->2, C=(0,1) on edge 3->2, D=(-1,0) on edge 0->3
    g_a = np.zeros(12, dtype=float)
    g_a[0], g_a[3] = -0.5, 0.5
    g_a[1] = g_a[4] = -0.25 * (y[1] - y[0])
    g_a[2] = g_a[5] = 0.25 * (x[1] - x[0])

    g_c = np.zeros(12, dtype=float)
    g_c[9], g_c[6] = -0.5, 0.5
    g_c[10] = g_c[7] = -0.25 * (y[2] - y[3])
    g_c[11] = g_c[8] = 0.25 * (x[2] - x[3])

    g_d = np.zeros(12, dtype=float)
    g_d[0], g_d[9] = -0.5, 0.5
    g_d[1] = g_d[10] = -0.25 * (y[3] - y[0])
    g_d[2] = g_d[11] = 0.25 * (x[3] - x[0])

    g_b = np.zeros(12, dtype=float)
    g_b[3], g_b[6] = -0.5, 0.5
    g_b[4] = g_b[7] = -0.25 * (y[2] - y[1])
    g_b[5] = g_b[8] = 0.25 * (x[2] - x[1])

    k_m = np.zeros((8, 8), dtype=float)
    k_b = np.zeros((12, 12), dtype=float)
    k_drill = np.zeros((12, 12), dtype=float)
    gt_mem = float(d_m[2, 2])
    m_shape_diag = np.zeros(4, dtype=float)
    area = 0.0

    for xi in _GP_COORDS:
        for eta in _GP_COORDS:
            n_vals, dn_dxi, dn_deta = _shape_and_derivs(xi, eta)
            j_mat = np.array(
                [
                    [float(np.dot(dn_dxi, x)), float(np.dot(dn_dxi, y))],
                    [float(np.dot(dn_deta, x)), float(np.dot(dn_deta, y))],
                ],
                dtype=float,
            )
            det_j = float(np.linalg.det(j_mat))
            if det_j <= 1e-14:
                raise ValueError(f"Invalid / non-CCW quad Jacobian detJ={det_j:.3e}")
            area += det_j
            inv_j = np.linalg.inv(j_mat)
            dn_dx_dy = inv_j @ np.vstack([dn_dxi, dn_deta])
            dn_dx, dn_dy = dn_dx_dy[0], dn_dx_dy[1]

            b_m = np.zeros((3, 8), dtype=float)
            b_b = np.zeros((3, 12), dtype=float)
            b_d = np.zeros(12, dtype=float)
            for k in range(4):
                b_m[0, 2 * k] = dn_dx[k]
                b_m[1, 2 * k + 1] = dn_dy[k]
                b_m[2, 2 * k] = dn_dy[k]
                b_m[2, 2 * k + 1] = dn_dx[k]

                b_b[0, 3 * k + 2] = dn_dx[k]
                b_b[1, 3 * k + 1] = -dn_dy[k]
                b_b[2, 3 * k + 1] = -dn_dx[k]
                b_b[2, 3 * k + 2] = dn_dy[k]

                # Hughes-Brezzi / MacNeal-Harder frame-invariant drilling strain:
                # theta_z - omega_z = sum_k [ 0.5*dN_k/dy * u_k - 0.5*dN_k/dx * v_k + N_k * theta_zk ]
                b_d[3 * k] = 0.5 * dn_dy[k]
                b_d[3 * k + 1] = -0.5 * dn_dx[k]
                b_d[3 * k + 2] = n_vals[k]

            g_cov = np.vstack(
                [
                    0.5 * (1.0 - eta) * g_a + 0.5 * (1.0 + eta) * g_c,
                    0.5 * (1.0 - xi) * g_d + 0.5 * (1.0 + xi) * g_b,
                ]
            )
            b_s = inv_j @ g_cov

            k_m += (b_m.T @ d_m @ b_m) * det_j
            k_b += (b_b.T @ d_b @ b_b + b_s.T @ d_s @ b_s) * det_j
            k_drill += (_ALPHA_DRILL * gt_mem) * np.outer(b_d, b_d) * det_j
            m_shape_diag += (n_vals * n_vals) * det_j

    k_loc = np.zeros((24, 24), dtype=float)
    mem_dofs = [6 * k + d for k in range(4) for d in (0, 1)]
    bend_dofs = [6 * k + d for k in range(4) for d in (2, 3, 4)]
    drill_dofs = [6 * k + d for k in range(4) for d in (0, 1, 5)]
    k_loc[np.ix_(mem_dofs, mem_dofs)] = k_m
    k_loc[np.ix_(bend_dofs, bend_dofs)] = k_b
    k_loc[np.ix_(drill_dofs, drill_dofs)] += k_drill
    k_loc = 0.5 * (k_loc + k_loc.T)

    # HRZ lumped mass matrix with exact total mass and rotary inertia (1/12)*m_node*t^2
    total_mass = max(float(areal_mass_kg_m2) * area, 0.0)
    m_nodes = total_mass * (m_shape_diag / max(float(np.sum(m_shape_diag)), 1e-18))
    m_loc = np.zeros((24, 24), dtype=float)
    for k in range(4):
        mn = float(m_nodes[k])
        m_loc[6 * k + 0, 6 * k + 0] = mn
        m_loc[6 * k + 1, 6 * k + 1] = mn
        m_loc[6 * k + 2, 6 * k + 2] = mn
        rot_i = mn * (t**2) / 12.0
        m_loc[6 * k + 3, 6 * k + 3] = rot_i
        m_loc[6 * k + 4, 6 * k + 4] = rot_i
        m_loc[6 * k + 5, 6 * k + 5] = max(1e-6 * rot_i, 1e-12)

    t_mat = _transformation_24x24(r_mat)
    k_glob = t_mat.T @ k_loc @ t_mat
    m_glob = t_mat.T @ m_loc @ t_mat
    k_glob = 0.5 * (k_glob + k_glob.T)
    m_glob = 0.5 * (m_glob + m_glob.T)

    f_grav = np.zeros(24, dtype=float)
    nodal_w = -0.25 * total_mass * 9.80665
    for k in range(4):
        f_grav[6 * k + 2] = nodal_w

    return k_glob, m_glob, f_grav


def mitc4_shell_von_mises_mpa(
    coords_3d: np.ndarray,
    u_elem_global: np.ndarray,
    e_pa: float,
    nu: float,
    thickness_m: float,
    d_bend_nm: float | None = None,
    e_mem_h_n_m: float | None = None,
) -> float:
    """Recover peak top/bottom surface von Mises stress (MPa) at element center (0, 0)."""
    r_mat, xy, _ = mitc4_local_frame(coords_3d)
    x = xy[:, 0]
    y = xy[:, 1]
    t = float(thickness_m)

    d_m, d_b, _, _ = _plane_stress_constitutive(
        e_pa=e_pa,
        nu=nu,
        thickness_m=t,
        d_bend_nm=d_bend_nm,
        e_mem_h_n_m=e_mem_h_n_m,
    )

    _, dn_dxi, dn_deta = _shape_and_derivs(0.0, 0.0)
    j_mat = np.array(
        [
            [float(np.dot(dn_dxi, x)), float(np.dot(dn_dxi, y))],
            [float(np.dot(dn_deta, x)), float(np.dot(dn_deta, y))],
        ],
        dtype=float,
    )
    inv_j = np.linalg.inv(j_mat)
    dn_dx_dy = inv_j @ np.vstack([dn_dxi, dn_deta])
    dn_dx, dn_dy = dn_dx_dy[0], dn_dx_dy[1]

    b_m = np.zeros((3, 8), dtype=float)
    b_b = np.zeros((3, 12), dtype=float)
    for k in range(4):
        b_m[0, 2 * k] = dn_dx[k]
        b_m[1, 2 * k + 1] = dn_dy[k]
        b_m[2, 2 * k] = dn_dy[k]
        b_m[2, 2 * k + 1] = dn_dx[k]

        b_b[0, 3 * k + 2] = dn_dx[k]
        b_b[1, 3 * k + 1] = -dn_dy[k]
        b_b[2, 3 * k + 1] = -dn_dx[k]
        b_b[2, 3 * k + 2] = dn_dy[k]

    t_mat = _transformation_24x24(r_mat)
    u_loc = t_mat @ np.asarray(u_elem_global, dtype=float)
    mem_dofs = [6 * k + d for k in range(4) for d in (0, 1)]
    bend_dofs = [6 * k + d for k in range(4) for d in (2, 3, 4)]

    n_res = d_m @ (b_m @ u_loc[mem_dofs])
    m_res = d_b @ (b_b @ u_loc[bend_dofs])

    sig_mem = n_res / max(t, 1e-9)
    sig_bend = 6.0 * m_res / max(t * t, 1e-12)

    def _vm(sig: np.ndarray) -> float:
        sx, sy, txy = float(sig[0]), float(sig[1]), float(sig[2])
        return math.sqrt(max(sx * sx - sx * sy + sy * sy + 3.0 * txy * txy, 0.0))

    vm_pa = max(_vm(sig_mem + sig_bend), _vm(sig_mem - sig_bend))
    return vm_pa * 1e-6


def _skew(vec: np.ndarray) -> np.ndarray:
    rx, ry, rz = float(vec[0]), float(vec[1]), float(vec[2])
    return np.array(
        [
            [0.0, -rz, ry],
            [rz, 0.0, -rx],
            [-ry, rx, 0.0],
        ],
        dtype=float,
    )


def fastener_spring_matrices(
    k_shear_n_m: float,
    k_normal_n_m: float,
    k_rot_nm_rad: float,
    normal_vec: np.ndarray,
    offset_vec: np.ndarray | None = None,
) -> np.ndarray:
    """Build exact symmetric 12x12 two-node 3D CBUSH spring stiffness matrix.

    Node 0 is the shell perimeter node and Node 1 is the matching frame extrusion node.
    If ``offset_vec`` (``p_shell - p_frame`` in global meters) is nonzero, applies an
    exact rigid-body eccentric lever-arm transformation on Node 1 so rigid rotations
    have zero strain energy and parallel-axis composite flange bending is captured.
    """
    n_arr = np.asarray(normal_vec, dtype=float)
    n_len = float(np.linalg.norm(n_arr))
    if n_len < 1e-12:
        raise ValueError("normal_vec must be non-zero")
    e3 = n_arr / n_len
    ref = np.array([1.0, 0.0, 0.0], dtype=float) if abs(float(e3[0])) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=float)
    e1 = np.cross(ref, e3)
    e1 /= max(float(np.linalg.norm(e1)), 1e-12)
    e2 = np.cross(e3, e1)
    e2 /= max(float(np.linalg.norm(e2)), 1e-12)
    r_mat = np.vstack([e1, e2, e3])

    k_t = r_mat.T @ np.diag([float(k_shear_n_m), float(k_shear_n_m), float(k_normal_n_m)]) @ r_mat
    k_r = r_mat.T @ np.diag([float(k_rot_nm_rad), float(k_rot_nm_rad), float(k_rot_nm_rad)]) @ r_mat

    k12 = np.zeros((12, 12), dtype=float)
    k12[0:3, 0:3] = k_t
    k12[6:9, 6:9] = k_t
    k12[0:3, 6:9] = -k_t
    k12[6:9, 0:3] = -k_t

    k12[3:6, 3:6] = k_r
    k12[9:12, 9:12] = k_r
    k12[3:6, 9:12] = -k_r
    k12[9:12, 3:6] = -k_r

    if offset_vec is not None:
        off = np.asarray(offset_vec, dtype=float)
        if float(np.linalg.norm(off)) > 1e-12:
            b_off = np.eye(12, dtype=float)
            b_off[6:9, 9:12] = -_skew(off)
            k12 = b_off.T @ k12 @ b_off

    return 0.5 * (k12 + k12.T)
