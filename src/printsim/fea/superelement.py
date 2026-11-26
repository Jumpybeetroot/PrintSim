"""Craig-Bampton Component Mode Synthesis (CMS) & Virtual Point Transformation (VPT).

Implements:
- Exact 6-DOF Virtual Point Transformation ``R_u`` and RBE3 weighted filter
  ``(R_u^T W R_u)^{-1} R_u^T W`` (van der Seijs et al., 2016) to couple 3D solid meshes
  to 6-DOF structural frame/beam nodes without artificial boss locking.
- Craig & Bampton (1968) fixed-interface substructuring supporting both dense NumPy
  and sparse SciPy ``(K, M)`` matrices.
- Multi-port VPT + CMS condensation for 3D solid parts and parametric Hex8 brackets
  with SHA256 disk/in-memory caching.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy import sparse
from scipy.linalg import block_diag, cho_factor, cho_solve, eigh, null_space, solve
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import LinearOperator, eigsh, splu

from printsim.fea.bracket_designer import _bracket_axis_spacings_mm, _bracket_layout_dims
from printsim.fea.solid_fem import (
    FaceName,
    constitutive_matrix_3d,
    hex8_element_mass,
    hex8_element_stiffness,
)
from printsim.schema import get_material

_BRACKET_CACHE: dict[str, "CraigBamptonSuperelement"] = {}


@dataclass
class CraigBamptonSuperelement:
    """Reduced Craig-Bampton dynamic superelement in strict SI units."""

    name: str
    n_boundary_dofs: int
    n_internal_modes: int
    K_cb: np.ndarray
    M_cb: np.ndarray
    T_cb: np.ndarray
    fixed_interface_freqs_hz: list[float]
    port_stiffness_diag: dict[str, float]
    mass_kg: float
    source_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of the superelement."""
        return {
            "name": self.name,
            "n_boundary_dofs": int(self.n_boundary_dofs),
            "n_internal_modes": int(self.n_internal_modes),
            "total_dofs": int(self.n_boundary_dofs + self.n_internal_modes),
            "fixed_interface_freqs_hz": [round(float(f), 2) for f in self.fixed_interface_freqs_hz],
            "port_stiffness_diag": {
                k: round(float(v), 3) for k, v in self.port_stiffness_diag.items()
            },
            "mass_kg": round(float(self.mass_kg), 6),
            "source_hash": self.source_hash,
        }


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


def virtual_point_transform(patch_coords_m: np.ndarray, vp_coord_m: np.ndarray) -> np.ndarray:
    """Build exact 3*n_p x 6 kinematic VPT matrix ``R_u`` mapping VP 6-DOFs to patch nodes.

    Each node ``j`` receives block ``[I_3, -[r_j - p_vp]_x]``, mapping virtual-point
    displacements/rotations ``[ux, uy, uz, rx, ry, rz]`` to 3D nodal translations
    ``u_j = u_vp + theta x (r_j - p_vp)``.
    """
    pts = np.asarray(patch_coords_m, dtype=float)
    if pts.ndim == 1 and pts.shape == (3,):
        pts = pts.reshape(1, 3)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
        raise ValueError(f"Expected patch_coords_m of shape (n_p, 3), got {pts.shape}")

    vp = np.asarray(vp_coord_m, dtype=float).reshape(3)
    d = pts - vp[None, :]
    n_p = pts.shape[0]
    R_u = np.zeros((3 * n_p, 6), dtype=float)

    # Identity blocks for translational DOFs
    R_u[0::3, 0] = 1.0
    R_u[1::3, 1] = 1.0
    R_u[2::3, 2] = 1.0

    # -[d]_x blocks for rotational DOFs: u = theta x d = -d x theta
    dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
    R_u[0::3, 4] = dz
    R_u[0::3, 5] = -dy
    R_u[1::3, 3] = -dz
    R_u[1::3, 5] = dx
    R_u[2::3, 3] = dy
    R_u[2::3, 4] = -dx
    return R_u


def virtual_point_filter_rbe3(
    R_u: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Compute weighted RBE3 virtual-point filter ``T_u = (R_u^T W R_u)^{-1} R_u^T W``.

    Maps 3*n_p patch nodal displacements to 6 virtual-point DOFs without rigidifying
    the interface (``T_u @ R_u == I_6`` for rigid body motions, while allowing local
    patch warping and Poisson relaxation).
    """
    R = np.asarray(R_u, dtype=float)
    n_dof, n_vp = R.shape
    if n_vp != 6:
        raise ValueError(f"Expected R_u with 6 columns, got {R.shape}")

    if weights is None:
        WR = R
    else:
        w = np.asarray(weights, dtype=float)
        if w.ndim == 1:
            if w.shape[0] == n_dof:
                w_diag = w
            elif w.shape[0] * 3 == n_dof:
                w_diag = np.repeat(w, 3)
            else:
                raise ValueError(
                    f"1D weights length {w.shape[0]} incompatible with R_u rows {n_dof}"
                )
            WR = w_diag[:, None] * R
        elif w.ndim == 2 and w.shape == (n_dof, n_dof):
            WR = w @ R
        else:
            raise ValueError(f"Incompatible weights shape {w.shape} for R_u shape {R.shape}")

    G = 0.5 * (R.T @ WR + (R.T @ WR).T)
    rhs = WR.T
    try:
        cond = float(np.linalg.cond(G))
        if np.isfinite(cond) and cond < 1e12:
            return solve(G, rhs, assume_a="sym")
    except np.linalg.LinAlgError:
        pass
    return np.linalg.pinv(G, rcond=1e-10) @ rhs


def _extract_port_stiffness_diag(K_cb: np.ndarray, n_boundary_dofs: int) -> dict[str, float]:
    """Extract effective 6-DOF port stiffnesses (N/m and Nm/rad).

    For multi-port superelements (``n_boundary_dofs >= 12``), grounds Port 0
    (``[0:6]``), statically condenses out internal modes and any extra DOFs,
    and returns ``1 / C_11[d, d]`` for Port 1 (``[6:12]``).
    """
    keys = (
        "k_x_n_m",
        "k_y_n_m",
        "k_z_n_m",
        "k_rx_nm_rad",
        "k_ry_nm_rad",
        "k_rz_nm_rad",
    )
    if n_boundary_dofs >= 12 and K_cb.shape[0] >= 12:
        K_free = 0.5 * (K_cb[6:, 6:] + K_cb[6:, 6:].T)
        n_free = K_free.shape[0]
        rhs = np.zeros((n_free, 6), dtype=float)
        rhs[:6, :6] = np.eye(6)
        # Diagonal Jacobi equilibration across translational (N/m), rotational (Nm/rad), and modal (rad^2/s^2) units
        diag_k = np.maximum(np.diag(K_free), 1e-18)
        s = 1.0 / np.sqrt(diag_k)
        K_scaled = (s[:, None] * K_free) * s[None, :]
        rhs_scaled = s[:, None] * rhs
        try:
            c_fac = cho_factor(K_scaled, check_finite=False)
            Y_scaled = cho_solve(c_fac, rhs_scaled, check_finite=False)
            C_full = s[:, None] * Y_scaled
        except np.linalg.LinAlgError:
            reg = 1e-10 * max(float(np.max(diag_k)), 1.0)
            C_full = np.linalg.pinv(K_free + reg * np.eye(n_free)) @ rhs
        C_11 = C_full[:6, :6]
        return {
            key: float(1.0 / max(float(C_11[i, i]), 1e-18))
            for i, key in enumerate(keys)
        }

    out: dict[str, float] = {}
    for i, key in enumerate(keys):
        if i < n_boundary_dofs and i < K_cb.shape[0]:
            out[key] = max(0.0, float(K_cb[i, i]))
        else:
            out[key] = 0.0
    return out


def _estimate_mass_from_matrix(M: np.ndarray | sparse.spmatrix) -> float:
    """Infer total translational mass (kg) from 3-DOF/node or 6-DOF/node mass matrix."""
    n_dof = M.shape[0]
    if n_dof == 0:
        return 0.0
    diag = np.asarray(M.diagonal(), dtype=float) if sparse.issparse(M) else np.diag(M)
    stride = 3
    if n_dof % 6 == 0:
        trans_med = float(np.median(np.abs(diag[0::6])))
        rot_med = float(np.median(np.abs(diag[3::6])))
        if trans_med > 0.0 and rot_med < 0.05 * trans_med:
            stride = 6
    r_x = np.zeros(n_dof, dtype=float)
    r_x[0::stride] = 1.0
    return float(r_x @ (M @ r_x))


def condense_craig_bampton(
    K: np.ndarray | sparse.spmatrix,
    M: np.ndarray | sparse.spmatrix,
    boundary_dofs: Sequence[int] | np.ndarray,
    n_modes: int = 8,
    name: str = "part",
    source_hash: str = "",
    mass_kg: float | None = None,
) -> CraigBamptonSuperelement:
    """Perform exact Craig & Bampton (1968) fixed-interface substructuring.

    Supports both dense NumPy arrays and SciPy sparse matrices. Preserves the exact
    ordering of ``boundary_dofs`` in the first ``n_b`` coordinates of ``K_cb, M_cb``.
    """
    n_total = K.shape[0]
    if K.shape != (n_total, n_total) or M.shape != (n_total, n_total):
        raise ValueError(f"K and M must be square with matching shape, got {K.shape} and {M.shape}")

    b_dofs = np.asarray(boundary_dofs, dtype=int).ravel()
    if len(np.unique(b_dofs)) != len(b_dofs):
        raise ValueError("boundary_dofs must not contain duplicates")
    if len(b_dofs) > 0 and (np.min(b_dofs) < 0 or np.max(b_dofs) >= n_total):
        raise ValueError("boundary_dofs index out of bounds")

    b_set = set(b_dofs.tolist())
    i_dofs = np.array([d for d in range(n_total) if d not in b_set], dtype=int)
    n_b = len(b_dofs)
    n_i = len(i_dofs)
    n_m = max(0, min(int(n_modes), n_i))

    is_sparse = sparse.issparse(K) or sparse.issparse(M)
    K_work = 0.5 * (K + K.T)
    M_work = 0.5 * (M + M.T)
    if is_sparse:
        K_csc = K_work.tocsc() if sparse.issparse(K_work) else csc_matrix(K_work)
        M_csc = M_work.tocsc() if sparse.issparse(M_work) else csc_matrix(M_work)
    else:
        K_dense = np.asarray(K_work, dtype=float)
        M_dense = np.asarray(M_work, dtype=float)

    if n_i == 0:
        T_cb = np.eye(n_total, dtype=float)[:, b_dofs] if n_b > 0 else np.zeros((n_total, 0))
        K_mat = K_csc.toarray() if is_sparse else K_dense
        M_mat = M_csc.toarray() if is_sparse else M_dense
        K_cb = 0.5 * (T_cb.T @ K_mat @ T_cb + (T_cb.T @ K_mat @ T_cb).T)
        M_cb = 0.5 * (T_cb.T @ M_mat @ T_cb + (T_cb.T @ M_mat @ T_cb).T)
        m_tot = float(mass_kg) if mass_kg is not None else _estimate_mass_from_matrix(M_work)
        return CraigBamptonSuperelement(
            name=name,
            n_boundary_dofs=n_b,
            n_internal_modes=0,
            K_cb=K_cb,
            M_cb=M_cb,
            T_cb=T_cb,
            fixed_interface_freqs_hz=[],
            port_stiffness_diag=_extract_port_stiffness_diag(K_cb, n_b),
            mass_kg=m_tot,
            source_hash=source_hash,
        )

    # Extract interior and coupling blocks
    if is_sparse:
        K_ii_sp = K_csc[i_dofs, :][:, i_dofs].tocsc()
        M_ii_sp = M_csc[i_dofs, :][:, i_dofs].tocsc()
        K_ib_dense = K_csc[i_dofs, :][:, b_dofs].toarray() if n_b > 0 else np.zeros((n_i, 0))
    else:
        K_ii_den = 0.5 * (K_dense[np.ix_(i_dofs, i_dofs)] + K_dense[np.ix_(i_dofs, i_dofs)].T)
        M_ii_den = 0.5 * (M_dense[np.ix_(i_dofs, i_dofs)] + M_dense[np.ix_(i_dofs, i_dofs)].T)
        K_ib_dense = K_dense[np.ix_(i_dofs, b_dofs)] if n_b > 0 else np.zeros((n_i, 0))

    # 1. Static constraint modes: Psi_cb = -K_ii^{-1} K_ib
    lu_kii = None
    K_ii_solve = None
    if is_sparse or n_i > 350:
        K_ii_csc = K_ii_sp if is_sparse else csc_matrix(K_ii_den)
        try:
            lu_kii = splu(K_ii_csc)
            K_ii_solve = K_ii_csc
            Psi_cb = lu_kii.solve(-K_ib_dense) if n_b > 0 else np.zeros((n_i, 0), dtype=float)
        except RuntimeError:
            k_diag = np.asarray(K_ii_csc.diagonal(), dtype=float)
            reg = 1e-9 * max(float(np.max(k_diag)), 1.0)
            K_ii_solve = (K_ii_csc + sparse.diags(np.full(n_i, reg), format="csc")).tocsc()
            lu_kii = splu(K_ii_solve)
            Psi_cb = lu_kii.solve(-K_ib_dense) if n_b > 0 else np.zeros((n_i, 0), dtype=float)
    else:
        try:
            c_fac = cho_factor(K_ii_den, check_finite=False)
            Psi_cb = cho_solve(c_fac, -K_ib_dense, check_finite=False) if n_b > 0 else np.zeros((n_i, 0))
        except np.linalg.LinAlgError:
            reg = 1e-9 * max(float(np.max(np.diag(K_ii_den))), 1.0)
            Psi_cb = (
                solve(K_ii_den + reg * np.eye(n_i), -K_ib_dense)
                if n_b > 0
                else np.zeros((n_i, 0), dtype=float)
            )

    # 2. Fixed-interface normal modes: (K_ii - omega^2 M_ii) phi = 0
    Phi_ii = np.zeros((n_i, n_m), dtype=float)
    fixed_freqs_hz: list[float] = []
    if n_m > 0:
        if is_sparse:
            m_diag = np.asarray(M_ii_sp.diagonal(), dtype=float)
            m_floor = max(float(np.max(m_diag)) * 1e-13, 1e-16)
            add_diag = np.where(m_diag < m_floor, m_floor, 0.0)
            M_ii_eig_sp = (M_ii_sp + sparse.diags(add_diag, format="csc")).tocsc()
        else:
            m_diag = np.diag(M_ii_den)
            m_floor = max(float(np.max(m_diag)) * 1e-13, 1e-16)
            M_ii_eig_den = M_ii_den + np.diag(np.where(m_diag < m_floor, m_floor, 0.0))

        solved_sparse_eig = False
        if (is_sparse or n_i > 350) and 1 <= n_m < n_i - 1 and lu_kii is not None and K_ii_solve is not None:
            try:
                op = LinearOperator(K_ii_solve.shape, matvec=lu_kii.solve, dtype=float)
                eigvals, eigvecs = eigsh(
                    K_ii_solve,
                    k=n_m,
                    M=M_ii_eig_sp if is_sparse else csc_matrix(M_ii_eig_den),
                    sigma=0.0,
                    which="LM",
                    OPinv=op,
                )
                order = np.argsort(eigvals)
                eigvals = np.maximum(eigvals[order], 0.0)
                eigvecs = eigvecs[:, order]
                solved_sparse_eig = True
            except RuntimeError:
                solved_sparse_eig = False

        if not solved_sparse_eig:
            K_ii_mat = K_ii_sp.toarray() if is_sparse else K_ii_den
            M_ii_mat = M_ii_eig_sp.toarray() if is_sparse else M_ii_eig_den
            eigvals, eigvecs = eigh(K_ii_mat, M_ii_mat, subset_by_index=[0, n_m - 1])
            eigvals = np.maximum(eigvals, 0.0)

        # Strict M_ii mass-normalization: Phi_ii^T M_ii Phi_ii = I
        M_norm_op = M_ii_eig_sp if is_sparse else M_ii_eig_den
        for j in range(n_m):
            v = eigvecs[:, j]
            m_gen = float(v @ (M_norm_op @ v))
            if m_gen > 1e-24:
                v = v / math.sqrt(m_gen)
            Phi_ii[:, j] = v
        fixed_freqs_hz = [math.sqrt(float(lam)) / (2.0 * math.pi) for lam in eigvals]

    # 3. Assemble global reduction matrix T_cb and exact symmetric K_cb, M_cb
    T_cb = np.zeros((n_total, n_b + n_m), dtype=float)
    if n_b > 0:
        T_cb[b_dofs, :n_b] = np.eye(n_b, dtype=float)
        T_cb[i_dofs, :n_b] = Psi_cb
    if n_m > 0:
        T_cb[i_dofs, n_b:] = Phi_ii

    if is_sparse:
        K_cb = T_cb.T @ (K_csc @ T_cb)
        M_cb = T_cb.T @ (M_csc @ T_cb)
    else:
        K_cb = T_cb.T @ K_dense @ T_cb
        M_cb = T_cb.T @ M_dense @ T_cb

    K_cb = 0.5 * (K_cb + K_cb.T)
    M_cb = 0.5 * (M_cb + M_cb.T)
    m_tot = float(mass_kg) if mass_kg is not None else _estimate_mass_from_matrix(M_work)

    return CraigBamptonSuperelement(
        name=name,
        n_boundary_dofs=n_b,
        n_internal_modes=n_m,
        K_cb=K_cb,
        M_cb=M_cb,
        T_cb=T_cb,
        fixed_interface_freqs_hz=fixed_freqs_hz,
        port_stiffness_diag=_extract_port_stiffness_diag(K_cb, n_b),
        mass_kg=m_tot,
        source_hash=source_hash,
    )


def condense_solid_patch_to_vp(
    K: np.ndarray | sparse.spmatrix,
    M: np.ndarray | sparse.spmatrix,
    port_node_sets: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_modes: int = 6,
    name: str = "solid_vp_part",
    source_hash: str = "",
    interface_mode: Literal["rbe3", "rbe2"] = "rbe3",
    mass_kg: float | None = None,
) -> CraigBamptonSuperelement:
    """Condense a multi-port 3D solid FEA model to 6-DOF virtual points + internal modes.

    Parameters
    ----------
    K, M:
        Global 3-DOF-per-node solid stiffness and mass matrices ``(N, N)``.
    port_node_sets:
        List of ``(patch_indices, patch_coords_m, vp_coord_m)`` per mounting port.
        ``patch_indices`` may be 1D node indices of length ``n_p`` or 1D DOF indices
        of length ``3 * n_p``; ``patch_coords_m`` has shape ``(n_p, 3)``.
    n_modes:
        Number of fixed-interface normal modes to retain.
    interface_mode:
        ``"rbe3"`` (default) relaxes patch warp/Poisson DOFs inside the internal CB
        partition via the nullspace of ``virtual_point_filter_rbe3`` (no artificial
        boss locking); ``"rbe2"`` constrains patch nodes rigidly to ``R_u``.
    """
    n_total = K.shape[0]
    if not port_node_sets:
        raise ValueError("port_node_sets must contain at least one mounting port")

    used_dofs: set[int] = set()
    port_dofs_list: list[np.ndarray] = []
    R_u_list: list[np.ndarray] = []
    T_u_list: list[np.ndarray] = []

    for port_idx, (idx_raw, coords_raw, vp_raw) in enumerate(port_node_sets):
        coords = np.asarray(coords_raw, dtype=float).reshape(-1, 3)
        n_p = coords.shape[0]
        idx_arr = np.asarray(idx_raw, dtype=int).ravel()

        if len(idx_arr) == n_p:
            dofs = np.empty(3 * n_p, dtype=int)
            dofs[0::3] = 3 * idx_arr
            dofs[1::3] = 3 * idx_arr + 1
            dofs[2::3] = 3 * idx_arr + 2
            node_mask = np.array(
                [
                    (3 * int(nd) not in used_dofs)
                    and (3 * int(nd) + 1 not in used_dofs)
                    and (3 * int(nd) + 2 not in used_dofs)
                    for nd in idx_arr
                ],
                dtype=bool,
            )
            if not np.all(node_mask):
                idx_arr = idx_arr[node_mask]
                coords = coords[node_mask]
                n_p = coords.shape[0]
                dofs = np.empty(3 * n_p, dtype=int)
                dofs[0::3] = 3 * idx_arr
                dofs[1::3] = 3 * idx_arr + 1
                dofs[2::3] = 3 * idx_arr + 2
        elif len(idx_arr) == 3 * n_p:
            dofs = idx_arr
            keep = np.array([int(d) not in used_dofs for d in dofs], dtype=bool)
            if not np.all(keep):
                # Drop whole nodes if any DOF already used
                node_keep = keep.reshape(n_p, 3).all(axis=1)
                coords = coords[node_keep]
                dofs = dofs.reshape(n_p, 3)[node_keep].ravel()
                n_p = coords.shape[0]
        else:
            raise ValueError(
                f"Port {port_idx}: index length {len(idx_arr)} does not match n_p={n_p} or 3*n_p={3*n_p}"
            )

        if n_p == 0:
            raise ValueError(f"Port {port_idx} has zero remaining non-overlapping patch nodes")

        used_dofs.update(dofs.tolist())
        R_u = virtual_point_transform(coords, vp_raw)
        T_u = virtual_point_filter_rbe3(R_u)
        port_dofs_list.append(dofs)
        R_u_list.append(R_u)
        T_u_list.append(T_u)

    n_ports = len(port_dofs_list)
    n_vp_total = 6 * n_ports
    m_tot = float(mass_kg) if mass_kg is not None else _estimate_mass_from_matrix(M)

    # Check if every port has full rank 6 and sufficient nodes for RBE3 nullspace relaxation
    can_rbe3 = interface_mode == "rbe3" and all(
        dofs.shape[0] > 6 and np.linalg.matrix_rank(R_u, tol=1e-9) == 6
        for dofs, R_u in zip(port_dofs_list, R_u_list)
    )

    if can_rbe3:
        # Build sparse square basis change B: [q_vp (6*P); w_warp; u_int] -> physical u
        rows_list: list[np.ndarray] = []
        cols_list: list[np.ndarray] = []
        vals_list: list[np.ndarray] = []

        vp_col_offset = 0
        warp_col_offset = n_vp_total
        for dofs, R_u, T_u in zip(port_dofs_list, R_u_list, T_u_list):
            # q_vp block
            rr, cc = np.meshgrid(dofs, np.arange(vp_col_offset, vp_col_offset + 6), indexing="ij")
            rows_list.append(rr.ravel())
            cols_list.append(cc.ravel())
            vals_list.append(R_u.ravel())
            vp_col_offset += 6

            # Nullspace warp block V_p: T_u @ V_p = 0
            V_p = null_space(T_u)
            n_w = V_p.shape[1]
            if n_w > 0:
                rr_w, cc_w = np.meshgrid(
                    dofs, np.arange(warp_col_offset, warp_col_offset + n_w), indexing="ij"
                )
                rows_list.append(rr_w.ravel())
                cols_list.append(cc_w.ravel())
                vals_list.append(V_p.ravel())
                warp_col_offset += n_w

        interior_dofs = np.array([d for d in range(n_total) if d not in used_dofs], dtype=int)
        n_int = len(interior_dofs)
        if n_int > 0:
            rows_list.append(interior_dofs)
            cols_list.append(np.arange(warp_col_offset, warp_col_offset + n_int, dtype=int))
            vals_list.append(np.ones(n_int, dtype=float))

        B_sp = sparse.coo_matrix(
            (np.concatenate(vals_list), (np.concatenate(rows_list), np.concatenate(cols_list))),
            shape=(n_total, n_total),
        ).tocsc()

        K_sp = K if sparse.issparse(K) else csc_matrix(K)
        M_sp = M if sparse.issparse(M) else csc_matrix(M)
        K_split = B_sp.T @ K_sp @ B_sp
        M_split = B_sp.T @ M_sp @ B_sp

        se_split = condense_craig_bampton(
            K_split,
            M_split,
            boundary_dofs=np.arange(n_vp_total, dtype=int),
            n_modes=n_modes,
            name=name,
            source_hash=source_hash,
            mass_kg=m_tot,
        )
        T_vp = B_sp @ se_split.T_cb
        return CraigBamptonSuperelement(
            name=name,
            n_boundary_dofs=n_vp_total,
            n_internal_modes=se_split.n_internal_modes,
            K_cb=se_split.K_cb,
            M_cb=se_split.M_cb,
            T_cb=T_vp,
            fixed_interface_freqs_hz=se_split.fixed_interface_freqs_hz,
            port_stiffness_diag=_extract_port_stiffness_diag(se_split.K_cb, n_vp_total),
            mass_kg=m_tot,
            source_hash=source_hash,
        )

    # RBE2 rigid-patch reduction fallback / explicit mode
    all_b_dofs = np.concatenate(port_dofs_list)
    se_raw = condense_craig_bampton(
        K,
        M,
        boundary_dofs=all_b_dofs,
        n_modes=n_modes,
        name=name,
        source_hash=source_hash,
        mass_kg=m_tot,
    )
    R_all = block_diag(*R_u_list)
    n_m = se_raw.n_internal_modes
    R_aug = block_diag(R_all, np.eye(n_m, dtype=float)) if n_m > 0 else R_all
    K_vp = 0.5 * (R_aug.T @ se_raw.K_cb @ R_aug + (R_aug.T @ se_raw.K_cb @ R_aug).T)
    M_vp = 0.5 * (R_aug.T @ se_raw.M_cb @ R_aug + (R_aug.T @ se_raw.M_cb @ R_aug).T)
    T_vp = se_raw.T_cb @ R_aug
    return CraigBamptonSuperelement(
        name=name,
        n_boundary_dofs=n_vp_total,
        n_internal_modes=n_m,
        K_cb=K_vp,
        M_cb=M_vp,
        T_cb=T_vp,
        fixed_interface_freqs_hz=se_raw.fixed_interface_freqs_hz,
        port_stiffness_diag=_extract_port_stiffness_diag(K_vp, n_vp_total),
        mass_kg=m_tot,
        source_hash=source_hash,
    )


def assemble_superelement_into_global(
    K_global: np.ndarray | sparse.spmatrix,
    M_global: np.ndarray | sparse.spmatrix,
    se: CraigBamptonSuperelement,
    boundary_global_dofs: Sequence[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Augment ``(K_global, M_global)`` with superelement internal modes and couple boundary DOFs."""
    b_glob = np.asarray(boundary_global_dofs, dtype=int).ravel()
    if len(b_glob) != se.n_boundary_dofs:
        raise ValueError(
            f"Expected {se.n_boundary_dofs} boundary_global_dofs for '{se.name}', got {len(b_glob)}"
        )

    K_g = K_global.toarray() if sparse.issparse(K_global) else np.asarray(K_global, dtype=float)
    M_g = M_global.toarray() if sparse.issparse(M_global) else np.asarray(M_global, dtype=float)
    n_g = K_g.shape[0]
    n_m = se.n_internal_modes
    n_new = n_g + n_m

    K_aug = np.zeros((n_new, n_new), dtype=float)
    M_aug = np.zeros((n_new, n_new), dtype=float)
    K_aug[:n_g, :n_g] = K_g
    M_aug[:n_g, :n_g] = M_g

    map_dofs = np.concatenate([b_glob, np.arange(n_g, n_new, dtype=int)])
    K_aug[np.ix_(map_dofs, map_dofs)] += se.K_cb
    M_aug[np.ix_(map_dofs, map_dofs)] += se.M_cb

    K_aug = 0.5 * (K_aug + K_aug.T)
    M_aug = 0.5 * (M_aug + M_aug.T)
    return K_aug, M_aug


def _build_coarse_bracket_rectilinear_grid(
    bracket_type: str,
    dims: dict[str, float],
    target_elements: int = 380,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Generate a feature-aligned Hex8 occupancy grid and per-axis brick sizes (dx, dy, dz) in mm."""
    Lx, Ly, Lz = dims["Lx"], dims["Ly"], dims["Lz"]
    min_dim = min(Lx, Ly, Lz)
    h_vol = ((0.35 * Lx * Ly * Lz) / max(float(target_elements), 64.0)) ** (1.0 / 3.0)
    h_mm = float(np.clip(h_vol, max(0.5, min_dim / 40.0), max(3.0, min_dim / 2.5)))

    dx, dy, dz, _ = _bracket_axis_spacings_mm(
        bracket_type, dims, h_mm, min_wall_cells=1
    )

    nx, ny, nz = len(dx), len(dy), len(dz)
    h_eq = float(np.mean(dx) * np.mean(dy) * np.mean(dz)) ** (1.0 / 3.0)
    xc = np.cumsum(dx) - 0.5 * dx
    yc = np.cumsum(dy) - 0.5 * dy
    zc = np.cumsum(dz) - 0.5 * dz
    X, Y, Z = np.meshgrid(xc, yc, zc, indexing="ij")
    grid = np.zeros((nx, ny, nz), dtype=bool)

    if bracket_type in ("corner_gusset", "xy_joint_l_bracket"):
        w = dims["w"]
        grid |= Y <= w
        grid |= X <= w
        r_eff = dims["r_eff"]
        if r_eff > 0.0:
            p0 = dims["p0"]
            px_end = dims["px_end"]
            py_end = dims["py_end"]
            tri_x = max(px_end - p0, 1e-6)
            tri_y = max(py_end - p0, 1e-6)
            in_tri = (X >= p0) & (Y >= p0) & (((X - p0) / tri_x + (Y - p0) / tri_y) <= 1.0)
            if bracket_type == "corner_gusset":
                grid |= in_tri & (np.abs(Z - 0.5 * Lz) <= 0.5 * r_eff)
            else:
                grid |= in_tri & (Z <= r_eff)

        d_eff = dims["d_eff"]
        if d_eff > 0.5 and h_eq < 0.7 * d_eff:
            rad = 0.5 * d_eff
            z_mid = 0.5 * Lz
            if bracket_type == "corner_gusset":
                grid &= ~((Y <= w) & (((X - dims["hole_x"]) ** 2 + (Z - z_mid) ** 2) <= rad**2))
                grid &= ~((X <= w) & (((Y - dims["hole_y"]) ** 2 + (Z - z_mid) ** 2) <= rad**2))
            else:
                for bx in (dims["hx1"], dims["hx2"]):
                    grid &= ~((Y <= w) & (((X - bx) ** 2 + (Z - z_mid) ** 2) <= rad**2))
                grid &= ~((X <= w) & (((Y - dims["hy"]) ** 2 + (Z - z_mid) ** 2) <= rad**2))

    else:
        plate_t = dims["plate_t"]
        flange_t = dims["flange_t"]
        rib_t = dims["rib_t"]
        grid |= np.abs(Z - 0.5 * Lz) <= 0.5 * plate_t
        grid |= Y <= flange_t
        if rib_t > 0.0:
            y0 = 0.5 * flange_t
            half_span_z = 0.5 * Lz - (0.5 * Lz - 0.5 * plate_t) * np.clip(
                (Y - y0) / max(Ly - y0, 1e-6), 0.0, 1.0
            )
            in_gusset = (Y >= y0) & (np.abs(Z - 0.5 * Lz) <= half_span_z)
            grid |= ((X <= rib_t) | (X >= (Lx - rib_t))) & in_gusset
        cx, cy = dims["cx"], dims["cy"]
        grid &= ~(((X - cx) ** 2 + (Y - cy) ** 2) <= dims["pilot_r"] ** 2)

    return grid, (dx, dy, dz)


def _build_coarse_bracket_grid(
    bracket_type: str,
    dims: dict[str, float],
    target_elements: int = 380,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Generate a coarse Hex8 occupancy grid and mean brick sizes (hx, hy, hz) in mm."""
    grid, (dx, dy, dz) = _build_coarse_bracket_rectilinear_grid(
        bracket_type, dims, target_elements=target_elements
    )
    return grid, (float(np.mean(dx)), float(np.mean(dy)), float(np.mean(dz)))


def _load_cached_superelement(cache_dir: Path, source_hash: str) -> CraigBamptonSuperelement | None:
    npz_path = cache_dir / f"{source_hash}.npz"
    meta_path = cache_dir / f"{source_hash}.json"
    if not (npz_path.exists() and meta_path.exists()):
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        with np.load(npz_path) as data:
            return CraigBamptonSuperelement(
                name=str(meta["name"]),
                n_boundary_dofs=int(meta["n_boundary_dofs"]),
                n_internal_modes=int(meta["n_internal_modes"]),
                K_cb=np.array(data["K_cb"], dtype=float),
                M_cb=np.array(data["M_cb"], dtype=float),
                T_cb=np.array(data["T_cb"], dtype=float),
                fixed_interface_freqs_hz=[float(v) for v in meta["fixed_interface_freqs_hz"]],
                port_stiffness_diag={k: float(v) for k, v in meta["port_stiffness_diag"].items()},
                mass_kg=float(meta["mass_kg"]),
                source_hash=str(meta["source_hash"]),
            )
    except Exception:
        return None


def _save_cached_superelement(cache_dir: Path, se: CraigBamptonSuperelement) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path = cache_dir / f"{se.source_hash}.npz"
    meta_path = cache_dir / f"{se.source_hash}.json"
    np.savez(npz_path, K_cb=se.K_cb, M_cb=se.M_cb, T_cb=se.T_cb)
    meta = {
        "name": se.name,
        "n_boundary_dofs": se.n_boundary_dofs,
        "n_internal_modes": se.n_internal_modes,
        "fixed_interface_freqs_hz": se.fixed_interface_freqs_hz,
        "port_stiffness_diag": se.port_stiffness_diag,
        "mass_kg": se.mass_kg,
        "source_hash": se.source_hash,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def condense_parametric_bracket(
    bracket_params: dict[str, Any],
    material_id: str,
    cache_dir: Path | None = None,
    fdm_anisotropic: bool | None = None,
) -> CraigBamptonSuperelement:
    """Build a coarse Hex8 mesh of a parametric bracket, apply VPT on 2 ports, and condense via CB CMS.

    Reuses ``constitutive_matrix_3d``, ``hex8_element_stiffness``, and ``hex8_element_mass``
    from ``solid_fem.py`` with exact rectangular brick geometry ``(hx, hy, hz)``. Caches
    results by SHA256 hash in memory and optionally on disk in ``cache_dir``.
    """
    bracket_type = str(bracket_params.get("bracket_type", "corner_gusset"))
    arm_length_mm = float(bracket_params.get("arm_length_mm", 40.0))
    height_mm = float(bracket_params.get("height_mm", 20.0))
    wall_thickness_mm = float(bracket_params.get("wall_thickness_mm", 6.0))
    rib_thickness_mm = float(bracket_params.get("rib_thickness_mm", 5.0))
    bolt_hole_mm = float(bracket_params.get("bolt_hole_mm", 5.2))
    fillet_mm = float(bracket_params.get("fillet_mm", 2.0))
    n_modes = int(bracket_params.get("n_modes", 6))
    target_elements = int(bracket_params.get("target_elements", 380))
    mat = get_material(material_id)
    raw_fdm = fdm_anisotropic if fdm_anisotropic is not None else bracket_params.get("fdm_anisotropic")
    is_fdm = bool(raw_fdm) if raw_fdm is not None else bool(mat.is_printed_polymer)
    interface_mode: Literal["rbe3", "rbe2"] = bracket_params.get("interface_mode", "rbe3")
    fixed_face: FaceName = bracket_params.get("fixed_face", "min_y")
    load_face: FaceName = bracket_params.get("load_face", "max_y")

    canonical_payload = {
        "schema_v": 3,
        "bracket_type": bracket_type,
        "arm_length_mm": arm_length_mm,
        "height_mm": height_mm,
        "wall_thickness_mm": wall_thickness_mm,
        "rib_thickness_mm": rib_thickness_mm,
        "bolt_hole_mm": bolt_hole_mm,
        "fillet_mm": fillet_mm,
        "n_modes": n_modes,
        "target_elements": target_elements,
        "fdm_anisotropic": is_fdm,
        "interface_mode": interface_mode,
        "fixed_face": fixed_face,
        "load_face": load_face,
        "material_id": material_id,
    }
    source_hash = hashlib.sha256(
        json.dumps(canonical_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()

    if cache_dir is not None:
        disk_hit = _load_cached_superelement(Path(cache_dir), source_hash)
        if disk_hit is not None:
            _BRACKET_CACHE[source_hash] = disk_hit
            return disk_hit
    if source_hash in _BRACKET_CACHE:
        cached = _BRACKET_CACHE[source_hash]
        if cache_dir is not None:
            _save_cached_superelement(Path(cache_dir), cached)
        return cached

    dims = _bracket_layout_dims(
        bracket_type=bracket_type,  # type: ignore[arg-type]
        arm_length_mm=arm_length_mm,
        height_mm=height_mm,
        wall_thickness_mm=wall_thickness_mm,
        rib_thickness_mm=rib_thickness_mm,
        bolt_hole_mm=bolt_hole_mm,
        fillet_mm=fillet_mm,
    )
    grid, (dx_mm, dy_mm, dz_mm) = _build_coarse_bracket_rectilinear_grid(
        bracket_type, dims, target_elements=target_elements
    )

    occ = np.argwhere(grid)
    n_elem = len(occ)
    if n_elem == 0:
        raise ValueError("Coarse bracket grid produced 0 occupied Hex8 elements")
    _, ny, nz = grid.shape

    offsets = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=int,
    )
    elem_corners = occ[:, None, :] + offsets[None, :, :]
    flat_keys = (
        elem_corners[:, :, 0] * ((ny + 1) * (nz + 1))
        + elem_corners[:, :, 1] * (nz + 1)
        + elem_corners[:, :, 2]
    )
    uniq_keys, inverse = np.unique(flat_keys, return_inverse=True)
    elem_nodes = inverse.reshape(n_elem, 8)
    n_nodes = len(uniq_keys)
    n_dof = 3 * n_nodes

    node_ix = uniq_keys // ((ny + 1) * (nz + 1))
    rem = uniq_keys % ((ny + 1) * (nz + 1))
    node_iy = rem // (nz + 1)
    node_iz = rem % (nz + 1)
    node_grid_coords = np.stack([node_ix, node_iy, node_iz], axis=1)

    x_edges_m = np.concatenate([[0.0], np.cumsum(dx_mm)]) * 1e-3
    y_edges_m = np.concatenate([[0.0], np.cumsum(dy_mm)]) * 1e-3
    z_edges_m = np.concatenate([[0.0], np.cumsum(dz_mm)]) * 1e-3
    node_coords_m = np.stack(
        [x_edges_m[node_ix], y_edges_m[node_iy], z_edges_m[node_iz]], axis=1
    )

    elem_sizes_m = np.stack(
        [dx_mm[occ[:, 0]], dy_mm[occ[:, 1]], dz_mm[occ[:, 2]]], axis=1
    ) * 1e-3
    elem_vols_m3 = np.prod(elem_sizes_m, axis=1)
    uniq_sizes_m, cls_idx = np.unique(np.round(elem_sizes_m, 9), axis=0, return_inverse=True)

    D = constitutive_matrix_3d(mat, fdm_anisotropic=is_fdm)
    ke_classes = np.empty((len(uniq_sizes_m), 24, 24), dtype=float)
    me_classes = np.empty((len(uniq_sizes_m), 24, 24), dtype=float)
    for c_i, sz_m in enumerate(uniq_sizes_m):
        brick_m = (float(sz_m[0]), float(sz_m[1]), float(sz_m[2]))
        ke_c, _ = hex8_element_stiffness(D, brick_m)
        me_c = hex8_element_mass(mat.density_kg_m3, brick_m)
        ke_classes[c_i] = ke_c
        me_classes[c_i] = me_c

    elem_dofs = np.empty((n_elem, 24), dtype=int)
    elem_dofs[:, 0::3] = 3 * elem_nodes
    elem_dofs[:, 1::3] = 3 * elem_nodes + 1
    elem_dofs[:, 2::3] = 3 * elem_nodes + 2

    rows = np.repeat(elem_dofs, 24, axis=1).ravel()
    cols = np.tile(elem_dofs, (1, 24)).ravel()
    K = sparse.coo_matrix(
        (ke_classes[cls_idx].ravel(), (rows, cols)), shape=(n_dof, n_dof)
    ).tocsc()
    M = sparse.coo_matrix(
        (me_classes[cls_idx].ravel(), (rows, cols)), shape=(n_dof, n_dof)
    ).tocsc()

    min_c = node_grid_coords.min(axis=0)
    max_c = node_grid_coords.max(axis=0)

    def _face_nodes(face: FaceName) -> np.ndarray:
        axis = {"x": 0, "y": 1, "z": 2}[face[-1]]
        target = min_c[axis] if face.startswith("min") else max_c[axis]
        return np.where(node_grid_coords[:, axis] == target)[0]

    p0_nodes = _face_nodes(fixed_face)
    p1_nodes = np.setdiff1d(_face_nodes(load_face), p0_nodes)
    if len(p0_nodes) < 3 or len(p1_nodes) < 3:
        raise ValueError("Selected mounting ports must contain at least 3 nodes each")

    p0_coords = node_coords_m[p0_nodes]
    p1_coords = node_coords_m[p1_nodes]
    p0_vp = np.mean(p0_coords, axis=0)
    p1_vp = np.mean(p1_coords, axis=0)

    mass_kg = float(np.sum(elem_vols_m3) * mat.density_kg_m3)
    se = condense_solid_patch_to_vp(
        K,
        M,
        port_node_sets=[
            (p0_nodes, p0_coords, p0_vp),
            (p1_nodes, p1_coords, p1_vp),
        ],
        n_modes=n_modes,
        name=f"{bracket_type}_{material_id}",
        source_hash=source_hash,
        interface_mode=interface_mode,
        mass_kg=mass_kg,
    )
    _BRACKET_CACHE[source_hash] = se
    if cache_dir is not None:
        _save_cached_superelement(Path(cache_dir), se)
    return se
