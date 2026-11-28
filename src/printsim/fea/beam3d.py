"""Exact 12-DOF 3D Euler-Bernoulli / Timoshenko frame finite element matrices."""

from __future__ import annotations

import math

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

from printsim.fea.shell_mitc4 import cross3, fastener_spring_matrices, mitc4_shell_matrices
from printsim.geometry.builder import (
    AssemblyModel,
    AttachedSuperelement,
    StructuralMember,
    StructuralNode,
)


def node_rotational_inertia_xyz(
    node: StructuralNode,
    assembly: AssemblyModel,
) -> tuple[float, float, float]:
    """Return diagonal rotational inertia ``(Ixx, Iyy, Izz)`` [kg*m^2] for a lumped node (B-11).

    - Bed surface: thin rectangular plate ``m*by^2/12, m*bx^2/12, m*(bx^2+by^2)/12``
      (or circular disk ``m*R^2/4, m*R^2/4, m*R^2/2`` for Delta printers).
    - Nozzle tip: compact ~7 mm gyration radius (heater block).
    - Other lumped masses (toolhead CoG, motors, carriage hubs): 25 mm box estimate.
    """
    m = float(node.lumped_mass_kg)
    if m <= 0.0:
        return (0.0, 0.0, 0.0)

    bed_id = getattr(assembly, "bed_node_id", "") or "bed_surface"
    if node.id == bed_id:
        spec = getattr(assembly, "spec", None)
        if spec is not None and hasattr(spec, "bed"):
            bx = max(float(spec.bed.size_mm[0]) * 1e-3, 0.05)
            by = max(float(spec.bed.size_mm[1]) * 1e-3, 0.05)
            kin_type = getattr(spec, "architecture", None)
            if kin_type == "delta":
                r_disk = 0.25 * (bx + by)
                ixx = m * (r_disk**2) / 4.0
                return (ixx, ixx, m * (r_disk**2) / 2.0)
            return (
                m * (by**2) / 12.0,
                m * (bx**2) / 12.0,
                m * (bx**2 + by**2) / 12.0,
            )

    r_gyr = 0.007 if node.role == "nozzle" else 0.025
    rot_i = m * (r_gyr**2)
    return (rot_i, rot_i, rot_i)


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

    ey = cross3(ref, ex)
    ey /= np.linalg.norm(ey)
    ez = cross3(ex, ey)
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


def beam_shear_coefficients(member: StructuralMember) -> tuple[float, float]:
    """Return Timoshenko shear correction factors ``(kappa_y, kappa_z)`` for anisotropic tubes or milled rails.

    Honors explicit Tet10-calibrated ``shear_kappa_y`` / ``shear_kappa_z`` when present on a
    profile (e.g., hollow/pocketed linear rails on unsupported X-gantries), otherwise returns
    ``(0.0, 0.0)`` (pure Euler-Bernoulli) unless ``member.profile.shape == "tube"`` and
    ``E / G > 4.5``, preserving exact isotropic Euler-Bernoulli benchmarks while capturing
    transverse shear deformation in high-E/G composite tubes.
    """
    if member.profile.shear_kappa_y is not None or member.profile.shear_kappa_z is not None:
        return (
            float(member.profile.shear_kappa_y or 0.0),
            float(member.profile.shear_kappa_z or 0.0),
        )
    if member.profile.shape != "tube":
        return (0.0, 0.0)
    e_pa = float(member.material.youngs_modulus_pa)
    g_pa = max(float(member.material.effective_shear_modulus_pa), 1.0)
    if e_pa / g_pa <= 4.5:
        return (0.0, 0.0)
    t_wall = member.profile.wall_thickness_m
    if t_wall is None or t_wall <= 0.0:
        return (0.434, 0.434)
    area = max(float(member.profile.area_m2), 1e-12)
    a_sy = 2.0 * float(member.profile.width_m) * float(t_wall)
    a_sz = 2.0 * float(member.profile.height_m) * float(t_wall)
    kappa_y = min(0.85, 0.87 * a_sy / area)
    kappa_z = min(0.85, 0.87 * a_sz / area)
    return (kappa_y, kappa_z)


def local_beam_stiffness(
    e_pa: float,
    g_pa: float,
    area: float,
    iy: float,
    iz: float,
    j_torsion: float,
    length: float,
    bending_scale: float = 1.0,
    kappa_shear_y: float = 0.0,
    kappa_shear_z: float = 0.0,
) -> np.ndarray:
    """12x12 local stiffness matrix for a 3D Euler-Bernoulli / Przemieniecki Timoshenko beam element.

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
    phi_y = (
        (12.0 * e_pa * iz * bending_scale) / (kappa_shear_y * max(g_pa, 1.0) * area * L2)
        if kappa_shear_y > 0.0
        else 0.0
    )
    c_z = e_pa * iz * bending_scale / ((1.0 + phi_y) * L3)
    idx_xy = [1, 5, 7, 11]
    k_xy = c_z * np.array(
        [
            [12.0, 6.0 * L, -12.0, 6.0 * L],
            [6.0 * L, (4.0 + phi_y) * L2, -6.0 * L, (2.0 - phi_y) * L2],
            [-12.0, -6.0 * L, 12.0, -6.0 * L],
            [6.0 * L, (2.0 - phi_y) * L2, -6.0 * L, (4.0 + phi_y) * L2],
        ]
    )
    k[np.ix_(idx_xy, idx_xy)] += k_xy

    # Bending in local x-z plane: (u_z1, th_y1, u_z2, th_y2), w'(x) = -th_y
    phi_z = (
        (12.0 * e_pa * iy * bending_scale) / (kappa_shear_z * max(g_pa, 1.0) * area * L2)
        if kappa_shear_z > 0.0
        else 0.0
    )
    c_y = e_pa * iy * bending_scale / ((1.0 + phi_z) * L3)
    idx_xz = [2, 4, 8, 10]
    k_xz = c_y * np.array(
        [
            [12.0, -6.0 * L, -12.0, -6.0 * L],
            [-6.0 * L, (4.0 + phi_z) * L2, 6.0 * L, (2.0 - phi_z) * L2],
            [-12.0, 6.0 * L, 12.0, 6.0 * L],
            [-6.0 * L, (2.0 - phi_z) * L2, 6.0 * L, (4.0 + phi_z) * L2],
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


def _skew_3x3(vec: np.ndarray) -> np.ndarray:
    vx, vy, vz = float(vec[0]), float(vec[1]), float(vec[2])
    return np.array(
        [
            [0.0, -vz, vy],
            [vz, 0.0, -vx],
            [-vy, vx, 0.0],
        ],
        dtype=float,
    )


def superelement_element_matrices(
    se: AttachedSuperelement,
    assembly: AssemblyModel,
    node_idx: dict[str, int],
    internal_dof_start: int,
) -> tuple[list[int], np.ndarray, np.ndarray | None]:
    """Return ``(global_dofs, K_eff, M_eff)`` for a 2-port Craig-Bampton superelement.

    Applies the exact rigid-arm port transform ``u1_cad = u1 + skew(dp_asm - dp_cad) @ th1``
    whenever ``se.port_coords_cad_m`` differs from the assembly port positions, guaranteeing
    exact zero strain energy under arbitrary 6-DOF global rigid-body motion.
    """
    i0 = node_idx[se.port_node_ids[0]]
    i1 = node_idx[se.port_node_ids[1]]
    p0_asm = assembly.nodes[i0].position_m
    p1_asm = assembly.nodes[i1].position_m
    dp_asm = p1_asm - p0_asm

    K_cb = np.asarray(se.K_cb, dtype=float)
    n_tot = int(K_cb.shape[0])
    n_m = max(0, int(getattr(se, "n_internal_modes", max(0, n_tot - 12))))
    if K_cb.shape != (12 + n_m, 12 + n_m):
        raise ValueError(
            f"Superelement '{se.id}' expected K_cb shape {(12 + n_m, 12 + n_m)}, got {K_cb.shape}"
        )

    M_raw = getattr(se, "M_cb", None)
    M_mat = np.asarray(M_raw, dtype=float) if M_raw is not None else None
    if M_mat is not None and M_mat.shape != K_cb.shape:
        raise ValueError(
            f"Superelement '{se.id}' M_cb shape {M_mat.shape} does not match K_cb {K_cb.shape}"
        )

    port_cad = getattr(se, "port_coords_cad_m", None)
    if port_cad is not None:
        p_cad = np.asarray(port_cad, dtype=float)
        dp_cad = p_cad[1] - p_cad[0]
        delta = dp_asm - dp_cad
        if float(np.linalg.norm(delta)) > 1e-12:
            J_b = np.eye(12, dtype=float)
            J_b[6:9, 9:12] = _skew_3x3(delta)
            if n_m > 0:
                J_aug = np.eye(12 + n_m, dtype=float)
                J_aug[:12, :12] = J_b
            else:
                J_aug = J_b
            K_eff = J_aug.T @ K_cb @ J_aug
            M_eff = (J_aug.T @ M_mat @ J_aug) if M_mat is not None else None
        else:
            K_eff = K_cb
            M_eff = M_mat
    else:
        K_eff = K_cb
        M_eff = M_mat

    K_eff = 0.5 * (K_eff + K_eff.T)
    if M_eff is not None:
        M_eff = 0.5 * (M_eff + M_eff.T)

    dofs = (
        list(range(6 * i0, 6 * i0 + 6))
        + list(range(6 * i1, 6 * i1 + 6))
        + list(range(internal_dof_start, internal_dof_start + n_m))
    )
    return dofs, K_eff, M_eff


class GlobalSystem:
    """Global FE system (K, M, gravity load, per-element blocks) assembled once per evaluation.

    ``evaluate_printer`` builds one instance and hands it to the static, modal, and sensitivity
    solvers, which otherwise each re-assemble identical global matrices and re-derive identical
    element blocks. Everything stored here is shared **read-only**; derived quantities
    (symmetrized free-free blocks, sparse LU factorization, eigenpairs) are memoized via
    :meth:`cached` so each is computed at most once per assembly.

    Element block lists are index-aligned with ``assembly.members`` / ``shell_elements`` /
    ``fastener_springs`` / ``superelements``:
      ``member_blocks[i] = (dofs, Ke_global, Me_global, length_m)``
      ``shell_blocks[i] = (dofs, Ke_global, Me_global)``
      ``spring_blocks[i] = (dofs, Ke_global)``
      ``superelement_blocks[i] = (dofs, Ke, Me_or_None)``
    """

    __slots__ = (
        "K",
        "M",
        "F_grav",
        "node_idx",
        "free_dofs",
        "member_blocks",
        "shell_blocks",
        "spring_blocks",
        "superelement_blocks",
        "_cache",
    )

    def __init__(
        self,
        K: np.ndarray,
        M: np.ndarray,
        F_grav: np.ndarray,
        node_idx: dict[str, int],
        free_dofs: list[int],
        member_blocks: list[tuple[list[int], np.ndarray, np.ndarray, float]],
        shell_blocks: list[tuple[list[int], np.ndarray, np.ndarray]],
        spring_blocks: list[tuple[list[int], np.ndarray]],
        superelement_blocks: list[tuple[list[int], np.ndarray, np.ndarray | None]],
    ) -> None:
        self.K = K
        self.M = M
        self.F_grav = F_grav
        self.node_idx = node_idx
        self.free_dofs = free_dofs
        self.member_blocks = member_blocks
        self.shell_blocks = shell_blocks
        self.spring_blocks = spring_blocks
        self.superelement_blocks = superelement_blocks
        self._cache: dict[object, object] = {}

    def as_tuple(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int], list[int]]:
        """Return the legacy ``(K, M, F_grav, node_idx, free_dofs)`` tuple."""
        return self.K, self.M, self.F_grav, self.node_idx, self.free_dofs

    def cached(self, key: object, factory):  # noqa: ANN001, ANN201
        """Return the memoized value for ``key``, computing it with ``factory()`` on first use.

        A raising factory caches nothing, so callers keep their own exception fallbacks.
        """
        try:
            return self._cache[key]
        except KeyError:
            value = factory()
            self._cache[key] = value
            return value

    def peek(self, key: object):  # noqa: ANN201
        """Return the memoized value for ``key`` or ``None`` without computing anything."""
        return self._cache.get(key)

    def store(self, key: object, value: object) -> None:
        self._cache[key] = value

    def reduced_stiffness(self) -> np.ndarray:
        """Symmetrized free-free stiffness ``K_ff`` (read-only)."""

        def _build() -> np.ndarray:
            ix = np.ix_(self.free_dofs, self.free_dofs)
            k_ff = 0.5 * (self.K[ix] + self.K[ix].T)
            k_ff.flags.writeable = False
            return k_ff

        return self.cached("k_ff", _build)

    def reduced_mass(self) -> np.ndarray:
        """Symmetrized free-free mass ``M_ff`` with the positive-definite diagonal floor (read-only)."""

        def _build() -> np.ndarray:
            ix = np.ix_(self.free_dofs, self.free_dofs)
            m_ff = 0.5 * (self.M[ix] + self.M[ix].T)
            # Strictly positive-definite mass diagonal floor (zero-mass rotational DOFs if any)
            m_diag = np.diag(m_ff)
            m_floor = max(float(np.max(m_diag)) * 1e-12, 1e-12)
            m_ff = m_ff + np.diag(np.where(m_diag < m_floor, m_floor, 0.0))
            m_ff.flags.writeable = False
            return m_ff

        return self.cached("m_ff", _build)

    def reduced_stiffness_csc(self) -> csc_matrix:
        """Sparse CSC copy of ``K_ff`` shared by the static, modal, and sensitivity solvers."""
        return self.cached("k_ff_csc", lambda: csc_matrix(self.reduced_stiffness()))

    def reduced_stiffness_lu(self):  # noqa: ANN201
        """SuperLU factorization of ``K_ff``; raises ``RuntimeError`` (uncached) if singular."""
        return self.cached("k_ff_lu", lambda: splu(self.reduced_stiffness_csc()))


def assemble_global_matrices(
    assembly: AssemblyModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int], list[int]]:
    """Assemble global K, M, gravity load vector F_g, node index map, and free DOF list."""
    return assemble_global_system(assembly, freeze=False).as_tuple()


def assemble_global_system(assembly: AssemblyModel, freeze: bool = True) -> GlobalSystem:
    """Assemble the global FE system once, keeping per-element blocks for downstream reuse.

    With ``freeze=True`` (default) every shared array is marked read-only so an accidental
    in-place edit by one solver raises instead of silently corrupting the others.
    """
    n_nodes = len(assembly.nodes)
    n_nodal_dof = 6 * n_nodes
    superelements = getattr(assembly, "superelements", ())
    n_internal_dofs = sum(
        max(0, int(getattr(se, "n_internal_modes", max(0, np.asarray(se.K_cb).shape[0] - 12))))
        for se in superelements
    )
    n_dof = n_nodal_dof + n_internal_dofs
    node_idx = {node.id: i for i, node in enumerate(assembly.nodes)}

    K = np.zeros((n_dof, n_dof), dtype=float)
    M = np.zeros((n_dof, n_dof), dtype=float)
    F_grav = np.zeros(n_dof, dtype=float)
    g_acc = 9.80665
    member_blocks: list[tuple[list[int], np.ndarray, np.ndarray, float]] = []
    shell_blocks: list[tuple[list[int], np.ndarray, np.ndarray]] = []
    spring_blocks: list[tuple[list[int], np.ndarray]] = []
    superelement_blocks: list[tuple[list[int], np.ndarray, np.ndarray | None]] = []

    # Assemble beam elements
    for member in assembly.members:
        i0 = node_idx[member.start_id]
        i1 = node_idx[member.end_id]
        p0 = assembly.nodes[i0].position_m
        p1 = assembly.nodes[i1].position_m
        r_mat, length = beam_rotation_matrix(p0, p1)
        t_mat = transformation_12x12(r_mat)

        bend_scale = 0.08 if member.is_truss else 1.0
        kappa_y, kappa_z = beam_shear_coefficients(member)
        k_loc = local_beam_stiffness(
            e_pa=member.material.youngs_modulus_pa,
            g_pa=member.material.effective_shear_modulus_pa,
            area=member.profile.area_m2,
            iy=member.profile.iy_m4,
            iz=member.profile.iz_m4,
            j_torsion=member.profile.j_m4,
            length=length,
            bending_scale=bend_scale,
            kappa_shear_y=kappa_y,
            kappa_shear_z=kappa_z,
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
        member_blocks.append((dofs, k_glob, m_glob, length))

        # Consistent Euler-Bernoulli nodal gravity load (-Z) including fixed-end moments.
        # Local bending conventions: v'(x) = +th_z => M_z = +/- w_y L^2 / 12;
        # w'(x) = -th_y => M_y = -/+ w_z L^2 / 12.
        elem_weight = member.profile.area_m2 * length * member.material.density_kg_m3 * g_acc
        w_loc = r_mat @ np.array([0.0, 0.0, -elem_weight], dtype=float)
        f_grav_loc = np.array(
            [
                0.5 * w_loc[0],
                0.5 * w_loc[1],
                0.5 * w_loc[2],
                0.0,
                -(w_loc[2] * length) / 12.0,
                +(w_loc[1] * length) / 12.0,
                0.5 * w_loc[0],
                0.5 * w_loc[1],
                0.5 * w_loc[2],
                0.0,
                +(w_loc[2] * length) / 12.0,
                -(w_loc[1] * length) / 12.0,
            ],
            dtype=float,
        )
        F_grav[dofs] += t_mat.T @ f_grav_loc

    # Assemble 24-DOF MITC4 shell panel elements
    for shell in assembly.shell_elements:
        n_indices = [node_idx[nid] for nid in shell.node_ids]
        coords = np.array([assembly.nodes[ni].position_m for ni in n_indices], dtype=float)
        k_sh, m_sh, f_sh = mitc4_shell_matrices(
            coords_3d=coords,
            e_pa=shell.material.youngs_modulus_pa,
            nu=shell.material.poisson_ratio,
            thickness_m=shell.thickness_m,
            areal_mass_kg_m2=shell.areal_mass_kg_m2,
            d_bend_nm=shell.d_bend_nm,
            g_transverse_h_n_m=shell.g_transverse_h_n_m,
            e_mem_h_n_m=shell.e_mem_h_n_m,
        )
        dofs = [6 * ni + d for ni in n_indices for d in range(6)]
        K[np.ix_(dofs, dofs)] += k_sh
        M[np.ix_(dofs, dofs)] += m_sh
        F_grav[dofs] += f_sh
        shell_blocks.append((dofs, k_sh, m_sh))

    # Assemble 12-DOF Huth-Schwarmann CBUSH fastener springs (shell perimeter <-> frame)
    for spring in assembly.fastener_springs:
        i_s = node_idx[spring.shell_node_id]
        i_f = node_idx[spring.frame_node_id]
        k_sp = fastener_spring_matrices(
            k_shear_n_m=spring.k_shear_n_m,
            k_normal_n_m=spring.k_normal_n_m,
            k_rot_nm_rad=spring.k_rot_nm_rad,
            normal_vec=np.asarray(spring.normal_vec, dtype=float),
            offset_vec=np.asarray(spring.offset_m, dtype=float),
            k_diag_6dof=spring.k_diag_6dof,
            offset_vec_frame=(
                None
                if getattr(spring, "offset_frame_m", None) is None
                else np.asarray(spring.offset_frame_m, dtype=float)
            ),
        )
        dofs = list(range(6 * i_s, 6 * i_s + 6)) + list(range(6 * i_f, 6 * i_f + 6))
        K[np.ix_(dofs, dofs)] += k_sp
        spring_blocks.append((dofs, k_sp))

    # Assemble 2-port Craig-Bampton CAD hardware superelements (+ optional internal CB modes)
    internal_dof_cursor = n_nodal_dof
    for se in superelements:
        se_dofs, k_se, m_se = superelement_element_matrices(
            se, assembly, node_idx, internal_dof_cursor
        )
        n_m_se = max(0, int(getattr(se, "n_internal_modes", max(0, k_se.shape[0] - 12))))
        internal_dof_cursor += n_m_se
        K[np.ix_(se_dofs, se_dofs)] += k_se
        superelement_blocks.append((se_dofs, k_se, m_se))
        if m_se is not None:
            M[np.ix_(se_dofs, se_dofs)] += m_se
            a_grav = np.zeros(12 + n_m_se, dtype=float)
            a_grav[2] = -g_acc
            a_grav[8] = -g_acc
            F_grav[se_dofs] += m_se @ a_grav

    # Add lumped nodal masses (motors, toolhead, gantry carriages, bed) + physical rotary inertia
    for i, node in enumerate(assembly.nodes):
        if node.lumped_mass_kg > 0.0:
            base = 6 * i
            M[base, base] += node.lumped_mass_kg
            M[base + 1, base + 1] += node.lumped_mass_kg
            M[base + 2, base + 2] += node.lumped_mass_kg
            ixx, iyy, izz = node_rotational_inertia_xyz(node, assembly)
            M[base + 3, base + 3] += ixx
            M[base + 4, base + 4] += iyy
            M[base + 5, base + 5] += izz
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
    if freeze:
        for arr in (K, M, F_grav):
            arr.flags.writeable = False
        for blocks in (member_blocks, shell_blocks, spring_blocks, superelement_blocks):
            for blk in blocks:
                for item in blk[1:]:
                    if isinstance(item, np.ndarray):
                        item.flags.writeable = False
    return GlobalSystem(
        K,
        M,
        F_grav,
        node_idx,
        free_dofs,
        member_blocks,
        shell_blocks,
        spring_blocks,
        superelement_blocks,
    )


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
    kappa_y, kappa_z = beam_shear_coefficients(member)
    k_loc = local_beam_stiffness(
        e_pa=member.material.youngs_modulus_pa,
        g_pa=member.material.effective_shear_modulus_pa,
        area=member.profile.area_m2,
        iy=member.profile.iy_m4,
        iz=member.profile.iz_m4,
        j_torsion=member.profile.j_m4,
        length=length,
        bending_scale=bend_scale,
        kappa_shear_y=kappa_y,
        kappa_shear_z=kappa_z,
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

    # Shear + torsional shear estimate (Bredt-Batho thin-wall for closed hollow tubes)
    tau_shear = 1.5 * np.hypot(shear_y_n, shear_z_n) / member.profile.area_m2
    prof = member.profile
    if prof.shape == "tube" and prof.wall_thickness_m is not None and prof.wall_thickness_m > 0.0:
        t_w = float(prof.wall_thickness_m)
        r_m = max(float(prof.corner_radius_m) - 0.5 * t_w, 0.0)
        a_m = max(
            (prof.width_m - t_w) * (prof.height_m - t_w) - (4.0 - math.pi) * (r_m**2),
            1e-12,
        )
        tau_torsion = torque_nm / (2.0 * a_m * t_w)
    else:
        r_eff = max(prof.max_c_y, prof.max_c_z)
        tau_torsion = torque_nm * r_eff / prof.j_m4
    tau_total = tau_shear + tau_torsion

    von_mises_pa = float(np.sqrt(sigma_normal**2 + 3.0 * tau_total**2))
    return {
        "axial_force_n": axial_n,
        "max_bending_moment_nm": float(np.hypot(moment_y_nm, moment_z_nm)),
        "torque_nm": torque_nm,
        "von_mises_mpa": von_mises_pa * 1e-6,
        "safety_factor": float(member.material.yield_strength_pa / max(von_mises_pa, 1.0)),
    }
