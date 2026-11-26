"""3D SIMP Topology Optimization and watertight STL mesh export for voxelized solid parts."""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    import trimesh

from printsim.schema import get_material

from .solid_fem import (
    FaceName,
    PartPreset,
    VoxelSize,
    _normalize_voxel_size,
    _scalar_voxel_size_mm,
    constitutive_matrix_3d,
    generate_parametric_part_grid,
    hex8_element_stiffness,
    solve_solid_voxel_fea,
)

_HEX_CORNER_OFFSETS = np.array(
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

_NEIGHBORS_6 = (
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)

# Direction (dx, dy, dz) -> outward-wound CCW quad corner offsets on unit cube [0, 1]^3
_EXPOSED_FACE_DEFS: tuple[tuple[tuple[int, int, int], np.ndarray], ...] = (
    ((-1, 0, 0), np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]], dtype=int)),
    ((1, 0, 0), np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]], dtype=int)),
    ((0, -1, 0), np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], dtype=int)),
    ((0, 1, 0), np.array([[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=int)),
    ((0, 0, -1), np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=int)),
    ((0, 0, 1), np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=int)),
)


@dataclass
class TopologyOptResult:
    """Result container for 3D SIMP topology optimization and verified final Hex8 FEA."""

    part_name: str
    material_id: str
    is_fdm_anisotropic: bool
    voxel_size_mm: float
    target_volume_fraction: float
    achieved_volume_fraction: float
    penalization: float
    filter_radius_voxels: float
    iterations_run: int
    baseline_elements: int
    optimized_elements: int
    baseline_mass_g: float
    optimized_mass_g: float
    mass_reduction_pct: float
    baseline_compliance_mj: float
    optimized_compliance_mj: float
    baseline_stiffness_n_per_mm: float
    optimized_stiffness_n_per_mm: float
    specific_stiffness_gain_pct: float
    max_displacement_um: float
    max_von_mises_mpa: float
    safety_factor: float
    failure_mode: str
    applied_force_n: list[float]
    history: list[dict] = field(default_factory=list)
    voxels: list[dict] = field(default_factory=list)
    optimized_grid: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0, 0), dtype=bool),
        repr=False,
    )
    voxel_dims_mm: tuple[float, float, float] | None = None

    def to_dict(self, include_voxels: bool = True) -> dict:
        """Serialize optimization results into JSON-friendly units without the raw 3D grid."""
        data = {
            "part_name": self.part_name,
            "material_id": self.material_id,
            "is_fdm_anisotropic": self.is_fdm_anisotropic,
            "voxel_size_mm": round(self.voxel_size_mm, 3),
            "target_volume_fraction": round(self.target_volume_fraction, 4),
            "achieved_volume_fraction": round(self.achieved_volume_fraction, 4),
            "penalization": round(self.penalization, 2),
            "filter_radius_voxels": round(self.filter_radius_voxels, 2),
            "iterations_run": self.iterations_run,
            "baseline_elements": self.baseline_elements,
            "optimized_elements": self.optimized_elements,
            "baseline_mass_g": round(self.baseline_mass_g, 2),
            "optimized_mass_g": round(self.optimized_mass_g, 2),
            "mass_reduction_pct": round(self.mass_reduction_pct, 2),
            "baseline_compliance_mj": round(self.baseline_compliance_mj, 4),
            "optimized_compliance_mj": round(self.optimized_compliance_mj, 4),
            "baseline_stiffness_n_per_mm": round(self.baseline_stiffness_n_per_mm, 2),
            "optimized_stiffness_n_per_mm": round(self.optimized_stiffness_n_per_mm, 2),
            "specific_stiffness_gain_pct": round(self.specific_stiffness_gain_pct, 2),
            "max_displacement_um": round(self.max_displacement_um, 2),
            "max_von_mises_mpa": round(self.max_von_mises_mpa, 3),
            "safety_factor": round(min(self.safety_factor, 999.0), 2),
            "failure_mode": self.failure_mode,
            "applied_force_n": [round(float(v), 2) for v in self.applied_force_n],
            "history": self.history,
        }
        if include_voxels:
            data["voxels"] = self.voxels
        return data


@dataclass
class _VoxelMeshSystem:
    occ: np.ndarray
    elem_dofs: np.ndarray
    n_dof: int
    free_dofs: np.ndarray
    F_free: np.ndarray
    free_rows: np.ndarray
    free_cols: np.ndarray
    free_elem_idx: np.ndarray
    free_ke_vals: np.ndarray
    is_fixed_elem: np.ndarray
    is_loaded_elem: np.ndarray


def _build_voxel_mesh_system(
    grid: np.ndarray,
    Ke: np.ndarray,
    fixed_face: FaceName,
    load_face: FaceName,
    force_n: tuple[float, float, float],
) -> _VoxelMeshSystem:
    """Build node/DOF mappings, boundary conditions, and precomputed free-DOF sparse indices."""
    occ = np.argwhere(grid)
    n_elem = len(occ)
    if n_elem == 0:
        raise ValueError("Empty voxel grid provided to topology optimizer")

    _, ny, nz = grid.shape
    elem_corners = occ[:, None, :] + _HEX_CORNER_OFFSETS[None, :, :]
    flat_keys = (
        elem_corners[:, :, 0] * ((ny + 1) * (nz + 1))
        + elem_corners[:, :, 1] * (nz + 1)
        + elem_corners[:, :, 2]
    )
    uniq_keys, inverse = np.unique(flat_keys.ravel(), return_inverse=True)
    elem_nodes = inverse.reshape(n_elem, 8)
    n_nodes = len(uniq_keys)
    n_dof = 3 * n_nodes

    node_ix = uniq_keys // ((ny + 1) * (nz + 1))
    rem = uniq_keys % ((ny + 1) * (nz + 1))
    node_iy = rem // (nz + 1)
    node_iz = rem % (nz + 1)
    node_grid_coords = np.stack([node_ix, node_iy, node_iz], axis=1)

    min_c = node_grid_coords.min(axis=0)
    max_c = node_grid_coords.max(axis=0)

    def _select_face_nodes(face: FaceName) -> np.ndarray:
        axis = {"x": 0, "y": 1, "z": 2}[face[-1]]
        target = min_c[axis] if face.startswith("min") else max_c[axis]
        return np.where(node_grid_coords[:, axis] == target)[0]

    fixed_nodes = _select_face_nodes(fixed_face)
    loaded_nodes = _select_face_nodes(load_face)
    loaded_nodes = np.setdiff1d(loaded_nodes, fixed_nodes)
    if len(loaded_nodes) == 0:
        loaded_nodes = np.setdiff1d(np.arange(n_nodes), fixed_nodes)

    fixed_dofs = np.concatenate([3 * fixed_nodes, 3 * fixed_nodes + 1, 3 * fixed_nodes + 2])
    free_mask = np.ones(n_dof, dtype=bool)
    free_mask[fixed_dofs] = False
    free_dofs = np.where(free_mask)[0]
    if len(free_dofs) == 0:
        raise ValueError("All degrees of freedom are constrained; no free DOFs to optimize")

    F = np.zeros(n_dof, dtype=float)
    f_vec = np.asarray(force_n, dtype=float)
    node_weights = np.bincount(elem_nodes.ravel(), minlength=n_nodes).astype(float)
    w_loaded = node_weights[loaded_nodes]
    w_sum = float(np.sum(w_loaded))
    if w_sum > 0.0:
        fracs = w_loaded / w_sum
        F[3 * loaded_nodes + 0] = f_vec[0] * fracs
        F[3 * loaded_nodes + 1] = f_vec[1] * fracs
        F[3 * loaded_nodes + 2] = f_vec[2] * fracs

    elem_dofs = np.empty((n_elem, 24), dtype=int)
    elem_dofs[:, 0::3] = 3 * elem_nodes
    elem_dofs[:, 1::3] = 3 * elem_nodes + 1
    elem_dofs[:, 2::3] = 3 * elem_nodes + 2

    # Precompute free-DOF COO indices so each SIMP step only scales element stiffness values
    dof_to_free = np.full(n_dof, -1, dtype=int)
    dof_to_free[free_dofs] = np.arange(len(free_dofs), dtype=int)
    elem_free_dofs = dof_to_free[elem_dofs]

    raw_rows = np.repeat(elem_free_dofs, 24, axis=1).ravel()
    raw_cols = np.tile(elem_free_dofs, (1, 24)).ravel()
    active_entries = (raw_rows >= 0) & (raw_cols >= 0)

    free_rows = raw_rows[active_entries]
    free_cols = raw_cols[active_entries]
    free_elem_idx = np.repeat(np.arange(n_elem, dtype=int), 576)[active_entries]
    free_ke_vals = np.tile(Ke.ravel(), n_elem)[active_entries]

    is_fixed_elem = np.isin(elem_nodes, fixed_nodes).any(axis=1)
    is_loaded_elem = np.isin(elem_nodes, loaded_nodes).any(axis=1)

    return _VoxelMeshSystem(
        occ=occ,
        elem_dofs=elem_dofs,
        n_dof=n_dof,
        free_dofs=free_dofs,
        F_free=F[free_dofs],
        free_rows=free_rows,
        free_cols=free_cols,
        free_elem_idx=free_elem_idx,
        free_ke_vals=free_ke_vals,
        is_fixed_elem=is_fixed_elem,
        is_loaded_elem=is_loaded_elem,
    )


def _solve_system_displacements(
    sys_data: _VoxelMeshSystem,
    scale: np.ndarray,
    diag_mat: sparse.spmatrix,
) -> np.ndarray:
    """Assemble K_ff(scale) and solve K_ff u_f = F_f in strict SI units."""
    n_free = len(sys_data.free_dofs)
    vals = scale[sys_data.free_elem_idx] * sys_data.free_ke_vals
    K_ff = sparse.coo_matrix(
        (vals, (sys_data.free_rows, sys_data.free_cols)),
        shape=(n_free, n_free),
    ).tocsr()
    K_ff = K_ff + diag_mat
    return np.asarray(spsolve(K_ff, sys_data.F_free, permc_spec="MMD_AT_PLUS_A"), dtype=float)


def _build_sensitivity_filter(
    occ: np.ndarray,
    radius_voxels: float,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Precompute vectorized Sigmund conical sensitivity filter matrix H and row sums Hs."""
    n_elem = len(occ)
    if radius_voxels <= 1.0:
        H = sparse.eye(n_elem, dtype=float, format="csr")
        Hs = np.ones(n_elem, dtype=float)
        return H, Hs

    coords = occ.astype(float)
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=radius_voxels, output_type="ndarray")

    if len(pairs) > 0:
        i_idx = pairs[:, 0]
        j_idx = pairs[:, 1]
        dists = np.linalg.norm(coords[i_idx] - coords[j_idx], axis=1)
        weights = np.maximum(0.0, radius_voxels - dists)
        keep = weights > 0.0
        i_idx = i_idx[keep]
        j_idx = j_idx[keep]
        weights = weights[keep]
        rows = np.concatenate([np.arange(n_elem), i_idx, j_idx])
        cols = np.concatenate([np.arange(n_elem), j_idx, i_idx])
        vals = np.concatenate([np.full(n_elem, radius_voxels), weights, weights])
    else:
        rows = np.arange(n_elem)
        cols = np.arange(n_elem)
        vals = np.full(n_elem, radius_voxels)

    H = sparse.coo_matrix((vals, (rows, cols)), shape=(n_elem, n_elem)).tocsr()
    Hs = np.asarray(H.sum(axis=1)).ravel()
    return H, Hs


def _oc_update(
    rho: np.ndarray,
    dc_filt: np.ndarray,
    target_volume_fraction: float,
    passive_mask: np.ndarray,
    preserve_boundaries: bool,
    move: float = 0.2,
    rho_min: float = 1e-3,
) -> np.ndarray:
    """Scale-invariant Optimality Criteria density update with normalized Lagrange bisection."""
    n_elem = len(rho)
    target_vol = target_volume_fraction * n_elem
    neg_dc = np.maximum(0.0, -dc_filt)
    dc_max = float(np.max(neg_dc))
    if dc_max <= 0.0:
        return rho.copy()

    # Normalize sensitivities by dc_max so the bisection bracket is invariant to SI force/modulus scale
    d_norm = neg_dc / dc_max
    lower_lim = np.maximum(rho_min, rho - move)
    upper_lim = np.minimum(1.0, rho + move)

    def _trial(lam_norm: float) -> np.ndarray:
        if lam_norm <= 0.0:
            rho_t = np.where(d_norm > 0.0, upper_lim, lower_lim)
        else:
            rho_cand = rho * np.sqrt(d_norm / lam_norm)
            rho_t = np.maximum(lower_lim, np.minimum(upper_lim, rho_cand))
        if preserve_boundaries:
            rho_t[passive_mask] = 1.0
        return rho_t

    rho_zero = _trial(0.0)
    if float(np.sum(rho_zero)) <= target_vol:
        return rho_zero

    l1 = 0.0
    l2 = 1e4
    rho_trial = _trial(l2)
    for _ in range(40):
        if float(np.sum(rho_trial)) <= target_vol or l2 >= 1e16:
            break
        l2 *= 4.0
        rho_trial = _trial(l2)

    rho_new = rho_trial
    for _ in range(80):
        if (l2 - l1) <= 1e-6 * max(l1 + l2, 1e-15):
            break
        lmid = 0.5 * (l1 + l2)
        rho_trial = _trial(lmid)
        rho_new = rho_trial
        if float(np.sum(rho_trial)) > target_vol:
            l1 = lmid
        else:
            l2 = lmid

    return rho_new


def _bfs_from_fixed(
    occ: np.ndarray,
    elem_id_grid: np.ndarray,
    kept_mask: np.ndarray,
    is_fixed_elem: np.ndarray,
) -> np.ndarray:
    """Find all kept voxels reachable from kept fixed voxels via 6-neighbor face adjacency."""
    nx, ny, nz = elem_id_grid.shape
    n_elem = len(occ)
    visited = np.zeros(n_elem, dtype=bool)
    queue: deque[int] = deque()

    for e_idx in np.where(kept_mask & is_fixed_elem)[0]:
        visited[e_idx] = True
        queue.append(int(e_idx))

    while queue:
        cur = queue.popleft()
        cx, cy, cz = int(occ[cur, 0]), int(occ[cur, 1]), int(occ[cur, 2])
        for dx, dy, dz in _NEIGHBORS_6:
            nx_i, ny_i, nz_i = cx + dx, cy + dy, cz + dz
            if 0 <= nx_i < nx and 0 <= ny_i < ny and 0 <= nz_i < nz:
                nb = int(elem_id_grid[nx_i, ny_i, nz_i])
                if nb >= 0 and kept_mask[nb] and not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)

    return visited


def _prune_disconnected_islands(
    grid_shape: tuple[int, int, int],
    occ: np.ndarray,
    rho: np.ndarray,
    kept_mask: np.ndarray,
    is_fixed_elem: np.ndarray,
    is_loaded_elem: np.ndarray,
    preserve_boundaries: bool,
) -> np.ndarray:
    """Remove floating islands not 6-connected to fixed boundary, bridging loaded faces if needed."""
    n_elem = len(occ)
    nx, ny, nz = grid_shape
    elem_id_grid = np.full(grid_shape, -1, dtype=int)
    elem_id_grid[occ[:, 0], occ[:, 1], occ[:, 2]] = np.arange(n_elem, dtype=int)

    kept = kept_mask.copy()
    if not np.any(kept & is_fixed_elem) and np.any(is_fixed_elem):
        fixed_candidates = np.where(is_fixed_elem)[0]
        best_fixed = fixed_candidates[int(np.argmax(rho[fixed_candidates]))]
        kept[best_fixed] = True

    visited = _bfs_from_fixed(occ, elem_id_grid, kept, is_fixed_elem)

    # Bridge any kept loaded boundary voxels disconnected from fixed boundary along highest-rho path
    unreached_loaded = np.where(kept & is_loaded_elem & (~visited))[0]
    if len(unreached_loaded) > 0 and np.any(visited):
        dist = np.full(n_elem, np.inf, dtype=float)
        parent = np.full(n_elem, -1, dtype=int)
        heap: list[tuple[float, int]] = []
        for src in np.where(visited)[0]:
            dist[src] = 0.0
            heapq.heappush(heap, (0.0, int(src)))

        while heap:
            cur_d, cur = heapq.heappop(heap)
            if cur_d > dist[cur]:
                continue
            cx, cy, cz = int(occ[cur, 0]), int(occ[cur, 1]), int(occ[cur, 2])
            for dx, dy, dz in _NEIGHBORS_6:
                nx_i, ny_i, nz_i = cx + dx, cy + dy, cz + dz
                if 0 <= nx_i < nx and 0 <= ny_i < ny and 0 <= nz_i < nz:
                    nb = int(elem_id_grid[nx_i, ny_i, nz_i])
                    if nb < 0:
                        continue
                    step_cost = 1e-3 if kept[nb] else (1.01 - float(rho[nb]))
                    cand_d = cur_d + step_cost
                    if cand_d < dist[nb]:
                        dist[nb] = cand_d
                        parent[nb] = cur
                        heapq.heappush(heap, (cand_d, nb))

        bridged = False
        for target in unreached_loaded:
            if np.isfinite(dist[target]):
                cur = int(target)
                while cur >= 0 and not visited[cur]:
                    kept[cur] = True
                    cur = int(parent[cur])
                bridged = True

        if bridged:
            visited = _bfs_from_fixed(occ, elem_id_grid, kept, is_fixed_elem)

    if preserve_boundaries:
        return visited | (kept & is_loaded_elem)
    return visited


def optimize_solid_topology(
    grid: np.ndarray,
    voxel_size_mm: VoxelSize,
    material_id: str = "pa12_cf_printed",
    fixed_face: FaceName = "min_x",
    load_face: FaceName = "max_x",
    force_n: tuple[float, float, float] = (0.0, 0.0, -50.0),
    target_volume_fraction: float = 0.45,
    iterations: int = 20,
    penalization: float = 3.0,
    filter_radius_voxels: float = 1.5,
    preserve_boundaries: bool = True,
    part_name: str = "Optimized Solid Part",
    fdm_anisotropic: bool | None = None,
) -> TopologyOptResult:
    """Run 3D SIMP compliance minimization on a boolean voxel domain and verify with Hex8 FEA."""
    domain = np.asarray(grid, dtype=bool)
    if domain.ndim != 3:
        raise ValueError(f"Expected 3D boolean grid, got shape {domain.shape}")
    hx_mm, hy_mm, hz_mm = _normalize_voxel_size(voxel_size_mm)
    if not (0.0 < target_volume_fraction <= 1.0):
        raise ValueError(
            f"target_volume_fraction must be in (0, 1], got {target_volume_fraction}"
        )

    n_iters = max(1, int(iterations))
    p = float(penalization)
    r_filt = float(filter_radius_voxels)
    emin_ratio = 1e-3

    mat = get_material(material_id)
    is_fdm = fdm_anisotropic if fdm_anisotropic is not None else bool(mat.is_printed_polymer)
    hx_m, hy_m, hz_m = hx_mm * 1e-3, hy_mm * 1e-3, hz_mm * 1e-3  # meters (strict SI)
    elem_vol_m3 = hx_m * hy_m * hz_m

    D = constitutive_matrix_3d(mat, fdm_anisotropic=is_fdm)
    Ke, _ = hex8_element_stiffness(D, (hx_m, hy_m, hz_m))
    diag_floor = float(Ke[0, 0]) * 1e-10

    sys_data = _build_voxel_mesh_system(domain, Ke, fixed_face, load_face, force_n)
    n_elem = len(sys_data.occ)
    diag_mat = sparse.diags(np.full(len(sys_data.free_dofs), diag_floor), format="csr")

    H, Hs = _build_sensitivity_filter(sys_data.occ, r_filt)

    passive_mask = sys_data.is_fixed_elem | sys_data.is_loaded_elem
    if preserve_boundaries:
        n_passive = int(np.sum(passive_mask))
        n_active = n_elem - n_passive
        target_vol = float(target_volume_fraction) * n_elem
        rho_active = float(np.clip((target_vol - n_passive) / max(n_active, 1), 1e-3, 1.0))
        rho = np.full(n_elem, rho_active, dtype=float)
        rho[passive_mask] = 1.0
    else:
        rho = np.full(n_elem, float(np.clip(target_volume_fraction, 1e-3, 1.0)), dtype=float)

    # Baseline full-solid compliance (J -> mJ)
    u_base_f = _solve_system_displacements(sys_data, np.ones(n_elem, dtype=float), diag_mat)
    baseline_compliance_mj = float(sys_data.F_free @ u_base_f) * 1e3

    history: list[dict] = []
    iterations_run = 0
    u_full = np.zeros(sys_data.n_dof, dtype=float)

    for it in range(1, n_iters + 1):
        scale = emin_ratio + (rho**p) * (1.0 - emin_ratio)
        u_f = _solve_system_displacements(sys_data, scale, diag_mat)
        u_full.fill(0.0)
        u_full[sys_data.free_dofs] = u_f
        u_elem = u_full[sys_data.elem_dofs]

        ce = np.sum((u_elem @ Ke) * u_elem, axis=1)
        ce = np.maximum(0.0, ce)
        compliance_j = float(np.sum(scale * ce))
        compliance_mj = compliance_j * 1e3

        dc = -p * (rho ** (p - 1.0)) * (1.0 - emin_ratio) * ce
        dc_filt = (H @ (rho * dc)) / np.maximum(1e-12, rho * Hs)

        rho_new = _oc_update(
            rho=rho,
            dc_filt=dc_filt,
            target_volume_fraction=target_volume_fraction,
            passive_mask=passive_mask,
            preserve_boundaries=preserve_boundaries,
        )
        max_change = float(np.max(np.abs(rho_new - rho)))
        rho = rho_new
        iterations_run = it

        mass_iter_g = float(np.sum(rho)) * elem_vol_m3 * mat.density_kg_m3 * 1000.0
        history.append(
            {
                "iteration": it,
                "compliance_mj": round(compliance_mj, 4),
                "volume_fraction": round(float(np.mean(rho)), 4),
                "mass_g": round(mass_iter_g, 2),
                "max_change": round(max_change, 4),
            }
        )

        if it >= 5 and max_change < 0.01:
            break

    # Final binary thresholding + 6-neighbor island pruning
    k_target = max(1, int(round(target_volume_fraction * n_elem)))
    kept_mask = np.zeros(n_elem, dtype=bool)
    if preserve_boundaries:
        kept_mask[passive_mask] = True
        n_remaining = max(0, k_target - int(np.sum(passive_mask)))
        if n_remaining > 0:
            active_idx = np.where(~passive_mask)[0]
            top_active = active_idx[np.argsort(-rho[active_idx], kind="mergesort")[:n_remaining]]
            kept_mask[top_active] = True
    else:
        top_all = np.argsort(-rho, kind="mergesort")[:k_target]
        kept_mask[top_all] = True

    kept_mask = _prune_disconnected_islands(
        grid_shape=domain.shape,
        occ=sys_data.occ,
        rho=rho,
        kept_mask=kept_mask,
        is_fixed_elem=sys_data.is_fixed_elem,
        is_loaded_elem=sys_data.is_loaded_elem,
        preserve_boundaries=preserve_boundaries,
    )

    optimized_grid = np.zeros(domain.shape, dtype=bool)
    optimized_grid[
        sys_data.occ[kept_mask, 0],
        sys_data.occ[kept_mask, 1],
        sys_data.occ[kept_mask, 2],
    ] = True

    # Verified final Hex8 FEA on both baseline full domain and binary optimized_grid
    baseline_fea = solve_solid_voxel_fea(
        grid=domain,
        voxel_size_mm=voxel_size_mm,
        material_id=material_id,
        fixed_face=fixed_face,
        load_face=load_face,
        force_n=force_n,
        part_name=part_name,
        fdm_anisotropic=is_fdm,
    )
    optimized_fea = solve_solid_voxel_fea(
        grid=optimized_grid,
        voxel_size_mm=voxel_size_mm,
        material_id=material_id,
        fixed_face=fixed_face,
        load_face=load_face,
        force_n=force_n,
        part_name=part_name,
        fdm_anisotropic=is_fdm,
    )

    opt_sys = _build_voxel_mesh_system(optimized_grid, Ke, fixed_face, load_face, force_n)
    opt_diag_mat = sparse.diags(np.full(len(opt_sys.free_dofs), diag_floor), format="csr")
    u_opt_f = _solve_system_displacements(
        opt_sys, np.ones(len(opt_sys.occ), dtype=float), opt_diag_mat
    )
    optimized_compliance_mj = float(opt_sys.F_free @ u_opt_f) * 1e3

    base_mass = float(baseline_fea.mass_g)
    opt_mass = float(optimized_fea.mass_g)
    mass_reduction_pct = (1.0 - opt_mass / max(base_mass, 1e-12)) * 100.0

    base_stiff = float(baseline_fea.stiffness_n_per_mm)
    opt_stiff = float(optimized_fea.stiffness_n_per_mm)
    specific_stiffness_gain_pct = (
        (opt_stiff / max(opt_mass, 1e-12)) / max(base_stiff / max(base_mass, 1e-12), 1e-12) - 1.0
    ) * 100.0
    achieved_volume_fraction = float(optimized_fea.num_elements) / float(n_elem)

    # Map verified displacements and stresses onto all original design-domain voxels
    occ_opt = np.argwhere(optimized_grid)
    opt_voxel_map = {
        (int(ix), int(iy), int(iz)): vox
        for (ix, iy, iz), vox in zip(occ_opt, optimized_fea.voxels)
    }

    voxels_out: list[dict] = []
    for e_idx in range(n_elem):
        ix = int(sys_data.occ[e_idx, 0])
        iy = int(sys_data.occ[e_idx, 1])
        iz = int(sys_data.occ[e_idx, 2])
        key = (ix, iy, iz)
        density_val = round(float(rho[e_idx]), 3)
        if key in opt_voxel_map:
            ov = opt_voxel_map[key]
            voxels_out.append(
                {
                    "center_mm": ov["center_mm"],
                    "disp_mm": ov["disp_mm"],
                    "von_mises_mpa": ov["von_mises_mpa"],
                    "density": density_val,
                    "kept": True,
                    "fixed": bool(ov["fixed"]),
                    "loaded": bool(ov["loaded"]),
                }
            )
        else:
            voxels_out.append(
                {
                    "center_mm": [
                        round((float(ix) + 0.5) * hx_mm, 3),
                        round((float(iy) + 0.5) * hy_mm, 3),
                        round((float(iz) + 0.5) * hz_mm, 3),
                    ],
                    "disp_mm": [0.0, 0.0, 0.0],
                    "von_mises_mpa": 0.0,
                    "density": density_val,
                    "kept": False,
                    "fixed": bool(sys_data.is_fixed_elem[e_idx]),
                    "loaded": bool(sys_data.is_loaded_elem[e_idx]),
                }
            )

    return TopologyOptResult(
        part_name=part_name,
        material_id=material_id,
        is_fdm_anisotropic=is_fdm,
        voxel_size_mm=_scalar_voxel_size_mm(voxel_size_mm),
        target_volume_fraction=float(target_volume_fraction),
        achieved_volume_fraction=achieved_volume_fraction,
        penalization=p,
        filter_radius_voxels=r_filt,
        iterations_run=iterations_run,
        baseline_elements=n_elem,
        optimized_elements=int(optimized_fea.num_elements),
        baseline_mass_g=base_mass,
        optimized_mass_g=opt_mass,
        mass_reduction_pct=mass_reduction_pct,
        baseline_compliance_mj=baseline_compliance_mj,
        optimized_compliance_mj=optimized_compliance_mj,
        baseline_stiffness_n_per_mm=base_stiff,
        optimized_stiffness_n_per_mm=opt_stiff,
        specific_stiffness_gain_pct=specific_stiffness_gain_pct,
        max_displacement_um=float(optimized_fea.max_displacement_um),
        max_von_mises_mpa=float(optimized_fea.max_von_mises_mpa),
        safety_factor=float(optimized_fea.safety_factor),
        failure_mode=str(optimized_fea.failure_mode),
        applied_force_n=[float(v) for v in force_n],
        history=history,
        voxels=voxels_out,
        optimized_grid=optimized_grid,
        voxel_dims_mm=(hx_mm, hy_mm, hz_mm),
    )


def optimize_preset_topology(
    preset: PartPreset,
    material_id: str = "pa12_cf_printed",
    force_n: tuple[float, float, float] | None = None,
    target_volume_fraction: float = 0.45,
    iterations: int = 20,
    penalization: float = 3.0,
    filter_radius_voxels: float = 1.5,
    preserve_boundaries: bool = True,
    fdm_anisotropic: bool | None = None,
) -> TopologyOptResult:
    """Run 3D SIMP topology optimization on one of PrintSim's parametric structural part presets."""
    grid, h_mm, fixed_face, load_face, default_f = generate_parametric_part_grid(preset)
    return optimize_solid_topology(
        grid=grid,
        voxel_size_mm=h_mm,
        material_id=material_id,
        fixed_face=fixed_face,
        load_face=load_face,
        force_n=force_n if force_n is not None else default_f,
        target_volume_fraction=target_volume_fraction,
        iterations=iterations,
        penalization=penalization,
        filter_radius_voxels=filter_radius_voxels,
        preserve_boundaries=preserve_boundaries,
        part_name=f"{preset.replace('_', ' ').title()} (Topology Optimized)",
        fdm_anisotropic=fdm_anisotropic,
    )


def voxel_grid_to_trimesh(
    grid: np.ndarray,
    voxel_size_mm: VoxelSize,
    smooth_iterations: int = 0,
) -> trimesh.Trimesh:
    """Convert a 3D boolean voxel grid into a watertight, outward-wound Trimesh without skimage."""
    domain = np.asarray(grid, dtype=bool)
    if domain.ndim != 3:
        raise ValueError(f"Expected 3D boolean grid, got shape {domain.shape}")
    hx_mm, hy_mm, hz_mm = _normalize_voxel_size(voxel_size_mm)
    if not np.any(domain):
        raise ValueError("Cannot build STL mesh from an empty voxel grid")

    nx, ny, nz = domain.shape
    padded = np.zeros((nx + 2, ny + 2, nz + 2), dtype=bool)
    padded[1:-1, 1:-1, 1:-1] = domain

    quad_corner_coords: list[np.ndarray] = []
    for (dx, dy, dz), quad_offsets in _EXPOSED_FACE_DEFS:
        exposed = domain & (~padded[1 + dx : nx + 1 + dx, 1 + dy : ny + 1 + dy, 1 + dz : nz + 1 + dz])
        if not np.any(exposed):
            continue
        face_voxels = np.argwhere(exposed)
        quad_corner_coords.append(face_voxels[:, None, :] + quad_offsets[None, :, :])

    all_corners = np.concatenate(quad_corner_coords, axis=0)  # (N_quads, 4, 3)
    stride_x = (ny + 1) * (nz + 1)
    stride_y = nz + 1
    corner_keys = (
        all_corners[:, :, 0] * stride_x
        + all_corners[:, :, 1] * stride_y
        + all_corners[:, :, 2]
    )

    uniq_keys, inverse = np.unique(corner_keys.ravel(), return_inverse=True)
    vx = uniq_keys // stride_x
    rem = uniq_keys % stride_x
    vy = rem // stride_y
    vz = rem % stride_y
    vertices = np.stack([vx * hx_mm, vy * hy_mm, vz * hz_mm], axis=1).astype(float)

    quads = inverse.reshape(-1, 4)
    faces = np.vstack([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])

    import trimesh

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    if smooth_iterations > 0:
        trimesh.smoothing.filter_taubin(mesh, iterations=int(smooth_iterations))
    return mesh


def topology_result_to_stl_bytes(
    result: TopologyOptResult,
    smooth_iterations: int = 0,
) -> bytes:
    """Serialize an optimized topology voxel grid into binary STL bytes."""
    vox_size: VoxelSize = result.voxel_dims_mm if result.voxel_dims_mm is not None else result.voxel_size_mm
    mesh = voxel_grid_to_trimesh(
        result.optimized_grid,
        voxel_size_mm=vox_size,
        smooth_iterations=smooth_iterations,
    )
    exported = mesh.export(file_type="stl")
    if isinstance(exported, str):
        return exported.encode("utf-8")
    return bytes(exported)


def export_topology_stl(
    result: TopologyOptResult,
    output_path: str | Path,
    smooth_iterations: int = 0,
) -> Path:
    """Write a watertight STL mesh of the topology-optimized part to disk."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(topology_result_to_stl_bytes(result, smooth_iterations=smooth_iterations))
    return out_path
