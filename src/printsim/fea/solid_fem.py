"""3D Solid Continuum Finite Element Analysis (8-node Hex8 brick elements) for brackets, mounts & STL parts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Literal

import numpy as np
import trimesh
from scipy import sparse
from scipy.sparse.linalg import spsolve

from printsim.schema import MaterialSpec, get_material

PartPreset = Literal["corner_gusset", "xy_motor_mount", "toolhead_carriage", "cantilever_bar"]
FaceName = Literal["min_x", "max_x", "min_y", "max_y", "min_z", "max_z"]
VoxelSize = float | tuple[float, float, float]


def _normalize_voxel_size(voxel_size: VoxelSize | Sequence[float]) -> tuple[float, float, float]:
    """Normalize scalar or 3-tuple voxel size (hx, hy, hz) and verify positivity."""
    if isinstance(voxel_size, (int, float, np.floating, np.integer)):
        h = float(voxel_size)
        if h <= 0.0:
            raise ValueError(f"voxel_size must be positive, got {voxel_size}")
        return (h, h, h)
    arr = tuple(float(v) for v in voxel_size)
    if len(arr) != 3:
        raise ValueError(f"Expected scalar or 3-tuple voxel_size, got {voxel_size}")
    if any(v <= 0.0 for v in arr):
        raise ValueError(f"All voxel dimensions must be positive, got {arr}")
    return (arr[0], arr[1], arr[2])


def _scalar_voxel_size_mm(voxel_size_mm: VoxelSize | Sequence[float]) -> float:
    """Return exact scalar float if isotropic, else volume-equivalent geometric mean."""
    if isinstance(voxel_size_mm, (int, float, np.floating, np.integer)):
        return float(voxel_size_mm)
    hx, hy, hz = _normalize_voxel_size(voxel_size_mm)
    return float((hx * hy * hz) ** (1.0 / 3.0))


@dataclass
class SolidFEAResult:
    part_name: str
    material_id: str
    is_fdm_anisotropic: bool
    voxel_size_mm: float
    num_elements: int
    num_nodes: int
    num_dofs: int
    mass_g: float
    applied_force_n: list[float]
    max_displacement_um: float
    stiffness_n_per_mm: float
    max_von_mises_mpa: float
    max_z_tensile_mpa: float
    yield_strength_mpa: float
    z_layer_strength_mpa: float
    safety_factor: float
    failure_mode: str  # "safe", "isotropic_yield", "fdm_layer_delamination"
    voxels: list[dict] = field(default_factory=list)  # [{center_mm, disp_mm, von_mises_mpa, fixed, loaded}]
    voxel_dims_mm: tuple[float, float, float] | None = None

    def to_dict(self) -> dict:
        return {
            "part_name": self.part_name,
            "material_id": self.material_id,
            "is_fdm_anisotropic": self.is_fdm_anisotropic,
            "voxel_size_mm": round(self.voxel_size_mm, 3),
            "num_elements": self.num_elements,
            "num_nodes": self.num_nodes,
            "num_dofs": self.num_dofs,
            "mass_g": round(self.mass_g, 2),
            "applied_force_n": [round(v, 2) for v in self.applied_force_n],
            "max_displacement_um": round(self.max_displacement_um, 2),
            "stiffness_n_per_mm": round(self.stiffness_n_per_mm, 2),
            "max_von_mises_mpa": round(self.max_von_mises_mpa, 3),
            "max_z_tensile_mpa": round(self.max_z_tensile_mpa, 3),
            "yield_strength_mpa": round(self.yield_strength_mpa, 1),
            "z_layer_strength_mpa": round(self.z_layer_strength_mpa, 1),
            "safety_factor": round(min(self.safety_factor, 999.0), 2),
            "failure_mode": self.failure_mode,
            "voxels": self.voxels,
        }


def constitutive_matrix_3d(mat: MaterialSpec, fdm_anisotropic: bool = False) -> np.ndarray:
    """Build 6x6 elasticity matrix D (Pa) for isotropic or transversely isotropic FDM-printed material."""
    E = mat.youngs_modulus_pa
    nu = mat.poisson_ratio

    if not fdm_anisotropic:
        c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
        D = np.zeros((6, 6), dtype=float)
        D[0:3, 0:3] = c * np.array(
            [
                [1.0 - nu, nu, nu],
                [nu, 1.0 - nu, nu],
                [nu, nu, 1.0 - nu],
            ]
        )
        g = E / (2.0 * (1.0 + nu))
        D[3, 3] = D[4, 4] = D[5, 5] = g
        return D

    # Transversely isotropic compliance matrix S for FDM printed parts (XY strong, Z layer-bond weaker)
    E_xy = E
    E_z = E * float(mat.e3_ratio)
    nu_xy = nu
    # Enforce thermodynamic positive-definiteness: 1 - nu_xy - 2 * nu_xz^2 * (E_z / E_xy) > 0
    max_nu_xz = 0.9 * math.sqrt(max(1.0 - nu_xy, 1e-6) * E_xy / max(2.0 * E_z, 1e-12))
    nu_xz = min(nu * 0.85, max_nu_xz)
    G_xy = E_xy / (2.0 * (1.0 + nu_xy))
    G_xz = E_z / (2.0 * (1.0 + nu_xz))

    S = np.array(
        [
            [1.0 / E_xy, -nu_xy / E_xy, -nu_xz / E_xy, 0.0, 0.0, 0.0],
            [-nu_xy / E_xy, 1.0 / E_xy, -nu_xz / E_xy, 0.0, 0.0, 0.0],
            [-nu_xz / E_xy, -nu_xz / E_xy, 1.0 / E_z, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0 / G_xy, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0 / G_xz, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / G_xz],
        ],
        dtype=float,
    )
    D = np.linalg.inv(S)
    return 0.5 * (D + D.T)


_HEX8_CORNER_SIGNS = np.array(
    [
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ],
    dtype=float,
)


def _hex8_b_matrix(
    xi: float,
    eta: float,
    zeta: float,
    h: VoxelSize,
) -> np.ndarray:
    """6x24 strain-displacement matrix B for a Hex8 brick of edge lengths h or (hx, hy, hz)."""
    hx, hy, hz = _normalize_voxel_size(h)
    inv_jx = 2.0 / hx  # d(xi)/dx = 2/hx
    inv_jy = 2.0 / hy  # d(eta)/dy = 2/hy
    inv_jz = 2.0 / hz  # d(zeta)/dz = 2/hz
    B = np.zeros((6, 24), dtype=float)
    for i in range(8):
        sx, sy, sz = _HEX8_CORNER_SIGNS[i]
        dN_dx = 0.125 * sx * (1.0 + sy * eta) * (1.0 + sz * zeta) * inv_jx
        dN_dy = 0.125 * sy * (1.0 + sx * xi) * (1.0 + sz * zeta) * inv_jy
        dN_dz = 0.125 * sz * (1.0 + sx * xi) * (1.0 + sy * eta) * inv_jz
        col = 3 * i
        B[0, col + 0] = dN_dx
        B[1, col + 1] = dN_dy
        B[2, col + 2] = dN_dz
        B[3, col + 0] = dN_dy
        B[3, col + 1] = dN_dx
        B[4, col + 1] = dN_dz
        B[4, col + 2] = dN_dy
        B[5, col + 0] = dN_dz
        B[5, col + 2] = dN_dx
    return B


def hex8_element_stiffness(
    D: np.ndarray,
    h: VoxelSize,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact 24x24 Hex8 stiffness matrix Ke via 2x2x2 Gauss quadrature and center B0."""
    hx, hy, hz = _normalize_voxel_size(h)
    gp = 1.0 / math.sqrt(3.0)
    det_j = (hx * hy * hz) * 0.125
    Ke = np.zeros((24, 24), dtype=float)
    for xi in (-gp, gp):
        for eta in (-gp, gp):
            for zeta in (-gp, gp):
                B = _hex8_b_matrix(xi, eta, zeta, (hx, hy, hz))
                Ke += (B.T @ D @ B) * det_j
    B_center = _hex8_b_matrix(0.0, 0.0, 0.0, (hx, hy, hz))
    return 0.5 * (Ke + Ke.T), B_center


def hex8_element_mass(rho_kg_m3: float, h_m: VoxelSize) -> np.ndarray:
    """Compute exact 24x24 consistent Hex8 element mass matrix via 2x2x2 Gauss quadrature."""
    hx, hy, hz = _normalize_voxel_size(h_m)
    gp = 1.0 / math.sqrt(3.0)
    det_j = (hx * hy * hz) * 0.125
    m_scalar = np.zeros((8, 8), dtype=float)
    for xi in (-gp, gp):
        for eta in (-gp, gp):
            for zeta in (-gp, gp):
                N = (
                    0.125
                    * (1.0 + _HEX8_CORNER_SIGNS[:, 0] * xi)
                    * (1.0 + _HEX8_CORNER_SIGNS[:, 1] * eta)
                    * (1.0 + _HEX8_CORNER_SIGNS[:, 2] * zeta)
                )
                m_scalar += np.outer(N, N) * (rho_kg_m3 * det_j)

    Me = np.zeros((24, 24), dtype=float)
    for d in range(3):
        Me[d::3, d::3] = m_scalar
    return 0.5 * (Me + Me.T)


def generate_parametric_part_grid(
    preset: PartPreset,
) -> tuple[np.ndarray, float, FaceName, FaceName, tuple[float, float, float]]:
    """Build a 3D boolean voxel occupancy grid for common 3D printer structural parts.

    Returns (grid, voxel_size_mm, fixed_face, load_face, default_force_n).
    """
    if preset == "corner_gusset":
        # 40 x 40 x 20 mm L-shaped corner gusset with diagonal stiffening web (voxel = 2.5mm -> 16x16x8)
        nx, ny, nz = 16, 16, 8
        h_mm = 2.5
        grid = np.zeros((nx, ny, nz), dtype=bool)
        # Base flange along X (y in [0..3]) and vertical arm along Y (x in [0..3])
        grid[:, :4, :] = True
        grid[:4, :, :] = True
        # Triangular stiffening rib in center plane (z in [2..6])
        for ix in range(nx):
            for iy in range(ny):
                if ix + iy <= 15:
                    grid[ix, iy, 2:6] = True
        # Bolt clearance holes on arms
        grid[10:12, :4, 3:5] = False
        grid[:4, 10:12, 3:5] = False
        return grid, h_mm, "min_y", "max_y", (60.0, 0.0, 0.0)

    if preset == "xy_motor_mount":
        # 60 x 52 x 16 mm CoreXY NEMA17 motor mount plate with side stiffening walls (voxel = 4mm -> 15x13x4)
        nx, ny, nz = 15, 13, 4
        h_mm = 4.0
        grid = np.zeros((nx, ny, nz), dtype=bool)
        # Main horizontal plate (top 2 layers)
        grid[:, :, 2:4] = True
        # Rear mounting flange bolted to frame extrusion
        grid[:, :2, :] = True
        # Side stiffening ribs
        grid[0:2, :, :] = True
        grid[-2:, :, :] = True
        # Central NEMA17 pilot bore cutout
        cx, cy = 7, 7
        for ix in range(nx):
            for iy in range(ny):
                if (ix - cx) ** 2 + (iy - cy) ** 2 <= 2.3**2:
                    grid[ix, iy, :] = False
        return grid, h_mm, "min_y", "max_y", (0.0, 45.0, -25.0)

    if preset == "toolhead_carriage":
        # 56 x 48 x 12 mm direct-drive carriage backing plate with rail bosses (voxel = 4mm -> 14x12x3)
        nx, ny, nz = 14, 12, 3
        h_mm = 4.0
        grid = np.zeros((nx, ny, nz), dtype=bool)
        grid[:, :, 0:2] = True
        # Stiffening perimeter ribs on front face
        grid[0:2, :, 2] = True
        grid[-2:, :, 2] = True
        grid[:, 0:2, 2] = True
        grid[:, -2:, 2] = True
        # Weight-reduction pocket in center
        grid[5:9, 4:8, :] = False
        return grid, h_mm, "min_z", "max_z", (35.0, -20.0, -10.0)

    if preset == "cantilever_bar":
        # Clean 60 x 12 x 12 mm rectangular cantilever bar for exact analytical benchmark validation
        nx, ny, nz = 20, 4, 4
        h_mm = 3.0
        grid = np.ones((nx, ny, nz), dtype=bool)
        return grid, h_mm, "min_x", "max_x", (0.0, 0.0, -50.0)

    raise ValueError(f"Unknown solid part preset: {preset}")


def voxelize_stl_bytes(
    stl_bytes: bytes,
    file_type: str = "stl",
    target_max_dim_voxels: int = 18,
) -> tuple[np.ndarray, float]:
    """Load an STL/OBJ byte buffer and voxelize it into a solid 3D boolean grid for Hex8 FEA."""
    loaded = trimesh.load(trimesh.util.wrap_as_stream(stl_bytes), file_type=file_type, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.to_geometry()
    extents = np.asarray(loaded.extents, dtype=float)
    max_ext = float(np.max(extents))
    if max_ext < 1e-6:
        raise ValueError("Uploaded mesh has degenerate zero dimensions")

    pitch_mm = max_ext / float(max(6, min(target_max_dim_voxels, 24)))
    vox = loaded.voxelized(pitch=pitch_mm)
    try:
        vox = vox.fill()
    except Exception:
        pass
    grid = np.asarray(vox.matrix, dtype=bool)
    if grid.sum() == 0:
        raise ValueError("Voxelization produced 0 solid elements; ensure mesh is closed/valid.")
    return grid, float(pitch_mm)


def voxelize_stl_file(path: str | Path, target_max_dim_voxels: int = 18) -> tuple[np.ndarray, float]:
    p = Path(path)
    return voxelize_stl_bytes(p.read_bytes(), file_type=p.suffix.lstrip(".") or "stl", target_max_dim_voxels=target_max_dim_voxels)


def solve_solid_voxel_fea(
    grid: np.ndarray,
    voxel_size_mm: VoxelSize,
    material_id: str = "abs_printed",
    fixed_face: FaceName = "min_x",
    load_face: FaceName = "max_x",
    force_n: tuple[float, float, float] = (0.0, 0.0, -50.0),
    part_name: str = "Custom Solid Part",
    fdm_anisotropic: bool | None = None,
) -> SolidFEAResult:
    """Solve 3D linear elastic Hex8 FEA on a boolean voxel grid using SciPy sparse direct solver."""
    mat = get_material(material_id)
    is_fdm = fdm_anisotropic if fdm_anisotropic is not None else bool(mat.is_printed_polymer)
    hx_mm, hy_mm, hz_mm = _normalize_voxel_size(voxel_size_mm)
    hx_m, hy_m, hz_m = hx_mm * 1e-3, hy_mm * 1e-3, hz_mm * 1e-3  # strict SI meters

    occ = np.argwhere(grid)
    n_elem = len(occ)
    if n_elem == 0:
        raise ValueError("Empty voxel grid provided to solid FEA solver")

    nx, ny, nz = grid.shape

    # Map occupied element corner grid coordinates (ix..ix+1, iy..iy+1, iz..iz+1) to compact node IDs
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
    elem_corners = occ[:, None, :] + offsets[None, :, :]  # (n_elem, 8, 3)
    flat_keys = (
        elem_corners[:, :, 0] * ((ny + 1) * (nz + 1))
        + elem_corners[:, :, 1] * (nz + 1)
        + elem_corners[:, :, 2]
    )
    uniq_keys, inverse = np.unique(flat_keys, return_inverse=True)
    elem_nodes = inverse.reshape(n_elem, 8)
    n_nodes = len(uniq_keys)
    n_dof = 3 * n_nodes

    # Recover integer grid coords for each unique node
    node_ix = uniq_keys // ((ny + 1) * (nz + 1))
    rem = uniq_keys % ((ny + 1) * (nz + 1))
    node_iy = rem // (nz + 1)
    node_iz = rem % (nz + 1)
    node_grid_coords = np.stack([node_ix, node_iy, node_iz], axis=1)  # (n_nodes, 3)

    # Assemble global sparse stiffness matrix K (vectorized COO -> CSR)
    D = constitutive_matrix_3d(mat, fdm_anisotropic=is_fdm)
    Ke, B_center = hex8_element_stiffness(D, (hx_m, hy_m, hz_m))

    elem_dofs = np.empty((n_elem, 24), dtype=int)
    elem_dofs[:, 0::3] = 3 * elem_nodes
    elem_dofs[:, 1::3] = 3 * elem_nodes + 1
    elem_dofs[:, 2::3] = 3 * elem_nodes + 2

    rows = np.repeat(elem_dofs, 24, axis=1).ravel()
    cols = np.tile(elem_dofs, (1, 24)).ravel()
    vals = np.tile(Ke.ravel(), n_elem)
    K = sparse.coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof)).tocsr()

    # Determine fixed and loaded node sets from occupied bounding box faces
    min_c = node_grid_coords.min(axis=0)
    max_c = node_grid_coords.max(axis=0)

    def _select_face_nodes(face: FaceName) -> np.ndarray:
        axis = {"x": 0, "y": 1, "z": 2}[face[-1]]
        target = min_c[axis] if face.startswith("min") else max_c[axis]
        return np.where(node_grid_coords[:, axis] == target)[0]

    fixed_nodes = _select_face_nodes(fixed_face)
    loaded_nodes = _select_face_nodes(load_face)
    # Exclude any overlap so load isn't lost on fixed boundary
    loaded_nodes = np.setdiff1d(loaded_nodes, fixed_nodes)
    if len(loaded_nodes) == 0:
        loaded_nodes = np.setdiff1d(np.arange(n_nodes), fixed_nodes)

    fixed_dofs = np.concatenate([3 * fixed_nodes, 3 * fixed_nodes + 1, 3 * fixed_nodes + 2])
    free_mask = np.ones(n_dof, dtype=bool)
    free_mask[fixed_dofs] = False
    free_dofs = np.where(free_mask)[0]

    # Distribute applied force vector using consistent element-face nodal weighting (1:2:4 corner:edge:interior)
    F = np.zeros(n_dof, dtype=float)
    f_vec = np.asarray(force_n, dtype=float)
    loaded_set_arr = np.zeros(n_nodes, dtype=bool)
    loaded_set_arr[loaded_nodes] = True
    # Count how many elements touch each loaded node -> exact area-consistent traction weight on regular voxel grid
    node_weights = np.bincount(elem_nodes.ravel(), minlength=n_nodes).astype(float)
    w_loaded = node_weights[loaded_nodes]
    w_sum = float(np.sum(w_loaded))
    if w_sum > 0:
        fracs = w_loaded / w_sum
        F[3 * loaded_nodes + 0] = f_vec[0] * fracs
        F[3 * loaded_nodes + 1] = f_vec[1] * fracs
        F[3 * loaded_nodes + 2] = f_vec[2] * fracs

    # Solve K_ff u_f = F_f
    K_ff = K[free_dofs, :][:, free_dofs]
    # Add microscopic diagonal stabilization only for disconnected floating voxels if any
    diag_floor = float(Ke[0, 0]) * 1e-10
    K_ff = K_ff + sparse.diags(np.full(len(free_dofs), diag_floor), format="csr")
    u_f = spsolve(K_ff, F[free_dofs])
    u = np.zeros(n_dof, dtype=float)
    u[free_dofs] = u_f

    # Element stress recovery at element centroid (DB @ u_e)
    DB = D @ B_center  # (6, 24)
    u_elem = u[elem_dofs]  # (n_elem, 24)
    stresses = u_elem @ DB.T  # (n_elem, 6): [sxx, syy, szz, txy, tyz, txz]

    sxx, syy, szz = stresses[:, 0], stresses[:, 1], stresses[:, 2]
    txy, tyz, txz = stresses[:, 3], stresses[:, 4], stresses[:, 5]
    von_mises = np.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3.0 * (txy**2 + tyz**2 + txz**2)
    )
    max_vm_mpa = float(np.max(von_mises)) * 1e-6
    max_z_tensile_mpa = float(max(0.0, np.max(szz))) * 1e-6

    yield_mpa = mat.yield_strength_pa * 1e-6
    z_layer_mpa = yield_mpa * (0.55 if is_fdm else 1.0)
    sf_iso = yield_mpa / max(max_vm_mpa, 1e-6)
    sf_z = z_layer_mpa / max(max_z_tensile_mpa, 1e-6)
    safety_factor = min(sf_iso, sf_z)

    if safety_factor >= 1.0:
        failure_mode = "safe"
    elif is_fdm and sf_z < sf_iso:
        failure_mode = "fdm_layer_delamination"
    else:
        failure_mode = "isotropic_yield"

    # Displacement & effective stiffness
    u_nodes = u.reshape(n_nodes, 3)
    disp_mags = np.linalg.norm(u_nodes, axis=1)
    max_disp_m = float(np.max(disp_mags))
    f_mag = float(np.linalg.norm(f_vec))
    avg_load_disp_m = float(np.mean(np.linalg.norm(u_nodes[loaded_nodes], axis=1)))
    stiffness_n_per_mm = f_mag / max(avg_load_disp_m * 1e3, 1e-9)

    mass_g = n_elem * (hx_m * hy_m * hz_m) * mat.density_kg_m3 * 1000.0

    # Build voxel list for 3D visualization
    fixed_set = set(fixed_nodes.tolist())
    loaded_set = set(loaded_nodes.tolist())
    voxels_out: list[dict] = []
    for e_idx in range(n_elem):
        ix, iy, iz = occ[e_idx]
        e_nds = elem_nodes[e_idx]
        mean_disp_mm = np.mean(u_nodes[e_nds], axis=0) * 1000.0
        is_fix = any(int(nd) in fixed_set for nd in e_nds)
        is_load = any(int(nd) in loaded_set for nd in e_nds)
        voxels_out.append(
            {
                "center_mm": [
                    round((float(ix) + 0.5) * hx_mm, 3),
                    round((float(iy) + 0.5) * hy_mm, 3),
                    round((float(iz) + 0.5) * hz_mm, 3),
                ],
                "disp_mm": [round(float(v), 6) for v in mean_disp_mm],
                "von_mises_mpa": round(float(von_mises[e_idx]) * 1e-6, 3),
                "fixed": is_fix,
                "loaded": is_load,
            }
        )

    return SolidFEAResult(
        part_name=part_name,
        material_id=material_id,
        is_fdm_anisotropic=is_fdm,
        voxel_size_mm=_scalar_voxel_size_mm(voxel_size_mm),
        num_elements=n_elem,
        num_nodes=n_nodes,
        num_dofs=n_dof,
        mass_g=mass_g,
        applied_force_n=[float(v) for v in f_vec],
        max_displacement_um=max_disp_m * 1e6,
        stiffness_n_per_mm=stiffness_n_per_mm,
        max_von_mises_mpa=max_vm_mpa,
        max_z_tensile_mpa=max_z_tensile_mpa,
        yield_strength_mpa=yield_mpa,
        z_layer_strength_mpa=z_layer_mpa,
        safety_factor=safety_factor,
        failure_mode=failure_mode,
        voxels=voxels_out,
        voxel_dims_mm=(hx_mm, hy_mm, hz_mm),
    )


def solve_preset_part_fea(
    preset: PartPreset,
    material_id: str = "abs_printed",
    force_n: tuple[float, float, float] | None = None,
    fdm_anisotropic: bool | None = None,
) -> SolidFEAResult:
    """Convenience wrapper to run 3D Hex8 FEA on a built-in printer part geometry."""
    grid, h_mm, fixed_face, load_face, default_f = generate_parametric_part_grid(preset)
    return solve_solid_voxel_fea(
        grid=grid,
        voxel_size_mm=h_mm,
        material_id=material_id,
        fixed_face=fixed_face,
        load_face=load_face,
        force_n=force_n if force_n is not None else default_f,
        part_name=preset.replace("_", " ").title(),
        fdm_anisotropic=fdm_anisotropic,
    )
