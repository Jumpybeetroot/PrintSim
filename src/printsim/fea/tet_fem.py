"""Conformal 3D tetrahedral continuum FEA powered by Gmsh + scikit-fem + meshio."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Literal

import numpy as np

from printsim.fea.solid_fem import FaceName, PartPreset, constitutive_matrix_3d
from printsim.schema import get_material


def is_tet_fea_available() -> bool:
    """Return True if optional Gmsh + scikit-fem + meshio dependencies are installed."""
    return (
        importlib.util.find_spec("gmsh") is not None
        and importlib.util.find_spec("skfem") is not None
        and importlib.util.find_spec("meshio") is not None
    )


def _require_tet_fea_deps() -> tuple[Any, Any]:
    """Lazily import scikit-fem and meshio (and verify gmsh is installed) without loading libgmsh in-process."""
    if importlib.util.find_spec("gmsh") is None:
        raise RuntimeError(
            "Conformal 3D tetrahedral FEA requires the optional tetfea extra: "
            "`uv sync --extra tetfea` (gmsh, scikit-fem, meshio)."
        )
    try:
        import meshio
        import skfem
    except ImportError as exc:
        raise RuntimeError(
            "Conformal 3D tetrahedral FEA requires the optional tetfea extra: "
            "`uv sync --extra tetfea` (gmsh, scikit-fem, meshio)."
        ) from exc
    return skfem, meshio


@dataclass
class TetFEAResult:
    """Results from 3D conformal tetrahedral continuum FEA."""

    backend: str
    element_type: str
    part_name: str
    material_id: str
    is_fdm_anisotropic: bool
    mesh_size_mm: float
    num_vertices: int
    num_elements: int
    num_dofs: int
    mass_g: float
    volume_mm3: float
    applied_force_n: list[float]
    max_displacement_um: float
    stiffness_n_per_mm: float
    max_von_mises_mpa: float
    max_z_tensile_mpa: float
    yield_strength_mpa: float
    z_layer_strength_mpa: float
    safety_factor: float
    failure_mode: str  # "safe", "isotropic_yield", "fdm_layer_delamination"
    surface_mesh: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_surface_mesh: bool = True) -> dict[str, Any]:
        """Serialize result to JSON-friendly dictionary with explicit unit suffixes."""
        out: dict[str, Any] = {
            "backend": self.backend,
            "element_type": self.element_type,
            "part_name": self.part_name,
            "material_id": self.material_id,
            "is_fdm_anisotropic": self.is_fdm_anisotropic,
            "mesh_size_mm": round(self.mesh_size_mm, 3),
            "num_vertices": self.num_vertices,
            "num_elements": self.num_elements,
            "num_dofs": self.num_dofs,
            "mass_g": round(self.mass_g, 2),
            "volume_mm3": round(self.volume_mm3, 2),
            "applied_force_n": [round(v, 2) for v in self.applied_force_n],
            "max_displacement_um": round(self.max_displacement_um, 2),
            "stiffness_n_per_mm": round(self.stiffness_n_per_mm, 2),
            "max_von_mises_mpa": round(self.max_von_mises_mpa, 3),
            "max_z_tensile_mpa": round(self.max_z_tensile_mpa, 3),
            "yield_strength_mpa": round(self.yield_strength_mpa, 1),
            "z_layer_strength_mpa": round(self.z_layer_strength_mpa, 1),
            "safety_factor": round(min(self.safety_factor, 999.0), 2),
            "failure_mode": self.failure_mode,
        }
        if include_surface_mesh:
            out["surface_mesh"] = self.surface_mesh
        return out


def _mesh_step_with_gmsh(
    step_path: Path,
    mesh_size_mm: float | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Import a STEP solid with OpenCASCADE via isolated Gmsh subprocess and return (points_mm, tets, h_mm)."""
    _require_tet_fea_deps()
    if not step_path.exists():
        raise FileNotFoundError(f"STEP file not found: {step_path}")

    with tempfile.TemporaryDirectory(prefix="printsim_gmsh_") as tmp_dir:
        out_npz = Path(tmp_dir) / "mesh.npz"
        mesh_arg = "none" if mesh_size_mm is None else str(float(mesh_size_mm))
        proc = subprocess.run(
            [sys.executable, "-m", "printsim.fea._gmsh_worker", str(step_path), str(out_npz), mesh_arg],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0 or not out_npz.exists():
            err_msg = (proc.stderr or proc.stdout or f"exit code {proc.returncode}").strip()
            raise RuntimeError(f"Gmsh STEP meshing failed: {err_msg}")
        with np.load(out_npz) as data:
            points_mm = np.ascontiguousarray(data["points_mm"], dtype=np.float64)
            tets = np.ascontiguousarray(data["tets"], dtype=np.int32)
            h_mm = float(data["h_mm"][0])
        return points_mm, tets, h_mm


def _face_axis_and_target(
    face: FaceName,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    mesh_size_m: float,
) -> tuple[int, float, float]:
    """Return (axis_index, target_coord_m, tolerance_m) for a bounding-box face selector."""
    if face not in ("min_x", "max_x", "min_y", "max_y", "min_z", "max_z"):
        raise ValueError(f"Invalid face selector '{face}'.")
    axis = {"x": 0, "y": 1, "z": 2}[face[-1]]
    target = float(bbox_min[axis] if face.startswith("min") else bbox_max[axis])
    span = float(bbox_max[axis] - bbox_min[axis])
    tol = max(1e-4 * span, 0.05 * mesh_size_m)
    return axis, target, tol


def _voigt_strain(field: Any) -> tuple[np.ndarray, ...]:
    """Extract 6 engineering-shear Voigt strains from a 3D vector DiscreteField gradient."""
    g = field.grad
    return (
        g[0, 0],
        g[1, 1],
        g[2, 2],
        g[0, 1] + g[1, 0],
        g[1, 2] + g[2, 1],
        g[0, 2] + g[2, 0],
    )


def _von_mises_from_voigt(sigma: np.ndarray) -> np.ndarray:
    """Compute von Mises equivalent stress from Voigt [sxx, syy, szz, txy, tyz, txz] (Pa)."""
    sxx, syy, szz = sigma[0], sigma[1], sigma[2]
    txy, tyz, txz = sigma[3], sigma[4], sigma[5]
    return np.sqrt(
        np.maximum(
            0.0,
            0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
            + 3.0 * (txy**2 + tyz**2 + txz**2),
        )
    )


def _extract_outward_surface_mesh(
    mesh: Any,
    u_vertices_m: np.ndarray,
    vertex_vm_pa: np.ndarray,
    fixed_vertex_ids: np.ndarray,
    load_vertex_ids: np.ndarray,
) -> dict[str, Any]:
    """Extract boundary surface triangles oriented outward and compact to surface vertices."""
    bnd_facets = mesh.boundary_facets()
    bnd_tris = mesh.facets[:, bnd_facets].T  # (n_bnd, 3)
    parent_tets = mesh.f2t[0, bnd_facets]  # (n_bnd,)

    # Each tet has 4 distinct vertices: sum(tet) - sum(facet) gives exact opposite interior vertex
    tet_verts = mesh.t[:, parent_tets].T.astype(np.int64)
    v_opp = np.sum(tet_verts, axis=1) - np.sum(bnd_tris.astype(np.int64), axis=1)

    p_m = mesh.p.T
    e1 = p_m[bnd_tris[:, 1]] - p_m[bnd_tris[:, 0]]
    e2 = p_m[bnd_tris[:, 2]] - p_m[bnd_tris[:, 0]]
    to_opp = p_m[v_opp] - p_m[bnd_tris[:, 0]]
    inward = np.sum(np.cross(e1, e2) * to_opp, axis=1) > 0.0

    oriented_tris = bnd_tris.copy()
    if np.any(inward):
        oriented_tris[inward, 1] = bnd_tris[inward, 2]
        oriented_tris[inward, 2] = bnd_tris[inward, 1]

    surf_vert_ids, surf_inv = np.unique(oriented_tris, return_inverse=True)
    surf_tris = surf_inv.reshape(-1, 3)

    surf_vertices_mm = mesh.p[:, surf_vert_ids].T * 1000.0
    surf_disp_mm = u_vertices_m[surf_vert_ids] * 1000.0
    surf_vm_mpa = vertex_vm_pa[surf_vert_ids] * 1e-6

    fixed_mask = np.isin(surf_vert_ids, fixed_vertex_ids)
    loaded_mask = np.isin(surf_vert_ids, load_vertex_ids)

    return {
        "vertices_mm": [[round(float(c), 4) for c in row] for row in surf_vertices_mm],
        "triangles": surf_tris.tolist(),
        "displacements_mm": [[round(float(c), 6) for c in row] for row in surf_disp_mm],
        "von_mises_mpa": [round(float(v), 4) for v in surf_vm_mpa],
        "fixed_mask": [bool(flag) for flag in fixed_mask],
        "loaded_mask": [bool(flag) for flag in loaded_mask],
    }


def solve_step_tet_fea(
    step_path: str | Path,
    material_id: str = "abs_printed",
    fixed_face: FaceName = "min_x",
    load_face: FaceName = "max_x",
    force_n: tuple[float, float, float] = (0.0, 0.0, -50.0),
    mesh_size_mm: float | None = None,
    order: Literal[1, 2] = 2,
    part_name: str | None = None,
    fdm_anisotropic: bool | None = None,
    export_vtu_path: str | Path | None = None,
) -> TetFEAResult:
    """Run conformal 3D tetrahedral continuum FEA (Tet10/Tet4) on a STEP file using Gmsh + scikit-fem."""
    if order not in (1, 2):
        raise ValueError(f"Unsupported tetrahedral element order={order}; expected 1 or 2.")

    skfem, meshio = _require_tet_fea_deps()
    path = Path(step_path)
    mat = get_material(material_id)
    is_fdm = fdm_anisotropic if fdm_anisotropic is not None else ("printed" in material_id)
    resolved_name = part_name if part_name is not None else path.name

    points_mm, tets, used_h_mm = _mesh_step_with_gmsh(path, mesh_size_mm)
    points_m = points_mm * 1e-3
    mesh_size_m = used_h_mm * 1e-3

    # Build oriented MeshTet1 (positive Jacobian determinant on every tetrahedron)
    mesh = skfem.MeshTet(points_m.T, tets.T).oriented()

    if order == 2:
        scalar_elem = skfem.ElementTetP2()
        element_label = "Tet10 (P2)"
        intorder = 2  # Exact for quadratic P2 stiffness (deg 1+1=2) on affine tets, 4 positive Gauss pts
    else:
        scalar_elem = skfem.ElementTetP1()
        element_label = "Tet4 (P1)"
        intorder = 1

    element = skfem.ElementVector(scalar_elem)
    basis = skfem.CellBasis(mesh, element, intorder=intorder)

    D = constitutive_matrix_3d(mat, fdm_anisotropic=is_fdm)
    nz_rows = [
        [(j, float(D[i, j])) for j in range(6) if abs(float(D[i, j])) > 0.0]
        for i in range(6)
    ]

    @skfem.BilinearForm
    def elasticity_form(u, v, w):
        eu = _voigt_strain(u)
        ev = _voigt_strain(v)
        out = None
        for i in range(6):
            terms = nz_rows[i]
            if not terms:
                continue
            s_i = terms[0][1] * eu[terms[0][0]]
            for j, d_ij in terms[1:]:
                s_i = s_i + d_ij * eu[j]
            term = s_i * ev[i]
            out = term if out is None else out + term
        return out

    K = elasticity_form.assemble(basis)

    # Identify fixed and loaded boundary faces in strict SI meters
    bbox_min = points_m.min(axis=0)
    bbox_max = points_m.max(axis=0)
    fix_axis, fix_target, fix_tol = _face_axis_and_target(fixed_face, bbox_min, bbox_max, mesh_size_m)
    load_axis, load_target, load_tol = _face_axis_and_target(load_face, bbox_min, bbox_max, mesh_size_m)

    fixed_dofs = basis.get_dofs(lambda x: np.abs(x[fix_axis] - fix_target) <= fix_tol).flatten()
    if len(fixed_dofs) < 9:
        dof_mask = np.abs(basis.doflocs[fix_axis] - fix_target) <= fix_tol
        if int(np.sum(dof_mask)) < 9:
            dof_mask = np.abs(basis.doflocs[fix_axis] - fix_target) <= max(fix_tol, 0.5 * mesh_size_m)
        fixed_dofs = np.nonzero(dof_mask)[0].astype(np.int32)
    if len(fixed_dofs) == 0:
        raise ValueError(f"No boundary DOFs found on fixed_face='{fixed_face}'.")

    fixed_vertex_ids = np.where(np.isin(basis.nodal_dofs[0], fixed_dofs))[0].astype(np.int32)

    # Consistent surface traction f = integral(t . v dS) on boundary facets lying flat on load_face
    f_vec = np.asarray(force_n, dtype=np.float64)
    bnd_facets = mesh.boundary_facets()
    bnd_facet_verts = mesh.facets[:, bnd_facets]
    on_load_face = np.abs(mesh.p[load_axis, bnd_facet_verts] - load_target) <= load_tol
    load_facets = bnd_facets[np.all(on_load_face, axis=0)]

    if len(load_facets) > 0:
        load_fbasis = basis.boundary(facets=load_facets)
        area_load_m2 = float(np.sum(load_fbasis.dx))
    else:
        load_fbasis = None
        area_load_m2 = 0.0

    if load_fbasis is not None and area_load_m2 > 1e-18:
        traction_pa = f_vec / area_load_m2

        @skfem.LinearForm
        def traction_form(v, w):
            return traction_pa[0] * v[0] + traction_pa[1] * v[1] + traction_pa[2] * v[2]

        f = traction_form.assemble(load_fbasis)
        load_vertex_ids = np.unique(mesh.facets[:, load_facets]).astype(np.int32)
    else:
        # Graceful fallback for curved geometries without flat facets on load_face
        bnd_nodes = mesh.boundary_nodes()
        non_fixed_bnd = np.setdiff1d(bnd_nodes, fixed_vertex_ids)
        if len(non_fixed_bnd) == 0:
            raise ValueError(f"No free boundary vertices found on load_face='{load_face}'.")
        cand = non_fixed_bnd[
            np.abs(mesh.p[load_axis, non_fixed_bnd] - load_target) <= max(load_tol, 0.5 * mesh_size_m)
        ]
        if len(cand) == 0:
            dists = np.abs(mesh.p[load_axis, non_fixed_bnd] - load_target)
            cand = non_fixed_bnd[dists <= float(np.min(dists)) + max(load_tol, 0.25 * mesh_size_m)]
        f = basis.zeros()
        nodal_force = f_vec / float(len(cand))
        for axis_i in range(3):
            f[basis.nodal_dofs[axis_i, cand]] = nodal_force[axis_i]
        load_vertex_ids = cand.astype(np.int32)

    u = skfem.solve(*skfem.condense(K, f, D=fixed_dofs))

    # Displacement extraction (corner vertices + P2 mid-edge nodes for true max displacement)
    u_vertices_m = u[basis.nodal_dofs].T  # (n_vertices, 3)
    if basis.edge_dofs.size > 0:
        u_all_nodes_m = np.vstack([u_vertices_m, u[basis.edge_dofs].T])
    else:
        u_all_nodes_m = u_vertices_m

    max_disp_m = float(np.max(np.linalg.norm(u_all_nodes_m, axis=1)))

    free_load_verts = np.setdiff1d(load_vertex_ids, fixed_vertex_ids)
    if len(free_load_verts) == 0:
        free_load_verts = load_vertex_ids
    mean_loaded_disp_m = float(np.mean(np.linalg.norm(u_vertices_m[free_load_verts], axis=1)))
    f_mag = float(np.linalg.norm(f_vec))
    stiffness_n_per_mm = f_mag / max(mean_loaded_disp_m * 1e3, 1e-9)

    # Stress recovery via element quadrature points -> centroid average & volume-weighted vertex projection
    grad_u = basis.interpolate(u).grad  # (3, 3, n_elem, n_qp)
    eps_qp = np.stack(
        [
            grad_u[0, 0],
            grad_u[1, 1],
            grad_u[2, 2],
            grad_u[0, 1] + grad_u[1, 0],
            grad_u[1, 2] + grad_u[2, 1],
            grad_u[0, 2] + grad_u[2, 0],
        ],
        axis=0,
    )
    sigma_qp = np.einsum("ij,jeq->ieq", D, eps_qp)  # (6, n_elem, n_qp) in Pa
    sigma_elem = np.mean(sigma_qp, axis=2)  # (6, n_elem) exact centroid stress for linear P1 or P2
    vm_elem = _von_mises_from_voigt(sigma_elem)  # (n_elem,) in Pa

    elem_vol_m3 = np.sum(basis.dx, axis=1)
    vertex_sigma = np.zeros((6, mesh.nvertices), dtype=np.float64)
    vertex_vol = np.zeros(mesh.nvertices, dtype=np.float64)
    for local_v in range(4):
        v_idx = mesh.t[local_v]
        np.add.at(vertex_vol, v_idx, elem_vol_m3)
        for comp in range(6):
            np.add.at(vertex_sigma[comp], v_idx, sigma_elem[comp] * elem_vol_m3)
    vertex_sigma /= np.maximum(vertex_vol[None, :], 1e-30)
    vertex_vm_pa = _von_mises_from_voigt(vertex_sigma)

    max_vm_mpa = float(np.max(vm_elem)) * 1e-6
    max_z_tensile_mpa = float(max(0.0, np.max(sigma_elem[2]))) * 1e-6

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

    volume_m3 = float(np.sum(elem_vol_m3))
    volume_mm3 = volume_m3 * 1e9
    mass_g = volume_m3 * mat.density_kg_m3 * 1000.0

    surface_mesh = _extract_outward_surface_mesh(
        mesh=mesh,
        u_vertices_m=u_vertices_m,
        vertex_vm_pa=vertex_vm_pa,
        fixed_vertex_ids=fixed_vertex_ids,
        load_vertex_ids=load_vertex_ids,
    )

    if export_vtu_path is not None:
        vtu_path = Path(export_vtu_path)
        vtu_path.parent.mkdir(parents=True, exist_ok=True)
        mio_mesh = meshio.Mesh(
            points=mesh.p.T * 1000.0,
            cells=[("tetra", mesh.t.T)],
            point_data={
                "displacement_mm": u_vertices_m * 1000.0,
                "von_mises_mpa": vertex_vm_pa * 1e-6,
            },
            cell_data={
                "von_mises_mpa": [vm_elem * 1e-6],
                "sigma_zz_mpa": [sigma_elem[2] * 1e-6],
            },
        )
        meshio.write(str(vtu_path), mio_mesh)

    return TetFEAResult(
        backend="gmsh+scikit-fem",
        element_type=element_label,
        part_name=resolved_name,
        material_id=material_id,
        is_fdm_anisotropic=is_fdm,
        mesh_size_mm=used_h_mm,
        num_vertices=int(mesh.nvertices),
        num_elements=int(mesh.nelements),
        num_dofs=int(basis.N),
        mass_g=mass_g,
        volume_mm3=volume_mm3,
        applied_force_n=[float(v) for v in f_vec],
        max_displacement_um=max_disp_m * 1e6,
        stiffness_n_per_mm=stiffness_n_per_mm,
        max_von_mises_mpa=max_vm_mpa,
        max_z_tensile_mpa=max_z_tensile_mpa,
        yield_strength_mpa=yield_mpa,
        z_layer_strength_mpa=z_layer_mpa,
        safety_factor=safety_factor,
        failure_mode=failure_mode,
        surface_mesh=surface_mesh,
    )


def solve_step_bytes_tet_fea(
    step_bytes: bytes,
    material_id: str = "abs_printed",
    fixed_face: FaceName = "min_x",
    load_face: FaceName = "max_x",
    force_n: tuple[float, float, float] = (0.0, 0.0, -50.0),
    mesh_size_mm: float | None = None,
    order: Literal[1, 2] = 2,
    part_name: str = "Uploaded STEP Part",
    fdm_anisotropic: bool | None = None,
    export_vtu_path: str | Path | None = None,
) -> TetFEAResult:
    """Solve conformal 3D tetrahedral FEA directly from raw STEP file bytes."""
    with tempfile.TemporaryDirectory(prefix="printsim_tet_fea_") as tmpdir:
        tmp_step = Path(tmpdir) / "part.step"
        tmp_step.write_bytes(step_bytes)
        return solve_step_tet_fea(
            step_path=tmp_step,
            material_id=material_id,
            fixed_face=fixed_face,
            load_face=load_face,
            force_n=force_n,
            mesh_size_mm=mesh_size_mm,
            order=order,
            part_name=part_name,
            fdm_anisotropic=fdm_anisotropic,
            export_vtu_path=export_vtu_path,
        )


def solve_preset_tet_fea(
    preset: PartPreset,
    material_id: str = "abs_printed",
    force_n: tuple[float, float, float] | None = None,
    mesh_size_mm: float | None = None,
    order: Literal[1, 2] = 2,
    fdm_anisotropic: bool | None = None,
) -> TetFEAResult:
    """Run conformal 3D Tet10/Tet4 FEA on a built-in OpenCASCADE STEP CAD part preset."""
    from printsim.geometry.cad_b3d import export_part_step_bytes, get_part_default_bcs

    step_bytes = export_part_step_bytes(preset)
    fixed_face, load_face, default_force = get_part_default_bcs(preset)
    return solve_step_bytes_tet_fea(
        step_bytes=step_bytes,
        material_id=material_id,
        fixed_face=fixed_face,
        load_face=load_face,
        force_n=force_n if force_n is not None else default_force,
        mesh_size_mm=mesh_size_mm,
        order=order,
        part_name=preset.replace("_", " ").title(),
        fdm_anisotropic=fdm_anisotropic,
    )
