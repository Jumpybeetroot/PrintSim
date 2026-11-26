"""Offline CAD-to-Craig-Bampton Superelement & PBR glTF Baker (strict SI internals).

Converts 3D OpenCASCADE B-Rep solids/assemblies (``build123d`` or STEP files) into:
- Conformal 3D Tet4 meshes via isolated Gmsh subprocesses (preventing OCCT symbol collisions).
- Exact multi-material or mono-material 3D linear elastic stiffness & consistent mass matrices
  via ``scikit-fem`` ``ElementTetP1()`` and ``constitutive_matrix_3d()``.
- Exact 12-DOF + n_m Craig-Bampton superelements via RBE3 Virtual Point Transformation
  (``condense_solid_patch_to_vp(..., interface_mode="rbe3")``).
- Decimated, multi-material PBR-colored binary glTF (``visual.glb``) in strict SI meters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import heapq
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Callable

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh
import trimesh

from printsim.cad_assets.registry import ASSETS_ROOT, clear_asset_cache
from printsim.fea.solid_fem import constitutive_matrix_3d
from printsim.fea.superelement import (
    condense_solid_patch_to_vp,
    virtual_point_transform,
)
from printsim.schema import get_material

PortSelectorFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class MaterialRegion:
    """A single B-Rep solid paired with its structural material and visual PBR styling."""

    solid: Any
    material_id: str
    label: str = ""
    fdm_anisotropic: bool | None = None
    color_hex: str | None = None
    metallic: float = 0.85
    roughness: float = 0.35


@dataclass
class VisualPart:
    """A surface visual geometry part (build123d Shape or trimesh.Trimesh in mm) for visual.glb."""

    shape_or_mesh: Any
    label: str
    color_hex: str
    metallic: float = 0.85
    roughness: float = 0.35
    double_sided: bool = False


@dataclass
class PortDefinition:
    """Virtual point coordinate (mm) and boundary vertex predicate (coords_mm -> bool mask)."""

    vp_coord_mm: tuple[float, float, float]
    selector: PortSelectorFn


@dataclass
class BakedAssetSpec:
    """Metadata and configuration for baking a CAD hardware asset bundle."""

    category: str
    asset_id: str
    name: str
    source_cad: str
    source_url: str
    reference_material_id: str
    cad_trust_pct: int
    port0: PortDefinition
    port1: PortDefinition
    n_internal_modes: int = 8
    mesh_size_mm: float = 1.5
    is_multi_material: bool = False
    is_fdm_anisotropic: bool = False
    part_recipe: dict[str, Any] = field(default_factory=dict)
    visual_max_triangles: int = 9000
    fea_surface_max_triangles: int = 60000


@dataclass
class BakedAssetResult:
    """Verification summary returned by :func:`bake_solid_superelement`."""

    category: str
    asset_id: str
    asset_dir: Path
    manifest: dict[str, Any]
    K_cb: np.ndarray
    M_cb: np.ndarray
    K_bb: np.ndarray
    port_coords_m: np.ndarray
    com_m: np.ndarray
    inertia_kg_m2: np.ndarray


def _hex_to_rgba(hex_color: str, alpha: int = 255) -> list[int]:
    h = hex_color.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    return [int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), alpha]


def select_z_annulus(
    z_mm: float,
    r_min_mm: float = 0.0,
    r_max_mm: float = 100.0,
    center_xy_mm: tuple[float, float] = (0.0, 0.0),
    tol_mm: float = 0.45,
) -> PortSelectorFn:
    """Select boundary nodes on a horizontal Z plane within an XY radial annulus."""

    def _sel(coords_mm: np.ndarray) -> np.ndarray:
        dx = coords_mm[:, 0] - center_xy_mm[0]
        dy = coords_mm[:, 1] - center_xy_mm[1]
        r = np.hypot(dx, dy)
        return (
            (np.abs(coords_mm[:, 2] - z_mm) <= tol_mm)
            & (r >= r_min_mm - 1e-6)
            & (r <= r_max_mm + 1e-6)
        )

    return _sel


def select_z_cap(
    z_max_mm: float,
    z_min_mm: float = -1e6,
    center_xy_mm: tuple[float, float] = (0.0, 0.0),
    r_max_mm: float = 100.0,
) -> PortSelectorFn:
    """Select boundary nodes inside a Z slab and XY radius (useful for conical nozzle tips)."""

    def _sel(coords_mm: np.ndarray) -> np.ndarray:
        r = np.hypot(coords_mm[:, 0] - center_xy_mm[0], coords_mm[:, 1] - center_xy_mm[1])
        return (coords_mm[:, 2] >= z_min_mm) & (coords_mm[:, 2] <= z_max_mm) & (r <= r_max_mm)

    return _sel


def select_plane_box(
    axis: int,
    coord_mm: float,
    tol_mm: float = 0.45,
    bounds_mm: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None,
) -> PortSelectorFn:
    """Select boundary nodes on ``coords[:, axis] ≈ coord_mm`` optionally cropped to a 3D box."""

    def _sel(coords_mm: np.ndarray) -> np.ndarray:
        mask = np.abs(coords_mm[:, axis] - coord_mm) <= tol_mm
        if bounds_mm is not None:
            for ax, (lo, hi) in enumerate(bounds_mm):
                mask &= (coords_mm[:, ax] >= lo) & (coords_mm[:, ax] <= hi)
        return mask

    return _sel


# ---------------------------------------------------------------------------
# Pure-NumPy Garland-Heckbert QEM Surface Mesh Decimator
# ---------------------------------------------------------------------------


def decimate_trimesh_qem(
    mesh: trimesh.Trimesh,
    target_triangles: int,
    preserve_boundaries: bool = True,
) -> trimesh.Trimesh:
    """Decimate a triangle mesh using Garland-Heckbert Quadric Error Metrics (QEM).

    Preserves open boundary edges and sharp creases while preventing face normal inversions.
    Works in pure NumPy/SciPy without requiring optional ``fast_simplification``.
    """
    m = mesh.copy()
    m.merge_vertices()
    m.update_faces(m.nondegenerate_faces())
    m.update_faces(m.unique_faces())
    m.remove_unreferenced_vertices()

    target = max(4, int(target_triangles))
    if len(m.faces) <= target:
        return m

    verts_arr = np.array(m.vertices, dtype=np.float64, copy=True)
    faces_arr = np.array(m.faces, dtype=np.int64, copy=True)
    n_verts = len(verts_arr)
    n_faces = len(faces_arr)

    # Build initial per-vertex quadrics from area-weighted face planes
    v0 = verts_arr[faces_arr[:, 0]]
    v1 = verts_arr[faces_arr[:, 1]]
    v2 = verts_arr[faces_arr[:, 2]]
    cross_vec = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross_vec, axis=1)
    safe_area = np.maximum(areas, 1e-18)
    normals_arr = cross_vec / (2.0 * safe_area[:, None])
    d_plane = -np.einsum("ij,ij->i", normals_arr, v0)
    planes = np.column_stack([normals_arr, d_plane])

    Q_faces = np.einsum("fi,fj->fij", planes, planes) * safe_area[:, None, None]
    Q_arr = np.zeros((n_verts, 4, 4), dtype=np.float64)
    for k in range(3):
        np.add.at(Q_arr, faces_arr[:, k], Q_faces)

    # Identify boundary edges and add strong perpendicular plane constraints
    dir_edges = faces_arr[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    face_ids_for_edge = np.repeat(np.arange(n_faces, dtype=np.int64), 3)
    sorted_e = np.sort(dir_edges, axis=1)
    u_edges, first_idx, counts = np.unique(
        sorted_e, axis=0, return_index=True, return_counts=True
    )

    if preserve_boundaries:
        bnd_mask = counts == 1
        if np.any(bnd_mask):
            be = dir_edges[first_idx[bnd_mask]]
            bf = face_ids_for_edge[first_idx[bnd_mask]]
            e_vec = verts_arr[be[:, 1]] - verts_arr[be[:, 0]]
            e_len = np.linalg.norm(e_vec, axis=1)
            ok = e_len > 1e-12
            if np.any(ok):
                e_hat = e_vec[ok] / e_len[ok, None]
                b_norm = np.cross(e_hat, normals_arr[bf[ok]])
                bn_len = np.linalg.norm(b_norm, axis=1)
                ok2 = bn_len > 1e-12
                if np.any(ok2):
                    bn = b_norm[ok2] / bn_len[ok2, None]
                    bd = -np.einsum("ij,ij->i", bn, verts_arr[be[ok][ok2, 0]])
                    bp = np.column_stack([bn, bd])
                    w_bnd = (100.0 * e_len[ok][ok2] ** 2)[:, None, None]
                    Qb = np.einsum("ei,ej->eij", bp, bp) * w_bnd
                    np.add.at(Q_arr, be[ok][ok2, 0], Qb)
                    np.add.at(Q_arr, be[ok][ok2, 1], Qb)

    # Pack symmetric 4x4 Q into 10 floats: [q00, q01, q02, q03, q11, q12, q13, q22, q23, q33]
    Q10 = np.column_stack(
        [
            Q_arr[:, 0, 0],
            Q_arr[:, 0, 1],
            Q_arr[:, 0, 2],
            Q_arr[:, 0, 3],
            Q_arr[:, 1, 1],
            Q_arr[:, 1, 2],
            Q_arr[:, 1, 3],
            Q_arr[:, 2, 2],
            Q_arr[:, 2, 3],
            Q_arr[:, 3, 3],
        ]
    )

    # Vectorized initial heap cost evaluation across all unique undirected edges
    valid_e = u_edges[:, 0] < u_edges[:, 1]
    u_edges = u_edges[valid_e]
    eu = u_edges[:, 0]
    ev = u_edges[:, 1]
    Qs = Q10[eu] + Q10[ev]
    pu_v = verts_arr[eu]
    pv_v = verts_arr[ev]
    pm_v = 0.5 * (pu_v + pv_v)

    def _batch_eval(q: np.ndarray, pts: np.ndarray) -> np.ndarray:
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        return np.maximum(
            0.0,
            q[:, 0] * x * x
            + q[:, 4] * y * y
            + q[:, 7] * z * z
            + 2.0
            * (
                q[:, 1] * x * y
                + q[:, 2] * x * z
                + q[:, 5] * y * z
                + q[:, 3] * x
                + q[:, 6] * y
                + q[:, 8] * z
            )
            + q[:, 9],
        )

    c_u = _batch_eval(Qs, pu_v)
    c_v = _batch_eval(Qs, pv_v)
    c_m = _batch_eval(Qs, pm_v)
    init_costs = np.minimum(np.minimum(c_u, c_v), c_m)

    a00, a01, a02, b0 = Qs[:, 0], Qs[:, 1], Qs[:, 2], -Qs[:, 3]
    a11, a12, b1 = Qs[:, 4], Qs[:, 5], -Qs[:, 6]
    a22, b2 = Qs[:, 7], -Qs[:, 8]
    c00 = a11 * a22 - a12 * a12
    c01 = a02 * a12 - a01 * a22
    c02 = a01 * a12 - a02 * a11
    det = a00 * c00 + a01 * c01 + a02 * c02
    good = np.abs(det) > 1e-18
    if np.any(good):
        inv_d = np.where(good, 1.0 / np.where(good, det, 1.0), 0.0)
        c11 = a00 * a22 - a02 * a02
        c12 = a01 * a02 - a00 * a12
        c22 = a00 * a11 - a01 * a01
        sx = (c00 * b0 + c01 * b1 + c02 * b2) * inv_d
        sy = (c01 * b0 + c11 * b1 + c12 * b2) * inv_d
        sz = (c02 * b0 + c12 * b1 + c22 * b2) * inv_d
        sol = np.column_stack([sx, sy, sz])
        elen = np.maximum(np.linalg.norm(pv_v - pu_v, axis=1), 1e-12)
        in_span = good & (np.linalg.norm(sol - pm_v, axis=1) <= 2.5 * elen)
        c_s = _batch_eval(Qs, np.where(in_span[:, None], sol, pm_v))
        init_costs = np.where(in_span & (c_s < init_costs), c_s, init_costs)

    # Pure-Python lists for O(1) scalar inner-loop collapse operations
    vx = verts_arr[:, 0].tolist()
    vy = verts_arr[:, 1].tolist()
    vz = verts_arr[:, 2].tolist()
    q_list = Q10.tolist()
    faces = faces_arr.tolist()
    fnx = normals_arr[:, 0].tolist()
    fny = normals_arr[:, 1].tolist()
    fnz = normals_arr[:, 2].tolist()

    vert_faces: list[set[int]] = [set() for _ in range(n_verts)]
    for fi, (a, b, c) in enumerate(faces):
        vert_faces[a].add(fi)
        vert_faces[b].add(fi)
        vert_faces[c].add(fi)

    face_alive = [True] * n_faces
    vert_alive = [True] * n_verts
    vert_ver = [0] * n_verts
    live_faces = n_faces

    def _edge_cost_scalar(u: int, v: int) -> tuple[float, float, float, float]:
        qu = q_list[u]
        qv = q_list[v]
        q00 = qu[0] + qv[0]
        q01 = qu[1] + qv[1]
        q02 = qu[2] + qv[2]
        q03 = qu[3] + qv[3]
        q11 = qu[4] + qv[4]
        q12 = qu[5] + qv[5]
        q13 = qu[6] + qv[6]
        q22 = qu[7] + qv[7]
        q23 = qu[8] + qv[8]
        q33 = qu[9] + qv[9]

        ux, uy, uz = vx[u], vy[u], vz[u]
        px, py, pz = vx[v], vy[v], vz[v]
        mx, my, mz = 0.5 * (ux + px), 0.5 * (uy + py), 0.5 * (uz + pz)

        best_c = (
            q00 * mx * mx
            + q11 * my * my
            + q22 * mz * mz
            + 2.0 * (q01 * mx * my + q02 * mx * mz + q12 * my * mz + q03 * mx + q13 * my + q23 * mz)
            + q33
        )
        if best_c < 0.0:
            best_c = 0.0
        bx, by, bz = mx, my, mz

        cu = (
            q00 * ux * ux
            + q11 * uy * uy
            + q22 * uz * uz
            + 2.0 * (q01 * ux * uy + q02 * ux * uz + q12 * uy * uz + q03 * ux + q13 * uy + q23 * uz)
            + q33
        )
        if cu < 0.0:
            cu = 0.0
        if cu < best_c:
            best_c = cu
            bx, by, bz = ux, uy, uz

        cv = (
            q00 * px * px
            + q11 * py * py
            + q22 * pz * pz
            + 2.0 * (q01 * px * py + q02 * px * pz + q12 * py * pz + q03 * px + q13 * py + q23 * pz)
            + q33
        )
        if cv < 0.0:
            cv = 0.0
        if cv < best_c:
            best_c = cv
            bx, by, bz = px, py, pz

        sc00 = q11 * q22 - q12 * q12
        sc01 = q02 * q12 - q01 * q22
        sc02 = q01 * q12 - q02 * q11
        s_det = q00 * sc00 + q01 * sc01 + q02 * sc02
        if s_det > 1e-18 or s_det < -1e-18:
            s_inv = 1.0 / s_det
            sc11 = q00 * q22 - q02 * q02
            sc12 = q01 * q02 - q00 * q12
            sc22 = q00 * q11 - q01 * q01
            sx_s = -(sc00 * q03 + sc01 * q13 + sc02 * q23) * s_inv
            sy_s = -(sc01 * q03 + sc11 * q13 + sc12 * q23) * s_inv
            sz_s = -(sc02 * q03 + sc12 * q13 + sc22 * q23) * s_inv
            dx, dy, dz = px - ux, py - uy, pz - uz
            elen2 = dx * dx + dy * dy + dz * dz
            if elen2 < 1e-24:
                elen2 = 1e-24
            dsx, dsy, dsz = sx_s - mx, sy_s - my, sz_s - mz
            if (dsx * dsx + dsy * dsy + dsz * dsz) <= 6.25 * elen2:
                cs = (
                    q00 * sx_s * sx_s
                    + q11 * sy_s * sy_s
                    + q22 * sz_s * sz_s
                    + 2.0
                    * (
                        q01 * sx_s * sy_s
                        + q02 * sx_s * sz_s
                        + q12 * sy_s * sz_s
                        + q03 * sx_s
                        + q13 * sy_s
                        + q23 * sz_s
                    )
                    + q33
                )
                if cs < 0.0:
                    cs = 0.0
                if cs < best_c:
                    best_c = cs
                    bx, by, bz = sx_s, sy_s, sz_s
        return best_c, bx, by, bz

    eu_l = eu.tolist()
    ev_l = ev.tolist()
    ic_l = init_costs.tolist()
    heap: list[tuple[float, int, int, int, int]] = [
        (ic_l[i], eu_l[i], ev_l[i], 0, 0) for i in range(len(eu_l))
    ]
    heapq.heapify(heap)

    while heap and live_faces > target:
        _, u, v, vu, vv = heapq.heappop(heap)
        if not (vert_alive[u] and vert_alive[v]):
            continue
        if vert_ver[u] != vu or vert_ver[v] != vv:
            continue

        vfu = vert_faces[u]
        vfv = vert_faces[v]
        shared_faces = vfu & vfv
        if not shared_faces:
            continue

        nbs_u: set[int] = set()
        for fi in vfu:
            if not face_alive[fi]:
                continue
            a, b, c = faces[fi]
            if a != u:
                nbs_u.add(a)
            if b != u:
                nbs_u.add(b)
            if c != u:
                nbs_u.add(c)
        nbs_v: set[int] = set()
        for fi in vfv:
            if not face_alive[fi]:
                continue
            a, b, c = faces[fi]
            if a != v:
                nbs_v.add(a)
            if b != v:
                nbs_v.add(b)
            if c != v:
                nbs_v.add(c)
        if len(nbs_u & nbs_v) != len(shared_faces):
            continue

        _, nx, ny, nz = _edge_cost_scalar(u, v)

        affected = (vfu | vfv) - shared_faces
        flip = False
        new_norms: list[tuple[int, float, float, float]] = []
        for fi in affected:
            if not face_alive[fi]:
                continue
            a, b, c = faces[fi]
            if a == u or a == v:
                x0, y0, z0 = nx, ny, nz
            else:
                x0, y0, z0 = vx[a], vy[a], vz[a]
            if b == u or b == v:
                x1, y1, z1 = nx, ny, nz
            else:
                x1, y1, z1 = vx[b], vy[b], vz[b]
            if c == u or c == v:
                x2, y2, z2 = nx, ny, nz
            else:
                x2, y2, z2 = vx[c], vy[c], vz[c]
            ex1, ey1, ez1 = x1 - x0, y1 - y0, z1 - z0
            ex2, ey2, ez2 = x2 - x0, y2 - y0, z2 - z0
            cx = ey1 * ez2 - ez1 * ey2
            cy = ez1 * ex2 - ex1 * ez2
            cz = ex1 * ey2 - ey1 * ex2
            n2 = cx * cx + cy * cy + cz * cz
            if n2 < 1e-36:
                flip = True
                break
            inv_n = 1.0 / math.sqrt(n2)
            cx *= inv_n
            cy *= inv_n
            cz *= inv_n
            if cx * fnx[fi] + cy * fny[fi] + cz * fnz[fi] < 0.15:
                flip = True
                break
            new_norms.append((fi, cx, cy, cz))
        if flip:
            continue

        vx[u], vy[u], vz[u] = nx, ny, nz
        qu = q_list[u]
        qv = q_list[v]
        for k in range(10):
            qu[k] += qv[k]
        vert_alive[v] = False
        vert_ver[u] += 1

        for fi in shared_faces:
            if face_alive[fi]:
                face_alive[fi] = False
                live_faces -= 1
            a, b, c = faces[fi]
            if a != u and a != v:
                vert_faces[a].discard(fi)
            if b != u and b != v:
                vert_faces[b].discard(fi)
            if c != u and c != v:
                vert_faces[c].discard(fi)

        for fi, cx, cy, cz in new_norms:
            tri = faces[fi]
            if tri[0] == v:
                tri[0] = u
            if tri[1] == v:
                tri[1] = u
            if tri[2] == v:
                tri[2] = u
            vfu.add(fi)
            fnx[fi], fny[fi], fnz[fi] = cx, cy, cz
        vfu.difference_update(shared_faces)
        vfv.clear()

        nbs_new: set[int] = set()
        for fi in vfu:
            if not face_alive[fi]:
                continue
            a, b, c = faces[fi]
            if a != u:
                nbs_new.add(a)
            if b != u:
                nbs_new.add(b)
            if c != u:
                nbs_new.add(c)
        vu_now = vert_ver[u]
        for nb in nbs_new:
            a, b = (u, nb) if u < nb else (nb, u)
            c_new, _, _, _ = _edge_cost_scalar(a, b)
            va, vb = (vu_now, vert_ver[nb]) if u < nb else (vert_ver[nb], vu_now)
            heapq.heappush(heap, (c_new, a, b, va, vb))

    kept_faces = np.asarray([faces[i] for i, al in enumerate(face_alive) if al], dtype=np.int64)
    out_verts = np.column_stack([vx, vy, vz])
    out = trimesh.Trimesh(vertices=out_verts, faces=kept_faces, process=False)
    out.remove_unreferenced_vertices()
    out.update_faces(out.nondegenerate_faces())
    out.remove_unreferenced_vertices()
    return out


# ---------------------------------------------------------------------------
# Isolated Gmsh Subprocess Workers (Single, Multi-Material, STL-Fallback)
# ---------------------------------------------------------------------------


def _export_shape_for_gmsh(shape: Any, path_stem: Path, prefer_brep: bool = False) -> Path:
    """Export a build123d Shape to STEP or native OCCT BREP."""
    import build123d as b3d

    if not prefer_brep:
        step_path = path_stem.with_suffix(".step")
        try:
            wrapped = shape if isinstance(shape, b3d.Compound) else b3d.Compound(children=[shape])
            b3d.export_step(wrapped, str(step_path))
            if step_path.exists() and step_path.stat().st_size > 100:
                return step_path
        except Exception:
            pass
    brep_path = path_stem.with_suffix(".brep")
    b3d.export_brep(shape, str(brep_path))
    return brep_path


def _fill_surface_holes_numpy(tm: trimesh.Trimesh) -> trimesh.Trimesh:
    """Close small open boundary loops in a surface mesh using pure NumPy (no networkx needed)."""
    if tm.is_watertight or len(tm.faces) == 0:
        return tm
    faces = np.asarray(tm.faces, dtype=np.int64)
    verts = np.asarray(tm.vertices, dtype=np.float64)
    dir_edges = faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    sorted_e = np.sort(dir_edges, axis=1)
    _, first_idx, counts = np.unique(sorted_e, axis=0, return_index=True, return_counts=True)
    bnd_edges = dir_edges[first_idx[counts == 1]]
    if len(bnd_edges) == 0:
        return tm

    # Trace closed boundary cycles; reverse each boundary edge (v->u) so cap normals match outward winding
    nxt_map: dict[int, list[int]] = {}
    for u, v in bnd_edges.tolist():
        nxt_map.setdefault(v, []).append(u)

    new_verts = verts.tolist()
    new_faces: list[list[int]] = []
    used_edges: set[tuple[int, int]] = set()

    for v_start, nbrs in list(nxt_map.items()):
        for u_next in nbrs:
            if (v_start, u_next) in used_edges:
                continue
            loop = [v_start]
            cur = u_next
            used_edges.add((v_start, u_next))
            while cur != v_start and len(loop) < 2000:
                loop.append(cur)
                cands = [n for n in nxt_map.get(cur, []) if (cur, n) not in used_edges]
                if not cands:
                    break
                nxt = cands[0]
                used_edges.add((cur, nxt))
                cur = nxt
            if cur == v_start and len(loop) >= 3:
                if len(loop) == 3:
                    new_faces.append([loop[0], loop[1], loop[2]])
                else:
                    c_idx = len(new_verts)
                    centroid = np.mean(verts[loop], axis=0).tolist()
                    new_verts.append(centroid)
                    n_l = len(loop)
                    for k in range(n_l):
                        new_faces.append([c_idx, loop[k], loop[(k + 1) % n_l]])

    if not new_faces:
        return tm
    out = trimesh.Trimesh(
        vertices=np.asarray(new_verts, dtype=np.float64),
        faces=np.vstack([faces, np.asarray(new_faces, dtype=np.int64)]),
        process=False,
    )
    out.update_faces(out.nondegenerate_faces())
    out.remove_unreferenced_vertices()
    return out


def _mesh_solid_voxel_tet4(
    shape: Any,
    mesh_size_mm: float,
    snap_coords_mm: tuple[tuple[float, float, float], ...] = (),
) -> tuple[np.ndarray, np.ndarray]:
    """Robust pure-NumPy conforming 5-tet-per-voxel fallback for self-intersecting SolidWorks B-Reps.

    Classifies voxel centers via exact Z-column triangle parity rasterization (zero rtree/networkx
    dependency), retains the primary 6-connected solid component, and decomposes each occupied cube
    with alternating-parity 5-tetrahedra splits (100% conforming shared diagonals, positive volume).
    Grid planes are snapped to ``snap_coords_mm`` (e.g., Port 0 / Port 1 mounting faces).
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    tm = _shape_to_trimesh_mm(shape, tolerance_mm=0.06, angular_tol=0.18)
    bmin, bmax = tm.bounds
    h = float(np.clip(mesh_size_mm * 0.62, 0.75, 1.80))

    # Align X, Y, Z grids cleanly: snap Y to Port 0 plane (e.g. MGN carriage face Y=0) and Z to Port 1 plane (e.g. Z=27.5)
    axes_grids: list[np.ndarray] = []
    for ax in range(3):
        anchor = 0.0
        if snap_coords_mm:
            if ax == 2 and len(snap_coords_mm) >= 2 and abs(float(snap_coords_mm[1][ax])) > 1e-6:
                anchor = float(snap_coords_mm[1][ax])
            else:
                anchor = float(snap_coords_mm[0][ax])
        shift = anchor - math.floor(anchor / h) * h
        lo = math.floor((float(bmin[ax]) - shift) / h) * h + shift - h
        hi = math.ceil((float(bmax[ax]) - shift) / h) * h + shift + 1.5 * h
        axes_grids.append(np.arange(lo, hi, h, dtype=np.float64))

    xs, ys, zs = axes_grids
    nx, ny, nz = len(xs) - 1, len(ys) - 1, len(zs) - 1
    xc = 0.5 * (xs[:-1] + xs[1:])
    yc = 0.5 * (ys[:-1] + ys[1:])
    zc = 0.5 * (zs[:-1] + zs[1:])

    tris = np.asarray(tm.triangles, dtype=np.float64)
    x0, y0, z0 = tris[:, 0, 0], tris[:, 0, 1], tris[:, 0, 2]
    x1, y1, z1 = tris[:, 1, 0], tris[:, 1, 1], tris[:, 1, 2]
    x2, y2, z2 = tris[:, 2, 0], tris[:, 2, 1], tris[:, 2, 2]
    det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    valid = np.abs(det) > 1e-10
    tris, det = tris[valid], det[valid]
    x0, y0, z0 = tris[:, 0, 0], tris[:, 0, 1], tris[:, 0, 2]
    x1, y1, z1 = tris[:, 1, 0], tris[:, 1, 1], tris[:, 1, 2]
    x2, y2, z2 = tris[:, 2, 0], tris[:, 2, 1], tris[:, 2, 2]

    # Tiny irrational jitter guarantees rays never graze exact mesh edges
    xcp = xc + 1.37e-5
    ycp = yc + 2.71e-5
    ix_lo = np.clip(np.searchsorted(xcp, np.minimum(np.minimum(x0, x1), x2)), 0, nx)
    ix_hi = np.clip(np.searchsorted(xcp, np.maximum(np.maximum(x0, x1), x2)), 0, nx)
    iy_lo = np.clip(np.searchsorted(ycp, np.minimum(np.minimum(y0, y1), y2)), 0, ny)
    iy_hi = np.clip(np.searchsorted(ycp, np.maximum(np.maximum(y0, y1), y2)), 0, ny)

    diff_z = np.zeros((nx, ny, nz + 1), dtype=np.int16)
    active = np.where((ix_hi > ix_lo) & (iy_hi > iy_lo))[0]
    for fi in active:
        a0, b0, c0 = float(x0[fi]), float(y0[fi]), float(z0[fi])
        e1x, e1y, e1z = float(x1[fi]) - a0, float(y1[fi]) - b0, float(z1[fi]) - c0
        e2x, e2y, e2z = float(x2[fi]) - a0, float(y2[fi]) - b0, float(z2[fi]) - c0
        inv_d = 1.0 / float(det[fi])
        for ix in range(int(ix_lo[fi]), int(ix_hi[fi])):
            px = float(xcp[ix]) - a0
            for iy in range(int(iy_lo[fi]), int(iy_hi[fi])):
                py = float(ycp[iy]) - b0
                u = (px * e2y - py * e2x) * inv_d
                if u < 0.0 or u > 1.0:
                    continue
                v = (e1x * py - e1y * px) * inv_d
                if v < 0.0 or (u + v) > 1.0:
                    continue
                zh = c0 + u * e1z + v * e2z
                iz = int(np.searchsorted(zc, zh))
                if iz < nz:
                    diff_z[ix, iy, iz] ^= 1

    occ = (np.cumsum(diff_z[:, :, :nz], axis=2) & 1).astype(bool)
    occ_idx = np.argwhere(occ)
    if len(occ_idx) == 0:
        raise RuntimeError("Voxel Tet4 fallback classified 0 interior voxels")

    # Keep only the primary 6-connected solid component
    lin = (occ_idx[:, 0] * ny + occ_idx[:, 1]) * nz + occ_idx[:, 2]
    idx_map = np.full(nx * ny * nz, -1, dtype=np.int32)
    idx_map[lin] = np.arange(len(occ_idx), dtype=np.int32)
    r_blocks: list[np.ndarray] = []
    c_blocks: list[np.ndarray] = []
    for axis, step in ((0, ny * nz), (1, nz), (2, 1)):
        mask = occ_idx[:, axis] + 1 < (nx, ny, nz)[axis]
        a = np.where(mask)[0]
        b = idx_map[lin[a] + step]
        ok = b >= 0
        if np.any(ok):
            r_blocks.append(a[ok])
            c_blocks.append(b[ok])
    if r_blocks:
        rr = np.concatenate(r_blocks)
        cc = np.concatenate(c_blocks)
        adj = csr_matrix(
            (np.ones(len(rr) * 2, dtype=np.int8), (np.concatenate([rr, cc]), np.concatenate([cc, rr]))),
            shape=(len(occ_idx), len(occ_idx)),
        )
        _, labels = connected_components(adj, directed=False)
        main_lbl = int(np.argmax(np.bincount(labels)))
        occ_idx = occ_idx[labels == main_lbl]

    # Map occupied cube corners (i+dx, j+dy, k+dz) to compact vertex indices
    nx1, ny1, nz1 = nx + 1, ny + 1, nz + 1
    ci, cj, ck = occ_idx[:, 0], occ_idx[:, 1], occ_idx[:, 2]
    # 8 corners ordered by bit 0=x, bit 1=y, bit 2=z
    corners = np.stack(
        [
            ((ci + dx) * ny1 + (cj + dy)) * nz1 + (ck + dz)
            for dz in (0, 1)
            for dy in (0, 1)
            for dx in (0, 1)
        ],
        axis=1,
    )  # (N_vox, 8)
    u_nodes, inv = np.unique(corners.ravel(), return_inverse=True)
    c_local = inv.reshape(-1, 8).astype(np.int32)

    ui = u_nodes // (ny1 * nz1)
    rem = u_nodes % (ny1 * nz1)
    uj = rem // nz1
    uk = rem % nz1
    points_mm = np.column_stack([xs[ui], ys[uj], zs[uk]]).astype(np.float64)

    # Alternating-parity 5-tet decomposition (100% conforming across all 6 faces!)
    even_pat = np.array(
        [[0, 3, 5, 6], [0, 1, 3, 5], [0, 2, 3, 6], [0, 4, 5, 6], [3, 5, 6, 7]], dtype=np.int32
    )
    odd_pat = np.array(
        [[1, 2, 4, 7], [0, 1, 2, 4], [1, 2, 3, 7], [1, 4, 5, 7], [2, 4, 6, 7]], dtype=np.int32
    )
    parity = ((ci + cj + ck) & 1).astype(bool)
    tet_blocks: list[np.ndarray] = []
    if np.any(~parity):
        tet_blocks.append(c_local[~parity][:, even_pat].reshape(-1, 4))
    if np.any(parity):
        tet_blocks.append(c_local[parity][:, odd_pat].reshape(-1, 4))
    tets = np.ascontiguousarray(np.vstack(tet_blocks), dtype=np.int32)
    return np.ascontiguousarray(points_mm, dtype=np.float64), tets


def _mesh_regions_tet4(
    regions: list[MaterialRegion],
    mesh_size_mm: float,
    fea_surface_max_triangles: int = 60000,
    snap_coords_mm: tuple[tuple[float, float, float], ...] = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a conformal 3D Tet4 mesh in mm with per-element ``region_id``.

    Uses ``printsim.fea._gmsh_worker`` for single-material STEP parts and falls back
    to watertight surface-STL-to-Tet4 (or ``occ.fragment`` for multi-material assemblies)
    inside an isolated Gmsh child process.
    """
    with tempfile.TemporaryDirectory(prefix="printsim_bake_gmsh_") as td:
        tmp = Path(td)
        out_npz = tmp / "tet_mesh.npz"

        if len(regions) == 1:
            geom_path = _export_shape_for_gmsh(regions[0].solid, tmp / "region_0")
            if geom_path.suffix == ".step":
                proc = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "printsim.fea._gmsh_worker",
                        str(geom_path),
                        str(out_npz),
                        str(float(mesh_size_mm)),
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if proc.returncode == 0 and out_npz.exists():
                    with np.load(out_npz) as data:
                        pts_mm = np.ascontiguousarray(data["points_mm"], dtype=np.float64)
                        tets = np.ascontiguousarray(data["tets"], dtype=np.int32)
                    if len(tets) > 0:
                        return pts_mm, tets, np.zeros(len(tets), dtype=np.int32)

            # Fallback for complex SolidWorks B-Reps (e.g. MBTCFD): native BREP -> 2D surface mesh -> merge -> decimate -> Tet4
            brep_path = _export_shape_for_gmsh(regions[0].solid, tmp / "region_0_fallback", prefer_brep=True)
            surf_npz = tmp / "surf.npz"
            proc2d = subprocess.run(
                [
                    sys.executable,
                    "-P",
                    str(Path(__file__).resolve()),
                    "--gmsh-worker-2d",
                    str(brep_path),
                    str(surf_npz),
                    str(float(mesh_size_mm)),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc2d.returncode != 0 or not surf_npz.exists():
                err = (proc2d.stderr or proc2d.stdout).strip()
                raise RuntimeError(f"Gmsh 2D surface meshing failed: {err}")

            with np.load(surf_npz) as sdata:
                tm = trimesh.Trimesh(
                    vertices=np.asarray(sdata["points_mm"], dtype=np.float64),
                    faces=np.asarray(sdata["tris"], dtype=np.int64),
                    process=False,
                )
            tm.merge_vertices(digits_vertex=3)
            tm.update_faces(tm.nondegenerate_faces())
            tm.update_faces(tm.unique_faces())
            tm.remove_unreferenced_vertices()
            if not tm.is_watertight:
                tm = _fill_surface_holes_numpy(tm)
            try:
                tm.fix_normals()
            except ModuleNotFoundError:
                trimesh.repair.fix_inversion(tm)
            if len(tm.faces) > fea_surface_max_triangles:
                tm = decimate_trimesh_qem(tm, target_triangles=fea_surface_max_triangles)

            stl_path = tmp / "watertight.stl"
            tm.export(str(stl_path))
            proc3d = subprocess.run(
                [
                    sys.executable,
                    "-P",
                    str(Path(__file__).resolve()),
                    "--gmsh-worker-stl",
                    str(stl_path),
                    str(out_npz),
                    str(float(mesh_size_mm)),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc3d.returncode == 0 and out_npz.exists():
                with np.load(out_npz) as data:
                    pts_mm = np.ascontiguousarray(data["points_mm"], dtype=np.float64)
                    tets = np.ascontiguousarray(data["tets"], dtype=np.int32)
                if len(tets) > 0:
                    return pts_mm, tets, np.zeros(len(tets), dtype=np.int32)

            # Tier-3 fallback for self-intersecting SolidWorks lofts (e.g. MBT_Chube_V1 CPAP duct):
            # pure-NumPy conforming 5-tet-per-voxel mesh snapped to exact Port 0 / Port 1 planes.
            pts_mm, tets = _mesh_solid_voxel_tet4(
                regions[0].solid,
                mesh_size_mm=mesh_size_mm,
                snap_coords_mm=snap_coords_mm,
            )
            return pts_mm, tets, np.zeros(len(tets), dtype=np.int32)

        # Multi-region conformal meshing via occ.fragment in isolated child process
        region_files: list[str] = []
        for idx, reg in enumerate(regions):
            gp = _export_shape_for_gmsh(reg.solid, tmp / f"region_{idx}", prefer_brep=True)
            region_files.append(str(gp))

        spec_json = tmp / "multi_regions.json"
        spec_json.write_text(
            json.dumps({"files": region_files, "mesh_size_mm": float(mesh_size_mm)}),
            encoding="utf-8",
        )
        proc_m = subprocess.run(
            [
                sys.executable,
                "-P",
                str(Path(__file__).resolve()),
                "--gmsh-worker-multi",
                str(spec_json),
                str(out_npz),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc_m.returncode != 0 or not out_npz.exists():
            err = (proc_m.stderr or proc_m.stdout).strip()
            raise RuntimeError(f"Gmsh multi-region meshing failed: {err}")

        with np.load(out_npz) as data:
            pts_mm = np.ascontiguousarray(data["points_mm"], dtype=np.float64)
            tets = np.ascontiguousarray(data["tets"], dtype=np.int32)
            reg_ids = np.ascontiguousarray(data["region_ids"], dtype=np.int32)
        return pts_mm, tets, reg_ids


# ---------------------------------------------------------------------------
# Visual GLB Builder (Strict SI Meters, Multi-Material PBR)
# ---------------------------------------------------------------------------


def _shape_to_trimesh_mm(
    shape: Any,
    tolerance_mm: float = 0.04,
    angular_tol: float = 0.14,
) -> trimesh.Trimesh:
    """Tessellate a build123d Shape (Solid/Compound/Shell) in mm to a clean trimesh.Trimesh."""
    if isinstance(shape, trimesh.Trimesh):
        return shape.copy()

    import build123d as b3d

    with tempfile.TemporaryDirectory(prefix="printsim_vis_stl_") as td:
        stl_path = Path(td) / "part.stl"
        b3d.export_stl(
            shape,
            str(stl_path),
            tolerance=tolerance_mm,
            angular_tolerance=angular_tol,
            ascii_format=False,
        )
        loaded = trimesh.load(str(stl_path), force="mesh")
        if isinstance(loaded, trimesh.Scene):
            loaded = loaded.dump(concatenate=True)
        return loaded


def _apply_creased_vertex_normals(
    tm: trimesh.Trimesh,
    crease_angle_deg: float = 34.0,
) -> trimesh.Trimesh:
    """Split sharp edges (> crease_angle_deg) and compute area-weighted smooth vertex normals."""
    m = tm.copy()
    m.merge_vertices()
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    if len(m.faces) == 0 or len(m.vertices) == 0:
        return m

    verts = np.asarray(m.vertices, dtype=np.float64)
    faces = np.asarray(m.faces, dtype=np.int64)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross_vec = np.cross(v1 - v0, v2 - v0)
    dbl_area = np.linalg.norm(cross_vec, axis=1)
    valid = dbl_area > 1e-18
    if not np.all(valid):
        m.update_faces(valid)
        m.remove_unreferenced_vertices()
        if len(m.faces) == 0 or len(m.vertices) == 0:
            return m
        verts = np.asarray(m.vertices, dtype=np.float64)
        faces = np.asarray(m.faces, dtype=np.int64)
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        cross_vec = np.cross(v1 - v0, v2 - v0)
        dbl_area = np.linalg.norm(cross_vec, axis=1)

    n_verts = len(verts)
    face_areas = 0.5 * dbl_area
    face_normals = cross_vec / dbl_area[:, None]
    cos_thresh = math.cos(math.radians(float(crease_angle_deg)))

    faces_list = faces.tolist()
    vert_inc: list[list[tuple[int, int]]] = [[] for _ in range(n_verts)]
    for fi, (a, b, c) in enumerate(faces_list):
        vert_inc[a].append((fi, 0))
        vert_inc[b].append((fi, 1))
        vert_inc[c].append((fi, 2))

    fn_list = face_normals.tolist()
    fa_list = face_areas.tolist()
    verts_list = verts.tolist()

    new_verts: list[list[float]] = [row[:] for row in verts_list]
    new_normals: list[list[float]] = [[0.0, 0.0, 1.0] for _ in range(n_verts)]
    new_faces = [row[:] for row in faces_list]

    for v in range(n_verts):
        inc = vert_inc[v]
        if not inc:
            continue
        if len(inc) == 1:
            fi, _ = inc[0]
            new_normals[v] = fn_list[fi]
            continue

        # Faces incident to `v` share an edge (v, nb) when they both contain `nb`
        nb_to_items: dict[int, list[int]] = {}
        for idx_i, (fi, _) in enumerate(inc):
            a, b, c = faces_list[fi]
            if a != v:
                nb_to_items.setdefault(a, []).append(idx_i)
            if b != v:
                nb_to_items.setdefault(b, []).append(idx_i)
            if c != v:
                nb_to_items.setdefault(c, []).append(idx_i)

        adj: list[list[int]] = [[] for _ in range(len(inc))]
        for shared_items in nb_to_items.values():
            k = len(shared_items)
            if k < 2:
                continue
            for p in range(k):
                ip = shared_items[p]
                npx, npy, npz = fn_list[inc[ip][0]]
                for q in range(p + 1, k):
                    iq = shared_items[q]
                    nqx, nqy, nqz = fn_list[inc[iq][0]]
                    if npx * nqx + npy * nqy + npz * nqz >= cos_thresh:
                        adj[ip].append(iq)
                        adj[iq].append(ip)

        visited = [False] * len(inc)
        clusters: list[list[int]] = []
        for start in range(len(inc)):
            if visited[start]:
                continue
            comp = [start]
            visited[start] = True
            head = 0
            while head < len(comp):
                cur = comp[head]
                head += 1
                for nxt in adj[cur]:
                    if not visited[nxt]:
                        visited[nxt] = True
                        comp.append(nxt)
            clusters.append(comp)

        for cl_idx, comp in enumerate(clusters):
            sx = sy = sz = 0.0
            for item_idx in comp:
                fi, _ = inc[item_idx]
                w = fa_list[fi]
                nx, ny, nz = fn_list[fi]
                sx += w * nx
                sy += w * ny
                sz += w * nz
            nrm = math.sqrt(sx * sx + sy * sy + sz * sz)
            if nrm > 1e-24:
                n_vec = [sx / nrm, sy / nrm, sz / nrm]
            else:
                n_vec = fn_list[inc[comp[0]][0]]

            if cl_idx == 0:
                target_v = v
                new_normals[target_v] = n_vec
            else:
                target_v = len(new_verts)
                new_verts.append(verts_list[v])
                new_normals.append(n_vec)
            for item_idx in comp:
                fi, col = inc[item_idx]
                new_faces[fi][col] = target_v

    return trimesh.Trimesh(
        vertices=np.asarray(new_verts, dtype=np.float64),
        faces=np.asarray(new_faces, dtype=np.int64),
        vertex_normals=np.asarray(new_normals, dtype=np.float64),
        process=False,
    )


def _allocate_visual_triangle_budgets(
    raw_meshes: list[tuple[VisualPart, trimesh.Trimesh]],
    max_total_triangles: int,
) -> list[int]:
    """Allocate per-part triangle budgets: cap minor fasteners <=96, weight primary parts by surface area."""
    n = len(raw_meshes)
    raw_counts = [len(tm.faces) for _, tm in raw_meshes]
    targets = list(raw_counts)
    if sum(raw_counts) <= max_total_triangles and all(
        not any(k in (vp.label or "").lower() for k in ("screw", "bolt", "fastener", "set_screw"))
        or raw_counts[i] <= 160
        for i, (vp, _) in enumerate(raw_meshes)
    ):
        return targets

    fastener_idxs: list[int] = []
    primary_idxs: list[int] = []
    for i, (vp, _) in enumerate(raw_meshes):
        lbl = (vp.label or "").lower()
        if any(k in lbl for k in ("screw", "bolt", "fastener", "set_screw")):
            fastener_idxs.append(i)
        else:
            primary_idxs.append(i)

    # 1. Cap minor fasteners at <= 160 triangles each
    fastener_tris = 0
    if fastener_idxs:
        raw_caps = [min(raw_counts[i], 160) for i in fastener_idxs]
        if sum(raw_caps) <= max_total_triangles or not primary_idxs:
            for i, cap in zip(fastener_idxs, raw_caps):
                targets[i] = min(cap, max(4, max_total_triangles // max(len(fastener_idxs), 1))) if not primary_idxs and sum(raw_caps) > max_total_triangles else cap
                fastener_tris += targets[i]
        else:
            fastener_budget = max(4 * len(fastener_idxs), int(round(0.25 * max_total_triangles)))
            for i in fastener_idxs:
                targets[i] = max(4, min(raw_counts[i], fastener_budget // len(fastener_idxs)))
                fastener_tris += targets[i]

    if not primary_idxs:
        return targets

    rem_budget = max(4 * len(primary_idxs), int(max_total_triangles) - fastener_tris)
    raw_primary_total = sum(raw_counts[i] for i in primary_idxs)
    if raw_primary_total <= rem_budget:
        for i in primary_idxs:
            targets[i] = raw_counts[i]
        return targets

    # 2. Weight primary visual parts by sqrt(surface_area) * raw_faces^0.35
    weights: dict[int, float] = {}
    for i in primary_idxs:
        tm = raw_meshes[i][1]
        area = max(float(tm.area), 1e-6)
        weights[i] = math.sqrt(area) * (max(raw_counts[i], 1) ** 0.35)

    # Apply per-part minimum floor (min(raw, 400)) when budget accommodates it
    floors = {i: min(raw_counts[i], 400) for i in primary_idxs}
    if sum(floors.values()) <= rem_budget:
        for i in primary_idxs:
            targets[i] = floors[i]
    else:
        tight_floors = {i: min(raw_counts[i], 24) for i in primary_idxs}
        if sum(tight_floors.values()) <= rem_budget:
            for i in primary_idxs:
                targets[i] = tight_floors[i]
        else:
            w_sum = max(sum(weights.values()), 1e-12)
            for i in primary_idxs:
                targets[i] = max(4, min(raw_counts[i], int(round(rem_budget * weights[i] / w_sum))))
            return targets

    # 3. Water-filling redistribution of remaining budget to uncapped primary parts
    extra = rem_budget - sum(targets[i] for i in primary_idxs)
    while extra > 0:
        active = [i for i in primary_idxs if targets[i] < raw_counts[i]]
        if not active:
            break
        w_sum = sum(weights[i] for i in active)
        if w_sum <= 0.0:
            break
        added_this_round = 0
        for i in active:
            room = raw_counts[i] - targets[i]
            add_i = min(room, int(extra * (weights[i] / w_sum)))
            if add_i > 0:
                targets[i] += add_i
                added_this_round += add_i
        extra -= added_this_round
        if added_this_round == 0:
            # Distribute leftover (< len(active)) 1-by-1 by descending weight
            for i in sorted(active, key=lambda idx: weights[idx], reverse=True):
                if extra <= 0:
                    break
                if targets[i] < raw_counts[i]:
                    targets[i] += 1
                    extra -= 1
            break

    return targets


def build_visual_glb(
    visual_parts: list[VisualPart],
    out_path: Path,
    max_total_triangles: int = 24000,
) -> tuple[int, int]:
    """Build a decimated, multi-material PBR ``visual.glb`` in strict SI meters.

    Returns ``(total_triangles, file_bytes)``.
    """
    raw_meshes: list[tuple[VisualPart, trimesh.Trimesh]] = []
    for vp in visual_parts:
        tm = _shape_to_trimesh_mm(vp.shape_or_mesh)
        if len(tm.faces) == 0:
            continue
        raw_meshes.append((vp, tm))

    if not raw_meshes:
        raise ValueError("No valid triangle meshes generated for visual.glb")

    target_budgets = _allocate_visual_triangle_budgets(raw_meshes, max_total_triangles)

    scene = trimesh.Scene()
    total_out_tris = 0
    for idx, ((vp, tm), target_tris) in enumerate(zip(raw_meshes, target_budgets)):
        if len(tm.faces) > target_tris:
            tm = decimate_trimesh_qem(tm, target_triangles=target_tris)

        # Convert mm -> strict SI meters, then apply angle-creased smooth vertex normals
        tm_m = tm.copy()
        tm_m.apply_scale(1e-3)
        tm_m = _apply_creased_vertex_normals(tm_m, crease_angle_deg=34.0)
        cached_normals = tm_m._cache.cache.get("vertex_normals")

        rgba = _hex_to_rgba(vp.color_hex)
        tm_m.visual = trimesh.visual.TextureVisuals(
            material=trimesh.visual.material.PBRMaterial(
                name=vp.label or f"part_{idx}",
                baseColorFactor=rgba,
                metallicFactor=float(vp.metallic),
                roughnessFactor=float(vp.roughness),
                doubleSided=bool(vp.double_sided),
            )
        )
        if cached_normals is not None:
            tm_m._cache["vertex_normals"] = cached_normals

        node_name = vp.label or f"part_{idx}"
        if node_name in scene.geometry:
            node_name = f"{node_name}_{idx}"
        scene.add_geometry(tm_m, node_name=node_name, geom_name=node_name)
        total_out_tris += len(tm_m.faces)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    glb_bytes = scene.export(file_type="glb", include_normals=True)
    if glb_bytes[:4] != b"glTF":
        raise RuntimeError("Exported visual.glb is missing binary glTF magic header")
    out_path.write_bytes(glb_bytes)
    return int(total_out_tris), int(len(glb_bytes))


# ---------------------------------------------------------------------------
# Core Baker: Tet4 Assembly, Mass/Inertia Extraction, CB CMS & Verification
# ---------------------------------------------------------------------------


def _assemble_tet4_km(
    points_m: np.ndarray,
    tets: np.ndarray,
    region_ids: np.ndarray,
    regions: list[MaterialRegion],
    default_fdm_anisotropic: bool,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    """Assemble global SI ``(K, M)`` matrices using ``skfem`` ``ElementTetP1()``."""
    import skfem

    mesh = skfem.MeshTet(points_m.T, tets.T).oriented()
    if mesh.nelements != len(tets):
        raise RuntimeError(
            f"MeshTet orientation changed element count ({len(tets)} -> {mesh.nelements})"
        )

    elem = skfem.ElementVector(skfem.ElementTetP1())
    basis_k = skfem.CellBasis(mesh, elem, intorder=1)
    basis_m = skfem.CellBasis(mesh, elem, intorder=2)

    @skfem.BilinearForm
    def mass_form(u, v, w):
        return w["rho"] * (u[0] * v[0] + u[1] * v[1] + u[2] * v[2])

    def _make_elast_form(D: np.ndarray):
        nz_rows = [
            [(j, float(D[i, j])) for j in range(6) if abs(float(D[i, j])) > 0.0]
            for i in range(6)
        ]

        @skfem.BilinearForm
        def elast_form(u, v, w):
            gu, gv = u.grad, v.grad
            eu = (
                gu[0, 0],
                gu[1, 1],
                gu[2, 2],
                gu[0, 1] + gu[1, 0],
                gu[1, 2] + gu[2, 1],
                gu[0, 2] + gu[2, 0],
            )
            ev = (
                gv[0, 0],
                gv[1, 1],
                gv[2, 2],
                gv[0, 1] + gv[1, 0],
                gv[1, 2] + gv[2, 1],
                gv[0, 2] + gv[2, 0],
            )
            out = None
            for i in range(6):
                terms = nz_rows[i]
                if not terms:
                    continue
                s_i = terms[0][1] * eu[terms[0][0]]
                for j, d_ij in terms[1:]:
                    s_i = s_i + d_ij * eu[j]
                t = s_i * ev[i]
                out = t if out is None else out + t
            return out

        return elast_form

    K_total = None
    M_total = None
    for reg_idx, reg in enumerate(regions):
        elem_ids = np.where(region_ids == reg_idx)[0]
        if len(elem_ids) == 0:
            continue
        mat = get_material(reg.material_id)
        is_fdm = (
            bool(reg.fdm_anisotropic)
            if reg.fdm_anisotropic is not None
            else (bool(default_fdm_anisotropic) and bool(mat.is_printed_polymer))
        )
        D = constitutive_matrix_3d(mat, fdm_anisotropic=is_fdm)

        if len(regions) == 1 and len(elem_ids) == mesh.nelements:
            bk = basis_k
            bm = basis_m
        else:
            bk = skfem.CellBasis(mesh, elem, intorder=1, elements=elem_ids)
            bm = skfem.CellBasis(mesh, elem, intorder=2, elements=elem_ids)

        K_r = _make_elast_form(D).assemble(bk)
        M_r = mass_form.assemble(bm, rho=float(mat.density_kg_m3))
        K_total = K_r if K_total is None else (K_total + K_r)
        M_total = M_r if M_total is None else (M_total + M_r)

    if K_total is None or M_total is None:
        raise RuntimeError("No tetrahedral elements assembled across material regions")

    # Guarantee canonical DOF permutation [3*node + 0, 3*node + 1, 3*node + 2]
    nodal_dofs = basis_k.nodal_dofs  # (3, n_vertices)
    perm = nodal_dofs.T.ravel()
    if not np.array_equal(perm, np.arange(basis_k.N)):
        K_total = K_total[perm, :][:, perm]
        M_total = M_total[perm, :][:, perm]

    return mesh, K_total, M_total, mesh.p.T


def _compute_mass_com_inertia(
    M: Any,
    coords_m: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute exact SI ``(mass_kg, com_m, inertia_kg_m2)`` from consistent mass matrix ``M``."""
    n_v = coords_m.shape[0]
    n_dof = 3 * n_v
    rx = np.zeros(n_dof, dtype=np.float64)
    ry = np.zeros(n_dof, dtype=np.float64)
    rz = np.zeros(n_dof, dtype=np.float64)
    rx[0::3] = 1.0
    ry[1::3] = 1.0
    rz[2::3] = 1.0

    mass_kg = float(rx @ (M @ rx))
    if mass_kg <= 1e-18:
        raise RuntimeError(f"Non-positive assembled mass: {mass_kg} kg")

    x_only = np.zeros(n_dof, dtype=np.float64)
    y_only = np.zeros(n_dof, dtype=np.float64)
    z_only = np.zeros(n_dof, dtype=np.float64)
    x_only[0::3] = coords_m[:, 0]
    y_only[1::3] = coords_m[:, 1]
    z_only[2::3] = coords_m[:, 2]

    com_m = np.array(
        [
            float(rx @ (M @ x_only)) / mass_kg,
            float(ry @ (M @ y_only)) / mass_kg,
            float(rz @ (M @ z_only)) / mass_kg,
        ],
        dtype=np.float64,
    )

    # Exact rotational kinetic energy identity for P1 elements: u_rot = omega x (r - com)
    d = coords_m - com_m[None, :]
    R_rot = np.zeros((n_dof, 3), dtype=np.float64)
    R_rot[0::3, 1] = d[:, 2]
    R_rot[0::3, 2] = -d[:, 1]
    R_rot[1::3, 0] = -d[:, 2]
    R_rot[1::3, 2] = d[:, 0]
    R_rot[2::3, 0] = d[:, 1]
    R_rot[2::3, 1] = -d[:, 0]

    I_com = R_rot.T @ (M @ R_rot)
    I_com = 0.5 * (I_com + I_com.T)
    return mass_kg, com_m, I_com


def _verify_and_extract_tip_stiffness(
    K_cb: np.ndarray,
    M_cb: np.ndarray,
) -> tuple[float, np.ndarray, dict[str, float], dict[str, float]]:
    """Verify CB symmetry, SPD mass, 6 rigid-body zero modes, and grounded tip stiffnesses."""
    K_s = 0.5 * (K_cb + K_cb.T)
    M_s = 0.5 * (M_cb + M_cb.T)

    if not np.allclose(K_cb, K_s, rtol=1e-7, atol=1e-7 * max(float(np.max(np.abs(K_cb))), 1.0)):
        raise RuntimeError("K_cb failed symmetry check")
    if not np.allclose(M_cb, M_s, rtol=1e-7, atol=1e-12):
        raise RuntimeError("M_cb failed symmetry check")

    m_eigs = np.linalg.eigvalsh(M_s)
    if float(np.min(m_eigs)) <= 0.0:
        raise RuntimeError(f"M_cb is not strictly positive definite (min eig={float(np.min(m_eigs))})")

    eigvals_free = eigh(K_s, M_s, eigvals_only=True)
    rigid_freqs = np.sqrt(np.maximum(np.abs(eigvals_free[:6]), 0.0)) / (2.0 * math.pi)
    rigid_max_hz = float(np.max(rigid_freqs))
    if rigid_max_hz >= 0.5:
        raise RuntimeError(
            f"Free-free rigid-body modes exceed 0.5 Hz tolerance (max={rigid_max_hz:.4f} Hz, eigs[:6]={eigvals_free[:6]})"
        )
    first_elastic_hz = math.sqrt(max(float(eigvals_free[6]), 0.0)) / (2.0 * math.pi)
    if float(eigvals_free[6]) <= 0.0 or first_elastic_hz <= 1.0:
        raise RuntimeError(
            f"First elastic mode is too low/non-positive ({first_elastic_hz:.4f} Hz, eig={eigvals_free[6]}); "
            "disconnected mesh or rank-deficient port!"
        )

    K_bb = K_s[:12, :12].copy()
    if K_s.shape[0] > 12:
        max_km = float(np.max(np.abs(K_s[:12, 12:])))
        max_k = max(float(np.max(np.abs(K_s))), 1.0)
        if max_km / max_k > 1e-6:
            raise RuntimeError(f"Craig-Bampton boundary-mode static coupling K_bm != 0 ({max_km / max_k:.3e})")

    # Ground Port 0 (DOFs 0..5) and verify SPD on Port 1 (DOFs 6..11)
    K_11 = 0.5 * (K_bb[6:12, 6:12] + K_bb[6:12, 6:12].T)
    k11_eigs = np.linalg.eigvalsh(K_11)
    if float(np.min(k11_eigs)) <= 0.0:
        raise RuntimeError(f"Grounded K_bb[6:12, 6:12] is not SPD (min eig={float(np.min(k11_eigs))})")

    diag_k = np.maximum(np.diag(K_11), 1e-18)
    s = 1.0 / np.sqrt(diag_k)
    K_scaled = (s[:, None] * K_11) * s[None, :]
    c_fac = cho_factor(K_scaled, check_finite=False)
    Y_scaled = cho_solve(c_fac, np.diag(s), check_finite=False)
    C_11 = s[:, None] * Y_scaled

    port_stiff_n_mm = {
        "kx": round(float(1.0 / (max(float(C_11[0, 0]), 1e-18) * 1000.0)), 3),
        "ky": round(float(1.0 / (max(float(C_11[1, 1]), 1e-18) * 1000.0)), 3),
        "kz": round(float(1.0 / (max(float(C_11[2, 2]), 1e-18) * 1000.0)), 3),
    }
    port_rot_nm_rad = {
        "krx": round(float(1.0 / max(float(C_11[3, 3]), 1e-18)), 3),
        "kry": round(float(1.0 / max(float(C_11[4, 4]), 1e-18)), 3),
        "krz": round(float(1.0 / max(float(C_11[5, 5]), 1e-18)), 3),
    }
    return rigid_max_hz, K_bb, port_stiff_n_mm, port_rot_nm_rad


def bake_solid_superelement(
    spec: BakedAssetSpec,
    regions: list[MaterialRegion],
    visual_parts: list[VisualPart] | None = None,
    assets_root: Path | None = None,
) -> BakedAssetResult:
    """Mesh, assemble, condense, verify, and persist a CAD hardware superelement bundle."""
    if not regions:
        raise ValueError("regions must contain at least one MaterialRegion")

    root = assets_root if assets_root is not None else ASSETS_ROOT
    asset_dir = root / spec.category / spec.asset_id
    asset_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate conformal 3D Tet4 mesh in mm
    pts_mm, tets, region_ids = _mesh_regions_tet4(
        regions=regions,
        mesh_size_mm=spec.mesh_size_mm,
        fea_surface_max_triangles=spec.fea_surface_max_triangles,
        snap_coords_mm=(spec.port0.vp_coord_mm, spec.port1.vp_coord_mm),
    )
    pts_m = pts_mm * 1e-3

    # 2. Assemble SI (K, M) using skfem ElementTetP1 + constitutive_matrix_3d
    mesh, K, M, coords_m = _assemble_tet4_km(
        points_m=pts_m,
        tets=tets,
        region_ids=region_ids,
        regions=regions,
        default_fdm_anisotropic=spec.is_fdm_anisotropic,
    )
    coords_mm = coords_m * 1000.0

    # 3. Exact SI mass, center of mass, and rotational inertia tensor
    mass_kg, com_m, inertia_kg_m2 = _compute_mass_com_inertia(M, coords_m)

    # 4. Select Port 0 and Port 1 boundary patches & validate rank-6 VPT
    bnd_nodes = np.asarray(mesh.boundary_nodes(), dtype=np.int64)
    bnd_coords_mm = coords_mm[bnd_nodes]

    p0_mask = np.asarray(spec.port0.selector(bnd_coords_mm), dtype=bool)
    p1_mask = np.asarray(spec.port1.selector(bnd_coords_mm), dtype=bool)
    p0_nodes = bnd_nodes[p0_mask]
    p1_nodes = np.setdiff1d(bnd_nodes[p1_mask], p0_nodes)

    if len(p0_nodes) < 4:
        raise ValueError(
            f"Port 0 for '{spec.asset_id}' matched only {len(p0_nodes)} boundary nodes (need >= 4)"
        )
    if len(p1_nodes) < 4:
        raise ValueError(
            f"Port 1 for '{spec.asset_id}' matched only {len(p1_nodes)} boundary nodes (need >= 4)"
        )

    p0_vp_m = np.asarray(spec.port0.vp_coord_mm, dtype=np.float64) * 1e-3
    p1_vp_m = np.asarray(spec.port1.vp_coord_mm, dtype=np.float64) * 1e-3

    for label, p_nodes, vp_m in (("Port 0", p0_nodes, p0_vp_m), ("Port 1", p1_nodes, p1_vp_m)):
        R_test = virtual_point_transform(coords_m[p_nodes], vp_m)
        if np.linalg.matrix_rank(R_test, tol=1e-9) != 6:
            raise ValueError(
                f"{label} patch for '{spec.asset_id}' is rank-deficient (<6) — nodes may be collinear"
            )

    port_coords_m = np.vstack([p0_vp_m, p1_vp_m]).astype(np.float64)

    # 5. Deterministic source hash
    hash_payload = {
        "schema_version": 1,
        "category": spec.category,
        "asset_id": spec.asset_id,
        "reference_material_id": spec.reference_material_id,
        "is_multi_material": spec.is_multi_material,
        "is_fdm_anisotropic": spec.is_fdm_anisotropic,
        "mesh_size_mm": spec.mesh_size_mm,
        "n_internal_modes": spec.n_internal_modes,
        "num_vertices": int(mesh.nvertices),
        "num_elements": int(mesh.nelements),
        "port0_mm": list(spec.port0.vp_coord_mm),
        "port1_mm": list(spec.port1.vp_coord_mm),
        "recipe": spec.part_recipe,
    }
    source_hash = hashlib.sha256(
        json.dumps(hash_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()

    # 6. Exact RBE3 VPT + Craig-Bampton CMS condensation
    se = condense_solid_patch_to_vp(
        K,
        M,
        port_node_sets=[
            (p0_nodes, coords_m[p0_nodes], p0_vp_m),
            (p1_nodes, coords_m[p1_nodes], p1_vp_m),
        ],
        n_modes=spec.n_internal_modes,
        name=spec.asset_id,
        source_hash=source_hash,
        interface_mode="rbe3",
        mass_kg=mass_kg,
    )

    # 7. Strict physics & matrix verification gate before writing
    rigid_max_hz, K_bb, port_stiff_n_mm, port_rot_nm_rad = _verify_and_extract_tip_stiffness(
        se.K_cb, se.M_cb
    )

    # 8. Build decimated PBR visual.glb in strict SI meters
    vis_list = visual_parts
    if vis_list is None:
        vis_list = []
        for idx, reg in enumerate(regions):
            mat = get_material(reg.material_id)
            vis_list.append(
                VisualPart(
                    shape_or_mesh=reg.solid,
                    label=reg.label or f"part_{idx}",
                    color_hex=reg.color_hex or mat.color_hex,
                    metallic=reg.metallic,
                    roughness=reg.roughness,
                )
            )
    glb_path = asset_dir / "visual.glb"
    vis_tris, vis_bytes = build_visual_glb(
        vis_list, glb_path, max_total_triangles=spec.visual_max_triangles
    )

    # 9. Write bundle artifacts
    ref_mat = get_material(spec.reference_material_id)
    recipe_out = {
        "schema_version": 1,
        "category": spec.category,
        "asset_id": spec.asset_id,
        "source_cad": spec.source_cad,
        "source_url": spec.source_url,
        "regions": [
            {
                "label": r.label or f"region_{i}",
                "material_id": r.material_id,
                "fdm_anisotropic": r.fdm_anisotropic,
            }
            for i, r in enumerate(regions)
        ],
        **spec.part_recipe,
    }
    (asset_dir / "part_recipe.json").write_text(
        json.dumps(recipe_out, indent=2), encoding="utf-8"
    )

    np.savez(
        asset_dir / "superelement.npz",
        K_cb=np.asarray(se.K_cb, dtype=np.float64),
        M_cb=np.asarray(se.M_cb, dtype=np.float64),
        K_bb=np.asarray(K_bb, dtype=np.float64),
        port_coords_m=port_coords_m,
        fixed_interface_freqs_hz=np.asarray(se.fixed_interface_freqs_hz, dtype=np.float64),
        com_m=com_m,
        inertia_kg_m2=inertia_kg_m2,
    )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "category": spec.category,
        "asset_id": spec.asset_id,
        "name": spec.name,
        "source_cad": spec.source_cad,
        "source_url": spec.source_url,
        "baked_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "reference_material_id": spec.reference_material_id,
        "reference_youngs_modulus_pa": float(ref_mat.youngs_modulus_pa),
        "reference_density_kg_m3": float(ref_mat.density_kg_m3),
        "is_multi_material": bool(spec.is_multi_material),
        "is_fdm_anisotropic": bool(spec.is_fdm_anisotropic),
        "element_type": "Tet4",
        "num_vertices": int(mesh.nvertices),
        "num_elements": int(mesh.nelements),
        "num_dofs": int(3 * mesh.nvertices),
        "n_boundary_dofs": 12,
        "n_internal_modes": int(se.n_internal_modes),
        "rigid_body_max_freq_hz": round(float(rigid_max_hz), 6),
        "mass_kg": round(float(mass_kg), 6),
        "com_mm": [round(float(v * 1000.0), 4) for v in com_m],
        "port0_mm": [round(float(v), 4) for v in spec.port0.vp_coord_mm],
        "port1_mm": [round(float(v), 4) for v in spec.port1.vp_coord_mm],
        "port_stiffness_n_per_mm": port_stiff_n_mm,
        "port_rot_stiffness_nm_per_rad": port_rot_nm_rad,
        "fixed_interface_freqs_hz": [round(float(f), 2) for f in se.fixed_interface_freqs_hz],
        "visual_glb": "visual.glb",
        "visual_triangles": int(vis_tris),
        "visual_bytes": int(vis_bytes),
        "cad_trust_pct": int(spec.cad_trust_pct),
        "source_hash": source_hash,
    }
    (asset_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    clear_asset_cache()

    return BakedAssetResult(
        category=spec.category,
        asset_id=spec.asset_id,
        asset_dir=asset_dir,
        manifest=manifest,
        K_cb=se.K_cb,
        M_cb=se.M_cb,
        K_bb=K_bb,
        port_coords_m=port_coords_m,
        com_m=com_m,
        inertia_kg_m2=inertia_kg_m2,
    )


def bake_step_file(
    step_path: str | Path,
    category: str,
    asset_id: str,
    material_id: str = "titanium_grade5",
    name: str | None = None,
    port0_axis: int = 2,
    port0_end: str = "max",
    port1_axis: int = 2,
    port1_end: str = "min",
    mesh_size_mm: float = 1.6,
    n_internal_modes: int = 8,
    cad_trust_pct: int = 85,
    assets_root: Path | None = None,
) -> BakedAssetResult:
    """Generic CLI helper to bake an arbitrary STEP solid into a 2-port hardware superelement."""
    import build123d as b3d

    p = Path(step_path)
    if not p.is_file():
        raise FileNotFoundError(f"STEP file not found: {p}")

    imported = b3d.import_step(str(p))
    solids = imported.solids()
    if not solids:
        raise ValueError(f"No 3D solid volumes found in {p}")
    solid = solids[0] if len(solids) == 1 else b3d.Compound(children=solids)

    bb = solid.bounding_box()
    mins = np.array([bb.min.X, bb.min.Y, bb.min.Z], dtype=np.float64)
    maxs = np.array([bb.max.X, bb.max.Y, bb.max.Z], dtype=np.float64)

    def _port_coord(axis: int, end: str) -> np.ndarray:
        c = 0.5 * (mins + maxs)
        c[axis] = mins[axis] if end == "min" else maxs[axis]
        return c

    p0_orig = _port_coord(port0_axis, port0_end)
    solid_shifted = solid.translate((-float(p0_orig[0]), -float(p0_orig[1]), -float(p0_orig[2])))
    bb_s = solid_shifted.bounding_box()
    s_mins = np.array([bb_s.min.X, bb_s.min.Y, bb_s.min.Z], dtype=np.float64)
    s_maxs = np.array([bb_s.max.X, bb_s.max.Y, bb_s.max.Z], dtype=np.float64)

    p0_mm = (0.0, 0.0, 0.0)
    p1_arr = 0.5 * (s_mins + s_maxs)
    p1_arr[port1_axis] = s_mins[port1_axis] if port1_end == "min" else s_maxs[port1_axis]
    p1_mm = (float(p1_arr[0]), float(p1_arr[1]), float(p1_arr[2]))

    tol = max(0.25, 0.18 * mesh_size_mm)
    mat = get_material(material_id)
    spec = BakedAssetSpec(
        category=category,
        asset_id=asset_id,
        name=name or asset_id.replace("_", " ").title(),
        source_cad=p.name,
        source_url="local",
        reference_material_id=material_id,
        cad_trust_pct=cad_trust_pct,
        port0=PortDefinition(p0_mm, select_plane_box(port0_axis, p0_mm[port0_axis], tol_mm=tol)),
        port1=PortDefinition(p1_mm, select_plane_box(port1_axis, p1_mm[port1_axis], tol_mm=tol)),
        n_internal_modes=n_internal_modes,
        mesh_size_mm=mesh_size_mm,
        is_multi_material=False,
        is_fdm_anisotropic=bool(mat.is_printed_polymer),
    )
    return bake_solid_superelement(
        spec=spec,
        regions=[MaterialRegion(solid=solid_shifted, material_id=material_id, label=asset_id)],
        assets_root=assets_root,
    )


# ---------------------------------------------------------------------------
# Isolated Gmsh Worker Entrypoints (NO build123d / cadquery-ocp imported here!)
# ---------------------------------------------------------------------------


def _gmsh_extract_tet4_arrays(gmsh: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_tags_arr = np.asarray(node_tags, dtype=np.int64)
    coords_mm = np.asarray(node_coords, dtype=np.float64).reshape(-1, 3)

    sort_order = np.argsort(node_tags_arr)
    sorted_tags = node_tags_arr[sort_order]

    elem_tags, elem_node_tags = gmsh.model.mesh.getElementsByType(4)
    if len(elem_tags) == 0:
        raise RuntimeError("Gmsh generated 0 Tet4 elements")

    raw_tet_tags = np.asarray(elem_node_tags, dtype=np.int64).reshape(-1, 4)
    pos = np.searchsorted(sorted_tags, raw_tet_tags.ravel())
    if not np.all(sorted_tags[pos] == raw_tet_tags.ravel()):
        raise RuntimeError("Gmsh returned tetrahedra referencing missing node tags")
    raw_indices = sort_order[pos].reshape(-1, 4)
    used_raw, compact_inv = np.unique(raw_indices, return_inverse=True)
    points_mm = np.ascontiguousarray(coords_mm[used_raw], dtype=np.float64)
    tets = np.ascontiguousarray(compact_inv.reshape(-1, 4), dtype=np.int32)
    return points_mm, tets, used_raw


def _gmsh_worker_2d_main(geom_path: Path, out_npz: Path, mesh_size_mm: float) -> None:
    import gmsh

    gmsh.initialize(argv=["gmsh"], interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setString("Geometry.OCCTargetUnit", "MM")
        gmsh.model.occ.importShapes(str(geom_path))
        gmsh.model.occ.synchronize()
        h = max(0.25, float(mesh_size_mm))
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", max(0.70, 0.50 * h))
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max(h * 1.65, 2.8))
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 4)
        try:
            gmsh.model.mesh.generate(2)
        except Exception:
            gmsh.model.mesh.clear()
            gmsh.option.setNumber("Mesh.Algorithm", 1)  # MeshAdapt fallback
            gmsh.model.mesh.generate(2)
        gmsh.model.mesh.removeDuplicateNodes()

        nt, nc, _ = gmsh.model.mesh.getNodes()
        et, ent = gmsh.model.mesh.getElementsByType(2)
        if len(et) == 0:
            raise RuntimeError("Gmsh generated 0 surface triangles")
        nt_arr = np.asarray(nt, dtype=np.int64)
        coords = np.asarray(nc, dtype=np.float64).reshape(-1, 3)
        raw_t = np.asarray(ent, dtype=np.int64).reshape(-1, 3)
        so = np.argsort(nt_arr)
        st = nt_arr[so]
        pos = np.searchsorted(st, raw_t.ravel())
        ri = so[pos].reshape(-1, 3)
        u, inv = np.unique(ri, return_inverse=True)
        np.savez(
            out_npz,
            points_mm=np.ascontiguousarray(coords[u], dtype=np.float64),
            tris=np.ascontiguousarray(inv.reshape(-1, 3), dtype=np.int32),
        )
    finally:
        if gmsh.isInitialized():
            gmsh.finalize()


def _gmsh_worker_stl_main(stl_path: Path, out_npz: Path, mesh_size_mm: float) -> None:
    import gmsh

    gmsh.initialize(argv=["gmsh"], interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.merge(str(stl_path))
        gmsh.model.mesh.createTopology()
        surfaces = gmsh.model.getEntities(2)
        if not surfaces:
            raise RuntimeError("No surface entities created from STL")
        loop = gmsh.model.geo.addSurfaceLoop([e[1] for e in surfaces])
        gmsh.model.geo.addVolume([loop])
        gmsh.model.geo.synchronize()
        h = max(0.25, float(mesh_size_mm))
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", max(0.25, 0.35 * h))
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h * 1.15)
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT (fast)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        try:
            gmsh.model.mesh.generate(3)
        except Exception:
            # Robust Delaunay fallback for intricate SLM fillets where HXT rejects thin edges
            gmsh.model.mesh.clear()
            gmsh.merge(str(stl_path))
            gmsh.model.mesh.createTopology()
            surfaces = gmsh.model.getEntities(2)
            loop = gmsh.model.geo.addSurfaceLoop([e[1] for e in surfaces])
            gmsh.model.geo.addVolume([loop])
            gmsh.model.geo.synchronize()
            gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay
            gmsh.option.setNumber("Mesh.Optimize", 1)
            gmsh.model.mesh.generate(3)
        points_mm, tets, _ = _gmsh_extract_tet4_arrays(gmsh)
        np.savez(out_npz, points_mm=points_mm, tets=tets)
    finally:
        if gmsh.isInitialized():
            gmsh.finalize()


def _gmsh_worker_multi_main(spec_json: Path, out_npz: Path) -> None:
    import gmsh

    cfg = json.loads(spec_json.read_text(encoding="utf-8"))
    files = [Path(f) for f in cfg["files"]]
    h = max(0.25, float(cfg["mesh_size_mm"]))

    gmsh.initialize(argv=["gmsh"], interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setString("Geometry.OCCTargetUnit", "MM")

        input_vols: list[tuple[int, int]] = []
        input_vol_region: list[int] = []
        for reg_idx, fp in enumerate(files):
            imported = gmsh.model.occ.importShapes(str(fp))
            vols = [dt for dt in imported if dt[0] == 3]
            if not vols:
                gmsh.model.occ.synchronize()
                all_v = gmsh.model.getEntities(3)
                known = set(input_vols)
                vols = [v for v in all_v if v not in known]
            for v in vols:
                input_vols.append(v)
                input_vol_region.append(reg_idx)

        if not input_vols:
            raise RuntimeError("No 3D volumes imported across multi-region inputs")

        _, out_map = gmsh.model.occ.fragment(input_vols, [])
        gmsh.model.occ.synchronize()

        # Assign each fragmented 3D entity tag to its first source region index
        frag_tag_to_region: dict[int, int] = {}
        for src_idx, mapped in enumerate(out_map):
            reg_idx = input_vol_region[src_idx]
            for dim, tag in mapped:
                if dim == 3 and tag not in frag_tag_to_region:
                    frag_tag_to_region[int(tag)] = reg_idx

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min(0.32, 0.22 * h))
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h * 1.1)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        try:
            gmsh.model.mesh.generate(3)
        except Exception:
            gmsh.model.mesh.clear()
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min(0.20, 0.16 * h))
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 14)
            gmsh.model.mesh.generate(3)

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_tags_arr = np.asarray(node_tags, dtype=np.int64)
        coords_mm = np.asarray(node_coords, dtype=np.float64).reshape(-1, 3)
        sort_order = np.argsort(node_tags_arr)
        sorted_tags = node_tags_arr[sort_order]

        tet_blocks: list[np.ndarray] = []
        reg_blocks: list[np.ndarray] = []
        for _, vtag in gmsh.model.getEntities(3):
            et, ent = gmsh.model.mesh.getElementsByType(4, tag=vtag)
            if len(et) == 0:
                continue
            r_idx = frag_tag_to_region.get(int(vtag), 0)
            arr = np.asarray(ent, dtype=np.int64).reshape(-1, 4)
            tet_blocks.append(arr)
            reg_blocks.append(np.full(len(arr), r_idx, dtype=np.int32))

        if not tet_blocks:
            raise RuntimeError("Gmsh multi-region generated 0 Tet4 elements")

        raw_tet_tags = np.vstack(tet_blocks)
        region_ids = np.concatenate(reg_blocks)
        pos = np.searchsorted(sorted_tags, raw_tet_tags.ravel())
        if not np.all(sorted_tags[pos] == raw_tet_tags.ravel()):
            raise RuntimeError("Gmsh returned tetrahedra referencing missing node tags")
        raw_indices = sort_order[pos].reshape(-1, 4)
        used_raw, compact_inv = np.unique(raw_indices, return_inverse=True)
        points_mm = np.ascontiguousarray(coords_mm[used_raw], dtype=np.float64)
        tets = np.ascontiguousarray(compact_inv.reshape(-1, 4), dtype=np.int32)

        np.savez(out_npz, points_mm=points_mm, tets=tets, region_ids=region_ids)
    finally:
        if gmsh.isInitialized():
            gmsh.finalize()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "--gmsh-worker-2d" and len(sys.argv) >= 5:
        _gmsh_worker_2d_main(Path(sys.argv[2]), Path(sys.argv[3]), float(sys.argv[4]))
    elif mode == "--gmsh-worker-stl" and len(sys.argv) >= 5:
        _gmsh_worker_stl_main(Path(sys.argv[2]), Path(sys.argv[3]), float(sys.argv[4]))
    elif mode == "--gmsh-worker-multi" and len(sys.argv) >= 4:
        _gmsh_worker_multi_main(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        sys.stderr.write(
            "Usage: python -m printsim.cad_assets.baker "
            "[--gmsh-worker-2d|--gmsh-worker-stl|--gmsh-worker-multi] ...\n"
        )
        raise SystemExit(2)
