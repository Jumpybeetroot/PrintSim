"""3D multi-material orthotropic continuum FEA calibration benchmark for PrintSim X-gantry beams & rails.

Constructs multi-material 3D CAD assemblies with OpenCASCADE (``gmsh.model.occ``), meshes with
Gmsh, and solves 3D orthotropic/isotropic continuum elasticity using ``scikit-fem`` Tet10 (P2)
finite elements (with Tet4 comparison for mesh-convergence verification).

Critical modeling principles:
- Discrete bolted-interface contact via ``occ.fragment``: a real 0.25 mm air gap separates rail
  and tube/extrusion between bolts so zero artificial continuous bonding forces ``gamma = 1.0``.
- Real MGN12/MGN9 counterbores (Ø6.0 mm) + clearance holes (Ø3.8 mm) so M3 bolt shafts (Ø3.0 mm)
  do not falsely weld to the rail bore wall; load transfers via bolt head seating, bolt shank
  bending/bearing, and Mindlin dry-friction bolted clamping pads (with Tribology/Sherif-Kossa
  interfacial asperity stiffness under full 2-wall / T-nut preload vs low-preload single-wall
  bolt-bearing).
- True 3D symmetric positive-definite orthotropic elasticity tensors ``C_ortho`` for CF laminates
  with Mahieux-Reifsnider thermal softening at 85 C / 100 C / 110 C.

Run:
    uv run python scripts/calibrate_xbeam_3d_fea.py
    uv run python scripts/calibrate_xbeam_3d_fea.py --only bare bending torsion
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import math
import time
from typing import Any

import gmsh
import numpy as np
import skfem
from scipy.sparse.linalg import splu

from printsim.fea.components import (
    hertzian_rail_carriage_stiffness,
    newmark_rail_composite_gamma,
)
from printsim.schema import get_material, get_profile


# ---------------------------------------------------------------------------
# 1. 3D Orthotropic & Isotropic Constitutive Elasticity Tensors
# ---------------------------------------------------------------------------


def ortho_constitutive_3d(
    e_l_pa: float,
    e_t_pa: float,
    g_lt_pa: float,
    g_tt_pa: float,
    nu_lt: float = 0.28,
    nu_tt: float = 0.35,
) -> np.ndarray:
    """Build SPD 6x6 3D orthotropic elasticity tensor D (Voigt [xx, yy, zz, xy, yz, xz], fibers along X)."""
    s_mat = np.array(
        [
            [1.0 / e_l_pa, -nu_lt / e_l_pa, -nu_lt / e_l_pa, 0.0, 0.0, 0.0],
            [-nu_lt / e_l_pa, 1.0 / e_t_pa, -nu_tt / e_t_pa, 0.0, 0.0, 0.0],
            [-nu_lt / e_l_pa, -nu_tt / e_t_pa, 1.0 / e_t_pa, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0 / g_lt_pa, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0 / g_tt_pa, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / g_lt_pa],
        ],
        dtype=np.float64,
    )
    d_mat = np.linalg.inv(s_mat)
    d_mat = 0.5 * (d_mat + d_mat.T)
    min_eig = float(np.linalg.eigvalsh(d_mat).min())
    if min_eig <= 0.0:
        raise ValueError(f"Non-SPD orthotropic elasticity tensor (min eig={min_eig})")
    return d_mat


def iso_constitutive_3d(e_pa: float, nu: float) -> np.ndarray:
    """Build SPD 6x6 3D isotropic elasticity tensor D."""
    c = e_pa / ((1.0 + nu) * (1.0 - 2.0 * nu))
    d_mat = np.zeros((6, 6), dtype=np.float64)
    d_mat[0:3, 0:3] = c * np.array(
        [
            [1.0 - nu, nu, nu],
            [nu, 1.0 - nu, nu],
            [nu, nu, 1.0 - nu],
        ],
        dtype=np.float64,
    )
    g = e_pa / (2.0 * (1.0 + nu))
    d_mat[3, 3] = d_mat[4, 4] = d_mat[5, 5] = g
    return d_mat


def normal_bearing_constitutive_3d(e_norm_pa: float = 100.0e9, axis: str = "y") -> np.ndarray:
    """Build orthotropic bearing-only pad (stiff normal compression/tension, compliant transverse shear)."""
    e_soft = 2.0e6
    g_soft = 0.8e6
    if axis == "y":
        e1, e2, e3 = e_soft, e_norm_pa, e_soft
    elif axis == "z":
        e1, e2, e3 = e_soft, e_soft, e_norm_pa
    else:
        e1, e2, e3 = e_norm_pa, e_soft, e_soft
    s_mat = np.diag([1.0 / e1, 1.0 / e2, 1.0 / e3, 1.0 / g_soft, 1.0 / g_soft, 1.0 / g_soft])
    return np.linalg.inv(s_mat)


def mindlin_pad_constitutive_3d(g_shear_gpa: float, e_norm_gpa: float = 80.0) -> np.ndarray:
    """Build Mindlin interfacial clamping pad: stiff normal contact (80 GPa), finite X/Z shear stiffness."""
    e_t = max(g_shear_gpa * 2.64, 1.0e-4) * 1e9
    g = max(g_shear_gpa, 1.0e-4) * 1e9
    s_mat = np.diag([1.0 / e_t, 1.0 / (e_norm_gpa * 1e9), 1.0 / e_t, 1.0 / g, 1.0 / g, 1.0 / e_t])
    return np.linalg.inv(s_mat)


def cf_constitutive_at_temp(material_id: str, temp_c: float = 25.0) -> tuple[np.ndarray, float, float]:
    """Return (D_6x6, E_L_eff_Pa, G_LT_eff_Pa) for CF tube laminate with Mahieux-Reifsnider retention."""
    mat = get_material(material_id)
    r_e = mat.modulus_retention_at_temp(temp_c)
    r_g = mat.shear_modulus_retention_at_temp(temp_c)
    e_l = mat.youngs_modulus_pa * r_e
    g_lt = mat.effective_shear_modulus_pa * r_g

    if material_id == "carbon_fiber_torsional_45":
        e_t_25, g_tt_25 = 22.0e9, 5.5e9
    elif material_id == "carbon_fiber_tube":
        e_t_25, g_tt_25 = 11.0e9, 3.8e9
    else:
        e_t_25 = mat.youngs_modulus_pa * float(mat.e2_ratio)
        g_tt_25 = 0.7 * mat.effective_shear_modulus_pa

    e_t = e_t_25 * r_g
    g_tt = g_tt_25 * r_g
    return ortho_constitutive_3d(e_l, e_t, g_lt, g_tt, nu_lt=0.28, nu_tt=0.35), e_l, g_lt


# ---------------------------------------------------------------------------
# 2. OpenCASCADE Geometry Builders
# ---------------------------------------------------------------------------


def _make_rounded_rect_face(cx: float, cy: float, w: float, h: float, r: float) -> int:
    occ = gmsh.model.occ
    hw, hh = 0.5 * w, 0.5 * h
    if r <= 1e-6:
        return occ.addRectangle(cx - hw, cy - hh, 0.0, w, h)
    r1 = occ.addRectangle(cx - hw + r, cy - hh, 0.0, w - 2.0 * r, h)
    r2 = occ.addRectangle(cx - hw, cy - hh + r, 0.0, w, h - 2.0 * r)
    pts = [
        (cx - hw + r, cy - hh + r),
        (cx + hw - r, cy - hh + r),
        (cx + hw - r, cy + hh - r),
        (cx - hw + r, cy + hh - r),
    ]
    objs = [(2, r1), (2, r2)] + [(2, occ.addDisk(px, py, 0.0, r, r)) for px, py in pts]
    fused, _ = occ.fuse([objs[0]], objs[1:])
    return fused[0][1]


def _make_2020_tslot_face() -> int:
    """Create connected Misumi HFS5-2020 T-slot face in XY plane (A ~ 184 mm^2, I ~ 0.742 cm^4)."""
    occ = gmsh.model.occ
    outer = _make_rounded_rect_face(0.0, 0.0, 20.0, 20.0, 0.8)
    bore = occ.addDisk(0.0, 0.0, 0.0, 2.5, 2.5)
    cuts = [(2, bore)]
    for s in (1, -1):
        cuts.append((2, occ.addRectangle(8.35 if s > 0 else -10.5, -3.1, 0.0, 2.15, 6.2)))
        cuts.append((2, occ.addRectangle(5.55 if s > 0 else -8.35, -5.4, 0.0, 2.8, 10.8)))
        cuts.append((2, occ.addRectangle(4.1 if s > 0 else -5.55, -3.1, 0.0, 1.45, 6.2)))
        cuts.append((2, occ.addRectangle(-3.1, 8.35 if s > 0 else -10.5, 0.0, 6.2, 2.15)))
        cuts.append((2, occ.addRectangle(-5.4, 5.55 if s > 0 else -8.35, 0.0, 10.8, 2.8)))
        cuts.append((2, occ.addRectangle(-3.1, 4.1 if s > 0 else -5.55, 0.0, 6.2, 1.45)))
    res, _ = occ.cut([(2, outer)], cuts)
    if len(res) != 1:
        raise RuntimeError(f"Expected 1 connected 2020 face, got {len(res)}")
    return res[0][1]


def _extrude_x(face_tag: int, x0: float, x1: float) -> list[tuple[int, int]]:
    occ = gmsh.model.occ
    ext = occ.extrude([(2, face_tag)], 0.0, 0.0, x1 - x0)
    vols = [e for e in ext if e[0] == 3]
    occ.rotate(vols, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, math.pi / 2.0)
    occ.translate(vols, x0, 0.0, 0.0)
    return vols


def _make_rail_solid_front(
    x0: float,
    x1: float,
    y_back: float,
    rail_id: str,
) -> list[tuple[int, int]]:
    """Build MGN12 (12x8) or MGN9 (9x6.5) rail along X, extending outward along -Y from y_back."""
    occ = gmsh.model.occ
    if rail_id == "mgn9_rail":
        w, h, gr_h, gr_d = 9.0, 6.5, 2.0, 1.05
    else:
        w, h, gr_h, gr_d = 12.0, 8.0, 2.6, 1.35
    box = occ.addBox(x0, y_back - h, -0.5 * w, x1 - x0, h, w)
    g1 = occ.addBox(x0 - 0.5, y_back - 0.65 * h - 0.5 * gr_h, -0.5 * w - 0.1, x1 - x0 + 1.0, gr_h, gr_d + 0.1)
    g2 = occ.addBox(x0 - 0.5, y_back - 0.65 * h - 0.5 * gr_h, 0.5 * w - gr_d, x1 - x0 + 1.0, gr_h, gr_d + 0.1)
    res, _ = occ.cut([(3, box)], [(3, g1), (3, g2)])
    return res


def _make_cf_insert_block(x0: float, x1: float, id_mm: float) -> list[tuple[int, int]]:
    """Create chamfered square PET-CF internal insert block fitting cleanly inside rounded CF ID."""
    occ = gmsh.model.occ
    h = 0.5 * id_mm
    c = h - 1.0
    b1 = occ.addBox(x0, -h, -c, x1 - x0, 2.0 * h, 2.0 * c)
    b2 = occ.addBox(x0, -c, -h, x1 - x0, 2.0 * c, 2.0 * h)
    fused, _ = occ.fuse([(3, b1)], [(3, b2)])
    return fused


def _fragment_and_tag(group_lists: list[tuple[int, list[tuple[int, int]]]]) -> None:
    """Run occ.fragment across ordered (gid, vols) priority groups and create Gmsh Physical Groups."""
    occ = gmsh.model.occ
    flat_inputs: list[tuple[int, int]] = []
    counts: list[tuple[int, int]] = []
    for gid, vols in group_lists:
        if vols:
            counts.append((gid, len(vols)))
            flat_inputs.extend(vols)

    if len(flat_inputs) == 1:
        occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [flat_inputs[0][1]], counts[0][0])
        return

    _, out_map = occ.fragment(flat_inputs[:1], flat_inputs[1:])
    occ.synchronize()

    claimed: set[int] = set()
    off = 0
    for gid, cnt in counts:
        raw: set[int] = set()
        for k in range(off, off + cnt):
            for d, tg in out_map[k]:
                if d == 3:
                    raw.add(tg)
        off += cnt
        owned = sorted(raw - claimed)
        claimed |= set(owned)
        if owned:
            gmsh.model.addPhysicalGroup(3, owned, gid)


def _mesh_and_assemble(
    mats_by_gid: dict[int, np.ndarray],
    h_bulk_mm: float = 5.0,
    h_min_mm: float = 0.30,
    order: int = 2,
    curvature_pts: int = 0,
    refine_bolts: bool = False,
) -> tuple[skfem.MeshTet, skfem.CellBasis, Any, np.ndarray]:
    """Generate 3D tetrahedral mesh in Gmsh and assemble multi-material skfem stiffness matrix K."""
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h_min_mm)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max(h_bulk_mm * 1.35, 6.0))
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", curvature_pts)
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h_bulk_mm)
    if refine_bolts:
        ents_b = gmsh.model.getEntitiesForPhysicalGroup(3, 3)
        if len(ents_b) > 0:
            pts_b = gmsh.model.getBoundary([(3, e) for e in ents_b], combined=False, oriented=False, recursive=True)
            gmsh.model.mesh.setSize([p for p in pts_b if p[0] == 0], 1.6)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.model.mesh.generate(3)

    ntags, ncoords, _ = gmsh.model.mesh.getNodes()
    all_tets, all_gids = [], []
    for _, gid in gmsh.model.getPhysicalGroups(3):
        for ent in gmsh.model.getEntitiesForPhysicalGroup(3, gid):
            _, enodes = gmsh.model.mesh.getElementsByType(4, ent)
            if len(enodes) > 0:
                arr = np.asarray(enodes, dtype=np.int64).reshape(-1, 4)
                all_tets.append(arr)
                all_gids.append(np.full(len(arr), gid, dtype=np.int32))
    gmsh.finalize()

    raw_tets = np.vstack(all_tets)
    elem_gid = np.concatenate(all_gids)
    ntags = np.asarray(ntags, dtype=np.int64)
    coords_mm = np.asarray(ncoords, dtype=np.float64).reshape(-1, 3)
    order_idx = np.argsort(ntags)
    pos = np.searchsorted(ntags[order_idx], raw_tets.ravel())
    idx = order_idx[pos].reshape(-1, 4)
    used, inv = np.unique(idx, return_inverse=True)
    mesh = skfem.MeshTet((coords_mm[used] * 1e-3).T, inv.reshape(-1, 4).T).oriented()

    scalar_elem = skfem.ElementTetP2() if order == 2 else skfem.ElementTetP1()
    intorder = 2 if order == 2 else 1
    element = skfem.ElementVector(scalar_elem)
    basis_full = skfem.CellBasis(mesh, element, intorder=intorder)

    k_tot = None
    for gid, d_mat in mats_by_gid.items():
        elems = np.nonzero(elem_gid == gid)[0]
        if len(elems) == 0:
            continue
        nz_rows = [[(j, float(d_mat[i, j])) for j in range(6) if abs(d_mat[i, j]) > 0.0] for i in range(6)]

        @skfem.BilinearForm
        def elas(u, v, w):
            g_u, g_v = u.grad, v.grad
            eu = (g_u[0, 0], g_u[1, 1], g_u[2, 2], g_u[0, 1] + g_u[1, 0], g_u[1, 2] + g_u[2, 1], g_u[0, 2] + g_u[2, 0])
            ev = (g_v[0, 0], g_v[1, 1], g_v[2, 2], g_v[0, 1] + g_v[1, 0], g_v[1, 2] + g_v[2, 1], g_v[0, 2] + g_v[2, 0])
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

        k_sub = elas.assemble(skfem.CellBasis(mesh, element, elements=elems, intorder=intorder))
        k_tot = k_sub if k_tot is None else (k_tot + k_sub)

    return mesh, basis_full, k_tot, elem_gid


def _solve_factored(k_mat: Any, fixed_dofs: np.ndarray, rhs_list: list[np.ndarray]) -> list[np.ndarray]:
    dummy_f = np.zeros(k_mat.shape[0], dtype=np.float64)
    k_ff, _, _, idx_free = skfem.condense(k_mat, dummy_f, D=fixed_dofs)
    lu = splu(k_ff.tocsc(), permc_spec="COLAMD")
    sols = []
    for f_vec in rhs_list:
        u = np.zeros_like(f_vec)
        u[idx_free] = lu.solve(f_vec[idx_free])
        sols.append(u)
    return sols


# ---------------------------------------------------------------------------
# 3. Sub-model 1: Bare Beam Analytical Validation (<3% vs Timoshenko / Bredt-Batho)
# ---------------------------------------------------------------------------


def run_bare_validation(kind: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    gmsh.initialize(["gmsh"], interruptible=False)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(f"bare_{kind}")
    occ = gmsh.model.occ

    l_mm = 380.0
    if kind == "cf_carbonara":
        f_out = _make_rounded_rect_face(0.0, 0.0, 20.0, 20.0, 2.5)
        f_in = _make_rounded_rect_face(0.0, 0.0, 16.4, 16.4, 0.7)
        sec, _ = occ.cut([(2, f_out)], [(2, f_in)])
        d_mat, e_l, g_lt = cf_constitutive_at_temp("carbon_fiber_torsional_45", 25.0)
        prof = get_profile("cf_tube_20x20_carbonara")
        a_sy = 2.0 * (prof.width_m - prof.wall_thickness_m) * prof.wall_thickness_m
        kappa = min(0.85, 0.833 * a_sy / prof.area_m2)
        h_b, curv = 3.2, 0
    elif kind == "al_2020":
        sec = [(2, _make_2020_tslot_face())]
        d_mat = iso_constitutive_3d(68.9e9, 0.33)
        e_l, g_lt = 68.9e9, 68.9e9 / 2.66
        prof = get_profile("extrusion_2020")
        kappa = 0.30  # Open T-slot web Timoshenko shear factor
        h_b, curv = 4.2, 0
    else:
        raise ValueError(kind)

    v_r = _extrude_x(sec[0][1], 0.0, l_mm / 2.0)
    v_l = occ.copy(v_r)
    occ.translate(v_l, -l_mm / 2.0, 0.0, 0.0)
    _fragment_and_tag([(1, v_r + v_l)])

    mesh, basis, k_mat, _ = _mesh_and_assemble({1: d_mat}, h_bulk_mm=h_b, h_min_mm=0.6, order=2, curvature_pts=curv)

    l_m = 0.380
    tol = 1e-5
    dl = basis.doflocs
    fix_bend = np.unique(
        np.concatenate(
            [
                np.nonzero((np.abs(np.abs(dl[0]) - l_m / 2.0) < tol) & ((np.arange(basis.N) % 3) != 0))[0],
                np.nonzero((np.abs(dl[0]) < tol) & ((np.arange(basis.N) % 3) == 0))[0],
            ]
        )
    )
    f_mid = np.nonzero(np.all(np.abs(mesh.p[0, mesh.facets]) < tol, axis=0))[0]
    fb_mid = skfem.FacetBasis(mesh, skfem.ElementVector(basis.elem.elem), facets=f_mid, intorder=2)
    area_mid = float(np.sum(fb_mid.dx))
    f_val = 20.0

    @skfem.LinearForm
    def form_fy(v, w):
        return (f_val / area_mid) * v[1]

    @skfem.LinearForm
    def form_fz(v, w):
        return (f_val / area_mid) * v[2]

    u_y, u_z = _solve_factored(k_mat, fix_bend, [form_fy.assemble(fb_mid), form_fz.assemble(fb_mid)])
    mid_verts = np.nonzero(np.abs(mesh.p[0]) < tol)[0]
    dy_3d = float(np.mean(u_y[basis.nodal_dofs[1, mid_verts]]))
    dz_3d = float(np.mean(u_z[basis.nodal_dofs[2, mid_verts]]))

    # Torsion (St. Venant free warping on open 2020; clamped left end on closed square CF)
    if kind == "al_2020":
        fix_tor_yz = np.nonzero((np.abs(dl[0] + l_m / 2.0) < tol) & ((np.arange(basis.N) % 3) != 0))[0]
        fix_tor_x1 = np.nonzero((np.abs(dl[0] + l_m / 2.0) < tol) & ((np.arange(basis.N) % 3) == 0))[0][:1]
        fix_tor = np.unique(np.concatenate([fix_tor_yz, fix_tor_x1]))
    else:
        fix_tor = np.nonzero(np.abs(dl[0] + l_m / 2.0) < tol)[0]

    f_right = np.nonzero(np.all(np.abs(mesh.p[0, mesh.facets] - l_m / 2.0) < tol, axis=0))[0]
    fb_right = skfem.FacetBasis(mesh, skfem.ElementVector(basis.elem.elem), facets=f_right, intorder=2)

    @skfem.Functional
    def ip_fn(w):
        return w.x[1] ** 2 + w.x[2] ** 2

    ip_end = float(ip_fn.assemble(fb_right))
    t_x = 10.0

    @skfem.LinearForm
    def form_tor(v, w):
        return (t_x / ip_end) * (-w.x[2] * v[1] + w.x[1] * v[2])

    (u_tor,) = _solve_factored(k_mat, fix_tor, [form_tor.assemble(fb_right)])
    right_verts = np.nonzero(np.abs(mesh.p[0] - l_m / 2.0) < tol)[0]
    yr, zr = mesh.p[1, right_verts], mesh.p[2, right_verts]
    uyr, uzr = u_tor[basis.nodal_dofs[1, right_verts]], u_tor[basis.nodal_dofs[2, right_verts]]
    theta_r = float(np.sum(yr * uzr - zr * uyr) / np.sum(yr**2 + zr**2))
    gj_3d = t_x * l_m / theta_r

    eb_y = f_val * (l_m**3) / (48.0 * e_l * prof.iz_m4)
    eb_z = f_val * (l_m**3) / (48.0 * e_l * prof.iy_m4)
    sh = f_val * l_m / (4.0 * kappa * g_lt * prof.area_m2) if kappa > 0.0 else 0.0
    timo_y, timo_z = eb_y + sh, eb_z + sh
    gj_ana = g_lt * prof.j_m4

    return {
        "kind": kind,
        "tets": int(mesh.nelements),
        "dofs": int(basis.N),
        "area_mesh_mm2": round(area_mid * 1e6, 2),
        "dy_3d_um": round(dy_3d * 1e6, 3),
        "dy_ana_um": round(timo_y * 1e6, 3),
        "err_y_pct": round((dy_3d / timo_y - 1.0) * 100.0, 2),
        "dz_3d_um": round(dz_3d * 1e6, 3),
        "dz_ana_um": round(timo_z * 1e6, 3),
        "err_z_pct": round((dz_3d / timo_z - 1.0) * 100.0, 2),
        "gj_3d": round(gj_3d, 2),
        "gj_ana": round(gj_ana, 2),
        "err_gj_pct": round((gj_3d / gj_ana - 1.0) * 100.0, 2),
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }


# ---------------------------------------------------------------------------
# 4. Shared Multi-Material Bolted Assembly Builder (Sub-models 2, 3, 4, 6)
# ---------------------------------------------------------------------------


@dataclass
class BeamRailSpec:
    label: str
    substrate: str  # "cf" | "al_2020"
    cf_profile_id: str = "cf_tube_20x20_carbonara"
    cf_material_id: str = "carbon_fiber_torsional_45"
    rail_id: str = "mgn12_rail"
    cf_mount: str = "two_wall_clamped_inserts"  # "single_wall_bare" | "two_wall_clamped_inserts"
    pitch_mm: float = 25.0
    temp_c: float = 25.0
    e_pad_gpa: float | None = None  # None -> auto physical contact regime
    with_backer: bool = False
    end_clamp_mode: str = "diaphragm"  # "diaphragm" | "m5_footprints_only"
    with_end_inserts: bool = True
    h_bulk_mm: float = 5.0
    h_min_mm: float = 0.35
    order: int = 2
    solve_torsion: bool = False
    solve_local_seat: bool = False
    carriage_len_mm: float = 32.4


def _build_beam_rail_model(cfg: BeamRailSpec) -> tuple[skfem.MeshTet, skfem.CellBasis, Any, np.ndarray, dict[str, Any]]:
    gmsh.initialize(["gmsh"], interruptible=False)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(cfg.label)
    occ = gmsh.model.occ

    l_mm = 380.0
    gap = 0.25  # physical air gap between rail/backer and beam outside discrete bolt zones

    # 1. Beam substrate
    if cfg.substrate == "al_2020":
        f_beam = _make_2020_tslot_face()
        beam_vols = _extrude_x(f_beam, -l_mm / 2.0, l_mm / 2.0)
        d_beam = iso_constitutive_3d(68.9e9, 0.33)
        e_l, g_lt = 68.9e9, 68.9e9 / 2.66
        beam_prof = get_profile("extrusion_2020")
        kappa = 0.30
        d_bolt = iso_constitutive_3d(205.0e9, 0.29)  # steel M3 bolts & T-nuts
        t_wall_mm = 1.7
    else:
        beam_prof = get_profile(cfg.cf_profile_id)
        t_wall_mm = (beam_prof.wall_thickness_m or 0.0018) * 1e3
        r_out = max(beam_prof.corner_radius_m * 1e3, 1.5)
        id_mm = 20.0 - 2.0 * t_wall_mm
        r_in = max(r_out - t_wall_mm, 0.35)
        f_out = _make_rounded_rect_face(0.0, 0.0, 20.0, 20.0, r_out)
        f_in = _make_rounded_rect_face(0.0, 0.0, id_mm, id_mm, r_in)
        sec, _ = occ.cut([(2, f_out)], [(2, f_in)])
        beam_vols = _extrude_x(sec[0][1], -l_mm / 2.0, l_mm / 2.0)
        d_beam, e_l, g_lt = cf_constitutive_at_temp(cfg.cf_material_id, cfg.temp_c)
        a_sy = 2.0 * beam_prof.width_m * (t_wall_mm * 1e-3)
        kappa = min(0.85, 0.87 * a_sy / beam_prof.area_m2)
        d_bolt = iso_constitutive_3d(114.0e9, 0.34)  # Ti Gr5 M3 bolts

    rail_prof = get_profile(cfg.rail_id)
    rail_h = rail_prof.height_m * 1e3
    rail_w = rail_prof.width_m * 1e3

    # 2. Front steel rail (350 mm centered)
    rail_vols = _make_rail_solid_front(-175.0, 175.0, -10.0 - gap, cfg.rail_id)

    # 3. Optional opposite-face 3 mm Steel 4140 backer bar (12 x 3 x 350 mm at y = +10 + gap)
    backer_vols: list[tuple[int, int]] = []
    if cfg.with_backer:
        bk_box = occ.addBox(-175.0, 10.0 + gap, -6.0, 350.0, 3.0, 12.0)
        backer_vols.append((3, bk_box))

    # 4. Bolts, pads, internal inserts, end inserts & M5 Al standoffs
    xs_bolts = np.arange(-162.5, 162.6, cfg.pitch_mm)
    bolt_vols: list[tuple[int, int]] = []
    pad_vols: list[tuple[int, int]] = []
    insert_vols: list[tuple[int, int]] = []
    standoff_vols: list[tuple[int, int]] = []
    carr_vols: list[tuple[int, int]] = []
    bear_vols: list[tuple[int, int]] = []

    is_two_wall = cfg.substrate == "cf" and cfg.cf_mount == "two_wall_clamped_inserts"
    pet_cf_ret = get_material("pet_cf_printed").modulus_retention_at_temp(cfg.temp_c)
    # Choose realistic Mindlin interfacial contact regime when e_pad_gpa is None:
    # - 2020 Al & 2-wall clamped CF: full ~2 kN M3 bolt preload -> k_t,area ~ 0.75e12 N/m^3 -> E_pad = 0.50 GPa across 0.25 mm gap
    # - At elevated chamber temp, Shigley bolted-stack compliance (k_bolt=19.8 MN/m vs PET-CF stack k_stack=12.3*pet_cf_ret)
    #   reduces preload P(T)/P_0, while Mindlin tangential stiffness k_t ~ G_surface(T) * (P(T)/P_0)^(1/3).
    # - Single-wall bare CF: un-inserted thin wall crushes/relaxes -> bolt-bearing only (E_pad = 0)
    if cfg.e_pad_gpa is None:
        if cfg.substrate == "al_2020":
            eff_pad_gpa = 0.50
        elif is_two_wall:
            r_g_cf = get_material(cfg.cf_material_id).shear_modulus_retention_at_temp(cfg.temp_c)
            r_preload = (0.050 + 0.081) / (0.050 + 0.081 / max(pet_cf_ret, 0.05))
            eff_pad_gpa = 0.50 * r_g_cf * (r_preload ** (1.0 / 3.0))
        else:
            eff_pad_gpa = 0.0
    else:
        eff_pad_gpa = cfg.e_pad_gpa

    for xb in xs_bolts:
        xb_f = float(xb)
        # M3 counterbore Ø6.0 + clearance hole Ø3.8 (r=1.9 vs bolt shaft r=1.5 -> 0.4 mm air clearance!)
        cb = occ.addCylinder(xb_f, -10.0 - gap - rail_h - 0.1, 0.0, 0.0, 3.6, 0.0, 3.0)
        thru = occ.addCylinder(xb_f, -10.0 - gap - rail_h + 3.4, 0.0, 0.0, rail_h - 3.2, 0.0, 1.9)
        rail_vols, _ = occ.cut(rail_vols, [(3, cb), (3, thru)])

        y_head = -10.0 - gap - rail_h + 3.5
        head = occ.addCylinder(xb_f, y_head - 1.5, 0.0, 0.0, 1.5, 0.0, 3.0)

        if cfg.substrate == "al_2020":
            shaft = occ.addCylinder(xb_f, y_head, 0.0, 0.0, (10.0 + gap + rail_h - 3.5) - 7.0, 0.0, 1.5)
            tnut = occ.addBox(xb_f - 3.5, -8.35, -5.3, 7.0, 2.5, 10.6)
            b_fused, _ = occ.fuse([(3, head)], [(3, shaft), (3, tnut)])
            bolt_vols.extend(b_fused)
            if eff_pad_gpa > 0.0:
                if cfg.rail_id == "mgn9_rail":
                    # MGN9 (9 mm wide) contacts two 1.4 mm shoulders z in [-4.5, -3.1] and [3.1, 4.5]
                    p1 = occ.addBox(xb_f - 3.5, -10.0 - gap, -4.5, 7.0, gap, 1.4)
                    p2 = occ.addBox(xb_f - 3.5, -10.0 - gap, 3.1, 7.0, gap, 1.4)
                else:
                    p1 = occ.addBox(xb_f - 3.5, -10.0 - gap, -6.0, 7.0, gap, 2.9)
                    p2 = occ.addBox(xb_f - 3.5, -10.0 - gap, 3.1, 7.0, gap, 2.9)
                pad_vols.extend([(3, p1), (3, p2)])
            if cfg.with_backer:
                # Rear M3 bolt + T-nut clamping 3 mm steel backer
                bk_hole = occ.addCylinder(xb_f, 9.9, 0.0, 0.0, 3.3 + gap, 0.0, 1.9)
                backer_vols, _ = occ.cut(backer_vols, [(3, bk_hole)])
                b_head = occ.addCylinder(xb_f, 13.0 + gap, 0.0, 0.0, 1.5, 0.0, 3.0)
                b_sh = occ.addCylinder(xb_f, 7.0, 0.0, 0.0, 6.0 + gap, 0.0, 1.5)
                b_tn = occ.addBox(xb_f - 3.5, 5.85, -5.3, 7.0, 2.5, 10.6)
                bk_b, _ = occ.fuse([(3, b_head)], [(3, b_sh), (3, b_tn)])
                bolt_vols.extend(bk_b)
                if eff_pad_gpa > 0.0:
                    pad_vols.append((3, occ.addBox(xb_f - 3.5, 10.0, -6.0, 7.0, gap, 2.9)))
                    pad_vols.append((3, occ.addBox(xb_f - 3.5, 10.0, 3.1, 7.0, gap, 2.9)))
        elif is_two_wall:
            y_end = 13.0 + gap if cfg.with_backer else 10.0
            if cfg.with_backer:
                bk_hole = occ.addCylinder(xb_f, 9.9, 0.0, 0.0, 3.3 + gap, 0.0, 1.9)
                backer_vols, _ = occ.cut(backer_vols, [(3, bk_hole)])
            shaft = occ.addCylinder(xb_f, y_head, 0.0, 0.0, y_end - y_head, 0.0, 1.5)
            nut = occ.addCylinder(xb_f, y_end, 0.0, 0.0, 1.8, 0.0, 3.5)
            b_fused, _ = occ.fuse([(3, head)], [(3, shaft), (3, nut)])
            bolt_vols.extend(b_fused)
            insert_vols.extend(_make_cf_insert_block(xb_f - 3.5, xb_f + 3.5, id_mm))
            if eff_pad_gpa > 0.0:
                pad_vols.append((3, occ.addCylinder(xb_f, -10.0 - gap, 0.0, 0.0, gap, 0.0, 4.0)))
                if cfg.with_backer:
                    pad_vols.append((3, occ.addCylinder(xb_f, 10.0, 0.0, 0.0, gap, 0.0, 4.0)))
        else:
            # Single-wall bare CF: M3x8 bolt into front wall only
            y_in_front = -10.0 + t_wall_mm
            shaft = occ.addCylinder(xb_f, y_head, 0.0, 0.0, y_in_front - y_head, 0.0, 1.5)
            nut = occ.addCylinder(xb_f, y_in_front, 0.0, 0.0, 1.6, 0.0, 3.5)
            b_fused, _ = occ.fuse([(3, head)], [(3, shaft), (3, nut)])
            bolt_vols.extend(b_fused)
            if eff_pad_gpa > 0.0:
                pad_vols.append((3, occ.addCylinder(xb_f, -10.0 - gap, 0.0, 0.0, gap, 0.0, 3.5)))

    # 5. Carbonara M5 End Inserts (31 mm at each end) + Al 6061 M5x15 hex standoff (Ø9.2 mm vertical along Z)
    if cfg.substrate == "cf" and (is_two_wall or cfg.end_clamp_mode == "m5_footprints_only") and cfg.with_end_inserts:
        for sign_x in (-1.0, 1.0):
            ex0, ex1 = (-190.0, -166.5) if sign_x < 0 else (166.5, 190.0)
            insert_vols.extend(_make_cf_insert_block(ex0, ex1, id_mm))
            # M5x15 Al 6061 standoff centered at x = +-178 mm, bridging top/bottom walls along Z
            standoff_vols.append((3, occ.addCylinder(sign_x * 178.0, 0.0, -0.5 * id_mm, 0.0, 0.0, id_mm, 4.6)))

    # 6. Optional local carriage platen + normal-bearing seat strip for Sub-model 2E/3B
    if cfg.solve_local_seat:
        c_half = 0.5 * cfg.carriage_len_mm
        y_carr_back = -10.0 - gap - rail_h
        carr_box = occ.addBox(-c_half, y_carr_back - 4.0, -0.5 * rail_w, cfg.carriage_len_mm, 4.0, rail_w)
        carr_vols.append((3, carr_box))
        # Normal-only bearing contact strip under the carriage footprint (excludes existing bolt pads)
        if cfg.substrate == "al_2020":
            if cfg.rail_id == "mgn9_rail":
                b1 = occ.addBox(-c_half, -10.0 - gap, -4.5, cfg.carriage_len_mm, gap, 1.4)
                b2 = occ.addBox(-c_half, -10.0 - gap, 3.1, cfg.carriage_len_mm, gap, 1.4)
            else:
                b1 = occ.addBox(-c_half, -10.0 - gap, -6.0, cfg.carriage_len_mm, gap, 2.9)
                b2 = occ.addBox(-c_half, -10.0 - gap, 3.1, cfg.carriage_len_mm, gap, 2.9)
            bear_vols.extend([(3, b1), (3, b2)])
        else:
            bear_vols.append((3, occ.addBox(-c_half, -10.0 - gap, -0.5 * rail_w, cfg.carriage_len_mm, gap, rail_w)))

    # Priority order: carriage > bolt > standoff > pad > bearing > rail > backer > insert > beam
    _fragment_and_tag(
        [
            (6, carr_vols),
            (3, bolt_vols),
            (7, standoff_vols),
            (4, pad_vols),
            (8, bear_vols),
            (2, rail_vols),
            (9, backer_vols),
            (5, insert_vols),
            (1, beam_vols),
        ]
    )

    e_insert = 5.5e9 * pet_cf_ret
    mats = {
        1: d_beam,
        2: iso_constitutive_3d(205.0e9, 0.29),
        3: d_bolt,
        4: mindlin_pad_constitutive_3d(max(eff_pad_gpa, 1e-4) / 2.64, e_norm_gpa=80.0),
        5: iso_constitutive_3d(e_insert, 0.35),
        6: iso_constitutive_3d(2000.0e9, 0.30),  # rigid steel carriage platen
        7: iso_constitutive_3d(68.9e9, 0.33),  # Al 6061 M5 standoff
        8: normal_bearing_constitutive_3d(100.0e9, axis="y"),
        9: iso_constitutive_3d(205.0e9, 0.29),  # Steel 4140 backer
    }
    mesh, basis, k_mat, elem_gid = _mesh_and_assemble(
        mats, h_bulk_mm=cfg.h_bulk_mm, h_min_mm=cfg.h_min_mm, order=cfg.order, refine_bolts=cfg.h_bulk_mm >= 5.5
    )
    meta = {
        "e_l": e_l,
        "g_lt": g_lt,
        "beam_prof": beam_prof,
        "rail_prof": rail_prof,
        "kappa": kappa,
        "gap_m": gap * 1e-3,
        "eff_pad_gpa": eff_pad_gpa,
    }
    return mesh, basis, k_mat, elem_gid, meta


def run_beam_rail_case(cfg: BeamRailSpec) -> dict[str, Any]:
    t0 = time.perf_counter()
    mesh, basis, k_mat, elem_gid, meta = _build_beam_rail_model(cfg)
    dl = basis.doflocs
    tol = 1e-4
    l_m = 0.380

    # End boundary conditions
    if cfg.end_clamp_mode == "m5_footprints_only":
        # Hold ONLY the 4 outer M5 washer footprints (r=5 mm at x=+-178 mm, z=+-10 mm)
        m5_mask = (np.abs(np.abs(dl[0]) - 0.178) < 0.005) & (np.abs(dl[1]) < 0.005) & (np.abs(np.abs(dl[2]) - 0.010) < tol)
        fix_dofs = np.nonzero(m5_mask)[0]
    else:
        # Diaphragm simply-supported on tube ends (u_y=u_z=0 on beam ends; rail/backer axial ends FREE!)
        beam_dofs = np.unique(basis.element_dofs[:, elem_gid == 1])
        is_beam = np.zeros(basis.N, dtype=bool)
        is_beam[beam_dofs] = True
        fix_ends = np.nonzero(is_beam & (np.abs(np.abs(dl[0]) - 0.190) < tol) & ((np.arange(basis.N) % 3) != 0))[0]
        fix_ux = np.nonzero(is_beam & (np.abs(dl[0]) < 0.006) & (np.abs(dl[1]) < 0.002) & ((np.arange(basis.N) % 3) == 0))[0][:2]
        fix_dofs = np.unique(np.concatenate([fix_ends, fix_ux]))

    # Identify tube neutral-web vertices at mid-span (|x| <= 5 mm, |y| <= 3 mm or top/bot walls)
    tube_verts = np.unique(mesh.t[:, elem_gid == 1])
    mid_webs_z = tube_verts[(np.abs(mesh.p[0, tube_verts]) <= 0.005) & (np.abs(np.abs(mesh.p[2, tube_verts]) - 0.010) < 0.002)]
    mid_webs_y = tube_verts[(np.abs(mesh.p[0, tube_verts]) <= 0.005) & (np.abs(np.abs(mesh.p[1, tube_verts]) - 0.010) < 0.002)]
    mid_all_tube = tube_verts[np.abs(mesh.p[0, tube_verts]) <= 0.005]

    out: dict[str, Any] = {
        "label": cfg.label,
        "tets": int(mesh.nelements),
        "dofs": int(basis.N),
        "e_pad_gpa": meta["eff_pad_gpa"],
    }

    # Global 3-pt bending F_y and F_z applied cleanly onto tube webs (zero local pad denting)
    beam_dofs_arr = np.unique(basis.element_dofs[:, elem_gid == 1])
    is_beam_d = np.zeros(basis.N, dtype=bool)
    is_beam_d[beam_dofs_arr] = True

    load_y_dofs = np.nonzero(is_beam_d & (np.abs(dl[0]) <= 0.004) & (np.abs(np.abs(dl[2]) - 0.010) < tol) & ((np.arange(basis.N) % 3) == 1))[0]
    f_y = basis.zeros()
    f_y[load_y_dofs] = -20.0 / len(load_y_dofs)

    load_z_dofs = np.nonzero(is_beam_d & (np.abs(dl[0]) <= 0.004) & (np.abs(np.abs(dl[1]) - 0.010) < tol) & ((np.arange(basis.N) % 3) == 2))[0]
    f_z = basis.zeros()
    f_z[load_z_dofs] = -20.0 / len(load_z_dofs)

    # 4-point pure-bending load (-10 N at x=-100 mm, -10 N at x=+100 mm) for shear-free EI_y
    f_4pt = basis.zeros()
    for xs_4 in (-0.100, 0.100):
        d4 = np.nonzero(is_beam_d & (np.abs(dl[0] - xs_4) <= 0.004) & (np.abs(np.abs(dl[2]) - 0.010) < tol) & ((np.arange(basis.N) % 3) == 1))[0]
        if len(d4) > 0:
            f_4pt[d4] = -10.0 / len(d4)

    rhs_list = [f_y, f_z, f_4pt]

    # Mid-span torque M_x (for end-clamping ratio test)
    if cfg.end_clamp_mode == "m5_footprints_only":
        mid_ring = np.nonzero(is_beam_d & (np.abs(dl[0]) <= 0.005))[0]
        # Torque couple on top/bottom/front/back mid-span ring
        f_tor_mid = basis.zeros()
        y_dofs = mid_ring[(mid_ring % 3) == 1]
        z_dofs = mid_ring[(mid_ring % 3) == 2]
        denom = float(np.sum(dl[2, y_dofs] ** 2) + np.sum(dl[1, z_dofs] ** 2))
        t_x_mid = 10.0
        f_tor_mid[y_dofs] = -(t_x_mid / denom) * dl[2, y_dofs]
        f_tor_mid[z_dofs] = +(t_x_mid / denom) * dl[1, z_dofs]
        rhs_list.append(f_tor_mid)

    # Local carriage-seat loads (F_y, M_x, M_z pitch) on rigid carriage platen (gid=6)
    if cfg.solve_local_seat:
        carr_dofs = np.unique(basis.element_dofs[:, elem_gid == 6])
        is_carr = np.zeros(basis.N, dtype=bool)
        is_carr[carr_dofs] = True
        y_front = np.min(dl[1, carr_dofs])
        face_mask = is_carr & (np.abs(dl[1] - y_front) < tol)
        fy_pad_dofs = np.nonzero(face_mask & ((np.arange(basis.N) % 3) == 1))[0]
        fx_pad_dofs = np.nonzero(face_mask & ((np.arange(basis.N) % 3) == 0))[0]

        f_pad_rad = basis.zeros()
        f_pad_rad[fy_pad_dofs] = 50.0 / len(fy_pad_dofs)  # push INTO front wall (+Y)

        # Roll M_x = 5.0 N*m via z-couple of F_y
        f_pad_roll = basis.zeros()
        zc = dl[2, fy_pad_dofs] - np.mean(dl[2, fy_pad_dofs])
        f_pad_roll[fy_pad_dofs] = (5.0 / np.sum(zc**2)) * zc

        # Pitch M_z = 5.0 N*m via x-couple of F_y
        f_pad_pitch = basis.zeros()
        xc = dl[0, fy_pad_dofs] - np.mean(dl[0, fy_pad_dofs])
        f_pad_pitch[fy_pad_dofs] = (5.0 / np.sum(xc**2)) * xc

        rhs_list.extend([f_pad_rad, f_pad_roll, f_pad_pitch])

    sols = _solve_factored(k_mat, fix_dofs, rhs_list)
    u_fy, u_fz, u_4pt = sols[0], sols[1], sols[2]

    uy_mid = float(np.mean(u_fy[basis.nodal_dofs[1, mid_webs_z]]))
    uz_mid = float(np.mean(u_fz[basis.nodal_dofs[2, mid_webs_y]]))
    out["dy_um"] = round(abs(uy_mid) * 1e6, 3)
    out["dz_um"] = round(abs(uz_mid) * 1e6, 3)

    # Pure shear-free 4-point bending sagitta inside [-75, +75 mm] (M_0 = 10 N * 0.090 m = 0.90 N*m)
    w_q = tube_verts[(np.abs(np.abs(mesh.p[0, tube_verts]) - 0.075) <= 0.004) & (np.abs(np.abs(mesh.p[2, tube_verts]) - 0.010) < 0.002)]
    uy_4_mid = float(np.mean(u_4pt[basis.nodal_dofs[1, mid_webs_z]]))
    uy_4_q = float(np.mean(u_4pt[basis.nodal_dofs[1, w_q]]))
    sag_4pt = abs(uy_4_mid - uy_4_q)
    m0_4pt = 10.0 * (0.190 - 0.100)
    ei_y_4pt = m0_4pt * (0.075**2) / (2.0 * max(sag_4pt, 1e-12))
    out["ei_y_4pt_nm2"] = round(ei_y_4pt, 1)

    # Extract composite EI_y, EI_z, gamma, k_s using Timoshenko composite shear:
    # Steel rail shares parallel shear with CF when matrix shear drops at elevated temp.
    e_l = meta["e_l"]
    g_lt = meta["g_lt"]
    b_prof = meta["beam_prof"]
    r_prof = meta["rail_prof"]
    kappa = meta["kappa"]
    bend_3pt = 20.0 * (l_m**3) / (48.0 * max(ei_y_4pt, 1.0))
    sh_3pt_est = max(0.0, abs(uy_mid) - bend_3pt)
    if sh_3pt_est > 1e-12:
        out["ga_comp_mn"] = round((20.0 * l_m / (4.0 * sh_3pt_est)) * 1e-6, 3)

    # For 3-pt PrintSim comparison, evaluate EI_y using PrintSim composite EI_y_4pt AND 3-pt total
    sh_z_bare = 20.0 * l_m / (4.0 * kappa * g_lt * b_prof.area_m2) if kappa > 0 else 0.0
    ei_y_comp = ei_y_4pt
    ei_z_comp = 20.0 * (l_m**3) / (48.0 * max(abs(uz_mid) - min(sh_z_bare, 0.45 * abs(uz_mid)), 1e-12))
    out["ei_y_nm2"] = round(ei_y_comp, 1)
    out["ei_z_nm2"] = round(ei_z_comp, 1)

    if not cfg.with_backer and cfg.end_clamp_mode == "diaphragm":
        i_y_comp = ei_y_comp / e_l
        n_e = 205.0e9 / e_l
        d_c = 0.5 * b_prof.width_m + meta["gap_m"] + 0.5 * r_prof.height_m
        i_par_max = ((b_prof.area_m2 * (n_e * r_prof.area_m2)) / (b_prof.area_m2 + n_e * r_prof.area_m2)) * (d_c**2)
        gamma = (i_y_comp - b_prof.iz_m4 - n_e * r_prof.iy_m4) / i_par_max
        ea_star = (e_l * b_prof.area_m2 * 205.0e9 * r_prof.area_m2) / (e_l * b_prof.area_m2 + 205.0e9 * r_prof.area_m2)
        k_s = (math.pi**2 * ea_star * (cfg.pitch_mm * 1e-3)) / ((l_m**2) * (1.0 / max(min(gamma, 0.9999), 1e-4) - 1.0))
        out["gamma_3d"] = round(gamma, 4)
        out["k_s_mn_m"] = round(k_s * 1e-6, 2)

    idx_s = 3
    if cfg.end_clamp_mode == "m5_footprints_only":
        u_t = sols[idx_s]; idx_s += 1
        ym, zm = mesh.p[1, mid_all_tube], mesh.p[2, mid_all_tube]
        uym, uzm = u_t[basis.nodal_dofs[1, mid_all_tube]], u_t[basis.nodal_dofs[2, mid_all_tube]]
        theta_mid = float(np.sum(ym * uzm - zm * uym) / np.sum(ym**2 + zm**2))
        out["theta_mid_mrad"] = round(abs(theta_mid) * 1e3, 4)

    if cfg.solve_local_seat:
        _, u_roll, _ = sols[idx_s], sols[idx_s + 1], sols[idx_s + 2]
        carr_verts = np.unique(mesh.t[:, elem_gid == 6])
        carr_front_v = carr_verts[np.abs(mesh.p[1, carr_verts] - np.min(mesh.p[1, carr_verts])) < 1e-4]

        # Pure local seat compliance: clamp rear corners (y >= 8 mm, |z| >= 6 mm) so global bending,
        # anticlastic Poisson expansion & global Saint-Venant twist/warping are identically zero.
        fix_local = np.nonzero(is_beam_d & (dl[1] >= 0.008) & (np.abs(dl[2]) >= 0.006))[0]
        u_rad_loc, u_rol_loc, u_pit_loc = _solve_factored(
            k_mat, fix_local, [rhs_list[idx_s], rhs_list[idx_s + 1], rhs_list[idx_s + 2]]
        )

        # 1. Local radial seat stiffness (N/um)
        uy_pad_pure = float(np.mean(u_rad_loc[basis.nodal_dofs[1, carr_front_v]]))
        out["k_wall_rad_n_um"] = round((50.0 / max(abs(uy_pad_pure), 1e-12)) * 1e-6, 2)

        # 2. Local roll stiffness (N*m/rad) under fixed rear-corner BC AND consistent-sign full-beam sub
        zc_p = mesh.p[2, carr_front_v] - np.mean(mesh.p[2, carr_front_v])
        th_x_loc = float(-np.sum(zc_p * u_rol_loc[basis.nodal_dofs[1, carr_front_v]]) / np.sum(zc_p**2))
        out["k_wall_roll_nm_rad"] = round(5.0 / max(abs(th_x_loc), 1e-12), 1)

        th_x_pad = float(-np.sum(zc_p * u_roll[basis.nodal_dofs[1, carr_front_v]]) / np.sum(zc_p**2))
        back_side_v = mid_all_tube[(mesh.p[1, mid_all_tube] >= 0.007) & (np.abs(mesh.p[2, mid_all_tube]) >= 0.006)]
        yb, zb = mesh.p[1, back_side_v], mesh.p[2, back_side_v]
        uyb, uzb = u_roll[basis.nodal_dofs[1, back_side_v]], u_roll[basis.nodal_dofs[2, back_side_v]]
        th_x_beam = float(np.sum(yb * uzb - zb * uyb) / np.sum(yb**2 + zb**2))
        out["k_wall_roll_sub_nm_rad"] = round(5.0 / max(abs(th_x_pad - th_x_beam), 1e-12), 1)

        # 3. Local pitch stiffness (N*m/rad)
        xc_p = mesh.p[0, carr_front_v] - np.mean(mesh.p[0, carr_front_v])
        th_z_loc = float(np.sum(xc_p * u_pit_loc[basis.nodal_dofs[1, carr_front_v]]) / np.sum(xc_p**2))
        out["k_wall_pitch_nm_rad"] = round(5.0 / max(abs(th_z_loc), 1e-12), 1)

    if cfg.solve_torsion:
        # Global end-to-end torsion GJ (clamp x=-190, torque on tube+rail at x=+190 or tube end)
        fix_t = np.nonzero(np.abs(dl[0] + 0.190) < tol)[0]
        f_right = np.nonzero(np.all(np.abs(mesh.p[0, mesh.facets] - 0.190) < tol, axis=0))[0]
        fb_r = skfem.FacetBasis(mesh, skfem.ElementVector(basis.elem.elem), facets=f_right, intorder=2)

        @skfem.Functional
        def ip_f(w):
            return w.x[1] ** 2 + w.x[2] ** 2

        ip_e = float(ip_f.assemble(fb_r))

        @skfem.LinearForm
        def t_f(v, w):
            return (10.0 / ip_e) * (-w.x[2] * v[1] + w.x[1] * v[2])

        (u_t,) = _solve_factored(k_mat, fix_t, [t_f.assemble(fb_r)])
        rv = np.nonzero(np.abs(mesh.p[0] - 0.190) < tol)[0]
        yr, zr = mesh.p[1, rv], mesh.p[2, rv]
        uyr, uzr = u_t[basis.nodal_dofs[1, rv]], u_t[basis.nodal_dofs[2, rv]]
        theta_r = float(np.sum(yr * uzr - zr * uyr) / np.sum(yr**2 + zr**2))
        out["gj_nm2"] = round(10.0 * l_m / theta_r, 2)

    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    return out


# ---------------------------------------------------------------------------
# 5. Sub-model 5: Dual Top+Bottom MGN12H + 5 mm Al 6061 Shuttle Steiner Coupling
# ---------------------------------------------------------------------------


def run_dual_rail_steiner_case(substrate: str) -> dict[str, Any]:
    """Extract 3D Steiner roll coupling efficiency for 5 mm Al 6061 wrap-around shuttle on dual MGN12H."""
    t0 = time.perf_counter()
    gap = 0.25
    l_mm = 380.0
    c_len = 32.4

    # Step A: Calibrate effective 3D raceway-track modulus on a RIGID rail base to match HIWIN MGN12H Z1
    car_cat = hertzian_rail_carriage_stiffness("mgn12_rail", preload="Z1", block_type="H")
    k_yy_target = car_cat["k_yy_n_m"]  # ~123.5 MN/m
    k_rx_target = car_cat["k_rx_nm_rad"]  # ~3540 N*m/rad

    clr = 0.50  # explicit radial/top clearance so carriage connects ONLY via ball raceways

    def _race_constitutive(ey_gpa: float, gz_gpa: float) -> np.ndarray:
        ey = ey_gpa * 1e9
        gz = gz_gpa * 1e9
        s_mat = np.diag([1.0 / (0.5 * ey), 1.0 / ey, 1.0 / gz, 1.0 / (0.4 * ey), 1.0 / (0.4 * gz), 1.0 / (0.4 * gz)])
        return np.linalg.inv(s_mat)

    def _single_rail_calib(ey_gpa: float, gz_gpa: float, rigid_block: bool = True) -> tuple[float, float]:
        gmsh.initialize(["gmsh"], interruptible=False)
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("single_race_cal")
        occ = gmsh.model.occ
        # Short 50 mm rail on rigid bed + real 27x10x32.4 mm MGN12H steel carriage + 2 raceway strips
        r_box = occ.addBox(-25.0, -6.0, 10.0, 50.0, 12.0, 8.0)
        g1 = occ.addBox(-26.0, -6.1, 10.0 + 4.0, 52.0, 1.45, 2.6)
        g2 = occ.addBox(-26.0, 6.0 - 1.35, 10.0 + 4.0, 52.0, 1.45, 2.6)
        rail_s, _ = occ.cut([(3, r_box)], [(3, g1), (3, g2)])
        race1 = occ.addBox(-0.5 * c_len, -6.0 - clr, 10.0 + 4.0, c_len, 1.35 + clr, 2.6)
        race2 = occ.addBox(-0.5 * c_len, 6.0 - 1.35, 10.0 + 4.0, c_len, 1.35 + clr, 2.6)
        blk = occ.addBox(-0.5 * c_len, -13.5, 13.25, c_len, 27.0, 10.0)
        ucut = occ.addBox(-0.5 * c_len - 1.0, -6.0 - clr, 13.0, c_len + 2.0, 12.0 + 2.0 * clr, 5.0 + clr)
        carr, _ = occ.cut([(3, blk)], [(3, ucut)])
        _fragment_and_tag([(6, carr), (10, [(3, race1), (3, race2)]), (2, rail_s)])
        e_blk = 2000.0e9 if rigid_block else 205.0e9
        mats = {
            2: iso_constitutive_3d(e_blk, 0.29),
            6: iso_constitutive_3d(e_blk, 0.29),
            10: _race_constitutive(ey_gpa, gz_gpa),
        }
        mesh, basis, k_m, _ = _mesh_and_assemble(mats, h_bulk_mm=3.2, h_min_mm=0.35, order=2)
        dl = basis.doflocs
        z_top_m = 0.02325
        fix = np.nonzero(np.abs(dl[2] - 0.010) < 1e-4)[0]
        top_mask = np.abs(dl[2] - z_top_m) < 1e-4
        fy_dofs = np.nonzero(top_mask & ((np.arange(basis.N) % 3) == 1))[0]
        fz_dofs = np.nonzero(top_mask & ((np.arange(basis.N) % 3) == 2))[0]
        f_lat = basis.zeros(); f_lat[fy_dofs] = 50.0 / len(fy_dofs)
        f_roll = basis.zeros()
        yc = dl[1, fz_dofs]
        f_roll[fz_dofs] = (5.0 / np.sum(yc**2)) * yc
        k1, _, _, i1 = skfem.condense(k_m, f_lat, D=np.unique(np.concatenate([fix, fz_dofs])))
        u_lat = np.zeros(basis.N); u_lat[i1] = splu(k1.tocsc(), permc_spec="COLAMD").solve(f_lat[i1])
        k2, _, _, i2 = skfem.condense(k_m, f_roll, D=fix)
        u_rol = np.zeros(basis.N); u_rol[i2] = splu(k2.tocsc(), permc_spec="COLAMD").solve(f_roll[i2])
        top_v = np.nonzero(np.abs(mesh.p[2] - z_top_m) < 1e-4)[0]
        dy = float(np.mean(u_lat[basis.nodal_dofs[1, top_v]]))
        yt = mesh.p[1, top_v]
        th_x = float(np.sum(yt * u_rol[basis.nodal_dofs[2, top_v]]) / np.sum(yt**2))
        return 50.0 / abs(dy), 5.0 / abs(th_x)

    # 2-step linear calibration of raceway Hertzian stiffness to match HIWIN MGN12H Z1 (k_yy, k_rx) within <1%
    k_yy_0, k_rx_0 = _single_rail_calib(1.45, 1.85, rigid_block=True)
    ey_cal = 1.45 * (k_yy_target / k_yy_0)
    gz_cal = 1.85 * (k_rx_target / k_rx_0)
    k_yy_cal, k_rx_cal = _single_rail_calib(ey_cal, gz_cal, rigid_block=True)

    # Step B: Full 380 mm dual top+bottom beam + 2 rails + 5 mm Al 6061-T6 wrap-around C-shuttle
    gmsh.initialize(["gmsh"], interruptible=False)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(f"dual_{substrate}")
    occ = gmsh.model.occ

    if substrate == "al_2020":
        beam_vols = _extrude_x(_make_2020_tslot_face(), -l_mm / 2.0, l_mm / 2.0)
        d_beam = iso_constitutive_3d(68.9e9, 0.33)
        d_bolt = iso_constitutive_3d(205.0e9, 0.29)
    elif substrate == "rigid_bed":
        f_out = _make_rounded_rect_face(0.0, 0.0, 20.0, 20.0, 0.5)
        beam_vols = _extrude_x(f_out, -l_mm / 2.0, l_mm / 2.0)
        d_beam = iso_constitutive_3d(2000.0e9, 0.30)
        d_bolt = iso_constitutive_3d(2000.0e9, 0.30)
    else:
        f_out = _make_rounded_rect_face(0.0, 0.0, 20.0, 20.0, 2.5)
        f_in = _make_rounded_rect_face(0.0, 0.0, 16.4, 16.4, 0.7)
        sec, _ = occ.cut([(2, f_out)], [(2, f_in)])
        beam_vols = _extrude_x(sec[0][1], -l_mm / 2.0, l_mm / 2.0)
        d_beam, _, _ = cf_constitutive_at_temp("carbon_fiber_torsional_45", 25.0)
        d_bolt = iso_constitutive_3d(114.0e9, 0.34)

    # Top (+Z) and Bottom (-Z) MGN12 rails along X in [-175, +175]
    rt_box = occ.addBox(-175.0, -6.0, 10.0 + gap, 350.0, 12.0, 8.0)
    gt1 = occ.addBox(-176.0, -6.1, 10.0 + gap + 4.0, 352.0, 1.45, 2.6)
    gt2 = occ.addBox(-176.0, 6.0 - 1.35, 10.0 + gap + 4.0, 352.0, 1.45, 2.6)
    top_rail, _ = occ.cut([(3, rt_box)], [(3, gt1), (3, gt2)])

    rb_box = occ.addBox(-175.0, -6.0, -18.0 - gap, 350.0, 12.0, 8.0)
    gb1 = occ.addBox(-176.0, -6.1, -18.0 - gap + 1.4, 352.0, 1.45, 2.6)
    gb2 = occ.addBox(-176.0, 6.0 - 1.35, -18.0 - gap + 1.4, 352.0, 1.45, 2.6)
    bot_rail, _ = occ.cut([(3, rb_box)], [(3, gb1), (3, gb2)])

    bolt_vols, pad_vols, ins_vols = [], [], []
    for xb in np.arange(-162.5, 162.6, 25.0):
        xb_f = float(xb)
        # Cut Ø6.0 counterbore + Ø3.8 clearance hole in both top & bottom rails
        cb_t = occ.addCylinder(xb_f, 0.0, 18.0 + gap + 0.1, 0.0, 0.0, -3.6, 3.0)
        th_t = occ.addCylinder(xb_f, 0.0, 18.0 + gap - 3.3, 0.0, 0.0, -5.0, 1.9)
        top_rail, _ = occ.cut(top_rail, [(3, cb_t), (3, th_t)])
        cb_b = occ.addCylinder(xb_f, 0.0, -18.0 - gap - 0.1, 0.0, 0.0, 3.6, 3.0)
        th_b = occ.addCylinder(xb_f, 0.0, -18.0 - gap + 3.3, 0.0, 0.0, 5.0, 1.9)
        bot_rail, _ = occ.cut(bot_rail, [(3, cb_b), (3, th_b)])

        if substrate == "al_2020":
            for sz in (1.0, -1.0):
                z_head = sz * (18.0 + gap - 3.5)
                hd = occ.addCylinder(xb_f, 0.0, z_head, 0.0, 0.0, -sz * 1.5, 3.0)
                sh = occ.addCylinder(xb_f, 0.0, z_head - sz * 1.5, 0.0, 0.0, -sz * (abs(z_head - sz * 7.1) - 1.5), 1.5)
                tn = occ.addBox(xb_f - 3.5, -5.3, sz * 7.1 - 1.25, 7.0, 10.6, 2.5)
                bf, _ = occ.fuse([(3, hd)], [(3, sh), (3, tn)])
                bolt_vols.extend(bf)
                p1 = occ.addBox(xb_f - 3.5, -6.0, 10.0 if sz > 0 else -10.0 - gap, 7.0, 2.9, gap)
                p2 = occ.addBox(xb_f - 3.5, 3.1, 10.0 if sz > 0 else -10.0 - gap, 7.0, 2.9, gap)
                pad_vols.extend([(3, p1), (3, p2)])
        elif substrate == "rigid_bed":
            bolt_vols.append((3, occ.addCylinder(xb_f, 0.0, -14.5 - gap, 0.0, 0.0, 29.0 + 2.0 * gap, 1.5)))
            pad_vols.append((3, occ.addCylinder(xb_f, 0.0, 10.0, 0.0, 0.0, gap, 5.5)))
            pad_vols.append((3, occ.addCylinder(xb_f, 0.0, -10.0 - gap, 0.0, 0.0, gap, 5.5)))
        else:
            # Through-bolt M3 + heads + 7 mm PET-CF insert + top/bottom clamping pads
            ht = occ.addCylinder(xb_f, 0.0, 14.5 + gap, 0.0, 0.0, -1.5, 3.0)
            hb = occ.addCylinder(xb_f, 0.0, -14.5 - gap, 0.0, 0.0, 1.5, 3.0)
            sh = occ.addCylinder(xb_f, 0.0, -13.1 - gap, 0.0, 0.0, 26.2 + 2.0 * gap, 1.5)
            bf, _ = occ.fuse([(3, sh)], [(3, ht), (3, hb)])
            bolt_vols.extend(bf)
            ins_vols.extend(_make_cf_insert_block(xb_f - 3.5, xb_f + 3.5, 16.4))
            pad_vols.append((3, occ.addCylinder(xb_f, 0.0, 10.0, 0.0, 0.0, gap, 4.0)))
            pad_vols.append((3, occ.addCylinder(xb_f, 0.0, -10.0 - gap, 0.0, 0.0, gap, 4.0)))
    rail_vols = top_rail + bot_rail

    # 4 Hertzian ball-raceway layers bridging rail grooves (|y|=4.65) to shuttle skirts (|y|=6.50)
    race_vols = [
        (3, occ.addBox(-0.5 * c_len, -6.0 - clr, 10.0 + gap + 4.0, c_len, 1.35 + clr, 2.6)),
        (3, occ.addBox(-0.5 * c_len, 6.0 - 1.35, 10.0 + gap + 4.0, c_len, 1.35 + clr, 2.6)),
        (3, occ.addBox(-0.5 * c_len, -6.0 - clr, -18.0 - gap + 1.4, c_len, 1.35 + clr, 2.6)),
        (3, occ.addBox(-0.5 * c_len, 6.0 - 1.35, -18.0 - gap + 1.4, c_len, 1.35 + clr, 2.6)),
    ]

    # Two hardened steel MGN12H carriage blocks (27 x 10 x 32.4 mm, E=205 GPa) around top/bottom rails
    blk_t = occ.addBox(-0.5 * c_len, -13.5, 13.25 + gap, c_len, 27.0, 10.0)
    uc_t = occ.addBox(-0.5 * c_len - 1.0, -6.0 - clr, 13.0, c_len + 2.0, 12.0 + 2.0 * clr, 5.0 + gap + clr)
    carr_t, _ = occ.cut([(3, blk_t)], [(3, uc_t)])

    blk_b = occ.addBox(-0.5 * c_len, -13.5, -23.25 - gap, c_len, 27.0, 10.0)
    uc_b = occ.addBox(-0.5 * c_len - 1.0, -6.0 - clr, -18.0 - gap - clr, c_len + 2.0, 12.0 + 2.0 * clr, 5.0 + gap + clr)
    carr_b, _ = occ.cut([(3, blk_b)], [(3, uc_b)])
    mgn_blocks = carr_t + carr_b

    # 5 mm Al 6061-T6 wrap-around C-carriage shuttle bolted to top & bottom MGN12H mounting faces (z = +-(23.25+gap))
    z_top = 23.25 + gap
    z_bot = -23.25 - gap
    p_top = occ.addBox(-0.5 * c_len, -19.0, z_top, c_len, 32.5, 5.0)
    p_bot = occ.addBox(-0.5 * c_len, -19.0, z_bot - 5.0, c_len, 32.5, 5.0)
    p_front = occ.addBox(-0.5 * c_len, -19.0, z_bot - 5.0, c_len, 5.0, (z_top + 5.0) - (z_bot - 5.0))
    shuttle, _ = occ.fuse([(3, p_top)], [(3, p_bot), (3, p_front)])

    _fragment_and_tag(
        [
            (6, shuttle),
            (11, mgn_blocks),
            (10, race_vols),
            (3, bolt_vols),
            (4, pad_vols),
            (2, rail_vols),
            (5, ins_vols),
            (1, beam_vols),
        ]
    )

    mats = {
        1: d_beam,
        2: iso_constitutive_3d(205.0e9, 0.29),
        3: d_bolt,
        4: iso_constitutive_3d(200.0e9 if substrate == "rigid_bed" else 0.50e9, 0.32),
        5: iso_constitutive_3d(5.5e9, 0.35),
        6: iso_constitutive_3d(68.9e9, 0.33),  # 5 mm Al 6061-T6 shuttle
        10: _race_constitutive(ey_cal, gz_cal),
        11: iso_constitutive_3d(205.0e9, 0.29),  # 27x10 mm steel MGN12H carriage blocks
    }
    mesh, basis, k_mat, elem_gid = _mesh_and_assemble(mats, h_bulk_mm=5.2, h_min_mm=0.25, order=2, refine_bolts=True)

    dl = basis.doflocs
    tol = 1e-4
    beam_dofs = np.unique(basis.element_dofs[:, elem_gid == 1])
    is_beam = np.zeros(basis.N, dtype=bool); is_beam[beam_dofs] = True
    fix_ends = np.nonzero(is_beam & (np.abs(np.abs(dl[0]) - 0.190) < tol))[0]

    shuttle_dofs = np.unique(basis.element_dofs[:, elem_gid == 6])
    is_sh = np.zeros(basis.N, dtype=bool); is_sh[shuttle_dofs] = True
    front_face_dofs = np.nonzero(is_sh & (np.abs(dl[1] + 0.019) < tol))[0]
    fy_d = front_face_dofs[(front_face_dofs % 3) == 1]
    # Pure roll moment M_x = 10 N*m on front face of 5 mm Al shuttle (y=-19 mm)
    denom = float(np.sum(dl[2, fy_d] ** 2))
    f_roll = basis.zeros()
    f_roll[fy_d] = -(10.0 / denom) * dl[2, fy_d]

    (u_roll,) = _solve_factored(k_mat, fix_ends, [f_roll])

    shuttle_verts = np.unique(mesh.t[:, elem_gid == 6])
    front_v = shuttle_verts[np.abs(mesh.p[1, shuttle_verts] + 0.019) < tol]
    zf = mesh.p[2, front_v]
    uyf = u_roll[basis.nodal_dofs[1, front_v]]
    th_x_front = float(-np.sum(zf * uyf) / np.sum(zf**2))

    # Also measure roll angle directly across the top & bottom MGN12H steel carriage mount faces (z = +-23.5 mm)
    mgn_verts = np.unique(mesh.t[:, elem_gid == 11])
    mgn_faces = mgn_verts[np.abs(np.abs(mesh.p[2, mgn_verts]) - 0.0235) < tol]
    zm = mesh.p[2, mgn_faces]
    uym = u_roll[basis.nodal_dofs[1, mgn_faces]]
    th_x_mgn = float(-np.sum(zm * uym) / np.sum(zm**2))

    # Subtract mid-span beam centroid twist (measured on back half of beam y >= 0, x in [-5, 5])
    tube_verts = np.unique(mesh.t[:, elem_gid == 1])
    mid_back = tube_verts[(np.abs(mesh.p[0, tube_verts]) <= 0.005) & (mesh.p[1, tube_verts] >= 0.0)]
    yb, zb = mesh.p[1, mid_back], mesh.p[2, mid_back]
    uyb, uzb = u_roll[basis.nodal_dofs[1, mid_back]], u_roll[basis.nodal_dofs[2, mid_back]]
    th_x_beam = float(np.sum(yb * uzb - zb * uyb) / np.sum(yb**2 + zb**2))

    k_rx_front_3d = 10.0 / max(abs(th_x_front - th_x_beam), 1e-12)
    k_rx_mgn_3d = 10.0 / max(abs(th_x_mgn - th_x_beam), 1e-12)
    z_track = 0.5 * (0.020 + 0.008) + gap * 1e-3
    eta_front = (k_rx_front_3d - 2.0 * k_rx_cal) / (2.0 * k_yy_cal * (z_track**2))
    eta_mgn = (k_rx_mgn_3d - 2.0 * k_rx_cal) / (2.0 * k_yy_cal * (z_track**2))

    return {
        "substrate": substrate,
        "tets": int(mesh.nelements),
        "dofs": int(basis.N),
        "k_yy_cal_n_um": round(k_yy_cal * 1e-6, 2),
        "k_rx_single_cal_nm_rad": round(k_rx_cal, 1),
        "k_rx_dual_front_nm_rad": round(k_rx_front_3d, 1),
        "k_rx_dual_mgn_nm_rad": round(k_rx_mgn_3d, 1),
        "eta_steiner_front_3d": round(eta_front, 4),
        "eta_steiner_mgn_3d": round(eta_mgn, 4),
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }


def run_adeo_torsion_check(temp_c: float) -> dict[str, Any]:
    """Check 200 mm bare Carbonara CF tube torsion (10 N*m) vs The_Adeo bench test (+2.4 / +2.4 / +2.6 deg)."""
    gmsh.initialize(["gmsh"], interruptible=False)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(f"adeo_{int(temp_c)}")
    occ = gmsh.model.occ
    f_out = _make_rounded_rect_face(0.0, 0.0, 20.0, 20.0, 2.5)
    f_in = _make_rounded_rect_face(0.0, 0.0, 16.4, 16.4, 0.7)
    sec, _ = occ.cut([(2, f_out)], [(2, f_in)])
    beam_vols = _extrude_x(sec[0][1], -100.0, 100.0)
    _fragment_and_tag([(1, beam_vols)])
    d_mat, _, g_lt = cf_constitutive_at_temp("carbon_fiber_torsional_45", temp_c)
    mesh, basis, k_mat, _ = _mesh_and_assemble({1: d_mat}, h_bulk_mm=4.2, h_min_mm=0.6, order=2)
    dl = basis.doflocs
    tol = 1e-4
    fix_tor = np.nonzero(np.abs(dl[0] + 0.100) < tol)[0]
    f_right = np.nonzero(np.all(np.abs(mesh.p[0, mesh.facets] - 0.100) < tol, axis=0))[0]
    fb_r = skfem.FacetBasis(mesh, skfem.ElementVector(basis.elem.elem), facets=f_right, intorder=2)

    @skfem.Functional
    def ip_fn(w):
        return w.x[1] ** 2 + w.x[2] ** 2

    ip_e = float(ip_fn.assemble(fb_r))

    @skfem.LinearForm
    def tor_fn(v, w):
        return (10.0 / ip_e) * (-w.x[2] * v[1] + w.x[1] * v[2])

    (u_t,) = _solve_factored(k_mat, fix_tor, [tor_fn.assemble(fb_r)])
    rv = np.nonzero(np.abs(mesh.p[0] - 0.100) < tol)[0]
    yr, zr = mesh.p[1, rv], mesh.p[2, rv]
    uyr, uzr = u_t[basis.nodal_dofs[1, rv]], u_t[basis.nodal_dofs[2, rv]]
    theta_rad = float(np.sum(yr * uzr - zr * uyr) / np.sum(yr**2 + zr**2))
    return {
        "temp_c": temp_c,
        "g_lt_gpa": round(g_lt * 1e-9, 2),
        "twist_deg_10nm_200mm": round(math.degrees(theta_rad), 3),
        "gj_nm2": round(10.0 * 0.200 / theta_rad, 2),
    }


def run_bare_end_clamp_case(spec_tuple: tuple[str, str, str, bool]) -> dict[str, Any]:
    """Isolate M5 XY-joint end-crushing compliance (bare CF tube clamped at x=+-178 mm, F_y/T_x at midspan)."""
    label, prof_id, clamp_kind, with_inserts = spec_tuple
    t0 = time.perf_counter()
    gmsh.initialize(["gmsh"], interruptible=False)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(label)
    occ = gmsh.model.occ

    prof = get_profile(prof_id)
    t_mm = (prof.wall_thickness_m or 0.0018) * 1e3
    r_out = max(prof.corner_radius_m * 1e3, 1.5)
    id_mm = 20.0 - 2.0 * t_mm
    r_in = max(r_out - t_mm, 0.35)
    f_out = _make_rounded_rect_face(0.0, 0.0, 20.0, 20.0, r_out)
    f_in = _make_rounded_rect_face(0.0, 0.0, id_mm, id_mm, r_in)
    sec, _ = occ.cut([(2, f_out)], [(2, f_in)])
    beam_vols = _extrude_x(sec[0][1], -190.0, 190.0)

    ins_vols, st_vols, m5_vols = [], [], []
    if clamp_kind == "datum_m5":
        for sx in (-1.0, 1.0):
            xc = sx * 178.0
            hd_b = occ.addCylinder(xc, 0.0, -12.0, 0.0, 0.0, 2.0, 5.0)
            hd_t = occ.addCylinder(xc, 0.0, 10.0, 0.0, 0.0, 2.0, 5.0)
            sh_b = occ.addCylinder(xc, 0.0, -10.2, 0.0, 0.0, 20.4, 2.5)
            bf, _ = occ.fuse([(3, sh_b)], [(3, hd_b), (3, hd_t)])
            m5_vols.extend(bf)
    if with_inserts:
        for sx in (-1.0, 1.0):
            ex0, ex1 = (-190.0, -159.0) if sx < 0 else (159.0, 190.0)
            ins_vols.extend(_make_cf_insert_block(ex0, ex1, id_mm))
            st_vols.append((3, occ.addCylinder(sx * 178.0, 0.0, -0.5 * id_mm, 0.0, 0.0, id_mm, 4.6)))

    _fragment_and_tag([(3, m5_vols), (7, st_vols), (5, ins_vols), (1, beam_vols)])
    d_beam, _, _ = cf_constitutive_at_temp("carbon_fiber_torsional_45", 25.0)
    mats = {
        1: d_beam,
        3: iso_constitutive_3d(205.0e9, 0.29),
        5: iso_constitutive_3d(5.5e9, 0.35),
        7: iso_constitutive_3d(68.9e9, 0.33),
    }
    mesh, basis, k_mat, elem_gid = _mesh_and_assemble(mats, h_bulk_mm=4.5, h_min_mm=0.5, order=2)

    dl = basis.doflocs
    tol = 1e-4
    beam_dofs = np.unique(basis.element_dofs[:, elem_gid == 1])
    is_beam = np.zeros(basis.N, dtype=bool); is_beam[beam_dofs] = True

    if clamp_kind == "ideal_ring":
        fix_dofs = np.nonzero(is_beam & (np.abs(np.abs(dl[0]) - 0.178) <= 0.005))[0]
    elif clamp_kind == "datum_m5":
        # Bottom XY-joint shelf supports bottom washer face (z <= -10 mm, r <= 5 mm);
        # top wall relies on the M5 screw + (if present) M5 Al standoff & 31 mm end insert.
        r2 = (np.abs(dl[0]) - 0.178) ** 2 + dl[1] ** 2
        fix_dofs = np.nonzero((r2 <= 0.0052**2) & (dl[2] <= -0.010 + tol))[0]
    else:
        r2 = (np.abs(dl[0]) - 0.178) ** 2 + dl[1] ** 2
        fix_dofs = np.nonzero(is_beam & (r2 <= 0.0050**2) & (np.abs(np.abs(dl[2]) - 0.010) < tol))[0]

    load_y_dofs = np.nonzero(is_beam & (np.abs(dl[0]) <= 0.005) & (np.abs(np.abs(dl[2]) - 0.010) < tol) & ((np.arange(basis.N) % 3) == 1))[0]
    f_y = basis.zeros(); f_y[load_y_dofs] = -20.0 / len(load_y_dofs)

    mid_ring = np.nonzero(is_beam & (np.abs(dl[0]) <= 0.005))[0]
    yd = mid_ring[(mid_ring % 3) == 1]; zd = mid_ring[(mid_ring % 3) == 2]
    den_t = float(np.sum(dl[2, yd] ** 2) + np.sum(dl[1, zd] ** 2))
    f_tor = basis.zeros()
    f_tor[yd] = -(10.0 / den_t) * dl[2, yd]
    f_tor[zd] = +(10.0 / den_t) * dl[1, zd]

    u_y, u_t = _solve_factored(k_mat, fix_dofs, [f_y, f_tor])
    tube_v = np.unique(mesh.t[:, elem_gid == 1])
    mw_z = tube_v[(np.abs(mesh.p[0, tube_v]) <= 0.005) & (np.abs(np.abs(mesh.p[2, tube_v]) - 0.010) < 0.002)]
    dy_mid = float(np.mean(u_y[basis.nodal_dofs[1, mw_z]]))

    ma = tube_v[np.abs(mesh.p[0, tube_v]) <= 0.005]
    ym, zm = mesh.p[1, ma], mesh.p[2, ma]
    uym, uzm = u_t[basis.nodal_dofs[1, ma]], u_t[basis.nodal_dofs[2, ma]]
    th_mid = float(np.sum(ym * uzm - zm * uym) / np.sum(ym**2 + zm**2))

    return {
        "label": label,
        "dy_um": round(abs(dy_mid) * 1e6, 3),
        "theta_mid_mrad": round(abs(th_mid) * 1e3, 4),
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }


# ---------------------------------------------------------------------------
# 6. Main Orchestration Across All 6 Study Phases
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="PrintSim 3D continuum FEA X-beam calibration suite.")
    parser.add_argument(
        "--only",
        nargs="+",
        default=["all"],
        choices=["all", "bare", "cf_gamma", "torsion", "end_clamp", "local_seat", "al_2020", "backer", "dual_rail", "convergence"],
    )
    args = parser.parse_args()
    sel = set(args.only)
    run_all = "all" in sel

    if run_all or "bare" in sel:
        print("\n=== [1] Sub-model 1: Bare Beam Analytical Validation (<3% Error) ===", flush=True)
        with ProcessPoolExecutor(max_workers=2) as ex:
            for r in ex.map(run_bare_validation, ["cf_carbonara", "al_2020"]):
                print(r, flush=True)
        print("\n=== [1B] The_Adeo 200 mm Carbonara Torsion Bench Validation ===", flush=True)
        with ProcessPoolExecutor(max_workers=4) as ex:
            for r in ex.map(run_adeo_torsion_check, [25.0, 85.0, 100.0, 110.0]):
                print(r, flush=True)

    if run_all or "cf_gamma" in sel:
        print("\n=== [2A/B] Sub-model 2: CF 2-Wall Clamped Inserts vs 1-Wall Bare Bending & Gamma ===", flush=True)
        cf_cases = [
            # Carbonara 2-wall (25/50 mm, 25C/85C)
            BeamRailSpec("carb_2w_p25_25C", "cf", "cf_tube_20x20_carbonara", "carbon_fiber_torsional_45", "mgn12_rail", "two_wall_clamped_inserts", 25.0, 25.0),
            BeamRailSpec("carb_2w_p50_25C", "cf", "cf_tube_20x20_carbonara", "carbon_fiber_torsional_45", "mgn12_rail", "two_wall_clamped_inserts", 50.0, 25.0),
            BeamRailSpec("carb_2w_p25_85C", "cf", "cf_tube_20x20_carbonara", "carbon_fiber_torsional_45", "mgn12_rail", "two_wall_clamped_inserts", 25.0, 85.0),
            # UD-rich Tg 80C 2-wall (25C/85C)
            BeamRailSpec("ud_2w_p25_25C", "cf", "cf_tube_20x20_carbonara", "carbon_fiber_tube", "mgn12_rail", "two_wall_clamped_inserts", 25.0, 25.0),
            BeamRailSpec("ud_2w_p25_85C", "cf", "cf_tube_20x20_carbonara", "carbon_fiber_tube", "mgn12_rail", "two_wall_clamped_inserts", 25.0, 85.0),
            # Single-wall bare CF across wall thicknesses (1.0, 1.8, 2.0 mm), pitch (25/50), & 85C
            BeamRailSpec("cf_1w_1.0mm_p25_25C", "cf", "cf_tube_20x20x1", "carbon_fiber_tube", "mgn12_rail", "single_wall_bare", 25.0, 25.0),
            BeamRailSpec("cf_1w_1.8mm_p25_25C", "cf", "cf_tube_20x20_carbonara", "carbon_fiber_torsional_45", "mgn12_rail", "single_wall_bare", 25.0, 25.0),
            BeamRailSpec("cf_1w_2.0mm_p25_25C", "cf", "cf_tube_20x20x2", "carbon_fiber_tube", "mgn12_rail", "single_wall_bare", 25.0, 25.0),
            BeamRailSpec("cf_1w_1.8mm_p50_25C", "cf", "cf_tube_20x20_carbonara", "carbon_fiber_torsional_45", "mgn12_rail", "single_wall_bare", 50.0, 25.0),
            BeamRailSpec("ud_1w_1.8mm_p25_85C", "cf", "cf_tube_20x20_carbonara", "carbon_fiber_tube", "mgn12_rail", "single_wall_bare", 25.0, 85.0),
            BeamRailSpec("carb_1w_1.8mm_p25_85C", "cf", "cf_tube_20x20_carbonara", "carbon_fiber_torsional_45", "mgn12_rail", "single_wall_bare", 25.0, 85.0),
        ]
        with ProcessPoolExecutor(max_workers=4) as ex:
            for r in ex.map(run_beam_rail_case, cf_cases):
                print(r, flush=True)

    if run_all or "torsion" in sel:
        print("\n=== [2C] Sub-model 2C: Global Torsion GJ (Inserts Corner Boost) ===", flush=True)
        tor_cases = [
            BeamRailSpec("tor_1w_bare_mgn12", "cf", "cf_tube_20x20_carbonara", "carbon_fiber_torsional_45", "mgn12_rail", "single_wall_bare", 25.0, 25.0, solve_torsion=True),
            BeamRailSpec("tor_2w_ins_mgn12", "cf", "cf_tube_20x20_carbonara", "carbon_fiber_torsional_45", "mgn12_rail", "two_wall_clamped_inserts", 25.0, 25.0, solve_torsion=True),
        ]
        with ProcessPoolExecutor(max_workers=2) as ex:
            for r in ex.map(run_beam_rail_case, tor_cases):
                print(r, flush=True)

    if run_all or "end_clamp" in sel:
        print("\n=== [2D] Sub-model 2D: Bare Tube Hollow-End vs M5-Standoff End-Insert Clamping ===", flush=True)
        bare_end_cases = [
            ("end_ideal_1.8", "cf_tube_20x20_carbonara", "ideal_ring", True),
            ("end_m5_ins_1.8", "cf_tube_20x20_carbonara", "m5", True),
            ("end_m5_hol_1.0", "cf_tube_20x20x1", "m5", False),
            ("end_m5_hol_1.8", "cf_tube_20x20_carbonara", "m5", False),
            ("end_m5_hol_2.0", "cf_tube_20x20x2", "m5", False),
            ("datum_ins_1.8", "cf_tube_20x20_carbonara", "datum_m5", True),
            ("datum_hol_1.0", "cf_tube_20x20x1", "datum_m5", False),
            ("datum_hol_1.8", "cf_tube_20x20_carbonara", "datum_m5", False),
            ("datum_hol_2.0", "cf_tube_20x20x2", "datum_m5", False),
        ]
        with ProcessPoolExecutor(max_workers=5) as ex:
            for r in ex.map(run_bare_end_clamp_case, bare_end_cases):
                print(r, flush=True)

    if run_all or "local_seat" in sel:
        print("\n=== [2E/3B] Local Carriage-Seat Compliance (MGN12H 32.4mm & MGN9H 29.9mm) ===", flush=True)
        seat_cases = [
            BeamRailSpec("seat_carb2w_mgn12h", "cf", "cf_tube_20x20_carbonara", "carbon_fiber_torsional_45", "mgn12_rail", "two_wall_clamped_inserts", 25.0, 25.0, solve_local_seat=True, carriage_len_mm=32.4),
            BeamRailSpec("seat_carb2w_mgn9h", "cf", "cf_tube_20x20_carbonara", "carbon_fiber_torsional_45", "mgn9_rail", "two_wall_clamped_inserts", 25.0, 25.0, solve_local_seat=True, carriage_len_mm=29.9),
            BeamRailSpec("seat_cf1w_1.0_mgn12h", "cf", "cf_tube_20x20x1", "carbon_fiber_torsional_45", "mgn12_rail", "single_wall_bare", 25.0, 25.0, solve_local_seat=True, carriage_len_mm=32.4),
            BeamRailSpec("seat_cf1w_1.8_mgn12h", "cf", "cf_tube_20x20_carbonara", "carbon_fiber_torsional_45", "mgn12_rail", "single_wall_bare", 25.0, 25.0, solve_local_seat=True, carriage_len_mm=32.4),
            BeamRailSpec("seat_cf1w_2.0_mgn12h", "cf", "cf_tube_20x20x2", "carbon_fiber_torsional_45", "mgn12_rail", "single_wall_bare", 25.0, 25.0, solve_local_seat=True, carriage_len_mm=32.4),
            BeamRailSpec("seat_cf1w_1.8_mgn9h", "cf", "cf_tube_20x20_carbonara", "carbon_fiber_torsional_45", "mgn9_rail", "single_wall_bare", 25.0, 25.0, solve_local_seat=True, carriage_len_mm=29.9),
            BeamRailSpec("seat_al2020_mgn12h", "al_2020", rail_id="mgn12_rail", solve_local_seat=True, carriage_len_mm=32.4),
            BeamRailSpec("seat_al2020_mgn9h", "al_2020", rail_id="mgn9_rail", solve_local_seat=True, carriage_len_mm=29.9),
        ]
        with ProcessPoolExecutor(max_workers=4) as ex:
            for r in ex.map(run_beam_rail_case, seat_cases):
                print(r, flush=True)

    if run_all or "al_2020" in sel:
        print("\n=== [3] Sub-model 3: 2020 Al Extrusion + MGN12/MGN9 (25 & 50 mm pitch) ===", flush=True)
        al_cases = [
            BeamRailSpec("al2020_mgn12_p25", "al_2020", rail_id="mgn12_rail", pitch_mm=25.0, solve_torsion=True),
            BeamRailSpec("al2020_mgn12_p50", "al_2020", rail_id="mgn12_rail", pitch_mm=50.0),
            BeamRailSpec("al2020_mgn9_p25", "al_2020", rail_id="mgn9_rail", pitch_mm=25.0, solve_torsion=True),
            BeamRailSpec("al2020_mgn9_p50", "al_2020", rail_id="mgn9_rail", pitch_mm=50.0),
        ]
        with ProcessPoolExecutor(max_workers=4) as ex:
            for r in ex.map(run_beam_rail_case, al_cases):
                print(r, flush=True)

    if run_all or "backer" in sel:
        print("\n=== [4] Sub-model 4: Opposite-Face 3 mm Steel 4140 Backer Sandwich ===", flush=True)
        bk_cases = [
            BeamRailSpec("backer_carb_2w", "cf", "cf_tube_20x20_carbonara", "carbon_fiber_torsional_45", "mgn12_rail", "two_wall_clamped_inserts", 25.0, 25.0, with_backer=True, solve_torsion=True),
            BeamRailSpec("backer_al2020", "al_2020", rail_id="mgn12_rail", pitch_mm=25.0, with_backer=True, solve_torsion=True),
        ]
        with ProcessPoolExecutor(max_workers=2) as ex:
            for r in ex.map(run_beam_rail_case, bk_cases):
                print(r, flush=True)

    if run_all or "dual_rail" in sel:
        print("\n=== [5] Sub-model 5: Dual Top+Bottom MGN12H + 5 mm Al 6061 Carriage Shuttle ===", flush=True)
        with ProcessPoolExecutor(max_workers=3) as ex:
            for r in ex.map(run_dual_rail_steiner_case, ["rigid_bed", "cf_carbonara", "al_2020"]):
                print(r, flush=True)

    if run_all or "convergence" in sel:
        print("\n=== [6] Sub-model 6: Mesh & Element Convergence (Tet4 vs Tet10 Coarse/Med/Fine) ===", flush=True)
        conv_cases = [
            BeamRailSpec("conv_2w_tet4_h4.5", "cf", "cf_tube_20x20_carbonara", "carbon_fiber_torsional_45", "mgn12_rail", "two_wall_clamped_inserts", 25.0, 25.0, h_bulk_mm=4.5, order=1),
            BeamRailSpec("conv_2w_tet10_coarse_h6.5", "cf", "cf_tube_20x20_carbonara", "carbon_fiber_torsional_45", "mgn12_rail", "two_wall_clamped_inserts", 25.0, 25.0, h_bulk_mm=6.5, order=2),
            BeamRailSpec("conv_2w_tet10_med_h5.0", "cf", "cf_tube_20x20_carbonara", "carbon_fiber_torsional_45", "mgn12_rail", "two_wall_clamped_inserts", 25.0, 25.0, h_bulk_mm=5.0, order=2),
            BeamRailSpec("conv_2w_tet10_fine_h3.6", "cf", "cf_tube_20x20_carbonara", "carbon_fiber_torsional_45", "mgn12_rail", "two_wall_clamped_inserts", 25.0, 25.0, h_bulk_mm=3.6, order=2),
            BeamRailSpec("conv_1w_tet10_coarse_h6.5", "cf", "cf_tube_20x20_carbonara", "carbon_fiber_torsional_45", "mgn12_rail", "single_wall_bare", 25.0, 25.0, h_bulk_mm=6.5, order=2),
            BeamRailSpec("conv_1w_tet10_fine_h3.6", "cf", "cf_tube_20x20_carbonara", "carbon_fiber_torsional_45", "mgn12_rail", "single_wall_bare", 25.0, 25.0, h_bulk_mm=3.6, order=2),
        ]
        with ProcessPoolExecutor(max_workers=4) as ex:
            for r in ex.map(run_beam_rail_case, conv_cases):
                print(r, flush=True)


if __name__ == "__main__":
    main()
