"""Regression tests for Group B Continuum FEA audit fixes (B-08, B-09, B-10, B-19, B-20) and C-09."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import trimesh
from scipy.linalg import solve

from printsim.fea.bracket_designer import (
    _bracket_layout_dims,
    _build_bracket_rectilinear_grid,
    design_and_evaluate_bracket,
)
from printsim.fea.shell_mitc4 import (
    _ALPHA_DRILL,
    mitc4_local_frame,
    mitc4_shell_matrices,
)
from printsim.fea.solid_fem import (
    constitutive_matrix_3d,
    generate_parametric_part_grid,
    hex8_incompatible_matrices,
    solve_preset_part_fea,
    solve_solid_voxel_fea,
)
from printsim.fea.superelement import (
    _BRACKET_CACHE,
    _build_coarse_bracket_rectilinear_grid,
    condense_parametric_bracket,
)
from printsim.fea.tet_fem import (
    is_tet_fea_available,
    solve_preset_tet_fea,
    solve_step_bytes_tet_fea,
)
from printsim.geometry.cad_b3d import is_cad_available
from printsim.schema import get_material


# ---------------------------------------------------------------------------
# B-10: Hex8 Wilson-Taylor (QM6-3D) incompatible modes
# ---------------------------------------------------------------------------


def test_b10_hex8_qm6_patch_test_rbm_and_1elem_bending_vs_timoshenko() -> None:
    """Verify QM6-3D Hex8 passes 3D patch test, preserves 6 RBMs, and eliminates shear locking."""
    mat = get_material("aluminum_6061_t6")
    D = constitutive_matrix_3d(mat, fdm_anisotropic=False)
    h_rect = (0.006, 0.004, 0.0025)

    Ke, B0, C_a = hex8_incompatible_matrices(D, h_rect)
    assert Ke.shape == (24, 24)
    assert B0.shape == (6, 24)
    assert C_a.shape == (9, 24)
    assert np.allclose(Ke, Ke.T, atol=1e-10)

    eigs = np.linalg.eigvalsh(Ke)
    assert np.max(np.abs(eigs[:6])) < 1e-10 * float(np.max(eigs))
    assert eigs[6] > 0.0

    # Constant-strain 3D patch test on a 3x2x2 non-cubic brick mesh (free interior nodes)
    nx, ny, nz = 3, 2, 2
    hx, hy, hz = 0.006, 0.004, 0.003
    n_x_nodes, n_y_nodes, n_z_nodes = nx + 1, ny + 1, nz + 1
    n_nodes = n_x_nodes * n_y_nodes * n_z_nodes
    coords = np.array(
        [
            [ix * hx, iy * hy, iz * hz]
            for ix in range(n_x_nodes)
            for iy in range(n_y_nodes)
            for iz in range(n_z_nodes)
        ],
        dtype=float,
    )
    # Prescribed arbitrary constant strain field (normal + shear)
    eps = np.array([1.2e-3, -0.4e-3, 0.7e-3, 0.6e-3, -0.3e-3, 0.5e-3], dtype=float)
    u_exact = np.zeros(3 * n_nodes, dtype=float)
    for nid, (x, y, z) in enumerate(coords):
        u_exact[3 * nid + 0] = eps[0] * x + 0.5 * eps[3] * y + 0.5 * eps[5] * z
        u_exact[3 * nid + 1] = 0.5 * eps[3] * x + eps[1] * y + 0.5 * eps[4] * z
        u_exact[3 * nid + 2] = 0.5 * eps[5] * x + 0.5 * eps[4] * y + eps[2] * z

    K_patch = np.zeros((3 * n_nodes, 3 * n_nodes), dtype=float)
    Ke_p, _, _ = hex8_incompatible_matrices(D, (hx, hy, hz))
    offsets = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                nds = [
                    (ix + ox) * (n_y_nodes * n_z_nodes) + (iy + oy) * n_z_nodes + (iz + oz)
                    for ox, oy, oz in offsets
                ]
                dofs = [3 * n + d for n in nds for d in range(3)]
                K_patch[np.ix_(dofs, dofs)] += Ke_p

    bnd_nodes = [
        ix * (n_y_nodes * n_z_nodes) + iy * n_z_nodes + iz
        for ix in range(n_x_nodes)
        for iy in range(n_y_nodes)
        for iz in range(n_z_nodes)
        if ix in (0, nx) or iy in (0, ny) or iz in (0, nz)
    ]
    presc = [3 * n + d for n in bnd_nodes for d in range(3)]
    free = [d for d in range(3 * n_nodes) if d not in set(presc)]
    assert len(free) > 0
    u_sol = u_exact.copy()
    u_sol[free] = solve(K_patch[np.ix_(free, free)], -K_patch[np.ix_(free, presc)] @ u_exact[presc])
    assert np.max(np.abs(u_sol - u_exact)) < 1e-12

    # 1-element-through-thickness bending beams vs exact Timoshenko deflection (> 95% accuracy)
    E = mat.youngs_modulus_pa
    G = mat.effective_shear_modulus_pa
    kappa = 5.0 / 6.0
    F_z = -50.0
    cases = [
        (5, 1, 1, 60.0, 12.0, 12.0),
        (10, 2, 2, 60.0, 12.0, 12.0),
        (10, 4, 1, 40.0, 20.0, 6.0),
    ]
    for n_ex, n_ey, n_ez, L_mm, b_mm, h_mm in cases:
        hx_mm, hy_mm, hz_mm = L_mm / n_ex, b_mm / n_ey, h_mm / n_ez
        res = solve_solid_voxel_fea(
            grid=np.ones((n_ex, n_ey, n_ez), dtype=bool),
            voxel_size_mm=(hx_mm, hy_mm, hz_mm),
            material_id="aluminum_6061_t6",
            fixed_face="min_x",
            load_face="max_x",
            force_n=(0.0, 0.0, F_z),
            fdm_anisotropic=False,
        )
        L_m, b_m, h_m = L_mm * 1e-3, b_mm * 1e-3, h_mm * 1e-3
        I_y = b_m * (h_m**3) / 12.0
        delta_tim_um = (
            abs(F_z) * (L_m**3) / (3.0 * E * I_y) + abs(F_z) * L_m / (kappa * G * b_m * h_m)
        ) * 1e6
        ratio = res.max_displacement_um / delta_tim_um
        assert 0.95 <= ratio <= 1.03, f"Mesh {n_ex}x{n_ey}x{n_ez}: ratio={ratio:.4f}"


# ---------------------------------------------------------------------------
# B-09: Surface / Gauss / corner-extrapolated stress recovery
# ---------------------------------------------------------------------------


def test_b09_hex8_surface_bending_and_z_tensile_stress_recovery() -> None:
    """Hex8 corner/nodal stress recovery matches 6PL/(bh^2) within [0.90, 1.05] and stays bounded at corners."""
    # Horizontal cantilever bar (60 x 12 x 12 mm)
    res_bar = solve_preset_part_fea(
        preset="cantilever_bar",
        material_id="aluminum_6061_t6",
        force_n=(0.0, 0.0, -50.0),
        fdm_anisotropic=False,
    )
    sigma_nom_mpa = (6.0 * 50.0 * 0.060 / (0.012 * (0.012**2))) * 1e-6
    ratio_vm = res_bar.max_von_mises_mpa / sigma_nom_mpa
    assert 0.90 <= ratio_vm <= 1.05, f"Hex8 cantilever ratio={ratio_vm:.4f}"

    # Vertical Z-built cantilever bar (12 x 12 x 60 mm) under lateral X load -> max_z_tensile_mpa = 6PL/(bh^2)
    res_z = solve_solid_voxel_fea(
        grid=np.ones((4, 4, 20), dtype=bool),
        voxel_size_mm=3.0,
        material_id="pa12_cf_printed",
        fixed_face="min_z",
        load_face="max_z",
        force_n=(50.0, 0.0, 0.0),
        fdm_anisotropic=True,
    )
    ratio_z = res_z.max_z_tensile_mpa / sigma_nom_mpa
    assert 0.90 <= ratio_z <= 1.05, f"Hex8 vertical bar Z-tensile ratio={ratio_z:.4f}"

    # Re-entrant corner gusset refinement must not diverge wildly under nodal averaging
    grid_coarse, h_c, ff, lf, f_def = generate_parametric_part_grid("corner_gusset")
    grid_fine = np.repeat(np.repeat(np.repeat(grid_coarse, 2, axis=0), 2, axis=1), 2, axis=2)
    res_c = solve_solid_voxel_fea(grid_coarse, h_c, "aluminum_6061_t6", ff, lf, f_def, fdm_anisotropic=False)
    res_f = solve_solid_voxel_fea(grid_fine, 0.5 * h_c, "aluminum_6061_t6", ff, lf, f_def, fdm_anisotropic=False)
    assert res_f.max_von_mises_mpa >= res_c.max_von_mises_mpa * 0.95
    assert res_f.max_von_mises_mpa < res_c.max_von_mises_mpa * 1.8


@pytest.mark.skipif(
    not (is_tet_fea_available() and is_cad_available()),
    reason="Requires optional [cad] and [tetfea] extras (build123d, gmsh, scikit-fem, meshio).",
)
def test_b09_tet10_surface_bending_and_z_tensile_stress_recovery(tmp_path: Path) -> None:
    """Tet10 vertex-extrapolated stress recovery matches 6PL/(bh^2) within [0.90, 1.05] for VM and Z-tensile."""
    res_bar = solve_preset_tet_fea(
        preset="cantilever_bar",
        material_id="aluminum_6061_t6",
        force_n=(0.0, 0.0, -50.0),
        order=2,
        fdm_anisotropic=False,
    )
    sigma_nom_mpa = (6.0 * 50.0 * 0.060 / (0.012 * (0.012**2))) * 1e-6
    ratio_vm = res_bar.max_von_mises_mpa / sigma_nom_mpa
    assert 0.90 <= ratio_vm <= 1.05, f"Tet10 cantilever VM ratio={ratio_vm:.4f}"

    import build123d as b3d

    solid_z = b3d.Solid.make_box(12.0, 12.0, 60.0, plane=b3d.Plane.XY)
    step_path = tmp_path / "vertical_bar.step"
    b3d.export_step(solid_z, str(step_path))
    res_z = solve_step_bytes_tet_fea(
        step_bytes=step_path.read_bytes(),
        material_id="pa12_cf_printed",
        fixed_face="min_z",
        load_face="max_z",
        force_n=(50.0, 0.0, 0.0),
        order=2,
        fdm_anisotropic=True,
    )
    ratio_z = res_z.max_z_tensile_mpa / sigma_nom_mpa
    assert 0.90 <= ratio_z <= 1.05, f"Tet10 vertical bar Z-tensile ratio={ratio_z:.4f}"


# ---------------------------------------------------------------------------
# B-08: Smooth feature-aligned bracket wall thickness & superelement convergence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("bracket_type", "rib_mm", "hole_mm", "fillet_mm"),
    [
        ("corner_gusset", 0.0, 0.0, 0.0),
        ("corner_gusset", 6.0, 5.5, 2.0),
        ("xy_joint_l_bracket", 6.0, 5.5, 2.0),
        ("motor_mount", 6.0, 5.5, 2.0),
    ],
)
def test_b08_bracket_wall_thickness_smooth_monotonic_no_staircase(
    bracket_type: str,
    rib_mm: float,
    hole_mm: float,
    fillet_mm: float,
) -> None:
    """Sweeping wall_thickness_mm across 4.9, 5.0, 6.0, 6.9, 7.0 produces smooth monotonic responses."""
    walls = [4.9, 5.0, 5.5, 6.0, 6.5, 6.9, 7.0]
    results = [
        design_and_evaluate_bracket(
            bracket_type=bracket_type,  # type: ignore[arg-type]
            arm_length_mm=40.0,
            height_mm=20.0,
            wall_thickness_mm=w,
            rib_thickness_mm=rib_mm,
            bolt_hole_mm=hole_mm,
            fillet_mm=fillet_mm,
            material_id="aluminum_6061_t6",
            engine="hex8",
        )
        for w in walls
    ]

    for w, r in zip(walls, results):
        assert r["realized_wall_thickness_mm"] == pytest.approx(w, rel=1e-3)

    stiffs = [r["stiffness_n_per_mm"] for r in results]
    vols = [r["volume_cm3"] for r in results]

    # Strictly monotonic with no 0-gradient plateaus
    for i in range(len(walls) - 1):
        assert stiffs[i + 1] > stiffs[i], f"{bracket_type} stiff not monotonic at {walls[i]}->{walls[i+1]}: {stiffs}"
        assert vols[i + 1] > vols[i], f"{bracket_type} vol not monotonic at {walls[i]}->{walls[i+1]}: {vols}"

    # No staircase jumps across integer grid crossings (4.9->5.0 or 6.9->7.0)
    idx_49, idx_50, idx_69, idx_70 = 0, 1, 5, 6
    assert stiffs[idx_50] / stiffs[idx_49] < 1.15
    assert stiffs[idx_70] / stiffs[idx_69] < 1.15

    # Compare against conformal Tet10 on clean unribbed corner_gusset (< 3% error across sweep)
    if bracket_type == "corner_gusset" and rib_mm == 0.0 and is_tet_fea_available() and is_cad_available():
        for w, r_hex in zip([4.9, 5.0, 6.0, 6.9, 7.0], [results[0], results[1], results[3], results[5], results[6]]):
            r_tet = design_and_evaluate_bracket(
                bracket_type="corner_gusset",
                arm_length_mm=40.0,
                height_mm=20.0,
                wall_thickness_mm=w,
                rib_thickness_mm=0.0,
                bolt_hole_mm=0.0,
                fillet_mm=0.0,
                material_id="aluminum_6061_t6",
                engine="tet10",
            )
            assert r_hex["volume_cm3"] == pytest.approx(r_tet["volume_cm3"], rel=1e-3)
            assert r_hex["stiffness_n_per_mm"] == pytest.approx(r_tet["stiffness_n_per_mm"], rel=0.03)


def test_b08_rectilinear_hex8_stl_export_watertight_and_exact_dimensions(tmp_path: Path) -> None:
    """Watertight rectilinear STL export preserves exact outer extents and solid volume at w=4.9 mm."""
    stl_path = tmp_path / "gusset_4p9.stl"
    res = design_and_evaluate_bracket(
        bracket_type="corner_gusset",
        arm_length_mm=40.0,
        height_mm=20.0,
        wall_thickness_mm=4.9,
        rib_thickness_mm=0.0,
        bolt_hole_mm=0.0,
        fillet_mm=0.0,
        material_id="aluminum_6061_t6",
        engine="hex8",
        output_stl_path=stl_path,
    )
    mesh = trimesh.load(str(stl_path), force="mesh")
    assert mesh.is_watertight
    extents = mesh.bounds[1] - mesh.bounds[0]
    assert extents == pytest.approx([40.0, 40.0, 20.0], rel=1e-4)
    assert mesh.volume * 1e-3 == pytest.approx(res["volume_cm3"], rel=1e-3)


def test_b08_superelement_pitch_floor_loosened_and_converges() -> None:
    """Superelement refines past old max(2.0, min_dim/6) floor and matches analytic mass & converged stiffness."""
    dims = _bracket_layout_dims("corner_gusset", 40.0, 20.0, 6.0, 5.0, 5.2, 2.0)
    grid_coarse, (dx_c, dy_c, dz_c) = _build_coarse_bracket_rectilinear_grid(
        "corner_gusset", dims, target_elements=380
    )
    grid_fine, (dx_f, dy_f, dz_f) = _build_coarse_bracket_rectilinear_grid(
        "corner_gusset", dims, target_elements=12000
    )
    assert int(grid_fine.sum()) > int(grid_coarse.sum()) * 10
    assert float(np.mean(dx_f)) < 1.5
    assert float(np.mean(dz_f)) < 1.5

    _BRACKET_CACHE.clear()
    se_380 = condense_parametric_bracket(
        {"bracket_type": "corner_gusset", "target_elements": 380, "fdm_anisotropic": False},
        "aluminum_6061_t6",
    )
    se_1600 = condense_parametric_bracket(
        {"bracket_type": "corner_gusset", "target_elements": 1600, "fdm_anisotropic": False},
        "aluminum_6061_t6",
    )
    se_4800 = condense_parametric_bracket(
        {"bracket_type": "corner_gusset", "target_elements": 4800, "fdm_anisotropic": False},
        "aluminum_6061_t6",
    )

    # Analytic Al 6061 corner_gusset (40/20/6/5/5.2/2) mass is ~25.1 g
    for se in (se_380, se_1600, se_4800):
        assert se.mass_kg * 1000.0 == pytest.approx(25.1, rel=0.03)

    # Default (380) is already within 15% of converged (4800) thanks to feature alignment + QM6-3D
    for key in ("k_x_n_m", "k_y_n_m", "k_z_n_m"):
        assert se_380.port_stiffness_diag[key] == pytest.approx(
            se_4800.port_stiffness_diag[key], rel=0.15
        )
        assert se_1600.port_stiffness_diag[key] == pytest.approx(
            se_4800.port_stiffness_diag[key], rel=0.12
        )


# ---------------------------------------------------------------------------
# C-09: Consistent directional stiffness definition
# ---------------------------------------------------------------------------


def test_c09_bracket_and_solid_stiffness_definition_aligned() -> None:
    """bracket_designer and solid_fem/tet_fem report identical F / mean(u . f_hat) stiffness."""
    dims = _bracket_layout_dims("corner_gusset", 40.0, 20.0, 6.0, 5.0, 5.5, 2.0)
    grid, (dx, dy, dz), ff, lf, _ = _build_bracket_rectilinear_grid("corner_gusset", dims)
    force = (60.0, 0.0, -20.0)

    hex_res = solve_solid_voxel_fea(
        grid=grid,
        voxel_size_mm=(float(np.mean(dx)), float(np.mean(dy)), float(np.mean(dz))),
        material_id="ppa_cf_printed",
        fixed_face=ff,
        load_face=lf,
        force_n=force,
        cell_sizes_mm=(dx, dy, dz),
    )
    bkt_hex = design_and_evaluate_bracket(
        bracket_type="corner_gusset",
        arm_length_mm=40.0,
        height_mm=20.0,
        wall_thickness_mm=6.0,
        rib_thickness_mm=5.0,
        bolt_hole_mm=5.5,
        fillet_mm=2.0,
        material_id="ppa_cf_printed",
        force_n=force,
        engine="hex8",
    )
    assert bkt_hex["stiffness_n_per_mm"] == pytest.approx(round(hex_res.stiffness_n_per_mm, 2), abs=0.02)

    # Pure bending cantilever stiffness along Z must exclude axial fiber-rotation u_x contamination
    bar_res = solve_preset_part_fea(
        preset="cantilever_bar",
        material_id="aluminum_6061_t6",
        force_n=(0.0, 0.0, -50.0),
        fdm_anisotropic=False,
    )
    mat = get_material("aluminum_6061_t6")
    E, G = mat.youngs_modulus_pa, mat.effective_shear_modulus_pa
    L, b, h = 0.060, 0.012, 0.012
    Iy = b * (h**3) / 12.0
    delta_tim_mm = (50.0 * L**3 / (3.0 * E * Iy) + 50.0 * L / ((5.0 / 6.0) * G * b * h)) * 1e3
    exact_k_n_mm = 50.0 / delta_tim_mm
    assert bar_res.stiffness_n_per_mm == pytest.approx(exact_k_n_mm, rel=0.04)

    if is_tet_fea_available() and is_cad_available():
        from printsim.fea.bracket_designer import _build_parametric_bracket_brep
        import tempfile

        b3d, solid, b_fix, b_load = _build_parametric_bracket_brep("corner_gusset", dims)
        with tempfile.TemporaryDirectory() as td:
            sp = Path(td) / "cg.step"
            b3d.export_step(solid, str(sp))
            tet_res = solve_step_bytes_tet_fea(
                sp.read_bytes(),
                material_id="ppa_cf_printed",
                fixed_face=b_fix,
                load_face=b_load,
                force_n=force,
                order=2,
            )
        bkt_tet = design_and_evaluate_bracket(
            bracket_type="corner_gusset",
            arm_length_mm=40.0,
            height_mm=20.0,
            wall_thickness_mm=6.0,
            rib_thickness_mm=5.0,
            bolt_hole_mm=5.5,
            fillet_mm=2.0,
            material_id="ppa_cf_printed",
            force_n=force,
            engine="tet10",
        )
        assert bkt_tet["stiffness_n_per_mm"] == pytest.approx(
            round(tet_res.stiffness_n_per_mm, 2), rel=0.02
        )


# ---------------------------------------------------------------------------
# B-19 & B-20: MITC4 QM6 incompatible membrane & drilling stabilization
# ---------------------------------------------------------------------------


def test_b19_mitc4_qm6_strip_cantilever_and_distorted_patch_test() -> None:
    """QM6 membrane recovers >95% in-plane bending at aspects 1, 2, 3.3 and passes distorted patch test."""
    e_pa = 70.3e9
    nu = 0.33
    t = 0.004
    rho = 2680.0
    g_pa = e_pa / (2.0 * (1.0 + nu))

    # 1. One-element-wide strip cantilever in in-plane bending at aspect ratios 1.0, 2.0, 3.3
    for aspect in (1.0, 2.0, 3.3):
        h_strip = 0.020
        n_el = 6
        dx = aspect * h_strip
        L = n_el * dx
        pts = np.array(
            [[i * dx, y, 0.0] for i in range(n_el + 1) for y in (0.0, h_strip)],
            dtype=float,
        )
        n_dof = 6 * len(pts)
        K = np.zeros((n_dof, n_dof), dtype=float)
        for i in range(n_el):
            q = [2 * i, 2 * (i + 1), 2 * (i + 1) + 1, 2 * i + 1]
            ke, _, _ = mitc4_shell_matrices(pts[q], e_pa, nu, t, rho * t)
            dofs = [6 * nid + d for nid in q for d in range(6)]
            K[np.ix_(dofs, dofs)] += ke

        P = 100.0
        F = np.zeros(n_dof, dtype=float)
        F[6 * (2 * n_el) + 1] = 0.5 * P
        F[6 * (2 * n_el + 1) + 1] = 0.5 * P
        fixed = [d for nid in (0, 1) for d in range(6 * nid, 6 * nid + 6)]
        free = [d for d in range(n_dof) if d not in fixed]
        u = np.zeros(n_dof, dtype=float)
        u[free] = solve(K[np.ix_(free, free)], F[free], assume_a="pos")

        v_tip = 0.5 * (u[6 * (2 * n_el) + 1] + u[6 * (2 * n_el + 1) + 1])
        I_z = t * (h_strip**3) / 12.0
        v_tim = P * (L**3) / (3.0 * e_pa * I_z) + P * L / ((5.0 / 6.0) * g_pa * t * h_strip)
        ratio = v_tip / v_tim
        assert 0.95 <= ratio <= 1.02, f"Aspect {aspect}: v_qm6/v_tim={ratio:.4f}"

    # 2. Distorted 5-element 3D tilted membrane patch test (free interior u, v, theta_z)
    pts_2d = np.array(
        [
            [0.0, 0.0],
            [0.40, 0.0],
            [0.40, 0.40],
            [0.0, 0.40],
            [0.08, 0.09],
            [0.29, 0.07],
            [0.32, 0.30],
            [0.10, 0.32],
        ],
        dtype=float,
    )
    pts_3d = np.column_stack([pts_2d[:, 0], 0.8 * pts_2d[:, 1], 0.6 * pts_2d[:, 1]])
    elems = [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (4, 5, 6, 7)]
    K_p = np.zeros((48, 48), dtype=float)
    for q in elems:
        ke, _, _ = mitc4_shell_matrices(pts_3d[list(q)], e_pa, nu, t, rho * t)
        dofs = [6 * nid + d for nid in q for d in range(6)]
        K_p[np.ix_(dofs, dofs)] += ke

    r_p, _, _ = mitc4_local_frame(pts_3d[:4])
    t_all = np.zeros((48, 48), dtype=float)
    for nid in range(8):
        t_all[6 * nid : 6 * nid + 3, 6 * nid : 6 * nid + 3] = r_p
        t_all[6 * nid + 3 : 6 * nid + 6, 6 * nid + 3 : 6 * nid + 6] = r_p
    K_loc = t_all @ K_p @ t_all.T

    eps_x, eps_y, gamma_xy, spin = 1.0e-3, -nu * 1.0e-3, 0.8e-3, 0.4e-3
    u_exact = np.zeros(48, dtype=float)
    for nid, (xk, yk) in enumerate(pts_2d):
        u_l = eps_x * xk + (0.5 * gamma_xy - spin) * yk
        v_l = eps_y * yk + (0.5 * gamma_xy + spin) * xk
        u_exact[6 * nid : 6 * nid + 3] = u_l * r_p[0] + v_l * r_p[1]
        u_exact[6 * nid + 3 : 6 * nid + 6] = spin * r_p[2]
    u_ex_loc = t_all @ u_exact
    presc = list(range(24)) + [6 * nid + d for nid in range(4, 8) for d in (2, 3, 4)]
    free_p = [d for d in range(48) if d not in presc]
    u_sol_loc = u_ex_loc.copy()
    u_sol_loc[free_p] = solve(
        K_loc[np.ix_(free_p, free_p)], -K_loc[np.ix_(free_p, presc)] @ u_ex_loc[presc], assume_a="pos"
    )
    assert np.max(np.abs(u_sol_loc - u_ex_loc)) < 1e-12


def test_b20_mitc4_drilling_stabilization_and_six_rbms() -> None:
    """_ALPHA_DRILL=1e-3 provides meaningful drilling stiffness while preserving 6 zero RBMs."""
    assert _ALPHA_DRILL == pytest.approx(1e-3, rel=1e-12)

    e_pa = 68.9e9
    nu = 0.33
    t = 0.003
    rho = 2700.0
    dist_2d = np.array([[0.0, 0.0], [0.27, 0.02], [0.22, 0.24], [-0.03, 0.21]], dtype=float)
    quad_dist = np.column_stack([dist_2d[:, 0], 0.8 * dist_2d[:, 1], 0.6 * dist_2d[:, 1]])
    K1, _, _ = mitc4_shell_matrices(quad_dist, e_pa, nu, t, rho * t)

    evals = np.linalg.eigvalsh(K1)
    assert int(np.sum(np.abs(evals) < 1e-5)) == 6
    assert evals[6] > 1e-2

    r_mat, _, pc = mitc4_local_frame(quad_dist)
    k_norm = float(np.linalg.norm(K1))
    for rv in r_mat:
        v_rot = np.zeros(24, dtype=float)
        for k in range(4):
            v_rot[6 * k : 6 * k + 3] = np.cross(rv, quad_dist[k] - pc)
            v_rot[6 * k + 3 : 6 * k + 6] = rv
        assert abs(0.5 * float(v_rot @ K1 @ v_rot)) / (k_norm * float(v_rot @ v_rot)) < 1e-14

    # Clamped 8x8 3mm Al plate center drilling torque stiffness (vs ~0.038 Nm/rad pre-audit)
    a = 0.40
    nx = ny = 8
    pts = np.array(
        [[i * a / nx, j * a / ny, 0.0] for j in range(ny + 1) for i in range(nx + 1)],
        dtype=float,
    )
    n_dof = 6 * len(pts)
    K_plate = np.zeros((n_dof, n_dof), dtype=float)
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            q = [n0, n0 + 1, n0 + nx + 2, n0 + nx + 1]
            ke, _, _ = mitc4_shell_matrices(pts[q], e_pa, nu, t, rho * t)
            dofs = [6 * nid + d for nid in q for d in range(6)]
            K_plate[np.ix_(dofs, dofs)] += ke

    fixed = [
        d
        for j in range(ny + 1)
        for i in range(nx + 1)
        if i in (0, nx) or j in (0, ny)
        for d in range(6 * (j * (nx + 1) + i), 6 * (j * (nx + 1) + i) + 6)
    ]
    free = [d for d in range(n_dof) if d not in set(fixed)]
    center_node = 4 * (nx + 1) + 4
    F = np.zeros(n_dof, dtype=float)
    F[6 * center_node + 5] = 1.0
    u = np.zeros(n_dof, dtype=float)
    u[free] = solve(K_plate[np.ix_(free, free)], F[free], assume_a="pos")
    k_drill_center = 1.0 / float(u[6 * center_node + 5])
    assert k_drill_center > 2.0
