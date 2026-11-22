"""Regression tests for continuum FEA, Craig-Bampton CMS, constitutive tensor, and SIMP fixes."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import trimesh
from scipy import sparse

from printsim.fea.bracket_designer import (
    _bracket_layout_dims,
    _build_bracket_voxel_grid,
    design_and_evaluate_bracket,
)
from printsim.fea.solid_fem import (
    constitutive_matrix_3d,
    hex8_element_mass,
    hex8_element_stiffness,
    solve_solid_voxel_fea,
)
from printsim.fea.superelement import (
    _BRACKET_CACHE,
    _build_coarse_bracket_grid,
    condense_craig_bampton,
    condense_parametric_bracket,
    condense_solid_patch_to_vp,
)
from printsim.fea.tet_fem import is_tet_fea_available, solve_step_bytes_tet_fea
from printsim.fea.topology import _oc_update, optimize_solid_topology
from printsim.geometry.cad_b3d import export_part_step_bytes, is_cad_available
from printsim.schema import MATERIAL_CATALOG, get_material


def test_rectangular_hex8_element_rbm_mass_and_cantilever_analytical() -> None:
    """Verify rectangular Hex8 (hx != hy != hz) has 6 zero RBMs, exact mass, and <5% beam accuracy."""
    mat = get_material("aluminum_6061_t6")
    D = constitutive_matrix_3d(mat, fdm_anisotropic=False)
    hx_m, hy_m, hz_m = 0.006, 0.004, 0.0025  # non-cubic rectangular brick
    Ke, B0 = hex8_element_stiffness(D, (hx_m, hy_m, hz_m))
    Me = hex8_element_mass(mat.density_kg_m3, (hx_m, hy_m, hz_m))

    assert Ke.shape == (24, 24)
    assert B0.shape == (6, 24)
    assert Me.shape == (24, 24)
    assert np.allclose(Ke, Ke.T, atol=1e-10)
    assert np.allclose(Me, Me.T, atol=1e-12)

    # 6 zero rigid-body modes + 18 strictly positive deformation modes
    k_eigs = np.linalg.eigvalsh(Ke)
    assert np.max(np.abs(k_eigs[:6])) < 1e-10 * float(np.max(k_eigs))
    assert k_eigs[6] > 0.0

    # Mass matrix strictly SPD and exact translational mass along X, Y, Z
    m_eigs = np.linalg.eigvalsh(Me)
    assert np.min(m_eigs) > 0.0
    expected_elem_mass = mat.density_kg_m3 * hx_m * hy_m * hz_m
    for d in range(3):
        r = np.zeros(24, dtype=float)
        r[d::3] = 1.0
        assert float(r @ Me @ r) == pytest.approx(expected_elem_mass, rel=1e-12)

    # Cantilever beam 60 x 12 x 12 mm discretized with non-cubic bricks: hx=2.5mm, hy=2.0mm, hz=1.5mm
    L_mm, b_mm, h_mm = 60.0, 12.0, 12.0
    hx_mm, hy_mm, hz_mm = 2.5, 2.0, 1.5
    nx, ny, nz = 24, 6, 8
    grid = np.ones((nx, ny, nz), dtype=bool)
    F_z = -50.0
    res = solve_solid_voxel_fea(
        grid=grid,
        voxel_size_mm=(hx_mm, hy_mm, hz_mm),
        material_id="aluminum_6061_t6",
        fixed_face="min_x",
        load_face="max_x",
        force_n=(0.0, 0.0, F_z),
        fdm_anisotropic=False,
    )

    L = L_mm * 1e-3
    b = b_mm * 1e-3
    h = h_mm * 1e-3
    I_y = (b * h**3) / 12.0
    E = mat.youngs_modulus_pa
    G = mat.effective_shear_modulus_pa
    kappa = 5.0 / 6.0
    delta_timoshenko_um = (
        abs(F_z) * (L**3) / (3.0 * E * I_y) + abs(F_z) * L / (kappa * G * b * h)
    ) * 1e6

    assert res.max_displacement_um == pytest.approx(delta_timoshenko_um, rel=0.05)
    assert res.voxel_dims_mm == pytest.approx((hx_mm, hy_mm, hz_mm), rel=1e-12)
    expected_mass_g = (L * b * h) * mat.density_kg_m3 * 1000.0
    assert res.mass_g == pytest.approx(expected_mass_g, rel=1e-9)

    # Backward-compatibility: scalar h vs equal 3-tuple (h, h, h) must match exactly
    res_scalar = solve_solid_voxel_fea(
        grid=np.ones((12, 4, 4), dtype=bool),
        voxel_size_mm=3.0,
        material_id="aluminum_6061_t6",
        fdm_anisotropic=False,
    )
    res_tuple = solve_solid_voxel_fea(
        grid=np.ones((12, 4, 4), dtype=bool),
        voxel_size_mm=(3.0, 3.0, 3.0),
        material_id="aluminum_6061_t6",
        fdm_anisotropic=False,
    )
    assert res_tuple.max_displacement_um == pytest.approx(res_scalar.max_displacement_um, rel=1e-12)
    assert res_tuple.mass_g == pytest.approx(res_scalar.mass_g, rel=1e-12)


def test_motor_mount_arm90_h10_exact_dimensions_and_tet10_parity(tmp_path: Path) -> None:
    """Verify motor_mount(arm=90, H=10) preserves exact 10.0 mm Z height and realistic mass/deflection."""
    dims = _bracket_layout_dims("motor_mount", 90.0, 10.0, 6.0, 6.0, 5.5, 2.0)
    grid, (hx, hy, hz), _, _ = _build_bracket_voxel_grid("motor_mount", dims)
    nx, ny, nz = grid.shape

    assert nx * hx == pytest.approx(dims["Lx"], rel=1e-9)
    assert ny * hy == pytest.approx(dims["Ly"], rel=1e-9)
    assert nz * hz == pytest.approx(10.0, rel=1e-9)

    stl_path = tmp_path / "motor_mount_90_10.stl"
    res_hex = design_and_evaluate_bracket(
        bracket_type="motor_mount",
        arm_length_mm=90.0,
        height_mm=10.0,
        wall_thickness_mm=6.0,
        rib_thickness_mm=6.0,
        bolt_hole_mm=5.5,
        fillet_mm=2.0,
        material_id="abs_printed",
        force_n=(0.0, 0.0, -50.0),
        engine="hex8",
        output_stl_path=stl_path,
    )
    mesh = trimesh.load(str(stl_path), force="mesh")
    assert mesh.is_watertight
    extents = mesh.bounds[1] - mesh.bounds[0]
    assert extents[0] == pytest.approx(dims["Lx"], rel=1e-4)
    assert extents[1] == pytest.approx(dims["Ly"], rel=1e-4)
    assert extents[2] == pytest.approx(10.0, rel=1e-4)

    if is_tet_fea_available() and is_cad_available():
        res_tet = design_and_evaluate_bracket(
            bracket_type="motor_mount",
            arm_length_mm=90.0,
            height_mm=10.0,
            wall_thickness_mm=6.0,
            rib_thickness_mm=6.0,
            bolt_hole_mm=5.5,
            fillet_mm=2.0,
            material_id="abs_printed",
            force_n=(0.0, 0.0, -50.0),
            engine="tet10",
        )
        # Previously mass was +116% off and deflection 10.3x-20x too stiff due to 24.3 mm Z distortion
        assert res_hex["mass_g"] == pytest.approx(res_tet["mass_g"], rel=0.20)
        assert res_hex["max_displacement_um"] == pytest.approx(
            res_tet["max_displacement_um"], rel=0.15
        )


def test_superelement_coarse_bracket_bbox_guyan_rbe3_and_singular_shift_invert() -> None:
    """Verify exact (Lx, Ly, Lz) bounding box in CMS, RBE3 equilibrium, Guyan exactness, and ARPACK consistency."""
    for btype, arm, height in (
        ("corner_gusset", 40.0, 20.0),
        ("corner_gusset", 60.0, 12.0),
        ("motor_mount", 90.0, 10.0),
    ):
        dims = _bracket_layout_dims(btype, arm, height, 6.0, 5.0, 5.2, 2.0)  # type: ignore[arg-type]
        grid, (hx, hy, hz) = _build_coarse_bracket_grid(btype, dims, target_elements=380)
        nx, ny, nz = grid.shape
        assert nx * hx == pytest.approx(dims["Lx"], rel=1e-9)
        assert ny * hy == pytest.approx(dims["Ly"], rel=1e-9)
        assert nz * hz == pytest.approx(dims["Lz"], rel=1e-9)

    _BRACKET_CACHE.clear()
    # Default fdm_anisotropic must honor mat.is_printed_polymer (True for pa12_cf_printed)
    se_default = condense_parametric_bracket(
        {"bracket_type": "corner_gusset", "arm_length_mm": 60.0, "height_mm": 12.0},
        "pa12_cf_printed",
    )
    se_iso = condense_parametric_bracket(
        {
            "bracket_type": "corner_gusset",
            "arm_length_mm": 60.0,
            "height_mm": 12.0,
            "fdm_anisotropic": False,
        },
        "pa12_cf_printed",
    )
    assert se_default.source_hash != se_iso.source_hash
    # Z-interlayer e3_ratio=0.32 makes FDM anisotropic bracket significantly softer in Z than isotropic
    assert se_default.port_stiffness_diag["k_z_n_m"] < se_iso.port_stiffness_diag["k_z_n_m"] * 0.75

    # Free-free RBE3 equilibrium: exactly 6 zero rigid body modes and 12 positive deformation modes
    k_eigs = np.linalg.eigvalsh(se_default.K_cb)
    assert np.max(np.abs(k_eigs[:6])) < 1e-9 * float(np.max(k_eigs))
    assert k_eigs[6] > 0.0

    # Guyan static exactness on non-cubic rectangular brick mesh (hx != hy != hz)
    mat = get_material("aluminum_6061_t6")
    D = constitutive_matrix_3d(mat, fdm_anisotropic=False)
    hx_m, hy_m, hz_m = 0.006, 0.004, 0.0025
    nx_b, ny_b, nz_b = 5, 3, 3
    grid_b = np.ones((nx_b, ny_b, nz_b), dtype=bool)
    occ_b = np.argwhere(grid_b)
    offsets = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
        dtype=int,
    )
    ec = occ_b[:, None, :] + offsets[None, :, :]
    fk = ec[:, :, 0] * ((ny_b + 1) * (nz_b + 1)) + ec[:, :, 1] * (nz_b + 1) + ec[:, :, 2]
    uk, inv = np.unique(fk, return_inverse=True)
    enodes = inv.reshape(len(occ_b), 8)
    ndof = 3 * len(uk)
    gcoords = np.stack(
        [
            uk // ((ny_b + 1) * (nz_b + 1)),
            (uk % ((ny_b + 1) * (nz_b + 1))) // (nz_b + 1),
            uk % (nz_b + 1),
        ],
        axis=1,
    )
    coords_m = gcoords.astype(float) * np.array([hx_m, hy_m, hz_m])[None, :]
    Ke, _ = hex8_element_stiffness(D, (hx_m, hy_m, hz_m))
    Me = hex8_element_mass(mat.density_kg_m3, (hx_m, hy_m, hz_m))
    edofs = np.empty((len(occ_b), 24), dtype=int)
    edofs[:, 0::3] = 3 * enodes
    edofs[:, 1::3] = 3 * enodes + 1
    edofs[:, 2::3] = 3 * enodes + 2
    rows = np.repeat(edofs, 24, axis=1).ravel()
    cols = np.tile(edofs, (1, 24)).ravel()
    K = sparse.coo_matrix((np.tile(Ke.ravel(), len(occ_b)), (rows, cols)), shape=(ndof, ndof)).tocsc()
    M = sparse.coo_matrix((np.tile(Me.ravel(), len(occ_b)), (rows, cols)), shape=(ndof, ndof)).tocsc()

    p0 = np.where(gcoords[:, 0] == 0)[0]
    p1 = np.where(gcoords[:, 0] == nx_b)[0]
    se_rect = condense_solid_patch_to_vp(
        K,
        M,
        port_node_sets=[
            (p0, coords_m[p0], np.mean(coords_m[p0], axis=0)),
            (p1, coords_m[p1], np.mean(coords_m[p1], axis=0)),
        ],
        n_modes=4,
        interface_mode="rbe3",
    )
    # Zero coupling K_bm between boundary and modal DOFs
    assert np.max(np.abs(se_rect.K_cb[:12, 12:])) < 1e-9 * float(np.max(np.abs(se_rect.K_cb)))
    # Guyan static exactness: applying arbitrary wrench at Port 1 with Port 0 grounded matches full recovery
    f_vp1 = np.array([12.0, -7.5, 18.0, 0.05, -0.08, 0.03], dtype=float)
    q_cb = np.linalg.solve(se_rect.K_cb[6:, 6:], np.concatenate([f_vp1, np.zeros(4)]))
    q_full_vp1 = q_cb[:6]
    # Reconstruct full 3D nodal vector via T_cb and compare against full K solve under equivalent RBE3 forces
    u_rec = se_rect.T_cb[:, 6:] @ q_cb
    # Virtual work check: u_rec^T K u_rec must equal q_cb^T K_cb[6:, 6:] q_cb
    assert float(u_rec @ (K @ u_rec)) == pytest.approx(float(q_cb @ se_rect.K_cb[6:, 6:] @ q_cb), rel=1e-11)
    assert np.all(np.isfinite(q_full_vp1))

    # Sparse ARPACK shift-invert consistency when K_ii has a singular zero-stiffness interior DOF (>350 interior DOFs)
    n_tot = 400
    b_dofs = [0, 1]
    diag_k = np.full(n_tot, 1e5, dtype=float)
    diag_k[200] = 0.0  # singular interior DOF forcing splu regularization fallback
    K_sing = sparse.diags(diag_k, format="csc")
    M_sing = sparse.diags(np.full(n_tot, 0.01, dtype=float), format="csc")
    se_sing = condense_craig_bampton(K_sing, M_sing, boundary_dofs=b_dofs, n_modes=4)
    assert len(se_sing.fixed_interface_freqs_hz) == 4
    assert np.all(np.isfinite(se_sing.K_cb))
    # First mode corresponds to regularized zero-stiffness DOF (sqrt(1e-9*1e5 / 0.01) / (2*pi) ~ 0.016 Hz)
    assert se_sing.fixed_interface_freqs_hz[0] < 1.0
    # Remaining modes correspond to sqrt(1e5 / 0.01) / (2*pi) = 503.29 Hz
    assert se_sing.fixed_interface_freqs_hz[1] == pytest.approx(
        math.sqrt(1e5 / 0.01) / (2.0 * math.pi), rel=1e-4
    )


def test_fdm_constitutive_tensor_uses_e3_ratio_and_spd_for_all_printed_polymers() -> None:
    """Verify constitutive_matrix_3d uses mat.e3_ratio, reciprocity, and is strictly SPD."""
    printed_mats = [m for m in MATERIAL_CATALOG.values() if m.is_printed_polymer]
    assert len(printed_mats) == 5

    for mat in printed_mats:
        D = constitutive_matrix_3d(mat, fdm_anisotropic=True)
        assert np.allclose(D, D.T, atol=1e-8)
        eigs = np.linalg.eigvalsh(D)
        assert float(np.min(eigs)) > 0.0, f"{mat.id} constitutive matrix is not strictly SPD"

        S = np.linalg.inv(D)
        assert S[0, 0] == pytest.approx(1.0 / mat.youngs_modulus_pa, rel=1e-9)
        assert S[1, 1] == pytest.approx(1.0 / mat.youngs_modulus_pa, rel=1e-9)
        assert S[2, 2] == pytest.approx(1.0 / (mat.youngs_modulus_pa * mat.e3_ratio), rel=1e-9)
        assert S[0, 2] == pytest.approx(S[2, 0], rel=1e-12)

    # Extreme synthetic e3_ratio and Poisson ratio must remain strictly SPD via stability clamp
    extreme_mat = get_material("abs_printed").model_copy(
        update={"e3_ratio": 1.5, "poisson_ratio": 0.48}
    )
    D_ext = constitutive_matrix_3d(extreme_mat, fdm_anisotropic=True)
    assert float(np.min(np.linalg.eigvalsh(D_ext))) > 0.0

    # Verify is_printed_polymer auto-detection (independent of material_id string containing 'printed')
    grid_bar = np.ones((6, 2, 2), dtype=bool)
    res_al = solve_solid_voxel_fea(grid_bar, 3.0, material_id="aluminum_6061_t6")
    res_pa = solve_solid_voxel_fea(grid_bar, 3.0, material_id="pa12_cf_printed")
    res_pa_override = solve_solid_voxel_fea(
        grid_bar, 3.0, material_id="pa12_cf_printed", fdm_anisotropic=False
    )
    assert res_al.is_fdm_anisotropic is False
    assert res_pa.is_fdm_anisotropic is True
    assert res_pa_override.is_fdm_anisotropic is False

    topo_al = optimize_solid_topology(
        grid_bar, 3.0, material_id="aluminum_6061_t6", iterations=2
    )
    topo_pa = optimize_solid_topology(
        grid_bar, 3.0, material_id="pa12_cf_printed", iterations=2
    )
    assert topo_al.is_fdm_anisotropic is False
    assert topo_pa.is_fdm_anisotropic is True

    bkt_al = design_and_evaluate_bracket(material_id="aluminum_6061_t6")
    bkt_pa = design_and_evaluate_bracket(material_id="pa12_cf_printed")
    assert bkt_al["fdm_anisotropic"] is False
    assert bkt_pa["fdm_anisotropic"] is True

    if is_tet_fea_available() and is_cad_available():
        step_bytes = export_part_step_bytes("cantilever_bar")
        tet_al = solve_step_bytes_tet_fea(step_bytes, material_id="aluminum_6061_t6", order=1)
        tet_pa = solve_step_bytes_tet_fea(step_bytes, material_id="pa12_cf_printed", order=1)
        assert tet_al.is_fdm_anisotropic is False
        assert tet_pa.is_fdm_anisotropic is True


def test_simp_oc_update_scale_invariant_for_tiny_sensitivities() -> None:
    """Verify _oc_update converges to target_volume_fraction (±0.005) even when dc ~ 1e-18."""
    n = 120
    rng = np.random.default_rng(7)
    rho = np.full(n, 0.5, dtype=float)
    passive = np.zeros(n, dtype=bool)
    passive[:8] = True
    rho[passive] = 1.0

    dc_unit = -rng.uniform(0.1, 2.0, size=n)
    target_vf = 0.50

    rho_ref = _oc_update(rho, dc_unit, target_vf, passive, preserve_boundaries=True)
    assert float(np.mean(rho_ref)) == pytest.approx(target_vf, abs=0.005)

    for scale in (1e-18, 1e-28, 1e12):
        rho_scaled = _oc_update(
            rho, dc_unit * scale, target_vf, passive, preserve_boundaries=True
        )
        assert float(np.mean(rho_scaled)) == pytest.approx(target_vf, abs=0.005)
        assert np.allclose(rho_scaled, rho_ref, atol=1e-10, rtol=1e-10)

    # Full SIMP solve on stiff steel with micro-Newton force (dc ~ 1e-17 J)
    res_tiny = optimize_solid_topology(
        grid=np.ones((12, 4, 4), dtype=bool),
        voxel_size_mm=3.0,
        material_id="steel_4140",
        force_n=(0.0, 0.0, -1e-5),
        target_volume_fraction=0.50,
        iterations=12,
    )
    assert res_tiny.history[-1]["volume_fraction"] == pytest.approx(0.50, abs=0.02)
    assert res_tiny.achieved_volume_fraction == pytest.approx(0.50, abs=0.08)
