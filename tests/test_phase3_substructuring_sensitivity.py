"""Phase 3 (v1.1) Physics Verification Tests:
- Craig-Bampton CMS & Virtual Point Transformation (superelement.py)
- Toolhead-Hotend-Nozzle RCSA / LM-FBS (rcsa.py)
- Exact Analytical Adjoint & Fox-Kapoor Modal Sensitivities (sensitivity.py)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse
from scipy.linalg import eigh

from printsim.fea import assemble_global_matrices, solve_modal_fea, solve_static_fea
from printsim.fea.beam3d import (
    beam_rotation_matrix,
    local_beam_stiffness,
    transformation_12x12,
)
from printsim.fea.rcsa import (
    analyze_toolhead_rcsa,
    beam_tip_receptance_4x4,
    lm_fbs_couple,
)
from printsim.fea.sensitivity import compute_structural_sensitivities
from printsim.fea.solver import _build_factorized_solver
from printsim.fea.superelement import (
    _BRACKET_CACHE,
    assemble_superelement_into_global,
    condense_craig_bampton,
    condense_parametric_bracket,
    condense_solid_patch_to_vp,
    virtual_point_filter_rbe3,
    virtual_point_transform,
)
from printsim.geometry import AssemblyModel, StructuralMember, StructuralNode, build_assembly
from printsim.schema import PrinterSpec, TOOLHEAD_PRESETS, get_material, get_profile


def test_craig_bampton_cantilever_exactness_dense_and_sparse() -> None:
    """Split a 10-element cantilever into two substructures and verify first 3 modes < 0.5%."""
    mat = get_material("aluminum_6063_t5")
    prof = get_profile("extrusion_2040")  # Asymmetric Iy != Iz
    length = 0.6
    n_elems = 10

    nodes_full = [
        StructuralNode(
            id=f"n{i}",
            position_m=np.array([length * i / n_elems, 0.0, 0.0]),
            fixed=(i == 0),
            role="toolhead" if i == n_elems else "frame",
        )
        for i in range(n_elems + 1)
    ]
    mems_full = [
        StructuralMember(
            id=f"m{i}",
            start_id=f"n{i}",
            end_id=f"n{i + 1}",
            profile=prof,
            material=mat,
        )
        for i in range(n_elems)
    ]
    asm_full = AssemblyModel(
        spec=PrinterSpec(),
        nodes=nodes_full,
        members=mems_full,
        toolhead_node_id=f"n{n_elems}",
    )
    K_full, M_full, _, _, free_full = assemble_global_matrices(asm_full)
    eigvals_mono, _ = eigh(
        K_full[np.ix_(free_full, free_full)],
        M_full[np.ix_(free_full, free_full)],
        subset_by_index=[0, 2],
    )
    freqs_mono = np.sqrt(eigvals_mono) / (2.0 * math.pi)

    # Substructure A: elements 0..4 (nodes 0..5, node 0 fixed)
    asm_a = AssemblyModel(
        spec=PrinterSpec(),
        nodes=nodes_full[:6],
        members=mems_full[:5],
        toolhead_node_id="n5",
    )
    K_a, M_a, _, _, _ = assemble_global_matrices(asm_a)

    # Substructure B: elements 5..9 (nodes 5..10, interface at local node 0 -> DOFs 0..5)
    nodes_b = [
        StructuralNode(
            id=f"n{i}",
            position_m=np.array([length * i / n_elems, 0.0, 0.0]),
            fixed=False,
        )
        for i in range(5, 11)
    ]
    asm_b = AssemblyModel(
        spec=PrinterSpec(),
        nodes=nodes_b,
        members=mems_full[5:],
        toolhead_node_id="n10",
    )
    K_b, M_b, _, _, _ = assemble_global_matrices(asm_b)

    for use_sparse in (False, True):
        Kb_in = sparse.csc_matrix(K_b) if use_sparse else K_b
        Mb_in = sparse.csc_matrix(M_b) if use_sparse else M_b
        se_b = condense_craig_bampton(
            Kb_in,
            Mb_in,
            boundary_dofs=list(range(6)),
            n_modes=4,
            name="cantilever_B",
        )
        assert se_b.K_cb.shape == (10, 10)
        assert se_b.M_cb.shape == (10, 10)
        assert np.allclose(se_b.K_cb, se_b.K_cb.T, atol=1e-10)
        assert np.allclose(se_b.M_cb, se_b.M_cb.T, atol=1e-10)

        K_aug, M_aug = assemble_superelement_into_global(
            K_a, M_a, se_b, boundary_global_dofs=list(range(30, 36))
        )
        free_aug = list(range(6, K_aug.shape[0]))
        eigvals_cb, _ = eigh(
            K_aug[np.ix_(free_aug, free_aug)],
            M_aug[np.ix_(free_aug, free_aug)],
            subset_by_index=[0, 2],
        )
        freqs_cb = np.sqrt(eigvals_cb) / (2.0 * math.pi)
        rel_err = np.abs(freqs_cb - freqs_mono) / freqs_mono
        assert np.all(rel_err < 0.005), f"CB frequencies {freqs_cb} vs monolithic {freqs_mono}"


def test_vpt_rigid_body_invariance_and_rbe3_filter() -> None:
    """Verify VPT reproduces arbitrary rigid translations and small rotations to < 1e-12."""
    rng = np.random.default_rng(42)
    patch_coords = rng.uniform(-0.03, 0.04, size=(12, 3))
    vp_coord = np.array([0.01, -0.005, 0.008], dtype=float)

    R_u = virtual_point_transform(patch_coords, vp_coord)
    assert R_u.shape == (36, 6)

    q_rigid = np.array([0.0012, -0.0007, 0.0023, 0.004, -0.0025, 0.0061], dtype=float)
    t_vec = q_rigid[:3]
    theta = q_rigid[3:]
    u_exact = np.array([t_vec + np.cross(theta, r - vp_coord) for r in patch_coords]).ravel()
    u_vpt = R_u @ q_rigid
    assert np.max(np.abs(u_vpt - u_exact)) < 1e-12

    # Unweighted and weighted RBE3 filters must invert R_u exactly
    T_u = virtual_point_filter_rbe3(R_u)
    assert T_u.shape == (6, 36)
    assert np.max(np.abs(T_u @ R_u - np.eye(6))) < 1e-12
    assert np.max(np.abs(T_u @ u_exact - q_rigid)) < 1e-12

    weights = rng.uniform(0.5, 3.0, size=12)
    T_w = virtual_point_filter_rbe3(R_u, weights=weights)
    assert np.max(np.abs(T_w @ R_u - np.eye(6))) < 1e-12
    assert np.max(np.abs(T_w @ u_exact - q_rigid)) < 1e-12


def test_parametric_bracket_condensation_spd_scaling_and_cache(tmp_path: Path) -> None:
    """Condense ABS vs CNC Al 6061 bracket: check (18,18), SPD, E scaling, and cache hit."""
    _BRACKET_CACHE.clear()
    params = {
        "bracket_type": "corner_gusset",
        "arm_length_mm": 40.0,
        "height_mm": 20.0,
        "wall_thickness_mm": 6.0,
        "rib_thickness_mm": 5.0,
        "bolt_hole_mm": 5.2,
        "fillet_mm": 2.0,
        "n_modes": 6,
        "fdm_anisotropic": False,
    }
    se_abs = condense_parametric_bracket(params, "abs_printed", cache_dir=tmp_path)
    se_al = condense_parametric_bracket(params, "aluminum_6061_t6", cache_dir=tmp_path)

    assert se_abs.K_cb.shape == (18, 18)
    assert se_abs.M_cb.shape == (18, 18)
    assert se_abs.n_boundary_dofs == 12
    assert se_abs.n_internal_modes == 6
    assert len(se_abs.fixed_interface_freqs_hz) == 6
    assert se_abs.mass_kg > 0.0

    # Symmetry & definiteness checks
    for se in (se_abs, se_al):
        assert np.allclose(se.K_cb, se.K_cb.T, atol=1e-10)
        assert np.allclose(se.M_cb, se.M_cb.T, atol=1e-10)
        m_eigs = np.linalg.eigvalsh(se.M_cb)
        assert np.all(m_eigs > 0.0), "Reduced mass matrix M_cb must be strictly SPD"

        # 6 rigid-body zero modes + 12 strictly positive deformation modes on free-free K_cb
        k_eigs_all = np.linalg.eigvalsh(se.K_cb)
        k_scale = float(np.max(k_eigs_all))
        assert np.all(k_eigs_all[:6] / k_scale < 1e-9)
        assert np.all(k_eigs_all[6:] > 0.0)

        # Grounded Port 0 (DOFs 0..5 fixed) -> K_cb[6:, 6:] must be strictly SPD
        k_grounded_eigs = np.linalg.eigvalsh(se.K_cb[6:, 6:])
        assert np.all(k_grounded_eigs > 0.0)

    # Linear scaling with Young's modulus E (ABS nu=0.35 vs Al 6061 nu=0.33 -> ~5% Poisson shift)
    e_ratio = (
        get_material("aluminum_6061_t6").youngs_modulus_pa
        / get_material("abs_printed").youngs_modulus_pa
    )
    kz_ratio = se_al.port_stiffness_diag["k_z_n_m"] / se_abs.port_stiffness_diag["k_z_n_m"]
    kx_ratio = se_al.port_stiffness_diag["k_x_n_m"] / se_abs.port_stiffness_diag["k_x_n_m"]
    krx_ratio = se_al.port_stiffness_diag["k_rx_nm_rad"] / se_abs.port_stiffness_diag["k_rx_nm_rad"]
    assert kz_ratio == pytest.approx(e_ratio, rel=0.05)
    assert kx_ratio == pytest.approx(e_ratio, rel=0.05)
    assert krx_ratio == pytest.approx(e_ratio, rel=0.05)

    # Exact linear scaling check at fixed Poisson ratio via condense_solid_patch_to_vp
    mat_1x = get_material("aluminum_6061_t6")
    mat_2x = mat_1x.model_copy(update={"youngs_modulus_pa": 2.0 * mat_1x.youngs_modulus_pa})
    from printsim.fea.solid_fem import constitutive_matrix_3d, hex8_element_stiffness
    from printsim.fea.superelement import hex8_element_mass

    h_bar = 0.005
    nx_b, ny_b, nz_b = 4, 2, 2
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
    n_nds = len(uk)
    ndof = 3 * n_nds
    gcoords = np.stack(
        [uk // ((ny_b + 1) * (nz_b + 1)), (uk % ((ny_b + 1) * (nz_b + 1))) // (nz_b + 1), uk % (nz_b + 1)],
        axis=1,
    )
    coords_m = gcoords.astype(float) * h_bar
    edofs = np.empty((len(occ_b), 24), dtype=int)
    edofs[:, 0::3] = 3 * enodes
    edofs[:, 1::3] = 3 * enodes + 1
    edofs[:, 2::3] = 3 * enodes + 2
    rows_b = np.repeat(edofs, 24, axis=1).ravel()
    cols_b = np.tile(edofs, (1, 24)).ravel()
    Me_b = hex8_element_mass(mat_1x.density_kg_m3, h_bar)
    M_b = sparse.coo_matrix((np.tile(Me_b.ravel(), len(occ_b)), (rows_b, cols_b)), shape=(ndof, ndof)).tocsc()
    p0_n = np.where(gcoords[:, 0] == 0)[0]
    p1_n = np.where(gcoords[:, 0] == nx_b)[0]
    ports = [
        (p0_n, coords_m[p0_n], np.mean(coords_m[p0_n], axis=0)),
        (p1_n, coords_m[p1_n], np.mean(coords_m[p1_n], axis=0)),
    ]

    Ke_1, _ = hex8_element_stiffness(constitutive_matrix_3d(mat_1x), h_bar)
    K_b1 = sparse.coo_matrix((np.tile(Ke_1.ravel(), len(occ_b)), (rows_b, cols_b)), shape=(ndof, ndof)).tocsc()
    Ke_2, _ = hex8_element_stiffness(constitutive_matrix_3d(mat_2x), h_bar)
    K_b2 = sparse.coo_matrix((np.tile(Ke_2.ravel(), len(occ_b)), (rows_b, cols_b)), shape=(ndof, ndof)).tocsc()

    se_b1 = condense_solid_patch_to_vp(K_b1, M_b, ports, n_modes=4)
    se_b2 = condense_solid_patch_to_vp(K_b2, M_b, ports, n_modes=4)
    assert np.allclose(se_b2.K_cb[:12, :12], 2.0 * se_b1.K_cb[:12, :12], rtol=1e-10, atol=1e-6)
    assert np.allclose(
        np.diag(se_b2.K_cb[12:, 12:]),
        2.0 * np.diag(se_b1.K_cb[12:, 12:]),
        rtol=1e-10,
    )
    eig1 = np.linalg.eigvalsh(se_b1.K_cb)
    eig2 = np.linalg.eigvalsh(se_b2.K_cb)
    assert np.max(np.abs(eig1[:6])) < 1e-12 * np.max(eig1)
    assert np.max(np.abs(eig2[:6])) < 1e-12 * np.max(eig2)
    assert np.allclose(eig2[6:], 2.0 * eig1[6:], rtol=1e-7)
    for key in se_b1.port_stiffness_diag:
        assert se_b2.port_stiffness_diag[key] == pytest.approx(
            2.0 * se_b1.port_stiffness_diag[key], rel=1e-10
        )
    assert np.allclose(se_b2.M_cb[:12, :12], se_b1.M_cb[:12, :12], rtol=1e-10)
    # Verify disk cache hit (clear memory cache first to force disk reload)
    _BRACKET_CACHE.clear()
    se_abs_disk_hit = condense_parametric_bracket(params, "abs_printed", cache_dir=tmp_path)
    assert se_abs_disk_hit.source_hash == se_abs.source_hash
    assert np.array_equal(se_abs_disk_hit.K_cb, se_abs.K_cb)
    assert np.array_equal(se_abs_disk_hit.M_cb, se_abs.M_cb)
    assert np.array_equal(se_abs_disk_hit.T_cb, se_abs.T_cb)

    # Verify in-memory cache hit
    se_abs_mem_hit = condense_parametric_bracket(params, "abs_printed", cache_dir=None)
    assert np.array_equal(se_abs_mem_hit.K_cb, se_abs.K_cb)


def test_fox_kapoor_and_adjoint_sensitivity_finite_difference() -> None:
    """Verify analytical Fox-Kapoor & adjoint sensitivities vs ±0.5% central FD on Voron Trident."""
    spec = PrinterSpec.from_yaml("examples/voron_trident_300.yaml")
    asm = build_assembly(spec)
    static_res = solve_static_fea(asm)
    modal_res = solve_modal_fea(asm)
    report = compute_structural_sensitivities(asm, static_res=static_res, modal_res=modal_res)

    # Verify report structure, compliance sum, and heat_score bounds
    assert set(report.panel_bay_sensitivities.keys()) == {"left", "right", "rear", "top"}
    assert len(report.top_stiffen_targets) == 5
    assert len(report.top_lighten_targets) == 3

    comp_sum = sum(v["compliance_share_pct"] for v in report.member_sensitivities.values()) + sum(
        v["compliance_share_pct"] for v in report.panel_bay_sensitivities.values()
    )
    assert comp_sum == pytest.approx(100.0, abs=0.25)

    for d in report.member_sensitivities.values():
        assert -1.0 <= d["heat_score"] <= 1.0

    # Moving X-gantry beam must have strictly NEGATIVE mass_sens_hz (lightening helps!)
    assert report.member_sensitivities["x_gantry_l"]["mass_sens_hz"] < -0.1
    assert report.member_sensitivities["x_gantry_r"]["mass_sens_hz"] < -0.1

    # Perturb frame column col_rl_low E and rho by ±0.5% and check central FD < 1%
    col_id = "col_rl_low"
    eps = 0.005
    dln = math.log(1.0 + eps) - math.log(1.0 - eps)

    def _eval_perturbed(e_scale: float = 1.0, rho_scale: float = 1.0) -> tuple[float, float, float]:
        asm_p = build_assembly(spec)
        for mem in asm_p.members:
            if mem.id == col_id:
                mem.material = mem.material.model_copy(
                    update={
                        "youngs_modulus_pa": mem.material.youngs_modulus_pa * e_scale,
                        "density_kg_m3": mem.material.density_kg_m3 * rho_scale,
                    }
                )
        mod_p = solve_modal_fea(asm_p)
        K_p, _, _, nidx, free_p = assemble_global_matrices(asm_p)
        solve_rhs = _build_factorized_solver(K_p, free_p)
        noz_i = nidx[asm_p.nozzle_node_id]
        bed_i = nidx[asm_p.bed_node_id]
        cog_i = nidx[asm_p.toolhead_cog_node_id]
        n_dof = K_p.shape[0]
        accel = asm_p.spec.kinematics.max_accel_mm_s2 * 1e-3

        F_x = np.zeros(n_dof)
        F_x[6 * cog_i + 0] = asm_p.x_moving_mass_kg * accel
        F_y = np.zeros(n_dof)
        th_share = asm_p.spec.toolhead.total_mass_g * 1e-3 / asm_p.y_moving_mass_kg
        F_y[6 * cog_i + 1] = asm_p.y_moving_mass_kg * accel * th_share
        other_g = [g for g in asm_p.gantry_node_ids if g != asm_p.toolhead_node_id]
        for gid in other_g:
            F_y[6 * nidx[gid] + 1] += asm_p.y_moving_mass_kg * accel * (1.0 - th_share) / len(other_g)

        u_x = solve_rhs(F_x)
        u_y = solve_rhs(F_y)
        u_rel_x = float(u_x[6 * noz_i + 0] - u_x[6 * bed_i + 0])
        u_rel_y = float(u_y[6 * noz_i + 1] - u_y[6 * bed_i + 1])
        return mod_p.fundamental_freq_hz, u_rel_x, u_rel_y

    f_ep, ux_ep, uy_ep = _eval_perturbed(e_scale=1.0 + eps)
    f_em, ux_em, uy_em = _eval_perturbed(e_scale=1.0 - eps)
    fd_stiff_hz = (f_ep - f_em) / dln
    fd_dux = (ux_ep - ux_em) / dln
    fd_duy = (uy_ep - uy_em) / dln

    col_sens = report.member_sensitivities[col_id]
    assert col_sens["stiffness_sens_hz"] == pytest.approx(fd_stiff_hz, rel=0.01)
    assert col_sens["du_rel_x_dlnE_m"] == pytest.approx(fd_dux, rel=0.01)
    assert col_sens["du_rel_y_dlnE_m"] == pytest.approx(fd_duy, rel=0.01)
    assert col_sens["du_rel_dlnE_m"] == pytest.approx(fd_dux + fd_duy, rel=0.01)

    f_rp, _, _ = _eval_perturbed(rho_scale=1.0 + eps)
    f_rm, _, _ = _eval_perturbed(rho_scale=1.0 - eps)
    fd_mass_hz = (f_rp - f_rm) / dln
    assert col_sens["mass_sens_hz"] == pytest.approx(fd_mass_hz, rel=0.01)


def test_dailey_repeated_root_symmetric_beam() -> None:
    """Verify sensitivity handles repeated eigenvalues (Iy == Iz symmetric square beam)."""
    mat = get_material("aluminum_6063_t5")
    prof = get_profile("extrusion_2020")  # Iy == Iz -> degenerate X/Y modes
    nodes = [
        StructuralNode("b0", np.array([0.0, 0.0, 0.0]), fixed=True, role="foot"),
        StructuralNode("mid", np.array([0.0, 0.0, 0.25]), role="frame"),
        StructuralNode("toolhead", np.array([0.0, 0.0, 0.50]), lumped_mass_kg=0.3, role="toolhead"),
    ]
    members = [
        StructuralMember("col_bot", "b0", "mid", prof, mat, role="frame"),
        StructuralMember("col_top", "mid", "toolhead", prof, mat, role="frame"),
    ]
    asm = AssemblyModel(spec=PrinterSpec(), nodes=nodes, members=members, toolhead_node_id="toolhead")
    rep = compute_structural_sensitivities(asm)
    assert rep.fundamental_freq_hz > 10.0
    assert rep.member_sensitivities["col_bot"]["stiffness_sens_hz"] > 0.0
    assert rep.member_sensitivities["col_top"]["mass_sens_hz"] < 0.0


def test_rcsa_timoshenko_beam_and_toolhead_whip_physics() -> None:
    """Verify 4x4 Timoshenko receptance, LM-FBS coupling, and monotonic whip reduction."""
    # 1. Exact static & modal cantilever check for beam_tip_receptance_4x4
    L, d, E, nu, rho = 0.20, 0.010, 200.0e9, 0.30, 7850.0
    G = E / (2.0 * (1.0 + nu))
    A = math.pi * (0.5 * d) ** 2
    I = 0.25 * math.pi * (0.5 * d) ** 4
    kappa = 5.0 / 6.0

    H0 = beam_tip_receptance_4x4(L, E, I, G, A, rho, 0.01, zeta=0.0, kappa=kappa)
    assert H0.shape == (4, 4)
    assert np.max(np.abs(H0 - H0.T)) < 1e-8
    H_tip = H0[2:, 2:] - H0[2:, :2] @ np.linalg.solve(H0[:2, :2], H0[:2, 2:])
    exact_tip_uu = (L**3) / (3.0 * E * I) + L / (kappa * G * A)
    assert abs(H_tip[0, 0].real - exact_tip_uu) / exact_tip_uu < 1e-5

    # Two half-beams coupled via lm_fbs_couple must match full-length beam receptance
    freqs_chk = np.array([40.0, 120.0, 250.0])
    H_full = beam_tip_receptance_4x4(L, E, I, G, A, rho, freqs_chk, zeta=0.02, kappa=kappa)
    H_half = beam_tip_receptance_4x4(0.5 * L, E, I, G, A, rho, freqs_chk, zeta=0.02, kappa=kappa)
    B_mid = np.zeros((2, 8))
    B_mid[0, 2], B_mid[0, 4] = 1.0, -1.0
    B_mid[1, 3], B_mid[1, 5] = 1.0, -1.0
    H_cpl = lm_fbs_couple([H_half, H_half], B_mid)
    keep = [0, 1, 6, 7]
    H_ends = H_cpl[:, np.ix_(keep, keep)[0], np.ix_(keep, keep)[1]]
    assert np.allclose(H_ends, H_full, rtol=1e-6)

    # 2. Toolhead RCSA on Voron Trident (baseline vs 2x mount stiffness vs Al 6061)
    spec = PrinterSpec.from_yaml("examples/voron_trident_300.yaml")
    asm = build_assembly(spec)
    mod = solve_modal_fea(asm)

    rcsa_1x = analyze_toolhead_rcsa(asm, mod, mount_stiffness_scale=1.0)
    rcsa_2x = analyze_toolhead_rcsa(asm, mod, mount_stiffness_scale=2.0)

    # Also test doubling toolhead_mount member stiffness directly on assembly.members
    asm_doubled = build_assembly(spec)
    for m in asm_doubled.members:
        if m.id == "toolhead_mount":
            m.material = m.material.model_copy(
                update={"youngs_modulus_pa": m.material.youngs_modulus_pa * 2.0}
            )
    rcsa_mem_2x = analyze_toolhead_rcsa(asm_doubled, mod)

    assert rcsa_1x.nozzle_to_sensor_amp_ratio >= 1.0
    assert rcsa_2x.nozzle_to_sensor_amp_ratio >= 1.0
    assert rcsa_2x.nozzle_to_sensor_amp_ratio < rcsa_1x.nozzle_to_sensor_amp_ratio
    assert rcsa_mem_2x.nozzle_to_sensor_amp_ratio < rcsa_1x.nozzle_to_sensor_amp_ratio
    assert rcsa_2x.hotend_bending_stiffness_n_per_mm > rcsa_1x.hotend_bending_stiffness_n_per_mm
    assert len(rcsa_1x.freqs_hz) == len(rcsa_1x.sensor_frf_mag) == len(rcsa_1x.nozzle_frf_mag)

    # Heavy UHF long ABS toolhead should trigger Nozzle-to-ADXL345 whip risk (> 1.35x)
    spec_whip = spec.model_copy(deep=True)
    heavy = TOOLHEAD_PRESETS["heavy_uhf_long"]
    spec_whip.toolhead.total_mass_g = heavy["total_mass_g"]
    spec_whip.toolhead.cog_offset_mm = tuple(heavy["cog_offset_mm"])
    spec_whip.toolhead.nozzle_offset_mm = tuple(heavy["nozzle_offset_mm"])
    spec_whip.toolhead.mount_material = "abs_printed"
    asm_whip = build_assembly(spec_whip)
    mod_whip = solve_modal_fea(asm_whip)
    rcsa_whip = analyze_toolhead_rcsa(asm_whip, mod_whip)
    assert rcsa_whip.nozzle_to_sensor_amp_ratio > 1.35
    assert rcsa_whip.whip_risk is True
    assert "Nozzle-to-ADXL345 whip risk" in rcsa_whip.warning_message
    d_rcsa = rcsa_whip.to_dict()
    assert d_rcsa["whip_risk"] is True

    # Verify high-frequency long-beam Timoshenko receptance (beta*L > 14 subdivision fallback)
    H_hf = beam_tip_receptance_4x4(0.6, E, I, G, A, rho, np.array([1500.0, 2500.0]), zeta=0.02)
    assert H_hf.shape == (2, 4, 4)
    assert np.all(np.isfinite(H_hf))
    assert np.allclose(H_hf, np.swapaxes(H_hf, 1, 2), atol=1e-10)


def test_bracket_variants_rbe3_vs_rbe2_and_multi_architecture_sensitivities() -> None:
    """Verify all bracket presets, RBE3 softer than RBE2 (no boss locking), and all printer architectures."""
    for btype in ("corner_gusset", "motor_mount", "xy_joint_l_bracket"):
        se_rbe3 = condense_parametric_bracket(
            {"bracket_type": btype, "interface_mode": "rbe3"}, "ppa_cf_printed"
        )
        se_rbe2 = condense_parametric_bracket(
            {"bracket_type": btype, "interface_mode": "rbe2"}, "ppa_cf_printed"
        )
        assert se_rbe3.K_cb.shape == (18, 18)
        assert se_rbe2.K_cb.shape == (18, 18)
        # RBE3 avoids artificial boss locking so port stiffness <= rigid RBE2
        assert se_rbe3.port_stiffness_diag["k_x_n_m"] <= se_rbe2.port_stiffness_diag["k_x_n_m"] * 1.001
        assert se_rbe3.port_stiffness_diag["k_z_n_m"] <= se_rbe2.port_stiffness_diag["k_z_n_m"] * 1.001
        assert se_rbe3.to_dict()["total_dofs"] == 18

    for yaml_path in (
        "examples/voron_monolith_awd.yaml",
        "examples/cartesian_bedslinger.yaml",
        "examples/delta_racer.yaml",
    ):
        asm = build_assembly(PrinterSpec.from_yaml(yaml_path))
        rep = compute_structural_sensitivities(asm)
        rep_dict = rep.to_dict()
        assert rep_dict["fundamental_freq_hz"] > 10.0
        assert set(rep.panel_bay_sensitivities.keys()) == {"left", "right", "rear", "top"}
        comp_sum = sum(v["compliance_share_pct"] for v in rep.member_sensitivities.values()) + sum(
            v["compliance_share_pct"] for v in rep.panel_bay_sensitivities.values()
        )
        assert comp_sum == pytest.approx(100.0, abs=0.5)
        assert len(rep.top_stiffen_targets) > 0
        assert len(rep.top_lighten_targets) > 0
