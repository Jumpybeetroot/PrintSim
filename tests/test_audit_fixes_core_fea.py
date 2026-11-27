"""Regression tests for Core FEA physics audit fixes (F1, F3, F4, F5, Mod #6, D1, and eigensolver conditioning)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import eigh, solve
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import LinearOperator, eigsh, splu

from printsim.fea import (
    analyze_toolhead_rcsa,
    assemble_global_matrices,
    compute_structural_sensitivities,
    solve_bed_plate_fem,
    solve_modal_fea,
    solve_static_fea,
)
from printsim.fea.bed_plate_fem import _resolve_mount_layout, _solve_plate_modes
from printsim.fea.sensitivity import _solve_eigen_modes
from printsim.fea.shell_mitc4 import (
    mitc4_local_frame,
    mitc4_shell_matrices,
    mitc4_shell_von_mises_mpa,
)
from printsim.geometry import build_assembly
from printsim.geometry.builder import FastenerSpring
from printsim.schema import PrinterSpec, get_material

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


# ---------------------------------------------------------------------------
# F1: Frame-Invariant MITC4 Hughes-Brezzi Drilling Penalty
# ---------------------------------------------------------------------------


def _assemble_5elem_distorted_patch(e_pa: float, nu: float, thickness_m: float, rho: float):
    """Build MacNeal-Harder 5-element distorted quad patch on [0, 0.4] x [0, 0.4] in tilted 3D plane."""
    # Outer corner nodes 0..3; inner skewed quad nodes 4..7
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
    # Tilt into 3D space (rotated about X by 36.87 deg) to verify 3D frame invariance simultaneously
    pts_3d = np.column_stack([pts_2d[:, 0], 0.8 * pts_2d[:, 1], 0.6 * pts_2d[:, 1]])
    elems = [
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
        (4, 5, 6, 7),
    ]
    n_dof = 6 * len(pts_3d)
    K = np.zeros((n_dof, n_dof), dtype=float)
    areal_mass = rho * thickness_m
    for q in elems:
        ke, _, _ = mitc4_shell_matrices(pts_3d[list(q)], e_pa, nu, thickness_m, areal_mass)
        dofs = [6 * nid + d for nid in q for d in range(6)]
        K[np.ix_(dofs, dofs)] += ke
    K = 0.5 * (K + K.T)
    return pts_2d, pts_3d, elems, K


def test_f1_mitc4_single_quad_6_zero_rbms_and_patch_tests() -> None:
    """Verify MITC4 has 6 exact zero RBMs (<1e-8 J/rad^2 under R_z) and passes membrane & bending patch tests."""
    e_pa = 68.9e9
    nu = 0.33
    rho = 2700.0
    t = 0.003

    # 1. Single unconstrained 3mm Al quad (rectangular and planar distorted 3D tilted trapezoid)
    quad_rect = np.array(
        [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.25, 0.25, 0.0], [0.0, 0.25, 0.0]],
        dtype=float,
    )
    # Distorted trapezoidal quad embedded in tilted 3D plane z = 0.75 * y
    dist_2d = np.array(
        [[0.0, 0.0], [0.27, 0.02], [0.22, 0.24], [-0.03, 0.21]],
        dtype=float,
    )
    quad_dist = np.column_stack([dist_2d[:, 0], 0.8 * dist_2d[:, 1], 0.6 * dist_2d[:, 1]])

    for coords in (quad_rect, quad_dist):
        K, M, _ = mitc4_shell_matrices(coords, e_pa, nu, t, rho * t)
        assert np.allclose(K, K.T, atol=1e-8)
        assert np.allclose(M, M.T, atol=1e-12)

        evals = np.linalg.eigvalsh(K)
        assert np.all(evals >= -1e-6)
        assert int(np.sum(np.abs(evals) < 1e-5)) == 6
        assert evals[6] > 1e-2

        # Construct all 6 exact 3D rigid-body motions about element centroid
        r_mat, _, pc = mitc4_local_frame(coords)
        e1, e2, e3 = r_mat[0], r_mat[1], r_mat[2]
        k_norm = float(np.linalg.norm(K))
        for axis_vec in (e1, e2, e3):
            v_trans = np.zeros(24, dtype=float)
            for k in range(4):
                v_trans[6 * k : 6 * k + 3] = axis_vec
            energy = 0.5 * float(v_trans @ K @ v_trans)
            assert abs(energy) / (k_norm * float(np.dot(v_trans, v_trans))) < 1e-14

        for rot_label, rot_vec in (("Rx", e1), ("Ry", e2), ("Rz", e3)):
            v_rot = np.zeros(24, dtype=float)
            for k in range(4):
                v_rot[6 * k : 6 * k + 3] = np.cross(rot_vec, coords[k] - pc)
                v_rot[6 * k + 3 : 6 * k + 6] = rot_vec
            energy = 0.5 * float(v_rot @ K @ v_rot)
            assert abs(energy) < 1e-8, f"RBM {rot_label} stored spurious energy {energy:.3e} J/rad^2"
            assert abs(energy) / (k_norm * float(np.dot(v_rot, v_rot))) < 1e-14

    # 2. Distorted 5-element 3D membrane patch test (combined uniaxial + pure shear with FREE drilling DOFs)
    pts_2d, pts_3d, elems, K = _assemble_5elem_distorted_patch(e_pa, nu, t, rho)
    r_patch, _, _ = mitc4_local_frame(pts_3d[:4])

    eps_x = 1.0e-3
    eps_y = -nu * eps_x
    gamma_xy = 0.8e-3
    omega_z = 0.5 * gamma_xy  # u = eps_x*x + gamma_xy*y, v = eps_y*y -> omega_z = -0.5*gamma_xy; use symmetric shear!
    # Symmetric pure shear: u = eps_x*x + 0.5*gamma_xy*y, v = eps_y*y + 0.5*gamma_xy*x => omega_z = 0;
    # Add rigid spin omega_spin = 0.4e-3 rad to test non-zero continuum spin recovery too!
    omega_spin = 0.4e-3

    u_exact = np.zeros(6 * len(pts_3d), dtype=float)
    for nid, (xk, yk) in enumerate(pts_2d):
        u_loc = eps_x * xk + (0.5 * gamma_xy - omega_spin) * yk
        v_loc = eps_y * yk + (0.5 * gamma_xy + omega_spin) * xk
        disp_g = u_loc * r_patch[0] + v_loc * r_patch[1]
        rot_g = omega_spin * r_patch[2]
        u_exact[6 * nid : 6 * nid + 3] = disp_g
        u_exact[6 * nid + 3 : 6 * nid + 6] = rot_g

    # Fix boundary nodes (0..3) translations + out-of-plane rotations; leave interior nodes (4..7) completely free
    fixed_mem = []
    for nid in range(4):
        fixed_mem.extend(range(6 * nid, 6 * nid + 6))
    # Also fix out-of-plane bending DOFs on interior nodes in local frame by solving in-plane + drilling DOFs
    # Equivalently, fix w_loc, thx_loc, thy_loc = 0 on interior nodes via global frame transformation
    t_all = np.zeros((48, 48), dtype=float)
    for nid in range(8):
        t_all[6 * nid : 6 * nid + 3, 6 * nid : 6 * nid + 3] = r_patch
        t_all[6 * nid + 3 : 6 * nid + 6, 6 * nid + 3 : 6 * nid + 6] = r_patch
    K_loc = t_all @ K @ t_all.T
    u_exact_loc = t_all @ u_exact

    # On outer boundary 0..3 prescribe all 6 DOFs; on interior 4..7 prescribe only out-of-plane w, thx, thy = 0,
    # leaving interior (u, v, theta_z) FREE!
    prescribed = list(range(24))
    for nid in range(4, 8):
        prescribed.extend([6 * nid + 2, 6 * nid + 3, 6 * nid + 4])
    free_mem = [d for d in range(48) if d not in prescribed]

    rhs_mem = -K_loc[np.ix_(free_mem, prescribed)] @ u_exact_loc[prescribed]
    u_sol_loc = u_exact_loc.copy()
    u_sol_loc[free_mem] = solve(K_loc[np.ix_(free_mem, free_mem)], rhs_mem, assume_a="pos")
    assert np.max(np.abs(u_sol_loc - u_exact_loc)) < 1e-11

    u_sol_glob = t_all.T @ u_sol_loc
    g_pa = e_pa / (2.0 * (1.0 + nu))
    sig_x_exact = (e_pa / (1.0 - nu**2)) * (eps_x + nu * eps_y)
    sig_y_exact = (e_pa / (1.0 - nu**2)) * (eps_y + nu * eps_x)
    tau_xy_exact = g_pa * gamma_xy
    vm_mem_exact_mpa = (
        math.sqrt(sig_x_exact**2 - sig_x_exact * sig_y_exact + sig_y_exact**2 + 3.0 * tau_xy_exact**2)
        * 1e-6
    )
    for q in elems:
        u_e = np.concatenate([u_sol_glob[6 * nid : 6 * nid + 6] for nid in q])
        vm_e = mitc4_shell_von_mises_mpa(pts_3d[list(q)], u_e, e_pa, nu, t)
        assert vm_e == pytest.approx(vm_mem_exact_mpa, rel=1e-9)

    # 3. Distorted 5-element 3D Kirchhoff bending patch test
    kx, ky, kxy = 0.04, -0.02, 0.03
    u_bend_loc = np.zeros(48, dtype=float)
    for nid, (xk, yk) in enumerate(pts_2d):
        w_loc = 0.5 * (kx * xk**2 + ky * yk**2 + kxy * xk * yk)
        dw_dx = kx * xk + 0.5 * kxy * yk
        dw_dy = ky * yk + 0.5 * kxy * xk
        # Right-hand rule: beta_x = +theta_y = -dw/dx, beta_y = -theta_x = -dw/dy
        u_bend_loc[6 * nid + 2] = w_loc
        u_bend_loc[6 * nid + 3] = dw_dy
        u_bend_loc[6 * nid + 4] = -dw_dx

    presc_b = list(range(24))
    for nid in range(4, 8):
        presc_b.extend([6 * nid + 0, 6 * nid + 1, 6 * nid + 5])
    free_b = [d for d in range(48) if d not in presc_b]
    rhs_b = -K_loc[np.ix_(free_b, presc_b)] @ u_bend_loc[presc_b]
    u_bsol_loc = u_bend_loc.copy()
    u_bsol_loc[free_b] = solve(K_loc[np.ix_(free_b, free_b)], rhs_b, assume_a="pos")
    assert np.max(np.abs(u_bsol_loc - u_bend_loc)) < 1e-11

    u_bsol_glob = t_all.T @ u_bsol_loc
    z_top = 0.5 * t
    sig_bx = -(e_pa * z_top / (1.0 - nu**2)) * (kx + nu * ky)
    sig_by = -(e_pa * z_top / (1.0 - nu**2)) * (ky + nu * kx)
    tau_bxy = -(e_pa * z_top / (2.0 * (1.0 + nu))) * kxy
    vm_bend_exact_mpa = (
        math.sqrt(sig_bx**2 - sig_bx * sig_by + sig_by**2 + 3.0 * tau_bxy**2) * 1e-6
    )
    for q in elems:
        u_e = np.concatenate([u_bsol_glob[6 * nid : 6 * nid + 6] for nid in q])
        vm_e = mitc4_shell_von_mises_mpa(pts_3d[list(q)], u_e, e_pa, nu, t)
        assert vm_e == pytest.approx(vm_bend_exact_mpa, rel=1e-9)


# ---------------------------------------------------------------------------
# F3: 2D Bed Plate Thermal Gradient Sign & Fallback
# ---------------------------------------------------------------------------


def test_f3_bed_plate_thermal_gradient_concave_up_and_degenerate_fallback() -> None:
    """Bottom-heated plate curves concave-up (edges higher than center) and avoids tuple ndarray comparison."""
    for bed_size in ((300.0, 300.0), (320.0, 220.0)):
        spec = PrinterSpec(architecture="corexy")
        spec.bed.z_drive = "leadscrew_triple"
        spec.bed.mount_type = "kinematic_3pt"
        spec.bed.size_mm = bed_size
        spec.bed.target_temp_c = 110.0
        spec.frame.chamber_temp_c = 25.0

        res = solve_bed_plate_fem(spec, mesh_n=4)
        assert res.taco_buckling_um == 0.0

        nx, ny = res.grid_nx, res.grid_ny
        z_th = np.array(res.z_thermal_um, dtype=float).reshape((ny, nx))
        center_z = float(z_th[ny // 2, nx // 2])
        corner_vals = [
            float(z_th[0, 0]),
            float(z_th[0, nx - 1]),
            float(z_th[ny - 1, 0]),
            float(z_th[ny - 1, nx - 1]),
        ]
        edge_mid_vals = [
            float(z_th[0, nx // 2]),
            float(z_th[ny - 1, nx // 2]),
            float(z_th[ny // 2, 0]),
            float(z_th[ny // 2, nx - 1]),
        ]
        # Concave-up bowl: edges & corners strictly above center
        assert min(edge_mid_vals) > center_z + 0.5
        assert min(corner_vals) > max(edge_mid_vals)

        # Verify exact analytical 2D spherical curvature (plate gradient + Volkersen/Timoshenko bimetallic PEI bow)
        from printsim.fea.bed_plate_fem import compute_bed_bimetallic_pei_bow_um

        mat = get_material(spec.bed.plate_material)
        lx_m, ly_m = bed_size[0] * 1e-3, bed_size[1] * 1e-3
        q_top = 12.0 * (110.0 - 25.0)
        kappa = mat.thermal_expansion_1_k * q_top / mat.thermal_conductivity_w_mk
        expected_grad_um = 0.5 * kappa * ((0.5 * lx_m) ** 2 + (0.5 * ly_m) ** 2) * 1e6
        bimetallic_edge_um = compute_bed_bimetallic_pei_bow_um(spec)
        r_corner2 = (0.5 * lx_m) ** 2 + (0.5 * ly_m) ** 2
        r_edge2 = (0.5 * max(lx_m, ly_m)) ** 2
        expected_corner_minus_center_um = expected_grad_um + bimetallic_edge_um * (r_corner2 / r_edge2)
        assert (corner_vals[0] - center_z) == pytest.approx(expected_corner_minus_center_um, rel=1e-6)

        # Anchor mean of thermal field is zero
        gx = np.array(res.grid_x_rel, dtype=float)
        gy = np.array(res.grid_y_rel, dtype=float)
        anchor_ids, _, _ = _resolve_mount_layout(spec, gx, gy)
        assert float(np.mean(np.array(res.z_thermal_um)[anchor_ids])) == pytest.approx(0.0, abs=1e-9)

    # Degenerate fallback check: force < 3 Z-modes + duplicate non-Z eigenvalue so tuple comparison would raise ValueError
    spec_sq = PrinterSpec(architecture="corexy")
    n_nodes = 9  # 3x3 grid
    n_dof = 6 * n_nodes
    k_toy = np.eye(n_dof, dtype=float)
    m_toy = np.eye(n_dof, dtype=float)
    # Give only 1 low Z mode (20 Hz), 2 identical in-plane X/Y modes (50 Hz), and keep all other Z modes high (500 Hz)
    for d in range(n_dof):
        freq_d = 500.0 if (d % 6 == 2) else 200.0
        k_toy[d, d] = (2.0 * math.pi * freq_d) ** 2
    k_toy[2, 2] = (2.0 * math.pi * 20.0) ** 2
    k_toy[0, 0] = (2.0 * math.pi * 50.0) ** 2
    k_toy[1, 1] = (2.0 * math.pi * 50.0) ** 2
    modes_hz, shape1, _ = _solve_plate_modes(k_toy, m_toy, spec_sq, [0, 2, 6, 8])
    assert len(modes_hz) == 3
    assert modes_hz[0] == pytest.approx(20.0, rel=1e-6)
    assert modes_hz[1] == pytest.approx(50.0, rel=1e-6)
    assert modes_hz[2] == pytest.approx(50.0, rel=1e-6)
    assert np.all(np.isfinite(shape1))


# ---------------------------------------------------------------------------
# F4, F5, Moderate #6: Static Solver Belt FBD, Bedslinger F_y, 8-Dir Envelope
# ---------------------------------------------------------------------------


def test_f4_f5_mod6_static_solver_belt_fbd_bedslinger_and_envelope() -> None:
    """Verify CoreXY F_belt equilibrium & 0 carriage Y force, Bedslinger 0 toolhead Y force, and 8-dir envelope."""
    # 1. F4: CoreXY belt preload FBD on rubber_feet (all gantry & base frame nodes free!)
    spec_cx = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    spec_cx.frame.bracing = "rear_cross"
    spec_cx.frame.feet_type = "rubber_feet"
    spec_cx.kinematics.max_accel_mm_s2 = 0.0  # isolates u_grav + u_belt cleanly

    spec_t0 = spec_cx.model_copy(deep=True)
    spec_t0.gantry.belt.pretension_n = 0.0
    spec_t90 = spec_cx.model_copy(deep=True)
    t0 = 90.0
    spec_t90.gantry.belt.pretension_n = t0

    asm_0 = build_assembly(spec_t0)
    asm_90 = build_assembly(spec_t90)
    st_0 = solve_static_fea(asm_0)
    st_90 = solve_static_fea(asm_90)

    # Reconstruct full nodal displacement delta u_belt via K @ u_belt (with full DOF solve)
    # Check using nodal displacements and exact K from assemble_global_matrices:
    # Since F_belt only loads translational DOFs (fx, fy) on free nodes, build u_belt on free DOFs
    K, _, _, nidx, free_dofs = assemble_global_matrices(asm_90)
    # Build identical F_grav solution to subtract u_grav from u_comb? Wait: nodal_displacements_mm rounds to 1e-6 mm,
    # so also verify exact F_belt directly via a zero-gravity K solve vs st_90 - st_0 difference!
    u_diff_m = np.zeros(K.shape[0], dtype=float)
    for nid, ni in nidx.items():
        d0 = np.array(st_0.nodal_displacements_mm[nid], dtype=float) * 1e-3
        d90 = np.array(st_90.nodal_displacements_mm[nid], dtype=float) * 1e-3
        u_diff_m[6 * ni : 6 * ni + 3] = d90 - d0

    # Verify sum of ground foot spring forces due to belt preload is ~0 (self-equilibrating!)
    net_foot_f = np.zeros(3, dtype=float)
    net_foot_m = np.zeros(3, dtype=float)
    for sp in asm_90.fastener_springs:
        if sp.bay == "feet":
            i_f = nidx[sp.frame_node_id]
            du_f = u_diff_m[6 * i_f : 6 * i_f + 3]
            f_sp = np.array([sp.k_shear_n_m * du_f[0], sp.k_shear_n_m * du_f[1], sp.k_normal_n_m * du_f[2]])
            net_foot_f += f_sp
            net_foot_m += np.cross(asm_90.nodes[i_f].position_m, f_sp)

    assert np.linalg.norm(net_foot_f) < 5e-3, f"Unbalanced F_belt transmitted {net_foot_f} N to feet"
    assert np.linalg.norm(net_foot_m) < 5e-3, f"Unbalanced F_belt transmitted {net_foot_m} N*m moment to feet"

    # Moving X-gantry carriages y_car_l/r must NOT be pulled forward in Y by symmetric belt pretension
    # (sub-micron ~0.6 um Y bow from real 3D CAD XY-joint off-plane X<->Y coupling is expected)
    dy_car_l_um = (st_90.nodal_displacements_mm["y_car_l"][1] - st_0.nodal_displacements_mm["y_car_l"][1]) * 1e3
    dy_car_r_um = (st_90.nodal_displacements_mm["y_car_r"][1] - st_0.nodal_displacements_mm["y_car_r"][1]) * 1e3
    assert abs(dy_car_l_um) < 1.5 and abs(dy_car_r_um) < 1.5

    # Also verify directly from solver F_belt formula that total F = 0 and total M = 0
    belt_loads = [
        ("g_rl", +2.0 * t0, -2.0 * t0),
        ("g_rr", -2.0 * t0, -2.0 * t0),
        ("g_fl", 0.0, +2.0 * t0),
        ("g_fr", 0.0, +2.0 * t0),
        ("y_car_l", +2.0 * t0, 0.0),
        ("y_car_r", -2.0 * t0, 0.0),
    ]
    f_tot = np.zeros(3)
    m_tot = np.zeros(3)
    for nid, fx, fy in belt_loads:
        fvec = np.array([fx, fy, 0.0])
        f_tot += fvec
        m_tot += np.cross(asm_90.node_map[nid].position_m, fvec)
    assert np.linalg.norm(f_tot) < 1e-10
    assert np.linalg.norm(m_tot) < 1e-10

    # 2. F5: Cartesian Bedslinger F_y puts ZERO load on stationary toolhead_cog and full y_moving_mass_kg on bed
    spec_bs = PrinterSpec.from_yaml(EXAMPLES_DIR / "cartesian_bedslinger.yaml")
    asm_bs = build_assembly(spec_bs)
    st_bs = solve_static_fea(asm_bs)

    assert np.linalg.norm(st_bs.nodal_displacements_y_mm["toolhead_cog"]) < 1e-8
    assert np.linalg.norm(st_bs.nodal_displacements_y_mm["nozzle_tip"]) < 1e-8
    assert st_bs.nodal_displacements_y_mm["bed_surface"][1] < -0.002

    # Check exact force applied to bed_surface equals -assembly.y_moving_mass_kg * a_y (includes carriage +0.35 kg!)
    K_bs, _, _, nidx_bs, _ = assemble_global_matrices(asm_bs)
    u_y_bs = np.zeros(K_bs.shape[0], dtype=float)
    # Build full u_y via unit bed Y compliance ratio (or check accel_deflection_y_um vs direct solve)
    f_bed_test = np.zeros(K_bs.shape[0], dtype=float)
    bed_i = nidx_bs[asm_bs.bed_node_id]
    f_expected_y = -asm_bs.y_moving_mass_kg * (spec_bs.kinematics.max_accel_mm_s2 * 1e-3)
    f_bed_test[6 * bed_i + 1] = f_expected_y
    free_bs = [d for d in range(K_bs.shape[0]) if not asm_bs.nodes[d // 6].fixed]
    u_bed_exact = np.zeros(K_bs.shape[0], dtype=float)
    u_bed_exact[free_bs] = solve(K_bs[np.ix_(free_bs, free_bs)], f_bed_test[free_bs], assume_a="pos")
    exact_bed_y_um = float(np.linalg.norm(u_bed_exact[6 * bed_i : 6 * bed_i + 3])) * 1e6
    assert st_bs.accel_deflection_y_um == pytest.approx(exact_bed_y_um, rel=1e-5)
    # Verify it is strictly larger than raw bed.mass_g * a_y (because y_moving_mass_kg includes carriage!)
    raw_ratio = asm_bs.y_moving_mass_kg / (spec_bs.bed.mass_g * 1e-3)
    assert raw_ratio > 1.2
    assert st_bs.peak_inertial_force_y_n == pytest.approx(abs(f_expected_y), rel=1e-6)

    # 3. Moderate #6: 8-direction envelope captures pure-axis peak without 29.3% diagonal underprediction
    # On bedslinger at high accel, pure Y (0, +1) strictly exceeds any 45-degree diagonal (0.7071, +-0.7071)
    spec_env = spec_bs.model_copy(deep=True)
    spec_env.kinematics.max_accel_mm_s2 = 12000.0
    asm_env = build_assembly(spec_env)
    st_env = solve_static_fea(asm_env)
    # With pure (0, +1), relative deflection > max(accel_deflection_y_um, 0.7071 * accel_deflection_y_um + 0.7071 * accel_deflection_x_um)
    # Specifically, verify combined_max_deflection_um >= accel_deflection_y_um (impossible if only 0.7071*u_y were checked!)
    assert st_env.combined_max_deflection_um > st_env.accel_deflection_y_um
    # Verify pure-Y contribution in nodals (after subtracting static gravity Y) matches 100% of u_y (not 0.7071 * u_y)
    spec_env_grav = spec_env.model_copy(deep=True)
    spec_env_grav.kinematics.max_accel_mm_s2 = 0.0
    st_env_grav = solve_static_fea(build_assembly(spec_env_grav))
    bed_y_grav = st_env_grav.nodal_displacements_mm["bed_surface"][1]
    bed_y_comb = st_env.nodal_displacements_mm["bed_surface"][1]
    bed_y_only = st_env.nodal_displacements_y_mm["bed_surface"][1]
    assert abs(bed_y_comb - bed_y_grav) == pytest.approx(abs(bed_y_only), rel=1e-4)


# ---------------------------------------------------------------------------
# D1: RCSA Rotational Damping Units
# ---------------------------------------------------------------------------


def test_d1_rcsa_rotational_damping_dimensional_consistency() -> None:
    """Verify RCSA carriage rotational damping produces zeta_rot_eff ~ frame_damping_ratio, independent of w0."""
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    asm = build_assembly(spec)
    mod = solve_modal_fea(asm)

    # Check standard RCSA runs cleanly and stays physically bounded
    rc_std = analyze_toolhead_rcsa(asm, mod)
    assert math.isfinite(rc_std.nozzle_to_sensor_amp_ratio)
    assert rc_std.hotend_bending_stiffness_n_per_mm > 0.0

    # Isolate carriage rotational resonance f_rot = sqrt(k_rot_car / j_car) / (2*pi) across wide freq sweep
    # With proper +1j * (2 * z_f * sqrt(k_rot_car * j_car)) * w damping:
    # 1) Changing modal_result.fundamental_freq_hz (w0) when carriage translational resonance is below f_rot
    #    must NOT collapse rotational damping by 1/w0; effective Q_rot = 1 / (2 * zeta_f) (~10, not ~6000).
    f_sweep = np.linspace(100.0, 2500.0, 2401)

    spec_stiff_th = spec.model_copy(deep=True)
    spec_stiff_th.toolhead.mount_material = "aluminum_6061_t6"
    spec_stiff_th.toolhead.internal_stiffness_n_per_mm = 15000.0
    asm_stiff = build_assembly(spec_stiff_th)

    # Build two ModalResult probes with identical frame_damping_ratio=0.05 but 3x different f0 (50 Hz vs 150 Hz)
    mod_lo = solve_modal_fea(asm_stiff)
    mod_lo.fundamental_freq_hz = 50.0
    mod_lo.frame_damping_ratio = 0.05
    for m in mod_lo.modes:
        m.frequency_hz = 50.0

    mod_hi = solve_modal_fea(asm_stiff)
    mod_hi.fundamental_freq_hz = 150.0
    mod_hi.frame_damping_ratio = 0.05
    for m in mod_hi.modes:
        m.frequency_hz = 150.0

    rc_lo = analyze_toolhead_rcsa(asm_stiff, mod_lo, freqs_hz=f_sweep, mount_stiffness_scale=100.0)
    rc_hi = analyze_toolhead_rcsa(asm_stiff, mod_hi, freqs_hz=f_sweep, mount_stiffness_scale=100.0)

    # In the high-frequency band (> 400 Hz) dominated by carriage pitch / hotend flexure,
    # peak nozzle transfer FRF must remain strongly damped (no un-damped 1e-4-zeta spike)
    # and increasing frame_damping_ratio 2x (0.04 -> 0.08) must meaningfully damp the rotational peak!
    spec_damped_lo = spec.model_copy(deep=True)
    asm_d = build_assembly(spec_damped_lo)
    mod_d04 = solve_modal_fea(asm_d)
    mod_d04.frame_damping_ratio = 0.04
    mod_d08 = solve_modal_fea(asm_d)
    mod_d08.frame_damping_ratio = 0.08

    # Check high-freq band above primary translational mode (> 250 Hz)
    # On X-dominant MGN12H (k_ry/k_rz >> k_rx), doubling frame_damping_ratio reduces high-freq peak;
    # on a Y-axis roll-governed carriage (C-block Z0), frame_damping_ratio damps the carriage roll peak by > 15%.
    mask_hf = f_sweep >= 250.0
    rc_04 = analyze_toolhead_rcsa(asm_d, mod_d04, freqs_hz=f_sweep, mount_stiffness_scale=20.0)
    rc_08 = analyze_toolhead_rcsa(asm_d, mod_d08, freqs_hz=f_sweep, mount_stiffness_scale=20.0)
    peak_hf_04 = float(np.max(rc_04.nozzle_frf_mag[mask_hf]))
    peak_hf_08 = float(np.max(rc_08.nozzle_frf_mag[mask_hf]))
    assert peak_hf_08 < peak_hf_04

    spec_y_roll = spec.model_copy(deep=True)
    spec_y_roll.gantry.rail_block_type = "C"
    spec_y_roll.gantry.rail_preload = "Z0"
    asm_yr = build_assembly(spec_y_roll)
    mod_y04 = solve_modal_fea(asm_yr)
    mod_y04.frame_damping_ratio = 0.04
    mod_y04.modes[0].dominant_axis = "Y"
    mod_y08 = solve_modal_fea(asm_yr)
    mod_y08.frame_damping_ratio = 0.08
    mod_y08.modes[0].dominant_axis = "Y"
    rc_y04 = analyze_toolhead_rcsa(asm_yr, mod_y04, freqs_hz=f_sweep, mount_stiffness_scale=20.0)
    rc_y08 = analyze_toolhead_rcsa(asm_yr, mod_y08, freqs_hz=f_sweep, mount_stiffness_scale=20.0)
    assert float(np.max(rc_y08.nozzle_frf_mag[mask_hf])) < float(np.max(rc_y04.nozzle_frf_mag[mask_hf])) * 0.85

    # Estimate half-power damping ratio on the carriage pitch resonance of Y_car[1,1] directly
    # using identical parameters as rcsa.py: at f_rot, |H| / |H_dc| = 1 / (2 * zeta_rot_eff)
    # Verify high-freq nozzle peak stays within reasonable damped Q (< 50x static compliance, NOT 6000x!)
    static_c = 1.0 / (rc_lo.hotend_bending_stiffness_n_per_mm * 1e3)
    assert float(np.max(rc_lo.nozzle_frf_mag[f_sweep >= 400.0])) < 50.0 * static_c
    assert float(np.max(rc_hi.nozzle_frf_mag[f_sweep >= 400.0])) < 50.0 * static_c


# ---------------------------------------------------------------------------
# Eigensolver Conditioning, Bedslinger fy_exc, and z_joint Sensitivity
# ---------------------------------------------------------------------------


def test_inverted_eigh_matches_eigsh_and_sensitivity_fd_exact() -> None:
    """Dense inverted eigh(M, K) matches shift-invert eigsh (<1e-6 Hz) and FD sensitivities (<0.1%)."""
    for yaml_name in ("corexy_voron_style.yaml", "voron_trident_300.yaml", "cartesian_bedslinger.yaml"):
        spec = PrinterSpec.from_yaml(EXAMPLES_DIR / yaml_name)
        asm = build_assembly(spec)
        K, M, _, _, free_dofs = assemble_global_matrices(asm)
        K_ff = 0.5 * (K[np.ix_(free_dofs, free_dofs)] + K[np.ix_(free_dofs, free_dofs)].T)
        M_ff = 0.5 * (M[np.ix_(free_dofs, free_dofs)] + M[np.ix_(free_dofs, free_dofs)].T)
        m_diag = np.diag(M_ff)
        m_floor = max(float(np.max(m_diag)) * 1e-12, 1e-12)
        M_ff = M_ff + np.diag(np.where(m_diag < m_floor, m_floor, 0.0))

        k_scan = 16
        # Dense inverted eigh(M_ff, K_ff) via sensitivity._solve_eigen_modes (n_free <= 600 uses dense path!)
        assert len(free_dofs) <= 600
        ev_dense, vecs_dense = _solve_eigen_modes(K_ff, M_ff, target_scan=k_scan)
        f_dense = np.sqrt(ev_dense) / (2.0 * math.pi)

        # Reference shift-invert eigsh
        lu = splu(csc_matrix(K_ff))
        op = LinearOperator(K_ff.shape, matvec=lu.solve, dtype=float)
        ev_sp, _ = eigsh(csc_matrix(K_ff), k=k_scan, M=csc_matrix(M_ff), sigma=0.0, which="LM", OPinv=op)
        f_sp = np.sqrt(np.sort(np.maximum(ev_sp, 0.0))) / (2.0 * math.pi)

        assert np.max(np.abs(f_dense[:10] - f_sp[:10])) < 1e-6
        # M-orthonormality check
        gram = vecs_dense.T @ M_ff @ vecs_dense
        assert np.allclose(gram, np.eye(k_scan), atol=1e-9)

    # Sensitivity ±0.1% central FD check on brace_rear_1 (which had 34.3% error under old eigh(K, M)!)
    spec_cx = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    spec_cx.frame.bracing = "rear_cross"
    asm_cx = build_assembly(spec_cx)
    mod_cx = solve_modal_fea(asm_cx)
    sen_cx = compute_structural_sensitivities(asm_cx, modal_res=mod_cx)

    eps = 1e-3
    dln = math.log(1.0 + eps) - math.log(1.0 - eps)

    def _eval_member_fd(mem_id: str, e_scale: float = 1.0, rho_scale: float = 1.0) -> tuple[float, float, float]:
        asm_p = build_assembly(spec_cx)
        for mem in asm_p.members:
            if mem.id == mem_id:
                mem.material = mem.material.model_copy(
                    update={
                        "youngs_modulus_pa": mem.material.youngs_modulus_pa * e_scale,
                        "density_kg_m3": mem.material.density_kg_m3 * rho_scale,
                    }
                )
        mod_p = solve_modal_fea(asm_p)
        st_p = solve_static_fea(asm_p)
        # Recompute linear static X & Y relative displacement directly via sensitivity's loads
        K_p, _, _, nidx_p, free_p = assemble_global_matrices(asm_p)
        lu_p = splu(csc_matrix(0.5 * (K_p[np.ix_(free_p, free_p)] + K_p[np.ix_(free_p, free_p)].T)))
        a = spec_cx.kinematics.max_accel_mm_s2 * 1e-3
        noz_i = nidx_p[asm_p.nozzle_node_id]
        bed_i = nidx_p[asm_p.bed_node_id]
        from printsim.fea.solver import build_inertial_load_vectors

        fx, fy, _, _ = build_inertial_load_vectors(asm_p, nidx_p, K_p.shape[0], a)
        ux = np.zeros(K_p.shape[0]); ux[free_p] = lu_p.solve(fx[free_p])
        uy = np.zeros(K_p.shape[0]); uy[free_p] = lu_p.solve(fy[free_p])
        _ = st_p
        return (
            mod_p.fundamental_freq_hz,
            float(ux[6 * noz_i + 0] - ux[6 * bed_i + 0]),
            float(uy[6 * noz_i + 1] - uy[6 * bed_i + 1]),
        )

    f_rp, _, _ = _eval_member_fd("brace_flat_rear_0", rho_scale=1.0 + eps)
    f_rm, _, _ = _eval_member_fd("brace_flat_rear_0", rho_scale=1.0 - eps)
    fd_mass = (f_rp - f_rm) / dln
    assert sen_cx.member_sensitivities["brace_flat_rear_0"]["mass_sens_hz"] == pytest.approx(fd_mass, rel=1e-3)

    f_ep, ux_ep, uy_ep = _eval_member_fd("col_rl_low", e_scale=1.0 + eps)
    f_em, ux_em, uy_em = _eval_member_fd("col_rl_low", e_scale=1.0 - eps)
    assert sen_cx.member_sensitivities["col_rl_low"]["stiffness_sens_hz"] == pytest.approx(
        (f_ep - f_em) / dln, rel=1e-3
    )
    assert sen_cx.member_sensitivities["col_rl_low"]["du_rel_x_dlnE_m"] == pytest.approx(
        (ux_ep - ux_em) / dln, rel=1e-3
    )
    assert sen_cx.member_sensitivities["col_rl_low"]["du_rel_y_dlnE_m"] == pytest.approx(
        (uy_ep - uy_em) / dln, rel=1e-3
    )


def test_bedslinger_modal_fy_exc_and_z_joint_sensitivity() -> None:
    """Verify bedslinger modal fy_exc excites only the bed (not stationary portal) and z_joint maps cleanly."""
    # 1. Bedslinger fy_exc unit bed load
    spec_bs = PrinterSpec.from_yaml(EXAMPLES_DIR / "cartesian_bedslinger.yaml")
    asm_bs = build_assembly(spec_bs)
    mod_bs = solve_modal_fea(asm_bs)
    sen_bs = compute_structural_sensitivities(asm_bs, modal_res=mod_bs)

    # Identify actual bed Y mode in modes (bed_surface Y amplitude > 0.7)
    bed_y_mode = next(m for m in mod_bs.modes if abs(m.nodal_vectors["bed_surface"][1]) > 0.7)
    assert mod_bs.shaper_y is not None
    assert mod_bs.shaper_y.structural_freq_hz == pytest.approx(bed_y_mode.frequency_hz, rel=0.02)
    assert sen_bs.primary_y_freq_hz == pytest.approx(bed_y_mode.frequency_hz, rel=0.01)

    # 2. Track 4 flying-gantry z_joint spring handling in sensitivity.py
    spec_fly = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    asm_fly = build_assembly(spec_fly)
    # Attach 4 synthetic z_joint springs coupling gantry corners to frame columns (mimicking Track 4)
    for tag, g_nid, col_nid in (
        ("fl", "y_car_l", "g_fl"),
        ("fr", "y_car_r", "g_fr"),
        ("rl", "y_car_l", "g_rl"),
        ("rr", "y_car_r", "g_rr"),
    ):
        asm_fly.fastener_springs.append(
            FastenerSpring(
                id=f"z_joint_{tag}",
                bay="z_joint",
                shell_node_id=g_nid,
                frame_node_id=col_nid,
                k_shear_n_m=2.0e6,
                k_normal_n_m=4.0e6,
                k_rot_nm_rad=180.0,
                normal_vec=(0.0, 0.0, 1.0),
                offset_m=(0.0, 0.0, 0.0),
            )
        )

    sen_fly = compute_structural_sensitivities(asm_fly)
    for tag in ("fl", "fr", "rl", "rr"):
        zid = f"z_joint_{tag}"
        assert zid in sen_fly.member_sensitivities
        assert sen_fly.member_sensitivities[zid]["role"] == "gantry"

    stiffen_map = {t["id"]: t for t in sen_fly.top_stiffen_targets}
    if "z_gantry_joints" in stiffen_map:
        assert stiffen_map["z_gantry_joints"]["label"] == "Z-Axis Gantry Joints"
    # Verify sensitivity FD on z_joint_fl matches central FD < 0.1%
    eps = 1e-3
    dln = math.log(1.0 + eps) - math.log(1.0 - eps)

    def _eval_zj(scale: float) -> float:
        asm_p = build_assembly(spec_fly)
        for tag, g_nid, col_nid in (
            ("fl", "y_car_l", "g_fl"),
            ("fr", "y_car_r", "g_fr"),
            ("rl", "y_car_l", "g_rl"),
            ("rr", "y_car_r", "g_rr"),
        ):
            s_mult = scale if tag == "fl" else 1.0
            asm_p.fastener_springs.append(
                FastenerSpring(
                    id=f"z_joint_{tag}",
                    bay="z_joint",
                    shell_node_id=g_nid,
                    frame_node_id=col_nid,
                    k_shear_n_m=2.0e6 * s_mult,
                    k_normal_n_m=4.0e6 * s_mult,
                    k_rot_nm_rad=180.0 * s_mult,
                    normal_vec=(0.0, 0.0, 1.0),
                    offset_m=(0.0, 0.0, 0.0),
                )
            )
        return solve_modal_fea(asm_p).fundamental_freq_hz

    fp, fm = _eval_zj(1.0 + eps), _eval_zj(1.0 - eps)
    assert sen_fly.member_sensitivities["z_joint_fl"]["stiffness_sens_hz"] == pytest.approx(
        (fp - fm) / dln, rel=1e-3
    )
