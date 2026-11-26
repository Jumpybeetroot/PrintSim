"""Regression and analytical verification tests for Group A FEA fixes (A-01..A-25)."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import solve
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

from printsim.fea import (
    assemble_global_matrices,
    compute_structural_sensitivities,
    solve_bed_plate_fem,
    solve_modal_fea,
    solve_static_fea,
)
from printsim.fea.beam3d import beam_rotation_matrix
from printsim.fea.bed_plate_fem import _apply_mount_springs, _resolve_mount_layout
from printsim.fea.shell_mitc4 import mitc4_shell_matrices
from printsim.fea.solver import _is_pseudo_profile_member
from printsim.geometry import AssemblyModel, StructuralMember, StructuralNode, build_assembly
from printsim.schema import BED_MOUNT_CATALOG, PrinterSpec, get_material, get_profile

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


# ---------------------------------------------------------------------------
# A-01: Y inertial load share includes hotend mass
# ---------------------------------------------------------------------------


def test_a01_y_inertial_load_includes_hotend_mass() -> None:
    """Verify solver and sensitivity Y inertial load shares use toolhead_moving_mass_kg (with hotend)."""
    for yaml_name, expected_approx_um in (
        ("corexy_voron_style.yaml", 25.92),  # D-04 Misumi HFS5-2020 + physical X-beam/rail cut-length + cog/nozzle split
        ("vzbot_330_awd.yaml", 9.25),  # D-02 0/90 CF tube + physical X-beam/rail cut-length + cog/nozzle split
    ):
        spec = PrinterSpec.from_yaml(EXAMPLES_DIR / yaml_name)
        asm = build_assembly(spec)
        assert asm.toolhead_moving_mass_kg > spec.toolhead.total_mass_g * 1e-3

        st = solve_static_fea(asm)
        sen = compute_structural_sensitivities(asm, static_res=st)

        # Exact manual static solve using toolhead_moving_mass_kg (split across cog/nozzle) vs buggy total_mass_g
        K, _, _, nidx, free = assemble_global_matrices(asm)
        lu = splu(csc_matrix(0.5 * (K[np.ix_(free, free)] + K[np.ix_(free, free)].T)))
        a = spec.kinematics.max_accel_mm_s2 * 1e-3
        f_y_tot = asm.y_moving_mass_kg * a
        cog_i = nidx[asm.toolhead_cog_node_id]
        noz_i = nidx[asm.nozzle_node_id]
        bed_i = nidx[asm.bed_node_id]
        m_cog = max(float(asm.nodes[cog_i].lumped_mass_kg), 0.0)
        m_noz = max(float(asm.nodes[noz_i].lumped_mass_kg), 0.0) if noz_i != cog_i else 0.0
        noz_frac = m_noz / (m_cog + m_noz) if (noz_i != cog_i and (m_cog + m_noz) > 1e-9) else 0.0
        cog_frac = 1.0 - noz_frac
        others = [g for g in asm.gantry_node_ids if g != asm.toolhead_node_id and g in nidx]

        def _solve_with_share(th_mass_kg: float) -> tuple[float, float]:
            fy = np.zeros(K.shape[0], dtype=float)
            sh = th_mass_kg / asm.y_moving_mass_kg
            fy[6 * cog_i + 1] = f_y_tot * sh * cog_frac
            if noz_frac > 0.0:
                fy[6 * noz_i + 1] += f_y_tot * sh * noz_frac
            for gid in others:
                fy[6 * nidx[gid] + 1] += f_y_tot * (1.0 - sh) / len(others)
            uy = np.zeros(K.shape[0], dtype=float)
            uy[free] = lu.solve(fy[free])
            rel_norm_um = float(np.linalg.norm(uy[6 * noz_i : 6 * noz_i + 3] - uy[6 * bed_i : 6 * bed_i + 3])) * 1e6
            rel_y_um = float(uy[6 * noz_i + 1] - uy[6 * bed_i + 1]) * 1e6
            return rel_norm_um, rel_y_um

        norm_exact_um, y_exact_um = _solve_with_share(asm.toolhead_moving_mass_kg)
        norm_buggy_um, _ = _solve_with_share(spec.toolhead.total_mass_g * 1e-3)

        assert st.accel_deflection_y_um == pytest.approx(norm_exact_um, rel=1e-7)
        assert st.accel_deflection_y_um == pytest.approx(expected_approx_um, abs=0.15)
        assert st.accel_deflection_y_um > norm_buggy_um * 1.05
        assert sen.static_rel_disp_y_um == pytest.approx(y_exact_um, rel=1e-7)

    # Central FD of X-gantry beam du_rel_y_dlnE_m matches sensitivity with toolhead_moving_mass_kg
    spec_vz = PrinterSpec.from_yaml(EXAMPLES_DIR / "vzbot_330_awd.yaml")
    asm_vz = build_assembly(spec_vz)
    sen_vz = compute_structural_sensitivities(asm_vz)
    eps = 1e-3
    dln = math.log(1.0 + eps) - math.log(1.0 - eps)
    uy_vals = []
    for sc in (1.0 + eps, 1.0 - eps):
        asm_p = build_assembly(spec_vz)
        for mem in asm_p.members:
            if mem.id == "x_gantry_l":
                upd: dict[str, float] = {"youngs_modulus_pa": mem.material.youngs_modulus_pa * sc}
                if mem.material.shear_modulus_pa is not None:
                    upd["shear_modulus_pa"] = mem.material.shear_modulus_pa * sc
                mem.material = mem.material.model_copy(update=upd)
        uy_vals.append(compute_structural_sensitivities(asm_p).static_rel_disp_y_um * 1e-6)
    fd_duy = (uy_vals[0] - uy_vals[1]) / dln
    assert sen_vz.member_sensitivities["x_gantry_l"]["du_rel_y_dlnE_m"] == pytest.approx(fd_duy, rel=1e-3)

    # Clamping guard when synthetic assembly sets toolhead_moving_mass_kg > y_moving_mass_kg
    asm_clamp = build_assembly(spec_vz)
    asm_clamp.toolhead_moving_mass_kg = asm_clamp.y_moving_mass_kg * 1.5
    st_clamp = solve_static_fea(asm_clamp)
    mod_clamp = solve_modal_fea(asm_clamp)
    sen_clamp = compute_structural_sensitivities(asm_clamp, static_res=st_clamp, modal_res=mod_clamp)
    assert math.isfinite(st_clamp.accel_deflection_y_um)
    assert math.isfinite(sen_clamp.static_rel_disp_y_um)


# ---------------------------------------------------------------------------
# A-02: Deltas apply 100% Y load and excitation at CoG (no tower carriage split)
# ---------------------------------------------------------------------------


def test_a02_delta_xy_isotropy_static_modal_and_sensitivity() -> None:
    """Symmetric 3-tower delta must give identical X and Y accel deflection, modal freq, and sensitivity."""
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "delta_racer.yaml")
    asm = build_assembly(spec)

    st = solve_static_fea(asm)
    mod = solve_modal_fea(asm)
    sen = compute_structural_sensitivities(asm, static_res=st, modal_res=mod)

    assert st.accel_deflection_x_um == pytest.approx(st.accel_deflection_y_um, rel=1e-6)
    assert mod.shaper_x is not None and mod.shaper_y is not None
    assert mod.shaper_x.structural_freq_hz == pytest.approx(mod.shaper_y.structural_freq_hz, rel=1e-5)
    assert mod.shaper_x.effective_freq_hz == pytest.approx(mod.shaper_y.effective_freq_hz, rel=1e-5)
    assert mod.shaper_x.max_accel_mm_s2 == pytest.approx(mod.shaper_y.max_accel_mm_s2, rel=1e-4)

    assert sen.static_rel_disp_x_um == pytest.approx(sen.static_rel_disp_y_um, rel=1e-6)
    assert sen.primary_x_freq_hz == pytest.approx(sen.primary_y_freq_hz, rel=1e-4)

    # Also test a delta without explicit CoG node (has_explicit_cog == False) to verify CoG moment sign
    spec_nc = spec.model_copy(deep=True)
    spec_nc.toolhead.cog_offset_mm = (0.0, 0.0, -20.0)
    spec_nc.toolhead.nozzle_offset_mm = (0.0, 0.0, -45.0)
    asm_nc = build_assembly(spec_nc)
    # Remove explicit toolhead_cog node & toolhead_mount to force has_explicit_cog = False path
    asm_nc.nodes = [n for n in asm_nc.nodes if n.id != "toolhead_cog"]
    asm_nc.members = [m for m in asm_nc.members if m.id not in ("toolhead_mount", "nozzle_stem")]
    asm_nc.toolhead_cog_node_id = asm_nc.toolhead_node_id
    asm_nc.nozzle_node_id = asm_nc.toolhead_node_id
    st_nc = solve_static_fea(asm_nc)
    assert st_nc.accel_deflection_x_um == pytest.approx(st_nc.accel_deflection_y_um, rel=1e-6)


# ---------------------------------------------------------------------------
# A-07: Deterministic eigensolve across sparse & dense paths
# ---------------------------------------------------------------------------


def test_a07_deterministic_eigensolve_bit_identical() -> None:
    """Consecutive modal & sensitivity runs must produce bit-identical dicts, freqs, and mode signs."""
    for yaml_name in ("corexy_voron_style.yaml", "cartesian_bedslinger.yaml"):
        spec = PrinterSpec.from_yaml(EXAMPLES_DIR / yaml_name)
        asm = build_assembly(spec)

        modal_hashes = []
        sens_hashes = []
        for _ in range(3):
            mod = solve_modal_fea(asm)
            sen = compute_structural_sensitivities(asm, modal_res=mod)
            modal_hashes.append(
                hashlib.md5(json.dumps(mod.to_dict(), sort_keys=True).encode()).hexdigest()
            )
            sens_hashes.append(
                hashlib.md5(json.dumps(sen.to_dict(), sort_keys=True).encode()).hexdigest()
            )

        assert len(set(modal_hashes)) == 1, f"Non-deterministic modal hashes on {yaml_name}: {modal_hashes}"
        assert len(set(sens_hashes)) == 1, f"Non-deterministic sens hashes on {yaml_name}: {sens_hashes}"


# ---------------------------------------------------------------------------
# A-15: Nozzle gyration radius 7 mm in sensitivity.py + Fox-Kapoor sum rule
# ---------------------------------------------------------------------------


def test_a15_nozzle_gyration_radius_and_fox_kapoor_sum_rule() -> None:
    """Verify Fox-Kapoor mass sensitivities sum to exact -f/2 (matching 7 mm nozzle r_gyr)."""
    for yaml_name in ("cartesian_bedslinger.yaml", "corexy_voron_style.yaml", "delta_racer.yaml"):
        asm = build_assembly(PrinterSpec.from_yaml(EXAMPLES_DIR / yaml_name))
        sen = compute_structural_sensitivities(asm)
        total_mass_sens = (
            sum(v["mass_sens_hz"] for v in sen.member_sensitivities.values())
            + sum(v["mass_sens_hz"] for v in sen.panel_bay_sensitivities.values())
            + sum(v["mass_sens_hz"] for v in sen.lumped_mass_sensitivities.values())
        )
        expected = -0.5 * sen.fundamental_freq_hz
        assert total_mass_sens == pytest.approx(expected, rel=1e-7)

    # Direct FD check on a cantilever with heavy rotational nozzle tip node (r_gyr = 0.007 vs 0.025)
    mat = get_material("aluminum_6063_t5")
    prof = get_profile("extrusion_2020")
    nodes = [
        StructuralNode("n0", np.array([0.0, 0.0, 0.0]), fixed=True, role="frame"),
        StructuralNode("n1", np.array([0.30, 0.0, 0.0]), role="toolhead"),
        StructuralNode("noz", np.array([0.30, 0.0, -0.05]), lumped_mass_kg=0.25, role="nozzle"),
    ]
    members = [
        StructuralMember("beam", "n0", "n1", prof, mat, role="gantry"),
        StructuralMember("stem", "n1", "noz", prof, mat, role="toolhead"),
    ]
    asm_noz = AssemblyModel(
        spec=PrinterSpec(),
        nodes=nodes,
        members=members,
        toolhead_node_id="n1",
        nozzle_node_id="noz",
    )
    sen_noz = compute_structural_sensitivities(asm_noz)
    eps = 1e-3
    dln = math.log(1.0 + eps) - math.log(1.0 - eps)
    f_vals = []
    for sc in (1.0 + eps, 1.0 - eps):
        n_p = [
            StructuralNode("n0", np.array([0.0, 0.0, 0.0]), fixed=True, role="frame"),
            StructuralNode("n1", np.array([0.30, 0.0, 0.0]), role="toolhead"),
            StructuralNode("noz", np.array([0.30, 0.0, -0.05]), lumped_mass_kg=0.25 * sc, role="nozzle"),
        ]
        a_p = AssemblyModel(
            spec=PrinterSpec(),
            nodes=n_p,
            members=members,
            toolhead_node_id="n1",
            nozzle_node_id="noz",
        )
        f_vals.append(compute_structural_sensitivities(a_p).fundamental_freq_hz)
    fd_noz_mass = (f_vals[0] - f_vals[1]) / dln
    assert sen_noz.lumped_mass_sensitivities["noz"]["mass_sens_hz"] == pytest.approx(fd_noz_mass, rel=1e-4)


# ---------------------------------------------------------------------------
# A-19: Signed u_rel compliance share (no zero-out when u_rel < 0)
# ---------------------------------------------------------------------------


def test_a19_signed_compliance_share_for_negative_u_rel() -> None:
    """Negative static u_rel_y must still yield positive compliance shares summing to 100%."""
    mat = get_material("aluminum_6063_t5")
    prof = get_profile("extrusion_2020")
    spec = PrinterSpec(architecture="corexy")
    spec.kinematics.max_accel_mm_s2 = 10000.0
    # cog_z = -0.25 m below toolhead with has_explicit_cog=False creates M_x = +F_y * 0.25,
    # yielding positive theta_x that tilts the nozzle at z = +0.80 m into -Y (u_rel_y < 0).
    spec.toolhead.cog_offset_mm = (0.0, 0.0, -250.0)
    nodes = [
        StructuralNode("base", np.array([0.0, 0.0, 0.0]), fixed=True, role="frame"),
        StructuralNode("mid", np.array([0.0, 0.0, 0.10]), role="frame"),
        StructuralNode("toolhead", np.array([0.0, 0.0, 0.20]), role="toolhead"),
        StructuralNode("nozzle_tip", np.array([0.0, 0.0, 0.80]), role="nozzle"),
    ]
    members = [
        StructuralMember("col_low", "base", "mid", prof, mat, role="frame"),
        StructuralMember("col_high", "mid", "toolhead", prof, mat, role="frame"),
        StructuralMember("stem", "toolhead", "nozzle_tip", prof, mat, role="toolhead"),
    ]
    asm = AssemblyModel(
        spec=spec,
        nodes=nodes,
        members=members,
        toolhead_node_id="toolhead",
        toolhead_cog_node_id="toolhead",
        nozzle_node_id="nozzle_tip",
        x_moving_mass_kg=0.0,  # isolates Y axis and exercises abs(u_rel_x) < 1e-12 guard too
        y_moving_mass_kg=0.5,
        toolhead_moving_mass_kg=0.5,
    )
    sen = compute_structural_sensitivities(asm)
    assert sen.static_rel_disp_x_um == pytest.approx(0.0, abs=1e-9)
    assert sen.static_rel_disp_y_um < -1.0

    comp_sum = sum(m["compliance_share_pct"] for m in sen.member_sensitivities.values())
    assert comp_sum == pytest.approx(100.0, abs=0.2)
    assert sen.member_sensitivities["col_low"]["compliance_share_pct"] > 0.0
    assert sen.member_sensitivities["col_high"]["compliance_share_pct"] > 0.0
    u_rel_y_m = sen.static_rel_disp_y_um * 1e-6
    signed_sum = sum(-m["du_rel_y_dlnE_m"] / u_rel_y_m for m in sen.member_sensitivities.values())
    assert signed_sum == pytest.approx(1.0, rel=1e-6)


# ---------------------------------------------------------------------------
# A-21: Member self-weight consistent Euler-Bernoulli fixed-end moments
# ---------------------------------------------------------------------------


def test_a21_beam3d_self_weight_consistent_fixed_end_moments() -> None:
    """1-element cantilever & 2-element fixed-fixed beams match exact Euler-Bernoulli self-weight sag."""
    mat = get_material("aluminum_6063_t5")
    prof = get_profile("extrusion_2040")  # Iy != Iz
    L = 0.8
    g_acc = 9.80665
    w_per_m = prof.area_m2 * mat.density_kg_m3 * g_acc
    E = mat.youngs_modulus_pa

    # 1. Single-element cantilevers in horizontal X, horizontal Y, and tilted 3D orientations
    directions = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([2.0, 1.0, 1.5]) / math.sqrt(2.0**2 + 1.0**2 + 1.5**2),
    ]
    for ex in directions:
        p0 = np.zeros(3, dtype=float)
        p1 = L * ex
        nodes = [
            StructuralNode("n0", p0, fixed=True, role="frame"),
            StructuralNode("n1", p1, fixed=False, role="frame"),
        ]
        members = [StructuralMember("m0", "n0", "n1", prof, mat, role="frame")]
        asm = AssemblyModel(spec=PrinterSpec(), nodes=nodes, members=members, toolhead_node_id="n1")
        K, _, F_grav, nidx, free = assemble_global_matrices(asm)

        # Total -Z force preserved identically
        assert float(np.sum(F_grav[2::6])) == pytest.approx(-w_per_m * L, rel=1e-12)

        u = np.zeros(K.shape[0], dtype=float)
        u[free] = solve(K[np.ix_(free, free)], F_grav[free])

        r_mat, _ = beam_rotation_matrix(p0, p1)
        u_tip_loc = r_mat @ u[6 * nidx["n1"] : 6 * nidx["n1"] + 3]
        th_tip_loc = r_mat @ u[6 * nidx["n1"] + 3 : 6 * nidx["n1"] + 6]
        q_loc = r_mat @ np.array([0.0, 0.0, -w_per_m])

        exact_u_x = (q_loc[0] * (L**2)) / (2.0 * E * prof.area_m2)
        exact_u_y = (q_loc[1] * (L**4)) / (8.0 * E * prof.iz_m4)
        exact_u_z = (q_loc[2] * (L**4)) / (8.0 * E * prof.iy_m4)
        exact_th_y = -(q_loc[2] * (L**3)) / (6.0 * E * prof.iy_m4)  # w'(x) = -th_y
        exact_th_z = +(q_loc[1] * (L**3)) / (6.0 * E * prof.iz_m4)  # v'(x) = +th_z

        assert u_tip_loc[0] == pytest.approx(exact_u_x, rel=1e-12, abs=1e-15)
        assert u_tip_loc[1] == pytest.approx(exact_u_y, rel=1e-12, abs=1e-15)
        assert u_tip_loc[2] == pytest.approx(exact_u_z, rel=1e-12, abs=1e-15)
        assert th_tip_loc[1] == pytest.approx(exact_th_y, rel=1e-12, abs=1e-15)
        assert th_tip_loc[2] == pytest.approx(exact_th_z, rel=1e-12, abs=1e-15)

    # 2. 1-element propped cantilever (fixed at n0, pinned at n1): exact tip rotation -w L^3 / (48 E Iy)
    # Using w'(L) = +w L^3 / (48 E Iy) for downward load => th_y(L) = -w'(L) = -w L^3 / (48 E Iy)
    nodes_prop = [
        StructuralNode("n0", np.array([0.0, 0.0, 0.0]), fixed=True, role="frame"),
        StructuralNode("n1", np.array([L, 0.0, 0.0]), pinned=True, role="frame"),
    ]
    asm_prop = AssemblyModel(
        spec=PrinterSpec(),
        nodes=nodes_prop,
        members=[StructuralMember("m0", "n0", "n1", prof, mat, role="frame")],
        toolhead_node_id="n1",
    )
    K_p, _, F_p, nidx_p, free_p = assemble_global_matrices(asm_prop)
    u_p = np.zeros(K_p.shape[0], dtype=float)
    u_p[free_p] = solve(K_p[np.ix_(free_p, free_p)], F_p[free_p])
    exact_prop_th_y = -(w_per_m * (L**3)) / (48.0 * E * prof.iy_m4)
    assert u_p[6 * nidx_p["n1"] + 4] == pytest.approx(exact_prop_th_y, rel=1e-12)

    # 3. 2-element fixed-fixed beam: exact midpoint deflection -w L^4 / (384 E Iy)
    nodes_ff = [
        StructuralNode("n0", np.array([0.0, 0.0, 0.0]), fixed=True, role="frame"),
        StructuralNode("n1", np.array([0.5 * L, 0.0, 0.0]), role="frame"),
        StructuralNode("n2", np.array([L, 0.0, 0.0]), fixed=True, role="frame"),
    ]
    members_ff = [
        StructuralMember("m0", "n0", "n1", prof, mat, role="frame"),
        StructuralMember("m1", "n1", "n2", prof, mat, role="frame"),
    ]
    asm_ff = AssemblyModel(spec=PrinterSpec(), nodes=nodes_ff, members=members_ff, toolhead_node_id="n1")
    K_ff, _, F_ff, nidx_ff, free_ff = assemble_global_matrices(asm_ff)
    u_ff = np.zeros(K_ff.shape[0], dtype=float)
    u_ff[free_ff] = solve(K_ff[np.ix_(free_ff, free_ff)], F_ff[free_ff])
    exact_mid_z = -(w_per_m * (L**4)) / (384.0 * E * prof.iy_m4)
    assert u_ff[6 * nidx_ff["n1"] + 2] == pytest.approx(exact_mid_z, rel=1e-12)


# ---------------------------------------------------------------------------
# A-22: Stress envelope excludes non-structural pseudo-profile members
# ---------------------------------------------------------------------------


def test_a22_stress_envelope_excludes_pseudo_profile_members() -> None:
    """max_von_mises_mpa, min_safety_factor, and peak_stress_part_label ignore virtual links."""
    for yaml_name in ("cartesian_bedslinger.yaml", "delta_racer.yaml", "corexy_voron_style.yaml"):
        asm = build_assembly(PrinterSpec.from_yaml(EXAMPLES_DIR / yaml_name))
        st = solve_static_fea(asm)

        assert st.peak_stress_part_label != "Toolhead & Nozzle Mount"
        mem_by_id = {m.id: m for m in asm.members}
        real_vm = [
            ms.von_mises_mpa
            for ms in st.member_stresses
            if not _is_pseudo_profile_member(mem_by_id[ms.member_id])
        ]
        shell_vm = [sh.von_mises_mpa for sh in st.shell_panel_stresses]
        expected_max_vm = max(real_vm + shell_vm)
        assert st.max_von_mises_mpa == pytest.approx(expected_max_vm, rel=1e-9)

        # Virtual members remain in member_stresses for viewport coloring
        if any(m.id == "toolhead_mount" for m in asm.members):
            assert any(ms.member_id == "toolhead_mount" for ms in st.member_stresses)

    # Synthetic fallback: only a virtual toolhead_mount member present
    mat_v = get_material("abs_printed").model_copy(update={"density_kg_m3": 1e-3})
    prof_v = get_profile("extrusion_2020").model_copy(update={"id": "toolhead_mount_profile"})
    asm_virt = AssemblyModel(
        spec=PrinterSpec(),
        nodes=[
            StructuralNode("n0", np.array([0.0, 0.0, 0.0]), fixed=True, role="frame"),
            StructuralNode("n1", np.array([0.0, 0.0, 0.05]), lumped_mass_kg=0.3, role="cog"),
        ],
        members=[StructuralMember("toolhead_mount", "n0", "n1", prof_v, mat_v, role="toolhead")],
        toolhead_node_id="n1",
        toolhead_cog_node_id="n1",
        nozzle_node_id="n1",
        x_moving_mass_kg=0.3,
        y_moving_mass_kg=0.3,
        toolhead_moving_mass_kg=0.3,
    )
    st_virt = solve_static_fea(asm_virt)
    assert st_virt.max_von_mises_mpa > 0.0
    assert st_virt.min_safety_factor < 999.0


# ---------------------------------------------------------------------------
# A-24: AWD sync damping PSD branch preserves monotonic shaper accel
# ---------------------------------------------------------------------------


def test_a24_awd_sync_damping_psd_monotonic_accel() -> None:
    """AWD sync modes raise PSD damping ratio without triggering spurious ZV->MZV accel drop."""
    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_monolith_awd.yaml")
    for dt in ("closed_loop_stepper", "foc_servo"):
        results = {}
        for mode in ("none", "torque_mirror", "position_equalizer"):
            s = base.model_copy(deep=True)
            s.kinematics.drive_type = dt  # type: ignore[assignment]
            s.kinematics.awd_sync_mode = mode  # type: ignore[assignment]
            results[mode] = solve_modal_fea(build_assembly(s))

        m_none = results["none"]
        m_tm = results["torque_mirror"]
        m_eq = results["position_equalizer"]
        assert m_eq.shaper_graph_x is not None and m_tm.shaper_graph_x is not None and m_none.shaper_graph_x is not None
        assert m_eq.shaper_graph_x.damping_ratio > m_tm.shaper_graph_x.damping_ratio > m_none.shaper_graph_x.damping_ratio
        assert m_eq.recommended_max_accel_mm_s2 >= m_tm.recommended_max_accel_mm_s2 >= m_none.recommended_max_accel_mm_s2


# ---------------------------------------------------------------------------
# A-25: Bed plate FEM odd mesh_n rear-center anchor detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mount_type", ["cnc_kelvin_magnetic", "trident_3pt_standoffs"])
def test_a25_bed_plate_odd_mesh_n_rear_center_anchor_pinned(mount_type: str) -> None:
    """Odd mesh_n (nx=8, no node at x=0) must still pin the rear-center 3pt anchor in X/Y."""
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    spec.bed.mount_type = mount_type  # type: ignore[assignment]

    for odd_n in (5, 7):
        res_odd = solve_bed_plate_fem(spec, mesh_n=odd_n)
        res_even = solve_bed_plate_fem(spec, mesh_n=odd_n + 1)
        assert res_odd.plate_first_mode_type == "elastic_drum"
        assert abs(res_odd.plate_first_mode_hz - res_even.plate_first_mode_hz) / res_even.plate_first_mode_hz < 0.08

        # Inspect assembled K matrix directly to verify rear-center anchor has full k_xy_each (not 500 N/m)
        nx = ny = odd_n + 1
        gx = np.linspace(-0.5, 0.5, nx, dtype=float)
        gy = np.linspace(-0.5, 0.5, ny, dtype=float)
        anc_ids, mount_pts, kz_c = _resolve_mount_layout(spec, gx, gy)
        assert len(anc_ids) == 3

        lx = spec.bed.size_mm[0] * 1e-3
        ly = spec.bed.size_mm[1] * 1e-3
        t = spec.bed.plate_thickness_mm * 1e-3
        mat = get_material(spec.bed.plate_material)
        areal_mass = (spec.bed.mass_g * 1e-3) / (lx * ly)
        coords = np.array([[x * lx, y * ly, 0.0] for y in gy for x in gx], dtype=float)
        n_dof = 6 * nx * ny
        k_plate = np.zeros((n_dof, n_dof), dtype=float)
        for ey in range(odd_n):
            for ex in range(odd_n):
                n0 = ey * nx + ex
                q = [n0, n0 + 1, (ey + 1) * nx + ex + 1, (ey + 1) * nx + ex]
                ke, _, _ = mitc4_shell_matrices(coords[q], mat.youngs_modulus_pa, mat.poisson_ratio, t, areal_mass)
                dofs = [6 * nid + d for nid in q for d in range(6)]
                k_plate[np.ix_(dofs, dofs)] += ke

        k_with_mounts = k_plate.copy()
        _apply_mount_springs(
            k_with_mounts,
            spec,
            anc_ids,
            kz_c,
            t,
            mount_points_rel=mount_pts,
            lx_m=lx,
            ly_m=ly,
        )
        cfg = BED_MOUNT_CATALOG[mount_type]
        k_xy_each = float(cfg["k_lateral_n_per_mm"]) * 1e3  # 3 * k_lat / 3 = k_lat
        rear_idx = max(range(len(mount_pts)), key=lambda i: mount_pts[i][1])
        rear_base = 6 * anc_ids[rear_idx]
        k_rear_xy = k_with_mounts[rear_base : rear_base + 2, rear_base : rear_base + 2] - k_plate[
            rear_base : rear_base + 2, rear_base : rear_base + 2
        ]
        eig_xy = np.linalg.eigvalsh(k_rear_xy)
        assert float(eig_xy[0]) == pytest.approx(k_xy_each, rel=1e-6)
        assert float(eig_xy[1]) == pytest.approx(k_xy_each, rel=1e-6)
