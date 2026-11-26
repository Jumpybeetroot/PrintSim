"""Comprehensive physics, numerical integration, FEA, and thermal-structural tests for CF X-gantry beams."""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import solve

from printsim.engine import evaluate_printer, optimize_design
from printsim.engine import optimizer as optimizer_mod
from printsim.engine import pareto as pareto_mod
from printsim.fea import (
    assemble_global_matrices,
    beam_shear_coefficients,
    compute_thermal_structural_report,
    element_internal_forces,
    newmark_rail_composite_gamma,
    solve_modal_fea,
    solve_static_fea,
)
from printsim.geometry import AssemblyModel, StructuralMember, StructuralNode, build_assembly, generate_bom
from printsim.geometry.builder import compute_toolhead_link_props
from printsim.schema import (
    MATERIAL_CATALOG,
    PROFILE_CATALOG,
    PrinterSpec,
    build_model_confidence_audit,
    get_material,
    get_profile,
)
from printsim.schema.profiles import _hollow_rect_tube

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def _corexy_cf_base(
    *,
    x_beam_profile: str = "cf_tube_20x20x2",
    x_beam_material: str = "carbon_fiber_tube",
    x_rail_mount: str = "front",
    chamber_temp_c: float = 25.0,
) -> PrinterSpec:
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    spec.frame.enclosed_panels = True
    spec.frame.panel_mounting = "bolted_structural"
    spec.frame.panel_material = "aluminum_5052_sheet"
    spec.frame.panel_thickness_mm = 4.0
    spec.frame.chamber_temp_c = chamber_temp_c
    spec.gantry.bracket_material = "aluminum_6061_t6"
    spec.toolhead.mount_material = "aluminum_6061_t6"
    spec.toolhead.internal_stiffness_n_per_mm = 1850.0
    spec.gantry.x_beam_profile = x_beam_profile
    spec.gantry.x_beam_material = x_beam_material
    spec.gantry.x_rail_mount = x_rail_mount  # type: ignore[assignment]
    return spec


# ---------------------------------------------------------------------------
# 1. Rounded-Corner Hollow Tube Section Geometry & Carbonara Scale-Weight
# ---------------------------------------------------------------------------


def test_hollow_rect_tube_rounded_corners_numerical_integration_and_sharp_limit() -> None:
    """Verify _hollow_rect_tube area, Iy, Iz against 2D grid integration (<0.2%) and exact r=0 limit."""
    # 1. Exact sharp-corner limit (r = 0)
    w_mm, h_mm, t_mm = 30.0, 20.0, 2.0
    sharp = _hollow_rect_tube("test_sharp", "Sharp 30x20x2", w_mm, h_mm, t_mm, corner_radius_mm=0.0)
    w, h, t = w_mm * 1e-3, h_mm * 1e-3, t_mm * 1e-3
    wi, hi = w - 2.0 * t, h - 2.0 * t
    assert sharp.area_m2 == pytest.approx(w * h - wi * hi, rel=1e-12)
    assert sharp.iy_m4 == pytest.approx((w * h**3 - wi * hi**3) / 12.0, rel=1e-12)
    assert sharp.iz_m4 == pytest.approx((h * w**3 - hi * wi**3) / 12.0, rel=1e-12)

    # 2. High-resolution 2D quadrant grid numerical integration for rounded-corner CF tubes
    for prof_id in ("cf_tube_20x20_carbonara", "cf_tube_30x20x2", "cf_tube_20x20x1"):
        prof = get_profile(prof_id)
        assert prof.wall_thickness_m is not None
        assert prof.corner_radius_m > 0.0

        hw = 0.5 * prof.width_m
        hh = 0.5 * prof.height_m
        t_w = prof.wall_thickness_m
        ro = prof.corner_radius_m
        ri = max(ro - t_w, 0.0)
        hwi = hw - t_w
        hhi = hh - t_w

        n_grid = 900
        dy = hw / n_grid
        dz = hh / n_grid
        ys = (np.arange(n_grid, dtype=float) + 0.5) * dy
        zs = (np.arange(n_grid, dtype=float) + 0.5) * dz
        Y, Z = np.meshgrid(ys, zs, indexing="ij")

        # Quadrant mask inside rounded rectangle: exterior corner circle check
        in_outer_corner = (Y <= hw - ro) | (Z <= hh - ro) | (
            (Y - (hw - ro)) ** 2 + (Z - (hh - ro)) ** 2 <= ro**2
        )
        in_outer = (Y <= hw) & (Z <= hh) & in_outer_corner

        in_inner_corner = (Y <= hwi - ri) | (Z <= hhi - ri) | (
            (Y - (hwi - ri)) ** 2 + (Z - (hhi - ri)) ** 2 <= ri**2
        )
        in_inner = (Y <= hwi) & (Z <= hhi) & in_inner_corner

        mask = in_outer & (~in_inner)
        da = dy * dz
        area_num = 4.0 * float(np.sum(mask)) * da
        iy_num = 4.0 * float(np.sum((Z**2) * mask)) * da
        iz_num = 4.0 * float(np.sum((Y**2) * mask)) * da

        assert prof.area_m2 == pytest.approx(area_num, rel=2e-3), prof_id
        assert prof.iy_m4 == pytest.approx(iy_num, rel=2e-3), prof_id
        assert prof.iz_m4 == pytest.approx(iz_num, rel=2e-3), prof_id


def test_carbonara_scale_weight_calibration() -> None:
    """Carbonara 20x20 CF X-beam matches 64.7 / 74.5 / 84.3 g at 330 / 380 / 430 mm within 1%."""
    prof = get_profile("cf_tube_20x20_carbonara")
    mat = get_material("carbon_fiber_torsional_45")
    assert mat.density_kg_m3 == pytest.approx(1560.0)

    for length_mm, expected_g in ((330.0, 64.7), (380.0, 74.5), (430.0, 84.3)):
        mass_g = prof.area_m2 * (length_mm * 1e-3) * mat.density_kg_m3 * 1000.0
        assert mass_g == pytest.approx(expected_g, rel=0.01)


# ---------------------------------------------------------------------------
# 2. CLT Material Catalog Hierarchy & M40J Negative CTE
# ---------------------------------------------------------------------------


def test_clt_material_catalog_hierarchy_and_m40j_negative_cte() -> None:
    """Verify CLT E, G, Tg, and negative axial CTE relationships across CF tube grades."""
    twill = MATERIAL_CATALOG["carbon_fiber_3k_twill"]
    ud_std = MATERIAL_CATALOG["carbon_fiber_tube"]
    ud_hi = MATERIAL_CATALOG["carbon_fiber_high_tg"]
    tor45 = MATERIAL_CATALOG["carbon_fiber_torsional_45"]
    m40j = MATERIAL_CATALOG["carbon_fiber_m40j_hm"]

    assert (
        twill.youngs_modulus_pa
        < tor45.youngs_modulus_pa
        < ud_std.youngs_modulus_pa
        < m40j.youngs_modulus_pa
    )
    assert (
        twill.effective_shear_modulus_pa
        < ud_std.effective_shear_modulus_pa
        < tor45.effective_shear_modulus_pa
        < m40j.effective_shear_modulus_pa
    )
    assert tor45.effective_shear_modulus_pa > 3.0 * ud_std.effective_shear_modulus_pa
    assert ud_hi.youngs_modulus_pa == pytest.approx(ud_std.youngs_modulus_pa)
    assert ud_hi.tg_c == pytest.approx(125.0)
    assert tor45.tg_c == pytest.approx(125.0)
    assert m40j.tg_c == pytest.approx(140.0)
    assert m40j.thermal_expansion_1_k < 0.0
    assert m40j.thermal_expansion_1_k == pytest.approx(-0.5e-6)


# ---------------------------------------------------------------------------
# 3. Exact Timoshenko Cantilever Benchmark & Bredt-Batho Torsional Stress
# ---------------------------------------------------------------------------


def _make_cantilever(profile_id: str, material_id: str, length_m: float = 0.35, n_elems: int = 4) -> AssemblyModel:
    prof = get_profile(profile_id)
    mat = get_material(material_id)
    nodes = [
        StructuralNode(
            id=f"n{i}",
            position_m=np.array([length_m * i / n_elems, 0.0, 0.0]),
            fixed=(i == 0),
            role="toolhead" if i == n_elems else "frame",
        )
        for i in range(n_elems + 1)
    ]
    members = [
        StructuralMember(
            id=f"m{i}",
            start_id=f"n{i}",
            end_id=f"n{i + 1}",
            profile=prof,
            material=mat,
        )
        for i in range(n_elems)
    ]
    return AssemblyModel(spec=PrinterSpec(), nodes=nodes, members=members, toolhead_node_id=f"n{n_elems}")


def test_timoshenko_cantilever_benchmark_and_bredt_batho_torsion_stress() -> None:
    """CF tube cantilever matches exact Timoshenko deflection <0.1% while Al stays Euler-Bernoulli."""
    L = 0.35
    P = 20.0

    # A. High-E/G composite tubes (both square and asymmetric 30x20 rectangular)
    for prof_id, mat_id in (
        ("cf_tube_20x20x2", "carbon_fiber_tube"),
        ("cf_tube_30x20x2", "carbon_fiber_tube"),
        ("cf_tube_20x20_carbonara", "carbon_fiber_torsional_45"),
    ):
        asm = _make_cantilever(prof_id, mat_id, length_m=L, n_elems=4)
        mem = asm.members[0]
        prof = mem.profile
        mat = mem.material
        ky, kz = beam_shear_coefficients(mem)
        assert 0.0 < ky <= 0.85 and 0.0 < kz <= 0.85

        K, M, _, nidx, free_dofs = assemble_global_matrices(asm)
        K_ff = K[np.ix_(free_dofs, free_dofs)]
        M_ff = M[np.ix_(free_dofs, free_dofs)]
        assert np.allclose(K_ff, K_ff.T, rtol=1e-12, atol=1e-8)
        assert np.allclose(M_ff, M_ff.T, rtol=1e-12, atol=1e-12)
        assert float(np.linalg.eigvalsh(0.5 * (K_ff + K_ff.T))[0]) > 0.0
        tip = nidx["n4"]

        # Local Y bending (uses Iz and kappa_y)
        Fy = np.zeros(K.shape[0])
        Fy[6 * tip + 1] = P
        uy = np.zeros(K.shape[0])
        uy[free_dofs] = solve(K[np.ix_(free_dofs, free_dofs)], Fy[free_dofs])
        eb_y = P * (L**3) / (3.0 * mat.youngs_modulus_pa * prof.iz_m4)
        timo_y = eb_y + P * L / (ky * mat.effective_shear_modulus_pa * prof.area_m2)
        assert abs(uy[6 * tip + 1]) == pytest.approx(timo_y, rel=1e-3)
        assert abs(uy[6 * tip + 1]) > eb_y * (1.05 if mat_id == "carbon_fiber_tube" else 1.01)

        # Local Z bending (uses Iy and kappa_z)
        Fz = np.zeros(K.shape[0])
        Fz[6 * tip + 2] = -P
        uz = np.zeros(K.shape[0])
        uz[free_dofs] = solve(K[np.ix_(free_dofs, free_dofs)], Fz[free_dofs])
        eb_z = P * (L**3) / (3.0 * mat.youngs_modulus_pa * prof.iy_m4)
        timo_z = eb_z + P * L / (kz * mat.effective_shear_modulus_pa * prof.area_m2)
        assert abs(uz[6 * tip + 2]) == pytest.approx(timo_z, rel=1e-3)
        assert abs(uz[6 * tip + 2]) > eb_z * (1.05 if mat_id == "carbon_fiber_tube" else 1.01)

    # B. Isotropic aluminum tube stays exact Euler-Bernoulli
    asm_al = _make_cantilever("al_tube_20x20x2", "aluminum_6061_t6", length_m=L, n_elems=4)
    assert beam_shear_coefficients(asm_al.members[0]) == (0.0, 0.0)
    K_al, _, _, nidx_al, free_al = assemble_global_matrices(asm_al)
    tip_al = nidx_al["n4"]
    Fy_al = np.zeros(K_al.shape[0])
    Fy_al[6 * tip_al + 1] = P
    uy_al = np.zeros(K_al.shape[0])
    uy_al[free_al] = solve(K_al[np.ix_(free_al, free_al)], Fy_al[free_al])
    eb_al = P * (L**3) / (3.0 * asm_al.members[0].material.youngs_modulus_pa * asm_al.members[0].profile.iz_m4)
    assert abs(uy_al[6 * tip_al + 1]) == pytest.approx(eb_al, rel=1e-4)

    # C. Bredt-Batho torsional shear stress on closed hollow CF tube under pure torque
    asm_tor = _make_cantilever("cf_tube_20x20x2", "carbon_fiber_torsional_45", length_m=L, n_elems=1)
    K_t, _, _, nidx_t, free_t = assemble_global_matrices(asm_tor)
    T_nm = 5.0
    Ft = np.zeros(K_t.shape[0])
    Ft[6 * nidx_t["n1"] + 3] = T_nm
    ut = np.zeros(K_t.shape[0])
    ut[free_t] = solve(K_t[np.ix_(free_t, free_t)], Ft[free_t])
    m0 = asm_tor.members[0]
    forces = element_internal_forces(
        m0,
        asm_tor.node_map[m0.start_id].position_m,
        asm_tor.node_map[m0.end_id].position_m,
        ut,
    )
    p = m0.profile
    assert p.wall_thickness_m is not None
    r_m = max(p.corner_radius_m - 0.5 * p.wall_thickness_m, 0.0)
    a_m = (p.width_m - p.wall_thickness_m) * (p.height_m - p.wall_thickness_m) - (4.0 - math.pi) * (r_m**2)
    tau_bb = T_nm / (2.0 * a_m * p.wall_thickness_m)
    expected_vm_mpa = math.sqrt(3.0) * tau_bb * 1e-6
    assert forces["torque_nm"] == pytest.approx(T_nm, rel=1e-9)
    assert forces["von_mises_mpa"] == pytest.approx(expected_vm_mpa, rel=1e-6)


# ---------------------------------------------------------------------------
# 4. Full-Printer CoreXY FEA: Torsional 45 vs 0/90 & M40J Combined Stiffness
# ---------------------------------------------------------------------------


def test_corexy_torsional_45_and_m40j_stiffness_hierarchy() -> None:
    """±45/0 torsional_45 beats 0/90 tube in cantilevered Y-stiffness; M40J leads 20x20 combined X/Y."""
    st_090 = solve_static_fea(
        build_assembly(_corexy_cf_base(x_beam_profile="cf_tube_20x20x2", x_beam_material="carbon_fiber_tube"))
    )
    st_tor = solve_static_fea(
        build_assembly(
            _corexy_cf_base(x_beam_profile="cf_tube_20x20x2", x_beam_material="carbon_fiber_torsional_45")
        )
    )
    st_m40j = solve_static_fea(
        build_assembly(_corexy_cf_base(x_beam_profile="cf_tube_20x20x2", x_beam_material="carbon_fiber_m40j_hm"))
    )

    # 3.1x higher GJ + shear stiffness beats 0/90 tube on cantilevered toolhead Y-stiffness & twist
    assert st_tor.toolhead_stiffness_y_n_per_mm > st_090.toolhead_stiffness_y_n_per_mm
    assert st_tor.nozzle_twist_angle_mrad < st_090.nozzle_twist_angle_mrad

    # M40J high-modulus ±45/0 achieves highest combined X/Y stiffness among 20x20 CF tubes
    assert st_m40j.toolhead_stiffness_x_n_per_mm >= st_tor.toolhead_stiffness_x_n_per_mm
    assert st_m40j.toolhead_stiffness_y_n_per_mm > st_tor.toolhead_stiffness_y_n_per_mm
    assert (
        st_m40j.toolhead_stiffness_x_n_per_mm + st_m40j.toolhead_stiffness_y_n_per_mm
        > st_090.toolhead_stiffness_x_n_per_mm + st_090.toolhead_stiffness_y_n_per_mm
    )


# ---------------------------------------------------------------------------
# 5. Heated Chamber (70 C) Matrix Softening & High-Tg Retention
# ---------------------------------------------------------------------------


def test_heated_chamber_70c_softening_tg_warning_and_high_tg_retention() -> None:
    """At 70 C, carbon_fiber_tube loses Y/torsional stiffness & warns; high_tg & torsional_45 retain >95%."""
    st_std_25 = solve_static_fea(
        build_assembly(_corexy_cf_base(x_beam_material="carbon_fiber_tube", chamber_temp_c=25.0))
    )
    rep_std_70 = evaluate_printer(
        _corexy_cf_base(x_beam_material="carbon_fiber_tube", chamber_temp_c=70.0)
    )
    assert rep_std_70.static_fea.toolhead_stiffness_y_n_per_mm < st_std_25.toolhead_stiffness_y_n_per_mm * 0.94
    assert rep_std_70.static_fea.beam_center_deflection_um > st_std_25.beam_center_deflection_um * 1.08
    assert rep_std_70.thermal_structural.x_beam_has_tg is True
    assert rep_std_70.thermal_structural.x_beam_tg_warning is True
    assert rep_std_70.thermal_structural.x_beam_shear_retention_pct < 75.0
    assert any("soften" in d.lower() and "carbon_fiber_torsional_45" in d for d in rep_std_70.diagnostics)

    for mat_id in ("carbon_fiber_high_tg", "carbon_fiber_torsional_45"):
        st_25 = solve_static_fea(
            build_assembly(_corexy_cf_base(x_beam_material=mat_id, chamber_temp_c=25.0))
        )
        rep_70 = evaluate_printer(
            _corexy_cf_base(x_beam_material=mat_id, chamber_temp_c=70.0)
        )
        assert rep_70.static_fea.toolhead_stiffness_y_n_per_mm > st_25.toolhead_stiffness_y_n_per_mm * 0.95
        assert rep_70.thermal_structural.x_beam_modulus_retention_pct > 95.0
        assert rep_70.thermal_structural.x_beam_shear_retention_pct > 95.0
        assert rep_70.thermal_structural.x_beam_tg_warning is False

    # Bedslinger CF X-gantry also applies chamber-temp laminate softening consistently
    bs_25 = PrinterSpec.from_yaml(EXAMPLES_DIR / "cartesian_bedslinger.yaml")
    bs_25.gantry.x_beam_profile = "cf_tube_20x20x2"
    bs_25.gantry.x_beam_material = "carbon_fiber_tube"
    bs_25.gantry.bracket_material = "aluminum_6061_t6"
    bs_25.toolhead.mount_material = "aluminum_6061_t6"
    bs_25.toolhead.hotend.preset_id = "chube_conduction"
    bs_70 = bs_25.model_copy(deep=True)
    bs_70.frame.chamber_temp_c = 70.0
    asm_bs_25 = build_assembly(bs_25)
    asm_bs_70 = build_assembly(bs_70)
    st_bs_25 = solve_static_fea(asm_bs_25)
    st_bs_70 = solve_static_fea(asm_bs_70)
    xg_25 = next(m for m in asm_bs_25.members if m.id == "x_gantry_l")
    xg_70 = next(m for m in asm_bs_70.members if m.id == "x_gantry_l")
    assert xg_70.material.effective_shear_modulus_pa < xg_25.material.effective_shear_modulus_pa * 0.75
    assert st_bs_70.toolhead_stiffness_y_n_per_mm < st_bs_25.toolhead_stiffness_y_n_per_mm * 0.96


# ---------------------------------------------------------------------------
# 6. Newmark Wall-Thickness Hierarchy & Epoxy Thermal Softening
# ---------------------------------------------------------------------------


def test_newmark_gamma_wall_thickness_hierarchy_and_thermal_softening() -> None:
    """Newmark gamma satisfies 1.0 mm < 1.5 mm < 2.0 mm wall and drops when low-Tg epoxy heats to 70 C."""
    steel = get_material("steel_4140")
    cf = get_material("carbon_fiber_tube")
    mgn12 = get_profile("mgn12_rail")
    p10 = get_profile("cf_tube_20x20x1")
    p15 = get_profile("cf_tube_20x20x1_5")
    p20 = get_profile("cf_tube_20x20x2")

    g10 = newmark_rail_composite_gamma(
        steel.youngs_modulus_pa,
        mgn12.area_m2,
        cf.youngs_modulus_pa,
        p10.area_m2,
        0.025,
        0.460,
        "carbon_tube",
        wall_thickness_m=p10.wall_thickness_m,
    )
    g15 = newmark_rail_composite_gamma(
        steel.youngs_modulus_pa,
        mgn12.area_m2,
        cf.youngs_modulus_pa,
        p15.area_m2,
        0.025,
        0.460,
        "carbon_tube",
        wall_thickness_m=p15.wall_thickness_m,
    )
    g20 = newmark_rail_composite_gamma(
        steel.youngs_modulus_pa,
        mgn12.area_m2,
        cf.youngs_modulus_pa,
        p20.area_m2,
        0.025,
        0.460,
        "carbon_tube",
        wall_thickness_m=p20.wall_thickness_m,
    )
    assert g10 < g15 < g20

    # Full assembly thermal_structural telemetry matches hierarchy & 70 C drop
    ts_10 = compute_thermal_structural_report(
        build_assembly(_corexy_cf_base(x_beam_profile="cf_tube_20x20x1", x_beam_material="carbon_fiber_tube"))
    )
    ts_15 = compute_thermal_structural_report(
        build_assembly(_corexy_cf_base(x_beam_profile="cf_tube_20x20x1_5", x_beam_material="carbon_fiber_tube"))
    )
    ts_20 = compute_thermal_structural_report(
        build_assembly(_corexy_cf_base(x_beam_profile="cf_tube_20x20x2", x_beam_material="carbon_fiber_tube"))
    )
    ts_20_hot = compute_thermal_structural_report(
        build_assembly(
            _corexy_cf_base(
                x_beam_profile="cf_tube_20x20x2",
                x_beam_material="carbon_fiber_tube",
                chamber_temp_c=70.0,
            )
        )
    )
    assert ts_10.rail_newmark_gamma < ts_15.rail_newmark_gamma < ts_20.rail_newmark_gamma
    assert ts_20_hot.rail_newmark_gamma < ts_20.rail_newmark_gamma


# ---------------------------------------------------------------------------
# 7. Opposite Bimetallic Bow Direction & Steel Backer Cancellation
# ---------------------------------------------------------------------------


def test_opposite_bimetallic_bow_direction_and_steel_backer_cancellation() -> None:
    """Al beam bows away_from_rail; CF beam bows toward_rail; steel backer cancels CF bow."""
    s_al = _corexy_cf_base(
        x_beam_profile="extrusion_2020",
        x_beam_material="aluminum_6063_t5",
        x_rail_mount="top",
        chamber_temp_c=70.0,
    )
    s_cf = _corexy_cf_base(
        x_beam_profile="cf_tube_20x20x2",
        x_beam_material="carbon_fiber_high_tg",
        x_rail_mount="top",
        chamber_temp_c=70.0,
    )
    s_cf_bk = s_cf.model_copy(deep=True)
    s_cf_bk.gantry.rail_backer_material = "steel_4140"
    s_cf_bk.gantry.rail_backer_thickness_mm = 3.0

    s_cf_cold = s_cf.model_copy(deep=True)
    s_cf_cold.frame.chamber_temp_c = 25.0

    ts_al = compute_thermal_structural_report(build_assembly(s_al))
    ts_cf = compute_thermal_structural_report(build_assembly(s_cf))
    ts_cf_bk = compute_thermal_structural_report(build_assembly(s_cf_bk))
    ts_cf_cold = compute_thermal_structural_report(build_assembly(s_cf_cold))

    assert ts_al.gantry_bimetallic_bow_um > 40.0
    assert ts_al.gantry_bow_direction == "away_from_rail"

    assert ts_cf.gantry_bimetallic_bow_um > 40.0
    assert ts_cf.gantry_bow_direction == "toward_rail"

    assert ts_cf_bk.gantry_bimetallic_bow_um < ts_cf.gantry_bimetallic_bow_um * 0.45
    assert ts_cf_bk.gantry_backer_reduction_pct > 55.0

    assert ts_cf_cold.gantry_bimetallic_bow_um == pytest.approx(0.0, abs=1e-9)
    assert ts_cf_cold.gantry_bow_direction == "neutral"


# ---------------------------------------------------------------------------
# 8. Diagnostics, Optimizer & Pareto CF Option Integration
# ---------------------------------------------------------------------------


def test_cf_diagnostics_and_optimizer_pareto_integration() -> None:
    """Thin-wall / low-G CF diagnostics fire appropriately; Pareto & optimizer expose new CF beams."""
    rep_1mm = evaluate_printer(_corexy_cf_base(x_beam_profile="cf_tube_20x20x1", x_beam_material="carbon_fiber_tube"))
    assert any("cf_tube_20x20x1" in d and "nut-bar" in d.lower() for d in rep_1mm.diagnostics)
    assert any("carbon_fiber_torsional_45" in d and "torsional" in d.lower() for d in rep_1mm.diagnostics)

    # Pareto X_BEAM_OPTIONS round-trip encoding & decoding within _XU bounds
    assert len(pareto_mod.X_BEAM_OPTIONS) >= 6
    assert int(pareto_mod._XU[3]) == len(pareto_mod.X_BEAM_OPTIONS) - 1
    opt_pairs = {(prof, mat) for _, _, prof, mat in pareto_mod.X_BEAM_OPTIONS}
    assert ("cf_tube_20x20_carbonara", "carbon_fiber_torsional_45") in opt_pairs
    assert ("cf_tube_20x20x2", "carbon_fiber_high_tg") in opt_pairs
    assert ("cf_tube_30x20x2", "carbon_fiber_torsional_45") in opt_pairs
    assert ("cf_tube_20x20_carbonara", "carbon_fiber_m40j_hm") in opt_pairs

    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    for idx, (_, _, xb_prof, xb_mat) in enumerate(pareto_mod.X_BEAM_OPTIONS):
        s_test = base.model_copy(deep=True)
        s_test.gantry.x_beam_profile = xb_prof
        s_test.gantry.x_beam_material = xb_mat
        vec = pareto_mod._encode_baseline_vector(s_test)
        assert vec[3] == idx
        dec, changes = pareto_mod._decode_candidate(s_test, vec)
        assert changes == ["Baseline (unchanged)"]
        assert dec.gantry.x_beam_profile == xb_prof
        assert dec.gantry.x_beam_material == xb_mat

    # Optimizer _mutate_spec reaches Carbonara / torsional / high-Tg / 30x20 / M40J options cleanly
    rng = random.Random(123)
    mutated_xbeams: set[tuple[str, str]] = set()
    for _ in range(120):
        cand, _ = optimizer_mod._mutate_spec(base, rng)
        mutated_xbeams.add((cand.gantry.x_beam_profile, cand.gantry.x_beam_material))
    assert ("cf_tube_20x20_carbonara", "carbon_fiber_torsional_45") in mutated_xbeams
    assert ("cf_tube_20x20x2", "carbon_fiber_high_tg") in mutated_xbeams
    assert ("cf_tube_30x20x2", "carbon_fiber_torsional_45") in mutated_xbeams
    assert ("cf_tube_20x20_carbonara", "carbon_fiber_m40j_hm") in mutated_xbeams

    res_opt = optimize_design(base, iterations=12, objective="rigidity", seed=42)
    assert res_opt.best_objective >= res_opt.baseline_objective


# ---------------------------------------------------------------------------
# 9. Carbonara Two-Wall Clamped Internal Inserts (cf_rail_mounting)
# ---------------------------------------------------------------------------


def test_cf_rail_mounting_auto_resolution_and_two_wall_insert_physics() -> None:
    """Carbonara auto-resolves to two_wall_clamped_inserts; generic CF tubes default to single_wall_bare."""
    # 1. Resolution rules on GantrySpec.effective_cf_rail_mounting
    s_al = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    assert s_al.gantry.cf_rail_mounting == "auto"
    assert s_al.gantry.effective_cf_rail_mounting == "none"

    s_al_forced = s_al.model_copy(deep=True)
    s_al_forced.gantry.cf_rail_mounting = "two_wall_clamped_inserts"
    assert s_al_forced.gantry.effective_cf_rail_mounting == "none"

    s_gen = _corexy_cf_base(
        x_beam_profile="cf_tube_20x20x2", x_beam_material="carbon_fiber_tube"
    )
    assert s_gen.gantry.effective_cf_rail_mounting == "single_wall_bare"

    s_carb = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_monolith_awd.yaml")
    assert s_carb.gantry.cf_rail_mounting == "auto"
    assert s_carb.gantry.effective_cf_rail_mounting == "two_wall_clamped_inserts"

    s_carb_bare = s_carb.model_copy(deep=True)
    s_carb_bare.gantry.cf_rail_mounting = "single_wall_bare"
    assert s_carb_bare.gantry.effective_cf_rail_mounting == "single_wall_bare"

    s_gen_ins = s_gen.model_copy(deep=True)
    s_gen_ins.gantry.cf_rail_mounting = "two_wall_clamped_inserts"
    assert s_gen_ins.gantry.effective_cf_rail_mounting == "two_wall_clamped_inserts"

    # 2. 25 C & 85 C comparison on Carbonara (two_wall_clamped_inserts vs single_wall_bare)
    asm_ins_25 = build_assembly(s_carb)
    asm_bare_25 = build_assembly(s_carb_bare)
    st_ins_25 = solve_static_fea(asm_ins_25)
    st_bare_25 = solve_static_fea(asm_bare_25)
    mo_ins_25 = solve_modal_fea(asm_ins_25)
    mo_bare_25 = solve_modal_fea(asm_bare_25)
    ts_ins_25 = compute_thermal_structural_report(asm_ins_25)
    ts_bare_25 = compute_thermal_structural_report(asm_bare_25)

    # Higher Newmark gamma, higher composite EI & GJ, higher carriage roll k_rx, higher modal damping
    assert asm_ins_25.x_rail_newmark_gamma > asm_bare_25.x_rail_newmark_gamma
    assert ts_ins_25.rail_newmark_gamma == pytest.approx(asm_ins_25.x_rail_newmark_gamma, rel=1e-9)
    assert ts_bare_25.rail_newmark_gamma == pytest.approx(asm_bare_25.x_rail_newmark_gamma, rel=1e-9)

    xg_ins = next(m for m in asm_ins_25.members if m.id == "x_gantry_l")
    xg_bare = next(m for m in asm_bare_25.members if m.id == "x_gantry_l")
    assert xg_ins.profile.iz_m4 > xg_bare.profile.iz_m4
    assert xg_ins.profile.j_m4 > xg_bare.profile.j_m4 * 1.11
    assert asm_ins_25.x_rail_ei_y_boost_pct > asm_bare_25.x_rail_ei_y_boost_pct
    assert asm_ins_25.x_rail_gj_boost_pct > asm_bare_25.x_rail_gj_boost_pct

    p_ins = compute_toolhead_link_props(s_carb, 0.02, 0.04)
    p_bare = compute_toolhead_link_props(s_carb_bare, 0.02, 0.04)
    assert p_ins["k_rx_car"] > p_bare["k_rx_car"]
    assert asm_ins_25.x_carriage_k_rx_nm_rad > asm_bare_25.x_carriage_k_rx_nm_rad
    assert st_ins_25.toolhead_stiffness_y_n_per_mm > st_bare_25.toolhead_stiffness_y_n_per_mm

    # Modal damping delta (+0.006) flows cleanly to ModalResult & Klipper PSD curves
    assert (mo_ins_25.frame_damping_ratio - mo_bare_25.frame_damping_ratio) == pytest.approx(
        0.006, abs=1e-9
    )
    assert mo_ins_25.shaper_graph_x is not None and mo_bare_25.shaper_graph_x is not None
    assert mo_ins_25.shaper_graph_y is not None and mo_bare_25.shaper_graph_y is not None
    assert mo_ins_25.shaper_graph_x.damping_ratio > mo_bare_25.shaper_graph_x.damping_ratio
    assert mo_ins_25.shaper_graph_y.damping_ratio > mo_bare_25.shaper_graph_y.damping_ratio

    # 3. Heated chamber 85 C retention: two_wall_clamped_inserts retains higher gamma fraction than single_wall_bare
    s_std_85 = _corexy_cf_base(
        x_beam_profile="cf_tube_20x20_carbonara",
        x_beam_material="carbon_fiber_tube",
        chamber_temp_c=85.0,
    )
    s_std_85_bare = s_std_85.model_copy(deep=True)
    s_std_85_bare.gantry.cf_rail_mounting = "single_wall_bare"
    s_std_25_ins = s_std_85.model_copy(deep=True)
    s_std_25_ins.frame.chamber_temp_c = 25.0
    s_std_25_bare = s_std_85_bare.model_copy(deep=True)
    s_std_25_bare.frame.chamber_temp_c = 25.0

    g_ins_25 = build_assembly(s_std_25_ins).x_rail_newmark_gamma
    g_ins_85 = build_assembly(s_std_85).x_rail_newmark_gamma
    g_bare_25 = build_assembly(s_std_25_bare).x_rail_newmark_gamma
    g_bare_85 = build_assembly(s_std_85_bare).x_rail_newmark_gamma
    assert (g_ins_85 / g_ins_25) > (g_bare_85 / g_bare_25)
    assert g_ins_85 > g_bare_85 * 1.4

    # 4. Exact insert mass conservation across physical x_beam_span, AssemblyModel, FEA M matrix, & zero double-counting
    from printsim.geometry.builder import corexy_x_beam_span_m

    for width_mm in (380.0, 460.0):
        s_w_ins = s_carb.model_copy(deep=True)
        s_w_ins.frame.width_mm = width_mm
        s_w_bare = s_w_ins.model_copy(deep=True)
        s_w_bare.gantry.cf_rail_mounting = "single_wall_bare"

        a_ins = build_assembly(s_w_ins)
        a_bare = build_assembly(s_w_bare)
        span_m = corexy_x_beam_span_m(s_w_ins)
        expected_ins_kg = (4.0 + 18.0 * span_m) * 1e-3

        assert a_ins.x_beam_insert_mass_kg == pytest.approx(expected_ins_kg, rel=1e-12)
        assert a_bare.x_beam_insert_mass_kg == 0.0
        # Bare x_beam_mass_kg is invariant (not polluted or double-counted by inserts)
        assert a_ins.x_beam_mass_kg == pytest.approx(a_bare.x_beam_mass_kg, rel=1e-12)
        assert (a_ins.y_moving_mass_kg - a_bare.y_moving_mass_kg) == pytest.approx(
            expected_ins_kg, rel=1e-12
        )
        assert (a_ins.total_mass_kg - a_bare.total_mass_kg) == pytest.approx(
            expected_ins_kg, rel=1e-12
        )

        # FEA consistent mass matrix diagonal & total translational X-mass reflect exact insert mass
        _, M_ins, _, _, _ = assemble_global_matrices(a_ins)
        _, M_bare, _, _, _ = assemble_global_matrices(a_bare)
        # Sum of X-DOF translational rows/cols equals total structural+lumped mass in the FE model
        assert float(np.sum(M_ins[0::6, 0::6]) - np.sum(M_bare[0::6, 0::6])) == pytest.approx(
            expected_ins_kg, rel=1e-9, abs=1e-9
        )

    # 5. Telemetry, BOM insert kit line, and confidence audit reflection
    lr_ins = asm_ins_25.to_geometry_dict()["linear_rails"]
    lr_bare = asm_bare_25.to_geometry_dict()["linear_rails"]
    assert lr_ins["cf_rail_mounting"] == "two_wall_clamped_inserts"
    assert lr_bare["cf_rail_mounting"] == "single_wall_bare"
    assert lr_ins["x_beam_insert_mass_g"] == pytest.approx(
        round(4.0 + 18.0 * corexy_x_beam_span_m(s_carb), 1)
    )
    assert lr_bare["x_beam_insert_mass_g"] == 0.0

    bom_ins = generate_bom(s_carb)
    bom_bare = generate_bom(s_carb_bare)
    ins_item = next((i for i in bom_ins.items if i.sku == "GNT-CF-INSERTS"), None)
    assert ins_item is not None
    assert ins_item.total_cost_usd == pytest.approx(18.0)
    assert ins_item.total_mass_g == pytest.approx(lr_ins["x_beam_insert_mass_g"], abs=0.1)
    assert all(i.sku != "GNT-CF-INSERTS" for i in bom_bare.items)

    audit_ins = {e["id"]: e for e in build_model_confidence_audit(s_carb)["entries"]}["x_gantry_beam"]
    audit_bare = {e["id"]: e for e in build_model_confidence_audit(s_carb_bare)["entries"]}["x_gantry_beam"]
    assert "2-Wall Clamped Inserts" in audit_ins["active_values"]
    assert "Single-Wall Bare" in audit_bare["active_values"]
    assert "125" in audit_ins["basis"] and "HVFDesigns" in audit_ins["basis"]

    # 6. Bedslinger with CF X-beam also honors cf_rail_mounting consistently
    bs_cf = PrinterSpec.from_yaml(EXAMPLES_DIR / "cartesian_bedslinger.yaml")
    bs_cf.gantry.x_beam_profile = "cf_tube_20x20_carbonara"
    bs_cf.gantry.x_beam_material = "carbon_fiber_torsional_45"
    bs_cf_bare = bs_cf.model_copy(deep=True)
    bs_cf_bare.gantry.cf_rail_mounting = "single_wall_bare"
    asm_bs_ins = build_assembly(bs_cf)
    asm_bs_bare = build_assembly(bs_cf_bare)
    exp_bs_ins_kg = (4.0 + 18.0 * (bs_cf.frame.width_mm * 1e-3)) * 1e-3
    assert asm_bs_ins.x_beam_insert_mass_kg == pytest.approx(exp_bs_ins_kg, rel=1e-12)
    assert asm_bs_ins.x_beam_mass_kg == pytest.approx(asm_bs_bare.x_beam_mass_kg, rel=1e-12)
    assert (asm_bs_ins.total_mass_kg - asm_bs_bare.total_mass_kg) == pytest.approx(
        exp_bs_ins_kg, rel=1e-12
    )


def test_corexy_physical_x_beam_and_x_rail_spans_match_voron_carbonara_bom() -> None:
    """CoreXY X-beam & X-rail moving mass and BOM match physical cut lengths (not outer frame width)."""
    from printsim.geometry.builder import corexy_x_beam_span_m, corexy_x_rail_len_m

    cases = [
        # (frame_w_mm, upright_prof, bx_mm, expected_beam_mm, expected_rail_mm, expected_carb_g)
        (410.0, "extrusion_2020", 250.0, 330.0, 300.0, 64.9),  # Voron 250 / Carbonara 330
        (460.0, "extrusion_2020", 300.0, 380.0, 350.0, 74.7),  # Voron 300 / Carbonara 380
        (510.0, "extrusion_2020", 350.0, 430.0, 400.0, 84.6),  # Voron 350 / Carbonara 430
        (490.0, "extrusion_2020", 300.0, 410.0, 350.0, 80.6),  # Wide 490mm 300-size Trident
        (500.0, "extrusion_4040", 300.0, 380.0, 350.0, 74.7),  # DoomCube 4040 300
    ]
    for fw, up, bx, exp_beam_mm, exp_rail_mm, exp_carb_g in cases:
        spec = _corexy_cf_base(
            x_beam_profile="cf_tube_20x20_carbonara",
            x_beam_material="carbon_fiber_torsional_45",
        )
        spec.frame.width_mm = fw
        spec.frame.depth_mm = fw
        spec.frame.profile = up
        spec.kinematics.build_volume_mm = (bx, bx, 250.0)

        assert corexy_x_beam_span_m(spec) * 1000.0 == pytest.approx(exp_beam_mm, abs=1e-6)
        assert corexy_x_rail_len_m(spec) * 1000.0 == pytest.approx(exp_rail_mm, abs=1e-6)

        asm = build_assembly(spec)
        assert asm.x_beam_mass_kg * 1000.0 == pytest.approx(exp_carb_g, abs=0.4)
        # MGN12R = 0.65 g/mm -> 195.0 / 227.5 / 260.0 g
        assert asm.x_rail_mass_kg * 1000.0 == pytest.approx(0.65 * exp_rail_mm, abs=0.2)

        # FEA consistent mass matrix & AssemblyModel member masses match physical X-beam + X-rail + inserts
        gantry_members = [m for m in asm.members if m.role == "gantry"]
        nmap = asm.node_map
        assert sum(m.rail_mass_kg for m in gantry_members) == pytest.approx(asm.x_rail_mass_kg, rel=1e-9)
        assert sum(m.mass_kg(nmap) for m in gantry_members) == pytest.approx(
            asm.x_beam_mass_kg + asm.x_rail_mass_kg + asm.x_backer_mass_kg + asm.x_beam_insert_mass_kg,
            rel=1e-9,
        )

        # BOM X-rail & cut-list net beam length agree with physical span
        bom = generate_bom(spec)
        x_rail_bom = next(i for i in bom.items if i.sku == "MOT-X-RAIL")
        assert f"L={exp_rail_mm:.0f} mm" in x_rail_bom.specification
        assert x_rail_bom.total_mass_g == pytest.approx(asm.x_rail_mass_kg * 1000.0, abs=0.2)
        gantry_cut = next(c for c in bom.cut_list if c.role == "gantry")
        assert gantry_cut.net_length_mm == pytest.approx(exp_beam_mm, abs=0.1)


def test_xbeam_3d_tet10_fea_calibrated_constants_and_confidence() -> None:
    """Lock in 3D Tet10 continuum FEA-calibrated gamma, end-clamp knockdowns, seat compliance & confidence."""
    steel = get_material("hardened_steel")
    al = get_material("aluminum_6063_t5")
    carb = get_material("carbon_fiber_torsional_45")
    mgn12 = get_profile("mgn12_rail")
    mgn9 = get_profile("mgn9_rail")
    p_carb = get_profile("cf_tube_20x20_carbonara")
    p_ext = get_profile("extrusion_2020")

    # 1. Calibrated Newmark composite gamma on L = 380 mm matches 3D Tet10 FEA ground truth
    g_carb_2w_25 = newmark_rail_composite_gamma(
        steel.youngs_modulus_pa,
        mgn12.area_m2,
        carb.youngs_modulus_pa,
        p_carb.area_m2,
        0.025,
        0.380,
        "carbon_tube",
        wall_thickness_m=p_carb.wall_thickness_m,
        cf_rail_mounting="two_wall_clamped_inserts",
    )
    g_carb_2w_50 = newmark_rail_composite_gamma(
        steel.youngs_modulus_pa,
        mgn12.area_m2,
        carb.youngs_modulus_pa,
        p_carb.area_m2,
        0.050,
        0.380,
        "carbon_tube",
        wall_thickness_m=p_carb.wall_thickness_m,
        cf_rail_mounting="two_wall_clamped_inserts",
    )
    g_carb_1w_25 = newmark_rail_composite_gamma(
        steel.youngs_modulus_pa,
        mgn12.area_m2,
        carb.youngs_modulus_pa,
        p_carb.area_m2,
        0.025,
        0.380,
        "carbon_tube",
        wall_thickness_m=p_carb.wall_thickness_m,
        cf_rail_mounting="single_wall_bare",
    )
    g_al_12_25 = newmark_rail_composite_gamma(
        steel.youngs_modulus_pa,
        mgn12.area_m2,
        al.youngs_modulus_pa,
        p_ext.area_m2,
        0.025,
        0.380,
        "extrusion",
    )
    g_al_9_25 = newmark_rail_composite_gamma(
        steel.youngs_modulus_pa,
        mgn9.area_m2,
        al.youngs_modulus_pa,
        p_ext.area_m2,
        0.025,
        0.380,
        "extrusion",
    )
    assert g_carb_2w_25 == pytest.approx(0.703, abs=0.025)
    assert g_carb_2w_50 == pytest.approx(0.542, abs=0.025)
    assert g_carb_1w_25 == pytest.approx(0.399, abs=0.025)
    assert g_al_12_25 == pytest.approx(0.675, abs=0.035)
    assert g_al_9_25 == pytest.approx(0.730, abs=0.025)
    assert g_carb_2w_25 > g_carb_2w_50 > g_carb_1w_25

    # 2. Local carriage seat compliance hierarchy: 1.0 < 1.8 < 2.0 mm single-wall & MGN9 < MGN12 < 2-wall
    s10 = _corexy_cf_base(x_beam_profile="cf_tube_20x20x1", x_beam_material="carbon_fiber_torsional_45")
    s18 = _corexy_cf_base(x_beam_profile="cf_tube_20x20_carbonara", x_beam_material="carbon_fiber_torsional_45")
    s18.gantry.cf_rail_mounting = "single_wall_bare"
    s20 = _corexy_cf_base(x_beam_profile="cf_tube_20x20x2", x_beam_material="carbon_fiber_torsional_45")
    s18_9 = s18.model_copy(deep=True)
    s18_9.gantry.x_rail_profile = "mgn9_rail"
    s_2w = _corexy_cf_base(x_beam_profile="cf_tube_20x20_carbonara", x_beam_material="carbon_fiber_torsional_45")

    p10 = compute_toolhead_link_props(s10, 0.02, 0.04)
    p18 = compute_toolhead_link_props(s18, 0.02, 0.04)
    p20 = compute_toolhead_link_props(s20, 0.02, 0.04)
    p18_9 = compute_toolhead_link_props(s18_9, 0.02, 0.04)
    p2w = compute_toolhead_link_props(s_2w, 0.02, 0.04)
    assert p18_9["k_rad_car"] < p18["k_rad_car"]
    assert p10["k_rad_car"] < p18["k_rad_car"] < p20["k_rad_car"] < p2w["k_rad_car"]
    assert p10["k_rx_car"] < p18["k_rx_car"] < p20["k_rx_car"] < p2w["k_rx_car"]
    assert p10["k_ry_car"] < p18["k_ry_car"] < p20["k_ry_car"] < p2w["k_ry_car"]

    # 3. Confidence audit reflects 3D Tet10 continuum FEA calibration (90% Carbonara / 88% CF / 96% Al; 90% rails)
    s_gen = _corexy_cf_base(x_beam_profile="cf_tube_20x20x2", x_beam_material="carbon_fiber_tube")
    s_al = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    e_carb = {e["id"]: e for e in build_model_confidence_audit(s_2w)["entries"]}
    e_gen = {e["id"]: e for e in build_model_confidence_audit(s_gen)["entries"]}
    e_al = {e["id"]: e for e in build_model_confidence_audit(s_al)["entries"]}
    assert e_carb["x_gantry_beam"]["trust_pct"] == 90
    assert e_carb["x_gantry_beam"]["tier"] == "exact"
    assert "3D orthotropic Tet10 continuum FEA" in e_carb["x_gantry_beam"]["basis"]
    assert "proprietary" in e_carb["x_gantry_beam"]["caveat"].lower()
    assert e_gen["x_gantry_beam"]["trust_pct"] == 88
    assert e_gen["x_gantry_beam"]["tier"] == "calibrated"
    assert e_al["x_gantry_beam"]["trust_pct"] == 96
    assert e_carb["rail_preload"]["trust_pct"] == 90
    assert e_al["rail_preload"]["trust_pct"] == 90
