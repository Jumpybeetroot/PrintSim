"""Physics, analytical FEA, mass/cost accounting, and BOM tests for composite bolted linear rails."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.linalg import eigh, solve

from printsim.api.server import create_app
from printsim.fea import assemble_global_matrices, beam_shear_coefficients, solve_modal_fea, solve_static_fea
from printsim.fea.components import (
    compute_thermal_structural_report,
    dual_rail_steiner_efficiency,
    hertzian_rail_carriage_stiffness,
    newmark_rail_composite_gamma,
)
from printsim.geometry import (
    AssemblyModel,
    StructuralMember,
    StructuralNode,
    build_assembly,
    generate_bom,
)
from printsim.geometry.builder import (
    _apply_bolted_linear_rail,
    compute_toolhead_link_props,
    x_carriage_block_mass_delta_g,
    y_carriage_block_mass_delta_g,
)
from printsim.schema import EXTRUSION_FILLINGS, PrinterSpec, ProfileSpec, get_material, get_profile

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def _corexy_spec(
    *,
    x_rail_mount: str = "front",
    y_rail_mount: str = "top",
    x_rail_profile: str = "mgn12_rail",
    y_rail_profile: str = "mgn12_rail",
    x_beam_profile: str = "extrusion_2020",
    x_beam_material: str = "aluminum_6063_t5",
    extrusion_filling: str = "none",
    dual_x_rails: bool = False,
) -> PrinterSpec:
    spec = PrinterSpec(architecture="corexy")
    spec.frame.bracing = "corner_gussets"
    spec.frame.enclosed_panels = False
    spec.frame.extrusion_filling = extrusion_filling  # type: ignore[assignment]
    spec.gantry.x_beam_profile = x_beam_profile
    spec.gantry.x_beam_material = x_beam_material
    spec.gantry.x_rail_profile = x_rail_profile
    spec.gantry.y_rail_profile = y_rail_profile
    spec.gantry.x_rail_mount = x_rail_mount  # type: ignore[assignment]
    spec.gantry.y_rail_mount = y_rail_mount  # type: ignore[assignment]
    spec.gantry.dual_x_rails = dual_x_rails
    return spec


# ---------------------------------------------------------------------------
# 1. Exact Analytical Composite Beam Mechanics & Cantilever Benchmark
# ---------------------------------------------------------------------------


def test_bolted_rail_analytical_transformed_section_and_cantilever_fea() -> None:
    """Verify exact transformed-section I_comp, eta_rail regimes, cantilever deflection, and modal f1."""
    steel = get_material("steel_4140")
    mgn12 = get_profile("mgn12_rail")
    al_mat = get_material("aluminum_6063_t5")
    cf_mat = get_material("carbon_fiber_tube")
    ext2020 = get_profile("extrusion_2020")
    cf2020 = get_profile("cf_tube_20x20x2")
    solid2020 = ProfileSpec(
        id="al_solid_2020",
        name="Solid Al 20x20 Bar",
        shape="solid_bar",
        width_m=0.020,
        height_m=0.020,
        area_m2=0.0004,
        iy_m4=(0.020**4) / 12.0,
        iz_m4=(0.020**4) / 12.0,
        j_m4=0.1406 * (0.020**4),
    )

    # Check Newmark (1951) partial-interaction gamma across extrusion, cf_*, solid_bar and dual symmetric flange action
    cases = [
        ("ext_front", ext2020, al_mat, "extrusion", [np.array([0.0, -1.0, 0.0])]),
        ("cf_top", cf2020, cf_mat, "carbon_tube", [np.array([0.0, 0.0, 1.0])]),
        ("solid_front", solid2020, al_mat, "solid_bar", [np.array([0.0, -1.0, 0.0])]),
        ("ext_dual", ext2020, al_mat, "extrusion", [np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])]),
    ]
    for label, beam_prof, beam_mat, sub_kind, normals in cases:
        L = 0.48
        n_elems = 8
        expected_eta = newmark_rail_composite_gamma(
            e_rail_pa=steel.youngs_modulus_pa,
            a_rail_m2=mgn12.area_m2,
            e_beam_pa=beam_mat.youngs_modulus_pa,
            a_beam_m2=beam_prof.area_m2,
            bolt_pitch_m=0.025,
            l_eff_m=L,
            substrate_kind=sub_kind,
        )
        nodes = [
            StructuralNode(
                id=f"n{i}",
                position_m=np.array([L * i / n_elems, 0.0, 0.0]),
                fixed=(i == 0),
                role="toolhead" if i == n_elems else "frame",
            )
            for i in range(n_elems + 1)
        ]
        nmap = {n.id: n for n in nodes}
        members = [
            StructuralMember(
                id=f"m{i}",
                start_id=f"n{i}",
                end_id=f"n{i + 1}",
                profile=beam_prof,
                material=beam_mat,
            )
            for i in range(n_elems)
        ]
        for m in members:
            _apply_bolted_linear_rail(
                m, nmap, mgn12, steel, normals, beam_shape=beam_prof.shape, l_eff_m=L
            )

        # Analytical transformed-section prediction along +X beam (ey=[0,1,0], ez=[0,0,1])
        n_e = steel.youngs_modulus_pa / beam_mat.youngs_modulus_pa
        d_c = 0.5 * beam_prof.width_m + 0.5 * mgn12.height_m
        if len(normals) == 2:
            i_par_each = expected_eta * (n_e * mgn12.area_m2) * (d_c**2)
            expected_diy = 2.0 * (n_e * mgn12.iy_m4 + i_par_each)
            expected_diz = 2.0 * (n_e * mgn12.iz_m4)
        elif abs(normals[0][1]) > 0.5:
            i_par = expected_eta * (beam_prof.area_m2 * n_e * mgn12.area_m2 / (beam_prof.area_m2 + n_e * mgn12.area_m2)) * (d_c**2)
            expected_diz = n_e * mgn12.iy_m4 + i_par
            expected_diy = n_e * mgn12.iz_m4
        else:
            i_par = expected_eta * (beam_prof.area_m2 * n_e * mgn12.area_m2 / (beam_prof.area_m2 + n_e * mgn12.area_m2)) * (d_c**2)
            expected_diy = n_e * mgn12.iy_m4 + i_par
            expected_diz = n_e * mgn12.iz_m4

        m0 = members[0]
        assert m0.profile.iy_m4 == pytest.approx(beam_prof.iy_m4 + expected_diy, rel=1e-9), label
        assert m0.profile.iz_m4 == pytest.approx(beam_prof.iz_m4 + expected_diz, rel=1e-9), label
        assert m0.profile.area_m2 == pytest.approx(beam_prof.area_m2, rel=1e-12), label

        asm = AssemblyModel(spec=PrinterSpec(), nodes=nodes, members=members, toolhead_node_id=f"n{n_elems}")
        K, M, _, node_idx, free_dofs = assemble_global_matrices(asm)
        tip = node_idx[f"n{n_elems}"]

        # Cantilever tip deflection PL^3 / (3 E I_comp) [+ Timoshenko PL / (kappa G A) for CF tubes]
        P = 15.0
        ky, kz = beam_shear_coefficients(m0)
        for axis, i_comp, kappa in ((1, m0.profile.iz_m4, ky), (2, m0.profile.iy_m4, kz)):
            F = np.zeros(K.shape[0])
            F[6 * tip + axis] = P
            u = np.zeros(K.shape[0])
            u[free_dofs] = solve(K[np.ix_(free_dofs, free_dofs)], F[free_dofs])
            exact_defl = P * (L**3) / (3.0 * beam_mat.youngs_modulus_pa * i_comp)
            if kappa > 0.0:
                exact_defl += P * L / (kappa * beam_mat.effective_shear_modulus_pa * beam_prof.area_m2)
            assert abs(u[6 * tip + axis]) == pytest.approx(exact_defl, rel=1e-4)

        # Cantilever consistent-mass modal fundamental frequency with distributed rail mass
        eigvals, _ = eigh(K[np.ix_(free_dofs, free_dofs)], M[np.ix_(free_dofs, free_dofs)], subset_by_index=[0, 0])
        f1_fea = math.sqrt(float(eigvals[0])) / (2.0 * math.pi)
        mu_per_m = beam_prof.area_m2 * beam_mat.density_kg_m3 + len(normals) * mgn12.area_m2 * steel.density_kg_m3
        i_min = min(m0.profile.iy_m4, m0.profile.iz_m4)
        f1_exact = (1.87510407**2 / (2.0 * math.pi * L**2)) * math.sqrt(
            (beam_mat.youngs_modulus_pa * i_min) / mu_per_m
        )
        k_min = ky if m0.profile.iz_m4 <= m0.profile.iy_m4 else kz
        if k_min > 0.0:
            eb_defl = (L**3) / (3.0 * beam_mat.youngs_modulus_pa * i_min)
            sh_defl = L / (k_min * beam_mat.effective_shear_modulus_pa * beam_prof.area_m2)
            f1_timo_est = f1_exact * math.sqrt(eb_defl / (eb_defl + sh_defl))
            assert f1_fea < f1_exact
            assert f1_fea == pytest.approx(f1_timo_est, rel=0.015)
        else:
            assert f1_fea == pytest.approx(f1_exact, rel=0.005)


def test_mgn12_substantial_boost_over_bare_beam_and_beats_mgn9() -> None:
    """MGN12H adds >150% primary-plane EI over bare 2020 and beats MGN9H in both Y and Z stiffness."""
    bare_2020 = get_profile("extrusion_2020")

    asm_mgn9 = build_assembly(_corexy_spec(x_rail_profile="mgn9_rail", y_rail_profile="mgn9_rail"))
    asm_mgn12 = build_assembly(_corexy_spec(x_rail_profile="mgn12_rail", y_rail_profile="mgn12_rail"))
    asm_mgn15 = build_assembly(_corexy_spec(x_rail_profile="mgn15_rail", y_rail_profile="mgn15_rail"))

    xg_9 = next(m for m in asm_mgn9.members if m.id == "x_gantry_l")
    xg_12 = next(m for m in asm_mgn12.members if m.id == "x_gantry_l")
    xg_15 = next(m for m in asm_mgn15.members if m.id == "x_gantry_l")

    # MGN12H adds substantial bending stiffness over hypothetical zero-rail bare beam (> +150% primary, > +45% secondary)
    assert xg_12.profile.iz_m4 > bare_2020.iz_m4 * 2.5
    assert xg_12.profile.iy_m4 > bare_2020.iy_m4 * 1.45

    st_9 = solve_static_fea(asm_mgn9)
    st_12 = solve_static_fea(asm_mgn12)
    st_15 = solve_static_fea(asm_mgn15)

    assert xg_15.profile.iz_m4 > xg_12.profile.iz_m4 > xg_9.profile.iz_m4
    assert xg_15.profile.iy_m4 > xg_12.profile.iy_m4 > xg_9.profile.iy_m4
    assert st_15.toolhead_stiffness_y_n_per_mm > st_12.toolhead_stiffness_y_n_per_mm > st_9.toolhead_stiffness_y_n_per_mm
    assert st_15.toolhead_stiffness_z_n_per_mm > st_12.toolhead_stiffness_z_n_per_mm > st_9.toolhead_stiffness_z_n_per_mm


# ---------------------------------------------------------------------------
# 2. X-Rail Mounting Orientations (CoreXY & Cartesian)
# ---------------------------------------------------------------------------


def test_x_rail_mount_orientations_stiffness_sag_twist_and_lever_arm() -> None:
    """Verify front vs top vs bottom vs dual_top_bottom stiffness, sag, nozzle drop, and twist."""
    asm_front = build_assembly(_corexy_spec(x_rail_mount="front"))
    asm_top = build_assembly(_corexy_spec(x_rail_mount="top"))
    asm_bottom = build_assembly(_corexy_spec(x_rail_mount="bottom"))
    asm_dual = build_assembly(_corexy_spec(x_rail_mount="dual_top_bottom"))

    st_front = solve_static_fea(asm_front)
    st_top = solve_static_fea(asm_top)
    st_bottom = solve_static_fea(asm_bottom)
    st_dual = solve_static_fea(asm_dual)

    xg_front = next(m for m in asm_front.members if m.id == "x_gantry_l")
    xg_top = next(m for m in asm_top.members if m.id == "x_gantry_l")
    xg_bottom = next(m for m in asm_bottom.members if m.id == "x_gantry_l")
    xg_dual = next(m for m in asm_dual.members if m.id == "x_gantry_l")

    # front has higher X-beam Y-bending stiffness (iz) & toolhead Y stiffness than top
    assert xg_front.profile.iz_m4 > xg_top.profile.iz_m4
    assert st_front.toolhead_stiffness_y_n_per_mm > st_top.toolhead_stiffness_y_n_per_mm

    # top & bottom have higher Z-bending stiffness (iy), lower gravity sag, and higher Z stiffness than front
    assert xg_top.profile.iy_m4 == pytest.approx(xg_bottom.profile.iy_m4, rel=1e-9)
    assert xg_top.profile.iy_m4 > xg_front.profile.iy_m4
    assert st_top.gravity_sag_toolhead_um < st_front.gravity_sag_toolhead_um
    assert st_bottom.gravity_sag_toolhead_um < st_front.gravity_sag_toolhead_um
    assert st_top.toolhead_stiffness_z_n_per_mm > st_front.toolhead_stiffness_z_n_per_mm
    assert st_bottom.toolhead_stiffness_z_n_per_mm > st_front.toolhead_stiffness_z_n_per_mm

    # bottom gives shorter nozzle Z drop & lower nozzle_twist_amplification_pct than top and front
    def _nozzle_z_drop_mm(asm: AssemblyModel) -> float:
        nmap = asm.node_map
        return float((nmap["toolhead"].position_m[2] - nmap["nozzle_tip"].position_m[2]) * 1e3)

    drop_front = _nozzle_z_drop_mm(asm_front)
    drop_top = _nozzle_z_drop_mm(asm_top)
    drop_bottom = _nozzle_z_drop_mm(asm_bottom)
    assert drop_bottom == pytest.approx(drop_front - 12.0, abs=1e-6)
    assert drop_top == pytest.approx(drop_front + 6.0, abs=1e-6)
    assert drop_bottom < drop_front < drop_top

    assert st_bottom.nozzle_twist_amplification_pct < st_front.nozzle_twist_amplification_pct
    assert st_bottom.nozzle_twist_amplification_pct < st_top.nozzle_twist_amplification_pct
    assert st_bottom.nozzle_twist_angle_mrad < st_front.nozzle_twist_angle_mrad
    assert st_bottom.nozzle_twist_angle_mrad < st_top.nozzle_twist_angle_mrad

    # dual_top_bottom gives highest X-beam Z-bending iy & Z stiffness; on a fixed-gantry (Trident) frame
    # it also gives lowest gravity sag (on a belted flying gantry the 2nd steel rail's mass stretches Z-belts)
    assert xg_dual.profile.iy_m4 > xg_top.profile.iy_m4
    assert xg_dual.profile.iy_m4 > xg_front.profile.iy_m4
    assert st_dual.toolhead_stiffness_z_n_per_mm > st_top.toolhead_stiffness_z_n_per_mm
    assert st_dual.toolhead_stiffness_z_n_per_mm > st_front.toolhead_stiffness_z_n_per_mm
    s_top_fix = _corexy_spec(x_rail_mount="top")
    s_top_fix.bed.z_drive = "leadscrew_triple"
    s_top_fix.bed.mount_type = "fixed_4pt"
    s_top_fix.bed.z_leadscrew_type = "sfu1204_bk10"
    s_dual_fix = _corexy_spec(x_rail_mount="dual_top_bottom")
    s_dual_fix.bed.z_drive = "leadscrew_triple"
    s_dual_fix.bed.mount_type = "fixed_4pt"
    s_dual_fix.bed.z_leadscrew_type = "sfu1204_bk10"
    assert solve_static_fea(build_assembly(s_dual_fix)).gravity_sag_toolhead_um < solve_static_fea(build_assembly(s_top_fix)).gravity_sag_toolhead_um

    # Clamping edge cases: small Y offset doesn't flip sign on top; small Z drop clamps to >= 10 mm on bottom
    s_clamp = _corexy_spec(x_rail_mount="top")
    s_clamp.toolhead.cog_offset_mm = (0.0, -3.0, -12.0)
    s_clamp.toolhead.nozzle_offset_mm = (0.0, -4.0, -16.0)
    asm_c_top = build_assembly(s_clamp)
    assert asm_c_top.node_map["toolhead_cog"].position_m[1] == pytest.approx(
        asm_c_top.node_map["toolhead"].position_m[1], abs=1e-9
    )
    assert asm_c_top.node_map["nozzle_tip"].position_m[1] == pytest.approx(
        asm_c_top.node_map["toolhead"].position_m[1], abs=1e-9
    )

    s_clamp_bot = s_clamp.model_copy(deep=True)
    s_clamp_bot.gantry.x_rail_mount = "bottom"
    asm_c_bot = build_assembly(s_clamp_bot)
    assert _nozzle_z_drop_mm(asm_c_bot) >= 10.0 - 1e-6


def test_cartesian_bedslinger_x_rail_mount_stiffening() -> None:
    """Cartesian bedslinger applies composite X-rail stiffening and nozzle lever-arm shifts."""
    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "cartesian_bedslinger.yaml")
    s_front = base.model_copy(deep=True)
    s_front.gantry.x_rail_mount = "front"
    s_top = base.model_copy(deep=True)
    s_top.gantry.x_rail_mount = "top"
    s_bot = base.model_copy(deep=True)
    s_bot.gantry.x_rail_mount = "bottom"
    s_dual = base.model_copy(deep=True)
    s_dual.gantry.x_rail_mount = "dual_top_bottom"

    asm_f = build_assembly(s_front)
    asm_t = build_assembly(s_top)
    asm_b = build_assembly(s_bot)
    asm_d = build_assembly(s_dual)

    st_f = solve_static_fea(asm_f)
    st_t = solve_static_fea(asm_t)
    st_b = solve_static_fea(asm_b)

    assert st_f.toolhead_stiffness_y_n_per_mm > st_t.toolhead_stiffness_y_n_per_mm
    assert st_t.toolhead_stiffness_z_n_per_mm > st_f.toolhead_stiffness_z_n_per_mm
    assert st_b.nozzle_twist_amplification_pct < st_f.nozzle_twist_amplification_pct
    assert st_b.nozzle_twist_amplification_pct < st_t.nozzle_twist_amplification_pct
    assert next(m for m in asm_d.members if m.id == "x_gantry_l").profile.iy_m4 > next(
        m for m in asm_t.members if m.id == "x_gantry_l"
    ).profile.iy_m4


# ---------------------------------------------------------------------------
# 3. Dual Top+Bottom Equivalence, Moving Mass & BOM Integration
# ---------------------------------------------------------------------------


def test_dual_top_bottom_matches_legacy_dual_x_rails_mass_and_bom() -> None:
    """dual_top_bottom matches legacy dual_x_rails=True, adds exactly 1 X rail mass, and updates BOM."""
    spec_front = _corexy_spec(x_rail_mount="front", dual_x_rails=False)
    spec_dual_mount = _corexy_spec(x_rail_mount="dual_top_bottom", dual_x_rails=False)
    spec_dual_legacy = _corexy_spec(x_rail_mount="front", dual_x_rails=True)

    asm_front = build_assembly(spec_front)
    asm_dual_mount = build_assembly(spec_dual_mount)
    asm_dual_legacy = build_assembly(spec_dual_legacy)

    # No legacy x_rail_aux_* duplicate members are created
    assert not any(m.id.startswith("x_rail_aux_") for m in asm_dual_mount.members)
    assert not any(m.id.startswith("x_rail_aux_") for m in asm_dual_legacy.members)

    # Exact equivalence between x_rail_mount="dual_top_bottom" and legacy dual_x_rails=True
    assert asm_dual_mount.y_moving_mass_kg == pytest.approx(asm_dual_legacy.y_moving_mass_kg, rel=1e-12)
    assert asm_dual_mount.total_mass_kg == pytest.approx(asm_dual_legacy.total_mass_kg, rel=1e-12)
    assert asm_dual_mount.estimated_material_cost_usd == pytest.approx(
        asm_dual_legacy.estimated_material_cost_usd, rel=1e-12
    )
    assert asm_dual_mount.to_geometry_dict()["linear_rails"] == asm_dual_legacy.to_geometry_dict()["linear_rails"]
    assert solve_static_fea(asm_dual_mount).toolhead_stiffness_z_n_per_mm == pytest.approx(
        solve_static_fea(asm_dual_legacy).toolhead_stiffness_z_n_per_mm, rel=1e-9
    )
    assert solve_modal_fea(asm_dual_mount).fundamental_freq_hz == pytest.approx(
        solve_modal_fea(asm_dual_legacy).fundamental_freq_hz, rel=1e-9
    )

    # Exact +1 extra X rail mass (+ 1 extra 54 g MGN12H carriage block on moving gantry/toolhead)
    from printsim.geometry.builder import corexy_x_rail_len_m

    x_rail_prof = get_profile(spec_front.gantry.x_rail_profile)
    rail_mat = get_material(spec_front.gantry.rail_material)
    x_rail_len_m = corexy_x_rail_len_m(spec_front)
    one_x_rail_kg = x_rail_prof.area_m2 * x_rail_len_m * rail_mat.density_kg_m3
    extra_block_kg = 0.054

    assert (asm_dual_mount.x_moving_mass_kg - asm_front.x_moving_mass_kg) == pytest.approx(extra_block_kg, rel=1e-9)
    assert (asm_dual_mount.toolhead_moving_mass_kg - asm_front.toolhead_moving_mass_kg) == pytest.approx(
        extra_block_kg, rel=1e-9
    )
    assert (asm_dual_mount.y_moving_mass_kg - asm_front.y_moving_mass_kg) == pytest.approx(
        one_x_rail_kg + extra_block_kg, rel=1e-9
    )
    assert (asm_dual_mount.frame_mass_kg - asm_front.frame_mass_kg) == pytest.approx(one_x_rail_kg, rel=1e-9)
    assert (asm_dual_mount.total_mass_kg - asm_front.total_mass_kg) == pytest.approx(
        one_x_rail_kg + extra_block_kg, rel=1e-9
    )
    assert (asm_dual_mount.total_rail_mass_kg - asm_front.total_rail_mass_kg) == pytest.approx(one_x_rail_kg, rel=1e-9)

    # BOM updates X-rail & carriage qty to 2, adds M3 rail screws, and records mount note
    bom_front = generate_bom(spec_front)
    bom_dual = generate_bom(spec_dual_mount)
    assert bom_dual.to_dict() == generate_bom(spec_dual_legacy).to_dict()

    x_rail_f = next(i for i in bom_front.items if i.sku == "MOT-X-RAIL")
    x_rail_d = next(i for i in bom_dual.items if i.sku == "MOT-X-RAIL")
    x_car_d = next(i for i in bom_dual.items if i.sku == "MOT-X-CAR")
    y_rail_f = next(i for i in bom_front.items if i.sku == "MOT-Y-RAIL")
    m3_f = next(i for i in bom_front.items if i.sku == "FST-M3-RAIL")
    m3_d = next(i for i in bom_dual.items if i.sku == "FST-M3-RAIL")

    assert x_rail_f.quantity == 1
    assert x_rail_d.quantity == 2
    assert x_car_d.quantity == 2
    assert m3_d.quantity > m3_f.quantity
    assert "front" in x_rail_f.specification.lower()
    assert "dual" in x_rail_d.specification.lower()
    assert "top" in y_rail_f.specification.lower()
    assert all(c.profile_id != spec_dual_mount.gantry.x_rail_profile for c in bom_dual.cut_list)


# ---------------------------------------------------------------------------
# 4. Y-Rail Mounting Orientations (top vs inner vs bottom)
# ---------------------------------------------------------------------------


def test_y_rail_mount_inner_boosts_x_bending_and_toolhead_x_stiffness() -> None:
    """y_rail_mount='inner' boosts Y-rail X-bending iz and toolhead X stiffness over 'top'/'bottom'."""
    for base_spec in (_corexy_spec(), PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_monolith_awd.yaml")):
        s_top = base_spec.model_copy(deep=True)
        s_top.gantry.y_rail_mount = "top"
        s_inner = base_spec.model_copy(deep=True)
        s_inner.gantry.y_rail_mount = "inner"
        s_bottom = base_spec.model_copy(deep=True)
        s_bottom.gantry.y_rail_mount = "bottom"

        asm_top = build_assembly(s_top)
        asm_inner = build_assembly(s_inner)
        asm_bottom = build_assembly(s_bottom)

        st_top = solve_static_fea(asm_top)
        st_inner = solve_static_fea(asm_inner)
        st_bottom = solve_static_fea(asm_bottom)

        yr_top = next(m for m in asm_top.members if m.id == "y_rail_l_f")
        yr_inner = next(m for m in asm_inner.members if m.id == "y_rail_l_f")
        yr_bottom = next(m for m in asm_bottom.members if m.id == "y_rail_l_f")

        assert yr_inner.profile.iz_m4 > yr_top.profile.iz_m4
        assert yr_top.profile.iy_m4 == pytest.approx(yr_bottom.profile.iy_m4, rel=1e-9)
        assert asm_inner.y_rail_ei_x_boost_pct > asm_top.y_rail_ei_x_boost_pct * 2.0
        assert st_inner.toolhead_stiffness_x_n_per_mm > st_top.toolhead_stiffness_x_n_per_mm
        assert st_inner.toolhead_stiffness_x_n_per_mm > st_bottom.toolhead_stiffness_x_n_per_mm

        bom_inner = generate_bom(s_inner)
        y_rail_item = next(i for i in bom_inner.items if i.sku == "MOT-Y-RAIL")
        assert "inner" in y_rail_item.specification.lower()


# ---------------------------------------------------------------------------
# 5. Zero Mass/Cost Double-Counting: Carbon Fiber X-Beam + Epoxy Granite Fill
# ---------------------------------------------------------------------------


def test_zero_mass_cost_double_counting_with_cf_beam_and_epoxy_granite() -> None:
    """Verify exact mass & cost separation when combining CF X-beam, steel rails, and epoxy granite."""
    s_cf_none_front = _corexy_spec(
        x_beam_profile="cf_tube_20x20x2",
        x_beam_material="carbon_fiber_tube",
        x_rail_profile="mgn12_rail",
        y_rail_profile="mgn12_rail",
        x_rail_mount="front",
        extrusion_filling="none",
    )
    s_cf_epoxy_front = s_cf_none_front.model_copy(deep=True)
    s_cf_epoxy_front.frame.extrusion_filling = "epoxy_granite"

    s_cf_epoxy_dual = s_cf_epoxy_front.model_copy(deep=True)
    s_cf_epoxy_dual.gantry.x_rail_mount = "dual_top_bottom"

    asm_0 = build_assembly(s_cf_none_front)
    asm_f = build_assembly(s_cf_epoxy_front)
    asm_d = build_assembly(s_cf_epoxy_dual)

    from printsim.geometry.builder import corexy_x_beam_span_m, corexy_x_rail_len_m

    x_beam_span_m = corexy_x_beam_span_m(s_cf_none_front)
    x_rail_len_m = corexy_x_rail_len_m(s_cf_none_front)
    from printsim.geometry.builder import corexy_frame_centerlines_m

    _, d_m, h_m = corexy_frame_centerlines_m(s_cf_none_front)
    cf_prof = get_profile("cf_tube_20x20x2")
    cf_mat = get_material("carbon_fiber_tube")
    steel_mat = get_material("steel_4140")
    x_rail_prof = get_profile("mgn12_rail")
    y_rail_prof = get_profile("mgn12_rail")

    bare_cf_kg = cf_prof.area_m2 * x_beam_span_m * cf_mat.density_kg_m3
    one_x_rail_kg = x_rail_prof.area_m2 * x_rail_len_m * steel_mat.density_kg_m3
    two_y_rails_kg = 2.0 * y_rail_prof.area_m2 * d_m * steel_mat.density_kg_m3
    z_rail_prof = get_profile(s_cf_none_front.bed.z_rail_profile)
    four_z_rails_kg = 4.0 * z_rail_prof.area_m2 * h_m * steel_mat.density_kg_m3

    # 1. Exact total_rail_mass_kg across X, Y, and Z rails
    assert asm_f.total_rail_mass_kg == pytest.approx(
        one_x_rail_kg + two_y_rails_kg + four_z_rails_kg, rel=1e-9
    )
    assert asm_d.total_rail_mass_kg == pytest.approx(
        2.0 * one_x_rail_kg + two_y_rails_kg + four_z_rails_kg, rel=1e-9
    )

    # 2. Exact y_moving_mass_kg (NO double-counting of x_rail_mass!)
    expected_y_mov_f = (
        (s_cf_epoxy_front.toolhead.total_mass_g + s_cf_epoxy_front.toolhead.hotend.mass_g) * 1e-3
        + s_cf_epoxy_front.toolhead.effective_umbilical_mass_g * 1e-3
        + bare_cf_kg
        + one_x_rail_kg
        + s_cf_epoxy_front.gantry.carriage_mass_g * 1e-3
    )
    expected_y_mov_d = expected_y_mov_f + one_x_rail_kg + 0.054
    assert asm_f.y_moving_mass_kg == pytest.approx(expected_y_mov_f, rel=1e-9)
    assert asm_d.y_moving_mass_kg == pytest.approx(expected_y_mov_d, rel=1e-9)

    # 3. X-gantry CF beam is NOT cavity-filled; stationary Y-rails ARE filled without wiping rail_mass_kg
    for xg in (m for m in asm_f.members if m.role == "gantry"):
        assert xg.fill_mass_kg == 0.0
        assert xg.rail_mass_kg > 0.0
    for yr in (m for m in asm_f.members if m.id.startswith("y_rail_")):
        assert yr.fill_mass_kg > 0.0
        assert yr.rail_mass_kg > 0.0

    # 4. Exact epoxy_granite mass & cost delta (no aluminum/CF $/kg double-charge)
    fill_cost_per_kg = float(EXTRUSION_FILLINGS["epoxy_granite"]["cost_per_kg_usd"])
    assert (asm_f.total_mass_kg - asm_0.total_mass_kg) == pytest.approx(asm_f.extrusion_fill_mass_kg, rel=1e-6)
    assert (asm_f.estimated_material_cost_usd - asm_0.estimated_material_cost_usd) == pytest.approx(
        asm_f.extrusion_fill_mass_kg * fill_cost_per_kg, rel=1e-6
    )

    # 5. Adding 2nd steel rail onto CF X-beam charges steel $/kg ($6/kg), NEVER CF $/kg ($95/kg)!
    expected_2nd_rail_cost = one_x_rail_kg * steel_mat.cost_per_kg_usd
    assert (asm_d.estimated_material_cost_usd - asm_f.estimated_material_cost_usd) == pytest.approx(
        expected_2nd_rail_cost, rel=1e-6
    )
    assert (asm_d.estimated_material_cost_usd - asm_f.estimated_material_cost_usd) < one_x_rail_kg * cf_mat.cost_per_kg_usd * 0.2


# ---------------------------------------------------------------------------
# 6. Geometry Dict Telemetry & FastAPI Integration
# ---------------------------------------------------------------------------


def test_geometry_dict_and_api_expose_linear_rails_telemetry() -> None:
    """Verify AssemblyModel.to_geometry_dict()['linear_rails'] and FastAPI /api/catalog + /api/simulate."""
    spec = _corexy_spec(x_rail_mount="bottom", y_rail_mount="inner", x_rail_profile="mgn12_rail", y_rail_profile="mgn9_rail")
    asm = build_assembly(spec)
    gdict = asm.to_geometry_dict()

    lr = gdict["linear_rails"]
    assert lr == {
        "x_rail_profile": "mgn12_rail",
        "y_rail_profile": "mgn9_rail",
        "rail_material": "steel_4140",
        "rail_quality": "hiwin_genuine",
        "rail_preload": "Z1",
        "x_rail_mount": "bottom",
        "y_rail_mount": "inner",
        "cf_rail_mounting": "none",
        "dual_x_rails": False,
        "x_rail_width_mm": 12.0,
        "x_rail_height_mm": 8.0,
        "y_rail_width_mm": 9.0,
        "y_rail_height_mm": 6.5,
        "x_beam_span_mm": 380.0,
        "x_beam_ei_y_boost_pct": round(asm.x_rail_ei_y_boost_pct, 1),
        "x_beam_ei_z_boost_pct": round(asm.x_rail_ei_z_boost_pct, 1),
        "x_beam_gj_boost_pct": round(asm.x_rail_gj_boost_pct, 1),
        "x_rail_newmark_gamma": round(asm.x_rail_newmark_gamma, 4),
        "x_beam_insert_mass_g": round(asm.x_beam_insert_mass_kg * 1e3, 1),
        "x_carriage_block_mass_g": round(asm.x_carriage_block_mass_g, 1),
        "x_carriage_block_delta_g": round(asm.x_carriage_block_delta_g, 1),
        "x_carriage_k_rad_n_per_um": round(asm.x_carriage_k_rad_n_per_um, 1),
        "x_carriage_k_rx_nm_rad": round(asm.x_carriage_k_rx_nm_rad, 1),
        "x_carriage_k_ry_nm_rad": round(asm.x_carriage_k_ry_nm_rad, 1),
        "y_rail_ei_x_boost_pct": round(asm.y_rail_ei_x_boost_pct, 1),
        "y_carriage_block_mass_g": round(asm.y_carriage_block_mass_g, 1),
        "y_carriage_block_delta_g": round(asm.y_carriage_block_delta_g, 1),
        "y_carriage_k_rx_nm_rad": round(asm.y_carriage_k_rx_nm_rad, 1),
        "total_rail_mass_g": round(asm.total_rail_mass_kg * 1e3, 1),
    }
    assert lr["x_beam_ei_z_boost_pct"] > lr["x_beam_ei_y_boost_pct"]
    assert lr["x_beam_gj_boost_pct"] > 0.0
    assert 0.0 < lr["x_rail_newmark_gamma"] < 1.0
    assert lr["x_carriage_block_mass_g"] == 54.0
    assert lr["x_carriage_block_delta_g"] == 0.0
    assert lr["x_carriage_k_rad_n_per_um"] > 50.0
    assert lr["x_carriage_k_rx_nm_rad"] > 500.0
    assert lr["x_carriage_k_ry_nm_rad"] > 1000.0
    assert lr["y_rail_ei_x_boost_pct"] > 90.0
    assert lr["y_carriage_block_mass_g"] == 26.0
    assert lr["y_carriage_k_rx_nm_rad"] > 300.0
    assert lr["total_rail_mass_g"] > 700.0

    client = TestClient(create_app())
    r_cat = client.get("/api/catalog")
    assert r_cat.status_code == 200
    cat = r_cat.json()
    assert {m["id"] for m in cat["x_rail_mounts"]} == {"front", "top", "bottom", "dual_top_bottom"}
    assert {m["id"] for m in cat["y_rail_mounts"]} == {"top", "inner", "bottom"}

    r_sim = client.post("/api/simulate", json={"spec": spec.model_dump(mode="json")})
    assert r_sim.status_code == 200
    sim = r_sim.json()
    assert sim["geometry"]["linear_rails"]["x_rail_mount"] == "bottom"
    assert sim["geometry"]["linear_rails"]["y_rail_mount"] == "inner"
    assert sim["geometry"]["linear_rails"]["x_beam_ei_z_boost_pct"] == lr["x_beam_ei_z_boost_pct"]
    assert sim["geometry"]["linear_rails"]["x_carriage_block_mass_g"] == 54.0


# ---------------------------------------------------------------------------
# 7. Official Carriage Masses, Dual-Rail Steiner Roll, Sandwich Backer & Telemetry
# ---------------------------------------------------------------------------


def test_official_hiwin_carriage_block_masses_and_moving_mass_deltas() -> None:
    """Verify official HIWIN/Misumi carriage block masses & exact moving-mass deltas across architectures."""
    expected_catalog_masses = [
        ("mgn9_rail", "C", 16.0),
        ("mgn9_rail", "H", 26.0),
        ("mgn12_rail", "C", 34.0),
        ("mgn12_rail", "H", 54.0),
        ("mgn15_rail", "C", 59.0),
        ("mgn15_rail", "H", 92.0),
        ("hgh15_rail", "C", 180.0),
        ("hgh15_rail", "H", 180.0),
        ("steel_rod_8mm", "C", 16.0),
        ("steel_rod_8mm", "H", 31.0),
        ("steel_rod_10mm", "C", 31.0),
        ("steel_rod_10mm", "H", 56.0),
        ("steel_rod_12mm", "C", 41.0),
        ("steel_rod_12mm", "H", 76.0),
    ]
    for rail_id, block_type, exp_g in expected_catalog_masses:
        res = hertzian_rail_carriage_stiffness(rail_id, block_type=block_type)
        assert res["block_mass_g"] == pytest.approx(exp_g, abs=1e-9)

    # Check helper x_carriage_block_mass_delta_g exact deltas vs single MGN12H (54.0 g)
    helper_cases = [
        ("mgn12_rail", "H", False, 54.0, 0.0),
        ("mgn12_rail", "C", False, 34.0, -20.0),
        ("mgn9_rail", "H", False, 26.0, -28.0),
        ("mgn9_rail", "C", False, 16.0, -38.0),
        ("mgn15_rail", "H", False, 92.0, 38.0),
        ("hgh15_rail", "C", False, 180.0, 126.0),
        ("mgn12_rail", "H", True, 108.0, 54.0),
        ("mgn9_rail", "H", True, 52.0, -2.0),
    ]
    for rail_id, bt, is_dual, exp_tot_g, exp_delta_g in helper_cases:
        s = _corexy_spec(x_rail_profile=rail_id, x_rail_mount="dual_top_bottom" if is_dual else "front")
        s.gantry.rail_block_type = bt  # type: ignore[assignment]
        tot_g, d_g = x_carriage_block_mass_delta_g(s)
        assert tot_g == pytest.approx(exp_tot_g, abs=1e-9)
        assert d_g == pytest.approx(exp_delta_g, abs=1e-9)

    # Check CoreXY & Bedslinger assembly moving-mass deltas and exact mass conservation
    base_corexy = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    asm_ref = build_assembly(base_corexy)
    assert x_carriage_block_mass_delta_g(base_corexy) == pytest.approx((54.0, 0.0), abs=1e-9)
    assert asm_ref.node_map["toolhead"].lumped_mass_kg == pytest.approx(0.054, abs=1e-9)

    for rail_id, bt, is_dual, exp_tot_g, exp_delta_g in [
        ("mgn9_rail", "C", False, 16.0, -38.0),
        ("mgn9_rail", "H", False, 26.0, -28.0),
        ("mgn12_rail", "C", False, 34.0, -20.0),
        ("mgn15_rail", "H", False, 92.0, 38.0),
        ("mgn12_rail", "H", True, 108.0, 54.0),
    ]:
        s = base_corexy.model_copy(deep=True)
        s.gantry.x_rail_profile = rail_id
        s.gantry.rail_block_type = bt  # type: ignore[assignment]
        s.gantry.x_rail_mount = "dual_top_bottom" if is_dual else "front"
        asm = build_assembly(s)
        nmap = asm.node_map

        assert (asm.toolhead_moving_mass_kg - asm_ref.toolhead_moving_mass_kg) * 1000.0 == pytest.approx(
            exp_delta_g, abs=1e-9
        )
        assert (asm.x_moving_mass_kg - asm_ref.x_moving_mass_kg) * 1000.0 == pytest.approx(
            exp_delta_g, abs=1e-9
        )
        assert nmap["toolhead"].lumped_mass_kg == pytest.approx(exp_tot_g * 1e-3, abs=1e-9)
        assert nmap["toolhead_cog"].lumped_mass_kg >= 0.02 - 1e-12
        assert (
            nmap["toolhead"].lumped_mass_kg
            + nmap["toolhead_cog"].lumped_mass_kg
            + nmap["nozzle_tip"].lumped_mass_kg
        ) == pytest.approx(asm.toolhead_moving_mass_kg, rel=1e-12)
        assert (
            asm.toolhead_moving_mass_kg
            + s.toolhead.effective_umbilical_mass_g * 1e-3
            + asm.x_beam_mass_kg
            + asm.x_rail_mass_kg
            + asm.x_backer_mass_kg
            + asm.y_carriage_joints_mass_kg
        ) == pytest.approx(asm.y_moving_mass_kg, rel=1e-12)

    # Bedslinger X moving mass also shifts by exact carriage block delta
    bs_base = PrinterSpec.from_yaml(EXAMPLES_DIR / "cartesian_bedslinger.yaml")
    bs_9c = bs_base.model_copy(deep=True)
    bs_9c.gantry.x_rail_profile = "mgn9_rail"
    bs_9c.gantry.rail_block_type = "C"
    assert (build_assembly(bs_9c).x_moving_mass_kg - build_assembly(bs_base).x_moving_mass_kg) * 1000.0 == pytest.approx(
        -38.0, abs=1e-9
    )

    # Delta architecture returns (block_mass_g, 0.0) and zero moving-mass delta
    dl_base = PrinterSpec.from_yaml(EXAMPLES_DIR / "delta_racer.yaml")
    dl_9c = dl_base.model_copy(deep=True)
    dl_9c.gantry.x_rail_profile = "mgn9_rail"
    dl_9c.gantry.rail_block_type = "C"
    assert x_carriage_block_mass_delta_g(dl_9c) == pytest.approx((16.0, 0.0), abs=1e-9)
    asm_dl_base = build_assembly(dl_base)
    asm_dl_9c = build_assembly(dl_9c)
    assert asm_dl_9c.x_moving_mass_kg == pytest.approx(asm_dl_base.x_moving_mass_kg, rel=1e-12)
    assert asm_dl_9c.y_moving_mass_kg == pytest.approx(asm_dl_base.y_moving_mass_kg, rel=1e-12)


def test_dual_top_bottom_steiner_roll_stiffness_and_monotonic_preload_quality() -> None:
    """dual_top_bottom parallel-axis Steiner roll stiffness exceeds 3x single-rail & stays strictly monotonic."""
    s_single = _corexy_spec(x_rail_mount="top")
    s_dual = _corexy_spec(x_rail_mount="dual_top_bottom")

    p_single = compute_toolhead_link_props(s_single, l_mount_m=0.02, l_stem_m=0.04)
    p_dual = compute_toolhead_link_props(s_dual, l_mount_m=0.02, l_stem_m=0.04)

    car = hertzian_rail_carriage_stiffness(
        s_dual.gantry.x_rail_profile,
        preload=s_dual.gantry.rail_preload,
        block_type=s_dual.gantry.rail_block_type,
        rail_quality=s_dual.gantry.rail_quality,
    )
    z_track = 0.5 * (
        get_profile(s_dual.gantry.x_beam_profile).height_m
        + get_profile(s_dual.gantry.x_rail_profile).height_m
    )
    eta_ext = dual_rail_steiner_efficiency(
        s_dual.gantry.x_beam_profile,
        s_dual.gantry.effective_cf_rail_mounting,
    )
    assert eta_ext == pytest.approx(0.45, abs=1e-9)
    expected_dual_krx = 2.0 * car["k_rx_nm_rad"] + eta_ext * 2.0 * car["k_yy_n_m"] * (z_track**2)

    assert p_dual["k_rx_car"] == pytest.approx(expected_dual_krx, rel=1e-9)
    assert p_dual["k_rx_car"] > 3.0 * p_single["k_rx_car"]
    assert p_dual["k_ry_car"] == pytest.approx(2.0 * p_single["k_ry_car"], rel=1e-9)
    assert p_dual["k_rz_car"] == pytest.approx(2.0 * p_single["k_rz_car"], rel=1e-9)
    assert p_dual["k_rad_car"] == pytest.approx(2.0 * p_single["k_rad_car"], rel=1e-9)

    asm_single = build_assembly(s_single)
    asm_dual = build_assembly(s_dual)
    lr_single = asm_single.to_geometry_dict()["linear_rails"]
    lr_dual = asm_dual.to_geometry_dict()["linear_rails"]
    assert lr_dual["x_carriage_k_rx_nm_rad"] > 3.0 * lr_single["x_carriage_k_rx_nm_rad"]
    assert compute_thermal_structural_report(asm_dual).rail_carriage_roll_nm_per_rad == pytest.approx(
        expected_dual_krx, rel=1e-9
    )

    # CF hollow tube dual_top_bottom captures 3D diamond-lozenge efficiency (0.07 inserts vs 0.03 bare)
    s_cf_dual = _corexy_spec(
        x_beam_profile="cf_tube_20x20_carbonara",
        x_beam_material="carbon_fiber_torsional_45",
        x_rail_mount="dual_top_bottom",
    )
    s_cf_dual_bare = s_cf_dual.model_copy(deep=True)
    s_cf_dual_bare.gantry.cf_rail_mounting = "single_wall_bare"
    eta_cf_ins = dual_rail_steiner_efficiency("cf_tube_20x20_carbonara", "two_wall_clamped_inserts")
    eta_cf_bare = dual_rail_steiner_efficiency("cf_tube_20x20_carbonara", "single_wall_bare")
    assert eta_cf_ins == pytest.approx(0.07, abs=1e-9)
    assert eta_cf_bare == pytest.approx(0.03, abs=1e-9)
    p_cf_dual = compute_toolhead_link_props(s_cf_dual, l_mount_m=0.02, l_stem_m=0.04)
    p_cf_dual_bare = compute_toolhead_link_props(s_cf_dual_bare, l_mount_m=0.02, l_stem_m=0.04)
    assert p_dual["k_rx_car"] > p_cf_dual["k_rx_car"] > p_cf_dual_bare["k_rx_car"]

    # overrides.py rail_preload telemetry stays synchronized with dual_rail_steiner_efficiency
    from printsim.schema import build_model_confidence_audit

    audit_dual = {e["id"]: e for e in build_model_confidence_audit(s_cf_dual)["entries"]}["rail_preload"]
    assert audit_dual["trust_pct"] == 90
    assert f"{round(p_cf_dual['k_rx_car']):,}" in audit_dual["active_values"]

    # 2020 extrusion local T-slot shoulder seat: MGN12H > MGN9H in radial, roll, and pitch
    p_ext_9 = compute_toolhead_link_props(_corexy_spec(x_rail_profile="mgn9_rail"), 0.02, 0.04)
    assert p_single["k_rad_car"] > p_ext_9["k_rad_car"]
    assert p_single["k_rx_car"] > p_ext_9["k_rx_car"]
    assert p_single["k_ry_car"] > p_ext_9["k_ry_car"]

    # Monotonic across preload & rail quality on dual_top_bottom
    preloads = ("ZF", "Z0", "Z1", "ZA", "ZB")
    krx_pre = []
    for pre in preloads:
        sp = s_dual.model_copy(deep=True)
        sp.gantry.rail_preload = pre  # type: ignore[assignment]
        krx_pre.append(compute_toolhead_link_props(sp, 0.02, 0.04)["k_rx_car"])
    assert all(krx_pre[i] < krx_pre[i + 1] for i in range(len(krx_pre) - 1))

    qualities = ("budget_generic", "midgrade_matched", "hiwin_genuine", "precision_thk_nsk")
    krx_qual = []
    for q in qualities:
        sq = s_dual.model_copy(deep=True)
        sq.gantry.rail_quality = q  # type: ignore[assignment]
        krx_qual.append(compute_toolhead_link_props(sq, 0.02, 0.04)["k_rx_car"])
    assert all(krx_qual[i] < krx_qual[i + 1] for i in range(len(qualities) - 1))


def test_opposite_face_rail_backer_sandwich_neutral_axis_composite_ei() -> None:
    """Opposite-face steel backer + MGN12 yields higher composite EI than 2 independent asymmetric additions."""
    for x_prof, x_mat in (
        ("extrusion_2020", "aluminum_6063_t5"),
        ("cf_tube_20x20_carbonara", "carbon_fiber_torsional_45"),
    ):
        s_single = _corexy_spec(x_beam_profile=x_prof, x_beam_material=x_mat, x_rail_mount="top")
        s_bk = s_single.model_copy(deep=True)
        s_bk.gantry.rail_backer_material = "steel_4140"
        s_bk.gantry.rail_backer_thickness_mm = 3.0

        asm_single = build_assembly(s_single)
        asm_bk = build_assembly(s_bk)

        base_prof = get_profile(x_prof)
        xg_single = next(m for m in asm_single.members if m.id == "x_gantry_l")
        xg_bk = next(m for m in asm_bk.members if m.id == "x_gantry_l")

        # Single-sided asymmetric rail increment
        d_iy_single = xg_single.profile.iy_m4 - base_prof.iy_m4
        two_independent_iy = base_prof.iy_m4 + 2.0 * d_iy_single
        assert xg_bk.profile.iy_m4 > two_independent_iy

        # Exact closed-form sandwich neutral-axis calculation check
        rail_prof = get_profile("mgn12_rail")
        rail_mat = get_material("steel_4140")
        beam_mat = get_material(x_mat)
        from printsim.geometry.builder import _make_x_backer_profile, corexy_frame_centerlines_m

        bk_prof = _make_x_backer_profile(base_prof, rail_prof, 3.0)
        l_w, _, _ = corexy_frame_centerlines_m(s_bk)
        sub_kind = "carbon_tube" if x_prof.startswith("cf_") else "extrusion"
        eta_r = newmark_rail_composite_gamma(
            rail_mat.youngs_modulus_pa,
            rail_prof.area_m2,
            beam_mat.youngs_modulus_pa,
            base_prof.area_m2,
            0.025,
            l_w,
            sub_kind,
            wall_thickness_m=base_prof.wall_thickness_m,
            cf_rail_mounting=s_bk.gantry.effective_cf_rail_mounting,
        )
        eta_b = newmark_rail_composite_gamma(
            rail_mat.youngs_modulus_pa,
            bk_prof.area_m2,
            beam_mat.youngs_modulus_pa,
            base_prof.area_m2,
            0.025,
            l_w,
            sub_kind,
            wall_thickness_m=base_prof.wall_thickness_m,
            cf_rail_mounting=s_bk.gantry.effective_cf_rail_mounting,
        )
        n_e = rail_mat.youngs_modulus_pa / beam_mat.youngs_modulus_pa
        d_1 = 0.5 * base_prof.height_m + 0.5 * rail_prof.height_m
        d_2 = 0.5 * base_prof.height_m + 0.5 * bk_prof.height_m
        a_r_eff = eta_r * n_e * rail_prof.area_m2
        a_b_eff = eta_b * n_e * bk_prof.area_m2
        y_na = (a_r_eff * d_1 - a_b_eff * d_2) / (base_prof.area_m2 + a_r_eff + a_b_eff)
        i_par = base_prof.area_m2 * (y_na**2) + a_r_eff * ((d_1 - y_na) ** 2) + a_b_eff * ((d_2 + y_na) ** 2)
        expected_iy = base_prof.iy_m4 + n_e * (rail_prof.iy_m4 + bk_prof.iy_m4) + i_par
        assert xg_bk.profile.iy_m4 == pytest.approx(expected_iy, rel=1e-9)

        # Bedslinger also uses exact sandwich neutral-axis composite when rail_backer_material is set
        bs_bk = PrinterSpec.from_yaml(EXAMPLES_DIR / "cartesian_bedslinger.yaml")
        bs_bk.gantry.x_beam_profile = x_prof
        bs_bk.gantry.x_beam_material = x_mat
        bs_bk.gantry.x_rail_mount = "top"
        bs_bk.gantry.rail_backer_material = "steel_4140"
        bs_bk.gantry.rail_backer_thickness_mm = 3.0
        asm_bs_bk = build_assembly(bs_bk)
        assert asm_bs_bk.to_geometry_dict()["linear_rails"]["x_beam_gj_boost_pct"] > 0.0
        assert next(m for m in asm_bs_bk.members if m.id == "x_gantry_l").profile.iy_m4 > two_independent_iy


def test_linear_rails_telemetry_and_y_carriage_compliance() -> None:
    """All new geometry.linear_rails fields populate validly & Y-carriage roll stiffness propagates to CoreXY."""
    required_new_keys = {
        "cf_rail_mounting",
        "x_beam_gj_boost_pct",
        "x_rail_newmark_gamma",
        "x_beam_insert_mass_g",
        "x_carriage_block_mass_g",
        "x_carriage_block_delta_g",
        "x_carriage_k_rad_n_per_um",
        "x_carriage_k_rx_nm_rad",
        "x_carriage_k_ry_nm_rad",
        "y_carriage_block_mass_g",
        "y_carriage_block_delta_g",
        "y_carriage_k_rx_nm_rad",
    }

    for yaml_name in (
        "corexy_voron_style.yaml",
        "voron_monolith_awd.yaml",
        "hyper_corexy_carbon.yaml",
        "cartesian_bedslinger.yaml",
        "delta_racer.yaml",
    ):
        spec = PrinterSpec.from_yaml(EXAMPLES_DIR / yaml_name)
        asm = build_assembly(spec)
        lr = asm.to_geometry_dict()["linear_rails"]
        assert required_new_keys.issubset(lr.keys()), yaml_name
        assert lr["x_carriage_block_mass_g"] > 0.0
        assert lr["x_carriage_k_rad_n_per_um"] > 0.0
        assert lr["x_carriage_k_rx_nm_rad"] > 0.0
        assert lr["x_carriage_k_ry_nm_rad"] > 0.0
        if spec.architecture == "delta":
            assert lr["x_carriage_block_delta_g"] == 0.0
            assert lr["x_rail_newmark_gamma"] == 0.0
            assert lr["y_carriage_block_mass_g"] == 0.0
            assert lr["y_carriage_k_rx_nm_rad"] == 0.0
        else:
            if spec.gantry.effective_cf_rail_mounting == "single_wall_bare":
                # Un-inserted hollow CF ends suffer torsional crush compliance vs bare clamped reference
                assert lr["x_beam_gj_boost_pct"] < 100.0
            else:
                assert lr["x_beam_gj_boost_pct"] > 0.0
            assert 0.0 < lr["x_rail_newmark_gamma"] < 1.0
            assert lr["y_carriage_block_mass_g"] > 0.0
            assert lr["y_carriage_k_rx_nm_rad"] > 0.0

    # CoreXY Y-rail upgrade (mgn9 -> mgn12 -> mgn15) monotonically increases Y-carriage telemetry & Z stiffness
    s9 = _corexy_spec(y_rail_profile="mgn9_rail")
    s12 = _corexy_spec(y_rail_profile="mgn12_rail")
    s15 = _corexy_spec(y_rail_profile="mgn15_rail")
    a9, a12, a15 = build_assembly(s9), build_assembly(s12), build_assembly(s15)
    lr9, lr12, lr15 = (
        a9.to_geometry_dict()["linear_rails"],
        a12.to_geometry_dict()["linear_rails"],
        a15.to_geometry_dict()["linear_rails"],
    )
    assert lr9["y_carriage_block_mass_g"] == 26.0
    assert lr12["y_carriage_block_mass_g"] == 54.0
    assert lr15["y_carriage_block_mass_g"] == 92.0
    assert lr9["y_carriage_k_rx_nm_rad"] < lr12["y_carriage_k_rx_nm_rad"] < lr15["y_carriage_k_rx_nm_rad"]
    assert solve_static_fea(a9).toolhead_stiffness_z_n_per_mm < solve_static_fea(a12).toolhead_stiffness_z_n_per_mm < solve_static_fea(a15).toolhead_stiffness_z_n_per_mm


def test_y_carriage_block_mass_deltas_and_bom_official_hiwin_weights() -> None:
    """Verify Y-carriage block weight deltas across CoreXY/Bedslinger/Delta and official BOM carriage weights."""
    # 1. Helper y_carriage_block_mass_delta_g exact deltas vs 2x MGN12H (108.0 g)
    helper_cases = [
        ("mgn9_rail", "C", 32.0, -76.0),
        ("mgn9_rail", "H", 52.0, -56.0),
        ("mgn12_rail", "C", 68.0, -40.0),
        ("mgn12_rail", "H", 108.0, 0.0),
        ("mgn15_rail", "C", 118.0, 10.0),
        ("mgn15_rail", "H", 184.0, 76.0),
        ("hgh15_rail", "C", 360.0, 252.0),
    ]
    for rail_id, bt, exp_tot_g, exp_delta_g in helper_cases:
        s = _corexy_spec(y_rail_profile=rail_id)
        s.gantry.rail_block_type = bt  # type: ignore[assignment]
        tot_g, d_g = y_carriage_block_mass_delta_g(s)
        assert tot_g == pytest.approx(exp_tot_g, abs=1e-9)
        assert d_g == pytest.approx(exp_delta_g, abs=1e-9)

    dl = PrinterSpec.from_yaml(EXAMPLES_DIR / "delta_racer.yaml")
    dl.gantry.y_rail_profile = "mgn9_rail"
    assert y_carriage_block_mass_delta_g(dl) == pytest.approx((26.0, 0.0), abs=1e-9)

    # 2. voron_monolith_awd: exact 256.0 g on stock MGN12H -> drops -56.0 g to 200.0 g on MGN9H
    s_mono_12 = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_monolith_awd.yaml")
    s_mono_9 = s_mono_12.model_copy(deep=True)
    s_mono_9.gantry.y_rail_profile = "mgn9_rail"
    a_mono_12 = build_assembly(s_mono_12)
    a_mono_9 = build_assembly(s_mono_9)
    assert a_mono_12.y_carriage_joints_mass_kg * 1e3 == pytest.approx(256.0, abs=1e-9)
    assert a_mono_9.y_carriage_joints_mass_kg * 1e3 == pytest.approx(200.0, abs=1e-9)
    assert a_mono_12.to_geometry_dict()["linear_rails"]["y_carriage_block_delta_g"] == 0.0
    assert a_mono_9.to_geometry_dict()["linear_rails"]["y_carriage_block_delta_g"] == -56.0
    assert (a_mono_9.y_moving_mass_kg - a_mono_12.y_moving_mass_kg) * 1e3 == pytest.approx(
        -56.0, abs=1e-9
    )
    assert (a_mono_9.node_map["y_car_l"].lumped_mass_kg - a_mono_12.node_map["y_car_l"].lumped_mass_kg) * 1e3 == pytest.approx(
        -28.0, abs=1e-9
    )

    # 3. voron_trident_300 & corexy_voron_style: exact 140.0 g & 145.0 g on stock MGN9H -> +56.0 g on MGN12H
    s_tri_9 = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    s_tri_12 = s_tri_9.model_copy(deep=True)
    s_tri_12.gantry.y_rail_profile = "mgn12_rail"
    a_tri_9 = build_assembly(s_tri_9)
    a_tri_12 = build_assembly(s_tri_12)
    assert a_tri_9.y_carriage_joints_mass_kg * 1e3 == pytest.approx(140.0, abs=1e-9)
    assert a_tri_12.y_carriage_joints_mass_kg * 1e3 == pytest.approx(196.0, abs=1e-9)
    assert a_tri_9.to_geometry_dict()["linear_rails"]["y_carriage_block_delta_g"] == -56.0
    assert a_tri_12.to_geometry_dict()["linear_rails"]["y_carriage_block_delta_g"] == 0.0

    s_v24_9 = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    s_v24_12 = s_v24_9.model_copy(deep=True)
    s_v24_12.gantry.y_rail_profile = "mgn12_rail"
    assert build_assembly(s_v24_9).y_carriage_joints_mass_kg * 1e3 == pytest.approx(145.0, abs=1e-9)
    assert build_assembly(s_v24_12).y_carriage_joints_mass_kg * 1e3 == pytest.approx(201.0, abs=1e-9)

    # 4. Bedslinger Y-rail block delta (-56.0 g when switching MGN12H -> MGN9H)
    bs_12 = PrinterSpec.from_yaml(EXAMPLES_DIR / "cartesian_bedslinger.yaml")
    bs_9 = bs_12.model_copy(deep=True)
    bs_9.gantry.y_rail_profile = "mgn9_rail"
    asm_bs_12 = build_assembly(bs_12)
    asm_bs_9 = build_assembly(bs_9)
    assert (asm_bs_9.y_carriage_joints_mass_kg - asm_bs_12.y_carriage_joints_mass_kg) * 1e3 == pytest.approx(
        -56.0, abs=1e-9
    )
    assert (asm_bs_9.y_moving_mass_kg - asm_bs_12.y_moving_mass_kg) * 1e3 == pytest.approx(
        -56.0, abs=1e-9
    )

    # 5. BOM uses official HIWIN C/H/HGH15CA block masses & dynamic C/H carriage names
    from printsim.geometry.bom import _rail_hw_info

    hw_cases = [
        ("mgn9_rail", "C", "MGN9C Linear Rail Carriage", 16.0),
        ("mgn9_rail", "H", "MGN9H Linear Rail Carriage", 26.0),
        ("mgn12_rail", "C", "MGN12C Linear Rail Carriage", 34.0),
        ("mgn12_rail", "H", "MGN12H Linear Rail Carriage", 54.0),
        ("mgn15_rail", "C", "MGN15C Linear Rail Carriage", 59.0),
        ("mgn15_rail", "H", "MGN15H Linear Rail Carriage", 92.0),
        ("hgh15_rail", "C", "HGH15CA Heavy Linear Carriage", 180.0),
        ("hgh15_rail", "H", "HGH15CA Heavy Linear Carriage", 180.0),
        ("steel_rod_8mm", "C", "LM8UU Linear Ball Bearing", 16.0),
        ("steel_rod_10mm", "C", "LM10UU Linear Ball Bearing", 31.0),
        ("steel_rod_12mm", "C", "LM12UU Linear Ball Bearing", 41.0),
    ]
    for rail_id, bt, exp_name, exp_unit_g in hw_cases:
        car_name, _, _, car_mass_g = _rail_hw_info(rail_id, block_type=bt)
        assert car_name == exp_name
        assert car_mass_g == pytest.approx(exp_unit_g, abs=1e-9)

    for rail_id, bt, exp_name, exp_unit_g in hw_cases[:6]:
        s_bom = _corexy_spec(x_rail_profile=rail_id, y_rail_profile=rail_id)
        s_bom.gantry.rail_block_type = bt  # type: ignore[assignment]
        bom = generate_bom(s_bom)
        x_car = next(i for i in bom.items if i.sku == "MOT-X-CAR")
        y_car = next(i for i in bom.items if i.sku == "MOT-Y-CAR")
        assert exp_name in x_car.name
        assert exp_name in y_car.name
        assert x_car.total_mass_g / x_car.quantity == pytest.approx(exp_unit_g, abs=1e-6)
        assert y_car.total_mass_g / y_car.quantity == pytest.approx(exp_unit_g, abs=1e-6)

    # Carbonara BOM includes GNT-CF-INSERTS line item
    bom_mono = generate_bom(s_mono_12)
    cf_ins = next(i for i in bom_mono.items if i.sku == "GNT-CF-INSERTS")
    assert cf_ins.category == "gantry"
    assert cf_ins.total_cost_usd == pytest.approx(18.0)
    assert cf_ins.total_mass_g == pytest.approx(
        a_mono_12.x_beam_insert_mass_kg * 1e3, abs=0.1
    )
