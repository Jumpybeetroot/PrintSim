"""Physics, analytical FEA, mass/cost accounting, and BOM tests for composite bolted linear rails."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.linalg import eigh, solve

from printsim.api.server import create_app
from printsim.fea import assemble_global_matrices, solve_modal_fea, solve_static_fea
from printsim.geometry import (
    AssemblyModel,
    StructuralMember,
    StructuralNode,
    build_assembly,
    generate_bom,
)
from printsim.geometry.builder import _apply_bolted_linear_rail
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

    # Check eta_rail = 0.45 (extrusion), 0.35 (cf_*), 0.55 (solid_bar) and dual symmetric flange action
    cases = [
        ("ext_front", ext2020, al_mat, 0.45, [np.array([0.0, -1.0, 0.0])]),
        ("cf_top", cf2020, cf_mat, 0.35, [np.array([0.0, 0.0, 1.0])]),
        ("solid_front", solid2020, al_mat, 0.55, [np.array([0.0, -1.0, 0.0])]),
        ("ext_dual", ext2020, al_mat, 0.45, [np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])]),
    ]
    for label, beam_prof, beam_mat, expected_eta, normals in cases:
        L = 0.48
        n_elems = 8
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
            _apply_bolted_linear_rail(m, nmap, mgn12, steel, normals, beam_shape=beam_prof.shape)

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

        # Cantilever tip deflection PL^3 / (3 E I_comp) in both Y and Z planes
        P = 15.0
        for axis, i_comp in ((1, m0.profile.iz_m4), (2, m0.profile.iy_m4)):
            F = np.zeros(K.shape[0])
            F[6 * tip + axis] = P
            u = np.zeros(K.shape[0])
            u[free_dofs] = solve(K[np.ix_(free_dofs, free_dofs)], F[free_dofs])
            exact_defl = P * (L**3) / (3.0 * beam_mat.youngs_modulus_pa * i_comp)
            assert abs(u[6 * tip + axis]) == pytest.approx(exact_defl, rel=1e-4)

        # Cantilever consistent-mass modal fundamental frequency with distributed rail mass
        eigvals, _ = eigh(K[np.ix_(free_dofs, free_dofs)], M[np.ix_(free_dofs, free_dofs)], subset_by_index=[0, 0])
        f1_fea = math.sqrt(float(eigvals[0])) / (2.0 * math.pi)
        mu_per_m = beam_prof.area_m2 * beam_mat.density_kg_m3 + len(normals) * mgn12.area_m2 * steel.density_kg_m3
        i_min = min(m0.profile.iy_m4, m0.profile.iz_m4)
        f1_exact = (1.87510407**2 / (2.0 * math.pi * L**2)) * math.sqrt(
            (beam_mat.youngs_modulus_pa * i_min) / mu_per_m
        )
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

    # dual_top_bottom gives highest X-beam Z-bending iy and lowest gravity sag
    assert xg_dual.profile.iy_m4 > xg_top.profile.iy_m4
    assert xg_dual.profile.iy_m4 > xg_front.profile.iy_m4
    assert st_dual.toolhead_stiffness_z_n_per_mm > st_top.toolhead_stiffness_z_n_per_mm
    assert st_dual.toolhead_stiffness_z_n_per_mm > st_front.toolhead_stiffness_z_n_per_mm
    assert st_dual.gravity_sag_toolhead_um < st_top.gravity_sag_toolhead_um

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

    # Exact +1 extra X rail mass on y_moving_mass_kg, frame_mass_kg, total_mass_kg, and total_rail_mass_kg
    x_rail_prof = get_profile(spec_front.gantry.x_rail_profile)
    rail_mat = get_material(spec_front.gantry.rail_material)
    w_m = spec_front.frame.width_mm * 1e-3
    one_x_rail_kg = x_rail_prof.area_m2 * w_m * rail_mat.density_kg_m3

    assert (asm_dual_mount.y_moving_mass_kg - asm_front.y_moving_mass_kg) == pytest.approx(one_x_rail_kg, rel=1e-9)
    assert (asm_dual_mount.frame_mass_kg - asm_front.frame_mass_kg) == pytest.approx(one_x_rail_kg, rel=1e-9)
    assert (asm_dual_mount.total_mass_kg - asm_front.total_mass_kg) == pytest.approx(one_x_rail_kg, rel=1e-9)
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

    w_m = s_cf_none_front.frame.width_mm * 1e-3
    d_m = s_cf_none_front.frame.depth_mm * 1e-3
    cf_prof = get_profile("cf_tube_20x20x2")
    cf_mat = get_material("carbon_fiber_tube")
    steel_mat = get_material("steel_4140")
    x_rail_prof = get_profile("mgn12_rail")
    y_rail_prof = get_profile("mgn12_rail")

    bare_cf_kg = cf_prof.area_m2 * w_m * cf_mat.density_kg_m3
    one_x_rail_kg = x_rail_prof.area_m2 * w_m * steel_mat.density_kg_m3
    two_y_rails_kg = 2.0 * y_rail_prof.area_m2 * d_m * steel_mat.density_kg_m3

    # 1. Exact total_rail_mass_kg across X and Y rails
    assert asm_f.total_rail_mass_kg == pytest.approx(one_x_rail_kg + two_y_rails_kg, rel=1e-9)
    assert asm_d.total_rail_mass_kg == pytest.approx(2.0 * one_x_rail_kg + two_y_rails_kg, rel=1e-9)

    # 2. Exact y_moving_mass_kg (NO double-counting of x_rail_mass!)
    expected_y_mov_f = (
        s_cf_epoxy_front.toolhead.total_mass_g * 1e-3
        + bare_cf_kg
        + one_x_rail_kg
        + s_cf_epoxy_front.gantry.carriage_mass_g * 1e-3
    )
    expected_y_mov_d = expected_y_mov_f + one_x_rail_kg
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
        "x_rail_mount": "bottom",
        "y_rail_mount": "inner",
        "dual_x_rails": False,
        "x_rail_width_mm": 12.0,
        "x_rail_height_mm": 8.0,
        "y_rail_width_mm": 9.0,
        "y_rail_height_mm": 6.5,
        "x_beam_ei_y_boost_pct": round(asm.x_rail_ei_y_boost_pct, 1),
        "x_beam_ei_z_boost_pct": round(asm.x_rail_ei_z_boost_pct, 1),
        "y_rail_ei_x_boost_pct": round(asm.y_rail_ei_x_boost_pct, 1),
        "total_rail_mass_g": round(asm.total_rail_mass_kg * 1e3, 1),
    }
    assert lr["x_beam_ei_z_boost_pct"] > lr["x_beam_ei_y_boost_pct"]
    assert lr["y_rail_ei_x_boost_pct"] > 90.0
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
