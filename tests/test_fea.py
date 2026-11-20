"""Analytical benchmark and physical sanity tests for the 3D Euler-Bernoulli FEA solver."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.linalg import eigh, solve

from printsim.fea import assemble_global_matrices, solve_modal_fea, solve_static_fea
from printsim.geometry import AssemblyModel, StructuralMember, StructuralNode, build_assembly
from printsim.schema import PrinterSpec, get_material, get_profile


def _cantilever_assembly(n_elems: int = 8, length_m: float = 0.5) -> AssemblyModel:
    mat = get_material("aluminum_6063_t5")
    prof = get_profile("extrusion_2040")  # Asymmetric Iy != Iz to test both bending planes!
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


def test_cantilever_exact_static_deflection_both_planes() -> None:
    """Verify FEA tip deflection matches exact analytical PL^3 / (3 E I) in both Y and Z."""
    L = 0.5
    P = 25.0  # Newtons
    asm = _cantilever_assembly(n_elems=6, length_m=L)
    mat = get_material("aluminum_6063_t5")
    prof = get_profile("extrusion_2040")

    K, _, _, node_idx, free_dofs = assemble_global_matrices(asm)
    tip = node_idx["n6"]

    # Local Z = Global Z (uses Iy)
    F_z = np.zeros(K.shape[0])
    F_z[6 * tip + 2] = -P
    u_z = np.zeros(K.shape[0])
    u_z[free_dofs] = solve(K[np.ix_(free_dofs, free_dofs)], F_z[free_dofs])
    exact_z = P * (L**3) / (3.0 * mat.youngs_modulus_pa * prof.iy_m4)
    assert abs(u_z[6 * tip + 2]) == pytest.approx(exact_z, rel=1e-4)

    # Local Y = Global Y (uses Iz)
    F_y = np.zeros(K.shape[0])
    F_y[6 * tip + 1] = P
    u_y = np.zeros(K.shape[0])
    u_y[free_dofs] = solve(K[np.ix_(free_dofs, free_dofs)], F_y[free_dofs])
    exact_y = P * (L**3) / (3.0 * mat.youngs_modulus_pa * prof.iz_m4)
    assert abs(u_y[6 * tip + 1]) == pytest.approx(exact_y, rel=1e-4)


def test_cantilever_exact_fundamental_frequency() -> None:
    """Verify consistent-mass modal FEA matches Euler-Bernoulli cantilever f1 formula."""
    L = 0.5
    asm = _cantilever_assembly(n_elems=8, length_m=L)
    mat = get_material("aluminum_6063_t5")
    prof = get_profile("extrusion_2040")

    K, M, _, _, free_dofs = assemble_global_matrices(asm)
    eigvals, _ = eigh(K[np.ix_(free_dofs, free_dofs)], M[np.ix_(free_dofs, free_dofs)], subset_by_index=[0, 0])
    f1_fea = math.sqrt(float(eigvals[0])) / (2.0 * math.pi)

    # Lowest bending mode uses min(Iy, Iz)
    i_min = min(prof.iy_m4, prof.iz_m4)
    beta1_l = 1.87510407
    f1_exact = (beta1_l**2 / (2.0 * math.pi * L**2)) * math.sqrt(
        (mat.youngs_modulus_pa * i_min) / (mat.density_kg_m3 * prof.area_m2)
    )
    assert f1_fea == pytest.approx(f1_exact, rel=0.005)


def test_bracing_increases_corexy_stiffness_and_resonance() -> None:
    """Adding full_truss bracing must increase natural frequency and reduce deflection."""
    spec_bare = PrinterSpec(architecture="corexy")
    spec_bare.frame.bracing = "none"
    spec_bare.frame.enclosed_panels = False

    spec_braced = spec_bare.model_copy(deep=True)
    spec_braced.frame.bracing = "full_truss"

    asm_bare = build_assembly(spec_bare)
    asm_braced = build_assembly(spec_braced)

    fea_bare = solve_static_fea(asm_bare)
    fea_braced = solve_static_fea(asm_braced)
    mod_bare = solve_modal_fea(asm_bare)
    mod_braced = solve_modal_fea(asm_braced)

    assert mod_braced.fundamental_freq_hz > mod_bare.fundamental_freq_hz * 1.15
    assert fea_braced.combined_max_deflection_um < fea_bare.combined_max_deflection_um
