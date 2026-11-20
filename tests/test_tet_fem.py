"""Analytical and integration tests for conformal 3D tetrahedral FEA (Gmsh + scikit-fem + meshio)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from printsim.fea import (
    TetFEAResult,
    is_tet_fea_available,
    solve_preset_tet_fea,
    solve_step_bytes_tet_fea,
    solve_step_tet_fea,
)
from printsim.geometry.cad_b3d import export_part_step, export_part_step_bytes, is_cad_available
from printsim.schema import get_material

pytestmark = pytest.mark.skipif(
    not (is_tet_fea_available() and is_cad_available()),
    reason="Requires optional [cad] and [tetfea] extras (build123d, gmsh, scikit-fem, meshio).",
)


def _signed_surface_volume_mm3(surface_mesh: dict) -> float:
    """Compute signed enclosed volume via divergence theorem; >0 iff outward-oriented."""
    verts = np.asarray(surface_mesh["vertices_mm"], dtype=np.float64)
    tris = np.asarray(surface_mesh["triangles"], dtype=np.int64)
    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    return float(np.sum(np.einsum("ij,ij->i", v0, np.cross(v1, v2))) / 6.0)


def test_tet10_uniaxial_tension_analytical() -> None:
    """Tet10 uniaxial bar extension must match FL/(EA) within 3% and 10.0 MPa within 10%."""
    # cantilever_bar is 60 x 12 x 12 mm -> L=0.06 m, A=144e-6 m^2
    F_x = 1440.0  # N -> exact nominal sigma_xx = 10.0 MPa
    res = solve_preset_tet_fea(
        preset="cantilever_bar",
        material_id="aluminum_6061_t6",
        force_n=(F_x, 0.0, 0.0),
        order=2,
        fdm_anisotropic=False,
    )
    assert isinstance(res, TetFEAResult)
    assert res.backend == "gmsh+scikit-fem"
    assert res.element_type == "Tet10 (P2)"
    assert res.is_fdm_anisotropic is False

    mat = get_material("aluminum_6061_t6")
    L = 0.060
    A = 0.012 * 0.012
    exact_disp_um = (F_x * L / (mat.youngs_modulus_pa * A)) * 1e6

    assert res.volume_mm3 == pytest.approx(60.0 * 12.0 * 12.0, rel=1e-3)
    assert res.max_displacement_um == pytest.approx(exact_disp_um, rel=0.03)
    assert res.max_von_mises_mpa == pytest.approx(10.0, rel=0.10)


def test_tet10_euler_bernoulli_timoshenko_bending_analytical() -> None:
    """Tet10 cantilever tip deflection must match Euler-Bernoulli + Timoshenko shear within 5%."""
    F_z = -50.0
    res = solve_preset_tet_fea(
        preset="cantilever_bar",
        material_id="aluminum_6061_t6",
        force_n=(0.0, 0.0, F_z),
        order=2,
        fdm_anisotropic=False,
    )
    mat = get_material("aluminum_6061_t6")
    L = 0.060
    b = 0.012
    h = 0.012
    A = b * h
    I_y = (b * h**3) / 12.0
    E = mat.youngs_modulus_pa
    G = mat.effective_shear_modulus_pa
    kappa = 5.0 / 6.0  # Rectangular cross-section Timoshenko shear factor

    delta_eb_m = abs(F_z) * L**3 / (3.0 * E * I_y)
    delta_shear_m = abs(F_z) * L / (kappa * G * A)
    exact_tip_um = (delta_eb_m + delta_shear_m) * 1e6

    assert res.max_displacement_um == pytest.approx(exact_tip_um, rel=0.05)


@pytest.mark.parametrize("preset", ["corner_gusset", "xy_motor_mount"])
def test_tet10_curved_brackets_outward_surface_and_fdm_vs_cnc(preset: str) -> None:
    """Curved STEP brackets mesh cleanly, orient outward, and show CNC Al >15x stiffer than FDM ABS."""
    res_abs = solve_preset_tet_fea(
        preset=preset,  # type: ignore[arg-type]
        material_id="abs_printed",
        order=2,
    )
    res_al = solve_preset_tet_fea(
        preset=preset,  # type: ignore[arg-type]
        material_id="aluminum_6061_t6",
        order=2,
    )

    assert res_abs.is_fdm_anisotropic is True
    assert res_al.is_fdm_anisotropic is False
    assert res_al.stiffness_n_per_mm > res_abs.stiffness_n_per_mm * 15.0
    assert res_al.safety_factor > res_abs.safety_factor * 5.0

    surf = res_al.surface_mesh
    n_surf = len(surf["vertices_mm"])
    assert n_surf > 20
    assert len(surf["triangles"]) > 30
    assert len(surf["displacements_mm"]) == n_surf
    assert len(surf["von_mises_mpa"]) == n_surf
    assert len(surf["fixed_mask"]) == n_surf
    assert len(surf["loaded_mask"]) == n_surf
    assert any(surf["fixed_mask"])
    assert any(surf["loaded_mask"])

    # Positive signed volume matching solid volume proves all boundary triangles point outward
    signed_vol = _signed_surface_volume_mm3(surf)
    assert signed_vol == pytest.approx(res_al.volume_mm3, rel=1e-3)

    d_compact = res_al.to_dict(include_surface_mesh=False)
    assert "surface_mesh" not in d_compact
    d_full = res_al.to_dict(include_surface_mesh=True)
    assert "surface_mesh" in d_full


def test_tet_fea_vtu_export_and_p1_order(tmp_path: Path) -> None:
    """Verify VTU export via meshio.write and Tet4 (order=1) execution from STEP file."""
    import meshio

    step_file = tmp_path / "bar.step"
    export_part_step("cantilever_bar", step_file)
    vtu_file = tmp_path / "bar_solution.vtu"

    res = solve_step_tet_fea(
        step_path=step_file,
        material_id="pa12_cf_printed",
        fixed_face="min_x",
        load_face="max_x",
        force_n=(0.0, 0.0, -40.0),
        mesh_size_mm=6.0,
        order=1,
        export_vtu_path=vtu_file,
    )
    assert res.element_type == "Tet4 (P1)"
    assert res.is_fdm_anisotropic is True
    assert vtu_file.exists() and vtu_file.stat().st_size > 200

    mio = meshio.read(str(vtu_file))
    assert "tetra" in mio.cells_dict
    assert len(mio.cells_dict["tetra"]) == res.num_elements
    assert "displacement_mm" in mio.point_data
    assert "von_mises_mpa" in mio.point_data
    assert "von_mises_mpa" in mio.cell_data

    # Also smoke-test raw STEP bytes entrypoint
    step_bytes = export_part_step_bytes("cantilever_bar")
    res_bytes = solve_step_bytes_tet_fea(step_bytes=step_bytes, material_id="aluminum_6061_t6")
    assert res_bytes.num_elements > 50
