"""Analytical benchmarks, physical sanity checks, and integration tests for 2D MITC4 bed plate FEM."""

from __future__ import annotations

import math
from pathlib import Path
import time

import numpy as np
import pytest

from printsim.engine import evaluate_printer
from printsim.fea import BedPlateFEAResult, solve_bed_plate_fem
from printsim.fea.bed_plate_fem import _apply_mount_springs, _resolve_mount_layout
from printsim.fea.shell_mitc4 import mitc4_shell_matrices
from printsim.schema import PrinterSpec, get_material

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"

_REQUIRED_KEYS = {
    "grid_nx",
    "grid_ny",
    "grid_x_rel",
    "grid_y_rel",
    "mount_points_rel",
    "z_gravity_um",
    "z_thermal_um",
    "z_total_um",
    "mode1_shape_z",
    "gravity_sag_um",
    "thermal_peak_to_valley_um",
    "taco_buckling_um",
    "total_flatness_error_um",
    "probe_center_deflection_um",
    "probe_corner_deflection_um",
    "plate_first_mode_hz",
    "plate_first_mode_type",
    "plate_modes_hz",
    "mesh_bed_leveling_required",
    "first_layer_risk",
    "summary",
}


@pytest.mark.parametrize("mount_type", ["rigid_4pt", "kinematic_3pt"])
def test_kirchhoff_leissa_thickness_scaling(mount_type: str) -> None:
    """Doubling plate thickness (4 -> 8 mm) follows Kirchhoff D ~ t^3 and Leissa f1 scaling."""
    spec_4 = PrinterSpec(architecture="corexy")
    spec_4.bed.z_drive = "belted_quad_z"
    spec_4.bed.mount_type = mount_type  # type: ignore[assignment]
    spec_4.bed.plate_thickness_mm = 4.0
    spec_4.bed.mass_g = 2400.0

    spec_8 = spec_4.model_copy(deep=True)
    spec_8.bed.plate_thickness_mm = 8.0

    res_4 = solve_bed_plate_fem(spec_4)
    res_8 = solve_bed_plate_fem(spec_8)

    # Holding areal mass constant: D ~ t^3 => f1 ~ t^1.5 (~2.83x), center deflection drops ~6-8x
    freq_ratio = res_8.plate_first_mode_hz / res_4.plate_first_mode_hz
    defl_ratio = res_4.probe_center_deflection_um / res_8.probe_center_deflection_um
    assert 2.4 <= freq_ratio <= 3.0
    assert defl_ratio > 6.0
    assert res_8.gravity_sag_um < res_4.gravity_sag_um * 0.25

    # Scaling mass proportionally with thickness (constant density): f1 ~ t (~2.0x)
    spec_8_prop = spec_8.model_copy(deep=True)
    spec_8_prop.bed.mass_g = 4800.0
    res_8_prop = solve_bed_plate_fem(spec_8_prop)
    prop_freq_ratio = res_8_prop.plate_first_mode_hz / res_4.plate_first_mode_hz
    assert 1.7 <= prop_freq_ratio <= 2.15


def test_rigid_4pt_thermal_taco_vs_kinematic_3pt_flat() -> None:
    """At 110 C bed temp, rigid_4pt suffers severe thermal taco warp while kinematic_3pt has 0 taco."""
    spec_rig = PrinterSpec(architecture="corexy")
    spec_rig.bed.z_drive = "leadscrew_triple"
    spec_rig.bed.mount_type = "rigid_4pt"
    spec_rig.bed.target_temp_c = 110.0

    spec_kin = spec_rig.model_copy(deep=True)
    spec_kin.bed.mount_type = "kinematic_3pt"

    res_rig = solve_bed_plate_fem(spec_rig)
    res_kin = solve_bed_plate_fem(spec_kin)

    assert res_kin.taco_buckling_um == 0.0
    assert res_rig.taco_buckling_um > 100.0
    assert res_rig.thermal_peak_to_valley_um > 100.0
    assert res_rig.total_flatness_error_um > res_kin.total_flatness_error_um * 5.0
    assert res_rig.mesh_bed_leveling_required is True
    assert res_rig.first_layer_risk in ("high", "critical")
    assert res_kin.first_layer_risk == "low"


@pytest.mark.parametrize("z_drive", ["belted_quad_z", "leadscrew_triple"])
def test_mount_bounce_hierarchy(z_drive: str) -> None:
    """springs_4pt < silicone_4pt < kinematic_3pt in plate natural frequency with expected mode types."""
    base = PrinterSpec(architecture="corexy")
    base.bed.z_drive = z_drive  # type: ignore[assignment]

    s_spr = base.model_copy(deep=True)
    s_spr.bed.mount_type = "springs_4pt"

    s_sil = base.model_copy(deep=True)
    s_sil.bed.mount_type = "silicone_4pt"

    s_kin = base.model_copy(deep=True)
    s_kin.bed.mount_type = "kinematic_3pt"

    r_spr = solve_bed_plate_fem(s_spr)
    r_sil = solve_bed_plate_fem(s_sil)
    r_kin = solve_bed_plate_fem(s_kin)

    assert r_spr.plate_first_mode_type == "mount_bounce"
    assert r_sil.plate_first_mode_type == "mount_bounce"
    assert r_kin.plate_first_mode_type == "elastic_drum"

    assert r_spr.plate_first_mode_hz < r_sil.plate_first_mode_hz < r_kin.plate_first_mode_hz
    assert r_spr.gravity_sag_um > r_sil.gravity_sag_um > r_kin.gravity_sag_um
    assert r_spr.probe_center_deflection_um > r_sil.probe_center_deflection_um > r_kin.probe_center_deflection_um


def test_z_drive_cantilever_vs_dual_vs_triple_vs_quad_z() -> None:
    """Single cantilever Z-drive shows cantilever_flap mode and far worse droop/corner flex."""
    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    results: dict[str, BedPlateFEAResult] = {}
    for zd in ("leadscrew_single", "leadscrew_dual", "leadscrew_triple", "belted_quad_z"):
        spec = base.model_copy(deep=True)
        spec.bed.z_drive = zd  # type: ignore[assignment]
        results[zd] = solve_bed_plate_fem(spec)

    r1 = results["leadscrew_single"]
    r2 = results["leadscrew_dual"]
    r3 = results["leadscrew_triple"]
    r4 = results["belted_quad_z"]

    assert r1.plate_first_mode_type == "cantilever_flap"
    assert r3.plate_first_mode_type == "elastic_drum"
    assert r4.plate_first_mode_type == "elastic_drum"

    # Rear-edge anchor band for single cantilever vs 3-point triangle for kinematic Trident
    assert len(r1.mount_points_rel) == 3
    assert all(pt[1] == pytest.approx(0.5) for pt in r1.mount_points_rel)
    assert len(r2.mount_points_rel) == 2
    assert all(pt[1] == pytest.approx(0.0) for pt in r2.mount_points_rel)

    assert r1.gravity_sag_um > r2.gravity_sag_um > r3.gravity_sag_um > r4.gravity_sag_um
    assert r1.gravity_sag_um > r3.gravity_sag_um * 5.0
    assert (
        r1.probe_corner_deflection_um
        > r2.probe_corner_deflection_um
        > r3.probe_corner_deflection_um
        > r4.probe_corner_deflection_um
    )
    assert r1.plate_first_mode_hz < r2.plate_first_mode_hz < r3.plate_first_mode_hz < r4.plate_first_mode_hz


@pytest.mark.parametrize(
    ("arch", "z_drive", "mount_type"),
    [
        ("corexy", "belted_quad_z", "kinematic_3pt"),
        ("corexy", "leadscrew_triple", "rigid_4pt"),
        ("corexy", "leadscrew_dual", "floating_4pt"),
        ("corexy", "leadscrew_single", "springs_4pt"),
        ("cartesian_bedslinger", "leadscrew_dual", "silicone_4pt"),
        ("delta", "leadscrew_triple", "kinematic_3pt"),
    ],
)
def test_assembled_matrices_spd_and_symmetric(arch: str, z_drive: str, mount_type: str) -> None:
    """Verify assembled MITC4 plate K and M matrices are strictly symmetric positive definite."""
    spec = PrinterSpec(architecture=arch)  # type: ignore[arg-type]
    spec.bed.z_drive = z_drive  # type: ignore[assignment]
    spec.bed.mount_type = mount_type  # type: ignore[assignment]

    nx = ny = 5
    lx = spec.bed.size_mm[0] * 1e-3
    ly = spec.bed.size_mm[1] * 1e-3
    t = spec.bed.plate_thickness_mm * 1e-3
    mat = get_material(spec.bed.plate_material)
    areal_mass = (spec.bed.mass_g * 1e-3) / (lx * ly)

    gx = np.linspace(-0.5, 0.5, nx)
    gy = np.linspace(-0.5, 0.5, ny)
    coords = np.array([[x * lx, y * ly, 0.0] for y in gy for x in gx], dtype=float)
    n_dof = 6 * nx * ny
    k_mat = np.zeros((n_dof, n_dof), dtype=float)
    m_mat = np.zeros((n_dof, n_dof), dtype=float)
    for ey in range(4):
        for ex in range(4):
            n0 = ey * nx + ex
            n1 = n0 + 1
            n2 = (ey + 1) * nx + ex + 1
            n3 = (ey + 1) * nx + ex
            ke, me, _ = mitc4_shell_matrices(
                coords[[n0, n1, n2, n3]],
                mat.youngs_modulus_pa,
                mat.poisson_ratio,
                t,
                areal_mass,
            )
            dofs = [6 * nid + d for nid in (n0, n1, n2, n3) for d in range(6)]
            k_mat[np.ix_(dofs, dofs)] += ke
            m_mat[np.ix_(dofs, dofs)] += me

    anc_ids, _, kz_c = _resolve_mount_layout(spec, gx, gy)
    _apply_mount_springs(k_mat, spec, anc_ids, kz_c, t)
    assert np.allclose(k_mat, k_mat.T, rtol=1e-12, atol=1e-8)
    assert np.allclose(m_mat, m_mat.T, rtol=1e-12, atol=1e-12)
    assert float(np.linalg.eigvalsh(0.5 * (k_mat + k_mat.T))[0]) > 0.0
    assert float(np.linalg.eigvalsh(0.5 * (m_mat + m_mat.T))[0]) > 0.0


def test_mesh_refinement_and_runtime() -> None:
    """5x5 mesh agrees closely with refined 7x7 mesh and runs well under 100 ms."""
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    t0 = time.perf_counter()
    coarse = solve_bed_plate_fem(spec, mesh_n=4)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    fine = solve_bed_plate_fem(spec, mesh_n=6)

    assert elapsed_ms < 100.0
    assert abs(coarse.plate_first_mode_hz - fine.plate_first_mode_hz) / fine.plate_first_mode_hz < 0.08
    assert abs(coarse.probe_center_deflection_um - fine.probe_center_deflection_um) / fine.probe_center_deflection_um < 0.12


def test_evaluator_integration_all_example_presets_and_diagnostics() -> None:
    """evaluate_printer(spec).to_dict()['bed_plate_fea'] matches exact schema across all YAML presets."""
    yamls = sorted(EXAMPLES_DIR.glob("*.yaml"))
    assert len(yamls) >= 4

    for yaml_path in yamls:
        spec = PrinterSpec.from_yaml(yaml_path)
        rep = evaluate_printer(spec)
        assert rep.bed_plate_fea is not None
        d = rep.to_dict(include_toolpath_segments=False)["bed_plate_fea"]
        assert d is not None
        assert set(d.keys()) == _REQUIRED_KEYS

        assert d["grid_nx"] == 5
        assert d["grid_ny"] == 5
        assert len(d["grid_x_rel"]) == 5
        assert len(d["grid_y_rel"]) == 5
        assert len(d["mount_points_rel"]) >= 2

        for grid_field in ("z_gravity_um", "z_thermal_um", "z_total_um", "mode1_shape_z"):
            assert len(d[grid_field]) == 25
            assert all(math.isfinite(float(v)) for v in d[grid_field])

        assert max(abs(v) for v in d["mode1_shape_z"]) == pytest.approx(1.0, abs=1e-4)
        assert len(d["plate_modes_hz"]) == 3
        assert d["plate_modes_hz"] == sorted(d["plate_modes_hz"])
        assert d["plate_first_mode_hz"] == d["plate_modes_hz"][0]
        assert d["plate_first_mode_type"] in ("elastic_drum", "mount_bounce", "cantilever_flap")
        assert d["first_layer_risk"] in ("low", "moderate", "high", "critical")
        assert isinstance(d["mesh_bed_leveling_required"], bool)
        assert len(d["summary"]) > 20
        for scalar_key in (
            "gravity_sag_um",
            "thermal_peak_to_valley_um",
            "taco_buckling_um",
            "total_flatness_error_um",
            "probe_center_deflection_um",
            "probe_corner_deflection_um",
            "plate_first_mode_hz",
        ):
            assert math.isfinite(float(d[scalar_key]))

    # include_advanced_analysis=False omits bed_plate_fea cleanly
    spec_fast = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    rep_fast = evaluate_printer(spec_fast, include_advanced_analysis=False)
    assert rep_fast.bed_plate_fea is None
    assert rep_fast.to_dict(include_toolpath_segments=False)["bed_plate_fea"] is None

    # Diagnostic checks: high flatness error & low bed mode frequency emit clear diagnostics
    spec_bad = spec_fast.model_copy(deep=True)
    spec_bad.bed.z_drive = "leadscrew_single"
    rep_bad = evaluate_printer(spec_bad)
    joined = " ".join(rep_bad.diagnostics).lower()
    assert "flatness error" in joined
    assert "bed plate natural frequency" in joined
