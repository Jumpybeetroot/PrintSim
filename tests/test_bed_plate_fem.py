"""Analytical benchmarks, physical sanity checks, and integration tests for 2D MITC4 bed plate FEM."""

from __future__ import annotations

import math
from pathlib import Path
import time

import numpy as np
import pytest

from printsim.engine import evaluate_printer
from printsim.fea import (
    BedPlateFEAResult,
    compute_thermal_structural_report,
    solve_bed_plate_fem,
)
from printsim.fea.bed_plate_fem import _apply_mount_springs, _resolve_mount_layout
from printsim.fea.shell_mitc4 import mitc4_shell_matrices
from printsim.geometry import build_assembly
from printsim.schema import BED_MOUNT_CATALOG, PrinterSpec, get_material

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
    """Default 9x9 mesh lands exactly on +-0.375 inset anchors, converges vs 17x17, and runs < 15 ms."""
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    solve_bed_plate_fem(spec)  # warm up BLAS/NumPy paths
    t0 = time.perf_counter()
    default_9x9 = solve_bed_plate_fem(spec)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    fine_17x17 = solve_bed_plate_fem(spec, mesh_n=16)

    assert default_9x9.grid_nx == 9 and default_9x9.grid_ny == 9
    assert default_9x9.mount_points_rel == [[-0.375, -0.375], [0.375, -0.375], [0.0, 0.375]]
    assert fine_17x17.mount_points_rel == default_9x9.mount_points_rel
    assert elapsed_ms < 15.0
    assert abs(default_9x9.plate_first_mode_hz - fine_17x17.plate_first_mode_hz) / fine_17x17.plate_first_mode_hz < 0.03
    assert abs(default_9x9.probe_center_deflection_um - fine_17x17.probe_center_deflection_um) / fine_17x17.probe_center_deflection_um < 0.05


@pytest.mark.parametrize(
    "mount_type",
    ["kinematic_3pt", "whopping_pochard_kinematic", "trident_3pt_standoffs"],
)
def test_3pt_inset_cantilevered_rear_corners_vs_supported_front_corners(mount_type: str) -> None:
    """3-point inset mounts give >3x higher probe deflection on cantilevered rear corners than supported front corners."""
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    spec.bed.mount_type = mount_type  # type: ignore[assignment]

    res = solve_bed_plate_fem(spec)
    assert res.mount_points_rel == [[-0.375, -0.375], [0.375, -0.375], [0.0, 0.375]]

    # Assemble 9x9 stiffness matrix directly to compare front vs rear corner 2N probe compliance
    nx = ny = 9
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
    ke_ref, _, _ = mitc4_shell_matrices(
        coords[[0, 1, nx + 1, nx]], mat.youngs_modulus_pa, mat.poisson_ratio, t, areal_mass
    )
    for ey in range(8):
        for ex in range(8):
            n0 = ey * nx + ex
            dofs = [6 * nid + d for nid in (n0, n0 + 1, n0 + nx + 1, n0 + nx) for d in range(6)]
            k_mat[np.ix_(dofs, dofs)] += ke_ref

    anc_ids, mpts, kz_c = _resolve_mount_layout(spec, gx, gy)
    _apply_mount_springs(k_mat, spec, anc_ids, kz_c, t, mount_points_rel=mpts, lx_m=lx, ly_m=ly)

    # Probe front-left (0), front-right (nx - 1), rear-left ((ny - 1) * nx), rear-right (ny * nx - 1)
    corners = [0, nx - 1, (ny - 1) * nx, ny * nx - 1]
    rhs = np.zeros((n_dof, len(corners)), dtype=float)
    for col, cid in enumerate(corners):
        rhs[6 * cid + 2, col] = -2.0
    sol = np.linalg.solve(0.5 * (k_mat + k_mat.T), rhs)
    fl_um, fr_um, rl_um, rr_um = [abs(float(sol[6 * cid + 2, col])) * 1e6 for col, cid in enumerate(corners)]

    assert fl_um == pytest.approx(fr_um, rel=1e-6)
    assert rl_um == pytest.approx(rr_um, rel=1e-6)
    assert rl_um > fl_um * 3.0
    assert rr_um > fr_um * 3.0
    assert res.probe_corner_deflection_um == pytest.approx(max(fl_um, fr_um, rl_um, rr_um), rel=1e-5)


def test_directional_inplane_mount_springs_rank_eigenvalues_and_silicone_temp_softening() -> None:
    """Verify exact rotated 2x2 K_xy = k_r r r^T + k_t t t^T eigenvalues/vectors and 70A silicone softening."""
    nx = ny = 9
    gx = np.linspace(-0.5, 0.5, nx)
    gy = np.linspace(-0.5, 0.5, ny)
    n_dof = 6 * nx * ny

    def _extract_mount_blocks(mount_type: str, target_temp_c: float = 60.0) -> tuple[list[np.ndarray], list[list[float]], list[int], np.ndarray]:
        spec = PrinterSpec(architecture="corexy")
        spec.bed.z_drive = "leadscrew_triple"
        spec.bed.mount_type = mount_type  # type: ignore[assignment]
        spec.bed.target_temp_c = target_temp_c
        lx = spec.bed.size_mm[0] * 1e-3
        ly = spec.bed.size_mm[1] * 1e-3
        anc_ids, mpts, kz_c = _resolve_mount_layout(spec, gx, gy)
        k_spring_only = np.zeros((n_dof, n_dof), dtype=float)
        _apply_mount_springs(
            k_spring_only,
            spec,
            anc_ids,
            kz_c,
            spec.bed.plate_thickness_mm * 1e-3,
            mount_points_rel=mpts,
            lx_m=lx,
            ly_m=ly,
        )
        blocks = [k_spring_only[6 * nid : 6 * nid + 2, 6 * nid : 6 * nid + 2] for nid in anc_ids]
        return blocks, mpts, anc_ids, k_spring_only

    # 1. Maxwell V-groove (kinematic_3pt): k_r = 0.5 N/mm, k_t = 24,000 N/mm, radial eigenvector || r_hat
    blocks_kin, mpts_kin, _, _ = _extract_mount_blocks("kinematic_3pt")
    for k_xy, (rx, ry) in zip(blocks_kin, mpts_kin):
        assert np.allclose(k_xy, k_xy.T, atol=1e-9)
        evals, evecs = np.linalg.eigh(k_xy)
        assert evals[0] == pytest.approx(500.0, rel=1e-9)
        assert evals[1] == pytest.approx(2.4e7, rel=1e-9)
        r_hat = np.array([rx, ry], dtype=float) / math.hypot(rx, ry)
        assert abs(float(np.dot(evecs[:, 0], r_hat))) == pytest.approx(1.0, abs=1e-9)

    # 2. Whopping Pochard flexure: k_r = 5.0 N/mm, k_t = 16,000 N/mm
    blocks_wp, _, _, _ = _extract_mount_blocks("whopping_pochard_kinematic")
    for k_xy in blocks_wp:
        evals = np.linalg.eigvalsh(k_xy)
        assert evals[0] == pytest.approx(5000.0, rel=1e-9)
        assert evals[1] == pytest.approx(1.6e7, rel=1e-9)

    # 3. Floating 4-pt: anchor #0 isotropic (6000 N/mm), anchors #1-3 slotted radial (0.08 * 6000)
    blocks_flt, _, _, _ = _extract_mount_blocks("floating_4pt")
    assert np.linalg.eigvalsh(blocks_flt[0]) == pytest.approx([6.0e6, 6.0e6], rel=1e-9)
    for k_xy in blocks_flt[1:]:
        assert np.linalg.eigvalsh(k_xy) == pytest.approx([0.08 * 6.0e6, 6.0e6], rel=1e-9)

    # 4. Trident 3-pt Bakelite: front anchors slotted (0.15 * 9500), rear-center isotropic (9500)
    blocks_tri, _, _, _ = _extract_mount_blocks("trident_3pt_standoffs")
    for k_xy in blocks_tri[:2]:
        assert np.linalg.eigvalsh(k_xy) == pytest.approx([0.15 * 9.5e6, 9.5e6], rel=1e-9)
    assert np.linalg.eigvalsh(blocks_tri[2]) == pytest.approx([9.5e6, 9.5e6], rel=1e-9)

    # 5. Rigid 4-pt: isotropic 40,000 N/mm + rotational bolt-flange clamping moment stiffness
    blocks_rig, _, anc_rig, k_rig_only = _extract_mount_blocks("rigid_4pt")
    for k_xy, nid in zip(blocks_rig, anc_rig):
        assert np.linalg.eigvalsh(k_xy) == pytest.approx([4.0e7, 4.0e7], rel=1e-9)
        assert k_rig_only[6 * nid + 3, 6 * nid + 3] > 10.0
        assert k_rig_only[6 * nid + 4, 6 * nid + 4] > 10.0

    # 6. Silicone 4-pt softens above 70 C (60 C -> 110 C drops stiffness by 16%)
    s_60 = PrinterSpec(architecture="corexy")
    s_60.bed.mount_type = "silicone_4pt"
    s_60.bed.target_temp_c = 60.0
    s_110 = s_60.model_copy(deep=True)
    s_110.bed.target_temp_c = 110.0
    r_60 = solve_bed_plate_fem(s_60)
    r_110 = solve_bed_plate_fem(s_110)
    assert r_110.probe_center_deflection_um > r_60.probe_center_deflection_um
    assert r_110.plate_first_mode_hz < r_60.plate_first_mode_hz


def test_z_drive_mechanism_coupling_and_belt_sync_layout() -> None:
    """Upgrading tr8x8_pom -> tr8x4_pom -> tr8x8_brass -> sfu1204_bk10 stiffens bed plate FEM; belt-sync supported."""
    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    screws = ("tr8x8_pom", "tr8x4_pom", "tr8x8_brass", "sfu1204_bk10")
    res = []
    for ls in screws:
        s = base.model_copy(deep=True)
        s.bed.z_leadscrew_type = ls  # type: ignore[assignment]
        res.append(solve_bed_plate_fem(s))

    for i in range(len(screws) - 1):
        assert res[i + 1].probe_center_deflection_um < res[i].probe_center_deflection_um
        assert res[i + 1].probe_corner_deflection_um < res[i].probe_corner_deflection_um
        assert res[i + 1].gravity_sag_um < res[i].gravity_sag_um
        assert res[i + 1].plate_first_mode_hz > res[i].plate_first_mode_hz

    # VzBot 330 AWD (leadscrew_dual_belt_sync: 4x smooth rods + 2x synced leadscrews)
    vz = PrinterSpec.from_yaml(EXAMPLES_DIR / "vzbot_330_awd.yaml")
    res_vz = solve_bed_plate_fem(vz)
    assert res_vz.mount_points_rel == [[-0.375, -0.375], [0.375, -0.375], [0.0, 0.375]]
    assert res_vz.plate_first_mode_type == "elastic_drum"
    assert res_vz.plate_first_mode_hz > 80.0


def test_thermal_warp_ordering_at_100c_and_components_parity() -> None:
    """At 100 C, kinematic == whopping < floating < trident_3pt < rigid_4pt and matches components.py."""
    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    base.bed.target_temp_c = 100.0

    tacos: dict[str, float] = {}
    ptvs: dict[str, float] = {}
    for mt in BED_MOUNT_CATALOG:
        s = base.model_copy(deep=True)
        s.bed.mount_type = mt  # type: ignore[assignment]
        plate = solve_bed_plate_fem(s)
        ts = compute_thermal_structural_report(build_assembly(s))
        assert plate.taco_buckling_um == pytest.approx(ts.bed_buckling_taco_um, rel=1e-9, abs=1e-9)
        tacos[mt] = plate.taco_buckling_um
        ptvs[mt] = plate.thermal_peak_to_valley_um

    assert tacos["kinematic_3pt"] == pytest.approx(0.0, abs=1e-9)
    assert tacos["whopping_pochard_kinematic"] == pytest.approx(0.0, abs=1e-9)
    assert (
        tacos["kinematic_3pt"]
        == tacos["whopping_pochard_kinematic"]
        < tacos["floating_4pt"]
        < tacos["trident_3pt_standoffs"]
        < tacos["rigid_4pt"]
    )
    assert (
        ptvs["kinematic_3pt"]
        == pytest.approx(ptvs["whopping_pochard_kinematic"], rel=1e-9)
    )
    assert (
        ptvs["kinematic_3pt"]
        < ptvs["floating_4pt"]
        < ptvs["trident_3pt_standoffs"]
        < ptvs["rigid_4pt"]
    )


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

        assert d["grid_nx"] == 9
        assert d["grid_ny"] == 9
        assert len(d["grid_x_rel"]) == 9
        assert len(d["grid_y_rel"]) == 9
        assert len(d["mount_points_rel"]) >= 2

        for grid_field in ("z_gravity_um", "z_thermal_um", "z_total_um", "mode1_shape_z"):
            assert len(d[grid_field]) == 81
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
