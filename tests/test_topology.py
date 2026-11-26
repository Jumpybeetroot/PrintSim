"""Analytical & engineering tests for 3D SIMP topology optimization and watertight STL export."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
import trimesh

from printsim.fea import (
    TopologyOptResult,
    export_topology_stl,
    optimize_preset_topology,
    optimize_solid_topology,
    topology_result_to_stl_bytes,
    voxel_grid_to_trimesh,
)
from printsim.fea.topology import _prune_disconnected_islands


def test_cantilever_bar_simp_convergence_and_specific_stiffness() -> None:
    """Cantilever bar SIMP decreases compliance from uniform start and improves specific stiffness."""
    # On a 4-voxel-tall cantilever bar, top & bottom outer skins are 50% of volume;
    # target_volume_fraction=0.55 lets SIMP carve the low-stress core while preserving bending flanges.
    res = optimize_preset_topology(
        preset="cantilever_bar",
        material_id="pa12_cf_printed",
        target_volume_fraction=0.55,
        iterations=20,
        penalization=3.0,
        filter_radius_voxels=1.5,
        preserve_boundaries=True,
    )

    assert isinstance(res, TopologyOptResult)
    assert res.iterations_run >= 5
    assert len(res.history) == res.iterations_run
    assert all("mass_g" in step and step["mass_g"] > 0.0 for step in res.history)

    # Compliance drops steeply during early SIMP iterations and finishes well below uniform start
    compliances = [step["compliance_mj"] for step in res.history]
    assert compliances[-1] < compliances[0] * 0.65
    for idx in range(1, min(5, len(compliances))):
        assert compliances[idx] < compliances[idx - 1]
    for idx in range(1, len(compliances)):
        assert compliances[idx] <= compliances[idx - 1] * 1.02

    # Final binary design dramatically beats uniform penalized density compliance
    assert res.optimized_compliance_mj < compliances[0]

    # Mass reduced near target volume fraction
    assert res.achieved_volume_fraction == pytest.approx(0.55, abs=0.08)
    expected_reduction = (1.0 - res.optimized_mass_g / res.baseline_mass_g) * 100.0
    assert res.mass_reduction_pct == pytest.approx(expected_reduction, rel=1e-4)
    assert res.optimized_elements < res.baseline_elements

    # Specific stiffness (stiffness / mass) improves over full solid baseline
    # (B-10 QM6-3D removes 1-voxel flange shear-locking inflation: gain ~2.0% vs 3.7% locked)
    opt_spec = res.optimized_stiffness_n_per_mm / res.optimized_mass_g
    base_spec = res.baseline_stiffness_n_per_mm / res.baseline_mass_g
    assert opt_spec > base_spec
    assert res.specific_stiffness_gain_pct > 1.5

    # Voxel payload and serialization invariants
    assert len(res.voxels) == res.baseline_elements
    assert sum(1 for v in res.voxels if v["kept"]) == res.optimized_elements
    payload_full = res.to_dict(include_voxels=True)
    assert "optimized_grid" not in payload_full
    assert len(payload_full["voxels"]) == res.baseline_elements
    payload_compact = res.to_dict(include_voxels=False)
    assert "voxels" not in payload_compact
    assert "optimized_grid" not in payload_compact


@pytest.mark.parametrize("preset", ["corner_gusset", "xy_motor_mount"])
def test_presets_corner_gusset_and_xy_motor_mount(preset: str) -> None:
    """Built-in printer brackets optimize in <3s, preserve mounting faces, and stay structurally valid."""
    t0 = time.perf_counter()
    res = optimize_preset_topology(
        preset=preset,  # type: ignore[arg-type]
        material_id="pa12_cf_printed",
        target_volume_fraction=0.45,
        iterations=18,
        penalization=3.0,
        filter_radius_voxels=1.5,
        preserve_boundaries=True,
    )
    elapsed_s = time.perf_counter() - t0

    import os

    assert elapsed_s < (15.0 if os.environ.get("CI") else 5.0)
    assert res.optimized_stiffness_n_per_mm > 0.0
    assert res.safety_factor > 0.0
    assert res.max_displacement_um > 0.0
    assert res.max_von_mises_mpa > 0.0
    assert res.failure_mode in {"safe", "isotropic_yield", "fdm_layer_delamination"}
    assert res.optimized_mass_g < res.baseline_mass_g

    # Mounting & loaded boundary elements must all remain kept
    boundary_voxels = [v for v in res.voxels if v["fixed"] or v["loaded"]]
    assert len(boundary_voxels) > 0
    assert all(v["kept"] for v in boundary_voxels)


def test_voxel_grid_to_trimesh_watertight_and_exact_volume() -> None:
    """Exposed boundary meshing produces a watertight, outward-wound mesh with exact voxel volume."""
    grid = np.zeros((10, 8, 5), dtype=bool)
    # L-shaped bracket with an internal through-hole (manifold 6-connected voxel solid)
    grid[:, :3, :] = True
    grid[:3, :, :] = True
    grid[5:7, :3, 1:3] = False

    h_mm = 2.5
    n_voxels = int(grid.sum())
    expected_volume_mm3 = n_voxels * (h_mm**3)

    mesh = voxel_grid_to_trimesh(grid, voxel_size_mm=h_mm, smooth_iterations=0)
    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    assert mesh.volume > 0.0
    assert mesh.volume == pytest.approx(expected_volume_mm3, rel=1e-6)

    # Taubin smoothing preserves mesh topology, watertightness, and positive volume
    smoothed = voxel_grid_to_trimesh(grid, voxel_size_mm=h_mm, smooth_iterations=4)
    assert smoothed.is_watertight
    assert smoothed.volume > 0.0
    assert len(smoothed.faces) == len(mesh.faces)


def test_export_topology_stl_and_bytes_roundtrip(tmp_path: Path) -> None:
    """STL bytes and file export produce valid reloadable STL meshes."""
    res = optimize_preset_topology(
        preset="cantilever_bar",
        material_id="pa12_cf_printed",
        target_volume_fraction=0.5,
        iterations=12,
    )

    stl_bytes = topology_result_to_stl_bytes(res, smooth_iterations=0)
    assert len(stl_bytes) > 84

    loaded_from_bytes = trimesh.load(
        trimesh.util.wrap_as_stream(stl_bytes),
        file_type="stl",
        force="mesh",
    )
    assert len(loaded_from_bytes.faces) > 0
    assert loaded_from_bytes.volume > 0.0
    expected_vol = res.optimized_elements * (res.voxel_size_mm**3)
    assert loaded_from_bytes.volume == pytest.approx(expected_vol, rel=1e-5)

    out_file = export_topology_stl(
        res,
        tmp_path / "exported" / "cantilever_opt.stl",
        smooth_iterations=2,
    )
    assert out_file.exists()
    assert out_file.stat().st_size > 84

    loaded_from_file = trimesh.load(out_file, force="mesh")
    assert len(loaded_from_file.faces) == len(loaded_from_bytes.faces)
    assert loaded_from_file.volume > 0.0


def test_island_pruning_and_load_path_bridging() -> None:
    """Floating islands are pruned while disconnected loaded regions are bridged along high-rho paths."""
    grid = np.ones((5, 2, 2), dtype=bool)
    occ = np.argwhere(grid)
    n_elem = len(occ)

    is_fixed = occ[:, 0] == 0
    is_loaded = occ[:, 0] == 4

    # Case 1: Floating island at x=3,4 with no load/fixed role is pruned
    kept_island = np.zeros(n_elem, dtype=bool)
    kept_island[occ[:, 0] <= 1] = True
    kept_island[occ[:, 0] >= 3] = True
    rho = np.full(n_elem, 0.2, dtype=float)
    pruned = _prune_disconnected_islands(
        grid_shape=grid.shape,
        occ=occ,
        rho=rho,
        kept_mask=kept_island,
        is_fixed_elem=is_fixed,
        is_loaded_elem=np.zeros(n_elem, dtype=bool),
        preserve_boundaries=True,
    )
    assert np.all(pruned[occ[:, 0] <= 1])
    assert not np.any(pruned[occ[:, 0] >= 3])

    # Case 2: Disconnected loaded face at x=4 is bridged through high-rho line (y=0, z=0)
    kept_bridge = is_fixed | is_loaded
    rho_bridge = np.full(n_elem, 0.1, dtype=float)
    rho_bridge[(occ[:, 1] == 0) & (occ[:, 2] == 0)] = 0.95
    bridged = _prune_disconnected_islands(
        grid_shape=grid.shape,
        occ=occ,
        rho=rho_bridge,
        kept_mask=kept_bridge,
        is_fixed_elem=is_fixed,
        is_loaded_elem=is_loaded,
        preserve_boundaries=True,
    )
    assert np.all(bridged[is_fixed | is_loaded])
    # All intermediate voxels along (y=0, z=0) should be bridged
    assert np.all(bridged[(occ[:, 1] == 0) & (occ[:, 2] == 0)])


def test_topology_input_validation() -> None:
    """Invalid grids, voxel sizes, or volume fractions raise explicit ValueErrors."""
    empty = np.zeros((4, 4, 4), dtype=bool)
    with pytest.raises(ValueError):
        voxel_grid_to_trimesh(empty, voxel_size_mm=2.0)
    with pytest.raises(ValueError):
        optimize_solid_topology(empty, voxel_size_mm=2.0)

    valid = np.ones((4, 3, 3), dtype=bool)
    with pytest.raises(ValueError):
        optimize_solid_topology(valid, voxel_size_mm=0.0)
    with pytest.raises(ValueError):
        optimize_solid_topology(valid, voxel_size_mm=2.0, target_volume_fraction=1.2)
