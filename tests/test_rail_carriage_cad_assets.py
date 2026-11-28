"""Verification tests for baked HIWIN solid linear rail & linear carriage 3D CAD assets."""

from __future__ import annotations

import numpy as np
import pytest

from printsim.cad_assets import (
    HARDWARE_CATEGORIES,
    get_hardware_asset_manifest,
    get_visual_glb_path,
    has_hardware_asset,
    has_visual_asset,
    list_hardware_assets,
    load_hardware_asset,
    load_superelement,
    verify_baked_asset,
)

# Expected HIWIN solid rail dimensions: (length_mm, width_mm, height_mm)
SOLID_RAIL_SPECS: dict[str, tuple[float, float, float]] = {
    "mgn9_rail": (360.0, 9.0, 6.5),
    "mgn12_rail": (375.0, 12.0, 8.0),
    "mgn15_rail": (360.0, 15.0, 10.0),
    "hgh15_rail": (360.0, 15.0, 15.0),
}

# Expected HIWIN carriage dimensions & datasheet masses:
# (length_mm, width_mm, block_height_mm, h_bolt_mm, target_mass_g)
CARRIAGE_SPECS: dict[str, tuple[float, float, float, float, float]] = {
    "mgn9c": (28.9, 20.0, 8.0, 6.1, 16.0),
    "mgn9h": (39.9, 20.0, 8.0, 6.1, 26.0),
    "mgn12c": (34.7, 27.0, 10.0, 8.2, 34.0),
    "mgn12h": (45.4, 27.0, 10.0, 8.2, 54.0),
    "mgn15c": (42.1, 32.0, 12.0, 10.0, 59.0),
    "mgn15h": (58.8, 32.0, 12.0, 10.0, 92.0),
    "hgh15ca": (61.4, 34.0, 19.7, 15.0, 180.0),
}


def test_carriages_category_registered() -> None:
    """'carriages' is registered in HARDWARE_CATEGORIES and discovered by list_hardware_assets()."""
    assert "carriages" in HARDWARE_CATEGORIES
    catalog = list_hardware_assets()
    assert "carriages" in catalog
    assert "x_beams" in catalog
    for cid in CARRIAGE_SPECS:
        assert cid in catalog["carriages"]
        assert catalog["carriages"][cid]["has_superelement"] is True
        assert catalog["carriages"][cid]["visual_url"] == f"/api/assets/hardware/carriages/{cid}/visual.glb"
    for rid in SOLID_RAIL_SPECS:
        assert rid in catalog["x_beams"]
        assert catalog["x_beams"][rid]["visual_url"] == f"/api/assets/hardware/x_beams/{rid}/visual.glb"


@pytest.mark.parametrize("rail_id", list(SOLID_RAIL_SPECS.keys()))
def test_solid_rail_cad_asset_loads_and_matches_bounds(rail_id: str) -> None:
    """Every solid profiled rail has a valid GLB, manifest, and exact HIWIN bounding box."""
    exp_len, exp_w, exp_h = SOLID_RAIL_SPECS[rail_id]
    assert has_visual_asset("x_beams", rail_id)
    glb_path = get_visual_glb_path("x_beams", rail_id)
    assert glb_path is not None and glb_path.is_file()
    assert glb_path.read_bytes()[:4] == b"glTF"

    man = get_hardware_asset_manifest("x_beams", rail_id)
    assert man is not None
    assert man["category"] == "x_beams"
    assert man["asset_id"] == rail_id
    assert man["has_superelement"] is False
    assert float(man["length_mm"]) == pytest.approx(exp_len, abs=0.1)
    assert float(man["width_mm"]) == pytest.approx(exp_w, abs=0.1)
    assert float(man["height_mm"]) == pytest.approx(exp_h, abs=0.1)
    assert int(man["visual_triangles"]) > 1000

    trimesh = pytest.importorskip("trimesh")
    scene = trimesh.load(glb_path)
    ext_mm = (scene.bounds[1] - scene.bounds[0]) * 1000.0
    ctr_mm = 0.5 * (scene.bounds[1] + scene.bounds[0]) * 1000.0
    assert ext_mm[0] == pytest.approx(exp_len, abs=0.2)
    assert ext_mm[1] == pytest.approx(exp_w, abs=0.15)
    assert ext_mm[2] == pytest.approx(exp_h, abs=0.15)
    # Centered at origin (matching mgn12_pocketed_rail convention)
    assert np.allclose(ctr_mm, [0.0, 0.0, 0.0], atol=0.15)


@pytest.mark.parametrize("carriage_id", list(CARRIAGE_SPECS.keys()))
def test_carriage_cad_asset_qa_superelement_and_bounds(carriage_id: str) -> None:
    """Every linear carriage passes 11/11 QA with rb_max < 0.01 Hz, HIWIN mass, and exact bounds."""
    exp_len, exp_w, exp_h, exp_h_bolt, exp_mass_g = CARRIAGE_SPECS[carriage_id]

    assert has_hardware_asset("carriages", carriage_id)
    assert has_visual_asset("carriages", carriage_id)

    qa = verify_baked_asset("carriages", carriage_id)
    assert qa.passed, qa.summary
    assert len(qa.checks) == 11
    assert all(c["passed"] for c in qa.checks)

    asset = load_hardware_asset("carriages", carriage_id)
    assert asset is not None
    asset_alias = load_superelement("carriages", carriage_id)
    assert asset_alias is not None
    assert asset.asset_id == carriage_id
    assert asset.category == "carriages"

    # Rigid-body zero-mode < 0.01 Hz
    rb_hz = float(asset.manifest["rigid_body_max_freq_hz"])
    assert rb_hz < 0.01, f"{carriage_id} rb_max={rb_hz:.6f} Hz >= 0.01 Hz"

    # Port 0 = [0, 0, 0] (ball-raceway center), Port 1 = [0, 0, +h_bolt] (4-bolt top face)
    assert np.allclose(asset.port_coords_m[0] * 1000.0, [0.0, 0.0, 0.0], atol=1e-4)
    assert np.allclose(asset.port_coords_m[1] * 1000.0, [0.0, 0.0, exp_h_bolt], atol=1e-3)

    # Steel saddle mass matches HIWIN datasheet within +-5%
    baked_mass_g = float(asset.manifest["mass_kg"]) * 1000.0
    assert baked_mass_g == pytest.approx(exp_mass_g, rel=0.05)

    # Multi-material PBR GLB dimensions match exact HIWIN carriage envelope
    trimesh = pytest.importorskip("trimesh")
    glb_path = get_visual_glb_path("carriages", carriage_id)
    assert glb_path is not None and glb_path.is_file()
    scene = trimesh.load(glb_path)
    ext_mm = (scene.bounds[1] - scene.bounds[0]) * 1000.0
    bmax_mm = scene.bounds[1] * 1000.0
    assert ext_mm[0] == pytest.approx(exp_len, abs=0.2)
    assert ext_mm[1] == pytest.approx(exp_w, abs=0.15)
    assert ext_mm[2] == pytest.approx(exp_h, abs=0.15)
    # Top mounting face sits at z = +h_bolt
    assert bmax_mm[2] == pytest.approx(exp_h_bolt, abs=0.15)
    # Multi-material visual scene (steel saddle + green end-caps + red wipers + brass grease port)
    if isinstance(scene, trimesh.Scene):
        assert len(scene.geometry) >= 4
