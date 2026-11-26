"""Tests for Track 2 zero-LLM local CAD-to-superelement pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
import yaml

from printsim.cad_assets import (
    CadAssetRecipe,
    bake_from_recipe,
    fetch_onshape_step,
    inspect_step_file,
    parse_onshape_reference,
    render_contact_sheet,
    verify_baked_asset,
)
from printsim.fea import is_tet_fea_available
from printsim.geometry import is_cad_available


def _build_synthetic_bracket_step(out_step: Path) -> Path:
    """Create a tiny flange bracket with a Z-axis mounting hole in STEP format."""
    import build123d as b3d

    with b3d.BuildPart() as bp:
        # Flange block centered in XY, Z in [-6, 6] (+Z/-Z area > -Y area)
        b3d.Box(24.0, 22.0, 12.0)
        # Through hole along Z (diameter 6 mm)
        b3d.Cylinder(radius=3.0, height=16.0, mode=b3d.Mode.SUBTRACT)

    b3d.export_step(b3d.Compound(children=[bp.part.solid()]), str(out_step))
    return out_step


def _build_overlapping_assembly_step(out_step: Path) -> Path:
    """Create a 2-solid STEP assembly with an intentional volumetric overlap."""
    import build123d as b3d

    box_a = b3d.Solid.make_box(12.0, 12.0, 12.0)
    box_b = b3d.Solid.make_box(10.0, 10.0, 10.0).translate((6.0, 6.0, 6.0))
    b3d.export_step(b3d.Compound(children=[box_a, box_b]), str(out_step))
    return out_step


@pytest.mark.skipif(not is_cad_available(), reason="Requires build123d")
def test_inspect_step_file_extracts_geometry_and_markdown(tmp_path: Path) -> None:
    step_path = _build_synthetic_bracket_step(tmp_path / "bracket.step")
    report = inspect_step_file(step_path, default_material="aluminum_6061_t6")

    assert report.file_format == "step"
    assert len(report.solids) == 1
    solid = report.solids[0]
    assert solid.index == 0
    assert solid.min_mm == pytest.approx([-12.0, -11.0, -6.0], abs=1e-2)
    assert solid.max_mm == pytest.approx([12.0, 11.0, 6.0], abs=1e-2)
    assert solid.size_mm == pytest.approx([24.0, 22.0, 12.0], abs=1e-2)

    # Box volume (24*22*12 = 6336 mm3) minus cylinder (pi*3^2*12 = 339.292 mm3) = 5996.71 mm3
    expected_vol_cm3 = (24.0 * 22.0 * 12.0 - 3.141592653589793 * 9.0 * 12.0) / 1000.0
    assert solid.volume_cm3 == pytest.approx(expected_vol_cm3, rel=1e-3)
    assert solid.mass_estimates_g["aluminum_6061"] == pytest.approx(expected_vol_cm3 * 2.70, rel=1e-2)
    assert solid.mass_estimates_g["titanium_grade5"] == pytest.approx(expected_vol_cm3 * 4.43, rel=1e-2)
    assert solid.mass_estimates_g["steel"] == pytest.approx(expected_vol_cm3 * 7.85, rel=1e-2)
    assert solid.mass_estimates_g["abs_pc"] == pytest.approx(expected_vol_cm3 * 1.18, rel=1e-2)

    # Planar faces (+Z/-Z, +Y/-Y, +X/-X) and Ø6 cylindrical hole
    assert len(report.planar_faces) == 6
    axes = {pf.axis for pf in report.planar_faces}
    assert {"+Z", "-Z", "+Y", "-Y", "+X", "-X"}.issubset(axes)

    assert len(report.cylindrical_features) >= 1
    hole = report.cylindrical_features[0]
    assert hole.kind == "hole"
    assert hole.diameter_mm == pytest.approx(6.0, abs=1e-2)
    assert hole.axis == pytest.approx([0.0, 0.0, 1.0], abs=1e-2)

    assert "port0" in report.suggested_ports
    assert "port1" in report.suggested_ports
    assert report.suggested_ports["port0"]["vp_coord_mm"][2] == pytest.approx(6.0, abs=0.1)
    assert report.suggested_ports["port1"]["vp_coord_mm"][2] == pytest.approx(-6.0, abs=0.1)

    md = report.to_markdown()
    assert len(md.splitlines()) <= 50
    assert "bracket.step" in md
    d = report.to_dict()
    assert d["total_volume_cm3"] == pytest.approx(expected_vol_cm3, rel=1e-3)


@pytest.mark.skipif(not is_cad_available(), reason="Requires build123d")
def test_inspect_step_file_detects_overlapping_solids(tmp_path: Path) -> None:
    step_path = _build_overlapping_assembly_step(tmp_path / "overlap.step")
    report = inspect_step_file(step_path)

    assert len(report.solids) == 2
    assert len(report.overlapping_pairs) == 1
    ov = report.overlapping_pairs[0]
    assert ov["solid_a"] == 0
    assert ov["solid_b"] == 1
    # Overlap is [6..12]^3 = 6*6*6 = 216 mm3 = 0.216 cm3
    assert ov["intersection_volume_cm3"] == pytest.approx(0.216, rel=1e-2)


@pytest.mark.skipif(
    not (is_cad_available() and is_tet_fea_available()),
    reason="Requires build123d, gmsh, and scikit-fem",
)
def test_bake_from_recipe_fast_cache_and_contact_sheet(tmp_path: Path) -> None:
    step_path = _build_synthetic_bracket_step(tmp_path / "mini_bracket.step")
    recipe_path = tmp_path / "mini_bracket.yaml"
    cache_dir = tmp_path / "cache"
    assets_root = tmp_path / "assets"
    preview_path = tmp_path / "preview.png"

    recipe = CadAssetRecipe(
        category="gantry_brackets",
        asset_id="mini_bracket",
        name="Mini Test Bracket",
        step_path=step_path.name,
        reference_material_id="aluminum_6061_t6",
        mesh_size_mm=2.0,
        n_internal_modes=6,
        visual_max_triangles=5000,
        port0={
            "kind": "plane_box",
            "vp_coord_mm": [0.0, 0.0, 6.0],
            "params": {"axis": 2, "coord_mm": 6.0, "tol_mm": 0.45},
        },
        port1={
            "kind": "auto_face",
        },
    )
    recipe.to_yaml(recipe_path)

    # 1. Cold bake (fast=True)
    res1 = bake_from_recipe(
        recipe_path,
        fast=True,
        use_cache=True,
        cache_dir=cache_dir,
        preview_png=preview_path,
        assets_root=assets_root,
    )
    assert res1["cache_hit"] is False
    assert res1["qa"]["passed"] is True
    assert res1["manifest"]["n_internal_modes"] == 4  # capped by fast=True
    assert preview_path.is_file()
    png_bytes = preview_path.read_bytes()
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(png_bytes) > 5000

    # Verify standalone render_contact_sheet and GLB inspect fallback work cleanly
    glb_path = assets_root / "gantry_brackets" / "mini_bracket" / "visual.glb"
    glb_report = inspect_step_file(glb_path)
    assert glb_report.file_format == "glb"
    assert glb_report.overall_size_mm == pytest.approx([24.0, 22.0, 12.0], abs=0.2)

    out_png2 = tmp_path / "preview_direct.png"
    render_contact_sheet(
        glb_path,
        out_png2,
        ports_mm={"port0": (0.0, 0.0, 6.0), "port1": (0.0, 0.0, -6.0)},
        com_mm=tuple(res1["manifest"]["com_mm"]),
        title="Direct GLB Contact Sheet",
    )
    assert out_png2.is_file() and out_png2.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    # 2. Warm bake hits cache cleanly and fast
    res2 = bake_from_recipe(
        recipe_path,
        fast=True,
        use_cache=True,
        cache_dir=cache_dir,
        preview_png=tmp_path / "preview_cached.png",
        assets_root=tmp_path / "assets_copy",
    )
    assert res2["cache_hit"] is True
    assert res2["qa"]["passed"] is True
    assert res2["elapsed_s"] < 0.5
    assert (tmp_path / "preview_cached.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    # 3. Visuals-only refresh without re-meshing
    res3 = bake_from_recipe(
        recipe_path,
        fast=True,
        visuals_only=True,
        assets_root=assets_root,
    )
    assert res3["qa"]["passed"] is True


def test_verify_baked_asset_chube_compact() -> None:
    qa = verify_baked_asset("hotends", "chube_compact")
    assert qa.passed is True, qa.summary
    d = qa.to_dict()
    assert d["passed"] is True
    assert all(c["passed"] for c in d["checks"])


def test_template_yaml_is_valid_recipe() -> None:
    tpl = Path(__file__).resolve().parent.parent / "cad_recipes" / "TEMPLATE.yaml"
    assert tpl.is_file()
    raw = yaml.safe_load(tpl.read_text(encoding="utf-8"))
    recipe = CadAssetRecipe.model_validate(raw)
    assert recipe.category == "gantry_brackets"
    assert recipe.port0.kind == "plane_box"
    assert recipe.port1.kind == "z_cap"


def test_fetch_onshape_step_parsing_and_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    url = "https://cad.onshape.com/documents/doc123abc/w/ws456def/e/el789ghi"
    assert parse_onshape_reference(url) == ("doc123abc", "ws456def", "el789ghi")
    assert parse_onshape_reference("doc123abc", workspace_id="ws1", element_id="el2") == (
        "doc123abc",
        "ws1",
        "el2",
    )
    with pytest.raises(ValueError, match="Invalid Onshape document URL"):
        parse_onshape_reference("https://cad.onshape.com/bad/url")

    monkeypatch.delenv("ONSHAPE_ACCESS_KEY", raising=False)
    monkeypatch.delenv("ONSHAPE_SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ONSHAPE_ACCESS_KEY"):
        fetch_onshape_step(url, tmp_path / "out.step")

    # Mock httpx.Client to verify exact Onshape v6 STEP endpoint + query params without network
    captured: dict[str, object] = {}

    class _FakeResp:
        content = b"ISO-10303-21;\nEND-ISO-10303-21;\n"

        def raise_for_status(self) -> None:
            return None

    class _FakeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, req_url, params=None, auth=None):
            captured["url"] = req_url
            captured["params"] = params
            captured["auth"] = auth
            return _FakeResp()

    monkeypatch.setenv("ONSHAPE_ACCESS_KEY", "test_ak")
    monkeypatch.setenv("ONSHAPE_SECRET_KEY", "test_sk")
    with patch.object(httpx, "Client", _FakeClient):
        out = fetch_onshape_step(url, tmp_path / "onshape_part.step", configuration="length=50mm")

    assert out.is_file()
    assert b"ISO-10303-21" in out.read_bytes()
    assert captured["url"] == "https://cad.onshape.com/api/v6/partstudios/d/doc123abc/w/ws456def/e/el789ghi/step"
    assert captured["params"] == {"configuration": "length=50mm"}
    assert captured["auth"] == ("test_ak", "test_sk")
