"""Tests for offline CAD-to-Craig-Bampton baker and runtime hardware superelement registry."""

from __future__ import annotations

import json
import math
from pathlib import Path
import time

import numpy as np
import pytest
from scipy.linalg import eigh
import trimesh
from typer.testing import CliRunner

from printsim.cad_assets import (
    ASSETS_ROOT,
    BakedAssetSpec,
    MaterialRegion,
    PortDefinition,
    VisualPart,
    bake_solid_superelement,
    build_visual_glb,
    clear_asset_cache,
    get_hardware_asset_manifest,
    get_visual_glb_path,
    has_hardware_asset,
    list_hardware_assets,
    load_superelement,
    select_plane_box,
)
from printsim.cli import app
from printsim.fea import is_tet_fea_available
from printsim.geometry import is_cad_available
from printsim.schema import get_material

PHASE1_ASSETS: list[tuple[str, str]] = [
    ("hotends", "chube_compact"),
    ("hotends", "chube_air"),
    ("hotends", "chube_conduction"),
    ("hotends", "tricorn_short"),
    ("hotends", "tricorn_long"),
    ("toolheads", "microwaved_bowden"),
]

REQUIRED_MANIFEST_KEYS = {
    "schema_version",
    "category",
    "asset_id",
    "name",
    "source_cad",
    "source_url",
    "baked_at",
    "reference_material_id",
    "reference_youngs_modulus_pa",
    "reference_density_kg_m3",
    "is_multi_material",
    "is_fdm_anisotropic",
    "element_type",
    "num_vertices",
    "num_elements",
    "num_dofs",
    "n_boundary_dofs",
    "n_internal_modes",
    "rigid_body_max_freq_hz",
    "mass_kg",
    "com_mm",
    "port0_mm",
    "port1_mm",
    "port_stiffness_n_per_mm",
    "port_rot_stiffness_nm_per_rad",
    "fixed_interface_freqs_hz",
    "visual_glb",
    "visual_triangles",
    "visual_bytes",
    "cad_trust_pct",
    "source_hash",
}


@pytest.mark.skipif(
    not (is_tet_fea_available() and is_cad_available()),
    reason="Requires optional [cad] and [tetfea] extras (build123d, gmsh, scikit-fem, meshio).",
)
def test_baker_analytical_cantilever_beam(tmp_path: Path) -> None:
    """Bake a 60 x 12 x 12 mm Al 6061-T6 cantilever beam and verify analytical accuracy."""
    import build123d as b3d

    bar = b3d.Solid.make_box(60.0, 12.0, 12.0, plane=b3d.Plane(origin=(0.0, -6.0, -6.0)))
    spec = BakedAssetSpec(
        category="test_parts",
        asset_id="cantilever_60x12x12",
        name="Analytical Cantilever 60x12x12 Al6061",
        source_cad="procedural_box",
        source_url="local",
        reference_material_id="aluminum_6061_t6",
        cad_trust_pct=88,
        port0=PortDefinition((0.0, 0.0, 0.0), select_plane_box(0, 0.0, tol_mm=0.3)),
        port1=PortDefinition((60.0, 0.0, 0.0), select_plane_box(0, 60.0, tol_mm=0.3)),
        n_internal_modes=6,
        mesh_size_mm=1.8,
        is_multi_material=False,
        is_fdm_anisotropic=False,
    )

    res = bake_solid_superelement(
        spec=spec,
        regions=[MaterialRegion(bar, "aluminum_6061_t6", "cantilever_bar")],
        assets_root=tmp_path,
    )

    mat = get_material("aluminum_6061_t6")
    L = 0.060
    b = 0.012
    h = 0.012
    A = b * h
    I_y = (b * h**3) / 12.0
    E = mat.youngs_modulus_pa
    G = mat.effective_shear_modulus_pa
    kappa = 5.0 / 6.0

    # Exact Euler-Bernoulli + Timoshenko lateral tip stiffness & axial stiffness
    k_lat_exact_n_mm = 1.0 / ((L**3 / (3.0 * E * I_y) + L / (kappa * G * A)) * 1000.0)
    k_ax_exact_n_mm = (E * A / L) / 1000.0
    m_exact_kg = L * A * mat.density_kg_m3

    ps = res.manifest["port_stiffness_n_per_mm"]
    assert ps["ky"] == pytest.approx(k_lat_exact_n_mm, rel=0.08)
    assert ps["kz"] == pytest.approx(k_lat_exact_n_mm, rel=0.08)
    assert ps["kx"] == pytest.approx(k_ax_exact_n_mm, rel=0.03)
    assert res.manifest["mass_kg"] == pytest.approx(m_exact_kg, rel=0.01)
    assert res.manifest["com_mm"] == pytest.approx([30.0, 0.0, 0.0], abs=0.05)

    # Check 6 rigid-body modes < 0.5 Hz and positive elastic modes
    assert res.manifest["rigid_body_max_freq_hz"] < 0.5
    eigvals_free = eigh(res.K_cb, res.M_cb, eigvals_only=True)
    rigid_hz = np.sqrt(np.maximum(np.abs(eigvals_free[:6]), 0.0)) / (2.0 * math.pi)
    assert np.max(rigid_hz) < 0.5
    assert eigvals_free[6] > 0.0

    # Check Guyan K_bb == K_cb[:12, :12] and grounded SPD
    np.testing.assert_allclose(res.K_bb, res.K_cb[:12, :12], atol=1e-12)
    assert np.min(np.linalg.eigvalsh(res.K_bb[6:12, 6:12])) > 0.0

    # Check GLB magic header & loadability
    glb_path = res.asset_dir / "visual.glb"
    assert glb_path.is_file()
    glb_bytes = glb_path.read_bytes()
    assert glb_bytes[:4] == b"glTF"
    loaded_scene = trimesh.load(glb_path)
    assert loaded_scene is not None


def test_phase1_baked_assets_load_verify_order_and_scale() -> None:
    """Verify all 6 Phase 1 baked assets, SPD matrices, physical ordering, scaling, and GLBs."""
    clear_asset_cache()
    catalog = list_hardware_assets()
    assert "hotends" in catalog and "toolheads" in catalog and "assemblies" in catalog and "gantry_brackets" in catalog

    loaded_manifests: dict[str, dict] = {}
    for category, asset_id in PHASE1_ASSETS:
        assert has_hardware_asset(category, asset_id), f"Missing baked asset {category}/{asset_id}"
        assert asset_id in catalog[category]

        manifest = get_hardware_asset_manifest(category, asset_id)
        assert manifest is not None
        assert REQUIRED_MANIFEST_KEYS.issubset(manifest.keys())
        assert manifest["schema_version"] == 1
        assert manifest["category"] == category
        assert manifest["asset_id"] == asset_id
        assert manifest["n_boundary_dofs"] == 12
        assert 84 <= manifest["cad_trust_pct"] <= 88
        assert manifest["rigid_body_max_freq_hz"] < 0.5
        assert manifest["visual_url"] == f"/api/assets/hardware/{category}/{asset_id}/visual.glb"
        loaded_manifests[asset_id] = manifest

        # Check all 4 bundle files on disk
        adir = ASSETS_ROOT / category / asset_id
        assert (adir / "part_recipe.json").is_file()
        assert (adir / "manifest.json").is_file()
        assert (adir / "superelement.npz").is_file()
        glb_path = get_visual_glb_path(category, asset_id)
        assert glb_path is not None and glb_path.is_file()
        glb_bytes = glb_path.read_bytes()
        assert glb_bytes[:4] == b"glTF"
        assert 1000 < len(glb_bytes) < 1200 * 1024

        # Verify smooth creased normals & anti-octagonal triangle allocation in visual.glb
        vis_scene = trimesh.load(glb_path)
        assert isinstance(vis_scene, trimesh.Scene)
        for part_name, geom in vis_scene.geometry.items():
            vn = geom._cache.cache.get("vertex_normals")
            assert vn is not None and len(vn) == len(geom.vertices), (
                f"{asset_id}/{part_name} missing explicit vertex normals in GLB"
            )
            if any(k in part_name.lower() for k in ("screw", "bolt", "fastener", "set_screw")):
                assert len(geom.faces) <= 160
            else:
                assert len(geom.faces) >= 300, f"{asset_id}/{part_name} under-tessellated ({len(geom.faces)} faces)"
        if asset_id in ("chube_air", "chube_conduction"):
            shell_geom = vis_scene.geometry["ti_monocoque_shell"]
            assert len(shell_geom.faces) >= 1500
            assert "conduction_heatsink" in vis_scene.geometry if asset_id == "chube_conduction" else "air_heatsink_fins" in vis_scene.geometry

        # Check NPZ payload & load_superelement
        with np.load(adir / "superelement.npz") as npz:
            for key in (
                "K_cb",
                "M_cb",
                "K_bb",
                "port_coords_m",
                "fixed_interface_freqs_hz",
                "com_m",
                "inertia_kg_m2",
            ):
                assert key in npz

        asset = load_superelement(category, asset_id)
        assert asset is not None
        n_m = int(manifest["n_internal_modes"])
        assert asset.K_cb.shape == (12 + n_m, 12 + n_m)
        assert asset.M_cb.shape == (12 + n_m, 12 + n_m)
        assert asset.K_bb.shape == (12, 12)
        assert asset.port_coords_m.shape == (2, 3)
        np.testing.assert_allclose(asset.port_coords_m[0], [0.0, 0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(asset.K_bb, asset.K_cb[:12, :12], atol=1e-10)

        # Symmetry & positive definiteness checks
        np.testing.assert_allclose(asset.K_cb, asset.K_cb.T, rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(asset.M_cb, asset.M_cb.T, rtol=1e-7, atol=1e-12)
        assert float(np.min(np.linalg.eigvalsh(asset.M_cb))) > 0.0
        assert float(np.min(np.linalg.eigvalsh(asset.K_bb[6:12, 6:12]))) > 0.0
        assert float(np.min(np.linalg.eigvalsh(asset.inertia_kg_m2))) > 0.0

        eigvals_free = eigh(asset.K_cb, asset.M_cb, eigvals_only=True)
        rigid_hz = np.sqrt(np.maximum(np.abs(eigvals_free[:6]), 0.0)) / (2.0 * math.pi)
        assert float(np.max(rigid_hz)) < 0.5
        assert float(eigvals_free[6]) > 0.0

    # Physical stiffness ordering checks
    k_cond = loaded_manifests["chube_conduction"]["port_stiffness_n_per_mm"]
    k_air = loaded_manifests["chube_air"]["port_stiffness_n_per_mm"]
    assert k_cond["kx"] > k_air["kx"]
    assert k_cond["ky"] > k_air["ky"]

    k_tshort = loaded_manifests["tricorn_short"]["port_stiffness_n_per_mm"]
    k_tlong = loaded_manifests["tricorn_long"]["port_stiffness_n_per_mm"]
    assert k_tshort["kx"] > k_tlong["kx"]
    assert k_tshort["ky"] > k_tlong["ky"]

    # Material scaling check on mono-material toolhead (Ti-6Al-4V -> Al 6061-T6)
    ti_mat = get_material("titanium_grade5")
    al_mat = get_material("aluminum_6061_t6")
    mbt_ti = load_superelement("toolheads", "microwaved_bowden")
    mbt_al = load_superelement("toolheads", "microwaved_bowden", material_id="aluminum_6061_t6")
    assert mbt_ti is not None and mbt_al is not None
    expected_ek = al_mat.youngs_modulus_pa / ti_mat.youngs_modulus_pa
    expected_em = al_mat.density_kg_m3 / ti_mat.density_kg_m3
    np.testing.assert_allclose(mbt_al.K_cb, mbt_ti.K_cb * expected_ek, rtol=1e-12)
    np.testing.assert_allclose(mbt_al.K_bb, mbt_ti.K_bb * expected_ek, rtol=1e-12)
    np.testing.assert_allclose(mbt_al.M_cb, mbt_ti.M_cb * expected_em, rtol=1e-12)
    np.testing.assert_allclose(
        mbt_al.fixed_interface_freqs_hz,
        np.asarray(mbt_ti.fixed_interface_freqs_hz) * math.sqrt(expected_ek / expected_em),
        rtol=1e-12,
    )

    # Temperature softening check on polymer scaling
    mbt_abs_cold = load_superelement("toolheads", "microwaved_bowden", material_id="abs_printed", chamber_temp_c=25.0)
    mbt_abs_hot = load_superelement("toolheads", "microwaved_bowden", material_id="abs_printed", chamber_temp_c=95.0)
    assert mbt_abs_cold is not None and mbt_abs_hot is not None
    assert mbt_abs_hot.stiffness_scale_applied < mbt_abs_cold.stiffness_scale_applied * 0.85

    # Multi-material asset ignores material_id scaling, respects stiffness_mult
    chube_ref = load_superelement("hotends", "chube_compact")
    chube_al = load_superelement("hotends", "chube_compact", material_id="aluminum_6061_t6")
    chube_scaled = load_superelement("hotends", "chube_compact", material_id="aluminum_6061_t6", stiffness_mult=1.25)
    assert chube_ref is not None and chube_al is not None and chube_scaled is not None
    np.testing.assert_allclose(chube_al.K_cb, chube_ref.K_cb, rtol=1e-12)
    np.testing.assert_allclose(chube_al.M_cb, chube_ref.M_cb, rtol=1e-12)
    np.testing.assert_allclose(chube_scaled.K_cb, chube_ref.K_cb * 1.25, rtol=1e-12)

    # Cached repeat load latency (< 1 ms per call, typically ~0.01-0.03 ms)
    t0 = time.perf_counter()
    for _ in range(100):
        load_superelement("toolheads", "microwaved_bowden", material_id="aluminum_6061_t6")
    per_call_ms = (time.perf_counter() - t0) * 1000.0 / 100.0
    assert per_call_ms < 0.5


def test_coupled_microwaved_tricorn_assembly() -> None:
    """Verify coupled assemblies/microwaved_bowden__tricorn_long CAD superelement & brace gain."""
    clear_asset_cache()
    category = "assemblies"
    asset_id = "microwaved_bowden__tricorn_long"

    catalog = list_hardware_assets()
    assert category in catalog
    assert asset_id in catalog[category]
    assert has_hardware_asset(category, asset_id)

    manifest = get_hardware_asset_manifest(category, asset_id)
    assert manifest is not None
    assert REQUIRED_MANIFEST_KEYS.issubset(manifest.keys())
    assert manifest["schema_version"] == 1
    assert manifest["category"] == category
    assert manifest["asset_id"] == asset_id
    assert manifest["name"] == "Tricorn Long + Microwaved Ti Bottom Brace"
    assert manifest["cad_trust_pct"] == 89
    assert manifest["is_multi_material"] is True
    assert manifest["n_boundary_dofs"] == 12
    assert manifest["rigid_body_max_freq_hz"] < 0.5
    assert manifest["visual_url"] == f"/api/assets/hardware/{category}/{asset_id}/visual.glb"

    adir = ASSETS_ROOT / category / asset_id
    assert (adir / "part_recipe.json").is_file()
    assert (adir / "manifest.json").is_file()
    assert (adir / "superelement.npz").is_file()
    recipe = json.loads((adir / "part_recipe.json").read_text(encoding="utf-8"))
    assert recipe["toolhead_id"] == "microwaved_bowden"
    assert recipe["hotend_id"] == "tricorn_long"
    assert recipe["brace_part"] == "Hotend_Brace (Ti-6Al-4V, 0.32 g)"

    glb_path = get_visual_glb_path(category, asset_id)
    assert glb_path is not None and glb_path.is_file()
    glb_bytes = glb_path.read_bytes()
    assert glb_bytes[:4] == b"glTF"
    assert 1000 < len(glb_bytes) < 1200 * 1024

    vis_scene = trimesh.load(glb_path)
    assert isinstance(vis_scene, trimesh.Scene)
    assert "hotend_brace" in vis_scene.geometry
    for part_name, geom in vis_scene.geometry.items():
        vn = geom._cache.cache.get("vertex_normals")
        assert vn is not None and len(vn) == len(geom.vertices)
        if any(k in part_name.lower() for k in ("screw", "bolt", "fastener", "set_screw")):
            assert len(geom.faces) <= 160
        else:
            assert len(geom.faces) >= 300

    asset = load_superelement(category, asset_id)
    assert asset is not None
    n_m = int(manifest["n_internal_modes"])
    assert asset.K_cb.shape == (12 + n_m, 12 + n_m)
    assert asset.M_cb.shape == (12 + n_m, 12 + n_m)
    assert asset.K_bb.shape == (12, 12)
    np.testing.assert_allclose(asset.port_coords_m[0], [0.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(asset.port_coords_m[1], [0.0, 0.0, -0.066], atol=1e-9)
    np.testing.assert_allclose(asset.K_bb, asset.K_cb[:12, :12], atol=1e-10)
    np.testing.assert_allclose(asset.K_cb, asset.K_cb.T, rtol=1e-7, atol=1e-6)
    np.testing.assert_allclose(asset.M_cb, asset.M_cb.T, rtol=1e-7, atol=1e-12)
    assert float(np.min(np.linalg.eigvalsh(asset.M_cb))) > 0.0
    assert float(np.min(np.linalg.eigvalsh(asset.K_bb[6:12, 6:12]))) > 0.0
    assert float(np.min(np.linalg.eigvalsh(asset.inertia_kg_m2))) > 0.0

    eigvals_free = eigh(asset.K_cb, asset.M_cb, eigvals_only=True)
    rigid_hz = np.sqrt(np.maximum(np.abs(eigvals_free[:6]), 0.0)) / (2.0 * math.pi)
    assert float(np.max(rigid_hz)) < 0.5
    assert float(eigvals_free[6]) > 0.0

    # Multi-material assembly ignores material_id scaling, respects stiffness_mult
    asset_al = load_superelement(category, asset_id, material_id="aluminum_6061_t6")
    asset_scaled = load_superelement(category, asset_id, stiffness_mult=1.2)
    assert asset_al is not None and asset_scaled is not None
    np.testing.assert_allclose(asset_al.K_cb, asset.K_cb, rtol=1e-12)
    np.testing.assert_allclose(asset_scaled.K_cb, asset.K_cb * 1.2, rtol=1e-12)

    # Parallel Ti-6Al-4V bottom brace strictly increases lateral tip stiffness vs unbraced Tricorn Long
    unbraced = get_hardware_asset_manifest("hotends", "tricorn_long")
    assert unbraced is not None
    k_braced = manifest["port_stiffness_n_per_mm"]
    k_unbraced = unbraced["port_stiffness_n_per_mm"]
    assert k_braced["kx"] > k_unbraced["kx"]
    assert k_braced["ky"] > k_unbraced["ky"]
    assert k_braced["kz"] > k_unbraced["kz"]


def test_cli_list_assets_and_missing_asset_graceful() -> None:
    """Verify `printsim list-assets --json` and graceful None on unknown assets."""
    assert has_hardware_asset("hotends", "does_not_exist") is False
    assert load_superelement("hotends", "does_not_exist") is None
    assert get_hardware_asset_manifest("hotends", None) is None
    assert get_visual_glb_path("toolheads", "../bad") is None

    runner = CliRunner()
    res = runner.invoke(app, ["list-assets", "--json"])
    assert res.exit_code == 0, res.output
    payload = json.loads(res.stdout)
    assert len(payload["hotends"]) >= 5
    assert "microwaved_bowden" in payload["toolheads"]
    assert "microwaved_bowden__tricorn_long" in payload["assemblies"]


def test_cli_bake_asset_step_roundtrip(tmp_path: Path) -> None:
    """Verify `printsim bake-asset` end-to-end from a STEP file to a loadable bundle."""
    if not (is_cad_available() and is_tet_fea_available()):
        pytest.skip("build123d / gmsh / scikit-fem required for CLI STEP bake test")

    import build123d as b3d

    step_path = tmp_path / "test_bracket.step"
    out_root = tmp_path / "baked_assets"
    box = b3d.Box(50.0, 12.0, 12.0)
    b3d.export_step(box, str(step_path))

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "bake-asset",
            str(step_path),
            "--category",
            "gantry_brackets",
            "--asset-id",
            "test_step_bar",
            "--material",
            "aluminum_6061_t6",
            "--port0-axis",
            "0",
            "--port0-end",
            "min",
            "--port1-axis",
            "0",
            "--port1-end",
            "max",
            "--mesh-size",
            "2.0",
            "--modes",
            "4",
            "--out-dir",
            str(out_root),
            "--json",
        ],
    )
    assert res.exit_code == 0, res.output
    manifest = json.loads(res.stdout)
    assert manifest["category"] == "gantry_brackets"
    assert manifest["asset_id"] == "test_step_bar"
    assert manifest["rigid_body_max_freq_hz"] < 0.5
    bundle_dir = out_root / "gantry_brackets" / "test_step_bar"
    for fname in ("part_recipe.json", "superelement.npz", "manifest.json", "visual.glb"):
        assert (bundle_dir / fname).is_file()
    # Closed-form EB + Timoshenko lateral tip stiffness for 50x12x12 Al6061-T6 bar:
    # I = 12*12^3 / 12 = 1728 mm^4, k_eb = 3*E*I/L^3 = 2857.4 N/mm, k_tim = 2714.6 N/mm
    ky = manifest["port_stiffness_n_per_mm"]["ky"]
    assert abs(ky - 2714.6) / 2714.6 < 0.10


def test_visual_glb_budget_creased_normals_and_doublesided(tmp_path: Path) -> None:
    """Verify fastener-capped budgeting, creased vertex normals, and doubleSided glTF flags."""
    main_cyl = trimesh.creation.cylinder(radius=12.7, height=60.0, sections=48)
    screw_hi = trimesh.creation.icosphere(subdivisions=3, radius=2.5)  # 1,280 faces
    assert len(screw_hi.faces) > 1000

    parts = [
        VisualPart(main_cyl, "outer_shell", "#b8c2ce", metallic=0.86, roughness=0.28, double_sided=True),
    ] + [
        VisualPart(screw_hi.copy(), f"torx_screw_{i}", "#d6dee8", metallic=0.92, roughness=0.24)
        for i in range(6)
    ]

    glb_path = tmp_path / "test_visual.glb"
    total_tris, total_bytes = build_visual_glb(parts, glb_path, max_total_triangles=1200)
    assert total_bytes > 1000
    assert total_tris <= 1200

    loaded = trimesh.load(glb_path)
    assert isinstance(loaded, trimesh.Scene)

    shell_mesh = loaded.geometry["outer_shell"]
    # 48-section cylinder has 192 raw faces (< 400 floor) -> preserved uncrushed!
    assert len(shell_mesh.faces) == len(main_cyl.faces)
    assert shell_mesh.visual.material.doubleSided is True
    # Crease-splitting separates top/bottom flat cap vertices from curved barrel vertices
    assert len(shell_mesh.vertices) > len(main_cyl.vertices)
    vn = shell_mesh._cache.cache.get("vertex_normals")
    assert vn is not None and len(vn) == len(shell_mesh.vertices)

    for i in range(6):
        sc_mesh = loaded.geometry[f"torx_screw_{i}"]
        assert len(sc_mesh.faces) <= 160
        assert sc_mesh.visual.material.doubleSided is False


