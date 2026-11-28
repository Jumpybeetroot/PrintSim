"""Tests for 3D Tet4 FEA-calibrated CNC-milled lightweight X-beam profiles and visual GLBs."""

from __future__ import annotations

from pathlib import Path

import pytest

from printsim.cad_assets.registry import get_hardware_asset_manifest, get_visual_glb_path, list_hardware_assets
from printsim.engine import evaluate_printer
from printsim.schema import PrinterSpec, get_profile


MILLED_XBEAM_IDS = (
    "cnc_al_monocoque_20x20",
    "cnc_al_truss_voron24_lw",
    "cnc_al_fysetc_v0_15x15",
)


@pytest.mark.parametrize("profile_id", MILLED_XBEAM_IDS)
def test_milled_xbeam_profile_and_visual_asset_present(profile_id: str) -> None:
    prof = get_profile(profile_id)
    assert prof.area_m2 > 0.0
    assert prof.iy_m4 > 0.0
    assert prof.iz_m4 > 0.0
    assert prof.j_m4 > 0.0

    glb = get_visual_glb_path("x_beams", profile_id)
    assert glb is not None and glb.is_file()
    man = get_hardware_asset_manifest("x_beams", profile_id)
    assert man is not None
    assert man["visual_url"] == f"/api/assets/hardware/x_beams/{profile_id}/visual.glb"
    assert float(man["length_mm"]) > 150.0

    catalog = list_hardware_assets()
    assert "x_beams" in catalog
    assert profile_id in catalog["x_beams"]


def test_milled_xbeam_physics_monotonicity() -> None:
    stock_2020 = get_profile("extrusion_2020")
    monocoque = get_profile("cnc_al_monocoque_20x20")
    voron24_lw = get_profile("cnc_al_truss_voron24_lw")
    fysetc_v0 = get_profile("cnc_al_fysetc_v0_15x15")

    # All milled beams must be lighter per meter than stock 2020 extrusion
    assert monocoque.area_m2 < stock_2020.area_m2 * 0.60
    assert voron24_lw.area_m2 < monocoque.area_m2
    assert fysetc_v0.area_m2 < voron24_lw.area_m2

    # Solid MGN12H front-flange monocoque truss must beat 4-face-cut truss on both Iz and specific stiffness (Iz/A)
    assert monocoque.iz_m4 > voron24_lw.iz_m4 * 2.0
    assert (monocoque.iz_m4 / monocoque.area_m2) > (voron24_lw.iz_m4 / voron24_lw.area_m2) * 1.4
    # Closed-flange monocoque must also beat open T-slot stock 2020 in torsion J
    assert monocoque.j_m4 > stock_2020.j_m4


def test_milled_xbeam_end_to_end_evaluation() -> None:
    spec_path = Path("examples/corexy_voron_style.yaml")
    spec = PrinterSpec.from_yaml(spec_path)
    spec.gantry.x_beam_profile = "cnc_al_monocoque_20x20"
    spec.gantry.x_beam_material = "aluminum_7075_t6"

    res = evaluate_printer(spec).to_dict()
    xvis = res["geometry"].get("x_beam_visual")
    assert xvis is not None
    assert xvis["asset_id"] == "cnc_al_monocoque_20x20"
    assert xvis["visual_glb_url"] == "/api/assets/hardware/x_beams/cnc_al_monocoque_20x20/visual.glb"
    assert xvis["length_mm"] == pytest.approx(380.0, abs=1.0)
    # 380-430mm beam mass should be ~90-115g in Al 7075-T6 (much lighter than ~170-185g stock 2020)
    assert 70.0 < res["geometry"]["mass_summary"]["x_beam_mass_g"] < 125.0


@pytest.mark.parametrize("profile_id", MILLED_XBEAM_IDS)
def test_milled_xbeam_rail_mount_face_is_front_minus_y(profile_id: str) -> None:
    """All milled X-beam visual GLBs must orient their rail-mounting face on CAD -Y (front)."""
    import numpy as np
    import trimesh

    glb = get_visual_glb_path("x_beams", profile_id)
    assert glb is not None and glb.is_file()
    sc = trimesh.load(str(glb))
    mesh = sc.to_geometry() if isinstance(sc, trimesh.Scene) else sc
    bb = mesh.bounds
    centers = mesh.triangles_center
    normals = mesh.face_normals

    # Inspect the middle 70% of the span along X, near each of the 4 outer side faces
    mid_x = np.abs(centers[:, 0]) < 0.35 * (bb[1][0] - bb[0][0])
    cl_flat_counts: dict[str, int] = {}
    for label, ax, sign in (("+Y", 1, 1), ("-Y", 1, -1), ("+Z", 2, 1), ("-Z", 2, -1)):
        bound = bb[1][ax] if sign > 0 else bb[0][ax]
        other_ax = 2 if ax == 1 else 1
        near_face = mid_x & (np.abs(centers[:, ax] - bound) < 0.0025)
        flat = near_face & (normals[:, ax] * sign > 0.95)
        on_cl = flat & (np.abs(centers[:, other_ax]) < 0.002)
        cl_flat_counts[label] = int(on_cl.sum())

    # The rail-mounting face (-Y, front) has the centerline mounting strip/flange,
    # whereas the 3 pocketed faces (+Y, +Z, -Z) have cutouts along the centerline.
    assert cl_flat_counts["-Y"] > max(
        cl_flat_counts["+Y"], cl_flat_counts["+Z"], cl_flat_counts["-Z"]
    ) * 2, f"{profile_id} rail-mount face is not on -Y (front): {cl_flat_counts}"
