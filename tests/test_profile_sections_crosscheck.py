"""Cross-check ``PROFILE_CATALOG`` section properties against a `sectionproperties` 2D FEM.

Ground truth here is the warping-function finite-element solution of the actual cross-section:
closed-form shapes are rebuilt from the catalog dimensions, and the 2020 T-slot outline comes
from vendor CAD (``tests/fixtures/profile_sections/``, written by
``scripts/derive_profile_sections.py``). Skipped when the optional ``sections`` extra is absent.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

pytest.importorskip("sectionproperties")

from sectionproperties.analysis import Section  # noqa: E402
from sectionproperties.pre import Geometry  # noqa: E402
from shapely.geometry import Point, Polygon, box  # noqa: E402

from printsim.schema.profiles import (  # noqa: E402
    PROFILE_CATALOG,
    ProfileSpec,
    _hollow_rect_tube,
    _solid_rect,
)

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "profile_sections"


def _fem_properties(poly: Polygon, mesh_size_mm2: float) -> dict[str, float]:
    geom = Geometry(poly)
    geom.create_mesh(mesh_sizes=[mesh_size_mm2])
    sec = Section(geom)
    sec.calculate_geometric_properties()
    sec.calculate_warping_properties()
    ixx, iyy, _ = sec.get_ic()
    return {"area": float(sec.get_area()), "ixx": float(ixx), "iyy": float(iyy), "j": float(sec.get_j())}


def _rounded_box(w: float, h: float, r: float) -> Polygon:
    if r <= 1e-9:
        return box(-0.5 * w, -0.5 * h, 0.5 * w, 0.5 * h)
    return box(-0.5 * w + r, -0.5 * h + r, 0.5 * w - r, 0.5 * h - r).buffer(r, quad_segs=48)


def _catalog_mm(p: ProfileSpec) -> dict[str, float]:
    # Catalog iy uses h^3 (bending about the axis parallel to the width) = sectionproperties ixx
    return {
        "area": p.area_m2 * 1e6,
        "ixx": p.iy_m4 * 1e12,
        "iyy": p.iz_m4 * 1e12,
        "j": p.j_m4 * 1e12,
    }


def _is_formula_profile(p: ProfileSpec, ref: ProfileSpec) -> bool:
    return all(
        # PrintDB hydration rounds the stored values at about 1e-5 relative
        math.isclose(getattr(p, f), getattr(ref, f), rel_tol=1e-4)
        for f in ("area_m2", "iy_m4", "iz_m4", "j_m4")
    )


def _analytic_tube_ids() -> list[str]:
    out = []
    for pid, p in PROFILE_CATALOG.items():
        if p.shape != "tube" or not p.wall_thickness_m:
            continue
        ref = _hollow_rect_tube(
            "x", "x", p.width_m * 1e3, p.height_m * 1e3, p.wall_thickness_m * 1e3, p.corner_radius_m * 1e3
        )
        if _is_formula_profile(p, ref):
            out.append(pid)
    return out


def _analytic_bar_ids() -> list[str]:
    out = []
    for pid, p in PROFILE_CATALOG.items():
        if p.shape != "solid_bar":
            continue
        if _is_formula_profile(p, _solid_rect("x", "x", p.width_m * 1e3, p.height_m * 1e3, "solid_bar")):
            out.append(pid)
    return out


_TUBE_IDS = _analytic_tube_ids()
_BAR_IDS = _analytic_bar_ids()
_ROD_IDS = [pid for pid, p in PROFILE_CATALOG.items() if p.shape == "rod"]


def test_formula_profiles_were_found():
    # Guards the parametrised tests below against silently collecting nothing
    assert len(_TUBE_IDS) >= 10
    assert len(_ROD_IDS) >= 3
    assert len(_BAR_IDS) >= 1


@pytest.mark.parametrize("profile_id", _TUBE_IDS)
def test_hollow_tube_area_and_bending_are_exact_and_bredt_torsion_is_a_close_lower_bound(profile_id: str):
    p = PROFILE_CATALOG[profile_id]
    w, h = p.width_m * 1e3, p.height_m * 1e3
    t, r = p.wall_thickness_m * 1e3, p.corner_radius_m * 1e3
    poly = _rounded_box(w, h, r).difference(_rounded_box(w - 2.0 * t, h - 2.0 * t, max(r - t, 0.0)))
    fem = _fem_properties(poly, mesh_size_mm2=min(w, h) ** 2 / 1500.0)
    cat = _catalog_mm(p)

    assert cat["area"] == pytest.approx(fem["area"], rel=1e-3)
    assert cat["ixx"] == pytest.approx(fem["ixx"], rel=1e-3)
    assert cat["iyy"] == pytest.approx(fem["iyy"], rel=1e-3)
    # Bredt-Batho assumes shear flow uniform through the wall, so it under-predicts J as the
    # wall thickens. Measured shortfall across the catalog: 0.8 % (1 mm wall, rounded corners)
    # to 9.1 % (3 mm wall on a 20 mm sharp-cornered tube), growing roughly linearly with t / b.
    ratio = cat["j"] / fem["j"]
    assert 0.90 < ratio <= 1.005, f"{profile_id}: Bredt J / FEM J = {ratio:.4f}"
    assert 1.0 - ratio < 0.65 * (t / min(w, h)) + 0.005, f"{profile_id}: shortfall {1.0 - ratio:.4f}"


@pytest.mark.parametrize("profile_id", _ROD_IDS)
def test_round_rod_matches_fem(profile_id: str):
    p = PROFILE_CATALOG[profile_id]
    d = p.width_m * 1e3
    fem = _fem_properties(Point(0.0, 0.0).buffer(0.5 * d, quad_segs=96), mesh_size_mm2=d**2 / 1500.0)
    cat = _catalog_mm(p)
    for key in ("area", "ixx", "iyy", "j"):
        assert cat[key] == pytest.approx(fem[key], rel=1e-3), key


@pytest.mark.parametrize("profile_id", _BAR_IDS)
def test_solid_bar_roark_torsion_matches_fem(profile_id: str):
    p = PROFILE_CATALOG[profile_id]
    w, h = p.width_m * 1e3, p.height_m * 1e3
    fem = _fem_properties(box(-0.5 * w, -0.5 * h, 0.5 * w, 0.5 * h), mesh_size_mm2=min(w, h) ** 2 / 1500.0)
    cat = _catalog_mm(p)
    for key in ("area", "ixx", "iyy", "j"):
        assert cat[key] == pytest.approx(fem[key], rel=2e-3), key


# --------------------------------------------------------------------------- vendor-CAD T-slot outlines


def _load_fixture(profile_id: str) -> tuple[dict, dict[str, float]]:
    data = json.loads((_FIXTURES / f"{profile_id}.json").read_text())
    poly = Polygon(data["exterior"], holes=data["interiors"])
    assert poly.is_valid
    return data, _fem_properties(poly, mesh_size_mm2=0.05)


def test_2020_fixture_outline_reproduces_the_cad_cut_face():
    data, fem = _load_fixture("extrusion_2020")
    # Discretised outline area vs the exact B-Rep cut-face area recorded at extraction time
    assert fem["area"] == pytest.approx(data["cad_cut_face_area_mm2"], rel=1e-3)
    # Four-fold symmetric profile
    assert fem["ixx"] == pytest.approx(fem["iyy"], rel=1e-6)


def test_2020_catalog_area_and_bending_match_vendor_cad():
    _, fem = _load_fixture("extrusion_2020")
    cat = _catalog_mm(PROFILE_CATALOG["extrusion_2020"])
    # Misumi HFS5-2020 datasheet: 183 mm^2, 0.742 cm^4. Vendor CAD: 182.2 mm^2, 0.7428 cm^4.
    assert cat["area"] == pytest.approx(fem["area"], rel=0.01)
    assert cat["ixx"] == pytest.approx(fem["ixx"], rel=0.01)
    assert cat["iyy"] == pytest.approx(fem["iyy"], rel=0.01)


def test_2020_vendor_cad_torsion_constant_value():
    _, fem = _load_fixture("extrusion_2020")
    # Open T-slot St. Venant torsion constant from the warping analysis: 0.078 cm^4
    assert fem["j"] == pytest.approx(783.0, rel=0.01)
    # Open thin-walled section: J is a small fraction of the polar second moment
    assert fem["j"] / (fem["ixx"] + fem["iyy"]) < 0.06


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known catalog discrepancy: extrusion_2020 j_m4 = 0.11 cm^4 (scaled from Bosch/item tables) is "
        "about 40 % above the 0.078 cm^4 the vendor-CAD outline gives. Changing it shifts every 2020 "
        "frame and X-beam torsion result and the shaper baselines, so it is left for a deliberate "
        "recalibration. Remove this marker when the catalog value is updated."
    ),
)
def test_2020_catalog_torsion_constant_matches_vendor_cad():
    _, fem = _load_fixture("extrusion_2020")
    cat = _catalog_mm(PROFILE_CATALOG["extrusion_2020"])
    assert cat["j"] == pytest.approx(fem["j"], rel=0.10)
