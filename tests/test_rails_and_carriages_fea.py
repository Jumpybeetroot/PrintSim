"""Unit & 3D Tet10 (P2) FEA physics parity tests for all solid linear rails, carriages & bushings."""

from __future__ import annotations

from pathlib import Path

import pytest

from printsim.engine import evaluate_printer
from printsim.fea.beam3d import beam_shear_coefficients
from printsim.fea.components import (
    _CARRIAGE_SADDLE_STIFFNESS_TET10,
    hertzian_rail_carriage_stiffness,
)
from printsim.geometry.builder import build_assembly
from printsim.schema import (
    PROFILE_CATALOG,
    PrinterSpec,
    get_printdb,
    get_profile,
    is_printdb_available,
)
from printsim.schema.printdb_bridge import PROFILE_PRINTDB_MAP, get_record_specs

RHO_STEEL = 7850.0

SOLID_RAIL_IDS = ("mgn9_rail", "mgn12_rail", "mgn15_rail", "hgh15_rail")

ALL_7_PROFILED_CARRIAGES: tuple[tuple[str, str], ...] = (
    ("mgn9_rail", "C"),
    ("mgn9_rail", "H"),
    ("mgn12_rail", "C"),
    ("mgn12_rail", "H"),
    ("mgn15_rail", "C"),
    ("mgn15_rail", "H"),
    ("hgh15_rail", "C"),
)


# ---------------------------------------------------------------------------
# 1. Solid Profiled Linear Rails — 3D Tet10 (P2) Section Properties & Invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("profile_id", "width_mm", "height_mm", "catalog_kg_m"),
    (
        ("mgn9_rail", 9.0, 6.5, 0.380),
        ("mgn12_rail", 12.0, 8.0, 0.650),
        ("mgn15_rail", 15.0, 10.0, 1.060),
        ("hgh15_rail", 15.0, 15.0, 1.450),
    ),
)
def test_solid_rails_tet10_sections_replace_solid_rect_approx(
    profile_id: str, width_mm: float, height_mm: float, catalog_kg_m: float
) -> None:
    """Every solid profiled rail uses Tet10 grooved+counterbored section properties < naive solid rect."""
    prof = get_profile(profile_id)
    assert prof.shape == "rail"
    assert prof.width_m == pytest.approx(width_mm * 1e-3, rel=1e-9)
    assert prof.height_m == pytest.approx(height_mm * 1e-3, rel=1e-9)
    # Exact HIWIN catalog linear mass @ 7850 kg/m^3 steel
    assert prof.area_m2 * RHO_STEEL == pytest.approx(catalog_kg_m, rel=1e-6)

    # Naive un-grooved, un-drilled rectangle upper bounds
    w, h = prof.width_m, prof.height_m
    rect_iy = w * (h**3) / 12.0
    rect_iz = h * (w**3) / 12.0
    a_long, b_short = max(w, h), min(w, h)
    rect_j = a_long * (b_short**3) * (1.0 / 3.0 - 0.21 * (b_short / a_long) * (1.0 - (b_short / a_long) ** 4 / 12.0))

    # Gothic-arch / circular-arc raceway grooves + M3/M4 counterbores strictly reduce Iy, Iz, J
    assert 0.65 * rect_iy < prof.iy_m4 < 0.95 * rect_iy
    assert 0.60 * rect_iz < prof.iz_m4 < 0.92 * rect_iz
    assert 0.50 * rect_j < prof.j_m4 < 0.90 * rect_j

    # Tet10 Timoshenko shear factors are present and physically bounded (0.45..0.90)
    assert prof.shear_kappa_y is not None and 0.45 < prof.shear_kappa_y < 0.90
    assert prof.shear_kappa_z is not None and 0.70 < prof.shear_kappa_z < 0.90


def test_solid_rail_hierarchy_and_waisted_hgr15_geometry() -> None:
    """Verify strict MGN9 < MGN12 < MGN15 < HGH15 hierarchy and HGR15 waisted Iy > Iz physics."""
    r9 = get_profile("mgn9_rail")
    r12 = get_profile("mgn12_rail")
    r15 = get_profile("mgn15_rail")
    h15 = get_profile("hgh15_rail")

    assert r9.area_m2 < r12.area_m2 < r15.area_m2 < h15.area_m2
    assert r9.iy_m4 < r12.iy_m4 < r15.iy_m4 < h15.iy_m4
    assert r9.iz_m4 < r12.iz_m4 < r15.iz_m4 < h15.iz_m4
    assert r9.j_m4 < r12.j_m4 < r15.j_m4 < h15.j_m4
    assert (r9.shear_kappa_y or 0.0) < (r12.shear_kappa_y or 0.0) < (r15.shear_kappa_y or 0.0) < (h15.shear_kappa_y or 0.0)

    # MGN rails are wider than tall (Iz > Iy); waisted 15x15 HGR15R cuts deeply into +-Y flanks (Iy > Iz)
    assert r9.iz_m4 > r9.iy_m4
    assert r12.iz_m4 > r12.iy_m4
    assert r15.iz_m4 > r15.iy_m4
    assert h15.iy_m4 > h15.iz_m4


def test_solid_vs_pocketed_and_hollow_rail_invariants() -> None:
    """Every solid rail strictly exceeds its underside-pocketed / hollow counterparts in A, Iy, Iz, J, ky."""
    pairs = (
        ("mgn9_rail", "mgn9_pocketed_rail"),
        ("mgn12_rail", "mgn12_pocketed_rail"),
        ("mgn12_rail", "mgn12_hollow_rail"),
        ("mgn15_rail", "mgn15_pocketed_rail"),
    )
    for solid_id, milled_id in pairs:
        sol = get_profile(solid_id)
        mil = get_profile(milled_id)
        assert mil.area_m2 < sol.area_m2
        assert mil.iy_m4 < sol.iy_m4
        assert mil.iz_m4 < sol.iz_m4
        assert mil.j_m4 < sol.j_m4
        assert (mil.shear_kappa_y or 0.0) < (sol.shear_kappa_y or 0.0)


# ---------------------------------------------------------------------------
# 2. All 7 HIWIN Profiled Carriages — 3D Tet10 5-DOF Saddle Compliance
# ---------------------------------------------------------------------------


def test_all_7_profiled_carriages_monotonic_roll_and_class_hierarchy() -> None:
    """Verify monotonic MGN9C < MGN9H < MGN12C < MGN12H < MGN15C < MGN15H < HGH15CA roll stiffness
    and strict monotonicity across all 5 DOFs within C-class, H-class, and C vs H pairs."""
    for pre in ("Z1", "ZA", "ZB"):
        res_all = [hertzian_rail_carriage_stiffness(rid, preload=pre, block_type=bt) for rid, bt in ALL_7_PROFILED_CARRIAGES]
        # Strict 7-carriage monotonic roll stiffness hierarchy
        for i in range(len(res_all) - 1):
            assert res_all[i]["k_rx_nm_rad"] < res_all[i + 1]["k_rx_nm_rad"], (
                f"Roll non-monotonic at preload={pre}: "
                f"{ALL_7_PROFILED_CARRIAGES[i]} ({res_all[i]['k_rx_nm_rad']:.1f}) vs "
                f"{ALL_7_PROFILED_CARRIAGES[i + 1]} ({res_all[i + 1]['k_rx_nm_rad']:.1f})"
            )

    # Strict same-block-class monotonicity across ALL 5 DOFs and ALL preloads
    for pre in ("ZF", "Z0", "Z1", "ZA", "ZB"):
        c_series = [
            hertzian_rail_carriage_stiffness(rid, preload=pre, block_type="C")
            for rid in ("mgn9_rail", "mgn12_rail", "mgn15_rail", "hgh15_rail")
        ]
        for i in range(len(c_series) - 1):
            for dof_key in ("k_yy_n_m", "k_zz_n_m", "k_rx_nm_rad", "k_ry_nm_rad", "k_rz_nm_rad"):
                assert c_series[i][dof_key] < c_series[i + 1][dof_key]

        h_series = [
            hertzian_rail_carriage_stiffness(rid, preload=pre, block_type="H")
            for rid in ("mgn9_rail", "mgn12_rail", "mgn15_rail", "hgh15_rail")
        ]
        for i in range(len(h_series) - 1):
            for dof_key in ("k_yy_n_m", "k_zz_n_m", "k_rx_nm_rad", "k_ry_nm_rad", "k_rz_nm_rad"):
                assert h_series[i][dof_key] < h_series[i + 1][dof_key]


def test_c_vs_h_block_roll_and_pitch_ratios() -> None:
    """Verify physically realistic C vs H block roll (~1.35-1.60x) and pitch/yaw (~2.5-4.1x) ratios."""
    for rid in ("mgn9_rail", "mgn12_rail", "mgn15_rail"):
        c_res = hertzian_rail_carriage_stiffness(rid, preload="Z1", block_type="C")
        h_res = hertzian_rail_carriage_stiffness(rid, preload="Z1", block_type="H")

        # H-block beats C-block on all 5 DOFs
        for dof_key in ("k_yy_n_m", "k_zz_n_m", "k_rx_nm_rad", "k_ry_nm_rad", "k_rz_nm_rad"):
            assert h_res[dof_key] > c_res[dof_key]

        # Roll scales roughly linearly with ball-row length L1 (~1.35x - 1.60x)
        roll_ratio = h_res["k_rx_nm_rad"] / c_res["k_rx_nm_rad"]
        assert 1.35 <= roll_ratio <= 1.65

        # Pitch & yaw scale roughly as L1^3 (~2.8x - 4.2x), moderated by Tet10 overhang flex
        pitch_ratio = h_res["k_ry_nm_rad"] / c_res["k_ry_nm_rad"]
        yaw_ratio = h_res["k_rz_nm_rad"] / c_res["k_rz_nm_rad"]
        assert 2.8 <= pitch_ratio <= 4.3
        assert 2.8 <= yaw_ratio <= 4.3


def test_tet10_saddle_anisotropy_and_preload_series_scaling() -> None:
    """3D Tet10 C-block saddle is stiffer vertically than laterally and preload-independent in series."""
    for rid, bt in ALL_7_PROFILED_CARRIAGES:
        k_sk_yy, k_sk_zz, k_sk_rx, k_sk_ry, k_sk_rz = _CARRIAGE_SADDLE_STIFFNESS_TET10[(rid, bt)]
        assert k_sk_zz > k_sk_yy > 0.0
        assert k_sk_rx > 0.0 and k_sk_ry > 0.0 and k_sk_rz > 0.0

        z1 = hertzian_rail_carriage_stiffness(rid, preload="Z1", block_type=bt)
        # Anisotropy: lateral skirt compliance exceeds vertical arch compliance (eta_yy < eta_zz)
        assert z1["saddle_eta_yy"] < z1["saddle_eta_zz"]
        assert 0.70 < z1["saddle_eta_yy"] < 0.92
        assert 0.85 < z1["saddle_eta_zz"] < 0.96
        assert 0.85 < z1["saddle_eta_rx"] < 0.96

        # Preload-independent series physics: heavier preload (ZB) increases saddle compliance share
        # relative to light preload (ZF) because k_hertz rises while steel k_skirt is constant!
        zf = hertzian_rail_carriage_stiffness(rid, preload="ZF", block_type=bt)
        zb = hertzian_rail_carriage_stiffness(rid, preload="ZB", block_type=bt)
        assert zb["saddle_eta_yy"] < z1["saddle_eta_yy"] < zf["saddle_eta_yy"]
        assert zb["saddle_eta_rx"] < z1["saddle_eta_rx"] < zf["saddle_eta_rx"]


def test_hgh15ca_hiwin_catalog_radial_stiffness_parity() -> None:
    """HGH15CA vertical/radial stiffness matches HIWIN HG15C catalog (Z0 ~ 196, ZA ~ 365 N/um)."""
    hg_z0 = hertzian_rail_carriage_stiffness("hgh15_rail", preload="Z0", block_type="C")
    hg_za = hertzian_rail_carriage_stiffness("hgh15_rail", preload="ZA", block_type="C")
    assert 180.0 <= hg_z0["k_zz_n_m"] * 1e-6 <= 215.0
    assert 320.0 <= hg_za["k_zz_n_m"] * 1e-6 <= 390.0
    # Tall 28 mm HGH15CA has thicker crown -> higher roll efficiency eta_rx than MGN12H
    m12h = hertzian_rail_carriage_stiffness("mgn12_rail", preload="Z1", block_type="H")
    hg_z1 = hertzian_rail_carriage_stiffness("hgh15_rail", preload="Z1", block_type="C")
    assert hg_z1["saddle_eta_rx"] > m12h["saddle_eta_rx"]


# ---------------------------------------------------------------------------
# 3. Misumi LM8UU / LM10UU / LM12UU Cylindrical Sleeve Compliance Validation
# ---------------------------------------------------------------------------


def test_lmxuu_cylindrical_sleeve_axisymmetry_and_hierarchy() -> None:
    """Closed cylindrical steel sleeves (LM8/10/12UU & LUU) are axisymmetric with >99% sleeve efficiency."""
    for bt in ("C", "H"):
        c8 = hertzian_rail_carriage_stiffness("steel_rod_8mm", preload="Z1", block_type=bt)
        c10 = hertzian_rail_carriage_stiffness("steel_rod_10mm", preload="Z1", block_type=bt)
        c12 = hertzian_rail_carriage_stiffness("steel_rod_12mm", preload="Z1", block_type=bt)

        for c in (c8, c10, c12):
            assert c["k_yy_n_m"] == pytest.approx(c["k_zz_n_m"], rel=1e-12)
            assert c["k_ry_nm_rad"] == pytest.approx(c["k_rz_nm_rad"], rel=1e-12)
            # Tet10 cylindrical sleeve FEA confirms <0.2% steel sleeve compliance share
            assert c["saddle_eta_yy"] > 0.995
            assert c["saddle_eta_zz"] > 0.995

        for dof_key in ("k_yy_n_m", "k_zz_n_m", "k_rx_nm_rad", "k_ry_nm_rad", "k_rz_nm_rad"):
            assert c8[dof_key] < c10[dof_key] < c12[dof_key]

    # LUU long bushing beats UU standard bushing on every rod diameter
    for rod_id in ("steel_rod_8mm", "steel_rod_10mm", "steel_rod_12mm"):
        uu = hertzian_rail_carriage_stiffness(rod_id, preload="Z1", block_type="C")
        luu = hertzian_rail_carriage_stiffness(rod_id, preload="Z1", block_type="H")
        assert luu["k_yy_n_m"] > uu["k_yy_n_m"]
        assert luu["k_ry_nm_rad"] > uu["k_ry_nm_rad"] * 2.0


# ---------------------------------------------------------------------------
# 4. PrintDB Persistence & End-to-End Assembly / FEA Smoke Verification
# ---------------------------------------------------------------------------


def test_printdb_solid_rails_and_tet10_saddle_reference_parity() -> None:
    """All 4 solid rails (including hgh15_rail) and Tet10 saddle constants live in tracked PrintDB/data/."""
    assert "hgh15_rail" in PROFILE_CATALOG
    assert PROFILE_PRINTDB_MAP["hgh15_rail"] == "components/profiles/profile-hgh15-rail"
    if not is_printdb_available():
        pytest.skip("PrintDB repository not accessible in this sandbox")

    db = get_printdb()
    assert db is not None
    for pid in SOLID_RAIL_IDS:
        rid = PROFILE_PRINTDB_MAP[pid]
        rec = db.get(rid)
        assert rec.origin == "community"
        si = get_record_specs(rid, si=True)
        assert si is not None
        prof = get_profile(pid)
        assert prof.iy_m4 == pytest.approx(float(si["iy_m4"]), rel=1e-6)
        assert prof.iz_m4 == pytest.approx(float(si["iz_m4"]), rel=1e-6)
        assert prof.j_m4 == pytest.approx(float(si["j_m4"]), rel=1e-6)

    ref = db.get("references/materials_mechanics/tet10-carriage-saddle-compliance")
    assert ref.origin == "community"
    assert float(ref.specs["mgn12h_k_skirt_yy_mn_per_m"]) == pytest.approx(1765.9)
    assert float(ref.specs["hgh15ca_k_skirt_rx_nm_per_rad"]) == pytest.approx(206196.0)
    assert float(ref.specs["hgh15_rail_shear_kappa_y_ratio"]) == pytest.approx(0.782)


def test_solid_rails_and_hgh15_end_to_end_evaluation() -> None:
    """Bare solid rail shear_kappa activates cleanly and HGH15 upgrades gantry stiffness over MGN12."""
    base = PrinterSpec.from_yaml(Path("examples/corexy_voron_style.yaml"))

    spec_mgn12 = base.model_copy(deep=True)
    spec_mgn12.gantry.x_rail_profile = "mgn12_rail"
    spec_hgh15 = base.model_copy(deep=True)
    spec_hgh15.gantry.x_rail_profile = "hgh15_rail"

    ev_12 = evaluate_printer(spec_mgn12)
    ev_h15 = evaluate_printer(spec_hgh15)
    assert ev_h15.assembly.x_rail_mass_kg > ev_12.assembly.x_rail_mass_kg
    assert ev_h15.static_fea.toolhead_stiffness_y_n_per_mm > ev_12.static_fea.toolhead_stiffness_y_n_per_mm
    assert ev_h15.modal_fea.fundamental_freq_hz > 20.0

    # Bare unsupported MGN12 rail uses Tet10-calibrated Timoshenko shear coefficients
    spec_bare = base.model_copy(deep=True)
    spec_bare.gantry.x_beam_profile = "unsupported_rail"
    spec_bare.gantry.x_rail_profile = "mgn12_rail"
    spec_bare.gantry.x_rail_mount = "front"
    asm_bare = build_assembly(spec_bare)
    m_x = next(m for m in asm_bare.members if m.id == "x_gantry_l")
    ky, kz = beam_shear_coefficients(m_x)
    assert ky == pytest.approx(0.812, abs=1e-3)
    assert kz == pytest.approx(0.646, abs=1e-3)
