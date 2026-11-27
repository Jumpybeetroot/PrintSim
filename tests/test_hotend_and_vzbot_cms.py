"""End-to-end QA, multiphysics, and calibration tests for CAL-12 hotends & CAL-07 VzBot brackets.

Covers:
  - CAL-12 multi-material Tet4 CAD CMS hotends:
      * ``hotends/phaetus_dragon_hf``
      * ``hotends/e3d_revo_voron``
      * ``hotends/phaetus_rapido_2_hf``
      * ``hotends/mellow_goliath_air`` (upgraded from visual-only GLB to full CAD CMS)
  - CAL-07 VzBot 330 gantry brackets QA re-bake (all 8 stations clearing ``< 0.05 Hz`` rigid-body gate)
  - CAL-07 PrintDB-calibrated printed-polymer bracket anisotropy, moisture, HDT & creep knockdown
"""

from __future__ import annotations

import math
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
import pytest

from printsim.api.server import create_app
from printsim.cad_assets.recipes import verify_baked_asset
from printsim.cad_assets.registry import (
    clear_asset_cache,
    gantry_bracket_polymer_knockdown,
    get_hardware_asset_manifest,
    get_visual_glb_path,
    has_hardware_asset,
    has_visual_asset,
    list_hardware_assets,
    load_superelement,
)
from printsim.engine import evaluate_printer
from printsim.schema import HotendSpec, PrinterSpec, build_model_confidence_audit

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"

CAL12_HOTENDS = (
    # (asset_id, expected_port1_z_mm, expected_trust_pct, mass_range_g, k_lat_range_n_per_mm)
    ("e3d_revo_voron", -49.08, 86, (20.0, 32.0), (90.0, 220.0)),
    ("mellow_goliath_air", -69.24, 85, (72.0, 102.0), (600.0, 1200.0)),
    ("phaetus_rapido_2_hf", -54.50, 86, (38.0, 58.0), (800.0, 1700.0)),
    ("phaetus_dragon_hf", -45.51, 86, (44.0, 66.0), (1000.0, 2100.0)),
)

VZBOT_STATIONS = (
    "vzbot_330_rear_left",
    "vzbot_330_rear_right",
    "vzbot_330_awd_front_left",
    "vzbot_330_awd_front_right",
    "vzbot_330_2wd_front_left",
    "vzbot_330_2wd_front_right",
    "vzbot_330_xy_left",
    "vzbot_330_xy_right",
)


# ---------------------------------------------------------------------------
# 1. CAL-12: Baked Hotend CAD Superelements QA, Ordering & Multiphysics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "asset_id,expected_z_mm,expected_trust_pct,mass_rng_g,k_lat_rng",
    CAL12_HOTENDS,
)
def test_cal12_hotend_bundle_qa_and_matrices(
    asset_id: str,
    expected_z_mm: float,
    expected_trust_pct: int,
    mass_rng_g: tuple[float, float],
    k_lat_rng: tuple[float, float],
) -> None:
    """Every CAL-12 hotend passes 11/11 QA checks (< 0.05 Hz RB), SPD K_11, and datum/mass bounds."""
    clear_asset_cache()
    catalog = list_hardware_assets()
    assert asset_id in catalog["hotends"]
    assert has_hardware_asset("hotends", asset_id)
    assert has_visual_asset("hotends", asset_id)

    qa = verify_baked_asset("hotends", asset_id)
    assert qa.passed, f"QA failed for hotends/{asset_id}: {qa.summary}"

    se = load_superelement("hotends", asset_id)
    assert se is not None
    man = se.manifest
    assert man["has_superelement"] is True
    assert man["is_multi_material"] is True
    assert man["element_type"] == "Tet4"
    assert man["num_elements"] >= 15000
    assert man["rigid_body_max_freq_hz"] < 0.05
    assert man["cad_trust_pct"] == expected_trust_pct
    assert man["port0_mm"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-3)
    assert man["port1_mm"] == pytest.approx([0.0, 0.0, expected_z_mm], abs=0.15)

    mass_g = float(man["mass_kg"]) * 1000.0
    assert mass_rng_g[0] <= mass_g <= mass_rng_g[1], f"{asset_id} mass {mass_g:.2f} g out of {mass_rng_g}"

    ps = man["port_stiffness_n_per_mm"]
    k_lat = 0.5 * (float(ps["kx"]) + float(ps["ky"]))
    assert k_lat_rng[0] <= k_lat <= k_lat_rng[1], f"{asset_id} k_lat {k_lat:.1f} out of {k_lat_rng}"
    assert float(ps["kz"]) > k_lat * 10.0

    # Grounded Port 1 stiffness K_11 strictly SPD & consistent mass M_cb strictly SPD
    k11 = 0.5 * (se.K_bb[6:12, 6:12] + se.K_bb[6:12, 6:12].T)
    assert float(np.min(np.linalg.eigvalsh(k11))) > 0.0
    assert float(np.min(np.linalg.eigvalsh(0.5 * (se.M_cb + se.M_cb.T)))) > 0.0

    glb = get_visual_glb_path("hotends", asset_id)
    assert glb is not None and glb.is_file()
    assert 1000 < man["visual_triangles"] <= 35000


def test_cal12_hotend_physical_stiffness_ordering_and_confidence() -> None:
    """Thin-wall Revo Voron < long-lever Goliath Air < Rapido 2 HF < 4-standoff Dragon HF < Chube Compact."""
    stiffs: dict[str, float] = {}
    for hid in ("e3d_revo_voron", "mellow_goliath_air", "phaetus_rapido_2_hf", "phaetus_dragon_hf", "chube_compact"):
        man = get_hardware_asset_manifest("hotends", hid)
        assert man is not None and man["has_superelement"] is True
        ps = man["port_stiffness_n_per_mm"]
        stiffs[hid] = 0.5 * (float(ps["kx"]) + float(ps["ky"]))

    assert stiffs["e3d_revo_voron"] < stiffs["mellow_goliath_air"] < stiffs["phaetus_rapido_2_hf"]
    assert stiffs["phaetus_rapido_2_hf"] < stiffs["phaetus_dragon_hf"] < stiffs["chube_compact"]

    # End-to-end multiphysics evaluation + Model Confidence audit lift (58-70% -> 85-86%)
    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    for asset_id, _z, expected_trust, _m, _k in CAL12_HOTENDS:
        spec_cad = base.model_copy(deep=True)
        spec_cad.use_cad_superelements = True
        spec_cad.toolhead.hotend = HotendSpec(preset_id=asset_id)
        rep = evaluate_printer(spec_cad, include_advanced_analysis=False)
        assert any(se.category == "hotends" and se.asset_id == asset_id for se in rep.assembly.superelements)
        assert rep.assembly.to_geometry_dict()["toolhead_structure"]["hotend_visual_glb_url"] == (
            f"/api/assets/hardware/hotends/{asset_id}/visual.glb"
        )
        assert math.isfinite(rep.static_fea.toolhead_stiffness_y_n_per_mm)
        assert math.isfinite(rep.modal_fea.shaper_x.effective_freq_hz)

        audit_cad = {e["id"]: e for e in build_model_confidence_audit(spec_cad)["entries"]}
        assert audit_cad["hotend_clamp"]["cad_superelement_active"] is True
        assert audit_cad["hotend_clamp"]["trust_pct"] == expected_trust

        spec_lumped = spec_cad.model_copy(deep=True)
        spec_lumped.use_cad_superelements = False
        audit_lumped = {e["id"]: e for e in build_model_confidence_audit(spec_lumped)["entries"]}
        assert audit_lumped["hotend_clamp"]["cad_superelement_active"] is False
        assert audit_lumped["hotend_clamp"]["trust_pct"] <= 70

    # FastAPI GLB route serves every newly baked hotend
    client = TestClient(create_app())
    for asset_id, *_ in CAL12_HOTENDS:
        r = client.get(f"/api/assets/hardware/hotends/{asset_id}/visual.glb")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("model/gltf-binary")
        assert r.content[:4] == b"glTF"


def test_vzbot_330_awd_now_uses_goliath_cad_cms_and_qa_verified_brackets() -> None:
    """VzBot 330 AWD upgrades hotend_clamp to 85% (Goliath CAD CMS) & gantry_bracket to 84%."""
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "vzbot_330_awd.yaml")
    rep = evaluate_printer(spec, include_advanced_analysis=False)
    se_ids = {se.asset_id for se in rep.assembly.superelements}
    assert "mellow_goliath_air" in se_ids
    assert {"vzbot_330_rear_left", "vzbot_330_rear_right", "vzbot_330_awd_front_left", "vzbot_330_awd_front_right", "vzbot_330_xy_left", "vzbot_330_xy_right"} <= se_ids

    audit = {e["id"]: e for e in build_model_confidence_audit(spec)["entries"]}
    assert audit["hotend_clamp"]["cad_superelement_active"] is True
    assert audit["hotend_clamp"]["trust_pct"] == 85
    assert audit["gantry_bracket"]["cad_superelement_active"] is True
    assert audit["gantry_bracket"]["trust_pct"] == 84


# ---------------------------------------------------------------------------
# 2. CAL-07: All 8 VzBot Brackets Clear < 0.05 Hz QA Gate + L/R Mirror Symmetry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("asset_id", VZBOT_STATIONS)
def test_cal07_all_vzbot_brackets_pass_qa_gate(asset_id: str) -> None:
    """All 8 VzBot 330 gantry bracket stations clear the < 0.05 Hz rigid-body QA gate (11/11 checks)."""
    qa = verify_baked_asset("gantry_brackets", asset_id)
    assert qa.passed, f"QA failed for gantry_brackets/{asset_id}: {qa.summary}"
    man = get_hardware_asset_manifest("gantry_brackets", asset_id)
    assert man is not None
    assert man["rigid_body_max_freq_hz"] < 0.05
    assert man["cad_trust_pct"] == 84


def test_cal07_vzbot_bracket_mirror_symmetry() -> None:
    """Stratified VzBot corner mounts maintain tight Left/Right stiffness & mass mirror symmetry."""
    for base in ("vzbot_330_rear", "vzbot_330_awd_front", "vzbot_330_2wd_front"):
        ml = get_hardware_asset_manifest("gantry_brackets", f"{base}_left")
        mr = get_hardware_asset_manifest("gantry_brackets", f"{base}_right")
        assert ml is not None and mr is not None
        assert ml["mass_kg"] == pytest.approx(mr["mass_kg"], rel=0.01)
        for ax in ("kx", "ky", "kz"):
            assert ml["port_stiffness_n_per_mm"][ax] == pytest.approx(
                mr["port_stiffness_n_per_mm"][ax], rel=0.08
            )


# ---------------------------------------------------------------------------
# 3. CAL-07: PrintDB Polymer HDT / Moisture / Creep Knockdown Calibration
# ---------------------------------------------------------------------------


def test_cal07_polymer_bracket_knockdown_printdb_calibration_and_monotonicity() -> None:
    """Printed-polymer bracket knockdown reflects PrintDB HDT, moisture uptake, and creep limits."""
    # Metals have neutral 1.0 knockdown
    kd_al = gantry_bracket_polymer_knockdown("aluminum_6061_t6", chamber_temp_c=80.0, belt_tension_n=90.0)
    assert kd_al["total_stiffness_factor"] == pytest.approx(1.0)

    # All 6 printed polymers have finite, physically bounded knockdowns
    polymers = ("abs_printed", "pa12_cf_printed", "pet_cf_printed", "pc_pbt_gf_printed", "ppa_cf_printed", "pps_cf_printed")
    results_75 = {
        mid: gantry_bracket_polymer_knockdown(mid, chamber_temp_c=75.0, reference_is_fdm=False, belt_tension_n=60.0)
        for mid in polymers
    }
    for mid, kd in results_75.items():
        assert 0.30 <= kd["total_stiffness_factor"] <= 0.85, f"{mid}: {kd}"
        assert kd["effective_creep_limit_n"] <= kd["base_creep_limit_n"]

    # High-HDT hydrophobic PPS-CF beats hygroscopic PA12-CF & low-HDT ABS at 75 C / 60 N
    assert results_75["pps_cf_printed"]["moisture_conditioning_factor"] > 0.98
    assert results_75["pa12_cf_printed"]["moisture_conditioning_factor"] < 0.88
    assert results_75["pps_cf_printed"]["hdt_creep_factor"] > results_75["abs_printed"]["hdt_creep_factor"]
    assert results_75["ppa_cf_printed"]["effective_creep_limit_n"] > results_75["abs_printed"]["effective_creep_limit_n"] * 2.0

    # Monotonic temperature & sustained belt-tension derating on ABS
    kd_abs_25 = gantry_bracket_polymer_knockdown("abs_printed", chamber_temp_c=25.0, belt_tension_n=30.0)
    kd_abs_75_low = gantry_bracket_polymer_knockdown("abs_printed", chamber_temp_c=75.0, belt_tension_n=30.0)
    kd_abs_75_awd = gantry_bracket_polymer_knockdown("abs_printed", chamber_temp_c=75.0, belt_tension_n=89.0)
    assert kd_abs_25["total_stiffness_factor"] > kd_abs_75_low["total_stiffness_factor"] > kd_abs_75_awd["total_stiffness_factor"]

    # Runtime load_superelement applies polymer knockdown to metal-baked gantry brackets
    se_al = load_superelement("gantry_brackets", "vzbot_330_rear_left", material_id="aluminum_6061_t6", chamber_temp_c=75.0)
    se_ppa = load_superelement("gantry_brackets", "vzbot_330_rear_left", material_id="ppa_cf_printed", chamber_temp_c=75.0)
    se_abs = load_superelement("gantry_brackets", "vzbot_330_rear_left", material_id="abs_printed", chamber_temp_c=75.0)
    assert se_al is not None and se_ppa is not None and se_abs is not None
    assert "polymer_knockdown_applied" not in se_al.manifest
    assert "polymer_knockdown_applied" in se_ppa.manifest
    assert "polymer_knockdown_applied" in se_abs.manifest
    assert se_al.stiffness_scale_applied > se_ppa.stiffness_scale_applied > se_abs.stiffness_scale_applied
