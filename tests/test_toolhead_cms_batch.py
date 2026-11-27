"""End-to-end QA & multiphysics tests for CAL-02 baked toolhead CAD CMS assets.

Covers:
  - ``toolheads/xol_compact``    (XOL 2 Compact Dual-4010 + Sherpa Mini)
  - ``toolheads/a4t_beacon``     (A4T Cowling + Dragon HE Duct + Xol Carriage)
  - ``toolheads/dragon_burner``  (Dragon Burner V8 + Orbiter 2 Mount + MGN12H Carriage)
  - ``toolheads/archetype_cpap`` (Archetype Breakneck v24 CPAP + Sherpa Micro + MGN12H)
"""

from __future__ import annotations

import math
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
import pytest
import trimesh

from printsim.api.server import create_app
from printsim.cad_assets.recipes import verify_baked_asset
from printsim.cad_assets.registry import (
    get_visual_glb_path,
    has_hardware_asset,
    has_visual_asset,
    list_hardware_assets,
    load_superelement,
)
from printsim.engine import evaluate_printer
from printsim.schema import PrinterSpec, TOOLHEAD_PRESETS
from printsim.schema.overrides import build_model_confidence_audit

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"

CAL02_TOOLHEADS = (
    # (asset_id, expected_port1_mm, expected_trust_pct, expected_body_range_n_per_mm, expected_tip_range_n_per_mm)
    ("xol_compact", (-0.1, -24.35, 4.68), 86, (100.0, 260.0), (50.0, 130.0)),
    ("a4t_beacon", (0.0, -24.35, 4.88), 84, (85.0, 240.0), (45.0, 120.0)),
    ("dragon_burner", (0.0, -20.0, -3.54), 83, (100.0, 260.0), (50.0, 130.0)),
    ("archetype_cpap", (0.0, -21.7, 13.5), 82, (45.0, 160.0), (30.0, 95.0)),
)


def _spec_with_toolhead(base: PrinterSpec, preset_id: str, *, use_cad: bool = True) -> PrinterSpec:
    p = TOOLHEAD_PRESETS[preset_id]
    s = base.model_copy(deep=True)
    s.use_cad_superelements = use_cad
    s.toolhead.preset_id = preset_id
    s.toolhead.name = str(p["name"])
    s.toolhead.total_mass_g = float(p["total_mass_g"])
    s.toolhead.cog_offset_mm = tuple(p["cog_offset_mm"])
    s.toolhead.nozzle_offset_mm = tuple(p["nozzle_offset_mm"])
    s.toolhead.mount_material = str(p["mount_material"])
    s.toolhead.body_construction = str(p["body_construction"])
    s.toolhead.internal_stiffness_n_per_mm = float(p["internal_stiffness_n_per_mm"])
    s.toolhead.z_probe_type = str(p["z_probe_type"])
    return s


@pytest.mark.parametrize(
    "asset_id,expected_port1_mm,expected_trust_pct,_body_rng,_tip_rng",
    CAL02_TOOLHEADS,
)
def test_cal02_baked_toolhead_bundle_qa_and_geometry(
    asset_id: str,
    expected_port1_mm: tuple[float, float, float],
    expected_trust_pct: int,
    _body_rng: tuple[float, float],
    _tip_rng: tuple[float, float],
) -> None:
    """Every CAL-02 baked toolhead passes 11/11 QA checks, compact GLB bounds, and datum invariants."""
    catalog = list_hardware_assets()
    assert asset_id in catalog["toolheads"]
    assert has_hardware_asset("toolheads", asset_id)
    assert has_visual_asset("toolheads", asset_id)

    qa = verify_baked_asset("toolheads", asset_id)
    assert qa.passed, f"QA failed for toolheads/{asset_id}: {qa.summary}"

    se = load_superelement("toolheads", asset_id)
    assert se is not None
    assert se.K_bb.shape == (12, 12)
    assert se.manifest["element_type"] == "Tet4"
    assert se.manifest["num_elements"] >= 20000
    assert se.manifest["rigid_body_max_freq_hz"] < 0.05
    assert se.manifest["is_fdm_anisotropic"] is True
    assert se.manifest["cad_trust_pct"] == expected_trust_pct
    assert se.manifest["port0_mm"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-3)
    assert se.manifest["port1_mm"] == pytest.approx(list(expected_port1_mm), abs=0.25)

    # First fixed-interface elastic mode must be realistic (> 150 Hz; no floppy detached ducts)
    assert len(se.fixed_interface_freqs_hz) == 6
    assert se.fixed_interface_freqs_hz[0] > 150.0

    # Grounded Port 1 stiffness submatrix K_11 must be strictly SPD
    k11 = 0.5 * (se.K_bb[6:12, 6:12] + se.K_bb[6:12, 6:12].T)
    assert np.all(np.linalg.eigvalsh(k11) > 0.0)

    # Visual GLB exists, within triangle budget, and has compact toolhead bounds (no stray 450mm belts/hoses)
    glb_path = get_visual_glb_path("toolheads", asset_id)
    assert glb_path is not None and glb_path.is_file()
    assert 1000 < se.manifest["visual_triangles"] <= 35000
    scene = trimesh.load(str(glb_path), force="scene")
    extents_mm = np.asarray(scene.extents, dtype=float) * 1000.0
    assert extents_mm[0] < 95.0, f"{asset_id} GLB X width too wide ({extents_mm[0]:.1f} mm — stray belt?)"
    assert extents_mm[1] < 110.0, f"{asset_id} GLB Y depth too deep ({extents_mm[1]:.1f} mm)"
    assert extents_mm[2] < 150.0, f"{asset_id} GLB Z height too tall ({extents_mm[2]:.1f} mm — stray CPAP hose?)"


@pytest.mark.parametrize(
    "asset_id,_port1,expected_trust_pct,body_rng,tip_rng",
    CAL02_TOOLHEADS,
)
def test_cal02_toolhead_multiphysics_stiffness_and_confidence(
    asset_id: str,
    _port1: tuple[float, float, float],
    expected_trust_pct: int,
    body_rng: tuple[float, float],
    tip_rng: tuple[float, float],
) -> None:
    """Evaluating each CAL-02 preset attaches its CAD CMS, yields realistic stiffness, and lifts confidence."""
    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")

    # 1. CAD-enabled evaluation
    spec_cad = _spec_with_toolhead(base, asset_id, use_cad=True)
    rep_cad = evaluate_printer(spec_cad, include_advanced_analysis=False)
    assert any(se.asset_id == asset_id for se in rep_cad.assembly.superelements)

    geom = rep_cad.assembly.to_geometry_dict()
    assert geom["toolhead_structure"]["toolhead_visual_glb_url"] == (
        f"/api/assets/hardware/toolheads/{asset_id}/visual.glb"
    )

    k_body = rep_cad.assembly.toolhead_stage_stiffnesses_n_per_mm["body_n_per_mm"]
    k_tip = rep_cad.static_fea.toolhead_flex_breakdown["effective_tip_stiffness_n_per_mm"]
    assert body_rng[0] <= k_body <= body_rng[1], f"{asset_id} body stiffness {k_body:.1f} out of {body_rng}"
    assert tip_rng[0] <= k_tip <= tip_rng[1], f"{asset_id} tip stiffness {k_tip:.1f} out of {tip_rng}"
    assert math.isfinite(rep_cad.modal_fea.shaper_x.effective_freq_hz)
    assert math.isfinite(rep_cad.modal_fea.shaper_y.effective_freq_hz)

    # 2. Model confidence jumps from lumped 48% to CAD 82-86%
    audit_cad = build_model_confidence_audit(spec_cad)
    tc_cad = next(e for e in audit_cad["entries"] if e["id"] == "toolhead_construction")
    assert tc_cad["cad_superelement_active"] is True
    assert tc_cad["trust_pct"] == expected_trust_pct

    spec_lumped = _spec_with_toolhead(base, asset_id, use_cad=False)
    audit_lumped = build_model_confidence_audit(spec_lumped)
    tc_lumped = next(e for e in audit_lumped["entries"] if e["id"] == "toolhead_construction")
    assert tc_lumped["cad_superelement_active"] is False
    assert tc_lumped["trust_pct"] <= 50

    # 3. Material scaling monotonicity: ABS < PPA-CF < CNC Al 6061 (with 0.58x bolted seam retention)
    s_abs = _spec_with_toolhead(base, asset_id, use_cad=True)
    s_abs.toolhead.mount_material = "abs_printed"
    s_ppa = _spec_with_toolhead(base, asset_id, use_cad=True)
    s_ppa.toolhead.mount_material = "ppa_cf_printed"
    s_al = _spec_with_toolhead(base, asset_id, use_cad=True)
    s_al.toolhead.mount_material = "aluminum_6061_t6"

    k_abs = evaluate_printer(s_abs, include_advanced_analysis=False).assembly.toolhead_stage_stiffnesses_n_per_mm["body_n_per_mm"]
    k_ppa = evaluate_printer(s_ppa, include_advanced_analysis=False).assembly.toolhead_stage_stiffnesses_n_per_mm["body_n_per_mm"]
    k_al = evaluate_printer(s_al, include_advanced_analysis=False).assembly.toolhead_stage_stiffnesses_n_per_mm["body_n_per_mm"]
    assert k_abs < k_ppa < k_al
    # CNC Al retains bolted clamshell seam knockdown (0.58x) rather than unphysical infinite billet
    assert k_al < 8000.0

    # 4. Printed Tap compliance monotonicity: printed_tap strongly softens body & tip stiffness and lowers X shaper freq
    s_tap = _spec_with_toolhead(base, asset_id, use_cad=True)
    s_tap.toolhead.z_probe_type = "printed_tap"
    rep_tap = evaluate_printer(s_tap, include_advanced_analysis=False)
    assert rep_tap.assembly.toolhead_stage_stiffnesses_n_per_mm["body_n_per_mm"] < k_body * 0.5
    assert rep_tap.static_fea.toolhead_flex_breakdown["effective_tip_stiffness_n_per_mm"] < k_tip * 0.75
    assert rep_tap.modal_fea.shaper_x.effective_freq_hz < rep_cad.modal_fea.shaper_x.effective_freq_hz


def test_cal02_example_yamls_and_api_glb_routes() -> None:
    """Example YAMLs (doomcube_4040, voron_monolith_awd) and FastAPI GLB endpoints work end-to-end."""
    for yaml_name, expected_th, expected_trust in (
        ("doomcube_4040.yaml", "a4t_beacon", 84),
        ("voron_monolith_awd.yaml", "archetype_cpap", 82),
    ):
        spec = PrinterSpec.from_yaml(EXAMPLES_DIR / yaml_name)
        assert spec.toolhead.preset_id == expected_th
        rep = evaluate_printer(spec, include_advanced_analysis=False)
        assert any(se.asset_id == expected_th for se in rep.assembly.superelements)
        assert rep.static_fea.toolhead_flex_breakdown["effective_tip_stiffness_n_per_mm"] > 35.0
        audit = build_model_confidence_audit(spec)
        tc = next(e for e in audit["entries"] if e["id"] == "toolhead_construction")
        assert tc["cad_superelement_active"] is True
        assert tc["trust_pct"] == expected_trust

    client = TestClient(create_app())
    for asset_id, *_ in CAL02_TOOLHEADS:
        resp = client.get(f"/api/assets/hardware/toolheads/{asset_id}/visual.glb")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("model/gltf-binary")
        assert len(resp.content) > 10_000
