"""Verification tests for newly baked catalog hotends (4) and Voron StealthBurner + Tap R8 CAD/FEA assets."""

from __future__ import annotations

from pathlib import Path

import pytest

from printsim.cad_assets.recipes import verify_baked_asset
from printsim.cad_assets.registry import has_hardware_asset, load_superelement
from printsim.engine import evaluate_printer
from printsim.schema import HotendSpec, PrinterSpec, TOOLHEAD_PRESETS

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"

NEW_HOTEND_IDS = (
    "e3d_v6_groove",
    "phaetus_dragon_uhf",
    "phaetus_rapido_2_uhf",
    "monolith_50mm_hf",
)

NEW_ASSETS = (
    ("hotends", "e3d_v6_groove", 25.0, 50.0, 80.0, 300.0),
    ("hotends", "phaetus_dragon_uhf", 45.0, 85.0, 300.0, 900.0),
    ("hotends", "phaetus_rapido_2_uhf", 40.0, 75.0, 450.0, 1150.0),
    ("hotends", "monolith_50mm_hf", 40.0, 75.0, 400.0, 1400.0),
    ("toolheads", "stealthburner_tap", 100.0, 190.0, 800.0, 3500.0),
)


@pytest.mark.parametrize(
    ("category", "asset_id", "min_mass_g", "max_mass_g", "min_kx", "max_kx"),
    NEW_ASSETS,
)
def test_baked_asset_qa_and_manifest_sanity(
    category: str,
    asset_id: str,
    min_mass_g: float,
    max_mass_g: float,
    min_kx: float,
    max_kx: float,
) -> None:
    """Every newly baked asset passes 11/11 QA checks with rb_max < 0.01 Hz and realistic mass/stiffness."""
    assert has_hardware_asset(category, asset_id)
    qa = verify_baked_asset(category, asset_id)
    assert qa.passed, f"{category}/{asset_id} QA failed: {qa.summary}"

    se = load_superelement(category, asset_id)
    assert se is not None
    m = se.manifest
    assert float(m["rigid_body_max_freq_hz"]) < 0.01
    mass_g = float(m["mass_kg"]) * 1000.0
    assert min_mass_g <= mass_g <= max_mass_g, f"{asset_id} mass {mass_g:.1f}g outside [{min_mass_g}, {max_mass_g}]"
    ps = m["port_stiffness_n_per_mm"]
    assert min_kx <= float(ps["kx"]) <= max_kx, f"{asset_id} kx={ps['kx']} outside [{min_kx}, {max_kx}]"
    assert float(ps["ky"]) > 0.0
    assert float(ps["kz"]) > 0.0
    assert se.visual_glb_path is not None and se.visual_glb_path.is_file()
    assert se.visual_glb_path.stat().st_size > 10_000


@pytest.mark.parametrize("hotend_id", NEW_HOTEND_IDS)
def test_evaluate_printer_attaches_new_hotend_cad_superelement(hotend_id: str) -> None:
    """evaluate_printer() automatically attaches each newly baked hotend Tet4 CMS superelement."""
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    spec.toolhead.hotend = HotendSpec(preset_id=hotend_id)
    rep = evaluate_printer(spec, include_advanced_analysis=False)

    nozzle_ses = [se for se in rep.assembly.superelements if se.id == "nozzle_stem"]
    assert len(nozzle_ses) == 1
    assert nozzle_ses[0].category == "hotends"
    assert nozzle_ses[0].asset_id == hotend_id
    assert rep.assembly.toolhead_stage_stiffnesses_n_per_mm["hotend_total_n_per_mm"] > 20.0


def test_hotend_hf_vs_uhf_and_v6_monotonic_stiffness() -> None:
    """Longer UHF melt extenders strictly reduce tip stiffness vs HF, and V6 groove is softest."""
    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")

    def _eval_he(hid: str) -> tuple[float, float]:
        se = load_superelement("hotends", hid)
        assert se is not None
        raw_kx = float(se.manifest["port_stiffness_n_per_mm"]["kx"])
        s = base.model_copy(deep=True)
        s.toolhead.hotend = HotendSpec(preset_id=hid)
        rep = evaluate_printer(s, include_advanced_analysis=False)
        he_tot = float(rep.assembly.toolhead_stage_stiffnesses_n_per_mm["hotend_total_n_per_mm"])
        return raw_kx, he_tot

    raw_dhf, tot_dhf = _eval_he("phaetus_dragon_hf")
    raw_duhf, tot_duhf = _eval_he("phaetus_dragon_uhf")
    assert raw_dhf > raw_duhf * 1.8
    assert tot_dhf > tot_duhf * 1.5

    raw_rhf, tot_rhf = _eval_he("phaetus_rapido_2_hf")
    raw_ruhf, tot_ruhf = _eval_he("phaetus_rapido_2_uhf")
    assert raw_rhf > raw_ruhf * 1.2
    assert tot_rhf > tot_ruhf * 1.15

    raw_v6, tot_v6 = _eval_he("e3d_v6_groove")
    raw_mono, tot_mono = _eval_he("monolith_50mm_hf")
    assert raw_mono > raw_v6 * 2.5
    assert tot_mono > tot_v6 * 2.5
    assert tot_dhf > tot_v6 * 4.0


def test_voron_tap_cad_superelement_and_monotonic_stiffness_order() -> None:
    """Voron Tap attaches dedicated stealthburner_tap CAD SE with static_eddy > cnc_tap > printed_tap."""
    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")

    def _run(preset_id: str, probe_type: str):
        s = base.model_copy(deep=True)
        s.toolhead.preset_id = preset_id
        s.toolhead.z_probe_type = probe_type
        if preset_id == "stealthburner_tap":
            p = TOOLHEAD_PRESETS["stealthburner_tap"]
            s.toolhead.total_mass_g = float(p["total_mass_g"])
            s.toolhead.cog_offset_mm = tuple(p["cog_offset_mm"])
            s.toolhead.nozzle_offset_mm = tuple(p["nozzle_offset_mm"])
        return evaluate_printer(s, include_advanced_analysis=False)

    rep_eddy = _run("stealthburner", "static_eddy")
    rep_cnc = _run("stealthburner", "cnc_tap")
    rep_printed = _run("stealthburner", "printed_tap")
    rep_tap_preset = _run("stealthburner_tap", "printed_tap")

    assert any(se.id == "toolhead_mount" and se.asset_id == "stealthburner" for se in rep_eddy.assembly.superelements)
    assert any(se.id == "toolhead_mount" and se.asset_id == "stealthburner_tap" for se in rep_cnc.assembly.superelements)
    assert any(se.id == "toolhead_mount" and se.asset_id == "stealthburner_tap" for se in rep_printed.assembly.superelements)
    assert any(se.id == "toolhead_mount" and se.asset_id == "stealthburner_tap" for se in rep_tap_preset.assembly.superelements)

    k_eddy = rep_eddy.assembly.toolhead_stage_stiffnesses_n_per_mm["body_n_per_mm"]
    k_cnc = rep_cnc.assembly.toolhead_stage_stiffnesses_n_per_mm["body_n_per_mm"]
    k_printed = rep_printed.assembly.toolhead_stage_stiffnesses_n_per_mm["body_n_per_mm"]
    assert k_eddy > k_cnc > k_printed

    tip_eddy = rep_eddy.static_fea.toolhead_flex_breakdown["effective_tip_stiffness_n_per_mm"]
    tip_cnc = rep_cnc.static_fea.toolhead_flex_breakdown["effective_tip_stiffness_n_per_mm"]
    tip_printed = rep_printed.static_fea.toolhead_flex_breakdown["effective_tip_stiffness_n_per_mm"]
    assert tip_eddy > tip_cnc > tip_printed

    assert rep_eddy.modal_fea.shaper_x.effective_freq_hz > rep_cnc.modal_fea.shaper_x.effective_freq_hz > rep_printed.modal_fea.shaper_x.effective_freq_hz
    assert rep_eddy.modal_fea.shaper_y.effective_freq_hz > rep_cnc.modal_fea.shaper_y.effective_freq_hz > rep_printed.modal_fea.shaper_y.effective_freq_hz

    assert rep_tap_preset.assembly.toolhead_stage_stiffnesses_n_per_mm["body_n_per_mm"] == pytest.approx(
        k_printed, rel=0.05
    )
