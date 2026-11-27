"""VzBot 330 AWD fidelity: gantry-on-top-frame topology, per-side panel materials, CAD kit & toolhead.

Ground truth: official VzBoT3D/VzBoT-Vz330 BOM, ``Vz-330_v1.2`` full-printer STEP, Klipper
``Firmware/330AWD-printer.cfg``, and community ``[input_shaper]`` results (docs/community_touchstones.md).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from printsim.engine.evaluator import evaluate_printer
from printsim.geometry.builder import build_assembly
from printsim.kinematics.collision import check_frame_clearance
from printsim.schema import PrinterSpec
from printsim.schema.printer import (
    GANTRY_HARDWARE_KIT_CATALOG,
    STEPPER_CATALOG,
    TOOLHEAD_PRESETS,
)

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
VZ_YAML = EXAMPLES_DIR / "vzbot_330_awd.yaml"


@pytest.fixture(scope="module")
def vz_spec() -> PrinterSpec:
    return PrinterSpec.from_yaml(VZ_YAML)


@pytest.fixture(scope="module")
def vz_report(vz_spec: PrinterSpec):
    return evaluate_printer(vz_spec, include_advanced_analysis=False)


# ---------------------------------------------------------------------------
# 1. Gantry-on-top-frame topology (VzBot Y rails on the top-ring Y extrusions)
# ---------------------------------------------------------------------------


def test_gantry_on_top_frame_removes_duplicate_top_ring(vz_spec: PrinterSpec) -> None:
    asm = build_assembly(vz_spec)
    ids = {m.id for m in asm.members}
    nmap = asm.node_map
    # No separate stub columns or duplicate top-ring Y / rear members
    assert not any(i.startswith("col_") and "_top" in i for i in ids)
    assert not any(i.startswith(("top_left", "top_right", "top_rear")) for i in ids)
    assert any(i.startswith("top_front") for i in ids)  # real front top X extrusion stays
    # Gantry plane == top-ring centerline, with coincident corners rigidly tied
    for tag in ("fl", "fr", "rl", "rr"):
        assert np.allclose(nmap[f"g_{tag}"].position_m, nmap[f"t_{tag}"].position_m)
    ties = {sp.id for sp in asm.fastener_springs if sp.bay == "rigid_tie"}
    assert {"rigid_tie_t_fl", "rigid_tie_t_fr", "rigid_tie_t_rl", "rigid_tie_t_rr"} <= ties
    # Each Y-rail chain is one continuous collinear member run from front to rear corner
    for side, x_tag in (("l", "l"), ("r", "r")):
        segs = [m for m in asm.members if m.id.startswith(f"y_rail_{side}_")]
        assert len(segs) >= 2 and all(m.role == "rail" for m in segs)
        total = sum(m.length_m(nmap) for m in segs)
        span = float(np.linalg.norm(nmap[f"g_r{x_tag}"].position_m - nmap[f"g_f{x_tag}"].position_m))
        assert total == pytest.approx(span, rel=1e-9)
        # 10 mm Z-rod top anchors sit on (and split) the same rail extrusion
        rod_nodes = {f"t_rod_{side}f", f"t_rod_{side}r"}
        seg_nodes = {m.start_id for m in segs} | {m.end_id for m in segs}
        assert rod_nodes <= seg_nodes


def test_gantry_on_top_frame_is_lighter_and_monotonic(vz_spec: PrinterSpec) -> None:
    """Dropping the phantom double ring must remove extrusion mass, never add stiffness artifacts."""
    off = vz_spec.model_copy(deep=True)
    off.frame.gantry_on_top_frame = False
    off.frame.gantry_top_clearance_mm = 25.0
    a_on = build_assembly(vz_spec)
    a_off = build_assembly(off)
    frame_kg = lambda a: sum(m.mass_kg(a.node_map) for m in a.members if m.role in ("frame", "rail"))  # noqa: E731
    assert frame_kg(a_on) < frame_kg(a_off)
    assert len(a_on.members) < len(a_off.members)


def test_gantry_on_top_frame_ignored_on_flying_gantry() -> None:
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    spec.frame.gantry_on_top_frame = True
    asm = build_assembly(spec)
    assert any(m.id.startswith("col_") and m.id.endswith("_top") for m in asm.members)
    assert not any(sp.bay == "rigid_tie" for sp in asm.fastener_springs)


# ---------------------------------------------------------------------------
# 2. Per-side panel materials (aluminium rear + polycarbonate sides)
# ---------------------------------------------------------------------------


def test_side_panel_material_softens_and_lightens_sides(vz_spec: PrinterSpec) -> None:
    all_al = vz_spec.model_copy(deep=True)
    all_al.frame.side_panel_material = None
    a_pc = build_assembly(vz_spec)
    a_al = build_assembly(all_al)
    side_pc = [s for s in a_pc.shell_elements if s.bay.startswith(("left", "right"))]
    side_al = [s for s in a_al.shell_elements if s.bay.startswith(("left", "right"))]
    rear_pc = [s for s in a_pc.shell_elements if s.bay.startswith("rear")]
    assert side_pc and rear_pc
    assert all(s.material.id == "polycarbonate" for s in side_pc)
    assert all(s.material.id == "aluminum_5052_sheet" for s in rear_pc)
    assert all(s.material.id == "aluminum_5052_sheet" for s in side_al)
    assert a_pc.panel_mass_kg < a_al.panel_mass_kg
    # Softer side panels can only lower the Y-axis (side-wall racking) shaper frequency
    y_pc = evaluate_printer(vz_spec, include_advanced_analysis=False).modal_fea.shaper_y.effective_freq_hz
    y_al = evaluate_printer(all_al, include_advanced_analysis=False).modal_fea.shaper_y.effective_freq_hz
    assert y_pc <= y_al + 1e-6


# ---------------------------------------------------------------------------
# 3. Catalog entries (kit, toolhead, stepper)
# ---------------------------------------------------------------------------


def test_vzbot_catalog_entries() -> None:
    kit = GANTRY_HARDWARE_KIT_CATALOG["vzbot_330_awd"]["kit_settings"]
    assert kit["belt_type"] == "gt2_6mm"
    assert kit["x_rail_profile"] == "mgn9_rail" and kit["x_rail_mount"] == "top"
    assert kit["y_rail_profile"] == "mgn12_rail" and kit["y_rail_mount"] == "inner"
    # 150-189 Hz pluck over 150 mm on Gates 6 mm (8.3 g/m): T = 4 L^2 f^2 mu = 16.8..26.7 N
    mu = 0.0083
    lo, hi = (4.0 * 0.15**2 * f**2 * mu for f in (150.0, 189.0))
    assert lo <= kit["belt_pretension_n"] <= hi
    assert GANTRY_HARDWARE_KIT_CATALOG["vzbot_330_2wd"]["kit_settings"]["drive_mode"] == "2wd"

    th = TOOLHEAD_PRESETS["vz_printhead_cnc"]
    assert th["body_construction"] == "plate_sandwich_cf"
    st = STEPPER_CATALOG["ldo_2804ac"]
    assert st["inductance_mh"] == pytest.approx(0.6) and st["insulation_class"] == "B"


def test_apply_vzbot_kit_syncs_settings() -> None:
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_fixed_gantry_300.yaml")
    changes = spec.apply_gantry_hardware_kit("vzbot_330_awd")
    assert changes
    g = spec.gantry
    assert (g.drive_mode, g.x_rail_profile, g.x_rail_mount, g.y_rail_mount) == ("awd", "mgn9_rail", "top", "inner")
    assert g.belt.type == "gt2_6mm"


# ---------------------------------------------------------------------------
# 4. Preset geometry, mass and CAD-measured toolhead datums
# ---------------------------------------------------------------------------


def test_vzbot_preset_matches_official_bom(vz_spec: PrinterSpec) -> None:
    f = vz_spec.frame
    assert (f.width_mm, f.depth_mm, f.height_mm) == (530, 500, 570)
    assert f.profile == "extrusion_2020" and f.effective_upright_profile == "extrusion_2040"
    assert f.gantry_on_top_frame
    assert vz_spec.kinematics.build_volume_mm[2] == 400
    assert vz_spec.bed.z_leadscrew_type == "tr8x8_brass"
    assert vz_spec.bed.z_rail_profile == "steel_rod_10mm"
    assert vz_spec.resolve_effective_gantry_hardware_kit() == "vzbot_330_awd"


def test_vzbot_toolhead_cad_datums(vz_spec: PrinterSpec) -> None:
    asm = build_assembly(vz_spec)
    nmap = asm.node_map
    base = nmap["toolhead"].position_m
    noz = (nmap["nozzle_tip"].position_m - base) * 1e3
    cog = (nmap["toolhead_cog"].position_m - base) * 1e3
    # Vz-330_v1.2 CAD: Goliath tip 22.4 mm in front of / 54.2 mm below the CF tube axis
    assert noz[1] == pytest.approx(-22.4, abs=0.2)
    assert noz[2] == pytest.approx(-54.2, abs=0.2)
    # Extruder above the top-mounted MGN9H: assembled CoG above the tube axis (T-branch chassis)
    assert cog[2] == pytest.approx(13.4, abs=0.2) and cog[1] == pytest.approx(-10.3, abs=0.2)
    assert "toolhead_chassis" in nmap
    # 244 g on-rail with the real 26 g MGN9H + 96 g Goliath Air = 340 g moving toolhead
    assert asm.toolhead_moving_mass_kg * 1e3 == pytest.approx(340.0, abs=2.0)


def test_vzbot_cad_gantry_superelements_active(vz_spec: PrinterSpec) -> None:
    asm = build_assembly(vz_spec)
    se_ids = {se.asset_id for se in asm.superelements}
    assert {"vzbot_330_xy_left", "vzbot_330_xy_right"} <= se_ids
    assert {"vzbot_330_rear_left", "vzbot_330_rear_right"} <= se_ids
    assert {"vzbot_330_awd_front_left", "vzbot_330_awd_front_right"} <= se_ids


# ---------------------------------------------------------------------------
# 5. Physics vs community ADXL measurements
# ---------------------------------------------------------------------------


def test_vzbot_shaper_bands_match_community_adxl(vz_report) -> None:
    """VzBot 330 AWD Klipper configs: X ~75-118 Hz, Y ~47-80 Hz, Y always the lower axis."""
    m = vz_report.modal_fea
    fx = m.shaper_x.effective_freq_hz
    fy = m.shaper_y.effective_freq_hz
    assert 75.0 <= fx <= 125.0
    assert 47.0 <= fy <= 85.0
    assert fx > fy
    assert not vz_report.collision.has_collision


def test_vzbot_z_clearance_matches_cad_stack(vz_spec: PrinterSpec) -> None:
    """CAD Z stack: 20 bottom ring + 80 bed carriage + 400 Z + 54.2 nozzle + 10 top half = 564.2 mm."""
    rep = check_frame_clearance(build_assembly(vz_spec))
    assert rep.clearance_z_mm == pytest.approx(570.0 - 564.2, abs=1.5)
    tall = vz_spec.model_copy(deep=True)
    tall.kinematics.build_volume_mm = (330.0, 320.0, 420.0)
    assert check_frame_clearance(build_assembly(tall)).has_collision


def test_awd_beats_2wd_on_same_vzbot_frame(vz_spec: PrinterSpec) -> None:
    two = vz_spec.model_copy(deep=True)
    two.apply_gantry_hardware_kit("vzbot_330_2wd")
    r4 = evaluate_printer(vz_spec, include_advanced_analysis=False).modal_fea
    r2 = evaluate_printer(two, include_advanced_analysis=False).modal_fea
    assert r4.shaper_y.effective_freq_hz >= r2.shaper_y.effective_freq_hz
    assert r4.recommended_max_accel_mm_s2 >= r2.recommended_max_accel_mm_s2


def test_offset_datum_rail_mount_only_shifts_when_mount_differs(vz_spec: PrinterSpec) -> None:
    """Top-datum offsets are used verbatim on a top rail; front-datum presets keep the legacy shift."""
    noz = lambda s: (build_assembly(s).node_map["nozzle_tip"].position_m - build_assembly(s).node_map["toolhead"].position_m) * 1e3  # noqa: E731
    top_native = noz(vz_spec)
    assert top_native[2] == pytest.approx(vz_spec.toolhead.nozzle_offset_mm[2], abs=1e-6)

    # Same printhead moved to a front-mounted rail: undo the top datum (6 mm further out & up)
    front = vz_spec.model_copy(deep=True)
    front.gantry.x_rail_mount = "front"
    n_front = noz(front)
    assert n_front[1] == pytest.approx(top_native[1] - 6.0, abs=1e-6)
    assert n_front[2] == pytest.approx(top_native[2] + 6.0, abs=1e-6)

    # Front-datum toolhead on a top rail still gets the legacy top-mount shift
    legacy = vz_spec.model_copy(deep=True)
    legacy.toolhead.offset_datum_rail_mount = "front"
    n_legacy = noz(legacy)
    assert n_legacy[2] == pytest.approx(top_native[2] - 6.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 6. Full Y-rail attachment & CAD toolhead visuals
# ---------------------------------------------------------------------------


def test_inner_rail_carriage_rides_raceway_with_rigid_arm(vz_spec: PrinterSpec) -> None:
    """Inner-face MGN12 on 2020: raceway = 10 mm half extrusion + 4.8 mm raceway height inboard."""
    from printsim.fea.components import y_carriage_raceway_offset_m

    assert y_carriage_raceway_offset_m("mgn12_rail", 0.010) == pytest.approx(0.0148)
    asm = build_assembly(vz_spec)
    nm = asm.node_map
    for side, sign in (("l", 1.0), ("r", -1.0)):
        rail = nm[f"y_rail_car_{side}"].position_m
        car = nm[f"y_car_{side}"].position_m
        assert (car - rail)[0] == pytest.approx(sign * 0.0148, abs=1e-9)
        assert car[1:] == pytest.approx(rail[1:])
        sp = next(s for s in asm.fastener_springs if s.id == f"y_car_bearing_{side}")
        # Rigid arm from the raceway (frame node = carriage) back to the extrusion axis (shell = rail)
        assert np.asarray(sp.offset_m) == pytest.approx(rail - car)
        # XY-joint superelement now spans raceway -> CF-tube clamp (43.5 - 14.8 = 28.7 mm)
        xb = nm[f"x_beam_{side}"].position_m
        assert abs(xb[0] - car[0]) == pytest.approx(0.0435 - 0.0148, abs=2e-4)


def test_vzbot_cad_toolhead_and_goliath_visuals_match_fea_datums(vz_spec: PrinterSpec) -> None:
    from printsim.cad_assets.registry import get_hardware_asset_manifest

    th = build_assembly(vz_spec).to_geometry_dict()["toolhead_structure"]
    assert th["toolhead_visual_glb_url"].endswith("toolheads/vz_printhead_cnc/visual.glb")
    assert th["hotend_visual_glb_url"].endswith("hotends/mellow_goliath_air/visual.glb")
    se = next(x for x in th["cad_superelements_active"] if x["category"] == "toolheads")
    assert se["asset_id"] == "vz_printhead_cnc" and th["cad_toolhead_self_contained"]
    port1 = se["port1_mm"]
    goliath = get_hardware_asset_manifest("hotends", "mellow_goliath_air")
    assert goliath is not None and goliath["has_superelement"] is False  # visual-only hotend
    l_goliath = -goliath["port1_mm"][2]
    assert l_goliath == pytest.approx(vz_spec.toolhead.hotend.overall_length_mm, abs=0.1)
    # Drawn nozzle = MGN9H top face (10 mm tube half + 10 mm MGN9H) + CAD Port 1 - Goliath length
    assert 20.0 + port1[2] - l_goliath == pytest.approx(vz_spec.toolhead.nozzle_offset_mm[2], abs=0.1)
    assert port1[1] == pytest.approx(vz_spec.toolhead.nozzle_offset_mm[1], abs=0.1)


def test_printed_and_cnc_vz_printheads_are_structural_and_ordered(vz_spec: PrinterSpec) -> None:
    """Exact printed v24 cage (ABS) vs fitted CNC 6061 cage on the same datums: CNC is stiffer."""
    from printsim.cad_assets.registry import get_hardware_asset_manifest

    cnc = get_hardware_asset_manifest("toolheads", "vz_printhead_cnc")
    pr = get_hardware_asset_manifest("toolheads", "vz_printhead_printed")
    assert cnc["has_superelement"] and pr["has_superelement"]
    assert cnc["multi_piece_bolted"] and pr["multi_piece_bolted"]
    assert cnc["port1_mm"] == pytest.approx(pr["port1_mm"])  # identical hotend datum
    assert cnc["rigid_body_max_freq_hz"] < 0.1 and pr["rigid_body_max_freq_hz"] < 0.1
    # Fitted CNC plates reproduce Mellow's 27 g full-alu figure (cage + 6061 adapter box)
    vols = json_recipe_th("vz_printhead_cnc")["part_volumes_cm3"]
    cage_g = 2.7 * sum(v for k, v in vols.items() if k != "hotend_adapter")
    assert cage_g == pytest.approx(27.0, abs=1.0)
    hybrid_g = 1.55 * (vols["top_plate"] + vols["bottom_plate"]) + 2.7 * (vols["front_piece"] + vols["back_piece"])
    assert hybrid_g == pytest.approx(22.0, abs=1.0)

    printed = vz_spec.model_copy(deep=True)
    for k, v in TOOLHEAD_PRESETS["vz_printhead_printed"].items():
        if k in type(printed.toolhead).model_fields:
            setattr(printed.toolhead, k, v)
    s_cnc = solve_static(vz_spec)
    s_pr = solve_static(printed)
    assert s_cnc > s_pr  # 6061 cage stiffer at the nozzle than the bolted ABS cage
    assert build_assembly(printed).toolhead_moving_mass_kg * 1e3 == pytest.approx(344.0, abs=1.0)
    assert build_assembly(vz_spec).toolhead_moving_mass_kg * 1e3 == pytest.approx(340.0, abs=1.0)


def solve_static(spec: PrinterSpec) -> float:
    from printsim.fea.solver import solve_static_fea

    return float(solve_static_fea(build_assembly(spec)).toolhead_stiffness_x_n_per_mm)


def json_recipe_th(asset_id: str) -> dict:
    import json

    from printsim.cad_assets.registry import ASSETS_ROOT

    return json.loads((ASSETS_ROOT / "toolheads" / asset_id / "part_recipe.json").read_text())


def test_vzbot_xy_joint_visual_includes_y_carriage_block() -> None:
    from printsim.cad_assets.registry import get_hardware_asset_manifest

    for side in ("left", "right"):
        man = get_hardware_asset_manifest("gantry_brackets", f"vzbot_330_xy_{side}")
        assert man is not None and man["has_superelement"] is True
        recipe = json_recipe(f"vzbot_330_xy_{side}")
        assert recipe.get("visual_includes") == "MGN12H Y carriage block"


def json_recipe(asset_id: str) -> dict:
    import json

    from printsim.cad_assets.registry import ASSETS_ROOT

    return json.loads((ASSETS_ROOT / "gantry_brackets" / asset_id / "part_recipe.json").read_text())
