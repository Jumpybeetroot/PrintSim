"""End-to-end tests for structural Voron / LDO / Siboor / Monolith CAD kits and StealthBurner."""

from __future__ import annotations

import math
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
import pytest

from printsim.api.server import create_app
from printsim.cad_assets.registry import (
    has_hardware_asset,
    list_hardware_assets,
    load_superelement,
)
from printsim.engine import evaluate_printer
from printsim.schema import PrinterSpec
from printsim.schema.overrides import build_model_confidence_audit, resolve_gantry_bracket

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


EXPECTED_NEW_PRESETS = (
    ("ldo_voron_24_revd_300.yaml", "ldo_voron_kit", "stealthburner"),
    ("ldo_voron_trident_revd_300.yaml", "ldo_voron_kit", "stealthburner"),
    ("siboor_voron_trident_awd_300.yaml", "siboor_cnc_awd", "stealthburner"),
    ("siboor_voron_24_cnc_300.yaml", "siboor_cnc_2wd", "stealthburner"),
    ("voron_monolith_awd.yaml", "monolith_ldo_cnc_awd", "archetype_cpap"),
    ("corexy_voron_style.yaml", "voron_stock_printed", "stealthburner"),
    ("voron_trident_300.yaml", "voron_stock_printed", "stealthburner"),
)


def test_baked_voron_monolith_siboor_stealthburner_assets_present_and_valid() -> None:
    """Verify all newly baked toolhead, gantry bracket, and printed part bundles pass QA invariants."""
    catalog = list_hardware_assets()

    # 1. Voron StealthBurner + CW2.5 CAD superelement
    assert "stealthburner" in catalog["toolheads"]
    assert has_hardware_asset("toolheads", "stealthburner")
    sb = load_superelement("toolheads", "stealthburner")
    assert sb is not None
    assert sb.K_bb.shape == (12, 12)
    assert sb.manifest["rigid_body_max_freq_hz"] < 0.05
    assert sb.visual_glb_path is not None and sb.visual_glb_path.is_file()
    # Port 1 must align with exact StealthBurner hotend axis & Revo Voron 4-bolt shelf in STEP
    assert sb.manifest["port1_mm"] == pytest.approx([0.2, -28.9, -1.9], abs=0.25)
    # Disjoint MGN12H bolt face (Port 0) and Revo 4-bolt shelf (Port 1): no overlapping sphere stiffening
    assert sb.manifest["port_stiffness_n_per_mm"]["kx"] < 4000.0
    assert sb.manifest["port_rot_stiffness_nm_per_rad"]["krx"] < 1200.0
    k11_sb = 0.5 * (sb.K_bb[6:12, 6:12] + sb.K_bb[6:12, 6:12].T)
    assert np.all(np.linalg.eigvalsh(k11_sb) > 0.0)

    # 2. Gantry bracket CMS bundles across Stock Voron, LDO Monolith (AWD + 2WD), and Siboor CNC
    required_brackets = [
        "monolith_awd_rear_left",
        "monolith_awd_rear_right",
        "monolith_awd_front_left",
        "monolith_awd_front_right",
        "monolith_awd_xy_left",
        "monolith_awd_xy_right",
        "monolith_2wd_front_left",
        "monolith_2wd_front_right",
        "monolith_2wd_xy_left",
        "monolith_2wd_xy_right",
        "monolith_v24_z_joint",
        "siboor_awd_rear_left",
        "siboor_awd_rear_right",
        "siboor_awd_front_left",
        "siboor_awd_front_right",
        "siboor_awd_xy_left",
        "siboor_awd_xy_right",
        "siboor_2wd_rear_left",
        "siboor_2wd_rear_right",
        "siboor_2wd_front_left",
        "siboor_2wd_front_right",
        "siboor_2wd_xy_left",
        "siboor_2wd_xy_right",
        "voron_stock_rear_left",
        "voron_stock_rear_right",
        "voron_stock_front_left",
        "voron_stock_front_right",
        "voron_stock_xy_left",
        "voron_stock_xy_right",
        "voron_24_z_drive_left",
        "voron_24_z_drive_right",
        "voron_24_z_joint",
        "voron_trident_z_mount",
        "voron_trident_z_mount_left",
        "voron_trident_z_mount_right",
        "voron_trident_z_mount_rear",
        "voron_trident_bed_arm",
        "voron_trident_bed_arm_left",
        "voron_trident_bed_arm_right",
        "voron_trident_bed_arm_rear",
    ]
    for bid in required_brackets:
        assert has_hardware_asset("gantry_brackets", bid), f"Missing gantry_brackets/{bid}"
        se = load_superelement("gantry_brackets", bid)
        assert se is not None
        assert se.manifest["rigid_body_max_freq_hz"] < 0.05
        assert se.manifest["mass_kg"] * 1000.0 < 320.0, f"{bid} FEA mass too large (motor leaked?)"
        k11 = 0.5 * (se.K_bb[6:12, 6:12] + se.K_bb[6:12, 6:12].T)
        assert np.all(np.linalg.eigvalsh(k11) > 0.0), f"{bid} K_11 not SPD"
        assert se.visual_glb_path is not None and se.visual_glb_path.is_file()


@pytest.mark.parametrize(
    "yaml_name,expected_gantry_kit,expected_th",
    EXPECTED_NEW_PRESETS,
)
def test_voron_kit_presets_evaluate_with_cad_gantry(
    yaml_name: str,
    expected_gantry_kit: str,
    expected_th: str,
) -> None:
    """Every Voron / LDO / Siboor / Monolith preset evaluates cleanly with structural CAD gantry & toolhead CMS."""
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / yaml_name)
    assert spec.resolve_effective_gantry_hardware_kit() == expected_gantry_kit
    assert spec.toolhead.preset_id == expected_th

    bkt = resolve_gantry_bracket(spec)
    assert bkt["cad_backed"] is True
    assert bkt["cad_kit_id"] == expected_gantry_kit
    assert bkt["stiffness_n_m"] > 1e5

    rep = evaluate_printer(spec, include_advanced_analysis=False)
    assert math.isfinite(rep.modal_fea.fundamental_freq_hz) and rep.modal_fea.fundamental_freq_hz > 25.0
    assert math.isfinite(rep.static_fea.toolhead_stiffness_y_n_per_mm)
    assert rep.scores.composite > 65.0

    geom = rep.assembly.to_geometry_dict()
    gb = geom["gantry_brackets"]
    assert gb["hardware_kit"] == expected_gantry_kit
    assert gb["cad_backed"] is True
    for st_key in ("rear_left", "rear_right", "front_left", "front_right", "xy_left", "xy_right"):
        assert st_key in gb["cad_assets"], f"{yaml_name} missing gantry station {st_key}"
        assert gb["cad_assets"][st_key]["visual_glb_url"].startswith("/api/assets/hardware/gantry_brackets/")

    if expected_th == "stealthburner":
        assert geom["toolhead_structure"]["toolhead_visual_glb_url"] == (
            "/api/assets/hardware/toolheads/stealthburner/visual.glb"
        )
        assert any(se.asset_id == "stealthburner" for se in rep.assembly.superelements)

    # Model confidence reports CAD-backed gantry trust
    audit = build_model_confidence_audit(spec)
    g_entry = next(e for e in audit["entries"] if e["id"] == "gantry_bracket")
    assert g_entry["cad_superelement_active"] is True
    assert g_entry["trust_pct"] >= 85


def test_cnc_kits_stiffer_than_stock_printed_voron_gantry_in_cad_cms() -> None:
    """Monolith LDO CNC AWD and Siboor CNC AWD produce substantially stiffer CAD brackets than Stock ABS."""
    s_stock = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    s_siboor = PrinterSpec.from_yaml(EXAMPLES_DIR / "siboor_voron_trident_awd_300.yaml")
    s_mono = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_monolith_awd.yaml")

    k_stock = resolve_gantry_bracket(s_stock)["stiffness_n_m"]
    k_siboor = resolve_gantry_bracket(s_siboor)["stiffness_n_m"]
    k_mono = resolve_gantry_bracket(s_mono)["stiffness_n_m"]

    assert k_siboor > k_stock * 3.0
    assert k_mono > k_stock * 3.0


def test_api_serves_structural_cad_kits() -> None:
    """FastAPI serves /api/catalog gantry_hardware_kits and structural gantry/toolhead visual GLBs."""
    client = TestClient(create_app())
    r_cat = client.get("/api/catalog")
    assert r_cat.status_code == 200
    cat = r_cat.json()
    kit_ids = {k["id"] for k in cat["gantry_hardware_kits"]}
    assert {"voron_stock_printed", "ldo_voron_kit", "siboor_cnc_awd", "monolith_ldo_cnc_awd"} <= kit_ids

    r_sb = client.get("/api/assets/hardware/toolheads/stealthburner/visual.glb")
    assert r_sb.status_code == 200
    assert r_sb.headers["content-type"].startswith("model/gltf-binary")
    assert len(r_sb.content) > 1000

    r_mono = client.get("/api/assets/hardware/gantry_brackets/monolith_awd_rear_left/visual.glb")
    assert r_mono.status_code == 200
    assert len(r_mono.content) > 1000


def test_stealthburner_cw2_stiffness_realism_clamshell_and_tap_parity() -> None:
    """Verify StealthBurner + CW2 ABS/Tap/metal stiffness realism and stealthburner_tap CAD parity."""
    from printsim.schema import TOOLHEAD_PRESETS

    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    rep_abs = evaluate_printer(base, include_advanced_analysis=False)
    k_body_abs = rep_abs.assembly.toolhead_stage_stiffnesses_n_per_mm["body_n_per_mm"]
    k_tip_abs = rep_abs.static_fea.toolhead_flex_breakdown["effective_tip_stiffness_n_per_mm"]

    # Disjoint MGN12H->Revo shelf ABS clamshell lands near ~80 N/mm body (~51 N/mm tip),
    # matching the analytical multi-piece clamshell heuristic (~79 N/mm body / ~51 N/mm tip).
    assert 55.0 <= k_body_abs <= 110.0
    assert 40.0 <= k_tip_abs <= 65.0

    # Printed Tap on stealthburner and stealthburner_tap preset both attach the stealthburner CAD SE
    # and strictly soften both stiffness and X/Y shaper frequencies vs static_eddy.
    spec_tap_probe = base.model_copy(deep=True)
    spec_tap_probe.toolhead.z_probe_type = "printed_tap"
    rep_tap_probe = evaluate_printer(spec_tap_probe, include_advanced_analysis=False)
    assert rep_tap_probe.assembly.toolhead_stage_stiffnesses_n_per_mm["body_n_per_mm"] < k_body_abs * 0.45
    assert rep_tap_probe.modal_fea.shaper_x.effective_freq_hz < rep_abs.modal_fea.shaper_x.effective_freq_hz
    assert rep_tap_probe.modal_fea.shaper_y.effective_freq_hz < rep_abs.modal_fea.shaper_y.effective_freq_hz

    tap_preset = TOOLHEAD_PRESETS["stealthburner_tap"]
    spec_tap_preset = base.model_copy(deep=True)
    spec_tap_preset.toolhead.preset_id = "stealthburner_tap"
    spec_tap_preset.toolhead.total_mass_g = float(tap_preset["total_mass_g"])
    spec_tap_preset.toolhead.cog_offset_mm = tuple(tap_preset["cog_offset_mm"])
    spec_tap_preset.toolhead.nozzle_offset_mm = tuple(tap_preset["nozzle_offset_mm"])
    spec_tap_preset.toolhead.z_probe_type = "printed_tap"
    rep_tap_preset = evaluate_printer(spec_tap_preset, include_advanced_analysis=False)
    geom_tap = rep_tap_preset.assembly.to_geometry_dict()
    assert geom_tap["toolhead_structure"]["toolhead_visual_glb_url"] == (
        "/api/assets/hardware/toolheads/stealthburner/visual.glb"
    )
    assert any(se.asset_id == "stealthburner" for se in rep_tap_preset.assembly.superelements)
    assert rep_tap_preset.assembly.toolhead_stage_stiffnesses_n_per_mm["body_n_per_mm"] == pytest.approx(
        rep_tap_probe.assembly.toolhead_stage_stiffnesses_n_per_mm["body_n_per_mm"], rel=0.05
    )

    # Upgrading ONLY the X-carriage to CNC Al 6061 while keeping the ABS StealthBurner body
    # combines 35% X-carriage + 65% forward body compliance in series: measurably stiffer than
    # all-ABS (~1.5x) yet still bounded by the printed body, and clears polymer belt creep warnings.
    spec_cnc_car = base.model_copy(deep=True)
    spec_cnc_car.toolhead.carriage_mount_material = "aluminum_6061_t6"
    spec_cnc_car.gantry.belt.pretension_n = 65.0
    rep_cnc_car = evaluate_printer(spec_cnc_car, include_advanced_analysis=False)
    k_body_cnc_car = rep_cnc_car.assembly.toolhead_stage_stiffnesses_n_per_mm["body_n_per_mm"]
    assert rep_cnc_car.assembly.to_geometry_dict()["toolhead_structure"]["cad_split_carriage_compatible"] is True
    assert k_body_abs * 1.35 < k_body_cnc_car < k_body_abs * 1.75
    assert not any("printed mount creep warning" in d.lower() for d in rep_cnc_car.diagnostics)

    # Upgrading StealthBurner chassis material to CNC Al 6061 retains multi-piece bolted seam compliance
    # (0.58x) rather than jumping to a welded solid billet (> 10,000 N/mm).
    spec_al = base.model_copy(deep=True)
    spec_al.toolhead.body_material = "aluminum_6061_t6"
    spec_al.toolhead.carriage_mount_material = "aluminum_6061_t6"
    spec_al.toolhead.hotend_mount_material = "aluminum_6061_t6"
    rep_al = evaluate_printer(spec_al, include_advanced_analysis=False)
    k_body_al = rep_al.assembly.toolhead_stage_stiffnesses_n_per_mm["body_n_per_mm"]
    assert k_body_cnc_car * 5.0 < k_body_al < 7000.0
    assert k_body_abs * 10.0 < k_body_al < 7000.0


def test_cad_xy_joint_superelements_wired_into_3d_corexy_fea() -> None:
    """CAD gantry kits attach 12x12 XY-joint superelements at x_beam_l/r with exact rigid-body modes & adjoints."""
    from printsim.fea import (
        assemble_global_matrices,
        compute_structural_sensitivities,
        solve_modal_fea,
        solve_static_fea,
    )
    from printsim.geometry import build_assembly
    from printsim.geometry._builder_gantry import corexy_x_beam_span_m

    def _skew(v: np.ndarray) -> np.ndarray:
        return np.array(
            [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
            dtype=float,
        )

    # 1. All 3 CAD kit families attach xy_joint_l / xy_joint_r & shorten X-beam span to real clamp ports
    for yaml_name, expected_inset_mm in (
        ("voron_trident_300.yaml", 45.0),
        ("siboor_voron_trident_awd_300.yaml", 32.0),
        ("voron_monolith_awd.yaml", 45.0),
    ):
        spec = PrinterSpec.from_yaml(EXAMPLES_DIR / yaml_name)
        asm = build_assembly(spec)
        nm = asm.node_map
        assert "x_beam_l" in nm and "x_beam_r" in nm
        assert nm["x_beam_l"].position_m[0] * 1000.0 == pytest.approx(expected_inset_mm, abs=0.2)
        # Frame-width reference is the Y-extrusion axis (rail station node); on inner-face rails
        # the moving carriage node rides the raceway inboard of it (Monolith / VzBot).
        w_cl_mm = nm["y_rail_car_r"].position_m[0] * 1000.0
        if spec.gantry.y_rail_mount == "inner":
            assert nm["y_car_l"].position_m[0] * 1000.0 == pytest.approx(14.8, abs=0.05)
            assert w_cl_mm - nm["y_car_r"].position_m[0] * 1000.0 == pytest.approx(14.8, abs=0.05)
        else:
            assert nm["y_car_r"].position_m[0] == pytest.approx(nm["y_rail_car_r"].position_m[0])
        assert (w_cl_mm - nm["x_beam_r"].position_m[0] * 1000.0) == pytest.approx(expected_inset_mm, abs=0.2)

        xg_l = next(m for m in asm.members if m.id == "x_gantry_l")
        xg_r = next(m for m in asm.members if m.id == "x_gantry_r")
        assert xg_l.start_id == "x_beam_l" and xg_l.end_id == "toolhead"
        assert xg_r.start_id == "toolhead" and xg_r.end_id == "x_beam_r"

        se_by_id = {se.id: se for se in asm.superelements}
        assert {"xy_joint_l", "xy_joint_r"} <= set(se_by_id.keys())
        assert asm.to_geometry_dict()["gantry_brackets"]["cad_xy_fea_active"] is True

        # Exact 6 rigid-body zero modes at (y_car_*, x_beam_*)
        for se_id in ("xy_joint_l", "xy_joint_r"):
            se = se_by_id[se_id]
            p0 = nm[se.port_node_ids[0]].position_m
            p1 = nm[se.port_node_ids[1]].position_m
            dp = p1 - p0
            K12 = np.asarray(se.K_cb, dtype=float)
            assert K12.shape == (12, 12)
            R = np.zeros((12, 6), dtype=float)
            R[0:3, 0:3] = np.eye(3)
            R[3:6, 3:6] = np.eye(3)
            R[6:9, 0:3] = np.eye(3)
            R[6:9, 3:6] = -_skew(dp)
            R[9:12, 3:6] = np.eye(3)
            rb_err = float(np.linalg.norm(K12 @ R) / max(np.linalg.norm(K12), 1.0))
            assert rb_err < 1e-10, f"{yaml_name} {se_id} rigid-body residual too large: {rb_err}"

        # Exact physical X-beam mass conservation (density-scaled to corexy_x_beam_span_m)
        spec_no_cad = spec.model_copy(deep=True)
        spec_no_cad.use_cad_superelements = False
        asm_no_cad = build_assembly(spec_no_cad)
        assert "x_beam_l" not in asm_no_cad.node_map
        assert asm_no_cad.to_geometry_dict()["gantry_brackets"]["cad_xy_fea_active"] is False
        assert asm.x_beam_mass_kg == pytest.approx(asm_no_cad.x_beam_mass_kg, rel=1e-6)
        assert asm.y_moving_mass_kg == pytest.approx(asm_no_cad.y_moving_mass_kg, rel=1e-6)
        assert corexy_x_beam_span_m(spec) > 0.3

    # 2. Explicit procedural kit wires analytical Timoshenko + Huth/VDI bolted-seam XY-joints
    spec_proc = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_fixed_gantry_300.yaml")
    spec_proc.gantry.hardware_kit = "procedural"
    asm_proc = build_assembly(spec_proc)
    assert "x_beam_l" in asm_proc.node_map and "x_beam_r" in asm_proc.node_map
    g_bkt = asm_proc.to_geometry_dict()["gantry_brackets"]
    assert g_bkt["cad_xy_fea_active"] is False
    assert g_bkt["xy_joint_fea_active"] is True
    assert g_bkt["xy_joint_model"] == "analytical_bolted"

    # 3. Monotonic static stiffness gain upgrading XY-joint bracket material (ABS -> PA12-CF -> Al 6061)
    trident = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    ky_vals = []
    for mat_id in ("abs_printed", "pa12_cf_printed", "aluminum_6061_t6"):
        s_m = trident.model_copy(deep=True)
        s_m.gantry.bracket_material = mat_id  # type: ignore[assignment]
        ky_vals.append(solve_static_fea(build_assembly(s_m)).toolhead_stiffness_y_n_per_mm)
    assert ky_vals[0] < ky_vals[1] < ky_vals[2]

    # 4. Exact adjoint static compliance derivative on xy_joint_l matches central finite difference
    asm_t = build_assembly(trident)
    st_t = solve_static_fea(asm_t)
    mo_t = solve_modal_fea(asm_t)
    sen_t = compute_structural_sensitivities(asm_t, static_res=st_t, modal_res=mo_t)
    assert "xy_joint_l" in sen_t.member_sensitivities
    assert sen_t.member_sensitivities["xy_joint_l"]["compliance_share_pct"] > 2.0

    eps = 1e-3
    dln = math.log(1.0 + eps) - math.log(1.0 - eps)
    uy_vals = []
    for sc in (1.0 + eps, 1.0 - eps):
        asm_p = build_assembly(trident)
        for se in asm_p.superelements:
            if se.id == "xy_joint_l":
                se.K_cb = se.K_cb * sc
        uy_vals.append(compute_structural_sensitivities(asm_p).static_rel_disp_y_um * 1e-6)
    fd_duy = (uy_vals[0] - uy_vals[1]) / dln
    assert sen_t.member_sensitivities["xy_joint_l"]["du_rel_y_dlnE_m"] == pytest.approx(fd_duy, rel=2e-2)


def test_siboor_awd_base_kit_settings_and_monolith_upgrade_sync() -> None:
    """Base kits configure full gantry/rail/Z/top-space specs, and gantry upgrades sync coupled settings."""
    from printsim.schema import apply_gantry_hardware_kit

    # 1. Siboor Voron Trident AWD base kit carries full authentic mechanical & Z/top-clearance specs
    siboor = PrinterSpec.from_yaml(EXAMPLES_DIR / "siboor_voron_trident_awd_300.yaml")
    assert siboor.gantry.hardware_kit == "siboor_cnc_awd"
    assert siboor.gantry.drive_mode == "awd"
    assert siboor.gantry.belt_routing == "stock_corexy"
    assert siboor.gantry.double_shear_pulleys is True
    assert siboor.gantry.bracket_material == "aluminum_6061_t6"
    assert siboor.gantry.x_rail_profile == "mgn12_rail"
    assert siboor.gantry.x_rail_mount == "front"
    assert siboor.gantry.y_rail_profile == "mgn9_rail"
    assert siboor.gantry.y_rail_mount == "bottom"
    assert siboor.gantry.rail_block_type == "H"
    assert siboor.gantry.belt.type == "gt2_9mm"
    assert siboor.bed.z_drive == "leadscrew_triple"
    assert siboor.bed.z_rail_profile == "mgn9_rail"
    assert siboor.bed.z_leadscrew_type == "tr8x4_pom"
    assert siboor.bed.mount_type == "trident_3pt_standoffs"
    assert siboor.frame.gantry_top_clearance_mm == pytest.approx(110.0)

    # 2. Upgrading Siboor AWD -> Monolith x LDO CNC AWD switches routing, rail orientation/size,
    #    belt tension, carriage mass, and M5 disc-spring axial preload while preserving Trident Z/frame
    changes = apply_gantry_hardware_kit(siboor, "monolith_ldo_cnc_awd")
    assert len(changes) >= 3
    assert siboor.gantry.hardware_kit == "monolith_ldo_cnc_awd"
    assert siboor.gantry.drive_mode == "awd"
    assert siboor.gantry.belt_routing == "monolith_short"
    assert siboor.gantry.bracket_material == "aluminum_6061_t6"
    assert siboor.gantry.double_shear_pulleys is True
    assert siboor.gantry.y_rail_mount == "inner"
    assert siboor.gantry.y_rail_profile == "mgn12_rail"
    assert siboor.gantry.x_rail_mount == "front"
    assert siboor.gantry.belt.type == "gt2_9mm"
    assert siboor.gantry.belt.pretension_n == pytest.approx(65.0)
    assert siboor.gantry.carriage_mass_g == pytest.approx(214.0)
    assert siboor.gantry.idler_axial_preload == "disc_spring_m5"
    # Base machine Z & top clearance stay intact across gantry upgrade
    assert siboor.bed.z_drive == "leadscrew_triple"
    assert siboor.frame.gantry_top_clearance_mm == pytest.approx(110.0)
    rep_up = evaluate_printer(siboor, include_advanced_analysis=False)
    assert rep_up.modal_fea.shaper_x.effective_freq_hz > 100.0
    assert rep_up.assembly.to_geometry_dict()["gantry_brackets"]["hardware_kit"] == "monolith_ldo_cnc_awd"

    # 3. Upgrading an LDO Voron 2.4r2 Rev D (2WD ABS 6mm flying gantry) -> Monolith x LDO CNC AWD
    #    upgrades drive mode, bracket material, double shear, belt routing/width/tension, inner Y rails,
    #    AND switches flying-gantry V2.4 Z-joints to milled
    ldo_v24 = PrinterSpec.from_yaml(EXAMPLES_DIR / "ldo_voron_24_revd_300.yaml")
    assert ldo_v24.gantry.hardware_kit == "ldo_voron_kit"
    assert ldo_v24.gantry.drive_mode == "2wd"
    assert ldo_v24.gantry.bracket_material == "abs_printed"
    assert ldo_v24.frame.gantry_top_clearance_mm == pytest.approx(55.0)

    v24_changes = ldo_v24.apply_gantry_hardware_kit("monolith_ldo_cnc_awd")
    assert len(v24_changes) >= 5
    assert ldo_v24.gantry.drive_mode == "awd"
    assert ldo_v24.gantry.belt_routing == "monolith_short"
    assert ldo_v24.gantry.bracket_material == "aluminum_6061_t6"
    assert ldo_v24.gantry.double_shear_pulleys is True
    assert ldo_v24.gantry.belt.type == "gt2_9mm"
    assert ldo_v24.gantry.belt.pretension_n == pytest.approx(65.0)
    assert ldo_v24.gantry.y_rail_mount == "inner"
    assert ldo_v24.gantry.z_joint_material == "milled"

    # 4. /api/catalog exposes kit_settings on concrete gantry_hardware_kits
    client = TestClient(create_app())
    cat = client.get("/api/catalog").json()
    kit_by_id = {k["id"]: k for k in cat["gantry_hardware_kits"]}
    assert kit_by_id["monolith_ldo_cnc_awd"]["kit_settings"]["belt_routing"] == "monolith_short"
    assert kit_by_id["monolith_ldo_cnc_awd"]["kit_settings"]["y_rail_mount"] == "inner"
    assert kit_by_id["siboor_cnc_awd"]["kit_settings"]["y_rail_mount"] == "bottom"
    assert kit_by_id["siboor_cnc_awd"]["kit_settings"]["belt_type"] == "gt2_9mm"


def test_full_gantry_3d_fea_y_carriages_corner_mounts_z_joints_and_trident_arms() -> None:
    """Verify full-gantry 3D FEA wiring: Y-rail bearings, CAD corner mounts, V2.4 Z-joints & Trident arms."""
    import numpy as np

    from printsim.fea.modal import solve_modal_fea
    from printsim.fea.solver import solve_static_fea
    from printsim.geometry.builder import build_assembly

    # 1. Monolith AWD: Y-rail bearings (6-DOF FastenerSpring) + XY-joints + 4 corner mounts
    mono = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_monolith_awd.yaml")
    mono_asm = build_assembly(mono)
    sp_by_id = {sp.id: sp for sp in mono_asm.fastener_springs}
    for sp_id in ("y_car_bearing_l", "y_car_bearing_r"):
        assert sp_id in sp_by_id, f"Missing 6-DOF Hertzian Y-carriage spring {sp_id}"
        assert sp_by_id[sp_id].k_diag_6dof is not None
        assert len(sp_by_id[sp_id].k_diag_6dof) == 6
    se_by_id = {se.id: se for se in mono_asm.superelements}
    for req_id in (
        "xy_joint_l",
        "xy_joint_r",
        "corner_mount_fl",
        "corner_mount_fr",
        "corner_mount_rl",
        "corner_mount_rr",
    ):
        assert req_id in se_by_id, f"Missing 3D FEA superelement {req_id} on Monolith AWD"
        # Every 12x12 superelement must be symmetric with 6 exact rigid-body zero modes
        K_se = se_by_id[req_id].K_cb[:12, :12]
        assert np.allclose(K_se, K_se.T, atol=1e-6)
        eigs = np.sort(np.linalg.eigvalsh(K_se))
        assert abs(eigs[5]) <= max(abs(eigs[-1]) * 1e-8, 1e-4)
        assert eigs[6] > 1.0

    # Y-rail extrusions terminate at y_rail_car_{l,r}, decoupled through y_car_bearing_{l,r} to y_car_{l,r}
    m_by_id = {m.id: m for m in mono_asm.members}
    assert m_by_id["y_rail_l_f"].end_id == "y_rail_car_l"
    assert m_by_id["y_rail_l_r"].start_id == "y_rail_car_l"
    assert (sp_by_id["y_car_bearing_l"].shell_node_id, sp_by_id["y_car_bearing_l"].frame_node_id) == (
        "y_rail_car_l",
        "y_car_l",
    )

    # 2. Corner motor mounts physically bend under CoreXY belt tension (ABS Stock > 5x more than CNC 6061)
    stock = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    stock.gantry.belt.pretension_n = 55.0
    stock_asm = build_assembly(stock)
    stock_st = solve_static_fea(stock_asm)

    cnc = stock.model_copy(deep=True)
    cnc.gantry.hardware_kit = "monolith_ldo_cnc_awd"
    cnc.gantry.bracket_material = "aluminum_6061_t6"
    cnc_asm = build_assembly(cnc)
    cnc_st = solve_static_fea(cnc_asm)

    def _pulley_rel_disp_um(st_res, tag: str) -> float:
        p_d = np.array(st_res.nodal_displacements_mm[f"pulley_{tag}"], dtype=float)
        g_d = np.array(st_res.nodal_displacements_mm[f"g_{tag}"], dtype=float)
        return float(np.linalg.norm(p_d - g_d)) * 1e3

    abs_rl_bend = _pulley_rel_disp_um(stock_st, "rl")
    cnc_rl_bend = _pulley_rel_disp_um(cnc_st, "rl")
    assert abs_rl_bend > 10.0, f"Expected real ABS motor-mount bend under belt tension, got {abs_rl_bend:.2f} um"
    assert abs_rl_bend > cnc_rl_bend * 5.0, (
        f"ABS motor mount ({abs_rl_bend:.2f} um) should bend >5x more than CNC 6061 ({cnc_rl_bend:.2f} um)"
    )

    # 3. Voron Trident CAD bed arms wired between z_bed_* and z_arm_tip_* in series with bed_arm_*
    stock_se = {se.id: se for se in stock_asm.superelements}
    for tag in ("fl", "fr", "r"):
        assert f"trident_bed_arm_{tag}" in stock_se
        assert f"z_arm_tip_{tag}" in stock_asm.node_map
        assert stock_se[f"trident_bed_arm_{tag}"].port_node_ids == (f"z_bed_{tag}", f"z_arm_tip_{tag}")

    # 4. Voron 2.4 flying gantry CAD Z-joints wired between col_g_* -> z_joint_mid_* -> z_joint_* -> g_*
    v24 = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    v24_asm = build_assembly(v24)
    v24_se = {se.id: se for se in v24_asm.superelements}
    for tag in ("fl", "fr", "rl", "rr"):
        assert f"cad_z_joint_{tag}" in v24_se
        assert v24_se[f"cad_z_joint_{tag}"].port_node_ids == (f"col_g_{tag}", f"z_joint_mid_{tag}")

    # Modal exports 6-DOF rotations for new pulley_*, y_rail_car_*, and z_arm_tip_* nodes
    stock_mo = solve_modal_fea(stock_asm)
    rot_keys = set(stock_mo.modes[0].nodal_rotations.keys())
    for k in ("y_rail_car_l", "y_car_l", "x_beam_l", "pulley_rl", "z_arm_tip_fl"):
        assert k in rot_keys, f"Missing modal rotation export for {k}"


def test_full_xy_joint_and_y_carriage_mechanics() -> None:
    """Verify all 5 XY-joint & Y-carriage FEA mechanics upgrades:
    1. Analytical Timoshenko + Huth/VDI bolted-seam XY-joints on procedural builds
    2. Explicit 4-bolt Y-carriage pad & X-beam clamp bolted-seam compliances
    3. Full coupled 6x6 CAD XY-joint compliance (off-plane cross-coupling preserved)
    4. Y-rail raceway-to-carriage lever-arm offset on Hertzian bearings (exact RB zero modes)
    5. Self-equilibrating eccentric stacked-belt & tension-mismatch moment loads on XY-joints
    """
    from printsim.fea import solve_static_fea
    from printsim.fea.components import (
        xy_joint_bolted_seam_compliances,
        y_carriage_lever_arm_m,
    )
    from printsim.fea.shell_mitc4 import fastener_spring_matrices
    from printsim.geometry.builder import build_assembly

    # 1. Procedural CoreXY gets analytical bolted XY-joint superelements with exact RB zero modes & monotonic material gain
    proc = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_fixed_gantry_300.yaml")
    proc.gantry.hardware_kit = "procedural"
    ky_proc = []
    for mat_id in ("abs_printed", "pa12_cf_printed", "aluminum_6061_t6"):
        sp = proc.model_copy(deep=True)
        sp.gantry.bracket_material = mat_id  # type: ignore[assignment]
        asm = build_assembly(sp)
        se_l = next(se for se in asm.superelements if se.id == "xy_joint_l")
        assert se_l.asset_id == "procedural_xy_joint"
        eigs = np.sort(np.linalg.eigvalsh(se_l.K_cb))
        assert abs(eigs[5]) <= max(abs(eigs[-1]) * 1e-8, 1e-4)
        assert eigs[6] > 1.0
        ky_proc.append(solve_static_fea(asm).toolhead_stiffness_y_n_per_mm)
    assert ky_proc[0] < ky_proc[1] < ky_proc[2]

    # 2. Explicit Huth/VDI bolted-seam compliances: H-block stiffer than C-block; Al 6061 stiffer than ABS
    c0_abs_c, c1_abs = xy_joint_bolted_seam_compliances(
        "mgn12_rail", "C", "bottom", "abs_printed", "extrusion_2020", "aluminum_6061_t6", 25.0
    )
    c0_abs_h, _ = xy_joint_bolted_seam_compliances(
        "mgn12_rail", "H", "bottom", "abs_printed", "extrusion_2020", "aluminum_6061_t6", 25.0
    )
    c0_al_h, c1_al = xy_joint_bolted_seam_compliances(
        "mgn12_rail", "H", "bottom", "aluminum_6061_t6", "extrusion_2020", "aluminum_6061_t6", 25.0
    )
    assert c0_abs_h[3, 3] < c0_abs_c[3, 3]  # Longer H-block bolt span stiffer in pitch
    assert c0_al_h[0, 0] < c0_abs_h[0, 0] and c1_al[3, 3] < c1_abs[3, 3]

    # 3. Full coupled 6x6 CAD XY-joint preserves off-plane cross-coupling (non-zero Y<->Z / Y<->Rx blocks)
    trident = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    asm_t = build_assembly(trident)
    se_xy = next(se for se in asm_t.superelements if se.id == "xy_joint_l")
    K11 = se_xy.K_cb[6:12, 6:12]
    assert abs(float(K11[1, 2])) > 1e2 or abs(float(K11[1, 3])) > 1e1

    # 4. Y-carriage bearing carries physical raceway-to-bolt-face lever arm with exact RB zero modes
    sp_l = next(sp for sp in asm_t.fastener_springs if sp.id == "y_car_bearing_l")
    assert sp_l.offset_m == y_carriage_lever_arm_m(
        trident.gantry.y_rail_profile, trident.gantry.y_rail_mount, "l"
    )
    assert float(np.linalg.norm(sp_l.offset_m)) > 0.004
    K_sp_off = fastener_spring_matrices(
        sp_l.k_shear_n_m,
        sp_l.k_normal_n_m,
        sp_l.k_rot_nm_rad,
        np.array(sp_l.normal_vec),
        offset_vec=np.array(sp_l.offset_m),
        k_diag_6dof=sp_l.k_diag_6dof,
    )
    K_sp_zero = fastener_spring_matrices(
        sp_l.k_shear_n_m,
        sp_l.k_normal_n_m,
        sp_l.k_rot_nm_rad,
        np.array(sp_l.normal_vec),
        offset_vec=np.zeros(3),
        k_diag_6dof=sp_l.k_diag_6dof,
    )
    # Co-located rigid-body modes [I_6; I_6] must have zero strain energy
    R_coloc = np.vstack([np.eye(6), np.eye(6)])
    assert float(np.linalg.norm(K_sp_off @ R_coloc)) < 1e-6 * float(np.linalg.norm(K_sp_off))
    # Lever arm induces off-diagonal translation<->rotation coupling
    assert float(np.linalg.norm(K_sp_off[:3, 3:6])) > 1.0
    assert float(np.linalg.norm(K_sp_zero[:3, 3:6])) == 0.0

    # 5. Eccentric stacked-belt & tension-mismatch moments increase XY-joint static deflection monotonically
    s_eq = trident.model_copy(deep=True)
    s_eq.gantry.belt.pretension_n = 70.0
    s_eq.gantry.belt.tension_mismatch_pct = 0.0
    s_mis = s_eq.model_copy(deep=True)
    s_mis.gantry.belt.tension_mismatch_pct = 25.0
    st_eq = solve_static_fea(build_assembly(s_eq))
    st_mis = solve_static_fea(build_assembly(s_mis))
    d_eq = np.linalg.norm(
        np.array(st_eq.nodal_displacements_mm["x_beam_l"]) - np.array(st_eq.nodal_displacements_mm["y_car_l"])
    )
    d_mis = np.linalg.norm(
        np.array(st_mis.nodal_displacements_mm["x_beam_l"]) - np.array(st_mis.nodal_displacements_mm["y_car_l"])
    )
    assert d_mis > d_eq


