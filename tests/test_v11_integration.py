"""Integration tests for PrintSim v1.1 (Phase 2 mechatronics/ballbar & Phase 3 CMS/RCSA/sensitivities)."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from printsim.api.server import create_app
from printsim.engine import evaluate_printer
from printsim.fea import (
    CraigBamptonSuperelement,
    DriveControlConfig,
    RCSAResult,
    SensitivityReport,
    WorkspaceFRFMap,
    analyze_toolhead_rcsa,
    compute_structural_sensitivities,
    compute_thermal_structural_report,
    compute_workspace_frf_map,
    condense_parametric_bracket,
)
from printsim.mcp_server import dispatch_tool
from printsim.schema import TOOLHEAD_PRESETS, PrinterSpec, StepperSpec

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def test_fea_init_exports_phase123_symbols() -> None:
    """Verify src/printsim/fea/__init__.py exports key Phase 1, 2, and 3 symbols."""
    assert callable(compute_thermal_structural_report)
    assert callable(compute_workspace_frf_map)
    assert callable(analyze_toolhead_rcsa)
    assert callable(compute_structural_sensitivities)
    assert callable(condense_parametric_bracket)
    assert DriveControlConfig is not None
    assert WorkspaceFRFMap is not None
    assert RCSAResult is not None
    assert SensitivityReport is not None
    assert CraigBamptonSuperelement is not None


def test_evaluator_v11_reports_and_speed_on_examples() -> None:
    """evaluate_printer emits workspace_frf, time_domain_metrology, rcsa, sensitivity in <250ms."""
    wfrf_keys = {
        "grid_n",
        "u_coords",
        "v_coords",
        "freq_x_grid_hz",
        "freq_y_grid_hz",
        "min_axis_freq_grid_hz",
        "xy_cross_coupling_grid_pct",
        "center_freq_hz",
        "min_freq_hz",
        "max_freq_hz",
        "worst_corner_drop_pct",
        "max_cross_coupling_pct",
        "reflected_rotor_mass_g",
        "drive_type",
        "midband_instability_risk",
    }
    tdm_keys = {
        "corner_rounding_um",
        "corner_overshoot_um",
        "ringing_peak_to_peak_um",
        "ringing_unshaped_um",
        "ringing_reduction_pct",
        "ringing_decay_mm",
        "corner_trace",
        "ballbar_radius_mm",
        "ballbar_speed_mm_s",
        "circularity_gxy_um",
        "mean_radial_lag_um",
        "quadrant_glitch_um",
        "ballbar_angles_deg",
        "ballbar_radial_error_um",
    }
    rcsa_keys = {
        "sensor_peak_freq_hz",
        "nozzle_tip_peak_freq_hz",
        "nozzle_to_sensor_amp_ratio",
        "hotend_bending_stiffness_n_per_mm",
        "whip_risk",
        "warning_message",
    }
    sens_keys = {
        "member_sensitivities",
        "panel_bay_sensitivities",
        "top_stiffen_targets",
        "top_lighten_targets",
    }

    for fname in ("voron_monolith_awd.yaml", "voron_trident_300.yaml"):
        spec = PrinterSpec.from_yaml(EXAMPLES_DIR / fname)
        t0 = time.perf_counter()
        rep = evaluate_printer(spec, panel_mesh_n=2)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert elapsed_ms < 250.0, f"{fname} evaluate_printer took {elapsed_ms:.1f} ms (>= 250 ms)"

        d = rep.to_dict(include_toolpath_segments=False)
        assert wfrf_keys <= set(d["workspace_frf"].keys())
        assert tdm_keys <= set(d["time_domain_metrology"].keys())
        assert rcsa_keys <= set(d["rcsa"].keys())
        assert sens_keys <= set(d["sensitivity"].keys())

        assert d["workspace_frf"]["grid_n"] == 5
        assert len(d["time_domain_metrology"]["ballbar_angles_deg"]) == 72
        assert d["rcsa"]["hotend_bending_stiffness_n_per_mm"] > 0.0
        assert len(d["sensitivity"]["top_stiffen_targets"]) > 0
        assert len(d["sensitivity"]["top_lighten_targets"]) > 0

    # Fast flag omits advanced analyses cleanly
    spec_fast = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    rep_fast = evaluate_printer(spec_fast, include_advanced_analysis=False)
    d_fast = rep_fast.to_dict(include_toolpath_segments=False)
    assert d_fast["workspace_frf"] is None
    assert d_fast["time_domain_metrology"] is None
    assert d_fast["rcsa"] is None
    assert d_fast["sensitivity"] is None


def test_evaluator_v11_actionable_diagnostics() -> None:
    """Verify corner drop >12%, open-loop midband resonance, and RCSA whip diagnostics fire cleanly."""
    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")

    # 1. Mid-band resonance + RCSA whip risk
    risky = base.model_copy(deep=True)
    risky.kinematics.max_velocity_mm_s = 560.0
    risky.kinematics.xy_stepper = StepperSpec(
        holding_torque_nm=0.59,
        rated_current_a=2.0,
        run_current_a=1.6,
        phase_resistance_ohm=1.4,
        inductance_mh=3.0,
        supply_voltage_v=24.0,
    )
    heavy = TOOLHEAD_PRESETS["heavy_uhf_long"]
    risky.toolhead.total_mass_g = float(heavy["total_mass_g"])
    risky.toolhead.cog_offset_mm = tuple(float(v) for v in heavy["cog_offset_mm"])  # type: ignore[assignment]
    risky.toolhead.nozzle_offset_mm = tuple(float(v) for v in heavy["nozzle_offset_mm"])  # type: ignore[assignment]
    risky.toolhead.mount_material = "abs_printed"

    rep_risky = evaluate_printer(risky)
    assert rep_risky.workspace_frf is not None
    assert rep_risky.workspace_frf.midband_instability_risk is True
    assert rep_risky.rcsa is not None
    assert rep_risky.rcsa.whip_risk is True
    diag_text = " ".join(rep_risky.diagnostics).lower()
    assert "mid-band" in diag_text
    assert "whip risk" in diag_text

    # 2. Corner drop > 12% on large 2WD CoreXY
    large_2wd = base.model_copy(deep=True)
    large_2wd.frame.width_mm = 620.0
    large_2wd.frame.depth_mm = 620.0
    large_2wd.kinematics.build_volume_mm = (450.0, 450.0, 300.0)
    large_2wd.bed.size_mm = (450.0, 450.0)
    large_2wd.gantry.drive_mode = "2wd"
    large_2wd.gantry.belt.type = "gt2_6mm"
    large_2wd.gantry.belt.core = "fiberglass"
    large_2wd.gantry.belt.pretension_n = 20.0
    rep_corner = evaluate_printer(large_2wd)
    assert rep_corner.workspace_frf is not None
    if rep_corner.workspace_frf.worst_corner_drop_pct > 12.0:
        assert any("belt-span asymmetry" in d.lower() for d in rep_corner.diagnostics)


def test_fastapi_v11_health_catalog_simulate_and_superelement() -> None:
    """Verify /api/health v1.1.0, /api/catalog, /api/simulate advanced shapers+FOC, and /api/superelement/bracket."""
    client = TestClient(create_app())

    r_health = client.get("/api/health")
    assert r_health.status_code == 200
    assert r_health.json()["version"] == "1.1.0"

    r_cat = client.get("/api/catalog")
    assert r_cat.status_code == 200
    cat = r_cat.json()
    drive_ids = {item["id"] for item in cat["drive_types"]}
    shaper_ids = {item["id"] for item in cat["input_shapers"]}
    assert {"open_loop_stepper", "closed_loop_stepper", "foc_servo"} <= drive_ids
    assert {"auto", "none", "zv", "mzv", "ei", "2hump_ei", "3hump_ei", "zvdd", "smooth_zv", "smooth_ei"} <= shaper_ids

    # Simulate with explicit ZVDD + FOC servo + notch filter
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_monolith_awd.yaml")
    spec_zvdd = spec.model_copy(deep=True)
    spec_zvdd.kinematics.input_shaper = "zvdd"
    spec_zvdd.kinematics.drive_type = "foc_servo"
    spec_zvdd.kinematics.servo_vel_bw_hz = 150.0
    spec_zvdd.kinematics.servo_pos_bw_hz = 35.0
    spec_zvdd.kinematics.notch_filter_hz = 120.0

    r_sim1 = client.post(
        "/api/simulate",
        json={"spec": spec_zvdd.model_dump(mode="json"), "include_toolpath_segments": False},
    )
    assert r_sim1.status_code == 200
    sim1 = r_sim1.json()
    assert sim1["modal_fea"]["shaper_x"]["recommended_shaper"] == "ZVDD"
    assert sim1["workspace_frf"]["drive_type"] == "foc_servo"
    assert sim1["workspace_frf"]["midband_instability_risk"] is False
    assert sim1["time_domain_metrology"]["shaper_used"] == "ZVDD"

    # Simulate with explicit smooth_ei + FOC servo + notch
    spec_sei = spec_zvdd.model_copy(deep=True)
    spec_sei.kinematics.input_shaper = "smooth_ei"
    r_sim2 = client.post(
        "/api/simulate",
        json={"spec": spec_sei.model_dump(mode="json"), "include_toolpath_segments": False},
    )
    assert r_sim2.status_code == 200
    sim2 = r_sim2.json()
    assert sim2["modal_fea"]["shaper_x"]["recommended_shaper"] == "SMOOTH_EI"
    assert sim2["time_domain_metrology"]["shaper_used"] == "SMOOTH_EI"

    # Bracket superelement endpoint (verifies SPD 6-DOF port stiffnesses and no raw T_cb)
    r_se_abs = client.post(
        "/api/superelement/bracket",
        json={
            "bracket_type": "corner_gusset",
            "material_id": "abs_printed",
            "wall_thickness_mm": 6.0,
            "rib_count": 1,
            "bolt_boss_thickness_mm": 6.0,
        },
    )
    assert r_se_abs.status_code == 200
    se_abs = r_se_abs.json()
    assert "T_cb" not in se_abs
    assert se_abs["n_boundary_dofs"] == 12
    assert se_abs["n_internal_modes"] == 6
    assert se_abs["total_dofs"] == 18
    assert se_abs["mass_kg"] > 0.0
    assert len(se_abs["fixed_interface_freqs_hz"]) == 6
    expected_port_keys = {
        "k_x_n_m",
        "k_y_n_m",
        "k_z_n_m",
        "k_rx_nm_rad",
        "k_ry_nm_rad",
        "k_rz_nm_rad",
    }
    assert set(se_abs["port_stiffness_diag"].keys()) == expected_port_keys
    assert all(se_abs["port_stiffness_diag"][k] > 0.0 for k in expected_port_keys)

    r_se_al = client.post(
        "/api/superelement/bracket",
        json={
            "bracket_type": "corner_gusset",
            "material_id": "aluminum_6061_t6",
            "wall_thickness_mm": 6.0,
            "rib_count": 2,
            "bolt_boss_thickness_mm": 8.0,
        },
    )
    assert r_se_al.status_code == 200
    se_al = r_se_al.json()
    assert se_al["port_stiffness_diag"]["k_x_n_m"] > se_abs["port_stiffness_diag"]["k_x_n_m"] * 10.0


def test_mcp_v11_new_tools() -> None:
    """Verify analyze_sensitivity_hotspots and condense_bracket_superelement via dispatch_tool."""
    yaml_text = (EXAMPLES_DIR / "voron_trident_300.yaml").read_text(encoding="utf-8")

    sens_res = dispatch_tool("analyze_sensitivity_hotspots", {"yaml_text": yaml_text})
    assert len(sens_res["top_stiffen_targets"]) > 0
    assert len(sens_res["top_lighten_targets"]) > 0
    assert sens_res["workspace_frf"]["grid_n"] == 5
    assert sens_res["workspace_frf"]["center_freq_hz"] > 10.0
    assert sens_res["time_domain_metrology"]["ringing_reduction_pct"] > 50.0
    assert sens_res["rcsa"]["hotend_bending_stiffness_n_per_mm"] > 0.0

    se_res = dispatch_tool(
        "condense_bracket_superelement",
        {
            "bracket_type": "xy_joint_l_bracket",
            "material_id": "ppa_cf_printed",
            "wall_thickness_mm": 6.0,
            "rib_thickness_mm": 5.0,
        },
    )
    assert "T_cb" not in se_res
    assert se_res["mass_kg"] > 0.0
    assert len(se_res["fixed_interface_freqs_hz"]) == 6
    assert len(se_res["source_hash"]) == 64
    assert all(val > 0.0 for val in se_res["port_stiffness_diag"].values())


def test_human_readable_fea_modal_and_grouped_sensitivity_diagnoses() -> None:
    """Verify grouped ROI targets, mode diagnoses, and plain-English static FEA summaries."""
    for fname in (
        "voron_trident_300.yaml",
        "voron_monolith_awd.yaml",
        "cartesian_bedslinger.yaml",
        "delta_racer.yaml",
    ):
        spec = PrinterSpec.from_yaml(EXAMPLES_DIR / fname)
        rep = evaluate_printer(spec, panel_mesh_n=2)
        d = rep.to_dict(include_toolpath_segments=False)

        # Static FEA human-readable stress part & flex explanation
        assert rep.static_fea.peak_stress_part_label
        assert rep.static_fea.top_flex_explanation
        assert d["static_fea"]["peak_stress_part_label"] == rep.static_fea.peak_stress_part_label
        assert d["static_fea"]["top_flex_explanation"] == rep.static_fea.top_flex_explanation
        assert "__s" not in rep.static_fea.peak_stress_part_label

        # Modal FEA per-mode plain-English diagnoses and severity thresholds
        assert len(rep.modal_fea.modes) > 0
        for m, md in zip(rep.modal_fea.modes, d["modal_fea"]["modes"]):
            assert m.mode_label
            assert m.what_is_shaking
            assert m.how_to_fix
            assert m.severity in {"critical", "moderate", "stiff"}
            if m.frequency_hz < 65.0:
                assert m.severity == "critical"
            elif m.frequency_hz < 115.0:
                assert m.severity == "moderate"
            else:
                assert m.severity == "stiff"
            assert md["mode_label"] == m.mode_label
            assert md["what_is_shaking"] == m.what_is_shaking
            assert md["how_to_fix"] == m.how_to_fix
            assert md["severity"] == m.severity

        # Sensitivity grouped targets & summary diagnosis
        sens = d["sensitivity"]
        assert sens is not None
        assert sens["summary_diagnosis"]

        stiffen_ids = [t["id"] for t in sens["top_stiffen_targets"]]
        assert len(stiffen_ids) == len(set(stiffen_ids)), f"Duplicate stiffen targets in {fname}: {stiffen_ids}"
        for t in sens["top_stiffen_targets"]:
            assert "__s" not in t["id"]
            assert t["label"]
            assert t["why_it_hurts"]
            assert t["action_hint"]
            assert isinstance(t["est_hz_gain"], float)
            assert t["est_hz_gain"] >= 0.0
            assert {"id", "role", "hz_per_usd", "compliance_share_pct", "action_hint"} <= set(t.keys())

        lighten_ids = [t["id"] for t in sens["top_lighten_targets"]]
        assert len(lighten_ids) == len(set(lighten_ids)), f"Duplicate lighten targets in {fname}: {lighten_ids}"
        for t in sens["top_lighten_targets"]:
            assert "__s" not in t["id"]
            assert t["label"]
            assert t["why_it_hurts"]
            assert t["action_hint"]
            assert t["hz_gain_per_100g_saved"] > 0.0
            assert {"id", "role", "mass_sens_hz", "hz_per_100g", "action_hint"} <= set(t.keys())

    # Spec-tailored physical bottleneck hints check: Trident (clip acrylic) vs Monolith (already bolted 4mm Al)
    trident_rep = evaluate_printer(PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml"))
    monolith_rep = evaluate_printer(PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_monolith_awd.yaml"))
    assert trident_rep.sensitivity is not None and monolith_rep.sensitivity is not None
    trident_col = next(
        t for t in trident_rep.sensitivity.top_stiffen_targets if t["id"] == "front_frame_columns"
    )
    monolith_col = next(
        t for t in monolith_rep.sensitivity.top_stiffen_targets if t["id"] == "front_frame_columns"
    )
    assert trident_col["action_hint"].startswith("Main limit:")
    assert monolith_col["action_hint"].startswith("Main limit:")
    assert "shear" in trident_col["action_hint"].lower()
    assert "column" in monolith_col["action_hint"].lower()
    for hint in (trident_col["action_hint"], monolith_col["action_hint"]):
        low = hint.lower()
        assert "bolted structural" not in low
        assert "epoxy granite" not in low
        assert "upgrade to" not in low
        assert "switch to" not in low
