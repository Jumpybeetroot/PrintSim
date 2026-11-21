"""Physics verification tests for stepper electromagnetic stiffness, nozzle-tip CoG flex, and panel series mount compliance."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from printsim.engine import evaluate_printer
from printsim.fea import solve_modal_fea, solve_static_fea
from printsim.geometry import build_assembly, generate_bom
from printsim.geometry.builder import panel_mechanical_properties
from printsim.kinematics.solvers import (
    analyze_kinematics,
    compute_drive_stiffness_n_per_m,
    stepper_available_torque_nm,
)
from printsim.schema import PrinterSpec, StepperSpec, get_material

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def _base_corexy(
    enclosed: bool = True,
    mounting: str = "bolted_structural",
    material: str = "acm_dibond",
    thickness_mm: float = 4.0,
    z_drive: str = "leadscrew_triple",
) -> PrinterSpec:
    spec = PrinterSpec(architecture="corexy")
    spec.frame.bracing = "corner_gussets"
    spec.frame.enclosed_panels = enclosed
    spec.frame.panel_mounting = mounting  # type: ignore[assignment]
    spec.frame.panel_material = material
    spec.frame.panel_thickness_mm = thickness_mm
    spec.bed.z_drive = z_drive  # type: ignore[assignment]
    return spec


# ---------------------------------------------------------------------------
# 1. Stepper Motor Electromagnetic Stiffness & Electromechanical Coupling
# ---------------------------------------------------------------------------


def test_stepper_electromagnetic_stiffness_analytical_and_current_sweep() -> None:
    """Verify exact k_theta = p*T_hold(I)*eta_V, series belt+motor coupling, and run_current_a monotonicity."""
    spec = _base_corexy()
    spec.gantry.drive_mode = "awd"
    spec.gantry.belt.type = "gt2_9mm"
    spec.gantry.belt.core = "aramid"
    spec.gantry.belt.pretension_n = 55.0

    # Exact analytical check on StepperSpec & compute_drive_stiffness_n_per_m
    asm_check = build_assembly(spec)
    st = spec.kinematics.xy_stepper
    r_p = spec.gantry.belt.pitch_radius_m
    motor_lin_v = (spec.kinematics.max_velocity_mm_s * 1e-3) * 1.4142
    max_rpm = (motor_lin_v / (2.0 * math.pi * r_p)) * 60.0
    rpm_op = max(120.0, 0.35 * max_rpm)

    k_theta_expected = (st.steps_per_rev / 4.0) * st.effective_holding_torque_nm * st.dynamic_voltage_efficiency(rpm_op)
    assert st.torsional_stiffness_nm_per_rad(rpm=rpm_op) == pytest.approx(k_theta_expected, rel=1e-9)

    k_belt, k_motor_axis, k_drive = compute_drive_stiffness_n_per_m(asm_check, "X")
    expected_k_motor_awd = 4.0 * k_theta_expected / (r_p**2)
    assert k_motor_axis == pytest.approx(expected_k_motor_awd, rel=1e-9)
    assert k_drive == pytest.approx((k_belt * k_motor_axis) / (k_belt + k_motor_axis), rel=1e-9)

    # Verify modal series coupling: 1/f_eff^2 = 1/f_struct^2 + 1/f_belt^2 + 1/f_motor^2
    mod = solve_modal_fea(asm_check)
    assert mod.shaper_x is not None and mod.shaper_y is not None
    expected_f_eff_x = 1.0 / math.sqrt(
        1.0 / mod.shaper_x.structural_freq_hz**2
        + 1.0 / mod.shaper_x.belt_freq_hz**2
        + 1.0 / mod.shaper_x.motor_freq_hz**2
    )
    assert mod.shaper_x.effective_freq_hz == pytest.approx(expected_f_eff_x, rel=1e-6)

    # Increasing run_current_a (0.8 A -> 1.6 A -> 2.4 A) strictly increases motor stiffness & effective shaper freq
    currents = [0.8, 1.6, 2.4]
    kins = []
    mods = []
    for curr in currents:
        s_i = spec.model_copy(deep=True)
        s_i.kinematics.xy_stepper.run_current_a = curr
        asm_i = build_assembly(s_i)
        kins.append(analyze_kinematics(asm_i))
        mods.append(solve_modal_fea(asm_i))

    for idx in range(len(currents) - 1):
        assert kins[idx + 1].motor_stiffness_n_per_mm > kins[idx].motor_stiffness_n_per_mm
        assert kins[idx + 1].drive_stiffness_n_per_mm > kins[idx].drive_stiffness_n_per_mm
        assert mods[idx + 1].shaper_x is not None and mods[idx].shaper_x is not None
        assert mods[idx + 1].shaper_y is not None and mods[idx].shaper_y is not None
        assert mods[idx + 1].shaper_x.motor_freq_hz > mods[idx].shaper_x.motor_freq_hz
        assert mods[idx + 1].shaper_x.effective_freq_hz > mods[idx].shaper_x.effective_freq_hz
        assert mods[idx + 1].shaper_y.effective_freq_hz > mods[idx].shaper_y.effective_freq_hz


def test_stepper_09deg_vs_18deg_and_24v_vs_48v_and_overcurrent() -> None:
    """Verify 0.9 deg ~2x static stiffness, 48V inductive headroom, and overcurrent heat/risk flag."""
    # 1. 0.9 deg (400 steps/rev) vs 1.8 deg (200 steps/rev) holding torque & inductance equal
    s_18 = StepperSpec(
        holding_torque_nm=0.55,
        rated_current_a=2.5,
        run_current_a=1.5,
        inductance_mh=1.5,
        supply_voltage_v=48.0,
        steps_per_rev=200,
    )
    s_09 = s_18.model_copy(update={"steps_per_rev": 400})
    assert s_09.pole_pairs == 100.0 and s_18.pole_pairs == 50.0
    # At low RPM (near static), 0.9 deg motor nearly doubles torsional stiffness
    ratio_static = s_09.torsional_stiffness_nm_per_rad(rpm=20.0) / s_18.torsional_stiffness_nm_per_rad(rpm=20.0)
    assert 1.90 <= ratio_static <= 2.01

    spec_18 = _base_corexy()
    spec_18.kinematics.xy_stepper = s_18
    spec_09 = _base_corexy()
    spec_09.kinematics.xy_stepper = s_09
    kin_18 = analyze_kinematics(build_assembly(spec_18))
    kin_09 = analyze_kinematics(build_assembly(spec_09))
    assert kin_09.motor_stiffness_n_per_mm > kin_18.motor_stiffness_n_per_mm * 1.7

    # 2. 24V vs 48V on higher-inductance 3.0 mH motor at 600 mm/s high speed
    spec_24v = _base_corexy()
    spec_24v.kinematics.max_velocity_mm_s = 600.0
    spec_24v.kinematics.xy_stepper = StepperSpec(
        holding_torque_nm=0.59,
        rated_current_a=2.0,
        run_current_a=1.6,
        phase_resistance_ohm=1.4,
        inductance_mh=3.0,
        supply_voltage_v=24.0,
        steps_per_rev=200,
    )
    spec_48v = spec_24v.model_copy(deep=True)
    spec_48v.kinematics.xy_stepper.supply_voltage_v = 48.0

    kin_24 = analyze_kinematics(build_assembly(spec_24v))
    kin_48 = analyze_kinematics(build_assembly(spec_48v))
    assert kin_48.motor_voltage_efficiency_pct > kin_24.motor_voltage_efficiency_pct + 15.0
    assert kin_48.available_torque_at_vmax_nm > kin_24.available_torque_at_vmax_nm * 1.25
    assert stepper_available_torque_nm(spec_48v.kinematics.xy_stepper, 1000.0) > stepper_available_torque_nm(
        spec_24v.kinematics.xy_stepper, 1000.0
    )

    # 3. Overcurrent > 1.05 * rated_current_a triggers flag, 2*I^2*R heating, and evaluator diagnostic
    spec_oc = _base_corexy()
    spec_oc.kinematics.xy_stepper.rated_current_a = 2.0
    spec_oc.kinematics.xy_stepper.run_current_a = 2.3
    spec_oc.kinematics.xy_stepper.phase_resistance_ohm = 1.4
    rep_oc = evaluate_printer(spec_oc)
    assert rep_oc.kinematics.motor_overcurrent_risk is True
    assert rep_oc.kinematics.motor_heat_w_each == pytest.approx(2.0 * (2.3**2) * 1.4, rel=1e-6)
    assert any("overcurrent" in d.lower() for d in rep_oc.diagnostics)

    # 4. 2WD vs AWD doubles parallel axis motor stiffness
    spec_2wd = _base_corexy()
    spec_2wd.gantry.drive_mode = "2wd"
    spec_awd = spec_2wd.model_copy(deep=True)
    spec_awd.gantry.drive_mode = "awd"
    kin_2wd = analyze_kinematics(build_assembly(spec_2wd))
    kin_awd = analyze_kinematics(build_assembly(spec_awd))
    assert kin_awd.motor_stiffness_n_per_mm == pytest.approx(2.0 * kin_2wd.motor_stiffness_n_per_mm, rel=1e-3)


# ---------------------------------------------------------------------------
# 2. Toolhead CoG & Nozzle-Tip 6-DOF Torsional/Pitch Vibration
# ---------------------------------------------------------------------------


def test_toolhead_cog_nozzle_drop_and_mount_material_stiffness() -> None:
    """Verify longer nozzle/CoG lever arms amplify flex and stiffer mounts reduce twist."""
    # 1. Moving nozzle_offset_mm Z from -45 mm -> -62 mm -> -80 mm strictly increases nozzle flex & twist amp
    nozzle_drops = [-45.0, -62.0, -80.0]
    noz_results = []
    for nz in nozzle_drops:
        s = _base_corexy()
        s.toolhead.mount_material = "abs_printed"
        s.toolhead.cog_offset_mm = (0.0, -18.0, -24.0)
        s.toolhead.nozzle_offset_mm = (0.0, -22.0, nz)
        noz_results.append(solve_static_fea(build_assembly(s)))

    for idx in range(len(nozzle_drops) - 1):
        assert noz_results[idx + 1].accel_deflection_y_um > noz_results[idx].accel_deflection_y_um
        assert noz_results[idx + 1].accel_deflection_x_um > noz_results[idx].accel_deflection_x_um
        assert noz_results[idx + 1].combined_max_deflection_um > noz_results[idx].combined_max_deflection_um
        assert noz_results[idx + 1].nozzle_twist_amplification_pct > noz_results[idx].nozzle_twist_amplification_pct

    # 2. Moving cog_offset_mm Z from -10 mm -> -22 mm -> -35 mm strictly increases torsion & nozzle deflection
    cog_drops = [-10.0, -22.0, -35.0]
    cog_results = []
    for cz in cog_drops:
        s = _base_corexy()
        s.toolhead.mount_material = "abs_printed"
        s.toolhead.cog_offset_mm = (0.0, -18.0, cz)
        s.toolhead.nozzle_offset_mm = (0.0, -22.0, -70.0)
        cog_results.append(solve_static_fea(build_assembly(s)))

    for idx in range(len(cog_drops) - 1):
        assert cog_results[idx + 1].accel_deflection_y_um > cog_results[idx].accel_deflection_y_um
        assert cog_results[idx + 1].accel_deflection_x_um > cog_results[idx].accel_deflection_x_um
        assert cog_results[idx + 1].nozzle_twist_amplification_pct > cog_results[idx].nozzle_twist_amplification_pct
        assert cog_results[idx + 1].nozzle_twist_angle_mrad > cog_results[idx].nozzle_twist_angle_mrad

    # 3. Mount material upgrade (abs_printed -> pa12_cf_printed -> aluminum_6061_t6) strictly improves nozzle stiffness & resonance
    mounts = ["abs_printed", "pa12_cf_printed", "aluminum_6061_t6"]
    mount_st = []
    mount_mo = []
    for mat in mounts:
        s = _base_corexy()
        s.toolhead.mount_material = mat  # type: ignore[assignment]
        s.toolhead.total_mass_g = 420.0
        s.toolhead.cog_offset_mm = (0.0, -22.0, -30.0)
        s.toolhead.nozzle_offset_mm = (0.0, -26.0, -74.0)
        asm = build_assembly(s)
        mount_st.append(solve_static_fea(asm))
        mount_mo.append(solve_modal_fea(asm))

    for idx in range(len(mounts) - 1):
        assert mount_st[idx + 1].accel_deflection_y_um < mount_st[idx].accel_deflection_y_um
        assert mount_st[idx + 1].combined_max_deflection_um < mount_st[idx].combined_max_deflection_um
        assert mount_st[idx + 1].nozzle_twist_angle_mrad < mount_st[idx].nozzle_twist_angle_mrad
        assert mount_st[idx + 1].toolhead_stiffness_y_n_per_mm > mount_st[idx].toolhead_stiffness_y_n_per_mm
        assert mount_mo[idx + 1].nozzle_pitch_twist_freq_hz > mount_mo[idx].nozzle_pitch_twist_freq_hz

    # 4. Long heavy ABS toolhead triggers high twist amplification diagnostic, and BOM skips toolhead sub-beams
    s_long = _base_corexy()
    s_long.toolhead.mount_material = "abs_printed"
    s_long.toolhead.total_mass_g = 460.0
    s_long.toolhead.cog_offset_mm = (0.0, -26.0, -34.0)
    s_long.toolhead.nozzle_offset_mm = (0.0, -28.0, -78.0)
    s_long.kinematics.max_accel_mm_s2 = 14000.0
    rep_long = evaluate_printer(s_long)
    assert rep_long.static_fea.nozzle_twist_amplification_pct > 45.0
    assert any("nozzle-tip" in d.lower() or "twist amplification" in d.lower() for d in rep_long.diagnostics)

    gdict = rep_long.assembly.to_geometry_dict()
    assert gdict["nozzle_node_id"] == "nozzle_tip"
    assert gdict["toolhead_cog_node_id"] == "toolhead_cog"
    assert gdict["mount_material"] == "abs_printed"
    assert gdict["nozzle_offset_mm"] == [0.0, -28.0, -78.0]

    bom = generate_bom(s_long)
    assert all(c.role != "toolhead" for c in bom.cut_list)


# ---------------------------------------------------------------------------
# 3. Panel Fastener Series Compliance & Rayleigh Vertical Mass Lumping
# ---------------------------------------------------------------------------


def test_panel_fastener_series_compliance_and_vertical_mass_distribution() -> None:
    """Verify clip <25% mild stiffening, bolted 4mm ACM (+gantry) dominance, and 0.35/0.15 mass lumping."""
    # 1. Exact series fastener compliance formula check (Huth bolted joint model)
    dibond = get_material("acm_dibond")
    al5052 = get_material("aluminum_5052_sheet")
    p_fly = panel_mechanical_properties(dibond, 0.004, 0.46, 0.50, "bolted_structural", bolted_to_gantry=False)
    p_gan = panel_mechanical_properties(dibond, 0.004, 0.46, 0.50, "bolted_structural", bolted_to_gantry=True)
    p_al4 = panel_mechanical_properties(al5052, 0.004, 0.46, 0.50, "bolted_structural", bolted_to_gantry=True)
    assert p_fly["k_panel_eff_n_m"] == pytest.approx(
        (p_fly["k_sheet_n_m"] * p_fly["k_mount_n_m"]) / (p_fly["k_sheet_n_m"] + p_fly["k_mount_n_m"]),
        rel=1e-9,
    )
    assert p_gan["k_panel_eff_n_m"] == pytest.approx(
        (p_gan["k_sheet_n_m"] * p_gan["k_mount_n_m"]) / (p_gan["k_sheet_n_m"] + p_gan["k_mount_n_m"]),
        rel=1e-9,
    )
    assert p_gan["k_mount_n_m"] > p_fly["k_mount_n_m"]
    assert p_gan["gt_eff_n_m"] > p_fly["gt_eff_n_m"]
    assert p_al4["k_panel_eff_n_m"] > p_gan["k_panel_eff_n_m"] * 5.0

    # 2. Open vs Clip 3mm Acrylic vs Bolted 3mm Acrylic vs Bolted 3/4/6mm ACM (+gantry)
    s_open = _base_corexy(enclosed=False, mounting="none", material="acm_dibond", thickness_mm=3.0)
    s_clip = _base_corexy(enclosed=True, mounting="clip_enclosure", material="acrylic_pmma", thickness_mm=3.0)
    s_bolt_acr = _base_corexy(enclosed=True, mounting="bolted_structural", material="acrylic_pmma", thickness_mm=3.0)
    s_bolt_3acm = _base_corexy(enclosed=True, mounting="bolted_structural", material="acm_dibond", thickness_mm=3.0)
    s_bolt_4acm = _base_corexy(enclosed=True, mounting="bolted_structural", material="acm_dibond", thickness_mm=4.0)
    s_bolt_6acm = _base_corexy(enclosed=True, mounting="bolted_structural", material="acm_dibond", thickness_mm=6.0)

    asm_open = build_assembly(s_open)
    asm_clip = build_assembly(s_clip)
    asm_bolt_acr = build_assembly(s_bolt_acr)
    asm_bolt_3acm = build_assembly(s_bolt_3acm)
    asm_bolt_4acm = build_assembly(s_bolt_4acm)
    asm_bolt_6acm = build_assembly(s_bolt_6acm)

    st_open = solve_static_fea(asm_open)
    st_clip = solve_static_fea(asm_clip)
    st_bolt_acr = solve_static_fea(asm_bolt_acr)
    st_bolt_3acm = solve_static_fea(asm_bolt_3acm)
    st_bolt_4acm = solve_static_fea(asm_bolt_4acm)
    st_bolt_6acm = solve_static_fea(asm_bolt_6acm)

    mo_open = solve_modal_fea(asm_open)
    mo_clip = solve_modal_fea(asm_clip)
    mo_bolt_acr = solve_modal_fea(asm_bolt_acr)
    mo_bolt_3acm = solve_modal_fea(asm_bolt_3acm)
    mo_bolt_4acm = solve_modal_fea(asm_bolt_4acm)
    mo_bolt_6acm = solve_modal_fea(asm_bolt_6acm)

    # Clip 3mm acrylic gives realistic mild stiffening (< 25% deflection drop vs Open)
    defl_drop_clip = (st_open.combined_max_deflection_um - st_clip.combined_max_deflection_um) / st_open.combined_max_deflection_um
    assert 0.0 < defl_drop_clip < 0.25
    assert st_clip.toolhead_stiffness_x_n_per_mm > st_open.toolhead_stiffness_x_n_per_mm
    assert mo_clip.fundamental_freq_hz < mo_open.fundamental_freq_hz * 1.15

    # Bolted 4mm ACM (+gantry) cuts deflection dramatically and outperforms Bolted 3mm Acrylic
    assert st_bolt_4acm.combined_max_deflection_um < st_open.combined_max_deflection_um * 0.50
    assert st_bolt_4acm.toolhead_stiffness_x_n_per_mm > st_bolt_acr.toolhead_stiffness_x_n_per_mm
    assert st_bolt_4acm.toolhead_stiffness_y_n_per_mm > st_bolt_acr.toolhead_stiffness_y_n_per_mm
    assert mo_bolt_4acm.fundamental_freq_hz > mo_bolt_acr.fundamental_freq_hz
    assert mo_bolt_4acm.fundamental_freq_hz >= mo_bolt_3acm.fundamental_freq_hz * 0.98

    # Monotonic static stiffness across thicknesses & 6mm ACM remains far stiffer than bolted acrylic/open
    assert st_bolt_6acm.toolhead_stiffness_x_n_per_mm > st_bolt_4acm.toolhead_stiffness_x_n_per_mm > st_bolt_3acm.toolhead_stiffness_x_n_per_mm
    assert mo_bolt_6acm.fundamental_freq_hz > mo_bolt_acr.fundamental_freq_hz > mo_open.fundamental_freq_hz * 1.45

    # Verify exact 0.35 / 0.15 vertical wall mass lumping on flying-gantry CoreXY (clean 4-bay geometry)
    s_fly_open = _base_corexy(enclosed=False, mounting="none", z_drive="belted_quad_z")
    s_fly_bolt = _base_corexy(enclosed=True, mounting="bolted_structural", material="acm_dibond", thickness_mm=4.0, z_drive="belted_quad_z")
    asm_fo = build_assembly(s_fly_open)
    asm_fb = build_assembly(s_fly_bolt)

    w = s_fly_bolt.frame.width_mm * 1e-3
    d = s_fly_bolt.frame.depth_mm * 1e-3
    h = s_fly_bolt.frame.height_mm * 1e-3
    areal = panel_mechanical_properties(dibond, 0.004, w, h, "bolted_structural")["areal_mass_kg_m2"]
    m_rear = w * h * areal
    m_side = d * h * areal
    m_top = w * d * areal

    nmap_o = asm_fo.node_map
    nmap_b = asm_fb.node_map
    # b_fl receives 0.35 of left wall; b_rl receives 0.35 of left + 0.35 of rear
    assert (nmap_b["b_fl"].lumped_mass_kg - nmap_o["b_fl"].lumped_mass_kg) == pytest.approx(0.35 * m_side, rel=1e-6)
    assert (nmap_b["b_rl"].lumped_mass_kg - nmap_o["b_rl"].lumped_mass_kg) == pytest.approx(0.35 * (m_side + m_rear), rel=1e-6)
    # t_fl receives 0.15 of left + 0.25 of top; t_rl receives 0.15 of left + 0.15 of rear + 0.25 of top
    assert (nmap_b["t_fl"].lumped_mass_kg - nmap_o["t_fl"].lumped_mass_kg) == pytest.approx(0.15 * m_side + 0.25 * m_top, rel=1e-6)
    assert (nmap_b["t_rl"].lumped_mass_kg - nmap_o["t_rl"].lumped_mass_kg) == pytest.approx(0.15 * (m_side + m_rear) + 0.25 * m_top, rel=1e-6)


def test_updated_trident_vs_monolith_awd_example_specs() -> None:
    """Verify updated Voron Trident 300 and Voron Monolith AWD example YAMLs simulate cleanly."""
    trident = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    monolith = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_monolith_awd.yaml")

    rep_t = evaluate_printer(trident)
    rep_m = evaluate_printer(monolith)

    assert not rep_t.collision.has_collision
    assert not rep_m.collision.has_collision
    assert not rep_t.kinematics.motor_overcurrent_risk
    assert not rep_m.kinematics.motor_overcurrent_risk

    assert rep_m.kinematics.motor_stiffness_n_per_mm > rep_t.kinematics.motor_stiffness_n_per_mm * 2.0
    assert rep_m.kinematics.drive_stiffness_n_per_mm > rep_t.kinematics.drive_stiffness_n_per_mm * 4.0
    assert rep_m.modal_fea.nozzle_pitch_twist_freq_hz > rep_t.modal_fea.nozzle_pitch_twist_freq_hz
    assert rep_m.modal_fea.recommended_max_accel_mm_s2 > rep_t.modal_fea.recommended_max_accel_mm_s2 * 3.0
    assert rep_m.static_fea.combined_max_deflection_um < rep_t.static_fea.combined_max_deflection_um
    assert rep_m.scores.composite > rep_t.scores.composite
