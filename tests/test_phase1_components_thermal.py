"""Comprehensive Phase 1 benchmarks: component mechanics, thermal-structural coupling, and horizontal FEA."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from printsim.api.server import create_app
from printsim.engine import evaluate_printer
from printsim.fea import solve_modal_fea, solve_static_fea
from printsim.fea.components import (
    belt_effective_ea_n,
    compute_thermal_structural_report,
    corner_joint_stiffness,
    gerbert_tooth_mesh_stiffness_n_m,
    hertzian_rail_carriage_stiffness,
    idler_shaft_and_bearing_stiffness,
    iwan_joint_damping_delta,
    newmark_rail_composite_gamma,
)
from printsim.geometry import build_assembly
from printsim.kinematics import analyze_kinematics
from printsim.schema import BeltSpec, PrinterSpec, StepperSpec, get_material, get_profile

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def _corexy_base() -> PrinterSpec:
    spec = PrinterSpec(architecture="corexy")
    spec.frame.width_mm = 350.0
    spec.frame.depth_mm = 350.0
    spec.frame.height_mm = 420.0
    spec.kinematics.build_volume_mm = (230.0, 230.0, 230.0)
    spec.frame.bracing = "corner_gussets"
    spec.frame.enclosed_panels = False
    spec.gantry.x_beam_profile = "extrusion_2020"
    spec.gantry.x_beam_material = "aluminum_6063_t5"
    spec.gantry.x_rail_profile = "mgn12_rail"
    spec.gantry.x_rail_mount = "top"
    return spec


# ---------------------------------------------------------------------------
# 1. Timoshenko (1925) Bimetallic Gantry Bow & Backer Cancellation
# ---------------------------------------------------------------------------


def test_timoshenko_bimetallic_bow_analytical_and_backers() -> None:
    """Verify MGN12 on 2020 (L=350mm, dT=35K) gives 100-220 um bow, >75% backer cut, and 0 um dual."""
    spec = _corexy_base()
    spec.frame.chamber_temp_c = 60.0  # dT = +35 K above 25 C reference

    rep_none = compute_thermal_structural_report(build_assembly(spec))
    assert rep_none.chamber_temp_c == pytest.approx(60.0, abs=1e-6)
    assert rep_none.delta_t_k == pytest.approx(35.0, abs=1e-6)
    assert 100.0 <= rep_none.gantry_bimetallic_bow_um <= 220.0
    assert rep_none.gantry_backer_reduction_pct == pytest.approx(0.0, abs=1e-6)

    # Closed-form Timoshenko (1925) + Newmark partial-interaction cross-check (read D-04 profile directly)
    prof_2020 = get_profile("extrusion_2020")
    eb, ab, ib, alphab = 68.9e9, prof_2020.area_m2, prof_2020.iy_m4, 23.4e-6
    mgn12 = get_profile("mgn12_rail")
    steel = get_material("steel_4140")
    er, ar, ir, alphar = steel.youngs_modulus_pa, mgn12.area_m2, mgn12.iy_m4, steel.thermal_expansion_1_k
    c = 0.010 + 0.5 * mgn12.height_m
    from printsim.geometry.builder import corexy_frame_centerlines_m

    l_x_cl = corexy_frame_centerlines_m(spec)[0]
    gamma = newmark_rail_composite_gamma(er, ar, eb, ab, 0.025, l_x_cl, "extrusion")
    kappa_timo = (alphab - alphar) * 35.0 * c / (c**2 + (eb * ib + er * ir) * (1.0 / (eb * ab) + 1.0 / (er * ar)))
    expected_bow_um = abs(gamma * kappa_timo) * (l_x_cl**2) / 8.0 * 1e6
    assert rep_none.gantry_bimetallic_bow_um == pytest.approx(expected_bow_um, rel=1e-6)
    assert rep_none.rail_newmark_gamma == pytest.approx(gamma, rel=1e-6)

    # Titanium Grade 5 and Steel 4140 backers each cut bimetallic bow by > 75% and add real stiffness
    asm_none = build_assembly(spec)
    st_none = solve_static_fea(asm_none)
    for bk_mat in ("titanium_grade5", "steel_4140"):
        s_bk = spec.model_copy(deep=True)
        s_bk.gantry.rail_backer_material = bk_mat  # type: ignore[assignment]
        s_bk.gantry.rail_backer_thickness_mm = 3.0
        asm_bk = build_assembly(s_bk)
        rep_bk = compute_thermal_structural_report(asm_bk)
        assert rep_bk.gantry_backer_reduction_pct > 75.0
        assert rep_bk.gantry_bimetallic_bow_um < rep_none.gantry_bimetallic_bow_um * 0.25
        assert asm_bk.y_moving_mass_kg > asm_none.y_moving_mass_kg
        assert solve_static_fea(asm_bk).toolhead_stiffness_z_n_per_mm > st_none.toolhead_stiffness_z_n_per_mm

    # Dual top+bottom rails cancel bimetallic bow identically to 0.0 um
    s_dual = spec.model_copy(deep=True)
    s_dual.gantry.x_rail_mount = "dual_top_bottom"
    rep_dual = compute_thermal_structural_report(build_assembly(s_dual))
    assert rep_dual.gantry_bimetallic_bow_um == pytest.approx(0.0, abs=1e-9)
    assert rep_dual.gantry_backer_reduction_pct == pytest.approx(100.0, abs=1e-6)

    # At 25 C reference temperature, bimetallic bow is 0.0 um and evaluator emits bow warning at 60 C
    s_cold = spec.model_copy(deep=True)
    s_cold.frame.chamber_temp_c = 25.0
    assert compute_thermal_structural_report(build_assembly(s_cold)).gantry_bimetallic_bow_um == pytest.approx(0.0, abs=1e-9)

    eval_hot = evaluate_printer(spec)
    assert any("bimetallic" in d.lower() for d in eval_hot.diagnostics)


# ---------------------------------------------------------------------------
# 2. Bed Thermal Gradient Bow & Rigid 4-Point Taco Buckling
# ---------------------------------------------------------------------------


def test_bed_thermal_buckling_rigid_vs_floating_vs_kinematic() -> None:
    """rigid_4pt at 110 C produces > 100 um taco buckling & diagnostic; kinematic_3pt gives 0 um."""
    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    base.bed.target_temp_c = 110.0

    s_kin = base.model_copy(deep=True)
    s_kin.bed.mount_type = "kinematic_3pt"

    s_flt = base.model_copy(deep=True)
    s_flt.bed.mount_type = "floating_4pt"

    s_rig = base.model_copy(deep=True)
    s_rig.bed.mount_type = "rigid_4pt"

    ev_kin = evaluate_printer(s_kin)
    ev_flt = evaluate_printer(s_flt)
    ev_rig = evaluate_printer(s_rig)

    assert ev_kin.thermal_structural.bed_buckling_taco_um == 0.0
    assert 0.0 < ev_flt.thermal_structural.bed_buckling_taco_um < 25.0
    assert ev_rig.thermal_structural.bed_buckling_taco_um > 100.0
    assert ev_kin.thermal_structural.bed_gradient_bow_um > 0.0
    assert ev_rig.thermal_structural.total_thermal_z_error_um > ev_kin.thermal_structural.total_thermal_z_error_um + 100.0
    assert any("taco buckling" in d.lower() for d in ev_rig.diagnostics)
    assert not any("taco buckling" in d.lower() for d in ev_kin.diagnostics)


# ---------------------------------------------------------------------------
# 3. Mahieux-Reifsnider (2002) Viscoelastic E(T) & Heated Chamber Softening
# ---------------------------------------------------------------------------


def test_mahieux_reifsnider_modulus_retention_and_chamber_softening() -> None:
    """PA12-CF loses >25% E at 60 C while PPA-CF & PPS-CF retain >93%; metals return 1.0."""
    pa12 = get_material("pa12_cf_printed")
    ppa = get_material("ppa_cf_printed")
    pps = get_material("pps_cf_printed")
    al = get_material("aluminum_6061_t6")

    assert pa12.modulus_retention_at_temp(25.0) == pytest.approx(1.0, rel=1e-9)
    assert ppa.modulus_retention_at_temp(25.0) == pytest.approx(1.0, rel=1e-9)
    assert pps.modulus_retention_at_temp(25.0) == pytest.approx(1.0, rel=1e-9)
    assert al.modulus_retention_at_temp(80.0) == pytest.approx(1.0, rel=1e-9)

    ret_pa12_60 = pa12.modulus_retention_at_temp(60.0)
    ret_ppa_60 = ppa.modulus_retention_at_temp(60.0)
    ret_pps_60 = pps.modulus_retention_at_temp(60.0)
    assert 0.58 <= ret_pa12_60 <= 0.68
    assert ret_pa12_60 < 0.75
    assert ret_ppa_60 > 0.93
    assert ret_pps_60 > 0.93
    assert pa12.effective_modulus_pa(60.0) == pytest.approx(pa12.youngs_modulus_pa * ret_pa12_60, rel=1e-9)

    # Chamber heating degrades PA12-CF bracket stiffness & triggers Tg warning, while PPA-CF holds firm
    s_pa12_25 = _corexy_base()
    s_pa12_25.gantry.bracket_material = "pa12_cf_printed"
    s_pa12_25.toolhead.mount_material = "pa12_cf_printed"
    s_pa12_25.frame.chamber_temp_c = 25.0

    s_pa12_60 = s_pa12_25.model_copy(deep=True)
    s_pa12_60.frame.chamber_temp_c = 60.0

    s_ppa_60 = s_pa12_60.model_copy(deep=True)
    s_ppa_60.gantry.bracket_material = "ppa_cf_printed"
    s_ppa_60.toolhead.mount_material = "ppa_cf_printed"

    ev_pa12_25 = evaluate_printer(s_pa12_25)
    ev_pa12_60 = evaluate_printer(s_pa12_60)
    ev_ppa_60 = evaluate_printer(s_ppa_60)

    assert ev_pa12_60.kinematics.bracket_stiffness_n_per_mm < ev_pa12_25.kinematics.bracket_stiffness_n_per_mm * 0.72
    assert ev_pa12_60.kinematics.bracket_modulus_retention_pct < 70.0
    assert ev_pa12_60.thermal_structural.bracket_tg_warning is True
    assert any("tg" in d.lower() or "softening" in d.lower() for d in ev_pa12_60.diagnostics)

    assert ev_ppa_60.thermal_structural.bracket_tg_warning is False
    assert ev_ppa_60.thermal_structural.bracket_modulus_retention_pct > 93.0


# ---------------------------------------------------------------------------
# 4. Newmark Partial-Interaction Linear Rail Shear Lag
# ---------------------------------------------------------------------------


def test_newmark_rail_composite_gamma_pitch_and_substrate_hierarchy() -> None:
    """25 mm pitch beats 50 mm skipped bolts; solid_bar > extrusion > carbon_tube; FEA responds."""
    steel = get_material("steel_4140")
    al = get_material("aluminum_6063_t5")
    cf = get_material("carbon_fiber_tube")
    mgn12 = get_profile("mgn12_rail")
    ext2020 = get_profile("extrusion_2020")
    cf2020 = get_profile("cf_tube_20x20x2")

    g_ext_25 = newmark_rail_composite_gamma(
        steel.youngs_modulus_pa, mgn12.area_m2, al.youngs_modulus_pa, ext2020.area_m2, 0.025, 0.350, "extrusion"
    )
    g_ext_50 = newmark_rail_composite_gamma(
        steel.youngs_modulus_pa, mgn12.area_m2, al.youngs_modulus_pa, ext2020.area_m2, 0.050, 0.350, "extrusion"
    )
    g_cf_25 = newmark_rail_composite_gamma(
        steel.youngs_modulus_pa, mgn12.area_m2, cf.youngs_modulus_pa, cf2020.area_m2, 0.025, 0.350, "carbon_tube"
    )
    g_solid_25 = newmark_rail_composite_gamma(
        steel.youngs_modulus_pa, mgn12.area_m2, al.youngs_modulus_pa, 4.0e-4, 0.025, 0.350, "solid_bar"
    )

    assert 0.50 <= g_ext_25 <= 0.75
    assert g_ext_25 > g_ext_50
    assert g_solid_25 > g_ext_25 > g_cf_25

    s_25 = _corexy_base()
    s_25.gantry.rail_bolt_pitch_mm = 25.0
    s_50 = s_25.model_copy(deep=True)
    s_50.gantry.rail_bolt_pitch_mm = 50.0

    asm_25 = build_assembly(s_25)
    asm_50 = build_assembly(s_50)
    assert asm_25.x_rail_ei_z_boost_pct > asm_50.x_rail_ei_z_boost_pct
    assert solve_static_fea(asm_25).toolhead_stiffness_z_n_per_mm > solve_static_fea(asm_50).toolhead_stiffness_z_n_per_mm


# ---------------------------------------------------------------------------
# 5. 6-DOF Hertzian Rail Carriage Stiffness (Preload ZF..ZB & Block C vs H)
# ---------------------------------------------------------------------------


def test_hertzian_rail_carriage_preload_and_block_type_in_fea() -> None:
    """ZF < Z0 < Z1 < ZA < ZB and H block > C block in both component stiffness and FEA nozzle twist."""
    preloads = ("ZF", "Z0", "Z1", "ZA", "ZB")
    h_vals = [hertzian_rail_carriage_stiffness("mgn12_rail", preload=p, block_type="H") for p in preloads]
    for i in range(len(preloads) - 1):
        assert h_vals[i + 1]["k_zz_n_m"] > h_vals[i]["k_zz_n_m"]
        assert h_vals[i + 1]["k_rx_nm_rad"] > h_vals[i]["k_rx_nm_rad"]
        assert h_vals[i + 1]["k_ry_nm_rad"] > h_vals[i]["k_ry_nm_rad"]
        assert h_vals[i + 1]["rolling_friction_n"] > h_vals[i]["rolling_friction_n"]

    c_z1 = hertzian_rail_carriage_stiffness("mgn12_rail", preload="Z1", block_type="C")
    h_z1 = hertzian_rail_carriage_stiffness("mgn12_rail", preload="Z1", block_type="H")
    assert h_z1["k_ry_nm_rad"] > c_z1["k_ry_nm_rad"] * 2.0
    assert h_z1["k_rz_nm_rad"] > c_z1["k_rz_nm_rad"] * 2.0
    assert h_z1["k_zz_n_m"] > c_z1["k_zz_n_m"]

    # HGH15 beats MGN15 beats MGN12 beats MGN9
    assert (
        hertzian_rail_carriage_stiffness("hgh15_rail")["k_zz_n_m"]
        > hertzian_rail_carriage_stiffness("mgn15_rail")["k_zz_n_m"]
        > hertzian_rail_carriage_stiffness("mgn12_rail")["k_zz_n_m"]
        > hertzian_rail_carriage_stiffness("mgn9_rail")["k_zz_n_m"]
    )

    # Full assembly FEA response with CNC toolhead mount
    base = _corexy_base()
    base.toolhead.mount_material = "aluminum_6061_t6"
    st_pre = []
    for p in preloads:
        s_p = base.model_copy(deep=True)
        s_p.gantry.rail_preload = p  # type: ignore[assignment]
        st_pre.append(solve_static_fea(build_assembly(s_p)))
    for i in range(len(preloads) - 1):
        assert st_pre[i + 1].toolhead_stiffness_y_n_per_mm > st_pre[i].toolhead_stiffness_y_n_per_mm
        assert st_pre[i + 1].nozzle_twist_angle_mrad < st_pre[i].nozzle_twist_angle_mrad

    s_c = base.model_copy(deep=True)
    s_c.gantry.rail_block_type = "C"
    s_h = base.model_copy(deep=True)
    s_h.gantry.rail_block_type = "H"
    asm_c = build_assembly(s_c)
    asm_h = build_assembly(s_h)
    st_c = solve_static_fea(asm_c)
    st_h = solve_static_fea(asm_h)
    assert (asm_h.x_moving_mass_kg - asm_c.x_moving_mass_kg) * 1000.0 == pytest.approx(20.0, abs=1e-9)
    assert st_h.toolhead_stiffness_x_n_per_mm > st_c.toolhead_stiffness_x_n_per_mm
    assert st_h.toolhead_stiffness_y_n_per_mm > st_c.toolhead_stiffness_y_n_per_mm
    assert (
        st_h.toolhead_flex_breakdown["rail_carriage_stiffness_n_per_mm"]
        > st_c.toolhead_flex_breakdown["rail_carriage_stiffness_n_per_mm"] * 1.3
    )
    # Equalizing moving mass (+20 g on MGN12C to match 54 g MGN12H) isolates H-block's lower twist angle
    s_c_eq = s_c.model_copy(deep=True)
    s_c_eq.toolhead.total_mass_g += 20.0
    st_c_eq = solve_static_fea(build_assembly(s_c_eq))
    assert st_h.nozzle_twist_angle_mrad < st_c_eq.nozzle_twist_angle_mrad


# ---------------------------------------------------------------------------
# 6. Semi-Rigid Corner Joint Hierarchy & Iwan Damping
# ---------------------------------------------------------------------------


def test_corner_joint_stiffness_hierarchy_and_damping() -> None:
    """l_bracket < blind_joint < gusset_plate < corner_cube < rigid in stiffness & modal f0."""
    joints = ("l_bracket", "blind_joint", "gusset_plate", "corner_cube", "rigid")
    cj_list = [corner_joint_stiffness("extrusion_2020", j) for j in joints]
    for i in range(len(joints) - 1):
        assert cj_list[i + 1]["k_r_inplane_nm_rad"] > cj_list[i]["k_r_inplane_nm_rad"]
        assert cj_list[i + 1]["k_r_torsion_nm_rad"] > cj_list[i]["k_r_torsion_nm_rad"]
        assert iwan_joint_damping_delta(joints[i + 1]) < iwan_joint_damping_delta(joints[i])

    trident_open = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    trident_open.frame.bracing = "none"
    trident_open.frame.enclosed_panels = False
    trident_open.frame.panel_mounting = "none"

    f0s = []
    kxs = []
    for j in joints:
        s_j = trident_open.model_copy(deep=True)
        s_j.frame.corner_joint_type = j  # type: ignore[assignment]
        asm = build_assembly(s_j)
        f0s.append(solve_modal_fea(asm).fundamental_freq_hz)
        kxs.append(solve_static_fea(asm).toolhead_stiffness_x_n_per_mm)
    for i in range(len(joints) - 1):
        assert f0s[i + 1] > f0s[i]
        assert kxs[i + 1] > kxs[i]

    # Calibrated baseline range checks at default blind_joint
    assert 35.0 <= f0s[1] <= 55.0
    bedslinger = PrinterSpec.from_yaml(EXAMPLES_DIR / "cartesian_bedslinger.yaml")
    assert 88.0 <= solve_modal_fea(build_assembly(bedslinger)).fundamental_freq_hz <= 102.0


# ---------------------------------------------------------------------------
# 7. Gerbert Tooth-Mesh, Idler Shaft Cantilever vs Double-Shear, & ISO Play
# ---------------------------------------------------------------------------


def test_gerbert_tooth_mesh_idler_shaft_and_bearing_clearance() -> None:
    """180 deg > 90 deg tooth mesh; double_shear > 8x cantilever; 6mm > 5mm shaft; ISO C2/CN/C3."""
    belt_180 = BeltSpec(type="gt2_6mm", core="fiberglass", pretension_n=35.0, wrap_angle_deg=180.0)
    belt_90 = belt_180.model_copy(update={"wrap_angle_deg": 90.0})
    ea_eff = belt_effective_ea_n(belt_180, span_m=0.5)
    assert belt_effective_ea_n(belt_180, 0.5) > belt_effective_ea_n(
        belt_180.model_copy(update={"pretension_n": 10.0}), 0.5
    )

    k_mesh_180 = gerbert_tooth_mesh_stiffness_n_m(belt_180, ea_eff)
    k_mesh_90 = gerbert_tooth_mesh_stiffness_n_m(belt_90, ea_eff)
    assert k_mesh_180 > k_mesh_90 * 1.25

    id_5_ss = idler_shaft_and_bearing_stiffness(belt_180, double_shear=False, bracket_e_pa=2.1e9)
    id_5_ds = idler_shaft_and_bearing_stiffness(belt_180, double_shear=True, bracket_e_pa=2.1e9)
    belt_6mm = belt_180.model_copy(update={"idler_shaft_diameter_mm": 6.0})
    id_6_ss = idler_shaft_and_bearing_stiffness(belt_6mm, double_shear=False, bracket_e_pa=68.9e9)
    id_5_ss_al = idler_shaft_and_bearing_stiffness(belt_180, double_shear=False, bracket_e_pa=68.9e9)

    assert id_5_ds["k_shaft_n_m"] > id_5_ss["k_shaft_n_m"] * 8.0
    assert id_6_ss["k_shaft_n_m"] > id_5_ss_al["k_shaft_n_m"] * 1.5
    assert id_5_ss["bearing_play_um"] == pytest.approx(7.5, abs=1e-6)
    assert (
        idler_shaft_and_bearing_stiffness(
            belt_180.model_copy(update={"bearing_clearance_class": "C2"}), False, 2.1e9
        )["bearing_play_um"]
        == 3.5
    )
    assert (
        idler_shaft_and_bearing_stiffness(
            belt_180.model_copy(update={"bearing_clearance_class": "C3"}), False, 2.1e9
        )["bearing_play_um"]
        == 15.5
    )

    # Low wrap angle raises minimum slack tension required against Euler-Eytelwein cam-out
    id_90 = idler_shaft_and_bearing_stiffness(belt_90, double_shear=False, bracket_e_pa=2.1e9)
    assert id_90["min_slack_tension_n"] > id_5_ss["min_slack_tension_n"] * 1.5

    # High pretension on 5 mm single-shear ABS triggers shaft_bending_risk via deflection > 12 um
    s_bend = _corexy_base()
    s_bend.gantry.double_shear_pulleys = False
    s_bend.gantry.belt.pretension_n = 42.0
    kin_bend = analyze_kinematics(build_assembly(s_bend))
    assert kin_bend.idler_shaft_deflection_um > 12.0
    assert kin_bend.shaft_bending_risk is True


# ---------------------------------------------------------------------------
# 8. Stepper Back-EMF Efficiency Model (24V vs 48V vs 60V)
# ---------------------------------------------------------------------------


def test_stepper_back_emf_voltage_efficiency_monotonicity() -> None:
    """48V & 60V deliver higher high-RPM efficiency than 24V; eta_V strictly decreases with RPM."""
    st_24 = StepperSpec(
        holding_torque_nm=0.59,
        rated_current_a=2.0,
        run_current_a=1.6,
        phase_resistance_ohm=1.4,
        inductance_mh=3.0,
        supply_voltage_v=24.0,
    )
    st_48 = st_24.model_copy(update={"supply_voltage_v": 48.0})
    st_60 = st_24.model_copy(update={"supply_voltage_v": 60.0})

    rpms = (50.0, 250.0, 600.0, 1000.0, 1400.0)
    for st in (st_24, st_48, st_60):
        effs = [st.dynamic_voltage_efficiency(r) for r in rpms]
        for i in range(len(effs) - 1):
            assert effs[i + 1] < effs[i]

    assert st_60.dynamic_voltage_efficiency(900.0) > st_48.dynamic_voltage_efficiency(900.0) > st_24.dynamic_voltage_efficiency(900.0)
    assert st_48.dynamic_voltage_efficiency(900.0) > st_24.dynamic_voltage_efficiency(900.0) + 0.15


# ---------------------------------------------------------------------------
# 9. Pure Horizontal XY Static Deflection Maps & Modal Vertical-Z Mode Filter
# ---------------------------------------------------------------------------


def test_horizontal_xy_fea_displacements_and_modal_z_filter() -> None:
    """Static XY maps exclude gravity Z-sag; modal modes filter pure vertical Z-bounce & tag excited_axis."""
    monolith = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_monolith_awd.yaml")
    asm = build_assembly(monolith)
    st = solve_static_fea(asm)
    mo = solve_modal_fea(asm, num_modes=8)

    # Pure horizontal X/Y/XY maps have ~0 Z droop on X-beam nodes (unlike gravity-included map)
    assert "toolhead" in st.nodal_displacements_x_mm
    assert "toolhead" in st.nodal_displacements_y_mm
    assert "toolhead" in st.nodal_displacements_xy_mm
    assert abs(st.nodal_displacements_x_mm["toolhead"][2]) < 1e-4
    assert abs(st.nodal_displacements_y_mm["toolhead"][2]) < 1e-4
    assert abs(st.nodal_displacements_xy_mm["toolhead"][2]) < 1e-4
    assert abs(st.nodal_displacements_x_mm["toolhead"][0]) > abs(st.nodal_displacements_x_mm["toolhead"][1])
    assert abs(st.nodal_displacements_y_mm["toolhead"][1]) > abs(st.nodal_displacements_y_mm["toolhead"][0])

    # Every mode in primary mo.modes has excited_axis in {X, Y, XY} and is not pure vertical bed Z-bounce
    assert len(mo.modes) >= 4
    for m in mo.modes:
        assert m.excited_axis in ("X", "Y", "XY")
        assert not (m.dominant_axis == "Z" and abs(m.nodal_vectors["bed_surface"][2]) > 0.7)


# ---------------------------------------------------------------------------
# 10. FastAPI v0.10.0 Catalog & Simulate Contract
# ---------------------------------------------------------------------------


def test_api_phase1_catalog_and_thermal_structural_simulate() -> None:
    """Verify /api/health (1.1.0), /api/catalog Phase 1 metadata, and /api/simulate payload."""
    client = TestClient(create_app())

    r_health = client.get("/api/health")
    assert r_health.status_code == 200
    assert r_health.json()["version"] == "1.1.0"

    cat = client.get("/api/catalog").json()
    for key in (
        "corner_joints",
        "rail_preloads",
        "rail_block_types",
        "rail_backers",
        "bed_mounts",
        "bearing_clearances",
    ):
        assert key in cat and len(cat[key]) >= 2

    spec = _corexy_base()
    spec.frame.chamber_temp_c = 60.0
    spec.bed.target_temp_c = 110.0
    spec.bed.mount_type = "rigid_4pt"
    spec.gantry.bracket_material = "pa12_cf_printed"

    r_sim = client.post("/api/simulate", json={"spec": spec.model_dump(mode="json")})
    assert r_sim.status_code == 200
    sim = r_sim.json()

    ts = sim["thermal_structural"]
    expected_keys = {
        "chamber_temp_c",
        "delta_t_k",
        "gantry_bimetallic_bow_um",
        "gantry_backer_reduction_pct",
        "bed_target_temp_c",
        "bed_mount_type",
        "bed_gradient_bow_um",
        "bed_buckling_taco_um",
        "frame_z_drift_um",
        "bracket_modulus_retention_pct",
        "bracket_tg_warning",
        "total_thermal_z_error_um",
        "rail_newmark_gamma",
        "rail_carriage_roll_nm_per_rad",
        "rail_carriage_radial_n_per_um",
        "x_beam_modulus_retention_pct",
        "x_beam_shear_retention_pct",
        "x_beam_tg_warning",
        "x_beam_has_tg",
        "gantry_bow_direction",
        "motor_mount_temp_c",
        "bearing_temp_c",
        "bearing_cte_clearance_um",
        "bearing_cte_pressfit_warning",
        "toolhead_belt_cog_z_offset_mm",
        "toolhead_nozzle_y_lever_mm",
        "toolhead_x_roll_warning",
    }
    assert set(ts.keys()) == expected_keys
    assert ts["bracket_tg_warning"] is True
    assert ts["x_beam_has_tg"] is False
    assert ts["x_beam_tg_warning"] is False
    assert ts["gantry_bow_direction"] == "away_from_rail"
    assert ts["bed_buckling_taco_um"] > 100.0
    assert 100.0 <= ts["gantry_bimetallic_bow_um"] <= 220.0

    assert "nodal_displacements_xy_mm" in sim["static_fea"]
    assert "excited_axis" in sim["modal_fea"]["modes"][0]
    assert "tooth_mesh_stiffness_n_per_mm" in sim["kinematics"]
    assert "bearing_play_um" in sim["kinematics"]
