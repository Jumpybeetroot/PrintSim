"""Regression tests for F2, H1-H4 structural assembly fixes and hardware preset accuracy."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from printsim.engine import evaluate_printer
from printsim.fea import assemble_global_matrices, solve_modal_fea, solve_static_fea
from printsim.fea.shell_mitc4 import fastener_spring_matrices
from printsim.geometry import build_assembly
from printsim.schema import BED_MOUNT_CATALOG, PrinterSpec

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def test_f2_bed_mount_spring_eccentric_offset_and_zero_rigid_pitch_energy() -> None:
    """F2: bed_mount springs use exact physical 3D offsets (including Z-stack t_bed + standoff + half 2020) with zero rigid pitch/roll energy."""
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    asm = build_assembly(spec)
    nmap = asm.node_map

    bed_springs = [s for s in asm.fastener_springs if s.bay == "bed_mount"]
    assert {s.id for s in bed_springs} == {
        "bed_mount_spring_fl",
        "bed_mount_spring_fr",
        "bed_mount_spring_r",
    }

    t_bed_m = float(spec.bed.plate_thickness_mm) * 1e-3
    h_standoff_m = float(BED_MOUNT_CATALOG[spec.bed.mount_type]["standoff_height_mm"]) * 1e-3
    half_subframe_m = 0.010
    expected_dz = -(t_bed_m + h_standoff_m + half_subframe_m)
    dz_actual = float(nmap["bed_frame"].position_m[2] - nmap["bed_surface"].position_m[2])
    assert dz_actual == pytest.approx(expected_dz, abs=1e-9)

    for sp in bed_springs:
        assert sp.frame_node_id == "bed_surface"
        r_vec = nmap[sp.shell_node_id].position_m - nmap["bed_surface"].position_m
        assert sp.offset_m == pytest.approx(tuple(float(c) for c in r_vec), abs=1e-9)
        assert sp.offset_m[2] == pytest.approx(expected_dz, abs=1e-9)

        K_sp = fastener_spring_matrices(
            sp.k_shear_n_m,
            sp.k_normal_n_m,
            sp.k_rot_nm_rad,
            np.array(sp.normal_vec, dtype=float),
            np.array(sp.offset_m, dtype=float),
        )
        # Pure 3D rigid-body rotation theta = (theta_x, theta_y, 0) centered at bed_surface:
        # u_anchor = theta x r_vec -> zero strain energy across the eccentric 6-DOF spring.
        theta = np.array([0.015, 0.02, 0.0], dtype=float)
        u_anchor = np.cross(theta, r_vec)
        u_rigid = np.zeros(12, dtype=float)
        u_rigid[0:3] = u_anchor
        u_rigid[3:6] = theta
        u_rigid[9:12] = theta
        strain_energy_j = 0.5 * float(u_rigid @ K_sp @ u_rigid)
        assert strain_energy_j == pytest.approx(0.0, abs=1e-10)


def test_h1_voron_v24_flying_gantry_joints_z_variation_and_trident_comparison() -> None:
    """H1: Voron 2.4 has separate g_* and col_g_* nodes, 4 z_joint_* springs, Z variation, and Trident > V2.4."""
    v24 = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    assert v24.is_flying_gantry

    asm_lo = build_assembly(v24, toolhead_rel_pos=(0.5, 0.5, 0.1))
    asm_hi = build_assembly(v24, toolhead_rel_pos=(0.5, 0.5, 0.9))

    for asm in (asm_lo, asm_hi):
        nmap = asm.node_map
        for tag in ("fl", "fr", "rl", "rr"):
            col_n = nmap[f"col_g_{tag}"]
            gan_n = nmap[f"g_{tag}"]
            assert col_n.role == "frame"
            assert gan_n.role == "gantry"
            assert gan_n.position_m == pytest.approx(col_n.position_m, abs=1e-9)

            zj = next(s for s in asm.fastener_springs if s.id == f"z_joint_{tag}")
            assert zj.bay == "z_joint"
            if f"z_joint_mid_{tag}" in nmap:
                assert zj.shell_node_id == f"z_joint_mid_{tag}"
                cad_zj = next(se for se in asm.superelements if se.id == f"cad_z_joint_{tag}")
                assert cad_zj.port_node_ids == (f"col_g_{tag}", f"z_joint_mid_{tag}")
            else:
                assert zj.shell_node_id == f"col_g_{tag}"
            assert zj.frame_node_id == f"g_{tag}"
            # CAL-10: two-span height-dependent Z-belt elasticity in series with voron_24_z_drive_* CAD CMS
            assert 1.5e5 < zj.k_normal_n_m < 2.4e5
            assert zj.k_shear_n_m == pytest.approx(1.2e6, rel=1e-6)
            assert zj.k_rot_nm_rad == pytest.approx(65.0, rel=1e-6)

            # Flying-gantry bed arms attach to base rail split nodes b_bed_*, not grounded feet b_*
            bed_arm = next(m for m in asm.members if m.id == f"bed_arm_{tag}")
            assert bed_arm.start_id == f"b_bed_{tag}"
            assert not nmap[bed_arm.start_id].fixed

    z_lo = float(asm_lo.node_map["g_fl"].position_m[2])
    z_hi = float(asm_hi.node_map["g_fl"].position_m[2])
    assert z_hi > z_lo + 0.15

    mod_lo = solve_modal_fea(asm_lo)
    mod_hi = solve_modal_fea(asm_hi)
    assert abs(mod_lo.fundamental_freq_hz - mod_hi.fundamental_freq_hz) > 5.0
    assert abs(mod_lo.shaper_x.effective_freq_hz - mod_hi.shaper_x.effective_freq_hz) > 2.0

    # Equally framed fixed-gantry Trident is stiffer than flying-gantry Voron 2.4
    trid_eq = v24.model_copy(deep=True)
    trid_eq.bed.z_drive = "leadscrew_triple"
    trid_eq.bed.z_leadscrew_type = "auto"
    asm_trid = build_assembly(trid_eq)
    asm_v24 = build_assembly(v24)
    st_trid, mo_trid = solve_static_fea(asm_trid), solve_modal_fea(asm_trid)
    st_v24, mo_v24 = solve_static_fea(asm_v24), solve_modal_fea(asm_v24)
    assert st_trid.toolhead_stiffness_x_n_per_mm > st_v24.toolhead_stiffness_x_n_per_mm
    assert st_trid.toolhead_stiffness_y_n_per_mm > st_v24.toolhead_stiffness_y_n_per_mm
    assert st_trid.toolhead_stiffness_z_n_per_mm > st_v24.toolhead_stiffness_z_n_per_mm
    assert mo_trid.fundamental_freq_hz > mo_v24.fundamental_freq_hz


def test_h2_corner_gussets_has_zero_diagonal_braces_and_modestly_stiffens_joints() -> None:
    """H2: cnc_corner_gussets emits zero diagonal brace rods on CoreXY/Bedslinger, unlike rear_flat_bar_x."""
    for fname in ("corexy_voron_style.yaml", "voron_trident_300.yaml", "doomcube_4040.yaml"):
        spec = PrinterSpec.from_yaml(EXAMPLES_DIR / fname)
        assert spec.frame.bracing == "cnc_corner_gussets"
        asm = build_assembly(spec)
        assert [m for m in asm.members if m.role == "brace" or m.id.startswith("brace_")] == []

    # Open-panel blind-joint CoreXY: cnc_corner_gussets has zero brace rods yet increases stiffness over bracing="none"
    open_none = PrinterSpec(architecture="corexy")
    open_none.frame.enclosed_panels = False
    open_none.frame.corner_joint_type = "blind_joint"
    open_none.frame.bracing = "none"

    open_gusset = open_none.model_copy(deep=True)
    open_gusset.frame.bracing = "cnc_corner_gussets"

    open_cross = open_none.model_copy(deep=True)
    open_cross.frame.bracing = "rear_flat_bar_x"

    asm_none = build_assembly(open_none)
    asm_gusset = build_assembly(open_gusset)
    asm_cross = build_assembly(open_cross)

    assert [m for m in asm_gusset.members if m.role == "brace"] == []
    assert len([m for m in asm_cross.members if m.role == "brace"]) >= 2
    assert solve_modal_fea(asm_gusset).fundamental_freq_hz > solve_modal_fea(asm_none).fundamental_freq_hz

    # Cartesian bedslinger with cnc_corner_gussets also emits zero full-face diagonal brace rods
    bs = PrinterSpec.from_yaml(EXAMPLES_DIR / "cartesian_bedslinger.yaml")
    bs.frame.bracing = "cnc_corner_gussets"
    asm_bs = build_assembly(bs)
    assert [m for m in asm_bs.members if m.role == "brace" or m.id.startswith("brace_")] == []


def test_h3_delta_6_rod_parallelogram_honors_delta_radius_and_no_pitch_modes() -> None:
    """H3: Delta uses 6 parallel-pair rods, honors delta_radius_mm, and eliminates unconstrained effector modes."""
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "delta_racer.yaml")
    asm = build_assembly(spec)
    nmap = asm.node_map

    arms = [m for m in asm.members if m.id.startswith("arm_")]
    expected_ids = {f"arm_{i}_{s}" for i in range(3) for s in ("a", "b")}
    assert {m.id for m in arms} == expected_ids

    expected_len_m = spec.kinematics.delta_arm_length_mm * 1e-3
    for m in arms:
        assert m.is_truss is True
        assert m.length_m(nmap) == pytest.approx(expected_len_m, abs=1e-6)

    for i in range(3):
        sep_eff = float(np.linalg.norm(nmap[f"eff_{i}_a"].position_m - nmap[f"eff_{i}_b"].position_m))
        sep_car = float(np.linalg.norm(nmap[f"c_{i}_a"].position_m - nmap[f"c_{i}_b"].position_m))
        assert sep_eff == pytest.approx(0.044, abs=1e-6)
        assert sep_car == pytest.approx(0.044, abs=1e-6)

    # Verify delta_radius_mm controls actual tower radial spacing
    r_delta_nom = float(np.linalg.norm(nmap["b_0"].position_m[:2] - nmap["toolhead"].position_m[:2]))
    wide = spec.model_copy(deep=True)
    wide.kinematics.delta_radius_mm = spec.kinematics.delta_radius_mm + 25.0
    asm_wide = build_assembly(wide)
    nmap_w = asm_wide.node_map
    r_delta_wide = float(np.linalg.norm(nmap_w["b_0"].position_m[:2] - nmap_w["toolhead"].position_m[:2]))
    assert (r_delta_wide - r_delta_nom) == pytest.approx(0.025, abs=1e-6)
    for m in [m for m in asm_wide.members if m.id.startswith("arm_")]:
        assert m.length_m(nmap_w) == pytest.approx(expected_len_m, abs=1e-6)

    # Check SPD stiffness and absence of soft/unconstrained pitch modes
    K, _, _, _, free = assemble_global_matrices(asm)
    evals = np.linalg.eigvalsh(K[np.ix_(free, free)])
    assert float(evals[0]) > 10.0

    modal = solve_modal_fea(asm, num_modes=6)
    assert modal.fundamental_freq_hz > 80.0
    xy_modes = [m for m in modal.modes if (m.x_axis_share_pct + m.y_axis_share_pct) > 10.0]
    assert len(xy_modes) >= 2
    assert abs(xy_modes[0].frequency_hz - xy_modes[1].frequency_hz) < 1.0


def test_h4_bedslinger_nonfixed_portal_base_and_moving_y_bed() -> None:
    """H4: Bedslinger p_bl/p_br are not fixed=True and moving Y bed tracks toolhead_rel_pos[1]."""
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "cartesian_bedslinger.yaml")
    asm_front = build_assembly(spec, toolhead_rel_pos=(0.5, 0.15, 0.5))
    asm_rear = build_assembly(spec, toolhead_rel_pos=(0.5, 0.85, 0.5))

    for nid in ("p_bl", "p_br"):
        assert asm_front.node_map[nid].fixed is False
    for nid in ("b_fl", "b_fr", "b_rl", "b_rr"):
        assert asm_front.node_map[nid].fixed is True

    y_bed_front = float(asm_front.node_map["bed_surface"].position_m[1])
    y_bed_rear = float(asm_rear.node_map["bed_surface"].position_m[1])
    assert abs(y_bed_front - y_bed_rear) > 0.10
    assert float(asm_front.node_map["bed_rail_l"].position_m[1]) == pytest.approx(y_bed_front, abs=1e-9)
    assert float(asm_rear.node_map["bed_rail_l"].position_m[1]) == pytest.approx(y_bed_rear, abs=1e-9)


def test_all_eight_example_presets_evaluate_cleanly_with_accurate_hardware_specs() -> None:
    """Verify all 8 reference YAML specs match corrected hardware catalogs and evaluate collision-free."""
    expected_files = [
        "cartesian_bedslinger.yaml",
        "corexy_voron_style.yaml",
        "delta_racer.yaml",
        "doomcube_4040.yaml",
        "hyper_corexy_carbon.yaml",
        "voron_monolith_awd.yaml",
        "voron_trident_300.yaml",
        "vzbot_330_awd.yaml",
    ]
    for fname in expected_files:
        spec = PrinterSpec.from_yaml(EXAMPLES_DIR / fname)
        rep = evaluate_printer(spec)
        assert not rep.collision.has_collision, f"{fname} collisions: {rep.collision.issues}"
        assert math.isfinite(rep.modal_fea.fundamental_freq_hz) and rep.modal_fea.fundamental_freq_hz > 15.0
        assert math.isfinite(rep.static_fea.combined_max_deflection_um)
        assert math.isfinite(rep.static_fea.toolhead_stiffness_z_n_per_mm)
        assert math.isfinite(rep.thermal.max_volumetric_flow_mm3_s)

    v24 = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    assert v24.frame.panel_mounting == "clip_enclosure"
    assert v24.frame.panel_material == "acrylic_pmma"
    assert v24.gantry.y_rail_profile == "mgn9_rail"
    assert v24.toolhead.preset_id == "stealthburner"
    assert v24.bed.mass_g == pytest.approx(2800.0)

    monolith = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_monolith_awd.yaml")
    assert monolith.gantry.belt_routing == "monolith_short"
    assert monolith.toolhead.preset_id == "archetype_cpap"
    assert monolith.kinematics.xy_stepper.preset_id == "ldo_2804ah"
    assert monolith.kinematics.xy_stepper.holding_torque_nm == pytest.approx(0.65)
    assert monolith.kinematics.xy_stepper.inductance_mh == pytest.approx(0.6)
    assert monolith.kinematics.xy_stepper.phase_resistance_ohm == pytest.approx(0.7)
    assert monolith.bed.mass_g == pytest.approx(2800.0)

    delta = PrinterSpec.from_yaml(EXAMPLES_DIR / "delta_racer.yaml")
    assert delta.kinematics.xy_stepper.preset_id == "ldo_09deg_2504"
    assert delta.kinematics.xy_stepper.steps_per_rev == 400
    assert tuple(delta.toolhead.nozzle_offset_mm) == (0.0, 0.0, -45.0)
    assert tuple(delta.toolhead.cog_offset_mm) == (0.0, 0.0, -18.0)
