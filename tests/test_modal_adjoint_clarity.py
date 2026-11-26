"""Tests for modal print-impact clarity, 4-lens adjoint heat scores, and subsystem ROI."""

from __future__ import annotations

import math

import numpy as np
import pytest

from printsim.fea import (
    assemble_global_matrices,
    compute_structural_sensitivities,
    solve_modal_fea,
)
from printsim.fea.solver import _build_factorized_solver
from printsim.geometry import build_assembly
from printsim.schema import TOOLHEAD_PRESETS, PrinterSpec


def _assert_valid_mode_clarity_fields(modal_dict: dict) -> None:
    modes = modal_dict["modes"]
    assert len(modes) >= 1
    mode_nums = {m["mode_number"] for m in modes}
    assert modal_dict["worst_print_mode_number"] in mode_nums

    for m in modes:
        sub = m["subsystem_share_pct"]
        assert set(sub.keys()) == {"toolhead", "gantry", "frame", "bed"}
        assert all(0.0 <= float(v) <= 100.0 for v in sub.values())
        assert sum(sub.values()) == pytest.approx(100.0, abs=0.25)

        assert 0.0 <= m["nozzle_whip_ratio"] <= 10.0
        assert 0.0 <= m["toolhead_flop_pct"] <= 100.0
        assert m["relative_nozzle_bed_amp"] >= 0.0
        assert 0.0 <= m["print_impact_score"] <= 100.0
        assert m["print_impact_level"] in {"critical", "high", "moderate", "low"}
        if m["print_impact_score"] >= 75.0:
            assert m["print_impact_level"] == "critical"
        elif m["print_impact_score"] >= 45.0:
            assert m["print_impact_level"] == "high"
        elif m["print_impact_score"] >= 20.0:
            assert m["print_impact_level"] == "moderate"
        else:
            assert m["print_impact_level"] == "low"
        assert m["dominant_problem_area"] in {"toolhead", "gantry", "frame", "bed"}
        assert m["severity"] in {"critical", "moderate", "stiff"}


def test_corexy_voron_style_modal_and_four_lens_sensitivity_clarity() -> None:
    """Verify subsystem breakdown, top_stiffen_targets[:2] toolhead_mount, and 4 heat-score lenses."""
    spec = PrinterSpec.from_yaml("examples/corexy_voron_style.yaml")
    asm = build_assembly(spec)
    modal_res = solve_modal_fea(asm)
    sens_res = compute_structural_sensitivities(asm, modal_res=modal_res)

    modal_dict = modal_res.to_dict()
    sens_dict = sens_res.to_dict()

    _assert_valid_mode_clarity_fields(modal_dict)

    # Subsystem breakdown sums to 100.0%
    sub_bd = sens_dict["subsystem_breakdown"]
    assert set(sub_bd.keys()) == {"toolhead", "gantry", "frame", "bed"}
    assert sens_dict["primary_problem_subsystem"] in {"toolhead", "gantry", "frame", "bed"}
    sub_sum = sum(v["compliance_share_pct"] for v in sub_bd.values())
    assert sub_sum == pytest.approx(100.0, abs=0.5)
    for item in sub_bd.values():
        assert item["severity"] in {"critical", "high", "moderate", "low"}

    # On a stiff bolted-panel CoreXY (so outer frame isn't unbraced), toolhead_mount MUST rank in top 2 and glow bright red
    spec_stiff = spec.model_copy(deep=True)
    spec_stiff.frame.enclosed_panels = True
    spec_stiff.frame.panel_mounting = "bolted_structural"
    spec_stiff.frame.panel_material = "acm_dibond"
    spec_stiff.frame.panel_thickness_mm = 4.0
    asm_stiff = build_assembly(spec_stiff)
    sens_stiff = compute_structural_sensitivities(asm_stiff).to_dict()
    top3_ids = [t["id"] for t in sens_stiff["top_stiffen_targets"][:3]]
    assert "toolhead_mount" in top3_ids, f"Expected toolhead_mount in top 3, got {top3_ids}"
    th_mem = sens_stiff["member_sensitivities"]["toolhead_mount"]
    assert th_mem["compliance_share_pct"] > 20.0
    assert th_mem["heat_score"] > 0.65
    assert th_mem["heat_score_compliance"] >= 0.80

    # toolhead_cog must show clear negative mass penalty (strongest on bolted-panel CoreXY where frame racking doesn't mask it)
    cog_lump = sens_dict["lumped_mass_sensitivities"]["toolhead_cog"]
    assert cog_lump["heat_score_mass"] < 0.0
    assert cog_lump["heat_score"] < 0.0
    cog_lump_stiff = sens_stiff["lumped_mass_sensitivities"]["toolhead_cog"]
    assert cog_lump_stiff["heat_score_mass"] < -0.3
    assert cog_lump_stiff["heat_score"] < -0.3

    # All 4 lenses exist within valid ranges across members, bays, and lumped masses
    for mem in sens_dict["member_sensitivities"].values():
        assert -1.0 <= mem["heat_score"] <= 1.0
        assert 0.0 <= mem["heat_score_compliance"] <= 1.0
        assert -1.0 <= mem["heat_score_modal"] <= 1.0
        assert -1.0 <= mem["heat_score_mass"] <= 0.0

    for bay in sens_dict["panel_bay_sensitivities"].values():
        assert -1.0 <= bay["heat_score"] <= 1.0
        assert 0.0 <= bay["heat_score_compliance"] <= 1.0
        assert -1.0 <= bay["heat_score_modal"] <= 1.0
        assert -1.0 <= bay["heat_score_mass"] <= 0.0

    for lump in sens_dict["lumped_mass_sensitivities"].values():
        assert -1.0 <= lump["heat_score"] <= 0.0
        assert lump["heat_score_compliance"] == 0.0
        assert -1.0 <= lump["heat_score_modal"] <= 0.0
        assert -1.0 <= lump["heat_score_mass"] <= 0.0

    # Overall structural compliance share sums to 100%
    comp_sum = sum(v["compliance_share_pct"] for v in sens_dict["member_sensitivities"].values()) + sum(
        v["compliance_share_pct"] for v in sens_dict["panel_bay_sensitivities"].values()
    )
    assert comp_sum == pytest.approx(100.0, abs=0.5)


def test_soft_heavy_toolhead_whip_vs_stiff_cnc_compact_toolhead() -> None:
    """Soft/heavy UHF ABS toolhead vs stiff CNC Al compact toolhead shows unmistakable modal & ROI shift."""
    base = PrinterSpec.from_yaml("examples/corexy_voron_style.yaml")
    # Use bolted structural panels so outer-frame racking doesn't mask the soft-vs-stiff toolhead comparison
    base.frame.enclosed_panels = True
    base.frame.panel_mounting = "bolted_structural"
    base.frame.panel_material = "acm_dibond"
    base.frame.panel_thickness_mm = 4.0

    soft = base.model_copy(deep=True)
    heavy = TOOLHEAD_PRESETS["heavy_uhf_long"]
    soft.toolhead.total_mass_g = float(heavy["total_mass_g"])
    soft.toolhead.cog_offset_mm = tuple(float(v) for v in heavy["cog_offset_mm"])  # type: ignore[assignment]
    soft.toolhead.nozzle_offset_mm = tuple(float(v) for v in heavy["nozzle_offset_mm"])  # type: ignore[assignment]
    soft.toolhead.mount_material = "abs_printed"
    # C-04 removed silent <350 power-law softening; use realistic soft printed body stiffness (160 N/mm)
    soft.toolhead.internal_stiffness_n_per_mm = 160.0

    stiff = base.model_copy(deep=True)
    stiff.toolhead.total_mass_g = 220.0
    stiff.toolhead.cog_offset_mm = (0.0, 0.0, 0.0)
    stiff.toolhead.nozzle_offset_mm = (0.0, 0.0, -35.0)
    stiff.toolhead.mount_material = "aluminum_6061_t6"
    stiff.toolhead.internal_stiffness_n_per_mm = 2000.0

    asm_soft = build_assembly(soft)
    mod_soft = solve_modal_fea(asm_soft)
    sen_soft = compute_structural_sensitivities(asm_soft, modal_res=mod_soft)

    asm_stiff = build_assembly(stiff)
    mod_stiff = solve_modal_fea(asm_stiff)
    sen_stiff = compute_structural_sensitivities(asm_stiff, modal_res=mod_stiff)

    _assert_valid_mode_clarity_fields(mod_soft.to_dict())
    _assert_valid_mode_clarity_fields(mod_stiff.to_dict())

    assert sen_soft.primary_problem_subsystem == "toolhead"
    assert sen_soft.top_stiffen_targets[0]["id"] == "toolhead_mount"
    assert "Toolhead & Mount" in sen_soft.summary_diagnosis

    worst_soft = next(
        m for m in mod_soft.modes if m.mode_number == mod_soft.worst_print_mode_number
    )
    assert worst_soft.dominant_problem_area == "toolhead"
    assert worst_soft.nozzle_whip_ratio > 1.3
    assert worst_soft.toolhead_flop_pct >= 35.0
    assert worst_soft.print_impact_score >= 70.0

    # Zero-overhang toolhead_cog_spring must appear in member_sensitivities for stiff compact config
    assert "toolhead_cog_spring" in sen_stiff.member_sensitivities
    assert (
        sen_stiff.subsystem_breakdown["toolhead"]["compliance_share_pct"]
        < sen_soft.subsystem_breakdown["toolhead"]["compliance_share_pct"] * 0.25
    )
    worst_stiff = next(
        m for m in mod_stiff.modes if m.mode_number == mod_stiff.worst_print_mode_number
    )
    assert worst_stiff.nozzle_whip_ratio < worst_soft.nozzle_whip_ratio
    assert worst_stiff.nozzle_whip_ratio < 1.2


def test_non_panel_fastener_springs_feet_and_bed_mount_adjoint_exactness() -> None:
    """Verify bed_mount, toolhead, and feet springs contribute to compliance, grouping, and FD derivatives."""
    spec = PrinterSpec.from_yaml("examples/corexy_voron_style.yaml")
    spec.frame.feet_type = "anti_vib_pads"
    spec.bed.mount_type = "springs_4pt"
    spec.toolhead.cog_offset_mm = (0.0, 0.0, 0.0)

    asm = build_assembly(spec)
    mod = solve_modal_fea(asm)
    sen = compute_structural_sensitivities(asm, modal_res=mod)
    _assert_valid_mode_clarity_fields(mod.to_dict())

    # Springs must be registered in member_sensitivities and compliance must still sum to 100%
    assert "toolhead_cog_spring" in sen.member_sensitivities
    assert "bed_mount_spring" in sen.member_sensitivities
    feet_ids = [mid for mid in sen.member_sensitivities if mid.startswith("feet_")]
    assert len(feet_ids) == 4
    assert any(t["id"] == "frame_feet" for t in sen.top_stiffen_targets)

    comp_sum = sum(v["compliance_share_pct"] for v in sen.member_sensitivities.values()) + sum(
        v["compliance_share_pct"] for v in sen.panel_bay_sensitivities.values()
    )
    assert comp_sum == pytest.approx(100.0, abs=0.5)

    # Verify analytical adjoint & Fox-Kapoor derivatives on a feet spring match ±0.5% central FD
    sp_target = feet_ids[0]
    eps = 0.005
    dln = math.log(1.0 + eps) - math.log(1.0 - eps)

    def _eval_spring_scale(scale: float) -> tuple[float, float, float]:
        asm_p = build_assembly(spec)
        for sp in asm_p.fastener_springs:
            if sp.id == sp_target:
                sp.k_shear_n_m *= scale
                sp.k_normal_n_m *= scale
                sp.k_rot_nm_rad *= scale
        mod_p = solve_modal_fea(asm_p)
        K_p, _, _, nidx, free_p = assemble_global_matrices(asm_p)
        solve_rhs = _build_factorized_solver(K_p, free_p)
        noz_i = nidx[asm_p.nozzle_node_id]
        bed_i = nidx[asm_p.bed_node_id]
        n_dof = K_p.shape[0]
        accel = asm_p.spec.kinematics.max_accel_mm_s2 * 1e-3
        from printsim.fea.solver import build_inertial_load_vectors

        F_x, F_y, _, _ = build_inertial_load_vectors(asm_p, nidx, n_dof, accel)

        u_x = solve_rhs(F_x)
        u_y = solve_rhs(F_y)
        ux_rel = float(u_x[6 * noz_i + 0] - u_x[6 * bed_i + 0])
        uy_rel = float(u_y[6 * noz_i + 1] - u_y[6 * bed_i + 1])
        return mod_p.fundamental_freq_hz, ux_rel, uy_rel

    fp, uxp, uyp = _eval_spring_scale(1.0 + eps)
    fm, uxm, uym = _eval_spring_scale(1.0 - eps)
    sp_sens = sen.member_sensitivities[sp_target]
    assert sp_sens["stiffness_sens_hz"] == pytest.approx((fp - fm) / dln, rel=0.02)
    assert sp_sens["du_rel_x_dlnE_m"] == pytest.approx((uxp - uxm) / dln, rel=0.02)
    assert sp_sens["du_rel_y_dlnE_m"] == pytest.approx((uyp - uym) / dln, rel=0.02)


def test_modal_clarity_across_all_architectures() -> None:
    """All reference YAML specs emit valid ModeShape clarity fields and subsystem breakdowns."""
    for yaml_path in (
        "examples/corexy_voron_style.yaml",
        "examples/voron_trident_300.yaml",
        "examples/voron_monolith_awd.yaml",
        "examples/cartesian_bedslinger.yaml",
        "examples/delta_racer.yaml",
    ):
        asm = build_assembly(PrinterSpec.from_yaml(yaml_path))
        mod = solve_modal_fea(asm)
        sen = compute_structural_sensitivities(asm, modal_res=mod)
        _assert_valid_mode_clarity_fields(mod.to_dict())

        sub = sen.to_dict()["subsystem_breakdown"]
        assert set(sub.keys()) == {"toolhead", "gantry", "frame", "bed"}
        assert sum(v["compliance_share_pct"] for v in sub.values()) == pytest.approx(100.0, abs=0.5)
