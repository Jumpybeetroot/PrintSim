"""Physics verification tests for the expanded industrial belt catalog, bend-radius, and creep models."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from printsim.engine.evaluator import evaluate_printer
from printsim.geometry.builder import build_assembly
from printsim.kinematics.solvers import analyze_kinematics, compute_drive_stiffness_details_n_per_m
from printsim.schema import BELT_CATALOG, BeltSpec, PrinterSpec

MONOLITH_YAML = Path(__file__).resolve().parents[1] / "examples" / "voron_monolith_awd.yaml"


def test_belt_catalog_ea_and_datasheet_ratios() -> None:
    """Verify datasheet EA / C_spez values and monotonic width/profile ordering."""
    b_gt2_6 = BeltSpec(type="gt2_6mm")
    b_gt2_9 = BeltSpec(type="gt2_9mm")
    b_gt2_12 = BeltSpec(type="gt2_12mm")
    b_gt2_15 = BeltSpec(type="gt2_15mm")
    b_gt2_dual9 = BeltSpec(type="gt2_dual_9mm")
    b_3gt_9 = BeltSpec(type="3gt_9mm")
    b_3gt_15 = BeltSpec(type="3gt_15mm")
    b_at3_10 = BeltSpec(type="at3_10mm")
    b_at3_16 = BeltSpec(type="at3_16mm")
    b_at3_g3 = BeltSpec(type="at3_gen3_16mm")
    b_htd3m_15 = BeltSpec(type="htd3m_15mm")
    b_at5_16 = BeltSpec(type="at5_16mm")

    # Existing GT2 baseline preserved exactly
    assert b_gt2_6.stiffness_ea_n == pytest.approx(50000.0)
    assert b_gt2_9.stiffness_ea_n == pytest.approx(75000.0)
    assert b_gt2_12.stiffness_ea_n == pytest.approx(100000.0)
    assert b_gt2_15.stiffness_ea_n == pytest.approx(125000.0)
    assert b_gt2_dual9.stiffness_ea_n == pytest.approx(150000.0)

    # Gates 3GT / 2GT unit-strain modulus ratio = 30k / 18k = 1.667x per mm width
    assert b_3gt_9.pitch_mm == pytest.approx(3.0)
    assert b_3gt_9.core == "fiberglass"
    assert b_3gt_9.stiffness_ea_n / b_gt2_9.stiffness_ea_n == pytest.approx(30.0 / 18.0, rel=0.01)
    assert b_3gt_15.stiffness_ea_n > b_gt2_15.stiffness_ea_n

    # Elatech / ContiTech steel-cord TPU datasheet C_spez values
    assert b_at3_10.pitch_mm == pytest.approx(3.0)
    assert b_at3_10.core == "steel"
    assert b_at3_10.body_material == "tpu"
    assert b_at3_10.stiffness_ea_n == pytest.approx(80000.0)
    assert b_at3_16.stiffness_ea_n == pytest.approx(135000.0)
    assert b_at3_g3.stiffness_ea_n == pytest.approx(250000.0)
    assert b_htd3m_15.stiffness_ea_n == pytest.approx(127500.0)
    assert b_at5_16.pitch_mm == pytest.approx(5.0)
    assert b_at5_16.stiffness_ea_n == pytest.approx(280000.0)
    assert len(BELT_CATALOG) >= 13


def test_pulley_radius_inverse_square_motor_stiffness_penalty() -> None:
    """Verify k_motor = K_theta / r_p^2 drops exactly as (p_2 / p_1)^2 for equal tooth count."""
    spec_2gt = PrinterSpec.from_yaml(MONOLITH_YAML)
    spec_2gt.gantry.belt = BeltSpec(type="gt2_9mm", pulley_teeth=20, pretension_n=65.0)
    d_2gt = compute_drive_stiffness_details_n_per_m(build_assembly(spec_2gt))

    spec_at5 = PrinterSpec.from_yaml(MONOLITH_YAML)
    spec_at5.gantry.belt = BeltSpec(
        type="at5_16mm", pulley_teeth=20, pretension_n=120.0, back_idler_diameter_mm=60.0
    )
    d_at5 = compute_drive_stiffness_details_n_per_m(build_assembly(spec_at5))

    # 5mm vs 2mm pitch at 20T -> 2.5x larger radius -> ~6.25x lower reflected electromagnetic motor stiffness
    # (slight additional shift from lower operating RPM at larger pulley radius, so check > 5.0x)
    ratio = float(d_2gt["k_motor_axis_n_m"]) / float(d_at5["k_motor_axis_n_m"])
    assert ratio > 5.0
    assert float(d_at5["k_belt_cord_n_m"]) > 3.0 * float(d_2gt["k_belt_cord_n_m"])


def test_reverse_bend_back_idler_and_min_pulley_fatigue_checks() -> None:
    """Verify minimum pulley teeth and reverse-bend back-idler diameter rules for AT3 and AT5."""
    # AT3 16mm on 16T pulley (< 20T reverse-bend limit) -> fatigue flagged
    spec_at3_bad = PrinterSpec.from_yaml(MONOLITH_YAML)
    spec_at3_bad.gantry.belt = BeltSpec(type="at3_16mm", pulley_teeth=16, pretension_n=90.0)
    kin_bad = analyze_kinematics(build_assembly(spec_at3_bad))
    assert kin_bad.belt_bend_fatigue_risk is True
    assert kin_bad.steel_core_fatigue_risk is True

    # AT3 16mm on 20T pulley with 20mm back-idler -> passes bend-radius limit
    spec_at3_ok = PrinterSpec.from_yaml(MONOLITH_YAML)
    spec_at3_ok.gantry.belt = BeltSpec(
        type="at3_16mm", pulley_teeth=20, pretension_n=90.0, back_idler_diameter_mm=20.0
    )
    kin_ok = analyze_kinematics(build_assembly(spec_at3_ok))
    assert kin_ok.belt_bend_fatigue_risk is False
    assert kin_ok.steel_core_fatigue_risk is False

    # AT5 16mm on 20T pulley (31.8mm PD default back-idler < 60mm min) -> fatigue flagged
    spec_at5_bad = PrinterSpec.from_yaml(MONOLITH_YAML)
    spec_at5_bad.gantry.belt = BeltSpec(type="at5_16mm", pulley_teeth=20, pretension_n=120.0)
    kin_at5_bad = analyze_kinematics(build_assembly(spec_at5_bad))
    assert kin_at5_bad.belt_bend_fatigue_risk is True
    assert kin_at5_bad.belt_min_back_idler_mm == pytest.approx(60.0)

    # AT5 16mm with explicit 60mm back-idlers -> passes bend-radius limit
    spec_at5_ok = PrinterSpec.from_yaml(MONOLITH_YAML)
    spec_at5_ok.gantry.belt = BeltSpec(
        type="at5_16mm", pulley_teeth=20, pretension_n=120.0, back_idler_diameter_mm=60.0
    )
    kin_at5_ok = analyze_kinematics(build_assembly(spec_at5_ok))
    assert kin_at5_ok.belt_bend_fatigue_risk is False


def test_tpu_vs_epdm_chamber_thermal_softening_and_aramid_creep() -> None:
    """Verify TPU belts soften & trigger creep risk in hot chambers (>55 C), while EPDM stays rigid."""
    # Open-frame 25 C AT3 16mm: 100% retention, no creep risk
    spec_cold = PrinterSpec.from_yaml(MONOLITH_YAML)
    spec_cold.frame.chamber_temp_c = 25.0
    spec_cold.gantry.belt = BeltSpec(
        type="at3_16mm", pulley_teeth=20, pretension_n=90.0, back_idler_diameter_mm=20.0
    )
    k_cold = analyze_kinematics(build_assembly(spec_cold))
    assert k_cold.belt_thermal_retention_pct == pytest.approx(100.0)
    assert k_cold.belt_creep_risk is False

    # Heated 65 C chamber AT3 16mm (TPU): softens noticeably & flags creep risk
    spec_hot_tpu = PrinterSpec.from_yaml(MONOLITH_YAML)
    spec_hot_tpu.frame.chamber_temp_c = 65.0
    spec_hot_tpu.gantry.belt = BeltSpec(
        type="at3_16mm", pulley_teeth=20, pretension_n=90.0, back_idler_diameter_mm=20.0
    )
    r_hot_tpu = evaluate_printer(spec_hot_tpu)
    assert r_hot_tpu.kinematics.belt_thermal_retention_pct < 80.0
    assert r_hot_tpu.kinematics.belt_creep_risk is True
    assert r_hot_tpu.kinematics.tooth_mesh_stiffness_n_per_mm < k_cold.tooth_mesh_stiffness_n_per_mm
    assert any("TPU belt matrix thermal softening" in d for d in r_hot_tpu.diagnostics)

    # Heated 65 C chamber Gates 3GT 15mm (EPDM/fiberglass): stays 100% rigid, zero creep risk
    spec_hot_epdm = PrinterSpec.from_yaml(MONOLITH_YAML)
    spec_hot_epdm.frame.chamber_temp_c = 65.0
    spec_hot_epdm.gantry.belt = BeltSpec(
        type="3gt_15mm", pulley_teeth=16, pretension_n=110.0, back_idler_diameter_mm=20.0
    )
    k_hot_epdm = analyze_kinematics(build_assembly(spec_hot_epdm))
    assert k_hot_epdm.belt_thermal_retention_pct == pytest.approx(100.0)
    assert k_hot_epdm.belt_creep_risk is False

    # Aramid core under sustained preload flags creep risk
    spec_aramid = PrinterSpec.from_yaml(MONOLITH_YAML)
    spec_aramid.gantry.belt = BeltSpec(type="gt2_9mm", core="aramid", pretension_n=55.0)
    r_aramid = evaluate_printer(spec_aramid)
    assert r_aramid.kinematics.belt_creep_risk is True
    assert any("Aramid belt cord viscoelastic relaxation" in d for d in r_aramid.diagnostics)


def test_belt_theoretical_validation_tiers_and_practical_issues() -> None:
    """Ensure industrial belts are explicitly tagged theoretical with real-world trade-offs."""
    assert BELT_CATALOG["gt2_6mm"]["validation_tier"] == "proven"
    assert BELT_CATALOG["gt2_9mm"]["validation_tier"] == "proven"
    assert BELT_CATALOG["gt2_12mm"]["validation_tier"] == "proven"
    assert BELT_CATALOG["gt2_15mm"]["validation_tier"] == "experimental"
    assert BELT_CATALOG["gt2_dual_9mm"]["validation_tier"] == "experimental"

    for belt_id in ("3gt_9mm", "3gt_12mm", "3gt_15mm", "at3_10mm", "at3_16mm", "at3_gen3_16mm", "htd3m_15mm", "at5_16mm"):
        b = BeltSpec(type=belt_id)
        assert b.validation_tier == "theoretical"
        assert "theoretical" in b.theoretical_summary.lower()
        assert len(b.practical_issues) >= 3

    spec = PrinterSpec.from_yaml(MONOLITH_YAML)
    spec.gantry.belt = BeltSpec(type="at3_gen3_16mm", pretension_n=90.0)
    report = evaluate_printer(spec, include_advanced_analysis=False)
    assert report.kinematics.belt_validation_tier == "theoretical"
    assert len(report.kinematics.belt_practical_issues) >= 3
    assert any("Theoretical / unproven on 3D printers" in d for d in report.diagnostics)
    # Stock proven 2GT should NOT emit a theoretical belt warning
    spec_proven = PrinterSpec.from_yaml(MONOLITH_YAML)
    report_proven = evaluate_printer(spec_proven, include_advanced_analysis=False)
    assert report_proven.kinematics.belt_validation_tier == "proven"
    assert not any("Theoretical / unproven on 3D printers" in d for d in report_proven.diagnostics)
