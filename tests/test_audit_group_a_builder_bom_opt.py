"""Regression tests for Physics Audit Group A fixes (A-10, A-11, A-12, A-18)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from printsim.engine import evaluate_printer, optimize_pareto_front
from printsim.engine.optimizer import _score_objective
from printsim.engine.pareto import _evaluate_spec_candidate
from printsim.geometry import build_assembly, generate_bom
from printsim.kinematics import analyze_kinematics
from printsim.schema import PrinterSpec, get_profile

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


# ---------------------------------------------------------------------------
# A-11: Dual-belt-sync side rail torsion constant uses j_m4 (not iy_m4)
# ---------------------------------------------------------------------------


def test_a11_belt_sync_side_rails_scale_j_m4_from_j_m4() -> None:
    """base/top left/right side rails in leadscrew_dual_belt_sync scale j_m4 from j_m4 * 2.5, not iy_m4."""
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "vzbot_330_awd.yaml")
    assert spec.bed.z_drive == "leadscrew_dual_belt_sync"

    base_prof = get_profile(spec.frame.profile)
    # Sanity: for T-slot extrusion, iy_m4 >> j_m4
    assert base_prof.iy_m4 > base_prof.j_m4 * 3.0

    asm = build_assembly(spec)
    base_sides = [m for m in asm.members if m.id.startswith(("base_left_", "base_right_"))]
    assert len(base_sides) > 0
    for m in base_sides:
        assert m.profile.j_m4 == pytest.approx(base_prof.j_m4 * 2.5, rel=1e-6)
        assert m.profile.iz_m4 == pytest.approx(base_prof.iz_m4 * 4.0, rel=1e-6)

    # The real VzBot has its Y rails on the top-ring Y extrusions (gantry_on_top_frame): those
    # extrusions are the y_rail_* chains (rail composite), so no separate top_left/right members.
    assert spec.frame.gantry_on_top_frame
    assert not any(m.id.startswith(("top_left_", "top_right_")) for m in asm.members)

    # Classic separate-gantry-ring topology keeps the top side-rail rod-holder scaling
    spec = spec.model_copy(deep=True)
    spec.frame.gantry_on_top_frame = False
    asm = build_assembly(spec)
    base_sides = [m for m in asm.members if m.id.startswith(("base_left_", "base_right_"))]
    top_sides = [m for m in asm.members if m.id.startswith(("top_left_", "top_right_"))]
    assert len(base_sides) > 0
    assert len(top_sides) > 0

    for m in base_sides:
        assert m.profile.j_m4 == pytest.approx(base_prof.j_m4 * 2.5, rel=1e-6)
        assert m.profile.iz_m4 == pytest.approx(base_prof.iz_m4 * 4.0, rel=1e-6)

    for m in top_sides:
        assert m.profile.j_m4 == pytest.approx(base_prof.j_m4 * 2.5, rel=1e-6)
        assert m.profile.iz_m4 == pytest.approx(base_prof.iz_m4 * 2.5, rel=1e-6)


# ---------------------------------------------------------------------------
# A-12: BOM sizes Z rails from spec.bed.z_rail_profile, skips "none", labels rods
# ---------------------------------------------------------------------------


def test_a12_bom_uses_z_rail_profile_across_architectures() -> None:
    """BOM uses spec.bed.z_rail_profile (not y_rail_profile) for CoreXY, Bedslinger, and Delta."""
    # 1. CoreXY Trident: y_rail=mgn9_rail, z_rail=mgn12_rail
    trident = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    assert trident.gantry.y_rail_profile == "mgn9_rail"
    assert trident.bed.z_rail_profile == "mgn12_rail"
    bom_tri = generate_bom(trident)
    tri_by_sku = {it.sku: it for it in bom_tri.items}
    assert "MGN9" in tri_by_sku["MOT-Y-RAIL"].name
    assert "MGN12" in tri_by_sku["MOT-Z-RAIL"].name
    assert "MGN12H" in tri_by_sku["MOT-Z-CAR"].name
    assert tri_by_sku["MOT-Z-RAIL"].quantity == 3
    # Mass should match MGN12 (650 g/m), not MGN9 (320 g/m)
    mgn12 = get_profile("mgn12_rail")
    mgn9 = get_profile("mgn9_rail")
    assert tri_by_sku["MOT-Z-RAIL"].total_mass_g > 3 * 0.35 * mgn9.area_m2 * 7850.0 * 1000.0 * 1.5
    assert tri_by_sku["MOT-Z-CAR"].specification == f"Preloaded block for {mgn12.id}"

    # 2. Bedslinger: y_rail=mgn12_rail, z_rail=mgn9_rail
    bs = PrinterSpec.from_yaml(EXAMPLES_DIR / "cartesian_bedslinger.yaml")
    assert bs.gantry.y_rail_profile == "mgn12_rail"
    assert bs.bed.z_rail_profile == "mgn9_rail"
    bom_bs = generate_bom(bs)
    bs_by_sku = {it.sku: it for it in bom_bs.items}
    assert "MGN12" in bs_by_sku["MOT-Y-RAIL"].name
    assert "MGN9" in bs_by_sku["MOT-Z-RAIL"].name
    assert "MGN9H" in bs_by_sku["MOT-Z-CAR"].name

    # 3. Delta: y_rail=mgn12_rail, z_rail=mgn9_rail
    delta = PrinterSpec.from_yaml(EXAMPLES_DIR / "delta_racer.yaml")
    assert delta.gantry.y_rail_profile == "mgn12_rail"
    assert delta.bed.z_rail_profile == "mgn9_rail"
    bom_delta = generate_bom(delta)
    delta_by_sku = {it.sku: it for it in bom_delta.items}
    assert "MGN9" in delta_by_sku["MOT-TWR-RAIL"].name
    assert "MGN9H" in delta_by_sku["MOT-TWR-CAR"].name
    assert delta_by_sku["MOT-TWR-RAIL"].quantity == 3


def test_a12_bom_skips_none_and_labels_steel_smooth_rods() -> None:
    """z_rail_profile='none' omits Z/tower rails; steel_rod_* emits smooth rods + linear bearings."""
    for yaml_name, rail_sku, car_sku, rod_sku, brg_sku, rod_prof, expected_brg in [
        ("voron_trident_300.yaml", "MOT-Z-RAIL", "MOT-Z-CAR", "MOT-Z-ROD", "MOT-Z-BRG", "steel_rod_10mm", "LM10UU"),
        ("cartesian_bedslinger.yaml", "MOT-Z-RAIL", "MOT-Z-CAR", "MOT-Z-ROD", "MOT-Z-BRG", "steel_rod_8mm", "LM8UU"),
        ("delta_racer.yaml", "MOT-TWR-RAIL", "MOT-TWR-CAR", "MOT-TWR-ROD", "MOT-TWR-BRG", "steel_rod_12mm", "LM12UU"),
    ]:
        base = PrinterSpec.from_yaml(EXAMPLES_DIR / yaml_name)

        # z_rail_profile == "none" skips Z/tower rail & carriage lines
        spec_none = base.model_copy(deep=True)
        spec_none.bed.z_rail_profile = "none"
        skus_none = {it.sku for it in generate_bom(spec_none).items}
        assert rail_sku not in skus_none
        assert car_sku not in skus_none
        assert rod_sku not in skus_none
        assert brg_sku not in skus_none

        # z_rail_profile == "steel_rod_*" emits smooth rod + linear bearing instead of "Linear Rail"
        spec_rod = base.model_copy(deep=True)
        spec_rod.bed.z_rail_profile = rod_prof  # type: ignore[assignment]
        by_sku_rod = {it.sku: it for it in generate_bom(spec_rod).items}
        assert rail_sku not in by_sku_rod
        assert car_sku not in by_sku_rod
        assert rod_sku in by_sku_rod
        assert brg_sku in by_sku_rod
        assert "Smooth Rod" in by_sku_rod[rod_sku].name
        assert "Linear Rail" not in by_sku_rod[rod_sku].name
        assert expected_brg in by_sku_rod[brg_sku].name


# ---------------------------------------------------------------------------
# A-18: Optimizer collision penalty is not double-counted and respects objective
# ---------------------------------------------------------------------------


def test_a18_optimizer_collision_penalty_single_and_objective_aware() -> None:
    """_score_objective avoids double-penalizing balanced and applies a single -30 on other objectives."""
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    ok_rep = evaluate_printer(spec, panel_mesh_n=2, include_advanced_analysis=False)
    assert not ok_rep.collision.has_collision

    # Force a real collision by shrinking outer frame width below build volume
    col_spec = spec.model_copy(deep=True)
    col_spec.frame.width_mm = 260.0
    col_rep = evaluate_printer(col_spec, panel_mesh_n=2, include_advanced_analysis=False)
    assert col_rep.collision.has_collision

    # Balanced: composite already includes the single -30 penalty; no extra -30 subtracted
    assert _score_objective(col_rep, "balanced") == pytest.approx(col_rep.scores.composite, rel=1e-9)
    assert _score_objective(col_rep, "balanced") >= 0.0

    # Other objectives: compute their own weighted score and subtract 30 once (clamped at 0)
    # C-06 rescaled smooth time bonus: min(40.0, 40.0 * (120.0 / max(t, 30.0)))
    time_bonus = min(40.0, 40.0 * (120.0 / max(col_rep.virtual_print.total_time_s, 30.0)))
    raw_speed = (
        0.35 * col_rep.scores.dynamics
        + 0.25 * col_rep.scores.thermal
        + 0.25 * col_rep.scores.print_quality
        + 0.15 * min(100.0, time_bonus * 2.5)
    )
    raw_rigidity = (
        0.65 * col_rep.scores.structural
        + 0.20 * col_rep.scores.dynamics
        + 0.15 * col_rep.scores.print_quality
    )
    raw_budget = (
        0.45 * col_rep.scores.efficiency
        + 0.30 * col_rep.scores.print_quality
        + 0.25 * col_rep.scores.structural
    )
    assert _score_objective(col_rep, "speed") == pytest.approx(max(0.0, raw_speed - 30.0), rel=1e-9)
    assert _score_objective(col_rep, "rigidity") == pytest.approx(max(0.0, raw_rigidity - 30.0), rel=1e-9)
    assert _score_objective(col_rep, "budget") == pytest.approx(max(0.0, raw_budget - 30.0), rel=1e-9)

    # Non-balanced colliding objectives remain distinct (not collapsed to composite - 30)
    non_bal = {
        round(_score_objective(col_rep, obj), 4)  # type: ignore[arg-type]
        for obj in ("speed", "rigidity", "budget")
    }
    assert len(non_bal) >= 2


# ---------------------------------------------------------------------------
# A-10: Pareto safe_max_accel == min(shaper, motor, belt) at 1.0x & constraints at safe_accel
# ---------------------------------------------------------------------------


def test_a10_pareto_safe_accel_respects_1x_motor_and_belt_limits() -> None:
    """Pareto safe_max_accel_mm_s2 equals min(shaper, motor, belt) at 1.0x and matches motion_limits."""
    # Use a high-frequency AWD build where motor/belt can bind below shaper
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_monolith_awd.yaml")
    # Weak motor so motor_max_accel_mm_s2 is strictly below shaper recommended accel
    spec.kinematics.xy_stepper.supply_voltage_v = 12.0
    spec.kinematics.xy_stepper.holding_torque_nm = 0.25
    spec.kinematics.xy_stepper.run_current_a = 0.50

    rep = evaluate_printer(spec, panel_mesh_n=2, include_advanced_analysis=False)
    expected_safe = min(
        float(rep.modal_fea.recommended_max_accel_mm_s2),
        float(rep.kinematics.motor_max_accel_mm_s2),
        float(rep.kinematics.belt_max_accel_mm_s2),
    )
    assert expected_safe < rep.modal_fea.recommended_max_accel_mm_s2

    cand = _evaluate_spec_candidate(spec, ["test"], "pla", "test_motor_cap")
    assert cand.point.safe_max_accel_mm_s2 == pytest.approx(expected_safe, rel=1e-6)
    assert cand.point.safe_max_accel_mm_s2 == pytest.approx(
        float(rep.motion_limits["achievable_max_accel_mm_s2"]), abs=1.0
    )
    assert cand.point.safe_max_accel_mm_s2 <= rep.kinematics.motor_max_accel_mm_s2 + 1e-6
    assert cand.point.safe_max_accel_mm_s2 <= rep.kinematics.belt_max_accel_mm_s2 + 1e-6

    # Belt-limited build: low wrap angle + modest pretension triggers tooth-cam-out belt slip below shaper limit
    belt_spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_monolith_awd.yaml")
    belt_spec.gantry.belt.wrap_angle_deg = 90.0
    belt_spec.gantry.belt.pretension_n = 12.0
    rep_belt = evaluate_printer(belt_spec, panel_mesh_n=2, include_advanced_analysis=False)
    assert rep_belt.kinematics.belt_max_accel_mm_s2 > 0.0
    assert rep_belt.kinematics.belt_max_accel_mm_s2 < rep_belt.kinematics.motor_max_accel_mm_s2
    assert rep_belt.kinematics.belt_max_accel_mm_s2 < rep_belt.modal_fea.recommended_max_accel_mm_s2
    cand_belt = _evaluate_spec_candidate(belt_spec, ["low_pretension"], "pla", "test_belt_cap")
    assert cand_belt.point.safe_max_accel_mm_s2 == pytest.approx(
        rep_belt.kinematics.belt_max_accel_mm_s2, rel=1e-6
    )


def test_a10_pareto_evaluates_kinematic_constraints_at_safe_accel() -> None:
    """Belt overtension at safe_accel marks candidate infeasible even if baseline max_accel_mm_s2 is safe."""
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_monolith_awd.yaml")
    # Configure 6mm fiberglass belt (max_working_tension_n = 58 N) with pretension = 55 N:
    # - At a low baseline max_accel_mm_s2 = 2000, dynamic tension stays < 58 N (overtension False).
    # - At advertised safe_accel (~15k-30k mm/s^2), dynamic tension exceeds 58 N (overtension True).
    spec.gantry.belt.type = "gt2_6mm"
    spec.gantry.belt.core = "fiberglass"
    spec.gantry.belt.pretension_n = 55.0
    spec.gantry.double_shear_pulleys = True
    spec.kinematics.max_accel_mm_s2 = 2000.0

    rep = evaluate_printer(spec, panel_mesh_n=2, include_advanced_analysis=False)
    assert rep.kinematics.belt_overtension_risk is False, "Baseline accel should not trigger overtension"

    cand = _evaluate_spec_candidate(spec, ["high_pretension_6mm"], "pla", "test_overtension_at_safe")
    assert cand.g[1] > 0.0, "belt_overtension_risk must be flagged at advertised safe_accel"
    assert cand.feasible is False


def test_a10_pareto_front_points_pass_kinematic_checks_at_advertised_accel() -> None:
    """Every point on the optimized Pareto front is kinematically valid at its advertised safe_max_accel."""
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    res = optimize_pareto_front(spec, population_size=12, generations=4, seed=42)
    assert len(res.pareto_front) >= 3

    for pt in res.pareto_front:
        rep = evaluate_printer(pt.spec, panel_mesh_n=2, include_advanced_analysis=False)
        assert pt.safe_max_accel_mm_s2 <= rep.modal_fea.recommended_max_accel_mm_s2 + 1e-3
        assert pt.safe_max_accel_mm_s2 <= rep.kinematics.motor_max_accel_mm_s2 + 1e-3
        assert pt.safe_max_accel_mm_s2 <= rep.kinematics.belt_max_accel_mm_s2 + 1e-3

        spec_at_safe = pt.spec.model_copy(
            update={
                "kinematics": pt.spec.kinematics.model_copy(
                    update={"max_accel_mm_s2": max(pt.safe_max_accel_mm_s2 * (1.0 - 1e-9), 1.0)}
                )
            }
        )
        kin_at_safe = analyze_kinematics(replace(rep.assembly, spec=spec_at_safe))
        assert not kin_at_safe.step_skip_risk, f"{pt.id} step-skips at advertised {pt.safe_max_accel_mm_s2}"
        assert not kin_at_safe.belt_slip_risk, f"{pt.id} belt-slips at advertised {pt.safe_max_accel_mm_s2}"
        assert not kin_at_safe.belt_overtension_risk, f"{pt.id} overtensions at {pt.safe_max_accel_mm_s2}"
        assert not kin_at_safe.shaft_bending_risk, f"{pt.id} shaft-bends at {pt.safe_max_accel_mm_s2}"
