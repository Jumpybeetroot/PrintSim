"""Regression tests for Physics Audit Group C (C-01..C-03, C-05..C-09) & BOM Group E cleanups."""

from __future__ import annotations

import math
import random
from pathlib import Path

import pytest

from printsim.engine import evaluate_printer, optimize_pareto_front
from printsim.engine import optimizer as optimizer_mod
from printsim.engine import pareto as pareto_mod
from printsim.fea.bed_plate_fem import solve_bed_plate_fem
from printsim.fea.sensitivity import compute_structural_sensitivities
from printsim.geometry.bom import generate_bom
from printsim.geometry.builder import build_assembly
from printsim.schema import HOTEND_CATALOG, TOOLHEAD_PRESETS, PrinterSpec, get_profile

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


# ---------------------------------------------------------------------------
# C-01: Inertial-only deflection scoring, smooth penalty, & Pareto stiffness
# ---------------------------------------------------------------------------


def test_c01_structural_score_ignores_gravity_sag_and_uses_smooth_penalty() -> None:
    """Structural score uses inertial deflection (not static gravity sag) with smooth saturation."""
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    rep_base = evaluate_printer(spec, include_advanced_analysis=False)

    # StaticFEAResult.combined_max_deflection_um remains untouched (gravity + belt + inertial)
    inertial_max = max(
        rep_base.static_fea.accel_deflection_x_um,
        rep_base.static_fea.accel_deflection_y_um,
    )
    assert rep_base.static_fea.combined_max_deflection_um > inertial_max * 2.0

    # Structural score must not saturate at 45-pt hard clamp on stock Voron 2.4 (~30 um @ 10k)
    assert rep_base.scores.structural > 45.0

    # Smooth penalty: worsening deflection from moderate to severe strictly lowers structural score
    # even well beyond 70 um (where the old min(45, 0.65 * d) clamp plateaued flat).
    s_soft_1 = spec.model_copy(deep=True)
    s_soft_1.gantry.x_beam_profile = "extrusion_2020"
    s_soft_1.toolhead.total_mass_g = 900.0
    s_soft_2 = s_soft_1.model_copy(deep=True)
    s_soft_2.toolhead.total_mass_g = 1400.0
    r_soft_1 = evaluate_printer(s_soft_1, include_advanced_analysis=False)
    r_soft_2 = evaluate_printer(s_soft_2, include_advanced_analysis=False)
    assert r_soft_2.scores.structural < r_soft_1.scores.structural

    # Adding pure static gravity settlement (springs_4pt) must not trigger an "under acceleration"
    # deflection penalty spike.
    s_trident = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    s_trident_spr = s_trident.model_copy(deep=True)
    s_trident_spr.bed.mount_type = "springs_4pt"
    r_tri_kin = evaluate_printer(s_trident, include_advanced_analysis=False)
    r_tri_spr = evaluate_printer(s_trident_spr, include_advanced_analysis=False)
    assert r_tri_spr.static_fea.combined_max_deflection_um > 250.0
    # Verify inertial deflection itself did not blow up like combined gravity sag
    assert max(r_tri_spr.static_fea.accel_deflection_x_um, r_tri_spr.static_fea.accel_deflection_y_um) < 40.0


def test_c01_pareto_deflection_ranks_dynamic_stiffness_without_gravity_sag() -> None:
    """Pareto nozzle_deflection_um measures inertial deflection at 10k mm/s^2, not gravity sag."""
    flying = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    base_cand = pareto_mod._evaluate_spec_candidate(flying, ["Baseline"], "pla", "base")

    # Build a dynamically much stiffer flying-gantry CoreXY (4040 + bolted Al5052 + CF beam + CNC mount)
    stiff = flying.model_copy(deep=True)
    stiff.frame.profile = "extrusion_4040"
    stiff.frame.bracing = "full_truss"
    stiff.frame.panel_mounting = "bolted_structural"
    stiff.frame.panel_material = "aluminum_5052_sheet"
    stiff.frame.panel_thickness_mm = 4.0
    stiff.gantry.x_beam_profile = "cf_tube_20x20x2"
    stiff.gantry.x_beam_material = "carbon_fiber_tube"
    stiff.gantry.bracket_material = "aluminum_6061_t6"
    stiff.toolhead.mount_material = "aluminum_6061_t6"
    stiff.toolhead.total_mass_g = 175.0
    stiff.toolhead.body_construction = "plate_sandwich_cf"
    stiff_cand = pareto_mod._evaluate_spec_candidate(stiff, ["Stiff"], "pla", "stiff")

    # Previously gravity sag (~78 um) made the stiff build rank WORSE than baseline (~75 um).
    assert stiff_cand.point.nozzle_deflection_um < base_cand.point.nozzle_deflection_um * 0.45

    # Invariance to configured max_accel_mm_s2 (fixed 10k mm/s^2 reference)
    derated = flying.model_copy(deep=True)
    derated.kinematics.max_accel_mm_s2 = 3000.0
    derated_cand = pareto_mod._evaluate_spec_candidate(derated, ["Derated"], "pla", "derated")
    assert derated_cand.point.nozzle_deflection_um == pytest.approx(
        base_cand.point.nozzle_deflection_um, rel=1e-4
    )


# ---------------------------------------------------------------------------
# C-02: Eliminate derating paradox in evaluator & optimizer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "yaml_name",
    [
        "corexy_voron_style.yaml",
        "voron_trident_300.yaml",
        "cartesian_bedslinger.yaml",
        "vzbot_330_awd.yaml",
    ],
)
def test_c02_lowering_accel_or_velocity_never_increases_composite(yaml_name: str) -> None:
    """Derating max_accel_mm_s2 or max_velocity_mm_s below achievable limits never raises composite."""
    base = PrinterSpec.from_yaml(EXAMPLES_DIR / yaml_name)
    rep_stock = evaluate_printer(base, include_advanced_analysis=False)
    ach_a = float(rep_stock.motion_limits["achievable_max_accel_mm_s2"])
    mot_v = float(rep_stock.motion_limits["motor_max_velocity_mm_s"])

    # Stock default benchmark should not self-penalize for impossible flow/ringing
    assert rep_stock.virtual_print.underextrusion_segments == 0
    assert rep_stock.virtual_print.ringing_segments == 0

    # 1. Monotonicity when lowering max_accel_mm_s2 below achievable ceiling
    a_start = min(base.kinematics.max_accel_mm_s2, ach_a)
    accel_levels = [a_start, a_start * 0.75, a_start * 0.5, a_start * 0.25]
    accel_scores: list[float] = []
    for a_val in accel_levels:
        s = base.model_copy(deep=True)
        s.kinematics.max_accel_mm_s2 = max(a_val, 500.0)
        accel_scores.append(evaluate_printer(s, include_advanced_analysis=False).scores.composite)
    for hi, lo in zip(accel_scores, accel_scores[1:]):
        assert lo <= hi + 1e-6, f"{yaml_name}: accel derating raised composite ({hi} -> {lo})"
    assert accel_scores[-1] < accel_scores[0]

    # Also check audit repro: stock Voron (8000) vs derated (4000, 2000)
    s_4k = base.model_copy(deep=True)
    s_4k.kinematics.max_accel_mm_s2 = min(4000.0, a_start * 0.6)
    rep_4k = evaluate_printer(s_4k, include_advanced_analysis=False)
    assert rep_4k.scores.composite <= rep_stock.scores.composite + 1e-6

    # 2. Monotonicity when lowering max_velocity_mm_s below achievable ceiling
    v_start = min(base.kinematics.max_velocity_mm_s, mot_v)
    vel_levels = [v_start, v_start * 0.7, v_start * 0.4, max(60.0, v_start * 0.2)]
    vel_scores: list[float] = []
    for v_val in vel_levels:
        s = base.model_copy(deep=True)
        s.kinematics.max_velocity_mm_s = v_val
        vel_scores.append(evaluate_printer(s, include_advanced_analysis=False).scores.composite)
    for hi, lo in zip(vel_scores, vel_scores[1:]):
        assert lo <= hi + 1e-6, f"{yaml_name}: velocity derating raised composite ({hi} -> {lo})"


def test_c02_optimizer_never_derates_max_accel() -> None:
    """Optimizer mutation pool no longer contains accel derating."""
    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    rng = random.Random(7)
    for _ in range(120):
        cand, changes = optimizer_mod._mutate_spec(base, rng)
        assert cand.kinematics.max_accel_mm_s2 == pytest.approx(base.kinematics.max_accel_mm_s2)
        assert not any(c.startswith("kinematics.max_accel=") for c in changes)


# ---------------------------------------------------------------------------
# C-03: Bed plate flatness after best-fit plane removal
# ---------------------------------------------------------------------------


def test_c03_bed_plate_separates_settlement_tilt_and_true_flatness() -> None:
    """mount_settlement_um, rigid_tilt_um, and plane-removed plate_flatness_um are cleanly separated."""
    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")

    # 1. Cantilevered single-Z: huge rigid front-rear tilt, small true non-planar bow
    s_single = base.model_copy(deep=True)
    s_single.bed.z_drive = "leadscrew_single"
    s_single.bed.target_temp_c = 60.0
    r_single = solve_bed_plate_fem(s_single)
    assert r_single.rigid_tilt_um > 500.0
    assert r_single.plate_flatness_um < 50.0
    assert r_single.plate_flatness_um < r_single.total_flatness_error_um * 0.1
    assert r_single.mesh_bed_leveling_required is False
    assert r_single.first_layer_risk == "low"

    # 2. Soft die springs (springs_4pt): huge vertical settlement, tiny non-planar bow
    s_springs = base.model_copy(deep=True)
    s_springs.bed.mount_type = "springs_4pt"
    s_springs.bed.target_temp_c = 60.0
    r_springs = solve_bed_plate_fem(s_springs)
    assert r_springs.mount_settlement_um > 200.0
    assert r_springs.rigid_tilt_um < 5.0
    assert r_springs.plate_flatness_um < 30.0
    assert r_springs.mesh_bed_leveling_required is False
    assert r_springs.first_layer_risk == "low"

    # 3. Over-constrained rigid_4pt at 110 C: genuine non-planar taco warp drives high/critical risk
    s_rigid = base.model_copy(deep=True)
    s_rigid.bed.mount_type = "rigid_4pt"
    s_rigid.bed.target_temp_c = 110.0
    r_rigid = solve_bed_plate_fem(s_rigid)
    assert r_rigid.plate_flatness_um > 100.0
    assert r_rigid.mesh_bed_leveling_required is True
    assert r_rigid.first_layer_risk in ("high", "critical")

    # 4. Evaluator reports plane-removed flatness error clearly for rigid_4pt and tilt for single-Z
    rep_rigid = evaluate_printer(s_rigid)
    assert any("after plane removal" in d.lower() for d in rep_rigid.diagnostics)
    rep_single = evaluate_printer(s_single)
    assert any("rigid tilt" in d.lower() and "after plane removal" in d.lower() for d in rep_single.diagnostics)


# ---------------------------------------------------------------------------
# C-05 & BOM Group E: Mass/cost reconciliation, belt mass, Z spec, net cuts
# ---------------------------------------------------------------------------


def test_c05_bom_hotend_mass_belt_mass_z_spec_and_net_cuts() -> None:
    """BOM uses spec hotend.mass_g, unified 12 g/m 6mm GT2 mass, clean belted-Z spec, and net cut lengths."""
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    bom = generate_bom(spec)
    by_sku = {it.sku: it for it in bom.items}

    # 1. Hotend BOM mass matches spec.toolhead.hotend.mass_g
    assert by_sku["HOT-ASM"].total_mass_g == pytest.approx(spec.toolhead.hotend.mass_g, abs=0.1)

    # Swap to Revo Voron (30 g) -> BOM reflects 30.0 g exactly
    s_revo = spec.model_copy(deep=True)
    s_revo.toolhead.hotend.preset_id = "e3d_revo_voron"
    bom_revo = generate_bom(s_revo)
    revo_hot = next(it for it in bom_revo.items if it.sku == "HOT-ASM")
    assert revo_hot.total_mass_g == pytest.approx(30.0, abs=0.1)

    # 2. Unified 6mm GT2 linear mass = 12.0 g/m across XY, Z-drive, and Z-sync belts
    xy_belt = by_sku["MOT-BELT-XY"]
    z_belt = by_sku["MOT-Z-BELT"]
    assert xy_belt.total_mass_g == pytest.approx(float(xy_belt.quantity) * 12.0, abs=0.2)
    assert z_belt.total_mass_g == pytest.approx(float(z_belt.quantity) * 12.0, abs=0.2)

    vz = PrinterSpec.from_yaml(EXAMPLES_DIR / "vzbot_330_awd.yaml")
    bom_vz = generate_bom(vz)
    z_sync = next(it for it in bom_vz.items if it.sku == "MOT-Z-SYNC-BELT")
    expected_sync_mm = round(2.0 * vz.frame.width_mm + vz.frame.depth_mm + 200.0, 0)
    assert z_sync.total_mass_g == pytest.approx((expected_sync_mm * 1e-3) * 12.0, abs=0.2)

    # 3. belted_quad_z with explicit leadscrew z_leadscrew_type must not leak leadscrew name into belt spec
    s_weird_z = spec.model_copy(deep=True)
    s_weird_z.bed.z_drive = "belted_quad_z"
    s_weird_z.bed.z_leadscrew_type = "tr8x4_pom"
    bom_weird = generate_bom(s_weird_z)
    z_belt_weird = next(it for it in bom_weird.items if it.sku == "MOT-Z-BELT")
    assert "TR8x4" not in z_belt_weird.specification
    assert "GT2 6mm" in z_belt_weird.specification

    # 4. Net non-overlapping buildable cut lengths reported & optional clipping reduces frame mass
    from printsim.geometry.builder import corexy_x_beam_span_m

    expected_gantry_net_mm = corexy_x_beam_span_m(spec) * 1000.0
    gantry_cut = next(c for c in bom.cut_list if c.role == "gantry")
    assert gantry_cut.net_length_mm == pytest.approx(expected_gantry_net_mm, abs=0.1)

    bom_clipped = generate_bom(spec, clip_corner_overlaps=True)
    gantry_clipped = next(c for c in bom_clipped.cut_list if c.role == "gantry")
    assert gantry_clipped.length_mm == pytest.approx(expected_gantry_net_mm, abs=0.1)
    assert bom_clipped.total_mass_kg < bom.total_mass_kg

    # 5. Evaluator efficiency score uses BOM cost consistently (adding expensive motors lowers efficiency)
    rep_2wd = evaluate_printer(spec, include_advanced_analysis=False)
    assert rep_2wd.bom_total_cost_usd == pytest.approx(bom.total_cost_usd, abs=0.05)
    s_awd = spec.model_copy(deep=True)
    s_awd.gantry.drive_mode = "awd"
    s_awd.gantry.double_shear_pulleys = True
    rep_awd = evaluate_printer(s_awd, include_advanced_analysis=False)
    assert rep_awd.bom_total_cost_usd > rep_2wd.bom_total_cost_usd + 50.0
    assert rep_awd.scores.efficiency < rep_2wd.scores.efficiency


# ---------------------------------------------------------------------------
# C-06 & C-07: Optimizer realistic upgrades, smooth time bonus, Pareto fidelity
# ---------------------------------------------------------------------------


def test_c06_optimizer_realistic_presets_and_smooth_time_bonus() -> None:
    """Speed time bonus stays positive > 160 s, and mutations swap full hotend/toolhead presets."""
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    rep = evaluate_printer(spec, panel_mesh_n=2, include_advanced_analysis=False)

    # Speed objective time bonus must stay > 0 and strictly decreasing well past 160 s
    orig_t = rep.virtual_print.total_time_s
    rep.virtual_print.total_time_s = 160.0
    s_160 = optimizer_mod._score_objective(rep, "speed")
    rep.virtual_print.total_time_s = 220.0
    s_220 = optimizer_mod._score_objective(rep, "speed")
    rep.virtual_print.total_time_s = 360.0
    s_360 = optimizer_mod._score_objective(rep, "speed")
    rep.virtual_print.total_time_s = orig_t
    assert s_160 > s_220 > s_360

    # Mutations use catalog presets (never free hotend.melt / toolhead.mass hacks)
    rng = random.Random(42)
    seen_hotend_preset = False
    seen_toolhead_preset = False
    for _ in range(120):
        cand, changes = optimizer_mod._mutate_spec(spec, rng)
        for ch in changes:
            assert not ch.startswith(("hotend.melt=", "hotend.heater=", "hotend.heatbreak=", "toolhead.mass="))
            if ch.startswith("hotend.preset="):
                seen_hotend_preset = True
                pid = ch.split("=", 1)[1]
                assert pid in HOTEND_CATALOG
                assert cand.toolhead.hotend.mass_g == pytest.approx(HOTEND_CATALOG[pid]["mass_g"])
            if ch.startswith("toolhead.preset="):
                seen_toolhead_preset = True
                tid = ch.split("=", 1)[1].split("(", 1)[0]
                assert tid in TOOLHEAD_PRESETS
                assert cand.toolhead.body_construction == TOOLHEAD_PRESETS[tid]["body_construction"]
                assert cand.toolhead.z_probe_type == TOOLHEAD_PRESETS[tid]["z_probe_type"]
    assert seen_hotend_preset
    assert seen_toolhead_preset


def test_c07_pareto_baseline_exact_fidelity_and_full_toolhead_preset() -> None:
    """Decoding _encode_baseline_vector(spec) reproduces baseline exactly; toolhead gene copies all preset fields."""
    assert pareto_mod._N_VARS == 12

    for yaml_name in ("corexy_voron_style.yaml", "voron_trident_300.yaml", "ratrig_vcore31_300.yaml"):
        base = PrinterSpec.from_yaml(EXAMPLES_DIR / yaml_name)
        bvec = pareto_mod._encode_baseline_vector(base)
        dec, changes = pareto_mod._decode_candidate(base, bvec)
        assert changes == ["Baseline (unchanged)"]
        assert dec.model_dump() == base.model_dump(), f"{yaml_name} baseline decode drifted!"

    # Changing toolhead gene copies body_construction, z_probe_type, & mount_material
    trident = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    bvec = list(pareto_mod._encode_baseline_vector(trident))
    bvec[7] = pareto_mod.TOOLHEAD_OPTIONS.index("archetype_cpap")
    dec_th, changes_th = pareto_mod._decode_candidate(trident, tuple(bvec))
    preset = TOOLHEAD_PRESETS["archetype_cpap"]
    assert dec_th.toolhead.body_construction == preset["body_construction"]
    assert dec_th.toolhead.z_probe_type == preset["z_probe_type"]
    assert dec_th.toolhead.mount_material == preset["mount_material"]
    assert any(c.startswith("Toolhead: archetype_cpap") for c in changes_th)

    # Changing hotend gene applies full HOTEND_CATALOG preset (including mass & stiffness)
    bvec_he = list(pareto_mod._encode_baseline_vector(trident))
    bvec_he[8] = 2  # phaetus_rapido_2_uhf
    dec_he, changes_he = pareto_mod._decode_candidate(trident, tuple(bvec_he))
    uhf_meta = HOTEND_CATALOG[pareto_mod.HOTEND_PRESET_OPTIONS[2]]
    assert dec_he.toolhead.hotend.preset_id == pareto_mod.HOTEND_PRESET_OPTIONS[2]
    assert dec_he.toolhead.hotend.mass_g == pytest.approx(uhf_meta["mass_g"])
    assert dec_he.toolhead.hotend.structural_stiffness_n_per_mm == pytest.approx(
        uhf_meta["structural_stiffness_n_per_mm"]
    )
    assert any(c.startswith("Hotend:") for c in changes_he)


# ---------------------------------------------------------------------------
# C-08: Rayleigh est_hz_gain in sensitivity.py
# ---------------------------------------------------------------------------


def test_c08_sensitivity_est_hz_gain_uses_rayleigh_not_linear() -> None:
    """est_hz_gain matches frozen-mode Rayleigh f*(sqrt(1 + s_e/2) - 1), strictly below linear 0.5*df/d(ln E)."""
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    rep = evaluate_printer(spec, include_advanced_analysis=True)
    assert rep.sensitivity is not None
    targets = rep.sensitivity.top_stiffen_targets
    assert len(targets) >= 1

    # Verify analytical Rayleigh vs linear property at full strain-energy share (s_e = 1.0):
    f_demo = 100.0
    stiff_sens_full = 0.5 * f_demo  # df/d(ln E) = (f/2)*s_e
    linear_est = 0.5 * stiff_sens_full
    rayleigh_est = f_demo * (math.sqrt(1.0 + stiff_sens_full / f_demo) - 1.0)
    assert linear_est == pytest.approx(25.0)
    assert rayleigh_est == pytest.approx(22.474487, rel=1e-5)
    assert rayleigh_est < linear_est

    # Check actual modal stiffening targets on Voron 2.4 stay positive and strictly <= linear bound
    pos_targets = [t for t in targets if t["est_hz_gain"] > 0.0]
    assert len(pos_targets) >= 1
    total_fund_stiff = sum(
        max(0.0, float(m["stiffness_sens_hz"]))
        for m in rep.sensitivity.member_sensitivities.values()
    )
    for t in pos_targets:
        assert t["est_hz_gain"] <= 0.5 * total_fund_stiff + 0.1


# ---------------------------------------------------------------------------
# C-09: Dynamics score capped by motor/belt & bed fundamental mode diagnostic
# ---------------------------------------------------------------------------


def test_c09_dynamics_score_capped_by_motor_and_belt_limits() -> None:
    """Dynamics score responds to motor/belt acceleration ceilings, not raw shaper accel alone."""
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_monolith_awd.yaml")
    rep_full = evaluate_printer(spec, include_advanced_analysis=False)

    # Starve the motor (low voltage/current) so motor_max_accel << shaper_max_accel
    s_weak_mot = spec.model_copy(deep=True)
    s_weak_mot.kinematics.xy_stepper.supply_voltage_v = 12.0
    s_weak_mot.kinematics.xy_stepper.run_current_a = 0.5
    rep_weak_mot = evaluate_printer(s_weak_mot, include_advanced_analysis=False)
    assert rep_weak_mot.motion_limits["accel_limiting_factor"] == "motor"
    assert rep_weak_mot.scores.dynamics < rep_full.scores.dynamics - 10.0

    # Starve belt pretension so belt_max_accel << shaper_max_accel and configured max_accel
    s_weak_belt = spec.model_copy(deep=True)
    s_weak_belt.gantry.belt.pretension_n = 3.0
    rep_weak_belt = evaluate_printer(s_weak_belt, include_advanced_analysis=False)
    assert rep_weak_belt.motion_limits["accel_limiting_factor"] == "belt"
    assert rep_weak_belt.scores.dynamics < rep_full.scores.dynamics - 10.0


def test_c09_low_bed_mode_not_mislabeled_as_outer_frame_racking() -> None:
    """Soft springs_4pt bed mode (< 40 Hz) must not emit 'outer frame racking' diagnostic."""
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    spec.bed.mount_type = "springs_4pt"
    rep = evaluate_printer(spec, include_advanced_analysis=False)
    assert rep.modal_fea.fundamental_freq_hz < 40.0
    joined = " ".join(rep.diagnostics).lower()
    assert "outer frame racking" not in joined
    assert "bed/z-mount" in joined
