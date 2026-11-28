"""Stage 2 Physics Audit verification tests (Group B, Group D datasheets, C-04, C-08)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from printsim.fea import solve_modal_fea
from printsim.fea.beam3d import assemble_global_matrices, node_rotational_inertia_xyz
from printsim.fea.components import (
    compute_bed_thermal_bow_and_taco_um,
    hertzian_rail_carriage_stiffness,
)
from printsim.fea.sensitivity import compute_structural_sensitivities
from printsim.geometry.builder import build_assembly, compute_toolhead_link_props
from printsim.kinematics.solvers import (
    compute_bracket_stiffness_n_per_m,
    compute_drive_stiffness_details_n_per_m,
    compute_drive_stiffness_n_per_m,
)
from printsim.schema import (
    BED_MOUNT_CATALOG,
    BeltSpec,
    PrinterSpec,
    ToolheadSpec,
    get_material,
    get_profile,
)

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


# ---------------------------------------------------------------------------
# Group D Datasheet Verification (D-01 .. D-05)
# ---------------------------------------------------------------------------


def test_d01_through_d05_datasheet_catalog_values() -> None:
    """Verify exact datasheet values for GT2 EA (D-01), CF G (D-02), MGN (D-03), 2020 (D-04), flat bar J (D-05)."""
    # D-01: Gates Table 6 quasi-static EA (16.9 kN at 6 mm, 3.8 lb at 0.1 % elongation) x the 1.85
    # dynamic small-amplitude ratio fitted to controlled same-machine measurements
    assert BeltSpec(type="gt2_6mm").static_stiffness_ea_n == pytest.approx(16900.0)
    assert BeltSpec(type="gt2_6mm").stiffness_ea_n == pytest.approx(16900.0 * 1.85)
    assert BeltSpec(type="gt2_9mm").stiffness_ea_n == pytest.approx(25500.0 * 1.85)
    assert BeltSpec(type="gt2_12mm").stiffness_ea_n == pytest.approx(34000.0 * 1.85)
    assert BeltSpec(type="gt2_dual_9mm").stiffness_ea_n == pytest.approx(51000.0 * 1.85)
    assert BeltSpec(type="gt2_15mm").stiffness_ea_n == pytest.approx(47250.0 * 1.85)
    assert BeltSpec(type="gt2_9mm").stiffness_ea_n < BeltSpec(type="at3_10mm").stiffness_ea_n
    # Gates Table 6: 3GT 9 mm 8.9 lb vs 2GT 9 mm 5.7 lb at 0.1 % elongation
    assert BeltSpec(type="3gt_9mm").stiffness_ea_n / BeltSpec(type="gt2_9mm").stiffness_ea_n == pytest.approx(
        39600.0 / 25500.0, rel=1e-6
    )

    # D-02: 0/90 roll-wrapped CF laminate in-plane shear modulus G_xy = 5.0 GPa
    for mat_id in ("carbon_fiber_tube", "carbon_fiber_high_tg"):
        mat = get_material(mat_id)
        assert mat.shear_modulus_pa == pytest.approx(5.0e9, rel=1e-9)

    # D-03: HIWIN MG catalog linear mass @ 7850 kg/m^3 steel
    rho_steel = 7850.0
    for pid, expected_kg_m in (("mgn9_rail", 0.380), ("mgn12_rail", 0.650), ("mgn15_rail", 1.060)):
        prof = get_profile(pid)
        assert prof.area_m2 * rho_steel == pytest.approx(expected_kg_m, rel=1e-6)

    # D-04: Misumi HFS5-2020 2020 T-slot extrusion cross-section & open torsion I_t
    p2020 = get_profile("extrusion_2020")
    assert p2020.area_m2 == pytest.approx(1.83e-4, rel=1e-9)
    assert p2020.iy_m4 == pytest.approx(0.742e-8, rel=1e-9)
    assert p2020.iz_m4 == pytest.approx(0.742e-8, rel=1e-9)
    assert p2020.j_m4 == pytest.approx(0.11e-8, rel=1e-9)

    # D-05: Roark rectangular St. Venant torsion constant for 25x5mm flat bar (9.104e-10 m^4)
    flat_bar = get_profile("al_flat_bar_25x5")
    assert flat_bar.j_m4 == pytest.approx(9.104e-10, rel=1e-3)


# ---------------------------------------------------------------------------
# B-01 & B-02: Coupled Series Compliance on Moving Mass & Signed Receptance
# ---------------------------------------------------------------------------


def test_b01_coupled_series_compliance_and_consistent_shares() -> None:
    """Verify f_eff and compliance_share_pct use c_struct + 1/k_drive on moving mass (not 1/f_s^2 + 1/f_d^2)."""
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    asm = build_assembly(spec)
    mo = solve_modal_fea(asm)
    assert mo.shaper_x is not None and mo.shaper_y is not None

    K, _, _, nidx, free_dofs = assemble_global_matrices(asm)
    K_ff = 0.5 * (K[np.ix_(free_dofs, free_dofs)] + K[np.ix_(free_dofs, free_dofs)].T)
    cog_i = nidx[asm.toolhead_cog_node_id]
    noz_i = nidx[asm.nozzle_node_id]
    th_i = nidx[asm.toolhead_node_id]
    bed_i = nidx[asm.bed_node_id]
    dr = asm.nodes[noz_i].position_m - asm.nodes[bed_i].position_m
    dr_car = asm.nodes[th_i].position_m - asm.nodes[bed_i].position_m

    m_cog = max(float(asm.nodes[cog_i].lumped_mass_kg), 0.0)
    m_noz = max(float(asm.nodes[noz_i].lumped_mass_kg), 0.0) if noz_i != cog_i else 0.0
    noz_frac = m_noz / (m_cog + m_noz) if (noz_i != cog_i and (m_cog + m_noz) > 1e-9) else 0.0
    fx = np.zeros(K.shape[0])
    fx[6 * cog_i + 0] = 1.0 - noz_frac
    if noz_frac > 0.0:
        fx[6 * noz_i + 0] += noz_frac
    cx = np.zeros(K.shape[0])
    cx[6 * noz_i + 0] = 1.0
    cx[6 * bed_i + 0] -= 1.0
    cx[6 * bed_i + 4] -= float(dr[2])
    cx[6 * bed_i + 5] += float(dr[1])
    cx_car = np.zeros(K.shape[0])
    cx_car[6 * th_i + 0] = 1.0
    cx_car[6 * bed_i + 0] -= 1.0
    cx_car[6 * bed_i + 4] -= float(dr_car[2])
    cx_car[6 * bed_i + 5] += float(dr_car[1])
    u_x = np.linalg.solve(K_ff, fx[free_dofs])
    c_noz_x = float(cx[free_dofs] @ u_x)
    c_car_x = float(cx_car[free_dofs] @ u_x)
    # Match solve_modal_fea non-collocated pitch-sign guard (Pitfall 3C): c_car + |c_noz - c_car|
    c_struct_x = c_car_x + abs(c_noz_x - c_car_x)

    k_belt_x, k_motor_x, _ = compute_drive_stiffness_n_per_m(asm, "X")
    k_bracket = compute_bracket_stiffness_n_per_m(asm)
    det_x = compute_drive_stiffness_details_n_per_m(asm, "X")
    r_p = spec.gantry.belt.pitch_radius_m
    m_rot = 2.0 * (spec.kinematics.xy_stepper.rotor_inertia_kg_m2 / (r_p**2))
    m_x = asm.x_moving_mass_kg + float(det_x["belt_moving_mass_kg"])
    # Exact 2-DOF [m_rot, m_x] coupled drive eigenproblem (replaces biased 1-DOF Rayleigh)
    k_bb = 1.0 / (1.0 / k_belt_x + 1.0 / k_bracket)
    b_q = (k_motor_x + k_bb) / m_rot + k_bb / m_x
    c_q = (k_motor_x * k_bb) / (m_rot * m_x)
    w1_sq = 0.5 * (b_q - math.sqrt(max(b_q * b_q - 4.0 * c_q, 0.0)))
    c_drv_exact = 1.0 / (m_x * w1_sq)
    alpha_1 = k_bb / (k_motor_x + k_bb - w1_sq * m_rot)
    u_mot = 0.5 * k_motor_x * (alpha_1**2)
    u_bb = 0.5 * k_bb * ((1.0 - alpha_1) ** 2)
    u_belt = u_bb * (k_bb / k_belt_x)
    u_bkt = u_bb * (k_bb / k_bracket)
    u_tot = u_mot + u_belt + u_bkt
    c_total_x = c_struct_x + c_drv_exact
    f_eff_exact = 1.0 / (2.0 * math.pi * math.sqrt(m_x * c_total_x))

    assert mo.shaper_x.effective_freq_hz == pytest.approx(f_eff_exact, rel=1e-6)

    # Verify old mismatched Dunkerley (1/f_struct^2 + 1/f_drive^2) differed substantially (>20%)
    f_dunk_buggy = 1.0 / math.sqrt(
        1.0 / mo.shaper_x.structural_freq_hz**2 + 1.0 / mo.shaper_x.drive_freq_hz**2
    )
    assert f_eff_exact > f_dunk_buggy * 1.25

    # Verify compliance shares reflect exact mode-1 modal strain energies summing to 100%
    shares = mo.shaper_x.compliance_share_pct
    assert sum(shares.values()) == pytest.approx(100.0, abs=0.2)
    assert shares["frame"] == pytest.approx(100.0 * c_struct_x / c_total_x, abs=0.2)
    assert shares["belt"] == pytest.approx(100.0 * (c_drv_exact * u_belt / u_tot) / c_total_x, abs=0.2)
    assert shares["motor"] == pytest.approx(100.0 * (c_drv_exact * u_mot / u_tot) / c_total_x, abs=0.2)
    assert shares["bracket"] == pytest.approx(100.0 * (c_drv_exact * u_bkt / u_tot) / c_total_x, abs=0.2)


def test_b02_mode_count_convergence_and_no_overhang_double_count() -> None:
    """Verify f_eff has zero mode-truncation drift across num_modes and structural_freq_hz is converged."""
    for yml in ("corexy_voron_style.yaml", "voron_trident_300.yaml", "delta_racer.yaml"):
        spec = PrinterSpec.from_yaml(EXAMPLES_DIR / yml)
        asm = build_assembly(spec)
        mo_lo = solve_modal_fea(asm, num_modes=6)
        mo_hi = solve_modal_fea(asm, num_modes=24)
        assert mo_lo.shaper_x is not None and mo_hi.shaper_x is not None
        assert mo_lo.shaper_y is not None and mo_hi.shaper_y is not None
        assert mo_lo.shaper_x.effective_freq_hz == pytest.approx(
            mo_hi.shaper_x.effective_freq_hz, rel=1e-6
        )
        assert mo_lo.shaper_y.effective_freq_hz == pytest.approx(
            mo_hi.shaper_y.effective_freq_hz, rel=1e-6
        )
        assert mo_lo.shaper_x.structural_freq_hz == pytest.approx(
            mo_hi.shaper_x.structural_freq_hz, rel=1e-3
        )


# ---------------------------------------------------------------------------
# B-11: Per-Node Rotational Inertia & Fox-Kapoor Mass Sum Rule
# ---------------------------------------------------------------------------


def test_b11_bed_plate_rotational_inertia_and_fox_kapoor_sum_rule() -> None:
    """Verify rectangular/disk plate rotary inertia on bed_surface and exact Fox-Kapoor sum rule."""
    # 1. CoreXY rectangular 300x300 mm bed plate
    spec_rect = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    asm_rect = build_assembly(spec_rect)
    bed_node = asm_rect.node_map[asm_rect.bed_node_id]
    bx = spec_rect.bed.size_mm[0] * 1e-3
    by = spec_rect.bed.size_mm[1] * 1e-3
    m_b = bed_node.lumped_mass_kg
    ixx, iyy, izz = node_rotational_inertia_xyz(bed_node, asm_rect)
    assert ixx == pytest.approx(m_b * (by**2) / 12.0, rel=1e-9)
    assert iyy == pytest.approx(m_b * (bx**2) / 12.0, rel=1e-9)
    assert izz == pytest.approx(m_b * (bx**2 + by**2) / 12.0, rel=1e-9)

    noz_node = asm_rect.node_map[asm_rect.nozzle_node_id]
    assert node_rotational_inertia_xyz(noz_node, asm_rect)[0] == pytest.approx(
        noz_node.lumped_mass_kg * (0.007**2), rel=1e-9
    )

    # 2. Delta circular disk bed plate
    spec_delta = PrinterSpec.from_yaml(EXAMPLES_DIR / "delta_racer.yaml")
    asm_delta = build_assembly(spec_delta)
    bed_delta = asm_delta.node_map[asm_delta.bed_node_id]
    r_disk = 0.25 * (spec_delta.bed.size_mm[0] + spec_delta.bed.size_mm[1]) * 1e-3
    d_ixx, d_iyy, d_izz = node_rotational_inertia_xyz(bed_delta, asm_delta)
    assert d_ixx == pytest.approx(bed_delta.lumped_mass_kg * (r_disk**2) / 4.0, rel=1e-9)
    assert d_iyy == pytest.approx(d_ixx, rel=1e-9)
    assert d_izz == pytest.approx(bed_delta.lumped_mass_kg * (r_disk**2) / 2.0, rel=1e-9)

    # 3. Fox-Kapoor mass sensitivity identity holds across M_ff including bed_surface plate inertia
    sen = compute_structural_sensitivities(asm_rect)
    sum_mass_sens = (
        sum(v["mass_sens_hz"] for v in sen.member_sensitivities.values())
        + sum(v["mass_sens_hz"] for v in sen.panel_bay_sensitivities.values())
        + sum(v["mass_sens_hz"] for v in sen.lumped_mass_sensitivities.values())
    )
    assert -2.0 * sum_mass_sens == pytest.approx(sen.fundamental_freq_hz, rel=1e-7)


# ---------------------------------------------------------------------------
# B-12: Hertzian 4-Row Factor & Convex-Shaft Round Linear Bushings
# ---------------------------------------------------------------------------


def test_b12_hgh15_datasheet_stiffness_and_rod_bushing_contact() -> None:
    """Verify HGH15C Z0/ZA match HIWIN HG catalog (~196/365 N/um) and steel_rod bushings model convex shaft."""
    hg_z0 = hertzian_rail_carriage_stiffness("hgh15_rail", preload="Z0", block_type="C")
    hg_z1 = hertzian_rail_carriage_stiffness("hgh15_rail", preload="Z1", block_type="C")
    hg_za = hertzian_rail_carriage_stiffness("hgh15_rail", preload="ZA", block_type="C")
    hg_zb = hertzian_rail_carriage_stiffness("hgh15_rail", preload="ZB", block_type="C")

    # HIWIN HG15C catalog radial stiffness (normal to mounting face -> k_zz): Z0 = 196 N/um, ZA = 365 N/um
    k_z0 = hg_z0["k_zz_n_m"] * 1e-6
    k_z1 = hg_z1["k_zz_n_m"] * 1e-6
    k_za = hg_za["k_zz_n_m"] * 1e-6
    k_zb = hg_zb["k_zz_n_m"] * 1e-6
    assert 180.0 <= k_z0 <= 215.0
    assert 310.0 <= k_za <= 390.0
    assert k_z0 < k_z1 < k_za < k_zb
    assert hg_z0["k_yy_n_m"] < hg_z1["k_yy_n_m"] < hg_za["k_yy_n_m"] < hg_zb["k_yy_n_m"]

    mgn15_z1 = hertzian_rail_carriage_stiffness("mgn15_rail", preload="Z1", block_type="C")
    assert mgn15_z1["k_zz_n_m"] * 1e-6 < k_z1
    assert mgn15_z1["k_yy_n_m"] < hg_z1["k_yy_n_m"]

    # Round LMxUU ball bushings on convex shaft have materially lower stiffness than profiled MGN rails
    rod8 = hertzian_rail_carriage_stiffness("steel_rod_8mm", preload="Z1", block_type="C")["k_yy_n_m"] * 1e-6
    rod10 = hertzian_rail_carriage_stiffness("steel_rod_10mm", preload="Z1", block_type="C")["k_yy_n_m"] * 1e-6
    rod12 = hertzian_rail_carriage_stiffness("steel_rod_12mm", preload="Z1", block_type="C")["k_yy_n_m"] * 1e-6
    mgn9_z1 = hertzian_rail_carriage_stiffness("mgn9_rail", preload="Z1", block_type="C")["k_yy_n_m"] * 1e-6

    assert rod8 < rod10 < rod12
    assert rod12 < 45.0 < mgn9_z1

    # Strict monotonicity across ALL block types (C and H), ALL preloads (ZF..ZB), and axisymmetric k_yy == k_zz
    for bt in ("C", "H"):
        for pre in ("ZF", "Z0", "Z1", "ZA", "ZB"):
            c8 = hertzian_rail_carriage_stiffness("steel_rod_8mm", preload=pre, block_type=bt)
            c10 = hertzian_rail_carriage_stiffness("steel_rod_10mm", preload=pre, block_type=bt)
            c12 = hertzian_rail_carriage_stiffness("steel_rod_12mm", preload=pre, block_type=bt)
            for k_key in ("k_yy_n_m", "k_zz_n_m", "k_rx_nm_rad", "k_ry_nm_rad", "k_rz_nm_rad"):
                assert c8[k_key] < c10[k_key] < c12[k_key]
            assert c8["k_yy_n_m"] == pytest.approx(c8["k_zz_n_m"], rel=1e-12)
            assert c10["k_yy_n_m"] == pytest.approx(c10["k_zz_n_m"], rel=1e-12)
            assert c12["k_yy_n_m"] == pytest.approx(c12["k_zz_n_m"], rel=1e-12)


# ---------------------------------------------------------------------------
# B-13: Taco-Bow Euler Buckling Threshold & Unit-Suffixed Restraint
# ---------------------------------------------------------------------------


def test_b13_taco_buckling_euler_threshold_and_unit_suffix() -> None:
    """Verify unit-suffixed k_restraint_thermal_n_per_m, sub-Pcr bow suppression, and post-Pcr buckling."""
    for mid, cfg in BED_MOUNT_CATALOG.items():
        assert "k_restraint_thermal_n_per_m" in cfg, f"Missing unit-suffixed field on {mid}"
        assert float(cfg["k_restraint_thermal_n_per_m"]) >= 0.0

    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    base.frame.chamber_temp_c = 25.0
    base.bed.mount_type = "rigid_4pt"

    # Thick 12mm plate at 60 C is well below Euler critical buckling load -> minor eccentric bow only
    s_12_60 = base.model_copy(deep=True)
    s_12_60.bed.plate_thickness_mm = 12.0
    s_12_60.bed.target_temp_c = 60.0
    _, taco_12_60 = compute_bed_thermal_bow_and_taco_um(s_12_60)
    assert taco_12_60 < 25.0

    # Thin 3mm plate at 110 C on rigid_4pt exceeds Euler threshold and experiences strong post-buckling bow
    s_3_110 = base.model_copy(deep=True)
    s_3_110.bed.plate_thickness_mm = 3.0
    s_3_110.bed.target_temp_c = 110.0
    _, taco_3_110 = compute_bed_thermal_bow_and_taco_um(s_3_110)
    assert taco_3_110 > 800.0

    # Strict ordering at 100 C on 8mm MIC-6 plate
    ordered = (
        "kinematic_3pt",
        "springs_4pt",
        "silicone_4pt",
        "spherical_3pt_ratrig",
        "floating_4pt",
        "trident_3pt_standoffs",
        "rigid_4pt",
    )
    tacos = []
    for mt in ordered:
        s_mt = base.model_copy(deep=True)
        s_mt.bed.plate_thickness_mm = 8.0
        s_mt.bed.target_temp_c = 100.0
        s_mt.bed.mount_type = mt  # type: ignore[assignment]
        tacos.append(compute_bed_thermal_bow_and_taco_um(s_mt)[1])
    assert tacos[0] == 0.0
    for idx in range(len(tacos) - 1):
        assert tacos[idx + 1] > tacos[idx]


# ---------------------------------------------------------------------------
# B-14: Feet-Sprung Receptance Without Fixed-45 Hz Heuristic
# ---------------------------------------------------------------------------


def test_b14_feet_sprung_receptance_no_fixed_45hz_penalty() -> None:
    """Verify benchtop feet use feet-sprung FE receptance + physical series compliance (no 45 Hz penalty)."""
    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    mods = {}
    for ft in ("rigid_bolted", "paver_on_foam", "rubber_feet", "anti_vib_pads"):
        s = base.model_copy(deep=True)
        s.frame.feet_type = ft  # type: ignore[assignment]
        mods[ft] = solve_modal_fea(build_assembly(s))

    assert mods["paver_on_foam"].shaper_x.structural_freq_hz < mods["rigid_bolted"].shaper_x.structural_freq_hz
    assert mods["rubber_feet"].shaper_x.structural_freq_hz < mods["rigid_bolted"].shaper_x.structural_freq_hz
    assert mods["anti_vib_pads"].shaper_x.structural_freq_hz < mods["rubber_feet"].shaper_x.structural_freq_hz
    # Mount modes couple through their FE modal receptance (inertia relief + frame recoil), so a
    # softer mount leaves less quasi-static mount compliance in the limiting axis and the heavier
    # paver base recoils less than bare feet.
    for ft in ("paver_on_foam", "rubber_feet", "anti_vib_pads"):
        md = mods[ft].mounting_dynamics
        assert md is not None and len(md["rigid_body_modes"]) >= 5
        assert 0.0 < md["x"]["frame_recoil_ratio"] < 0.2
        ratio = mods[ft].shaper_x.effective_freq_hz / mods["rigid_bolted"].shaper_x.effective_freq_hz
        assert 0.90 <= ratio <= 1.15
    assert (
        mods["paver_on_foam"].mounting_dynamics["x"]["frame_recoil_ratio"]
        < mods["rubber_feet"].mounting_dynamics["x"]["frame_recoil_ratio"]
    )
    assert mods["rigid_bolted"].mounting_dynamics is None


# ---------------------------------------------------------------------------
# C-04 & C-08: Linear Stage 2 Body Stiffness & Rayleigh Sensitivity Gain
# ---------------------------------------------------------------------------


def test_c04_linear_body_stiffness_without_sub350_power_law_distortion() -> None:
    """Verify internal_stiffness_n_per_mm scales Stage 2 body stiffness linearly below & above 350 N/mm."""
    desc = ToolheadSpec.model_fields["internal_stiffness_n_per_mm"].description or ""
    assert "Stage 2" in desc
    assert "body_construction is None" in desc

    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    base.toolhead.body_construction = None
    ks = {}
    for k_in in (100.0, 175.0, 250.0, 350.0, 700.0):
        s = base.model_copy(deep=True)
        s.toolhead.internal_stiffness_n_per_mm = k_in
        props = compute_toolhead_link_props(s, l_mount_m=0.025, l_stem_m=0.040)
        ks[k_in] = props["k_int_n_m"]

    assert ks[350.0] / ks[175.0] == pytest.approx(2.0, rel=1e-9)
    assert ks[250.0] / ks[100.0] == pytest.approx(2.5, rel=1e-9)
    assert ks[700.0] / ks[350.0] == pytest.approx(2.0, rel=1e-9)


def test_c08_modal_root_cause_est_hz_gain_uses_rayleigh_bound() -> None:
    """Verify ModeShape stiffness est_hz_gain matches Rayleigh f*(sqrt(1 + 0.5*s) - 1)."""
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    mo = solve_modal_fea(build_assembly(spec))
    m1 = mo.modes[0]
    stiff_top = max(
        (c for c in m1.top_causes if c["kind"] == "stiffness"),
        key=lambda c: c["share_pct"],
    )
    share_frac = stiff_top["share_pct"] / 100.0
    expected_rayleigh = m1.frequency_hz * (math.sqrt(1.0 + 0.5 * share_frac) - 1.0)
    # 0.1% share rounding gives at most ~0.02 Hz uncertainty on top of 2-decimal rounding
    assert stiff_top["est_hz_gain"] == pytest.approx(expected_rayleigh, abs=0.02)
