"""Batch 2A structural calibration tests: CAL-09 (corner joints), CAL-04 (frame feet),
CAL-05 (rail quality clearance/ball-sharing), and CAL-06 (extrusion cavity fillings).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from printsim.fea.components import (
    RAIL_RADIAL_CLEARANCE_UM_TABLE,
    corner_joint_stiffness,
    extrusion_filling_composite_properties,
    frame_feet_isolator_mechanics,
    hertzian_rail_carriage_stiffness,
    iwan_joint_damping_delta,
)
from printsim.fea.modal import solve_modal_fea
from printsim.fea.solver import solve_static_fea
from printsim.geometry.builder import build_assembly
from printsim.schema.printer import (
    CORNER_JOINT_BASE_2020,
    EXTRUSION_FILLINGS,
    FRAME_FEET_CATALOG,
    RAIL_QUALITY_CATALOG,
    PrinterSpec,
)

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


# ---------------------------------------------------------------------------
# CAL-09: 3D Tet10-Calibrated 2020 Corner Joint Stiffnesses
# ---------------------------------------------------------------------------


def test_cal09_corner_joint_tet10_calibrated_values_and_hierarchy() -> None:
    """Verify Tet10-calibrated 2020 corner joint stiffness hierarchy and physical bounds."""
    joints = ("l_bracket", "blind_joint", "gusset_plate", "corner_cube", "rigid")
    results = [corner_joint_stiffness("extrusion_2020", j) for j in joints]

    # Strict monotonicity across all 3 stiffness DOFs (translational, in-plane, torsional)
    for i in range(len(joints) - 1):
        assert results[i + 1]["k_t_n_m"] > results[i]["k_t_n_m"]
        assert results[i + 1]["k_r_inplane_nm_rad"] > results[i]["k_r_inplane_nm_rad"]
        assert results[i + 1]["k_r_torsion_nm_rad"] > results[i]["k_r_torsion_nm_rad"]
        assert iwan_joint_damping_delta(joints[i + 1]) < iwan_joint_damping_delta(joints[i])

    # Tet10 calibrated 2020 bounds (1.65 mm T-slot lip flexure + Back-Burdekin/Yoshimura contact)
    assert 2000.0 <= CORNER_JOINT_BASE_2020["l_bracket"][1] <= 4000.0
    assert 10000.0 <= CORNER_JOINT_BASE_2020["blind_joint"][1] <= 18000.0
    assert 30000.0 <= CORNER_JOINT_BASE_2020["gusset_plate"][1] <= 45000.0
    assert 75000.0 <= CORNER_JOINT_BASE_2020["corner_cube"][1] <= 105000.0

    # Torsional stiffness stays physically below in-plane bending stiffness for all non-rigid joints
    for j in ("l_bracket", "blind_joint", "gusset_plate", "corner_cube"):
        k_t, k_r_in, k_r_tor = CORNER_JOINT_BASE_2020[j]
        assert k_t > 0.0
        assert 0.10 * k_r_in <= k_r_tor <= 0.55 * k_r_in

    # Larger profiles (3030, 4040) scale monotonically above 2020
    cj_2020 = corner_joint_stiffness("extrusion_2020", "blind_joint")
    cj_3030 = corner_joint_stiffness("extrusion_3030", "blind_joint")
    cj_4040 = corner_joint_stiffness("extrusion_4040", "blind_joint")
    assert cj_2020["k_r_inplane_nm_rad"] < cj_3030["k_r_inplane_nm_rad"] < cj_4040["k_r_inplane_nm_rad"]
    assert cj_2020["k_t_n_m"] < cj_3030["k_t_n_m"] < cj_4040["k_t_n_m"]


# ---------------------------------------------------------------------------
# CAL-04: Benchtop Isolator Feet (HULA / Sorbothane / Rubber / Paver-on-Foam)
# ---------------------------------------------------------------------------


def test_cal04_frame_feet_isolator_stiffness_loss_factor_and_modal_sway() -> None:
    """Verify Sorbothane/HULA, 65A rubber, and 18 kg paver-on-foam isolator frequencies and loss factors."""
    m_printer = 20.0
    mech_rub = frame_feet_isolator_mechanics("rubber_feet", printer_mass_kg=m_printer)
    mech_av = frame_feet_isolator_mechanics("anti_vib_pads", printer_mass_kg=m_printer)
    mech_pav = frame_feet_isolator_mechanics("paver_on_foam", printer_mass_kg=m_printer)
    mech_rig = frame_feet_isolator_mechanics("rigid_bolted", printer_mass_kg=m_printer)

    assert math.isinf(mech_rig["f_sway_hz"]) and math.isinf(mech_rig["f_bounce_hz"])

    # HULA / Sorbothane 50D: ~8-13 Hz lateral sway isolation, high viscoelastic loss factor tan(delta) ~ 0.56
    assert 8.0 <= mech_av["f_sway_hz"] <= 13.0
    assert 14.0 <= mech_av["f_bounce_hz"] <= 24.0
    assert 0.50 <= mech_av["loss_factor"] <= 0.66
    assert mech_av["mount_damping_ratio"] == pytest.approx(0.5 * mech_av["loss_factor"], rel=1e-6)

    # 18 kg concrete paver on 25 mm foam: ~11-17 Hz lateral isolation on 20+18=38 kg total mass
    assert mech_pav["base_mass_kg"] == pytest.approx(18.0, abs=1e-6)
    assert mech_pav["effective_isolated_mass_kg"] == pytest.approx(38.0, abs=1e-6)
    assert 11.0 <= mech_pav["f_sway_hz"] <= 17.5
    assert 0.22 <= mech_pav["loss_factor"] <= 0.36

    # Standard 65A rubber / TPU feet: stiffer ~22-30 Hz sway, lower elastomer loss factor ~0.12
    assert 22.0 <= mech_rub["f_sway_hz"] <= 30.0
    assert 0.08 <= mech_rub["loss_factor"] <= 0.16

    # Frequency ordering: HULA/Sorbothane < Paver+Foam < Rubber Feet
    assert mech_av["f_sway_hz"] < mech_pav["f_sway_hz"] < mech_rub["f_sway_hz"]
    # Loss factor ordering: Rubber < Paver+Foam < HULA/Sorbothane
    assert mech_rub["loss_factor"] < mech_pav["loss_factor"] < mech_av["loss_factor"]

    # Full FE modal check: mounting_dynamics exposes calibrated mount_loss_factor
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    spec.frame.feet_type = "anti_vib_pads"
    modal = solve_modal_fea(build_assembly(spec))
    assert modal.mounting_dynamics is not None
    assert modal.mounting_dynamics["mount_loss_factor"] == pytest.approx(
        FRAME_FEET_CATALOG["anti_vib_pads"]["loss_factor"], abs=1e-3
    )
    assert 6.0 <= modal.mounting_dynamics["lowest_sway_hz"] <= 18.0


# ---------------------------------------------------------------------------
# CAL-05: HIWIN / THK / NSK Radial Clearance & Ball-Tolerance Contact Tables
# ---------------------------------------------------------------------------


def test_cal05_rail_quality_preload_dependent_contact_tables() -> None:
    """Verify non-flat preload-dependent Hertzian ball-sharing replaces flat scalar multipliers."""
    # 1. HIWIN genuine remains exact 1.000 reference across all rail sizes and preloads
    for rail in ("mgn9_rail", "mgn12_rail", "mgn15_rail", "hgh15_rail"):
        for pre in ("ZF", "Z0", "Z1", "ZA", "ZB"):
            res = hertzian_rail_carriage_stiffness(rail, preload=pre, block_type="H", rail_quality="hiwin_genuine")
            assert res["effective_quality_stiffness_ratio"] == pytest.approx(1.0, rel=1e-9)
            assert res["effective_radial_clearance_um"] == pytest.approx(
                RAIL_RADIAL_CLEARANCE_UM_TABLE[rail][pre], abs=1e-9
            )

    # 2. Strict quality hierarchy across every preload (budget < midgrade < hiwin < precision THK/NSK)
    for pre in ("ZF", "Z0", "Z1", "ZA", "ZB"):
        c_bud = hertzian_rail_carriage_stiffness("mgn12_rail", preload=pre, rail_quality="budget_generic")
        c_mid = hertzian_rail_carriage_stiffness("mgn12_rail", preload=pre, rail_quality="midgrade_matched")
        c_hiw = hertzian_rail_carriage_stiffness("mgn12_rail", preload=pre, rail_quality="hiwin_genuine")
        c_thk = hertzian_rail_carriage_stiffness("mgn12_rail", preload=pre, rail_quality="precision_thk_nsk")

        assert c_bud["k_yy_n_m"] < c_mid["k_yy_n_m"] < c_hiw["k_yy_n_m"] < c_thk["k_yy_n_m"]
        assert c_bud["k_rx_nm_rad"] < c_mid["k_rx_nm_rad"] < c_hiw["k_rx_nm_rad"] < c_thk["k_rx_nm_rad"]
        assert c_bud["effective_radial_clearance_um"] > c_mid["effective_radial_clearance_um"] > c_hiw["effective_radial_clearance_um"] > c_thk["effective_radial_clearance_um"]

    # 3. Non-flat preload dependence (the whole point of replacing scalar multipliers!):
    # On loose budget clones (4.2 um scatter), heavier preload ZA/ZB compresses more balls into
    # active load-sharing than light Z0/ZF preload -> higher effective quality ratio & active fraction.
    bud_z0 = hertzian_rail_carriage_stiffness("mgn12_rail", preload="Z0", rail_quality="budget_generic")
    bud_z1 = hertzian_rail_carriage_stiffness("mgn12_rail", preload="Z1", rail_quality="budget_generic")
    bud_za = hertzian_rail_carriage_stiffness("mgn12_rail", preload="ZA", rail_quality="budget_generic")
    assert bud_z0["effective_quality_stiffness_ratio"] < bud_z1["effective_quality_stiffness_ratio"] < bud_za["effective_quality_stiffness_ratio"]
    assert bud_z0["active_ball_fraction"] <= bud_z1["active_ball_fraction"] <= bud_za["active_ball_fraction"]
    assert bud_za["effective_quality_stiffness_ratio"] - bud_z0["effective_quality_stiffness_ratio"] > 0.08

    # 4. THK/NSK Grade P tight conformity + G5 balls maintain near-100% active ball fraction
    thk_z0 = hertzian_rail_carriage_stiffness("mgn12_rail", preload="Z0", rail_quality="precision_thk_nsk")
    thk_za = hertzian_rail_carriage_stiffness("mgn12_rail", preload="ZA", rail_quality="precision_thk_nsk")
    assert thk_z0["active_ball_fraction"] >= 0.99
    assert thk_z0["groove_conformity_f"] < RAIL_QUALITY_CATALOG["hiwin_genuine"]["conformity_f"]
    assert 1.10 <= thk_za["effective_quality_stiffness_ratio"] <= 1.35
    assert thk_z0["effective_quality_stiffness_ratio"] > thk_za["effective_quality_stiffness_ratio"]

    # 5. Explicit user stiffness_mult_override scales cleanly when different from catalog baseline
    base_hiw = hertzian_rail_carriage_stiffness("mgn12_rail", preload="Z1", rail_quality="hiwin_genuine")
    ov_half = hertzian_rail_carriage_stiffness(
        "mgn12_rail", preload="Z1", rail_quality="hiwin_genuine", stiffness_mult_override=0.50
    )
    assert ov_half["k_yy_n_m"] < base_hiw["k_yy_n_m"] * 0.65


# ---------------------------------------------------------------------------
# CAL-06: Extrusion Cavity Fillings — Bond Efficiency, Composite EI & Damping
# ---------------------------------------------------------------------------


def test_cal06_extrusion_filling_composite_ei_bond_efficiency_and_damping() -> None:
    """Verify bond-efficiency composite EI_eff (epoxy granite vs UHPC) and tap-test damping order."""
    for prof_id in ("extrusion_2020", "extrusion_4040"):
        c_none = extrusion_filling_composite_properties(prof_id, "none")
        c_foam = extrusion_filling_composite_properties(prof_id, "pu_foam")
        c_sand = extrusion_filling_composite_properties(prof_id, "silica_sand")
        c_lead = extrusion_filling_composite_properties(prof_id, "lead_shot")
        c_eg = extrusion_filling_composite_properties(prof_id, "epoxy_granite")
        c_uhpc = extrusion_filling_composite_properties(prof_id, "uhpc_grout")

        # Unbonded fills contribute zero composite EI gain
        for unb in (c_none, c_foam, c_sand, c_lead):
            assert unb["delta_iy_m4"] == 0.0
            assert unb["ei_gain_pct"] == 0.0
            assert unb["bond_efficiency"] == 0.0

        # Epoxy granite has higher adhesive bond efficiency than cementitious UHPC (shrinkage slip),
        # while UHPC's higher E (42 vs 30 GPa) still delivers a larger net composite EI gain
        # (~16-20% on 2020; ~26-33% on 4040 due to larger hollow core void fraction)
        assert c_eg["bond_efficiency"] > c_uhpc["bond_efficiency"] > 0.70
        assert 10.0 <= c_eg["ei_gain_pct"] <= 30.0
        assert 12.0 <= c_uhpc["ei_gain_pct"] <= 36.0
        assert c_uhpc["ei_gain_pct"] > c_eg["ei_gain_pct"]

        # Tap-test calibrated damping ordering: none < pu_foam < silica_sand < uhpc_grout < epoxy_granite < lead_shot
        assert (
            c_none["damping_boost"]
            < c_foam["damping_boost"]
            < c_sand["damping_boost"]
            < c_uhpc["damping_boost"]
            < c_eg["damping_boost"]
            < c_lead["damping_boost"]
        )
        # Literature damping ranges (Cortes/Suh/Friend/Marhadi/Mohring)
        assert 0.005 <= c_foam["damping_boost"] <= 0.012
        assert 0.015 <= c_sand["damping_boost"] <= 0.028
        assert 0.020 <= c_uhpc["damping_boost"] <= 0.030
        assert 0.024 <= c_eg["damping_boost"] <= 0.035
        assert 0.032 <= c_lead["damping_boost"] <= 0.044

    # End-to-end assembly & static FEA check: bonded fill stiffens open CoreXY frame without touching moving mass
    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    base.frame.enclosed_panels = False
    base.frame.bracing = "none"
    s_eg = base.model_copy(deep=True)
    s_eg.frame.extrusion_filling = "epoxy_granite"
    s_uhpc = base.model_copy(deep=True)
    s_uhpc.frame.extrusion_filling = "uhpc_grout"

    asm_0 = build_assembly(base)
    asm_eg = build_assembly(s_eg)
    asm_uhpc = build_assembly(s_uhpc)
    assert asm_eg.x_moving_mass_kg == pytest.approx(asm_0.x_moving_mass_kg, rel=1e-9)
    assert solve_static_fea(asm_uhpc).toolhead_stiffness_x_n_per_mm > solve_static_fea(asm_eg).toolhead_stiffness_x_n_per_mm > solve_static_fea(asm_0).toolhead_stiffness_x_n_per_mm
    assert asm_eg.frame_damping_ratio > asm_uhpc.frame_damping_ratio > asm_0.frame_damping_ratio
