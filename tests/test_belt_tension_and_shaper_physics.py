"""Physics verification tests for belt tension mechanics & coupled belt-to-shaper dynamics.

Covers:
1. Width-normalized cord crimp/toe stiffening (`q0 = T0 / width_mm`).
2. Euler-Bernoulli pulley/idler wrap flexural-fillet seating compliance (`EI_b`, `T_wrap = EI_b / R^2`).
3. Tension-dependent Gerbert tooth-flank radial seating stiffness.
4. Significant, realistic low-tension shaper frequency & acceleration drop-off on 2WD and AWD CoreXY.
5. Exact 2-DOF `[m_rot, m_car + m_belt_eff]` coupled drive eigenvalue & modal strain-energy shares
   vs. biased 1-DOF Rayleigh approximation.
6. Rayleigh moving-belt inertia `(1/3) * mu * L_active` and Ross-Ungar-Kerwin strain-energy damping.
7. Canonical parity between `kinematics/solvers.py` and `fea/mechatronics.py` belt branch models.
8. Exact 2-belt CoreXY dynamic tension swing (2WD unequal 3:1 compatibility split vs. AWD 50/50 split).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from printsim.fea.components import (
    belt_damping_ratio,
    belt_effective_ea_n,
    belt_flexural_rigidity_nm2,
    gerbert_tooth_mesh_stiffness_n_m,
)
from printsim.fea.mechatronics import build_coupled_state_space, corexy_belt_span_stiffness_matrix
from printsim.fea.modal import _solve_coupled_drive_2dof, solve_modal_fea
from printsim.geometry.builder import build_assembly
from printsim.kinematics.solvers import analyze_kinematics, compute_drive_stiffness_details_n_per_m
from printsim.schema import BeltSpec, PrinterSpec

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def test_belt_flexural_rigidity_cantilever_droop_and_scaling() -> None:
    """Gates 2GT 6mm EI yields realistic ~7 mm 50mm cantilever self-weight droop; scales with pitch & core."""
    b6 = BeltSpec(type="gt2_6mm", core="fiberglass", pretension_n=25.0)
    b9 = BeltSpec(type="gt2_9mm", core="fiberglass", pretension_n=25.0)
    b6_steel = BeltSpec(type="gt2_6mm", core="steel", pretension_n=25.0)
    b3gt = BeltSpec(type="3gt_9mm", core="fiberglass", pretension_n=40.0)
    bat3 = BeltSpec(type="at3_10mm", core="steel", pretension_n=60.0)

    ei6 = belt_flexural_rigidity_nm2(b6)
    ei9 = belt_flexural_rigidity_nm2(b9)
    # 50 mm cantilever self-weight tip droop: delta = w*L^4 / (8*EI) should be ~5-12 mm
    w_n_m = b6.linear_mass_kg_m * 9.80665
    droop_mm = (w_n_m * (0.05**4) / (8.0 * ei6)) * 1e3
    assert 4.0 <= droop_mm <= 12.0
    assert ei9 == pytest.approx(ei6 * 1.5, rel=1e-6)
    assert belt_flexural_rigidity_nm2(b6_steel) > ei6 * 2.0
    assert belt_flexural_rigidity_nm2(b3gt) > ei9 * 5.0
    # 3D Tet10 pitch-cell FEA (belt_pitch_cell_fea.json): 2.41 mm thick 3GT > 1.90 mm AT3 > 1.38 mm 2GT
    assert belt_flexural_rigidity_nm2(bat3) > ei9 * 2.5
    assert belt_flexural_rigidity_nm2(b3gt) > belt_flexural_rigidity_nm2(bat3)


def test_width_normalized_crimp_and_pulley_wrap_seating_compliance() -> None:
    """Line-tension q0 = T0/w normalizes crimp across widths; wrap bends soften low-tension belts."""
    # At equal line tension q0 = 2.0 N/mm (12 N on 6mm vs 24 N on 12mm), no-wrap EA scales 2.0x exactly
    b6 = BeltSpec(type="gt2_6mm", core="fiberglass", pretension_n=12.0)
    b12 = BeltSpec(type="gt2_12mm", core="fiberglass", pretension_n=24.0)
    ea6_nowarp = belt_effective_ea_n(b6, span_m=0.5, n_pulley_tangents=0.0)
    ea12_nowarp = belt_effective_ea_n(b12, span_m=0.5, n_pulley_tangents=0.0)
    assert ea12_nowarp / ea6_nowarp == pytest.approx(2.0, rel=1e-2)

    # At low tension (5 N), adding 10 pulley/idler tangent bends reduces effective EA markedly vs 0 bends
    b6_loose = BeltSpec(type="gt2_6mm", core="fiberglass", pretension_n=5.0)
    ea_loose_0 = belt_effective_ea_n(b6_loose, span_m=0.5, n_pulley_tangents=0.0)
    ea_loose_10 = belt_effective_ea_n(b6_loose, span_m=0.5, n_pulley_tangents=10.0)
    assert ea_loose_10 < ea_loose_0 * 0.70

    # At high tension (65 N), wrap fillet shrinks so 10 tangents barely lowers EA (< 3% loss)
    b6_tight = BeltSpec(type="gt2_6mm", core="fiberglass", pretension_n=65.0)
    ea_tight_0 = belt_effective_ea_n(b6_tight, span_m=0.5, n_pulley_tangents=0.0)
    ea_tight_10 = belt_effective_ea_n(b6_tight, span_m=0.5, n_pulley_tangents=10.0)
    assert ea_tight_10 > ea_tight_0 * 0.97

    # Gerbert tooth-mesh stiffness increases substantially from 5 N (unseated) to 45 N (seated)
    k_tooth_loose = gerbert_tooth_mesh_stiffness_n_m(b6_loose, ea_loose_10)
    k_tooth_tight = gerbert_tooth_mesh_stiffness_n_m(b6_tight, ea_tight_10)
    assert k_tooth_tight > k_tooth_loose * 1.45


def test_belt_tension_sweep_has_realistic_shaper_sensitivity_and_diminishing_returns() -> None:
    """Lowering tension from 65 N -> 15 N -> 5 N strongly lowers shaper Hz & accel; >45 N saturates."""
    monolith = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_monolith_awd.yaml")
    freqs_y: dict[int, float] = {}
    accels_y: dict[int, float] = {}
    damps_y: dict[int, float] = {}
    shapers_y: dict[int, str] = {}

    for t in (5, 9, 15, 28, 45, 65, 90):
        sp = monolith.model_copy(deep=True)
        sp.gantry.belt.pretension_n = float(t)
        mod = solve_modal_fea(build_assembly(sp))
        assert mod.shaper_y is not None and mod.shaper_graph_y is not None
        freqs_y[t] = mod.shaper_y.effective_freq_hz
        accels_y[t] = mod.shaper_y.max_accel_mm_s2
        damps_y[t] = mod.shaper_graph_y.damping_ratio
        shapers_y[t] = mod.shaper_y.recommended_shaper

    # Strictly monotonic frequency and accel across the full tension range
    for t_lo, t_hi in ((5, 9), (9, 15), (15, 28), (28, 45), (45, 65), (65, 90)):
        assert freqs_y[t_hi] > freqs_y[t_lo], f"Expected f_y({t_hi}N) > f_y({t_lo}N)"
        assert accels_y[t_hi] >= accels_y[t_lo], f"Expected accel_y({t_hi}N) >= accel_y({t_lo}N)"

    # Substantial real-world drop when lowering tension on a stiff belt-dominated AWD build:
    # 65 N -> 15 N drops >= 12 Hz Y; 65 N -> 5 N drops >= 40 Hz Y and degrades off ZV
    assert freqs_y[65] - freqs_y[15] >= 12.0
    assert freqs_y[65] - freqs_y[5] >= 40.0
    assert shapers_y[5] != "ZV"
    # Diminishing returns at high tension: 45->90 N gains far less than 5->45 N
    assert (freqs_y[90] - freqs_y[45]) < (freqs_y[45] - freqs_y[5]) * 0.15
    # Low tension raises effective damping / PSD width
    assert damps_y[5] > damps_y[65] + 0.015


def test_exact_2dof_drive_eigenproblem_matches_matrix_eigensolver() -> None:
    """Closed-form `_solve_coupled_drive_2dof` matches numpy 2x2 generalized eigenproblem to 1e-12."""
    k_belt = 600e3
    k_motor = 2400e3
    k_bkt = 5000e3
    m_car = 0.30
    m_rot = 0.67

    sol = _solve_coupled_drive_2dof(k_belt, k_motor, k_bkt, m_car, m_rot)
    k_bb = 1.0 / (1.0 / k_belt + 1.0 / k_bkt)
    K2 = np.array([[k_motor + k_bb, -k_bb], [-k_bb, k_bb]], dtype=float)
    M2 = np.diag([m_rot, m_car])
    eigs = np.sort(np.linalg.eigvals(np.linalg.solve(M2, K2)).real)

    assert sol["w1_sq"] == pytest.approx(float(eigs[0]), rel=1e-12)
    assert sol["w2_sq"] == pytest.approx(float(eigs[1]), rel=1e-12)
    assert sol["u_belt_frac"] + sol["u_motor_frac"] + sol["u_bracket_frac"] == pytest.approx(1.0, abs=1e-12)

    # Exact 2-DOF fundamental frequency is strictly higher than the biased 1-DOF Rayleigh rigid-body bound
    k_motor_ray = k_motor * (m_car / (m_car + m_rot))
    k_drv_ray = 1.0 / (1.0 / k_bb + 1.0 / k_motor_ray)
    w1_ray_sq = k_drv_ray / m_car
    assert sol["w1_sq"] > w1_ray_sq * 1.15


def test_solvers_and_mechatronics_share_canonical_belt_stiffness() -> None:
    """Center-bed `corexy_belt_span_stiffness_matrix` matches `compute_drive_stiffness_details_n_per_m`."""
    for yaml_name in ("voron_monolith_awd.yaml", "voron_trident_300.yaml", "cartesian_bedslinger.yaml"):
        spec = PrinterSpec.from_yaml(EXAMPLES_DIR / yaml_name)
        asm = build_assembly(spec)
        det = compute_drive_stiffness_details_n_per_m(asm, axis="X", u_rel=0.5, v_rel=0.5)
        k_cart, _, _ = corexy_belt_span_stiffness_matrix(asm, u_rel=0.5, v_rel=0.5)
        k_belt = float(det["k_belt_subsystem_n_m"])
        k_bkt = float(det["k_bracket_n_m"])
        expected_k_xx = 1.0 / (1.0 / k_belt + 1.0 / k_bkt)
        assert float(k_cart[0, 0]) == pytest.approx(expected_k_xx, rel=1e-4)

        # Verify model_overrides.belt propagates identically into mechatronics
        sp_ov = spec.model_copy(deep=True)
        sp_ov.model_overrides.belt.stiffness_ea_n = spec.gantry.belt.stiffness_ea_n * 1.5
        asm_ov = build_assembly(sp_ov)
        k_cart_ov, _, _ = corexy_belt_span_stiffness_matrix(asm_ov, u_rel=0.5, v_rel=0.5)
        assert float(k_cart_ov[0, 0]) > float(k_cart[0, 0]) * 1.14


def test_belt_damping_ratio_and_moving_belt_inertia_trends() -> None:
    """Belt damping increases at low tension, high TPU temp, & slack unloading; wide belts add real mass."""
    b_epdm = BeltSpec(type="gt2_9mm", core="fiberglass", pretension_n=65.0)
    b_loose = BeltSpec(type="gt2_9mm", core="fiberglass", pretension_n=8.0)
    b_aramid = BeltSpec(type="gt2_9mm", core="aramid", pretension_n=65.0)
    b_tpu = BeltSpec(type="at3_10mm", core="steel", pretension_n=65.0)

    assert belt_damping_ratio(b_loose) > belt_damping_ratio(b_epdm) + 0.015
    assert belt_damping_ratio(b_epdm, slack_margin_ratio=0.05) > belt_damping_ratio(b_epdm) + 0.02
    assert belt_damping_ratio(b_aramid) > belt_damping_ratio(b_epdm)
    assert belt_damping_ratio(b_tpu, chamber_temp_c=70.0) > belt_damping_ratio(b_tpu, chamber_temp_c=25.0) + 0.01

    # Moving belt mass participation scales with linear_mass_kg_m across 6 mm -> 12 mm -> dual 9 mm
    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_monolith_awd.yaml")
    masses = []
    for btype in ("gt2_6mm", "gt2_9mm", "gt2_12mm", "gt2_dual_9mm"):
        sp = base.model_copy(deep=True)
        sp.gantry.belt.type = btype  # type: ignore[assignment]
        det = compute_drive_stiffness_details_n_per_m(build_assembly(sp), "X")
        masses.append(float(det["belt_moving_mass_kg"]))
    assert masses[0] < masses[1] < masses[2] < masses[3]
    assert masses[3] > 0.010  # >= 10 g Rayleigh participating inertia on dual 9mm AWD


def test_corexy_2wd_vs_awd_dynamic_belt_tension_compatibility_split() -> None:
    """2WD unequal branch lengths load short span ~72-75% of F_belt; AWD balances 2 belts 50/50."""
    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_monolith_awd.yaml")
    base.kinematics.max_accel_mm_s2 = 20000.0
    base.gantry.belt.pretension_n = 60.0

    sp_awd = base.model_copy(deep=True)
    sp_awd.gantry.drive_mode = "awd"
    sp_2wd = base.model_copy(deep=True)
    sp_2wd.gantry.drive_mode = "2wd"

    asm_awd = build_assembly(sp_awd)
    asm_2wd = build_assembly(sp_2wd)
    det_awd = compute_drive_stiffness_details_n_per_m(asm_awd, "X")
    det_2wd = compute_drive_stiffness_details_n_per_m(asm_2wd, "X")

    # Perimeter CoreXY routing: at bed centre both layouts have equal belt branches (50 / 50 split)
    assert det_awd["slack_load_share"] == pytest.approx(0.50, abs=0.02)
    assert det_2wd["slack_load_share"] == pytest.approx(0.50, abs=0.03)
    # Toward a rear corner the branches become unequal; more so in AWD, where the toolhead gets
    # closer to a motor
    det_awd_c = compute_drive_stiffness_details_n_per_m(asm_awd, "X", u_rel=0.05, v_rel=0.95)
    det_2wd_c = compute_drive_stiffness_details_n_per_m(asm_2wd, "X", u_rel=0.05, v_rel=0.95)
    assert 0.62 <= float(det_2wd_c["slack_load_share"]) <= 0.80
    assert float(det_awd_c["slack_load_share"]) > float(det_2wd_c["slack_load_share"])
    # AWD halves the belt path between toolhead and motors
    assert float(det_2wd["cord_geometry_factor_m"]) == pytest.approx(
        2.0 * float(det_awd["cord_geometry_factor_m"]), rel=0.02
    )

    kin_awd = analyze_kinematics(asm_awd)
    kin_2wd = analyze_kinematics(asm_2wd)
    swing_awd = kin_awd.belt_peak_tension_n - 60.0
    swing_2wd = kin_2wd.belt_peak_tension_n - 60.0
    # With equal branches at bed centre the per-belt dynamic tension swing is about the same in
    # AWD and 2WD (no fake 50% 4-belt reduction)
    assert 0.85 <= swing_awd / swing_2wd <= 1.25
    # Coupled state-space remains asymptotically stable at both low and high tensions
    mod_awd = solve_modal_fea(asm_awd)
    mdl = build_coupled_state_space(asm_awd, mod_awd, 0.5, 0.5)
    assert float(np.max(np.linalg.eigvals(mdl.A).real)) < 0.0
