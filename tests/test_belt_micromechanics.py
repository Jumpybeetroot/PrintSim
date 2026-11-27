"""Unit tests for first-principles belt micromechanics & 3D pitch-cell Tet10 FEA calibration."""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from printsim.fea.belt_micromechanics import (
    CORD_MICROMECHANICS_CATALOG,
    belt_cord_tangent_ea_n,
    cord_crimp_factor_micromechanics,
    cord_crimp_secant_ea_n,
    cord_crimp_tangent_ea_n,
    get_cord_micromechanics_spec,
    load_pitch_cell_fea_calibration,
    pitch_cell_bending_energy_fractions,
    pitch_cell_flexural_rigidity_nm2,
    pitch_cell_gerbert_mesh_stiffness_n_m,
    pitch_cell_tooth_mesh_k,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FEA_JSON_PATH = REPO_ROOT / "data" / "touchstones" / "belt_pitch_cell_fea.json"


# ---------------------------------------------------------------------------
# 1. Costello Helical-Cord Micromechanics Analytical Limits
# ---------------------------------------------------------------------------


def test_costello_straight_cord_limit_and_asymptotic_ceiling() -> None:
    """Verify straight-cord limit (alpha -> 0) and high-tension asymptote N*E_f*A_f*cos^3(alpha)."""
    n_cords = 7.0
    e_f = 72.0e9
    a_f = 5.294e-8

    # 1. Straight cord (alpha = 0): exact N * E_f * A_f at EVERY tension (including T = 0)
    ea_straight_exact = n_cords * e_f * a_f
    for tension in (0.0, 1.0, 25.0, 500.0):
        ea_tan = cord_crimp_tangent_ea_n(
            tension,
            n_cords=n_cords,
            filament_modulus_pa=e_f,
            filament_area_m2=a_f,
            helix_angle_rad=0.0,
        )
        ea_sec = cord_crimp_secant_ea_n(
            tension,
            n_cords=n_cords,
            filament_modulus_pa=e_f,
            filament_area_m2=a_f,
            helix_angle_rad=0.0,
        )
        assert ea_tan == pytest.approx(ea_straight_exact, rel=1e-12)
        assert ea_sec == pytest.approx(ea_straight_exact, rel=1e-12)

    # 2. Helical cord (alpha = 16 deg): strictly bounded above by N * E_f * A_f * cos^3(alpha)
    #    and approaches it asymptotically as T -> inf
    alpha = math.radians(16.0)
    ea_ceiling = n_cords * e_f * a_f * (math.cos(alpha) ** 3)
    ea_low = cord_crimp_tangent_ea_n(
        0.5,
        n_cords=n_cords,
        filament_modulus_pa=e_f,
        filament_area_m2=a_f,
        helix_angle_rad=alpha,
    )
    ea_high = cord_crimp_tangent_ea_n(
        2000.0,
        n_cords=n_cords,
        filament_modulus_pa=e_f,
        filament_area_m2=a_f,
        helix_angle_rad=alpha,
    )
    assert 0.0 < ea_low < ea_ceiling
    assert ea_high <= ea_ceiling
    assert ea_high == pytest.approx(ea_ceiling, rel=1e-4)

    # 3. Secant modulus matches numerical integral of 1 / EA_tan(T)
    t_eval = 20.0
    n_steps = 2000
    dt = t_eval / n_steps
    eps_num = sum(
        dt
        / cord_crimp_tangent_ea_n(
            (i + 0.5) * dt,
            n_cords=n_cords,
            filament_modulus_pa=e_f,
            filament_area_m2=a_f,
            helix_angle_rad=alpha,
        )
        for i in range(n_steps)
    )
    ea_sec_num = t_eval / eps_num
    ea_sec_ana = cord_crimp_secant_ea_n(
        t_eval,
        n_cords=n_cords,
        filament_modulus_pa=e_f,
        filament_area_m2=a_f,
        helix_angle_rad=alpha,
    )
    assert ea_sec_ana == pytest.approx(ea_sec_num, rel=1e-6)


def test_costello_monotonic_stiffening_and_gates_2gt_physical_range() -> None:
    """Verify monotonic stiffening vs tension, matrix confinement sensitivity, and Gates 2GT bounds."""
    tensions = [0.0, 2.0, 5.0, 10.0, 16.9, 25.0, 45.0, 80.0, 200.0]
    for (profile, core) in CORD_MICROMECHANICS_CATALOG:
        width = 6.0 if profile == "2gt" else (9.0 if profile == "3gt" else 10.0)
        vals = [belt_cord_tangent_ea_n(profile, width, t, core=core) for t in tensions]
        assert all(v > 0.0 for v in vals)
        assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1)), (
            f"Non-monotonic tangent EA for {(profile, core)}: {vals}"
        )

    # Gates 2GT 6 mm E-glass 135 tex physical touchstones:
    # - Glass elastic ceiling ~21-24 kN
    # - 0.1% strain secant EA (at ~16.9 N) near Gates Table 6 static EA (16.9 kN)
    cspec = get_cord_micromechanics_spec("2gt", "fiberglass")
    n_c_6mm = cspec.n_cords_for_width(6.0)
    ea_ceil_6mm = n_c_6mm * cspec.filament_modulus_pa * cspec.filament_area_m2 * (math.cos(cspec.helix_angle_rad) ** 3)
    assert 21_000.0 <= ea_ceil_6mm <= 24_000.0

    ea_sec_16_9 = cord_crimp_secant_ea_n(
        16.9,
        n_cords=n_c_6mm,
        filament_modulus_pa=cspec.filament_modulus_pa,
        filament_area_m2=cspec.filament_area_m2,
        helix_angle_rad=cspec.helix_angle_rad,
        helix_radius_m=cspec.helix_radius_m,
        matrix_modulus_pa=cspec.e_matrix_25c_pa,
        core_transverse_stiffness_pa=cspec.core_transverse_stiffness_pa,
        matrix_confinement_factor=cspec.matrix_confinement_factor,
        radial_compaction_clearance_m=cspec.radial_compaction_clearance_m,
    )
    assert 15_000.0 <= ea_sec_16_9 <= 19_000.0

    # Inside the low-tension toe (T < T_comp), softer matrix lowers radial confinement k_r0
    # and therefore lowers the initial crimp factor EA_tan(T) / EA_inf
    f_epdm = cord_crimp_factor_micromechanics(2.0, profile_family="2gt", width_mm=6.0, elastomer_modulus_scale=1.0)
    f_soft = cord_crimp_factor_micromechanics(2.0, profile_family="2gt", width_mm=6.0, elastomer_modulus_scale=0.5)
    assert 0.0 < f_soft < f_epdm < 1.0

    # Steel cord has tighter lay angle -> higher low-tension crimp factor than fiberglass at same line tension
    f_steel = cord_crimp_factor_micromechanics(15.0, profile_family="at3", width_mm=10.0, core="steel")
    f_glass = cord_crimp_factor_micromechanics(15.0, profile_family="3gt", width_mm=10.0, core="fiberglass")
    assert f_steel > f_glass


# ---------------------------------------------------------------------------
# 2. 3D Pitch-Cell Tet10 FEA Calibration JSON & Surrogate Properties
# ---------------------------------------------------------------------------


def test_fea_calibration_json_exists_converged_and_matches_embedded() -> None:
    """Verify Tet10 FEA touchstone JSON exists, passes bar/mesh convergence, and matches embedded constants."""
    assert FEA_JSON_PATH.is_file(), f"Missing Tet10 FEA calibration JSON at {FEA_JSON_PATH}"
    raw = json.loads(FEA_JSON_PATH.read_text(encoding="utf-8"))

    # Analytical plane-strain bar benchmark < 0.5% error
    assert raw["analytical_bar_benchmark"]["rel_error_pct"] < 0.5

    loaded = load_pitch_cell_fea_calibration(FEA_JSON_PATH)
    embedded = load_pitch_cell_fea_calibration(None)

    for pid in ("2gt", "3gt", "at3"):
        assert pid in raw["profiles"]
        conv = raw["profiles"][pid]["mesh_convergence"]
        # Fine vs coarse Tet10 convergence checks
        assert conv["fine_n_tets"] > conv["coarse_n_tets"]
        assert conv["k_t_full_diff_pct"] < 6.0
        assert conv["ei_diff_pct"] < 5.0

        # Embedded constants must be synchronized with JSON
        c_json = loaded[pid]
        c_emb = embedded[pid]
        assert c_emb.k_t_full_n_per_m_mm == pytest.approx(c_json.k_t_full_n_per_m_mm, rel=1e-5)
        assert c_emb.seating_floor_s0 == pytest.approx(c_json.seating_floor_s0, rel=1e-5)
        assert c_emb.seating_scale_q_n_per_mm == pytest.approx(c_json.seating_scale_q_n_per_mm, rel=1e-5)
        assert c_emb.ei_25c_nm2_per_m == pytest.approx(c_json.ei_25c_nm2_per_m, rel=1e-5)
        assert c_emb.ei_matrix_retention_slope == pytest.approx(c_json.ei_matrix_retention_slope, rel=1e-5)


def test_pitch_cell_tooth_mesh_and_bending_monotonicity_and_positivity() -> None:
    """Verify positive-definite stiffness, monotonic tension seating, and thermal softening."""
    q_grid = [0.25, 1.0, 2.0, 4.0, 8.0, 16.0]
    ret_grid = [1.0, 0.8, 0.6, 0.4]

    for pf in ("2gt", "3gt", "at3", "htd3m", "at5"):
        # Monotonic stiffening of k_t with line tension q0
        kt_vals = [pitch_cell_tooth_mesh_k(pf, q) for q in q_grid]
        assert all(k > 0.0 for k in kt_vals)
        assert all(kt_vals[i] < kt_vals[i + 1] for i in range(len(kt_vals) - 1))

        # Monotonic softening of k_t and EI_b as matrix modulus retention drops
        kt_ret = [pitch_cell_tooth_mesh_k(pf, 4.0, matrix_modulus_retention=r) for r in ret_grid]
        ei_ret = [pitch_cell_flexural_rigidity_nm2(pf, 10.0, matrix_modulus_retention=r) for r in ret_grid]
        assert all(kt_ret[i] > kt_ret[i + 1] > 0.0 for i in range(len(kt_ret) - 1))
        assert all(ei_ret[i] > ei_ret[i + 1] > 0.0 for i in range(len(ei_ret) - 1))

        # Engaged Gerbert tooth-mesh chain is strictly positive and scales with tension
        km_low = pitch_cell_gerbert_mesh_stiffness_n_m(pf, 10.0, 10.0, 30_000.0)
        km_high = pitch_cell_gerbert_mesh_stiffness_n_m(pf, 10.0, 80.0, 30_000.0)
        assert 0.0 < km_low < km_high

        # Bending energy fractions sum to 1 and cord fraction rises as elastomer softens
        fc_cold, fe_cold = pitch_cell_bending_energy_fractions(pf, matrix_modulus_retention=1.0)
        fc_hot, fe_hot = pitch_cell_bending_energy_fractions(pf, matrix_modulus_retention=0.4)
        assert fc_cold + fe_cold == pytest.approx(1.0, abs=1e-12)
        assert fc_hot + fe_hot == pytest.approx(1.0, abs=1e-12)
        assert 0.0 < fc_cold < fc_hot < 1.0

    # Physical cross-family ordering: AT3 (92 ShA TPU) > 3GT (EPDM) > 2GT (EPDM) for tooth k_t,
    # and 3GT (thick 2.41 mm backing) > AT3 (1.90 mm TPU/steel) > 2GT (1.38 mm EPDM/glass) for EI_b
    assert pitch_cell_tooth_mesh_k("at3", 4.0) > pitch_cell_tooth_mesh_k("3gt", 4.0) > pitch_cell_tooth_mesh_k("2gt", 4.0)
    assert pitch_cell_flexural_rigidity_nm2("3gt", 10.0) > 5.0 * pitch_cell_flexural_rigidity_nm2("2gt", 10.0)
    assert pitch_cell_flexural_rigidity_nm2("at3", 10.0) > 2.5 * pitch_cell_flexural_rigidity_nm2("2gt", 10.0)
    assert pitch_cell_flexural_rigidity_nm2("3gt", 10.0) > pitch_cell_flexural_rigidity_nm2("at3", 10.0)


# ---------------------------------------------------------------------------
# 3. Pyodide WASM Import-Safety Check
# ---------------------------------------------------------------------------


def test_belt_micromechanics_pyodide_import_safety() -> None:
    """Verify belt_micromechanics never imports build123d, gmsh, skfem, or heavy scipy subpackages."""
    probe = (
        "import sys\n"
        "blocked = (\n"
        "    'build123d', 'gmsh', 'skfem', 'trimesh', 'pymoo', 'fastapi',\n"
        "    'scipy.optimize', 'scipy.signal', 'scipy.stats', 'scipy.spatial', 'scipy.interpolate',\n"
        "    '_multiprocessing', 'multiprocessing', 'sqlite3',\n"
        ")\n"
        "for mod in blocked:\n"
        "    sys.modules[mod] = None\n"
        "from printsim.fea.belt_micromechanics import (\n"
        "    cord_crimp_tangent_ea_n,\n"
        "    pitch_cell_tooth_mesh_k,\n"
        "    pitch_cell_flexural_rigidity_nm2,\n"
        "    pitch_cell_gerbert_mesh_stiffness_n_m,\n"
        ")\n"
        "ea = cord_crimp_tangent_ea_n(25.0, n_cords=7, filament_modulus_pa=72e9, filament_area_m2=5.294e-8, helix_angle_rad=0.28)\n"
        "kt = pitch_cell_tooth_mesh_k('2gt', 4.0)\n"
        "ei = pitch_cell_flexural_rigidity_nm2('2gt', 6.0)\n"
        "km = pitch_cell_gerbert_mesh_stiffness_n_m('2gt', 6.0, 25.0, ea)\n"
        "assert ea > 10000.0 and kt > 10000.0 and ei > 1e-6 and km > 1e5\n"
    )
    proc = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"Pyodide import-safety probe failed:\n{proc.stderr}"
