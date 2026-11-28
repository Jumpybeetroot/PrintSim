"""Tests for per-mode Fox-Kapoor root-cause analysis and unified limiting-resonance summary."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import eigh

from printsim.fea.beam3d import assemble_global_matrices
from printsim.fea.modal import (
    _classify_lumped_mass_group,
    _classify_modal_group,
    _compute_mode_root_causes,
    _precompute_modal_element_blocks,
    solve_modal_fea,
)
from printsim.geometry import AssemblyModel, StructuralMember, StructuralNode, build_assembly
from printsim.schema import PrinterSpec, get_material, get_profile

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
ALL_PRESETS = [
    "voron_trident_300",
    "voron_monolith_awd",
    "cartesian_bedslinger",
    "corexy_voron_style",
    "delta_racer",
]


def _load(name: str) -> PrinterSpec:
    return PrinterSpec.from_yaml(EXAMPLES_DIR / f"{name}.yaml")


def _scale_group_stiffness(asm: AssemblyModel, group_id: str, scale: float) -> None:
    """Scale stiffness matrices (Ke) of all beams, shells, and springs belonging to group_id."""
    for mem in asm.members:
        gid, _ = _classify_modal_group(mem.id, mem.role)
        if gid == group_id:
            updates: dict[str, float] = {
                "youngs_modulus_pa": mem.material.youngs_modulus_pa * scale,
            }
            if mem.material.shear_modulus_pa is not None:
                updates["shear_modulus_pa"] = mem.material.shear_modulus_pa * scale
            mem.material = mem.material.model_copy(update=updates)

    for sh in asm.shell_elements:
        bay_prefix = sh.bay.split("_")[0]
        gid, _ = _classify_modal_group(f"bay_{bay_prefix}", "panel")
        if gid == group_id:
            sh.material = sh.material.model_copy(
                update={"youngs_modulus_pa": sh.material.youngs_modulus_pa * scale}
            )
            if sh.d_bend_nm is not None:
                sh.d_bend_nm *= scale
            if sh.g_transverse_h_n_m is not None:
                sh.g_transverse_h_n_m *= scale
            if sh.e_mem_h_n_m is not None:
                sh.e_mem_h_n_m *= scale

    for sp in asm.fastener_springs:
        bay_raw = sp.bay
        bay_prefix = bay_raw.split("_")[0]
        if bay_raw not in ("bed_mount", "toolhead", "feet", "z_joint") and bay_prefix in (
            "left",
            "right",
            "rear",
            "top",
        ):
            gid, _ = _classify_modal_group(f"bay_{bay_prefix}", "panel")
        else:
            sp_role = (
                "bed"
                if bay_raw == "bed_mount"
                else (
                    "toolhead"
                    if bay_raw == "toolhead"
                    else ("gantry" if bay_raw == "z_joint" else "frame")
                )
            )
            gid, _ = _classify_modal_group(sp.id, sp_role, bay_raw)
        if gid == group_id:
            sp.k_shear_n_m *= scale
            sp.k_normal_n_m *= scale
            sp.k_rot_nm_rad *= scale
            if sp.k_diag_6dof is not None:
                sp.k_diag_6dof = tuple(v * scale for v in sp.k_diag_6dof)

    for se in asm.superelements:
        se_gid = getattr(se, "part_group_id", "") or _classify_modal_group(se.id, getattr(se, "role", ""))[0]
        if se_gid == group_id:
            se.K_cb = se.K_cb * scale


def _scale_group_mass(asm: AssemblyModel, group_id: str, scale: float) -> None:
    """Scale mass matrices (Me + lumped nodal masses) belonging to group_id."""
    for mem in asm.members:
        gid, _ = _classify_modal_group(mem.id, mem.role)
        if gid == group_id:
            mem.material = mem.material.model_copy(
                update={"density_kg_m3": mem.material.density_kg_m3 * scale}
            )

    for sh in asm.shell_elements:
        bay_prefix = sh.bay.split("_")[0]
        gid, _ = _classify_modal_group(f"bay_{bay_prefix}", "panel")
        if gid == group_id:
            sh.areal_mass_kg_m2 *= scale

    for node in asm.nodes:
        if node.lumped_mass_kg <= 0.0 or node.fixed:
            continue
        gid, _, _ = _classify_lumped_mass_group(asm, node.id, node.role)
        if gid == group_id:
            node.lumped_mass_kg *= scale


@pytest.mark.parametrize("preset_name", ALL_PRESETS)
def test_per_mode_top_causes_and_energy_conservation_across_presets(preset_name: str) -> None:
    """Per-mode top_causes are valid, serialized, and full strain/kinetic energy is conserved."""
    spec = _load(preset_name)
    asm = build_assembly(spec)
    res = solve_modal_fea(asm)
    res_dict = res.to_dict()
    json.dumps(res_dict)  # must be pure JSON serializable

    assert len(res.modes) >= 1
    assert len(res_dict["modes"]) == len(res.modes)

    for m, md in zip(res.modes, res_dict["modes"]):
        assert len(m.top_causes) >= 1
        assert len(m.top_causes) <= 5
        assert md["top_causes"] == m.top_causes

        stiff_causes = [c for c in m.top_causes if c["kind"] == "stiffness"]
        mass_causes = [c for c in m.top_causes if c["kind"] == "mass"]
        assert len(stiff_causes) >= 1
        assert len(stiff_causes) <= 3
        assert len(mass_causes) <= 2

        for c in m.top_causes:
            assert set(c.keys()) == {
                "id",
                "label",
                "subsystem",
                "kind",
                "share_pct",
                "est_hz_gain",
                "detail",
            }
            assert c["subsystem"] in {"toolhead", "gantry", "frame", "bed"}
            assert c["kind"] in {"stiffness", "mass"}
            assert 0.0 <= c["share_pct"] <= 100.0
            assert c["est_hz_gain"] >= 0.0
            assert isinstance(c["detail"], str) and len(c["detail"]) > 10

        assert sum(c["share_pct"] for c in stiff_causes) <= 100.5
        assert sum(c["share_pct"] for c in mass_causes) <= 100.5

        gains = [c["est_hz_gain"] for c in m.top_causes]
        assert gains == sorted(gains, reverse=True)

        assert 0.0 <= m.x_axis_share_pct <= 100.0
        assert 0.0 <= m.y_axis_share_pct <= 100.0
        assert 0.0 <= m.limiting_axis_share_pct <= 100.0

    assert sum(m.x_axis_share_pct for m in res.modes) <= 100.5
    assert sum(m.y_axis_share_pct for m in res.modes) <= 100.5

    # Exact Rayleigh conservation check on Mode 1 eigenvectors: sum(U_group) == phi^T K phi == lam, sum(T_group) == 1.0
    K, M, _, node_idx, free_dofs = assemble_global_matrices(asm)
    K_ff = 0.5 * (K[np.ix_(free_dofs, free_dofs)] + K[np.ix_(free_dofs, free_dofs)].T)
    M_ff = 0.5 * (M[np.ix_(free_dofs, free_dofs)] + M[np.ix_(free_dofs, free_dofs)].T)
    m_diag = np.diag(M_ff)
    m_floor = max(float(np.max(m_diag)) * 1e-12, 1e-12)
    M_ff = M_ff + np.diag(np.where(m_diag < m_floor, m_floor, 0.0))

    n_free = len(free_dofs)
    mu, eigvecs = eigh(M_ff, K_ff, subset_by_index=[n_free - 1, n_free - 1])
    lam0 = 1.0 / max(float(mu[0]), 1e-30)
    phi0 = eigvecs[:, 0]
    phi0 = phi0 / math.sqrt(float(phi0 @ M_ff @ phi0))
    full_vec0 = np.zeros(K.shape[0], dtype=float)
    full_vec0[free_dofs] = phi0

    elem_blocks, lumped_blocks = _precompute_modal_element_blocks(asm, node_idx)
    _, u_groups, u_total, t_groups, t_total = _compute_mode_root_causes(
        full_vec0,
        lam0,
        math.sqrt(lam0) / (2.0 * math.pi),
        elem_blocks,
        lumped_blocks,
    )
    assert u_total == pytest.approx(float(phi0 @ K_ff @ phi0), rel=1e-10)
    assert u_total == pytest.approx(lam0, rel=1e-8)
    assert sum(float(g["energy"]) for g in u_groups.values()) == pytest.approx(u_total, rel=1e-9)
    assert t_total == pytest.approx(1.0, rel=1e-6)
    assert sum(float(g["energy"]) for g in t_groups.values()) == pytest.approx(t_total, rel=1e-9)


@pytest.mark.parametrize(
    "preset_name",
    ["voron_trident_300", "corexy_voron_style", "cartesian_bedslinger", "voron_monolith_awd"],
)
def test_analytical_fox_kapoor_mode1_stiffening_and_mass_agreement(preset_name: str) -> None:
    """Stiffening Mode 1's #1 stiffness cause by +10% (and lightening mass by -10%) matches Fox-Kapoor."""
    spec = _load(preset_name)
    asm_base = build_assembly(spec)
    res_base = solve_modal_fea(asm_base)
    m1 = res_base.modes[0]

    # Pick #1 stiffness cause of Mode 1 (highest est_hz_gain)
    stiff_top = max(
        (c for c in m1.top_causes if c["kind"] == "stiffness"),
        key=lambda c: c["est_hz_gain"],
    )
    asm_stiff = build_assembly(spec)
    # Use a small +2% perturbation to stay linear and avoid Mode 1/Mode 2 crossing on closely-spaced modes
    _scale_group_stiffness(asm_stiff, stiff_top["id"], 1.02)
    res_stiff = solve_modal_fea(asm_stiff)

    # est_hz_gain is exact linear +Hz for +50% stiffness => +2% stiffness is 0.04 * est_hz_gain
    df_actual = res_stiff.modes[0].frequency_hz - m1.frequency_hz
    df_pred = 0.04 * stiff_top["est_hz_gain"]
    assert df_actual > 0.0
    assert df_actual == pytest.approx(df_pred, rel=0.15, abs=0.03)

    # If Mode 1 also has a top mass cause, verify -2% mass agrees with 0.08 * est_hz_gain (-25% mass)
    # (small perturbation stays linear and avoids Mode 1/Mode 2 eigenvalue crossing on closely-spaced modes)
    mass_causes = [c for c in m1.top_causes if c["kind"] == "mass"]
    if mass_causes:
        mass_top = max(mass_causes, key=lambda c: c["est_hz_gain"])
        asm_light = build_assembly(spec)
        _scale_group_mass(asm_light, mass_top["id"], 0.98)
        res_light = solve_modal_fea(asm_light)

        df_mass_actual = res_light.modes[0].frequency_hz - m1.frequency_hz
        df_mass_pred = 0.08 * mass_top["est_hz_gain"]
        assert df_mass_actual > 0.0
        assert df_mass_actual == pytest.approx(df_mass_pred, rel=0.10, abs=0.05)


@pytest.mark.parametrize("preset_name", ["voron_monolith_awd", "cartesian_bedslinger"])
def test_belt_stretch_monotonicity_gt2_6mm_to_12mm(preset_name: str) -> None:
    """Upgrading belt gt2_6mm -> gt2_12mm reduces belt_cord_stretch share and raises effective_freq_hz."""
    spec_6 = _load(preset_name)
    spec_6.gantry.belt.type = "gt2_6mm"
    spec_12 = spec_6.model_copy(deep=True)
    spec_12.gantry.belt.type = "gt2_12mm"

    res_6 = solve_modal_fea(build_assembly(spec_6))
    res_12 = solve_modal_fea(build_assembly(spec_12))

    lms_6 = res_6.limiting_modal_summary
    lms_12 = res_12.limiting_modal_summary
    assert lms_6 is not None and lms_12 is not None

    assert lms_12["effective_freq_hz"] > lms_6["effective_freq_hz"]
    assert res_12.recommended_max_accel_mm_s2 > res_6.recommended_max_accel_mm_s2
    assert (
        lms_12["belt_sub_breakdown"]["k_belt_cord_n_per_mm"]
        > lms_6["belt_sub_breakdown"]["k_belt_cord_n_per_mm"]
    )

    cord_6 = next(
        (c["share_pct"] for c in lms_6["top_overall_causes"] if c["id"] == "belt_cord_stretch"),
        0.0,
    )
    cord_12 = next(
        (c["share_pct"] for c in lms_12["top_overall_causes"] if c["id"] == "belt_cord_stretch"),
        0.0,
    )
    assert cord_6 > 0.0
    assert cord_12 < cord_6


@pytest.mark.parametrize("preset_name", ALL_PRESETS)
def test_primary_limiting_mode_uniqueness_and_consistency(preset_name: str) -> None:
    """is_primary_limiting_mode is True for exactly 1 mode and matches limiting_modal_summary."""
    spec = _load(preset_name)
    res = solve_modal_fea(build_assembly(spec))
    lms = res.limiting_modal_summary
    assert lms is not None

    primary_modes = [m for m in res.modes if m.is_primary_limiting_mode]
    assert len(primary_modes) == 1
    pm = primary_modes[0]

    assert pm.mode_number == lms["primary_structural_mode_number"]
    assert pm.is_limiting_axis_mode is True
    assert pm.limiting_axis_share_pct == max(m.limiting_axis_share_pct for m in res.modes)

    if preset_name == "cartesian_bedslinger":
        assert lms["limiting_axis"] == "Y"
        assert pm.excited_axis == "Y"
        assert pm.mode_number in (4, 5)  # B-11 bed plate rotary inertia shifts Y limiting mode to #4
    elif preset_name == "voron_trident_300":
        assert lms["limiting_axis"] == "Y"
        assert pm.excited_axis == "Y"
        assert pm.mode_number in (2, 4)  # Frame Y sway (#2) or ABS StealthBurner pitch whip (#4)


@pytest.mark.parametrize("preset_name", ALL_PRESETS)
def test_limiting_modal_summary_shares_and_contracts(preset_name: str) -> None:
    """belt_sub_breakdown sums to 100.0%, top_overall_causes <= 100.5%, and all fields valid."""
    spec = _load(preset_name)
    res = solve_modal_fea(build_assembly(spec))
    lms = res.limiting_modal_summary
    assert lms is not None

    bsb = lms["belt_sub_breakdown"]
    belt_share_sum = (
        bsb["belt_cord_share_pct"] + bsb["tooth_mesh_share_pct"] + bsb["idler_shaft_share_pct"]
    )
    assert belt_share_sum == pytest.approx(100.0, abs=0.15)
    assert 0.0 < bsb["k_belt_total_n_per_mm"] <= bsb["k_belt_cord_n_per_mm"]

    causes = lms["top_overall_causes"]
    assert 1 <= len(causes) <= 6
    assert sum(c["share_pct"] for c in causes) <= 100.5
    assert all(c["share_pct"] >= 1.5 for c in causes)
    shares = [c["share_pct"] for c in causes]
    assert shares == sorted(shares, reverse=True)
    for c in causes:
        assert set(c.keys()) == {
            "id",
            "label",
            "category",
            "subsystem",
            "share_pct",
            "est_hz_gain",
            "est_accel_gain_mm_s2",
            "detail",
        }
        assert c["category"] in {"belt", "motor", "bracket", "structural"}
        assert c["subsystem"] in {"drive", "toolhead", "gantry", "frame", "bed"}
        assert c["est_hz_gain"] >= 0.0
        assert c["est_accel_gain_mm_s2"] >= 0.0

    mmi = lms["moving_mass_info"]
    assert mmi["axis"] == lms["limiting_axis"]
    assert mmi["mass_g"] > 0.0
    assert mmi["est_hz_gain_25pct_lighter"] >= 0.0
    assert isinstance(mmi["label"], str) and len(mmi["label"]) > 3

    assert lms["limiting_kind"] in {"structural", "belt", "motor", "bracket"}
    assert "Belt stretch IS included" in lms["belt_included_note"]
    assert len(lms["headline"]) > 20


def test_custom_truss_and_empty_assembly_edge_cases() -> None:
    """Minimal custom truss and zero-DOF assemblies handle root-cause and primary-mode fallbacks."""
    # Zero free DOFs
    n_fixed = StructuralNode("n0", np.array([0.0, 0.0, 0.0]), fixed=True, role="foot")
    asm_empty = AssemblyModel(spec=PrinterSpec(), nodes=[n_fixed], members=[], toolhead_node_id="n0")
    res_empty = solve_modal_fea(asm_empty)
    assert res_empty.fundamental_freq_hz == 0.0
    assert res_empty.modes == []
    assert res_empty.limiting_modal_summary is None

    # Minimal cantilever beam along X
    mat = get_material("aluminum_6063_t5")
    prof = get_profile("extrusion_2020")
    nodes = [
        StructuralNode("n0", np.array([0.0, 0.0, 0.0]), fixed=True, role="foot"),
        StructuralNode("n1", np.array([0.25, 0.0, 0.0]), role="frame"),
        StructuralNode("n2", np.array([0.50, 0.0, 0.0]), lumped_mass_kg=0.35, role="toolhead"),
    ]
    members = [
        StructuralMember("m0", "n0", "n1", prof, mat),
        StructuralMember("m1", "n1", "n2", prof, mat),
    ]
    asm_min = AssemblyModel(spec=PrinterSpec(), nodes=nodes, members=members, toolhead_node_id="n2")
    res_min = solve_modal_fea(asm_min)
    assert len(res_min.modes) >= 1
    assert sum(1 for m in res_min.modes if m.is_primary_limiting_mode) == 1
    assert res_min.limiting_modal_summary is not None
    json.dumps(res_min.to_dict())
