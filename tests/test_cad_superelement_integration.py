"""Integration, mathematical, and regression tests for CAD Craig-Bampton superelement injection."""

from __future__ import annotations

import math
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.linalg import eigh

from printsim.api.server import app
from printsim.engine import evaluate_printer
from printsim.fea import (
    assemble_global_matrices,
    compute_structural_sensitivities,
    solve_modal_fea,
    solve_static_fea,
)
from printsim.fea.beam3d import local_beam_mass, local_beam_stiffness, transformation_12x12, beam_rotation_matrix
from printsim.fea.superelement import condense_craig_bampton, condense_parametric_bracket
from printsim.geometry import AssemblyModel, StructuralNode, build_assembly
from printsim.geometry.builder import (
    AttachedSuperelement,
    _series_combine_superelement_kbb,
)
from printsim.schema import (
    HotendSpec,
    PrinterSpec,
    build_model_confidence_audit,
    get_material,
    get_profile,
)

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def _make_beam_cb_superelement(
    *,
    p0_cad: np.ndarray,
    p1_cad: np.ndarray,
    n_elems: int = 4,
    n_modes: int = 4,
    e_pa: float = 110.0e9,
    rho: float = 4430.0,
    radius_m: float = 0.005,
    name: str = "synthetic_cb_beam",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """Condense a multi-element 3D Euler-Bernoulli cantilever into a 2-port CB superelement."""
    area = math.pi * (radius_m**2)
    iy = iz = 0.25 * math.pi * (radius_m**4)
    j_t = 0.5 * math.pi * (radius_m**4)
    g_pa = e_pa / (2.0 * 1.33)

    n_nodes = n_elems + 1
    n_dof = 6 * n_nodes
    K_full = np.zeros((n_dof, n_dof), dtype=float)
    M_full = np.zeros((n_dof, n_dof), dtype=float)
    pts = [p0_cad + (p1_cad - p0_cad) * (i / n_elems) for i in range(n_nodes)]

    for i in range(n_elems):
        r_mat, length = beam_rotation_matrix(pts[i], pts[i + 1])
        t_mat = transformation_12x12(r_mat)
        k_loc = local_beam_stiffness(e_pa, g_pa, area, iy, iz, j_t, length)
        m_loc = local_beam_mass(rho, area, iy, iz, length)
        dofs = list(range(6 * i, 6 * i + 12))
        K_full[np.ix_(dofs, dofs)] += t_mat.T @ k_loc @ t_mat
        M_full[np.ix_(dofs, dofs)] += t_mat.T @ m_loc @ t_mat

    b_dofs = list(range(0, 6)) + list(range(6 * (n_nodes - 1), 6 * n_nodes))
    cb = condense_craig_bampton(
        K_full,
        M_full,
        boundary_dofs=b_dofs,
        n_modes=n_modes,
        name=name,
    )
    K_bb = cb.K_cb[:12, :12].copy()
    return cb.K_cb, cb.M_cb, K_bb, cb.fixed_interface_freqs_hz


@dataclass
class _MockLoadedHardwareAsset:
    category: str
    asset_id: str
    manifest: dict[str, Any]
    K_cb: np.ndarray
    M_cb: np.ndarray
    K_bb: np.ndarray
    port_coords_m: np.ndarray
    fixed_interface_freqs_hz: list[float]
    com_m: np.ndarray
    inertia_kg_m2: np.ndarray
    visual_glb_path: Path | None


def _install_mock_cad_registry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Install an in-memory printsim.cad_assets.registry module matching Track A's contract."""
    hotend_glb = tmp_path / "chube_conduction.glb"
    hotend_glb.write_bytes(b"glTF\x02\x00\x00\x00mock_hotend_glb")
    tricorn_glb = tmp_path / "tricorn_long.glb"
    tricorn_glb.write_bytes(b"glTF\x02\x00\x00\x00mock_tricorn_glb")
    toolhead_glb = tmp_path / "stealthburner.glb"
    toolhead_glb.write_bytes(b"glTF\x02\x00\x00\x00mock_toolhead_glb")
    mb_glb = tmp_path / "microwaved_bowden.glb"
    mb_glb.write_bytes(b"glTF\x02\x00\x00\x00mock_mb_glb")
    braced_glb = tmp_path / "microwaved_bowden__tricorn_long.glb"
    braced_glb.write_bytes(b"glTF\x02\x00\x00\x00mock_braced_glb")

    # Build a stiff hotend CAD CB superelement (k_tip ~ 3600 N/mm) & a softer V6 one (~45 N/mm)
    p0_he = np.array([0.0, 0.0, 0.0])
    p1_he = np.array([0.0, 0.0, -0.066])
    K_cb_chube, M_cb_chube, K_bb_chube, freqs_chube = _make_beam_cb_superelement(
        p0_cad=p0_he,
        p1_cad=p1_he,
        n_elems=3,
        n_modes=4,
        e_pa=114.0e9,
        radius_m=0.00985,  # 3*E*I/L^3 ~ 3490 N/mm
    )
    K_cb_v6, M_cb_v6, K_bb_v6, freqs_v6 = _make_beam_cb_superelement(
        p0_cad=p0_he,
        p1_cad=np.array([0.0, 0.0, -0.0623]),
        n_elems=3,
        n_modes=4,
        e_pa=193.0e9,
        radius_m=0.0022,  # ~44 N/mm
    )
    # Unbraced Tricorn Long (~263 N/mm) vs Braced Microwaved Bowden + Tricorn Long (~3400 N/mm)
    K_cb_tri, M_cb_tri, K_bb_tri, freqs_tri = _make_beam_cb_superelement(
        p0_cad=p0_he,
        p1_cad=p1_he,
        n_elems=3,
        n_modes=4,
        e_pa=113.8e9,
        radius_m=0.0041,
    )
    K_cb_braced, M_cb_braced, K_bb_braced, freqs_braced = _make_beam_cb_superelement(
        p0_cad=p0_he,
        p1_cad=p1_he,
        n_elems=3,
        n_modes=4,
        e_pa=113.8e9,
        radius_m=0.00777,
    )
    # Build toolhead CAD CB superelements (StealthBurner ABS, Microwaved Bowden Ti-6Al-4V)
    p0_th = np.array([0.0, 0.0, 0.0])
    p1_th = np.array([0.0, -0.022, -0.028])
    K_cb_th, M_cb_th, K_bb_th, freqs_th = _make_beam_cb_superelement(
        p0_cad=p0_th,
        p1_cad=p1_th,
        n_elems=3,
        n_modes=4,
        e_pa=2.1e9,
        rho=1040.0,
        radius_m=0.014,
    )
    p1_mb = np.array([0.0, -0.00976, 0.0305])
    K_cb_mb, M_cb_mb, K_bb_mb, freqs_mb = _make_beam_cb_superelement(
        p0_cad=p0_th,
        p1_cad=p1_mb,
        n_elems=3,
        n_modes=4,
        e_pa=113.8e9,
        rho=4430.0,
        radius_m=0.011,
    )

    assets: dict[tuple[str, str], dict[str, Any]] = {
        ("hotends", "chube_conduction"): {
            "manifest": {
                "id": "chube_conduction",
                "name": "Luke's Lab Chube Conduction",
                "cad_trust_pct": 93,
                "mass_kg": 0.090,
                "visual_url": "/api/assets/hardware/hotends/chube_conduction/visual.glb",
            },
            "K_cb": K_cb_chube,
            "M_cb": M_cb_chube,
            "K_bb": K_bb_chube,
            "port_coords_m": np.vstack([p0_he, p1_he]),
            "freqs": freqs_chube,
            "glb": hotend_glb,
            "ref_mat": None,
        },
        ("hotends", "e3d_v6_groove"): {
            "manifest": {
                "id": "e3d_v6_groove",
                "name": "E3D V6 Groove-Mount",
                "cad_trust_pct": 91,
                "mass_kg": 0.042,
                "visual_url": "/api/assets/hardware/hotends/e3d_v6_groove/visual.glb",
            },
            "K_cb": K_cb_v6,
            "M_cb": M_cb_v6,
            "K_bb": K_bb_v6,
            "port_coords_m": np.vstack([p0_he, np.array([0.0, 0.0, -0.0623])]),
            "freqs": freqs_v6,
            "glb": hotend_glb,
            "ref_mat": None,
        },
        ("hotends", "tricorn_long"): {
            "manifest": {
                "id": "tricorn_long",
                "name": "Tricorn Long (120W 3x40W Inverted Ti)",
                "cad_trust_pct": 88,
                "mass_kg": 0.0464,
                "element_type": "Tet4",
                "num_elements": 116445,
                "port_stiffness_n_per_mm": {"kx": 262.2, "ky": 263.7, "kz": 52757.7},
                "visual_url": "/api/assets/hardware/hotends/tricorn_long/visual.glb",
            },
            "K_cb": K_cb_tri,
            "M_cb": M_cb_tri,
            "K_bb": K_bb_tri,
            "port_coords_m": np.vstack([p0_he, p1_he]),
            "freqs": freqs_tri,
            "glb": tricorn_glb,
            "ref_mat": None,
        },
        ("toolheads", "stealthburner"): {
            "manifest": {
                "id": "stealthburner",
                "name": "Voron StealthBurner + CW2",
                "cad_trust_pct": 90,
                "mass_kg": 0.365,
                "visual_url": "/api/assets/hardware/toolheads/stealthburner/visual.glb",
            },
            "K_cb": K_cb_th,
            "M_cb": M_cb_th,
            "K_bb": K_bb_th,
            "port_coords_m": np.vstack([p0_th, p1_th]),
            "freqs": freqs_th,
            "glb": toolhead_glb,
            "ref_mat": "abs_printed",
        },
        ("toolheads", "microwaved_bowden"): {
            "manifest": {
                "id": "microwaved_bowden",
                "name": "Microwaved Bowden Toolhead (SLM Ti-6Al-4V)",
                "cad_trust_pct": 85,
                "mass_kg": 0.0719,
                "is_multi_material": False,
                "reference_youngs_modulus_pa": 113.8e9,
                "reference_density_kg_m3": 4430.0,
                "port_stiffness_n_per_mm": {"kx": 10727.5, "ky": 3563.1, "kz": 24677.8},
                "visual_url": "/api/assets/hardware/toolheads/microwaved_bowden/visual.glb",
            },
            "K_cb": K_cb_mb,
            "M_cb": M_cb_mb,
            "K_bb": K_bb_mb,
            "port_coords_m": np.vstack([p0_th, p1_mb]),
            "freqs": freqs_mb,
            "glb": mb_glb,
            "ref_mat": "titanium_grade5",
        },
        ("assemblies", "microwaved_bowden__tricorn_long"): {
            "manifest": {
                "id": "microwaved_bowden__tricorn_long",
                "asset_id": "microwaved_bowden__tricorn_long",
                "category": "assemblies",
                "name": "Microwaved Bowden + Tricorn Long (Ti Braced Assembly)",
                "cad_trust_pct": 89,
                "mass_kg": 0.048,
                "is_multi_material": True,
                "element_type": "Tet4",
                "num_elements": 132000,
                "port_stiffness_n_per_mm": {"kx": 3420.0, "ky": 3380.0, "kz": 68000.0},
                "visual_url": "/api/assets/hardware/assemblies/microwaved_bowden__tricorn_long/visual.glb",
            },
            "K_cb": K_cb_braced,
            "M_cb": M_cb_braced,
            "K_bb": K_bb_braced,
            "port_coords_m": np.vstack([p0_he, p1_he]),
            "freqs": freqs_braced,
            "glb": braced_glb,
            "ref_mat": None,
        },
    }

    mod = types.ModuleType("printsim.cad_assets.registry")

    def list_hardware_assets() -> dict[str, dict[str, dict[str, Any]]]:
        out: dict[str, dict[str, dict[str, Any]]] = {"hotends": {}, "toolheads": {}, "assemblies": {}}
        for (cat, aid), entry in assets.items():
            out.setdefault(cat, {})[aid] = dict(entry["manifest"])
        return out

    def has_hardware_asset(category: str, asset_id: str | None) -> bool:
        return bool(asset_id and (category, asset_id) in assets)

    def get_hardware_asset_manifest(category: str, asset_id: str | None) -> dict[str, Any] | None:
        if not asset_id or (category, asset_id) not in assets:
            return None
        return dict(assets[(category, asset_id)]["manifest"])

    def get_visual_glb_path(category: str, asset_id: str | None) -> Path | None:
        if not asset_id or (category, asset_id) not in assets:
            return None
        return assets[(category, asset_id)]["glb"]

    def load_superelement(
        category: str,
        asset_id: str,
        material_id: str | None = None,
        chamber_temp_c: float = 25.0,
        stiffness_mult: float = 1.0,
    ) -> _MockLoadedHardwareAsset | None:
        entry = assets.get((category, asset_id))
        if entry is None:
            return None
        scale = float(stiffness_mult)
        if material_id is not None and entry["ref_mat"] is not None:
            e_ref = get_material(entry["ref_mat"]).effective_modulus_pa(25.0)
            e_cur = get_material(material_id).effective_modulus_pa(chamber_temp_c)
            scale *= e_cur / max(e_ref, 1.0)
        return _MockLoadedHardwareAsset(
            category=category,
            asset_id=asset_id,
            manifest=dict(entry["manifest"]),
            K_cb=entry["K_cb"] * scale,
            M_cb=entry["M_cb"].copy(),
            K_bb=entry["K_bb"] * scale,
            port_coords_m=entry["port_coords_m"].copy(),
            fixed_interface_freqs_hz=list(entry["freqs"]),
            com_m=0.5 * (entry["port_coords_m"][0] + entry["port_coords_m"][1]),
            inertia_kg_m2=np.eye(3) * 1e-5,
            visual_glb_path=entry["glb"],
        )

    mod.list_hardware_assets = list_hardware_assets  # type: ignore[attr-defined]
    mod.has_hardware_asset = has_hardware_asset  # type: ignore[attr-defined]
    mod.get_hardware_asset_manifest = get_hardware_asset_manifest  # type: ignore[attr-defined]
    mod.get_visual_glb_path = get_visual_glb_path  # type: ignore[attr-defined]
    mod.load_superelement = load_superelement  # type: ignore[attr-defined]

    pkg = types.ModuleType("printsim.cad_assets")
    pkg.registry = mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "printsim.cad_assets", pkg)
    monkeypatch.setitem(sys.modules, "printsim.cad_assets.registry", mod)


# ---------------------------------------------------------------------------
# 1. Exact Rigid-Arm Port Transform & Series Equilibrium Math
# ---------------------------------------------------------------------------


def test_rigid_arm_port_transform_exact_zero_rigid_body_strain_energy() -> None:
    """K_transformed has zero strain energy (<1e-10) under arbitrary 6-DOF global rigid-body motion."""
    p0_cad = np.array([0.0, 0.0, 0.0])
    p1_cad = np.array([0.0, 0.0, -0.066])
    K_cb, M_cb, _, _ = _make_beam_cb_superelement(
        p0_cad=p0_cad, p1_cad=p1_cad, n_elems=4, n_modes=4
    )

    # Place assembly nodes at completely different 3D positions (non-zero delta in X, Y, Z)
    p0_asm = np.array([0.14, 0.18, 0.27])
    p1_asm = np.array([0.152, 0.165, 0.222])
    dp_asm = p1_asm - p0_asm

    nodes = [
        StructuralNode("n0", p0_asm, fixed=False, role="toolhead"),
        StructuralNode("n1", p1_asm, fixed=False, role="nozzle"),
    ]
    se = AttachedSuperelement(
        id="test_se",
        category="hotends",
        asset_id="test_beam",
        port_node_ids=("n0", "n1"),
        K_cb=K_cb,
        M_cb=M_cb,
        port_coords_cad_m=np.vstack([p0_cad, p1_cad]),
        n_internal_modes=4,
    )
    asm = AssemblyModel(
        spec=PrinterSpec(),
        nodes=nodes,
        members=[],
        superelements=[se],
        toolhead_node_id="n0",
        nozzle_node_id="n1",
    )
    K, M, F_grav, _, _ = assemble_global_matrices(asm)
    assert K.shape == (16, 16)
    assert M.shape == (16, 16)
    np.testing.assert_allclose(K, K.T, atol=1e-9)
    np.testing.assert_allclose(M, M.T, atol=1e-9)

    k_scale = float(np.max(np.diag(K)))
    rng = np.random.default_rng(7)
    for _ in range(10):
        t = rng.normal(size=3)
        omega = rng.normal(size=3)
        u_rb = np.zeros(16, dtype=float)
        u_rb[0:3] = t
        u_rb[3:6] = omega
        u_rb[6:9] = t + np.cross(omega, dp_asm)
        u_rb[9:12] = omega
        # u_rb[12:16] == 0 for rigid-body motion in Craig-Bampton coordinates
        energy = float(u_rb @ K @ u_rb)
        assert abs(energy) < 1e-10 * k_scale
        assert float(np.linalg.norm(K @ u_rb)) < 1e-8 * k_scale

    # Free-free eigenvalues: exactly 6 zero rigid-body modes + positive elastic modes
    eigvals, _ = eigh(K, M)
    freqs_hz = np.sqrt(np.maximum(eigvals, 0.0)) / (2.0 * math.pi)
    assert np.all(freqs_hz[:6] < 1e-2)
    assert freqs_hz[6] > 100.0
    # Verify gravity load sums to total mass * g in -Z
    m_tot = float(np.sum(M[[2, 8], :][:, [2, 8]]))  # cross-check rigid-Z mass projection
    r_z = np.zeros(16, dtype=float)
    r_z[[2, 8]] = 1.0
    assert float(r_z @ F_grav) == pytest.approx(-9.80665 * float(r_z @ M @ r_z), rel=1e-9)


def test_series_combine_superelement_kbb_exact_equilibrium_and_series_compliance() -> None:
    """K_12_eff = B^T K_11_eff B preserves 6 zero RB modes & matches exact analytical series compliance."""
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([0.0, 0.0, -0.060])
    _, _, K_bb, _ = _make_beam_cb_superelement(
        p0_cad=p0, p1_cad=p1, n_elems=4, n_modes=0, e_pa=110.0e9, radius_m=0.005
    )
    k_cad_x = 1.0 / float(np.linalg.inv(K_bb[6:12, 6:12])[0, 0])

    k_clamp_t = 2.0e6  # 2000 N/mm
    k_clamp_r = 5000.0  # Nm/rad
    C_clamp = np.diag(
        [
            1.0 / k_clamp_t,
            1.0 / k_clamp_t,
            1.0 / (4.0 * k_clamp_t),
            1.0 / k_clamp_r,
            1.0 / k_clamp_r,
            1.0 / k_clamp_r,
        ]
    )
    dp_asm = np.array([0.005, -0.010, -0.042])
    K_12_eff, K_11_eff, C_eff = _series_combine_superelement_kbb(K_bb, C_clamp, dp_asm)

    assert K_12_eff.shape == (12, 12)
    np.testing.assert_allclose(K_12_eff, K_12_eff.T, atol=1e-9)
    eigs = np.linalg.eigvalsh(K_12_eff)
    assert np.all(np.abs(eigs[:6]) < 1e-8 * eigs[-1])
    assert np.all(eigs[6:] > 0.0)

    # Exact series tip compliance in X: 1/k_cad_x + 1/k_clamp_t
    assert float(C_eff[0, 0]) == pytest.approx(1.0 / k_cad_x + 1.0 / k_clamp_t, rel=1e-9)
    np.testing.assert_allclose(K_12_eff[6:12, 6:12], K_11_eff, rtol=1e-9, atol=1e-9)

    # Verify exact rigid-body zero energy on K_12_eff at dp_asm
    rng = np.random.default_rng(11)
    for _ in range(8):
        t = rng.normal(size=3)
        omega = rng.normal(size=3)
        u_rb = np.concatenate([t, omega, t + np.cross(omega, dp_asm), omega])
        assert abs(float(u_rb @ K_12_eff @ u_rb)) < 1e-10 * eigs[-1]


# ---------------------------------------------------------------------------
# 2. Full Multiphysics Pipeline, Solvers, Adjoint Sensitivities & Audit
# ---------------------------------------------------------------------------


def test_full_pipeline_with_mock_cad_superelements_and_internal_cb_dofs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify builder, beam3d, solver, modal (incl. feet), sensitivity, audit & API with CAD SEs."""
    _install_mock_cad_registry(monkeypatch, tmp_path)

    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    base.frame.enclosed_panels = True
    base.frame.panel_mounting = "bolted_structural"
    base.frame.panel_material = "aluminum_5052_sheet"
    base.frame.panel_thickness_mm = 4.0
    base.toolhead.hotend = HotendSpec(preset_id="chube_conduction")

    s_cad = base.model_copy(deep=True)
    s_cad.use_cad_superelements = True
    s_heur = base.model_copy(deep=True)
    s_heur.use_cad_superelements = False

    asm_cad = build_assembly(s_cad)
    asm_heur = build_assembly(s_heur)

    # Heuristic members omitted, CAD superelements injected, exact mass conserved
    assert len(asm_cad.superelements) == 2
    assert {se.id for se in asm_cad.superelements} == {"toolhead_mount", "nozzle_stem"}
    assert not any(m.id in ("toolhead_mount", "nozzle_stem") for m in asm_cad.members)
    assert len(asm_heur.superelements) == 0
    assert any(m.id == "toolhead_mount" for m in asm_heur.members)
    assert any(m.id == "nozzle_stem" for m in asm_heur.members)

    assert asm_cad.total_mass_kg == pytest.approx(asm_heur.total_mass_kg, abs=1e-7)
    assert asm_cad.x_moving_mass_kg == pytest.approx(asm_heur.x_moving_mass_kg, rel=1e-12)
    assert asm_cad.y_moving_mass_kg == pytest.approx(asm_heur.y_moving_mass_kg, rel=1e-12)
    assert asm_cad.node_map["toolhead_cog"].lumped_mass_kg == pytest.approx(
        asm_heur.node_map["toolhead_cog"].lumped_mass_kg, rel=1e-12
    )

    # Geometry dict export
    gdict_cad = asm_cad.to_geometry_dict()["toolhead_structure"]
    assert gdict_cad["use_cad_superelements"] is True
    assert gdict_cad["cad_toolhead_self_contained"] is True
    assert gdict_cad["cad_hotend_self_contained"] is True
    assert gdict_cad["cad_includes_backplate"] is True
    assert len(gdict_cad["cad_superelements_active"]) == 2
    assert gdict_cad["hotend_visual_glb_url"] == "/api/assets/hardware/hotends/chube_conduction/visual.glb"
    assert gdict_cad["toolhead_visual_glb_url"] == "/api/assets/hardware/toolheads/stealthburner/visual.glb"

    gdict_heur = asm_heur.to_geometry_dict()["toolhead_structure"]
    assert gdict_heur["use_cad_superelements"] is False
    assert gdict_heur["cad_toolhead_self_contained"] is False
    assert gdict_heur["cad_hotend_self_contained"] is False
    assert gdict_heur["cad_includes_backplate"] is False
    assert gdict_heur["cad_superelements_active"] == []

    # Static, modal, sensitivity solves
    st_cad = solve_static_fea(asm_cad)
    mo_cad = solve_modal_fea(asm_cad)
    sen_cad = compute_structural_sensitivities(asm_cad, static_res=st_cad, modal_res=mo_cad)

    assert st_cad.toolhead_flex_breakdown["hotend_and_nozzle_pct"] < 20.0
    assert (
        st_cad.toolhead_flex_breakdown["rail_and_xbeam_pct"]
        + st_cad.toolhead_flex_breakdown["carriage_and_body_pct"]
        + st_cad.toolhead_flex_breakdown["hotend_and_nozzle_pct"]
    ) == pytest.approx(100.0, abs=0.2)
    assert math.isfinite(mo_cad.fundamental_freq_hz) and mo_cad.fundamental_freq_hz > 20.0

    # Sensitivity attribution & 100% compliance sum
    assert "toolhead_mount" in sen_cad.member_sensitivities
    assert "nozzle_stem" in sen_cad.member_sensitivities
    comp_sum = sum(v["compliance_share_pct"] for v in sen_cad.member_sensitivities.values()) + sum(
        v["compliance_share_pct"] for v in sen_cad.panel_bay_sensitivities.values()
    )
    assert comp_sum == pytest.approx(100.0, abs=0.5)
    assert any(t["id"] == "toolhead_mount" for t in sen_cad.top_stiffen_targets)

    # Analytical adjoint & Fox-Kapoor derivatives on toolhead_mount SE match central finite difference!
    eps = 0.005
    dln = math.log(1.0 + eps) - math.log(1.0 - eps)

    def _eval_se_scale(scale: float) -> tuple[float, float, float]:
        asm_p = build_assembly(s_cad)
        for se in asm_p.superelements:
            if se.id == "toolhead_mount":
                se.K_cb = se.K_cb * scale
        mod_p = solve_modal_fea(asm_p)
        K_p, _, _, nidx, free_p = assemble_global_matrices(asm_p)
        from printsim.fea.solver import _build_factorized_solver, build_inertial_load_vectors

        solve_rhs = _build_factorized_solver(K_p, free_p)
        noz_i = nidx[asm_p.nozzle_node_id]
        bed_i = nidx[asm_p.bed_node_id]
        n_dof = K_p.shape[0]
        accel = asm_p.spec.kinematics.max_accel_mm_s2 * 1e-3
        F_x, F_y, _, _ = build_inertial_load_vectors(asm_p, nidx, n_dof, accel)

        ux = solve_rhs(F_x)
        uy = solve_rhs(F_y)
        ux_rel = float(ux[6 * noz_i + 0] - ux[6 * bed_i + 0])
        uy_rel = float(uy[6 * noz_i + 1] - uy[6 * bed_i + 1])
        return mod_p.fundamental_freq_hz, ux_rel, uy_rel

    fp, uxp, uyp = _eval_se_scale(1.0 + eps)
    fm, uxm, uym = _eval_se_scale(1.0 - eps)
    th_sens = sen_cad.member_sensitivities["toolhead_mount"]
    assert th_sens["stiffness_sens_hz"] == pytest.approx((fp - fm) / dln, rel=0.02)
    assert th_sens["du_rel_x_dlnE_m"] == pytest.approx((uxp - uxm) / dln, rel=0.02)
    assert th_sens["du_rel_y_dlnE_m"] == pytest.approx((uyp - uym) / dln, rel=0.02)

    # Series rail-carriage/clamp/body sensitivity preserved without double-counting generic backplate
    s_rail_zb = s_cad.model_copy(deep=True)
    s_rail_zb.gantry.rail_preload = "ZB"
    s_rail_zb.gantry.rail_quality = "precision_thk_nsk"
    s_hm_al = s_cad.model_copy(deep=True)
    s_hm_al.toolhead.hotend_mount_material = "aluminum_6061_t6"
    s_body_al = s_cad.model_copy(deep=True)
    s_body_al.toolhead.body_material = "aluminum_6061_t6"
    asm_zb = build_assembly(s_rail_zb)
    st_zb = solve_static_fea(asm_zb)
    st_hm = solve_static_fea(build_assembly(s_hm_al))
    st_body = solve_static_fea(build_assembly(s_body_al))
    assert (
        asm_zb.toolhead_stage_stiffnesses_n_per_mm["carriage_mount_n_per_mm"]
        > asm_cad.toolhead_stage_stiffnesses_n_per_mm["carriage_mount_n_per_mm"] * 1.35
    )
    assert st_zb.toolhead_stiffness_y_n_per_mm > st_cad.toolhead_stiffness_y_n_per_mm
    assert st_hm.toolhead_flex_breakdown["hotend_stiffness_n_per_mm"] > st_cad.toolhead_flex_breakdown["hotend_stiffness_n_per_mm"] * 1.25
    assert st_body.toolhead_stiffness_y_n_per_mm > st_cad.toolhead_stiffness_y_n_per_mm

    # V6 vs Chube hotend CAD superelements show large stiffness & twist gap
    s_v6 = s_cad.model_copy(deep=True)
    s_v6.toolhead.hotend = HotendSpec(preset_id="e3d_v6_groove")
    st_v6 = solve_static_fea(build_assembly(s_v6))
    assert st_v6.toolhead_flex_breakdown["hotend_and_nozzle_pct"] > 45.0
    assert st_v6.toolhead_flex_breakdown["dominant_flex_source"] == "hotend_nozzle"
    assert st_cad.toolhead_stiffness_y_n_per_mm > st_v6.toolhead_stiffness_y_n_per_mm * 2.5

    # On stiff CNC CoreXY (voron_monolith_awd), V6 hotend CAD superelement exceeds 70% flex share
    mono_v6 = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_monolith_awd.yaml")
    mono_v6.toolhead.hotend = HotendSpec(preset_id="e3d_v6_groove")
    st_mono_v6 = solve_static_fea(build_assembly(mono_v6))
    assert st_mono_v6.toolhead_flex_breakdown["hotend_and_nozzle_pct"] > 70.0
    assert st_mono_v6.toolhead_flex_breakdown["dominant_flex_source"] == "hotend_nozzle"

    # Internal CB DOFs + benchtop feet compliance (guards gdof // 6 in modal.py!)
    s_feet = s_cad.model_copy(deep=True)
    s_feet.frame.feet_type = "rubber_feet"
    asm_int = build_assembly(s_feet)
    p0_he = asm_int.node_map["toolhead_cog"].position_m
    p1_he = asm_int.node_map["nozzle_tip"].position_m
    K_cb_int, M_cb_int, _, freqs_int = _make_beam_cb_superelement(
        p0_cad=p0_he, p1_cad=p1_he, n_elems=3, n_modes=4
    )
    asm_int.superelements = [
        se
        for se in asm_int.superelements
        if se.id != "nozzle_stem"
    ]
    asm_int.superelements.append(
        AttachedSuperelement(
            id="nozzle_stem",
            category="hotends",
            asset_id="chube_conduction",
            port_node_ids=("toolhead_cog", "nozzle_tip"),
            K_cb=K_cb_int,
            M_cb=M_cb_int,
            port_coords_cad_m=np.vstack([p0_he, p1_he]),
            n_internal_modes=4,
            part_group_id="toolhead_mount",
            part_group_label="Toolhead & Nozzle Mount",
            role="toolhead",
            mass_kg=0.05,
            fixed_interface_freqs_hz=freqs_int,
        )
    )
    st_int = solve_static_fea(asm_int)
    mo_int = solve_modal_fea(asm_int)
    sen_int = compute_structural_sensitivities(asm_int, static_res=st_int, modal_res=mo_int)
    assert math.isfinite(st_int.combined_max_deflection_um)
    assert math.isfinite(mo_int.fundamental_freq_hz) and mo_int.fundamental_freq_hz > 5.0
    comp_sum_int = sum(v["compliance_share_pct"] for v in sen_int.member_sensitivities.values()) + sum(
        v["compliance_share_pct"] for v in sen_int.panel_bay_sensitivities.values()
    )
    assert comp_sum_int == pytest.approx(100.0, abs=0.5)

    # Confidence audit integration
    audit_cad = {e["id"]: e for e in build_model_confidence_audit(s_cad)["entries"]}
    audit_heur = {e["id"]: e for e in build_model_confidence_audit(s_heur)["entries"]}
    assert audit_cad["toolhead_preset"]["cad_superelement_active"] is True
    assert audit_cad["toolhead_preset"]["trust_pct"] == 90
    assert audit_cad["toolhead_construction"]["cad_superelement_active"] is True
    assert audit_cad["toolhead_construction"]["trust_pct"] == 90
    assert audit_cad["hotend_clamp"]["cad_superelement_active"] is True
    assert audit_cad["hotend_clamp"]["trust_pct"] == 93
    assert audit_heur["hotend_clamp"]["cad_superelement_active"] is False
    assert audit_heur["hotend_clamp"]["trust_pct"] == 78

    # Options breakdown active trust invariant
    active_he_opts = [o for o in audit_cad["hotend_clamp"]["options_breakdown"] if o["active"]]
    assert len(active_he_opts) == 1
    assert active_he_opts[0]["trust_pct"] == audit_cad["hotend_clamp"]["trust_pct"]

    # FastAPI endpoints
    client = TestClient(app)
    r_list = client.get("/api/assets/hardware")
    assert r_list.status_code == 200
    data_list = r_list.json()
    assert "chube_conduction" in data_list["hotends"]
    assert "stealthburner" in data_list["toolheads"]

    r_glb = client.get("/api/assets/hardware/hotends/chube_conduction/visual.glb")
    assert r_glb.status_code == 200
    assert r_glb.headers["content-type"].startswith("model/gltf-binary")
    assert r_glb.headers["cache-control"] == "public, max-age=3600"
    assert r_glb.content.startswith(b"glTF")

    # 404 & traversal protection
    assert client.get("/api/assets/hardware/evil/chube_conduction/visual.glb").status_code == 404
    assert client.get("/api/assets/hardware/hotends/bad.id/visual.glb").status_code == 404
    assert client.get("/api/assets/hardware/hotends/missing_asset/visual.glb").status_code == 404


def test_body_construction_wall_thickness_and_no_double_counting_with_cad_se(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Self-contained CAD toolheads resist generic wall/construction/backplate distortion & scale cleanly."""
    _install_mock_cad_registry(monkeypatch, tmp_path)

    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    base.toolhead.hotend = HotendSpec(preset_id="chube_conduction")
    base.toolhead.body_construction = "compact_cage"

    asm_base = build_assembly(base)
    st_base = solve_static_fea(asm_base)
    stiff_base = asm_base.toolhead_stage_stiffnesses_n_per_mm
    gdict_base = asm_base.to_geometry_dict()["toolhead_structure"]

    # 1. No fake generic Stage 1 backplate double-counted on self-contained CAD toolhead
    assert gdict_base["cad_toolhead_self_contained"] is True
    assert gdict_base["cad_hotend_self_contained"] is True
    assert gdict_base["cad_includes_backplate"] is True
    assert stiff_base["backplate_only_n_per_mm"] == 0.0
    assert stiff_base["carriage_mount_n_per_mm"] == pytest.approx(
        stiff_base["rail_carriage_n_per_mm"], rel=1e-12
    )

    # 2. Generic parametric sliders (wall thickness, carriage plate thickness/material,
    #    body_construction, internal_stiffness_n_per_mm) must NOT distort self-contained CAD toolhead
    s_mut = base.model_copy(deep=True)
    s_mut.toolhead.body_wall_thickness_mm = 10.0
    s_mut.toolhead.carriage_plate_thickness_mm = 12.0
    s_mut.toolhead.carriage_mount_material = "aluminum_6061_t6"
    s_mut.toolhead.body_construction = "cnc_monoblock"
    s_mut.toolhead.internal_stiffness_n_per_mm = 3000.0
    asm_mut = build_assembly(s_mut)
    st_mut = solve_static_fea(asm_mut)
    assert asm_mut.toolhead_stage_stiffnesses_n_per_mm["body_n_per_mm"] == pytest.approx(
        stiff_base["body_n_per_mm"], rel=1e-12
    )
    assert asm_mut.toolhead_stage_stiffnesses_n_per_mm["carriage_mount_n_per_mm"] == pytest.approx(
        stiff_base["carriage_mount_n_per_mm"], rel=1e-12
    )
    assert st_mut.toolhead_stiffness_y_n_per_mm == pytest.approx(
        st_base.toolhead_stiffness_y_n_per_mm, rel=1e-12
    )

    s_none_bc = base.model_copy(deep=True)
    s_none_bc.toolhead.body_construction = None
    s_none_bc.toolhead.internal_stiffness_n_per_mm = 150.0
    asm_none_bc = build_assembly(s_none_bc)
    assert asm_none_bc.toolhead_stage_stiffnesses_n_per_mm["body_n_per_mm"] == pytest.approx(
        stiff_base["body_n_per_mm"], rel=1e-12
    )

    # 3. Material swaps (abs_printed -> ppa_cf_printed -> aluminum_6061_t6 -> titanium_grade5) scale cleanly
    mat_ks: list[float] = []
    for mat_id in ("abs_printed", "ppa_cf_printed", "aluminum_6061_t6", "titanium_grade5"):
        s_mat = base.model_copy(deep=True)
        s_mat.toolhead.body_material = mat_id
        mat_ks.append(build_assembly(s_mat).toolhead_stage_stiffnesses_n_per_mm["body_n_per_mm"])
    for idx in range(len(mat_ks) - 1):
        assert mat_ks[idx + 1] > mat_ks[idx]

    # 4. Explicit Advanced model_overrides.toolhead_construction & Z-probe Tap compliance scale cleanly
    s_ov = base.model_copy(deep=True)
    s_ov.model_overrides.toolhead_construction.body_stiffness_base_n_per_mm = 1000.0
    s_ov.model_overrides.toolhead_construction.torsion_mult = 2.0
    asm_ov = build_assembly(s_ov)
    assert (
        asm_ov.toolhead_stage_stiffnesses_n_per_mm["body_n_per_mm"]
        > stiff_base["body_n_per_mm"] * 3.0
    )

    s_tap = base.model_copy(deep=True)
    s_tap.toolhead.z_probe_type = "printed_tap"
    asm_tap = build_assembly(s_tap)
    assert (
        asm_tap.toolhead_stage_stiffnesses_n_per_mm["body_n_per_mm"]
        < stiff_base["body_n_per_mm"] * 0.5
    )

    # 5. Hotend clamp thickness continues to scale Stage 3 clamp compliance in series
    s_hm_thin = base.model_copy(deep=True)
    s_hm_thin.toolhead.hotend_mount_thickness_mm = 4.0
    s_hm_thick = base.model_copy(deep=True)
    s_hm_thick.toolhead.hotend_mount_thickness_mm = 16.0
    assert (
        build_assembly(s_hm_thick).toolhead_stage_stiffnesses_n_per_mm["hotend_total_n_per_mm"]
        > build_assembly(s_hm_thin).toolhead_stage_stiffnesses_n_per_mm["hotend_total_n_per_mm"] * 1.15
    )

    # 6. Full end-to-end multiphysics evaluation
    rep = evaluate_printer(base, include_advanced_analysis=False)
    assert rep.static_fea.combined_max_deflection_um > 0.0
    assert rep.modal_fea.fundamental_freq_hz > 10.0
    assert len(rep.assembly.superelements) == 2


def test_solid_hex8_vpt_superelement_integration(tmp_path: Path) -> None:
    """Real 3D Hex8 VPT Craig-Bampton superelement (18x18 w/ 6 internal modes) integrates cleanly."""
    se_hex = condense_parametric_bracket(
        {
            "bracket_type": "corner_gusset",
            "arm_length_mm": 40.0,
            "height_mm": 20.0,
            "wall_thickness_mm": 6.0,
            "n_modes": 6,
        },
        material_id="aluminum_6061_t6",
        cache_dir=tmp_path,
    )
    assert se_hex.K_cb.shape == (18, 18)
    assert se_hex.M_cb.shape == (18, 18)

    p0_cad = np.array([0.0, 0.0, 0.0])
    p1_cad = np.array([0.0, 0.030, 0.0])
    p0_asm = np.array([0.10, 0.20, 0.30])
    p1_asm = np.array([0.12, 0.23, 0.28])

    nodes = [
        StructuralNode("root", p0_asm, fixed=True, role="frame"),
        StructuralNode("tip", p1_asm, fixed=False, lumped_mass_kg=0.15, role="toolhead"),
    ]
    se = AttachedSuperelement(
        id="hex_bracket",
        category="gantry_brackets",
        asset_id="corner_gusset",
        port_node_ids=("root", "tip"),
        K_cb=se_hex.K_cb,
        M_cb=se_hex.M_cb,
        port_coords_cad_m=np.vstack([p0_cad, p1_cad]),
        n_internal_modes=6,
        part_group_id="toolhead_mount",
        part_group_label="Toolhead & Nozzle Mount",
        role="toolhead",
        mass_kg=se_hex.mass_kg,
        fixed_interface_freqs_hz=list(se_hex.fixed_interface_freqs_hz),
    )
    asm = AssemblyModel(
        spec=PrinterSpec(),
        nodes=nodes,
        members=[],
        superelements=[se],
        toolhead_node_id="tip",
        toolhead_cog_node_id="tip",
        nozzle_node_id="tip",
        bed_node_id="",
        x_moving_mass_kg=0.15,
        y_moving_mass_kg=0.15,
        toolhead_moving_mass_kg=0.15,
    )
    K, M, _, _, free_dofs = assemble_global_matrices(asm)
    assert K.shape == (18, 18)
    assert len(free_dofs) == 12  # 6 DOFs at tip + 6 internal CB modes

    st = solve_static_fea(asm)
    mo = solve_modal_fea(asm)
    sen = compute_structural_sensitivities(asm, static_res=st, modal_res=mo)
    assert math.isfinite(st.toolhead_stiffness_x_n_per_mm) and st.toolhead_stiffness_x_n_per_mm > 0.0
    assert math.isfinite(mo.fundamental_freq_hz) and mo.fundamental_freq_hz > 10.0
    assert "hex_bracket" in sen.member_sensitivities
    assert sen.member_sensitivities["hex_bracket"]["compliance_share_pct"] == pytest.approx(100.0, abs=0.5)


def test_real_cad_assets_registry_if_present() -> None:
    """When Track A's baked cad_assets exist on disk, verify every asset end-to-end."""
    try:
        from printsim.cad_assets.registry import list_hardware_assets, load_superelement
    except ImportError:
        pytest.skip("printsim.cad_assets.registry not yet available on disk")

    catalog = list_hardware_assets()
    all_items = [
        (cat, aid, manifest)
        for cat, items in catalog.items()
        for aid, manifest in items.items()
    ]
    if not all_items:
        pytest.skip("No pre-baked hardware assets present in cad_assets yet")

    client = TestClient(app)
    resp = client.get("/api/assets/hardware")
    assert resp.status_code == 200

    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_monolith_awd.yaml")
    for cat, aid, manifest in all_items:
        asset = load_superelement(cat, aid)
        assert asset is not None
        assert asset.K_bb.shape == (12, 12)
        eigs_11 = np.linalg.eigvalsh(0.5 * (asset.K_bb[6:12, 6:12] + asset.K_bb[6:12, 6:12].T))
        assert np.all(eigs_11 > 0.0), f"{cat}/{aid} K_11 not strictly SPD"

        r_glb = client.get(f"/api/assets/hardware/{cat}/{aid}/visual.glb")
        assert r_glb.status_code == 200
        assert r_glb.headers["content-type"].startswith("model/gltf-binary")

        s = base.model_copy(deep=True)
        if cat == "hotends":
            s.toolhead.hotend = HotendSpec(preset_id=aid)
        elif cat == "toolheads":
            s.toolhead.preset_id = aid
        elif cat == "assemblies" and "__" in aid:
            th_id, he_id = aid.split("__", 1)
            s.toolhead.preset_id = th_id
            s.toolhead.hotend = HotendSpec(preset_id=he_id)
        asm = build_assembly(s)
        assert any(se.category == cat and se.asset_id == aid for se in asm.superelements)
        st = solve_static_fea(asm)
        mo = solve_modal_fea(asm)
        sen = compute_structural_sensitivities(asm, static_res=st, modal_res=mo)
        assert math.isfinite(st.combined_max_deflection_um)
        assert math.isfinite(mo.fundamental_freq_hz)
        comp_sum = sum(v["compliance_share_pct"] for v in sen.member_sensitivities.values()) + sum(
            v["compliance_share_pct"] for v in sen.panel_bay_sensitivities.values()
        )
        assert comp_sum == pytest.approx(100.0, abs=0.5)
        assert manifest.get("cad_trust_pct", 0) > 0

    # Real 75k-elem microwaved_bowden CAD toolhead is self-contained (immune to generic wall/plate/construction,
    # no double-counted Stage 1 backplate, scales cleanly with titanium_grade5 <-> aluminum_6061_t6 <-> ppa_cf_printed)
    if "microwaved_bowden" in catalog.get("toolheads", {}):
        from printsim.schema import TOOLHEAD_PRESETS

        mb_base = base.model_copy(deep=True)
        mb_preset = TOOLHEAD_PRESETS["microwaved_bowden"]
        mb_base.toolhead.preset_id = "microwaved_bowden"
        mb_base.toolhead.total_mass_g = float(mb_preset["total_mass_g"])
        mb_base.toolhead.cog_offset_mm = tuple(mb_preset["cog_offset_mm"])
        mb_base.toolhead.nozzle_offset_mm = tuple(mb_preset["nozzle_offset_mm"])
        mb_base.toolhead.mount_material = mb_preset["mount_material"]
        mb_base.toolhead.body_construction = mb_preset["body_construction"]
        mb_base.toolhead.hotend = HotendSpec(preset_id="tricorn_long")

        asm_mb = build_assembly(mb_base)
        stiff_mb = asm_mb.toolhead_stage_stiffnesses_n_per_mm
        gdict_mb = asm_mb.to_geometry_dict()["toolhead_structure"]
        assert gdict_mb["cad_toolhead_self_contained"] is True
        assert gdict_mb["cad_hotend_self_contained"] is True
        assert gdict_mb["cad_includes_backplate"] is True
        assert stiff_mb["backplate_only_n_per_mm"] == 0.0
        assert stiff_mb["carriage_mount_n_per_mm"] == pytest.approx(
            stiff_mb["rail_carriage_n_per_mm"], rel=1e-12
        )

        # Generic wall/plate/construction changes do NOT distort real microwaved_bowden CAD
        mb_slider = mb_base.model_copy(deep=True)
        mb_slider.toolhead.body_wall_thickness_mm = 11.0
        mb_slider.toolhead.carriage_plate_thickness_mm = 14.0
        mb_slider.toolhead.carriage_mount_material = "abs_printed"
        mb_slider.toolhead.body_construction = "multi_piece_clamshell"
        asm_slider = build_assembly(mb_slider)
        assert asm_slider.toolhead_stage_stiffnesses_n_per_mm["body_n_per_mm"] == pytest.approx(
            stiff_mb["body_n_per_mm"], rel=1e-12
        )

        # Material swaps (ppa_cf_printed < aluminum_6061_t6 < titanium_grade5) scale real CAD cleanly
        mb_ppa = mb_base.model_copy(deep=True)
        mb_ppa.toolhead.body_material = "ppa_cf_printed"
        mb_ti = mb_base.model_copy(deep=True)
        mb_ti.toolhead.body_material = "titanium_grade5"
        k_ppa = build_assembly(mb_ppa).toolhead_stage_stiffnesses_n_per_mm["body_n_per_mm"]
        k_al = stiff_mb["body_n_per_mm"]
        k_ti = build_assembly(mb_ti).toolhead_stage_stiffnesses_n_per_mm["body_n_per_mm"]
        assert k_ppa < k_al * 0.5
        assert k_ti > k_al * 1.5

        # Real coupled assembly bundle check when baked assemblies/microwaved_bowden__tricorn_long exists on disk
        if "microwaved_bowden__tricorn_long" in catalog.get("assemblies", {}):
            assert gdict_mb["cad_hotend_braced"] is True
            assert gdict_mb["hotend_visual_glb_url"] == "/api/assets/hardware/hotends/tricorn_long/visual.glb"
            assert any(
                se.id == "nozzle_stem"
                and se.category == "assemblies"
                and se.asset_id == "microwaved_bowden__tricorn_long"
                for se in asm_mb.superelements
            )

            sb_unbraced = base.model_copy(deep=True)
            sb_unbraced.toolhead.preset_id = "stealthburner"
            sb_unbraced.toolhead.hotend = HotendSpec(preset_id="tricorn_long")
            asm_unbraced = build_assembly(sb_unbraced)
            assert asm_unbraced.to_geometry_dict()["toolhead_structure"]["cad_hotend_braced"] is False
            assert any(
                se.id == "nozzle_stem"
                and se.category == "hotends"
                and se.asset_id == "tricorn_long"
                for se in asm_unbraced.superelements
            )

            st_mb = solve_static_fea(asm_mb)
            st_unbraced = solve_static_fea(asm_unbraced)
            assert st_mb.toolhead_stiffness_x_n_per_mm > st_unbraced.toolhead_stiffness_x_n_per_mm
            assert st_mb.toolhead_stiffness_y_n_per_mm > st_unbraced.toolhead_stiffness_y_n_per_mm
            assert (
                st_mb.toolhead_flex_breakdown["hotend_stiffness_n_per_mm"]
                > st_unbraced.toolhead_flex_breakdown["hotend_stiffness_n_per_mm"]
            )

            audit_mb = {e["id"]: e for e in build_model_confidence_audit(mb_base)["entries"]}
            hc_entry = audit_mb["hotend_clamp"]
            assert hc_entry["cad_superelement_active"] is True
            assert hc_entry["cad_superelement_id"] == "microwaved_bowden__tricorn_long"
            assert hc_entry["trust_pct"] == 89
            assert "+ Microwaved Ti Bottom Brace" in hc_entry["active_selection"]
            active_opts = [o for o in hc_entry["options_breakdown"] if o["active"]]
            assert len(active_opts) == 1
            assert active_opts[0]["id"] == "tricorn_long"
            assert active_opts[0]["trust_pct"] == hc_entry["trust_pct"]


def test_coupled_braced_assembly_superelement_mock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Mocked coupled assembly (microwaved_bowden__tricorn_long) raises tip stiffness, updates audit, and serves GLB."""
    _install_mock_cad_registry(monkeypatch, tmp_path)
    from printsim.schema import TOOLHEAD_PRESETS

    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_monolith_awd.yaml")
    mb_preset = TOOLHEAD_PRESETS["microwaved_bowden"]

    s_braced = base.model_copy(deep=True)
    s_braced.use_cad_superelements = True
    s_braced.toolhead.preset_id = "microwaved_bowden"
    s_braced.toolhead.total_mass_g = float(mb_preset["total_mass_g"])
    s_braced.toolhead.cog_offset_mm = tuple(mb_preset["cog_offset_mm"])
    s_braced.toolhead.nozzle_offset_mm = tuple(mb_preset["nozzle_offset_mm"])
    s_braced.toolhead.mount_material = mb_preset["mount_material"]
    s_braced.toolhead.body_construction = mb_preset["body_construction"]
    s_braced.toolhead.hotend = HotendSpec(preset_id="tricorn_long")

    s_unbraced = base.model_copy(deep=True)
    s_unbraced.use_cad_superelements = True
    s_unbraced.toolhead.preset_id = "stealthburner"
    s_unbraced.toolhead.hotend = HotendSpec(preset_id="tricorn_long")

    asm_braced = build_assembly(s_braced)
    asm_unbraced = build_assembly(s_unbraced)
    st_braced = solve_static_fea(asm_braced)
    st_unbraced = solve_static_fea(asm_unbraced)

    # 1. Superelement category/asset_id and geometry flags
    assert any(
        se.id == "nozzle_stem"
        and se.category == "assemblies"
        and se.asset_id == "microwaved_bowden__tricorn_long"
        for se in asm_braced.superelements
    )
    assert any(
        se.id == "nozzle_stem"
        and se.category == "hotends"
        and se.asset_id == "tricorn_long"
        for se in asm_unbraced.superelements
    )
    gdict_braced = asm_braced.to_geometry_dict()["toolhead_structure"]
    gdict_unbraced = asm_unbraced.to_geometry_dict()["toolhead_structure"]
    assert gdict_braced["cad_hotend_braced"] is True
    assert gdict_braced["cad_hotend_self_contained"] is True
    assert gdict_braced["cad_toolhead_self_contained"] is True
    # Keep hotend_visual_glb_url pointing to hotends/{he_pid}/visual.glb (no double-draw)
    assert gdict_braced["hotend_visual_glb_url"] == "/api/assets/hardware/hotends/tricorn_long/visual.glb"
    assert gdict_unbraced["cad_hotend_braced"] is False

    # 2. Strictly higher tip & hotend stiffness than unbraced stealthburner + tricorn_long
    assert st_braced.toolhead_stiffness_x_n_per_mm > st_unbraced.toolhead_stiffness_x_n_per_mm * 2.0
    assert st_braced.toolhead_stiffness_y_n_per_mm > st_unbraced.toolhead_stiffness_y_n_per_mm * 2.0
    assert (
        st_braced.toolhead_flex_breakdown["hotend_stiffness_n_per_mm"]
        > st_unbraced.toolhead_flex_breakdown["hotend_stiffness_n_per_mm"] * 5.0
    )

    # 3. Braced metal/SLM multi-point clamp vs polymer compliance scaling
    s_ti = s_braced.model_copy(deep=True)
    s_ti.toolhead.body_material = "titanium_grade5"
    s_al_thin = s_braced.model_copy(deep=True)
    s_al_thin.toolhead.hotend_mount_thickness_mm = 4.0
    s_abs = s_braced.model_copy(deep=True)
    s_abs.toolhead.body_material = "abs_printed"

    asm_ti = build_assembly(s_ti)
    asm_al_thin = build_assembly(s_al_thin)
    asm_abs = build_assembly(s_abs)
    assert asm_al_thin.toolhead_stage_stiffnesses_n_per_mm["hotend_clamp_n_per_mm"] >= 8000.0
    assert (
        asm_ti.toolhead_stage_stiffnesses_n_per_mm["hotend_clamp_n_per_mm"]
        > asm_braced.toolhead_stage_stiffnesses_n_per_mm["hotend_clamp_n_per_mm"]
        > asm_abs.toolhead_stage_stiffnesses_n_per_mm["hotend_clamp_n_per_mm"]
    )
    assert (
        asm_ti.toolhead_stage_stiffnesses_n_per_mm["hotend_total_n_per_mm"]
        > asm_braced.toolhead_stage_stiffnesses_n_per_mm["hotend_total_n_per_mm"]
        > asm_abs.toolhead_stage_stiffnesses_n_per_mm["hotend_total_n_per_mm"]
    )
    assert asm_abs.toolhead_stage_stiffnesses_n_per_mm["hotend_clamp_n_per_mm"] < 3000.0

    # 4. Confidence audit entry #18 (hotend_clamp) reports braced assembly manifest & matching breakdown trust
    audit_braced = {e["id"]: e for e in build_model_confidence_audit(s_braced)["entries"]}
    hc_braced = audit_braced["hotend_clamp"]
    assert hc_braced["cad_superelement_active"] is True
    assert hc_braced["cad_superelement_id"] == "microwaved_bowden__tricorn_long"
    assert hc_braced["trust_pct"] == 89
    assert "+ Microwaved Ti Bottom Brace" in hc_braced["active_selection"]
    assert "3,420/3,380 N/mm" in hc_braced["active_values"]
    active_opts = [o for o in hc_braced["options_breakdown"] if o["active"]]
    assert len(active_opts) == 1
    assert active_opts[0]["id"] == "tricorn_long"
    assert active_opts[0]["trust_pct"] == hc_braced["trust_pct"]

    # Disabling CAD superelements reverts braced flags & trust
    s_off = s_braced.model_copy(deep=True)
    s_off.use_cad_superelements = False
    asm_off = build_assembly(s_off)
    assert asm_off.superelements == []
    assert asm_off.to_geometry_dict()["toolhead_structure"]["cad_hotend_braced"] is False
    hc_off = {e["id"]: e for e in build_model_confidence_audit(s_off)["entries"]}["hotend_clamp"]
    assert hc_off["cad_superelement_active"] is False
    assert hc_off["cad_superelement_id"] is None
    assert hc_off["trust_pct"] == 78

    # 5. FastAPI serves assemblies GLB & catalog
    client = TestClient(app)
    r_list = client.get("/api/assets/hardware")
    assert r_list.status_code == 200
    assert "microwaved_bowden__tricorn_long" in r_list.json().get("assemblies", {})
    r_glb = client.get("/api/assets/hardware/assemblies/microwaved_bowden__tricorn_long/visual.glb")
    assert r_glb.status_code == 200
    assert r_glb.headers["content-type"].startswith("model/gltf-binary")
    assert r_glb.content.startswith(b"glTF")


def test_ti_bottom_brace_toggle_and_nozzle_shield_mass_cog() -> None:
    """Verify Ti bottom brace toggle & nozzle shield selector update stiffness, mass, CoG, and audit."""
    from printsim.geometry.builder import build_assembly
    from printsim.fea.solver import solve_static_fea
    from printsim.schema.overrides import build_model_confidence_audit
    from printsim.schema.printer import HOTEND_CATALOG, TOOLHEAD_PRESETS, PrinterSpec

    base = PrinterSpec.from_yaml("examples/corexy_voron_style.yaml")
    th_p = TOOLHEAD_PRESETS["microwaved_bowden"]
    he_p = HOTEND_CATALOG["tricorn_long"]
    base.use_cad_superelements = True
    base.toolhead.preset_id = "microwaved_bowden"
    base.toolhead.name = str(th_p["name"])
    base.toolhead.total_mass_g = float(th_p["total_mass_g"])
    base.toolhead.cog_offset_mm = tuple(float(v) for v in th_p["cog_offset_mm"])
    base.toolhead.nozzle_offset_mm = tuple(float(v) for v in th_p["nozzle_offset_mm"])
    base.toolhead.mount_material = str(th_p["mount_material"])
    base.toolhead.body_material = "titanium_grade5"
    base.toolhead.hotend.preset_id = "tricorn_long"
    base.toolhead.hotend.name = str(he_p["name"])
    base.toolhead.hotend.mass_g = float(he_p["mass_g"])
    base.toolhead.hotend.mount_style = str(he_p["mount_style"])
    base.toolhead.hotend.structural_stiffness_n_per_mm = float(he_p["structural_stiffness_n_per_mm"])
    base.toolhead.hotend.overall_length_mm = float(he_p["overall_length_mm"])

    # 1. Default: Ti bottom brace ON + Microwaved SLM Ti shield (removes 2.67 g stock Tricorn bucket)
    s_on = base.model_copy(deep=True)
    s_on.toolhead.use_hotend_bottom_brace = True
    s_on.toolhead.nozzle_shield_mode = "microwaved"
    asm_on = build_assembly(s_on)
    st_on = solve_static_fea(asm_on)
    g_on = asm_on.to_geometry_dict()["toolhead_structure"]
    assert g_on["cad_hotend_brace_available"] is True
    assert g_on["cad_hotend_braced"] is True
    assert g_on["use_hotend_bottom_brace"] is True
    assert g_on["nozzle_shield_mode"] == "microwaved"
    assert g_on["nozzle_shield_options"] == ["microwaved", "stock", "none"]
    assert g_on["hotend_stiffness_n_per_mm"] > 2000.0
    assert abs(g_on["hotend_mass_g"] - (62.0 - 2.67)) < 0.05
    assert abs(g_on["toolhead_body_mass_g"] - 130.0) < 0.05

    # 2. Toggle Ti bottom brace OFF: drops to standalone tricorn_long (~263 N/mm) & removes 1.67 g brace
    s_off = s_on.model_copy(deep=True)
    s_off.toolhead.use_hotend_bottom_brace = False
    asm_off = build_assembly(s_off)
    st_off = solve_static_fea(asm_off)
    g_off = asm_off.to_geometry_dict()["toolhead_structure"]
    assert g_off["cad_hotend_brace_available"] is True
    assert g_off["cad_hotend_braced"] is False
    assert g_off["use_hotend_bottom_brace"] is False
    assert g_off["hotend_stiffness_n_per_mm"] < 300.0
    assert st_on.toolhead_flex_breakdown["hotend_stiffness_n_per_mm"] > st_off.toolhead_flex_breakdown["hotend_stiffness_n_per_mm"] * 5.0
    from printsim.fea.modal import solve_modal_fea

    mf_on = solve_modal_fea(asm_on)
    mf_off = solve_modal_fea(asm_off)
    assert mf_on.shaper_x is not None and mf_off.shaper_x is not None
    assert mf_on.shaper_x.effective_freq_hz > mf_off.shaper_x.effective_freq_hz
    assert mf_on.shaper_x.max_accel_mm_s2 >= mf_off.shaper_x.max_accel_mm_s2
    assert abs((g_on["assembled_toolhead_mass_g"] - g_off["assembled_toolhead_mass_g"]) - 1.67) < 0.05
    # Removing lower brace shifts CoG upward (less negative Z)
    assert g_off["effective_cog_offset_mm"][2] > g_on["effective_cog_offset_mm"][2]
    audit_off = {e["id"]: e for e in build_model_confidence_audit(s_off)["entries"]}["hotend_clamp"]
    assert audit_off["cad_superelement_id"] == "tricorn_long"
    assert audit_off["trust_pct"] == 88

    # 3. Nozzle shield modes: stock Tricorn bucket (+2.67 g, -1.27 g MBT shield = +1.40 g vs microwaved)
    #    and none (-1.27 g vs microwaved, -2.67 g vs stock), updating both mass and CoG
    s_stock = s_on.model_copy(deep=True)
    s_stock.toolhead.nozzle_shield_mode = "stock"
    g_stock = build_assembly(s_stock).to_geometry_dict()["toolhead_structure"]
    assert g_stock["nozzle_shield_mode"] == "stock"
    assert abs(g_stock["hotend_mass_g"] - 62.0) < 0.05
    assert abs(g_stock["toolhead_body_mass_g"] - (130.0 - 1.27)) < 0.05
    assert abs((g_stock["assembled_toolhead_mass_g"] - g_on["assembled_toolhead_mass_g"]) - 1.40) < 0.05

    s_none = s_on.model_copy(deep=True)
    s_none.toolhead.nozzle_shield_mode = "none"
    g_none = build_assembly(s_none).to_geometry_dict()["toolhead_structure"]
    assert g_none["nozzle_shield_mode"] == "none"
    assert abs(g_none["hotend_mass_g"] - (62.0 - 2.67)) < 0.05
    assert abs(g_none["toolhead_body_mass_g"] - (130.0 - 1.27)) < 0.05
    assert g_none["assembled_toolhead_mass_g"] < g_on["assembled_toolhead_mass_g"] < g_stock["assembled_toolhead_mass_g"]
    # Bare nozzle has highest (least negative) CoG Z; heavy stock bucket pulls CoG lowest (most negative Z)
    assert g_none["effective_cog_offset_mm"][2] > g_on["effective_cog_offset_mm"][2] > g_stock["effective_cog_offset_mm"][2]


def test_braced_microwaved_chassis_pitch_stiffness_nozzle_load_and_accessories() -> None:
    """Verify braced Microwaved chassis pitch stiffness, material scaling, nozzle_stem inertial load, and accessory gating."""
    from printsim.fea.beam3d import superelement_element_matrices
    from printsim.fea.solver import build_inertial_load_vectors
    from printsim.geometry.builder import resolve_toolhead_hardware_accessories
    from printsim.schema import TOOLHEAD_PRESETS

    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_monolith_awd.yaml")
    th_p = TOOLHEAD_PRESETS["microwaved_bowden"]
    base.use_cad_superelements = True
    base.toolhead.preset_id = "microwaved_bowden"
    base.toolhead.name = str(th_p["name"])
    base.toolhead.total_mass_g = float(th_p["total_mass_g"])
    base.toolhead.cog_offset_mm = tuple(float(v) for v in th_p["cog_offset_mm"])
    base.toolhead.nozzle_offset_mm = tuple(float(v) for v in th_p["nozzle_offset_mm"])
    base.toolhead.mount_material = str(th_p["mount_material"])
    base.toolhead.body_material = "aluminum_6061_t6"
    base.toolhead.hotend = HotendSpec(preset_id="tricorn_long")

    s_on = base.model_copy(deep=True)
    s_on.toolhead.use_hotend_bottom_brace = True
    s_off = base.model_copy(deep=True)
    s_off.toolhead.use_hotend_bottom_brace = False

    asm_on = build_assembly(s_on)
    asm_off = build_assembly(s_off)
    st_on = solve_static_fea(asm_on)
    st_off = solve_static_fea(asm_off)
    mo_on = solve_modal_fea(asm_on)
    mo_off = solve_modal_fea(asm_off)
    sen_on = compute_structural_sensitivities(asm_on, static_res=st_on, modal_res=mo_on)
    sen_off = compute_structural_sensitivities(asm_off, static_res=st_off, modal_res=mo_off)

    # 1. Al 6061-T6 Microwaved bulk monoblock chassis stays invariant (~2000-2800 N/mm) when
    #    toggling the hotend bottom brace OFF (only hotend_total_n_per_mm & nozzle whip drop!)
    body_k_on = asm_on.toolhead_stage_stiffnesses_n_per_mm["body_n_per_mm"]
    body_k_off = asm_off.toolhead_stage_stiffnesses_n_per_mm["body_n_per_mm"]
    assert 2000.0 <= body_k_on <= 2800.0
    assert body_k_on == pytest.approx(body_k_off, rel=0.02)
    assert (
        asm_on.toolhead_stage_stiffnesses_n_per_mm["hotend_total_n_per_mm"]
        > asm_off.toolhead_stage_stiffnesses_n_per_mm["hotend_total_n_per_mm"] * 5.0
    )
    # 35.5 mm compact Microwaved drop uses true 35.5 mm lever arm (> 4500 N/mm vs penalized 45 mm floor)
    assert asm_on.toolhead_stage_stiffnesses_n_per_mm["rail_carriage_n_per_mm"] > 4500.0

    # 2. Static FEA & adjoint split inertial load onto nozzle_tip -> positive nozzle_stem energy & brace lowers accel_deflection_x_um
    assert sen_off.member_sensitivities["nozzle_stem"]["compliance_share_pct"] > 5.0
    assert sen_off.member_sensitivities["nozzle_stem"]["du_rel_x_dlnE_m"] < 0.0
    assert sen_on.member_sensitivities["nozzle_stem"]["compliance_share_pct"] > 1.0
    assert sen_on.member_sensitivities["nozzle_stem"]["du_rel_x_dlnE_m"] < 0.0
    assert st_on.accel_deflection_x_um < st_off.accel_deflection_x_um * 0.85
    assert st_on.accel_deflection_y_um < st_off.accel_deflection_y_um * 0.85

    # Explicit static strain energy check on nozzle_stem under static X inertial load
    K_off, _, _, nidx_off, free_off = assemble_global_matrices(asm_off)
    from printsim.fea.solver import _build_factorized_solver

    fx_off, _, _, _ = build_inertial_load_vectors(
        asm_off, nidx_off, K_off.shape[0], s_off.kinematics.max_accel_mm_s2 * 1e-3
    )
    u_x_off = _build_factorized_solver(K_off, free_off)(fx_off)
    cursor = 6 * len(asm_off.nodes)
    se_strain_e = 0.0
    for se in asm_off.superelements:
        dofs_se, Ke_se, _ = superelement_element_matrices(se, asm_off, nidx_off, cursor)
        cursor += max(0, int(getattr(se, "n_internal_modes", max(0, Ke_se.shape[0] - 12))))
        if se.id == "nozzle_stem":
            u_se = u_x_off[dofs_se]
            se_strain_e = 0.5 * float(u_se @ Ke_se @ u_se)
    assert se_strain_e > 0.0

    # 3. Al-scaled toolhead_mount SE reports ~43.8 g mass & material-scaled port_stiffness_n_per_mm
    se_al = next(se for se in asm_on.superelements if se.id == "toolhead_mount")
    s_ti = s_on.model_copy(deep=True)
    s_ti.toolhead.body_material = "titanium_grade5"
    asm_ti = build_assembly(s_ti)
    se_ti = next(se for se in asm_ti.superelements if se.id == "toolhead_mount")
    assert se_al.mass_kg == pytest.approx(0.0438, abs=0.001)
    assert se_ti.mass_kg == pytest.approx(0.0719, abs=0.001)
    assert se_al.to_summary_dict()["port_stiffness_n_per_mm"]["kx"] < se_ti.to_summary_dict()["port_stiffness_n_per_mm"]["kx"] * 0.7

    # 4. Accessory envelope gating: Chube does NOT offer microwaved shield or subtract -1.67 g brace delta
    s_chube = s_off.model_copy(deep=True)
    s_chube.toolhead.hotend = HotendSpec(preset_id="chube_conduction")
    acc_chube = resolve_toolhead_hardware_accessories(s_chube)
    assert acc_chube["nozzle_shield_options"] == []
    assert "microwaved" not in acc_chube["nozzle_shield_options"]
    assert acc_chube["body_delta_g"] == 0.0
    assert acc_chube["effective_body_mass_g"] == pytest.approx(130.0, abs=1e-6)

    s_short = s_off.model_copy(deep=True)
    s_short.toolhead.hotend = HotendSpec(preset_id="tricorn_short")
    acc_short = resolve_toolhead_hardware_accessories(s_short)
    assert acc_short["nozzle_shield_options"] == ["microwaved", "stock", "none"]
    assert acc_short["body_delta_g"] == 0.0  # -1.67 g brace delta gated to tricorn_long only


def test_microwaved_bowden_chube_and_exact_cutout_chube_superelements() -> None:
    """Verify baked Microwaved Bowden Chube V1 + exact-cutout Chube Compact/Air/Conduction assets."""
    from printsim.schema import TOOLHEAD_PRESETS

    client = TestClient(app)
    for url in (
        "/api/assets/hardware/toolheads/microwaved_bowden_chube/visual.glb",
        "/api/assets/hardware/hotends/chube_conduction/visual.glb",
        "/api/assets/hardware/hotends/chube_air/visual.glb",
        "/api/assets/hardware/hotends/chube_compact/visual.glb",
    ):
        resp = client.get(url)
        assert resp.status_code == 200, url
        assert resp.content[:4] == b"glTF", url

    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_monolith_awd.yaml")
    th_p = TOOLHEAD_PRESETS["microwaved_bowden_chube"]
    base.use_cad_superelements = True
    base.toolhead.preset_id = "microwaved_bowden_chube"
    base.toolhead.name = str(th_p["name"])
    base.toolhead.total_mass_g = float(th_p["total_mass_g"])
    base.toolhead.cog_offset_mm = tuple(float(v) for v in th_p["cog_offset_mm"])
    base.toolhead.nozzle_offset_mm = tuple(float(v) for v in th_p["nozzle_offset_mm"])
    base.toolhead.mount_material = str(th_p["mount_material"])
    base.toolhead.body_material = "aluminum_6061_t6"
    base.toolhead.hotend = HotendSpec(preset_id="chube_conduction")

    asm_al = build_assembly(base)
    se_th_al = next(se for se in asm_al.superelements if se.id == "toolhead_mount")
    se_he_cond = next(se for se in asm_al.superelements if se.id == "nozzle_stem")
    assert se_th_al.asset_id == "microwaved_bowden_chube"
    assert se_he_cond.asset_id == "chube_conduction"
    assert se_th_al.mass_kg == pytest.approx(0.0565, abs=0.002)

    # Material scaling: aluminum_6061_t6 -> titanium_grade5 scales stiffness (~1.65x) and mass (~1.64x)
    s_ti = base.model_copy(deep=True)
    s_ti.toolhead.body_material = "titanium_grade5"
    asm_ti = build_assembly(s_ti)
    se_th_ti = next(se for se in asm_ti.superelements if se.id == "toolhead_mount")
    assert se_th_ti.mass_kg > se_th_al.mass_kg * 1.55
    assert (
        se_th_ti.to_summary_dict()["port_stiffness_n_per_mm"]["kx"]
        > se_th_al.to_summary_dict()["port_stiffness_n_per_mm"]["kx"] * 1.55
    )

    # Exact-cutout Chube stiffness ordering: Compact (~2975) > Conduction (~815) > Air (~560)
    k_by_hotend: dict[str, float] = {}
    for he_id in ("chube_compact", "chube_conduction", "chube_air"):
        sp = base.model_copy(deep=True)
        sp.toolhead.hotend = HotendSpec(preset_id=he_id)
        asm = build_assembly(sp)
        se_he = next(se for se in asm.superelements if se.id == "nozzle_stem")
        pk = se_he.to_summary_dict()["port_stiffness_n_per_mm"]
        k_by_hotend[he_id] = 0.5 * (float(pk["kx"]) + float(pk["ky"]))

    assert 2500.0 <= k_by_hotend["chube_compact"] <= 3500.0
    assert 700.0 <= k_by_hotend["chube_conduction"] <= 950.0
    assert 480.0 <= k_by_hotend["chube_air"] <= 650.0
    assert k_by_hotend["chube_compact"] > k_by_hotend["chube_conduction"] > k_by_hotend["chube_air"]

    # Confidence audit elevates toolhead_preset and hotend_clamp to CAD CMS trust
    audit = {e["id"]: e for e in build_model_confidence_audit(base)["entries"]}
    assert audit["toolhead_preset"]["trust_pct"] >= 86
    assert audit["hotend_clamp"]["cad_superelement_id"] == "chube_conduction"
    assert audit["hotend_clamp"]["trust_pct"] >= 86


def test_el_fuego_slm_toolhead_superelement_and_chube_integration() -> None:
    """Verify baked El Fuego 3-part SLM toolhead CMS, material scaling, and Chube/Tricorn pairing."""
    from printsim.schema import TOOLHEAD_PRESETS

    client = TestClient(app)
    resp = client.get("/api/assets/hardware/toolheads/el_fuego/visual.glb")
    assert resp.status_code == 200
    assert resp.content[:4] == b"glTF"

    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_monolith_awd.yaml")
    th_p = TOOLHEAD_PRESETS["el_fuego"]
    base.use_cad_superelements = True
    base.toolhead.preset_id = "el_fuego"
    base.toolhead.name = str(th_p["name"])
    base.toolhead.total_mass_g = float(th_p["total_mass_g"])
    base.toolhead.cog_offset_mm = tuple(float(v) for v in th_p["cog_offset_mm"])
    base.toolhead.nozzle_offset_mm = tuple(float(v) for v in th_p["nozzle_offset_mm"])
    base.toolhead.mount_material = str(th_p["mount_material"])
    base.toolhead.body_material = "slm_aluminum"
    base.toolhead.cpap_blower_id = str(th_p["cpap_blower_id"])
    base.toolhead.shared_heatsink_cpap_duct = True
    base.toolhead.hotend = HotendSpec(preset_id="chube_conduction")

    asm_al = build_assembly(base)
    se_th_al = next(se for se in asm_al.superelements if se.id == "toolhead_mount")
    se_he_cond = next(se for se in asm_al.superelements if se.id == "nozzle_stem")
    assert se_th_al.asset_id == "el_fuego"
    assert se_he_cond.asset_id == "chube_conduction"
    assert se_th_al.mass_kg == pytest.approx(0.09385, abs=0.002)
    assert asm_al.toolhead_stage_stiffnesses_n_per_mm["backplate_only_n_per_mm"] == 0.0
    assert 850.0 <= asm_al.toolhead_stage_stiffnesses_n_per_mm["body_n_per_mm"] <= 1250.0

    # Material scaling: slm_aluminum (68 GPa) -> titanium_grade5 (113.8 GPa) scales stiffness ~1.67x
    s_ti = base.model_copy(deep=True)
    s_ti.toolhead.body_material = "titanium_grade5"
    asm_ti = build_assembly(s_ti)
    se_th_ti = next(se for se in asm_ti.superelements if se.id == "toolhead_mount")
    assert se_th_ti.mass_kg > se_th_al.mass_kg * 1.60
    assert (
        se_th_ti.to_summary_dict()["port_stiffness_n_per_mm"]["kx"]
        > se_th_al.to_summary_dict()["port_stiffness_n_per_mm"]["kx"] * 1.60
    )

    # Full multiphysics evaluation succeeds cleanly with Chube Conduction & Tricorn Long
    rep = evaluate_printer(base)
    assert rep.modal_fea.shaper_x.effective_freq_hz > 40.0
    assert rep.modal_fea.shaper_y.effective_freq_hz > 40.0

    audit = {e["id"]: e for e in build_model_confidence_audit(base)["entries"]}
    assert audit["toolhead_preset"]["cad_superelement_id"] == "el_fuego"
    assert audit["toolhead_preset"]["trust_pct"] >= 85
