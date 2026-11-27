"""Unit tests for per-part toolhead FEA, modal flex breakdown, and 3D CB sub-part recovery."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from printsim.cad_assets.registry import SubpartRecoveryData
from printsim.engine import evaluate_printer
from printsim.fea.toolhead_parts import (
    compute_modal_toolhead_parts,
    compute_static_toolhead_parts,
)
from printsim.geometry.builder import build_assembly
from printsim.schema import PrinterSpec


def _load_voron_spec() -> PrinterSpec:
    p = Path(__file__).resolve().parent.parent / "examples" / "corexy_voron_style.yaml"
    return PrinterSpec.model_validate(yaml.safe_load(p.read_text(encoding="utf-8")))


def test_static_and_modal_toolhead_parts_populated() -> None:
    spec = _load_voron_spec()
    report = evaluate_printer(spec)

    static_parts = report.static_fea.toolhead_parts
    assert len(static_parts) >= 5
    stages = {p["stage"] for p in static_parts}
    assert "rail_carriage" in stages
    assert "chassis" in stages or "chassis_carrier" in stages
    assert "hotend_neck" in stages
    assert "heater_nozzle" in stages
    assert "cooling_shroud" in stages

    for p in static_parts:
        assert p["compliance_share_pct"] >= 0.0
        assert p["local_flex_um"] >= 0.0
        assert p["total_disp_um"] >= 0.0
        assert p["source"] in ("stage_fea", "cad_3d_recovery")

    # Structural toolhead compliance shares sum to ~100% of toolhead-only compliance
    th_share_sum = sum(p["toolhead_share_pct"] for p in static_parts if p["is_structural"])
    assert 98.0 <= th_share_sum <= 102.0

    # Every vibration mode carries per-part toolhead modal flex breakdown
    assert len(report.modal_fea.modes) >= 1
    for mode in report.modal_fea.modes:
        assert len(mode.toolhead_part_flex) >= 5
        modal_flex_sum = sum(p["modal_flex_pct"] for p in mode.toolhead_part_flex)
        assert modal_flex_sum == 0.0 or (98.0 <= modal_flex_sum <= 102.0)
        kin_sum = sum(p["kinetic_share_pct"] for p in mode.toolhead_part_flex)
        assert kin_sum == 0.0 or (98.0 <= kin_sum <= 102.0)


def test_subpart_recovery_energy_conservation_and_scaling() -> None:
    # Build two synthetic symmetric positive semi-definite 12x12 part matrices
    rng = np.random.default_rng(42)
    A1 = rng.normal(size=(12, 12))
    A2 = rng.normal(size=(12, 12))
    K1 = A1.T @ A1 * 1e6
    K2 = A2.T @ A2 * 2e6
    M1 = np.eye(12) * 0.03
    M2 = np.eye(12) * 0.05

    disp_op = np.zeros((2, 3, 12), dtype=np.float64)
    disp_op[0, :, 6:9] = 0.4 * np.eye(3)
    disp_op[1, :, 6:9] = 0.8 * np.eye(3)

    stress_op = np.zeros((2, 6, 12), dtype=np.float64)
    stress_op[0, 0, 6] = 1.0e10  # 10 MPa per mm
    stress_op[1, 3, 6] = 2.0e10

    rec = SubpartRecoveryData(
        names=("carriage_mount_zone", "main_chassis"),
        materials=("abs_printed", "abs_printed"),
        mass_kg=np.array([0.03, 0.05]),
        com_m=np.array([[0.0, 0.0, 0.0], [0.0, -0.02, 0.0]]),
        K_bb=np.stack([K1, K2], axis=0),
        M_bb=np.stack([M1, M2], axis=0),
        disp_op=disp_op,
        stress_op=stress_op,
    )

    q_b = np.zeros(12, dtype=np.float64)
    q_b[6] = 1e-4  # 0.1 mm X deflection at Port 1
    out = rec.recover(q_b)
    assert len(out) == 2
    assert abs(sum(r["strain_energy_pct"] for r in out) - 100.0) < 0.05
    assert abs(sum(r["kinetic_energy_pct"] for r in out) - 100.0) < 0.05
    assert out[0]["local_flex_um"] > 0.0
    assert out[1]["local_flex_um"] > out[0]["local_flex_um"]
    assert out[0]["stress_mpa"] > 0.0
    assert out[1]["stress_mpa"] > out[0]["stress_mpa"]

    # Scaling stiffness by 2x scales strain energy and stress by 2x while preserving percentages
    rec2 = rec.scaled(scale_k=2.0, scale_m=1.5)
    out2 = rec2.recover(q_b)
    assert np.isclose(out2[0]["stress_mpa"], 2.0 * out[0]["stress_mpa"], rtol=1e-3)
    assert np.isclose(out2[0]["mass_g"], 1.5 * out[0]["mass_g"], rtol=1e-3)
    assert np.isclose(out2[0]["strain_energy_pct"], out[0]["strain_energy_pct"], atol=0.05)


def test_toolhead_parts_upgrades_to_cad_3d_recovery_when_attached() -> None:
    spec = _load_voron_spec()
    asm = build_assembly(spec)
    th_se = next((se for se in asm.superelements if se.id == "toolhead_mount"), None)
    assert th_se is not None

    K1 = np.eye(12) * 1e6
    K2 = np.eye(12) * 3e6
    M1 = np.eye(12) * 0.04
    M2 = np.eye(12) * 0.06
    disp_op = np.zeros((2, 3, 12))
    disp_op[0, :, 6:9] = 0.3 * np.eye(3)
    disp_op[1, :, 6:9] = 0.7 * np.eye(3)
    stress_op = np.zeros((2, 6, 12))
    stress_op[0, 0, 6] = 5e9
    stress_op[1, 0, 6] = 1.5e10

    th_se.subpart_recovery = SubpartRecoveryData(
        names=("carriage_mount_zone", "main_chassis"),
        materials=("abs_printed", "abs_printed"),
        mass_kg=np.array([0.04, 0.06]),
        com_m=np.array([[0.0, 0.0, 0.0], [0.0, -0.02, 0.0]]),
        K_bb=np.stack([K1, K2], axis=0),
        M_bb=np.stack([M1, M2], axis=0),
        disp_op=disp_op,
        stress_op=stress_op,
    )

    node_idx = {n.id: i for i, n in enumerate(asm.nodes)}
    u_vec = np.zeros(6 * len(asm.nodes), dtype=np.float64)
    junc_id = "toolhead_chassis" if "toolhead_chassis" in node_idx else asm.toolhead_cog_node_id
    u_vec[6 * node_idx[junc_id]] = 2e-5
    u_vec[6 * node_idx[asm.nozzle_node_id]] = 3e-5

    static_parts = compute_static_toolhead_parts(
        asm,
        node_idx,
        u_vec,
        {"carriage_and_body_pct": 60.0, "hotend_and_nozzle_pct": 10.0},
    )
    cad_th_parts = [
        p for p in static_parts if p["source"] == "cad_3d_recovery" and p["id"].startswith("cad_th_")
    ]
    assert len(cad_th_parts) == 2
    assert {p["id"] for p in cad_th_parts} == {"cad_th_carriage_mount_zone", "cad_th_main_chassis"}

    modal_parts = compute_modal_toolhead_parts(asm, node_idx, u_vec, max_node_disp=3e-5)
    cad_th_modal = [
        p for p in modal_parts if p["source"] == "cad_3d_recovery" and p["id"].startswith("cad_th_")
    ]
    assert len(cad_th_modal) == 2


def test_baked_3d_intra_part_field_recovery_and_rigid_invariance() -> None:
    """Verify baked archetype_cpap & chube_conduction expose 3D Tet4 internal fields with zero rigid-body stress."""
    from printsim.cad_assets.registry import load_superelement

    for cat, aid in (("toolheads", "archetype_cpap"), ("hotends", "chube_conduction")):
        asset = load_superelement(cat, aid)
        assert asset is not None and asset.subpart_recovery is not None
        rec = asset.subpart_recovery
        assert rec.field_points_m is not None and len(rec.field_points_m) >= 32
        assert rec.field_disp_op is not None and rec.field_stress_op is not None

        # Pure rigid-body translation of both ports must yield zero flex and zero stress
        q_rb = np.array([1e-3, -2e-3, 5e-4, 0.0, 0.0, 0.0, 1e-3, -2e-3, 5e-4, 0.0, 0.0, 0.0])
        rb_field = rec.recover_field(q_rb)
        assert rb_field is not None
        assert rb_field["max_flex_um"] < 1e-2
        assert rb_field["max_stress_mpa"] < 1e-2

        # Cantilever X deflection at Port 1 must produce a non-trivial spatial gradient
        q_bend = np.zeros(12, dtype=np.float64)
        q_bend[6] = 1e-4  # +0.1 mm X at Port 1
        bend_field = rec.recover_field(q_bend)
        assert bend_field is not None
        assert bend_field["max_flex_um"] > 10.0
        assert bend_field["max_stress_mpa"] > 0.5
        assert min(bend_field["stress_mpa"]) < bend_field["max_stress_mpa"]

    # End-to-end evaluation populates static toolhead_part_fields and modal toolhead_mode_fields
    spec = _load_voron_spec()
    spec.toolhead.preset_id = "archetype_cpap"
    spec.toolhead.hotend.preset_id = "chube_conduction"
    report = evaluate_printer(spec)

    assert "toolhead_mount" in report.static_fea.toolhead_part_fields
    assert "nozzle_stem" in report.static_fea.toolhead_part_fields
    for key in ("toolhead_mount", "nozzle_stem"):
        fld = report.static_fea.toolhead_part_fields[key]
        assert fld["source"] == "cad_3d_tet4"
        assert len(fld["points_mm"]) >= 32
        assert fld["max_stress_mpa"] > 0.0

    assert len(report.modal_fea.modes) >= 1
    m0 = report.modal_fea.modes[0]
    assert "toolhead_mount" in m0.toolhead_mode_fields
    assert "nozzle_stem" in m0.toolhead_mode_fields
    assert len(m0.toolhead_mode_fields["toolhead_mount"]["strain_norm"]) >= 32
