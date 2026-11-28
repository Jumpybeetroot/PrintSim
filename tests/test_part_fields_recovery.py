"""Unit tests for printer-wide 3D Tet4 intra-part FEA field recovery (`part_fields` & `mode_part_fields`)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from printsim.cad_assets.registry import load_superelement
from printsim.engine import evaluate_printer
from printsim.fea.beam3d import assemble_global_system
from printsim.fea.toolhead_parts import compute_printer_part_fields, compute_toolhead_part_fields
from printsim.geometry.builder import AttachedSuperelement, build_assembly
from printsim.schema import PrinterSpec

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"

REQUIRED_FIELD_KEYS = {
    "source",
    "asset_id",
    "label",
    "node_ids",
    "points_mm",
    "flex_um",
    "stress_mpa",
    "max_flex_um",
    "max_stress_mpa",
    "subpart_idx",
    "subparts",
}


def _assert_valid_field_entry(entry: dict[str, Any], *, require_positive: bool = True, is_modal: bool = False) -> None:
    missing = REQUIRED_FIELD_KEYS - set(entry.keys())
    assert not missing, f"Missing required keys: {missing}"
    assert entry["source"] == "cad_3d_tet4"
    assert isinstance(entry["asset_id"], str) and len(entry["asset_id"]) > 0
    assert isinstance(entry["label"], str) and len(entry["label"]) > 0
    assert isinstance(entry["node_ids"], list) and len(entry["node_ids"]) == 2
    assert all(isinstance(n, str) and len(n) > 0 for n in entry["node_ids"])

    n_pts = len(entry["points_mm"])
    assert n_pts >= 16
    assert len(entry["flex_um"]) == n_pts
    assert len(entry["stress_mpa"]) == n_pts
    assert isinstance(entry["subparts"], list) and len(entry["subparts"]) >= 1
    if entry["subpart_idx"]:
        assert len(entry["subpart_idx"]) == n_pts

    assert entry["max_flex_um"] >= 0.0
    assert entry["max_stress_mpa"] >= 0.0
    if require_positive:
        assert entry["max_flex_um"] > 0.0
        assert entry["max_stress_mpa"] > 0.0

    if is_modal:
        assert "flex_norm" in entry and len(entry["flex_norm"]) == n_pts
        assert "strain_norm" in entry and len(entry["strain_norm"]) == n_pts


def _build_global_rigid_body_displacement(
    assembly: Any,
    n_dofs: int,
    trans_m: np.ndarray,
    rot_rad: np.ndarray,
) -> np.ndarray:
    """Construct an exact 6-DOF global rigid-body motion vector ``u(p) = t + theta x p, r = theta``."""
    u_rb = np.zeros(n_dofs, dtype=np.float64)
    for i, node in enumerate(assembly.nodes):
        u_rb[6 * i : 6 * i + 3] = trans_m + np.cross(rot_rad, node.position_m)
        u_rb[6 * i + 3 : 6 * i + 6] = rot_rad
    return u_rb


@pytest.mark.parametrize(
    "yaml_name,expected_carriages",
    [
        ("corexy_voron_style.yaml", ("x_carriage", "y_car_bearing_l", "y_car_bearing_r")),
        ("voron_monolith_awd.yaml", ("x_carriage", "y_car_bearing_l", "y_car_bearing_r")),
    ],
)
def test_printer_wide_static_and_modal_part_fields(
    yaml_name: str,
    expected_carriages: tuple[str, ...],
) -> None:
    """Verify default Voron & Monolith specs recover static & modal 3D Tet4 fields for toolhead, hotend & HIWIN carriages."""
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / yaml_name)
    report = evaluate_printer(spec)

    static_pf = report.static_fea.part_fields
    expected_keys = ("toolhead_mount", "nozzle_stem", *expected_carriages)
    for key in expected_keys:
        assert key in static_pf, f"Expected {key!r} in static part_fields for {yaml_name}; got {list(static_pf)}"
        _assert_valid_field_entry(static_pf[key], require_positive=True, is_modal=False)

    # Backward compatibility: toolhead_part_fields remains populated
    assert "toolhead_mount" in report.static_fea.toolhead_part_fields
    assert "nozzle_stem" in report.static_fea.toolhead_part_fields
    assert report.static_fea.toolhead_part_fields["toolhead_mount"] == static_pf["toolhead_mount"]
    assert report.static_fea.toolhead_part_fields["nozzle_stem"] == static_pf["nozzle_stem"]

    # JSON serialization includes part_fields and mode_part_fields
    static_dict = report.static_fea.to_dict()
    assert "part_fields" in static_dict
    assert set(expected_keys).issubset(static_dict["part_fields"].keys())

    assert len(report.modal_fea.modes) >= 1
    m0 = report.modal_fea.modes[0]
    for key in expected_keys:
        assert key in m0.mode_part_fields, f"Expected {key!r} in mode_part_fields for {yaml_name}"
        _assert_valid_field_entry(m0.mode_part_fields[key], require_positive=True, is_modal=True)

    # Backward compatibility: toolhead_mode_fields remains populated with normalized arrays
    assert "toolhead_mount" in m0.toolhead_mode_fields
    assert "nozzle_stem" in m0.toolhead_mode_fields
    assert len(m0.toolhead_mode_fields["toolhead_mount"]["strain_norm"]) >= 16

    modal_dict = report.modal_fea.to_dict()
    assert "mode_part_fields" in modal_dict["modes"][0]
    assert set(expected_keys).issubset(modal_dict["modes"][0]["mode_part_fields"].keys())


@pytest.mark.parametrize("yaml_name", ["corexy_voron_style.yaml", "voron_monolith_awd.yaml"])
def test_exact_6dof_rigid_body_invariance_printer_wide(yaml_name: str) -> None:
    """Every recovered part (including offset ports and rotated carriages) must yield exact zero flex & stress under 6-DOF rigid-body motion."""
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / yaml_name)
    asm = build_assembly(spec)
    sys = assemble_global_system(asm)

    u_rb = _build_global_rigid_body_displacement(
        asm,
        sys.K.shape[0],
        trans_m=np.array([2.5e-3, -1.8e-3, 9.0e-4], dtype=np.float64),
        rot_rad=np.array([0.015, -0.022, 0.018], dtype=np.float64),
    )

    pf = compute_printer_part_fields(asm, sys.node_idx, u_rb, normalized=False)
    assert len(pf) >= 5
    for key, entry in pf.items():
        _assert_valid_field_entry(entry, require_positive=False, is_modal=False)
        assert entry["max_flex_um"] == 0.0, f"{key} leaked rigid-body flex: {entry['max_flex_um']}"
        assert entry["max_stress_mpa"] == 0.0, f"{key} leaked rigid-body stress: {entry['max_stress_mpa']}"

    # Legacy wrapper must also be exact rigid-body invariant
    th_pf = compute_toolhead_part_fields(asm, sys.node_idx, u_rb, normalized=False)
    for key, entry in th_pf.items():
        assert entry["max_flex_um"] == 0.0
        assert entry["max_stress_mpa"] == 0.0


def test_rotated_and_offset_superelement_local_transform_and_compactness() -> None:
    """Verify arbitrary attached superelements transform q_b via rotation + port offset, and un-baked procedural parts stay omitted."""
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    spec.gantry.hardware_kit = "procedural"
    asm = build_assembly(spec)
    sys = assemble_global_system(asm)

    # Procedural XY-joints have no baked field_points_m and must be omitted for payload compactness
    u_zero = np.zeros(sys.K.shape[0], dtype=np.float64)
    pf_base = compute_printer_part_fields(asm, sys.node_idx, u_zero, normalized=False)
    assert "xy_joint_l" not in pf_base
    assert "xy_joint_r" not in pf_base

    # Attach a synthetic rotated CAD superelement using a real baked carriage recovery operator
    car_asset = load_superelement("carriages", "mgn12h")
    assert car_asset is not None and car_asset.subpart_recovery is not None

    angle = math.pi / 2.0
    r_z90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    synth_se = AttachedSuperelement(
        id="synthetic_rotated_bracket",
        category="carriages",
        asset_id="mgn12h",
        port_node_ids=("y_car_l", "x_beam_l"),
        K_cb=car_asset.K_bb,
        port_coords_cad_m=car_asset.port_coords_m,
        part_group_label="Synthetic Rotated Bracket",
        manifest={**car_asset.manifest, "rotation_z_rad": angle},
        subpart_recovery=car_asset.subpart_recovery,
    )
    asm.superelements.append(synth_se)

    # 1. Exact 6-DOF rigid-body motion must produce zero flex and zero stress despite rotation & port offset
    u_rb = _build_global_rigid_body_displacement(
        asm,
        sys.K.shape[0],
        trans_m=np.array([-1.0e-3, 3.0e-3, -2.0e-3]),
        rot_rad=np.array([0.02, 0.01, -0.03]),
    )
    pf_rb = compute_printer_part_fields(asm, sys.node_idx, u_rb, normalized=False)
    assert "synthetic_rotated_bracket" in pf_rb
    assert pf_rb["synthetic_rotated_bracket"]["max_flex_um"] == 0.0
    assert pf_rb["synthetic_rotated_bracket"]["max_stress_mpa"] == 0.0

    # 2. Pure elastic shear displacement at Port 1 in assembly +Y corresponds to local CAD +X under R_z(90 deg)
    u_el = np.zeros(sys.K.shape[0], dtype=np.float64)
    i1 = sys.node_idx["x_beam_l"]
    u_el[6 * i1 + 1] = 2.0e-5  # +20 um along assembly +Y
    pf_el = compute_printer_part_fields(asm, sys.node_idx, u_el, normalized=False)
    ent = pf_el["synthetic_rotated_bracket"]
    _assert_valid_field_entry(ent, require_positive=True, is_modal=False)

    # Confirm exact equivalence to directly passing R_z(90)^T @ [0, 20um, 0] = [20um, 0, 0] into recover_field
    q_expected = np.zeros(12, dtype=np.float64)
    q_expected[6:9] = r_z90.T @ np.array([0.0, 2.0e-5, 0.0])
    ref = car_asset.subpart_recovery.recover_field(q_expected)
    assert ref is not None
    assert ent["max_flex_um"] == pytest.approx(ref["max_flex_um"], rel=1e-6)
    assert ent["max_stress_mpa"] == pytest.approx(ref["max_stress_mpa"], rel=1e-6)
