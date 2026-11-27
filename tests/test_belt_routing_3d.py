"""Analytical, geometric, and parametric tests for CAL-13 3D CoreXY belt routing graph."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from printsim.fea.mechatronics import corexy_belt_span_stiffness_matrix
from printsim.geometry import build_assembly
from printsim.geometry._builder_gantry import (
    CoreXYBeltGraph3D,
    build_corexy_belt_routing_graph_3d,
)
from printsim.kinematics.solvers import (
    analyze_kinematics,
    compute_branch_idler_stiffness_n_m,
    compute_drive_stiffness_details_n_per_m,
    compute_per_idler_stiffness_n_m,
)
from printsim.schema import BeltSpec, PrinterSpec

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def _load(name: str) -> PrinterSpec:
    return PrinterSpec.from_yaml(EXAMPLES_DIR / name)


def test_3d_belt_graph_segments_and_cad_station_coordinates() -> None:
    """Every segment matches exact 3D Euclidean norm, unit direction, and CAD/procedural provenance."""
    for name, expected_cad in (
        ("voron_trident_300.yaml", True),
        ("siboor_voron_trident_awd_300.yaml", True),
        ("voron_monolith_awd.yaml", True),
        ("vzbot_330_awd.yaml", True),
    ):
        spec = _load(name)
        asm = build_assembly(spec)
        graph = build_corexy_belt_routing_graph_3d(asm)
        assert isinstance(graph, CoreXYBeltGraph3D)
        assert graph.uses_cad_stations is expected_cad
        assert set(graph.branches.keys()) == {"A1", "A2", "B1", "B2"}

        for bid, br in graph.branches.items():
            assert len(br.segments) == len(br.node_ids) - 1
            seg_sum = 0.0
            for seg in br.segments:
                p0 = np.asarray(seg.start_m, dtype=float)
                p1 = np.asarray(seg.end_m, dtype=float)
                dist = float(np.linalg.norm(p1 - p0))
                assert seg.length_m == pytest.approx(dist, rel=1e-12, abs=1e-12)
                assert np.linalg.norm(seg.direction_unit) == pytest.approx(1.0, abs=1e-9)
                seg_sum += seg.length_m
            assert br.total_length_m >= seg_sum - 1e-12
            assert br.drive_node_id == br.node_ids[-1]
            assert graph.pulleys[br.drive_node_id].role == "drive"

        # JSON-serializable telemetry dict with explicit unit keys
        payload = graph.to_dict()
        dumped = json.dumps(payload)
        assert "z_upper_plane_mm" in dumped
        assert "effective_span_ratio_vs_stock" in dumped

    # VzBot rear CAD mount has -2.0 mm Z offset -> rear drive pulley carries real 3D Z shift
    vz_graph = build_corexy_belt_routing_graph_3d(build_assembly(_load("vzbot_330_awd.yaml")))
    drv_rr = vz_graph.pulleys[vz_graph.branches["A1"].drive_node_id]
    assert drv_rr.position_m[2] == pytest.approx(vz_graph.z_upper_plane_m - 0.002, abs=1e-5)

    # Procedural build falls back cleanly without CAD corner nodes
    proc_spec = _load("voron_trident_300.yaml")
    proc_spec.gantry.hardware_kit = "procedural"
    proc_graph = build_corexy_belt_routing_graph_3d(build_assembly(proc_spec))
    assert proc_graph.uses_cad_stations is False
    assert all(p.source == "parametric" for p in proc_graph.pulleys.values())


def test_stacked_upper_lower_belt_planes_and_z_lever_arms() -> None:
    """Upper/lower CoreXY belt planes track belt width + 3 mm flange gap and toolhead Z offsets."""
    base = _load("voron_monolith_awd.yaml")
    for btype, expected_dz_mm in (("gt2_6mm", 9.0), ("gt2_9mm", 12.0), ("gt2_10mm", 13.0)):
        sp = base.model_copy(deep=True)
        sp.gantry.belt.type = btype  # type: ignore[assignment]
        sp.toolhead.belt_plane_z_offset_mm = -2.0
        sp.toolhead.cog_offset_mm = (0.0, -12.0, 10.0)
        graph = build_corexy_belt_routing_graph_3d(build_assembly(sp))

        assert graph.dz_belt_planes_m * 1e3 == pytest.approx(expected_dz_mm, abs=1e-6)
        assert (graph.z_upper_plane_m - graph.z_lower_plane_m) * 1e3 == pytest.approx(
            expected_dz_mm, abs=1e-6
        )
        assert graph.upper_plane_z_lever_mm == pytest.approx(-2.0 + 0.5 * expected_dz_mm, abs=1e-4)
        assert graph.lower_plane_z_lever_mm == pytest.approx(-2.0 - 0.5 * expected_dz_mm, abs=1e-4)
        assert graph.belt_to_cog_z_lever_mm == pytest.approx(12.0, abs=1e-4)
        assert graph.stacked_belt_twist_moment_nm_per_n == pytest.approx(
            expected_dz_mm * 1e-3, rel=1e-9
        )

        # Plane A nodes live at z_upper_plane_m; Plane B nodes live at z_lower_plane_m
        for p in graph.pulleys.values():
            expected_z = graph.z_upper_plane_m if p.belt_plane == "A" else graph.z_lower_plane_m
            assert p.position_m[2] == pytest.approx(expected_z, abs=1e-6)


def test_non_scalar_frame_size_upright_and_aspect_ratio_dependence() -> None:
    """3D coordinates replace legacy 0.86x scalar: span savings depend on frame size, upright, & aspect."""
    # 1. Small 230 mm M0 vs 460 mm 300-build vs large 560 mm 350-build: fixed S-wrap removal saves
    #    a larger fraction of total active belt on smaller frames than on larger frames.
    g_m0 = build_corexy_belt_routing_graph_3d(build_assembly(_load("monolith_zero_m0.yaml")))
    g_300 = build_corexy_belt_routing_graph_3d(build_assembly(_load("voron_monolith_awd.yaml")))

    sp_350 = _load("voron_monolith_awd.yaml")
    sp_350.frame.width_mm = 560.0
    sp_350.frame.depth_mm = 560.0
    sp_350.kinematics.build_volume_mm = (400.0, 400.0, 300.0)
    g_350 = build_corexy_belt_routing_graph_3d(build_assembly(sp_350))

    assert g_m0.effective_span_ratio_vs_stock < g_300.effective_span_ratio_vs_stock < g_350.effective_span_ratio_vs_stock < 1.0
    # Confirm none of them is pinned to the legacy hardcoded 0.86 constant
    assert abs(g_m0.effective_span_ratio_vs_stock - 0.86) > 0.05

    # 2. Wider 4040 uprights shift pulley stations inboard vs 2020 uprights on the same 460x460 outer frame
    sp_2020 = _load("voron_monolith_awd.yaml")
    sp_4040 = sp_2020.model_copy(deep=True)
    sp_4040.frame.upright_profile = "extrusion_4040"
    g_2020 = build_corexy_belt_routing_graph_3d(build_assembly(sp_2020))
    g_4040 = build_corexy_belt_routing_graph_3d(build_assembly(sp_4040))
    assert g_4040.total_active_belt_length_m < g_2020.total_active_belt_length_m

    # 3. Rectangular aspect ratio (560x360) yields distinct X-dominant vs Y-dominant 2WD branch spans
    sp_rect = _load("voron_trident_300.yaml")
    sp_rect.frame.width_mm = 560.0
    sp_rect.frame.depth_mm = 360.0
    sp_rect.kinematics.build_volume_mm = (400.0, 200.0, 250.0)
    g_rect = build_corexy_belt_routing_graph_3d(build_assembly(sp_rect))
    # Branch A2 crosses the long rear X bar (~560 mm), whereas A1 runs along the short Y side (~360 mm)
    assert g_rect.branches["A2"].total_length_m > g_rect.branches["A1"].total_length_m + 0.12


def test_per_idler_cantilever_vs_double_shear_vs_live_shaft_and_wrap_projection() -> None:
    """Verify Euler-Bernoulli cantilever overhang, live-shaft vs bolt stiffness, and Castigliano wrap math."""
    belt_9 = BeltSpec(type="gt2_9mm", core="fiberglass", pretension_n=60.0)
    e_al = 68.9e9
    e_abs = 2.1e9

    # Analytical stiffness ordering on aluminum housing
    r_single_lo = compute_per_idler_stiffness_n_m(belt_9, "single_shear_bolt", e_al, z_overhang_extra_m=0.0)
    r_single_hi = compute_per_idler_stiffness_n_m(belt_9, "single_shear_bolt", e_al, z_overhang_extra_m=0.012)
    r_ds_bolt = compute_per_idler_stiffness_n_m(belt_9, "double_shear_bolt", e_al, z_overhang_extra_m=0.006)
    r_live = compute_per_idler_stiffness_n_m(belt_9, "live_shaft", e_al, axial_preload_active=False)
    r_live_pre = compute_per_idler_stiffness_n_m(belt_9, "live_shaft", e_al, axial_preload_active=True)

    assert r_single_hi["k_shaft_n_m"] < r_single_lo["k_shaft_n_m"]
    assert r_single_hi["shaft_deflection_um"] > r_single_lo["shaft_deflection_um"] * 2.0
    assert r_single_lo["k_idler_n_m"] < r_ds_bolt["k_idler_n_m"] < r_live["k_idler_n_m"] < r_live_pre["k_idler_n_m"]

    # Polymer ABS root compliance softens both single-shear and double-shear bolts vs 6061 aluminum
    r_abs_single = compute_per_idler_stiffness_n_m(belt_9, "single_shear_bolt", e_abs, z_overhang_extra_m=0.0)
    r_abs_ds = compute_per_idler_stiffness_n_m(belt_9, "double_shear_bolt", e_abs, z_overhang_extra_m=0.006)
    assert r_abs_single["k_shaft_n_m"] < r_single_lo["k_shaft_n_m"] * 0.78
    assert r_abs_ds["k_shaft_n_m"] < r_ds_bolt["k_shaft_n_m"] * 0.88

    # Thermal CTE press-fit clearance degrades bearing radial stiffness
    r_hot_loose = compute_per_idler_stiffness_n_m(belt_9, "live_shaft", e_al, thermal_clearance_um=5.0)
    assert r_hot_loose["k_bearing_n_m"] < r_live["k_bearing_n_m"]

    # Castigliano virtual work check: 180-deg wrap idler (alpha^2 = 4) has ~2x belt-line compliance
    # vs 90-deg wrap idler (alpha^2 = 2) when evaluated on an otherwise identical single-idler branch.
    class _StubNode:
        def __init__(self, wrap_deg: float) -> None:
            self.station = "fl"
            self.shaft_architecture = "double_shear_bolt"
            self.z_overhang_extra_m = 0.0
            self.wrap_angle_deg = wrap_deg

    class _StubBranch:
        idler_node_ids = ("i0",)
        drive_node_id = "d0"

    b90 = compute_branch_idler_stiffness_n_m(
        _StubBranch(), {"i0": _StubNode(90.0)}, belt_9, e_al, e_al, include_drive_pulley=False
    )
    b180 = compute_branch_idler_stiffness_n_m(
        _StubBranch(), {"i0": _StubNode(180.0)}, belt_9, e_al, e_al, include_drive_pulley=False
    )
    # At 180 deg, alpha^2 doubles (4 vs 2) while higher radial preload stiffens Hertzian bearing slightly (~1.12x)
    ratio_k = b90["k_branch_idler_n_m"] / b180["k_branch_idler_n_m"]
    assert 1.65 < ratio_k < 2.05


def test_canonical_solvers_mechatronics_parity_and_non_corexy_compat() -> None:
    """solvers.py and mechatronics.py maintain exact parity across routings; non-CoreXY unchanged."""
    for name in ("voron_trident_300.yaml", "voron_monolith_awd.yaml", "siboor_voron_trident_awd_300.yaml"):
        spec = _load(name)
        asm = build_assembly(spec)
        det = compute_drive_stiffness_details_n_per_m(asm, "X", u_rel=0.5, v_rel=0.5)
        k_cart, k_a, k_b = corexy_belt_span_stiffness_matrix(asm, u_rel=0.5, v_rel=0.5)
        k_belt = float(det["k_belt_subsystem_n_m"])
        k_bkt = float(det["k_bracket_n_m"])
        expected_k_xx = 1.0 / (1.0 / k_belt + 1.0 / k_bkt)
        assert float(k_cart[0, 0]) == pytest.approx(expected_k_xx, rel=1e-4)
        assert float(k_cart[0, 0]) == pytest.approx(k_a + k_b, rel=1e-9)
        assert det["belt_plane_dz_mm"] == pytest.approx(spec.gantry.belt.width_mm + 3.0, abs=1e-6)

        # Off-center parity: K_cart[0, 0] == k_A + k_B and non-zero cross-coupling K_cart[0, 1] == k_A - k_B
        k_cart_off, k_a_off, k_b_off = corexy_belt_span_stiffness_matrix(asm, u_rel=0.15, v_rel=0.85)
        assert float(k_cart_off[0, 0]) == pytest.approx(k_a_off + k_b_off, rel=1e-9)
        assert float(k_cart_off[0, 1]) == pytest.approx(k_a_off - k_b_off, rel=1e-9)

    # Non-CoreXY architectures do not invoke CoreXY 3D belt graph and remain valid
    for name in ("cartesian_bedslinger.yaml", "flsun_v400_delta.yaml"):
        spec = _load(name)
        asm = build_assembly(spec)
        det = compute_drive_stiffness_details_n_per_m(asm, "X")
        assert "belt_graph_3d" not in det
        kin = analyze_kinematics(asm)
        assert kin.belt_stiffness_n_per_mm > 0.0
        assert kin.drive_stiffness_n_per_mm > 0.0
