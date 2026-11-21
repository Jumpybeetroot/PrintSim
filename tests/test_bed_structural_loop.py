"""Comprehensive tests for Nozzle-Relative-to-Bed Structural Loop Metrology (u_rel = u_nozzle - u_bed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from printsim.api.server import create_app
from printsim.fea import assemble_global_matrices, solve_modal_fea, solve_static_fea
from printsim.fea.solver import _solve_linear_system
from printsim.geometry import (
    AssemblyModel,
    StructuralMember,
    StructuralNode,
    build_assembly,
    generate_bom,
)
from printsim.schema import PrinterSpec, get_material, get_profile

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def _load(name: str) -> PrinterSpec:
    return PrinterSpec.from_yaml(EXAMPLES_DIR / f"{name}.yaml")


# ---------------------------------------------------------------------------
# 1. Structural Bed Node, Mass Accounting, Fallback, and BOM
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "preset_name",
    [
        "voron_trident_300",
        "voron_monolith_awd",
        "corexy_voron_style",
        "cartesian_bedslinger",
        "delta_racer",
    ],
)
def test_bed_node_mass_accounting_zero_double_counting_and_bom(preset_name: str) -> None:
    """Every printer architecture mounts bed_surface with exact bed mass, zero double-count, and clean BOM."""
    spec = _load(preset_name)
    asm = build_assembly(spec)
    nmap = asm.node_map

    assert asm.bed_node_id == "bed_surface"
    assert "bed_surface" in nmap
    bed_node = nmap["bed_surface"]
    assert bed_node.role == "bed"

    expected_bed_kg = spec.bed.mass_g * 1e-3
    assert bed_node.lumped_mass_kg == pytest.approx(expected_bed_kg, rel=1e-9)

    # Zero double-counting in total_mass_kg: bed mass is inside lumped_mass, NOT added twice
    lumped_sum = sum(n.lumped_mass_kg for n in asm.nodes)
    assert asm.total_mass_kg == pytest.approx(asm.frame_mass_kg + lumped_sum, rel=1e-9)

    # Virtual bed mount members have virtually zero mass & zero fill mass
    bed_members = [m for m in asm.members if m.role == "bed"]
    assert len(bed_members) >= 3
    assert sum(m.mass_kg(nmap) for m in bed_members) < 1e-4
    assert all(m.fill_mass_kg == 0.0 for m in bed_members)
    assert all(m.material.cost_per_kg_usd == 0.0 for m in bed_members)

    # Increasing spec.bed.mass_g by 1000 g increases total_mass_kg by EXACTLY 1.0 kg (not 2.0 kg)
    spec_heavy = spec.model_copy(deep=True)
    spec_heavy.bed.mass_g += 1000.0
    asm_heavy = build_assembly(spec_heavy)
    assert (asm_heavy.total_mass_kg - asm.total_mass_kg) == pytest.approx(1.0, rel=1e-9)

    # Geometry dict exposes bed_node_id
    gdict = asm.to_geometry_dict()
    assert gdict["bed_node_id"] == "bed_surface"

    # BOM excludes all virtual role="bed" struts
    bom = generate_bom(spec)
    assert all(c.role != "bed" for c in bom.cut_list)


def test_trident_bom_merges_split_columns_and_rear_members() -> None:
    """Trident column split (col_fl_bot/mid/top) and rear split merge into exact full-length BOM cuts."""
    spec = _load("voron_trident_300")
    asm = build_assembly(spec)
    member_ids = {m.id for m in asm.members}

    for tag in ("fl", "fr"):
        assert f"col_{tag}_bot" in member_ids
        assert f"col_{tag}_mid" in member_ids
        assert f"col_{tag}_top" in member_ids
        assert f"col_{tag}_low" not in member_ids

    assert "z_bed_fl" in asm.node_map
    assert "z_bed_fr" in asm.node_map
    assert "z_bed_r" in asm.node_map

    bom = generate_bom(spec)
    col_cuts = [
        c
        for c in bom.cut_list
        if c.role == "frame"
        and c.profile_id == spec.frame.profile
        and c.length_mm == pytest.approx(spec.frame.height_mm, abs=0.1)
    ]
    assert len(col_cuts) == 1
    assert col_cuts[0].quantity == 4
    assert all(c.role != "bed" for c in bom.cut_list)


def test_custom_minimal_truss_without_bed_falls_back_gracefully() -> None:
    """Custom trusses without bed_surface clear bed_node_id and match absolute nozzle metrology."""
    mat = get_material("aluminum_6063_t5")
    prof = get_profile("extrusion_2020")
    nodes = [
        StructuralNode("n0", np.array([0.0, 0.0, 0.0]), fixed=True, role="foot"),
        StructuralNode("n1", np.array([0.25, 0.0, 0.0]), role="frame"),
        StructuralNode("n2", np.array([0.50, 0.0, 0.0]), lumped_mass_kg=0.4, role="toolhead"),
    ]
    members = [
        StructuralMember("m0", "n0", "n1", prof, mat),
        StructuralMember("m1", "n1", "n2", prof, mat),
    ]
    asm = AssemblyModel(spec=PrinterSpec(), nodes=nodes, members=members, toolhead_node_id="n2")
    assert asm.bed_node_id == ""
    assert asm.to_geometry_dict()["bed_node_id"] == ""

    st = solve_static_fea(asm)
    mo = solve_modal_fea(asm)
    assert st.bed_deflection_um == 0.0
    assert st.combined_max_deflection_um == pytest.approx(st.nozzle_absolute_deflection_um, rel=1e-9)
    assert mo.fundamental_freq_hz > 10.0


# ---------------------------------------------------------------------------
# 2. Exact Structural-Loop Identities (u_rel = u_nozzle - u_bed, +-1N Loop)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "preset_name",
    ["voron_trident_300", "corexy_voron_style", "cartesian_bedslinger", "delta_racer"],
)
def test_exact_relative_displacement_and_slocum_loop_stiffness_identity(preset_name: str) -> None:
    """Verify combined_max_deflection_um == norm(u_noz - u_bed) and k == 1 / |u_noz - u_bed| under +-1N."""
    spec = _load(preset_name)
    asm = build_assembly(spec)
    st = solve_static_fea(asm)

    u_noz_um = np.array(st.nodal_displacements_mm["nozzle_tip"], dtype=float) * 1e3
    u_bed_um = np.array(st.nodal_displacements_mm["bed_surface"], dtype=float) * 1e3
    u_th_um = np.array(st.nodal_displacements_mm["toolhead"], dtype=float) * 1e3

    expected_rel_um = float(np.linalg.norm(u_noz_um - u_bed_um))
    expected_abs_um = float(np.linalg.norm(u_noz_um))
    expected_bed_um = float(np.linalg.norm(u_bed_um))
    expected_th_rel_um = float(np.linalg.norm(u_th_um - u_bed_um))

    assert st.combined_max_deflection_um == pytest.approx(expected_rel_um, abs=2e-3)
    assert st.nozzle_absolute_deflection_um == pytest.approx(expected_abs_um, abs=2e-3)
    assert st.bed_deflection_um == pytest.approx(expected_bed_um, abs=2e-3)
    assert st.beam_center_deflection_um == pytest.approx(expected_th_rel_um, abs=2e-3)

    # Serialized dict includes new telemetry fields
    st_dict = st.to_dict()
    assert st_dict["nozzle_absolute_deflection_um"] == pytest.approx(expected_abs_um, abs=0.02)
    assert st_dict["bed_deflection_um"] == pytest.approx(expected_bed_um, abs=0.02)

    # Exact manual +-1 N unit structural-loop stiffness check in X, Y, Z
    K, _, _, node_idx, free_dofs = assemble_global_matrices(asm)
    noz_dof = 6 * node_idx["nozzle_tip"]
    bed_dof = 6 * node_idx["bed_surface"]
    reported_k = [
        st.toolhead_stiffness_x_n_per_mm,
        st.toolhead_stiffness_y_n_per_mm,
        st.toolhead_stiffness_z_n_per_mm,
    ]
    for axis in range(3):
        f_loop = np.zeros(K.shape[0], dtype=float)
        f_loop[noz_dof + axis] = +1.0
        f_loop[bed_dof + axis] = -1.0
        u_loop = _solve_linear_system(K, f_loop, free_dofs)
        rel_mm = abs(float(u_loop[noz_dof + axis] - u_loop[bed_dof + axis])) * 1e3
        expected_k = 1.0 / max(rel_mm, 1e-9)
        assert reported_k[axis] == pytest.approx(expected_k, rel=1e-9)


# ---------------------------------------------------------------------------
# 3. Architecture-Specific Structural Loop Physics
# ---------------------------------------------------------------------------


def test_trident_height_dependent_co_sway_and_plate_thickness() -> None:
    """On unpaneled Trident, first-layer co-sway cancels frame sway; tall prints grow relative error."""
    spec = _load("voron_trident_300")
    spec.frame.bracing = "none"
    spec.frame.enclosed_panels = False
    spec.frame.panel_mounting = "none"

    asm_low = build_assembly(spec, toolhead_rel_pos=(0.5, 0.5, 0.05))
    asm_mid = build_assembly(spec, toolhead_rel_pos=(0.5, 0.5, 0.50))
    asm_tall = build_assembly(spec, toolhead_rel_pos=(0.5, 0.5, 0.95))

    # Bed moves DOWN as print height w_rel increases
    assert asm_low.bed_center_m[2] > asm_mid.bed_center_m[2] > asm_tall.bed_center_m[2]
    assert asm_low.bed_center_m[2] > 0.35
    assert asm_tall.bed_center_m[2] < 0.20

    st_low = solve_static_fea(asm_low)
    st_mid = solve_static_fea(asm_mid)
    st_tall = solve_static_fea(asm_tall)

    # First layer (w_rel=0.05): bed & gantry sway together -> relative deflection << absolute nozzle deflection
    assert st_low.bed_deflection_um > 10.0
    assert st_low.combined_max_deflection_um < st_low.nozzle_absolute_deflection_um * 0.65

    # Tall print (w_rel=0.95): bed near base -> larger relative deflection & lower loop stiffness
    assert st_low.combined_max_deflection_um < st_mid.combined_max_deflection_um < st_tall.combined_max_deflection_um
    assert st_low.toolhead_stiffness_x_n_per_mm > st_mid.toolhead_stiffness_x_n_per_mm > st_tall.toolhead_stiffness_x_n_per_mm
    assert st_low.toolhead_stiffness_y_n_per_mm > st_mid.toolhead_stiffness_y_n_per_mm > st_tall.toolhead_stiffness_y_n_per_mm

    # Modal: bed Z bounce does NOT hijack shaper_x / shaper_y or fundamental_freq_hz at w_rel=0.50
    mo_mid = solve_modal_fea(asm_mid)
    assert mo_mid.shaper_x is not None and mo_mid.shaper_y is not None
    assert 30.0 < mo_mid.fundamental_freq_hz < 60.0
    assert 30.0 < mo_mid.shaper_x.structural_freq_hz < 65.0
    assert 30.0 < mo_mid.shaper_y.structural_freq_hz < 65.0

    # Thicker MIC-6 plate (4 -> 8 -> 12 mm) raises Z structural-loop stiffness
    st_z = []
    for t_mm in (4.0, 8.0, 12.0):
        s_t = spec.model_copy(deep=True)
        s_t.bed.plate_thickness_mm = t_mm
        st_z.append(solve_static_fea(build_assembly(s_t)).toolhead_stiffness_z_n_per_mm)
    assert st_z[0] < st_z[1] < st_z[2]


@pytest.mark.parametrize("preset_name", ["corexy_voron_style", "delta_racer"])
def test_fixed_base_bed_has_tiny_deflection_and_matches_absolute(preset_name: str) -> None:
    """Voron 2.4 (belted_quad_z) and Delta anchor bed to fixed feet: bed_deflection_um < 1.0 um."""
    spec = _load(preset_name)
    asm = build_assembly(spec)
    st = solve_static_fea(asm)
    mo = solve_modal_fea(asm)

    assert st.bed_deflection_um < 1.0
    assert st.combined_max_deflection_um == pytest.approx(st.nozzle_absolute_deflection_um, rel=0.05)
    assert mo.fundamental_freq_hz > 60.0


def test_cartesian_bedslinger_y_accel_flexes_bed_relative_to_nozzle() -> None:
    """Bedslinger Y acceleration drives bed in -Y opposite to nozzle (+Y), increasing relative error."""
    spec = _load("cartesian_bedslinger")
    asm = build_assembly(spec)
    st = solve_static_fea(asm)
    mo = solve_modal_fea(asm)

    # Bed has real Y deflection under Y inertia
    assert st.bed_deflection_um > 2.0
    noz_y_mm = st.nodal_displacements_mm["nozzle_tip"][1]
    bed_y_mm = st.nodal_displacements_mm["bed_surface"][1]
    # Nozzle and bed deflect in opposite Y directions under relative inertial reaction
    assert noz_y_mm * bed_y_mm < 0.0
    assert st.combined_max_deflection_um > st.nozzle_absolute_deflection_um

    # Bed is much stiffer in X/Z than in Y carriage direction
    K, _, _, node_idx, free_dofs = assemble_global_matrices(asm)
    bd = 6 * node_idx["bed_surface"]
    bed_k = []
    for axis in range(3):
        f = np.zeros(K.shape[0])
        f[bd + axis] = 1.0
        u = _solve_linear_system(K, f, free_dofs)
        bed_k.append(1.0 / max(abs(float(u[bd + axis])) * 1e3, 1e-9))
    assert bed_k[0] > bed_k[1] * 10.0  # X >> Y
    assert bed_k[2] > bed_k[1] * 2.0   # Z > Y

    # Heavier bed increases bed deflection and relative Y accel deflection
    spec_heavy = spec.model_copy(deep=True)
    spec_heavy.bed.mass_g = 2600.0
    st_heavy = solve_static_fea(build_assembly(spec_heavy))
    assert st_heavy.bed_deflection_um > st.bed_deflection_um * 1.4
    assert st_heavy.accel_deflection_y_um > st.accel_deflection_y_um

    # Fundamental mode remains the portal Y mode (~104 Hz), not hijacked by bed bounce
    assert mo.fundamental_freq_hz == pytest.approx(103.9, abs=3.0)
    assert any(
        abs(m.nodal_vectors["bed_surface"][1]) > 0.7 for m in mo.modes
    )


def test_api_simulate_exposes_bed_structural_loop_fields() -> None:
    """FastAPI /api/simulate exposes bed_node_id, nozzle_absolute_deflection_um, and bed_deflection_um."""
    spec = _load("voron_trident_300")
    client = TestClient(create_app())
    resp = client.post("/api/simulate", json={"spec": spec.model_dump(mode="json")})
    assert resp.status_code == 200
    data = resp.json()
    assert data["geometry"]["bed_node_id"] == "bed_surface"
    assert "bed_surface" in data["static_fea"]["nodal_displacements_mm"]
    assert data["static_fea"]["nozzle_absolute_deflection_um"] > data["static_fea"]["combined_max_deflection_um"]
    assert data["static_fea"]["bed_deflection_um"] > 5.0
    bn = data["modal_fea"]["accel_bottleneck"]
    assert bn is not None
    assert sum(bn["shares_pct"].values()) == pytest.approx(100.0, abs=0.2)


def test_accel_bottleneck_compliance_shares_sum_to_100_and_track_upgrades() -> None:
    """Series compliance shares (frame + belt + motor + bracket) sum to 100% and shift accurately."""
    # 1. Stock unpaneled Trident -> Frame is #1 bottleneck
    s_stock = _load("voron_trident_300")
    mo_stock = solve_modal_fea(build_assembly(s_stock))
    bn_stock = mo_stock.accel_bottleneck
    assert bn_stock is not None
    assert sum(bn_stock["shares_pct"].values()) == pytest.approx(100.0, abs=0.2)
    assert bn_stock["primary_source"] == "frame"

    # 2. Bolted 4mm Al panels on 2WD 6mm Trident -> Belt Stretch becomes #1 bottleneck
    s_pan = s_stock.model_copy(deep=True)
    s_pan.frame.panel_mounting = "bolted_structural"
    s_pan.frame.panel_material = "aluminum_5052_sheet"
    s_pan.frame.panel_thickness_mm = 4.0
    mo_pan = solve_modal_fea(build_assembly(s_pan))
    bn_pan = mo_pan.accel_bottleneck
    assert bn_pan is not None
    assert sum(bn_pan["shares_pct"].values()) == pytest.approx(100.0, abs=0.2)
    assert bn_pan["primary_source"] == "belt"
    assert bn_pan["shares_pct"]["frame"] < bn_stock["shares_pct"]["frame"]

    # 3. Monolith AWD with ABS brackets -> Bracket Flex is #1; upgrading to CNC 6061 drops bracket share < 15%
    s_abs = _load("voron_monolith_awd")
    s_abs.gantry.bracket_material = "abs_printed"
    mo_abs = solve_modal_fea(build_assembly(s_abs))
    assert mo_abs.accel_bottleneck is not None
    assert mo_abs.accel_bottleneck["primary_source"] == "bracket"

    s_cnc = _load("voron_monolith_awd")
    mo_cnc = solve_modal_fea(build_assembly(s_cnc))
    assert mo_cnc.accel_bottleneck is not None
    assert sum(mo_cnc.accel_bottleneck["shares_pct"].values()) == pytest.approx(100.0, abs=0.2)
    assert mo_cnc.accel_bottleneck["shares_pct"]["bracket"] < 15.0
    assert mo_cnc.accel_bottleneck["primary_source"] != "bracket"
