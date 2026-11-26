"""Tests for non-blocking open-center Top Braces and open-door Front Braces."""

from __future__ import annotations

import io
from pathlib import Path

import ezdxf
import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.linalg import eigvalsh

from printsim.api.server import create_app
from printsim.engine import evaluate_printer
from printsim.fea import (
    assemble_global_matrices,
    compute_structural_sensitivities,
    solve_modal_fea,
    solve_static_fea,
)
from printsim.fea.modal import _classify_modal_group
from printsim.fea.solver import classify_part_group
from printsim.geometry import (
    build_assembly,
    export_panels_dxf_bytes,
    front_brace_dimensions_mm,
    generate_bom,
    list_dxf_exportable_parts,
    list_dxf_parts,
    top_brace_dimensions_mm,
)
from printsim.schema import (
    FRONT_BRACE_CATALOG,
    TOP_BRACE_CATALOG,
    PrinterSpec,
)

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def _open_top_bolted_al_trident() -> PrinterSpec:
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    spec.frame.enclosed_panels = True
    spec.frame.panel_mounting = "bolted_structural"
    spec.frame.panel_material = "aluminum_5052_sheet"
    spec.frame.panel_thickness_mm = 5.0
    spec.frame.top_panel = False
    return spec


def _assert_spd(spec: PrinterSpec) -> None:
    asm = build_assembly(spec)
    K, M, _, _, free = assemble_global_matrices(asm)
    K_ff = K[np.ix_(free, free)]
    M_ff = M[np.ix_(free, free)]
    assert np.allclose(K_ff, K_ff.T, atol=1e-7)
    assert np.allclose(M_ff, M_ff.T, atol=1e-12)
    assert float(np.min(eigvalsh(K_ff))) > 0.0
    assert float(np.min(eigvalsh(M_ff))) > 0.0


def test_defaults_preserve_exact_backward_compatibility() -> None:
    """Default top_brace='none' and front_brace='none' preserve exact baseline f1, Kx, mass, and cost."""
    spec_default = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    assert spec_default.frame.top_brace == "none"
    assert spec_default.frame.front_brace == "none"

    spec_explicit = spec_default.model_copy(deep=True)
    spec_explicit.frame.top_brace = "none"
    spec_explicit.frame.front_brace = "none"

    asm_def = build_assembly(spec_default)
    asm_exp = build_assembly(spec_explicit)
    gd_def = asm_def.to_geometry_dict()
    gd_exp = asm_exp.to_geometry_dict()
    assert gd_def == gd_exp
    assert gd_def["top_brace"]["active"] is False
    assert gd_def["top_brace"]["mass_kg"] == 0.0
    assert gd_def["top_brace"]["cost_usd"] == 0.0
    assert gd_def["front_brace"]["active"] is False
    assert gd_def["front_brace"]["mass_kg"] == 0.0
    assert gd_def["front_brace"]["cost_usd"] == 0.0

    st_def = solve_static_fea(asm_def)
    st_exp = solve_static_fea(asm_exp)
    mo_def = solve_modal_fea(asm_def)
    mo_exp = solve_modal_fea(asm_exp)
    assert st_def.toolhead_stiffness_x_n_per_mm == pytest.approx(
        st_exp.toolhead_stiffness_x_n_per_mm, rel=1e-12
    )
    assert mo_def.fundamental_freq_hz == pytest.approx(mo_exp.fundamental_freq_hz, rel=1e-12)
    assert asm_def.total_mass_kg == pytest.approx(asm_exp.total_mass_kg, rel=1e-12)
    assert asm_def.estimated_material_cost_usd == pytest.approx(
        asm_exp.estimated_material_cost_usd, rel=1e-12
    )


@pytest.mark.parametrize(
    "top_brace",
    ["cnc_halo_ring", "rear_v_and_front_corners", "corner_45_bars"],
)
def test_all_top_brace_variants_stiffen_bolted_al_trident_and_preserve_open_cutout(
    top_brace: str,
) -> None:
    """All 3 top_brace variants build SPD matrices, boost f1 and Kx, keep open center, and bill accurately."""
    base_spec = _open_top_bolted_al_trident()
    asm0 = build_assembly(base_spec)
    st0 = solve_static_fea(asm0)
    mo0 = solve_modal_fea(asm0)
    bom0 = generate_bom(base_spec)

    spec = base_spec.model_copy(deep=True)
    spec.frame.top_brace = top_brace  # type: ignore[assignment]
    _assert_spd(spec)

    asm = build_assembly(spec)
    st = solve_static_fea(asm)
    mo = solve_modal_fea(asm)
    sen = compute_structural_sensitivities(asm, static_res=st, modal_res=mo)
    bom = generate_bom(spec)
    gd = asm.to_geometry_dict()

    assert mo.fundamental_freq_hz > mo0.fundamental_freq_hz + 5.0
    assert st.toolhead_stiffness_x_n_per_mm > st0.toolhead_stiffness_x_n_per_mm + 0.3
    assert st.accel_deflection_x_um < st0.accel_deflection_x_um

    tb_meta = gd["top_brace"]
    assert set(tb_meta.keys()) == {
        "id",
        "active",
        "label",
        "material_name",
        "thickness_mm",
        "flange_width_mm",
        "cutout_w_mm",
        "cutout_d_mm",
        "mass_kg",
        "cost_usd",
    }
    assert tb_meta["id"] == top_brace
    assert tb_meta["active"] is True
    assert tb_meta["thickness_mm"] == pytest.approx(5.0)
    assert tb_meta["mass_kg"] > 0.1
    assert tb_meta["cost_usd"] == pytest.approx(TOP_BRACE_CATALOG[top_brace]["cost_usd"])

    if top_brace == "cnc_halo_ring":
        top_shells = [s for s in asm.shell_elements if s.bay == "top_brace"]
        assert len(top_shells) == 12
        nmap = asm.node_map
        top_nodes = [n for n in asm.nodes if n.id.startswith("pan_top_brace_")]
        assert len(top_nodes) == 24
        assert all(n.role == "panel" for n in top_nodes)
        used_ids = {nid for sh in top_shells for nid in sh.node_ids}
        assert {n.id for n in top_nodes} == used_ids

        w_m = spec.frame.width_mm * 1e-3
        d_m = spec.frame.depth_mm * 1e-3
        cw_m = tb_meta["cutout_w_mm"] * 1e-3
        cd_m = tb_meta["cutout_d_mm"] * 1e-3
        x0, x1 = 0.5 * (w_m - cw_m), 0.5 * (w_m + cw_m)
        y0, y1 = 0.5 * (d_m - cd_m), 0.5 * (d_m + cd_m)
        for sh in top_shells:
            c = np.mean([nmap[nid].position_m for nid in sh.node_ids], axis=0)
            assert not (x0 < c[0] < x1 and y0 < c[1] < y1)
        assert "top_brace" in sen.panel_bay_sensitivities
    else:
        brace_bars = [m for m in asm.members if m.id.startswith("brace_top_open_")]
        assert len(brace_bars) == 4
        assert all(m.role == "brace" and not m.is_truss for m in brace_bars)
        assert all(m.profile.id == "al_flat_bar_25x5" for m in brace_bars)
        assert all(m.material.id == "aluminum_6061_t6" for m in brace_bars)

    assert (asm.total_mass_kg - asm0.total_mass_kg) == pytest.approx(asm.top_brace_mass_kg, abs=1e-4)
    assert (asm.estimated_material_cost_usd - asm0.estimated_material_cost_usd) == pytest.approx(
        asm.top_brace_cost_usd, abs=0.05
    )
    assert (bom.total_mass_kg - bom0.total_mass_kg) == pytest.approx(asm.top_brace_mass_kg, abs=2e-3)
    assert (bom.total_cost_usd - bom0.total_cost_usd) == pytest.approx(asm.top_brace_cost_usd, abs=0.05)
    assert any(item.sku.startswith("FRM-TOP-") for item in bom.items)


@pytest.mark.parametrize(
    "front_brace",
    ["cnc_portal_frame", "header_and_skirt", "corner_45_bars"],
)
def test_all_front_brace_variants_stiffen_bolted_al_trident_and_preserve_open_door(
    front_brace: str,
) -> None:
    """All 3 front_brace variants build SPD matrices, boost f1 and Kx, keep door open, and bill accurately."""
    base_spec = _open_top_bolted_al_trident()
    asm0 = build_assembly(base_spec)
    st0 = solve_static_fea(asm0)
    mo0 = solve_modal_fea(asm0)
    bom0 = generate_bom(base_spec)

    spec = base_spec.model_copy(deep=True)
    spec.frame.front_brace = front_brace  # type: ignore[assignment]
    _assert_spd(spec)

    asm = build_assembly(spec)
    st = solve_static_fea(asm)
    mo = solve_modal_fea(asm)
    sen = compute_structural_sensitivities(asm, static_res=st, modal_res=mo)
    bom = generate_bom(spec)
    gd = asm.to_geometry_dict()

    assert mo.fundamental_freq_hz > mo0.fundamental_freq_hz + 2.0
    assert st.toolhead_stiffness_x_n_per_mm > st0.toolhead_stiffness_x_n_per_mm + 0.3
    assert st.accel_deflection_x_um < st0.accel_deflection_x_um

    fb_meta = gd["front_brace"]
    assert set(fb_meta.keys()) == {
        "id",
        "active",
        "label",
        "material_name",
        "thickness_mm",
        "flange_width_mm",
        "header_height_mm",
        "skirt_height_mm",
        "cutout_w_mm",
        "cutout_h_mm",
        "mass_kg",
        "cost_usd",
    }
    assert fb_meta["id"] == front_brace
    assert fb_meta["active"] is True
    assert fb_meta["thickness_mm"] == pytest.approx(5.0)
    assert fb_meta["mass_kg"] > 0.1
    assert fb_meta["cost_usd"] == pytest.approx(FRONT_BRACE_CATALOG[front_brace]["cost_usd"])

    from printsim.geometry.builder import corexy_frame_centerlines_m

    nmap = asm.node_map
    w_m, _, h_m = corexy_frame_centerlines_m(spec)
    if front_brace == "cnc_portal_frame":
        front_shells = [s for s in asm.shell_elements if s.bay == "front_brace"]
        assert len(front_shells) == 12
        front_nodes = [n for n in asm.nodes if n.id.startswith("pan_front_brace_")]
        assert len(front_nodes) == 24
        assert all(n.role == "panel" for n in front_nodes)
        cw_m = fb_meta["cutout_w_mm"] * 1e-3
        ch_m = fb_meta["cutout_h_mm"] * 1e-3
        x0, x1 = 0.5 * (w_m - cw_m), 0.5 * (w_m + cw_m)
        z0, z1 = 0.5 * (h_m - ch_m), 0.5 * (h_m + ch_m)
        for sh in front_shells:
            c = np.mean([nmap[nid].position_m for nid in sh.node_ids], axis=0)
            assert not (x0 < c[0] < x1 and z0 < c[2] < z1)
        assert "front_brace" in sen.panel_bay_sensitivities
    elif front_brace == "header_and_skirt":
        front_shells = [s for s in asm.shell_elements if s.bay == "front_brace"]
        assert len(front_shells) >= 8
        assert fb_meta["header_height_mm"] == pytest.approx(55.0)
        assert fb_meta["skirt_height_mm"] == pytest.approx(90.0)
        hdr_h_m = fb_meta["header_height_mm"] * 1e-3
        skr_h_m = fb_meta["skirt_height_mm"] * 1e-3
        # No shells inside open door gap z in (skr_h_m, h_m - hdr_h_m)
        for sh in front_shells:
            c = np.mean([nmap[nid].position_m for nid in sh.node_ids], axis=0)
            assert not (skr_h_m < c[2] < h_m - hdr_h_m)
        # Verify floating inner horizontal edges (z=z_gantry middle & z=90mm middle) have NO CBUSH springs
        front_springs = [sp for sp in asm.fastener_springs if sp.bay == "front_brace"]
        assert len(front_springs) >= 10
        for sp in front_springs:
            p_sh = nmap[sp.shell_node_id].position_m
            on_outer_rail = (
                p_sh[0] <= 1e-4
                or p_sh[0] >= w_m - 1e-4
                or p_sh[2] <= 1e-4
                or p_sh[2] >= h_m - 1e-4
            )
            assert on_outer_rail, f"Spring {sp.id} attached to floating inner edge at {p_sh}"
        assert "front_brace" in sen.panel_bay_sensitivities
    else:
        brace_bars = [m for m in asm.members if m.id.startswith("brace_front_open_")]
        assert len(brace_bars) == 4
        assert all(m.role == "brace" and not m.is_truss for m in brace_bars)
        assert all(m.profile.id == "al_flat_bar_25x5" for m in brace_bars)
        assert all(m.material.id == "aluminum_6061_t6" for m in brace_bars)

    assert (asm.total_mass_kg - asm0.total_mass_kg) == pytest.approx(asm.front_brace_mass_kg, abs=1e-4)
    assert (asm.estimated_material_cost_usd - asm0.estimated_material_cost_usd) == pytest.approx(
        asm.front_brace_cost_usd, abs=0.05
    )
    assert (bom.total_mass_kg - bom0.total_mass_kg) == pytest.approx(
        asm.front_brace_mass_kg, abs=2e-3
    )
    assert (bom.total_cost_usd - bom0.total_cost_usd) == pytest.approx(
        asm.front_brace_cost_usd, abs=0.05
    )
    assert any(item.sku.startswith("FRM-FRONT-") for item in bom.items)


def test_combined_braces_panel_combinations_and_non_corexy_guard() -> None:
    """Combined top+front braces work with enclosed=False, top_panel=True, V2.4, and stay off on non-CoreXY."""
    # 1. Combined CNC Halo + Header & Skirt on open-frame (enclosed_panels=False)
    spec_open = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    spec_open.frame.enclosed_panels = False
    spec_open.frame.top_brace = "cnc_halo_ring"
    spec_open.frame.front_brace = "header_and_skirt"
    _assert_spd(spec_open)
    rep_open = evaluate_printer(spec_open, include_advanced_analysis=False)
    assert rep_open.scores.composite > 0.0
    assert any(s.bay == "top_brace" for s in rep_open.assembly.shell_elements)
    assert any(s.bay == "front_brace" for s in rep_open.assembly.shell_elements)

    # 2. Combined CNC Halo + CNC Portal with enclosed_panels=True & top_panel=True
    spec_enc = _open_top_bolted_al_trident()
    spec_enc.frame.top_panel = True
    spec_enc.frame.top_brace = "cnc_halo_ring"
    spec_enc.frame.front_brace = "cnc_portal_frame"
    _assert_spd(spec_enc)
    asm_enc = build_assembly(spec_enc)
    assert any(s.bay == "top" for s in asm_enc.shell_elements)
    assert any(s.bay == "top_brace" for s in asm_enc.shell_elements)
    assert any(s.bay == "front_brace" for s in asm_enc.shell_elements)

    # 3. Flying-gantry CoreXY (Voron 2.4) works cleanly
    v24 = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    v24.frame.top_panel = False
    v24.frame.top_brace = "rear_v_and_front_corners"
    v24.frame.front_brace = "header_and_skirt"
    _assert_spd(v24)

    # 4. Non-CoreXY keeps top_brace and front_brace inactive
    for non_corexy in ("cartesian_bedslinger.yaml", "delta_racer.yaml"):
        nc_base = PrinterSpec.from_yaml(EXAMPLES_DIR / non_corexy)
        nc_braced = nc_base.model_copy(deep=True)
        nc_braced.frame.top_brace = "cnc_halo_ring"
        nc_braced.frame.front_brace = "cnc_portal_frame"
        asm0 = build_assembly(nc_base)
        asm1 = build_assembly(nc_braced)
        gd1 = asm1.to_geometry_dict()
        assert gd1["top_brace"]["active"] is False
        assert gd1["front_brace"]["active"] is False
        assert asm1.total_mass_kg == pytest.approx(asm0.total_mass_kg, rel=1e-12)
        assert asm1.estimated_material_cost_usd == pytest.approx(
            asm0.estimated_material_cost_usd, rel=1e-12
        )


def test_modal_and_sensitivity_classification_for_braces() -> None:
    """Modal and sensitivity classifiers label top_brace, front_brace, and flat bars cleanly."""
    assert _classify_modal_group("bay_top_brace", "panel", bay="top_brace") == (
        "top_brace",
        "Top Open-Center Brace",
    )
    assert _classify_modal_group("bay_front_brace", "panel", bay="front_brace") == (
        "front_brace",
        "Front Open-Door Brace",
    )
    assert classify_part_group("brace_top_open_0", "brace") == (
        "frame_bracing",
        "Frame Diagonal Bracing",
    )
    assert classify_part_group("brace_front_open_2", "brace") == (
        "frame_bracing",
        "Frame Diagonal Bracing",
    )


def test_dxf_export_and_catalog_endpoint_for_top_front_braces() -> None:
    """DXF export supports top_halo_ring, front_portal_frame, front_header_skirt, all_plates, and /api/catalog."""
    spec = _open_top_bolted_al_trident()
    spec.frame.top_brace = "cnc_halo_ring"
    spec.frame.front_brace = "cnc_portal_frame"

    dims_top = top_brace_dimensions_mm(spec)
    assert dims_top["flange_width_mm"] == pytest.approx(50.0)
    assert dims_top["cutout_w_mm"] == pytest.approx(spec.frame.width_mm - 100.0)
    assert dims_top["cutout_d_mm"] == pytest.approx(spec.frame.depth_mm - 100.0)

    dims_front = front_brace_dimensions_mm(spec)
    assert dims_front["flange_width_mm"] == pytest.approx(55.0)
    assert dims_front["cutout_w_mm"] == pytest.approx(spec.frame.width_mm - 110.0)
    assert dims_front["cutout_h_mm"] == pytest.approx(spec.frame.height_mm - 110.0)

    # 1. top_halo_ring DXF: outer rect on CUT_OUTER, R20 rounded inner cutout on HOLES w/ bulges, M5 Ø5.5 mm holes
    doc_halo = ezdxf.read(io.StringIO(export_panels_dxf_bytes(spec, "top_halo_ring").decode("utf-8")))
    msp_halo = doc_halo.modelspace()
    assert len(list(msp_halo.query('LWPOLYLINE[layer=="CUT_OUTER"]'))) == 1
    halo_cutouts = list(msp_halo.query('LWPOLYLINE[layer=="HOLES"]'))
    assert len(halo_cutouts) == 1
    assert any(abs(pt[4]) > 0.1 for pt in halo_cutouts[0].get_points("xyseb"))
    halo_bolts = list(msp_halo.query('CIRCLE[layer=="HOLES"]'))
    assert len(halo_bolts) >= 16
    assert all(c.dxf.radius == pytest.approx(2.75, abs=0.02) for c in halo_bolts)

    # 2. front_portal_frame DXF
    doc_portal = ezdxf.read(
        io.StringIO(export_panels_dxf_bytes(spec, "front_portal_frame").decode("utf-8"))
    )
    msp_portal = doc_portal.modelspace()
    assert len(list(msp_portal.query('LWPOLYLINE[layer=="CUT_OUTER"]'))) == 1
    portal_cutouts = list(msp_portal.query('LWPOLYLINE[layer=="HOLES"]'))
    assert len(portal_cutouts) == 1
    assert any(abs(pt[4]) > 0.1 for pt in portal_cutouts[0].get_points("xyseb"))
    portal_bolts = list(msp_portal.query('CIRCLE[layer=="HOLES"]'))
    assert len(portal_bolts) >= 16
    assert all(c.dxf.radius == pytest.approx(2.75, abs=0.02) for c in portal_bolts)

    # 3. front_header_skirt DXF (2 plates: upper header + lower skirt)
    spec_hs = spec.model_copy(deep=True)
    spec_hs.frame.front_brace = "header_and_skirt"
    doc_hs = ezdxf.read(
        io.StringIO(export_panels_dxf_bytes(spec_hs, "front_header_skirt").decode("utf-8"))
    )
    msp_hs = doc_hs.modelspace()
    assert len(list(msp_hs.query('LWPOLYLINE[layer=="CUT_OUTER"]'))) == 2
    hs_bolts = list(msp_hs.query('CIRCLE[layer=="HOLES"]'))
    assert len(hs_bolts) >= 12
    assert all(c.dxf.radius == pytest.approx(2.75, abs=0.02) for c in hs_bolts)

    # 4. list_dxf_parts & all_plates include active CNC top/front braces automatically
    parts_portal = [p["part"] for p in list_dxf_parts(spec)]
    assert "top_halo_ring" in parts_portal
    assert "front_portal_frame" in parts_portal
    assert list_dxf_parts(spec) == list_dxf_exportable_parts(spec)

    parts_hs = [p["part"] for p in list_dxf_parts(spec_hs)]
    assert "top_halo_ring" in parts_hs
    assert "front_header_skirt" in parts_hs

    doc_all = ezdxf.read(io.StringIO(export_panels_dxf_bytes(spec_hs, "all_plates").decode("utf-8")))
    # 2 side + 1 rear + 1 halo + 2 (header+skirt) + 4 gussets = 10 plates
    assert len(list(doc_all.modelspace().query('LWPOLYLINE[layer=="CUT_OUTER"]'))) >= 10

    # 5. FastAPI /api/catalog exposes top_braces and front_braces
    client = TestClient(create_app())
    resp = client.get("/api/catalog")
    assert resp.status_code == 200
    cat = resp.json()
    assert [item["id"] for item in cat["top_braces"]] == list(TOP_BRACE_CATALOG.keys())
    assert [item["id"] for item in cat["front_braces"]] == list(FRONT_BRACE_CATALOG.keys())
