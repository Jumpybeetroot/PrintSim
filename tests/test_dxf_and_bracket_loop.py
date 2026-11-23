"""Tests for 2D CNC/Laser DXF exporter and Custom CAD/CMS bracket modal FEA injection."""

from __future__ import annotations

import io
import json
from pathlib import Path

import ezdxf
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from printsim.api.server import create_app
from printsim.cli import app as cli_app
from printsim.engine import evaluate_printer
from printsim.geometry import (
    export_panels_dxf,
    export_panels_dxf_bytes,
    list_dxf_exportable_parts,
)
from printsim.mcp_server import handle_jsonrpc_message
from printsim.schema import CustomBracketOverride, PrinterSpec

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "examples"


def _parse_dxf_bytes(dxf_bytes: bytes) -> ezdxf.document.Drawing:
    return ezdxf.read(io.StringIO(dxf_bytes.decode("utf-8")))


def _assert_standard_layers(doc: ezdxf.document.Drawing) -> None:
    assert doc.units == ezdxf.units.MM
    layer_names = {layer.dxf.name for layer in doc.layers}
    assert {"CUT_OUTER", "HOLES", "ANNOTATIONS"} <= layer_names
    assert doc.layers.get("CUT_OUTER").color == 7
    assert doc.layers.get("HOLES").color == 1
    assert doc.layers.get("ANNOTATIONS").color == 3


def test_dxf_export_layers_entities_and_parts(tmp_path: Path) -> None:
    """Verify DXF export produces valid R2010 CAD entities across all supported plate types."""
    base_spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    bolted_bb_spec = base_spec.model_copy(
        update={
            "frame": base_spec.frame.model_copy(
                update={
                    "panel_mounting": "bolted_structural",
                    "trident_backbone": True,
                }
            )
        }
    )

    parts_meta = list_dxf_exportable_parts(bolted_bb_spec)
    assert [p["part"] for p in parts_meta] == [
        "side_panel",
        "rear_panel",
        "top_panel",
        "trident_backbone",
        "corner_gusset",
        "all_plates",
    ]

    for part in ("all_plates", "side_panel", "rear_panel", "top_panel", "trident_backbone", "corner_gusset"):
        raw = export_panels_dxf_bytes(bolted_bb_spec, part=part)
        doc = _parse_dxf_bytes(raw)
        _assert_standard_layers(doc)
        msp = doc.modelspace()

        cut_polys = list(msp.query('LWPOLYLINE[layer=="CUT_OUTER"]'))
        hole_circles = list(msp.query('CIRCLE[layer=="HOLES"]'))
        ann_texts = list(msp.query('TEXT[layer=="ANNOTATIONS"]'))

        assert len(cut_polys) >= 1
        assert all(pl.closed for pl in cut_polys)
        assert len(hole_circles) >= 4
        assert len(ann_texts) >= 2

    # Side panel mounting modes: bolted_structural (~75 mm pitch) vs clip_enclosure (4 corner notches)
    doc_bolted_side = _parse_dxf_bytes(export_panels_dxf_bytes(bolted_bb_spec, part="side_panel"))
    doc_clip_side = _parse_dxf_bytes(export_panels_dxf_bytes(base_spec, part="side_panel"))
    bolted_circles = list(doc_bolted_side.modelspace().query('CIRCLE[layer=="HOLES"]'))
    clip_circles = list(doc_clip_side.modelspace().query('CIRCLE[layer=="HOLES"]'))
    clip_poly = list(doc_clip_side.modelspace().query('LWPOLYLINE[layer=="CUT_OUTER"]'))[0]
    assert len(bolted_circles) > len(clip_circles)
    assert len(clip_poly) == 12  # 4 corner clip notches

    # Rear panel includes outer profile + lower cable/electronics pass-through cutout
    doc_rear = _parse_dxf_bytes(export_panels_dxf_bytes(bolted_bb_spec, part="rear_panel"))
    assert len(list(doc_rear.modelspace().query('LWPOLYLINE[layer=="CUT_OUTER"]'))) == 2

    # Trident backbone: central weight-relief cutout + 3x Ø22.5 mm pilot bores + 12x NEMA17 M3 holes
    doc_bb = _parse_dxf_bytes(export_panels_dxf_bytes(bolted_bb_spec, part="trident_backbone"))
    bb_polys = list(doc_bb.modelspace().query('LWPOLYLINE[layer=="CUT_OUTER"]'))
    bb_circles = list(doc_bb.modelspace().query('CIRCLE[layer=="HOLES"]'))
    assert len(bb_polys) == 2
    pilot_bores = [c for c in bb_circles if c.dxf.radius == pytest.approx(11.25, abs=0.05)]
    nema_m3_holes = [c for c in bb_circles if c.dxf.radius == pytest.approx(1.7, abs=0.05)]
    assert len(pilot_bores) == 3
    assert len(nema_m3_holes) == 12

    # Corner gusset scales with frame.profile width (2020 -> 40x40, 3030 -> 60x60) with 4x M5 holes
    doc_g20 = _parse_dxf_bytes(export_panels_dxf_bytes(base_spec, part="corner_gusset"))
    g20_poly = list(doc_g20.modelspace().query('LWPOLYLINE[layer=="CUT_OUTER"]'))[0]
    g20_pts = list(g20_poly.get_points("xy"))
    assert max(p[0] for p in g20_pts) == pytest.approx(40.0)
    assert max(p[1] for p in g20_pts) == pytest.approx(40.0)
    g20_holes = list(doc_g20.modelspace().query('CIRCLE[layer=="HOLES"]'))
    assert len(g20_holes) == 4
    assert all(c.dxf.radius == pytest.approx(2.75, abs=0.02) for c in g20_holes)

    spec_3030 = base_spec.model_copy(
        update={"frame": base_spec.frame.model_copy(update={"profile": "extrusion_3030"})}
    )
    doc_g30 = _parse_dxf_bytes(export_panels_dxf_bytes(spec_3030, part="corner_gusset"))
    g30_poly = list(doc_g30.modelspace().query('LWPOLYLINE[layer=="CUT_OUTER"]'))[0]
    g30_pts = list(g30_poly.get_points("xy"))
    assert max(p[0] for p in g30_pts) == pytest.approx(60.0)

    # All-plates nested layout keeps 30 mm spacing between plates
    doc_all = _parse_dxf_bytes(export_panels_dxf_bytes(bolted_bb_spec, part="all_plates"))
    all_polys = list(doc_all.modelspace().query('LWPOLYLINE[layer=="CUT_OUTER"]'))
    # Verify leftmost two outer plates (side panels) have exact 30 mm gap
    side_xs = [list(pl.get_points("xy")) for pl in all_polys[:2]]
    max_x_first = max(p[0] for p in side_xs[0])
    min_x_second = min(p[0] for p in side_xs[1])
    assert (min_x_second - max_x_first) == pytest.approx(30.0, abs=0.5)

    # File write helper and error handling
    out_file = export_panels_dxf(bolted_bb_spec, tmp_path / "kit.dxf", part="all_plates")
    assert out_file.exists() and out_file.stat().st_size > 1000
    with pytest.raises(ValueError, match="Unknown DXF part"):
        export_panels_dxf_bytes(base_spec, part="invalid_plate")


def test_custom_bracket_override_modal_injection_monotonicity() -> None:
    """Verify CustomBracketOverride directly controls drive stiffness & full-printer shaper Hz."""
    base_spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    rep_base = evaluate_printer(base_spec, include_advanced_analysis=False)

    assert rep_base.kinematics.custom_bracket_active is False
    assert rep_base.kinematics.custom_bracket_label == ""
    assert rep_base.kinematics.to_dict()["custom_bracket_active"] is False

    stiff_spec = base_spec.model_copy(
        update={
            "gantry": base_spec.gantry.model_copy(
                update={
                    "custom_bracket_override": CustomBracketOverride(
                        label="CNC 6061 Billet Motor Mount",
                        stiffness_n_per_mm=4500.0,
                        mass_delta_g=12.0,
                        source="tet10_fea",
                    )
                }
            )
        }
    )
    rep_stiff = evaluate_printer(stiff_spec, include_advanced_analysis=False)
    assert rep_stiff.kinematics.custom_bracket_active is True
    assert rep_stiff.kinematics.custom_bracket_label == "CNC 6061 Billet Motor Mount"
    assert rep_stiff.kinematics.bracket_stiffness_n_per_mm == pytest.approx(4500.0, rel=1e-3)
    assert rep_stiff.kinematics.drive_stiffness_n_per_mm > rep_base.kinematics.drive_stiffness_n_per_mm
    assert rep_stiff.modal_fea.shaper_x is not None and rep_base.modal_fea.shaper_x is not None
    assert rep_stiff.modal_fea.shaper_y is not None and rep_base.modal_fea.shaper_y is not None
    assert rep_stiff.modal_fea.shaper_x.effective_freq_hz > rep_base.modal_fea.shaper_x.effective_freq_hz
    assert rep_stiff.modal_fea.shaper_y.effective_freq_hz > rep_base.modal_fea.shaper_y.effective_freq_hz

    soft_spec = base_spec.model_copy(
        update={
            "gantry": base_spec.gantry.model_copy(
                update={
                    "custom_bracket_override": CustomBracketOverride(
                        label="Compliant Prototype Mount",
                        stiffness_n_per_mm=25.0,
                        source="custom_cad",
                    )
                }
            )
        }
    )
    rep_soft = evaluate_printer(soft_spec, include_advanced_analysis=False)
    assert rep_soft.kinematics.custom_bracket_active is True
    assert rep_soft.kinematics.bracket_stiffness_n_per_mm == pytest.approx(25.0, rel=1e-3)
    assert rep_soft.kinematics.drive_stiffness_n_per_mm < rep_base.kinematics.drive_stiffness_n_per_mm
    assert rep_soft.modal_fea.shaper_x is not None and rep_soft.modal_fea.shaper_y is not None
    assert rep_soft.modal_fea.shaper_x.effective_freq_hz < rep_base.modal_fea.shaper_x.effective_freq_hz
    assert rep_soft.modal_fea.shaper_y.effective_freq_hz < rep_base.modal_fea.shaper_y.effective_freq_hz


def test_api_cli_and_mcp_dxf_and_bracket_inject_end_to_end(tmp_path: Path) -> None:
    """Verify FastAPI /api/export/dxf, /api/brackets/inject-evaluate, CLI export-dxf, and MCP tool."""
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    spec_dict = spec.model_dump(mode="json")

    client = TestClient(create_app())

    # 1. POST /api/export/dxf
    r_dxf = client.post("/api/export/dxf", json={"spec": spec_dict, "part": "trident_backbone"})
    assert r_dxf.status_code == 200
    assert "application/dxf" in r_dxf.headers["content-type"]
    assert "trident_backbone.dxf" in r_dxf.headers.get("content-disposition", "")
    doc_api = _parse_dxf_bytes(r_dxf.content)
    _assert_standard_layers(doc_api)

    r_bad = client.post("/api/export/dxf", json={"spec": spec_dict, "part": "bad_part"})
    assert r_bad.status_code == 400

    # 2. POST /api/brackets/inject-evaluate (CMS superelement path when custom_stiffness_n_per_mm is None)
    r_cms = client.post(
        "/api/brackets/inject-evaluate",
        json={
            "spec": spec_dict,
            "bracket_type": "xy_motor_mount",
            "material": "aluminum_6061_t6",
        },
    )
    assert r_cms.status_code == 200, r_cms.text
    cms_data = r_cms.json()
    expected_keys = {
        "baseline_x_hz",
        "injected_x_hz",
        "delta_x_hz",
        "baseline_y_hz",
        "injected_y_hz",
        "delta_y_hz",
        "bracket_stiffness_n_per_mm",
        "injected_spec",
        "summary",
    }
    assert expected_keys <= set(cms_data.keys())
    assert cms_data["bracket_stiffness_n_per_mm"] > 1000.0
    assert cms_data["delta_x_hz"] > 0.0
    assert cms_data["delta_y_hz"] > 0.0
    assert cms_data["injected_x_hz"] == pytest.approx(
        cms_data["baseline_x_hz"] + cms_data["delta_x_hz"], abs=0.05
    )
    assert cms_data["injected_spec"]["gantry"]["custom_bracket_override"]["source"] == "cms_superelement"

    # Explicit custom stiffness override (soft mount -> negative delta)
    r_soft = client.post(
        "/api/brackets/inject-evaluate",
        json={
            "spec": spec_dict,
            "bracket_type": "corner_gusset",
            "material": "abs_printed",
            "custom_stiffness_n_per_mm": 25.0,
            "mass_delta_g": 5.0,
            "label": "Soft Printed Test Mount",
        },
    )
    assert r_soft.status_code == 200, r_soft.text
    soft_data = r_soft.json()
    assert soft_data["bracket_stiffness_n_per_mm"] == pytest.approx(25.0)
    assert soft_data["delta_x_hz"] < 0.0
    assert soft_data["delta_y_hz"] < 0.0
    assert soft_data["summary"]["label"] == "Soft Printed Test Mount"

    # 3. CLI export-dxf
    cli_out = tmp_path / "cli_kit.dxf"
    runner = CliRunner()
    cli_res = runner.invoke(
        cli_app,
        [
            "export-dxf",
            str(EXAMPLES_DIR / "voron_trident_300.yaml"),
            "--out",
            str(cli_out),
            "--part",
            "all_plates",
        ],
    )
    assert cli_res.exit_code == 0, cli_res.output
    assert cli_out.exists()
    _assert_standard_layers(_parse_dxf_bytes(cli_out.read_bytes()))

    # 4. MCP export_printer_dxf
    mcp_out = tmp_path / "mcp_gusset.dxf"
    yaml_text = (EXAMPLES_DIR / "voron_trident_300.yaml").read_text(encoding="utf-8")
    resp = handle_jsonrpc_message(
        {
            "jsonrpc": "2.0",
            "id": 42,
            "method": "tools/call",
            "params": {
                "name": "export_printer_dxf",
                "arguments": {
                    "yaml_spec": yaml_text,
                    "output_path": str(mcp_out),
                    "part": "corner_gusset",
                },
            },
        }
    )
    assert resp is not None
    assert resp["result"]["isError"] is False, resp["result"]["content"][0]["text"]
    payload = json.loads(resp["result"]["content"][0]["text"])
    assert payload["part"] == "corner_gusset"
    assert payload["bytes"] > 500
    assert mcp_out.exists()
    _assert_standard_layers(_parse_dxf_bytes(mcp_out.read_bytes()))
