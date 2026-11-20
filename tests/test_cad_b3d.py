"""Tests for OpenCASCADE B-Rep parametric CAD parts and STEP/STL export via build123d."""

from __future__ import annotations

from pathlib import Path

import pytest
import trimesh

from printsim.fea.solid_fem import generate_parametric_part_grid
from printsim.geometry import (
    build_assembly,
    build_assembly_brep,
    build_part_brep,
    export_assembly_step,
    export_assembly_step_bytes,
    export_part_step,
    export_part_step_bytes,
    export_part_stl_bytes,
    get_part_default_bcs,
    inspect_custom_mesh,
    is_cad_available,
)
from printsim.geometry.cad_b3d import PartPreset
from printsim.schema import CustomMeshSpec, PrinterSpec

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"

PART_SPECS: list[tuple[PartPreset, tuple[float, float, float], tuple[float, float]]] = [
    ("cantilever_bar", (60.0, 12.0, 12.0), (8639.9, 8640.1)),
    ("corner_gusset", (40.0, 40.0, 20.0), (11000.0, 12600.0)),
    ("xy_motor_mount", (60.0, 52.0, 16.0), (28000.0, 33000.0)),
    ("toolhead_carriage", (56.0, 48.0, 12.0), (23500.0, 26000.0)),
]


@pytest.mark.parametrize("preset", [p[0] for p in PART_SPECS])
def test_part_default_bcs_match_solid_fem(preset: PartPreset) -> None:
    _, _, expected_fixed, expected_load, expected_force = generate_parametric_part_grid(preset)
    fixed, load, force = get_part_default_bcs(preset)
    assert fixed == expected_fixed
    assert load == expected_load
    assert force == expected_force


def test_unknown_preset_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unknown solid part preset"):
        get_part_default_bcs("nonexistent_part")  # type: ignore[arg-type]


@pytest.mark.skipif(not is_cad_available(), reason="build123d optional CAD extra not installed")
@pytest.mark.parametrize("preset,expected_dims,vol_range", PART_SPECS)
def test_build_part_brep_single_valid_solid_and_exact_bounds(
    preset: PartPreset,
    expected_dims: tuple[float, float, float],
    vol_range: tuple[float, float],
) -> None:
    solid = build_part_brep(preset)

    assert solid.is_valid
    assert len(solid.solids()) == 1
    vol = float(solid.volume)
    assert vol_range[0] <= vol <= vol_range[1]

    if preset == "cantilever_bar":
        assert vol == pytest.approx(60.0 * 12.0 * 12.0, rel=1e-6)

    bbox = solid.bounding_box()
    assert bbox.min.X == pytest.approx(0.0, abs=1e-3)
    assert bbox.min.Y == pytest.approx(0.0, abs=1e-3)
    assert bbox.min.Z == pytest.approx(0.0, abs=1e-3)
    assert bbox.max.X == pytest.approx(expected_dims[0], abs=1e-3)
    assert bbox.max.Y == pytest.approx(expected_dims[1], abs=1e-3)
    assert bbox.max.Z == pytest.approx(expected_dims[2], abs=1e-3)


@pytest.mark.skipif(not is_cad_available(), reason="build123d optional CAD extra not installed")
@pytest.mark.parametrize("preset", [p[0] for p in PART_SPECS])
def test_part_step_and_stl_exports(preset: PartPreset, tmp_path: Path) -> None:
    import build123d as b3d

    solid = build_part_brep(preset)
    ref_vol = float(solid.volume)

    # STEP file export & re-import
    step_path = export_part_step(preset, tmp_path / f"{preset}.step")
    assert step_path.exists()
    raw_file = step_path.read_bytes()
    assert b"ISO-10303-21;" in raw_file
    assert b"END-ISO-10303-21;" in raw_file

    reimported = b3d.import_step(str(step_path))
    assert float(reimported.volume) == pytest.approx(ref_vol, rel=1e-3)

    # STEP bytes export
    step_bytes = export_part_step_bytes(preset)
    assert b"ISO-10303-21;" in step_bytes
    assert len(step_bytes) > 500

    # STL bytes export & trimesh validation
    stl_bytes = export_part_stl_bytes(preset)
    assert len(stl_bytes) > 100
    mesh = trimesh.load(trimesh.util.wrap_as_stream(stl_bytes), file_type="stl", force="mesh")
    assert mesh.is_watertight
    assert abs(float(mesh.volume)) == pytest.approx(ref_vol, rel=2e-2)


@pytest.mark.skipif(not is_cad_available(), reason="build123d optional CAD extra not installed")
def test_export_assembly_step_corexy_roundtrip_and_inspect(tmp_path: Path) -> None:
    import build123d as b3d

    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    asm = build_assembly(spec)

    # Verify panel_shear_* virtual members are excluded from B-Rep compound
    physical_members = [m for m in asm.members if not m.id.startswith("panel_shear_")]
    compound = build_assembly_brep(asm)
    assert len(compound.solids()) == len(physical_members) + 2  # + print_bed + toolhead

    step_path = export_assembly_step(asm, tmp_path / "corexy_voron.step")
    assert step_path.exists()
    step_bytes = step_path.read_bytes()
    assert b"ISO-10303-21;" in step_bytes
    assert b"END-ISO-10303-21;" in step_bytes

    # Bytes helper matches header and non-trivial payload
    asm_bytes = export_assembly_step_bytes(asm)
    assert b"ISO-10303-21;" in asm_bytes
    assert len(asm_bytes) > 5000

    # Re-import with build123d and verify against inspect_custom_mesh
    reimported = b3d.import_step(str(step_path))
    re_vol = float(reimported.volume)
    re_bbox = reimported.bounding_box()
    assert re_vol > 0.0
    assert (re_bbox.max.X - re_bbox.min.X) >= spec.frame.width_mm - 1.0
    assert (re_bbox.max.Y - re_bbox.min.Y) >= spec.frame.depth_mm - 1.0
    assert (re_bbox.max.Z - re_bbox.min.Z) >= spec.frame.height_mm - 1.0

    inspected = inspect_custom_mesh(
        CustomMeshSpec(
            id="corexy_step_check",
            name="CoreXY STEP Check",
            file_path=str(step_path),
            material="aluminum_6063_t5",
        )
    )
    assert inspected["format"] == "step"
    assert inspected["watertight"] is True
    assert inspected["volume_mm3"] == pytest.approx(re_vol, rel=1e-4)
    assert inspected["mass_g"] > 0.0
    assert inspected["bounds_mm"][0] == pytest.approx(
        [re_bbox.min.X, re_bbox.min.Y, re_bbox.min.Z], abs=1e-3
    )
    assert inspected["bounds_mm"][1] == pytest.approx(
        [re_bbox.max.X, re_bbox.max.Y, re_bbox.max.Z], abs=1e-3
    )


@pytest.mark.skipif(not is_cad_available(), reason="build123d optional CAD extra not installed")
def test_export_assembly_step_delta_with_rods_and_diagonals(tmp_path: Path) -> None:
    import build123d as b3d

    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "delta_racer.yaml")
    asm = build_assembly(spec)
    step_path = export_assembly_step(asm, tmp_path / "delta_racer.step")
    assert step_path.exists()
    assert b"ISO-10303-21;" in step_path.read_bytes()

    reimported = b3d.import_step(str(step_path))
    assert float(reimported.volume) > 0.0
    assert len(reimported.solids()) >= 10
