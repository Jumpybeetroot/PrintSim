"""Extract a prismatic part's cross-section from vendor CAD and derive exact section properties.

Slices one extrusion/tube solid out of a STEP file (optionally picked by an assembly label
substring), writes the 2D outline as a small JSON fixture, and prints the area, second moments
and St. Venant torsion constant from a `sectionproperties` warping analysis next to the
matching ``PROFILE_CATALOG`` entry.

    uv run --extra cad python scripts/derive_profile_sections.py \
        --step cad_sources/voron_trident/extracted/trident_assembly.step \
        --label "HFSB5-2020-240-D" --profile extrusion_2020 \
        --out tests/fixtures/profile_sections/extrusion_2020.json

Needs the optional ``cad`` extra (build123d) and ``sectionproperties``. The fixture is what
``tests/test_profile_sections_crosscheck.py`` reads, so CI never needs the CAD file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _iter_leaf_solids(node: Any, prefix: str = ""):
    label = (getattr(node, "label", "") or "").strip()
    path = f"{prefix}/{label}" if (prefix and label) else (label or prefix)
    children = getattr(node, "children", None) or []
    if children:
        for child in children:
            yield from _iter_leaf_solids(child, path)
        return
    for solid in node.solids() if hasattr(node, "solids") else []:
        yield path, solid


def _wire_points(wire: Any, b3d: Any, arc_points: int) -> list[tuple[float, float, float]]:
    pts: list[tuple[float, float, float]] = []
    for edge in wire.order_edges():
        if edge.geom_type == b3d.GeomType.LINE:
            params = [0.0]
        else:
            params = np.linspace(0.0, 1.0, arc_points, endpoint=False)
        pts.extend(tuple(edge.position_at(float(t))) for t in params)
    return pts


def extract_section(
    step_path: Path, label: str | None, slice_fraction: float, arc_points: int
) -> dict[str, Any]:
    """Return the outline (mm, centred on the bounding-box centre) of the picked solid's cross-section."""
    import build123d as b3d

    root = b3d.import_step(str(step_path))
    candidates = [(p, s) for p, s in _iter_leaf_solids(root) if label is None or label in p]
    if not candidates:
        raise SystemExit(f"No solid matching label {label!r} in {step_path}")
    path, solid = candidates[0]

    bb = solid.bounding_box()
    sizes = np.array([bb.size.X, bb.size.Y, bb.size.Z])
    mins = np.array([bb.min.X, bb.min.Y, bb.min.Z])
    axis = int(np.argmax(sizes))
    origin = mins + 0.5 * sizes
    origin[axis] = mins[axis] + slice_fraction * sizes[axis]
    z_dir = [0.0, 0.0, 0.0]
    z_dir[axis] = 1.0
    faces = solid.intersect(b3d.Plane(origin=tuple(origin), z_dir=tuple(z_dir))).faces()
    if len(faces) != 1:
        raise SystemExit(f"Expected one cut face, got {len(faces)}; pick another --slice-fraction")
    face = faces[0]

    in_plane = [i for i in range(3) if i != axis]
    centre = mins + 0.5 * sizes

    def to_2d(pts: list[tuple[float, float, float]]) -> list[list[float]]:
        return [[round(p[in_plane[0]] - centre[in_plane[0]], 5), round(p[in_plane[1]] - centre[in_plane[1]], 5)] for p in pts]

    return {
        "source_step": step_path.name,
        "source_label": path,
        "length_mm": round(float(sizes[axis]), 3),
        "slice_fraction": slice_fraction,
        "cad_cut_face_area_mm2": round(float(face.area), 4),
        "units": "mm",
        "exterior": to_2d(_wire_points(face.outer_wire(), b3d, arc_points)),
        "interiors": [to_2d(_wire_points(w, b3d, arc_points)) for w in face.inner_wires()],
    }


def section_properties(section: dict[str, Any], mesh_size_mm2: float = 0.05) -> dict[str, float]:
    """Area, centroidal second moments and torsion constant of a fixture outline (mm units)."""
    from sectionproperties.analysis import Section
    from sectionproperties.pre import Geometry
    from shapely.geometry import Polygon

    poly = Polygon(section["exterior"], holes=section["interiors"])
    geom = Geometry(poly)
    geom.create_mesh(mesh_sizes=[mesh_size_mm2])
    sec = Section(geom)
    sec.calculate_geometric_properties()
    sec.calculate_warping_properties()
    ixx, iyy, ixy = sec.get_ic()
    return {
        "area_mm2": float(sec.get_area()),
        "ixx_mm4": float(ixx),
        "iyy_mm4": float(iyy),
        "ixy_mm4": float(ixy),
        "j_mm4": float(sec.get_j()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", type=Path, required=True, help="STEP file (single part or assembly)")
    ap.add_argument("--label", default=None, help="Substring of the assembly label path to pick")
    ap.add_argument("--profile", default=None, help="PROFILE_CATALOG id to compare against")
    ap.add_argument("--slice-fraction", type=float, default=0.37, help="Cut position along the long axis (0..1)")
    ap.add_argument("--arc-points", type=int, default=64, help="Points per curved edge")
    ap.add_argument("--out", type=Path, default=None, help="Write the outline fixture JSON here")
    ap.add_argument("--source-note", default=None, help="Provenance text stored in the fixture")
    args = ap.parse_args()

    section = extract_section(args.step, args.label, args.slice_fraction, args.arc_points)
    if args.profile:
        section["profile_id"] = args.profile
    if args.source_note:
        section["source_note"] = args.source_note
    props = section_properties(section)
    print(f"picked: {section['source_label']}  (length {section['length_mm']} mm)")
    print(f"CAD cut-face area: {section['cad_cut_face_area_mm2']} mm^2")
    for key, val in props.items():
        print(f"  {key:9s} = {val:12.3f}")
    if args.profile:
        from printsim.schema.profiles import PROFILE_CATALOG

        cat = PROFILE_CATALOG[args.profile]
        print(f"catalog {args.profile}:")
        print(f"  area_mm2  = {cat.area_m2 * 1e6:12.3f}  ({(cat.area_m2 * 1e6 / props['area_mm2'] - 1) * 100:+.1f} %)")
        print(f"  iy_mm4    = {cat.iy_m4 * 1e12:12.3f}  ({(cat.iy_m4 * 1e12 / props['ixx_mm4'] - 1) * 100:+.1f} %)")
        print(f"  iz_mm4    = {cat.iz_m4 * 1e12:12.3f}  ({(cat.iz_m4 * 1e12 / props['iyy_mm4'] - 1) * 100:+.1f} %)")
        print(f"  j_mm4     = {cat.j_m4 * 1e12:12.3f}  ({(cat.j_m4 * 1e12 / props['j_mm4'] - 1) * 100:+.1f} %)")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(section, separators=(",", ":")) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
