"""Local B-Rep STEP / Mesh Inspector for Zero-LLM CAD-to-Superelement Pipelines.

Inspects `.step`/`.stp` assemblies via ``build123d`` (or `.stl`/`.glb`/`.obj` via ``trimesh``),
extracting solid volumes, material mass estimates, axis-aligned planar mounting faces,
cylindrical holes/bosses, solid-pair overlaps, and suggested Port 0 / Port 1 selectors.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from printsim.schema import get_material

MASS_DENSITIES_G_CM3: dict[str, float] = {
    "aluminum_6061": 2.70,
    "titanium_grade5": 4.43,
    "steel": 7.85,
    "abs_pc": 1.18,
}

_AXIS_LABELS: tuple[tuple[int, int, str], ...] = (
    (0, 1, "+X"),
    (0, -1, "-X"),
    (1, 1, "+Y"),
    (1, -1, "-Y"),
    (2, 1, "+Z"),
    (2, -1, "-Z"),
)


@dataclass
class SolidInspectInfo:
    """Geometric and mass summary of a single solid or sub-mesh."""

    index: int
    label: str
    name: str
    min_mm: list[float]
    max_mm: list[float]
    size_mm: list[float]
    volume_cm3: float
    mass_estimates_g: dict[str, float]
    default_material_mass_g: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PlanarFaceGroup:
    """Axis-aligned planar face cluster grouped by (±axis, coordinate)."""

    axis: str
    axis_index: int
    sign: int
    coord_mm: float
    area_mm2: float
    face_count: int
    center_mm: list[float]
    bounds_mm: list[list[float]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CylindricalFeature:
    """Detected cylindrical B-Rep hole or boss."""

    kind: str
    radius_mm: float
    diameter_mm: float
    center_mm: list[float]
    axis: list[float]
    area_mm2: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StepInspectReport:
    """Complete structural & geometric inspection report for a STEP or mesh file."""

    path: str
    file_format: str
    default_material: str
    solids: list[SolidInspectInfo] = field(default_factory=list)
    total_volume_cm3: float = 0.0
    total_mass_estimates_g: dict[str, float] = field(default_factory=dict)
    default_material_mass_g: float = 0.0
    overall_min_mm: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    overall_max_mm: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    overall_size_mm: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    planar_faces: list[PlanarFaceGroup] = field(default_factory=list)
    cylindrical_features: list[CylindricalFeature] = field(default_factory=list)
    overlapping_pairs: list[dict[str, Any]] = field(default_factory=list)
    suggested_ports: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "file_format": self.file_format,
            "default_material": self.default_material,
            "solids": [s.to_dict() for s in self.solids],
            "total_volume_cm3": self.total_volume_cm3,
            "total_mass_estimates_g": dict(self.total_mass_estimates_g),
            "default_material_mass_g": self.default_material_mass_g,
            "overall_min_mm": list(self.overall_min_mm),
            "overall_max_mm": list(self.overall_max_mm),
            "overall_size_mm": list(self.overall_size_mm),
            "planar_faces": [f.to_dict() for f in self.planar_faces],
            "cylindrical_features": [c.to_dict() for c in self.cylindrical_features],
            "overlapping_pairs": list(self.overlapping_pairs),
            "suggested_ports": dict(self.suggested_ports),
        }

    def to_markdown(self) -> str:
        """Render a compact engineering summary table (strictly <= 50 lines)."""
        lines: list[str] = []
        name = Path(self.path).name
        sx, sy, sz = self.overall_size_mm
        lines.append(
            f"### CAD Inspect: `{name}` ({self.file_format.upper()}, "
            f"{len(self.solids)} solid{'s' if len(self.solids) != 1 else ''})"
        )
        lines.append(
            f"- **Size (mm):** `{sx:.2f} x {sy:.2f} x {sz:.2f}` | "
            f"**Min/Max:** `{[round(v, 2) for v in self.overall_min_mm]}` .. "
            f"`{[round(v, 2) for v in self.overall_max_mm]}`"
        )
        m = self.total_mass_estimates_g
        lines.append(
            f"- **Vol:** `{self.total_volume_cm3:.3f} cm³` | "
            f"**Mass ({self.default_material}):** `{self.default_material_mass_g:.2f} g` | "
            f"Al6061 `{m.get('aluminum_6061', 0.0):.2f}g`, Ti64 `{m.get('titanium_grade5', 0.0):.2f}g`, "
            f"Steel `{m.get('steel', 0.0):.2f}g`, ABS/PC `{m.get('abs_pc', 0.0):.2f}g`"
        )

        lines.append("| # | Solid | Size (mm) | Vol (cm³) | Default Mass (g) |")
        lines.append("|---:|---|---:|---:|---:|")
        for s in self.solids[:8]:
            lbl = s.label or s.name
            sz_s = f"{s.size_mm[0]:.1f}×{s.size_mm[1]:.1f}×{s.size_mm[2]:.1f}"
            lines.append(
                f"| {s.index} | `{lbl}` | {sz_s} | {s.volume_cm3:.3f} | {s.default_material_mass_g:.2f} |"
            )
        if len(self.solids) > 8:
            lines.append(f"| … | *+{len(self.solids) - 8} more solids* | | | |")

        if self.planar_faces:
            lines.append("| Axis | Coord (mm) | Area (mm²) | Faces |")
            lines.append("|---|---:|---:|---:|")
            for pf in self.planar_faces[:5]:
                lines.append(
                    f"| {pf.axis} | {pf.coord_mm:.3f} | {pf.area_mm2:.1f} | {pf.face_count} |"
                )

        if self.cylindrical_features:
            lines.append("| Feature | Ø (mm) | Center (mm) | Axis |")
            lines.append("|---|---:|---|---|")
            for cf in self.cylindrical_features[:5]:
                c_s = f"[{cf.center_mm[0]:.1f}, {cf.center_mm[1]:.1f}, {cf.center_mm[2]:.1f}]"
                a_s = f"[{cf.axis[0]:.2f}, {cf.axis[1]:.2f}, {cf.axis[2]:.2f}]"
                lines.append(f"| {cf.kind} | {cf.diameter_mm:.2f} | `{c_s}` | `{a_s}` |")

        if self.overlapping_pairs:
            ov_s = ", ".join(
                f"({p['solid_a']},{p['solid_b']})" for p in self.overlapping_pairs[:5]
            )
            lines.append(f"- **Overlaps ({len(self.overlapping_pairs)}):** {ov_s}")
        else:
            lines.append("- **Overlaps:** None")

        for pname in ("port0", "port1"):
            ps = self.suggested_ports.get(pname)
            if ps:
                lines.append(
                    f"- **Suggested `{pname}`:** `{ps.get('kind')}` @ "
                    f"`{ps.get('vp_coord_mm')}` ({ps.get('reason', '')})"
                )

        if len(lines) > 50:
            lines = lines[:49] + ["- *(truncated to 50 lines)*"]
        return "\n".join(lines)


def _resolve_default_density_g_cm3(material_id: str) -> float:
    try:
        mat = get_material(material_id)
        return float(mat.density_kg_m3) / 1000.0
    except KeyError:
        return MASS_DENSITIES_G_CM3.get(material_id, 4.43)


def _mass_estimates_for_vol(vol_cm3: float) -> dict[str, float]:
    return {
        key: round(float(vol_cm3) * rho, 3)
        for key, rho in MASS_DENSITIES_G_CM3.items()
    }


def _canonical_unit_axis(vec: tuple[float, float, float]) -> list[float]:
    arr = np.asarray(vec, dtype=np.float64)
    n = float(np.linalg.norm(arr))
    if n <= 1e-12:
        return [0.0, 0.0, 1.0]
    u = arr / n
    # Canonicalize sign so dominant component is positive
    dom = int(np.argmax(np.abs(u)))
    if u[dom] < 0.0:
        u = -u
    return [round(float(x), 4) for x in u]


def _collect_step_solid_labels(root_shape: Any) -> list[tuple[Any, str]]:
    """Collect solids from an imported build123d Shape along with any compound/solid labels."""
    results: list[tuple[Any, str]] = []
    seen_wrapped: list[Any] = []

    def _walk(node: Any, inherited_label: str) -> None:
        own_lbl = str(getattr(node, "label", "") or "").strip()
        lbl = own_lbl or inherited_label
        children = getattr(node, "children", None)
        if children:
            for ch in children:
                _walk(ch, lbl)
            return
        solids_attr = getattr(node, "solids", None)
        if callable(solids_attr):
            for s in solids_attr():
                s_lbl = str(getattr(s, "label", "") or "").strip() or lbl
                w = getattr(s, "wrapped", s)
                is_dup = False
                for prev in seen_wrapped:
                    try:
                        if prev is w or (hasattr(w, "IsSame") and w.IsSame(prev)):
                            is_dup = True
                            break
                    except Exception:
                        pass
                if not is_dup:
                    seen_wrapped.append(w)
                    results.append((s, s_lbl))

    _walk(root_shape, "")
    if not results:
        solids_attr = getattr(root_shape, "solids", None)
        if callable(solids_attr):
            for s in solids_attr():
                results.append((s, str(getattr(s, "label", "") or "").strip()))
    return results


def _cluster_planar_faces(
    raw_planes: list[tuple[int, int, str, float, float, list[float], list[float], list[float]]],
) -> list[PlanarFaceGroup]:
    """Group planar faces by (axis_label, 0.05 mm quantized coordinate) and return top 8."""
    clusters: dict[tuple[str, int], dict[str, Any]] = {}
    for ax_idx, sign, ax_lbl, coord_mm, area_mm2, center_mm, fmin, fmax in raw_planes:
        bucket = int(round(coord_mm / 0.05))
        key = (ax_lbl, bucket)
        entry = clusters.get(key)
        if entry is None:
            clusters[key] = {
                "axis": ax_lbl,
                "axis_index": ax_idx,
                "sign": sign,
                "w_coord": coord_mm * area_mm2,
                "w_center": np.asarray(center_mm, dtype=np.float64) * area_mm2,
                "area": area_mm2,
                "count": 1,
                "bmin": np.asarray(fmin, dtype=np.float64),
                "bmax": np.asarray(fmax, dtype=np.float64),
            }
        else:
            entry["w_coord"] += coord_mm * area_mm2
            entry["w_center"] += np.asarray(center_mm, dtype=np.float64) * area_mm2
            entry["area"] += area_mm2
            entry["count"] += 1
            entry["bmin"] = np.minimum(entry["bmin"], np.asarray(fmin, dtype=np.float64))
            entry["bmax"] = np.maximum(entry["bmax"], np.asarray(fmax, dtype=np.float64))

    groups: list[PlanarFaceGroup] = []
    for entry in clusters.values():
        tot_a = max(float(entry["area"]), 1e-12)
        wc = entry["w_center"] / tot_a
        bmin = entry["bmin"]
        bmax = entry["bmax"]
        groups.append(
            PlanarFaceGroup(
                axis=str(entry["axis"]),
                axis_index=int(entry["axis_index"]),
                sign=int(entry["sign"]),
                coord_mm=round(float(entry["w_coord"] / tot_a), 3),
                area_mm2=round(float(entry["area"]), 2),
                face_count=int(entry["count"]),
                center_mm=[round(float(wc[0]), 3), round(float(wc[1]), 3), round(float(wc[2]), 3)],
                bounds_mm=[
                    [round(float(bmin[0]), 3), round(float(bmax[0]), 3)],
                    [round(float(bmin[1]), 3), round(float(bmax[1]), 3)],
                    [round(float(bmin[2]), 3), round(float(bmax[2]), 3)],
                ],
            )
        )
    groups.sort(key=lambda g: g.area_mm2, reverse=True)
    return groups[:8]


def _extract_brep_features(
    solids: list[Any],
) -> tuple[list[PlanarFaceGroup], list[CylindricalFeature]]:
    """Extract major axis-aligned planar faces (top 8) & cylindrical holes/bosses (top 16)."""
    import build123d as b3d

    try:
        from OCP.BRepAdaptor import BRepAdaptor_Surface
    except Exception:
        BRepAdaptor_Surface = None  # type: ignore[assignment]

    raw_planes: list[tuple[int, int, str, float, float, list[float], list[float], list[float]]] = []
    raw_cyls: list[dict[str, Any]] = []

    for solid in solids:
        try:
            faces = solid.faces()
        except Exception:
            continue

        for face in faces:
            try:
                area_mm2 = float(face.area)
                if not math.isfinite(area_mm2) or area_mm2 <= 1e-4:
                    continue
                gtype = face.geom_type
            except Exception:
                continue

            if gtype == b3d.GeomType.PLANE:
                try:
                    n = face.normal_at()
                    n_arr = np.array([float(n.X), float(n.Y), float(n.Z)], dtype=np.float64)
                    n_norm = float(np.linalg.norm(n_arr))
                    if n_norm <= 1e-12:
                        continue
                    n_arr /= n_norm
                    ax_idx = int(np.argmax(np.abs(n_arr)))
                    if abs(n_arr[ax_idx]) < 0.985:
                        continue
                    sign = 1 if n_arr[ax_idx] >= 0.0 else -1
                    ax_lbl = f"{'+' if sign > 0 else '-'}{'XYZ'[ax_idx]}"
                    c = face.center()
                    c_list = [float(c.X), float(c.Y), float(c.Z)]
                    bb = face.bounding_box()
                    fmin = [float(bb.min.X), float(bb.min.Y), float(bb.min.Z)]
                    fmax = [float(bb.max.X), float(bb.max.Y), float(bb.max.Z)]
                    raw_planes.append(
                        (ax_idx, sign, ax_lbl, c_list[ax_idx], area_mm2, c_list, fmin, fmax)
                    )
                except Exception:
                    continue

            elif gtype == b3d.GeomType.CYLINDER:
                try:
                    radius_mm: float | None = None
                    ax_vec: np.ndarray | None = None
                    loc_vec: np.ndarray | None = None

                    if BRepAdaptor_Surface is not None:
                        adaptor = BRepAdaptor_Surface(face.wrapped)
                        cyl = adaptor.Cylinder()
                        radius_mm = float(cyl.Radius())
                        ax_d = cyl.Axis().Direction()
                        loc_p = cyl.Location()
                        ax_vec = np.array([ax_d.X(), ax_d.Y(), ax_d.Z()], dtype=np.float64)
                        loc_vec = np.array([loc_p.X(), loc_p.Y(), loc_p.Z()], dtype=np.float64)

                    if radius_mm is None:
                        r_val = getattr(face, "radius", None)
                        if r_val is None:
                            continue
                        radius_mm = float(r_val)
                    if not math.isfinite(radius_mm) or radius_mm <= 1e-4:
                        continue

                    bb_c = face.center(b3d.CenterOf.BOUNDING_BOX)
                    bb_center = np.array([float(bb_c.X), float(bb_c.Y), float(bb_c.Z)], dtype=np.float64)

                    if ax_vec is None:
                        try:
                            aor = face.axis_of_rotation
                            ax_vec = np.array(
                                [float(aor.direction.X), float(aor.direction.Y), float(aor.direction.Z)],
                                dtype=np.float64,
                            )
                            loc_vec = np.array(
                                [float(aor.position.X), float(aor.position.Y), float(aor.position.Z)],
                                dtype=np.float64,
                            )
                        except Exception:
                            ax_vec = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                            loc_vec = bb_center.copy()

                    ax_norm = float(np.linalg.norm(ax_vec))
                    if ax_norm <= 1e-12:
                        continue
                    ax_u = ax_vec / ax_norm
                    assert loc_vec is not None
                    proj_t = float(np.dot(bb_center - loc_vec, ax_u))
                    center_on_axis = loc_vec + proj_t * ax_u

                    kind = "cylinder"
                    try:
                        surf_pt = face.center()
                        surf_arr = np.array([float(surf_pt.X), float(surf_pt.Y), float(surf_pt.Z)], dtype=np.float64)
                        surf_normal = face.normal_at()
                        sn_arr = np.array([float(surf_normal.X), float(surf_normal.Y), float(surf_normal.Z)], dtype=np.float64)
                        radial = surf_arr - (loc_vec + float(np.dot(surf_arr - loc_vec, ax_u)) * ax_u)
                        if float(np.linalg.norm(radial)) > 1e-6:
                            dot_r = float(np.dot(sn_arr, radial))
                            if dot_r < -1e-6:
                                kind = "hole"
                            elif dot_r > 1e-6:
                                kind = "boss"
                    except Exception:
                        pass

                    canon_ax = _canonical_unit_axis((float(ax_u[0]), float(ax_u[1]), float(ax_u[2])))
                    raw_cyls.append(
                        {
                            "kind": kind,
                            "radius_mm": radius_mm,
                            "center": center_on_axis,
                            "axis": np.asarray(canon_ax, dtype=np.float64),
                            "area_mm2": area_mm2,
                        }
                    )
                except Exception:
                    continue

    # Merge split half-cylinder faces belonging to the same coaxial hole/boss
    merged: list[dict[str, Any]] = []
    for item in raw_cyls:
        matched = False
        for m in merged:
            if abs(m["radius_mm"] - item["radius_mm"]) > 0.02:
                continue
            if abs(float(np.dot(m["axis"], item["axis"]))) < 0.995:
                continue
            diff = item["center"] - m["center"]
            perp = diff - float(np.dot(diff, m["axis"])) * m["axis"]
            if float(np.linalg.norm(perp)) <= 0.05 and abs(float(np.dot(diff, m["axis"]))) <= 0.25:
                tot_a = m["area_mm2"] + item["area_mm2"]
                m["center"] = (m["center"] * m["area_mm2"] + item["center"] * item["area_mm2"]) / max(tot_a, 1e-12)
                m["area_mm2"] = tot_a
                if m["kind"] == "cylinder" and item["kind"] != "cylinder":
                    m["kind"] = item["kind"]
                matched = True
                break
        if not matched:
            merged.append(dict(item))

    merged.sort(key=lambda d: d["area_mm2"], reverse=True)
    cyl_features = [
        CylindricalFeature(
            kind=str(d["kind"]),
            radius_mm=round(float(d["radius_mm"]), 4),
            diameter_mm=round(2.0 * float(d["radius_mm"]), 4),
            center_mm=[round(float(v), 3) for v in d["center"]],
            axis=[round(float(v), 4) for v in d["axis"]],
            area_mm2=round(float(d["area_mm2"]), 2),
        )
        for d in merged[:16]
    ]
    return _cluster_planar_faces(raw_planes), cyl_features


def _detect_overlapping_solids(
    solids: list[Any],
    solid_infos: list[SolidInspectInfo],
) -> list[dict[str, Any]]:
    """Detect overlapping solid pairs using AABB check + B-Rep intersection if <= 24 solids."""
    n = len(solids)
    if n < 2:
        return []

    check_brep_vol = n <= 24
    overlaps: list[dict[str, Any]] = []
    tol = 0.01

    for i in range(n):
        min_a = solid_infos[i].min_mm
        max_a = solid_infos[i].max_mm
        for j in range(i + 1, n):
            min_b = solid_infos[j].min_mm
            max_b = solid_infos[j].max_mm
            if (
                min_a[0] >= max_b[0] - tol
                or min_b[0] >= max_a[0] - tol
                or min_a[1] >= max_b[1] - tol
                or min_b[1] >= max_a[1] - tol
                or min_a[2] >= max_b[2] - tol
                or min_b[2] >= max_a[2] - tol
            ):
                continue

            inter_cm3: float | None = None
            if check_brep_vol:
                try:
                    inter = solids[i].intersect(solids[j])
                    if inter is None:
                        continue
                    solids_fn = getattr(inter, "solids", None)
                    inter_parts = solids_fn() if callable(solids_fn) else [inter]
                    vol_mm3 = sum(float(getattr(p, "volume", 0.0)) for p in inter_parts)
                    if vol_mm3 <= 1e-3:
                        continue
                    inter_cm3 = round(vol_mm3 / 1000.0, 4)
                except Exception:
                    inter_cm3 = None

            overlaps.append(
                {
                    "solid_a": i,
                    "solid_b": j,
                    "label_a": solid_infos[i].label or solid_infos[i].name,
                    "label_b": solid_infos[j].label or solid_infos[j].name,
                    "intersection_volume_cm3": inter_cm3,
                }
            )
    return overlaps


def suggest_ports_from_inspection(
    overall_min_mm: list[float],
    overall_max_mm: list[float],
    planar_faces: list[PlanarFaceGroup],
) -> dict[str, dict[str, Any]]:
    """Suggest Port 0 (mount face near Z_max or Y_min) and Port 1 (tip/output near Z_min)."""
    mins = np.asarray(overall_min_mm, dtype=np.float64)
    maxs = np.asarray(overall_max_mm, dtype=np.float64)
    sizes = np.maximum(maxs - mins, 1e-6)
    cx, cy, cz = (0.5 * (mins + maxs)).tolist()

    z_tol = max(0.5, 0.08 * float(sizes[2]))
    y_tol = max(0.5, 0.08 * float(sizes[1]))

    # Port 0: largest planar face near Z_max (+Z) or Y_min (-Y)
    p0_candidates = [
        pf
        for pf in planar_faces
        if (pf.axis == "+Z" and pf.coord_mm >= float(maxs[2]) - z_tol)
        or (pf.axis == "-Y" and pf.coord_mm <= float(mins[1]) + y_tol)
    ]
    if not p0_candidates and planar_faces:
        p0_candidates = [planar_faces[0]]

    if p0_candidates:
        best0 = max(p0_candidates, key=lambda pf: pf.area_mm2)
        vp0 = list(best0.center_mm)
        vp0[best0.axis_index] = best0.coord_mm
        port0_sug: dict[str, Any] = {
            "kind": "plane_box",
            "vp_coord_mm": [round(float(v), 3) for v in vp0],
            "params": {
                "axis": best0.axis_index,
                "coord_mm": round(float(best0.coord_mm), 3),
                "tol_mm": 0.45,
            },
            "reason": f"largest planar face near {best0.axis} ({best0.area_mm2:.1f} mm²)",
        }
    else:
        port0_sug = {
            "kind": "plane_box",
            "vp_coord_mm": [round(cx, 3), round(cy, 3), round(float(maxs[2]), 3)],
            "params": {
                "axis": 2,
                "coord_mm": round(float(maxs[2]), 3),
                "tol_mm": 0.45,
            },
            "reason": "Z_max bounding face fallback",
        }

    # Port 1: tip/output near Z_min
    p1_candidates = [
        pf
        for pf in planar_faces
        if pf.axis == "-Z" and pf.coord_mm <= float(mins[2]) + z_tol
    ]
    if p1_candidates:
        best1 = max(p1_candidates, key=lambda pf: pf.area_mm2)
        # Guard against picking the exact same plane as port0 on thin X/Y parts
        if not (
            port0_sug["params"].get("axis") == 2
            and abs(float(port0_sug["params"].get("coord_mm", 0.0)) - best1.coord_mm) < 0.1
        ):
            vp1 = list(best1.center_mm)
            vp1[2] = best1.coord_mm
            port1_sug: dict[str, Any] = {
                "kind": "plane_box",
                "vp_coord_mm": [round(float(v), 3) for v in vp1],
                "params": {
                    "axis": 2,
                    "coord_mm": round(float(best1.coord_mm), 3),
                    "tol_mm": 0.45,
                },
                "reason": f"planar -Z tip/output face ({best1.area_mm2:.1f} mm²)",
            }
            return {"port0": port0_sug, "port1": port1_sug}

    z_cap_height = max(0.8, min(3.0, 0.06 * float(sizes[2])))
    r_max = round(max(float(sizes[0]), float(sizes[1])) * 0.65, 3)
    port1_sug = {
        "kind": "z_cap",
        "vp_coord_mm": [round(cx, 3), round(cy, 3), round(float(mins[2]), 3)],
        "params": {
            "z_min_mm": round(float(mins[2]) - 0.1, 3),
            "z_max_mm": round(float(mins[2]) + z_cap_height, 3),
            "center_xy_mm": [round(cx, 3), round(cy, 3)],
            "r_max_mm": r_max,
        },
        "reason": "Z_min tip/output cap",
    }
    return {"port0": port0_sug, "port1": port1_sug}


def _inspect_trimesh_fallback(
    path: Path,
    default_material: str,
    default_rho: float,
) -> StepInspectReport:
    """Inspect an `.stl`, `.glb`, `.gltf`, or `.obj` mesh file via ``trimesh``."""
    loaded = trimesh.load(str(path))
    meshes: list[tuple[str, trimesh.Trimesh]] = []
    if isinstance(loaded, trimesh.Scene):
        for node_name in loaded.graph.nodes_geometry:
            mat, geom_name = loaded.graph[node_name]
            geom = loaded.geometry.get(geom_name)
            if isinstance(geom, trimesh.Trimesh) and len(geom.faces) > 0:
                tm = geom.copy()
                tm.apply_transform(mat)
                meshes.append((str(node_name), tm))
    elif isinstance(loaded, trimesh.Trimesh):
        meshes.append((path.stem, loaded))

    if not meshes:
        raise ValueError(f"No triangle meshes found in {path}")

    # GLB/glTF in PrintSim is in SI meters -> convert to mm for inspection if extent < 5.0
    scale_to_mm = 1.0
    if path.suffix.lower() in {".glb", ".gltf"}:
        all_pts = np.vstack([m.bounds for _, m in meshes])
        ext = all_pts.max(axis=0) - all_pts.min(axis=0)
        if float(np.max(ext)) < 5.0:
            scale_to_mm = 1000.0

    solid_infos: list[SolidInspectInfo] = []
    raw_planes: list[tuple[int, int, str, float, float, list[float], list[float], list[float]]] = []
    ov_mins = np.full(3, np.inf, dtype=np.float64)
    ov_maxs = np.full(3, -np.inf, dtype=np.float64)

    for idx, (lbl, tm_orig) in enumerate(meshes):
        tm = tm_orig.copy()
        if scale_to_mm != 1.0:
            tm.apply_scale(scale_to_mm)
        bmin = tm.bounds[0]
        bmax = tm.bounds[1]
        ext = bmax - bmin
        ov_mins = np.minimum(ov_mins, bmin)
        ov_maxs = np.maximum(ov_maxs, bmax)

        vol_mm3 = abs(float(tm.volume)) if tm.is_volume else float(np.prod(ext) * 0.35)
        vol_cm3 = vol_mm3 / 1000.0
        solid_infos.append(
            SolidInspectInfo(
                index=idx,
                label=lbl,
                name=lbl or f"solid_{idx}",
                min_mm=[round(float(v), 3) for v in bmin],
                max_mm=[round(float(v), 3) for v in bmax],
                size_mm=[round(float(v), 3) for v in ext],
                volume_cm3=round(vol_cm3, 4),
                mass_estimates_g=_mass_estimates_for_vol(vol_cm3),
                default_material_mass_g=round(vol_cm3 * default_rho, 3),
            )
        )

        # Group axis-aligned planar triangles into synthetic planar face clusters
        fn = np.asarray(tm.face_normals, dtype=np.float64)
        fa = np.asarray(tm.area_faces, dtype=np.float64)
        tris = np.asarray(tm.triangles, dtype=np.float64)
        if len(fn) > 0:
            centroids = tris.mean(axis=1)
            for ax_idx, sign, ax_lbl in _AXIS_LABELS:
                mask = (fn[:, ax_idx] * sign) >= 0.985
                if not np.any(mask):
                    continue
                coords = centroids[mask, ax_idx]
                buckets = np.round(coords / 0.05).astype(np.int64)
                for b_id in np.unique(buckets):
                    sub_mask = buckets == b_id
                    sub_area = float(np.sum(fa[mask][sub_mask]))
                    if sub_area <= 1e-3:
                        continue
                    sub_tris = tris[mask][sub_mask].reshape(-1, 3)
                    sub_c = np.average(centroids[mask][sub_mask], axis=0, weights=fa[mask][sub_mask])
                    sub_min = sub_tris.min(axis=0)
                    sub_max = sub_tris.max(axis=0)
                    raw_planes.append(
                        (
                            ax_idx,
                            sign,
                            ax_lbl,
                            float(sub_c[ax_idx]),
                            sub_area,
                            sub_c.tolist(),
                            sub_min.tolist(),
                            sub_max.tolist(),
                        )
                    )

    total_vol_cm3 = round(sum(s.volume_cm3 for s in solid_infos), 4)
    planar_faces = _cluster_planar_faces(raw_planes)
    ov_min_l = [round(float(v), 3) for v in ov_mins]
    ov_max_l = [round(float(v), 3) for v in ov_maxs]
    ov_size_l = [round(float(v), 3) for v in (ov_maxs - ov_mins)]
    suggested_ports = suggest_ports_from_inspection(ov_min_l, ov_max_l, planar_faces)

    return StepInspectReport(
        path=str(path),
        file_format=path.suffix.lstrip(".").lower(),
        default_material=default_material,
        solids=solid_infos,
        total_volume_cm3=total_vol_cm3,
        total_mass_estimates_g=_mass_estimates_for_vol(total_vol_cm3),
        default_material_mass_g=round(total_vol_cm3 * default_rho, 3),
        overall_min_mm=ov_min_l,
        overall_max_mm=ov_max_l,
        overall_size_mm=ov_size_l,
        planar_faces=planar_faces,
        cylindrical_features=[],
        overlapping_pairs=[],
        suggested_ports=suggested_ports,
    )


def inspect_step_file(
    path: str | Path,
    default_material: str = "titanium_grade5",
) -> StepInspectReport:
    """Inspect a STEP (or STL/GLB fallback) CAD file and return a structured report."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"CAD file not found: {p}")

    default_rho = _resolve_default_density_g_cm3(default_material)
    suffix = p.suffix.lower()
    if suffix not in {".step", ".stp"}:
        return _inspect_trimesh_fallback(p, default_material, default_rho)

    import build123d as b3d

    imported = b3d.import_step(str(p))
    labeled_solids = _collect_step_solid_labels(imported)
    if not labeled_solids:
        raise ValueError(f"No 3D solids found in STEP file: {p}")

    solids = [s for s, _ in labeled_solids]
    solid_infos: list[SolidInspectInfo] = []
    ov_mins = np.full(3, np.inf, dtype=np.float64)
    ov_maxs = np.full(3, -np.inf, dtype=np.float64)

    for idx, (solid, lbl) in enumerate(labeled_solids):
        bb = solid.bounding_box()
        bmin = np.array([float(bb.min.X), float(bb.min.Y), float(bb.min.Z)], dtype=np.float64)
        bmax = np.array([float(bb.max.X), float(bb.max.Y), float(bb.max.Z)], dtype=np.float64)
        ext = bmax - bmin
        ov_mins = np.minimum(ov_mins, bmin)
        ov_maxs = np.maximum(ov_maxs, bmax)

        try:
            vol_mm3 = abs(float(solid.volume))
        except Exception:
            vol_mm3 = 0.0
        vol_cm3 = vol_mm3 / 1000.0
        solid_infos.append(
            SolidInspectInfo(
                index=idx,
                label=lbl,
                name=lbl or f"solid_{idx}",
                min_mm=[round(float(v), 3) for v in bmin],
                max_mm=[round(float(v), 3) for v in bmax],
                size_mm=[round(float(v), 3) for v in ext],
                volume_cm3=round(vol_cm3, 4),
                mass_estimates_g=_mass_estimates_for_vol(vol_cm3),
                default_material_mass_g=round(vol_cm3 * default_rho, 3),
            )
        )

    planar_faces, cyl_features = _extract_brep_features(solids)
    overlaps = _detect_overlapping_solids(solids, solid_infos)

    total_vol_cm3 = round(sum(s.volume_cm3 for s in solid_infos), 4)
    ov_min_l = [round(float(v), 3) for v in ov_mins]
    ov_max_l = [round(float(v), 3) for v in ov_maxs]
    ov_size_l = [round(float(v), 3) for v in (ov_maxs - ov_mins)]
    suggested_ports = suggest_ports_from_inspection(ov_min_l, ov_max_l, planar_faces)

    return StepInspectReport(
        path=str(p),
        file_format="step",
        default_material=default_material,
        solids=solid_infos,
        total_volume_cm3=total_vol_cm3,
        total_mass_estimates_g=_mass_estimates_for_vol(total_vol_cm3),
        default_material_mass_g=round(total_vol_cm3 * default_rho, 3),
        overall_min_mm=ov_min_l,
        overall_max_mm=ov_max_l,
        overall_size_mm=ov_size_l,
        planar_faces=planar_faces,
        cylindrical_features=cyl_features,
        overlapping_pairs=overlaps,
        suggested_ports=suggested_ports,
    )
