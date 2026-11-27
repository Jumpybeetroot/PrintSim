#!/usr/bin/env python3
"""Bake the VzBot 330 gantry hardware kit into ``src/printsim/assets/hardware/gantry_brackets/``.

Source: official VzBoT3D/VzBoT-Vz330 full-printer CAD ``CAD/Full Printer/Vz-330_v1.2.zip``
(``Vz-330_WEB v2.step``, 409 MB, Z-up, -Y = front). Every station is extracted in exact
assembled world coordinates and re-anchored to the PrintSim frame-corner datum
(``docs/CAD_AND_FEA_PITFALLS.md`` Pitfall 1F):

* Frame: 530 x 500 x 570 mm outer; left/right top Y 2020 centerlines at X = -/+255, front/rear
  2040-upright centers at Y = -175 / +285, gantry plane = top-ring centerline Z = +45.
* X beam: 20x20x1 CF tube axis at Y = +58.26, Z = +45 (MGN9H rail on top, Z = 55..61.5).

Stations (``(0, 0, 0)`` = frame corner centerline at the gantry plane, or Y-extrusion
centerline at the X-beam axis for the XY joints):

* ``vzbot_330_rear_{left,right}``      Motor mounts with integrated tensioners (parts A/B + motor plate)
* ``vzbot_330_awd_front_{left,right}`` AWD front idler / motor mounts (pulley mount top + bottom)
* ``vzbot_330_2wd_front_{left,right}`` 2WD front idler mounts (two stacked 20T idlers)
* ``vzbot_330_xy_{left,right}``        Y-gantry XY joints (gantry bottom + top) clamping the CF tube

Port 0 = frame / Y-carriage bolted contact faces, Port 1 = motor-shaft / idler-shaft bores or the
CF-tube clamp pocket (disjoint node clouds, Pitfall 1G). The v1.2 CAD models the printed gantry;
the Mellow CNC 6061 kit shares the same envelope, so assets are baked at the aluminium 6061-T6
reference and rescaled at load time to ``gantry.bracket_material`` (``cad_trust_pct`` 72).

Usage::

    uv run python scripts/bake_vzbot_kit.py --step "cad_sources/vzbot/Vz-330_WEB v2.step"
    uv run python scripts/bake_vzbot_kit.py --parts .cache/vzbot/vz_parts.pkl   # cached solids
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
import trimesh

from printsim.cad_assets import (
    ASSETS_ROOT,
    BakedAssetSpec,
    MaterialRegion,
    PortDefinition,
    VisualPart,
    bake_solid_superelement,
    build_visual_glb,
    clear_asset_cache,
)

SOURCE_URL = "https://github.com/VzBoT3D/VzBoT-Vz330/tree/master/CAD/Full%20Printer"
SOURCE_CAD = "Vz-330_v1.2.zip / Vz-330_WEB v2.step"
ROOT_LABEL = "Vz-330_WEB_v2"
GANTRY = f"{ROOT_LABEL}/gantry_2WD_+_AWD/"
DEFAULT_CACHE = Path(".cache/vzbot/vz_parts.pkl")

X_LEFT_CL = -255.0
X_RIGHT_CL = 255.0
Y_FRONT_CL = -175.0
Y_REAR_CL = 285.0
Y_XBEAM = 58.26
Z_GANTRY = 45.0

COL_VZ_RED = "#d7262e"
COL_CNC_DARK = "#2b313b"
COL_MOTOR = "#1b202a"
COL_PULLEY = "#cfd7e2"
COL_STEEL = "#9ca8b8"
COL_TI = "#b8c2ce"

REFERENCE_MATERIAL = "aluminum_6061_t6"
CAD_TRUST_PCT = 84
FORCE_REBAKE = False  # set by --force; otherwise stations with a complete bundle are skipped
VISUALS_ONLY = False  # set by --visuals-only; rebuild visual.glb of complete bundles without re-meshing


# ---------------------------------------------------------------------------
# CAD extraction (slow: ~5 min for the 409 MB STEP) with pickle cache
# ---------------------------------------------------------------------------


def extract_parts(step_path: Path, cache: Path) -> list[dict[str, Any]]:
    """Tessellate every gantry solid of the full VzBot STEP in world coordinates (mm)."""
    import build123d as b3d

    from bake_voron_kits import _shape_to_trimesh  # type: ignore[import-not-found]

    keep = (f"{ROOT_LABEL}/gantry_2WD_+_AWD", f"{ROOT_LABEL}/Vz-Printhead_Printed_MGN9-revised_v24")
    root = b3d.import_step(str(step_path))
    out: list[dict[str, Any]] = []

    def walk(node: Any, prefix: str = "", loc: Any = None) -> None:
        lbl = (getattr(node, "label", "") or "").strip()
        path = f"{prefix}/{lbl}" if prefix else lbl
        loc = b3d.Location() if loc is None else loc
        children = getattr(node, "children", None) or []
        if children:
            if not any(path.startswith(k) or k.startswith(path) for k in keep):
                return
            nl = getattr(node, "location", None)
            nxt = loc * nl if nl is not None else loc
            for ch in children:
                walk(ch, path, nxt)
            return
        if not any(path.startswith(k) for k in keep):
            return
        for i, solid in enumerate(node.solids() if hasattr(node, "solids") else []):
            s2 = solid.moved(loc)
            tm = _shape_to_trimesh(s2, deflection=0.35)
            try:
                com = s2.center(b3d.CenterOf.MASS)
                com_xyz = [com.X, com.Y, com.Z]
                vol = float(s2.volume)
            except Exception:
                com_xyz, vol = None, None
            out.append(
                {
                    "path": path,
                    "i": i,
                    "vol_mm3": vol,
                    "com": com_xyz,
                    "v": np.asarray(tm.vertices, dtype=np.float32),
                    "f": np.asarray(tm.faces, dtype=np.int32),
                }
            )

    walk(root)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as fh:
        pickle.dump(out, fh)
    return out


def load_parts(parts_path: Path | None, step_path: Path | None) -> list[dict[str, Any]]:
    if parts_path is not None and parts_path.is_file():
        with parts_path.open("rb") as fh:
            return pickle.load(fh)
    if step_path is None or not step_path.is_file():
        raise FileNotFoundError("Provide --parts <cached pkl> or --step <Vz-330_WEB v2.step>")
    return extract_parts(step_path, parts_path or DEFAULT_CACHE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mesh(part: dict[str, Any], origin: np.ndarray) -> trimesh.Trimesh:
    m = trimesh.Trimesh(vertices=part["v"].astype(np.float64), faces=part["f"], process=False)
    m.apply_translation(-origin)
    return m


def _leaf(path: str) -> str:
    return path.rsplit("/", 1)[-1].lower()


def _style(path: str, ext: np.ndarray) -> tuple[str, float, float]:
    leaf = _leaf(path)
    lo = path.lower()
    if "nema17" in lo or "stepper" in lo:
        return COL_MOTOR, 0.55, 0.42
    if "sliding_block" in lo or "mgn" in leaf:
        return COL_STEEL, 0.88, 0.24
    if "titanium" in leaf or "shoulder_bolt" in leaf:
        return COL_TI, 0.86, 0.30
    if "pulley" in lo or "gt2" in lo or "bearing" in leaf or "component" in leaf:
        return COL_PULLEY, 0.84, 0.26
    if max(ext[0], ext[1]) <= 14.0 or "screw" in leaf or "nut" in leaf or "bolt" in leaf:
        return COL_STEEL, 0.84, 0.30
    if "top" in leaf or "knob" in lo or "cap" in leaf:
        return COL_VZ_RED, 0.80, 0.34
    return COL_CNC_DARK, 0.82, 0.36


def _near_boxes(boxes: list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]], tol: float):
    """Select nodes lying on (within ``tol`` of) the surface of any axis-aligned frame-extrusion box."""

    def _sel(c: np.ndarray) -> np.ndarray:
        mask = np.zeros(len(c), dtype=bool)
        for (x0, x1), (y0, y1), (z0, z1) in boxes:
            dx = np.maximum(np.maximum(x0 - c[:, 0], c[:, 0] - x1), 0.0)
            dy = np.maximum(np.maximum(y0 - c[:, 1], c[:, 1] - y1), 0.0)
            dz = np.maximum(np.maximum(z0 - c[:, 2], c[:, 2] - z1), 0.0)
            mask |= np.sqrt(dx * dx + dy * dy + dz * dz) <= tol
        return mask

    return _sel


def _near_axes(axes_xy: list[tuple[float, float]], r_max: float, z_range: tuple[float, float]):
    """Select nodes within ``r_max`` of vertical shaft axes (motor shaft / idler shoulder bolts)."""

    def _sel(c: np.ndarray) -> np.ndarray:
        mask = np.zeros(len(c), dtype=bool)
        zin = (c[:, 2] >= z_range[0]) & (c[:, 2] <= z_range[1])
        for ax, ay in axes_xy:
            mask |= np.hypot(c[:, 0] - ax, c[:, 1] - ay) <= r_max
        return mask & zin

    return _sel


def _min_count(sel: Callable[[np.ndarray], np.ndarray], fallback_center: tuple[float, float, float], k: int = 16):
    def _wrapped(c: np.ndarray) -> np.ndarray:
        mask = sel(c)
        if int(mask.sum()) >= k:
            return mask
        d = np.linalg.norm(c - np.asarray(fallback_center), axis=1)
        out = np.zeros(len(c), dtype=bool)
        out[np.argsort(d)[:k]] = True
        return out

    return _wrapped


def _stratify_port(
    sel: Callable[[np.ndarray], np.ndarray],
    cell_mm: float = 7.5,
    min_nodes: int = 24,
) -> Callable[[np.ndarray], np.ndarray]:
    """Mirror-symmetrically stratify dense contact patches to keep RBE3 nullspace well-conditioned.

    A raw 100 x 20 mm extrusion contact face on a 1.6 mm voxel mesh contains ~2,000+ boundary nodes
    (>6,000 dense RBE3 nullspace DOFs in ``K_split``), which causes SuperLU roundoff drift in the
    6 free-free rigid-body modes (0.05-0.09 Hz). Selecting the single node closest to each ``cell_mm``
    voxel center preserves the full contact footprint & L/R mirror symmetry while tightening rigid-body
    residuals to < 0.015 Hz (well inside the < 0.05 Hz QA gate).
    """

    def _wrapped(c: np.ndarray) -> np.ndarray:
        mask = np.asarray(sel(c), dtype=bool)
        idx = np.flatnonzero(mask)
        if len(idx) <= max(min_nodes, 48):
            return mask
        pts = c[idx]
        # Mirror-symmetric grid about X=0, Y=0, Z=0
        cell_keys = np.round(pts / float(cell_mm)).astype(np.int64)
        cell_centers = cell_keys.astype(np.float64) * float(cell_mm)
        dist2 = np.sum((pts - cell_centers) ** 2, axis=1)
        # Sort by (cell_key, dist2) deterministically using symmetric |x| tie-breaking
        order = np.lexsort((np.abs(pts[:, 0]), pts[:, 1], pts[:, 2], dist2, cell_keys[:, 2], cell_keys[:, 1], cell_keys[:, 0]))
        sorted_keys = cell_keys[order]
        keep_first = np.ones(len(order), dtype=bool)
        keep_first[1:] = np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)
        chosen = idx[order[keep_first]]
        if len(chosen) < min_nodes:
            return mask
        out = np.zeros(len(c), dtype=bool)
        out[chosen] = True
        return out

    return _wrapped


def _bake_station(
    *,
    asset_id: str,
    name: str,
    parts: list[dict[str, Any]],
    origin: np.ndarray,
    is_structural: Callable[[str, np.ndarray], bool],
    port0: PortDefinition,
    port1: PortDefinition,
    mesh_size_mm: float,
    recipe: dict[str, Any],
    visual_only_parts: list[dict[str, Any]] | None = None,
) -> None:
    bundle = ASSETS_ROOT / "gantry_brackets" / asset_id
    complete = all(
        (bundle / f).is_file() for f in ("manifest.json", "superelement.npz", "visual.glb")
    )
    if complete and not FORCE_REBAKE and not VISUALS_ONLY:
        print(f"  [skip] {asset_id}: complete bundle already baked (use --force to re-bake)")
        return
    visual: list[VisualPart] = []
    structs: list[trimesh.Trimesh] = []
    for idx, part in enumerate(list(parts) + list(visual_only_parts or [])):
        is_visual_only = idx >= len(parts)
        m = _mesh(part, origin)
        if len(m.faces) == 0:
            continue
        ext = m.extents
        col, met, rough = _style(part["path"], ext)
        visual.append(VisualPart(m, (_leaf(part["path"])[:24] or f"part_{idx}"), col, metallic=met, roughness=rough))
        if not is_visual_only and is_structural(part["path"], ext):
            structs.append(m)
    if complete and VISUALS_ONLY and not FORCE_REBAKE:
        import json

        tris, nbytes = build_visual_glb(visual, bundle / "visual.glb", max_total_triangles=22000)
        man_path = bundle / "manifest.json"
        man = json.loads(man_path.read_text(encoding="utf-8"))
        man["visual_triangles"] = int(tris)
        man["visual_bytes"] = int(nbytes)
        man_path.write_text(json.dumps(man, indent=2), encoding="utf-8")
        rec_path = bundle / "part_recipe.json"
        if rec_path.is_file():
            rec = json.loads(rec_path.read_text(encoding="utf-8"))
            rec.update(recipe)
            rec_path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(f"  [visual] {asset_id}: {tris} tris, {nbytes / 1024:.0f} KiB (superelement unchanged)")
        return
    if not structs:
        raise RuntimeError(f"{asset_id}: no structural solids selected")
    structural = trimesh.util.concatenate(structs) if len(structs) > 1 else structs[0]
    spec = BakedAssetSpec(
        category="gantry_brackets",
        asset_id=asset_id,
        name=name,
        source_cad=SOURCE_CAD,
        source_url=SOURCE_URL,
        reference_material_id=REFERENCE_MATERIAL,
        cad_trust_pct=CAD_TRUST_PCT,
        port0=port0,
        port1=port1,
        n_internal_modes=6,
        mesh_size_mm=mesh_size_mm,
        is_multi_material=False,
        is_fdm_anisotropic=False,
        part_recipe=recipe,
        visual_max_triangles=22000,
    )
    t0 = time.perf_counter()
    res = bake_solid_superelement(
        spec,
        regions=[MaterialRegion(solid=structural, material_id=REFERENCE_MATERIAL, label=asset_id, fdm_anisotropic=False)],
        visual_parts=visual,
    )
    ps = res.manifest["port_stiffness_n_per_mm"]
    rb_hz = float(res.manifest["rigid_body_max_freq_hz"])
    print(
        f"  [CMS] {asset_id}: {res.manifest['num_elements']} tets, mass={res.manifest['mass_kg'] * 1e3:.1f} g "
        f"(Al ref), rb={rb_hz:.5f} Hz, "
        f"k=({ps['kx']:.0f}, {ps['ky']:.0f}, {ps['kz']:.0f}) N/mm ({time.perf_counter() - t0:.1f}s)"
    )
    if rb_hz >= 0.05:
        raise RuntimeError(f"{asset_id} failed < 0.05 Hz rigid-body QA gate (rb={rb_hz:.5f} Hz)")


def _select(parts: list[dict[str, Any]], prefix: str, x_pred: Callable[[float], bool] | None = None):
    out = []
    for p in parts:
        if not p["path"].startswith(prefix):
            continue
        if x_pred is not None:
            cx = 0.5 * (float(p["v"][:, 0].min()) + float(p["v"][:, 0].max()))
            if not x_pred(cx):
                continue
        out.append(p)
    return out


# ---------------------------------------------------------------------------
# Stations
# ---------------------------------------------------------------------------


def _frame_contact_boxes(side_sign: float, end: str) -> list:
    """Top-ring extrusion boxes (local mm about the corner datum) a corner mount bolts onto."""
    # Y extrusion (20 x 20) along the side; front/rear X extrusion along the end
    y_ext = ((-10.0, 10.0), (-40.0, 480.0) if end == "front" else (-480.0, 40.0), (-10.0, 10.0))
    if end == "front":
        x_ext = ((-10.0, 520.0) if side_sign < 0 else (-520.0, 10.0), (-20.0, 0.0), (-10.0, 10.0))
    else:
        x_ext = ((-10.0, 520.0) if side_sign < 0 else (-520.0, 10.0), (0.0, 20.0), (-10.0, 10.0))
    return [y_ext, x_ext]


def bake_rear(parts: list[dict[str, Any]]) -> None:
    for side, sgn, sub in (("left", -1.0, "Motor_Mount_Left"), ("right", 1.0, "Motor_Mount_right")):
        origin = np.array([X_LEFT_CL if sgn < 0 else X_RIGHT_CL, Y_REAR_CL, Z_GANTRY])
        sel = _select(parts, f"{GANTRY}Motormounts_with_tensioners_v41/{sub}")
        shaft = [p for p in sel if "nema17" in p["path"].lower() and float(np.ptp(p["v"][:, 2])) > 50.0][0]
        sx = 0.5 * (float(shaft["v"][:, 0].min()) + float(shaft["v"][:, 0].max())) - origin[0]
        sy = 0.5 * (float(shaft["v"][:, 1].min()) + float(shaft["v"][:, 1].max())) - origin[1]

        def _is_struct(path: str, ext: np.ndarray) -> bool:
            leaf = _leaf(path)
            return ("motor_mount_part" in leaf or "motor_plate" in leaf) and "nema" not in path.lower()

        p1 = (sx, sy, -2.0)
        _bake_station(
            asset_id=f"vzbot_330_rear_{side}",
            name=f"VzBot 330 Rear-{side.title()} Motor Mount + Tensioner",
            parts=sel,
            origin=origin,
            is_structural=_is_struct,
            port0=PortDefinition((0.0, 0.0, 0.0), _stratify_port(_min_count(_near_boxes(_frame_contact_boxes(sgn, "rear"), 1.6), (0.0, 0.0, 10.0)), cell_mm=7.5)),
            port1=PortDefinition(p1, _min_count(_near_axes([(sx, sy)], 6.5, (-26.0, 15.0)), p1)),
            mesh_size_mm=2.6,
            recipe={"station": f"rear_{side}", "port1": "motor shaft bore (tensioner plate + top bearing)", "origin_cad_mm": origin.tolist()},
        )


def bake_front_awd(parts: list[dict[str, Any]]) -> None:
    base = f"{GANTRY}AWD_Front_idlers_motormounts_v12/VZ330_AWD_Idlers/motormounts/"
    for side, sgn, sub in (("left", -1.0, "AWD_idler_left"), ("right", 1.0, "AWD_idler_right")):
        origin = np.array([X_LEFT_CL if sgn < 0 else X_RIGHT_CL, Y_FRONT_CL, Z_GANTRY])
        sel = _select(parts, base + sub)
        shaft = [p for p in sel if "nema17" in p["path"].lower() and float(np.ptp(p["v"][:, 2])) > 50.0][0]
        sx = 0.5 * (float(shaft["v"][:, 0].min()) + float(shaft["v"][:, 0].max())) - origin[0]
        sy = 0.5 * (float(shaft["v"][:, 1].min()) + float(shaft["v"][:, 1].max())) - origin[1]

        def _is_struct(path: str, ext: np.ndarray) -> bool:
            leaf = _leaf(path)
            return leaf.startswith("pulleymount") and "nema" not in path.lower()

        p1 = (sx, sy, 0.0)
        _bake_station(
            asset_id=f"vzbot_330_awd_front_{side}",
            name=f"VzBot 330 AWD Front-{side.title()} Idler/Motor Mount",
            parts=sel,
            origin=origin,
            is_structural=_is_struct,
            port0=PortDefinition((0.0, 0.0, 0.0), _stratify_port(_min_count(_near_boxes(_frame_contact_boxes(sgn, "front"), 1.6), (0.0, 0.0, 10.0)), cell_mm=7.5)),
            port1=PortDefinition(p1, _min_count(_near_axes([(sx, sy)], 6.5, (-16.0, 15.0)), p1)),
            mesh_size_mm=2.6,
            recipe={"station": f"front_{side}", "drive": "awd", "port1": "front motor shaft bore", "origin_cad_mm": origin.tolist()},
        )


def bake_front_2wd(parts: list[dict[str, Any]]) -> None:
    for side, sgn, sub in (("left", -1.0, "2WD_Front_Idlers_mounts_Left"), ("right", 1.0, "2WD_Front_Idlers_mounts_right")):
        origin = np.array([X_LEFT_CL if sgn < 0 else X_RIGHT_CL, Y_FRONT_CL, Z_GANTRY])
        sel = _select(parts, GANTRY + sub)
        bolts = [p for p in sel if "shoulder_bolt" in p["path"].lower()]
        axes = [
            (
                0.5 * (float(b["v"][:, 0].min()) + float(b["v"][:, 0].max())) - origin[0],
                0.5 * (float(b["v"][:, 1].min()) + float(b["v"][:, 1].max())) - origin[1],
            )
            for b in bolts
        ]
        cx = float(np.mean([a[0] for a in axes]))
        cy = float(np.mean([a[1] for a in axes]))

        def _is_struct(path: str, ext: np.ndarray) -> bool:
            return _leaf(path).startswith("pulleymount")

        p1 = (cx, cy, 0.0)
        _bake_station(
            asset_id=f"vzbot_330_2wd_front_{side}",
            name=f"VzBot 330 2WD Front-{side.title()} Idler Mount",
            parts=sel,
            origin=origin,
            is_structural=_is_struct,
            port0=PortDefinition((0.0, 0.0, 0.0), _stratify_port(_min_count(_near_boxes(_frame_contact_boxes(sgn, "front"), 1.6), (0.0, 0.0, 10.0)), cell_mm=7.5)),
            port1=PortDefinition(p1, _min_count(_near_axes(axes, 5.5, (-16.0, 15.0)), p1)),
            mesh_size_mm=2.4,
            recipe={"station": f"front_{side}", "drive": "2wd", "port1": "stacked 20T idler shoulder-bolt bores", "origin_cad_mm": origin.tolist()},
        )


def bake_xy(parts: list[dict[str, Any]]) -> None:
    for side, sgn, sub in (("left", -1.0, "Y_gantry_printed_-_TS_v50"), ("right", 1.0, "Y_gantry_printed_-_ST_v7")):
        origin = np.array([X_LEFT_CL if sgn < 0 else X_RIGHT_CL, Y_XBEAM, Z_GANTRY])
        sel = _select(parts, GANTRY + sub)
        # MGN12H Y carriage block riding the inner-face rail (visual only: its ball-raceway
        # compliance is the Hertzian y_car_bearing in the FEA, not part of this CMS body)
        y_block = _select(parts, GANTRY + ("Y_sliding_block_left" if sgn < 0 else "Y_sliding_block_right"))
        # Y-carriage pad face: joint face mating the MGN12H block on the inner rail face (X = -/+232)
        x_pad = -232.0 - origin[0] if sgn < 0 else 232.0 - origin[0]
        # CF tube clamp pocket: tube spans |X| <= 220, joint inner face at |X| = 203
        x_t0, x_t1 = (-220.0 - origin[0], -203.0 - origin[0]) if sgn < 0 else (203.0 - origin[0], 220.0 - origin[0])

        def _pad(c: np.ndarray, _x=x_pad) -> np.ndarray:
            return (np.abs(c[:, 0] - _x) <= 1.4) & (np.abs(c[:, 1]) <= 23.0) & (np.abs(c[:, 2]) <= 14.0)

        def _pocket(c: np.ndarray, _a=min(x_t0, x_t1), _b=max(x_t0, x_t1)) -> np.ndarray:
            in_x = (c[:, 0] >= _a) & (c[:, 0] <= _b)
            on_wall = ((np.abs(np.abs(c[:, 1]) - 10.0) <= 1.4) & (np.abs(c[:, 2]) <= 11.0)) | (
                (np.abs(np.abs(c[:, 2]) - 10.0) <= 1.4) & (np.abs(c[:, 1]) <= 11.0)
            )
            return in_x & on_wall

        def _is_struct(path: str, ext: np.ndarray) -> bool:
            return _leaf(path).startswith(("gantry_bottom", "gantry_top"))

        p0 = (x_pad, 0.0, 0.0)
        p1 = (0.5 * (x_t0 + x_t1), 0.0, 0.0)
        _bake_station(
            asset_id=f"vzbot_330_xy_{side}",
            name=f"VzBot 330 Y-Gantry XY-Joint {side.title()}",
            parts=sel,
            origin=origin,
            is_structural=_is_struct,
            port0=PortDefinition(p0, _stratify_port(_min_count(_pad, p0), cell_mm=4.5)),
            port1=PortDefinition(p1, _stratify_port(_min_count(_pocket, p1), cell_mm=4.5)),
            mesh_size_mm=2.0,
            recipe={"station": f"xy_{side}", "port0": "MGN12H Y-carriage pad face", "port1": "20x20 CF tube clamp pocket", "origin_cad_mm": origin.tolist(), "visual_includes": "MGN12H Y carriage block"},
            visual_only_parts=y_block,
        )


# ---------------------------------------------------------------------------
# Toolhead & hotend visual assets (Vz-Printhead + Goliath Air)
# ---------------------------------------------------------------------------

PRINTHEAD = f"{ROOT_LABEL}/Vz-Printhead_Printed_MGN9-revised_v24/"
# CAD datums: MGN9H block top face center (toolhead Port 0) and Goliath top mounting face / tip
MGN9H_TOP_FACE_CAD_MM = np.array([0.0, 58.26, 65.0])
GOLIATH_TIP_CAD_MM = np.array([-0.06, 35.9, -9.24])
GOLIATH_TOP_Z_CAD_MM = 60.0
COL_PRINTED_RED = "#c81e2b"
COL_PRINTED_BLACK = "#1f242d"


def _write_visual_bundle(
    category: str,
    asset_id: str,
    name: str,
    visual: list[VisualPart],
    extra_meta: dict[str, Any],
    max_triangles: int,
) -> None:
    import json
    from datetime import datetime, timezone

    bundle = ASSETS_ROOT / category / asset_id
    if (bundle / "superelement.npz").is_file():
        print(f"  [skip] {category}/{asset_id}: structural bundle present (visual comes with it)")
        return
    if not FORCE_REBAKE and (bundle / "manifest.json").is_file() and (bundle / "visual.glb").is_file():
        print(f"  [skip] {category}/{asset_id}: visual bundle already baked (use --force to re-bake)")
        return
    bundle.mkdir(parents=True, exist_ok=True)
    tris, nbytes = build_visual_glb(visual, bundle / "visual.glb", max_total_triangles=max_triangles)
    lo = np.min(np.vstack([vp.shape_or_mesh.bounds[0] for vp in visual]), axis=0)
    hi = np.max(np.vstack([vp.shape_or_mesh.bounds[1] for vp in visual]), axis=0)
    manifest = {
        "schema_version": 1,
        "category": category,
        "asset_id": asset_id,
        "name": name,
        "source_cad": SOURCE_CAD,
        "source_url": SOURCE_URL,
        "baked_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "has_superelement": False,
        "visual_glb": "visual.glb",
        "visual_triangles": int(tris),
        "visual_bytes": int(nbytes),
        "ref_size_mm": [round(float(v), 2) for v in (hi - lo)],
        **extra_meta,
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  [visual] {category}/{asset_id}: {tris} tris, {nbytes / 1024:.0f} KiB, size={manifest['ref_size_mm']} mm")


def _printhead_style(path: str, ext: np.ndarray) -> tuple[str, float, float]:
    lo = path.lower()
    leaf = _leaf(path)
    if "hextrudort" in lo:
        return COL_CNC_DARK, 0.78, 0.36
    if "fan_v2" in lo or "blades" in leaf or "body" in leaf and "25x25" in lo:
        return COL_PRINTED_BLACK, 0.10, 0.62
    if "cpap_tube" in leaf:
        return "#3a414d", 0.05, 0.72
    if "carriage" in lo or "adapter_plate" in lo:
        return COL_CNC_DARK, 0.82, 0.34
    if max(ext) <= 12.0 or "screw" in leaf or "standoff" in leaf or "insert" in leaf or "flathead" in leaf:
        return COL_STEEL, 0.84, 0.30
    if "duct" in lo or "cpap" in lo or "endstop" in lo or "enstop" in lo or "cable_holder" in lo:
        return COL_PRINTED_RED, 0.05, 0.58
    return COL_PRINTED_BLACK, 0.10, 0.6


def bake_toolhead_visuals(parts: list[dict[str, Any]]) -> None:
    """Vz-Printhead (toolheads/vz_printhead_cnc) and Goliath Air (hotends/mellow_goliath_air) GLBs.

    Visual-only bundles: the official CAD models the *printed* Vz-Printhead, so its geometry is not a
    valid structural proxy for the 27 g CNC plates the preset simulates; the FEA keeps the calibrated
    plate-sandwich Stage-2 chain while the Studio renders the exact CAD envelope.
    """
    include = (
        "hardware__7_",
        "Hotend_universal_adapter_plate",
        "Duct/",
        "carriage/",
        "cable_holder_Vz",
        "Rear_sliding_mount_CPAP_15mm",
        "Extruders/",
        "X-enstop",
        "Hotends/Goliath_Mockup_AIR_v1/goliath_heatsink_fan_duct",
    )
    th_parts = []
    for p in parts:
        if not p["path"].startswith(PRINTHEAD):
            continue
        rel = p["path"][len(PRINTHEAD):]
        if "Top_plate_MGN9C" in rel:  # mutually exclusive MGN9C variant at identical coordinates
            continue
        if rel.startswith(include):
            th_parts.append(p)
    origin = MGN9H_TOP_FACE_CAD_MM
    visual: list[VisualPart] = []
    for idx, part in enumerate(th_parts):
        m = _mesh(part, origin)
        if len(m.faces) == 0:
            continue
        col, met, rough = _printhead_style(part["path"], m.extents)
        visual.append(VisualPart(m, (_leaf(part["path"])[:24] or f"part_{idx}"), col, metallic=met, roughness=rough))
    hot_top = np.array([GOLIATH_TIP_CAD_MM[0], GOLIATH_TIP_CAD_MM[1], GOLIATH_TOP_Z_CAD_MM])
    port1 = [round(float(v), 3) for v in (hot_top - origin)]
    _write_visual_bundle(
        "toolheads",
        "vz_printhead_cnc",
        "VzBot Vz-Printhead (MGN9H top carriage, HextrudORT Low, CPAP)",
        visual,
        {
            "port0_mm": [0.0, 0.0, 0.0],
            "port1_mm": port1,
            "datum": "Port 0 = MGN9H block top face center; Port 1 = Goliath top mounting face",
            "x_rail_mount": "top",
            "cad_trust_pct": 70,
        },
        max_triangles=24000,
    )

    goliath = [
        p
        for p in parts
        if p["path"].startswith(PRINTHEAD + "Hotends/Goliath_Mockup_AIR_v1/Goliath_Air_cooled")
    ]
    hot_vis = [
        VisualPart(_mesh(p, hot_top), "goliath_air", "#d6dbe3", metallic=0.86, roughness=0.3) for p in goliath
    ]
    tip_drop = float(GOLIATH_TOP_Z_CAD_MM - GOLIATH_TIP_CAD_MM[2])
    _write_visual_bundle(
        "hotends",
        "mellow_goliath_air",
        "Mellow Goliath Air (VzBoT CAD mock-up)",
        hot_vis,
        {
            "port0_mm": [0.0, 0.0, 0.0],
            "port1_mm": [0.0, 0.0, round(-tip_drop, 2)],
            "datum": "Port 0 = top mounting face center; Port 1 = nozzle tip",
            "cad_trust_pct": 65,
        },
        max_triangles=12000,
    )


# ---------------------------------------------------------------------------
# Structural Vz-Printhead superelements: printed (exact CAD) and CNC (CAD layout, fitted plates)
# ---------------------------------------------------------------------------

# Fitted CNC plate set (Mellow full-alu 27 g AND hybrid CF top/bottom 22 g both reproduced):
# 1.5 mm top/bottom plates (window scale 0.93), 2.5 mm front/back side pieces (window scale 0.80)
CNC_TOP_BOTTOM_T_MM = 1.5
CNC_SIDE_T_MM = 2.5
CNC_TB_WINDOW = 0.93
CNC_SIDE_WINDOW = 0.80


def _b3d_box(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float):
    import build123d as b3d

    return b3d.Box(x1 - x0, y1 - y0, z1 - z0).moved(
        b3d.Location(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2))
    )


def _cnc_window(x0, x1, y0, y1, z0, z1, f):
    cx, cy, cz = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
    if z1 - z0 < 4.0:  # through-thickness window in a horizontal plate (scale x & y)
        return _b3d_box(cx - (x1 - x0) * f / 2, cx + (x1 - x0) * f / 2, cy - (y1 - y0) * f / 2, cy + (y1 - y0) * f / 2, z0, z1)
    return _b3d_box(cx - (x1 - x0) * f / 2, cx + (x1 - x0) * f / 2, y0, y1, cz - (z1 - z0) * f / 2, cz + (z1 - z0) * f / 2)


def build_cnc_cage_cad_mm():
    """Return ``{name: build123d Solid}`` for the CNC Vz-Printhead cage in VzBot CAD coordinates (mm).

    Same cage topology & datums as the official printed v24 head (top plate on the MGN9H block top
    face Z=65, front/back pieces either side of the 20 mm CF tube, bottom plate below it, rear belt /
    cable pillar), as in the Mellow kit assembly guide; thicknesses & windows fitted to the published
    27 g (6061) / 22 g (CF top-bottom hybrid) masses.
    """
    tt, ts, ftb, fs = CNC_TOP_BOTTOM_T_MM, CNC_SIDE_T_MM, CNC_TB_WINDOW, CNC_SIDE_WINDOW
    top = _b3d_box(-27.5, 27.5, 26.2, 82.91, 65, 65 + tt)
    for w in ((-26, -11, 49, 80), (11, 26, 49, 80), (-10, 10, 70, 81)):
        top = top - _cnc_window(*w, 64.9, 65.1 + tt, ftb)
    front = _b3d_box(-27.5, 27.5, 47.67 - ts, 47.67, 33 - tt, 65)
    for w in ((-26, -3), (3, 26)):
        front = front - _cnc_window(w[0], w[1], 47.6 - ts, 47.7, 34, 62, fs)
    back = _b3d_box(-22.5, 22.5, 72.91, 72.91 + ts, 33 - tt, 65) + _b3d_box(-5, 5, 72.91, 72.91 + ts, 65, 114)
    for w in ((-21, -3, 34, 62), (3, 21, 34, 62), (-3.5, 3.5, 72, 108)):
        back = back - _cnc_window(w[0], w[1], 72.8, 73.0 + ts, w[2], w[3], fs)
    bottom = _b3d_box(-27.5, 27.5, 33.64, 82.9, 33 - tt, 33)
    for w in ((-26, -3), (3, 26)):
        bottom = bottom - _cnc_window(w[0], w[1], 36, 80, 32.9 - tt, 33.1, ftb)
    # CNC 6061 universal hotend adapter (official outline 41.2 x 20.7 x 5 mm under the top plate front)
    adapter = _b3d_box(-20.62, 20.61, 26.2, 46.86, 60, 65)
    import build123d as b3d

    for hx, hy in ((-5.66, 30.21), (5.66, 30.21), (5.66, 41.52), (-5.66, 41.52)):
        adapter = adapter - b3d.Cylinder(1.3, 5.2).moved(b3d.Location((hx, hy, 62.5)))
    return {"top_plate": top, "front_piece": front, "back_piece": back, "bottom_plate": bottom, "hotend_adapter": adapter}


def _shape_mesh(shape: Any, origin: np.ndarray) -> trimesh.Trimesh:
    from bake_voron_kits import _shape_to_trimesh  # type: ignore[import-not-found]

    m = _shape_to_trimesh(shape, deflection=0.2)
    m.apply_translation(-origin)
    return m


def _dressing_parts(parts: list[dict[str, Any]], include_carriage: bool) -> list[dict[str, Any]]:
    keep = (
        "hardware__7_/Heat_insert",
        "hardware__7_/M3_8mm",
        "hardware__7_/M3_10mm",
        "Duct/",
        "cable_holder_Vz",
        "Rear_sliding_mount_CPAP_15mm",
        "Extruders/",
        "X-enstop",
        "Hotends/Goliath_Mockup_AIR_v1/goliath_heatsink_fan_duct",
    )
    carriage = ("carriage/", "Hotend_universal_adapter_plate", "hardware__7_/M3__30mm_round_standoffs")
    out = []
    for p in parts:
        if not p["path"].startswith(PRINTHEAD):
            continue
        rel = p["path"][len(PRINTHEAD):]
        if "Top_plate_MGN9C" in rel:
            continue
        if rel.startswith(keep) or (include_carriage and rel.startswith(carriage)):
            out.append(p)
    return out


def _toolhead_ports() -> tuple[PortDefinition, PortDefinition]:
    p1 = (-0.06, -22.36, -5.0)

    def _p0(c: np.ndarray) -> np.ndarray:  # top-plate underside on the MGN9H 4x M3 pad
        return (np.abs(c[:, 2]) <= 0.8) & (np.abs(c[:, 0]) <= 10.5) & (np.abs(c[:, 1]) <= 10.5)

    def _p1(c: np.ndarray) -> np.ndarray:  # hotend adapter underside around the Goliath top face
        return (np.abs(c[:, 2] - p1[2]) <= 0.8) & (np.hypot(c[:, 0] - p1[0], c[:, 1] - p1[1]) <= 12.0)

    return (
        PortDefinition((0.0, 0.0, 0.0), _min_count(_p0, (0.0, 0.0, 0.0))),
        PortDefinition(p1, _min_count(_p1, p1)),
    )


def _mark_multi_piece(category: str, asset_id: str, note: str) -> None:
    import json

    man_path = ASSETS_ROOT / category / asset_id / "manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    man["multi_piece_bolted"] = True
    man["x_rail_mount"] = "top"
    man["datum"] = "Port 0 = MGN9H block top face center; Port 1 = Goliath top mounting face"
    man["construction_note"] = note
    man_path.write_text(json.dumps(man, indent=2), encoding="utf-8")


def bake_toolhead_structural(parts: list[dict[str, Any]]) -> None:
    origin = MGN9H_TOP_FACE_CAD_MM
    port0, port1 = _toolhead_ports()

    # ---- Printed Vz-Printhead (exact official v24 MGN9 printed cage) ----
    struct_keys = (
        "carriage/Top_plate_MGN9H",
        "carriage/back_plate",
        "carriage/bottom_plate",
        "carriage/Front",
        "Hotend_universal_adapter_plate/SOLID",
        "hardware__7_/M3__30mm_round_standoffs",
    )
    printed_struct, printed_vis = [], []
    for idx, part in enumerate(_dressing_parts(parts, include_carriage=True)):
        m = _mesh(part, origin)
        if len(m.faces) == 0:
            continue
        rel = part["path"][len(PRINTHEAD):]
        col, met, rough = _printhead_style(part["path"], m.extents)
        if rel.startswith(("carriage/", "Hotend_universal_adapter_plate/SOLID")):
            col, met, rough = COL_PRINTED_BLACK, 0.08, 0.62
        printed_vis.append(VisualPart(m, (_leaf(part["path"])[:24] or f"part_{idx}"), col, metallic=met, roughness=rough))
        if rel.startswith(struct_keys):
            printed_struct.append(m)
    _bake_toolhead_asset(
        "vz_printhead_printed",
        "VzBot Vz-Printhead Printed v24 (ABS/ASA, MGN9H top carriage)",
        material="abs_printed",
        fdm=True,
        region=trimesh.util.concatenate(printed_struct),
        visual=printed_vis,
        port0=port0,
        port1=port1,
        mesh_size_mm=1.1,
        trust=80,
        recipe={"structural": list(struct_keys), "cage": "official printed v24 MGN9 cage (exact CAD)"},
    )
    _mark_multi_piece("toolheads", "vz_printhead_printed", "4 printed plates + adapter bolted at 8 corners with 4x M3 standoffs")

    # ---- CNC Vz-Printhead (official cage layout, plates fitted to Mellow 27 g / 22 g) ----
    import build123d as b3d

    cage = build_cnc_cage_cad_mm()
    fused = None
    for solid in cage.values():
        fused = solid if fused is None else fused.fuse(solid)
    fused_local = fused.moved(b3d.Location((-origin[0], -origin[1], -origin[2])))
    cnc_vis = []
    for name, solid in cage.items():
        cnc_vis.append(VisualPart(_shape_mesh(solid, origin), f"cnc_{name}", COL_STEEL if name != "hotend_adapter" else COL_CNC_DARK, metallic=0.88, roughness=0.3))
    for idx, part in enumerate(_dressing_parts(parts, include_carriage=False)):
        m = _mesh(part, origin)
        if len(m.faces) == 0:
            continue
        col, met, rough = _printhead_style(part["path"], m.extents)
        cnc_vis.append(VisualPart(m, (_leaf(part["path"])[:24] or f"part_{idx}"), col, metallic=met, roughness=rough))
    vol_cm3 = {k: round(float(v.volume) / 1000.0, 3) for k, v in cage.items()}
    _bake_toolhead_asset(
        "vz_printhead_cnc",
        "VzBot Vz-Printhead CNC 6061 (Mellow, MGN9H top carriage)",
        material="aluminum_6061_t6",
        fdm=False,
        region=fused_local,
        visual=cnc_vis,
        port0=port0,
        port1=port1,
        mesh_size_mm=1.0,
        trust=62,
        recipe={
            "cage": "official v24 cage layout & datums; CNC plate thickness/windows fitted to Mellow 27 g (6061) & 22 g (CF hybrid)",
            "top_bottom_t_mm": CNC_TOP_BOTTOM_T_MM,
            "side_t_mm": CNC_SIDE_T_MM,
            "part_volumes_cm3": vol_cm3,
        },
    )
    _mark_multi_piece("toolheads", "vz_printhead_cnc", "4 CNC plates + adapter bolted at 8 corners (Mellow kit)")


def _bake_toolhead_asset(
    asset_id: str,
    name: str,
    *,
    material: str,
    fdm: bool,
    region: Any,
    visual: list[VisualPart],
    port0: PortDefinition,
    port1: PortDefinition,
    mesh_size_mm: float,
    trust: int,
    recipe: dict[str, Any],
) -> None:
    import shutil

    bundle = ASSETS_ROOT / "toolheads" / asset_id
    if (bundle / "superelement.npz").is_file() and not FORCE_REBAKE:
        print(f"  [skip] toolheads/{asset_id}: structural bundle already baked (use --force)")
        return
    if bundle.is_dir() and not (bundle / "superelement.npz").is_file():
        shutil.rmtree(bundle)  # replace an earlier visual-only bundle
    spec = BakedAssetSpec(
        category="toolheads",
        asset_id=asset_id,
        name=name,
        source_cad=SOURCE_CAD,
        source_url=SOURCE_URL,
        reference_material_id=material,
        cad_trust_pct=trust,
        port0=port0,
        port1=port1,
        n_internal_modes=8,
        mesh_size_mm=mesh_size_mm,
        is_multi_material=False,
        is_fdm_anisotropic=fdm,
        part_recipe=recipe,
        visual_max_triangles=24000,
        fea_surface_max_triangles=80000,
    )
    t0 = time.perf_counter()
    res = bake_solid_superelement(
        spec,
        regions=[MaterialRegion(solid=region, material_id=material, label=asset_id, fdm_anisotropic=fdm)],
        visual_parts=visual,
    )
    ps = res.manifest["port_stiffness_n_per_mm"]
    print(
        f"  [CMS] toolheads/{asset_id}: {res.manifest['num_elements']} tets, mass={res.manifest['mass_kg'] * 1e3:.1f} g "
        f"({material}), rb={res.manifest['rigid_body_max_freq_hz']:.4f} Hz, k=({ps['kx']:.0f}, {ps['ky']:.0f}, {ps['kz']:.0f}) N/mm "
        f"({time.perf_counter() - t0:.0f}s)"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--step", type=Path, default=Path("cad_sources/vzbot/Vz-330_WEB v2.step"))
    ap.add_argument("--parts", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--group", choices=["all", "rear", "front_awd", "front_2wd", "xy", "toolhead", "toolhead_structural"], default="all")
    ap.add_argument("--force", action="store_true", help="Re-bake stations that already have a complete bundle")
    ap.add_argument("--visuals-only", action="store_true", help="Rebuild visual.glb of complete bundles without re-meshing")
    args = ap.parse_args()
    global FORCE_REBAKE, VISUALS_ONLY
    FORCE_REBAKE = bool(args.force)
    VISUALS_ONLY = bool(args.visuals_only)
    parts = load_parts(args.parts, args.step)
    print(f"Loaded {len(parts)} VzBot CAD solids")
    if args.group in ("all", "rear"):
        bake_rear(parts)
    if args.group in ("all", "front_awd"):
        bake_front_awd(parts)
    if args.group in ("all", "front_2wd"):
        bake_front_2wd(parts)
    if args.group in ("all", "xy"):
        bake_xy(parts)
    if args.group in ("all", "toolhead"):
        bake_toolhead_visuals(parts)  # Goliath hotend visual (toolheads are structural below)
    if args.group in ("all", "toolhead_structural"):
        bake_toolhead_structural(parts)
    clear_asset_cache()


if __name__ == "__main__":
    main()
