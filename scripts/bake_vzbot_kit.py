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
CAD_TRUST_PCT = 72
FORCE_REBAKE = False  # set by --force; otherwise stations with a complete bundle are skipped


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
) -> None:
    bundle = ASSETS_ROOT / "gantry_brackets" / asset_id
    if not FORCE_REBAKE and all(
        (bundle / f).is_file() for f in ("manifest.json", "superelement.npz", "visual.glb")
    ):
        print(f"  [skip] {asset_id}: complete bundle already baked (use --force to re-bake)")
        return
    visual: list[VisualPart] = []
    structs: list[trimesh.Trimesh] = []
    for idx, part in enumerate(parts):
        m = _mesh(part, origin)
        if len(m.faces) == 0:
            continue
        ext = m.extents
        col, met, rough = _style(part["path"], ext)
        visual.append(VisualPart(m, (_leaf(part["path"])[:24] or f"part_{idx}"), col, metallic=met, roughness=rough))
        if is_structural(part["path"], ext):
            structs.append(m)
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
    print(
        f"  [CMS] {asset_id}: {res.manifest['num_elements']} tets, mass={res.manifest['mass_kg'] * 1e3:.1f} g "
        f"(Al ref), rb={res.manifest['rigid_body_max_freq_hz']:.4f} Hz, "
        f"k=({ps['kx']:.0f}, {ps['ky']:.0f}, {ps['kz']:.0f}) N/mm ({time.perf_counter() - t0:.1f}s)"
    )


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
            port0=PortDefinition((0.0, 0.0, 0.0), _min_count(_near_boxes(_frame_contact_boxes(sgn, "rear"), 1.6), (0.0, 0.0, 10.0))),
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
            port0=PortDefinition((0.0, 0.0, 0.0), _min_count(_near_boxes(_frame_contact_boxes(sgn, "front"), 1.6), (0.0, 0.0, 10.0))),
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
            port0=PortDefinition((0.0, 0.0, 0.0), _min_count(_near_boxes(_frame_contact_boxes(sgn, "front"), 1.6), (0.0, 0.0, 10.0))),
            port1=PortDefinition(p1, _min_count(_near_axes(axes, 5.5, (-16.0, 15.0)), p1)),
            mesh_size_mm=2.4,
            recipe={"station": f"front_{side}", "drive": "2wd", "port1": "stacked 20T idler shoulder-bolt bores", "origin_cad_mm": origin.tolist()},
        )


def bake_xy(parts: list[dict[str, Any]]) -> None:
    for side, sgn, sub in (("left", -1.0, "Y_gantry_printed_-_TS_v50"), ("right", 1.0, "Y_gantry_printed_-_ST_v7")):
        origin = np.array([X_LEFT_CL if sgn < 0 else X_RIGHT_CL, Y_XBEAM, Z_GANTRY])
        sel = _select(parts, GANTRY + sub)
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
            port0=PortDefinition(p0, _min_count(_pad, p0)),
            port1=PortDefinition(p1, _min_count(_pocket, p1)),
            mesh_size_mm=2.0,
            recipe={"station": f"xy_{side}", "port0": "MGN12H Y-carriage pad face", "port1": "20x20 CF tube clamp pocket", "origin_cad_mm": origin.tolist()},
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--step", type=Path, default=Path("cad_sources/vzbot/Vz-330_WEB v2.step"))
    ap.add_argument("--parts", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--group", choices=["all", "rear", "front_awd", "front_2wd", "xy"], default="all")
    ap.add_argument("--force", action="store_true", help="Re-bake stations that already have a complete bundle")
    args = ap.parse_args()
    global FORCE_REBAKE
    FORCE_REBAKE = bool(args.force)
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
    clear_asset_cache()


if __name__ == "__main__":
    main()
