"""Declarative YAML CAD-to-Superelement Recipes, Caching, QA, Contact-Sheet & Onshape Export."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from typing import Any, Literal

import httpx
import numpy as np
from pydantic import BaseModel, Field
from scipy.linalg import eigh
import trimesh
import yaml

from printsim.cad_assets.baker import (
    BakedAssetSpec,
    MaterialRegion,
    PortDefinition,
    bake_solid_superelement,
    build_visual_glb,
    decimate_trimesh_qem,
    select_plane_box,
    select_z_annulus,
    select_z_cap,
)
from printsim.cad_assets.inspect import (
    _collect_step_solid_labels,
    _extract_brep_features,
    suggest_ports_from_inspection,
)
from printsim.cad_assets.registry import ASSETS_ROOT, clear_asset_cache
from printsim.schema import get_material

DEFAULT_CACHE_DIR = Path(".cache") / "cad_fea"

_ONSHAPE_URL_RE = re.compile(
    r"^https?://[^/]+/documents/([A-Za-z0-9_-]+)/w/([A-Za-z0-9_-]+)/e/([A-Za-z0-9_-]+)"
)


# ---------------------------------------------------------------------------
# Recipe Schema (YAML-driven)
# ---------------------------------------------------------------------------


class RecipeTransform(BaseModel):
    """Optional rigid-body transform applied to imported CAD geometry (in mm / degrees)."""

    rotate_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    translate_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: float = 1.0


class RecipePortSpec(BaseModel):
    """Port 0 / Port 1 virtual-point and boundary patch selector definition."""

    kind: Literal["z_annulus", "z_cap", "plane_box", "auto_face"] = "auto_face"
    params: dict[str, Any] = Field(default_factory=dict)
    vp_coord_mm: tuple[float, float, float] | None = None


class RecipeRegionSpec(BaseModel):
    """Optional per-solid material & PBR appearance override."""

    solid_index: int | None = None
    solid_name: str | None = None
    material_id: str
    label: str = ""
    color_hex: str | None = None
    metallic: float = 0.85
    roughness: float = 0.35
    fdm_anisotropic: bool | None = None


class RecipeOnshapeSpec(BaseModel):
    """Optional Onshape Part Studio provenance metadata."""

    document_id: str
    workspace_id: str
    element_id: str
    part_id: str | None = None
    configuration: str | None = None


class CadAssetRecipe(BaseModel):
    """Declarative YAML recipe for zero-LLM CAD-to-Craig-Bampton superelement baking."""

    schema_version: int = 1
    category: str
    asset_id: str
    name: str
    step_path: str
    source_url: str = "local"
    reference_material_id: str = "titanium_grade5"
    cad_trust_pct: int = 85
    mesh_size_mm: float = 1.6
    n_internal_modes: int = 8
    visual_max_triangles: int = 24000
    is_fdm_anisotropic: bool = False
    transform: RecipeTransform | None = None
    port0: RecipePortSpec = Field(
        default_factory=lambda: RecipePortSpec(kind="auto_face", params={"role": "port0"})
    )
    port1: RecipePortSpec = Field(
        default_factory=lambda: RecipePortSpec(kind="auto_face", params={"role": "port1"})
    )
    regions: list[RecipeRegionSpec] = Field(default_factory=list)
    onshape: RecipeOnshapeSpec | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> CadAssetRecipe:
        p = Path(path)
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Recipe YAML must be a mapping: {p}")
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump(mode="json", exclude_none=True)
        p.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return p


# ---------------------------------------------------------------------------
# QA Verification Gate (CAD_AND_FEA_PITFALLS.md Invariants)
# ---------------------------------------------------------------------------


@dataclass
class AssetQAReport:
    """Verification report validating baked superelement & visual GLB invariants."""

    passed: bool
    checks: list[dict[str, Any]]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sym_rel_err(mat: np.ndarray) -> float:
    denom = max(float(np.max(np.abs(mat))), 1e-18)
    return float(np.max(np.abs(mat - mat.T)) / denom)


def verify_baked_asset(
    category: str,
    asset_id: str,
    assets_root: Path | None = None,
) -> AssetQAReport:
    """Enforce all CAD/FEA invariants from ``docs/CAD_AND_FEA_PITFALLS.md`` on a baked asset."""
    root = assets_root if assets_root is not None else ASSETS_ROOT
    asset_dir = root / category / asset_id
    checks: list[dict[str, Any]] = []

    def _rec(name: str, ok: bool, value: Any, threshold: str, detail: str = "") -> bool:
        checks.append(
            {
                "name": name,
                "passed": bool(ok),
                "value": value,
                "threshold": threshold,
                "detail": detail,
            }
        )
        return bool(ok)

    manifest_path = asset_dir / "manifest.json"
    npz_path = asset_dir / "superelement.npz"
    glb_path = asset_dir / "visual.glb"

    files_ok = manifest_path.is_file() and npz_path.is_file() and glb_path.is_file()
    _rec(
        "files_exist",
        files_ok,
        {
            "manifest.json": manifest_path.is_file(),
            "superelement.npz": npz_path.is_file(),
            "visual.glb": glb_path.is_file(),
        },
        "all 3 bundle files exist",
        str(asset_dir),
    )
    if not files_ok:
        return AssetQAReport(
            passed=False,
            checks=checks,
            summary=f"FAIL {category}/{asset_id}: missing bundle files in {asset_dir}",
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # GLB file & triangle budget
    glb_bytes = glb_path.read_bytes()
    glb_ok = len(glb_bytes) > 12 and glb_bytes[:4] == b"glTF"
    _rec("glb_non_empty", glb_ok, len(glb_bytes), "> 12 bytes with glTF header")

    vis_tris = int(manifest.get("visual_triangles", 0))
    _rec("visual_triangles", 0 < vis_tris <= 35000, vis_tris, "0 < visual_triangles <= 35000")

    # Mass, CoM, Inertia
    with np.load(npz_path) as data:
        K_cb = np.asarray(data["K_cb"], dtype=np.float64)
        M_cb = np.asarray(data["M_cb"], dtype=np.float64)
        K_bb = np.asarray(data["K_bb"], dtype=np.float64)
        com_m = np.asarray(data["com_m"], dtype=np.float64)
        inertia = np.asarray(data["inertia_kg_m2"], dtype=np.float64)

    mass_kg = float(manifest.get("mass_kg", 0.0))
    mass_ok = math.isfinite(mass_kg) and mass_kg > 0.0
    _rec("mass_kg_positive", mass_ok, mass_kg, "> 0")

    com_ok = com_m.shape == (3,) and bool(np.all(np.isfinite(com_m)))
    _rec("com_m_finite", com_ok, com_m.tolist() if com_ok else None, "finite (3,)")

    inert_ok = (
        inertia.shape == (3, 3)
        and bool(np.all(np.isfinite(inertia)))
        and _sym_rel_err(inertia) < 1e-6
        and float(np.min(np.linalg.eigvalsh(0.5 * (inertia + inertia.T)))) > 0.0
    )
    _rec("inertia_kg_m2_valid", inert_ok, inertia.shape, "finite symmetric SPD (3,3)")

    # K_cb, K_bb, M_cb symmetry & finiteness
    kcb_finite = bool(np.all(np.isfinite(K_cb)))
    kcb_sym = _sym_rel_err(K_cb) if kcb_finite else float("inf")
    _rec("K_cb_symmetric_finite", kcb_finite and kcb_sym < 1e-6, kcb_sym, "< 1e-6")

    kbb_finite = bool(np.all(np.isfinite(K_bb)))
    kbb_sym = _sym_rel_err(K_bb) if kbb_finite else float("inf")
    _rec("K_bb_symmetric_finite", kbb_finite and kbb_sym < 1e-6, kbb_sym, "< 1e-6")

    mcb_finite = bool(np.all(np.isfinite(M_cb)))
    M_s = 0.5 * (M_cb + M_cb.T)
    m_min_eig = float(np.min(np.linalg.eigvalsh(M_s))) if mcb_finite else -1.0
    _rec("M_cb_spd", mcb_finite and _sym_rel_err(M_cb) < 1e-6 and m_min_eig > 0.0, m_min_eig, "> 0")

    # Rigid-body zero-mode invariant (< 0.05 Hz)
    manifest_rb_hz = float(manifest.get("rigid_body_max_freq_hz", 999.0))
    computed_rb_hz = float("inf")
    if kcb_finite and mcb_finite and m_min_eig > 0.0:
        K_s = 0.5 * (K_cb + K_cb.T)
        eigvals = eigh(K_s, M_s, eigvals_only=True)
        computed_rb_hz = float(np.max(np.sqrt(np.maximum(np.abs(eigvals[:6]), 0.0)) / (2.0 * math.pi)))
    rb_ok = manifest_rb_hz < 0.05 and computed_rb_hz < 0.05
    _rec(
        "rigid_body_max_freq_hz",
        rb_ok,
        {"manifest_hz": round(manifest_rb_hz, 6), "computed_hz": round(computed_rb_hz, 6)},
        "< 0.05 Hz",
    )

    # Port stiffness positive & grounded K_bb[6:12, 6:12] SPD
    pst = manifest.get("port_stiffness_n_per_mm") or {}
    kx = float(pst.get("kx", 0.0))
    ky = float(pst.get("ky", 0.0))
    kz = float(pst.get("kz", 0.0))
    k11 = 0.5 * (K_bb[6:12, 6:12] + K_bb[6:12, 6:12].T)
    k11_min_eig = float(np.min(np.linalg.eigvalsh(k11))) if kbb_finite else -1.0
    stiff_ok = kx > 0.0 and ky > 0.0 and kz > 0.0 and k11_min_eig > 0.0
    _rec(
        "port_stiffness_positive",
        stiff_ok,
        {"kx": kx, "ky": ky, "kz": kz, "k11_min_eig": k11_min_eig},
        "kx, ky, kz > 0 & K_bb[6:12] SPD",
    )

    all_passed = all(c["passed"] for c in checks)
    n_pass = sum(1 for c in checks if c["passed"])
    summary = (
        f"{'PASS' if all_passed else 'FAIL'} {category}/{asset_id}: "
        f"{n_pass}/{len(checks)} checks passed "
        f"(mass={mass_kg * 1000.0:.2f} g, rb_max={computed_rb_hz:.5f} Hz, "
        f"k=[{kx:.0f}, {ky:.0f}, {kz:.0f}] N/mm, tris={vis_tris})"
    )
    return AssetQAReport(passed=all_passed, checks=checks, summary=summary)


# ---------------------------------------------------------------------------
# 2x2 Orthographic + Isometric Contact Sheet Renderer
# ---------------------------------------------------------------------------


def _extract_colored_mesh_parts(
    mesh_or_glb_path: trimesh.Trimesh | trimesh.Scene | str | Path,
    ports_mm: dict[str, tuple[float, float, float]] | None,
    com_mm: tuple[float, float, float] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(verts_mm [N,3], faces [F,3], face_rgb [F,3])`` in millimeters."""
    from_glb = False
    if isinstance(mesh_or_glb_path, (str, Path)):
        p = Path(mesh_or_glb_path)
        from_glb = p.suffix.lower() in {".glb", ".gltf"}
        loaded = trimesh.load(str(p))
    else:
        loaded = mesh_or_glb_path

    parts: list[tuple[trimesh.Trimesh, np.ndarray]] = []
    default_rgb = np.array([0.62, 0.69, 0.78], dtype=np.float64)

    def _mesh_rgb(tm: trimesh.Trimesh) -> np.ndarray:
        try:
            vis = tm.visual
            mat = getattr(vis, "material", None)
            if mat is not None and getattr(mat, "baseColorFactor", None) is not None:
                c = np.asarray(mat.baseColorFactor[:3], dtype=np.float64)
                if np.max(c) > 1.0:
                    c = c / 255.0
                return np.clip(c, 0.05, 0.98)
            fc = getattr(vis, "face_colors", None)
            if fc is not None and len(fc) > 0:
                c = np.asarray(fc[0][:3], dtype=np.float64)
                if np.max(c) > 1.0:
                    c = c / 255.0
                return np.clip(c, 0.05, 0.98)
        except Exception:
            pass
        return default_rgb

    if isinstance(loaded, trimesh.Scene):
        for node_name in loaded.graph.nodes_geometry:
            mat, geom_name = loaded.graph[node_name]
            geom = loaded.geometry.get(geom_name)
            if isinstance(geom, trimesh.Trimesh) and len(geom.faces) > 0:
                tm = geom.copy()
                tm.apply_transform(mat)
                parts.append((tm, _mesh_rgb(tm)))
    elif isinstance(loaded, trimesh.Trimesh):
        parts.append((loaded.copy(), _mesh_rgb(loaded)))
    else:
        raise TypeError(f"Unsupported mesh type for contact sheet: {type(loaded)}")

    if not parts:
        raise ValueError("No non-empty triangle meshes available to render contact sheet")

    all_bounds = np.vstack([tm.bounds for tm, _ in parts])
    ext = all_bounds.max(axis=0) - all_bounds.min(axis=0)
    max_ext = float(np.max(ext))
    scale_mm = 1.0
    if from_glb and max_ext < 5.0:
        scale_mm = 1000.0
    elif max_ext < 2.0:
        ref_pts: list[tuple[float, float, float]] = []
        if ports_mm:
            ref_pts.extend(ports_mm.values())
        if com_mm is not None:
            ref_pts.append(com_mm)
        if ref_pts and float(np.max(np.abs(np.asarray(ref_pts)))) > 5.0 * max(max_ext, 1e-6):
            scale_mm = 1000.0

    # Keep preview fast (< 0.25s) while retaining crisp silhouettes
    total_faces = sum(len(tm.faces) for tm, _ in parts)
    budget = 6500
    v_blocks: list[np.ndarray] = []
    f_blocks: list[np.ndarray] = []
    c_blocks: list[np.ndarray] = []
    v_offset = 0

    for tm_raw, rgb in parts:
        tm = tm_raw
        if scale_mm != 1.0:
            tm = tm.copy()
            tm.apply_scale(scale_mm)
        if total_faces > budget and len(tm.faces) > 300:
            part_target = max(160, int(round(budget * (len(tm.faces) / total_faces))))
            if len(tm.faces) > part_target:
                try:
                    tm = decimate_trimesh_qem(tm, target_triangles=part_target)
                except Exception:
                    pass

        verts = np.asarray(tm.vertices, dtype=np.float64)
        faces = np.asarray(tm.faces, dtype=np.int64) + v_offset
        colors = np.repeat(rgb[None, :], len(faces), axis=0)
        v_blocks.append(verts)
        f_blocks.append(faces)
        c_blocks.append(colors)
        v_offset += len(verts)

    return np.vstack(v_blocks), np.vstack(f_blocks), np.vstack(c_blocks)


def render_contact_sheet(
    mesh_or_glb_path: trimesh.Trimesh | trimesh.Scene | str | Path,
    out_png: str | Path,
    ports_mm: dict[str, tuple[float, float, float]] | None = None,
    com_mm: tuple[float, float, float] | None = None,
    title: str = "",
) -> Path:
    """Render a crisp 2x2 orthographic + isometric contact sheet PNG with port & CoG overlays."""
    os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.collections import PolyCollection
    import matplotlib.pyplot as plt

    verts, faces, face_rgb = _extract_colored_mesh_parts(mesh_or_glb_path, ports_mm, com_mm)

    bmin = verts.min(axis=0)
    bmax = verts.max(axis=0)
    dims = bmax - bmin

    tris = verts[faces]  # (F, 3, 3)
    e1 = tris[:, 1] - tris[:, 0]
    e2 = tris[:, 2] - tris[:, 0]
    raw_n = np.cross(e1, e2)
    n_len = np.maximum(np.linalg.norm(raw_n, axis=1, keepdims=True), 1e-18)
    normals = raw_n / n_len
    tri_centers = tris.mean(axis=1)

    # Right-handed view orthonormal bases (u, v, depth)
    az = math.radians(-45.0)
    el = math.radians(30.0)
    iso_depth = np.array(
        [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)],
        dtype=np.float64,
    )
    iso_u = np.cross(np.array([0.0, 0.0, 1.0]), iso_depth)
    iso_u /= max(float(np.linalg.norm(iso_u)), 1e-12)
    iso_v = np.cross(iso_depth, iso_u)

    views = [
        ("Front (X-Z)", "X (mm)", "Z (mm)", np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([0.0, -1.0, 0.0])),
        ("Side (Y-Z)", "Y (mm)", "Z (mm)", np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])),
        ("Top (X-Y)", "X (mm)", "Y (mm)", np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
        ("Isometric", "U (mm)", "V (mm)", iso_u, iso_v, iso_depth),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.5), facecolor="#0f172a")
    fig.subplots_adjust(hspace=0.28, wspace=0.22, top=0.89, bottom=0.07, left=0.07, right=0.96)

    for ax, (v_title, xlabel, ylabel, u_ax, v_ax, d_ax) in zip(axes.flat, views):
        ax.set_facecolor("#1e293b")
        uv_verts = np.column_stack([verts @ u_ax, verts @ v_ax])
        depth_tri = tri_centers @ d_ax
        order = np.argsort(depth_tri, kind="mergesort")

        light_dir = 0.25 * u_ax + 0.40 * v_ax + 0.88 * d_ax
        light_dir /= float(np.linalg.norm(light_dir))
        n_dot_d = normals @ d_ax
        n_dot_l = normals @ light_dir
        oriented_l = np.where(n_dot_d >= 0.0, n_dot_l, -n_dot_l)
        shade = 0.34 + 0.66 * np.clip(oriented_l, 0.0, 1.0)

        shaded_rgb = np.clip(face_rgb[order] * shade[order, None], 0.0, 1.0)
        rgba = np.column_stack([shaded_rgb, np.ones(len(order), dtype=np.float64)])
        polys = uv_verts[faces[order]]

        coll = PolyCollection(polys, facecolors=rgba, edgecolors="none", antialiaseds=False)
        ax.add_collection(coll)

        u_min, u_max = float(uv_verts[:, 0].min()), float(uv_verts[:, 0].max())
        v_min, v_max = float(uv_verts[:, 1].min()), float(uv_verts[:, 1].max())
        span = max(u_max - u_min, v_max - v_min, 1.0)
        pad = 0.14 * span
        u_mid = 0.5 * (u_min + u_max)
        v_mid = 0.5 * (v_min + v_max)
        half = 0.5 * span + pad
        ax.set_xlim(u_mid - half, u_mid + half)
        ax.set_ylim(v_mid - half, v_mid + half)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color="#334155", linestyle="--", linewidth=0.5, alpha=0.6)

        # Overlay Port 0 (green), Port 1 (red), CoG (blue)
        markers: list[tuple[str, tuple[float, float, float], str, str]] = []
        if ports_mm:
            if "port0" in ports_mm:
                markers.append(("P0", ports_mm["port0"], "#22c55e", "o"))
            if "port1" in ports_mm:
                markers.append(("P1", ports_mm["port1"], "#ef4444", "o"))
        if com_mm is not None:
            markers.append(("CoG", com_mm, "#3b82f6", "D"))

        for m_lbl, pt3, color, marker in markers:
            pt_arr = np.asarray(pt3, dtype=np.float64)
            pu = float(pt_arr @ u_ax)
            pv = float(pt_arr @ v_ax)
            ax.scatter(
                [pu],
                [pv],
                s=58,
                c=color,
                marker=marker,
                edgecolors="#ffffff",
                linewidths=1.1,
                zorder=10,
            )
            ax.annotate(
                m_lbl,
                (pu, pv),
                xytext=(5, 5),
                textcoords="offset points",
                color="#f8fafc",
                fontsize=8,
                fontweight="bold",
                zorder=11,
            )

        ax.set_title(v_title, color="#f1f5f9", fontsize=10, fontweight="bold", pad=6)
        ax.set_xlabel(xlabel, color="#cbd5e1", fontsize=8.5)
        ax.set_ylabel(ylabel, color="#cbd5e1", fontsize=8.5)
        ax.tick_params(colors="#94a3b8", labelsize=7.5)
        for spine in ax.spines.values():
            spine.set_color("#475569")

    main_title = title or "CAD Superelement Contact Sheet"
    dim_str = (
        f"Dims: {dims[0]:.2f} × {dims[1]:.2f} × {dims[2]:.2f} mm   |   "
        f"● P0 (Mount)   ● P1 (Tip)   ◆ CoG"
    )
    fig.suptitle(main_title, color="#f8fafc", fontsize=12.5, fontweight="bold", y=0.965)
    fig.text(0.5, 0.922, dim_str, ha="center", color="#cbd5e1", fontsize=9)

    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Recipe Resolution, Hashing, Caching & Bake Pipeline
# ---------------------------------------------------------------------------


def _resolve_step_path(recipe: CadAssetRecipe, recipe_path: Path) -> Path:
    raw = Path(recipe.step_path)
    candidates = [raw] if raw.is_absolute() else [
        recipe_path.parent / raw,
        Path.cwd() / raw,
        Path.cwd() / "cad_sources" / raw,
    ]
    for cand in candidates:
        if cand.is_file():
            return cand.resolve()
    hint = ""
    if recipe.onshape is not None:
        hint = " (or fetch via `fetch_onshape_step` / `printsim cad fetch-onshape`)"
    raise FileNotFoundError(
        f"STEP file '{recipe.step_path}' not found in {[str(c) for c in candidates]}{hint}"
    )


def _apply_recipe_transform(shape: Any, tf: RecipeTransform | None) -> Any:
    if tf is None:
        return shape
    import build123d as b3d

    cur = shape
    if abs(float(tf.scale) - 1.0) > 1e-9:
        cur = cur.scale(float(tf.scale))
    rx, ry, rz = (float(x) for x in tf.rotate_deg)
    if abs(rx) > 1e-9:
        cur = cur.rotate(b3d.Axis.X, rx)
    if abs(ry) > 1e-9:
        cur = cur.rotate(b3d.Axis.Y, ry)
    if abs(rz) > 1e-9:
        cur = cur.rotate(b3d.Axis.Z, rz)
    tx, ty, tz = (float(x) for x in tf.translate_mm)
    if abs(tx) > 1e-9 or abs(ty) > 1e-9 or abs(tz) > 1e-9:
        cur = cur.translate((tx, ty, tz))
    return cur


def _build_port_definition(
    port_spec: RecipePortSpec,
    role: str,
    solids: list[Any],
    eff_mesh_size_mm: float,
    auto_suggestions: dict[str, dict[str, Any]] | None,
) -> tuple[PortDefinition, dict[str, dict[str, Any]]]:
    if port_spec.kind == "auto_face":
        if auto_suggestions is None or role not in auto_suggestions:
            mins = np.full(3, np.inf)
            maxs = np.full(3, -np.inf)
            for s in solids:
                bb = s.bounding_box()
                mins = np.minimum(mins, [bb.min.X, bb.min.Y, bb.min.Z])
                maxs = np.maximum(maxs, [bb.max.X, bb.max.Y, bb.max.Z])
            planar_faces, _ = _extract_brep_features(solids)
            auto_suggestions = suggest_ports_from_inspection(mins.tolist(), maxs.tolist(), planar_faces)
        sug = auto_suggestions[role]
        resolved_kind = str(sug["kind"])
        params = dict(sug["params"])
        params.update(port_spec.params)
        vp = tuple(float(v) for v in (port_spec.vp_coord_mm or sug["vp_coord_mm"]))
    else:
        resolved_kind = port_spec.kind
        params = dict(port_spec.params)
        if port_spec.vp_coord_mm is not None:
            vp = tuple(float(v) for v in port_spec.vp_coord_mm)
        elif resolved_kind == "z_annulus":
            cxy = params.get("center_xy_mm", (0.0, 0.0))
            vp = (float(cxy[0]), float(cxy[1]), float(params["z_mm"]))
        elif resolved_kind == "z_cap":
            cxy = params.get("center_xy_mm", (0.0, 0.0))
            vp = (float(cxy[0]), float(cxy[1]), float(params["z_max_mm"]))
        else:
            mins = np.full(3, np.inf)
            maxs = np.full(3, -np.inf)
            for s in solids:
                bb = s.bounding_box()
                mins = np.minimum(mins, [bb.min.X, bb.min.Y, bb.min.Z])
                maxs = np.maximum(maxs, [bb.max.X, bb.max.Y, bb.max.Z])
            center = 0.5 * (mins + maxs)
            ax = int(params.get("axis", 2))
            center[ax] = float(params["coord_mm"])
            vp = (float(center[0]), float(center[1]), float(center[2]))

    default_tol = max(0.35, 0.25 * eff_mesh_size_mm)
    if resolved_kind == "z_annulus":
        sel = select_z_annulus(
            z_mm=float(params["z_mm"]),
            r_min_mm=float(params.get("r_min_mm", 0.0)),
            r_max_mm=float(params.get("r_max_mm", 100.0)),
            center_xy_mm=tuple(float(v) for v in params.get("center_xy_mm", (0.0, 0.0))),
            tol_mm=float(params.get("tol_mm", default_tol)),
        )
    elif resolved_kind == "z_cap":
        sel = select_z_cap(
            z_max_mm=float(params["z_max_mm"]),
            z_min_mm=float(params.get("z_min_mm", -1e6)),
            center_xy_mm=tuple(float(v) for v in params.get("center_xy_mm", (0.0, 0.0))),
            r_max_mm=float(params.get("r_max_mm", 100.0)),
        )
    elif resolved_kind == "plane_box":
        raw_bounds = params.get("bounds_mm")
        bounds_mm = (
            tuple((float(lo), float(hi)) for lo, hi in raw_bounds)
            if raw_bounds is not None
            else None
        )
        sel = select_plane_box(
            axis=int(params.get("axis", 2)),
            coord_mm=float(params["coord_mm"]),
            tol_mm=float(params.get("tol_mm", default_tol)),
            bounds_mm=bounds_mm,
        )
    else:
        raise ValueError(f"Unsupported port selector kind '{resolved_kind}'")

    return PortDefinition((float(vp[0]), float(vp[1]), float(vp[2])), sel), (auto_suggestions or {})


def _build_material_regions(
    recipe: CadAssetRecipe,
    labeled_solids: list[tuple[Any, str]],
) -> tuple[list[MaterialRegion], bool]:
    regions: list[MaterialRegion] = []
    by_index = {r.solid_index: r for r in recipe.regions if r.solid_index is not None}
    by_name = {r.solid_name: r for r in recipe.regions if r.solid_name}

    for idx, (solid, lbl) in enumerate(labeled_solids):
        ov = by_index.get(idx) or (by_name.get(lbl) if lbl else None)
        mat_id = ov.material_id if ov is not None else recipe.reference_material_id
        mat = get_material(mat_id)
        regions.append(
            MaterialRegion(
                solid=solid,
                material_id=mat_id,
                label=(ov.label if ov and ov.label else lbl) or f"{recipe.asset_id}_{idx}",
                fdm_anisotropic=(
                    ov.fdm_anisotropic
                    if ov is not None and ov.fdm_anisotropic is not None
                    else (bool(recipe.is_fdm_anisotropic) and bool(mat.is_printed_polymer))
                ),
                color_hex=(ov.color_hex if ov and ov.color_hex else mat.color_hex),
                metallic=ov.metallic if ov is not None else (0.15 if mat.is_printed_polymer else 0.85),
                roughness=ov.roughness if ov is not None else (0.65 if mat.is_printed_polymer else 0.35),
            )
        )
    mat_ids = {r.material_id for r in regions}
    return regions, len(mat_ids) > 1


def _compute_recipe_content_hash(
    step_bytes: bytes,
    recipe: CadAssetRecipe,
    eff_mesh_size_mm: float,
    eff_n_modes: int,
    eff_vis_tris: int,
) -> str:
    step_sha = hashlib.sha256(step_bytes).hexdigest()
    payload = {
        "schema_version": 1,
        "step_sha256": step_sha,
        "category": recipe.category,
        "asset_id": recipe.asset_id,
        "reference_material_id": recipe.reference_material_id,
        "is_fdm_anisotropic": recipe.is_fdm_anisotropic,
        "mesh_size_mm": float(eff_mesh_size_mm),
        "n_internal_modes": int(eff_n_modes),
        "visual_max_triangles": int(eff_vis_tris),
        "transform": recipe.transform.model_dump(mode="json") if recipe.transform else None,
        "port0": recipe.port0.model_dump(mode="json"),
        "port1": recipe.port1.model_dump(mode="json"),
        "regions": [r.model_dump(mode="json") for r in recipe.regions],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def bake_from_recipe(
    recipe_path: str | Path,
    *,
    fast: bool = False,
    visuals_only: bool = False,
    use_cache: bool = True,
    cache_dir: Path | None = None,
    preview_png: Path | None = None,
    assets_root: Path | None = None,
) -> dict[str, Any]:
    """Bake a declarative YAML CAD recipe into a verified superelement bundle with caching."""
    t0 = time.perf_counter()
    rpath = Path(recipe_path)
    recipe = CadAssetRecipe.from_yaml(rpath)
    step_file = _resolve_step_path(recipe, rpath)
    step_bytes = step_file.read_bytes()

    eff_mesh_size_mm = max(recipe.mesh_size_mm * 1.8, 2.8) if fast else float(recipe.mesh_size_mm)
    eff_n_modes = min(recipe.n_internal_modes, 4) if fast else int(recipe.n_internal_modes)
    eff_vis_tris = min(recipe.visual_max_triangles, 8000) if fast else int(recipe.visual_max_triangles)

    root = assets_root if assets_root is not None else ASSETS_ROOT
    asset_dir = root / recipe.category / recipe.asset_id
    asset_dir.mkdir(parents=True, exist_ok=True)

    content_hash = _compute_recipe_content_hash(
        step_bytes, recipe, eff_mesh_size_mm, eff_n_modes, eff_vis_tris
    )
    croot = cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR
    cdir = croot / content_hash

    out_preview: Path | None = preview_png
    if fast and out_preview is None:
        out_preview = asset_dir / "preview.png"

    # Visual-only refresh path (no re-meshing)
    if visuals_only:
        manifest_path = asset_dir / "manifest.json"
        npz_path = asset_dir / "superelement.npz"
        if not manifest_path.is_file() or not npz_path.is_file():
            raise FileNotFoundError(
                f"visuals_only=True requires existing manifest.json and superelement.npz in {asset_dir}"
            )
        import build123d as b3d
        from printsim.cad_assets.baker import VisualPart

        imported = b3d.import_step(str(step_file))
        raw_labeled = _collect_step_solid_labels(imported)
        tf_labeled = [(_apply_recipe_transform(s, recipe.transform), lbl) for s, lbl in raw_labeled]
        regions, _ = _build_material_regions(recipe, tf_labeled)
        vis_parts = [
            VisualPart(
                shape_or_mesh=reg.solid,
                label=reg.label,
                color_hex=reg.color_hex or get_material(reg.material_id).color_hex,
                metallic=reg.metallic,
                roughness=reg.roughness,
            )
            for reg in regions
        ]
        vis_tris, vis_bytes = build_visual_glb(
            vis_parts, asset_dir / "visual.glb", max_total_triangles=eff_vis_tris
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["visual_triangles"] = int(vis_tris)
        manifest["visual_bytes"] = int(vis_bytes)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        clear_asset_cache()
        if out_preview is not None:
            render_contact_sheet(
                asset_dir / "visual.glb",
                out_preview,
                ports_mm={
                    "port0": tuple(float(v) for v in manifest["port0_mm"]),
                    "port1": tuple(float(v) for v in manifest["port1_mm"]),
                },
                com_mm=tuple(float(v) for v in manifest["com_mm"]),
                title=recipe.name,
            )
        qa = verify_baked_asset(recipe.category, recipe.asset_id, assets_root=root)
        return {
            "manifest": manifest,
            "qa": qa.to_dict(),
            "cache_hit": False,
            "preview_png": str(out_preview) if out_preview is not None else None,
            "elapsed_s": round(time.perf_counter() - t0, 4),
            "content_hash": content_hash,
        }

    # Cache hit check
    cached_manifest = cdir / "manifest.json"
    cached_npz = cdir / "superelement.npz"
    cached_glb = cdir / "visual.glb"
    if use_cache and cached_manifest.is_file() and cached_npz.is_file() and cached_glb.is_file():
        for fname in ("manifest.json", "superelement.npz", "visual.glb", "part_recipe.json"):
            src = cdir / fname
            if src.is_file():
                shutil.copy2(src, asset_dir / fname)
        clear_asset_cache()
        manifest = json.loads((asset_dir / "manifest.json").read_text(encoding="utf-8"))
        if out_preview is not None:
            cached_png = cdir / "preview.png"
            if cached_png.is_file():
                out_preview.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(cached_png, out_preview)
            else:
                render_contact_sheet(
                    asset_dir / "visual.glb",
                    out_preview,
                    ports_mm={
                        "port0": tuple(float(v) for v in manifest["port0_mm"]),
                        "port1": tuple(float(v) for v in manifest["port1_mm"]),
                    },
                    com_mm=tuple(float(v) for v in manifest["com_mm"]),
                    title=recipe.name,
                )
        qa = verify_baked_asset(recipe.category, recipe.asset_id, assets_root=root)
        return {
            "manifest": manifest,
            "qa": qa.to_dict(),
            "cache_hit": True,
            "preview_png": str(out_preview) if out_preview is not None else None,
            "elapsed_s": round(time.perf_counter() - t0, 4),
            "content_hash": content_hash,
        }

    # Full B-Rep import -> transform -> bake
    import build123d as b3d

    imported = b3d.import_step(str(step_file))
    raw_labeled = _collect_step_solid_labels(imported)
    if not raw_labeled:
        raise ValueError(f"No 3D solid volumes found in {step_file}")

    tf_labeled = [(_apply_recipe_transform(s, recipe.transform), lbl) for s, lbl in raw_labeled]
    tf_solids = [s for s, _ in tf_labeled]
    regions, is_multi = _build_material_regions(recipe, tf_labeled)

    port0_def, auto_sug = _build_port_definition(
        recipe.port0, "port0", tf_solids, eff_mesh_size_mm, None
    )
    port1_def, _ = _build_port_definition(
        recipe.port1, "port1", tf_solids, eff_mesh_size_mm, auto_sug
    )

    spec = BakedAssetSpec(
        category=recipe.category,
        asset_id=recipe.asset_id,
        name=recipe.name,
        source_cad=step_file.name,
        source_url=recipe.source_url,
        reference_material_id=recipe.reference_material_id,
        cad_trust_pct=recipe.cad_trust_pct,
        port0=port0_def,
        port1=port1_def,
        n_internal_modes=eff_n_modes,
        mesh_size_mm=eff_mesh_size_mm,
        is_multi_material=is_multi,
        is_fdm_anisotropic=recipe.is_fdm_anisotropic,
        part_recipe={
            "recipe_file": rpath.name,
            "content_hash": content_hash,
            "fast": bool(fast),
        },
        visual_max_triangles=eff_vis_tris,
    )

    res = bake_solid_superelement(spec=spec, regions=regions, assets_root=root)

    if out_preview is not None:
        render_contact_sheet(
            asset_dir / "visual.glb",
            out_preview,
            ports_mm={
                "port0": tuple(float(v) for v in res.manifest["port0_mm"]),
                "port1": tuple(float(v) for v in res.manifest["port1_mm"]),
            },
            com_mm=tuple(float(v) for v in res.manifest["com_mm"]),
            title=recipe.name,
        )

    if use_cache:
        cdir.mkdir(parents=True, exist_ok=True)
        for fname in ("manifest.json", "superelement.npz", "visual.glb", "part_recipe.json"):
            src = asset_dir / fname
            if src.is_file():
                shutil.copy2(src, cdir / fname)
        if out_preview is not None and out_preview.is_file():
            shutil.copy2(out_preview, cdir / "preview.png")

    qa = verify_baked_asset(recipe.category, recipe.asset_id, assets_root=root)
    return {
        "manifest": res.manifest,
        "qa": qa.to_dict(),
        "cache_hit": False,
        "preview_png": str(out_preview) if out_preview is not None else None,
        "elapsed_s": round(time.perf_counter() - t0, 4),
        "content_hash": content_hash,
    }


# ---------------------------------------------------------------------------
# Direct Onshape STEP Fetcher (Zero-LLM REST Export)
# ---------------------------------------------------------------------------


def parse_onshape_reference(
    url_or_doc: str,
    *,
    workspace_id: str | None = None,
    element_id: str | None = None,
) -> tuple[str, str, str]:
    """Parse an Onshape document URL or explicit IDs into ``(did, wid, eid)``."""
    raw = str(url_or_doc).strip()
    if raw.startswith("http://") or raw.startswith("https://"):
        m = _ONSHAPE_URL_RE.match(raw)
        if not m:
            raise ValueError(
                f"Invalid Onshape document URL '{raw}'. Expected format: "
                "https://cad.onshape.com/documents/{did}/w/{wid}/e/{eid}"
            )
        return m.group(1), m.group(2), m.group(3)

    if not raw or not workspace_id or not element_id:
        raise ValueError(
            "Provide either a full Onshape URL or explicit document_id + workspace_id + element_id"
        )
    return raw, str(workspace_id).strip(), str(element_id).strip()


def fetch_onshape_step(
    url_or_doc: str,
    out_path: str | Path,
    *,
    workspace_id: str | None = None,
    element_id: str | None = None,
    configuration: str | None = None,
) -> Path:
    """Download a STEP export directly from Onshape Part Studios REST API using Basic Auth."""
    did, wid, eid = parse_onshape_reference(
        url_or_doc, workspace_id=workspace_id, element_id=element_id
    )

    access_key = os.environ.get("ONSHAPE_ACCESS_KEY", "").strip()
    secret_key = os.environ.get("ONSHAPE_SECRET_KEY", "").strip()
    if not access_key or not secret_key:
        raise RuntimeError(
            "Missing Onshape API credentials: set ONSHAPE_ACCESS_KEY and ONSHAPE_SECRET_KEY "
            "environment variables, or export the STEP once via a single `onshape_export` call."
        )

    api_url = f"https://cad.onshape.com/api/v6/partstudios/d/{did}/w/{wid}/e/{eid}/step"
    params = {"configuration": configuration} if configuration else None

    with httpx.Client(follow_redirects=True, timeout=120.0) as client:
        resp = client.get(api_url, params=params, auth=(access_key, secret_key))
        resp.raise_for_status()
        content = resp.content

    if not content:
        raise RuntimeError(f"Onshape returned empty STEP response from {api_url}")

    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    return dest
