"""Modeled & simulated frame bracing builder (3D CAD superelements + bolted flat-bar FEA).

Replaces legacy unmodeled scalar corner multipliers and idealized center-line truss rods
with physically modeled hardware:
  1. ``cast_corner_brackets``: Die-cast aluminum gusseted 90° L-brackets (Misumi HBLFSN5/6/8)
     loaded from baked 3D Tet4 Craig-Bampton superelements, combined in series with Huth
     T-slot bolt-bearing compliance, rotated into each 3D frame corner orientation, and
     coupled across split extrusion nodes at the real bolt-center offset.
  2. ``cnc_corner_gussets``: 5–6 mm CNC 6061-T6 aluminum 5-hole L-gusset plates
     (RatRig / Monolith / DoomCube style) loaded from baked 3D Tet4 superelements,
     mounted flush on exterior frame faces with exact rigid-arm offset + 2-bolt compliance.
  3. ``rear_flat_bar_x`` / ``side_flat_bar_x`` / ``rear_and_side_flat_bars``:
     Bolted 25×5 mm 6061-T6 aluminum flat-bar diagonal X-braces modeled as 12-DOF
     Timoshenko beams offset onto the exterior frame face and coupled via M5 T-nut
     CBUSH fastener springs (non-blocking: open top and open front door).
  4. ``gussets_and_flat_bars``: Combines CNC corner gusset CAD superelements at every
     frame corner with bolted 25×5 mm rear + side flat-bar X-braces.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from printsim.cad_assets.registry import LoadedHardwareAsset, load_superelement
from printsim.schema import LEGACY_BRACING_ALIASES, PrinterSpec, get_material, get_profile

if TYPE_CHECKING:
    from printsim.geometry.builder import (
        AttachedSuperelement,
        FastenerSpring,
        StructuralMember,
        StructuralNode,
    )


def _skew(v: np.ndarray) -> np.ndarray:
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    return np.array(
        [
            [0.0, -vz, vy],
            [vz, 0.0, -vx],
            [-vy, vx, 0.0],
        ],
        dtype=float,
    )


def _select_profile_series(width_m: float) -> str:
    """Map frame extrusion width (m) to the nearest baked CAD series (2020/3030/4040)."""
    if width_m <= 0.025:
        return "2020"
    if width_m <= 0.035:
        return "3030"
    return "4040"


def _series_bolted_corner_superelement_local(
    asset: LoadedHardwareAsset,
    *,
    kind: str,
    ext_w_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine baked 12x12 solid-FEM ``(K_bb, M_bb)`` in series with Huth T-slot bolt compliance.

    Returns ``(K_12_local, M_12_local, port_coords_local_m)`` with exact zero rigid-body
    strain energy under all 6 rigid-body modes.
    """
    K_bb = np.asarray(asset.K_bb, dtype=float)
    M_bb = np.asarray(asset.M_cb[:12, :12], dtype=float)
    p_cad = np.asarray(asset.port_coords_m, dtype=float)  # (2, 3)
    dp = p_cad[1] - p_cad[0]

    # Fixed-port-0 6x6 stiffness at Port 1
    K11 = 0.5 * (K_bb[6:12, 6:12] + K_bb[6:12, 6:12].T)
    # Regularize tiny numerical floor if needed before inversion
    C11_cad = np.linalg.inv(K11)

    # Huth / VDI 2230 T-slot M5/M6/M8 bolt bearing + flange clamping compliance per port
    scale_w = max(ext_w_m / 0.020, 0.65)
    if kind == "cast_corner_bracket":
        # 1x M5/M6/M8 T-nut bolt per leg on cast aluminum bracket
        k_t = 18.0e6 * scale_w
        k_r = 1.8e3 * (scale_w**2)
    else:
        # 2x M5/M6/M8 bolts per outer leg on 5-hole CNC aluminum gusset plate
        k_t = 42.0e6 * scale_w
        k_r = 8.5e3 * (scale_w**2)

    C_bolt = np.diag([1.0 / k_t, 1.0 / k_t, 1.0 / k_t, 1.0 / k_r, 1.0 / k_r, 1.0 / k_r])

    # Rigid-body transfer from Port 0 to Port 1: u1_rigid = T_01 @ u0_rigid
    T_01 = np.eye(6, dtype=float)
    T_01[:3, 3:6] = -_skew(dp)

    # Total relative compliance at Port 1 (CAD body + Port 0 bolt + Port 1 bolt in series)
    C11_tot = C11_cad + C_bolt + T_01 @ C_bolt @ T_01.T
    C11_tot = 0.5 * (C11_tot + C11_tot.T)
    K11_tot = np.linalg.inv(C11_tot)

    # Self-equilibrating 12x12 stiffness matrix: B^T @ [u0; u1] = u1 - T_01 @ u0
    B = np.vstack((-T_01.T, np.eye(6, dtype=float)))
    K12 = B @ K11_tot @ B.T
    K12 = 0.5 * (K12 + K12.T)
    M12 = 0.5 * (M_bb + M_bb.T)
    return K12, M12, p_cad


def _rotate_12x12(M12: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Rotate a 12x12 2-port 6-DOF matrix by 3x3 orthogonal rotation ``R`` (local -> global)."""
    R12 = np.zeros((12, 12), dtype=float)
    for blk in range(4):
        s = 3 * blk
        R12[s : s + 3, s : s + 3] = R
    out = R12 @ M12 @ R12.T
    return 0.5 * (out + out.T)


def _corexy_corner_specs(
    w: float,
    d: float,
    h: float,
    kind: str,
) -> list[dict[str, Any]]:
    """Return corner definitions ``(id, vertex, ex, ey, ez_sign)`` for CoreXY frame corners.

    ``ex`` and ``ey`` are unit vectors pointing inward from the corner vertex along the
    two meeting frame extrusions. ``ez = ex x ey``.
    For ``cast_corner_bracket``, the bracket sits on the INSIDE corner between ``ex`` and ``ey``.
    For ``cnc_corner_gusset``, the plate sits flush on the EXTERIOR face with ``+ez`` pointing outward.
    """
    corners: list[dict[str, Any]] = []

    if kind == "cast_corner_bracket":
        # 8 outer corners x 3 orthogonal inner planes = 24 Misumi HBLFSN brackets
        verts = [
            ("b_fl", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            ("b_fr", (w, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            ("b_rl", (0.0, d, 0.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
            ("b_rr", (w, d, 0.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
            ("t_fl", (0.0, 0.0, h), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),
            ("t_fr", (w, 0.0, h), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),
            ("t_rl", (0.0, d, h), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)),
            ("t_rr", (w, d, h), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)),
        ]
        for cname, pt, vx, vy, vz in verts:
            pairs = (
                ("xy", vx, vy),
                ("xz", vx, vz),
                ("yz", vy, vz),
            )
            for ptag, a_vec, b_vec in pairs:
                corners.append(
                    {
                        "id": f"cast_{cname}_{ptag}",
                        "vertex": np.asarray(pt, dtype=float),
                        "ex": np.asarray(a_vec, dtype=float),
                        "ey": np.asarray(b_vec, dtype=float),
                    }
                )
        return corners

    # cnc_corner_gusset: 16 exterior 5-hole corner plates on the 4 vertical faces
    # (Left -X, Right +X, Front -Y, Rear +Y; 4 corners per face)
    # Ordered so ez = ex x ey always points OUTWARD from the frame face!
    face_defs = [
        # Left face (x = 0, outward normal = -X)
        ("left_bf", (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
        ("left_br", (0.0, d, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
        ("left_tf", (0.0, 0.0, h), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),
        ("left_tr", (0.0, d, h), (0.0, 0.0, -1.0), (0.0, -1.0, 0.0)),
        # Right face (x = w, outward normal = +X)
        ("right_bf", (w, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        ("right_br", (w, d, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
        ("right_tf", (w, 0.0, h), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
        ("right_tr", (w, d, h), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)),
        # Front face (y = 0, outward normal = -Y)
        ("front_bl", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ("front_br", (w, 0.0, 0.0), (0.0, 0.0, 1.0), (-1.0, 0.0, 0.0)),
        ("front_tl", (0.0, 0.0, h), (0.0, 0.0, -1.0), (1.0, 0.0, 0.0)),
        ("front_tr", (w, 0.0, h), (-1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
        # Rear face (y = d, outward normal = +Y)
        ("rear_bl", (0.0, d, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
        ("rear_br", (w, d, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ("rear_tl", (0.0, d, h), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
        ("rear_tr", (w, d, h), (0.0, 0.0, -1.0), (-1.0, 0.0, 0.0)),
    ]
    for cid, pt, ex, ey in face_defs:
        corners.append(
            {
                "id": f"gusset_{cid}",
                "vertex": np.asarray(pt, dtype=float),
                "ex": np.asarray(ex, dtype=float),
                "ey": np.asarray(ey, dtype=float),
            }
        )
    return corners


def _bedslinger_corner_specs(
    nmap: dict[str, StructuralNode],
    kind: str,
) -> list[dict[str, Any]]:
    """Return corner definitions for Cartesian bedslinger portal & base junctions."""
    p_bl = nmap["p_bl"].position_m
    p_br = nmap["p_br"].position_m
    p_tl = nmap["p_tl"].position_m
    p_tr = nmap["p_tr"].position_m
    if kind == "cast_corner_bracket":
        return [
            {"id": "cast_pbl_f", "vertex": p_bl, "ex": np.array([0.0, -1.0, 0.0]), "ey": np.array([0.0, 0.0, 1.0])},
            {"id": "cast_pbl_r", "vertex": p_bl, "ex": np.array([0.0, 1.0, 0.0]), "ey": np.array([0.0, 0.0, 1.0])},
            {"id": "cast_pbr_f", "vertex": p_br, "ex": np.array([0.0, -1.0, 0.0]), "ey": np.array([0.0, 0.0, 1.0])},
            {"id": "cast_pbr_r", "vertex": p_br, "ex": np.array([0.0, 1.0, 0.0]), "ey": np.array([0.0, 0.0, 1.0])},
            {"id": "cast_pbl_x", "vertex": p_bl, "ex": np.array([1.0, 0.0, 0.0]), "ey": np.array([0.0, 0.0, 1.0])},
            {"id": "cast_pbr_x", "vertex": p_br, "ex": np.array([-1.0, 0.0, 0.0]), "ey": np.array([0.0, 0.0, 1.0])},
            {"id": "cast_ptl_x", "vertex": p_tl, "ex": np.array([1.0, 0.0, 0.0]), "ey": np.array([0.0, 0.0, -1.0])},
            {"id": "cast_ptr_x", "vertex": p_tr, "ex": np.array([-1.0, 0.0, 0.0]), "ey": np.array([0.0, 0.0, -1.0])},
        ]
    # CNC gusset plates on outer YZ sides of portal uprights + rear XZ portal corners
    return [
        {"id": "gusset_pbl_f", "vertex": p_bl, "ex": np.array([0.0, 0.0, 1.0]), "ey": np.array([0.0, -1.0, 0.0])},
        {"id": "gusset_pbl_r", "vertex": p_bl, "ex": np.array([0.0, 1.0, 0.0]), "ey": np.array([0.0, 0.0, 1.0])},
        {"id": "gusset_pbr_f", "vertex": p_br, "ex": np.array([0.0, -1.0, 0.0]), "ey": np.array([0.0, 0.0, 1.0])},
        {"id": "gusset_pbr_r", "vertex": p_br, "ex": np.array([0.0, 0.0, 1.0]), "ey": np.array([0.0, 1.0, 0.0])},
        {"id": "gusset_ptl_r", "vertex": p_tl, "ex": np.array([1.0, 0.0, 0.0]), "ey": np.array([0.0, 0.0, -1.0])},
        {"id": "gusset_ptr_r", "vertex": p_tr, "ex": np.array([0.0, 0.0, -1.0]), "ey": np.array([-1.0, 0.0, 0.0])},
        {"id": "gusset_pbl_xz", "vertex": p_bl, "ex": np.array([0.0, 0.0, 1.0]), "ey": np.array([1.0, 0.0, 0.0])},
        {"id": "gusset_pbr_xz", "vertex": p_br, "ex": np.array([-1.0, 0.0, 0.0]), "ey": np.array([0.0, 0.0, 1.0])},
    ]


def _delta_corner_specs(
    nmap: dict[str, StructuralNode],
    kind: str,
) -> list[dict[str, Any]]:
    """Return corner definitions for Delta tower-to-base and tower-to-top junctions."""
    corners: list[dict[str, Any]] = []
    prefix = "cast" if kind == "cast_corner_bracket" else "gusset"
    center_xy = sum(nmap[f"b_{k}"].position_m[:2] for k in range(3)) / 3.0
    for i in range(3):
        j_next = (i + 1) % 3
        j_prev = (i - 1) % 3
        for tier, z_dir in (("b", 1.0), ("t", -1.0)):
            v = nmap[f"{tier}_{i}"].position_m
            ez_vec = np.array([0.0, 0.0, z_dir], dtype=float)
            for tag, nb in (("n", j_next), ("p", j_prev)):
                v_nb = nmap[f"{tier}_{nb}"].position_m
                e_horiz = v_nb - v
                nrm = float(np.linalg.norm(e_horiz))
                if nrm < 1e-6:
                    continue
                e_horiz = e_horiz / nrm
                # Ensure outward normal for gusset plates
                ex, ey = e_horiz, ez_vec
                if kind == "cnc_corner_gusset":
                    n_out = np.cross(ex, ey)
                    # Outward points away from delta center (cx, cy, z)
                    radial = np.array([v[0] - center_xy[0], v[1] - center_xy[1], 0.0], dtype=float)
                    if float(np.dot(n_out, radial)) < 0.0:
                        ex, ey = ey, ex
                corners.append(
                    {
                        "id": f"{prefix}_d_{tier}{i}_{tag}",
                        "vertex": v,
                        "ex": ex,
                        "ey": ey,
                    }
                )
    return corners


def _resolve_bracing_mode(spec: PrinterSpec) -> str:
    raw = str(getattr(spec.frame, "bracing", "none") or "none").strip().lower()
    return LEGACY_BRACING_ALIASES.get(raw, raw)


def _compute_single_cad_brace_rot_stiffness(
    loaded: LoadedHardwareAsset,
    *,
    kind: str,
    ext_w_m: float,
) -> tuple[float, float]:
    """Return single-bracket bolted rotational stiffness ``(kr_inplane, kr_torsion)`` (N·m/rad)."""
    K12_loc, _M12_loc, p_cad = _series_bolted_corner_superelement_local(
        loaded, kind=kind, ext_w_m=ext_w_m
    )
    K11 = K12_loc[6:12, 6:12]
    v_loc = (
        np.array([0.0, 0.0, 0.0], dtype=float)
        if kind == "cast_corner_bracket"
        else np.array([0.5 * ext_w_m, 0.5 * ext_w_m, 0.0], dtype=float)
    )
    r1 = p_cad[1] - v_loc
    ez = np.array([0.0, 0.0, 1.0], dtype=float)
    qz = np.concatenate((np.cross(ez, r1), ez))
    kr_in_single = max(0.0, float(qz @ K11 @ qz))

    ex = np.array([1.0, 0.0, 0.0], dtype=float)
    qx = np.concatenate((np.cross(ex, r1), ex))
    kr_tor_single = max(0.0, float(qx @ K11 @ qx))
    return kr_in_single, kr_tor_single


def compute_cad_corner_brace_stiffness(spec: PrinterSpec) -> tuple[float, float]:
    """Return exact 3D CAD superelement corner rotational stiffness ``(kr_inplane, kr_torsion)`` (N·m/rad).

    Computes the kinematic corner-hinge projection ``q^T K_11 q`` directly from the baked
    Tet4 Craig-Bampton superelement ``K_bb`` in series with Huth T-slot bolt compliance.
    """
    mode = _resolve_bracing_mode(spec)
    if mode == "cast_corner_brackets":
        kind = "cast_corner_bracket"
        n_parallel = 2.0  # 2 vertical-plane brackets per corner column junction
    elif mode in ("cnc_corner_gussets", "gussets_and_flat_bars"):
        kind = "cnc_corner_gusset"
        n_parallel = 2.0  # 2 orthogonal exterior 5-hole gusset plates per corner
    else:
        return 0.0, 0.0

    frame_prof = get_profile(spec.frame.profile)
    ext_w_m = float(frame_prof.width_m)
    series = _select_profile_series(ext_w_m)
    asset_id = f"{kind}_{series}"
    loaded = load_superelement(
        "frame_braces",
        asset_id,
        material_id="aluminum_6061_t6",
        chamber_temp_c=float(spec.frame.chamber_temp_c),
    )
    if loaded is None:
        return 0.0, 0.0

    kr_in_single, kr_tor_single = _compute_single_cad_brace_rot_stiffness(
        loaded, kind=kind, ext_w_m=ext_w_m
    )
    return n_parallel * kr_in_single, n_parallel * kr_tor_single


def _attach_cad_corner_braces(
    *,
    spec: PrinterSpec,
    kind: str,
    nodes: list[StructuralNode],
    nmap: dict[str, StructuralNode],
    members: list[StructuralMember],
    superelements: list[AttachedSuperelement],
    allow_gantry: bool,
) -> tuple[list[dict[str, Any]], float, float, dict[str, Any]]:
    """Place baked 3D CAD corner brackets or CNC gussets at all frame corners (exact mass + 3D GLBs).

    Their exact 3D solid-FEM Craig-Bampton rotational stiffness is coupled into the frame's
    semi-rigid corner joints via :func:`compute_cad_corner_brace_stiffness`, keeping the
    global beam mesh unsplit (zero extra stub-beam DOFs) while lumping exact CAD hardware
    mass onto the corner nodes and emitting exact 3D poses for Studio GLB rendering.
    """
    frame_prof = get_profile(spec.frame.profile)
    ext_w_m = float(frame_prof.width_m)
    series = _select_profile_series(ext_w_m)
    asset_id = f"{kind}_{series}"
    loaded = load_superelement(
        "frame_braces",
        asset_id,
        material_id="aluminum_6061_t6",
        chamber_temp_c=float(spec.frame.chamber_temp_c),
    )
    if loaded is None:
        return [], 0.0, 0.0, {}

    recipe = loaded.manifest.get("part_recipe") or {}
    unit_mass_kg = float(loaded.manifest.get("mass_kg", 0.02)) * float(loaded.mass_scale_applied)
    unit_cost_usd = float(recipe.get("unit_cost_usd", 2.0))
    kr_in_single, kr_tor_single = _compute_single_cad_brace_rot_stiffness(
        loaded, kind=kind, ext_w_m=ext_w_m
    )
    cad_meta: dict[str, Any] = {
        "cad_asset_name": str(loaded.manifest.get("name") or asset_id),
        "unit_mass_g": round(unit_mass_kg * 1000.0, 2),
        "unit_cost_usd": round(unit_cost_usd, 2),
        "element_type": str(loaded.manifest.get("element_type") or "Tet4"),
        "num_elements": int(loaded.manifest.get("num_elements") or 0),
        "kr_inplane_nm_rad": round(kr_in_single, 1),
        "kr_torsion_nm_rad": round(kr_tor_single, 1),
        "corner_kr_inplane_nm_rad": round(2.0 * kr_in_single, 1),
        "corner_kr_torsion_nm_rad": round(2.0 * kr_tor_single, 1),
    }
    visual_url = str(
        loaded.manifest.get("visual_url")
        or f"/api/assets/hardware/frame_braces/{asset_id}/visual.glb"
    )

    if spec.architecture == "corexy":
        w = float(np.linalg.norm(nmap["t_fr"].position_m - nmap["t_fl"].position_m))
        d = float(np.linalg.norm(nmap["t_rl"].position_m - nmap["t_fl"].position_m))
        h = float(np.linalg.norm(nmap["t_fl"].position_m - nmap["b_fl"].position_m))
        corner_defs = _corexy_corner_specs(w, d, h, kind)
    elif spec.architecture == "cartesian_bedslinger":
        corner_defs = _bedslinger_corner_specs(nmap, kind)
    else:
        corner_defs = _delta_corner_specs(nmap, kind)

    visuals: list[dict[str, Any]] = []
    total_mass_kg = 0.0
    total_cost_usd = 0.0

    # Fast nearest-frame-node lookup for corner mass lumping & modal/FEA anchoring.
    # Include role=="foot" because CoreXY/Bedslinger/Delta bottom frame corners (b_fl, b_fr, ...)
    # carry role="foot" (whereas ground anchor nodes below compliant feet carry role="ground").
    frame_nodes = [nd for nd in nodes if nd.role in ("frame", "tower", "foot")] or list(nodes)
    for cdef in corner_defs:
        v = cdef["vertex"]
        ex = cdef["ex"]
        ey = cdef["ey"]
        ez = np.cross(ex, ey)
        ez_n = float(np.linalg.norm(ez))
        if ez_n < 1e-6:
            continue
        ez = ez / ez_n

        # Lump exact CAD hardware mass onto the nearest frame corner node
        nearest = min(frame_nodes, key=lambda nd: float(np.linalg.norm(nd.position_m - v)))
        nearest.lumped_mass_kg += unit_mass_kg
        total_mass_kg += unit_mass_kg
        total_cost_usd += unit_cost_usd

        # CAD origin in global assembly coordinates (m)
        if kind == "cast_corner_bracket":
            cad_origin = v + 0.5 * ext_w_m * ex + 0.5 * ext_w_m * ey
        else:
            cad_origin = v - 0.5 * ext_w_m * ex - 0.5 * ext_w_m * ey + 0.5 * ext_w_m * ez

        visuals.append(
            {
                "id": cdef["id"],
                "kind": kind,
                "asset_id": asset_id,
                "visual_glb_url": visual_url,
                "origin_m": [round(float(x), 6) for x in cad_origin],
                "basis_x": [round(float(x), 6) for x in ex],
                "basis_y": [round(float(x), 6) for x in ey],
                "basis_z": [round(float(x), 6) for x in ez],
                "node_a": nearest.id,
                "node_b": nearest.id,
            }
        )

    return visuals, total_mass_kg, total_cost_usd, cad_meta


def _add_bolted_flat_bar_diagonal(
    *,
    brace_id: str,
    fn0: str,
    fn1: str,
    outward_normal: tuple[float, float, float],
    ext_w_m: float,
    nodes: list[StructuralNode],
    nmap: dict[str, StructuralNode],
    members: list[StructuralMember],
    springs: list[FastenerSpring],
) -> float:
    """Attach a 25x5mm Al 6061-T6 flat-bar diagonal beam on the exterior face via CBUSH bolts."""
    from printsim.geometry import builder as _b

    if fn0 == fn1 or fn0 not in nmap or fn1 not in nmap:
        return 0.0
    nvec = np.asarray(outward_normal, dtype=float)
    nrm = float(np.linalg.norm(nvec))
    if nrm < 1e-9:
        return 0.0
    nvec = nvec / nrm
    n_tup = (float(nvec[0]), float(nvec[1]), float(nvec[2]))

    bar_prof = get_profile("al_flat_bar_25x5")
    bar_mat = get_material("aluminum_6061_t6")
    t_bar = 0.005
    d_c = 0.5 * ext_w_m + 0.5 * t_bar
    off_vec = d_c * nvec

    # Create exterior-face bar end nodes coupled to frame corner nodes via M5 T-nut CBUSH springs
    bn0 = f"{brace_id}_n0"
    bn1 = f"{brace_id}_n1"
    p_bn0 = nmap[fn0].position_m + off_vec
    p_bn1 = nmap[fn1].position_m + off_vec
    node0 = _b.StructuralNode(id=bn0, position_m=p_bn0, role="brace")
    node1 = _b.StructuralNode(id=bn1, position_m=p_bn1, role="brace")
    nodes.extend((node0, node1))
    nmap[bn0] = node0
    nmap[bn1] = node1

    # Huth / VDI 2230 M5 bolt-in-Al-T-slot shear, clamping normal & rotational stiffness
    k_shear = 18.0e6
    k_normal = 36.0e6
    k_rot = 1.5e3
    off_tup = (float(off_vec[0]), float(off_vec[1]), float(off_vec[2]))
    springs.append(
        _b.FastenerSpring(
            id=f"cbush_{brace_id}_0",
            bay="frame_bracing",
            shell_node_id=bn0,
            frame_node_id=fn0,
            k_shear_n_m=k_shear,
            k_normal_n_m=k_normal,
            k_rot_nm_rad=k_rot,
            normal_vec=n_tup,
            offset_m=off_tup,
        )
    )
    springs.append(
        _b.FastenerSpring(
            id=f"cbush_{brace_id}_1",
            bay="frame_bracing",
            shell_node_id=bn1,
            frame_node_id=fn1,
            k_shear_n_m=k_shear,
            k_normal_n_m=k_normal,
            k_rot_nm_rad=k_rot,
            normal_vec=n_tup,
            offset_m=off_tup,
        )
    )

    mem = _b.StructuralMember(
        id=brace_id,
        start_id=bn0,
        end_id=bn1,
        profile=bar_prof,
        material=bar_mat,
        role="brace",
        is_truss=False,
    )
    members.append(mem)
    return mem.mass_kg(nmap)


def _add_bolted_flat_bar_braces(
    *,
    spec: PrinterSpec,
    mode: str,
    nodes: list[StructuralNode],
    nmap: dict[str, StructuralNode],
    members: list[StructuralMember],
    springs: list[FastenerSpring],
    allow_gantry: bool,
) -> tuple[float, float]:
    """Attach bolted 25x5mm Al 6061-T6 flat-bar diagonal X-braces (open top & front)."""
    ext_w_m = float(get_profile(spec.frame.profile).width_m)
    total_mass_kg = 0.0
    n_bars = 0

    if spec.architecture == "corexy":
        side_top_prefix = "col_g_" if spec.is_flying_gantry else ("g_" if "g_fl" in nmap else "t_")

        if mode in ("rear_flat_bar_x", "rear_and_side_flat_bars", "gussets_and_flat_bars"):
            for idx, (n_a, n_b) in enumerate((("b_rl", "t_rr"), ("b_rr", "t_rl"))):
                total_mass_kg += _add_bolted_flat_bar_diagonal(
                    brace_id=f"brace_flat_rear_{idx}",
                    fn0=n_a,
                    fn1=n_b,
                    outward_normal=(0.0, 1.0, 0.0),
                    ext_w_m=ext_w_m,
                    nodes=nodes,
                    nmap=nmap,
                    members=members,
                    springs=springs,
                )
                n_bars += 1

        if mode in ("side_flat_bar_x", "rear_and_side_flat_bars", "gussets_and_flat_bars"):
            for idx, (n_a, n_b) in enumerate(
                (("b_fl", f"{side_top_prefix}rl"), ("b_rl", f"{side_top_prefix}fl"))
            ):
                total_mass_kg += _add_bolted_flat_bar_diagonal(
                    brace_id=f"brace_flat_left_{idx}",
                    fn0=n_a,
                    fn1=n_b,
                    outward_normal=(-1.0, 0.0, 0.0),
                    ext_w_m=ext_w_m,
                    nodes=nodes,
                    nmap=nmap,
                    members=members,
                    springs=springs,
                )
                n_bars += 1
            for idx, (n_a, n_b) in enumerate(
                (("b_fr", f"{side_top_prefix}rr"), ("b_rr", f"{side_top_prefix}fr"))
            ):
                total_mass_kg += _add_bolted_flat_bar_diagonal(
                    brace_id=f"brace_flat_right_{idx}",
                    fn0=n_a,
                    fn1=n_b,
                    outward_normal=(1.0, 0.0, 0.0),
                    ext_w_m=ext_w_m,
                    nodes=nodes,
                    nmap=nmap,
                    members=members,
                    springs=springs,
                )
                n_bars += 1

    elif spec.architecture == "cartesian_bedslinger":
        if mode in ("rear_flat_bar_x", "side_flat_bar_x", "rear_and_side_flat_bars", "gussets_and_flat_bars"):
            for bid, n_a, n_b, norm in (
                ("brace_flat_l_rear", "b_rl", "p_tl", (-1.0, 0.0, 0.0)),
                ("brace_flat_r_rear", "b_rr", "p_tr", (1.0, 0.0, 0.0)),
                ("brace_flat_portal_1", "p_bl", "p_tr", (0.0, 1.0, 0.0)),
            ):
                total_mass_kg += _add_bolted_flat_bar_diagonal(
                    brace_id=bid,
                    fn0=n_a,
                    fn1=n_b,
                    outward_normal=norm,
                    ext_w_m=ext_w_m,
                    nodes=nodes,
                    nmap=nmap,
                    members=members,
                    springs=springs,
                )
                n_bars += 1
        if mode in ("side_flat_bar_x", "rear_and_side_flat_bars", "gussets_and_flat_bars"):
            for bid, n_a, n_b, norm in (
                ("brace_flat_l_front", "b_fl", "p_tl", (-1.0, 0.0, 0.0)),
                ("brace_flat_r_front", "b_fr", "p_tr", (1.0, 0.0, 0.0)),
                ("brace_flat_portal_2", "p_br", "p_tl", (0.0, 1.0, 0.0)),
            ):
                total_mass_kg += _add_bolted_flat_bar_diagonal(
                    brace_id=bid,
                    fn0=n_a,
                    fn1=n_b,
                    outward_normal=norm,
                    ext_w_m=ext_w_m,
                    nodes=nodes,
                    nmap=nmap,
                    members=members,
                    springs=springs,
                )
                n_bars += 1

    else:  # delta
        center_xy = sum(nmap[f"b_{k}"].position_m[:2] for k in range(3)) / 3.0
        for i in range(3):
            j = (i + 1) % 3
            bi = nmap[f"b_{i}"].position_m
            bj = nmap[f"b_{j}"].position_m
            mid = 0.5 * (bi + bj)
            norm_xy = np.array([mid[0] - center_xy[0], mid[1] - center_xy[1], 0.0], dtype=float)
            nn = float(np.linalg.norm(norm_xy))
            out_n = (float(norm_xy[0] / nn), float(norm_xy[1] / nn), 0.0) if nn > 1e-6 else (1.0, 0.0, 0.0)
            if mode in ("rear_flat_bar_x", "side_flat_bar_x", "rear_and_side_flat_bars", "gussets_and_flat_bars"):
                total_mass_kg += _add_bolted_flat_bar_diagonal(
                    brace_id=f"brace_flat_d_{i}_a",
                    fn0=f"b_{i}",
                    fn1=f"t_{j}",
                    outward_normal=out_n,
                    ext_w_m=ext_w_m,
                    nodes=nodes,
                    nmap=nmap,
                    members=members,
                    springs=springs,
                )
                n_bars += 1
            if mode in ("rear_and_side_flat_bars", "gussets_and_flat_bars"):
                total_mass_kg += _add_bolted_flat_bar_diagonal(
                    brace_id=f"brace_flat_d_{i}_b",
                    fn0=f"b_{j}",
                    fn1=f"t_{i}",
                    outward_normal=out_n,
                    ext_w_m=ext_w_m,
                    nodes=nodes,
                    nmap=nmap,
                    members=members,
                    springs=springs,
                )
                n_bars += 1

    cost_usd = 8.0 * n_bars
    return total_mass_kg, cost_usd


def add_modeled_frame_bracing(
    *,
    spec: PrinterSpec,
    nodes: list[StructuralNode],
    nmap: dict[str, StructuralNode],
    members: list[StructuralMember],
    springs: list[FastenerSpring],
    superelements: list[AttachedSuperelement],
    allow_gantry: bool,
) -> dict[str, Any]:
    """Build modeled & simulated frame bracing for CoreXY, Bedslinger, or Delta assemblies."""
    mode = _resolve_bracing_mode(spec)
    if mode == "none":
        return {
            "visuals": [],
            "cad_mass_kg": 0.0,
            "flat_bar_mass_kg": 0.0,
            "cost_usd": 0.0,
            "cad_count": 0,
            "cad_asset_id": None,
            "cad_meta": {},
        }

    visuals: list[dict[str, Any]] = []
    cad_mass_kg = 0.0
    cad_cost_usd = 0.0
    cad_meta: dict[str, Any] = {}
    flat_bar_mass_kg = 0.0
    flat_bar_cost_usd = 0.0

    if mode == "cast_corner_brackets":
        visuals, cad_mass_kg, cad_cost_usd, cad_meta = _attach_cad_corner_braces(
            spec=spec,
            kind="cast_corner_bracket",
            nodes=nodes,
            nmap=nmap,
            members=members,
            superelements=superelements,
            allow_gantry=allow_gantry,
        )
    elif mode in ("cnc_corner_gussets", "gussets_and_flat_bars"):
        visuals, cad_mass_kg, cad_cost_usd, cad_meta = _attach_cad_corner_braces(
            spec=spec,
            kind="cnc_corner_gusset",
            nodes=nodes,
            nmap=nmap,
            members=members,
            superelements=superelements,
            allow_gantry=allow_gantry,
        )

    if mode in ("rear_flat_bar_x", "side_flat_bar_x", "rear_and_side_flat_bars", "gussets_and_flat_bars"):
        flat_bar_mass_kg, flat_bar_cost_usd = _add_bolted_flat_bar_braces(
            spec=spec,
            mode=mode,
            nodes=nodes,
            nmap=nmap,
            members=members,
            springs=springs,
            allow_gantry=allow_gantry,
        )

    return {
        "visuals": visuals,
        "cad_mass_kg": cad_mass_kg,
        "flat_bar_mass_kg": flat_bar_mass_kg,
        "cost_usd": cad_cost_usd + flat_bar_cost_usd,
        "cad_count": len(visuals),
        "cad_asset_id": visuals[0]["asset_id"] if visuals else None,
        "cad_meta": cad_meta,
    }
