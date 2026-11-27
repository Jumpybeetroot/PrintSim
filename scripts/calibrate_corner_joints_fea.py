#!/usr/bin/env python3
"""3D Tet10 Continuum FEA Submodel Calibration for 2020 Frame Corner Joints (CAL-09).

Constructs realistic 3D OpenCASCADE CAD submodels (``gmsh.model.occ``) of Misumi HFS5-2020
aluminum T-slot L-frame corner joints and solves 3D linear elasticity with ``scikit-fem``
Tet10 (P2) elements to extract semi-rigid corner joint stiffnesses:
  - ``k_t_n_m``: translational / shear joint stiffness (N/m)
  - ``k_r_inplane_nm_rad``: in-plane bending moment-rotation stiffness (N*m/rad)
  - ``k_r_torsion_nm_rad``: out-of-plane / torsional joint stiffness (N*m/rad)

Physics & Contact Model (Back, Burdekin & Cowley 1973; Yoshimura 1979; Sherif & Kossa 1991; VDI 2230):
  - Misumi HFS5-2020 T-slot cross-section (1.65 mm T-slot lips, 6.2 mm slot opening, 4.6 mm
    center bore, A ≈ 186 mm^2, I ≈ 0.75 cm^4, Al 6063-T5 E = 69 GPa, nu = 0.33).
  - Preloaded M5/M6 steel SHCS fasteners & T-nuts (E = 207 GPa, nu = 0.29, nominal preload
    F_p ≈ 3.5-5.0 kN at 3.5-5.0 N*m torque -> mean clamped pressure p ≈ 25-35 MPa).
  - Interfacial anodized-Al bolted contact pads (t_pad = 0.60 mm) placed ONLY inside each
    fastener's VDI-2230 clamping frustum, with Yoshimura / Back-Burdekin areal contact
    stiffness k_n = 12.0 GPa/mm (E_n = k_n * t_pad = 7.20 GPa) and Mindlin tangential
    micro-slip stiffness k_t = 3.6 GPa/mm (G_t = 0.30 * E_n = 2.16 GPa). Single-bolt butt
    faces apply a 0.65x unilateral tension-side decompression factor under bending moment.
  - Exact 2x2 compliance transformation to corner centerline intersection O = (0, 0, Z_c) via
    ``C_O = T @ C_tip @ T.T`` separates pure translational/shear compliance from rotational
    compliance without lever-arm pollution.

Joint architectures modeled (matching ``CORNER_JOINT_BASE_2020``):
  1. ``rigid_control``: monolithic continuous 2020 L-frame reference.
  2. ``l_bracket``: standard 20x20x17 mm die-cast aluminum 90-deg 2020 internal corner bracket
     (2x M5 SHCS + T-nuts, 3.5 mm walls, unpreloaded butt gap).
  3. ``blind_joint``: Voron standard M5 SHCS blind joint (M5 head bearing on Column A's
     1.65 mm inner T-slot lips, shank threaded 14 mm into Beam B core, butt-face frustum pads).
  4. ``gusset_plate``: 60x60x5.0 mm CNC 6061-T6 external 5-hole L-gusset plate bolted on the
     front face (Y = -10 mm) with 5x M5 bolts + T-nuts + VDI-2230 bolt frustum pads. Also
     compared against ``gusset_welded_control`` to isolate pure 5-bolt T-nut joint compliance.
  5. ``corner_cube``: 20x20x20 mm 3-way aluminum corner cube (Misumi HBLCR5 style) with
     T-slot locating anti-rotation tabs and M5/M6 end-bore bolts clamping both full 20x20
     extrusion end faces onto solid cube faces (normalized per member end since 2 interfaces
     act in series along the L-specimen).

Run:
    uv run python scripts/calibrate_corner_joints_fea.py
    uv run python scripts/calibrate_corner_joints_fea.py --order 2 --h-bulk 6.0
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass

import gmsh
import numpy as np
import skfem
from scipy.sparse.linalg import splu


# ---------------------------------------------------------------------------
# 1. Constitutive Matrices (strict SI: Pa)
# ---------------------------------------------------------------------------


def iso_constitutive_3d(e_pa: float, nu: float) -> np.ndarray:
    """Build SPD 6x6 3D isotropic elasticity tensor D (Voigt [xx, yy, zz, xy, yz, xz])."""
    c = e_pa / ((1.0 + nu) * (1.0 - 2.0 * nu))
    d_mat = np.zeros((6, 6), dtype=np.float64)
    d_mat[0:3, 0:3] = c * np.array(
        [
            [1.0 - nu, nu, nu],
            [nu, 1.0 - nu, nu],
            [nu, nu, 1.0 - nu],
        ],
        dtype=np.float64,
    )
    g = e_pa / (2.0 * (1.0 + nu))
    d_mat[3, 3] = d_mat[4, 4] = d_mat[5, 5] = g
    return d_mat


def mindlin_contact_pad_3d(
    normal_axis: str,
    e_norm_pa: float,
    g_tang_pa: float,
) -> np.ndarray:
    """Build orthotropic Mindlin preloaded bolted contact pad elasticity tensor."""
    e_t = max(2.6 * g_tang_pa, 1.0e7)
    g_t = max(g_tang_pa, 5.0e6)
    e_n = max(e_norm_pa, 5.0e7)
    if normal_axis == "x":
        diag_s = [1.0 / e_n, 1.0 / e_t, 1.0 / e_t, 1.0 / g_t, 1.0 / e_t, 1.0 / g_t]
    elif normal_axis == "y":
        diag_s = [1.0 / e_t, 1.0 / e_n, 1.0 / e_t, 1.0 / g_t, 1.0 / g_t, 1.0 / e_t]
    else:
        diag_s = [1.0 / e_t, 1.0 / e_t, 1.0 / e_n, 1.0 / e_t, 1.0 / g_t, 1.0 / g_t]
    return np.diag([1.0 / s for s in diag_s])


# ---------------------------------------------------------------------------
# 2. OpenCASCADE Geometry Builders (mm)
# ---------------------------------------------------------------------------

Z_TOP = 150.0  # Top of Column A (mm)
Z_CENTER = 140.0  # Beam B centerline Z (mm)
X_TIP = 160.0  # Beam B tip X (mm) -> L_O from corner centerline X=0 is 160 mm
T_PAD = 0.60  # Interfacial Mindlin pad thickness (mm)


def _make_2020_tslot_face() -> int:
    """Create connected Misumi HFS5-2020 T-slot face in XY plane (A ~ 186 mm^2, I ~ 0.75 cm^4)."""
    occ = gmsh.model.occ
    outer = occ.addRectangle(-10.0, -10.0, 0.0, 20.0, 20.0)
    bore = occ.addDisk(0.0, 0.0, 0.0, 2.3, 2.3)
    cuts = [(2, bore)]
    for s in (1, -1):
        cuts.append((2, occ.addRectangle(8.35 if s > 0 else -10.5, -3.1, 0.0, 2.15, 6.2)))
        cuts.append((2, occ.addRectangle(5.55 if s > 0 else -8.35, -5.4, 0.0, 2.8, 10.8)))
        cuts.append((2, occ.addRectangle(4.1 if s > 0 else -5.55, -3.1, 0.0, 1.45, 6.2)))
        cuts.append((2, occ.addRectangle(-3.1, 8.35 if s > 0 else -10.5, 0.0, 6.2, 2.15)))
        cuts.append((2, occ.addRectangle(-5.4, 5.55 if s > 0 else -8.35, 0.0, 10.8, 2.8)))
        cuts.append((2, occ.addRectangle(-3.1, 4.1 if s > 0 else -5.55, 0.0, 6.2, 1.45)))
    res, _ = occ.cut([(2, outer)], cuts)
    if len(res) != 1:
        raise RuntimeError(f"Expected 1 connected 2020 face, got {len(res)}")
    return res[0][1]


def _extrude_z(face_tag: int, z0: float, z1: float) -> list[tuple[int, int]]:
    occ = gmsh.model.occ
    ext = occ.extrude([(2, face_tag)], 0.0, 0.0, z1 - z0)
    vols = [e for e in ext if e[0] == 3]
    if abs(z0) > 1e-9:
        occ.translate(vols, 0.0, 0.0, z0)
    return vols


def _extrude_x_at_z(face_tag: int, x0: float, x1: float, z_center: float) -> list[tuple[int, int]]:
    occ = gmsh.model.occ
    ext = occ.extrude([(2, face_tag)], 0.0, 0.0, x1 - x0)
    vols = [e for e in ext if e[0] == 3]
    occ.rotate(vols, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, math.pi / 2.0)
    occ.translate(vols, x0, 0.0, z_center)
    return vols


def _fragment_and_tag(group_lists: list[tuple[int, list[tuple[int, int]]]]) -> None:
    """Fragment ordered priority volume groups and assign Gmsh Physical Groups."""
    occ = gmsh.model.occ
    flat_inputs: list[tuple[int, int]] = []
    counts: list[tuple[int, int]] = []
    for gid, vols in group_lists:
        if vols:
            counts.append((gid, len(vols)))
            flat_inputs.extend(vols)

    if len(flat_inputs) == 1:
        occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [flat_inputs[0][1]], counts[0][0])
        return

    _, out_map = occ.fragment(flat_inputs[:1], flat_inputs[1:])
    occ.synchronize()

    claimed: set[int] = set()
    off = 0
    for gid, cnt in counts:
        raw: set[int] = set()
        for k in range(off, off + cnt):
            for d, tg in out_map[k]:
                if d == 3:
                    raw.add(tg)
        off += cnt
        owned = sorted(raw - claimed)
        claimed |= set(owned)
        if owned:
            gmsh.model.addPhysicalGroup(3, owned, gid)


# ---------------------------------------------------------------------------
# 3. Joint Specimen Builders
# ---------------------------------------------------------------------------


def _build_rigid_control() -> None:
    occ = gmsh.model.occ
    col_a = _extrude_z(_make_2020_tslot_face(), 0.0, Z_TOP)
    beam_b = _extrude_x_at_z(_make_2020_tslot_face(), 0.0, X_TIP, Z_CENTER)
    fused, _ = occ.fuse(col_a, beam_b)
    _fragment_and_tag([(1, fused)])


def _build_gusset_welded_control() -> None:
    """Monolithic welded 5-hole gusset reference (isolates bolted-interface compliance of gusset_plate)."""
    occ = gmsh.model.occ
    col_a = _extrude_z(_make_2020_tslot_face(), 0.0, Z_TOP)
    beam_b = _extrude_x_at_z(_make_2020_tslot_face(), 0.0, X_TIP, Z_CENTER)
    pts_xz = [
        (-10.0, Z_TOP),
        (50.0, Z_TOP),
        (50.0, Z_TOP - 20.0),
        (10.0, Z_TOP - 60.0),
        (-10.0, Z_TOP - 60.0),
    ]
    pt_tags = [occ.addPoint(px, -10.0, pz) for px, pz in pts_xz]
    lines = [occ.addLine(pt_tags[i], pt_tags[(i + 1) % len(pt_tags)]) for i in range(len(pt_tags))]
    cl = occ.addCurveLoop(lines)
    sf = occ.addPlaneSurface([cl])
    ext = occ.extrude([(2, sf)], 0.0, -5.0, 0.0)
    plate_vols = [e for e in ext if e[0] == 3]
    fused, _ = occ.fuse(col_a, beam_b + plate_vols)
    _fragment_and_tag([(1, fused)])


def _build_cube_welded_control() -> None:
    """Monolithic 20x20x20 solid corner cube fused to extrusions (isolates bolted end-face compliance)."""
    occ = gmsh.model.occ
    z_cube_bot = Z_CENTER - 10.0
    col_a = _extrude_z(_make_2020_tslot_face(), 0.0, z_cube_bot)
    beam_b = _extrude_x_at_z(_make_2020_tslot_face(), 10.0, X_TIP, Z_CENTER)
    cube = occ.addBox(-10.0, -10.0, z_cube_bot, 20.0, 20.0, 20.0)
    fused, _ = occ.fuse([(3, cube)], col_a + beam_b)
    _fragment_and_tag([(1, fused)])


def _build_blind_joint() -> None:
    """Voron standard M5 SHCS blind joint on 2020 extrusion."""
    occ = gmsh.model.occ
    col_a_raw = _extrude_z(_make_2020_tslot_face(), 0.0, Z_TOP)
    access_hole = occ.addCylinder(-11.0, 0.0, Z_CENTER, 15.0, 0.0, 0.0, 2.75)
    col_a, _ = occ.cut(col_a_raw, [(3, access_hole)])

    beam_b = _extrude_x_at_z(_make_2020_tslot_face(), 10.0 + T_PAD, X_TIP, Z_CENTER)

    bolt_head = occ.addBox(5.90, -4.40, Z_CENTER - 4.40, 2.80, 8.80, 8.80)
    bolt_shank = occ.addBox(8.35, -2.50, Z_CENTER - 2.50, 16.65, 5.00, 5.00)
    bolt_vols, _ = occ.fuse([(3, bolt_head)], [(3, bolt_shank)])

    # Anodized butt-face Mindlin pads within the M5 VDI-2230 frustum (flanking A's 6.2 mm T-slot)
    pad_pos = occ.addBox(10.0, 3.2, Z_CENTER - 9.0, T_PAD, 6.4, 18.0)
    pad_neg = occ.addBox(10.0, -9.6, Z_CENTER - 9.0, T_PAD, 6.4, 18.0)
    pads = [(3, pad_pos), (3, pad_neg)]

    _fragment_and_tag([(3, bolt_vols), (1, col_a + beam_b), (7, pads)])


def _build_l_bracket() -> None:
    """Standard 20x20x17 mm die-cast aluminum 90-deg 2020 internal corner L-bracket (2x M5 + T-nuts)."""
    occ = gmsh.model.occ
    col_a = _extrude_z(_make_2020_tslot_face(), 0.0, Z_TOP)
    beam_b = _extrude_x_at_z(_make_2020_tslot_face(), 10.60, X_TIP, Z_CENTER)

    x0 = 10.0 + T_PAD
    z0 = (Z_CENTER - 10.0) - T_PAD
    arm = 20.0
    wall = 3.5
    w_bkt = 17.0
    rib_t = 2.6

    flange_b = occ.addBox(x0, -0.5 * w_bkt, z0 - wall, arm, w_bkt, wall)
    flange_a = occ.addBox(x0, -0.5 * w_bkt, z0 - arm, wall, w_bkt, arm)
    bkt_parts = [(3, flange_b), (3, flange_a)]

    for sy in (-1.0, 1.0):
        y_min = -0.5 * w_bkt if sy < 0 else 0.5 * w_bkt - rib_t
        p1 = occ.addPoint(x0 + wall - 0.3, y_min, z0 - wall + 0.3)
        p2 = occ.addPoint(x0 + arm - 1.5, y_min, z0 - wall + 0.3)
        p3 = occ.addPoint(x0 + arm - 1.5, y_min, z0 - wall - 1.8)
        p4 = occ.addPoint(x0 + wall + 1.8, y_min, z0 - arm + 1.5)
        p5 = occ.addPoint(x0 + wall - 0.3, y_min, z0 - arm + 1.5)
        lines = [
            occ.addLine(p1, p2),
            occ.addLine(p2, p3),
            occ.addLine(p3, p4),
            occ.addLine(p4, p5),
            occ.addLine(p5, p1),
        ]
        cl = occ.addCurveLoop(lines)
        sf = occ.addPlaneSurface([cl])
        ext = occ.extrude([(2, sf)], 0.0, rib_t, 0.0)
        bkt_parts.extend([e for e in ext if e[0] == 3])

    bkt_fused, _ = occ.fuse(bkt_parts[:1], bkt_parts[1:])

    z_ba = z0 - 11.0
    x_bb = x0 + 11.0
    tnut_a = occ.addBox(5.90, -4.8, z_ba - 4.8, 2.75, 9.6, 9.6)
    bolt_a = occ.addBox(7.0, -2.2, z_ba - 2.2, 8.8, 4.4, 4.4)
    tnut_b = occ.addBox(x_bb - 4.8, -4.8, (Z_CENTER - 10.0) + 1.35, 9.6, 9.6, 2.75)
    bolt_b = occ.addBox(x_bb - 2.2, -2.2, z0 - wall - 1.5, 4.4, 4.4, 8.8)
    fasteners, _ = occ.fuse([(3, tnut_a)], [(3, bolt_a), (3, tnut_b), (3, bolt_b)])

    pads_x = [
        (3, occ.addBox(10.0, 3.2, z_ba - 6.0, T_PAD, 5.2, 12.0)),
        (3, occ.addBox(10.0, -8.4, z_ba - 6.0, T_PAD, 5.2, 12.0)),
    ]
    pads_z = [
        (3, occ.addBox(x_bb - 6.0, 3.2, (Z_CENTER - 10.0) - T_PAD, 12.0, 5.2, T_PAD)),
        (3, occ.addBox(x_bb - 6.0, -8.4, (Z_CENTER - 10.0) - T_PAD, 12.0, 5.2, T_PAD)),
    ]

    _fragment_and_tag(
        [
            (3, fasteners),
            (2, bkt_fused),
            (1, col_a + beam_b),
            (4, pads_x),
            (6, pads_z),
        ]
    )


def _build_gusset_plate() -> None:
    """5-hole CNC 6061-T6 60x60x5.0 mm external corner gusset plate (RatRig / Monolith style)."""
    occ = gmsh.model.occ
    col_a = _extrude_z(_make_2020_tslot_face(), 0.0, Z_TOP)
    beam_b = _extrude_x_at_z(_make_2020_tslot_face(), 10.0 + T_PAD, X_TIP, Z_CENTER)

    y_face = -10.0 - T_PAD
    t_plate = 5.0
    pts_xz = [
        (-10.0, Z_TOP),
        (50.0, Z_TOP),
        (50.0, Z_TOP - 20.0),
        (10.0, Z_TOP - 60.0),
        (-10.0, Z_TOP - 60.0),
    ]
    pt_tags = [occ.addPoint(px, y_face, pz) for px, pz in pts_xz]
    lines = [occ.addLine(pt_tags[i], pt_tags[(i + 1) % len(pt_tags)]) for i in range(len(pt_tags))]
    cl = occ.addCurveLoop(lines)
    sf = occ.addPlaneSurface([cl])
    ext = occ.extrude([(2, sf)], 0.0, -t_plate, 0.0)
    plate_vols = [e for e in ext if e[0] == 3]

    bolt_vols: list[tuple[int, int]] = []
    pads_y: list[tuple[int, int]] = []
    for zb in (Z_CENTER, Z_CENTER - 20.0, Z_CENTER - 40.0):
        tn = occ.addBox(-4.8, -8.65, zb - 4.8, 9.6, 2.75, 9.6)
        sh = occ.addBox(-2.2, y_face - t_plate - 1.5, zb - 2.2, 4.4, t_plate + T_PAD + 4.5, 4.4)
        bolt_vols.extend([(3, tn), (3, sh)])
        pads_y.append((3, occ.addBox(3.2, y_face, zb - 6.5, 6.4, T_PAD, 13.0)))
        pads_y.append((3, occ.addBox(-9.6, y_face, zb - 6.5, 6.4, T_PAD, 13.0)))

    for xb in (20.0, 40.0):
        tn = occ.addBox(xb - 4.8, -8.65, Z_CENTER - 4.8, 9.6, 2.75, 9.6)
        sh = occ.addBox(xb - 2.2, y_face - t_plate - 1.5, Z_CENTER - 2.2, 4.4, t_plate + T_PAD + 4.5, 4.4)
        bolt_vols.extend([(3, tn), (3, sh)])
        pads_y.append((3, occ.addBox(xb - 6.5, y_face, Z_CENTER + 3.2, 13.0, T_PAD, 6.4)))
        pads_y.append((3, occ.addBox(xb - 6.5, y_face, Z_CENTER - 9.6, 13.0, T_PAD, 6.4)))

    fasteners, _ = occ.fuse(bolt_vols[:1], bolt_vols[1:])

    pads_x = [
        (3, occ.addBox(10.0, 3.2, Z_CENTER - 9.0, T_PAD, 6.4, 18.0)),
        (3, occ.addBox(10.0, -9.6, Z_CENTER - 9.0, T_PAD, 6.4, 18.0)),
    ]

    _fragment_and_tag(
        [
            (3, fasteners),
            (2, plate_vols),
            (1, col_a + beam_b),
            (5, pads_y),
            (7, pads_x),
        ]
    )


def _build_corner_cube() -> None:
    """20x20x20 mm 3-way aluminum corner cube (Misumi HBLCR5 style) with T-slot tabs & M5/M6 end bolts."""
    occ = gmsh.model.occ
    z_cube_bot = Z_CENTER - 10.0  # 130.0 mm
    col_a = _extrude_z(_make_2020_tslot_face(), 0.0, z_cube_bot - T_PAD)
    beam_b = _extrude_x_at_z(_make_2020_tslot_face(), 10.0 + T_PAD, X_TIP, Z_CENTER)

    cube = occ.addBox(-10.0, -10.0, z_cube_bot, 20.0, 20.0, 20.0)
    # Anti-rotation locating tabs nesting 2.5 mm into the 6.2 mm T-slots of Column A & Beam B
    # (prevents torsional slip across the flat butt faces on real Misumi/OpenBuilds corner cubes)
    tabs = [
        (3, occ.addBox(-3.0, 6.0, z_cube_bot - 3.0, 6.0, 3.8, 3.5)),
        (3, occ.addBox(-3.0, -9.8, z_cube_bot - 3.0, 6.0, 3.8, 3.5)),
        (3, occ.addBox(9.5, 6.0, Z_CENTER - 3.0, 3.5, 3.8, 6.0)),
        (3, occ.addBox(9.5, -9.8, Z_CENTER - 3.0, 3.5, 3.8, 6.0)),
    ]
    cube_fused, _ = occ.fuse([(3, cube)], tabs)

    # Central M5/M6 end-bore bolts clamping Column A and Beam B flush against the cube
    bolt_a = occ.addBox(-2.6, -2.6, z_cube_bot - 15.0, 5.2, 5.2, 30.0)
    bolt_b = occ.addBox(-5.0, -2.6, Z_CENTER - 2.6, 30.0, 5.2, 5.2)
    fasteners, _ = occ.fuse([(3, bolt_a)], [(3, bolt_b)])

    pad_z = _extrude_z(_make_2020_tslot_face(), z_cube_bot - T_PAD, z_cube_bot)
    pad_x = _extrude_x_at_z(_make_2020_tslot_face(), 10.0, 10.0 + T_PAD, Z_CENTER)

    _fragment_and_tag(
        [
            (3, fasteners),
            (2, cube_fused),
            (1, col_a + beam_b),
            (6, pad_z),
            (4, pad_x),
        ]
    )


# ---------------------------------------------------------------------------
# 4. Meshing, Multi-Material Assembly & Exact 2x2 Corner Compliance Extraction
# ---------------------------------------------------------------------------


@dataclass
class JointFEAResult:
    joint_id: str
    order: int
    n_tets: int
    n_dofs: int
    solve_time_s: float
    # Work-conjugate tip responses
    c_ry_my: float  # theta_y / My (rad / N*m)
    c_uz_my: float  # u_z / My (m / N*m)
    c_ry_fz: float  # theta_y / Fz (rad / N)
    c_uz_fz: float  # u_z / Fz (m / N)
    c_rz_mz: float  # theta_z / Mz (rad / N*m)
    c_rx_mx: float  # theta_x / Mx (rad / N*m)
    c_ux_fx: float  # u_x / Fx (m / N)
    # Isolated single-joint stiffnesses (referenced to monolithic control)
    k_r_inplane_nm_rad: float | None = None
    k_r_outplane_nm_rad: float | None = None
    k_r_torsion_nm_rad: float | None = None
    k_t_axial_n_m: float | None = None
    k_t_shear_n_m: float | None = None
    k_t_eff_n_m: float | None = None


def _build_and_solve_specimen(
    joint_id: str,
    *,
    order: int = 2,
    h_bulk_mm: float = 6.5,
    k_n_gpa_per_mm: float = 12.0,
    k_t_gpa_per_mm: float = 3.6,
) -> JointFEAResult:
    """Mesh one 2020 corner-joint specimen with Gmsh and solve 5 load cases in scikit-fem."""
    t0 = time.perf_counter()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(f"joint_{joint_id}")

    builders = {
        "rigid_control": _build_rigid_control,
        "gusset_welded_control": _build_gusset_welded_control,
        "cube_welded_control": _build_cube_welded_control,
        "l_bracket": _build_l_bracket,
        "blind_joint": _build_blind_joint,
        "gusset_plate": _build_gusset_plate,
        "corner_cube": _build_corner_cube,
    }
    builders[joint_id]()

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.45)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h_bulk_mm * 1.45)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 10)
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h_bulk_mm)

    existing_gids = {gid for _, gid in gmsh.model.getPhysicalGroups(3)}
    for gid_ref, sz in ((2, 3.2), (3, 2.8), (4, 2.8), (5, 2.8), (6, 2.8), (7, 2.8)):
        if gid_ref not in existing_gids:
            continue
        ents = gmsh.model.getEntitiesForPhysicalGroup(3, gid_ref)
        if len(ents) > 0:
            pts = gmsh.model.getBoundary(
                [(3, e) for e in ents], combined=False, oriented=False, recursive=True
            )
            gmsh.model.mesh.setSize([p for p in pts if p[0] == 0], sz)

    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.model.mesh.generate(3)

    ntags, ncoords, _ = gmsh.model.mesh.getNodes()
    all_tets, all_gids = [], []
    for _, gid in gmsh.model.getPhysicalGroups(3):
        for ent in gmsh.model.getEntitiesForPhysicalGroup(3, gid):
            _, enodes = gmsh.model.mesh.getElementsByType(4, ent)
            if len(enodes) > 0:
                arr = np.asarray(enodes, dtype=np.int64).reshape(-1, 4)
                all_tets.append(arr)
                all_gids.append(np.full(len(arr), gid, dtype=np.int32))
    gmsh.finalize()

    raw_tets = np.vstack(all_tets)
    elem_gid = np.concatenate(all_gids)
    ntags = np.asarray(ntags, dtype=np.int64)
    coords_mm = np.asarray(ncoords, dtype=np.float64).reshape(-1, 3)
    order_idx = np.argsort(ntags)
    pos = np.searchsorted(ntags[order_idx], raw_tets.ravel())
    idx = order_idx[pos].reshape(-1, 4)
    used, inv = np.unique(idx, return_inverse=True)
    mesh = skfem.MeshTet((coords_mm[used] * 1e-3).T, inv.reshape(-1, 4).T).oriented()

    e_norm_pa = k_n_gpa_per_mm * T_PAD * 1e9
    g_tang_pa = k_t_gpa_per_mm * T_PAD * 1e9

    d_al_ext = iso_constitutive_3d(69.0e9, 0.33)
    d_hw = iso_constitutive_3d(71.0e9 if joint_id in ("l_bracket", "corner_cube") else 69.0e9, 0.33)
    d_steel = iso_constitutive_3d(207.0e9, 0.29)
    # Unilateral tension-side decompression factors:
    # - blind_joint butt face (1 central M5 bolt): 0.65x
    # - gusset_plate butt face (unbolted butt seam): 0.30x
    # - corner_cube full 20x20 end-face on solid cube: 0.85x
    g7_scale = 0.30 if joint_id == "gusset_plate" else 0.65
    g46_scale = 0.85 if joint_id == "corner_cube" else (0.75 if joint_id == "l_bracket" else 1.0)
    mats_by_gid = {
        1: d_al_ext,
        2: d_hw,
        3: d_steel,
        4: mindlin_contact_pad_3d("x", e_norm_pa * g46_scale, g_tang_pa * g46_scale),
        5: mindlin_contact_pad_3d("y", e_norm_pa, g_tang_pa),
        6: mindlin_contact_pad_3d("z", e_norm_pa * g46_scale, g_tang_pa * g46_scale),
        7: mindlin_contact_pad_3d("x", e_norm_pa * g7_scale, g_tang_pa * g7_scale),
    }

    scalar_elem = skfem.ElementTetP2() if order == 2 else skfem.ElementTetP1()
    intorder = 2 if order == 2 else 1
    element = skfem.ElementVector(scalar_elem)
    basis_full = skfem.CellBasis(mesh, element, intorder=intorder)

    k_tot = None
    for gid, d_mat in mats_by_gid.items():
        elems = np.nonzero(elem_gid == gid)[0]
        if len(elems) == 0:
            continue
        nz_rows = [[(j, float(d_mat[i, j])) for j in range(6) if abs(d_mat[i, j]) > 0.0] for i in range(6)]

        @skfem.BilinearForm
        def elas(u, v, w):
            g_u, g_v = u.grad, v.grad
            eu = (
                g_u[0, 0],
                g_u[1, 1],
                g_u[2, 2],
                g_u[0, 1] + g_u[1, 0],
                g_u[1, 2] + g_u[2, 1],
                g_u[0, 2] + g_u[2, 0],
            )
            ev = (
                g_v[0, 0],
                g_v[1, 1],
                g_v[2, 2],
                g_v[0, 1] + g_v[1, 0],
                g_v[1, 2] + g_v[2, 1],
                g_v[0, 2] + g_v[2, 0],
            )
            out = None
            for i in range(6):
                si = None
                for j, cij in nz_rows[i]:
                    term = cij * eu[j]
                    si = term if si is None else si + term
                if si is not None:
                    prod = si * ev[i]
                    out = prod if out is None else out + prod
            return out

        sub_basis = skfem.CellBasis(mesh, element, elements=elems, intorder=intorder)
        k_sub = elas.assemble(sub_basis)
        k_tot = k_sub if k_tot is None else k_tot + k_sub

    assert k_tot is not None
    k_tot = k_tot.tocsc()

    dofs_base = basis_full.get_dofs(lambda x: x[2] <= 0.00015).all()
    all_dofs = np.arange(basis_full.N)
    free_dofs = np.setdiff1d(all_dofs, dofs_base)
    k_ff = k_tot[free_dofs, :][:, free_dofs].tocsc()
    lu = splu(k_ff)

    x_tip_m = X_TIP * 1e-3
    z_c_m = Z_CENTER * 1e-3
    tip_facets = mesh.facets_satisfying(lambda x: x[0] >= x_tip_m - 0.00015)
    fbasis = skfem.FacetBasis(mesh, element, facets=tip_facets, intorder=intorder + 1)

    @skfem.Functional
    def fn_area(w):
        return 1.0 + 0.0 * w.x[0]

    @skfem.Functional
    def fn_iy(w):
        return (w.x[2] - z_c_m) ** 2

    @skfem.Functional
    def fn_iz(w):
        return (w.x[1]) ** 2

    area_tip = float(fn_area.assemble(fbasis))
    iy_tip = float(fn_iy.assemble(fbasis))
    iz_tip = float(fn_iz.assemble(fbasis))
    jp_tip = iy_tip + iz_tip

    def _solve_load(rhs_full: np.ndarray) -> np.ndarray:
        u = np.zeros(basis_full.N, dtype=np.float64)
        u[free_dofs] = lu.solve(rhs_full[free_dofs])
        return u

    def _measure_tip_kinematics(u: np.ndarray) -> tuple[float, float, float, float, float, float]:
        uh = fbasis.interpolate(u)

        @skfem.Functional
        def f_ux(w):
            return w.uh[0]

        @skfem.Functional
        def f_uy(w):
            return w.uh[1]

        @skfem.Functional
        def f_uz(w):
            return w.uh[2]

        @skfem.Functional
        def f_ry_num(w):
            return w.uh[0] * (w.x[2] - z_c_m)

        @skfem.Functional
        def f_rz_num(w):
            return -w.uh[0] * w.x[1]

        @skfem.Functional
        def f_rx_num(w):
            return w.uh[2] * w.x[1] - w.uh[1] * (w.x[2] - z_c_m)

        ux = float(f_ux.assemble(fbasis, uh=uh)) / area_tip
        uy = float(f_uy.assemble(fbasis, uh=uh)) / area_tip
        uz = float(f_uz.assemble(fbasis, uh=uh)) / area_tip
        rx = float(f_rx_num.assemble(fbasis, uh=uh)) / jp_tip
        ry = float(f_ry_num.assemble(fbasis, uh=uh)) / iy_tip
        rz = float(f_rz_num.assemble(fbasis, uh=uh)) / iz_tip
        return ux, uy, uz, rx, ry, rz

    @skfem.LinearForm
    def lf_my(v, w):
        return ((w.x[2] - z_c_m) / iy_tip) * v[0]

    @skfem.LinearForm
    def lf_mz(v, w):
        return (-w.x[1] / iz_tip) * v[0]

    @skfem.LinearForm
    def lf_mx(v, w):
        return (-(w.x[2] - z_c_m) / jp_tip) * v[1] + (w.x[1] / jp_tip) * v[2]

    @skfem.LinearForm
    def lf_fx(v, w):
        return (1.0 / area_tip) * v[0]

    @skfem.LinearForm
    def lf_fz(v, w):
        return (1.0 / area_tip) * v[2]

    _, _, uz_my, _, ry_my, _ = _measure_tip_kinematics(_solve_load(lf_my.assemble(fbasis)))
    _, _, _, _, _, rz_mz = _measure_tip_kinematics(_solve_load(lf_mz.assemble(fbasis)))
    _, _, _, rx_mx, _, _ = _measure_tip_kinematics(_solve_load(lf_mx.assemble(fbasis)))
    ux_fx, _, _, _, _, _ = _measure_tip_kinematics(_solve_load(lf_fx.assemble(fbasis)))
    _, _, uz_fz, _, ry_fz, _ = _measure_tip_kinematics(_solve_load(lf_fz.assemble(fbasis)))

    elapsed = time.perf_counter() - t0
    return JointFEAResult(
        joint_id=joint_id,
        order=order,
        n_tets=int(mesh.nelements),
        n_dofs=int(basis_full.N),
        solve_time_s=round(elapsed, 2),
        c_ry_my=ry_my,
        c_uz_my=uz_my,
        c_ry_fz=ry_fz,
        c_uz_fz=uz_fz,
        c_rz_mz=rz_mz,
        c_rx_mx=rx_mx,
        c_ux_fx=ux_fx,
    )


def _corner_2x2_compliance(r: JointFEAResult, l_shift_m: float) -> np.ndarray:
    """Transform tip [u_z, theta_y] compliance matrix rigidly back to corner node at x_tip - l_shift."""
    # Kinematics: u_tip = u_O - L * theta_y (positive ry tilts +X beam tip downward in -Z)
    # Wait: check sign from Maxwell-Betti reciprocity directly using measured c_uz_my!
    sgn = 1.0 if r.c_uz_my >= 0.0 else -1.0
    c_sym = 0.5 * (r.c_uz_my + r.c_ry_fz)
    c_tip = np.array([[r.c_uz_fz, c_sym], [c_sym, r.c_ry_my]], dtype=float)
    # Shift along -X by l_shift_m to cancel the lever-arm cross term
    t_mat = np.array([[1.0, -sgn * l_shift_m], [0.0, 1.0]], dtype=float)
    return t_mat @ c_tip @ t_mat.T


def run_calibration(
    *,
    order: int = 2,
    h_bulk_mm: float = 6.5,
    k_n_gpa_per_mm: float = 12.0,
    k_t_gpa_per_mm: float = 3.6,
) -> list[JointFEAResult]:
    """Run rigid controls + all 4 corner joint types and extract isolated joint stiffnesses."""
    order_ids = [
        "rigid_control",
        "gusset_welded_control",
        "cube_welded_control",
        "l_bracket",
        "blind_joint",
        "gusset_plate",
        "corner_cube",
    ]
    results: dict[str, JointFEAResult] = {}
    for jid in order_ids:
        res = _build_and_solve_specimen(
            jid,
            order=order,
            h_bulk_mm=h_bulk_mm,
            k_n_gpa_per_mm=k_n_gpa_per_mm,
            k_t_gpa_per_mm=k_t_gpa_per_mm,
        )
        results[jid] = res
        print(
            f"  [{jid:21s}] Tet{10 if order == 2 else 4}: {res.n_tets:6d} elems, {res.n_dofs:7d} DOFs "
            f"({res.solve_time_s:5.2f}s) | ry/My={res.c_ry_my:.4e}, rz/Mz={res.c_rz_mz:.4e}, "
            f"rx/Mx={res.c_rx_mx:.4e}, ux/Fx={res.c_ux_fx:.4e}, uz/Fz={res.c_uz_fz:.4e}",
            flush=True,
        )

    ref_bare = results["rigid_control"]
    ref_gusset = results["gusset_welded_control"]
    ref_cube = results["cube_welded_control"]
    l_shift_o = (X_TIP - 10.0) * 1e-3  # 0.150 m from tip back to joint plane X = 10 mm

    out: list[JointFEAResult] = [ref_bare, ref_gusset, ref_cube]
    for jid in ("l_bracket", "blind_joint", "gusset_plate", "corner_cube"):
        r = results[jid]
        # Subtract the geometry-matched monolithic welded control for hardware that reinforces
        # the corner panel zone (gusset_plate -> gusset_welded_control; corner_cube -> cube_welded_control)
        # so pure bolted-interface compliance is isolated without panel-zone sign inversion.
        if jid == "gusset_plate":
            ref = ref_gusset
        elif jid == "corner_cube":
            ref = ref_cube
        else:
            ref = ref_bare
        # For corner_cube, TWO bolted member-to-cube interfaces sit in series along the L-specimen
        # (Column A -> Cube and Cube -> Beam B), so per-member joint stiffness is 2x the L-specimen value.
        n_series = 2.0 if jid == "corner_cube" else 1.0

        c_o_joint = _corner_2x2_compliance(r, l_shift_o)
        c_o_ref = _corner_2x2_compliance(ref, l_shift_o)
        dc_o = c_o_joint - c_o_ref

        d_ry = max(float(dc_o[1, 1]), 1e-9) / n_series
        d_uz_sh = max(float(dc_o[0, 0]), 1e-12) / n_series
        d_rz = max(r.c_rz_mz - ref.c_rz_mz, 1e-9) / n_series
        d_rx = max(r.c_rx_mx - ref.c_rx_mx, 1e-9) / n_series
        d_ux = max(r.c_ux_fx - ref.c_ux_fx, 1e-12) / n_series

        r.k_r_inplane_nm_rad = 1.0 / d_ry
        r.k_r_outplane_nm_rad = 1.0 / d_rz
        r.k_r_torsion_nm_rad = 1.0 / d_rx
        r.k_t_axial_n_m = 1.0 / d_ux
        r.k_t_shear_n_m = 1.0 / d_uz_sh
        r.k_t_eff_n_m = math.sqrt(r.k_t_axial_n_m * r.k_t_shear_n_m)
        out.append(r)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="CAL-09 3D Tet10 FEA corner-joint calibration.")
    parser.add_argument("--order", type=int, default=2, choices=(1, 2), help="Tet order (1=Tet4, 2=Tet10)")
    parser.add_argument("--h-bulk", type=float, default=6.5, help="Bulk mesh size (mm)")
    parser.add_argument("--kn-gpa-mm", type=float, default=12.0, help="Normal areal contact stiffness (GPa/mm)")
    parser.add_argument("--kt-gpa-mm", type=float, default=3.6, help="Tangential areal contact stiffness (GPa/mm)")
    parser.add_argument("--json-out", type=str, default="", help="Optional JSON output path")
    args = parser.parse_args()

    print(
        f"Running CAL-09 3D Tet{10 if args.order == 2 else 4} corner-joint FEA "
        f"(h_bulk={args.h_bulk} mm, k_n={args.kn_gpa_mm} GPa/mm, k_t={args.kt_gpa_mm} GPa/mm)..."
    )
    results = run_calibration(
        order=args.order,
        h_bulk_mm=args.h_bulk,
        k_n_gpa_per_mm=args.kn_gpa_mm,
        k_t_gpa_per_mm=args.kt_gpa_mm,
    )

    print("\n=== CAL-09 Isolated 2020 Corner Joint Stiffnesses (3D Continuum FEA) ===")
    print(
        f"{'Joint':14s} | {'K_t (N/mm)':>11s} | {'K_t_ax':>10s} | {'K_t_sh':>10s} | "
        f"{'K_r_in (Nm/rad)':>15s} | {'K_r_out':>11s} | {'K_r_tor (Nm/rad)':>16s}"
    )
    print("-" * 102)
    for r in results:
        if r.joint_id.endswith("_control"):
            continue
        print(
            f"{r.joint_id:14s} | {r.k_t_eff_n_m * 1e-3:11.0f} | {r.k_t_axial_n_m * 1e-3:10.0f} | "
            f"{r.k_t_shear_n_m * 1e-3:10.0f} | {r.k_r_inplane_nm_rad:15.0f} | "
            f"{r.k_r_outplane_nm_rad:11.0f} | {r.k_r_torsion_nm_rad:16.0f}"
        )

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump([asdict(r) for r in results], fh, indent=2)
        print(f"\nSaved JSON report to {args.json_out}")


if __name__ == "__main__":
    main()
