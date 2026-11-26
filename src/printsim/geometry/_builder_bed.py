"""Bed mount compliance, Z-drive mechanisms, smooth rods, and CoreXY bed subframe builders extracted from ``builder.py``."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from printsim.schema import (
    Z_RAIL_PROFILES,
    MaterialSpec,
    PrinterSpec,
    ProfileSpec,
    get_material,
    get_profile,
    resolve_bed_mount,
    resolve_rail_quality,
    resolve_z_drive_mechanism,
)

if TYPE_CHECKING:
    from printsim.geometry.builder import (
        FastenerSpring,
        StructuralMember,
        StructuralNode,
    )


def _make_bed_virtual_material(mat: MaterialSpec) -> MaterialSpec:
    """Create a virtually massless, zero-cost stiffness copy for bed structural mounts."""
    return mat.model_copy(update={"density_kg_m3": 1e-3, "cost_per_kg_usd": 0.0})


def _apply_smooth_rod_guideway(
    member: StructuralMember,
    node_map: dict[str, StructuralNode],
    rod_profile: ProfileSpec,
    rod_material: MaterialSpec,
) -> tuple[float, float, float]:
    """Add an end-supported smooth steel guide rod's standalone EI/GJ & mass (no Steiner boost)."""
    p0 = node_map[member.start_id].position_m
    p1 = node_map[member.end_id].position_m
    length = max(float(np.linalg.norm(p1 - p0)), 1e-4)
    e_ext = max(member.material.youngs_modulus_pa, 1e6)
    g_ext = max(member.material.effective_shear_modulus_pa, 1e6)
    n_e = rod_material.youngs_modulus_pa / e_ext
    n_g = rod_material.effective_shear_modulus_pa / g_ext

    # End-supported smooth rods lose ~50% flexural sharing at rod-holder mounts vs bolted MGN rails
    end_clamp = 0.50 if rod_profile.id.startswith("steel_rod") else 1.0
    d_iy = end_clamp * n_e * rod_profile.iy_m4
    d_iz = end_clamp * n_e * rod_profile.iz_m4
    d_j = end_clamp * n_g * rod_profile.j_m4
    member.profile = member.profile.model_copy(
        update={
            "iy_m4": member.profile.iy_m4 + d_iy,
            "iz_m4": member.profile.iz_m4 + d_iz,
            "j_m4": member.profile.j_m4 + d_j,
        }
    )

    m_rod = rod_profile.area_m2 * length * rod_material.density_kg_m3
    a_ext = max(member.profile.area_m2, 1e-12)
    rho_delta = m_rod / max(a_ext * length, 1e-12)
    member.material = member.material.model_copy(
        update={"density_kg_m3": member.material.density_kg_m3 + rho_delta}
    )
    member.rail_mass_kg += m_rod
    return d_iy, d_iz, d_j


def _rebalance_virtual_bed_rail_mass(
    member: StructuralMember,
    node_map: dict[str, StructuralNode],
    old_rho: float,
    old_rail_mass: float,
) -> None:
    """Keep virtual bed struts near-massless while conserving added Z-guideway mass at column nodes."""
    if member.role != "bed":
        return
    added_kg = max(member.rail_mass_kg - old_rail_mass, 0.0)
    member.material = member.material.model_copy(update={"density_kg_m3": old_rho})
    if added_kg <= 0.0:
        return
    targets = [
        node_map[nid]
        for nid in (member.start_id, member.end_id)
        if nid in node_map and nid not in ("bed_surface", "bed_frame")
    ]
    if not targets:
        targets = [node_map[member.start_id]]
    per_node = added_kg / len(targets)
    for n_obj in targets:
        n_obj.lumped_mass_kg += per_node


def _apply_z_guideway(
    member: StructuralMember,
    node_map: dict[str, StructuralNode],
    z_rail_id: str,
    rail_material: MaterialSpec,
    normals: list[np.ndarray],
    bolt_pitch_m: float,
    l_eff_m: float,
) -> None:
    """Apply Z-axis linear rail (Steiner-coupled) or smooth guide rod (no Steiner) to a column."""
    from printsim.geometry import builder as _b

    if z_rail_id == "none":
        return
    old_rho = member.material.density_kg_m3
    old_rail_mass = member.rail_mass_kg
    z_prof = get_profile(z_rail_id)
    if z_rail_id.startswith("steel_rod"):
        _apply_smooth_rod_guideway(member, node_map, z_prof, rail_material)
    else:
        _b._apply_bolted_linear_rail(
            member,
            node_map,
            z_prof,
            rail_material,
            normals,
            beam_shape=member.profile.shape,
            bolt_pitch_m=bolt_pitch_m,
            l_eff_m=l_eff_m,
        )
    _rebalance_virtual_bed_rail_mass(member, node_map, old_rho, old_rail_mass)


def _compute_mount_offsets_m(spec: PrinterSpec) -> list[tuple[float, float]]:
    """Return 2D (dx, dy) mount point offsets (m) relative to bed center.

    Uses the 3/8-span (``0.375``) inset matching the 9x9 MITC4 bed plate FEM anchors
    so 3D viewport mount pins and 2D plate FEM supports align to the millimeter.
    """
    bx_m = float(spec.bed.size_mm[0]) * 1e-3
    by_m = float(spec.bed.size_mm[1]) * 1e-3
    inset = 0.375
    if spec.architecture == "delta":
        r_m = inset * min(bx_m, by_m)
        return [
            (
                r_m * math.cos(math.radians(deg)),
                r_m * math.sin(math.radians(deg)),
            )
            for deg in (90.0, 210.0, 330.0)
        ]
    if int(resolve_bed_mount(spec)["mount_points"]) == 3:
        return [
            (-inset * bx_m, -inset * by_m),
            (inset * bx_m, -inset * by_m),
            (0.0, inset * by_m),
        ]
    return [
        (-inset * bx_m, -inset * by_m),
        (inset * bx_m, -inset * by_m),
        (-inset * bx_m, inset * by_m),
        (inset * bx_m, inset * by_m),
    ]


def _attach_bed_mount_compliance(
    nodes: list[StructuralNode],
    fastener_springs: list[FastenerSpring],
    spec: PrinterSpec,
    bed_center_m: np.ndarray,
    anchor_nodes: tuple[tuple[str, str], ...] | None = None,
) -> None:
    """Create ``bed_frame`` / ``bed_surface`` nodes and 6-DOF ``bed_mount`` spring(s).

    When ``anchor_nodes`` is provided (e.g. Trident 3-point T-subframe mounts
    ``(("fl", "bed_t_fl"), ("fr", "bed_t_fr"), ("r", "bed_t_rear_mount"))``),
    attaches individual 6-DOF standoff springs directly from each physical subframe
    anchor node to ``bed_surface`` using its exact 3D ``(dx, dy, dz)`` lever-arm
    offset so bed roll/pitch moments transfer directly into the wide mount stance
    instead of twisting the central Y-spine in torsion. Otherwise attaches a single
    centroidal 6-DOF ``bed_mount_spring`` from ``bed_frame`` to ``bed_surface``.
    """
    from printsim.geometry import builder as _b

    cfg = resolve_bed_mount(spec)
    t_bed_m = float(spec.bed.plate_thickness_mm) * 1e-3
    h_standoff_m = float(cfg.get("standoff_height_mm", 14.0)) * 1e-3
    half_subframe_m = 0.010  # half of a 2020 bed subframe extrusion

    bed_kg = spec.bed.mass_g * 1e-3
    bed_frame_kg = 0.0
    bed_surface_kg = bed_kg

    bed_frame_pos = bed_center_m.copy()
    bed_frame_pos[2] -= t_bed_m + h_standoff_m + half_subframe_m
    dz_bed_mount_m = float(bed_frame_pos[2] - bed_center_m[2])
    nodes.append(
        _b.StructuralNode("bed_frame", bed_frame_pos, lumped_mass_kg=bed_frame_kg, role="bed")
    )
    nodes.append(
        _b.StructuralNode(
            "bed_surface", bed_center_m.copy(), lumped_mass_kg=bed_surface_kg, role="bed"
        )
    )

    n_pts = int(cfg["mount_points"])
    k_axial_n_m = float(cfg["k_axial_n_per_mm"]) * 1e3
    k_lat_n_m = float(cfg["k_lateral_n_per_mm"]) * 1e3
    if str(cfg.get("radial_dof_mode", "")) == "elastomer_shear":
        temp_soft = max(0.75, 1.0 - 0.004 * max(0.0, float(spec.bed.target_temp_c) - 70.0))
        k_axial_n_m *= temp_soft
        k_lat_n_m *= temp_soft

    nmap_local = {n.id: n for n in nodes}
    if anchor_nodes and len(anchor_nodes) == n_pts and all(nid in nmap_local for _, nid in anchor_nodes):
        # Local bearing-seat rotational stiffness per standoff (~10 mm OD M3/M4 spacer);
        # global roll (0.28125*k_z*bx^2), pitch (0.375*k_z*by^2), and yaw stiffnesses
        # emerge directly from the exact 3D eccentric lever-arm offsets (p_anchor - p_bed).
        k_seat_rot_nm_rad = max(k_axial_n_m * (0.005**2), 5.0)
        for tag, anchor_nid in anchor_nodes:
            off_vec = nmap_local[anchor_nid].position_m - bed_center_m
            fastener_springs.append(
                _b.FastenerSpring(
                    id=f"bed_mount_spring_{tag}",
                    bay="bed_mount",
                    shell_node_id=anchor_nid,
                    frame_node_id="bed_surface",
                    k_shear_n_m=k_lat_n_m,
                    k_normal_n_m=k_axial_n_m,
                    k_rot_nm_rad=k_seat_rot_nm_rad,
                    normal_vec=(0.0, 0.0, 1.0),
                    offset_m=(float(off_vec[0]), float(off_vec[1]), float(off_vec[2])),
                )
            )
        return

    pts = _compute_mount_offsets_m(spec)
    n_eff = max(len(pts), 1)
    x_bar = sum(x for x, _ in pts) / n_eff
    y_bar = sum(y for _, y in pts) / n_eff
    # Exact second moments of axial mount stiffness about the mount-pattern centroid (x_bar, y_bar):
    # - 3-pt triangle [(-0.375 bx, -0.375 by), (+0.375 bx, -0.375 by), (0, +0.375 by)] has
    #   centroid (0, -0.125 by) -> K_roll = 0.28125 k_z bx^2, K_pitch = 0.375 k_z by^2.
    # - 4-pt pattern [(+-0.375 bx, +-0.375 by)] has centroid (0, 0) -> K_roll = 0.5625 k_z bx^2,
    #   K_pitch = 0.5625 k_z by^2.
    k_roll = sum(k_axial_n_m * ((x - x_bar) ** 2) for x, _ in pts)
    k_pitch = sum(k_axial_n_m * ((y - y_bar) ** 2) for _, y in pts)
    # FastenerSpring uses scalar k_rot_nm_rad; geometric mean preserves 3-pt vs 4-pt inertia scaling.
    k_rot_nm_rad = max(math.sqrt(max(k_roll * k_pitch, 0.0)), 50.0)

    fastener_springs.append(
        _b.FastenerSpring(
            id="bed_mount_spring",
            bay="bed_mount",
            shell_node_id="bed_frame",
            frame_node_id="bed_surface",
            k_shear_n_m=n_pts * k_lat_n_m,
            k_normal_n_m=n_pts * k_axial_n_m,
            k_rot_nm_rad=k_rot_nm_rad,
            normal_vec=(0.0, 0.0, 1.0),
            offset_m=(0.0, 0.0, dz_bed_mount_m),
        )
    )


def _compute_z_carriage_spring_stiffness(
    spec: PrinterSpec,
    z_drive: str,
    k_z_drive_col_n_m: float,
) -> tuple[float, float, float]:
    """Return 6-DOF Z-carriage + drive spring stiffnesses ``(k_normal_z, k_shear_xy, k_rot_xyz)`` in SI."""
    if z_drive == "leadscrew_dual_belt_sync":
        # Floating TR8/SFU leadscrew nuts transmit axial Z thrust; the 4 corner smooth rods
        # + LMK10LUU bearings (z_rod_brg_*) carry lateral and pitch/roll guideway loads.
        return float(k_z_drive_col_n_m), 1.5e6, 40.0

    z_rail_id = spec.bed.z_rail_profile
    if z_rail_id == "none":
        return float(k_z_drive_col_n_m), 2.0e6, 120.0

    from printsim.fea.components import hertzian_rail_carriage_stiffness

    rail_qual = resolve_rail_quality(spec)
    car = hertzian_rail_carriage_stiffness(
        z_rail_id,
        preload=spec.gantry.rail_preload,
        block_type=spec.gantry.rail_block_type,
        rail_quality=spec.gantry.rail_quality,
        preload_fraction_override=spec.model_overrides.rail_preload.preload_fraction,
        stiffness_mult_override=rail_qual["stiffness_mult"],
        drag_mult_override=rail_qual["drag_mult"],
    )
    k_rad_hertz = 0.5 * (float(car["k_yy_n_m"]) + float(car["k_zz_n_m"]))
    # Vertical column bending in XZ/YZ loads the Z-carriage along its raceway length L1
    # (pitch/yaw k_ry, k_rz), whereas narrow-track roll k_rx governs column torsion around Z.
    k_rot_hertz = 0.5 * (float(car["k_ry_nm_rad"]) + float(car["k_rz_nm_rad"]))

    z_rail_cfg = Z_RAIL_PROFILES.get(z_rail_id, Z_RAIL_PROFILES["mgn12_rail"])
    rail_kind = str(z_rail_cfg.get("kind", "linear_rail"))
    if rail_kind == "linear_rail":
        # MGN carriage block in series with Z-carriage bracket & 4x M3 T-slot mounting compliance
        is_hgh15 = "hgh15" in z_rail_id
        is_mgn15 = "15" in z_rail_id and not is_hgh15
        is_mgn12 = "12" in z_rail_id
        if is_hgh15:
            k_house_shear = 9.0e7
            k_house_rot = 90000.0
        elif is_mgn15:
            k_house_shear = 6.5e7
            k_house_rot = 55000.0
        elif is_mgn12:
            k_house_shear = 4.5e7
            k_house_rot = 35000.0
        else:
            k_house_shear = 2.6e7
            k_house_rot = 22000.0
    else:
        diam_mm = float(z_rail_cfg.get("width_mm", 10.0))
        d_scale = max(diam_mm / 10.0, 0.7)
        k_house_shear = 1.1e7 * d_scale
        k_house_rot = 3800.0 * (d_scale**1.5)

    k_shear = (k_rad_hertz * k_house_shear) / max(k_rad_hertz + k_house_shear, 1.0)
    k_rot = (k_rot_hertz * k_house_rot) / max(k_rot_hertz + k_house_rot, 1.0)
    return float(k_z_drive_col_n_m), float(k_shear), float(k_rot)


def _compute_corexy_z_drive_stiffness(
    spec: PrinterSpec,
    z_gantry: float,
    z_bed: float,
    is_flying: bool,
    trident_backbone_active: bool,
) -> float:
    """Compute axial Z-drive column/screw/nut/thrust stiffness (N/m) for CoreXY architectures."""
    z_mech = resolve_z_drive_mechanism(spec)
    if str(z_mech["kind"]) == "belt":
        return float(z_mech["k_drive_n_per_um_per_corner"] or 0.18) * 1.0e6
    e_steel_z = 200.0e9
    l_screw_m = max(z_gantry if is_flying else z_bed, 0.05)
    k_shaft = e_steel_z * float(z_mech["root_area_m2"]) / l_screw_m
    k_nut = max(float(z_mech["k_nut_n_per_um"]) * 1.0e6, 1.0e5)
    k_thrust = max(float(z_mech["k_thrust_n_per_um"]) * 1.0e6, 1.0e5)
    if trident_backbone_active:
        k_thrust *= 2.4
    return 1.0 / (1.0 / k_shaft + 1.0 / k_nut + 1.0 / k_thrust)


def _setup_corexy_z_drive_nodes_and_bed(
    spec: PrinterSpec,
    nodes: list[StructuralNode],
    fastener_springs: list[FastenerSpring],
    z_drive: str,
    bed_cfg: dict[str, Any],
    bed_center: np.ndarray,
    w: float,
    d: float,
    h: float,
    z_gantry: float,
    z_subframe: float,
    z_motor_kg: float,
    k_z_drive_col_n_m: float,
) -> tuple[tuple[str, str], ...] | None:
    """Create CoreXY Z-drive carriage/column/rod nodes, 6-DOF drive springs, and bed mount springs."""
    from printsim.geometry import builder as _b

    y_mid = d * 0.5
    trident_bed_anchors: tuple[tuple[str, str], ...] | None = None
    k_z_car_normal, k_z_car_shear, k_z_car_rot = _compute_z_carriage_spring_stiffness(
        spec, z_drive, k_z_drive_col_n_m
    )

    def _add_z_drive_springs(tags: tuple[str, ...]) -> None:
        for tag in tags:
            fastener_springs.append(
                _b.FastenerSpring(
                    id=f"z_bed_drive_{tag}",
                    bay="bed_z_drive",
                    shell_node_id=f"z_col_bed_{tag}",
                    frame_node_id=f"z_bed_{tag}",
                    k_shear_n_m=k_z_car_shear,
                    k_normal_n_m=k_z_car_normal,
                    k_rot_nm_rad=k_z_car_rot,
                    normal_vec=(0.0, 0.0, 1.0),
                    offset_m=(0.0, 0.0, 0.0),
                )
            )

    if z_drive == "leadscrew_triple":
        bed_x_m = float(spec.bed.size_mm[0]) * 1e-3
        bed_y_m = float(spec.bed.size_mm[1]) * 1e-3
        x_fl = w * 0.5 - 0.375 * bed_x_m
        x_fr = w * 0.5 + 0.375 * bed_x_m
        y_f = d * 0.5 - 0.375 * bed_y_m
        y_r = d * 0.5 + 0.375 * bed_y_m
        nodes.extend(
            [
                # Stationary Z-rail/column attachment nodes on front columns & rear Z-post
                _b.StructuralNode("z_col_bed_fl", np.array([0.0, 0.0, z_subframe]), role="frame"),
                _b.StructuralNode("z_col_bed_fr", np.array([w, 0.0, z_subframe]), role="frame"),
                _b.StructuralNode("b_rear_mid", np.array([w * 0.5, d, 0.0]), lumped_mass_kg=z_motor_kg, role="frame"),
                _b.StructuralNode("g_rear_mid", np.array([w * 0.5, d, z_gantry]), role="frame"),
                _b.StructuralNode("z_col_bed_r", np.array([w * 0.5, d, z_subframe]), role="frame"),
                # Moving Z-carriage nodes coupled vertically on leadscrew/nut + MGN block compliance
                _b.StructuralNode("z_bed_fl", np.array([0.0, 0.0, z_subframe]), role="bed"),
                _b.StructuralNode("z_bed_fr", np.array([w, 0.0, z_subframe]), role="bed"),
                _b.StructuralNode("z_bed_r", np.array([w * 0.5, d, z_subframe]), role="bed"),
                _b.StructuralNode("bed_t_fl", np.array([x_fl, y_f, z_subframe]), role="bed"),
                _b.StructuralNode("bed_t_fr", np.array([x_fr, y_f, z_subframe]), role="bed"),
                _b.StructuralNode("bed_t_front", np.array([w * 0.5, y_f, z_subframe]), role="bed"),
                _b.StructuralNode("bed_t_rear_mount", np.array([w * 0.5, y_r, z_subframe]), role="bed"),
            ]
        )
        _add_z_drive_springs(("fl", "fr", "r"))
        if int(bed_cfg["mount_points"]) == 3:
            trident_bed_anchors = (
                ("fl", "bed_t_fl"),
                ("fr", "bed_t_fr"),
                ("r", "bed_t_rear_mount"),
            )
    elif z_drive == "leadscrew_dual":
        # Anchor top of dual-Z side guides to top perimeter rails (t_left_mid / t_right_mid)
        # so the moving Y carriages y_car_l/r never collide or weld to Z guide nodes at mid-Y.
        nodes.extend(
            [
                _b.StructuralNode("b_left_mid", np.array([0.0, y_mid, 0.0]), lumped_mass_kg=z_motor_kg, role="frame"),
                _b.StructuralNode("b_right_mid", np.array([w, y_mid, 0.0]), lumped_mass_kg=z_motor_kg, role="frame"),
                _b.StructuralNode("t_left_mid", np.array([0.0, y_mid, h]), role="frame"),
                _b.StructuralNode("t_right_mid", np.array([w, y_mid, h]), role="frame"),
                _b.StructuralNode("z_col_bed_l", np.array([0.0, y_mid, z_subframe]), role="frame"),
                _b.StructuralNode("z_col_bed_r", np.array([w, y_mid, z_subframe]), role="frame"),
                _b.StructuralNode("z_bed_l", np.array([0.0, y_mid, z_subframe]), role="bed"),
                _b.StructuralNode("z_bed_r", np.array([w, y_mid, z_subframe]), role="bed"),
            ]
        )
        _add_z_drive_springs(("l", "r"))
    elif z_drive == "leadscrew_dual_belt_sync":
        # Real VzBot 330/235 Z-architecture: 1 NEMA17 Z stepper at rear-center driving 2 belt-synced
        # mid-side TR8 leadscrews + 4 vertical hardened smooth rods (y = 0.25d, 0.75d on left & right)
        # anchored between base and top perimeter rails with LMK10LUU flanged linear bearings.
        y_rod_f = d * 0.25
        y_rod_r = d * 0.75
        nodes.extend(
            [
                _b.StructuralNode("b_rear_mid", np.array([w * 0.5, d, 0.0]), lumped_mass_kg=z_motor_kg, role="frame"),
                _b.StructuralNode("b_left_mid", np.array([0.0, y_mid, 0.0]), role="frame"),
                _b.StructuralNode("b_right_mid", np.array([w, y_mid, 0.0]), role="frame"),
                _b.StructuralNode("t_left_mid", np.array([0.0, y_mid, h]), role="frame"),
                _b.StructuralNode("t_right_mid", np.array([w, y_mid, h]), role="frame"),
                _b.StructuralNode("z_col_bed_l", np.array([0.0, y_mid, z_subframe]), role="frame"),
                _b.StructuralNode("z_col_bed_r", np.array([w, y_mid, z_subframe]), role="frame"),
                _b.StructuralNode("z_bed_l", np.array([0.0, y_mid, z_subframe]), role="bed"),
                _b.StructuralNode("z_bed_r", np.array([w, y_mid, z_subframe]), role="bed"),
                # 4 smooth-rod base anchors, top anchors, rod-at-bed nodes, and LMK bearing housings
                _b.StructuralNode("b_rod_lf", np.array([0.0, y_rod_f, 0.0]), role="frame"),
                _b.StructuralNode("b_rod_lr", np.array([0.0, y_rod_r, 0.0]), role="frame"),
                _b.StructuralNode("b_rod_rf", np.array([w, y_rod_f, 0.0]), role="frame"),
                _b.StructuralNode("b_rod_rr", np.array([w, y_rod_r, 0.0]), role="frame"),
                _b.StructuralNode("t_rod_lf", np.array([0.0, y_rod_f, h]), role="frame"),
                _b.StructuralNode("t_rod_lr", np.array([0.0, y_rod_r, h]), role="frame"),
                _b.StructuralNode("t_rod_rf", np.array([w, y_rod_f, h]), role="frame"),
                _b.StructuralNode("t_rod_rr", np.array([w, y_rod_r, h]), role="frame"),
                _b.StructuralNode("z_rod_lf", np.array([0.0, y_rod_f, z_subframe]), role="bed"),
                _b.StructuralNode("z_rod_lr", np.array([0.0, y_rod_r, z_subframe]), role="bed"),
                _b.StructuralNode("z_rod_rf", np.array([w, y_rod_f, z_subframe]), role="bed"),
                _b.StructuralNode("z_rod_rr", np.array([w, y_rod_r, z_subframe]), role="bed"),
                _b.StructuralNode("z_brg_lf", np.array([0.0, y_rod_f, z_subframe]), role="bed"),
                _b.StructuralNode("z_brg_lr", np.array([0.0, y_rod_r, z_subframe]), role="bed"),
                _b.StructuralNode("z_brg_rf", np.array([w, y_rod_f, z_subframe]), role="bed"),
                _b.StructuralNode("z_brg_rr", np.array([w, y_rod_r, z_subframe]), role="bed"),
            ]
        )
        _add_z_drive_springs(("l", "r"))
        for tag in ("lf", "lr", "rf", "rr"):
            fastener_springs.append(
                _b.FastenerSpring(
                    id=f"z_rod_brg_{tag}",
                    bay="z_guide",
                    shell_node_id=f"z_rod_{tag}",
                    frame_node_id=f"z_brg_{tag}",
                    k_shear_n_m=1.2e7,
                    k_normal_n_m=1.0e2,
                    k_rot_nm_rad=5000.0,
                    normal_vec=(0.0, 0.0, 1.0),
                    offset_m=(0.0, 0.0, 0.0),
                )
            )
    elif z_drive == "leadscrew_single":
        nodes.extend(
            [
                _b.StructuralNode("b_rear_mid", np.array([w * 0.5, d, 0.0]), lumped_mass_kg=z_motor_kg, role="frame"),
                _b.StructuralNode("g_rear_mid", np.array([w * 0.5, d, z_gantry]), role="frame"),
                _b.StructuralNode("z_col_bed_r", np.array([w * 0.5, d, z_subframe]), role="frame"),
                _b.StructuralNode("z_bed_r", np.array([w * 0.5, d, z_subframe]), role="bed"),
            ]
        )
        _add_z_drive_springs(("r",))

    _attach_bed_mount_compliance(
        nodes,
        fastener_springs,
        spec,
        bed_center,
        anchor_nodes=trident_bed_anchors,
    )


    return trident_bed_anchors


def _add_corexy_bed_subframe_members(
    spec: PrinterSpec,
    nodes: list[StructuralNode],
    members: list[StructuralMember],
    z_drive: str,
    is_flying: bool,
    fp: str,
    fm: str,
    h: float,
    bz: float,
    trident_bed_anchors: tuple[tuple[str, str], ...] | None,
) -> None:
    """Add CoreXY bed structural arms, T-subframe, dual-Z columns, belt-synced smooth rods, or cantilever arm."""
    from printsim.geometry import builder as _b

    z_mech_id = spec.bed.effective_z_leadscrew_type
    z_mech = resolve_z_drive_mechanism(spec)
    # Structural bed mounting links into intermediate bed_frame (virtually massless; exact mass on bed nodes)
    bed_vmat = _make_bed_virtual_material(get_material(spec.bed.plate_material))
    t_mm = max(float(spec.bed.plate_thickness_mm), 2.0)
    t_ratio = t_mm / 8.0
    iy_triple_arm = 1.8e-8 + 7.2e-8 * (t_ratio**3)
    iz_triple_arm = 1.5e-7
    j_triple_arm = 6.0e-8

    if is_flying:
        bed_arm_prof = ProfileSpec.model_construct(
            id="bed_fixed_arm_profile",
            name=f"Fixed Bed Base Mount ({t_mm:.1f}mm {spec.bed.plate_material})",
            shape="solid_bar",
            width_m=0.020,
            height_m=max(0.014, 0.012 + t_mm * 1e-3),
            area_m2=8.0e-4,
            iy_m4=8.0e-8 + 2.4e-7 * (t_ratio**3),
            iz_m4=1.6e-7,
            j_m4=6.0e-8,
        )
        for tag in ("fl", "fr", "rl", "rr"):
            members.append(
                _b.StructuralMember(
                    id=f"bed_arm_{tag}",
                    start_id=f"b_bed_{tag}",
                    end_id="bed_frame",
                    profile=bed_arm_prof,
                    material=bed_vmat,
                    role="bed",
                )
            )
    else:
        frame_prof = get_profile(fp)
        z_rail_vmat = _make_bed_virtual_material(get_material(fm))
        z_post_prof = ProfileSpec.model_construct(
            id="bed_z_rail_profile",
            name=f"Z-Axis Guideway Backing ({fp})",
            shape=frame_prof.shape,
            width_m=frame_prof.width_m,
            height_m=frame_prof.height_m,
            area_m2=frame_prof.area_m2,
            iy_m4=frame_prof.iy_m4,
            iz_m4=frame_prof.iz_m4,
            j_m4=frame_prof.j_m4,
        )
        if z_drive in ("leadscrew_triple", "leadscrew_single"):
            members.append(
                _b.StructuralMember(
                    id="bed_rear_z_bot",
                    start_id="b_rear_mid",
                    end_id="z_col_bed_r",
                    profile=z_post_prof,
                    material=z_rail_vmat,
                    role="bed",
                )
            )
            members.append(
                _b.StructuralMember(
                    id="bed_rear_z_top",
                    start_id="z_col_bed_r",
                    end_id="g_rear_mid",
                    profile=z_post_prof,
                    material=z_rail_vmat,
                    role="bed",
                )
            )

        if z_drive == "leadscrew_triple":
            ext_2020 = get_profile("extrusion_2020")
            # Bare 2020 Z-arm + Z-joint gusset + cast-Al bed edge stiffness (axial leadscrew/nut
            # compliance lives cleanly in the 6-DOF z_bed_drive_* CBUSH springs, not arm Iy dilution).
            bed_arm_prof = ProfileSpec.model_construct(
                id="bed_trident_arm_profile",
                name=f"3-Point Z Bed Arm + Carriage ({t_mm:.1f}mm {spec.bed.plate_material})",
                shape="extrusion",
                width_m=ext_2020.width_m,
                height_m=ext_2020.height_m,
                area_m2=1.4e-3,
                iy_m4=25.0 * iy_triple_arm,
                iz_m4=5.0e-7,
                j_m4=6.25e-7,
            )
            for tag, start_nid, end_nid in (
                ("fl", "z_bed_fl", "bed_t_fl"),
                ("fr", "z_bed_fr", "bed_t_fr"),
                ("r", "bed_t_rear_mount", "z_bed_r"),
            ):
                members.append(
                    _b.StructuralMember(
                        id=f"bed_arm_{tag}",
                        start_id=start_nid,
                        end_id=end_nid,
                        profile=bed_arm_prof,
                        material=bed_vmat,
                        role="bed",
                    )
                )

            # 2020 T-subframe composite with bolted cast-Al bed plate diaphragm:
            # - Front crossbar (X): out-of-plane composite Iy + 2020 torsion J around X.
            # - Central spine (Y): in-plane cast-Al plate diaphragm Iz + composite Iy + 2020-scale torsion J.
            bed_crossbar_prof = ProfileSpec.model_construct(
                id="bed_trident_subframe_crossbar_profile",
                name=f"2020 T-Subframe Crossbar + Plate ({t_mm:.1f}mm {spec.bed.plate_material})",
                shape="extrusion",
                width_m=ext_2020.width_m,
                height_m=ext_2020.height_m,
                area_m2=1.4e-3,
                iy_m4=ext_2020.iy_m4 + 1.9932e-6 * (t_ratio**3),
                iz_m4=ext_2020.iz_m4 + 2.5e-7 * t_ratio,
                j_m4=ext_2020.j_m4 + 1.0e-6 * (t_ratio**3),
            )
            # For 3-pt mounts, individual standoff springs attach directly at bed_t_fl/fr/rear_mount
            # so roll loads bypass spine torsion. For 4-pt mounts (centroidal spring at bed_frame),
            # include the 4-corner carrier roll transfer into the front crossbar.
            spine_j_m4 = (
                ext_2020.j_m4 + 1.489e-7 * (t_ratio**3)
                if trident_bed_anchors is not None
                else ext_2020.j_m4 + 1.9932e-6 * (t_ratio**3)
            )
            bed_spine_prof = ProfileSpec.model_construct(
                id="bed_trident_subframe_spine_profile",
                name=f"2020 T-Subframe Spine + Plate ({t_mm:.1f}mm {spec.bed.plate_material})",
                shape="extrusion",
                width_m=ext_2020.width_m,
                height_m=ext_2020.height_m,
                area_m2=1.4e-3,
                iy_m4=ext_2020.iy_m4 + 6.0e-6 * (t_ratio**3),
                iz_m4=ext_2020.iz_m4 + 4.0e-6 * t_ratio,
                j_m4=spine_j_m4,
            )
            for mid, n0, n1, prof in (
                ("bed_subframe_fl", "bed_t_fl", "bed_t_front", bed_crossbar_prof),
                ("bed_subframe_fr", "bed_t_front", "bed_t_fr", bed_crossbar_prof),
                ("bed_subframe_spine_f", "bed_t_front", "bed_frame", bed_spine_prof),
                ("bed_subframe_spine_r", "bed_frame", "bed_t_rear_mount", bed_spine_prof),
            ):
                members.append(
                    _b.StructuralMember(
                        id=mid,
                        start_id=n0,
                        end_id=n1,
                        profile=prof,
                        material=bed_vmat,
                        role="bed",
                    )
                )
        elif z_drive == "leadscrew_dual":
            z_dual_rail_prof = ProfileSpec.model_construct(
                id="bed_dual_z_rail_profile",
                name="Dual-Z Side Linear Guide Column",
                shape=frame_prof.shape,
                width_m=frame_prof.width_m,
                height_m=frame_prof.height_m,
                area_m2=frame_prof.area_m2,
                iy_m4=0.22 * frame_prof.iy_m4,
                iz_m4=0.22 * frame_prof.iz_m4,
                j_m4=0.22 * frame_prof.j_m4,
            )
            for side, b_nid, z_nid, t_nid in (
                ("left", "b_left_mid", "z_col_bed_l", "t_left_mid"),
                ("right", "b_right_mid", "z_col_bed_r", "t_right_mid"),
            ):
                members.append(
                    _b.StructuralMember(
                        id=f"bed_{side}_z_bot",
                        start_id=b_nid,
                        end_id=z_nid,
                        profile=z_dual_rail_prof,
                        material=z_rail_vmat,
                        role="bed",
                    )
                )
                members.append(
                    _b.StructuralMember(
                        id=f"bed_{side}_z_top",
                        start_id=z_nid,
                        end_id=t_nid,
                        profile=z_dual_rail_prof,
                        material=z_rail_vmat,
                        role="bed",
                    )
                )
            bed_dual_prof = ProfileSpec.model_construct(
                id="bed_dual_arm_profile",
                name=f"Dual-Z Bed Arm + Plate ({t_mm:.1f}mm {spec.bed.plate_material})",
                shape="solid_bar",
                width_m=0.020,
                height_m=max(0.012, 0.012 + t_mm * 1e-3),
                area_m2=5.0e-4,
                iy_m4=3.5 * iy_triple_arm,
                iz_m4=0.45 * iz_triple_arm,
                j_m4=0.35 * j_triple_arm,
            )
            for tag, z_nid in (("l", "z_bed_l"), ("r", "z_bed_r")):
                members.append(
                    _b.StructuralMember(
                        id=f"bed_arm_{tag}",
                        start_id=z_nid,
                        end_id="bed_frame",
                        profile=bed_dual_prof,
                        material=bed_vmat,
                        role="bed",
                    )
                )
        elif z_drive == "leadscrew_dual_belt_sync":
            nmap_pre = {n.id: n for n in nodes}
            # 4x hardened steel smooth rods (default Ø10 mm) spanning base-to-top rails.
            # Real VzBot 330/235 rods span the Z-travel bay (~bz + 100 mm) minus 55 mm double-length
            # LMK10LUU flanged linear bearings and CNC aluminum top/bottom rod holders.
            rod_prof_id = (
                spec.bed.z_rail_profile
                if spec.bed.z_rail_profile.startswith("steel_rod")
                else "steel_rod_10mm"
            )
            rod_base_prof = get_profile(rod_prof_id)
            rod_mat = get_material(spec.gantry.rail_material)
            l_clear_rod = min(h - 0.10, max(bz + 0.015, 0.24))
            sleeve_ei_mult = min(5.5, max(1.0, (h / max(l_clear_rod, 0.12)) ** 3))
            rod_eff_prof = rod_base_prof.model_copy(
                update={
                    "iy_m4": rod_base_prof.iy_m4 * sleeve_ei_mult,
                    "iz_m4": rod_base_prof.iz_m4 * sleeve_ei_mult,
                    "j_m4": rod_base_prof.j_m4 * sleeve_ei_mult,
                }
            )
            # Bottom Z-rod cross-extrusions + CNC rod-holder clamp torsional/lateral backing
            members.append(
                _b.StructuralMember(
                    id="base_z_cross_f",
                    start_id="b_rod_lf",
                    end_id="b_rod_rf",
                    profile=frame_prof,
                    material=get_material(fm),
                    role="bed",
                )
            )
            members.append(
                _b.StructuralMember(
                    id="base_z_cross_r",
                    start_id="b_rod_lr",
                    end_id="b_rod_rr",
                    profile=frame_prof,
                    material=get_material(fm),
                    role="bed",
                )
            )
            for m_side in members:
                if m_side.id.startswith(("base_left_", "base_right_")):
                    m_side.profile = m_side.profile.model_copy(
                        update={
                            "iz_m4": m_side.profile.iz_m4 * 4.0,
                            "j_m4": m_side.profile.j_m4 * 2.5,
                        }
                    )
                elif m_side.id.startswith(("top_left_", "top_right_")):
                    m_side.profile = m_side.profile.model_copy(
                        update={
                            "iz_m4": m_side.profile.iz_m4 * 2.5,
                            "j_m4": m_side.profile.j_m4 * 2.5,
                        }
                    )
            for tag in ("lf", "lr", "rf", "rr"):
                b_nid = f"b_rod_{tag}"
                z_nid = f"z_rod_{tag}"
                t_nid = f"t_rod_{tag}"
                l_bot = float(np.linalg.norm(nmap_pre[z_nid].position_m - nmap_pre[b_nid].position_m))
                l_top = float(np.linalg.norm(nmap_pre[t_nid].position_m - nmap_pre[z_nid].position_m))
                members.append(
                    _b.StructuralMember(
                        id=f"bed_rod_{tag}_bot",
                        start_id=b_nid,
                        end_id=z_nid,
                        profile=rod_eff_prof,
                        material=rod_mat,
                        role="bed",
                        rail_mass_kg=rod_eff_prof.area_m2 * l_bot * rod_mat.density_kg_m3,
                    )
                )
                members.append(
                    _b.StructuralMember(
                        id=f"bed_rod_{tag}_top",
                        start_id=z_nid,
                        end_id=t_nid,
                        profile=rod_eff_prof,
                        material=rod_mat,
                        role="bed",
                        rail_mass_kg=rod_eff_prof.area_m2 * l_top * rod_mat.density_kg_m3,
                    )
                )
            # 2x mid-side belt-synced Z leadscrew columns (axial Z compliance lives in z_bed_drive_l/r)
            ls_diam_m = float(z_mech.get("diameter_mm", 8.0)) * 1e-3
            ls_r_m = 0.5 * ls_diam_m
            ls_i_m4 = 0.25 * math.pi * (ls_r_m**4)
            ls_j_m4 = 0.5 * math.pi * (ls_r_m**4)
            ls_vmat = _make_bed_virtual_material(rod_mat)
            ls_bot_prof = ProfileSpec.model_construct(
                id="bed_sync_leadscrew_bot_profile",
                name=f"Belt-Synced Z Leadscrew Lower ({z_mech_id})",
                shape="rod",
                width_m=ls_diam_m,
                height_m=ls_diam_m,
                area_m2=5.0e-4,  # stiff axial transfer to z_col_bed_l/r; shaft/nut/thrust live in z_bed_drive_*
                iy_m4=ls_i_m4 * sleeve_ei_mult,
                iz_m4=ls_i_m4 * sleeve_ei_mult,
                j_m4=ls_j_m4 * sleeve_ei_mult,
            )
            ls_top_prof = ls_bot_prof.model_copy(
                update={
                    "id": "bed_sync_leadscrew_top_profile",
                    "name": f"Belt-Synced Z Leadscrew Upper ({z_mech_id})",
                    "area_m2": 6.0e-7,  # floating upper 608 pillow block (no axial bind)
                }
            )
            for side, b_nid, z_nid, t_nid in (
                ("left", "b_left_mid", "z_col_bed_l", "t_left_mid"),
                ("right", "b_right_mid", "z_col_bed_r", "t_right_mid"),
            ):
                members.append(
                    _b.StructuralMember(
                        id=f"bed_{side}_z_bot",
                        start_id=b_nid,
                        end_id=z_nid,
                        profile=ls_bot_prof,
                        material=ls_vmat,
                        role="bed",
                    )
                )
                members.append(
                    _b.StructuralMember(
                        id=f"bed_{side}_z_top",
                        start_id=z_nid,
                        end_id=t_nid,
                        profile=ls_top_prof,
                        material=ls_vmat,
                        role="bed",
                    )
                )
            # 2x mid-side Z-lift bed arms + 4x rigid 2020 bed-carrier arms into LMK10LUU bearings
            bed_sync_arm_prof = ProfileSpec.model_construct(
                id="bed_sync_dual_arm_profile",
                name=f"Belt-Synced Dual-Z Bed Subframe ({t_mm:.1f}mm {spec.bed.plate_material})",
                shape="solid_bar",
                width_m=0.020,
                height_m=max(0.014, 0.014 + t_mm * 1e-3),
                area_m2=6.0e-4,
                iy_m4=5.0 * iy_triple_arm,
                iz_m4=0.90 * iz_triple_arm,
                j_m4=0.75 * j_triple_arm,
            )
            for tag, z_nid in (("l", "z_bed_l"), ("r", "z_bed_r")):
                members.append(
                    _b.StructuralMember(
                        id=f"bed_arm_{tag}",
                        start_id=z_nid,
                        end_id="bed_frame",
                        profile=bed_sync_arm_prof,
                        material=bed_vmat,
                        role="bed",
                    )
                )
            bed_carrier_prof = ProfileSpec.model_construct(
                id="bed_rod_carrier_arm_profile",
                name="VzBot 2020 Bed Carrier Subframe Arm",
                shape="solid_bar",
                width_m=0.020,
                height_m=0.020,
                area_m2=6.0e-4,
                iy_m4=2.2e-7,
                iz_m4=2.2e-7,
                j_m4=8.0e-8,
            )
            for tag in ("lf", "lr", "rf", "rr"):
                members.append(
                    _b.StructuralMember(
                        id=f"bed_carrier_{tag}",
                        start_id=f"z_brg_{tag}",
                        end_id="bed_frame",
                        profile=bed_carrier_prof,
                        material=bed_vmat,
                        role="bed",
                    )
                )
        elif z_drive == "leadscrew_single":
            bed_cantilever_prof = ProfileSpec.model_construct(
                id="bed_cantilever_arm_profile",
                name=f"Single-Z Cantilever Bed Arm ({t_mm:.1f}mm {spec.bed.plate_material})",
                shape="solid_bar",
                width_m=0.020,
                height_m=max(0.012, 0.010 + t_mm * 1e-3),
                area_m2=4.0e-4,
                iy_m4=0.55 * iy_triple_arm,
                iz_m4=0.45 * iz_triple_arm,
                j_m4=0.22 * j_triple_arm,
            )
            members.append(
                _b.StructuralMember(
                    id="bed_arm_cantilever",
                    start_id="z_bed_r",
                    end_id="bed_frame",
                    profile=bed_cantilever_prof,
                    material=bed_vmat,
                    role="bed",
                )
            )


def _apply_corexy_z_guideways(
    spec: PrinterSpec,
    member_by_id: dict[str, StructuralMember],
    nmap: dict[str, StructuralNode],
    z_drive: str,
    is_flying: bool,
    fp: str,
    rail_mat: MaterialSpec,
    bolt_pitch_m: float,
    h: float,
    z_gantry: float,
) -> None:
    """Apply physical Z-axis linear rails or smooth rods along CoreXY vertical Z guideways."""
    from printsim.geometry import builder as _b

    frame_prof = get_profile(fp)
    z_rail_id = spec.bed.z_rail_profile
    if z_rail_id != "none":
        if is_flying:
            for tag, nx_sign in (("fl", 1.0), ("fr", -1.0), ("rl", 1.0), ("rr", -1.0)):
                n_z = [np.array([nx_sign, 0.0, 0.0])]
                for mid in (f"col_{tag}_low", f"col_{tag}_top"):
                    if mid in member_by_id:
                        _apply_z_guideway(
                            member_by_id[mid], nmap, z_rail_id, rail_mat, n_z, bolt_pitch_m, h
                        )
        elif z_drive == "leadscrew_triple":
            n_front_y = [np.array([0.0, 1.0, 0.0])]
            for tag in ("fl", "fr"):
                for seg in ("bot", "mid", "top"):
                    mid = f"col_{tag}_{seg}"
                    if mid in member_by_id:
                        _apply_z_guideway(
                            member_by_id[mid], nmap, z_rail_id, rail_mat, n_front_y, bolt_pitch_m, h
                        )
            n_rear = [np.array([0.0, -1.0, 0.0])]
            for mid in ("bed_rear_z_bot", "bed_rear_z_top"):
                if mid in member_by_id:
                    _apply_z_guideway(
                        member_by_id[mid], nmap, z_rail_id, rail_mat, n_rear, bolt_pitch_m, z_gantry
                    )
        elif z_drive == "leadscrew_dual":
            z_rail_prof_d = get_profile(z_rail_id)
            for side in ("left", "right"):
                nx = 1.0 if side == "left" else -1.0
                for mid in (f"bed_{side}_z_bot", f"bed_{side}_z_top"):
                    if mid in member_by_id:
                        m_obj = member_by_id[mid]
                        old_rho = m_obj.material.density_kg_m3
                        old_rail_mass = m_obj.rail_mass_kg
                        if z_rail_prof_d.shape == "rail":
                            # Linear-rail dual-Z uses twin side guideway columns with bolted steel rail
                            # (using frame_prof wall area during Newmark composite EI, then keeping
                            # leadscrew-equivalent axial area_m2 since rail carriages slide freely along Z).
                            m_obj.profile = ProfileSpec.model_construct(
                                id="bed_dual_z_rail_profile",
                                name="Dual-Z Side Linear Rail + Guide Column",
                                shape=frame_prof.shape,
                                width_m=frame_prof.width_m,
                                height_m=frame_prof.height_m,
                                area_m2=frame_prof.area_m2,
                                iy_m4=6.0 * frame_prof.iy_m4,
                                iz_m4=6.0 * frame_prof.iz_m4,
                                j_m4=0.22 * frame_prof.j_m4,
                            )
                            _b._apply_bolted_linear_rail(
                                m_obj,
                                nmap,
                                z_rail_prof_d,
                                rail_mat,
                                [np.array([nx, 0.0, 0.0])],
                                beam_shape=frame_prof.shape,
                                bolt_pitch_m=float(spec.gantry.rail_bolt_pitch_mm) * 1e-3,
                            )
                            # Keep bending EI from composite column+rail and full extrusion axial area_m2
                            # since z_bed_drive_l/r CBUSH springs supply leadscrew/ballscrew/belt Z compliance.
                            m_obj.profile = m_obj.profile.model_copy(
                                update={"area_m2": frame_prof.area_m2}
                            )
                            _rebalance_virtual_bed_rail_mass(m_obj, nmap, old_rho, old_rail_mass)
                            continue
                        else:
                            _apply_smooth_rod_guideway(m_obj, nmap, z_rail_prof_d, rail_mat)
                        _rebalance_virtual_bed_rail_mass(m_obj, nmap, old_rho, old_rail_mass)
        elif z_drive == "leadscrew_single":
            n_rear = [np.array([0.0, -1.0, 0.0])]
            for mid in ("bed_rear_z_bot", "bed_rear_z_top"):
                if mid in member_by_id:
                    _apply_z_guideway(
                        member_by_id[mid], nmap, z_rail_id, rail_mat, n_rear, bolt_pitch_m, z_gantry
                    )
