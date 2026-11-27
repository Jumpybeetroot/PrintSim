"""Cartesian bedslinger structural assembly builder extracted from ``builder.py``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from printsim.schema import (
    PrinterSpec,
    ProfileSpec,
    get_material,
    get_profile,
    resolve_belt,
    resolve_gantry_bracket,
)

if TYPE_CHECKING:
    from printsim.geometry.builder import AssemblyModel, FastenerSpring, StructuralMember


def _build_cartesian_bedslinger(
    spec: PrinterSpec, toolhead_rel_pos: tuple[float, float, float]
) -> AssemblyModel:
    """Build a 3D FEA structural assembly for a Cartesian H-frame bedslinger printer."""
    from printsim.geometry import builder as _b

    up_prof = get_profile(spec.frame.effective_upright_profile)
    fr_prof = get_profile(spec.frame.profile)
    w = max(spec.frame.width_mm * 1e-3 - up_prof.width_m, 0.08)
    d = max(spec.frame.depth_mm * 1e-3 - fr_prof.width_m, 0.08)
    h = max(spec.frame.height_mm * 1e-3 - fr_prof.height_m, 0.08)
    bx, by, bz = np.array(spec.kinematics.build_volume_mm) * 1e-3
    u, v_rel, w_rel = np.clip(toolhead_rel_pos, 0.05, 0.95)

    y_portal = d * 0.55
    orig_x = (w - bx) * 0.5
    orig_y = (d - by) * 0.5
    orig_z = 0.075
    z_gantry = min(h - 0.06, orig_z + w_rel * bz)
    x_tool = orig_x + u * bx
    # Moving Y-bed slides along base Y rails to position build coord v_rel under stationary nozzle at y_portal
    y_bed = float(np.clip(y_portal + (0.5 - v_rel) * by, 0.05, max(d - 0.05, 0.06)))

    xy_motor_kg = spec.kinematics.xy_stepper.motor_mass_g * 1e-3
    z_motor_kg = spec.kinematics.z_stepper.motor_mass_g * 1e-3
    bkt_resolved = resolve_gantry_bracket(spec)
    bkt_delta_kg = float(bkt_resolved["carriage_mass_delta_g"]) * 1e-3
    _, y_block_delta_g = _b.y_carriage_block_mass_delta_g(spec)
    y_block_delta_kg = y_block_delta_g * 1e-3
    y_carriage_total_kg = max(spec.gantry.carriage_mass_g * 1e-3 + y_block_delta_kg, 0.04)
    carriage_side_kg = 0.5 * y_carriage_total_kg
    toolhead_kg = _b._effective_toolhead_mass_kg(spec)

    nodes = [
        # H-frame base feet
        _b.StructuralNode("b_fl", np.array([0.0, 0.0, 0.0]), fixed=True, role="foot"),
        _b.StructuralNode("b_fr", np.array([w, 0.0, 0.0]), fixed=True, role="foot"),
        _b.StructuralNode("b_rl", np.array([0.0, d, 0.0]), fixed=True, role="foot"),
        _b.StructuralNode("b_rr", np.array([w, d, 0.0]), fixed=True, role="foot"),
        # Portal base junctions (not ground-fixed; transfer portal loads via base rails to feet)
        _b.StructuralNode(
            "p_bl", np.array([0.0, y_portal, 0.0]), fixed=False, lumped_mass_kg=z_motor_kg, role="mount"
        ),
        _b.StructuralNode(
            "p_br", np.array([w, y_portal, 0.0]), fixed=False, lumped_mass_kg=z_motor_kg, role="mount"
        ),
        # Z-axis gantry height nodes on vertical uprights
        _b.StructuralNode(
            "z_l",
            np.array([0.0, y_portal, z_gantry]),
            lumped_mass_kg=carriage_side_kg + xy_motor_kg,
            role="gantry",
        ),
        _b.StructuralNode(
            "z_r", np.array([w, y_portal, z_gantry]), lumped_mass_kg=carriage_side_kg, role="gantry"
        ),
        # Portal top corners
        _b.StructuralNode("p_tl", np.array([0.0, y_portal, h]), role="frame"),
        _b.StructuralNode("p_tr", np.array([w, y_portal, h]), role="frame"),
        # Toolhead on X gantry
        _b.StructuralNode(
            "toolhead",
            np.array([x_tool, y_portal, z_gantry]),
            lumped_mass_kg=toolhead_kg,
            role="toolhead",
        ),
    ]

    fp = spec.frame.profile
    up = spec.frame.effective_upright_profile
    fm = spec.frame.material
    bp = spec.frame.brace_profile
    bm = spec.frame.brace_material
    members: list[StructuralMember] = []

    # Base H-frame
    _b._add_member(members, "base_l_f", "b_fl", "p_bl", fp, fm)
    _b._add_member(members, "base_l_r", "p_bl", "b_rl", fp, fm)
    _b._add_member(members, "base_r_f", "b_fr", "p_br", fp, fm)
    _b._add_member(members, "base_r_r", "p_br", "b_rr", fp, fm)
    _b._add_member(members, "base_cross", "p_bl", "p_br", fp, fm)
    _b._add_member(members, "base_front", "b_fl", "b_fr", fp, fm)
    _b._add_member(members, "base_rear", "b_rl", "b_rr", fp, fm)

    # Vertical Z portal uprights + top bar
    _b._add_member(members, "z_col_l_low", "p_bl", "z_l", up, fm)
    _b._add_member(members, "z_col_l_top", "z_l", "p_tl", up, fm)
    _b._add_member(members, "z_col_r_low", "p_br", "z_r", up, fm)
    _b._add_member(members, "z_col_r_top", "z_r", "p_tr", up, fm)
    _b._add_member(members, "portal_top", "p_tl", "p_tr", fp, fm)

    # Moving X gantry beam
    xp = spec.gantry.x_beam_profile
    xm = spec.gantry.x_beam_material
    _b._add_member(members, "x_gantry_l", "z_l", "toolhead", xp, xm, role="gantry")
    _b._add_member(members, "x_gantry_r", "toolhead", "z_r", xp, xm, role="gantry")

    # Modeled & simulated frame bracing is attached below via add_modeled_frame_bracing()
    nmap = {n.id: n for n in nodes}
    rail_mat = get_material(spec.gantry.rail_material)
    x_rail_prof = get_profile(spec.gantry.x_rail_profile)
    base_x_prof = get_profile(xp)
    x_normals = _b._x_rail_mount_normals(spec.gantry.effective_x_rail_mount)
    bolt_pitch_m = spec.gantry.effective_x_rail_bolt_pitch_mm * 1e-3
    member_by_id = {m.id: m for m in members}
    chamber_c = float(spec.frame.chamber_temp_c)
    x_beam_mat = get_material(xm)
    if x_beam_mat.tg_c is not None and abs(chamber_c - 25.0) > 1e-6:
        eff_e = x_beam_mat.effective_modulus_pa(chamber_c)
        eff_g = x_beam_mat.effective_shear_modulus_at_temp_pa(chamber_c)
        for mid in ("x_gantry_l", "x_gantry_r"):
            m_xb = member_by_id[mid]
            m_xb.material = m_xb.material.model_copy(
                update={
                    "youngs_modulus_pa": eff_e,
                    "shear_modulus_pa": eff_g,
                }
            )

    x_boost_iy = 0.0
    x_boost_iz = 0.0
    cf_mount = spec.gantry.effective_cf_rail_mounting
    for mid in ("x_gantry_l", "x_gantry_r"):
        d_iy_x, d_iz_x, _ = _b._apply_bolted_linear_rail(
            member_by_id[mid],
            nmap,
            x_rail_prof,
            rail_mat,
            x_normals,
            beam_shape=base_x_prof.shape,
            bolt_pitch_m=bolt_pitch_m,
            l_eff_m=w,
            chamber_temp_c=chamber_c,
            cf_rail_mounting=cf_mount,
        )
        x_boost_iy += d_iy_x
        x_boost_iz += d_iz_x
    x_boost_iy *= 0.5
    x_boost_iz *= 0.5

    if spec.gantry.rail_backer_material != "none" and not spec.gantry.effective_dual_x_rails:
        backer_mat = get_material(spec.gantry.rail_backer_material)
        backer_prof = _b._make_x_backer_profile(
            base_x_prof, x_rail_prof, spec.gantry.rail_backer_thickness_mm
        )
        backer_normals = [-x_normals[0]]
        bk_iy_acc = 0.0
        bk_iz_acc = 0.0
        for mid in ("x_gantry_l", "x_gantry_r"):
            d_iy_b, d_iz_b, _ = _b._apply_bolted_linear_rail(
                member_by_id[mid],
                nmap,
                backer_prof,
                backer_mat,
                backer_normals,
                beam_shape=base_x_prof.shape,
                bolt_pitch_m=bolt_pitch_m,
                l_eff_m=w,
                track_as_rail=False,
                chamber_temp_c=chamber_c,
                cf_rail_mounting=cf_mount,
            )
            bk_iy_acc += d_iy_b
            bk_iz_acc += d_iz_b
        x_boost_iy += 0.5 * bk_iy_acc
        x_boost_iz += 0.5 * bk_iz_acc

    x_beam_insert_mass_kg = _b._apply_cf_two_wall_clamped_inserts(
        member_by_id, nmap, spec, w
    )
    x_rail_ei_y_boost_pct = 100.0 * x_boost_iz / max(base_x_prof.iz_m4, 1e-18)
    x_rail_ei_z_boost_pct = 100.0 * x_boost_iy / max(base_x_prof.iy_m4, 1e-18)

    z_rail_id = spec.bed.z_rail_profile
    if z_rail_id != "none":
        n_z = [np.array([0.0, -1.0, 0.0])]
        for side in ("l", "r"):
            for mid in (f"z_col_{side}_low", f"z_col_{side}_top"):
                if mid in member_by_id:
                    _b._apply_z_guideway(
                        member_by_id[mid], nmap, z_rail_id, rail_mat, n_z, spec.gantry.rail_bolt_pitch_mm * 1e-3, h
                    )

    _b._apply_corner_joint_compliance(members, spec, col_length_m=h)

    # Bedslinger: X moves toolhead (+ bracket delta); Y moves bed + carriage (+ bracket delta when specified)
    bed_kg = spec.bed.mass_g * 1e-3
    x_beam_mass = sum(
        max(
            m.mass_kg(nmap)
            - m.rail_mass_kg
            - m.backer_mass_kg
            - m.fill_mass_kg
            - m.insert_mass_kg,
            0.0,
        )
        for m in members
        if m.role == "gantry"
    )
    x_backer_mass = sum(m.backer_mass_kg for m in members if m.role == "gantry")
    x_rail_mass = x_rail_prof.area_m2 * w * rail_mat.density_kg_m3
    if spec.gantry.effective_dual_x_rails:
        x_rail_mass *= 2.0
    x_moving = toolhead_kg + 0.03 + bkt_delta_kg
    y_moving = bed_kg + y_carriage_total_kg + bkt_delta_kg

    # Explicit bedslinger Y-bed node mounted to base-level Y-rails (z=0.015 m, split at y_bed)
    bed_center = np.array([w * 0.5, y_bed, orig_z])
    z_y_rail = min(orig_z - 0.014, 0.015)
    fastener_springs: list[FastenerSpring] = []
    nodes.extend(
        [
            _b.StructuralNode("bed_rail_l", np.array([0.0, y_bed, z_y_rail]), role="bed"),
            _b.StructuralNode("bed_rail_r", np.array([w, y_bed, z_y_rail]), role="bed"),
        ]
    )
    _b._attach_bed_mount_compliance(nodes, fastener_springs, spec, bed_center)
    bed_vmat = _b._make_bed_virtual_material(get_material(spec.bed.plate_material))
    rail_vmat = _b._make_bed_virtual_material(get_material(spec.gantry.rail_material))
    t_mm = max(float(spec.bed.plate_thickness_mm), 2.0)
    t_ratio = t_mm / 4.0
    y_rail_prof = get_profile(spec.gantry.y_rail_profile)
    y_scale_i = 4.0 if spec.gantry.y_rail_profile.startswith("steel_rod") else 50.0
    y_rail_guide_prof = ProfileSpec.model_construct(
        id="bed_y_rail_guide_profile",
        name=f"Bedslinger Y-Rail Guide ({spec.gantry.y_rail_profile})",
        shape=y_rail_prof.shape,
        width_m=y_rail_prof.width_m,
        height_m=y_rail_prof.height_m,
        area_m2=y_rail_prof.area_m2,
        iy_m4=y_rail_prof.iy_m4 * y_scale_i,
        iz_m4=y_rail_prof.iz_m4 * y_scale_i,
        j_m4=y_rail_prof.j_m4 * y_scale_i,
    )
    for tag, f_nid, r_nid, rail_nid in (
        ("l", "b_fl", "b_rl", "bed_rail_l"),
        ("r", "b_fr", "b_rr", "bed_rail_r"),
    ):
        members.append(
            _b.StructuralMember(
                id=f"bed_y_rail_{tag}_f",
                start_id=f_nid,
                end_id=rail_nid,
                profile=y_rail_guide_prof,
                material=rail_vmat,
                role="bed",
            )
        )
        members.append(
            _b.StructuralMember(
                id=f"bed_y_rail_{tag}_r",
                start_id=r_nid,
                end_id=rail_nid,
                profile=y_rail_guide_prof,
                material=rail_vmat,
                role="bed",
            )
        )

    l_arm = max(w * 0.5, 0.05)
    belt_ea_n = float(resolve_belt(spec)["stiffness_ea_n"])
    k_y_target = 1.15e6 * ((belt_ea_n / 50000.0) ** 0.35)
    iz_y_carriage = k_y_target * (l_arm**3) / (24.0 * max(bed_vmat.youngs_modulus_pa, 1e6))
    bed_car_prof = ProfileSpec.model_construct(
        id="bed_carriage_arm_profile",
        name=f"Bedslinger Y-Carriage & Belt Mount ({t_mm:.1f}mm {spec.bed.plate_material})",
        shape="solid_bar",
        width_m=0.014,
        height_m=max(0.010, 0.008 + t_mm * 1e-3),
        area_m2=2.5e-4,
        iy_m4=4.0e-8 + 4.0e-8 * (t_ratio**3),
        iz_m4=iz_y_carriage,
        j_m4=6.0e-8,
    )
    for tag, rail_nid in (("l", "bed_rail_l"), ("r", "bed_rail_r")):
        members.append(
            _b.StructuralMember(
                id=f"bed_arm_{tag}",
                start_id=rail_nid,
                end_id="bed_frame",
                profile=bed_car_prof,
                material=bed_vmat,
                role="bed",
            )
        )

    from printsim.geometry._builder_bracing import add_modeled_frame_bracing

    frame_superelements: list[_b.AttachedSuperelement] = []
    frame_brace_res = add_modeled_frame_bracing(
        spec=spec,
        nodes=nodes,
        nmap=nmap,
        members=members,
        springs=fastener_springs,
        superelements=frame_superelements,
        allow_gantry=False,
    )

    stage_stiffnesses = _b._attach_toolhead_cog_and_nozzle(
        nodes, members, spec, base_node_id="toolhead", fastener_springs=fastener_springs
    )
    if bkt_delta_kg > 0.0:
        next(n for n in nodes if n.id == "toolhead").lumped_mass_kg += bkt_delta_kg

    return _b.AssemblyModel(
        spec=spec,
        nodes=nodes,
        members=members,
        fastener_springs=fastener_springs,
        superelements=frame_superelements,
        frame_brace_cad_mass_kg=float(frame_brace_res["cad_mass_kg"]),
        frame_brace_flat_bar_mass_kg=float(frame_brace_res["flat_bar_mass_kg"]),
        frame_brace_cost_usd=float(frame_brace_res["cost_usd"]),
        frame_brace_cad_count=int(frame_brace_res["cad_count"]),
        frame_brace_cad_asset_id=frame_brace_res["cad_asset_id"],
        frame_brace_cad_meta=dict(frame_brace_res.get("cad_meta") or {}),
        frame_brace_visuals=list(frame_brace_res["visuals"]),
        toolhead_node_id="toolhead",
        toolhead_cog_node_id="toolhead_cog",
        nozzle_node_id="nozzle_tip",
        bed_node_id="bed_surface",
        gantry_node_ids=["z_l", "toolhead", "z_r"],
        bed_center_m=bed_center,
        build_origin_m=np.array([orig_x, orig_y, orig_z]),
        x_moving_mass_kg=x_moving,
        y_moving_mass_kg=y_moving,
        x_beam_mass_kg=x_beam_mass,
        x_rail_mass_kg=x_rail_mass,
        x_backer_mass_kg=x_backer_mass,
        x_beam_insert_mass_kg=x_beam_insert_mass_kg,
        y_carriage_joints_mass_kg=y_carriage_total_kg,
        y_carriage_block_delta_g=y_block_delta_g,
        toolhead_moving_mass_kg=toolhead_kg,
        x_rail_ei_y_boost_pct=x_rail_ei_y_boost_pct,
        x_rail_ei_z_boost_pct=x_rail_ei_z_boost_pct,
        total_rail_mass_kg=sum(m.rail_mass_kg for m in members),
        toolhead_stage_stiffnesses_n_per_mm=stage_stiffnesses,
    )
