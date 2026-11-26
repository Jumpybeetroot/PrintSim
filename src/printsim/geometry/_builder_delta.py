"""Delta 3-tower structural assembly builder extracted from ``builder.py``."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from printsim.schema import (
    PrinterSpec,
    ProfileSpec,
    get_material,
    get_profile,
    resolve_gantry_bracket,
)

if TYPE_CHECKING:
    from printsim.geometry.builder import (
        AssemblyModel,
        FastenerSpring,
        StructuralMember,
        StructuralNode,
    )


def _build_delta(
    spec: PrinterSpec, toolhead_rel_pos: tuple[float, float, float]
) -> AssemblyModel:
    """Build a 3D FEA structural assembly for a 3-tower parallel-arm Delta printer."""
    from printsim.geometry import builder as _b

    fr_prof = get_profile(spec.frame.profile)
    w = spec.frame.width_mm * 1e-3
    d = spec.frame.depth_mm * 1e-3
    h = max(spec.frame.height_mm * 1e-3 - fr_prof.height_m, 0.08)
    bx, by, bz = np.array(spec.kinematics.build_volume_mm) * 1e-3
    _, _, w_rel = np.clip(toolhead_rel_pos, 0.1, 0.8)

    r_eff = 0.030  # 30 mm effector rod pivot radius
    r_car = 0.015  # 15 mm inward carriage rod pivot offset from tower axis
    r_delta = max(spec.kinematics.delta_radius_mm * 1e-3, 0.04)
    r_tower = r_delta + r_eff + r_car
    arm_len = max(spec.kinematics.delta_arm_length_mm * 1e-3, r_delta + 0.03)
    pair_half_spacing = 0.022  # 44 mm tangential parallel-pair rod spacing

    # Center delta prism around (cx, cy) so coordinates stay positive & honor delta_radius_mm
    cx = max(r_tower, 0.5 * w)
    cy = max(r_tower, 0.5 * d)
    angles = [math.radians(90.0), math.radians(210.0), math.radians(330.0)]
    dz_car = math.sqrt(max(arm_len**2 - r_delta**2, 1e-4))
    z_eff_min = 0.06
    z_eff_max = max(z_eff_min + 0.01, h - 0.05 - dz_car)
    z_effector = float(np.clip(0.08 + w_rel * min(bz, h * 0.55), z_eff_min, z_eff_max))
    z_car = min(h - 0.04, z_effector + dz_car)

    motor_kg = spec.kinematics.xy_stepper.motor_mass_g * 1e-3
    toolhead_kg = _b._effective_toolhead_mass_kg(spec)
    bkt_resolved = resolve_gantry_bracket(spec)
    bkt_delta_kg = float(bkt_resolved["carriage_mass_delta_g"]) * 1e-3
    car_kg = (spec.gantry.carriage_mass_g / 3.0) * 1e-3

    nodes: list[StructuralNode] = []
    for i, ang in enumerate(angles):
        er_x, er_y = math.cos(ang), math.sin(ang)
        et_x, et_y = -er_y, er_x
        px = cx + r_tower * er_x
        py = cy + r_tower * er_y
        nodes.append(
            _b.StructuralNode(
                f"b_{i}", np.array([px, py, 0.0]), fixed=True, lumped_mass_kg=motor_kg, role="foot"
            )
        )
        nodes.append(
            _b.StructuralNode(
                f"c_{i}", np.array([px, py, z_car]), lumped_mass_kg=car_kg, role="gantry"
            )
        )
        nodes.append(_b.StructuralNode(f"t_{i}", np.array([px, py, h]), role="frame"))
        for side, s_sign in (("a", -1.0), ("b", 1.0)):
            lat = s_sign * pair_half_spacing
            cx_p = cx + (r_tower - r_car) * er_x + lat * et_x
            cy_p = cy + (r_tower - r_car) * er_y + lat * et_y
            ex_p = cx + r_eff * er_x + lat * et_x
            ey_p = cy + r_eff * er_y + lat * et_y
            nodes.append(
                _b.StructuralNode(f"c_{i}_{side}", np.array([cx_p, cy_p, z_car]), role="gantry")
            )
            nodes.append(
                _b.StructuralNode(
                    f"eff_{i}_{side}", np.array([ex_p, ey_p, z_effector]), role="toolhead"
                )
            )

    nodes.append(
        _b.StructuralNode(
            "toolhead",
            np.array([cx, cy, z_effector]),
            lumped_mass_kg=toolhead_kg,
            role="toolhead",
        )
    )

    fp = spec.frame.profile
    up = spec.frame.effective_upright_profile
    fm = spec.frame.material
    bp = spec.frame.brace_profile
    bm = spec.frame.brace_material
    members: list[StructuralMember] = []

    eff_link_vmat = _b._make_bed_virtual_material(get_material("carbon_fiber_tube")).model_copy(
        update={"youngs_modulus_pa": 120.0e9}
    )
    eff_link_prof = ProfileSpec.model_construct(
        id="delta_rigid_link_profile",
        name="Delta Rigid Carriage/Effector Link",
        shape="solid_bar",
        width_m=0.014,
        height_m=0.014,
        area_m2=2.0e-4,
        iy_m4=3.0e-8,
        iz_m4=3.0e-8,
        j_m4=3.0e-8,
    )

    for i in range(3):
        j = (i + 1) % 3
        _b._add_member(members, f"base_{i}", f"b_{i}", f"b_{j}", fp, fm)
        _b._add_member(members, f"top_{i}", f"t_{i}", f"t_{j}", fp, fm)
        _b._add_member(members, f"tower_{i}_low", f"b_{i}", f"c_{i}", up, fm, role="tower")
        _b._add_member(members, f"tower_{i}_top", f"c_{i}", f"t_{i}", up, fm, role="tower")
        for side in ("a", "b"):
            members.append(
                _b.StructuralMember(
                    id=f"car_link_{i}_{side}",
                    start_id=f"c_{i}",
                    end_id=f"c_{i}_{side}",
                    profile=eff_link_prof,
                    material=eff_link_vmat,
                    role="toolhead",
                )
            )
            members.append(
                _b.StructuralMember(
                    id=f"eff_link_{i}_{side}",
                    start_id="toolhead",
                    end_id=f"eff_{i}_{side}",
                    profile=eff_link_prof,
                    material=eff_link_vmat,
                    role="toolhead",
                )
            )
            # Twin parallel carbon delta rods constraining effector X/Y/Z and pitch/roll/yaw
            _b._add_member(
                members,
                f"arm_{i}_{side}",
                f"c_{i}_{side}",
                f"eff_{i}_{side}",
                "brace_rod_10mm",
                "carbon_fiber_tube",
                role="gantry",
                is_truss=True,
            )

        if spec.frame.bracing in ("rear_cross", "full_truss"):
            _b._add_member(
                members, f"brace_{i}_a", f"b_{i}", f"t_{j}", bp, bm, role="brace", is_truss=True
            )
        if spec.frame.bracing == "full_truss":
            _b._add_member(
                members, f"brace_{i}_b", f"b_{j}", f"t_{i}", bp, bm, role="brace", is_truss=True
            )

    z_rail_id = spec.bed.z_rail_profile
    if z_rail_id != "none":
        nmap_d = {n.id: n for n in nodes}
        rail_mat_d = get_material(spec.gantry.rail_material)
        bolt_pitch_d = spec.gantry.rail_bolt_pitch_mm * 1e-3
        member_by_id_d = {m.id: m for m in members}
        for i, ang in enumerate(angles):
            n_rad = [np.array([-math.cos(ang), -math.sin(ang), 0.0])]
            for mid in (f"tower_{i}_low", f"tower_{i}_top"):
                if mid in member_by_id_d:
                    mem_t = member_by_id_d[mid]
                    _b._apply_z_guideway(
                        mem_t, nmap_d, z_rail_id, rail_mat_d, n_rad, bolt_pitch_d, h
                    )
                    # Preserve exact 120-deg C3 rotational symmetry without off-diagonal I_yz
                    i_iso = 0.5 * (mem_t.profile.iy_m4 + mem_t.profile.iz_m4)
                    mem_t.profile = mem_t.profile.model_copy(
                        update={"iy_m4": i_iso, "iz_m4": i_iso}
                    )

    panels_active, panel_mounting, pmat, panel_t_m = _b._panel_shear_params(spec)
    total_panel_mass_kg = 0.0
    rep_panel_props: dict[str, float] | None = None
    if panels_active:
        nmap = {n.id: n for n in nodes}
        for i in range(3):
            j = (i + 1) % 3
            _, bay_mass, props = _b._add_shear_panel_bay(
                members,
                nmap,
                f"bay{i}",
                f"b_{i}",
                f"b_{j}",
                f"t_{i}",
                f"t_{j}",
                pmat,
                panel_t_m,
                panel_mounting,
            )
            total_panel_mass_kg += bay_mass
            if rep_panel_props is None:
                rep_panel_props = props

    bed_center = np.array([cx, cy, 0.06])
    fastener_springs: list[FastenerSpring] = []
    _b._attach_bed_mount_compliance(nodes, fastener_springs, spec, bed_center)
    bed_vmat = _b._make_bed_virtual_material(get_material(spec.bed.plate_material))
    t_mm = max(float(spec.bed.plate_thickness_mm), 2.0)
    t_ratio = t_mm / 6.0
    delta_bed_prof = ProfileSpec.model_construct(
        id="bed_delta_mount_profile",
        name=f"Delta Base Bed Mount ({t_mm:.1f}mm {spec.bed.plate_material})",
        shape="solid_bar",
        width_m=0.020,
        height_m=max(0.012, 0.010 + t_mm * 1e-3),
        area_m2=8.0e-4,
        iy_m4=3.0e-8 + 9.0e-8 * (t_ratio**3),
        iz_m4=1.4e-7,
        j_m4=5.0e-8,
    )
    for i in range(3):
        members.append(
            _b.StructuralMember(
                id=f"bed_arm_{i}",
                start_id=f"b_{i}",
                end_id="bed_frame",
                profile=delta_bed_prof,
                material=bed_vmat,
                role="bed",
            )
        )

    eff_mass = toolhead_kg + 0.09  # ultra-light effector + arm portion
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
        toolhead_node_id="toolhead",
        toolhead_cog_node_id="toolhead_cog",
        nozzle_node_id="nozzle_tip",
        bed_node_id="bed_surface",
        gantry_node_ids=["c_0", "c_1", "c_2", "toolhead"],
        bed_center_m=bed_center,
        build_origin_m=np.array([cx - bx * 0.5, cy - by * 0.5, 0.06]),
        x_moving_mass_kg=eff_mass + bkt_delta_kg,
        y_moving_mass_kg=eff_mass + bkt_delta_kg,
        y_carriage_joints_mass_kg=spec.gantry.carriage_mass_g * 1e-3 + bkt_delta_kg,
        toolhead_moving_mass_kg=toolhead_kg,
        panel_mass_kg=total_panel_mass_kg,
        total_rail_mass_kg=sum(m.rail_mass_kg for m in members),
        panel_rep_props=rep_panel_props,
        toolhead_stage_stiffnesses_n_per_mm=stage_stiffnesses,
    )
