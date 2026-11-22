"""3D structural assembly graph generator from declarative PrinterSpec."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from printsim.schema import (
    BED_MOUNT_CATALOG,
    EXTRUSION_FILLINGS,
    FRAME_FEET_CATALOG,
    GANTRY_BRACKET_CATALOG,
    HOTEND_CATALOG,
    PROFILE_CATALOG,
    TOOLHEAD_CONSTRUCTION_CATALOG,
    Z_DRIVE_MECHANISM_CATALOG,
    Z_PROBE_CATALOG,
    Z_RAIL_PROFILES,
    MaterialSpec,
    PrinterSpec,
    ProfileSpec,
    get_material,
    get_profile,
)


@dataclass
class StructuralNode:
    """A 3D structural FEA node in SI units (meters, kg)."""

    id: str
    position_m: np.ndarray  # shape (3,) [x, y, z] in meters
    fixed: bool = False  # all 6 DOFs constrained (foot bolted/seated on bench)
    pinned: bool = False  # 3 translations constrained
    lumped_mass_kg: float = 0.0
    role: str = "frame"


@dataclass
class StructuralMember:
    """A 3D beam/truss element connecting two nodes."""

    id: str
    start_id: str
    end_id: str
    profile: ProfileSpec
    material: MaterialSpec
    role: str = "frame"
    is_truss: bool = False  # if True, reduced rotational coupling (rod/brace)
    fill_mass_kg: float = 0.0
    rail_mass_kg: float = 0.0
    backer_mass_kg: float = 0.0
    backer_material_id: str = ""

    def length_m(self, node_map: dict[str, StructuralNode]) -> float:
        p0 = node_map[self.start_id].position_m
        p1 = node_map[self.end_id].position_m
        return float(np.linalg.norm(p1 - p0))

    def mass_kg(self, node_map: dict[str, StructuralNode]) -> float:
        return self.profile.area_m2 * self.length_m(node_map) * self.material.density_kg_m3


@dataclass
class ShellPanelElement:
    """A 4-node 24-DOF MITC4 flat shell element representing an enclosure/structural panel quad."""

    id: str
    bay: str
    node_ids: tuple[str, str, str, str]  # CCW ordered
    material: MaterialSpec
    thickness_m: float
    areal_mass_kg_m2: float
    d_bend_nm: float | None = None
    g_transverse_h_n_m: float | None = None
    e_mem_h_n_m: float | None = None
    normal_vec: tuple[float, float, float] = (0.0, 0.0, 1.0)


@dataclass
class FastenerSpring:
    """A 2-node 12-DOF 3D CBUSH spring connecting a shell perimeter node to a frame node."""

    id: str
    bay: str
    shell_node_id: str
    frame_node_id: str
    k_shear_n_m: float
    k_normal_n_m: float
    k_rot_nm_rad: float
    normal_vec: tuple[float, float, float] = (0.0, 0.0, 1.0)
    offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class AssemblyModel:
    """Complete physical 3D assembly model ready for FEA, kinematics, and visualization."""

    spec: PrinterSpec
    nodes: list[StructuralNode] = field(default_factory=list)
    members: list[StructuralMember] = field(default_factory=list)
    shell_elements: list[ShellPanelElement] = field(default_factory=list)
    fastener_springs: list[FastenerSpring] = field(default_factory=list)
    panel_mesh_n: int = 2
    toolhead_node_id: str = "toolhead"
    toolhead_cog_node_id: str = "toolhead_cog"
    nozzle_node_id: str = "nozzle_tip"
    bed_node_id: str = "bed_surface"
    gantry_node_ids: list[str] = field(default_factory=list)
    bed_center_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    build_origin_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    x_moving_mass_kg: float = 0.0
    y_moving_mass_kg: float = 0.0
    x_beam_mass_kg: float = 0.0
    x_rail_mass_kg: float = 0.0
    x_backer_mass_kg: float = 0.0
    y_carriage_joints_mass_kg: float = 0.0
    toolhead_moving_mass_kg: float = 0.0
    frame_mass_kg: float = 0.0
    panel_mass_kg: float = 0.0
    extrusion_fill_mass_kg: float = 0.0
    x_rail_ei_y_boost_pct: float = 0.0
    x_rail_ei_z_boost_pct: float = 0.0
    y_rail_ei_x_boost_pct: float = 0.0
    total_rail_mass_kg: float = 0.0
    frame_damping_ratio: float = 0.042
    total_mass_kg: float = 0.0
    estimated_material_cost_usd: float = 0.0
    panel_rep_props: dict[str, float] | None = None
    toolhead_stage_stiffnesses_n_per_mm: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.nodes:
            node_ids = {n.id for n in self.nodes}
            if self.toolhead_cog_node_id not in node_ids:
                self.toolhead_cog_node_id = self.toolhead_node_id
            if self.nozzle_node_id not in node_ids:
                self.nozzle_node_id = self.toolhead_node_id
            if self.bed_node_id and self.bed_node_id not in node_ids:
                self.bed_node_id = ""

    @property
    def node_map(self) -> dict[str, StructuralNode]:
        return {n.id: n for n in self.nodes}

    def _compute_bed_mount_points_mm(self) -> list[list[float]]:
        """Return bed-underside mounting coordinates (mm) strictly inside heated bed footprint."""
        bc = self.bed_center_m * 1000.0
        bx = float(self.spec.bed.size_mm[0])
        by = float(self.spec.bed.size_mm[1])
        bz = float(bc[2]) - 0.5 * float(self.spec.bed.plate_thickness_mm)
        if self.spec.architecture == "delta":
            r_m = 0.36 * min(bx, by)
            return [
                [
                    round(float(bc[0]) + r_m * math.cos(math.radians(deg)), 2),
                    round(float(bc[1]) + r_m * math.sin(math.radians(deg)), 2),
                    round(bz, 2),
                ]
                for deg in (90.0, 210.0, 330.0)
            ]
        if self.spec.bed.mount_type in ("kinematic_3pt", "whopping_pochard_kinematic"):
            return [
                [round(float(bc[0]) - 0.36 * bx, 2), round(float(bc[1]) - 0.36 * by, 2), round(bz, 2)],
                [round(float(bc[0]) + 0.36 * bx, 2), round(float(bc[1]) - 0.36 * by, 2), round(bz, 2)],
                [round(float(bc[0]), 2), round(float(bc[1]) + 0.36 * by, 2), round(bz, 2)],
            ]
        return [
            [round(float(bc[0]) - 0.36 * bx, 2), round(float(bc[1]) - 0.36 * by, 2), round(bz, 2)],
            [round(float(bc[0]) + 0.36 * bx, 2), round(float(bc[1]) - 0.36 * by, 2), round(bz, 2)],
            [round(float(bc[0]) - 0.36 * bx, 2), round(float(bc[1]) + 0.36 * by, 2), round(bz, 2)],
            [round(float(bc[0]) + 0.36 * bx, 2), round(float(bc[1]) + 0.36 * by, 2), round(bz, 2)],
        ]

    def _compute_z_rails_geometry(self) -> list[dict]:
        """Return Z-axis linear rails / smooth rods metadata (mm) for CAD/Studio rendering."""
        z_prof = self.spec.bed.z_rail_profile
        if z_prof == "none":
            return []
        nmap = self.node_map
        arch = self.spec.architecture
        z_drive = self.spec.bed.z_drive
        specs: list[tuple[str, str, str, str, tuple[float, float, float]]] = []
        if arch == "corexy":
            if z_drive == "belted_quad_z":
                specs = [
                    ("z_rail_fl", "b_fl", "t_fl", "col_g_fl", (1.0, 0.0, 0.0)),
                    ("z_rail_fr", "b_fr", "t_fr", "col_g_fr", (-1.0, 0.0, 0.0)),
                    ("z_rail_rl", "b_rl", "t_rl", "col_g_rl", (1.0, 0.0, 0.0)),
                    ("z_rail_rr", "b_rr", "t_rr", "col_g_rr", (-1.0, 0.0, 0.0)),
                ]
            elif z_drive == "leadscrew_triple":
                specs = [
                    ("z_rail_fl", "b_fl", "t_fl", "z_bed_fl", (0.0, 1.0, 0.0)),
                    ("z_rail_fr", "b_fr", "t_fr", "z_bed_fr", (0.0, 1.0, 0.0)),
                    ("z_rail_rear", "b_rear_mid", "g_rear_mid", "z_bed_r", (0.0, -1.0, 0.0)),
                ]
            elif z_drive == "leadscrew_dual":
                specs = [
                    ("z_rail_left", "b_left_mid", "g_left_mid", "z_bed_l", (1.0, 0.0, 0.0)),
                    ("z_rail_right", "b_right_mid", "g_right_mid", "z_bed_r", (-1.0, 0.0, 0.0)),
                ]
            elif z_drive == "leadscrew_single":
                specs = [
                    ("z_rail_rear", "b_rear_mid", "g_rear_mid", "z_bed_r", (0.0, -1.0, 0.0)),
                ]
        elif arch == "cartesian_bedslinger":
            specs = [
                ("z_rail_left", "p_bl", "p_tl", "z_l", (0.0, -1.0, 0.0)),
                ("z_rail_right", "p_br", "p_tr", "z_r", (0.0, -1.0, 0.0)),
            ]
        elif arch == "delta":
            for i in range(3):
                ang = math.radians(90.0 + 120.0 * i)
                specs.append(
                    (
                        f"z_rail_{i}",
                        f"b_{i}",
                        f"t_{i}",
                        f"c_{i}",
                        (-math.cos(ang), -math.sin(ang), 0.0),
                    )
                )

        out: list[dict] = []
        for rid, b_id, t_id, car_id, nvec in specs:
            if b_id not in nmap or t_id not in nmap:
                continue
            p0 = nmap[b_id].position_m * 1000.0
            p1 = nmap[t_id].position_m * 1000.0
            car_z = float(nmap[car_id].position_m[2] * 1000.0) if car_id in nmap else float(self.bed_center_m[2] * 1000.0)
            out.append(
                {
                    "id": rid,
                    "start_mm": [round(float(v), 2) for v in p0],
                    "end_mm": [round(float(v), 2) for v in p1],
                    "profile_id": z_prof,
                    "carriage_z_mm": round(car_z, 2),
                    "normal": [round(float(v), 4) for v in nvec],
                }
            )
        return out

    def _compute_z_leadscrews_geometry(self) -> list[dict]:
        """Return Z-axis leadscrews / ballscrews / Z-belts metadata (mm) for CAD/Studio."""
        mech_id = self.spec.bed.effective_z_leadscrew_type
        mech = Z_DRIVE_MECHANISM_CATALOG.get(mech_id, Z_DRIVE_MECHANISM_CATALOG["tr8x4_pom"])
        kind = str(mech["kind"])
        diam_mm = float(mech["diameter_mm"])
        nmap = self.node_map
        arch = self.spec.architecture
        z_drive = self.spec.bed.z_drive
        specs: list[tuple[str, str, str, str, tuple[float, float, float]]] = []
        if arch == "corexy":
            if z_drive == "belted_quad_z":
                specs = [
                    ("z_drive_fl", "b_fl", "t_fl", "col_g_fl", (18.0, 12.0, 0.0)),
                    ("z_drive_fr", "b_fr", "t_fr", "col_g_fr", (-18.0, 12.0, 0.0)),
                    ("z_drive_rl", "b_rl", "t_rl", "col_g_rl", (18.0, -12.0, 0.0)),
                    ("z_drive_rr", "b_rr", "t_rr", "col_g_rr", (-18.0, -12.0, 0.0)),
                ]
            elif z_drive == "leadscrew_triple":
                specs = [
                    ("z_drive_fl", "b_fl", "g_fl", "z_bed_fl", (20.0, 0.0, 0.0)),
                    ("z_drive_fr", "b_fr", "g_fr", "z_bed_fr", (-20.0, 0.0, 0.0)),
                    ("z_drive_rear", "b_rear_mid", "g_rear_mid", "z_bed_r", (0.0, -20.0, 0.0)),
                ]
            elif z_drive == "leadscrew_dual":
                specs = [
                    ("z_drive_left", "b_left_mid", "g_left_mid", "z_bed_l", (20.0, 0.0, 0.0)),
                    ("z_drive_right", "b_right_mid", "g_right_mid", "z_bed_r", (-20.0, 0.0, 0.0)),
                ]
            elif z_drive == "leadscrew_single":
                specs = [
                    ("z_drive_rear", "b_rear_mid", "g_rear_mid", "z_bed_r", (0.0, -20.0, 0.0)),
                ]
        elif arch == "cartesian_bedslinger":
            specs = [
                ("z_drive_left", "p_bl", "p_tl", "z_l", (20.0, 0.0, 0.0)),
                ("z_drive_right", "p_br", "p_tr", "z_r", (-20.0, 0.0, 0.0)),
            ]
        elif arch == "delta":
            for i in range(3):
                ang = math.radians(90.0 + 120.0 * i)
                specs.append(
                    (
                        f"z_drive_{i}",
                        f"b_{i}",
                        f"t_{i}",
                        f"c_{i}",
                        (-18.0 * math.cos(ang), -18.0 * math.sin(ang), 0.0),
                    )
                )

        out: list[dict] = []
        for zid, b_id, t_id, nut_id, off_mm in specs:
            if b_id not in nmap or t_id not in nmap:
                continue
            off = np.asarray(off_mm, dtype=float)
            p0 = nmap[b_id].position_m * 1000.0 + off
            p1 = nmap[t_id].position_m * 1000.0 + off
            nut_z = float(nmap[nut_id].position_m[2] * 1000.0) if nut_id in nmap else float(self.bed_center_m[2] * 1000.0)
            out.append(
                {
                    "id": zid,
                    "kind": kind,
                    "mechanism_id": mech_id,
                    "start_mm": [round(float(v), 2) for v in p0],
                    "end_mm": [round(float(v), 2) for v in p1],
                    "nut_z_mm": round(nut_z, 2),
                    "motor_pos_mm": [round(float(v), 2) for v in p0],
                    "diameter_mm": round(diam_mm, 2),
                }
            )
        return out

    def to_geometry_dict(self) -> dict:
        """Serialize assembly geometry (mm) for the 3D web viewer and AI tools."""
        nmap = self.node_map
        pmat = get_material(self.spec.frame.panel_material)
        fill_id = self.spec.frame.extrusion_filling
        fill_spec = EXTRUSION_FILLINGS.get(fill_id, EXTRUSION_FILLINGS["none"])
        bkt_id = self.spec.gantry.bracket_material
        bkt_info = GANTRY_BRACKET_CATALOG.get(bkt_id, GANTRY_BRACKET_CATALOG["abs_printed"])
        panels_active = bool(
            self.spec.frame.enclosed_panels and self.spec.frame.panel_mounting != "none"
        )
        rep_props = self.panel_rep_props
        if panels_active and rep_props is None:
            rep_h_m = (
                max(self.spec.frame.height_mm * 1e-3 - 0.055, 0.04)
                if self.spec.panels_bolted_to_gantry_effective
                else self.spec.frame.height_mm * 1e-3
            )
            rep_props = panel_mechanical_properties(
                pmat,
                self.spec.frame.panel_thickness_mm * 1e-3,
                self.spec.frame.width_mm * 1e-3,
                rep_h_m,
                self.spec.frame.panel_mounting,
                bolted_to_gantry=self.spec.panels_bolted_to_gantry_effective,
            )
        return {
            "architecture": self.spec.architecture,
            "drive_mode": self.spec.gantry.drive_mode,
            "belt_routing": self.spec.gantry.belt_routing,
            "double_shear_pulleys": self.spec.gantry.double_shear_pulleys,
            "extrusion_filling": {
                "id": fill_id,
                "name": fill_spec["name"],
                "short_name": fill_spec["short_name"],
                "bonded": fill_spec["bonded"],
                "added_mass_kg": round(self.extrusion_fill_mass_kg, 3),
                "damping_ratio": round(self.frame_damping_ratio, 4),
                "color_hex": fill_spec["color_hex"],
                "notes": fill_spec["notes"],
            },
            "gantry_brackets": {
                "material_id": bkt_id,
                "name": bkt_info["name"],
                "short_name": bkt_info["short_name"],
                "youngs_modulus_gpa": bkt_info["youngs_modulus_gpa"],
                "hdt_c": bkt_info["hdt_c"],
                "creep_tension_limit_n": bkt_info["creep_tension_limit_n"],
                "cnc_machined": bkt_info["cnc_machined"],
                "kit_cost_usd": bkt_info["kit_cost_usd"],
                "carriage_mass_delta_g": bkt_info["carriage_mass_delta_g"],
                "color_hex": bkt_info["color_hex"],
            },
            "linear_rails": {
                "x_rail_profile": self.spec.gantry.x_rail_profile,
                "y_rail_profile": self.spec.gantry.y_rail_profile,
                "rail_material": self.spec.gantry.rail_material,
                "rail_quality": self.spec.gantry.rail_quality,
                "rail_preload": self.spec.gantry.rail_preload,
                "x_rail_mount": self.spec.gantry.effective_x_rail_mount,
                "y_rail_mount": self.spec.gantry.y_rail_mount,
                "dual_x_rails": self.spec.gantry.effective_dual_x_rails,
                "x_rail_width_mm": round(get_profile(self.spec.gantry.x_rail_profile).width_m * 1e3, 1),
                "x_rail_height_mm": round(get_profile(self.spec.gantry.x_rail_profile).height_m * 1e3, 1),
                "y_rail_width_mm": round(get_profile(self.spec.gantry.y_rail_profile).width_m * 1e3, 1),
                "y_rail_height_mm": round(get_profile(self.spec.gantry.y_rail_profile).height_m * 1e3, 1),
                "x_beam_ei_y_boost_pct": round(self.x_rail_ei_y_boost_pct, 1),
                "x_beam_ei_z_boost_pct": round(self.x_rail_ei_z_boost_pct, 1),
                "y_rail_ei_x_boost_pct": round(self.y_rail_ei_x_boost_pct, 1),
                "total_rail_mass_g": round(self.total_rail_mass_kg * 1e3, 1),
            },
            "panels": {
                "enabled": panels_active,
                "mounting": self.spec.frame.panel_mounting if panels_active else "none",
                "material_id": self.spec.frame.panel_material,
                "material_name": pmat.name,
                "thickness_mm": self.spec.frame.panel_thickness_mm,
                "color_hex": pmat.color_hex,
                "bolted_to_gantry": self.spec.panels_bolted_to_gantry_effective,
                "is_flying_gantry": self.spec.is_flying_gantry,
                "effective_shear_gt_kn_mm": (
                    round(rep_props["gt_eff_n_m"] * 1e-6, 3)
                    if panels_active and rep_props is not None
                    else 0.0
                ),
                "buckling_factor": (
                    round(rep_props["psi_buckle"], 3)
                    if panels_active and rep_props is not None
                    else 0.0
                ),
                "areal_mass_kg_m2": (
                    round(rep_props["areal_mass_kg_m2"], 2)
                    if panels_active and rep_props is not None
                    else 0.0
                ),
                "total_panel_mass_kg": round(self.panel_mass_kg, 3) if panels_active else 0.0,
            },
            "panel_mesh_n": self.panel_mesh_n,
            "shell_panels": [
                {
                    "id": s.id,
                    "bay": s.bay,
                    "node_ids": list(s.node_ids),
                    "thickness_mm": round(s.thickness_m * 1e3, 2),
                }
                for s in self.shell_elements
            ],
            "frame_dims_mm": [
                self.spec.frame.width_mm,
                self.spec.frame.depth_mm,
                self.spec.frame.height_mm,
            ],
            "build_volume_mm": list(self.spec.kinematics.build_volume_mm),
            "build_origin_mm": (self.build_origin_m * 1000.0).tolist(),
            "bed_size_mm": list(self.spec.bed.size_mm),
            "bed_center_mm": (self.bed_center_m * 1000.0).tolist(),
            "bed_mount_type": self.spec.bed.mount_type,
            "bed_z_drive": self.spec.bed.z_drive,
            "z_rail_profile": self.spec.bed.z_rail_profile,
            "z_leadscrew_type": self.spec.bed.z_leadscrew_type,
            "effective_z_leadscrew_type": self.spec.bed.effective_z_leadscrew_type,
            "feet_type": self.spec.frame.feet_type,
            "bed_frame_node_id": "bed_frame" if "bed_frame" in nmap else "",
            "bed_support_points_mm": [
                (nmap[m.start_id].position_m * 1000.0).tolist()
                for m in self.members
                if m.id.startswith("bed_arm_") and m.start_id in nmap
            ],
            "bed_mount_points_mm": self._compute_bed_mount_points_mm(),
            "z_rails": self._compute_z_rails_geometry(),
            "z_leadscrews": self._compute_z_leadscrews_geometry(),
            "toolhead_node_id": self.toolhead_node_id,
            "toolhead_cog_node_id": self.toolhead_cog_node_id,
            "nozzle_node_id": self.nozzle_node_id,
            "bed_node_id": self.bed_node_id,
            "cog_offset_mm": list(self.spec.toolhead.cog_offset_mm),
            "nozzle_offset_mm": list(self.spec.toolhead.nozzle_offset_mm),
            "mount_material": self.spec.toolhead.mount_material,
            "internal_stiffness_n_per_mm": self.spec.toolhead.internal_stiffness_n_per_mm,
            "z_probe_type": self.spec.toolhead.z_probe_type,
            "toolhead_envelope_mm": list(self.spec.toolhead.envelope_mm),
            "toolhead_structure": {
                "body_construction": self.spec.toolhead.body_construction,
                "construction_mass_delta_g": 0.0,
                "carriage_mount_material": self.spec.toolhead.effective_carriage_mount_material,
                "body_material": self.spec.toolhead.effective_body_material,
                "hotend_mount_material": self.spec.toolhead.effective_hotend_mount_material,
                "carriage_plate_thickness_mm": round(
                    float(self.spec.toolhead.carriage_plate_thickness_mm), 2
                ),
                "body_wall_thickness_mm": round(
                    float(self.spec.toolhead.body_wall_thickness_mm), 2
                ),
                "hotend_mount_thickness_mm": round(
                    float(self.spec.toolhead.hotend_mount_thickness_mm), 2
                ),
                "thickness_mass_delta_g": 0.0,
                "hotend_mass_g": round(float(self.spec.toolhead.hotend.mass_g), 1),
                "toolhead_body_mass_g": round(float(self.spec.toolhead.total_mass_g), 1),
                "assembled_toolhead_mass_g": round(
                    float(self.spec.toolhead.total_mass_g + self.spec.toolhead.hotend.mass_g), 1
                ),
                "hotend_preset_id": self.spec.toolhead.hotend.preset_id,
                "hotend_mount_style": self.spec.toolhead.hotend.mount_style,
                "hotend_stiffness_n_per_mm": round(
                    float(self.spec.toolhead.hotend.structural_stiffness_n_per_mm), 2
                ),
                "hotend_overall_length_mm": round(
                    float(self.spec.toolhead.hotend.overall_length_mm), 2
                ),
                "stage_stiffnesses_n_per_mm": {
                    k: round(float(v), 2)
                    for k, v in self.toolhead_stage_stiffnesses_n_per_mm.items()
                },
            },
            "nodes": [
                {
                    "id": n.id,
                    "position_mm": (n.position_m * 1000.0).tolist(),
                    "fixed": n.fixed,
                    "lumped_mass_g": n.lumped_mass_kg * 1000.0,
                    "role": n.role,
                }
                for n in self.nodes
            ],
            "members": [
                {
                    "id": m.id,
                    "start_id": m.start_id,
                    "end_id": m.end_id,
                    "start_mm": (nmap[m.start_id].position_m * 1000.0).tolist(),
                    "end_mm": (nmap[m.end_id].position_m * 1000.0).tolist(),
                    "profile": m.profile.id,
                    "width_mm": m.profile.width_m * 1000.0,
                    "height_mm": m.profile.height_m * 1000.0,
                    "shape": m.profile.shape,
                    "material": m.material.id,
                    "color_hex": m.material.color_hex,
                    "role": m.role,
                    "mass_g": m.mass_kg(nmap) * 1000.0,
                }
                for m in self.members
            ],
            "mass_summary": {
                "frame_mass_kg": round(self.frame_mass_kg, 3),
                "total_mass_kg": round(self.total_mass_kg, 3),
                "x_moving_mass_g": round(self.x_moving_mass_kg * 1000.0, 1),
                "y_moving_mass_g": round(self.y_moving_mass_kg * 1000.0, 1),
                "x_beam_mass_g": round(self.x_beam_mass_kg * 1000.0, 1),
                "x_rail_mass_g": round(self.x_rail_mass_kg * 1000.0, 1),
                "x_backer_mass_g": round(self.x_backer_mass_kg * 1000.0, 1),
                "y_carriage_joints_mass_g": round(self.y_carriage_joints_mass_kg * 1000.0, 1),
                "toolhead_moving_mass_g": round(self.toolhead_moving_mass_kg * 1000.0, 1),
                "estimated_material_cost_usd": round(self.estimated_material_cost_usd, 2),
            },
        }


def _add_member(
    members: list[StructuralMember],
    mid: str,
    start_id: str,
    end_id: str,
    profile_id: str,
    material_id: str,
    role: str = "frame",
    is_truss: bool = False,
) -> None:
    members.append(
        StructuralMember(
            id=mid,
            start_id=start_id,
            end_id=end_id,
            profile=get_profile(profile_id),
            material=get_material(material_id),
            role=role,
            is_truss=is_truss,
        )
    )


def panel_mechanical_properties(
    pmat: MaterialSpec,
    thickness_m: float,
    bay_w_m: float,
    bay_h_m: float,
    mounting: str,
    bolted_to_gantry: bool = False,
) -> dict[str, float]:
    """Compute panel shear, flexural, Timoshenko buckling, Huth joint, and areal mass properties (SI).

    - ACM / Dibond (``acm_dibond``) uses a 3-layer Aluminum/LDPE/Aluminum sandwich model
      calibrated to 3A Composites DIBOND datasheets (0.3 mm Al skins + LDPE core) with
      Timoshenko core-shear knockdown on flexural rigidity.
    - Homogeneous solid sheets use isotropic thin-plate theory (``D = E t^3 / (12 (1-nu^2))``).
    - Applies Timoshenko shear-buckling / oil-canning knockdown ``psi_buckle in [0.22, 1.0]``.
    - Models perimeter bolted fastener shear stiffness via the Huth-Schwarmann (1986) single-shear
      bolted joint formula (M5 fasteners on ~85 mm pitch into 2020/3030 T-slot flange) with
      material-specific bearing thickness and clamping preload efficiency (solid metal friction-lock
      vs unpotted ACM soft LDPE core crush vs polymer thermal-clearance holes).
    """
    t = max(float(thickness_m), 1e-4)
    is_acm = pmat.id == "acm_dibond"
    is_polymer = pmat.youngs_modulus_pa < 10.0e9
    is_composite = pmat.id == "carbon_fiber_plate"

    b_min = max(min(float(bay_w_m), float(bay_h_m)), 0.04)
    b_max = max(max(float(bay_w_m), float(bay_h_m)), b_min)

    if is_acm:
        t_skin = min(0.0003, 0.25 * t)
        e_al = 70.0e9
        nu_al = 0.33
        g_al = 26.3e9
        rho_al = 2700.0
        t_core = max(t - 2.0 * t_skin, 0.0)
        g_core = 0.17e9
        rho_core = 920.0

        gt_0 = g_al * (2.0 * t_skin) + g_core * t_core
        d_bend = e_al * (t_skin * (t - t_skin) ** 2) / (2.0 * (1.0 - nu_al**2))
        s_core = g_core * ((t - t_skin) ** 2) / max(t_core, 1e-5) if t_core > 1e-6 else 1e12
        # Sandwich Timoshenko flexural rigidity including soft LDPE core shear deformation
        flexural_d = 1.0 / (
            1.0 / max(d_bend, 1e-9) + ((math.pi / b_min) ** 2) / max(s_core, 1e-9)
        )
        areal_mass = rho_al * (2.0 * t_skin) + rho_core * t_core
        # Unpotted sandwich bolts bear primarily on the single 0.3 mm Al skin against the extrusion
        t_bear = t_skin
        e_bear = e_al
    else:
        gt_0 = pmat.effective_shear_modulus_pa * t
        flexural_d = (pmat.youngs_modulus_pa * (t**3)) / (
            12.0 * (1.0 - pmat.poisson_ratio**2)
        )
        areal_mass = pmat.density_kg_m3 * t
        t_bear = t
        e_bear = pmat.youngs_modulus_pa

    k_s = 5.34 + 4.0 * ((b_min / b_max) ** 2)
    n_xy_cr = k_s * (math.pi**2) * flexural_d / (b_min**2)
    # Critical shear-buckling & thin-plate oil-canning transition (~40 kN/m -> 3-4 mm Al plate)
    n_xy_ref = 40_000.0
    psi_buckle = 0.22 + 0.78 * (n_xy_cr / (n_xy_cr + n_xy_ref))

    h_eff = max(float(bay_h_m), 0.04)
    w_eff = max(float(bay_w_m), 0.04)
    k_sheet = psi_buckle * gt_0 * (w_eff / h_eff)

    if mounting == "clip_enclosure":
        k_mount = 28_000.0
        eta_comp = 0.0
    elif mounting == "bolted_structural":
        # Huth-Schwarmann (1986) single-shear M5 bolted fastener flexibility
        d_bolt = 0.005
        e_bolt = 200.0e9
        t_ext = 0.0022  # 2020/3030 aluminum extrusion T-slot flange thickness
        e_ext = 69.0e9
        a_huth = 2.0 / 3.0
        b_huth = 3.0 if (not is_acm and not is_polymer and not is_composite) else 4.2
        # Clamp Huth bolt-bending lever thickness to ~1.5*t_ext so thick clamped plates (>= 4 mm Al)
        # saturate monotonically at the extrusion T-slot flange stiffness instead of rolling over.
        t_lever = min(t_bear, 1.5 * t_ext)
        c_bolt = (
            (((t_lever + t_ext) / (2.0 * d_bolt)) ** a_huth)
            * b_huth
            * (
                1.0 / (t_bear * e_bear)
                + 1.0 / (t_ext * e_ext)
                + 1.0 / (2.0 * t_bear * e_bolt)
                + 1.0 / (2.0 * t_ext * e_bolt)
            )
        )
        k_bolt = 1.0 / max(c_bolt, 1e-12)

        # Clamping / preload efficiency (metal-on-metal friction lock vs soft LDPE core crush vs polymer)
        if is_acm:
            clamp_eff = 0.22
        elif is_polymer:
            clamp_eff = 0.30
        elif is_composite:
            clamp_eff = 0.80
        else:
            clamp_eff = 1.0

        # M5 bolts at ~85 mm pitch along loaded edge; top & bottom rows act in series (N_edge / 2)
        n_edge = max(3.0, round(w_eff / 0.085) + 1.0)
        n_eff = (n_edge * 0.5) * (1.35 if bolted_to_gantry else 1.0)
        k_mount = n_eff * k_bolt * clamp_eff
        eta_comp = 0.55 * clamp_eff * psi_buckle
    else:
        k_mount = 0.0
        eta_comp = 0.0

    if k_mount <= 0.0:
        k_panel_eff = 0.0
        gt_eff = 0.0
        eta_mount = 0.0
    else:
        k_panel_eff = (k_sheet * k_mount) / max(k_sheet + k_mount, 1e-9)
        gt_eff = k_panel_eff * (h_eff / w_eff)
        eta_mount = k_panel_eff / max(k_sheet, 1e-9)

    return {
        "gt_0_n_m": gt_0,
        "gt_eff_n_m": gt_eff,
        "k_sheet_n_m": k_sheet,
        "k_mount_n_m": k_mount,
        "k_panel_eff_n_m": k_panel_eff,
        "flexural_d_nm": flexural_d,
        "psi_buckle": psi_buckle,
        "eta_mount": eta_mount,
        "eta_comp": eta_comp,
        "t_bear_m": t_bear,
        "e_bear_pa": e_bear,
        "areal_mass_kg_m2": areal_mass,
    }


def _panel_shear_params(spec: PrinterSpec) -> tuple[bool, str, MaterialSpec, float]:
    """Return (active, mounting, panel_material, thickness_m)."""
    pmat = get_material(spec.frame.panel_material)
    mounting = spec.frame.panel_mounting
    thickness_m = spec.frame.panel_thickness_mm * 1e-3
    if not spec.frame.enclosed_panels or mounting == "none":
        return False, "none", pmat, thickness_m
    return True, mounting, pmat, thickness_m


def _make_panel_virtual_material(pmat: MaterialSpec) -> MaterialSpec:
    """Create a virtually massless, zero-cost stiffness copy for equivalent shear struts."""
    return pmat.model_copy(update={"density_kg_m3": 1e-3, "cost_per_kg_usd": 0.0})


def _make_bed_virtual_material(mat: MaterialSpec) -> MaterialSpec:
    """Create a virtually massless, zero-cost stiffness copy for bed structural mounts."""
    return mat.model_copy(update={"density_kg_m3": 1e-3, "cost_per_kg_usd": 0.0})


def _add_shear_panel_bay(
    members: list[StructuralMember],
    node_map: dict[str, StructuralNode],
    bay: str,
    bl_id: str,
    br_id: str,
    tl_id: str,
    tr_id: str,
    pmat: MaterialSpec,
    thickness_m: float,
    mounting: str,
    bolted_to_gantry: bool = False,
    wall_height_m: float | None = None,
) -> tuple[float, float, dict[str, float]]:
    """Model a rectangular shear panel via massless equivalent X-pair diagonal truss struts.

    Uses ``A_eq = (gt_eff * L_d^3) / (2 * E * b * h)`` with a virtually massless material
    copy (0.000 kg phantom mass) and lumps the exact physical panel mass
    ``b * h * areal_mass_kg_m2`` onto the corner nodes (0.35 bottom / 0.15 top on
    vertical walls matching Rayleigh base-anchored modal mass, 0.25 each on top lid).
    """
    b = float(np.linalg.norm(node_map[br_id].position_m - node_map[bl_id].position_m))
    h = float(np.linalg.norm(node_map[tl_id].position_m - node_map[bl_id].position_m))
    if b <= 1e-6 or h <= 1e-6:
        props = panel_mechanical_properties(
            pmat,
            thickness_m,
            max(b, 0.04),
            max(h, 0.04),
            mounting,
            bolted_to_gantry=bolted_to_gantry,
        )
        return 0.0, 0.0, props

    prop_h = wall_height_m if (wall_height_m is not None and wall_height_m > h) else h
    props = panel_mechanical_properties(
        pmat,
        thickness_m,
        b,
        prop_h,
        mounting,
        bolted_to_gantry=bolted_to_gantry,
    )
    virtual_mat = _make_panel_virtual_material(pmat)
    diag_len = math.hypot(b, h)
    a_eq = max(
        (props["gt_eff_n_m"] * (diag_len**3)) / (2.0 * virtual_mat.youngs_modulus_pa * b * h),
        1e-10,
    )

    prof = ProfileSpec.model_construct(
        id=f"panel_shear_{bay}",
        name=f"Equivalent Shear Panel ({bay}, {thickness_m * 1e3:.1f}mm {pmat.id})",
        shape="plate",
        width_m=max(thickness_m, 0.002),
        height_m=max(thickness_m, 0.004),
        area_m2=a_eq,
        iy_m4=1e-11,
        iz_m4=1e-11,
        j_m4=1e-11,
    )
    members.append(
        StructuralMember(
            id=f"panel_shear_{bay}_1",
            start_id=bl_id,
            end_id=tr_id,
            profile=prof,
            material=virtual_mat,
            role="brace",
            is_truss=True,
        )
    )
    members.append(
        StructuralMember(
            id=f"panel_shear_{bay}_2",
            start_id=br_id,
            end_id=tl_id,
            profile=prof,
            material=virtual_mat,
            role="brace",
            is_truss=True,
        )
    )

    actual_panel_mass = b * h * props["areal_mass_kg_m2"]
    if bay == "top":
        for nid in (bl_id, br_id, tl_id, tr_id):
            node_map[nid].lumped_mass_kg += 0.25 * actual_panel_mass
    else:
        for nid in (bl_id, br_id):
            node_map[nid].lumped_mass_kg += 0.35 * actual_panel_mass
        for nid in (tl_id, tr_id):
            node_map[nid].lumped_mass_kg += 0.15 * actual_panel_mass

    return a_eq, actual_panel_mass, props


def _member_local_axes(p0: np.ndarray, p1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return local (ey, ez) unit vectors matching ``fea.beam3d.beam_rotation_matrix``."""
    vec = p1 - p0
    length = float(np.linalg.norm(vec))
    ex = vec / max(length, 1e-12)
    global_z = np.array([0.0, 0.0, 1.0])
    ref = np.array([0.0, 1.0, 0.0]) if abs(float(np.dot(ex, global_z))) > 0.99 else global_z
    ey = np.cross(ref, ex)
    ey /= max(float(np.linalg.norm(ey)), 1e-12)
    ez = np.cross(ex, ey)
    ez /= max(float(np.linalg.norm(ez)), 1e-12)
    return ey, ez


def _x_rail_mount_normals(x_mount: str) -> list[np.ndarray]:
    """Return global outward normal vector(s) for X-gantry linear rail mounting."""
    if x_mount == "top":
        return [np.array([0.0, 0.0, 1.0])]
    if x_mount == "bottom":
        return [np.array([0.0, 0.0, -1.0])]
    if x_mount == "dual_top_bottom":
        return [np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])]
    return [np.array([0.0, -1.0, 0.0])]


def _y_rail_mount_normals(y_mount: str, side: str) -> list[np.ndarray]:
    """Return global outward normal vector(s) for CoreXY Y-rail mounting ('l' or 'r')."""
    if y_mount == "bottom":
        return [np.array([0.0, 0.0, -1.0])]
    if y_mount == "inner":
        return [np.array([1.0 if side == "l" else -1.0, 0.0, 0.0])]
    return [np.array([0.0, 0.0, 1.0])]


def _apply_bolted_linear_rail(
    member: StructuralMember,
    node_map: dict[str, StructuralNode],
    rail_profile: ProfileSpec,
    rail_material: MaterialSpec,
    mount_normals: list[np.ndarray],
    beam_shape: str | None = None,
    bolt_pitch_m: float = 0.025,
    l_eff_m: float | None = None,
    track_as_rail: bool = True,
) -> tuple[float, float, float]:
    """Model 1 or 2 steel linear rails (or backer bar) bolted to beam face(s) via Newmark partial interaction.

    Applies transformed-section parallel-axis bending and torsional boosts to ``member.profile``
    while keeping ``area_m2`` equal to the bare beam cross-section, and folds distributed rail/backer
    mass into ``member.material.density_kg_m3`` and ``member.rail_mass_kg`` / ``member.backer_mass_kg``
    with zero double-counting.
    """
    if not mount_normals:
        return 0.0, 0.0, 0.0

    p0 = node_map[member.start_id].position_m
    p1 = node_map[member.end_id].position_m
    length = float(np.linalg.norm(p1 - p0))
    if length <= 1e-9:
        return 0.0, 0.0, 0.0

    ey, ez = _member_local_axes(p0, p1)
    shape = beam_shape or member.profile.shape

    # Newmark (1951) partial-interaction composite shear-lag factor gamma from M3 bolt pitch & substrate
    if member.profile.id.startswith("cf_"):
        sub_kind = "carbon_tube"
    elif shape == "solid_bar":
        sub_kind = "solid_bar"
    else:
        sub_kind = "extrusion"

    from printsim.fea.components import newmark_rail_composite_gamma

    eta_rail = newmark_rail_composite_gamma(
        e_rail_pa=rail_material.youngs_modulus_pa,
        a_rail_m2=rail_profile.area_m2,
        e_beam_pa=member.material.youngs_modulus_pa,
        a_beam_m2=member.profile.area_m2,
        bolt_pitch_m=bolt_pitch_m,
        l_eff_m=l_eff_m if l_eff_m is not None else length,
        substrate_kind=sub_kind,
    )

    n_e = rail_material.youngs_modulus_pa / max(member.material.youngs_modulus_pa, 1e6)
    n_g = rail_material.effective_shear_modulus_pa / max(
        member.material.effective_shear_modulus_pa, 1e6
    )

    i_local_norm = rail_profile.iy_m4
    i_local_tang = rail_profile.iz_m4
    j_local = rail_profile.j_m4
    a_ext = member.profile.area_m2
    n_a_rail = n_e * rail_profile.area_m2
    is_symmetric_dual = len(mount_normals) == 2

    d_iy = 0.0
    d_iz = 0.0
    d_j = 0.0
    for n_mount in mount_normals:
        n_norm = float(np.linalg.norm(n_mount))
        n_unit = n_mount / n_norm if n_norm > 1e-12 else n_mount
        cy = float(np.dot(n_unit, ey)) ** 2
        cz = float(np.dot(n_unit, ez)) ** 2
        beam_half = 0.5 * (cy * member.profile.width_m + cz * member.profile.height_m)
        d_c = beam_half + 0.5 * rail_profile.height_m

        if is_symmetric_dual:
            i_par = eta_rail * n_a_rail * (d_c**2)
        else:
            i_par = eta_rail * ((a_ext * n_a_rail) / max(a_ext + n_a_rail, 1e-12)) * (d_c**2)

        d_iz += cy * (n_e * i_local_norm + i_par) + cz * (n_e * i_local_tang)
        d_iy += cz * (n_e * i_local_norm + i_par) + cy * (n_e * i_local_tang)
        d_j += n_g * j_local

    member.profile = member.profile.model_copy(
        update={
            "iy_m4": member.profile.iy_m4 + d_iy,
            "iz_m4": member.profile.iz_m4 + d_iz,
            "j_m4": member.profile.j_m4 + d_j,
        }
    )

    m_rail_member = (
        len(mount_normals) * rail_profile.area_m2 * length * rail_material.density_kg_m3
    )
    rho_delta = m_rail_member / max(a_ext * length, 1e-12)
    member.material = member.material.model_copy(
        update={"density_kg_m3": member.material.density_kg_m3 + rho_delta}
    )
    if track_as_rail:
        member.rail_mass_kg += m_rail_member
    else:
        member.backer_mass_kg += m_rail_member
        member.backer_material_id = rail_material.id
    return d_iy, d_iz, d_j


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
    if z_rail_id == "none":
        return
    old_rho = member.material.density_kg_m3
    old_rail_mass = member.rail_mass_kg
    z_prof = get_profile(z_rail_id)
    if z_rail_id.startswith("steel_rod"):
        _apply_smooth_rod_guideway(member, node_map, z_prof, rail_material)
    else:
        _apply_bolted_linear_rail(
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


def _make_x_backer_profile(
    x_beam_prof: ProfileSpec, x_rail_prof: ProfileSpec, backer_thickness_mm: float
) -> ProfileSpec:
    """Create a rectangular bar profile for an opposite-face X-gantry thermal backer."""
    w = min(x_beam_prof.width_m, max(x_rail_prof.width_m, 0.012))
    h = max(backer_thickness_mm * 1e-3, 0.001)
    area = w * h
    iy = w * (h**3) / 12.0
    iz = h * (w**3) / 12.0
    a, b = max(w, h), min(w, h)
    j_val = a * (b**3) * (1.0 / 3.0 - 0.21 * (b / a) * (1.0 - (b**4) / (12.0 * a**4)))
    return ProfileSpec.model_construct(
        id="x_rail_backer_bar",
        name=f"Thermal Rail Backer ({w*1e3:.0f}x{backer_thickness_mm:.1f}mm)",
        shape="solid_bar",
        width_m=w,
        height_m=h,
        area_m2=area,
        iy_m4=iy,
        iz_m4=iz,
        j_m4=j_val,
    )


def _apply_corner_joint_compliance(
    members: list[StructuralMember],
    spec: PrinterSpec,
    col_length_m: float,
) -> None:
    """Apply semi-rigid corner joint end-fixity knockdown to unpaneled vertical frame columns."""
    jtype = spec.frame.corner_joint_type
    bolted_panels = spec.frame.enclosed_panels and spec.frame.panel_mounting == "bolted_structural"
    if jtype == "rigid" or bolted_panels:
        return

    from printsim.fea.components import corner_joint_stiffness

    cj = corner_joint_stiffness(spec.frame.profile, jtype)
    bp = get_profile(spec.frame.profile)
    # Any un-railed horizontal frame beam carries the bare frame profile (or bonded core if already applied)
    unrailed = next(
        (m for m in members if m.role == "frame" and m.rail_mass_kg == 0.0 and m.id.startswith(("base_", "top_"))),
        None,
    )
    i_ext_y = unrailed.profile.iy_m4 if unrailed is not None else bp.iy_m4
    i_ext_z = unrailed.profile.iz_m4 if unrailed is not None else bp.iz_m4
    fill_spec = EXTRUSION_FILLINGS.get(spec.frame.extrusion_filling, EXTRUSION_FILLINGS["none"])
    joint_fill_boost = 1.0
    if spec.frame.extrusion_filling != "none" and bool(fill_spec.get("bonded", False)):
        e_fill = float(fill_spec.get("youngs_modulus_pa", 0.0))
        # Bonded cast mineral/epoxy core backs up hollow T-slot corner flanges against bolt-bearing ovalization
        joint_fill_boost = 1.0 + 0.45 * min(1.0, e_fill / 25.0e9)
    kr_in = cj["k_r_inplane_nm_rad"] * joint_fill_boost
    kr_tor = cj["k_r_torsion_nm_rad"] * joint_fill_boost
    if spec.frame.bracing == "corner_gussets":
        kr_in *= 1.50
        kr_tor *= 1.40
    l_col = max(col_length_m, 0.15)
    # Bolted steel Z rails stop ~10% short of each corner joint; Euler-Bernoulli bending moment
    # M(x) peaks at the corner ends (~88% of portal span flexibility resides in the un-railed end zones).
    w_end = 0.88

    for member in members:
        is_corexy_col = member.id.startswith("col_")
        is_cart_col = member.id.startswith("z_col_")
        if not (is_corexy_col or is_cart_col):
            continue

        iy_span = member.profile.iy_m4
        iz_span = member.profile.iz_m4
        if is_corexy_col and member.rail_mass_kg > 0.0:
            iy_span = 1.0 / (w_end / max(i_ext_y, 1e-18) + (1.0 - w_end) / max(member.profile.iy_m4, 1e-18))
            iz_span = 1.0 / (w_end / max(i_ext_z, 1e-18) + (1.0 - w_end) / max(member.profile.iz_m4, 1e-18))

        e_pa = member.material.youngs_modulus_pa
        g_pa = member.material.effective_shear_modulus_pa
        eta_iy = 1.0 / (1.0 + (6.0 * e_pa * iy_span) / (l_col * kr_in))
        eta_iz = 1.0 / (1.0 + (6.0 * e_pa * iz_span) / (l_col * kr_in))
        eta_j = 1.0 / (1.0 + (2.0 * g_pa * member.profile.j_m4) / (l_col * kr_tor))

        if is_cart_col:
            # Bedslinger H-frame uprights bolt flat to base frame with tee plates
            eta_iy = 1.0 - 0.06 * (1.0 - eta_iy)
            eta_iz = 1.0 - 0.06 * (1.0 - eta_iz)
            eta_j = 1.0 - 0.06 * (1.0 - eta_j)

        member.profile = member.profile.model_copy(
            update={
                "iy_m4": iy_span * eta_iy,
                "iz_m4": iz_span * eta_iz,
                "j_m4": member.profile.j_m4 * eta_j,
            }
        )


def _effective_toolhead_mass_kg(spec: PrinterSpec) -> float:
    """Return pure user-entered assembled toolhead mass (kg): toolhead.total_mass_g + hotend.mass_g."""
    return max((float(spec.toolhead.total_mass_g) + float(spec.toolhead.hotend.mass_g)) * 1e-3, 0.03)


def compute_toolhead_link_props(
    spec: PrinterSpec,
    l_mount_m: float,
    l_stem_m: float,
) -> dict[str, float]:
    """Compute decoupled 3-stage series stiffness properties for carriage plate, body, and hotend.

    Stage 1 (``carriage_mount_material`` + rail carriage + Z-probe): rail Hertzian roll/pitch
    compliance in series with carriage backplate stiffness.
    Stage 2 (``body_material`` + ``internal_stiffness_n_per_mm``): toolhead housing flexure.
    Stage 3 (``hotend_mount_material`` + ``hotend.mount_style`` + ``hotend.structural_stiffness_n_per_mm``):
    hotend clamp in series with heatbreak/hotend neck cantilever stiffness.
    """
    probe_cfg = Z_PROBE_CATALOG.get(spec.toolhead.z_probe_type, Z_PROBE_CATALOG["static_eddy"])
    probe_comp = float(probe_cfg["compliance_factor"])
    chamber_c = spec.frame.chamber_temp_c

    car_mat = get_material(spec.toolhead.effective_carriage_mount_material)
    body_mat = get_material(spec.toolhead.effective_body_material)
    hm_mat = get_material(spec.toolhead.effective_hotend_mount_material)

    e_car = car_mat.effective_modulus_pa(chamber_c) * probe_comp
    e_body = body_mat.effective_modulus_pa(chamber_c) * probe_comp
    e_hm = hm_mat.effective_modulus_pa(chamber_c) * probe_comp
    body_retention = body_mat.modulus_retention_at_temp(chamber_c)

    cog_x_mm, cog_y_mm, _ = spec.toolhead.cog_offset_mm
    noz_z_drop_m = max(abs(float(spec.toolhead.nozzle_offset_mm[2])) * 1e-3, 0.045)
    is_axisymmetric = (
        spec.architecture == "delta"
        and abs(float(cog_x_mm)) < 1e-4
        and abs(float(cog_y_mm)) < 1e-4
    )

    # Stage 1: Linear rail carriage Hertzian stiffness + carriage mount backplate
    if spec.architecture != "delta":
        from printsim.fea.components import hertzian_rail_carriage_stiffness

        car = hertzian_rail_carriage_stiffness(
            spec.gantry.x_rail_profile,
            preload=spec.gantry.rail_preload,
            block_type=spec.gantry.rail_block_type,
            rail_quality=spec.gantry.rail_quality,
        )
        car_mult = 2.0 if spec.gantry.effective_dual_x_rails else 1.0
        k_rx_car = car["k_rx_nm_rad"] * car_mult
        k_ry_car = car["k_ry_nm_rad"] * car_mult
        k_rz_car = car["k_rz_nm_rad"] * car_mult
        k_rad_car = 0.5 * (car["k_yy_n_m"] + car["k_zz_n_m"]) * car_mult
    else:
        k_rx_car = k_ry_car = k_rz_car = 2500.0
        k_rad_car = 2.0e8

    r_car = float(spec.toolhead.carriage_plate_thickness_mm) / 5.0
    car_ratio = max(e_car / 2.1e9, 0.04)
    k_plate_t = 2.4e6 * car_ratio * r_car
    k_plate_rot_y = 1160.0 * car_ratio * (r_car**2.4)
    k_plate_rot_z = k_plate_rot_y if is_axisymmetric else 4500.0 * car_ratio * (r_car**2.4)

    k_car_t = 1.0 / (1.0 / max(k_rad_car, 1.0) + 1.0 / max(k_plate_t, 1.0))
    k_car_rot_y = 1.0 / (1.0 / max(k_rx_car, 1.0) + 1.0 / max(k_plate_rot_y, 1.0))
    k_car_rot_z = 1.0 / (
        1.0 / max(0.5 * (k_ry_car + k_rz_car), 1.0) + 1.0 / max(k_plate_rot_z, 1.0)
    )
    k_car_rot_tor = 1.0 / (1.0 / max(k_rz_car, 1.0) + 1.0 / max(k_plate_rot_z, 1.0))
    k_r_car_eff = 3.0 / (1.0 / k_car_rot_y + 2.0 / k_car_rot_z)

    k_rail_carriage_n_m = 1.0 / (
        1.0 / max(k_rad_car, 1.0) + (noz_z_drop_m**2) / max(k_rx_car, 1.0)
    )
    k_backplate_only_n_m = 1.0 / (
        1.0 / max(k_plate_t, 1.0) + (noz_z_drop_m**2) / max(k_plate_rot_y, 1.0)
    )
    k_carriage_plate_n_m = 1.0 / (
        1.0 / max(k_car_t, 1.0) + (noz_z_drop_m**2) / max(k_car_rot_y, 1.0)
    )

    # Stage 2: Toolhead body housing (body_construction architecture OR legacy internal_stiffness)
    r_body = float(spec.toolhead.body_wall_thickness_mm) / 4.0
    body_ratio = max(e_body / 2.1e9, 0.04)
    bc = spec.toolhead.body_construction
    if bc is not None and bc in TOOLHEAD_CONSTRUCTION_CATALOG:
        cfg = TOOLHEAD_CONSTRUCTION_CATALOG[bc]
        k_body_n_m = (
            (float(cfg["body_stiffness_base_n_per_mm"]) * 1e3)
            * (body_ratio**0.72)
            * (r_body**1.35)
        )
        torsion_mult = float(cfg["torsion_mult"])
    else:
        k_raw = max(float(spec.toolhead.internal_stiffness_n_per_mm), 10.0)
        k_int_scaled = 350.0e3 * (
            (k_raw / 350.0) ** 1.90 if k_raw < 350.0 else (k_raw / 350.0)
        )
        k_body_n_m = k_int_scaled * (body_ratio**0.55) * (r_body**1.35)
        torsion_mult = 1.0
    k_body_rot_y = k_body_n_m * (0.091**2)
    k_body_rot_z = k_body_rot_y if is_axisymmetric else (3.3 * k_body_rot_y)

    # Combine Stage 1 & Stage 2 cleanly in series once (zero double-knockdown)
    k_stage12_t = 1.0 / (1.0 / max(k_car_t, 1.0) + 1.0 / max(k_body_n_m, 1.0))
    k_stage12_rot_y = 1.0 / (1.0 / max(k_car_rot_y, 1.0) + 1.0 / max(k_body_rot_y, 1.0))
    k_stage12_rot_z = 1.0 / (1.0 / max(k_car_rot_z, 1.0) + 1.0 / max(k_body_rot_z, 1.0))
    k_stage12_rot_tor = 1.0 / (1.0 / max(k_car_rot_tor, 1.0) + 1.0 / max(k_body_rot_z, 1.0))

    # Zero-cog fallback spring must stay strictly stiffer than infinitesimal cog > 0 beam
    k_plate_t_0 = 5000.0 * k_body_n_m * (car_ratio**0.35)
    k_t_spring = (
        (10.0 * k_rad_car * k_plate_t_0) / max(10.0 * k_rad_car + k_plate_t_0, 1e-9)
    ) * probe_comp
    k_r_spring = 1.35 * max(k_stage12_rot_y, k_stage12_rot_z)

    e_mount_eff = math.sqrt(max(e_car, 1.0e5) * max(e_body, 1.0e5))
    nu_mount = 0.5 * (car_mat.poisson_ratio + body_mat.poisson_ratio)
    g_mount = e_mount_eff / (2.0 * (1.0 + nu_mount))

    l_m = max(float(l_mount_m), 1e-4)
    ei_y_raw = 62.0 * (k_body_n_m / 450.0e3)
    ei_z_raw = ei_y_raw if is_axisymmetric else (2.60 * ei_y_raw)
    gj_raw = ei_y_raw * (0.65 if is_axisymmetric else 0.85) * torsion_mult

    ei_y_eff = ei_y_raw * k_stage12_rot_y / max(k_stage12_rot_y + ei_y_raw / l_m, 1e-9)
    ei_z_eff = ei_z_raw * k_stage12_rot_z / max(k_stage12_rot_z + ei_z_raw / l_m, 1e-9)
    gj_eff = gj_raw * k_stage12_rot_tor / max(k_stage12_rot_tor + gj_raw / l_m, 1e-9)

    iy_mount = ei_y_eff / max(e_mount_eff, 1e5)
    iz_mount = ei_z_eff / max(e_mount_eff, 1e5)
    j_mount = gj_eff / max(g_mount, 1e5)
    area_mount = min(2.2e-4, 5.0e8 * l_m / max(e_mount_eff, 1e6))

    # Stage 3: Hotend clamp compliance + physical hotend structural stiffness
    mount_style = spec.toolhead.hotend.mount_style
    clamp_base_map = {
        "groove_mount": 800.0e3,
        "rigid_4bolt_m3": 2200.0e3,
        "dragon_4bolt_m25": 3200.0e3,
        "chube_6bolt_m3_18pcd": 6500.0e3,
    }
    r_hm = float(spec.toolhead.hotend_mount_thickness_mm) / 8.0
    hm_ratio = max(e_hm / 2.1e9, 0.04)
    k_hotend_clamp_n_m = clamp_base_map.get(mount_style, 3200.0e3) * (hm_ratio**0.45) * (r_hm**1.8)

    k_hotend_unit_n_m = float(spec.toolhead.hotend.structural_stiffness_n_per_mm) * 1e3
    if spec.toolhead.hotend.preset_id is None:
        hb_mat_custom = get_material(spec.toolhead.hotend.heatbreak_material)
        hb_wall = max(float(spec.toolhead.hotend.heatbreak_wall_mm), 0.10)
        hb_scale = ((hb_mat_custom.youngs_modulus_pa / 110.0e9) ** 0.35) * (
            (hb_wall / 0.25) ** 0.30
        )
        k_hotend_unit_n_m *= float(np.clip(hb_scale, 0.55, 1.85))

    k_hotend_total_n_m = 1.0 / (
        1.0 / max(k_hotend_clamp_n_m, 1.0) + 1.0 / max(k_hotend_unit_n_m, 1.0)
    )

    hb_mat = get_material(spec.toolhead.hotend.heatbreak_material)
    e_stem = hb_mat.youngs_modulus_pa
    l_s = max(float(l_stem_m), 1e-4)
    k_stem_eff = k_hotend_total_n_m
    i_stem = float(np.clip(k_stem_eff * (l_s**3) / (3.0 * e_stem), 1e-15, 1e-4))
    j_stem = max(1.3 * i_stem, 1e-15)
    area_stem = min(1.2e-4, 5.0e8 * l_s / max(e_stem, 1e6))

    k_combined_tip_n_m = 1.0 / (
        1.0 / max(k_carriage_plate_n_m, 1.0)
        + 1.0 / max(k_body_n_m, 1.0)
        + 1.0 / max(k_hotend_total_n_m, 1.0)
    )

    return {
        "e_mount_eff": e_mount_eff,
        "g_mount": g_mount,
        "k_int_n_m": k_body_n_m,
        "k_rot_th": k_body_rot_y,
        "k_t_spring": k_t_spring,
        "k_r_spring": k_r_spring,
        "area_mount": area_mount,
        "iy_mount": iy_mount,
        "iz_mount": iz_mount,
        "j_mount": j_mount,
        "e_stem": e_stem,
        "k_stem_eff": k_stem_eff,
        "area_stem": area_stem,
        "i_stem": i_stem,
        "j_stem": j_stem,
        "k_rail_carriage_n_per_mm": k_rail_carriage_n_m * 1e-3,
        "k_backplate_only_n_per_mm": k_backplate_only_n_m * 1e-3,
        "k_carriage_plate_n_per_mm": k_carriage_plate_n_m * 1e-3,
        "k_body_n_per_mm": k_body_n_m * 1e-3,
        "k_hotend_clamp_n_per_mm": k_hotend_clamp_n_m * 1e-3,
        "k_hotend_total_n_per_mm": k_hotend_total_n_m * 1e-3,
        "k_combined_toolhead_n_per_mm": k_combined_tip_n_m * 1e-3,
    }


def _attach_toolhead_cog_and_nozzle(
    nodes: list[StructuralNode],
    members: list[StructuralMember],
    spec: PrinterSpec,
    base_node_id: str = "toolhead",
    fastener_springs: list[FastenerSpring] | None = None,
) -> dict[str, float]:
    """Attach explicit 6-DOF toolhead_cog and nozzle_tip nodes via carriage mount & hotend stem."""
    nmap = {n.id: n for n in nodes}
    if base_node_id not in nmap:
        return {}

    base_node = nmap[base_node_id]
    base_pos = base_node.position_m
    toolhead_kg = _effective_toolhead_mass_kg(spec)
    hotend_kg = float(spec.toolhead.hotend.mass_g) * 1e-3
    probe_cfg = Z_PROBE_CATALOG.get(spec.toolhead.z_probe_type, Z_PROBE_CATALOG["static_eddy"])
    cog_y_shift_m = float(probe_cfg["cog_y_shift_mm"]) * 1e-3

    # Linear carriage block stays on X-beam centerline; Rayleigh-equivalent lower cantilevered hotend mass (~28% of hotend mass) sits at nozzle tip
    nozzle_lumped_kg = max(0.010, min(0.28 * hotend_kg, max(toolhead_kg - 0.06, 0.015)))
    base_node.lumped_mass_kg = 0.04
    cog_lumped_kg = max(toolhead_kg - 0.04 - nozzle_lumped_kg, 0.02)

    cog_offset = np.array(spec.toolhead.cog_offset_mm, dtype=float) * 1e-3
    nozzle_offset = np.array(spec.toolhead.nozzle_offset_mm, dtype=float) * 1e-3

    if spec.architecture != "delta":
        x_mount = spec.gantry.effective_x_rail_mount
        if x_mount == "top":
            for off in (cog_offset, nozzle_offset):
                if off[1] < 0.0:
                    off[1] = min(0.0, off[1] + 0.006)
                elif off[1] > 0.0:
                    off[1] = max(0.0, off[1] - 0.006)
                if abs(off[2]) > 1e-12:
                    off[2] -= 0.006
        elif x_mount == "bottom":
            for off in (cog_offset, nozzle_offset):
                if off[1] < 0.0:
                    off[1] = min(0.0, off[1] + 0.005)
                elif off[1] > 0.0:
                    off[1] = max(0.0, off[1] - 0.005)
            if cog_offset[2] < 0.0:
                cog_offset[2] = min(0.0, cog_offset[2] + 0.012)
            if nozzle_offset[2] < 0.0:
                nozzle_offset[2] = min(-0.010, nozzle_offset[2] + 0.012)

    if abs(cog_y_shift_m) > 0.0:
        cog_offset[1] += cog_y_shift_m
        nozzle_offset[1] += cog_y_shift_m

    cog_norm = float(np.linalg.norm(cog_offset))
    is_zero_cog = cog_norm < 1e-4
    cog_pos = base_pos.copy() if is_zero_cog else (base_pos + cog_offset)

    nozzle_pos = base_pos + nozzle_offset
    stem_vec = nozzle_pos - cog_pos
    stem_len = float(np.linalg.norm(stem_vec))
    is_zero_stem = stem_len < 1e-4
    if is_zero_stem:
        nozzle_pos = cog_pos.copy()

    nodes.append(
        StructuralNode(
            id="toolhead_cog",
            position_m=cog_pos,
            lumped_mass_kg=cog_lumped_kg,
            role="cog",
        )
    )
    nodes.append(
        StructuralNode(
            id="nozzle_tip",
            position_m=nozzle_pos,
            lumped_mass_kg=nozzle_lumped_kg,
            role="nozzle",
        )
    )

    l_mount_m = 0.0 if is_zero_cog else float(np.linalg.norm(cog_pos - base_pos))
    l_stem_m = 0.0 if is_zero_stem else stem_len
    props = compute_toolhead_link_props(spec, l_mount_m=l_mount_m, l_stem_m=l_stem_m)

    base_mount_mat = get_material(spec.toolhead.effective_carriage_mount_material)
    mount_mat = base_mount_mat.model_copy(
        update={
            "youngs_modulus_pa": props["e_mount_eff"],
            "shear_modulus_pa": props["g_mount"],
            "density_kg_m3": 1e-3,
            "cost_per_kg_usd": 0.0,
        }
    )

    if is_zero_cog and fastener_springs is not None:
        fastener_springs.append(
            FastenerSpring(
                id="toolhead_cog_spring",
                bay="toolhead",
                shell_node_id=base_node_id,
                frame_node_id="toolhead_cog",
                k_shear_n_m=props["k_t_spring"],
                k_normal_n_m=props["k_t_spring"],
                k_rot_nm_rad=props["k_r_spring"],
                normal_vec=(0.0, 0.0, 1.0),
                offset_m=(0.0, 0.0, 0.0),
            )
        )
    else:
        mount_prof = ProfileSpec.model_construct(
            id="toolhead_mount_profile",
            name=(
                f"Toolhead Carriage+Body ({spec.toolhead.effective_carriage_mount_material}/"
                f"{spec.toolhead.effective_body_material})"
            ),
            shape="solid_bar",
            width_m=0.026,
            height_m=0.018,
            area_m2=props["area_mount"],
            iy_m4=props["iy_mount"],
            iz_m4=props["iz_mount"],
            j_m4=props["j_mount"],
        )
        members.append(
            StructuralMember(
                id="toolhead_mount",
                start_id=base_node_id,
                end_id="toolhead_cog",
                profile=mount_prof,
                material=mount_mat,
                role="toolhead",
            )
        )

    if is_zero_stem and fastener_springs is not None:
        fastener_springs.append(
            FastenerSpring(
                id="nozzle_stem_spring",
                bay="toolhead",
                shell_node_id="toolhead_cog",
                frame_node_id="nozzle_tip",
                k_shear_n_m=props["k_stem_eff"],
                k_normal_n_m=props["k_stem_eff"],
                k_rot_nm_rad=max(props["k_rot_th"] * 1.5, 500.0),
                normal_vec=(0.0, 0.0, 1.0),
                offset_m=(0.0, 0.0, 0.0),
            )
        )
    else:
        hb_mat = get_material(spec.toolhead.hotend.heatbreak_material)
        stem_mat = hb_mat.model_copy(
            update={
                "youngs_modulus_pa": props["e_stem"],
                "density_kg_m3": 1e-3,
                "cost_per_kg_usd": 0.0,
            }
        )
        stem_prof = ProfileSpec.model_construct(
            id="nozzle_stem_profile",
            name=f"Hotend Nozzle Stem ({spec.toolhead.hotend.mount_style})",
            shape="rod",
            width_m=0.012,
            height_m=0.012,
            area_m2=props["area_stem"],
            iy_m4=props["i_stem"],
            iz_m4=props["i_stem"],
            j_m4=props["j_stem"],
        )
        members.append(
            StructuralMember(
                id="nozzle_stem",
                start_id="toolhead_cog",
                end_id="nozzle_tip",
                profile=stem_prof,
                material=stem_mat,
                role="toolhead",
            )
        )

    return {
        "rail_carriage_n_per_mm": props["k_rail_carriage_n_per_mm"],
        "backplate_only_n_per_mm": props["k_backplate_only_n_per_mm"],
        "carriage_mount_n_per_mm": props["k_carriage_plate_n_per_mm"],
        "body_n_per_mm": props["k_body_n_per_mm"],
        "hotend_clamp_n_per_mm": props["k_hotend_clamp_n_per_mm"],
        "hotend_total_n_per_mm": props["k_hotend_total_n_per_mm"],
        "combined_toolhead_n_per_mm": props["k_combined_toolhead_n_per_mm"],
    }


def _apply_extrusion_filling(model: AssemblyModel) -> None:
    """Apply stationary extrusion cavity filling mass, bonded stiffness, and structural damping."""
    fill_id = model.spec.frame.extrusion_filling
    fill_spec = EXTRUSION_FILLINGS.get(fill_id, EXTRUSION_FILLINGS["none"])
    bkt_id = model.spec.gantry.bracket_material
    bkt_info = GANTRY_BRACKET_CATALOG.get(bkt_id, GANTRY_BRACKET_CATALOG["abs_printed"])

    nmap = model.node_map
    total_fill_mass_kg = 0.0
    filled_count = 0

    if fill_id != "none":
        rho_fill = float(fill_spec["density_kg_m3"])
        e_fill = float(fill_spec["youngs_modulus_pa"])
        is_bonded = bool(fill_spec["bonded"])
        g_fill = e_fill / (2.0 * (1.0 + 0.25))  # mineral/grout Poisson ~0.25

        for member in model.members:
            if member.profile.shape != "extrusion" or member.role not in ("frame", "rail", "tower"):
                continue

            raw_prof = PROFILE_CATALOG.get(member.profile.id, member.profile)
            w = raw_prof.width_m
            h = raw_prof.height_m
            a_ext = raw_prof.area_m2
            a_void = max(w * h - a_ext, 0.0)
            a_cav = 0.58 * a_void
            i_cav_y = 0.52 * max((w * (h**3)) / 12.0 - raw_prof.iy_m4, 0.0)
            i_cav_z = 0.52 * max((h * (w**3)) / 12.0 - raw_prof.iz_m4, 0.0)

            length = member.length_m(nmap)
            m_fill = rho_fill * a_cav * length
            member.fill_mass_kg = m_fill
            total_fill_mass_kg += m_fill
            filled_count += 1

            base_rho = member.material.density_kg_m3
            member.material = member.material.model_copy(
                update={"density_kg_m3": base_rho + rho_fill * (a_cav / max(a_ext, 1e-12))}
            )

            if is_bonded:
                n_e = e_fill / max(member.material.youngs_modulus_pa, 1e6)
                n_g = g_fill / max(member.material.effective_shear_modulus_pa, 1e6)
                member.profile = member.profile.model_copy(
                    update={
                        "iy_m4": member.profile.iy_m4 + n_e * i_cav_y,
                        "iz_m4": member.profile.iz_m4 + n_e * i_cav_z,
                        "j_m4": member.profile.j_m4 + n_g * 0.7 * (i_cav_y + i_cav_z),
                    }
                )

    model.extrusion_fill_mass_kg = total_fill_mass_kg
    fill_boost = float(fill_spec["damping_boost"]) if (fill_id == "none" or filled_count > 0) else 0.0
    bracket_damping = -0.003 if bkt_info["cnc_machined"] else 0.004
    from printsim.fea.components import iwan_joint_damping_delta

    joint_damping = iwan_joint_damping_delta(model.spec.frame.corner_joint_type)
    model.frame_damping_ratio = max(
        0.025, min(0.12, 0.039 + joint_damping + fill_boost + bracket_damping)
    )


def _attach_bed_mount_compliance(
    nodes: list[StructuralNode],
    fastener_springs: list[FastenerSpring],
    spec: PrinterSpec,
    bed_center_m: np.ndarray,
) -> None:
    """Create intermediate ``bed_frame`` node and 6-DOF ``bed_mount_spring`` to ``bed_surface``."""
    bed_kg = spec.bed.mass_g * 1e-3
    bed_frame_kg = 0.0
    bed_surface_kg = bed_kg

    bed_frame_pos = bed_center_m.copy()
    bed_frame_pos[2] -= 0.014
    nodes.append(
        StructuralNode("bed_frame", bed_frame_pos, lumped_mass_kg=bed_frame_kg, role="bed")
    )
    nodes.append(
        StructuralNode(
            "bed_surface", bed_center_m.copy(), lumped_mass_kg=bed_surface_kg, role="bed"
        )
    )

    cfg = BED_MOUNT_CATALOG.get(spec.bed.mount_type, BED_MOUNT_CATALOG["kinematic_3pt"])
    n_pts = int(cfg["mount_points"])
    k_axial_n_m = float(cfg["k_axial_n_per_mm"]) * 1e3
    k_lat_n_m = float(cfg["k_lateral_n_per_mm"]) * 1e3
    bed_x_m = float(spec.bed.size_mm[0]) * 1e-3
    bed_y_m = float(spec.bed.size_mm[1]) * 1e-3
    r_mount = 0.42 * min(bed_x_m, bed_y_m)
    k_rot_nm_rad = max(0.5 * n_pts * k_axial_n_m * (r_mount**2), 50.0)

    fastener_springs.append(
        FastenerSpring(
            id="bed_mount_spring",
            bay="bed_mount",
            shell_node_id="bed_frame",
            frame_node_id="bed_surface",
            k_shear_n_m=n_pts * k_lat_n_m,
            k_normal_n_m=n_pts * k_axial_n_m,
            k_rot_nm_rad=k_rot_nm_rad,
            normal_vec=(0.0, 0.0, 1.0),
            offset_m=(0.0, 0.0, -0.014),
        )
    )


def _apply_feet_grounding(model: AssemblyModel) -> None:
    """Replace rigid fixed feet with compliant benchtop isolator springs when configured."""
    feet_type = model.spec.frame.feet_type
    if feet_type == "rigid_bolted":
        return

    f_cfg = FRAME_FEET_CATALOG.get(feet_type, FRAME_FEET_CATALOG["rigid_bolted"])
    k_normal_n_m = float(f_cfg["k_axial_n_per_mm"]) * 1e3
    k_shear_n_m = float(f_cfg["k_lateral_n_per_mm"]) * 1e3
    k_rot_nm_rad = max(k_normal_n_m * (0.020**2), 40.0)

    grounded_any = False
    foot_nodes: list[StructuralNode] = []
    for node in list(model.nodes):
        if node.fixed and node.role in ("foot", "frame", "tower"):
            node.fixed = False
            foot_nodes.append(node)
            gnd_pos = node.position_m.copy()
            gnd_pos[2] -= 0.015
            gnd_id = f"gnd_{node.id}"
            model.nodes.append(
                StructuralNode(gnd_id, gnd_pos, fixed=True, role="ground")
            )
            model.fastener_springs.append(
                FastenerSpring(
                    id=f"feet_spring_{node.id}",
                    bay="feet",
                    shell_node_id=gnd_id,
                    frame_node_id=node.id,
                    k_shear_n_m=k_shear_n_m,
                    k_normal_n_m=k_normal_n_m,
                    k_rot_nm_rad=k_rot_nm_rad,
                    normal_vec=(0.0, 0.0, 1.0),
                    offset_m=(0.0, 0.0, 0.0),
                )
            )
            grounded_any = True

    base_mass_kg = float(f_cfg.get("base_mass_kg", 0.0))
    if base_mass_kg > 0.0 and foot_nodes:
        per_foot_kg = base_mass_kg / len(foot_nodes)
        for fn in foot_nodes:
            fn.lumped_mass_kg += per_foot_kg

    if grounded_any and model.spec.architecture == "cartesian_bedslinger":
        for node in model.nodes:
            if node.fixed and node.role == "mount":
                node.fixed = False


def _apply_mount_and_feet_damping(model: AssemblyModel) -> None:
    """Apply damping increments from frame feet isolators and bed mounting hardware."""
    f_cfg = FRAME_FEET_CATALOG.get(model.spec.frame.feet_type, FRAME_FEET_CATALOG["rigid_bolted"])
    b_cfg = BED_MOUNT_CATALOG.get(model.spec.bed.mount_type, BED_MOUNT_CATALOG["kinematic_3pt"])
    damping_delta = float(f_cfg["damping_delta"]) + float(b_cfg["damping_delta"]) * 0.25
    model.frame_damping_ratio = max(0.025, min(0.13, model.frame_damping_ratio + damping_delta))


def build_assembly(
    spec: PrinterSpec,
    toolhead_rel_pos: tuple[float, float, float] = (0.5, 0.5, 0.5),
    panel_mesh_n: int = 2,
) -> AssemblyModel:
    """Construct a 3D FEA-ready structural assembly from a PrinterSpec.

    Parameters
    ----------
    spec:
        Declarative PrinterSpec.
    toolhead_rel_pos:
        Normalized (u, v, w) in [0, 1]^3 specifying where the toolhead is positioned
        inside the build volume for structural stiffness evaluation. Defaults to center.
    panel_mesh_n:
        Subdivision grid density (n x n MITC4 shells per CoreXY panel bay). Defaults to 2
        for fast interactive solves; use 6 for deep offline verification.
    """
    mesh_n = max(2, int(panel_mesh_n))
    if spec.frame.custom_nodes and spec.frame.custom_members:
        model = _build_custom_assembly(spec)
        model.panel_mesh_n = mesh_n
        return model

    if spec.architecture == "corexy":
        model = _build_corexy(spec, toolhead_rel_pos, panel_mesh_n=mesh_n)
    elif spec.architecture == "cartesian_bedslinger":
        model = _build_cartesian_bedslinger(spec, toolhead_rel_pos)
    elif spec.architecture == "delta":
        model = _build_delta(spec, toolhead_rel_pos)
    else:
        raise ValueError(f"Unsupported architecture: {spec.architecture}")

    model.panel_mesh_n = mesh_n

    # Append any supplemental custom nodes/members defined alongside a preset frame
    nmap = model.node_map
    for cn in spec.frame.custom_nodes:
        if cn.id not in nmap:
            node = StructuralNode(
                id=cn.id,
                position_m=np.array(cn.position_mm, dtype=float) * 1e-3,
                fixed=cn.fixed,
                lumped_mass_kg=cn.lumped_mass_g * 1e-3,
                role=cn.role,
            )
            model.nodes.append(node)
            nmap[cn.id] = node

    for cm in spec.frame.custom_members:
        if cm.start_node in nmap and cm.end_node in nmap:
            _add_member(
                model.members,
                cm.id,
                cm.start_node,
                cm.end_node,
                cm.profile,
                cm.material,
                role=cm.role,
            )

    _apply_feet_grounding(model)
    _apply_extrusion_filling(model)
    _apply_mount_and_feet_damping(model)
    _finalize_mass_and_cost(model)
    return model


def _build_custom_assembly(spec: PrinterSpec) -> AssemblyModel:
    nodes = [
        StructuralNode(
            id=n.id,
            position_m=np.array(n.position_mm, dtype=float) * 1e-3,
            fixed=n.fixed,
            lumped_mass_kg=n.lumped_mass_g * 1e-3,
            role=n.role,
        )
        for n in spec.frame.custom_nodes
    ]
    members: list[StructuralMember] = []
    for m in spec.frame.custom_members:
        _add_member(members, m.id, m.start_node, m.end_node, m.profile, m.material, role=m.role)

    toolhead_id = next((n.id for n in nodes if n.role == "toolhead"), nodes[-1].id)
    w, d, h = spec.frame.width_mm * 1e-3, spec.frame.depth_mm * 1e-3, spec.frame.height_mm * 1e-3
    bx, by, bz = np.array(spec.kinematics.build_volume_mm) * 1e-3
    bkt_info = GANTRY_BRACKET_CATALOG.get(
        spec.gantry.bracket_material, GANTRY_BRACKET_CATALOG["abs_printed"]
    )
    bkt_delta_kg = float(bkt_info["carriage_mass_delta_g"]) * 1e-3
    th_mass_kg = _effective_toolhead_mass_kg(spec)
    model = AssemblyModel(
        spec=spec,
        nodes=nodes,
        members=members,
        toolhead_node_id=toolhead_id,
        bed_center_m=np.array([w * 0.5, d * 0.5, 0.06]),
        build_origin_m=np.array([(w - bx) * 0.5, (d - by) * 0.5, 0.07]),
        x_moving_mass_kg=th_mass_kg,
        y_moving_mass_kg=th_mass_kg + (spec.gantry.carriage_mass_g * 1e-3) + bkt_delta_kg,
        y_carriage_joints_mass_kg=(spec.gantry.carriage_mass_g * 1e-3) + bkt_delta_kg,
        toolhead_moving_mass_kg=th_mass_kg,
    )
    _apply_feet_grounding(model)
    _apply_extrusion_filling(model)
    _apply_mount_and_feet_damping(model)
    _finalize_mass_and_cost(model)
    return model


def _split_member_at_node(
    members: list[StructuralMember],
    member_idx: int,
    split_node_id: str,
    frac: float,
) -> None:
    """Split ``members[member_idx]`` into two collinear segments at ``split_node_id``."""
    orig = members[member_idx]
    f_clamped = max(1e-4, min(1.0 - 1e-4, float(frac)))
    seg_a = StructuralMember(
        id=orig.id,
        start_id=orig.start_id,
        end_id=split_node_id,
        profile=orig.profile,
        material=orig.material,
        role=orig.role,
        is_truss=orig.is_truss,
        fill_mass_kg=orig.fill_mass_kg * f_clamped,
        rail_mass_kg=orig.rail_mass_kg * f_clamped,
        backer_mass_kg=orig.backer_mass_kg * f_clamped,
        backer_material_id=orig.backer_material_id,
    )
    suffix = 1
    existing_ids = {m.id for m in members}
    seg_b_id = f"{orig.id}__s{suffix}"
    while seg_b_id in existing_ids:
        suffix += 1
        seg_b_id = f"{orig.id}__s{suffix}"
    seg_b = StructuralMember(
        id=seg_b_id,
        start_id=split_node_id,
        end_id=orig.end_id,
        profile=orig.profile,
        material=orig.material,
        role=orig.role,
        is_truss=orig.is_truss,
        fill_mass_kg=orig.fill_mass_kg * (1.0 - f_clamped),
        rail_mass_kg=orig.rail_mass_kg * (1.0 - f_clamped),
        backer_mass_kg=orig.backer_mass_kg * (1.0 - f_clamped),
        backer_material_id=orig.backer_material_id,
    )
    members[member_idx] = seg_a
    members.insert(member_idx + 1, seg_b)


def _find_or_split_frame_node(
    nodes: list[StructuralNode],
    nmap: dict[str, StructuralNode],
    members: list[StructuralMember],
    p_target: np.ndarray,
    snap_tol_m: float = 5e-4,
    allow_gantry: bool = True,
) -> str:
    """Locate an existing structural frame/rail node near ``p_target`` or split the spanning beam."""
    excluded_roles = {"panel", "toolhead", "cog", "nozzle", "bed"}
    if not allow_gantry:
        excluded_roles.add("gantry")
    best_id: str | None = None
    best_dist = snap_tol_m
    for n in nodes:
        if n.role in excluded_roles:
            continue
        dist = float(np.linalg.norm(n.position_m - p_target))
        if dist < best_dist:
            best_dist = dist
            best_id = n.id
    if best_id is not None:
        return best_id

    for idx, m in enumerate(members):
        if m.role not in ("frame", "rail", "tower"):
            continue
        p0 = nmap[m.start_id].position_m
        p1 = nmap[m.end_id].position_m
        vec = p1 - p0
        length = float(np.linalg.norm(vec))
        if length < 1e-6:
            continue
        u = vec / length
        proj = float(np.dot(p_target - p0, u))
        if snap_tol_m < proj < length - snap_tol_m:
            perp = float(np.linalg.norm((p_target - p0) - proj * u))
            if perp < snap_tol_m:
                split_id = f"fsplit_{len(nodes)}"
                while split_id in nmap:
                    split_id = f"fsplit_{len(nodes)}_{idx}"
                node = StructuralNode(
                    id=split_id,
                    position_m=(p0 + proj * u).copy(),
                    role="frame",
                )
                nodes.append(node)
                nmap[split_id] = node
                _split_member_at_node(members, idx, split_id, proj / length)
                return split_id

    # Fallback: closest frame/rail node
    non_panel = [n for n in nodes if n.role not in excluded_roles]
    return min(non_panel, key=lambda n: float(np.linalg.norm(n.position_m - p_target))).id


def _add_corexy_shell_panels(
    spec: PrinterSpec,
    nodes: list[StructuralNode],
    nmap: dict[str, StructuralNode],
    members: list[StructuralMember],
    pmat: MaterialSpec,
    panel_t_m: float,
    mounting: str,
    bolted_to_gantry: bool,
    panel_mesh_n: int,
) -> tuple[list[ShellPanelElement], list[FastenerSpring], float, dict[str, float] | None]:
    """Generate MITC4 shell grids + Huth CBUSH fastener springs for CoreXY enclosure bays."""
    h = spec.frame.height_mm * 1e-3
    frame_prof = get_profile(spec.frame.profile)
    d_c = 0.5 * frame_prof.width_m + 0.5 * panel_t_m

    is_acm = pmat.id == "acm_dibond"
    if is_acm:
        t_skin = min(0.0003, 0.25 * panel_t_m)
        t_core = max(panel_t_m - 2.0 * t_skin, 0.0)
        e_al = 70.0e9
        nu_al = 0.33
        g_core = 0.17e9
        e_core = 2.0 * (1.0 + 0.42) * g_core
        e_mem_base = e_al * (2.0 * t_skin) + e_core * t_core
        d_bend_base: float | None = e_al * (t_skin * (panel_t_m - t_skin) ** 2) / (
            2.0 * (1.0 - nu_al**2)
        )
        g_trans_base: float | None = max(g_core * t_core, 1e3)
        panel_mat_shell = pmat.model_copy(update={"poisson_ratio": nu_al})
    else:
        e_mem_base = pmat.youngs_modulus_pa * panel_t_m
        d_bend_base = None
        g_trans_base = None
        panel_mat_shell = pmat

    if bolted_to_gantry:
        bays: tuple[tuple[str, str, str, str, str, tuple[float, float, float], float | None], ...] = (
            ("rear_low", "b_rl", "b_rr", "g_rl", "g_rr", (0.0, 1.0, 0.0), h),
            ("rear_top", "g_rl", "g_rr", "t_rl", "t_rr", (0.0, 1.0, 0.0), h),
            ("left_low", "b_fl", "b_rl", "g_fl", "g_rl", (-1.0, 0.0, 0.0), h),
            ("left_top", "g_fl", "g_rl", "t_fl", "t_rl", (-1.0, 0.0, 0.0), h),
            ("right_low", "b_fr", "b_rr", "g_fr", "g_rr", (1.0, 0.0, 0.0), h),
            ("right_top", "g_fr", "g_rr", "t_fr", "t_rr", (1.0, 0.0, 0.0), h),
            ("top", "t_fl", "t_fr", "t_rl", "t_rr", (0.0, 0.0, 1.0), None),
        )
    else:
        bays = (
            ("rear", "b_rl", "b_rr", "t_rl", "t_rr", (0.0, 1.0, 0.0), None),
            ("left", "b_fl", "b_rl", "t_fl", "t_rl", (-1.0, 0.0, 0.0), None),
            ("right", "b_fr", "b_rr", "t_fr", "t_rr", (1.0, 0.0, 0.0), None),
            ("top", "t_fl", "t_fr", "t_rl", "t_rr", (0.0, 0.0, 1.0), None),
        )

    clamp_eff = (
        0.22
        if is_acm
        else (
            0.30
            if pmat.youngs_modulus_pa < 10.0e9
            else (0.80 if pmat.id == "carbon_fiber_plate" else 1.0)
        )
    )
    mesh_n = max(2, int(panel_mesh_n))

    shell_elements: list[ShellPanelElement] = []
    fastener_springs: list[FastenerSpring] = []
    total_panel_mass_kg = 0.0
    rep_panel_props: dict[str, float] | None = None
    # Share shell nodes across *_low / *_top seam so bolted-to-gantry side & rear walls
    # behave as continuous one-piece sheets (preserving C1 out-of-plane bending continuity).
    seam_nodes: dict[tuple[str, int], str] = {}
    seam_springs: dict[tuple[str, int], FastenerSpring] = {}

    for bay, bl_id, br_id, tl_id, tr_id, n_tup, wall_h in bays:
        p00 = nmap[bl_id].position_m
        p10 = nmap[br_id].position_m
        p01 = nmap[tl_id].position_m
        p11 = nmap[tr_id].position_m
        bw = float(np.linalg.norm(p10 - p00))
        bh = float(np.linalg.norm(p01 - p00))
        if bw <= 1e-6 or bh <= 1e-6:
            continue

        is_split_wall = wall_h is not None
        wall_base = bay[:-4] if bay.endswith("_low") or bay.endswith("_top") else ""
        is_low_wall = bay.endswith("_low")
        is_top_wall = bay.endswith("_top")
        prop_h = wall_h if (wall_h is not None and wall_h > bh) else bh
        props = panel_mechanical_properties(
            pmat,
            panel_t_m,
            bw,
            prop_h,
            mounting,
            bolted_to_gantry=is_split_wall,
        )
        total_panel_mass_kg += bw * bh * props["areal_mass_kg_m2"]
        if rep_panel_props is None and bay in ("rear_low", "rear"):
            rep_panel_props = props

        e_mem_eff = e_mem_base * props["psi_buckle"]
        if mounting == "clip_enclosure":
            k_shear_per_m = 2.0 * 28_000.0 / max(bw, 0.04)
            k_normal_per_m = 2.5e5
            k_rot_per_m = 0.2
        else:
            n_edge_w = max(3.0, round(bw / 0.085) + 1.0)
            gantry_mult = 1.35 if is_split_wall else 1.0
            k_mount_edge = n_edge_w * (props["k_mount_n_m"] / (0.5 * n_edge_w * gantry_mult))
            k_shear_per_m = k_mount_edge / max(bw, 0.04)
            k_normal_per_m = max(1.0e6, 2.0 * k_shear_per_m)
            k_rot_per_m = 45.0 * clamp_eff

        nx_g = mesh_n
        ny_g = mesh_n if bh > 0.12 else max(1, mesh_n // 2)
        nvec = np.asarray(n_tup, dtype=float)
        off_nominal = d_c * nvec

        grid = np.empty((ny_g + 1, nx_g + 1), dtype=object)
        for j in range(ny_g + 1):
            eta = j / ny_g
            for i in range(nx_g + 1):
                if is_top_wall and j == 0 and (wall_base, i) in seam_nodes:
                    # Stitch top sub-bay seamlessly onto lower sub-bay's gantry-level shell nodes
                    sid = seam_nodes[(wall_base, i)]
                    grid[j, i] = sid
                    if i in (0, nx_g) and (wall_base, i) in seam_springs:
                        # Add upper vertical column tributary length to the shared corner bolt
                        ds_col_top = 0.5 * (bh / ny_g)
                        sp_existing = seam_springs[(wall_base, i)]
                        sp_existing.k_shear_n_m += k_shear_per_m * ds_col_top
                        sp_existing.k_normal_n_m += k_normal_per_m * ds_col_top
                        sp_existing.k_rot_nm_rad += k_rot_per_m * ds_col_top
                    continue

                xi = i / nx_g
                p_frame = (
                    (1.0 - xi) * (1.0 - eta) * p00
                    + xi * (1.0 - eta) * p10
                    + xi * eta * p11
                    + (1.0 - xi) * eta * p01
                )
                p_shell = p_frame + off_nominal
                sid = f"pan_{bay}_{j}_{i}"
                node = StructuralNode(id=sid, position_m=p_shell, role="panel")
                nodes.append(node)
                nmap[sid] = node
                grid[j, i] = sid
                if is_low_wall and j == ny_g:
                    seam_nodes[(wall_base, i)] = sid

                if i in (0, nx_g) or j in (0, ny_g):
                    fid = _find_or_split_frame_node(
                        nodes, nmap, members, p_frame, allow_gantry=bolted_to_gantry
                    )
                    ds = 0.0
                    if j in (0, ny_g):
                        ds += (bw / nx_g) * (0.5 if i in (0, nx_g) else 1.0)
                    if i in (0, nx_g):
                        ds += (bh / ny_g) * (0.5 if j in (0, ny_g) else 1.0)
                    actual_off = p_shell - nmap[fid].position_m
                    sp = FastenerSpring(
                        id=f"cbush_{bay}_{j}_{i}",
                        bay=bay,
                        shell_node_id=sid,
                        frame_node_id=fid,
                        k_shear_n_m=k_shear_per_m * ds,
                        k_normal_n_m=k_normal_per_m * ds,
                        k_rot_nm_rad=k_rot_per_m * ds,
                        normal_vec=n_tup,
                        offset_m=(
                            float(actual_off[0]),
                            float(actual_off[1]),
                            float(actual_off[2]),
                        ),
                    )
                    fastener_springs.append(sp)
                    if is_low_wall and j == ny_g:
                        seam_springs[(wall_base, i)] = sp

        for j in range(ny_g):
            for i in range(nx_g):
                n0 = str(grid[j, i])
                n1 = str(grid[j, i + 1])
                n2 = str(grid[j + 1, i + 1])
                n3 = str(grid[j + 1, i])
                c3 = np.array([nmap[nid].position_m for nid in (n0, n1, n2, n3)])
                if float(np.dot(np.cross(c3[2] - c3[0], c3[3] - c3[1]), nvec)) < 0.0:
                    nids = (n0, n3, n2, n1)
                else:
                    nids = (n0, n1, n2, n3)
                shell_elements.append(
                    ShellPanelElement(
                        id=f"shell_{bay}_{j}_{i}",
                        bay=bay,
                        node_ids=nids,
                        material=panel_mat_shell,
                        thickness_m=panel_t_m,
                        areal_mass_kg_m2=props["areal_mass_kg_m2"],
                        d_bend_nm=d_bend_base,
                        g_transverse_h_n_m=g_trans_base,
                        e_mem_h_n_m=e_mem_eff,
                        normal_vec=n_tup,
                    )
                )

    return shell_elements, fastener_springs, total_panel_mass_kg, rep_panel_props


def _build_corexy(
    spec: PrinterSpec,
    toolhead_rel_pos: tuple[float, float, float],
    panel_mesh_n: int = 2,
) -> AssemblyModel:
    w = spec.frame.width_mm * 1e-3
    d = spec.frame.depth_mm * 1e-3
    h = spec.frame.height_mm * 1e-3
    bx, by, bz = np.array(spec.kinematics.build_volume_mm) * 1e-3
    u, v, w_rel = np.clip(toolhead_rel_pos, 0.05, 0.95)

    xy_motor_kg = spec.kinematics.xy_stepper.motor_mass_g * 1e-3
    z_motor_kg = spec.kinematics.z_stepper.motor_mass_g * 1e-3
    bkt_info = GANTRY_BRACKET_CATALOG.get(
        spec.gantry.bracket_material, GANTRY_BRACKET_CATALOG["abs_printed"]
    )
    bkt_delta_kg = float(bkt_info["carriage_mass_delta_g"]) * 1e-3
    carriage_side_kg = (spec.gantry.carriage_mass_g * 0.5) * 1e-3 + 0.5 * bkt_delta_kg
    toolhead_kg = _effective_toolhead_mass_kg(spec)
    bed_kg = spec.bed.mass_g * 1e-3

    # Build origin centered in frame
    orig_x = (w - bx) * 0.5
    orig_y = (d - by) * 0.5
    orig_z = 0.08
    y_gantry = orig_y + v * by
    x_tool = orig_x + u * bx

    is_flying = spec.is_flying_gantry
    z_drive = spec.bed.z_drive
    if is_flying:
        z_bed = orig_z
        z_gantry_min = min(z_bed + 0.055, max(h - 0.08, 0.06))
        z_gantry_max = max(z_gantry_min + 0.02, h - 0.055)
        z_gantry = float(
            np.clip(
                z_bed + 0.055 + w_rel * max(bz - 0.04, 0.05),
                z_gantry_min,
                z_gantry_max,
            )
        )
    else:
        # Gantry plane near top of CoreXY frame
        z_gantry = h - 0.055
        z_min = min(0.08, 0.5 * z_gantry)
        z_max = max(z_min + 0.01, z_gantry - 0.045)
        z_stroke = min(bz, max(z_gantry - 0.14, 0.02))
        z_bed = float(np.clip(z_gantry - 0.045 - w_rel * z_stroke, z_min, z_max))

    is_awd = spec.gantry.drive_mode == "awd"
    front_motor_kg = (xy_motor_kg + 0.03) if is_awd else 0.0
    front_role = "mount" if is_awd else "frame"
    bed_center = np.array([w * 0.5, d * 0.5, z_bed])
    fastener_springs: list[FastenerSpring] = []

    z_mech_id = spec.bed.effective_z_leadscrew_type
    z_mech = Z_DRIVE_MECHANISM_CATALOG.get(z_mech_id, Z_DRIVE_MECHANISM_CATALOG["tr8x4_pom"])
    if str(z_mech["kind"]) == "belt":
        k_z_drive_col_n_m = float(z_mech["k_drive_n_per_um_per_corner"] or 0.18) * 1.0e6
    else:
        e_steel_z = 200.0e9
        l_screw_m = max(z_gantry if is_flying else z_bed, 0.05)
        k_shaft = e_steel_z * float(z_mech["root_area_m2"]) / l_screw_m
        k_nut = max(float(z_mech["k_nut_n_per_um"]) * 1.0e6, 1.0e5)
        k_thrust = max(float(z_mech["k_thrust_n_per_um"]) * 1.0e6, 1.0e5)
        k_z_drive_col_n_m = 1.0 / (1.0 / k_shaft + 1.0 / k_nut + 1.0 / k_thrust)

    # Place Z stepper motor masses on their physical drive nodes per Z architecture
    corner_z_kg = z_motor_kg if is_flying else 0.0
    front_z_kg = z_motor_kg if z_drive == "leadscrew_triple" else corner_z_kg
    nodes = [
        # Bottom 4 corner feet (fixed to table)
        StructuralNode("b_fl", np.array([0.0, 0.0, 0.0]), fixed=True, lumped_mass_kg=front_z_kg, role="foot"),
        StructuralNode("b_fr", np.array([w, 0.0, 0.0]), fixed=True, lumped_mass_kg=front_z_kg, role="foot"),
        StructuralNode("b_rl", np.array([0.0, d, 0.0]), fixed=True, lumped_mass_kg=corner_z_kg, role="foot"),
        StructuralNode("b_rr", np.array([w, d, 0.0]), fixed=True, lumped_mass_kg=corner_z_kg, role="foot"),
    ]
    if is_flying:
        # Voron 2.4 flying gantry: split vertical columns at col_g_* and couple separate
        # floating gantry corners g_* via 6-DOF Z-belt + MGN rail carriage + GE5C Z-joint springs.
        y_bed_f = d * 0.14
        y_bed_r = d * 0.86
        nodes.extend(
            [
                StructuralNode("col_g_fl", np.array([0.0, 0.0, z_gantry]), role="frame"),
                StructuralNode("col_g_fr", np.array([w, 0.0, z_gantry]), role="frame"),
                StructuralNode("col_g_rl", np.array([0.0, d, z_gantry]), role="frame"),
                StructuralNode("col_g_rr", np.array([w, d, z_gantry]), role="frame"),
                StructuralNode("g_fl", np.array([0.0, 0.0, z_gantry]), lumped_mass_kg=front_motor_kg, role=front_role if is_awd else "gantry"),
                StructuralNode("g_fr", np.array([w, 0.0, z_gantry]), lumped_mass_kg=front_motor_kg, role=front_role if is_awd else "gantry"),
                StructuralNode("g_rl", np.array([0.0, d, z_gantry]), lumped_mass_kg=xy_motor_kg, role="gantry"),
                StructuralNode("g_rr", np.array([w, d, z_gantry]), lumped_mass_kg=xy_motor_kg, role="gantry"),
                StructuralNode("b_bed_fl", np.array([0.0, y_bed_f, 0.0]), role="frame"),
                StructuralNode("b_bed_rl", np.array([0.0, y_bed_r, 0.0]), role="frame"),
                StructuralNode("b_bed_fr", np.array([w, y_bed_f, 0.0]), role="frame"),
                StructuralNode("b_bed_rr", np.array([w, y_bed_r, 0.0]), role="frame"),
            ]
        )
        for tag in ("fl", "fr", "rl", "rr"):
            fastener_springs.append(
                FastenerSpring(
                    id=f"z_joint_{tag}",
                    bay="z_joint",
                    shell_node_id=f"col_g_{tag}",
                    frame_node_id=f"g_{tag}",
                    k_shear_n_m=1.2e6,
                    k_normal_n_m=k_z_drive_col_n_m,
                    k_rot_nm_rad=65.0,
                    normal_vec=(0.0, 0.0, 1.0),
                    offset_m=(0.0, 0.0, 0.0),
                )
            )
    else:
        nodes.extend(
            [
                # Fixed-gantry elevation nodes directly on the 4 vertical columns
                StructuralNode("g_fl", np.array([0.0, 0.0, z_gantry]), lumped_mass_kg=front_motor_kg, role=front_role),
                StructuralNode("g_fr", np.array([w, 0.0, z_gantry]), lumped_mass_kg=front_motor_kg, role=front_role),
                StructuralNode("g_rl", np.array([0.0, d, z_gantry]), lumped_mass_kg=xy_motor_kg, role="mount"),
                StructuralNode("g_rr", np.array([w, d, z_gantry]), lumped_mass_kg=xy_motor_kg, role="mount"),
            ]
        )
    nodes.extend(
        [
            # Top 4 corners
            StructuralNode("t_fl", np.array([0.0, 0.0, h]), role="frame"),
            StructuralNode("t_fr", np.array([w, 0.0, h]), role="frame"),
            StructuralNode("t_rl", np.array([0.0, d, h]), role="frame"),
            StructuralNode("t_rr", np.array([w, d, h]), role="frame"),
            # Moving Y carriages along left/right Y rails
            StructuralNode("y_car_l", np.array([0.0, y_gantry, z_gantry]), lumped_mass_kg=carriage_side_kg, role="gantry"),
            StructuralNode("y_car_r", np.array([w, y_gantry, z_gantry]), lumped_mass_kg=carriage_side_kg, role="gantry"),
            # Toolhead node on the X gantry beam
            StructuralNode("toolhead", np.array([x_tool, y_gantry, z_gantry]), lumped_mass_kg=toolhead_kg, role="toolhead"),
        ]
    )
    _attach_bed_mount_compliance(nodes, fastener_springs, spec, bed_center)

    y_mid = d * 0.5
    if z_drive == "leadscrew_triple":
        nodes.extend(
            [
                StructuralNode("z_bed_fl", np.array([0.0, 0.0, z_bed]), role="frame"),
                StructuralNode("z_bed_fr", np.array([w, 0.0, z_bed]), role="frame"),
                StructuralNode("b_rear_mid", np.array([w * 0.5, d, 0.0]), lumped_mass_kg=z_motor_kg, role="frame"),
                StructuralNode("g_rear_mid", np.array([w * 0.5, d, z_gantry]), role="frame"),
                StructuralNode("z_bed_r", np.array([w * 0.5, d, z_bed]), role="bed"),
            ]
        )
    elif z_drive == "leadscrew_dual":
        nodes.extend(
            [
                StructuralNode("b_left_mid", np.array([0.0, y_mid, 0.0]), lumped_mass_kg=z_motor_kg, role="frame"),
                StructuralNode("b_right_mid", np.array([w, y_mid, 0.0]), lumped_mass_kg=z_motor_kg, role="frame"),
                StructuralNode("g_left_mid", np.array([0.0, y_mid, z_gantry]), role="frame"),
                StructuralNode("g_right_mid", np.array([w, y_mid, z_gantry]), role="frame"),
                StructuralNode("z_bed_l", np.array([0.0, y_mid, z_bed]), role="bed"),
                StructuralNode("z_bed_r", np.array([w, y_mid, z_bed]), role="bed"),
            ]
        )
    elif z_drive == "leadscrew_single":
        nodes.extend(
            [
                StructuralNode("b_rear_mid", np.array([w * 0.5, d, 0.0]), lumped_mass_kg=z_motor_kg, role="frame"),
                StructuralNode("g_rear_mid", np.array([w * 0.5, d, z_gantry]), role="frame"),
                StructuralNode("z_bed_r", np.array([w * 0.5, d, z_bed]), role="bed"),
            ]
        )

    fp = spec.frame.profile
    fm = spec.frame.material
    bp = spec.frame.brace_profile
    bm = spec.frame.brace_material
    members: list[StructuralMember] = []

    # Base perimeter (split only where Z-rail columns anchor)
    _add_member(members, "base_front", "b_fl", "b_fr", fp, fm)
    if z_drive in ("leadscrew_triple", "leadscrew_single"):
        _add_member(members, "base_rear_l", "b_rl", "b_rear_mid", fp, fm)
        _add_member(members, "base_rear_r", "b_rear_mid", "b_rr", fp, fm)
    else:
        _add_member(members, "base_rear", "b_rl", "b_rr", fp, fm)

    if z_drive == "leadscrew_dual":
        _add_member(members, "base_left_f", "b_fl", "b_left_mid", fp, fm)
        _add_member(members, "base_left_r", "b_left_mid", "b_rl", fp, fm)
        _add_member(members, "base_right_f", "b_fr", "b_right_mid", fp, fm)
        _add_member(members, "base_right_r", "b_right_mid", "b_rr", fp, fm)
    elif is_flying:
        _add_member(members, "base_left_f", "b_fl", "b_bed_fl", fp, fm)
        _add_member(members, "base_left_m", "b_bed_fl", "b_bed_rl", fp, fm)
        _add_member(members, "base_left_r", "b_bed_rl", "b_rl", fp, fm)
        _add_member(members, "base_right_f", "b_fr", "b_bed_fr", fp, fm)
        _add_member(members, "base_right_m", "b_bed_fr", "b_bed_rr", fp, fm)
        _add_member(members, "base_right_r", "b_bed_rr", "b_rr", fp, fm)
    else:
        _add_member(members, "base_left", "b_fl", "b_rl", fp, fm)
        _add_member(members, "base_right", "b_fr", "b_rr", fp, fm)

    # 4 vertical columns split at gantry height (and front columns split at bed height on triple-Z)
    for tag in ("fl", "fr", "rl", "rr"):
        col_mid = f"col_g_{tag}" if is_flying else f"g_{tag}"
        if z_drive == "leadscrew_triple" and tag in ("fl", "fr"):
            _add_member(members, f"col_{tag}_bot", f"b_{tag}", f"z_bed_{tag}", fp, fm)
            _add_member(members, f"col_{tag}_mid", f"z_bed_{tag}", col_mid, fp, fm)
        else:
            _add_member(members, f"col_{tag}_low", f"b_{tag}", col_mid, fp, fm)
        _add_member(members, f"col_{tag}_top", col_mid, f"t_{tag}", fp, fm)

    # Top perimeter
    _add_member(members, "top_front", "t_fl", "t_fr", fp, fm)
    _add_member(members, "top_rear", "t_rl", "t_rr", fp, fm)
    _add_member(members, "top_left", "t_fl", "t_rl", fp, fm)
    _add_member(members, "top_right", "t_fr", "t_rr", fp, fm)

    # Mid gantry frame rails (left/right Y rails + rear motor bar, open front door)
    if z_drive == "leadscrew_dual":
        if abs(y_gantry - y_mid) >= 2e-3:
            if y_gantry < y_mid:
                _add_member(members, "y_rail_l_f", "g_fl", "y_car_l", fp, fm, role="rail")
                _add_member(members, "y_rail_l_m", "y_car_l", "g_left_mid", fp, fm, role="rail")
                _add_member(members, "y_rail_l_r", "g_left_mid", "g_rl", fp, fm, role="rail")
                _add_member(members, "y_rail_r_f", "g_fr", "y_car_r", fp, fm, role="rail")
                _add_member(members, "y_rail_r_m", "y_car_r", "g_right_mid", fp, fm, role="rail")
                _add_member(members, "y_rail_r_r", "g_right_mid", "g_rr", fp, fm, role="rail")
            else:
                _add_member(members, "y_rail_l_f", "g_fl", "g_left_mid", fp, fm, role="rail")
                _add_member(members, "y_rail_l_m", "g_left_mid", "y_car_l", fp, fm, role="rail")
                _add_member(members, "y_rail_l_r", "y_car_l", "g_rl", fp, fm, role="rail")
                _add_member(members, "y_rail_r_f", "g_fr", "g_right_mid", fp, fm, role="rail")
                _add_member(members, "y_rail_r_m", "g_right_mid", "y_car_r", fp, fm, role="rail")
                _add_member(members, "y_rail_r_r", "y_car_r", "g_rr", fp, fm, role="rail")
        else:
            _add_member(members, "y_rail_l_f", "g_fl", "y_car_l", fp, fm, role="rail")
            _add_member(members, "y_rail_l_r", "y_car_l", "g_rl", fp, fm, role="rail")
            _add_member(members, "y_rail_r_f", "g_fr", "y_car_r", fp, fm, role="rail")
            _add_member(members, "y_rail_r_r", "y_car_r", "g_rr", fp, fm, role="rail")
            nmap_tmp = {n.id: n for n in nodes}
            off_l = nmap_tmp["g_left_mid"].position_m - nmap_tmp["y_car_l"].position_m
            off_r = nmap_tmp["g_right_mid"].position_m - nmap_tmp["y_car_r"].position_m
            fastener_springs.append(
                FastenerSpring(
                    id="z_rail_top_tie_l",
                    bay="frame",
                    shell_node_id="g_left_mid",
                    frame_node_id="y_car_l",
                    k_shear_n_m=1.0e9,
                    k_normal_n_m=1.0e9,
                    k_rot_nm_rad=1.0e7,
                    normal_vec=(0.0, 0.0, 1.0),
                    offset_m=(float(off_l[0]), float(off_l[1]), float(off_l[2])),
                )
            )
            fastener_springs.append(
                FastenerSpring(
                    id="z_rail_top_tie_r",
                    bay="frame",
                    shell_node_id="g_right_mid",
                    frame_node_id="y_car_r",
                    k_shear_n_m=1.0e9,
                    k_normal_n_m=1.0e9,
                    k_rot_nm_rad=1.0e7,
                    normal_vec=(0.0, 0.0, 1.0),
                    offset_m=(float(off_r[0]), float(off_r[1]), float(off_r[2])),
                )
            )
    else:
        _add_member(members, "y_rail_l_f", "g_fl", "y_car_l", fp, fm, role="rail")
        _add_member(members, "y_rail_l_r", "y_car_l", "g_rl", fp, fm, role="rail")
        _add_member(members, "y_rail_r_f", "g_fr", "y_car_r", fp, fm, role="rail")
        _add_member(members, "y_rail_r_r", "y_car_r", "g_rr", fp, fm, role="rail")

    if z_drive in ("leadscrew_triple", "leadscrew_single"):
        _add_member(members, "gantry_rear_bar_l", "g_rl", "g_rear_mid", fp, fm)
        _add_member(members, "gantry_rear_bar_r", "g_rear_mid", "g_rr", fp, fm)
    else:
        _add_member(members, "gantry_rear_bar", "g_rl", "g_rr", fp, fm)

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
            iy_m4=4.0e-8 + 1.2e-7 * (t_ratio**3),
            iz_m4=1.6e-7,
            j_m4=6.0e-8,
        )
        for tag in ("fl", "fr", "rl", "rr"):
            members.append(
                StructuralMember(
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
        nmap_pre = {n.id: n for n in nodes}

        def _series_z_arm_iy(iy_nom: float, start_nid: str) -> float:
            l_arm = max(
                float(np.linalg.norm(nmap_pre[start_nid].position_m - nmap_pre["bed_frame"].position_m)),
                0.05,
            )
            iy_bare = (1.40 if z_drive == "leadscrew_triple" else 1.0) * iy_nom
            k_arm_z = 3.0 * max(bed_vmat.youngs_modulus_pa, 1e6) * iy_bare / (l_arm**3)
            return iy_bare * k_z_drive_col_n_m / max(k_z_drive_col_n_m + k_arm_z, 1.0)

        if z_drive in ("leadscrew_triple", "leadscrew_single"):
            members.append(
                StructuralMember(
                    id="bed_rear_z_bot",
                    start_id="b_rear_mid",
                    end_id="z_bed_r",
                    profile=z_post_prof,
                    material=z_rail_vmat,
                    role="bed",
                )
            )
            members.append(
                StructuralMember(
                    id="bed_rear_z_top",
                    start_id="z_bed_r",
                    end_id="g_rear_mid",
                    profile=z_post_prof,
                    material=z_rail_vmat,
                    role="bed",
                )
            )

        if z_drive == "leadscrew_triple":
            for tag, z_nid in (("fl", "z_bed_fl"), ("fr", "z_bed_fr"), ("r", "z_bed_r")):
                bed_arm_prof = ProfileSpec.model_construct(
                    id="bed_trident_arm_profile",
                    name=f"3-Point Z Bed Spider + Plate ({t_mm:.1f}mm {spec.bed.plate_material})",
                    shape="solid_bar",
                    width_m=0.020,
                    height_m=max(0.012, 0.012 + t_mm * 1e-3),
                    area_m2=6.0e-4,
                    iy_m4=_series_z_arm_iy(iy_triple_arm, z_nid),
                    iz_m4=iz_triple_arm,
                    j_m4=j_triple_arm,
                )
                members.append(
                    StructuralMember(
                        id=f"bed_arm_{tag}",
                        start_id=z_nid,
                        end_id="bed_frame",
                        profile=bed_arm_prof,
                        material=bed_vmat,
                        role="bed",
                    )
                )
        elif z_drive == "leadscrew_dual":
            z_dual_rail_prof = ProfileSpec.model_construct(
                id="bed_dual_z_rail_profile",
                name="Dual-Z Side Linear Guide + Leadscrew",
                shape="solid_bar",
                width_m=0.012,
                height_m=0.012,
                area_m2=2.8e-5,
                iy_m4=0.22 * frame_prof.iy_m4,
                iz_m4=0.22 * frame_prof.iz_m4,
                j_m4=0.22 * frame_prof.j_m4,
            )
            for side, b_nid, z_nid, g_nid in (
                ("left", "b_left_mid", "z_bed_l", "g_left_mid"),
                ("right", "b_right_mid", "z_bed_r", "g_right_mid"),
            ):
                members.append(
                    StructuralMember(
                        id=f"bed_{side}_z_bot",
                        start_id=b_nid,
                        end_id=z_nid,
                        profile=z_dual_rail_prof,
                        material=z_rail_vmat,
                        role="bed",
                    )
                )
                members.append(
                    StructuralMember(
                        id=f"bed_{side}_z_top",
                        start_id=z_nid,
                        end_id=g_nid,
                        profile=z_dual_rail_prof,
                        material=z_rail_vmat,
                        role="bed",
                    )
                )
            for tag, z_nid in (("l", "z_bed_l"), ("r", "z_bed_r")):
                bed_dual_prof = ProfileSpec.model_construct(
                    id="bed_dual_arm_profile",
                    name=f"Dual-Z Bed Arm + Plate ({t_mm:.1f}mm {spec.bed.plate_material})",
                    shape="solid_bar",
                    width_m=0.020,
                    height_m=max(0.012, 0.012 + t_mm * 1e-3),
                    area_m2=5.0e-4,
                    iy_m4=_series_z_arm_iy(0.22 * iy_triple_arm, z_nid),
                    iz_m4=0.45 * iz_triple_arm,
                    j_m4=0.35 * j_triple_arm,
                )
                members.append(
                    StructuralMember(
                        id=f"bed_arm_{tag}",
                        start_id=z_nid,
                        end_id="bed_frame",
                        profile=bed_dual_prof,
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
                iy_m4=_series_z_arm_iy(0.22 * iy_triple_arm, "z_bed_r"),
                iz_m4=0.45 * iz_triple_arm,
                j_m4=0.22 * j_triple_arm,
            )
            members.append(
                StructuralMember(
                    id="bed_arm_cantilever",
                    start_id="z_bed_r",
                    end_id="bed_frame",
                    profile=bed_cantilever_prof,
                    material=bed_vmat,
                    role="bed",
                )
            )

    # Moving X gantry beam (split at toolhead node)
    xp = spec.gantry.x_beam_profile
    xm = spec.gantry.x_beam_material
    _add_member(members, "x_gantry_l", "y_car_l", "toolhead", xp, xm, role="gantry")
    _add_member(members, "x_gantry_r", "toolhead", "y_car_r", xp, xm, role="gantry")

    # Bracing strategies (corner_gussets stiffens corner joints via _apply_corner_joint_compliance,
    # reserving full-face diagonal rods strictly for rear_cross and full_truss)
    bracing = spec.frame.bracing
    brace_g = "col_g_" if is_flying else "g_"
    if bracing == "rear_cross":
        _add_member(members, "brace_rear_1", "b_rl", "t_rr", bp, bm, role="brace", is_truss=True)
        _add_member(members, "brace_rear_2", "b_rr", "t_rl", bp, bm, role="brace", is_truss=True)
        _add_member(members, "brace_top_1", "t_fl", "t_rr", bp, bm, role="brace", is_truss=True)
    elif bracing == "full_truss":
        _add_member(members, "brace_rear_1", "b_rl", "t_rr", bp, bm, role="brace", is_truss=True)
        _add_member(members, "brace_rear_2", "b_rr", "t_rl", bp, bm, role="brace", is_truss=True)
        _add_member(members, "brace_left_1", "b_fl", f"{brace_g}rl", bp, bm, role="brace", is_truss=True)
        _add_member(members, "brace_left_2", "b_rl", f"{brace_g}fl", bp, bm, role="brace", is_truss=True)
        _add_member(members, "brace_right_1", "b_fr", f"{brace_g}rr", bp, bm, role="brace", is_truss=True)
        _add_member(members, "brace_right_2", "b_rr", f"{brace_g}fr", bp, bm, role="brace", is_truss=True)
        _add_member(members, "brace_top_1", "t_fl", "t_rr", bp, bm, role="brace", is_truss=True)
        _add_member(members, "brace_top_2", "t_fr", "t_rl", bp, bm, role="brace", is_truss=True)

    nmap = {n.id: n for n in nodes}

    # Composite bolted steel linear rails on Y extrusions and moving X gantry beam
    rail_mat = get_material(spec.gantry.rail_material)
    x_rail_prof = get_profile(spec.gantry.x_rail_profile)
    y_rail_prof = get_profile(spec.gantry.y_rail_profile)
    base_y_prof = get_profile(fp)
    base_x_prof = get_profile(xp)
    y_mount = spec.gantry.y_rail_mount
    x_mount = spec.gantry.effective_x_rail_mount
    x_normals = _x_rail_mount_normals(x_mount)
    bolt_pitch_m = spec.gantry.rail_bolt_pitch_mm * 1e-3
    member_by_id = {m.id: m for m in members}

    y_boost_iz = 0.0
    for m in members:
        if m.role == "rail" and m.id.startswith(("y_rail_l_", "y_rail_r_")):
            side = "l" if m.id.startswith("y_rail_l_") else "r"
            _, d_iz_y, _ = _apply_bolted_linear_rail(
                m,
                nmap,
                y_rail_prof,
                rail_mat,
                _y_rail_mount_normals(y_mount, side),
                beam_shape=base_y_prof.shape,
                bolt_pitch_m=bolt_pitch_m,
                l_eff_m=d,
            )
            y_boost_iz = d_iz_y

    x_boost_iy = 0.0
    x_boost_iz = 0.0
    for mid in ("x_gantry_l", "x_gantry_r"):
        d_iy_x, d_iz_x, _ = _apply_bolted_linear_rail(
            member_by_id[mid],
            nmap,
            x_rail_prof,
            rail_mat,
            x_normals,
            beam_shape=base_x_prof.shape,
            bolt_pitch_m=bolt_pitch_m,
            l_eff_m=w,
        )
        x_boost_iy += d_iy_x
        x_boost_iz += d_iz_x
    x_boost_iy *= 0.5
    x_boost_iz *= 0.5

    if spec.gantry.rail_backer_material != "none" and not spec.gantry.effective_dual_x_rails:
        backer_mat = get_material(spec.gantry.rail_backer_material)
        backer_prof = _make_x_backer_profile(
            base_x_prof, x_rail_prof, spec.gantry.rail_backer_thickness_mm
        )
        backer_normals = [-x_normals[0]]
        bk_iy_acc = 0.0
        bk_iz_acc = 0.0
        for mid in ("x_gantry_l", "x_gantry_r"):
            d_iy_b, d_iz_b, _ = _apply_bolted_linear_rail(
                member_by_id[mid],
                nmap,
                backer_prof,
                backer_mat,
                backer_normals,
                beam_shape=base_x_prof.shape,
                bolt_pitch_m=bolt_pitch_m,
                l_eff_m=w,
                track_as_rail=False,
            )
            bk_iy_acc += d_iy_b
            bk_iz_acc += d_iz_b
        x_boost_iy += 0.5 * bk_iy_acc
        x_boost_iz += 0.5 * bk_iz_acc

    x_rail_ei_y_boost_pct = 100.0 * x_boost_iz / max(base_x_prof.iz_m4, 1e-18)
    x_rail_ei_z_boost_pct = 100.0 * x_boost_iy / max(base_x_prof.iy_m4, 1e-18)
    y_rail_ei_x_boost_pct = 100.0 * y_boost_iz / max(base_y_prof.iz_m4, 1e-18)

    # Physical Z-axis linear rails or smooth rods along vertical Z guideways
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
                            _apply_bolted_linear_rail(
                                m_obj,
                                nmap,
                                z_rail_prof_d,
                                rail_mat,
                                [np.array([nx, 0.0, 0.0])],
                                beam_shape=frame_prof.shape,
                                bolt_pitch_m=float(spec.gantry.rail_bolt_pitch_mm) * 1e-3,
                            )
                            # Keep bending EI from composite column+rail; axial Z load passes primarily
                            # through the lower leadscrew span (bed_*_z_bot) since the MGN carriage
                            # slides along the upper rail guide (bed_*_z_top) with only drag/idler coupling.
                            m_obj.profile = m_obj.profile.model_copy(update={"area_m2": 2.8e-5})
                            _rebalance_virtual_bed_rail_mass(m_obj, nmap, old_rho, old_rail_mass)
                            if mid.endswith("_top"):
                                top_area = 6.0e-7
                                m_obj.material = m_obj.material.model_copy(
                                    update={"density_kg_m3": m_obj.material.density_kg_m3 * (2.8e-5 / top_area)}
                                )
                                m_obj.profile = m_obj.profile.model_copy(update={"area_m2": top_area})
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

    _apply_corner_joint_compliance(members, spec, col_length_m=h)

    # Enclosure / structural MITC4 shell panels + Huth CBUSH fastener springs across rear, side, and top bays
    panels_active, panel_mounting, pmat, panel_t_m = _panel_shear_params(spec)
    shell_elements: list[ShellPanelElement] = []
    total_panel_mass_kg = 0.0
    rep_panel_props: dict[str, float] | None = None
    if panels_active:
        bolted_to_gantry = spec.panels_bolted_to_gantry_effective
        shell_elements, panel_springs, total_panel_mass_kg, rep_panel_props = (
            _add_corexy_shell_panels(
                spec=spec,
                nodes=nodes,
                nmap=nmap,
                members=members,
                pmat=pmat,
                panel_t_m=panel_t_m,
                mounting=panel_mounting,
                bolted_to_gantry=bolted_to_gantry,
                panel_mesh_n=panel_mesh_n,
            )
        )
        fastener_springs.extend(panel_springs)

    # Calculate moving masses (strip composite rail & backer mass from gantry beam before adding x_rail_mass)
    x_beam_mass = sum(
        max(m.mass_kg(nmap) - m.rail_mass_kg - m.backer_mass_kg - m.fill_mass_kg, 0.0)
        for m in members
        if m.role == "gantry"
    )
    x_backer_mass = sum(m.backer_mass_kg for m in members if m.role == "gantry")
    x_rail_mass = x_rail_prof.area_m2 * w * rail_mat.density_kg_m3
    if spec.gantry.effective_dual_x_rails:
        x_rail_mass *= 2.0

    # In CoreXY, X moves toolhead; Y moves toolhead + X beam + X rail + backer + Y carriages + gantry bracket delta
    x_moving = toolhead_kg + 0.04  # plus belt segment
    y_moving = (
        toolhead_kg
        + x_beam_mass
        + x_rail_mass
        + x_backer_mass
        + spec.gantry.carriage_mass_g * 1e-3
        + bkt_delta_kg
    )

    stage_stiffnesses = _attach_toolhead_cog_and_nozzle(
        nodes, members, spec, base_node_id="toolhead", fastener_springs=fastener_springs
    )

    return AssemblyModel(
        spec=spec,
        nodes=nodes,
        members=members,
        shell_elements=shell_elements,
        fastener_springs=fastener_springs,
        panel_mesh_n=max(2, int(panel_mesh_n)),
        toolhead_node_id="toolhead",
        toolhead_cog_node_id="toolhead_cog",
        nozzle_node_id="nozzle_tip",
        bed_node_id="bed_surface",
        gantry_node_ids=["y_car_l", "toolhead", "y_car_r"],
        bed_center_m=np.array([w * 0.5, d * 0.5, z_bed]),
        build_origin_m=np.array([orig_x, orig_y, orig_z]),
        x_moving_mass_kg=x_moving,
        y_moving_mass_kg=y_moving,
        x_beam_mass_kg=x_beam_mass,
        x_rail_mass_kg=x_rail_mass,
        x_backer_mass_kg=x_backer_mass,
        y_carriage_joints_mass_kg=spec.gantry.carriage_mass_g * 1e-3 + bkt_delta_kg,
        toolhead_moving_mass_kg=toolhead_kg,
        panel_mass_kg=total_panel_mass_kg,
        x_rail_ei_y_boost_pct=x_rail_ei_y_boost_pct,
        x_rail_ei_z_boost_pct=x_rail_ei_z_boost_pct,
        y_rail_ei_x_boost_pct=y_rail_ei_x_boost_pct,
        total_rail_mass_kg=sum(m.rail_mass_kg for m in members),
        panel_rep_props=rep_panel_props,
        toolhead_stage_stiffnesses_n_per_mm=stage_stiffnesses,
    )


def _build_cartesian_bedslinger(
    spec: PrinterSpec, toolhead_rel_pos: tuple[float, float, float]
) -> AssemblyModel:
    w = spec.frame.width_mm * 1e-3
    d = spec.frame.depth_mm * 1e-3
    h = spec.frame.height_mm * 1e-3
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
    bkt_info = GANTRY_BRACKET_CATALOG.get(
        spec.gantry.bracket_material, GANTRY_BRACKET_CATALOG["abs_printed"]
    )
    bkt_delta_kg = float(bkt_info["carriage_mass_delta_g"]) * 1e-3
    carriage_side_kg = (spec.gantry.carriage_mass_g * 0.5) * 1e-3
    toolhead_kg = _effective_toolhead_mass_kg(spec)

    nodes = [
        # H-frame base feet
        StructuralNode("b_fl", np.array([0.0, 0.0, 0.0]), fixed=True, role="foot"),
        StructuralNode("b_fr", np.array([w, 0.0, 0.0]), fixed=True, role="foot"),
        StructuralNode("b_rl", np.array([0.0, d, 0.0]), fixed=True, role="foot"),
        StructuralNode("b_rr", np.array([w, d, 0.0]), fixed=True, role="foot"),
        # Portal base junctions (not ground-fixed; transfer portal loads via base rails to feet)
        StructuralNode("p_bl", np.array([0.0, y_portal, 0.0]), fixed=False, lumped_mass_kg=z_motor_kg, role="mount"),
        StructuralNode("p_br", np.array([w, y_portal, 0.0]), fixed=False, lumped_mass_kg=z_motor_kg, role="mount"),
        # Z-axis gantry height nodes on vertical uprights
        StructuralNode("z_l", np.array([0.0, y_portal, z_gantry]), lumped_mass_kg=carriage_side_kg + xy_motor_kg, role="gantry"),
        StructuralNode("z_r", np.array([w, y_portal, z_gantry]), lumped_mass_kg=carriage_side_kg, role="gantry"),
        # Portal top corners
        StructuralNode("p_tl", np.array([0.0, y_portal, h]), role="frame"),
        StructuralNode("p_tr", np.array([w, y_portal, h]), role="frame"),
        # Toolhead on X gantry
        StructuralNode("toolhead", np.array([x_tool, y_portal, z_gantry]), lumped_mass_kg=toolhead_kg, role="toolhead"),
    ]

    fp = spec.frame.profile
    fm = spec.frame.material
    bp = spec.frame.brace_profile
    bm = spec.frame.brace_material
    members: list[StructuralMember] = []

    # Base H-frame
    _add_member(members, "base_l_f", "b_fl", "p_bl", fp, fm)
    _add_member(members, "base_l_r", "p_bl", "b_rl", fp, fm)
    _add_member(members, "base_r_f", "b_fr", "p_br", fp, fm)
    _add_member(members, "base_r_r", "p_br", "b_rr", fp, fm)
    _add_member(members, "base_cross", "p_bl", "p_br", fp, fm)
    _add_member(members, "base_front", "b_fl", "b_fr", fp, fm)
    _add_member(members, "base_rear", "b_rl", "b_rr", fp, fm)

    # Vertical Z portal uprights + top bar
    _add_member(members, "z_col_l_low", "p_bl", "z_l", fp, fm)
    _add_member(members, "z_col_l_top", "z_l", "p_tl", fp, fm)
    _add_member(members, "z_col_r_low", "p_br", "z_r", fp, fm)
    _add_member(members, "z_col_r_top", "z_r", "p_tr", fp, fm)
    _add_member(members, "portal_top", "p_tl", "p_tr", fp, fm)

    # Moving X gantry beam
    xp = spec.gantry.x_beam_profile
    xm = spec.gantry.x_beam_material
    _add_member(members, "x_gantry_l", "z_l", "toolhead", xp, xm, role="gantry")
    _add_member(members, "x_gantry_r", "toolhead", "z_r", xp, xm, role="gantry")

    # Portal diagonal braces (corner_gussets only stiffens joints via _apply_corner_joint_compliance)
    if spec.frame.bracing in ("rear_cross", "full_truss"):
        _add_member(members, "brace_l_rear", "b_rl", "p_tl", bp, bm, role="brace", is_truss=True)
        _add_member(members, "brace_r_rear", "b_rr", "p_tr", bp, bm, role="brace", is_truss=True)
        _add_member(members, "brace_portal_x1", "p_bl", "p_tr", bp, bm, role="brace", is_truss=True)
    if spec.frame.bracing == "full_truss":
        _add_member(members, "brace_l_front", "b_fl", "p_tl", bp, bm, role="brace", is_truss=True)
        _add_member(members, "brace_r_front", "b_fr", "p_tr", bp, bm, role="brace", is_truss=True)
        _add_member(members, "brace_portal_x2", "p_br", "p_tl", bp, bm, role="brace", is_truss=True)

    nmap = {n.id: n for n in nodes}
    rail_mat = get_material(spec.gantry.rail_material)
    x_rail_prof = get_profile(spec.gantry.x_rail_profile)
    base_x_prof = get_profile(xp)
    x_normals = _x_rail_mount_normals(spec.gantry.effective_x_rail_mount)
    bolt_pitch_m = spec.gantry.rail_bolt_pitch_mm * 1e-3
    member_by_id = {m.id: m for m in members}
    x_boost_iy = 0.0
    x_boost_iz = 0.0
    for mid in ("x_gantry_l", "x_gantry_r"):
        d_iy_x, d_iz_x, _ = _apply_bolted_linear_rail(
            member_by_id[mid],
            nmap,
            x_rail_prof,
            rail_mat,
            x_normals,
            beam_shape=base_x_prof.shape,
            bolt_pitch_m=bolt_pitch_m,
            l_eff_m=w,
        )
        x_boost_iy += d_iy_x
        x_boost_iz += d_iz_x
    x_boost_iy *= 0.5
    x_boost_iz *= 0.5

    if spec.gantry.rail_backer_material != "none" and not spec.gantry.effective_dual_x_rails:
        backer_mat = get_material(spec.gantry.rail_backer_material)
        backer_prof = _make_x_backer_profile(
            base_x_prof, x_rail_prof, spec.gantry.rail_backer_thickness_mm
        )
        backer_normals = [-x_normals[0]]
        bk_iy_acc = 0.0
        bk_iz_acc = 0.0
        for mid in ("x_gantry_l", "x_gantry_r"):
            d_iy_b, d_iz_b, _ = _apply_bolted_linear_rail(
                member_by_id[mid],
                nmap,
                backer_prof,
                backer_mat,
                backer_normals,
                beam_shape=base_x_prof.shape,
                bolt_pitch_m=bolt_pitch_m,
                l_eff_m=w,
                track_as_rail=False,
            )
            bk_iy_acc += d_iy_b
            bk_iz_acc += d_iz_b
        x_boost_iy += 0.5 * bk_iy_acc
        x_boost_iz += 0.5 * bk_iz_acc

    x_rail_ei_y_boost_pct = 100.0 * x_boost_iz / max(base_x_prof.iz_m4, 1e-18)
    x_rail_ei_z_boost_pct = 100.0 * x_boost_iy / max(base_x_prof.iy_m4, 1e-18)

    z_rail_id = spec.bed.z_rail_profile
    if z_rail_id != "none":
        n_z = [np.array([0.0, -1.0, 0.0])]
        for side in ("l", "r"):
            for mid in (f"z_col_{side}_low", f"z_col_{side}_top"):
                if mid in member_by_id:
                    _apply_z_guideway(
                        member_by_id[mid], nmap, z_rail_id, rail_mat, n_z, bolt_pitch_m, h
                    )

    _apply_corner_joint_compliance(members, spec, col_length_m=h)

    # Bedslinger: X moves toolhead (+ bracket delta); Y moves bed + carriage (+ bracket delta when specified)
    bed_kg = spec.bed.mass_g * 1e-3
    x_beam_mass = sum(
        max(m.mass_kg(nmap) - m.rail_mass_kg - m.backer_mass_kg - m.fill_mass_kg, 0.0)
        for m in members
        if m.role == "gantry"
    )
    x_backer_mass = sum(m.backer_mass_kg for m in members if m.role == "gantry")
    x_rail_mass = x_rail_prof.area_m2 * w * rail_mat.density_kg_m3
    if spec.gantry.effective_dual_x_rails:
        x_rail_mass *= 2.0
    x_moving = toolhead_kg + 0.03 + bkt_delta_kg
    y_moving = bed_kg + spec.gantry.carriage_mass_g * 1e-3 + bkt_delta_kg

    # Explicit bedslinger Y-bed node mounted to base-level Y-rails (z=0.015 m, split at y_bed)
    bed_center = np.array([w * 0.5, y_bed, orig_z])
    z_y_rail = min(orig_z - 0.014, 0.015)
    fastener_springs: list[FastenerSpring] = []
    nodes.extend(
        [
            StructuralNode("bed_rail_l", np.array([0.0, y_bed, z_y_rail]), role="bed"),
            StructuralNode("bed_rail_r", np.array([w, y_bed, z_y_rail]), role="bed"),
        ]
    )
    _attach_bed_mount_compliance(nodes, fastener_springs, spec, bed_center)
    bed_vmat = _make_bed_virtual_material(get_material(spec.bed.plate_material))
    rail_vmat = _make_bed_virtual_material(get_material(spec.gantry.rail_material))
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
            StructuralMember(
                id=f"bed_y_rail_{tag}_f",
                start_id=f_nid,
                end_id=rail_nid,
                profile=y_rail_guide_prof,
                material=rail_vmat,
                role="bed",
            )
        )
        members.append(
            StructuralMember(
                id=f"bed_y_rail_{tag}_r",
                start_id=r_nid,
                end_id=rail_nid,
                profile=y_rail_guide_prof,
                material=rail_vmat,
                role="bed",
            )
        )

    l_arm = max(w * 0.5, 0.05)
    k_y_target = 1.15e6 * ((spec.gantry.belt.stiffness_ea_n / 26000.0) ** 0.35)
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
            StructuralMember(
                id=f"bed_arm_{tag}",
                start_id=rail_nid,
                end_id="bed_frame",
                profile=bed_car_prof,
                material=bed_vmat,
                role="bed",
            )
        )

    stage_stiffnesses = _attach_toolhead_cog_and_nozzle(
        nodes, members, spec, base_node_id="toolhead", fastener_springs=fastener_springs
    )
    if bkt_delta_kg > 0.0:
        next(n for n in nodes if n.id == "toolhead").lumped_mass_kg += bkt_delta_kg

    return AssemblyModel(
        spec=spec,
        nodes=nodes,
        members=members,
        fastener_springs=fastener_springs,
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
        y_carriage_joints_mass_kg=spec.gantry.carriage_mass_g * 1e-3,
        toolhead_moving_mass_kg=toolhead_kg,
        x_rail_ei_y_boost_pct=x_rail_ei_y_boost_pct,
        x_rail_ei_z_boost_pct=x_rail_ei_z_boost_pct,
        total_rail_mass_kg=sum(m.rail_mass_kg for m in members),
        toolhead_stage_stiffnesses_n_per_mm=stage_stiffnesses,
    )


def _build_delta(
    spec: PrinterSpec, toolhead_rel_pos: tuple[float, float, float]
) -> AssemblyModel:
    w = spec.frame.width_mm * 1e-3
    d = spec.frame.depth_mm * 1e-3
    h = spec.frame.height_mm * 1e-3
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
    toolhead_kg = _effective_toolhead_mass_kg(spec)
    bkt_info = GANTRY_BRACKET_CATALOG.get(
        spec.gantry.bracket_material, GANTRY_BRACKET_CATALOG["abs_printed"]
    )
    bkt_delta_kg = float(bkt_info["carriage_mass_delta_g"]) * 1e-3
    car_kg = (spec.gantry.carriage_mass_g / 3.0) * 1e-3

    nodes: list[StructuralNode] = []
    for i, ang in enumerate(angles):
        er_x, er_y = math.cos(ang), math.sin(ang)
        et_x, et_y = -er_y, er_x
        px = cx + r_tower * er_x
        py = cy + r_tower * er_y
        nodes.append(StructuralNode(f"b_{i}", np.array([px, py, 0.0]), fixed=True, lumped_mass_kg=motor_kg, role="foot"))
        nodes.append(StructuralNode(f"c_{i}", np.array([px, py, z_car]), lumped_mass_kg=car_kg, role="gantry"))
        nodes.append(StructuralNode(f"t_{i}", np.array([px, py, h]), role="frame"))
        for side, s_sign in (("a", -1.0), ("b", 1.0)):
            lat = s_sign * pair_half_spacing
            cx_p = cx + (r_tower - r_car) * er_x + lat * et_x
            cy_p = cy + (r_tower - r_car) * er_y + lat * et_y
            ex_p = cx + r_eff * er_x + lat * et_x
            ey_p = cy + r_eff * er_y + lat * et_y
            nodes.append(StructuralNode(f"c_{i}_{side}", np.array([cx_p, cy_p, z_car]), role="gantry"))
            nodes.append(StructuralNode(f"eff_{i}_{side}", np.array([ex_p, ey_p, z_effector]), role="toolhead"))

    nodes.append(StructuralNode("toolhead", np.array([cx, cy, z_effector]), lumped_mass_kg=toolhead_kg, role="toolhead"))

    fp = spec.frame.profile
    fm = spec.frame.material
    bp = spec.frame.brace_profile
    bm = spec.frame.brace_material
    members: list[StructuralMember] = []

    eff_link_vmat = _make_bed_virtual_material(get_material("carbon_fiber_tube")).model_copy(
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
        _add_member(members, f"base_{i}", f"b_{i}", f"b_{j}", fp, fm)
        _add_member(members, f"top_{i}", f"t_{i}", f"t_{j}", fp, fm)
        _add_member(members, f"tower_{i}_low", f"b_{i}", f"c_{i}", fp, fm, role="tower")
        _add_member(members, f"tower_{i}_top", f"c_{i}", f"t_{i}", fp, fm, role="tower")
        for side in ("a", "b"):
            members.append(
                StructuralMember(
                    id=f"car_link_{i}_{side}",
                    start_id=f"c_{i}",
                    end_id=f"c_{i}_{side}",
                    profile=eff_link_prof,
                    material=eff_link_vmat,
                    role="toolhead",
                )
            )
            members.append(
                StructuralMember(
                    id=f"eff_link_{i}_{side}",
                    start_id="toolhead",
                    end_id=f"eff_{i}_{side}",
                    profile=eff_link_prof,
                    material=eff_link_vmat,
                    role="toolhead",
                )
            )
            # Twin parallel carbon delta rods constraining effector X/Y/Z and pitch/roll/yaw
            _add_member(
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
            _add_member(members, f"brace_{i}_a", f"b_{i}", f"t_{j}", bp, bm, role="brace", is_truss=True)
        if spec.frame.bracing == "full_truss":
            _add_member(members, f"brace_{i}_b", f"b_{j}", f"t_{i}", bp, bm, role="brace", is_truss=True)

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
                    _apply_z_guideway(
                        mem_t, nmap_d, z_rail_id, rail_mat_d, n_rad, bolt_pitch_d, h
                    )
                    # Preserve exact 120-deg C3 rotational symmetry without off-diagonal I_yz
                    i_iso = 0.5 * (mem_t.profile.iy_m4 + mem_t.profile.iz_m4)
                    mem_t.profile = mem_t.profile.model_copy(
                        update={"iy_m4": i_iso, "iz_m4": i_iso}
                    )

    panels_active, panel_mounting, pmat, panel_t_m = _panel_shear_params(spec)
    total_panel_mass_kg = 0.0
    rep_panel_props: dict[str, float] | None = None
    if panels_active:
        nmap = {n.id: n for n in nodes}
        for i in range(3):
            j = (i + 1) % 3
            _, bay_mass, props = _add_shear_panel_bay(
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
    _attach_bed_mount_compliance(nodes, fastener_springs, spec, bed_center)
    bed_vmat = _make_bed_virtual_material(get_material(spec.bed.plate_material))
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
            StructuralMember(
                id=f"bed_arm_{i}",
                start_id=f"b_{i}",
                end_id="bed_frame",
                profile=delta_bed_prof,
                material=bed_vmat,
                role="bed",
            )
        )

    eff_mass = toolhead_kg + 0.09  # ultra-light effector + arm portion
    stage_stiffnesses = _attach_toolhead_cog_and_nozzle(
        nodes, members, spec, base_node_id="toolhead", fastener_springs=fastener_springs
    )
    if bkt_delta_kg > 0.0:
        next(n for n in nodes if n.id == "toolhead").lumped_mass_kg += bkt_delta_kg

    return AssemblyModel(
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


def _finalize_mass_and_cost(model: AssemblyModel) -> None:
    nmap = model.node_map
    frame_mass = 0.0
    cost = 0.0
    total_rail_mass = 0.0
    rail_mat = get_material(model.spec.gantry.rail_material)
    for m in model.members:
        mk = m.mass_kg(nmap)
        frame_mass += mk
        base_mk = max(mk - m.fill_mass_kg - m.rail_mass_kg - m.backer_mass_kg, 0.0)
        cost += base_mk * m.material.cost_per_kg_usd
        if m.rail_mass_kg > 0.0:
            total_rail_mass += m.rail_mass_kg
            cost += m.rail_mass_kg * rail_mat.cost_per_kg_usd
        if m.backer_mass_kg > 0.0 and m.backer_material_id:
            cost += m.backer_mass_kg * get_material(m.backer_material_id).cost_per_kg_usd
    model.total_rail_mass_kg = total_rail_mass

    if model.extrusion_fill_mass_kg > 0.0:
        fill_spec = EXTRUSION_FILLINGS.get(
            model.spec.frame.extrusion_filling, EXTRUSION_FILLINGS["none"]
        )
        cost += model.extrusion_fill_mass_kg * float(fill_spec["cost_per_kg_usd"])

    bkt_info = GANTRY_BRACKET_CATALOG.get(
        model.spec.gantry.bracket_material, GANTRY_BRACKET_CATALOG["abs_printed"]
    )
    cost += float(bkt_info["kit_cost_usd"])

    has_bed_node = bool(model.bed_node_id and model.bed_node_id in nmap)
    lumped_mass = sum(n.lumped_mass_kg for n in model.nodes)
    bed_mass = model.spec.bed.mass_g * 1e-3
    bed_frame_mass = nmap["bed_frame"].lumped_mass_kg if "bed_frame" in nmap else 0.0
    bed_node_mass = (nmap[model.bed_node_id].lumped_mass_kg if has_bed_node else 0.0) + bed_frame_mass
    model.frame_mass_kg = frame_mass
    # Shell elements carry their physical panel mass via MITC4 element mass matrices (not node lumps)
    shell_panel_mass = model.panel_mass_kg if model.shell_elements else 0.0
    model.total_mass_kg = (
        frame_mass + lumped_mass + shell_panel_mass + max(bed_mass - bed_node_mass, 0.0)
    )
    if model.panel_mass_kg > 0.0:
        pmat = get_material(model.spec.frame.panel_material)
        cost += model.panel_mass_kg * pmat.cost_per_kg_usd
    # Add baseline estimate for motors, bed plate, rails, and hotend hardware
    bed_mat = get_material(model.spec.bed.plate_material)
    cost += bed_mass * bed_mat.cost_per_kg_usd + 140.0  # steppers/electronics/hotend baseline
    model.estimated_material_cost_usd = cost
