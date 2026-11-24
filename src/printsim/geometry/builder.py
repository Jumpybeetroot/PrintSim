"""3D structural assembly graph generator from declarative PrinterSpec."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from printsim.schema import (
    BED_MOUNT_CATALOG,
    EXTRUSION_FILLINGS,
    FRAME_FEET_CATALOG,
    FRONT_BRACE_CATALOG,
    GANTRY_BRACKET_CATALOG,
    HOTEND_CATALOG,
    PROFILE_CATALOG,
    TOOLHEAD_CONSTRUCTION_CATALOG,
    TOP_BRACE_CATALOG,
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
    backbone_mass_kg: float = 0.0
    backbone_cost_usd: float = 0.0
    trident_backbone_active: bool = False
    top_brace_mass_kg: float = 0.0
    top_brace_cost_usd: float = 0.0
    top_brace_active: bool = False
    top_brace_is_shell: bool = False
    front_brace_mass_kg: float = 0.0
    front_brace_cost_usd: float = 0.0
    front_brace_active: bool = False
    front_brace_is_shell: bool = False
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
        bz = float(bc[2]) - float(self.spec.bed.plate_thickness_mm)
        return [
            [
                round(float(bc[0]) + dx_m * 1000.0, 2),
                round(float(bc[1]) + dy_m * 1000.0, 2),
                round(bz, 2),
            ]
            for dx_m, dy_m in _compute_mount_offsets_m(self.spec)
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
                    ("z_rail_left", "b_left_mid", "t_left_mid", "z_bed_l", (1.0, 0.0, 0.0)),
                    ("z_rail_right", "b_right_mid", "t_right_mid", "z_bed_r", (-1.0, 0.0, 0.0)),
                ]
            elif z_drive == "leadscrew_dual_belt_sync":
                specs = [
                    ("z_rod_lf", "b_rod_lf", "t_rod_lf", "z_rod_lf", (1.0, 0.0, 0.0)),
                    ("z_rod_lr", "b_rod_lr", "t_rod_lr", "z_rod_lr", (1.0, 0.0, 0.0)),
                    ("z_rod_rf", "b_rod_rf", "t_rod_rf", "z_rod_rf", (-1.0, 0.0, 0.0)),
                    ("z_rod_rr", "b_rod_rr", "t_rod_rr", "z_rod_rr", (-1.0, 0.0, 0.0)),
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

        vis_prof = (
            z_prof
            if (z_drive != "leadscrew_dual_belt_sync" or z_prof.startswith("steel_rod"))
            else "steel_rod_10mm"
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
                    "profile_id": vis_prof,
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
            elif z_drive in ("leadscrew_dual", "leadscrew_dual_belt_sync"):
                specs = [
                    ("z_drive_left", "b_left_mid", "t_left_mid", "z_bed_l", (20.0, 0.0, 0.0)),
                    ("z_drive_right", "b_right_mid", "t_right_mid", "z_bed_r", (-20.0, 0.0, 0.0)),
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

        shared_motor_pos = (
            nmap["b_rear_mid"].position_m * 1000.0
            if (z_drive == "leadscrew_dual_belt_sync" and "b_rear_mid" in nmap)
            else None
        )
        out: list[dict] = []
        for zid, b_id, t_id, nut_id, off_mm in specs:
            if b_id not in nmap or t_id not in nmap:
                continue
            off = np.asarray(off_mm, dtype=float)
            p0 = nmap[b_id].position_m * 1000.0 + off
            p1 = nmap[t_id].position_m * 1000.0 + off
            nut_z = float(nmap[nut_id].position_m[2] * 1000.0) if nut_id in nmap else float(self.bed_center_m[2] * 1000.0)
            mpos = shared_motor_pos if shared_motor_pos is not None else p0
            out.append(
                {
                    "id": zid,
                    "kind": kind,
                    "mechanism_id": mech_id,
                    "start_mm": [round(float(v), 2) for v in p0],
                    "end_mm": [round(float(v), 2) for v in p1],
                    "nut_z_mm": round(nut_z, 2),
                    "motor_pos_mm": [round(float(v), 2) for v in mpos],
                    "diameter_mm": round(diam_mm, 2),
                }
            )
        return out

    def _build_trident_backbone_dict(self) -> dict:
        dims = trident_backbone_dimensions_mm(self.spec)
        return {
            "enabled": bool(self.spec.frame.trident_backbone),
            "active": bool(self.trident_backbone_active),
            "geometry_kind": "open_perimeter_frame",
            "material_name": '3.18mm (1/8") Powder-Coated Mild Steel (Clee)',
            "thickness_mm": 3.18,
            "flange_width_mm": round(dims["flange_width_mm"], 1),
            "cutout_w_mm": round(dims["cutout_w_mm"], 1),
            "cutout_d_mm": round(dims["cutout_d_mm"], 1),
            "mass_kg": round(self.backbone_mass_kg, 3),
            "cost_usd": round(self.backbone_cost_usd, 2) if self.trident_backbone_active else 0.0,
            "z_thrust_boost_factor": 2.4 if self.trident_backbone_active else 1.0,
            "notes": (
                'Unified 3.18mm (1/8") mild-steel open perimeter backbone plate replacing 3 printed Z-motor mounts; '
                "keeps Z0/Z1/Z2 coplanar, eliminates rear-Z sag, and lowers CoG."
            ),
        }

    def _build_top_brace_dict(self) -> dict:
        mode = self.spec.frame.top_brace
        entry = TOP_BRACE_CATALOG.get(mode, TOP_BRACE_CATALOG["none"])
        dims = top_brace_dimensions_mm(self.spec)
        if not self.top_brace_active:
            mat_name = "None"
        elif entry.get("kind") == "cnc_plate":
            mat_name = get_material("aluminum_5052_sheet").name
        else:
            mat_name = get_material("aluminum_6061_t6").name
        return {
            "id": mode,
            "active": bool(self.top_brace_active),
            "label": str(entry.get("label", "None (Open Top)")),
            "material_name": mat_name,
            "thickness_mm": round(dims["thickness_mm"], 2) if self.top_brace_active else 0.0,
            "flange_width_mm": round(dims["flange_width_mm"], 1) if self.top_brace_active else 0.0,
            "cutout_w_mm": round(dims["cutout_w_mm"], 1),
            "cutout_d_mm": round(dims["cutout_d_mm"], 1),
            "mass_kg": round(self.top_brace_mass_kg, 3) if self.top_brace_active else 0.0,
            "cost_usd": round(self.top_brace_cost_usd, 2) if self.top_brace_active else 0.0,
        }

    def _build_front_brace_dict(self) -> dict:
        mode = self.spec.frame.front_brace
        entry = FRONT_BRACE_CATALOG.get(mode, FRONT_BRACE_CATALOG["none"])
        dims = front_brace_dimensions_mm(self.spec)
        if not self.front_brace_active:
            mat_name = "None"
        elif entry.get("kind") == "cnc_plate":
            mat_name = get_material("aluminum_5052_sheet").name
        else:
            mat_name = get_material("aluminum_6061_t6").name
        return {
            "id": mode,
            "active": bool(self.front_brace_active),
            "label": str(entry.get("label", "None (Open Front Door)")),
            "material_name": mat_name,
            "thickness_mm": round(dims["thickness_mm"], 2) if self.front_brace_active else 0.0,
            "flange_width_mm": round(dims["flange_width_mm"], 1) if self.front_brace_active else 0.0,
            "header_height_mm": round(dims["header_height_mm"], 1) if self.front_brace_active else 0.0,
            "skirt_height_mm": round(dims["skirt_height_mm"], 1) if self.front_brace_active else 0.0,
            "cutout_w_mm": round(dims["cutout_w_mm"], 1),
            "cutout_h_mm": round(dims["cutout_h_mm"], 1),
            "mass_kg": round(self.front_brace_mass_kg, 3) if self.front_brace_active else 0.0,
            "cost_usd": round(self.front_brace_cost_usd, 2) if self.front_brace_active else 0.0,
        }

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
                max(
                    self.spec.frame.height_mm * 1e-3
                    - self.spec.frame.gantry_top_clearance_mm * 1e-3,
                    0.04,
                )
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
        mount_pts = self._compute_bed_mount_points_mm()
        is_trident_3pt_subframe = (
            self.spec.architecture == "corexy"
            and self.spec.bed.z_drive == "leadscrew_triple"
            and self.spec.bed.mount_type
            in ("kinematic_3pt", "whopping_pochard_kinematic", "trident_3pt_standoffs")
            and all(nid in nmap for nid in ("bed_t_fl", "bed_t_fr", "bed_t_rear_mount"))
        )
        bed_mount_anchor_nodes = (
            ["bed_t_fl", "bed_t_fr", "bed_t_rear_mount"]
            if is_trident_3pt_subframe
            else (["bed_frame"] * len(mount_pts) if "bed_frame" in nmap else [])
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
                "top_panel": bool(self.spec.frame.top_panel),
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
            "gantry_top_clearance_mm": round(float(self.spec.frame.gantry_top_clearance_mm), 1),
            "trident_backbone": self._build_trident_backbone_dict(),
            "top_brace": self._build_top_brace_dict(),
            "front_brace": self._build_front_brace_dict(),
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
            "bed_subframe_z_mm": (
                round(float(nmap["bed_frame"].position_m[2] * 1000.0), 2)
                if "bed_frame" in nmap
                else None
            ),
            "bed_support_points_mm": [
                (
                    nmap[
                        m.end_id
                        if (m.start_id.startswith("bed_t_") and m.end_id in nmap)
                        else m.start_id
                    ].position_m
                    * 1000.0
                ).tolist()
                for m in self.members
                if m.id.startswith("bed_arm_") and m.start_id in nmap
            ],
            "bed_mount_points_mm": mount_pts,
            "bed_mount_anchor_nodes": bed_mount_anchor_nodes,
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
        # Perimeter twist-clips + closed-cell EPDM foam gasket shear coupling (replaces uncoupled soft spring)
        k_mount = 220_000.0
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

    up_prof_id = spec.frame.effective_upright_profile
    cj_h = corner_joint_stiffness(spec.frame.profile, jtype)
    if up_prof_id != spec.frame.profile:
        cj_up = corner_joint_stiffness(up_prof_id, jtype)
        cj = {
            "k_r_inplane_nm_rad": math.sqrt(
                cj_h["k_r_inplane_nm_rad"] * cj_up["k_r_inplane_nm_rad"]
            ),
            "k_r_torsion_nm_rad": math.sqrt(
                cj_h["k_r_torsion_nm_rad"] * cj_up["k_r_torsion_nm_rad"]
            ),
        }
    else:
        cj = cj_h
    bp = get_profile(up_prof_id)
    # Any un-railed vertical column carries the bare upright profile (or bonded core if already applied)
    unrailed = next(
        (m for m in members if m.role == "frame" and m.rail_mass_kg == 0.0 and m.id.startswith("col_")),
        None,
    )
    if unrailed is None and up_prof_id == spec.frame.profile:
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
    backbone_lower_col_boost = (
        spec.architecture == "corexy"
        and bool(spec.frame.trident_backbone)
        and not spec.is_flying_gantry
    )
    l_col = max(col_length_m, 0.15)
    # Bolted steel Z rails stop ~10% short of each corner joint; Castigliano bending-energy
    # integral M(x)^2 over the un-railed outer 10% ends is 1 - 0.8^3 = 0.488 (~49%).
    w_end = 0.49

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
        kr_in_m = kr_in
        kr_tor_m = kr_tor
        if backbone_lower_col_boost and is_corexy_col and member.id.endswith(("_bot", "_low")):
            # Bolted 3 mm mild-steel base plate acts as a rigid base-plane corner gusset/squaring plate
            kr_in_m *= 1.85
            kr_tor_m *= 1.75
        eta_iy = 1.0 / (1.0 + (6.0 * e_pa * iy_span) / (l_col * kr_in_m))
        eta_iz = 1.0 / (1.0 + (6.0 * e_pa * iz_span) / (l_col * kr_in_m))
        eta_j = 1.0 / (1.0 + (2.0 * g_pa * member.profile.j_m4) / (l_col * kr_tor_m))

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
    if spec.bed.mount_type in (
        "kinematic_3pt",
        "whopping_pochard_kinematic",
        "trident_3pt_standoffs",
    ):
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
) -> None:
    """Create intermediate ``bed_frame`` node at subframe centerline and 6-DOF ``bed_mount_spring`` to ``bed_surface``."""
    cfg = BED_MOUNT_CATALOG.get(spec.bed.mount_type, BED_MOUNT_CATALOG["kinematic_3pt"])
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
        StructuralNode("bed_frame", bed_frame_pos, lumped_mass_kg=bed_frame_kg, role="bed")
    )
    nodes.append(
        StructuralNode(
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
        FastenerSpring(
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
    allow_rear_z: bool = False,
) -> str:
    """Locate an existing structural frame/rail node near ``p_target`` or split the spanning beam."""
    excluded_roles = {"panel", "backbone", "toolhead", "cog", "nozzle", "bed"}
    if not allow_gantry:
        excluded_roles.add("gantry")
    best_id: str | None = None
    best_dist = snap_tol_m
    for n in nodes:
        if n.role in excluded_roles and not (allow_rear_z and n.id == "z_bed_r"):
            continue
        dist = float(np.linalg.norm(n.position_m - p_target))
        if dist < best_dist:
            best_dist = dist
            best_id = n.id
    if best_id is not None:
        return best_id

    for idx, m in enumerate(members):
        is_allowed_rear_z = allow_rear_z and m.id.startswith("bed_rear_z_")
        if m.role not in ("frame", "rail", "tower") and not is_allowed_rear_z:
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
                    role="bed" if is_allowed_rear_z else "frame",
                )
                nodes.append(node)
                nmap[split_id] = node
                _split_member_at_node(members, idx, split_id, proj / length)
                return split_id

    # Fallback: closest frame/rail node
    non_panel = [
        n
        for n in nodes
        if n.role not in excluded_roles or (allow_rear_z and n.id == "z_bed_r")
    ]
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
        bay_list: list[tuple[str, str, str, str, str, tuple[float, float, float], float | None]] = [
            ("rear_low", "b_rl", "b_rr", "g_rl", "g_rr", (0.0, 1.0, 0.0), h),
            ("rear_top", "g_rl", "g_rr", "t_rl", "t_rr", (0.0, 1.0, 0.0), h),
            ("left_low", "b_fl", "b_rl", "g_fl", "g_rl", (-1.0, 0.0, 0.0), h),
            ("left_top", "g_fl", "g_rl", "t_fl", "t_rl", (-1.0, 0.0, 0.0), h),
            ("right_low", "b_fr", "b_rr", "g_fr", "g_rr", (1.0, 0.0, 0.0), h),
            ("right_top", "g_fr", "g_rr", "t_fr", "t_rr", (1.0, 0.0, 0.0), h),
        ]
    else:
        bay_list = [
            ("rear", "b_rl", "b_rr", "t_rl", "t_rr", (0.0, 1.0, 0.0), None),
            ("left", "b_fl", "b_rl", "t_fl", "t_rl", (-1.0, 0.0, 0.0), None),
            ("right", "b_fr", "b_rr", "t_fr", "t_rr", (1.0, 0.0, 0.0), None),
        ]
    if spec.frame.top_panel:
        bay_list.append(("top", "t_fl", "t_fr", "t_rl", "t_rr", (0.0, 0.0, 1.0), None))
    bays = tuple(bay_list)

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
            k_shear_per_m = 2.0 * 220_000.0 / max(bw, 0.04)
            k_normal_per_m = 2.5e5
            k_rot_per_m = 0.2
        else:
            n_edge_w = max(3.0, round(bw / 0.085) + 1.0)
            gantry_mult = 1.35 if is_split_wall else 1.0
            k_mount_edge = n_edge_w * (props["k_mount_n_m"] / (0.5 * n_edge_w * gantry_mult))
            k_shear_per_m = k_mount_edge / max(bw, 0.04)
            k_normal_per_m = max(1.0e6, 2.0 * k_shear_per_m)
            k_rot_per_m = 45.0 * clamp_eff

        has_rear_z_col = (
            mounting != "clip_enclosure"
            and bay in ("rear", "rear_low", "rear_top")
            and spec.bed.z_drive in ("leadscrew_triple", "leadscrew_single")
            and "b_rear_mid" in nmap
            and "g_rear_mid" in nmap
            and "z_bed_r" in nmap
        )
        # Ensure rear bays with a rear-center vertical Z-column (Trident / single-Z)
        # have an exact centerline grid column (xi = 0.5) so the rear panel contacts
        # b_rear_mid -> bed_rear_z_bot -> z_bed_r -> bed_rear_z_top -> g_rear_mid
        # without unphysical out-of-plane pass-through in modal/static views.
        # Keep ny_g uniform (identical to left/right bays) so corner columns share
        # exact split nodes with zero stub segments or extra vertical DOFs.
        nx_g = (mesh_n if mesh_n % 2 == 0 else mesh_n + 1) if has_rear_z_col else mesh_n
        ny_g = mesh_n if bh > 0.12 else max(1, mesh_n // 2)
        rear_z_top_z: float | None = (
            float(nmap["g_rear_mid"].position_m[2])
            if (has_rear_z_col and bay in ("rear_low", "rear"))
            else None
        )
        nvec = np.asarray(n_tup, dtype=float)
        off_nominal = d_c * nvec
        mid_i = nx_g // 2

        grid = np.empty((ny_g + 1, nx_g + 1), dtype=object)
        for j in range(ny_g + 1):
            eta = j / ny_g
            ds_vert = (bh / ny_g) * (0.5 if j in (0, ny_g) else 1.0)
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

                is_perim = i in (0, nx_g) or j in (0, ny_g)
                on_rear_z_interior = (
                    not is_perim
                    and has_rear_z_col
                    and bay in ("rear_low", "rear")
                    and i == mid_i
                    and rear_z_top_z is not None
                    and float(p_frame[2]) <= rear_z_top_z + 1e-3
                )
                if is_perim:
                    fid = _find_or_split_frame_node(
                        nodes, nmap, members, p_frame, allow_gantry=bolted_to_gantry
                    )
                    ds = 0.0
                    if j in (0, ny_g):
                        ds += (bw / nx_g) * (0.5 if i in (0, nx_g) else 1.0)
                    if i in (0, nx_g):
                        ds += ds_vert
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
                elif on_rear_z_interior:
                    # Rear panel rests flush against the outer face of the rear-center
                    # vertical 2020 Z-extrusion (bed_rear_z_bot / z_bed_r / bed_rear_z_top):
                    # enforce out-of-plane normal contact (zero pass-through in Y) without
                    # adding artificial in-plane shear bolts that do not exist down the center.
                    fid = _find_or_split_frame_node(
                        nodes,
                        nmap,
                        members,
                        p_frame,
                        allow_gantry=bolted_to_gantry,
                        allow_rear_z=True,
                    )
                    actual_off = p_shell - nmap[fid].position_m
                    k_contact_n_m = max(k_normal_per_m * ds_vert, 5.0e6)
                    fastener_springs.append(
                        FastenerSpring(
                            id=f"cbush_{bay}_{j}_{i}",
                            bay=bay,
                            shell_node_id=sid,
                            frame_node_id=fid,
                            k_shear_n_m=0.0,
                            k_normal_n_m=k_contact_n_m,
                            k_rot_nm_rad=0.0,
                            normal_vec=n_tup,
                            offset_m=(
                                float(actual_off[0]),
                                float(actual_off[1]),
                                float(actual_off[2]),
                            ),
                        )
                    )

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


def trident_backbone_dimensions_mm(spec: PrinterSpec) -> dict[str, float]:
    """Compute Clee Trident Backbone perimeter flange & open central cutout dimensions (mm)."""
    fw = max(float(spec.frame.width_mm), 120.0)
    fd = max(float(spec.frame.depth_mm), 120.0)
    min_dim = min(fw, fd)
    flange = min(80.0, max(min(55.0, 0.30 * min_dim), min_dim * 0.24))
    cutout_w = max(min(80.0, 0.30 * fw), fw - 2.0 * flange)
    cutout_d = max(min(80.0, 0.30 * fd), fd - 2.0 * flange - 20.0)
    recess = min(20.0, max(0.0, fd - 2.0 * flange - cutout_d))
    return {
        "width_mm": fw,
        "depth_mm": fd,
        "thickness_mm": 3.18,
        "flange_width_mm": flange,
        "cutout_w_mm": cutout_w,
        "cutout_d_mm": cutout_d,
        "rear_recess_mm": recess,
    }


def _trident_backbone_cost_usd(spec: PrinterSpec) -> float:
    """West3D retail price for Clee's 3.18 mm (1/8") powder-coated mild-steel Trident Backbone."""
    max_xy_mm = max(
        float(spec.kinematics.build_volume_mm[0]),
        float(spec.kinematics.build_volume_mm[1]),
    )
    if max_xy_mm <= 275.0:
        return 44.99
    if max_xy_mm <= 325.0:
        return 49.99
    return 54.99


def _add_trident_backbone(
    spec: PrinterSpec,
    nodes: list[StructuralNode],
    nmap: dict[str, StructuralNode],
    members: list[StructuralMember],
) -> tuple[list[ShellPanelElement], list[FastenerSpring], float, float]:
    """Add Clee Trident Backbone: 3.18 mm mild-steel open-perimeter MITC4 ring & rear Z-wing stiffener."""
    w = spec.frame.width_mm * 1e-3
    d = spec.frame.depth_mm * 1e-3
    dims = trident_backbone_dimensions_mm(spec)
    flange_m = dims["flange_width_mm"] * 1e-3
    cutout_w_m = dims["cutout_w_mm"] * 1e-3
    cutout_d_m = dims["cutout_d_mm"] * 1e-3

    t_bb = 0.00318
    e_steel = 205.0e9
    nu_steel = 0.29
    rho_steel = 7850.0
    fill_factor = 0.94
    membrane_eff = 0.92

    ring_area_m2 = max(w * d - cutout_w_m * cutout_d_m, 1e-4)
    areal_mass = fill_factor * rho_steel * t_bb
    e_mem_eff = membrane_eff * e_steel * t_bb
    backbone_mass_kg = ring_area_m2 * areal_mass
    backbone_cost_usd = _trident_backbone_cost_usd(spec)

    bb_mat = MaterialSpec(
        id="mild_steel_powdercoated",
        name='3.18mm (1/8") Powder-Coated Mild Steel (Clee)',
        youngs_modulus_pa=e_steel,
        poisson_ratio=nu_steel,
        density_kg_m3=rho_steel,
        yield_strength_pa=250.0e6,
        thermal_conductivity_w_mk=50.0,
        specific_heat_j_kgk=490.0,
        thermal_expansion_1_k=12.0e-6,
        damping_ratio=0.012,
        cost_per_kg_usd=backbone_cost_usd / max(backbone_mass_kg, 0.5),
        color_hex="#232830",
    )

    frame_prof = get_profile(spec.frame.profile)
    d_c = 0.5 * frame_prof.height_m + 0.5 * t_bb
    n_tup = (0.0, 0.0, -1.0)
    nvec = np.asarray(n_tup, dtype=float)
    off_nominal = d_c * nvec

    # Stiffen lower perimeter extrusions (especially rear Z-motor wing) via Newmark partial-composite
    # 3.18 mm mild-steel bottom flange without adding duplicate mass (mass lives on MITC4 shell).
    from printsim.fea.components import newmark_rail_composite_gamma

    g_steel = bb_mat.effective_shear_modulus_pa
    for m in members:
        if not m.id.startswith("base_"):
            continue
        is_rear = m.id.startswith("base_rear")
        is_x_span = is_rear or m.id.startswith("base_front")
        span_m = max(w if is_x_span else d, 0.10)
        w_flange = 0.085 if is_rear else 0.030
        pitch_m = 0.045 if is_rear else 0.055
        a_flange = t_bb * w_flange
        eta_bb = newmark_rail_composite_gamma(
            e_rail_pa=e_steel,
            a_rail_m2=a_flange,
            e_beam_pa=m.material.youngs_modulus_pa,
            a_beam_m2=m.profile.area_m2,
            bolt_pitch_m=pitch_m,
            l_eff_m=span_m,
            substrate_kind="extrusion",
        )
        n_e = e_steel / max(m.material.youngs_modulus_pa, 1e6)
        n_g_ratio = g_steel / max(m.material.effective_shear_modulus_pa, 1e6)
        a_ext = m.profile.area_m2
        n_a = n_e * a_flange
        i_par = eta_bb * ((a_ext * n_a) / max(a_ext + n_a, 1e-12)) * (d_c**2)
        i_loc = n_e * (w_flange * (t_bb**3) / 12.0)
        j_flange = n_g_ratio * (w_flange * (t_bb**3) / 3.0) + (
            0.50 * i_par if is_rear else 0.20 * i_par
        )
        ey, ez = _member_local_axes(nmap[m.start_id].position_m, nmap[m.end_id].position_m)
        cy = float(np.dot(nvec, ey)) ** 2
        cz = float(np.dot(nvec, ez)) ** 2
        d_iy = cz * (i_loc + i_par)
        d_iz = cy * (i_loc + i_par)
        m.profile = m.profile.model_copy(
            update={
                "iy_m4": m.profile.iy_m4 + d_iy,
                "iz_m4": m.profile.iz_m4 + d_iz,
                "j_m4": m.profile.j_m4 + j_flange,
            }
        )

    # Bolted M5 steel-to-extrusion Huth CBUSH fastener springs around bottom perimeter
    props = panel_mechanical_properties(bb_mat, t_bb, w, d, "bolted_structural")
    n_edge_w = max(3.0, round(w / 0.085) + 1.0)
    k_mount_edge = n_edge_w * (props["k_mount_n_m"] / (0.5 * n_edge_w))
    k_shear_per_m = k_mount_edge / max(w, 0.04)
    k_normal_per_m = max(1.0e6, 2.0 * k_shear_per_m)
    k_rot_per_m = 450.0

    p00 = nmap["b_fl"].position_m
    p10 = nmap["b_fr"].position_m
    p01 = nmap["b_rl"].position_m
    p11 = nmap["b_rr"].position_m
    bw = float(np.linalg.norm(p10 - p00))
    bh = float(np.linalg.norm(p01 - p00))

    # Open-center 4x4 perimeter ring aligned to real ~80 mm flanges (skipping 4 center quads & center node)
    x_cut0_m = 0.5 * (w - cutout_w_m)
    x_cut1_m = 0.5 * (w + cutout_w_m)
    y_cut0_m = min(flange_m, 0.5 * (d - cutout_d_m))
    y_cut1_m = y_cut0_m + cutout_d_m

    u1 = max(0.05, min(0.45, x_cut0_m / max(w, 1e-6)))
    u3 = min(0.95, max(0.55, x_cut1_m / max(w, 1e-6)))
    u_stops = (0.0, u1, 0.5, u3, 1.0)

    v1 = max(0.05, min(0.45, y_cut0_m / max(d, 1e-6)))
    v3 = min(0.95, max(v1 + 0.10, y_cut1_m / max(d, 1e-6)))
    v2 = 0.5 if (v1 + 0.05 < 0.5 < v3 - 0.05) else 0.5 * (v1 + v3)
    v_stops = (0.0, v1, v2, v3, 1.0)

    nx_g = len(u_stops) - 1  # 4
    ny_g = len(v_stops) - 1  # 4
    xs_m = [u * bw for u in u_stops]
    ys_m = [v * bh for v in v_stops]

    grid = np.empty((ny_g + 1, nx_g + 1), dtype=object)
    shell_elements: list[ShellPanelElement] = []
    fastener_springs: list[FastenerSpring] = []
    for j in range(ny_g + 1):
        eta = v_stops[j]
        for i in range(nx_g + 1):
            if i == 2 and j == 2:
                # Center point lies inside open central cutout; omit to avoid an unattached orphan node
                continue
            xi = u_stops[i]
            p_frame = (
                (1.0 - xi) * (1.0 - eta) * p00
                + xi * (1.0 - eta) * p10
                + xi * eta * p11
                + (1.0 - xi) * eta * p01
            )
            p_shell = p_frame + off_nominal
            sid = f"pan_backbone_{j}_{i}"
            node = StructuralNode(id=sid, position_m=p_shell, role="backbone")
            nodes.append(node)
            nmap[sid] = node
            grid[j, i] = sid

            if i in (0, nx_g) or j in (0, ny_g):
                fid = _find_or_split_frame_node(
                    nodes, nmap, members, p_frame, allow_gantry=False
                )
                ds = 0.0
                if j in (0, ny_g):
                    if i > 0:
                        ds += 0.5 * (xs_m[i] - xs_m[i - 1])
                    if i < nx_g:
                        ds += 0.5 * (xs_m[i + 1] - xs_m[i])
                if i in (0, nx_g):
                    if j > 0:
                        ds += 0.5 * (ys_m[j] - ys_m[j - 1])
                    if j < ny_g:
                        ds += 0.5 * (ys_m[j + 1] - ys_m[j])
                actual_off = p_shell - nmap[fid].position_m
                fastener_springs.append(
                    FastenerSpring(
                        id=f"cbush_backbone_{j}_{i}",
                        bay="backbone",
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
                )

    for j in range(ny_g):
        for i in range(nx_g):
            if i not in (0, nx_g - 1) and j not in (0, ny_g - 1):
                continue
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
                    id=f"shell_backbone_{j}_{i}",
                    bay="backbone",
                    node_ids=nids,
                    material=bb_mat,
                    thickness_m=t_bb,
                    areal_mass_kg_m2=areal_mass,
                    e_mem_h_n_m=e_mem_eff,
                    normal_vec=n_tup,
                )
            )

    return shell_elements, fastener_springs, backbone_mass_kg, backbone_cost_usd


def top_brace_dimensions_mm(spec: PrinterSpec, variant: str | None = None) -> dict[str, float]:
    """Compute open-center CoreXY top brace plate or flat-bar dimensions (mm)."""
    fw = max(float(spec.frame.width_mm), 120.0)
    fd = max(float(spec.frame.depth_mm), 120.0)
    min_dim = min(fw, fd)
    mode = spec.frame.top_brace if variant is None else variant
    if mode == "cnc_halo_ring":
        flange = min(50.0, 0.28 * min_dim)
        cutout_w = max(40.0, fw - 2.0 * flange)
        cutout_d = max(40.0, fd - 2.0 * flange)
        fillet = min(20.0, 0.25 * min(cutout_w, cutout_d))
        return {
            "width_mm": fw,
            "depth_mm": fd,
            "thickness_mm": 5.0,
            "flange_width_mm": flange,
            "cutout_w_mm": cutout_w,
            "cutout_d_mm": cutout_d,
            "fillet_radius_mm": fillet,
            "arm_length_mm": 0.0,
        }
    if mode in ("rear_v_and_front_corners", "corner_45_bars"):
        arm_mm = min(130.0, 0.28 * min_dim)
        bar_w = 25.0
        return {
            "width_mm": fw,
            "depth_mm": fd,
            "thickness_mm": 5.0,
            "flange_width_mm": bar_w,
            "cutout_w_mm": max(40.0, fw - 2.0 * bar_w),
            "cutout_d_mm": max(40.0, fd - 2.0 * bar_w),
            "fillet_radius_mm": 0.0,
            "arm_length_mm": arm_mm,
        }
    return {
        "width_mm": fw,
        "depth_mm": fd,
        "thickness_mm": 0.0,
        "flange_width_mm": 0.0,
        "cutout_w_mm": fw,
        "cutout_d_mm": fd,
        "fillet_radius_mm": 0.0,
        "arm_length_mm": 0.0,
    }


def front_brace_dimensions_mm(spec: PrinterSpec, variant: str | None = None) -> dict[str, float]:
    """Compute open-door CoreXY front brace plate or flat-bar dimensions (mm)."""
    fw = max(float(spec.frame.width_mm), 120.0)
    fh = max(float(spec.frame.height_mm), 120.0)
    min_dim = min(fw, fh)
    mode = spec.frame.front_brace if variant is None else variant
    try:
        ext_w = float(get_profile(spec.frame.effective_upright_profile).width_m) * 1000.0
    except KeyError:
        ext_w = 20.0

    if mode == "cnc_portal_frame":
        border = min(55.0, 0.28 * min_dim)
        cutout_w = max(40.0, fw - 2.0 * border)
        cutout_h = max(40.0, fh - 2.0 * border)
        fillet = min(20.0, 0.25 * min(cutout_w, cutout_h))
        return {
            "width_mm": fw,
            "height_mm": fh,
            "thickness_mm": 5.0,
            "flange_width_mm": border,
            "header_height_mm": 0.0,
            "skirt_height_mm": 0.0,
            "cutout_w_mm": cutout_w,
            "cutout_h_mm": cutout_h,
            "fillet_radius_mm": fillet,
            "arm_length_mm": 0.0,
        }
    if mode == "header_and_skirt":
        top_clear_mm = min(float(spec.frame.gantry_top_clearance_mm), max(fh - 160.0, 25.0))
        z_gantry_mm = max(80.0, fh - top_clear_mm)
        header_h = max(20.0, fh - z_gantry_mm)
        skirt_h = min(90.0, 0.25 * z_gantry_mm)
        cutout_w = max(40.0, fw - 2.0 * ext_w)
        cutout_h = max(40.0, z_gantry_mm - skirt_h)
        return {
            "width_mm": fw,
            "height_mm": fh,
            "thickness_mm": 5.0,
            "flange_width_mm": ext_w,
            "header_height_mm": header_h,
            "skirt_height_mm": skirt_h,
            "cutout_w_mm": cutout_w,
            "cutout_h_mm": cutout_h,
            "fillet_radius_mm": 0.0,
            "arm_length_mm": 0.0,
        }
    if mode == "corner_45_bars":
        arm_mm = min(120.0, 0.26 * min_dim)
        bar_w = 25.0
        return {
            "width_mm": fw,
            "height_mm": fh,
            "thickness_mm": 5.0,
            "flange_width_mm": bar_w,
            "header_height_mm": 0.0,
            "skirt_height_mm": 0.0,
            "cutout_w_mm": max(40.0, fw - 2.0 * bar_w),
            "cutout_h_mm": max(40.0, fh - 2.0 * bar_w),
            "fillet_radius_mm": 0.0,
            "arm_length_mm": arm_mm,
        }
    return {
        "width_mm": fw,
        "height_mm": fh,
        "thickness_mm": 0.0,
        "flange_width_mm": 0.0,
        "header_height_mm": 0.0,
        "skirt_height_mm": 0.0,
        "cutout_w_mm": fw,
        "cutout_h_mm": fh,
        "fillet_radius_mm": 0.0,
        "arm_length_mm": 0.0,
    }


def _point_to_segment_dist_m(pt: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    len_sq = float(np.dot(ab, ab))
    if len_sq < 1e-12:
        return float(np.linalg.norm(pt - a))
    t = max(0.0, min(1.0, float(np.dot(pt - a, ab) / len_sq)))
    proj = a + t * ab
    return float(np.linalg.norm(pt - proj))


def _add_open_ring_shell_plate(
    *,
    bay: str,
    node_prefix: str,
    shell_prefix: str,
    cbush_prefix: str,
    p00: np.ndarray,
    p10: np.ndarray,
    p01: np.ndarray,
    p11: np.ndarray,
    cutout_w_m: float,
    cutout_h_m: float,
    n_tup: tuple[float, float, float],
    d_c: float,
    t_plate_m: float,
    mat: MaterialSpec,
    areal_mass: float,
    e_mem_eff: float,
    k_shear_per_m: float,
    k_normal_per_m: float,
    k_rot_per_m: float,
    nodes: list[StructuralNode],
    nmap: dict[str, StructuralNode],
    members: list[StructuralMember],
    allow_gantry: bool,
) -> tuple[list[ShellPanelElement], list[FastenerSpring], float]:
    """Generate a 12-quad open-center MITC4 ring (4x4 grid omitting center 2x2) + perimeter CBUSH springs."""
    bw = float(np.linalg.norm(p10 - p00))
    bh = float(np.linalg.norm(p01 - p00))
    x_cut0_m = 0.5 * (bw - cutout_w_m)
    x_cut1_m = 0.5 * (bw + cutout_w_m)
    y_cut0_m = 0.5 * (bh - cutout_h_m)
    y_cut1_m = 0.5 * (bh + cutout_h_m)

    u1 = max(0.05, min(0.45, x_cut0_m / max(bw, 1e-6)))
    u3 = min(0.95, max(0.55, x_cut1_m / max(bw, 1e-6)))
    u_stops = (0.0, u1, 0.5, u3, 1.0)

    v1 = max(0.05, min(0.45, y_cut0_m / max(bh, 1e-6)))
    v3 = min(0.95, max(0.55, y_cut1_m / max(bh, 1e-6)))
    v_stops = (0.0, v1, 0.5, v3, 1.0)

    nx_g = len(u_stops) - 1  # 4
    ny_g = len(v_stops) - 1  # 4
    xs_m = [u * bw for u in u_stops]
    ys_m = [v * bh for v in v_stops]
    nvec = np.asarray(n_tup, dtype=float)
    off_nominal = d_c * nvec

    grid = np.empty((ny_g + 1, nx_g + 1), dtype=object)
    shells: list[ShellPanelElement] = []
    springs: list[FastenerSpring] = []
    for j in range(ny_g + 1):
        eta = v_stops[j]
        for i in range(nx_g + 1):
            if i == 2 and j == 2:
                continue
            xi = u_stops[i]
            p_frame = (
                (1.0 - xi) * (1.0 - eta) * p00
                + xi * (1.0 - eta) * p10
                + xi * eta * p11
                + (1.0 - xi) * eta * p01
            )
            p_shell = p_frame + off_nominal
            sid = f"{node_prefix}_{j}_{i}"
            node = StructuralNode(id=sid, position_m=p_shell, role="panel")
            nodes.append(node)
            nmap[sid] = node
            grid[j, i] = sid

            if i in (0, nx_g) or j in (0, ny_g):
                fid = _find_or_split_frame_node(
                    nodes, nmap, members, p_frame, allow_gantry=allow_gantry
                )
                ds = 0.0
                if j in (0, ny_g):
                    if i > 0:
                        ds += 0.5 * (xs_m[i] - xs_m[i - 1])
                    if i < nx_g:
                        ds += 0.5 * (xs_m[i + 1] - xs_m[i])
                if i in (0, nx_g):
                    if j > 0:
                        ds += 0.5 * (ys_m[j] - ys_m[j - 1])
                    if j < ny_g:
                        ds += 0.5 * (ys_m[j + 1] - ys_m[j])
                actual_off = p_shell - nmap[fid].position_m
                springs.append(
                    FastenerSpring(
                        id=f"{cbush_prefix}_{j}_{i}",
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
                )

    for j in range(ny_g):
        for i in range(nx_g):
            if i not in (0, nx_g - 1) and j not in (0, ny_g - 1):
                continue
            n0 = str(grid[j, i])
            n1 = str(grid[j, i + 1])
            n2 = str(grid[j + 1, i + 1])
            n3 = str(grid[j + 1, i])
            c3 = np.array([nmap[nid].position_m for nid in (n0, n1, n2, n3)])
            if float(np.dot(np.cross(c3[2] - c3[0], c3[3] - c3[1]), nvec)) < 0.0:
                nids = (n0, n3, n2, n1)
            else:
                nids = (n0, n1, n2, n3)
            shells.append(
                ShellPanelElement(
                    id=f"{shell_prefix}_{j}_{i}",
                    bay=bay,
                    node_ids=nids,
                    material=mat,
                    thickness_m=t_plate_m,
                    areal_mass_kg_m2=areal_mass,
                    e_mem_h_n_m=e_mem_eff,
                    normal_vec=n_tup,
                )
            )

    actual_area_m2 = max(bw * bh * (1.0 - (u3 - u1) * (v3 - v1)), 1e-4)
    return shells, springs, actual_area_m2 * areal_mass


def _add_flat_bar_brace_kit(
    *,
    id_prefix: str,
    endpoints_m: list[tuple[tuple[float, float, float], tuple[float, float, float]]],
    nodes: list[StructuralNode],
    nmap: dict[str, StructuralNode],
    members: list[StructuralMember],
    allow_gantry: bool,
) -> float:
    """Attach 25x5 mm Al 6061-T6 flat-bar brace beams directly into frame nodes/rails."""
    bar_prof = get_profile("al_flat_bar_25x5")
    bar_mat = get_material("aluminum_6061_t6")
    total_mass_kg = 0.0
    for idx, (p0_tup, p1_tup) in enumerate(endpoints_m):
        p0 = np.asarray(p0_tup, dtype=float)
        p1 = np.asarray(p1_tup, dtype=float)
        n0 = _find_or_split_frame_node(nodes, nmap, members, p0, allow_gantry=allow_gantry)
        n1 = _find_or_split_frame_node(nodes, nmap, members, p1, allow_gantry=allow_gantry)
        if n0 == n1:
            continue
        mem = StructuralMember(
            id=f"{id_prefix}_{idx}",
            start_id=n0,
            end_id=n1,
            profile=bar_prof,
            material=bar_mat,
            role="brace",
            is_truss=False,
        )
        members.append(mem)
        total_mass_kg += mem.mass_kg(nmap)
    return total_mass_kg


def _add_top_and_front_braces(
    *,
    spec: PrinterSpec,
    nodes: list[StructuralNode],
    nmap: dict[str, StructuralNode],
    members: list[StructuralMember],
    panel_mesh_n: int,
) -> dict[str, Any]:
    """Build CoreXY non-blocking open-center Top Braces and open-door Front Braces."""
    w = spec.frame.width_mm * 1e-3
    d = spec.frame.depth_mm * 1e-3
    h = spec.frame.height_mm * 1e-3
    allow_gantry = not spec.is_flying_gantry

    shells: list[ShellPanelElement] = []
    springs: list[FastenerSpring] = []
    top_mode = spec.frame.top_brace
    front_mode = spec.frame.front_brace

    al5052 = get_material("aluminum_5052_sheet")
    t_plate = 0.005
    areal_mass_5052 = al5052.density_kg_m3 * t_plate
    frame_prof = get_profile(spec.frame.profile)

    def _plate_coupling_params(span_w_m: float, span_h_m: float) -> tuple[float, float, float, float]:
        props = panel_mechanical_properties(al5052, t_plate, span_w_m, span_h_m, "bolted_structural")
        n_edge_w = max(3.0, round(span_w_m / 0.085) + 1.0)
        k_mount_edge = n_edge_w * (props["k_mount_n_m"] / (0.5 * n_edge_w))
        k_shear_per_m = k_mount_edge / max(span_w_m, 0.04)
        k_normal_per_m = max(1.0e6, 2.0 * k_shear_per_m)
        k_rot_per_m = 450.0
        e_mem_eff = 0.92 * al5052.youngs_modulus_pa * t_plate
        return e_mem_eff, k_shear_per_m, k_normal_per_m, k_rot_per_m

    # 1. Open-center Top Brace
    top_mass_kg = 0.0
    top_cost_usd = 0.0
    top_active = top_mode != "none"
    top_is_shell = False
    if top_active:
        top_entry = TOP_BRACE_CATALOG.get(top_mode, TOP_BRACE_CATALOG["none"])
        top_cost_usd = float(top_entry.get("cost_usd", 0.0))
        if top_mode == "cnc_halo_ring":
            dims_t = top_brace_dimensions_mm(spec, variant="cnc_halo_ring")
            e_mem_eff, k_shear_per_m, k_normal_per_m, k_rot_per_m = _plate_coupling_params(w, d)
            d_c = 0.5 * frame_prof.height_m + 0.5 * t_plate
            t_shells, t_springs, top_mass_kg = _add_open_ring_shell_plate(
                bay="top_brace",
                node_prefix="pan_top_brace",
                shell_prefix="shell_top_brace",
                cbush_prefix="cbush_top_brace",
                p00=nmap["t_fl"].position_m,
                p10=nmap["t_fr"].position_m,
                p01=nmap["t_rl"].position_m,
                p11=nmap["t_rr"].position_m,
                cutout_w_m=dims_t["cutout_w_mm"] * 1e-3,
                cutout_h_m=dims_t["cutout_d_mm"] * 1e-3,
                n_tup=(0.0, 0.0, 1.0),
                d_c=d_c,
                t_plate_m=t_plate,
                mat=al5052,
                areal_mass=areal_mass_5052,
                e_mem_eff=e_mem_eff,
                k_shear_per_m=k_shear_per_m,
                k_normal_per_m=k_normal_per_m,
                k_rot_per_m=k_rot_per_m,
                nodes=nodes,
                nmap=nmap,
                members=members,
                allow_gantry=allow_gantry,
            )
            shells.extend(t_shells)
            springs.extend(t_springs)
            top_is_shell = True
        elif top_mode == "rear_v_and_front_corners":
            l_arm = min(0.130, 0.28 * min(w, d))
            top_mass_kg = _add_flat_bar_brace_kit(
                id_prefix="brace_top_open",
                endpoints_m=[
                    ((0.5 * w, d, h), (0.0, 0.5 * d, h)),
                    ((0.5 * w, d, h), (w, 0.5 * d, h)),
                    ((l_arm, 0.0, h), (0.0, l_arm, h)),
                    ((w - l_arm, 0.0, h), (w, l_arm, h)),
                ],
                nodes=nodes,
                nmap=nmap,
                members=members,
                allow_gantry=allow_gantry,
            )
        elif top_mode == "corner_45_bars":
            l_arm = min(0.130, 0.28 * min(w, d))
            top_mass_kg = _add_flat_bar_brace_kit(
                id_prefix="brace_top_open",
                endpoints_m=[
                    ((l_arm, 0.0, h), (0.0, l_arm, h)),
                    ((w - l_arm, 0.0, h), (w, l_arm, h)),
                    ((l_arm, d, h), (0.0, d - l_arm, h)),
                    ((w - l_arm, d, h), (w, d - l_arm, h)),
                ],
                nodes=nodes,
                nmap=nmap,
                members=members,
                allow_gantry=allow_gantry,
            )

    # 2. Open-door Front Brace
    front_mass_kg = 0.0
    front_cost_usd = 0.0
    front_active = front_mode != "none"
    front_is_shell = False
    if front_active:
        front_entry = FRONT_BRACE_CATALOG.get(front_mode, FRONT_BRACE_CATALOG["none"])
        front_cost_usd = float(front_entry.get("cost_usd", 0.0))
        if front_mode == "cnc_portal_frame":
            dims_f = front_brace_dimensions_mm(spec, variant="cnc_portal_frame")
            e_mem_eff, k_shear_per_m, k_normal_per_m, k_rot_per_m = _plate_coupling_params(w, h)
            d_c = 0.5 * frame_prof.width_m + 0.5 * t_plate
            f_shells, f_springs, front_mass_kg = _add_open_ring_shell_plate(
                bay="front_brace",
                node_prefix="pan_front_brace",
                shell_prefix="shell_front_brace",
                cbush_prefix="cbush_front_brace",
                p00=nmap["b_fl"].position_m,
                p10=nmap["b_fr"].position_m,
                p01=nmap["t_fl"].position_m,
                p11=nmap["t_fr"].position_m,
                cutout_w_m=dims_f["cutout_w_mm"] * 1e-3,
                cutout_h_m=dims_f["cutout_h_mm"] * 1e-3,
                n_tup=(0.0, -1.0, 0.0),
                d_c=d_c,
                t_plate_m=t_plate,
                mat=al5052,
                areal_mass=areal_mass_5052,
                e_mem_eff=e_mem_eff,
                k_shear_per_m=k_shear_per_m,
                k_normal_per_m=k_normal_per_m,
                k_rot_per_m=k_rot_per_m,
                nodes=nodes,
                nmap=nmap,
                members=members,
                allow_gantry=allow_gantry,
            )
            shells.extend(f_shells)
            springs.extend(f_springs)
            front_is_shell = True
        elif front_mode == "header_and_skirt":
            dims_f = front_brace_dimensions_mm(spec, variant="header_and_skirt")
            hdr_h_m = dims_f["header_height_mm"] * 1e-3
            skr_h_m = dims_f["skirt_height_mm"] * 1e-3
            e_mem_eff, k_shear_per_m, k_normal_per_m, k_rot_per_m = _plate_coupling_params(w, h)
            d_c = 0.5 * frame_prof.width_m + 0.5 * t_plate
            n_tup = (0.0, -1.0, 0.0)
            nvec = np.asarray(n_tup, dtype=float)
            off_nominal = d_c * nvec

            p_bfl = nmap["b_fl"].position_m
            p_bfr = nmap["b_fr"].position_m
            p_tfl = nmap["t_fl"].position_m
            p_tfr = nmap["t_fr"].position_m
            front_rails = (
                (p_bfl, p_bfr),
                (p_tfl, p_tfr),
                (p_bfl, p_tfl),
                (p_bfr, p_tfr),
            )
            nx_strip = max(4, panel_mesh_n)

            def _build_front_strip(
                *,
                node_prefix: str,
                shell_prefix: str,
                cbush_prefix: str,
                z0_m: float,
                z1_m: float,
                is_header: bool,
            ) -> float:
                bh = max(z1_m - z0_m, 1e-4)
                ny_strip = 1 if bh < 0.075 else 2
                p00 = np.array([0.0, 0.0, z0_m], dtype=float)
                p10 = np.array([w, 0.0, z0_m], dtype=float)
                p01 = np.array([0.0, 0.0, z1_m], dtype=float)
                p11 = np.array([w, 0.0, z1_m], dtype=float)

                grid = np.empty((ny_strip + 1, nx_strip + 1), dtype=object)
                for j in range(ny_strip + 1):
                    eta = j / ny_strip
                    for i in range(nx_strip + 1):
                        xi = i / nx_strip
                        p_frame = (
                            (1.0 - xi) * (1.0 - eta) * p00
                            + xi * (1.0 - eta) * p10
                            + xi * eta * p11
                            + (1.0 - xi) * eta * p01
                        )
                        p_shell = p_frame + off_nominal
                        sid = f"{node_prefix}_{j}_{i}"
                        node = StructuralNode(id=sid, position_m=p_shell, role="panel")
                        nodes.append(node)
                        nmap[sid] = node
                        grid[j, i] = sid

                        is_perim = i in (0, nx_strip) or j in (0, ny_strip)
                        if is_perim:
                            rail_dist = min(
                                _point_to_segment_dist_m(p_frame, ra, rb) for ra, rb in front_rails
                            )
                            if rail_dist <= 0.005:
                                fid = _find_or_split_frame_node(
                                    nodes, nmap, members, p_frame, allow_gantry=allow_gantry
                                )
                                ds = 0.0
                                attached_horiz = (j == ny_strip) if is_header else (j == 0)
                                if attached_horiz:
                                    ds += (w / nx_strip) * (0.5 if i in (0, nx_strip) else 1.0)
                                if i in (0, nx_strip):
                                    ds += (bh / ny_strip) * (0.5 if j in (0, ny_strip) else 1.0)
                                actual_off = p_shell - nmap[fid].position_m
                                springs.append(
                                    FastenerSpring(
                                        id=f"{cbush_prefix}_{j}_{i}",
                                        bay="front_brace",
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
                                )

                for j in range(ny_strip):
                    for i in range(nx_strip):
                        n0 = str(grid[j, i])
                        n1 = str(grid[j, i + 1])
                        n2 = str(grid[j + 1, i + 1])
                        n3 = str(grid[j + 1, i])
                        c3 = np.array([nmap[nid].position_m for nid in (n0, n1, n2, n3)])
                        if float(np.dot(np.cross(c3[2] - c3[0], c3[3] - c3[1]), nvec)) < 0.0:
                            nids = (n0, n3, n2, n1)
                        else:
                            nids = (n0, n1, n2, n3)
                        shells.append(
                            ShellPanelElement(
                                id=f"{shell_prefix}_{j}_{i}",
                                bay="front_brace",
                                node_ids=nids,
                                material=al5052,
                                thickness_m=t_plate,
                                areal_mass_kg_m2=areal_mass_5052,
                                e_mem_h_n_m=e_mem_eff,
                                normal_vec=n_tup,
                            )
                        )
                return w * bh * areal_mass_5052

            hdr_mass = _build_front_strip(
                node_prefix="pan_front_hdr",
                shell_prefix="shell_front_hdr",
                cbush_prefix="cbush_front_hdr",
                z0_m=h - hdr_h_m,
                z1_m=h,
                is_header=True,
            )
            skr_mass = _build_front_strip(
                node_prefix="pan_front_skr",
                shell_prefix="shell_front_skr",
                cbush_prefix="cbush_front_skr",
                z0_m=0.0,
                z1_m=skr_h_m,
                is_header=False,
            )
            front_mass_kg = hdr_mass + skr_mass
            front_is_shell = True
        elif front_mode == "corner_45_bars":
            l_arm = min(0.120, 0.26 * min(w, h))
            front_mass_kg = _add_flat_bar_brace_kit(
                id_prefix="brace_front_open",
                endpoints_m=[
                    ((l_arm, 0.0, 0.0), (0.0, 0.0, l_arm)),
                    ((w - l_arm, 0.0, 0.0), (w, 0.0, l_arm)),
                    ((l_arm, 0.0, h), (0.0, 0.0, h - l_arm)),
                    ((w - l_arm, 0.0, h), (w, 0.0, h - l_arm)),
                ],
                nodes=nodes,
                nmap=nmap,
                members=members,
                allow_gantry=allow_gantry,
            )

    return {
        "shells": shells,
        "springs": springs,
        "top_brace_mass_kg": top_mass_kg,
        "top_brace_cost_usd": top_cost_usd,
        "top_brace_active": top_active,
        "top_brace_is_shell": top_is_shell,
        "front_brace_mass_kg": front_mass_kg,
        "front_brace_cost_usd": front_cost_usd,
        "front_brace_active": front_active,
        "front_brace_is_shell": front_is_shell,
    }


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
    if spec.gantry.custom_bracket_override is not None:
        bkt_delta_kg += float(spec.gantry.custom_bracket_override.mass_delta_g) * 1e-3
    y_carriage_total_kg = max(spec.gantry.carriage_mass_g * 1e-3 + bkt_delta_kg, 0.01)
    carriage_side_kg = 0.5 * y_carriage_total_kg
    toolhead_kg = _effective_toolhead_mass_kg(spec)
    bed_kg = spec.bed.mass_g * 1e-3

    # Build origin centered in frame
    orig_x = (w - bx) * 0.5
    orig_y = (d - by) * 0.5
    orig_z = 0.08
    y_gantry = orig_y + v * by
    x_tool = orig_x + u * bx

    is_flying = spec.is_flying_gantry
    trident_backbone_active = bool(spec.frame.trident_backbone and not is_flying)
    z_drive = spec.bed.z_drive
    bed_cfg = BED_MOUNT_CATALOG.get(spec.bed.mount_type, BED_MOUNT_CATALOG["kinematic_3pt"])
    t_bed_m = float(spec.bed.plate_thickness_mm) * 1e-3
    h_standoff_m = float(bed_cfg.get("standoff_height_mm", 14.0)) * 1e-3
    half_subframe_m = 0.010
    top_clear_m = min(float(spec.frame.gantry_top_clearance_mm) * 1e-3, max(h - 0.16, 0.025))
    if is_flying:
        z_bed = orig_z
        z_gantry_min = min(z_bed + 0.055, max(h - 0.08, 0.06))
        z_gantry_max = max(z_gantry_min + 0.02, h - top_clear_m)
        z_gantry = float(
            np.clip(
                z_bed + 0.055 + w_rel * max(bz - 0.04, 0.05),
                z_gantry_min,
                z_gantry_max,
            )
        )
    else:
        # Gantry plane near top of CoreXY frame
        z_gantry = h - top_clear_m
        z_min_req = max(0.08, 0.030 + t_bed_m + h_standoff_m + half_subframe_m)
        z_min = min(z_min_req, 0.5 * z_gantry)
        z_max = max(z_min + 0.01, z_gantry - 0.045)
        z_stroke = min(bz, max(z_gantry - 0.14, 0.02))
        z_bed = float(np.clip(z_gantry - 0.045 - w_rel * z_stroke, z_min, z_max))
    z_subframe = max(0.025, z_bed - t_bed_m - h_standoff_m - half_subframe_m)

    is_awd = spec.gantry.drive_mode == "awd"
    cnc_motor_mount_kg = 0.11 if bool(bkt_info["cnc_machined"]) else 0.0
    rear_motor_kg = xy_motor_kg + cnc_motor_mount_kg
    front_motor_kg = (xy_motor_kg + 0.03 + cnc_motor_mount_kg) if is_awd else 0.0
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
        if trident_backbone_active:
            k_thrust *= 2.4
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
                StructuralNode("g_rl", np.array([0.0, d, z_gantry]), lumped_mass_kg=rear_motor_kg, role="gantry"),
                StructuralNode("g_rr", np.array([w, d, z_gantry]), lumped_mass_kg=rear_motor_kg, role="gantry"),
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
                StructuralNode("g_rl", np.array([0.0, d, z_gantry]), lumped_mass_kg=rear_motor_kg, role="mount"),
                StructuralNode("g_rr", np.array([w, d, z_gantry]), lumped_mass_kg=rear_motor_kg, role="mount"),
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
        bed_x_m = float(spec.bed.size_mm[0]) * 1e-3
        bed_y_m = float(spec.bed.size_mm[1]) * 1e-3
        x_fl = w * 0.5 - 0.375 * bed_x_m
        x_fr = w * 0.5 + 0.375 * bed_x_m
        y_f = d * 0.5 - 0.375 * bed_y_m
        y_r = d * 0.5 + 0.375 * bed_y_m
        nodes.extend(
            [
                StructuralNode("z_bed_fl", np.array([0.0, 0.0, z_subframe]), role="frame"),
                StructuralNode("z_bed_fr", np.array([w, 0.0, z_subframe]), role="frame"),
                StructuralNode("b_rear_mid", np.array([w * 0.5, d, 0.0]), lumped_mass_kg=z_motor_kg, role="frame"),
                StructuralNode("g_rear_mid", np.array([w * 0.5, d, z_gantry]), role="frame"),
                StructuralNode("z_bed_r", np.array([w * 0.5, d, z_subframe]), role="bed"),
                StructuralNode("bed_t_fl", np.array([x_fl, y_f, z_subframe]), role="bed"),
                StructuralNode("bed_t_fr", np.array([x_fr, y_f, z_subframe]), role="bed"),
                StructuralNode("bed_t_front", np.array([w * 0.5, y_f, z_subframe]), role="bed"),
                StructuralNode("bed_t_rear_mount", np.array([w * 0.5, y_r, z_subframe]), role="bed"),
            ]
        )
    elif z_drive == "leadscrew_dual":
        # Anchor top of dual-Z side guides to top perimeter rails (t_left_mid / t_right_mid)
        # so the moving Y carriages y_car_l/r never collide or weld to Z guide nodes at mid-Y.
        nodes.extend(
            [
                StructuralNode("b_left_mid", np.array([0.0, y_mid, 0.0]), lumped_mass_kg=z_motor_kg, role="frame"),
                StructuralNode("b_right_mid", np.array([w, y_mid, 0.0]), lumped_mass_kg=z_motor_kg, role="frame"),
                StructuralNode("t_left_mid", np.array([0.0, y_mid, h]), role="frame"),
                StructuralNode("t_right_mid", np.array([w, y_mid, h]), role="frame"),
                StructuralNode("z_bed_l", np.array([0.0, y_mid, z_subframe]), role="bed"),
                StructuralNode("z_bed_r", np.array([w, y_mid, z_subframe]), role="bed"),
            ]
        )
    elif z_drive == "leadscrew_dual_belt_sync":
        # Real VzBot 330/235 Z-architecture: 1 NEMA17 Z stepper at rear-center driving 2 belt-synced
        # mid-side TR8 leadscrews + 4 vertical hardened smooth rods (y = 0.25d, 0.75d on left & right)
        # anchored between base and top perimeter rails with LMK10LUU flanged linear bearings.
        y_rod_f = d * 0.25
        y_rod_r = d * 0.75
        nodes.extend(
            [
                StructuralNode("b_rear_mid", np.array([w * 0.5, d, 0.0]), lumped_mass_kg=z_motor_kg, role="frame"),
                StructuralNode("b_left_mid", np.array([0.0, y_mid, 0.0]), role="frame"),
                StructuralNode("b_right_mid", np.array([w, y_mid, 0.0]), role="frame"),
                StructuralNode("t_left_mid", np.array([0.0, y_mid, h]), role="frame"),
                StructuralNode("t_right_mid", np.array([w, y_mid, h]), role="frame"),
                StructuralNode("z_bed_l", np.array([0.0, y_mid, z_subframe]), role="bed"),
                StructuralNode("z_bed_r", np.array([w, y_mid, z_subframe]), role="bed"),
                # 4 smooth-rod base anchors, top anchors, rod-at-bed nodes, and LMK bearing housings
                StructuralNode("b_rod_lf", np.array([0.0, y_rod_f, 0.0]), role="frame"),
                StructuralNode("b_rod_lr", np.array([0.0, y_rod_r, 0.0]), role="frame"),
                StructuralNode("b_rod_rf", np.array([w, y_rod_f, 0.0]), role="frame"),
                StructuralNode("b_rod_rr", np.array([w, y_rod_r, 0.0]), role="frame"),
                StructuralNode("t_rod_lf", np.array([0.0, y_rod_f, h]), role="frame"),
                StructuralNode("t_rod_lr", np.array([0.0, y_rod_r, h]), role="frame"),
                StructuralNode("t_rod_rf", np.array([w, y_rod_f, h]), role="frame"),
                StructuralNode("t_rod_rr", np.array([w, y_rod_r, h]), role="frame"),
                StructuralNode("z_rod_lf", np.array([0.0, y_rod_f, z_subframe]), role="bed"),
                StructuralNode("z_rod_lr", np.array([0.0, y_rod_r, z_subframe]), role="bed"),
                StructuralNode("z_rod_rf", np.array([w, y_rod_f, z_subframe]), role="bed"),
                StructuralNode("z_rod_rr", np.array([w, y_rod_r, z_subframe]), role="bed"),
                StructuralNode("z_brg_lf", np.array([0.0, y_rod_f, z_subframe]), role="bed"),
                StructuralNode("z_brg_lr", np.array([0.0, y_rod_r, z_subframe]), role="bed"),
                StructuralNode("z_brg_rf", np.array([w, y_rod_f, z_subframe]), role="bed"),
                StructuralNode("z_brg_rr", np.array([w, y_rod_r, z_subframe]), role="bed"),
            ]
        )
        for tag in ("lf", "lr", "rf", "rr"):
            fastener_springs.append(
                FastenerSpring(
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
                StructuralNode("b_rear_mid", np.array([w * 0.5, d, 0.0]), lumped_mass_kg=z_motor_kg, role="frame"),
                StructuralNode("g_rear_mid", np.array([w * 0.5, d, z_gantry]), role="frame"),
                StructuralNode("z_bed_r", np.array([w * 0.5, d, z_subframe]), role="bed"),
            ]
        )

    fp = spec.frame.profile
    up = spec.frame.effective_upright_profile
    fm = spec.frame.material
    bp = spec.frame.brace_profile
    bm = spec.frame.brace_material
    members: list[StructuralMember] = []

    # Base perimeter (split only where Z-rail/rod columns or Z-motors anchor)
    _add_member(members, "base_front", "b_fl", "b_fr", fp, fm)
    if z_drive in ("leadscrew_triple", "leadscrew_single", "leadscrew_dual_belt_sync"):
        _add_member(members, "base_rear_l", "b_rl", "b_rear_mid", fp, fm)
        _add_member(members, "base_rear_r", "b_rear_mid", "b_rr", fp, fm)
    else:
        _add_member(members, "base_rear", "b_rl", "b_rr", fp, fm)

    if z_drive == "leadscrew_dual":
        _add_member(members, "base_left_f", "b_fl", "b_left_mid", fp, fm)
        _add_member(members, "base_left_r", "b_left_mid", "b_rl", fp, fm)
        _add_member(members, "base_right_f", "b_fr", "b_right_mid", fp, fm)
        _add_member(members, "base_right_r", "b_right_mid", "b_rr", fp, fm)
    elif z_drive == "leadscrew_dual_belt_sync":
        _add_member(members, "base_left_1", "b_fl", "b_rod_lf", fp, fm)
        _add_member(members, "base_left_2", "b_rod_lf", "b_left_mid", fp, fm)
        _add_member(members, "base_left_3", "b_left_mid", "b_rod_lr", fp, fm)
        _add_member(members, "base_left_4", "b_rod_lr", "b_rl", fp, fm)
        _add_member(members, "base_right_1", "b_fr", "b_rod_rf", fp, fm)
        _add_member(members, "base_right_2", "b_rod_rf", "b_right_mid", fp, fm)
        _add_member(members, "base_right_3", "b_right_mid", "b_rod_rr", fp, fm)
        _add_member(members, "base_right_4", "b_rod_rr", "b_rr", fp, fm)
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
            _add_member(members, f"col_{tag}_bot", f"b_{tag}", f"z_bed_{tag}", up, fm)
            _add_member(members, f"col_{tag}_mid", f"z_bed_{tag}", col_mid, up, fm)
        else:
            _add_member(members, f"col_{tag}_low", f"b_{tag}", col_mid, up, fm)
        _add_member(members, f"col_{tag}_top", col_mid, f"t_{tag}", up, fm)

    # Top perimeter (split along left/right rails where dual-Z guides / 4x smooth rods anchor)
    _add_member(members, "top_front", "t_fl", "t_fr", fp, fm)
    _add_member(members, "top_rear", "t_rl", "t_rr", fp, fm)
    if z_drive == "leadscrew_dual":
        _add_member(members, "top_left_f", "t_fl", "t_left_mid", fp, fm)
        _add_member(members, "top_left_r", "t_left_mid", "t_rl", fp, fm)
        _add_member(members, "top_right_f", "t_fr", "t_right_mid", fp, fm)
        _add_member(members, "top_right_r", "t_right_mid", "t_rr", fp, fm)
    elif z_drive == "leadscrew_dual_belt_sync":
        _add_member(members, "top_left_1", "t_fl", "t_rod_lf", fp, fm)
        _add_member(members, "top_left_2", "t_rod_lf", "t_left_mid", fp, fm)
        _add_member(members, "top_left_3", "t_left_mid", "t_rod_lr", fp, fm)
        _add_member(members, "top_left_4", "t_rod_lr", "t_rl", fp, fm)
        _add_member(members, "top_right_1", "t_fr", "t_rod_rf", fp, fm)
        _add_member(members, "top_right_2", "t_rod_rf", "t_right_mid", fp, fm)
        _add_member(members, "top_right_3", "t_right_mid", "t_rod_rr", fp, fm)
        _add_member(members, "top_right_4", "t_rod_rr", "t_rr", fp, fm)
    else:
        _add_member(members, "top_left", "t_fl", "t_rl", fp, fm)
        _add_member(members, "top_right", "t_fr", "t_rr", fp, fm)

    # Mid gantry frame rails (left/right Y rails + rear motor bar, open front door)
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
            iy_m4=8.0e-8 + 2.4e-7 * (t_ratio**3),
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

        def _series_z_arm_iy(
            iy_nom: float, start_nid: str, end_nid: str = "bed_frame"
        ) -> float:
            l_arm = max(
                float(np.linalg.norm(nmap_pre[start_nid].position_m - nmap_pre[end_nid].position_m)),
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
            ext_2020 = get_profile("extrusion_2020")
            for tag, start_nid, end_nid in (
                ("fl", "z_bed_fl", "bed_t_fl"),
                ("fr", "z_bed_fr", "bed_t_fr"),
                ("r", "bed_t_rear_mount", "z_bed_r"),
            ):
                bed_arm_prof = ProfileSpec.model_construct(
                    id="bed_trident_arm_profile",
                    name=f"3-Point Z Bed Arm + Carriage ({t_mm:.1f}mm {spec.bed.plate_material})",
                    shape="extrusion",
                    width_m=ext_2020.width_m,
                    height_m=ext_2020.height_m,
                    area_m2=1.4e-3,
                    iy_m4=_series_z_arm_iy(iy_triple_arm, start_nid, end_nid),
                    iz_m4=5.0e-7,
                    j_m4=1.5e-7,
                )
                members.append(
                    StructuralMember(
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
                j_m4=ext_2020.j_m4 + 1.789e-7 * (t_ratio**3),
            )
            bed_spine_prof = ProfileSpec.model_construct(
                id="bed_trident_subframe_spine_profile",
                name=f"2020 T-Subframe Spine + Plate ({t_mm:.1f}mm {spec.bed.plate_material})",
                shape="extrusion",
                width_m=ext_2020.width_m,
                height_m=ext_2020.height_m,
                area_m2=1.4e-3,
                iy_m4=ext_2020.iy_m4 + 1.9932e-6 * (t_ratio**3),
                iz_m4=ext_2020.iz_m4 + 4.0e-6 * t_ratio,
                j_m4=ext_2020.j_m4 + 1.489e-7 * (t_ratio**3),
            )
            for mid, n0, n1, prof in (
                ("bed_subframe_fl", "bed_t_fl", "bed_t_front", bed_crossbar_prof),
                ("bed_subframe_fr", "bed_t_front", "bed_t_fr", bed_crossbar_prof),
                ("bed_subframe_spine_f", "bed_t_front", "bed_frame", bed_spine_prof),
                ("bed_subframe_spine_r", "bed_frame", "bed_t_rear_mount", bed_spine_prof),
            ):
                members.append(
                    StructuralMember(
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
                name="Dual-Z Side Linear Guide + Leadscrew",
                shape="solid_bar",
                width_m=0.012,
                height_m=0.012,
                area_m2=8.0e-5,  # steel-equivalent TR8 axial EA in aluminum-modulus virtual material
                iy_m4=0.22 * frame_prof.iy_m4,
                iz_m4=0.22 * frame_prof.iz_m4,
                j_m4=0.22 * frame_prof.j_m4,
            )
            for side, b_nid, z_nid, t_nid in (
                ("left", "b_left_mid", "z_bed_l", "t_left_mid"),
                ("right", "b_right_mid", "z_bed_r", "t_right_mid"),
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
                        end_id=t_nid,
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
                    iy_m4=_series_z_arm_iy(0.95 * iy_triple_arm, z_nid),
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
        elif z_drive == "leadscrew_dual_belt_sync":
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
                StructuralMember(
                    id="base_z_cross_f",
                    start_id="b_rod_lf",
                    end_id="b_rod_rf",
                    profile=frame_prof,
                    material=get_material(fm),
                    role="bed",
                )
            )
            members.append(
                StructuralMember(
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
                            "j_m4": m_side.profile.iy_m4 * 2.5,
                        }
                    )
                elif m_side.id.startswith(("top_left_", "top_right_")):
                    m_side.profile = m_side.profile.model_copy(
                        update={
                            "iz_m4": m_side.profile.iz_m4 * 2.5,
                            "j_m4": m_side.profile.iy_m4 * 2.5,
                        }
                    )
            for tag in ("lf", "lr", "rf", "rr"):
                b_nid = f"b_rod_{tag}"
                z_nid = f"z_rod_{tag}"
                t_nid = f"t_rod_{tag}"
                l_bot = float(np.linalg.norm(nmap_pre[z_nid].position_m - nmap_pre[b_nid].position_m))
                l_top = float(np.linalg.norm(nmap_pre[t_nid].position_m - nmap_pre[z_nid].position_m))
                members.append(
                    StructuralMember(
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
                    StructuralMember(
                        id=f"bed_rod_{tag}_top",
                        start_id=z_nid,
                        end_id=t_nid,
                        profile=rod_eff_prof,
                        material=rod_mat,
                        role="bed",
                        rail_mass_kg=rod_eff_prof.area_m2 * l_top * rod_mat.density_kg_m3,
                    )
                )
            # 2x mid-side belt-synced TR8 leadscrews carrying vertical Z bed load
            ls_diam_m = float(z_mech.get("diameter_mm", 8.0)) * 1e-3
            ls_r_m = 0.5 * ls_diam_m
            ls_i_m4 = 0.25 * math.pi * (ls_r_m**4)
            ls_j_m4 = 0.5 * math.pi * (ls_r_m**4)
            ls_root_a = float(z_mech.get("root_area_m2", 2.38e-5))
            ls_vmat = _make_bed_virtual_material(rod_mat)
            ls_bot_prof = ProfileSpec.model_construct(
                id="bed_sync_leadscrew_bot_profile",
                name=f"Belt-Synced Z Leadscrew Lower ({z_mech_id})",
                shape="rod",
                width_m=ls_diam_m,
                height_m=ls_diam_m,
                area_m2=ls_root_a,
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
                ("left", "b_left_mid", "z_bed_l", "t_left_mid"),
                ("right", "b_right_mid", "z_bed_r", "t_right_mid"),
            ):
                members.append(
                    StructuralMember(
                        id=f"bed_{side}_z_bot",
                        start_id=b_nid,
                        end_id=z_nid,
                        profile=ls_bot_prof,
                        material=ls_vmat,
                        role="bed",
                    )
                )
                members.append(
                    StructuralMember(
                        id=f"bed_{side}_z_top",
                        start_id=z_nid,
                        end_id=t_nid,
                        profile=ls_top_prof,
                        material=ls_vmat,
                        role="bed",
                    )
                )
            # 2x mid-side Z-lift bed arms + 4x rigid 2020 bed-carrier arms into LMK10LUU bearings
            for tag, z_nid in (("l", "z_bed_l"), ("r", "z_bed_r")):
                bed_sync_arm_prof = ProfileSpec.model_construct(
                    id="bed_sync_dual_arm_profile",
                    name=f"Belt-Synced Dual-Z Bed Subframe ({t_mm:.1f}mm {spec.bed.plate_material})",
                    shape="solid_bar",
                    width_m=0.020,
                    height_m=max(0.014, 0.014 + t_mm * 1e-3),
                    area_m2=6.0e-4,
                    iy_m4=_series_z_arm_iy(0.75 * iy_triple_arm, z_nid),
                    iz_m4=0.90 * iz_triple_arm,
                    j_m4=0.75 * j_triple_arm,
                )
                members.append(
                    StructuralMember(
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
                    StructuralMember(
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
    # reserving full-face diagonal rods for explicit truss/cross modes)
    bracing = spec.frame.bracing
    brace_g = "col_g_" if is_flying else "g_"
    if bracing == "top_cross":
        _add_member(members, "brace_top_1", "t_fl", "t_rr", bp, bm, role="brace", is_truss=True)
        _add_member(members, "brace_top_2", "t_fr", "t_rl", bp, bm, role="brace", is_truss=True)
    elif bracing == "rear_cross":
        _add_member(members, "brace_rear_1", "b_rl", "t_rr", bp, bm, role="brace", is_truss=True)
        _add_member(members, "brace_rear_2", "b_rr", "t_rl", bp, bm, role="brace", is_truss=True)
        _add_member(members, "brace_top_1", "t_fl", "t_rr", bp, bm, role="brace", is_truss=True)
    elif bracing == "side_cross":
        _add_member(members, "brace_left_1", "b_fl", f"{brace_g}rl", bp, bm, role="brace", is_truss=True)
        _add_member(members, "brace_left_2", "b_rl", f"{brace_g}fl", bp, bm, role="brace", is_truss=True)
        _add_member(members, "brace_right_1", "b_fr", f"{brace_g}rr", bp, bm, role="brace", is_truss=True)
        _add_member(members, "brace_right_2", "b_rr", f"{brace_g}fr", bp, bm, role="brace", is_truss=True)
    elif bracing == "rear_and_sides":
        _add_member(members, "brace_rear_1", "b_rl", "t_rr", bp, bm, role="brace", is_truss=True)
        _add_member(members, "brace_rear_2", "b_rr", "t_rl", bp, bm, role="brace", is_truss=True)
        _add_member(members, "brace_left_1", "b_fl", f"{brace_g}rl", bp, bm, role="brace", is_truss=True)
        _add_member(members, "brace_left_2", "b_rl", f"{brace_g}fl", bp, bm, role="brace", is_truss=True)
        _add_member(members, "brace_right_1", "b_fr", f"{brace_g}rr", bp, bm, role="brace", is_truss=True)
        _add_member(members, "brace_right_2", "b_rr", f"{brace_g}fr", bp, bm, role="brace", is_truss=True)
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
                            m_obj.profile = m_obj.profile.model_copy(update={"area_m2": 8.0e-5})
                            _rebalance_virtual_bed_rail_mass(m_obj, nmap, old_rho, old_rail_mass)
                            if mid.endswith("_top"):
                                top_area = 6.0e-7
                                m_obj.material = m_obj.material.model_copy(
                                    update={"density_kg_m3": m_obj.material.density_kg_m3 * (8.0e-5 / top_area)}
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
    shell_elements: list[ShellPanelElement] = []
    backbone_mass_kg = 0.0
    backbone_cost_usd = 0.0
    if trident_backbone_active:
        bb_shells, bb_springs, backbone_mass_kg, backbone_cost_usd = _add_trident_backbone(
            spec=spec,
            nodes=nodes,
            nmap=nmap,
            members=members,
        )
        shell_elements.extend(bb_shells)
        fastener_springs.extend(bb_springs)

    panels_active, panel_mounting, pmat, panel_t_m = _panel_shear_params(spec)
    total_panel_mass_kg = 0.0
    rep_panel_props: dict[str, float] | None = None
    if panels_active:
        bolted_to_gantry = spec.panels_bolted_to_gantry_effective
        pan_shells, panel_springs, total_panel_mass_kg, rep_panel_props = (
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
        shell_elements.extend(pan_shells)
        fastener_springs.extend(panel_springs)

    brace_res = _add_top_and_front_braces(
        spec=spec,
        nodes=nodes,
        nmap=nmap,
        members=members,
        panel_mesh_n=panel_mesh_n,
    )
    shell_elements.extend(brace_res["shells"])
    fastener_springs.extend(brace_res["springs"])

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
        + y_carriage_total_kg
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
        y_carriage_joints_mass_kg=y_carriage_total_kg,
        toolhead_moving_mass_kg=toolhead_kg,
        panel_mass_kg=total_panel_mass_kg,
        backbone_mass_kg=backbone_mass_kg,
        backbone_cost_usd=backbone_cost_usd,
        trident_backbone_active=trident_backbone_active,
        top_brace_mass_kg=float(brace_res["top_brace_mass_kg"]),
        top_brace_cost_usd=float(brace_res["top_brace_cost_usd"]),
        top_brace_active=bool(brace_res["top_brace_active"]),
        top_brace_is_shell=bool(brace_res["top_brace_is_shell"]),
        front_brace_mass_kg=float(brace_res["front_brace_mass_kg"]),
        front_brace_cost_usd=float(brace_res["front_brace_cost_usd"]),
        front_brace_active=bool(brace_res["front_brace_active"]),
        front_brace_is_shell=bool(brace_res["front_brace_is_shell"]),
        x_rail_ei_y_boost_pct=x_rail_ei_y_boost_pct,
        x_rail_ei_z_boost_pct=x_rail_ei_z_boost_pct,
        y_rail_ei_x_boost_pct=y_rail_ei_x_boost_pct,
        total_rail_mass_kg=sum(m.rail_mass_kg for m in members),
        panel_rep_props=rep_panel_props,
        toolhead_stage_stiffnesses_n_per_mm=stage_stiffnesses,
    )


# Architecture-specific builders extracted to sibling modules (re-exported here for backwards compatibility)
from printsim.geometry._builder_bedslinger import _build_cartesian_bedslinger  # noqa: E402
from printsim.geometry._builder_delta import _build_delta  # noqa: E402


def _finalize_mass_and_cost(model: AssemblyModel) -> None:
    nmap = model.node_map
    frame_mass = 0.0
    cost = 0.0
    total_rail_mass = 0.0
    rail_mat = get_material(model.spec.gantry.rail_material)
    for m in model.members:
        mk = m.mass_kg(nmap)
        frame_mass += mk
        if m.id.startswith(("brace_top_open_", "brace_front_open_")):
            continue
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
    # Shell elements carry their physical panel/backbone/CNC-brace mass via MITC4 element mass matrices (not node lumps)
    top_shell_mass = model.top_brace_mass_kg if model.top_brace_is_shell else 0.0
    front_shell_mass = model.front_brace_mass_kg if model.front_brace_is_shell else 0.0
    shell_panel_mass = (
        (model.panel_mass_kg + model.backbone_mass_kg + top_shell_mass + front_shell_mass)
        if model.shell_elements
        else 0.0
    )
    model.total_mass_kg = (
        frame_mass + lumped_mass + shell_panel_mass + max(bed_mass - bed_node_mass, 0.0)
    )
    if model.panel_mass_kg > 0.0:
        pmat = get_material(model.spec.frame.panel_material)
        cost += model.panel_mass_kg * pmat.cost_per_kg_usd
    if model.trident_backbone_active:
        cost += model.backbone_cost_usd
    if model.top_brace_active:
        cost += model.top_brace_cost_usd
    if model.front_brace_active:
        cost += model.front_brace_cost_usd
    # Add baseline estimate for motors, bed plate, rails, and hotend hardware
    bed_mat = get_material(model.spec.bed.plate_material)
    cost += bed_mass * bed_mat.cost_per_kg_usd + 140.0  # steppers/electronics/hotend baseline
    model.estimated_material_cost_usd = cost
