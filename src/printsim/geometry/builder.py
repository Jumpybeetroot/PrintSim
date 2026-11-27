"""3D structural assembly graph generator from declarative PrinterSpec."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from printsim.schema import (
    BED_MOUNT_CATALOG,
    EXTRUSION_FILLINGS,
    FRAME_FEET_CATALOG,
    FRONT_BRACE_CATALOG,
    GANTRY_BRACKET_CATALOG,
    GANTRY_HARDWARE_KIT_CATALOG,
    HOTEND_CATALOG,
    PROFILE_CATALOG,
    TOOLHEAD_CONSTRUCTION_CATALOG,
    TOOLHEAD_PRESETS,
    TOP_BRACE_CATALOG,
    Z_DRIVE_MECHANISM_CATALOG,
    Z_PROBE_CATALOG,
    Z_RAIL_PROFILES,
    MaterialSpec,
    PrinterSpec,
    ProfileSpec,
    get_material,
    get_profile,
    resolve_bed_mount,
    resolve_corner_joint,
    resolve_extrusion_filling,
    resolve_frame_feet,
    resolve_gantry_bracket,
    resolve_hotend_clamp,
    resolve_rail_quality,
    resolve_toolhead_construction,
    resolve_z_drive_mechanism,
    resolve_z_probe,
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
    insert_mass_kg: float = 0.0

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
    k_diag_6dof: tuple[float, float, float, float, float, float] | None = None


@dataclass
class AttachedSuperelement:
    """A 2-port Craig-Bampton condensed CAD superelement attached between two AssemblyModel nodes."""

    id: str
    category: str
    asset_id: str
    port_node_ids: tuple[str, str]
    K_cb: np.ndarray
    M_cb: np.ndarray | None = None
    port_coords_cad_m: np.ndarray | None = None
    n_internal_modes: int = 0
    part_group_id: str = "toolhead_mount"
    part_group_label: str = "Toolhead & Nozzle Mount"
    role: str = "toolhead"
    mass_kg: float = 0.0
    fixed_interface_freqs_hz: list[float] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)
    visual_glb_url: str | None = None
    cad_trust_pct: int | None = None

    def to_summary_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of this active CAD superelement."""
        m = self.manifest or {}
        return {
            "id": self.id,
            "category": self.category,
            "asset_id": self.asset_id,
            "name": str(m.get("name") or self.asset_id),
            "element_type": str(m.get("element_type") or "Tet4"),
            "num_elements": int(m.get("num_elements") or 0),
            "port_stiffness_n_per_mm": m.get("port_stiffness_n_per_mm") or {},
            "port0_mm": m.get("port0_mm") or [0.0, 0.0, 0.0],
            "port1_mm": m.get("port1_mm") or None,
            "port_node_ids": list(self.port_node_ids),
            "n_internal_modes": int(self.n_internal_modes),
            "part_group_id": self.part_group_id,
            "part_group_label": self.part_group_label,
            "role": self.role,
            "mass_kg": round(float(self.mass_kg), 6),
            "fixed_interface_freqs_hz": [
                round(float(f), 2) for f in self.fixed_interface_freqs_hz
            ],
            "visual_glb_url": self.visual_glb_url,
            "cad_trust_pct": self.cad_trust_pct,
        }


@dataclass
class AssemblyModel:
    """Complete physical 3D assembly model ready for FEA, kinematics, and visualization."""

    spec: PrinterSpec
    nodes: list[StructuralNode] = field(default_factory=list)
    members: list[StructuralMember] = field(default_factory=list)
    shell_elements: list[ShellPanelElement] = field(default_factory=list)
    fastener_springs: list[FastenerSpring] = field(default_factory=list)
    superelements: list[AttachedSuperelement] = field(default_factory=list)
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
    x_beam_insert_mass_kg: float = 0.0
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
    x_rail_gj_boost_pct: float = 0.0
    x_rail_newmark_gamma: float = 0.0
    x_carriage_block_mass_g: float = 0.0
    x_carriage_block_delta_g: float = 0.0
    x_carriage_k_rad_n_per_um: float = 0.0
    x_carriage_k_rx_nm_rad: float = 0.0
    x_carriage_k_ry_nm_rad: float = 0.0
    y_rail_ei_x_boost_pct: float = 0.0
    y_carriage_block_mass_g: float = 0.0
    y_carriage_block_delta_g: float = 0.0
    y_carriage_k_rx_nm_rad: float = 0.0
    total_rail_mass_kg: float = 0.0
    frame_damping_ratio: float = 0.042
    total_mass_kg: float = 0.0
    ballast_mass_kg: float = 0.0
    ballast_node_ids: list[str] = field(default_factory=list)
    # Toolhead position in the build volume as fractions of X / Y / Z travel (belt branch lengths)
    toolhead_rel_pos: tuple[float, float, float] = (0.5, 0.5, 0.5)
    suspension_line_tension_n: float = 0.0
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
                # Official Voron 2.4r2: Z-rails bolt onto inner ±Y faces of the 4 vertical columns
                specs = [
                    ("z_rail_fl", "b_fl", "t_fl", "col_g_fl", (0.0, 1.0, 0.0)),
                    ("z_rail_fr", "b_fr", "t_fr", "col_g_fr", (0.0, 1.0, 0.0)),
                    ("z_rail_rl", "b_rl", "t_rl", "col_g_rl", (0.0, -1.0, 0.0)),
                    ("z_rail_rr", "b_rr", "t_rr", "col_g_rr", (0.0, -1.0, 0.0)),
                ]
            elif z_drive == "leadscrew_triple":
                # Official Voron Trident: front Z-rails on +Y inner face of front columns,
                # rear Z-rail on -X side face of rear vertical extrusion (clears rear leadscrew at Y=-32mm)
                specs = [
                    ("z_rail_fl", "b_fl", "t_fl", "z_bed_fl", (0.0, 1.0, 0.0)),
                    ("z_rail_fr", "b_fr", "t_fr", "z_bed_fr", (0.0, 1.0, 0.0)),
                    ("z_rail_rear", "b_rear_mid", "g_rear_mid", "z_bed_r", (-1.0, 0.0, 0.0)),
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
                # Official Voron 2.4r2: 20T Z-belt pulley axis in Z_Belt_Drive_A/B is at (±16.8, ±38.0) mm
                specs = [
                    ("z_drive_fl", "b_fl", "t_fl", "col_g_fl", (16.8, 38.0, 0.0)),
                    ("z_drive_fr", "b_fr", "t_fr", "col_g_fr", (-16.8, 38.0, 0.0)),
                    ("z_drive_rl", "b_rl", "t_rl", "col_g_rl", (16.8, -38.0, 0.0)),
                    ("z_drive_rr", "b_rr", "t_rr", "col_g_rr", (-16.8, -38.0, 0.0)),
                ]
            elif z_drive == "leadscrew_triple":
                # Official Voron Trident: Z leadscrew axes in Z_Motors & Z_Left/Right/Rear are at (±32, +36) & (0, -32) mm
                specs = [
                    ("z_drive_fl", "b_fl", "g_fl", "z_bed_fl", (32.0, 36.0, 0.0)),
                    ("z_drive_fr", "b_fr", "g_fr", "z_bed_fr", (-32.0, 36.0, 0.0)),
                    ("z_drive_rear", "b_rear_mid", "g_rear_mid", "z_bed_r", (0.0, -32.0, 0.0)),
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

    def _frame_centerline_dims_mm(self) -> list[float]:
        """Return frame beam centerline spans [w_cl, d_cl, h_cl] (mm) matching node coordinates."""
        if self.spec.architecture == "corexy":
            w_cl, d_cl, h_cl = corexy_frame_centerlines_m(self.spec)
            return [round(w_cl * 1000.0, 2), round(d_cl * 1000.0, 2), round(h_cl * 1000.0, 2)]
        if self.spec.architecture == "cartesian_bedslinger":
            up_prof = get_profile(self.spec.frame.effective_upright_profile)
            fr_prof = get_profile(self.spec.frame.profile)
            return [
                round(max(self.spec.frame.width_mm - up_prof.width_m * 1000.0, 80.0), 2),
                round(max(self.spec.frame.depth_mm - fr_prof.width_m * 1000.0, 80.0), 2),
                round(max(self.spec.frame.height_mm - fr_prof.height_m * 1000.0, 80.0), 2),
            ]
        if self.spec.architecture == "delta":
            fr_prof = get_profile(self.spec.frame.profile)
            return [
                round(float(self.spec.frame.width_mm), 2),
                round(float(self.spec.frame.depth_mm), 2),
                round(max(self.spec.frame.height_mm - fr_prof.height_m * 1000.0, 80.0), 2),
            ]
        return [
            round(float(self.spec.frame.width_mm), 2),
            round(float(self.spec.frame.depth_mm), 2),
            round(float(self.spec.frame.height_mm), 2),
        ]

    def to_geometry_dict(self) -> dict:
        """Serialize assembly geometry (mm) for the 3D web viewer and AI tools."""
        nmap = self.node_map
        pmat = get_material(self.spec.frame.panel_material)
        fill_id = self.spec.frame.extrusion_filling
        fill_spec = EXTRUSION_FILLINGS.get(fill_id, EXTRUSION_FILLINGS["none"])
        fill_cfg = resolve_extrusion_filling(self.spec)
        bkt_id = self.spec.gantry.bracket_material
        bkt_info = GANTRY_BRACKET_CATALOG.get(bkt_id, GANTRY_BRACKET_CATALOG["abs_printed"])
        bkt_resolved = resolve_gantry_bracket(self.spec)
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
            and int(resolve_bed_mount(self.spec)["mount_points"]) == 3
            and all(nid in nmap for nid in ("bed_t_fl", "bed_t_fr", "bed_t_rear_mount"))
        )
        bed_mount_anchor_nodes = (
            ["bed_t_fl", "bed_t_fr", "bed_t_rear_mount"]
            if is_trident_3pt_subframe
            else (["bed_frame"] * len(mount_pts) if "bed_frame" in nmap else [])
        )
        hid_vis = (self.spec.toolhead.hotend.preset_id or "").lower().strip() or None
        tid_vis = resolve_toolhead_cad_asset_id(self.spec.toolhead.preset_id) or None
        th_acc = resolve_toolhead_hardware_accessories(self.spec)

        # CAD gantry hardware kit visual & stiffness metadata
        hw_kit_id = self.spec.resolve_effective_gantry_hardware_kit()
        hw_kit_meta = GANTRY_HARDWARE_KIT_CATALOG.get(hw_kit_id, {})
        hw_kit_stations: dict[str, str] = dict(hw_kit_meta.get("stations") or {})
        gantry_cad_assets: dict[str, dict[str, Any]] = {}

        if hw_kit_stations and hw_kit_id != "procedural":
            try:
                from printsim.cad_assets.registry import (
                    get_hardware_asset_manifest,
                    get_visual_glb_path,
                )
            except ImportError:
                get_hardware_asset_manifest = None  # type: ignore[assignment]
                get_visual_glb_path = None  # type: ignore[assignment]

            if get_visual_glb_path is not None:
                bkt_mat_obj = get_material(bkt_id)
                station_stiffs = bkt_resolved.get("cad_station_stiffnesses_n_per_mm") or {}
                for st_key, aid in hw_kit_stations.items():
                    if not aid:
                        continue
                    if get_visual_glb_path("gantry_brackets", aid) is None:
                        continue
                    man = (
                        get_hardware_asset_manifest("gantry_brackets", aid)
                        if get_hardware_asset_manifest is not None
                        else None
                    ) or {}
                    raw_mass_kg = float(man.get("mass_kg", 0.0))
                    if (
                        raw_mass_kg > 0.0
                        and not bool(man.get("is_multi_material", False))
                        and float(man.get("reference_density_kg_m3", 0.0)) > 0.0
                    ):
                        raw_mass_kg *= float(bkt_mat_obj.density_kg_m3) / float(
                            man["reference_density_kg_m3"]
                        )
                    st_k = station_stiffs.get(st_key)
                    if st_k is None:
                        ps = man.get("port_stiffness_n_per_mm")
                        if isinstance(ps, dict) and "kx" in ps and "ky" in ps:
                            kx_v = max(float(ps["kx"]), 1e-6)
                            ky_v = max(float(ps["ky"]), 1e-6)
                            k_planar = 2.0 / (1.0 / kx_v + 1.0 / ky_v)
                            if (
                                not bool(man.get("is_multi_material", False))
                                and float(man.get("reference_youngs_modulus_pa", 0.0)) > 0.0
                            ):
                                k_planar *= float(
                                    bkt_mat_obj.effective_modulus_pa(
                                        float(self.spec.frame.chamber_temp_c)
                                    )
                                ) / float(man["reference_youngs_modulus_pa"])
                            st_k = round(k_planar, 2)
                    p0_raw = man.get("port0_mm", [0.0, 0.0, 0.0])
                    p1_raw = man.get("port1_mm", [0.0, 0.0, 0.0])
                    gantry_cad_assets[st_key] = {
                        "asset_id": aid,
                        "visual_glb_url": f"/api/assets/hardware/gantry_brackets/{aid}/visual.glb",
                        "mass_g": round(raw_mass_kg * 1000.0, 2),
                        "stiffness_n_per_mm": st_k,
                        "port0_mm": [float(v) for v in p0_raw[:3]],
                        "port1_mm": [float(v) for v in p1_raw[:3]],
                    }

        return {
            "architecture": self.spec.architecture,
            "drive_mode": self.spec.gantry.drive_mode,
            "belt_routing": self.spec.gantry.belt_routing,
            "double_shear_pulleys": self.spec.gantry.double_shear_pulleys,
            "extrusion_filling": {
                "id": fill_id,
                "name": fill_spec["name"],
                "short_name": fill_spec["short_name"],
                "bonded": fill_cfg["bonded"],
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
                "creep_tension_limit_n": round(float(bkt_resolved["creep_tension_limit_n"]), 2),
                "cnc_machined": bkt_info["cnc_machined"],
                "kit_cost_usd": bkt_info["kit_cost_usd"],
                "carriage_mass_delta_g": round(float(bkt_resolved["carriage_mass_delta_g"]), 2),
                "color_hex": bkt_info["color_hex"],
                "hardware_kit": hw_kit_id,
                "hardware_kit_requested": self.spec.gantry.hardware_kit,
                "hardware_kit_label": str(hw_kit_meta.get("name") or hw_kit_id),
                "cad_backed": bool(bkt_resolved.get("cad_backed", False)),
                "cad_xy_fea_active": bool(
                    any(
                        se.id in ("xy_joint_l", "xy_joint_r")
                        and se.asset_id != "procedural_xy_joint"
                        for se in self.superelements
                    )
                ),
                "xy_joint_fea_active": bool(
                    any(se.id in ("xy_joint_l", "xy_joint_r") for se in self.superelements)
                ),
                "xy_joint_model": (
                    "cad_tet4"
                    if any(
                        se.id in ("xy_joint_l", "xy_joint_r")
                        and se.asset_id != "procedural_xy_joint"
                        for se in self.superelements
                    )
                    else (
                        "analytical_bolted"
                        if any(se.id in ("xy_joint_l", "xy_joint_r") for se in self.superelements)
                        else "rigid"
                    )
                ),
                "cad_stiffness_n_per_mm": round(float(bkt_resolved["stiffness_n_m"]) * 1e-3, 2),
                "cad_assets": gantry_cad_assets,
            },
            "linear_rails": {
                "x_rail_profile": self.spec.gantry.x_rail_profile,
                "y_rail_profile": self.spec.gantry.y_rail_profile,
                "rail_material": self.spec.gantry.rail_material,
                "rail_quality": self.spec.gantry.rail_quality,
                "rail_preload": self.spec.gantry.rail_preload,
                "x_rail_mount": self.spec.gantry.effective_x_rail_mount,
                "y_rail_mount": self.spec.gantry.y_rail_mount,
                "cf_rail_mounting": self.spec.gantry.effective_cf_rail_mounting,
                "dual_x_rails": self.spec.gantry.effective_dual_x_rails,
                "x_rail_width_mm": round(get_profile(self.spec.gantry.x_rail_profile).width_m * 1e3, 1),
                "x_rail_height_mm": round(get_profile(self.spec.gantry.x_rail_profile).height_m * 1e3, 1),
                "y_rail_width_mm": round(get_profile(self.spec.gantry.y_rail_profile).width_m * 1e3, 1),
                "y_rail_height_mm": round(get_profile(self.spec.gantry.y_rail_profile).height_m * 1e3, 1),
                "x_beam_ei_y_boost_pct": round(self.x_rail_ei_y_boost_pct, 1),
                "x_beam_ei_z_boost_pct": round(self.x_rail_ei_z_boost_pct, 1),
                "x_beam_gj_boost_pct": round(self.x_rail_gj_boost_pct, 1),
                "x_rail_newmark_gamma": round(self.x_rail_newmark_gamma, 4),
                "x_beam_insert_mass_g": round(self.x_beam_insert_mass_kg * 1e3, 1),
                "x_carriage_block_mass_g": round(self.x_carriage_block_mass_g, 1),
                "x_carriage_block_delta_g": round(self.x_carriage_block_delta_g, 1),
                "x_carriage_k_rad_n_per_um": round(self.x_carriage_k_rad_n_per_um, 1),
                "x_carriage_k_rx_nm_rad": round(self.x_carriage_k_rx_nm_rad, 1),
                "x_carriage_k_ry_nm_rad": round(self.x_carriage_k_ry_nm_rad, 1),
                "y_rail_ei_x_boost_pct": round(self.y_rail_ei_x_boost_pct, 1),
                "y_carriage_block_mass_g": round(self.y_carriage_block_mass_g, 1),
                "y_carriage_block_delta_g": round(self.y_carriage_block_delta_g, 1),
                "y_carriage_k_rx_nm_rad": round(self.y_carriage_k_rx_nm_rad, 1),
                "total_rail_mass_g": round(self.total_rail_mass_kg * 1e3, 1),
            },
            "panels": {
                "enabled": panels_active,
                "top_panel": bool(self.spec.frame.top_panel),
                "mounting": self.spec.frame.panel_mounting if panels_active else "none",
                "material_id": self.spec.frame.panel_material,
                "material_name": pmat.name,
                "side_material_id": self.spec.frame.side_panel_material or self.spec.frame.panel_material,
                "side_material_name": (
                    get_material(self.spec.frame.side_panel_material).name
                    if self.spec.frame.side_panel_material
                    else pmat.name
                ),
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
            "frame_dims_mm": self._frame_centerline_dims_mm(),
            "frame_outer_dims_mm": [
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
            "suspension": (
                {
                    "line": self.spec.frame.suspension_line,
                    "length_mm": round(float(self.spec.frame.suspension_length_mm), 1),
                    "line_tension_n": round(self.suspension_line_tension_n, 2),
                    "lines": [
                        {
                            "anchor_mm": (nmap[sp.shell_node_id].position_m * 1000.0).tolist(),
                            "attach_mm": (nmap[sp.frame_node_id].position_m * 1000.0).tolist(),
                            "attach_node_id": sp.frame_node_id,
                            "anchor_node_id": sp.shell_node_id,
                        }
                        for sp in self.fastener_springs
                        if sp.bay == "feet"
                        and sp.shell_node_id in nmap
                        and sp.frame_node_id in nmap
                    ],
                }
                if self.spec.frame.feet_type == "suspended_lines"
                else None
            ),
            "ballast_mass_kg": round(self.ballast_mass_kg, 3),
            "ballast_location": self.spec.frame.ballast_location,
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
            "cog_offset_mm": [round(v, 2) for v in th_acc["effective_cog_offset_mm"]],
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
                "hotend_mass_g": round(float(th_acc["effective_hotend_mass_g"]), 2),
                "toolhead_body_mass_g": round(float(th_acc["effective_body_mass_g"]), 2),
                "assembled_toolhead_mass_g": round(float(th_acc["effective_total_mass_g"]), 2),
                "effective_cog_offset_mm": [round(v, 2) for v in th_acc["effective_cog_offset_mm"]],
                "accessory_mass_delta_g": float(th_acc["accessory_mass_delta_g"]),
                "nozzle_shield_mode": str(th_acc["nozzle_shield_mode"]),
                "nozzle_shield_options": list(th_acc["nozzle_shield_options"]),
                "hotend_preset_id": self.spec.toolhead.hotend.preset_id,
                "hotend_mount_style": self.spec.toolhead.hotend.mount_style,
                "hotend_stiffness_n_per_mm": round(
                    float(
                        next(
                            (
                                (
                                    2.0
                                    / (
                                        1.0 / max(float(se.manifest["port_stiffness_n_per_mm"]["kx"]), 1e-6)
                                        + 1.0 / max(float(se.manifest["port_stiffness_n_per_mm"]["ky"]), 1e-6)
                                    )
                                )
                                for se in self.superelements
                                if se.category in ("assemblies", "hotends")
                                and isinstance(se.manifest.get("port_stiffness_n_per_mm"), dict)
                                and float(se.manifest["port_stiffness_n_per_mm"].get("kx", 0.0)) > 0.0
                                and float(se.manifest["port_stiffness_n_per_mm"].get("ky", 0.0)) > 0.0
                            ),
                            self.spec.toolhead.hotend.structural_stiffness_n_per_mm,
                        )
                    ),
                    2,
                ),
                "hotend_overall_length_mm": round(
                    float(self.spec.toolhead.hotend.overall_length_mm), 2
                ),
                "stage_stiffnesses_n_per_mm": {
                    k: round(float(v), 2)
                    for k, v in self.toolhead_stage_stiffnesses_n_per_mm.items()
                },
                "use_cad_superelements": bool(self.spec.use_cad_superelements),
                "use_hotend_bottom_brace": bool(
                    getattr(self.spec.toolhead, "use_hotend_bottom_brace", True)
                ),
                "cad_hotend_brace_available": bool(
                    bool(self.spec.use_cad_superelements)
                    and tid_vis
                    and hid_vis
                    and _has_cad_asset("assemblies", f"{tid_vis}__{hid_vis}")
                ),
                "cad_toolhead_self_contained": bool(
                    any(se.category == "toolheads" for se in self.superelements)
                ),
                "cad_hotend_self_contained": bool(
                    any(se.category in ("hotends", "assemblies") for se in self.superelements)
                ),
                "cad_hotend_braced": bool(
                    any(se.category == "assemblies" for se in self.superelements)
                ),
                "cad_includes_backplate": bool(
                    any(se.category == "toolheads" for se in self.superelements)
                ),
                "cad_superelements_active": [se.to_summary_dict() for se in self.superelements],
                "hotend_visual_glb_url": (
                    f"/api/assets/hardware/hotends/{hid_vis}/visual.glb"
                    if (hid_vis and _has_cad_visual("hotends", hid_vis))
                    else None
                ),
                "toolhead_visual_glb_url": (
                    f"/api/assets/hardware/toolheads/{tid_vis}/visual.glb"
                    if (tid_vis and _has_cad_visual("toolheads", tid_vis))
                    else None
                ),
                # Port 0 / Port 1 datums of visual-only CAD toolheads (no structural superelement),
                # e.g. the printed-geometry VzBot Vz-Printhead on a top-mounted MGN9H
                "toolhead_visual_ports_mm": _cad_visual_ports_mm("toolheads", tid_vis),
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


# Subsystem builders and helpers extracted to sibling modules (re-exported for backwards compatibility)
from printsim.geometry._builder_bed import (  # noqa: E402
    _add_corexy_bed_subframe_members,
    _apply_corexy_z_guideways,
    _apply_smooth_rod_guideway,
    _apply_z_guideway,
    _attach_bed_mount_compliance,
    _compute_corexy_z_drive_stiffness,
    _compute_flying_z_joint_spring_stiffness,
    _compute_mount_offsets_m,
    _compute_z_carriage_spring_stiffness,
    _make_bed_virtual_material,
    _rebalance_virtual_bed_rail_mass,
    _setup_corexy_z_drive_nodes_and_bed,
)
from printsim.geometry._builder_gantry import (  # noqa: E402
    _BASELINE_X_CARRIAGE_BLOCK_MASS_G,
    _BASELINE_Y_CARRIAGE_PAIR_MASS_G,
    _apply_bedslinger_x_rail_backer_sandwich,
    _apply_bolted_linear_rail,
    _apply_bolted_rail_with_opposite_backer,
    _apply_cf_two_wall_clamped_inserts,
    _apply_corexy_gantry_rails,
    _build_cad_corner_mount_superelement,
    _build_cad_trident_bed_arm_superelement,
    _build_cad_xy_joint_superelement,
    _build_procedural_xy_joint_superelement,
    _build_cad_z_joint_superelement,
    _build_y_carriage_bearing_superelement,
    _compute_newmark_rail_gamma,
    _make_x_backer_profile,
    _member_local_axes,
    _oriented_bare_rail_profile,
    _populate_linear_rail_telemetry,
    _resolve_cad_corner_mount_assets,
    _resolve_cad_trident_bed_arm_assets,
    _resolve_cad_xy_joint_assets,
    _resolve_cad_z_joint_asset,
    _x_rail_mount_normals,
    _y_rail_mount_normals,
    corexy_x_beam_span_m,
    corexy_x_rail_len_m,
    x_carriage_block_mass_delta_g,
    y_carriage_block_mass_delta_g,
)
from printsim.geometry._builder_panels import (  # noqa: E402
    _add_corexy_shell_panels,
    _add_flat_bar_brace_kit,
    _add_open_ring_shell_plate,
    _add_shear_panel_bay,
    _add_top_and_front_braces,
    _add_trident_backbone,
    _find_or_split_frame_node,
    _make_panel_virtual_material,
    _panel_shear_params,
    _point_to_segment_dist_m,
    _split_member_at_node,
    _trident_backbone_cost_usd,
    bolted_flange_rot_stiffness_per_m,
    front_brace_dimensions_mm,
    panel_mechanical_properties,
    top_brace_dimensions_mm,
    trident_backbone_dimensions_mm,
)
from printsim.geometry._builder_toolhead import (  # noqa: E402
    _LEGACY_PRE_FEA_HOTEND_STIFFNESS,
    _MBT_SHIELD_MASS_G,
    _MBT_SHIELD_REL_NOZ_MM,
    _MBT_TI_BRACE_MASS_G,
    _MBT_TI_BRACE_REL_NOZ_MM,
    _TRICORN_STOCK_SHIELD_MASS_G,
    _TRICORN_STOCK_SHIELD_REL_NOZ_MM,
    _attach_toolhead_cog_and_nozzle,
    _effective_toolhead_mass_kg,
    _has_cad_asset,
    _inject_cad_toolhead_superelements,
    _invert_6x6_spd,
    _load_cad_superelement,
    _series_combine_superelement_kbb,
    _skew_3x3,
    compute_toolhead_link_props,
    resolve_toolhead_cad_asset_id,
    resolve_toolhead_hardware_accessories,
)


def _apply_corner_joint_compliance(
    members: list[StructuralMember],
    spec: PrinterSpec,
    col_length_m: float,
) -> None:
    """Apply semi-rigid corner joint end-fixity knockdown to unpaneled vertical frame columns."""
    jtype = spec.frame.corner_joint_type
    bolted_panels = spec.frame.enclosed_panels and spec.frame.panel_mounting == "bolted_structural"
    if bolted_panels:
        return
    ov_cj = spec.model_overrides.corner_joint
    has_cj_override = (
        ov_cj.k_t_n_per_mm is not None
        or ov_cj.k_r_inplane_nm_rad is not None
        or ov_cj.k_r_torsion_nm_rad is not None
    )
    if jtype == "rigid" and not has_cj_override:
        return

    up_prof_id = spec.frame.effective_upright_profile
    cj = resolve_corner_joint(spec)
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
    fill_cfg = resolve_extrusion_filling(spec)
    joint_fill_boost = 1.0
    if fill_cfg["is_active"] and bool(fill_cfg["bonded"]):
        e_fill = float(fill_cfg["youngs_modulus_pa"])
        # Bonded cast mineral/epoxy core backs up hollow T-slot corner flanges against bolt-bearing ovalization
        joint_fill_boost = 1.0 + 0.45 * min(1.0, e_fill / 25.0e9)
    kr_in = float(cj["k_r_inplane_nm_rad"]) * joint_fill_boost
    kr_tor = float(cj["k_r_torsion_nm_rad"]) * joint_fill_boost
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


def _apply_extrusion_filling(model: AssemblyModel) -> None:
    """Apply stationary extrusion cavity filling mass, bonded stiffness, and structural damping."""
    fill_cfg = resolve_extrusion_filling(model.spec)
    bkt_resolved = resolve_gantry_bracket(model.spec)
    cj_resolved = resolve_corner_joint(model.spec)

    nmap = model.node_map
    total_fill_mass_kg = 0.0
    filled_count = 0

    if fill_cfg["is_active"]:
        rho_fill = float(fill_cfg["density_kg_m3"])
        e_fill = float(fill_cfg["youngs_modulus_pa"])
        is_bonded = bool(fill_cfg["bonded"])
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
            filled_count += 1
            if rho_fill > 0.0:
                m_fill = rho_fill * a_cav * length
                member.fill_mass_kg = m_fill
                total_fill_mass_kg += m_fill
                base_rho = member.material.density_kg_m3
                member.material = member.material.model_copy(
                    update={"density_kg_m3": base_rho + rho_fill * (a_cav / max(a_ext, 1e-12))}
                )

            if is_bonded and e_fill > 0.0:
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
    fill_boost = (
        float(fill_cfg["damping_boost"])
        if (not fill_cfg["is_active"] or filled_count > 0)
        else 0.0
    )
    bracket_damping = float(bkt_resolved["damping_delta"])
    joint_damping = float(cj_resolved["damping_delta"])
    model.frame_damping_ratio = max(
        0.025, min(0.12, 0.039 + joint_damping + fill_boost + bracket_damping)
    )


_MOUNT_BASE_ROLES = ("foot", "frame", "tower")
_FRAME_STRUCTURE_ROLES = ("foot", "frame", "tower", "mount")
_G_ACC = 9.80665  # standard gravity (m/s^2)


def _mount_base_nodes(model: AssemblyModel) -> list[StructuralNode]:
    """Ground-contact nodes of the frame (the fixed feet before any compliant mount is applied)."""
    return [n for n in model.nodes if n.fixed and n.role in _MOUNT_BASE_ROLES]


def _node_above(model: AssemblyModel, base: StructuralNode) -> StructuralNode:
    """Highest frame-structure node directly above ``base`` (same XY within 1 mm), else ``base``."""
    best = base
    for n in model.nodes:
        if n.role not in _FRAME_STRUCTURE_ROLES or n.fixed and n is not base:
            continue
        dxy = float(np.hypot(*(n.position_m[:2] - base.position_m[:2])))
        if dxy <= 1e-3 and float(n.position_m[2]) > float(best.position_m[2]) + 1e-6:
            best = n
    return best


def _frame_top_nodes(model: AssemblyModel) -> list[StructuralNode]:
    """Frame-structure nodes within 1 mm of the highest frame-structure elevation."""
    cand = [n for n in model.nodes if n.role in _FRAME_STRUCTURE_ROLES]
    if not cand:
        return []
    z_max = max(float(n.position_m[2]) for n in cand)
    return [n for n in cand if float(n.position_m[2]) >= z_max - 1e-3]


def _apply_frame_ballast(model: AssemblyModel) -> None:
    """Lump ``frame.ballast_mass_kg`` of dead weight onto base, top, or all frame corner nodes.

    Must run before `_apply_feet_grounding` so base ballast lands on the foot nodes: on a
    rigidly bolted frame those stay fixed (the ballast is then dynamically inert, exactly as
    weight bolted to the bench would be), while on compliant feet or suspension lines it rides
    above the mount springs and lowers the rigid-body mount modes.
    """
    m_ballast = max(float(model.spec.frame.ballast_mass_kg), 0.0)
    if m_ballast <= 0.0:
        return
    loc = model.spec.frame.ballast_location
    base_nodes = _mount_base_nodes(model)
    top_nodes = _frame_top_nodes(model)
    if loc == "top":
        targets = top_nodes or base_nodes
    elif loc == "distributed":
        seen: set[str] = set()
        targets = []
        for n in base_nodes + top_nodes:
            if n.id not in seen:
                seen.add(n.id)
                targets.append(n)
    else:
        targets = base_nodes or top_nodes
    if not targets:
        return
    per_node_kg = m_ballast / len(targets)
    for n in targets:
        n.lumped_mass_kg += per_node_kg
    model.ballast_mass_kg = m_ballast
    model.ballast_node_ids = [n.id for n in targets]


def _apply_feet_grounding(model: AssemblyModel) -> None:
    """Replace rigid fixed feet with compliant benchtop isolator springs or overhead suspension lines."""
    f_cfg = resolve_frame_feet(model.spec)
    if not f_cfg["is_compliant"]:
        return

    is_suspended = bool(f_cfg.get("is_suspended", False))
    k_normal_n_m = float(f_cfg["k_axial_n_m"])
    k_shear_n_m = float(f_cfg["k_lateral_n_m"])
    if k_normal_n_m <= 0.0:
        k_normal_n_m = 1.0e8
    if k_shear_n_m <= 0.0:
        k_shear_n_m = 1.0e8
    # A rope or line carries no moment; elastomer feet resist tilt over their ~40 mm contact patch.
    k_rot_nm_rad = 0.0 if is_suspended else max(k_normal_n_m * (0.020**2), 40.0)
    line_length_m = float(f_cfg.get("line_length_m", 0.0))

    grounded_any = False
    foot_nodes: list[StructuralNode] = []
    for node in _mount_base_nodes(model):
        node.fixed = False
        foot_nodes.append(node)
        # Suspension lines pick the frame up at the corner directly above each foot (top frame
        # corners on a CoreXY cube / delta towers); frames with no upright there hang by the base.
        attach = _node_above(model, node) if is_suspended else node
        gnd_pos = attach.position_m.copy()
        gnd_pos[2] += line_length_m if is_suspended else -0.015
        gnd_id = f"gnd_{node.id}"
        model.nodes.append(StructuralNode(gnd_id, gnd_pos, fixed=True, role="ground"))
        model.fastener_springs.append(
            FastenerSpring(
                id=f"feet_spring_{node.id}",
                bay="feet",
                shell_node_id=gnd_id,
                frame_node_id=attach.id,
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


def _finalize_suspension_tension(model: AssemblyModel) -> None:
    """Set each suspension line's lateral stiffness to the pendulum geometric stiffness ``T / L``.

    With the printer's weight shared equally by ``n`` near-vertical lines, ``T = M g / n`` and
    the summed lateral stiffness ``M g / L`` reproduces the simple-pendulum sway frequency
    ``(1 / 2 pi) sqrt(g / L)`` independent of mass. Runs after `_finalize_mass_and_cost` because
    it needs the total suspended mass (frame, hardware, ballast).
    """
    f_cfg = resolve_frame_feet(model.spec)
    if not f_cfg.get("is_suspended", False):
        return
    lines = [sp for sp in model.fastener_springs if sp.bay == "feet"]
    if not lines:
        return
    tension_n = max(float(model.total_mass_kg), 0.1) * _G_ACC / len(lines)
    model.suspension_line_tension_n = tension_n
    if not f_cfg.get("lateral_from_tension", False):
        return
    k_lat_n_m = tension_n / max(float(f_cfg["line_length_m"]), 0.05)
    for sp in lines:
        sp.k_shear_n_m = k_lat_n_m


def _apply_mount_and_feet_damping(model: AssemblyModel) -> None:
    """Apply damping increments from frame feet isolators and bed mounting hardware."""
    f_cfg = resolve_frame_feet(model.spec)
    b_cfg = resolve_bed_mount(model.spec)
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
    model.toolhead_rel_pos = (
        float(toolhead_rel_pos[0]),
        float(toolhead_rel_pos[1]),
        float(toolhead_rel_pos[2]),
    )

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

    _apply_bedslinger_x_rail_backer_sandwich(model)
    _populate_linear_rail_telemetry(model)
    _inject_cad_toolhead_superelements(model)
    _apply_frame_ballast(model)
    _apply_feet_grounding(model)
    _apply_extrusion_filling(model)
    _apply_mount_and_feet_damping(model)
    _finalize_mass_and_cost(model)
    _finalize_suspension_tension(model)
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
    bkt_resolved = resolve_gantry_bracket(spec)
    bkt_delta_kg = float(bkt_resolved["carriage_mass_delta_g"]) * 1e-3
    _, y_block_delta_g = y_carriage_block_mass_delta_g(spec)
    y_block_delta_kg = y_block_delta_g * 1e-3
    y_carriage_total_kg = max(
        (spec.gantry.carriage_mass_g * 1e-3) + bkt_delta_kg + y_block_delta_kg,
        0.04 if spec.architecture != "delta" else 0.01,
    )
    th_mass_kg = _effective_toolhead_mass_kg(spec)
    model = AssemblyModel(
        spec=spec,
        nodes=nodes,
        members=members,
        toolhead_node_id=toolhead_id,
        bed_center_m=np.array([w * 0.5, d * 0.5, 0.06]),
        build_origin_m=np.array([(w - bx) * 0.5, (d - by) * 0.5, 0.07]),
        x_moving_mass_kg=th_mass_kg,
        y_moving_mass_kg=th_mass_kg + y_carriage_total_kg,
        y_carriage_joints_mass_kg=y_carriage_total_kg,
        y_carriage_block_delta_g=y_block_delta_g,
        toolhead_moving_mass_kg=th_mass_kg,
    )
    _apply_bedslinger_x_rail_backer_sandwich(model)
    _populate_linear_rail_telemetry(model)
    _inject_cad_toolhead_superelements(model)
    _apply_frame_ballast(model)
    _apply_feet_grounding(model)
    _apply_extrusion_filling(model)
    _apply_mount_and_feet_damping(model)
    _finalize_mass_and_cost(model)
    _finalize_suspension_tension(model)
    return model


def _has_cad_visual(category: str, asset_id: str | None) -> bool:
    """True when a baked ``visual.glb`` exists (structural or visual-only bundle)."""
    if not asset_id:
        return False
    if _has_cad_asset(category, asset_id):
        return True
    try:
        from printsim.cad_assets.registry import has_visual_asset
    except ImportError:
        return False
    return bool(has_visual_asset(category, asset_id))


def _cad_visual_ports_mm(category: str, asset_id: str | None) -> dict[str, list[float]] | None:
    """Return ``{"port0_mm", "port1_mm"}`` from a visual-only bundle manifest, else ``None``."""
    if not asset_id:
        return None
    try:
        from printsim.cad_assets.registry import get_hardware_asset_manifest
    except ImportError:
        return None
    man = get_hardware_asset_manifest(category, asset_id)
    if not man or man.get("has_superelement", True):
        return None
    p0, p1 = man.get("port0_mm"), man.get("port1_mm")
    if not (isinstance(p0, list) and isinstance(p1, list) and len(p0) == 3 and len(p1) == 3):
        return None
    return {"port0_mm": [float(v) for v in p0], "port1_mm": [float(v) for v in p1]}


def corexy_frame_centerlines_m(spec: PrinterSpec) -> tuple[float, float, float]:
    """Return CoreXY beam centerline spans (w_cl, d_cl, h_cl) in meters from outer edge-to-edge dims.

    ``spec.frame.width_mm``, ``depth_mm``, and ``height_mm`` specify outer edge-to-edge
    frame dimensions (e.g., 460 × 460 × 480 mm for Voron 2.4 300, 460 × 460 × 500 mm for
    Voron Trident 300). Deducting one upright profile cross-section in X/Y and one
    horizontal frame profile height in Z yields exact beam centerline spans so the FEA
    model, 3D viewport outer faces, BOM cut lengths, and DXF panels agree to the millimeter.
    """
    up_prof = get_profile(spec.frame.effective_upright_profile)
    fr_prof = get_profile(spec.frame.profile)
    w_cl = max(spec.frame.width_mm * 1e-3 - up_prof.width_m, 0.08)
    d_cl = max(spec.frame.depth_mm * 1e-3 - up_prof.height_m, 0.08)
    h_cl = max(spec.frame.height_mm * 1e-3 - fr_prof.height_m, 0.08)
    return w_cl, d_cl, h_cl


# Penalty stiffness tying coincident top-ring / gantry-plane corner nodes into one rigid joint
# (same order as the CORNER_JOINT_BASE_2020 "rigid" reference: 1e10 N/m, 1e9 N*m/rad).
_RIGID_TIE_K_DIAG_6DOF = (1.0e10, 1.0e10, 1.0e10, 1.0e9, 1.0e9, 1.0e9)


def _add_on_top_frame_y_rail_chains(
    members: list[StructuralMember],
    nodes: list[StructuralNode],
    fastener_springs: list[FastenerSpring],
    fp: str,
    fm: str,
    h: float,
) -> None:
    """Build gantry-on-top-frame Y-rail beams as the single top-ring Y extrusion per side.

    With ``frame.gantry_on_top_frame`` the gantry plane coincides with the top ring, so each side
    is one continuous 2020 + bolted-rail member chain from the front corner through any top Z-rod /
    Z-guide anchor nodes and the stationary rail-carriage station to the rear corner. The coincident
    ``t_*`` top corners are rigidly tied to the ``g_*`` gantry corners (one physical corner joint).
    """
    nmap = {n.id: n for n in nodes}
    for side, x_tag in (("l", "l"), ("r", "r")):
        corner_f = f"g_f{x_tag}"
        corner_r = f"g_r{x_tag}"
        x_line = float(nmap[corner_f].position_m[0])
        y_f = float(nmap[corner_f].position_m[1])
        y_r = float(nmap[corner_r].position_m[1])
        mid_ids = [
            n.id
            for n in nodes
            if n.id.startswith("t_")
            and n.id not in ("t_fl", "t_fr", "t_rl", "t_rr")
            and abs(float(n.position_m[0]) - x_line) < 1e-6
            and abs(float(n.position_m[2]) - h) < 1e-6
            and y_f + 1e-6 < float(n.position_m[1]) < y_r - 1e-6
        ]
        car_id = f"y_rail_car_{side}"
        y_car = float(nmap[car_id].position_m[1])
        # Merge a frame anchor that coincides with the carriage station (avoid zero-length beams)
        chain_mid: list[str] = []
        for nid in sorted(mid_ids, key=lambda i: float(nmap[i].position_m[1])):
            if abs(float(nmap[nid].position_m[1]) - y_car) < 1e-3:
                fastener_springs.append(
                    FastenerSpring(
                        id=f"rigid_tie_{nid}",
                        bay="rigid_tie",
                        shell_node_id=nid,
                        frame_node_id=car_id,
                        k_shear_n_m=_RIGID_TIE_K_DIAG_6DOF[0],
                        k_normal_n_m=_RIGID_TIE_K_DIAG_6DOF[2],
                        k_rot_nm_rad=_RIGID_TIE_K_DIAG_6DOF[3],
                        k_diag_6dof=_RIGID_TIE_K_DIAG_6DOF,
                    )
                )
                continue
            chain_mid.append(nid)
        chain = [corner_f]
        inserted = False
        for nid in chain_mid:
            if not inserted and float(nmap[nid].position_m[1]) > y_car:
                chain.append(car_id)
                inserted = True
            chain.append(nid)
        if not inserted:
            chain.append(car_id)
        chain.append(corner_r)
        i_car = chain.index(car_id)
        n_seg = len(chain) - 1
        for i in range(n_seg):
            # Front of the carriage: y_rail_{s}_f, _f1, _f2 ...; behind it: ..., _r1, y_rail_{s}_r
            if i < i_car:
                mid = f"y_rail_{side}_f{i}" if i else f"y_rail_{side}_f"
            else:
                k_back = n_seg - 1 - i
                mid = f"y_rail_{side}_r{k_back}" if k_back else f"y_rail_{side}_r"
            _add_member(members, mid, chain[i], chain[i + 1], fp, fm, role="rail")
    for tag in ("fl", "fr", "rl", "rr"):
        fastener_springs.append(
            FastenerSpring(
                id=f"rigid_tie_t_{tag}",
                bay="rigid_tie",
                shell_node_id=f"t_{tag}",
                frame_node_id=f"g_{tag}",
                k_shear_n_m=_RIGID_TIE_K_DIAG_6DOF[0],
                k_normal_n_m=_RIGID_TIE_K_DIAG_6DOF[2],
                k_rot_nm_rad=_RIGID_TIE_K_DIAG_6DOF[3],
                k_diag_6dof=_RIGID_TIE_K_DIAG_6DOF,
            )
        )


def _build_corexy(
    spec: PrinterSpec,
    toolhead_rel_pos: tuple[float, float, float],
    panel_mesh_n: int = 2,
) -> AssemblyModel:
    w, d, h = corexy_frame_centerlines_m(spec)
    bx, by, bz = np.array(spec.kinematics.build_volume_mm) * 1e-3
    u, v, w_rel = np.clip(toolhead_rel_pos, 0.05, 0.95)

    xy_motor_kg = spec.kinematics.xy_stepper.motor_mass_g * 1e-3
    z_motor_kg = spec.kinematics.z_stepper.motor_mass_g * 1e-3
    bkt_info = GANTRY_BRACKET_CATALOG.get(
        spec.gantry.bracket_material, GANTRY_BRACKET_CATALOG["abs_printed"]
    )
    bkt_resolved = resolve_gantry_bracket(spec)
    bkt_delta_kg = float(bkt_resolved["carriage_mass_delta_g"]) * 1e-3
    if spec.gantry.custom_bracket_override is not None:
        bkt_delta_kg += float(spec.gantry.custom_bracket_override.mass_delta_g) * 1e-3
    _, y_block_delta_g = y_carriage_block_mass_delta_g(spec)
    y_block_delta_kg = y_block_delta_g * 1e-3
    y_carriage_total_kg = max(
        spec.gantry.carriage_mass_g * 1e-3 + bkt_delta_kg + y_block_delta_kg, 0.04
    )
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
    on_top_frame = bool(spec.frame.gantry_on_top_frame) and not is_flying
    trident_backbone_active = bool(spec.frame.trident_backbone and not is_flying)
    z_drive = spec.bed.z_drive
    bed_cfg = resolve_bed_mount(spec)
    t_bed_m = float(spec.bed.plate_thickness_mm) * 1e-3
    h_standoff_m = float(bed_cfg.get("standoff_height_mm", 14.0)) * 1e-3
    half_subframe_m = 0.5 * min(get_profile(spec.frame.profile).height_m, 0.020)
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
        # Gantry plane near top of CoreXY frame (or exactly on the top ring when the Y rails
        # bolt straight onto the top-perimeter Y extrusions, e.g. VzBot inner-face MGN12)
        z_gantry = h if on_top_frame else h - top_clear_m
        z_min_req = max(0.08, 0.030 + t_bed_m + h_standoff_m + half_subframe_m)
        z_min = min(z_min_req, 0.5 * z_gantry)
        z_max = max(z_min + 0.01, z_gantry - 0.045)
        z_stroke = min(bz, max(z_gantry - 0.14, 0.02))
        z_bed = float(np.clip(z_gantry - 0.045 - w_rel * z_stroke, z_min, z_max))
    z_subframe = max(0.025, z_bed - t_bed_m - h_standoff_m - half_subframe_m)

    is_awd = spec.gantry.drive_mode == "awd"
    if bkt_info.get("manufacturing_process") == "slm":
        # Topology-optimized hollow SLM motor mounts save ~40% volume vs solid 3-axis CNC billet,
        # scaled by alloy density relative to SLM AlSi10Mg (2670 kg/m³).
        bkt_mat_obj = get_material(spec.gantry.bracket_material)
        cnc_motor_mount_kg = 0.065 * (bkt_mat_obj.density_kg_m3 / 2670.0)
    else:
        cnc_motor_mount_kg = 0.11 if bool(bkt_info["cnc_machined"]) else 0.0
    rear_motor_kg = xy_motor_kg + cnc_motor_mount_kg
    front_motor_kg = (xy_motor_kg + 0.03 + cnc_motor_mount_kg) if is_awd else 0.0
    front_role = "mount" if is_awd else "frame"
    bed_center = np.array([w * 0.5, d * 0.5, z_bed])
    fastener_springs: list[FastenerSpring] = []


    k_z_drive_col_n_m = _compute_corexy_z_drive_stiffness(
        spec, z_gantry, z_bed, is_flying, trident_backbone_active
    )
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
    gantry_superelements: list[AttachedSuperelement] = []
    corner_mount_assets = _resolve_cad_corner_mount_assets(spec)
    cad_z_joint_asset = _resolve_cad_z_joint_asset(spec) if is_flying else None
    g_rear_l_mass = 0.0 if "rl" in corner_mount_assets else rear_motor_kg
    g_rear_r_mass = 0.0 if "rr" in corner_mount_assets else rear_motor_kg
    g_front_l_mass = 0.0 if "fl" in corner_mount_assets else front_motor_kg
    g_front_r_mass = 0.0 if "fr" in corner_mount_assets else front_motor_kg

    if is_flying:
        # Voron 2.4 flying gantry: split vertical columns at col_g_* and couple separate
        # floating gantry corners g_* via 6-DOF Z-belt + MGN rail carriage + GE5C Z-joint springs
        # (in series with the baked 3D CAD Z-joint superelement when a CAD kit provides one).
        y_bed_f = d * 0.14
        y_bed_r = d * 0.86
        nodes.extend(
            [
                StructuralNode("col_g_fl", np.array([0.0, 0.0, z_gantry]), role="frame"),
                StructuralNode("col_g_fr", np.array([w, 0.0, z_gantry]), role="frame"),
                StructuralNode("col_g_rl", np.array([0.0, d, z_gantry]), role="frame"),
                StructuralNode("col_g_rr", np.array([w, d, z_gantry]), role="frame"),
                StructuralNode("g_fl", np.array([0.0, 0.0, z_gantry]), lumped_mass_kg=g_front_l_mass, role=front_role if is_awd else "gantry"),
                StructuralNode("g_fr", np.array([w, 0.0, z_gantry]), lumped_mass_kg=g_front_r_mass, role=front_role if is_awd else "gantry"),
                StructuralNode("g_rl", np.array([0.0, d, z_gantry]), lumped_mass_kg=g_rear_l_mass, role="gantry"),
                StructuralNode("g_rr", np.array([w, d, z_gantry]), lumped_mass_kg=g_rear_r_mass, role="gantry"),
                StructuralNode("b_bed_fl", np.array([0.0, y_bed_f, 0.0]), role="frame"),
                StructuralNode("b_bed_rl", np.array([0.0, y_bed_r, 0.0]), role="frame"),
                StructuralNode("b_bed_fr", np.array([w, y_bed_f, 0.0]), role="frame"),
                StructuralNode("b_bed_rr", np.array([w, y_bed_r, 0.0]), role="frame"),
            ]
        )
        k_zj_n, k_zj_s, k_zj_r = _compute_flying_z_joint_spring_stiffness(
            spec, k_z_drive_col_n_m
        )
        zj_corner_coords = {
            "fl": (np.array([0.0, 0.0, z_gantry]), 0.0),
            "fr": (np.array([w, 0.0, z_gantry]), 0.0),
            "rl": (np.array([0.0, d, z_gantry]), math.pi),
            "rr": (np.array([w, d, z_gantry]), math.pi),
        }
        for tag in ("fl", "fr", "rl", "rr"):
            if cad_z_joint_asset is not None:
                c_pos, rot_z = zj_corner_coords[tag]
                mid_id = f"z_joint_mid_{tag}"
                nodes.append(StructuralNode(mid_id, c_pos.copy(), lumped_mass_kg=1e-7, role="gantry"))
                gantry_superelements.append(
                    _build_cad_z_joint_superelement(
                        f"cad_z_joint_{tag}",
                        cad_z_joint_asset,
                        (f"col_g_{tag}", mid_id),
                        rot_z,
                        spec,
                    )
                )
                fastener_springs.append(
                    FastenerSpring(
                        id=f"z_joint_{tag}",
                        bay="z_joint",
                        shell_node_id=mid_id,
                        frame_node_id=f"g_{tag}",
                        k_shear_n_m=k_zj_s,
                        k_normal_n_m=k_zj_n,
                        k_rot_nm_rad=k_zj_r,
                        normal_vec=(0.0, 0.0, 1.0),
                        offset_m=(0.0, 0.0, 0.0),
                    )
                )
            else:
                fastener_springs.append(
                    FastenerSpring(
                        id=f"z_joint_{tag}",
                        bay="z_joint",
                        shell_node_id=f"col_g_{tag}",
                        frame_node_id=f"g_{tag}",
                        k_shear_n_m=k_zj_s,
                        k_normal_n_m=k_zj_n,
                        k_rot_nm_rad=k_zj_r,
                        normal_vec=(0.0, 0.0, 1.0),
                        offset_m=(0.0, 0.0, 0.0),
                    )
                )
    else:
        nodes.extend(
            [
                # Fixed-gantry elevation nodes directly on the 4 vertical columns
                StructuralNode("g_fl", np.array([0.0, 0.0, z_gantry]), lumped_mass_kg=g_front_l_mass, role=front_role),
                StructuralNode("g_fr", np.array([w, 0.0, z_gantry]), lumped_mass_kg=g_front_r_mass, role=front_role),
                StructuralNode("g_rl", np.array([0.0, d, z_gantry]), lumped_mass_kg=g_rear_l_mass, role="mount"),
                StructuralNode("g_rr", np.array([w, d, z_gantry]), lumped_mass_kg=g_rear_r_mass, role="mount"),
            ]
        )

    # Wire baked 3D CAD corner motor-mount & front-idler superelements (g_* -> pulley_*)
    if corner_mount_assets:
        corner_base_pos = {
            "fl": (np.array([0.0, 0.0, z_gantry]), front_motor_kg),
            "fr": (np.array([w, 0.0, z_gantry]), front_motor_kg),
            "rl": (np.array([0.0, d, z_gantry]), rear_motor_kg),
            "rr": (np.array([w, d, z_gantry]), rear_motor_kg),
        }
        for tag in ("fl", "fr", "rl", "rr"):
            asset = corner_mount_assets.get(tag)
            if asset is None:
                continue
            p_corner, m_pulley = corner_base_pos[tag]
            p1_cad_m = np.asarray(asset.port_coords_m[1], dtype=float)  # type: ignore[attr-defined]
            p_pulley = p_corner + p1_cad_m
            # Keep pulley shaft cleanly inside the CoreXY envelope for tiny custom frames
            p_pulley[0] = float(np.clip(p_pulley[0], -0.02, w + 0.02))
            p_pulley[1] = float(np.clip(p_pulley[1], -0.02, d + 0.02))
            pulley_id = f"pulley_{tag}"
            nodes.append(
                StructuralNode(
                    pulley_id,
                    p_pulley,
                    lumped_mass_kg=m_pulley if m_pulley > 0.0 else 1e-7,
                    role="mount" if m_pulley > 0.0 else "gantry",
                )
            )
            gantry_superelements.append(
                _build_cad_corner_mount_superelement(
                    f"corner_mount_{tag}",
                    asset,
                    (f"g_{tag}", pulley_id),
                    p_corner,
                    p_pulley,
                    spec,
                )
            )

    nodes.extend(
        [
            # Top 4 corners
            StructuralNode("t_fl", np.array([0.0, 0.0, h]), role="frame"),
            StructuralNode("t_fr", np.array([w, 0.0, h]), role="frame"),
            StructuralNode("t_rl", np.array([0.0, d, h]), role="frame"),
            StructuralNode("t_rr", np.array([w, d, h]), role="frame"),
            # Stationary Y-rail beam nodes at the current Y-gantry station (bolted Y-rail + panel attach)
            StructuralNode("y_rail_car_l", np.array([0.0, y_gantry, z_gantry]), role="rail"),
            StructuralNode("y_rail_car_r", np.array([w, y_gantry, z_gantry]), role="rail"),
            # Moving Y carriages coupled to y_rail_car_{l,r} through 6-DOF Hertzian linear-rail bearings
            StructuralNode("y_car_l", np.array([0.0, y_gantry, z_gantry]), lumped_mass_kg=carriage_side_kg, role="gantry"),
            StructuralNode("y_car_r", np.array([w, y_gantry, z_gantry]), lumped_mass_kg=carriage_side_kg, role="gantry"),
            # Toolhead node on the X gantry beam
            StructuralNode("toolhead", np.array([x_tool, y_gantry, z_gantry]), lumped_mass_kg=toolhead_kg, role="toolhead"),
        ]
    )
    if spec.gantry.y_rail_mount == "inner":
        # Inner-face Y rails: the carriage rolls on a raceway inboard of the Y-extrusion axis, so the
        # moving carriage node sits on that raceway (rigid arm back to y_rail_car_* in the bearing).
        from printsim.fea.components import y_carriage_raceway_offset_m

        r_race = y_carriage_raceway_offset_m(
            spec.gantry.y_rail_profile, 0.5 * get_profile(spec.frame.profile).width_m
        )
        _nm = {n.id: n for n in nodes}
        _nm["y_car_l"].position_m = np.array([r_race, y_gantry, z_gantry])
        _nm["y_car_r"].position_m = np.array([w - r_race, y_gantry, z_gantry])


    trident_bed_anchors = _setup_corexy_z_drive_nodes_and_bed(
        spec=spec,
        nodes=nodes,
        fastener_springs=fastener_springs,
        z_drive=z_drive,
        bed_cfg=bed_cfg,
        bed_center=bed_center,
        w=w,
        d=d,
        h=h,
        z_gantry=z_gantry,
        z_subframe=z_subframe,
        z_motor_kg=z_motor_kg,
        k_z_drive_col_n_m=k_z_drive_col_n_m,
    )
    fp = spec.frame.profile
    up = spec.frame.effective_upright_profile
    fm = spec.frame.material
    bp = spec.frame.brace_profile
    bm = spec.frame.brace_material
    members: list[StructuralMember] = []

    # Base perimeter (split only where Z-rail/rod columns or Z-motors anchor)
    _add_member(members, "base_front", "b_fl", "b_fr", fp, fm)
    if spec.frame.rear_bottom_extrusion:
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
            _add_member(members, f"col_{tag}_bot", f"b_{tag}", f"z_col_bed_{tag}", up, fm)
            _add_member(members, f"col_{tag}_mid", f"z_col_bed_{tag}", col_mid, up, fm)
        else:
            _add_member(members, f"col_{tag}_low", f"b_{tag}", col_mid, up, fm)
        if not on_top_frame:
            _add_member(members, f"col_{tag}_top", col_mid, f"t_{tag}", up, fm)

    # Top perimeter (split along left/right rails where dual-Z guides / 4x smooth rods anchor)
    _add_member(members, "top_front", "t_fl", "t_fr", fp, fm)
    if not on_top_frame:
        _add_member(members, "top_rear", "t_rl", "t_rr", fp, fm)
    if on_top_frame:
        # The top-ring Y extrusions ARE the Y-rail beams (built as y_rail_* chains below) and the
        # rear top extrusion IS gantry_rear_bar: no duplicate collinear top-ring members.
        pass
    elif z_drive == "leadscrew_dual":
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
    if on_top_frame:
        _add_on_top_frame_y_rail_chains(members, nodes, fastener_springs, fp, fm, h)
    else:
        _add_member(members, "y_rail_l_f", "g_fl", "y_rail_car_l", fp, fm, role="rail")
        _add_member(members, "y_rail_l_r", "y_rail_car_l", "g_rl", fp, fm, role="rail")
        _add_member(members, "y_rail_r_f", "g_fr", "y_rail_car_r", fp, fm, role="rail")
        _add_member(members, "y_rail_r_r", "y_rail_car_r", "g_rr", fp, fm, role="rail")

    if z_drive in ("leadscrew_triple", "leadscrew_single"):
        _add_member(members, "gantry_rear_bar_l", "g_rl", "g_rear_mid", fp, fm)
        _add_member(members, "gantry_rear_bar_r", "g_rear_mid", "g_rr", fp, fm)
    else:
        _add_member(members, "gantry_rear_bar", "g_rl", "g_rr", fp, fm)

    # Wire baked 3D CAD Trident bed-arm superelements (z_bed_* -> z_arm_tip_*) in series with 2020 bed_arm_*
    trident_arm_tips: dict[str, str] = {}
    if z_drive == "leadscrew_triple":
        trident_arm_assets = _resolve_cad_trident_bed_arm_assets(spec)
        if trident_arm_assets:
            nmap_pre = {n.id: n for n in nodes}
            arm_targets = {
                "fl": ("z_bed_fl", "bed_t_fl"),
                "fr": ("z_bed_fr", "bed_t_fr"),
                "r": ("z_bed_r", "bed_t_rear_mount"),
            }
            for tag, (start_id, target_id) in arm_targets.items():
                asset = trident_arm_assets.get(tag)
                if asset is None or start_id not in nmap_pre or target_id not in nmap_pre:
                    continue
                p_start = nmap_pre[start_id].position_m
                p_target = nmap_pre[target_id].position_m
                p1_cad_m = np.asarray(asset.port_coords_m[1], dtype=float)  # type: ignore[attr-defined]
                dp_xy = np.array([float(p1_cad_m[0]), float(p1_cad_m[1]), 0.0], dtype=float)
                span_vec = p_target - p_start
                span_len = float(np.linalg.norm(span_vec[:2]))
                dp_len = float(np.linalg.norm(dp_xy[:2]))
                if span_len > 0.025 and dp_len > 1e-4:
                    # Keep z_arm_tip_* strictly horizontal at z_subframe and <= 80% of span to bed_t_*
                    if dp_len > 0.80 * span_len:
                        dp_xy *= (0.80 * span_len) / dp_len
                    p_tip = p_start + dp_xy
                    tip_id = f"z_arm_tip_{tag}"
                    nodes.append(StructuralNode(tip_id, p_tip, role="bed"))
                    # Keep z_bed_* strictly positive-definite in M when decoupled from bed_arm_*
                    if nmap_pre[start_id].lumped_mass_kg <= 0.0:
                        nmap_pre[start_id].lumped_mass_kg = 1e-7
                    trident_arm_tips[tag] = tip_id
                    gantry_superelements.append(
                        _build_cad_trident_bed_arm_superelement(
                            f"trident_bed_arm_{tag}",
                            asset,
                            (start_id, tip_id),
                            p_start,
                            p_tip,
                            spec,
                        )
                    )

    _add_corexy_bed_subframe_members(
        spec=spec,
        nodes=nodes,
        members=members,
        z_drive=z_drive,
        is_flying=is_flying,
        fp=fp,
        fm=fm,
        h=h,
        bz=bz,
        trident_bed_anchors=trident_bed_anchors,
        trident_arm_tips=trident_arm_tips or None,
    )
    # Moving X gantry beam (split at toolhead node).
    # When a CAD gantry hardware kit provides baked 3D XY-joint superelements (xy_left / xy_right),
    # terminate the X-beam at the exact CAD inner clamp ports (x_beam_l / x_beam_r) and connect
    # y_car_* -> x_beam_* with the full 12x12 Craig-Bampton XY-joint stiffness matrices.
    xp = spec.gantry.x_beam_profile
    xm = spec.gantry.x_beam_material
    xy_assets = _resolve_cad_xy_joint_assets(spec)
    nmap_ycar = {n.id: n for n in nodes if n.id in ("y_car_l", "y_car_r")}
    if xy_assets is not None:
        xy_l_asset, xy_r_asset = xy_assets
        raw_in_l = abs(float(xy_l_asset.port_coords_m[1][0]))  # type: ignore[attr-defined]
        raw_in_r = abs(float(xy_r_asset.port_coords_m[1][0]))  # type: ignore[attr-defined]
        max_in_l = max(0.015, min(0.25 * w, x_tool - 0.015))
        max_in_r = max(0.015, min(0.25 * w, (w - x_tool) - 0.015))
        x_in_l = float(np.clip(raw_in_l, 0.015, max_in_l))
        x_in_r = float(np.clip(raw_in_r, 0.015, max_in_r))
        p_ycar_l = np.array(nmap_ycar["y_car_l"].position_m, dtype=float)
        p_ycar_r = np.array(nmap_ycar["y_car_r"].position_m, dtype=float)
        p_xbeam_l = np.array([x_in_l, y_gantry, z_gantry], dtype=float)
        p_xbeam_r = np.array([w - x_in_r, y_gantry, z_gantry], dtype=float)
        nodes.extend(
            [
                StructuralNode("x_beam_l", p_xbeam_l, role="gantry"),
                StructuralNode("x_beam_r", p_xbeam_r, role="gantry"),
            ]
        )
        _add_member(members, "x_gantry_l", "x_beam_l", "toolhead", xp, xm, role="gantry")
        _add_member(members, "x_gantry_r", "toolhead", "x_beam_r", xp, xm, role="gantry")
        gantry_superelements.append(
            _build_cad_xy_joint_superelement(
                "xy_joint_l",
                xy_l_asset,
                ("y_car_l", "x_beam_l"),
                p_ycar_l,
                p_xbeam_l,
                spec,
            )
        )
        gantry_superelements.append(
            _build_cad_xy_joint_superelement(
                "xy_joint_r",
                xy_r_asset,
                ("y_car_r", "x_beam_r"),
                p_ycar_r,
                p_xbeam_r,
                spec,
            )
        )
    elif bool(getattr(spec, "use_cad_superelements", True)):
        # Procedural CoreXY builds (with superelements enabled, the default) still wire analytical
        # Timoshenko + Huth/VDI bolted-seam XY-joint superelements so y_car_* -> x_beam_* flex is
        # captured even without a CAD gantry kit. Explicit use_cad_superelements=False retains the
        # pure-beam baseline for isolated beam-only verification.
        default_in = get_profile(fp).width_m + 0.012
        max_in_l = max(0.015, min(0.25 * w, x_tool - 0.015))
        max_in_r = max(0.015, min(0.25 * w, (w - x_tool) - 0.015))
        x_in_l = float(np.clip(default_in, 0.015, max_in_l))
        x_in_r = float(np.clip(default_in, 0.015, max_in_r))
        p_ycar_l = np.array(nmap_ycar["y_car_l"].position_m, dtype=float)
        p_ycar_r = np.array(nmap_ycar["y_car_r"].position_m, dtype=float)
        p_xbeam_l = np.array([x_in_l, y_gantry, z_gantry], dtype=float)
        p_xbeam_r = np.array([w - x_in_r, y_gantry, z_gantry], dtype=float)
        nodes.extend(
            [
                StructuralNode("x_beam_l", p_xbeam_l, role="gantry"),
                StructuralNode("x_beam_r", p_xbeam_r, role="gantry"),
            ]
        )
        _add_member(members, "x_gantry_l", "x_beam_l", "toolhead", xp, xm, role="gantry")
        _add_member(members, "x_gantry_r", "toolhead", "x_beam_r", xp, xm, role="gantry")
        gantry_superelements.append(
            _build_procedural_xy_joint_superelement(
                "xy_joint_l",
                ("y_car_l", "x_beam_l"),
                p_ycar_l,
                p_xbeam_l,
                spec,
            )
        )
        gantry_superelements.append(
            _build_procedural_xy_joint_superelement(
                "xy_joint_r",
                ("y_car_r", "x_beam_r"),
                p_ycar_r,
                p_xbeam_r,
                spec,
            )
        )
    else:
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


    bolt_pitch_m = spec.gantry.rail_bolt_pitch_mm * 1e-3
    member_by_id = {m.id: m for m in members}
    (
        x_rail_prof,
        rail_mat,
        y_car,
        x_beam_insert_mass_kg,
        x_rail_ei_y_boost_pct,
        x_rail_ei_z_boost_pct,
        x_rail_gj_boost_pct,
        y_rail_ei_x_boost_pct,
    ) = _apply_corexy_gantry_rails(
        spec=spec,
        members=members,
        member_by_id=member_by_id,
        nmap=nmap,
        w=w,
        d=d,
        fp=fp,
        xp=xp,
        xm=xm,
    )

    # Wire explicit 6-DOF Hertzian linear-rail carriage contact bearings (y_rail_car_* -> y_car_*)
    fastener_springs.append(
        _build_y_carriage_bearing_superelement(
            "y_car_bearing_l",
            ("y_rail_car_l", "y_car_l"),
            y_car,
            spec,
            rail_and_car_pos_m=(nmap["y_rail_car_l"].position_m, nmap["y_car_l"].position_m),
        )
    )
    fastener_springs.append(
        _build_y_carriage_bearing_superelement(
            "y_car_bearing_r",
            ("y_rail_car_r", "y_car_r"),
            y_car,
            spec,
            rail_and_car_pos_m=(nmap["y_rail_car_r"].position_m, nmap["y_car_r"].position_m),
        )
    )

    _apply_corexy_z_guideways(
        spec=spec,
        member_by_id=member_by_id,
        nmap=nmap,
        z_drive=z_drive,
        is_flying=is_flying,
        fp=fp,
        rail_mat=rail_mat,
        bolt_pitch_m=bolt_pitch_m,
        h=h,
        z_gantry=z_gantry,
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

    # Calculate moving masses (strip composite rail, backer & insert mass from gantry beam before adding x_rail_mass)
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
    if xp == "unsupported_rail":
        # Bare rail IS the X-gantry beam (x_beam_mass == 0.0; rail_mass_kg holds exact bare rail mass)
        x_rail_mass = sum(m.rail_mass_kg for m in members if m.role == "gantry")
    else:
        x_rail_len = corexy_x_rail_len_m(spec)
        x_rail_mass = x_rail_prof.area_m2 * x_rail_len * rail_mat.density_kg_m3
        if spec.gantry.effective_dual_x_rails:
            x_rail_mass *= 2.0

    # In CoreXY, X moves toolhead; Y moves toolhead + X beam + X rail + backer + CF inserts + Y carriages + gantry bracket delta
    x_moving = toolhead_kg + 0.04  # plus belt segment
    y_moving = (
        toolhead_kg
        + x_beam_mass
        + x_rail_mass
        + x_backer_mass
        + x_beam_insert_mass_kg
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
        superelements=gantry_superelements,
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
        x_beam_insert_mass_kg=x_beam_insert_mass_kg,
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
        x_rail_gj_boost_pct=x_rail_gj_boost_pct,
        y_rail_ei_x_boost_pct=y_rail_ei_x_boost_pct,
        y_carriage_block_mass_g=float(y_car["block_mass_g"]),
        y_carriage_block_delta_g=y_block_delta_g,
        y_carriage_k_rx_nm_rad=float(y_car["k_rx_nm_rad"]),
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
        base_mk = max(
            mk - m.fill_mass_kg - m.rail_mass_kg - m.backer_mass_kg - m.insert_mass_kg, 0.0
        )
        cost += base_mk * m.material.cost_per_kg_usd
        if m.rail_mass_kg > 0.0:
            total_rail_mass += m.rail_mass_kg
            cost += m.rail_mass_kg * rail_mat.cost_per_kg_usd
        if m.backer_mass_kg > 0.0 and m.backer_material_id:
            cost += m.backer_mass_kg * get_material(m.backer_material_id).cost_per_kg_usd
    model.total_rail_mass_kg = total_rail_mass

    if model.x_beam_insert_mass_kg > 0.0:
        cost += 18.0

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
