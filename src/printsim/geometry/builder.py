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
        tid_vis = (self.spec.toolhead.preset_id or "").lower().strip() or None
        th_acc = resolve_toolhead_hardware_accessories(self.spec)
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
                    if (hid_vis and _has_cad_asset("hotends", hid_vis))
                    else None
                ),
                "toolhead_visual_glb_url": (
                    f"/api/assets/hardware/toolheads/{tid_vis}/visual.glb"
                    if (tid_vis and _has_cad_asset("toolheads", tid_vis))
                    else None
                ),
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
    _compute_newmark_rail_gamma,
    _make_x_backer_profile,
    _member_local_axes,
    _populate_linear_rail_telemetry,
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


def _apply_feet_grounding(model: AssemblyModel) -> None:
    """Replace rigid fixed feet with compliant benchtop isolator springs when configured."""
    f_cfg = resolve_frame_feet(model.spec)
    if not f_cfg["is_compliant"]:
        return

    k_normal_n_m = float(f_cfg["k_axial_n_m"])
    k_shear_n_m = float(f_cfg["k_lateral_n_m"])
    if k_normal_n_m <= 0.0:
        k_normal_n_m = 1.0e8
    if k_shear_n_m <= 0.0:
        k_shear_n_m = 1.0e8
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
    _apply_feet_grounding(model)
    _apply_extrusion_filling(model)
    _apply_mount_and_feet_damping(model)
    _finalize_mass_and_cost(model)
    return model


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
    trident_backbone_active = bool(spec.frame.trident_backbone and not is_flying)
    z_drive = spec.bed.z_drive
    bed_cfg = resolve_bed_mount(spec)
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
