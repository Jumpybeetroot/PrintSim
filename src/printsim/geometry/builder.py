"""3D structural assembly graph generator from declarative PrinterSpec."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from printsim.schema import (
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

    def length_m(self, node_map: dict[str, StructuralNode]) -> float:
        p0 = node_map[self.start_id].position_m
        p1 = node_map[self.end_id].position_m
        return float(np.linalg.norm(p1 - p0))

    def mass_kg(self, node_map: dict[str, StructuralNode]) -> float:
        return self.profile.area_m2 * self.length_m(node_map) * self.material.density_kg_m3


@dataclass
class AssemblyModel:
    """Complete physical 3D assembly model ready for FEA, kinematics, and visualization."""

    spec: PrinterSpec
    nodes: list[StructuralNode] = field(default_factory=list)
    members: list[StructuralMember] = field(default_factory=list)
    toolhead_node_id: str = "toolhead"
    gantry_node_ids: list[str] = field(default_factory=list)
    bed_center_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    build_origin_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    x_moving_mass_kg: float = 0.0
    y_moving_mass_kg: float = 0.0
    frame_mass_kg: float = 0.0
    total_mass_kg: float = 0.0
    estimated_material_cost_usd: float = 0.0

    @property
    def node_map(self) -> dict[str, StructuralNode]:
        return {n.id: n for n in self.nodes}

    def to_geometry_dict(self) -> dict:
        """Serialize assembly geometry (mm) for the 3D web viewer and AI tools."""
        nmap = self.node_map
        pmat = get_material(self.spec.frame.panel_material)
        panels_enabled = bool(
            self.spec.frame.enclosed_panels and self.spec.frame.panel_mounting != "none"
        )
        return {
            "architecture": self.spec.architecture,
            "drive_mode": self.spec.gantry.drive_mode,
            "double_shear_pulleys": self.spec.gantry.double_shear_pulleys,
            "panels": {
                "enabled": panels_enabled,
                "mounting": (
                    self.spec.frame.panel_mounting if self.spec.frame.enclosed_panels else "none"
                ),
                "material_id": self.spec.frame.panel_material,
                "material_name": pmat.name,
                "thickness_mm": self.spec.frame.panel_thickness_mm,
                "color_hex": pmat.color_hex,
            },
            "frame_dims_mm": [
                self.spec.frame.width_mm,
                self.spec.frame.depth_mm,
                self.spec.frame.height_mm,
            ],
            "build_volume_mm": list(self.spec.kinematics.build_volume_mm),
            "build_origin_mm": (self.build_origin_m * 1000.0).tolist(),
            "bed_size_mm": list(self.spec.bed.size_mm),
            "bed_center_mm": (self.bed_center_m * 1000.0).tolist(),
            "toolhead_node_id": self.toolhead_node_id,
            "toolhead_envelope_mm": list(self.spec.toolhead.envelope_mm),
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


def _panel_shear_params(spec: PrinterSpec) -> tuple[bool, float, MaterialSpec, float]:
    """Return (active, shear_efficiency_eta, panel_material, thickness_m)."""
    pmat = get_material(spec.frame.panel_material)
    mounting = spec.frame.panel_mounting
    if not spec.frame.enclosed_panels or mounting == "none":
        return False, 0.0, pmat, spec.frame.panel_thickness_mm * 1e-3
    eta = 0.65 if mounting == "bolted_structural" else 0.12
    return True, eta, pmat, spec.frame.panel_thickness_mm * 1e-3


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
    eta: float,
) -> None:
    """Model a rectangular shear panel via equivalent X-pair diagonal truss struts.

    Uses the textbook diagonal shear-diaphragm equivalence:
        A_eq = eta * (G * t * L_d^3) / (2 * E * b * h)
    and distributes any remaining true static panel mass onto the four corner nodes.
    """
    b = float(np.linalg.norm(node_map[br_id].position_m - node_map[bl_id].position_m))
    h = float(np.linalg.norm(node_map[tl_id].position_m - node_map[bl_id].position_m))
    if b <= 1e-6 or h <= 1e-6:
        return
    diag_len = math.hypot(b, h)
    e_pa = pmat.youngs_modulus_pa
    g_pa = pmat.effective_shear_modulus_pa
    a_eq = max(eta * (g_pa * thickness_m * (diag_len**3)) / (2.0 * e_pa * b * h), 1e-9)

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
            material=pmat,
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
            material=pmat,
            role="brace",
            is_truss=True,
        )
    )

    actual_panel_mass = b * h * thickness_m * pmat.density_kg_m3
    strut_mass = 2.0 * a_eq * diag_len * pmat.density_kg_m3
    rem_mass = max(0.0, actual_panel_mass - strut_mass)
    if rem_mass > 0.0:
        per_node = 0.25 * rem_mass
        for nid in (bl_id, br_id, tl_id, tr_id):
            node_map[nid].lumped_mass_kg += per_node


def build_assembly(
    spec: PrinterSpec,
    toolhead_rel_pos: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> AssemblyModel:
    """Construct a 3D FEA-ready structural assembly from a PrinterSpec.

    Parameters
    ----------
    spec:
        Declarative PrinterSpec.
    toolhead_rel_pos:
        Normalized (u, v, w) in [0, 1]^3 specifying where the toolhead is positioned
        inside the build volume for structural stiffness evaluation. Defaults to center.
    """
    if spec.frame.custom_nodes and spec.frame.custom_members:
        return _build_custom_assembly(spec)

    if spec.architecture == "corexy":
        model = _build_corexy(spec, toolhead_rel_pos)
    elif spec.architecture == "cartesian_bedslinger":
        model = _build_cartesian_bedslinger(spec, toolhead_rel_pos)
    elif spec.architecture == "delta":
        model = _build_delta(spec, toolhead_rel_pos)
    else:
        raise ValueError(f"Unsupported architecture: {spec.architecture}")

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
    model = AssemblyModel(
        spec=spec,
        nodes=nodes,
        members=members,
        toolhead_node_id=toolhead_id,
        bed_center_m=np.array([w * 0.5, d * 0.5, 0.06]),
        build_origin_m=np.array([(w - bx) * 0.5, (d - by) * 0.5, 0.07]),
        x_moving_mass_kg=spec.toolhead.total_mass_g * 1e-3,
        y_moving_mass_kg=(spec.toolhead.total_mass_g + spec.gantry.carriage_mass_g) * 1e-3,
    )
    _finalize_mass_and_cost(model)
    return model


def _build_corexy(
    spec: PrinterSpec, toolhead_rel_pos: tuple[float, float, float]
) -> AssemblyModel:
    w = spec.frame.width_mm * 1e-3
    d = spec.frame.depth_mm * 1e-3
    h = spec.frame.height_mm * 1e-3
    bx, by, bz = np.array(spec.kinematics.build_volume_mm) * 1e-3
    u, v, _ = np.clip(toolhead_rel_pos, 0.05, 0.95)

    # Gantry plane near top of CoreXY frame
    z_gantry = h - 0.055
    xy_motor_kg = spec.kinematics.xy_stepper.motor_mass_g * 1e-3
    z_motor_kg = spec.kinematics.z_stepper.motor_mass_g * 1e-3
    carriage_side_kg = (spec.gantry.carriage_mass_g * 0.5) * 1e-3
    toolhead_kg = spec.toolhead.total_mass_g * 1e-3

    # Build origin centered in frame
    orig_x = (w - bx) * 0.5
    orig_y = (d - by) * 0.5
    orig_z = 0.08
    y_gantry = orig_y + v * by
    x_tool = orig_x + u * bx

    is_awd = spec.gantry.drive_mode == "awd"
    front_motor_kg = (xy_motor_kg + 0.03) if is_awd else 0.0
    front_role = "mount" if is_awd else "frame"

    nodes = [
        # Bottom 4 corner feet (fixed to table)
        StructuralNode("b_fl", np.array([0.0, 0.0, 0.0]), fixed=True, lumped_mass_kg=z_motor_kg, role="foot"),
        StructuralNode("b_fr", np.array([w, 0.0, 0.0]), fixed=True, lumped_mass_kg=z_motor_kg, role="foot"),
        StructuralNode("b_rl", np.array([0.0, d, 0.0]), fixed=True, lumped_mass_kg=z_motor_kg, role="foot"),
        StructuralNode("b_rr", np.array([w, d, 0.0]), fixed=True, lumped_mass_kg=z_motor_kg, role="foot"),
        # Gantry elevation nodes on the 4 vertical columns
        StructuralNode("g_fl", np.array([0.0, 0.0, z_gantry]), lumped_mass_kg=front_motor_kg, role=front_role),
        StructuralNode("g_fr", np.array([w, 0.0, z_gantry]), lumped_mass_kg=front_motor_kg, role=front_role),
        StructuralNode("g_rl", np.array([0.0, d, z_gantry]), lumped_mass_kg=xy_motor_kg, role="mount"),
        StructuralNode("g_rr", np.array([w, d, z_gantry]), lumped_mass_kg=xy_motor_kg, role="mount"),
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

    fp = spec.frame.profile
    fm = spec.frame.material
    bp = spec.frame.brace_profile
    bm = spec.frame.brace_material
    members: list[StructuralMember] = []

    # Base perimeter
    _add_member(members, "base_front", "b_fl", "b_fr", fp, fm)
    _add_member(members, "base_rear", "b_rl", "b_rr", fp, fm)
    _add_member(members, "base_left", "b_fl", "b_rl", fp, fm)
    _add_member(members, "base_right", "b_fr", "b_rr", fp, fm)

    # 4 vertical columns split at gantry height
    for tag in ("fl", "fr", "rl", "rr"):
        _add_member(members, f"col_{tag}_low", f"b_{tag}", f"g_{tag}", fp, fm)
        _add_member(members, f"col_{tag}_top", f"g_{tag}", f"t_{tag}", fp, fm)

    # Top perimeter
    _add_member(members, "top_front", "t_fl", "t_fr", fp, fm)
    _add_member(members, "top_rear", "t_rl", "t_rr", fp, fm)
    _add_member(members, "top_left", "t_fl", "t_rl", fp, fm)
    _add_member(members, "top_right", "t_fr", "t_rr", fp, fm)

    # Mid gantry frame rails (left/right split at Y carriage, plus rear motor bar)
    _add_member(members, "y_rail_l_f", "g_fl", "y_car_l", fp, fm, role="rail")
    _add_member(members, "y_rail_l_r", "y_car_l", "g_rl", fp, fm, role="rail")
    _add_member(members, "y_rail_r_f", "g_fr", "y_car_r", fp, fm, role="rail")
    _add_member(members, "y_rail_r_r", "y_car_r", "g_rr", fp, fm, role="rail")
    _add_member(members, "gantry_rear_bar", "g_rl", "g_rr", fp, fm)

    # Moving X gantry beam (split at toolhead node)
    xp = spec.gantry.x_beam_profile
    xm = spec.gantry.x_beam_material
    _add_member(members, "x_gantry_l", "y_car_l", "toolhead", xp, xm, role="gantry")
    _add_member(members, "x_gantry_r", "toolhead", "y_car_r", xp, xm, role="gantry")

    # Optional dual X rail stiffening member modeled in parallel
    if spec.gantry.dual_x_rails:
        _add_member(
            members,
            "x_rail_aux_l",
            "y_car_l",
            "toolhead",
            spec.gantry.x_rail_profile,
            spec.gantry.rail_material,
            role="rail",
        )
        _add_member(
            members,
            "x_rail_aux_r",
            "toolhead",
            "y_car_r",
            spec.gantry.x_rail_profile,
            spec.gantry.rail_material,
            role="rail",
        )

    # Bracing strategies
    bracing = spec.frame.bracing
    if bracing == "corner_gussets":
        # Small stiffening corner nodes/struts on rear face & top corners
        _add_member(members, "brace_rear_1", "b_rl", "g_rr", bp, bm, role="brace", is_truss=True)
    elif bracing == "rear_cross":
        _add_member(members, "brace_rear_1", "b_rl", "t_rr", bp, bm, role="brace", is_truss=True)
        _add_member(members, "brace_rear_2", "b_rr", "t_rl", bp, bm, role="brace", is_truss=True)
        _add_member(members, "brace_top_1", "t_fl", "t_rr", bp, bm, role="brace", is_truss=True)
    elif bracing == "full_truss":
        _add_member(members, "brace_rear_1", "b_rl", "t_rr", bp, bm, role="brace", is_truss=True)
        _add_member(members, "brace_rear_2", "b_rr", "t_rl", bp, bm, role="brace", is_truss=True)
        _add_member(members, "brace_left_1", "b_fl", "g_rl", bp, bm, role="brace", is_truss=True)
        _add_member(members, "brace_left_2", "b_rl", "g_fl", bp, bm, role="brace", is_truss=True)
        _add_member(members, "brace_right_1", "b_fr", "g_rr", bp, bm, role="brace", is_truss=True)
        _add_member(members, "brace_right_2", "b_rr", "g_fr", bp, bm, role="brace", is_truss=True)
        _add_member(members, "brace_top_1", "t_fl", "t_rr", bp, bm, role="brace", is_truss=True)
        _add_member(members, "brace_top_2", "t_fr", "t_rl", bp, bm, role="brace", is_truss=True)

    nmap = {n.id: n for n in nodes}

    # Enclosure / structural shear panels across rear, side, and top bays
    panels_active, panel_eta, pmat, panel_t_m = _panel_shear_params(spec)
    if panels_active:
        for bay, bl_id, br_id, tl_id, tr_id in (
            ("rear", "b_rl", "b_rr", "t_rl", "t_rr"),
            ("left", "b_fl", "b_rl", "t_fl", "t_rl"),
            ("right", "b_fr", "b_rr", "t_fr", "t_rr"),
            ("top", "t_fl", "t_fr", "t_rl", "t_rr"),
        ):
            _add_shear_panel_bay(
                members,
                nmap,
                bay,
                bl_id,
                br_id,
                tl_id,
                tr_id,
                pmat,
                panel_t_m,
                panel_eta,
            )

    # Calculate moving masses
    x_beam_mass = sum(m.mass_kg(nmap) for m in members if m.role == "gantry")
    x_rail_mass = get_profile(spec.gantry.x_rail_profile).area_m2 * w * get_material(spec.gantry.rail_material).density_kg_m3
    if spec.gantry.dual_x_rails:
        x_rail_mass *= 2.0

    # In CoreXY, X moves toolhead; Y moves toolhead + X beam + X rail + Y carriages
    x_moving = toolhead_kg + 0.04  # plus belt segment
    y_moving = toolhead_kg + x_beam_mass + x_rail_mass + spec.gantry.carriage_mass_g * 1e-3

    return AssemblyModel(
        spec=spec,
        nodes=nodes,
        members=members,
        toolhead_node_id="toolhead",
        gantry_node_ids=["y_car_l", "toolhead", "y_car_r"],
        bed_center_m=np.array([w * 0.5, d * 0.5, orig_z]),
        build_origin_m=np.array([orig_x, orig_y, orig_z]),
        x_moving_mass_kg=x_moving,
        y_moving_mass_kg=y_moving,
    )


def _build_cartesian_bedslinger(
    spec: PrinterSpec, toolhead_rel_pos: tuple[float, float, float]
) -> AssemblyModel:
    w = spec.frame.width_mm * 1e-3
    d = spec.frame.depth_mm * 1e-3
    h = spec.frame.height_mm * 1e-3
    bx, by, bz = np.array(spec.kinematics.build_volume_mm) * 1e-3
    u, _, w_rel = np.clip(toolhead_rel_pos, 0.05, 0.95)

    y_portal = d * 0.55
    orig_x = (w - bx) * 0.5
    orig_y = (d - by) * 0.5
    orig_z = 0.075
    z_gantry = min(h - 0.06, orig_z + w_rel * bz)
    x_tool = orig_x + u * bx

    xy_motor_kg = spec.kinematics.xy_stepper.motor_mass_g * 1e-3
    z_motor_kg = spec.kinematics.z_stepper.motor_mass_g * 1e-3
    carriage_side_kg = (spec.gantry.carriage_mass_g * 0.5) * 1e-3
    toolhead_kg = spec.toolhead.total_mass_g * 1e-3

    nodes = [
        # H-frame base feet
        StructuralNode("b_fl", np.array([0.0, 0.0, 0.0]), fixed=True, role="foot"),
        StructuralNode("b_fr", np.array([w, 0.0, 0.0]), fixed=True, role="foot"),
        StructuralNode("b_rl", np.array([0.0, d, 0.0]), fixed=True, role="foot"),
        StructuralNode("b_rr", np.array([w, d, 0.0]), fixed=True, role="foot"),
        # Portal base junctions
        StructuralNode("p_bl", np.array([0.0, y_portal, 0.0]), fixed=True, lumped_mass_kg=z_motor_kg, role="mount"),
        StructuralNode("p_br", np.array([w, y_portal, 0.0]), fixed=True, lumped_mass_kg=z_motor_kg, role="mount"),
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

    # Portal diagonal braces (huge impact on bedslinger Y-ringing!)
    if spec.frame.bracing in ("corner_gussets", "rear_cross", "full_truss"):
        _add_member(members, "brace_l_rear", "b_rl", "p_tl", bp, bm, role="brace", is_truss=True)
        _add_member(members, "brace_r_rear", "b_rr", "p_tr", bp, bm, role="brace", is_truss=True)
    if spec.frame.bracing in ("rear_cross", "full_truss"):
        _add_member(members, "brace_portal_x1", "p_bl", "p_tr", bp, bm, role="brace", is_truss=True)
    if spec.frame.bracing == "full_truss":
        _add_member(members, "brace_l_front", "b_fl", "p_tl", bp, bm, role="brace", is_truss=True)
        _add_member(members, "brace_r_front", "b_fr", "p_tr", bp, bm, role="brace", is_truss=True)
        _add_member(members, "brace_portal_x2", "p_br", "p_tl", bp, bm, role="brace", is_truss=True)

    # Bedslinger: X moves toolhead; Y moves the entire heavy heated bed + carriage!
    x_moving = toolhead_kg + 0.03
    y_moving = spec.bed.mass_g * 1e-3 + 0.35

    return AssemblyModel(
        spec=spec,
        nodes=nodes,
        members=members,
        toolhead_node_id="toolhead",
        gantry_node_ids=["z_l", "toolhead", "z_r"],
        bed_center_m=np.array([w * 0.5, d * 0.5, orig_z]),
        build_origin_m=np.array([orig_x, orig_y, orig_z]),
        x_moving_mass_kg=x_moving,
        y_moving_mass_kg=y_moving,
    )


def _build_delta(
    spec: PrinterSpec, toolhead_rel_pos: tuple[float, float, float]
) -> AssemblyModel:
    radius = max(spec.frame.width_mm, spec.frame.depth_mm) * 0.5 * 1e-3
    h = spec.frame.height_mm * 1e-3
    bx, by, bz = np.array(spec.kinematics.build_volume_mm) * 1e-3
    _, _, w_rel = np.clip(toolhead_rel_pos, 0.1, 0.8)

    # Center the delta prism around (radius, radius) so coordinates are positive
    cx, cy = radius, radius
    angles = [math.radians(90.0), math.radians(210.0), math.radians(330.0)]
    z_effector = 0.08 + w_rel * min(bz, h * 0.55)
    arm_len = spec.kinematics.delta_arm_length_mm * 1e-3
    z_car = min(h - 0.08, z_effector + math.sqrt(max(arm_len**2 - (radius * 0.75) ** 2, 0.01)))

    motor_kg = spec.kinematics.xy_stepper.motor_mass_g * 1e-3
    toolhead_kg = spec.toolhead.total_mass_g * 1e-3
    car_kg = (spec.gantry.carriage_mass_g / 3.0) * 1e-3

    nodes: list[StructuralNode] = []
    for i, ang in enumerate(angles):
        px = cx + radius * math.cos(ang)
        py = cy + radius * math.sin(ang)
        nodes.append(StructuralNode(f"b_{i}", np.array([px, py, 0.0]), fixed=True, lumped_mass_kg=motor_kg, role="foot"))
        nodes.append(StructuralNode(f"c_{i}", np.array([px, py, z_car]), lumped_mass_kg=car_kg, role="gantry"))
        nodes.append(StructuralNode(f"t_{i}", np.array([px, py, h]), role="frame"))

    nodes.append(StructuralNode("toolhead", np.array([cx, cy, z_effector]), lumped_mass_kg=toolhead_kg, role="toolhead"))

    fp = spec.frame.profile
    fm = spec.frame.material
    bp = spec.frame.brace_profile
    bm = spec.frame.brace_material
    members: list[StructuralMember] = []

    for i in range(3):
        j = (i + 1) % 3
        _add_member(members, f"base_{i}", f"b_{i}", f"b_{j}", fp, fm)
        _add_member(members, f"top_{i}", f"t_{i}", f"t_{j}", fp, fm)
        _add_member(members, f"tower_{i}_low", f"b_{i}", f"c_{i}", fp, fm, role="tower")
        _add_member(members, f"tower_{i}_top", f"c_{i}", f"t_{i}", fp, fm, role="tower")
        # Carbon delta rods connecting tower carriages to effector
        _add_member(members, f"arm_{i}", f"c_{i}", "toolhead", "brace_rod_10mm", "carbon_fiber_tube", role="gantry", is_truss=True)

        if spec.frame.bracing in ("rear_cross", "full_truss"):
            _add_member(members, f"brace_{i}_a", f"b_{i}", f"t_{j}", bp, bm, role="brace", is_truss=True)
        if spec.frame.bracing == "full_truss":
            _add_member(members, f"brace_{i}_b", f"b_{j}", f"t_{i}", bp, bm, role="brace", is_truss=True)

    panels_active, panel_eta, pmat, panel_t_m = _panel_shear_params(spec)
    if panels_active:
        nmap = {n.id: n for n in nodes}
        for i in range(3):
            j = (i + 1) % 3
            _add_shear_panel_bay(
                members,
                nmap,
                f"bay{i}",
                f"b_{i}",
                f"b_{j}",
                f"t_{i}",
                f"t_{j}",
                pmat,
                panel_t_m,
                panel_eta,
            )

    eff_mass = toolhead_kg + 0.09  # ultra-light effector + arm portion
    return AssemblyModel(
        spec=spec,
        nodes=nodes,
        members=members,
        toolhead_node_id="toolhead",
        gantry_node_ids=["c_0", "c_1", "c_2", "toolhead"],
        bed_center_m=np.array([cx, cy, 0.06]),
        build_origin_m=np.array([cx - bx * 0.5, cy - by * 0.5, 0.06]),
        x_moving_mass_kg=eff_mass,
        y_moving_mass_kg=eff_mass,
    )


def _finalize_mass_and_cost(model: AssemblyModel) -> None:
    nmap = model.node_map
    frame_mass = 0.0
    cost = 0.0
    for m in model.members:
        mk = m.mass_kg(nmap)
        frame_mass += mk
        cost += mk * m.material.cost_per_kg_usd

    lumped_mass = sum(n.lumped_mass_kg for n in model.nodes)
    bed_mass = model.spec.bed.mass_g * 1e-3
    model.frame_mass_kg = frame_mass
    model.total_mass_kg = frame_mass + lumped_mass + bed_mass
    # Add baseline estimate for motors, bed plate, rails, and hotend hardware
    bed_mat = get_material(model.spec.bed.plate_material)
    cost += bed_mass * bed_mat.cost_per_kg_usd + 140.0  # steppers/electronics/hotend baseline
    model.estimated_material_cost_usd = cost
