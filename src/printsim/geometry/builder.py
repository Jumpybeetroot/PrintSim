"""3D structural assembly graph generator from declarative PrinterSpec."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from printsim.schema import (
    EXTRUSION_FILLINGS,
    GANTRY_BRACKET_CATALOG,
    PROFILE_CATALOG,
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
            "toolhead_node_id": self.toolhead_node_id,
            "toolhead_cog_node_id": self.toolhead_cog_node_id,
            "nozzle_node_id": self.nozzle_node_id,
            "bed_node_id": self.bed_node_id,
            "cog_offset_mm": list(self.spec.toolhead.cog_offset_mm),
            "nozzle_offset_mm": list(self.spec.toolhead.nozzle_offset_mm),
            "mount_material": self.spec.toolhead.mount_material,
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


def _apply_composite_panel_stiffening(
    members: list[StructuralMember],
    node_map: dict[str, StructuralNode],
    pmat: MaterialSpec,
    thickness_m: float,
    props: dict[str, float],
    bolted_to_gantry: bool,
) -> None:
    """Apply parallel-axis composite flange / rail-backer bending stiffening to bolted frame beams.

    Rigidly bolted structural panels act compositely with the aluminum extrusions they are
    bolted against (columns, top rails, and fixed gantry Y-rails / rear bar). Uses von Karman /
    Eurocode 3 effective flange width ``b_eff = min(15 * t_bear, 60 mm)`` projected onto exact
    3D beam local axes ``ey, ez`` without altering ``area_m2`` (no mass double-counting).
    """
    eta_c = float(props.get("eta_comp", 0.0))
    if eta_c <= 1e-4:
        return

    t_load = float(props.get("t_bear_m", thickness_m))
    e_load = float(props.get("e_bear_pa", pmat.youngs_modulus_pa))
    g_load = 26.3e9 if pmat.id == "acm_dibond" else pmat.effective_shear_modulus_pa
    b_eff = min(15.0 * t_load, 0.060)
    a_f = b_eff * t_load

    nx = np.array([1.0, 0.0, 0.0])
    ny = np.array([0.0, 1.0, 0.0])
    nz = np.array([0.0, 0.0, 1.0])
    panel_attachments: dict[str, list[np.ndarray]] = {
        "col_fl_low": [nx],
        "col_fl_bot": [nx],
        "col_fl_mid": [nx],
        "col_fl_top": [nx],
        "col_fr_low": [nx],
        "col_fr_bot": [nx],
        "col_fr_mid": [nx],
        "col_fr_top": [nx],
        "col_rl_low": [nx, ny],
        "col_rl_top": [nx, ny],
        "col_rr_low": [nx, ny],
        "col_rr_top": [nx, ny],
        "top_front": [nz],
        "top_rear": [ny, nz],
        "top_left": [nx, nz],
        "top_right": [nx, nz],
    }
    if bolted_to_gantry:
        panel_attachments.update(
            {
                "y_rail_l_f": [nx],
                "y_rail_l_r": [nx],
                "y_rail_r_f": [nx],
                "y_rail_r_r": [nx],
                "gantry_rear_bar": [ny],
                "gantry_rear_bar_l": [ny],
                "gantry_rear_bar_r": [ny],
            }
        )

    for member in members:
        normals = panel_attachments.get(member.id)
        if not normals:
            continue
        p0 = node_map[member.start_id].position_m
        p1 = node_map[member.end_id].position_m
        ey, ez = _member_local_axes(p0, p1)
        n_e = e_load / max(member.material.youngs_modulus_pa, 1e6)
        n_g = g_load / max(member.material.effective_shear_modulus_pa, 1e6)
        a_ext = member.profile.area_m2
        d_c = 0.5 * member.profile.width_m + 0.5 * thickness_m

        i_par = (a_ext * (n_e * a_f)) / max(a_ext + n_e * a_f, 1e-12) * (d_c**2)
        i_loc = n_e * (b_eff * (t_load**3)) / 12.0
        i_oop = eta_c * (i_par + i_loc)
        i_inp = eta_c * n_e * (t_load * (b_eff**3)) / 12.0 * 0.25
        dj = eta_c * n_g * (b_eff * (t_load**3)) / 3.0

        d_iy = 0.0
        d_iz = 0.0
        d_jt = 0.0
        for n_p in normals:
            cy = float(np.dot(n_p, ey)) ** 2
            cz = float(np.dot(n_p, ez)) ** 2
            # Local iz governs bending in ey; local iy governs bending in ez
            d_iz += cy * i_oop + cz * i_inp
            d_iy += cz * i_oop + cy * i_inp
            d_jt += dj

        member.profile = ProfileSpec.model_construct(
            id=member.profile.id,
            name=member.profile.name,
            shape=member.profile.shape,
            width_m=member.profile.width_m,
            height_m=member.profile.height_m,
            area_m2=member.profile.area_m2,
            iy_m4=member.profile.iy_m4 + d_iy,
            iz_m4=member.profile.iz_m4 + d_iz,
            j_m4=member.profile.j_m4 + d_jt,
        )


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
    kr_in = cj["k_r_inplane_nm_rad"]
    kr_tor = cj["k_r_torsion_nm_rad"]
    l_col = max(col_length_m, 0.15)

    for member in members:
        is_corexy_col = member.id.startswith("col_")
        is_cart_col = member.id.startswith("z_col_")
        if not (is_corexy_col or is_cart_col):
            continue

        e_pa = member.material.youngs_modulus_pa
        g_pa = member.material.effective_shear_modulus_pa
        eta_iy = 1.0 / (1.0 + (6.0 * e_pa * member.profile.iy_m4) / (l_col * kr_in))
        eta_iz = 1.0 / (1.0 + (6.0 * e_pa * member.profile.iz_m4) / (l_col * kr_in))
        eta_j = 1.0 / (1.0 + (2.0 * g_pa * member.profile.j_m4) / (l_col * kr_tor))

        if is_cart_col:
            # Bedslinger H-frame uprights bolt flat to base frame with tee plates
            eta_iy = 1.0 - 0.06 * (1.0 - eta_iy)
            eta_iz = 1.0 - 0.06 * (1.0 - eta_iz)
            eta_j = 1.0 - 0.06 * (1.0 - eta_j)

        member.profile = member.profile.model_copy(
            update={
                "iy_m4": member.profile.iy_m4 * eta_iy,
                "iz_m4": member.profile.iz_m4 * eta_iz,
                "j_m4": member.profile.j_m4 * eta_j,
            }
        )


def _attach_toolhead_cog_and_nozzle(
    nodes: list[StructuralNode],
    members: list[StructuralMember],
    spec: PrinterSpec,
    base_node_id: str = "toolhead",
) -> None:
    """Attach explicit 6-DOF toolhead_cog and nozzle_tip nodes via carriage mount & hotend stem."""
    nmap = {n.id: n for n in nodes}
    if base_node_id not in nmap:
        return

    base_node = nmap[base_node_id]
    base_pos = base_node.position_m
    toolhead_kg = spec.toolhead.total_mass_g * 1e-3

    # Linear carriage block mass stays on X-beam centerline
    base_node.lumped_mass_kg = 0.04

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
                off[2] -= 0.006
        elif x_mount == "bottom":
            for off in (cog_offset, nozzle_offset):
                if off[1] < 0.0:
                    off[1] = min(0.0, off[1] + 0.005)
                elif off[1] > 0.0:
                    off[1] = max(0.0, off[1] - 0.005)
            if cog_offset[2] < 0.0:
                cog_offset[2] = min(-0.004, cog_offset[2] + 0.012)
            if nozzle_offset[2] < 0.0:
                nozzle_offset[2] = min(-0.010, nozzle_offset[2] + 0.012)

    if float(np.linalg.norm(cog_offset)) < 0.004:
        cog_offset = np.array([0.0, 0.0, -0.004], dtype=float)
    cog_pos = base_pos + cog_offset

    nozzle_pos = base_pos + nozzle_offset
    stem_vec = nozzle_pos - cog_pos
    stem_len = float(np.linalg.norm(stem_vec))
    if stem_len < 0.005:
        if stem_len > 1e-9:
            nozzle_pos = cog_pos + (stem_vec / stem_len) * 0.005
        else:
            nozzle_pos = cog_pos + np.array([0.0, 0.0, -0.005], dtype=float)

    nodes.append(
        StructuralNode(
            id="toolhead_cog",
            position_m=cog_pos,
            lumped_mass_kg=max(toolhead_kg - 0.06, 0.02),
            role="cog",
        )
    )
    nodes.append(
        StructuralNode(
            id="nozzle_tip",
            position_m=nozzle_pos,
            lumped_mass_kg=0.02,
            role="nozzle",
        )
    )

    base_mount_mat = get_material(spec.toolhead.mount_material)
    e_mount_eff = base_mount_mat.effective_modulus_pa(spec.frame.chamber_temp_c)
    mount_mat = base_mount_mat.model_copy(
        update={
            "youngs_modulus_pa": e_mount_eff,
            "shear_modulus_pa": e_mount_eff / (2.0 * (1.0 + base_mount_mat.poisson_ratio)),
            "density_kg_m3": 1e-3,
            "cost_per_kg_usd": 0.0,
        }
    )
    iy_mount = 1.2e-8
    iz_mount = 1.8e-8
    j_mount = 1.4e-8

    if spec.architecture != "delta":
        from printsim.fea.components import hertzian_rail_carriage_stiffness

        l_mount = max(float(np.linalg.norm(cog_pos - base_pos)), 0.004)
        car = hertzian_rail_carriage_stiffness(
            spec.gantry.x_rail_profile,
            preload=spec.gantry.rail_preload,
            block_type=spec.gantry.rail_block_type,
        )
        car_mult = 2.0 if spec.gantry.effective_dual_x_rails else 1.0
        g_mount = mount_mat.effective_shear_modulus_pa

        # Series coupling: mount beam rotational/tip stiffness with 6-DOF Hertzian rail carriage
        k_rot_car_y = car["k_rx_nm_rad"] * car_mult
        k_rot_beam_y = e_mount_eff * iy_mount / l_mount
        eta_rot_y = k_rot_car_y / max(k_rot_car_y + k_rot_beam_y, 1e-9)

        k_tip_beam_y = 3.0 * e_mount_eff * iy_mount / (l_mount**3)
        k_rad_car_y = car["k_zz_n_m"] * car_mult
        eta_tip_y = k_rad_car_y / max(k_rad_car_y + k_tip_beam_y, 1e-9)

        k_rot_car_z = 0.5 * (car["k_ry_nm_rad"] + car["k_rz_nm_rad"]) * car_mult
        k_rot_beam_z = e_mount_eff * iz_mount / l_mount
        eta_rot_z = k_rot_car_z / max(k_rot_car_z + k_rot_beam_z, 1e-9)

        k_tip_beam_z = 3.0 * e_mount_eff * iz_mount / (l_mount**3)
        k_rad_car_z = car["k_yy_n_m"] * car_mult
        eta_tip_z = k_rad_car_z / max(k_rad_car_z + k_tip_beam_z, 1e-9)

        k_tor_car = car["k_rz_nm_rad"] * car_mult
        k_tor_beam = g_mount * j_mount / l_mount
        eta_tor = k_tor_car / max(k_tor_car + k_tor_beam, 1e-9)

        iy_mount *= eta_rot_y * eta_tip_y
        iz_mount *= eta_rot_z * eta_tip_z
        j_mount *= eta_tor

    mount_prof = ProfileSpec.model_construct(
        id="toolhead_mount_profile",
        name=f"Toolhead Carriage Mount ({spec.toolhead.mount_material})",
        shape="solid_bar",
        width_m=0.026,
        height_m=0.018,
        area_m2=2.2e-4,
        iy_m4=iy_mount,
        iz_m4=iz_mount,
        j_m4=j_mount,
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

    stem_mat = get_material("aluminum_6061_t6").model_copy(
        update={"density_kg_m3": 1e-3, "cost_per_kg_usd": 0.0}
    )
    stem_prof = ProfileSpec.model_construct(
        id="nozzle_stem_profile",
        name="Hotend Nozzle Stem",
        shape="rod",
        width_m=0.012,
        height_m=0.012,
        area_m2=1.2e-4,
        iy_m4=6.0e-9,
        iz_m4=6.0e-9,
        j_m4=8.0e-9,
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

    _apply_extrusion_filling(model)
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
    model = AssemblyModel(
        spec=spec,
        nodes=nodes,
        members=members,
        toolhead_node_id=toolhead_id,
        bed_center_m=np.array([w * 0.5, d * 0.5, 0.06]),
        build_origin_m=np.array([(w - bx) * 0.5, (d - by) * 0.5, 0.07]),
        x_moving_mass_kg=spec.toolhead.total_mass_g * 1e-3,
        y_moving_mass_kg=(spec.toolhead.total_mass_g + spec.gantry.carriage_mass_g) * 1e-3 + bkt_delta_kg,
    )
    _apply_extrusion_filling(model)
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
) -> str:
    """Locate an existing structural frame/rail node near ``p_target`` or split the spanning beam."""
    best_id: str | None = None
    best_dist = snap_tol_m
    for n in nodes:
        if n.role == "panel":
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
    non_panel = [n for n in nodes if n.role != "panel"]
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
            k_rot_per_m = 15.0 * clamp_eff

        nx_g = mesh_n
        ny_g = mesh_n if bh > 0.12 else max(1, mesh_n // 2)
        nvec = np.asarray(n_tup, dtype=float)
        off_nominal = d_c * nvec

        grid = np.empty((ny_g + 1, nx_g + 1), dtype=object)
        for j in range(ny_g + 1):
            eta = j / ny_g
            for i in range(nx_g + 1):
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

                if i in (0, nx_g) or j in (0, ny_g):
                    fid = _find_or_split_frame_node(nodes, nmap, members, p_frame)
                    ds = 0.0
                    if j in (0, ny_g):
                        ds += (bw / nx_g) * (0.5 if i in (0, nx_g) else 1.0)
                    if i in (0, nx_g):
                        ds += (bh / ny_g) * (0.5 if j in (0, ny_g) else 1.0)
                    actual_off = p_shell - nmap[fid].position_m
                    fastener_springs.append(
                        FastenerSpring(
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

    # Gantry plane near top of CoreXY frame
    z_gantry = h - 0.055
    xy_motor_kg = spec.kinematics.xy_stepper.motor_mass_g * 1e-3
    z_motor_kg = spec.kinematics.z_stepper.motor_mass_g * 1e-3
    bkt_info = GANTRY_BRACKET_CATALOG.get(
        spec.gantry.bracket_material, GANTRY_BRACKET_CATALOG["abs_printed"]
    )
    bkt_delta_kg = float(bkt_info["carriage_mass_delta_g"]) * 1e-3
    carriage_side_kg = (spec.gantry.carriage_mass_g * 0.5) * 1e-3 + 0.5 * bkt_delta_kg
    toolhead_kg = spec.toolhead.total_mass_g * 1e-3
    bed_kg = spec.bed.mass_g * 1e-3

    # Build origin centered in frame
    orig_x = (w - bx) * 0.5
    orig_y = (d - by) * 0.5
    orig_z = 0.08
    y_gantry = orig_y + v * by
    x_tool = orig_x + u * bx

    is_flying = spec.is_flying_gantry
    if is_flying:
        z_bed = orig_z
    else:
        z_min = min(0.08, 0.5 * z_gantry)
        z_max = max(z_min + 0.01, z_gantry - 0.045)
        z_stroke = min(bz, max(z_gantry - 0.14, 0.02))
        z_bed = float(np.clip(z_gantry - 0.045 - w_rel * z_stroke, z_min, z_max))

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
        # Physical heated bed center node
        StructuralNode("bed_surface", np.array([w * 0.5, d * 0.5, z_bed]), lumped_mass_kg=bed_kg, role="bed"),
    ]
    if not is_flying:
        nodes.extend(
            [
                StructuralNode("z_bed_fl", np.array([0.0, 0.0, z_bed]), role="frame"),
                StructuralNode("z_bed_fr", np.array([w, 0.0, z_bed]), role="frame"),
                StructuralNode("b_rear_mid", np.array([w * 0.5, d, 0.0]), role="frame"),
                StructuralNode("g_rear_mid", np.array([w * 0.5, d, z_gantry]), role="frame"),
                StructuralNode("z_bed_r", np.array([w * 0.5, d, z_bed]), role="bed"),
            ]
        )

    fp = spec.frame.profile
    fm = spec.frame.material
    bp = spec.frame.brace_profile
    bm = spec.frame.brace_material
    members: list[StructuralMember] = []

    # Base perimeter
    _add_member(members, "base_front", "b_fl", "b_fr", fp, fm)
    if is_flying:
        _add_member(members, "base_rear", "b_rl", "b_rr", fp, fm)
    else:
        _add_member(members, "base_rear_l", "b_rl", "b_rear_mid", fp, fm)
        _add_member(members, "base_rear_r", "b_rear_mid", "b_rr", fp, fm)
    _add_member(members, "base_left", "b_fl", "b_rl", fp, fm)
    _add_member(members, "base_right", "b_fr", "b_rr", fp, fm)

    # 4 vertical columns split at gantry height (and front columns split at bed height on Trident)
    for tag in ("fl", "fr", "rl", "rr"):
        if not is_flying and tag in ("fl", "fr"):
            _add_member(members, f"col_{tag}_bot", f"b_{tag}", f"z_bed_{tag}", fp, fm)
            _add_member(members, f"col_{tag}_mid", f"z_bed_{tag}", f"g_{tag}", fp, fm)
        else:
            _add_member(members, f"col_{tag}_low", f"b_{tag}", f"g_{tag}", fp, fm)
        _add_member(members, f"col_{tag}_top", f"g_{tag}", f"t_{tag}", fp, fm)

    # Top perimeter
    _add_member(members, "top_front", "t_fl", "t_fr", fp, fm)
    _add_member(members, "top_rear", "t_rl", "t_rr", fp, fm)
    _add_member(members, "top_left", "t_fl", "t_rl", fp, fm)
    _add_member(members, "top_right", "t_fr", "t_rr", fp, fm)

    # Mid gantry frame rails (standard Voron Trident / CoreXY layout: left/right Y rails + rear motor bar, open front door)
    _add_member(members, "y_rail_l_f", "g_fl", "y_car_l", fp, fm, role="rail")
    _add_member(members, "y_rail_l_r", "y_car_l", "g_rl", fp, fm, role="rail")
    _add_member(members, "y_rail_r_f", "g_fr", "y_car_r", fp, fm, role="rail")
    _add_member(members, "y_rail_r_r", "y_car_r", "g_rr", fp, fm, role="rail")
    if is_flying:
        _add_member(members, "gantry_rear_bar", "g_rl", "g_rr", fp, fm)
    else:
        _add_member(members, "gantry_rear_bar_l", "g_rl", "g_rear_mid", fp, fm)
        _add_member(members, "gantry_rear_bar_r", "g_rear_mid", "g_rr", fp, fm)

    # Structural bed mounting links (virtually massless so bed_surface.lumped_mass_kg carries exact bed mass)
    bed_vmat = _make_bed_virtual_material(get_material(spec.bed.plate_material))
    t_mm = max(float(spec.bed.plate_thickness_mm), 2.0)
    t_ratio = t_mm / 8.0
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
                    start_id=f"b_{tag}",
                    end_id="bed_surface",
                    profile=bed_arm_prof,
                    material=bed_vmat,
                    role="bed",
                )
            )
    else:
        frame_prof = get_profile(fp)
        z_rail_prof_ref = get_profile(spec.gantry.y_rail_profile)
        z_rail_vmat = _make_bed_virtual_material(get_material(fm))
        z_rail_prof = ProfileSpec.model_construct(
            id="bed_rear_z_rail_profile",
            name="Rear Z-Rail Backing Extrusion + MGN12",
            shape="solid_bar",
            width_m=frame_prof.width_m,
            height_m=frame_prof.height_m,
            area_m2=frame_prof.area_m2 + 0.5 * z_rail_prof_ref.area_m2,
            iy_m4=frame_prof.iy_m4 * 2.6,
            iz_m4=frame_prof.iz_m4 * 1.5,
            j_m4=frame_prof.j_m4 * 1.8,
        )
        members.append(
            StructuralMember(
                id="bed_rear_z_bot",
                start_id="b_rear_mid",
                end_id="z_bed_r",
                profile=z_rail_prof,
                material=z_rail_vmat,
                role="bed",
            )
        )
        members.append(
            StructuralMember(
                id="bed_rear_z_top",
                start_id="z_bed_r",
                end_id="g_rear_mid",
                profile=z_rail_prof,
                material=z_rail_vmat,
                role="bed",
            )
        )
        bed_arm_prof = ProfileSpec.model_construct(
            id="bed_trident_arm_profile",
            name=f"3-Point Z Bed Spider + Plate ({t_mm:.1f}mm {spec.bed.plate_material})",
            shape="solid_bar",
            width_m=0.020,
            height_m=max(0.012, 0.012 + t_mm * 1e-3),
            area_m2=6.0e-4,
            iy_m4=1.8e-8 + 7.2e-8 * (t_ratio**3),
            iz_m4=1.5e-7,
            j_m4=6.0e-8,
        )
        for tag, z_nid in (("fl", "z_bed_fl"), ("fr", "z_bed_fr"), ("r", "z_bed_r")):
            members.append(
                StructuralMember(
                    id=f"bed_arm_{tag}",
                    start_id=z_nid,
                    end_id="bed_surface",
                    profile=bed_arm_prof,
                    material=bed_vmat,
                    role="bed",
                )
            )

    # Moving X gantry beam (split at toolhead node)
    xp = spec.gantry.x_beam_profile
    xm = spec.gantry.x_beam_material
    _add_member(members, "x_gantry_l", "y_car_l", "toolhead", xp, xm, role="gantry")
    _add_member(members, "x_gantry_r", "toolhead", "y_car_r", xp, xm, role="gantry")

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
    for mid, side in (
        ("y_rail_l_f", "l"),
        ("y_rail_l_r", "l"),
        ("y_rail_r_f", "r"),
        ("y_rail_r_r", "r"),
    ):
        _, d_iz_y, _ = _apply_bolted_linear_rail(
            member_by_id[mid],
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

    _apply_corner_joint_compliance(members, spec, col_length_m=h)

    # Enclosure / structural MITC4 shell panels + Huth CBUSH fastener springs across rear, side, and top bays
    panels_active, panel_mounting, pmat, panel_t_m = _panel_shear_params(spec)
    shell_elements: list[ShellPanelElement] = []
    fastener_springs: list[FastenerSpring] = []
    total_panel_mass_kg = 0.0
    rep_panel_props: dict[str, float] | None = None
    if panels_active:
        bolted_to_gantry = spec.panels_bolted_to_gantry_effective
        shell_elements, fastener_springs, total_panel_mass_kg, rep_panel_props = (
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

    _attach_toolhead_cog_and_nozzle(nodes, members, spec, base_node_id="toolhead")

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
        panel_mass_kg=total_panel_mass_kg,
        x_rail_ei_y_boost_pct=x_rail_ei_y_boost_pct,
        x_rail_ei_z_boost_pct=x_rail_ei_z_boost_pct,
        y_rail_ei_x_boost_pct=y_rail_ei_x_boost_pct,
        total_rail_mass_kg=sum(m.rail_mass_kg for m in members),
        panel_rep_props=rep_panel_props,
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
    bkt_info = GANTRY_BRACKET_CATALOG.get(
        spec.gantry.bracket_material, GANTRY_BRACKET_CATALOG["abs_printed"]
    )
    bkt_delta_kg = float(bkt_info["carriage_mass_delta_g"]) * 1e-3
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

    _apply_corner_joint_compliance(members, spec, col_length_m=h)

    # Bedslinger: X moves toolhead; Y moves the entire heavy heated bed + carriage!
    bed_kg = spec.bed.mass_g * 1e-3
    x_moving = toolhead_kg + 0.03
    y_moving = bed_kg + 0.35 + bkt_delta_kg

    # Explicit bedslinger Y-bed node mounted to base frame via Y-rails (stiff X/Z, compliant Y carriage/belt)
    bed_center = np.array([w * 0.5, d * 0.5, orig_z])
    nodes.extend(
        [
            StructuralNode("bed_rail_l", np.array([0.0, d * 0.5, orig_z]), role="bed"),
            StructuralNode("bed_rail_r", np.array([w, d * 0.5, orig_z]), role="bed"),
            StructuralNode("bed_surface", bed_center.copy(), lumped_mass_kg=bed_kg, role="bed"),
        ]
    )
    bed_vmat = _make_bed_virtual_material(get_material(spec.bed.plate_material))
    rail_vmat = _make_bed_virtual_material(get_material(spec.gantry.rail_material))
    t_mm = max(float(spec.bed.plate_thickness_mm), 2.0)
    t_ratio = t_mm / 4.0
    y_rail_guide_prof = ProfileSpec.model_construct(
        id="bed_y_rail_guide_profile",
        name="Bedslinger Y-Rail Guide Support",
        shape="rod",
        width_m=0.010,
        height_m=0.010,
        area_m2=6.0e-4,
        iy_m4=8.0e-8,
        iz_m4=8.0e-8,
        j_m4=4.0e-8,
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
        iy_m4=1.2e-8 + 1.5e-8 * (t_ratio**3),
        iz_m4=iz_y_carriage,
        j_m4=2.0e-8,
    )
    for tag, rail_nid in (("l", "bed_rail_l"), ("r", "bed_rail_r")):
        members.append(
            StructuralMember(
                id=f"bed_arm_{tag}",
                start_id=rail_nid,
                end_id="bed_surface",
                profile=bed_car_prof,
                material=bed_vmat,
                role="bed",
            )
        )

    _attach_toolhead_cog_and_nozzle(nodes, members, spec, base_node_id="toolhead")
    if bkt_delta_kg > 0.0:
        next(n for n in nodes if n.id == "toolhead").lumped_mass_kg += bkt_delta_kg

    return AssemblyModel(
        spec=spec,
        nodes=nodes,
        members=members,
        toolhead_node_id="toolhead",
        toolhead_cog_node_id="toolhead_cog",
        nozzle_node_id="nozzle_tip",
        bed_node_id="bed_surface",
        gantry_node_ids=["z_l", "toolhead", "z_r"],
        bed_center_m=bed_center,
        build_origin_m=np.array([orig_x, orig_y, orig_z]),
        x_moving_mass_kg=x_moving,
        y_moving_mass_kg=y_moving,
        x_rail_ei_y_boost_pct=x_rail_ei_y_boost_pct,
        x_rail_ei_z_boost_pct=x_rail_ei_z_boost_pct,
        total_rail_mass_kg=sum(m.rail_mass_kg for m in members),
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
    bkt_info = GANTRY_BRACKET_CATALOG.get(
        spec.gantry.bracket_material, GANTRY_BRACKET_CATALOG["abs_printed"]
    )
    bkt_delta_kg = float(bkt_info["carriage_mass_delta_g"]) * 1e-3
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

    bed_kg = spec.bed.mass_g * 1e-3
    bed_center = np.array([cx, cy, 0.06])
    nodes.append(
        StructuralNode("bed_surface", bed_center.copy(), lumped_mass_kg=bed_kg, role="bed")
    )
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
                end_id="bed_surface",
                profile=delta_bed_prof,
                material=bed_vmat,
                role="bed",
            )
        )

    eff_mass = toolhead_kg + 0.09  # ultra-light effector + arm portion
    _attach_toolhead_cog_and_nozzle(nodes, members, spec, base_node_id="toolhead")
    if bkt_delta_kg > 0.0:
        next(n for n in nodes if n.id == "toolhead").lumped_mass_kg += bkt_delta_kg

    return AssemblyModel(
        spec=spec,
        nodes=nodes,
        members=members,
        toolhead_node_id="toolhead",
        toolhead_cog_node_id="toolhead_cog",
        nozzle_node_id="nozzle_tip",
        bed_node_id="bed_surface",
        gantry_node_ids=["c_0", "c_1", "c_2", "toolhead"],
        bed_center_m=bed_center,
        build_origin_m=np.array([cx - bx * 0.5, cy - by * 0.5, 0.06]),
        x_moving_mass_kg=eff_mass,
        y_moving_mass_kg=eff_mass + bkt_delta_kg,
        panel_mass_kg=total_panel_mass_kg,
        panel_rep_props=rep_panel_props,
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
    bed_node_mass = nmap[model.bed_node_id].lumped_mass_kg if has_bed_node else 0.0
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
