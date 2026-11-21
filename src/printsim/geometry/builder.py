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
    toolhead_cog_node_id: str = "toolhead_cog"
    nozzle_node_id: str = "nozzle_tip"
    gantry_node_ids: list[str] = field(default_factory=list)
    bed_center_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    build_origin_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    x_moving_mass_kg: float = 0.0
    y_moving_mass_kg: float = 0.0
    frame_mass_kg: float = 0.0
    panel_mass_kg: float = 0.0
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

    @property
    def node_map(self) -> dict[str, StructuralNode]:
        return {n.id: n for n in self.nodes}

    def to_geometry_dict(self) -> dict:
        """Serialize assembly geometry (mm) for the 3D web viewer and AI tools."""
        nmap = self.node_map
        pmat = get_material(self.spec.frame.panel_material)
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
        c_bolt = (
            (((t_bear + t_ext) / (2.0 * d_bolt)) ** a_huth)
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
        "col_fl_top": [nx],
        "col_fr_low": [nx],
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
    if float(np.linalg.norm(cog_offset)) < 0.004:
        cog_offset = np.array([0.0, 0.0, -0.004], dtype=float)
    cog_pos = base_pos + cog_offset

    nozzle_offset = np.array(spec.toolhead.nozzle_offset_mm, dtype=float) * 1e-3
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

    mount_mat = get_material(spec.toolhead.mount_material).model_copy(
        update={"density_kg_m3": 1e-3, "cost_per_kg_usd": 0.0}
    )
    mount_prof = ProfileSpec.model_construct(
        id="toolhead_mount_profile",
        name=f"Toolhead Carriage Mount ({spec.toolhead.mount_material})",
        shape="solid_bar",
        width_m=0.026,
        height_m=0.018,
        area_m2=2.2e-4,
        iy_m4=1.2e-8,
        iz_m4=1.8e-8,
        j_m4=1.4e-8,
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

    # Mid gantry frame rails (standard Voron Trident / CoreXY layout: left/right Y rails + rear motor bar, open front door)
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
    panels_active, panel_mounting, pmat, panel_t_m = _panel_shear_params(spec)
    total_panel_mass_kg = 0.0
    rep_panel_props: dict[str, float] | None = None
    if panels_active:
        bolted_to_gantry = spec.panels_bolted_to_gantry_effective
        if bolted_to_gantry:
            bays = (
                ("rear_low", "b_rl", "b_rr", "g_rl", "g_rr"),
                ("rear_top", "g_rl", "g_rr", "t_rl", "t_rr"),
                ("left_low", "b_fl", "b_rl", "g_fl", "g_rl"),
                ("left_top", "g_fl", "g_rl", "t_fl", "t_rl"),
                ("right_low", "b_fr", "b_rr", "g_fr", "g_rr"),
                ("right_top", "g_fr", "g_rr", "t_fr", "t_rr"),
                ("top", "t_fl", "t_fr", "t_rl", "t_rr"),
            )
        else:
            bays = (
                ("rear", "b_rl", "b_rr", "t_rl", "t_rr"),
                ("left", "b_fl", "b_rl", "t_fl", "t_rl"),
                ("right", "b_fr", "b_rr", "t_fr", "t_rr"),
                ("top", "t_fl", "t_fr", "t_rl", "t_rr"),
            )

        bay_aeq: dict[str, float] = {}
        for bay, bl_id, br_id, tl_id, tr_id in bays:
            is_split_wall = bolted_to_gantry and bay != "top"
            a_eq, bay_mass, props = _add_shear_panel_bay(
                members,
                nmap,
                bay,
                bl_id,
                br_id,
                tl_id,
                tr_id,
                pmat,
                panel_t_m,
                panel_mounting,
                bolted_to_gantry=is_split_wall,
                wall_height_m=h if is_split_wall else None,
            )
            bay_aeq[bay] = a_eq
            total_panel_mass_kg += bay_mass
            if rep_panel_props is None and bay in ("rear_low", "rear"):
                rep_panel_props = props

        if panel_mounting == "bolted_structural" and rep_panel_props is not None:
            _apply_composite_panel_stiffening(
                members,
                nmap,
                pmat,
                panel_t_m,
                rep_panel_props,
                bolted_to_gantry=bolted_to_gantry,
            )

        if bolted_to_gantry:
            virtual_mat = _make_panel_virtual_material(pmat)
            for side, f_bot, r_bot, car_id, bay_key in (
                ("l", "b_fl", "b_rl", "y_car_l", "left_low"),
                ("r", "b_fr", "b_rr", "y_car_r", "right_low"),
            ):
                yrail_prof = ProfileSpec.model_construct(
                    id=f"panel_shear_yrail_{side}",
                    name=f"Gantry Y-Rail Panel Tie-In ({side}, {panel_t_m * 1e3:.1f}mm {pmat.id})",
                    shape="plate",
                    width_m=max(panel_t_m, 0.002),
                    height_m=max(panel_t_m, 0.004),
                    area_m2=max(0.30 * bay_aeq[bay_key], 1e-10),
                    iy_m4=1e-11,
                    iz_m4=1e-11,
                    j_m4=1e-11,
                )
                members.append(
                    StructuralMember(
                        id=f"panel_shear_yrail_{side}_1",
                        start_id=f_bot,
                        end_id=car_id,
                        profile=yrail_prof,
                        material=virtual_mat,
                        role="brace",
                        is_truss=True,
                    )
                )
                members.append(
                    StructuralMember(
                        id=f"panel_shear_yrail_{side}_2",
                        start_id=r_bot,
                        end_id=car_id,
                        profile=yrail_prof,
                        material=virtual_mat,
                        role="brace",
                        is_truss=True,
                    )
                )

    # Calculate moving masses
    x_beam_mass = sum(m.mass_kg(nmap) for m in members if m.role == "gantry")
    x_rail_mass = get_profile(spec.gantry.x_rail_profile).area_m2 * w * get_material(spec.gantry.rail_material).density_kg_m3
    if spec.gantry.dual_x_rails:
        x_rail_mass *= 2.0

    # In CoreXY, X moves toolhead; Y moves toolhead + X beam + X rail + Y carriages
    x_moving = toolhead_kg + 0.04  # plus belt segment
    y_moving = toolhead_kg + x_beam_mass + x_rail_mass + spec.gantry.carriage_mass_g * 1e-3

    _attach_toolhead_cog_and_nozzle(nodes, members, spec, base_node_id="toolhead")

    return AssemblyModel(
        spec=spec,
        nodes=nodes,
        members=members,
        toolhead_node_id="toolhead",
        toolhead_cog_node_id="toolhead_cog",
        nozzle_node_id="nozzle_tip",
        gantry_node_ids=["y_car_l", "toolhead", "y_car_r"],
        bed_center_m=np.array([w * 0.5, d * 0.5, orig_z]),
        build_origin_m=np.array([orig_x, orig_y, orig_z]),
        x_moving_mass_kg=x_moving,
        y_moving_mass_kg=y_moving,
        panel_mass_kg=total_panel_mass_kg,
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

    _attach_toolhead_cog_and_nozzle(nodes, members, spec, base_node_id="toolhead")

    return AssemblyModel(
        spec=spec,
        nodes=nodes,
        members=members,
        toolhead_node_id="toolhead",
        toolhead_cog_node_id="toolhead_cog",
        nozzle_node_id="nozzle_tip",
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

    eff_mass = toolhead_kg + 0.09  # ultra-light effector + arm portion
    _attach_toolhead_cog_and_nozzle(nodes, members, spec, base_node_id="toolhead")

    return AssemblyModel(
        spec=spec,
        nodes=nodes,
        members=members,
        toolhead_node_id="toolhead",
        toolhead_cog_node_id="toolhead_cog",
        nozzle_node_id="nozzle_tip",
        gantry_node_ids=["c_0", "c_1", "c_2", "toolhead"],
        bed_center_m=np.array([cx, cy, 0.06]),
        build_origin_m=np.array([cx - bx * 0.5, cy - by * 0.5, 0.06]),
        x_moving_mass_kg=eff_mass,
        y_moving_mass_kg=eff_mass,
        panel_mass_kg=total_panel_mass_kg,
        panel_rep_props=rep_panel_props,
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
    if model.panel_mass_kg > 0.0:
        pmat = get_material(model.spec.frame.panel_material)
        cost += model.panel_mass_kg * pmat.cost_per_kg_usd
    # Add baseline estimate for motors, bed plate, rails, and hotend hardware
    bed_mat = get_material(model.spec.bed.plate_material)
    cost += bed_mass * bed_mat.cost_per_kg_usd + 140.0  # steppers/electronics/hotend baseline
    model.estimated_material_cost_usd = cost
