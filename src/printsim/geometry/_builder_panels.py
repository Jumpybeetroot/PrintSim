"""Panel mechanics, MITC4 enclosure shells, Trident backbone, and top/front braces extracted from ``builder.py``."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from printsim.schema import (
    FRONT_BRACE_CATALOG,
    TOP_BRACE_CATALOG,
    MaterialSpec,
    PrinterSpec,
    ProfileSpec,
    get_material,
    get_profile,
)

if TYPE_CHECKING:
    from printsim.geometry.builder import (
        FastenerSpring,
        ShellPanelElement,
        StructuralMember,
        StructuralNode,
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


def bolted_flange_rot_stiffness_per_m(
    flexural_d_nm: float,
    panel_t_m: float,
    frame_width_m: float,
    clamp_eff: float,
    bolt_pitch_m: float = 0.085,
) -> float:
    """Compute semi-rigid bolted extrusion-flange edge rotational stiffness (N*m / (rad * m)).

    Models a structural panel bolted along a single-row T-slot extrusion centerline (M5 bolts at
    ``bolt_pitch_m`` pitch on face width ``frame_width_m``) as two physical compliances in series:

    1. **VDI 2230 bolt + T-slot lip prying stiffness**: M5 steel bolt + 2.2 mm Al 6063 T-slot lip
       normal stiffness (``k_n_bolt ≈ 35 MN/m`` per bolt) acting at effective flange fulcrum lever
       arm ``e_eff = 0.35 * frame_width_m`` from the T-slot centerline.
    2. **Discrete-bolt VDI 2230 pressure-cone plate-edge stiffness**: along a single-row bolted
       edge at pitch ``p``, only the VDI 2230 clamped pressure cone ``d_cone = d_w + t`` (M5 DIN 125
       washer ``d_w = 10 mm``) actively resists edge lift-off (duty fraction ``alpha = d_cone / p``),
       yielding ``k_rot_plate = rot_eff * alpha * (1.5 * D / p)``.

    Rotational clamping requires both axial bolt preload and non-crushing face contact, so soft
    polymer thermal-clearance holes (``clamp_eff = 0.30``) and unpotted ACM soft LDPE cores
    (``clamp_eff = 0.22``) scale as ``clamp_eff**2``, whereas solid metal-on-metal bolted plates
    (``clamp_eff = 1.0``) achieve realistic semi-rigid clamping (naturally bounded in the global
    3D FEM by the extrusion beam's own finite Saint-Venant torsional stiffness ``G * J``).
    """
    rot_eff = max(float(clamp_eff), 0.0) ** 2
    if rot_eff <= 1e-6:
        return 0.2
    pitch = max(float(bolt_pitch_m), 0.025)
    d_cone = 0.010 + max(float(panel_t_m), 0.0)  # M5 DIN 125 washer OD + VDI 2230 cone spread
    cone_frac = min(1.0, d_cone / pitch)
    e_eff = 0.35 * max(float(frame_width_m), 0.015)
    k_n_bolt = 3.5e7  # N/m per M5 bolt + 2020/3030 T-slot lip in series
    k_rot_bolt = rot_eff * (k_n_bolt / pitch) * (e_eff * e_eff)
    k_rot_plate = rot_eff * cone_frac * (1.5 * max(float(flexural_d_nm), 1e-6) / pitch)
    return 1.0 / (1.0 / max(k_rot_bolt, 1e-6) + 1.0 / max(k_rot_plate, 1e-6))


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
    from printsim.geometry import builder as _b

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
        _b.StructuralMember(
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
        _b.StructuralMember(
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


def _split_member_at_node(
    members: list[StructuralMember],
    member_idx: int,
    split_node_id: str,
    frac: float,
) -> None:
    """Split ``members[member_idx]`` into two collinear segments at ``split_node_id``."""
    from printsim.geometry import builder as _b

    orig = members[member_idx]
    f_clamped = max(1e-4, min(1.0 - 1e-4, float(frac)))
    seg_a = _b.StructuralMember(
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
        insert_mass_kg=orig.insert_mass_kg * f_clamped,
    )
    suffix = 1
    existing_ids = {m.id for m in members}
    seg_b_id = f"{orig.id}__s{suffix}"
    while seg_b_id in existing_ids:
        suffix += 1
        seg_b_id = f"{orig.id}__s{suffix}"
    seg_b = _b.StructuralMember(
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
        insert_mass_kg=orig.insert_mass_kg * (1.0 - f_clamped),
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
    from printsim.geometry import builder as _b

    excluded_roles = {"panel", "backbone", "toolhead", "cog", "nozzle", "bed"}
    if not allow_gantry:
        excluded_roles.add("gantry")
    best_id: str | None = None
    best_dist = snap_tol_m
    for n in nodes:
        if n.role in excluded_roles:
            continue
        if n.id == "z_col_bed_r" and not allow_rear_z:
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
                node = _b.StructuralNode(
                    id=split_id,
                    position_m=(p0 + proj * u).copy(),
                    role="frame",
                )
                nodes.append(node)
                nmap[split_id] = node
                _split_member_at_node(members, idx, split_id, proj / length)
                return split_id

    # Fallback: closest frame/rail node
    non_panel = [
        n
        for n in nodes
        if n.role not in excluded_roles and (allow_rear_z or n.id != "z_col_bed_r")
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
    from printsim.geometry import builder as _b

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

    shell_elements: list[_b.ShellPanelElement] = []
    fastener_springs: list[_b.FastenerSpring] = []
    total_panel_mass_kg = 0.0
    rep_panel_props: dict[str, float] | None = None
    # Share shell nodes across *_low / *_top seam so bolted-to-gantry side & rear walls
    # behave as continuous one-piece sheets (preserving C1 out-of-plane bending continuity).
    seam_nodes: dict[tuple[str, int], str] = {}
    seam_springs: dict[tuple[str, int], _b.FastenerSpring] = {}

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
            k_rot_per_m = bolted_flange_rot_stiffness_per_m(
                flexural_d_nm=props["flexural_d_nm"],
                panel_t_m=panel_t_m,
                frame_width_m=frame_prof.width_m,
                clamp_eff=clamp_eff,
            )

        has_rear_z_col = (
            mounting != "clip_enclosure"
            and bay in ("rear", "rear_low", "rear_top")
            and spec.bed.z_drive in ("leadscrew_triple", "leadscrew_single")
            and "b_rear_mid" in nmap
            and "g_rear_mid" in nmap
            and "z_col_bed_r" in nmap
        )
        # Ensure rear bays with a rear-center vertical Z-column (Trident / single-Z)
        # have an exact centerline grid column (xi = 0.5) so the rear panel contacts
        # b_rear_mid -> bed_rear_z_bot -> z_col_bed_r -> bed_rear_z_top -> g_rear_mid
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
                node = _b.StructuralNode(id=sid, position_m=p_shell, role="panel")
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
                    sp = _b.FastenerSpring(
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
                    # vertical 2020 Z-extrusion (bed_rear_z_bot / z_col_bed_r / bed_rear_z_top):
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
                        _b.FastenerSpring(
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
                    _b.ShellPanelElement(
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
    from printsim.geometry import builder as _b

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
        ey, ez = _b._member_local_axes(nmap[m.start_id].position_m, nmap[m.end_id].position_m)
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
    k_rot_per_m = bolted_flange_rot_stiffness_per_m(
        flexural_d_nm=props["flexural_d_nm"],
        panel_t_m=t_bb,
        frame_width_m=get_profile(spec.frame.profile).width_m,
        clamp_eff=1.0,
    )

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
    shell_elements: list[_b.ShellPanelElement] = []
    fastener_springs: list[_b.FastenerSpring] = []
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
            node = _b.StructuralNode(id=sid, position_m=p_shell, role="backbone")
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
                    _b.FastenerSpring(
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
                _b.ShellPanelElement(
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
    edge_subdiv: int = 2,
) -> tuple[list[ShellPanelElement], list[FastenerSpring], float]:
    """Generate an open-center MITC4 ring (perimeter strips around cutout) + perimeter CBUSH springs."""
    from printsim.geometry import builder as _b

    bw = float(np.linalg.norm(p10 - p00))
    bh = float(np.linalg.norm(p01 - p00))
    x_cut0_m = 0.5 * (bw - cutout_w_m)
    x_cut1_m = 0.5 * (bw + cutout_w_m)
    y_cut0_m = 0.5 * (bh - cutout_h_m)
    y_cut1_m = 0.5 * (bh + cutout_h_m)

    u1 = max(0.05, min(0.45, x_cut0_m / max(bw, 1e-6)))
    u3 = min(0.95, max(0.55, x_cut1_m / max(bw, 1e-6)))
    n_mid = max(2, min(int(edge_subdiv), 6))
    u_stops = (
        0.0,
        u1,
        *(u1 + (u3 - u1) * k / n_mid for k in range(1, n_mid)),
        u3,
        1.0,
    )

    v1 = max(0.05, min(0.45, y_cut0_m / max(bh, 1e-6)))
    v3 = min(0.95, max(0.55, y_cut1_m / max(bh, 1e-6)))
    v_stops = (
        0.0,
        v1,
        *(v1 + (v3 - v1) * k / n_mid for k in range(1, n_mid)),
        v3,
        1.0,
    )

    nx_g = len(u_stops) - 1  # n_mid + 2
    ny_g = len(v_stops) - 1  # n_mid + 2
    xs_m = [u * bw for u in u_stops]
    ys_m = [v * bh for v in v_stops]
    nvec = np.asarray(n_tup, dtype=float)
    off_nominal = d_c * nvec

    grid = np.empty((ny_g + 1, nx_g + 1), dtype=object)
    shells: list[_b.ShellPanelElement] = []
    springs: list[_b.FastenerSpring] = []
    for j in range(ny_g + 1):
        eta = v_stops[j]
        for i in range(nx_g + 1):
            if 1 < i < nx_g - 1 and 1 < j < ny_g - 1:
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
            node = _b.StructuralNode(id=sid, position_m=p_shell, role="panel")
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
                    _b.FastenerSpring(
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
                _b.ShellPanelElement(
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
    from printsim.geometry import builder as _b

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
        mem = _b.StructuralMember(
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
    from printsim.geometry import builder as _b

    w = spec.frame.width_mm * 1e-3
    d = spec.frame.depth_mm * 1e-3
    h = spec.frame.height_mm * 1e-3
    allow_gantry = not spec.is_flying_gantry

    shells: list[_b.ShellPanelElement] = []
    springs: list[_b.FastenerSpring] = []
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
        # 5 mm CNC halo / portal plates bolt on dense ~45 mm CNC pitch across the full extrusion face
        k_rot_per_m = bolted_flange_rot_stiffness_per_m(
            flexural_d_nm=props["flexural_d_nm"],
            panel_t_m=t_plate,
            frame_width_m=frame_prof.width_m,
            clamp_eff=1.0,
            bolt_pitch_m=0.045,
        )
        e_mem_eff = 0.92 * al5052.youngs_modulus_pa * t_plate
        return e_mem_eff, k_shear_per_m, k_normal_per_m, k_rot_per_m

    def _close_tslot_torsion(prefixes: tuple[str, ...]) -> None:
        """Add Bredt-Batho closed-T-slot torsional rigidity when a 5 mm CNC plate bridges an extrusion face."""
        for m in members:
            if m.role == "frame" and m.id.startswith(prefixes):
                w_f = m.profile.width_m
                j_cell = 0.0235 * (w_f**4)
                m.profile = m.profile.model_copy(update={"j_m4": m.profile.j_m4 + j_cell})

    # 1. Open-center Top Brace
    top_mass_kg = 0.0
    top_cost_usd = 0.0
    top_active = top_mode != "none"
    top_is_shell = False
    if top_active:
        top_entry = TOP_BRACE_CATALOG.get(top_mode, TOP_BRACE_CATALOG["none"])
        top_cost_usd = float(top_entry.get("cost_usd", 0.0))
        if top_mode == "cnc_halo_ring":
            _close_tslot_torsion(("top_",))
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
                edge_subdiv=panel_mesh_n,
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
            _close_tslot_torsion(("top_front", "base_front", "col_fl", "col_fr"))
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
                edge_subdiv=panel_mesh_n,
            )
            shells.extend(f_shells)
            springs.extend(f_springs)
            front_is_shell = True
        elif front_mode == "header_and_skirt":
            _close_tslot_torsion(("top_front", "base_front"))
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
                        node = _b.StructuralNode(id=sid, position_m=p_shell, role="panel")
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
                                    _b.FastenerSpring(
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
                            _b.ShellPanelElement(
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
