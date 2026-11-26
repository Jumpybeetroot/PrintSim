"""Linear rail Newmark composite mechanics, CF inserts, and gantry carriage helpers extracted from ``builder.py``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from printsim.schema import (
    MaterialSpec,
    PrinterSpec,
    ProfileSpec,
    get_material,
    get_profile,
    resolve_rail_quality,
)

if TYPE_CHECKING:
    from printsim.geometry.builder import (
        AssemblyModel,
        StructuralMember,
        StructuralNode,
    )


def corexy_x_beam_span_m(spec: PrinterSpec) -> float:
    """Physical CoreXY X-gantry beam cut length between left/right XY-joint mounting faces (m).

    Subtracts the left/right vertical frame uprights (``2 * b_upright``) plus left/right
    XY-joint bracket and Y-carriage mounting offsets (``2 * 20 mm = 40 mm``), matching real
    Voron 2.4 / Trident / Carbonara beams (330 / 380 / 430 mm on 410 / 460 / 510 mm 2020
    frames; 380 mm on 500 mm 4040 DoomCube 300; 400 mm on 500 mm 3030 RatRig 300).
    """
    w = float(spec.frame.width_mm) * 1e-3
    if spec.architecture != "corexy":
        return max(w, 0.12)
    bx = float(spec.kinematics.build_volume_mm[0]) * 1e-3
    b_up = float(get_profile(spec.frame.effective_upright_profile).width_m)
    inner_span = w - 2.0 * (b_up + 0.020)
    return float(min(w, max(inner_span, min(w - 2.0 * b_up, bx + 0.050), 0.12)))


def corexy_x_rail_len_m(spec: PrinterSpec) -> float:
    """Physical CoreXY X-axis linear rail cut length bolted to the X-gantry beam (m).

    Covers X build travel plus carriage-block and endstop clearance (``bx + 50 mm`` ->
    exact official Voron BOM 300 / 350 / 400 mm MGN12R rails for 250 / 300 / 350 mm
    builds), bounded by the physical X-beam span.
    """
    beam_span = corexy_x_beam_span_m(spec)
    if spec.architecture != "corexy":
        return beam_span
    bx = float(spec.kinematics.build_volume_mm[0]) * 1e-3
    return float(min(beam_span, max(bx + 0.050, 0.12)))


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


def _compute_newmark_rail_gamma(
    member: StructuralMember,
    rail_profile: ProfileSpec,
    rail_material: MaterialSpec,
    beam_shape: str | None,
    bolt_pitch_m: float,
    l_eff_m: float,
    chamber_temp_c: float,
    cf_rail_mounting: str = "single_wall_bare",
) -> float:
    """Compute Newmark (1951) partial-interaction shear-lag gamma for a rail/backer on a member."""
    shape = beam_shape or member.profile.shape
    if member.profile.id.startswith("cf_") or member.material.id.startswith("carbon_fiber"):
        sub_kind = "carbon_tube"
    elif shape == "solid_bar":
        sub_kind = "solid_bar"
    else:
        sub_kind = "extrusion"

    from printsim.fea.components import newmark_rail_composite_gamma

    return newmark_rail_composite_gamma(
        e_rail_pa=rail_material.youngs_modulus_pa,
        a_rail_m2=rail_profile.area_m2,
        e_beam_pa=member.material.youngs_modulus_pa,
        a_beam_m2=member.profile.area_m2,
        bolt_pitch_m=bolt_pitch_m,
        l_eff_m=l_eff_m,
        substrate_kind=sub_kind,
        wall_thickness_m=member.profile.wall_thickness_m,
        matrix_shear_retention=member.material.shear_modulus_retention_at_temp(chamber_temp_c),
        cf_rail_mounting=cf_rail_mounting,
    )


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
    chamber_temp_c: float = 25.0,
    cf_rail_mounting: str = "single_wall_bare",
    mass_length_scale: float = 1.0,
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
    eta_rail = _compute_newmark_rail_gamma(
        member=member,
        rail_profile=rail_profile,
        rail_material=rail_material,
        beam_shape=beam_shape,
        bolt_pitch_m=bolt_pitch_m,
        l_eff_m=l_eff_m if l_eff_m is not None else length,
        chamber_temp_c=chamber_temp_c,
        cf_rail_mounting=cf_rail_mounting,
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

    is_cf_sub = member.profile.id.startswith("cf_") or member.material.id.startswith("carbon_fiber")
    # 3D Tet10 FEA torsion calibration: discrete M3 center bolts mobilize partial rail J
    # (~61% on T-nut bolted 2020 extrusion, ~18-28% on CF tube before insert corner boost).
    j_coup = (
        min(0.70, 0.04 + 0.35 * eta_rail)
        if is_cf_sub
        else min(0.85, 0.12 + 0.72 * eta_rail)
    )

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
        d_j += j_coup * n_g * j_local

    member.profile = member.profile.model_copy(
        update={
            "iy_m4": member.profile.iy_m4 + d_iy,
            "iz_m4": member.profile.iz_m4 + d_iz,
            "j_m4": member.profile.j_m4 + d_j,
        }
    )

    m_rail_member = (
        len(mount_normals)
        * rail_profile.area_m2
        * (length * max(float(mass_length_scale), 0.0))
        * rail_material.density_kg_m3
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


def _apply_bolted_rail_with_opposite_backer(
    member: StructuralMember,
    node_map: dict[str, StructuralNode],
    rail_profile: ProfileSpec,
    rail_material: MaterialSpec,
    backer_profile: ProfileSpec,
    backer_material: MaterialSpec,
    mount_normal: np.ndarray,
    beam_shape: str | None = None,
    bolt_pitch_m: float = 0.025,
    l_eff_m: float | None = None,
    chamber_temp_c: float = 25.0,
    cf_rail_mounting: str = "single_wall_bare",
    mass_length_scale: float = 1.0,
) -> tuple[float, float, float]:
    """Model a bolted linear rail paired with an opposite-face backer using exact sandwich neutral-axis shift."""
    p0 = node_map[member.start_id].position_m
    p1 = node_map[member.end_id].position_m
    length = float(np.linalg.norm(p1 - p0))
    if length <= 1e-9:
        return 0.0, 0.0, 0.0

    ey, ez = _member_local_axes(p0, p1)
    l_span = l_eff_m if l_eff_m is not None else length
    eta_r = _compute_newmark_rail_gamma(
        member=member,
        rail_profile=rail_profile,
        rail_material=rail_material,
        beam_shape=beam_shape,
        bolt_pitch_m=bolt_pitch_m,
        l_eff_m=l_span,
        chamber_temp_c=chamber_temp_c,
        cf_rail_mounting=cf_rail_mounting,
    )
    eta_b = _compute_newmark_rail_gamma(
        member=member,
        rail_profile=backer_profile,
        rail_material=backer_material,
        beam_shape=beam_shape,
        bolt_pitch_m=bolt_pitch_m,
        l_eff_m=l_span,
        chamber_temp_c=chamber_temp_c,
        cf_rail_mounting=cf_rail_mounting,
    )

    e_beam = max(member.material.youngs_modulus_pa, 1e6)
    g_beam = max(member.material.effective_shear_modulus_pa, 1e6)
    n_r = rail_material.youngs_modulus_pa / e_beam
    n_b = backer_material.youngs_modulus_pa / e_beam
    n_gr = rail_material.effective_shear_modulus_pa / g_beam
    n_gb = backer_material.effective_shear_modulus_pa / g_beam

    n_norm = float(np.linalg.norm(mount_normal))
    n_unit = mount_normal / n_norm if n_norm > 1e-12 else mount_normal
    cy = float(np.dot(n_unit, ey)) ** 2
    cz = float(np.dot(n_unit, ez)) ** 2
    beam_half = 0.5 * (cy * member.profile.width_m + cz * member.profile.height_m)
    d_1 = beam_half + 0.5 * rail_profile.height_m
    d_2 = beam_half + 0.5 * backer_profile.height_m

    a_beam = member.profile.area_m2
    a_r_eff = eta_r * n_r * rail_profile.area_m2
    a_b_eff = eta_b * n_b * backer_profile.area_m2
    y_na = (a_r_eff * d_1 - a_b_eff * d_2) / max(a_beam + a_r_eff + a_b_eff, 1e-12)
    i_par = (
        a_beam * (y_na**2)
        + a_r_eff * ((d_1 - y_na) ** 2)
        + a_b_eff * ((d_2 + y_na) ** 2)
    )

    i_local_norm = n_r * rail_profile.iy_m4 + n_b * backer_profile.iy_m4
    i_local_tang = n_r * rail_profile.iz_m4 + n_b * backer_profile.iz_m4
    d_iz = cy * (i_local_norm + i_par) + cz * i_local_tang
    d_iy = cz * (i_local_norm + i_par) + cy * i_local_tang
    is_cf_sub = member.profile.id.startswith("cf_") or member.material.id.startswith("carbon_fiber")
    # 3D Tet10 FEA torsion calibration on opposite-face rail+backer sandwich:
    # (~35% rail+backer St. Venant J transfer on CF before 2-wall insert boost -> ~231 vs 227 Nm^2 FEA;
    #  ~78% flange-coupled sandwich closure on 2020 Al -> ~100.5 vs 101.8 Nm^2 FEA).
    j_coup_bk = 0.35 if is_cf_sub else 0.78
    d_j = j_coup_bk * (n_gr * rail_profile.j_m4 + n_gb * backer_profile.j_m4)

    member.profile = member.profile.model_copy(
        update={
            "iy_m4": member.profile.iy_m4 + d_iy,
            "iz_m4": member.profile.iz_m4 + d_iz,
            "j_m4": member.profile.j_m4 + d_j,
        }
    )

    eff_mass_len = length * max(float(mass_length_scale), 0.0)
    m_rail_member = rail_profile.area_m2 * eff_mass_len * rail_material.density_kg_m3
    m_backer_member = backer_profile.area_m2 * eff_mass_len * backer_material.density_kg_m3
    rho_delta = (m_rail_member + m_backer_member) / max(a_beam * length, 1e-12)
    member.material = member.material.model_copy(
        update={"density_kg_m3": member.material.density_kg_m3 + rho_delta}
    )
    member.rail_mass_kg += m_rail_member
    member.backer_mass_kg += m_backer_member
    member.backer_material_id = backer_material.id
    return d_iy, d_iz, d_j


def _apply_cf_two_wall_clamped_inserts(
    member_by_id: dict[str, StructuralMember],
    node_map: dict[str, StructuralNode],
    spec: PrinterSpec,
    span_m: float,
) -> float:
    """Apply CF tube end-clamping & Carbonara 2-wall clamped internal insert mechanics and mass.

    - ``single_wall_bare``: Hollow thin-wall CF tube ends cannot be torqued rigidly into XY-joints
      without crushing/delaminating the 1–2 mm walls, yielding semi-rigid end-clamping knockdown on
      bending (``iy_m4``, ``iz_m4``) and torsion (``j_m4``).
    - ``two_wall_clamped_inserts`` (Carbonara): Solid 100%-infill annealed CF/GF internal inserts +
      M5 hex end standoffs + M3x28 Grade 5 Ti bolts clamp both CF walls rigidly at the XY-joints
      (eliminating hollow-end crush compliance), add a +14% closed-box torsional corner transfer
      boost, and carry incremental internal-rail-insert + longer-Ti-bolt mass
      ``(4.0 + 18.0 * span_m) g`` (~11–13 g; XY-joint end fasteners are already in ``carriage_mass_g``).
    """
    cf_mount = spec.gantry.effective_cf_rail_mounting
    if cf_mount == "none":
        return 0.0
    gantry_mids = [mid for mid in ("x_gantry_l", "x_gantry_r") if mid in member_by_id]
    if not gantry_mids:
        return 0.0
    if cf_mount == "single_wall_bare":
        x_beam_prof = get_profile(spec.gantry.x_beam_profile)
        t_wall_mm = max(
            float(
                x_beam_prof.wall_thickness_m
                if x_beam_prof.wall_thickness_m is not None
                else 2.0e-3
            )
            * 1e3,
            0.8,
        )
        # Hollow thin-wall CF tube ends crush under M5 XY-joint torque without internal end inserts.
        # Calibrated to 3D Tet10 datum+M5-bolt FEA across 1.0 / 1.8 / 2.0 mm walls:
        # eta_bend = 0.523 / 0.840 / 0.909; eta_tor = 0.355 / 0.677 / 0.749.
        eta_end_bend = min(0.92, max(0.50, 0.90 * ((t_wall_mm / 2.0) ** 0.78)))
        eta_end_tor = min(0.80, max(0.35, 0.75 * ((t_wall_mm / 2.0) ** 1.05)))
        for mid in gantry_mids:
            m_xb = member_by_id[mid]
            m_xb.profile = m_xb.profile.model_copy(
                update={
                    "iy_m4": m_xb.profile.iy_m4 * eta_end_bend,
                    "iz_m4": m_xb.profile.iz_m4 * eta_end_bend,
                    "j_m4": m_xb.profile.j_m4 * eta_end_tor,
                }
            )
        return 0.0

    # Incremental mass of internal printed rail inserts + M3x28 Ti bolt overhang (~11-13 g)
    insert_mass_kg = (4.0 + 18.0 * max(float(span_m), 0.10)) * 1e-3
    per_member_kg = insert_mass_kg / len(gantry_mids)
    for mid in gantry_mids:
        m_xb = member_by_id[mid]
        length = m_xb.length_m(node_map)
        # 3D Tet10 FEA corner load-transfer boost from compressed 2-wall PET-CF inserts (+14% -> ~222 Nm^2 vs 221.9 FEA)
        m_xb.profile = m_xb.profile.model_copy(update={"j_m4": m_xb.profile.j_m4 * 1.14})
        if length > 1e-9:
            rho_delta = per_member_kg / max(m_xb.profile.area_m2 * length, 1e-12)
            m_xb.material = m_xb.material.model_copy(
                update={"density_kg_m3": m_xb.material.density_kg_m3 + rho_delta}
            )
            m_xb.insert_mass_kg += per_member_kg
    return insert_mass_kg


def _oriented_bare_rail_profile(x_rail_prof: ProfileSpec, x_mount: str) -> ProfileSpec:
    """Orient a bare linear rail cross-section to match its mounting face on an unsupported X-gantry.

    In local beam coordinates along +X, local y is global Y (front-to-back) and local z is global Z.
    - ``top``/``bottom``/``dual_top_bottom``: rail lies flat (wide dimension along Y, height along Z).
    - ``front``: rail stands vertically (8 mm thickness along Y -> iz resists Y-bending, 12 mm width along Z -> iy resists Z-bending).
    """
    if x_mount in ("top", "bottom", "dual_top_bottom"):
        return x_rail_prof.model_copy(
            update={"id": x_rail_prof.id, "shape": "rail"}
        )
    return x_rail_prof.model_copy(
        update={
            "id": x_rail_prof.id,
            "shape": "rail",
            "width_m": x_rail_prof.height_m,
            "height_m": x_rail_prof.width_m,
            "iy_m4": x_rail_prof.iz_m4,
            "iz_m4": x_rail_prof.iy_m4,
        }
    )


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


_BASELINE_X_CARRIAGE_BLOCK_MASS_G = 54.0  # Standard single HIWIN MGN12H carriage block mass (g)
_BASELINE_Y_CARRIAGE_PAIR_MASS_G = 108.0  # Standard 2x HIWIN MGN12H Y-carriage blocks (g)


def x_carriage_block_mass_delta_g(spec: PrinterSpec) -> tuple[float, float]:
    """Return ``(total_x_block_mass_g, delta_vs_mgn12h_g)`` from official HIWIN/Misumi carriage weights."""
    from printsim.fea.components import hertzian_rail_carriage_stiffness

    rail_qual = resolve_rail_quality(spec)
    car = hertzian_rail_carriage_stiffness(
        spec.gantry.x_rail_profile,
        preload=spec.gantry.rail_preload,
        block_type=spec.gantry.rail_block_type,
        rail_quality=spec.gantry.rail_quality,
        preload_fraction_override=spec.model_overrides.rail_preload.preload_fraction,
        stiffness_mult_override=rail_qual["stiffness_mult"],
        drag_mult_override=rail_qual["drag_mult"],
    )
    block_mass_g = float(car["block_mass_g"])
    if spec.architecture == "delta":
        return block_mass_g, 0.0
    n_blocks = 2 if spec.gantry.effective_dual_x_rails else 1
    total_g = n_blocks * block_mass_g
    return total_g, total_g - _BASELINE_X_CARRIAGE_BLOCK_MASS_G


def y_carriage_block_mass_delta_g(spec: PrinterSpec) -> tuple[float, float]:
    """Return ``(total_y_block_mass_g, delta_vs_mgn12h_pair_g)`` from official HIWIN/Misumi carriage weights."""
    from printsim.fea.components import hertzian_rail_carriage_stiffness

    rail_qual = resolve_rail_quality(spec)
    car = hertzian_rail_carriage_stiffness(
        spec.gantry.y_rail_profile,
        preload=spec.gantry.rail_preload,
        block_type=spec.gantry.rail_block_type,
        rail_quality=spec.gantry.rail_quality,
        preload_fraction_override=spec.model_overrides.rail_preload.preload_fraction,
        stiffness_mult_override=rail_qual["stiffness_mult"],
        drag_mult_override=rail_qual["drag_mult"],
    )
    block_mass_g = float(car["block_mass_g"])
    if spec.architecture == "delta":
        return block_mass_g, 0.0
    total_g = 2.0 * block_mass_g
    return total_g, total_g - _BASELINE_Y_CARRIAGE_PAIR_MASS_G


def _apply_bedslinger_x_rail_backer_sandwich(model: AssemblyModel) -> None:
    """Upgrade Cartesian bedslinger X-gantry opposite-face backer to exact sandwich neutral-axis composite."""
    spec = model.spec
    if (
        spec.architecture != "cartesian_bedslinger"
        or spec.gantry.rail_backer_material == "none"
        or spec.gantry.effective_dual_x_rails
    ):
        return

    nmap = model.node_map
    member_by_id = {m.id: m for m in model.members}
    if "x_gantry_l" not in member_by_id or "x_gantry_r" not in member_by_id:
        return

    xp = spec.gantry.x_beam_profile
    xm = spec.gantry.x_beam_material
    base_x_prof = get_profile(xp)
    base_x_mat = get_material(xm)
    chamber_c = float(spec.frame.chamber_temp_c)
    if base_x_mat.tg_c is not None and abs(chamber_c - 25.0) > 1e-6:
        base_x_mat = base_x_mat.model_copy(
            update={
                "youngs_modulus_pa": base_x_mat.effective_modulus_pa(chamber_c),
                "shear_modulus_pa": base_x_mat.effective_shear_modulus_at_temp_pa(chamber_c),
            }
        )

    rail_mat = get_material(spec.gantry.rail_material)
    x_rail_prof = get_profile(spec.gantry.x_rail_profile)
    backer_mat = get_material(spec.gantry.rail_backer_material)
    backer_prof = _make_x_backer_profile(
        base_x_prof, x_rail_prof, spec.gantry.rail_backer_thickness_mm
    )
    x_normals = _x_rail_mount_normals(spec.gantry.effective_x_rail_mount)
    bolt_pitch_m = spec.gantry.rail_bolt_pitch_mm * 1e-3
    w = (
        float(np.linalg.norm(nmap["z_r"].position_m - nmap["z_l"].position_m))
        if ("z_l" in nmap and "z_r" in nmap)
        else spec.frame.width_mm * 1e-3
    )

    x_boost_iy = 0.0
    x_boost_iz = 0.0
    cf_mount = spec.gantry.effective_cf_rail_mounting
    for mid in ("x_gantry_l", "x_gantry_r"):
        m_xb = member_by_id[mid]
        m_xb.profile = base_x_prof
        m_xb.material = base_x_mat
        m_xb.rail_mass_kg = 0.0
        m_xb.backer_mass_kg = 0.0
        m_xb.backer_material_id = ""
        m_xb.insert_mass_kg = 0.0
        d_iy, d_iz, _ = _apply_bolted_rail_with_opposite_backer(
            m_xb,
            nmap,
            x_rail_prof,
            rail_mat,
            backer_prof,
            backer_mat,
            x_normals[0],
            beam_shape=base_x_prof.shape,
            bolt_pitch_m=bolt_pitch_m,
            l_eff_m=w,
            chamber_temp_c=chamber_c,
            cf_rail_mounting=cf_mount,
        )
        x_boost_iy += d_iy
        x_boost_iz += d_iz
    model.x_beam_insert_mass_kg = _apply_cf_two_wall_clamped_inserts(
        member_by_id, nmap, spec, w
    )
    x_boost_j = 0.5 * sum(
        max(member_by_id[mid].profile.j_m4 - base_x_prof.j_m4, 0.0)
        for mid in ("x_gantry_l", "x_gantry_r")
    )

    model.x_rail_ei_y_boost_pct = 100.0 * (0.5 * x_boost_iz) / max(base_x_prof.iz_m4, 1e-18)
    model.x_rail_ei_z_boost_pct = 100.0 * (0.5 * x_boost_iy) / max(base_x_prof.iy_m4, 1e-18)
    model.x_rail_gj_boost_pct = 100.0 * x_boost_j / max(base_x_prof.j_m4, 1e-18)


def _populate_linear_rail_telemetry(model: AssemblyModel) -> None:
    """Populate X/Y linear rail composite boost, Newmark gamma, and carriage block/stiffness telemetry."""
    from printsim.geometry import builder as _b

    spec = model.spec
    block_total_g, block_delta_g = x_carriage_block_mass_delta_g(spec)
    model.x_carriage_block_mass_g = block_total_g
    model.x_carriage_block_delta_g = block_delta_g

    props = _b.compute_toolhead_link_props(spec, l_mount_m=0.02, l_stem_m=0.04)
    model.x_carriage_k_rad_n_per_um = float(props["k_rad_car"]) * 1e-6
    model.x_carriage_k_rx_nm_rad = float(props["k_rx_car"])
    model.x_carriage_k_ry_nm_rad = float(props["k_ry_car"])

    if spec.architecture in ("corexy", "cartesian_bedslinger"):
        is_unsupported_rail = spec.gantry.x_beam_profile == "unsupported_rail"
        x_rail_prof = get_profile(spec.gantry.x_rail_profile)
        rail_mat = get_material(spec.gantry.rail_material)
        base_x_prof = (
            _oriented_bare_rail_profile(x_rail_prof, spec.gantry.effective_x_rail_mount)
            if is_unsupported_rail
            else get_profile(spec.gantry.x_beam_profile)
        )
        x_gantry_members = [m for m in model.members if m.id in ("x_gantry_l", "x_gantry_r")]
        if x_gantry_members:
            mean_iz = sum(m.profile.iz_m4 for m in x_gantry_members) / len(x_gantry_members)
            mean_iy = sum(m.profile.iy_m4 for m in x_gantry_members) / len(x_gantry_members)
            mean_j = sum(m.profile.j_m4 for m in x_gantry_members) / len(x_gantry_members)
            model.x_rail_ei_y_boost_pct = (
                100.0 * (mean_iz - base_x_prof.iz_m4) / max(base_x_prof.iz_m4, 1e-18)
            )
            model.x_rail_ei_z_boost_pct = (
                100.0 * (mean_iy - base_x_prof.iy_m4) / max(base_x_prof.iy_m4, 1e-18)
            )
            model.x_rail_gj_boost_pct = (
                100.0 * (mean_j - base_x_prof.j_m4) / max(base_x_prof.j_m4, 1e-18)
            )

        x_beam_mat = rail_mat if is_unsupported_rail else get_material(spec.gantry.x_beam_material)
        chamber_c = float(spec.frame.chamber_temp_c)
        eb_eff = x_beam_mat.effective_modulus_pa(chamber_c)
        shear_ret = x_beam_mat.shear_modulus_retention_at_temp(chamber_c)
        if is_unsupported_rail:
            sub_kind = "solid_bar"
        elif base_x_prof.id.startswith("cf_") or x_beam_mat.id.startswith("carbon_fiber"):
            sub_kind = "carbon_tube"
        elif base_x_prof.shape == "solid_bar":
            sub_kind = "solid_bar"
        else:
            sub_kind = "extrusion"
        from printsim.fea.components import (
            hertzian_rail_carriage_stiffness,
            newmark_rail_composite_gamma,
        )

        x_span_m = (
            sum(m.length_m(model.node_map) for m in x_gantry_members)
            if x_gantry_members
            else spec.frame.width_mm * 1e-3
        )
        if is_unsupported_rail:
            if spec.gantry.rail_backer_material == "none":
                model.x_rail_newmark_gamma = 1.0
            else:
                backer_mat_tel = get_material(spec.gantry.rail_backer_material)
                backer_prof_tel = _make_x_backer_profile(
                    x_rail_prof, x_rail_prof, spec.gantry.rail_backer_thickness_mm
                )
                model.x_rail_newmark_gamma = newmark_rail_composite_gamma(
                    e_rail_pa=backer_mat_tel.youngs_modulus_pa,
                    a_rail_m2=backer_prof_tel.area_m2,
                    e_beam_pa=rail_mat.youngs_modulus_pa,
                    a_beam_m2=x_rail_prof.area_m2,
                    bolt_pitch_m=spec.gantry.rail_bolt_pitch_mm * 1e-3,
                    l_eff_m=max(x_span_m, 0.15),
                    substrate_kind="solid_bar",
                )
        else:
            model.x_rail_newmark_gamma = newmark_rail_composite_gamma(
                e_rail_pa=rail_mat.youngs_modulus_pa,
                a_rail_m2=x_rail_prof.area_m2,
                e_beam_pa=eb_eff,
                a_beam_m2=base_x_prof.area_m2,
                bolt_pitch_m=spec.gantry.rail_bolt_pitch_mm * 1e-3,
                l_eff_m=max(x_span_m, 0.15),
                substrate_kind=sub_kind,
                wall_thickness_m=base_x_prof.wall_thickness_m,
                matrix_shear_retention=shear_ret,
                cf_rail_mounting=spec.gantry.effective_cf_rail_mounting,
            )
        model.x_beam_insert_mass_kg = sum(
            m.insert_mass_kg for m in model.members if m.role == "gantry"
        )

        rail_qual = resolve_rail_quality(spec)
        y_car = hertzian_rail_carriage_stiffness(
            spec.gantry.y_rail_profile,
            preload=spec.gantry.rail_preload,
            block_type=spec.gantry.rail_block_type,
            rail_quality=spec.gantry.rail_quality,
            preload_fraction_override=spec.model_overrides.rail_preload.preload_fraction,
            stiffness_mult_override=rail_qual["stiffness_mult"],
            drag_mult_override=rail_qual["drag_mult"],
        )
        _, y_block_delta_g = y_carriage_block_mass_delta_g(spec)
        model.y_carriage_block_mass_g = float(y_car["block_mass_g"])
        model.y_carriage_block_delta_g = y_block_delta_g
        model.y_carriage_k_rx_nm_rad = float(y_car["k_rx_nm_rad"])
    else:
        model.x_rail_gj_boost_pct = 0.0
        model.x_rail_newmark_gamma = 0.0
        model.x_beam_insert_mass_kg = 0.0
        model.y_carriage_block_mass_g = 0.0
        model.y_carriage_block_delta_g = 0.0
        model.y_carriage_k_rx_nm_rad = 0.0


def _apply_corexy_gantry_rails(
    spec: PrinterSpec,
    members: list[StructuralMember],
    member_by_id: dict[str, StructuralMember],
    nmap: dict[str, StructuralNode],
    w: float,
    d: float,
    fp: str,
    xp: str,
    xm: str,
) -> tuple[ProfileSpec, MaterialSpec, dict[str, float], float, float, float, float, float]:
    """Apply bolted steel linear rails, viscoelastic laminate softening, and CF inserts on CoreXY X/Y beams."""
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

    from printsim.fea.components import hertzian_rail_carriage_stiffness

    rail_qual_y = resolve_rail_quality(spec)
    y_car = hertzian_rail_carriage_stiffness(
        spec.gantry.y_rail_profile,
        preload=spec.gantry.rail_preload,
        block_type=spec.gantry.rail_block_type,
        rail_quality=spec.gantry.rail_quality,
        preload_fraction_override=spec.model_overrides.rail_preload.preload_fraction,
        stiffness_mult_override=rail_qual_y["stiffness_mult"],
        drag_mult_override=rail_qual_y["drag_mult"],
    )
    k_rx_y_car = max(float(y_car["k_rx_nm_rad"]), 1.0)

    chamber_c = float(spec.frame.chamber_temp_c)
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
                chamber_temp_c=chamber_c,
            )
            y_boost_iz = d_iz_y
            # Wire Y-rail carriage Hertzian roll compliance in series with Y-rail beam torsion
            g_y_eff = max(m.material.effective_shear_modulus_pa, 1e6)
            k_tor_y_beam = (4.0 * g_y_eff * m.profile.j_m4) / max(d, 0.15)
            eta_j_y = k_rx_y_car / max(k_rx_y_car + k_tor_y_beam, 1.0)
            m.profile = m.profile.model_copy(update={"j_m4": m.profile.j_m4 * eta_j_y})

    # Chamber-temperature viscoelastic laminate softening on composite X-gantry beam
    is_unsupported_rail = xp == "unsupported_rail"
    x_beam_span = corexy_x_beam_span_m(spec)
    x_rail_len = corexy_x_rail_len_m(spec)
    beam_len_scale = x_beam_span / max(w, 1e-9)
    rail_len_scale = x_rail_len / max(w, 1e-9)

    if is_unsupported_rail:
        # Unsupported bare linear rail X-gantry (e.g. Monolith Zero): rail IS the X-gantry beam.
        # Construct x_gantry_l/r directly from oriented x_rail_profile & rail_material with zero
        # phantom beam double-counting, and bolt any opposite-face backer directly onto the steel rail.
        bare_rail_prof = _oriented_bare_rail_profile(x_rail_prof, x_mount)
        base_x_prof = bare_rail_prof
        for mid in ("x_gantry_l", "x_gantry_r"):
            m_xb = member_by_id[mid]
            m_xb.profile = bare_rail_prof
            m_xb.material = rail_mat.model_copy(
                update={"density_kg_m3": rail_mat.density_kg_m3 * beam_len_scale}
            )
            m_xb.rail_mass_kg = m_xb.mass_kg(nmap)
            m_xb.backer_mass_kg = 0.0
            m_xb.backer_material_id = ""
            m_xb.insert_mass_kg = 0.0

        x_boost_iy = 0.0
        x_boost_iz = 0.0
        if spec.gantry.rail_backer_material != "none":
            backer_mat = get_material(spec.gantry.rail_backer_material)
            backer_prof = _make_x_backer_profile(
                x_rail_prof, x_rail_prof, spec.gantry.rail_backer_thickness_mm
            )
            opposite_normal = [-x_normals[0]]
            for mid in ("x_gantry_l", "x_gantry_r"):
                d_iy_x, d_iz_x, _ = _apply_bolted_linear_rail(
                    member_by_id[mid],
                    nmap,
                    backer_prof,
                    backer_mat,
                    opposite_normal,
                    beam_shape="solid_bar",
                    bolt_pitch_m=bolt_pitch_m,
                    l_eff_m=x_beam_span,
                    track_as_rail=False,
                    chamber_temp_c=chamber_c,
                    mass_length_scale=beam_len_scale,
                )
                x_boost_iy += d_iy_x
                x_boost_iz += d_iz_x
        x_beam_insert_mass_kg = 0.0
    else:
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

        # Physical X-beam cut length (between XY-joint faces) and bolted X-rail cut length
        # (build_X + 50 mm) are shorter than the outer frame width w; scale bare beam and
        # folded rail/backer densities across geometric [0, w] so FEA M, y_moving_mass, and BOM match.
        for mid in ("x_gantry_l", "x_gantry_r"):
            m_xb = member_by_id[mid]
            m_xb.material = m_xb.material.model_copy(
                update={"density_kg_m3": m_xb.material.density_kg_m3 * beam_len_scale}
            )

        x_boost_iy = 0.0
        x_boost_iz = 0.0
        cf_mount = spec.gantry.effective_cf_rail_mounting
        if spec.gantry.rail_backer_material != "none" and not spec.gantry.effective_dual_x_rails:
            backer_mat = get_material(spec.gantry.rail_backer_material)
            backer_prof = _make_x_backer_profile(
                base_x_prof, x_rail_prof, spec.gantry.rail_backer_thickness_mm
            )
            for mid in ("x_gantry_l", "x_gantry_r"):
                d_iy_x, d_iz_x, _ = _apply_bolted_rail_with_opposite_backer(
                    member_by_id[mid],
                    nmap,
                    x_rail_prof,
                    rail_mat,
                    backer_prof,
                    backer_mat,
                    x_normals[0],
                    beam_shape=base_x_prof.shape,
                    bolt_pitch_m=bolt_pitch_m,
                    l_eff_m=w,
                    chamber_temp_c=chamber_c,
                    cf_rail_mounting=cf_mount,
                    mass_length_scale=rail_len_scale,
                )
                x_boost_iy += d_iy_x
                x_boost_iz += d_iz_x
        else:
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
                    chamber_temp_c=chamber_c,
                    cf_rail_mounting=cf_mount,
                    mass_length_scale=rail_len_scale,
                )
                x_boost_iy += d_iy_x
                x_boost_iz += d_iz_x
        x_beam_insert_mass_kg = _apply_cf_two_wall_clamped_inserts(
            member_by_id, nmap, spec, x_beam_span
        )
    x_boost_iy *= 0.5
    x_boost_iz *= 0.5
    x_boost_j = 0.5 * sum(
        max(member_by_id[mid].profile.j_m4 - base_x_prof.j_m4, 0.0)
        for mid in ("x_gantry_l", "x_gantry_r")
    )

    x_rail_ei_y_boost_pct = 100.0 * x_boost_iz / max(base_x_prof.iz_m4, 1e-18)
    x_rail_ei_z_boost_pct = 100.0 * x_boost_iy / max(base_x_prof.iy_m4, 1e-18)
    x_rail_gj_boost_pct = 100.0 * x_boost_j / max(base_x_prof.j_m4, 1e-18)
    y_rail_ei_x_boost_pct = 100.0 * y_boost_iz / max(base_y_prof.iz_m4, 1e-18)


    return (
        x_rail_prof,
        rail_mat,
        y_car,
        x_beam_insert_mass_kg,
        x_rail_ei_y_boost_pct,
        x_rail_ei_z_boost_pct,
        x_rail_gj_boost_pct,
        y_rail_ei_x_boost_pct,
    )
