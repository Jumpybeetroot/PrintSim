"""Linear rail Newmark composite mechanics, CF inserts, and gantry carriage helpers extracted from ``builder.py``."""

from __future__ import annotations

import math
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
    bolt_pitch_m = spec.gantry.effective_x_rail_bolt_pitch_mm * 1e-3
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

        if spec.architecture == "corexy":
            from printsim.geometry.builder import corexy_frame_centerlines_m

            x_span_m = float(corexy_frame_centerlines_m(spec)[0])
        else:
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
                    bolt_pitch_m=spec.gantry.effective_x_rail_bolt_pitch_mm * 1e-3,
                    l_eff_m=max(x_span_m, 0.15),
                    substrate_kind="solid_bar",
                )
        else:
            model.x_rail_newmark_gamma = newmark_rail_composite_gamma(
                e_rail_pa=rail_mat.youngs_modulus_pa,
                a_rail_m2=x_rail_prof.area_m2,
                e_beam_pa=eb_eff,
                a_beam_m2=base_x_prof.area_m2,
                bolt_pitch_m=spec.gantry.effective_x_rail_bolt_pitch_mm * 1e-3,
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
    bolt_pitch_m = spec.gantry.effective_x_rail_bolt_pitch_mm * 1e-3
    y_bolt_pitch_m = spec.gantry.effective_y_rail_bolt_pitch_mm * 1e-3
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
                bolt_pitch_m=y_bolt_pitch_m,
                l_eff_m=d,
                chamber_temp_c=chamber_c,
            )
            y_boost_iz = d_iz_y
            # Note: Y-rail carriage Hertzian roll/pitch/yaw/radial compliance (k_rx_y_car) is
            # modeled explicitly by the 6-DOF y_car_bearing_{l,r} superelements between
            # y_rail_car_{l,r} and y_car_{l,r} (zero double-counting on Y-rail beam torsion).

    # Chamber-temperature viscoelastic laminate softening on composite X-gantry beam
    is_unsupported_rail = xp == "unsupported_rail"
    x_beam_span = corexy_x_beam_span_m(spec)
    x_rail_len = corexy_x_rail_len_m(spec)
    x_geom_span = sum(
        member_by_id[mid].length_m(nmap)
        for mid in ("x_gantry_l", "x_gantry_r")
        if mid in member_by_id
    )
    if x_geom_span <= 1e-6:
        x_geom_span = w
    beam_len_scale = x_beam_span / max(x_geom_span, 1e-9)
    rail_len_scale = x_rail_len / max(x_geom_span, 1e-9)

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
        # (build_X + 50 mm) may differ slightly from the FEA beam span x_geom_span; scale bare
        # beam and folded rail/backer densities across x_geom_span so FEA M, y_moving_mass, and BOM match.
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


def _skew_3x3_local(vec: np.ndarray) -> np.ndarray:
    vx, vy, vz = float(vec[0]), float(vec[1]), float(vec[2])
    return np.array(
        [
            [0.0, -vz, vy],
            [vz, 0.0, -vx],
            [-vy, vx, 0.0],
        ],
        dtype=float,
    )


def _resolve_cad_xy_joint_assets(spec: PrinterSpec) -> tuple[object, object] | None:
    """Load baked left & right 3D CAD XY-joint superelements for CoreXY CAD hardware kits."""
    if spec.architecture != "corexy" or not bool(getattr(spec, "use_cad_superelements", True)):
        return None
    kit_id = spec.resolve_effective_gantry_hardware_kit()
    if kit_id == "procedural":
        return None

    from printsim.schema.printer import GANTRY_HARDWARE_KIT_CATALOG

    stations = dict((GANTRY_HARDWARE_KIT_CATALOG.get(kit_id) or {}).get("stations") or {})
    aid_l = stations.get("xy_left")
    aid_r = stations.get("xy_right")
    if not aid_l or not aid_r:
        return None

    try:
        from printsim.cad_assets.registry import has_hardware_asset, load_superelement
    except ImportError:
        return None

    if not (
        has_hardware_asset("gantry_brackets", aid_l)
        and has_hardware_asset("gantry_brackets", aid_r)
    ):
        return None

    chamber_c = float(spec.frame.chamber_temp_c)
    bkt_mat_id = spec.gantry.bracket_material
    se_l = load_superelement(
        "gantry_brackets",
        aid_l,
        material_id=bkt_mat_id,
        chamber_temp_c=chamber_c,
    )
    se_r = load_superelement(
        "gantry_brackets",
        aid_r,
        material_id=bkt_mat_id,
        chamber_temp_c=chamber_c,
    )
    if se_l is None or se_r is None:
        return None
    return se_l, se_r


def _invert_sym_6x6(mat: np.ndarray) -> np.ndarray:
    """Invert a symmetric positive-definite 6x6 matrix with diagonal preconditioning."""
    m_sym = 0.5 * (mat + mat.T)
    diag = np.maximum(np.diag(m_sym), 1e-15)
    s_inv = 1.0 / np.sqrt(diag)
    m_sc = (s_inv[:, None] * m_sym) * s_inv[None, :]
    try:
        inv_sc = np.linalg.inv(m_sc)
    except np.linalg.LinAlgError:
        inv_sc = np.linalg.pinv(m_sc, rcond=1e-10)
    out = (s_inv[:, None] * inv_sc) * s_inv[None, :]
    return 0.5 * (out + out.T)


def _assemble_xy_joint_with_bolted_seams(
    c_body_6x6: np.ndarray,
    dp_asm: np.ndarray,
    spec: PrinterSpec,
) -> tuple[np.ndarray, dict[str, float]]:
    """Combine XY-joint body compliance in series with explicit Port-0 (4x M3 Y-carriage)
    and Port-1 (X-beam clamp) bolted-seam compliances, returning ``(K_12x12, diag_stiffness_info)``.
    """
    from printsim.fea.components import xy_joint_bolted_seam_compliances

    c_pad0, c_pad1 = xy_joint_bolted_seam_compliances(
        rail_profile_id=spec.gantry.y_rail_profile,
        block_type=spec.gantry.rail_block_type,
        y_mount=spec.gantry.y_rail_mount,
        bracket_material_id=spec.gantry.bracket_material,
        x_beam_profile_id=spec.gantry.x_beam_profile,
        x_beam_material_id=spec.gantry.x_beam_material,
        chamber_temp_c=float(spec.frame.chamber_temp_c),
    )
    t_01 = np.eye(6, dtype=float)
    t_01[:3, 3:6] = -_skew_3x3_local(dp_asm)
    c_total = t_01 @ c_pad0 @ t_01.T + 0.5 * (c_body_6x6 + c_body_6x6.T) + c_pad1
    c_total = 0.5 * (c_total + c_total.T)
    k_11_eff = _invert_sym_6x6(c_total)
    b_mat = np.hstack([-t_01, np.eye(6, dtype=float)])
    k_asm = b_mat.T @ k_11_eff @ b_mat
    k_asm = 0.5 * (k_asm + k_asm.T)
    info = {
        "kx_n_per_mm": round(1e-3 / max(float(c_total[0, 0]), 1e-15), 2),
        "ky_n_per_mm": round(1e-3 / max(float(c_total[1, 1]), 1e-15), 2),
        "kz_n_per_mm": round(1e-3 / max(float(c_total[2, 2]), 1e-15), 2),
    }
    return k_asm, info


def _build_cad_xy_joint_superelement(
    se_id: str,
    asset: object,
    port_node_ids: tuple[str, str],
    p0_asm_m: np.ndarray,
    p1_asm_m: np.ndarray,
    spec: PrinterSpec,
):
    """Build a 12x12 rigid-arm-transformed AttachedSuperelement for a CoreXY CAD XY-joint.

    Preserves the full coupled 6x6 Tet4 Craig-Bampton compliance ``C_11 = inv(K_11)`` (including
    real 3D off-plane twist-to-sag and lateral-to-roll cross-coupling) and places it in series
    with explicit Huth/VDI-2230 bolted-seam compliances at Port 0 (4x M3 Y-carriage pad) and
    Port 1 (X-beam clamp), preserving machine-precision 6-DOF rigid-body zero modes.
    """
    from printsim.geometry import builder as _b

    K_bb = np.asarray(asset.K_bb, dtype=float)  # type: ignore[attr-defined]
    dp_asm = np.asarray(p1_asm_m, dtype=float) - np.asarray(p0_asm_m, dtype=float)

    # Full coupled SPD 6x6 CAD body compliance at Port 1 relative to Port 0 (zero off-plane dropping).
    # Bolted-seam compliance is modeled explicitly via C_pad0 + C_pad1 (zero double-counting).
    c_body = _invert_sym_6x6(K_bb[6:12, 6:12])
    K_asm, eff_diag = _assemble_xy_joint_with_bolted_seams(c_body, dp_asm, spec)

    mass_scale = float(getattr(asset, "mass_scale_applied", 1.0))
    stiff_scale = float(getattr(asset, "stiffness_scale_applied", 1.0))
    mass_ref = float(asset.manifest.get("mass_kg", 0.05)) * mass_scale  # type: ignore[attr-defined]

    manifest_out = dict(asset.manifest)  # type: ignore[attr-defined]
    if not bool(manifest_out.get("is_multi_material", False)):
        if isinstance(manifest_out.get("port_stiffness_n_per_mm"), dict) and abs(stiff_scale - 1.0) > 1e-6:
            manifest_out["port_stiffness_n_per_mm"] = {
                k: round(float(v) * stiff_scale, 3)
                for k, v in manifest_out["port_stiffness_n_per_mm"].items()
            }
        if abs(mass_scale - 1.0) > 1e-6:
            manifest_out["mass_kg"] = round(mass_ref, 6)
    manifest_out["effective_bolted_stiffness_n_per_mm"] = eff_diag

    aid = str(asset.asset_id)  # type: ignore[attr-defined]
    return _b.AttachedSuperelement(
        id=se_id,
        category="gantry_brackets",
        asset_id=aid,
        port_node_ids=port_node_ids,
        K_cb=K_asm,
        M_cb=None,  # Mass is already accounted for on y_car_l / y_car_r (zero double-counting)
        port_coords_cad_m=None,  # Already congruence-transformed to exact (p0_asm_m, p1_asm_m)
        n_internal_modes=0,
        part_group_id="xy_joints",
        part_group_label="XY-Joint Brackets (CAD)",
        role="gantry",
        mass_kg=mass_ref,
        fixed_interface_freqs_hz=list(asset.fixed_interface_freqs_hz),  # type: ignore[attr-defined]
        manifest=manifest_out,
        visual_glb_url=f"/api/assets/hardware/gantry_brackets/{aid}/visual.glb",
        cad_trust_pct=int(asset.manifest.get("cad_trust_pct", 88)),  # type: ignore[attr-defined]
    )


def _build_procedural_xy_joint_superelement(
    se_id: str,
    port_node_ids: tuple[str, str],
    p0_asm_m: np.ndarray,
    p1_asm_m: np.ndarray,
    spec: PrinterSpec,
):
    """Build an analytical 12x12 Timoshenko + bolted-seam XY-joint superelement for procedural CoreXY builds.

    Ensures non-CAD / procedural CoreXY printers capture real XY-joint bracket flex (scaled by
    ``bracket_material`` E(T) and G(T)) plus the exact Port-0 4x M3 Y-carriage bolted pad and
    Port-1 X-beam clamp seam compliances.
    """
    from printsim.geometry import builder as _b

    dp_asm = np.asarray(p1_asm_m, dtype=float) - np.asarray(p0_asm_m, dtype=float)
    span_m = max(float(np.linalg.norm(dp_asm)), 0.015)
    sign_x = 1.0 if float(dp_asm[0]) >= 0.0 else -1.0

    bkt_mat = get_material(spec.gantry.bracket_material)
    chamber_c = float(spec.frame.chamber_temp_c)
    e_pa = max(bkt_mat.effective_modulus_pa(chamber_c), 1.0e8)
    g_pa = max(bkt_mat.effective_shear_modulus_at_temp_pa(chamber_c), 3.0e7)

    # Effective pocketed CoreXY dual-idler XY-joint cross-section (~38 mm Y width x 28 mm Z height,
    # ~42% net structural wall fill after idler pockets & belt channels)
    w_y = 0.038
    h_z = 0.028
    fill = 0.42
    area = fill * w_y * h_z
    iy = fill * (w_y * (h_z**3)) / 12.0  # bending about Y (Z deflection)
    iz = fill * (h_z * (w_y**3)) / 12.0  # bending about Z (Y deflection)
    j_tor = fill * 0.22 * min(w_y, h_z) ** 3 * max(w_y, h_z)
    kappa = 0.72

    # Exact Timoshenko cantilever 6x6 compliance at Port 1 (X-beam end) with Port 0 fixed
    c_body = np.zeros((6, 6), dtype=float)
    c_body[0, 0] = span_m / (e_pa * area)
    c_body[3, 3] = span_m / (g_pa * j_tor)
    # In-plane Y-bending + shear + yaw coupling ([1, 5]): sign_x handles left (+X) vs right (-X)
    c_body[1, 1] = (span_m**3) / (3.0 * e_pa * iz) + span_m / (kappa * g_pa * area)
    c_body[5, 5] = span_m / (e_pa * iz)
    c_body[1, 5] = c_body[5, 1] = sign_x * (span_m**2) / (2.0 * e_pa * iz)
    # Vertical Z-bending + shear + pitch coupling ([2, 4])
    c_body[2, 2] = (span_m**3) / (3.0 * e_pa * iy) + span_m / (kappa * g_pa * area)
    c_body[4, 4] = span_m / (e_pa * iy)
    c_body[2, 4] = c_body[4, 2] = -sign_x * (span_m**2) / (2.0 * e_pa * iy)

    K_asm, eff_diag = _assemble_xy_joint_with_bolted_seams(c_body, dp_asm, spec)
    mass_ref = max(area * span_m * bkt_mat.density_kg_m3, 0.015)

    return _b.AttachedSuperelement(
        id=se_id,
        category="gantry_brackets",
        asset_id="procedural_xy_joint",
        port_node_ids=port_node_ids,
        K_cb=K_asm,
        M_cb=None,
        port_coords_cad_m=None,
        n_internal_modes=0,
        part_group_id="xy_joints",
        part_group_label="XY-Joint Brackets (Analytical)",
        role="gantry",
        mass_kg=mass_ref,
        fixed_interface_freqs_hz=[],
        manifest={
            "name": f"Analytical XY-Joint ({bkt_mat.name})",
            "element_type": "Timoshenko+Huth",
            "num_elements": 1,
            "mass_kg": round(mass_ref, 6),
            "port_stiffness_n_per_mm": eff_diag,
            "effective_bolted_stiffness_n_per_mm": eff_diag,
        },
        visual_glb_url="",
        cad_trust_pct=72,
    )


def _build_y_carriage_bearing_superelement(
    se_id: str,
    port_node_ids: tuple[str, str],
    y_car: dict[str, float],
    spec: PrinterSpec,
    rail_and_car_pos_m: tuple[np.ndarray, np.ndarray] | None = None,
):
    """Build a 12x12 Hertzian linear-rail carriage contact superelement for ``y_rail_car_* -> y_car_*``.

    Models the 5 constrained linear-bearing DOFs (radial, lateral, roll about Y-rail, pitch, yaw)
    using exact Hertzian ball-raceway contact stiffnesses from ``hertzian_rail_carriage_stiffness``,
    plus the physical raceway-to-carriage-bolt-face lever-arm offset (``y_carriage_lever_arm_m``),
    while keeping the axial Y travel DOF structurally locked (``1e9 N/m``) so axial CoreXY belt
    compliance remains cleanly isolated in ``modal.py``'s 2-DOF belt-drive eigenproblem with zero
    double-counting.
    """
    from printsim.fea.components import y_carriage_lever_arm_m
    from printsim.geometry import builder as _b

    y_mount = spec.gantry.y_rail_mount
    side = "l" if se_id.endswith("_l") else "r"
    lever_arm = y_carriage_lever_arm_m(spec.gantry.y_rail_profile, y_mount, side)
    if y_mount == "inner" and rail_and_car_pos_m is not None:
        # Carriage node already sits on the raceway: rigid arm back to the Y-extrusion axis
        # (p_shell - p_frame = y_rail_car - y_car), Hertzian contact acting at the raceway.
        p_rail, p_car = rail_and_car_pos_m
        lever_arm = tuple(float(v) for v in (np.asarray(p_rail) - np.asarray(p_car)))
    k_yy_lock = 1.0e9
    k_ry_roll = max(float(y_car["k_rx_nm_rad"]), 1.0)
    if y_mount == "inner":
        k_xx = max(float(y_car["k_zz_n_m"]), 1.0)
        k_zz = max(float(y_car["k_yy_n_m"]), 1.0)
        k_rx = max(float(y_car["k_rz_nm_rad"]), 1.0)
        k_rz = max(float(y_car["k_ry_nm_rad"]), 1.0)
    else:
        k_xx = max(float(y_car["k_yy_n_m"]), 1.0)
        k_zz = max(float(y_car["k_zz_n_m"]), 1.0)
        k_rx = max(float(y_car["k_ry_nm_rad"]), 1.0)
        k_rz = max(float(y_car["k_rz_nm_rad"]), 1.0)

    return _b.FastenerSpring(
        id=se_id,
        bay="y_carriage",
        shell_node_id=port_node_ids[0],
        frame_node_id=port_node_ids[1],
        k_shear_n_m=0.5 * (k_xx + k_zz),
        k_normal_n_m=k_yy_lock,
        k_rot_nm_rad=k_ry_roll,
        normal_vec=(0.0, 1.0, 0.0),
        offset_m=lever_arm,
        k_diag_6dof=(k_xx, k_yy_lock, k_zz, k_rx, k_ry_roll, k_rz),
    )


def _congruence_shift_two_port_k(
    K_bb: np.ndarray,
    p0_cad_m: np.ndarray,
    p1_cad_m: np.ndarray,
    dp_asm_m: np.ndarray,
) -> np.ndarray:
    """Shift a 12x12 CAD superelement ``K_bb`` from ``(p0_cad_m, p1_cad_m)`` to ``(0, dp_asm_m)``.

    Uses exact rigid-arm congruence transforms ``J_0`` (port 0 -> origin) and ``J_1`` (port 1 ->
    ``dp_asm_m``), preserving all 6 rigid-body zero modes to machine precision.
    """
    J0 = np.eye(6, dtype=float)
    J0[:3, 3:6] = -_skew_3x3_local(np.asarray(p0_cad_m, dtype=float))
    r1_shift = np.asarray(p1_cad_m, dtype=float) - np.asarray(dp_asm_m, dtype=float)
    J1 = np.eye(6, dtype=float)
    J1[:3, 3:6] = -_skew_3x3_local(r1_shift)
    J = np.zeros((12, 12), dtype=float)
    J[:6, :6] = J0
    J[6:12, 6:12] = J1
    K_asm = J.T @ K_bb @ J
    return 0.5 * (K_asm + K_asm.T)


def _resolve_cad_corner_mount_assets(spec: PrinterSpec) -> dict[str, object]:
    """Load baked 3D CAD corner motor-mount & front-idler superelements for CoreXY CAD kits."""
    if spec.architecture != "corexy" or not bool(getattr(spec, "use_cad_superelements", True)):
        return {}
    kit_id = spec.resolve_effective_gantry_hardware_kit()
    if kit_id == "procedural":
        return {}

    from printsim.fea.components import compute_stepper_mount_temp_c
    from printsim.schema.printer import GANTRY_HARDWARE_KIT_CATALOG

    stations = dict((GANTRY_HARDWARE_KIT_CATALOG.get(kit_id) or {}).get("stations") or {})
    try:
        from printsim.cad_assets.registry import has_hardware_asset, load_superelement
    except ImportError:
        return {}

    chamber_c = float(spec.frame.chamber_temp_c)
    motor_temp_c, _ = compute_stepper_mount_temp_c(spec)
    bkt_mat_id = spec.gantry.bracket_material
    is_awd = spec.gantry.drive_mode == "awd"

    out: dict[str, object] = {}
    for tag, st_key in (
        ("rl", "rear_left"),
        ("rr", "rear_right"),
        ("fl", "front_left"),
        ("fr", "front_right"),
    ):
        aid = stations.get(st_key)
        if not aid or not has_hardware_asset("gantry_brackets", aid):
            continue
        eval_temp_c = motor_temp_c if (tag in ("rl", "rr") or is_awd) else chamber_c
        se = load_superelement(
            "gantry_brackets",
            aid,
            material_id=bkt_mat_id,
            chamber_temp_c=eval_temp_c,
        )
        if se is not None:
            out[tag] = se
    return out


def _build_cad_corner_mount_superelement(
    se_id: str,
    asset: object,
    port_node_ids: tuple[str, str],
    p0_asm_m: np.ndarray,
    p1_asm_m: np.ndarray,
    spec: PrinterSpec,
):
    """Build a 12x12 rigid-arm-transformed AttachedSuperelement for a CoreXY corner mount/idler."""
    from printsim.geometry import builder as _b

    kit_id = spec.resolve_effective_gantry_hardware_kit()
    bkt_mat = get_material(spec.gantry.bracket_material)
    bolt_factor = float(bkt_mat.bolted_joint_stiffness_factor)
    cad_is_double_shear = kit_id in (
        "monolith_ldo_cnc_awd",
        "monolith_ldo_cnc_2wd",
        "siboor_cnc_awd",
        "siboor_cnc_2wd",
        "vzbot_330_awd",
        "vzbot_330_2wd",
    )
    ds_mult = 1.45 if (spec.gantry.double_shear_pulleys and not cad_is_double_shear) else 1.0
    extra_mult = max(bolt_factor * ds_mult, 1e-4)

    K_bb = np.asarray(asset.K_bb, dtype=float) * extra_mult  # type: ignore[attr-defined]
    p0_cad_m = np.asarray(asset.port_coords_m[0], dtype=float)  # type: ignore[attr-defined]
    p1_cad_m = np.asarray(asset.port_coords_m[1], dtype=float)  # type: ignore[attr-defined]
    dp_asm_m = np.asarray(p1_asm_m, dtype=float) - np.asarray(p0_asm_m, dtype=float)
    K_asm = _congruence_shift_two_port_k(K_bb, p0_cad_m, p1_cad_m, dp_asm_m)

    mass_scale = float(getattr(asset, "mass_scale_applied", 1.0))
    stiff_scale = float(getattr(asset, "stiffness_scale_applied", 1.0)) * extra_mult
    mass_ref = float(asset.manifest.get("mass_kg", 0.08)) * mass_scale  # type: ignore[attr-defined]

    manifest_out = dict(asset.manifest)  # type: ignore[attr-defined]
    if not bool(manifest_out.get("is_multi_material", False)):
        if isinstance(manifest_out.get("port_stiffness_n_per_mm"), dict) and abs(stiff_scale - 1.0) > 1e-6:
            manifest_out["port_stiffness_n_per_mm"] = {
                k: round(float(v) * stiff_scale, 3)
                for k, v in manifest_out["port_stiffness_n_per_mm"].items()
            }
        if abs(mass_scale - 1.0) > 1e-6:
            manifest_out["mass_kg"] = round(mass_ref, 6)

    aid = str(asset.asset_id)  # type: ignore[attr-defined]
    return _b.AttachedSuperelement(
        id=se_id,
        category="gantry_brackets",
        asset_id=aid,
        port_node_ids=port_node_ids,
        K_cb=K_asm,
        M_cb=None,  # Motor/mount mass is placed on pulley_* (zero double-counting)
        port_coords_cad_m=None,
        n_internal_modes=0,
        part_group_id="gantry_brackets",
        part_group_label="Gantry Motor & Idler Mounts (CAD)",
        role="gantry",
        mass_kg=mass_ref,
        fixed_interface_freqs_hz=list(asset.fixed_interface_freqs_hz),  # type: ignore[attr-defined]
        manifest=manifest_out,
        visual_glb_url=f"/api/assets/hardware/gantry_brackets/{aid}/visual.glb",
        cad_trust_pct=int(asset.manifest.get("cad_trust_pct", 88)),  # type: ignore[attr-defined]
    )


def _resolve_cad_z_joint_asset(spec: PrinterSpec) -> object | None:
    """Load baked 3D CAD flying-gantry Z-joint superelement for Voron 2.4 CAD kits."""
    if (
        spec.architecture != "corexy"
        or not spec.is_flying_gantry
        or not bool(getattr(spec, "use_cad_superelements", True))
    ):
        return None
    kit_id = spec.resolve_effective_gantry_hardware_kit()
    if kit_id == "procedural":
        return None

    from printsim.schema.printer import GANTRY_HARDWARE_KIT_CATALOG

    stations = dict((GANTRY_HARDWARE_KIT_CATALOG.get(kit_id) or {}).get("stations") or {})
    aid = stations.get("z_joint")
    if not aid:
        return None
    try:
        from printsim.cad_assets.registry import has_hardware_asset, load_superelement
    except ImportError:
        return None
    if not has_hardware_asset("gantry_brackets", aid):
        return None

    zj_mat = spec.gantry.z_joint_material
    mat_id = "abs_printed" if zj_mat == "printed" else "aluminum_6061_t6"
    return load_superelement(
        "gantry_brackets",
        aid,
        material_id=mat_id,
        chamber_temp_c=float(spec.frame.chamber_temp_c),
    )


def _build_cad_z_joint_superelement(
    se_id: str,
    asset: object,
    port_node_ids: tuple[str, str],
    rot_z_rad: float,
    spec: PrinterSpec,
):
    """Build a 12x12 zero-span CAD Z-joint bracket superelement in series with ``z_joint_*``."""
    from printsim.geometry import builder as _b

    zj_mat = spec.gantry.z_joint_material
    mat_id = "abs_printed" if zj_mat == "printed" else "aluminum_6061_t6"
    mat_obj = get_material(mat_id)
    bolt_factor = float(mat_obj.bolted_joint_stiffness_factor)
    type_mult = 1.65 if spec.gantry.z_joint_type == "double" else 1.0
    hybrid_mult = 0.55 if zj_mat == "hybrid" else 1.0
    extra_mult = max(bolt_factor * type_mult * hybrid_mult, 1e-4)

    K_bb = np.asarray(asset.K_bb, dtype=float) * extra_mult  # type: ignore[attr-defined]
    K_11 = 0.5 * (K_bb[6:12, 6:12] + K_bb[6:12, 6:12].T)

    c_r, s_r = float(np.cos(rot_z_rad)), float(np.sin(rot_z_rad))
    R3 = np.array([[c_r, -s_r, 0.0], [s_r, c_r, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    R6 = np.zeros((6, 6), dtype=float)
    R6[:3, :3] = R3
    R6[3:6, 3:6] = R3
    K_11_rot = R6 @ K_11 @ R6.T

    B = np.hstack([-np.eye(6, dtype=float), np.eye(6, dtype=float)])
    K_asm = B.T @ K_11_rot @ B
    K_asm = 0.5 * (K_asm + K_asm.T)

    mass_scale = float(getattr(asset, "mass_scale_applied", 1.0))
    mass_ref = float(asset.manifest.get("mass_kg", 0.025)) * mass_scale  # type: ignore[attr-defined]
    aid = str(asset.asset_id)  # type: ignore[attr-defined]
    return _b.AttachedSuperelement(
        id=se_id,
        category="gantry_brackets",
        asset_id=aid,
        port_node_ids=port_node_ids,
        K_cb=K_asm,
        M_cb=None,
        port_coords_cad_m=None,
        n_internal_modes=0,
        part_group_id="z_gantry_joints",
        part_group_label="Flying-Gantry Z-Joint Brackets (CAD)",
        role="gantry",
        mass_kg=mass_ref,
        fixed_interface_freqs_hz=list(asset.fixed_interface_freqs_hz),  # type: ignore[attr-defined]
        manifest=dict(asset.manifest),  # type: ignore[attr-defined]
        visual_glb_url=f"/api/assets/hardware/gantry_brackets/{aid}/visual.glb",
        cad_trust_pct=int(asset.manifest.get("cad_trust_pct", 88)),  # type: ignore[attr-defined]
    )


def _resolve_cad_trident_bed_arm_assets(spec: PrinterSpec) -> dict[str, object]:
    """Load baked 3D CAD Trident Z bed-arm superelements (FL/FR/Rear) for triple-Z CoreXY."""
    if (
        spec.architecture != "corexy"
        or spec.bed.z_drive != "leadscrew_triple"
        or not bool(getattr(spec, "use_cad_superelements", True))
    ):
        return {}
    kit_id = spec.resolve_effective_gantry_hardware_kit()
    if kit_id == "procedural":
        return {}

    from printsim.schema.printer import GANTRY_HARDWARE_KIT_CATALOG

    stations = dict((GANTRY_HARDWARE_KIT_CATALOG.get(kit_id) or {}).get("stations") or {})
    try:
        from printsim.cad_assets.registry import has_hardware_asset, load_superelement
    except ImportError:
        return {}

    chamber_c = float(spec.frame.chamber_temp_c)
    bkt_mat_id = spec.gantry.bracket_material
    out: dict[str, object] = {}
    for tag, st_key in (
        ("fl", "trident_bed_arm_left"),
        ("fr", "trident_bed_arm_right"),
        ("r", "trident_bed_arm_rear"),
    ):
        aid = stations.get(st_key) or stations.get("trident_bed_arm")
        if not aid or not has_hardware_asset("gantry_brackets", aid):
            continue
        se = load_superelement(
            "gantry_brackets",
            aid,
            material_id=bkt_mat_id,
            chamber_temp_c=chamber_c,
        )
        if se is not None:
            out[tag] = se
    return out


def _build_cad_trident_bed_arm_superelement(
    se_id: str,
    asset: object,
    port_node_ids: tuple[str, str],
    p0_asm_m: np.ndarray,
    p1_asm_m: np.ndarray,
    spec: PrinterSpec,
):
    """Build a 12x12 Trident CAD bed-arm superelement between ``z_bed_*`` and ``z_arm_tip_*``.

    Uses the real baked Tet4 Craig-Bampton lateral/bending/torsional compliance while accounting
    for the intermediate TR8 leadscrew nut boss supporting vertical Z shear partway along the arm
    (preventing an unsupported full-length ABS cantilever from dominating Z gravity droop).
    """
    from printsim.geometry import builder as _b

    bkt_mat = get_material(spec.gantry.bracket_material)
    bolt_factor = float(bkt_mat.bolted_joint_stiffness_factor)
    extra_mult = max(bolt_factor, 1e-4)

    K_bb = np.asarray(asset.K_bb, dtype=float) * extra_mult  # type: ignore[attr-defined]
    K_11_raw = 0.5 * (K_bb[6:12, 6:12] + K_bb[6:12, 6:12].T)
    diag_k = np.maximum(np.diag(K_11_raw), 1e-12)
    s_inv = 1.0 / np.sqrt(diag_k)
    k_sc = (s_inv[:, None] * K_11_raw) * s_inv[None, :]
    try:
        c_sc = np.linalg.inv(k_sc)
    except np.linalg.LinAlgError:
        c_sc = np.linalg.pinv(k_sc, rcond=1e-10)
    C_11 = (s_inv[:, None] * c_sc) * s_inv[None, :]
    C_11 = 0.5 * (C_11 + C_11.T)

    # Account for TR8 leadscrew nut flange + M5 GE5C clevis + sleeved 2020 extrusion pocket
    # supporting the Trident bed arm at the Z-carriage (vs an un-nutted full-length ABS cantilever):
    # apply symmetric congruence scaling D @ C_11 @ D (strictly SPD-preserving) to bound the
    # supported clevis compliance while retaining baked 3D CAD anisotropy & material scaling.
    e_ratio = max(
        bkt_mat.effective_modulus_pa(float(spec.frame.chamber_temp_c)) / 2.1e9, 0.2
    )
    k_floors = (
        4.5e7 * e_ratio * bolt_factor,  # kx reinforced 2020-pocket lateral shear (N/m)
        4.5e7 * e_ratio * bolt_factor,  # ky reinforced 2020-pocket lateral shear (N/m)
        4.5e7 * e_ratio * bolt_factor,  # kz TR8 leadscrew nut-boss vertical support (N/m)
        2.5e4 * e_ratio * bolt_factor,  # krx supported clevis roll (N*m/rad)
        2.5e4 * e_ratio * bolt_factor,  # kry supported clevis pitch (N*m/rad)
        2.5e4 * e_ratio * bolt_factor,  # krz supported clevis yaw (N*m/rad)
    )
    d_scale = np.ones(6, dtype=float)
    for idx_dof, k_min in enumerate(k_floors):
        c_max = 1.0 / max(k_min, 1.0)
        if float(C_11[idx_dof, idx_dof]) > c_max:
            d_scale[idx_dof] = math.sqrt(c_max / float(C_11[idx_dof, idx_dof]))
    C_11 = (d_scale[:, None] * C_11) * d_scale[None, :]
    C_11 = 0.5 * (C_11 + C_11.T)

    # Symmetrize & invert cleanly back to SPD K_11_eff
    diag_c = np.maximum(np.diag(C_11), 1e-15)
    sc = 1.0 / np.sqrt(diag_c)
    c_s = (sc[:, None] * C_11) * sc[None, :]
    try:
        k_s = np.linalg.inv(c_s)
    except np.linalg.LinAlgError:
        k_s = np.linalg.pinv(c_s, rcond=1e-10)
    K_11_eff = (sc[:, None] * k_s) * sc[None, :]
    K_11_eff = 0.5 * (K_11_eff + K_11_eff.T)

    dp_asm = np.asarray(p1_asm_m, dtype=float) - np.asarray(p0_asm_m, dtype=float)
    T_01 = np.eye(6, dtype=float)
    T_01[:3, 3:6] = -_skew_3x3_local(dp_asm)
    B = np.hstack([-T_01, np.eye(6, dtype=float)])
    K_asm = B.T @ K_11_eff @ B
    K_asm = 0.5 * (K_asm + K_asm.T)

    mass_scale = float(getattr(asset, "mass_scale_applied", 1.0))
    mass_ref = float(asset.manifest.get("mass_kg", 0.045)) * mass_scale  # type: ignore[attr-defined]
    aid = str(asset.asset_id)  # type: ignore[attr-defined]
    return _b.AttachedSuperelement(
        id=se_id,
        category="gantry_brackets",
        asset_id=aid,
        port_node_ids=port_node_ids,
        K_cb=K_asm,
        M_cb=None,
        port_coords_cad_m=None,
        n_internal_modes=0,
        part_group_id="bed_support",
        part_group_label="Trident Z Bed-Arm Brackets (CAD)",
        role="bed",
        mass_kg=mass_ref,
        fixed_interface_freqs_hz=list(asset.fixed_interface_freqs_hz),  # type: ignore[attr-defined]
        manifest=dict(asset.manifest),  # type: ignore[attr-defined]
        visual_glb_url=f"/api/assets/hardware/gantry_brackets/{aid}/visual.glb",
        cad_trust_pct=int(asset.manifest.get("cad_trust_pct", 88)),  # type: ignore[attr-defined]
    )
