"""Linear rail Newmark composite mechanics, CF inserts, and gantry carriage helpers extracted from ``builder.py``."""

from __future__ import annotations

import math
from dataclasses import dataclass
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
    elif "_tube_" in member.profile.id:
        sub_kind = "metal_tube"
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
        rail_profile_id=rail_profile.id,
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
            "shear_kappa_y": x_rail_prof.shear_kappa_z,
            "shear_kappa_z": x_rail_prof.shear_kappa_y,
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
        elif "_tube_" in base_x_prof.id:
            sub_kind = "metal_tube"
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
                rail_profile_id=x_rail_prof.id,
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
    frame_arm: tuple[float, float, float] | None = None
    if y_mount == "inner" and rail_and_car_pos_m is not None:
        # Carriage node already sits on the raceway, apart from the rail node on the Y-extrusion
        # axis. The Hertzian contact acts at the raceway: rigid arm from the rail node (shell) out
        # to the carriage node, zero arm on the carriage node (frame). One shared arm for both
        # nodes is only valid when they coincide and would break rigid-body invariance here.
        p_rail, p_car = rail_and_car_pos_m
        lever_arm = tuple(float(v) for v in (np.asarray(p_car) - np.asarray(p_rail)))
        frame_arm = (0.0, 0.0, 0.0)
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
        offset_frame_m=frame_arm,
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


# ---------------------------------------------------------------------------
# CAL-13: Explicit 3D Pulley-Coordinate Belt Path & Idler Routing Graph
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PulleyNode3D:
    """Physical 3D pulley, idler, or belt-clamp station in the CoreXY belt routing graph."""

    id: str
    station: str  # "fl" | "fr" | "rl" | "rr" | "xy_l" | "xy_r" | "toolhead"
    role: str  # "drive" | "corner_idler" | "xy_joint_idler" | "guide_idler" | "clamp"
    belt_plane: str  # "A" (upper) | "B" (lower)
    position_m: tuple[float, float, float]
    shaft_architecture: str  # "motor_single_shear" | "motor_double_shear" | "single_shear_bolt" | "double_shear_bolt" | "live_shaft" | "clamp"
    z_overhang_extra_m: float
    wrap_angle_deg: float
    pulley_radius_m: float
    source: str  # "cad_station" | "parametric"

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "station": self.station,
            "role": self.role,
            "belt_plane": self.belt_plane,
            "position_mm": [round(v * 1e3, 2) for v in self.position_m],
            "shaft_architecture": self.shaft_architecture,
            "z_overhang_extra_mm": round(self.z_overhang_extra_m * 1e3, 2),
            "wrap_angle_deg": round(self.wrap_angle_deg, 1),
            "pulley_radius_mm": round(self.pulley_radius_m * 1e3, 2),
            "source": self.source,
        }


@dataclass(frozen=True)
class BeltSegment3D:
    """Directed straight 3D belt segment between two adjacent stations on a branch."""

    branch_id: str
    belt_plane: str
    start_id: str
    end_id: str
    start_m: tuple[float, float, float]
    end_m: tuple[float, float, float]
    length_m: float
    direction_unit: tuple[float, float, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "branch_id": self.branch_id,
            "belt_plane": self.belt_plane,
            "start_id": self.start_id,
            "end_id": self.end_id,
            "length_mm": round(self.length_m * 1e3, 2),
            "direction_unit": [round(v, 4) for v in self.direction_unit],
        }


@dataclass(frozen=True)
class BeltBranch3D:
    """One carriage-to-motor belt branch in the CoreXY 3D belt routing graph."""

    branch_id: str  # "A1" | "A2" | "B1" | "B2"
    belt_plane: str  # "A" (upper) | "B" (lower)
    node_ids: tuple[str, ...]
    segments: tuple[BeltSegment3D, ...]
    total_length_m: float
    n_pulley_tangents: float
    idler_node_ids: tuple[str, ...]
    drive_node_id: str

    def to_dict(self) -> dict[str, object]:
        return {
            "branch_id": self.branch_id,
            "belt_plane": self.belt_plane,
            "node_ids": list(self.node_ids),
            "total_length_mm": round(self.total_length_m * 1e3, 2),
            "n_pulley_tangents": round(self.n_pulley_tangents, 1),
            "idler_count": len(self.idler_node_ids),
            "drive_node_id": self.drive_node_id,
            "segments": [s.to_dict() for s in self.segments],
        }


@dataclass(frozen=True)
class CoreXYBeltGraph3D:
    """Explicit 3D pulley-coordinate CoreXY belt graph with stacked A/B planes & idler metadata."""

    routing: str
    drive_mode: str
    hardware_kit: str
    uses_cad_stations: bool
    u_rel: float
    v_rel: float
    z_gantry_m: float
    z_belt_mid_m: float
    z_upper_plane_m: float
    z_lower_plane_m: float
    dz_belt_planes_m: float
    upper_plane_z_lever_mm: float
    lower_plane_z_lever_mm: float
    belt_to_cog_z_lever_mm: float
    stacked_belt_twist_moment_nm_per_n: float
    pulleys: dict[str, PulleyNode3D]
    branches: dict[str, BeltBranch3D]
    longest_free_span_m: float
    total_active_belt_length_m: float
    effective_span_ratio_vs_stock: float

    def to_dict(self) -> dict[str, object]:
        return {
            "routing": self.routing,
            "drive_mode": self.drive_mode,
            "hardware_kit": self.hardware_kit,
            "uses_cad_stations": self.uses_cad_stations,
            "u_rel": round(self.u_rel, 4),
            "v_rel": round(self.v_rel, 4),
            "z_gantry_mm": round(self.z_gantry_m * 1e3, 2),
            "z_belt_mid_mm": round(self.z_belt_mid_m * 1e3, 2),
            "z_upper_plane_mm": round(self.z_upper_plane_m * 1e3, 2),
            "z_lower_plane_mm": round(self.z_lower_plane_m * 1e3, 2),
            "dz_belt_planes_mm": round(self.dz_belt_planes_m * 1e3, 2),
            "upper_plane_z_lever_mm": round(self.upper_plane_z_lever_mm, 2),
            "lower_plane_z_lever_mm": round(self.lower_plane_z_lever_mm, 2),
            "belt_to_cog_z_lever_mm": round(self.belt_to_cog_z_lever_mm, 2),
            "stacked_belt_twist_moment_nm_per_n": round(self.stacked_belt_twist_moment_nm_per_n, 5),
            "longest_free_span_mm": round(self.longest_free_span_m * 1e3, 2),
            "total_active_belt_length_mm": round(self.total_active_belt_length_m * 1e3, 2),
            "effective_span_ratio_vs_stock": round(self.effective_span_ratio_vs_stock, 4),
            "branch_lengths_mm": {
                bid: round(br.total_length_m * 1e3, 2) for bid, br in self.branches.items()
            },
            "pulleys": {pid: p.to_dict() for pid, p in self.pulleys.items()},
            "branches": {bid: br.to_dict() for bid, br in self.branches.items()},
        }


def _segment_turn_wrap_deg(
    p_prev: np.ndarray,
    p_curr: np.ndarray,
    p_next: np.ndarray,
) -> float:
    """Exact 3D belt deflection/wrap angle (deg) at an interior pulley node."""
    v_in = p_curr - p_prev
    v_out = p_next - p_curr
    n_in = float(np.linalg.norm(v_in))
    n_out = float(np.linalg.norm(v_out))
    if n_in <= 1e-9 or n_out <= 1e-9:
        return 0.0
    cos_turn = max(-1.0, min(1.0, float(np.dot(v_in, v_out)) / (n_in * n_out)))
    return float(math.degrees(math.acos(cos_turn)))


def _resolve_corexy_corner_pulley_coords_m(
    assembly: AssemblyModel,
    routing: str,
    use_cad_override: bool | None = None,
) -> tuple[dict[str, np.ndarray], bool]:
    """Resolve physical 3D corner pulley shaft coordinates ``{fl, fr, rl, rr}`` in meters.

    Uses baked CAD corner-mount stations (``pulley_fl``..``pulley_rr``) when present on the
    assembly and consistent with ``routing``, shifting 2020-baked stations inboard by
    ``0.5 * (b_upright - 20 mm)`` on wider 3030/4040 columns so pulley shafts clear the inner
    upright face. Falls back cleanly to parametric Voron/Monolith station geometry when
    procedural hardware is active or when evaluating a routing counterfactual.
    """
    from printsim.geometry.builder import corexy_frame_centerlines_m

    spec = assembly.spec
    nm = assembly.node_map
    w, d, h = corexy_frame_centerlines_m(spec)
    bx, by, _ = [float(v) * 1e-3 for v in spec.kinematics.build_volume_mm]
    z_g = (
        float(nm["y_rail_car_l"].position_m[2])
        if "y_rail_car_l" in nm
        else max(h - float(spec.frame.gantry_top_clearance_mm) * 1e-3, 0.08)
    )
    b_up = float(get_profile(spec.frame.effective_upright_profile).width_m)
    extra_up_in = max(0.0, 0.5 * (b_up - 0.020))
    is_awd = spec.gantry.drive_mode == "awd"

    has_cad_nodes = all(f"pulley_{t}" in nm for t in ("fl", "fr", "rl", "rr"))
    use_cad = has_cad_nodes if use_cad_override is None else (use_cad_override and has_cad_nodes)

    if use_cad:
        p_fl = np.asarray(nm["pulley_fl"].position_m, dtype=float).copy()
        p_fr = np.asarray(nm["pulley_fr"].position_m, dtype=float).copy()
        p_rl = np.asarray(nm["pulley_rl"].position_m, dtype=float).copy()
        p_rr = np.asarray(nm["pulley_rr"].position_m, dtype=float).copy()
        if extra_up_in > 1e-6:
            p_fl[0] += extra_up_in
            p_fl[1] += extra_up_in
            p_fr[0] -= extra_up_in
            p_fr[1] += extra_up_in
            p_rl[0] += extra_up_in
            p_rl[1] -= extra_up_in
            p_rr[0] -= extra_up_in
            p_rr[1] -= extra_up_in
    else:
        if routing == "monolith_short":
            dx_f = 0.0450 + extra_up_in
            dx_r = 0.0450 + extra_up_in
            dy_f = 0.0450 + extra_up_in
            dy_r = 0.0450 + extra_up_in
        else:
            # Voron 2.4 / Trident stock printed front-idler & rear A/B drive station offsets
            dx_f = 0.0190 + extra_up_in
            dy_f = 0.0240 + extra_up_in
            dx_r = 0.0490 + extra_up_in
            dy_r = 0.0140 + extra_up_in
        p_fl = np.array([dx_f, dy_f, z_g], dtype=float)
        p_fr = np.array([w - dx_f, dy_f, z_g], dtype=float)
        p_rl = np.array([dx_r, d - dy_r, z_g], dtype=float)
        p_rr = np.array([w - dx_r, d - dy_r, z_g], dtype=float)

    if routing == "monolith_short":
        # Monolith short-path (No-Protrusion / inboard pulley pocket) tucks corner motor/idler
        # centers inside the 45 mm CAD corner bracket envelope (monolith_awd_front_left/xy_left)
        # and aligns front & rear stations collinearly on the Y-rail axis for both 2WD and AWD
        # while preserving build travel clearance.
        max_y_in = max(0.015, 0.5 * (d - by) - 0.012)
        target_y_in = min(0.0450 + extra_up_in, max_y_in)
        if float(p_fl[1]) < target_y_in:
            p_fl[1] = target_y_in
            p_fr[1] = target_y_in
        if (d - float(p_rl[1])) < target_y_in:
            p_rl[1] = d - target_y_in
            p_rr[1] = d - target_y_in
        max_x_in = max(0.020, 0.5 * (w - bx) - 0.012)
        target_x_in = min(0.0450 + extra_up_in, max_x_in)
        x_col = min(max(float(p_fl[0]), float(p_rl[0]), target_x_in), max_x_in)
        p_fl[0] = x_col
        p_rl[0] = x_col
        p_fr[0] = w - x_col
        p_rr[0] = w - x_col

    # Physical inner-envelope safety clamp for very tight custom frames
    for pt_l, pt_r in ((p_fl, p_fr), (p_rl, p_rr)):
        x_l = float(np.clip(pt_l[0], 0.010, max(0.5 * w - 0.03, 0.015)))
        x_r = float(np.clip(pt_r[0], min(0.5 * w + 0.03, w - 0.015), max(w - 0.010, 0.02)))
        pt_l[0] = x_l
        pt_r[0] = x_r
    for pt_f in (p_fl, p_fr):
        pt_f[1] = float(np.clip(pt_f[1], 0.004, max(0.5 * d - 0.03, 0.010)))
    for pt_r in (p_rl, p_rr):
        pt_r[1] = float(np.clip(pt_r[1], min(0.5 * d + 0.03, d - 0.010), max(d - 0.004, 0.02)))

    return {"fl": p_fl, "fr": p_fr, "rl": p_rl, "rr": p_rr}, use_cad


def build_corexy_belt_routing_graph_3d(
    assembly: AssemblyModel,
    u_rel: float | None = None,
    v_rel: float | None = None,
    routing_override: str | None = None,
    include_stock_comparison: bool = True,
) -> CoreXYBeltGraph3D:
    """Build the explicit 3D pulley-coordinate CoreXY belt routing graph at ``(u_rel, v_rel)``.

    Models:
      - Stacked upper (Belt A, ``z_A = z_belt_mid + 0.5 * dz_belt``) and lower (Belt B,
        ``z_B = z_belt_mid - 0.5 * dz_belt``) belt planes and exact vertical Z lever arms
        relative to the gantry neutral axis and toolhead CoG.
      - Exact 3D ``(x, y, z)`` pulley/idler/clamp coordinates from CAD corner stations when
        present (falling back to parametric Voron/Monolith station geometry otherwise).
      - ``stock_corexy`` rear A/B drive guide-idler S-wrap vs ``monolith_short`` direct-entry
        collinear live-shaft routing (replacing the legacy scalar ``0.86x`` span factor).
      - Per-station shaft architecture (single-shear vs double-shear motor shaft, single-shear
        vs double-shear M5 bolt idlers, and Monolith press-fit live-shaft idlers) with exact
        3D wrap angles and upper/lower Z overhang lever arms.
    """
    from printsim.geometry.builder import corexy_frame_centerlines_m

    spec = assembly.spec
    nm = assembly.node_map
    w, d, h = corexy_frame_centerlines_m(spec)
    bx, by, _ = [float(v) * 1e-3 for v in spec.kinematics.build_volume_mm]
    rel_pos = getattr(assembly, "toolhead_rel_pos", (0.5, 0.5, 0.5))
    u = float(min(max(rel_pos[0] if u_rel is None else u_rel, 0.05), 0.95))
    v = float(min(max(rel_pos[1] if v_rel is None else v_rel, 0.05), 0.95))

    orig_x = 0.5 * (w - bx)
    orig_y = 0.5 * (d - by)
    x_th = orig_x + u * bx
    y_th = orig_y + v * by

    z_g = (
        float(nm["y_rail_car_l"].position_m[2])
        if "y_rail_car_l" in nm
        else max(h - float(spec.frame.gantry_top_clearance_mm) * 1e-3, 0.08)
    )
    belt = spec.gantry.belt
    r_p = float(belt.pitch_radius_m)
    dz_belt = (float(belt.width_mm) + 3.0) * 1e-3  # belt width + 3.0 mm flange/washer gap
    z_belt_mid = z_g + float(spec.toolhead.belt_plane_z_offset_mm) * 1e-3
    z_a = z_belt_mid + 0.5 * dz_belt
    z_b = z_belt_mid - 0.5 * dz_belt
    z_cog = z_g + float(spec.toolhead.cog_offset_mm[2]) * 1e-3

    routing = routing_override if routing_override is not None else spec.gantry.belt_routing
    is_awd = spec.gantry.drive_mode == "awd"
    hw_kit = spec.resolve_effective_gantry_hardware_kit()
    use_cad_override = None if routing_override is None else False
    corners, uses_cad = _resolve_corexy_corner_pulley_coords_m(
        assembly, routing=routing, use_cad_override=use_cad_override
    )
    p_fl, p_fr, p_rl, p_rr = corners["fl"], corners["fr"], corners["rl"], corners["rr"]

    # Inner Y-run tangent line aligned with front corner pulley inner pitch tangent
    x_yrun_l = min(float(p_fl[0]) + r_p, max(orig_x - 0.005, float(p_fl[0])))
    x_yrun_r = max(float(p_fr[0]) - r_p, min(orig_x + bx + 0.005, float(p_fr[0])))

    # Rear drive pulley pitch-tangent X coordinate: collinear with Y-run when motor shafts align
    # (Monolith AWD), or exact CAD rear motor shaft X when laterally inset (Voron Stock / Siboor / VzBot).
    x_drive_rl = x_yrun_l if abs(float(p_rl[0]) - float(p_fl[0])) < 1.5e-3 else float(p_rl[0])
    x_drive_rr = x_yrun_r if abs(float(p_rr[0]) - float(p_fr[0])) < 1.5e-3 else float(p_rr[0])

    # Compact carriage belt-clamp half-spacing (belt teeth lock ±6 mm from carriage center)
    x_clamp = 0.006

    # CAD station Z offsets (e.g. VzBot rear motor at -2.0 mm relative to gantry plane)
    dz_fl = float(p_fl[2]) - z_g
    dz_fr = float(p_fr[2]) - z_g
    dz_rl = float(p_rl[2]) - z_g
    dz_rr = float(p_rr[2]) - z_g

    has_rear_swrap = routing == "stock_corexy"
    swrap_dx = 0.010
    swrap_dy = 0.006

    motor_arch = "motor_double_shear" if spec.gantry.double_shear_pulleys else "motor_single_shear"
    if routing == "monolith_short":
        idler_arch = "live_shaft"
        guide_arch = "live_shaft"
    elif spec.gantry.double_shear_pulleys:
        idler_arch = "double_shear_bolt"
        guide_arch = "double_shear_bolt"
    else:
        # Stock Voron XY-joint & front-idler clevis pockets capture both ends of the M5 bolt
        # inside printed ABS ears (double-shear bolt with ABS root & stacked overhang), whereas
        # single-shear rear A/B drive guide posts and NEMA 17 motor shafts act as cantilevers.
        idler_arch = "double_shear_bolt"
        guide_arch = "single_shear_bolt"

    src_corner = "cad_station" if uses_cad else "parametric"
    src_xy = "cad_station" if (uses_cad and "x_beam_l" in nm) else "parametric"

    # Build per-branch ordered 3D waypoint descriptors:
    # (node_id, station, role, xyz_np, shaft_arch, z_overhang_extra_m, source)
    def _branch_specs(bid: str, plane: str, pz: float) -> list[tuple[str, str, str, np.ndarray, str, float, str]]:
        # Upper belt plane (A) sits dz_belt further along a bottom-rooted single-shear motor shaft
        # on stock Voron A/B drives, and +0.5*dz_belt on Monolith's compact inverted hub or a
        # stacked dual-plane M5 bolt; live-shafts straddle each belt plane individually (0 extra).
        if motor_arch == "motor_single_shear":
            if plane == "A":
                z_extra_motor = 0.5 * dz_belt if routing == "monolith_short" else dz_belt
            else:
                z_extra_motor = 0.0
        elif motor_arch == "motor_double_shear" and routing == "stock_corexy":
            z_extra_motor = 0.5 * dz_belt
        else:
            z_extra_motor = 0.0
        z_extra_idler = (
            0.0
            if idler_arch == "live_shaft"
            else (0.5 * dz_belt if plane == "A" else 0.0)
        )

        th_l = np.array([x_th - x_clamp, y_th, pz], dtype=float)
        th_r = np.array([x_th + x_clamp, y_th, pz], dtype=float)
        # Exact orthogonal 90-deg belt-tangent intersection at left/right XY-joint idlers
        xy_lf = np.array([x_yrun_l, y_th, pz], dtype=float)
        xy_lr = np.array([x_yrun_l, y_th, pz], dtype=float)
        xy_rf = np.array([x_yrun_r, y_th, pz], dtype=float)
        xy_rr = np.array([x_yrun_r, y_th, pz], dtype=float)

        # Inner pitch-tangent stations at front/rear corner pulleys (offset inboard by r_p along Y)
        y_tan_fl = min(float(p_fl[1]) + r_p, orig_y - 0.002)
        y_tan_fr = min(float(p_fr[1]) + r_p, orig_y - 0.002)
        y_tan_rl = max(float(p_rl[1]) - r_p, orig_y + by + 0.002)
        y_tan_rr = max(float(p_rr[1]) - r_p, orig_y + by + 0.002)
        c_fl = np.array([x_yrun_l, y_tan_fl, pz + dz_fl], dtype=float)
        c_fr = np.array([x_yrun_r, y_tan_fr, pz + dz_fr], dtype=float)
        c_rl = np.array([x_drive_rl, y_tan_rl, pz + dz_rl], dtype=float)
        c_rr = np.array([x_drive_rr, y_tan_rr, pz + dz_rr], dtype=float)
        rg_l = np.array([x_yrun_l, y_tan_rl, pz + dz_rl], dtype=float)
        rg_r = np.array([x_yrun_r, y_tan_rr, pz + dz_rr], dtype=float)
        sw_l_in = np.array([x_drive_rl + swrap_dx, y_tan_rl + swrap_dy, pz + dz_rl], dtype=float)
        sw_l_out = np.array([x_drive_rl - swrap_dx, y_tan_rl + swrap_dy, pz + dz_rl], dtype=float)
        sw_r_in = np.array([x_drive_rr - swrap_dx, y_tan_rr + swrap_dy, pz + dz_rr], dtype=float)
        sw_r_out = np.array([x_drive_rr + swrap_dx, y_tan_rr + swrap_dy, pz + dz_rr], dtype=float)

        need_rg_l = abs(x_drive_rl - x_yrun_l) > 1e-4 or has_rear_swrap
        need_rg_r = abs(x_drive_rr - x_yrun_r) > 1e-4 or has_rear_swrap

        if is_awd:
            if bid == "A1":
                pts = [
                    ("clamp_r_A", "toolhead", "clamp", th_r, "clamp", 0.0, "parametric"),
                    ("xy_rr_A", "xy_r", "xy_joint_idler", xy_rr, idler_arch, z_extra_idler, src_xy),
                ]
                if need_rg_r:
                    pts.append(("rear_guide_r_A", "rr", "guide_idler", rg_r, guide_arch, z_extra_idler, src_corner))
                if has_rear_swrap:
                    pts.append(("rear_swrap_r_A1", "rr", "guide_idler", sw_r_in, guide_arch, z_extra_idler, src_corner))
                pts.append(("drive_rr_A", "rr", "drive", c_rr, motor_arch, z_extra_motor, src_corner))
                return pts
            if bid == "A2":
                return [
                    ("clamp_l_A", "toolhead", "clamp", th_l, "clamp", 0.0, "parametric"),
                    ("xy_lf_A", "xy_l", "xy_joint_idler", xy_lf, idler_arch, z_extra_idler, src_xy),
                    ("drive_fl_A", "fl", "drive", c_fl, motor_arch, z_extra_motor, src_corner),
                ]
            if bid == "B1":
                pts = [
                    ("clamp_l_B", "toolhead", "clamp", th_l, "clamp", 0.0, "parametric"),
                    ("xy_lr_B", "xy_l", "xy_joint_idler", xy_lr, idler_arch, z_extra_idler, src_xy),
                ]
                if need_rg_l:
                    pts.append(("rear_guide_l_B", "rl", "guide_idler", rg_l, guide_arch, z_extra_idler, src_corner))
                if has_rear_swrap:
                    pts.append(("rear_swrap_l_B1", "rl", "guide_idler", sw_l_in, guide_arch, z_extra_idler, src_corner))
                pts.append(("drive_rl_B", "rl", "drive", c_rl, motor_arch, z_extra_motor, src_corner))
                return pts
            return [
                ("clamp_r_B", "toolhead", "clamp", th_r, "clamp", 0.0, "parametric"),
                ("xy_rf_B", "xy_r", "xy_joint_idler", xy_rf, idler_arch, z_extra_idler, src_xy),
                ("drive_fr_B", "fr", "drive", c_fr, motor_arch, z_extra_motor, src_corner),
            ]

        # 2WD perimeter CoreXY routing: motors at Rear-Right (Belt A) and Rear-Left (Belt B)
        if bid == "A1":
            pts = [
                ("clamp_r_A", "toolhead", "clamp", th_r, "clamp", 0.0, "parametric"),
                ("xy_rf_A", "xy_r", "xy_joint_idler", xy_rf, idler_arch, z_extra_idler, src_xy),
                ("corner_fr_A", "fr", "corner_idler", c_fr, idler_arch, z_extra_idler, src_corner),
            ]
            if need_rg_r:
                pts.append(("rear_guide_r_A", "rr", "guide_idler", rg_r, guide_arch, z_extra_idler, src_corner))
            if has_rear_swrap:
                pts.append(("rear_swrap_r_A1", "rr", "guide_idler", sw_r_in, guide_arch, z_extra_idler, src_corner))
            pts.append(("drive_rr_A", "rr", "drive", c_rr, motor_arch, z_extra_motor, src_corner))
            return pts
        if bid == "A2":
            pts = [
                ("clamp_l_A", "toolhead", "clamp", th_l, "clamp", 0.0, "parametric"),
                ("xy_lr_A", "xy_l", "xy_joint_idler", xy_lr, idler_arch, z_extra_idler, src_xy),
                ("corner_rl_A", "rl", "corner_idler", rg_l, idler_arch, z_extra_idler, src_corner),
            ]
            if has_rear_swrap:
                pts.append(("rear_swrap_r_A2", "rr", "guide_idler", sw_r_out, guide_arch, z_extra_idler, src_corner))
            pts.append(("drive_rr_A", "rr", "drive", c_rr, motor_arch, z_extra_motor, src_corner))
            return pts
        if bid == "B1":
            pts = [
                ("clamp_l_B", "toolhead", "clamp", th_l, "clamp", 0.0, "parametric"),
                ("xy_lf_B", "xy_l", "xy_joint_idler", xy_lf, idler_arch, z_extra_idler, src_xy),
                ("corner_fl_B", "fl", "corner_idler", c_fl, idler_arch, z_extra_idler, src_corner),
            ]
            if need_rg_l:
                pts.append(("rear_guide_l_B", "rl", "guide_idler", rg_l, guide_arch, z_extra_idler, src_corner))
            if has_rear_swrap:
                pts.append(("rear_swrap_l_B1", "rl", "guide_idler", sw_l_in, guide_arch, z_extra_idler, src_corner))
            pts.append(("drive_rl_B", "rl", "drive", c_rl, motor_arch, z_extra_motor, src_corner))
            return pts
        pts = [
            ("clamp_r_B", "toolhead", "clamp", th_r, "clamp", 0.0, "parametric"),
            ("xy_rr_B", "xy_r", "xy_joint_idler", xy_rr, idler_arch, z_extra_idler, src_xy),
            ("corner_rr_B", "rr", "corner_idler", rg_r, idler_arch, z_extra_idler, src_corner),
        ]
        if has_rear_swrap:
            pts.append(("rear_swrap_l_B2", "rl", "guide_idler", sw_l_out, guide_arch, z_extra_idler, src_corner))
        pts.append(("drive_rl_B", "rl", "drive", c_rl, motor_arch, z_extra_motor, src_corner))
        return pts

    pulleys: dict[str, PulleyNode3D] = {}
    branches: dict[str, BeltBranch3D] = {}
    drive_wrap_deg = float(belt.wrap_angle_deg)
    longest_seg = 0.0

    for bid, plane, pz in (("A1", "A", z_a), ("A2", "A", z_a), ("B1", "B", z_b), ("B2", "B", z_b)):
        raw_pts = _branch_specs(bid, plane, pz)
        segments: list[BeltSegment3D] = []
        idler_ids: list[str] = []
        node_ids: list[str] = []
        total_len = 0.0
        n_pts = len(raw_pts)

        for i, (nid, st, role, xyz, s_arch, z_ex, src) in enumerate(raw_pts):
            node_ids.append(nid)
            if role == "drive":
                w_deg = drive_wrap_deg
            elif role == "clamp":
                w_deg = 0.0
            elif 0 < i < n_pts - 1:
                w_deg = _segment_turn_wrap_deg(raw_pts[i - 1][3], xyz, raw_pts[i + 1][3])
                idler_ids.append(nid)
            else:
                w_deg = 90.0
                idler_ids.append(nid)
            # Keep maximum wrap angle if a drive node is shared by both branches in 2WD
            prev_node = pulleys.get(nid)
            if prev_node is None or w_deg > prev_node.wrap_angle_deg:
                pulleys[nid] = PulleyNode3D(
                    id=nid,
                    station=st,
                    role=role,
                    belt_plane=plane,
                    position_m=(float(xyz[0]), float(xyz[1]), float(xyz[2])),
                    shaft_architecture=s_arch,
                    z_overhang_extra_m=float(z_ex),
                    wrap_angle_deg=float(w_deg),
                    pulley_radius_m=r_p,
                    source=src,
                )

        for i in range(n_pts - 1):
            nid0, _, _, p0, _, _, _ = raw_pts[i]
            nid1, _, _, p1, _, _, _ = raw_pts[i + 1]
            diff = p1 - p0
            seg_len = float(np.linalg.norm(diff))
            u_vec = diff / max(seg_len, 1e-12)
            segments.append(
                BeltSegment3D(
                    branch_id=bid,
                    belt_plane=plane,
                    start_id=nid0,
                    end_id=nid1,
                    start_m=(float(p0[0]), float(p0[1]), float(p0[2])),
                    end_m=(float(p1[0]), float(p1[1]), float(p1[2])),
                    length_m=seg_len,
                    direction_unit=(float(u_vec[0]), float(u_vec[1]), float(u_vec[2])),
                )
            )
            total_len += seg_len
            if seg_len > longest_seg:
                longest_seg = seg_len

        # Pulley tangent points contributing Euler-Bernoulli wrap flexural-fillet compliance:
        # 2 tangents per active 90/180-deg idler (entry + exit) + drive entry + clamp exit
        active_idlers = sum(1 for iid in idler_ids if pulleys[iid].wrap_angle_deg >= 15.0)
        n_tangents = max(2.0 * float(active_idlers) + 2.0, 4.0)
        branches[bid] = BeltBranch3D(
            branch_id=bid,
            belt_plane=plane,
            node_ids=tuple(node_ids),
            segments=tuple(segments),
            total_length_m=max(total_len, 0.12 if is_awd else 0.20),
            n_pulley_tangents=n_tangents,
            idler_node_ids=tuple(idler_ids),
            drive_node_id=raw_pts[-1][0],
        )

    total_active_len = sum(br.total_length_m for br in branches.values())
    # Exact longest unobstructed Y or X free span between corner stations for pluck/whip diagnostics
    span_x_free = max(x_yrun_r - x_yrun_l, 0.10)
    span_y_free = max(
        float(p_rl[1]) - float(p_fl[1]),
        float(p_rr[1]) - float(p_fr[1]),
        0.10,
    )
    longest_free_span = max(longest_seg, span_x_free, span_y_free)

    eff_ratio_vs_stock = 1.0
    if include_stock_comparison and routing != "stock_corexy":
        ref_stock = build_corexy_belt_routing_graph_3d(
            assembly,
            u_rel=u,
            v_rel=v,
            routing_override="stock_corexy",
            include_stock_comparison=False,
        )
        eff_ratio_vs_stock = total_active_len / max(ref_stock.total_active_belt_length_m, 1e-6)

    return CoreXYBeltGraph3D(
        routing=routing,
        drive_mode="awd" if is_awd else "2wd",
        hardware_kit=hw_kit,
        uses_cad_stations=uses_cad,
        u_rel=u,
        v_rel=v,
        z_gantry_m=z_g,
        z_belt_mid_m=z_belt_mid,
        z_upper_plane_m=z_a,
        z_lower_plane_m=z_b,
        dz_belt_planes_m=dz_belt,
        upper_plane_z_lever_mm=(z_a - z_g) * 1e3,
        lower_plane_z_lever_mm=(z_b - z_g) * 1e3,
        belt_to_cog_z_lever_mm=(z_cog - z_belt_mid) * 1e3,
        stacked_belt_twist_moment_nm_per_n=dz_belt,
        pulleys=pulleys,
        branches=branches,
        longest_free_span_m=longest_free_span,
        total_active_belt_length_m=total_active_len,
        effective_span_ratio_vs_stock=eff_ratio_vs_stock,
    )
