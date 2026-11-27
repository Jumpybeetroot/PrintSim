"""Toolhead accessories, 3-stage link compliance, and CAD superelements extracted from ``builder.py``."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from printsim.schema import (
    HOTEND_CATALOG,
    PROFILE_CATALOG,
    TOOLHEAD_CONSTRUCTION_CATALOG,
    TOOLHEAD_PRESETS,
    PrinterSpec,
    ProfileSpec,
    get_material,
    get_profile,
    resolve_hotend_clamp,
    resolve_rail_quality,
    resolve_toolhead_construction,
    resolve_z_probe,
)

if TYPE_CHECKING:
    from printsim.geometry.builder import (
        AssemblyModel,
        AttachedSuperelement,
        FastenerSpring,
        StructuralMember,
        StructuralNode,
    )


# Exact CAD-derived hardware accessory masses (g) & CoG offsets relative to nozzle tip (mm)
# Measured from watertight B-Rep/mesh solids in mbt_tricorn_v2.step & tricorn_hotends.step:
#   - Microwaved Ti bottom brace (0.31 g Ti-6Al-4V) + 3x M3 SS screws (1.35 g) = 1.67 g @ [0.00, +9.79, +18.00] mm from nozzle tip
#   - Microwaved optional SLM Ti nozzle shield (0.82 g Ti-6Al-4V) + 1x M3 SS screw (0.45 g) = 1.27 g @ [+3.91, -2.71, +11.23] mm from nozzle tip
#   - Stock Tricorn SLM Ti nozzle bucket shield (602.3 mm^3 Ti-6Al-4V) = 2.67 g @ [0.00, +0.47, +8.14] mm from nozzle tip
_MBT_TI_BRACE_MASS_G = 1.67
_MBT_TI_BRACE_REL_NOZ_MM = np.array([0.0, 9.79, 18.0], dtype=float)
_MBT_SHIELD_MASS_G = 1.27
_MBT_SHIELD_REL_NOZ_MM = np.array([3.91, -2.71, 11.23], dtype=float)
_TRICORN_STOCK_SHIELD_MASS_G = 2.67
_TRICORN_STOCK_SHIELD_REL_NOZ_MM = np.array([0.0, 0.47, 8.14], dtype=float)


def resolve_toolhead_hardware_accessories(spec: PrinterSpec) -> dict[str, Any]:
    """Resolve optional CAD toolhead/hotend accessories (Ti bottom brace & nozzle shield) into exact mass and CoG."""
    th_pid = (spec.toolhead.preset_id or "").lower().strip()
    he_pid = (spec.toolhead.hotend.preset_id or "").lower().strip()
    is_mbt = th_pid == "microwaved_bowden"
    is_tricorn = he_pid.startswith("tricorn_")
    brace_compatible = is_mbt and he_pid == "tricorn_long"
    shield_compatible = is_mbt and (is_tricorn or not he_pid)

    use_brace = bool(getattr(spec.toolhead, "use_hotend_bottom_brace", True))
    raw_shield = str(getattr(spec.toolhead, "nozzle_shield_mode", "auto") or "auto").lower().strip()

    options: list[str] = []
    if is_mbt and is_tricorn:
        options = ["microwaved", "stock", "none"]
    elif is_tricorn:
        options = ["stock", "none"]
    elif shield_compatible:
        options = ["microwaved", "none"]

    if not options:
        resolved_shield = "none" if (is_mbt and he_pid and not is_tricorn) else "stock"
    elif raw_shield == "auto":
        resolved_shield = "microwaved" if shield_compatible else "stock"
    elif raw_shield in options:
        resolved_shield = raw_shield
    else:
        resolved_shield = options[0]

    base_body_g = float(spec.toolhead.total_mass_g)
    base_hotend_g = float(spec.toolhead.hotend.mass_g)
    base_total_g = max(base_body_g + base_hotend_g, 1.0)
    base_cog_mm = np.array(spec.toolhead.cog_offset_mm, dtype=float)
    noz_mm = np.array(spec.toolhead.nozzle_offset_mm, dtype=float)

    body_delta_g = 0.0
    hotend_delta_g = 0.0
    moment_delta_g_mm = np.zeros(3, dtype=float)

    # Microwaved dressed preset baseline (130 g) includes its Ti bottom brace (1.67 g) and optional Ti shield (1.27 g).
    # Only include the Ti bottom brace toggle delta when paired with the matching 66 mm Tricorn Long envelope.
    if brace_compatible and not use_brace:
        body_delta_g -= _MBT_TI_BRACE_MASS_G
        moment_delta_g_mm -= _MBT_TI_BRACE_MASS_G * (noz_mm + _MBT_TI_BRACE_REL_NOZ_MM)

    if shield_compatible and resolved_shield != "microwaved":
        body_delta_g -= _MBT_SHIELD_MASS_G
        moment_delta_g_mm -= _MBT_SHIELD_MASS_G * (noz_mm + _MBT_SHIELD_REL_NOZ_MM)

    # Stock Tricorn catalog baseline (62 g Long / 52 g Short) includes the stock Ti bucket shield (2.67 g).
    if is_tricorn and resolved_shield != "stock":
        hotend_delta_g -= _TRICORN_STOCK_SHIELD_MASS_G
        moment_delta_g_mm -= _TRICORN_STOCK_SHIELD_MASS_G * (noz_mm + _TRICORN_STOCK_SHIELD_REL_NOZ_MM)

    eff_body_g = max(base_body_g + body_delta_g, 10.0)
    eff_hotend_g = max(base_hotend_g + hotend_delta_g, 5.0)
    eff_total_g = eff_body_g + eff_hotend_g
    eff_cog_mm = (base_total_g * base_cog_mm + moment_delta_g_mm) / max(eff_total_g, 1.0)

    return {
        "use_hotend_bottom_brace": bool(use_brace and brace_compatible),
        "nozzle_shield_mode": resolved_shield,
        "nozzle_shield_options": options,
        "body_delta_g": round(body_delta_g, 2),
        "hotend_delta_g": round(hotend_delta_g, 2),
        "accessory_mass_delta_g": round(body_delta_g + hotend_delta_g, 2),
        "effective_body_mass_g": eff_body_g,
        "effective_hotend_mass_g": eff_hotend_g,
        "effective_total_mass_g": eff_total_g,
        "effective_cog_offset_mm": (
            float(eff_cog_mm[0]),
            float(eff_cog_mm[1]),
            float(eff_cog_mm[2]),
        ),
    }


def _effective_toolhead_mass_kg(spec: PrinterSpec) -> float:
    """Return effective assembled toolhead mass (kg), including CAD accessories & X-carriage block delta."""
    from printsim.geometry import builder as _b

    acc = resolve_toolhead_hardware_accessories(spec)
    _, block_delta_g = _b.x_carriage_block_mass_delta_g(spec)
    return max((float(acc["effective_total_mass_g"]) + block_delta_g) * 1e-3, 0.03)


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
    probe_cfg = resolve_z_probe(spec)
    probe_comp = float(probe_cfg["compliance_factor"])
    chamber_c = spec.frame.chamber_temp_c

    car_mat = get_material(spec.toolhead.effective_carriage_mount_material)
    body_mat = get_material(spec.toolhead.effective_body_material)
    hm_mat = get_material(spec.toolhead.effective_hotend_mount_material)

    e_car = (
        car_mat.effective_modulus_pa(chamber_c)
        * float(car_mat.bolted_joint_stiffness_factor)
        * probe_comp
    )
    e_body = body_mat.effective_modulus_pa(chamber_c) * probe_comp
    e_hm = (
        hm_mat.effective_modulus_pa(chamber_c)
        * float(hm_mat.bolted_joint_stiffness_factor)
        * probe_comp
    )
    body_retention = body_mat.modulus_retention_at_temp(chamber_c)

    cog_x_mm, cog_y_mm, _ = spec.toolhead.cog_offset_mm
    noz_z_drop_m = max(abs(float(spec.toolhead.nozzle_offset_mm[2])) * 1e-3, 0.030)
    is_axisymmetric = (
        spec.architecture == "delta"
        and abs(float(cog_x_mm)) < 1e-4
        and abs(float(cog_y_mm)) < 1e-4
    )

    # Stage 1: Linear rail carriage Hertzian stiffness + carriage mount backplate
    if spec.architecture != "delta":
        from printsim.fea.components import (
            dual_rail_steiner_efficiency,
            hertzian_rail_carriage_stiffness,
        )

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
        is_dual = spec.gantry.effective_dual_x_rails
        car_mult = 2.0 if is_dual else 1.0
        cf_mount = spec.gantry.effective_cf_rail_mounting
        if is_dual:
            x_beam_h = get_profile(spec.gantry.x_beam_profile).height_m
            x_rail_prof_opt = PROFILE_CATALOG.get(spec.gantry.x_rail_profile)
            x_rail_h = x_rail_prof_opt.height_m if x_rail_prof_opt is not None else 0.015
            z_track = 0.5 * (x_beam_h + x_rail_h)
            eta_steiner = dual_rail_steiner_efficiency(spec.gantry.x_beam_profile, cf_mount)
            k_rx_car = 2.0 * car["k_rx_nm_rad"] + eta_steiner * 2.0 * car["k_yy_n_m"] * (z_track**2)
        else:
            k_rx_car = car["k_rx_nm_rad"]
        k_ry_car = car["k_ry_nm_rad"] * car_mult
        k_rz_car = car["k_rz_nm_rad"] * car_mult
        k_rad_car = 0.5 * (car["k_yy_n_m"] + car["k_zz_n_m"]) * car_mult

        x_beam_prof = get_profile(spec.gantry.x_beam_profile)
        x_rail_prof_opt = PROFILE_CATALOG.get(spec.gantry.x_rail_profile)
        w_rail_m = x_rail_prof_opt.width_m if x_rail_prof_opt is not None else 0.012
        if spec.gantry.rail_block_type == "H":
            l_block_m = 0.0299 if "mgn9" in spec.gantry.x_rail_profile else 0.0324
        else:
            l_block_m = 0.022

        if cf_mount != "none":
            # 3D orthotropic Tet10 FEA calibrated rail-seat shell/wall stiffness on hollow CF tubes
            # (scripts/calibrate_xbeam_3d_fea.py): accounts for steel rail base width bridging
            # (b_eff = b_face - 0.68*w_rail) across 1.0 / 1.8 / 2.0 mm walls & MGN9/MGN12 widths.
            x_beam_mat = get_material(spec.gantry.x_beam_material)
            t_wall_m = max(
                float(
                    x_beam_prof.wall_thickness_m
                    if x_beam_prof.wall_thickness_m is not None
                    else 2.0e-3
                ),
                0.8e-3,
            )
            b_face_m = max(x_beam_prof.width_m - 2.0 * t_wall_m, 0.004)
            e_axial_t = x_beam_mat.effective_modulus_pa(chamber_c)
            r_g_cf = x_beam_mat.shear_modulus_retention_at_temp(chamber_c)
            e_trans_pa = max(0.25 * e_axial_t, 14.0e9 * r_g_cf)
            l_eff_m = l_block_m + b_face_m
            b_eff_m = max(b_face_m - 0.68 * w_rail_m, 0.30 * b_face_m)

            k_wall_rad_1 = (
                62.9e6
                * (e_trans_pa / 21.78e9)
                * ((t_wall_m / 0.0018) ** 1.75)
                * (l_eff_m / 0.0488)
                * ((0.00824 / b_eff_m) ** 2.15)
            )
            k_wall_roll_1 = (
                2550.0
                * (e_trans_pa / 21.78e9)
                * ((t_wall_m / 0.0018) ** 1.40)
                * (l_eff_m / 0.0488)
                * ((0.00824 / b_eff_m) ** 2.20)
            )
            k_wall_pitch_1 = 10.8 * k_wall_roll_1

            if cf_mount == "two_wall_clamped_inserts":
                # 100%-infill annealed PET-CF core (E~5.5 GPa) + front-to-back M3 clamping
                # (3D FEA local_seat: 195.2 N/um MGN12H, 143.1 N/um MGN9H radial; ~80 kNm/rad seat roll)
                e_core_pa = 5.5e9 * max(0.45, r_g_cf)
                k_core_rad = e_core_pa * (w_rail_m * l_block_m) / (0.95 * b_face_m)
                k_seat_rad = (k_core_rad + k_wall_rad_1) * car_mult
                k_rad_car = (k_rad_car * k_seat_rad) / max(k_rad_car + k_seat_rad, 1e-9)
                if not is_dual:
                    k_seat_roll_2w = 80000.0 * (w_rail_m / 0.012) * max(0.45, r_g_cf)
                    k_rx_car = (k_rx_car * k_seat_roll_2w) / max(k_rx_car + k_seat_roll_2w, 1e-9)
            else:
                # Single-wall bare CF: unsupported thin wall oil-cans under radial, roll & pitch loads
                k_seat_rad = k_wall_rad_1 * car_mult
                k_rad_car = (k_rad_car * k_seat_rad) / max(k_rad_car + k_seat_rad, 1e-9)
                if not is_dual:
                    k_rx_car = (k_rx_car * k_wall_roll_1) / max(k_rx_car + k_wall_roll_1, 1e-9)
                k_seat_pitch = k_wall_pitch_1 * car_mult
                k_ry_car = (k_ry_car * k_seat_pitch) / max(k_ry_car + k_seat_pitch, 1e-9)
        elif x_beam_prof.shape == "extrusion":
            # 3D Tet10 FEA calibrated HFS5-2020 T-slot shoulder seat compliance:
            # 271.7 N/um MGN12H & 183.2 N/um MGN9H radial; 8170 / 4980 Nm/rad single-rail lip roll;
            # 82.4 / 42.5 kNm/rad pitch (dual_top_bottom roll uses 3D-calibrated eta_steiner directly).
            l_scale = (l_block_m / 0.0324) ** 0.50
            k_seat_rad_ext = 270.0e6 * ((w_rail_m / 0.012) ** 1.35) * l_scale * car_mult
            k_rad_car = (k_rad_car * k_seat_rad_ext) / max(k_rad_car + k_seat_rad_ext, 1e-9)
            if not is_dual:
                # Isolate local T-slot shoulder lip flex (excluding half-beam torsion already in FEM)
                k_seat_roll_ext = 24000.0 * ((w_rail_m / 0.012) ** 1.70) * l_scale
                k_rx_car = (k_rx_car * k_seat_roll_ext) / max(k_rx_car + k_seat_roll_ext, 1e-9)
            k_seat_pitch_ext = 82400.0 * ((w_rail_m / 0.012) ** 2.20) * l_scale * car_mult
            k_ry_car = (k_ry_car * k_seat_pitch_ext) / max(k_ry_car + k_seat_pitch_ext, 1e-9)
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
    th_cfg = resolve_toolhead_construction(spec)
    torsion_mult = float(th_cfg["torsion_mult"])
    bc = spec.toolhead.body_construction
    if bc is not None and bc in TOOLHEAD_CONSTRUCTION_CATALOG:
        k_body_n_m = (
            (float(th_cfg["body_stiffness_base_n_per_mm"]) * 1e3)
            * (body_ratio**0.72)
            * (r_body**1.35)
        )
    else:
        # C-04: Linear Stage 2 body/housing stiffness directly from internal_stiffness_n_per_mm (no < 350 power-law distortion)
        k_raw = max(float(th_cfg["body_stiffness_base_n_per_mm"]), 10.0)
        k_body_n_m = (k_raw * 1e3) * (body_ratio**0.55) * (r_body**1.35)
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
    clamp_cfg = resolve_hotend_clamp(spec)
    r_hm = float(spec.toolhead.hotend_mount_thickness_mm) / 8.0
    hm_ratio = max(e_hm / 2.1e9, 0.04)
    k_hotend_clamp_n_m = (
        (float(clamp_cfg["clamp_stiffness_base_n_per_mm"]) * 1e3)
        * (hm_ratio**0.45)
        * (r_hm**1.8)
    )

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
        "k_car_t": k_car_t,
        "k_car_rot_y": k_car_rot_y,
        "k_car_rot_z": k_car_rot_z,
        "k_car_rot_tor": k_car_rot_tor,
        "k_rad_car": k_rad_car,
        "k_rx_car": k_rx_car,
        "k_ry_car": k_ry_car,
        "k_rz_car": k_rz_car,
        "probe_comp": probe_comp,
    }


def _skew_3x3(vec: np.ndarray) -> np.ndarray:
    vx, vy, vz = float(vec[0]), float(vec[1]), float(vec[2])
    return np.array(
        [
            [0.0, -vz, vy],
            [vz, 0.0, -vx],
            [-vy, vx, 0.0],
        ],
        dtype=float,
    )


def _has_cad_asset(category: str, asset_id: str | None) -> bool:
    if not asset_id:
        return False
    try:
        from printsim.cad_assets.registry import has_hardware_asset
    except ImportError:
        return False
    return bool(has_hardware_asset(category, asset_id))


def _load_cad_superelement(
    category: str,
    asset_id: str | None,
    *,
    material_id: str | None = None,
    chamber_temp_c: float = 25.0,
    stiffness_mult: float = 1.0,
):
    if not asset_id:
        return None
    try:
        from printsim.cad_assets.registry import load_superelement
    except ImportError:
        return None
    return load_superelement(
        category,
        asset_id,
        material_id=material_id,
        chamber_temp_c=chamber_temp_c,
        stiffness_mult=stiffness_mult,
    )


def _invert_6x6_spd(mat: np.ndarray) -> np.ndarray:
    m_sym = 0.5 * (np.asarray(mat, dtype=float) + np.asarray(mat, dtype=float).T)
    diag_m = np.maximum(np.diag(m_sym), 1e-18)
    s = 1.0 / np.sqrt(diag_m)
    m_scaled = (s[:, None] * m_sym) * s[None, :]
    try:
        inv_scaled = np.linalg.inv(m_scaled)
    except np.linalg.LinAlgError:
        inv_scaled = np.linalg.pinv(m_scaled, rcond=1e-10)
    inv_m = (s[:, None] * inv_scaled) * s[None, :]
    return 0.5 * (inv_m + inv_m.T)


def _series_combine_superelement_kbb(
    K_bb: np.ndarray,
    C_series_6x6: np.ndarray,
    dp_asm_m: np.ndarray,
    c_cad_scale_diag: np.ndarray | None = None,
    dp_cad_m: np.ndarray | None = None,
    c_cad_pre_scale_diag: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine 12x12 CAD boundary stiffness in series with 6x6 mount/clamp compliance.

    When ``dp_cad_m`` is provided (e.g. a CAD toolhead whose Port 1 hotend pocket sits at a
    different offset from Port 0 than the global assembly's ``toolhead_cog`` node), rigidly
    shifts ``C_cad`` from ``dp_cad_m`` to ``dp_asm_m`` via ``T_shift @ C_cad @ T_shift.T`` so
    lateral-to-pitch cross-coupling has the exact physical sign and shear center at ``dp_asm_m``.
    Optional ``c_cad_pre_scale_diag`` applies a congruence compliance scaling on ``C_cad``
    before shifting (e.g., dual-point braced pitch/roll stiffening across a boxed chassis).

    Returns ``(K_12_eff, K_11_eff, C_eff)`` with exact 6 rigid-body zero modes at ``dp_asm_m``.
    """
    K_11 = 0.5 * (K_bb[6:12, 6:12] + K_bb[6:12, 6:12].T)
    C_cad = _invert_6x6_spd(K_11)
    if c_cad_pre_scale_diag is not None:
        s_pre = np.sqrt(np.asarray(c_cad_pre_scale_diag, dtype=float).reshape(6))
        C_cad = (s_pre[:, None] * C_cad) * s_pre[None, :]
        C_cad = 0.5 * (C_cad + C_cad.T)
    if dp_cad_m is not None:
        dr_shift = np.asarray(dp_asm_m, dtype=float) - np.asarray(dp_cad_m, dtype=float)
        if float(np.linalg.norm(dr_shift)) > 1e-9:
            T_shift = np.eye(6, dtype=float)
            T_shift[:3, 3:6] = -_skew_3x3(dr_shift)
            C_cad = T_shift @ C_cad @ T_shift.T
            C_cad = 0.5 * (C_cad + C_cad.T)
    if c_cad_scale_diag is not None:
        s_vec = np.sqrt(np.asarray(c_cad_scale_diag, dtype=float).reshape(6))
        C_cad = (s_vec[:, None] * C_cad) * s_vec[None, :]
        C_cad = 0.5 * (C_cad + C_cad.T)
    C_eff = 0.5 * ((C_cad + C_series_6x6) + (C_cad + C_series_6x6).T)
    K_11_eff = _invert_6x6_spd(C_eff)

    T_01 = np.eye(6, dtype=float)
    T_01[:3, 3:6] = -_skew_3x3(dp_asm_m)
    B = np.hstack([-T_01, np.eye(6, dtype=float)])
    K_12_eff = B.T @ K_11_eff @ B
    K_12_eff = 0.5 * (K_12_eff + K_12_eff.T)
    return K_12_eff, K_11_eff, C_eff


# Pre-exact-cutout catalog defaults so legacy YAMLs load unscaled CAD hotend superelements
_LEGACY_PRE_FEA_HOTEND_STIFFNESS: dict[str, tuple[float, ...]] = {
    "tricorn_long": (1100.0,),
    "tricorn_short": (1650.0,),
    "chube_air": (2600.0, 2250.0),
    "chube_compact": (2200.0, 2600.0),
    "chube_conduction": (3600.0, 5670.0),
}


_TOOLHEAD_CAD_ASSET_ALIASES: dict[str, str] = {
    "stealthburner_tap": "stealthburner",
}


def resolve_toolhead_cad_asset_id(preset_id: str | None) -> str:
    """Resolve toolhead preset ID to its baked CAD asset ID (e.g., stealthburner_tap -> stealthburner)."""
    pid = (preset_id or "").lower().strip()
    return _TOOLHEAD_CAD_ASSET_ALIASES.get(pid, pid)


def _inject_cad_toolhead_superelements(model: AssemblyModel) -> None:
    """Replace heuristic toolhead/hotend links with baked CAD superelements when enabled & available."""
    from printsim.geometry import builder as _b

    spec = model.spec
    if not getattr(spec, "use_cad_superelements", True):
        return

    nmap = model.node_map
    base_id = model.toolhead_node_id
    cog_id = model.toolhead_cog_node_id
    noz_id = model.nozzle_node_id
    if base_id not in nmap or cog_id not in nmap or noz_id not in nmap:
        return

    # For top-heavy direct-drive toolheads (cog_z > 0, noz_z < 0), _attach_toolhead_cog_and_nozzle
    # introduces a rail-height T-branch junction node ("toolhead_chassis") so the hotend hangs
    # from the lower chassis rather than from the elevated extruder CoG.
    junction_id = "toolhead_chassis" if "toolhead_chassis" in nmap else cog_id
    p_base = nmap[base_id].position_m
    p_junc = nmap[junction_id].position_m
    p_noz = nmap[noz_id].position_m
    l_mount_m = 0.0 if junction_id == base_id else float(np.linalg.norm(p_junc - p_base))
    l_stem_m = 0.0 if noz_id == junction_id else float(np.linalg.norm(p_noz - p_junc))
    props = compute_toolhead_link_props(spec, l_mount_m=l_mount_m, l_stem_m=l_stem_m)
    chamber_c = float(spec.frame.chamber_temp_c)
    updated_stage = False
    asset_th = None

    # 1. Self-contained Toolhead CAD Superelement (includes backplate + body + hotend mount;
    #    replaces heuristic toolhead_mount beam and places ONLY Hertzian MGN rail carriage contact in series)
    th_pid = (spec.toolhead.preset_id or "").lower().strip()
    th_asset_id = resolve_toolhead_cad_asset_id(th_pid)
    he_pid = (spec.toolhead.hotend.preset_id or "").lower().strip()
    combo_id = f"{th_asset_id}__{he_pid}" if (th_asset_id and he_pid) else ""
    use_bottom_brace = bool(getattr(spec.toolhead, "use_hotend_bottom_brace", True))
    will_use_braced_combo = bool(
        combo_id
        and use_bottom_brace
        and noz_id != junction_id
        and any(m.id == "nozzle_stem" for m in model.members)
        and _has_cad_asset("assemblies", combo_id)
    )
    if (
        th_asset_id
        and junction_id != base_id
        and any(m.id == "toolhead_mount" for m in model.members)
    ):
        # Guard: ToolheadSpec() defaults preset_id="stealthburner" even for legacy/synthetic
        # configs with cog_z <= 0 (e.g. generic -25 mm CoG or heavy UHF overrides). Real
        # StealthBurner + Clockwork 2 direct-drive assemblies have their extruder CoG above
        # the X-rail (cog_z > 0), so only attach the StealthBurner CAD SE when cog_z > 0.
        th_geom_matches_preset = not (
            th_asset_id == "stealthburner" and float(spec.toolhead.cog_offset_mm[2]) <= 0.0
        )

        probe_comp = float(props.get("probe_comp", 1.0))
        tc_ov = spec.model_overrides.toolhead_construction
        bc_mult = 1.0
        torsion_ratio = 1.0
        if tc_ov.body_stiffness_base_n_per_mm is not None or tc_ov.torsion_mult is not None:
            tc_cfg = resolve_toolhead_construction(spec)
            if tc_ov.body_stiffness_base_n_per_mm is not None:
                bc_mult = float(tc_ov.body_stiffness_base_n_per_mm) / max(
                    float(tc_cfg["baseline_body_stiffness_base_n_per_mm"]), 1e-6
                )
            if tc_ov.torsion_mult is not None:
                torsion_ratio = float(tc_ov.torsion_mult) / max(
                    float(tc_cfg["baseline_torsion_mult"]), 1e-6
                )

        th_stiff_mult = max(bc_mult * probe_comp, 1e-4)
        if th_geom_matches_preset:
            asset_th = _load_cad_superelement(
                "toolheads",
                th_asset_id,
                material_id=spec.toolhead.effective_body_material,
                chamber_temp_c=chamber_c,
                stiffness_mult=th_stiff_mult,
            )
        if asset_th is not None:
            # Multi-piece FDM-printed toolhead assemblies (e.g. Voron StealthBurner + CW2.5)
            # are voxel-bonded from separate bolted STEP halves:
            # - Printed polymers (< 25 GPa): 0.55x (40%-gyroid infill + layer-line anisotropy)
            #   * 0.58x (multi-piece M3 heat-set clamshell seam compliance) = 0.32x.
            # - CNC/SLM metals (>= 25 GPa): drop the FDM infill penalty but retain the 0.58x
            #   multi-piece bolted clamshell seam compliance (not a welded one-piece monoblock).
            eff_body_mat = get_material(spec.toolhead.effective_body_material)
            # Bolted multi-plate CNC cages baked as one fused solid (VzBot Vz-Printhead CNC) keep
            # the same 0.58x bolted-seam compliance even though they were not FDM-printed.
            if bool(asset_th.manifest.get("is_fdm_anisotropic", False)) or bool(
                asset_th.manifest.get("multi_piece_bolted", False)
            ):
                clamshell_mult = (
                    0.32 if float(eff_body_mat.youngs_modulus_pa) < 25e9 else 0.58
                )
                asset_th.K_bb = asset_th.K_bb * clamshell_mult
                asset_th.K_cb = asset_th.K_cb * clamshell_mult
                asset_th.stiffness_scale_applied = (
                    float(getattr(asset_th, "stiffness_scale_applied", 1.0)) * clamshell_mult
                )
            dp_asm_th = p_junc - p_base
            # Only Hertzian MGN linear rail carriage ball-raceway compliance in series
            # (do NOT double-count generic Stage 1 backplate compliance on self-contained CAD toolheads).
            C_rail_0 = np.diag(
                [
                    1.0 / max(float(props["k_rad_car"]), 1.0),
                    1.0 / max(float(props["k_rad_car"]), 1.0),
                    1.0 / max(float(props["k_rad_car"]), 1.0),
                    1.0 / max(float(props["k_rx_car"]), 1.0),
                    1.0 / max(0.5 * (float(props["k_ry_car"]) + float(props["k_rz_car"])), 1.0),
                    1.0 / max(float(props["k_rz_car"]), 1.0),
                ]
            )
            T_01_th = np.eye(6, dtype=float)
            T_01_th[:3, 3:6] = -_skew_3x3(dp_asm_th)
            C_rail_1 = T_01_th @ C_rail_0 @ T_01_th.T

            c_scale = (
                np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0 / max(torsion_ratio, 1e-3)], dtype=float)
                if abs(torsion_ratio - 1.0) > 1e-6
                else None
            )
            dp_cad_th = (
                asset_th.port_coords_m[1] - asset_th.port_coords_m[0]
                if asset_th.port_coords_m is not None and len(asset_th.port_coords_m) >= 2
                else None
            )
            # When a self-contained CAD toolhead's Port 1 sits high above the rail/CoG
            # (e.g., Microwaved hotend top ring at Z = +30.5 mm vs CoG at -1.6 mm), Port 1's
            # local rotational compliance reflects the thin hotend-pocket collar lip (already
            # accounted for in Stage 3 hotend_clamp), not the bulk monoblock chassis stiffness.
            # Keep bulk toolhead_mount stiffness invariant to hotend brace toggles so removing
            # a hotend brace increases hotend whip rather than softening the whole toolhead body!
            has_elevated_hotend_port = (
                dp_cad_th is not None and float(dp_cad_th[2] - dp_asm_th[2]) > 0.015
            )
            c_pre_scale = (
                np.array([1.0, 1.0, 1.0, 1.0 / 3.2, 1.0 / 3.2, 1.0], dtype=float)
                if has_elevated_hotend_port
                else None
            )
            K_12_eff_th, _, C_eff_th = _series_combine_superelement_kbb(
                asset_th.K_bb,
                C_rail_1,
                dp_asm_th,
                c_cad_scale_diag=c_scale,
                dp_cad_m=dp_cad_th,
                c_cad_pre_scale_diag=c_pre_scale,
            )
            C_body_only = 0.5 * ((C_eff_th - C_rail_1) + (C_eff_th - C_rail_1).T)
            dr_cog_noz = p_noz - p_junc
            P_noz = np.hstack([np.eye(3, dtype=float), -_skew_3x3(dr_cog_noz)])
            C_body_noz = P_noz @ C_body_only @ P_noz.T
            k_body_x = 1.0 / max(float(C_body_noz[0, 0]), 1e-12)
            k_body_y = 1.0 / max(float(C_body_noz[1, 1]), 1e-12)
            k_body_n_m = 2.0 / (1.0 / k_body_x + 1.0 / k_body_y)
            model.toolhead_stage_stiffnesses_n_per_mm["carriage_mount_n_per_mm"] = float(
                props["k_rail_carriage_n_per_mm"]
            )
            model.toolhead_stage_stiffnesses_n_per_mm["backplate_only_n_per_mm"] = 0.0
            model.toolhead_stage_stiffnesses_n_per_mm["body_n_per_mm"] = k_body_n_m * 1e-3
            updated_stage = True

            model.members = [m for m in model.members if m.id != "toolhead_mount"]
            th_mass_scale = float(getattr(asset_th, "mass_scale_applied", 1.0))
            th_stiff_scale = float(getattr(asset_th, "stiffness_scale_applied", 1.0))
            th_mass_ref = (
                float(asset_th.manifest.get("mass_kg", float(spec.toolhead.total_mass_g) * 1e-3))
                * th_mass_scale
            )
            th_manifest = dict(asset_th.manifest)
            if not bool(th_manifest.get("is_multi_material", False)):
                if isinstance(th_manifest.get("port_stiffness_n_per_mm"), dict) and abs(th_stiff_scale - 1.0) > 1e-6:
                    th_manifest["port_stiffness_n_per_mm"] = {
                        k: round(float(v) * th_stiff_scale, 3)
                        for k, v in th_manifest["port_stiffness_n_per_mm"].items()
                    }
                if abs(th_mass_scale - 1.0) > 1e-6:
                    th_manifest["mass_kg"] = round(th_mass_ref, 6)
            th_manifest["_baseline_k11_trace"] = float(np.trace(K_12_eff_th[6:9, 6:9]))
            model.superelements.append(
                _b.AttachedSuperelement(
                    id="toolhead_mount",
                    category="toolheads",
                    asset_id=th_asset_id,
                    port_node_ids=(base_id, junction_id),
                    K_cb=K_12_eff_th,
                    M_cb=None,
                    port_coords_cad_m=None,
                    n_internal_modes=0,
                    part_group_id="toolhead_mount",
                    part_group_label="Toolhead & Nozzle Mount",
                    role="toolhead",
                    mass_kg=th_mass_ref,
                    fixed_interface_freqs_hz=list(asset_th.fixed_interface_freqs_hz),
                    manifest=th_manifest,
                    visual_glb_url=f"/api/assets/hardware/toolheads/{th_asset_id}/visual.glb",
                    cad_trust_pct=int(asset_th.manifest.get("cad_trust_pct", 90)),
                )
            )

    # 2. Hotend CAD Superelement (replaces heuristic nozzle_stem beam, keeps Stage 3 clamp in series;
    #    prefers a coupled toolhead+hotend assembly superelement when available for braced toolheads)
    if (
        he_pid
        and noz_id != junction_id
        and any(m.id == "nozzle_stem" for m in model.members)
    ):
        he = spec.toolhead.hotend
        cat_he = HOTEND_CATALOG.get(he_pid)
        he_stiff_mult = 1.0
        if cat_he is not None:
            cat_k = float(cat_he["structural_stiffness_n_per_mm"])
            he_k = float(he.structural_stiffness_n_per_mm)
            legacy_ks = _LEGACY_PRE_FEA_HOTEND_STIFFNESS.get(he_pid, ())
            is_catalog_default = (cat_k > 0.0 and abs(he_k - cat_k) / cat_k <= 1e-4) or any(
                abs(he_k - lk) / lk <= 1e-4 for lk in legacy_ks
            )
            if cat_k > 0.0 and not is_catalog_default:
                he_stiff_mult = max(he_k / cat_k, 1e-3)

        asset_he = None
        se_category = "hotends"
        se_asset_id = he_pid
        is_braced_combo = False
        if combo_id and use_bottom_brace:
            asset_he = _load_cad_superelement(
                "assemblies",
                combo_id,
                material_id=None,
                chamber_temp_c=chamber_c,
                stiffness_mult=he_stiff_mult,
            )
            if asset_he is not None:
                se_category = "assemblies"
                se_asset_id = combo_id
                is_braced_combo = True
        if asset_he is None:
            asset_he = _load_cad_superelement(
                "hotends",
                he_pid,
                material_id=None,
                chamber_temp_c=chamber_c,
                stiffness_mult=he_stiff_mult,
            )

        if asset_he is not None:
            dp_asm_he = p_noz - p_junc
            dp_cad_he = asset_he.port_coords_m[1] - asset_he.port_coords_m[0]
            l_eff_he = max(
                float(np.linalg.norm(dp_cad_he)),
                float(np.linalg.norm(dp_asm_he)),
                0.040,
            )
            k_clamp_t = max(float(props["k_hotend_clamp_n_per_mm"]) * 1e3, 1.0)
            if (
                asset_th is not None
                and not bool(asset_th.manifest.get("is_multi_material", False))
                and spec.toolhead.hotend_mount_material is None
                and spec.toolhead.body_material is not None
            ):
                hm_mat_eff = get_material(spec.toolhead.effective_body_material)
                hm_ratio_eff = max(
                    (hm_mat_eff.effective_modulus_pa(chamber_c) * float(props.get("probe_comp", 1.0)))
                    / 2.1e9,
                    0.04,
                )
                clamp_cfg = resolve_hotend_clamp(spec)
                r_hm = float(spec.toolhead.hotend_mount_thickness_mm) / 8.0
                k_clamp_t = max(
                    (float(clamp_cfg["clamp_stiffness_base_n_per_mm"]) * 1e3)
                    * (hm_ratio_eff**0.45)
                    * (r_hm**1.8),
                    1.0,
                )
                model.toolhead_stage_stiffnesses_n_per_mm["hotend_clamp_n_per_mm"] = (
                    k_clamp_t * 1e-3
                )
            if is_braced_combo:
                # Dual-point top 4xM3 crown + lower 3xM3 Ti brace parallel clamping into metal/SLM body
                body_mat_eff = get_material(spec.toolhead.effective_body_material)
                e_body_eff = float(body_mat_eff.effective_modulus_pa(chamber_c))
                is_metal_toolhead = (
                    not bool(body_mat_eff.is_printed_polymer)
                    and e_body_eff >= 40.0e9
                )
                if (
                    is_metal_toolhead
                    and spec.model_overrides.hotend_clamp.clamp_stiffness_base_n_per_mm is None
                ):
                    e_al6061 = float(get_material("aluminum_6061_t6").youngs_modulus_pa)
                    k_clamp_eff_n_mm = max(k_clamp_t * 1e-3, 8000.0) * (
                        e_body_eff / max(e_al6061, 1.0)
                    )
                    k_clamp_t = max(k_clamp_eff_n_mm * 1e3, 1.0)
                    model.toolhead_stage_stiffnesses_n_per_mm["hotend_clamp_n_per_mm"] = (
                        k_clamp_t * 1e-3
                    )
            k_clamp_r = max(k_clamp_t * (l_eff_he**2), 50.0)
            C_clamp = np.diag(
                [
                    1.0 / k_clamp_t,
                    1.0 / k_clamp_t,
                    1.0 / (4.0 * k_clamp_t),
                    1.0 / k_clamp_r,
                    1.0 / k_clamp_r,
                    1.0 / k_clamp_r,
                ]
            )
            K_12_eff_he, _, C_eff_he = _series_combine_superelement_kbb(
                asset_he.K_bb, C_clamp, dp_asm_he
            )
            k_he_x = 1.0 / max(float(C_eff_he[0, 0]), 1e-12)
            k_he_y = 1.0 / max(float(C_eff_he[1, 1]), 1e-12)
            k_he_tot_n_mm = (2.0 / (1.0 / k_he_x + 1.0 / k_he_y)) * 1e-3
            model.toolhead_stage_stiffnesses_n_per_mm["hotend_total_n_per_mm"] = k_he_tot_n_mm
            updated_stage = True

            model.members = [m for m in model.members if m.id != "nozzle_stem"]
            he_mass_scale = float(getattr(asset_he, "mass_scale_applied", 1.0))
            he_mass_ref = (
                float(asset_he.manifest.get("mass_kg", float(he.mass_g) * 1e-3))
                * he_mass_scale
            )
            he_manifest = dict(asset_he.manifest)
            he_eff_scale = float(getattr(asset_he, "stiffness_scale_applied", he_stiff_mult))
            if (
                isinstance(he_manifest.get("port_stiffness_n_per_mm"), dict)
                and abs(he_eff_scale - 1.0) > 1e-6
            ):
                he_manifest["port_stiffness_n_per_mm"] = {
                    k: round(float(v) * he_eff_scale, 3)
                    for k, v in he_manifest["port_stiffness_n_per_mm"].items()
                }
            se_visual_url = (
                f"/api/assets/hardware/{se_category}/{se_asset_id}/visual.glb"
                if asset_he.visual_glb_path is not None
                else f"/api/assets/hardware/hotends/{he_pid}/visual.glb"
            )
            model.superelements.append(
                _b.AttachedSuperelement(
                    id="nozzle_stem",
                    category=se_category,
                    asset_id=se_asset_id,
                    port_node_ids=(junction_id, noz_id),
                    K_cb=K_12_eff_he,
                    M_cb=None,
                    port_coords_cad_m=None,
                    n_internal_modes=0,
                    part_group_id="toolhead_mount",
                    part_group_label="Toolhead & Nozzle Mount",
                    role="toolhead",
                    mass_kg=he_mass_ref,
                    fixed_interface_freqs_hz=list(asset_he.fixed_interface_freqs_hz),
                    manifest=he_manifest,
                    visual_glb_url=se_visual_url,
                    cad_trust_pct=int(asset_he.manifest.get("cad_trust_pct", 92)),
                )
            )

    if updated_stage:
        st_map = model.toolhead_stage_stiffnesses_n_per_mm
        st_map["combined_toolhead_n_per_mm"] = 1.0 / (
            1.0 / max(float(st_map.get("carriage_mount_n_per_mm", 1e9)), 1e-6)
            + 1.0 / max(float(st_map.get("body_n_per_mm", 1e9)), 1e-6)
            + 1.0 / max(float(st_map.get("hotend_total_n_per_mm", 1e9)), 1e-6)
        )


def _attach_toolhead_cog_and_nozzle(
    nodes: list[StructuralNode],
    members: list[StructuralMember],
    spec: PrinterSpec,
    base_node_id: str = "toolhead",
    fastener_springs: list[FastenerSpring] | None = None,
) -> dict[str, float]:
    """Attach explicit 6-DOF toolhead_cog and nozzle_tip nodes via carriage mount & hotend stem."""
    from printsim.geometry import builder as _b

    nmap = {n.id: n for n in nodes}
    if base_node_id not in nmap:
        return {}

    base_node = nmap[base_node_id]
    base_pos = base_node.position_m
    acc = resolve_toolhead_hardware_accessories(spec)
    block_total_g, block_delta_g = _b.x_carriage_block_mass_delta_g(spec)
    toolhead_kg = max((float(acc["effective_total_mass_g"]) + block_delta_g) * 1e-3, 0.03)
    hotend_kg = float(acc["effective_hotend_mass_g"]) * 1e-3
    probe_cfg = resolve_z_probe(spec)
    cog_y_shift_m = float(probe_cfg["cog_y_shift_mm"]) * 1e-3

    # Linear carriage block stays on X-beam centerline; Rayleigh-equivalent lower cantilevered hotend mass (~28% of hotend mass) sits at nozzle tip
    nozzle_lumped_kg = max(0.010, min(0.28 * hotend_kg, max(toolhead_kg - 0.06, 0.015)))
    desired_base_kg = 0.04 if spec.architecture == "delta" else (block_total_g * 1e-3)
    max_base_kg = max(toolhead_kg - nozzle_lumped_kg - 0.02, 0.005)
    base_lumped_kg = min(desired_base_kg, max_base_kg)
    base_node.lumped_mass_kg = base_lumped_kg
    cog_lumped_kg = max(toolhead_kg - base_lumped_kg - nozzle_lumped_kg, 0.02)

    cog_offset = np.array(acc["effective_cog_offset_mm"], dtype=float) * 1e-3
    nozzle_offset = np.array(spec.toolhead.nozzle_offset_mm, dtype=float) * 1e-3

    if spec.architecture != "delta":
        x_mount = spec.gantry.effective_x_rail_mount
        datum = getattr(spec.toolhead, "offset_datum_rail_mount", "front")
        if datum == "top":
            if x_mount == "top":
                x_mount = "datum"  # offsets already measured on a top-mounted rail: no shift
            else:
                # Undo the top-mount datum shift first, then apply the actual mount's shift
                for off in (cog_offset, nozzle_offset):
                    if off[1] < 0.0:
                        off[1] -= 0.006
                    elif off[1] > 0.0:
                        off[1] += 0.006
                    if abs(off[2]) > 1e-12:
                        off[2] += 0.006
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

    # For top-heavy direct-drive toolheads (CoG ABOVE rail Z while nozzle hangs BELOW rail Z),
    # a 2-beam series chain (base -> +Z cog -> -Z nozzle) forms an unphysical C-shaped hairpin
    # that stretches nozzle_stem by +cog_z and levers upper-extruder flex backwards onto the nozzle.
    # Insert a rail-height T-branch junction ("toolhead_chassis") so toolhead_mount carries rail->chassis,
    # nozzle_stem hangs from rail-level chassis->nozzle, and toolhead_mount_upper carries chassis->cog.
    use_t_branch = (
        not is_zero_cog
        and float(cog_pos[2] - base_pos[2]) > 1e-4
        and float(nozzle_pos[2] - base_pos[2]) < -1e-4
    )
    if use_t_branch:
        chassis_pos = np.array([cog_pos[0], cog_pos[1], base_pos[2]], dtype=float)
        chassis_xy_norm = float(np.linalg.norm(chassis_pos - base_pos))
        is_zero_mount = chassis_xy_norm < 1e-4
        if is_zero_mount:
            chassis_pos = base_pos.copy()
        junc_id = "toolhead_chassis"
        junc_pos = chassis_pos
        nodes.append(
            _b.StructuralNode(
                id=junc_id,
                position_m=chassis_pos,
                lumped_mass_kg=0.0,
                role="toolhead",
            )
        )
    else:
        is_zero_mount = is_zero_cog
        junc_id = "toolhead_cog"
        junc_pos = cog_pos

    stem_vec = nozzle_pos - junc_pos
    stem_len = float(np.linalg.norm(stem_vec))
    is_zero_stem = stem_len < 1e-4
    if is_zero_stem:
        nozzle_pos = junc_pos.copy()

    nodes.append(
        _b.StructuralNode(
            id="toolhead_cog",
            position_m=cog_pos,
            lumped_mass_kg=cog_lumped_kg,
            role="cog",
        )
    )
    nodes.append(
        _b.StructuralNode(
            id="nozzle_tip",
            position_m=nozzle_pos,
            lumped_mass_kg=nozzle_lumped_kg,
            role="nozzle",
        )
    )

    l_mount_m = 0.0 if is_zero_mount else float(np.linalg.norm(junc_pos - base_pos))
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

    if is_zero_mount and fastener_springs is not None:
        fastener_springs.append(
            _b.FastenerSpring(
                id="toolhead_cog_spring",
                bay="toolhead",
                shell_node_id=base_node_id,
                frame_node_id=junc_id,
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
            _b.StructuralMember(
                id="toolhead_mount",
                start_id=base_node_id,
                end_id=junc_id,
                profile=mount_prof,
                material=mount_mat,
                role="toolhead",
            )
        )

    if use_t_branch:
        upper_stiff_mult = 4.0
        upper_prof = ProfileSpec.model_construct(
            id="toolhead_mount_profile",
            name=(
                f"Toolhead Upper Extruder Arm ({spec.toolhead.effective_body_material})"
            ),
            shape="solid_bar",
            width_m=0.026,
            height_m=0.018,
            area_m2=props["area_mount"],
            iy_m4=props["iy_mount"] * upper_stiff_mult,
            iz_m4=props["iz_mount"] * upper_stiff_mult,
            j_m4=props["j_mount"] * upper_stiff_mult,
        )
        members.append(
            _b.StructuralMember(
                id="toolhead_mount_upper",
                start_id=junc_id,
                end_id="toolhead_cog",
                profile=upper_prof,
                material=mount_mat,
                role="toolhead",
            )
        )

    if is_zero_stem and fastener_springs is not None:
        fastener_springs.append(
            _b.FastenerSpring(
                id="nozzle_stem_spring",
                bay="toolhead",
                shell_node_id=junc_id,
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
            _b.StructuralMember(
                id="nozzle_stem",
                start_id=junc_id,
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
