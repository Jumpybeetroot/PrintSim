"""Per-part toolhead flex, compliance, stress, and modal participation breakdown (strict SI internals).

Supports both:
- **Option 1 (Stage FEA)**: Universal breakdown across every physical toolhead stage
  (MGN linear rail carriage, X-carriage backplate, toolhead chassis, upper extruder/motor CoG,
  hotend mount collar, optional Ti bottom brace, hotend heatbreak/heatsink, heater block & nozzle,
  and non-structural cooling shroud/accessories). Works on every printer & in Pyodide WASM.
- **Option 2 (3D CAD Craig-Bampton Recovery)**: When a baked CAD superelement bundle includes
  ``subpart_recovery`` operators (``subpart_K_bb``, ``subpart_M_bb``, ``subpart_disp_op``,
  ``subpart_stress_op``), evaluates exact 3D Tet4 per-subpart strain energy, kinetic energy,
  local elastic flex, and peak von Mises stress directly from the solved global boundary DOFs.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from printsim.geometry._builder_toolhead import resolve_toolhead_hardware_accessories
from printsim.geometry.builder import AssemblyModel
from printsim.schema import get_material


def _get_node_dof6(u_vec: np.ndarray, node_idx: dict[str, int], nid: str) -> np.ndarray:
    idx = node_idx.get(nid)
    if idx is None or 6 * idx + 6 > len(u_vec):
        return np.zeros(6, dtype=np.float64)
    return np.asarray(u_vec[6 * idx : 6 * idx + 6], dtype=np.float64)


def _node_pos(assembly: AssemblyModel, node_idx: dict[str, int], nid: str) -> np.ndarray:
    idx = node_idx.get(nid)
    if idx is None:
        return np.zeros(3, dtype=np.float64)
    return np.asarray(assembly.nodes[idx].position_m, dtype=np.float64)


def _rel_elastic_flex_m(
    u0: np.ndarray, p0: np.ndarray, u1: np.ndarray, p1: np.ndarray
) -> tuple[float, float]:
    """Return ``(elastic_disp_m, elastic_rot_rad)`` of node 1 relative to node 0 rigid-body motion."""
    dp = p1 - p0
    du = u1[:3] - (u0[:3] + np.cross(u0[3:6], dp))
    dth = u1[3:6] - u0[3:6]
    return float(np.linalg.norm(du)), float(np.linalg.norm(dth))


def _find_superelement(assembly: AssemblyModel, se_id: str) -> Any | None:
    for se in getattr(assembly, "superelements", ()):
        if getattr(se, "id", None) == se_id:
            return se
    return None


def _short_mat_name(name: str) -> str:
    """Compact human-readable material name without nested parentheticals."""
    s = name.split("(")[0].strip()
    s = s.replace("Aluminum ", "Al ").replace("Titanium ", "Ti ")
    return s or name


def _pretty_hotend_mount_style(raw_style: str) -> str:
    known = {
        "chube_6bolt_m3_18pcd": "Chube 6-Bolt M3",
        "rigid_flange": "Rigid 4-Bolt Flange",
        "rigid_flange_4bolt": "Rigid 4-Bolt Flange",
        "groove_mount": "V6 Groove Mount",
        "threaded_m12": "Threaded M12 Collar",
        "dragon_4bolt": "Dragon 4-Bolt",
        "rapido_4bolt": "Rapido 4-Bolt",
    }
    key = str(raw_style or "rigid_flange").strip().lower()
    return known.get(key, key.replace("_", " ").title())


def _is_chube_hotend(hotend_spec: Any) -> bool:
    pid = str(getattr(hotend_spec, "preset_id", "") or "").lower()
    nm = str(getattr(hotend_spec, "name", "") or "").lower()
    return "chube" in pid or "chube" in nm


def _multi_piece_chassis_labels(spec_th: Any) -> tuple[tuple[str, str, str], tuple[str, str, str], tuple[str, str, str]]:
    """Return architecture-accurate (id, label, stage) tuples for the 3 multi-piece chassis sub-zones."""
    pid = str(getattr(spec_th, "preset_id", "") or "").lower()
    if "archetype" in pid:
        return (
            ("chassis_carrier", "Archetype Hotend Core & MGN12 Mount", "chassis_carrier"),
            ("chassis_clamshell", "Breakneck Core Bridge & Duct Mount", "chassis_clamshell"),
            ("chassis_extruder", "Sherpa Micro Extruder Block", "chassis_extruder"),
        )
    return (
        ("chassis_carrier", "Hotend Carrier (Lower Body)", "chassis_carrier"),
        ("chassis_clamshell", "Front Clamshell & Motor Plate", "chassis_clamshell"),
        ("chassis_extruder", "Extruder Housing (Upper Body)", "chassis_extruder"),
    )


def _pretty_subpart_label(raw_name: str, fallback_prefix: str = "") -> str:
    known = {
        "carriage_mount_zone": "Carriage Mount Zone (CAD 3D)",
        "main_chassis": "Main Chassis Body (CAD 3D)",
        "hotend_mount_zone": "Hotend Mount Zone (CAD 3D)",
        "upper_extruder_zone": "Upper Extruder / Motor Zone (CAD 3D)",
    }
    key = raw_name.strip().lower()
    if key in known:
        return known[key]
    nice = raw_name.replace("_", " ").strip().title()
    return f"{fallback_prefix}{nice}" if fallback_prefix else nice


def compute_static_toolhead_parts(
    assembly: AssemblyModel,
    node_idx: dict[str, int],
    u_peak_vec: np.ndarray,
    toolhead_flex_breakdown: dict[str, Any],
    member_stress_map: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Compute per-part static FEA flex, compliance share, stiffness, mass, and stress."""
    spec = assembly.spec
    spec_th = spec.toolhead
    spec_gan = spec.gantry
    is_delta = spec.architecture == "delta"

    th_id = assembly.toolhead_node_id
    cog_id = assembly.toolhead_cog_node_id if assembly.toolhead_cog_node_id in node_idx else th_id
    noz_id = assembly.nozzle_node_id if assembly.nozzle_node_id in node_idx else cog_id
    junc_id = "toolhead_chassis" if "toolhead_chassis" in node_idx else cog_id
    has_t_branch = junc_id == "toolhead_chassis" and cog_id != junc_id

    p_th = _node_pos(assembly, node_idx, th_id)
    p_junc = _node_pos(assembly, node_idx, junc_id)
    p_cog = _node_pos(assembly, node_idx, cog_id)
    p_noz = _node_pos(assembly, node_idx, noz_id)

    u_th = _get_node_dof6(u_peak_vec, node_idx, th_id)
    u_junc = _get_node_dof6(u_peak_vec, node_idx, junc_id)
    u_cog = _get_node_dof6(u_peak_vec, node_idx, cog_id)
    u_noz = _get_node_dof6(u_peak_vec, node_idx, noz_id)

    flex_12_m, twist_12_rad = _rel_elastic_flex_m(u_th, p_th, u_junc, p_junc)
    flex_up_m, twist_up_rad = (
        _rel_elastic_flex_m(u_junc, p_junc, u_cog, p_cog) if has_t_branch else (0.0, 0.0)
    )
    flex_3_m, twist_3_rad = _rel_elastic_flex_m(u_junc, p_junc, u_noz, p_noz)

    disp_th_um = float(np.linalg.norm(u_th[:3])) * 1e6
    disp_junc_um = float(np.linalg.norm(u_junc[:3])) * 1e6
    disp_cog_um = float(np.linalg.norm(u_cog[:3])) * 1e6
    disp_noz_um = float(np.linalg.norm(u_noz[:3])) * 1e6

    stage_stiff = getattr(assembly, "toolhead_stage_stiffnesses_n_per_mm", {}) or {}
    k_rail = float(stage_stiff.get("rail_carriage_n_per_mm", stage_stiff.get("carriage_mount_n_per_mm", 0.0)))
    k_bp = float(stage_stiff.get("backplate_only_n_per_mm", 0.0))
    k_body = float(stage_stiff.get("body_n_per_mm", 0.0))
    k_clamp = float(stage_stiff.get("hotend_clamp_n_per_mm", 0.0))
    k_hotend_tot = float(stage_stiff.get("hotend_total_n_per_mm", spec_th.hotend.structural_stiffness_n_per_mm))

    # Separate hotend-only stiffness from clamp stiffness in series
    c_he_tot = (1.0 / k_hotend_tot) if k_hotend_tot > 1e-9 else 0.0
    c_clamp = (1.0 / k_clamp) if k_clamp > 1e-9 else 0.0
    c_clamp_eff = min(c_clamp, c_he_tot * 0.85) if c_he_tot > 1e-12 else c_clamp
    c_he_only = max(c_he_tot - c_clamp_eff, c_he_tot * 0.15) if c_he_tot > 1e-12 else 0.0
    k_he_only = (1.0 / c_he_only) if c_he_only > 1e-12 else k_hotend_tot

    c_rail = (1.0 / k_rail) if k_rail > 1e-9 else 0.0
    c_bp = (1.0 / k_bp) if k_bp > 1e-9 else 0.0
    c_body = (1.0 / k_body) if k_body > 1e-9 else 0.0
    c_12_sum = max(c_rail + c_bp + c_body, 1e-15)

    frac_rail = c_rail / c_12_sum if c_rail > 0 else 0.0
    frac_bp = c_bp / c_12_sum if c_bp > 0 else 0.0
    frac_body = max(0.0, 1.0 - frac_rail - frac_bp)

    c_3_sum = max(c_clamp_eff + c_he_only, 1e-15)
    frac_clamp = c_clamp_eff / c_3_sum if c_clamp_eff > 0 else 0.0
    frac_he_only = max(0.0, 1.0 - frac_clamp)

    pct_car_body = float(toolhead_flex_breakdown.get("carriage_and_body_pct", 0.0))
    pct_hotend = float(toolhead_flex_breakdown.get("hotend_and_nozzle_pct", 0.0))
    pct_th_total = max(pct_car_body + pct_hotend, 1e-6)

    acc = resolve_toolhead_hardware_accessories(spec)
    total_body_g = float(acc["effective_body_mass_g"])
    total_hotend_g = float(acc["effective_hotend_mass_g"])
    is_braced = bool(acc.get("use_hotend_bottom_brace", False))

    # Realistic sub-mass budget breakdown (sums cleanly to assembled toolhead mass)
    m_rail_car_g = 0.0 if is_delta else (26.0 if "mgn9" in spec_gan.x_rail_profile else 54.0)
    m_rail_car_g = min(m_rail_car_g, total_body_g * 0.45)
    rem_body_g = max(total_body_g - m_rail_car_g, 5.0)
    m_bp_g = round(rem_body_g * 0.16, 1) if k_bp > 1e-9 else 0.0
    ext_type = str(getattr(spec_th.extruder, "type", "direct_drive") or "direct_drive")
    ext_name = str(getattr(spec_th.extruder, "name", "Direct Drive") or "Direct Drive")
    is_direct_drive = ext_type == "direct_drive"
    m_ext_g = round(max(rem_body_g - m_bp_g - 15.0, 0.0) * 0.48, 1) if is_direct_drive else 0.0
    m_shroud_g = round(max(rem_body_g - m_bp_g - m_ext_g, 0.0) * 0.25, 1)
    m_chassis_g = round(max(rem_body_g - m_bp_g - m_ext_g - m_shroud_g, 5.0), 1)
    m_heater_noz_g = round(
        min(max(float(getattr(spec_th.hotend, "block_mass_g", 14.0)) + 2.5, 6.0), total_hotend_g * 0.65),
        1,
    )
    m_he_neck_g = round(max(total_hotend_g - m_heater_noz_g, 4.0), 1)

    ms_map = member_stress_map or {}
    mount_ms = ms_map.get("toolhead_mount")
    upper_ms = ms_map.get("toolhead_mount_upper")
    noz_ms = ms_map.get("nozzle_stem")

    car_mat = get_material(spec_th.effective_carriage_mount_material)
    body_mat = get_material(spec_th.effective_body_material)
    hm_mat = get_material(spec_th.effective_hotend_mount_material)
    hb_mat = get_material(spec_th.hotend.heatbreak_material)

    parts: list[dict[str, Any]] = []

    # 1. Linear Rail Carriage (Hertzian contact)
    rail_comp_pct = round(pct_car_body * frac_rail, 2)
    rail_short = f"{spec_gan.x_rail_profile.upper().replace('_RAIL', '')}{spec_gan.rail_block_type}"
    parts.append(
        {
            "id": "rail_carriage",
            "name": "Delta Effector Joint" if is_delta else f"{rail_short} Linear Carriage",
            "stage": "rail_carriage",
            "is_structural": True,
            "source": "stage_fea",
            "material": "Hardened Bearing Steel (Hertzian)" if not is_delta else "Delta Effector",
            "mass_g": round(m_rail_car_g, 1),
            "stiffness_n_per_mm": round(k_rail, 1) if k_rail > 1e-9 else None,
            "compliance_share_pct": rail_comp_pct,
            "toolhead_share_pct": round(rail_comp_pct / pct_th_total * 100.0, 1),
            "local_flex_um": round(flex_12_m * frac_rail * 1e6, 3),
            "total_disp_um": round(disp_th_um + flex_12_m * frac_rail * 1e6, 2),
            "twist_mrad": round(twist_12_rad * frac_rail * 1e3, 3),
            "stress_mpa": round(float(mount_ms.von_mises_mpa) * 0.45, 2) if mount_ms else None,
            "yield_mpa": 650.0,
            "why_it_hurts": (
                "Ball-raceway Hertzian roll compliance lets the entire toolhead pitch around the X-rail."
                if rail_comp_pct >= 10.0
                else "Linear carriage ball-raceway contact is stiff relative to the rest of the loop."
            ),
            "action_hint": (
                "Increase rail preload (ZA/ZB), switch to an H-block, or use dual X rails."
                if rail_comp_pct >= 10.0
                else "Carriage preload and block size are already well-matched."
            ),
        }
    )

    # 2. X-Carriage Backplate / CNC X-Carriage Mount
    if k_bp > 1e-9:
        bp_comp_pct = round(pct_car_body * frac_bp, 2)
        is_cnc_cad_car = (
            spec.architecture in ("corexy", "cartesian")
            and spec.gantry.effective_x_rail_mount == "front"
            and not bool(car_mat.is_printed_polymer)
        )
        parts.append(
            {
                "id": "carriage_backplate",
                "name": (
                    f"CNC X-Carriage ({_short_mat_name(car_mat.name)})"
                    if is_cnc_cad_car
                    else f"X-Carriage Plate ({spec_th.carriage_plate_thickness_mm:.0f}mm)"
                ),
                "stage": "backplate",
                "is_structural": True,
                "source": "cad_3d_recovery" if is_cnc_cad_car else "stage_fea",
                "material": car_mat.name,
                "mass_g": 20.2 if is_cnc_cad_car else m_bp_g,
                "stiffness_n_per_mm": round(k_bp, 1),
                "compliance_share_pct": bp_comp_pct,
                "toolhead_share_pct": round(bp_comp_pct / pct_th_total * 100.0, 1),
                "local_flex_um": round(flex_12_m * frac_bp * 1e6, 3),
                "total_disp_um": round(disp_th_um + flex_12_m * (frac_rail + frac_bp) * 1e6, 2),
                "twist_mrad": round(twist_12_rad * frac_bp * 1e3, 3),
                "stress_mpa": round(float(mount_ms.von_mises_mpa), 2) if mount_ms else None,
                "yield_mpa": round(float(car_mat.yield_strength_pa) * 1e-6, 1),
                "why_it_hurts": (
                    "Bending at the 4× M3 carriage mounting plate amplifies nozzle deflection through the full Z drop."
                    if bp_comp_pct >= 8.0
                    else "Backplate bending is small compared to other compliance sources."
                ),
                "action_hint": (
                    "Upgrade carriage mount to CNC 6061-T6 aluminum or thicken the backplate."
                    if bp_comp_pct >= 8.0
                    else "Current backplate material and thickness are sufficient."
                ),
            }
        )

    # 3. Toolhead Chassis / Body — Option 2 (3D CAD CB Sub-Part Recovery) or Option 1 (Stage FEA)
    se_th = _find_superelement(assembly, "toolhead_mount")
    se_th_rec = getattr(se_th, "subpart_recovery", None) if se_th is not None else None
    body_comp_pct = round(pct_car_body * frac_body, 2)
    if se_th_rec is not None:
        q_b_th = np.concatenate([u_th, u_junc])
        recovered_th = se_th_rec.recover(q_b_th)
        for sp in recovered_th:
            sp_share_body = float(sp["strain_energy_pct"]) / 100.0
            sp_comp_pct = round(body_comp_pct * sp_share_body, 2)
            sp_mat = get_material(sp["material_id"]) if sp["material_id"] else body_mat
            sp_id = f"cad_th_{sp['name']}"
            parts.append(
                {
                    "id": sp_id,
                    "name": _pretty_subpart_label(sp["name"]),
                    "stage": "extruder_upper" if "extruder" in sp["name"].lower() else "chassis",
                    "is_structural": True,
                    "source": "cad_3d_recovery",
                    "material": sp_mat.name,
                    "mass_g": float(sp["mass_g"]),
                    "stiffness_n_per_mm": round(k_body / max(sp_share_body, 0.05), 1) if k_body > 1e-9 else None,
                    "compliance_share_pct": sp_comp_pct,
                    "toolhead_share_pct": round(sp_comp_pct / pct_th_total * 100.0, 1),
                    "local_flex_um": round(float(sp["local_flex_um"]), 3),
                    "total_disp_um": round(disp_junc_um, 2),
                    "twist_mrad": round(twist_12_rad * frac_body * sp_share_body * 1e3, 3),
                    "stress_mpa": round(float(sp["stress_mpa"]), 3),
                    "yield_mpa": round(float(sp_mat.yield_strength_pa) * 1e-6, 1),
                    "why_it_hurts": (
                        f"3D Tet4 recovery shows {sp['strain_energy_pct']:.0f}% of chassis strain energy concentrates here."
                        if sp["strain_energy_pct"] >= 30.0
                        else f"Carries {sp['strain_energy_pct']:.0f}% of the 3D CAD chassis strain energy."
                    ),
                    "action_hint": (
                        "Stiffen this zone via stiffer chassis material (CNC Al / SLM Ti / PPA-CF) or gusseting."
                        if sp_comp_pct >= 8.0
                        else "Local 3D chassis stress and flex in this zone are well controlled."
                    ),
                }
            )
    else:
        is_unibody = bool(
            se_th is not None and (getattr(se_th, "manifest", {}) or {}).get("is_unibody", False)
        )
        if is_unibody:
            parts.append(
                {
                    "id": "toolhead_chassis",
                    "name": f"{spec_th.name or 'Toolhead'} Unibody",
                    "stage": "chassis",
                    "is_structural": True,
                    "source": "stage_fea",
                    "material": body_mat.name,
                    "mass_g": m_chassis_g,
                    "stiffness_n_per_mm": round(k_body, 1) if k_body > 1e-9 else None,
                    "compliance_share_pct": body_comp_pct,
                    "toolhead_share_pct": round(body_comp_pct / pct_th_total * 100.0, 1),
                    "local_flex_um": round(flex_12_m * frac_body * 1e6, 3),
                    "total_disp_um": round(disp_junc_um, 2),
                    "twist_mrad": round(twist_12_rad * frac_body * 1e3, 3),
                    "stress_mpa": round(float(mount_ms.von_mises_mpa) * 0.85, 2) if mount_ms else None,
                    "yield_mpa": round(float(body_mat.yield_strength_pa) * 1e-6, 1),
                    "why_it_hurts": (
                        "Monolithic chassis flexure pitches the nozzle under Y acceleration."
                        if body_comp_pct >= 10.0
                        else "Unibody chassis stiffness is strong relative to the frame and hotend."
                    ),
                    "action_hint": (
                        "Switch chassis to CNC aluminum, SLM Ti, or stiffer CF polymer (PPA-CF/PPS-CF)."
                        if body_comp_pct >= 10.0
                        else "Unibody chassis flex is already minor."
                    ),
                }
            )
        else:
            # Multi-piece toolhead body (e.g., StealthBurner + CW2, XOL, Dragon Burner, A4T, Archetype):
            # Split into its 3 physical structural sub-components so users can inspect & hide each piece.
            (c1_id, c1_name, c1_stage), (c2_id, c2_name, c2_stage), (c3_id, c3_name, c3_stage) = (
                _multi_piece_chassis_labels(spec_th)
            )
            sub_body_defs = [
                (
                    c1_id,
                    c1_name,
                    c1_stage,
                    0.55,
                    0.38,
                    "Lower hotend carrier bridges the X-carriage bolts to the hotend mount — carries peak bending moment.",
                    "Upgrade body material to PPA-CF/PPS-CF or CNC aluminum to stiffen the hotend carrier.",
                ),
                (
                    c2_id,
                    c2_name,
                    c2_stage,
                    0.30,
                    0.34,
                    "Bolted mid-body bridge / front clamshell carries torsional & shear loads across the bolted seam.",
                    "Use tighter M3 heat-set clamping, stiffer CF polymer, or a CNC front plate.",
                ),
                (
                    c3_id,
                    c3_name,
                    c3_stage,
                    0.15,
                    0.28,
                    "Upper extruder housing carries direct-drive motor inertia above the X-rail.",
                    "Reduce upper extruder mass or switch to a lower-CoG planetary extruder.",
                ),
            ]
            for sid, sname, sstage, comp_w, mass_w, why_txt, hint_txt in sub_body_defs:
                s_comp = round(body_comp_pct * comp_w, 2)
                parts.append(
                    {
                        "id": sid,
                        "name": sname,
                        "stage": sstage,
                        "is_structural": True,
                        "source": "stage_fea",
                        "material": body_mat.name,
                        "mass_g": round(m_chassis_g * mass_w, 1),
                        "stiffness_n_per_mm": round(k_body / comp_w, 1) if k_body > 1e-9 else None,
                        "compliance_share_pct": s_comp,
                        "toolhead_share_pct": round(s_comp / pct_th_total * 100.0, 1),
                        "local_flex_um": round(flex_12_m * frac_body * comp_w * 1e6, 3),
                        "total_disp_um": round(disp_junc_um, 2),
                        "twist_mrad": round(twist_12_rad * frac_body * comp_w * 1e3, 3),
                        "stress_mpa": round(float(mount_ms.von_mises_mpa) * (0.5 + 0.5 * comp_w), 2) if mount_ms else None,
                        "yield_mpa": round(float(body_mat.yield_strength_pa) * 1e-6, 1),
                        "why_it_hurts": why_txt,
                        "action_hint": hint_txt,
                    }
                )

    # 4. Upper Extruder & Motor Assembly (when direct-drive or T-branch present)
    if has_t_branch or is_direct_drive:
        parts.append(
            {
                "id": "extruder_upper",
                "name": "Extruder Stepper Motor",
                "stage": "extruder_upper",
                "is_structural": has_t_branch,
                "source": "stage_fea",
                "material": f"{body_mat.name} / NEMA14",
                "mass_g": m_ext_g if m_ext_g > 0 else 85.0,
                "stiffness_n_per_mm": None,
                "compliance_share_pct": 0.0,
                "toolhead_share_pct": 0.0,
                "local_flex_um": round(flex_up_m * 1e6, 3),
                "total_disp_um": round(disp_cog_um, 2),
                "twist_mrad": round(twist_up_rad * 1e3, 3),
                "stress_mpa": round(float(upper_ms.von_mises_mpa), 2) if upper_ms else None,
                "yield_mpa": round(float(body_mat.yield_strength_pa) * 1e-6, 1),
                "why_it_hurts": (
                    "High-mounted extruder/motor mass adds pitch inertia above the X-rail, amplifying Y-axis ringing."
                    if (m_ext_g >= 80.0 or float(spec_th.cog_offset_mm[2]) > 5.0)
                    else "Compact extruder mass keeps pitch inertia low."
                ),
                "action_hint": (
                    "Use a lighter pancake stepper / planetary extruder (Orbiter/Sherpa/Vz-Hextrudort) or lower CoG."
                    if (m_ext_g >= 80.0 or float(spec_th.cog_offset_mm[2]) > 5.0)
                    else "Extruder mass and CoG placement are already well-optimized."
                ),
            }
        )

    # 5. Hotend Mount Collar / Clamp
    clamp_comp_pct = round(pct_hotend * frac_clamp, 2)
    parts.append(
        {
            "id": "hotend_clamp",
            "name": "Hotend Mount Collar",
            "stage": "hotend_clamp",
            "is_structural": True,
            "source": "stage_fea",
            "material": hm_mat.name,
            "mass_g": 8.0,
            "stiffness_n_per_mm": round(k_clamp, 1) if k_clamp > 1e-9 else None,
            "compliance_share_pct": clamp_comp_pct,
            "toolhead_share_pct": round(clamp_comp_pct / pct_th_total * 100.0, 1),
            "local_flex_um": round(flex_3_m * frac_clamp * 1e6, 3),
            "total_disp_um": round(disp_junc_um + flex_3_m * frac_clamp * 1e6, 2),
            "twist_mrad": round(twist_3_rad * frac_clamp * 1e3, 3),
            "stress_mpa": round(float(noz_ms.von_mises_mpa) * 0.75, 2) if noz_ms else None,
            "yield_mpa": round(float(hm_mat.yield_strength_pa) * 1e-6, 1),
            "why_it_hurts": (
                "Hotend collar seating compliance tilts the entire hotend stem under XY acceleration."
                if clamp_comp_pct >= 6.0
                else "Hotend collar clamping is rigid."
            ),
            "action_hint": (
                "Switch from groove mount to a 4/6-bolt rigid flange or thicker metal/CF hotend mount."
                if clamp_comp_pct >= 6.0
                else "Hotend mount clamping is already well-supported."
            ),
        }
    )

    # 6. Optional Ti Bottom Brace (when enabled on braced CAD toolheads)
    if is_braced:
        parts.append(
            {
                "id": "hotend_brace",
                "name": "Hotend Bottom Brace (Ti)",
                "stage": "hotend_brace",
                "is_structural": True,
                "source": "stage_fea",
                "material": "Titanium Grade 5 (Ti-6Al-4V)",
                "mass_g": round(float(acc.get("hotend_brace_mass_g", 2.0)), 1),
                "stiffness_n_per_mm": round(k_hotend_tot * 0.65, 1) if k_hotend_tot > 1e-9 else None,
                "compliance_share_pct": round(pct_hotend * frac_he_only * 0.2, 2),
                "toolhead_share_pct": round((pct_hotend * frac_he_only * 0.2) / pct_th_total * 100.0, 1),
                "local_flex_um": round(flex_3_m * frac_he_only * 0.25 * 1e6, 3),
                "total_disp_um": round(disp_junc_um + flex_3_m * 0.55 * 1e6, 2),
                "twist_mrad": round(twist_3_rad * 0.2 * 1e3, 3),
                "stress_mpa": round(float(noz_ms.von_mises_mpa) * 0.6, 2) if noz_ms else None,
                "yield_mpa": 880.0,
                "why_it_hurts": "Doubly supports the lower hotend heatsink to eliminate cantilever neck whip.",
                "action_hint": "Keep enabled — disabling this brace increases hotend cantilever flex dramatically.",
            }
        )

    # 7. Hotend Heatbreak & Heatsink — Option 2 (3D CAD Recovery) or Option 1 (Stage FEA)
    se_he = _find_superelement(assembly, "nozzle_stem")
    se_he_rec = getattr(se_he, "subpart_recovery", None) if se_he is not None else None
    he_neck_comp_pct = round(pct_hotend * frac_he_only * (0.8 if is_braced else 1.0), 2)
    if se_he_rec is not None:
        q_b_he = np.concatenate([u_junc, u_noz])
        recovered_he = se_he_rec.recover(q_b_he)
        for sp in recovered_he:
            sp_share_he = float(sp["strain_energy_pct"]) / 100.0
            sp_comp_pct = round(he_neck_comp_pct * sp_share_he, 2)
            sp_mat = get_material(sp["material_id"]) if sp["material_id"] else hb_mat
            parts.append(
                {
                    "id": f"cad_he_{sp['name']}",
                    "name": _pretty_subpart_label(sp["name"], fallback_prefix="Hotend "),
                    "stage": "hotend_neck",
                    "is_structural": True,
                    "source": "cad_3d_recovery",
                    "material": sp_mat.name,
                    "mass_g": float(sp["mass_g"]),
                    "stiffness_n_per_mm": round(k_he_only / max(sp_share_he, 0.05), 1) if k_he_only > 1e-9 else None,
                    "compliance_share_pct": sp_comp_pct,
                    "toolhead_share_pct": round(sp_comp_pct / pct_th_total * 100.0, 1),
                    "local_flex_um": round(float(sp["local_flex_um"]), 3),
                    "total_disp_um": round(disp_noz_um, 2),
                    "twist_mrad": round(twist_3_rad * frac_he_only * sp_share_he * 1e3, 3),
                    "stress_mpa": round(float(sp["stress_mpa"]), 3),
                    "yield_mpa": round(float(sp_mat.yield_strength_pa) * 1e-6, 1),
                    "why_it_hurts": f"3D Tet4 recovery shows {sp['strain_energy_pct']:.0f}% of hotend strain energy in this region.",
                    "action_hint": (
                        "Switch to a stiffer structural hotend (Chube / braced Tricorn) to reduce tip whip."
                        if sp_comp_pct >= 8.0
                        else "Hotend structural region stiffness is well controlled."
                    ),
                }
            )
    else:
        he_part_prefix = "Hotend Ti Shell" if _is_chube_hotend(spec_th.hotend) else "Hotend Heatbreak"
        parts.append(
            {
                "id": "hotend_neck",
                "name": f"{he_part_prefix} ({spec_th.hotend.name})",
                "stage": "hotend_neck",
                "is_structural": True,
                "source": "stage_fea",
                "material": hb_mat.name,
                "mass_g": m_he_neck_g,
                "stiffness_n_per_mm": round(k_he_only, 1) if k_he_only > 1e-9 else None,
                "compliance_share_pct": he_neck_comp_pct,
                "toolhead_share_pct": round(he_neck_comp_pct / pct_th_total * 100.0, 1),
                "local_flex_um": round(flex_3_m * frac_he_only * 1e6, 3),
                "total_disp_um": round(disp_noz_um, 2),
                "twist_mrad": round(twist_3_rad * frac_he_only * 1e3, 3),
                "stress_mpa": round(float(noz_ms.von_mises_mpa), 2) if noz_ms else None,
                "yield_mpa": round(float(hb_mat.yield_strength_pa) * 1e-6, 1),
                "why_it_hurts": (
                    "Thin-walled heatbreak neck / standoffs bend like a cantilever under nozzle & heater-block inertia."
                    if he_neck_comp_pct >= 10.0
                    else "Hotend heatbreak & heatsink stiffness is strong."
                ),
                "action_hint": (
                    "Switch to a braced or large-diameter titanium-shell hotend (Chube / Tricorn Braced)."
                    if he_neck_comp_pct >= 10.0
                    else "Hotend structural stem is already stiff."
                ),
            }
        )

    # 8. Heater Block & Nozzle Tip (rigid tip mass rider)
    parts.append(
        {
            "id": "heater_nozzle",
            "name": f"Heater Block & {spec_th.hotend.nozzle_diameter_mm}mm Nozzle",
            "stage": "heater_nozzle",
            "is_structural": False,
            "source": "stage_fea",
            "material": "Copper Alloy / Hardened Steel",
            "mass_g": m_heater_noz_g,
            "stiffness_n_per_mm": None,
            "compliance_share_pct": 0.0,
            "toolhead_share_pct": 0.0,
            "local_flex_um": 0.0,
            "total_disp_um": round(disp_noz_um, 2),
            "twist_mrad": round(float(np.linalg.norm(u_noz[3:6])) * 1e3, 3),
            "stress_mpa": None,
            "yield_mpa": None,
            "why_it_hurts": (
                f"Tip mass ({m_heater_noz_g:.0f} g) sits at the full {abs(float(spec_th.nozzle_offset_mm[2])):.0f} mm Z drop — drives cantilever bending in the heatbreak above it."
            ),
            "action_hint": "Reduce nozzle drop height or use a lighter compact melt zone to cut tip inertia.",
        }
    )

    # 9. Cooling Shroud, Ducts & Accessories (non-structural rider)
    if m_shroud_g >= 2.0:
        parts.append(
            {
                "id": "cooling_shroud",
                "name": "Shroud, Fans, PCB & Bolts",
                "stage": "cooling_shroud",
                "is_structural": False,
                "source": "stage_fea",
                "material": f"{body_mat.name} (Non-Structural Rider)",
                "mass_g": m_shroud_g,
                "stiffness_n_per_mm": None,
                "compliance_share_pct": 0.0,
                "toolhead_share_pct": 0.0,
                "local_flex_um": 0.0,
                "total_disp_um": round(disp_junc_um, 2),
                "twist_mrad": round(float(np.linalg.norm(u_junc[3:6])) * 1e3, 3),
                "stress_mpa": 0.0,
                "yield_mpa": None,
                "why_it_hurts": "Non-load-bearing mass rider — does not carry print forces, but adds moving inertia to the gantry.",
                "action_hint": "Switch to remote CPAP part cooling or lighter printed ducts to shed moving mass.",
            }
        )

    return parts


def compute_modal_toolhead_parts(
    assembly: AssemblyModel,
    node_idx: dict[str, int],
    phi_vec: np.ndarray,
    max_node_disp: float,
) -> list[dict[str, Any]]:
    """Compute per-part toolhead modal flex share & kinetic shake for a single vibration mode."""
    spec = assembly.spec
    spec_th = spec.toolhead
    spec_gan = spec.gantry
    is_delta = spec.architecture == "delta"

    th_id = assembly.toolhead_node_id
    cog_id = assembly.toolhead_cog_node_id if assembly.toolhead_cog_node_id in node_idx else th_id
    noz_id = assembly.nozzle_node_id if assembly.nozzle_node_id in node_idx else cog_id
    junc_id = "toolhead_chassis" if "toolhead_chassis" in node_idx else cog_id
    has_t_branch = junc_id == "toolhead_chassis" and cog_id != junc_id

    norm = max(float(max_node_disp), 1e-12)
    p_th = _node_pos(assembly, node_idx, th_id)
    p_junc = _node_pos(assembly, node_idx, junc_id)
    p_cog = _node_pos(assembly, node_idx, cog_id)
    p_noz = _node_pos(assembly, node_idx, noz_id)

    u_th = _get_node_dof6(phi_vec, node_idx, th_id)
    u_junc = _get_node_dof6(phi_vec, node_idx, junc_id)
    u_cog = _get_node_dof6(phi_vec, node_idx, cog_id)
    u_noz = _get_node_dof6(phi_vec, node_idx, noz_id)

    flex_12, twist_12 = _rel_elastic_flex_m(u_th, p_th, u_junc, p_junc)
    flex_up, twist_up = (
        _rel_elastic_flex_m(u_junc, p_junc, u_cog, p_cog) if has_t_branch else (0.0, 0.0)
    )
    flex_3, twist_3 = _rel_elastic_flex_m(u_junc, p_junc, u_noz, p_noz)

    disp_th_n = float(np.linalg.norm(u_th[:3])) / norm
    disp_junc_n = float(np.linalg.norm(u_junc[:3])) / norm
    disp_cog_n = float(np.linalg.norm(u_cog[:3])) / norm
    disp_noz_n = float(np.linalg.norm(u_noz[:3])) / norm

    stage_stiff = getattr(assembly, "toolhead_stage_stiffnesses_n_per_mm", {}) or {}
    k_rail = float(stage_stiff.get("rail_carriage_n_per_mm", stage_stiff.get("carriage_mount_n_per_mm", 0.0)))
    k_bp = float(stage_stiff.get("backplate_only_n_per_mm", 0.0))
    k_body = float(stage_stiff.get("body_n_per_mm", 0.0))
    k_clamp = float(stage_stiff.get("hotend_clamp_n_per_mm", 0.0))
    k_hotend_tot = float(stage_stiff.get("hotend_total_n_per_mm", spec_th.hotend.structural_stiffness_n_per_mm))

    c_rail = (1.0 / k_rail) if k_rail > 1e-9 else 0.0
    c_bp = (1.0 / k_bp) if k_bp > 1e-9 else 0.0
    c_body = (1.0 / k_body) if k_body > 1e-9 else 0.0
    c_12_sum = max(c_rail + c_bp + c_body, 1e-15)
    frac_rail = c_rail / c_12_sum if c_rail > 0 else 0.0
    frac_bp = c_bp / c_12_sum if c_bp > 0 else 0.0
    frac_body = max(0.0, 1.0 - frac_rail - frac_bp)

    c_he_tot = (1.0 / k_hotend_tot) if k_hotend_tot > 1e-9 else 0.0
    c_clamp = (1.0 / k_clamp) if k_clamp > 1e-9 else 0.0
    c_clamp_eff = min(c_clamp, c_he_tot * 0.85) if c_he_tot > 1e-12 else c_clamp
    c_he_only = max(c_he_tot - c_clamp_eff, c_he_tot * 0.15) if c_he_tot > 1e-12 else 0.0
    c_3_sum = max(c_clamp_eff + c_he_only, 1e-15)
    frac_clamp = c_clamp_eff / c_3_sum if c_clamp_eff > 0 else 0.0
    frac_he_only = max(0.0, 1.0 - frac_clamp)

    # Elastic strain energy proxy per stage: U_i = 0.5 * k_i * (flex_i)^2 = 0.5 * F_series^2 * c_i
    k_12_eff = (1.0 / c_12_sum) * 1e3 if c_12_sum > 1e-12 else 1e6
    u_12_tot = 0.5 * k_12_eff * (flex_12**2)
    k_3_eff = (1.0 / c_3_sum) * 1e3 if c_3_sum > 1e-12 else 1e6
    u_3_tot = 0.5 * k_3_eff * (flex_3**2)
    u_up_tot = 0.5 * (k_12_eff * 0.5) * (flex_up**2) if has_t_branch else 0.0

    acc = resolve_toolhead_hardware_accessories(spec)
    is_braced = bool(acc.get("use_hotend_bottom_brace", False))
    ext_type = str(getattr(spec_th.extruder, "type", "direct_drive") or "direct_drive")
    ext_name = str(getattr(spec_th.extruder, "name", "Direct Drive") or "Direct Drive")
    is_direct_drive = ext_type == "direct_drive"

    raw_items: list[dict[str, Any]] = []

    raw_items.append(
        {
            "id": "rail_carriage",
            "name": "Delta Effector Joint" if is_delta else f"Linear Rail Carriage ({spec_gan.x_rail_profile.upper().replace('_RAIL', '')} {spec_gan.rail_block_type})",
            "stage": "rail_carriage",
            "is_structural": True,
            "source": "stage_fea",
            "strain_u": u_12_tot * frac_rail,
            "rel_flex_norm": (flex_12 * frac_rail) / norm,
            "disp_norm": disp_th_n + (flex_12 * frac_rail) / norm,
            "twist_norm": twist_12 * frac_rail / norm,
            "kinetic_w": 0.15 * (disp_th_n**2),
        }
    )

    if k_bp > 1e-9:
        raw_items.append(
            {
                "id": "carriage_backplate",
                "name": f"X-Carriage Backplate ({spec_th.carriage_plate_thickness_mm:.1f} mm)",
                "stage": "backplate",
                "is_structural": True,
                "source": "stage_fea",
                "strain_u": u_12_tot * frac_bp,
                "rel_flex_norm": (flex_12 * frac_bp) / norm,
                "disp_norm": disp_th_n + (flex_12 * (frac_rail + frac_bp)) / norm,
                "twist_norm": twist_12 * frac_bp / norm,
                "kinetic_w": 0.10 * (disp_junc_n**2),
            }
        )

    se_th = _find_superelement(assembly, "toolhead_mount")
    se_th_rec = getattr(se_th, "subpart_recovery", None) if se_th is not None else None
    if se_th_rec is not None:
        q_b_th = np.concatenate([u_th, u_junc])
        rec_th = se_th_rec.recover(q_b_th)
        for sp in rec_th:
            sp_frac = float(sp["strain_energy_pct"]) / 100.0
            raw_items.append(
                {
                    "id": f"cad_th_{sp['name']}",
                    "name": _pretty_subpart_label(sp["name"]),
                    "stage": "extruder_upper" if "extruder" in sp["name"].lower() else "chassis",
                    "is_structural": True,
                    "source": "cad_3d_recovery",
                    "strain_u": u_12_tot * frac_body * sp_frac,
                    "rel_flex_norm": (float(sp["local_flex_um"]) * 1e-6) / norm,
                    "disp_norm": disp_junc_n,
                    "twist_norm": twist_12 * frac_body * sp_frac / norm,
                    "kinetic_w": float(sp["kinetic_energy"]) if float(sp["kinetic_energy"]) > 0 else 0.25 * (disp_junc_n**2),
                }
            )
    else:
        is_unibody_m = bool(
            se_th is not None and (getattr(se_th, "manifest", {}) or {}).get("is_unibody", False)
        )
        if is_unibody_m:
            raw_items.append(
                {
                    "id": "toolhead_chassis",
                    "name": f"{spec_th.name or 'Toolhead'} Unibody",
                    "stage": "chassis",
                    "is_structural": True,
                    "source": "stage_fea",
                    "strain_u": u_12_tot * frac_body,
                    "rel_flex_norm": (flex_12 * frac_body) / norm,
                    "disp_norm": disp_junc_n,
                    "twist_norm": twist_12 * frac_body / norm,
                    "kinetic_w": 0.30 * (disp_junc_n**2),
                }
            )
        else:
            (c1_id, c1_name, c1_stage), (c2_id, c2_name, c2_stage), (c3_id, c3_name, c3_stage) = (
                _multi_piece_chassis_labels(spec_th)
            )
            for sid, sname, sstage, comp_w, kin_w in (
                (c1_id, c1_name, c1_stage, 0.55, 0.12),
                (c2_id, c2_name, c2_stage, 0.30, 0.10),
                (c3_id, c3_name, c3_stage, 0.15, 0.08),
            ):
                raw_items.append(
                    {
                        "id": sid,
                        "name": sname,
                        "stage": sstage,
                        "is_structural": True,
                        "source": "stage_fea",
                        "strain_u": u_12_tot * frac_body * comp_w,
                        "rel_flex_norm": (flex_12 * frac_body * comp_w) / norm,
                        "disp_norm": disp_junc_n,
                        "twist_norm": twist_12 * frac_body * comp_w / norm,
                        "kinetic_w": kin_w * (disp_junc_n**2),
                    }
                )

    if has_t_branch or is_direct_drive:
        raw_items.append(
            {
                "id": "extruder_upper",
                "name": f"Extruder & Motor ({ext_name})",
                "stage": "extruder_upper",
                "is_structural": has_t_branch,
                "source": "stage_fea",
                "strain_u": u_up_tot,
                "rel_flex_norm": flex_up / norm,
                "disp_norm": disp_cog_n,
                "twist_norm": twist_up / norm,
                "kinetic_w": 0.35 * (disp_cog_n**2),
            }
        )

    raw_items.append(
        {
            "id": "hotend_clamp",
            "name": f"Hotend Mount Collar ({_pretty_hotend_mount_style(getattr(spec_th.hotend, 'mount_style', 'rigid_flange'))})",
            "stage": "hotend_clamp",
            "is_structural": True,
            "source": "stage_fea",
            "strain_u": u_3_tot * frac_clamp,
            "rel_flex_norm": (flex_3 * frac_clamp) / norm,
            "disp_norm": disp_junc_n + (flex_3 * frac_clamp) / norm,
            "twist_norm": twist_3 * frac_clamp / norm,
            "kinetic_w": 0.04 * (disp_junc_n**2),
        }
    )

    if is_braced:
        raw_items.append(
            {
                "id": "hotend_brace",
                "name": "Hotend Bottom Brace (Ti)",
                "stage": "hotend_brace",
                "is_structural": True,
                "source": "stage_fea",
                "strain_u": u_3_tot * frac_he_only * 0.2,
                "rel_flex_norm": (flex_3 * frac_he_only * 0.25) / norm,
                "disp_norm": disp_junc_n + (flex_3 * 0.55) / norm,
                "twist_norm": twist_3 * 0.2 / norm,
                "kinetic_w": 0.02 * (disp_noz_n**2),
            }
        )

    se_he = _find_superelement(assembly, "nozzle_stem")
    se_he_rec = getattr(se_he, "subpart_recovery", None) if se_he is not None else None
    he_scale = 0.8 if is_braced else 1.0
    if se_he_rec is not None:
        q_b_he = np.concatenate([u_junc, u_noz])
        rec_he = se_he_rec.recover(q_b_he)
        for sp in rec_he:
            sp_frac = float(sp["strain_energy_pct"]) / 100.0
            raw_items.append(
                {
                    "id": f"cad_he_{sp['name']}",
                    "name": _pretty_subpart_label(sp["name"], fallback_prefix="Hotend "),
                    "stage": "hotend_neck",
                    "is_structural": True,
                    "source": "cad_3d_recovery",
                    "strain_u": u_3_tot * frac_he_only * he_scale * sp_frac,
                    "rel_flex_norm": (float(sp["local_flex_um"]) * 1e-6) / norm,
                    "disp_norm": disp_noz_n,
                    "twist_norm": twist_3 * frac_he_only * sp_frac / norm,
                    "kinetic_w": float(sp["kinetic_energy"]) if float(sp["kinetic_energy"]) > 0 else 0.12 * (disp_noz_n**2),
                }
            )
    else:
        he_modal_prefix = "Hotend Ti Shell & Heatsink" if _is_chube_hotend(spec_th.hotend) else "Hotend Heatbreak & Heatsink"
        raw_items.append(
            {
                "id": "hotend_neck",
                "name": f"{he_modal_prefix} ({spec_th.hotend.name})",
                "stage": "hotend_neck",
                "is_structural": True,
                "source": "stage_fea",
                "strain_u": u_3_tot * frac_he_only * he_scale,
                "rel_flex_norm": (flex_3 * frac_he_only) / norm,
                "disp_norm": disp_noz_n,
                "twist_norm": twist_3 * frac_he_only / norm,
                "kinetic_w": 0.12 * (disp_noz_n**2),
            }
        )

    raw_items.append(
        {
            "id": "heater_nozzle",
            "name": f"Heater Block & {spec_th.hotend.nozzle_diameter_mm} mm Nozzle",
            "stage": "heater_nozzle",
            "is_structural": False,
            "source": "stage_fea",
            "strain_u": 0.0,
            "rel_flex_norm": 0.0,
            "disp_norm": disp_noz_n,
            "twist_norm": float(np.linalg.norm(u_noz[3:6])) / norm,
            "kinetic_w": 0.08 * (disp_noz_n**2),
        }
    )

    raw_items.append(
        {
            "id": "cooling_shroud",
            "name": "Cooling Shroud, Fans & Probe",
            "stage": "cooling_shroud",
            "is_structural": False,
            "source": "stage_fea",
            "strain_u": 0.0,
            "rel_flex_norm": 0.0,
            "disp_norm": disp_junc_n,
            "twist_norm": float(np.linalg.norm(u_junc[3:6])) / norm,
            "kinetic_w": 0.10 * (disp_junc_n**2),
        }
    )

    u_sum = sum(float(it["strain_u"]) for it in raw_items)
    k_sum = sum(float(it["kinetic_w"]) for it in raw_items)
    out: list[dict[str, Any]] = []
    for it in raw_items:
        out.append(
            {
                "id": it["id"],
                "name": it["name"],
                "stage": it["stage"],
                "is_structural": bool(it["is_structural"]),
                "source": it["source"],
                "modal_flex_pct": round((float(it["strain_u"]) / u_sum * 100.0) if u_sum > 1e-24 else 0.0, 1),
                "kinetic_share_pct": round((float(it["kinetic_w"]) / k_sum * 100.0) if k_sum > 1e-24 else 0.0, 1),
                "rel_flex_norm": round(float(it["rel_flex_norm"]), 4),
                "disp_norm": round(float(it["disp_norm"]), 4),
            }
        )
    return out


def compute_toolhead_part_fields(
    assembly: AssemblyModel,
    node_idx: dict[str, int],
    u_vec: np.ndarray,
    *,
    normalized: bool = False,
) -> dict[str, dict[str, Any]]:
    """Recover 3D intra-part Tet4 flex & von Mises stress fields for toolhead & hotend CAD superelements."""
    th_id = assembly.toolhead_node_id
    cog_id = assembly.toolhead_cog_node_id if assembly.toolhead_cog_node_id in node_idx else th_id
    noz_id = assembly.nozzle_node_id if assembly.nozzle_node_id in node_idx else cog_id
    junc_id = "toolhead_chassis" if "toolhead_chassis" in node_idx else cog_id

    u_th = _get_node_dof6(u_vec, node_idx, th_id)
    u_junc = _get_node_dof6(u_vec, node_idx, junc_id)
    u_noz = _get_node_dof6(u_vec, node_idx, noz_id)

    fields: dict[str, dict[str, Any]] = {}
    for se_id, q_b in (
        ("toolhead_mount", np.concatenate([u_th, u_junc])),
        ("nozzle_stem", np.concatenate([u_junc, u_noz])),
    ):
        se = _find_superelement(assembly, se_id)
        rec = getattr(se, "subpart_recovery", None) if se is not None else None
        if rec is None or not hasattr(rec, "recover_field"):
            continue
        f_data = rec.recover_field(q_b)
        if not f_data:
            continue
        if normalized:
            fl = np.asarray(f_data["flex_um"], dtype=np.float64)
            st = np.asarray(f_data["stress_mpa"], dtype=np.float64)
            fl_max = float(np.max(fl)) if len(fl) > 0 else 0.0
            st_max = float(np.max(st)) if len(st) > 0 else 0.0
            fields[se_id] = {
                "source": f_data["source"],
                "points_mm": f_data["points_mm"],
                "flex_norm": np.round(fl / fl_max, 3).tolist() if fl_max > 1e-12 else [0.0] * len(fl),
                "strain_norm": np.round(st / st_max, 3).tolist() if st_max > 1e-12 else [0.0] * len(st),
            }
        else:
            fields[se_id] = f_data
    return fields
