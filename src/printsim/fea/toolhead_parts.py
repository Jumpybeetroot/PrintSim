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


def _resolve_carriage_asset_id(
    rail_profile_id: str | None,
    block_type: str = "H",
) -> str | None:
    """Map a rail profile ID + block type ('C' | 'H') to its baked ``carriages/*`` asset ID."""
    if not rail_profile_id:
        return None
    rp = str(rail_profile_id).lower().strip()
    suffix = "c" if str(block_type).upper().strip() == "C" else "h"
    if "hgh15" in rp:
        return "hgh15ca"
    if "mgn15" in rp:
        return f"mgn15{suffix}"
    if "mgn9" in rp:
        return f"mgn9{suffix}"
    if "mgn12" in rp:
        return f"mgn12{suffix}"
    return None


def _has_field_recovery(rec: Any) -> bool:
    """Return True if ``rec`` is a ``SubpartRecoveryData`` with sampled 3D Tet4 field operators."""
    if rec is None or not hasattr(rec, "recover_field"):
        return False
    pts = getattr(rec, "field_points_m", None)
    return pts is not None and len(pts) > 0


def _load_superelement_recovery(
    se: Any,
    spec: Any,
) -> tuple[Any | None, np.ndarray | None]:
    """Return ``(subpart_recovery, port_coords_cad_m)`` for an attached CAD superelement.

    Uses ``se.subpart_recovery`` when already attached, otherwise lazily resolves the cached
    bundle via ``printsim.cad_assets.registry.load_superelement`` (pure NumPy, Pyodide-safe).
    """
    port_coords_cad_m: np.ndarray | None = getattr(se, "port_coords_cad_m", None)
    if port_coords_cad_m is not None:
        port_coords_cad_m = np.asarray(port_coords_cad_m, dtype=np.float64)
    else:
        man = getattr(se, "manifest", None) or {}
        p0_mm = man.get("port0_mm")
        p1_mm = man.get("port1_mm")
        if (
            isinstance(p0_mm, (list, tuple))
            and isinstance(p1_mm, (list, tuple))
            and len(p0_mm) == 3
            and len(p1_mm) == 3
        ):
            port_coords_cad_m = np.array([p0_mm, p1_mm], dtype=np.float64) * 1e-3

    rec = getattr(se, "subpart_recovery", None)
    if _has_field_recovery(rec):
        return rec, port_coords_cad_m

    cat = str(getattr(se, "category", "") or "")
    aid = str(getattr(se, "asset_id", "") or "")
    if not cat or not aid or aid == "procedural_xy_joint":
        return None, None

    try:
        from printsim.cad_assets.registry import load_superelement
    except ImportError:
        return None, None

    mat_id: str | None = None
    chamber_c = float(getattr(getattr(spec, "frame", None), "chamber_temp_c", 25.0))
    if cat == "gantry_brackets":
        se_id = str(getattr(se, "id", "") or "")
        gantry = getattr(spec, "gantry", None)
        if se_id.startswith("cad_z_joint_") and gantry is not None:
            zj_mat = getattr(gantry, "z_joint_material", "printed")
            mat_id = "abs_printed" if zj_mat == "printed" else "aluminum_6061_t6"
        elif gantry is not None:
            mat_id = getattr(gantry, "bracket_material", None)
    elif cat == "frame_braces":
        mat_id = "aluminum_6061_t6"

    loaded = load_superelement(
        cat,
        aid,
        material_id=mat_id,
        chamber_temp_c=chamber_c,
    )
    loaded_rec = getattr(loaded, "subpart_recovery", None) if loaded is not None else None
    if not _has_field_recovery(loaded_rec):
        return None, None
    loaded_ports = getattr(loaded, "port_coords_m", None)
    if port_coords_cad_m is None and loaded_ports is not None:
        port_coords_cad_m = np.asarray(loaded_ports, dtype=np.float64)
    return loaded_rec, port_coords_cad_m


def _rot_z_3x3(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _superelement_rotation_cad_to_asm(se: Any) -> np.ndarray:
    """Return the 3x3 rotation matrix ``R`` mapping CAD-local vectors to assembly coordinates."""
    rot = getattr(se, "rotation", None)
    if rot is not None:
        arr = np.asarray(rot, dtype=np.float64)
        if arr.shape == (3, 3):
            return arr
        if arr.ndim == 0:
            return _rot_z_3x3(float(arr))

    man = getattr(se, "manifest", None) or {}
    if "rotation_matrix" in man:
        arr = np.asarray(man["rotation_matrix"], dtype=np.float64)
        if arr.shape == (3, 3):
            return arr
    if "rotation_z_rad" in man:
        return _rot_z_3x3(float(man["rotation_z_rad"]))

    se_id = str(getattr(se, "id", "") or "")
    if se_id in ("cad_z_joint_rl", "cad_z_joint_rr"):
        return _rot_z_3x3(math.pi)
    return np.eye(3, dtype=np.float64)


def _build_local_qb(
    p0_asm: np.ndarray,
    p1_asm: np.ndarray,
    u0: np.ndarray,
    u1: np.ndarray,
    r_cad_to_asm: np.ndarray,
    *,
    dp_cad_m: np.ndarray | None = None,
    apply_port1_shift: bool = False,
    scale: float = 1.0,
) -> np.ndarray:
    """Build a rigid-body-invariant 12-DOF boundary displacement vector ``q_b`` in CAD coordinates.

    Subtracts Port 0's exact 6-DOF rigid-body motion in the assembly frame, rotates the elastic
    relative deformation into the CAD local frame via ``R.T``, and optionally shifts Port 1 from
    ``dp_asm`` back to ``dp_cad`` (matching ``superelement_element_matrices``). Guarantees exact
    zero flex and zero stress under arbitrary 6-DOF global rigid-body translation and rotation.
    """
    dp_asm = np.asarray(p1_asm, dtype=np.float64) - np.asarray(p0_asm, dtype=np.float64)
    du_el_asm = (u1[:3] - (u0[:3] + np.cross(u0[3:6], dp_asm))) * float(scale)
    dr_el_asm = (u1[3:6] - u0[3:6]) * float(scale)

    rt = r_cad_to_asm.T
    du_el_cad = rt @ du_el_asm
    dr_el_cad = rt @ dr_el_asm

    if apply_port1_shift and dp_cad_m is not None:
        dp_asm_cad = rt @ dp_asm
        du_el_cad = du_el_cad - np.cross(dr_el_cad, dp_asm_cad - dp_cad_m)

    q_b = np.zeros(12, dtype=np.float64)
    q_b[6:9] = du_el_cad
    q_b[9:12] = dr_el_cad
    return q_b


def _format_part_field_entry(
    rec: Any,
    q_b: np.ndarray,
    *,
    asset_id: str,
    label: str,
    node_ids: tuple[str, str] | list[str],
    normalized: bool = False,
) -> dict[str, Any] | None:
    """Evaluate ``rec.recover_field(q_b)`` + ``rec.recover(q_b)`` into the canonical part-field dict."""
    f_data = rec.recover_field(q_b)
    if not f_data:
        return None

    flex_list = f_data["flex_um"]
    stress_list = f_data["stress_mpa"]
    sp_idx_list = f_data["subpart_idx"]

    subparts_out: list[dict[str, Any]] = []
    if hasattr(rec, "recover"):
        raw_subparts = rec.recover(q_b)
        fl_arr = np.asarray(flex_list, dtype=np.float64)
        st_arr = np.asarray(stress_list, dtype=np.float64)
        idx_arr = np.asarray(sp_idx_list, dtype=np.int64) if sp_idx_list else np.empty(0, dtype=np.int64)
        for sp_i, sp in enumerate(raw_subparts):
            mask = (idx_arr == sp_i) if len(idx_arr) == len(fl_arr) else None
            pt_count = int(np.sum(mask)) if mask is not None else 0
            if mask is not None and pt_count > 0:
                sp_max_stress = round(float(np.max(st_arr[mask])), 3)
                sp_mean_stress = round(float(np.mean(st_arr[mask])), 3)
                sp_max_flex = round(float(np.max(fl_arr[mask])), 3)
            else:
                sp_max_stress = float(sp["stress_mpa"])
                sp_mean_stress = float(sp["stress_mpa"])
                sp_max_flex = float(sp["local_flex_um"])
            subparts_out.append(
                {
                    "name": sp["name"],
                    "label": _pretty_subpart_label(sp["name"]),
                    "material_id": sp["material_id"],
                    "mass_g": sp["mass_g"],
                    "strain_energy_pct": sp["strain_energy_pct"],
                    "kinetic_energy_pct": sp["kinetic_energy_pct"],
                    "local_flex_um": sp["local_flex_um"],
                    "stress_mpa": sp["stress_mpa"],
                    "max_stress_mpa": sp_max_stress,
                    "mean_stress_mpa": sp_mean_stress,
                    "max_flex_um": sp_max_flex,
                    "point_count": pt_count,
                }
            )

    entry: dict[str, Any] = {
        "source": f_data["source"],
        "asset_id": str(asset_id),
        "label": str(label),
        "node_ids": [str(node_ids[0]), str(node_ids[1])],
        "points_mm": f_data["points_mm"],
        "flex_um": flex_list,
        "stress_mpa": stress_list,
        "max_flex_um": float(f_data["max_flex_um"]),
        "max_stress_mpa": float(f_data["max_stress_mpa"]),
        "subpart_idx": sp_idx_list,
        "subparts": subparts_out,
    }
    if normalized:
        fl = np.asarray(flex_list, dtype=np.float64)
        st = np.asarray(stress_list, dtype=np.float64)
        fl_max = float(np.max(fl)) if len(fl) > 0 else 0.0
        st_max = float(np.max(st)) if len(st) > 0 else 0.0
        entry["flex_norm"] = np.round(fl / fl_max, 3).tolist() if fl_max > 1e-12 else [0.0] * len(fl)
        entry["strain_norm"] = np.round(st / st_max, 3).tolist() if st_max > 1e-12 else [0.0] * len(st)
    return entry


def _carriage_rotation_x(x_rail_mount: str) -> np.ndarray:
    """CAD-to-assembly rotation matrix for an X-rail HIWIN carriage (CAD +X=rail, +Z=bolt face)."""
    ex = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if x_rail_mount == "front":
        ez = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    elif x_rail_mount == "bottom":
        ez = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    else:
        ez = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    ey = np.cross(ez, ex)
    return np.column_stack([ex, ey, ez])


def _carriage_rotation_y(y_rail_mount: str, side: str) -> np.ndarray:
    """CAD-to-assembly rotation matrix for a Y-rail HIWIN carriage (CAD +X=rail, +Z=bolt face)."""
    ex = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if y_rail_mount == "top":
        ez = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    elif y_rail_mount == "bottom":
        ez = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    elif y_rail_mount == "inner":
        ez = np.array([1.0 if side == "l" else -1.0, 0.0, 0.0], dtype=np.float64)
    else:
        ez = np.array([-1.0 if side == "l" else 1.0, 0.0, 0.0], dtype=np.float64)
    ey = np.cross(ez, ex)
    return np.column_stack([ex, ey, ez])


def compute_printer_part_fields(
    assembly: AssemblyModel,
    node_idx: dict[str, int],
    u_vec: np.ndarray,
    *,
    u_bed_vec: np.ndarray | None = None,
    normalized: bool = False,
) -> dict[str, dict[str, Any]]:
    """Recover 3D intra-part Tet4 flex (µm) and von Mises stress (MPa) across the entire printer.

    Recovers fields for:
    1. Every attached 2-port ``AttachedSuperelement`` whose loaded asset provides
       ``subpart_recovery`` with ``field_points_m`` (toolheads, hotends, braced assemblies,
       XY joints, A/B motor mounts, front idlers, Z-drives, Z-joints, Trident bed arms, etc.).
    2. Attached HIWIN linear-rail carriages (``y_car_bearing_l``, ``y_car_bearing_r``, and
       ``x_carriage``) backed by baked ``carriages/*`` Tet4 superelements.
    3. Modeled frame corner braces / gussets in ``assembly.frame_brace_visuals`` whenever their
       baked ``frame_braces/*`` asset provides ``field_points_m``.

    All boundary vectors ``q_b`` are transformed into each component's local CAD coordinate frame
    with exact 6-DOF rigid-body invariance.
    """
    spec = assembly.spec
    fields: dict[str, dict[str, Any]] = {}

    # 1. Every attached 2-port CAD superelement on the printer
    for se in getattr(assembly, "superelements", ()):
        port_ids = getattr(se, "port_node_ids", ())
        if len(port_ids) != 2:
            continue
        p0_id, p1_id = str(port_ids[0]), str(port_ids[1])
        if p0_id not in node_idx or p1_id not in node_idx:
            continue

        rec, port_coords_cad_m = _load_superelement_recovery(se, spec)
        if rec is None:
            continue

        vec = u_bed_vec if (u_bed_vec is not None and getattr(se, "role", "") == "bed") else u_vec
        u0 = _get_node_dof6(vec, node_idx, p0_id)
        u1 = _get_node_dof6(vec, node_idx, p1_id)
        p0_asm = _node_pos(assembly, node_idx, p0_id)
        p1_asm = _node_pos(assembly, node_idx, p1_id)

        r_mat = _superelement_rotation_cad_to_asm(se)
        dp_cad_m = (
            port_coords_cad_m[1] - port_coords_cad_m[0]
            if port_coords_cad_m is not None and len(port_coords_cad_m) >= 2
            else None
        )
        se_id = str(se.id)
        apply_shift = (
            se_id == "toolhead_mount"
            or se_id.startswith("corner_mount_")
            or getattr(se, "port_coords_cad_m", None) is not None
        )
        q_b = _build_local_qb(
            p0_asm,
            p1_asm,
            u0,
            u1,
            r_mat,
            dp_cad_m=dp_cad_m,
            apply_port1_shift=apply_shift,
        )
        man = getattr(se, "manifest", None) or {}
        label = str(
            man.get("name")
            or getattr(se, "part_group_label", "")
            or getattr(se, "asset_id", "")
            or se_id
        )
        entry = _format_part_field_entry(
            rec,
            q_b,
            asset_id=str(getattr(se, "asset_id", "") or se_id),
            label=label,
            node_ids=(p0_id, p1_id),
            normalized=normalized,
        )
        if entry is not None:
            fields[se_id] = entry

    # 2. HIWIN linear-rail carriages (Y-rail bearings & X-rail toolhead carriage)
    try:
        from printsim.cad_assets.registry import load_superelement
    except ImportError:
        load_superelement = None  # type: ignore[assignment]

    if load_superelement is not None:
        gantry = getattr(spec, "gantry", None)
        block_type = str(getattr(gantry, "rail_block_type", "H") or "H") if gantry is not None else "H"

        # 2a. Explicit 6-DOF Y-carriage bearings (y_car_bearing_l / y_car_bearing_r)
        y_cid = (
            _resolve_carriage_asset_id(getattr(gantry, "y_rail_profile", None), block_type)
            if gantry is not None
            else None
        )
        y_car_asset = load_superelement("carriages", y_cid) if y_cid else None
        y_car_rec = getattr(y_car_asset, "subpart_recovery", None) if y_car_asset is not None else None
        if _has_field_recovery(y_car_rec):
            y_mount = str(getattr(gantry, "y_rail_mount", "top") or "top")
            for sp in getattr(assembly, "fastener_springs", ()):
                sp_id = str(getattr(sp, "id", "") or "")
                if not sp_id.startswith("y_car_bearing_") or sp_id in fields:
                    continue
                p0_id = str(sp.shell_node_id)
                p1_id = str(sp.frame_node_id)
                if p0_id not in node_idx or p1_id not in node_idx:
                    continue
                side = "l" if sp_id.endswith("_l") else "r"
                r_mat = _carriage_rotation_y(y_mount, side)
                u0 = _get_node_dof6(u_vec, node_idx, p0_id)
                u1 = _get_node_dof6(u_vec, node_idx, p1_id)
                p0_asm = _node_pos(assembly, node_idx, p0_id)
                p1_asm = _node_pos(assembly, node_idx, p1_id)
                q_b = _build_local_qb(
                    p0_asm,
                    p1_asm,
                    u0,
                    u1,
                    r_mat,
                    apply_port1_shift=False,
                )
                side_lbl = "Left" if side == "l" else "Right"
                entry = _format_part_field_entry(
                    y_car_rec,
                    q_b,
                    asset_id=y_cid,
                    label=f"Y-Rail Carriage ({side_lbl}, {y_cid.upper()})",
                    node_ids=(p0_id, p1_id),
                    normalized=normalized,
                )
                if entry is not None:
                    fields[sp_id] = entry

        # 2b. X-rail carriage under toolhead (Stage-1 Hertzian carriage compliance share)
        x_cid = (
            _resolve_carriage_asset_id(getattr(gantry, "x_rail_profile", None), block_type)
            if gantry is not None and getattr(spec, "architecture", "corexy") in ("corexy", "cartesian", "cartesian_bedslinger")
            else None
        )
        if x_cid and "x_carriage" not in fields:
            th_id = assembly.toolhead_node_id
            cog_id = assembly.toolhead_cog_node_id if assembly.toolhead_cog_node_id in node_idx else th_id
            junc_id = "toolhead_chassis" if "toolhead_chassis" in node_idx else cog_id
            if th_id in node_idx and junc_id in node_idx and junc_id != th_id:
                x_car_asset = load_superelement("carriages", x_cid)
                x_car_rec = getattr(x_car_asset, "subpart_recovery", None) if x_car_asset is not None else None
                if _has_field_recovery(x_car_rec):
                    st_map = getattr(assembly, "toolhead_stage_stiffnesses_n_per_mm", None) or {}
                    k_rail = float(st_map.get("carriage_mount_n_per_mm", 0.0))
                    k_bp = float(st_map.get("backplate_only_n_per_mm", 0.0))
                    k_body = float(st_map.get("body_n_per_mm", 0.0))
                    c_rail = (1.0 / k_rail) if k_rail > 1e-6 else 0.0
                    c_bp = (1.0 / k_bp) if k_bp > 1e-6 else 0.0
                    c_body = (1.0 / k_body) if k_body > 1e-6 else 0.0
                    c_tot = c_rail + c_bp + c_body
                    car_frac = float(np.clip(c_rail / c_tot, 0.02, 1.0)) if (c_tot > 1e-12 and c_rail > 0.0) else 0.15

                    x_mount = str(getattr(gantry, "effective_x_rail_mount", "front") or "front")
                    r_mat = _carriage_rotation_x(x_mount)
                    u0 = _get_node_dof6(u_vec, node_idx, th_id)
                    u1 = _get_node_dof6(u_vec, node_idx, junc_id)
                    p0_asm = _node_pos(assembly, node_idx, th_id)
                    p1_asm = _node_pos(assembly, node_idx, junc_id)
                    q_b = _build_local_qb(
                        p0_asm,
                        p1_asm,
                        u0,
                        u1,
                        r_mat,
                        apply_port1_shift=False,
                        scale=car_frac,
                    )
                    entry = _format_part_field_entry(
                        x_car_rec,
                        q_b,
                        asset_id=x_cid,
                        label=f"X-Rail Carriage ({x_cid.upper()})",
                        node_ids=(th_id, junc_id),
                        normalized=normalized,
                    )
                    if entry is not None:
                        fields["x_carriage"] = entry

        # 3. Frame corner braces / gussets (activates automatically once frame_braces/* include field_points_m)
        brace_vis_list = getattr(assembly, "frame_brace_visuals", None) or ()
        if brace_vis_list:
            brace_cache: dict[str, Any] = {}
            chamber_c = float(getattr(getattr(spec, "frame", None), "chamber_temp_c", 25.0))
            adj_map: dict[str, list[str]] | None = None
            for vis in brace_vis_list:
                vid = str(vis.get("id") or "")
                aid = str(vis.get("asset_id") or "")
                if not vid or not aid or vid in fields:
                    continue
                if aid not in brace_cache:
                    brace_cache[aid] = load_superelement(
                        "frame_braces",
                        aid,
                        material_id="aluminum_6061_t6",
                        chamber_temp_c=chamber_c,
                    )
                b_asset = brace_cache[aid]
                b_rec = getattr(b_asset, "subpart_recovery", None) if b_asset is not None else None
                if not _has_field_recovery(b_rec):
                    continue

                na = str(vis.get("node_a") or "")
                if na not in node_idx:
                    continue
                ex = np.asarray(vis.get("basis_x", [1.0, 0.0, 0.0]), dtype=np.float64)
                ey = np.asarray(vis.get("basis_y", [0.0, 1.0, 0.0]), dtype=np.float64)
                ez = np.asarray(vis.get("basis_z", [0.0, 0.0, 1.0]), dtype=np.float64)
                r_mat = np.column_stack([ex, ey, ez])

                if adj_map is None:
                    adj_map = {}
                    for mem in assembly.members:
                        adj_map.setdefault(mem.start_id, []).append(mem.end_id)
                        adj_map.setdefault(mem.end_id, []).append(mem.start_id)

                p_na = _node_pos(assembly, node_idx, na)
                best_nb: str | None = None
                best_align = 0.5
                for nb in adj_map.get(na, ()):
                    if nb not in node_idx:
                        continue
                    v_leg = _node_pos(assembly, node_idx, nb) - p_na
                    ln = float(np.linalg.norm(v_leg))
                    if ln < 1e-6:
                        continue
                    align = float(np.dot(v_leg / ln, ey))
                    if align > best_align:
                        best_align = align
                        best_nb = nb
                if best_nb is None:
                    continue

                p_nb = _node_pos(assembly, node_idx, best_nb)
                leg_len = max(float(np.linalg.norm(p_nb - p_na)), 1e-6)
                b_ports = getattr(b_asset, "port_coords_m", None)
                if b_ports is not None and len(b_ports) >= 2:
                    dp_cad_b = np.asarray(b_ports[1], dtype=np.float64) - np.asarray(b_ports[0], dtype=np.float64)
                    arm_scale = min(1.0, float(np.linalg.norm(dp_cad_b)) / leg_len)
                else:
                    arm_scale = min(1.0, 0.04 / leg_len)
                u0 = _get_node_dof6(u_vec, node_idx, na)
                u1 = _get_node_dof6(u_vec, node_idx, best_nb)
                q_b = _build_local_qb(
                    p_na,
                    p_nb,
                    u0,
                    u1,
                    r_mat,
                    apply_port1_shift=False,
                    scale=arm_scale,
                )
                b_man = getattr(b_asset, "manifest", None) or {}
                label = str(b_man.get("name") or f"Frame Corner Brace ({vid})")
                entry = _format_part_field_entry(
                    b_rec,
                    q_b,
                    asset_id=aid,
                    label=label,
                    node_ids=(na, best_nb),
                    normalized=normalized,
                )
                if entry is not None:
                    fields[vid] = entry

    return fields


def compute_toolhead_part_fields(
    assembly: AssemblyModel,
    node_idx: dict[str, int],
    u_vec: np.ndarray,
    *,
    normalized: bool = False,
) -> dict[str, dict[str, Any]]:
    """Recover 3D intra-part Tet4 flex & von Mises stress fields for toolhead & hotend CAD superelements."""
    all_fields = compute_printer_part_fields(
        assembly=assembly,
        node_idx=node_idx,
        u_vec=u_vec,
        normalized=normalized,
    )
    return {
        k: v for k, v in all_fields.items() if k in ("toolhead_mount", "nozzle_stem")
    }
