"""Parametric 3D printer assembly DSL (YAML / JSON / Python)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

Architecture = Literal["corexy", "cartesian_bedslinger", "delta"]
BracingType = Literal[
    "none",
    "corner_gussets",
    "top_cross",
    "rear_cross",
    "side_cross",
    "rear_and_sides",
    "full_truss",
]
PanelMountingType = Literal["none", "clip_enclosure", "bolted_structural"]
ExtrusionFillingType = Literal[
    "none",
    "pu_foam",
    "silica_sand",
    "lead_shot",
    "epoxy_granite",
    "uhpc_grout",
]
BracketMaterialType = Literal[
    "abs_printed",
    "pet_cf_printed",
    "pa12_cf_printed",
    "ppa_cf_printed",
    "pps_cf_printed",
    "aluminum_6061_t6",
    "aluminum_7075_t6",
]
DriveModeType = Literal["2wd", "awd"]
XRailMountType = Literal["front", "top", "bottom", "dual_top_bottom"]
YRailMountType = Literal["top", "inner", "bottom"]
DriveType = Literal["open_loop_stepper", "closed_loop_stepper", "foc_servo"]
InputShaperType = Literal[
    "auto",
    "none",
    "zv",
    "mzv",
    "ei",
    "2hump_ei",
    "3hump_ei",
    "zvdd",
    "smooth_zv",
    "smooth_ei",
]
CornerJointType = Literal["rigid", "l_bracket", "blind_joint", "gusset_plate", "corner_cube"]
RailPreloadType = Literal["ZF", "Z0", "Z1", "ZA", "ZB"]
RailQualityType = Literal[
    "budget_generic",
    "midgrade_matched",
    "hiwin_genuine",
    "precision_thk_nsk",
]
RailBlockType = Literal["C", "H"]
RailBackerMaterialType = Literal["none", "steel_4140", "titanium_grade5"]
BedMountType = Literal[
    "kinematic_3pt",
    "whopping_pochard_kinematic",
    "trident_3pt_standoffs",
    "rigid_4pt",
    "floating_4pt",
    "silicone_4pt",
    "springs_4pt",
]
FrameFeetType = Literal["rigid_bolted", "rubber_feet", "anti_vib_pads", "paver_on_foam"]
BearingClearanceType = Literal["C2", "CN", "C3"]
ZProbeType = Literal["static_eddy", "klicky_inductive", "cnc_tap", "printed_tap"]
BeltRoutingType = Literal["stock_corexy", "monolith_short"]
BeltType = Literal[
    "gt2_6mm",
    "gt2_9mm",
    "gt2_12mm",
    "gt2_15mm",
    "gt2_dual_9mm",
    "3gt_9mm",
    "3gt_12mm",
    "3gt_15mm",
    "at3_10mm",
    "at3_16mm",
    "at3_gen3_16mm",
    "htd3m_15mm",
    "at5_16mm",
]
BeltBodyMaterialType = Literal["epdm", "tpu"]
BeltValidationTierType = Literal["proven", "experimental", "theoretical"]

# Commercial open-ended & linear synchronous timing belt catalog
# Sources:
# - Gates PowerGrip GT2/GT3 Synchronous Belt Data Summary (2M/3M/5M tensile modulus ratio 18k:30k:86k lbf/in)
# - Elatech / SIT Type-M open-ended PU steel-cord linear catalogs (C_spez: AT3 10mm=80kN, 16mm=135kN;
#   HTD 3M 15mm=127.5kN; AT5 16mm=280kN)
# - ContiTech Synchroflex AT3 Gen III technical sheet (F_zul 16mm=1002N @ 0.4% -> C_spez~250kN, z_min=20, d_min=20mm)
# - Breco / Mulco contraflexure (reverse-bend back-idler) minimum diameter guidelines
# NOTE: 2GT 6/9/12 mm belts are empirically calibrated against community CoreXY ADXL shaper logs;
# 2GT 15 mm / Dual 9 mm are experimental custom-hardware builds; 3GT / AT3 / HTD3M / AT5 entries are
# THEORETICAL datasheet-extrapolated models (unproven on real CoreXY 3D printers).
BELT_CATALOG: dict[str, dict[str, Any]] = {
    "gt2_6mm": {
        "id": "gt2_6mm",
        "label": "Gates 2GT 6mm",
        "profile_family": "2gt",
        "pitch_mm": 2.0,
        "width_mm": 6.0,
        "hardware_width_mm": 6.0,
        "default_core": "fiberglass",
        "body_material": "epdm",
        "ea_default_core_n": 50000.0,
        "linear_mass_kg_m": 0.0083,
        "max_working_tension_n": 58.0,
        "tooth_shear_k_n_per_m_mm": 5.5e4,
        "min_pulley_teeth": 16,
        "min_back_idler_mm": 12.0,
        "max_temp_c": 110.0,
        "cost_usd_per_m": 4.50,
        "validation_tier": "proven",
        "theoretical_summary": "Empirically calibrated against stock Voron Trident / 2.4 / bedslinger / delta ADXL shaper logs.",
        "practical_issues": [],
    },
    "gt2_9mm": {
        "id": "gt2_9mm",
        "label": "Gates 2GT 9mm",
        "profile_family": "2gt",
        "pitch_mm": 2.0,
        "width_mm": 9.0,
        "hardware_width_mm": 9.0,
        "default_core": "fiberglass",
        "body_material": "epdm",
        "ea_default_core_n": 75000.0,
        "linear_mass_kg_m": 0.01245,
        "max_working_tension_n": 92.8,
        "tooth_shear_k_n_per_m_mm": 5.5e4,
        "min_pulley_teeth": 16,
        "min_back_idler_mm": 12.0,
        "max_temp_c": 110.0,
        "cost_usd_per_m": 6.50,
        "validation_tier": "proven",
        "theoretical_summary": "Empirically calibrated against Monolith AWD, RatRig V-Core, and VzBot 330 ADXL shaper logs.",
        "practical_issues": [],
    },
    "gt2_12mm": {
        "id": "gt2_12mm",
        "label": "Gates 2GT 12mm",
        "profile_family": "2gt",
        "pitch_mm": 2.0,
        "width_mm": 12.0,
        "hardware_width_mm": 12.0,
        "default_core": "fiberglass",
        "body_material": "epdm",
        "ea_default_core_n": 100000.0,
        "linear_mass_kg_m": 0.0166,
        "max_working_tension_n": 127.6,
        "tooth_shear_k_n_per_m_mm": 5.5e4,
        "min_pulley_teeth": 16,
        "min_back_idler_mm": 12.0,
        "max_temp_c": 110.0,
        "cost_usd_per_m": 8.50,
        "validation_tier": "proven",
        "theoretical_summary": "Validated on high-tension Monolith / custom CNC AWD CoreXY builds.",
        "practical_issues": [
            "Requires 12 mm pulleys/idlers, double-shear motor supports, and taller XY-joint belt paths than stock 6 mm Voron kits.",
        ],
    },
    "gt2_15mm": {
        "id": "gt2_15mm",
        "label": "Gates 2GT 15mm",
        "profile_family": "2gt",
        "pitch_mm": 2.0,
        "width_mm": 15.0,
        "hardware_width_mm": 15.0,
        "default_core": "fiberglass",
        "body_material": "epdm",
        "ea_default_core_n": 125000.0,
        "linear_mass_kg_m": 0.02075,
        "max_working_tension_n": 162.0,
        "tooth_shear_k_n_per_m_mm": 5.5e4,
        "min_pulley_teeth": 16,
        "min_back_idler_mm": 12.0,
        "max_temp_c": 110.0,
        "cost_usd_per_m": 11.50,
        "validation_tier": "experimental",
        "theoretical_summary": "Experimental (extrapolated from Gates 2GT per-mm stiffness): rarely built on 3D printers due to custom wide-belt hardware.",
        "practical_issues": [
            "Scarce 15 mm 2GT pulleys and flanged idlers; requires custom CNC XY-joints and +6 mm taller belt stacks over 9 mm.",
            "High radial belt loads (>80–120 N) bend standard 5 mm NEMA17 shafts and overload F695 bearings without double-shear live shafts.",
            "Limited empirical shaper benchmarks compared to 6/9/12 mm 2GT.",
        ],
    },
    "gt2_dual_9mm": {
        "id": "gt2_dual_9mm",
        "label": "Dual Parallel Gates 2GT 9mm (18mm eff.)",
        "profile_family": "2gt",
        "pitch_mm": 2.0,
        "width_mm": 18.0,
        "hardware_width_mm": 20.0,
        "default_core": "fiberglass",
        "body_material": "epdm",
        "ea_default_core_n": 150000.0,
        "linear_mass_kg_m": 0.0249,
        "max_working_tension_n": 185.6,
        "tooth_shear_k_n_per_m_mm": 5.5e4,
        "min_pulley_teeth": 16,
        "min_back_idler_mm": 12.0,
        "max_temp_c": 110.0,
        "cost_usd_per_m": 13.00,
        "validation_tier": "experimental",
        "theoretical_summary": "Experimental parallel-belt concept (ideal 2× 9 mm load sharing assumed; real tension mismatch reduces effective stiffness).",
        "practical_issues": [
            "Equalizing static tension between two parallel 9 mm loops is difficult; mismatch shifts load onto one belt and causes tracking drift.",
            "Needs ~20 mm tall stacked pulley/idler assemblies, long double-shear shafts, and custom dual-track carriage clamps.",
            "Double belt count doubles routing complexity and axial motor/frame preload.",
        ],
    },
    "3gt_9mm": {
        "id": "3gt_9mm",
        "label": "Gates 3GT (3MGT) 9mm",
        "profile_family": "3gt",
        "pitch_mm": 3.0,
        "width_mm": 9.0,
        "hardware_width_mm": 9.0,
        "default_core": "fiberglass",
        "body_material": "epdm",
        "ea_default_core_n": 125000.0,
        "linear_mass_kg_m": 0.0180,
        "max_working_tension_n": 180.0,
        "tooth_shear_k_n_per_m_mm": 7.8e4,
        "min_pulley_teeth": 14,
        "min_back_idler_mm": 20.0,
        "max_temp_c": 100.0,
        "cost_usd_per_m": 9.50,
        "validation_tier": "theoretical",
        "theoretical_summary": "Theoretical (datasheet-extrapolated from Gates 3GT tensile ratios; not empirically validated on CoreXY 3D printers).",
        "practical_issues": [
            "Coarser 3 mm tooth pitch increases tooth-mesh VFA ripple and chordal polygon action on small pulleys vs. 2GT.",
            "1.5× larger pulley diameter at the same tooth count (60 mm/rev @ 20T) reduces stepper force per N·m and microstep resolution by 33%.",
            "Requires ≥20 mm OD smooth back-idlers for reverse-bend life, forcing oversized CoreXY idler housings.",
            "3GT pulleys, clamps, and open-ended 9 mm belts have minimal off-the-shelf 3D printer ecosystem support.",
        ],
    },
    "3gt_12mm": {
        "id": "3gt_12mm",
        "label": "Gates 3GT (3MGT) 12mm",
        "profile_family": "3gt",
        "pitch_mm": 3.0,
        "width_mm": 12.0,
        "hardware_width_mm": 12.0,
        "default_core": "fiberglass",
        "body_material": "epdm",
        "ea_default_core_n": 166700.0,
        "linear_mass_kg_m": 0.0240,
        "max_working_tension_n": 240.0,
        "tooth_shear_k_n_per_m_mm": 7.8e4,
        "min_pulley_teeth": 14,
        "min_back_idler_mm": 20.0,
        "max_temp_c": 100.0,
        "cost_usd_per_m": 12.50,
        "validation_tier": "theoretical",
        "theoretical_summary": "Theoretical (datasheet-extrapolated from Gates 3GT tensile ratios; not empirically validated on CoreXY 3D printers).",
        "practical_issues": [
            "Coarser 3 mm tooth pitch worsens VFA surface ripple and chordal polygon excitation relative to 2GT.",
            "Larger 3 mm-pitch pulleys (60 mm/rev @ 20T) cut motor mechanical advantage & step resolution by 1.5× without gear reduction.",
            "Requires ≥20 mm smooth back-idlers and heavy custom CNC double-shear brackets for 12 mm-wide 3GT hardware.",
            "Higher belt mass (24 g/m) and bending resistance add parasitic drag on NEMA17 drives.",
        ],
    },
    "3gt_15mm": {
        "id": "3gt_15mm",
        "label": "Gates 3GT (3MGT) 15mm",
        "profile_family": "3gt",
        "pitch_mm": 3.0,
        "width_mm": 15.0,
        "hardware_width_mm": 15.0,
        "default_core": "fiberglass",
        "body_material": "epdm",
        "ea_default_core_n": 208300.0,
        "linear_mass_kg_m": 0.0300,
        "max_working_tension_n": 300.0,
        "tooth_shear_k_n_per_m_mm": 7.8e4,
        "min_pulley_teeth": 14,
        "min_back_idler_mm": 20.0,
        "max_temp_c": 100.0,
        "cost_usd_per_m": 15.50,
        "validation_tier": "theoretical",
        "theoretical_summary": "Theoretical (datasheet-extrapolated from Gates 3GT catalogs; unproven on real CoreXY printers).",
        "practical_issues": [
            "3 mm tooth pitch degrades fine VFA surface quality and cuts stepper torque advantage by 1.5× vs. 20T 2GT.",
            "Requires ≥20 mm OD back-idlers, 15 mm custom CNC pulleys/idlers, and rigid live-shaft mounts — does not fit standard Voron/VzBot frames.",
            "Heavy 30 g/m belt mass and high bending stiffness increase bearing friction and frame preload.",
        ],
    },
    "at3_10mm": {
        "id": "at3_10mm",
        "label": "Breco/Elatech AT3 10mm (Steel/TPU)",
        "profile_family": "at3",
        "pitch_mm": 3.0,
        "width_mm": 10.0,
        "hardware_width_mm": 10.0,
        "default_core": "steel",
        "body_material": "tpu",
        "ea_default_core_n": 80000.0,
        "linear_mass_kg_m": 0.0220,
        "max_working_tension_n": 320.0,
        "tooth_shear_k_n_per_m_mm": 9.5e4,
        "min_pulley_teeth": 20,
        "min_back_idler_mm": 20.0,
        "max_temp_c": 80.0,
        "cost_usd_per_m": 14.00,
        "validation_tier": "theoretical",
        "theoretical_summary": "Theoretical (extrapolated from Breco/Elatech linear-actuator C_spez datasheets; unproven in high-reversal CoreXY 3D printers).",
        "practical_issues": [
            "Steel cords suffer rapid reverse-bend (contraflexure) fatigue in CoreXY serpentine paths unless every smooth back-idler is ≥20 mm OD and pulleys ≥20T.",
            "92 ShA TPU polyurethane softens sharply above 55 °C chamber temperature (tooth cold-flow & stiffness loss), unlike EPDM 2GT.",
            "3 mm pitch at ≥20T gives ≥60 mm/rev (1.5× coarser than 20T 2GT), hurting VFA surface quality and reducing motor thrust per N·m by 33%.",
            "Stiff steel-cord tracking requires precision-crowned/flanged CNC pulleys and oversized custom XY-joints.",
        ],
    },
    "at3_16mm": {
        "id": "at3_16mm",
        "label": "Breco/Elatech AT3 16mm (Steel/TPU)",
        "profile_family": "at3",
        "pitch_mm": 3.0,
        "width_mm": 16.0,
        "hardware_width_mm": 16.0,
        "default_core": "steel",
        "body_material": "tpu",
        "ea_default_core_n": 135000.0,
        "linear_mass_kg_m": 0.0352,
        "max_working_tension_n": 540.0,
        "tooth_shear_k_n_per_m_mm": 9.5e4,
        "min_pulley_teeth": 20,
        "min_back_idler_mm": 20.0,
        "max_temp_c": 80.0,
        "cost_usd_per_m": 21.00,
        "validation_tier": "theoretical",
        "theoretical_summary": "Theoretical (extrapolated from Breco/Elatech linear-actuator C_spez datasheets; unproven in CoreXY 3D printers).",
        "practical_issues": [
            "Steel cords fracture under CoreXY reverse-bending unless smooth back-idlers are ≥20 mm OD and pulleys ≥20T (19.1 mm PD).",
            "92 ShA TPU body softens above 55 °C enclosed chamber temps, losing tooth shear stiffness and creeping under preload.",
            "16 mm belt width + ≥20 mm idlers require a ground-up CNC gantry redesign (massive stack height, 6–8 mm double-shear live shafts).",
            "60 mm/rev @ 20T (3 mm pitch) worsens VFA ripple and reduces stepper mechanical advantage by 1.5×; heavy 35 g/m belt mass.",
        ],
    },
    "at3_gen3_16mm": {
        "id": "at3_gen3_16mm",
        "label": "ContiTech AT3 Gen III 16mm (HD Steel/TPU)",
        "profile_family": "at3_gen3",
        "pitch_mm": 3.0,
        "width_mm": 16.0,
        "hardware_width_mm": 16.0,
        "default_core": "steel",
        "body_material": "tpu",
        "ea_default_core_n": 250000.0,
        "linear_mass_kg_m": 0.0420,
        "max_working_tension_n": 1002.0,
        "tooth_shear_k_n_per_m_mm": 1.15e5,
        "min_pulley_teeth": 20,
        "min_back_idler_mm": 20.0,
        "max_temp_c": 80.0,
        "cost_usd_per_m": 34.00,
        "validation_tier": "theoretical",
        "theoretical_summary": "Theoretical upper-bound model (extrapolated from ContiTech Synchroflex Gen III C_spez; never validated in 3D printer CoreXY motion).",
        "practical_issues": [
            "High-density steel cords demand ≥20 mm OD smooth back-idlers (≥25 mm recommended) and ≥20T pulleys to avoid rapid contraflexure wire fatigue.",
            "92 ShA TPU matrix loses up to 30–65% tooth stiffness in 65–80 °C ABS/ASA chambers and can cold-flow in clamps.",
            "3 mm pitch @ ≥20T (60 mm/rev) reduces stepper torque advantage by 33% and introduces coarser tooth-mesh VFA artifacts.",
            "Very high bending stiffness, 42 g/m linear mass, expensive industrial pulleys/clamps, and requires CNC frame/gantry mounts capable of >150 N preload.",
        ],
    },
    "htd3m_15mm": {
        "id": "htd3m_15mm",
        "label": "Elatech HTD 3M 15mm (Steel/TPU)",
        "profile_family": "htd3m",
        "pitch_mm": 3.0,
        "width_mm": 15.0,
        "hardware_width_mm": 15.0,
        "default_core": "steel",
        "body_material": "tpu",
        "ea_default_core_n": 127500.0,
        "linear_mass_kg_m": 0.0300,
        "max_working_tension_n": 510.0,
        "tooth_shear_k_n_per_m_mm": 8.5e4,
        "min_pulley_teeth": 20,
        "min_back_idler_mm": 30.0,
        "max_temp_c": 80.0,
        "cost_usd_per_m": 18.50,
        "validation_tier": "theoretical",
        "theoretical_summary": "Theoretical (extrapolated from Elatech HTD 3M steel/TPU industrial catalogs; unproven on 3D printers).",
        "practical_issues": [
            "Requires ≥30 mm OD smooth back-idlers for reverse-bend steel-cord life — practically incompatible with compact CoreXY XY-joint geometry.",
            "HTD semicircular tooth profile has noticeably higher reversal backlash than 2GT/3GT/AT3 unless zero-clearance pulleys are custom-cut.",
            "92 ShA TPU softens above 55 °C chamber temperatures and 60 mm/rev (20T) reduces stepper resolution & torque advantage by 1.5×.",
        ],
    },
    "at5_16mm": {
        "id": "at5_16mm",
        "label": "Breco/Elatech AT5 16mm (Heavy Steel/TPU)",
        "profile_family": "at5",
        "pitch_mm": 5.0,
        "width_mm": 16.0,
        "hardware_width_mm": 16.0,
        "default_core": "steel",
        "body_material": "tpu",
        "ea_default_core_n": 280000.0,
        "linear_mass_kg_m": 0.0544,
        "max_working_tension_n": 1120.0,
        "tooth_shear_k_n_per_m_mm": 1.30e5,
        "min_pulley_teeth": 20,
        "min_back_idler_mm": 60.0,
        "max_temp_c": 80.0,
        "cost_usd_per_m": 26.00,
        "validation_tier": "theoretical",
        "theoretical_summary": "Theoretical CNC upper-bound benchmark only (extrapolated from heavy industrial AT5 datasheets; completely impractical on standard CoreXY printers).",
        "practical_issues": [
            "Requires ≥60 mm OD smooth back-idlers and ≥20T pulleys (31.8 mm PD) — impossible inside Voron/VzBot-scale CoreXY gantries.",
            "5 mm pitch @ 20T gives 100 mm/rev (2.5× coarser than 20T 2GT), destroying step resolution and stepper torque advantage without planetary gearboxes.",
            "Severe 5 mm tooth-pitch chordal/VFA artifacts and heavy 54.4 g/m belt mass + stiffness drag.",
            "92 ShA TPU body softens above 55 °C enclosed chamber temperatures.",
        ],
    },
}
HotendMountStyleType = Literal[
    "groove_mount", "rigid_4bolt_m3", "dragon_4bolt_m25", "chube_6bolt_m3_18pcd"
]
ToolheadConstructionType = Literal[
    "multi_piece_clamshell",
    "compact_cage",
    "plate_sandwich_cf",
    "extruder_as_chassis",
    "cnc_monoblock",
    "cantilever_bowden",
]
ZDriveType = Literal[
    "leadscrew_single",
    "leadscrew_dual",
    "leadscrew_dual_belt_sync",
    "leadscrew_triple",
    "belted_quad_z",
]
ZRailProfileType = Literal[
    "mgn12_rail",
    "mgn9_rail",
    "steel_rod_8mm",
    "steel_rod_10mm",
    "steel_rod_12mm",
    "none",
]
ZLeadscrewType = Literal[
    "auto",
    "tr8x4_pom",
    "tr8x8_pom",
    "tr8x8_brass",
    "sfu1204_bk10",
    "gt2_6mm_z_belt",
    "gt2_9mm_z_belt",
]

# Commercial hotend thermomechanical & structural stiffness catalog
# Sources: manufacturer CAD/datasheets, CNC kitchen / Luke's Laboratory Chube structural benchmarks
HOTEND_CATALOG: dict[str, dict] = {
    "e3d_v6_groove": {
        "id": "e3d_v6_groove",
        "preset_id": "e3d_v6_groove",
        "name": "E3D V6 Groove-Mount (40W Al Block)",
        "short_name": "E3D V6 Groove",
        "mount_style": "groove_mount",
        "structural_stiffness_n_per_mm": 45.0,
        "mass_g": 42.0,
        "overall_length_mm": 62.3,
        "melt_zone_length_mm": 12.5,
        "heater_power_w": 40.0,
        "block_material": "aluminum_6061_t6",
        "block_mass_g": 12.0,
        "nozzle_material": "brass_c360",
        "heatbreak_material": "stainless_304",
        "heatbreak_wall_mm": 0.30,
        "heatbreak_length_mm": 11.0,
        "heatbreak_id_mm": 2.0,
        "heatsink_thermal_resistance_k_w": 4.2,
        "max_temp_c": 285.0,
        "cost_usd": 55.0,
        "notes": "Classic J-head groove-mount collar with 0.30 mm SS304 heatbreak and Al block; compliant collar permits substantial nozzle whip",
    },
    "e3d_revo_voron": {
        "id": "e3d_revo_voron",
        "preset_id": "e3d_revo_voron",
        "name": "E3D Revo Voron (40W RapidChange)",
        "short_name": "Revo Voron",
        "mount_style": "rigid_4bolt_m3",
        "structural_stiffness_n_per_mm": 165.0,
        "mass_g": 30.0,
        "overall_length_mm": 52.0,
        "melt_zone_length_mm": 15.5,
        "heater_power_w": 40.0,
        "block_material": "brass_c360",
        "block_mass_g": 8.0,
        "nozzle_material": "brass_c360",
        "heatbreak_material": "stainless_304",
        "heatbreak_wall_mm": 0.15,
        "heatbreak_length_mm": 10.0,
        "heatbreak_id_mm": 2.0,
        "heatsink_thermal_resistance_k_w": 3.6,
        "max_temp_c": 300.0,
        "cost_usd": 95.0,
        "notes": "Ultra-light 30g 4-bolt M3 rigid heatsink with 0.15 mm SS304 thin-wall integrated nozzle/heatbreak",
    },
    "phaetus_dragon_hf": {
        "id": "phaetus_dragon_hf",
        "preset_id": "phaetus_dragon_hf",
        "name": "Phaetus Dragon HF (4-Bolt Rigid)",
        "short_name": "Dragon HF",
        "mount_style": "dragon_4bolt_m25",
        "structural_stiffness_n_per_mm": 550.0,
        "mass_g": 46.0,
        "overall_length_mm": 62.2,
        "melt_zone_length_mm": 17.5,
        "heater_power_w": 50.0,
        "block_material": "copper_c110",
        "block_mass_g": 13.0,
        "nozzle_material": "brass_c360",
        "heatbreak_material": "titanium_grade5",
        "heatbreak_wall_mm": 0.20,
        "heatbreak_length_mm": 12.0,
        "heatbreak_id_mm": 2.0,
        "heatsink_thermal_resistance_k_w": 3.4,
        "max_temp_c": 350.0,
        "cost_usd": 85.0,
        "notes": "4x M2.5 rigid mount with 4 titanium structural standoffs bypassing the 0.20 mm Ti bi-metal heatbreak",
    },
    "phaetus_dragon_uhf": {
        "id": "phaetus_dragon_uhf",
        "preset_id": "phaetus_dragon_uhf",
        "name": "Phaetus Dragon UHF (Extended Melt)",
        "short_name": "Dragon UHF",
        "mount_style": "dragon_4bolt_m25",
        "structural_stiffness_n_per_mm": 340.0,
        "mass_g": 56.0,
        "overall_length_mm": 81.6,
        "melt_zone_length_mm": 30.0,
        "heater_power_w": 70.0,
        "block_material": "copper_c110",
        "block_mass_g": 18.0,
        "nozzle_material": "copper_c110",
        "heatbreak_material": "titanium_grade5",
        "heatbreak_wall_mm": 0.25,
        "heatbreak_length_mm": 12.0,
        "heatbreak_id_mm": 2.0,
        "heatsink_thermal_resistance_k_w": 3.2,
        "max_temp_c": 350.0,
        "cost_usd": 105.0,
        "notes": "Extended 81.6 mm UHF copper melt zone; longer unsupported lower extender lowers tip stiffness vs Dragon HF",
    },
    "phaetus_rapido_2_hf": {
        "id": "phaetus_rapido_2_hf",
        "preset_id": "phaetus_rapido_2_hf",
        "name": "Phaetus Rapido 2 HF (85W Ceramic)",
        "short_name": "Rapido 2 HF",
        "mount_style": "dragon_4bolt_m25",
        "structural_stiffness_n_per_mm": 500.0,
        "mass_g": 50.0,
        "overall_length_mm": 71.0,
        "melt_zone_length_mm": 22.0,
        "heater_power_w": 85.0,
        "block_material": "copper_c110",
        "block_mass_g": 14.0,
        "nozzle_material": "copper_c110",
        "heatbreak_material": "titanium_grade5",
        "heatbreak_wall_mm": 0.25,
        "heatbreak_length_mm": 12.0,
        "heatbreak_id_mm": 2.0,
        "heatsink_thermal_resistance_k_w": 3.0,
        "max_temp_c": 350.0,
        "cost_usd": 95.0,
        "notes": "4-bolt rigid mount with 3 Ti screws and 85W cylindrical ceramic copper melt core",
    },
    "phaetus_rapido_2_uhf": {
        "id": "phaetus_rapido_2_uhf",
        "preset_id": "phaetus_rapido_2_uhf",
        "name": "Phaetus Rapido 2 UHF (85W Extended)",
        "short_name": "Rapido 2 UHF",
        "mount_style": "dragon_4bolt_m25",
        "structural_stiffness_n_per_mm": 320.0,
        "mass_g": 62.0,
        "overall_length_mm": 79.2,
        "melt_zone_length_mm": 34.0,
        "heater_power_w": 85.0,
        "block_material": "copper_c110",
        "block_mass_g": 17.0,
        "nozzle_material": "copper_c110",
        "heatbreak_material": "titanium_grade5",
        "heatbreak_wall_mm": 0.25,
        "heatbreak_length_mm": 12.0,
        "heatbreak_id_mm": 2.0,
        "heatsink_thermal_resistance_k_w": 2.9,
        "max_temp_c": 350.0,
        "cost_usd": 115.0,
        "notes": "79.2 mm UHF ceramic-ring hotend with copper melt extender; extra lower lever arm trades tip stiffness for flow",
    },
    "mellow_goliath_air": {
        "id": "mellow_goliath_air",
        "preset_id": "mellow_goliath_air",
        "name": "Mellow Goliath Air (100W Long-Melt)",
        "short_name": "Goliath Air",
        "mount_style": "dragon_4bolt_m25",
        "structural_stiffness_n_per_mm": 750.0,
        "mass_g": 96.0,
        "overall_length_mm": 69.2,
        "melt_zone_length_mm": 50.0,
        "heater_power_w": 100.0,
        "block_material": "copper_c110",
        "block_mass_g": 26.0,
        "nozzle_material": "copper_c110",
        "heatbreak_material": "titanium_grade5",
        "heatbreak_wall_mm": 0.30,
        "heatbreak_length_mm": 12.0,
        "heatbreak_id_mm": 2.0,
        "heatsink_thermal_resistance_k_w": 2.8,
        "max_temp_c": 400.0,
        "cost_usd": 145.0,
        "notes": "Heavy 96g high-flow VzBot hotend with 50 mm copper melt zone and wide Ti structural posts",
    },
    "chube_compact": {
        "id": "chube_compact",
        "preset_id": "chube_compact",
        "name": "Luke's Lab Chube Compact (80W Ti Shell)",
        "short_name": "Chube Compact",
        "mount_style": "chube_6bolt_m3_18pcd",
        "structural_stiffness_n_per_mm": 2200.0,
        "mass_g": 62.0,
        "overall_length_mm": 48.0,
        "melt_zone_length_mm": 33.0,
        "heater_power_w": 80.0,
        "block_material": "copper_c110",
        "block_mass_g": 15.0,
        "nozzle_material": "copper_c110",
        "heatbreak_material": "titanium_grade5",
        "heatbreak_wall_mm": 0.25,
        "heatbreak_length_mm": 11.0,
        "heatbreak_id_mm": 2.0,
        "heatsink_thermal_resistance_k_w": 2.6,
        "max_temp_c": 450.0,
        "cost_usd": 185.0,
        "notes": "Compact 48 mm monocoque Ti structural shell with 6-bolt M3 18 mm PCD flange (~2200 N/mm tip stiffness)",
    },
    "chube_air": {
        "id": "chube_air",
        "preset_id": "chube_air",
        "name": "Luke's Lab Chube Air (100W Ti Shell)",
        "short_name": "Chube Air",
        "mount_style": "chube_6bolt_m3_18pcd",
        "structural_stiffness_n_per_mm": 2600.0,
        "mass_g": 86.0,
        "overall_length_mm": 60.0,
        "melt_zone_length_mm": 42.0,
        "heater_power_w": 100.0,
        "block_material": "copper_c110",
        "block_mass_g": 18.0,
        "nozzle_material": "copper_c110",
        "heatbreak_material": "titanium_grade5",
        "heatbreak_wall_mm": 0.25,
        "heatbreak_length_mm": 12.0,
        "heatbreak_id_mm": 2.0,
        "heatsink_thermal_resistance_k_w": 2.5,
        "max_temp_c": 450.0,
        "cost_usd": 210.0,
        "notes": "Finned air-cooled Ti monocoque shell (42 mm Cu melt core, 6-bolt M3 18 mm PCD, ~2600 N/mm tip stiffness)",
    },
    "chube_conduction": {
        "id": "chube_conduction",
        "preset_id": "chube_conduction",
        "name": "Luke's Lab Chube Conduction (120W Ti Shell)",
        "short_name": "Chube Conduction",
        "mount_style": "chube_6bolt_m3_18pcd",
        "structural_stiffness_n_per_mm": 3600.0,
        "mass_g": 82.0,
        "overall_length_mm": 60.0,
        "melt_zone_length_mm": 52.0,
        "heater_power_w": 120.0,
        "block_material": "copper_c110",
        "block_mass_g": 20.0,
        "nozzle_material": "copper_c110",
        "heatbreak_material": "titanium_grade5",
        "heatbreak_wall_mm": 0.25,
        "heatbreak_length_mm": 12.0,
        "heatbreak_id_mm": 2.0,
        "heatsink_thermal_resistance_k_w": 2.2,
        "max_temp_c": 450.0,
        "cost_usd": 235.0,
        "notes": "Conductive-mount full Ti monocoque shell with 52 mm Cu melt core and 6-bolt M3 18 mm PCD mount (~3600 N/mm)",
    },
}

# Z-axis drive mechanism catalog (ISO 2901 TR8 leadscrews, HIWIN/TBI SFU1204+BK10 ballscrew, Gates 2GT Z belts)
Z_DRIVE_MECHANISM_CATALOG: dict[str, dict] = {
    "tr8x4_pom": {
        "id": "tr8x4_pom",
        "name": "TR8x4 Leadscrew + POM Nut (2-Start)",
        "short_name": "TR8x4 POM",
        "kind": "leadscrew",
        "lead_mm": 4.0,
        "root_area_m2": 2.38e-5,
        "k_nut_n_per_um": 11.0,
        "k_thrust_n_per_um": 18.0,
        "k_drive_n_per_um_per_corner": None,
        "backlash_um": 28.0,
        "mass_kg_per_m": 0.28,
        "diameter_mm": 8.0,
        "notes": "Voron Trident standard 2-start 4 mm lead screw with anti-backdrive POM flanged nut",
    },
    "tr8x8_pom": {
        "id": "tr8x8_pom",
        "name": "TR8x8 Leadscrew + POM Nut (4-Start)",
        "short_name": "TR8x8 POM",
        "kind": "leadscrew",
        "lead_mm": 8.0,
        "root_area_m2": 2.38e-5,
        "k_nut_n_per_um": 6.5,
        "k_thrust_n_per_um": 14.0,
        "k_drive_n_per_um_per_corner": None,
        "backlash_um": 45.0,
        "mass_kg_per_m": 0.27,
        "diameter_mm": 8.0,
        "notes": "Common 4-start 8 mm lead screw with Delrin/POM nut; higher helix reduces nut axial stiffness and allows backdrive",
    },
    "tr8x8_brass": {
        "id": "tr8x8_brass",
        "name": "TR8x8 Leadscrew + Brass Nut (4-Start)",
        "short_name": "TR8x8 Brass",
        "kind": "leadscrew",
        "lead_mm": 8.0,
        "root_area_m2": 2.38e-5,
        "k_nut_n_per_um": 38.0,
        "k_thrust_n_per_um": 22.0,
        "k_drive_n_per_um_per_corner": None,
        "backlash_um": 35.0,
        "mass_kg_per_m": 0.27,
        "diameter_mm": 8.0,
        "notes": "Machined C360 brass nut raises thread-flank stiffness substantially over POM on 8 mm leadscrews",
    },
    "sfu1204_bk10": {
        "id": "sfu1204_bk10",
        "name": "SFU1204 Ballscrew + BK10 Angular Bearing",
        "short_name": "SFU1204 + BK10",
        "kind": "ballscrew",
        "lead_mm": 4.0,
        "root_area_m2": 7.09e-5,
        "k_nut_n_per_um": 85.0,
        "k_thrust_n_per_um": 150.0,
        "k_drive_n_per_um_per_corner": None,
        "backlash_um": 8.0,
        "mass_kg_per_m": 0.68,
        "diameter_mm": 12.0,
        "notes": "12 mm rolled ballscrew with recirculating steel ball nut and preloaded BK10 angular-contact thrust block",
    },
    "gt2_6mm_z_belt": {
        "id": "gt2_6mm_z_belt",
        "name": "Gates GT2 6mm Z-Belt Drive (5:1 Reduction)",
        "short_name": "GT2 6mm Z-Belt",
        "kind": "belt",
        "lead_mm": 8.0,
        "root_area_m2": 9.6e-6,
        "k_nut_n_per_um": 0.0,
        "k_thrust_n_per_um": 0.0,
        "k_drive_n_per_um_per_corner": 0.18,
        "backlash_um": 18.0,
        "mass_kg_per_m": 0.017,
        "diameter_mm": 6.0,
        "notes": "Voron 2.4 standard 6 mm fiberglass GT2 Z-belt loop (~180 N/mm effective axial stiffness per corner)",
    },
    "gt2_9mm_z_belt": {
        "id": "gt2_9mm_z_belt",
        "name": "Gates GT2 9mm Z-Belt Drive (5:1 Reduction)",
        "short_name": "GT2 9mm Z-Belt",
        "kind": "belt",
        "lead_mm": 8.0,
        "root_area_m2": 1.44e-5,
        "k_nut_n_per_um": 0.0,
        "k_thrust_n_per_um": 0.0,
        "k_drive_n_per_um_per_corner": 0.27,
        "backlash_um": 12.0,
        "mass_kg_per_m": 0.025,
        "diameter_mm": 9.0,
        "notes": "Heavy-duty 9 mm fiberglass GT2 Z-belt upgrade (+50% Z-corner axial stiffness vs 6 mm belt)",
    },
}

# Z-axis linear guideway profile catalog
Z_RAIL_PROFILES: dict[str, dict] = {
    "mgn12_rail": {
        "id": "mgn12_rail",
        "name": "MGN12H Linear Rail (12x8mm)",
        "short_name": "MGN12H Rail",
        "kind": "linear_rail",
        "width_mm": 12.0,
        "height_mm": 8.0,
        "notes": "Bolted 12 mm steel guideway providing strong Newmark composite column stiffening and high carriage pitch rigidity",
    },
    "mgn9_rail": {
        "id": "mgn9_rail",
        "name": "MGN9H Linear Rail (9x6.5mm)",
        "short_name": "MGN9H Rail",
        "kind": "linear_rail",
        "width_mm": 9.0,
        "height_mm": 6.5,
        "notes": "Lightweight 9 mm steel linear guideway bolted to Z columns/uprights",
    },
    "steel_rod_8mm": {
        "id": "steel_rod_8mm",
        "name": "8mm Hardened Linear Rod + LM8UU",
        "short_name": "8mm Smooth Rod",
        "kind": "smooth_rod",
        "width_mm": 8.0,
        "height_mm": 8.0,
        "notes": "Unsupported 8 mm chrome steel shaft with LM8UU ball bushing (no extrusion composite flange action)",
    },
    "steel_rod_10mm": {
        "id": "steel_rod_10mm",
        "name": "10mm Hardened Linear Rod + LM10UU",
        "short_name": "10mm Smooth Rod",
        "kind": "smooth_rod",
        "width_mm": 10.0,
        "height_mm": 10.0,
        "notes": "Unsupported 10 mm chrome steel shaft with LM10UU ball bushing (2.44x bending EI vs 8 mm rod)",
    },
    "steel_rod_12mm": {
        "id": "steel_rod_12mm",
        "name": "12mm Hardened Linear Rod + LM12UU",
        "short_name": "12mm Smooth Rod",
        "kind": "smooth_rod",
        "width_mm": 12.0,
        "height_mm": 12.0,
        "notes": "Heavy 12 mm chrome steel shaft with LM12UU ball bushing (5.06x bending EI vs 8 mm rod)",
    },
    "none": {
        "id": "none",
        "name": "None (Bare Frame / No Z Guide Rail)",
        "short_name": "None",
        "kind": "none",
        "width_mm": 0.0,
        "height_mm": 0.0,
        "notes": "Baseline bare frame column without bolted Z rail or steel guide rod",
    },
}

# Z-probe carriage architecture catalog (mass, forward CoG overhang shift, sliding-rail pitch compliance)
Z_PROBE_CATALOG: dict[str, dict] = {
    "static_eddy": {
        "id": "static_eddy",
        "name": "Static Eddy / Rigid (Beacon / Cartographer)",
        "short_name": "Static Eddy",
        "added_mass_g": 0.0,
        "cog_y_shift_mm": 0.0,
        "compliance_factor": 1.0,
        "cost_usd": 75.0,
        "notes": "Rigidly bolted non-contact eddy probe; zero carriage play and lowest moving mass",
    },
    "klicky_inductive": {
        "id": "klicky_inductive",
        "name": "Klicky / Inductive / Omron (Static Dock)",
        "short_name": "Klicky / Inductive",
        "added_mass_g": 12.0,
        "cog_y_shift_mm": -1.5,
        "compliance_factor": 0.98,
        "cost_usd": 15.0,
        "notes": "Docked microswitch or inductive probe; near-zero carriage compliance penalty",
    },
    "cnc_tap": {
        "id": "cnc_tap",
        "name": "CNC Voron Tap (ChaoticLab / Al Rail)",
        "short_name": "CNC Tap",
        "added_mass_g": 62.0,
        "cog_y_shift_mm": -6.5,
        "compliance_factor": 0.58,
        "cost_usd": 55.0,
        "notes": "Nozzle-probing CNC optical rail; adds ~62g mass, forward overhang, and sliding-block pitch compliance (~12-20 Hz drop)",
    },
    "printed_tap": {
        "id": "printed_tap",
        "name": "Printed Voron Tap (ABS/ASA Sliding Rail)",
        "short_name": "Printed Tap",
        "added_mass_g": 78.0,
        "cog_y_shift_mm": -8.5,
        "compliance_factor": 0.34,
        "cost_usd": 18.0,
        "notes": "Classic printed Tap carriage; adds ~78g and major sliding-mount flop (often 18-30 Hz resonance drop)",
    },
}

# CoreXY belt routing & idler shaft architecture catalog
BELT_ROUTING_CATALOG: dict[str, dict] = {
    "stock_corexy": {
        "id": "stock_corexy",
        "name": "Stock CoreXY Routing (Standard Bolt Idlers)",
        "short_name": "Stock CoreXY",
        "span_factor": 1.0,
        "idler_stiffness_factor": 1.0,
        "notes": "Standard perimeter CoreXY belt path with conventional M5 bolt idlers",
    },
    "monolith_short": {
        "id": "monolith_short",
        "name": "Monolith Short Path + Live-Shaft Idlers",
        "short_name": "Monolith Short",
        "span_factor": 0.86,
        "idler_stiffness_factor": 1.45,
        "notes": "14% shorter belt path with press-fit live-shaft idlers eliminating M5 bolt bending",
    },
}

# Bed frame mounting hardware catalog (SI-friendly explicit units; Slocum Hertzian kinematic vs elastomeric/spring stiffness)
BED_MOUNT_CATALOG: dict[str, dict] = {
    "kinematic_3pt": {
        "id": "kinematic_3pt",
        "name": "3-Point Maxwell Kinematic",
        "short_name": "Kinematic 3-Pt",
        "mount_points": 3,
        "k_axial_n_per_mm": 32000.0,
        "k_lateral_n_per_mm": 24000.0,
        "k_restraint_thermal": 0.0,
        "damping_delta": 0.0,
        "standoff_height_mm": 16.0,
        "radial_dof_mode": "maxwell_vgroove",
        "tap_rated": False,
        "mount_inset_ratio": 0.38,
        "cost_usd": 45.0,
        "notes": "Zero radial thermal constraint (no taco bow) with high Hertzian hardened-steel ball-groove stiffness",
    },
    "whopping_pochard_kinematic": {
        "id": "whopping_pochard_kinematic",
        "name": "Whopping Pochard Leaf-Spring Kinematic",
        "short_name": "Whopping Pochard",
        "mount_points": 3,
        "k_axial_n_per_mm": 24000.0,
        "k_lateral_n_per_mm": 16000.0,
        "k_restraint_thermal": 0.0,
        "damping_delta": 0.002,
        "standoff_height_mm": 14.0,
        "radial_dof_mode": "radial_flexure",
        "tap_rated": True,
        "mount_inset_ratio": 0.38,
        "cost_usd": 65.0,
        "notes": "Preloaded spring-steel leaf kinematic mount: zero thermal taco bow and rated for Voron Tap probing loads",
    },
    "trident_3pt_standoffs": {
        "id": "trident_3pt_standoffs",
        "name": "3-Point Trident Bakelite Standoffs + GE5C",
        "short_name": "Trident 3-Pt Bakelite",
        "mount_points": 3,
        "k_axial_n_per_mm": 18000.0,
        "k_lateral_n_per_mm": 9500.0,
        "k_restraint_thermal": 22.0,
        "damping_delta": 0.003,
        "standoff_height_mm": 10.0,
        "radial_dof_mode": "slotted_3pt_bakelite",
        "tap_rated": True,
        "mount_inset_ratio": 0.38,
        "cost_usd": 16.0,
        "notes": "Stock Voron/Siboor Trident 3-point Bakelite thermal-isolation standoffs on 2020 T-subframe with GE5C spherical Z-joints",
    },
    "rigid_4pt": {
        "id": "rigid_4pt",
        "name": "4-Point Rigid Hard-Bolted Standoffs",
        "short_name": "Rigid 4-Pt",
        "mount_points": 4,
        "k_axial_n_per_mm": 65000.0,
        "k_lateral_n_per_mm": 40000.0,
        "k_restraint_thermal": 8000.0,
        "damping_delta": -0.002,
        "standoff_height_mm": 12.0,
        "radial_dof_mode": "rigid_all",
        "tap_rated": True,
        "mount_inset_ratio": 0.38,
        "cost_usd": 12.0,
        "notes": "Highest mechanical rigidity (solid metal standoffs), but over-constrains thermal expansion (taco buckling above 80 C)",
    },
    "floating_4pt": {
        "id": "floating_4pt",
        "name": "4-Point Floating / Slotted Tabs",
        "short_name": "Floating 4-Pt",
        "mount_points": 4,
        "k_axial_n_per_mm": 12000.0,
        "k_lateral_n_per_mm": 6000.0,
        "k_restraint_thermal": 18.0,
        "damping_delta": 0.002,
        "standoff_height_mm": 12.0,
        "radial_dof_mode": "slotted_1fixed_3float",
        "tap_rated": True,
        "mount_inset_ratio": 0.38,
        "cost_usd": 22.0,
        "notes": "Slotted/shoulder-bolt tabs relieve radial thermal expansion with solid vertical support",
    },
    "silicone_4pt": {
        "id": "silicone_4pt",
        "name": "4-Point Silicone Spacers (70A)",
        "short_name": "Silicone 4-Pt",
        "mount_points": 4,
        "k_axial_n_per_mm": 48.0,
        "k_lateral_n_per_mm": 16.0,
        "k_restraint_thermal": 12.0,
        "damping_delta": 0.008,
        "standoff_height_mm": 16.0,
        "radial_dof_mode": "elastomer_shear",
        "tap_rated": False,
        "mount_inset_ratio": 0.38,
        "cost_usd": 10.0,
        "notes": "Compliant 70A silicone columns relieve thermal stress and add damping, but allow noticeable bed bounce/tilt",
    },
    "springs_4pt": {
        "id": "springs_4pt",
        "name": "4-Point Yellow Die Springs",
        "short_name": "Springs 4-Pt",
        "mount_points": 4,
        "k_axial_n_per_mm": 16.0,
        "k_lateral_n_per_mm": 4.5,
        "k_restraint_thermal": 8.0,
        "damping_delta": 0.004,
        "standoff_height_mm": 20.0,
        "radial_dof_mode": "coil_spring_shear",
        "tap_rated": False,
        "mount_inset_ratio": 0.38,
        "cost_usd": 6.0,
        "notes": "Classic leveling die springs; very low lateral & vertical stiffness prone to bed ringing and Z-bounce",
    },
}

# Machine benchtop feet / anti-vibration pad catalog (per-foot vertical & lateral stiffness)
FRAME_FEET_CATALOG: dict[str, dict] = {
    "rigid_bolted": {
        "id": "rigid_bolted",
        "name": "Rigid Benchtop / Bolted",
        "short_name": "Rigid Bolted",
        "k_axial_n_per_mm": 0.0,
        "k_lateral_n_per_mm": 0.0,
        "damping_delta": 0.0,
        "notes": "Idealized rigidly anchored frame feet",
    },
    "rubber_feet": {
        "id": "rubber_feet",
        "name": "Standard Rubber / TPU Feet",
        "short_name": "Rubber Feet",
        "k_axial_n_per_mm": 360.0,
        "k_lateral_n_per_mm": 140.0,
        "damping_delta": 0.006,
        "notes": "Typical Voron/printer rubber feet; allows whole-frame benchtop rocking",
    },
    "anti_vib_pads": {
        "id": "anti_vib_pads",
        "name": "Soft Anti-Vibration Pads (HULA / Sorbothane)",
        "short_name": "Anti-Vib Pads",
        "k_axial_n_per_mm": 85.0,
        "k_lateral_n_per_mm": 32.0,
        "damping_delta": 0.015,
        "notes": "Soft isolators lower whole-frame sway frequency while isolating table noise; frame & bed rock mostly together",
    },
    "paver_on_foam": {
        "id": "paver_on_foam",
        "name": "Concrete Paver on Foam Pad (18 kg Base)",
        "short_name": "Paver + Foam",
        "k_axial_n_per_mm": 240.0,
        "k_lateral_n_per_mm": 95.0,
        "damping_delta": 0.022,
        "base_mass_kg": 18.0,
        "notes": "Classic 18 kg concrete paver on dense foam: lowers CoG and strongly damps desk vibration while keeping frame & bed moving together",
    },
}

# Engineering catalog for frame extrusion cavity fillings (SI units; literature / CNC casting data)
EXTRUSION_FILLINGS: dict[str, dict] = {
    "none": {
        "id": "none",
        "name": "Hollow (Unfilled)",
        "short_name": "Hollow",
        "density_kg_m3": 0.0,
        "youngs_modulus_pa": 0.0,
        "bonded": False,
        "damping_boost": 0.0,
        "cost_per_kg_usd": 0.0,
        "color_hex": "#94a3b8",
        "notes": "Standard hollow aluminum extrusion (low structural damping)",
    },
    "pu_foam": {
        "id": "pu_foam",
        "name": "Expanding PU Acoustic Foam",
        "short_name": "PU Foam",
        "density_kg_m3": 45.0,
        "youngs_modulus_pa": 0.02e9,
        "bonded": False,
        "damping_boost": 0.010,
        "cost_per_kg_usd": 18.0,
        "color_hex": "#fde68a",
        "notes": "Ultra-light acoustic deadening; negligible added mass or stiffness",
    },
    "silica_sand": {
        "id": "silica_sand",
        "name": "Kiln-Dried Silica Sand",
        "short_name": "Silica Sand",
        "density_kg_m3": 1550.0,
        "youngs_modulus_pa": 0.0,
        "bonded": False,
        "damping_boost": 0.028,
        "cost_per_kg_usd": 1.5,
        "color_hex": "#d6b88d",
        "notes": "Low-cost granular Coulomb friction damping; zero bonded bending stiffness",
    },
    "lead_shot": {
        "id": "lead_shot",
        "name": "Fine Lead Shot (#8–#9)",
        "short_name": "Lead Shot",
        "density_kg_m3": 6800.0,
        "youngs_modulus_pa": 0.0,
        "bonded": False,
        "damping_boost": 0.042,
        "cost_per_kg_usd": 11.0,
        "color_hex": "#475569",
        "notes": "Maximum granular impact damping & base inertia; heavy mass lowers raw Hz",
    },
    "epoxy_granite": {
        "id": "epoxy_granite",
        "name": "Epoxy Granite (85% Quartz + Epoxy)",
        "short_name": "Epoxy Granite",
        "density_kg_m3": 2250.0,
        "youngs_modulus_pa": 30.0e9,
        "bonded": True,
        "damping_boost": 0.036,
        "cost_per_kg_usd": 9.5,
        "color_hex": "#64748b",
        "notes": "Mineral casting bonds to cavity walls: +10-16% bending stiffness & strong resin damping",
    },
    "uhpc_grout": {
        "id": "uhpc_grout",
        "name": "Non-Shrink UHPC / CSA Grout",
        "short_name": "UHPC Grout",
        "density_kg_m3": 2350.0,
        "youngs_modulus_pa": 40.0e9,
        "bonded": True,
        "damping_boost": 0.030,
        "cost_per_kg_usd": 4.5,
        "color_hex": "#9ca3af",
        "notes": "High-modulus bonded cementitious grout: highest core EI boost + solid damping",
    },
}

# Gantry bracket / motor-mount / XY-joint material catalog (ISO 527 / ISO 75 TDS + Monolith data)
GANTRY_BRACKET_CATALOG: dict[str, dict] = {
    "abs_printed": {
        "id": "abs_printed",
        "name": "3D-Printed ABS / ASA (2.1 GPa)",
        "short_name": "Printed ABS",
        "youngs_modulus_gpa": 2.1,
        "hdt_c": 95.0,
        "creep_tension_limit_n": 45.0,
        "cnc_machined": False,
        "kit_cost_usd": 28.0,
        "carriage_mass_delta_g": 0.0,
        "color_hex": "#e63946",
    },
    "pet_cf_printed": {
        "id": "pet_cf_printed",
        "name": "3D-Printed PET-CF15 Annealed (5.2 GPa)",
        "short_name": "Printed PET-CF15",
        "youngs_modulus_gpa": 5.2,
        "hdt_c": 182.0,
        "creep_tension_limit_n": 75.0,
        "cnc_machined": False,
        "kit_cost_usd": 52.0,
        "carriage_mass_delta_g": 10.0,
        "color_hex": "#3b4252",
    },
    "pa12_cf_printed": {
        "id": "pa12_cf_printed",
        "name": "3D-Printed PA12-CF15 (6.2 GPa)",
        "short_name": "Printed PA12-CF",
        "youngs_modulus_gpa": 6.2,
        "hdt_c": 155.0,
        "creep_tension_limit_n": 65.0,
        "cnc_machined": False,
        "kit_cost_usd": 72.0,
        "carriage_mass_delta_g": 6.0,
        "color_hex": "#333840",
    },
    "ppa_cf_printed": {
        "id": "ppa_cf_printed",
        "name": "3D-Printed PPA-CF Annealed (10.8 GPa)",
        "short_name": "Printed PPA-CF",
        "youngs_modulus_gpa": 10.8,
        "hdt_c": 196.0,
        "creep_tension_limit_n": 105.0,
        "cnc_machined": False,
        "kit_cost_usd": 98.0,
        "carriage_mass_delta_g": 9.0,
        "color_hex": "#2d333b",
    },
    "pps_cf_printed": {
        "id": "pps_cf_printed",
        "name": "3D-Printed PPS-CF Annealed (9.2 GPa, 252°C HDT)",
        "short_name": "Printed PPS-CF",
        "youngs_modulus_gpa": 9.2,
        "hdt_c": 252.0,
        "creep_tension_limit_n": 115.0,
        "cnc_machined": False,
        "kit_cost_usd": 120.0,
        "carriage_mass_delta_g": 12.0,
        "color_hex": "#23272e",
    },
    "aluminum_6061_t6": {
        "id": "aluminum_6061_t6",
        "name": "CNC Aluminum 6061-T6 (68.9 GPa)",
        "short_name": "CNC Al 6061",
        "youngs_modulus_gpa": 68.9,
        "hdt_c": 300.0,
        "creep_tension_limit_n": 250.0,
        "cnc_machined": True,
        "kit_cost_usd": 210.0,
        "carriage_mass_delta_g": 42.0,
        "color_hex": "#c6d0db",
    },
    "aluminum_7075_t6": {
        "id": "aluminum_7075_t6",
        "name": "CNC Aluminum 7075-T6 Billet (71.7 GPa)",
        "short_name": "CNC Al 7075",
        "youngs_modulus_gpa": 71.7,
        "hdt_c": 300.0,
        "creep_tension_limit_n": 280.0,
        "cnc_machined": True,
        "kit_cost_usd": 285.0,
        "carriage_mass_delta_g": 46.0,
        "color_hex": "#cbd5e1",
    },
}


# Y-gantry / XY-joint / carriage / idler / umbilical moving hardware weight classes
GANTRY_WEIGHT_CLASSES: dict[str, dict] = {
    "ultralight_printed": {
        "id": "ultralight_printed",
        "name": "Ultralight Printed (Minimalist / Umbilical)",
        "short_name": "Ultralight Printed",
        "carriage_mass_g": 85.0,
        "suggested_bracket_material": "abs_printed",
        "notes": "Minimalist printed XY-joints with umbilical wiring (no drag chain) and lightweight MGN9/12 blocks.",
    },
    "standard_printed": {
        "id": "standard_printed",
        "name": "Standard Printed (Stock Voron ABS/ASA)",
        "short_name": "Standard Printed",
        "carriage_mass_g": 140.0,
        "suggested_bracket_material": "abs_printed",
        "notes": "Stock Voron Trident/2.4 printed ABS/ASA XY-joints, MGN blocks, idlers, and cable chain hardware.",
    },
    "stiffness_opt_printed": {
        "id": "stiffness_opt_printed",
        "name": "Stiffness-Optimized Printed (CF/GF + Double-Shear)",
        "short_name": "Stiff Printed (CF)",
        "carriage_mass_g": 185.0,
        "suggested_bracket_material": "pet_cf_printed",
        "notes": "Thicker ribbed PET-CF/PA-CF XY-joints with M5 double-shear shafts and steel idler stacks.",
    },
    "cnc_aluminum_light": {
        "id": "cnc_aluminum_light",
        "name": "CNC Aluminum Lightweight (Pocketed 6061)",
        "short_name": "CNC Al (Light)",
        "carriage_mass_g": 210.0,
        "suggested_bracket_material": "aluminum_6061_t6",
        "notes": "Pocketed billet CNC 6061 XY-joints (ChaoticLab/Vitalii/Funssor lightweight style) with zero creep.",
    },
    "monolith_cnc_awd": {
        "id": "monolith_cnc_awd",
        "name": "LDO Monolith CNC 6061 AWD (Scale-Weighed)",
        "short_name": "Monolith CNC AWD",
        "carriage_mass_g": 214.0,
        "suggested_bracket_material": "aluminum_6061_t6",
        "notes": "Scale-weighed LDO Monolith CNC 6061 AWD (CloakedWayne #updates): 2x 75.9 g assembled CNC XY-joints (151.8 g) + 2x 48.3 g LDO MGN12H blocks (96.6 g) + ~7.6 g X-beam fasteners = 256 g dressed (214 g base + 42 g Al delta; MGN9H builds save ~46 g -> ~210 g dressed).",
    },
    "cnc_aluminum_hd": {
        "id": "cnc_aluminum_hd",
        "name": "CNC Aluminum Heavy-Duty (Monolith / AWD Billet)",
        "short_name": "CNC Al (Heavy-Duty)",
        "carriage_mass_g": 280.0,
        "suggested_bracket_material": "aluminum_6061_t6",
        "notes": "Thick solid billet CNC 6061/7075 double-shear XY-joints for 9mm/12mm AWD belt tension and MGN12H/15H blocks.",
    },
}


# Linear rail & carriage manufacturing/grinding precision grade catalog
# Calibrated to Hertzian ball-sharing across tolerance grades (ISO P / HIWIN C/H / sorted / generic)
RAIL_QUALITY_CATALOG: dict[str, dict] = {
    "budget_generic": {
        "id": "budget_generic",
        "name": "Budget / Generic Clone (Loose Raceway)",
        "short_name": "Budget Clone",
        "stiffness_mult": 0.58,
        "drag_mult": 1.45,
        "cost_mult": 0.45,
        "notes": "Wider ball-diameter tolerance (~60% active load-sharing balls) and rougher raceway grinding; noticeable roll play even with Z0/Z1 preload.",
    },
    "midgrade_matched": {
        "id": "midgrade_matched",
        "name": "Matched 3D-Printer Grade (LDO / HoneyBadger / Fysetc)",
        "short_name": "LDO / Matched",
        "stiffness_mult": 0.88,
        "drag_mult": 1.05,
        "cost_mult": 0.85,
        "notes": "Hand-sorted balls and consistent preload matched for high-speed CoreXY printers.",
    },
    "hiwin_genuine": {
        "id": "hiwin_genuine",
        "name": "Genuine HIWIN / PMI (Grade C/H)",
        "short_name": "HIWIN Genuine",
        "stiffness_mult": 1.00,
        "drag_mult": 1.00,
        "cost_mult": 1.00,
        "notes": "Catalog HIWIN Gothic-arch ground raceways and precision-sorted balls (baseline Hertzian stiffness).",
    },
    "precision_thk_nsk": {
        "id": "precision_thk_nsk",
        "name": "Precision Ground (THK / NSK / IKO Grade P)",
        "short_name": "THK / NSK P",
        "stiffness_mult": 1.22,
        "drag_mult": 0.82,
        "cost_mult": 2.10,
        "notes": "High-conformity ground raceways with G5 ball sorting for near-100% ball sharing, higher roll stiffness, and lower seal drag.",
    },
}


class FrameNodeSpec(BaseModel):
    """Explicit 3D structural joint node (coordinates in mm)."""

    id: str
    position_mm: tuple[float, float, float]
    fixed: bool = False
    lumped_mass_g: float = Field(0.0, ge=0)
    role: Literal["frame", "gantry", "foot", "mount", "toolhead"] = "frame"


class FrameMemberSpec(BaseModel):
    """Explicit structural beam/bar/rail connecting two nodes."""

    id: str
    start_node: str
    end_node: str
    profile: str = "extrusion_2020"
    material: str = "aluminum_6063_t5"
    role: Literal["frame", "brace", "gantry", "rail", "tower"] = "frame"


class CustomMeshSpec(BaseModel):
    """Reference to an external STL / OBJ / STEP part attached to the printer."""

    id: str
    name: str
    file_path: str
    attach_to: Literal["frame", "gantry", "toolhead", "bed"] = "frame"
    position_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    material: str = "abs_printed"
    override_mass_g: float | None = Field(None, gt=0)


class FrameSpec(BaseModel):
    """Parametric frame specification with optional custom nodes/members override."""

    width_mm: float = Field(460.0, gt=100, description="Outer frame X span (mm)")
    depth_mm: float = Field(460.0, gt=100, description="Outer frame Y span (mm)")
    height_mm: float = Field(500.0, gt=100, description="Outer frame Z height (mm)")
    profile: str = Field("extrusion_2020", description="Primary structural frame profile ID")
    upright_profile: str | None = Field(
        None,
        description="Optional separate profile for vertical corner uprights/columns (e.g. extrusion_4040 uprights with extrusion_2020 horizontals on DoomCube); defaults to frame.profile",
    )
    material: str = Field("aluminum_6063_t5", description="Primary structural frame material ID")
    bracing: BracingType = Field("corner_gussets", description="Structural triangulation strategy")
    brace_profile: str = Field("brace_rod_10mm", description="Profile for diagonal braces")
    brace_material: str = Field("aluminum_6063_t5", description="Material for diagonal braces")
    enclosed_panels: bool = Field(
        False, description="Whether side/rear/top panels are installed on the frame"
    )
    top_panel: bool = Field(
        True,
        description="Include top enclosure/structural panel on CoreXY frames; disable for open-top or braced-top builds",
    )
    panel_mounting: PanelMountingType = Field(
        "bolted_structural",
        description="Panel attachment stiffness: none, clip_enclosure (loose), or bolted_structural (shear diaphragm)",
    )
    panel_material: str = Field(
        "acm_dibond",
        description="Material ID for enclosure/structural panels (acrylic_pmma, polycarbonate, acm_dibond, aluminum_5052_sheet, carbon_fiber_plate)",
    )
    panel_thickness_mm: float = Field(3.0, ge=1.0, le=12.0, description="Panel sheet thickness (mm)")
    bolt_panels_to_gantry: bool = Field(
        True,
        description="On non-flying-gantry CoreXY machines, bolt structural panels directly to fixed Y-rail & rear gantry beams",
    )
    gantry_top_clearance_mm: float = Field(
        55.0,
        ge=25.0,
        le=300.0,
        description="Vertical distance from top frame extrusion down to the XY gantry plane (mm); 55 mm stock Voron, 80-200 mm for tall toolheads or integrated tophat",
    )
    trident_backbone: bool = Field(
        False,
        description="Clee Trident Backbone (West3D/KB-3D): 3 mm powder-coated mild steel lower-frame plate unifying all 3 Z-motor mounts & stiffening the bottom perimeter (fixed-gantry CoreXY / Voron Trident)",
    )
    extrusion_filling: ExtrusionFillingType = Field(
        "none",
        description="Frame extrusion cavity filling for damping & bonded core stiffness (none, pu_foam, silica_sand, lead_shot, epoxy_granite, uhpc_grout)",
    )
    corner_joint_type: CornerJointType = Field(
        "blind_joint",
        description="Frame corner connection hardware: rigid, l_bracket, blind_joint, gusset_plate, corner_cube",
    )
    feet_type: FrameFeetType = Field(
        "rigid_bolted",
        description="Benchtop foot mounting compliance: rigid_bolted, rubber_feet, anti_vib_pads, or paver_on_foam",
    )
    chamber_temp_c: float = Field(
        25.0, ge=15.0, le=120.0, description="Ambient/heated enclosure chamber air temperature (C)"
    )
    custom_nodes: list[FrameNodeSpec] = Field(default_factory=list)
    custom_members: list[FrameMemberSpec] = Field(default_factory=list)

    @property
    def effective_upright_profile(self) -> str:
        """Vertical corner upright/column profile ID (falls back to primary frame.profile)."""
        return self.upright_profile or self.profile


class BeltSpec(BaseModel):
    """Drive belt specification calibrated to Gates PowerGrip 2GT/3GT & Breco/Elatech/ContiTech datasheets."""

    type: BeltType = "gt2_6mm"
    core: Literal["fiberglass", "steel", "aramid"] = "fiberglass"
    pulley_teeth: int = Field(20, ge=12, le=48)
    pitch_mm: float = Field(2.0, gt=0)
    pretension_n: float = Field(25.0, gt=0, le=400.0)
    wrap_angle_deg: float = Field(180.0, ge=60.0, le=240.0)
    idler_shaft_diameter_mm: float = Field(5.0, ge=3.0, le=12.0)
    back_idler_diameter_mm: float | None = Field(
        None,
        ge=8.0,
        le=80.0,
        description="Optional smooth back-side idler OD (mm) for reverse-bend fatigue check; defaults to pulley pitch diameter",
    )
    bearing_clearance_class: BearingClearanceType = "CN"

    @model_validator(mode="after")
    def _reconcile_catalog_defaults(self) -> BeltSpec:
        entry = BELT_CATALOG.get(self.type, BELT_CATALOG["gt2_6mm"])
        if "pitch_mm" not in self.model_fields_set:
            self.pitch_mm = float(entry["pitch_mm"])
        if "core" not in self.model_fields_set:
            self.core = entry["default_core"]
        return self

    @property
    def catalog_entry(self) -> dict[str, Any]:
        return BELT_CATALOG.get(self.type, BELT_CATALOG["gt2_6mm"])

    @property
    def label(self) -> str:
        return f"{self.catalog_entry['label']} ({self.core})"

    @property
    def width_mm(self) -> float:
        return float(self.catalog_entry["width_mm"])

    @property
    def hardware_width_mm(self) -> float:
        return float(self.catalog_entry["hardware_width_mm"])

    @property
    def body_material(self) -> BeltBodyMaterialType:
        return self.catalog_entry["body_material"]

    @property
    def validation_tier(self) -> BeltValidationTierType:
        return self.catalog_entry.get("validation_tier", "proven")

    @property
    def theoretical_summary(self) -> str:
        return str(self.catalog_entry.get("theoretical_summary", ""))

    @property
    def practical_issues(self) -> list[str]:
        return list(self.catalog_entry.get("practical_issues", []))

    @property
    def tooth_shear_k_n_per_m_mm(self) -> float:
        return float(self.catalog_entry["tooth_shear_k_n_per_m_mm"])

    @property
    def min_pulley_teeth(self) -> int:
        base_min = int(self.catalog_entry["min_pulley_teeth"])
        # Steel-core cords in 2GT pitch suffer severe flex fatigue on small <=20T pulleys
        if self.type.startswith("gt2_") and self.core == "steel":
            return max(base_min, 24)
        return base_min

    @property
    def min_back_idler_diameter_mm(self) -> float:
        base_d = float(self.catalog_entry["min_back_idler_mm"])
        if self.type.startswith("gt2_") and self.core == "steel":
            return max(base_d, 20.0)
        return base_d

    @property
    def pitch_diameter_mm(self) -> float:
        return (self.pulley_teeth * self.pitch_mm) / 3.141592653589793

    @property
    def effective_back_idler_diameter_mm(self) -> float:
        if self.back_idler_diameter_mm is not None:
            return float(self.back_idler_diameter_mm)
        return self.pitch_diameter_mm

    def tooth_modulus_retention(self, chamber_temp_c: float = 25.0) -> float:
        """Elastomer tooth-shear & backing stiffness retention vs. chamber temperature.

        EPDM / polychloroprene (Gates 2GT/3GT) stays stable past 100 C; 92 ShA TPU
        (Breco/Elatech/ContiTech linear belts) softens significantly above 50-55 C.
        """
        t = float(chamber_temp_c)
        if self.body_material == "tpu":
            if t <= 45.0:
                return 1.0
            # ~0.72 at 65 C, ~0.50 at 80 C, floor 0.35
            return max(0.35, 1.0 - 0.014 * (t - 45.0))
        # EPDM / polychloroprene
        if t <= 80.0:
            return 1.0
        return max(0.65, 1.0 - 0.006 * (t - 80.0))

    @property
    def stiffness_ea_n(self) -> float:
        """Dynamic secant tensile modulus EA (Newtons) calibrated to Gates & Elatech/ContiTech datasheets."""
        entry = self.catalog_entry
        base_ea = float(entry["ea_default_core_n"])
        default_core = str(entry["default_core"])
        if self.core == default_core:
            return base_ea
        core_rel = {"fiberglass": 1.0, "aramid": 1.45, "steel": 2.70}
        return base_ea * (core_rel[self.core] / core_rel[default_core])

    @property
    def linear_mass_kg_m(self) -> float:
        """Belt linear mass density mu (kg/m) calibrated to manufacturer datasheets."""
        entry = self.catalog_entry
        base_mu = float(entry["linear_mass_kg_m"])
        default_core = str(entry["default_core"])
        if self.core == default_core:
            return base_mu
        core_density_rel = {"fiberglass": 1.0, "aramid": 0.94, "steel": 1.35}
        return base_mu * (core_density_rel[self.core] / core_density_rel[default_core])

    @property
    def max_working_tension_n(self) -> float:
        """Safe continuous peak working tension (N) per manufacturer engineering ratings."""
        entry = self.catalog_entry
        base_t = float(entry["max_working_tension_n"])
        default_core = str(entry["default_core"])
        if self.core == default_core:
            return base_t
        core_rel = {"fiberglass": 58.0, "aramid": 85.0, "steel": 105.0}
        return base_t * (core_rel[self.core] / core_rel[default_core])

    def pluck_freq_hz(self, span_mm: float = 150.0) -> float:
        """Transverse acoustic belt pluck frequency at given free span (standard Voron/VzBot 150 mm)."""
        span_m = max(span_mm * 1e-3, 0.05)
        return (1.0 / (2.0 * span_m)) * ((self.pretension_n / self.linear_mass_kg_m) ** 0.5)

    @property
    def pitch_radius_m(self) -> float:
        return (self.pulley_teeth * self.pitch_mm * 1e-3) / (2.0 * 3.141592653589793)


class CustomBracketOverride(BaseModel):
    """Injected stiffness & mass override from a 3D Solid/Tet10/CMS bracket evaluation."""

    label: str = Field(..., min_length=1, description="Human-readable CAD bracket label")
    stiffness_n_per_mm: float = Field(
        ..., gt=1.0, le=500000.0, description="Condensed effective bracket stiffness (N/mm)"
    )
    mass_delta_g: float = Field(
        0.0, ge=-200.0, le=500.0, description="Net moving gantry mass delta vs baseline bracket (g)"
    )
    source: str = Field(
        "cms_superelement",
        description="Origin of override: cms_superelement, tet10_fea, hex8_fea, or custom_cad",
    )


class GantrySpec(BaseModel):
    """X/Y gantry beam, rails, drive architecture (2WD/AWD), and belt parameters."""

    drive_mode: DriveModeType = Field(
        "2wd", description="XY drive layout: 2wd (2-motor CoreXY) or awd (4-motor All-Wheel Drive)"
    )
    belt_routing: BeltRoutingType = Field(
        "stock_corexy",
        description="CoreXY belt routing & idler shaft architecture: stock_corexy or monolith_short",
    )
    double_shear_pulleys: bool = Field(
        False, description="Double-shear bearing supports on XY motor shafts & idlers for high belt tension"
    )
    bracket_material: BracketMaterialType = Field(
        "abs_printed",
        description="Material for XY motor mounts, front idlers, and XY-joint gantry brackets",
    )
    x_beam_profile: str = "extrusion_2020"
    x_beam_material: str = "aluminum_6063_t5"
    x_rail_profile: str = "mgn12_rail"
    y_rail_profile: str = "mgn12_rail"
    rail_material: str = "steel_4140"
    x_rail_mount: XRailMountType = Field(
        "front",
        description="X-axis linear rail mounting face on the X gantry beam: front (Voron standard, max Y stiffness), top (RatRig/Annex, max Z stiffness), bottom (under-slung, shorter nozzle drop), or dual_top_bottom (symmetric dual-rail I-beam)",
    )
    y_rail_mount: YRailMountType = Field(
        "top",
        description="Y-axis linear rail mounting face on the left/right Y extrusions: top (Voron standard), inner (side-mounted Monolith/VzBot, max X stiffness), or bottom (inverted)",
    )
    rail_preload: RailPreloadType = Field(
        "Z1", description="HIWIN MGN/HGH carriage ball preload class (ZF, Z0, Z1, ZA, ZB)"
    )
    rail_quality: RailQualityType = Field(
        "hiwin_genuine",
        description="Linear rail & carriage manufacturing/grinding precision grade affecting Hertzian ball-sharing stiffness and rolling drag",
    )
    rail_block_type: RailBlockType = Field(
        "H", description="Linear carriage block length class (C standard vs H long)"
    )
    rail_bolt_pitch_mm: float = Field(
        25.0, ge=15.0, le=100.0, description="Linear rail M3 mounting fastener spacing (mm)"
    )
    rail_backer_material: RailBackerMaterialType = Field(
        "none", description="Optional X-beam opposite-face thermal bow cancellation backer"
    )
    rail_backer_thickness_mm: float = Field(
        3.0, ge=1.0, le=8.0, description="Opposite-face rail backer bar thickness (mm)"
    )
    dual_x_rails: bool = False
    carriage_mass_g: float = Field(
        140.0, ge=20.0, le=1200.0, description="Combined X/Y gantry carriages & joints (g)"
    )
    belt: BeltSpec = Field(default_factory=BeltSpec)
    custom_bracket_override: CustomBracketOverride | None = Field(
        None,
        description="Optional custom CAD/CMS-condensed bracket stiffness & mass injected into full-printer FEA",
    )

    @property
    def effective_x_rail_mount(self) -> XRailMountType:
        """Resolve effective X rail mount, honoring legacy ``dual_x_rails=True``."""
        if self.dual_x_rails or self.x_rail_mount == "dual_top_bottom":
            return "dual_top_bottom"
        return self.x_rail_mount

    @property
    def effective_dual_x_rails(self) -> bool:
        """True when dual top+bottom X rails are active."""
        return self.dual_x_rails or self.x_rail_mount == "dual_top_bottom"


class StepperSpec(BaseModel):
    """Stepper motor electromechanical model calibrated to manufacturer datasheets."""

    preset_id: str = "ldo_2504ac"
    name: str = "NEMA17-48mm (LDO 42STH48-2504AC)"
    holding_torque_nm: float = Field(0.55, gt=0)
    rated_current_a: float = Field(2.5, gt=0, description="Motor datasheet rated RMS current per phase (A)")
    run_current_a: float = Field(
        1.2, gt=0.1, le=5.0, description="Configured driver RMS run current per phase (A)"
    )
    phase_resistance_ohm: float = Field(1.2, gt=0.05, description="Winding phase resistance R (Ohms)")
    inductance_mh: float = Field(1.5, gt=0, description="Winding phase inductance L (mH)")
    supply_voltage_v: float = Field(24.0, gt=0, le=96.0, description="Stepper driver bus voltage (V)")
    rotor_inertia_kg_m2: float = Field(6.8e-6, gt=0)
    motor_mass_g: float = Field(340.0, gt=0)
    steps_per_rev: int = Field(200, ge=100, description="200 for 1.8 deg, 400 for 0.9 deg")

    @property
    def pole_pairs(self) -> float:
        """Electrical pole pairs p = steps_per_rev / 4 (50 for 1.8°, 100 for 0.9°)."""
        return self.steps_per_rev / 4.0

    @property
    def effective_holding_torque_nm(self) -> float:
        """Current-scaled static holding torque: linear below rated current, soft saturation above."""
        import math

        ratio = max(self.run_current_a, 0.0) / max(self.rated_current_a, 0.1)
        if ratio <= 1.0:
            return self.holding_torque_nm * ratio
        sat = 1.8
        return self.holding_torque_nm * (1.0 + math.tanh(sat * (ratio - 1.0)) / sat)

    def dynamic_voltage_efficiency(self, rpm: float = 250.0) -> float:
        """Back-EMF + inductive current-tracking efficiency eta_V in (0, 1] at speed/vibration operating point."""
        import math

        l_h = self.inductance_mh * 1e-3
        omega_mech = max(rpm, 10.0) * (2.0 * math.pi / 60.0)
        k_e = self.holding_torque_nm / max(self.rated_current_a, 0.1)
        v_bemf = k_e * omega_mech
        v_inductive = omega_mech * self.pole_pairs * l_h * self.run_current_a
        v_res = self.run_current_a * self.phase_resistance_ohm
        v_headroom = max(self.supply_voltage_v - v_res, 1.0)
        v_bus = max(self.supply_voltage_v, 1.0)
        return 1.0 / math.sqrt(
            1.0 + ((v_res + v_bemf) / v_bus) ** 2 + (v_inductive / v_headroom) ** 2
        )

    def torsional_stiffness_nm_per_rad(self, rpm: float = 250.0) -> float:
        """Electromagnetic torsional stiffness k_theta = p * T_hold(I) * eta_V (N·m/rad)."""
        return self.pole_pairs * self.effective_holding_torque_nm * self.dynamic_voltage_efficiency(rpm)

    @property
    def power_dissipation_w(self) -> float:
        """Approximate steady-state 2-phase copper heat dissipation P = 2 * I_rms^2 * R (W)."""
        return 2.0 * (self.run_current_a**2) * self.phase_resistance_ohm


STEPPER_CATALOG: dict[str, dict] = {
    "ldo_2504ac": {
        "preset_id": "ldo_2504ac",
        "name": "LDO 42STH48-2504AC (48mm 1.8° Speedy Power)",
        "holding_torque_nm": 0.55,
        "rated_current_a": 2.5,
        "run_current_a": 1.2,
        "phase_resistance_ohm": 1.2,
        "inductance_mh": 1.5,
        "rotor_inertia_kg_m2": 6.8e-6,
        "motor_mass_g": 340.0,
        "steps_per_rev": 200,
    },
    "ldo_2804ah": {
        "preset_id": "ldo_2804ah",
        "name": "LDO 42STH48-2804AH (48mm 1.8° Ultra-Low 0.6mH)",
        "holding_torque_nm": 0.42,
        "rated_current_a": 2.8,
        "run_current_a": 2.0,
        "phase_resistance_ohm": 0.7,
        "inductance_mh": 0.6,
        "rotor_inertia_kg_m2": 6.2e-6,
        "motor_mass_g": 350.0,
        "steps_per_rev": 200,
    },
    "ldo_60_3004": {
        "preset_id": "ldo_60_3004",
        "name": "LDO 42STH60-3004MAC (60mm 1.8° Kraken High-Torque)",
        "holding_torque_nm": 0.80,
        "rated_current_a": 3.0,
        "run_current_a": 2.2,
        "phase_resistance_ohm": 0.9,
        "inductance_mh": 1.8,
        "rotor_inertia_kg_m2": 8.6e-6,
        "motor_mass_g": 520.0,
        "steps_per_rev": 200,
    },
    "ldo_09deg_2504": {
        "preset_id": "ldo_09deg_2504",
        "name": "LDO 42STH48-2504MAC (48mm 0.9° 2× Magnetic Stiffness)",
        "holding_torque_nm": 0.50,
        "rated_current_a": 2.5,
        "run_current_a": 1.6,
        "phase_resistance_ohm": 1.2,
        "inductance_mh": 2.2,
        "rotor_inertia_kg_m2": 6.8e-6,
        "motor_mass_g": 340.0,
        "steps_per_rev": 400,
    },
    "moons_420i": {
        "preset_id": "moons_420i",
        "name": "MOONS' MS17HD6P420I (48mm 1.8° Voron Standard)",
        "holding_torque_nm": 0.65,
        "rated_current_a": 2.0,
        "run_current_a": 1.2,
        "phase_resistance_ohm": 1.3,
        "inductance_mh": 2.8,
        "rotor_inertia_kg_m2": 7.2e-6,
        "motor_mass_g": 360.0,
        "steps_per_rev": 200,
    },
    "stepperonline_2004": {
        "preset_id": "stepperonline_2004",
        "name": "STEPPERONLINE 17HS19-2004S1 (48mm 1.8° Budget)",
        "holding_torque_nm": 0.59,
        "rated_current_a": 2.0,
        "run_current_a": 1.0,
        "phase_resistance_ohm": 1.4,
        "inductance_mh": 3.0,
        "rotor_inertia_kg_m2": 6.8e-6,
        "motor_mass_g": 390.0,
        "steps_per_rev": 200,
    },
}


class KinematicsSpec(BaseModel):
    """Motion limits, build volume, motors, and input shaping."""

    build_volume_mm: tuple[float, float, float] = (300.0, 300.0, 300.0)
    max_velocity_mm_s: float = Field(450.0, gt=0)
    max_accel_mm_s2: float = Field(8000.0, gt=0)
    max_z_velocity_mm_s: float = Field(25.0, gt=0)
    max_z_accel_mm_s2: float = Field(300.0, gt=0)
    square_corner_velocity_mm_s: float = Field(5.0, gt=0)
    input_shaper: InputShaperType = "auto"
    drive_type: DriveType = "open_loop_stepper"
    servo_vel_bw_hz: float = Field(120.0, ge=20.0, le=500.0)
    servo_pos_bw_hz: float = Field(25.0, ge=5.0, le=150.0)
    notch_filter_hz: float | None = Field(None, ge=15.0, le=300.0)
    xy_stepper: StepperSpec = Field(default_factory=StepperSpec)
    z_stepper: StepperSpec = Field(
        default_factory=lambda: StepperSpec(
            name="NEMA17-40mm Z", holding_torque_nm=0.45, motor_mass_g=280.0
        )
    )
    # Delta-specific parameters (ignored for CoreXY/Cartesian)
    delta_arm_length_mm: float = Field(280.0, gt=50)
    delta_radius_mm: float = Field(145.0, gt=30)


_HOTEND_PRESET_FIELDS = (
    "name",
    "mount_style",
    "structural_stiffness_n_per_mm",
    "mass_g",
    "overall_length_mm",
    "melt_zone_length_mm",
    "heater_power_w",
    "block_material",
    "block_mass_g",
    "nozzle_material",
    "heatbreak_material",
    "heatbreak_wall_mm",
    "heatbreak_length_mm",
    "heatbreak_id_mm",
    "heatsink_thermal_resistance_k_w",
    "max_temp_c",
)


class HotendSpec(BaseModel):
    """Thermomechanical hotend & melt-zone specification."""

    preset_id: str | None = None
    mount_style: HotendMountStyleType = "dragon_4bolt_m25"
    structural_stiffness_n_per_mm: float = Field(450.0, ge=15.0, le=6000.0)
    mass_g: float = Field(
        45.0,
        ge=5.0,
        le=400.0,
        description="Assembled bare hotend weight (g) including heatsink, heatbreak, heater block, heater/thermistor, and nozzle",
    )
    overall_length_mm: float = Field(62.0, ge=30.0, le=120.0)
    name: str = "High-Flow Bi-Metal Hotend"
    heater_power_w: float = Field(60.0, gt=5)
    block_material: str = "copper_c110"
    block_mass_g: float = Field(14.0, gt=1)
    nozzle_material: str = "brass_c360"
    nozzle_diameter_mm: float = Field(0.4, gt=0.1, le=2.0)
    melt_zone_length_mm: float = Field(16.0, gt=4.0, le=80.0)
    heatbreak_material: str = "titanium_grade5"
    heatbreak_length_mm: float = Field(12.0, gt=3.0)
    heatbreak_wall_mm: float = Field(0.25, gt=0.05)
    heatbreak_id_mm: float = Field(2.0, gt=1.5)
    heatsink_thermal_resistance_k_w: float = Field(
        3.2, gt=0.2, description="Heatsink-to-ambient thermal resistance with fan running (K/W)"
    )
    part_cooling_cfm: float = Field(8.5, ge=0.0, description="Part cooling fan airflow (CFM)")
    max_temp_c: float = Field(300.0, gt=180.0, le=550.0)
    ambient_temp_c: float = Field(25.0, ge=0.0, le=100.0)

    @model_validator(mode="before")
    @classmethod
    def _apply_hotend_preset_defaults(cls, data: Any) -> Any:
        if isinstance(data, dict):
            pid = data.get("preset_id")
            if isinstance(pid, str) and pid in HOTEND_CATALOG:
                preset = HOTEND_CATALOG[pid]
                merged = dict(data)
                for field_name in _HOTEND_PRESET_FIELDS:
                    if field_name not in merged or merged[field_name] is None:
                        if field_name in preset:
                            merged[field_name] = preset[field_name]
                return merged
        return data

    def __setattr__(self, name: str, value: Any) -> None:
        prev = self.__dict__.get("preset_id") if name == "preset_id" else None
        super().__setattr__(name, value)
        if name == "preset_id" and isinstance(value, str) and value in HOTEND_CATALOG and value != prev:
            preset = HOTEND_CATALOG[value]
            for field_name in _HOTEND_PRESET_FIELDS:
                if field_name in preset:
                    super().__setattr__(field_name, preset[field_name])


class ExtruderSpec(BaseModel):
    """Filament drive extruder specification."""

    type: Literal["direct_drive", "bowden"] = "direct_drive"
    gear_ratio: float = Field(5.0, gt=0.5)
    max_push_force_n: float = Field(55.0, gt=5.0)
    filament_diameter_mm: float = Field(1.75, gt=1.0)


TOOLHEAD_CONSTRUCTION_CATALOG: dict[str, dict] = {
    "multi_piece_clamshell": {
        "id": "multi_piece_clamshell",
        "name": "Multi-Piece Clamshell (StealthBurner / Afterburner)",
        "short_name": "Clamshell",
        "body_stiffness_base_n_per_mm": 110.0,
        "torsion_mult": 0.72,
        "mass_delta_g": 0.0,
        "notes": "Split bolted housing halves with heat-set seam compliance along the carriage-to-hotend load path; easy maintenance but lower torsional rigidity.",
    },
    "compact_cage": {
        "id": "compact_cage",
        "name": "Compact Open-Cage Box (XOL / Dragon Burner / A4T)",
        "short_name": "Open Cage",
        "body_stiffness_base_n_per_mm": 185.0,
        "torsion_mult": 0.92,
        "mass_delta_g": 0.0,
        "notes": "Single-piece core cage with stiffening side cheeks and short X-carriage load path.",
    },
    "plate_sandwich_cf": {
        "id": "plate_sandwich_cf",
        "name": "Plate Sandwich + Standoffs (Archetype / VzBot / EVA)",
        "short_name": "Plate Sandwich",
        "body_stiffness_base_n_per_mm": 270.0,
        "torsion_mult": 1.05,
        "mass_delta_g": 0.0,
        "notes": "Twin flat structural plates tied with metal standoffs for high in-plane roll and bending stiffness.",
    },
    "extruder_as_chassis": {
        "id": "extruder_as_chassis",
        "name": "Integrated Extruder-as-Chassis (Direct-Bolted Core)",
        "short_name": "Extruder Core",
        "body_stiffness_base_n_per_mm": 320.0,
        "torsion_mult": 1.15,
        "mass_delta_g": 0.0,
        "notes": "Extruder gear frame & hotend clamp bolt directly to the X-carriage with minimal shroud load path.",
    },
    "cnc_monoblock": {
        "id": "cnc_monoblock",
        "name": "CNC Monoblock / Ribbed Billet Chassis",
        "short_name": "CNC Monoblock",
        "body_stiffness_base_n_per_mm": 480.0,
        "torsion_mult": 1.35,
        "mass_delta_g": 0.0,
        "notes": "Single-piece machined chassis uniting carriage mount, gusseted webs, and hotend clamp with zero seam slip.",
    },
    "cantilever_bowden": {
        "id": "cantilever_bowden",
        "name": "Minimalist Bowden / Remote-Drive Mount",
        "short_name": "Bowden Bracket",
        "body_stiffness_base_n_per_mm": 230.0,
        "torsion_mult": 1.00,
        "mass_delta_g": 0.0,
        "notes": "Compact hotend-only bracket providing a short, direct cantilever load path to the X-carriage for remote-drive setups.",
    },
}


class ToolheadSpec(BaseModel):
    """Complete toolhead assembly (extruder, hotend, shroud, fans, carriage, CoG & nozzle lever arm)."""

    preset_id: str = "stealthburner"
    name: str = "Stealth-Style Direct Toolhead"
    total_mass_g: float = Field(380.0, gt=20.0)
    cog_offset_mm: tuple[float, float, float] = Field(
        (0.0, -18.0, -25.0),
        description="Toolhead center-of-mass offset [dx, dy, dz] from X-beam centerline (mm)",
    )
    nozzle_offset_mm: tuple[float, float, float] = Field(
        (0.0, -22.0, -62.0),
        description="Nozzle tip offset [dx, dy, dz] from X-beam centerline (mm)",
    )
    mount_material: BracketMaterialType = Field(
        "abs_printed",
        description="Toolhead carriage & XY-joint mount material affecting nozzle pitch/twist stiffness",
    )
    carriage_mount_material: str | None = Field(
        None,
        description="Optional override material for X-carriage backplate (falls back to mount_material)",
    )
    body_material: str | None = Field(
        None,
        description="Optional override material for toolhead body/shroud (falls back to mount_material)",
    )
    hotend_mount_material: str | None = Field(
        None,
        description="Optional override material for hotend mounting collar/clamp (falls back to mount_material)",
    )
    carriage_plate_thickness_mm: float = Field(
        5.0,
        ge=2.0,
        le=15.0,
        description="X-carriage backplate thickness (mm)",
    )
    body_wall_thickness_mm: float = Field(
        4.0,
        ge=2.0,
        le=12.0,
        description="Toolhead/extruder body housing wall thickness (mm)",
    )
    hotend_mount_thickness_mm: float = Field(
        8.0,
        ge=3.0,
        le=20.0,
        description="Hotend mount/clamp collar thickness (mm)",
    )
    body_construction: ToolheadConstructionType | None = Field(
        None,
        description="Structural architecture of the toolhead chassis (when set, derives Stage 2 body stiffness from construction + body_material + body_wall_thickness_mm instead of internal_stiffness_n_per_mm)",
    )
    internal_stiffness_n_per_mm: float = Field(
        450.0,
        ge=25.0,
        le=20000.0,
        description="Overall internal toolhead structural stiffness from linear carriage to nozzle tip (N/mm), modeling flex inside the printed/CNC toolhead body, carriage plate, and hotend clamp.",
    )
    z_probe_type: ZProbeType = Field(
        "static_eddy",
        description="Bed probing carriage architecture: static_eddy (Beacon/Cartographer), klicky_inductive, cnc_tap, or printed_tap",
    )
    envelope_mm: tuple[float, float, float] = (68.0, 74.0, 95.0)
    hotend: HotendSpec = Field(default_factory=HotendSpec)
    extruder: ExtruderSpec = Field(default_factory=ExtruderSpec)

    @property
    def effective_carriage_mount_material(self) -> str:
        return self.carriage_mount_material or self.mount_material

    @property
    def effective_body_material(self) -> str:
        return self.body_material or self.mount_material

    @property
    def effective_hotend_mount_material(self) -> str:
        return self.hotend_mount_material or self.mount_material


# NOTE: total_mass_g specifies toolhead housing + extruder + cooling shroud mass EXCLUDING the hotend
# (_effective_toolhead_mass_kg adds hotend.mass_g, matching DesignPanel.tsx and ToolheadSpec).
TOOLHEAD_PRESETS: dict[str, dict] = {
    "stealthburner": {
        "preset_id": "stealthburner",
        "name": "Voron StealthBurner + CW2",
        "total_mass_g": 365.0,
        "cog_offset_mm": [0.0, -22.0, -28.0],
        "nozzle_offset_mm": [0.0, -26.0, -66.0],
        "mount_material": "abs_printed",
        "body_construction": "multi_piece_clamshell",
        "internal_stiffness_n_per_mm": 350.0,
        "z_probe_type": "static_eddy",
    },
    "stealthburner_tap": {
        "preset_id": "stealthburner_tap",
        "name": "Voron StealthBurner + Printed Tap",
        "total_mass_g": 385.0,
        "cog_offset_mm": [0.0, -22.0, -28.0],
        "nozzle_offset_mm": [0.0, -26.0, -66.0],
        "mount_material": "abs_printed",
        "body_construction": "multi_piece_clamshell",
        "internal_stiffness_n_per_mm": 350.0,
        "z_probe_type": "printed_tap",
    },
    "xol_compact": {
        "preset_id": "xol_compact",
        "name": "XOL / A4T Compact Dual-5015",
        "total_mass_g": 235.0,
        "cog_offset_mm": [0.0, -15.0, -18.0],
        "nozzle_offset_mm": [0.0, -18.0, -54.0],
        "mount_material": "pa12_cf_printed",
        "body_construction": "compact_cage",
        "internal_stiffness_n_per_mm": 650.0,
        "z_probe_type": "static_eddy",
    },
    "a4t_beacon": {
        "preset_id": "a4t_beacon",
        "name": "A4T / XOL 2 + Static Beacon",
        "total_mass_g": 215.0,
        "cog_offset_mm": [0.0, -13.0, -15.0],
        "nozzle_offset_mm": [0.0, -16.0, -52.0],
        "mount_material": "ppa_cf_printed",
        "body_construction": "compact_cage",
        "internal_stiffness_n_per_mm": 1150.0,
        "z_probe_type": "static_eddy",
    },
    "dragon_burner": {
        "preset_id": "dragon_burner",
        "name": "Dragon Burner / Rapid Burner Compact",
        "total_mass_g": 195.0,
        "cog_offset_mm": [0.0, -14.0, -16.0],
        "nozzle_offset_mm": [0.0, -16.0, -50.0],
        "mount_material": "pa12_cf_printed",
        "body_construction": "compact_cage",
        "internal_stiffness_n_per_mm": 720.0,
        "z_probe_type": "static_eddy",
    },
    "archetype_cpap": {
        "preset_id": "archetype_cpap",
        "name": "Archetype / Monolith CPAP Ultra-Low CoG",
        "total_mass_g": 175.0,
        "cog_offset_mm": [0.0, -11.0, -12.0],
        "nozzle_offset_mm": [0.0, -14.0, -48.0],
        "mount_material": "aluminum_6061_t6",
        "body_construction": "plate_sandwich_cf",
        "internal_stiffness_n_per_mm": 1800.0,
        "z_probe_type": "static_eddy",
    },
    "heavy_uhf_long": {
        "preset_id": "heavy_uhf_long",
        "name": "Heavy Long-Nozzle UHF Toolhead",
        "total_mass_g": 440.0,
        "cog_offset_mm": [0.0, -26.0, -34.0],
        "nozzle_offset_mm": [0.0, -28.0, -78.0],
        "mount_material": "abs_printed",
        "body_construction": "multi_piece_clamshell",
        "internal_stiffness_n_per_mm": 250.0,
        "z_probe_type": "static_eddy",
    },
}


class BedSpec(BaseModel):
    """Heated print bed & Z-axis mechanism."""

    size_mm: tuple[float, float] = (300.0, 300.0)
    plate_material: str = "mic6_aluminum"
    plate_thickness_mm: float = Field(8.0, gt=1.0)
    heater_power_w: float = Field(600.0, gt=50.0)
    mass_g: float = Field(2400.0, gt=100.0)
    z_drive: ZDriveType = "belted_quad_z"
    z_rail_profile: ZRailProfileType = Field(
        "mgn12_rail",
        description="Z-axis vertical guideway profile: mgn12_rail, mgn9_rail, steel_rod_8mm/10mm/12mm, or none",
    )
    z_leadscrew_type: ZLeadscrewType = Field(
        "auto",
        description="Z-axis drive screw/belt mechanism (auto resolves from z_drive)",
    )
    target_temp_c: float = Field(60.0, ge=20.0, le=160.0, description="Heated bed target temperature (C)")
    mount_type: BedMountType = Field(
        "kinematic_3pt",
        description=(
            "Bed frame attachment constraint: kinematic_3pt (Maxwell), whopping_pochard_kinematic, "
            "trident_3pt_standoffs, rigid_4pt, floating_4pt, silicone_4pt, or springs_4pt"
        ),
    )

    @property
    def effective_z_leadscrew_type(self) -> str:
        """Resolve 'auto' Z-drive mechanism from z_drive architecture."""
        if self.z_leadscrew_type != "auto":
            return self.z_leadscrew_type
        if self.z_drive == "belted_quad_z":
            return "gt2_6mm_z_belt"
        if self.z_drive == "leadscrew_triple":
            return "tr8x4_pom"
        return "tr8x8_pom"


class PrinterSpec(BaseModel):
    """Root declarative specification for a complete 3D printer design."""

    name: str = "Custom 3D Printer"
    description: str = ""
    architecture: Architecture = "corexy"
    frame: FrameSpec = Field(default_factory=FrameSpec)
    gantry: GantrySpec = Field(default_factory=GantrySpec)
    kinematics: KinematicsSpec = Field(default_factory=KinematicsSpec)
    toolhead: ToolheadSpec = Field(default_factory=ToolheadSpec)
    bed: BedSpec = Field(default_factory=BedSpec)
    custom_meshes: list[CustomMeshSpec] = Field(default_factory=list)

    @property
    def is_flying_gantry(self) -> bool:
        """True for flying-gantry CoreXY machines (e.g. Voron 2.4 quad-belted Z) where the gantry moves in Z."""
        return self.architecture == "corexy" and self.bed.z_drive == "belted_quad_z"

    @property
    def panels_bolted_to_gantry_effective(self) -> bool:
        """True when bolted structural panels can and do tie directly into fixed gantry beams."""
        return (
            self.architecture == "corexy"
            and self.frame.enclosed_panels
            and self.frame.panel_mounting == "bolted_structural"
            and self.frame.bolt_panels_to_gantry
            and not self.is_flying_gantry
        )

    @model_validator(mode="after")
    def _validate_build_fits_frame(self) -> "PrinterSpec":
        # Warn/enforce only absurd negative clearances; collision engine handles fine clearances
        bx, by, bz = self.kinematics.build_volume_mm
        if bx <= 0 or by <= 0 or bz <= 0:
            raise ValueError("build_volume_mm dimensions must be positive")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PrinterSpec":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)

    def save_yaml(self, path: str | Path) -> None:
        Path(path).write_text(self.to_yaml(), encoding="utf-8")
