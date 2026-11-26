"""Parametric 3D printer assembly DSL (YAML / JSON / Python)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from .overrides import ModelOverridesSpec

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
TopBraceType = Literal[
    "none",
    "cnc_halo_ring",
    "rear_v_and_front_corners",
    "corner_45_bars",
]
FrontBraceType = Literal[
    "none",
    "cnc_portal_frame",
    "header_and_skirt",
    "corner_45_bars",
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
    "pc_pbt_gf_printed",
    "ppa_cf_printed",
    "pps_cf_printed",
    "slm_aluminum",
    "slm_ti6al4v",
    "slm_maraging_steel",
    "slm_316l_steel",
    "slm_17_4ph_steel",
    "sheet_metal_steel",
    "aluminum_6061_t6",
    "aluminum_7075_t6",
]
ChamberHeaterType = Literal["none", "monolith_100w_ptc", "monolith_500w_ptc"]
CpapBlowerType = Literal["none", "ws7040_24v", "wm9290_24v", "wm7575_220vac"]
StepperThermalGasketType = Literal["none", "ptfe_2mm"]
IdlerAxialPreloadType = Literal["auto", "none", "disc_spring_m5"]
ZJointType = Literal["single", "double"]
ZJointMaterialType = Literal["printed", "hybrid", "slm", "milled"]
DriveModeType = Literal["2wd", "awd"]
XRailMountType = Literal["front", "top", "bottom", "dual_top_bottom"]
YRailMountType = Literal["top", "inner", "bottom"]
DriveType = Literal["open_loop_stepper", "closed_loop_stepper", "foc_servo"]
AwdSyncMode = Literal["none", "torque_mirror", "position_equalizer"]
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
    "smooth_flatness",
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
CFRailMountingType = Literal["auto", "single_wall_bare", "two_wall_clamped_inserts"]
BedMountType = Literal[
    "kinematic_3pt",
    "whopping_pochard_kinematic",
    "trident_3pt_standoffs",
    "cnc_kelvin_magnetic",
    "captive_belleville_kinematic",
    "dual_leaf_ti_kinematic",
    "spherical_3pt_ratrig",
    "craz_kinematic_ge5c",
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
    "gt2_10mm",
    "gt3_10mm",
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
        "ea_static_n": 18900.0,  # Exact Gates PowerGrip 2GT quasi-static catalog EA (18,000 lbf/in * 6/25.4 mm)
        "dynamic_stiffening_factor": 2.0,  # 40-150 Hz preloaded small-amplitude EPDM/fiberglass dynamic tangent ratio
        "ea_default_core_n": 37800.0,  # ea_static_n * dynamic_stiffening_factor
        "linear_mass_kg_m": 0.0083,
        "max_working_tension_n": 58.0,
        "tooth_shear_k_n_per_m_mm": 5.5e4,
        "min_pulley_teeth": 16,
        "min_back_idler_mm": 12.0,
        "max_temp_c": 110.0,
        "cost_usd_per_m": 4.50,
        "validation_tier": "proven",
        "theoretical_summary": "Gates PowerGrip 2GT 6mm (18.9 kN quasi-static datasheet EA x 2.0 dynamic small-amplitude factor, exact pulley-to-pulley spans).",
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
        "ea_static_n": 28350.0,  # Exact Gates PowerGrip 2GT 9mm quasi-static catalog EA (1.5x 6mm)
        "dynamic_stiffening_factor": 2.0,
        "ea_default_core_n": 56700.0,  # ea_static_n * dynamic_stiffening_factor (below steel-cord AT3 80 kN)
        "linear_mass_kg_m": 0.01245,
        "max_working_tension_n": 92.8,
        "tooth_shear_k_n_per_m_mm": 5.5e4,
        "min_pulley_teeth": 16,
        "min_back_idler_mm": 12.0,
        "max_temp_c": 110.0,
        "cost_usd_per_m": 6.50,
        "validation_tier": "proven",
        "theoretical_summary": "Gates PowerGrip 2GT 9mm (28.35 kN quasi-static datasheet EA x 2.0 dynamic factor, exact pulley-to-pulley spans).",
        "practical_issues": [],
    },
    "gt2_10mm": {
        "id": "gt2_10mm",
        "label": "Gates 2GT 10mm (Monolith)",
        "profile_family": "2gt",
        "pitch_mm": 2.0,
        "width_mm": 10.0,
        "hardware_width_mm": 10.0,
        "default_core": "fiberglass",
        "body_material": "epdm",
        "ea_static_n": 31500.0,  # Exact Gates PowerGrip 2GT 10mm quasi-static catalog EA (10/9 x 9mm)
        "dynamic_stiffening_factor": 2.0,
        "ea_default_core_n": 63000.0,
        "linear_mass_kg_m": 0.01236,  # Scale-weighed Monolith 2x1658 mm = 41.0 g (12.36 g/m)
        "max_working_tension_n": 105.0,
        "tooth_shear_k_n_per_m_mm": 5.5e4,
        "elastomer_seating_factor": 1.0,
        "rolling_drag_mult": 1.0,
        "min_pulley_teeth": 16,
        "min_back_idler_mm": 12.0,
        "max_temp_c": 110.0,
        "cost_usd_per_m": 7.20,
        "validation_tier": "proven",
        "theoretical_summary": "Gates PowerGrip 2GT 10mm EPDM fiberglass (scale-weighed 12.36 g/m; Monolith Road-to-50k baseline: 113.8 Hz ZV @ 283 Hz / ~89 N).",
        "practical_issues": [],
    },
    "gt3_10mm": {
        "id": "gt3_10mm",
        "label": "Gates GT3 (2MGT) 10mm",
        "profile_family": "2mgt",
        "pitch_mm": 2.0,
        "width_mm": 10.0,
        "hardware_width_mm": 10.0,
        "default_core": "fiberglass",
        "body_material": "epdm",
        "ea_static_n": 31500.0,  # Same tension cords as 2GT, softer elastomer matrix
        "dynamic_stiffening_factor": 1.94,
        "ea_default_core_n": 61110.0,
        "linear_mass_kg_m": 0.01499,  # Scale-weighed Monolith 2x1658 mm = 49.7 g (14.99 g/m)
        "max_working_tension_n": 105.0,
        "tooth_shear_k_n_per_m_mm": 4.9e4,
        "elastomer_seating_factor": 0.90,  # Softer elastomer needs more tension/elongation to reach cord stiffness
        "rolling_drag_mult": 1.18,  # Higher pulley-wrap hysteresis & rolling resistance (Monolith belt shootout)
        "min_pulley_teeth": 16,
        "min_back_idler_mm": 12.0,
        "max_temp_c": 110.0,
        "cost_usd_per_m": 8.00,
        "validation_tier": "proven",
        "theoretical_summary": "Gates GT3 2MGT 10mm (scale-weighed 14.99 g/m; Monolith matched-tension shootout: 110.6 Hz ZV @ 257 Hz pluck vs 113.8 Hz for GT2 EPDM due to softer base elastomer & higher rolling drag).",
        "practical_issues": [
            "Softer base elastomer requires higher tension/elongation to match GT2 EPDM cord stiffness and increases pulley rolling resistance.",
        ],
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
        "ea_static_n": 37800.0,  # Exact Gates PowerGrip 2GT 12mm quasi-static catalog EA (2.0x 6mm)
        "dynamic_stiffening_factor": 2.0,
        "ea_default_core_n": 75600.0,  # ea_static_n * dynamic_stiffening_factor
        "linear_mass_kg_m": 0.0166,
        "max_working_tension_n": 127.6,
        "tooth_shear_k_n_per_m_mm": 5.5e4,
        "min_pulley_teeth": 16,
        "min_back_idler_mm": 12.0,
        "max_temp_c": 110.0,
        "cost_usd_per_m": 8.50,
        "validation_tier": "proven",
        "theoretical_summary": "Gates PowerGrip 2GT 12mm (37.8 kN quasi-static datasheet EA x 2.0 dynamic factor) on high-tension AWD CoreXY builds.",
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
        "ea_static_n": 47250.0,
        "dynamic_stiffening_factor": 2.0,
        "ea_default_core_n": 94500.0,
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
        "ea_static_n": 56700.0,
        "dynamic_stiffening_factor": 2.0,
        "ea_default_core_n": 113400.0,
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
        "ea_static_n": 47250.0,  # 30/18 = 5/3x Gates 3GT/2GT fiberglass modulus ratio
        "dynamic_stiffening_factor": 2.0,
        "ea_default_core_n": 94500.0,
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
        "ea_static_n": 63000.0,
        "dynamic_stiffening_factor": 2.0,
        "ea_default_core_n": 126000.0,
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
        "ea_static_n": 78750.0,
        "dynamic_stiffening_factor": 2.0,
        "ea_default_core_n": 157500.0,
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
        "ea_static_n": 80000.0,  # Exact Elatech/SIT C_spez = 80 kN (Hookean steel cords -> dynamic factor 1.0)
        "dynamic_stiffening_factor": 1.0,
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
from printsim.schema.printdb_bridge import (
    hydrate_bed_mount_catalog as _hydrate_bed_mount_catalog,
    hydrate_belt_catalog as _hydrate_belt_catalog,
    hydrate_hotend_catalog as _hydrate_hotend_catalog,
    hydrate_stepper_catalog as _hydrate_stepper_catalog,
    hydrate_z_drive_catalog as _hydrate_z_drive_catalog,
)

_hydrate_belt_catalog(BELT_CATALOG)

HotendMountStyleType = Literal[
    "groove_mount", "rigid_4bolt_m3", "dragon_4bolt_m25", "chube_6bolt_m3_18pcd"
]
HOTEND_MOUNT_CLAMP_BASE_N_PER_MM: dict[str, float] = {
    "groove_mount": 800.0,
    "rigid_4bolt_m3": 2200.0,
    "dragon_4bolt_m25": 3200.0,
    "chube_6bolt_m3_18pcd": 6500.0,
}
# Corner joint base stiffness for 2020 Al extrusion: (k_t_n_m, k_r_inplane_nm_rad, k_r_torsion_nm_rad)
CORNER_JOINT_BASE_2020: dict[str, tuple[float, float, float]] = {
    "l_bracket": (3.5e6, 4500.0, 1800.0),
    "blind_joint": (1.2e7, 22000.0, 7500.0),
    "gusset_plate": (2.2e7, 42000.0, 14000.0),
    "corner_cube": (3.8e7, 85000.0, 28000.0),
    "rigid": (1.0e10, 1.0e9, 1.0e9),
}
CORNER_JOINT_IWAN_DAMPING: dict[str, float] = {
    "rigid": 0.0,
    "corner_cube": 0.001,
    "gusset_plate": 0.002,
    "blind_joint": 0.003,
    "l_bracket": 0.005,
}
RAIL_PRELOAD_FRACTIONS: dict[str, float] = {
    "ZF": 0.002,
    "Z0": 0.010,
    "Z1": 0.030,
    "ZA": 0.060,
    "ZB": 0.100,
}
BEARING_RADIAL_PLAY_UM: dict[str, float] = {
    "C2": 3.5,
    "CN": 7.5,
    "C3": 15.5,
}
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
    "tr8x4_wobblex",
    "tr8x8_pom",
    "tr8x8_brass",
    "sfu1204_bk10",
    "sfu1204_wobblex",
    "sfu1605_fk12",
    "gt2_6mm_z_belt",
    "gt2_9mm_z_belt",
    "gt2_9mm_planetary_z",
    "gt2_9mm_worm_z",
    "kevinakasam_belted_z",
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
        "mount_style": "rigid_4bolt_m3",
        "structural_stiffness_n_per_mm": 2975.0,
        "mass_g": 63.0,
        "overall_length_mm": 47.2,
        "melt_zone_length_mm": 33.0,
        "heater_power_w": 80.0,
        "block_material": "copper_c110",
        "block_mass_g": 18.0,
        "nozzle_material": "copper_c110",
        "heatbreak_material": "titanium_grade5",
        "heatbreak_wall_mm": 0.25,
        "heatbreak_length_mm": 10.0,
        "heatbreak_id_mm": 2.0,
        "heatsink_thermal_resistance_k_w": 2.6,
        "max_temp_c": 500.0,
        "cost_usd": 170.0,
        "notes": "Official SolidWorks STEP (Chube_Compact_integration_assembly.STEP): 47.2 mm OAL w/ V6 nozzle, Ø22.23×0.89 mm Ti monocoque shell w/ exact 4× Ø12.50 mm cooling cutouts @ 45/135/225/315°, 4× M3 Ø16 mm Dragon PCD mount, 33 mm meltzone, Chalice Ti heatbreak, 50 g bare / 63 g dressed; 49.3k-elem Tet4 FEA calibrated stiffness 3,020/2,931 N/mm (short 47.2 mm lever arm + solid lower shell)",
    },
    "chube_air": {
        "id": "chube_air",
        "preset_id": "chube_air",
        "name": "Luke's Lab Chube Air (100W Ti Shell)",
        "short_name": "Chube Air",
        "mount_style": "chube_6bolt_m3_18pcd",
        "structural_stiffness_n_per_mm": 560.0,
        "mass_g": 79.0,
        "overall_length_mm": 66.0,
        "melt_zone_length_mm": 42.0,
        "heater_power_w": 100.0,
        "block_material": "copper_c110",
        "block_mass_g": 26.0,
        "nozzle_material": "copper_c110",
        "heatbreak_material": "titanium_grade5",
        "heatbreak_wall_mm": 0.25,
        "heatbreak_length_mm": 11.0,
        "heatbreak_id_mm": 2.0,
        "heatsink_thermal_resistance_k_w": 2.5,
        "max_temp_c": 500.0,
        "cost_usd": 260.0,
        "notes": "Official OEM STEP & Rev A drawing (CHUBE_INTEGRATION_REVA.STEP): 66.0 mm OAL w/ V6 nozzle (60.0 mm bare), Ø25.4×0.90 mm Ti monocoque shell w/ exact 6× staggered Ø19.0 mm cutouts, 6× M3 Ø18 mm PCD, 42 mm meltzone, 6×21 mm heater, 69 g bare (66.9 g FEA) / 79 g dressed, 500 °C rated; 70.7k-elem Tet4 FEA calibrated stiffness 549/568 N/mm",
    },
    "chube_conduction": {
        "id": "chube_conduction",
        "preset_id": "chube_conduction",
        "name": "Luke's Lab Chube Conduction (120W Ti Shell)",
        "short_name": "Chube Conduction",
        "mount_style": "chube_6bolt_m3_18pcd",
        "structural_stiffness_n_per_mm": 815.0,
        "mass_g": 90.0,
        "overall_length_mm": 66.0,
        "melt_zone_length_mm": 52.0,
        "heater_power_w": 120.0,
        "block_material": "copper_c110",
        "block_mass_g": 34.0,
        "nozzle_material": "copper_c110",
        "heatbreak_material": "titanium_grade5",
        "heatbreak_wall_mm": 0.25,
        "heatbreak_length_mm": 10.0,
        "heatbreak_id_mm": 2.0,
        "heatsink_thermal_resistance_k_w": 2.2,
        "max_temp_c": 500.0,
        "cost_usd": 320.0,
        "notes": "Official OEM STEP & Rev A drawing (CHUBE_INTEGRATION_REVA.STEP): 66.0 mm OAL w/ V6 nozzle (60.0 mm bare), Ø25.4×0.90 mm Ti monocoque shell w/ exact 6× staggered Ø19.0 mm cutouts + 8 mm solid CNC Al conduction top block, 6× M3 Ø18 mm PCD (requires ≥80 cm² metal mount), 52 mm meltzone, 6×46.5 mm 120W Supervolcano heater, 74 g bare (78.2 g FEA, CoM -35.2 mm) / 90 g dressed, 500 °C rated; 80.2k-elem Tet4 FEA calibrated stiffness 801/830 N/mm",
    },
    "tricorn_short": {
        "id": "tricorn_short",
        "preset_id": "tricorn_short",
        "name": "Tricorn Short (120W 3×40W Inverted Ti)",
        "short_name": "Tricorn Short",
        "mount_style": "rigid_4bolt_m3",
        "structural_stiffness_n_per_mm": 440.0,
        "mass_g": 50.0,
        "overall_length_mm": 54.0,
        "melt_zone_length_mm": 41.0,
        "heater_power_w": 120.0,
        "block_material": "copper_c110",
        "block_mass_g": 23.0,
        "nozzle_material": "copper_c110",
        "heatbreak_material": "titanium_grade5",
        "heatbreak_wall_mm": 0.45,
        "heatbreak_length_mm": 13.0,
        "heatbreak_id_mm": 1.9,
        "heatsink_thermal_resistance_k_w": 2.6,
        "max_temp_c": 450.0,
        "cost_usd": 215.0,
        "notes": "Open-source Okano Tricorn Short (STEP CAD: 54 mm overall, 41 mm melt zone w/ V6 nozzle, 23 g CuCrZr block, 3×40W 4×40mm heaters, inverted Ti-6Al-4V heatbreak 0.45 mm wall, SLM Ti crown + M3 Dragon mount); 112k-elem Tet4 FEA calibrated unbraced stiffness 437/442 N/mm",
    },
    "tricorn_long": {
        "id": "tricorn_long",
        "preset_id": "tricorn_long",
        "name": "Tricorn Long (120W 3×40W Inverted Ti)",
        "short_name": "Tricorn Long",
        "mount_style": "rigid_4bolt_m3",
        "structural_stiffness_n_per_mm": 265.0,
        "mass_g": 62.0,
        "overall_length_mm": 66.0,
        "melt_zone_length_mm": 53.5,
        "heater_power_w": 120.0,
        "block_material": "copper_c110",
        "block_mass_g": 31.4,
        "nozzle_material": "copper_c110",
        "heatbreak_material": "titanium_grade5",
        "heatbreak_wall_mm": 0.45,
        "heatbreak_length_mm": 13.0,
        "heatbreak_id_mm": 1.9,
        "heatsink_thermal_resistance_k_w": 2.4,
        "max_temp_c": 450.0,
        "cost_usd": 230.0,
        "notes": "Open-source Okano Tricorn Long (STEP CAD: 66 mm overall, 53.5 mm melt zone w/ V6 nozzle, 31.4 g CuCrZr block, 3×40W 4×40mm heaters, inverted Ti-6Al-4V heatbreak 0.45 mm wall, SLM Ti crown/shield + M3 Dragon mount); 116k-elem Tet4 FEA calibrated unbraced stiffness 262/264 N/mm (higher w/ Microwaved Ti bottom brace)",
    },
    "monolith_50mm_hf": {
        "id": "monolith_50mm_hf",
        "preset_id": "monolith_50mm_hf",
        "name": "Monolith 50mm+V6 HF Hotend (80W PT1000)",
        "short_name": "Monolith 50mm HF",
        "mount_style": "rigid_4bolt_m3",
        "structural_stiffness_n_per_mm": 920.0,
        "mass_g": 52.0,
        "overall_length_mm": 68.0,
        "melt_zone_length_mm": 50.0,
        "heater_power_w": 80.0,
        "block_material": "copper_c110",
        "block_mass_g": 22.0,
        "nozzle_material": "copper_c110",
        "heatbreak_material": "titanium_grade5",
        "heatbreak_wall_mm": 0.25,
        "heatbreak_length_mm": 10.5,
        "heatbreak_id_mm": 2.0,
        "heatsink_thermal_resistance_k_w": 2.4,
        "max_temp_c": 350.0,
        "cost_usd": 145.0,
        "notes": "Monolith Toolhead 50 mm + V6 cloverleaf HF meltzone with vertical-preload threadless heatbreak & PT1000; bench-tested to 83 mm³/s LDO ABS @ 300 °C / 85 °C chamber (80 W heater, ~90% PWM).",
    },
}
_hydrate_hotend_catalog(HOTEND_CATALOG)

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
        "wobble_decoupled": False,
        "self_locking": True,
        "unpowered_drop_risk": False,
        "gear_ratio": 1.0,
        "cost_per_axis_usd": 19.5,
        "notes": "Voron Trident standard 2-start 4 mm lead screw with anti-backdrive POM flanged nut",
    },
    "tr8x4_wobblex": {
        "id": "tr8x4_wobblex",
        "name": "TR8x4 Leadscrew + WobbleX Decoupler",
        "short_name": "TR8x4 + WobbleX",
        "kind": "leadscrew",
        "lead_mm": 4.0,
        "root_area_m2": 2.38e-5,
        "k_nut_n_per_um": 10.2,
        "k_thrust_n_per_um": 18.0,
        "k_drive_n_per_um_per_corner": None,
        "backlash_um": 18.0,
        "mass_kg_per_m": 0.31,
        "diameter_mm": 8.0,
        "wobble_decoupled": True,
        "self_locking": True,
        "unpowered_drop_risk": False,
        "gear_ratio": 1.0,
        "cost_per_axis_usd": 34.0,
        "notes": "TR8x4 leadscrew paired with MirageC WobbleX WS8 self-aligning pin-and-ball Oldham decoupler to eliminate Z-banding",
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
        "wobble_decoupled": False,
        "self_locking": False,
        "unpowered_drop_risk": True,
        "gear_ratio": 1.0,
        "cost_per_axis_usd": 19.5,
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
        "wobble_decoupled": False,
        "self_locking": False,
        "unpowered_drop_risk": True,
        "gear_ratio": 1.0,
        "cost_per_axis_usd": 22.0,
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
        "wobble_decoupled": False,
        "self_locking": False,
        "unpowered_drop_risk": True,
        "gear_ratio": 1.0,
        "cost_per_axis_usd": 52.0,
        "notes": "12 mm rolled ballscrew with recirculating steel ball nut and preloaded BK10 angular-contact thrust block",
    },
    "sfu1204_wobblex": {
        "id": "sfu1204_wobblex",
        "name": "SFU1204 Ballscrew + BK10 + WobbleX (V-Core 4 / Trident)",
        "short_name": "SFU1204 + WobbleX",
        "kind": "ballscrew",
        "lead_mm": 4.0,
        "root_area_m2": 7.09e-5,
        "k_nut_n_per_um": 72.0,
        "k_thrust_n_per_um": 150.0,
        "k_drive_n_per_um_per_corner": None,
        "backlash_um": 6.0,
        "mass_kg_per_m": 0.72,
        "diameter_mm": 12.0,
        "wobble_decoupled": True,
        "self_locking": False,
        "unpowered_drop_risk": True,
        "gear_ratio": 1.0,
        "cost_per_axis_usd": 68.0,
        "notes": "Rolled SFU1204 ballscrew + BK10 angular-contact thrust block paired with WobbleX XY decoupler to isolate screw runout",
    },
    "sfu1605_fk12": {
        "id": "sfu1605_fk12",
        "name": "SFU1605 Heavy Ballscrew + FK12/BK12",
        "short_name": "SFU1605 + FK12",
        "kind": "ballscrew",
        "lead_mm": 5.0,
        "root_area_m2": 1.33e-4,
        "k_nut_n_per_um": 145.0,
        "k_thrust_n_per_um": 240.0,
        "k_drive_n_per_um_per_corner": None,
        "backlash_um": 5.0,
        "mass_kg_per_m": 1.25,
        "diameter_mm": 16.0,
        "wobble_decoupled": False,
        "self_locking": False,
        "unpowered_drop_risk": True,
        "gear_ratio": 1.0,
        "cost_per_axis_usd": 82.0,
        "notes": "Heavy-duty 16 mm C7 ballscrew with FK12/BK12 preloaded thrust bearings for large high-mass moving beds",
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
        "wobble_decoupled": True,
        "self_locking": False,
        "unpowered_drop_risk": True,
        "gear_ratio": 5.0,
        "cost_per_axis_usd": 28.0,
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
        "wobble_decoupled": True,
        "self_locking": False,
        "unpowered_drop_risk": True,
        "gear_ratio": 5.0,
        "cost_per_axis_usd": 36.0,
        "notes": "Heavy-duty 9 mm fiberglass GT2 Z-belt upgrade (+50% Z-corner axial stiffness vs 6 mm belt)",
    },
    "gt2_9mm_planetary_z": {
        "id": "gt2_9mm_planetary_z",
        "name": "Trident / Moving-Bed 9mm Belted-Z (5:1 Planetary)",
        "short_name": "9mm Planetary Z",
        "kind": "belt",
        "lead_mm": 8.0,
        "root_area_m2": 1.44e-5,
        "k_nut_n_per_um": 0.0,
        "k_thrust_n_per_um": 0.0,
        "k_drive_n_per_um_per_corner": 0.31,
        "backlash_um": 14.0,
        "mass_kg_per_m": 0.025,
        "diameter_mm": 9.0,
        "wobble_decoupled": True,
        "self_locking": False,
        "unpowered_drop_risk": True,
        "gear_ratio": 5.0,
        "cost_per_axis_usd": 54.0,
        "notes": "Moving-bed 9 mm GT2 belted-Z conversion driven through compact 5:1 planetary gearboxes: zero Z-wobble and high lift speed",
    },
    "gt2_9mm_worm_z": {
        "id": "gt2_9mm_worm_z",
        "name": "9mm Belted-Z + Self-Locking Worm / 20:1 Gearbox",
        "short_name": "9mm Worm Belted Z",
        "kind": "belt",
        "lead_mm": 2.0,
        "root_area_m2": 1.44e-5,
        "k_nut_n_per_um": 0.0,
        "k_thrust_n_per_um": 0.0,
        "k_drive_n_per_um_per_corner": 0.34,
        "backlash_um": 16.0,
        "mass_kg_per_m": 0.025,
        "diameter_mm": 9.0,
        "wobble_decoupled": True,
        "self_locking": True,
        "unpowered_drop_risk": False,
        "gear_ratio": 20.0,
        "cost_per_axis_usd": 66.0,
        "notes": "9 mm GT2 belted-Z with 20:1 self-locking worm reducer: zero leadscrew wobble, highest belted-Z axial stiffness, and zero unpowered drop",
    },
    "kevinakasam_belted_z": {
        "id": "kevinakasam_belted_z",
        "name": "KevinAkaSam Belted-Z (80:16 5:1 + Keybak)",
        "short_name": "KevinAkaSam Z",
        "kind": "belt",
        "lead_mm": 8.0,
        "root_area_m2": 9.6e-6,
        "k_nut_n_per_um": 0.0,
        "k_thrust_n_per_um": 0.0,
        "k_drive_n_per_um_per_corner": 0.21,
        "backlash_um": 15.0,
        "mass_kg_per_m": 0.017,
        "diameter_mm": 6.0,
        "wobble_decoupled": True,
        "self_locking": False,
        "unpowered_drop_risk": False,
        "gear_ratio": 5.0,
        "cost_per_axis_usd": 32.0,
        "notes": "Popular open-source 80T:16T (5:1) dual-loop 6 mm GT2 belted-Z mod with Keybak retractable counterbalance preventing unpowered drop",
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
        "k_restraint_thermal_n_per_m": 0.0,
        "k_restraint_thermal": 0.0,  # deprecated alias
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
        "k_restraint_thermal_n_per_m": 0.0,
        "k_restraint_thermal": 0.0,  # deprecated alias
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
        "k_restraint_thermal_n_per_m": 3.8e6,
        "k_restraint_thermal": 22.0,  # deprecated alias
        "damping_delta": 0.003,
        "standoff_height_mm": 10.0,
        "radial_dof_mode": "slotted_3pt_bakelite",
        "tap_rated": True,
        "mount_inset_ratio": 0.38,
        "cost_usd": 16.0,
        "notes": "Stock Voron/Siboor Trident 3-point Bakelite thermal-isolation standoffs on 2020 T-subframe with GE5C spherical Z-joints",
    },
    "cnc_kelvin_magnetic": {
        "id": "cnc_kelvin_magnetic",
        "name": "CNC N48SH Magnetic Kelvin Kinematic (MRW / ChaoticLab)",
        "short_name": "CNC Kelvin Mag",
        "mount_points": 3,
        "k_axial_n_per_mm": 42000.0,
        "k_lateral_n_per_mm": 30000.0,
        "k_restraint_thermal_n_per_m": 0.0,
        "k_restraint_thermal": 0.0,  # deprecated alias
        "damping_delta": 0.001,
        "standoff_height_mm": 15.0,
        "radial_dof_mode": "kelvin_cone_vee_flat",
        "tap_rated": False,
        "mount_inset_ratio": 0.38,
        "cost_usd": 75.0,
        "notes": "3-point CNC aluminum Kelvin coupling (cone + V-groove + flat) with hardened steel balls and 150C N48SH magnetic preload: zero thermal taco and very high Hertzian Z stiffness",
    },
    "captive_belleville_kinematic": {
        "id": "captive_belleville_kinematic",
        "name": "Super-Stiff Captive Belleville Kinematic (Tap-Rated)",
        "short_name": "Captive Belleville",
        "mount_points": 3,
        "k_axial_n_per_mm": 52000.0,
        "k_lateral_n_per_mm": 36000.0,
        "k_restraint_thermal_n_per_m": 0.0,
        "k_restraint_thermal": 0.0,  # deprecated alias
        "damping_delta": 0.002,
        "standoff_height_mm": 16.0,
        "radial_dof_mode": "maxwell_vgroove",
        "tap_rated": True,
        "mount_inset_ratio": 0.38,
        "cost_usd": 95.0,
        "notes": "Hardened steel ball-in-V-groove kinematic contacts mechanically preloaded with captive shoulder bolts and Belleville disc springs: highest kinematic Z stiffness, zero thermal taco, fully Tap-rated",
    },
    "dual_leaf_ti_kinematic": {
        "id": "dual_leaf_ti_kinematic",
        "name": "Dual-Blade Ti/301 Leaf-Spring Kinematic (Annex / Heavy Pochard)",
        "short_name": "Dual-Blade Ti Leaf",
        "mount_points": 3,
        "k_axial_n_per_mm": 38000.0,
        "k_lateral_n_per_mm": 26000.0,
        "k_restraint_thermal_n_per_m": 0.0,
        "k_restraint_thermal": 0.0,  # deprecated alias
        "damping_delta": 0.003,
        "standoff_height_mm": 14.0,
        "radial_dof_mode": "radial_flexure",
        "tap_rated": True,
        "mount_inset_ratio": 0.38,
        "cost_usd": 88.0,
        "notes": "Stacked twin spring-steel/Ti-6Al-4V radial leaf flexures: +58% higher Z stiffness than single-blade Pochard with zero play, zero thermal taco, and Tap rating",
    },
    "spherical_3pt_ratrig": {
        "id": "spherical_3pt_ratrig",
        "name": "RatRig V-Core 3/4 3-Pt Spherical Pivot Kinematic",
        "short_name": "RatRig Spherical 3-Pt",
        "mount_points": 3,
        "k_axial_n_per_mm": 30000.0,
        "k_lateral_n_per_mm": 20000.0,
        "k_restraint_thermal_n_per_m": 1.5e6,
        "k_restraint_thermal": 8.0,  # deprecated alias
        "damping_delta": 0.002,
        "standoff_height_mm": 15.0,
        "radial_dof_mode": "radial_flexure",
        "tap_rated": True,
        "mount_inset_ratio": 0.38,
        "cost_usd": 58.0,
        "notes": "3-point spherical ball-socket Z-arm pivots with radial thermal glide slot: eliminates tramming bind on large 300-500mm beds with high vertical Tap stiffness",
    },
    "craz_kinematic_ge5c": {
        "id": "craz_kinematic_ge5c",
        "name": "Monolith Cra-Z 2+2+2 Double-Wishbone GE5C",
        "short_name": "Monolith Cra-Z",
        "mount_points": 3,
        "k_axial_n_per_mm": 34000.0,
        "k_lateral_n_per_mm": 24000.0,
        "k_restraint_thermal_n_per_m": 0.0,
        "k_restraint_thermal": 0.0,
        "damping_delta": 0.002,
        "standoff_height_mm": 15.0,
        "radial_dof_mode": "double_wishbone_ge5c",
        "tap_rated": True,
        "mount_inset_ratio": 0.38,
        "cost_usd": 78.0,
        "notes": "Monolith SLM 2+2+2 double-wishbone bed suspension with GE5C spherical bearings close to Z rails + cylindrical bed-edge joints, gravity-preloaded in XY; zero thermal taco bow, drop-in for 3-screw Trident or 4-belt fixed-gantry V2.",
    },
    "rigid_4pt": {
        "id": "rigid_4pt",
        "name": "4-Point Rigid Hard-Bolted Standoffs",
        "short_name": "Rigid 4-Pt",
        "mount_points": 4,
        "k_axial_n_per_mm": 65000.0,
        "k_lateral_n_per_mm": 40000.0,
        "k_restraint_thermal_n_per_m": 8.0e7,  # B-13: 2x opposing rigid steel standoff lateral pair (N/m)
        "k_restraint_thermal": 8000.0,  # deprecated alias
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
        "k_restraint_thermal_n_per_m": 2.8e6,
        "k_restraint_thermal": 18.0,  # deprecated alias
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
        "k_restraint_thermal_n_per_m": 32000.0,
        "k_restraint_thermal": 12.0,  # deprecated alias
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
        "k_restraint_thermal_n_per_m": 9000.0,
        "k_restraint_thermal": 8.0,  # deprecated alias
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
    "pc_pbt_gf_printed": {
        "id": "pc_pbt_gf_printed",
        "name": "3D-Printed PC-PBT-GF (4.6 GPa, 132°C HDT)",
        "short_name": "Printed PC-PBT-GF",
        "youngs_modulus_gpa": 4.6,
        "hdt_c": 132.0,
        "creep_tension_limit_n": 72.0,
        "cnc_machined": False,
        "kit_cost_usd": 78.0,
        "carriage_mass_delta_g": 8.0,
        "color_hex": "#334155",
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
    "slm_aluminum": {
        "id": "slm_aluminum",
        "name": "SLM 3D-Printed Aluminum AlSi10Mg (68 GPa)",
        "short_name": "SLM AlSi10Mg",
        "youngs_modulus_gpa": 68.0,
        "hdt_c": 300.0,
        "creep_tension_limit_n": 230.0,
        "cnc_machined": True,
        "manufacturing_process": "slm",
        "kit_cost_usd": 135.0,
        "carriage_mass_delta_g": 10.0,
        "color_hex": "#b8c4d0",
    },
    "slm_ti6al4v": {
        "id": "slm_ti6al4v",
        "name": "SLM 3D-Printed Titanium Ti-6Al-4V (110 GPa)",
        "short_name": "SLM Ti-6Al-4V",
        "youngs_modulus_gpa": 110.0,
        "hdt_c": 450.0,
        "creep_tension_limit_n": 340.0,
        "cnc_machined": True,
        "manufacturing_process": "slm",
        "kit_cost_usd": 295.0,
        "carriage_mass_delta_g": 28.0,
        "color_hex": "#7c7d8c",
    },
    "slm_maraging_steel": {
        "id": "slm_maraging_steel",
        "name": "SLM Maraging Steel MS1 1.2709 (180 GPa)",
        "short_name": "SLM MS1 Steel",
        "youngs_modulus_gpa": 180.0,
        "hdt_c": 450.0,
        "creep_tension_limit_n": 380.0,
        "cnc_machined": True,
        "manufacturing_process": "slm",
        "kit_cost_usd": 275.0,
        "carriage_mass_delta_g": 64.0,
        "color_hex": "#6b7280",
    },
    "slm_316l_steel": {
        "id": "slm_316l_steel",
        "name": "SLM 316L Stainless Steel (185 GPa)",
        "short_name": "SLM 316L Steel",
        "youngs_modulus_gpa": 185.0,
        "hdt_c": 450.0,
        "creep_tension_limit_n": 340.0,
        "cnc_machined": True,
        "manufacturing_process": "slm",
        "kit_cost_usd": 220.0,
        "carriage_mass_delta_g": 62.0,
        "color_hex": "#94a3b8",
    },
    "slm_17_4ph_steel": {
        "id": "slm_17_4ph_steel",
        "name": "SLM 17-4PH Stainless Steel H900 (190 GPa)",
        "short_name": "SLM 17-4PH",
        "youngs_modulus_gpa": 190.0,
        "hdt_c": 450.0,
        "creep_tension_limit_n": 380.0,
        "cnc_machined": True,
        "manufacturing_process": "slm",
        "kit_cost_usd": 245.0,
        "carriage_mass_delta_g": 60.0,
        "color_hex": "#7c8798",
    },
    "sheet_metal_steel": {
        "id": "sheet_metal_steel",
        "name": "Monolith Laser-Cut Steel Sheet Metal (3–5mm, 200 GPa)",
        "short_name": "Sheet Metal Steel",
        "youngs_modulus_gpa": 200.0,
        "hdt_c": 400.0,
        "creep_tension_limit_n": 320.0,
        "cnc_machined": True,
        "manufacturing_process": "sheet_metal",
        "kit_cost_usd": 85.0,
        "carriage_mass_delta_g": 55.0,
        "color_hex": "#64748b",
    },
    "aluminum_6061_t6": {
        "id": "aluminum_6061_t6",
        "name": "CNC Aluminum 6061-T6 (68.9 GPa)",
        "short_name": "CNC Al 6061",
        "youngs_modulus_gpa": 68.9,
        "hdt_c": 300.0,
        "creep_tension_limit_n": 250.0,
        "cnc_machined": True,
        "manufacturing_process": "cnc",
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
        "manufacturing_process": "cnc",
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
        "carriage_mass_g": 141.0,
        "suggested_bracket_material": "abs_printed",
        "notes": "Minimalist printed XY-joints with umbilical wiring (no drag chain): 141 g dressed on MGN12H / 85 g on MGN9H.",
    },
    "standard_printed": {
        "id": "standard_printed",
        "name": "Standard Printed (Stock Voron ABS/ASA)",
        "short_name": "Standard Printed",
        "carriage_mass_g": 196.0,
        "suggested_bracket_material": "abs_printed",
        "notes": "Stock Voron Trident/2.4 printed ABS/ASA XY-joints, MGN blocks, idlers, and cable chain hardware (196 g dressed on MGN12H / 140 g on MGN9H).",
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
    "monolith_slm_mgn12h": {
        "id": "monolith_slm_mgn12h",
        "name": "Monolith SLM Aluminum XY-Joints (Scale-Weighed)",
        "short_name": "Monolith SLM Al",
        "carriage_mass_g": 182.4,
        "suggested_bracket_material": "slm_aluminum",
        "notes": "1-piece SLM AlSi10Mg XY-joints (47.9 g bare each; 96.2 g w/ MGN12H carriage -> 192.4 g dressed pair, saving 56 g/pair vs LDO CNC MGN12H).",
    },
    "dderg_ti_mgn9h": {
        "id": "dderg_ti_mgn9h",
        "name": "dderg Ultralight Milled + Hollow Ti Live-Shafts (MGN9H)",
        "short_name": "dderg Ti MGN9H",
        "carriage_mass_g": 134.6,
        "suggested_bracket_material": "aluminum_6061_t6",
        "notes": "dderg lightweight milled XY-joints with hollow Ti live shafts & Ti hardware (42.1 g bare / 67.3 g w/ MGN9H -> 134.6 g pair, -67.6 g vs LDO MGN9H).",
    },
    "monolith_sheet_metal": {
        "id": "monolith_sheet_metal",
        "name": "Monolith v1.5 Sheet Metal (3–3.2mm XY-Joints)",
        "short_name": "Monolith Sheet Metal",
        "carriage_mass_g": 235.0,
        "suggested_bracket_material": "sheet_metal_steel",
        "notes": "Monolith Builder v1.5 laser-cut steel gantry (3–3.2 mm XY-joint plates, 4.7–5 mm motor mounts & tensioners, 40x9 mm Z belt clamps).",
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


# Non-blocking open-center CoreXY top bracing & open-door front bracing catalogs
TOP_BRACE_CATALOG: dict[str, dict[str, Any]] = {
    "none": {
        "id": "none",
        "label": "None (Open Top)",
        "short_label": "None",
        "kind": "none",
        "thickness_mm": 0.0,
        "description": "Unbraced open top frame perimeter (baseline).",
        "cost_usd": 0.0,
    },
    "cnc_halo_ring": {
        "id": "cnc_halo_ring",
        "label": "CNC 5mm Al 5052 Halo Ring Plate",
        "short_label": "CNC Halo",
        "kind": "cnc_plate",
        "thickness_mm": 5.0,
        "description": "5.0 mm CNC-routed 5052-H32 aluminum open-center perimeter halo ring bolted to top frame rails with R20 inner fillets (full build-volume overhead access).",
        "cost_usd": 65.0,
    },
    "rear_v_and_front_corners": {
        "id": "rear_v_and_front_corners",
        "label": "Rear V + Front 45° Corner Bars (25x5mm Al)",
        "short_label": "Rear-V + Corners",
        "kind": "flat_bar",
        "thickness_mm": 5.0,
        "description": "25x5 mm 6061-T6 aluminum flat-bar kit forming a rear V-truss into the side rails plus two front 45° corner braces while keeping the top center open.",
        "cost_usd": 28.0,
    },
    "corner_45_bars": {
        "id": "corner_45_bars",
        "label": "4x Top 45° Corner Bars (25x5mm Al)",
        "short_label": "4x Corner Bars",
        "kind": "flat_bar",
        "thickness_mm": 5.0,
        "description": "Four 25x5 mm 6061-T6 aluminum 45° corner brace bars bolted across all four top frame corners with full open-top access.",
        "cost_usd": 24.0,
    },
}

FRONT_BRACE_CATALOG: dict[str, dict[str, Any]] = {
    "none": {
        "id": "none",
        "label": "None (Open Front Door)",
        "short_label": "None",
        "kind": "none",
        "thickness_mm": 0.0,
        "description": "Unbraced open front door portal (baseline).",
        "cost_usd": 0.0,
    },
    "cnc_portal_frame": {
        "id": "cnc_portal_frame",
        "label": "CNC 5mm Al 5052 Open-Door Portal Frame",
        "short_label": "CNC Portal",
        "kind": "cnc_plate",
        "thickness_mm": 5.0,
        "description": "5.0 mm CNC-routed 5052-H32 aluminum perimeter portal frame bolted around the front uprights and horizontal rails with R20 inner fillets and full open door.",
        "cost_usd": 78.0,
    },
    "header_and_skirt": {
        "id": "header_and_skirt",
        "label": "CNC 5mm Al 5052 Header + Lower Skirt",
        "short_label": "Header + Skirt",
        "kind": "cnc_plate",
        "thickness_mm": 5.0,
        "description": "Twin 5.0 mm CNC 5052 aluminum plates: upper gantry-clearance header + 90 mm lower front electronics skirt, leaving full-width front door access.",
        "cost_usd": 46.0,
    },
    "corner_45_bars": {
        "id": "corner_45_bars",
        "label": "4x Front 45° Corner Bars (25x5mm Al)",
        "short_label": "4x Corner Bars",
        "kind": "flat_bar",
        "thickness_mm": 5.0,
        "description": "Four 25x5 mm 6061-T6 aluminum 45° corner brace bars bolted across all four front portal corners without obstructing front door access.",
        "cost_usd": 24.0,
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
    top_brace: TopBraceType = Field(
        "none",
        description="Non-blocking open-center CoreXY top frame bracing: none, cnc_halo_ring (5mm Al 5052 perimeter ring), rear_v_and_front_corners (25x5mm Al bars), or corner_45_bars (4x 25x5mm Al 45-deg corner bars)",
    )
    front_brace: FrontBraceType = Field(
        "none",
        description="Non-blocking open-door CoreXY front portal bracing: none, cnc_portal_frame (5mm Al 5052 full perimeter door frame), header_and_skirt (5mm Al 5052 upper header + lower skirt), or corner_45_bars (4x 25x5mm Al 45-deg corner bars)",
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
    chamber_heater_id: ChamberHeaterType = Field(
        "none",
        description="Active chamber PTC heater kit: none, monolith_100w_ptc (100W 24V + 2x 3628), or monolith_500w_ptc (500W AC + 2x 5032)",
    )
    chamber_insulation_mm: float = Field(
        0.0,
        ge=0.0,
        le=50.0,
        description="Internal PIR rigid chamber insulation foam thickness (mm; e.g., 10 mm Monolith Zero, 20 mm Fixed Gantry)",
    )
    rear_bottom_extrusion: bool = Field(
        True,
        description="Include lower rear horizontal perimeter extrusion (affects whole-frame torsion and spatial IS uniformity across bed)",
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
    def elastomer_seating_factor(self) -> float:
        return float(self.catalog_entry.get("elastomer_seating_factor", 1.0))

    @property
    def rolling_drag_mult(self) -> float:
        return float(self.catalog_entry.get("rolling_drag_mult", 1.0))

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
    def static_stiffness_ea_n(self) -> float:
        """Quasi-static manufacturer datasheet tensile modulus EA (Newtons)."""
        entry = self.catalog_entry
        base_static = float(entry.get("ea_static_n", entry["ea_default_core_n"]))
        default_core = str(entry["default_core"])
        if self.core == default_core:
            return base_static
        core_rel = {"fiberglass": 1.0, "aramid": 1.45, "steel": 2.70}
        return base_static * (core_rel[self.core] / core_rel[default_core])

    @property
    def dynamic_stiffening_factor(self) -> float:
        """Small-amplitude 40-150 Hz dynamic-to-static tangent modulus factor."""
        entry = self.catalog_entry
        return float(entry.get("dynamic_stiffening_factor", 1.0))

    @property
    def stiffness_ea_n(self) -> float:
        """Dynamic small-amplitude tensile modulus EA (Newtons) = static_stiffness_ea_n * dynamic_stiffening_factor."""
        return self.static_stiffness_ea_n * self.dynamic_stiffening_factor

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

    def pluck_freq_hz(
        self, span_mm: float = 150.0, linear_mass_kg_m: float | None = None
    ) -> float:
        """Transverse acoustic belt pluck frequency at given free span (standard Voron/VzBot 150 mm)."""
        span_m = max(span_mm * 1e-3, 0.05)
        mu = self.linear_mass_kg_m if linear_mass_kg_m is None else float(linear_mass_kg_m)
        return (1.0 / (2.0 * span_m)) * ((self.pretension_n / mu) ** 0.5)

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
    cf_rail_mounting: CFRailMountingType = Field(
        "auto",
        description=(
            "Carbon-fiber X-beam linear rail mounting method: auto (2-wall clamped internal "
            "inserts on Carbonara, single-wall bare on generic CF tubes), single_wall_bare, "
            "or two_wall_clamped_inserts"
        ),
    )
    dual_x_rails: bool = False
    carriage_mass_g: float = Field(
        140.0, ge=20.0, le=1200.0, description="Combined X/Y gantry carriages & joints (g)"
    )
    belt: BeltSpec = Field(default_factory=BeltSpec)
    stepper_thermal_gasket: StepperThermalGasketType = Field(
        "none",
        description="Stepper-to-bracket thermal isolation gasket: none or ptfe_2mm (drops motor mount temp ~14 C in 80 C chambers)",
    )
    idler_axial_preload: IdlerAxialPreloadType = Field(
        "auto",
        description="Axial preload on F695/695 live-shaft bearings to prevent Al-housing CTE press-fit loss >66 C: auto (disc_spring_m5 on monolith_short), none, or disc_spring_m5",
    )
    z_joint_type: ZJointType = Field(
        "single",
        description="Flying-gantry V2 Z-joint carriage count per corner: single (1 carriage) or double (2 carriages per Z rail)",
    )
    z_joint_material: ZJointMaterialType = Field(
        "printed",
        description="Flying-gantry V2 Z-joint construction: printed, hybrid, slm, or milled (LDO Milled Z-Joint Kit w/ compliant spacers)",
    )
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

    @property
    def effective_cf_rail_mounting(
        self,
    ) -> Literal["none", "single_wall_bare", "two_wall_clamped_inserts"]:
        """Resolve effective carbon-fiber X-beam rail mounting architecture."""
        if not (
            self.x_beam_profile.startswith("cf_")
            or self.x_beam_material.startswith("carbon_fiber")
        ):
            return "none"
        if self.cf_rail_mounting == "auto":
            return (
                "two_wall_clamped_inserts"
                if self.x_beam_profile == "cf_tube_20x20_carbonara"
                else "single_wall_bare"
            )
        return self.cf_rail_mounting

    @property
    def effective_idler_axial_preload(self) -> Literal["none", "disc_spring_m5"]:
        """Resolve 'auto' axial bearing preload (Monolith short-path kits use M5 Belleville disc springs)."""
        if self.idler_axial_preload == "auto":
            return "disc_spring_m5" if self.belt_routing == "monolith_short" else "none"
        return self.idler_axial_preload


class StepperSpec(BaseModel):
    """Stepper motor electromechanical & thermal model calibrated to manufacturer datasheets."""

    preset_id: str = "ldo_2504ac"
    name: str = "NEMA17-48mm (LDO 42STH48-2504AC)"
    holding_torque_nm: float = Field(0.55, gt=0)
    rated_current_a: float = Field(2.5, gt=0, description="Motor datasheet rated RMS current per phase (A)")
    run_current_a: float = Field(
        1.2, gt=0.1, le=8.0, description="Configured driver RMS run current per phase (A)"
    )
    phase_resistance_ohm: float = Field(1.2, gt=0.05, description="Winding phase resistance R at 25C (Ohms)")
    inductance_mh: float = Field(1.5, gt=0, description="Winding phase inductance L (mH)")
    supply_voltage_v: float = Field(24.0, gt=0, le=96.0, description="Stepper driver bus voltage (V)")
    rotor_inertia_kg_m2: float = Field(6.8e-6, gt=0)
    motor_mass_g: float = Field(340.0, gt=0)
    steps_per_rev: int = Field(200, ge=100, description="200 for 1.8 deg, 400 for 0.9 deg")
    nema_size: int = Field(17, ge=14, le=34, description="NEMA faceplate frame size (14, 17, 23)")
    body_length_mm: float = Field(48.0, gt=10.0, le=150.0, description="Motor stack body length (mm)")
    insulation_class: Literal["B", "F", "H"] = Field(
        "B", description="Winding thermal insulation class: B=130C, F=155C, H=180C"
    )
    has_encoder: bool = Field(
        False, description="Whether motor includes an integrated optical/magnetic shaft encoder"
    )

    @property
    def pole_pairs(self) -> float:
        """Electrical pole pairs p = steps_per_rev / 4 (50 for 1.8°, 100 for 0.9°)."""
        return self.steps_per_rev / 4.0

    @property
    def max_winding_temp_c(self) -> float:
        """IEC 60085 continuous winding insulation limit (Class B=130°C, F=155°C, H=180°C)."""
        return {"B": 130.0, "F": 155.0, "H": 180.0}.get(self.insulation_class, 130.0)

    @property
    def shaft_diameter_mm(self) -> float:
        """Standard output shaft diameter by NEMA frame size (NEMA 14/17: 5.0 mm, NEMA 23: 6.35 mm)."""
        return 6.35 if self.nema_size >= 23 else 5.0

    @property
    def thermal_resistance_k_per_w(self) -> float:
        """Steady-state winding-to-ambient thermal resistance R_th (K/W) scaled by motor surface area.

        Calibrated to NEMA 17 48 mm benchmark (~4.6 K/W mounted on aluminum/bracket in enclosure).
        """
        face_mm = {14: 35.2, 17: 42.3, 23: 57.0}.get(self.nema_size, 2.54 * self.nema_size)
        area_mm2 = 4.0 * face_mm * self.body_length_mm + 2.0 * (face_mm**2)
        area_ref_mm2 = 4.0 * 42.3 * 48.0 + 2.0 * (42.3**2)  # 11,700 mm^2 (NEMA 17 48 mm)
        return 4.6 * ((area_ref_mm2 / max(area_mm2, 1000.0)) ** 0.85)

    def torque_at_rms_current_nm(self, i_rms_a: float) -> float:
        """Static electromagnetic torque at RMS phase current: linear below rated, soft tanh saturation above."""
        import math

        ratio = max(i_rms_a, 0.0) / max(self.rated_current_a, 0.1)
        if ratio <= 1.0:
            return self.holding_torque_nm * ratio
        sat = 1.8
        return self.holding_torque_nm * (1.0 + math.tanh(sat * (ratio - 1.0)) / sat)

    @property
    def effective_holding_torque_nm(self) -> float:
        """Current-scaled static holding torque: linear below rated current, soft saturation above."""
        return self.torque_at_rms_current_nm(self.run_current_a)

    def _phase_constants(self) -> tuple[float, float]:
        """Return consistent per-phase peak back-EMF constant Kt_phase (V_pk/(rad/s)) and peak current I_pk (A)."""
        import math

        # Two-phases-on holding torque: T_hold = sqrt(2) * Kt_phase * I_rated (RMS)
        k_t_phase = self.holding_torque_nm / (math.sqrt(2.0) * max(self.rated_current_a, 0.1))
        i_pk = math.sqrt(2.0) * self.run_current_a
        return k_t_phase, i_pk

    def dynamic_voltage_efficiency(self, rpm: float = 250.0, bus_utilization: float = 0.90) -> float:
        """Smooth back-EMF + inductive current-tracking efficiency eta_V in (0, 1] for EM stiffness & damping."""
        import math

        l_h = self.inductance_mh * 1e-3
        omega_mech = max(rpm, 10.0) * (2.0 * math.pi / 60.0)
        k_e, i_pk = self._phase_constants()
        v_bemf = k_e * omega_mech
        v_inductive = omega_mech * self.pole_pairs * l_h * i_pk
        v_res = i_pk * self.phase_resistance_ohm
        v_bus_eff = max(bus_utilization * self.supply_voltage_v, 1.0)
        v_headroom = max(v_bus_eff - v_res, 1.0)
        return 1.0 / math.sqrt(
            1.0 + ((v_res + v_bemf) / v_bus_eff) ** 2 + (v_inductive / v_headroom) ** 2
        )

    def pullout_torque_efficiency(self, rpm: float = 250.0, bus_utilization: float = 0.90) -> float:
        """Piecewise two-phase phasor voltage-limited pull-out torque ratio in [0, 1].

        Constant torque (1.0) up to corner speed where |(R + j*w_e*L)*I_pk + E| <= V_bus_eff;
        above corner speed solves (R*I_q + E)^2 + (w_e*L*I_q)^2 = V_bus_eff^2 for I_q / I_pk.
        """
        import math

        k_e, i_pk = self._phase_constants()
        omega_mech = max(rpm, 0.0) * (2.0 * math.pi / 60.0)
        v_bemf = k_e * omega_mech
        x_l = self.pole_pairs * omega_mech * (self.inductance_mh * 1e-3)
        r_ohm = self.phase_resistance_ohm
        v_bus_eff = max(bus_utilization * self.supply_voltage_v, 1.0)

        if math.hypot(r_ohm * i_pk + v_bemf, x_l * i_pk) <= v_bus_eff:
            return 1.0
        if v_bemf >= v_bus_eff:
            return 0.0

        z2 = r_ohm * r_ohm + x_l * x_l
        disc = (r_ohm * v_bemf) ** 2 + z2 * (v_bus_eff * v_bus_eff - v_bemf * v_bemf)
        i_q = (-r_ohm * v_bemf + math.sqrt(max(0.0, disc))) / max(z2, 1e-12)
        return max(0.0, min(1.0, i_q / max(i_pk, 1e-9)))

    def torsional_stiffness_nm_per_rad(self, rpm: float = 250.0) -> float:
        """Electromagnetic torsional stiffness k_theta = p * T_hold(I) * eta_V (N·m/rad)."""
        return self.pole_pairs * self.effective_holding_torque_nm * self.dynamic_voltage_efficiency(rpm)

    @property
    def power_dissipation_w(self) -> float:
        """Steady-state 2-phase 25°C copper heat dissipation P = 2 * I_rms^2 * R (W)."""
        return 2.0 * (self.run_current_a**2) * self.phase_resistance_ohm

    def estimated_winding_temp_c(self, ambient_temp_c: float = 25.0) -> float:
        """Self-consistent steady-state winding temperature including copper TCR (+0.3854%/K at 25°C).

        Solves T_w = T_amb + R_th * P_25 * (1 + alpha_cu * (T_w - 25)) in closed form.
        """
        alpha_cu = 0.003854  # Copper TCR referenced to 25°C (0.00393 / (1 + 5*0.00393))
        r_th_p = self.thermal_resistance_k_per_w * self.power_dissipation_w
        denom = max(1.0 - r_th_p * alpha_cu, 0.15)
        t_w = (ambient_temp_c + r_th_p * (1.0 - 25.0 * alpha_cu)) / denom
        return max(ambient_temp_c, min(t_w, 300.0))


STEPPER_CATALOG: dict[str, dict] = {
    "excit3d_max": {
        "preset_id": "excit3d_max",
        "name": "Excit3D Max Motor (48mm 1.8° 4.0A 0.6mH Class H)",
        "holding_torque_nm": 0.56,
        "rated_current_a": 4.0,
        "run_current_a": 3.5,
        "phase_resistance_ohm": 0.45,
        "inductance_mh": 0.6,
        "rotor_inertia_kg_m2": 8.2e-6,
        "motor_mass_g": 370.0,
        "steps_per_rev": 200,
        "nema_size": 17,
        "body_length_mm": 48.0,
        "insulation_class": "H",
        "has_encoder": False,
        "unit_cost_usd": 35.0,
    },
    "isik_ouroboros": {
        "preset_id": "isik_ouroboros",
        "name": "Isik's Tech Ouroboros (48mm 1.8° 1000 PPR Encoder)",
        "holding_torque_nm": 0.55,
        "rated_current_a": 2.5,
        "run_current_a": 1.8,
        "phase_resistance_ohm": 1.2,
        "inductance_mh": 1.6,
        "rotor_inertia_kg_m2": 8.2e-6,
        "motor_mass_g": 375.0,
        "steps_per_rev": 200,
        "nema_size": 17,
        "body_length_mm": 48.0,
        "insulation_class": "B",
        "has_encoder": True,
        "unit_cost_usd": 39.0,
    },
    "ldo_2504ac": {
        "preset_id": "ldo_2504ac",
        "name": "LDO 42STH48-2504AC (48mm 1.8° Speedy Power)",
        "holding_torque_nm": 0.55,
        "rated_current_a": 2.5,
        "run_current_a": 1.2,
        "phase_resistance_ohm": 1.2,
        "inductance_mh": 1.5,
        "rotor_inertia_kg_m2": 8.45e-6,
        "motor_mass_g": 340.0,
        "steps_per_rev": 200,
        "nema_size": 17,
        "body_length_mm": 48.0,
        "insulation_class": "B",
        "has_encoder": False,
        "unit_cost_usd": 22.0,
    },
    "ldo_2504ah": {
        "preset_id": "ldo_2504ah",
        "name": "LDO 42STH48-2504AH (48mm 1.8° High-Temp 180°C)",
        "holding_torque_nm": 0.55,
        "rated_current_a": 2.5,
        "run_current_a": 2.0,
        "phase_resistance_ohm": 1.2,
        "inductance_mh": 1.5,
        "rotor_inertia_kg_m2": 8.45e-6,
        "motor_mass_g": 360.0,
        "steps_per_rev": 200,
        "nema_size": 17,
        "body_length_mm": 48.0,
        "insulation_class": "H",
        "has_encoder": False,
        "unit_cost_usd": 26.0,
    },
    "ldo_2504ah_s45r": {
        "preset_id": "ldo_2504ah_s45r",
        "name": "Monolith LDO 42STH48-2504AH-S45R (45mm Round Shaft, 2m Cable)",
        "holding_torque_nm": 0.55,
        "rated_current_a": 2.5,
        "run_current_a": 2.0,
        "phase_resistance_ohm": 1.2,
        "inductance_mh": 1.5,
        "rotor_inertia_kg_m2": 8.45e-6,
        "motor_mass_g": 372.0,
        "steps_per_rev": 200,
        "nema_size": 17,
        "body_length_mm": 48.0,
        "insulation_class": "H",
        "has_encoder": False,
        "unit_cost_usd": 29.0,
    },
    "ldo_2504ah_s80": {
        "preset_id": "ldo_2504ah_s80",
        "name": "Monolith Zero LDO 42STH48-2504AH-S80 (80mm Shaft)",
        "holding_torque_nm": 0.55,
        "rated_current_a": 2.5,
        "run_current_a": 1.8,
        "phase_resistance_ohm": 1.2,
        "inductance_mh": 1.5,
        "rotor_inertia_kg_m2": 8.6e-6,
        "motor_mass_g": 385.0,
        "steps_per_rev": 200,
        "nema_size": 17,
        "body_length_mm": 48.0,
        "insulation_class": "H",
        "has_encoder": False,
        "unit_cost_usd": 31.0,
    },
    "ldo_2804ah": {
        "preset_id": "ldo_2804ah",
        "name": "LDO 42STH48-2804AH (48mm 1.8° Super Power 0.6mH)",
        "holding_torque_nm": 0.42,
        "rated_current_a": 2.8,
        "run_current_a": 2.0,
        "phase_resistance_ohm": 0.7,
        "inductance_mh": 0.6,
        "rotor_inertia_kg_m2": 8.45e-6,
        "motor_mass_g": 350.0,
        "steps_per_rev": 200,
        "nema_size": 17,
        "body_length_mm": 48.0,
        "insulation_class": "H",
        "has_encoder": False,
        "unit_cost_usd": 28.0,
    },
    "kraken_motor_v2": {
        "preset_id": "kraken_motor_v2",
        "name": "Kraken Motor V2 (60mm 1.8° 3.0A 1.6mH High-Speed)",
        "holding_torque_nm": 0.80,
        "rated_current_a": 3.0,
        "run_current_a": 2.4,
        "phase_resistance_ohm": 1.0,
        "inductance_mh": 1.6,
        "rotor_inertia_kg_m2": 10.0e-6,
        "motor_mass_g": 500.0,
        "steps_per_rev": 200,
        "nema_size": 17,
        "body_length_mm": 60.0,
        "insulation_class": "H",
        "has_encoder": False,
        "unit_cost_usd": 32.0,
    },
    "ldo_60_3004": {
        "preset_id": "ldo_60_3004",
        "name": "LDO 42STH60-3004AC (60mm 1.8° 3.0A High-Torque)",
        "holding_torque_nm": 0.90,
        "rated_current_a": 3.0,
        "run_current_a": 2.2,
        "phase_resistance_ohm": 1.1,
        "inductance_mh": 2.3,
        "rotor_inertia_kg_m2": 12.0e-6,
        "motor_mass_g": 500.0,
        "steps_per_rev": 200,
        "nema_size": 17,
        "body_length_mm": 60.0,
        "insulation_class": "H",
        "has_encoder": False,
        "unit_cost_usd": 30.0,
    },
    "ldo_60_2004ah": {
        "preset_id": "ldo_60_2004ah",
        "name": "LDO 42STH60-2004AH (60mm 1.8° 2.0A High-Temp)",
        "holding_torque_nm": 0.83,
        "rated_current_a": 2.0,
        "run_current_a": 1.6,
        "phase_resistance_ohm": 2.0,
        "inductance_mh": 3.3,
        "rotor_inertia_kg_m2": 10.2e-6,
        "motor_mass_g": 500.0,
        "steps_per_rev": 200,
        "nema_size": 17,
        "body_length_mm": 60.0,
        "insulation_class": "H",
        "has_encoder": False,
        "unit_cost_usd": 27.0,
    },
    "ldo_09deg_2504": {
        "preset_id": "ldo_09deg_2504",
        "name": "LDO 42STH48-2504MAC (48mm 0.9° 2× Magnetic Stiffness)",
        "holding_torque_nm": 0.45,
        "rated_current_a": 2.5,
        "run_current_a": 1.6,
        "phase_resistance_ohm": 1.2,
        "inductance_mh": 1.5,
        "rotor_inertia_kg_m2": 8.5e-6,
        "motor_mass_g": 340.0,
        "steps_per_rev": 400,
        "nema_size": 17,
        "body_length_mm": 48.0,
        "insulation_class": "B",
        "has_encoder": False,
        "unit_cost_usd": 25.0,
    },
    "ldo_09deg_2004mah": {
        "preset_id": "ldo_09deg_2004mah",
        "name": "LDO 42STH48-2004MAH(VRN) (48mm 0.9° 2.0A 1.45Ω 2.0mH Class H)",
        "holding_torque_nm": 0.40,
        "rated_current_a": 2.0,
        "run_current_a": 1.4,
        "phase_resistance_ohm": 1.45,
        "inductance_mh": 2.0,
        "rotor_inertia_kg_m2": 7.1e-6,
        "motor_mass_g": 350.0,
        "steps_per_rev": 400,
        "nema_size": 17,
        "body_length_mm": 48.0,
        "insulation_class": "H",
        "has_encoder": False,
        "unit_cost_usd": 24.0,
    },
    "moons_09deg_ha2p4200": {
        "preset_id": "moons_09deg_ha2p4200",
        "name": "MOONS' MS17HA2P4200 (40mm 0.9° 2.0A 1.05Ω 2.8mH)",
        "holding_torque_nm": 0.39,
        "rated_current_a": 2.0,
        "run_current_a": 1.4,
        "phase_resistance_ohm": 1.05,
        "inductance_mh": 2.8,
        "rotor_inertia_kg_m2": 5.7e-6,
        "motor_mass_g": 280.0,
        "steps_per_rev": 400,
        "nema_size": 17,
        "body_length_mm": 39.8,
        "insulation_class": "B",
        "has_encoder": False,
        "unit_cost_usd": 22.0,
    },
    "leadshine_42cm06": {
        "preset_id": "leadshine_42cm06",
        "name": "Leadshine 42CM06 (47mm 1.8° 0.60 N·m Low-R)",
        "holding_torque_nm": 0.60,
        "rated_current_a": 2.5,
        "run_current_a": 2.0,
        "phase_resistance_ohm": 0.9,
        "inductance_mh": 1.6,
        "rotor_inertia_kg_m2": 7.2e-6,
        "motor_mass_g": 350.0,
        "steps_per_rev": 200,
        "nema_size": 17,
        "body_length_mm": 47.0,
        "insulation_class": "B",
        "has_encoder": False,
        "unit_cost_usd": 24.0,
    },
    "moons_420i": {
        "preset_id": "moons_420i",
        "name": "MOONS' MS17HD6P420I (48mm 1.8° Voron Standard)",
        "holding_torque_nm": 0.67,
        "rated_current_a": 2.0,
        "run_current_a": 1.2,
        "phase_resistance_ohm": 1.3,
        "inductance_mh": 2.7,
        "rotor_inertia_kg_m2": 8.2e-6,
        "motor_mass_g": 360.0,
        "steps_per_rev": 200,
        "nema_size": 17,
        "body_length_mm": 48.3,
        "insulation_class": "B",
        "has_encoder": False,
        "unit_cost_usd": 23.0,
    },
    "stepperonline_2504s": {
        "preset_id": "stepperonline_2504s",
        "name": "STEPPERONLINE 17HS19-2504S (48mm 1.8° 2.5A 1.75mH)",
        "holding_torque_nm": 0.60,
        "rated_current_a": 2.5,
        "run_current_a": 1.8,
        "phase_resistance_ohm": 1.1,
        "inductance_mh": 1.75,
        "rotor_inertia_kg_m2": 6.8e-6,
        "motor_mass_g": 390.0,
        "steps_per_rev": 200,
        "nema_size": 17,
        "body_length_mm": 48.0,
        "insulation_class": "B",
        "has_encoder": False,
        "unit_cost_usd": 17.0,
    },
    "stepperonline_2004": {
        "preset_id": "stepperonline_2004",
        "name": "STEPPERONLINE 17HS19-2004S1 (48mm 1.8° Budget)",
        "holding_torque_nm": 0.59,
        "rated_current_a": 2.0,
        "run_current_a": 1.0,
        "phase_resistance_ohm": 1.4,
        "inductance_mh": 3.0,
        "rotor_inertia_kg_m2": 8.2e-6,
        "motor_mass_g": 390.0,
        "steps_per_rev": 200,
        "nema_size": 17,
        "body_length_mm": 48.0,
        "insulation_class": "B",
        "has_encoder": False,
        "unit_cost_usd": 14.0,
    },
    "stepperonline_09deg_2004": {
        "preset_id": "stepperonline_09deg_2004",
        "name": "STEPPERONLINE 17HM19-2004S (48mm 0.9° Budget)",
        "holding_torque_nm": 0.46,
        "rated_current_a": 2.0,
        "run_current_a": 1.4,
        "phase_resistance_ohm": 1.45,
        "inductance_mh": 4.0,
        "rotor_inertia_kg_m2": 8.2e-6,
        "motor_mass_g": 390.0,
        "steps_per_rev": 400,
        "nema_size": 17,
        "body_length_mm": 48.0,
        "insulation_class": "B",
        "has_encoder": False,
        "unit_cost_usd": 19.0,
    },
    "ldo_40_1684ac": {
        "preset_id": "ldo_40_1684ac",
        "name": "LDO 42STH40-1684AC (40mm 1.8° Compact)",
        "holding_torque_nm": 0.45,
        "rated_current_a": 1.68,
        "run_current_a": 1.0,
        "phase_resistance_ohm": 1.65,
        "inductance_mh": 3.6,
        "rotor_inertia_kg_m2": 5.3e-6,
        "motor_mass_g": 280.0,
        "steps_per_rev": 200,
        "nema_size": 17,
        "body_length_mm": 40.0,
        "insulation_class": "B",
        "has_encoder": False,
        "unit_cost_usd": 18.0,
    },
    "ldo_57_2804mac": {
        "preset_id": "ldo_57_2804mac",
        "name": "LDO 57STH41-2804MAC (NEMA 23 41mm 0.9° HevORT)",
        "holding_torque_nm": 0.60,
        "rated_current_a": 2.8,
        "run_current_a": 2.2,
        "phase_resistance_ohm": 0.7,
        "inductance_mh": 1.4,
        "rotor_inertia_kg_m2": 12.0e-6,
        "motor_mass_g": 420.0,
        "steps_per_rev": 400,
        "nema_size": 23,
        "body_length_mm": 41.0,
        "insulation_class": "B",
        "has_encoder": False,
        "unit_cost_usd": 38.0,
    },
    "stepperonline_23hs22_2804": {
        "preset_id": "stepperonline_23hs22_2804",
        "name": "STEPPERONLINE 23HS22-2804S (NEMA 23 56mm 1.8° 1.26 N·m)",
        "holding_torque_nm": 1.26,
        "rated_current_a": 2.8,
        "run_current_a": 2.4,
        "phase_resistance_ohm": 0.9,
        "inductance_mh": 2.5,
        "rotor_inertia_kg_m2": 30.0e-6,
        "motor_mass_g": 700.0,
        "steps_per_rev": 200,
        "nema_size": 23,
        "body_length_mm": 56.0,
        "insulation_class": "B",
        "has_encoder": False,
        "unit_cost_usd": 29.0,
    },
}
_hydrate_stepper_catalog(STEPPER_CATALOG)
_hydrate_z_drive_catalog(Z_DRIVE_MECHANISM_CATALOG)
_hydrate_bed_mount_catalog(BED_MOUNT_CATALOG)


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
    lpv_workspace_scheduling: bool = False
    awd_sync_mode: AwdSyncMode = "none"
    awd_equalizer_bw_hz: float = Field(2.0, ge=0.5, le=15.0)
    notch_depth_db: float = Field(12.0, ge=3.0, le=30.0)
    notch_q: float = Field(1.75, ge=0.5, le=8.0)
    pin_rotor_ff: bool = False
    toolhead_imu_active_damping: bool = False
    mode_space_torque_ff: bool = False
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
        description="Stage 2 toolhead body/housing stiffness (N/mm) used when body_construction is None; scaled by body_material and body_wall_thickness_mm, then combined in series with Stage 1 carriage plate and Stage 3 hotend clamp.",
    )
    z_probe_type: ZProbeType = Field(
        "static_eddy",
        description="Bed probing carriage architecture: static_eddy (Beacon/Cartographer), klicky_inductive, cnc_tap, or printed_tap",
    )
    use_hotend_bottom_brace: bool = Field(
        default=True,
        description="Enable the toolhead's lower hotend brace (e.g., Microwaved Bowden Ti bottom brace on Tricorn Long) when a coupled braced CAD superelement is available",
    )
    nozzle_shield_mode: Literal["auto", "stock", "microwaved", "none"] = Field(
        default="auto",
        description=(
            "Nozzle thermal/airflow shield configuration for supported toolhead/hotend pairs "
            "(auto = Microwaved SLM Ti shield on Microwaved+Tricorn, stock Tricorn bucket otherwise; "
            "stock = stock Tricorn Ti bucket; microwaved = Microwaved optional SLM Ti shield; none = bare nozzle)"
        ),
    )
    envelope_mm: tuple[float, float, float] = (68.0, 74.0, 95.0)
    cpap_blower_id: CpapBlowerType = Field(
        "none",
        description="Remote CPAP part-cooling blower: none (onboard fans), ws7040_24v (5.8 dm3/s), wm9290_24v (9.2 dm3/s), or wm7575_220vac (11.6 dm3/s)",
    )
    cpap_hose_id_mm: float = Field(
        15.0,
        ge=10.0,
        le=25.0,
        description="CPAP air hose internal diameter (mm; 15 mm HT hose vs 18 mm standard hose)",
    )
    cpap_pwm_pct: float = Field(
        60.0,
        ge=0.0,
        le=100.0,
        description="Nominal CPAP blower PWM duty cycle (%)",
    )
    shared_heatsink_cpap_duct: bool = Field(
        False,
        description="Enforce Kalico [heater_fan] minimum CPAP airflow through hotend heatsink duct when hotend > 50 C",
    )
    belt_plane_z_offset_mm: float = Field(
        0.0,
        ge=-40.0,
        le=40.0,
        description="Vertical offset from X-beam centerline to XY drive belt plane (mm); 0 mm = centered/coplanar",
    )
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
        "default_hardware_summary": (
            "Clockwork 2 (CW2) direct-drive extruder + NEMA14 pancake, ABS shroud, "
            "5015 + 4010 fans, MGN12H carriage & static eddy probe (hotend added separately)."
        ),
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
        "default_hardware_summary": (
            "Clockwork 2 (CW2) direct-drive extruder + NEMA14 pancake, ABS shroud, "
            "5015 + 4010 fans & Printed Voron Tap carriage (hotend added separately)."
        ),
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
        "default_hardware_summary": (
            "Sherpa Mini direct-drive extruder + NEMA14 pancake, PA12-CF cage, "
            "dual cooling blowers, MGN12H carriage & static eddy probe (hotend added separately)."
        ),
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
        "default_hardware_summary": (
            "Wristwatch G2 / compact direct-drive extruder + NEMA14 pancake, PPA-CF cage, "
            "dual blowers, MGN12H carriage & static Beacon probe (hotend added separately)."
        ),
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
        "default_hardware_summary": (
            "Orbiter 2 / Sherpa compact direct-drive extruder, PA12-CF shroud, "
            "dual 4010 blowers, MGN12H carriage & static eddy probe (hotend added separately)."
        ),
    },
    "archetype_cpap": {
        "preset_id": "archetype_cpap",
        "name": "Archetype CPAP Ultra-Low CoG",
        "total_mass_g": 175.0,
        "cog_offset_mm": [0.0, -11.0, -12.0],
        "nozzle_offset_mm": [0.0, -14.0, -48.0],
        "mount_material": "aluminum_6061_t6",
        "body_construction": "plate_sandwich_cf",
        "internal_stiffness_n_per_mm": 1800.0,
        "z_probe_type": "static_eddy",
        "default_hardware_summary": (
            "Lightweight direct-drive extruder (Sherpa/Orbiter-class), CF/Al plate sandwich, "
            "remote CPAP duct (no onboard blowers), MGN12H carriage & static eddy probe (hotend added separately)."
        ),
    },
    "monolith_toolhead_cpap": {
        "preset_id": "monolith_toolhead_cpap",
        "name": "Monolith Toolhead (Belt-Coplanar CPAP + 37:1 Worm)",
        "total_mass_g": 143.0,
        "cog_offset_mm": [0.0, -9.5, -1.5],
        "nozzle_offset_mm": [0.0, -14.0, -44.0],
        "mount_material": "slm_aluminum",
        "body_construction": "extruder_as_chassis",
        "internal_stiffness_n_per_mm": 2400.0,
        "z_probe_type": "static_eddy",
        "cpap_blower_id": "ws7040_24v",
        "cpap_hose_id_mm": 15.0,
        "belt_plane_z_offset_mm": -1.5,
        "default_hardware_summary": (
            "Integrated 37:1 worm-gear direct-drive extruder, SLM Al belt-coplanar CPAP chassis "
            "(WS7040 24V, 15 mm hose), MGN12H carriage & static eddy probe (hotend added separately)."
        ),
    },
    "microwaved_bowden": {
        "preset_id": "microwaved_bowden",
        "name": "Microwaved Bowden SLM (CPAP + Beacon)",
        "total_mass_g": 130.0,
        "cog_offset_mm": [0.0, -8.5, -2.0],
        "nozzle_offset_mm": [0.0, -17.5, -35.5],
        "mount_material": "aluminum_6061_t6",
        "body_construction": "cnc_monoblock",
        "internal_stiffness_n_per_mm": 5340.0,
        "z_probe_type": "static_eddy",
        "extruder_type": "bowden",
        "default_hardware_summary": (
            "Remote Bowden extruder (no onboard motor) — 130 g includes SLM Ti/Al body (43.4 g), "
            "Ti bottom brace & shield (2.9 g), MGN12H carriage (48.3 g), Monolith SLM belt clamp (~16 g), "
            "Beacon RevH & ECAS coupler (hotend added separately)."
        ),
    },
    "microwaved_bowden_chube": {
        "preset_id": "microwaved_bowden_chube",
        "name": "Microwaved Bowden Chube V1 SLM (CPAP + Beacon)",
        "total_mass_g": 135.0,
        "cog_offset_mm": [0.0, -10.5, -3.0],
        "nozzle_offset_mm": [0.0, -20.5, -38.5],
        "mount_material": "aluminum_6061_t6",
        "body_construction": "cnc_monoblock",
        "internal_stiffness_n_per_mm": 2990.0,
        "z_probe_type": "static_eddy",
        "extruder_type": "bowden",
        "default_hardware_summary": (
            "Remote Bowden extruder (no onboard motor) — 135 g includes SLM Al Chube V1 body (~56.5 g), "
            "MGN12H carriage (48.3 g), Monolith SLM belt clamp (~16 g), Beacon RevH & ECAS coupler "
            "(hotend added separately)."
        ),
    },
    "el_fuego": {
        "preset_id": "el_fuego",
        "name": "El Fuego SLM (CPAP + Sherpa Micro + Beacon)",
        "total_mass_g": 262.0,
        "cog_offset_mm": [0.0, -10.5, 2.0],
        "nozzle_offset_mm": [-0.3, -27.7, -63.7],
        "mount_material": "slm_aluminum",
        "body_construction": "cnc_monoblock",
        "internal_stiffness_n_per_mm": 1007.0,
        "z_probe_type": "static_eddy",
        "cpap_blower_id": "wm7575_220vac",
        "cpap_hose_id_mm": 19.0,
        "shared_heatsink_cpap_duct": True,
        "extruder_type": "direct_drive",
        "default_hardware_summary": (
            "Sherpa Micro direct-drive extruder + NEMA14 pancake (~105 g, pulls CoG to Z +2.0 mm), "
            "3-part SLM CAD chassis (94.2 g: AlSi10Mg main body + Al belt plate + Ti extruder plate), "
            "MGN12H carriage (48.3 g), Beacon RevH & fasteners, plus WM7575 CPAP (19 mm hose) w/ shared "
            "heatsink bleed. Hotend added separately; 3D mesh renders chassis only."
        ),
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
        "default_hardware_summary": (
            "Heavy direct-drive extruder, multi-blower UHF ABS shroud, long-nozzle drop spacer, "
            "MGN12H carriage & static eddy probe (hotend added separately)."
        ),
    },
}

# Remote CPAP part-cooling blower catalog (Monolith3D ducted flow & exhaust-heating benchmarks)
CPAP_BLOWER_CATALOG: dict[str, dict[str, Any]] = {
    "none": {
        "id": "none",
        "name": "Onboard Toolhead Fans (No CPAP)",
        "short_name": "Onboard Fans",
        "ducted_max_dm3_s": 0.0,
        "ducted_max_cfm": 0.0,
        "min_stable_pwm_pct": 0.0,
        "exhaust_heating_at_100pct_c": 0.0,
        "max_static_pressure_pa": 0.0,
        "voltage_v": 24.0,
        "acoustic_profile": "standard",
        "cost_usd": 0.0,
        "notes": "Uses HotendSpec.part_cooling_cfm directly without remote CPAP hose dynamics.",
    },
    "ws7040_24v": {
        "id": "ws7040_24v",
        "name": "WS7040-24-V200 Brushless CPAP (5.8 dm³/s)",
        "short_name": "WS7040 24V",
        "ducted_max_dm3_s": 5.8,
        "ducted_max_cfm": 12.29,
        "min_stable_pwm_pct": 18.0,
        "exhaust_heating_at_100pct_c": 5.5,
        "max_static_pressure_pa": 6500.0,
        "voltage_v": 24.0,
        "acoustic_profile": "moderate",
        "cost_usd": 42.0,
        "printdb_id": "components/fans/ws7040-24-v200-cpap",
        "notes": "Monolith benchmarked 5.8 dm³/s ducted max (~12.3 CFM) with +5.5 °C exhaust heating at 100% PWM.",
    },
    "wm9290_24v": {
        "id": "wm9290_24v",
        "name": "WM9290-24V Brushless CPAP (9.2 dm³/s, Quietest)",
        "short_name": "WM9290 24V",
        "ducted_max_dm3_s": 9.2,
        "ducted_max_cfm": 19.49,
        "min_stable_pwm_pct": 12.0,
        "exhaust_heating_at_100pct_c": 4.2,
        "max_static_pressure_pa": 7500.0,
        "voltage_v": 24.0,
        "acoustic_profile": "quiet",
        "cost_usd": 78.0,
        "printdb_id": "components/fans/wm9290-24v-cpap",
        "notes": "Monolith benchmarked 9.2 dm³/s ducted max (~19.5 CFM), smooth control down to 1.0 dm³/s at 25% PWM, quietest acoustic spectrum.",
    },
    "wm7575_220vac": {
        "id": "wm7575_220vac",
        "name": "WM7575-220VAC High-Output CPAP (11.6 dm³/s)",
        "short_name": "WM7575 AC",
        "ducted_max_dm3_s": 11.6,
        "ducted_max_cfm": 24.58,
        "min_stable_pwm_pct": 20.0,
        "exhaust_heating_at_100pct_c": 6.8,
        "max_static_pressure_pa": 9000.0,
        "voltage_v": 220.0,
        "acoustic_profile": "loud_hf",
        "cost_usd": 95.0,
        "printdb_id": "components/fans/wm7575-220vac-cpap",
        "notes": "Monolith benchmarked 11.6 dm³/s ducted max (~24.6 CFM) with loud high-frequency acoustic spectrum.",
    },
}

# Active PTC chamber heater duct kits (Monolith3D Builder / SLM_Chamber_Heater_Ducts)
CHAMBER_HEATER_CATALOG: dict[str, dict[str, Any]] = {
    "none": {
        "id": "none",
        "name": "Passive Bed Heating Only (No Active Heater)",
        "short_name": "None (Passive)",
        "power_w": 0.0,
        "voltage_v": 0.0,
        "fan_count": 0,
        "fan_size": "none",
        "ptc_dims_mm": (0.0, 0.0),
        "circulation_cfm": 0.0,
        "cost_usd": 0.0,
        "notes": "Chamber heated solely by bed convection/radiation.",
    },
    "monolith_100w_ptc": {
        "id": "monolith_100w_ptc",
        "name": "Monolith 100W 24V SLM PTC Heater (2× 3628 Fans)",
        "short_name": "Monolith 100W PTC",
        "power_w": 100.0,
        "voltage_v": 24.0,
        "fan_count": 2,
        "fan_size": "3628",
        "ptc_dims_mm": (98.0, 32.0),
        "circulation_cfm": 14.0,
        "cost_usd": 58.0,
        "printdb_id": "assemblies/printers/monolith-chamber-heater-100w-slm",
        "notes": "Monolith Builder 100W 24V PTC (98x32 mm) + SLM duct + 2x 3628 fans + 4x M3x15 F-F standoffs (panel or extrusion mount).",
    },
    "monolith_500w_ptc": {
        "id": "monolith_500w_ptc",
        "name": "Monolith 500W AC SLM PTC Heater (2× 5032 Fans)",
        "short_name": "Monolith 500W PTC",
        "power_w": 500.0,
        "voltage_v": 120.0,
        "fan_count": 2,
        "fan_size": "5032",
        "ptc_dims_mm": (140.0, 50.0),
        "circulation_cfm": 32.0,
        "cost_usd": 115.0,
        "printdb_id": "assemblies/printers/monolith-chamber-heater-500w-slm",
        "notes": "Monolith Builder 500W AC PTC (140x50 mm) + SLM duct + 2x 5032 fans + 4x M3x15 F-F standoffs (panel, fixed, or flying gantry mount).",
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
            "Bed frame attachment constraint: kinematic_3pt, whopping_pochard_kinematic, "
            "trident_3pt_standoffs, cnc_kelvin_magnetic, captive_belleville_kinematic, "
            "dual_leaf_ti_kinematic, spherical_3pt_ratrig, rigid_4pt, floating_4pt, "
            "silicone_4pt, or springs_4pt"
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
    model_overrides: ModelOverridesSpec = Field(default_factory=ModelOverridesSpec)
    use_cad_superelements: bool = Field(
        default=True,
        description="Use pre-baked 3D CAD Craig-Bampton superelements for toolheads/hotends when available",
    )

    @field_validator("model_overrides", mode="before")
    @classmethod
    def _coerce_none_model_overrides(cls, v: Any) -> Any:
        return {} if v is None else v

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
        data = self.model_dump(mode="json")
        compact_overrides = self.model_overrides.to_compact_dict()
        if compact_overrides:
            data["model_overrides"] = compact_overrides
        else:
            data.pop("model_overrides", None)
        return yaml.safe_dump(data, sort_keys=False)

    def save_yaml(self, path: str | Path) -> None:
        Path(path).write_text(self.to_yaml(), encoding="utf-8")
