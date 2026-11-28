import React, { useEffect, useRef, useState } from "react";
import {
  AWD_SYNC_LABELS,
  BELT_LABELS,
  SHAPER_LABELS,
  getBeltDefaults,
  getBeltTensionSliderMax,
} from "../constants/printerLabels";
import {
  AwdSyncMode,
  BearingClearanceClass,
  BeltRoutingType,
  BracketMaterialType,
  CFRailMountingType,
  CatalogResponse,
  DriveType,
  GantryBracketInfo,
  GantryHardwareKitItem,
  GantryHardwareKitType,
  GantryWeightClassItem,
  IdlerAxialPreloadType,
  PrinterSpecData,
  RailBackerMaterialType,
  RailBlockType,
  RailMountOption,
  RailPreloadType,
  RailQualityType,
  SimulationResponse,
  StepperThermalGasketType,
  XRailMountType,
  YRailMountType,
  ZJointMaterialType,
  ZJointType,
} from "../types";
import {
  ModelOverrideControl,
  PresetAffectsStrip,
  clearModelOverrideGroup,
} from "./ModelOverrideControl";
import {
  applyGantryHardwareKit,
  getGantryKitAffectsItems,
  syncDriveModeWithHardwareKit,
} from "../utils/kitPresets";

const DRIVE_TYPE_OPTIONS: Array<{ id: DriveType; label: string; shortLabel: string }> = [
  { id: "open_loop_stepper", label: "Open-Loop Stepper", shortLabel: "Open-Loop Stepper" },
  { id: "closed_loop_stepper", label: "Closed-Loop Stepper", shortLabel: "Closed-Loop Stepper" },
  { id: "foc_servo", label: "FOC Servo (Closed-Loop)", shortLabel: "FOC Servo" },
];

const AWD_SYNC_OPTIONS: Array<{ id: AwdSyncMode; label: string; shortLabel: string }> = [
  { id: "none", label: AWD_SYNC_LABELS.none, shortLabel: "Off" },
  { id: "torque_mirror", label: AWD_SYNC_LABELS.torque_mirror, shortLabel: "Torque Mirror" },
  { id: "position_equalizer", label: AWD_SYNC_LABELS.position_equalizer, shortLabel: "Pos + Equalizer" },
];

function computeNotchPhaseMetrics(depthDb: number, q: number): {
  maxLagDeg: number;
  peakLagFreqRatio: number;
} {
  const safeDepth = Math.max(0.5, Math.min(30, depthDb));
  const safeQ = Math.max(0.5, Math.min(8, q));
  const k = Math.pow(10, -safeDepth / 20);
  const sqrtK = Math.sqrt(k);
  const maxLagDeg = (Math.atan((1 - k) / (2 * sqrtK)) * 180) / Math.PI;
  const peakLagFreqRatio =
    Math.sqrt(1 + k / (4 * safeQ * safeQ)) - sqrtK / (2 * safeQ);
  return { maxLagDeg, peakLagFreqRatio };
}

const X_BEAM_METAL_PROFILE_IDS = new Set([
  "extrusion_2020",
  "extrusion_2040",
  "cnc_al_beam_20x20",
  "cnc_al_monocoque_20x20",
  "cnc_al_truss_voron24_lw",
  "cnc_al_fysetc_v0_15x15",
  "al_tube_12x12x1",
  "al_tube_20x20x2",
  "al_tube_20x20x3",
  "extrusion_3030",
]);

const X_BEAM_LIGHTWEIGHT_TUBE_IDS = new Set([
  "cf_tube_20x20x1",
  "cf_tube_20x20x1_5",
  "cf_tube_20x20_carbonara",
  "cf_tube_20x20_carbonara_2_5",
  "cf_tube_20x20x2",
  "cf_tube_30x20x2",
  "cf_tube_25x25x2",
  "ti_tube_20x20x1_2",
  "ti_tube_20x20x2",
  "unsupported_rail",
]);

const X_BEAM_MATERIAL_IDS = new Set([
  "aluminum_6063_t5",
  "aluminum_6061_t6",
  "aluminum_7075_t6",
  "carbon_fiber_3k_twill",
  "carbon_fiber_tube",
  "carbon_fiber_high_tg",
  "carbon_fiber_torsional_45",
  "carbon_fiber_m40j_hm",
  "titanium_grade5",
  "steel_4140",
]);

export const FALLBACK_GANTRY_HARDWARE_KITS: GantryHardwareKitItem[] = [
  {
    id: "auto",
    name: "Auto (Match Preset & Drive Mode)",
    short_name: "Auto",
  },
  {
    id: "voron_stock_printed",
    name: "Voron Stock Printed Gantry (2WD)",
    short_name: "Voron Stock",
    default_bracket_material: "abs_printed",
    drive_mode: "2wd",
  },
  {
    id: "ldo_voron_kit",
    name: "LDO Voron Kit (Printed + CNC Upgrades)",
    short_name: "LDO Kit",
    default_bracket_material: "abs_printed",
    drive_mode: "2wd",
  },
  {
    id: "siboor_cnc_awd",
    name: "Siboor CNC Gantry Kit (AWD)",
    short_name: "Siboor CNC AWD",
    default_bracket_material: "aluminum_6061_t6",
    drive_mode: "awd",
  },
  {
    id: "siboor_cnc_2wd",
    name: "Siboor CNC Gantry Kit (2WD)",
    short_name: "Siboor CNC 2WD",
    default_bracket_material: "aluminum_6061_t6",
    drive_mode: "2wd",
  },
  {
    id: "monolith_ldo_cnc_awd",
    name: "Monolith / LDO CNC Gantry (AWD)",
    short_name: "Monolith CNC AWD",
    default_bracket_material: "aluminum_6061_t6",
    drive_mode: "awd",
  },
  {
    id: "monolith_ldo_cnc_2wd",
    name: "Monolith / LDO CNC Gantry (2WD)",
    short_name: "Monolith CNC 2WD",
    default_bracket_material: "aluminum_6061_t6",
    drive_mode: "2wd",
  },
  {
    id: "vzbot_330_awd",
    name: "VzBot 330 Gantry (AWD · 6mm · MGN9 CF)",
    short_name: "VzBot 330 AWD",
    default_bracket_material: "aluminum_6061_t6",
    drive_mode: "awd",
  },
  {
    id: "vzbot_330_2wd",
    name: "VzBot 330 Gantry (2WD · 6mm · MGN9 CF)",
    short_name: "VzBot 330 2WD",
    default_bracket_material: "aluminum_6061_t6",
    drive_mode: "2wd",
  },
  {
    id: "procedural",
    name: "Procedural Simplified Geometry",
    short_name: "Procedural",
  },
];

export const FALLBACK_GANTRY_BRACKETS: GantryBracketInfo[] = [
  {
    id: "abs_printed",
    name: "3D-Printed ABS / ASA (2.1 GPa)",
    short_name: "Printed ABS",
    youngs_modulus_gpa: 2.1,
    hdt_c: 95,
    creep_tension_limit_n: 45,
    cnc_machined: false,
    kit_cost_usd: 28,
    carriage_mass_delta_g: 0,
    color_hex: "#e63946",
  },
  {
    id: "pet_cf_printed",
    name: "3D-Printed PET-CF15 Annealed (5.2 GPa)",
    short_name: "Printed PET-CF15",
    youngs_modulus_gpa: 5.2,
    hdt_c: 182,
    creep_tension_limit_n: 75,
    cnc_machined: false,
    kit_cost_usd: 52,
    carriage_mass_delta_g: 10,
    color_hex: "#3b4252",
  },
  {
    id: "pa12_cf_printed",
    name: "3D-Printed PA12-CF15 (6.2 GPa)",
    short_name: "Printed PA12-CF",
    youngs_modulus_gpa: 6.2,
    hdt_c: 155,
    creep_tension_limit_n: 65,
    cnc_machined: false,
    kit_cost_usd: 72,
    carriage_mass_delta_g: 6,
    color_hex: "#333840",
  },
  {
    id: "pc_pbt_gf_printed",
    name: "3D-Printed PC-PBT-GF (6.1 GPa, 120°C HDT)",
    short_name: "Printed PC-PBT-GF",
    youngs_modulus_gpa: 6.1,
    hdt_c: 120,
    creep_tension_limit_n: 72,
    cnc_machined: false,
    kit_cost_usd: 78,
    carriage_mass_delta_g: 8,
    color_hex: "#334155",
  },
  {
    id: "ppa_cf_printed",
    name: "3D-Printed PPA-CF Annealed (11.8 GPa)",
    short_name: "Printed PPA-CF",
    youngs_modulus_gpa: 11.8,
    hdt_c: 196,
    creep_tension_limit_n: 105,
    cnc_machined: false,
    kit_cost_usd: 98,
    carriage_mass_delta_g: 9,
    color_hex: "#2d333b",
  },
  {
    id: "pps_cf_printed",
    name: "3D-Printed PPS-CF Annealed (8.5 GPa, 235°C HDT)",
    short_name: "Printed PPS-CF",
    youngs_modulus_gpa: 8.5,
    hdt_c: 235,
    creep_tension_limit_n: 115,
    cnc_machined: false,
    kit_cost_usd: 120,
    carriage_mass_delta_g: 12,
    color_hex: "#23272e",
  },
  {
    id: "slm_aluminum",
    name: "SLM 3D-Printed Aluminum AlSi10Mg (68 GPa)",
    short_name: "SLM AlSi10Mg",
    youngs_modulus_gpa: 68.0,
    hdt_c: 300,
    creep_tension_limit_n: 230,
    cnc_machined: true,
    manufacturing_process: "slm",
    kit_cost_usd: 135,
    carriage_mass_delta_g: 10,
    color_hex: "#b8c4d0",
  },
  {
    id: "slm_ti6al4v",
    name: "SLM 3D-Printed Titanium Ti-6Al-4V (110 GPa)",
    short_name: "SLM Ti-6Al-4V",
    youngs_modulus_gpa: 110.0,
    hdt_c: 450,
    creep_tension_limit_n: 340,
    cnc_machined: true,
    manufacturing_process: "slm",
    kit_cost_usd: 295,
    carriage_mass_delta_g: 28,
    color_hex: "#7c7d8c",
  },
  {
    id: "slm_maraging_steel",
    name: "SLM Maraging Steel MS1 1.2709 (180 GPa)",
    short_name: "SLM MS1 Steel",
    youngs_modulus_gpa: 180.0,
    hdt_c: 450,
    creep_tension_limit_n: 380,
    cnc_machined: true,
    manufacturing_process: "slm",
    kit_cost_usd: 275,
    carriage_mass_delta_g: 64,
    color_hex: "#6b7280",
  },
  {
    id: "slm_316l_steel",
    name: "SLM 316L Stainless Steel (185 GPa)",
    short_name: "SLM 316L Steel",
    youngs_modulus_gpa: 185.0,
    hdt_c: 450,
    creep_tension_limit_n: 340,
    cnc_machined: true,
    manufacturing_process: "slm",
    kit_cost_usd: 220,
    carriage_mass_delta_g: 62,
    color_hex: "#94a3b8",
  },
  {
    id: "slm_17_4ph_steel",
    name: "SLM 17-4PH Stainless Steel H900 (190 GPa)",
    short_name: "SLM 17-4PH",
    youngs_modulus_gpa: 190.0,
    hdt_c: 450,
    creep_tension_limit_n: 380,
    cnc_machined: true,
    manufacturing_process: "slm",
    kit_cost_usd: 245,
    carriage_mass_delta_g: 60,
    color_hex: "#7c8798",
  },
  {
    id: "sheet_metal_steel",
    name: "Monolith Laser-Cut Steel Sheet Metal (3–5mm, 200 GPa)",
    short_name: "Sheet Metal Steel",
    youngs_modulus_gpa: 200.0,
    hdt_c: 400,
    creep_tension_limit_n: 320,
    cnc_machined: true,
    manufacturing_process: "sheet_metal",
    kit_cost_usd: 85,
    carriage_mass_delta_g: 55,
    color_hex: "#64748b",
  },
  {
    id: "aluminum_6061_t6",
    name: "CNC Aluminum 6061-T6 (68.9 GPa)",
    short_name: "CNC Al 6061",
    youngs_modulus_gpa: 68.9,
    hdt_c: 300,
    creep_tension_limit_n: 250,
    cnc_machined: true,
    manufacturing_process: "cnc",
    kit_cost_usd: 210,
    carriage_mass_delta_g: 42,
    color_hex: "#c6d0db",
  },
  {
    id: "aluminum_7075_t6",
    name: "CNC Aluminum 7075-T6 Billet (71.7 GPa)",
    short_name: "CNC Al 7075",
    youngs_modulus_gpa: 71.7,
    hdt_c: 300,
    creep_tension_limit_n: 280,
    cnc_machined: true,
    manufacturing_process: "cnc",
    kit_cost_usd: 285,
    carriage_mass_delta_g: 46,
    color_hex: "#cbd5e1",
  },
];

const FALLBACK_GANTRY_WEIGHT_CLASSES: GantryWeightClassItem[] = [
  {
    id: "ultralight_printed",
    name: "Ultralight Printed (141g)",
    short_name: "Ultralight Printed",
    carriage_mass_g: 141,
    suggested_bracket_material: "abs_printed",
    notes:
      "Minimalist printed XY joints & lightweight Y-carriages (141 g on MGN12H / 85 g on MGN9H).",
  },
  {
    id: "standard_printed",
    name: "Standard Printed (196g)",
    short_name: "Standard Printed",
    carriage_mass_g: 196,
    suggested_bracket_material: "abs_printed",
    notes:
      "Stock Voron 2.4 / Trident printed ABS/ASA XY joints, idlers & Y-carriages (196 g on MGN12H / 140 g on MGN9H).",
  },
  {
    id: "stiffness_printed_cf",
    name: "Stiffness-Optimized Printed CF (185g)",
    short_name: "Printed CF Stiff",
    carriage_mass_g: 185,
    suggested_bracket_material: "ppa_cf_printed",
    notes: "Beefed-up double-shear PPA-CF / PET-CF XY joints & gusseted Y-carriages.",
  },
  {
    id: "cnc_al_light",
    name: "CNC Aluminum Light (210g)",
    short_name: "CNC Al Light",
    carriage_mass_g: 210,
    suggested_bracket_material: "aluminum_6061_t6",
    notes: "Pocketed CNC 6061-T6 XY joints & lightweight machined Y-carriages.",
  },
  {
    id: "monolith_cnc_awd",
    name: "LDO Monolith CNC 6061 AWD (Scale-Weighed)",
    short_name: "Monolith CNC AWD",
    carriage_mass_g: 214,
    suggested_bracket_material: "aluminum_6061_t6",
    notes:
      "Scale-weighed LDO Monolith CNC 6061 AWD (CloakedWayne): 2× 75.9 g CNC XY-joints + 2× 48.3 g LDO MGN12H + fasteners = 256 g dressed (214 g base + 42 g Al; MGN9H ~210 g).",
  },
  {
    id: "monolith_slm_mgn12h",
    name: "Monolith SLM Aluminum XY-Joints (Scale-Weighed)",
    short_name: "Monolith SLM Al",
    carriage_mass_g: 182.4,
    suggested_bracket_material: "slm_aluminum",
    notes:
      "1-piece SLM AlSi10Mg XY-joints (47.9 g bare each; 96.2 g w/ MGN12H carriage -> 192.4 g dressed pair, saving 56 g/pair vs LDO CNC MGN12H).",
  },
  {
    id: "dderg_ti_mgn9h",
    name: "dderg Ultralight Milled + Hollow Ti Live-Shafts (MGN9H)",
    short_name: "dderg Ti MGN9H",
    carriage_mass_g: 134.6,
    suggested_bracket_material: "aluminum_6061_t6",
    notes:
      "dderg lightweight milled XY-joints with hollow Ti live shafts & Ti hardware (42.1 g bare / 67.3 g w/ MGN9H -> 134.6 g pair, -67.6 g vs LDO MGN9H).",
  },
  {
    id: "monolith_sheet_metal",
    name: "Monolith v1.5 Sheet Metal (3–3.2mm XY-Joints)",
    short_name: "Monolith Sheet Metal",
    carriage_mass_g: 235,
    suggested_bracket_material: "sheet_metal_steel",
    notes:
      "Monolith Builder v1.5 laser-cut steel gantry (3–3.2 mm XY-joint plates, 4.7–5 mm motor mounts & tensioners, 40×9 mm Z belt clamps).",
  },
  {
    id: "cnc_al_heavy",
    name: "CNC Aluminum Heavy-Duty (280g)",
    short_name: "CNC Al Heavy",
    carriage_mass_g: 280,
    suggested_bracket_material: "aluminum_7075_t6",
    notes: "Heavy billet CNC 7075-T6 AWD / double-shear XY blocks & thick Y-carriages.",
  },
];

function isCarbonTubeMaterial(matId: string): boolean {
  return matId.startsWith("carbon_fiber_") && matId !== "carbon_fiber_plate";
}

function inferAutoXBeamMaterial(profileId: string, currentMaterial: string): string | null {
  if (profileId.startsWith("cf_tube_20x20_carbonara")) {
    if (
      currentMaterial !== "carbon_fiber_torsional_45" &&
      currentMaterial !== "carbon_fiber_m40j_hm"
    ) {
      return "carbon_fiber_torsional_45";
    }
    return null;
  }
  if (profileId.startsWith("cf_tube_")) {
    return isCarbonTubeMaterial(currentMaterial) ? null : "carbon_fiber_torsional_45";
  }
  if (profileId.startsWith("ti_tube_")) return "titanium_grade5";
  if (
    profileId === "unsupported_rail" ||
    profileId.startsWith("steel_tube_") ||
    profileId.startsWith("steel_rod_")
  ) {
    return "steel_4140";
  }
  if (profileId === "cnc_al_monocoque_20x20") {
    return currentMaterial === "aluminum_6061_t6" || currentMaterial === "aluminum_7075_t6"
      ? null
      : "aluminum_7075_t6";
  }
  if (profileId === "cnc_al_truss_voron24_lw" || profileId === "cnc_al_fysetc_v0_15x15") {
    return currentMaterial === "aluminum_6061_t6" || currentMaterial === "aluminum_7075_t6"
      ? null
      : "aluminum_6061_t6";
  }
  if (
    (profileId.startsWith("extrusion_") ||
      profileId.startsWith("al_tube_") ||
      profileId.startsWith("cnc_al_")) &&
    (isCarbonTubeMaterial(currentMaterial) ||
      currentMaterial === "titanium_grade5" ||
      currentMaterial === "steel_4140")
  ) {
    return "aluminum_6063_t5";
  }
  return null;
}

const FALLBACK_RAIL_PRELOAD_FRACTION: Record<RailPreloadType, number> = {
  ZF: 0.002,
  Z0: 0.01,
  Z1: 0.03,
  ZA: 0.06,
  ZB: 0.1,
};

const FALLBACK_BEARING_PLAY_UM: Record<BearingClearanceClass, number> = {
  C2: 3.5,
  CN: 7.5,
  C3: 15.5,
};


const FALLBACK_BELT_ROUTING_PARAMS: Record<
  BeltRoutingType,
  { span_factor: number; idler_stiffness_factor: number }
> = {
  stock_corexy: { span_factor: 1.0, idler_stiffness_factor: 1.0 },
  monolith_short: { span_factor: 0.86, idler_stiffness_factor: 1.45 },
};


const BUS_VOLTAGE_PRESETS = [24, 36, 48, 60];

const LINEAR_RAIL_PROFILES: Array<{ id: string; label: string; shortLabel: string }> = [
  { id: "mgn9_rail", label: "MGN9H (9×6.5mm)", shortLabel: "MGN9H" },
  { id: "mgn12_rail", label: "MGN12H (12×8mm)", shortLabel: "MGN12H" },
  { id: "mgn15_rail", label: "MGN15H (15×10mm)", shortLabel: "MGN15H" },
];

const X_LINEAR_RAIL_PROFILES: Array<{ id: string; label: string; shortLabel: string }> = [
  { id: "mgn9_rail", label: "MGN9H (9×6.5mm, 380 g/m)", shortLabel: "MGN9H" },
  {
    id: "mgn9_pocketed_rail",
    label: "Pocketed MGN9H (9×6.5mm, 299 g/m · Tet10)",
    shortLabel: "Pocketed MGN9H",
  },
  { id: "mgn12_rail", label: "MGN12H (12×8mm, 650 g/m)", shortLabel: "MGN12H" },
  {
    id: "mgn12_hollow_rail",
    label: "Hollow MGN12H (12×8mm, 441 g/m · Tet10)",
    shortLabel: "Hollow MGN12H",
  },
  {
    id: "mgn12_pocketed_rail",
    label: "Pocketed MGN12H (12×8mm, 485 g/m · Tet10)",
    shortLabel: "Pocketed MGN12H",
  },
  { id: "mgn15_rail", label: "MGN15H (15×10mm, 1060 g/m)", shortLabel: "MGN15H" },
  {
    id: "mgn15_pocketed_rail",
    label: "Pocketed MGN15H (15×10mm, 731 g/m · Tet10)",
    shortLabel: "Pocketed MGN15H",
  },
];

const FALLBACK_X_RAIL_MOUNTS: RailMountOption<XRailMountType>[] = [
  {
    id: "front",
    name: "Front Face (-Y)",
    short_name: "Front",
    notes: "Standard Voron front mount — maximum Y-axis composite bending stiffness",
  },
  {
    id: "top",
    name: "Top Face (+Z)",
    short_name: "Top",
    notes: "Top-mounted X rail — maximum vertical Z-axis stiffness",
  },
  {
    id: "bottom",
    name: "Bottom Face (-Z)",
    short_name: "Bottom",
    notes: "Under-slung X rail — shorter nozzle drop and lower X-beam pitch twist",
  },
  {
    id: "dual_top_bottom",
    name: "Dual Top + Bottom (±Z)",
    short_name: "Dual Top+Bot",
    notes: "Symmetric twin steel rails top & bottom — highest Z/torsional stiffness",
  },
];

const FALLBACK_Y_RAIL_MOUNTS: RailMountOption<YRailMountType>[] = [
  {
    id: "top",
    name: "Top Face (+Z)",
    short_name: "Top",
    notes: "Standard top-face Y linear rails",
  },
  {
    id: "inner",
    name: "Inner Side (±X)",
    short_name: "Inner Side",
    notes: "Inward side-mounted Y rails — maximum lateral X-axis bending stiffness",
  },
  {
    id: "bottom",
    name: "Bottom Face (-Z)",
    short_name: "Bottom",
    notes: "Under-slung bottom-face Y rails",
  },
];

function railProfileShort(id?: string): string {
  if (!id) return "MGN12H";
  const found = X_LINEAR_RAIL_PROFILES.find((r) => r.id === id);
  if (found) return found.shortLabel;
  if (id.includes("mgn9")) return "MGN9H";
  if (id.includes("mgn12")) return "MGN12H";
  return id;
}

function shortMotorName(name: string): string {
  return name.split("(")[0].trim() || name;
}

const RAIL_PRELOAD_OPTIONS: Array<{ id: RailPreloadType; label: string; shortLabel: string }> = [
  { id: "ZF", label: "ZF Clearance", shortLabel: "ZF" },
  { id: "Z0", label: "Z0 Very Light", shortLabel: "Z0" },
  { id: "Z1", label: "Z1 Light Preload", shortLabel: "Z1" },
  { id: "ZA", label: "ZA Medium", shortLabel: "ZA" },
  { id: "ZB", label: "ZB Heavy", shortLabel: "ZB" },
];

const FALLBACK_RAIL_QUALITIES: Array<{
  id: RailQualityType;
  name: string;
  short_name: string;
  stiffness_mult: number;
  drag_mult: number;
  cost_mult: number;
  notes: string;
}> = [
  {
    id: "budget_generic",
    name: "Budget / Generic Clone",
    short_name: "Budget Clone",
    stiffness_mult: 0.55,
    drag_mult: 1.45,
    cost_mult: 0.45,
    notes: "Loose ball-groove tolerance & uneven preload — noticeable carriage roll play.",
  },
  {
    id: "midgrade_matched",
    name: "Matched 3D-Printer Grade (LDO / HoneyBadger)",
    short_name: "LDO / Matched",
    stiffness_mult: 0.85,
    drag_mult: 1.05,
    cost_mult: 0.78,
    notes: "QC-matched carriage & rail set; smooth glide and solid value for CoreXY builds.",
  },
  {
    id: "hiwin_genuine",
    name: "Genuine HIWIN / PMI (Grade C/H)",
    short_name: "HIWIN",
    stiffness_mult: 1.0,
    drag_mult: 1.0,
    cost_mult: 1.0,
    notes: "Datasheet Hertzian ball contact stiffness with consistent preload and low drag.",
  },
  {
    id: "precision_thk_nsk",
    name: "Precision Ground (THK / NSK / IKO Grade P)",
    short_name: "THK / NSK P",
    stiffness_mult: 1.28,
    drag_mult: 0.85,
    cost_mult: 2.1,
    notes: "Precision-ground raceways & tight ball sorting for peak roll stiffness and ultra-low drag.",
  },
];

const RAIL_BLOCK_OPTIONS: Array<{ id: RailBlockType; label: string; shortLabel: string }> = [
  { id: "C", label: "C — Standard Block (Lighter)", shortLabel: "C" },
  { id: "H", label: "H — Long Block (High Roll/Pitch)", shortLabel: "H" },
];

const RAIL_BACKER_OPTIONS: Array<{ id: RailBackerMaterialType; label: string; shortLabel: string }> = [
  { id: "none", label: "None", shortLabel: "None" },
  { id: "titanium_grade5", label: "Titanium Gr5 (Anti-Bow)", shortLabel: "Ti Gr5" },
  { id: "steel_4140", label: "Steel 4140 (Anti-Bow)", shortLabel: "Steel 4140" },
];

const CF_RAIL_MOUNTING_OPTIONS: Array<{
  id: CFRailMountingType;
  label: string;
  shortLabel: string;
}> = [
  { id: "auto", label: "Auto (Carbonara 2-Wall / Bare)", shortLabel: "Auto" },
  {
    id: "two_wall_clamped_inserts",
    label: "2-Wall Clamped Inserts (Carbonara · M3×28)",
    shortLabel: "2-Wall Inserts",
  },
  {
    id: "single_wall_bare",
    label: "Single-Wall Bare Bolted (1 Wall)",
    shortLabel: "Single-Wall Bare",
  },
];

const BEARING_CLEARANCE_OPTIONS: Array<{ id: BearingClearanceClass; label: string; shortLabel: string }> = [
  { id: "C2", label: "C2 Tight", shortLabel: "C2" },
  { id: "CN", label: "CN Standard", shortLabel: "CN" },
  { id: "C3", label: "C3 Loose", shortLabel: "C3" },
];


const BELT_ROUTING_OPTIONS: Array<{
  id: BeltRoutingType;
  label: string;
  shortLabel: string;
  hint: string;
}> = [
  {
    id: "stock_corexy",
    label: "Stock CoreXY Routing",
    shortLabel: "Stock CoreXY",
    hint: "Standard full-perimeter CoreXY belt loop with conventional fixed-pin idlers.",
  },
  {
    id: "monolith_short",
    label: "Monolith Short Path + Live-Shaft Idlers",
    shortLabel: "Monolith Short",
    hint: "Shortened belt spans + rigid live-shaft idler bearings boost belt frequency and curb idler deflection.",
  },
];

export interface DesignGantrySectionProps {
  spec: PrinterSpecData;
  catalog: CatalogResponse;
  sim: SimulationResponse | null;
  showHints: boolean;
  advancedMode: boolean;
  update: (mutator: (draft: PrinterSpecData) => void, changeLabel?: string) => void;
  updatePositiveNum: (
    raw: string,
    mutator: (draft: PrinterSpecData, val: number) => void,
    changeLabelBuilder: (val: number) => string
  ) => void;
  profileLabel: (id: string) => string;
  materialLabel: (id: string) => string;
}

export const DesignGantrySection: React.FC<DesignGantrySectionProps> = ({
  spec,
  catalog,
  sim,
  showHints,
  advancedMode,
  update,
  updatePositiveNum,
  profileLabel,
  materialLabel,
}) => {
  // Smooth local drafts for gantry sliders (avoid API spam while dragging)
  const [tensionDraft, setTensionDraft] = useState<number>(spec.gantry.belt.pretension_n ?? 28);
  const beltDyn = sim?.modal_fea?.belt_dynamics ?? null;
  const tensionTimerRef = useRef<number | null>(null);

  const [carriageMassDraft, setCarriageMassDraft] = useState<number>(
    Math.round(spec.gantry.carriage_mass_g ?? 140)
  );
  const carriageMassTimerRef = useRef<number | null>(null);

  const [runCurrentDraft, setRunCurrentDraft] = useState<number>(
    spec.kinematics.xy_stepper.run_current_a ?? 1.2
  );
  const currentTimerRef = useRef<number | null>(null);

  const [notchDraft, setNotchDraft] = useState<string>(
    spec.kinematics.notch_filter_hz != null ? String(Math.round(spec.kinematics.notch_filter_hz)) : ""
  );

  const [awdEqBwDraft, setAwdEqBwDraft] = useState<number>(
    Number((spec.kinematics.awd_equalizer_bw_hz ?? 2.0).toFixed(1))
  );
  const awdEqBwTimerRef = useRef<number | null>(null);

  const [notchDepthDraft, setNotchDepthDraft] = useState<number>(
    Number((spec.kinematics.notch_depth_db ?? 12.0).toFixed(1))
  );
  const notchDepthTimerRef = useRef<number | null>(null);

  const [notchQDraft, setNotchQDraft] = useState<number>(
    Number((spec.kinematics.notch_q ?? 1.75).toFixed(2))
  );
  const notchQTimerRef = useRef<number | null>(null);

  useEffect(() => {
    if (spec.gantry.belt.pretension_n != null) {
      setTensionDraft(spec.gantry.belt.pretension_n);
    }
  }, [spec.gantry.belt.pretension_n]);

  useEffect(() => {
    if (spec.gantry.carriage_mass_g != null) {
      setCarriageMassDraft(Math.round(spec.gantry.carriage_mass_g));
    }
  }, [spec.gantry.carriage_mass_g]);

  useEffect(() => {
    const nextA =
      spec.kinematics.xy_stepper.run_current_a ??
      Math.min(spec.kinematics.xy_stepper.rated_current_a || 1.2, 1.2);
    setRunCurrentDraft(Number(nextA.toFixed(1)));
  }, [
    spec.kinematics.xy_stepper.run_current_a,
    spec.kinematics.xy_stepper.rated_current_a,
    spec.kinematics.xy_stepper.preset_id,
    spec.kinematics.xy_stepper.name,
  ]);

  useEffect(() => {
    const nf = spec.kinematics.notch_filter_hz;
    setNotchDraft(nf != null && nf > 0 ? String(Math.round(nf)) : "");
  }, [spec.kinematics.notch_filter_hz]);

  useEffect(() => {
    setAwdEqBwDraft(Number((spec.kinematics.awd_equalizer_bw_hz ?? 2.0).toFixed(1)));
  }, [spec.kinematics.awd_equalizer_bw_hz]);

  useEffect(() => {
    setNotchDepthDraft(Number((spec.kinematics.notch_depth_db ?? 12.0).toFixed(1)));
  }, [spec.kinematics.notch_depth_db]);

  useEffect(() => {
    setNotchQDraft(Number((spec.kinematics.notch_q ?? 1.75).toFixed(2)));
  }, [spec.kinematics.notch_q]);

  useEffect(() => {
    return () => {
      if (tensionTimerRef.current != null) window.clearTimeout(tensionTimerRef.current);
      if (carriageMassTimerRef.current != null) window.clearTimeout(carriageMassTimerRef.current);
      if (currentTimerRef.current != null) window.clearTimeout(currentTimerRef.current);
      if (awdEqBwTimerRef.current != null) window.clearTimeout(awdEqBwTimerRef.current);
      if (notchDepthTimerRef.current != null) window.clearTimeout(notchDepthTimerRef.current);
      if (notchQTimerRef.current != null) window.clearTimeout(notchQTimerRef.current);
    };
  }, []);

  const beltTensionMax = spec
    ? Math.max(
        getBeltTensionSliderMax(spec.gantry.belt.type),
        Math.min(250, Math.round(spec.gantry.belt.pretension_n))
      )
    : 120;

  const commitBeltTension = (rawVal: number) => {
    const val = Math.max(5, Math.min(beltTensionMax, Math.round(rawVal)));
    if (tensionTimerRef.current != null) {
      window.clearTimeout(tensionTimerRef.current);
      tensionTimerRef.current = null;
    }
    if (val === spec.gantry.belt.pretension_n) return;
    update(
      (d) => {
        d.gantry.belt.pretension_n = val;
      },
      `Belt Tension: ${spec.gantry.belt.pretension_n} N → ${val} N`
    );
  };

  const handleTensionSliderChange = (nextVal: number) => {
    setTensionDraft(nextVal);
    if (tensionTimerRef.current != null) window.clearTimeout(tensionTimerRef.current);
    tensionTimerRef.current = window.setTimeout(() => {
      commitBeltTension(nextVal);
    }, 220);
  };

  const commitCarriageMass = (rawVal: number) => {
    const val = Math.max(20, Math.min(600, Math.round(rawVal)));
    if (carriageMassTimerRef.current != null) {
      window.clearTimeout(carriageMassTimerRef.current);
      carriageMassTimerRef.current = null;
    }
    if (Math.round(spec.gantry.carriage_mass_g) === val) return;
    update(
      (d) => {
        d.gantry.carriage_mass_g = val;
      },
      `XY-Joints & Y-Carriages Mass: ${Math.round(spec.gantry.carriage_mass_g)}g → ${val}g`
    );
  };

  const handleCarriageMassSliderChange = (nextVal: number) => {
    const val = Math.max(20, Math.min(600, Math.round(nextVal)));
    setCarriageMassDraft(val);
    if (carriageMassTimerRef.current != null) window.clearTimeout(carriageMassTimerRef.current);
    carriageMassTimerRef.current = window.setTimeout(() => {
      commitCarriageMass(val);
    }, 220);
  };

  const specRunCurrentA = Number(
    (
      spec.kinematics.xy_stepper.run_current_a ??
      Math.min(spec.kinematics.xy_stepper.rated_current_a || 1.2, 1.2)
    ).toFixed(1)
  );

  const runCurrentMaxA = Math.max(
    3.2,
    Math.min(8.0, Math.ceil((spec.kinematics.xy_stepper.rated_current_a || 2.5) * 1.25 * 10) / 10)
  );

  const commitRunCurrent = (rawVal: number) => {
    const val = Math.max(0.4, Math.min(runCurrentMaxA, Math.round(rawVal * 10) / 10));
    if (currentTimerRef.current != null) {
      window.clearTimeout(currentTimerRef.current);
      currentTimerRef.current = null;
    }
    if (Math.abs(val - specRunCurrentA) < 0.01) return;
    update(
      (d) => {
        d.kinematics.xy_stepper.run_current_a = val;
      },
      `Motor Current: ${specRunCurrentA.toFixed(1)}A → ${val.toFixed(1)}A`
    );
  };

  const handleRunCurrentSliderChange = (nextVal: number) => {
    const rounded = Math.round(nextVal * 10) / 10;
    setRunCurrentDraft(rounded);
    if (currentTimerRef.current != null) window.clearTimeout(currentTimerRef.current);
    currentTimerRef.current = window.setTimeout(() => {
      commitRunCurrent(rounded);
    }, 220);
  };

  const currentRailPreload: RailPreloadType = spec.gantry.rail_preload ?? "Z0";
  const currentRailQuality: RailQualityType = spec.gantry.rail_quality ?? "hiwin_genuine";
  const railQualityOptions =
    catalog.rail_qualities && catalog.rail_qualities.length > 0
      ? catalog.rail_qualities
      : FALLBACK_RAIL_QUALITIES;
  const selectedRailQuality =
    railQualityOptions.find((q) => q.id === currentRailQuality) ?? railQualityOptions[2] ?? railQualityOptions[0];
  const currentRailBlock: RailBlockType = spec.gantry.rail_block_type ?? "H";
  const currentRailBoltPitch: number = spec.gantry.rail_bolt_pitch_mm ?? 25;
  const currentRailBacker: RailBackerMaterialType = spec.gantry.rail_backer_material ?? "none";
  const currentCfRailMounting: CFRailMountingType = spec.gantry.cf_rail_mounting ?? "auto";
  const isHollowTubeXBeam =
    spec.gantry.x_beam_profile.includes("_tube_") ||
    spec.gantry.x_beam_profile.startsWith("cf_") ||
    spec.gantry.x_beam_material.startsWith("carbon_fiber");
  const currentIdlerShaftMm: number = spec.gantry.belt.idler_shaft_diameter_mm ?? 5;
  const currentBearingClass: BearingClearanceClass = spec.gantry.belt.bearing_clearance_class ?? "CN";
  const currentDriveType: DriveType = spec.kinematics.drive_type ?? "open_loop_stepper";
  const currentBeltRouting: BeltRoutingType = spec.gantry.belt_routing ?? "stock_corexy";

  const beltRoutingCatalogEntry = catalog.belt_routings?.find((r) => r.id === currentBeltRouting);
  const beltRoutingOptionMeta =
    BELT_ROUTING_OPTIONS.find((o) => o.id === currentBeltRouting) ?? BELT_ROUTING_OPTIONS[0];

  const currentServoVelBw: number = spec.kinematics.servo_vel_bw_hz ?? 120;

  const commitNotchFilter = (raw: string) => {
    const trimmed = raw.trim();
    const prevVal = spec.kinematics.notch_filter_hz ?? null;
    if (trimmed === "") {
      if (prevVal === null) return;
      update(
        (d) => {
          d.kinematics.notch_filter_hz = null;
        },
        `Notch Filter: ${Math.round(prevVal)} Hz → Off`
      );
      return;
    }
    const parsed = Number(trimmed);
    if (!Number.isFinite(parsed) || parsed <= 0) return;
    const clamped = Math.max(20, Math.min(500, Math.round(parsed)));
    setNotchDraft(String(clamped));
    if (prevVal != null && Math.round(prevVal) === clamped) return;
    const prevLbl = prevVal != null && prevVal > 0 ? `${Math.round(prevVal)} Hz` : "Off";
    update(
      (d) => {
        d.kinematics.notch_filter_hz = clamped;
      },
      `Notch Filter: ${prevLbl} → ${clamped} Hz`
    );
  };

  const driveMode: "2wd" | "awd" = spec.gantry.drive_mode || "2wd";
  const isOpenLoopDrive = currentDriveType === "open_loop_stepper";
  const currentLpv = Boolean(spec.kinematics.lpv_workspace_scheduling ?? false);
  const currentAwdSync: AwdSyncMode = spec.kinematics.awd_sync_mode ?? "none";
  const currentAwdEqBw = spec.kinematics.awd_equalizer_bw_hz ?? 2.0;
  const currentNotchDepthDb = spec.kinematics.notch_depth_db ?? 12.0;
  const currentNotchQ = spec.kinematics.notch_q ?? 1.75;
  const currentPinRotor = Boolean(spec.kinematics.pin_rotor_ff ?? false);
  const currentImuDamping = Boolean(spec.kinematics.toolhead_imu_active_damping ?? false);
  const currentModeSpaceFf = Boolean(spec.kinematics.mode_space_torque_ff ?? false);

  const activeBetaCount =
    (spec.kinematics.input_shaper === "smooth_flatness" ? 1 : 0) +
    (currentLpv ? 1 : 0) +
    (currentAwdSync !== "none" ? 1 : 0) +
    (currentPinRotor ? 1 : 0) +
    (currentImuDamping ? 1 : 0) +
    (currentModeSpaceFf ? 1 : 0);

  const expectedBetaAwdSync: AwdSyncMode =
    driveMode === "awd" ? "position_equalizer" : "none";
  const isFocBetaStackActive =
    currentDriveType === "foc_servo" &&
    spec.kinematics.input_shaper === "smooth_flatness" &&
    currentLpv &&
    currentAwdSync === expectedBetaAwdSync &&
    currentPinRotor &&
    currentModeSpaceFf;

  const notchPhaseMetrics = computeNotchPhaseMetrics(notchDepthDraft, notchQDraft);

  const commitAwdEqBw = (nextVal: number) => {
    const clamped = Math.max(0.5, Math.min(15.0, Math.round(nextVal * 2) / 2));
    if (Math.abs(clamped - currentAwdEqBw) < 0.05) return;
    update(
      (d) => {
        d.kinematics.awd_equalizer_bw_hz = clamped;
      },
      `AWD Equalizer BW: ${currentAwdEqBw.toFixed(1)} Hz → ${clamped.toFixed(1)} Hz`
    );
  };

  const handleAwdEqBwSliderChange = (nextVal: number) => {
    const rounded = Math.round(nextVal * 2) / 2;
    setAwdEqBwDraft(rounded);
    if (awdEqBwTimerRef.current != null) window.clearTimeout(awdEqBwTimerRef.current);
    awdEqBwTimerRef.current = window.setTimeout(() => commitAwdEqBw(rounded), 200);
  };

  const commitNotchDepth = (nextVal: number) => {
    const clamped = Math.max(3.0, Math.min(30.0, Math.round(nextVal * 2) / 2));
    if (Math.abs(clamped - currentNotchDepthDb) < 0.05) return;
    update(
      (d) => {
        d.kinematics.notch_depth_db = clamped;
      },
      `Notch Depth: ${currentNotchDepthDb.toFixed(1)} dB → ${clamped.toFixed(1)} dB`
    );
  };

  const handleNotchDepthSliderChange = (nextVal: number) => {
    const rounded = Math.round(nextVal * 2) / 2;
    setNotchDepthDraft(rounded);
    if (notchDepthTimerRef.current != null) window.clearTimeout(notchDepthTimerRef.current);
    notchDepthTimerRef.current = window.setTimeout(() => commitNotchDepth(rounded), 200);
  };

  const commitNotchQ = (nextVal: number) => {
    const clamped = Math.max(0.5, Math.min(8.0, Math.round(nextVal * 20) / 20));
    if (Math.abs(clamped - currentNotchQ) < 0.01) return;
    update(
      (d) => {
        d.kinematics.notch_q = clamped;
      },
      `Notch Q: ${currentNotchQ.toFixed(2)} → ${clamped.toFixed(2)}`
    );
  };

  const handleNotchQSliderChange = (nextVal: number) => {
    const rounded = Math.round(nextVal * 20) / 20;
    setNotchQDraft(rounded);
    if (notchQTimerRef.current != null) window.clearTimeout(notchQTimerRef.current);
    notchQTimerRef.current = window.setTimeout(() => commitNotchQ(rounded), 200);
  };

  const handleLoadFocBetaStack = () => {
    update(
      (d) => {
        d.kinematics.drive_type = "foc_servo";
        if (d.kinematics.servo_vel_bw_hz == null) {
          d.kinematics.servo_vel_bw_hz = 180;
        }
        d.kinematics.input_shaper = "smooth_flatness";
        d.kinematics.lpv_workspace_scheduling = true;
        d.kinematics.awd_sync_mode =
          d.gantry.drive_mode === "awd" ? "position_equalizer" : "none";
        d.kinematics.awd_equalizer_bw_hz = 2.0;
        d.kinematics.pin_rotor_ff = true;
        d.kinematics.mode_space_torque_ff = true;
      },
      "Loaded Kalico FOC Beta Stack"
    );
  };

  const handleDisableBetaStack = () => {
    update(
      (d) => {
        if (d.kinematics.input_shaper === "smooth_flatness") {
          d.kinematics.input_shaper = "auto";
        }
        d.kinematics.lpv_workspace_scheduling = false;
        d.kinematics.awd_sync_mode = "none";
        d.kinematics.pin_rotor_ff = false;
        d.kinematics.toolhead_imu_active_damping = false;
        d.kinematics.mode_space_torque_ff = false;
      },
      "Disabled Kalico / FOC Beta Stack"
    );
  };

  const doubleShearPulleys = Boolean(spec.gantry.double_shear_pulleys);
  const beltPluckHz = sim?.kinematics.belt_pluck_freq_150mm_hz;
  const beltPeakTensionN = sim?.kinematics.belt_peak_tension_n;
  const beltMaxWorkingN = sim?.kinematics.belt_max_working_tension_n;
  const beltOvertension = Boolean(sim?.kinematics.belt_overtension_risk);
  const shaftBendingRisk = Boolean(sim?.kinematics.shaft_bending_risk);

  const gantryBrackets: GantryBracketInfo[] =
    catalog.gantry_brackets && catalog.gantry_brackets.length > 0
      ? catalog.gantry_brackets
      : FALLBACK_GANTRY_BRACKETS;
  const gantryHardwareKits: GantryHardwareKitItem[] =
    catalog.gantry_hardware_kits && catalog.gantry_hardware_kits.length > 0
      ? catalog.gantry_hardware_kits
      : FALLBACK_GANTRY_HARDWARE_KITS;
  const currentHardwareKit: GantryHardwareKitType = spec.gantry.hardware_kit ?? "auto";
  const simGantryBrackets = sim?.geometry?.gantry_brackets;
  const isCadGantryKitActive = Boolean(simGantryBrackets?.cad_backed);
  const resolvedHardwareKitLabel =
    simGantryBrackets?.hardware_kit_label ??
    gantryHardwareKits.find((k) => k.id === simGantryBrackets?.hardware_kit)?.short_name ??
    null;
  const currentBracketId: BracketMaterialType = spec.gantry.bracket_material ?? "abs_printed";
  const selectedBracket =
    gantryBrackets.find((b) => b.id === currentBracketId) ?? gantryBrackets[0];
  const customBracketOverride = spec.gantry.custom_bracket_override ?? null;
  const simBracketStiffness = sim?.kinematics.bracket_stiffness_n_per_mm;
  const bracketCreepRisk = Boolean(sim?.kinematics.bracket_creep_risk);

  const gantryWeightClasses: GantryWeightClassItem[] =
    catalog.gantry_weight_classes && catalog.gantry_weight_classes.length > 0
      ? catalog.gantry_weight_classes
      : FALLBACK_GANTRY_WEIGHT_CLASSES;
  const matchedWeightClass =
    gantryWeightClasses.find(
      (cls) => Math.abs(cls.carriage_mass_g - spec.gantry.carriage_mass_g) < 0.5
    ) ?? null;
  const bracketDeltaG = customBracketOverride
    ? Math.round(customBracketOverride.mass_delta_g ?? 0)
    : Math.round(selectedBracket.carriage_mass_delta_g ?? 0);
  const isSelectedBracketSlm =
    selectedBracket.manufacturing_process === "slm" || selectedBracket.id.startsWith("slm_");
  const isSelectedBracketSheetMetal =
    selectedBracket.manufacturing_process === "sheet_metal" ||
    selectedBracket.id === "sheet_metal_steel";
  const bracketDeltaShortLabel = customBracketOverride
    ? "CAD"
    : isSelectedBracketSlm
      ? "SLM"
      : isSelectedBracketSheetMetal
        ? "Steel"
        : selectedBracket.cnc_machined
          ? "CNC Al"
          : selectedBracket.short_name.replace(/^Printed\s+/i, "");
  const baseCarriageMassG = Math.round(carriageMassDraft);
  const dressedMovingXYJointsG = baseCarriageMassG + bracketDeltaG;
  const simMassSummary = sim?.geometry?.mass_summary ?? null;

  const xRailMounts: RailMountOption<XRailMountType>[] =
    catalog.x_rail_mounts && catalog.x_rail_mounts.length > 0
      ? catalog.x_rail_mounts
      : FALLBACK_X_RAIL_MOUNTS;
  const yRailMounts: RailMountOption<YRailMountType>[] =
    catalog.y_rail_mounts && catalog.y_rail_mounts.length > 0
      ? catalog.y_rail_mounts
      : FALLBACK_Y_RAIL_MOUNTS;
  const currentXRailMount: XRailMountType =
    spec.gantry.x_rail_mount ?? (spec.gantry.dual_x_rails ? "dual_top_bottom" : "front");
  const currentYRailMount: YRailMountType = spec.gantry.y_rail_mount ?? "top";
  const selectedXRailMount =
    xRailMounts.find((m) => m.id === currentXRailMount) ?? xRailMounts[0];
  const selectedYRailMount =
    yRailMounts.find((m) => m.id === currentYRailMount) ?? yRailMounts[0];
  const isFlyingGantry =
    spec.architecture === "corexy" && spec.bed.z_drive === "belted_quad_z";
  const currentXRailProfile = spec.gantry.x_rail_profile || "mgn12_rail";
  const currentYRailProfile = spec.gantry.y_rail_profile || "mgn9_rail";
  const xRailProfileOptions = X_LINEAR_RAIL_PROFILES.some((r) => r.id === currentXRailProfile)
    ? X_LINEAR_RAIL_PROFILES
    : [
        ...X_LINEAR_RAIL_PROFILES,
        { id: currentXRailProfile, label: profileLabel(currentXRailProfile), shortLabel: railProfileShort(currentXRailProfile) },
      ];
  const yRailProfileOptions = LINEAR_RAIL_PROFILES.some((r) => r.id === currentYRailProfile)
    ? LINEAR_RAIL_PROFILES
    : [
        ...LINEAR_RAIL_PROFILES,
        { id: currentYRailProfile, label: profileLabel(currentYRailProfile), shortLabel: railProfileShort(currentYRailProfile) },
      ];

  const stepperCatalog = catalog.steppers ?? [];
  const xyStepper = spec.kinematics.xy_stepper;
  const matchedStepper =
    stepperCatalog.find((s) => s.preset_id === xyStepper.preset_id) ??
    stepperCatalog.find((s) => s.name === xyStepper.name);
  const selectedStepperValue = matchedStepper ? matchedStepper.preset_id : "__custom__";
  const ratedCurrentA =
    typeof xyStepper.rated_current_a === "number" &&
    Number.isFinite(xyStepper.rated_current_a) &&
    xyStepper.rated_current_a > 0
      ? xyStepper.rated_current_a
      : 2.0;
  const currentPctOfRated = Math.round((runCurrentDraft / ratedCurrentA) * 100);
  const phaseResistanceOhm = xyStepper.phase_resistance_ohm ?? 1.2;
  const estMotorHeatW = 2.0 * runCurrentDraft * runCurrentDraft * phaseResistanceOhm;
  const simXyStepper = sim?.spec?.kinematics?.xy_stepper;
  const simMatchesMotor = Boolean(
    simXyStepper &&
      simXyStepper.preset_id === xyStepper.preset_id &&
      Math.abs((simXyStepper.rated_current_a ?? 0) - ratedCurrentA) < 1e-3 &&
      Math.abs(
        (sim?.kinematics.motor_run_current_a ?? simXyStepper.run_current_a ?? 0) - runCurrentDraft
      ) < 0.05
  );
  const motorOvercurrentDraft =
    runCurrentDraft > ratedCurrentA * 1.05 + 1e-6 ||
    (simMatchesMotor && Boolean(sim?.kinematics.motor_overcurrent_risk));

  return (
            <div className="section-card" data-design-section="gantry">
              <div className="section-title">
                <span>Gantry, Belts &amp; Kinematics</span>
                {simMassSummary && (
                  <span className="sub-block-tag" style={{ fontVariantNumeric: "tabular-nums" }}>
                    X {Math.round(simMassSummary.x_moving_mass_g)}g / Y{" "}
                    {Math.round(simMassSummary.y_moving_mass_g)}g
                  </span>
                )}
              </div>

              {customBracketOverride && (
                <div className="custom-bracket-pill">
                  <div className="custom-bracket-pill-text">
                    <span className="custom-bracket-dot" />
                    <span>
                      Custom CAD Bracket Active:{" "}
                      <strong>{customBracketOverride.label}</strong> (
                      {Math.round(customBracketOverride.stiffness_n_per_mm)} N/mm)
                    </span>
                  </div>
                  <button
                    type="button"
                    className="custom-bracket-clear-btn"
                    onClick={() =>
                      update(
                        (d) => {
                          d.gantry.custom_bracket_override = null;
                        },
                        `Custom CAD Bracket: ${customBracketOverride.label} → Cleared`
                      )
                    }
                  >
                    Clear
                  </button>
                </div>
              )}

              {spec.architecture === "corexy" && (() => {
                const kitAffectsItems = getGantryKitAffectsItems(
                  currentHardwareKit,
                  gantryHardwareKits
                );
                return (
                  <div className="field-row">
                    <label htmlFor="dp-gantry-hardware-kit">
                      <span>Gantry Kit / Upgrade</span>
                      {isCadGantryKitActive && (
                        <span
                          className="sub-block-tag ok"
                          title={
                            simGantryBrackets?.cad_stiffness_n_per_mm
                              ? `CAD-backed gantry hardware (${Math.round(simGantryBrackets.cad_stiffness_n_per_mm)} N/mm)`
                              : "Real 3D CAD gantry hardware meshes & stiffness active"
                          }
                        >
                          {currentHardwareKit === "auto" && resolvedHardwareKitLabel
                            ? `CAD · ${resolvedHardwareKitLabel}`
                            : "CAD-backed"}
                        </span>
                      )}
                    </label>
                    <select
                      id="dp-gantry-hardware-kit"
                      value={currentHardwareKit}
                      onChange={(e) => {
                        const nextKit = e.target.value as GantryHardwareKitType;
                        const prevLabel =
                          gantryHardwareKits.find((k) => k.id === currentHardwareKit)?.short_name ??
                          currentHardwareKit;
                        const nextLabel =
                          gantryHardwareKits.find((k) => k.id === nextKit)?.short_name ?? nextKit;
                        const draftPreview: PrinterSpecData = JSON.parse(JSON.stringify(spec));
                        const syncedChanges = applyGantryHardwareKit(
                          draftPreview,
                          nextKit,
                          gantryHardwareKits
                        );
                        const suffix =
                          syncedChanges.length > 0 ? ` (${syncedChanges.join(", ")})` : "";
                        update(
                          (d) => {
                            applyGantryHardwareKit(d, nextKit, gantryHardwareKits);
                          },
                          `Gantry Upgrade: ${prevLabel} → ${nextLabel}${suffix}`
                        );
                      }}
                    >
                      {gantryHardwareKits
                        .filter((kit) => kit.id !== "auto" || currentHardwareKit === "auto")
                        .map((kit) => (
                          <option key={kit.id} value={kit.id}>
                            {kit.name}
                          </option>
                        ))}
                    </select>
                    <PresetAffectsStrip
                      visible={kitAffectsItems.length > 0}
                      items={kitAffectsItems}
                      suffix="· customize any field below"
                    />
                    {showHints && (
                      <div className="hint-line">
                        Upgrading your gantry kit automatically applies its drive mode, belt
                        routing, bracket material, double-shear pulleys, rail size &amp;
                        orientation, belt width/tension, and V2.4 Z-joint hardware.
                      </div>
                    )}
                  </div>
                );
              })()}

              {spec.architecture === "corexy" && (
                <div className="field-row">
                  <label id="dp-xy-drive-mode-1">
                    <span>XY Drive Mode</span>
                    <span style={{ color: "var(--accent)", fontWeight: 500 }}>
                      {driveMode === "awd" ? "4 Motors (AWD)" : "2 Motors (2WD)"}
                    </span>
                  </label>
                  <div role="group" aria-labelledby="dp-xy-drive-mode-1" className="engine-toggle">
                    <button
                      type="button"
                      className={`engine-toggle-btn ${driveMode === "2wd" ? "active" : ""}`}
 aria-pressed={driveMode === "2wd"}
                      onClick={() => {
                        if (driveMode === "2wd") return;
                        const preview: PrinterSpecData = JSON.parse(JSON.stringify(spec));
                        const { switchedKitId, changes } = syncDriveModeWithHardwareKit(
                          preview,
                          "2wd",
                          gantryHardwareKits
                        );
                        const kitNote = switchedKitId
                          ? ` + ${gantryHardwareKits.find((k) => k.id === switchedKitId)?.short_name ?? switchedKitId}`
                          : "";
                        const detailNote =
                          changes.length > 1 ? ` (${changes.slice(1).join(", ")})` : "";
                        update(
                          (d) => {
                            syncDriveModeWithHardwareKit(d, "2wd", gantryHardwareKits);
                          },
                          `Drive Mode: AWD → 2WD${kitNote}${detailNote}`
                        );
                      }}
                    >
                      2WD CoreXY
                      <span className="engine-toggle-sub">2 Motors</span>
                    </button>
                    <button
                      type="button"
                      className={`engine-toggle-btn ${driveMode === "awd" ? "active" : ""}`}
 aria-pressed={driveMode === "awd"}
                      onClick={() => {
                        if (driveMode === "awd") return;
                        const preview: PrinterSpecData = JSON.parse(JSON.stringify(spec));
                        const { switchedKitId, changes } = syncDriveModeWithHardwareKit(
                          preview,
                          "awd",
                          gantryHardwareKits
                        );
                        const kitNote = switchedKitId
                          ? ` + ${gantryHardwareKits.find((k) => k.id === switchedKitId)?.short_name ?? switchedKitId}`
                          : "";
                        const detailNote =
                          changes.length > 1 ? ` (${changes.slice(1).join(", ")})` : "";
                        update(
                          (d) => {
                            syncDriveModeWithHardwareKit(d, "awd", gantryHardwareKits);
                          },
                          `Drive Mode: 2WD → AWD${kitNote}${detailNote}`
                        );
                      }}
                    >
                      AWD CoreXY
                      <span className="engine-toggle-sub">4 Motors</span>
                    </button>
                  </div>
                  {showHints && (
                    <div className="hint-line">
                      AWD adds 2 front XY motors for 2× torque &amp; half belt span; switching on
                      Siboor CNC or Monolith CNC also switches to the matching 2WD/AWD CAD kit.
                    </div>
                  )}
                </div>
              )}

              {showHints && (
                <div className="hint-line">Lighter X-beam = faster Y acceleration &amp; less ringing</div>
              )}

              <div className="field-grid-2 stack">
                <div className="field-row">
                  <label htmlFor="dp-x-beam-profile-2">X Beam Profile</label>
                  <select id="dp-x-beam-profile-2"
                    value={spec.gantry.x_beam_profile}
                    onChange={(e) => {
                      const nextVal = e.target.value;
                      const autoMat = inferAutoXBeamMaterial(nextVal, spec.gantry.x_beam_material);
                      const shouldPairMat =
                        autoMat != null && autoMat !== spec.gantry.x_beam_material;
                      update(
                        (d) => {
                          d.gantry.x_beam_profile = nextVal;
                          if (shouldPairMat && autoMat) {
                            d.gantry.x_beam_material = autoMat;
                          }
                        },
                        shouldPairMat && autoMat
                          ? `X Beam Profile: ${profileLabel(spec.gantry.x_beam_profile)} → ${profileLabel(nextVal)} (${materialLabel(autoMat)})`
                          : `X Beam Profile: ${profileLabel(spec.gantry.x_beam_profile)} → ${profileLabel(nextVal)}`
                      );
                    }}
                  >
                    {(() => {
                      const metalBeams = catalog.profiles.filter((p) =>
                        X_BEAM_METAL_PROFILE_IDS.has(p.id)
                      );
                      const lightTubes = catalog.profiles.filter((p) =>
                        X_BEAM_LIGHTWEIGHT_TUBE_IDS.has(p.id)
                      );
                      const hasCurrent =
                        metalBeams.some((p) => p.id === spec.gantry.x_beam_profile) ||
                        lightTubes.some((p) => p.id === spec.gantry.x_beam_profile);
                      return (
                        <>
                          {!hasCurrent && (
                            <option value={spec.gantry.x_beam_profile}>
                              {profileLabel(spec.gantry.x_beam_profile)} (Custom)
                            </option>
                          )}
                          <optgroup label="Extrusions & Aluminum Beams">
                            {metalBeams.map((p) => (
                              <option key={p.id} value={p.id}>
                                {p.name}
                              </option>
                            ))}
                          </optgroup>
                          {lightTubes.length > 0 && (
                            <optgroup label="Carbon Fiber, Titanium & Bare Rail">
                              {lightTubes.map((p) => (
                                <option key={p.id} value={p.id}>
                                  {p.name}
                                </option>
                              ))}
                            </optgroup>
                          )}
                        </>
                      );
                    })()}
                  </select>
                </div>
                <div className="field-row">
                  <label htmlFor="dp-x-beam-material-3">X Beam Material</label>
                  <select id="dp-x-beam-material-3"
                    value={spec.gantry.x_beam_material}
                    onChange={(e) => {
                      const nextVal = e.target.value;
                      update(
                        (d) => {
                          d.gantry.x_beam_material = nextVal;
                        },
                        `X Beam Material: ${materialLabel(spec.gantry.x_beam_material)} → ${materialLabel(nextVal)}`
                      );
                    }}
                  >
                    {(() => {
                      const beamMats = catalog.materials.filter((m) =>
                        X_BEAM_MATERIAL_IDS.has(m.id)
                      );
                      const hasCurrent = beamMats.some(
                        (m) => m.id === spec.gantry.x_beam_material
                      );
                      return (
                        <>
                          {!hasCurrent && (
                            <option value={spec.gantry.x_beam_material}>
                              {materialLabel(spec.gantry.x_beam_material)} (Custom)
                            </option>
                          )}
                          {beamMats.map((m) => (
                            <option key={m.id} value={m.id}>
                              {m.name}
                            </option>
                          ))}
                        </>
                      );
                    })()}
                  </select>
                </div>
              </div>

              {spec.architecture !== "delta" && isHollowTubeXBeam && (
                <div className="field-row stack">
                  <label htmlFor="dp-cf-rail-mounting">Tube Rail Mounting</label>
                  <select
                    id="dp-cf-rail-mounting"
                    value={currentCfRailMounting}
                    onChange={(e) => {
                      const nextVal = e.target.value as CFRailMountingType;
                      const prevLbl =
                        CF_RAIL_MOUNTING_OPTIONS.find((o) => o.id === currentCfRailMounting)
                          ?.shortLabel ?? currentCfRailMounting;
                      const nextLbl =
                        CF_RAIL_MOUNTING_OPTIONS.find((o) => o.id === nextVal)?.shortLabel ??
                        nextVal;
                      update(
                        (d) => {
                          d.gantry.cf_rail_mounting = nextVal;
                        },
                        `Tube Rail Mounting: ${prevLbl} → ${nextLbl}`
                      );
                    }}
                  >
                    {CF_RAIL_MOUNTING_OPTIONS.map((opt) => (
                      <option key={opt.id} value={opt.id}>
                        {opt.label}
                      </option>
                    ))}
                  </select>
                </div>
              )}

              {showHints && (
                <div className="hint-line">Wider/stiffer belts reduce stretch and sharpen corners</div>
              )}

              <div className="field-grid-2">
                <div className="field-row">
                  <label htmlFor="dp-belt-type-4">Belt Type</label>
                  <select id="dp-belt-type-4"
                    value={spec.gantry.belt.type}
                    onChange={(e) => {
                      const nextVal = e.target.value as PrinterSpecData["gantry"]["belt"]["type"];
                      const defaults = getBeltDefaults(nextVal);
                      const hadOv = Boolean(
                        spec.model_overrides?.belt &&
                          Object.values(spec.model_overrides.belt).some((v) => v != null)
                      );
                      update(
                        (d) => {
                          d.gantry.belt.type = nextVal;
                          d.gantry.belt.pitch_mm = defaults.pitch_mm;
                          d.gantry.belt.core = defaults.core;
                          clearModelOverrideGroup(d, "belt");
                        },
                        `Belt: ${BELT_LABELS[spec.gantry.belt.type] ?? spec.gantry.belt.type} → ${BELT_LABELS[nextVal]}${hadOv ? " (override reset)" : ""}`
                      );
                    }}
                  >
                    <optgroup label="2GT — Proven on 3D Printers (2 mm pitch)">
                      <option value="gt2_6mm">{BELT_LABELS.gt2_6mm}</option>
                      <option value="gt2_9mm">{BELT_LABELS.gt2_9mm}</option>
                      <option value="gt2_10mm">{BELT_LABELS.gt2_10mm}</option>
                      <option value="gt3_10mm">{BELT_LABELS.gt3_10mm}</option>
                      <option value="gt2_12mm">{BELT_LABELS.gt2_12mm}</option>
                    </optgroup>
                    <optgroup label="2GT — Experimental Custom Hardware">
                      <option value="gt2_15mm">{BELT_LABELS.gt2_15mm} (Exp.)</option>
                      <option value="gt2_dual_9mm">{BELT_LABELS.gt2_dual_9mm} (Exp.)</option>
                    </optgroup>
                    <optgroup label="Gates 3GT — Theoretical / Unproven (3 mm pitch)">
                      <option value="3gt_9mm">{BELT_LABELS["3gt_9mm"]} [Theoretical]</option>
                      <option value="3gt_12mm">{BELT_LABELS["3gt_12mm"]} [Theoretical]</option>
                      <option value="3gt_15mm">{BELT_LABELS["3gt_15mm"]} [Theoretical]</option>
                    </optgroup>
                    <optgroup label="Industrial Steel / TPU — Theoretical / Unproven (3–5 mm)">
                      <option value="at3_10mm">{BELT_LABELS.at3_10mm} [Theoretical]</option>
                      <option value="at3_16mm">{BELT_LABELS.at3_16mm} [Theoretical]</option>
                      <option value="at3_gen3_16mm">{BELT_LABELS.at3_gen3_16mm} [Theoretical]</option>
                      <option value="htd3m_15mm">{BELT_LABELS.htd3m_15mm} [Theoretical]</option>
                      <option value="at5_16mm">{BELT_LABELS.at5_16mm} [Theoretical]</option>
                    </optgroup>
                  </select>
                </div>
                <div className="field-row">
                  <label htmlFor="dp-belt-core-5">Belt Core</label>
                  <select id="dp-belt-core-5"
                    value={spec.gantry.belt.core}
                    onChange={(e) => {
                      const nextVal = e.target.value as PrinterSpecData["gantry"]["belt"]["core"];
                      const hadOv = Boolean(
                        spec.model_overrides?.belt &&
                          Object.values(spec.model_overrides.belt).some((v) => v != null)
                      );
                      update(
                        (d) => {
                          d.gantry.belt.core = nextVal;
                          clearModelOverrideGroup(d, "belt");
                        },
                        `Belt Core: ${spec.gantry.belt.core} → ${nextVal}${hadOv ? " (override reset)" : ""}`
                      );
                    }}
                  >
                    <option value="fiberglass">Fiberglass</option>
                    <option value="aramid">Aramid</option>
                    <option value="steel">Steel</option>
                  </select>
                </div>
              </div>

              {(() => {
                const beltMeta = catalog?.belts?.find((b) => b.id === spec.gantry.belt.type);
                return (
                  <ModelOverrideControl
                    visible={advancedMode}
                    idPrefix="ov-belt"
                    title="Belt & Core"
                    group="belt"
                    spec={spec}
                    sim={sim}
                    update={update}
                    fields={[
                      {
                        key: "stiffness_ea_n",
                        label: "Tensile EA",
                        unit: "N",
                        min: 5000,
                        max: 2000000,
                        step: 1000,
                        decimals: 0,
                        fallbackBaseline: beltMeta?.ea_default_core_n ?? 65000,
                        tooltip: "Axial tensile stiffness product E·A of the timing belt cord",
                      },
                      {
                        key: "tooth_shear_k_n_per_m_mm",
                        label: "Tooth Shear k",
                        unit: "N/(m·mm)",
                        min: 100,
                        max: 100000,
                        step: 100,
                        decimals: 0,
                        fallbackBaseline: beltMeta?.tooth_shear_k_n_per_m_mm ?? 3600,
                        tooltip: "Tooth-engagement shear stiffness per mm of belt width",
                      },
                      {
                        key: "linear_mass_g_per_m",
                        label: "Linear Mass",
                        unit: "g/m",
                        min: 1,
                        max: 300,
                        step: 0.5,
                        decimals: 1,
                        fallbackBaseline: beltMeta
                          ? Number((beltMeta.linear_mass_kg_m * 1000).toFixed(1))
                          : 11,
                        tooltip: "Belt mass per meter of length (affects pluck frequency & moving mass)",
                      },
                      {
                        key: "max_working_tension_n",
                        label: "Max Tension",
                        unit: "N",
                        min: 10,
                        max: 1000,
                        step: 5,
                        decimals: 0,
                        fallbackBaseline: beltMeta?.max_working_tension_n ?? 55,
                        tooltip: "Rated maximum dynamic working tension before cord fatigue/overtension",
                      },
                    ]}
                  />
                );
              })()}

              {(() => {
                const beltMeta = catalog?.belts?.find((b) => b.id === spec.gantry.belt.type);
                const tier = beltMeta?.validation_tier ?? sim?.kinematics?.belt_validation_tier ?? "proven";
                const summary =
                  beltMeta?.theoretical_summary ?? sim?.kinematics?.belt_theoretical_summary ?? "";
                const issues =
                  beltMeta?.practical_issues ?? sim?.kinematics?.belt_practical_issues ?? [];
                if (tier === "proven" && issues.length === 0) return null;
                const isTheoretical = tier === "theoretical";
                const minT = beltMeta?.min_pulley_teeth ?? sim?.kinematics?.belt_min_pulley_teeth;
                const minIdler = beltMeta?.min_back_idler_mm ?? sim?.kinematics?.belt_min_back_idler_mm;
                const bodyMat = (beltMeta?.body_material ?? sim?.kinematics?.belt_body_material ?? "epdm").toUpperCase();
                return (
                  <div className={`belt-theory-callout ${isTheoretical ? "theoretical" : "experimental"}`}>
                    <div className="belt-theory-header">
                      <span className={`belt-tier-badge ${isTheoretical ? "theoretical" : "experimental"}`}>
                        {isTheoretical
                          ? "THEORETICAL — UNPROVEN ON 3D PRINTERS"
                          : "EXPERIMENTAL — CUSTOM HARDWARE"}
                      </span>
                      {minT != null && minIdler != null && (
                        <span className="belt-theory-limits">
                          ≥{minT}T · ≥{Math.round(minIdler)}mm idler · {bodyMat}
                        </span>
                      )}
                    </div>
                    {summary && <div className="belt-theory-summary">{summary}</div>}
                    {issues.length > 0 && (
                      <ul className="belt-theory-issues">
                        {issues.map((issue, idx) => (
                          <li key={idx}>{issue}</li>
                        ))}
                      </ul>
                    )}
                  </div>
                );
              })()}

              <div className="slider-row">
                <label htmlFor="dp-belt-tension-n-6">
                  <span>Belt Tension (N)</span>
                  <span className="tension-readout">
                    <strong>{tensionDraft} N</strong>
                    {beltPluckHz != null && (
                      <span className="tension-hz-pill">~{Math.round(beltPluckHz)} Hz @ 150 mm</span>
                    )}
                  </span>
                </label>
                <div className="tension-control-row">
                  <input id="dp-belt-tension-n-6"
                    type="range"
                    min={5}
                    max={beltTensionMax}
                    step={1}
                    value={Math.max(5, Math.min(beltTensionMax, tensionDraft))}
                    onChange={(e) => handleTensionSliderChange(Number(e.target.value))}
                    onPointerUp={() => commitBeltTension(tensionDraft)}
                    onKeyUp={() => commitBeltTension(tensionDraft)}
                  />
                  <input aria-label="Belt Tension (N) value"
                    type="number"
                    className="tension-num-input"
                    min={5}
                    max={beltTensionMax}
                    step={1}
                    value={tensionDraft}
                    onChange={(e) => {
                      const v = Number(e.target.value);
                      if (!Number.isFinite(v)) return;
                      setTensionDraft(v);
                      if (v >= 5 && v <= beltTensionMax) {
                        commitBeltTension(v);
                      }
                    }}
                    onBlur={() => commitBeltTension(tensionDraft)}
                  />
                </div>
                {(beltOvertension || shaftBendingRisk || showHints) && (
                  <div className={`hint-line ${beltOvertension ? "danger" : shaftBendingRisk ? "warn" : ""}`}>
                    {beltOvertension
                      ? beltPeakTensionN != null && beltMaxWorkingN != null
                        ? `Peak dynamic tension ${beltPeakTensionN.toFixed(1)} N (${Math.round(
                            spec.gantry.belt.pretension_n
                          )} N static + ${Math.max(
                            0,
                            beltPeakTensionN - spec.gantry.belt.pretension_n
                          ).toFixed(1)} N accel load) exceeds ${Math.round(
                            beltMaxWorkingN
                          )} N working rating.`
                        : `Peak dynamic belt tension exceeds ${
                            beltMaxWorkingN != null ? `${Math.round(beltMaxWorkingN)} N ` : ""
                          }working tension rating.`
                      : shaftBendingRisk
                        ? "High tension (>45 N) without double-shear bearings risks shaft bending."
                        : "High tension >45 N needs 9mm+ belts & double-shear bearings."}
                  </div>
                )}
              </div>

              <div className="field-grid-2">
                {spec.architecture === "corexy" && (
                  <div className="field-row">
                    <label htmlFor="dp-belt-mismatch-71">A/B Tension Mismatch (%)</label>
                    <input
                      id="dp-belt-mismatch-71"
                      type="number"
                      min={0}
                      max={80}
                      step={5}
                      value={spec.gantry.belt.tension_mismatch_pct ?? 0}
                      title="Tension difference between the two CoreXY belts as a percentage of their mean; couples X and Y resonances into a double peak"
                      onChange={(e) => {
                        const v = Number(e.target.value);
                        if (!Number.isFinite(v) || v < 0 || v > 80) return;
                        const prev = spec.gantry.belt.tension_mismatch_pct ?? 0;
                        if (v === prev) return;
                        update((d) => {
                          d.gantry.belt.tension_mismatch_pct = v;
                        }, `Belt A/B Tension Mismatch: ${prev}% → ${v}%`);
                      }}
                    />
                  </div>
                )}
                <div className="field-row">
                  <label htmlFor="dp-belt-set-temp-72">Tension Set At (°C)</label>
                  <input
                    id="dp-belt-set-temp-72"
                    type="number"
                    min={0}
                    max={120}
                    step={1}
                    placeholder="Chamber temp"
                    value={spec.gantry.belt.pretension_set_temp_c ?? ""}
                    title="Frame temperature when the belts were tensioned. Leave empty if the tension above was measured at chamber temperature"
                    onChange={(e) => {
                      const raw = e.target.value.trim();
                      const prev = spec.gantry.belt.pretension_set_temp_c ?? null;
                      if (raw === "") {
                        if (prev === null) return;
                        update((d) => {
                          d.gantry.belt.pretension_set_temp_c = null;
                        }, `Belt Tension Set Temperature: ${prev} °C → chamber temperature`);
                        return;
                      }
                      const v = Number(raw);
                      if (!Number.isFinite(v) || v < 0 || v > 120 || v === prev) return;
                      update((d) => {
                        d.gantry.belt.pretension_set_temp_c = v;
                      }, `Belt Tension Set Temperature: ${prev ?? "chamber"} → ${v} °C`);
                    }}
                  />
                </div>
              </div>
              {beltDyn && (beltDyn.thermal_tension_delta_n !== 0 || beltDyn.coupling_kappa !== 0) && (
                <div className="hint-line">
                  {beltDyn.thermal_tension_delta_n !== 0 &&
                    `Operating tension ${beltDyn.operating_pretension_n.toFixed(1)} N at chamber temperature (${
                      beltDyn.thermal_tension_delta_n > 0 ? "+" : ""
                    }${beltDyn.thermal_tension_delta_n.toFixed(1)} N, ~${Math.round(
                      beltDyn.pluck_hz_150mm_operating
                    )} Hz @ 150 mm). `}
                  {beltDyn.coupling_kappa !== 0 &&
                    beltDyn.cross_peak_in_x != null &&
                    beltDyn.cross_peak_in_y != null &&
                    `Belt A ${beltDyn.tension_a_n?.toFixed(1)} N / B ${beltDyn.tension_b_n?.toFixed(
                      1
                    )} N: cross-axis peak ${Math.round(100 * beltDyn.cross_peak_in_x)}% on X, ${Math.round(
                      100 * beltDyn.cross_peak_in_y
                    )}% on Y.`}
                </div>
              )}

              <div className="sub-block">
                <div className="sub-block-title">
                  <span>XY Stepper Motor &amp; Driver Power</span>
                  <span
                    className={`sub-block-tag ${
                      motorOvercurrentDraft ? "danger" : currentPctOfRated > 88 ? "warn" : ""
                    }`}
                  >
                    {runCurrentDraft.toFixed(1)}A @ {xyStepper.supply_voltage_v}V
                  </span>
                </div>

                {showHints && (
                  <div className="hint-line">
                    Higher Amps strengthen magnetic holding stiffness (raises input-shaper Hz); higher
                    Voltage &amp; lower mH keep torque &amp; stiffness strong at high speed.
                  </div>
                )}

                <div className="field-row">
                  <label htmlFor="dp-motor-model-7">Motor Model</label>
                  <select id="dp-motor-model-7"
                    value={selectedStepperValue}
                    onChange={(e) => {
                      const nextId = e.target.value;
                      const chosen = stepperCatalog.find((s) => s.preset_id === nextId);
                      if (!chosen) return;
                      if (currentTimerRef.current != null) {
                        window.clearTimeout(currentTimerRef.current);
                        currentTimerRef.current = null;
                      }
                      const nextRunA = Number(chosen.run_current_a.toFixed(1));
                      setRunCurrentDraft(nextRunA);
                      const prevName = shortMotorName(xyStepper.name);
                      const nextName = shortMotorName(chosen.name);
                      update(
                        (d) => {
                          d.kinematics.xy_stepper.preset_id = chosen.preset_id;
                          d.kinematics.xy_stepper.name = chosen.name;
                          d.kinematics.xy_stepper.holding_torque_nm = chosen.holding_torque_nm;
                          d.kinematics.xy_stepper.rated_current_a = chosen.rated_current_a;
                          d.kinematics.xy_stepper.run_current_a = nextRunA;
                          d.kinematics.xy_stepper.phase_resistance_ohm = chosen.phase_resistance_ohm;
                          d.kinematics.xy_stepper.inductance_mh = chosen.inductance_mh;
                          d.kinematics.xy_stepper.rotor_inertia_kg_m2 = chosen.rotor_inertia_kg_m2;
                          d.kinematics.xy_stepper.motor_mass_g = chosen.motor_mass_g;
                          d.kinematics.xy_stepper.steps_per_rev = chosen.steps_per_rev;
                          d.kinematics.xy_stepper.nema_size = chosen.nema_size ?? 17;
                          d.kinematics.xy_stepper.body_length_mm = chosen.body_length_mm ?? 48;
                          d.kinematics.xy_stepper.insulation_class = chosen.insulation_class ?? "B";
                          d.kinematics.xy_stepper.has_encoder = Boolean(chosen.has_encoder);
                        },
                        `XY Motor: ${prevName} → ${nextName}`
                      );
                    }}
                  >
                    {selectedStepperValue === "__custom__" && (
                      <option value="__custom__">{xyStepper.name} (Custom)</option>
                    )}
                    {stepperCatalog.map((m) => (
                      <option key={m.preset_id} value={m.preset_id}>
                        {m.name}
                      </option>
                    ))}
                  </select>
                  <div className="motor-spec-strip">
                    <span>{xyStepper.holding_torque_nm.toFixed(2)} N·m</span>
                    <span>•</span>
                    <span>{xyStepper.inductance_mh.toFixed(2).replace(/0$/, "")} mH</span>
                    <span>•</span>
                    <span>{phaseResistanceOhm.toFixed(2).replace(/0$/, "")} Ω</span>
                    <span>•</span>
                    <span>{ratedCurrentA.toFixed(1)}A</span>
                    <span>•</span>
                    <span>{Math.round((xyStepper.rotor_inertia_kg_m2 ?? 8.2e-6) * 1e7)} g·cm²</span>
                    <span>•</span>
                    <span>{xyStepper.steps_per_rev === 400 ? "0.9°" : "1.8°"}</span>
                    <span>•</span>
                    <span>Class {xyStepper.insulation_class ?? "B"}</span>
                    {xyStepper.has_encoder && (
                      <>
                        <span>•</span>
                        <span>1000 PPR</span>
                      </>
                    )}
                  </div>
                </div>

                <div className="slider-row">
                  <label htmlFor="dp-run-current-amps-max-a-8">
                    <span>
                      Run Current (Amps){" "}
                      <span style={{ color: "var(--muted)", fontWeight: 500 }}>
                        max {ratedCurrentA.toFixed(1)}A
                      </span>
                    </span>
                    <span className="tension-readout">
                      <span
                        className={`tension-hz-pill ${
                          motorOvercurrentDraft ? "danger" : currentPctOfRated > 88 ? "warn" : ""
                        }`}
                      >
                        {currentPctOfRated}% rated • ~{estMotorHeatW.toFixed(1)} W/ea
                      </span>
                    </span>
                  </label>
                  <div className="tension-control-row">
                    <input id="dp-run-current-amps-max-a-8"
                      type="range"
                      min={0.4}
                      max={runCurrentMaxA}
                      step={0.1}
                      value={Math.max(0.4, Math.min(runCurrentMaxA, runCurrentDraft))}
                      onChange={(e) => handleRunCurrentSliderChange(Number(e.target.value))}
                      onPointerUp={() => commitRunCurrent(runCurrentDraft)}
                      onKeyUp={() => commitRunCurrent(runCurrentDraft)}
                    />
                    <input aria-label="Run Current (Amps) value"
                      type="number"
                      className="tension-num-input"
                      min={0.4}
                      max={runCurrentMaxA}
                      step={0.1}
                      value={runCurrentDraft}
                      onChange={(e) => {
                        const v = Number(e.target.value);
                        if (!Number.isFinite(v)) return;
                        setRunCurrentDraft(v);
                        if (v >= 0.4 && v <= runCurrentMaxA) {
                          commitRunCurrent(v);
                        }
                      }}
                      onBlur={() => commitRunCurrent(runCurrentDraft)}
                    />
                  </div>
                  {motorOvercurrentDraft && (
                    <div className="hint-line danger">
                      Exceeds {ratedCurrentA.toFixed(1)}A motor rating — overheating &amp; torque saturation risk!
                    </div>
                  )}
                </div>

                <div className="field-row">
                  <label id="dp-bus-voltage-v-9">
                    <span>Bus Voltage (V)</span>
                    <span style={{ color: "var(--text)", fontWeight: 500 }}>
                      {xyStepper.supply_voltage_v}V DC
                    </span>
                  </label>
                  <div role="group" aria-labelledby="dp-bus-voltage-v-9" className="voltage-selector-row">
                    <div className="voltage-preset-group">
                      {BUS_VOLTAGE_PRESETS.map((v) => (
                        <button
                          key={v}
                          type="button"
                          className={`voltage-pill-btn ${
                            Math.abs(xyStepper.supply_voltage_v - v) < 0.5 ? "active" : ""
                          }`}
                          onClick={() => {
                            if (Math.abs(xyStepper.supply_voltage_v - v) < 0.5) return;
                            update(
                              (d) => {
                                d.kinematics.xy_stepper.supply_voltage_v = v;
                              },
                              `Stepper Voltage: ${xyStepper.supply_voltage_v}V → ${v}V`
                            );
                          }}
                        >
                          {v}V
                        </button>
                      ))}
                    </div>
                    <input aria-label="Bus Voltage (V) value"
                      type="number"
                      className="tension-num-input"
                      min={12}
                      max={96}
                      step={1}
                      value={xyStepper.supply_voltage_v}
                      onChange={(e) =>
                        updatePositiveNum(
                          e.target.value,
                          (d, v) => {
                            d.kinematics.xy_stepper.supply_voltage_v = v;
                          },
                          (v) => `Stepper Voltage: ${xyStepper.supply_voltage_v}V → ${v}V`
                        )
                      }
                      title="Custom stepper bus voltage (V)"
                    />
                  </div>
                </div>
              </div>

              <div className="field-grid-2">
                <div className="field-row">
                  <label htmlFor="dp-max-vel-mm-s-10">Max Vel (mm/s)</label>
                  <input id="dp-max-vel-mm-s-10"
                    type="number"
                    value={spec.kinematics.max_velocity_mm_s}
                    onChange={(e) =>
                      updatePositiveNum(
                        e.target.value,
                        (d, v) => {
                          d.kinematics.max_velocity_mm_s = v;
                        },
                        (v) => `Max Vel: ${spec.kinematics.max_velocity_mm_s} → ${v} mm/s`
                      )
                    }
                  />
                </div>
                <div className="field-row">
                  <label htmlFor="dp-max-accel-mm-s-11">Max Accel (mm/s²)</label>
                  <input id="dp-max-accel-mm-s-11"
                    type="number"
                    step={500}
                    value={spec.kinematics.max_accel_mm_s2}
                    onChange={(e) =>
                      updatePositiveNum(
                        e.target.value,
                        (d, v) => {
                          d.kinematics.max_accel_mm_s2 = v;
                        },
                        (v) =>
                          `Max Accel: ${spec.kinematics.max_accel_mm_s2.toLocaleString()} → ${v.toLocaleString()} mm/s²`
                      )
                    }
                  />
                </div>
              </div>

              <div className="field-row">
                <label htmlFor="dp-input-shaper-12">Input Shaper</label>
                <select id="dp-input-shaper-12"
                  value={spec.kinematics.input_shaper}
                  onChange={(e) => {
                    const nextVal = e.target.value as PrinterSpecData["kinematics"]["input_shaper"];
                    update(
                      (d) => {
                        d.kinematics.input_shaper = nextVal;
                      },
                      `Input Shaper: ${SHAPER_LABELS[spec.kinematics.input_shaper] ?? spec.kinematics.input_shaper} → ${SHAPER_LABELS[nextVal] ?? nextVal}`
                    );
                  }}
                >
                  <option value="auto">Auto (Klipper)</option>
                  <option value="none">None</option>
                  <option value="zv">ZV</option>
                  <option value="mzv">MZV</option>
                  <option value="ei">EI</option>
                  <option value="2hump_ei">2HUMP_EI</option>
                  <option value="3hump_ei">3HUMP_EI</option>
                  <option value="zvdd">ZVDD</option>
                  <option value="smooth_zv">SMOOTH_ZV (Kalico)</option>
                  <option value="smooth_ei">SMOOTH_EI (Kalico)</option>
                  <option value="smooth_flatness">SMOOTH_FLATNESS (Kalico C4 Beta)</option>
                </select>
                {spec.kinematics.input_shaper === "smooth_flatness" && (
                  <div className="beta-guide-box">
                    <div className="beta-guide-head">
                      <span>SMOOTH_FLATNESS Guide</span>
                      <span className="beta-pill">Kalico C⁴</span>
                    </div>
                    <p className="beta-guide-line">
                      <strong>What it is:</strong> C⁴ septic differential-flatness smoother (
                      <code>t_sm = 0.70 / f_n</code>) that pre-leads inertial belt stretch to cancel
                      the primary resonance with ~35% less corner rounding than <code>MZV</code>.
                    </p>
                    <p className="beta-guide-line">
                      <strong>How to set it:</strong> Use your <strong>locked-rotor</strong> belt
                      frequency (<em>f</em><sub>b</sub>), ensure a clean single-peak spectrum and{" "}
                      <code>≥ 1.25×</code> motor torque headroom (handles <code>1.17×</code> peak
                      accel spikes on tight zig-zags), and enable{" "}
                      <strong>LPV 3×3 Bed Scheduling</strong> below if corner resonance shifts &gt;
                      5% (mandatory on 2WD CoreXY).
                    </p>
                  </div>
                )}
              </div>

              <details className="sub-accordion">
                <summary>
                  <span>Gantry Hardware &amp; Brackets</span>
                  <span className="sub-accordion-tag">
                    {customBracketOverride ? "Custom CAD" : selectedBracket.short_name} ·{" "}
                    {dressedMovingXYJointsG}g
                  </span>
                </summary>
                <div className="sub-accordion-body">
                  {spec.architecture === "corexy" && (
                    <div className="field-row">
                      <label id="dp-pulley-mounts-13">Pulley Mounts</label>
                      <div role="group" aria-labelledby="dp-pulley-mounts-13" className="engine-toggle">
                        <button
                          type="button"
                          className={`engine-toggle-btn ${!doubleShearPulleys ? "active" : ""}`}
 aria-pressed={!doubleShearPulleys}
                          onClick={() => {
                            if (!doubleShearPulleys) return;
                            update(
                              (d) => {
                                d.gantry.double_shear_pulleys = false;
                              },
                              "Pulley Mounts: Double-Shear → Single-Shear"
                            );
                          }}
                        >
                          Single-Shear
                          <span className="engine-toggle-sub">Standard</span>
                        </button>
                        <button
                          type="button"
                          className={`engine-toggle-btn ${doubleShearPulleys ? "active" : ""}`}
 aria-pressed={doubleShearPulleys}
                          onClick={() => {
                            if (doubleShearPulleys) return;
                            update(
                              (d) => {
                                d.gantry.double_shear_pulleys = true;
                              },
                              "Pulley Mounts: Single-Shear → Double-Shear"
                            );
                          }}
                        >
                          Double-Shear
                          <span className="engine-toggle-sub">Bearings</span>
                        </button>
                      </div>
                    </div>
                  )}

                  <div className="field-row">
                    <label htmlFor="dp-gantry-brackets-preset">
                      <span>Gantry Brackets &amp; Motor Mounts</span>
                      <span
                        className={`sub-block-tag ${
                          customBracketOverride
                            ? "ok"
                            : bracketCreepRisk
                              ? "danger"
                              : selectedBracket.cnc_machined
                                ? "ok"
                                : ""
                        }`}
                      >
                        <span
                          className="swatch-dot"
                          style={{
                            background: customBracketOverride ? "#22d3ee" : selectedBracket.color_hex,
                            marginRight: 4,
                          }}
                        />
                        {customBracketOverride ? "Custom CAD Override" : selectedBracket.short_name}
                      </span>
                    </label>
                    {customBracketOverride && (
                      <div className="custom-bracket-pill" style={{ marginBottom: 4 }}>
                        <div className="custom-bracket-pill-text">
                          <span className="custom-bracket-dot" />
                          <span>
                            Custom CAD Bracket Active:{" "}
                            <strong>{customBracketOverride.label}</strong> (
                            {Math.round(customBracketOverride.stiffness_n_per_mm)} N/mm)
                          </span>
                        </div>
                        <button
                          type="button"
                          className="custom-bracket-clear-btn"
                          onClick={() =>
                            update(
                              (d) => {
                                d.gantry.custom_bracket_override = null;
                              },
                              `Custom CAD Bracket: ${customBracketOverride.label} → Cleared`
                            )
                          }
                        >
                          Clear
                        </button>
                      </div>
                    )}
                    <select id="dp-gantry-brackets-preset"
                      value={currentBracketId}
                      onChange={(e) => {
                        const nextId = e.target.value as BracketMaterialType;
                        const nextItem =
                          gantryBrackets.find((b) => b.id === nextId) ?? gantryBrackets[0];
                        const hadOv = Boolean(
                          spec.model_overrides?.gantry_bracket &&
                            Object.values(spec.model_overrides.gantry_bracket).some((v) => v != null)
                        );
                        update(
                          (d) => {
                            d.gantry.bracket_material = nextId;
                            clearModelOverrideGroup(d, "gantry_bracket");
                          },
                          `Gantry Brackets: ${selectedBracket.short_name} → ${nextItem.short_name}${hadOv ? " (override reset)" : ""}`
                        );
                      }}
                    >
                      {gantryBrackets.map((b) => (
                        <option key={b.id} value={b.id}>
                          {b.name}
                        </option>
                      ))}
                    </select>
                    {showHints && (
                      <div className="hint-line">
                        Motor mounts &amp; XY joints flex in series with belts. Printed ABS creeps
                        above ~45 N in heated chambers; annealed PPA-CF/PPS-CF, SLM printed metals
                        (lighter hollow geometry, 0.92–0.94× rough-face contact, lower CTE), or CNC
                        billet aluminum prevent creep &amp; sharpen resonance.
                      </div>
                    )}
                    <div className="bracket-readout-row">
                      <div className="inline-readout-strip">
                        {simBracketStiffness != null && (
                          <>
                            <strong>{Math.round(simBracketStiffness)} N/mm</strong>
                            <span>·</span>
                          </>
                        )}
                        <span>{Math.round(selectedBracket.hdt_c)}°C HDT</span>
                        <span>·</span>
                        <span>{Math.round(selectedBracket.creep_tension_limit_n)} N limit</span>
                      </div>
                      <div className="bracket-pill-row">
                        {bracketCreepRisk ? (
                          <span className="badge-pill bad">Creep Risk</span>
                        ) : isSelectedBracketSlm ? (
                          <span className="badge-pill ok">SLM Metal</span>
                        ) : isSelectedBracketSheetMetal ? (
                          <span className="badge-pill ok">Sheet Metal</span>
                        ) : selectedBracket.cnc_machined ? (
                          <span className="badge-pill ok">CNC Rigid</span>
                        ) : (
                          <span className="badge-pill ok">No Creep</span>
                        )}
                        <span className="bracket-mass-note">
                          {selectedBracket.carriage_mass_delta_g > 0
                            ? `+${Math.round(selectedBracket.carriage_mass_delta_g)}g moving`
                            : "Base Y-carriage mass"}
                        </span>
                      </div>
                    </div>
                    <ModelOverrideControl
                      visible={advancedMode}
                      idPrefix="ov-gantry-bracket"
                      title="Gantry Brackets"
                      group="gantry_bracket"
                      spec={spec}
                      sim={sim}
                      update={update}
                      fields={[
                        {
                          key: "stiffness_n_per_mm",
                          label: "Stiffness",
                          unit: "N/mm",
                          min: 10,
                          max: 50000,
                          step: 25,
                          decimals: 0,
                          fallbackBaseline:
                            simBracketStiffness ?? (selectedBracket.cnc_machined ? 1800 : 280),
                          tooltip: "Lumped in-series stiffness of motor mounts & XY-joint brackets",
                        },
                        {
                          key: "carriage_mass_delta_g",
                          label: "Mass Delta",
                          unit: "g",
                          min: -100,
                          max: 500,
                          step: 2,
                          decimals: 0,
                          fallbackBaseline: selectedBracket.carriage_mass_delta_g,
                          tooltip: "Added moving Y-carriage mass relative to baseline printed ABS",
                        },
                        {
                          key: "creep_tension_limit_n",
                          label: "Creep Limit",
                          unit: "N",
                          min: 10,
                          max: 1000,
                          step: 5,
                          decimals: 0,
                          fallbackBaseline: selectedBracket.creep_tension_limit_n,
                          tooltip: "Maximum sustained belt tension before polymer creep risk",
                        },
                        {
                          key: "damping_delta",
                          label: "Damping Delta",
                          unit: "ζ",
                          min: -0.02,
                          max: 0.1,
                          step: 0.001,
                          decimals: 3,
                          fallbackBaseline: isSelectedBracketSlm
                            ? -0.0015
                            : selectedBracket.cnc_machined
                              ? -0.003
                              : 0.004,
                          tooltip: "Modal damping contribution from bracket material hysteresis",
                        },
                      ]}
                    />
                  </div>

                  <div className="field-row">
                    <label htmlFor="dp-y-gantry-weight-class-preset-14">
                      <span>Y-Gantry Weight Class Preset</span>
                      <span className="sub-block-tag">
                        {matchedWeightClass ? matchedWeightClass.short_name : "Custom"}
                      </span>
                    </label>
                    <select id="dp-y-gantry-weight-class-preset-14"
                      value={matchedWeightClass ? matchedWeightClass.id : "__custom__"}
                      onChange={(e) => {
                        const nextId = e.target.value;
                        if (nextId === "__custom__") return;
                        const cls = gantryWeightClasses.find((c) => c.id === nextId);
                        if (!cls) return;
                        if (carriageMassTimerRef.current != null) {
                          window.clearTimeout(carriageMassTimerRef.current);
                          carriageMassTimerRef.current = null;
                        }
                        setCarriageMassDraft(Math.round(cls.carriage_mass_g));
                        const nextBracketValid = gantryBrackets.some(
                          (b) => b.id === cls.suggested_bracket_material
                        );
                        update(
                          (d) => {
                            d.gantry.carriage_mass_g = cls.carriage_mass_g;
                            if (nextBracketValid) {
                              if (d.gantry.bracket_material !== cls.suggested_bracket_material) {
                                clearModelOverrideGroup(d, "gantry_bracket");
                              }
                              d.gantry.bracket_material =
                                cls.suggested_bracket_material as BracketMaterialType;
                            }
                          },
                          `Y-Gantry Weight Class: ${matchedWeightClass ? matchedWeightClass.short_name : `Custom (${Math.round(spec.gantry.carriage_mass_g)}g)`} → ${cls.short_name} (${Math.round(cls.carriage_mass_g)}g)`
                        );
                      }}
                    >
                      {gantryWeightClasses.map((cls) => (
                        <option key={cls.id} value={cls.id}>
                          {cls.name}
                        </option>
                      ))}
                      {!matchedWeightClass && (
                        <option value="__custom__">
                          Custom Weight ({Math.round(spec.gantry.carriage_mass_g)}g)
                        </option>
                      )}
                    </select>
                    <PresetAffectsStrip
                      visible={advancedMode}
                      items={[
                        {
                          label: "Base Carriage Mass",
                          value: `${Math.round(spec.gantry.carriage_mass_g)} g`,
                        },
                        {
                          label: "Bracket Material",
                          value: selectedBracket.short_name,
                        },
                      ]}
                      suffix="(editable above & below)"
                    />
                    {showHints && (
                      <div className="hint-line">
                        {matchedWeightClass?.notes ??
                          "Combined mass of XY-joint blocks, idler hardware & Y-rail carriages moving on Y."}
                      </div>
                    )}
                  </div>

                  <div className="slider-row">
                    <label htmlFor="dp-xy-joints-and-y-carriages-15">
                      <span>XY-Joints &amp; Y-Carriages Mass (g)</span>
                      <span className="tension-readout">
                        <strong>{Math.round(carriageMassDraft)} g</strong>
                      </span>
                    </label>
                    <div className="tension-control-row">
                      <input id="dp-xy-joints-and-y-carriages-15"
                        type="range"
                        min={20}
                        max={600}
                        step={5}
                        value={Math.max(20, Math.min(600, Math.round(carriageMassDraft)))}
                        onChange={(e) => handleCarriageMassSliderChange(Number(e.target.value))}
                        onPointerUp={() => commitCarriageMass(carriageMassDraft)}
                        onKeyUp={() => commitCarriageMass(carriageMassDraft)}
                      />
                      <input aria-label="XY-Joints & Y-Carriages Mass (g) value"
                        type="number"
                        className="tension-num-input"
                        min={20}
                        max={600}
                        step={5}
                        value={carriageMassDraft}
                        onChange={(e) => {
                          const v = Number(e.target.value);
                          if (!Number.isFinite(v)) return;
                          setCarriageMassDraft(v);
                          if (v >= 20 && v <= 600) {
                            commitCarriageMass(v);
                          }
                        }}
                        onBlur={() => commitCarriageMass(carriageMassDraft)}
                      />
                    </div>
                    <div
                      className="dressed-mass-note"
                      title="Moving XY-joint & Y-carriage total sent to FEA (bracket material mass delta is added automatically — no need to double-count it in the slider)"
                    >
                      {baseCarriageMassG}g base {bracketDeltaG >= 0 ? "+" : "−"}{" "}
                      {Math.abs(bracketDeltaG)}g {bracketDeltaShortLabel} ={" "}
                      <strong>{dressedMovingXYJointsG}g</strong> dressed moving XY-joints
                    </div>
                  </div>

                  <div className="field-row">
                    <label htmlFor="dp-stepper-thermal-gasket">Stepper Thermal Gasket</label>
                    <select
                      id="dp-stepper-thermal-gasket"
                      value={spec.gantry.stepper_thermal_gasket ?? "none"}
                      title="2mm PTFE thermal barrier between NEMA 17/23 motor face and printed/metal bracket cuts conducted mount heat by ~14 °C"
                      onChange={(e) => {
                        const nextVal = e.target.value as StepperThermalGasketType;
                        const prevLbl =
                          (spec.gantry.stepper_thermal_gasket ?? "none") === "ptfe_2mm"
                            ? "2mm PTFE"
                            : "None";
                        const nextLbl = nextVal === "ptfe_2mm" ? "2mm PTFE" : "None";
                        update(
                          (d) => {
                            d.gantry.stepper_thermal_gasket = nextVal;
                          },
                          `Stepper Thermal Gasket: ${prevLbl} → ${nextLbl}`
                        );
                      }}
                    >
                      <option value="none">None (Direct Mount)</option>
                      <option value="ptfe_2mm">2mm PTFE (-14 °C Mount)</option>
                    </select>
                    {showHints && (
                      <div className="hint-line">
                        2 mm PTFE motor gaskets insulate printed mounts from stepper face heat (−14 °C)
                        to preserve polymer modulus and prevent belt-tension creep.
                      </div>
                    )}
                  </div>

                  {isFlyingGantry && (
                    <div className="field-grid-2">
                      <div className="field-row">
                        <label htmlFor="dp-z-joint-type">V2 Z-Joints</label>
                        <select
                          id="dp-z-joint-type"
                          value={spec.gantry.z_joint_type ?? "single"}
                          title="Voron 2.4 flying-gantry corner Z-joint topology: single stock vs Monolith double-shear twin-arm"
                          onChange={(e) => {
                            const nextVal = e.target.value as ZJointType;
                            const prevLbl =
                              (spec.gantry.z_joint_type ?? "single") === "double"
                                ? "Double"
                                : "Single";
                            const nextLbl = nextVal === "double" ? "Double" : "Single";
                            update(
                              (d) => {
                                d.gantry.z_joint_type = nextVal;
                              },
                              `V2 Z-Joints: ${prevLbl} → ${nextLbl}`
                            );
                          }}
                        >
                          <option value="single">Single</option>
                          <option value="double">Double</option>
                        </select>
                      </div>
                      <div className="field-row">
                        <label htmlFor="dp-z-joint-material">Z-Joint Material</label>
                        <select
                          id="dp-z-joint-material"
                          value={spec.gantry.z_joint_material ?? "printed"}
                          title="Voron 2.4 flying-gantry corner Z-joint material stiffness and thermal stability"
                          onChange={(e) => {
                            const nextVal = e.target.value as ZJointMaterialType;
                            const matLabels: Record<ZJointMaterialType, string> = {
                              printed: "Printed",
                              hybrid: "Hybrid",
                              slm: "SLM Al",
                              milled: "Milled 6061",
                            };
                            const prevKey = spec.gantry.z_joint_material ?? "printed";
                            update(
                              (d) => {
                                d.gantry.z_joint_material = nextVal;
                              },
                              `Z-Joint Material: ${matLabels[prevKey]} → ${matLabels[nextVal]}`
                            );
                          }}
                        >
                          <option value="printed">Printed</option>
                          <option value="hybrid">Hybrid</option>
                          <option value="slm">SLM Al</option>
                          <option value="milled">Milled 6061</option>
                        </select>
                      </div>
                    </div>
                  )}

                  {simMassSummary && (
                    <div
                      style={{
                        display: "flex",
                        flexWrap: "wrap",
                        alignItems: "center",
                        gap: "4px 6px",
                        padding: "6px 8px",
                        borderRadius: 6,
                        background: "rgba(13, 19, 29, 0.78)",
                        border: "1px solid rgba(56, 189, 248, 0.22)",
                        fontSize: 11,
                        lineHeight: 1.4,
                        color: "var(--muted)",
                        fontVariantNumeric: "tabular-nums",
                      }}
                    >
                      {simMassSummary.x_beam_mass_g != null && (
                        <>
                          <span>
                            X-Beam:{" "}
                            <strong style={{ color: "var(--text)" }}>
                              {Math.round(simMassSummary.x_beam_mass_g)}g
                            </strong>
                          </span>
                          <span style={{ opacity: 0.45 }}>·</span>
                        </>
                      )}
                      {simMassSummary.x_rail_mass_g != null && (
                        <>
                          <span>
                            X-Rail:{" "}
                            <strong style={{ color: "var(--text)" }}>
                              {Math.round(simMassSummary.x_rail_mass_g)}g
                            </strong>
                          </span>
                          <span style={{ opacity: 0.45 }}>·</span>
                        </>
                      )}
                      {(simMassSummary.x_backer_mass_g ?? 0) > 0 && (
                        <>
                          <span>
                            Backer:{" "}
                            <strong style={{ color: "var(--text)" }}>
                              {Math.round(simMassSummary.x_backer_mass_g!)}g
                            </strong>
                          </span>
                          <span style={{ opacity: 0.45 }}>·</span>
                        </>
                      )}
                      <span>
                        XY-Joints:{" "}
                        <strong style={{ color: "var(--text)" }}>
                          {Math.round(
                            simMassSummary.y_carriage_joints_mass_g ?? spec.gantry.carriage_mass_g
                          )}
                          g
                        </strong>
                      </span>
                      <span style={{ opacity: 0.45 }}>·</span>
                      <span>
                        Toolhead:{" "}
                        <strong style={{ color: "var(--text)" }}>
                          {Math.round(simMassSummary.x_moving_mass_g)}g
                        </strong>
                      </span>
                      <span style={{ color: "var(--accent)", fontWeight: 600, marginLeft: 2 }}>
                        → Total Y: {Math.round(simMassSummary.y_moving_mass_g)}g
                      </span>
                    </div>
                  )}
                </div>
              </details>

              <details className="sub-accordion">
                <summary>
                  <span>Linear Rails &amp; Mounting</span>
                  <span className="sub-accordion-tag">
                    X: {selectedXRailMount.short_name}
                    {spec.architecture === "corexy" ? ` · Y: ${selectedYRailMount.short_name}` : ""}
                  </span>
                </summary>
                <div className="sub-accordion-body">
                  {showHints && (
                    <div className="hint-line">
                      Front = max Y stiffness; Top = max Z stiffness; Bottom = shorter nozzle drop
                      &amp; less pitch twist; Inner Y = max X stiffness.
                    </div>
                  )}

                  <div className={spec.architecture === "corexy" ? "field-grid-2" : "field-row"}>
                    <div className="field-row">
                      <label htmlFor="dp-x-rail-profile-16">X Rail Profile</label>
                      <select id="dp-x-rail-profile-16"
                        value={currentXRailProfile}
                        onChange={(e) => {
                          const nextId = e.target.value;
                          update(
                            (d) => {
                              d.gantry.x_rail_profile = nextId;
                            },
                            `X Rail: ${railProfileShort(currentXRailProfile)} → ${railProfileShort(nextId)}`
                          );
                        }}
                      >
                        {xRailProfileOptions.map((r) => (
                          <option key={r.id} value={r.id}>
                            {r.label}
                          </option>
                        ))}
                      </select>
                    </div>
                    {spec.architecture === "corexy" && (
                      <div className="field-row">
                        <label htmlFor="dp-y-rail-profile-17">Y Rail Profile</label>
                        <select id="dp-y-rail-profile-17"
                          value={currentYRailProfile}
                          onChange={(e) => {
                            const nextId = e.target.value;
                            update(
                              (d) => {
                                d.gantry.y_rail_profile = nextId;
                              },
                              `Y Rail: ${railProfileShort(currentYRailProfile)} → ${railProfileShort(nextId)}`
                            );
                          }}
                        >
                          {yRailProfileOptions.map((r) => (
                            <option key={r.id} value={r.id}>
                              {r.label}
                            </option>
                          ))}
                        </select>
                      </div>
                    )}
                  </div>

                  <div className="field-row">
                    <label id="dp-x-rail-mount-18">
                      <span>X Rail Mount</span>
                      <span style={{ color: "var(--text)", fontWeight: 500 }}>
                        {selectedXRailMount.name}
                      </span>
                    </label>
                    <div role="group" aria-labelledby="dp-x-rail-mount-18" className="voltage-preset-group" style={{ gridTemplateColumns: "repeat(4, 1fr)" }}>
                      {xRailMounts.map((opt) => (
                        <button
                          key={opt.id}
                          type="button"
                          className={`voltage-pill-btn ${currentXRailMount === opt.id ? "active" : ""}`}
 aria-pressed={currentXRailMount === opt.id}
                          style={{ fontSize: 11 }}
                          title={opt.notes}
                          onClick={() => {
                            if (currentXRailMount === opt.id) return;
                            update(
                              (d) => {
                                d.gantry.x_rail_mount = opt.id;
                                d.gantry.dual_x_rails = opt.id === "dual_top_bottom";
                              },
                              `X Rail Mount: ${selectedXRailMount.short_name} → ${opt.short_name}`
                            );
                          }}
                        >
                          {opt.short_name}
                        </button>
                      ))}
                    </div>
                  </div>

                  {spec.architecture === "corexy" && (
                    <div className="field-row">
                      <label id="dp-y-rail-mount-19">
                        <span>Y Rail Mount</span>
                        <span style={{ color: "var(--text)", fontWeight: 500 }}>
                          {selectedYRailMount.name}
                        </span>
                      </label>
                      <div role="group" aria-labelledby="dp-y-rail-mount-19" className="voltage-preset-group" style={{ gridTemplateColumns: "repeat(3, 1fr)" }}>
                        {yRailMounts.map((opt) => (
                          <button
                            key={opt.id}
                            type="button"
                            className={`voltage-pill-btn ${currentYRailMount === opt.id ? "active" : ""}`}
 aria-pressed={currentYRailMount === opt.id}
                            style={{ fontSize: 11 }}
                            title={opt.notes}
                            onClick={() => {
                              if (currentYRailMount === opt.id) return;
                              update(
                                (d) => {
                                  d.gantry.y_rail_mount = opt.id;
                                },
                                `Y Rail Mount: ${selectedYRailMount.short_name} → ${opt.short_name}`
                              );
                            }}
                          >
                            {opt.short_name}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}

                  <div className="field-grid-2">
                    <div className="field-row">
                      <label htmlFor="dp-rail-preload-20">Rail Preload</label>
                      <select id="dp-rail-preload-20"
                        value={currentRailPreload}
                        onChange={(e) => {
                          const nextVal = e.target.value as RailPreloadType;
                          const hadOv = Boolean(
                            spec.model_overrides?.rail_preload &&
                              Object.values(spec.model_overrides.rail_preload).some((v) => v != null)
                          );
                          update(
                            (d) => {
                              d.gantry.rail_preload = nextVal;
                              clearModelOverrideGroup(d, "rail_preload");
                            },
                            `Rail Preload: ${currentRailPreload} → ${nextVal}${hadOv ? " (override reset)" : ""}`
                          );
                        }}
                      >
                        {RAIL_PRELOAD_OPTIONS.map((opt) => (
                          <option key={opt.id} value={opt.id}>
                            {opt.label}
                          </option>
                        ))}
                      </select>
                    </div>
                    <div className="field-row">
                      <label htmlFor="dp-rail-quality-grade-21">Rail Quality / Grade</label>
                      <select id="dp-rail-quality-grade-21"
                        value={currentRailQuality}
                        onChange={(e) => {
                          const nextVal = e.target.value as RailQualityType;
                          const prevLbl = selectedRailQuality.short_name;
                          const nextLbl =
                            railQualityOptions.find((q) => q.id === nextVal)?.short_name ?? nextVal;
                          const hadOv = Boolean(
                            spec.model_overrides?.rail_quality &&
                              Object.values(spec.model_overrides.rail_quality).some((v) => v != null)
                          );
                          update(
                            (d) => {
                              d.gantry.rail_quality = nextVal;
                              clearModelOverrideGroup(d, "rail_quality");
                            },
                            `Rail Quality: ${prevLbl} → ${nextLbl}${hadOv ? " (override reset)" : ""}`
                          );
                        }}
                      >
                        {railQualityOptions.map((opt) => (
                          <option key={opt.id} value={opt.id}>
                            {opt.name}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>
                  {showHints && selectedRailQuality.notes && (
                    <div
                      style={{
                        fontSize: 11,
                        color: "var(--muted)",
                        marginTop: -2,
                        marginBottom: 6,
                        lineHeight: 1.35,
                      }}
                    >
                      {selectedRailQuality.notes}
                    </div>
                  )}
                  <ModelOverrideControl
                    visible={advancedMode}
                    idPrefix="ov-rail-preload"
                    title="Rail Preload"
                    group="rail_preload"
                    spec={spec}
                    sim={sim}
                    update={update}
                    fields={[
                      {
                        key: "preload_fraction",
                        label: "Preload Fraction",
                        unit: "×C",
                        min: 0.0005,
                        max: 0.25,
                        step: 0.005,
                        decimals: 3,
                        fallbackBaseline:
                          catalog.rail_preloads?.find((p) => p.id === currentRailPreload)
                            ?.preload_fraction ?? FALLBACK_RAIL_PRELOAD_FRACTION[currentRailPreload],
                        tooltip: "Ball preload as fraction of dynamic load rating C (sets Hertzian stiffness & drag)",
                      },
                    ]}
                  />
                  <ModelOverrideControl
                    visible={advancedMode}
                    idPrefix="ov-rail-quality"
                    title="Rail Quality"
                    group="rail_quality"
                    spec={spec}
                    sim={sim}
                    update={update}
                    fields={[
                      {
                        key: "stiffness_mult",
                        label: "Stiffness Mult",
                        unit: "×",
                        min: 0.1,
                        max: 3.0,
                        step: 0.05,
                        decimals: 2,
                        fallbackBaseline: selectedRailQuality.stiffness_mult,
                        tooltip: "Ball-sharing & raceway conformity multiplier on carriage contact stiffness",
                      },
                      {
                        key: "drag_mult",
                        label: "Drag Mult",
                        unit: "×",
                        min: 0.1,
                        max: 5.0,
                        step: 0.05,
                        decimals: 2,
                        fallbackBaseline: selectedRailQuality.drag_mult,
                        tooltip: "Seal & rolling friction multiplier on rail carriage drag",
                      },
                    ]}
                  />

                  <div className="field-grid-2 stack">
                    <div className="field-row">
                      <label htmlFor="dp-carriage-block-22">Carriage Block</label>
                      <select id="dp-carriage-block-22"
                        value={currentRailBlock}
                        onChange={(e) => {
                          const nextVal = e.target.value as RailBlockType;
                          const prevLbl =
                            RAIL_BLOCK_OPTIONS.find((o) => o.id === currentRailBlock)?.label ?? currentRailBlock;
                          const nextLbl =
                            RAIL_BLOCK_OPTIONS.find((o) => o.id === nextVal)?.label ?? nextVal;
                          update(
                            (d) => {
                              d.gantry.rail_block_type = nextVal;
                            },
                            `Carriage Block: ${prevLbl} → ${nextLbl}`
                          );
                        }}
                      >
                        {RAIL_BLOCK_OPTIONS.map((opt) => (
                          <option key={opt.id} value={opt.id}>
                            {opt.label}
                          </option>
                        ))}
                      </select>
                    </div>
                    <div className="field-row">
                      <label htmlFor="dp-rail-bolt-pitch-23">Rail Bolt Pitch</label>
                      <select id="dp-rail-bolt-pitch-23"
                        value={currentRailBoltPitch}
                        onChange={(e) => {
                          const nextVal = Number(e.target.value);
                          if (!Number.isFinite(nextVal) || nextVal <= 0) return;
                          update(
                            (d) => {
                              d.gantry.rail_bolt_pitch_mm = nextVal;
                            },
                            `Rail Bolt Pitch: ${currentRailBoltPitch}mm → ${nextVal}mm`
                          );
                        }}
                      >
                        <option value={25}>25 mm (Every Hole)</option>
                        <option value={50}>50 mm (Every Other)</option>
                      </select>
                    </div>
                  </div>

                  <div className="field-row">
                    <label htmlFor="dp-thermal-rail-backer-24">Thermal Rail Backer</label>
                    <select id="dp-thermal-rail-backer-24"
                      value={currentRailBacker}
                      onChange={(e) => {
                        const nextVal = e.target.value as RailBackerMaterialType;
                        const prevLbl =
                          RAIL_BACKER_OPTIONS.find((o) => o.id === currentRailBacker)?.shortLabel ??
                          currentRailBacker;
                        const nextLbl =
                          RAIL_BACKER_OPTIONS.find((o) => o.id === nextVal)?.shortLabel ?? nextVal;
                        update(
                          (d) => {
                            d.gantry.rail_backer_material = nextVal;
                          },
                          `Rail Backer: ${prevLbl} → ${nextLbl}`
                        );
                      }}
                    >
                      {RAIL_BACKER_OPTIONS.map((opt) => (
                        <option key={opt.id} value={opt.id}>
                          {opt.label}
                        </option>
                      ))}
                    </select>
                  </div>

                  {spec.architecture !== "delta" &&
                    sim?.geometry?.linear_rails &&
                    (() => {
                      const lr = sim.geometry.linear_rails;
                      const fmtSignedPct = (v: number) => `${v >= 0 ? "+" : ""}${Math.round(v)}%`;

                      const beamParts: string[] = [];
                      if (Number.isFinite(lr.x_beam_ei_y_boost_pct)) {
                        beamParts.push(`EI_y ${fmtSignedPct(lr.x_beam_ei_y_boost_pct)}`);
                      }
                      if (Number.isFinite(lr.x_beam_ei_z_boost_pct)) {
                        beamParts.push(`EI_z ${fmtSignedPct(lr.x_beam_ei_z_boost_pct)}`);
                      }
                      if (
                        lr.x_beam_gj_boost_pct != null &&
                        Number.isFinite(lr.x_beam_gj_boost_pct)
                      ) {
                        beamParts.push(`GJ ${fmtSignedPct(lr.x_beam_gj_boost_pct)}`);
                      }
                      if (
                        lr.x_rail_newmark_gamma != null &&
                        Number.isFinite(lr.x_rail_newmark_gamma)
                      ) {
                        beamParts.push(`γ ${lr.x_rail_newmark_gamma.toFixed(2)}`);
                      }
                      if (lr.cf_rail_mounting === "two_wall_clamped_inserts") {
                        if (
                          lr.x_beam_insert_mass_g != null &&
                          Number.isFinite(lr.x_beam_insert_mass_g) &&
                          lr.x_beam_insert_mass_g > 0
                        ) {
                          beamParts.push(
                            `2-Wall Inserts (+${Math.round(lr.x_beam_insert_mass_g)} g)`
                          );
                        } else {
                          beamParts.push("2-Wall Inserts");
                        }
                      } else if (lr.cf_rail_mounting === "single_wall_bare") {
                        beamParts.push("Single-Wall Bare");
                      }

                      const carriageParts: string[] = [];
                      if (
                        lr.x_carriage_block_mass_g != null &&
                        Number.isFinite(lr.x_carriage_block_mass_g)
                      ) {
                        let massStr = `${Math.round(lr.x_carriage_block_mass_g)} g`;
                        if (
                          lr.x_carriage_block_delta_g != null &&
                          Number.isFinite(lr.x_carriage_block_delta_g) &&
                          Math.abs(lr.x_carriage_block_delta_g) >= 0.5
                        ) {
                          const roundedDelta = Math.round(lr.x_carriage_block_delta_g);
                          if (roundedDelta !== 0) {
                            massStr += ` (${roundedDelta > 0 ? "+" : ""}${roundedDelta} g vs MGN12H)`;
                          }
                        }
                        carriageParts.push(massStr);
                      }
                      const hasRx =
                        lr.x_carriage_k_rx_nm_rad != null &&
                        Number.isFinite(lr.x_carriage_k_rx_nm_rad);
                      const hasRy =
                        lr.x_carriage_k_ry_nm_rad != null &&
                        Number.isFinite(lr.x_carriage_k_ry_nm_rad);
                      if (hasRx && hasRy) {
                        carriageParts.push(
                          `Roll ${Math.round(lr.x_carriage_k_rx_nm_rad!).toLocaleString()} / Pitch ${Math.round(lr.x_carriage_k_ry_nm_rad!).toLocaleString()} N·m/rad`
                        );
                      } else if (hasRx) {
                        carriageParts.push(
                          `Roll ${Math.round(lr.x_carriage_k_rx_nm_rad!).toLocaleString()} N·m/rad`
                        );
                      } else if (hasRy) {
                        carriageParts.push(
                          `Pitch ${Math.round(lr.x_carriage_k_ry_nm_rad!).toLocaleString()} N·m/rad`
                        );
                      }
                      if (
                        lr.x_carriage_k_rad_n_per_um != null &&
                        Number.isFinite(lr.x_carriage_k_rad_n_per_um)
                      ) {
                        carriageParts.push(
                          `${Math.round(lr.x_carriage_k_rad_n_per_um).toLocaleString()} N/µm`
                        );
                      }

                      const yCarriageParts: string[] = [];
                      if (
                        spec.architecture === "corexy" &&
                        lr.y_carriage_block_mass_g != null &&
                        Number.isFinite(lr.y_carriage_block_mass_g) &&
                        lr.y_carriage_block_mass_g > 0
                      ) {
                        let yMassStr = `2× ${Math.round(lr.y_carriage_block_mass_g)} g`;
                        if (
                          lr.y_carriage_block_delta_g != null &&
                          Number.isFinite(lr.y_carriage_block_delta_g) &&
                          Math.abs(lr.y_carriage_block_delta_g) >= 0.5
                        ) {
                          const roundedYDelta = Math.round(lr.y_carriage_block_delta_g);
                          if (roundedYDelta !== 0) {
                            yMassStr += ` (${roundedYDelta > 0 ? "+" : ""}${roundedYDelta} g vs 2×12H)`;
                          }
                        }
                        yCarriageParts.push(yMassStr);
                        if (
                          lr.y_carriage_k_rx_nm_rad != null &&
                          Number.isFinite(lr.y_carriage_k_rx_nm_rad)
                        ) {
                          yCarriageParts.push(
                            `Roll ${Math.round(lr.y_carriage_k_rx_nm_rad).toLocaleString()} N·m/rad`
                          );
                        }
                      }

                      if (
                        beamParts.length === 0 &&
                        carriageParts.length === 0 &&
                        yCarriageParts.length === 0
                      ) {
                        return null;
                      }

                      return (
                        <div
                          className="cad-driven-summary cad-driven-summary-lines"
                          title="Composite X-beam stiffness boost from bolted linear rail (γ = Newmark partial-composite shear coupling, 0 = uncoupled, 1 = fully bonded), CF rail insert clamping, and Hertzian X/Y carriage block mass & contact stiffness."
                        >
                          {beamParts.length > 0 && (
                            <div>Composite X-Beam: {beamParts.join(" · ")}</div>
                          )}
                          {carriageParts.length > 0 && (
                            <div>Carriage: {carriageParts.join(" · ")}</div>
                          )}
                          {yCarriageParts.length > 0 && (
                            <div>Y-Carriages: {yCarriageParts.join(" · ")}</div>
                          )}
                        </div>
                      );
                    })()}
                </div>
              </details>

              <details className="sub-accordion">
                <summary>
                  <span>Belt Routing &amp; Servo Control</span>
                  <span className="sub-accordion-tag">
                    {spec.architecture === "corexy" ? `${beltRoutingOptionMeta.shortLabel} · ` : ""}
                    {DRIVE_TYPE_OPTIONS.find((o) => o.id === currentDriveType)?.shortLabel ??
                      currentDriveType}
                  </span>
                </summary>
                <div className="sub-accordion-body">
                  {spec.architecture === "corexy" && (
                    <div className="field-row">
                      <label htmlFor="dp-belt-path-and-idlers-25">
                        <span>Belt Path &amp; Idlers</span>
                        <span
                          style={{
                            color:
                              currentBeltRouting === "monolith_short"
                                ? "var(--accent)"
                                : "var(--muted)",
                            fontWeight: 500,
                          }}
                        >
                          {beltRoutingOptionMeta.shortLabel}
                        </span>
                      </label>
                      <select id="dp-belt-path-and-idlers-25"
                        value={currentBeltRouting}
                        onChange={(e) => {
                          const nextVal = e.target.value as BeltRoutingType;
                          const prevLbl =
                            BELT_ROUTING_OPTIONS.find((o) => o.id === currentBeltRouting)
                              ?.shortLabel ?? currentBeltRouting;
                          const nextLbl =
                            BELT_ROUTING_OPTIONS.find((o) => o.id === nextVal)?.shortLabel ??
                            nextVal;
                          const hadOv = Boolean(
                            spec.model_overrides?.belt_routing &&
                              Object.values(spec.model_overrides.belt_routing).some((v) => v != null)
                          );
                          update(
                            (d) => {
                              d.gantry.belt_routing = nextVal;
                              clearModelOverrideGroup(d, "belt_routing");
                            },
                            `Belt Path: ${prevLbl} → ${nextLbl}${hadOv ? " (override reset)" : ""}`
                          );
                        }}
                      >
                        {BELT_ROUTING_OPTIONS.map((opt) => {
                          const catItem = catalog.belt_routings?.find((r) => r.id === opt.id);
                          return (
                            <option key={opt.id} value={opt.id}>
                              {catItem?.name ?? opt.label}
                            </option>
                          );
                        })}
                      </select>
                      {showHints && (
                        <div className="hint-line">
                          {beltRoutingCatalogEntry?.notes || beltRoutingOptionMeta.hint}
                        </div>
                      )}
                      {(() => {
                        const fb = FALLBACK_BELT_ROUTING_PARAMS[currentBeltRouting];
                        return (
                          <ModelOverrideControl
                            visible={advancedMode}
                            idPrefix="ov-belt-routing"
                            title="Belt Routing"
                            group="belt_routing"
                            spec={spec}
                            sim={sim}
                            update={update}
                            fields={[
                              {
                                key: "span_factor",
                                label: "Span Factor",
                                unit: "×",
                                min: 0.3,
                                max: 2.0,
                                step: 0.02,
                                decimals: 2,
                                fallbackBaseline: beltRoutingCatalogEntry?.span_factor ?? fb.span_factor,
                                tooltip: "Multiplier on effective free belt span length",
                              },
                              {
                                key: "idler_stiffness_factor",
                                label: "Idler Stiffness",
                                unit: "×",
                                min: 0.2,
                                max: 5.0,
                                step: 0.05,
                                decimals: 2,
                                fallbackBaseline:
                                  beltRoutingCatalogEntry?.idler_stiffness_factor ??
                                  fb.idler_stiffness_factor,
                                tooltip: "Multiplier on idler pin/shaft bending stiffness",
                              },
                            ]}
                          />
                        );
                      })()}
                    </div>
                  )}

                  <div className="field-grid-2">
                    <div className="field-row">
                      <label htmlFor="dp-idler-shaft-26">Idler Shaft</label>
                      <select id="dp-idler-shaft-26"
                        value={currentIdlerShaftMm}
                        onChange={(e) => {
                          const nextVal = Number(e.target.value);
                          if (!Number.isFinite(nextVal) || nextVal <= 0) return;
                          update(
                            (d) => {
                              d.gantry.belt.idler_shaft_diameter_mm = nextVal;
                            },
                            `Idler Shaft: ${currentIdlerShaftMm}mm → ${nextVal}mm Pin`
                          );
                        }}
                      >
                        <option value={5}>5 mm Pin</option>
                        <option value={6}>6 mm Pin</option>
                      </select>
                    </div>
                    <div className="field-row">
                      <label htmlFor="dp-bearing-play-27">Bearing Play</label>
                      <select id="dp-bearing-play-27"
                        value={currentBearingClass}
                        onChange={(e) => {
                          const nextVal = e.target.value as BearingClearanceClass;
                          const prevLbl =
                            BEARING_CLEARANCE_OPTIONS.find((o) => o.id === currentBearingClass)
                              ?.shortLabel ?? currentBearingClass;
                          const nextLbl =
                            BEARING_CLEARANCE_OPTIONS.find((o) => o.id === nextVal)?.shortLabel ??
                            nextVal;
                          const hadOv = Boolean(
                            spec.model_overrides?.bearing_clearance &&
                              Object.values(spec.model_overrides.bearing_clearance).some((v) => v != null)
                          );
                          update(
                            (d) => {
                              d.gantry.belt.bearing_clearance_class = nextVal;
                              clearModelOverrideGroup(d, "bearing_clearance");
                            },
                            `Bearing Play: ${prevLbl} → ${nextLbl}${hadOv ? " (override reset)" : ""}`
                          );
                        }}
                      >
                        {BEARING_CLEARANCE_OPTIONS.map((opt) => (
                          <option key={opt.id} value={opt.id}>
                            {opt.label}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>

                  <div className="field-row">
                    <label htmlFor="dp-idler-axial-preload">Idler Axial Preload</label>
                    <select
                      id="dp-idler-axial-preload"
                      value={spec.gantry.idler_axial_preload ?? "auto"}
                      title="M5 Belleville disc springs axially preload F695/695-2RS idler bearing stacks to eliminate axial play and reduce CTE press-fit binding"
                      onChange={(e) => {
                        const nextVal = e.target.value as IdlerAxialPreloadType;
                        const labels: Record<IdlerAxialPreloadType, string> = {
                          auto: "Auto",
                          none: "None",
                          disc_spring_m5: "M5 Disc Springs",
                        };
                        const prevKey = spec.gantry.idler_axial_preload ?? "auto";
                        update(
                          (d) => {
                            d.gantry.idler_axial_preload = nextVal;
                          },
                          `Idler Axial Preload: ${labels[prevKey]} → ${labels[nextVal]}`
                        );
                      }}
                    >
                      <option value="auto">Auto</option>
                      <option value="none">None</option>
                      <option value="disc_spring_m5">M5 Disc Springs</option>
                    </select>
                  </div>
                  <ModelOverrideControl
                    visible={advancedMode}
                    idPrefix="ov-bearing-clearance"
                    title="Bearing Clearance"
                    group="bearing_clearance"
                    spec={spec}
                    sim={sim}
                    update={update}
                    fields={[
                      {
                        key: "radial_play_um",
                        label: "Bearing Radial Play",
                        unit: "µm",
                        min: 0,
                        max: 100,
                        step: 0.5,
                        decimals: 1,
                        fallbackBaseline:
                          catalog.bearing_clearances?.find((b) => b.id === currentBearingClass)
                            ?.radial_play_um ?? FALLBACK_BEARING_PLAY_UM[currentBearingClass],
                        tooltip: "ISO 5753 radial internal clearance of idler pulley bearings",
                      },
                    ]}
                  />

                  <div className="field-row">
                    <label htmlFor="dp-drive-control-28">
                      <span>Drive Control</span>
                      <span style={{ color: "var(--accent)", fontWeight: 500 }}>
                        {DRIVE_TYPE_OPTIONS.find((o) => o.id === currentDriveType)?.shortLabel ??
                          currentDriveType}
                      </span>
                    </label>
                    <select id="dp-drive-control-28"
                      value={currentDriveType}
                      onChange={(e) => {
                        const nextVal = e.target.value as DriveType;
                        if (nextVal === currentDriveType) return;
                        const prevLbl =
                          DRIVE_TYPE_OPTIONS.find((o) => o.id === currentDriveType)?.shortLabel ??
                          currentDriveType;
                        const nextLbl =
                          DRIVE_TYPE_OPTIONS.find((o) => o.id === nextVal)?.shortLabel ?? nextVal;
                        update(
                          (d) => {
                            d.kinematics.drive_type = nextVal;
                            if (
                              nextVal !== "open_loop_stepper" &&
                              d.kinematics.servo_vel_bw_hz == null
                            ) {
                              d.kinematics.servo_vel_bw_hz = nextVal === "foc_servo" ? 180 : 120;
                            }
                          },
                          `Drive Control: ${prevLbl} → ${nextLbl}`
                        );
                      }}
                    >
                      {DRIVE_TYPE_OPTIONS.map((opt) => (
                        <option key={opt.id} value={opt.id}>
                          {opt.label}
                        </option>
                      ))}
                    </select>
                  </div>

                  {currentDriveType !== "open_loop_stepper" && (
                    <div className="field-grid-2">
                      <div className="field-row">
                        <label htmlFor="dp-notch-filter-hz-29">
                          <span>Notch Filter (Hz)</span>
                          <span style={{ fontSize: 11, color: "var(--muted)" }}>blank = Off</span>
                        </label>
                        <input id="dp-notch-filter-hz-29"
                          type="number"
                          min={20}
                          max={500}
                          step={5}
                          placeholder="Off (e.g. 120)"
                          value={notchDraft}
                          onChange={(e) => {
                            const raw = e.target.value;
                            setNotchDraft(raw);
                            if (raw.trim() === "") {
                              commitNotchFilter("");
                            }
                          }}
                          onBlur={() => commitNotchFilter(notchDraft)}
                          onKeyDown={(e) => {
                            if (e.key === "Enter") {
                              commitNotchFilter(notchDraft);
                            }
                          }}
                        />
                      </div>
                      <div className="field-row">
                        <label htmlFor="dp-vel-bw-hz-30">Vel BW (Hz)</label>
                        <input id="dp-vel-bw-hz-30"
                          type="number"
                          min={20}
                          max={1000}
                          step={10}
                          value={currentServoVelBw}
                          onChange={(e) =>
                            updatePositiveNum(
                              e.target.value,
                              (d, v) => {
                                d.kinematics.servo_vel_bw_hz = Math.max(20, Math.min(1000, v));
                              },
                              (v) =>
                                `Servo Vel BW: ${Math.round(currentServoVelBw)} Hz → ${Math.round(v)} Hz`
                            )
                          }
                        />
                      </div>
                    </div>
                  )}
                </div>
              </details>

              <details className="sub-accordion beta-accordion">
                <summary>
                  <span>
                    <span>Kalico / FOC Beta Features</span>
                    <span className="beta-pill">BETA</span>
                  </span>
                  <span
                    className={`sub-accordion-tag ${activeBetaCount > 0 ? "beta-active-tag" : ""}`}
                  >
                    {activeBetaCount > 0 ? `${activeBetaCount} active` : "Off"}
                  </span>
                </summary>
                <div className="sub-accordion-body">
                  {/* One-click Kalico FOC Beta Stack bar */}
                  <div className="beta-stack-bar">
                    {!isFocBetaStackActive ? (
                      <button
                        type="button"
                        className="beta-stack-btn"
                        onClick={handleLoadFocBetaStack}
                        title="Enable FOC Servo, SMOOTH_FLATNESS, LPV 3×3 scheduling, AWD Position Equalizer (2.0 Hz), Pin-Rotor FF, and Mode-Space Torque FF in one click"
                      >
                        Load Kalico FOC Beta Stack
                      </button>
                    ) : (
                      <span
                        className="beta-pill ok"
                        style={{ flex: "1 1 auto", justifyContent: "center", padding: "4px 7px" }}
                      >
                        Kalico FOC Stack Active
                      </span>
                    )}
                    {activeBetaCount > 0 && (
                      <button
                        type="button"
                        className="beta-stack-btn disable"
                        onClick={handleDisableBetaStack}
                        title="Reset Kalico / FOC beta features back to off"
                      >
                        Disable Beta Stack
                      </button>
                    )}
                  </div>

                  {/* Open-loop drive info banner */}
                  {isOpenLoopDrive && (
                    <div className="beta-foc-banner">
                      <div>
                        Real-time FOC loops (<code>0x65</code>/<code>0x67</code> Pin-Rotor,
                        Equalizer, IMU Damping, Mode-Space FF) require Closed-Loop Stepper or FOC
                        Servo drive.
                      </div>
                      <div className="beta-foc-banner-actions">
                        <button
                          type="button"
                          className="beta-mini-btn"
                          onClick={() => {
                            update(
                              (d) => {
                                d.kinematics.drive_type = "foc_servo";
                                if (d.kinematics.servo_vel_bw_hz == null) {
                                  d.kinematics.servo_vel_bw_hz = 180;
                                }
                              },
                              "Drive Control: Open-Loop Stepper → FOC Servo"
                            );
                          }}
                        >
                          Switch to FOC Servo
                        </button>
                      </div>
                    </div>
                  )}

                  {/* 1. LPV 3x3 Bed Resonance Scheduling */}
                  <div className="beta-control-group">
                    <label
                      className={`beta-toggle-row ${currentLpv ? "active" : ""}`}
                      title="Schedules smoother_freq(x, y) across a 3×3 bed map so narrow shapers (SMOOTH_FLATNESS / ZV) stay tuned in every corner"
                    >
                      <div className="beta-toggle-main">
                        <span className="beta-toggle-head">
                          <span>LPV 3×3 Bed Resonance Scheduling</span>
                          <span className="beta-pill">f(x, y)</span>
                        </span>
                      </div>
                      <input
                        type="checkbox"
                        checked={currentLpv}
                        onChange={(e) => {
                          const checked = e.target.checked;
                          update(
                            (d) => {
                              d.kinematics.lpv_workspace_scheduling = checked;
                            },
                            `LPV 3×3 Scheduling: ${currentLpv ? "On" : "Off"} → ${checked ? "On" : "Off"}`
                          );
                        }}
                      />
                    </label>
                    <div className="beta-setting-note">
                      <span>
                        <strong>What:</strong> Schedules <code>smoother_freq(x, y)</code> across a
                        3×3 bed map so narrow shapers (<code>SMOOTH_FLATNESS</code> /{" "}
                        <code>ZV</code>) stay tuned in every corner.
                      </span>{" "}
                      <span>
                        <strong>How to set:</strong> Turn ON for CoreXY with{" "}
                        <code>SMOOTH_FLATNESS</code>; capture 9 ADXL sweeps across the bed on
                        hardware to populate the map.
                      </span>
                    </div>
                  </div>

                  {/* 2. AWD FOC Sync Mode + Equalizer BW */}
                  <div className="beta-subcard">
                    <div className="beta-subcard-head">
                      <span>AWD FOC Sync Mode</span>
                      {driveMode !== "awd" ? (
                        <span className="beta-pill muted">2WD Build</span>
                      ) : currentAwdSync === "position_equalizer" ? (
                        <span
                          className={`beta-pill ${
                            awdEqBwDraft > 4.0
                              ? "warn"
                              : awdEqBwDraft >= 1.5 && awdEqBwDraft <= 2.5
                                ? "ok"
                                : ""
                          }`}
                          title={
                            awdEqBwDraft > 4.0
                              ? "Equalizer BW > 4 Hz risks fighting 1×/rev pulley runout and injecting belt ripple"
                              : awdEqBwDraft >= 1.5 && awdEqBwDraft <= 2.5
                                ? "Recommended 1.5–2.5 Hz sweet spot: rejects static load sharing drift without fighting 1×/rev pulley runout"
                                : "Equalizer loop crossover frequency between paired AWD motors"
                          }
                        >
                          {awdEqBwDraft > 4.0
                            ? "1×/rev Runout Risk (>4 Hz)"
                            : awdEqBwDraft >= 1.5 && awdEqBwDraft <= 2.5
                              ? "Runout-Safe (1.5–2.5 Hz)"
                              : awdEqBwDraft < 1.5
                                ? "Ultra-Soft (<1.5 Hz)"
                                : "Moderate (2.5–4.0 Hz)"}
                        </span>
                      ) : (
                        <span className="beta-pill muted">
                          {AWD_SYNC_OPTIONS.find((o) => o.id === currentAwdSync)?.shortLabel ??
                            currentAwdSync}
                        </span>
                      )}
                    </div>

                    <select aria-label="AWD FOC Sync Mode"
                      value={currentAwdSync}
                      onChange={(e) => {
                        const nextVal = e.target.value as AwdSyncMode;
                        if (nextVal === currentAwdSync) return;
                        update(
                          (d) => {
                            d.kinematics.awd_sync_mode = nextVal;
                            if (
                              nextVal === "position_equalizer" &&
                              d.kinematics.awd_equalizer_bw_hz == null
                            ) {
                              d.kinematics.awd_equalizer_bw_hz = 2.0;
                            }
                          },
                          `AWD Sync: ${AWD_SYNC_LABELS[currentAwdSync] ?? currentAwdSync} → ${AWD_SYNC_LABELS[nextVal] ?? nextVal}`
                        );
                      }}
                    >
                      {AWD_SYNC_OPTIONS.map((opt) => (
                        <option key={opt.id} value={opt.id}>
                          {opt.label}
                        </option>
                      ))}
                    </select>

                    {currentAwdSync === "position_equalizer" && (
                      <div className="slider-row" style={{ marginTop: 2 }}>
                        <label htmlFor="dp-equalizer-bw-hz-31">
                          <span>Equalizer BW (Hz)</span>
                          <span style={{ color: "var(--accent)", fontWeight: 600 }}>
                            {awdEqBwDraft.toFixed(1)} Hz
                          </span>
                        </label>
                        <input id="dp-equalizer-bw-hz-31"
                          type="range"
                          min={0.5}
                          max={15.0}
                          step={0.5}
                          value={awdEqBwDraft}
                          onChange={(e) => handleAwdEqBwSliderChange(Number(e.target.value))}
                          onMouseUp={() => commitAwdEqBw(awdEqBwDraft)}
                          onTouchEnd={() => commitAwdEqBw(awdEqBwDraft)}
                        />
                      </div>
                    )}

                    <div className="beta-setting-note">
                      <span>
                        <strong>What:</strong> Keeps front &amp; rear AWD motors sharing torque
                        cleanly without belt slosh or static fighting.
                      </span>{" "}
                      <span>
                        <strong>How to set:</strong> Pick{" "}
                        <strong>Stage 2: Position + Equalizer</strong> at{" "}
                        <strong>1.5–2.5 Hz</strong> (<code>2.0 Hz</code> default) with
                        velocity/accel freeze; keep <code>≤ 4 Hz</code> so it never chases 1×/rev
                        pulley runout (6–25 Hz).
                      </span>
                    </div>
                  </div>

                  {/* 3. TMC4671 Hardware Notch (BIQUAD_T) Depth & Q */}
                  <div className="beta-subcard">
                    <div className="beta-subcard-head">
                      <span>TMC4671 Notch (BIQUAD_T)</span>
                      <span
                        className={`beta-pill ${notchPhaseMetrics.maxLagDeg <= 45 ? "ok" : "warn"}`}
                        title={`Exact BIQUAD_T max phase lag at ${(notchPhaseMetrics.peakLagFreqRatio * 100).toFixed(0)}% of notch freq (≤45° preserves FOC velocity loop phase margin; BW ≈ f₀ / ${notchQDraft.toFixed(2)})`}
                      >
                        Max Lag -{Math.round(notchPhaseMetrics.maxLagDeg)}°
                      </span>
                    </div>

                    <div className="field-row">
                      <label htmlFor="dp-notch-freq-hz-32">
                        <span>Notch Freq (Hz)</span>
                        <span style={{ fontSize: 11, color: "var(--muted)" }}>
                          {spec.kinematics.notch_filter_hz != null &&
                          spec.kinematics.notch_filter_hz > 0
                            ? `Peak lag @ ~${Math.round(spec.kinematics.notch_filter_hz * notchPhaseMetrics.peakLagFreqRatio)} Hz`
                            : "blank = Off"}
                        </span>
                      </label>
                      <input id="dp-notch-freq-hz-32"
                        type="number"
                        min={20}
                        max={500}
                        step={5}
                        placeholder="Off (e.g. 120)"
                        value={notchDraft}
                        onChange={(e) => {
                          const raw = e.target.value;
                          setNotchDraft(raw);
                          if (raw.trim() === "") {
                            commitNotchFilter("");
                          }
                        }}
                        onBlur={() => commitNotchFilter(notchDraft)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") {
                            commitNotchFilter(notchDraft);
                          }
                        }}
                      />
                    </div>

                    <div className="field-grid-2">
                      <div className="slider-row">
                        <label htmlFor="dp-depth-db-33">
                          <span>Depth (dB)</span>
                          <span style={{ color: "var(--accent)", fontWeight: 600 }}>
                            {notchDepthDraft.toFixed(1)} dB
                          </span>
                        </label>
                        <input id="dp-depth-db-33"
                          type="range"
                          min={3}
                          max={24}
                          step={0.5}
                          value={Math.min(24, Math.max(3, notchDepthDraft))}
                          onChange={(e) => handleNotchDepthSliderChange(Number(e.target.value))}
                          onMouseUp={() => commitNotchDepth(notchDepthDraft)}
                          onTouchEnd={() => commitNotchDepth(notchDepthDraft)}
                        />
                      </div>
                      <div className="slider-row">
                        <label htmlFor="dp-notch-q-34">
                          <span>Notch Q</span>
                          <span style={{ color: "var(--accent)", fontWeight: 600 }}>
                            {notchQDraft.toFixed(2)}
                          </span>
                        </label>
                        <input id="dp-notch-q-34"
                          type="range"
                          min={0.8}
                          max={5.0}
                          step={0.05}
                          value={Math.min(5.0, Math.max(0.8, notchQDraft))}
                          onChange={(e) => handleNotchQSliderChange(Number(e.target.value))}
                          onMouseUp={() => commitNotchQ(notchQDraft)}
                          onTouchEnd={() => commitNotchQ(notchQDraft)}
                        />
                      </div>
                    </div>

                    <div className="beta-setting-note">
                      <span>
                        <strong>What:</strong> Carves out belt/frame resonance inside the TMC4671
                        velocity feedback loop at 25 kHz PWM.
                      </span>{" "}
                      <span>
                        <strong>How to set:</strong> Set <strong>Freq</strong> to your primary
                        resonance peak, <strong>Depth = 10–12 dB</strong>,{" "}
                        <strong>Q = 1.5–2.0</strong> (caps phase lag at <code>-31°</code> to{" "}
                        <code>-44°</code>; avoid <code>&gt; 15 dB</code> or servo phase margin
                        drops when belt frequency shifts across the bed).
                      </span>
                    </div>
                  </div>

                  {/* 4. Pin-Rotor Belt-Reaction FF (0x65) */}
                  <div className="beta-control-group">
                    <label
                      className={`beta-toggle-row ${currentPinRotor ? "active" : ""} ${
                        isOpenLoopDrive ? "foc-inactive" : ""
                      }`}
                      title="Cancels belt recoil on the rotor via register 0x65, recovering locked-rotor belt resonance f_b without extra sensors"
                    >
                      <div className="beta-toggle-main">
                        <span className="beta-toggle-head">
                          <span>Pin-Rotor Belt-Reaction FF</span>
                          <span className="beta-pill">0x65</span>
                        </span>
                      </div>
                      <input
                        type="checkbox"
                        checked={currentPinRotor}
                        onChange={(e) => {
                          const checked = e.target.checked;
                          update(
                            (d) => {
                              d.kinematics.pin_rotor_ff = checked;
                            },
                            `Pin-Rotor FF (0x65): ${currentPinRotor ? "On" : "Off"} → ${checked ? "On" : "Off"}`
                          );
                        }}
                      />
                    </label>
                    <div className="beta-setting-note">
                      <span>
                        <strong>What:</strong> Sensorless 2-state belt-deflection observer in the
                        servo ISR injecting torque offset (<code>0x65</code>) to cancel rotor
                        recoil, raising resonance toward the stiffer locked-rotor frequency{" "}
                        <em>f</em><sub>b</sub>.
                      </span>{" "}
                      <span>
                        <strong>How to set:</strong> Enable on FOC/Closed-Loop drives; tune{" "}
                        <code>pin_mass</code> &amp; <code>pin_lead_us</code> via encoder
                        following-error sweep before measuring <em>f</em><sub>b</sub>.
                      </span>
                    </div>
                  </div>

                  {/* 5. Stage 4 Toolhead IMU Active Damping (ICM-42688-P) */}
                  <div className="beta-control-group">
                    <label
                      className={`beta-toggle-row ${currentImuDamping ? "active" : ""} ${
                        isOpenLoopDrive ? "foc-inactive" : ""
                      }`}
                      title="Injects real-time skyhook damping (+0.07 ζ) from an ICM-42688-P toolhead IMU via TIM6 -> 0x65"
                    >
                      <div className="beta-toggle-main">
                        <span className="beta-toggle-head">
                          <span>Stage 4 Toolhead IMU Active Damping</span>
                          <span className="beta-pill ok">+0.07 ζ</span>
                        </span>
                      </div>
                      <input
                        type="checkbox"
                        checked={currentImuDamping}
                        onChange={(e) => {
                          const checked = e.target.checked;
                          update(
                            (d) => {
                              d.kinematics.toolhead_imu_active_damping = checked;
                            },
                            `Toolhead IMU Damping: ${currentImuDamping ? "On" : "Off"} → ${checked ? "On" : "Off"}`
                          );
                        }}
                      />
                    </label>
                    <div className="beta-setting-note">
                      <span>
                        <strong>What:</strong> Real-time toolhead accelerometer (
                        <code>ICM-42688-P</code>) feedback into <code>TIM6 → 0x65</code> adding
                        active skyhook damping (<code>+0.07 ζ</code>) to flatten PSD peaks.
                      </span>{" "}
                      <span>
                        <strong>How to set:</strong> Experimental — requires low-latency
                        SPI/differential IMU link; start with conservative gain and verify stability
                        before increasing accel.
                      </span>
                    </div>
                  </div>

                  {/* 6. Mode-Space CoreXY Torque FF + Direction Split */}
                  <div className="beta-control-group">
                    <label
                      className={`beta-toggle-row ${currentModeSpaceFf ? "active" : ""} ${
                        isOpenLoopDrive ? "foc-inactive" : ""
                      }`}
                      title="Compensates pulling vs. pushing belt asymmetry and high-accel tracking lag in CoreXY mode space"
                    >
                      <div className="beta-toggle-main">
                        <span className="beta-toggle-head">
                          <span>Mode-Space CoreXY Torque FF</span>
                          <span className="beta-pill">Dir Split</span>
                        </span>
                      </div>
                      <input
                        type="checkbox"
                        checked={currentModeSpaceFf}
                        onChange={(e) => {
                          const checked = e.target.checked;
                          update(
                            (d) => {
                              d.kinematics.mode_space_torque_ff = checked;
                            },
                            `Mode-Space Torque FF: ${currentModeSpaceFf ? "On" : "Off"} → ${checked ? "On" : "Off"}`
                          );
                        }}
                      />
                    </label>
                    <div className="beta-setting-note">
                      <span>
                        <strong>What:</strong> Injects CoreXY diagonal inertia/friction FF (
                        <code>0x67</code>/<code>0x65</code>) with pulling/pushing belt-span
                        compensation.
                      </span>{" "}
                      <span>
                        <strong>How to set:</strong> Enable on FOC/Closed-Loop after steady-plateau
                        inertia &amp; friction ID to cut corner tracking lag and circularity error (
                        <code>G_XY</code>).
                      </span>
                    </div>
                  </div>
                </div>
              </details>
            </div>

  );
};
