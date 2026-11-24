import React, { useEffect, useRef, useState } from "react";
import { onTabListKeyDown } from "../utils/a11y";
import { DesignAnchoringSection, DesignFrameSection } from "./DesignFrameSection";
import { DesignBedZSection } from "./DesignBedZSection";
import {
  AWD_SYNC_LABELS,
  BELT_LABELS,
  SHAPER_LABELS,
  getBeltDefaults,
  getBeltTensionSliderMax,
} from "../constants/printerLabels";
import {
  loadShowHintsPreference,
  saveShowHintsPreference,
} from "../utils/studioStorage";
import {
  AwdSyncMode,
  BearingClearanceClass,
  BeltRoutingType,
  BracketMaterialType,
  CatalogResponse,
  DriveType,
  GantryBracketInfo,
  GantryWeightClassItem,
  HotendMountStyle,
  HotendPresetItem,
  PrinterSpecData,
  RailBackerMaterialType,
  RailBlockType,
  RailMountOption,
  RailPreloadType,
  RailQualityType,
  SimulationResponse,
  SolidFeaEngine,
  ToolheadConstructionCatalogItem,
  ToolheadConstructionType,
  ToolheadMountMaterial,
  TopoOptRequestParams,
  XRailMountType,
  YRailMountType,
  ZProbeType,
} from "../types";

/** Top-level sections of the Printer tab, in order, for the jump bar. */
const DESIGN_SECTIONS = [
  { id: "frame", label: "Frame" },
  { id: "gantry", label: "Gantry" },
  { id: "toolhead", label: "Toolhead" },
  { id: "hotend", label: "Hotend" },
  { id: "bed", label: "Bed" },
  { id: "sim", label: "Sim" },
] as const;

interface Props {
  catalog: CatalogResponse | null;
  sim: SimulationResponse | null;
  selectedPreset: string;
  onSelectPreset: (id: string) => void;
  filamentId: string;
  onFilamentChange: (id: string) => void;
  spec: PrinterSpecData | null;
  onSpecChange: (next: PrinterSpecData, changeLabel?: string) => void;
  savedBuildsCount?: number;
  lastAutoSavedAt?: string | null;
  onOpenSavedBuilds?: () => void;
  toolheadRelPos: [number, number, number];
  onToolheadRelPosChange: (pos: [number, number, number], changeLabel?: string) => void;
  panelMeshN: number;
  onPanelMeshNChange: (n: number) => void;
  yamlText: string;
  onYamlChange: (txt: string) => void;
  onSimulateYaml: () => void;
  gcodeText: string;
  onGcodeChange: (txt: string) => void;
  /** Runs the virtual print with the current G-code box contents. */
  onRunGcode?: () => void;
  onRunPartFea: (params: {
    preset: string;
    materialId: string;
    forceN: [number, number, number];
    fdmAnisotropic: boolean;
    engine: SolidFeaEngine;
  }) => void;
  onRunStlFea: (params: {
    file: File;
    materialId: string;
    fixedFace: string;
    loadFace: string;
    forceN: [number, number, number];
    fdmAnisotropic: boolean;
  }) => void;
  onDownloadPartStep: (preset: string) => void;
  exportingStep: boolean;
  onRunTopoOpt: (params: TopoOptRequestParams) => void;
  onDownloadTopoStl: (params: TopoOptRequestParams) => void;
  hasTopoResult: boolean;
  topoLoading: boolean;
  loading: boolean;
}

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

const HEATBREAK_LABELS: Record<string, string> = {
  titanium_grade5: "Titanium Gr.5",
  stainless_304: "Stainless 304",
};

const X_BEAM_METAL_PROFILE_IDS = new Set([
  "extrusion_2020",
  "extrusion_2040",
  "cnc_al_beam_20x20",
  "al_tube_20x20x2",
  "extrusion_3030",
]);

const X_BEAM_LIGHTWEIGHT_TUBE_IDS = new Set([
  "cf_tube_20x20x1_5",
  "cf_tube_20x20x2",
  "cf_tube_25x25x2",
  "ti_tube_20x20x1_2",
]);

const X_BEAM_MATERIAL_IDS = new Set([
  "aluminum_6063_t5",
  "aluminum_6061_t6",
  "aluminum_7075_t6",
  "carbon_fiber_tube",
  "carbon_fiber_high_tg",
  "titanium_grade5",
  "steel_4140",
]);

const PART_FEA_PRINTED_IDS = new Set([
  "abs_printed",
  "pet_cf_printed",
  "pa12_cf_printed",
  "ppa_cf_printed",
  "pps_cf_printed",
]);

const PART_FEA_METAL_IDS = new Set([
  "aluminum_6061_t6",
  "aluminum_7075_t6",
  "carbon_fiber_plate",
  "steel_4140",
  "stainless_304",
  "titanium_grade5",
]);

const FALLBACK_GANTRY_BRACKETS: GantryBracketInfo[] = [
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
    id: "aluminum_6061_t6",
    name: "CNC Aluminum 6061-T6 (68.9 GPa)",
    short_name: "CNC Al 6061",
    youngs_modulus_gpa: 68.9,
    hdt_c: 300,
    creep_tension_limit_n: 250,
    cnc_machined: true,
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
    kit_cost_usd: 285,
    carriage_mass_delta_g: 46,
    color_hex: "#cbd5e1",
  },
];

const FALLBACK_GANTRY_WEIGHT_CLASSES: GantryWeightClassItem[] = [
  {
    id: "ultralight_printed",
    name: "Ultralight Printed (85g)",
    short_name: "Ultralight Printed",
    carriage_mass_g: 85,
    suggested_bracket_material: "abs_printed",
    notes: "Minimalist printed XY joints & lightweight Y-carriages (Annex / VzBot style).",
  },
  {
    id: "standard_printed",
    name: "Standard Printed (140g)",
    short_name: "Standard Printed",
    carriage_mass_g: 140,
    suggested_bracket_material: "abs_printed",
    notes: "Stock Voron 2.4 / Trident printed ABS/ASA XY joints, idlers & Y-carriages.",
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
    id: "cnc_al_heavy",
    name: "CNC Aluminum Heavy-Duty (280g)",
    short_name: "CNC Al Heavy",
    carriage_mass_g: 280,
    suggested_bracket_material: "aluminum_7075_t6",
    notes: "Heavy billet CNC 7075-T6 AWD / double-shear XY blocks & thick Y-carriages.",
  },
];

function inferAutoXBeamMaterial(profileId: string, currentMaterial: string): string | null {
  if (profileId.startsWith("cf_tube_")) return "carbon_fiber_tube";
  if (profileId.startsWith("ti_tube_")) return "titanium_grade5";
  if (profileId.startsWith("steel_tube_") || profileId.startsWith("steel_rod_")) {
    return "steel_4140";
  }
  if (
    (profileId.startsWith("extrusion_") ||
      profileId.startsWith("al_tube_") ||
      profileId.startsWith("cnc_al_")) &&
    (currentMaterial === "carbon_fiber_tube" ||
      currentMaterial === "titanium_grade5" ||
      currentMaterial === "steel_4140")
  ) {
    return "aluminum_6063_t5";
  }
  return null;
}

const MOUNT_MATERIAL_OPTIONS: Array<{
  id: ToolheadMountMaterial;
  label: string;
  shortLabel: string;
}> = [
  { id: "abs_printed", label: "Printed ABS (2.1 GPa)", shortLabel: "Printed ABS" },
  { id: "pet_cf_printed", label: "Printed PET-CF15 (5.2 GPa)", shortLabel: "Printed PET-CF15" },
  { id: "pa12_cf_printed", label: "Printed PA12-CF (6.2 GPa)", shortLabel: "Printed PA12-CF" },
  { id: "ppa_cf_printed", label: "Printed PPA-CF (11.8 GPa)", shortLabel: "Printed PPA-CF" },
  { id: "pps_cf_printed", label: "Printed PPS-CF (8.5 GPa)", shortLabel: "Printed PPS-CF" },
  { id: "aluminum_6061_t6", label: "CNC Al 6061-T6 (68.9 GPa)", shortLabel: "CNC Al 6061" },
  { id: "aluminum_7075_t6", label: "CNC Al 7075-T6 (71.7 GPa)", shortLabel: "CNC Al 7075" },
];

const TOOLHEAD_PART_MATERIAL_OPTIONS: Array<{
  id: string;
  label: string;
  shortLabel: string;
}> = [
  { id: "abs_printed", label: "Printed ABS (2.1 GPa)", shortLabel: "ABS" },
  { id: "pet_cf_printed", label: "Printed PET-CF15 (5.2 GPa)", shortLabel: "PET-CF" },
  { id: "pa12_cf_printed", label: "Printed PA12-CF (6.2 GPa)", shortLabel: "PA12-CF" },
  { id: "ppa_cf_printed", label: "Printed PPA-CF (11.8 GPa)", shortLabel: "PPA-CF" },
  { id: "pps_cf_printed", label: "Printed PPS-CF (8.5 GPa)", shortLabel: "PPS-CF" },
  { id: "carbon_fiber_plate", label: "3K Carbon Fiber Plate (55 GPa)", shortLabel: "3K CF Plate" },
  { id: "aluminum_6061_t6", label: "CNC Al 6061-T6 (68.9 GPa)", shortLabel: "CNC Al 6061" },
  { id: "aluminum_7075_t6", label: "CNC Al 7075-T6 (71.7 GPa)", shortLabel: "CNC Al 7075" },
];

const TOOLHEAD_CONSTRUCTION_OPTIONS: ToolheadConstructionCatalogItem[] = [
  {
    id: "multi_piece_clamshell",
    name: "Multi-Piece Clamshell (StealthBurner / Afterburner)",
    short_name: "Multi-Piece Clamshell",
    body_stiffness_base_n_per_mm: 420,
    torsion_mult: 0.72,
    mass_delta_g: 15,
    notes: "Stacked printed halves + separate shroud; extra bolted seams add compliance and mass.",
  },
  {
    id: "compact_cage",
    name: "Compact Open-Cage Box (XOL / Dragon Burner / A4T)",
    short_name: "Compact Cage",
    body_stiffness_base_n_per_mm: 680,
    torsion_mult: 1.0,
    mass_delta_g: 0,
    notes: "Single-piece boxed truss around the hotend; balanced stiffness-to-weight baseline.",
  },
  {
    id: "plate_sandwich_cf",
    name: "Plate Sandwich + Standoffs (Archetype / VzBot / EVA)",
    short_name: "Plate Sandwich",
    body_stiffness_base_n_per_mm: 950,
    torsion_mult: 1.18,
    mass_delta_g: -12,
    notes: "Front/rear structural plates tied with metal standoffs; high bending stiffness with low mass.",
  },
  {
    id: "extruder_as_chassis",
    name: "Integrated Extruder-as-Chassis (Direct-Bolted Core)",
    short_name: "Extruder-as-Chassis",
    body_stiffness_base_n_per_mm: 1250,
    torsion_mult: 1.32,
    mass_delta_g: -25,
    notes: "Direct-bolted extruder housing acts as the load path, eliminating redundant brackets and overhang.",
  },
  {
    id: "cnc_monoblock",
    name: "CNC Monoblock / Ribbed Billet Chassis",
    short_name: "CNC Monoblock",
    body_stiffness_base_n_per_mm: 2400,
    torsion_mult: 1.75,
    mass_delta_g: 18,
    notes: "One-piece machined ribbed chassis; eliminates inter-part joint slip for maximum tip rigidity.",
  },
  {
    id: "cantilever_bowden",
    name: "Minimalist Bowden / Remote-Drive Mount",
    short_name: "Bowden Mount",
    body_stiffness_base_n_per_mm: 520,
    torsion_mult: 0.85,
    mass_delta_g: -65,
    notes: "Minimalist hotend-only bracket with remote extruder; drastically cuts moving mass.",
  },
];

function inferToolheadConstruction(
  spec: PrinterSpecData,
  sim: SimulationResponse | null
): ToolheadConstructionType {
  if (spec.toolhead.body_construction) return spec.toolhead.body_construction;
  const simConstruction = sim?.geometry?.toolhead_structure?.body_construction;
  if (simConstruction) return simConstruction;
  if (spec.toolhead.extruder?.type === "bowden") return "cantilever_bowden";
  const pid = (spec.toolhead.preset_id || "").toLowerCase();
  if (pid.includes("stealthburner") || pid.includes("afterburner") || pid.includes("heavy_uhf")) {
    return "multi_piece_clamshell";
  }
  if (pid.includes("archetype") || pid.includes("vz") || pid.includes("eva")) {
    return "plate_sandwich_cf";
  }
  if (pid.includes("cnc")) {
    return "cnc_monoblock";
  }
  return "compact_cage";
}

const HOTEND_MOUNT_STYLE_OPTIONS: Array<{
  id: HotendMountStyle;
  label: string;
  shortLabel: string;
}> = [
  { id: "groove_mount", label: "Groove Mount (E3D V6)", shortLabel: "Groove Mount" },
  { id: "rigid_4bolt_m3", label: "4× M3 Rigid Flange", shortLabel: "4× M3 Rigid" },
  { id: "dragon_4bolt_m25", label: "Dragon 4× M2.5 Standoffs", shortLabel: "Dragon 4× M2.5" },
  { id: "chube_6bolt_m3_18pcd", label: "Chube 6× M3 Ti Shell (18mm PCD)", shortLabel: "Chube 6× M3 Ti" },
];

const FALLBACK_HOTEND_PRESETS: HotendPresetItem[] = [
  {
    id: "e3d_v6_groove",
    name: "E3D V6 Groove Mount",
    short_name: "E3D V6",
    mount_style: "groove_mount",
    mass_g: 42,
    overall_length_mm: 62,
    heater_power_w: 40,
    block_material: "aluminum_6061_t6",
    block_mass_g: 12,
    nozzle_material: "brass_c360",
    melt_zone_length_mm: 12,
    heatbreak_material: "stainless_304",
    heatbreak_length_mm: 14,
    heatbreak_wall_mm: 0.45,
    heatbreak_id_mm: 2.0,
    heatsink_thermal_resistance_k_w: 8.5,
    structural_stiffness_n_per_mm: 45,
    max_temp_c: 285,
  },
  {
    id: "e3d_revo_voron",
    name: "E3D Revo Voron (4-Bolt M3)",
    short_name: "Revo Voron",
    mount_style: "rigid_4bolt_m3",
    mass_g: 36,
    overall_length_mm: 52,
    heater_power_w: 40,
    block_material: "brass_c360",
    block_mass_g: 8,
    nozzle_material: "brass_c360",
    melt_zone_length_mm: 13,
    heatbreak_material: "titanium_grade5",
    heatbreak_length_mm: 12,
    heatbreak_wall_mm: 0.35,
    heatbreak_id_mm: 2.0,
    heatsink_thermal_resistance_k_w: 7.2,
    structural_stiffness_n_per_mm: 240,
    max_temp_c: 300,
  },
  {
    id: "phaetus_dragon_hf",
    name: "Phaetus Dragon HF (4× M2.5 Standoffs)",
    short_name: "Dragon HF",
    mount_style: "dragon_4bolt_m25",
    mass_g: 52,
    overall_length_mm: 61,
    heater_power_w: 60,
    block_material: "copper_c110",
    block_mass_g: 16,
    nozzle_material: "brass_c360",
    melt_zone_length_mm: 17,
    heatbreak_material: "titanium_grade5",
    heatbreak_length_mm: 14,
    heatbreak_wall_mm: 0.25,
    heatbreak_id_mm: 2.0,
    heatsink_thermal_resistance_k_w: 6.8,
    structural_stiffness_n_per_mm: 520,
    max_temp_c: 500,
  },
  {
    id: "phaetus_rapido_2_uhf",
    name: "Phaetus Rapido 2 UHF (4× M2.5 Standoffs)",
    short_name: "Rapido 2 UHF",
    mount_style: "dragon_4bolt_m25",
    mass_g: 58,
    overall_length_mm: 68,
    heater_power_w: 115,
    block_material: "copper_c110",
    block_mass_g: 19,
    nozzle_material: "copper_c110",
    melt_zone_length_mm: 27,
    heatbreak_material: "titanium_grade5",
    heatbreak_length_mm: 14,
    heatbreak_wall_mm: 0.25,
    heatbreak_id_mm: 2.0,
    heatsink_thermal_resistance_k_w: 6.5,
    structural_stiffness_n_per_mm: 480,
    max_temp_c: 350,
  },
  {
    id: "vz_goliath",
    name: "VzBot Goliath Long (4× M2.5 Rigid)",
    short_name: "Goliath",
    mount_style: "dragon_4bolt_m25",
    mass_g: 74,
    overall_length_mm: 76,
    heater_power_w: 115,
    block_material: "copper_c110",
    block_mass_g: 28,
    nozzle_material: "copper_c110",
    melt_zone_length_mm: 36,
    heatbreak_material: "titanium_grade5",
    heatbreak_length_mm: 15,
    heatbreak_wall_mm: 0.3,
    heatbreak_id_mm: 2.0,
    heatsink_thermal_resistance_k_w: 5.8,
    structural_stiffness_n_per_mm: 780,
    max_temp_c: 450,
  },
  {
    id: "lukes_lab_chube_air",
    name: "Luke's Lab Chube Air (6× M3 Ti Shell)",
    short_name: "Chube Air",
    mount_style: "chube_6bolt_m3_18pcd",
    mass_g: 86,
    overall_length_mm: 68,
    heater_power_w: 120,
    block_material: "copper_c110",
    block_mass_g: 26,
    nozzle_material: "copper_c110",
    melt_zone_length_mm: 36,
    heatbreak_material: "titanium_grade5",
    heatbreak_length_mm: 15,
    heatbreak_wall_mm: 0.35,
    heatbreak_id_mm: 2.0,
    heatsink_thermal_resistance_k_w: 5.2,
    structural_stiffness_n_per_mm: 2400,
    max_temp_c: 450,
  },
  {
    id: "lukes_lab_chube_conduction",
    name: "Luke's Lab Chube Conduction (6× M3 Ti Shell + CNC Block)",
    short_name: "Chube Conduction",
    mount_style: "chube_6bolt_m3_18pcd",
    mass_g: 98,
    overall_length_mm: 65,
    heater_power_w: 120,
    block_material: "copper_c110",
    block_mass_g: 28,
    nozzle_material: "copper_c110",
    melt_zone_length_mm: 38,
    heatbreak_material: "titanium_grade5",
    heatbreak_length_mm: 14,
    heatbreak_wall_mm: 0.35,
    heatbreak_id_mm: 2.0,
    heatsink_thermal_resistance_k_w: 3.8,
    structural_stiffness_n_per_mm: 3600,
    max_temp_c: 450,
  },
];

function toolheadPartMatShort(id?: string | null): string {
  if (!id) return "ABS";
  return TOOLHEAD_PART_MATERIAL_OPTIONS.find((m) => m.id === id)?.shortLabel ?? id.replace(/_/g, " ");
}

function hotendStiffnessTier(k: number): { label: string; color: string } {
  if (k < 100) return { label: "Floppy", color: "var(--red)" };
  if (k < 350) return { label: "Moderate", color: "var(--amber)" };
  if (k < 1200) return { label: "Rigid", color: "var(--accent)" };
  return { label: "Ultra-Rigid", color: "var(--green)" };
}

const BUS_VOLTAGE_PRESETS = [24, 36, 48, 60];

const LINEAR_RAIL_PROFILES: Array<{ id: string; label: string; shortLabel: string }> = [
  { id: "mgn9_rail", label: "MGN9H (9×6.5mm)", shortLabel: "MGN9H" },
  { id: "mgn12_rail", label: "MGN12H (12×8mm)", shortLabel: "MGN12H" },
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
  const found = LINEAR_RAIL_PROFILES.find((r) => r.id === id);
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
  { id: "C", label: "C Standard", shortLabel: "C" },
  { id: "H", label: "H Long", shortLabel: "H" },
];

const RAIL_BACKER_OPTIONS: Array<{ id: RailBackerMaterialType; label: string; shortLabel: string }> = [
  { id: "none", label: "None", shortLabel: "None" },
  { id: "titanium_grade5", label: "Titanium Gr5 (Anti-Bow)", shortLabel: "Ti Gr5" },
  { id: "steel_4140", label: "Steel 4140 (Anti-Bow)", shortLabel: "Steel 4140" },
];

const BEARING_CLEARANCE_OPTIONS: Array<{ id: BearingClearanceClass; label: string; shortLabel: string }> = [
  { id: "C2", label: "C2 Tight", shortLabel: "C2" },
  { id: "CN", label: "CN Standard", shortLabel: "CN" },
  { id: "C3", label: "C3 Loose", shortLabel: "C3" },
];

const Z_PROBE_OPTIONS: Array<{
  id: ZProbeType;
  label: string;
  shortLabel: string;
  hint: string;
}> = [
  {
    id: "static_eddy",
    label: "Static Eddy (Beacon/Cartographer)",
    shortLabel: "Static Eddy",
    hint: "Rigid direct-bolt carriage mount — avoids Voron Tap's sliding-rail compliance penalty.",
  },
  {
    id: "klicky_inductive",
    label: "Klicky / Inductive",
    shortLabel: "Klicky / Inductive",
    hint: "Rigid carriage mount — avoids Tap's sliding-rail compliance penalty while keeping high nozzle resonance.",
  },
  {
    id: "cnc_tap",
    label: "CNC Voron Tap",
    shortLabel: "CNC Tap",
    hint: "Nozzle-tap MGN9 sliding rail adds extra carriage compliance in the nozzle load path (include probe weight in Toolhead Mass below).",
  },
  {
    id: "printed_tap",
    label: "Printed Voron Tap (soft)",
    shortLabel: "Printed Tap",
    hint: "Printed sliding-rail Tap adds soft compliance in the nozzle load path — sharp resonance drop vs a rigid Beacon/eddy mount (include probe weight in Toolhead Mass below).",
  },
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

export const DesignPanel: React.FC<Props> = ({
  catalog,
  sim,
  selectedPreset,
  onSelectPreset,
  filamentId,
  onFilamentChange,
  spec,
  onSpecChange,
  savedBuildsCount = 0,
  lastAutoSavedAt = null,
  onOpenSavedBuilds,
  toolheadRelPos,
  onToolheadRelPosChange,
  panelMeshN,
  onPanelMeshNChange,
  yamlText,
  onYamlChange,
  onSimulateYaml,
  gcodeText,
  onGcodeChange,
  onRunGcode,
  onRunPartFea,
  onRunStlFea,
  onDownloadPartStep,
  exportingStep,
  onRunTopoOpt,
  onDownloadTopoStl,
  hasTopoResult,
  topoLoading,
  loading,
}) => {
  const [tab, setTab] = useState<"params" | "part_fea" | "yaml" | "gcode">("params");

  // Section jump bar: track which Printer-tab section is at the top of the scroll area.
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const [activeSection, setActiveSection] = useState<string>(DESIGN_SECTIONS[0].id);
  // After a chip click, the clicked section stays highlighted until the panel stops scrolling
  // to it. The lock is released by a pause in scroll events, not a fixed time, so a long
  // smooth scroll cannot outlast it.
  const jumpLockedRef = useRef<boolean>(false);
  const jumpLockTimerRef = useRef<number | null>(null);
  const holdJumpLock = (ms: number) => {
    jumpLockedRef.current = true;
    if (jumpLockTimerRef.current !== null) window.clearTimeout(jumpLockTimerRef.current);
    jumpLockTimerRef.current = window.setTimeout(() => {
      jumpLockedRef.current = false;
      jumpLockTimerRef.current = null;
    }, ms);
  };
  const designReady = Boolean(catalog && spec);
  useEffect(() => {
    const el = scrollRef.current;
    if (!el || tab !== "params" || !designReady) return;
    const measure = () => {
      if (jumpLockedRef.current) {
        holdJumpLock(160);
        return;
      }
      // The last sections are short and can never reach the top; at the bottom, pick the last one.
      if (el.scrollTop + el.clientHeight >= el.scrollHeight - 2 && el.scrollTop > 0) {
        setActiveSection(DESIGN_SECTIONS[DESIGN_SECTIONS.length - 1].id);
        return;
      }
      const line = el.getBoundingClientRect().top + 64;
      let current: string = DESIGN_SECTIONS[0].id;
      for (const sec of DESIGN_SECTIONS) {
        const node = el.querySelector<HTMLElement>(`[data-design-section="${sec.id}"]`);
        if (node && node.getBoundingClientRect().top <= line) current = sec.id;
      }
      setActiveSection(current);
    };
    el.addEventListener("scroll", measure, { passive: true });
    measure();
    return () => {
      el.removeEventListener("scroll", measure);
      if (jumpLockTimerRef.current !== null) window.clearTimeout(jumpLockTimerRef.current);
      jumpLockedRef.current = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tab, designReady]);

  const jumpToSection = (id: string) => {
    const node = scrollRef.current?.querySelector<HTMLElement>(`[data-design-section="${id}"]`);
    if (!node) return;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    setActiveSection(id);
    holdJumpLock(600);
    node.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
  };
  const [showHints, setShowHints] = useState<boolean>(() => loadShowHintsPreference());
  const [feaEngine, setFeaEngine] = useState<SolidFeaEngine>("tet10");
  const [partPreset, setPartPreset] = useState<string>("corner_gusset");
  const [partMat, setPartMat] = useState<string>("pa12_cf_printed");
  const [partFdm, setPartFdm] = useState<boolean>(true);
  const [fx, setFx] = useState<number>(60);
  const [fy, setFy] = useState<number>(0);
  const [fz, setFz] = useState<number>(0);
  const [fixedFace, setFixedFace] = useState<string>("min_z");
  const [loadFace, setLoadFace] = useState<string>("max_z");

  // 3D SIMP Topology Optimization state
  const [targetVolFrac, setTargetVolFrac] = useState<number>(0.45);
  const [topoIterations, setTopoIterations] = useState<number>(18);
  const [preserveBoundaries, setPreserveBoundaries] = useState<boolean>(true);

  // Smooth local drafts for sliders (avoid API spam while dragging)
  const [tensionDraft, setTensionDraft] = useState<number>(spec?.gantry.belt.pretension_n ?? 28);
  const tensionTimerRef = useRef<number | null>(null);

  const [carriageMassDraft, setCarriageMassDraft] = useState<number>(
    Math.round(spec?.gantry.carriage_mass_g ?? 140)
  );
  const carriageMassTimerRef = useRef<number | null>(null);

  const [runCurrentDraft, setRunCurrentDraft] = useState<number>(
    spec?.kinematics.xy_stepper.run_current_a ?? 1.2
  );
  const currentTimerRef = useRef<number | null>(null);

  const [cogYDraft, setCogYDraft] = useState<number>(
    Math.round(Math.abs(spec?.toolhead.cog_offset_mm?.[1] ?? 18))
  );
  const cogYTimerRef = useRef<number | null>(null);

  const [cogZDraft, setCogZDraft] = useState<number>(
    Math.round(Math.abs(spec?.toolhead.cog_offset_mm?.[2] ?? 25))
  );
  const cogZTimerRef = useRef<number | null>(null);

  const [nozzleZDraft, setNozzleZDraft] = useState<number>(
    Math.round(Math.abs(spec?.toolhead.nozzle_offset_mm?.[2] ?? 62))
  );
  const nozzleZTimerRef = useRef<number | null>(null);

  const [hotendStiffnessDraft, setHotendStiffnessDraft] = useState<number>(
    Math.round(spec?.toolhead.hotend.structural_stiffness_n_per_mm ?? 450)
  );
  const hotendStiffnessTimerRef = useRef<number | null>(null);

  const [hotendMassDraft, setHotendMassDraft] = useState<string>(
    String(Math.round(spec?.toolhead.hotend.mass_g ?? 45))
  );
  const hotendMassTimerRef = useRef<number | null>(null);

  const [carriagePlateThickDraft, setCarriagePlateThickDraft] = useState<number>(
    Number((spec?.toolhead.carriage_plate_thickness_mm ?? 5.0).toFixed(1))
  );
  const carriagePlateThickTimerRef = useRef<number | null>(null);

  const [bodyWallThickDraft, setBodyWallThickDraft] = useState<number>(
    Number((spec?.toolhead.body_wall_thickness_mm ?? 4.0).toFixed(1))
  );
  const bodyWallThickTimerRef = useRef<number | null>(null);

  const [hotendMountThickDraft, setHotendMountThickDraft] = useState<number>(
    Number((spec?.toolhead.hotend_mount_thickness_mm ?? 8.0).toFixed(1))
  );
  const hotendMountThickTimerRef = useRef<number | null>(null);

  const [notchDraft, setNotchDraft] = useState<string>(
    spec?.kinematics.notch_filter_hz != null ? String(Math.round(spec.kinematics.notch_filter_hz)) : ""
  );

  const [awdEqBwDraft, setAwdEqBwDraft] = useState<number>(
    Number((spec?.kinematics.awd_equalizer_bw_hz ?? 2.0).toFixed(1))
  );
  const awdEqBwTimerRef = useRef<number | null>(null);

  const [notchDepthDraft, setNotchDepthDraft] = useState<number>(
    Number((spec?.kinematics.notch_depth_db ?? 12.0).toFixed(1))
  );
  const notchDepthTimerRef = useRef<number | null>(null);

  const [notchQDraft, setNotchQDraft] = useState<number>(
    Number((spec?.kinematics.notch_q ?? 1.75).toFixed(2))
  );
  const notchQTimerRef = useRef<number | null>(null);

  useEffect(() => {
    if (spec?.gantry.belt.pretension_n != null) {
      setTensionDraft(spec.gantry.belt.pretension_n);
    }
  }, [spec?.gantry.belt.pretension_n]);

  useEffect(() => {
    if (spec?.gantry.carriage_mass_g != null) {
      setCarriageMassDraft(Math.round(spec.gantry.carriage_mass_g));
    }
  }, [spec?.gantry.carriage_mass_g]);

  useEffect(() => {
    if (spec?.kinematics.xy_stepper) {
      const nextA =
        spec.kinematics.xy_stepper.run_current_a ??
        Math.min(spec.kinematics.xy_stepper.rated_current_a || 1.2, 1.2);
      setRunCurrentDraft(Number(nextA.toFixed(1)));
    }
  }, [
    spec?.kinematics.xy_stepper.run_current_a,
    spec?.kinematics.xy_stepper.rated_current_a,
    spec?.kinematics.xy_stepper.preset_id,
    spec?.kinematics.xy_stepper.name,
  ]);

  useEffect(() => {
    if (spec?.toolhead.cog_offset_mm) {
      setCogYDraft(Math.round(Math.abs(spec.toolhead.cog_offset_mm[1] ?? 18)));
      setCogZDraft(Math.round(Math.abs(spec.toolhead.cog_offset_mm[2] ?? 25)));
    }
  }, [spec?.toolhead.cog_offset_mm?.[1], spec?.toolhead.cog_offset_mm?.[2]]);

  useEffect(() => {
    if (spec?.toolhead.nozzle_offset_mm) {
      setNozzleZDraft(Math.round(Math.abs(spec.toolhead.nozzle_offset_mm[2] ?? 62)));
    } else if (spec) {
      setNozzleZDraft(62);
    }
  }, [spec?.toolhead.nozzle_offset_mm?.[2]]);

  useEffect(() => {
    if (spec) {
      setHotendStiffnessDraft(
        Math.round(spec.toolhead.hotend.structural_stiffness_n_per_mm ?? 450)
      );
    }
  }, [spec?.toolhead.hotend.structural_stiffness_n_per_mm]);

  useEffect(() => {
    if (spec) {
      setHotendMassDraft(String(Math.round(spec.toolhead.hotend.mass_g ?? 45)));
    }
  }, [spec?.toolhead.hotend.mass_g]);

  useEffect(() => {
    if (spec) {
      setCarriagePlateThickDraft(
        Number((spec.toolhead.carriage_plate_thickness_mm ?? 5.0).toFixed(1))
      );
    }
  }, [spec?.toolhead.carriage_plate_thickness_mm]);

  useEffect(() => {
    if (spec) {
      setBodyWallThickDraft(Number((spec.toolhead.body_wall_thickness_mm ?? 4.0).toFixed(1)));
    }
  }, [spec?.toolhead.body_wall_thickness_mm]);

  useEffect(() => {
    if (spec) {
      setHotendMountThickDraft(
        Number((spec.toolhead.hotend_mount_thickness_mm ?? 8.0).toFixed(1))
      );
    }
  }, [spec?.toolhead.hotend_mount_thickness_mm]);

  useEffect(() => {
    if (spec) {
      const nf = spec.kinematics.notch_filter_hz;
      setNotchDraft(nf != null && nf > 0 ? String(Math.round(nf)) : "");
    }
  }, [spec?.kinematics.notch_filter_hz]);

  useEffect(() => {
    if (spec) {
      setAwdEqBwDraft(Number((spec.kinematics.awd_equalizer_bw_hz ?? 2.0).toFixed(1)));
    }
  }, [spec?.kinematics.awd_equalizer_bw_hz]);

  useEffect(() => {
    if (spec) {
      setNotchDepthDraft(Number((spec.kinematics.notch_depth_db ?? 12.0).toFixed(1)));
    }
  }, [spec?.kinematics.notch_depth_db]);

  useEffect(() => {
    if (spec) {
      setNotchQDraft(Number((spec.kinematics.notch_q ?? 1.75).toFixed(2)));
    }
  }, [spec?.kinematics.notch_q]);

  useEffect(() => {
    return () => {
      if (tensionTimerRef.current != null) window.clearTimeout(tensionTimerRef.current);
      if (carriageMassTimerRef.current != null) window.clearTimeout(carriageMassTimerRef.current);
      if (currentTimerRef.current != null) window.clearTimeout(currentTimerRef.current);
      if (cogYTimerRef.current != null) window.clearTimeout(cogYTimerRef.current);
      if (cogZTimerRef.current != null) window.clearTimeout(cogZTimerRef.current);
      if (nozzleZTimerRef.current != null) window.clearTimeout(nozzleZTimerRef.current);
      if (hotendStiffnessTimerRef.current != null) window.clearTimeout(hotendStiffnessTimerRef.current);
      if (hotendMassTimerRef.current != null) window.clearTimeout(hotendMassTimerRef.current);
      if (carriagePlateThickTimerRef.current != null)
        window.clearTimeout(carriagePlateThickTimerRef.current);
      if (bodyWallThickTimerRef.current != null) window.clearTimeout(bodyWallThickTimerRef.current);
      if (hotendMountThickTimerRef.current != null)
        window.clearTimeout(hotendMountThickTimerRef.current);
      if (awdEqBwTimerRef.current != null) window.clearTimeout(awdEqBwTimerRef.current);
      if (notchDepthTimerRef.current != null) window.clearTimeout(notchDepthTimerRef.current);
      if (notchQTimerRef.current != null) window.clearTimeout(notchQTimerRef.current);
    };
  }, []);

  if (!spec || !catalog) {
    return (
      <aside className="sidebar">
        <div className="sidebar-scroll">Loading catalog...</div>
      </aside>
    );
  }

  const profileLabel = (id: string) => {
    const found = catalog.profiles.find((p) => p.id === id);
    return found ? found.name.replace(/\s*\([^)]*\)\s*/g, "").trim() : id;
  };

  const materialLabel = (id: string) => {
    const found = catalog.materials.find((m) => m.id === id);
    return found ? found.name.replace(/\s*\([^)]*\)\s*/g, "").trim() : id;
  };

  const update = (mutator: (draft: PrinterSpecData) => void, changeLabel?: string) => {
    const clone: PrinterSpecData = JSON.parse(JSON.stringify(spec));
    mutator(clone);
    onSpecChange(clone, changeLabel);
  };

  const updatePositiveNum = (
    raw: string,
    mutator: (draft: PrinterSpecData, val: number) => void,
    changeLabelBuilder: (val: number) => string
  ) => {
    const val = Number(raw);
    if (!Number.isFinite(val) || val <= 0) return;
    update((d) => mutator(d, val), changeLabelBuilder(val));
  };

  const beltTensionMax = spec
    ? Math.max(
        getBeltTensionSliderMax(spec.gantry.belt.type),
        Math.min(250, Math.round(spec.gantry.belt.pretension_n))
      )
    : 120;

  const commitBeltTension = (rawVal: number) => {
    const val = Math.max(15, Math.min(beltTensionMax, Math.round(rawVal)));
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

  const commitRunCurrent = (rawVal: number) => {
    const val = Math.max(0.4, Math.min(3.2, Math.round(rawVal * 10) / 10));
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

  const specCogYAbs = Math.round(Math.abs(spec.toolhead.cog_offset_mm?.[1] ?? 18));
  const specCogZAbs = Math.round(Math.abs(spec.toolhead.cog_offset_mm?.[2] ?? 25));
  const specNozzleZAbs = Math.round(Math.abs(spec.toolhead.nozzle_offset_mm?.[2] ?? 62));

  const commitCogY = (rawVal: number) => {
    const val = Math.max(0, Math.min(250, Math.round(rawVal)));
    if (cogYTimerRef.current != null) {
      window.clearTimeout(cogYTimerRef.current);
      cogYTimerRef.current = null;
    }
    if (val === specCogYAbs) return;
    update(
      (d) => {
        const cur = d.toolhead.cog_offset_mm || [0, -18, -25];
        d.toolhead.cog_offset_mm = [cur[0], val === 0 ? 0 : -val, cur[2]];
        d.toolhead.preset_id = "custom";
      },
      `CoG Forward Y: ${specCogYAbs}mm → ${val}mm`
    );
  };

  const handleCogYSliderChange = (nextVal: number) => {
    setCogYDraft(nextVal);
    if (cogYTimerRef.current != null) window.clearTimeout(cogYTimerRef.current);
    cogYTimerRef.current = window.setTimeout(() => commitCogY(nextVal), 220);
  };

  const commitCogZ = (rawVal: number) => {
    const val = Math.max(0, Math.min(250, Math.round(rawVal)));
    if (cogZTimerRef.current != null) {
      window.clearTimeout(cogZTimerRef.current);
      cogZTimerRef.current = null;
    }
    if (val === specCogZAbs) return;
    update(
      (d) => {
        const cur = d.toolhead.cog_offset_mm || [0, -18, -25];
        d.toolhead.cog_offset_mm = [cur[0], cur[1], val === 0 ? 0 : -val];
        d.toolhead.preset_id = "custom";
      },
      `CoG Drop Z: ${specCogZAbs}mm → ${val}mm`
    );
  };

  const handleCogZSliderChange = (nextVal: number) => {
    setCogZDraft(nextVal);
    if (cogZTimerRef.current != null) window.clearTimeout(cogZTimerRef.current);
    cogZTimerRef.current = window.setTimeout(() => commitCogZ(nextVal), 220);
  };

  const commitNozzleZ = (rawVal: number) => {
    const val = Math.max(0, Math.min(250, Math.round(rawVal)));
    if (nozzleZTimerRef.current != null) {
      window.clearTimeout(nozzleZTimerRef.current);
      nozzleZTimerRef.current = null;
    }
    if (val === specNozzleZAbs) return;
    update(
      (d) => {
        const cur = d.toolhead.nozzle_offset_mm || [0, -22, -62];
        d.toolhead.nozzle_offset_mm = [cur[0], cur[1], val === 0 ? 0 : -val];
        d.toolhead.preset_id = "custom";
      },
      `Nozzle Drop Z: ${specNozzleZAbs}mm → ${val}mm`
    );
  };

  const handleNozzleZSliderChange = (nextVal: number) => {
    setNozzleZDraft(nextVal);
    if (nozzleZTimerRef.current != null) window.clearTimeout(nozzleZTimerRef.current);
    nozzleZTimerRef.current = window.setTimeout(() => commitNozzleZ(nextVal), 220);
  };

  const specHotendStiffness = Math.round(spec.toolhead.hotend.structural_stiffness_n_per_mm ?? 450);
  const commitHotendStiffness = (rawVal: number) => {
    const val = Math.max(25, Math.min(4000, Math.round(rawVal)));
    if (hotendStiffnessTimerRef.current != null) {
      window.clearTimeout(hotendStiffnessTimerRef.current);
      hotendStiffnessTimerRef.current = null;
    }
    if (val === specHotendStiffness) return;
    update(
      (d) => {
        d.toolhead.hotend.structural_stiffness_n_per_mm = val;
        d.toolhead.hotend.preset_id = null;
      },
      `Hotend Stiffness: ${specHotendStiffness} → ${val} N/mm`
    );
  };

  const handleHotendStiffnessSliderChange = (nextVal: number) => {
    setHotendStiffnessDraft(nextVal);
    if (hotendStiffnessTimerRef.current != null) {
      window.clearTimeout(hotendStiffnessTimerRef.current);
    }
    hotendStiffnessTimerRef.current = window.setTimeout(
      () => commitHotendStiffness(nextVal),
      220
    );
  };

  const specHotendMassG = Math.round(spec.toolhead.hotend.mass_g ?? 45);
  const assembledToolheadMassG = Math.round(spec.toolhead.total_mass_g + specHotendMassG);
  const commitHotendMass = (rawVal: number) => {
    const val = Math.max(5, Math.min(400, Math.round(rawVal)));
    if (hotendMassTimerRef.current != null) {
      window.clearTimeout(hotendMassTimerRef.current);
      hotendMassTimerRef.current = null;
    }
    if (val === specHotendMassG) return;
    update(
      (d) => {
        d.toolhead.hotend.mass_g = val;
      },
      `Hotend Weight: ${specHotendMassG}g → ${val}g`
    );
  };

  const specCarriagePlateThickMm = Number(
    (spec.toolhead.carriage_plate_thickness_mm ?? 5.0).toFixed(1)
  );
  const commitCarriagePlateThick = (rawVal: number) => {
    const val = Math.max(2.0, Math.min(15.0, Math.round(rawVal * 2) / 2));
    if (carriagePlateThickTimerRef.current != null) {
      window.clearTimeout(carriagePlateThickTimerRef.current);
      carriagePlateThickTimerRef.current = null;
    }
    if (Math.abs(val - specCarriagePlateThickMm) < 0.05) return;
    update(
      (d) => {
        d.toolhead.carriage_plate_thickness_mm = val;
        d.toolhead.preset_id = "custom";
      },
      `X-Carriage Plate Thickness: ${specCarriagePlateThickMm.toFixed(1)}mm → ${val.toFixed(1)}mm`
    );
  };
  const handleCarriagePlateThickSliderChange = (nextVal: number) => {
    const rounded = Math.round(nextVal * 2) / 2;
    setCarriagePlateThickDraft(rounded);
    if (carriagePlateThickTimerRef.current != null) {
      window.clearTimeout(carriagePlateThickTimerRef.current);
    }
    carriagePlateThickTimerRef.current = window.setTimeout(
      () => commitCarriagePlateThick(rounded),
      220
    );
  };

  const specBodyWallThickMm = Number((spec.toolhead.body_wall_thickness_mm ?? 4.0).toFixed(1));
  const commitBodyWallThick = (rawVal: number) => {
    const val = Math.max(2.0, Math.min(12.0, Math.round(rawVal * 2) / 2));
    if (bodyWallThickTimerRef.current != null) {
      window.clearTimeout(bodyWallThickTimerRef.current);
      bodyWallThickTimerRef.current = null;
    }
    if (Math.abs(val - specBodyWallThickMm) < 0.05) return;
    update(
      (d) => {
        d.toolhead.body_wall_thickness_mm = val;
        d.toolhead.preset_id = "custom";
      },
      `Toolhead Body Wall: ${specBodyWallThickMm.toFixed(1)}mm → ${val.toFixed(1)}mm`
    );
  };
  const handleBodyWallThickSliderChange = (nextVal: number) => {
    const rounded = Math.round(nextVal * 2) / 2;
    setBodyWallThickDraft(rounded);
    if (bodyWallThickTimerRef.current != null) {
      window.clearTimeout(bodyWallThickTimerRef.current);
    }
    bodyWallThickTimerRef.current = window.setTimeout(() => commitBodyWallThick(rounded), 220);
  };

  const specHotendMountThickMm = Number(
    (spec.toolhead.hotend_mount_thickness_mm ?? 8.0).toFixed(1)
  );
  const commitHotendMountThick = (rawVal: number) => {
    const val = Math.max(3.0, Math.min(20.0, Math.round(rawVal * 2) / 2));
    if (hotendMountThickTimerRef.current != null) {
      window.clearTimeout(hotendMountThickTimerRef.current);
      hotendMountThickTimerRef.current = null;
    }
    if (Math.abs(val - specHotendMountThickMm) < 0.05) return;
    update(
      (d) => {
        d.toolhead.hotend_mount_thickness_mm = val;
        d.toolhead.preset_id = "custom";
      },
      `Hotend Mount Thickness: ${specHotendMountThickMm.toFixed(1)}mm → ${val.toFixed(1)}mm`
    );
  };
  const handleHotendMountThickSliderChange = (nextVal: number) => {
    const rounded = Math.round(nextVal * 2) / 2;
    setHotendMountThickDraft(rounded);
    if (hotendMountThickTimerRef.current != null) {
      window.clearTimeout(hotendMountThickTimerRef.current);
    }
    hotendMountThickTimerRef.current = window.setTimeout(
      () => commitHotendMountThick(rounded),
      220
    );
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
  const currentIdlerShaftMm: number = spec.gantry.belt.idler_shaft_diameter_mm ?? 5;
  const currentBearingClass: BearingClearanceClass = spec.gantry.belt.bearing_clearance_class ?? "CN";
  const currentDriveType: DriveType = spec.kinematics.drive_type ?? "open_loop_stepper";
  const currentBeltRouting: BeltRoutingType = spec.gantry.belt_routing ?? "stock_corexy";
  const currentZProbe: ZProbeType = spec.toolhead.z_probe_type ?? "static_eddy";

  const beltRoutingCatalogEntry = catalog.belt_routings?.find((r) => r.id === currentBeltRouting);
  const beltRoutingOptionMeta =
    BELT_ROUTING_OPTIONS.find((o) => o.id === currentBeltRouting) ?? BELT_ROUTING_OPTIONS[0];
  const zProbeCatalogEntry = catalog.z_probes?.find((p) => p.id === currentZProbe);
  const zProbeOptionMeta =
    Z_PROBE_OPTIONS.find((o) => o.id === currentZProbe) ?? Z_PROBE_OPTIONS[0];

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
  const beltMaxWorkingN = sim?.kinematics.belt_max_working_tension_n;
  const beltOvertension = Boolean(sim?.kinematics.belt_overtension_risk);
  const shaftBendingRisk = Boolean(sim?.kinematics.shaft_bending_risk);

  const gantryBrackets: GantryBracketInfo[] =
    catalog.gantry_brackets && catalog.gantry_brackets.length > 0
      ? catalog.gantry_brackets
      : FALLBACK_GANTRY_BRACKETS;
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
  const bracketDeltaShortLabel = customBracketOverride
    ? "CAD"
    : selectedBracket.cnc_machined
      ? "Al"
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
  const currentXRailProfile = spec.gantry.x_rail_profile || "mgn12_rail";
  const currentYRailProfile = spec.gantry.y_rail_profile || "mgn9_rail";
  const xRailProfileOptions = LINEAR_RAIL_PROFILES.some((r) => r.id === currentXRailProfile)
    ? LINEAR_RAIL_PROFILES
    : [
        ...LINEAR_RAIL_PROFILES,
        { id: currentXRailProfile, label: profileLabel(currentXRailProfile), shortLabel: railProfileShort(currentXRailProfile) },
      ];
  const yRailProfileOptions = LINEAR_RAIL_PROFILES.some((r) => r.id === currentYRailProfile)
    ? LINEAR_RAIL_PROFILES
    : [
        ...LINEAR_RAIL_PROFILES,
        { id: currentYRailProfile, label: profileLabel(currentYRailProfile), shortLabel: railProfileShort(currentYRailProfile) },
      ];

  const partFeaPrintedMaterials = (() => {
    const fromCatalog = catalog.materials
      .filter((m) => PART_FEA_PRINTED_IDS.has(m.id))
      .map((m) => ({ id: m.id, name: m.name }));
    const existingIds = new Set(fromCatalog.map((m) => m.id));
    const extras = gantryBrackets
      .filter((b) => PART_FEA_PRINTED_IDS.has(b.id) && !existingIds.has(b.id))
      .map((b) => ({ id: b.id, name: b.name }));
    return [...fromCatalog, ...extras];
  })();
  const partFeaMetalMaterials = (() => {
    const fromCatalog = catalog.materials
      .filter((m) => PART_FEA_METAL_IDS.has(m.id))
      .map((m) => ({ id: m.id, name: m.name }));
    const existingIds = new Set(fromCatalog.map((m) => m.id));
    const extras = gantryBrackets
      .filter((b) => PART_FEA_METAL_IDS.has(b.id) && !existingIds.has(b.id))
      .map((b) => ({ id: b.id, name: b.name }));
    return [...fromCatalog, ...extras];
  })();

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

  const toolheadPresets = catalog.toolhead_presets ?? [];
  const toolheadConstructions: ToolheadConstructionCatalogItem[] =
    catalog.toolhead_constructions && catalog.toolhead_constructions.length > 0
      ? catalog.toolhead_constructions
      : TOOLHEAD_CONSTRUCTION_OPTIONS;
  const currentBodyConstruction: ToolheadConstructionType = inferToolheadConstruction(spec, sim);
  const selectedBodyConstruction: ToolheadConstructionCatalogItem =
    toolheadConstructions.find((c) => c.id === currentBodyConstruction) ??
    TOOLHEAD_CONSTRUCTION_OPTIONS[1];
  const currentMountMaterial: ToolheadMountMaterial =
    spec.toolhead.mount_material ?? "abs_printed";
  const currentCarriageMountMat: string =
    spec.toolhead.carriage_mount_material || currentMountMaterial;
  const currentBodyMat: string =
    spec.toolhead.body_material || currentMountMaterial;
  const currentHotendMountMat: string =
    spec.toolhead.hotend_mount_material || currentBodyMat;

  const hotendPresets: HotendPresetItem[] =
    catalog.hotend_presets && catalog.hotend_presets.length > 0
      ? catalog.hotend_presets
      : FALLBACK_HOTEND_PRESETS;
  const currentHotendMountStyle: HotendMountStyle =
    spec.toolhead.hotend.mount_style ?? "rigid_4bolt_m3";
  const matchedHotendPreset = hotendPresets.find(
    (hp) => hp.id === spec.toolhead.hotend.preset_id
  );
  const selectedHotendPresetVal = matchedHotendPreset ? matchedHotendPreset.id : "__custom__";
  const hotendTier = hotendStiffnessTier(hotendStiffnessDraft);

  const matchedToolheadPreset = toolheadPresets.find(
    (tp) =>
      tp.preset_id === spec.toolhead.preset_id &&
      Math.abs(tp.total_mass_g - spec.toolhead.total_mass_g) < 0.5 &&
      Math.round(Math.abs(tp.cog_offset_mm[1])) === specCogYAbs &&
      Math.round(Math.abs(tp.cog_offset_mm[2])) === specCogZAbs &&
      Math.round(Math.abs(tp.nozzle_offset_mm[2])) === specNozzleZAbs &&
      tp.mount_material === currentMountMaterial &&
      (tp.z_probe_type ?? "static_eddy") === currentZProbe &&
      (!tp.body_construction ||
        spec.toolhead.body_construction == null ||
        tp.body_construction === spec.toolhead.body_construction)
  );
  const selectedToolheadPresetVal = matchedToolheadPreset
    ? matchedToolheadPreset.preset_id
    : "__custom__";

  return (
    <aside className="sidebar">
      {/* Preset & filament apply to every tab, so they sit above the tabs, once. */}
      <div className="design-header">
        <div className="design-header-row">
          <label htmlFor="design-preset-select">Printer</label>
          <select
            id="design-preset-select"
            value={selectedPreset}
            onChange={(e) => onSelectPreset(e.target.value)}
          >
            {!selectedPreset && <option value="">Custom / Saved Configuration</option>}
            {catalog.presets.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name} ({p.architecture})
              </option>
            ))}
          </select>
        </div>
        <div className="design-header-row">
          <label htmlFor="design-filament-select" title="Filament material (sets rheology and glass transition)">
            Filament
          </label>
          <select
            id="design-filament-select"
            value={filamentId}
            onChange={(e) => onFilamentChange(e.target.value)}
          >
            {catalog.filaments.map((f) => (
              <option key={f.id} value={f.id}>
                {f.name} ({f.print_temp_c}°C)
              </option>
            ))}
          </select>
        </div>
        {onOpenSavedBuilds && (
          <div className="session-save-strip">
            <span
              className="session-autosave-pill"
              title={
                lastAutoSavedAt
                  ? `Last auto-saved at ${new Date(lastAutoSavedAt).toLocaleTimeString()} — your settings persist across page refreshes`
                  : "Active settings auto-save in this browser across page refreshes"
              }
            >
              <span className="saved-autosave-dot" />
              <span>Auto-saved</span>
            </span>
            <div className="session-save-btns">
              <button
                type="button"
                className="btn saved-builds-btn session-mini-btn"
                onClick={onOpenSavedBuilds}
              >
                {savedBuildsCount > 0 ? `Save / Load (${savedBuildsCount})` : "Save / Load"}
              </button>
            </div>
          </div>
        )}
      </div>

      <div className="tabs" role="tablist" aria-label="Design inputs" onKeyDown={onTabListKeyDown}>
        <button role="tab"
          aria-selected={tab === "params"}
          tabIndex={tab === "params" ? 0 : -1}
          className={`tab-btn ${tab === "params" ? "active" : ""}`} onClick={() => setTab("params")}>
          Printer
        </button>
        <button role="tab"
          aria-selected={tab === "part_fea"}
          tabIndex={tab === "part_fea" ? 0 : -1}
          className={`tab-btn ${tab === "part_fea" ? "active" : ""}`} onClick={() => setTab("part_fea")}>
          Parts &amp; Topo
        </button>
        <button role="tab"
          aria-selected={tab === "yaml"}
          tabIndex={tab === "yaml" ? 0 : -1}
          className={`tab-btn ${tab === "yaml" ? "active" : ""}`} onClick={() => setTab("yaml")}>
          YAML
        </button>
        <button role="tab"
          aria-selected={tab === "gcode"}
          tabIndex={tab === "gcode" ? 0 : -1}
          className={`tab-btn ${tab === "gcode" ? "active" : ""}`} onClick={() => setTab("gcode")}>
          G-code
        </button>
      </div>

      <div className="sidebar-scroll" ref={scrollRef}>
        {tab === "params" && (
          <div className="hints-toggle-bar">
            <span className="hints-toggle-label">What each control affects</span>
            <button
              type="button"
              className={`hints-pill ${showHints ? "active" : ""}`}
 aria-pressed={showHints}
              onClick={() =>
                setShowHints((v) => {
                  const next = !v;
                  saveShowHintsPreference(next);
                  return next;
                })
              }
              title="Toggle plain-English explanations on printer controls"
            >
              Plain-English Hints: {showHints ? "ON" : "OFF"}
            </button>
          </div>
        )}

        {tab === "params" && (
          <nav className="section-jump" aria-label="Jump to section">
            {DESIGN_SECTIONS.map((sec) => (
              <button
                key={sec.id}
                type="button"
                className={`chip-btn ${activeSection === sec.id ? "active" : ""}`}
                aria-current={activeSection === sec.id ? "true" : undefined}
                onClick={() => jumpToSection(sec.id)}
              >
                {sec.label}
              </button>
            ))}
          </nav>
        )}

        {tab === "params" && (
          <>
            <DesignFrameSection
              spec={spec}
              catalog={catalog}
              sim={sim}
              showHints={showHints}
              update={update}
              updatePositiveNum={updatePositiveNum}
              profileLabel={profileLabel}
              materialLabel={materialLabel}
            />

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
                        update(
                          (d) => {
                            d.gantry.drive_mode = "2wd";
                          },
                          "Drive Mode: AWD (4-Motor) → 2WD (2-Motor)"
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
                        update(
                          (d) => {
                            d.gantry.drive_mode = "awd";
                          },
                          "Drive Mode: 2WD → AWD (4-Motor)"
                        );
                      }}
                    >
                      AWD CoreXY
                      <span className="engine-toggle-sub">4 Motors</span>
                    </button>
                  </div>
                  {showHints && (
                    <div className="hint-line">
                      AWD adds 2 front XY motors for 2× torque &amp; half belt span; double-shear
                      bearings prevent pulley shaft bending under high tension.
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
                            <optgroup label="Carbon Fiber & Titanium Tubes">
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
                      update(
                        (d) => {
                          d.gantry.belt.type = nextVal;
                          d.gantry.belt.pitch_mm = defaults.pitch_mm;
                          d.gantry.belt.core = defaults.core;
                        },
                        `Belt: ${BELT_LABELS[spec.gantry.belt.type] ?? spec.gantry.belt.type} → ${BELT_LABELS[nextVal]}`
                      );
                    }}
                  >
                    <optgroup label="2GT — Proven on 3D Printers (2 mm pitch)">
                      <option value="gt2_6mm">{BELT_LABELS.gt2_6mm}</option>
                      <option value="gt2_9mm">{BELT_LABELS.gt2_9mm}</option>
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
                      update(
                        (d) => {
                          d.gantry.belt.core = nextVal;
                        },
                        `Belt Core: ${spec.gantry.belt.core} → ${nextVal}`
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
                    min={15}
                    max={beltTensionMax}
                    step={1}
                    value={Math.max(15, Math.min(beltTensionMax, tensionDraft))}
                    onChange={(e) => handleTensionSliderChange(Number(e.target.value))}
                    onPointerUp={() => commitBeltTension(tensionDraft)}
                    onKeyUp={() => commitBeltTension(tensionDraft)}
                  />
                  <input aria-label="Belt Tension (N) value"
                    type="number"
                    className="tension-num-input"
                    min={15}
                    max={beltTensionMax}
                    step={1}
                    value={tensionDraft}
                    onChange={(e) => {
                      const v = Number(e.target.value);
                      if (!Number.isFinite(v)) return;
                      setTensionDraft(v);
                      if (v >= 15 && v <= beltTensionMax) {
                        commitBeltTension(v);
                      }
                    }}
                    onBlur={() => commitBeltTension(tensionDraft)}
                  />
                </div>
                {(beltOvertension || shaftBendingRisk || showHints) && (
                  <div className={`hint-line ${beltOvertension ? "danger" : shaftBendingRisk ? "warn" : ""}`}>
                    {beltOvertension
                      ? `Belt overtensioned — exceeds ${
                          beltMaxWorkingN != null ? `${Math.round(beltMaxWorkingN)} N ` : ""
                        }working tension rating.`
                      : shaftBendingRisk
                        ? "High tension (>45 N) without double-shear bearings risks shaft bending."
                        : "High tension >45 N needs 9mm+ belts & double-shear bearings."}
                  </div>
                )}
              </div>

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
                    <span>{xyStepper.inductance_mh.toFixed(1)} mH</span>
                    <span>•</span>
                    <span>{ratedCurrentA.toFixed(1)}A rated</span>
                    <span>•</span>
                    <span>{xyStepper.steps_per_rev === 400 ? "0.9°" : "1.8°"}</span>
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
                      max={3.2}
                      step={0.1}
                      value={Math.max(0.4, Math.min(3.2, runCurrentDraft))}
                      onChange={(e) => handleRunCurrentSliderChange(Number(e.target.value))}
                      onPointerUp={() => commitRunCurrent(runCurrentDraft)}
                      onKeyUp={() => commitRunCurrent(runCurrentDraft)}
                    />
                    <input aria-label="Run Current (Amps) value"
                      type="number"
                      className="tension-num-input"
                      min={0.4}
                      max={3.2}
                      step={0.1}
                      value={runCurrentDraft}
                      onChange={(e) => {
                        const v = Number(e.target.value);
                        if (!Number.isFinite(v)) return;
                        setRunCurrentDraft(v);
                        if (v >= 0.4 && v <= 3.2) {
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
                        update(
                          (d) => {
                            d.gantry.bracket_material = nextId;
                          },
                          `Gantry Brackets: ${selectedBracket.short_name} → ${nextItem.short_name}`
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
                        above ~45 N belt tension in heated chambers; annealed PPA-CF/PPS-CF or CNC
                        aluminum prevent creep &amp; sharpen resonance.
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
                          update(
                            (d) => {
                              d.gantry.rail_preload = nextVal;
                            },
                            `Rail Preload: ${currentRailPreload} → ${nextVal}`
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
                          update(
                            (d) => {
                              d.gantry.rail_quality = nextVal;
                            },
                            `Rail Quality: ${prevLbl} → ${nextLbl}`
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

                  <div className="field-grid-2">
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
                          update(
                            (d) => {
                              d.gantry.belt_routing = nextVal;
                            },
                            `Belt Path: ${prevLbl} → ${nextLbl}`
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
                          update(
                            (d) => {
                              d.gantry.belt.bearing_clearance_class = nextVal;
                            },
                            `Bearing Play: ${prevLbl} → ${nextLbl}`
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

            <div className="section-card" data-design-section="toolhead">
              <div className="section-title">
                <span>Toolhead CoG, Nozzle &amp; Mount</span>
                <span
                  className="sub-block-tag"
                  style={{ fontVariantNumeric: "tabular-nums" }}
                  title="Toolhead Mass (excl. hotend) + Hotend Weight"
                >
                  {assembledToolheadMassG}g total
                </span>
              </div>

              {showHints && (
                <div className="hint-line">
                  We care about vibration AT THE NOZZLE TIP — lowering CoG &amp; nozzle drop and stiffening
                  the mount stops the X-beam from twisting like a lever arm during rapid Y reversals.
                </div>
              )}

              <div className="field-row">
                <label htmlFor="dp-toolhead-preset-35">Toolhead Preset</label>
                <select id="dp-toolhead-preset-35"
                  value={selectedToolheadPresetVal}
                  onChange={(e) => {
                    const nextId = e.target.value;
                    const chosen = toolheadPresets.find((tp) => tp.preset_id === nextId);
                    if (!chosen) return;
                    const prevName = spec.toolhead.name;
                    update(
                      (d) => {
                        d.toolhead.preset_id = chosen.preset_id;
                        d.toolhead.name = chosen.name;
                        d.toolhead.total_mass_g = chosen.total_mass_g;
                        d.toolhead.cog_offset_mm = [...chosen.cog_offset_mm];
                        d.toolhead.nozzle_offset_mm = [...chosen.nozzle_offset_mm];
                        d.toolhead.mount_material = chosen.mount_material;
                        d.toolhead.carriage_mount_material = chosen.mount_material;
                        d.toolhead.body_material = chosen.mount_material;
                        d.toolhead.hotend_mount_material = chosen.mount_material;
                        if (chosen.body_construction != null) {
                          d.toolhead.body_construction = chosen.body_construction;
                        }
                        d.toolhead.z_probe_type = chosen.z_probe_type ?? "static_eddy";
                        if (chosen.internal_stiffness_n_per_mm != null) {
                          d.toolhead.internal_stiffness_n_per_mm =
                            chosen.internal_stiffness_n_per_mm;
                        }
                      },
                      `Toolhead Preset: ${prevName} → ${chosen.name}`
                    );
                  }}
                >
                  {selectedToolheadPresetVal === "__custom__" && (
                    <option value="__custom__">
                      {spec.toolhead.name || "Custom Toolhead"} (Custom)
                    </option>
                  )}
                  {toolheadPresets.map((tp) => (
                    <option key={tp.preset_id} value={tp.preset_id}>
                      {tp.name} ({tp.total_mass_g}g)
                    </option>
                  ))}
                </select>
              </div>

              <div className="field-row">
                <label htmlFor="dp-z-probe-carriage-interface-36">
                  <span>Z-Probe / Carriage Interface</span>
                  <span
                    style={{
                      color:
                        currentZProbe === "printed_tap" || currentZProbe === "cnc_tap"
                          ? "var(--amber)"
                          : "var(--accent)",
                      fontWeight: 500,
                    }}
                  >
                    {zProbeOptionMeta.shortLabel}
                  </span>
                </label>
                <select id="dp-z-probe-carriage-interface-36"
                  value={currentZProbe}
                  onChange={(e) => {
                    const nextVal = e.target.value as ZProbeType;
                    const prevLbl =
                      Z_PROBE_OPTIONS.find((o) => o.id === currentZProbe)?.shortLabel ??
                      currentZProbe;
                    const nextLbl =
                      Z_PROBE_OPTIONS.find((o) => o.id === nextVal)?.shortLabel ?? nextVal;
                    update(
                      (d) => {
                        d.toolhead.z_probe_type = nextVal;
                      },
                      `Z-Probe: ${prevLbl} → ${nextLbl}`
                    );
                  }}
                >
                  {Z_PROBE_OPTIONS.map((opt) => {
                    const catItem = catalog.z_probes?.find((p) => p.id === opt.id);
                    return (
                      <option key={opt.id} value={opt.id}>
                        {catItem?.name ?? opt.label}
                      </option>
                    );
                  })}
                </select>
                {showHints && (
                  <div
                    className={`hint-line ${
                      currentZProbe === "printed_tap" || currentZProbe === "cnc_tap" ? "warn" : ""
                    }`}
                  >
                    {zProbeCatalogEntry?.notes || zProbeOptionMeta.hint}
                  </div>
                )}
              </div>

              <div className="field-row">
                <label htmlFor="dp-toolhead-mass-excl-hotend-g-37">
                  <span>Toolhead Mass, excl. Hotend (g)</span>
                  <span
                    className="sub-block-tag"
                    style={{ color: "var(--accent)", fontVariantNumeric: "tabular-nums" }}
                    title="Assembled toolhead weight: Toolhead Mass + Hotend Weight"
                  >
                    + Hotend = {assembledToolheadMassG}g
                  </span>
                </label>
                <input id="dp-toolhead-mass-excl-hotend-g-37"
                  type="number"
                  min={50}
                  max={1200}
                  step={5}
                  value={spec.toolhead.total_mass_g}
                  onChange={(e) =>
                    updatePositiveNum(
                      e.target.value,
                      (d, v) => {
                        d.toolhead.total_mass_g = v;
                        d.toolhead.preset_id = "custom";
                      },
                      (v) => `Toolhead Mass: ${spec.toolhead.total_mass_g}g → ${v}g`
                    )
                  }
                />
                {showHints && (
                  <div className="hint-line">
                    Weight of body, extruder, fans &amp; probe WITHOUT hotend — Hotend Weight (
                    {specHotendMassG}g) is added separately below.
                  </div>
                )}
              </div>

              <div className="slider-row">
                <label htmlFor="dp-nozzle-drop-z-mm-38">
                  <span>Nozzle Drop Z (mm)</span>
                  <span className="tension-readout">
                    <strong style={{ color: "#fbbf24" }}>{nozzleZDraft} mm</strong>
                    <span className="kpi-sub">
                      {nozzleZDraft === 0 ? "· centered on X-beam" : "lever arm"}
                    </span>
                  </span>
                </label>
                <div className="tension-control-row">
                  <input id="dp-nozzle-drop-z-mm-38"
                    type="range"
                    min={0}
                    max={120}
                    step={1}
                    value={Math.max(0, Math.min(120, nozzleZDraft))}
                    onChange={(e) => handleNozzleZSliderChange(Number(e.target.value))}
                    onPointerUp={() => commitNozzleZ(nozzleZDraft)}
                    onKeyUp={() => commitNozzleZ(nozzleZDraft)}
                  />
                  <input aria-label="Nozzle Drop Z (mm) value"
                    type="number"
                    className="tension-num-input"
                    min={0}
                    max={250}
                    step={1}
                    value={nozzleZDraft}
                    onChange={(e) => {
                      const v = Number(e.target.value);
                      if (!Number.isFinite(v)) return;
                      setNozzleZDraft(v);
                      if (v >= 0 && v <= 250) commitNozzleZ(v);
                    }}
                    onBlur={() => {
                      const clamped = Math.max(0, Math.min(250, Math.round(nozzleZDraft)));
                      setNozzleZDraft(clamped);
                      commitNozzleZ(clamped);
                    }}
                  />
                </div>
              </div>

              <details className="sub-accordion">
                <summary>
                  <span>Toolhead Structure &amp; CoG Offsets</span>
                  <span className="sub-accordion-tag">
                    {selectedBodyConstruction.short_name} · {toolheadPartMatShort(currentBodyMat)}
                  </span>
                </summary>
                <div className="sub-accordion-body">
                  {showHints && (
                    <div className="hint-line">
                      Choose the structural architecture and tune material &amp; section thickness per
                      stage — stiffness updates directly while weight stays strictly what you enter
                      above.
                    </div>
                  )}

                  {/* 0. Body Construction / Architecture */}
                  <div
                    className="field-row"
                    style={{
                      marginBottom: 8,
                      paddingBottom: 8,
                      borderBottom: "1px dashed rgba(255, 255, 255, 0.08)",
                    }}
                  >
                    <label htmlFor="dp-body-construction-architecture-39">
                      <span>Body Construction / Architecture</span>
                      <span className="sub-block-tag" style={{ color: "var(--accent)" }}>
                        {selectedBodyConstruction.short_name}
                      </span>
                    </label>
                    <select id="dp-body-construction-architecture-39"
                      value={currentBodyConstruction}
                      onChange={(e) => {
                        const nextVal = e.target.value as ToolheadConstructionType;
                        const prevShort = selectedBodyConstruction.short_name;
                        const nextItem =
                          toolheadConstructions.find((c) => c.id === nextVal) ??
                          selectedBodyConstruction;
                        update(
                          (d) => {
                            d.toolhead.body_construction = nextVal;
                            d.toolhead.preset_id = "custom";
                          },
                          `Toolhead Construction: ${prevShort} → ${nextItem.short_name}`
                        );
                      }}
                    >
                      {toolheadConstructions.map((opt) => (
                        <option key={opt.id} value={opt.id}>
                          {opt.name}
                        </option>
                      ))}
                    </select>
                    {showHints && selectedBodyConstruction.notes && (
                      <div
                        style={{
                          marginTop: 5,
                          fontSize: 11,
                          lineHeight: 1.35,
                          color: "var(--muted)",
                        }}
                      >
                        {selectedBodyConstruction.notes}
                      </div>
                    )}
                  </div>

                  {/* 1. X-Carriage Backplate */}
                  <div className="field-row" style={{ marginBottom: 6 }}>
                    <label htmlFor="dp-x-carriage-backplate-40">
                      <span>X-Carriage Backplate</span>
                      <span
                        className="sub-block-tag"
                        style={{ fontVariantNumeric: "tabular-nums" }}
                      >
                        {carriagePlateThickDraft.toFixed(1)} mm
                      </span>
                    </label>
                    <select id="dp-x-carriage-backplate-40"
                      value={currentCarriageMountMat}
                      onChange={(e) => {
                        const nextMat = e.target.value;
                        update(
                          (d) => {
                            d.toolhead.carriage_mount_material = nextMat;
                            if (MOUNT_MATERIAL_OPTIONS.some((o) => o.id === nextMat)) {
                              d.toolhead.mount_material = nextMat as ToolheadMountMaterial;
                            }
                            d.toolhead.preset_id = "custom";
                          },
                          `X-Carriage Backplate: ${toolheadPartMatShort(currentCarriageMountMat)} → ${toolheadPartMatShort(nextMat)}`
                        );
                      }}
                    >
                      {TOOLHEAD_PART_MATERIAL_OPTIONS.map((opt) => (
                        <option key={opt.id} value={opt.id}>
                          {opt.label}
                        </option>
                      ))}
                    </select>
                    <div
                      className="tension-control-row"
                      style={{ marginTop: 4, alignItems: "center", gap: 6 }}
                    >
                      <span style={{ fontSize: 11, color: "var(--muted)", whiteSpace: "nowrap" }}>
                        Plate
                      </span>
                      <input aria-label="X-Carriage Backplate thickness (mm)"
                        type="range"
                        min={2.0}
                        max={15.0}
                        step={0.5}
                        value={carriagePlateThickDraft}
                        onChange={(e) => handleCarriagePlateThickSliderChange(Number(e.target.value))}
                        onPointerUp={() => commitCarriagePlateThick(carriagePlateThickDraft)}
                        onKeyUp={() => commitCarriagePlateThick(carriagePlateThickDraft)}
                      />
                    </div>
                  </div>

                  {/* 2. Toolhead / Extruder Body */}
                  <div className="field-row" style={{ marginBottom: 6 }}>
                    <label htmlFor="dp-toolhead-extruder-body-41">
                      <span>Toolhead / Extruder Body</span>
                      <span
                        className="sub-block-tag"
                        style={{ fontVariantNumeric: "tabular-nums" }}
                      >
                        {bodyWallThickDraft.toFixed(1)} mm
                      </span>
                    </label>
                    <select id="dp-toolhead-extruder-body-41"
                      value={currentBodyMat}
                      onChange={(e) => {
                        const nextMat = e.target.value;
                        update(
                          (d) => {
                            d.toolhead.body_material = nextMat;
                            d.toolhead.preset_id = "custom";
                          },
                          `Toolhead Body: ${toolheadPartMatShort(currentBodyMat)} → ${toolheadPartMatShort(nextMat)}`
                        );
                      }}
                    >
                      {TOOLHEAD_PART_MATERIAL_OPTIONS.map((opt) => (
                        <option key={opt.id} value={opt.id}>
                          {opt.label}
                        </option>
                      ))}
                    </select>
                    <div
                      className="tension-control-row"
                      style={{ marginTop: 4, alignItems: "center", gap: 6 }}
                    >
                      <span style={{ fontSize: 11, color: "var(--muted)", whiteSpace: "nowrap" }}>
                        Wall
                      </span>
                      <input aria-label="Toolhead / Extruder Body wall thickness (mm)"
                        type="range"
                        min={2.0}
                        max={12.0}
                        step={0.5}
                        value={bodyWallThickDraft}
                        onChange={(e) => handleBodyWallThickSliderChange(Number(e.target.value))}
                        onPointerUp={() => commitBodyWallThick(bodyWallThickDraft)}
                        onKeyUp={() => commitBodyWallThick(bodyWallThickDraft)}
                      />
                    </div>
                  </div>

                  {/* 3. Hotend Mount / Clamp */}
                  <div className="field-row" style={{ marginBottom: 6 }}>
                    <label htmlFor="dp-hotend-mount-clamp-42">
                      <span>Hotend Mount / Clamp</span>
                      <span
                        className="sub-block-tag"
                        style={{ fontVariantNumeric: "tabular-nums" }}
                      >
                        {hotendMountThickDraft.toFixed(1)} mm
                      </span>
                    </label>
                    <select id="dp-hotend-mount-clamp-42"
                      value={currentHotendMountMat}
                      onChange={(e) => {
                        const nextMat = e.target.value;
                        update(
                          (d) => {
                            d.toolhead.hotend_mount_material = nextMat;
                            d.toolhead.preset_id = "custom";
                          },
                          `Hotend Clamp: ${toolheadPartMatShort(currentHotendMountMat)} → ${toolheadPartMatShort(nextMat)}`
                        );
                      }}
                    >
                      {TOOLHEAD_PART_MATERIAL_OPTIONS.map((opt) => (
                        <option key={opt.id} value={opt.id}>
                          {opt.label}
                        </option>
                      ))}
                    </select>
                    <div
                      className="tension-control-row"
                      style={{ marginTop: 4, alignItems: "center", gap: 6 }}
                    >
                      <span style={{ fontSize: 11, color: "var(--muted)", whiteSpace: "nowrap" }}>
                        Clamp
                      </span>
                      <input aria-label="Hotend Mount / Clamp thickness (mm)"
                        type="range"
                        min={3.0}
                        max={20.0}
                        step={0.5}
                        value={hotendMountThickDraft}
                        onChange={(e) => handleHotendMountThickSliderChange(Number(e.target.value))}
                        onPointerUp={() => commitHotendMountThick(hotendMountThickDraft)}
                        onKeyUp={() => commitHotendMountThick(hotendMountThickDraft)}
                      />
                    </div>
                  </div>

                  <div className="slider-row">
                    <label htmlFor="dp-cog-forward-y-mm-43">
                      <span>CoG Forward Y (mm)</span>
                      <span className="tension-readout">
                        <strong>{cogYDraft} mm</strong>
                        <span className="kpi-sub">
                          {cogYDraft === 0 ? "· centered on X-beam" : "in front of X-beam"}
                        </span>
                      </span>
                    </label>
                    <div className="tension-control-row">
                      <input id="dp-cog-forward-y-mm-43"
                        type="range"
                        min={0}
                        max={80}
                        step={1}
                        value={Math.max(0, Math.min(80, cogYDraft))}
                        onChange={(e) => handleCogYSliderChange(Number(e.target.value))}
                        onPointerUp={() => commitCogY(cogYDraft)}
                        onKeyUp={() => commitCogY(cogYDraft)}
                      />
                      <input aria-label="CoG Forward Y (mm) value"
                        type="number"
                        className="tension-num-input"
                        min={0}
                        max={250}
                        step={1}
                        value={cogYDraft}
                        onChange={(e) => {
                          const v = Number(e.target.value);
                          if (!Number.isFinite(v)) return;
                          setCogYDraft(v);
                          if (v >= 0 && v <= 250) commitCogY(v);
                        }}
                        onBlur={() => {
                          const clamped = Math.max(0, Math.min(250, Math.round(cogYDraft)));
                          setCogYDraft(clamped);
                          commitCogY(clamped);
                        }}
                      />
                    </div>
                  </div>

                  <div className="slider-row">
                    <label htmlFor="dp-cog-drop-z-mm-44">
                      <span>CoG Drop Z (mm)</span>
                      <span className="tension-readout">
                        <strong>{cogZDraft} mm</strong>
                        <span className="kpi-sub">
                          {cogZDraft === 0 ? "· centered on X-beam" : "below X-beam"}
                        </span>
                      </span>
                    </label>
                    <div className="tension-control-row">
                      <input id="dp-cog-drop-z-mm-44"
                        type="range"
                        min={0}
                        max={100}
                        step={1}
                        value={Math.max(0, Math.min(100, cogZDraft))}
                        onChange={(e) => handleCogZSliderChange(Number(e.target.value))}
                        onPointerUp={() => commitCogZ(cogZDraft)}
                        onKeyUp={() => commitCogZ(cogZDraft)}
                      />
                      <input aria-label="CoG Drop Z (mm) value"
                        type="number"
                        className="tension-num-input"
                        min={0}
                        max={250}
                        step={1}
                        value={cogZDraft}
                        onChange={(e) => {
                          const v = Number(e.target.value);
                          if (!Number.isFinite(v)) return;
                          setCogZDraft(v);
                          if (v >= 0 && v <= 250) commitCogZ(v);
                        }}
                        onBlur={() => {
                          const clamped = Math.max(0, Math.min(250, Math.round(cogZDraft)));
                          setCogZDraft(clamped);
                          commitCogZ(clamped);
                        }}
                      />
                    </div>
                  </div>
                </div>
              </details>
            </div>

            <div className="section-card" data-design-section="hotend">
              <div className="section-title">
                <span>Hotend Stiffness, Thermal &amp; Cooling</span>
                <span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
                  <span className="sub-block-tag" style={{ fontVariantNumeric: "tabular-nums" }}>
                    {specHotendMassG}g
                  </span>
                  <span className="sub-block-tag" style={{ color: hotendTier.color }}>
                    {hotendStiffnessDraft} N/mm
                  </span>
                </span>
              </div>

              {showHints && (
                <div className="hint-line">
                  Hotend mount architecture governs nozzle-tip bending stiffness (V6 groove-mount
                  whip vs. rigid 4-bolt standoffs or Chube 6-bolt Ti shell); melt zone &amp; wattage
                  set max volumetric flow.
                </div>
              )}

              <div className="field-row">
                <label htmlFor="dp-hotend-preset-45">Hotend Preset</label>
                <select id="dp-hotend-preset-45"
                  value={selectedHotendPresetVal}
                  onChange={(e) => {
                    const nextId = e.target.value;
                    const chosen = hotendPresets.find((hp) => hp.id === nextId);
                    if (!chosen) return;
                    const prevName = spec.toolhead.hotend.name;
                    const chosenMassG = chosen.mass_g ?? 45;
                    update(
                      (d) => {
                        d.toolhead.hotend.preset_id = chosen.id;
                        d.toolhead.hotend.name = chosen.name;
                        d.toolhead.hotend.mass_g = chosenMassG;
                        d.toolhead.hotend.mount_style = chosen.mount_style as HotendMountStyle;
                        d.toolhead.hotend.structural_stiffness_n_per_mm =
                          chosen.structural_stiffness_n_per_mm;
                        d.toolhead.hotend.overall_length_mm = chosen.overall_length_mm;
                        d.toolhead.hotend.heater_power_w = chosen.heater_power_w;
                        d.toolhead.hotend.block_material = chosen.block_material;
                        d.toolhead.hotend.block_mass_g = chosen.block_mass_g;
                        d.toolhead.hotend.nozzle_material = chosen.nozzle_material;
                        d.toolhead.hotend.melt_zone_length_mm = chosen.melt_zone_length_mm;
                        d.toolhead.hotend.heatbreak_material = chosen.heatbreak_material;
                        d.toolhead.hotend.heatbreak_length_mm = chosen.heatbreak_length_mm;
                        d.toolhead.hotend.heatbreak_wall_mm = chosen.heatbreak_wall_mm;
                        d.toolhead.hotend.heatbreak_id_mm = chosen.heatbreak_id_mm;
                        d.toolhead.hotend.heatsink_thermal_resistance_k_w =
                          chosen.heatsink_thermal_resistance_k_w;
                        d.toolhead.hotend.max_temp_c = chosen.max_temp_c;
                      },
                      `Hotend: ${prevName} → ${chosen.short_name || chosen.name} (${Math.round(chosen.structural_stiffness_n_per_mm)} N/mm · ${Math.round(chosenMassG)}g)`
                    );
                  }}
                >
                  {selectedHotendPresetVal === "__custom__" && (
                    <option value="__custom__">
                      {spec.toolhead.hotend.name || "Custom Hotend"} — {specHotendStiffness} N/mm ·{" "}
                      {specHotendMassG}g (Custom)
                    </option>
                  )}
                  {hotendPresets.map((hp) => {
                    const tier = hotendStiffnessTier(hp.structural_stiffness_n_per_mm);
                    const wtStr = hp.mass_g != null ? ` · ${Math.round(hp.mass_g)}g` : "";
                    return (
                      <option key={hp.id} value={hp.id}>
                        {hp.name} — {Math.round(hp.structural_stiffness_n_per_mm)} N/mm{wtStr} (
                        {tier.label})
                      </option>
                    );
                  })}
                </select>
                {matchedHotendPreset?.notes && showHints && (
                  <div className="hint-line">{matchedHotendPreset.notes}</div>
                )}
              </div>

              <div className="field-grid-2">
                <div className="field-row">
                  <label htmlFor="dp-nozzle-dia-mm-46">Nozzle Dia (mm)</label>
                  <input id="dp-nozzle-dia-mm-46"
                    type="number"
                    step={0.1}
                    value={spec.toolhead.hotend.nozzle_diameter_mm}
                    onChange={(e) =>
                      updatePositiveNum(
                        e.target.value,
                        (d, v) => {
                          d.toolhead.hotend.nozzle_diameter_mm = v;
                        },
                        (v) => `Nozzle Dia: ${spec.toolhead.hotend.nozzle_diameter_mm} → ${v} mm`
                      )
                    }
                  />
                </div>
                <div className="field-row">
                  <label htmlFor="dp-cooling-fan-cfm-47">Cooling Fan (CFM)</label>
                  <input id="dp-cooling-fan-cfm-47"
                    type="number"
                    step={0.5}
                    value={spec.toolhead.hotend.part_cooling_cfm}
                    onChange={(e) =>
                      updatePositiveNum(
                        e.target.value,
                        (d, v) => {
                          d.toolhead.hotend.part_cooling_cfm = v;
                        },
                        (v) => `Cooling Fan: ${spec.toolhead.hotend.part_cooling_cfm} → ${v} CFM`
                      )
                    }
                  />
                </div>
              </div>

              <details className="sub-accordion">
                <summary>
                  <span>Custom Hotend Thermal &amp; Stiffness</span>
                  <span className="sub-accordion-tag">
                    {spec.toolhead.hotend.heater_power_w}W ·{" "}
                    {spec.toolhead.hotend.melt_zone_length_mm}mm
                  </span>
                </summary>
                <div className="sub-accordion-body">
                  <div className="field-row">
                    <label htmlFor="dp-hotend-weight-g-48">
                      <span>Hotend Weight (g)</span>
                      <span
                        className="sub-block-tag"
                        style={{ color: "var(--accent)", fontVariantNumeric: "tabular-nums" }}
                        title="Added directly on top of Toolhead Mass (excl. Hotend) to form moving mass"
                      >
                        + {specHotendMassG}g to moving mass
                      </span>
                    </label>
                    <input id="dp-hotend-weight-g-48"
                      type="number"
                      min={5}
                      max={400}
                      step={1}
                      value={hotendMassDraft}
                      onChange={(e) => {
                        const raw = e.target.value;
                        setHotendMassDraft(raw);
                        const v = Number(raw);
                        if (raw.trim() !== "" && Number.isFinite(v) && v >= 5 && v <= 400) {
                          if (hotendMassTimerRef.current != null) {
                            window.clearTimeout(hotendMassTimerRef.current);
                          }
                          hotendMassTimerRef.current = window.setTimeout(
                            () => commitHotendMass(v),
                            220
                          );
                        }
                      }}
                      onBlur={() => {
                        const v = Number(hotendMassDraft);
                        const clamped =
                          hotendMassDraft.trim() !== "" && Number.isFinite(v)
                            ? Math.max(5, Math.min(400, Math.round(v)))
                            : specHotendMassG;
                        setHotendMassDraft(String(clamped));
                        commitHotendMass(clamped);
                      }}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") {
                          const v = Number(hotendMassDraft);
                          if (hotendMassDraft.trim() !== "" && Number.isFinite(v)) {
                            const clamped = Math.max(5, Math.min(400, Math.round(v)));
                            setHotendMassDraft(String(clamped));
                            commitHotendMass(clamped);
                          }
                        }
                      }}
                    />
                  </div>

                  <div className="field-row">
                    <label htmlFor="dp-hotend-mount-style-49">Hotend Mount Style</label>
                    <select id="dp-hotend-mount-style-49"
                      value={currentHotendMountStyle}
                      onChange={(e) => {
                        const nextStyle = e.target.value as HotendMountStyle;
                        const prevLbl =
                          HOTEND_MOUNT_STYLE_OPTIONS.find((o) => o.id === currentHotendMountStyle)
                            ?.shortLabel ?? currentHotendMountStyle;
                        const nextLbl =
                          HOTEND_MOUNT_STYLE_OPTIONS.find((o) => o.id === nextStyle)?.shortLabel ??
                          nextStyle;
                        update(
                          (d) => {
                            d.toolhead.hotend.mount_style = nextStyle;
                            d.toolhead.hotend.preset_id = null;
                          },
                          `Hotend Mount: ${prevLbl} → ${nextLbl}`
                        );
                      }}
                    >
                      {HOTEND_MOUNT_STYLE_OPTIONS.map((opt) => (
                        <option key={opt.id} value={opt.id}>
                          {opt.label}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div className="slider-row">
                    <label htmlFor="dp-hotend-structural-stiffness-n-mm-50">
                      <span>Hotend Structural Stiffness (N/mm)</span>
                      <span className="tension-readout">
                        <strong>{hotendStiffnessDraft.toLocaleString()} N/mm</strong>
                        <span className="kpi-sub" style={{ color: hotendTier.color }}>
                          {hotendTier.label}
                        </span>
                      </span>
                    </label>
                    <div className="tension-control-row">
                      <input id="dp-hotend-structural-stiffness-n-mm-50"
                        type="range"
                        min={25}
                        max={4000}
                        step={25}
                        value={Math.max(25, Math.min(4000, hotendStiffnessDraft))}
                        onChange={(e) => handleHotendStiffnessSliderChange(Number(e.target.value))}
                        onPointerUp={() => commitHotendStiffness(hotendStiffnessDraft)}
                        onKeyUp={() => commitHotendStiffness(hotendStiffnessDraft)}
                      />
                      <input aria-label="Hotend Structural Stiffness (N/mm) value"
                        type="number"
                        className="tension-num-input"
                        min={25}
                        max={4000}
                        step={25}
                        value={hotendStiffnessDraft}
                        onChange={(e) => {
                          const v = Number(e.target.value);
                          if (!Number.isFinite(v)) return;
                          setHotendStiffnessDraft(v);
                          if (v >= 25 && v <= 4000) commitHotendStiffness(v);
                        }}
                        onBlur={() => {
                          const clamped = Math.max(
                            25,
                            Math.min(4000, Math.round(hotendStiffnessDraft))
                          );
                          setHotendStiffnessDraft(clamped);
                          commitHotendStiffness(clamped);
                        }}
                      />
                    </div>
                  </div>

                  <div className="field-grid-2">
                    <div className="field-row">
                      <label htmlFor="dp-heater-power-w-51">Heater Power (W)</label>
                      <input id="dp-heater-power-w-51"
                        type="number"
                        value={spec.toolhead.hotend.heater_power_w}
                        onChange={(e) =>
                          updatePositiveNum(
                            e.target.value,
                            (d, v) => {
                              d.toolhead.hotend.heater_power_w = v;
                            },
                            (v) => `Heater Power: ${spec.toolhead.hotend.heater_power_w}W → ${v}W`
                          )
                        }
                      />
                    </div>
                    <div className="field-row">
                      <label htmlFor="dp-melt-zone-mm-52">Melt Zone (mm)</label>
                      <input id="dp-melt-zone-mm-52"
                        type="number"
                        value={spec.toolhead.hotend.melt_zone_length_mm}
                        onChange={(e) =>
                          updatePositiveNum(
                            e.target.value,
                            (d, v) => {
                              d.toolhead.hotend.melt_zone_length_mm = v;
                            },
                            (v) =>
                              `Melt Zone: ${spec.toolhead.hotend.melt_zone_length_mm} → ${v} mm`
                          )
                        }
                      />
                    </div>
                  </div>

                  {showHints && (
                    <div className="hint-line">
                      Titanium stops heat-creep clogs; higher CFM improves overhangs &amp; bridges
                    </div>
                  )}

                  <div className="field-row">
                    <label htmlFor="dp-heatbreak-material-53">Heatbreak Material</label>
                    <select id="dp-heatbreak-material-53"
                      value={spec.toolhead.hotend.heatbreak_material}
                      onChange={(e) => {
                        const nextVal = e.target.value;
                        update(
                          (d) => {
                            d.toolhead.hotend.heatbreak_material = nextVal;
                          },
                          `Heatbreak: ${HEATBREAK_LABELS[spec.toolhead.hotend.heatbreak_material] ?? spec.toolhead.hotend.heatbreak_material} → ${HEATBREAK_LABELS[nextVal] ?? nextVal}`
                        );
                      }}
                    >
                      {spec.toolhead.hotend.heatbreak_material !== "titanium_grade5" &&
                        spec.toolhead.hotend.heatbreak_material !== "stainless_304" && (
                          <option value={spec.toolhead.hotend.heatbreak_material}>
                            {materialLabel(spec.toolhead.hotend.heatbreak_material)} (Custom)
                          </option>
                        )}
                      <option value="titanium_grade5">Titanium Gr.5 (Low Heat Creep)</option>
                      <option value="stainless_304">Stainless 304 (Standard)</option>
                    </select>
                  </div>
                </div>
              </details>
            </div>

            <DesignBedZSection
              spec={spec}
              catalog={catalog}
              showHints={showHints}
              update={update}
              updatePositiveNum={updatePositiveNum}
              materialLabel={materialLabel}
            />

            <DesignAnchoringSection
              spec={spec}
              catalog={catalog}
              showHints={showHints}
              toolheadRelPos={toolheadRelPos}
              onToolheadRelPosChange={onToolheadRelPosChange}
              panelMeshN={panelMeshN}
              onPanelMeshNChange={onPanelMeshNChange}
              update={update}
            />
          </>
        )}

        {tab === "part_fea" && (
          <>
            <div className="section-card">
              <div className="section-title">
                <span>3D Solid Part FEA</span>
                <span style={{ color: feaEngine === "tet10" ? "var(--accent)" : "var(--muted)", fontSize: 11 }}>
                  {feaEngine === "tet10" ? "Tet10 (P2)" : "Voxel Hex8"}
                </span>
              </div>

              <div className="field-row">
                <label id="dp-solver-engine-54">Solver Engine</label>
                <div role="group" aria-labelledby="dp-solver-engine-54" className="engine-toggle">
                  <button
                    type="button"
                    className={`engine-toggle-btn ${feaEngine === "tet10" ? "active" : ""}`}
 aria-pressed={feaEngine === "tet10"}
                    onClick={() => setFeaEngine("tet10")}
                  >
                    Curved Tet10
                    <span className="engine-toggle-sub">Gmsh + scikit-fem</span>
                  </button>
                  <button
                    type="button"
                    className={`engine-toggle-btn ${feaEngine === "hex8" ? "active" : ""}`}
 aria-pressed={feaEngine === "hex8"}
                    onClick={() => setFeaEngine("hex8")}
                  >
                    Fast Voxel Hex8
                    <span className="engine-toggle-sub">Structured Grid</span>
                  </button>
                </div>
              </div>

              <div className="field-row">
                <label htmlFor="dp-parametric-bracket-mount-55">Parametric Bracket / Mount</label>
                <select id="dp-parametric-bracket-mount-55" value={partPreset} onChange={(e) => setPartPreset(e.target.value)}>
                  {catalog.part_presets.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.name}
                    </option>
                  ))}
                </select>
              </div>
              <div className="field-grid-2 wide-first">
                <div className="field-row">
                  <label htmlFor="dp-material-56">Material</label>
                  <select id="dp-material-56"
                    value={partMat}
                    onChange={(e) => {
                      setPartMat(e.target.value);
                      setPartFdm(e.target.value.includes("printed"));
                    }}
                  >
                    <optgroup label="3D-Printed Engineering Polymers">
                      {partFeaPrintedMaterials.map((m) => (
                        <option key={m.id} value={m.id}>
                          {m.name}
                        </option>
                      ))}
                    </optgroup>
                    <optgroup label="CNC Metals & Carbon Plate">
                      {partFeaMetalMaterials.map((m) => (
                        <option key={m.id} value={m.id}>
                          {m.name}
                        </option>
                      ))}
                    </optgroup>
                  </select>
                </div>
                <div className="field-row">
                  <label htmlFor="dp-fdm-z-anisotropy-57">FDM Z-Anisotropy</label>
                  <select id="dp-fdm-z-anisotropy-57" value={partFdm ? "yes" : "no"} onChange={(e) => setPartFdm(e.target.value === "yes")}>
                    <option value="yes">FDM Layer Model</option>
                    <option value="no">Isotropic (CNC)</option>
                  </select>
                </div>
              </div>
              <div className="field-grid-3">
                <div className="field-row">
                  <label htmlFor="dp-fx-n-58">Fx (N)</label>
                  <input id="dp-fx-n-58" type="number" value={fx} onChange={(e) => setFx(Number(e.target.value))} />
                </div>
                <div className="field-row">
                  <label htmlFor="dp-fy-n-59">Fy (N)</label>
                  <input id="dp-fy-n-59" type="number" value={fy} onChange={(e) => setFy(Number(e.target.value))} />
                </div>
                <div className="field-row">
                  <label htmlFor="dp-fz-n-60">Fz (N)</label>
                  <input id="dp-fz-n-60" type="number" value={fz} onChange={(e) => setFz(Number(e.target.value))} />
                </div>
              </div>
              <div className="btn-row">
                <button
                  className="btn primary"
                  disabled={loading || topoLoading}
                  onClick={() =>
                    onRunPartFea({
                      preset: partPreset,
                      materialId: partMat,
                      forceN: [fx, fy, fz],
                      fdmAnisotropic: partFdm,
                      engine: feaEngine,
                    })
                  }
                >
                  {loading ? "Solving..." : "Run Part FEA"}
                </button>
                <button
                  className="btn"
                  disabled={loading || topoLoading || exportingStep}
                  onClick={() => onDownloadPartStep(partPreset)}
                  title="Download real OpenCASCADE B-Rep STEP CAD file for this parametric bracket"
                >
                  {exportingStep ? "Exporting..." : "Download Bracket STEP"}
                </button>
              </div>

              <div style={{ borderTop: "1px solid var(--border)", paddingTop: 10, marginTop: 4 }}>
                <div className="section-title" style={{ marginBottom: 4 }}>
                  Or Upload Custom STEP / STL / OBJ
                </div>
                <div style={{ fontSize: 11, color: "var(--muted)", marginBottom: 6, lineHeight: 1.3 }}>
                  <code>.step</code>/<code>.stp</code> → Curved Tet10 (Gmsh) • <code>.stl</code>/<code>.obj</code> → Voxel Hex8
                </div>
                <div className="field-grid-2" style={{ marginBottom: 6 }}>
                  <div className="field-row">
                    <label htmlFor="dp-fixed-clamp-face-61">Fixed Clamp Face</label>
                    <select id="dp-fixed-clamp-face-61" value={fixedFace} onChange={(e) => setFixedFace(e.target.value)}>
                      <option value="min_z">Bottom (-Z)</option>
                      <option value="min_x">Left (-X)</option>
                      <option value="min_y">Front (-Y)</option>
                    </select>
                  </div>
                  <div className="field-row">
                    <label htmlFor="dp-loaded-face-62">Loaded Face</label>
                    <select id="dp-loaded-face-62" value={loadFace} onChange={(e) => setLoadFace(e.target.value)}>
                      <option value="max_z">Top (+Z)</option>
                      <option value="max_x">Right (+X)</option>
                      <option value="max_y">Rear (+Y)</option>
                    </select>
                  </div>
                </div>
                <input aria-label="Upload a custom STEP, STL or OBJ part"
                  type="file"
                  accept=".stl,.obj,.step,.stp"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) {
                      onRunStlFea({
                        file,
                        materialId: partMat,
                        fixedFace,
                        loadFace,
                        forceN: [fx, fy, fz],
                        fdmAnisotropic: partFdm,
                      });
                      e.target.value = "";
                    }
                  }}
                />
              </div>
            </div>

            <div className="section-card" style={{ borderColor: "rgba(168, 85, 247, 0.4)" }}>
              <div className="section-title">
                <span>3D Topology Optimization (SIMP)</span>
                <span style={{ color: "var(--purple)", fontSize: 11 }}>Generative</span>
              </div>
              <div style={{ fontSize: 11, color: "var(--muted)", lineHeight: 1.35 }}>
                Carves low-stress voxels from the selected preset &amp; load case above while maximizing
                stiffness-to-weight ratio.
              </div>

              <div className="slider-row">
                <label htmlFor="dp-target-volume-fraction-63">
                  <span>Target Volume Fraction</span>
                  <strong style={{ color: "var(--text)" }}>{Math.round(targetVolFrac * 100)}%</strong>
                </label>
                <input id="dp-target-volume-fraction-63"
                  type="range"
                  min={0.25}
                  max={0.8}
                  step={0.05}
                  value={targetVolFrac}
                  onChange={(e) => setTargetVolFrac(Number(e.target.value))}
                />
              </div>

              <div className="slider-row">
                <label htmlFor="dp-simp-iterations-64">
                  <span>SIMP Iterations</span>
                  <strong style={{ color: "var(--text)" }}>{topoIterations}</strong>
                </label>
                <input id="dp-simp-iterations-64"
                  type="range"
                  min={8}
                  max={30}
                  step={1}
                  value={topoIterations}
                  onChange={(e) => setTopoIterations(Number(e.target.value))}
                />
              </div>

              <label className="checkbox-row">
                <input
                  type="checkbox"
                  checked={preserveBoundaries}
                  onChange={(e) => setPreserveBoundaries(e.target.checked)}
                />
                <span>Preserve Mounting Faces (Fixed &amp; Loaded)</span>
              </label>

              <button
                className="btn primary"
                disabled={loading || topoLoading}
                onClick={() =>
                  onRunTopoOpt({
                    preset: partPreset,
                    materialId: partMat,
                    forceN: [fx, fy, fz],
                    targetVolumeFraction: targetVolFrac,
                    iterations: topoIterations,
                    preserveBoundaries,
                    fdmAnisotropic: partFdm,
                  })
                }
              >
                {topoLoading ? "Running SIMP Optimization..." : "Run Topology Opt"}
              </button>

              <button
                className="btn"
                disabled={loading || topoLoading}
                style={hasTopoResult ? { borderColor: "var(--accent)", color: "var(--accent)" } : undefined}
                onClick={() =>
                  onDownloadTopoStl({
                    preset: partPreset,
                    materialId: partMat,
                    forceN: [fx, fy, fz],
                    targetVolumeFraction: targetVolFrac,
                    iterations: topoIterations,
                    preserveBoundaries,
                    fdmAnisotropic: partFdm,
                  })
                }
              >
                Download Optimized STL
              </button>
            </div>
          </>
        )}

        {tab === "yaml" && (
          <div className="section-card">
            <div className="section-title">
              <span>Declarative Printer YAML</span>
              <button className="btn primary" style={{ padding: "4px 8px" }} disabled={loading} onClick={onSimulateYaml}>
                Apply YAML
              </button>
            </div>
            <textarea
              aria-label="Printer YAML"
              className="code-editor"
              value={yamlText}
              onChange={(e) => onYamlChange(e.target.value)}
              spellCheck={false}
            />
          </div>
        )}

        {tab === "gcode" && (
          <div className="section-card">
            <div className="section-title">
              <span>Custom G-code Input</span>
              {onRunGcode && (
                <button
                  type="button"
                  className="btn primary"
                  style={{ padding: "4px 8px" }}
                  disabled={loading}
                  onClick={onRunGcode}
                  title="Run the virtual print with this G-code and show the result in the Print view"
                >
                  {loading ? "Solving..." : "Run Print Sim"}
                </button>
              )}
            </div>
            <p style={{ margin: 0, color: "var(--muted)", fontSize: 12 }}>
              Leave blank to run the built-in high-speed calibration cube &amp; ringing torture benchmark, or paste
              any Marlin/Klipper/OrcaSlicer G-code below.
            </p>
            <textarea
              aria-label="G-code input"
              className="code-editor"
              placeholder="; Paste G-code here (G0/G1 moves)..."
              value={gcodeText}
              onChange={(e) => onGcodeChange(e.target.value)}
              spellCheck={false}
            />
          </div>
        )}
      </div>
    </aside>
  );
};
