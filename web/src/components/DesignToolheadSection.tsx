import React, { useEffect, useRef, useState } from "react";
import { humanizeId } from "../utils/displayText";
import {
  CadSuperelementSummary,
  CatalogResponse,
  HotendMountStyle,
  PrinterSpecData,
  SimulationResponse,
  ToolheadConstructionCatalogItem,
  ToolheadConstructionType,
  ToolheadMountMaterial,
  ToolheadPresetItem,
  ZProbeType,
} from "../types";
import {
  ModelOverrideControl,
  PresetAffectsStrip,
  clearModelOverrideGroup,
} from "./ModelOverrideControl";

export interface BakedHardwareAssetManifest {
  asset_id?: string;
  category?: string;
  name?: string;
  reference_material_id?: string;
  mass_kg?: number;
  com_mm?: [number, number, number];
  port0_mm?: [number, number, number];
  port1_mm?: [number, number, number];
  port_stiffness_n_per_mm?: { kx?: number; ky?: number; kz?: number };
  element_type?: string;
  num_elements?: number;
  cad_trust_pct?: number;
  visual_glb_url?: string;
}

export interface BakedHardwareAssetsResponse {
  hotends?: Record<string, BakedHardwareAssetManifest>;
  toolheads?: Record<string, BakedHardwareAssetManifest>;
  assemblies?: Record<string, BakedHardwareAssetManifest>;
  gantry_brackets?: Record<string, BakedHardwareAssetManifest>;
}

let bakedHardwareAssetsCache: BakedHardwareAssetsResponse | null = null;
let bakedHardwareAssetsPromise: Promise<BakedHardwareAssetsResponse> | null = null;

function fetchBakedHardwareAssets(): Promise<BakedHardwareAssetsResponse> {
  if (bakedHardwareAssetsCache) return Promise.resolve(bakedHardwareAssetsCache);
  if (!bakedHardwareAssetsPromise) {
    bakedHardwareAssetsPromise = fetch("/api/assets/hardware")
      .then((res) => (res.ok ? (res.json() as Promise<BakedHardwareAssetsResponse>) : {}))
      .then((data) => {
        bakedHardwareAssetsCache = data && typeof data === "object" ? data : {};
        return bakedHardwareAssetsCache;
      })
      .catch(() => {
        bakedHardwareAssetsPromise = null;
        return {};
      });
  }
  return bakedHardwareAssetsPromise;
}

export function useBakedHardwareAssets(): BakedHardwareAssetsResponse {
  const [assets, setAssets] = useState<BakedHardwareAssetsResponse>(
    () => bakedHardwareAssetsCache ?? {}
  );
  useEffect(() => {
    let active = true;
    fetchBakedHardwareAssets().then((data) => {
      if (active) setAssets(data);
    });
    return () => {
      active = false;
    };
  }, []);
  return assets;
}

export const fmtMassG = (g: number): string =>
  Math.abs(g - Math.round(g)) < 0.05 ? `${Math.round(g)}` : g.toFixed(1);

export const cadHarmonicTipStiffness = (se?: CadSuperelementSummary | null): number | null => {
  const kx = se?.port_stiffness_n_per_mm?.kx;
  const ky = se?.port_stiffness_n_per_mm?.ky;
  if (kx && ky && Number.isFinite(kx) && Number.isFinite(ky) && kx > 0 && ky > 0) {
    return 2 / (1 / kx + 1 / ky);
  }
  return null;
};

export const formatElemCount = (n?: number): string => {
  if (!Number.isFinite(n) || !n) return "—";
  return n >= 1000 ? `${(n / 1000).toFixed(1)}k` : n.toLocaleString();
};

export const formatCadCmsTooltip = (se: CadSuperelementSummary): string => {
  const elemLabel = se.element_type ? se.element_type.toUpperCase() : "TET";
  const elems = Number.isFinite(se.num_elements) ? se.num_elements.toLocaleString() : "—";
  const kx = Number.isFinite(se.port_stiffness_n_per_mm?.kx)
    ? Math.round(se.port_stiffness_n_per_mm.kx).toLocaleString()
    : "—";
  const ky = Number.isFinite(se.port_stiffness_n_per_mm?.ky)
    ? Math.round(se.port_stiffness_n_per_mm.ky).toLocaleString()
    : "—";
  const kz = Number.isFinite(se.port_stiffness_n_per_mm?.kz)
    ? Math.round(se.port_stiffness_n_per_mm.kz).toLocaleString()
    : "—";
  const trust = se.cad_trust_pct != null ? ` · ${Math.round(se.cad_trust_pct)}% trust` : "";
  return `CAD CMS 3D Superelement: ${se.name} (${elems} ${elemLabel} elements · tip stiffness Kx ${kx} / Ky ${ky} / Kz ${kz} N/mm${trust})`;
};

const MOUNT_MATERIAL_OPTIONS: Array<{
  id: ToolheadMountMaterial;
  label: string;
  shortLabel: string;
}> = [
  { id: "abs_printed", label: "Printed ABS (2.1 GPa)", shortLabel: "Printed ABS" },
  { id: "pet_cf_printed", label: "Printed PET-CF15 (5.2 GPa)", shortLabel: "Printed PET-CF15" },
  { id: "pa12_cf_printed", label: "Printed PA12-CF (6.2 GPa)", shortLabel: "Printed PA12-CF" },
  { id: "pc_pbt_gf_printed", label: "Printed PC-PBT-GF (6.1 GPa)", shortLabel: "Printed PC-PBT-GF" },
  { id: "ppa_cf_printed", label: "Printed PPA-CF (10.8 GPa)", shortLabel: "Printed PPA-CF" },
  { id: "pps_cf_printed", label: "Printed PPS-CF (9.2 GPa)", shortLabel: "Printed PPS-CF" },
  { id: "slm_aluminum", label: "SLM AlSi10Mg (68 GPa)", shortLabel: "SLM AlSi10Mg" },
  { id: "slm_ti6al4v", label: "SLM Ti-6Al-4V Gr.5/23 (110 GPa)", shortLabel: "SLM Ti-6Al-4V" },
  { id: "slm_maraging_steel", label: "SLM MS1 Steel (180 GPa)", shortLabel: "SLM MS1" },
  { id: "slm_316l_steel", label: "SLM 316L Steel (185 GPa)", shortLabel: "SLM 316L" },
  { id: "slm_17_4ph_steel", label: "SLM 17-4PH H900 (190 GPa)", shortLabel: "SLM 17-4PH" },
  { id: "sheet_metal_steel", label: "Sheet Metal Steel (200 GPa)", shortLabel: "Sheet Metal Steel" },
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
  { id: "pc_pbt_gf_printed", label: "Printed PC-PBT-GF (6.1 GPa)", shortLabel: "PC-PBT-GF" },
  { id: "ppa_cf_printed", label: "Printed PPA-CF (10.8 GPa)", shortLabel: "PPA-CF" },
  { id: "pps_cf_printed", label: "Printed PPS-CF (9.2 GPa)", shortLabel: "PPS-CF" },
  { id: "carbon_fiber_plate", label: "3K Carbon Fiber Plate (58 GPa)", shortLabel: "3K CF Plate" },
  { id: "slm_aluminum", label: "SLM AlSi10Mg (68 GPa)", shortLabel: "SLM Al" },
  { id: "slm_ti6al4v", label: "SLM Ti-6Al-4V Gr.5/23 (110 GPa)", shortLabel: "SLM Ti64" },
  { id: "slm_maraging_steel", label: "SLM MS1 Steel (180 GPa)", shortLabel: "SLM MS1" },
  { id: "slm_316l_steel", label: "SLM 316L Steel (185 GPa)", shortLabel: "SLM 316L" },
  { id: "slm_17_4ph_steel", label: "SLM 17-4PH H900 (190 GPa)", shortLabel: "SLM 17-4PH" },
  { id: "sheet_metal_steel", label: "Sheet Metal Steel (200 GPa)", shortLabel: "Steel Sheet" },
  { id: "aluminum_6061_t6", label: "CNC Al 6061-T6 (68.9 GPa)", shortLabel: "CNC Al 6061" },
  { id: "aluminum_7075_t6", label: "CNC Al 7075-T6 (71.7 GPa)", shortLabel: "CNC Al 7075" },
  { id: "titanium_grade5", label: "CNC Ti-6Al-4V Gr.5 (113.8 GPa)", shortLabel: "CNC Ti64" },
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


const FALLBACK_Z_PROBE_PARAMS: Record<
  ZProbeType,
  { compliance_factor: number; cog_y_shift_mm: number }
> = {
  static_eddy: { compliance_factor: 1.0, cog_y_shift_mm: 0.0 },
  klicky_inductive: { compliance_factor: 0.98, cog_y_shift_mm: -1.5 },
  cnc_tap: { compliance_factor: 0.58, cog_y_shift_mm: -6.5 },
  printed_tap: { compliance_factor: 0.34, cog_y_shift_mm: -8.5 },
};


const FALLBACK_TOOLHEAD_PRESETS: ToolheadPresetItem[] = [
  {
    preset_id: "stealthburner",
    name: "Voron StealthBurner + CW2",
    total_mass_g: 398,
    cog_offset_mm: [0, -24, 16],
    nozzle_offset_mm: [0, -34, -60],
    mount_material: "abs_printed",
    body_construction: "multi_piece_clamshell",
    internal_stiffness_n_per_mm: 350,
    z_probe_type: "static_eddy",
    default_hardware_summary:
      "Official Stealthburner_CW2.5 STEP (198 cm³ ABS ~158 g) + LDO-36STH20 (85 g) + BMG (26 g) + 5015/4010 fans (42 g) + HIWIN MGN12H carriage (54 g) + PCB/LEDs/hardware (~33 g) = 398 g on-rail excl. hotend (444 g w/ Dragon HF; CW2 motor pulls CoG to Z +16 mm).",
  },
  {
    preset_id: "stealthburner_tap",
    name: "Voron StealthBurner + Printed Tap",
    total_mass_g: 438,
    cog_offset_mm: [0, -24, 14],
    nozzle_offset_mm: [0, -34, -60],
    mount_material: "abs_printed",
    body_construction: "multi_piece_clamshell",
    internal_stiffness_n_per_mm: 350,
    z_probe_type: "printed_tap",
    default_hardware_summary:
      "StealthBurner + CW2 with Printed Voron Tap carriage (+40 g net over stock X-carriage for 50 mm MGN9H rail/block, PCB & magnets; -8.5 mm additional Y shift applied by printed_tap) = 438 g on-rail excl. hotend (484 g w/ Dragon HF).",
  },
  {
    preset_id: "xol_compact",
    name: "XOL 2 Compact Dual-4010",
    total_mass_g: 278,
    cog_offset_mm: [0, -18, 10],
    nozzle_offset_mm: [0, -29, -49.5],
    mount_material: "pa12_cf_printed",
    body_construction: "compact_cage",
    internal_stiffness_n_per_mm: 650,
    z_probe_type: "static_eddy",
    default_hardware_summary:
      "Official Xol Assembly STEP (60.4 cm³ ASA ~51 g) + Sherpa Mini w/ 36STH20 (115 g) + 2×4010 blowers & 2510 fan (36.4 g) + HIWIN MGN12H carriage (54 g) + EBB36/probe/hardware (~22 g) = 278 g on-rail excl. hotend (328 g w/ Rapido HF; ~274 g off-rail).",
  },
  {
    preset_id: "a4t_beacon",
    name: "A4T + WW-BMG/WWG2 + Static Beacon",
    total_mass_g: 298,
    cog_offset_mm: [0, -19, 8],
    nozzle_offset_mm: [0, -28, -50],
    mount_material: "ppa_cf_printed",
    body_construction: "compact_cage",
    internal_stiffness_n_per_mm: 1150,
    z_probe_type: "static_eddy",
    default_hardware_summary:
      "Official A4T Cowling + HE Duct + Xol Carriage STEPs (71.8 cm³ ~60 g ASA) + WW-BMG/WWG2 (~134 g) + 2×4010 blowers & Delta 2510 (36.4 g) + HIWIN MGN12H carriage (54 g) + Beacon/LEDs/hardware (~13 g) = 298 g on-rail excl. hotend (344 g w/ Dragon HF; ~290 g off-rail).",
  },
  {
    preset_id: "dragon_burner",
    name: "Dragon Burner V8 (Orbiter 2 + Dual-4010)",
    total_mass_g: 272,
    cog_offset_mm: [0, -14, 4.5],
    nozzle_offset_mm: [0, -20, -50],
    mount_material: "pa12_cf_printed",
    body_construction: "compact_cage",
    internal_stiffness_n_per_mm: 720,
    z_probe_type: "static_eddy",
    default_hardware_summary:
      "Official Dragon_Burner_v0.2_v8 STEP (27.7 cm³ core + ~18 cm³ MGN12H carriage = 37.5 g ABS) + Orbiter v2.0 (135 g) + 2×4010 blowers & 3010 fan (32 g) + HIWIN MGN12H carriage (54 g) + Beacon/hardware (~13.5 g) = 272 g on-rail excl. hotend (318 g w/ Dragon HF; ~264 g off-rail).",
  },
  {
    preset_id: "archetype_cpap",
    name: "Archetype Breakneck CPAP (Sherpa Micro)",
    total_mass_g: 248,
    cog_offset_mm: [0, -15.5, 12],
    nozzle_offset_mm: [0, -21, -64],
    mount_material: "ppa_cf_printed",
    body_construction: "compact_cage",
    internal_stiffness_n_per_mm: 950,
    z_probe_type: "static_eddy",
    cpap_blower_id: "ws7040_24v",
    cpap_hose_id_mm: 19,
    default_hardware_summary:
      "Official Archetype Public CAD STEP (Breakneck v24: 81.5 cm³ printed core/duct/rear-bar ~68.4 g) + Sherpa Micro (105 g) + 2510 heatsink fan (6.4 g) + HIWIN MGN12H carriage (54 g) + Beacon & hardware (~14.3 g) + remote 19 mm CPAP hose = 248 g on-rail excl. hotend (344 g w/ Goliath Air; nozzle Z -64 mm UHF / -74 mm Goliath).",
  },
  {
    preset_id: "monolith_toolhead_cpap",
    name: "Monolith Toolhead (Belt-Coplanar CPAP + 37:1 Worm)",
    total_mass_g: 218,
    cog_offset_mm: [0, -11.5, -1.5],
    nozzle_offset_mm: [0, -16, -52],
    mount_material: "slm_aluminum",
    body_construction: "extruder_as_chassis",
    internal_stiffness_n_per_mm: 2400,
    z_probe_type: "static_eddy",
    cpap_blower_id: "ws7040_24v",
    cpap_hose_id_mm: 15,
    belt_plane_z_offset_mm: -1.5,
    default_hardware_summary:
      "Back-solved from CloakedWayne ~750 g 300 mm build & monolith3d.xyz SLM BOM: SLM AlSi10Mg main body w/ CPAP ducts-as-heatsink (~48 g) + exact CAD SLM Al belt clamp (6.6 g / 2.47 cm³) + SLM Ti hotend brace (~7 g) + 37:1 worm w/ LDO-36STH17 (96 g; +15 g w/ 20 mm) + LDO MGN12H (48.3 g) + Beacon & hardware (~12 g) = 218 g on-rail excl. hotend (270 g dressed w/ 52 g 50 mm HF hotend).",
  },
  {
    preset_id: "microwaved_bowden",
    name: "Microwaved Bowden SLM (CPAP + Beacon)",
    total_mass_g: 130,
    cog_offset_mm: [0, -8.5, -2],
    nozzle_offset_mm: [0, -17.5, -35.5],
    mount_material: "aluminum_6061_t6",
    body_material: "titanium_grade5",
    body_construction: "cnc_monoblock",
    internal_stiffness_n_per_mm: 5340,
    z_probe_type: "static_eddy",
    extruder_type: "bowden",
    default_hardware_summary:
      "Remote Bowden extruder (no onboard motor) — 130 g on-rail includes SLM Al body (43.4 g), Ti bottom brace & shield (2.9 g), LDO MGN12H carriage (48.3 g), Monolith SLM belt clamp (6.6 g Al / 19.7 g 316L SS), Beacon RevH & ECAS/fasteners (hotend added separately).",
  },
  {
    preset_id: "microwaved_bowden_chube",
    name: "Microwaved Bowden Chube V1 SLM (CPAP + Beacon)",
    total_mass_g: 135,
    cog_offset_mm: [0, -10.5, -3],
    nozzle_offset_mm: [0, -20.5, -38.5],
    mount_material: "aluminum_6061_t6",
    body_material: "aluminum_6061_t6",
    body_construction: "cnc_monoblock",
    internal_stiffness_n_per_mm: 2990,
    z_probe_type: "static_eddy",
    extruder_type: "bowden",
    default_hardware_summary:
      "Remote Bowden extruder (no onboard motor) — 135 g on-rail includes SLM Al Chube V1 body (~56.5 g), LDO MGN12H carriage (48.3 g), Monolith SLM belt clamp (6.6 g Al / 19.7 g 316L SS), Beacon RevH & ECAS/fasteners (hotend added separately).",
  },
  {
    preset_id: "el_fuego",
    name: "El Fuego SLM (CPAP + Sherpa Micro + Beacon)",
    total_mass_g: 262,
    cog_offset_mm: [0, -10.5, 2],
    nozzle_offset_mm: [-0.3, -27.7, -63.7],
    mount_material: "slm_aluminum",
    body_material: "slm_aluminum",
    body_construction: "cnc_monoblock",
    internal_stiffness_n_per_mm: 1007,
    z_probe_type: "static_eddy",
    cpap_blower_id: "wm7575_220vac",
    cpap_hose_id_mm: 19,
    shared_heatsink_cpap_duct: true,
    extruder_type: "direct_drive",
    default_hardware_summary:
      "Sherpa Micro direct-drive extruder + NEMA14 pancake (~105 g, pulls CoG to Z +2.0 mm), 3-part SLM CAD chassis (94.2 g: AlSi10Mg main body + Al belt plate + Ti extruder plate), LDO MGN12H carriage (48.3 g), Beacon RevH & fasteners, plus WM7575 CPAP (19 mm hose) w/ shared heatsink bleed. Hotend added separately; 3D mesh renders chassis only.",
  },
  {
    preset_id: "vz_printhead_cnc",
    name: "VzBot Vz-Printhead CNC 6061 (HextrudORT Low + CPAP + Beacon)",
    total_mass_g: 272,
    cog_offset_mm: [1.9, -10.3, 13.4],
    nozzle_offset_mm: [-0.1, -22.4, -54.2],
    offset_datum_rail_mount: "top",
    mount_material: "aluminum_6061_t6",
    body_construction: "plate_sandwich_cf",
    internal_stiffness_n_per_mm: 1200,
    z_probe_type: "static_eddy",
    cpap_blower_id: "ws7040_24v",
    cpap_hose_id_mm: 15,
    extruder_type: "direct_drive",
    default_hardware_summary:
      "Mellow CNC Vz-Printhead on the official v24 cage layout (1.5 mm top/bottom, 2.5 mm side 6061 plates fitted to the published 27 g / 22 g masses) + HextrudORT Low + Moons NEMA14 + CPAP + Beacon = 244 g on-rail with the MGN9H (272 g MGN12H-convention) excl. hotend.",
  },
  {
    preset_id: "vz_printhead_printed",
    name: "VzBot Vz-Printhead Printed v24 (ABS/ASA, HextrudORT Low + CPAP + Beacon)",
    total_mass_g: 276,
    cog_offset_mm: [1.9, -10.3, 13.4],
    nozzle_offset_mm: [-0.1, -22.4, -54.2],
    offset_datum_rail_mount: "top",
    mount_material: "abs_printed",
    body_construction: "plate_sandwich_cf",
    internal_stiffness_n_per_mm: 450,
    z_probe_type: "static_eddy",
    cpap_blower_id: "ws7040_24v",
    cpap_hose_id_mm: 15,
    extruder_type: "direct_drive",
    default_hardware_summary:
      "Exact official printed v24 Vz-Printhead cage (27.4 g ABS/ASA at 4 walls / 40-50% infill) + aluminium standoffs + HextrudORT Low + Moons NEMA14 + CPAP + Beacon = 248 g on-rail with the MGN9H (276 g MGN12H-convention) excl. hotend.",
  },
  {
    preset_id: "trinity_1_6_dav",
    name: "WV-Design Trinity 1.6 (Dragon Ace Volcano + Dual-3628)",
    total_mass_g: 382,
    cog_offset_mm: [0, -28.2, 3.6],
    nozzle_offset_mm: [0, -32.74, -81.0],
    mount_material: "ppa_cf_printed",
    body_construction: "compact_cage",
    internal_stiffness_n_per_mm: 1350,
    z_probe_type: "static_eddy",
    belt_plane_z_offset_mm: -0.3,
    cpap_blower_id: "none",
    extruder_type: "direct_drive",
    extruder_gear_ratio: 6.25,
    default_hotend_preset_id: "trianglelab_dragon_ace_volcano",
    part_cooling_cfm: 20,
    envelope_mm: [114.3, 62.9, 155],
    default_hardware_summary:
      "Official WV-Design Trinity 1.6-stable STEP (65.9 cm³ unibody + 24.4 cm³ ducts = 88.9 g ABS-GF/PPA-CF) + FYSETC CNC Sherpa Mini v3 (135 g, 6.25:1) + dual Delta 3628 12V PWM fans (64.4 g, ~20 CFM) + Delta 2510 HEF + Monolith SLM clamp + Beacon + HIWIN MGN12H = 382 g on-rail excl. hotend (373 g off-rail dressed w/ 45 g DAV; CoG dead between rail balls).",
  },
  {
    preset_id: "trinity_1_6_chube",
    name: "WV-Design Trinity 1.6 (Chube Compact + Dual-3628)",
    total_mass_g: 394,
    cog_offset_mm: [0, -28.5, 1.8],
    nozzle_offset_mm: [0, -32.74, -81.18],
    mount_material: "ppa_cf_printed",
    body_construction: "compact_cage",
    internal_stiffness_n_per_mm: 1350,
    z_probe_type: "static_eddy",
    belt_plane_z_offset_mm: -0.3,
    cpap_blower_id: "none",
    extruder_type: "direct_drive",
    extruder_gear_ratio: 6.25,
    default_hotend_preset_id: "chube_compact",
    part_cooling_cfm: 20,
    envelope_mm: [119.2, 60.9, 155],
    default_hardware_summary:
      "Official WV-Design Trinity 1.6-stable Chube Compact STEP (78.1 cm³ unibody + 24.2 cm³ ducts = 100.7 g ABS-GF/PPA-CF) + FYSETC CNC Sherpa Mini v3 (135 g, 6.25:1) + dual Delta 3628 12V PWM fans (64.4 g, ~20 CFM) + Delta 2510 HEF + Monolith SLM clamp + Beacon + HIWIN MGN12H = 394 g on-rail excl. hotend (403 g off-rail dressed w/ 63 g Chube Compact).",
  },
  {
    preset_id: "heavy_uhf_long",
    name: "Heavy Long-Nozzle UHF Toolhead",
    total_mass_g: 475,
    cog_offset_mm: [0, -26, -34],
    nozzle_offset_mm: [0, -28, -78],
    mount_material: "abs_printed",
    body_construction: "multi_piece_clamshell",
    internal_stiffness_n_per_mm: 250,
    z_probe_type: "static_eddy",
    default_hardware_summary:
      "Worst-case heavy direct-drive extruder, multi-blower UHF ABS shroud, long-nozzle drop spacer, MGN12H carriage (54 g) & static eddy probe = 475 g on-rail excl. hotend (hotend added separately).",
  },
];

function toolheadPartMatShort(id?: string | null): string {
  if (!id) return "ABS";
  return TOOLHEAD_PART_MATERIAL_OPTIONS.find((m) => m.id === id)?.shortLabel ?? id.replace(/_/g, " ");
}


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

export interface DesignToolheadSectionProps {
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
  materialLabel: (id: string) => string;
}

export const DesignToolheadSection: React.FC<DesignToolheadSectionProps> = ({
  spec,
  catalog,
  sim,
  showHints,
  advancedMode,
  update,
  updatePositiveNum,
  materialLabel,
}) => {
  const bakedAssets = useBakedHardwareAssets();

  const [cogYDraft, setCogYDraft] = useState<number>(
    Math.round(Math.abs(spec.toolhead.cog_offset_mm?.[1] ?? 18))
  );
  const cogYTimerRef = useRef<number | null>(null);

  const [cogZDraft, setCogZDraft] = useState<number>(
    Math.round(Math.abs(spec.toolhead.cog_offset_mm?.[2] ?? 25))
  );
  const cogZTimerRef = useRef<number | null>(null);

  const [nozzleZDraft, setNozzleZDraft] = useState<number>(
    Math.round(Math.abs(spec.toolhead.nozzle_offset_mm?.[2] ?? 62))
  );
  const nozzleZTimerRef = useRef<number | null>(null);

  const [carriagePlateThickDraft, setCarriagePlateThickDraft] = useState<number>(
    Number((spec.toolhead.carriage_plate_thickness_mm ?? 5.0).toFixed(1))
  );
  const carriagePlateThickTimerRef = useRef<number | null>(null);

  const [bodyWallThickDraft, setBodyWallThickDraft] = useState<number>(
    Number((spec.toolhead.body_wall_thickness_mm ?? 4.0).toFixed(1))
  );
  const bodyWallThickTimerRef = useRef<number | null>(null);

  const [hotendMountThickDraft, setHotendMountThickDraft] = useState<number>(
    Number((spec.toolhead.hotend_mount_thickness_mm ?? 8.0).toFixed(1))
  );
  const hotendMountThickTimerRef = useRef<number | null>(null);

  useEffect(() => {
    if (spec.toolhead.cog_offset_mm) {
      setCogYDraft(Math.round(Math.abs(spec.toolhead.cog_offset_mm[1] ?? 18)));
      setCogZDraft(Math.round(Math.abs(spec.toolhead.cog_offset_mm[2] ?? 25)));
    }
  }, [spec.toolhead.cog_offset_mm?.[1], spec.toolhead.cog_offset_mm?.[2]]);

  useEffect(() => {
    if (spec.toolhead.nozzle_offset_mm) {
      setNozzleZDraft(Math.round(Math.abs(spec.toolhead.nozzle_offset_mm[2] ?? 62)));
    } else {
      setNozzleZDraft(62);
    }
  }, [spec.toolhead.nozzle_offset_mm?.[2]]);

  useEffect(() => {
    setCarriagePlateThickDraft(
      Number((spec.toolhead.carriage_plate_thickness_mm ?? 5.0).toFixed(1))
    );
  }, [spec.toolhead.carriage_plate_thickness_mm]);

  useEffect(() => {
    setBodyWallThickDraft(Number((spec.toolhead.body_wall_thickness_mm ?? 4.0).toFixed(1)));
  }, [spec.toolhead.body_wall_thickness_mm]);

  useEffect(() => {
    setHotendMountThickDraft(
      Number((spec.toolhead.hotend_mount_thickness_mm ?? 8.0).toFixed(1))
    );
  }, [spec.toolhead.hotend_mount_thickness_mm]);

  useEffect(() => {
    return () => {
      if (cogYTimerRef.current != null) window.clearTimeout(cogYTimerRef.current);
      if (cogZTimerRef.current != null) window.clearTimeout(cogZTimerRef.current);
      if (nozzleZTimerRef.current != null) window.clearTimeout(nozzleZTimerRef.current);
      if (carriagePlateThickTimerRef.current != null)
        window.clearTimeout(carriagePlateThickTimerRef.current);
      if (bodyWallThickTimerRef.current != null) window.clearTimeout(bodyWallThickTimerRef.current);
      if (hotendMountThickTimerRef.current != null)
        window.clearTimeout(hotendMountThickTimerRef.current);
    };
  }, []);

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

  const specHotendMassG = Math.round(spec.toolhead.hotend.mass_g ?? 45);
  const assembledToolheadMassG = Math.round(
    sim?.geometry?.toolhead_structure?.assembled_toolhead_mass_g ??
      spec.toolhead.total_mass_g + specHotendMassG
  );

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

  const currentZProbe: ZProbeType = spec.toolhead.z_probe_type ?? "static_eddy";
  const zProbeCatalogEntry = catalog.z_probes?.find((p) => p.id === currentZProbe);
  const zProbeOptionMeta =
    Z_PROBE_OPTIONS.find((o) => o.id === currentZProbe) ?? Z_PROBE_OPTIONS[0];

  const toolheadPresets: ToolheadPresetItem[] =
    catalog.toolhead_presets && catalog.toolhead_presets.length > 0
      ? catalog.toolhead_presets
      : FALLBACK_TOOLHEAD_PRESETS;
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

  const useCadSuperelements = spec.use_cad_superelements ?? true;
  const activeCadSuperelements =
    sim?.geometry?.toolhead_structure?.cad_superelements_active ?? [];
  const activeToolheadCadCms = useCadSuperelements
    ? activeCadSuperelements.find((se) => se.category.toLowerCase().includes("toolhead"))
    : undefined;

  const effectiveHotendMassG =
    sim?.geometry?.toolhead_structure?.hotend_mass_g ?? specHotendMassG;
  const effectiveBodyMassG =
    sim?.geometry?.toolhead_structure?.toolhead_body_mass_g ?? spec.toolhead.total_mass_g;
  const effectiveCogMm =
    sim?.geometry?.toolhead_structure?.effective_cog_offset_mm ?? spec.toolhead.cog_offset_mm;

  const toolheadPresetIds = new Set(toolheadPresets.map((tp) => tp.preset_id));
  const extraBakedToolheads = Object.entries(bakedAssets.toolheads ?? {}).filter(
    ([assetId]) => !toolheadPresetIds.has(assetId)
  );

  const matchedToolheadPreset = toolheadPresets.find(
    (tp) =>
      tp.preset_id === spec.toolhead.preset_id &&
      Math.abs(tp.total_mass_g - spec.toolhead.total_mass_g) < 0.5 &&
      Math.round(Math.abs(tp.cog_offset_mm[1])) === specCogYAbs &&
      Math.round(Math.abs(tp.cog_offset_mm[2])) === specCogZAbs &&
      Math.round(Math.abs(tp.nozzle_offset_mm[2])) === specNozzleZAbs &&
      (Boolean(activeToolheadCadCms) || tp.mount_material === currentMountMaterial) &&
      (tp.z_probe_type ?? "static_eddy") === currentZProbe &&
      (!tp.body_construction ||
        spec.toolhead.body_construction == null ||
        tp.body_construction === spec.toolhead.body_construction)
  );
  const matchedBakedToolheadId =
    !matchedToolheadPreset &&
    spec.toolhead.preset_id &&
    extraBakedToolheads.some(([assetId]) => assetId === spec.toolhead.preset_id)
      ? spec.toolhead.preset_id
      : null;
  const selectedToolheadPresetVal = matchedToolheadPreset
    ? matchedToolheadPreset.preset_id
    : matchedBakedToolheadId ?? "__custom__";

  return (
            <div className="section-card" data-design-section="toolhead">
              <div className="section-title">
                <span>Toolhead CoG, Nozzle &amp; Mount</span>
                <span className="section-title-pills">
                  {activeToolheadCadCms && (
                    <span
                      className="badge-pill info"
                      title={formatCadCmsTooltip(activeToolheadCadCms)}
                    >
                      CAD CMS
                    </span>
                  )}
                  <span
                    className="sub-block-tag tabular-nums"
                    title="Toolhead Mass (excl. hotend) + Hotend Weight"
                  >
                    {assembledToolheadMassG}g total
                  </span>
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
                    if (!chosen) {
                      const baked = bakedAssets.toolheads?.[nextId];
                      if (!baked) return;
                      const prevName = spec.toolhead.name;
                      const bakedName = baked.name || humanizeId(nextId);
                      const bakedMassG = Math.max(10, Math.round((baked.mass_kg ?? 0.1) * 1000));
                      update(
                        (d) => {
                          d.toolhead.preset_id = nextId;
                          d.toolhead.name = bakedName;
                          d.toolhead.total_mass_g = bakedMassG;
                          if (baked.com_mm && baked.com_mm.length === 3) {
                            d.toolhead.cog_offset_mm = [
                              baked.com_mm[0],
                              baked.com_mm[1],
                              baked.com_mm[2],
                            ];
                          }
                          if (baked.reference_material_id) {
                            d.toolhead.carriage_mount_material = baked.reference_material_id;
                            d.toolhead.body_material = baked.reference_material_id;
                            d.toolhead.hotend_mount_material = baked.reference_material_id;
                            if (
                              MOUNT_MATERIAL_OPTIONS.some(
                                (o) => o.id === baked.reference_material_id
                              )
                            ) {
                              d.toolhead.mount_material =
                                baked.reference_material_id as ToolheadMountMaterial;
                            }
                          }
                          d.toolhead.nozzle_shield_mode = "auto";
                          d.use_cad_superelements = true;
                        },
                        `Toolhead Preset: ${prevName} → ${bakedName}`
                      );
                      return;
                    }
                    const prevName = spec.toolhead.name;
                    update(
                      (d) => {
                        d.toolhead.preset_id = chosen.preset_id;
                        d.toolhead.name = chosen.name;
                        d.toolhead.total_mass_g = chosen.total_mass_g;
                        d.toolhead.cog_offset_mm = [...chosen.cog_offset_mm];
                        d.toolhead.nozzle_offset_mm = [...chosen.nozzle_offset_mm];
                        d.toolhead.offset_datum_rail_mount = chosen.offset_datum_rail_mount ?? "front";
                        const chosenBodyMat = chosen.body_material ?? chosen.mount_material;
                        d.toolhead.mount_material = chosen.mount_material;
                        d.toolhead.carriage_mount_material = chosenBodyMat;
                        d.toolhead.body_material = chosenBodyMat;
                        d.toolhead.hotend_mount_material = chosenBodyMat;
                        if (chosen.body_construction != null) {
                          if (d.toolhead.body_construction !== chosen.body_construction) {
                            clearModelOverrideGroup(d, "toolhead_construction");
                          }
                          d.toolhead.body_construction = chosen.body_construction;
                        }
                        const nextProbe = chosen.z_probe_type ?? "static_eddy";
                        if ((d.toolhead.z_probe_type ?? "static_eddy") !== nextProbe) {
                          clearModelOverrideGroup(d, "z_probe");
                        }
                        d.toolhead.z_probe_type = nextProbe;
                        if (chosen.internal_stiffness_n_per_mm != null) {
                          d.toolhead.internal_stiffness_n_per_mm =
                            chosen.internal_stiffness_n_per_mm;
                        }
                        if (d.toolhead.extruder) {
                          d.toolhead.extruder.type = chosen.extruder_type ?? "direct_drive";
                          if (chosen.extruder_gear_ratio != null) {
                            d.toolhead.extruder.gear_ratio = chosen.extruder_gear_ratio;
                          }
                        }
                        if (chosen.envelope_mm != null && chosen.envelope_mm.length === 3) {
                          d.toolhead.envelope_mm = [...chosen.envelope_mm];
                        }
                        if (chosen.default_hotend_preset_id) {
                          const hp = catalog.hotend_presets?.find(
                            (h) => h.id === chosen.default_hotend_preset_id
                          );
                          if (hp && d.toolhead.hotend) {
                            d.toolhead.hotend.preset_id = hp.id;
                            d.toolhead.hotend.name = hp.name;
                            d.toolhead.hotend.mount_style = hp.mount_style as HotendMountStyle;
                            d.toolhead.hotend.mass_g = hp.mass_g;
                            d.toolhead.hotend.overall_length_mm = hp.overall_length_mm;
                            d.toolhead.hotend.heater_power_w = hp.heater_power_w;
                            d.toolhead.hotend.block_material = hp.block_material;
                            d.toolhead.hotend.block_mass_g = hp.block_mass_g;
                            d.toolhead.hotend.nozzle_material = hp.nozzle_material;
                            d.toolhead.hotend.melt_zone_length_mm = hp.melt_zone_length_mm;
                            d.toolhead.hotend.heatbreak_material = hp.heatbreak_material;
                            d.toolhead.hotend.heatbreak_length_mm = hp.heatbreak_length_mm;
                            d.toolhead.hotend.heatbreak_wall_mm = hp.heatbreak_wall_mm;
                            d.toolhead.hotend.heatbreak_id_mm = hp.heatbreak_id_mm;
                            d.toolhead.hotend.heatsink_thermal_resistance_k_w =
                              hp.heatsink_thermal_resistance_k_w;
                            d.toolhead.hotend.structural_stiffness_n_per_mm =
                              hp.structural_stiffness_n_per_mm;
                            d.toolhead.hotend.max_temp_c = hp.max_temp_c;
                          }
                        }
                        if (chosen.part_cooling_cfm != null && d.toolhead.hotend) {
                          d.toolhead.hotend.part_cooling_cfm = chosen.part_cooling_cfm;
                        }
                        if (chosen.cpap_blower_id != null) {
                          d.toolhead.cpap_blower_id = chosen.cpap_blower_id;
                        }
                        if (chosen.cpap_hose_id_mm != null) {
                          d.toolhead.cpap_hose_id_mm = chosen.cpap_hose_id_mm;
                        }
                        d.toolhead.shared_heatsink_cpap_duct = Boolean(
                          chosen.shared_heatsink_cpap_duct
                        );
                        if (chosen.belt_plane_z_offset_mm != null) {
                          d.toolhead.belt_plane_z_offset_mm = chosen.belt_plane_z_offset_mm;
                        }
                        d.toolhead.nozzle_shield_mode = "auto";
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
                  {toolheadPresets.map((tp) => {
                    const isSelectedTh = tp.preset_id === spec.toolhead.preset_id;
                    const displayMassG = isSelectedTh
                      ? fmtMassG(effectiveBodyMassG)
                      : String(tp.total_mass_g);
                    return (
                      <option key={tp.preset_id} value={tp.preset_id}>
                        {tp.name} ({displayMassG}g)
                      </option>
                    );
                  })}
                  {extraBakedToolheads.length > 0 && (
                    <optgroup label="Baked CAD Assets">
                      {extraBakedToolheads.map(([assetId, manifest]) => {
                        const isSelectedTh = assetId === spec.toolhead.preset_id;
                        const massG = Math.round((manifest.mass_kg ?? 0.1) * 1000);
                        const displayMassG = isSelectedTh
                          ? fmtMassG(effectiveBodyMassG)
                          : String(massG);
                        const label = manifest.name || humanizeId(assetId);
                        return (
                          <option key={assetId} value={assetId}>
                            {label} ({displayMassG}g)
                          </option>
                        );
                      })}
                    </optgroup>
                  )}
                </select>
                {(() => {
                  const presetMeta =
                    matchedToolheadPreset ??
                    toolheadPresets.find((tp) => tp.preset_id === spec.toolhead.preset_id);
                  if (!presetMeta?.default_hardware_summary) return null;
                  return (
                    <div className="hint-line">
                      <strong>Included by default:</strong> {presetMeta.default_hardware_summary}
                    </div>
                  );
                })()}
                <PresetAffectsStrip
                  visible={advancedMode}
                  items={[
                    { label: "Mass", value: `${fmtMassG(effectiveBodyMassG)} g` },
                    {
                      label: "CoG",
                      value: `(${Math.round(effectiveCogMm?.[0] ?? 0)}, ${Math.round(effectiveCogMm?.[1] ?? -18)}, ${Math.round(effectiveCogMm?.[2] ?? -25)}) mm`,
                    },
                    {
                      label: "Nozzle Z",
                      value: `${Math.round(spec.toolhead.nozzle_offset_mm?.[2] ?? -62)} mm`,
                    },
                    { label: "", value: toolheadPartMatShort(currentMountMaterial) },
                    { label: "", value: selectedBodyConstruction.short_name },
                    { label: "", value: zProbeOptionMeta.shortLabel },
                  ]}
                  suffix="(editable below)"
                />
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
                    const hadOv = Boolean(
                      spec.model_overrides?.z_probe &&
                        Object.values(spec.model_overrides.z_probe).some((v) => v != null)
                    );
                    update(
                      (d) => {
                        d.toolhead.z_probe_type = nextVal;
                        clearModelOverrideGroup(d, "z_probe");
                      },
                      `Z-Probe: ${prevLbl} → ${nextLbl}${hadOv ? " (override reset)" : ""}`
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
                {(() => {
                  const fb = FALLBACK_Z_PROBE_PARAMS[currentZProbe];
                  return (
                    <ModelOverrideControl
                      visible={advancedMode}
                      idPrefix="ov-z-probe"
                      title="Z-Probe Interface"
                      group="z_probe"
                      spec={spec}
                      sim={sim}
                      update={update}
                      fields={[
                        {
                          key: "compliance_factor",
                          label: "Stiffness Factor",
                          unit: "×",
                          min: 0.1,
                          max: 2.0,
                          step: 0.02,
                          decimals: 2,
                          fallbackBaseline:
                            zProbeCatalogEntry?.compliance_factor ?? fb.compliance_factor,
                          tooltip: "Carriage-to-toolhead interface stiffness multiplier (1.0 = rigid bolt, <1 = sliding Tap compliance)",
                        },
                        {
                          key: "cog_y_shift_mm",
                          label: "CoG Y Shift",
                          unit: "mm",
                          min: -50,
                          max: 50,
                          step: 0.5,
                          decimals: 1,
                          fallbackBaseline: zProbeCatalogEntry?.cog_y_shift_mm ?? fb.cog_y_shift_mm,
                          tooltip: "Additional forward Y center-of-gravity shift from probe mounting hardware",
                        },
                      ]}
                    />
                  );
                })()}
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
                    {fmtMassG(effectiveHotendMassG)}g) is added separately below.
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

              {activeToolheadCadCms ? (
                <div className="cad-driven-card">
                  <div className="cad-driven-header">
                    <span className="cad-driven-title">CAD-Modeled Toolhead Structure</span>
                    <span className="cad-driven-pills">
                      <span
                        className="badge-pill info"
                        title={formatCadCmsTooltip(activeToolheadCadCms)}
                      >
                        CAD CMS
                      </span>
                      <span className="sub-block-tag tabular-nums">
                        {formatElemCount(activeToolheadCadCms.num_elements)}{" "}
                        {activeToolheadCadCms.element_type || "Tet4"}
                      </span>
                    </span>
                  </div>

                  <div className="cad-driven-grid">
                    {(() => {
                      const nozzleBodyK =
                        sim?.static_fea?.toolhead_flex_breakdown?.body_stiffness_n_per_mm ??
                        sim?.geometry?.toolhead_structure?.stage_stiffnesses_n_per_mm?.body_n_per_mm;
                      const portKx = activeToolheadCadCms.port_stiffness_n_per_mm?.kx;
                      const portKy = activeToolheadCadCms.port_stiffness_n_per_mm?.ky;
                      const portKz = activeToolheadCadCms.port_stiffness_n_per_mm?.kz;
                      const portDetail =
                        Number.isFinite(portKx) && Number.isFinite(portKy)
                          ? `Nozzle-tip projected CAD chassis stiffness (harmonic XY bending). Raw Port-0→Port-1 flange stiffness: Kx ${Math.round(portKx!).toLocaleString()} / Ky ${Math.round(portKy!).toLocaleString()}${Number.isFinite(portKz) ? ` / Kz ${Math.round(portKz!).toLocaleString()}` : ""} N/mm.`
                          : formatCadCmsTooltip(activeToolheadCadCms);
                      return (
                        <div className="cad-driven-stat" title={portDetail}>
                          <span className="cad-driven-stat-label">Chassis @ Nozzle</span>
                          <span className="cad-driven-stat-value tabular-nums">
                            {nozzleBodyK != null && Number.isFinite(nozzleBodyK) && nozzleBodyK > 0
                              ? `${Math.round(nozzleBodyK).toLocaleString()} N/mm`
                              : Number.isFinite(portKx) && Number.isFinite(portKy)
                                ? `${Math.round(portKx!).toLocaleString()} / ${Math.round(portKy!).toLocaleString()} N/mm`
                                : "—"}
                          </span>
                        </div>
                      );
                    })()}
                    <div className="cad-driven-stat">
                      <span className="cad-driven-stat-label">CoG Offset</span>
                      <span className="cad-driven-stat-value">
                        {Math.abs(effectiveCogMm[0]) >= 0.05
                          ? `X ${effectiveCogMm[0] >= 0 ? "+" : ""}${effectiveCogMm[0].toFixed(1)} · `
                          : ""}
                        Y {effectiveCogMm[1] >= 0 ? "+" : ""}
                        {effectiveCogMm[1].toFixed(1)} · Z{" "}
                        {effectiveCogMm[2] >= 0 ? "+" : ""}
                        {effectiveCogMm[2].toFixed(1)} mm
                      </span>
                    </div>
                    <div className="cad-driven-stat">
                      <span className="cad-driven-stat-label">Nozzle Drop</span>
                      <span className="cad-driven-stat-value">{specNozzleZAbs} mm</span>
                    </div>
                    <div className="cad-driven-stat">
                      <span className="cad-driven-stat-label">Body Mass</span>
                      <span className="cad-driven-stat-value">
                        {fmtMassG(effectiveBodyMassG)} g
                      </span>
                    </div>
                  </div>

                  <div className="field-row">
                    <label htmlFor="dp-cad-toolhead-material">Chassis Material</label>
                    <select
                      id="dp-cad-toolhead-material"
                      value={currentBodyMat}
                      onChange={(e) => {
                        const nextMat = e.target.value;
                        update(
                          (d) => {
                            d.toolhead.body_material = nextMat;
                            d.toolhead.carriage_mount_material = nextMat;
                            d.toolhead.hotend_mount_material = nextMat;
                            if (MOUNT_MATERIAL_OPTIONS.some((o) => o.id === nextMat)) {
                              d.toolhead.mount_material = nextMat as ToolheadMountMaterial;
                            }
                          },
                          `CAD Toolhead Material: ${toolheadPartMatShort(currentBodyMat)} → ${toolheadPartMatShort(nextMat)}`
                        );
                      }}
                    >
                      {!TOOLHEAD_PART_MATERIAL_OPTIONS.some((opt) => opt.id === currentBodyMat) && (
                        <option value={currentBodyMat}>{materialLabel(currentBodyMat)}</option>
                      )}
                      {TOOLHEAD_PART_MATERIAL_OPTIONS.map((opt) => (
                        <option key={opt.id} value={opt.id}>
                          {opt.label}
                        </option>
                      ))}
                    </select>
                  </div>

                  <p className="cad-driven-note">
                    Chassis stiffness &amp; load path come directly from the baked 3D CAD
                    superelement; simulated mass &amp; CoG include the preset&apos;s dressed hardware
                    (extruder, carriage &amp; probe, excl. hotend) even when the 3D mesh renders only
                    the structural chassis. Switch preset to Custom (or toggle CAD Superelements off
                    in Advanced Mode) to use generic structure sliders.
                  </p>
                </div>
              ) : (
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
                        const hadOv = Boolean(
                          spec.model_overrides?.toolhead_construction &&
                            Object.values(spec.model_overrides.toolhead_construction).some((v) => v != null)
                        );
                        update(
                          (d) => {
                            d.toolhead.body_construction = nextVal;
                            d.toolhead.preset_id = "custom";
                            clearModelOverrideGroup(d, "toolhead_construction");
                          },
                          `Toolhead Construction: ${prevShort} → ${nextItem.short_name}${hadOv ? " (override reset)" : ""}`
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
                    <ModelOverrideControl
                      visible={advancedMode}
                      idPrefix="ov-toolhead-construction"
                      title="Body Construction"
                      group="toolhead_construction"
                      spec={spec}
                      sim={sim}
                      update={update}
                      fields={[
                        {
                          key: "body_stiffness_base_n_per_mm",
                          label: "Body Stiffness",
                          unit: "N/mm",
                          min: 50,
                          max: 20000,
                          step: 25,
                          decimals: 0,
                          fallbackBaseline: selectedBodyConstruction.body_stiffness_base_n_per_mm,
                          tooltip: "Baseline toolhead housing bending stiffness before material/wall scaling",
                        },
                        {
                          key: "torsion_mult",
                          label: "Torsion Mult",
                          unit: "×",
                          min: 0.2,
                          max: 4.0,
                          step: 0.05,
                          decimals: 2,
                          fallbackBaseline: selectedBodyConstruction.torsion_mult,
                          tooltip: "Torsional rigidity multiplier for multi-seam vs monoblock body geometry",
                        },
                      ]}
                    />
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

                  <div className="field-row">
                    <label htmlFor="dp-belt-plane-z-offset-mm">
                      <span>Belt-Plane Z Offset (mm)</span>
                      <span className="sub-block-tag">
                        {(spec.toolhead.belt_plane_z_offset_mm ?? 0) === 0
                          ? "Coplanar (0 mm)"
                          : `${(spec.toolhead.belt_plane_z_offset_mm ?? 0) > 0 ? "+" : ""}${spec.toolhead.belt_plane_z_offset_mm} mm`}
                      </span>
                    </label>
                    <input
                      id="dp-belt-plane-z-offset-mm"
                      type="number"
                      min={-40}
                      max={40}
                      step={0.5}
                      value={spec.toolhead.belt_plane_z_offset_mm ?? 0}
                      title="Vertical Z offset between belt drive plane and X-rail / toolhead CoG (0 mm = belt-coplanar like Monolith Toolhead, minimizing X-accel roll coupling)"
                      onChange={(e) => {
                        const v = Number(e.target.value);
                        if (!Number.isFinite(v)) return;
                        const clamped = Math.max(-40, Math.min(40, v));
                        const prev = spec.toolhead.belt_plane_z_offset_mm ?? 0;
                        if (clamped === prev) return;
                        update(
                          (d) => {
                            d.toolhead.belt_plane_z_offset_mm = clamped;
                          },
                          `Belt-Plane Z Offset: ${prev} → ${clamped} mm`
                        );
                      }}
                    />
                    {showHints && (
                      <div className="hint-line">
                        Aligning the belt plane with toolhead CoG (≈ 0 mm) minimizes X-axis roll torque
                        and resonance splitting.
                      </div>
                    )}
                  </div>
                </div>
                </details>
              )}
            </div>

  );
};
