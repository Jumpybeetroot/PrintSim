import React, { useEffect, useRef, useState } from "react";
import { humanizeId } from "../utils/displayText";
import {
  CadSuperelementSummary,
  CatalogResponse,
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
  { id: "titanium_grade5", label: "SLM / CNC Ti-6Al-4V Gr.5 (113.8 GPa)", shortLabel: "Ti-6Al-4V" },
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
    total_mass_g: 365,
    cog_offset_mm: [0, -22, -28],
    nozzle_offset_mm: [0, -26, -66],
    mount_material: "abs_printed",
    body_construction: "multi_piece_clamshell",
    internal_stiffness_n_per_mm: 350,
    z_probe_type: "static_eddy",
  },
  {
    preset_id: "stealthburner_tap",
    name: "Voron StealthBurner + Printed Tap",
    total_mass_g: 385,
    cog_offset_mm: [0, -22, -28],
    nozzle_offset_mm: [0, -26, -66],
    mount_material: "abs_printed",
    body_construction: "multi_piece_clamshell",
    internal_stiffness_n_per_mm: 350,
    z_probe_type: "printed_tap",
  },
  {
    preset_id: "xol_compact",
    name: "XOL / A4T Compact Dual-5015",
    total_mass_g: 235,
    cog_offset_mm: [0, -15, -18],
    nozzle_offset_mm: [0, -18, -54],
    mount_material: "pa12_cf_printed",
    body_construction: "compact_cage",
    internal_stiffness_n_per_mm: 650,
    z_probe_type: "static_eddy",
  },
  {
    preset_id: "a4t_beacon",
    name: "A4T / XOL 2 + Static Beacon",
    total_mass_g: 215,
    cog_offset_mm: [0, -13, -15],
    nozzle_offset_mm: [0, -16, -52],
    mount_material: "ppa_cf_printed",
    body_construction: "compact_cage",
    internal_stiffness_n_per_mm: 1150,
    z_probe_type: "static_eddy",
  },
  {
    preset_id: "dragon_burner",
    name: "Dragon Burner / Rapid Burner Compact",
    total_mass_g: 195,
    cog_offset_mm: [0, -14, -16],
    nozzle_offset_mm: [0, -16, -50],
    mount_material: "pa12_cf_printed",
    body_construction: "compact_cage",
    internal_stiffness_n_per_mm: 720,
    z_probe_type: "static_eddy",
  },
  {
    preset_id: "archetype_cpap",
    name: "Archetype CPAP Ultra-Low CoG",
    total_mass_g: 175,
    cog_offset_mm: [0, -11, -12],
    nozzle_offset_mm: [0, -14, -48],
    mount_material: "aluminum_6061_t6",
    body_construction: "plate_sandwich_cf",
    internal_stiffness_n_per_mm: 1800,
    z_probe_type: "static_eddy",
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
  },
  {
    preset_id: "heavy_uhf_long",
    name: "Heavy Long-Nozzle UHF Toolhead",
    total_mass_g: 440,
    cog_offset_mm: [0, -26, -34],
    nozzle_offset_mm: [0, -28, -78],
    mount_material: "abs_printed",
    body_construction: "multi_piece_clamshell",
    internal_stiffness_n_per_mm: 250,
    z_probe_type: "static_eddy",
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
                    <div className="cad-driven-stat">
                      <span className="cad-driven-stat-label">CAD Stiffness (Kx / Ky)</span>
                      <span className="cad-driven-stat-value">
                        {Number.isFinite(activeToolheadCadCms.port_stiffness_n_per_mm?.kx)
                          ? Math.round(activeToolheadCadCms.port_stiffness_n_per_mm.kx).toLocaleString()
                          : "—"}{" "}
                        /{" "}
                        {Number.isFinite(activeToolheadCadCms.port_stiffness_n_per_mm?.ky)
                          ? Math.round(activeToolheadCadCms.port_stiffness_n_per_mm.ky).toLocaleString()
                          : "—"}{" "}
                        N/mm
                      </span>
                    </div>
                    <div className="cad-driven-stat">
                      <span className="cad-driven-stat-label">CoG Offset</span>
                      <span className="cad-driven-stat-value">
                        {Math.abs(effectiveCogMm[0]) >= 0.05
                          ? `X ${effectiveCogMm[0] >= 0 ? "+" : ""}${effectiveCogMm[0].toFixed(1)} · `
                          : ""}
                        Y {Math.abs(effectiveCogMm[1]).toFixed(1)} · Z{" "}
                        {Math.abs(effectiveCogMm[2]).toFixed(1)} mm
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
                    Geometry, backplate, ribs &amp; CoG come directly from the baked 3D CAD
                    superelement. Switch preset to Custom (or toggle CAD Superelements off in
                    Advanced Mode) to use generic structure sliders.
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
                </div>
                </details>
              )}
            </div>

  );
};
