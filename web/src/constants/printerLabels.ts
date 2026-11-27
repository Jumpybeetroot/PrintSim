import {
  AwdSyncMode,
  BracingType,
  FrontBraceType,
  InputShaperType,
  PrinterSpecData,
  TopBraceType,
} from "../types";

export const BRACING_LABELS: Record<BracingType, string> = {
  none: "None (Open Frame)",
  cast_corner_brackets: "Cast Al Corner Brackets (HBLFSN5/6/8 — CAD)",
  cnc_corner_gussets: "CNC Al 5-Hole Corner Gussets (5–6mm — CAD)",
  rear_flat_bar_x: "Bolted Rear Flat-Bar X-Brace (25×5mm Al)",
  side_flat_bar_x: "Bolted Side Flat-Bar X-Braces (25×5mm Al)",
  rear_and_side_flat_bars: "Bolted Rear + Side Flat-Bar X-Braces (25×5mm)",
  gussets_and_flat_bars: "CNC Corner Gussets + Rear/Side Flat-Bar X-Braces",
};

const LEGACY_BRACING_MAP: Record<string, BracingType> = {
  corner_gussets: "cnc_corner_gussets",
  top_cross: "cnc_corner_gussets",
  rear_cross: "rear_flat_bar_x",
  side_cross: "side_flat_bar_x",
  rear_and_sides: "rear_and_side_flat_bars",
  full_truss: "gussets_and_flat_bars",
};

export function normalizeBracingType(raw: string | undefined | null): BracingType {
  if (!raw) return "cnc_corner_gussets";
  const k = raw.trim().toLowerCase();
  if (k in BRACING_LABELS) return k as BracingType;
  return LEGACY_BRACING_MAP[k] ?? "cnc_corner_gussets";
}

export const TOP_BRACE_LABELS: Record<TopBraceType, string> = {
  none: "None (Open Top)",
  cnc_halo_ring: "CNC Halo Ring (5mm Al, 50mm flange)",
  rear_v_and_front_corners: "Rear-V + Front Corners (25×5mm Bar)",
  corner_45_bars: "4× Top Corner 45° Bars (25×5mm)",
};

export const FRONT_BRACE_LABELS: Record<FrontBraceType, string> = {
  none: "None (Open Front Door)",
  cnc_portal_frame: "CNC Portal Frame (5mm Al, 55mm border)",
  header_and_skirt: "Upper Header + Lower Skirt (5mm Al)",
  corner_45_bars: "4× Front Corner 45° Bars (25×5mm)",
};

export const ARCH_LABELS: Record<PrinterSpecData["architecture"], string> = {
  corexy: "CoreXY",
  cartesian_bedslinger: "Cartesian Bedslinger",
  delta: "Delta",
};

export const Z_DRIVE_LABELS: Record<string, string> = {
  leadscrew_triple: "Fixed Gantry (Trident)",
  belted_quad_z: "Flying Gantry (Voron 2.4)",
  leadscrew_dual: "Fixed Gantry (Dual-Z)",
  leadscrew_dual_belt_sync: "Fixed Gantry (Belt-Synced Dual-Z)",
  leadscrew_single: "Fixed Gantry (Single-Z)",
};

export const BELT_LABELS: Record<PrinterSpecData["gantry"]["belt"]["type"], string> = {
  gt2_6mm: "GT2 6mm",
  gt2_9mm: "GT2 9mm",
  gt2_10mm: "GT2 10mm (Monolith)",
  gt3_10mm: "GT3 (2MGT) 10mm",
  gt2_12mm: "GT2 12mm",
  gt2_15mm: "GT2 15mm",
  gt2_dual_9mm: "Dual 2GT 9mm (18mm)",
  "3gt_9mm": "Gates 3GT 9mm",
  "3gt_12mm": "Gates 3GT 12mm",
  "3gt_15mm": "Gates 3GT 15mm",
  at3_10mm: "AT3 10mm Steel/TPU",
  at3_16mm: "AT3 16mm Steel/TPU",
  at3_gen3_16mm: "AT3 Gen III 16mm",
  htd3m_15mm: "HTD 3M 15mm Steel/TPU",
  at5_16mm: "AT5 16mm Heavy Steel",
};

export const CHAMBER_HEATER_LABELS: Record<string, string> = {
  none: "None (Passive)",
  monolith_100w_ptc: "Monolith 100W 24V PTC",
  monolith_500w_ptc: "Monolith 500W AC PTC",
};

export const CPAP_BLOWER_LABELS: Record<string, string> = {
  none: "Onboard Fans",
  ws7040_24v: "WS7040 24V (5.8 dm³/s)",
  wm9290_24v: "WM9290 24V (9.2 dm³/s)",
  wm7575_220vac: "WM7575 220VAC (11.6 dm³/s)",
};

export const SHAPER_LABELS: Record<InputShaperType, string> = {
  auto: "Auto (Klipper)",
  none: "None",
  zv: "ZV",
  mzv: "MZV",
  ei: "EI",
  "2hump_ei": "2HUMP_EI",
  "3hump_ei": "3HUMP_EI",
  zvdd: "ZVDD",
  smooth_zv: "SMOOTH_ZV (Kalico)",
  smooth_ei: "SMOOTH_EI (Kalico)",
  smooth_flatness: "SMOOTH_FLATNESS (Kalico C4 Beta)",
};

export const AWD_SYNC_LABELS: Record<AwdSyncMode, string> = {
  none: "Off (Independent)",
  torque_mirror: "Stage 1: Torque Mirror",
  position_equalizer: "Stage 2: Position + Equalizer",
};

export function getBeltDefaults(
  beltType: PrinterSpecData["gantry"]["belt"]["type"]
): { pitch_mm: number; core: PrinterSpecData["gantry"]["belt"]["core"] } {
  if (beltType.startsWith("gt2_") || beltType.startsWith("gt3_")) {
    return { pitch_mm: 2.0, core: "fiberglass" };
  }
  if (beltType.startsWith("3gt_")) {
    return { pitch_mm: 3.0, core: "fiberglass" };
  }
  if (beltType.startsWith("at5_")) {
    return { pitch_mm: 5.0, core: "steel" };
  }
  // at3_* and htd3m_*
  return { pitch_mm: 3.0, core: "steel" };
}

export function getBeltTensionSliderMax(
  beltType: PrinterSpecData["gantry"]["belt"]["type"]
): number {
  if (beltType === "gt2_6mm" || beltType === "gt2_9mm") {
    return 120;
  }
  return 250;
}
