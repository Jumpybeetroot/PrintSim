import {
  AwdSyncMode,
  FrontBraceType,
  InputShaperType,
  PrinterSpecData,
  TopBraceType,
} from "../types";

export const BRACING_LABELS: Record<PrinterSpecData["frame"]["bracing"], string> = {
  none: "None (Open Frame)",
  corner_gussets: "Corner Gussets",
  top_cross: "Top Frame X-Brace",
  rear_cross: "Rear X-Brace + Top",
  side_cross: "Side X-Braces (Left/Right)",
  rear_and_sides: "Rear + Side X-Braces (Open Top)",
  full_truss: "Full 3D Truss (All Faces)",
};

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
  if (beltType.startsWith("gt2_")) {
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
