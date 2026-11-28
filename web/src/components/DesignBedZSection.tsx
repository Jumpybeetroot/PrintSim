import React, { useEffect, useRef, useState } from "react";
import {
  BedMountType,
  CatalogResponse,
  PrinterSpecData,
  SimulationResponse,
  ZLeadscrewType,
  ZRailProfileType,
} from "../types";
import { BED_MOUNT_STANDOFF_HEIGHT_MM } from "../viewport/bedMounts";
import {
  ModelOverrideControl,
  clearModelOverrideGroup,
} from "./ModelOverrideControl";

const FALLBACK_Z_RAIL_PROFILES: Array<{ id: ZRailProfileType; name: string }> = [
  { id: "mgn12_rail", name: "MGN12H Linear Rail (12×8mm)" },
  { id: "mgn9_rail", name: "MGN9H Linear Rail (9×6.5mm)" },
  { id: "steel_rod_8mm", name: "8mm Smooth Steel Rods" },
  { id: "steel_rod_10mm", name: "10mm Smooth Steel Rods" },
  { id: "steel_rod_12mm", name: "12mm Smooth Steel Rods" },
  { id: "none", name: "None (Leadscrew Guided)" },
];

interface ZDriveOptionMeta {
  id: ZLeadscrewType;
  name: string;
  kind: "auto" | "leadscrew" | "ballscrew" | "belt";
  k_nut_n_per_um?: number;
  k_thrust_n_per_um?: number;
  k_drive_n_per_um_per_corner?: number | null;
  wobble_decoupled?: boolean;
  self_locking?: boolean;
  unpowered_drop_risk?: boolean;
  backlash_um?: number;
  gear_ratio?: number;
  cost_per_axis_usd?: number;
  notes?: string;
}

const FALLBACK_Z_DRIVE_MECHANISMS: ZDriveOptionMeta[] = [
  { id: "auto", name: "Auto (Match Z Architecture)", kind: "auto" },
  {
    id: "tr8x4_pom",
    name: "TR8×4 Trident POM Nut",
    kind: "leadscrew",
    k_nut_n_per_um: 11.0,
    k_thrust_n_per_um: 18.0,
    wobble_decoupled: false,
    self_locking: true,
    unpowered_drop_risk: false,
    backlash_um: 28,
    notes: "Standard 4mm-lead Trident leadscrew with self-locking POM nut.",
  },
  {
    id: "tr8x4_wobblex",
    name: "TR8×4 + WobbleX Decoupler",
    kind: "leadscrew",
    k_nut_n_per_um: 10.2,
    k_thrust_n_per_um: 18.0,
    wobble_decoupled: true,
    self_locking: true,
    unpowered_drop_risk: false,
    backlash_um: 18,
    notes: "Floating kinematic WobbleX nut decouples XY screw runout while staying self-locking.",
  },
  {
    id: "tr8x8_pom",
    name: "TR8×8 POM Nut",
    kind: "leadscrew",
    k_nut_n_per_um: 6.5,
    k_thrust_n_per_um: 14.0,
    wobble_decoupled: false,
    self_locking: false,
    unpowered_drop_risk: true,
    backlash_um: 45,
    notes: "Fast 8mm-lead 4-start screw; lower friction can back-drive heavy beds when unpowered.",
  },
  {
    id: "tr8x8_brass",
    name: "TR8×8 Brass Nut",
    kind: "leadscrew",
    k_nut_n_per_um: 38.0,
    k_thrust_n_per_um: 22.0,
    wobble_decoupled: false,
    self_locking: false,
    unpowered_drop_risk: true,
    backlash_um: 35,
    notes: "Budget brass leadscrew nut with higher axial backlash and runout coupling.",
  },
  {
    id: "sfu1204_bk10",
    name: "SFU1204 C7 Ballscrew + BK10",
    kind: "ballscrew",
    k_nut_n_per_um: 85.0,
    k_thrust_n_per_um: 150.0,
    wobble_decoupled: false,
    self_locking: false,
    unpowered_drop_risk: true,
    backlash_um: 8,
    notes: "High-stiffness 12mm C7 rolled ballscrew on BK10 angular-contact bearings; back-drives unpowered.",
  },
  {
    id: "sfu1204_wobblex",
    name: "SFU1204 + BK10 + WobbleX",
    kind: "ballscrew",
    k_nut_n_per_um: 72.0,
    k_thrust_n_per_um: 150.0,
    wobble_decoupled: true,
    self_locking: false,
    unpowered_drop_risk: true,
    backlash_um: 6,
    notes: "SFU1204 ballscrew axial rigidity paired with a WobbleX XY runout decoupler.",
  },
  {
    id: "sfu1605_fk12",
    name: "SFU1605 Heavy Ballscrew + FK12/BK12",
    kind: "ballscrew",
    k_nut_n_per_um: 145.0,
    k_thrust_n_per_um: 240.0,
    wobble_decoupled: false,
    self_locking: false,
    unpowered_drop_risk: true,
    backlash_um: 5,
    notes: "Industrial 16mm ballscrew + heavy FK12 flange support for large high-mass beds.",
  },
  {
    id: "gt2_6mm_z_belt",
    name: "GT2 6mm Z-Belt",
    kind: "belt",
    k_drive_n_per_um_per_corner: 0.18,
    wobble_decoupled: true,
    self_locking: false,
    unpowered_drop_risk: true,
    backlash_um: 18,
    notes: "Stock Voron 2.4 6mm GT2 Z-belt drive — zero screw wobble.",
  },
  {
    id: "gt2_9mm_z_belt",
    name: "GT2 9mm Z-Belt",
    kind: "belt",
    k_drive_n_per_um_per_corner: 0.27,
    wobble_decoupled: true,
    self_locking: false,
    unpowered_drop_risk: true,
    backlash_um: 12,
    notes: "Upgraded 9mm GT2 Z-belt with +50% tensile stiffness and zero Z-wobble.",
  },
  {
    id: "gt2_9mm_planetary_z",
    name: "9mm Belted-Z (5:1 Planetary)",
    kind: "belt",
    k_drive_n_per_um_per_corner: 0.31,
    wobble_decoupled: true,
    self_locking: false,
    unpowered_drop_risk: true,
    backlash_um: 14,
    gear_ratio: 5,
    notes: "9mm belted Z with compact 5:1 planetary gearbox for high thrust and zero screw wobble.",
  },
  {
    id: "gt2_9mm_worm_z",
    name: "9mm Belted-Z (Self-Locking Worm / 20:1)",
    kind: "belt",
    k_drive_n_per_um_per_corner: 0.34,
    wobble_decoupled: true,
    self_locking: true,
    unpowered_drop_risk: false,
    backlash_um: 16,
    gear_ratio: 20,
    notes: "Self-locking 20:1 worm-geared 9mm belted Z — zero Z-wobble and zero bed drop on power loss.",
  },
  {
    id: "kevinakasam_belted_z",
    name: "KevinAkaSam Belted-Z (80:16 + Keybak)",
    kind: "belt",
    k_drive_n_per_um_per_corner: 0.21,
    wobble_decoupled: true,
    self_locking: false,
    unpowered_drop_risk: false,
    backlash_um: 15,
    gear_ratio: 5,
    notes: "80:16 belted-Z conversion with Keybak spring counterbalance to prevent unpowered bed drop.",
  },
];

const FALLBACK_BED_MOUNT_PARAMS: Record<
  BedMountType,
  {
    k_axial_n_per_mm: number;
    k_lateral_n_per_mm: number;
    k_restraint_thermal_n_per_mm: number;
    damping_delta: number;
    standoff_height_mm: number;
  }
> = {
  kinematic_3pt: { k_axial_n_per_mm: 32000, k_lateral_n_per_mm: 24000, k_restraint_thermal_n_per_mm: 0, damping_delta: 0, standoff_height_mm: 16 },
  whopping_pochard_kinematic: { k_axial_n_per_mm: 24000, k_lateral_n_per_mm: 16000, k_restraint_thermal_n_per_mm: 0, damping_delta: 0.002, standoff_height_mm: 14 },
  trident_3pt_standoffs: { k_axial_n_per_mm: 18000, k_lateral_n_per_mm: 9500, k_restraint_thermal_n_per_mm: 3800, damping_delta: 0.003, standoff_height_mm: 10 },
  cnc_kelvin_magnetic: { k_axial_n_per_mm: 42000, k_lateral_n_per_mm: 30000, k_restraint_thermal_n_per_mm: 0, damping_delta: 0.001, standoff_height_mm: 15 },
  captive_belleville_kinematic: { k_axial_n_per_mm: 52000, k_lateral_n_per_mm: 36000, k_restraint_thermal_n_per_mm: 0, damping_delta: 0.002, standoff_height_mm: 16 },
  dual_leaf_ti_kinematic: { k_axial_n_per_mm: 38000, k_lateral_n_per_mm: 26000, k_restraint_thermal_n_per_mm: 0, damping_delta: 0.003, standoff_height_mm: 14 },
  spherical_3pt_ratrig: { k_axial_n_per_mm: 30000, k_lateral_n_per_mm: 20000, k_restraint_thermal_n_per_mm: 1500, damping_delta: 0.002, standoff_height_mm: 15 },
  craz_kinematic_ge5c: { k_axial_n_per_mm: 34000, k_lateral_n_per_mm: 24000, k_restraint_thermal_n_per_mm: 0, damping_delta: 0.002, standoff_height_mm: 15 },
  rigid_4pt: { k_axial_n_per_mm: 65000, k_lateral_n_per_mm: 40000, k_restraint_thermal_n_per_mm: 80000, damping_delta: -0.002, standoff_height_mm: 12 },
  floating_4pt: { k_axial_n_per_mm: 12000, k_lateral_n_per_mm: 6000, k_restraint_thermal_n_per_mm: 2800, damping_delta: 0.002, standoff_height_mm: 12 },
  silicone_4pt: { k_axial_n_per_mm: 48, k_lateral_n_per_mm: 16, k_restraint_thermal_n_per_mm: 32, damping_delta: 0.008, standoff_height_mm: 16 },
  silicone_4pt_clamped: { k_axial_n_per_mm: 2000, k_lateral_n_per_mm: 400, k_restraint_thermal_n_per_mm: 64, damping_delta: 0.006, standoff_height_mm: 16 },
  springs_4pt: { k_axial_n_per_mm: 16, k_lateral_n_per_mm: 4.5, k_restraint_thermal_n_per_mm: 9, damping_delta: 0.004, standoff_height_mm: 20 },
};

const FALLBACK_Z_DRIVE_BY_ID: Record<string, ZDriveOptionMeta> = Object.fromEntries(
  FALLBACK_Z_DRIVE_MECHANISMS.map((m) => [m.id, m])
);

const BED_MOUNT_OPTIONS: Array<{
  id: BedMountType;
  group: "kinematic" | "standard";
  label: string;
  shortLabel: string;
  hint: string;
  points: 3 | 4;
  tapRated: boolean;
  tapLabel: string;
  tapTitle: string;
  tacoLabel: string;
  tacoTone: "ok" | "info" | "danger";
  tacoTitle: string;
}> = [
  {
    id: "captive_belleville_kinematic",
    group: "kinematic",
    label: "Super-Stiff Captive Belleville Kinematic (Tap-Rated)",
    shortLabel: "Captive Belleville Kinematic",
    hint: "Captive shoulder bolt + stacked Belleville disc springs preload hardened ball in CNC V-groove — ultra-high axial stiffness and zero thermal taco.",
    points: 3,
    tapRated: true,
    tapLabel: "Tap-Rated ✓",
    tapTitle: "High-preload captive Belleville stack locks the ball solidly under Voron Tap probing",
    tacoLabel: "Zero Taco",
    tacoTone: "ok",
    tacoTitle: "Exact 3-point kinematic V-grooves allow free radial thermal expansion with zero plate buckling",
  },
  {
    id: "dual_leaf_ti_kinematic",
    group: "kinematic",
    label: "Dual-Blade Ti/301 Leaf-Spring Kinematic (Annex / Heavy Pochard)",
    shortLabel: "Dual-Leaf Ti Kinematic",
    hint: "Twin stacked Titanium / 301 spring-steel leaf flexures double vertical & lateral stiffness while eliminating thermal taco.",
    points: 3,
    tapRated: true,
    tapLabel: "Tap-Rated ✓",
    tapTitle: "Dual-blade parallel Ti/301 flexures provide zero-play rigidity under Voron Tap probing",
    tacoLabel: "Zero Taco",
    tacoTone: "ok",
    tacoTitle: "Radial parallel leaf flexures absorb bed thermal expansion smoothly without taco buckling",
  },
  {
    id: "whopping_pochard_kinematic",
    group: "kinematic",
    label: "Whopping Pochard Leaf-Spring (Tap-Rated)",
    shortLabel: "Whopping Pochard",
    hint: "Zero-play kinematic leaf-spring mount — eliminates thermal taco while staying rigid under Voron Tap probing.",
    points: 3,
    tapRated: true,
    tapLabel: "Tap-Rated ✓",
    tapTitle: "Bolted 301 spring-steel leaf flexures resist vertical Tap probing with zero play",
    tacoLabel: "Zero Taco",
    tacoTone: "ok",
    tacoTitle: "Radial leaf-spring flexures absorb bed thermal expansion without taco buckling",
  },
  {
    id: "craz_kinematic_ge5c",
    group: "kinematic",
    label: "Monolith Cra-Z 2+2+2 GE5C",
    shortLabel: "Monolith Cra-Z",
    hint: "Monolith SLM 2+2+2 double-wishbone bed suspension with GE5C spherical bearings close to Z rails — zero thermal taco bow and Tap-rated vertical stiffness.",
    points: 3,
    tapRated: true,
    tapLabel: "Tap-Rated ✓",
    tapTitle: "Gravity-preloaded GE5C double-wishbone links resist vertical Tap probing with zero play",
    tacoLabel: "Zero Taco",
    tacoTone: "ok",
    tacoTitle: "2+2+2 double-wishbone GE5C geometry decouples radial thermal growth with zero plate buckling",
  },
  {
    id: "spherical_3pt_ratrig",
    group: "kinematic",
    label: "RatRig V-Core 3/4 3-Pt Spherical Pivot Kinematic",
    shortLabel: "RatRig 3-Pt Spherical",
    hint: "Three spherical ball-socket pivots with radial expansion slots — proven Tap-rated stiffness on large 300–500mm beds.",
    points: 3,
    tapRated: true,
    tapLabel: "Tap-Rated ✓",
    tapTitle: "Hard-bolted spherical bearing pivots resist vertical Tap probing forces cleanly",
    tacoLabel: "Near-Zero Taco",
    tacoTone: "ok",
    tacoTitle: "Spherical pivots + radial expansion slots relieve thermal growth with near-zero plate warp",
  },
  {
    id: "cnc_kelvin_magnetic",
    group: "kinematic",
    label: "CNC N48SH Magnetic Kelvin Kinematic (MRW / ChaoticLab)",
    shortLabel: "CNC Kelvin Magnetic",
    hint: "True 3-2-1 cone/V-groove/flat CNC billet Kelvin mount with high-temp N48SH magnets — zero thermal taco and removable bed.",
    points: 3,
    tapRated: false,
    tapLabel: "Magnetic — Not Tap-Rated",
    tapTitle: "Magnetic preload can lift or shift under Voron Tap nozzle probing forces",
    tacoLabel: "Zero Taco",
    tacoTone: "ok",
    tacoTitle: "True 3-2-1 Kelvin constraint (cone + V-groove + flat) allows completely stress-free thermal expansion",
  },
  {
    id: "kinematic_3pt",
    group: "kinematic",
    label: "3-Pt Kinematic (Maxwell)",
    shortLabel: "3-Pt Kinematic",
    hint: "Zero thermal taco buckling at high bed temps with high axial/lateral stiffness.",
    points: 3,
    tapRated: false,
    tapLabel: "Magnetic — Not Tap-Rated",
    tapTitle: "Magnetic preload can lift or shift under Voron Tap nozzle probing forces",
    tacoLabel: "Zero Taco",
    tacoTone: "ok",
    tacoTitle: "Exact 6-DOF Maxwell radial V-grooves allow unconstrained radial thermal expansion",
  },
  {
    id: "trident_3pt_standoffs",
    group: "standard",
    label: "3-Pt Trident Bakelite Standoffs + GE5C",
    shortLabel: "Trident 3-Pt Bakelite",
    hint: "Stock Voron/Siboor Trident 3-point Bakelite thermal standoffs on 2020 T-subframe with GE5C spherical Z-joints.",
    points: 3,
    tapRated: true,
    tapLabel: "Tap-Rated ✓",
    tapTitle: "Hard-bolted phenolic Bakelite columns stay rigid under Voron Tap probing",
    tacoLabel: "Low Slotted Relief",
    tacoTone: "info",
    tacoTitle: "Rear-center fixed anchor with two front slotted thermal-relief washers",
  },
  {
    id: "floating_4pt",
    group: "standard",
    label: "4-Pt Floating (1 Fixed + 3 Slotted)",
    shortLabel: "4-Pt Floating",
    hint: "Stiff vertically while slotted corners relieve thermal expansion without buckling.",
    points: 4,
    tapRated: true,
    tapLabel: "Tap-Rated ✓",
    tapTitle: "Solid vertical shoulder-bolt support handles nozzle probing loads cleanly",
    tacoLabel: "Low Slotted Relief",
    tacoTone: "info",
    tacoTitle: "1 fixed locator anchor + 3 radially slotted glide tabs relieve expansion",
  },
  {
    id: "rigid_4pt",
    group: "standard",
    label: "4-Pt Rigid (Hard Spacers)",
    shortLabel: "4-Pt Rigid",
    hint: "Stiffest mechanically, but over-constrains thermal expansion and tacos above 80°C.",
    points: 4,
    tapRated: true,
    tapLabel: "Tap-Rated ✓",
    tapTitle: "Solid metal hex standoffs provide highest axial probing rigidity",
    tacoLabel: "High Thermal Taco Risk",
    tacoTone: "danger",
    tacoTitle: "Over-constrained 4-corner hard bolting buckles the bed plate above 80°C",
  },
  {
    id: "silicone_4pt",
    group: "standard",
    label: "4-Pt Silicone Spacers",
    shortLabel: "4-Pt Silicone",
    hint: "Thermally decoupled with moderate damping; softer than kinematic/rigid mounts.",
    points: 4,
    tapRated: false,
    tapLabel: "Compliant — Not Tap-Rated",
    tapTitle: "70A elastomer compresses under Tap probing force, degrading Z-offset repeatability",
    tacoLabel: "Low Compliant Relief",
    tacoTone: "info",
    tacoTitle: "Compliant silicone columns shear laterally to relieve thermal expansion",
  },
  {
    id: "silicone_4pt_clamped",
    group: "standard",
    label: "4-Pt Silicone Spacers, Torqued Through-Bolt (VzBot)",
    shortLabel: "4-Pt Silicone Clamped",
    hint: "Silicone spacers clamped by M3x50 through-bolts: a preloaded joint that stays stiff for vibration while remaining leveling-adjustable.",
    points: 4,
    tapRated: false,
    tapLabel: "Compliant — Not Tap-Rated",
    tapTitle: "Large probing loads can exceed the knob preload and fall back to elastomer compliance",
    tacoLabel: "Low Compliant Relief",
    tacoTone: "info",
    tacoTitle: "Thermal growth slips through the M3 hole clearance and shears the silicone",
  },
  {
    id: "springs_4pt",
    group: "standard",
    label: "4-Pt Bed Springs (Yellow)",
    shortLabel: "4-Pt Springs",
    hint: "Low-cost leveling springs; softest mount with highest bed-bounce under Y accel.",
    points: 4,
    tapRated: false,
    tapLabel: "Compliant — Not Tap-Rated",
    tapTitle: "Low axial spring stiffness deflects heavily under nozzle probing forces",
    tacoLabel: "Low Compliant Relief",
    tacoTone: "info",
    tacoTitle: "Compliant coil springs absorb radial expansion with minimal plate buckling",
  },
];

const PLATE_MATERIAL_FALLBACKS: Array<{ id: string; name: string }> = [
  { id: "mic6_aluminum", name: "MIC-6 Cast Aluminum Plate" },
  { id: "atp5_aluminum", name: "ATP-5 Cast Aluminum Plate" },
  { id: "aluminum_6061_t6", name: "Aluminum 6061-T6" },
  { id: "aluminum_7075_t6", name: "Aluminum 7075-T6" },
  { id: "carbon_fiber_plate", name: "Carbon Fiber 3K Plate" },
  { id: "steel_4140", name: "4140 Alloy Steel" },
];

const PLATE_MATERIAL_IDS = new Set([
  "mic6_aluminum",
  "atp5_aluminum",
  "aluminum_6061_t6",
  "aluminum_7075_t6",
  "carbon_fiber_plate",
  "steel_4140",
]);

function bedZTopologyMeta(arch: PrinterSpecData["architecture"], zDrive: string): {
  badge: string;
  caption: string;
} {
  if (arch === "cartesian_bedslinger") {
    return { badge: "Moving Y-Bed", caption: "Y-axis bed carriage on linear rails/rods" };
  }
  if (arch === "delta") {
    return { badge: "Fixed Delta Base", caption: "Stationary circular plate anchored to bottom triangle" };
  }
  if (zDrive === "belted_quad_z") {
    return { badge: "Fixed Base Bed", caption: "Voron 2.4 style — bed fixed to base extrusions, flying XY gantry" };
  }
  if (zDrive === "leadscrew_triple") {
    return { badge: "3-Pt Trident Z", caption: "3 independent Z leadscrews (front-left, front-right, rear-center)" };
  }
  if (zDrive === "leadscrew_dual") {
    return { badge: "2-Pt Left/Right Z", caption: "Dual left/right Z leadscrews & linear guides" };
  }
  if (zDrive === "leadscrew_dual_belt_sync") {
    return {
      badge: "Belt-Synced Dual-Z",
      caption: "VzBot style — dual Z leadscrews mechanically synced via closed-loop timing belt",
    };
  }
  if (zDrive === "leadscrew_single") {
    return { badge: "Cantilever Rear-Z", caption: "Single rear Z rail/screw — test front/low Z for cantilever droop" };
  }
  return { badge: zDrive, caption: "Z-axis bed support" };
}

export interface DesignBedZSectionProps {
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

export const DesignBedZSection: React.FC<DesignBedZSectionProps> = ({
  spec,
  catalog,
  sim,
  showHints,
  advancedMode,
  update,
  updatePositiveNum,
  materialLabel,
}) => {
  const [bedTempDraft, setBedTempDraft] = useState<number>(
    Math.round(spec.bed.target_temp_c ?? 60)
  );
  const bedTempTimerRef = useRef<number | null>(null);

  const [bedWidthDraft, setBedWidthDraft] = useState<string>(
    String(Math.round(spec.bed.size_mm[0]))
  );
  const [bedDepthDraft, setBedDepthDraft] = useState<string>(
    String(Math.round(spec.bed.size_mm[1]))
  );
  const [bedHeaterDraft, setBedHeaterDraft] = useState<string>(
    String(Math.round(spec.bed.heater_power_w))
  );

  useEffect(() => {
    setBedTempDraft(Math.round(spec.bed.target_temp_c ?? 60));
  }, [spec.bed.target_temp_c]);

  useEffect(() => {
    setBedWidthDraft(String(Math.round(spec.bed.size_mm[0])));
  }, [spec.bed.size_mm[0]]);

  useEffect(() => {
    setBedDepthDraft(String(Math.round(spec.bed.size_mm[1])));
  }, [spec.bed.size_mm[1]]);

  useEffect(() => {
    setBedHeaterDraft(String(Math.round(spec.bed.heater_power_w)));
  }, [spec.bed.heater_power_w]);

  useEffect(() => {
    return () => {
      if (bedTempTimerRef.current != null) window.clearTimeout(bedTempTimerRef.current);
    };
  }, []);

  const commitBedSizeAxis = (axis: 0 | 1, raw: string) => {
    const current = Math.round(spec.bed.size_mm[axis]);
    const parsed = Number(raw);
    const setDraft = axis === 0 ? setBedWidthDraft : setBedDepthDraft;
    if (raw.trim() === "" || !Number.isFinite(parsed)) {
      setDraft(String(current));
      return;
    }
    const clamped = Math.max(80, Math.min(1300, Math.round(parsed)));
    setDraft(String(clamped));
    if (clamped === current) return;
    const axisLabel = axis === 0 ? "Bed Width" : "Bed Depth";
    update(
      (d) => {
        const nextSize: [number, number] = [d.bed.size_mm[0], d.bed.size_mm[1]];
        nextSize[axis] = clamped;
        d.bed.size_mm = nextSize;
      },
      `${axisLabel}: ${current} → ${clamped} mm`
    );
  };

  const commitBedHeaterW = (raw: string) => {
    const current = Math.round(spec.bed.heater_power_w);
    const parsed = Number(raw);
    if (raw.trim() === "" || !Number.isFinite(parsed)) {
      setBedHeaterDraft(String(current));
      return;
    }
    const clamped = Math.max(50, Math.min(3000, Math.round(parsed)));
    setBedHeaterDraft(String(clamped));
    if (clamped === current) return;
    update(
      (d) => {
        d.bed.heater_power_w = clamped;
      },
      `Bed Heater: ${current}W → ${clamped}W`
    );
  };

  const specBedTempC = Math.round(spec.bed.target_temp_c ?? 60);
  const commitBedTemp = (rawVal: number) => {
    const val = Math.max(25, Math.min(130, Math.round(rawVal)));
    if (bedTempTimerRef.current != null) {
      window.clearTimeout(bedTempTimerRef.current);
      bedTempTimerRef.current = null;
    }
    if (val === specBedTempC) return;
    update(
      (d) => {
        d.bed.target_temp_c = val;
      },
      `Bed Temp: ${specBedTempC}°C → ${val}°C`
    );
  };

  const handleBedTempSliderChange = (nextVal: number) => {
    setBedTempDraft(nextVal);
    if (bedTempTimerRef.current != null) window.clearTimeout(bedTempTimerRef.current);
    bedTempTimerRef.current = window.setTimeout(() => commitBedTemp(nextVal), 220);
  };

  const currentBedMount: BedMountType = spec.bed.mount_type ?? "kinematic_3pt";
  const bedMountCatalogEntry = catalog.bed_mounts?.find((m) => m.id === currentBedMount);
  const bedMountOptionMeta =
    BED_MOUNT_OPTIONS.find((o) => o.id === currentBedMount) ?? BED_MOUNT_OPTIONS[0];

  const plateMaterialOptions = (() => {
    const filtered = catalog.materials
      .filter((m) => PLATE_MATERIAL_IDS.has(m.id) || m.id === spec.bed.plate_material)
      .map((m) => ({ id: m.id, name: m.name }));
    const existingIds = new Set(filtered.map((m) => m.id));
    for (const fb of PLATE_MATERIAL_FALLBACKS) {
      if (!existingIds.has(fb.id)) {
        filtered.push(fb);
        existingIds.add(fb.id);
      }
    }
    if (!existingIds.has(spec.bed.plate_material)) {
      filtered.unshift({ id: spec.bed.plate_material, name: spec.bed.plate_material });
    }
    return filtered;
  })();

  const zTopoMeta = bedZTopologyMeta(spec.architecture, spec.bed.z_drive);

  const zRailProfileOptions =
    catalog.z_rail_profiles && catalog.z_rail_profiles.length > 0
      ? catalog.z_rail_profiles
      : FALLBACK_Z_RAIL_PROFILES;
  const currentZRailProfile: ZRailProfileType = spec.bed.z_rail_profile ?? "mgn12_rail";

  const zDriveMechanismOptions: ZDriveOptionMeta[] =
    catalog.z_drive_mechanisms && catalog.z_drive_mechanisms.length > 0
      ? [
          { id: "auto", name: "Auto (Match Z Architecture)", kind: "auto" },
          ...catalog.z_drive_mechanisms.map((m) => {
            const fb = FALLBACK_Z_DRIVE_BY_ID[m.id];
            return {
              id: m.id as ZLeadscrewType,
              name: m.name,
              kind: m.kind,
              k_nut_n_per_um: m.k_nut_n_per_um ?? fb?.k_nut_n_per_um,
              k_thrust_n_per_um: m.k_thrust_n_per_um ?? fb?.k_thrust_n_per_um,
              k_drive_n_per_um_per_corner:
                m.k_drive_n_per_um_per_corner ?? fb?.k_drive_n_per_um_per_corner,
              wobble_decoupled: m.wobble_decoupled ?? fb?.wobble_decoupled,
              self_locking: m.self_locking ?? fb?.self_locking,
              unpowered_drop_risk: m.unpowered_drop_risk ?? fb?.unpowered_drop_risk,
              backlash_um: m.backlash_um ?? fb?.backlash_um,
              gear_ratio: m.gear_ratio ?? fb?.gear_ratio,
              cost_per_axis_usd: m.cost_per_axis_usd ?? fb?.cost_per_axis_usd,
              notes: m.notes ?? fb?.notes,
            };
          }),
        ]
      : FALLBACK_Z_DRIVE_MECHANISMS;
  const currentZLeadscrewType: ZLeadscrewType = spec.bed.z_leadscrew_type ?? "auto";
  const isBeltedQuadZ = spec.bed.z_drive === "belted_quad_z";
  const isMovingBedZ = !isBeltedQuadZ && spec.architecture !== "delta";
  const screwDriveOptions = zDriveMechanismOptions.filter(
    (m) => m.id !== "auto" && m.kind !== "belt"
  );
  const beltDriveOptions = zDriveMechanismOptions.filter(
    (m) => m.id !== "auto" && m.kind === "belt"
  );

  const effectiveZDriveId: ZLeadscrewType =
    currentZLeadscrewType !== "auto"
      ? currentZLeadscrewType
      : isBeltedQuadZ
        ? "gt2_6mm_z_belt"
        : spec.bed.z_drive === "leadscrew_triple"
          ? "tr8x4_pom"
          : "tr8x8_pom";
  const activeZDriveMeta: ZDriveOptionMeta | undefined =
    zDriveMechanismOptions.find((m) => m.id === effectiveZDriveId) ??
    FALLBACK_Z_DRIVE_BY_ID[effectiveZDriveId];

  const kinematicBedMountOptions = BED_MOUNT_OPTIONS.filter((o) => o.group === "kinematic");
  const standardBedMountOptions = BED_MOUNT_OPTIONS.filter((o) => o.group === "standard");

  return (
    <div className="section-card" data-design-section="bed">
      <div className="section-title">
        <span>Build Plate &amp; Bed Mount</span>
        <span className="sub-block-tag">{bedTempDraft}°C</span>
      </div>

      <div className="slider-row">
        <label htmlFor="db-bed-temp-c-1">
          <span>Bed Temp (°C)</span>
          <span className="tension-readout">
            <strong style={{ color: bedTempDraft >= 100 ? "var(--amber)" : "var(--text)" }}>
              {bedTempDraft} °C
            </strong>
          </span>
        </label>
        <div className="tension-control-row">
          <input id="db-bed-temp-c-1"
            type="range"
            min={25}
            max={130}
            step={5}
            value={Math.max(25, Math.min(130, bedTempDraft))}
            onChange={(e) => handleBedTempSliderChange(Number(e.target.value))}
            onPointerUp={() => commitBedTemp(bedTempDraft)}
            onKeyUp={() => commitBedTemp(bedTempDraft)}
          />
          <input aria-label="Bed Temp (°C) value"
            type="number"
            className="tension-num-input"
            min={25}
            max={130}
            step={5}
            value={bedTempDraft}
            onChange={(e) => {
              const v = Number(e.target.value);
              if (!Number.isFinite(v)) return;
              setBedTempDraft(v);
              if (v >= 25 && v <= 130) commitBedTemp(v);
            }}
            onBlur={() => commitBedTemp(bedTempDraft)}
          />
        </div>
      </div>

      <div className="field-row">
        <label htmlFor="db-bed-mount-2">
          <span>Bed Mount</span>
          {bedMountCatalogEntry?.k_axial_n_per_mm != null && (
            <span style={{ fontSize: 11, color: "var(--muted)" }}>
              k_z {Math.round(bedMountCatalogEntry.k_axial_n_per_mm)} N/mm
            </span>
          )}
        </label>
        <select id="db-bed-mount-2"
          value={currentBedMount}
          onChange={(e) => {
            const nextVal = e.target.value as BedMountType;
            const prevLbl =
              BED_MOUNT_OPTIONS.find((o) => o.id === currentBedMount)?.shortLabel ??
              currentBedMount;
            const nextLbl =
              BED_MOUNT_OPTIONS.find((o) => o.id === nextVal)?.shortLabel ?? nextVal;
            const hadOv = Boolean(
              spec.model_overrides?.bed_mount &&
                Object.values(spec.model_overrides.bed_mount).some((v) => v != null)
            );
            update(
              (d) => {
                d.bed.mount_type = nextVal;
                clearModelOverrideGroup(d, "bed_mount");
              },
              `Bed Mount: ${prevLbl} → ${nextLbl}${hadOv ? " (override reset)" : ""}`
            );
          }}
        >
          {(
            [
              ["complete", "Complete Models"],
              ["wip_specific", "WIP Specific Models"],
              ["generic_standin", "Generic Stand-Ins"],
            ] as const
          ).map(([statusKey, groupLabel]) => {
            const groupItems = [...kinematicBedMountOptions, ...standardBedMountOptions].filter(
              (opt) => {
                const catItem = catalog.bed_mounts?.find((m) => m.id === opt.id);
                return (catItem?.model_status ?? "generic_standin") === statusKey;
              }
            );
            if (groupItems.length === 0) return null;
            return (
              <optgroup key={statusKey} label={groupLabel}>
                {groupItems.map((opt) => {
                  const catItem = catalog.bed_mounts?.find((m) => m.id === opt.id);
                  return (
                    <option key={opt.id} value={opt.id}>
                      {catItem?.name ?? opt.label}
                    </option>
                  );
                })}
              </optgroup>
            );
          })}
        </select>
        <div className="mount-physics-strip" aria-label="Bed mount physics summary">
          <span
            className="mount-pill neutral"
            title={`${bedMountCatalogEntry?.mount_points ?? bedMountOptionMeta.points}-point bed support layout`}
          >
            {bedMountCatalogEntry?.mount_points ?? bedMountOptionMeta.points}-Pt
          </span>
          <span
            className={`mount-pill ${bedMountOptionMeta.tapRated ? "ok" : "warn"}`}
            title={bedMountOptionMeta.tapTitle}
          >
            {bedMountOptionMeta.tapLabel}
          </span>
          <span
            className={`mount-pill ${bedMountOptionMeta.tacoTone}`}
            title={bedMountOptionMeta.tacoTitle}
          >
            {bedMountOptionMeta.tacoLabel}
          </span>
          <span
            className="mount-pill neutral"
            title="Nominal standoff stack height between heated bed plate and subframe"
          >
            {BED_MOUNT_STANDOFF_HEIGHT_MM[currentBedMount]} mm
          </span>
        </div>
        {showHints && (
          <div className="hint-line">
            {bedMountCatalogEntry?.notes || bedMountOptionMeta.hint}
          </div>
        )}
        {(() => {
          const fb = FALLBACK_BED_MOUNT_PARAMS[currentBedMount];
          const thermalNPerMm =
            bedMountCatalogEntry?.k_restraint_thermal_n_per_m != null
              ? bedMountCatalogEntry.k_restraint_thermal_n_per_m / 1000
              : fb.k_restraint_thermal_n_per_mm;
          return (
            <ModelOverrideControl
              visible={advancedMode}
              idPrefix="ov-bed-mount"
              title="Bed Mount"
              group="bed_mount"
              spec={spec}
              sim={sim}
              update={update}
              fields={[
                {
                  key: "k_axial_n_per_mm",
                  label: "Axial k_z",
                  unit: "N/mm",
                  min: 1,
                  max: 500000,
                  step: 100,
                  decimals: 0,
                  fallbackBaseline: bedMountCatalogEntry?.k_axial_n_per_mm ?? fb.k_axial_n_per_mm,
                  tooltip: "Per-point vertical Z stiffness of bed mount hardware",
                },
                {
                  key: "k_lateral_n_per_mm",
                  label: "Lateral k_xy",
                  unit: "N/mm",
                  min: 0.5,
                  max: 500000,
                  step: 100,
                  decimals: 0,
                  fallbackBaseline: bedMountCatalogEntry?.k_lateral_n_per_mm ?? fb.k_lateral_n_per_mm,
                  tooltip: "Per-point lateral shear stiffness against bed sway",
                },
                {
                  key: "k_restraint_thermal_n_per_mm",
                  label: "Thermal Restraint",
                  unit: "N/mm",
                  min: 0,
                  max: 500000,
                  step: 100,
                  decimals: 0,
                  fallbackBaseline: thermalNPerMm,
                  tooltip: "In-plane radial thermal expansion restraint driving bed taco buckling",
                },
                {
                  key: "damping_delta",
                  label: "Mount Damping",
                  unit: "ζ",
                  min: -0.02,
                  max: 0.1,
                  step: 0.001,
                  decimals: 3,
                  fallbackBaseline: bedMountCatalogEntry?.damping_delta ?? fb.damping_delta,
                  tooltip: "Modal damping increment contributed by bed mount interface",
                },
                {
                  key: "standoff_height_mm",
                  label: "Standoff Height",
                  unit: "mm",
                  min: 2,
                  max: 60,
                  step: 1,
                  decimals: 1,
                  fallbackBaseline: bedMountCatalogEntry?.standoff_height_mm ?? fb.standoff_height_mm,
                  tooltip: "Nominal standoff stack height between bed plate and Z subframe",
                },
              ]}
            />
          );
        })()}
      </div>

      <div className="field-grid-3">
        <div className="field-row">
          <label htmlFor="db-bed-w-mm-3" title="Build Plate Width (mm)">Bed W (mm)</label>
          <input id="db-bed-w-mm-3"
            type="number"
            min={80}
            max={1300}
            step={10}
            value={bedWidthDraft}
            onChange={(e) => setBedWidthDraft(e.target.value)}
            onBlur={() => commitBedSizeAxis(0, bedWidthDraft)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                commitBedSizeAxis(0, bedWidthDraft);
                e.currentTarget.blur();
              } else if (e.key === "Escape") {
                setBedWidthDraft(String(Math.round(spec.bed.size_mm[0])));
                e.currentTarget.blur();
              }
            }}
          />
        </div>
        <div className="field-row">
          <label htmlFor="db-bed-d-mm-4" title="Build Plate Depth (mm)">Bed D (mm)</label>
          <input id="db-bed-d-mm-4"
            type="number"
            min={80}
            max={1300}
            step={10}
            value={bedDepthDraft}
            onChange={(e) => setBedDepthDraft(e.target.value)}
            onBlur={() => commitBedSizeAxis(1, bedDepthDraft)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                commitBedSizeAxis(1, bedDepthDraft);
                e.currentTarget.blur();
              } else if (e.key === "Escape") {
                setBedDepthDraft(String(Math.round(spec.bed.size_mm[1])));
                e.currentTarget.blur();
              }
            }}
          />
        </div>
        <div className="field-row">
          <label htmlFor="db-heater-w-5" title="Bed Heater Wattage (W)">Heater (W)</label>
          <input id="db-heater-w-5"
            type="number"
            min={50}
            max={3000}
            step={25}
            value={bedHeaterDraft}
            onChange={(e) => setBedHeaterDraft(e.target.value)}
            onBlur={() => commitBedHeaterW(bedHeaterDraft)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                commitBedHeaterW(bedHeaterDraft);
                e.currentTarget.blur();
              } else if (e.key === "Escape") {
                setBedHeaterDraft(String(Math.round(spec.bed.heater_power_w)));
                e.currentTarget.blur();
              }
            }}
          />
        </div>
      </div>

      <div className="field-grid-2 wide-last">
        <div className="field-row">
          <label htmlFor="db-plate-thickness-mm-6">Plate Thickness (mm)</label>
          <input id="db-plate-thickness-mm-6"
            type="number"
            min={4}
            max={12}
            step={1}
            value={spec.bed.plate_thickness_mm}
            onChange={(e) =>
              updatePositiveNum(
                e.target.value,
                (d, v) => {
                  d.bed.plate_thickness_mm = Math.max(4, Math.min(12, v));
                },
                (v) =>
                  `Plate Thickness: ${spec.bed.plate_thickness_mm} → ${Math.max(
                    4,
                    Math.min(12, v)
                  )} mm`
              )
            }
          />
        </div>
        <div className="field-row">
          <label htmlFor="db-plate-material-7">Plate Material</label>
          <select id="db-plate-material-7"
            value={spec.bed.plate_material}
            onChange={(e) => {
              const nextVal = e.target.value;
              update(
                (d) => {
                  d.bed.plate_material = nextVal;
                },
                `Plate Material: ${materialLabel(spec.bed.plate_material)} → ${materialLabel(
                  nextVal
                )}`
              );
            }}
          >
            {plateMaterialOptions.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name}
              </option>
            ))}
          </select>
        </div>
      </div>

      <details className="sub-accordion">
        <summary>
          <span>Z-Axis Hardware &amp; Bed Mass</span>
          <span className="sub-accordion-tag">
            {zTopoMeta.badge} · {spec.bed.mass_g}g
          </span>
        </summary>
        <div className="sub-accordion-body">
          <div className="inline-readout-strip" title={zTopoMeta.caption}>
            <strong style={{ color: "var(--accent)" }}>{zTopoMeta.badge}</strong>
            <span>·</span>
            <span>{zTopoMeta.caption}</span>
          </div>

          <div className="field-row">
            <label htmlFor="db-z-linear-rails-guideway-8">Z Linear Rails / Guideway</label>
            <select id="db-z-linear-rails-guideway-8"
              value={currentZRailProfile}
              onChange={(e) => {
                const nextVal = e.target.value as ZRailProfileType;
                const prevLbl =
                  zRailProfileOptions.find((o) => o.id === currentZRailProfile)?.name ??
                  currentZRailProfile;
                const nextLbl =
                  zRailProfileOptions.find((o) => o.id === nextVal)?.name ?? nextVal;
                update(
                  (d) => {
                    d.bed.z_rail_profile = nextVal;
                  },
                  `Z Rails: ${prevLbl} → ${nextLbl}`
                );
              }}
            >
              {!zRailProfileOptions.some((opt) => opt.id === currentZRailProfile) && (
                <optgroup label="Generic Stand-Ins">
                  <option value={currentZRailProfile}>{currentZRailProfile} (Custom)</option>
                </optgroup>
              )}
              {(
                [
                  ["complete", "Complete Models"],
                  ["wip_specific", "WIP Specific Models"],
                  ["generic_standin", "Generic Stand-Ins"],
                ] as const
              ).map(([statusKey, groupLabel]) => {
                const groupItems = zRailProfileOptions.filter((opt) => {
                  const catItem = catalog.z_rail_profiles?.find((r) => r.id === opt.id);
                  const st =
                    catItem?.model_status ??
                    (opt.id === "mgn12_rail" || opt.id === "mgn9_rail"
                      ? "complete"
                      : "generic_standin");
                  return st === statusKey;
                });
                if (groupItems.length === 0) return null;
                return (
                  <optgroup key={statusKey} label={groupLabel}>
                    {groupItems.map((opt) => (
                      <option key={opt.id} value={opt.id}>
                        {opt.name}
                      </option>
                    ))}
                  </optgroup>
                );
              })}
            </select>
          </div>

          <div className="field-row">
            <label htmlFor="db-field-9">
              <span>{isBeltedQuadZ ? "Z Belt Drive" : "Z Leadscrew / Drive"}</span>
              {currentZLeadscrewType === "auto" && activeZDriveMeta && (
                <span style={{ fontSize: 11, color: "var(--muted)" }}>
                  → {activeZDriveMeta.name}
                </span>
              )}
            </label>
            <select id="db-field-9"
              value={currentZLeadscrewType}
              onChange={(e) => {
                const nextVal = e.target.value as ZLeadscrewType;
                const prevLbl =
                  zDriveMechanismOptions.find((o) => o.id === currentZLeadscrewType)
                    ?.name ?? currentZLeadscrewType;
                const nextLbl =
                  zDriveMechanismOptions.find((o) => o.id === nextVal)?.name ?? nextVal;
                const hadOv = Boolean(
                  spec.model_overrides?.z_drive_mechanism &&
                    Object.values(spec.model_overrides.z_drive_mechanism).some((v) => v != null)
                );
                update(
                  (d) => {
                    d.bed.z_leadscrew_type = nextVal;
                    clearModelOverrideGroup(d, "z_drive_mechanism");
                  },
                  `Z Drive: ${prevLbl} → ${nextLbl}${hadOv ? " (override reset)" : ""}`
                );
              }}
            >
              <option value="auto">Auto (Match Z Architecture)</option>
              {(() => {
                const visibleDrives = isBeltedQuadZ
                  ? beltDriveOptions
                  : [...screwDriveOptions, ...beltDriveOptions];
                const hasCurrent =
                  currentZLeadscrewType === "auto" ||
                  visibleDrives.some((o) => o.id === currentZLeadscrewType);
                return (
                  <>
                    {!hasCurrent && (
                      <optgroup label="Generic Stand-Ins">
                        <option value={currentZLeadscrewType}>
                          {zDriveMechanismOptions.find((o) => o.id === currentZLeadscrewType)
                            ?.name ?? currentZLeadscrewType}{" "}
                          (Custom)
                        </option>
                      </optgroup>
                    )}
                    {(
                      [
                        ["complete", "Complete Models"],
                        ["wip_specific", "WIP Specific Models"],
                        ["generic_standin", "Generic Stand-Ins"],
                      ] as const
                    ).map(([statusKey, groupLabel]) => {
                      const groupItems = visibleDrives.filter((opt) => {
                        const catItem = catalog.z_drive_mechanisms?.find((z) => z.id === opt.id);
                        return (catItem?.model_status ?? "generic_standin") === statusKey;
                      });
                      if (groupItems.length === 0) return null;
                      return (
                        <optgroup key={statusKey} label={groupLabel}>
                          {groupItems.map((opt) => (
                            <option key={opt.id} value={opt.id}>
                              {opt.name}
                            </option>
                          ))}
                        </optgroup>
                      );
                    })}
                  </>
                );
              })()}
            </select>
            {activeZDriveMeta && (
              <div className="mount-physics-strip" aria-label="Z drive physics summary">
                <span
                  className={`mount-pill ${activeZDriveMeta.wobble_decoupled ? "ok" : "neutral"}`}
                  title={
                    activeZDriveMeta.wobble_decoupled
                      ? "Belted Z or WobbleX floating nut decouples XY screw runout from print walls"
                      : "Rigidly coupled nut transmits leadscrew/ballscrew runout laterally unless well-aligned"
                  }
                >
                  {activeZDriveMeta.wobble_decoupled ? "Wobble-Free" : "Screw-Coupled"}
                </span>
                {activeZDriveMeta.self_locking ? (
                  <span
                    className="mount-pill ok"
                    title="High mechanical lead/worm friction holds Z height securely when stepper power is cut"
                  >
                    Self-Locking
                  </span>
                ) : !activeZDriveMeta.unpowered_drop_risk ? (
                  <span
                    className="mount-pill ok"
                    title="Keybak spring retractor counterbalances bed weight to prevent unpowered drop"
                  >
                    Keybak Balanced
                  </span>
                ) : isMovingBedZ ? (
                  <span
                    className="mount-pill warn"
                    title="Low back-drive friction can allow a heavy moving bed to drop when Z steppers power off"
                  >
                    Drop Risk (Unpowered)
                  </span>
                ) : (
                  <span
                    className="mount-pill info"
                    title="Flying-gantry belt drive relies on motor holding torque or physical gantry stops when unpowered"
                  >
                    Powered Hold
                  </span>
                )}
                {activeZDriveMeta.gear_ratio != null && activeZDriveMeta.gear_ratio > 1 && (
                  <span
                    className="mount-pill info"
                    title={`Mechanical drive reduction ratio (${activeZDriveMeta.gear_ratio}:1)`}
                  >
                    {activeZDriveMeta.gear_ratio}:1 Gear
                  </span>
                )}
                {activeZDriveMeta.backlash_um != null && (
                  <span
                    className="mount-pill neutral"
                    title="Nominal axial Z backlash under direction reversal"
                  >
                    Backlash ~{Math.round(activeZDriveMeta.backlash_um)} µm
                  </span>
                )}
              </div>
            )}
            {showHints && activeZDriveMeta?.notes && (
              <div className="hint-line">{activeZDriveMeta.notes}</div>
            )}
            <ModelOverrideControl
              visible={advancedMode}
              idPrefix="ov-z-drive-mechanism"
              title="Z-Drive Mechanism"
              group="z_drive_mechanism"
              spec={spec}
              sim={sim}
              update={update}
              fields={
                activeZDriveMeta?.kind === "belt"
                  ? [
                      {
                        key: "k_drive_n_per_um_per_corner",
                        label: "Corner Drive k",
                        unit: "N/µm",
                        min: 0.01,
                        max: 100,
                        step: 0.05,
                        decimals: 2,
                        fallbackBaseline: activeZDriveMeta.k_drive_n_per_um_per_corner ?? 0.18,
                        tooltip: "Effective axial Z-belt loop stiffness per driven corner",
                      },
                      {
                        key: "backlash_um",
                        label: "Z Backlash",
                        unit: "µm",
                        min: 0,
                        max: 200,
                        step: 1,
                        decimals: 1,
                        fallbackBaseline: activeZDriveMeta.backlash_um ?? 18,
                        tooltip: "Nominal axial Z reversal backlash",
                      },
                    ]
                  : [
                      {
                        key: "k_nut_n_per_um",
                        label: "Nut Stiffness",
                        unit: "N/µm",
                        min: 0.5,
                        max: 1000,
                        step: 1,
                        decimals: 1,
                        fallbackBaseline: activeZDriveMeta?.k_nut_n_per_um ?? 11.0,
                        tooltip: "Axial thread/ball-nut stiffness per leadscrew or ballscrew",
                      },
                      {
                        key: "k_thrust_n_per_um",
                        label: "Thrust Bearing k",
                        unit: "N/µm",
                        min: 1,
                        max: 2000,
                        step: 2,
                        decimals: 1,
                        fallbackBaseline: activeZDriveMeta?.k_thrust_n_per_um ?? 18.0,
                        tooltip: "Axial stiffness of the Z motor or BK/FK thrust bearing block",
                      },
                      {
                        key: "backlash_um",
                        label: "Z Backlash",
                        unit: "µm",
                        min: 0,
                        max: 200,
                        step: 1,
                        decimals: 1,
                        fallbackBaseline: activeZDriveMeta?.backlash_um ?? 28,
                        tooltip: "Nominal axial Z reversal backlash",
                      },
                    ]
              }
            />
          </div>

          <div className="field-row">
            <label htmlFor="db-bed-mass-g-10">Bed Mass (g)</label>
            <input id="db-bed-mass-g-10"
              type="number"
              min={800}
              max={6000}
              step={100}
              value={spec.bed.mass_g}
              onChange={(e) =>
                updatePositiveNum(
                  e.target.value,
                  (d, v) => {
                    d.bed.mass_g = Math.max(800, Math.min(6000, Math.round(v)));
                  },
                  (v) =>
                    `Bed Mass: ${spec.bed.mass_g}g → ${Math.max(
                      800,
                      Math.min(6000, Math.round(v))
                    )}g`
                )
              }
            />
          </div>
        </div>
      </details>
    </div>
  );
};
