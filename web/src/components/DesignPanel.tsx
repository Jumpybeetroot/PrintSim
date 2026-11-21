import React, { useEffect, useRef, useState } from "react";
import {
  BearingClearanceClass,
  BedMountType,
  BracketMaterialType,
  CatalogResponse,
  CornerJointType,
  DriveType,
  ExtrusionFillingInfo,
  ExtrusionFillingType,
  GantryBracketInfo,
  PrinterSpecData,
  RailBackerMaterialType,
  RailBlockType,
  RailMountOption,
  RailPreloadType,
  SimulationResponse,
  SolidFeaEngine,
  ToolheadMountMaterial,
  TopoOptRequestParams,
  XRailMountType,
  YRailMountType,
} from "../types";

interface Props {
  catalog: CatalogResponse | null;
  sim: SimulationResponse | null;
  selectedPreset: string;
  onSelectPreset: (id: string) => void;
  filamentId: string;
  onFilamentChange: (id: string) => void;
  spec: PrinterSpecData | null;
  onSpecChange: (next: PrinterSpecData, changeLabel?: string) => void;
  yamlText: string;
  onYamlChange: (txt: string) => void;
  onSimulateYaml: () => void;
  gcodeText: string;
  onGcodeChange: (txt: string) => void;
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

const BRACING_LABELS: Record<PrinterSpecData["frame"]["bracing"], string> = {
  none: "None",
  corner_gussets: "Corner Gussets",
  rear_cross: "Rear Cross + Top",
  full_truss: "Full 3D Truss",
};

const ARCH_LABELS: Record<PrinterSpecData["architecture"], string> = {
  corexy: "CoreXY",
  cartesian_bedslinger: "Cartesian Bedslinger",
  delta: "Delta",
};

const Z_DRIVE_SHORT_LABELS: Record<string, string> = {
  leadscrew_triple: "Fixed Gantry (Trident)",
  belted_quad_z: "Flying Gantry (Voron 2.4)",
  leadscrew_dual: "Fixed Gantry (Dual-Z)",
  leadscrew_single: "Fixed Gantry (Single-Z)",
};

const BELT_LABELS: Record<PrinterSpecData["gantry"]["belt"]["type"], string> = {
  gt2_6mm: "GT2 6mm",
  gt2_9mm: "GT2 9mm",
  gt2_12mm: "GT2 12mm",
};

const SHAPER_LABELS: Record<PrinterSpecData["kinematics"]["input_shaper"], string> = {
  auto: "Auto (Klipper)",
  none: "None",
  zv: "ZV",
  mzv: "MZV",
  ei: "EI",
  "2hump_ei": "2HUMP_EI",
  zvdd: "ZVDD",
  smooth_zv: "SMOOTH_ZV (Kalico)",
  smooth_ei: "SMOOTH_EI (Kalico)",
};

const DRIVE_TYPE_OPTIONS: Array<{ id: DriveType; label: string; shortLabel: string }> = [
  { id: "open_loop_stepper", label: "Open-Loop Stepper", shortLabel: "Open-Loop Stepper" },
  { id: "closed_loop_stepper", label: "Closed-Loop Stepper", shortLabel: "Closed-Loop Stepper" },
  { id: "foc_servo", label: "FOC Servo (Closed-Loop)", shortLabel: "FOC Servo" },
];

const HEATBREAK_LABELS: Record<string, string> = {
  titanium_grade5: "Titanium Gr.5",
  stainless_304: "Stainless 304",
  brass_c360: "Brass C360",
};

const FALLBACK_EXTRUSION_FILLINGS: ExtrusionFillingInfo[] = [
  {
    id: "none",
    name: "Hollow (Unfilled)",
    short_name: "Hollow",
    density_kg_m3: 0,
    youngs_modulus_pa: 0,
    bonded: false,
    damping_boost: 0,
    cost_per_kg_usd: 0,
    color_hex: "#94a3b8",
    notes: "Standard hollow aluminum extrusion (low structural damping)",
  },
  {
    id: "pu_foam",
    name: "Expanding PU Acoustic Foam",
    short_name: "PU Foam",
    density_kg_m3: 45,
    youngs_modulus_pa: 0.02e9,
    bonded: false,
    damping_boost: 0.01,
    cost_per_kg_usd: 18,
    color_hex: "#fde68a",
    notes: "Ultra-light acoustic deadening; negligible added mass or stiffness",
  },
  {
    id: "silica_sand",
    name: "Kiln-Dried Silica Sand",
    short_name: "Silica Sand",
    density_kg_m3: 1550,
    youngs_modulus_pa: 0,
    bonded: false,
    damping_boost: 0.028,
    cost_per_kg_usd: 1.5,
    color_hex: "#d6b88d",
    notes: "Low-cost granular Coulomb friction damping; zero bonded bending stiffness",
  },
  {
    id: "lead_shot",
    name: "Fine Lead Shot (#8–#9)",
    short_name: "Lead Shot",
    density_kg_m3: 6800,
    youngs_modulus_pa: 0,
    bonded: false,
    damping_boost: 0.042,
    cost_per_kg_usd: 11,
    color_hex: "#475569",
    notes: "Maximum granular impact damping & base inertia; heavy mass lowers raw Hz",
  },
  {
    id: "epoxy_granite",
    name: "Epoxy Granite (85% Quartz + Epoxy)",
    short_name: "Epoxy Granite",
    density_kg_m3: 2250,
    youngs_modulus_pa: 30e9,
    bonded: true,
    damping_boost: 0.036,
    cost_per_kg_usd: 9.5,
    color_hex: "#64748b",
    notes: "Mineral casting bonds to cavity walls: +10-16% bending stiffness & strong resin damping",
  },
  {
    id: "uhpc_grout",
    name: "Non-Shrink UHPC / CSA Grout",
    short_name: "UHPC Grout",
    density_kg_m3: 2350,
    youngs_modulus_pa: 40e9,
    bonded: true,
    damping_boost: 0.03,
    cost_per_kg_usd: 4.5,
    color_hex: "#9ca3af",
    notes: "High-modulus bonded cementitious grout: highest core EI boost + solid damping",
  },
];

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

function mountMaterialShort(id?: string): string {
  return MOUNT_MATERIAL_OPTIONS.find((m) => m.id === id)?.shortLabel ?? "Printed ABS";
}

function toolheadStiffnessTier(k: number): { label: string; color: string } {
  if (k < 250) return { label: "Soft Printed", color: "var(--amber)" };
  if (k <= 700) return { label: "Standard Printed", color: "var(--muted)" };
  if (k <= 1500) return { label: "Rigid CF-Polymer", color: "var(--accent)" };
  return { label: "CNC Monoblock", color: "var(--green)" };
}

function shortMotorName(name: string): string {
  return name.split("(")[0].trim() || name;
}

const PANEL_MATERIALS: Array<{ id: string; label: string; shortLabel: string }> = [
  { id: "acrylic_pmma", label: "Cast Acrylic (PMMA)", shortLabel: "Acrylic" },
  { id: "polycarbonate", label: "Polycarbonate (PC)", shortLabel: "Polycarbonate" },
  { id: "acm_dibond", label: "ACM / Dibond", shortLabel: "ACM / Dibond" },
  { id: "aluminum_5052_sheet", label: "Aluminum 5052 Sheet", shortLabel: "Aluminum 5052" },
  { id: "carbon_fiber_plate", label: "Carbon Fiber 3K Sheet", shortLabel: "Carbon Fiber 3K" },
];

const PANEL_MOUNTING_SHORT: Record<string, string> = {
  none: "Open",
  clip_enclosure: "Clip-In",
  bolted_structural: "Bolted Structural",
};

const PANEL_THICKNESS_OPTIONS = [2, 3, 4, 5, 6];

function panelMaterialShort(id: string): string {
  return PANEL_MATERIALS.find((m) => m.id === id)?.shortLabel ?? id;
}

function formatPanelSummary(frame: PrinterSpecData["frame"]): string {
  const mounting = !frame.enclosed_panels ? "none" : frame.panel_mounting || "bolted_structural";
  if (!frame.enclosed_panels || mounting === "none") return "Open";
  const mat = panelMaterialShort(frame.panel_material || "acrylic_pmma");
  const mnt = PANEL_MOUNTING_SHORT[mounting] ?? mounting;
  return `${frame.panel_thickness_mm}mm ${mat} (${mnt})`;
}

const CORNER_JOINT_OPTIONS: Array<{ id: CornerJointType; label: string; shortLabel: string }> = [
  { id: "blind_joint", label: "Blind Joint (T-Slot)", shortLabel: "Blind Joint" },
  { id: "l_bracket", label: "L-Bracket (Die-Cast)", shortLabel: "L-Bracket" },
  { id: "gusset_plate", label: "Gusset Plate", shortLabel: "Gusset Plate" },
  { id: "corner_cube", label: "3-Way Corner Cube", shortLabel: "Corner Cube" },
  { id: "rigid", label: "Ideal Rigid", shortLabel: "Ideal Rigid" },
];

const RAIL_PRELOAD_OPTIONS: Array<{ id: RailPreloadType; label: string; shortLabel: string }> = [
  { id: "ZF", label: "ZF Clearance", shortLabel: "ZF" },
  { id: "Z0", label: "Z0 Very Light", shortLabel: "Z0" },
  { id: "Z1", label: "Z1 Light Preload", shortLabel: "Z1" },
  { id: "ZA", label: "ZA Medium", shortLabel: "ZA" },
  { id: "ZB", label: "ZB Heavy", shortLabel: "ZB" },
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

const BED_MOUNT_OPTIONS: Array<{ id: BedMountType; label: string; shortLabel: string }> = [
  { id: "kinematic_3pt", label: "3-Pt Kinematic (Maxwell)", shortLabel: "3-Pt Kinematic" },
  { id: "floating_4pt", label: "4-Pt Floating", shortLabel: "4-Pt Floating" },
  { id: "rigid_4pt", label: "4-Pt Rigid (Over-Constrained)", shortLabel: "4-Pt Rigid" },
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
  yamlText,
  onYamlChange,
  onSimulateYaml,
  gcodeText,
  onGcodeChange,
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
  const [showHints, setShowHints] = useState<boolean>(true);
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

  const [toolheadStiffnessDraft, setToolheadStiffnessDraft] = useState<number>(
    Math.round(spec?.toolhead.internal_stiffness_n_per_mm ?? 450)
  );
  const toolheadStiffnessTimerRef = useRef<number | null>(null);

  const [chamberTempDraft, setChamberTempDraft] = useState<number>(
    Math.round(spec?.frame.chamber_temp_c ?? 25)
  );
  const chamberTempTimerRef = useRef<number | null>(null);

  const [bedTempDraft, setBedTempDraft] = useState<number>(
    Math.round(spec?.bed.target_temp_c ?? 60)
  );
  const bedTempTimerRef = useRef<number | null>(null);

  const [notchDraft, setNotchDraft] = useState<string>(
    spec?.kinematics.notch_filter_hz != null ? String(Math.round(spec.kinematics.notch_filter_hz)) : ""
  );

  useEffect(() => {
    if (spec?.gantry.belt.pretension_n != null) {
      setTensionDraft(spec.gantry.belt.pretension_n);
    }
  }, [spec?.gantry.belt.pretension_n]);

  useEffect(() => {
    if (spec?.kinematics.xy_stepper) {
      const nextA =
        spec.kinematics.xy_stepper.run_current_a ??
        Math.min(spec.kinematics.xy_stepper.rated_current_a || 1.2, 1.2);
      setRunCurrentDraft(Number(nextA.toFixed(1)));
    }
  }, [spec?.kinematics.xy_stepper.run_current_a, spec?.kinematics.xy_stepper.preset_id]);

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
      setToolheadStiffnessDraft(Math.round(spec.toolhead.internal_stiffness_n_per_mm ?? 450));
    }
  }, [spec?.toolhead.internal_stiffness_n_per_mm]);

  useEffect(() => {
    if (spec) {
      setChamberTempDraft(Math.round(spec.frame.chamber_temp_c ?? 25));
    }
  }, [spec?.frame.chamber_temp_c]);

  useEffect(() => {
    if (spec) {
      setBedTempDraft(Math.round(spec.bed.target_temp_c ?? 60));
    }
  }, [spec?.bed.target_temp_c]);

  useEffect(() => {
    if (spec) {
      const nf = spec.kinematics.notch_filter_hz;
      setNotchDraft(nf != null && nf > 0 ? String(Math.round(nf)) : "");
    }
  }, [spec?.kinematics.notch_filter_hz]);

  useEffect(() => {
    return () => {
      if (tensionTimerRef.current != null) window.clearTimeout(tensionTimerRef.current);
      if (currentTimerRef.current != null) window.clearTimeout(currentTimerRef.current);
      if (cogYTimerRef.current != null) window.clearTimeout(cogYTimerRef.current);
      if (cogZTimerRef.current != null) window.clearTimeout(cogZTimerRef.current);
      if (nozzleZTimerRef.current != null) window.clearTimeout(nozzleZTimerRef.current);
      if (toolheadStiffnessTimerRef.current != null) window.clearTimeout(toolheadStiffnessTimerRef.current);
      if (chamberTempTimerRef.current != null) window.clearTimeout(chamberTempTimerRef.current);
      if (bedTempTimerRef.current != null) window.clearTimeout(bedTempTimerRef.current);
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

  const commitBeltTension = (rawVal: number) => {
    const val = Math.max(15, Math.min(120, Math.round(rawVal)));
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

  const specRunCurrentA =
    spec.kinematics.xy_stepper.run_current_a ??
    Math.min(spec.kinematics.xy_stepper.rated_current_a || 1.2, 1.2);

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
  const specToolheadStiffness = Math.round(spec.toolhead.internal_stiffness_n_per_mm ?? 450);

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

  const commitToolheadStiffness = (rawVal: number) => {
    const val = Math.max(25, Math.min(20000, Math.round(rawVal)));
    if (toolheadStiffnessTimerRef.current != null) {
      window.clearTimeout(toolheadStiffnessTimerRef.current);
      toolheadStiffnessTimerRef.current = null;
    }
    if (val === specToolheadStiffness) return;
    update(
      (d) => {
        d.toolhead.internal_stiffness_n_per_mm = val;
        d.toolhead.preset_id = "custom";
      },
      `Toolhead Stiffness: ${specToolheadStiffness} → ${val} N/mm`
    );
  };

  const handleToolheadStiffnessSliderChange = (nextVal: number) => {
    setToolheadStiffnessDraft(nextVal);
    if (toolheadStiffnessTimerRef.current != null) {
      window.clearTimeout(toolheadStiffnessTimerRef.current);
    }
    toolheadStiffnessTimerRef.current = window.setTimeout(
      () => commitToolheadStiffness(nextVal),
      220
    );
  };

  const specChamberTempC = Math.round(spec.frame.chamber_temp_c ?? 25);
  const commitChamberTemp = (rawVal: number) => {
    const val = Math.max(25, Math.min(85, Math.round(rawVal)));
    if (chamberTempTimerRef.current != null) {
      window.clearTimeout(chamberTempTimerRef.current);
      chamberTempTimerRef.current = null;
    }
    if (val === specChamberTempC) return;
    update(
      (d) => {
        d.frame.chamber_temp_c = val;
      },
      `Chamber Temp: ${specChamberTempC}°C → ${val}°C`
    );
  };

  const handleChamberTempSliderChange = (nextVal: number) => {
    setChamberTempDraft(nextVal);
    if (chamberTempTimerRef.current != null) window.clearTimeout(chamberTempTimerRef.current);
    chamberTempTimerRef.current = window.setTimeout(() => commitChamberTemp(nextVal), 220);
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

  const currentCornerJoint: CornerJointType = spec.frame.corner_joint_type ?? "blind_joint";
  const currentRailPreload: RailPreloadType = spec.gantry.rail_preload ?? "Z0";
  const currentRailBlock: RailBlockType = spec.gantry.rail_block_type ?? "H";
  const currentRailBoltPitch: number = spec.gantry.rail_bolt_pitch_mm ?? 25;
  const currentRailBacker: RailBackerMaterialType = spec.gantry.rail_backer_material ?? "none";
  const currentIdlerShaftMm: number = spec.gantry.belt.idler_shaft_diameter_mm ?? 5;
  const currentBearingClass: BearingClearanceClass = spec.gantry.belt.bearing_clearance_class ?? "CN";
  const currentBedMount: BedMountType = spec.bed.mount_type ?? "kinematic_3pt";
  const currentDriveType: DriveType = spec.kinematics.drive_type ?? "open_loop_stepper";
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

  const effectivePanelMounting: PrinterSpecData["frame"]["panel_mounting"] = !spec.frame.enclosed_panels
    ? "none"
    : spec.frame.panel_mounting || "bolted_structural";
  const panelsEnabled = spec.frame.enclosed_panels && effectivePanelMounting !== "none";
  const currentPanelMaterial = spec.frame.panel_material || "acrylic_pmma";
  const currentPanelThickness = spec.frame.panel_thickness_mm || 3;
  const isFlyingGantry = spec.bed.z_drive === "belted_quad_z";
  const boltPanelsToGantry = spec.frame.bolt_panels_to_gantry ?? true;
  const thicknessChoices = PANEL_THICKNESS_OPTIONS.includes(currentPanelThickness)
    ? PANEL_THICKNESS_OPTIONS
    : [...PANEL_THICKNESS_OPTIONS, currentPanelThickness].sort((a, b) => a - b);

  const driveMode: "2wd" | "awd" = spec.gantry.drive_mode || "2wd";
  const doubleShearPulleys = Boolean(spec.gantry.double_shear_pulleys);
  const beltPluckHz = sim?.kinematics.belt_pluck_freq_150mm_hz;
  const beltMaxWorkingN = sim?.kinematics.belt_max_working_tension_n;
  const beltOvertension = Boolean(sim?.kinematics.belt_overtension_risk);
  const shaftBendingRisk = Boolean(sim?.kinematics.shaft_bending_risk);

  const extrusionFillings: ExtrusionFillingInfo[] =
    catalog.extrusion_fillings && catalog.extrusion_fillings.length > 0
      ? catalog.extrusion_fillings
      : FALLBACK_EXTRUSION_FILLINGS;
  const currentFillingId: ExtrusionFillingType = spec.frame.extrusion_filling ?? "none";
  const selectedFilling =
    extrusionFillings.find((f) => f.id === currentFillingId) ?? extrusionFillings[0];
  const simFilling =
    sim?.geometry.extrusion_filling && sim.geometry.extrusion_filling.id === currentFillingId
      ? sim.geometry.extrusion_filling
      : null;
  const fillingAddedMassKg = simFilling?.added_mass_kg ?? 0;
  const fillingDampingRatio =
    simFilling?.damping_ratio ??
    sim?.modal_fea.frame_damping_ratio ??
    0.042 + (selectedFilling?.damping_boost ?? 0);

  const gantryBrackets: GantryBracketInfo[] =
    catalog.gantry_brackets && catalog.gantry_brackets.length > 0
      ? catalog.gantry_brackets
      : FALLBACK_GANTRY_BRACKETS;
  const currentBracketId: BracketMaterialType = spec.gantry.bracket_material ?? "abs_printed";
  const selectedBracket =
    gantryBrackets.find((b) => b.id === currentBracketId) ?? gantryBrackets[0];
  const simBracketStiffness = sim?.kinematics.bracket_stiffness_n_per_mm;
  const bracketCreepRisk = Boolean(sim?.kinematics.bracket_creep_risk);

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
  const simLinearRails = sim?.geometry.linear_rails;

  const partFeaMaterials = (() => {
    const existingIds = new Set(catalog.materials.map((m) => m.id));
    const extras = gantryBrackets
      .filter((b) => !existingIds.has(b.id))
      .map((b) => ({ id: b.id, name: b.name }));
    return [...catalog.materials, ...extras];
  })();

  const stepperCatalog = catalog.steppers ?? [];
  const xyStepper = spec.kinematics.xy_stepper;
  const matchedStepper =
    stepperCatalog.find((s) => s.preset_id === xyStepper.preset_id) ??
    stepperCatalog.find((s) => s.name === xyStepper.name);
  const selectedStepperValue = matchedStepper ? matchedStepper.preset_id : "__custom__";
  const ratedCurrentA = Math.max(xyStepper.rated_current_a || 2.0, 0.1);
  const currentPctOfRated = Math.round((runCurrentDraft / ratedCurrentA) * 100);
  const phaseResistanceOhm = xyStepper.phase_resistance_ohm ?? 1.2;
  const estMotorHeatW = 2.0 * runCurrentDraft * runCurrentDraft * phaseResistanceOhm;
  const motorOvercurrentDraft = runCurrentDraft > ratedCurrentA + 1e-6 || Boolean(sim?.kinematics.motor_overcurrent_risk);

  const toolheadPresets = catalog.toolhead_presets ?? [];
  const currentMountMaterial: ToolheadMountMaterial =
    spec.toolhead.mount_material ?? "abs_printed";
  const matchedToolheadPreset = toolheadPresets.find(
    (tp) =>
      tp.preset_id === spec.toolhead.preset_id &&
      Math.abs(tp.total_mass_g - spec.toolhead.total_mass_g) < 0.5 &&
      Math.round(Math.abs(tp.cog_offset_mm[1])) === specCogYAbs &&
      Math.round(Math.abs(tp.cog_offset_mm[2])) === specCogZAbs &&
      Math.round(Math.abs(tp.nozzle_offset_mm[2])) === specNozzleZAbs &&
      tp.mount_material === currentMountMaterial &&
      (tp.internal_stiffness_n_per_mm == null ||
        Math.round(tp.internal_stiffness_n_per_mm) === specToolheadStiffness)
  );
  const selectedToolheadPresetVal = matchedToolheadPreset
    ? matchedToolheadPreset.preset_id
    : "__custom__";
  const stiffnessTier = toolheadStiffnessTier(toolheadStiffnessDraft);

  return (
    <aside className="sidebar">
      <div className="tabs">
        <button className={`tab-btn ${tab === "params" ? "active" : ""}`} onClick={() => setTab("params")}>
          Printer
        </button>
        <button className={`tab-btn ${tab === "part_fea" ? "active" : ""}`} onClick={() => setTab("part_fea")}>
          Parts &amp; Topo
        </button>
        <button className={`tab-btn ${tab === "yaml" ? "active" : ""}`} onClick={() => setTab("yaml")}>
          YAML
        </button>
        <button className={`tab-btn ${tab === "gcode" ? "active" : ""}`} onClick={() => setTab("gcode")}>
          G-code
        </button>
      </div>

      <div className="sidebar-scroll">
        {tab === "params" && (
          <div className="hints-toggle-bar">
            <span className="hints-toggle-label">What each control affects</span>
            <button
              type="button"
              className={`hints-pill ${showHints ? "active" : ""}`}
              onClick={() => setShowHints((v) => !v)}
              title="Toggle plain-English explanations on printer controls"
            >
              Plain-English Hints: {showHints ? "ON" : "OFF"}
            </button>
          </div>
        )}

        {/* Preset & Filament selectors always available */}
        <div className="section-card">
          <div className="section-title">Preset &amp; Filament</div>
          <div className="field-row">
            <label>Printer Template</label>
            <select value={selectedPreset} onChange={(e) => onSelectPreset(e.target.value)}>
              {catalog.presets.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name} ({p.architecture})
                </option>
              ))}
            </select>
          </div>
          <div className="field-row">
            <label>Filament Material (Rheology &amp; Tg)</label>
            <select value={filamentId} onChange={(e) => onFilamentChange(e.target.value)}>
              {catalog.filaments.map((f) => (
                <option key={f.id} value={f.id}>
                  {f.name} ({f.print_temp_c}°C)
                </option>
              ))}
            </select>
          </div>
        </div>

        {tab === "params" && (
          <>
            <div className="section-card">
              <div className="section-title">Frame &amp; Structure (FEA)</div>
              <div className="field-row">
                <label>Architecture</label>
                <select
                  value={spec.architecture}
                  onChange={(e) => {
                    const nextVal = e.target.value as PrinterSpecData["architecture"];
                    update(
                      (d) => {
                        d.architecture = nextVal;
                      },
                      `Architecture: ${ARCH_LABELS[spec.architecture]} → ${ARCH_LABELS[nextVal]}`
                    );
                  }}
                >
                  <option value="corexy">CoreXY</option>
                  <option value="cartesian_bedslinger">Cartesian Bedslinger</option>
                  <option value="delta">Delta (3-Tower)</option>
                </select>
              </div>

              {spec.architecture === "corexy" && (
                <>
                  <div className="field-row">
                    <label>
                      <span>Gantry &amp; Z Architecture</span>
                      <span style={{ color: isFlyingGantry ? "var(--amber)" : "var(--accent)", fontWeight: 600 }}>
                        {isFlyingGantry ? "Flying Gantry" : "Fixed Gantry"}
                      </span>
                    </label>
                    <select
                      value={spec.bed.z_drive}
                      onChange={(e) => {
                        const nextZ = e.target.value;
                        const prevLabel = Z_DRIVE_SHORT_LABELS[spec.bed.z_drive] ?? spec.bed.z_drive;
                        const nextLabel = Z_DRIVE_SHORT_LABELS[nextZ] ?? nextZ;
                        update(
                          (d) => {
                            d.bed.z_drive = nextZ;
                          },
                          `Gantry Type: ${prevLabel} → ${nextLabel}`
                        );
                      }}
                    >
                      <option value="leadscrew_triple">Fixed Gantry — Triple-Z Bed (Trident/VzBot)</option>
                      <option value="belted_quad_z">Flying Gantry — Quad-Belted Z (Voron 2.4)</option>
                      <option value="leadscrew_dual">Fixed Gantry — Dual-Z Bed</option>
                      {spec.bed.z_drive === "leadscrew_single" && (
                        <option value="leadscrew_single">Fixed Gantry — Single-Z Bed</option>
                      )}
                    </select>
                  </div>
                  {showHints && (
                    <div className="hint-line">
                      Fixed gantry (bed moves Z) lets structural panels bolt straight to the Y-rail &amp; motor
                      beams; flying gantry (Voron 2.4) moves XY in Z, so panels only stiffen the outer cube.
                    </div>
                  )}
                </>
              )}

              {showHints && (
                <div className="hint-line">
                  Bigger/thicker extrusions = stiffer frame &amp; higher resonance, but heavier &amp; pricier
                </div>
              )}

              <div className="field-grid-3">
                <div className="field-row">
                  <label>Width mm</label>
                  <input
                    type="number"
                    value={spec.frame.width_mm}
                    onChange={(e) =>
                      updatePositiveNum(
                        e.target.value,
                        (d, v) => {
                          d.frame.width_mm = v;
                        },
                        (v) => `Frame Width: ${spec.frame.width_mm} → ${v} mm`
                      )
                    }
                  />
                </div>
                <div className="field-row">
                  <label>Depth mm</label>
                  <input
                    type="number"
                    value={spec.frame.depth_mm}
                    onChange={(e) =>
                      updatePositiveNum(
                        e.target.value,
                        (d, v) => {
                          d.frame.depth_mm = v;
                        },
                        (v) => `Frame Depth: ${spec.frame.depth_mm} → ${v} mm`
                      )
                    }
                  />
                </div>
                <div className="field-row">
                  <label>Height mm</label>
                  <input
                    type="number"
                    value={spec.frame.height_mm}
                    onChange={(e) =>
                      updatePositiveNum(
                        e.target.value,
                        (d, v) => {
                          d.frame.height_mm = v;
                        },
                        (v) => `Frame Height: ${spec.frame.height_mm} → ${v} mm`
                      )
                    }
                  />
                </div>
              </div>

              <div className="field-grid-2">
                <div className="field-row">
                  <label>Frame Profile</label>
                  <select
                    value={spec.frame.profile}
                    onChange={(e) => {
                      const nextVal = e.target.value;
                      update(
                        (d) => {
                          d.frame.profile = nextVal;
                        },
                        `Frame Profile: ${profileLabel(spec.frame.profile)} → ${profileLabel(nextVal)}`
                      );
                    }}
                  >
                    {catalog.profiles.map((p) => (
                      <option key={p.id} value={p.id}>
                        {p.name}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="field-row">
                  <label>Frame Material</label>
                  <select
                    value={spec.frame.material}
                    onChange={(e) => {
                      const nextVal = e.target.value;
                      update(
                        (d) => {
                          d.frame.material = nextVal;
                        },
                        `Frame Material: ${materialLabel(spec.frame.material)} → ${materialLabel(nextVal)}`
                      );
                    }}
                  >
                    {catalog.materials.map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.name}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {showHints && (
                <div className="hint-line">Cuts frame wobble &amp; ghosting dramatically for low cost</div>
              )}

              <div className="field-grid-2">
                <div className="field-row">
                  <label>Bracing</label>
                  <select
                    value={spec.frame.bracing}
                    onChange={(e) => {
                      const nextVal = e.target.value as PrinterSpecData["frame"]["bracing"];
                      update(
                        (d) => {
                          d.frame.bracing = nextVal;
                        },
                        `Bracing: ${BRACING_LABELS[spec.frame.bracing]} → ${BRACING_LABELS[nextVal]}`
                      );
                    }}
                  >
                    <option value="none">None (Open Frame)</option>
                    <option value="corner_gussets">Corner Gussets</option>
                    <option value="rear_cross">Rear Cross + Top</option>
                    <option value="full_truss">Full 3D Truss</option>
                  </select>
                </div>
                <div className="field-row">
                  <label>Corner Joint Type</label>
                  <select
                    value={currentCornerJoint}
                    onChange={(e) => {
                      const nextVal = e.target.value as CornerJointType;
                      const prevLbl =
                        CORNER_JOINT_OPTIONS.find((o) => o.id === currentCornerJoint)?.shortLabel ??
                        currentCornerJoint;
                      const nextLbl =
                        CORNER_JOINT_OPTIONS.find((o) => o.id === nextVal)?.shortLabel ?? nextVal;
                      update(
                        (d) => {
                          d.frame.corner_joint_type = nextVal;
                        },
                        `Corner Joint: ${prevLbl} → ${nextLbl}`
                      );
                    }}
                  >
                    {CORNER_JOINT_OPTIONS.map((opt) => (
                      <option key={opt.id} value={opt.id}>
                        {opt.label}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="field-row">
                <label>
                  <span>Extrusion Cavity Filling</span>
                  <span style={{ display: "inline-flex", alignItems: "center", gap: 4, color: currentFillingId !== "none" ? "var(--accent)" : "var(--muted)", fontWeight: 600 }}>
                    <span
                      className="swatch-dot"
                      style={{ background: selectedFilling.color_hex }}
                    />
                    {selectedFilling.short_name}
                  </span>
                </label>
                <select
                  value={currentFillingId}
                  onChange={(e) => {
                    const nextId = e.target.value as ExtrusionFillingType;
                    const nextItem =
                      extrusionFillings.find((f) => f.id === nextId) ?? extrusionFillings[0];
                    update(
                      (d) => {
                        d.frame.extrusion_filling = nextId;
                      },
                      `Extrusion Fill: ${selectedFilling.short_name} → ${nextItem.short_name}`
                    );
                  }}
                >
                  {extrusionFillings.map((f) => (
                    <option key={f.id} value={f.id}>
                      {f.name}
                    </option>
                  ))}
                </select>
                <div className="inline-readout-strip">
                  {currentFillingId === "none" ? (
                    <>
                      <span>Hollow extrusion</span>
                      <span>·</span>
                      <span>ζ = {fillingDampingRatio.toFixed(3)}</span>
                      <span>·</span>
                      <span>Base stiffness</span>
                    </>
                  ) : (
                    <>
                      <strong>+{fillingAddedMassKg.toFixed(2)} kg fill</strong>
                      <span>·</span>
                      <span>ζ = {fillingDampingRatio.toFixed(3)}</span>
                      <span>·</span>
                      <span style={{ color: selectedFilling.bonded ? "var(--accent)" : "var(--green)" }}>
                        {selectedFilling.bonded ? "Bonded +EI" : "Granular Damping"}
                      </span>
                    </>
                  )}
                </div>
                {showHints && (
                  <div className="hint-line">
                    Fillings apply to stationary frame extrusions only (never the moving X-beam).
                    Epoxy granite &amp; UHPC bond to cavity walls (+EI + damping); sand &amp; lead shot
                    damp resonance without raising moving mass.
                  </div>
                )}
              </div>

              <div className="sub-block">
                <div className="sub-block-title">
                  <span>Structural &amp; Enclosure Panels</span>
                  {panelsEnabled && (
                    <span className="sub-block-tag">
                      {currentPanelThickness}mm {panelMaterialShort(currentPanelMaterial)}
                    </span>
                  )}
                </div>

                {showHints && (
                  <div className="hint-line">
                    Bolted ACM, aluminum, or carbon panels act as shear walls that lock the frame square;
                    thicker/stiffer sheets boost resonance &amp; cut flex.
                  </div>
                )}

                <div className="field-row">
                  <label>Panel Mounting</label>
                  <select
                    value={effectivePanelMounting}
                    onChange={(e) => {
                      const nextMount = e.target.value as PrinterSpecData["frame"]["panel_mounting"];
                      const prevDesc = formatPanelSummary(spec.frame);
                      const nextFrame: PrinterSpecData["frame"] = {
                        ...spec.frame,
                        enclosed_panels: nextMount !== "none",
                        panel_mounting: nextMount,
                        panel_material: spec.frame.panel_material || "acrylic_pmma",
                        panel_thickness_mm: spec.frame.panel_thickness_mm || 3,
                      };
                      const nextDesc = formatPanelSummary(nextFrame);
                      update(
                        (d) => {
                          d.frame.enclosed_panels = nextMount !== "none";
                          d.frame.panel_mounting = nextMount;
                          if (!d.frame.panel_material) d.frame.panel_material = "acrylic_pmma";
                          if (!d.frame.panel_thickness_mm) d.frame.panel_thickness_mm = 3;
                        },
                        `Panels: ${prevDesc} → ${nextDesc}`
                      );
                    }}
                  >
                    <option value="none">Open (No Panels)</option>
                    <option value="clip_enclosure">Clip-In Enclosure</option>
                    <option value="bolted_structural">Bolted Structural Panels</option>
                  </select>
                </div>

                {panelsEnabled && (
                  <div className="field-grid-2">
                    <div className="field-row">
                      <label>Panel Material</label>
                      <select
                        value={currentPanelMaterial}
                        onChange={(e) => {
                          const nextMat = e.target.value;
                          const prevDesc = formatPanelSummary(spec.frame);
                          const nextDesc = formatPanelSummary({
                            ...spec.frame,
                            panel_material: nextMat,
                          });
                          update(
                            (d) => {
                              d.frame.panel_material = nextMat;
                            },
                            `Panels: ${prevDesc} → ${nextDesc}`
                          );
                        }}
                      >
                        {PANEL_MATERIALS.map((pm) => (
                          <option key={pm.id} value={pm.id}>
                            {pm.label}
                          </option>
                        ))}
                      </select>
                    </div>
                    <div className="field-row">
                      <label>Thickness (mm)</label>
                      <select
                        value={currentPanelThickness}
                        onChange={(e) => {
                          const nextThick = Number(e.target.value);
                          if (!Number.isFinite(nextThick) || nextThick <= 0) return;
                          const prevDesc = formatPanelSummary(spec.frame);
                          const nextDesc = formatPanelSummary({
                            ...spec.frame,
                            panel_thickness_mm: nextThick,
                          });
                          update(
                            (d) => {
                              d.frame.panel_thickness_mm = nextThick;
                            },
                            `Panels: ${prevDesc} → ${nextDesc}`
                          );
                        }}
                      >
                        {thicknessChoices.map((mm) => (
                          <option key={mm} value={mm}>
                            {mm} mm
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>
                )}

                {spec.architecture === "corexy" &&
                  panelsEnabled &&
                  effectivePanelMounting === "bolted_structural" && (
                    <div className="gantry-bolt-block">
                      {isFlyingGantry ? (
                        <>
                          <label className="checkbox-row disabled">
                            <input type="checkbox" checked={false} disabled />
                            <span>Bolt Panels Directly to Fixed Gantry Beams</span>
                          </label>
                          <div className="hint-line warn">
                            Unavailable on Flying Gantry (Voron 2.4) — the XY gantry moves up/down in Z, so
                            panels can only bolt to the outer cube.
                          </div>
                        </>
                      ) : (
                        <>
                          <label className="checkbox-row">
                            <input
                              type="checkbox"
                              checked={boltPanelsToGantry}
                              onChange={(e) => {
                                const nextVal = e.target.checked;
                                update(
                                  (d) => {
                                    d.frame.bolt_panels_to_gantry = nextVal;
                                  },
                                  nextVal
                                    ? "Panel Bolts: Outer Frame Only → Bolted to Fixed Gantry"
                                    : "Panel Bolts: Bolted to Fixed Gantry → Outer Frame Only"
                                );
                              }}
                            />
                            <span>Bolt Panels Directly to Fixed Gantry Beams</span>
                          </label>
                          {showHints && (
                            <div className="hint-line">
                              Ties side &amp; rear panels straight into the fixed Y-rail and rear motor
                              extrusions for a major stiffness &amp; resonance boost.
                            </div>
                          )}
                        </>
                      )}
                    </div>
                  )}

                <div className="slider-row" style={{ marginTop: 4 }}>
                  <label>
                    <span>Chamber Temp (°C)</span>
                    <span className="tension-readout">
                      <strong style={{ color: chamberTempDraft >= 55 ? "var(--amber)" : "var(--text)" }}>
                        {chamberTempDraft} °C
                      </strong>
                      <span className="kpi-sub">
                        {chamberTempDraft <= 25 ? "ambient" : `ΔT +${chamberTempDraft - 25} K`}
                      </span>
                    </span>
                  </label>
                  <div className="tension-control-row">
                    <input
                      type="range"
                      min={25}
                      max={85}
                      step={5}
                      value={Math.max(25, Math.min(85, chamberTempDraft))}
                      onChange={(e) => handleChamberTempSliderChange(Number(e.target.value))}
                      onPointerUp={() => commitChamberTemp(chamberTempDraft)}
                      onKeyUp={() => commitChamberTemp(chamberTempDraft)}
                    />
                    <input
                      type="number"
                      className="tension-num-input"
                      min={25}
                      max={85}
                      step={5}
                      value={chamberTempDraft}
                      onChange={(e) => {
                        const v = Number(e.target.value);
                        if (!Number.isFinite(v)) return;
                        setChamberTempDraft(v);
                        if (v >= 25 && v <= 85) commitChamberTemp(v);
                      }}
                      onBlur={() => commitChamberTemp(chamberTempDraft)}
                    />
                  </div>
                  {showHints && (
                    <div className="hint-line">
                      Heated chambers drive aluminum-steel bimetallic gantry bow, frame Z-drift, and
                      printed bracket E(T) softening near polymer Tg.
                    </div>
                  )}
                </div>
              </div>
            </div>

            <div className="section-card">
              <div className="section-title">Gantry, Belts &amp; Kinematics</div>

              {spec.architecture === "corexy" && (
                <div className="sub-block">
                  <div className="sub-block-title">
                    <span>XY Drive System</span>
                    <span className="sub-block-tag">
                      {driveMode === "awd" ? "4 Motors (AWD)" : "2 Motors (2WD)"}
                    </span>
                  </div>

                  {showHints && (
                    <div className="hint-line">
                      AWD adds 2 front XY motors for 2× torque &amp; half belt span; double-shear bearings
                      prevent pulley shaft bending under high tension.
                    </div>
                  )}

                  <div className="field-row">
                    <label>Drive Mode</label>
                    <div className="engine-toggle">
                      <button
                        type="button"
                        className={`engine-toggle-btn ${driveMode === "2wd" ? "active" : ""}`}
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
                  </div>

                  <div className="field-row">
                    <label>Pulley Mounts</label>
                    <div className="engine-toggle">
                      <button
                        type="button"
                        className={`engine-toggle-btn ${!doubleShearPulleys ? "active" : ""}`}
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
                </div>
              )}

              <div className="sub-block">
                <div className="sub-block-title">
                  <span>Gantry Brackets &amp; Motor Mounts</span>
                  <span
                    className={`sub-block-tag ${
                      bracketCreepRisk ? "danger" : selectedBracket.cnc_machined ? "ok" : ""
                    }`}
                  >
                    <span
                      className="swatch-dot"
                      style={{ background: selectedBracket.color_hex, marginRight: 4 }}
                    />
                    {selectedBracket.short_name}
                  </span>
                </div>

                {showHints && (
                  <div className="hint-line">
                    Motor mounts &amp; XY joints flex in series with belts. Printed ABS creeps above
                    ~45 N belt tension in heated chambers; annealed PPA-CF/PPS-CF or CNC aluminum
                    prevent creep &amp; sharpen resonance.
                  </div>
                )}

                <div className="field-row">
                  <label>Bracket Material</label>
                  <select
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
                </div>

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

              {showHints && (
                <div className="hint-line">Lighter X-beam = faster Y acceleration &amp; less ringing</div>
              )}

              <div className="field-grid-2">
                <div className="field-row">
                  <label>X Beam Profile</label>
                  <select
                    value={spec.gantry.x_beam_profile}
                    onChange={(e) => {
                      const nextVal = e.target.value;
                      update(
                        (d) => {
                          d.gantry.x_beam_profile = nextVal;
                        },
                        `X Beam Profile: ${profileLabel(spec.gantry.x_beam_profile)} → ${profileLabel(nextVal)}`
                      );
                    }}
                  >
                    {catalog.profiles.map((p) => (
                      <option key={p.id} value={p.id}>
                        {p.name}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="field-row">
                  <label>X Beam Material</label>
                  <select
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
                    {catalog.materials.map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.name}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="sub-block">
                <div className="sub-block-title">
                  <span>Linear Rails &amp; Mounting</span>
                  <span className="sub-block-tag">
                    X: {selectedXRailMount.short_name}
                    {spec.architecture === "corexy" ? ` · Y: ${selectedYRailMount.short_name}` : ""}
                  </span>
                </div>

                {showHints && (
                  <div className="hint-line">
                    Front = max Y stiffness; Top = max Z stiffness; Bottom = shorter nozzle drop &amp; less
                    pitch twist; Inner Y = max X stiffness.
                  </div>
                )}

                <div className={spec.architecture === "corexy" ? "field-grid-2" : "field-row"}>
                  <div className="field-row">
                    <label>X Rail Profile</label>
                    <select
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
                      <label>Y Rail Profile</label>
                      <select
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
                  <label>
                    <span>X Rail Mount</span>
                    <span style={{ color: "var(--text)", fontWeight: 600 }}>
                      {selectedXRailMount.name}
                    </span>
                  </label>
                  <div className="voltage-preset-group" style={{ gridTemplateColumns: "repeat(4, 1fr)" }}>
                    {xRailMounts.map((opt) => (
                      <button
                        key={opt.id}
                        type="button"
                        className={`voltage-pill-btn ${currentXRailMount === opt.id ? "active" : ""}`}
                        style={{ fontSize: 10 }}
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
                    <label>
                      <span>Y Rail Mount</span>
                      <span style={{ color: "var(--text)", fontWeight: 600 }}>
                        {selectedYRailMount.name}
                      </span>
                    </label>
                    <div className="voltage-preset-group" style={{ gridTemplateColumns: "repeat(3, 1fr)" }}>
                      {yRailMounts.map((opt) => (
                        <button
                          key={opt.id}
                          type="button"
                          className={`voltage-pill-btn ${currentYRailMount === opt.id ? "active" : ""}`}
                          style={{ fontSize: 10.5 }}
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
                    <label>Rail Preload</label>
                    <select
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
                    <label>Carriage Block</label>
                    <select
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
                </div>

                <div className="field-grid-2">
                  <div className="field-row">
                    <label>Rail Bolt Pitch</label>
                    <select
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
                  <div className="field-row">
                    <label>Thermal Rail Backer</label>
                    <select
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

                {simLinearRails && (
                  <div className="inline-readout-strip">
                    <span>
                      X-Beam Composite Boost:{" "}
                      <strong style={{ color: "var(--accent)" }}>
                        +{Math.round(simLinearRails.x_beam_ei_y_boost_pct)}% Y
                      </strong>{" "}
                      ·{" "}
                      <strong style={{ color: "var(--accent)" }}>
                        +{Math.round(simLinearRails.x_beam_ei_z_boost_pct)}% Z
                      </strong>
                    </span>
                    {spec.architecture === "corexy" && (
                      <>
                        <span>·</span>
                        <span>
                          Y-Rail{" "}
                          <strong style={{ color: "var(--green)" }}>
                            +{Math.round(simLinearRails.y_rail_ei_x_boost_pct)}% X
                          </strong>
                        </span>
                      </>
                    )}
                    <span>·</span>
                    <span>{Math.round(simLinearRails.total_rail_mass_g)} g steel</span>
                  </div>
                )}
              </div>

              {showHints && (
                <div className="hint-line">Wider/stiffer belts reduce stretch and sharpen corners</div>
              )}

              <div className="field-grid-2">
                <div className="field-row">
                  <label>Belt Width</label>
                  <select
                    value={spec.gantry.belt.type}
                    onChange={(e) => {
                      const nextVal = e.target.value as PrinterSpecData["gantry"]["belt"]["type"];
                      update(
                        (d) => {
                          d.gantry.belt.type = nextVal;
                        },
                        `Belt Width: ${BELT_LABELS[spec.gantry.belt.type]} → ${BELT_LABELS[nextVal]}`
                      );
                    }}
                  >
                    <option value="gt2_6mm">GT2 6mm</option>
                    <option value="gt2_9mm">GT2 9mm</option>
                    <option value="gt2_12mm">GT2 12mm</option>
                  </select>
                </div>
                <div className="field-row">
                  <label>Belt Core</label>
                  <select
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

              <div className="field-grid-2">
                <div className="field-row">
                  <label>Idler Shaft</label>
                  <select
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
                  <label>Bearing Play</label>
                  <select
                    value={currentBearingClass}
                    onChange={(e) => {
                      const nextVal = e.target.value as BearingClearanceClass;
                      const prevLbl =
                        BEARING_CLEARANCE_OPTIONS.find((o) => o.id === currentBearingClass)?.shortLabel ??
                        currentBearingClass;
                      const nextLbl =
                        BEARING_CLEARANCE_OPTIONS.find((o) => o.id === nextVal)?.shortLabel ?? nextVal;
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

              <div className="slider-row">
                <label>
                  <span>Belt Tension (N)</span>
                  <span className="tension-readout">
                    <strong>{tensionDraft} N</strong>
                    {beltPluckHz != null && (
                      <span className="tension-hz-pill">~{Math.round(beltPluckHz)} Hz @ 150 mm</span>
                    )}
                  </span>
                </label>
                <div className="tension-control-row">
                  <input
                    type="range"
                    min={15}
                    max={120}
                    step={1}
                    value={Math.max(15, Math.min(120, tensionDraft))}
                    onChange={(e) => handleTensionSliderChange(Number(e.target.value))}
                    onPointerUp={() => commitBeltTension(tensionDraft)}
                    onKeyUp={() => commitBeltTension(tensionDraft)}
                  />
                  <input
                    type="number"
                    className="tension-num-input"
                    min={15}
                    max={120}
                    step={1}
                    value={tensionDraft}
                    onChange={(e) => {
                      const v = Number(e.target.value);
                      if (!Number.isFinite(v)) return;
                      setTensionDraft(v);
                      if (v >= 15 && v <= 120) {
                        commitBeltTension(v);
                      }
                    }}
                    onBlur={() => commitBeltTension(tensionDraft)}
                  />
                </div>
                {(showHints || beltOvertension || shaftBendingRisk) && (
                  <div className={`hint-line ${beltOvertension ? "danger" : shaftBendingRisk ? "warn" : ""}`}>
                    {beltPluckHz != null && (
                      <span>
                        Pluck tune: <strong>~{Math.round(beltPluckHz)} Hz @ 150 mm</strong>
                        {beltMaxWorkingN != null ? ` (max ${Math.round(beltMaxWorkingN)} N)` : ""}.{" "}
                      </span>
                    )}
                    {beltOvertension
                      ? "Belt overtensioned — upgrade to 9mm/12mm Aramid!"
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
                  <label>Motor Model</label>
                  <select
                    value={selectedStepperValue}
                    onChange={(e) => {
                      const nextId = e.target.value;
                      const chosen = stepperCatalog.find((s) => s.preset_id === nextId);
                      if (!chosen) return;
                      const prevName = shortMotorName(xyStepper.name);
                      const nextName = shortMotorName(chosen.name);
                      update(
                        (d) => {
                          d.kinematics.xy_stepper.preset_id = chosen.preset_id;
                          d.kinematics.xy_stepper.name = chosen.name;
                          d.kinematics.xy_stepper.holding_torque_nm = chosen.holding_torque_nm;
                          d.kinematics.xy_stepper.rated_current_a = chosen.rated_current_a;
                          d.kinematics.xy_stepper.run_current_a = chosen.run_current_a;
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
                  <label>
                    <span>Run Current (Amps)</span>
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
                    <input
                      type="range"
                      min={0.4}
                      max={3.2}
                      step={0.1}
                      value={Math.max(0.4, Math.min(3.2, runCurrentDraft))}
                      onChange={(e) => handleRunCurrentSliderChange(Number(e.target.value))}
                      onPointerUp={() => commitRunCurrent(runCurrentDraft)}
                      onKeyUp={() => commitRunCurrent(runCurrentDraft)}
                    />
                    <input
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
                  <label>
                    <span>Bus Voltage (V)</span>
                    <span style={{ color: "var(--text)", fontWeight: 600 }}>
                      {xyStepper.supply_voltage_v}V DC
                    </span>
                  </label>
                  <div className="voltage-selector-row">
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
                    <input
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
                <div className="field-row">
                  <label>
                    <span>Drive Control</span>
                    <span style={{ color: "var(--accent)", fontWeight: 600 }}>
                      {DRIVE_TYPE_OPTIONS.find((o) => o.id === currentDriveType)?.shortLabel ??
                        currentDriveType}
                    </span>
                  </label>
                  <select
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
                          if (nextVal !== "open_loop_stepper" && d.kinematics.servo_vel_bw_hz == null) {
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
                      <label>
                        <span>Notch Filter (Hz)</span>
                        <span style={{ fontSize: 10, color: "var(--muted)" }}>blank = Off</span>
                      </label>
                      <input
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
                      <label>Vel BW (Hz)</label>
                      <input
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
                            (v) => `Servo Vel BW: ${Math.round(currentServoVelBw)} Hz → ${Math.round(v)} Hz`
                          )
                        }
                      />
                    </div>
                  </div>
                )}
              </div>

              <div className="field-grid-2">
                <div className="field-row">
                  <label>Max Vel (mm/s)</label>
                  <input
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
                  <label>Max Accel (mm/s²)</label>
                  <input
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
                <label>Input Shaper</label>
                <select
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
                  <option value="zvdd">ZVDD</option>
                  <option value="smooth_zv">SMOOTH_ZV (Kalico)</option>
                  <option value="smooth_ei">SMOOTH_EI (Kalico)</option>
                </select>
              </div>
            </div>

            <div className="section-card">
              <div className="section-title">
                <span>Toolhead CoG, Nozzle &amp; Mount</span>
                <span className="sub-block-tag">{spec.toolhead.total_mass_g}g</span>
              </div>

              {showHints && (
                <div className="hint-line">
                  We care about vibration AT THE NOZZLE TIP — lowering CoG &amp; nozzle drop and stiffening
                  the mount stops the X-beam from twisting like a lever arm during rapid Y reversals.
                </div>
              )}

              <div className="field-row">
                <label>Toolhead Preset</label>
                <select
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

              <div className="field-grid-2">
                <div className="field-row">
                  <label>Carriage / Mount</label>
                  <select
                    value={currentMountMaterial}
                    onChange={(e) => {
                      const nextMat = e.target.value as ToolheadMountMaterial;
                      update(
                        (d) => {
                          d.toolhead.mount_material = nextMat;
                          d.toolhead.preset_id = "custom";
                        },
                        `Toolhead Mount: ${mountMaterialShort(currentMountMaterial)} → ${mountMaterialShort(
                          nextMat
                        )}`
                      );
                    }}
                  >
                    {MOUNT_MATERIAL_OPTIONS.map((opt) => (
                      <option key={opt.id} value={opt.id}>
                        {opt.label}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="field-row">
                  <label>Toolhead Mass (g)</label>
                  <input
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
                </div>
              </div>

              <div className="slider-row">
                <label>
                  <span>Toolhead Internal Stiffness (N/mm)</span>
                  <span className="tension-readout">
                    <strong>{toolheadStiffnessDraft.toLocaleString()} N/mm</strong>
                    <span className="kpi-sub" style={{ color: stiffnessTier.color }}>
                      {stiffnessTier.label}
                    </span>
                  </span>
                </label>
                <div className="tension-control-row">
                  <input
                    type="range"
                    min={50}
                    max={3000}
                    step={25}
                    value={Math.max(50, Math.min(3000, toolheadStiffnessDraft))}
                    onChange={(e) => handleToolheadStiffnessSliderChange(Number(e.target.value))}
                    onPointerUp={() => commitToolheadStiffness(toolheadStiffnessDraft)}
                    onKeyUp={() => commitToolheadStiffness(toolheadStiffnessDraft)}
                  />
                  <input
                    type="number"
                    className="tension-num-input"
                    min={25}
                    max={20000}
                    step={25}
                    value={toolheadStiffnessDraft}
                    onChange={(e) => {
                      const v = Number(e.target.value);
                      if (!Number.isFinite(v)) return;
                      setToolheadStiffnessDraft(v);
                      if (v >= 25 && v <= 20000) commitToolheadStiffness(v);
                    }}
                    onBlur={() => {
                      const clamped = Math.max(25, Math.min(20000, Math.round(toolheadStiffnessDraft)));
                      setToolheadStiffnessDraft(clamped);
                      commitToolheadStiffness(clamped);
                    }}
                  />
                </div>
                {showHints && (
                  <div className="hint-line">
                    Accounts for flex inside the toolhead body &amp; hotend clamp (&lt;250 Soft
                    Printed · 250–700 Standard Printed · 700–1500 Rigid CF-Polymer · &gt;1500 CNC
                    Monoblock).
                  </div>
                )}
              </div>

              <div className="slider-row">
                <label>
                  <span>CoG Forward Y (mm)</span>
                  <span className="tension-readout">
                    <strong>{cogYDraft} mm</strong>
                    <span className="kpi-sub">
                      {cogYDraft === 0 ? "· centered on X-beam" : "in front of X-beam"}
                    </span>
                  </span>
                </label>
                <div className="tension-control-row">
                  <input
                    type="range"
                    min={0}
                    max={80}
                    step={1}
                    value={Math.max(0, Math.min(80, cogYDraft))}
                    onChange={(e) => handleCogYSliderChange(Number(e.target.value))}
                    onPointerUp={() => commitCogY(cogYDraft)}
                    onKeyUp={() => commitCogY(cogYDraft)}
                  />
                  <input
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
                <label>
                  <span>CoG Drop Z (mm)</span>
                  <span className="tension-readout">
                    <strong>{cogZDraft} mm</strong>
                    <span className="kpi-sub">
                      {cogZDraft === 0 ? "· centered on X-beam" : "below X-beam"}
                    </span>
                  </span>
                </label>
                <div className="tension-control-row">
                  <input
                    type="range"
                    min={0}
                    max={100}
                    step={1}
                    value={Math.max(0, Math.min(100, cogZDraft))}
                    onChange={(e) => handleCogZSliderChange(Number(e.target.value))}
                    onPointerUp={() => commitCogZ(cogZDraft)}
                    onKeyUp={() => commitCogZ(cogZDraft)}
                  />
                  <input
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

              <div className="slider-row">
                <label>
                  <span>Nozzle Drop Z (mm)</span>
                  <span className="tension-readout">
                    <strong style={{ color: "#fbbf24" }}>{nozzleZDraft} mm</strong>
                    <span className="kpi-sub">
                      {nozzleZDraft === 0 ? "· centered on X-beam" : "lever arm"}
                    </span>
                  </span>
                </label>
                <div className="tension-control-row">
                  <input
                    type="range"
                    min={0}
                    max={120}
                    step={1}
                    value={Math.max(0, Math.min(120, nozzleZDraft))}
                    onChange={(e) => handleNozzleZSliderChange(Number(e.target.value))}
                    onPointerUp={() => commitNozzleZ(nozzleZDraft)}
                    onKeyUp={() => commitNozzleZ(nozzleZDraft)}
                  />
                  <input
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
            </div>

            <div className="section-card">
              <div className="section-title">Hotend Thermal &amp; Cooling</div>

              {showHints && (
                <div className="hint-line">
                  Longer melt zone &amp; higher wattage unlock faster print flow (mm³/s)
                </div>
              )}

              <div className="field-grid-2">
                <div className="field-row">
                  <label>Heater Power (W)</label>
                  <input
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
                  <label>Melt Zone (mm)</label>
                  <input
                    type="number"
                    value={spec.toolhead.hotend.melt_zone_length_mm}
                    onChange={(e) =>
                      updatePositiveNum(
                        e.target.value,
                        (d, v) => {
                          d.toolhead.hotend.melt_zone_length_mm = v;
                        },
                        (v) => `Melt Zone: ${spec.toolhead.hotend.melt_zone_length_mm} → ${v} mm`
                      )
                    }
                  />
                </div>
              </div>

              <div className="field-grid-2">
                <div className="field-row">
                  <label>Nozzle Dia (mm)</label>
                  <input
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
                  <label>Cooling Fan (CFM)</label>
                  <input
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

              {showHints && (
                <div className="hint-line">
                  Titanium stops heat-creep clogs; higher CFM improves overhangs &amp; bridges
                </div>
              )}

              <div className="field-row">
                <label>Heatbreak Material</label>
                <select
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
                  <option value="titanium_grade5">Titanium Gr.5</option>
                  <option value="stainless_304">Stainless 304</option>
                  <option value="brass_c360">Brass C360</option>
                </select>
              </div>
            </div>

            <div className="section-card">
              <div className="section-title">
                <span>Build Plate &amp; Bed Mount</span>
                <span className="sub-block-tag">{bedTempDraft}°C</span>
              </div>

              {showHints && (
                <div className="hint-line">
                  Rigid 4-point mounts constrain aluminum expansion at high bed temps, causing thermal
                  &quot;taco&quot; buckling; 3-point Maxwell kinematic mounts keep the plate flat.
                </div>
              )}

              <div className="slider-row">
                <label>
                  <span>Bed Temp (°C)</span>
                  <span className="tension-readout">
                    <strong style={{ color: bedTempDraft >= 100 ? "var(--amber)" : "var(--text)" }}>
                      {bedTempDraft} °C
                    </strong>
                  </span>
                </label>
                <div className="tension-control-row">
                  <input
                    type="range"
                    min={25}
                    max={130}
                    step={5}
                    value={Math.max(25, Math.min(130, bedTempDraft))}
                    onChange={(e) => handleBedTempSliderChange(Number(e.target.value))}
                    onPointerUp={() => commitBedTemp(bedTempDraft)}
                    onKeyUp={() => commitBedTemp(bedTempDraft)}
                  />
                  <input
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
                <label>Bed Mount</label>
                <select
                  value={currentBedMount}
                  onChange={(e) => {
                    const nextVal = e.target.value as BedMountType;
                    const prevLbl =
                      BED_MOUNT_OPTIONS.find((o) => o.id === currentBedMount)?.shortLabel ??
                      currentBedMount;
                    const nextLbl =
                      BED_MOUNT_OPTIONS.find((o) => o.id === nextVal)?.shortLabel ?? nextVal;
                    update(
                      (d) => {
                        d.bed.mount_type = nextVal;
                      },
                      `Bed Mount: ${prevLbl} → ${nextLbl}`
                    );
                  }}
                >
                  {BED_MOUNT_OPTIONS.map((opt) => (
                    <option key={opt.id} value={opt.id}>
                      {opt.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </>
        )}

        {tab === "part_fea" && (
          <>
            <div className="section-card">
              <div className="section-title">
                <span>3D Solid Part FEA</span>
                <span style={{ color: feaEngine === "tet10" ? "var(--accent)" : "var(--muted)", fontSize: 10 }}>
                  {feaEngine === "tet10" ? "Tet10 (P2)" : "Voxel Hex8"}
                </span>
              </div>

              <div className="field-row">
                <label>Solver Engine</label>
                <div className="engine-toggle">
                  <button
                    type="button"
                    className={`engine-toggle-btn ${feaEngine === "tet10" ? "active" : ""}`}
                    onClick={() => setFeaEngine("tet10")}
                  >
                    Curved Tet10
                    <span className="engine-toggle-sub">Gmsh + scikit-fem</span>
                  </button>
                  <button
                    type="button"
                    className={`engine-toggle-btn ${feaEngine === "hex8" ? "active" : ""}`}
                    onClick={() => setFeaEngine("hex8")}
                  >
                    Fast Voxel Hex8
                    <span className="engine-toggle-sub">Structured Grid</span>
                  </button>
                </div>
              </div>

              <div className="field-row">
                <label>Parametric Bracket / Mount</label>
                <select value={partPreset} onChange={(e) => setPartPreset(e.target.value)}>
                  {catalog.part_presets.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.name}
                    </option>
                  ))}
                </select>
              </div>
              <div className="field-grid-2">
                <div className="field-row">
                  <label>Material</label>
                  <select
                    value={partMat}
                    onChange={(e) => {
                      setPartMat(e.target.value);
                      setPartFdm(e.target.value.includes("printed"));
                    }}
                  >
                    {partFeaMaterials.map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.name}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="field-row">
                  <label>FDM Z-Anisotropy</label>
                  <select value={partFdm ? "yes" : "no"} onChange={(e) => setPartFdm(e.target.value === "yes")}>
                    <option value="yes">FDM Layer Model</option>
                    <option value="no">Isotropic (CNC)</option>
                  </select>
                </div>
              </div>
              <div className="field-grid-3">
                <div className="field-row">
                  <label>Fx (N)</label>
                  <input type="number" value={fx} onChange={(e) => setFx(Number(e.target.value))} />
                </div>
                <div className="field-row">
                  <label>Fy (N)</label>
                  <input type="number" value={fy} onChange={(e) => setFy(Number(e.target.value))} />
                </div>
                <div className="field-row">
                  <label>Fz (N)</label>
                  <input type="number" value={fz} onChange={(e) => setFz(Number(e.target.value))} />
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
                <div style={{ fontSize: 10.5, color: "var(--muted)", marginBottom: 6, lineHeight: 1.3 }}>
                  <code>.step</code>/<code>.stp</code> → Curved Tet10 (Gmsh) • <code>.stl</code>/<code>.obj</code> → Voxel Hex8
                </div>
                <div className="field-grid-2" style={{ marginBottom: 6 }}>
                  <div className="field-row">
                    <label>Fixed Clamp Face</label>
                    <select value={fixedFace} onChange={(e) => setFixedFace(e.target.value)}>
                      <option value="min_z">Bottom (-Z)</option>
                      <option value="min_x">Left (-X)</option>
                      <option value="min_y">Front (-Y)</option>
                    </select>
                  </div>
                  <div className="field-row">
                    <label>Loaded Face</label>
                    <select value={loadFace} onChange={(e) => setLoadFace(e.target.value)}>
                      <option value="max_z">Top (+Z)</option>
                      <option value="max_x">Right (+X)</option>
                      <option value="max_y">Rear (+Y)</option>
                    </select>
                  </div>
                </div>
                <input
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
                <span style={{ color: "var(--purple)", fontSize: 10 }}>Generative</span>
              </div>
              <div style={{ fontSize: 11, color: "var(--muted)", lineHeight: 1.35 }}>
                Carves low-stress voxels from the selected preset &amp; load case above while maximizing
                stiffness-to-weight ratio.
              </div>

              <div className="slider-row">
                <label>
                  <span>Target Volume Fraction</span>
                  <strong style={{ color: "var(--text)" }}>{Math.round(targetVolFrac * 100)}%</strong>
                </label>
                <input
                  type="range"
                  min={0.25}
                  max={0.8}
                  step={0.05}
                  value={targetVolFrac}
                  onChange={(e) => setTargetVolFrac(Number(e.target.value))}
                />
              </div>

              <div className="slider-row">
                <label>
                  <span>SIMP Iterations</span>
                  <strong style={{ color: "var(--text)" }}>{topoIterations}</strong>
                </label>
                <input
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
              className="code-editor"
              value={yamlText}
              onChange={(e) => onYamlChange(e.target.value)}
              spellCheck={false}
            />
          </div>
        )}

        {tab === "gcode" && (
          <div className="section-card">
            <div className="section-title">Custom G-code Input</div>
            <p style={{ margin: 0, color: "var(--muted)", fontSize: 11.5 }}>
              Leave blank to run the built-in high-speed calibration cube &amp; ringing torture benchmark, or paste
              any Marlin/Klipper/OrcaSlicer G-code below.
            </p>
            <textarea
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
