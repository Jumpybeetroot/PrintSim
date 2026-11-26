import React, { useEffect, useRef, useState } from "react";
import {
  ARCH_LABELS,
  BRACING_LABELS,
  FRONT_BRACE_LABELS,
  TOP_BRACE_LABELS,
  Z_DRIVE_LABELS,
} from "../constants/printerLabels";
import {
  CatalogResponse,
  CornerJointType,
  ExtrusionFillingInfo,
  ExtrusionFillingType,
  FrameFeetType,
  FrontBraceType,
  PrinterSpecData,
  ProfileItem,
  SimulationResponse,
  TopBraceType,
} from "../types";
import {
  ModelOverrideControl,
  PresetAffectsStrip,
  clearModelOverrideGroup,
} from "./ModelOverrideControl";

interface FrameSizePreset {
  id: string;
  label: string;
  frame: [number, number, number];
  build: [number, number, number];
  bed: [number, number];
}

const COREXY_FRAME_SIZE_PRESETS: FrameSizePreset[] = [
  { id: "250", label: "250 mm", frame: [440, 440, 450], build: [250, 250, 230], bed: [250, 250] },
  { id: "300", label: "300 mm", frame: [490, 490, 500], build: [300, 300, 250], bed: [300, 300] },
  { id: "350", label: "350 mm", frame: [540, 540, 550], build: [350, 350, 300], bed: [350, 350] },
  { id: "400", label: "400 mm", frame: [590, 590, 600], build: [400, 400, 350], bed: [400, 400] },
];

const BEDSLINGER_FRAME_SIZE_PRESETS: FrameSizePreset[] = [
  { id: "250", label: "250 mm", frame: [390, 410, 450], build: [250, 250, 260], bed: [250, 250] },
  { id: "300", label: "300 mm", frame: [440, 460, 500], build: [300, 300, 300], bed: [300, 300] },
  { id: "350", label: "350 mm", frame: [490, 510, 550], build: [350, 350, 350], bed: [350, 350] },
  { id: "400", label: "400 mm", frame: [540, 560, 600], build: [400, 400, 400], bed: [400, 400] },
];

function inferProfileOuterWidthMm(profileId: string, profiles: ProfileItem[]): number {
  if (profileId.includes("4040")) return 40;
  if (profileId.includes("3030")) return 30;
  if (profileId.includes("2040") || profileId.includes("2020")) return 20;
  const found = profiles.find((p) => p.id === profileId);
  if (found && Number.isFinite(found.width_m) && found.width_m > 0) {
    return Math.round(found.width_m * 1000);
  }
  return 20;
}

function shortProfileCode(profileId: string, profileLabelFn: (id: string) => string): string {
  if (profileId.startsWith("extrusion_")) {
    return profileId.replace(/^extrusion_/, "");
  }
  return profileLabelFn(profileId).split(" ")[0] || profileId;
}

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

const PANEL_MATERIALS: Array<{ id: string; label: string; shortLabel: string }> = [
  { id: "acrylic_pmma", label: "Cast Acrylic (PMMA)", shortLabel: "Acrylic" },
  { id: "polycarbonate", label: "Polycarbonate (PC)", shortLabel: "Polycarbonate" },
  { id: "acm_dibond", label: "ACM / Dibond", shortLabel: "ACM / Dibond" },
  { id: "aluminum_5052_sheet", label: "Aluminum 5052 Sheet", shortLabel: "Aluminum 5052" },
  { id: "carbon_fiber_plate", label: "Carbon Fiber 3K Sheet", shortLabel: "Carbon Fiber 3K" },
];

const FRAME_EXTRUSION_IDS = new Set([
  "extrusion_2020",
  "extrusion_2040",
  "extrusion_3030",
  "extrusion_4040",
  "extrusion_3060",
]);

const FRAME_TUBE_IDS = new Set([
  "al_tube_20x20x2",
  "al_tube_30x30x3",
  "steel_tube_25x25x2",
]);

const FRAME_MATERIAL_IDS = new Set([
  "aluminum_6063_t5",
  "aluminum_6061_t6",
  "aluminum_7075_t6",
  "steel_4140",
  "stainless_304",
  "titanium_grade5",
]);

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
  const topSuffix = frame.top_panel === false ? ", No Top" : "";
  return `${frame.panel_thickness_mm}mm ${mat} (${mnt}${topSuffix})`;
}

const CORNER_JOINT_OPTIONS: Array<{ id: CornerJointType; label: string; shortLabel: string }> = [
  { id: "blind_joint", label: "Blind Joint (T-Slot)", shortLabel: "Blind Joint" },
  { id: "l_bracket", label: "L-Bracket (Die-Cast)", shortLabel: "L-Bracket" },
  { id: "gusset_plate", label: "Gusset Plate", shortLabel: "Gusset Plate" },
  { id: "corner_cube", label: "3-Way Corner Cube", shortLabel: "Corner Cube" },
  { id: "rigid", label: "Ideal Rigid", shortLabel: "Ideal Rigid" },
];

const FALLBACK_CORNER_JOINT_PARAMS: Record<
  CornerJointType,
  { k_t_n_per_mm: number; k_r_inplane_nm_rad: number; k_r_torsion_nm_rad: number; damping_delta: number }
> = {
  l_bracket: { k_t_n_per_mm: 3500, k_r_inplane_nm_rad: 4500, k_r_torsion_nm_rad: 1800, damping_delta: 0.005 },
  blind_joint: { k_t_n_per_mm: 12000, k_r_inplane_nm_rad: 22000, k_r_torsion_nm_rad: 7500, damping_delta: 0.003 },
  gusset_plate: { k_t_n_per_mm: 22000, k_r_inplane_nm_rad: 42000, k_r_torsion_nm_rad: 14000, damping_delta: 0.002 },
  corner_cube: { k_t_n_per_mm: 38000, k_r_inplane_nm_rad: 85000, k_r_torsion_nm_rad: 28000, damping_delta: 0.001 },
  rigid: { k_t_n_per_mm: 10000000, k_r_inplane_nm_rad: 1000000000, k_r_torsion_nm_rad: 1000000000, damping_delta: 0 },
};

const FALLBACK_FRAME_FEET_PARAMS: Record<
  FrameFeetType,
  { k_axial_n_per_mm: number; k_lateral_n_per_mm: number; damping_delta: number; base_mass_kg: number }
> = {
  rigid_bolted: { k_axial_n_per_mm: 0, k_lateral_n_per_mm: 0, damping_delta: 0, base_mass_kg: 0 },
  rubber_feet: { k_axial_n_per_mm: 360, k_lateral_n_per_mm: 140, damping_delta: 0.006, base_mass_kg: 0 },
  anti_vib_pads: { k_axial_n_per_mm: 85, k_lateral_n_per_mm: 32, damping_delta: 0.015, base_mass_kg: 0 },
  paver_on_foam: { k_axial_n_per_mm: 240, k_lateral_n_per_mm: 95, damping_delta: 0.022, base_mass_kg: 18 },
};

const FRAME_FEET_OPTIONS: Array<{
  id: FrameFeetType;
  label: string;
  shortLabel: string;
  hint: string;
}> = [
  {
    id: "rigid_bolted",
    label: "Rigid Bolted",
    shortLabel: "Rigid Bolted",
    hint: "Frame rigidly anchored to heavy benchtop — suppresses whole-frame sway so all modes are internal structural flex.",
  },
  {
    id: "rubber_feet",
    label: "Rubber / TPU Feet",
    shortLabel: "Rubber Feet",
    hint: "Compliant benchtop feet allow low-Hz whole-frame rocking (low nozzle-to-bed error!) while isolating bench vibrations.",
  },
  {
    id: "anti_vib_pads",
    label: "Anti-Vib Pads / HULA",
    shortLabel: "Anti-Vib Pads",
    hint: "Ultra-soft omnidirectional isolation (HULA/sorbothane) lets the whole chassis float together with minimal nozzle-to-bed print error.",
  },
  {
    id: "paver_on_foam",
    label: "Concrete Paver on Foam (18 kg Inert Base)",
    shortLabel: "Paver on Foam",
    hint: "18 kg concrete paver over acoustic foam decouples benchtop noise while adding massive base inertia to resist frame rocking.",
  },
];

const TEST_POS_PRESETS: Array<{
  id: string;
  label: string;
  pos: [number, number, number];
}> = [
  { id: "center", label: "Center", pos: [0.5, 0.5, 0.5] },
  { id: "front_left", label: "Front-Left", pos: [0.15, 0.15, 0.5] },
  { id: "top_z", label: "Top Z", pos: [0.5, 0.5, 0.85] },
  { id: "low_z", label: "Low Z", pos: [0.5, 0.5, 0.15] },
];

export interface DesignFrameSectionProps {
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

export const DesignFrameSection: React.FC<DesignFrameSectionProps> = ({
  spec,
  catalog,
  sim,
  showHints,
  advancedMode,
  update,
  profileLabel,
  materialLabel,
}) => {
  const [chamberTempDraft, setChamberTempDraft] = useState<number>(
    Math.round(spec.frame.chamber_temp_c ?? 25)
  );
  const chamberTempTimerRef = useRef<number | null>(null);

  const [gantryClearanceDraft, setGantryClearanceDraft] = useState<number>(
    Math.round(spec.frame.gantry_top_clearance_mm ?? 55)
  );
  const gantryClearanceTimerRef = useRef<number | null>(null);

  const [frameWidthDraft, setFrameWidthDraft] = useState<string>(
    String(Math.round(spec.frame.width_mm))
  );
  const [frameDepthDraft, setFrameDepthDraft] = useState<string>(
    String(Math.round(spec.frame.depth_mm))
  );
  const [frameHeightDraft, setFrameHeightDraft] = useState<string>(
    String(Math.round(spec.frame.height_mm))
  );
  const frameWidthInputRef = useRef<HTMLInputElement | null>(null);

  const [buildXDraft, setBuildXDraft] = useState<string>(
    String(Math.round(spec.kinematics.build_volume_mm[0]))
  );
  const [buildYDraft, setBuildYDraft] = useState<string>(
    String(Math.round(spec.kinematics.build_volume_mm[1]))
  );
  const [buildZDraft, setBuildZDraft] = useState<string>(
    String(Math.round(spec.kinematics.build_volume_mm[2]))
  );

  useEffect(() => {
    setChamberTempDraft(Math.round(spec.frame.chamber_temp_c ?? 25));
  }, [spec.frame.chamber_temp_c]);

  useEffect(() => {
    setGantryClearanceDraft(Math.round(spec.frame.gantry_top_clearance_mm ?? 55));
  }, [spec.frame.gantry_top_clearance_mm]);

  useEffect(() => {
    setFrameWidthDraft(String(Math.round(spec.frame.width_mm)));
  }, [spec.frame.width_mm]);

  useEffect(() => {
    setFrameDepthDraft(String(Math.round(spec.frame.depth_mm)));
  }, [spec.frame.depth_mm]);

  useEffect(() => {
    setFrameHeightDraft(String(Math.round(spec.frame.height_mm)));
  }, [spec.frame.height_mm]);

  useEffect(() => {
    setBuildXDraft(String(Math.round(spec.kinematics.build_volume_mm[0])));
  }, [spec.kinematics.build_volume_mm[0]]);

  useEffect(() => {
    setBuildYDraft(String(Math.round(spec.kinematics.build_volume_mm[1])));
  }, [spec.kinematics.build_volume_mm[1]]);

  useEffect(() => {
    setBuildZDraft(String(Math.round(spec.kinematics.build_volume_mm[2])));
  }, [spec.kinematics.build_volume_mm[2]]);

  useEffect(() => {
    return () => {
      if (chamberTempTimerRef.current != null) window.clearTimeout(chamberTempTimerRef.current);
      if (gantryClearanceTimerRef.current != null)
        window.clearTimeout(gantryClearanceTimerRef.current);
    };
  }, []);

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

  const specGantryClearanceMm = Math.round(spec.frame.gantry_top_clearance_mm ?? 55);
  const commitGantryClearance = (rawVal: number) => {
    const val = Math.max(25, Math.min(300, Math.round(rawVal)));
    if (gantryClearanceTimerRef.current != null) {
      window.clearTimeout(gantryClearanceTimerRef.current);
      gantryClearanceTimerRef.current = null;
    }
    if (val === specGantryClearanceMm) return;
    update(
      (d) => {
        d.frame.gantry_top_clearance_mm = val;
      },
      `Top-to-Gantry Clearance: ${specGantryClearanceMm} mm → ${val} mm`
    );
  };

  const handleGantryClearanceSliderChange = (nextVal: number) => {
    setGantryClearanceDraft(nextVal);
    if (gantryClearanceTimerRef.current != null)
      window.clearTimeout(gantryClearanceTimerRef.current);
    gantryClearanceTimerRef.current = window.setTimeout(
      () => commitGantryClearance(nextVal),
      220
    );
  };

  const currentCornerJoint: CornerJointType = spec.frame.corner_joint_type ?? "blind_joint";
  const effectivePanelMounting: PrinterSpecData["frame"]["panel_mounting"] = !spec.frame.enclosed_panels
    ? "none"
    : spec.frame.panel_mounting || "bolted_structural";
  const panelsEnabled = spec.frame.enclosed_panels && effectivePanelMounting !== "none";
  const currentPanelMaterial = spec.frame.panel_material || "acrylic_pmma";
  const currentPanelThickness = spec.frame.panel_thickness_mm || 3;
  const isFlyingGantry = spec.bed.z_drive === "belted_quad_z";
  const boltPanelsToGantry = spec.frame.bolt_panels_to_gantry ?? true;
  const topPanelInstalled = spec.frame.top_panel ?? true;
  const tridentBackboneEnabled = Boolean(spec.frame.trident_backbone);
  const simBackbone = sim?.geometry.trident_backbone ?? null;
  const currentTopBrace: TopBraceType = spec.frame.top_brace ?? "none";
  const currentFrontBrace: FrontBraceType = spec.frame.front_brace ?? "none";
  const simTopBrace =
    sim?.geometry.top_brace && sim.geometry.top_brace.id === currentTopBrace
      ? sim.geometry.top_brace
      : null;
  const simFrontBrace =
    sim?.geometry.front_brace && sim.geometry.front_brace.id === currentFrontBrace
      ? sim.geometry.front_brace
      : null;
  const topBraceCatalogItem = catalog.top_braces?.find((b) => b.id === currentTopBrace);
  const frontBraceCatalogItem = catalog.front_braces?.find((b) => b.id === currentFrontBrace);
  const hasCncOpenBracePlate =
    currentTopBrace === "cnc_halo_ring" ||
    currentFrontBrace === "cnc_portal_frame" ||
    currentFrontBrace === "header_and_skirt";
  const thicknessChoices = PANEL_THICKNESS_OPTIONS.includes(currentPanelThickness)
    ? PANEL_THICKNESS_OPTIONS
    : [...PANEL_THICKNESS_OPTIONS, currentPanelThickness].sort((a, b) => a - b);

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

  const sizePresets: FrameSizePreset[] | null =
    spec.architecture === "corexy"
      ? COREXY_FRAME_SIZE_PRESETS
      : spec.architecture === "cartesian_bedslinger"
        ? BEDSLINGER_FRAME_SIZE_PRESETS
        : null;

  const activeSizePresetId: string | null = sizePresets
    ? (sizePresets.find(
        (p) =>
          Math.round(spec.frame.width_mm) === p.frame[0] &&
          Math.round(spec.frame.depth_mm) === p.frame[1] &&
          Math.round(spec.frame.height_mm) === p.frame[2] &&
          Math.round(spec.kinematics.build_volume_mm[0]) === p.build[0] &&
          Math.round(spec.kinematics.build_volume_mm[1]) === p.build[1] &&
          Math.round(spec.kinematics.build_volume_mm[2]) === p.build[2] &&
          Math.round(spec.bed.size_mm[0]) === p.bed[0] &&
          Math.round(spec.bed.size_mm[1]) === p.bed[1]
      )?.id ?? null)
    : null;

  const commitFrameDimension = (
    axis: "width_mm" | "depth_mm" | "height_mm",
    raw: string
  ) => {
    const current = Math.round(spec.frame[axis]);
    const setDraft =
      axis === "width_mm"
        ? setFrameWidthDraft
        : axis === "depth_mm"
          ? setFrameDepthDraft
          : setFrameHeightDraft;
    const parsed = Number(raw);
    if (raw.trim() === "" || !Number.isFinite(parsed)) {
      setDraft(String(current));
      return;
    }
    const clamped = Math.max(200, Math.min(1500, Math.round(parsed)));
    setDraft(String(clamped));
    if (clamped === current) return;
    const label =
      axis === "width_mm"
        ? "Frame Width"
        : axis === "depth_mm"
          ? "Frame Depth"
          : "Frame Height";
    update(
      (d) => {
        d.frame[axis] = clamped;
      },
      `${label}: ${current} → ${clamped} mm`
    );
  };

  const commitBuildDim = (axis: 0 | 1 | 2, raw: string) => {
    const current = Math.round(spec.kinematics.build_volume_mm[axis]);
    const setDraft =
      axis === 0 ? setBuildXDraft : axis === 1 ? setBuildYDraft : setBuildZDraft;
    const parsed = Number(raw);
    if (raw.trim() === "" || !Number.isFinite(parsed)) {
      setDraft(String(current));
      return;
    }
    const clamped = Math.max(80, Math.min(1300, Math.round(parsed)));
    setDraft(String(clamped));
    if (clamped === current) return;
    const axisName = axis === 0 ? "X" : axis === 1 ? "Y" : "Z";
    update(
      (d) => {
        const nextVol: [number, number, number] = [
          d.kinematics.build_volume_mm[0],
          d.kinematics.build_volume_mm[1],
          d.kinematics.build_volume_mm[2],
        ];
        nextVol[axis] = clamped;
        d.kinematics.build_volume_mm = nextVol;
      },
      `Build Volume ${axisName}: ${current} → ${clamped} mm`
    );
  };

  const handleApplyFramePreset = (preset: FrameSizePreset) => {
    setFrameWidthDraft(String(preset.frame[0]));
    setFrameDepthDraft(String(preset.frame[1]));
    setFrameHeightDraft(String(preset.frame[2]));
    setBuildXDraft(String(preset.build[0]));
    setBuildYDraft(String(preset.build[1]));
    setBuildZDraft(String(preset.build[2]));
    update(
      (d) => {
        d.frame.width_mm = preset.frame[0];
        d.frame.depth_mm = preset.frame[1];
        d.frame.height_mm = preset.frame[2];
        d.kinematics.build_volume_mm = [...preset.build] as [number, number, number];
        d.bed.size_mm = [...preset.bed] as [number, number];
      },
      `Frame Size Preset: ${preset.label} (${preset.frame.join("×")} mm)`
    );
  };

  const handleFitBedAndBuildToFrame = () => {
    const parseOrCurrent = (raw: string, fallback: number) => {
      const n = Number(raw);
      return raw.trim() !== "" && Number.isFinite(n)
        ? Math.max(200, Math.min(1500, Math.round(n)))
        : Math.round(fallback);
    };
    const fw = parseOrCurrent(frameWidthDraft, spec.frame.width_mm);
    const fd = parseOrCurrent(frameDepthDraft, spec.frame.depth_mm);
    const fh = parseOrCurrent(frameHeightDraft, spec.frame.height_mm);
    const bx = Math.max(120, fw - 190);
    const by = Math.max(120, fd - 190);
    const bz = Math.max(120, fh - 240);
    setFrameWidthDraft(String(fw));
    setFrameDepthDraft(String(fd));
    setFrameHeightDraft(String(fh));
    setBuildXDraft(String(bx));
    setBuildYDraft(String(by));
    setBuildZDraft(String(bz));
    update(
      (d) => {
        d.frame.width_mm = fw;
        d.frame.depth_mm = fd;
        d.frame.height_mm = fh;
        d.kinematics.build_volume_mm = [bx, by, bz];
        d.bed.size_mm = [bx, by];
      },
      `Fit Bed & Build to Frame: ${bx}×${by}×${bz} mm`
    );
  };

  const frameProfileWidthMm = inferProfileOuterWidthMm(spec.frame.profile, catalog.profiles);
  const uprightProfileId = spec.frame.upright_profile || spec.frame.profile;
  const uprightProfileWidthMm = inferProfileOuterWidthMm(uprightProfileId, catalog.profiles);
  const outerWidthMm = Math.round(spec.frame.width_mm) + uprightProfileWidthMm;
  const outerDepthMm = Math.round(spec.frame.depth_mm) + uprightProfileWidthMm;
  const outerHeightMm = Math.round(spec.frame.height_mm) + frameProfileWidthMm;
  const frameProfileCode = shortProfileCode(spec.frame.profile, profileLabel);
  const uprightProfileCode = shortProfileCode(uprightProfileId, profileLabel);
  const footprintDetail =
    uprightProfileWidthMm !== frameProfileWidthMm
      ? `${uprightProfileCode} uprights + ${frameProfileCode} rails`
      : `${frameProfileCode} centerlines + ${frameProfileWidthMm} mm`;

  return (
    <div className="section-card" data-design-section="frame">
      <div className="section-title">Frame &amp; Structure (FEA)</div>
      <div className="field-row">
        <label htmlFor="df-architecture-1">Architecture</label>
        <select id="df-architecture-1"
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
            <label htmlFor="df-gantry-and-z-architecture-2">
              <span>Gantry &amp; Z Architecture</span>
              <span style={{ color: isFlyingGantry ? "var(--amber)" : "var(--accent)", fontWeight: 500 }}>
                {isFlyingGantry ? "Flying Gantry" : "Fixed Gantry"}
              </span>
            </label>
            <select id="df-gantry-and-z-architecture-2"
              value={spec.bed.z_drive}
              onChange={(e) => {
                const nextZ = e.target.value as PrinterSpecData["bed"]["z_drive"];
                const prevLabel = Z_DRIVE_LABELS[spec.bed.z_drive] ?? spec.bed.z_drive;
                const nextLabel = Z_DRIVE_LABELS[nextZ] ?? nextZ;
                update(
                  (d) => {
                    d.bed.z_drive = nextZ;
                    if (!d.bed.z_leadscrew_type || d.bed.z_leadscrew_type === "auto") {
                      clearModelOverrideGroup(d, "z_drive_mechanism");
                    }
                  },
                  `Gantry Type: ${prevLabel} → ${nextLabel}`
                );
              }}
            >
              <option value="leadscrew_triple">Fixed Gantry — Triple-Z Bed (Trident/VzBot)</option>
              <option value="belted_quad_z">Flying Gantry — Quad-Belted Z (Voron 2.4)</option>
              <option value="leadscrew_dual">Fixed Gantry — Dual-Z Bed</option>
              <option value="leadscrew_dual_belt_sync">
                Fixed Gantry — Belt-Synced Dual-Z (VzBot)
              </option>
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

      {sizePresets && (
        <div className="field-row">
          <label id="df-quick-frame-size-3">
            <span>Quick Frame Size</span>
            <span style={{ fontSize: 11, color: "var(--muted)" }}>
              {activeSizePresetId ? `${activeSizePresetId} Class` : "Custom"}
            </span>
          </label>
          <div role="group" aria-labelledby="df-quick-frame-size-3" className="frame-size-presets">
            {sizePresets.map((preset) => (
              <button
                key={preset.id}
                type="button"
                className={`frame-size-pill ${activeSizePresetId === preset.id ? "active" : ""}`}
 aria-pressed={activeSizePresetId === preset.id}
                title={`Set frame ${preset.frame.join("×")} mm, build ${preset.build.join("×")} mm, bed ${preset.bed.join("×")} mm`}
                onClick={() => handleApplyFramePreset(preset)}
              >
                {preset.label}
              </button>
            ))}
            <button
              type="button"
              className={`frame-size-pill ${activeSizePresetId === null ? "active" : ""}`}
 aria-pressed={activeSizePresetId === null}
              title="Enter custom frame centerline dimensions below"
              onClick={() => {
                frameWidthInputRef.current?.focus();
                frameWidthInputRef.current?.select();
              }}
            >
              Custom
            </button>
          </div>
          <PresetAffectsStrip
            visible={advancedMode}
            items={[
              {
                label: "Frame",
                value: `${Math.round(spec.frame.width_mm)}×${Math.round(spec.frame.depth_mm)}×${Math.round(spec.frame.height_mm)} mm`,
              },
              {
                label: "Build",
                value: `${Math.round(spec.kinematics.build_volume_mm[0])}×${Math.round(spec.kinematics.build_volume_mm[1])}×${Math.round(spec.kinematics.build_volume_mm[2])} mm`,
              },
              {
                label: "Bed",
                value: `${Math.round(spec.bed.size_mm[0])}×${Math.round(spec.bed.size_mm[1])} mm`,
              },
            ]}
            suffix="(editable below)"
          />
        </div>
      )}

      <div className="field-grid-3">
        <div className="field-row">
          <label htmlFor="df-width-mm-4">Width (mm)</label>
          <input id="df-width-mm-4"
            ref={frameWidthInputRef}
            type="number"
            min={200}
            max={1500}
            step={10}
            value={frameWidthDraft}
            onChange={(e) => setFrameWidthDraft(e.target.value)}
            onBlur={() => commitFrameDimension("width_mm", frameWidthDraft)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                commitFrameDimension("width_mm", frameWidthDraft);
                e.currentTarget.blur();
              } else if (e.key === "Escape") {
                setFrameWidthDraft(String(Math.round(spec.frame.width_mm)));
                e.currentTarget.blur();
              }
            }}
          />
        </div>
        <div className="field-row">
          <label htmlFor="df-depth-mm-5">Depth (mm)</label>
          <input id="df-depth-mm-5"
            type="number"
            min={200}
            max={1500}
            step={10}
            value={frameDepthDraft}
            onChange={(e) => setFrameDepthDraft(e.target.value)}
            onBlur={() => commitFrameDimension("depth_mm", frameDepthDraft)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                commitFrameDimension("depth_mm", frameDepthDraft);
                e.currentTarget.blur();
              } else if (e.key === "Escape") {
                setFrameDepthDraft(String(Math.round(spec.frame.depth_mm)));
                e.currentTarget.blur();
              }
            }}
          />
        </div>
        <div className="field-row">
          <label htmlFor="df-height-mm-6">Height (mm)</label>
          <input id="df-height-mm-6"
            type="number"
            min={200}
            max={1500}
            step={10}
            value={frameHeightDraft}
            onChange={(e) => setFrameHeightDraft(e.target.value)}
            onBlur={() => commitFrameDimension("height_mm", frameHeightDraft)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                commitFrameDimension("height_mm", frameHeightDraft);
                e.currentTarget.blur();
              } else if (e.key === "Escape") {
                setFrameHeightDraft(String(Math.round(spec.frame.height_mm)));
                e.currentTarget.blur();
              }
            }}
          />
        </div>
      </div>

      <div className="frame-footprint-note">
        Outer footprint:{" "}
        <strong>
          {outerWidthMm} × {outerDepthMm} × {outerHeightMm} mm
        </strong>{" "}
        ({footprintDetail})
      </div>

      <div className="field-row">
        <div className="frame-build-header">
          <span className="frame-build-title">Build Volume X / Y / Z (mm)</span>
          <button
            type="button"
            className="frame-sync-btn"
            title="Resize build volume & bed plate to safely fit inside the current frame dimensions"
            onClick={handleFitBedAndBuildToFrame}
          >
            Fit Bed &amp; Build to Frame
          </button>
        </div>
        <div className="field-grid-3">
          <div className="field-row">
            <label htmlFor="df-build-x-mm-7">Build X (mm)</label>
            <input id="df-build-x-mm-7"
              type="number"
              min={80}
              max={1300}
              step={10}
              value={buildXDraft}
              onChange={(e) => setBuildXDraft(e.target.value)}
              onBlur={() => commitBuildDim(0, buildXDraft)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  commitBuildDim(0, buildXDraft);
                  e.currentTarget.blur();
                } else if (e.key === "Escape") {
                  setBuildXDraft(String(Math.round(spec.kinematics.build_volume_mm[0])));
                  e.currentTarget.blur();
                }
              }}
            />
          </div>
          <div className="field-row">
            <label htmlFor="df-build-y-mm-8">Build Y (mm)</label>
            <input id="df-build-y-mm-8"
              type="number"
              min={80}
              max={1300}
              step={10}
              value={buildYDraft}
              onChange={(e) => setBuildYDraft(e.target.value)}
              onBlur={() => commitBuildDim(1, buildYDraft)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  commitBuildDim(1, buildYDraft);
                  e.currentTarget.blur();
                } else if (e.key === "Escape") {
                  setBuildYDraft(String(Math.round(spec.kinematics.build_volume_mm[1])));
                  e.currentTarget.blur();
                }
              }}
            />
          </div>
          <div className="field-row">
            <label htmlFor="df-build-z-mm-9">Build Z (mm)</label>
            <input id="df-build-z-mm-9"
              type="number"
              min={80}
              max={1300}
              step={10}
              value={buildZDraft}
              onChange={(e) => setBuildZDraft(e.target.value)}
              onBlur={() => commitBuildDim(2, buildZDraft)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  commitBuildDim(2, buildZDraft);
                  e.currentTarget.blur();
                } else if (e.key === "Escape") {
                  setBuildZDraft(String(Math.round(spec.kinematics.build_volume_mm[2])));
                  e.currentTarget.blur();
                }
              }}
            />
          </div>
        </div>
      </div>

      <div className="field-grid-2 stack">
        <div className="field-row">
          <label htmlFor="df-frame-profile-10">Frame Profile</label>
          <select id="df-frame-profile-10"
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
            {(() => {
              const extrusions = catalog.profiles.filter((p) => FRAME_EXTRUSION_IDS.has(p.id));
              const tubes = catalog.profiles.filter((p) => FRAME_TUBE_IDS.has(p.id));
              const hasCurrent =
                extrusions.some((p) => p.id === spec.frame.profile) ||
                tubes.some((p) => p.id === spec.frame.profile);
              return (
                <>
                  {!hasCurrent && (
                    <option value={spec.frame.profile}>
                      {profileLabel(spec.frame.profile)} (Custom)
                    </option>
                  )}
                  <optgroup label="T-Slot Extrusions">
                    {extrusions.map((p) => (
                      <option key={p.id} value={p.id}>
                        {p.name}
                      </option>
                    ))}
                  </optgroup>
                  {tubes.length > 0 && (
                    <optgroup label="Structural Frame Tubes">
                      {tubes.map((p) => (
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
          <label htmlFor="df-frame-material-11">Frame Material</label>
          <select id="df-frame-material-11"
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
            {(() => {
              const frameMats = catalog.materials.filter((m) => FRAME_MATERIAL_IDS.has(m.id));
              const hasCurrent = frameMats.some((m) => m.id === spec.frame.material);
              return (
                <>
                  {!hasCurrent && (
                    <option value={spec.frame.material}>
                      {materialLabel(spec.frame.material)} (Custom)
                    </option>
                  )}
                  {frameMats.map((m) => (
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

      {(() => {
        const extrusionProfiles = catalog.profiles.filter((p) =>
          p.id.startsWith("extrusion_")
        );
        const fallbackUprights = [
          { id: "extrusion_2020", name: "2020 Aluminum Extrusion" },
          { id: "extrusion_2040", name: "2040 Aluminum Extrusion" },
          { id: "extrusion_3030", name: "3030 Aluminum Extrusion" },
          { id: "extrusion_4040", name: "4040 Aluminum Extrusion" },
        ];
        const uprightOptions =
          extrusionProfiles.length > 0 ? extrusionProfiles : fallbackUprights;
        const currentUpright = spec.frame.upright_profile || "";
        const hasCustomUpright =
          Boolean(currentUpright) &&
          !uprightOptions.some((p) => p.id === currentUpright);
        const isMixed =
          Boolean(currentUpright) && currentUpright !== spec.frame.profile;

        return (
          <div className="field-row">
            <label htmlFor="df-corner-upright-profile-12">
              <span>Corner Upright Profile</span>
              {isMixed && (
                <span style={{ color: "var(--accent)", fontWeight: 500 }}>
                  Mixed Frame
                </span>
              )}
            </label>
            <select id="df-corner-upright-profile-12"
              value={currentUpright}
              onChange={(e) => {
                const raw = e.target.value;
                const nextUpright = raw ? raw : null;
                const prevLbl = spec.frame.upright_profile
                  ? profileLabel(spec.frame.upright_profile)
                  : "Same as Frame";
                const nextLbl = nextUpright
                  ? profileLabel(nextUpright)
                  : "Same as Frame";
                update(
                  (d) => {
                    d.frame.upright_profile = nextUpright;
                  },
                  `Corner Uprights: ${prevLbl} → ${nextLbl}`
                );
              }}
            >
              <option value="">
                Same as Frame ({profileLabel(spec.frame.profile)})
              </option>
              {uprightOptions.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
              {hasCustomUpright && (
                <option value={currentUpright}>
                  {profileLabel(currentUpright)}
                </option>
              )}
            </select>
            {showHints && (
              <div className="hint-line">
                Use heavier vertical corner columns (e.g. 4040 uprights + 2020
                horizontals, DoomCube style) for high frame rigidity without full-frame
                weight/cost.
              </div>
            )}
          </div>
        );
      })()}

      {spec.architecture === "corexy" && (
        <div className="slider-row">
          <label htmlFor="df-top-to-gantry-clearance-mm-13">
            <span>Top-to-Gantry Clearance (mm)</span>
            <span className="tension-readout">
              <strong>{gantryClearanceDraft} mm</strong>
              <span className="kpi-sub">
                {gantryClearanceDraft < 50
                  ? "compact"
                  : gantryClearanceDraft <= 60
                    ? "stock"
                    : `+${gantryClearanceDraft - 55} mm tophat`}
              </span>
            </span>
          </label>
          <div className="tension-control-row">
            <input id="df-top-to-gantry-clearance-mm-13"
              type="range"
              min={25}
              max={250}
              step={5}
              value={Math.max(25, Math.min(250, gantryClearanceDraft))}
              onChange={(e) => handleGantryClearanceSliderChange(Number(e.target.value))}
              onPointerUp={() => commitGantryClearance(gantryClearanceDraft)}
              onKeyUp={() => commitGantryClearance(gantryClearanceDraft)}
            />
            <input aria-label="Top-to-Gantry Clearance (mm) value"
              type="number"
              className="tension-num-input"
              min={25}
              max={300}
              step={5}
              value={gantryClearanceDraft}
              onChange={(e) => {
                const v = Number(e.target.value);
                if (!Number.isFinite(v)) return;
                setGantryClearanceDraft(v);
                if (v >= 25 && v <= 300) commitGantryClearance(v);
              }}
              onBlur={() => commitGantryClearance(gantryClearanceDraft)}
            />
          </div>
          {showHints && (
            <div className="hint-line">
              Adds headroom above the gantry for tall toolheads, umbilicals, or a tophat
              while lengthening the upper frame columns.
            </div>
          )}
        </div>
      )}

      {showHints && (
        <div className="hint-line">Cuts frame wobble &amp; ghosting dramatically for low cost</div>
      )}

      <div className="field-row">
        <label htmlFor="df-bracing-14">Bracing</label>
        <select id="df-bracing-14"
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
          <option value="top_cross">Top Frame X-Brace</option>
          <option value="rear_cross">Rear X-Brace + Top</option>
          <option value="side_cross">Side X-Braces (Left/Right)</option>
          <option value="rear_and_sides">Rear + Side X-Braces (Open Top)</option>
          <option value="full_truss">Full 3D Truss (All Faces)</option>
        </select>
      </div>

      {spec.architecture === "corexy" && (
        <div className="gantry-bolt-block">
          {isFlyingGantry ? (
            <>
              <label className="checkbox-row disabled">
                <input type="checkbox" checked={false} disabled />
                <span>Trident Backbone (Clee — 3mm Steel)</span>
              </label>
              <div className="hint-line warn">
                Designed for fixed-gantry Voron Trident (replaces 3 bottom Z-motor mounts);
                unavailable on flying gantry.
              </div>
            </>
          ) : (
            <>
              <label className="checkbox-row">
                <input
                  type="checkbox"
                  checked={tridentBackboneEnabled}
                  onChange={(e) => {
                    const nextVal = e.target.checked;
                    update(
                      (d) => {
                        d.frame.trident_backbone = nextVal;
                      },
                      nextVal
                        ? "Trident Backbone (Clee): Off → Installed (3mm Steel)"
                        : "Trident Backbone (Clee): Installed → Off"
                    );
                  }}
                />
                <span>Trident Backbone (Clee — 3mm Mild Steel)</span>
              </label>
              {tridentBackboneEnabled && (
                <div className="inline-readout-strip">
                  <strong>
                    +{(simBackbone && simBackbone.mass_kg > 0 ? simBackbone.mass_kg : 4.6).toFixed(2)} kg steel
                  </strong>
                  <span>·</span>
                  <span>
                    {(simBackbone && simBackbone.thickness_mm > 0 ? simBackbone.thickness_mm : 3).toFixed(0)}mm solid plate
                  </span>
                  {sim?.static_fea?.bed_deflection_um != null && (
                    <>
                      <span>·</span>
                      <span>Bed sag {sim.static_fea.bed_deflection_um.toFixed(1)} µm</span>
                    </>
                  )}
                  <span>·</span>
                  <span style={{ color: "var(--accent)" }}>
                    {(simBackbone && simBackbone.z_thrust_boost_factor > 0 ? simBackbone.z_thrust_boost_factor : 2.4).toFixed(1)}× Z-mount
                  </span>
                  <span>·</span>
                  <span>
                    +${(simBackbone && simBackbone.cost_usd > 0 ? simBackbone.cost_usd : 50).toFixed(0)}
                  </span>
                </div>
              )}
              {showHints && (
                <div className="hint-line">
                  West3D / Clee 3mm laser-cut powder-coated mild steel solid full-footprint
                  deck plate unifying all 3 bottom Z-motor mounts across the lower frame.
                </div>
              )}
            </>
          )}
        </div>
      )}

      {spec.architecture === "corexy" && (
        <>
          <div className="field-row">
            <label htmlFor="df-top-brace-open-center-15">
              <span>Top Brace (Open Center)</span>
              {currentTopBrace !== "none" && (
                <span style={{ color: "var(--accent)", fontWeight: 500 }}>
                  {topBraceCatalogItem?.short_label ??
                    (currentTopBrace === "cnc_halo_ring"
                      ? "CNC Halo"
                      : currentTopBrace === "rear_v_and_front_corners"
                        ? "Rear-V + Corners"
                        : "4× Corner Bars")}
                </span>
              )}
            </label>
            <select id="df-top-brace-open-center-15"
              value={currentTopBrace}
              onChange={(e) => {
                const nextVal = e.target.value as TopBraceType;
                update(
                  (d) => {
                    d.frame.top_brace = nextVal;
                  },
                  `Top Brace: ${TOP_BRACE_LABELS[currentTopBrace]} → ${TOP_BRACE_LABELS[nextVal]}`
                );
              }}
            >
              <option value="none">{TOP_BRACE_LABELS.none}</option>
              <option value="cnc_halo_ring">{TOP_BRACE_LABELS.cnc_halo_ring}</option>
              <option value="rear_v_and_front_corners">
                {TOP_BRACE_LABELS.rear_v_and_front_corners}
              </option>
              <option value="corner_45_bars">{TOP_BRACE_LABELS.corner_45_bars}</option>
            </select>
            {currentTopBrace !== "none" && (
              <div className="inline-readout-strip">
                {simTopBrace && simTopBrace.cutout_w_mm > 0 && simTopBrace.cutout_d_mm > 0 ? (
                  <strong>
                    {Math.round(simTopBrace.cutout_w_mm)}×{Math.round(simTopBrace.cutout_d_mm)} mm
                    open center
                  </strong>
                ) : (
                  <strong>
                    {Math.max(120, Math.round(spec.frame.width_mm - 100))}×
                    {Math.max(120, Math.round(spec.frame.depth_mm - 100))} mm open center
                  </strong>
                )}
                <span>•</span>
                <span>
                  +
                  {(simTopBrace && simTopBrace.mass_kg > 0
                    ? simTopBrace.mass_kg
                    : currentTopBrace === "cnc_halo_ring"
                      ? 1.18
                      : 0.35
                  ).toFixed(2)}{" "}
                  kg
                </span>
                <span>•</span>
                <span>
                  ~$
                  {Math.round(
                    simTopBrace && simTopBrace.cost_usd > 0
                      ? simTopBrace.cost_usd
                      : (topBraceCatalogItem?.cost_usd ??
                          (currentTopBrace === "cnc_halo_ring" ? 55 : 22))
                  )}
                </span>
              </div>
            )}
            {showHints && topBraceCatalogItem?.description && currentTopBrace !== "none" && (
              <div className="hint-line">{topBraceCatalogItem.description}</div>
            )}
          </div>

          <div className="field-row">
            <label htmlFor="df-front-brace-open-door-16">
              <span>Front Brace (Open Door)</span>
              {currentFrontBrace !== "none" && (
                <span style={{ color: "var(--accent)", fontWeight: 500 }}>
                  {frontBraceCatalogItem?.short_label ??
                    (currentFrontBrace === "cnc_portal_frame"
                      ? "CNC Portal"
                      : currentFrontBrace === "header_and_skirt"
                        ? "Header + Skirt"
                        : "4× Corner Bars")}
                </span>
              )}
            </label>
            <select id="df-front-brace-open-door-16"
              value={currentFrontBrace}
              onChange={(e) => {
                const nextVal = e.target.value as FrontBraceType;
                update(
                  (d) => {
                    d.frame.front_brace = nextVal;
                  },
                  `Front Brace: ${FRONT_BRACE_LABELS[currentFrontBrace]} → ${FRONT_BRACE_LABELS[nextVal]}`
                );
              }}
            >
              <option value="none">{FRONT_BRACE_LABELS.none}</option>
              <option value="cnc_portal_frame">{FRONT_BRACE_LABELS.cnc_portal_frame}</option>
              <option value="header_and_skirt">{FRONT_BRACE_LABELS.header_and_skirt}</option>
              <option value="corner_45_bars">{FRONT_BRACE_LABELS.corner_45_bars}</option>
            </select>
            {currentFrontBrace !== "none" && (
              <div className="inline-readout-strip">
                {simFrontBrace && simFrontBrace.cutout_w_mm > 0 && simFrontBrace.cutout_h_mm > 0 ? (
                  <strong>
                    {Math.round(simFrontBrace.cutout_w_mm)}×{Math.round(simFrontBrace.cutout_h_mm)}{" "}
                    mm clear door
                  </strong>
                ) : (
                  <strong>
                    {Math.max(120, Math.round(spec.frame.width_mm - 110))}×
                    {Math.max(120, Math.round(spec.frame.height_mm - 110))} mm clear door
                  </strong>
                )}
                <span>•</span>
                <span>
                  +
                  {(simFrontBrace && simFrontBrace.mass_kg > 0
                    ? simFrontBrace.mass_kg
                    : currentFrontBrace === "cnc_portal_frame"
                      ? 1.3
                      : currentFrontBrace === "header_and_skirt"
                        ? 0.95
                        : 0.35
                  ).toFixed(2)}{" "}
                  kg
                </span>
                <span>•</span>
                <span>
                  ~$
                  {Math.round(
                    simFrontBrace && simFrontBrace.cost_usd > 0
                      ? simFrontBrace.cost_usd
                      : (frontBraceCatalogItem?.cost_usd ??
                          (currentFrontBrace === "cnc_portal_frame"
                            ? 60
                            : currentFrontBrace === "header_and_skirt"
                              ? 45
                              : 22))
                  )}
                </span>
              </div>
            )}
            {showHints && frontBraceCatalogItem?.description && currentFrontBrace !== "none" && (
              <div className="hint-line">{frontBraceCatalogItem.description}</div>
            )}
          </div>

          {hasCncOpenBracePlate && (
            <div className="hint-line" style={{ color: "var(--accent)", marginTop: -2 }}>
              CNC Halo / Front Portal / Header+Skirt plates are automatically included in{" "}
              <strong>DXF Plates</strong> export.
            </div>
          )}
        </>
      )}

      <details className="sub-accordion">
        <summary>
          <span>Advanced Frame (Joints &amp; Cavity Fill)</span>
          <span className="sub-accordion-tag">
            {CORNER_JOINT_OPTIONS.find((o) => o.id === currentCornerJoint)?.shortLabel ??
              currentCornerJoint}{" "}
            · {selectedFilling.short_name}
          </span>
        </summary>
        <div className="sub-accordion-body">
          <div className="field-row">
            <label htmlFor="df-corner-joint-type-17">Corner Joint Type</label>
            <select id="df-corner-joint-type-17"
              value={currentCornerJoint}
              onChange={(e) => {
                const nextVal = e.target.value as CornerJointType;
                const prevLbl =
                  CORNER_JOINT_OPTIONS.find((o) => o.id === currentCornerJoint)?.shortLabel ??
                  currentCornerJoint;
                const nextLbl =
                  CORNER_JOINT_OPTIONS.find((o) => o.id === nextVal)?.shortLabel ?? nextVal;
                const hadOv = Boolean(
                  spec.model_overrides?.corner_joint &&
                    Object.values(spec.model_overrides.corner_joint).some((v) => v != null)
                );
                update(
                  (d) => {
                    d.frame.corner_joint_type = nextVal;
                    clearModelOverrideGroup(d, "corner_joint");
                  },
                  `Corner Joint: ${prevLbl} → ${nextLbl}${hadOv ? " (override reset)" : ""}`
                );
              }}
            >
              {CORNER_JOINT_OPTIONS.map((opt) => (
                <option key={opt.id} value={opt.id}>
                  {opt.label}
                </option>
              ))}
            </select>
            {(() => {
              const catJoint = catalog.corner_joints?.find((c) => c.id === currentCornerJoint);
              const fb = FALLBACK_CORNER_JOINT_PARAMS[currentCornerJoint];
              return (
                <ModelOverrideControl
                  visible={advancedMode}
                  idPrefix="ov-corner-joint"
                  title="Corner Joint"
                  group="corner_joint"
                  spec={spec}
                  sim={sim}
                  update={update}
                  fields={[
                    {
                      key: "k_t_n_per_mm",
                      label: "Translational k",
                      unit: "N/mm",
                      min: 10,
                      max: 100000000,
                      step: 500,
                      decimals: 0,
                      fallbackBaseline: catJoint?.k_t_n_per_mm ?? fb.k_t_n_per_mm,
                      tooltip: "Corner joint translational shear/axial spring stiffness",
                    },
                    {
                      key: "k_r_inplane_nm_rad",
                      label: "In-Plane k_r",
                      unit: "N·m/rad",
                      min: 10,
                      max: 10000000000,
                      step: 500,
                      decimals: 0,
                      fallbackBaseline: catJoint?.k_r_inplane_nm_rad ?? fb.k_r_inplane_nm_rad,
                      tooltip: "In-plane bending moment stiffness of each frame corner joint",
                    },
                    {
                      key: "k_r_torsion_nm_rad",
                      label: "Torsion k_t",
                      unit: "N·m/rad",
                      min: 10,
                      max: 10000000000,
                      step: 500,
                      decimals: 0,
                      fallbackBaseline: catJoint?.k_r_torsion_nm_rad ?? fb.k_r_torsion_nm_rad,
                      tooltip: "Out-of-plane / torsional rotational stiffness of each frame corner joint",
                    },
                    {
                      key: "damping_delta",
                      label: "Joint Damping",
                      unit: "ζ",
                      min: 0,
                      max: 0.15,
                      step: 0.001,
                      decimals: 3,
                      fallbackBaseline: catJoint?.damping_delta ?? fb.damping_delta,
                      tooltip: "Iwan micro-slip modal damping increment contributed by corner joints",
                    },
                  ]}
                />
              );
            })()}
          </div>

          <div className="field-row">
            <label htmlFor="df-extrusion-cavity-filling-18">
              <span>Extrusion Cavity Filling</span>
              <span
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: 4,
                  color: currentFillingId !== "none" ? "var(--accent)" : "var(--muted)",
                  fontWeight: 500,
                }}
              >
                <span
                  className="swatch-dot"
                  style={{ background: selectedFilling.color_hex }}
                />
                {selectedFilling.short_name}
              </span>
            </label>
            <select id="df-extrusion-cavity-filling-18"
              value={currentFillingId}
              onChange={(e) => {
                const nextId = e.target.value as ExtrusionFillingType;
                const nextItem =
                  extrusionFillings.find((f) => f.id === nextId) ?? extrusionFillings[0];
                const hadOv = Boolean(
                  spec.model_overrides?.extrusion_filling &&
                    Object.values(spec.model_overrides.extrusion_filling).some((v) => v != null)
                );
                update(
                  (d) => {
                    d.frame.extrusion_filling = nextId;
                    clearModelOverrideGroup(d, "extrusion_filling");
                  },
                  `Extrusion Fill: ${selectedFilling.short_name} → ${nextItem.short_name}${hadOv ? " (override reset)" : ""}`
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
                  <span
                    style={{
                      color: selectedFilling.bonded ? "var(--accent)" : "var(--green)",
                    }}
                  >
                    {selectedFilling.bonded ? "Bonded +EI" : "Granular Damping"}
                  </span>
                </>
              )}
            </div>
            {showHints && (
              <div className="hint-line">
                Fillings apply to stationary frame extrusions only (never the moving X-beam).
                Epoxy granite &amp; UHPC bond to cavity walls (+EI + damping); sand &amp;
                lead shot damp resonance without raising moving mass.
              </div>
            )}
            <ModelOverrideControl
              visible={advancedMode}
              idPrefix="ov-extrusion-filling"
              title="Extrusion Cavity Fill"
              group="extrusion_filling"
              spec={spec}
              sim={sim}
              update={update}
              fields={[
                {
                  key: "density_kg_m3",
                  label: "Density",
                  unit: "kg/m³",
                  min: 0,
                  max: 15000,
                  step: 50,
                  decimals: 0,
                  fallbackBaseline: selectedFilling.density_kg_m3,
                  tooltip: "Bulk density of material packed inside extrusion hollow core",
                },
                {
                  key: "youngs_modulus_gpa",
                  label: "Core E",
                  unit: "GPa",
                  min: 0,
                  max: 120,
                  step: 0.5,
                  decimals: 2,
                  fallbackBaseline: Number((selectedFilling.youngs_modulus_pa / 1e9).toFixed(2)),
                  tooltip: "Elastic modulus of cured cavity core (used when bonded to walls)",
                },
                {
                  key: "damping_boost",
                  label: "Damping Boost",
                  unit: "ζ",
                  min: 0,
                  max: 0.2,
                  step: 0.002,
                  decimals: 3,
                  fallbackBaseline: selectedFilling.damping_boost,
                  tooltip: "Added structural modal damping ratio from granular or resin fill",
                },
                {
                  key: "bonded",
                  label: "Bonded to Cavity Walls (+EI)",
                  type: "boolean",
                  fallbackBaseline: selectedFilling.bonded,
                  tooltip: "When enabled, core modulus adds composite bending stiffness to frame beams",
                },
              ]}
            />
          </div>
        </div>
      </details>

      <div className="sub-block">
        <div className="sub-block-title">
          <span>Structural &amp; Enclosure Panels</span>
          {panelsEnabled && (
            <span className="sub-block-tag">
              {currentPanelThickness}mm {panelMaterialShort(currentPanelMaterial)}
              {spec.architecture === "corexy" && !topPanelInstalled ? " · No Top" : ""}
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
          <label htmlFor="df-panel-mounting-19">Panel Mounting</label>
          <select id="df-panel-mounting-19"
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
          <div className="field-grid-2 wide-first">
            <div className="field-row">
              <label htmlFor="df-panel-material-20">Panel Material</label>
              <select id="df-panel-material-20"
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
                {!PANEL_MATERIALS.some((pm) => pm.id === currentPanelMaterial) && (
                  <option value={currentPanelMaterial}>
                    {materialLabel(currentPanelMaterial)} (Custom)
                  </option>
                )}
                {PANEL_MATERIALS.map((pm) => (
                  <option key={pm.id} value={pm.id}>
                    {pm.label}
                  </option>
                ))}
              </select>
            </div>
            <div className="field-row">
              <label htmlFor="df-thickness-mm-21">Thickness (mm)</label>
              <select id="df-thickness-mm-21"
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

        {spec.architecture === "corexy" && panelsEnabled && (
          <label className="checkbox-row" style={{ marginTop: 4 }}>
            <input
              type="checkbox"
              checked={topPanelInstalled}
              onChange={(e) => {
                const nextVal = e.target.checked;
                update(
                  (d) => {
                    d.frame.top_panel = nextVal;
                  },
                  nextVal
                    ? "Top Panel: Open Top → Installed"
                    : "Top Panel: Installed → Open Top"
                );
              }}
            />
            <span>Include Top Panel</span>
          </label>
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
          <label htmlFor="df-chamber-temp-c-22">
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
            <input id="df-chamber-temp-c-22"
              type="range"
              min={25}
              max={85}
              step={5}
              value={Math.max(25, Math.min(85, chamberTempDraft))}
              onChange={(e) => handleChamberTempSliderChange(Number(e.target.value))}
              onPointerUp={() => commitChamberTemp(chamberTempDraft)}
              onKeyUp={() => commitChamberTemp(chamberTempDraft)}
            />
            <input aria-label="Chamber Temp (°C) value"
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
  );
};

export interface DesignAnchoringSectionProps {
  spec: PrinterSpecData;
  catalog: CatalogResponse;
  sim: SimulationResponse | null;
  showHints: boolean;
  advancedMode: boolean;
  toolheadRelPos: [number, number, number];
  onToolheadRelPosChange: (pos: [number, number, number], changeLabel?: string) => void;
  panelMeshN: number;
  onPanelMeshNChange: (n: number) => void;
  update: (mutator: (draft: PrinterSpecData) => void, changeLabel?: string) => void;
}

export const DesignAnchoringSection: React.FC<DesignAnchoringSectionProps> = ({
  spec,
  catalog,
  sim,
  showHints,
  advancedMode,
  toolheadRelPos,
  onToolheadRelPosChange,
  panelMeshN,
  onPanelMeshNChange,
  update,
}) => {
  const [posDraft, setPosDraft] = useState<[number, number, number]>(toolheadRelPos);
  const posTimerRef = useRef<number | null>(null);

  useEffect(() => {
    setPosDraft(toolheadRelPos);
  }, [toolheadRelPos[0], toolheadRelPos[1], toolheadRelPos[2]]);

  useEffect(() => {
    return () => {
      if (posTimerRef.current != null) window.clearTimeout(posTimerRef.current);
    };
  }, []);

  const currentFeetType: FrameFeetType = spec.frame.feet_type ?? "rigid_bolted";
  const frameFeetCatalogEntry = catalog.frame_feet?.find((f) => f.id === currentFeetType);
  const frameFeetOptionMeta =
    FRAME_FEET_OPTIONS.find((o) => o.id === currentFeetType) ?? FRAME_FEET_OPTIONS[0];

  const commitToolheadRelPos = (nextPos: [number, number, number], label?: string) => {
    if (posTimerRef.current != null) {
      window.clearTimeout(posTimerRef.current);
      posTimerRef.current = null;
    }
    if (
      Math.abs(nextPos[0] - toolheadRelPos[0]) < 0.005 &&
      Math.abs(nextPos[1] - toolheadRelPos[1]) < 0.005 &&
      Math.abs(nextPos[2] - toolheadRelPos[2]) < 0.005
    ) {
      return;
    }
    onToolheadRelPosChange(nextPos, label);
  };

  const handlePosAxisSlider = (axisIdx: 0 | 1 | 2, pctVal: number) => {
    const frac = Math.max(0.1, Math.min(0.9, Math.round(pctVal) / 100));
    const next: [number, number, number] = [posDraft[0], posDraft[1], posDraft[2]];
    next[axisIdx] = frac;
    setPosDraft(next);
    if (posTimerRef.current != null) window.clearTimeout(posTimerRef.current);
    posTimerRef.current = window.setTimeout(() => commitToolheadRelPos(next), 240);
  };

  return (
    <div className="section-card" data-design-section="sim">
      <details className="sub-accordion">
        <summary>
          <span>Simulation &amp; Anchoring</span>
          <span className="sub-accordion-tag">
            {frameFeetOptionMeta.shortLabel} · {panelMeshN}×{panelMeshN}
          </span>
        </summary>
        <div className="sub-accordion-body">
          <div className="field-row">
            <label htmlFor="df-benchtop-feet-23">
              <span>Benchtop Feet</span>
              <span
                style={{
                  color: currentFeetType === "rigid_bolted" ? "var(--muted)" : "var(--accent)",
                  fontWeight: 500,
                }}
              >
                {frameFeetOptionMeta.shortLabel}
              </span>
            </label>
            <select id="df-benchtop-feet-23"
              value={currentFeetType}
              onChange={(e) => {
                const nextVal = e.target.value as FrameFeetType;
                const prevLbl =
                  FRAME_FEET_OPTIONS.find((o) => o.id === currentFeetType)?.shortLabel ??
                  currentFeetType;
                const nextLbl =
                  FRAME_FEET_OPTIONS.find((o) => o.id === nextVal)?.shortLabel ?? nextVal;
                const hadOv = Boolean(
                  spec.model_overrides?.frame_feet &&
                    Object.values(spec.model_overrides.frame_feet).some((v) => v != null)
                );
                update(
                  (d) => {
                    d.frame.feet_type = nextVal;
                    clearModelOverrideGroup(d, "frame_feet");
                  },
                  `Benchtop Feet: ${prevLbl} → ${nextLbl}${hadOv ? " (override reset)" : ""}`
                );
              }}
            >
              {FRAME_FEET_OPTIONS.map((opt) => {
                const catItem = catalog.frame_feet?.find((f) => f.id === opt.id);
                return (
                  <option key={opt.id} value={opt.id}>
                    {catItem?.name ?? opt.label}
                  </option>
                );
              })}
            </select>
            {showHints && (
              <div className="hint-line">
                {frameFeetCatalogEntry?.notes || frameFeetOptionMeta.hint}
              </div>
            )}
            {(() => {
              const fb = FALLBACK_FRAME_FEET_PARAMS[currentFeetType];
              return (
                <ModelOverrideControl
                  visible={advancedMode}
                  idPrefix="ov-frame-feet"
                  title="Benchtop Feet"
                  group="frame_feet"
                  spec={spec}
                  sim={sim}
                  update={update}
                  note={
                    currentFeetType === "rigid_bolted"
                      ? "0 N/mm = rigidly anchored to benchtop; enter positive stiffness to model compliant feet."
                      : undefined
                  }
                  fields={[
                    {
                      key: "k_axial_n_per_mm",
                      label: "Vertical k_z",
                      unit: "N/mm",
                      min: 0,
                      max: 100000,
                      step: 10,
                      decimals: 0,
                      fallbackBaseline: frameFeetCatalogEntry?.k_axial_n_per_mm ?? fb.k_axial_n_per_mm,
                      tooltip: "Per-foot vertical compression stiffness",
                    },
                    {
                      key: "k_lateral_n_per_mm",
                      label: "Lateral k_xy",
                      unit: "N/mm",
                      min: 0,
                      max: 100000,
                      step: 5,
                      decimals: 0,
                      fallbackBaseline: frameFeetCatalogEntry?.k_lateral_n_per_mm ?? fb.k_lateral_n_per_mm,
                      tooltip: "Per-foot horizontal shear stiffness",
                    },
                    {
                      key: "damping_delta",
                      label: "Foot Damping",
                      unit: "ζ",
                      min: 0,
                      max: 0.15,
                      step: 0.002,
                      decimals: 3,
                      fallbackBaseline: frameFeetCatalogEntry?.damping_delta ?? fb.damping_delta,
                      tooltip: "Added modal damping ratio from elastomer feet or foam pad",
                    },
                    {
                      key: "base_mass_kg",
                      label: "Base Mass",
                      unit: "kg",
                      min: 0,
                      max: 200,
                      step: 1,
                      decimals: 1,
                      fallbackBaseline: frameFeetCatalogEntry?.base_mass_kg ?? fb.base_mass_kg,
                      tooltip: "Inertial base mass rigidly coupled to bottom frame (e.g. concrete paver)",
                    },
                  ]}
                />
              );
            })()}
          </div>

          <div className="sub-block">
            <div className="sub-block-title">
              <span>Test Position (X, Y, Z)</span>
              <span className="sub-block-tag">
                {Math.round(posDraft[0] * 100)} / {Math.round(posDraft[1] * 100)} /{" "}
                {Math.round(posDraft[2] * 100)}%
              </span>
            </div>

            <div className="chip-group">
              {TEST_POS_PRESETS.map((preset) => {
                const isActive =
                  Math.abs(posDraft[0] - preset.pos[0]) < 0.03 &&
                  Math.abs(posDraft[1] - preset.pos[1]) < 0.03 &&
                  Math.abs(posDraft[2] - preset.pos[2]) < 0.03;
                return (
                  <button
                    key={preset.id}
                    type="button"
                    className={`chip-btn ${isActive ? "active" : ""}`}
 aria-pressed={isActive}
                    onClick={() => {
                      setPosDraft(preset.pos);
                      commitToolheadRelPos(preset.pos, `Test Position: ${preset.label}`);
                    }}
                  >
                    {preset.label}
                  </button>
                );
              })}
            </div>

            <div className="pos-slider-grid">
              {(
                [
                  [0, "X"],
                  [1, "Y"],
                  [2, "Z"],
                ] as Array<[0 | 1 | 2, string]>
              ).map(([idx, axisName]) => {
                const pct = Math.round(posDraft[idx] * 100);
                return (
                  <div key={axisName} className="pos-slider-row">
                    <span className="pos-axis-lbl">{axisName}</span>
                    <input aria-label={`Toolhead test position ${axisName} (%)`}
                      type="range"
                      min={10}
                      max={90}
                      step={5}
                      value={pct}
                      onChange={(e) => handlePosAxisSlider(idx, Number(e.target.value))}
                      onPointerUp={() => commitToolheadRelPos(posDraft)}
                      onKeyUp={() => commitToolheadRelPos(posDraft)}
                    />
                    <span className="pos-axis-val">{pct}%</span>
                  </div>
                );
              })}
            </div>

            {showHints && (
              <div className="hint-line">
                Probe corners or Top/Low Z to reveal cantilever bed droop, Z-column flex, or
                asymmetric gantry modes.
              </div>
            )}
          </div>

          <div className="field-row">
            <label id="df-panel-shell-mesh-mitc4-24">Panel Shell Mesh (MITC4)</label>
            <div role="group" aria-labelledby="df-panel-shell-mesh-mitc4-24"
              className="engine-toggle engine-toggle-4"
              style={{ gridTemplateColumns: "repeat(4, minmax(0, 1fr))" }}
            >
              {(
                [
                  [2, "2×2 Fast"],
                  [3, "3×3 Balanced"],
                  [4, "4×4 Med"],
                  [8, "8×8 Max"],
                ] as Array<[number, string]>
              ).map(([n, label]) => (
                <button
                  key={n}
                  type="button"
                  className={`engine-toggle-btn ${panelMeshN === n ? "active" : ""}`}
 aria-pressed={panelMeshN === n}
                  onClick={() => onPanelMeshNChange(n)}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
        </div>
      </details>
    </div>
  );
};
