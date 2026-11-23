import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  ARCH_LABELS,
  BELT_LABELS,
  BRACING_LABELS,
  Z_DRIVE_LABELS,
} from "../constants/printerLabels";
import {
  AxisShaperGraph,
  CatalogResponse,
  PrinterBOMResponse,
  PrinterSpecData,
  SimulationResponse,
} from "../types";

export interface BaselineCompareDrawerProps {
  open: boolean;
  onClose: () => void;
  baselineSim: SimulationResponse | null;
  baselineSpec: PrinterSpecData | null;
  baselineBom: PrinterBOMResponse | null;
  baselineLabel: string;
  currentSim: SimulationResponse | null;
  currentSpec: PrinterSpecData | null;
  currentBom: PrinterBOMResponse | null;
  catalog: CatalogResponse | null;
  onPinCurrent?: () => void;
}

interface CompareMetricRow {
  id: string;
  label: string;
  unit: string;
  baseVal: number;
  curVal: number;
  baseText: string;
  curText: string;
  deltaText: string;
  deltaCls: "better" | "worse" | "same";
}

interface SpecDiffRow {
  field: string;
  before: string;
  after: string;
}

function shortCatalogName(
  id: string | null | undefined,
  items?: Array<{ id: string; name: string }>
): string {
  if (!id) return "—";
  const found = items?.find((it) => it.id === id);
  if (!found) return id;
  return found.name.replace(/\s*\([^)]*\)\s*/g, "").trim() || found.name;
}

function formatPanelSummary(frame: PrinterSpecData["frame"]): string {
  const mounting = !frame.enclosed_panels ? "none" : frame.panel_mounting || "bolted_structural";
  if (!frame.enclosed_panels || mounting === "none") return "Open";
  const mat = frame.panel_material || "acrylic_pmma";
  const mnt = mounting === "clip_enclosure" ? "Clip-In" : "Bolted";
  return `${frame.panel_thickness_mm}mm ${mat.replace(/_/g, " ")} (${mnt})`;
}

function collectSpecDiffs(
  base: PrinterSpecData | null,
  cur: PrinterSpecData | null,
  catalog: CatalogResponse | null
): SpecDiffRow[] {
  if (!base || !cur) return [];
  const diffs: SpecDiffRow[] = [];
  const push = (field: string, before: string, after: string) => {
    if (before !== after) diffs.push({ field, before, after });
  };

  if (base.architecture !== cur.architecture) {
    push(
      "Architecture",
      ARCH_LABELS[base.architecture] ?? base.architecture,
      ARCH_LABELS[cur.architecture] ?? cur.architecture
    );
  }
  if (
    base.frame.width_mm !== cur.frame.width_mm ||
    base.frame.depth_mm !== cur.frame.depth_mm ||
    base.frame.height_mm !== cur.frame.height_mm
  ) {
    push(
      "Frame Size",
      `${base.frame.width_mm}×${base.frame.depth_mm}×${base.frame.height_mm} mm`,
      `${cur.frame.width_mm}×${cur.frame.depth_mm}×${cur.frame.height_mm} mm`
    );
  }
  if (base.frame.profile !== cur.frame.profile) {
    push(
      "Frame Profile",
      shortCatalogName(base.frame.profile, catalog?.profiles),
      shortCatalogName(cur.frame.profile, catalog?.profiles)
    );
  }
  if ((base.frame.upright_profile || null) !== (cur.frame.upright_profile || null)) {
    push(
      "Corner Uprights",
      base.frame.upright_profile
        ? shortCatalogName(base.frame.upright_profile, catalog?.profiles)
        : "Same as Frame",
      cur.frame.upright_profile
        ? shortCatalogName(cur.frame.upright_profile, catalog?.profiles)
        : "Same as Frame"
    );
  }
  if (base.frame.material !== cur.frame.material) {
    push(
      "Frame Material",
      shortCatalogName(base.frame.material, catalog?.materials),
      shortCatalogName(cur.frame.material, catalog?.materials)
    );
  }
  if (base.frame.bracing !== cur.frame.bracing) {
    push(
      "Bracing",
      BRACING_LABELS[base.frame.bracing] ?? base.frame.bracing,
      BRACING_LABELS[cur.frame.bracing] ?? cur.frame.bracing
    );
  }
  if ((base.frame.extrusion_filling ?? "none") !== (cur.frame.extrusion_filling ?? "none")) {
    push(
      "Extrusion Fill",
      base.frame.extrusion_filling ?? "none",
      cur.frame.extrusion_filling ?? "none"
    );
  }
  if ((base.frame.corner_joint_type ?? "blind_joint") !== (cur.frame.corner_joint_type ?? "blind_joint")) {
    push(
      "Corner Joints",
      base.frame.corner_joint_type ?? "blind_joint",
      cur.frame.corner_joint_type ?? "blind_joint"
    );
  }
  if ((base.frame.feet_type ?? "rigid_bolted") !== (cur.frame.feet_type ?? "rigid_bolted")) {
    push(
      "Benchtop Feet",
      base.frame.feet_type ?? "rigid_bolted",
      cur.frame.feet_type ?? "rigid_bolted"
    );
  }
  if (
    base.frame.enclosed_panels !== cur.frame.enclosed_panels ||
    base.frame.panel_mounting !== cur.frame.panel_mounting ||
    base.frame.panel_material !== cur.frame.panel_material ||
    base.frame.panel_thickness_mm !== cur.frame.panel_thickness_mm
  ) {
    push("Panels", formatPanelSummary(base.frame), formatPanelSummary(cur.frame));
  }
  if (Boolean(base.frame.bolt_panels_to_gantry ?? true) !== Boolean(cur.frame.bolt_panels_to_gantry ?? true)) {
    push(
      "Panel Bolts",
      (base.frame.bolt_panels_to_gantry ?? true) ? "Bolted to Gantry" : "Outer Frame Only",
      (cur.frame.bolt_panels_to_gantry ?? true) ? "Bolted to Gantry" : "Outer Frame Only"
    );
  }
  if (Boolean(base.frame.trident_backbone) !== Boolean(cur.frame.trident_backbone)) {
    push(
      "Trident Backbone",
      base.frame.trident_backbone ? "3mm Steel" : "Off",
      cur.frame.trident_backbone ? "3mm Steel" : "Off"
    );
  }
  if (Math.round(base.frame.chamber_temp_c ?? 25) !== Math.round(cur.frame.chamber_temp_c ?? 25)) {
    push(
      "Chamber Temp",
      `${Math.round(base.frame.chamber_temp_c ?? 25)}°C`,
      `${Math.round(cur.frame.chamber_temp_c ?? 25)}°C`
    );
  }
  if ((base.gantry.drive_mode ?? "2wd") !== (cur.gantry.drive_mode ?? "2wd")) {
    push(
      "Drive Mode",
      (base.gantry.drive_mode ?? "2wd").toUpperCase(),
      (cur.gantry.drive_mode ?? "2wd").toUpperCase()
    );
  }
  if (base.gantry.x_beam_profile !== cur.gantry.x_beam_profile) {
    push(
      "X Beam Profile",
      shortCatalogName(base.gantry.x_beam_profile, catalog?.profiles),
      shortCatalogName(cur.gantry.x_beam_profile, catalog?.profiles)
    );
  }
  if (base.gantry.x_beam_material !== cur.gantry.x_beam_material) {
    push(
      "X Beam Material",
      shortCatalogName(base.gantry.x_beam_material, catalog?.materials),
      shortCatalogName(cur.gantry.x_beam_material, catalog?.materials)
    );
  }
  if (base.gantry.x_rail_profile !== cur.gantry.x_rail_profile) {
    push("X Rail", base.gantry.x_rail_profile, cur.gantry.x_rail_profile);
  }
  if (base.gantry.y_rail_profile !== cur.gantry.y_rail_profile) {
    push("Y Rail", base.gantry.y_rail_profile, cur.gantry.y_rail_profile);
  }
  if ((base.gantry.x_rail_mount ?? "front") !== (cur.gantry.x_rail_mount ?? "front")) {
    push("X Rail Mount", base.gantry.x_rail_mount ?? "front", cur.gantry.x_rail_mount ?? "front");
  }
  if ((base.gantry.y_rail_mount ?? "top") !== (cur.gantry.y_rail_mount ?? "top")) {
    push("Y Rail Mount", base.gantry.y_rail_mount ?? "top", cur.gantry.y_rail_mount ?? "top");
  }
  if (base.gantry.belt.type !== cur.gantry.belt.type) {
    push(
      "Belt",
      BELT_LABELS[base.gantry.belt.type] ?? base.gantry.belt.type,
      BELT_LABELS[cur.gantry.belt.type] ?? cur.gantry.belt.type
    );
  }
  if (base.gantry.belt.core !== cur.gantry.belt.core) {
    push("Belt Core", base.gantry.belt.core, cur.gantry.belt.core);
  }
  if (base.gantry.belt.pretension_n !== cur.gantry.belt.pretension_n) {
    push("Belt Tension", `${base.gantry.belt.pretension_n} N`, `${cur.gantry.belt.pretension_n} N`);
  }
  if ((base.gantry.belt_routing ?? "stock_corexy") !== (cur.gantry.belt_routing ?? "stock_corexy")) {
    push("Belt Routing", base.gantry.belt_routing ?? "stock_corexy", cur.gantry.belt_routing ?? "stock_corexy");
  }
  if (Boolean(base.gantry.double_shear_pulleys) !== Boolean(cur.gantry.double_shear_pulleys)) {
    push(
      "Pulley Mounts",
      base.gantry.double_shear_pulleys ? "Double-Shear" : "Single-Shear",
      cur.gantry.double_shear_pulleys ? "Double-Shear" : "Single-Shear"
    );
  }
  if ((base.gantry.bracket_material ?? "abs_printed") !== (cur.gantry.bracket_material ?? "abs_printed")) {
    push(
      "Gantry Brackets",
      base.gantry.bracket_material ?? "abs_printed",
      cur.gantry.bracket_material ?? "abs_printed"
    );
  }
  const baseOv = base.gantry.custom_bracket_override ?? null;
  const curOv = cur.gantry.custom_bracket_override ?? null;
  if (
    (baseOv?.label ?? null) !== (curOv?.label ?? null) ||
    Math.round(baseOv?.stiffness_n_per_mm ?? -1) !== Math.round(curOv?.stiffness_n_per_mm ?? -1)
  ) {
    push(
      "Custom CAD Bracket",
      baseOv ? `${baseOv.label} (${Math.round(baseOv.stiffness_n_per_mm)} N/mm)` : "None",
      curOv ? `${curOv.label} (${Math.round(curOv.stiffness_n_per_mm)} N/mm)` : "None"
    );
  }
  if (Math.round(base.gantry.carriage_mass_g) !== Math.round(cur.gantry.carriage_mass_g)) {
    push(
      "XY-Joints Mass",
      `${Math.round(base.gantry.carriage_mass_g)} g`,
      `${Math.round(cur.gantry.carriage_mass_g)} g`
    );
  }
  if (base.kinematics.xy_stepper.name !== cur.kinematics.xy_stepper.name) {
    push(
      "XY Motor",
      base.kinematics.xy_stepper.name.split("(")[0].trim(),
      cur.kinematics.xy_stepper.name.split("(")[0].trim()
    );
  }
  if (base.kinematics.xy_stepper.supply_voltage_v !== cur.kinematics.xy_stepper.supply_voltage_v) {
    push(
      "Bus Voltage",
      `${base.kinematics.xy_stepper.supply_voltage_v} V`,
      `${cur.kinematics.xy_stepper.supply_voltage_v} V`
    );
  }
  const baseA = base.kinematics.xy_stepper.run_current_a ?? Math.min(base.kinematics.xy_stepper.rated_current_a, 1.2);
  const curA = cur.kinematics.xy_stepper.run_current_a ?? Math.min(cur.kinematics.xy_stepper.rated_current_a, 1.2);
  if (Math.abs(baseA - curA) >= 0.05) {
    push("Motor Current", `${baseA.toFixed(1)} A`, `${curA.toFixed(1)} A`);
  }
  if (base.kinematics.max_accel_mm_s2 !== cur.kinematics.max_accel_mm_s2) {
    push(
      "Target Accel",
      `${base.kinematics.max_accel_mm_s2.toLocaleString()} mm/s²`,
      `${cur.kinematics.max_accel_mm_s2.toLocaleString()} mm/s²`
    );
  }
  if (base.kinematics.input_shaper !== cur.kinematics.input_shaper) {
    push("Input Shaper", base.kinematics.input_shaper.toUpperCase(), cur.kinematics.input_shaper.toUpperCase());
  }
  if (base.toolhead.total_mass_g !== cur.toolhead.total_mass_g) {
    push("Toolhead Mass", `${base.toolhead.total_mass_g} g`, `${cur.toolhead.total_mass_g} g`);
  }
  if ((base.toolhead.mount_material ?? "abs_printed") !== (cur.toolhead.mount_material ?? "abs_printed")) {
    push(
      "Toolhead Mount",
      base.toolhead.mount_material ?? "abs_printed",
      cur.toolhead.mount_material ?? "abs_printed"
    );
  }
  if ((base.toolhead.z_probe_type ?? "static_eddy") !== (cur.toolhead.z_probe_type ?? "static_eddy")) {
    push(
      "Z-Probe",
      base.toolhead.z_probe_type ?? "static_eddy",
      cur.toolhead.z_probe_type ?? "static_eddy"
    );
  }
  if (base.toolhead.hotend.name !== cur.toolhead.hotend.name) {
    push("Hotend", base.toolhead.hotend.name, cur.toolhead.hotend.name);
  }
  if (base.bed.z_drive !== cur.bed.z_drive) {
    push(
      "Z Architecture",
      Z_DRIVE_LABELS[base.bed.z_drive] ?? base.bed.z_drive,
      Z_DRIVE_LABELS[cur.bed.z_drive] ?? cur.bed.z_drive
    );
  }
  if ((base.bed.mount_type ?? "kinematic_3pt") !== (cur.bed.mount_type ?? "kinematic_3pt")) {
    push("Bed Mount", base.bed.mount_type ?? "kinematic_3pt", cur.bed.mount_type ?? "kinematic_3pt");
  }
  if (base.bed.plate_thickness_mm !== cur.bed.plate_thickness_mm) {
    push("Bed Plate Thickness", `${base.bed.plate_thickness_mm} mm`, `${cur.bed.plate_thickness_mm} mm`);
  }
  if (base.bed.plate_material !== cur.bed.plate_material) {
    push(
      "Bed Plate Material",
      shortCatalogName(base.bed.plate_material, catalog?.materials),
      shortCatalogName(cur.bed.plate_material, catalog?.materials)
    );
  }
  if (Math.round(base.bed.target_temp_c ?? 60) !== Math.round(cur.bed.target_temp_c ?? 60)) {
    push("Bed Temp", `${Math.round(base.bed.target_temp_c ?? 60)}°C`, `${Math.round(cur.bed.target_temp_c ?? 60)}°C`);
  }

  return diffs;
}

function getAxisGraph(sim: SimulationResponse | null, axis: "x" | "y"): AxisShaperGraph | null {
  if (!sim) return null;
  const fromGraphs = sim.modal_fea.shaper_graphs?.[axis];
  if (fromGraphs && Array.isArray(fromGraphs.freqs_hz) && fromGraphs.freqs_hz.length >= 2) {
    return fromGraphs;
  }
  const shaper = axis === "x" ? sim.modal_fea.shaper_x : sim.modal_fea.shaper_y;
  if (shaper?.klipper_graph && Array.isArray(shaper.klipper_graph.freqs_hz) && shaper.klipper_graph.freqs_hz.length >= 2) {
    return shaper.klipper_graph;
  }
  return null;
}

function buildRow(
  id: string,
  label: string,
  unit: string,
  baseVal: number,
  curVal: number,
  higherIsBetter: boolean,
  eps: number,
  formatVal: (v: number) => string,
  formatDelta: (d: number) => string
): CompareMetricRow {
  const delta = curVal - baseVal;
  let deltaCls: "better" | "worse" | "same" = "same";
  let deltaText = "0";
  if (Math.abs(delta) >= eps) {
    const better = higherIsBetter ? delta > 0 : delta < 0;
    deltaCls = better ? "better" : "worse";
    deltaText = formatDelta(delta);
  }
  return {
    id,
    label,
    unit,
    baseVal,
    curVal,
    baseText: formatVal(baseVal),
    curText: formatVal(curVal),
    deltaText,
    deltaCls,
  };
}

export const BaselineCompareDrawer: React.FC<BaselineCompareDrawerProps> = ({
  open,
  onClose,
  baselineSim,
  baselineSpec,
  baselineBom,
  baselineLabel,
  currentSim,
  currentSpec,
  currentBom,
  catalog,
  onPinCurrent,
}) => {
  const [psdAxis, setPsdAxis] = useState<"x" | "y">("x");
  const [copiedMd, setCopiedMd] = useState<boolean>(false);
  const copyTimerRef = useRef<number | null>(null);

  useEffect(() => {
    if (!open) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open, onClose]);

  useEffect(() => {
    return () => {
      if (copyTimerRef.current != null) window.clearTimeout(copyTimerRef.current);
    };
  }, []);

  const metrics = useMemo<CompareMetricRow[]>(() => {
    if (!baselineSim || !currentSim) return [];

    const baseXHz = baselineSim.modal_fea.shaper_x?.effective_freq_hz ?? baselineSim.modal_fea.fundamental_freq_hz;
    const curXHz = currentSim.modal_fea.shaper_x?.effective_freq_hz ?? currentSim.modal_fea.fundamental_freq_hz;
    const baseYHz = baselineSim.modal_fea.shaper_y?.effective_freq_hz ?? baselineSim.modal_fea.fundamental_freq_hz;
    const curYHz = currentSim.modal_fea.shaper_y?.effective_freq_hz ?? currentSim.modal_fea.fundamental_freq_hz;

    const baseCost =
      baselineBom?.total_cost_usd ?? baselineSim.geometry.mass_summary.estimated_material_cost_usd;
    const curCost =
      currentBom?.total_cost_usd ?? currentSim.geometry.mass_summary.estimated_material_cost_usd;

    const baseLimHz = Math.max(1, Math.min(baseXHz, baseYHz));
    const curLimHz = Math.max(1, Math.min(curXHz, curYHz));
    const baseCostPerHz = baseCost / baseLimHz;
    const curCostPerHz = curCost / curLimHz;

    const sign = (d: number) => (d > 0 ? "+" : "");

    return [
      buildRow(
        "composite",
        "Composite Score",
        "/ 100",
        baselineSim.scores.composite,
        currentSim.scores.composite,
        true,
        0.1,
        (v) => v.toFixed(1),
        (d) => `${sign(d)}${d.toFixed(1)}`
      ),
      buildRow(
        "x_shaper_hz",
        "X Shaper Hz",
        "Hz",
        baseXHz,
        curXHz,
        true,
        0.1,
        (v) => `${v.toFixed(1)} Hz`,
        (d) => `${sign(d)}${d.toFixed(1)} Hz`
      ),
      buildRow(
        "y_shaper_hz",
        "Y Shaper Hz",
        "Hz",
        baseYHz,
        curYHz,
        true,
        0.1,
        (v) => `${v.toFixed(1)} Hz`,
        (d) => `${sign(d)}${d.toFixed(1)} Hz`
      ),
      buildRow(
        "safe_accel",
        "Safe Max Accel",
        "mm/s²",
        baselineSim.modal_fea.recommended_max_accel_mm_s2,
        currentSim.modal_fea.recommended_max_accel_mm_s2,
        true,
        25,
        (v) => `${Math.round(v).toLocaleString()} mm/s²`,
        (d) => `${sign(d)}${Math.round(d).toLocaleString()}`
      ),
      buildRow(
        "nozzle_flex",
        "Nozzle Flex",
        "µm",
        baselineSim.static_fea.combined_max_deflection_um,
        currentSim.static_fea.combined_max_deflection_um,
        false,
        0.1,
        (v) => `${v.toFixed(1)} µm`,
        (d) => `${sign(d)}${d.toFixed(1)} µm`
      ),
      buildRow(
        "x_moving_mass",
        "Moving X Mass",
        "g",
        baselineSim.geometry.mass_summary.x_moving_mass_g,
        currentSim.geometry.mass_summary.x_moving_mass_g,
        false,
        1,
        (v) => `${Math.round(v)} g`,
        (d) => `${sign(d)}${Math.round(d)} g`
      ),
      buildRow(
        "y_moving_mass",
        "Moving Y Mass",
        "g",
        baselineSim.geometry.mass_summary.y_moving_mass_g,
        currentSim.geometry.mass_summary.y_moving_mass_g,
        false,
        1,
        (v) => `${Math.round(v)} g`,
        (d) => `${sign(d)}${Math.round(d)} g`
      ),
      buildRow(
        "total_mass",
        "Total Printer Mass",
        "kg",
        baselineSim.geometry.mass_summary.total_mass_kg,
        currentSim.geometry.mass_summary.total_mass_kg,
        false,
        0.05,
        (v) => `${v.toFixed(2)} kg`,
        (d) => `${sign(d)}${d.toFixed(2)} kg`
      ),
      buildRow(
        "bom_cost",
        "BOM Cost",
        "$",
        baseCost,
        curCost,
        false,
        1,
        (v) => `$${Math.round(v).toLocaleString()}`,
        (d) => `${d > 0 ? "+$" : "-$"}${Math.abs(Math.round(d)).toLocaleString()}`
      ),
      buildRow(
        "cost_per_hz",
        "Cost per Hz (Limiting Axis)",
        "$/Hz",
        baseCostPerHz,
        curCostPerHz,
        false,
        0.05,
        (v) => `$${v.toFixed(2)}/Hz`,
        (d) => `${d > 0 ? "+$" : "-$"}${Math.abs(d).toFixed(2)}/Hz`
      ),
    ];
  }, [baselineBom, baselineSim, currentBom, currentSim]);

  const specDiffs = useMemo(
    () => collectSpecDiffs(baselineSpec, currentSpec, catalog),
    [baselineSpec, currentSpec, catalog]
  );

  const baseGraph = useMemo(() => getAxisGraph(baselineSim, psdAxis), [baselineSim, psdAxis]);
  const curGraph = useMemo(() => getAxisGraph(currentSim, psdAxis), [currentSim, psdAxis]);

  const psdSvg = useMemo(() => {
    if (!baseGraph && !curGraph) return null;
    const W = 520;
    const H = 175;
    const padL = 24;
    const padR = 16;
    const padT = 22;
    const padB = 24;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    const allFreqs = [
      ...(baseGraph?.freqs_hz ?? []),
      ...(curGraph?.freqs_hz ?? []),
    ];
    const fMin = Math.min(5, ...(allFreqs.length ? [Math.min(...allFreqs)] : [5]));
    const fMax = Math.max(200, ...(allFreqs.length ? [Math.max(...allFreqs)] : [200]));
    const fSpan = Math.max(fMax - fMin, 1);

    const baseMax = Math.max(...(baseGraph?.raw_psd ?? [0]), 1e-9);
    const curMax = Math.max(...(curGraph?.raw_psd ?? [0]), 1e-9);
    const sharedMax = Math.max(baseMax, curMax, 1e-9);

    const xOf = (f: number) => padL + ((f - fMin) / fSpan) * plotW;
    const yOf = (v: number) => padT + plotH - (Math.max(0, v) / sharedMax) * plotH;

    const buildPath = (g: AxisShaperGraph | null): string => {
      if (!g) return "";
      const n = Math.min(g.freqs_hz.length, g.raw_psd.length);
      if (n < 2) return "";
      const pts: string[] = [];
      for (let i = 0; i < n; i++) {
        pts.push(`${i === 0 ? "M" : "L"}${xOf(g.freqs_hz[i]).toFixed(1)},${yOf(g.raw_psd[i]).toFixed(1)}`);
      }
      return pts.join(" ");
    };

    const basePath = buildPath(baseGraph);
    const curPath = buildPath(curGraph);
    let curArea = "";
    if (curGraph && curPath) {
      const n = Math.min(curGraph.freqs_hz.length, curGraph.raw_psd.length);
      curArea = `${curPath} L${xOf(curGraph.freqs_hz[n - 1]).toFixed(1)},${(padT + plotH).toFixed(1)} L${xOf(curGraph.freqs_hz[0]).toFixed(1)},${(padT + plotH).toFixed(1)} Z`;
    }

    const ticks = [20, 60, 100, 140, 180, 220].filter((t) => t >= fMin && t <= fMax);

    return {
      W,
      H,
      padL,
      padR,
      padT,
      padB,
      plotW,
      plotH,
      basePath,
      curPath,
      curArea,
      ticks,
      xOf,
      basePeakHz: baseGraph?.peak_freq_hz ?? null,
      curPeakHz: curGraph?.peak_freq_hz ?? null,
    };
  }, [baseGraph, curGraph]);

  const markdownReport = useMemo(() => {
    const curName = currentSpec?.name || currentSim?.printer_name || "Current Design";
    const lines: string[] = [
      `## PrintSim Trade Study: ${baselineLabel} vs. ${curName}`,
      ``,
      `| Metric | Baseline | Current | Delta |`,
      `| :--- | ---: | ---: | ---: |`,
      ...metrics.map((m) => `| ${m.label} | ${m.baseText} | ${m.curText} | **${m.deltaText}** |`),
      ``,
      `### Spec Changes`,
    ];
    if (specDiffs.length === 0) {
      lines.push(`- _No parameter changes from pinned baseline._`);
    } else {
      for (const d of specDiffs) {
        lines.push(`- **${d.field}**: ${d.before} → ${d.after}`);
      }
    }
    return lines.join("\n");
  }, [baselineLabel, currentSim, currentSpec, metrics, specDiffs]);

  const handleCopyMarkdown = async () => {
    let copied = false;
    try {
      if (navigator.clipboard && typeof navigator.clipboard.writeText === "function") {
        await navigator.clipboard.writeText(markdownReport);
        copied = true;
      }
    } catch {
      copied = false;
    }
    if (!copied) {
      try {
        const ta = document.createElement("textarea");
        ta.value = markdownReport;
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.focus();
        ta.select();
        document.execCommand("copy");
        document.body.removeChild(ta);
        copied = true;
      } catch {
        copied = false;
      }
    }
    if (copied) {
      setCopiedMd(true);
      if (copyTimerRef.current != null) window.clearTimeout(copyTimerRef.current);
      copyTimerRef.current = window.setTimeout(() => {
        setCopiedMd(false);
        copyTimerRef.current = null;
      }, 2200);
    }
  };

  if (!open || !baselineSim || !currentSim) return null;

  const curName = currentSpec?.name || currentSim.printer_name;
  const xBaseHz = baselineSim.modal_fea.shaper_x?.effective_freq_hz ?? baselineSim.modal_fea.fundamental_freq_hz;
  const xCurHz = currentSim.modal_fea.shaper_x?.effective_freq_hz ?? currentSim.modal_fea.fundamental_freq_hz;
  const yBaseHz = baselineSim.modal_fea.shaper_y?.effective_freq_hz ?? baselineSim.modal_fea.fundamental_freq_hz;
  const yCurHz = currentSim.modal_fea.shaper_y?.effective_freq_hz ?? currentSim.modal_fea.fundamental_freq_hz;

  return (
    <div
      className="compare-drawer-backdrop"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="compare-drawer" role="dialog" aria-modal="true" aria-label="Baseline Trade-Study Comparison">
        <div className="compare-drawer-header">
          <div>
            <div className="compare-drawer-title">Trade-Study Compare</div>
            <div className="compare-drawer-sub">
              <span className="compare-chip-base">{baselineLabel}</span>
              <span style={{ color: "var(--muted)", margin: "0 6px" }}>vs.</span>
              <span className="compare-chip-cur">{curName}</span>
            </div>
          </div>
          <div className="compare-drawer-header-actions">
            {onPinCurrent && (
              <button
                type="button"
                className="btn"
                onClick={onPinCurrent}
                title="Pin the current design as the new comparison baseline"
              >
                Pin Current
              </button>
            )}
            <button
              type="button"
              className={`btn ${copiedMd ? "share-copied" : "primary"}`}
              onClick={handleCopyMarkdown}
            >
              {copiedMd ? "Copied ✓" : "Copy Markdown Trade Study"}
            </button>
            <button
              type="button"
              className="btn"
              onClick={onClose}
              aria-label="Close comparison drawer"
            >
              ✕
            </button>
          </div>
        </div>

        <div className="compare-drawer-body">
          {/* Side-by-side KPI table */}
          <div className="compare-section">
            <div className="section-title">
              <span>Performance &amp; Cost Comparison</span>
              <span className="sub-block-tag">{metrics.length} metrics</span>
            </div>
            <div className="compare-table-wrap">
              <table className="compare-table">
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th>Baseline</th>
                    <th>Current</th>
                    <th>Delta</th>
                  </tr>
                </thead>
                <tbody>
                  {metrics.map((row) => (
                    <tr key={row.id}>
                      <td className="compare-metric-name">{row.label}</td>
                      <td className="compare-num muted">{row.baseText}</td>
                      <td className="compare-num strong">{row.curText}</td>
                      <td className={`compare-num compare-delta ${row.deltaCls}`}>
                        {row.deltaText}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Dual-PSD Overlay SVG */}
          <div className="compare-section">
            <div className="section-title">
              <span>Dual-PSD Resonance Overlay</span>
              <div className="compare-psd-tabs">
                <button
                  type="button"
                  className={`psd-axis-btn ${psdAxis === "x" ? "active" : ""}`}
                  onClick={() => setPsdAxis("x")}
                >
                  X Axis{" "}
                  <span className="psd-axis-peak">
                    {xBaseHz.toFixed(0)}→{xCurHz.toFixed(0)} Hz
                  </span>
                </button>
                <button
                  type="button"
                  className={`psd-axis-btn ${psdAxis === "y" ? "active" : ""}`}
                  onClick={() => setPsdAxis("y")}
                >
                  Y Axis{" "}
                  <span className="psd-axis-peak">
                    {yBaseHz.toFixed(0)}→{yCurHz.toFixed(0)} Hz
                  </span>
                </button>
              </div>
            </div>

            {psdSvg ? (
              <div className="compare-psd-card">
                <div className="compare-psd-legend">
                  <span className="compare-legend-item">
                    <span className="compare-swatch-dashed" />
                    Baseline ({baseGraph?.recommended_shaper ?? "—"} ·{" "}
                    {baseGraph ? `${baseGraph.peak_freq_hz.toFixed(1)} Hz` : "—"})
                  </span>
                  <span className="compare-legend-item">
                    <span className="compare-swatch-solid" />
                    Current ({curGraph?.recommended_shaper ?? "—"} ·{" "}
                    {curGraph ? `${curGraph.peak_freq_hz.toFixed(1)} Hz` : "—"})
                  </span>
                  {psdSvg.basePeakHz != null && psdSvg.curPeakHz != null && (
                    <span
                      className={`compare-delta-pill ${
                        Math.abs(psdSvg.curPeakHz - psdSvg.basePeakHz) < 0.1
                          ? "same"
                          : psdSvg.curPeakHz > psdSvg.basePeakHz
                            ? "better"
                            : "worse"
                      }`}
                    >
                      {psdSvg.curPeakHz - psdSvg.basePeakHz > 0 ? "+" : ""}
                      {(psdSvg.curPeakHz - psdSvg.basePeakHz).toFixed(1)} Hz peak
                    </span>
                  )}
                </div>

                <svg
                  viewBox={`0 0 ${psdSvg.W} ${psdSvg.H}`}
                  className="compare-psd-svg"
                  role="img"
                  aria-label={`Dual PSD overlay for ${psdAxis.toUpperCase()} axis`}
                >
                  <defs>
                    <linearGradient id="curPsdFillGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#22d3ee" stopOpacity="0.28" />
                      <stop offset="100%" stopColor="#22d3ee" stopOpacity="0.02" />
                    </linearGradient>
                  </defs>

                  {/* Horizontal grid */}
                  {[0.25, 0.5, 0.75, 1.0].map((frac) => {
                    const y = psdSvg.padT + psdSvg.plotH * (1 - frac);
                    return (
                      <line
                        key={frac}
                        x1={psdSvg.padL}
                        x2={psdSvg.W - psdSvg.padR}
                        y1={y}
                        y2={y}
                        stroke="rgba(148, 163, 184, 0.14)"
                        strokeWidth={1}
                      />
                    );
                  })}

                  {/* Vertical frequency grid + labels */}
                  {psdSvg.ticks.map((t) => {
                    const x = psdSvg.xOf(t);
                    return (
                      <g key={t}>
                        <line
                          x1={x}
                          x2={x}
                          y1={psdSvg.padT}
                          y2={psdSvg.padT + psdSvg.plotH}
                          stroke="rgba(148, 163, 184, 0.12)"
                          strokeWidth={1}
                        />
                        <text
                          x={x}
                          y={psdSvg.H - 6}
                          textAnchor="middle"
                          fontSize={10}
                          fill="#8b9bb4"
                        >
                          {t} Hz
                        </text>
                      </g>
                    );
                  })}

                  {/* Baseline axis bottom line */}
                  <line
                    x1={psdSvg.padL}
                    x2={psdSvg.W - psdSvg.padR}
                    y1={psdSvg.padT + psdSvg.plotH}
                    y2={psdSvg.padT + psdSvg.plotH}
                    stroke="rgba(148, 163, 184, 0.35)"
                    strokeWidth={1}
                  />

                  {/* Current area fill */}
                  {psdSvg.curArea && <path d={psdSvg.curArea} fill="url(#curPsdFillGrad)" />}

                  {/* Baseline PSD curve (dashed amber) */}
                  {psdSvg.basePath && (
                    <path
                      d={psdSvg.basePath}
                      fill="none"
                      stroke="#f59e0b"
                      strokeWidth={1.9}
                      strokeDasharray="5 3"
                      opacity={0.9}
                    />
                  )}

                  {/* Current PSD curve (solid cyan/emerald) */}
                  {psdSvg.curPath && (
                    <path
                      d={psdSvg.curPath}
                      fill="none"
                      stroke="#22d3ee"
                      strokeWidth={2.2}
                    />
                  )}

                  {/* Peak markers */}
                  {psdSvg.basePeakHz != null && (
                    <line
                      x1={psdSvg.xOf(psdSvg.basePeakHz)}
                      x2={psdSvg.xOf(psdSvg.basePeakHz)}
                      y1={psdSvg.padT}
                      y2={psdSvg.padT + psdSvg.plotH}
                      stroke="#f59e0b"
                      strokeWidth={1}
                      strokeDasharray="3 3"
                      opacity={0.7}
                    />
                  )}
                  {psdSvg.curPeakHz != null && (
                    <line
                      x1={psdSvg.xOf(psdSvg.curPeakHz)}
                      x2={psdSvg.xOf(psdSvg.curPeakHz)}
                      y1={psdSvg.padT}
                      y2={psdSvg.padT + psdSvg.plotH}
                      stroke="#34d399"
                      strokeWidth={1.25}
                      opacity={0.85}
                    />
                  )}
                </svg>
              </div>
            ) : (
              <div className="hint-line">
                No Klipper PSD curves available for this comparison.
              </div>
            )}
          </div>

          {/* Exact Spec Diffs */}
          <div className="compare-section">
            <div className="section-title">
              <span>Spec Differences</span>
              <span className="sub-block-tag">
                {specDiffs.length === 0 ? "Identical" : `${specDiffs.length} changed`}
              </span>
            </div>
            {specDiffs.length === 0 ? (
              <div className="compare-empty-diff">
                Current specification matches the pinned baseline. Edit any printer parameter in the
                left panel to see live trade-study deltas.
              </div>
            ) : (
              <div className="compare-diff-list">
                {specDiffs.map((d) => (
                  <div key={d.field} className="compare-diff-row">
                    <span className="compare-diff-field">{d.field}</span>
                    <span className="compare-diff-vals">
                      <span className="compare-diff-before">{d.before}</span>
                      <span className="compare-diff-arrow">→</span>
                      <span className="compare-diff-after">{d.after}</span>
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
