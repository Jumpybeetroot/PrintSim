import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  AdxlNozzleAxisDiagnostic,
  AxisShaperGraph,
  GeometryExtrusionFillingData,
  KlipperShaperCandidate,
  MotionLimitsData,
  PrinterSpecData,
  StepResponseData,
} from "../types";

export interface KlipperShaperPsdCardProps {
  graphs: { x: AxisShaperGraph; y: AxisShaperGraph };
  adxlVsNozzle?: { x: AdxlNozzleAxisDiagnostic; y: AdxlNozzleAxisDiagnostic };
  frameDampingRatio?: number;
  extrusionFilling?: GeometryExtrusionFillingData;
  spec?: PrinterSpecData;
  motionLimits?: MotionLimitsData;
  changed: boolean;
  embedded?: boolean;
}

type PsdAxisMode = "x" | "y" | "both";
type ShaperChartMode = "psd" | "step";

interface GhostSnapshot {
  freqs_hz: number[];
  raw_psd: number[];
  peak_freq_hz: number;
}

interface VfaMarker {
  id: "belt" | "elec" | "step";
  label: string;
  freqHz: number;
  color: string;
}

const DEFAULT_SHAPER_DAMPING = 0.1;
const EI_VIB_TOL = 0.05;

function resolveBeltPitchMm(spec?: PrinterSpecData): number {
  const explicit = spec?.gantry.belt.pitch_mm;
  if (explicit && Number.isFinite(explicit) && explicit > 0) return explicit;
  const bType = (spec?.gantry.belt.type ?? "gt2_6mm").toLowerCase();
  if (bType.startsWith("at5")) return 5.0;
  if (bType.startsWith("3gt") || bType.startsWith("at3") || bType.startsWith("htd3m")) {
    return 3.0;
  }
  return 2.0;
}

function getKlipperShaperPulses(
  name: string,
  shaperFreqHz: number,
  dampingRatio: number = DEFAULT_SHAPER_DAMPING
): { amps: number[]; timesSec: number[] } | null {
  if (!Number.isFinite(shaperFreqHz) || shaperFreqHz <= 0) return null;
  const d = Math.max(0.005, Math.min(0.6, dampingRatio));
  const df = Math.sqrt(1 - d * d);
  const tD = 1 / (shaperFreqHz * df);
  const k = Math.exp((-d * Math.PI) / df);
  const key = name.trim().toUpperCase();

  let rawAmps: number[];
  let timesSec: number[];

  if (key === "NONE") {
    rawAmps = [1];
    timesSec = [0];
  } else if (key === "ZV" || key === "SMOOTH_ZV") {
    rawAmps = [1, k];
    timesSec = [0, 0.5 * tD];
  } else if (key === "ZVD") {
    rawAmps = [1, 2 * k, k * k];
    timesSec = [0, 0.5 * tD, tD];
  } else if (key === "MZV") {
    const km = Math.exp((-0.75 * d * Math.PI) / df);
    const a1 = 1 - 1 / Math.SQRT2;
    const a2 = (Math.SQRT2 - 1) * km;
    const a3 = a1 * km * km;
    rawAmps = [a1, a2, a3];
    timesSec = [0, 0.375 * tD, 0.75 * tD];
  } else if (key === "EI" || key === "SMOOTH_EI") {
    const a1 = 0.25 * (1 + EI_VIB_TOL);
    const a2 = 0.5 * (1 - EI_VIB_TOL) * k;
    const a3 = a1 * k * k;
    rawAmps = [a1, a2, a3];
    timesSec = [0, 0.5 * tD, tD];
  } else if (key === "2HUMP_EI") {
    const v2 = EI_VIB_TOL * EI_VIB_TOL;
    const x = Math.cbrt(v2 * (Math.sqrt(1 - v2) + 1));
    const a1 = (3 * x * x + 2 * x + 3 * v2) / (16 * x);
    const a2 = (0.5 - a1) * k;
    const a3 = a2 * k;
    const a4 = a1 * k * k * k;
    rawAmps = [a1, a2, a3, a4];
    timesSec = [0, 0.5 * tD, tD, 1.5 * tD];
  } else if (key === "3HUMP_EI") {
    const k2 = k * k;
    const a1 =
      0.0625 * (1 + 3 * EI_VIB_TOL + 2 * Math.sqrt(2 * (EI_VIB_TOL + 1) * EI_VIB_TOL));
    const a2 = 0.25 * (1 - EI_VIB_TOL) * k;
    const a3 = (0.5 * (1 + EI_VIB_TOL) - 2 * a1) * k2;
    const a4 = a2 * k2;
    const a5 = a1 * k2 * k2;
    rawAmps = [a1, a2, a3, a4, a5];
    timesSec = [0, 0.5 * tD, tD, 1.5 * tD, 2.0 * tD];
  } else if (key === "ZVDD") {
    rawAmps = [1, 3 * k, 3 * k * k, k * k * k];
    timesSec = [0, 0.5 * tD, tD, 1.5 * tD];
  } else {
    return null;
  }

  const sumA = rawAmps.reduce((acc, v) => acc + v, 0);
  if (sumA <= 1e-12) return null;
  return {
    amps: rawAmps.map((a) => a / sumA),
    timesSec,
  };
}

function evalUnitStepSecondOrder(tSec: number, freqHz: number, zeta: number): number {
  if (tSec <= 0) return 0;
  const wn = 2 * Math.PI * freqHz;
  const z = Math.max(0.01, Math.min(0.85, zeta));
  const sqrtOneMinusZ2 = Math.sqrt(1 - z * z);
  const wd = wn * sqrtOneMinusZ2;
  const phi = Math.atan2(sqrtOneMinusZ2, z);
  return 1 - (Math.exp(-z * wn * tSec) / sqrtOneMinusZ2) * Math.sin(wd * tSec + phi);
}

function evalShapedStepSeries(
  timesMs: number[],
  plantFreqHz: number,
  plantZeta: number,
  pulses: { amps: number[]; timesSec: number[] }
): number[] {
  return timesMs.map((tMs) => {
    const tSec = tMs * 0.001;
    let y = 0;
    for (let i = 0; i < pulses.amps.length; i++) {
      const dt = tSec - pulses.timesSec[i];
      if (dt > 0) {
        y += pulses.amps[i] * evalUnitStepSecondOrder(dt, plantFreqHz, plantZeta);
      }
    }
    return y;
  });
}

function computeStepMetrics(timesMs: number[], vals: number[]): {
  overshootPct: number;
  settleTimeMs: number;
} {
  let maxVal = 0;
  let lastOutsideIdx = -1;
  for (let i = 0; i < vals.length; i++) {
    if (vals[i] > maxVal) maxVal = vals[i];
    if (Math.abs(vals[i] - 1) > 0.02) {
      lastOutsideIdx = i;
    }
  }
  const overshootPct = Math.max(0, (maxVal - 1) * 100);
  const settleIdx = Math.min(vals.length - 1, Math.max(0, lastOutsideIdx + 1));
  const settleTimeMs = timesMs[settleIdx] ?? timesMs[timesMs.length - 1] ?? 0;
  return { overshootPct, settleTimeMs };
}

interface StepResponseComputed {
  W: number;
  H: number;
  padL: number;
  padR: number;
  padT: number;
  padB: number;
  plotW: number;
  plotH: number;
  tMaxMs: number;
  yMax: number;
  toX: (tMs: number) => number;
  toY: (v: number) => number;
  gridTimesMs: number[];
  unshapedLine: string;
  shapedLine: string;
  bandArea: string;
  bandLowLine: string;
  bandHighLine: string;
  overshootUnshapedPct: number;
  overshootShapedPct: number;
  settleUnshapedMs: number;
  settleShapedMs: number;
  bandWorstOvershootPct: number | null;
  shaperLabel: string;
}

function computeStepSvgData(
  graph: AxisShaperGraph,
  activeCandidate: KlipperShaperCandidate | null,
  plantZeta: number,
  bandDeltaHz: number
): StepResponseComputed | null {
  const plantFreq = Math.max(8, graph.peak_freq_hz || graph.recommended_freq_hz || 45);
  const shaperName = activeCandidate?.name ?? graph.recommended_shaper ?? "MZV";
  const shaperFreq = activeCandidate?.freq_hz ?? graph.recommended_freq_hz ?? plantFreq;
  const pulses = getKlipperShaperPulses(shaperName, shaperFreq, DEFAULT_SHAPER_DAMPING);

  const backendStep: StepResponseData | undefined = graph.step_response;
  let timeMs: number[];
  let unshaped: number[];
  let shaped: number[];
  let overshootUnshapedPct: number;
  let overshootShapedPct: number;
  let settleUnshapedMs: number;
  let settleShapedMs: number;
  let bandLow: number[] | null = null;
  let bandHigh: number[] | null = null;
  let bandWorstOvershootPct: number | null = null;

  if (pulses) {
    const periodMs = 1000 / plantFreq;
    const tMaxMs = Math.max(60, Math.min(180, Math.ceil((periodMs * 5.2) / 10) * 10));
    const nPts = 180;
    timeMs = Array.from({ length: nPts }, (_, i) => (i / (nPts - 1)) * tMaxMs);
    unshaped = timeMs.map((t) => evalUnitStepSecondOrder(t * 0.001, plantFreq, plantZeta));
    shaped = evalShapedStepSeries(timeMs, plantFreq, plantZeta, pulses);
    const uM = computeStepMetrics(timeMs, unshaped);
    const sM = computeStepMetrics(timeMs, shaped);
    overshootUnshapedPct = uM.overshootPct;
    overshootShapedPct = sM.overshootPct;
    settleUnshapedMs = uM.settleTimeMs;
    settleShapedMs = sM.settleTimeMs;

    if (bandDeltaHz > 0) {
      const fLo = Math.max(8, plantFreq - bandDeltaHz);
      const fHi = plantFreq + bandDeltaHz;
      bandLow = evalShapedStepSeries(timeMs, fLo, plantZeta, pulses);
      bandHigh = evalShapedStepSeries(timeMs, fHi, plantZeta, pulses);
      const loM = computeStepMetrics(timeMs, bandLow);
      const hiM = computeStepMetrics(timeMs, bandHigh);
      bandWorstOvershootPct = Math.max(loM.overshootPct, hiM.overshootPct);
    }
  } else if (backendStep && backendStep.time_ms.length >= 2) {
    timeMs = backendStep.time_ms;
    unshaped = backendStep.unshaped;
    shaped = backendStep.shaped;
    overshootUnshapedPct = backendStep.overshoot_unshaped_pct;
    overshootShapedPct = backendStep.overshoot_shaped_pct;
    settleUnshapedMs = backendStep.settle_time_unshaped_ms;
    settleShapedMs = backendStep.settle_time_shaped_ms;
  } else {
    return null;
  }

  const n = Math.min(timeMs.length, unshaped.length, shaped.length);
  if (n < 2) return null;

  const W = 316;
  const H = 178;
  const padL = 26;
  const padR = 12;
  const padT = 18;
  const padB = 22;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;
  const tMaxMs = Math.max(timeMs[n - 1] || 100, 10);

  let rawMaxY = 1.25;
  for (let i = 0; i < n; i++) {
    if (unshaped[i] > rawMaxY) rawMaxY = unshaped[i];
    if (shaped[i] > rawMaxY) rawMaxY = shaped[i];
    if (bandLow && bandLow[i] > rawMaxY) rawMaxY = bandLow[i];
    if (bandHigh && bandHigh[i] > rawMaxY) rawMaxY = bandHigh[i];
  }
  const yMax = Math.max(1.5, Math.min(2.0, Math.ceil((rawMaxY + 0.08) * 10) / 10));

  const toX = (t: number) => padL + (Math.max(0, Math.min(tMaxMs, t)) / tMaxMs) * plotW;
  const toY = (v: number) => padT + (1 - Math.max(0, Math.min(yMax, v)) / yMax) * plotH;

  const unPts: string[] = [];
  const shPts: string[] = [];
  const loPts: string[] = [];
  const hiPts: string[] = [];
  const envTop: string[] = [];
  const envBot: string[] = [];

  for (let i = 0; i < n; i++) {
    const x = toX(timeMs[i]).toFixed(1);
    unPts.push(`${x},${toY(unshaped[i]).toFixed(1)}`);
    shPts.push(`${x},${toY(shaped[i]).toFixed(1)}`);
    if (bandLow && bandHigh) {
      const yLo = bandLow[i];
      const yHi = bandHigh[i];
      loPts.push(`${x},${toY(yLo).toFixed(1)}`);
      hiPts.push(`${x},${toY(yHi).toFixed(1)}`);
      envTop.push(`${x},${toY(Math.max(yLo, yHi)).toFixed(1)}`);
      envBot.push(`${x},${toY(Math.min(yLo, yHi)).toFixed(1)}`);
    }
  }

  const bandArea =
    envTop.length >= 2
      ? `M ${envTop.join(" L ")} L ${envBot.reverse().join(" L ")} Z`
      : "";

  const stepMs = tMaxMs <= 80 ? 20 : tMaxMs <= 140 ? 30 : 40;
  const gridTimesMs: number[] = [];
  for (let t = stepMs; t < tMaxMs - 4; t += stepMs) {
    gridTimesMs.push(t);
  }

  return {
    W,
    H,
    padL,
    padR,
    padT,
    padB,
    plotW,
    plotH,
    tMaxMs,
    yMax,
    toX,
    toY,
    gridTimesMs,
    unshapedLine: unPts.join(" "),
    shapedLine: shPts.join(" "),
    bandArea,
    bandLowLine: loPts.join(" "),
    bandHighLine: hiPts.join(" "),
    overshootUnshapedPct,
    overshootShapedPct,
    settleUnshapedMs,
    settleShapedMs,
    bandWorstOvershootPct,
    shaperLabel: `${shaperName} @ ${shaperFreq.toFixed(1)} Hz`,
  };
}

function buildKlipperPrinterCfg(
  graphs: { x: AxisShaperGraph; y: AxisShaperGraph },
  spec?: PrinterSpecData,
  motionLimits?: MotionLimitsData
): string {
  const name = spec?.name?.trim() || "Custom Printer";
  const arch = spec?.architecture ?? "corexy";
  const kinematics =
    arch === "cartesian_bedslinger"
      ? "cartesian"
      : arch === "delta"
        ? "delta"
        : "corexy";

  const configuredMaxVel = Math.round(
    motionLimits?.configured_max_velocity_mm_s ?? spec?.kinematics.max_velocity_mm_s ?? 400
  );
  const maxVel =
    motionLimits && motionLimits.motor_max_velocity_mm_s > 0
      ? Math.round(Math.min(configuredMaxVel, motionLimits.motor_max_velocity_mm_s))
      : configuredMaxVel;
  const shaperAccelMin = Math.min(
    graphs.x.recommended_max_accel_mm_s2 || Infinity,
    graphs.y.recommended_max_accel_mm_s2 || Infinity
  );
  const maxAccel =
    motionLimits && motionLimits.achievable_max_accel_mm_s2 > 0
      ? Math.round(motionLimits.achievable_max_accel_mm_s2)
      : Number.isFinite(shaperAccelMin) && shaperAccelMin > 0
        ? Math.round(shaperAccelMin)
        : Math.round(spec?.kinematics.max_accel_mm_s2 ?? 10000);
  const scv = (spec?.kinematics.square_corner_velocity_mm_s ?? 5.0).toFixed(1);

  let motionLimitComment: string | null = null;
  if (motionLimits) {
    const shaperA = Math.round(motionLimits.shaper_max_accel_mm_s2);
    const motorA = Math.round(motionLimits.motor_max_accel_mm_s2);
    const beltA = Math.round(motionLimits.belt_max_accel_mm_s2);
    const motorV = Math.round(motionLimits.motor_max_velocity_mm_s);
    if (motionLimits.accel_limiting_factor === "motor") {
      motionLimitComment = `# Accel limit: Motor Torque (${motorA} mm/s²) < Shaper (${shaperA} mm/s²), Belt (${beltA} mm/s²) | Motor Vmax: ${motorV} mm/s`;
    } else if (motionLimits.accel_limiting_factor === "belt") {
      motionLimitComment = `# Accel limit: Belt Tension (${beltA} mm/s²) < Shaper (${shaperA} mm/s²), Motor (${motorA} mm/s²) | Motor Vmax: ${motorV} mm/s`;
    } else {
      motionLimitComment = `# Accel limit: Input Shaper (${shaperA} mm/s²) < Motor (${motorA} mm/s²), Belt (${beltA} mm/s²) | Motor Vmax: ${motorV} mm/s`;
    }
  }

  const shaperX = (graphs.x.recommended_shaper || "mzv").toLowerCase();
  const freqX = graphs.x.recommended_freq_hz.toFixed(1);
  const shaperY = (graphs.y.recommended_shaper || "mzv").toLowerCase();
  const freqY = graphs.y.recommended_freq_hz.toFixed(1);

  const rawRunCurrent =
    spec?.kinematics.xy_stepper.run_current_a ??
    Math.min(spec?.kinematics.xy_stepper.rated_current_a ?? 1.2, 1.2);
  const runCurrentStr = Number(rawRunCurrent.toFixed(2)).toString();
  const supplyV = Math.round(spec?.kinematics.xy_stepper.supply_voltage_v ?? 24);

  const lines = [
    `# PrintSim Recommended Klipper Config — ${name}`,
    ...(motionLimitComment ? [motionLimitComment] : []),
    `[printer]`,
    `kinematics: ${kinematics}`,
    `max_velocity: ${maxVel}`,
    `max_accel: ${maxAccel}`,
    `square_corner_velocity: ${scv}`,
    ``,
    `[input_shaper]`,
    `shaper_type_x: ${shaperX}`,
    `shaper_freq_x: ${freqX}`,
    `shaper_type_y: ${shaperY}`,
    `shaper_freq_y: ${freqY}`,
    ``,
    `# Stepper XY run_current: ${runCurrentStr} A (supply ${supplyV} V)`,
  ];

  const kin = spec?.kinematics;
  const betaNotes: string[] = [];
  if (kin?.input_shaper === "smooth_flatness") {
    betaNotes.push(`#   shaper_type: smooth_flatness (Kalico C4 Beta)`);
  }
  if (kin?.lpv_workspace_scheduling) {
    betaNotes.push(`#   lpv_workspace_scheduling: True (3x3 bed smoother_freq(x, y) map)`);
  }
  if (kin?.awd_sync_mode && kin.awd_sync_mode !== "none") {
    if (kin.awd_sync_mode === "position_equalizer") {
      const bw = (kin.awd_equalizer_bw_hz ?? 2.0).toFixed(1);
      betaNotes.push(`#   awd_sync_mode: position_equalizer (equalizer_bw_hz: ${bw} Hz)`);
    } else {
      betaNotes.push(`#   awd_sync_mode: torque_mirror (Stage 1 FOC torque mirror)`);
    }
  }
  const hasNotch = kin?.notch_filter_hz != null && kin.notch_filter_hz > 0;
  if (hasNotch) {
    const nf = Math.round(kin!.notch_filter_hz!);
    const depthDb = (kin!.notch_depth_db ?? 12.0).toFixed(1);
    const qVal = (kin!.notch_q ?? 1.75).toFixed(2);
    betaNotes.push(`#   tmc4671_biquad_t_notch: ${nf} Hz (depth: ${depthDb} dB, Q: ${qVal})`);
  }
  if (kin?.pin_rotor_ff) {
    betaNotes.push(`#   pin_rotor_ff (0x65): True (belt-reaction cancellation)`);
  }
  if (kin?.toolhead_imu_active_damping) {
    betaNotes.push(`#   toolhead_imu_active_damping (ICM-42688-P -> 0x65): True (+0.07 zeta)`);
  }
  if (kin?.mode_space_torque_ff) {
    betaNotes.push(`#   mode_space_torque_ff: True (CoreXY direction-split torque FF)`);
  }

  if (betaNotes.length > 0) {
    lines.push(``, `# Kalico / FOC Beta Features:`, ...betaNotes);
  }

  return lines.join("\n");
}

function findRecommendedCandidate(graph: AxisShaperGraph): KlipperShaperCandidate | null {
  const list = graph.candidates || [];
  if (!list.length) return null;
  return list.find((c) => c.name === graph.recommended_shaper) ?? list[0];
}

interface PsdSvgComputed {
  W: number;
  H: number;
  padL: number;
  padR: number;
  padT: number;
  padB: number;
  plotW: number;
  plotH: number;
  toX: (f: number) => number;
  toAttY: (v: number) => number;
  rawLine: string;
  rawArea: string;
  shapedLine: string;
  shapedArea: string;
  attLine: string;
  ghostLine: string;
  carriageLine: string;
  vfaMarkers: Array<{
    id: string;
    label: string;
    freqHz: number;
    x: number;
    color: string;
    labelY: number;
    anchor: "start" | "end";
  }>;
  gridFreqs: number[];
  markerFreq: number;
  markerX: number;
  markerLabelRight: boolean;
  peakFreq: number;
  peakX: number;
  peakY: number;
  showDistinctPeak: boolean;
}

interface PsdSvgExtras {
  ghost?: GhostSnapshot | null;
  carriageRawPsd?: number[] | null;
  vfaMarkers?: VfaMarker[];
}

function computePsdSvgData(
  graph: AxisShaperGraph,
  activeCandidate: KlipperShaperCandidate | null,
  height: number,
  extras?: PsdSvgExtras
): PsdSvgComputed | null {
  const freqs = graph.freqs_hz || [];
  const raw = graph.raw_psd || [];
  const recShaped = graph.shaped_psd || [];
  const n = Math.min(freqs.length, raw.length);
  if (n < 2) return null;

  const W = 316;
  const H = height;
  const padL = 12;
  const padR = 26;
  const padT = 18;
  const padB = height >= 160 ? 22 : 19;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const fMin = Math.min(5, freqs[0] ?? 5);
  const fMax = Math.max(220, freqs[n - 1] ?? 220);
  const fSpan = Math.max(fMax - fMin, 1);

  let peakIdx = 0;
  for (let i = 1; i < n; i++) {
    if (raw[i] > raw[peakIdx]) peakIdx = i;
  }

  // Match backend PSD attenuation scaling (V vs V^2) when previewing an alternate shaper
  let useSquared = true;
  const recCand = graph.candidates?.find((c) => c.name === graph.recommended_shaper);
  if (recCand && recCand.attenuation_curve.length >= n && recShaped.length >= n) {
    const rVal = raw[peakIdx];
    const att = recCand.attenuation_curve[peakIdx];
    if (rVal > 1e-12 && att > 0.02 && att < 0.98) {
      const ratio = recShaped[peakIdx] / rVal;
      useSquared = Math.abs(ratio - att * att) <= Math.abs(ratio - att);
    }
  }

  const isRecommended = !activeCandidate || activeCandidate.name === graph.recommended_shaper;
  const shaped: number[] =
    isRecommended && recShaped.length >= n
      ? recShaped.slice(0, n)
      : raw.slice(0, n).map((val, i) => {
          const att = activeCandidate?.attenuation_curve?.[i] ?? 1;
          return val * (useSquared ? att * att : att);
        });

  const ghost = extras?.ghost ?? null;
  const carriageRaw = extras?.carriageRawPsd ?? null;

  let maxPsdRaw = 1e-12;
  for (let i = 0; i < n; i++) {
    if (raw[i] > maxPsdRaw) maxPsdRaw = raw[i];
    if (shaped[i] > maxPsdRaw) maxPsdRaw = shaped[i];
  }
  if (ghost) {
    const gn = Math.min(ghost.freqs_hz.length, ghost.raw_psd.length);
    for (let i = 0; i < gn; i++) {
      if (ghost.raw_psd[i] > maxPsdRaw) maxPsdRaw = ghost.raw_psd[i];
    }
  }
  if (carriageRaw) {
    const cn = Math.min(n, carriageRaw.length);
    for (let i = 0; i < cn; i++) {
      if (carriageRaw[i] > maxPsdRaw) maxPsdRaw = carriageRaw[i];
    }
  }

  const maxPsd = maxPsdRaw * 1.08;
  const toX = (f: number) => padL + ((f - fMin) / fSpan) * plotW;
  const toPsdY = (v: number) => padT + (1 - Math.max(0, v) / maxPsd) * plotH;
  const toAttY = (v: number) => padT + (1 - Math.max(0, Math.min(v, 1.05)) / 1.05) * plotH;

  const rawPts: string[] = [];
  const shapedPts: string[] = [];
  const attPts: string[] = [];
  const carriagePts: string[] = [];
  const attCurve = activeCandidate?.attenuation_curve ?? [];

  for (let i = 0; i < n; i++) {
    const x = toX(freqs[i]).toFixed(1);
    rawPts.push(`${x},${toPsdY(raw[i]).toFixed(1)}`);
    shapedPts.push(`${x},${toPsdY(shaped[i]).toFixed(1)}`);
    if (i < attCurve.length) {
      attPts.push(`${x},${toAttY(attCurve[i]).toFixed(1)}`);
    }
    if (carriageRaw && i < carriageRaw.length) {
      carriagePts.push(`${x},${toPsdY(carriageRaw[i]).toFixed(1)}`);
    }
  }

  const ghostPts: string[] = [];
  if (ghost) {
    const gn = Math.min(ghost.freqs_hz.length, ghost.raw_psd.length);
    for (let i = 0; i < gn; i++) {
      const gx = toX(ghost.freqs_hz[i]).toFixed(1);
      ghostPts.push(`${gx},${toPsdY(ghost.raw_psd[i]).toFixed(1)}`);
    }
  }

  const baseLineY = (padT + plotH).toFixed(1);
  const firstX = toX(freqs[0]).toFixed(1);
  const lastX = toX(freqs[n - 1]).toFixed(1);
  const rawArea = `M ${firstX},${baseLineY} L ${rawPts.join(" L ")} L ${lastX},${baseLineY} Z`;
  const shapedArea = `M ${firstX},${baseLineY} L ${shapedPts.join(" L ")} L ${lastX},${baseLineY} Z`;

  const gridFreqs = [50, 100, 150, 200].filter((f) => f >= fMin && f <= fMax);
  const markerFreq = activeCandidate?.freq_hz ?? graph.recommended_freq_hz;
  const markerX = Math.max(padL, Math.min(padL + plotW, toX(markerFreq)));
  const markerLabelRight = markerX <= padL + plotW * 0.62;

  const peakFreq = graph.peak_freq_hz || freqs[peakIdx];
  const peakX = Math.max(padL, Math.min(padL + plotW, toX(peakFreq)));
  const peakY = toPsdY(raw[peakIdx]);
  const showDistinctPeak = Math.abs(peakFreq - markerFreq) >= 3.5;

  const vfaMarkers: PsdSvgComputed["vfaMarkers"] = [];
  const rawVfa = (extras?.vfaMarkers ?? [])
    .filter((m) => Number.isFinite(m.freqHz) && m.freqHz >= fMin && m.freqHz <= fMax)
    .sort((a, b) => a.freqHz - b.freqHz);
  for (let i = 0; i < rawVfa.length; i++) {
    const m = rawVfa[i];
    const x = Math.max(padL, Math.min(padL + plotW, toX(m.freqHz)));
    const prev = vfaMarkers[vfaMarkers.length - 1];
    const nearPrev = prev && Math.abs(x - prev.x) < 34;
    const labelY = nearPrev ? prev.labelY + 10 : padT + 20;
    const anchor: "start" | "end" = x > padL + plotW - 34 ? "end" : "start";
    vfaMarkers.push({
      id: m.id,
      label: m.label,
      freqHz: m.freqHz,
      x,
      color: m.color,
      labelY,
      anchor,
    });
  }

  return {
    W,
    H,
    padL,
    padR,
    padT,
    padB,
    plotW,
    plotH,
    toX,
    toAttY,
    rawLine: rawPts.join(" "),
    rawArea,
    shapedLine: shapedPts.join(" "),
    shapedArea,
    attLine: attPts.join(" "),
    ghostLine: ghostPts.join(" "),
    carriageLine: carriagePts.join(" "),
    vfaMarkers,
    gridFreqs,
    markerFreq,
    markerX,
    markerLabelRight,
    peakFreq,
    peakX,
    peakY,
    showDistinctPeak,
  };
}

const PsdSvgChart: React.FC<{
  svgData: PsdSvgComputed;
  gradId: string;
  markerName: string;
}> = ({ svgData, gradId, markerName }) => (
  <svg viewBox={`0 0 ${svgData.W} ${svgData.H}`} className="psd-svg">
    <defs>
      <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stopColor="#38bdf8" stopOpacity="0.38" />
        <stop offset="100%" stopColor="#38bdf8" stopOpacity="0.03" />
      </linearGradient>
    </defs>

    <rect
      x="0"
      y="0"
      width={svgData.W}
      height={svgData.H}
      rx="6"
      fill="#0b1019"
      stroke="#223045"
      strokeWidth="1"
    />

    {/* Horizontal gridlines */}
    {[0.25, 0.5, 0.75, 1].map((frac) => {
      const y = svgData.padT + svgData.plotH * (1 - frac);
      return (
        <line
          key={`hg-${frac}`}
          x1={svgData.padL}
          y1={y}
          x2={svgData.padL + svgData.plotW}
          y2={y}
          stroke="#172233"
          strokeDasharray="2 2"
        />
      );
    })}

    {/* Baseline */}
    <line
      x1={svgData.padL}
      y1={svgData.padT + svgData.plotH}
      x2={svgData.padL + svgData.plotW}
      y2={svgData.padT + svgData.plotH}
      stroke="#243248"
    />

    {/* Vertical frequency gridlines (50, 100, 150, 200 Hz) */}
    {svgData.gridFreqs.map((f) => {
      const x = svgData.toX(f);
      return (
        <g key={`fg-${f}`}>
          <line
            x1={x}
            y1={svgData.padT}
            x2={x}
            y2={svgData.padT + svgData.plotH}
            stroke="#1c293d"
            strokeDasharray="2 3"
          />
          <text
            x={x}
            y={svgData.H - 5}
            fill="#7d90ac"
            fontSize="8.5"
            textAnchor="middle"
            style={{ fontVariantNumeric: "tabular-nums" }}
          >
            {f} Hz
          </text>
        </g>
      );
    })}

    {/* Right-axis V(f) scale labels */}
    {[1, 0.5, 0].map((v) => (
      <text
        key={`att-tick-${v}`}
        x={svgData.W - 4}
        y={svgData.toAttY(v) + 3}
        fill="#c084fc"
        fillOpacity="0.82"
        fontSize="7.5"
        textAnchor="end"
      >
        {v.toFixed(v === 0.5 ? 1 : 0)}
      </text>
    ))}

    {/* Raw PSD filled area & stroke */}
    <path d={svgData.rawArea} fill={`url(#${gradId})`} />
    <polyline fill="none" stroke="#38bdf8" strokeWidth="1.6" points={svgData.rawLine} />

    {/* Optional pinned ghost reference PSD curve */}
    {svgData.ghostLine && (
      <polyline
        fill="none"
        stroke="#94a3b8"
        strokeWidth="1.25"
        strokeOpacity="0.72"
        strokeDasharray="3 2.5"
        points={svgData.ghostLine}
      />
    )}

    {/* Optional Carriage ADXL PSD overlay */}
    {svgData.carriageLine && (
      <polyline
        fill="none"
        stroke="#f97316"
        strokeWidth="1.45"
        strokeOpacity="0.9"
        strokeDasharray="4 2.5"
        points={svgData.carriageLine}
      />
    )}

    {/* Shaped Residual PSD filled area & stroke */}
    <path d={svgData.shapedArea} fill="rgba(34, 197, 94, 0.14)" />
    <polyline fill="none" stroke="#22c55e" strokeWidth="1.8" points={svgData.shapedLine} />

    {/* Selected / Recommended Shaper Attenuation V(f) curve */}
    {svgData.attLine && (
      <polyline
        fill="none"
        stroke="#c084fc"
        strokeWidth="1.45"
        strokeDasharray="4 2.5"
        points={svgData.attLine}
      />
    )}

    {/* Optional non-scored VFA harmonic vertical markers */}
    {svgData.vfaMarkers.map((m) => (
      <g key={`vfa-${m.id}`}>
        <line
          x1={m.x}
          y1={svgData.padT}
          x2={m.x}
          y2={svgData.padT + svgData.plotH}
          stroke={m.color}
          strokeOpacity="0.72"
          strokeWidth="1.15"
          strokeDasharray="2 2"
        />
        <text
          x={m.anchor === "start" ? m.x + 3 : m.x - 3}
          y={m.labelY}
          fill={m.color}
          fontSize="7.5"
          fontWeight="600"
          textAnchor={m.anchor}
        >
          {m.label}
        </text>
      </g>
    ))}

    {/* Peak resonance dot on raw PSD when distinct from shaper marker */}
    {svgData.showDistinctPeak && (
      <g>
        <line
          x1={svgData.peakX}
          y1={svgData.peakY}
          x2={svgData.peakX}
          y2={svgData.padT + svgData.plotH}
          stroke="#38bdf8"
          strokeOpacity="0.45"
          strokeWidth="1"
          strokeDasharray="2 2"
        />
        <circle
          cx={svgData.peakX}
          cy={svgData.peakY}
          r="2.6"
          fill="#38bdf8"
          stroke="#0b1019"
          strokeWidth="1"
        />
      </g>
    )}

    {/* Vertical dashed marker at selected shaper frequency */}
    <line
      x1={svgData.markerX}
      y1={svgData.padT - 2}
      x2={svgData.markerX}
      y2={svgData.padT + svgData.plotH}
      stroke="#f59e0b"
      strokeWidth="1.3"
      strokeDasharray="3 2.5"
    />
    <circle cx={svgData.markerX} cy={svgData.padT} r="2.6" fill="#f59e0b" />
    <text
      x={svgData.markerLabelRight ? svgData.markerX + 5 : svgData.markerX - 5}
      y={svgData.padT + 9}
      fill="#fcd34d"
      fontSize="8.5"
      fontWeight="700"
      textAnchor={svgData.markerLabelRight ? "start" : "end"}
    >
      {markerName} @ {svgData.markerFreq.toFixed(1)} Hz
    </text>
  </svg>
);

const StepResponseSvgChart: React.FC<{
  stepData: StepResponseComputed;
}> = ({ stepData }) => {
  const y1 = stepData.toY(1.0);
  const yBandTop = stepData.toY(1.02);
  const yBandBot = stepData.toY(0.98);

  return (
    <svg viewBox={`0 0 ${stepData.W} ${stepData.H}`} className="psd-svg">
      <rect
        x="0"
        y="0"
        width={stepData.W}
        height={stepData.H}
        rx="6"
        fill="#0b1019"
        stroke="#223045"
        strokeWidth="1"
      />

      {/* Horizontal gridlines & left Y-axis ticks */}
      {[0, 0.5, 1.0, 1.5].map((v) => {
        if (v > stepData.yMax + 0.01) return null;
        const y = stepData.toY(v);
        return (
          <g key={`sy-${v}`}>
            <line
              x1={stepData.padL}
              y1={y}
              x2={stepData.padL + stepData.plotW}
              y2={y}
              stroke={v === 1.0 ? "#334155" : "#172233"}
              strokeDasharray={v === 0 ? undefined : "2 2"}
            />
            <text
              x={stepData.padL - 4}
              y={y + 3}
              fill="#7d90ac"
              fontSize="7.5"
              textAnchor="end"
            >
              {v.toFixed(1)}
            </text>
          </g>
        );
      })}

      {/* ±2% settle tolerance band around 1.0 */}
      <rect
        x={stepData.padL}
        y={yBandTop}
        width={stepData.plotW}
        height={Math.max(1, yBandBot - yBandTop)}
        fill="rgba(34, 197, 94, 0.1)"
      />
      <line
        x1={stepData.padL}
        y1={y1}
        x2={stepData.padL + stepData.plotW}
        y2={y1}
        stroke="#94a3b8"
        strokeOpacity="0.55"
        strokeWidth="0.9"
        strokeDasharray="3 3"
      />

      {/* Vertical time gridlines */}
      {stepData.gridTimesMs.map((t) => {
        const x = stepData.toX(t);
        return (
          <g key={`st-${t}`}>
            <line
              x1={x}
              y1={stepData.padT}
              x2={x}
              y2={stepData.padT + stepData.plotH}
              stroke="#1c293d"
              strokeDasharray="2 3"
            />
            <text
              x={x}
              y={stepData.H - 5}
              fill="#7d90ac"
              fontSize="8.5"
              textAnchor="middle"
            >
              {t} ms
            </text>
          </g>
        );
      })}

      {/* Optional ±5 Hz resonance band envelope */}
      {stepData.bandArea && (
        <>
          <path d={stepData.bandArea} fill="rgba(245, 158, 11, 0.16)" />
          <polyline
            fill="none"
            stroke="#f59e0b"
            strokeWidth="1.05"
            strokeOpacity="0.82"
            strokeDasharray="3 2"
            points={stepData.bandLowLine}
          />
          <polyline
            fill="none"
            stroke="#f59e0b"
            strokeWidth="1.05"
            strokeOpacity="0.82"
            strokeDasharray="3 2"
            points={stepData.bandHighLine}
          />
        </>
      )}

      {/* Unshaped 2nd-order step response */}
      <polyline
        fill="none"
        stroke="#ef4444"
        strokeWidth="1.45"
        strokeOpacity="0.72"
        points={stepData.unshapedLine}
      />

      {/* Shaped step response */}
      <polyline
        fill="none"
        stroke="#22c55e"
        strokeWidth="1.9"
        points={stepData.shapedLine}
      />

      {/* Header shaper tag */}
      <text
        x={stepData.W - stepData.padR}
        y={stepData.padT - 5}
        fill="#7ee2b0"
        fontSize="8.5"
        fontWeight="700"
        textAnchor="end"
      >
        {stepData.shaperLabel}
      </text>
    </svg>
  );
};

export const KlipperShaperPsdCard: React.FC<KlipperShaperPsdCardProps> = ({
  graphs,
  adxlVsNozzle,
  frameDampingRatio,
  extrusionFilling,
  spec,
  motionLimits,
  changed,
  embedded,
}) => {
  const [axis, setAxis] = useState<PsdAxisMode>("x");
  const [chartMode, setChartMode] = useState<ShaperChartMode>("psd");
  const [selectedShaperName, setSelectedShaperName] = useState<string | null>(null);
  const [ghostByAxis, setGhostByAxis] = useState<Partial<Record<"x" | "y", GhostSnapshot>>>({});
  const [showVfaOverlay, setShowVfaOverlay] = useState<boolean>(false);
  const defaultVfaSpeed = Math.max(
    150,
    Math.min(300, Math.round(spec?.kinematics.max_velocity_mm_s ?? 200))
  );
  const [vfaSpeedMmS, setVfaSpeedMmS] = useState<number>(defaultVfaSpeed);
  const [showCarriageAdxl, setShowCarriageAdxl] = useState<boolean>(false);
  const [bandDeltaHz, setBandDeltaHz] = useState<number>(0);
  const [cfgCopied, setCfgCopied] = useState<boolean>(false);
  const [showCfgPreview, setShowCfgPreview] = useState<boolean>(false);
  const copyTimerRef = useRef<number | null>(null);

  const singleAxisKey: "x" | "y" = axis === "y" ? "y" : "x";
  const singleGraph = singleAxisKey === "y" ? graphs.y : graphs.x;
  const currentGhost = ghostByAxis[singleAxisKey] ?? null;
  const axisDiag: AdxlNozzleAxisDiagnostic | undefined = adxlVsNozzle?.[singleAxisKey];

  useEffect(() => {
    return () => {
      if (copyTimerRef.current !== null) {
        window.clearTimeout(copyTimerRef.current);
      }
    };
  }, []);

  // Reset shaper inspection to recommended when axis or simulation graph refreshes
  useEffect(() => {
    setSelectedShaperName(null);
  }, [axis, singleGraph.recommended_shaper, singleGraph.peak_freq_hz]);

  const klipperCfgText = useMemo(
    () => buildKlipperPrinterCfg(graphs, spec, motionLimits),
    [graphs, spec, motionLimits]
  );

  const handleCopyPrinterCfg = async () => {
    let copied = false;
    try {
      if (navigator.clipboard && typeof navigator.clipboard.writeText === "function") {
        await navigator.clipboard.writeText(klipperCfgText);
        copied = true;
      }
    } catch {
      copied = false;
    }

    if (!copied) {
      try {
        const ta = document.createElement("textarea");
        ta.value = klipperCfgText;
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
      setCfgCopied(true);
      if (copyTimerRef.current !== null) {
        window.clearTimeout(copyTimerRef.current);
      }
      copyTimerRef.current = window.setTimeout(() => {
        setCfgCopied(false);
        copyTimerRef.current = null;
      }, 2200);
    }
  };

  const recCandX = useMemo(() => findRecommendedCandidate(graphs.x), [graphs.x]);
  const recCandY = useMemo(() => findRecommendedCandidate(graphs.y), [graphs.y]);

  const activeCandidate: KlipperShaperCandidate | null = useMemo(() => {
    const list = singleGraph.candidates || [];
    if (!list.length) return null;
    if (selectedShaperName) {
      const found = list.find((c) => c.name === selectedShaperName);
      if (found) return found;
    }
    return list.find((c) => c.name === singleGraph.recommended_shaper) ?? list[0];
  }, [singleGraph.candidates, singleGraph.recommended_shaper, selectedShaperName]);

  const vfaInfo = useMemo(() => {
    const beltPitchMm = resolveBeltPitchMm(spec);
    const pulleyTeeth = Math.max(10, spec?.gantry.belt.pulley_teeth ?? 20);
    const stepsPerRev = Math.max(100, spec?.kinematics.xy_stepper.steps_per_rev ?? 200);
    const rotorTeeth = stepsPerRev / 4;
    const circMm = pulleyTeeth * beltPitchMm;
    const rps = vfaSpeedMmS / Math.max(circMm, 1);
    const beltHz = vfaSpeedMmS / beltPitchMm;
    const elecHz = rps * rotorTeeth;
    const stepHz = 4 * elecHz;
    const markers: VfaMarker[] = [
      { id: "belt", label: "Belt", freqHz: beltHz, color: "#fb923c" },
      { id: "elec", label: "Elec", freqHz: elecHz, color: "#2dd4bf" },
      { id: "step", label: "Step", freqHz: stepHz, color: "#fb7185" },
    ];
    return { beltHz, elecHz, stepHz, markers };
  }, [spec, vfaSpeedMmS]);

  const effDamping =
    frameDampingRatio ?? extrusionFilling?.damping_ratio ?? singleGraph.damping_ratio;

  const singleSvgData = useMemo(
    () =>
      axis === "both"
        ? null
        : computePsdSvgData(singleGraph, activeCandidate, 178, {
            ghost: currentGhost,
            carriageRawPsd:
              showCarriageAdxl && axisDiag?.carriage_raw_psd?.length
                ? axisDiag.carriage_raw_psd
                : null,
            vfaMarkers: showVfaOverlay ? vfaInfo.markers : undefined,
          }),
    [
      axis,
      singleGraph,
      activeCandidate,
      currentGhost,
      showCarriageAdxl,
      axisDiag,
      showVfaOverlay,
      vfaInfo.markers,
    ]
  );

  const singleStepData = useMemo(
    () =>
      axis === "both" || chartMode !== "step"
        ? null
        : computeStepSvgData(singleGraph, activeCandidate, effDamping, bandDeltaHz),
    [axis, chartMode, singleGraph, activeCandidate, effDamping, bandDeltaHz]
  );

  const bothSvgX = useMemo(
    () => (axis === "both" ? computePsdSvgData(graphs.x, recCandX, 124) : null),
    [axis, graphs.x, recCandX]
  );

  const bothSvgY = useMemo(
    () => (axis === "both" ? computePsdSvgData(graphs.y, recCandY, 124) : null),
    [axis, graphs.y, recCandY]
  );

  const handleToggleGhost = () => {
    setGhostByAxis((prev) => {
      if (prev[singleAxisKey]) {
        const next = { ...prev };
        delete next[singleAxisKey];
        return next;
      }
      return {
        ...prev,
        [singleAxisKey]: {
          freqs_hz: [...(singleGraph.freqs_hz || [])],
          raw_psd: [...(singleGraph.raw_psd || [])],
          peak_freq_hz: singleGraph.peak_freq_hz,
        },
      };
    });
  };
  const isFilled = Boolean(extrusionFilling && extrusionFilling.id !== "none");
  const fillingShortName = isFilled && extrusionFilling ? extrusionFilling.short_name : "Hollow";

  const minRecAccel = Math.min(
    graphs.x.recommended_max_accel_mm_s2 || Infinity,
    graphs.y.recommended_max_accel_mm_s2 || Infinity
  );
  const limitingAxis: "X" | "Y" =
    graphs.x.recommended_max_accel_mm_s2 <= graphs.y.recommended_max_accel_mm_s2 ? "X" : "Y";

  const peakPillText =
    axis === "both"
      ? `X ${graphs.x.peak_freq_hz.toFixed(0)} / Y ${graphs.y.peak_freq_hz.toFixed(0)} Hz`
      : `Peak ${singleGraph.peak_freq_hz.toFixed(1)} Hz`;

  const exportButtons = (
    <span className="klipper-cfg-actions">
      <button
        type="button"
        className={`klipper-cfg-btn ${cfgCopied ? "copied" : ""}`}
        onClick={() => void handleCopyPrinterCfg()}
        title="Copy ready-to-paste Klipper [printer] & [input_shaper] block to clipboard"
      >
        {cfgCopied ? "Copied ✓" : "Copy printer.cfg"}
      </button>
      <button
        type="button"
        className={`klipper-cfg-preview-btn ${showCfgPreview ? "active" : ""}`}
 aria-pressed={showCfgPreview}
        onClick={() => setShowCfgPreview((v) => !v)}
        title="Toggle Klipper printer.cfg snippet preview"
      >
        {showCfgPreview ? "Hide" : "Preview"}
      </button>
    </span>
  );

  const vibClass = (pct?: number) => {
    if (pct == null) return "";
    return pct <= 2.0 ? "tag-ok" : pct <= 6.0 ? "tag-warn" : "tag-bad";
  };

  return (
    <div
      className={
        embedded
          ? "psd-card"
          : `section-card psd-card ${changed ? "card-changed" : ""}`
      }
      style={embedded ? { display: "flex", flexDirection: "column", gap: 8 } : undefined}
    >
      {!embedded ? (
        <div className="section-title" style={{ flexWrap: "wrap", gap: 6 }}>
          <span>Klipper Input Shaper (PSD)</span>
          <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
            <span className="psd-peak-pill">{peakPillText}</span>
            {exportButtons}
          </span>
        </div>
      ) : (
        <div className="klipper-cfg-header-row">
          <span className="psd-peak-pill">{peakPillText}</span>
          {exportButtons}
        </div>
      )}

      {showCfgPreview && (
        <pre className="klipper-cfg-preview">
          <code>{klipperCfgText}</code>
        </pre>
      )}

      <div className="psd-damping-strip">
        <span>Frame Damping</span>
        <span className="psd-damping-val">
          {isFilled && extrusionFilling && (
            <span
              className="swatch-dot"
              style={{ background: extrusionFilling.color_hex }}
            />
          )}
          <strong>ζ = {effDamping.toFixed(3)}</strong>
          <span>·</span>
          <span style={{ color: isFilled ? "var(--accent)" : "var(--muted)" }}>
            {fillingShortName}
          </span>
        </span>
      </div>

      {/* X Axis | Y Axis | Both (X+Y) Toggle Pills */}
      <div
        className="psd-axis-toggle"
        style={{ gridTemplateColumns: "1fr 1fr 1fr", gap: 4 }}
      >
        <button
          type="button"
          className={`psd-axis-btn ${axis === "x" ? "active" : ""}`}
 aria-pressed={axis === "x"}
          style={{ padding: "5px 6px" }}
          onClick={() => setAxis("x")}
          title={`X Axis: ${graphs.x.recommended_shaper} @ ${graphs.x.recommended_freq_hz.toFixed(1)} Hz · ≤${Math.round(graphs.x.recommended_max_accel_mm_s2).toLocaleString()} mm/s²`}
        >
          <span className="psd-axis-top">
            <strong>X Axis</strong>
            <span className="psd-axis-peak">{graphs.x.peak_freq_hz.toFixed(0)} Hz</span>
          </span>
          <span className="psd-axis-rec" style={{ fontSize: 11 }}>
            {graphs.x.recommended_shaper} {graphs.x.recommended_freq_hz.toFixed(0)}Hz
          </span>
        </button>

        <button
          type="button"
          className={`psd-axis-btn ${axis === "y" ? "active" : ""}`}
 aria-pressed={axis === "y"}
          style={{ padding: "5px 6px" }}
          onClick={() => setAxis("y")}
          title={`Y Axis: ${graphs.y.recommended_shaper} @ ${graphs.y.recommended_freq_hz.toFixed(1)} Hz · ≤${Math.round(graphs.y.recommended_max_accel_mm_s2).toLocaleString()} mm/s²`}
        >
          <span className="psd-axis-top">
            <strong>Y Axis</strong>
            <span className="psd-axis-peak">{graphs.y.peak_freq_hz.toFixed(0)} Hz</span>
          </span>
          <span className="psd-axis-rec" style={{ fontSize: 11 }}>
            {graphs.y.recommended_shaper} {graphs.y.recommended_freq_hz.toFixed(0)}Hz
          </span>
        </button>

        <button
          type="button"
          className={`psd-axis-btn ${axis === "both" ? "active" : ""}`}
 aria-pressed={axis === "both"}
          style={{ padding: "5px 6px" }}
          onClick={() => setAxis("both")}
          title="View stacked X and Y Klipper Shake&Tune PSD spectra side-by-side"
        >
          <span className="psd-axis-top">
            <strong>Both (X+Y)</strong>
          </span>
          <span className="psd-axis-rec" style={{ fontSize: 11, color: "var(--accent)" }}>
            {Number.isFinite(minRecAccel)
              ? `≤${Math.round(minRecAccel).toLocaleString()}`
              : "Compare"}
          </span>
        </button>
      </div>

      {/* Single-Axis Mode (X Axis or Y Axis) */}
      {axis !== "both" && (
        <>
          {/* Mode Toggle (PSD | Step Response) & Chart Overlay Chips */}
          <div className="psd-toolbar-row">
            <div
              className="bottleneck-axis-tabs"
              role="group"
              aria-label="Shaper chart view mode"
            >
              <button
                type="button"
                className={`bottleneck-axis-btn ${chartMode === "psd" ? "active" : ""}`}
                aria-pressed={chartMode === "psd"}
                onClick={() => setChartMode("psd")}
              >
                PSD
              </button>
              <button
                type="button"
                className={`bottleneck-axis-btn ${chartMode === "step" ? "active" : ""}`}
                aria-pressed={chartMode === "step"}
                onClick={() => setChartMode("step")}
              >
                Step Response
              </button>
            </div>

            <div className="psd-overlay-chips">
              {chartMode === "psd" ? (
                <>
                  <button
                    type="button"
                    className={`chip-btn ${currentGhost ? "active" : ""}`}
                    aria-pressed={Boolean(currentGhost)}
                    onClick={handleToggleGhost}
                    title={
                      currentGhost
                        ? "Clear pinned ghost PSD reference curve"
                        : "Freeze current raw PSD as a dashed reference curve while tweaking printer specs"
                    }
                  >
                    {currentGhost ? "Clear Ghost" : "Pin Ghost"}
                  </button>
                  <button
                    type="button"
                    className={`chip-btn ${showVfaOverlay ? "active" : ""}`}
                    aria-pressed={showVfaOverlay}
                    onClick={() => setShowVfaOverlay((v) => !v)}
                    title="Overlay belt tooth, motor electrical, and full-step VFA harmonic markers at print speed"
                  >
                    VFA @ Speed
                  </button>
                  {axisDiag && (
                    <button
                      type="button"
                      className={`chip-btn ${showCarriageAdxl ? "active" : ""}`}
                      aria-pressed={showCarriageAdxl}
                      onClick={() => setShowCarriageAdxl((v) => !v)}
                      title="Overlay dashed Carriage ADXL345 PSD against true Nozzle-Tip PSD"
                    >
                      Carriage ADXL
                    </button>
                  )}
                </>
              ) : (
                <button
                  type="button"
                  className={`chip-btn ${bandDeltaHz > 0 ? "active" : ""}`}
                  aria-pressed={bandDeltaHz > 0}
                  onClick={() => setBandDeltaHz((v) => (v > 0 ? 0 : 5))}
                  title="Show ±5 Hz resonance band envelope to compare shaper robustness against wide or split peaks"
                >
                  Band ±5 Hz
                </button>
              )}
            </div>
          </div>

          {chartMode === "psd" && showVfaOverlay && (
            <div className="psd-vfa-strip">
              <div className="psd-vfa-control-row">
                <label htmlFor={`psd-vfa-speed-${singleAxisKey}`}>VFA Speed (mm/s)</label>
                <input
                  id={`psd-vfa-speed-${singleAxisKey}`}
                  type="range"
                  min={50}
                  max={600}
                  step={5}
                  value={vfaSpeedMmS}
                  onChange={(e) => setVfaSpeedMmS(Number(e.target.value))}
                />
                <span className="psd-vfa-speed-val">{vfaSpeedMmS} mm/s</span>
              </div>
              <div className="psd-vfa-freqs">
                <span>Belt {vfaInfo.beltHz.toFixed(0)} Hz</span>
                <span>·</span>
                <span>Elec {vfaInfo.elecHz.toFixed(0)} Hz</span>
                <span>·</span>
                <span>Step {vfaInfo.stepHz.toFixed(0)} Hz</span>
              </div>
              <div className="psd-vfa-note">
                Continuous periodic forcing — input shapers cancel transient ringing, not steady-state VFA.
              </div>
            </div>
          )}

          {chartMode === "psd" && singleSvgData && (
            <div className="psd-chart-wrap">
              <div className="psd-legend-row">
                <span className="psd-legend-item">
                  <span className="psd-swatch raw" /> Raw PSD
                </span>
                <span className="psd-legend-item">
                  <span className="psd-swatch shaped" /> Shaped
                </span>
                <span className="psd-legend-item">
                  <span className="psd-swatch att" /> V(f) {activeCandidate?.name ?? ""}
                </span>
                {currentGhost && (
                  <span className="psd-legend-item">
                    <span className="psd-swatch ghost" /> Ghost ({currentGhost.peak_freq_hz.toFixed(0)} Hz)
                  </span>
                )}
                {showCarriageAdxl && axisDiag && (
                  <span className="psd-legend-item">
                    <span className="psd-swatch carriage" /> Carriage ADXL
                  </span>
                )}
              </div>

              <PsdSvgChart
                svgData={singleSvgData}
                gradId={`psdRawGrad-${axis}-full`}
                markerName={activeCandidate?.name ?? singleGraph.recommended_shaper}
              />
            </div>
          )}

          {chartMode === "psd" && axisDiag && (
            <div
              className={`psd-adxl-diag ${axisDiag.has_significant_divergence ? "warn" : ""}`}
            >
              <div className="psd-adxl-diag-head">
                <span>Carriage ADXL vs. Nozzle Tip</span>
                <span
                  className={`badge-pill ${axisDiag.has_significant_divergence ? "warn" : "ok"}`}
                >
                  {axisDiag.has_significant_divergence ? "Divergence" : "Aligned"} ·{" "}
                  {axisDiag.nozzle_to_carriage_ratio.toFixed(2)}×
                </span>
              </div>
              <div className="psd-adxl-diag-grid">
                <div className="psd-adxl-diag-col">
                  <span className="psd-adxl-diag-lbl">Carriage ADXL</span>
                  <strong>
                    {axisDiag.carriage_recommended_shaper} @{" "}
                    {axisDiag.carriage_recommended_freq_hz.toFixed(1)} Hz
                  </strong>
                  <span className="psd-adxl-diag-sub">
                    Peak {axisDiag.carriage_peak_hz.toFixed(1)} Hz · ≤
                    {Math.round(axisDiag.carriage_max_accel_mm_s2).toLocaleString()} mm/s²
                  </span>
                </div>
                <div className="psd-adxl-diag-col">
                  <span className="psd-adxl-diag-lbl">Nozzle Tip (Printed)</span>
                  <strong>
                    {axisDiag.nozzle_recommended_shaper} @{" "}
                    {axisDiag.nozzle_recommended_freq_hz.toFixed(1)} Hz
                  </strong>
                  <span className="psd-adxl-diag-sub">
                    Peak {axisDiag.nozzle_peak_hz.toFixed(1)} Hz · ≤
                    {Math.round(axisDiag.nozzle_max_accel_mm_s2).toLocaleString()} mm/s²
                  </span>
                </div>
              </div>
            </div>
          )}

          {chartMode === "step" && singleStepData && (
            <div className="psd-chart-wrap">
              <div className="psd-legend-row">
                <span className="psd-legend-item">
                  <span className="psd-swatch step-raw" /> Unshaped
                </span>
                <span className="psd-legend-item">
                  <span className="psd-swatch shaped" /> Shaped ({activeCandidate?.name ?? singleGraph.recommended_shaper})
                </span>
                {bandDeltaHz > 0 && (
                  <span className="psd-legend-item">
                    <span className="psd-swatch band" /> ±{bandDeltaHz} Hz band
                  </span>
                )}
              </div>

              <StepResponseSvgChart stepData={singleStepData} />

              <div className="kpi-grid">
                <div
                  className="kpi-box"
                  title="Peak unit-step overshoot before and after input shaping"
                >
                  <span className="kpi-label">Overshoot</span>
                  <span
                    className={`kpi-value psd-step-kpi-val ${
                      singleStepData.overshootShapedPct <= 5
                        ? "tag-ok"
                        : singleStepData.overshootShapedPct <= 15
                          ? "tag-warn"
                          : "tag-bad"
                    }`}
                  >
                    {singleStepData.overshootUnshapedPct.toFixed(0)}% →{" "}
                    {singleStepData.overshootShapedPct.toFixed(1)}%
                  </span>
                  {singleStepData.bandWorstOvershootPct !== null && (
                    <span className="kpi-sub">
                      ±{bandDeltaHz} Hz worst: {singleStepData.bandWorstOvershootPct.toFixed(1)}%
                    </span>
                  )}
                </div>
                <div
                  className="kpi-box"
                  title="Time to settle and remain within ±2% of the commanded step target"
                >
                  <span className="kpi-label">2% Settle</span>
                  <span className="kpi-value psd-step-kpi-val">
                    {singleStepData.settleUnshapedMs.toFixed(0)} →{" "}
                    {singleStepData.settleShapedMs.toFixed(0)} ms
                  </span>
                  <span className="kpi-sub">
                    fₙ {singleGraph.peak_freq_hz.toFixed(1)} Hz · ζ {effDamping.toFixed(3)}
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* Compact 5-Row Shaper Comparison Table */}
          {singleGraph.candidates && singleGraph.candidates.length > 0 && (
            <div className="psd-table-wrap">
              <table className="psd-table">
                <thead>
                  <tr>
                    <th>Shaper</th>
                    <th className="num">Freq</th>
                    <th className="num">Vib %</th>
                    <th className="num">Smooth</th>
                    <th className="num">Max Accel</th>
                  </tr>
                </thead>
                <tbody>
                  {singleGraph.candidates.map((c) => {
                    const isRec = c.name === singleGraph.recommended_shaper;
                    const isSelected = activeCandidate?.name === c.name;
                    return (
                      <tr
                        key={c.name}
                        className={`psd-row ${isSelected ? "selected" : ""} ${isRec ? "recommended" : ""}`}
                        onClick={() => setSelectedShaperName(c.name)}
                        title={`Click to preview ${c.name} attenuation V(f) curve on chart`}
                      >
                        <td>
                          <span className="psd-shaper-cell">
                            <strong>{c.name}</strong>
                            {isRec && <span className="psd-rec-badge">REC</span>}
                          </span>
                        </td>
                        <td className="num">{c.freq_hz.toFixed(1)} Hz</td>
                        <td className={`num ${vibClass(c.vibrations_pct)}`}>
                          {c.vibrations_pct.toFixed(1)}%
                        </td>
                        <td className="num">{c.smoothing.toFixed(3)}</td>
                        <td className="num">
                          {Math.round(c.max_accel_mm_s2).toLocaleString()}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              <div className="psd-table-hint">
                {chartMode === "step"
                  ? "Click any shaper row to compare its step response & overshoot"
                  : "Click any shaper row to inspect its V(f) attenuation & residual curve"}
              </div>
            </div>
          )}
        </>
      )}

      {/* Both (X+Y) Stacked Shake&Tune Mode */}
      {axis === "both" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {/* Side-by-side X & Y recommended shaper summary */}
          <div className="kpi-grid">
            {(
              [
                { key: "X" as const, g: graphs.x, rec: recCandX },
                { key: "Y" as const, g: graphs.y, rec: recCandY },
              ] as const
            ).map(({ key, g, rec }) => {
              const isLim = key === limitingAxis;
              return (
                <div
                  key={key}
                  className="kpi-box"
                  style={{
                    borderColor: isLim ? "rgba(245, 158, 11, 0.55)" : undefined,
                    padding: "6px 7px",
                    gap: 3,
                  }}
                >
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      gap: 4,
                    }}
                  >
                    <span className="kpi-label" style={{ color: "var(--text)" }}>
                      {key} Axis
                    </span>
                    {isLim ? (
                      <span
                        className="psd-rec-badge"
                        style={{
                          background: "rgba(245, 158, 11, 0.16)",
                          borderColor: "rgba(245, 158, 11, 0.5)",
                          color: "var(--amber)",
                        }}
                      >
                        LIMIT
                      </span>
                    ) : (
                      <span className="psd-rec-badge">REC</span>
                    )}
                  </div>
                  <div
                    style={{
                      fontSize: 12,
                      fontWeight: 700,
                      color: "var(--green)",
                      fontVariantNumeric: "tabular-nums",
                    }}
                  >
                    {g.recommended_shaper} @ {g.recommended_freq_hz.toFixed(1)} Hz
                  </div>
                  <div className="kpi-sub" style={{ display: "flex", justifyContent: "space-between", gap: 4 }}>
                    <span>≤{Math.round(g.recommended_max_accel_mm_s2).toLocaleString()}</span>
                    {rec && (
                      <span className={vibClass(rec.vibrations_pct)}>
                        {rec.vibrations_pct.toFixed(1)}% vib
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>

          <div className="psd-legend-row">
            <span className="psd-legend-item">
              <span className="psd-swatch raw" /> Raw PSD
            </span>
            <span className="psd-legend-item">
              <span className="psd-swatch shaped" /> Shaped Residual
            </span>
            <span className="psd-legend-item">
              <span className="psd-swatch att" /> V(f) Recommended
            </span>
          </div>

          {/* Stacked Compact X-Axis PSD */}
          {bothSvgX && (
            <div className="psd-chart-wrap" style={{ gap: 3 }}>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  gap: 6,
                  fontSize: 11,
                  padding: "0 2px",
                }}
              >
                <span style={{ fontWeight: 600, color: "var(--text)" }}>
                  X-Axis{" "}
                  <span style={{ color: "var(--muted)", fontWeight: 500 }}>
                    (Peak {graphs.x.peak_freq_hz.toFixed(1)} Hz)
                  </span>
                </span>
                <button
                  type="button"
                  className="klipper-cfg-preview-btn"
                  onClick={() => setAxis("x")}
                  title="Inspect all X-axis shaper candidates"
                >
                  Table →
                </button>
              </div>
              <PsdSvgChart
                svgData={bothSvgX}
                gradId="psdRawGrad-x-cmp"
                markerName={recCandX?.name ?? graphs.x.recommended_shaper}
              />
            </div>
          )}

          {/* Stacked Compact Y-Axis PSD */}
          {bothSvgY && (
            <div className="psd-chart-wrap" style={{ gap: 3 }}>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  gap: 6,
                  fontSize: 11,
                  padding: "0 2px",
                }}
              >
                <span style={{ fontWeight: 600, color: "var(--text)" }}>
                  Y-Axis{" "}
                  <span style={{ color: "var(--muted)", fontWeight: 500 }}>
                    (Peak {graphs.y.peak_freq_hz.toFixed(1)} Hz)
                  </span>
                </span>
                <button
                  type="button"
                  className="klipper-cfg-preview-btn"
                  onClick={() => setAxis("y")}
                  title="Inspect all Y-axis shaper candidates"
                >
                  Table →
                </button>
              </div>
              <PsdSvgChart
                svgData={bothSvgY}
                gradId="psdRawGrad-y-cmp"
                markerName={recCandY?.name ?? graphs.y.recommended_shaper}
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
};
