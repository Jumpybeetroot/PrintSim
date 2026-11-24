import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  AxisShaperGraph,
  GeometryExtrusionFillingData,
  KlipperShaperCandidate,
  MotionLimitsData,
  PrinterSpecData,
} from "../types";

export interface KlipperShaperPsdCardProps {
  graphs: { x: AxisShaperGraph; y: AxisShaperGraph };
  frameDampingRatio?: number;
  extrusionFilling?: GeometryExtrusionFillingData;
  spec?: PrinterSpecData;
  motionLimits?: MotionLimitsData;
  changed: boolean;
  embedded?: boolean;
}

type PsdAxisMode = "x" | "y" | "both";

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
  gridFreqs: number[];
  markerFreq: number;
  markerX: number;
  markerLabelRight: boolean;
  peakFreq: number;
  peakX: number;
  peakY: number;
  showDistinctPeak: boolean;
}

function computePsdSvgData(
  graph: AxisShaperGraph,
  activeCandidate: KlipperShaperCandidate | null,
  height: number
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

  const maxPsd = Math.max(...raw.slice(0, n), ...shaped, 1e-12) * 1.08;
  const toX = (f: number) => padL + ((f - fMin) / fSpan) * plotW;
  const toPsdY = (v: number) => padT + (1 - Math.max(0, v) / maxPsd) * plotH;
  const toAttY = (v: number) => padT + (1 - Math.max(0, Math.min(v, 1.05)) / 1.05) * plotH;

  const rawPts: string[] = [];
  const shapedPts: string[] = [];
  const attPts: string[] = [];
  const attCurve = activeCandidate?.attenuation_curve ?? [];

  for (let i = 0; i < n; i++) {
    const x = toX(freqs[i]).toFixed(1);
    rawPts.push(`${x},${toPsdY(raw[i]).toFixed(1)}`);
    shapedPts.push(`${x},${toPsdY(shaped[i]).toFixed(1)}`);
    if (i < attCurve.length) {
      attPts.push(`${x},${toAttY(attCurve[i]).toFixed(1)}`);
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

export const KlipperShaperPsdCard: React.FC<KlipperShaperPsdCardProps> = ({
  graphs,
  frameDampingRatio,
  extrusionFilling,
  spec,
  motionLimits,
  changed,
  embedded,
}) => {
  const [axis, setAxis] = useState<PsdAxisMode>("x");
  const [selectedShaperName, setSelectedShaperName] = useState<string | null>(null);
  const [cfgCopied, setCfgCopied] = useState<boolean>(false);
  const [showCfgPreview, setShowCfgPreview] = useState<boolean>(false);
  const copyTimerRef = useRef<number | null>(null);

  const singleGraph = axis === "y" ? graphs.y : graphs.x;

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

  const singleSvgData = useMemo(
    () => (axis === "both" ? null : computePsdSvgData(singleGraph, activeCandidate, 178)),
    [axis, singleGraph, activeCandidate]
  );

  const bothSvgX = useMemo(
    () => (axis === "both" ? computePsdSvgData(graphs.x, recCandX, 124) : null),
    [axis, graphs.x, recCandX]
  );

  const bothSvgY = useMemo(
    () => (axis === "both" ? computePsdSvgData(graphs.y, recCandY, 124) : null),
    [axis, graphs.y, recCandY]
  );

  const effDamping =
    frameDampingRatio ?? extrusionFilling?.damping_ratio ?? singleGraph.damping_ratio;
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
          {singleSvgData && (
            <div className="psd-chart-wrap">
              <div className="psd-legend-row">
                <span className="psd-legend-item">
                  <span className="psd-swatch raw" /> Raw PSD
                </span>
                <span className="psd-legend-item">
                  <span className="psd-swatch shaped" /> Shaped Residual
                </span>
                <span className="psd-legend-item">
                  <span className="psd-swatch att" /> V(f) {activeCandidate?.name ?? ""}
                </span>
              </div>

              <PsdSvgChart
                svgData={singleSvgData}
                gradId={`psdRawGrad-${axis}-full`}
                markerName={activeCandidate?.name ?? singleGraph.recommended_shaper}
              />
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
                Click any shaper row to inspect its V(f) attenuation &amp; residual curve
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
