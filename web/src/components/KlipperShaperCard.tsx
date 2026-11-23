import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  AxisShaperGraph,
  GeometryExtrusionFillingData,
  KlipperShaperCandidate,
  PrinterSpecData,
} from "../types";

export interface KlipperShaperPsdCardProps {
  graphs: { x: AxisShaperGraph; y: AxisShaperGraph };
  frameDampingRatio?: number;
  extrusionFilling?: GeometryExtrusionFillingData;
  spec?: PrinterSpecData;
  changed: boolean;
  embedded?: boolean;
}

function buildKlipperPrinterCfg(
  graphs: { x: AxisShaperGraph; y: AxisShaperGraph },
  spec?: PrinterSpecData
): string {
  const name = spec?.name?.trim() || "Custom Printer";
  const arch = spec?.architecture ?? "corexy";
  const kinematics =
    arch === "cartesian_bedslinger"
      ? "cartesian"
      : arch === "delta"
        ? "delta"
        : "corexy";

  const maxVel = Math.round(spec?.kinematics.max_velocity_mm_s ?? 400);
  const shaperAccelMin = Math.min(
    graphs.x.recommended_max_accel_mm_s2 || Infinity,
    graphs.y.recommended_max_accel_mm_s2 || Infinity
  );
  const maxAccel = Number.isFinite(shaperAccelMin) && shaperAccelMin > 0
    ? Math.round(shaperAccelMin)
    : Math.round(spec?.kinematics.max_accel_mm_s2 ?? 10000);
  const scv = (spec?.kinematics.square_corner_velocity_mm_s ?? 5.0).toFixed(1);

  const shaperX = (graphs.x.recommended_shaper || "mzv").toLowerCase();
  const freqX = graphs.x.recommended_freq_hz.toFixed(1);
  const shaperY = (graphs.y.recommended_shaper || "mzv").toLowerCase();
  const freqY = graphs.y.recommended_freq_hz.toFixed(1);

  const rawRunCurrent =
    spec?.kinematics.xy_stepper.run_current_a ??
    Math.min(spec?.kinematics.xy_stepper.rated_current_a ?? 1.2, 1.2);
  const runCurrentStr = Number(rawRunCurrent.toFixed(2)).toString();
  const supplyV = Math.round(spec?.kinematics.xy_stepper.supply_voltage_v ?? 24);

  return [
    `# PrintSim Recommended Klipper Config — ${name}`,
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
  ].join("\n");
}

export const KlipperShaperPsdCard: React.FC<KlipperShaperPsdCardProps> = ({
  graphs,
  frameDampingRatio,
  extrusionFilling,
  spec,
  changed,
  embedded,
}) => {
  const [axis, setAxis] = useState<"x" | "y">("x");
  const [selectedShaperName, setSelectedShaperName] = useState<string | null>(null);
  const [cfgCopied, setCfgCopied] = useState<boolean>(false);
  const [showCfgPreview, setShowCfgPreview] = useState<boolean>(false);
  const copyTimerRef = useRef<number | null>(null);

  const graph = axis === "x" ? graphs.x : graphs.y;

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
  }, [axis, graph.recommended_shaper, graph.peak_freq_hz]);

  const klipperCfgText = useMemo(
    () => buildKlipperPrinterCfg(graphs, spec),
    [graphs, spec]
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

  const activeCandidate: KlipperShaperCandidate | null = useMemo(() => {
    const list = graph.candidates || [];
    if (!list.length) return null;
    if (selectedShaperName) {
      const found = list.find((c) => c.name === selectedShaperName);
      if (found) return found;
    }
    return list.find((c) => c.name === graph.recommended_shaper) ?? list[0];
  }, [graph.candidates, graph.recommended_shaper, selectedShaperName]);

  const svgData = useMemo(() => {
    const freqs = graph.freqs_hz || [];
    const raw = graph.raw_psd || [];
    const recShaped = graph.shaped_psd || [];
    const n = Math.min(freqs.length, raw.length);
    if (n < 2) return null;

    const W = 316;
    const H = 150;
    const padL = 12;
    const padR = 26;
    const padT = 18;
    const padB = 22;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    const fMin = Math.min(5, freqs[0] ?? 5);
    const fMax = Math.max(220, freqs[n - 1] ?? 220);
    const fSpan = Math.max(fMax - fMin, 1);

    // Match backend PSD attenuation scaling (V vs V^2) when previewing an alternate shaper
    let useSquared = true;
    const recCand = graph.candidates?.find((c) => c.name === graph.recommended_shaper);
    if (recCand && recCand.attenuation_curve.length >= n && recShaped.length >= n) {
      let peakIdx = 0;
      for (let i = 1; i < n; i++) {
        if (raw[i] > raw[peakIdx]) peakIdx = i;
      }
      const rVal = raw[peakIdx];
      const att = recCand.attenuation_curve[peakIdx];
      if (rVal > 1e-12 && att > 0.02 && att < 0.98) {
        const ratio = recShaped[peakIdx] / rVal;
        useSquared = Math.abs(ratio - att * att) <= Math.abs(ratio - att);
      }
    }

    const isRecommended =
      !activeCandidate || activeCandidate.name === graph.recommended_shaper;
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
    };
  }, [graph, activeCandidate]);

  const gradId = `psdRawGrad-${axis}`;
  const effDamping =
    frameDampingRatio ?? extrusionFilling?.damping_ratio ?? graph.damping_ratio;
  const isFilled = Boolean(extrusionFilling && extrusionFilling.id !== "none");
  const fillingShortName = isFilled && extrusionFilling ? extrusionFilling.short_name : "Hollow";

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
        onClick={() => setShowCfgPreview((v) => !v)}
        title="Toggle Klipper printer.cfg snippet preview"
      >
        {showCfgPreview ? "Hide" : "Preview"}
      </button>
    </span>
  );

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
        <div className="section-title">
          <span>Klipper ADXL345 Resonance (PSD)</span>
          <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
            <span className="psd-peak-pill">Peak {graph.peak_freq_hz.toFixed(1)} Hz</span>
            {exportButtons}
          </span>
        </div>
      ) : (
        <div className="klipper-cfg-header-row">
          <span className="psd-peak-pill">Peak {graph.peak_freq_hz.toFixed(1)} Hz</span>
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

      {/* X Axis | Y Axis Toggle Pills */}
      <div className="psd-axis-toggle">
        <button
          type="button"
          className={`psd-axis-btn ${axis === "x" ? "active" : ""}`}
          onClick={() => setAxis("x")}
        >
          <span className="psd-axis-top">
            <strong>X Axis</strong>
            <span className="psd-axis-peak">{graphs.x.peak_freq_hz.toFixed(0)} Hz</span>
          </span>
          <span className="psd-axis-rec">
            {graphs.x.recommended_shaper} @ {graphs.x.recommended_freq_hz.toFixed(0)} Hz · ≤
            {Math.round(graphs.x.recommended_max_accel_mm_s2).toLocaleString()} mm/s²
          </span>
        </button>
        <button
          type="button"
          className={`psd-axis-btn ${axis === "y" ? "active" : ""}`}
          onClick={() => setAxis("y")}
        >
          <span className="psd-axis-top">
            <strong>Y Axis</strong>
            <span className="psd-axis-peak">{graphs.y.peak_freq_hz.toFixed(0)} Hz</span>
          </span>
          <span className="psd-axis-rec">
            {graphs.y.recommended_shaper} @ {graphs.y.recommended_freq_hz.toFixed(0)} Hz · ≤
            {Math.round(graphs.y.recommended_max_accel_mm_s2).toLocaleString()} mm/s²
          </span>
        </button>
      </div>

      {/* Inline SVG Shake&Tune PSD Plot */}
      {svgData && (
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
                    y={svgData.H - 6}
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
                fillOpacity="0.8"
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

            {/* Selected Shaper Attenuation V(f) curve */}
            {svgData.attLine && (
              <polyline
                fill="none"
                stroke="#c084fc"
                strokeWidth="1.45"
                strokeDasharray="4 2.5"
                points={svgData.attLine}
              />
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
              {activeCandidate?.name ?? graph.recommended_shaper} @ {svgData.markerFreq.toFixed(1)} Hz
            </text>
          </svg>
        </div>
      )}

      {/* Compact 5-Row Shaper Comparison Table */}
      {graph.candidates && graph.candidates.length > 0 && (
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
              {graph.candidates.map((c) => {
                const isRec = c.name === graph.recommended_shaper;
                const isSelected = activeCandidate?.name === c.name;
                const vibCls =
                  c.vibrations_pct <= 2.0 ? "tag-ok" : c.vibrations_pct <= 6.0 ? "tag-warn" : "tag-bad";
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
                    <td className={`num ${vibCls}`}>{c.vibrations_pct.toFixed(1)}%</td>
                    <td className="num">{c.smoothing.toFixed(3)}</td>
                    <td className="num">{Math.round(c.max_accel_mm_s2).toLocaleString()}</td>
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
    </div>
  );
};
