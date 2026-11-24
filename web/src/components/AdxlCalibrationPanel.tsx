import React, { useMemo, useRef, useState } from "react";
import {
  AdxlAxisCalibrationData,
  AdxlCalibrationResponse,
  AdxlFitResponse,
  PrinterSpecData,
} from "../types";

interface Props {
  spec: PrinterSpecData;
  onApplyFittedSpec?: (fittedSpec: PrinterSpecData, changeLabel?: string) => void;
}

function formatFitValue(val: string | number | boolean | null): string {
  if (val === null || val === undefined) return "—";
  if (typeof val === "boolean") return val ? "On" : "Off";
  if (typeof val === "number") {
    return Number.isInteger(val) ? String(val) : val.toFixed(2).replace(/\.?0+$/, "");
  }
  return val.replace(/_/g, " ");
}

/**
 * Generate a realistic synthetic Klipper `resonances_<axis>.csv` file with a dominant
 * Lorentzian mechanical peak + secondary harmonic so users can immediately test the overlay.
 */
function buildSyntheticKlipperCsv(
  axis: "X" | "Y",
  peakHz: number,
  damping: number,
  secondaryHz: number,
  secondaryWeight: number
): string {
  const lines: string[] = ["freq,psd_x,psd_y,psd_z,psd_xyz,mzv"];
  const gamma1 = peakHz * damping * 2;
  const gamma2 = secondaryHz * (damping * 1.15) * 2;

  for (let f = 5; f <= 200; f += 1) {
    // Second-order SDOF resonance Lorentzian envelope
    const r1 = f / peakHz;
    const denom1 = (1 - r1 * r1) ** 2 + (2 * damping * r1) ** 2;
    const p1 = 1.0 / Math.max(denom1, 1e-4);

    const r2 = f / secondaryHz;
    const denom2 = (1 - r2 * r2) ** 2 + (2 * damping * 1.15 * r2) ** 2;
    const p2 = secondaryWeight / Math.max(denom2, 1e-4);

    // Deterministic gentle spectral ripple (no Math.random flicker)
    const ripple =
      1 +
      0.04 * Math.sin(f * 0.63 + (axis === "X" ? 0.5 : 1.7)) +
      0.025 * Math.cos(f * 1.29);
    const floor = 0.015 * (1 + 0.3 * Math.sin(f * 0.25));
    const dominant = Math.max(0, (p1 + p2) * 1800 * ripple + floor * 1800);
    const cross = dominant * 0.08 + 12 * (gamma1 / (gamma1 + gamma2));
    const zVal = dominant * 0.05 + 8;

    const psdX = axis === "X" ? dominant : cross;
    const psdY = axis === "Y" ? dominant : cross;
    const psdXyz = psdX + psdY + zVal;

    lines.push(
      `${f.toFixed(1)},${psdX.toExponential(4)},${psdY.toExponential(4)},${zVal.toExponential(
        4
      )},${psdXyz.toExponential(4)},0.0`
    );
  }
  return lines.join("\n");
}

function peakDeltaClass(errorPct: number): "ok" | "warn" | "bad" {
  const absErr = Math.abs(errorPct);
  if (absErr <= 8) return "ok";
  if (absErr <= 18) return "warn";
  return "bad";
}

function correlationClass(r: number): string {
  if (r >= 0.85) return "tag-ok";
  if (r >= 0.65) return "tag-warn";
  return "tag-bad";
}

const AdxlAxisOverlayCard: React.FC<{ data: AdxlAxisCalibrationData }> = ({ data }) => {
  const svg = useMemo(() => {
    const freqs = data.freqs_hz || [];
    const meas = data.measured_psd_norm || [];
    const sim = data.simulated_psd_norm || [];
    const n = Math.min(freqs.length, meas.length, sim.length);
    if (n < 2) return null;

    const W = 316;
    const H = 142;
    const padL = 12;
    const padR = 12;
    const padT = 20;
    const padB = 22;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    const fMin = Math.min(5, freqs[0] ?? 5);
    const fMax = Math.max(180, freqs[n - 1] ?? 200);
    const fSpan = Math.max(fMax - fMin, 1);

    const maxVal = Math.max(...meas.slice(0, n), ...sim.slice(0, n), 1.0) * 1.08;
    const toX = (f: number) => padL + ((f - fMin) / fSpan) * plotW;
    const toY = (v: number) => padT + (1 - Math.max(0, v) / maxVal) * plotH;

    const measPts: string[] = [];
    const simPts: string[] = [];
    for (let i = 0; i < n; i++) {
      const x = toX(freqs[i]).toFixed(1);
      measPts.push(`${x},${toY(meas[i]).toFixed(1)}`);
      simPts.push(`${x},${toY(sim[i]).toFixed(1)}`);
    }

    const baseLineY = (padT + plotH).toFixed(1);
    const firstX = toX(freqs[0]).toFixed(1);
    const lastX = toX(freqs[n - 1]).toFixed(1);
    const measArea = `M ${firstX},${baseLineY} L ${measPts.join(" L ")} L ${lastX},${baseLineY} Z`;

    const gridFreqs = [50, 100, 150, 200].filter((f) => f >= fMin && f <= fMax);

    const measPeakX = Math.max(padL, Math.min(padL + plotW, toX(data.measured_peak_hz)));
    const simPeakX = Math.max(padL, Math.min(padL + plotW, toX(data.simulated_peak_hz)));

    // Find Y value nearest each peak frequency for marker dots
    let measPeakY = toY(1.0);
    let simPeakY = toY(1.0);
    let bestMeasDist = Infinity;
    let bestSimDist = Infinity;
    for (let i = 0; i < n; i++) {
      const dm = Math.abs(freqs[i] - data.measured_peak_hz);
      if (dm < bestMeasDist) {
        bestMeasDist = dm;
        measPeakY = toY(meas[i]);
      }
      const ds = Math.abs(freqs[i] - data.simulated_peak_hz);
      if (ds < bestSimDist) {
        bestSimDist = ds;
        simPeakY = toY(sim[i]);
      }
    }

    const peaksClose = Math.abs(measPeakX - simPeakX) < 58;
    const measRight = peaksClose
      ? measPeakX >= simPeakX
        ? measPeakX <= padL + plotW - 52
        : false
      : measPeakX <= padL + plotW * 0.65;
    const simRight = peaksClose
      ? !measRight
      : simPeakX <= padL + plotW * 0.65;

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
      measLine: measPts.join(" "),
      measArea,
      simLine: simPts.join(" "),
      gridFreqs,
      measPeakX,
      measPeakY,
      simPeakX,
      simPeakY,
      measRight,
      simRight,
    };
  }, [data]);

  const gradId = `adxlMeasGrad-${data.axis}`;
  const deltaCls = peakDeltaClass(data.peak_error_pct);
  const deltaSign = data.peak_delta_hz >= 0 ? "+" : "";
  const formatLabel = data.source_format === "raw_accel_csv" ? "Raw ADXL" : "Klipper PSD";

  return (
    <div className="adxl-axis-card">
      <div className="adxl-axis-header">
        <div style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
          <strong className="adxl-axis-title">{data.axis}-Axis Overlay</strong>
          <span className="adxl-format-tag">{formatLabel}</span>
        </div>
        <span
          className={`badge-pill ${deltaCls}`}
          title={`Measured ${data.measured_peak_hz.toFixed(1)} Hz vs Simulated ${data.simulated_peak_hz.toFixed(
            1
          )} Hz (${data.peak_error_pct >= 0 ? "+" : ""}${data.peak_error_pct.toFixed(1)}%)`}
        >
          Δ {deltaSign}
          {data.peak_delta_hz.toFixed(1)} Hz ({data.peak_error_pct >= 0 ? "+" : ""}
          {data.peak_error_pct.toFixed(0)}%)
        </span>
      </div>

      {svg && (
        <div className="psd-chart-wrap">
          <div className="psd-legend-row">
            <span className="psd-legend-item">
              <span className="adxl-swatch measured" />
              <span>Measured ({data.measured_peak_hz.toFixed(1)} Hz)</span>
            </span>
            <span className="psd-legend-item">
              <span className="adxl-swatch simulated" />
              <span>Simulated ({data.simulated_peak_hz.toFixed(1)} Hz)</span>
            </span>
          </div>

          <svg viewBox={`0 0 ${svg.W} ${svg.H}`} className="psd-svg">
            <defs>
              <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#10b981" stopOpacity="0.28" />
                <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.02" />
              </linearGradient>
            </defs>

            <rect
              x="0"
              y="0"
              width={svg.W}
              height={svg.H}
              rx="6"
              fill="#0b1019"
              stroke="#243044"
              strokeWidth="1"
            />

            {/* Horizontal grid lines */}
            {[0.25, 0.5, 0.75, 1.0].map((frac) => {
              const y = svg.padT + svg.plotH * (1 - frac);
              return (
                <line
                  key={frac}
                  x1={svg.padL}
                  y1={y}
                  x2={svg.padL + svg.plotW}
                  y2={y}
                  stroke="#182234"
                  strokeWidth="1"
                  strokeDasharray={frac === 1 ? undefined : "2 3"}
                />
              );
            })}

            {/* Vertical frequency grid lines */}
            {svg.gridFreqs.map((f) => {
              const x = svg.toX(f);
              return (
                <g key={f}>
                  <line
                    x1={x}
                    y1={svg.padT}
                    x2={x}
                    y2={svg.padT + svg.plotH}
                    stroke="#182234"
                    strokeWidth="1"
                    strokeDasharray="2 3"
                  />
                  <text
                    x={x}
                    y={svg.H - 6}
                    fill="#64748b"
                    fontSize="8"
                    textAnchor="middle"
                  >
                    {f} Hz
                  </text>
                </g>
              );
            })}

            {/* Baseline */}
            <line
              x1={svg.padL}
              y1={svg.padT + svg.plotH}
              x2={svg.padL + svg.plotW}
              y2={svg.padT + svg.plotH}
              stroke="#273247"
              strokeWidth="1"
            />

            {/* Simulated PSD (dashed amber) */}
            <polyline
              fill="none"
              stroke="#f59e0b"
              strokeWidth="1.7"
              strokeDasharray="4 2.5"
              points={svg.simLine}
            />

            {/* Measured PSD (solid emerald/cyan + subtle fill) */}
            <path d={svg.measArea} fill={`url(#${gradId})`} />
            <polyline
              fill="none"
              stroke="#10b981"
              strokeWidth="1.9"
              points={svg.measLine}
            />

            {/* Simulated peak marker */}
            <line
              x1={svg.simPeakX}
              y1={svg.padT}
              x2={svg.simPeakX}
              y2={svg.padT + svg.plotH}
              stroke="#f59e0b"
              strokeWidth="1"
              strokeDasharray="2 2"
              opacity="0.7"
            />
            <circle
              cx={svg.simPeakX}
              cy={svg.simPeakY}
              r="3"
              fill="#f59e0b"
              stroke="#0b1019"
              strokeWidth="1"
            />
            <text
              x={svg.simRight ? svg.simPeakX + 4 : svg.simPeakX - 4}
              y={svg.padT + 9}
              fill="#fcd34d"
              fontSize="7.8"
              fontWeight="700"
              textAnchor={svg.simRight ? "start" : "end"}
            >
              Sim {data.simulated_peak_hz.toFixed(0)}Hz
            </text>

            {/* Measured peak marker */}
            <line
              x1={svg.measPeakX}
              y1={svg.padT}
              x2={svg.measPeakX}
              y2={svg.padT + svg.plotH}
              stroke="#10b981"
              strokeWidth="1.1"
              opacity="0.8"
            />
            <circle
              cx={svg.measPeakX}
              cy={svg.measPeakY}
              r="3.2"
              fill="#10b981"
              stroke="#0b1019"
              strokeWidth="1"
            />
            <text
              x={svg.measRight ? svg.measPeakX + 4 : svg.measPeakX - 4}
              y={svg.padT + 18}
              fill="#6ee7b7"
              fontSize="7.8"
              fontWeight="700"
              textAnchor={svg.measRight ? "start" : "end"}
            >
              Meas {data.measured_peak_hz.toFixed(0)}Hz
            </text>

            <text x={svg.padL} y={svg.H - 6} fill="#64748b" fontSize="8">
              0
            </text>
          </svg>
        </div>
      )}

      {/* Comparison Metrics Table */}
      <div className="adxl-metrics-list">
        <div className="metric-kv">
          <span>Shaper (Meas vs Sim)</span>
          <span className="metric-val" style={{ textAlign: "right" }}>
            <span style={{ color: "#6ee7b7" }}>
              {data.measured_shaper.toUpperCase()} @ {data.measured_shaper_freq_hz.toFixed(1)} Hz
            </span>
            <span style={{ color: "var(--muted)", margin: "0 4px" }}>vs</span>
            <span style={{ color: "#fcd34d" }}>
              {data.simulated_shaper.toUpperCase()} @ {data.simulated_shaper_freq_hz.toFixed(1)} Hz
            </span>
          </span>
        </div>
        <div className="metric-kv">
          <span>Max Accel (Meas vs Sim)</span>
          <span className="metric-val">
            <span style={{ color: "#6ee7b7" }}>
              {Math.round(data.measured_max_accel_mm_s2).toLocaleString()}
            </span>
            <span style={{ color: "var(--muted)", margin: "0 4px" }}>vs</span>
            <span style={{ color: "#fcd34d" }}>
              {Math.round(data.simulated_max_accel_mm_s2).toLocaleString()} mm/s²
            </span>
          </span>
        </div>
        <div className="metric-kv">
          <span>Damping Ratio (ζ)</span>
          <span className="metric-val">
            <span style={{ color: "#6ee7b7" }}>{data.measured_damping_ratio.toFixed(3)}</span>
            <span style={{ color: "var(--muted)", margin: "0 4px" }}>vs</span>
            <span style={{ color: "#fcd34d" }}>{data.simulated_damping_ratio.toFixed(3)}</span>
          </span>
        </div>
        <div className="metric-kv">
          <span>Spectral Correlation (r)</span>
          <span className={`metric-val ${correlationClass(data.spectral_correlation)}`}>
            r = {data.spectral_correlation.toFixed(2)}
            <span style={{ color: "var(--muted)", fontWeight: 400, marginLeft: 5 }}>
              ({data.sample_count.toLocaleString()} pts)
            </span>
          </span>
        </div>
      </div>

      {data.diagnostics && data.diagnostics.length > 0 && (
        <ul className="adxl-diag-list">
          {data.diagnostics.map((d, idx) => (
            <li key={idx}>{d}</li>
          ))}
        </ul>
      )}
    </div>
  );
};

export const AdxlCalibrationPanel: React.FC<Props> = ({ spec, onApplyFittedSpec }) => {
  const [xCsvText, setXCsvText] = useState<string | null>(null);
  const [xFileName, setXFileName] = useState<string | null>(null);
  const [yCsvText, setYCsvText] = useState<string | null>(null);
  const [yFileName, setYFileName] = useState<string | null>(null);
  const [dragAxis, setDragAxis] = useState<"X" | "Y" | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [fitting, setFitting] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AdxlCalibrationResponse | null>(null);
  const [fitResult, setFitResult] = useState<AdxlFitResponse | null>(null);
  const [fitApplied, setFitApplied] = useState<boolean>(false);

  const xInputRef = useRef<HTMLInputElement | null>(null);
  const yInputRef = useRef<HTMLInputElement | null>(null);

  const handleFileLoad = async (axis: "X" | "Y", file: File | null | undefined) => {
    if (!file) return;
    try {
      const text = await file.text();
      if (!text.trim()) {
        setError(`${axis}-axis CSV file is empty.`);
        return;
      }
      setError(null);
      setFitResult(null);
      setFitApplied(false);
      if (axis === "X") {
        setXCsvText(text);
        setXFileName(file.name);
      } else {
        setYCsvText(text);
        setYFileName(file.name);
      }
    } catch {
      setError(`Failed to read ${axis}-axis CSV file.`);
    }
  };

  const handleDrop = (axis: "X" | "Y", e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragAxis(null);
    const file = e.dataTransfer.files?.[0];
    void handleFileLoad(axis, file);
  };

  const handleLoadSample = () => {
    setError(null);
    setFitResult(null);
    setFitApplied(false);
    setXCsvText(buildSyntheticKlipperCsv("X", 54.0, 0.048, 108.0, 0.16));
    setXFileName("sample_resonances_x.csv");
    setYCsvText(buildSyntheticKlipperCsv("Y", 46.0, 0.055, 89.0, 0.22));
    setYFileName("sample_resonances_y.csv");
  };

  const handleCompare = async () => {
    if (!xCsvText && !yCsvText) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/calibration/adxl", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          spec,
          axis_x_csv: xCsvText,
          axis_y_csv: yCsvText,
          max_freq_hz: 200,
        }),
      });
      if (!res.ok) {
        const errBody = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(errBody.detail || "ADXL345 calibration comparison failed");
      }
      const data: AdxlCalibrationResponse = await res.json();
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const handleAutoFit = async () => {
    if (!xCsvText && !yCsvText) return;
    setFitting(true);
    setError(null);
    setFitApplied(false);
    try {
      const res = await fetch("/api/calibration/adxl/fit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          spec,
          axis_x_csv: xCsvText,
          axis_y_csv: yCsvText,
          max_freq_hz: 200,
        }),
      });
      if (!res.ok) {
        const errBody = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(errBody.detail || "ADXL345 digital-twin auto-fit failed");
      }
      const data: AdxlFitResponse = await res.json();
      setFitResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setFitting(false);
    }
  };

  const handleApplyFit = () => {
    if (!fitResult || !onApplyFittedSpec) return;
    onApplyFittedSpec(fitResult.fitted_spec, "ADXL Auto-Fit Calibration");
    setFitApplied(true);
  };

  const busy = loading || fitting;

  const calibratedAxes = (["X", "Y"] as const)
    .map((ax) => result?.axes?.[ax])
    .filter((d): d is AdxlAxisCalibrationData => Boolean(d));

  return (
    <div className="adxl-panel">
      <div className="adxl-intro">
        Drop Klipper <code>resonances_x/y_*.csv</code> (Shake&amp;Tune /{" "}
        <code>calibrate_shaper</code>) or raw ADXL345 CSVs to overlay real hardware peaks onto
        your simulated PSD.
      </div>

      {/* Two CSV Drop / Pick Slots */}
      <div className="adxl-slot-grid">
        {(["X", "Y"] as const).map((axis) => {
          const fileName = axis === "X" ? xFileName : yFileName;
          const hasCsv = axis === "X" ? Boolean(xCsvText) : Boolean(yCsvText);
          const inputRef = axis === "X" ? xInputRef : yInputRef;
          return (
            <div
              key={axis}
              className={`adxl-drop-slot ${hasCsv ? "loaded" : ""} ${
                dragAxis === axis ? "dragging" : ""
              }`}
              onClick={() => inputRef.current?.click()}
              onDragOver={(e) => {
                e.preventDefault();
                setDragAxis(axis);
              }}
              onDragLeave={() => setDragAxis((prev) => (prev === axis ? null : prev))}
              onDrop={(e) => handleDrop(axis, e)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  inputRef.current?.click();
                }
              }}
              title={`Click or drop ${axis}-axis Klipper/ADXL345 CSV`}
            >
              <input aria-label={`${axis}-axis accelerometer CSV file`}
                ref={inputRef}
                type="file"
                accept=".csv,.txt,text/csv"
                style={{ display: "none" }}
                onChange={(e) => {
                  void handleFileLoad(axis, e.target.files?.[0]);
                  e.target.value = "";
                }}
              />
              <div className="adxl-slot-top">
                <span className="adxl-slot-axis">{axis}-Axis CSV</span>
                {hasCsv && (
                  <button
                    type="button"
                    className="adxl-slot-clear"
                    onClick={(e) => {
                      e.stopPropagation();
                      if (axis === "X") {
                        setXCsvText(null);
                        setXFileName(null);
                      } else {
                        setYCsvText(null);
                        setYFileName(null);
                      }
                    }}
                    title={`Clear ${axis}-axis CSV`}
                  >
                    ×
                  </button>
                )}
              </div>
              <span className="adxl-slot-name">
                {fileName || `resonances_${axis.toLowerCase()}_*.csv`}
              </span>
            </div>
          );
        })}
      </div>

      {/* Action Buttons */}
      <div className="adxl-actions">
        <button
          type="button"
          className="btn"
          style={{ padding: "5px 8px", fontSize: 11 }}
          onClick={handleLoadSample}
          disabled={busy}
          title="Load realistic synthetic Klipper X (~54 Hz) & Y (~46 Hz) CSVs for quick testing"
        >
          Load Sample ADXL
        </button>
        <button
          type="button"
          className="btn primary"
          style={{ padding: "5px 8px", fontSize: 11 }}
          onClick={handleCompare}
          disabled={busy || (!xCsvText && !yCsvText)}
        >
          {loading ? "Comparing..." : "Compare Measured vs. Simulated"}
        </button>
        <button
          type="button"
          className="btn ai"
          style={{ padding: "5px 8px", fontSize: 11 }}
          onClick={handleAutoFit}
          disabled={busy || (!xCsvText && !yCsvText)}
          title="Optimize physical printer parameters (belt tension, joint stiffness, toolhead stiffness) to match your measured ADXL345 resonance peaks"
        >
          {fitting ? "Auto-Fitting..." : "Auto-Fit Model"}
        </button>
      </div>

      {error && <div className="diag-item bad">{error}</div>}

      {fitResult && (
        <div className="adxl-fit-card">
          <div className="adxl-fit-header">
            <strong className="adxl-fit-title">Digital-Twin Auto-Fit</strong>
            <span
              className={`badge-pill ${
                fitResult.after_mean_abs_error_pct <= fitResult.before_mean_abs_error_pct - 0.5 ||
                fitResult.after_mean_abs_error_pct <= 8
                  ? "ok"
                  : "warn"
              }`}
              title={`Mean peak frequency error reduced from ${fitResult.before_mean_abs_error_pct.toFixed(
                1
              )}% to ${fitResult.after_mean_abs_error_pct.toFixed(1)}%`}
            >
              Error: {fitResult.before_mean_abs_error_pct.toFixed(1)}% →{" "}
              {fitResult.after_mean_abs_error_pct.toFixed(1)}%
            </span>
          </div>

          {fitResult.summary && <div className="adxl-fit-summary">{fitResult.summary}</div>}

          {fitResult.changes && fitResult.changes.length > 0 ? (
            <div className="adxl-fit-table-wrap">
              <table className="adxl-fit-table">
                <thead>
                  <tr>
                    <th>Parameter</th>
                    <th>Change</th>
                  </tr>
                </thead>
                <tbody>
                  {fitResult.changes.map((chg) => (
                    <tr key={chg.field} title={chg.reason}>
                      <td>
                        <div className="adxl-fit-param">{chg.label}</div>
                        {chg.reason && <div className="adxl-fit-reason">{chg.reason}</div>}
                      </td>
                      <td className="adxl-fit-vals">
                        <span className="adxl-fit-before">{formatFitValue(chg.before)}</span>
                        <span className="adxl-fit-arrow">→</span>
                        <span className="adxl-fit-after">{formatFitValue(chg.after)}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="adxl-fit-summary" style={{ color: "var(--muted)" }}>
              Model already closely matches measured peaks — no parameter updates required.
            </div>
          )}

          {onApplyFittedSpec && (
            <button
              type="button"
              className={`btn adxl-fit-apply-btn ${fitApplied ? "applied" : ""}`}
              onClick={handleApplyFit}
            >
              {fitApplied ? "Fitted Model Applied ✓" : "Apply Fitted Model to Studio"}
            </button>
          )}
        </div>
      )}

      {result && (
        <div className="adxl-results">
          {result.overall_summary && (
            <div className="diag-item" style={{ fontSize: 11 }}>
              {result.overall_summary}
            </div>
          )}

          {calibratedAxes.map((axisData) => (
            <AdxlAxisOverlayCard key={axisData.axis} data={axisData} />
          ))}

          {result.recommended_adjustments && result.recommended_adjustments.length > 0 && (
            <div className="adxl-recs-box">
              <div className="adxl-recs-title">Model Calibration Suggestions</div>
              <ul className="adxl-diag-list">
                {result.recommended_adjustments.map((rec, idx) => (
                  <li key={idx}>{rec}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
