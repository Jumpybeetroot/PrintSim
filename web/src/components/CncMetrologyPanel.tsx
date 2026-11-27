import React, { useMemo, useState } from "react";
import {
  PrinterSpecData,
  RCSAData,
  SensitivityData,
  SubsystemKey,
  TimeDomainMetrologyData,
  ViewMode,
  WorkspaceFRFData,
} from "../types";
import { formatHzGain } from "../viewport/partClassifier";
import {
  FlexIsolationTarget,
  inferSubsystemForPartGroup,
  isSameFlexIsolation,
  subsystemLabel,
} from "../viewport/flexIsolation";
import { SurfaceTexturePreview } from "./SurfaceTexturePreview";

const SUBSYSTEM_ORDER: SubsystemKey[] = ["toolhead", "gantry", "frame", "bed"];

const SUBSYSTEM_DISPLAY: Record<
  SubsystemKey,
  { short: string; full: string; color: string }
> = {
  toolhead: { short: "Toolhead", full: "Toolhead & Mount", color: "#f43f5e" },
  gantry: { short: "Gantry", full: "X/Y Gantry", color: "#f59e0b" },
  frame: { short: "Frame", full: "Outer Frame", color: "#38bdf8" },
  bed: { short: "Bed", full: "Bed & Z-Mount", color: "#a855f7" },
};

export function workspaceHeatColors(
  hz: number,
  minHz: number,
  maxHz: number
): { bg: string; border: string } {
  const span = maxHz - minHz;
  const t = span > 0.5 ? Math.max(0, Math.min(1, (hz - minHz) / span)) : 0.65;
  // Red (#ef4444) -> Amber (#f59e0b) -> Emerald/Cyan (#10b981 / #06b6d4)
  let r: number;
  let g: number;
  let b: number;
  if (t < 0.5) {
    const u = t / 0.5;
    r = Math.round(239 + (245 - 239) * u);
    g = Math.round(68 + (158 - 68) * u);
    b = Math.round(68 + (11 - 68) * u);
  } else {
    const u = (t - 0.5) / 0.5;
    r = Math.round(245 + (16 - 245) * u);
    g = Math.round(158 + (185 - 158) * u);
    b = Math.round(11 + (129 - 11) * u);
  }
  return {
    bg: `rgba(${r}, ${g}, ${b}, 0.28)`,
    border: `rgba(${r}, ${g}, ${b}, 0.62)`,
  };
}

function perpErrorHeatColor(errUm: number, scaleUm: number): string {
  const t = Math.max(0, Math.min(1, Math.abs(errUm) / Math.max(scaleUm, 1)));
  // Cyan (#22d3ee) -> Amber (#f59e0b) -> Red (#ef4444)
  let r: number;
  let g: number;
  let b: number;
  if (t < 0.5) {
    const u = t / 0.5;
    r = Math.round(34 + (245 - 34) * u);
    g = Math.round(211 + (158 - 211) * u);
    b = Math.round(238 + (11 - 238) * u);
  } else {
    const u = (t - 0.5) / 0.5;
    r = Math.round(245 + (239 - 245) * u);
    g = Math.round(158 + (68 - 158) * u);
    b = Math.round(11 + (68 - 11) * u);
  }
  return `rgb(${r}, ${g}, ${b})`;
}

export const CncMetrologyContent: React.FC<{
  metrology?: TimeDomainMetrologyData;
  workspaceFrf?: WorkspaceFRFData;
  rcsa?: RCSAData;
  spec?: PrinterSpecData;
}> = ({ metrology, workspaceFrf, rcsa, spec }) => {
  const [traceScenario, setTraceScenario] = useState<"corner" | "zigzag">("corner");
  const hasZigzag = Boolean(
    metrology?.zigzag_trace && (metrology.zigzag_trace.ideal_x_mm?.length ?? 0) >= 2
  );
  const effectiveScenario: "corner" | "zigzag" = hasZigzag ? traceScenario : "corner";
  const ballbarSvg = useMemo(() => {
    const angles = metrology?.ballbar_angles_deg;
    const errs = metrology?.ballbar_radial_error_um;
    if (!angles || !errs) return null;
    const n = Math.min(angles.length, errs.length);
    if (n < 3) return null;

    const cx = 90;
    const cy = 90;
    const r0 = 52;
    const maxPix = 21;

    let maxAbs = 0;
    for (let i = 0; i < n; i++) {
      const a = Math.abs(errs[i] ?? 0);
      if (a > maxAbs) maxAbs = a;
    }
    const scaleUm = Math.max(5, Math.ceil(Math.max(maxAbs, 1) / 5) * 5);

    const pts: string[] = [];
    for (let i = 0; i < n; i++) {
      const rad = ((angles[i] ?? 0) * Math.PI) / 180;
      const err = errs[i] ?? 0;
      const r = Math.max(12, Math.min(78, r0 + (err / scaleUm) * maxPix));
      const x = cx + r * Math.cos(rad);
      const y = cy - r * Math.sin(rad);
      pts.push(`${x.toFixed(1)},${y.toFixed(1)}`);
    }

    return {
      cx,
      cy,
      r0,
      innerR: r0 - maxPix * 0.72,
      outerR: r0 + maxPix * 0.72,
      scaleUm,
      polygonPts: pts.join(" "),
    };
  }, [metrology?.ballbar_angles_deg, metrology?.ballbar_radial_error_um]);

  const cornerSvg = useMemo(() => {
    const trace = metrology?.corner_trace;
    if (!trace) return null;
    const ix = trace.ideal_x_mm || [];
    const iy = trace.ideal_y_mm || [];
    const ax = trace.actual_x_mm || [];
    const ay = trace.actual_y_mm || [];
    const ux = trace.unshaped_x_mm || [];
    const uy = trace.unshaped_y_mm || [];
    const cx = trace.shaped_cmd_x_mm || [];
    const cy = trace.shaped_cmd_y_mm || [];
    const n = Math.min(ix.length, iy.length);
    if (n < 2) return null;

    let iMinX = Infinity,
      iMaxX = -Infinity,
      iMinY = Infinity,
      iMaxY = -Infinity;
    for (let i = 0; i < n; i++) {
      if (ix[i] < iMinX) iMinX = ix[i];
      if (ix[i] > iMaxX) iMaxX = ix[i];
      if (iy[i] < iMinY) iMinY = iy[i];
      if (iy[i] > iMaxY) iMaxY = iy[i];
    }
    const idealSpan = Math.max(iMaxX - iMinX, iMaxY - iMinY, 1e-3);

    const nAct = Math.min(n, ax.length, ay.length);
    const nUn = Math.min(n, ux.length, uy.length);
    const nCmd = Math.min(n, cx.length, cy.length);

    let maxDev = 0;
    for (let i = 0; i < nAct; i++) {
      const d = Math.hypot((ax[i] ?? ix[i]) - ix[i], (ay[i] ?? iy[i]) - iy[i]);
      if (d > maxDev) maxDev = d;
    }
    for (let i = 0; i < nUn; i++) {
      const d = Math.hypot((ux[i] ?? ix[i]) - ix[i], (uy[i] ?? iy[i]) - iy[i]);
      if (d > maxDev) maxDev = d;
    }

    let devGain = 1;
    if (maxDev > 1e-5 && maxDev / idealSpan < 0.08) {
      const rawGain = (0.14 * idealSpan) / maxDev;
      const steps = [1, 2, 5, 10, 15, 20, 25];
      devGain = steps.find((s) => s >= rawGain) ?? 25;
    }

    const scaledAct: Array<[number, number]> = [];
    for (let i = 0; i < nAct; i++) {
      scaledAct.push([
        ix[i] + ((ax[i] ?? ix[i]) - ix[i]) * devGain,
        iy[i] + ((ay[i] ?? iy[i]) - iy[i]) * devGain,
      ]);
    }
    const scaledUn: Array<[number, number]> = [];
    for (let i = 0; i < nUn; i++) {
      scaledUn.push([
        ix[i] + ((ux[i] ?? ix[i]) - ix[i]) * devGain,
        iy[i] + ((uy[i] ?? iy[i]) - iy[i]) * devGain,
      ]);
    }
    const scaledCmd: Array<[number, number]> = [];
    for (let i = 0; i < nCmd; i++) {
      scaledCmd.push([
        ix[i] + ((cx[i] ?? ix[i]) - ix[i]) * devGain,
        iy[i] + ((cy[i] ?? iy[i]) - iy[i]) * devGain,
      ]);
    }

    let minX = iMinX,
      maxX = iMaxX,
      minY = iMinY,
      maxY = iMaxY;
    for (const [x, y] of scaledAct) {
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    for (const [x, y] of scaledUn) {
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    for (const [x, y] of scaledCmd) {
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }

    const W = 240;
    const H = 140;
    const padL = 12;
    const padR = 12;
    const padT = 22;
    const padB = 12;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;
    const spanX = Math.max(maxX - minX, 1e-4);
    const spanY = Math.max(maxY - minY, 1e-4);
    const scale = Math.min(plotW / spanX, plotH / spanY);
    const offX = padL + (plotW - spanX * scale) * 0.5;
    const offY = padT + (plotH - spanY * scale) * 0.5;

    const mapPt = (x: number, y: number): string => {
      const sx = offX + (x - minX) * scale;
      const sy = offY + (maxY - y) * scale;
      return `${sx.toFixed(1)},${sy.toFixed(1)}`;
    };

    const idealPts: string[] = [];
    for (let i = 0; i < n; i++) {
      idealPts.push(mapPt(ix[i], iy[i]));
    }
    const actPts = scaledAct.map(([x, y]) => mapPt(x, y));
    const unPts = scaledUn.map(([x, y]) => mapPt(x, y));
    const cmdPts = scaledCmd.map(([x, y]) => mapPt(x, y));

    return {
      W,
      H,
      padL,
      padT,
      plotW,
      plotH,
      devGain,
      idealLine: idealPts.join(" "),
      actualLine: actPts.join(" "),
      unshapedLine: unPts.join(" "),
      shapedCmdLine: cmdPts.length >= 2 ? cmdPts.join(" ") : "",
    };
  }, [metrology?.corner_trace]);

  const zigzagSvg = useMemo(() => {
    const trace = metrology?.zigzag_trace;
    if (!trace) return null;
    const ix = trace.ideal_x_mm || [];
    const iy = trace.ideal_y_mm || [];
    const ax = trace.actual_x_mm || [];
    const ay = trace.actual_y_mm || [];
    const ux = trace.unshaped_x_mm || [];
    const uy = trace.unshaped_y_mm || [];
    const perp = trace.perp_error_um || [];
    const n = Math.min(ix.length, iy.length);
    if (n < 2) return null;

    let iMinX = Infinity,
      iMaxX = -Infinity,
      iMinY = Infinity,
      iMaxY = -Infinity;
    for (let i = 0; i < n; i++) {
      if (ix[i] < iMinX) iMinX = ix[i];
      if (ix[i] > iMaxX) iMaxX = ix[i];
      if (iy[i] < iMinY) iMinY = iy[i];
      if (iy[i] > iMaxY) iMaxY = iy[i];
    }
    const idealSpan = Math.max(iMaxX - iMinX, iMaxY - iMinY, 1e-3);

    const nAct = Math.min(n, ax.length, ay.length);
    const nUn = Math.min(n, ux.length, uy.length);

    let maxDev = 0;
    for (let i = 0; i < nAct; i++) {
      const d = Math.hypot((ax[i] ?? ix[i]) - ix[i], (ay[i] ?? iy[i]) - iy[i]);
      if (d > maxDev) maxDev = d;
    }
    for (let i = 0; i < nUn; i++) {
      const d = Math.hypot((ux[i] ?? ix[i]) - ix[i], (uy[i] ?? iy[i]) - iy[i]);
      if (d > maxDev) maxDev = d;
    }

    let devGain = 1;
    if (maxDev > 1e-5 && maxDev / idealSpan < 0.06) {
      const rawGain = (0.11 * idealSpan) / maxDev;
      const steps = [1, 2, 5, 10, 15, 20];
      devGain = steps.find((s) => s >= rawGain) ?? 20;
    }

    const scaledAct: Array<[number, number]> = [];
    for (let i = 0; i < nAct; i++) {
      scaledAct.push([
        ix[i] + ((ax[i] ?? ix[i]) - ix[i]) * devGain,
        iy[i] + ((ay[i] ?? iy[i]) - iy[i]) * devGain,
      ]);
    }
    const scaledUn: Array<[number, number]> = [];
    for (let i = 0; i < nUn; i++) {
      scaledUn.push([
        ix[i] + ((ux[i] ?? ix[i]) - ix[i]) * devGain,
        iy[i] + ((uy[i] ?? iy[i]) - iy[i]) * devGain,
      ]);
    }

    let minX = iMinX,
      maxX = iMaxX,
      minY = iMinY,
      maxY = iMaxY;
    for (const [x, y] of scaledAct) {
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    for (const [x, y] of scaledUn) {
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }

    const W = 240;
    const H = 148;
    const padL = 12;
    const padR = 12;
    const padT = 24;
    const padB = 12;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;
    const spanX = Math.max(maxX - minX, 1e-4);
    const spanY = Math.max(maxY - minY, 1e-4);
    const scale = Math.min(plotW / spanX, plotH / spanY);
    const offX = padL + (plotW - spanX * scale) * 0.5;
    const offY = padT + (plotH - spanY * scale) * 0.5;

    const toSvgXY = (x: number, y: number): [number, number] => [
      offX + (x - minX) * scale,
      offY + (maxY - y) * scale,
    ];

    const idealPts: string[] = [];
    for (let i = 0; i < n; i++) {
      const [sx, sy] = toSvgXY(ix[i], iy[i]);
      idealPts.push(`${sx.toFixed(1)},${sy.toFixed(1)}`);
    }
    const unPts = scaledUn.map(([x, y]) => {
      const [sx, sy] = toSvgXY(x, y);
      return `${sx.toFixed(1)},${sy.toFixed(1)}`;
    });

    let maxPerp = 0;
    for (let i = 0; i < perp.length; i++) {
      const v = Math.abs(perp[i] ?? 0);
      if (v > maxPerp) maxPerp = v;
    }
    const perpScaleUm = Math.max(25, Math.ceil(Math.max(maxPerp, 1) / 10) * 10);

    const segments: Array<{
      x1: number;
      y1: number;
      x2: number;
      y2: number;
      color: string;
    }> = [];
    for (let i = 0; i < scaledAct.length - 1; i++) {
      const [x1, y1] = toSvgXY(scaledAct[i][0], scaledAct[i][1]);
      const [x2, y2] = toSvgXY(scaledAct[i + 1][0], scaledAct[i + 1][1]);
      const e0 = Math.abs(perp[i] ?? 0);
      const e1 = Math.abs(perp[i + 1] ?? e0);
      segments.push({
        x1,
        y1,
        x2,
        y2,
        color: perpErrorHeatColor(0.5 * (e0 + e1), perpScaleUm),
      });
    }

    return {
      W,
      H,
      padL,
      padT,
      plotW,
      plotH,
      devGain,
      perpScaleUm,
      idealLine: idealPts.join(" "),
      unshapedLine: unPts.join(" "),
      segments,
    };
  }, [metrology?.zigzag_trace]);

  const hasMetrology = Boolean(metrology);
  const hasWorkspace = Boolean(
    workspaceFrf && (workspaceFrf.min_axis_freq_grid_hz?.length ?? 0) > 0
  );
  const hasRcsa = Boolean(rcsa);

  if (!hasMetrology && !hasWorkspace && !hasRcsa) {
    return null;
  }

  const gridN =
    workspaceFrf?.grid_n || workspaceFrf?.min_axis_freq_grid_hz?.length || 5;
  const vCoords = workspaceFrf?.v_coords ?? [];
  const uCoords = workspaceFrf?.u_coords ?? [];
  const minGrid = workspaceFrf?.min_axis_freq_grid_hz ?? [];
  const fxGrid = workspaceFrf?.freq_x_grid_hz ?? [];
  const fyGrid = workspaceFrf?.freq_y_grid_hz ?? [];
  const ccGrid = workspaceFrf?.xy_cross_coupling_grid_pct ?? [];
  const rowIndices = Array.from({ length: minGrid.length }, (_, i) => i);
  if (vCoords.length >= 2 && vCoords[0] < vCoords[vCoords.length - 1]) {
    rowIndices.reverse();
  }

  return (
    <div className="cnc-metrology-card" style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      {workspaceFrf?.drive_type && (
        <div style={{ display: "flex", justifyContent: "flex-end" }}>
          <span className="badge-pill" style={{ fontSize: 11, padding: "1px 6px" }}>
            {workspaceFrf.drive_type.replace(/_/g, " ")}
          </span>
        </div>
      )}

      {/* 1. Virtual ISO 230-4 Circular Ballbar Polar SVG Plot */}
      {metrology && (
        <div className="metrology-subblock">
          <div className="metrology-subhead">
            <span>Virtual ISO 230-4 Ballbar</span>
            <span className="metrology-submeta">
              R={metrology.ballbar_radius_mm ?? 50}mm @ {metrology.ballbar_speed_mm_s ?? 150} mm/s
            </span>
          </div>

          {ballbarSvg && (
            <div className="ballbar-svg-wrap">
              <svg viewBox="0 0 180 180" className="ballbar-svg">
                <rect
                  x="1"
                  y="1"
                  width="178"
                  height="178"
                  rx="8"
                  fill="#0b111b"
                  stroke="#1e293b"
                  strokeWidth="1"
                />
                {/* Outer & inner tolerance rings */}
                <circle
                  cx={ballbarSvg.cx}
                  cy={ballbarSvg.cy}
                  r={ballbarSvg.outerR}
                  fill="none"
                  stroke="#1e293b"
                  strokeWidth="0.9"
                  strokeDasharray="2 2"
                />
                <circle
                  cx={ballbarSvg.cx}
                  cy={ballbarSvg.cy}
                  r={ballbarSvg.innerR}
                  fill="none"
                  stroke="#1e293b"
                  strokeWidth="0.9"
                  strokeDasharray="2 2"
                />
                {/* Diagonal 45° / 135° guides */}
                <line
                  x1={ballbarSvg.cx - 52}
                  y1={ballbarSvg.cy - 52}
                  x2={ballbarSvg.cx + 52}
                  y2={ballbarSvg.cy + 52}
                  stroke="#162033"
                  strokeWidth="0.8"
                />
                <line
                  x1={ballbarSvg.cx + 52}
                  y1={ballbarSvg.cy - 52}
                  x2={ballbarSvg.cx - 52}
                  y2={ballbarSvg.cy + 52}
                  stroke="#162033"
                  strokeWidth="0.8"
                />
                {/* 0°/180° and 90°/270° quadrant crosshairs */}
                <line
                  x1="14"
                  y1={ballbarSvg.cy}
                  x2="166"
                  y2={ballbarSvg.cy}
                  stroke="#334155"
                  strokeWidth="0.9"
                />
                <line
                  x1={ballbarSvg.cx}
                  y1="14"
                  x2={ballbarSvg.cx}
                  y2="166"
                  stroke="#334155"
                  strokeWidth="0.9"
                />
                {/* Ideal dashed 0 µm reference circle */}
                <circle
                  cx={ballbarSvg.cx}
                  cy={ballbarSvg.cy}
                  r={ballbarSvg.r0}
                  fill="none"
                  stroke="#94a3b8"
                  strokeWidth="1.1"
                  strokeDasharray="3 3"
                  strokeOpacity="0.8"
                />
                {/* Magnified polar error trace */}
                <polygon
                  points={ballbarSvg.polygonPts}
                  fill="rgba(34, 211, 238, 0.15)"
                  stroke="#22d3ee"
                  strokeWidth="1.8"
                  strokeLinejoin="round"
                />
                {/* Quadrant markers */}
                <text x="169" y={ballbarSvg.cy + 3} fill="#94a3b8" fontSize="8" textAnchor="end" fontWeight="600">
                  0°
                </text>
                <text x={ballbarSvg.cx} y="11" fill="#94a3b8" fontSize="8" textAnchor="middle" fontWeight="600">
                  90°
                </text>
                <text x="11" y={ballbarSvg.cy + 3} fill="#94a3b8" fontSize="8" textAnchor="start" fontWeight="600">
                  180°
                </text>
                <text x={ballbarSvg.cx} y="175" fill="#94a3b8" fontSize="8" textAnchor="middle" fontWeight="600">
                  270°
                </text>
                <text x="7" y="13" fill="#64748b" fontSize="7.5">
                  0 µm ideal
                </text>
                <text x="173" y="13" fill="#22d3ee" fontSize="7.5" textAnchor="end">
                  ±{ballbarSvg.scaleUm} µm
                </text>
              </svg>
            </div>
          )}

          <div className="kpi-grid-3">
            <div className="kpi-box" title="ISO 230-4 Circularity error G_XY (max - min radius)">
              <span className="kpi-label">G_XY Circ.</span>
              <span
                className={`kpi-value ${
                  (metrology.circularity_gxy_um ?? 0) > 80
                    ? "tag-bad"
                    : (metrology.circularity_gxy_um ?? 0) > 40
                      ? "tag-warn"
                      : "tag-ok"
                }`}
                style={{ fontSize: 13 }}
              >
                {(metrology.circularity_gxy_um ?? 0).toFixed(1)} µm
              </span>
            </div>
            <div className="kpi-box" title="Mean radial servo/shaper lag M_SE">
              <span className="kpi-label">Lag M_SE</span>
              <span className="kpi-value" style={{ fontSize: 13, color: "var(--accent)" }}>
                {(metrology.mean_radial_lag_um ?? 0).toFixed(1)} µm
              </span>
            </div>
            <div className="kpi-box" title="Axis-reversal quadrant protrusion glitch at 0°/90°/180°/270°">
              <span className="kpi-label">Quad Glitch</span>
              <span
                className={`kpi-value ${
                  (metrology.quadrant_glitch_um ?? 0) > 25 ? "tag-warn" : ""
                }`}
                style={{ fontSize: 13 }}
              >
                {(metrology.quadrant_glitch_um ?? 0).toFixed(1)} µm
              </span>
            </div>
          </div>
        </div>
      )}

      {/* 2. Corner & Zig-Zag Ringing Zoom SVG Plot */}
      {metrology && (
        <div className="metrology-subblock">
          <div className="metrology-subhead metrology-subhead-wrap">
            <span>{hasZigzag ? "Corner & Zig-Zag Ringing" : "90° Corner Ringing Zoom"}</span>
            <span className="metrology-head-right">
              {hasZigzag && (
                <span
                  className="bottleneck-axis-tabs"
                  role="group"
                  aria-label="Metrology trace scenario"
                >
                  <button
                    type="button"
                    className={`bottleneck-axis-btn ${effectiveScenario === "corner" ? "active" : ""}`}
                    aria-pressed={effectiveScenario === "corner"}
                    onClick={() => setTraceScenario("corner")}
                  >
                    90° Corner
                  </button>
                  <button
                    type="button"
                    className={`bottleneck-axis-btn ${effectiveScenario === "zigzag" ? "active" : ""}`}
                    aria-pressed={effectiveScenario === "zigzag"}
                    onClick={() => setTraceScenario("zigzag")}
                  >
                    Zig-Zag S
                  </button>
                </span>
              )}
              {effectiveScenario === "corner" && cornerSvg && cornerSvg.devGain > 1 && (
                <span className="metrology-submeta">×{cornerSvg.devGain} zoom</span>
              )}
              {effectiveScenario === "zigzag" && zigzagSvg && zigzagSvg.devGain > 1 && (
                <span className="metrology-submeta">×{zigzagSvg.devGain} zoom</span>
              )}
            </span>
          </div>

          {effectiveScenario === "corner" && (
            <>
              {cornerSvg && (
                <svg viewBox={`0 0 ${cornerSvg.W} ${cornerSvg.H}`} className="corner-zoom-svg">
                  <rect
                    x="0.5"
                    y="0.5"
                    width={cornerSvg.W - 1}
                    height={cornerSvg.H - 1}
                    rx="6"
                    fill="#0b111b"
                    stroke="#1e293b"
                    strokeWidth="1"
                  />
                  {/* Subtle reference cross */}
                  <line
                    x1={cornerSvg.padL}
                    y1={cornerSvg.padT + cornerSvg.plotH * 0.5}
                    x2={cornerSvg.padL + cornerSvg.plotW}
                    y2={cornerSvg.padT + cornerSvg.plotH * 0.5}
                    stroke="#162033"
                    strokeDasharray="2 3"
                  />
                  <line
                    x1={cornerSvg.padL + cornerSvg.plotW * 0.5}
                    y1={cornerSvg.padT}
                    x2={cornerSvg.padL + cornerSvg.plotW * 0.5}
                    y2={cornerSvg.padT + cornerSvg.plotH}
                    stroke="#162033"
                    strokeDasharray="2 3"
                  />
                  {/* Ideal dashed white 90° corner */}
                  <polyline
                    fill="none"
                    stroke="#e2e8f0"
                    strokeWidth="1.2"
                    strokeDasharray="4 3"
                    strokeOpacity="0.78"
                    points={cornerSvg.idealLine}
                  />
                  {/* Optional shaped command reference curve */}
                  {cornerSvg.shapedCmdLine && (
                    <polyline
                      fill="none"
                      stroke="#38bdf8"
                      strokeWidth="1.2"
                      strokeDasharray="3 2"
                      strokeOpacity="0.82"
                      points={cornerSvg.shapedCmdLine}
                    />
                  )}
                  {/* Faded red unshaped ringing trace */}
                  {cornerSvg.unshapedLine && (
                    <polyline
                      fill="none"
                      stroke="#ef4444"
                      strokeWidth="1.4"
                      strokeOpacity="0.56"
                      strokeLinejoin="round"
                      points={cornerSvg.unshapedLine}
                    />
                  )}
                  {/* Bright emerald input-shaped trace */}
                  {cornerSvg.actualLine && (
                    <polyline
                      fill="none"
                      stroke="#10b981"
                      strokeWidth="2.0"
                      strokeLinejoin="round"
                      points={cornerSvg.actualLine}
                    />
                  )}
                  {/* Legend bar */}
                  <line x1="8" y1="11" x2="18" y2="11" stroke="#e2e8f0" strokeWidth="1.2" strokeDasharray="3 2" />
                  <text x="21" y="14" fill="#94a3b8" fontSize="8">
                    Ideal
                  </text>
                  <line x1="50" y1="11" x2="60" y2="11" stroke="#ef4444" strokeWidth="1.5" strokeOpacity="0.7" />
                  <text x="63" y="14" fill="#fca5a5" fontSize="8">
                    Unshaped
                  </text>
                  <line x1="110" y1="11" x2="120" y2="11" stroke="#10b981" strokeWidth="2" />
                  <text x="123" y="14" fill="#6ee7b7" fontSize="8">
                    Shaped
                  </text>
                  {cornerSvg.shapedCmdLine && (
                    <>
                      <line x1="162" y1="11" x2="172" y2="11" stroke="#38bdf8" strokeWidth="1.3" strokeDasharray="3 2" />
                      <text x="175" y="14" fill="#bfe3fb" fontSize="8">
                        Cmd
                      </text>
                    </>
                  )}
                </svg>
              )}

              {(metrology.printed_path_error_um != null ||
                metrology.shaper_path_smoothing_um != null ||
                metrology.cross_track_error_um != null ||
                metrology.along_track_lag_um != null) && (
                <div className="kpi-grid">
                  {metrology.printed_path_error_um != null && (
                    <div
                      className="kpi-box"
                      title="True geometric nearest-point wall deviation from the ideal 90° path (excludes timing lag)"
                    >
                      <span className="kpi-label">Printed Path Error</span>
                      <span
                        className={`kpi-value metrology-kpi-sm ${
                          metrology.printed_path_error_um > 60
                            ? "tag-bad"
                            : metrology.printed_path_error_um > 30
                              ? "tag-warn"
                              : "tag-ok"
                        }`}
                      >
                        {metrology.printed_path_error_um.toFixed(1)} µm
                      </span>
                    </div>
                  )}
                  {metrology.shaper_path_smoothing_um != null && (
                    <div
                      className="kpi-box"
                      title="Geometric corner rounding introduced purely by the input shaper's command filter"
                    >
                      <span className="kpi-label">Shaper Smoothing</span>
                      <span className="kpi-value metrology-kpi-sm">
                        {metrology.shaper_path_smoothing_um.toFixed(1)} µm
                      </span>
                    </div>
                  )}
                  {metrology.cross_track_error_um != null && (
                    <div
                      className="kpi-box"
                      title="Perpendicular wall wobble normal to the toolpath direction (visible ghosting/ringing)"
                    >
                      <span className="kpi-label">Cross-Track Wobble</span>
                      <span
                        className={`kpi-value metrology-kpi-sm ${
                          metrology.cross_track_error_um > 40
                            ? "tag-bad"
                            : metrology.cross_track_error_um > 18
                              ? "tag-warn"
                              : "tag-ok"
                        }`}
                      >
                        {metrology.cross_track_error_um.toFixed(1)} µm
                      </span>
                    </div>
                  )}
                  {metrology.along_track_lag_um != null && (
                    <div
                      className="kpi-box"
                      title="Timing delay along the travel direction from shaper convolution — NOT physical wall displacement (extrusion stays synchronized)"
                    >
                      <span className="kpi-label">Along-Track Lag</span>
                      <span className="kpi-value metrology-kpi-sm metrology-kpi-muted">
                        {metrology.along_track_lag_um.toFixed(1)} µm
                      </span>
                    </div>
                  )}
                </div>
              )}

              <div className="kpi-grid-3">
                <div
                  className="kpi-box"
                  title={`Corner rounding ${(metrology.corner_rounding_um ?? 0).toFixed(1)} µm • Overshoot ${(metrology.corner_overshoot_um ?? 0).toFixed(1)} µm`}
                >
                  <span className="kpi-label">Rounding</span>
                  <span className="kpi-value metrology-kpi-sm">
                    {(metrology.corner_rounding_um ?? 0).toFixed(1)} µm
                  </span>
                  <span className="kpi-sub">
                    +{(metrology.corner_overshoot_um ?? 0).toFixed(0)} µm over
                  </span>
                </div>
                <div
                  className="kpi-box"
                  title={`Peak-to-peak ringing ${(metrology.ringing_peak_to_peak_um ?? 0).toFixed(1)} µm vs ${(metrology.ringing_unshaped_um ?? 0).toFixed(1)} µm unshaped`}
                >
                  <span className="kpi-label">Ringing</span>
                  <span
                    className={`kpi-value metrology-kpi-sm ${
                      (metrology.ringing_peak_to_peak_um ?? 0) > 45
                        ? "tag-bad"
                        : (metrology.ringing_peak_to_peak_um ?? 0) > 20
                          ? "tag-warn"
                          : "tag-ok"
                    }`}
                  >
                    {(metrology.ringing_peak_to_peak_um ?? 0).toFixed(1)} µm
                  </span>
                  <span className="kpi-sub tag-ok">
                    -{Math.round(metrology.ringing_reduction_pct ?? 0)}% vs raw
                  </span>
                </div>
                <div className="kpi-box" title="Distance for residual ringing to decay below visible threshold">
                  <span className="kpi-label">Decay</span>
                  <span className="kpi-value metrology-kpi-sm">
                    {(metrology.ringing_decay_mm ?? 0).toFixed(1)} mm
                  </span>
                  <span className="kpi-sub">
                    raw {(metrology.ringing_unshaped_um ?? 0).toFixed(0)} µm
                  </span>
                </div>
              </div>
            </>
          )}

          {effectiveScenario === "zigzag" && zigzagSvg && (
            <>
              <svg viewBox={`0 0 ${zigzagSvg.W} ${zigzagSvg.H}`} className="corner-zoom-svg">
                <defs>
                  <linearGradient id="zigzagPerpScaleGrad" x1="0" y1="0" x2="1" y2="0">
                    <stop offset="0%" stopColor="#22d3ee" />
                    <stop offset="50%" stopColor="#f59e0b" />
                    <stop offset="100%" stopColor="#ef4444" />
                  </linearGradient>
                </defs>
                <rect
                  x="0.5"
                  y="0.5"
                  width={zigzagSvg.W - 1}
                  height={zigzagSvg.H - 1}
                  rx="6"
                  fill="#0b111b"
                  stroke="#1e293b"
                  strokeWidth="1"
                />
                {/* Ideal dashed S-curve */}
                <polyline
                  fill="none"
                  stroke="#e2e8f0"
                  strokeWidth="1.15"
                  strokeDasharray="4 3"
                  strokeOpacity="0.72"
                  points={zigzagSvg.idealLine}
                />
                {/* Faded red unshaped trace */}
                {zigzagSvg.unshapedLine && (
                  <polyline
                    fill="none"
                    stroke="#ef4444"
                    strokeWidth="1.25"
                    strokeOpacity="0.45"
                    strokeLinejoin="round"
                    points={zigzagSvg.unshapedLine}
                  />
                )}
                {/* Perpendicular-error heatmap-colored shaped trace */}
                {zigzagSvg.segments.map((seg, idx) => (
                  <line
                    key={`zz-${idx}`}
                    x1={seg.x1.toFixed(1)}
                    y1={seg.y1.toFixed(1)}
                    x2={seg.x2.toFixed(1)}
                    y2={seg.y2.toFixed(1)}
                    stroke={seg.color}
                    strokeWidth="2.1"
                    strokeLinecap="round"
                  />
                ))}
                {/* Legend bar */}
                <line x1="8" y1="11" x2="18" y2="11" stroke="#e2e8f0" strokeWidth="1.2" strokeDasharray="3 2" />
                <text x="21" y="14" fill="#94a3b8" fontSize="8">
                  Ideal
                </text>
                <line x1="48" y1="11" x2="58" y2="11" stroke="#ef4444" strokeWidth="1.3" strokeOpacity="0.55" />
                <text x="61" y="14" fill="#fca5a5" fontSize="8">
                  Raw
                </text>
                <text x="128" y="14" fill="#94a3b8" fontSize="7.5" textAnchor="end">
                  0 µm
                </text>
                <rect x="132" y="8" width="52" height="5" rx="2" fill="url(#zigzagPerpScaleGrad)" />
                <text x="188" y="14" fill="#fca5a5" fontSize="7.5">
                  {zigzagSvg.perpScaleUm} µm
                </text>
              </svg>

              <div className="kpi-grid-3">
                <div
                  className="kpi-box"
                  title="Geometric printed path error across rapid Zig-Zag S reversals"
                >
                  <span className="kpi-label">Path Error</span>
                  <span
                    className={`kpi-value metrology-kpi-sm ${
                      (metrology.zigzag_path_error_um ?? 0) > 65
                        ? "tag-bad"
                        : (metrology.zigzag_path_error_um ?? 0) > 30
                          ? "tag-warn"
                          : "tag-ok"
                    }`}
                  >
                    {(metrology.zigzag_path_error_um ?? 0).toFixed(1)} µm
                  </span>
                </div>
                <div
                  className="kpi-box"
                  title="Geometric corner rounding introduced by input shaper across S-curve reversals"
                >
                  <span className="kpi-label">Smoothing</span>
                  <span className="kpi-value metrology-kpi-sm">
                    {(metrology.zigzag_shaper_smoothing_um ?? 0).toFixed(1)} µm
                  </span>
                </div>
                <div
                  className="kpi-box"
                  title={`Residual zig-zag ringing ${(metrology.zigzag_residual_ringing_um ?? 0).toFixed(1)} µm vs ${(metrology.zigzag_unshaped_ringing_um ?? 0).toFixed(1)} µm unshaped`}
                >
                  <span className="kpi-label">Ringing</span>
                  <span
                    className={`kpi-value metrology-kpi-sm ${
                      (metrology.zigzag_residual_ringing_um ?? 0) > 45
                        ? "tag-bad"
                        : (metrology.zigzag_residual_ringing_um ?? 0) > 20
                          ? "tag-warn"
                          : "tag-ok"
                    }`}
                  >
                    {(metrology.zigzag_residual_ringing_um ?? 0).toFixed(1)} µm
                  </span>
                  <span className="kpi-sub">
                    raw {(metrology.zigzag_unshaped_ringing_um ?? 0).toFixed(0)} µm
                  </span>
                </div>
              </div>
            </>
          )}
        </div>
      )}

      {/* 2b. Interactive Printed Wall Surface Texture Preview */}
      {metrology && <SurfaceTexturePreview metrology={metrology} spec={spec} />}

      {/* 3. 5×5 Bed Workspace (x, y) Resonance Heatmap & RCSA Nozzle Whip */}
      {(hasWorkspace || hasRcsa) && (
        <div className="metrology-subblock">
          {hasWorkspace && workspaceFrf && (
            <>
              <div className="metrology-subhead">
                <span>
                  {gridN}×{gridN} Bed Workspace Resonance (Hz)
                </span>
                <span className="metrology-submeta">
                  {workspaceFrf.min_freq_hz.toFixed(0)}–{workspaceFrf.max_freq_hz.toFixed(0)} Hz
                </span>
              </div>

              <div
                className="workspace-heatmap-grid"
                style={{ gridTemplateColumns: `repeat(${gridN}, 1fr)` }}
              >
                {rowIndices.map((r) => {
                  const row = minGrid[r] ?? [];
                  return row.map((hz, c) => {
                    const fx = fxGrid[r]?.[c] ?? hz;
                    const fy = fyGrid[r]?.[c] ?? hz;
                    const cc = ccGrid[r]?.[c] ?? 0;
                    const uPct = Math.round((uCoords[c] ?? c / Math.max(gridN - 1, 1)) * 100);
                    const vPct = Math.round((vCoords[r] ?? r / Math.max(gridN - 1, 1)) * 100);
                    const style = workspaceHeatColors(
                      hz,
                      workspaceFrf.min_freq_hz,
                      workspaceFrf.max_freq_hz
                    );
                    return (
                      <div
                        key={`${r}-${c}`}
                        className="workspace-cell"
                        style={{ background: style.bg, borderColor: style.border }}
                        title={`Bed (${uPct}% X, ${vPct}% Y) • Min ${hz.toFixed(1)} Hz | fX ${fx.toFixed(1)} Hz | fY ${fy.toFixed(1)} Hz | X-Y Coupling ${cc.toFixed(1)}%`}
                      >
                        {Math.round(hz)}
                      </div>
                    );
                  });
                })}
              </div>

              <div className="workspace-axis-caption">
                <span>Front (-Y) Bottom · Rear (+Y) Top</span>
                <span>Center: {workspaceFrf.center_freq_hz.toFixed(1)} Hz</span>
              </div>

              <div className="badge-pill-row" style={{ marginTop: 2 }}>
                <span
                  className={`badge-pill ${
                    workspaceFrf.worst_corner_drop_pct > 25
                      ? "bad"
                      : workspaceFrf.worst_corner_drop_pct > 15
                        ? "warn"
                        : "ok"
                  }`}
                  title="Worst corner resonance drop relative to bed center"
                >
                  Corner Drop: -{workspaceFrf.worst_corner_drop_pct.toFixed(1)}%
                </span>
                <span
                  className={`badge-pill ${workspaceFrf.max_cross_coupling_pct > 15 ? "warn" : ""}`}
                  title="Peak X-Y off-axis modal cross-coupling across the bed"
                >
                  X-Y Coupling: {workspaceFrf.max_cross_coupling_pct.toFixed(1)}%
                </span>
                <span
                  className="badge-pill"
                  style={{ borderColor: "var(--border)", color: "var(--text)", background: "#0d131d" }}
                  title="Effective reflected stepper/servo rotor inertia mass at the carriage"
                >
                  Refl. Rotor: {workspaceFrf.reflected_rotor_mass_g.toFixed(0)} g
                </span>
                {workspaceFrf.midband_instability_risk && (
                  <span
                    className="badge-pill bad"
                    title="Open-loop stepper mid-band resonance instability risk — consider closed-loop or higher damping"
                  >
                    ⚠ Mid-Band Risk
                  </span>
                )}
              </div>
            </>
          )}

          {rcsa && (
            <div className={`rcsa-strip ${rcsa.whip_risk ? "warn" : "ok"}`}>
              <div className="rcsa-header">
                <span>RCSA Nozzle Whip Ratio</span>
                <span className={`badge-pill ${rcsa.whip_risk ? "bad" : "ok"}`}>
                  {rcsa.nozzle_to_sensor_amp_ratio.toFixed(2)}×
                  {rcsa.whip_risk ? " WHIP RISK" : " OK"}
                </span>
              </div>
              <div className="rcsa-sub">
                <span>Sensor {rcsa.sensor_peak_freq_hz.toFixed(1)} Hz</span>
                <span>·</span>
                <span>Nozzle {rcsa.nozzle_tip_peak_freq_hz.toFixed(1)} Hz</span>
                <span>·</span>
                <span>k={rcsa.hotend_bending_stiffness_n_per_mm.toFixed(0)} N/mm</span>
              </div>
              {rcsa.whip_risk && rcsa.warning_message && (
                <div className="rcsa-warning">⚠ {rcsa.warning_message}</div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export const AdjointSensitivityContent: React.FC<{
  sensitivity: SensitivityData;
  viewMode: ViewMode;
  onViewModeChange: (m: ViewMode) => void;
  flexIsolation?: FlexIsolationTarget;
  onIsolateFlexTarget?: (target: FlexIsolationTarget) => void;
}> = ({ sensitivity, viewMode, onViewModeChange, flexIsolation, onIsolateFlexTarget }) => {
  const topStiffen = sensitivity.top_stiffen_targets?.slice(0, 5) ?? [];
  const topLighten = sensitivity.top_lighten_targets?.slice(0, 3) ?? [];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <div className="metrology-subhead">
        <span>3D Adjoint Stiffness &amp; Mass ROI</span>
        <button
          type="button"
          className={`deep-verify-cta ${viewMode === "sensitivity" ? "active" : ""}`}
 aria-pressed={viewMode === "sensitivity"}
          style={{ padding: "2px 8px", fontSize: 11 }}
          onClick={() => onViewModeChange("sensitivity")}
          title="Color structural frame & panel bays by adjoint Hz/USD & Hz/100g ROI in the 3D viewport"
        >
          {viewMode === "sensitivity" ? "Viewing in 3D" : "View in 3D"}
        </button>
      </div>

      {sensitivity.summary_diagnosis && (
        <div className="sens-diagnosis-banner">{sensitivity.summary_diagnosis}</div>
      )}

      {sensitivity.subsystem_breakdown && (() => {
        const bd = sensitivity.subsystem_breakdown;
        const primary = sensitivity.primary_problem_subsystem;
        const thPct = bd.toolhead?.compliance_share_pct ?? 0;
        const ganPct = bd.gantry?.compliance_share_pct ?? 0;
        const frPct = bd.frame?.compliance_share_pct ?? 0;
        const bedPct = bd.bed?.compliance_share_pct ?? 0;

        let calloutText: string | null = null;
        if (thPct > frPct && thPct >= ganPct && thPct >= 15) {
          calloutText = `Primary compliance limit: Toolhead & Mount (${Math.round(
            thPct
          )}%) vs. Outer Frame (${Math.round(frPct)}%)`;
        } else if (ganPct > frPct && ganPct >= 15) {
          calloutText = `Primary compliance limit: X/Y Gantry (${Math.round(
            ganPct
          )}%) vs. Outer Frame (${Math.round(frPct)}%)`;
        } else if (bedPct > frPct && bedPct >= 15) {
          calloutText = `Primary compliance limit: Bed & Z-Mount (${Math.round(
            bedPct
          )}%) vs. Outer Frame (${Math.round(frPct)}%)`;
        }

        return (
          <div className="sens-breakdown-card">
            <div className="sens-breakdown-head">
              <span>4-Subsystem Nozzle-Error Share</span>
              {primary && (
                <span className="badge-pill bad" style={{ padding: "0 6px", fontSize: 11 }}>
                  #1 {SUBSYSTEM_DISPLAY[primary]?.short ?? primary}
                </span>
              )}
            </div>

            <div className="subsystem-bar-track">
              {SUBSYSTEM_ORDER.map((k) => {
                const pct = Math.max(0, bd[k]?.compliance_share_pct ?? 0);
                if (pct <= 0.5) return null;
                return (
                  <div
                    key={k}
                    className="subsystem-bar-seg"
                    style={{ width: `${pct}%`, background: SUBSYSTEM_DISPLAY[k].color }}
                    title={`${bd[k]?.label ?? SUBSYSTEM_DISPLAY[k].full}: ${Math.round(pct)}% of flex`}
                  />
                );
              })}
            </div>

            <div className="sens-subsystem-grid">
              {SUBSYSTEM_ORDER.map((k) => {
                const entry = bd[k];
                const sev = entry?.severity ?? "low";
                const isPrimary = primary === k;
                const subTarget: FlexIsolationTarget = {
                  kind: "subsystem",
                  subsystem: k,
                  label: subsystemLabel(k),
                };
                const isIso = isSameFlexIsolation(flexIsolation, subTarget);
                return (
                  <button
                    key={k}
                    type="button"
                    className={`sens-sub-chip sev-${sev} ${isPrimary ? "primary" : ""} ${
                      isIso ? "is-isolated" : ""
                    }`}
                    aria-pressed={isIso}
                    onClick={() => onIsolateFlexTarget?.(subTarget)}
                    title={
                      entry
                        ? `${entry.label}: ${entry.compliance_share_pct.toFixed(1)}% flex • Stiff sens ${entry.modal_stiff_hz.toFixed(
                            2
                          )} Hz • Mass sens ${entry.modal_mass_hz.toFixed(2)} Hz — Click to isolate in 3D`
                        : `${SUBSYSTEM_DISPLAY[k].full} — Click to isolate in 3D`
                    }
                  >
                    <span className="sens-sub-chip-top">
                      <span className="sens-sub-chip-name">
                        <span
                          className="swatch-dot"
                          style={{ background: SUBSYSTEM_DISPLAY[k].color }}
                        />
                        {SUBSYSTEM_DISPLAY[k].full}
                      </span>
                      <strong className="sens-sub-chip-pct">
                        {Math.round(entry?.compliance_share_pct ?? 0)}%
                      </strong>
                    </span>
                  </button>
                );
              })}
            </div>

            {calloutText && <div className="sens-priority-callout">{calloutText}</div>}
          </div>
        );
      })()}

      {topStiffen.length > 0 && (
        <div className="sens-group">
          <div className="sens-group-label stiffen">Primary Compliance Hotspots (Stiffness Limits)</div>
          {topStiffen.map((t) => {
            const label = t.label ?? t.id;
            const target: FlexIsolationTarget = {
              kind: "part_group",
              partGroupId: t.id,
              label,
              subsystem: inferSubsystemForPartGroup(t.id, t.role),
            };
            const isIso = isSameFlexIsolation(flexIsolation, target);
            return (
              <div
                key={`stiff-${t.id}`}
                className={`sens-row stiffen ${isIso ? "is-isolated" : ""}`}
                title={t.action_hint || `${label} (${t.id})`}
              >
                <div className="sens-row-top">
                  <span className="sens-id">
                    <strong>{label}</strong>
                    {!t.label && <span className="sens-role">{t.role}</span>}
                  </span>
                  {onIsolateFlexTarget && (
                    <button
                      type="button"
                      className={`sens-isolate-btn ${isIso ? "active" : ""}`}
                      aria-pressed={isIso}
                      onClick={() => onIsolateFlexTarget(target)}
                      title={
                        isIso
                          ? `Exit 3D isolation of ${label}`
                          : `Isolate and focus ${label} in 3D viewport`
                      }
                    >
                      {isIso ? "Isolated" : "Isolate 3D"}
                    </button>
                  )}
                </div>
                <div className="sens-badge-row">
                  {(t.est_hz_gain ?? 0) > 0 && (
                    <span className="sens-badge gain">+{formatHzGain(t.est_hz_gain!)} Hz</span>
                  )}
                  {t.compliance_share_pct >= 1 && (
                    <span className="sens-badge flex">
                      {t.compliance_share_pct.toFixed(0)}% of flex
                    </span>
                  )}
                  <span className="sens-badge roi">+{t.hz_per_usd.toFixed(2)} Hz/$</span>
                </div>
                {t.why_it_hurts && <div className="sens-why">{t.why_it_hurts}</div>}
                {t.action_hint && (
                  <div className="sens-fix">
                    <span className="sens-fix-prefix">Main limit:</span> {t.action_hint}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {topLighten.length > 0 && (
        <div className="sens-group">
          <div className="sens-group-label lighten">Primary Inertia Hotspots (Moving-Mass Limits)</div>
          {topLighten.map((t) => {
            const gainPer100g = t.hz_gain_per_100g_saved ?? Math.abs(t.hz_per_100g);
            const label = t.label ?? t.id;
            const target: FlexIsolationTarget = {
              kind: "part_group",
              partGroupId: t.id,
              label,
              subsystem: inferSubsystemForPartGroup(t.id, t.role),
            };
            const isIso = isSameFlexIsolation(flexIsolation, target);
            return (
              <div
                key={`light-${t.id}`}
                className={`sens-row lighten ${isIso ? "is-isolated" : ""}`}
                title={t.action_hint || `${label} (${t.id})`}
              >
                <div className="sens-row-top">
                  <span className="sens-id">
                    <strong>{label}</strong>
                    {!t.label && <span className="sens-role">{t.role}</span>}
                  </span>
                  {onIsolateFlexTarget && (
                    <button
                      type="button"
                      className={`sens-isolate-btn ${isIso ? "active" : ""}`}
                      aria-pressed={isIso}
                      onClick={() => onIsolateFlexTarget(target)}
                      title={
                        isIso
                          ? `Exit 3D isolation of ${label}`
                          : `Isolate and focus ${label} in 3D viewport`
                      }
                    >
                      {isIso ? "Isolated" : "Isolate 3D"}
                    </button>
                  )}
                </div>
                <div className="sens-badge-row">
                  <span className="sens-badge lighten-gain">
                    +{gainPer100g.toFixed(2)} Hz / 100g saved
                  </span>
                </div>
                {t.why_it_hurts && <div className="sens-why">{t.why_it_hurts}</div>}
                {t.action_hint && (
                  <div className="sens-fix">
                    <span className="sens-fix-prefix">Main limit:</span> {t.action_hint}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
