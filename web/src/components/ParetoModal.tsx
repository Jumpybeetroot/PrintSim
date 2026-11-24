import React, { useEffect, useMemo, useState } from "react";
import { ParetoOptResponse, ParetoPointData } from "../types";

interface Props {
  open: boolean;
  loading: boolean;
  error: string | null;
  data: ParetoOptResponse | null;
  onClose: () => void;
  onRerun: () => void;
  onApply: (point: ParetoPointData) => void;
}

function formatBackendBadge(backend: string): string {
  const low = backend.toLowerCase();
  if (low.includes("pymoo") && low.includes("nsga")) return "pymoo (NSGA-II)";
  if (low === "pymoo") return "pymoo (NSGA-II)";
  return backend;
}

function scoreDotColor(score: number): string {
  if (score >= 86) return "#22c55e";
  if (score >= 76) return "#10b981";
  if (score >= 66) return "#38bdf8";
  if (score >= 55) return "#a855f7";
  return "#f59e0b";
}

function formatDeltaVal(
  val: number,
  baseVal: number,
  higherIsBetter: boolean,
  eps: number,
  formatter: (d: number) => string
): { text: string; cls: "better" | "worse" | "same" } {
  const delta = val - baseVal;
  if (Math.abs(delta) < eps) {
    return { text: "0", cls: "same" };
  }
  const better = higherIsBetter ? delta > 0 : delta < 0;
  return { text: formatter(delta), cls: better ? "better" : "worse" };
}

function pickDefaultPoint(front: ParetoPointData[]): ParetoPointData | null {
  if (!front.length) return null;
  const knee = front.find((p) => {
    const low = p.label.toLowerCase();
    return low.includes("sweet") || low.includes("knee") || low.includes("balanced");
  });
  if (knee) return knee;
  return [...front].sort((a, b) => b.composite_score - a.composite_score)[0] ?? front[0];
}

function isStandoutLabel(label: string): boolean {
  const low = label.toLowerCase();
  return (
    low.includes("sweet") ||
    low.includes("knee") ||
    low.includes("fastest") ||
    low.includes("cheapest") ||
    low.includes("budget") ||
    low.includes("stiffest")
  );
}

function shortPointLabel(label: string): string {
  if (label.length <= 22) return label;
  return `${label.slice(0, 20)}…`;
}

export const ParetoModal: React.FC<Props> = ({
  open,
  loading,
  error,
  data,
  onClose,
  onRerun,
  onApply,
}) => {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [hoveredId, setHoveredId] = useState<string | null>(null);

  // Close on Escape
  useEffect(() => {
    if (!open) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open, onClose]);

  const sortedFront = useMemo(() => {
    if (!data?.pareto_front) return [];
    return [...data.pareto_front].sort(
      (a, b) => a.total_cost_usd - b.total_cost_usd || a.safe_max_accel_mm_s2 - b.safe_max_accel_mm_s2
    );
  }, [data]);

  // Auto-select sweet spot / knee whenever fresh Pareto results arrive
  useEffect(() => {
    if (!sortedFront.length) {
      setSelectedId(null);
      return;
    }
    const def = pickDefaultPoint(sortedFront);
    setSelectedId(def?.id ?? sortedFront[0].id);
  }, [sortedFront]);

  const selectedPoint = useMemo(() => {
    if (!sortedFront.length) return null;
    return sortedFront.find((p) => p.id === selectedId) ?? pickDefaultPoint(sortedFront);
  }, [sortedFront, selectedId]);

  // SVG Scatter geometry
  const W = 540;
  const H = 370;
  const padL = 62;
  const padR = 24;
  const padT = 24;
  const padB = 46;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const plotModel = useMemo(() => {
    if (!data) return null;
    const allPts = [data.baseline, ...sortedFront];
    const costs = allPts.map((p) => p.total_cost_usd);
    const accels = allPts.map((p) => p.safe_max_accel_mm_s2);

    const rawMinX = Math.min(...costs);
    const rawMaxX = Math.max(...costs);
    const rawMinY = Math.min(...accels);
    const rawMaxY = Math.max(...accels);

    const spanX = Math.max(rawMaxX - rawMinX, 80);
    const spanY = Math.max(rawMaxY - rawMinY, 2000);

    const minX = Math.max(0, rawMinX - spanX * 0.12);
    const maxX = rawMaxX + spanX * 0.14;
    const minY = Math.max(0, rawMinY - spanY * 0.14);
    const maxY = rawMaxY + spanY * 0.14;

    const toX = (cost: number) => padL + ((cost - minX) / Math.max(maxX - minX, 1e-6)) * plotW;
    const toY = (accel: number) => padT + (1 - (accel - minY) / Math.max(maxY - minY, 1e-6)) * plotH;

    const xTicks = Array.from({ length: 5 }, (_, i) => {
      const v = minX + ((maxX - minX) * i) / 4;
      return { val: v, x: toX(v) };
    });

    const yTicks = Array.from({ length: 5 }, (_, i) => {
      const v = minY + ((maxY - minY) * i) / 4;
      return { val: v, y: toY(v) };
    });

    const frontierPolyline = sortedFront
      .map((p) => `${toX(p.total_cost_usd).toFixed(1)},${toY(p.safe_max_accel_mm_s2).toFixed(1)}`)
      .join(" ");

    const baseX = toX(data.baseline.total_cost_usd);
    const baseY = toY(data.baseline.safe_max_accel_mm_s2);

    // Compute collision-avoided labels for standout + selected points
    const labeledCandidates = sortedFront
      .filter((p) => p.id === selectedPoint?.id || isStandoutLabel(p.label))
      .map((p) => {
        const cx = toX(p.total_cost_usd);
        const cy = toY(p.safe_max_accel_mm_s2);
        const alignRight = cx > padL + plotW * 0.62;
        return {
          id: p.id,
          label: shortPointLabel(p.label),
          cx,
          cy,
          lx: alignRight ? cx - 12 : cx + 12,
          ly: cy < padT + 26 ? cy + 16 : cy - 11,
          anchor: (alignRight ? "end" : "start") as "start" | "end",
          isSelected: p.id === selectedPoint?.id,
        };
      });

    // Simple vertical de-confliction so labels don't overlap
    labeledCandidates.sort((a, b) => a.ly - b.ly);
    for (let i = 1; i < labeledCandidates.length; i++) {
      const prev = labeledCandidates[i - 1];
      const cur = labeledCandidates[i];
      if (Math.abs(cur.lx - prev.lx) < 120 && cur.ly - prev.ly < 15) {
        cur.ly = Math.min(padT + plotH - 6, prev.ly + 15);
      }
    }

    return {
      toX,
      toY,
      xTicks,
      yTicks,
      frontierPolyline,
      baseX,
      baseY,
      labeledCandidates,
    };
  }, [data, sortedFront, selectedPoint, plotW, plotH]);

  if (!open) return null;

  const base = data?.baseline;

  return (
    <div
      className="pareto-backdrop"
      role="dialog"
      aria-modal="true"
      aria-label="Multi-Goal Trade-Off Frontier"
      onClick={onClose}
    >
      <div className="pareto-modal" onClick={(e) => e.stopPropagation()}>
        <div className="pareto-header">
          <div className="pareto-header-left">
            <div className="pareto-title-row">
              <h2 className="pareto-title">Multi-Goal Trade-Off Frontier</h2>
              {data && (
                <span className="badge-pill tet">
                  {formatBackendBadge(data.backend)} • {data.evaluations} evaluations
                </span>
              )}
              {loading && <span className="badge-pill">Solving NSGA-II...</span>}
            </div>
            <div className="pareto-subtitle">
              Simultaneously balancing Build Cost ($), Safe Max Accel (mm/s²), and Nozzle Stiffness (µm)
            </div>
          </div>

          <div className="pareto-header-actions">
            {!loading && (
              <button
                type="button"
                className="btn"
                style={{ padding: "5px 10px", fontSize: 12 }}
                onClick={onRerun}
                title="Re-run Pareto multi-objective optimization from current design"
              >
                Re-Run Search
              </button>
            )}
            <button
              type="button"
              className="pareto-close-btn"
              onClick={onClose}
              aria-label="Close modal"
              title="Close (Esc)"
            >
              ×
            </button>
          </div>
        </div>

        {loading && (
          <div className="pareto-state-box">
            <div className="pareto-spinner" />
            <div className="pareto-state-title">Evolving Pareto-Optimal Printer Architectures...</div>
            <div className="pareto-state-sub">
              Evaluating frame extrusions, structural panels, AWD/2WD drives, belts, motors, toolheads &amp; hotends
              across multi-objective generations.
            </div>
          </div>
        )}

        {!loading && error && (
          <div className="pareto-state-box">
            <div className="diag-item bad" style={{ maxWidth: 460 }}>
              <strong>Frontier Search Failed:</strong> {error}
            </div>
            <button type="button" className="btn primary" onClick={onRerun}>
              Retry Frontier Search
            </button>
          </div>
        )}

        {!loading && !error && data && plotModel && base && (
          <div className="pareto-body">
            {/* Left Side: Interactive SVG Scatter Plot */}
            <div className="pareto-plot-pane">
              <div className="pareto-plot-header">
                <span>Cost vs. Safe Max Acceleration</span>
                <span className="pareto-plot-meta">
                  {sortedFront.length} Pareto builds • Pop {data.population_size} × {data.generations} gen
                </span>
              </div>

              <div className="pareto-svg-wrap">
                <svg viewBox={`0 0 ${W} ${H}`} className="pareto-svg">
                  <defs>
                    <radialGradient id="paretoSelectedHalo" cx="50%" cy="50%" r="50%">
                      <stop offset="0%" stopColor="#38bdf8" stopOpacity="0.45" />
                      <stop offset="100%" stopColor="#38bdf8" stopOpacity="0" />
                    </radialGradient>
                  </defs>

                  <rect
                    x={padL}
                    y={padT}
                    width={plotW}
                    height={plotH}
                    rx="6"
                    fill="#0b1019"
                    stroke="#243045"
                    strokeWidth="1"
                  />

                  {/* Horizontal Gridlines & Y ticks */}
                  {plotModel.yTicks.map((t, i) => (
                    <g key={`y-${i}`}>
                      <line
                        x1={padL}
                        y1={t.y}
                        x2={padL + plotW}
                        y2={t.y}
                        stroke="#182233"
                        strokeDasharray={i === 0 ? undefined : "3 3"}
                      />
                      <text
                        x={padL - 8}
                        y={t.y + 3.5}
                        fill="#8b9bb4"
                        fontSize="10"
                        textAnchor="end"
                        style={{ fontVariantNumeric: "tabular-nums" }}
                      >
                        {t.val >= 1000 ? `${(t.val / 1000).toFixed(t.val >= 10000 ? 0 : 1)}k` : Math.round(t.val)}
                      </text>
                    </g>
                  ))}

                  {/* Vertical Gridlines & X ticks */}
                  {plotModel.xTicks.map((t, i) => (
                    <g key={`x-${i}`}>
                      <line
                        x1={t.x}
                        y1={padT}
                        x2={t.x}
                        y2={padT + plotH}
                        stroke="#182233"
                        strokeDasharray={i === 0 ? undefined : "3 3"}
                      />
                      <text
                        x={t.x}
                        y={padT + plotH + 16}
                        fill="#8b9bb4"
                        fontSize="10"
                        textAnchor="middle"
                        style={{ fontVariantNumeric: "tabular-nums" }}
                      >
                        ${Math.round(t.val).toLocaleString()}
                      </text>
                    </g>
                  ))}

                  {/* Axis Titles */}
                  <text
                    x={padL + plotW / 2}
                    y={H - 10}
                    fill="#9fb0c9"
                    fontSize="11"
                    fontWeight="600"
                    textAnchor="middle"
                  >
                    Estimated Build Cost ($)
                  </text>
                  <text
                    x={16}
                    y={padT + plotH / 2}
                    fill="#9fb0c9"
                    fontSize="11"
                    fontWeight="600"
                    textAnchor="middle"
                    transform={`rotate(-90 16 ${padT + plotH / 2})`}
                  >
                    Safe Max Accel (mm/s²)
                  </text>

                  {/* Pareto Frontier Dashed Curve */}
                  {sortedFront.length > 1 && (
                    <polyline
                      fill="none"
                      stroke="#38bdf8"
                      strokeOpacity="0.45"
                      strokeWidth="2"
                      strokeDasharray="5 4"
                      points={plotModel.frontierPolyline}
                    />
                  )}

                  {/* Baseline Amber Diamond Marker */}
                  <g className="pareto-baseline-marker">
                    <polygon
                      points={`${plotModel.baseX},${plotModel.baseY - 9} ${plotModel.baseX + 9},${plotModel.baseY} ${plotModel.baseX},${plotModel.baseY + 9} ${plotModel.baseX - 9},${plotModel.baseY}`}
                      fill="#f59e0b"
                      stroke="#0c1017"
                      strokeWidth="2"
                    />
                    <text
                      x={plotModel.baseX > padL + plotW * 0.65 ? plotModel.baseX - 13 : plotModel.baseX + 13}
                      y={plotModel.baseY + 4}
                      fill="#fcd34d"
                      fontSize="10"
                      fontWeight="700"
                      textAnchor={plotModel.baseX > padL + plotW * 0.65 ? "end" : "start"}
                    >
                      Current Build (${Math.round(base.total_cost_usd).toLocaleString()})
                    </text>
                    <title>
                      Current Build (Baseline): ${Math.round(base.total_cost_usd)} •{" "}
                      {Math.round(base.safe_max_accel_mm_s2).toLocaleString()} mm/s² •{" "}
                      {base.nozzle_deflection_um.toFixed(1)} µm flex
                    </title>
                  </g>

                  {/* Pareto Frontier Points */}
                  {sortedFront.map((pt) => {
                    const cx = plotModel.toX(pt.total_cost_usd);
                    const cy = plotModel.toY(pt.safe_max_accel_mm_s2);
                    const isSelected = pt.id === selectedPoint?.id;
                    const isHovered = pt.id === hoveredId;
                    const dotColor = scoreDotColor(pt.composite_score);

                    return (
                      <g
                        key={pt.id}
                        style={{ cursor: "pointer" }}
                        onClick={() => setSelectedId(pt.id)}
                        onMouseEnter={() => setHoveredId(pt.id)}
                        onMouseLeave={() => setHoveredId((prev) => (prev === pt.id ? null : prev))}
                      >
                        {isSelected && (
                          <>
                            <circle cx={cx} cy={cy} r={18} fill="url(#paretoSelectedHalo)" />
                            <circle
                              cx={cx}
                              cy={cy}
                              r={11.5}
                              fill="none"
                              stroke="#38bdf8"
                              strokeWidth="2"
                            />
                          </>
                        )}
                        <circle
                          cx={cx}
                          cy={cy}
                          r={isSelected ? 7.5 : isHovered ? 7.5 : 6}
                          fill={dotColor}
                          stroke={isSelected ? "#ffffff" : "#0c1017"}
                          strokeWidth={isSelected ? 2 : 1.5}
                        />
                        <title>
                          {pt.label}: ${Math.round(pt.total_cost_usd)} •{" "}
                          {Math.round(pt.safe_max_accel_mm_s2).toLocaleString()} mm/s² •{" "}
                          {pt.nozzle_deflection_um.toFixed(1)} µm flex • Score {pt.composite_score.toFixed(1)}
                        </title>
                      </g>
                    );
                  })}

                  {/* Standout & Selected Callout Labels */}
                  {plotModel.labeledCandidates.map((lbl) => (
                    <g key={`lbl-${lbl.id}`} style={{ pointerEvents: "none" }}>
                      <line
                        x1={lbl.cx}
                        y1={lbl.cy}
                        x2={lbl.lx + (lbl.anchor === "end" ? 3 : -3)}
                        y2={lbl.ly - 3}
                        stroke={lbl.isSelected ? "#38bdf8" : "rgba(148, 163, 184, 0.45)"}
                        strokeWidth="1"
                      />
                      <text
                        x={lbl.lx}
                        y={lbl.ly}
                        fill={lbl.isSelected ? "#e0f2fe" : "#cbd5e1"}
                        fontSize={lbl.isSelected ? "10.5" : "9.5"}
                        fontWeight={lbl.isSelected ? "700" : "600"}
                        textAnchor={lbl.anchor}
                      >
                        {lbl.label}
                      </text>
                    </g>
                  ))}
                </svg>
              </div>

              <div className="pareto-legend-bar">
                <span className="pareto-legend-item">
                  <span className="pareto-legend-diamond" /> Current Build
                </span>
                <span className="pareto-legend-item">
                  <span className="pareto-legend-dot" style={{ background: "#22c55e" }} /> Score ≥86
                </span>
                <span className="pareto-legend-item">
                  <span className="pareto-legend-dot" style={{ background: "#38bdf8" }} /> 66–85
                </span>
                <span className="pareto-legend-item">
                  <span className="pareto-legend-dot" style={{ background: "#f59e0b" }} /> &lt;66
                </span>
                <span className="pareto-legend-hint">Click any dot to inspect &amp; apply</span>
              </div>
            </div>

            {/* Right Side: Selected Build Inspector + Scrollable Pareto List */}
            <div className="pareto-side-pane">
              {selectedPoint && (() => {
                const costD = formatDeltaVal(
                  selectedPoint.total_cost_usd,
                  base.total_cost_usd,
                  false,
                  0.5,
                  (d) => `${d > 0 ? "+" : "-"}$${Math.round(Math.abs(d)).toLocaleString()}`
                );
                const accelD = formatDeltaVal(
                  selectedPoint.safe_max_accel_mm_s2,
                  base.safe_max_accel_mm_s2,
                  true,
                  25,
                  (d) => `${d > 0 ? "+" : ""}${Math.round(d).toLocaleString()}`
                );
                const flexD = formatDeltaVal(
                  selectedPoint.nozzle_deflection_um,
                  base.nozzle_deflection_um,
                  false,
                  0.1,
                  (d) => `${d > 0 ? "+" : ""}${d.toFixed(1)} µm`
                );
                const freqD = formatDeltaVal(
                  selectedPoint.fundamental_freq_hz,
                  base.fundamental_freq_hz,
                  true,
                  0.1,
                  (d) => `${d > 0 ? "+" : ""}${d.toFixed(1)} Hz`
                );
                const flowD = formatDeltaVal(
                  selectedPoint.max_flow_mm3_s,
                  base.max_flow_mm3_s,
                  true,
                  0.1,
                  (d) => `${d > 0 ? "+" : ""}${d.toFixed(1)}`
                );
                const massD = formatDeltaVal(
                  selectedPoint.total_mass_kg,
                  base.total_mass_kg,
                  false,
                  0.05,
                  (d) => `${d > 0 ? "+" : ""}${d.toFixed(2)} kg`
                );
                const scoreD = formatDeltaVal(
                  selectedPoint.composite_score,
                  base.composite_score,
                  true,
                  0.05,
                  (d) => `${d > 0 ? "+" : ""}${d.toFixed(1)}`
                );

                return (
                  <div className="pareto-inspector">
                    <div className="pareto-inspector-head">
                      <div>
                        <div className="pareto-inspector-eyebrow">Selected Pareto Build</div>
                        <div className="pareto-inspector-title">{selectedPoint.label}</div>
                      </div>
                      <div className="pareto-score-badge">
                        <span
                          className="pareto-score-num"
                          style={{ color: scoreDotColor(selectedPoint.composite_score) }}
                        >
                          {selectedPoint.composite_score.toFixed(1)}
                        </span>
                        {scoreD.cls !== "same" && (
                          <span className={`delta-inline ${scoreD.cls}`}>{scoreD.text}</span>
                        )}
                      </div>
                    </div>

                    <div className="pareto-kpi-grid">
                      <div className="pareto-kpi-tile">
                        <div className="pareto-kpi-top">
                          <span>Build Cost</span>
                          <span className={`impact-badge ${costD.cls}`}>
                            {costD.cls === "same" ? "—" : costD.text}
                          </span>
                        </div>
                        <div className="pareto-kpi-val">
                          ${Math.round(selectedPoint.total_cost_usd).toLocaleString()}
                        </div>
                      </div>

                      <div className="pareto-kpi-tile">
                        <div className="pareto-kpi-top">
                          <span>Safe Max Accel</span>
                          <span className={`impact-badge ${accelD.cls}`}>
                            {accelD.cls === "same" ? "—" : accelD.text}
                          </span>
                        </div>
                        <div className="pareto-kpi-val">
                          {Math.round(selectedPoint.safe_max_accel_mm_s2).toLocaleString()} mm/s²
                        </div>
                      </div>

                      <div className="pareto-kpi-tile">
                        <div className="pareto-kpi-top">
                          <span>Nozzle Flex</span>
                          <span className={`impact-badge ${flexD.cls}`}>
                            {flexD.cls === "same" ? "—" : flexD.text}
                          </span>
                        </div>
                        <div className="pareto-kpi-val">
                          {selectedPoint.nozzle_deflection_um.toFixed(1)} µm
                        </div>
                      </div>

                      <div className="pareto-kpi-tile">
                        <div className="pareto-kpi-top">
                          <span>Resonance f₁</span>
                          <span className={`impact-badge ${freqD.cls}`}>
                            {freqD.cls === "same" ? "—" : freqD.text}
                          </span>
                        </div>
                        <div className="pareto-kpi-val">
                          {selectedPoint.fundamental_freq_hz.toFixed(1)} Hz
                        </div>
                      </div>

                      <div className="pareto-kpi-tile">
                        <div className="pareto-kpi-top">
                          <span>Max Flow</span>
                          <span className={`impact-badge ${flowD.cls}`}>
                            {flowD.cls === "same" ? "—" : flowD.text}
                          </span>
                        </div>
                        <div className="pareto-kpi-val">
                          {selectedPoint.max_flow_mm3_s.toFixed(1)} mm³/s
                        </div>
                      </div>

                      <div className="pareto-kpi-tile">
                        <div className="pareto-kpi-top">
                          <span>Total Mass</span>
                          <span className={`impact-badge ${massD.cls}`}>
                            {massD.cls === "same" ? "—" : massD.text}
                          </span>
                        </div>
                        <div className="pareto-kpi-val">
                          {selectedPoint.total_mass_kg.toFixed(2)} kg
                        </div>
                      </div>
                    </div>

                    <div className="pareto-changes-block">
                      <div className="pareto-changes-label">Upgrades vs. Current Build</div>
                      {selectedPoint.changes.length > 0 ? (
                        <div className="pareto-chips">
                          {selectedPoint.changes.map((chg, idx) => (
                            <span key={idx} className="pareto-change-chip">
                              {chg}
                            </span>
                          ))}
                        </div>
                      ) : (
                        <div className="pareto-changes-empty">Matches baseline hardware configuration</div>
                      )}
                    </div>

                    <button
                      type="button"
                      className="btn pareto-apply-btn"
                      disabled={!selectedPoint.spec}
                      onClick={() => onApply(selectedPoint)}
                    >
                      Apply This Build to Studio
                    </button>
                  </div>
                );
              })()}

              <div className="pareto-list-section">
                <div className="pareto-list-head">
                  <span>All Frontier Builds ({sortedFront.length})</span>
                  <span>Cost • Accel • Flex</span>
                </div>
                <div className="pareto-list-scroll">
                  {sortedFront.map((pt) => {
                    const active = pt.id === selectedPoint?.id;
                    return (
                      <button
                        type="button"
                        key={pt.id}
                        className={`pareto-list-row ${active ? "active" : ""}`}
 aria-pressed={active}
                        onClick={() => setSelectedId(pt.id)}
                        onMouseEnter={() => setHoveredId(pt.id)}
                        onMouseLeave={() => setHoveredId((prev) => (prev === pt.id ? null : prev))}
                      >
                        <div className="pareto-list-row-left">
                          <span
                            className="pareto-legend-dot"
                            style={{ background: scoreDotColor(pt.composite_score) }}
                          />
                          <div className="pareto-list-row-text">
                            <span className="pareto-list-label">{pt.label}</span>
                            <span className="pareto-list-sub">
                              Score {pt.composite_score.toFixed(1)} • {pt.changes.length} change
                              {pt.changes.length === 1 ? "" : "s"}
                            </span>
                          </div>
                        </div>
                        <div className="pareto-list-row-metrics">
                          <strong>${Math.round(pt.total_cost_usd).toLocaleString()}</strong>
                          <span>
                            {(pt.safe_max_accel_mm_s2 / 1000).toFixed(1)}k • {pt.nozzle_deflection_um.toFixed(0)}µm
                          </span>
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
