import React, { useEffect, useMemo, useState } from "react";
import { onTabListKeyDown } from "../utils/a11y";
import { AdxlCalibrationPanel } from "./AdxlCalibrationPanel";
import { AdjointSensitivityContent, CncMetrologyContent } from "./CncMetrologyPanel";
import { KlipperShaperPsdCard } from "./KlipperShaperCard";
import { prettifyDiagnostic, splitBomName } from "../utils/displayText";
import {
  AccelBottleneckData,
  AccelBottleneckShares,
  BedPlateFeaData,
  CompareMode,
  CustomBracketOverrideData,
  DeepVerifyResponse,
  LimitingModalSummaryData,
  ModeShapeData,
  OverallLimitingCauseItem,
  PrinterBOMResponse,
  PrinterSpecData,
  ShaperData,
  SimulationResponse,
  SolidFeaEngine,
  SolidFEAResponse,
  SubsystemKey,
  TetFEAResponse,
  ThermalStructuralReport,
  ToolheadFlexBreakdownData,
  ToolheadStructureGeometryData,
  TopologyOptResponse,
  ViewMode,
  WeightShaperGainsSummary,
} from "../types";

interface Props {
  sim: SimulationResponse | null;
  deepVerifyResult: DeepVerifyResponse | null;
  viewingDeepSim: boolean;
  onToggleViewingDeepSim: (useDeep: boolean) => void;
  onInspectDeepFea: () => void;
  onDismissDeepVerify: () => void;
  baselineSim: SimulationResponse | null;
  baselineBom: PrinterBOMResponse | null;
  baselineLabel: string;
  prevSim: SimulationResponse | null;
  prevBom: PrinterBOMResponse | null;
  lastChangeSummary: string | null;
  compareMode: CompareMode;
  onCompareModeChange: (m: CompareMode) => void;
  canUndo: boolean;
  onUndo: () => void;
  onResetToBaseline: () => void;
  onSetBaseline: () => void;
  solidFea: SolidFEAResponse | null;
  tetFea: TetFEAResponse | null;
  activeSolidEngine: SolidFeaEngine;
  topoResult: TopologyOptResponse | null;
  viewMode: ViewMode;
  onViewModeChange: (m: ViewMode) => void;
  onInspectModalMode?: (modeIdx: number) => void;
  bom: PrinterBOMResponse | null;
  bomLoading: boolean;
  stockLengthMm: number;
  onStockLengthChange: (lenMm: number) => void;
  onDownloadBomCsv: () => void;
  showBedWarp?: boolean;
  onInspectBedWarp?: () => void;
  toolheadPunchIn?: boolean;
  onToggleToolheadPunchIn?: () => void;
  onApplyFittedSpec?: (fittedSpec: PrinterSpecData, changeLabel?: string) => void;
  onInjectCustomBracket?: (override: CustomBracketOverrideData) => void;
  error: string | null;
}

function formatBackendLabel(backend: string): string {
  const low = backend.toLowerCase();
  if (low.includes("gmsh") && (low.includes("skfem") || low.includes("scikit"))) {
    return "Gmsh + scikit-fem";
  }
  return backend;
}

function formatElementLabel(elementType: string): string {
  const low = elementType.toLowerCase();
  if (low === "tet10" || low === "p2" || low === "tet10_p2") return "Tet10 (P2)";
  if (low === "tet4" || low === "p1" || low === "tet4_p1") return "Tet4 (P1)";
  return elementType;
}

function formatHzGain(hz: number): string {
  return hz >= 0.1 ? hz.toFixed(1) : hz.toFixed(2);
}

function formatCompactAccel(v: number): string {
  if (!Number.isFinite(v) || v <= 0) return "—";
  if (v >= 10000) return `${Math.round(v / 1000)}k`;
  if (v >= 1000) return `${(v / 1000).toFixed(1).replace(/\.0$/, "")}k`;
  return `${Math.round(v)}`;
}

function modeSeverityMeta(severity?: "critical" | "moderate" | "stiff"): {
  label: string;
  cls: string;
} | null {
  if (severity === "critical") return { label: "Limiting", cls: "bad" };
  if (severity === "moderate") return { label: "Moderate", cls: "warn" };
  if (severity === "stiff") return { label: "Stiff", cls: "ok" };
  return null;
}

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

function impactLevelCls(level?: "critical" | "high" | "moderate" | "low", score?: number): string {
  const eff =
    level ??
    (score != null
      ? score >= 75
        ? "critical"
        : score >= 50
          ? "high"
          : score >= 25
            ? "moderate"
            : "low"
      : "low");
  if (eff === "critical") return "bad";
  if (eff === "high" || eff === "moderate") return "warn";
  return "ok";
}

const CATEGORY_ORDER = ["frame", "motion", "motors", "hotend", "bed", "enclosure", "fasteners"];

const CATEGORY_COLORS: Record<string, string> = {
  frame: "#38bdf8",
  motion: "#22c55e",
  motors: "#a855f7",
  hotend: "#f97316",
  bed: "#eab308",
  enclosure: "#06b6d4",
  fasteners: "#94a3b8",
};

const NESTING_PALETTE = ["#38bdf8", "#22c55e", "#f59e0b", "#a855f7", "#2dd4bf", "#fb7185"];

function scoreColor(v: number): string {
  if (v >= 80) return "var(--green)";
  if (v >= 60) return "var(--accent)";
  if (v >= 45) return "var(--amber)";
  return "var(--red)";
}

type DeltaStatus = "better" | "worse" | "same";

interface ImpactMetricRow {
  key: string;
  label: string;
  oldVal: number;
  newVal: number;
  delta: number;
  pct: number;
  status: DeltaStatus;
  oldText: string;
  newText: string;
  deltaText: string;
}

function evalDelta(
  oldVal: number,
  newVal: number,
  higherIsBetter: boolean,
  eps: number
): { delta: number; pct: number; status: DeltaStatus } {
  const delta = newVal - oldVal;
  const pct = Math.abs(oldVal) > 1e-6 ? (delta / Math.abs(oldVal)) * 100 : 0;
  if (Math.abs(delta) < eps) {
    return { delta: 0, pct: 0, status: "same" };
  }
  const improved = higherIsBetter ? delta > 0 : delta < 0;
  return { delta, pct, status: improved ? "better" : "worse" };
}

function buildImpactMetrics(
  cur: SimulationResponse,
  cmp: SimulationResponse,
  curBom: PrinterBOMResponse | null,
  cmpBom: PrinterBOMResponse | null
): ImpactMetricRow[] {
  const curCost =
    curBom?.total_cost_usd ??
    cur.bom_total_cost_usd ??
    cur.geometry.mass_summary.estimated_material_cost_usd;
  const cmpCost =
    cmpBom?.total_cost_usd ??
    cmp.bom_total_cost_usd ??
    cmp.geometry.mass_summary.estimated_material_cost_usd;
  const curInertialFlex = Math.max(
    cur.static_fea.accel_deflection_x_um,
    cur.static_fea.accel_deflection_y_um
  );
  const cmpInertialFlex = Math.max(
    cmp.static_fea.accel_deflection_x_um,
    cmp.static_fea.accel_deflection_y_um
  );

  const defs: Array<{
    key: string;
    label: string;
    oldVal: number;
    newVal: number;
    higherIsBetter: boolean;
    eps: number;
    fmt: (v: number) => string;
    fmtDelta: (d: number) => string;
  }> = [
    {
      key: "score",
      label: "Overall Score",
      oldVal: cmp.scores.composite,
      newVal: cur.scores.composite,
      higherIsBetter: true,
      eps: 0.05,
      fmt: (v) => v.toFixed(1),
      fmtDelta: (d) => `${d > 0 ? "+" : ""}${d.toFixed(1)}`,
    },
    {
      key: "accel",
      label: "Achievable Max Accel",
      oldVal: cmp.motion_limits?.achievable_max_accel_mm_s2 ?? cmp.modal_fea.recommended_max_accel_mm_s2,
      newVal: cur.motion_limits?.achievable_max_accel_mm_s2 ?? cur.modal_fea.recommended_max_accel_mm_s2,
      higherIsBetter: true,
      eps: 15,
      fmt: (v) => `${Math.round(v).toLocaleString()} mm/s²`,
      fmtDelta: (d) => `${d > 0 ? "+" : ""}${Math.round(d).toLocaleString()}`,
    },
    {
      key: "freq",
      label: "Resonance (f₁)",
      oldVal: cmp.modal_fea.fundamental_freq_hz,
      newVal: cur.modal_fea.fundamental_freq_hz,
      higherIsBetter: true,
      eps: 0.05,
      fmt: (v) => `${v.toFixed(1)} Hz`,
      fmtDelta: (d) => `${d > 0 ? "+" : ""}${d.toFixed(1)} Hz`,
    },
    {
      key: "flex",
      label: "Inertial Flex",
      oldVal: cmpInertialFlex,
      newVal: curInertialFlex,
      higherIsBetter: false,
      eps: 0.05,
      fmt: (v) => `${v.toFixed(1)} µm`,
      fmtDelta: (d) => `${d > 0 ? "+" : ""}${d.toFixed(1)} µm`,
    },
    {
      key: "flow",
      label: "Max Flow",
      oldVal: cmp.thermal.max_volumetric_flow_mm3_s,
      newVal: cur.thermal.max_volumetric_flow_mm3_s,
      higherIsBetter: true,
      eps: 0.05,
      fmt: (v) => `${v.toFixed(1)} mm³/s`,
      fmtDelta: (d) => `${d > 0 ? "+" : ""}${d.toFixed(1)}`,
    },
    {
      key: "time",
      label: "Print Time",
      oldVal: cmp.virtual_print.total_time_s,
      newVal: cur.virtual_print.total_time_s,
      higherIsBetter: false,
      eps: 0.05,
      fmt: (v) => `${v.toFixed(1)} s`,
      fmtDelta: (d) => `${d > 0 ? "+" : ""}${d.toFixed(1)} s`,
    },
    {
      key: "cost",
      label: "Build Cost",
      oldVal: cmpCost,
      newVal: curCost,
      higherIsBetter: false,
      eps: 0.5,
      fmt: (v) => `$${Math.round(v).toLocaleString()}`,
      fmtDelta: (d) => `${d > 0 ? "+" : "-"}$${Math.round(Math.abs(d)).toLocaleString()}`,
    },
    {
      key: "mass",
      label: "Total Mass",
      oldVal: cmp.geometry.mass_summary.total_mass_kg,
      newVal: cur.geometry.mass_summary.total_mass_kg,
      higherIsBetter: false,
      eps: 0.01,
      fmt: (v) => `${v.toFixed(2)} kg`,
      fmtDelta: (d) => `${d > 0 ? "+" : ""}${d.toFixed(2)} kg`,
    },
  ];

  return defs.map((def) => {
    const res = evalDelta(def.oldVal, def.newVal, def.higherIsBetter, def.eps);
    return {
      key: def.key,
      label: def.label,
      oldVal: def.oldVal,
      newVal: def.newVal,
      delta: res.delta,
      pct: res.pct,
      status: res.status,
      oldText: def.fmt(def.oldVal),
      newText: def.fmt(def.newVal),
      deltaText: res.status === "same" ? "0" : def.fmtDelta(res.delta),
    };
  });
}

function buildPlainEnglishSummary(metrics: ImpactMetricRow[]): string {
  const byKey = Object.fromEntries(metrics.map((m) => [m.key, m])) as Record<string, ImpactMetricRow>;

  const gains: Array<{ weight: number; text: string }> = [];
  const tradeoffs: Array<{ weight: number; text: string }> = [];

  const pctTag = (m: ImpactMetricRow) =>
    Math.abs(m.pct) >= 1 ? ` (${m.pct > 0 ? "+" : ""}${Math.round(m.pct)}%)` : "";

  const accel = byKey.accel;
  if (accel.status === "better") {
    gains.push({
      weight: Math.abs(accel.pct) + 12,
      text: `raised safe accel by +${Math.round(accel.delta).toLocaleString()} mm/s²${pctTag(accel)}`,
    });
  } else if (accel.status === "worse") {
    tradeoffs.push({
      weight: Math.abs(accel.pct) + 12,
      text: `lowered safe accel by ${Math.round(accel.delta).toLocaleString()} mm/s²${pctTag(accel)}`,
    });
  }

  const freq = byKey.freq;
  if (freq.status === "better") {
    gains.push({
      weight: Math.abs(freq.pct) + 10,
      text: `boosted resonance by +${freq.delta.toFixed(1)} Hz${pctTag(freq)}`,
    });
  } else if (freq.status === "worse") {
    tradeoffs.push({
      weight: Math.abs(freq.pct) + 10,
      text: `dropped resonance by ${freq.delta.toFixed(1)} Hz${pctTag(freq)}`,
    });
  }

  const flex = byKey.flex;
  if (flex.status === "better") {
    gains.push({
      weight: Math.abs(flex.pct) + 11,
      text: `cut frame flex by ${Math.round(flex.pct)}%`,
    });
  } else if (flex.status === "worse") {
    tradeoffs.push({
      weight: Math.abs(flex.pct) + 11,
      text: `increased frame flex by +${Math.round(flex.pct)}%`,
    });
  }

  const flow = byKey.flow;
  if (flow.status === "better") {
    gains.push({
      weight: Math.abs(flow.pct) + 11,
      text: `increased max flow by +${flow.delta.toFixed(1)} mm³/s${pctTag(flow)}`,
    });
  } else if (flow.status === "worse") {
    tradeoffs.push({
      weight: Math.abs(flow.pct) + 11,
      text: `reduced max flow by ${flow.delta.toFixed(1)} mm³/s${pctTag(flow)}`,
    });
  }

  const time = byKey.time;
  if (time.status === "better") {
    gains.push({
      weight: Math.abs(time.pct) + 8,
      text: `cut print time by ${time.delta.toFixed(1)} s${pctTag(time)}`,
    });
  } else if (time.status === "worse") {
    tradeoffs.push({
      weight: Math.abs(time.pct) + 8,
      text: `slowed print time by +${time.delta.toFixed(1)} s`,
    });
  }

  const cost = byKey.cost;
  if (cost.status === "better") {
    gains.push({
      weight: Math.abs(cost.pct) + 5,
      text: `saved $${Math.round(Math.abs(cost.delta)).toLocaleString()} in build cost`,
    });
  } else if (cost.status === "worse") {
    tradeoffs.push({
      weight: Math.abs(cost.pct) + 7,
      text: `adding +$${Math.round(cost.delta).toLocaleString()} to build cost`,
    });
  }

  const mass = byKey.mass;
  if (mass.status === "better") {
    gains.push({
      weight: Math.abs(mass.pct) + 4,
      text: `trimmed ${Math.abs(mass.delta).toFixed(2)} kg total mass`,
    });
  } else if (mass.status === "worse") {
    tradeoffs.push({
      weight: Math.abs(mass.pct) + 6,
      text: `adding +${mass.delta.toFixed(2)} kg mass`,
    });
  }

  const score = byKey.score;
  if (gains.length === 0 && tradeoffs.length === 0) {
    if (score.status === "better") {
      return `Improved overall engineering score by +${score.delta.toFixed(1)} pts.`;
    }
    if (score.status === "worse") {
      return `Lowered overall engineering score by ${score.delta.toFixed(1)} pts.`;
    }
    return "No measurable change to performance.";
  }

  gains.sort((a, b) => b.weight - a.weight);
  tradeoffs.sort((a, b) => b.weight - a.weight);

  const topGains = gains.slice(0, 2).map((g) => g.text);
  const topTrades = tradeoffs.slice(0, 2).map((t) => t.text);

  const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);
  const joinAnd = (arr: string[]) => (arr.length <= 1 ? arr[0] ?? "" : `${arr[0]} and ${arr[1]}`);

  if (topGains.length > 0 && topTrades.length > 0) {
    return `${cap(joinAnd(topGains))}, while ${joinAnd(topTrades)}.`;
  }
  if (topGains.length > 0) {
    return `${cap(joinAnd(topGains))} with no measurable downsides.`;
  }
  // Only tradeoffs
  const normalizedTrades = topTrades.map((t) =>
    t.startsWith("adding ") ? t.replace("adding ", "added ") : t
  );
  return `${cap(joinAnd(normalizedTrades))}.`;
}

const InlineDeltaBadge: React.FC<{
  oldVal: number;
  newVal: number;
  higherIsBetter: boolean;
  eps?: number;
  formatDelta: (d: number) => string;
}> = ({ oldVal, newVal, higherIsBetter, eps = 0.05, formatDelta }) => {
  const res = evalDelta(oldVal, newVal, higherIsBetter, eps);
  if (res.status === "same") return null;
  return (
    <span className={`delta-inline ${res.status}`}>
      {formatDelta(res.delta)}
    </span>
  );
};

const DeepDiveAccordion: React.FC<{
  title: string;
  badge?: string;
  updated?: boolean;
  open: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}> = ({ title, badge, updated, open, onToggle, children }) => (
  <div
    className={`section-card deep-accordion ${open ? "open" : ""} ${updated ? "card-changed" : ""}`}
  >
    <button
      type="button"
      className="deep-accordion-head"
      onClick={onToggle}
      aria-expanded={open}
    >
      <span
        style={{
          display: "inline-flex",
          alignItems: "center",
          gap: 6,
          minWidth: 0,
          flex: "1 1 auto",
          textAlign: "left",
        }}
      >
        <span
          aria-hidden="true"
          style={{
            fontSize: 11,
            color: "var(--muted)",
            width: 10,
            flexShrink: 0,
          }}
        >
          {open ? "▾" : "▸"}
        </span>
        <span
          style={{
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
        >
          {title}
        </span>
        {updated && <span className="changed-dot-label">Updated</span>}
      </span>
      {badge && <span className="deep-accordion-badge">{badge}</span>}
    </button>
    {open && <div className="deep-accordion-body">{children}</div>}
  </div>
);

const TopologyOptCard: React.FC<{ topo: TopologyOptResponse }> = ({ topo }) => {
  const hist = topo.history || [];
  const w = 276;
  const h = 88;
  const padL = 8;
  const padR = 8;
  const padT = 18;
  const padB = 16;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  let compPoints = "";
  let massPoints = "";
  if (hist.length > 0) {
    const comps = hist.map((p) => p.compliance_mj);
    const masses = hist.map((p) => p.mass_g);
    const minC = Math.min(...comps);
    const maxC = Math.max(...comps);
    const spanC = Math.max(maxC - minC, 1e-6);
    const minM = Math.min(...masses, 0);
    const maxM = Math.max(...masses, topo.baseline_mass_g, 1);
    const spanM = Math.max(maxM - minM, 1e-6);
    const denom = Math.max(hist.length - 1, 1);

    compPoints = hist
      .map((p, idx) => {
        const x = padL + (idx / denom) * plotW;
        const y = padT + (1 - (p.compliance_mj - minC) / spanC) * plotH;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");

    massPoints = hist
      .map((p, idx) => {
        const x = padL + (idx / denom) * plotW;
        const y = padT + (1 - (p.mass_g - minM) / spanM) * plotH;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
  }

  return (
    <div className="section-card" style={{ borderColor: "rgba(168, 85, 247, 0.5)" }}>
      <div className="section-title">
        <span>3D SIMP Topology Opt</span>
        <span style={{ color: "var(--purple)", fontSize: 11 }}>
          {topo.part_name} ({topo.is_fdm_anisotropic ? "FDM" : "ISO"})
        </span>
      </div>

      <div className="kpi-grid">
        <div className="kpi-box">
          <span className="kpi-label">Mass Saved</span>
          <span className="kpi-value tag-ok">-{topo.mass_reduction_pct.toFixed(1)}%</span>
          <span className="kpi-sub">
            {topo.baseline_mass_g.toFixed(1)}g → {topo.optimized_mass_g.toFixed(1)}g
          </span>
        </div>
        <div className="kpi-box">
          <span className="kpi-label">Specific Stiffness</span>
          <span className={`kpi-value ${topo.specific_stiffness_gain_pct >= 0 ? "tag-ok" : "tag-warn"}`}>
            {topo.specific_stiffness_gain_pct >= 0 ? "+" : ""}
            {topo.specific_stiffness_gain_pct.toFixed(1)}%
          </span>
          <span className="kpi-sub">
            {topo.baseline_stiffness_n_per_mm.toFixed(0)} → {topo.optimized_stiffness_n_per_mm.toFixed(0)} N/mm
          </span>
        </div>
        <div className="kpi-box">
          <span className="kpi-label">Safety Factor</span>
          <span className={`kpi-value ${topo.safety_factor < 1.5 ? "tag-bad" : "tag-ok"}`}>
            {topo.safety_factor.toFixed(2)}×
          </span>
          <span className="kpi-sub">{topo.failure_mode.replace(/_/g, " ")}</span>
        </div>
        <div className="kpi-box">
          <span className="kpi-label">Iterations</span>
          <span className="kpi-value" style={{ color: "var(--accent)" }}>
            {topo.iterations_run}
          </span>
          <span className="kpi-sub">
            Vol {(topo.achieved_volume_fraction * 100).toFixed(0)}% / {(topo.target_volume_fraction * 100).toFixed(0)}%
          </span>
        </div>
      </div>

      <div className="metric-kv">
        <span>Voxels Kept / Total</span>
        <span className="metric-val">
          {topo.optimized_elements} / {topo.baseline_elements} ({topo.voxel_size_mm.toFixed(1)} mm)
        </span>
      </div>
      <div className="metric-kv">
        <span>Peak Stress / Max Disp</span>
        <span className="metric-val">
          {topo.max_von_mises_mpa.toFixed(2)} MPa • {topo.max_displacement_um.toFixed(1)} µm
        </span>
      </div>
      <div className="metric-kv">
        <span>Compliance (Base → Opt)</span>
        <span className="metric-val">
          {topo.baseline_compliance_mj.toFixed(2)} → {topo.optimized_compliance_mj.toFixed(2)} mJ
        </span>
      </div>

      {hist.length > 0 && (
        <svg viewBox={`0 0 ${w} ${h}`} style={{ width: "100%", height: 90, marginTop: 2 }}>
          <rect x="0" y="0" width={w} height={h} rx="5" fill="#0d131d" stroke="#273247" strokeWidth="1" />
          <line x1={padL} y1={padT + plotH * 0.5} x2={w - padR} y2={padT + plotH * 0.5} stroke="#1e293b" strokeDasharray="2 2" />
          <line x1={padL} y1={padT + plotH} x2={w - padR} y2={padT + plotH} stroke="#1e293b" />
          {hist.length > 1 ? (
            <>
              <polyline fill="none" stroke="#38bdf8" strokeWidth="2" points={compPoints} />
              <polyline fill="none" stroke="#22c55e" strokeWidth="2" strokeDasharray="4 2" points={massPoints} />
            </>
          ) : (
            <circle cx={padL + plotW * 0.5} cy={padT + plotH * 0.5} r="3" fill="#38bdf8" />
          )}
          <text x="8" y="12" fill="#8b9bb4" fontSize="8.5" fontWeight="600">
            SIMP Convergence (Iter 1–{hist[hist.length - 1]?.iteration ?? topo.iterations_run})
          </text>
          <line x1="158" y1="9" x2="170" y2="9" stroke="#38bdf8" strokeWidth="2" />
          <text x="173" y="12" fill="#38bdf8" fontSize="8">
            Compliance
          </text>
          <line x1="226" y1="9" x2="238" y2="9" stroke="#22c55e" strokeWidth="2" strokeDasharray="3 2" />
          <text x="241" y="12" fill="#22c55e" fontSize="8">
            Mass
          </text>
          <text x="8" y={h - 4} fill="#64748b" fontSize="7.5">
            Iter 1
          </text>
          <text x={w - 8} y={h - 4} fill="#64748b" fontSize="7.5" textAnchor="end">
            Iter {hist[hist.length - 1]?.iteration ?? topo.iterations_run} ({topo.optimized_compliance_mj.toFixed(1)} mJ /{" "}
            {topo.optimized_mass_g.toFixed(1)}g)
          </text>
        </svg>
      )}
    </div>
  );
};

const STOCK_PRINTED_BRACKET_MASS_G = 45;

const TetFEACard: React.FC<{
  tet: TetFEAResponse;
  activeCustomBracket?: CustomBracketOverrideData | null;
  onInjectCustomBracket?: (override: CustomBracketOverrideData) => void;
}> = ({ tet, activeCustomBracket, onInjectCustomBracket }) => {
  const isSafe = tet.safety_factor >= 1.5 && tet.failure_mode === "safe";
  const backendLabel = formatBackendLabel(tet.backend);
  const elemLabel = formatElementLabel(tet.element_type);
  const injectLabel = `${tet.part_name} (${tet.material_id})`;
  const massDeltaG = Number((tet.mass_g - STOCK_PRINTED_BRACKET_MASS_G).toFixed(1));
  const stiffNPerMm = Number(tet.stiffness_n_per_mm.toFixed(1));
  const isAlreadyInjected = Boolean(
    activeCustomBracket &&
      activeCustomBracket.source === "tet10_fea" &&
      activeCustomBracket.label === injectLabel &&
      Math.abs(activeCustomBracket.stiffness_n_per_mm - stiffNPerMm) < 0.2
  );

  return (
    <div className="section-card" style={{ borderColor: "rgba(56, 189, 248, 0.55)" }}>
      <div className="section-title">
        <span>Curved 3D Tet FEA</span>
        <span className={`badge-pill ${isSafe ? "ok" : "bad"}`}>
          SF {tet.safety_factor.toFixed(2)}×
        </span>
      </div>

      <div className="badge-pill-row">
        <span className="badge-pill tet">{backendLabel}</span>
        <span className="badge-pill">{elemLabel}</span>
        <span className="badge-pill" style={{ borderColor: "var(--border)", color: "var(--muted)", background: "#0d131d" }}>
          {tet.is_fdm_anisotropic ? "FDM Anisotropic" : "Isotropic"}
        </span>
      </div>

      <div className="kpi-grid">
        <div className="kpi-box">
          <span className="kpi-label">Max Displacement</span>
          <span className="kpi-value" style={{ color: "var(--accent)" }}>
            {tet.max_displacement_um.toFixed(1)} µm
          </span>
          <span className="kpi-sub">Stiffness {tet.stiffness_n_per_mm.toFixed(1)} N/mm</span>
        </div>
        <div className="kpi-box">
          <span className="kpi-label">Safety Factor</span>
          <span className={`kpi-value ${isSafe ? "tag-ok" : "tag-bad"}`}>
            {tet.safety_factor.toFixed(2)}×
          </span>
          <span className="kpi-sub">{tet.failure_mode.replace(/_/g, " ")}</span>
        </div>
      </div>

      <div className="metric-kv">
        <span>Part / Material</span>
        <span className="metric-val">
          {tet.part_name} ({tet.material_id})
        </span>
      </div>
      <div className="metric-kv">
        <span>Backend</span>
        <span className="metric-val">{backendLabel}</span>
      </div>
      <div className="metric-kv">
        <span>Element Type</span>
        <span className="metric-val">{elemLabel}</span>
      </div>
      <div className="metric-kv">
        <span>Mesh Size</span>
        <span className="metric-val">{tet.mesh_size_mm.toFixed(2)} mm</span>
      </div>
      <div className="metric-kv">
        <span>Elements / DOFs</span>
        <span className="metric-val">
          {tet.num_elements.toLocaleString()} / {tet.num_dofs.toLocaleString()}
        </span>
      </div>
      <div className="metric-kv">
        <span>Volume</span>
        <span className="metric-val">{tet.volume_mm3.toFixed(1)} mm³</span>
      </div>
      <div className="metric-kv">
        <span>Mass</span>
        <span className="metric-val">{tet.mass_g.toFixed(2)} g</span>
      </div>
      <div className="metric-kv">
        <span>Max von Mises</span>
        <span className="metric-val">
          {tet.max_von_mises_mpa.toFixed(2)} MPa
          <span style={{ color: "var(--muted)", fontWeight: 400 }}> / {tet.yield_strength_mpa.toFixed(0)}</span>
        </span>
      </div>
      <div className="metric-kv">
        <span>Max Z-Layer Tension</span>
        <span className="metric-val">
          {tet.max_z_tensile_mpa.toFixed(2)} MPa
          <span style={{ color: "var(--muted)", fontWeight: 400 }}> / {tet.z_layer_strength_mpa.toFixed(0)}</span>
        </span>
      </div>
      {onInjectCustomBracket && (
        <div className="inject-bracket-row">
          <button
            type="button"
            className={`btn inject-bracket-btn ${isAlreadyInjected ? "injected" : ""}`}
            onClick={() =>
              onInjectCustomBracket({
                label: injectLabel,
                stiffness_n_per_mm: stiffNPerMm,
                mass_delta_g: massDeltaG,
                source: "tet10_fea",
              })
            }
            title={`Inject ${stiffNPerMm} N/mm stiffness and ${massDeltaG >= 0 ? "+" : ""}${massDeltaG}g mass delta (vs 45g stock bracket) into the full-printer FEA model`}
          >
            {isAlreadyInjected ? "Active in Full Printer FEA ✓" : "Inject into Full Printer FEA"}
          </button>
          <span className="inject-bracket-sub">
            k = {stiffNPerMm} N/mm · Δm = {massDeltaG >= 0 ? "+" : ""}
            {massDeltaG}g vs 45g stock
          </span>
        </div>
      )}
    </div>
  );
};

const SolidHexFEACard: React.FC<{
  solidFea: SolidFEAResponse;
  activeCustomBracket?: CustomBracketOverrideData | null;
  onInjectCustomBracket?: (override: CustomBracketOverrideData) => void;
}> = ({ solidFea, activeCustomBracket, onInjectCustomBracket }) => {
  const injectLabel = `${solidFea.part_name} (${solidFea.material_id})`;
  const massDeltaG = Number((solidFea.mass_g - STOCK_PRINTED_BRACKET_MASS_G).toFixed(1));
  const stiffNPerMm = Number(solidFea.stiffness_n_per_mm.toFixed(1));
  const isAlreadyInjected = Boolean(
    activeCustomBracket &&
      activeCustomBracket.source === "hex8_fea" &&
      activeCustomBracket.label === injectLabel &&
      Math.abs(activeCustomBracket.stiffness_n_per_mm - stiffNPerMm) < 0.2
  );

  return (
    <div className="section-card" style={{ borderColor: "rgba(56,189,248,0.45)" }}>
      <div className="section-title">
        <span>3D Solid Hex8 Part FEA</span>
        <span className={`badge-pill ${solidFea.safety_factor < 1.5 ? "bad" : "ok"}`}>
          SF {solidFea.safety_factor.toFixed(2)}×
        </span>
      </div>
      <div className="metric-kv">
        <span>Part / Material</span>
        <span className="metric-val">
          {solidFea.part_name} ({solidFea.is_fdm_anisotropic ? "FDM" : "ISO"})
        </span>
      </div>
      <div className="metric-kv">
        <span>Mesh / Mass</span>
        <span className="metric-val">
          {solidFea.num_elements} Hex8 ({solidFea.num_dofs} DOF) • {solidFea.mass_g}g
        </span>
      </div>
      <div className="metric-kv">
        <span>Max Disp / Stiffness</span>
        <span className="metric-val">
          {solidFea.max_displacement_um.toFixed(1)} µm • {solidFea.stiffness_n_per_mm.toFixed(0)} N/mm
        </span>
      </div>
      <div className="metric-kv">
        <span>Peak von Mises / Z-Layer</span>
        <span className="metric-val">
          {solidFea.max_von_mises_mpa.toFixed(2)} / {solidFea.max_z_tensile_mpa.toFixed(2)} MPa
        </span>
      </div>
      <div className="metric-kv">
        <span>Status</span>
        <span className={`metric-val ${solidFea.failure_mode === "safe" ? "tag-ok" : "tag-bad"}`}>
          {solidFea.failure_mode.replace(/_/g, " ")}
        </span>
      </div>
      {onInjectCustomBracket && (
        <div className="inject-bracket-row">
          <button
            type="button"
            className={`btn inject-bracket-btn ${isAlreadyInjected ? "injected" : ""}`}
            onClick={() =>
              onInjectCustomBracket({
                label: injectLabel,
                stiffness_n_per_mm: stiffNPerMm,
                mass_delta_g: massDeltaG,
                source: "hex8_fea",
              })
            }
            title={`Inject ${stiffNPerMm} N/mm stiffness and ${massDeltaG >= 0 ? "+" : ""}${massDeltaG}g mass delta (vs 45g stock bracket) into the full-printer FEA model`}
          >
            {isAlreadyInjected ? "Active in Full Printer FEA ✓" : "Inject into Full Printer FEA"}
          </button>
          <span className="inject-bracket-sub">
            k = {stiffNPerMm} N/mm · Δm = {massDeltaG >= 0 ? "+" : ""}
            {massDeltaG}g vs 45g stock
          </span>
        </div>
      )}
    </div>
  );
};

const BOTTLENECK_SOURCES: Array<{
  key: keyof AccelBottleneckShares;
  label: string;
  shortLabel: string;
  color: string;
}> = [
  { key: "frame", label: "Frame & Gantry", shortLabel: "Frame", color: "#38bdf8" },
  { key: "belt", label: "Belt Stretch", shortLabel: "Belt", color: "#f59e0b" },
  { key: "motor", label: "Motor Stiffness", shortLabel: "Motor", color: "#a78bfa" },
  { key: "bracket", label: "Brackets", shortLabel: "Bkt", color: "#fb7185" },
];

const OVERALL_CAUSE_META: Record<
  OverallLimitingCauseItem["category"],
  { badge: string; color: string }
> = {
  belt: { badge: "BELT", color: "#f59e0b" },
  structural: { badge: "STRUCT", color: "#38bdf8" },
  motor: { badge: "MOTOR", color: "#a78bfa" },
  bracket: { badge: "BKT", color: "#fb7185" },
  mass: { badge: "MASS", color: "#22d3ee" },
};

const AccelBottleneckCard: React.FC<{
  bottleneck?: AccelBottleneckData;
  limitingSummary?: LimitingModalSummaryData;
  shaperX?: ShaperData | null;
  shaperY?: ShaperData | null;
  modes?: ModeShapeData[];
  onInspectModalMode?: (modeIdx: number) => void;
  changed: boolean;
}> = ({ bottleneck, limitingSummary, shaperX, shaperY, modes, onInspectModalMode, changed }) => {
  const defaultLimitingAxis: "X" | "Y" =
    limitingSummary?.limiting_axis ?? bottleneck?.limiting_axis ?? "Y";
  const [selectedAxis, setSelectedAxis] = useState<"X" | "Y">(defaultLimitingAxis);

  useEffect(() => {
    setSelectedAxis(defaultLimitingAxis);
  }, [defaultLimitingAxis]);

  const fallbackShares: AccelBottleneckShares = limitingSummary?.compliance_share_pct ??
    bottleneck?.shares_pct ?? { frame: 25, belt: 25, motor: 25, bracket: 25 };

  const activeShares: AccelBottleneckShares =
    selectedAxis === "X"
      ? shaperX?.compliance_share_pct ?? bottleneck?.x_shares_pct ?? fallbackShares
      : shaperY?.compliance_share_pct ?? bottleneck?.y_shares_pct ?? fallbackShares;

  const totalShare = Math.max(
    BOTTLENECK_SOURCES.reduce((acc, s) => acc + Math.max(0, activeShares[s.key] ?? 0), 0),
    1e-6
  );

  const primarySourceKey: keyof AccelBottleneckShares = limitingSummary
    ? limitingSummary.limiting_kind === "structural"
      ? "frame"
      : limitingSummary.limiting_kind
    : bottleneck?.primary_source ?? "frame";

  const primaryMeta =
    BOTTLENECK_SOURCES.find((s) => s.key === primarySourceKey) ?? BOTTLENECK_SOURCES[0];

  // Dominant source for the currently viewed axis (highlights the biggest contributor in the legend)
  let activeDominantKey: keyof AccelBottleneckShares = primarySourceKey;
  let activeMaxVal = -1;
  for (const s of BOTTLENECK_SOURCES) {
    const v = activeShares[s.key] ?? 0;
    if (v > activeMaxVal) {
      activeMaxVal = v;
      activeDominantKey = s.key;
    }
  }

  // Component frequencies for the currently viewed axis
  const activeShaper = selectedAxis === "X" ? shaperX : shaperY;
  const useSummaryFreqs = Boolean(
    limitingSummary && selectedAxis === limitingSummary.limiting_axis
  );
  const compFreqs: Record<keyof AccelBottleneckShares, number | undefined> = {
    frame: useSummaryFreqs
      ? limitingSummary?.structural_freq_hz
      : activeShaper?.structural_freq_hz,
    belt: useSummaryFreqs ? limitingSummary?.belt_freq_hz : activeShaper?.belt_freq_hz,
    motor: useSummaryFreqs ? limitingSummary?.motor_freq_hz : activeShaper?.motor_freq_hz,
    bracket: useSummaryFreqs ? limitingSummary?.bracket_freq_hz : activeShaper?.bracket_freq_hz,
  };

  // X vs Y summary info for comparison strip
  const xFreq =
    shaperX?.effective_freq_hz ??
    (limitingSummary
      ? limitingSummary.limiting_axis === "X"
        ? limitingSummary.effective_freq_hz
        : limitingSummary.other_axis_effective_freq_hz
      : undefined);
  const yFreq =
    shaperY?.effective_freq_hz ??
    (limitingSummary
      ? limitingSummary.limiting_axis === "Y"
        ? limitingSummary.effective_freq_hz
        : limitingSummary.other_axis_effective_freq_hz
      : undefined);
  const xAccel =
    shaperX?.max_accel_mm_s2 ??
    (limitingSummary
      ? limitingSummary.limiting_axis === "X"
        ? limitingSummary.max_accel_mm_s2
        : limitingSummary.other_axis_max_accel_mm_s2
      : undefined);
  const yAccel =
    shaperY?.max_accel_mm_s2 ??
    (limitingSummary
      ? limitingSummary.limiting_axis === "Y"
        ? limitingSummary.max_accel_mm_s2
        : limitingSummary.other_axis_max_accel_mm_s2
      : undefined);

  const beltSub = limitingSummary?.belt_sub_breakdown;
  const beltShareForSub = Math.max(
    limitingSummary?.compliance_share_pct?.belt ?? 0,
    activeShares.belt ?? 0
  );
  const showBeltSub = Boolean(beltSub && beltShareForSub >= 10);

  const primaryModeIdx =
    limitingSummary && modes && modes.length > 0
      ? (() => {
          const byNum = modes.findIndex(
            (m) => m.mode_number === limitingSummary.primary_structural_mode_number
          );
          if (byNum >= 0) return byNum;
          return modes.findIndex((m) => Boolean(m.is_primary_limiting_mode));
        })()
      : -1;

  const formatAxisLine = (shares: AccelBottleneckShares) =>
    BOTTLENECK_SOURCES.map((s) => `${s.shortLabel} ${Math.round(shares[s.key] ?? 0)}%`).join(" · ");

  return (
    <div className={`section-card bottleneck-card ${changed ? "card-changed" : ""}`}>
      <div className="section-title bottleneck-header">
        <span>{limitingSummary ? "Limiting Resonance & Top Causes" : "Accel Limit Breakdown"}</span>
        <span
          className="bottleneck-pill"
          style={{
            background: `${primaryMeta.color}22`,
            border: `1px solid ${primaryMeta.color}66`,
            color: primaryMeta.color,
          }}
        >
          {limitingSummary
            ? `Limiting: ${limitingSummary.limiting_axis} · ${limitingSummary.effective_freq_hz.toFixed(1)} Hz`
            : bottleneck
              ? `Limiting: ${bottleneck.limiting_axis} · ${Math.round(bottleneck.primary_share_pct)}% ${bottleneck.primary_label}`
              : `Limiting: ${defaultLimitingAxis}`}
        </span>
      </div>

      {limitingSummary && (
        <div className="limiting-headline-box" style={{ borderLeftColor: primaryMeta.color }}>
          <div className="limiting-headline-text">{limitingSummary.headline}</div>
          {xFreq != null && yFreq != null && xAccel != null && yAccel != null && (
            <div className="limiting-xy-compare">
              {(limitingSummary.limiting_axis === "X"
                ? (["X", "Y"] as const)
                : (["Y", "X"] as const)
              ).map((ax) => {
                  const isLim = ax === limitingSummary.limiting_axis;
                  const f = ax === "X" ? xFreq : yFreq;
                  const a = ax === "X" ? xAccel : yAccel;
                  const shaperName =
                    ax === "X"
                      ? shaperX?.recommended_shaper ??
                        (isLim ? limitingSummary.recommended_shaper : "")
                      : shaperY?.recommended_shaper ??
                        (isLim ? limitingSummary.recommended_shaper : "");
                  return (
                    <div
                      key={ax}
                      className={`limiting-xy-chip ${isLim ? "is-limiting" : ""}`}
                    >
                      <span className="limiting-xy-axis">{ax}:</span>
                      <strong>{f.toFixed(1)} Hz</strong>
                      <span className="limiting-xy-sep">·</span>
                      <span>{Math.round(a).toLocaleString()} mm/s²</span>
                      {shaperName && <span className="limiting-xy-shaper">({shaperName})</span>}
                      {isLim && <span className="limiting-tag-mini">LIMITING</span>}
                    </div>
                  );
                })}
            </div>
          )}
        </div>
      )}

      <div className="bottleneck-axis-row">
        <span>Compliance Share ({selectedAxis}-Axis)</span>
        <div className="bottleneck-axis-tabs">
          {(["X", "Y"] as const).map((ax) => {
            const isLimiting = ax === defaultLimitingAxis;
            return (
              <button
                key={ax}
                type="button"
                className={`bottleneck-axis-btn ${selectedAxis === ax ? "active" : ""}`}
 aria-pressed={selectedAxis === ax}
                onClick={() => setSelectedAxis(ax)}
                title={
                  isLimiting
                    ? `${ax}-axis is the limiting acceleration axis`
                    : `Inspect ${ax}-axis compliance share`
                }
              >
                {ax}
                {isLimiting ? " ★" : ""}
              </button>
            );
          })}
        </div>
      </div>

      <div
        className="bottleneck-bar"
        role="img"
        aria-label={`Acceleration compliance share on ${selectedAxis}-axis`}
      >
        {BOTTLENECK_SOURCES.map((src) => {
          const rawPct = Math.max(0, activeShares[src.key] ?? 0);
          const normPct = (rawPct / totalShare) * 100;
          if (normPct < 0.4) return null;
          return (
            <div
              key={src.key}
              className="bottleneck-seg"
              style={{ width: `${normPct}%`, background: src.color }}
              title={`${src.label}: ${rawPct.toFixed(1)}%`}
            >
              {normPct >= 14 ? `${Math.round(rawPct)}%` : ""}
            </div>
          );
        })}
      </div>

      <div className="bottleneck-legend">
        {BOTTLENECK_SOURCES.map((src) => {
          const val = activeShares[src.key] ?? 0;
          const isDom = src.key === activeDominantKey;
          const freq = compFreqs[src.key];
          const hasFreq = freq != null && Number.isFinite(freq) && freq > 0 && freq < 9999;
          return (
            <div key={src.key} className={`bottleneck-legend-item ${isDom ? "primary" : ""}`}>
              <span className="bottleneck-legend-left">
                <span className="swatch-dot" style={{ background: src.color }} />
                <span>{src.label}</span>
              </span>
              <span className="bottleneck-legend-right">
                <span
                  className="bottleneck-legend-pct"
                  style={{ color: isDom ? src.color : undefined }}
                >
                  {val.toFixed(0)}%
                </span>
                {hasFreq && (
                  <span className="bottleneck-legend-hz">· {Math.round(freq)} Hz</span>
                )}
              </span>
            </div>
          );
        })}
      </div>

      {showBeltSub && beltSub && (
        <div
          className="belt-sub-strip"
          title={`Total belt drive stiffness: ${Math.round(beltSub.k_belt_total_n_per_mm)} N/mm (Cord ${Math.round(
            beltSub.k_belt_cord_n_per_mm
          )} · Tooth Mesh ${Math.round(beltSub.k_tooth_mesh_n_per_mm)} · Idler Shafts ${Math.round(
            beltSub.k_idler_shaft_n_per_mm
          )} N/mm)`}
        >
          <span className="belt-sub-prefix">Belt Stretch Split:</span>{" "}
          <strong>Cord EA/L {Math.round(beltSub.belt_cord_share_pct)}%</strong> (
          {Math.round(beltSub.k_belt_cord_n_per_mm)} N/mm) · Tooth Mesh{" "}
          <strong>{Math.round(beltSub.tooth_mesh_share_pct)}%</strong> · Idlers{" "}
          <strong>{Math.round(beltSub.idler_shaft_share_pct)}%</strong>
        </div>
      )}

      {limitingSummary &&
        limitingSummary.top_overall_causes &&
        limitingSummary.top_overall_causes.length > 0 && (
          <div className="top-causes-block">
            <div className="top-causes-head">
              Top Causes Limiting Your {limitingSummary.limiting_axis}-Axis
            </div>
            <div className="top-causes-list">
              {limitingSummary.top_overall_causes.slice(0, 5).map((cause) => {
                const meta = OVERALL_CAUSE_META[cause.category] ?? {
                  badge: "CAUSE",
                  color: "#38bdf8",
                };
                const barWidth = Math.max(4, Math.min(100, cause.share_pct));
                return (
                  <div
                    key={cause.id}
                    className="top-cause-row"
                    title={cause.detail}
                  >
                    <div className="top-cause-main">
                      <span
                        className="cause-cat-pill"
                        style={{
                          background: `${meta.color}20`,
                          borderColor: `${meta.color}66`,
                          color: meta.color,
                        }}
                      >
                        {meta.badge}
                      </span>
                      <span className="top-cause-label">{cause.label}</span>
                      <span className="top-cause-metrics">
                        <strong className="top-cause-pct">{Math.round(cause.share_pct)}%</strong>
                        {cause.est_hz_gain > 0 && (
                          <span className="top-cause-hz">
                            +{formatHzGain(cause.est_hz_gain)} Hz
                          </span>
                        )}
                        {cause.est_accel_gain_mm_s2 >= 50 && (
                          <span className="top-cause-accel">
                            +{Math.round(cause.est_accel_gain_mm_s2).toLocaleString()} mm/s²
                          </span>
                        )}
                      </span>
                    </div>
                    <div className="top-cause-bar-track">
                      <div
                        className="top-cause-bar-fill"
                        style={{ width: `${barWidth}%`, background: meta.color }}
                      />
                    </div>
                    {cause.detail && <div className="top-cause-detail">{cause.detail}</div>}
                  </div>
                );
              })}
            </div>
          </div>
        )}

      {limitingSummary?.moving_mass_info && (
        <div className="moving-mass-lever-note">
          <span className="moving-mass-lever-key">Moving Mass Lever:</span> Cutting 25% off{" "}
          <strong>{limitingSummary.moving_mass_info.label}</strong> (
          {Math.round(limitingSummary.moving_mass_info.mass_g)}g) adds{" "}
          <strong className="moving-mass-lever-gain">
            +{formatHzGain(limitingSummary.moving_mass_info.est_hz_gain_25pct_lighter)} Hz
          </strong>
        </div>
      )}

      {limitingSummary && onInspectModalMode && primaryModeIdx >= 0 && (
        <button
          type="button"
          className="animate-limiting-mode-btn"
          onClick={() => onInspectModalMode(primaryModeIdx)}
          title={`Switch 3D viewport to Modal mode and animate Mode ${limitingSummary.primary_structural_mode_number} (${limitingSummary.primary_structural_mode_label})`}
        >
          ▶ Animate Limiting Mode M{limitingSummary.primary_structural_mode_number} (
          {limitingSummary.primary_structural_mode_freq_hz.toFixed(1)} Hz) in 3D
        </button>
      )}

      {!limitingSummary && bottleneck && (
        <>
          <div className="bottleneck-xy-strip">
            <div
              className={`bottleneck-xy-line ${bottleneck.limiting_axis === "X" ? "is-limiting" : ""}`}
            >
              <span className="bottleneck-xy-axis">X:</span>
              <span>{formatAxisLine(bottleneck.x_shares_pct ?? bottleneck.shares_pct)}</span>
            </div>
            <div
              className={`bottleneck-xy-line ${bottleneck.limiting_axis === "Y" ? "is-limiting" : ""}`}
            >
              <span className="bottleneck-xy-axis">Y:</span>
              <span>{formatAxisLine(bottleneck.y_shares_pct ?? bottleneck.shares_pct)}</span>
            </div>
          </div>

          {bottleneck.recommendation && (
            <div className="bottleneck-tip" style={{ borderLeftColor: primaryMeta.color }}>
              {bottleneck.recommendation}
            </div>
          )}
        </>
      )}
    </div>
  );
};

const WeightShaperPotentialCard: React.FC<{
  gains: WeightShaperGainsSummary;
  changed: boolean;
  embedded?: boolean;
}> = ({ gains, changed, embedded }) => {
  const axisTagStyle = (
    axis: "X" | "Y" | "XY",
    affectsLimiting: boolean
  ): React.CSSProperties => ({
    display: "inline-flex",
    alignItems: "center",
    padding: "0 4px",
    borderRadius: 4,
    fontSize: 11,
    fontWeight: 600,
    letterSpacing: "0.02em",
    marginLeft: 5,
    verticalAlign: "middle",
    background: affectsLimiting
      ? axis === "XY"
        ? "rgba(56, 189, 248, 0.16)"
        : "rgba(16, 185, 129, 0.16)"
      : "rgba(148, 163, 184, 0.1)",
    border: affectsLimiting
      ? axis === "XY"
        ? "1px solid rgba(56, 189, 248, 0.38)"
        : "1px solid rgba(16, 185, 129, 0.38)"
      : "1px solid rgba(148, 163, 184, 0.22)",
    color: affectsLimiting ? (axis === "XY" ? "#7dd3fc" : "#6ee7b7") : "#64748b",
  });

  return (
    <div
      className={embedded ? undefined : `section-card ${changed ? "card-changed" : ""}`}
      style={
        embedded
          ? { display: "flex", flexDirection: "column", gap: 8 }
          : {
              gap: 8,
              borderColor: "rgba(56, 189, 248, 0.32)",
            }
      }
    >
      {!embedded && (
        <div className="section-title bottleneck-header">
          <span>Weight-Reduction Shaper Potential</span>
          <span
            className="bottleneck-pill"
            style={{
              background: "rgba(56, 189, 248, 0.14)",
              border: "1px solid rgba(56, 189, 248, 0.45)",
              color: "#38bdf8",
            }}
          >
            Limiting: {gains.limiting_axis}-Axis ({gains.limiting_shaper_hz.toFixed(1)} Hz)
          </span>
        </div>
      )}

      <div
        style={{
          padding: "7px 9px",
          borderRadius: 6,
          background: "rgba(13, 19, 29, 0.78)",
          border: "1px solid rgba(39, 50, 71, 0.9)",
          borderLeft: "2.5px solid #34d399",
          fontSize: 11,
          lineHeight: 1.42,
          color: "var(--text)",
          fontVariantNumeric: "tabular-nums",
        }}
      >
        Reducing <strong>{gains.limiting_axis}-axis</strong> moving weight by 25% (
        <span style={{ color: "var(--muted)" }}>−{Math.round(gains.total_potential_cut_g)}g</span>
        ) adds up to{" "}
        <strong style={{ color: "#34d399" }}>
          +{gains.total_potential_hz_gain.toFixed(1)} Hz
        </strong>{" "}
        <span style={{ color: "#7dd3fc", fontWeight: 500 }}>
          (+{Math.round(gains.total_potential_accel_gain_mm_s2).toLocaleString()} mm/s²)
        </span>
      </div>

      <div className="psd-table-wrap" style={{ overflowX: "auto" }}>
        <table className="psd-table" style={{ width: "100%", fontSize: 11 }}>
          <thead>
            <tr>
              <th>Component</th>
              <th className="num">Mass / Cut</th>
              <th className="num">Shaper Gain</th>
              <th className="num">Hz/100g</th>
            </tr>
          </thead>
          <tbody>
            {gains.items.map((item) => {
              const affectsLimiting =
                item.affects_axis === "XY" || item.affects_axis === gains.limiting_axis;
              const isMuted = !affectsLimiting || item.est_hz_gain < 0.05;
              const isHighGain = affectsLimiting && item.est_hz_gain >= 1.0;

              return (
                <tr
                  key={item.id}
                  title={item.note || undefined}
                  style={{
                    background: isHighGain ? "rgba(16, 185, 129, 0.07)" : undefined,
                    opacity: isMuted ? 0.58 : 1,
                  }}
                >
                  <td
                    style={{
                      padding: "5px 4px",
                      borderBottom: "1px dashed rgba(255, 255, 255, 0.05)",
                      boxShadow: isHighGain ? "inset 2px 0 0 #34d399" : undefined,
                      color: isMuted ? "var(--muted)" : "var(--text)",
                      fontWeight: isHighGain ? 600 : 500,
                    }}
                  >
                    <span>{item.label}</span>
                    <span style={axisTagStyle(item.affects_axis, affectsLimiting)}>
                      {item.affects_axis}
                    </span>
                  </td>
                  <td
                    className="num"
                    style={{
                      padding: "5px 4px",
                      borderBottom: "1px dashed rgba(255, 255, 255, 0.05)",
                      whiteSpace: "nowrap",
                    }}
                  >
                    <div style={{ color: isMuted ? "var(--muted)" : "var(--text)" }}>
                      {Math.round(item.current_mass_g)}g
                    </div>
                    <div style={{ fontSize: 11, color: "var(--muted)", lineHeight: 1.15 }}>
                      −{Math.round(item.example_cut_g)}g
                    </div>
                  </td>
                  <td
                    className="num"
                    style={{
                      padding: "5px 4px",
                      borderBottom: "1px dashed rgba(255, 255, 255, 0.05)",
                      whiteSpace: "nowrap",
                    }}
                  >
                    <div
                      style={{
                        fontWeight: isHighGain ? 700 : 600,
                        color: isHighGain
                          ? "#34d399"
                          : isMuted
                            ? "var(--muted)"
                            : "#7dd3fc",
                      }}
                    >
                      +{item.est_hz_gain.toFixed(1)} Hz
                    </div>
                    <div
                      style={{
                        fontSize: 11,
                        color: isHighGain ? "rgba(125, 211, 252, 0.85)" : "var(--muted)",
                        lineHeight: 1.15,
                      }}
                    >
                      +{Math.round(item.est_accel_gain_mm_s2).toLocaleString()} mm/s²
                    </div>
                  </td>
                  <td
                    className="num"
                    style={{
                      padding: "5px 4px",
                      borderBottom: "1px dashed rgba(255, 255, 255, 0.05)",
                      whiteSpace: "nowrap",
                      color: isHighGain
                        ? "#7dd3fc"
                        : isMuted
                          ? "var(--muted)"
                          : "var(--text)",
                    }}
                  >
                    +{item.hz_per_100g.toFixed(1)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};

const DeepVerifyCard: React.FC<{
  result: DeepVerifyResponse;
  viewingDeepSim: boolean;
  onToggleViewingDeepSim: (useDeep: boolean) => void;
  onInspectDeepFea: () => void;
  onDismiss: () => void;
}> = ({
  result,
  viewingDeepSim,
  onToggleViewingDeepSim,
  onInspectDeepFea,
  onDismiss,
}) => {
  const { fast_mesh_n, deep_mesh_n, fast_summary: fast, deep_summary: deep, convergence } = result;
  const fmtSignedPct = (v: number) => `${v > 0 ? "+" : ""}${v.toFixed(1)}%`;
  const deltaClass = (pct: number) =>
    Math.abs(pct) <= 2.5 ? "tag-ok" : Math.abs(pct) <= 6.0 ? "tag-warn" : "tag-bad";

  const hasDrum = fast.panel_drum_freq_hz > 0 || deep.panel_drum_freq_hz > 0;
  const drumDelta = deep.panel_drum_freq_hz - fast.panel_drum_freq_hz;
  const drumColor =
    deep.panel_drum_freq_hz > 0
      ? deep.panel_drum_freq_hz < 65
        ? "tag-warn"
        : deep.panel_drum_freq_hz >= 85
          ? "tag-ok"
          : ""
      : "";

  const hasPanelStress = fast.panel_max_von_mises_mpa > 0 || deep.panel_max_von_mises_mpa > 0;
  const stressDelta = deep.panel_max_von_mises_mpa - fast.panel_max_von_mises_mpa;

  const isSignoff = Boolean(result.workspace_fea && result.uncertainty);
  const verdict = result.verdict ?? (convergence.converged ? "pass" : "caution");
  const verdictBadgeClass =
    verdict === "pass" ? "ok" : verdict === "fail" ? "bad" : "warn";
  const verdictLabel =
    verdict === "pass"
      ? isSignoff
        ? "Sign-Off Pass"
        : "Converged"
      : verdict === "fail"
        ? "Needs Attention"
        : isSignoff
          ? "Sign-Off Caution"
          : "Mesh Sensitive";

  const ws = result.workspace_fea;
  const unc = result.uncertainty;
  const checks = result.checks ?? [];
  // Relative workspace position (0–1) shown as a percentage of travel.
  const pct = (rel: number) => `${Math.round(rel * 100)}%`;
  // Colour a workspace cell by its resonance drop against the bed-centre solve.
  const dropClass = (hz: number) => {
    const centerHz = ws?.center_min_axis_hz ?? 0;
    const drop = centerHz > 0 ? ((centerHz - hz) / centerHz) * 100 : 0;
    return drop <= 5 ? "ok" : drop <= 12 ? "warn" : "bad";
  };

  return (
    <div
      className={`section-card deep-verify-card ${
        verdict === "pass" ? "converged" : verdict === "fail" ? "failed" : "refining"
      }`}
    >
      <div className="deep-verify-header">
        <span className="section-title" style={{ margin: 0 }}>
          {isSignoff ? "Sign-Off Verification" : "Deep Verify Convergence"}
        </span>
        <div className="deep-verify-header-right">
          <span className={`badge-pill ${verdictBadgeClass}`}>{verdictLabel}</span>
          <button
            type="button"
            className="deep-verify-close"
            onClick={onDismiss}
            title="Dismiss verification results"
            aria-label="Dismiss verification results"
          >
            ×
          </button>
        </div>
      </div>

      <div className="deep-verify-meta">
        <span>
          <strong>
            {fast_mesh_n}×{fast_mesh_n} → {deep_mesh_n}×{deep_mesh_n}
          </strong>{" "}
          MITC4
          {result.total_fea_solves ? (
            <>
              {" "}
              · <strong>{result.total_fea_solves}</strong> FEA solves
            </>
          ) : null}
        </span>
        <span>
          <strong>{deep.num_shell_elements.toLocaleString()}</strong> shells ·{" "}
          <strong>
            {result.total_solve_time_ms != null
              ? `${(result.total_solve_time_ms / 1000).toFixed(1)} s`
              : `${Math.round(deep.solve_time_ms)} ms`}
          </strong>
        </span>
      </div>

      {unc && (
        <div className="signoff-block">
          <div className="signoff-subhead">
            <span title={unc.coverage_note}>90% Parametric Bands</span>
            <span className="signoff-submeta">
              Model Trust {unc.overall_trust_pct}% ({unc.num_groups_perturbed} perturbed)
            </span>
          </div>
          <table className="deep-verify-table">
            <thead>
              <tr>
                <th>Metric</th>
                <th className="num">Nominal</th>
                <th className="num">90% Band</th>
                <th className="num">±%</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Shaper X (Hz)</td>
                <td className="num">
                  <strong>{unc.bands.shaper_x_hz.nominal.toFixed(1)}</strong>
                </td>
                <td className="num">
                  {unc.bands.shaper_x_hz.p05.toFixed(1)}–{unc.bands.shaper_x_hz.p95.toFixed(1)}
                </td>
                <td className={`num ${deltaClass(unc.bands.shaper_x_hz.ci90_pct)}`}>
                  ±{unc.bands.shaper_x_hz.ci90_pct.toFixed(1)}%
                </td>
              </tr>
              <tr>
                <td>Shaper Y (Hz)</td>
                <td className="num">
                  <strong>{unc.bands.shaper_y_hz.nominal.toFixed(1)}</strong>
                </td>
                <td className="num">
                  {unc.bands.shaper_y_hz.p05.toFixed(1)}–{unc.bands.shaper_y_hz.p95.toFixed(1)}
                </td>
                <td className={`num ${deltaClass(unc.bands.shaper_y_hz.ci90_pct)}`}>
                  ±{unc.bands.shaper_y_hz.ci90_pct.toFixed(1)}%
                </td>
              </tr>
              <tr>
                <td>Safe Accel (mm/s²)</td>
                <td className="num">
                  <strong>{Math.round(unc.bands.recommended_max_accel_mm_s2.nominal).toLocaleString()}</strong>
                </td>
                <td className="num">
                  {Math.round(unc.bands.recommended_max_accel_mm_s2.p05).toLocaleString()}–
                  {Math.round(unc.bands.recommended_max_accel_mm_s2.p95).toLocaleString()}
                </td>
                <td className={`num ${deltaClass(unc.bands.recommended_max_accel_mm_s2.ci90_pct)}`}>
                  ±{unc.bands.recommended_max_accel_mm_s2.ci90_pct.toFixed(1)}%
                </td>
              </tr>
              <tr>
                <td>Nozzle–Bed Flex (µm)</td>
                <td className="num">
                  <strong>{unc.bands.combined_max_deflection_um.nominal.toFixed(1)}</strong>
                </td>
                <td className="num">
                  {unc.bands.combined_max_deflection_um.p05.toFixed(1)}–
                  {unc.bands.combined_max_deflection_um.p95.toFixed(1)}
                </td>
                <td className={`num ${deltaClass(unc.bands.combined_max_deflection_um.ci90_pct)}`}>
                  ±{unc.bands.combined_max_deflection_um.ci90_pct.toFixed(1)}%
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      )}

      {ws && (
        <div className="signoff-block">
          <div className="signoff-subhead">
            <span>
              True 3D FEA Workspace (
              {ws.xy_position_modeled === false ? "Z sweep" : `${ws.grid_n}×${ws.grid_n} + Z`})
            </span>
            <span className="signoff-submeta">
              Worst: <strong>{ws.worst_case_min_axis_hz.toFixed(1)} Hz</strong> (-{ws.worst_freq_drop_pct.toFixed(1)}%)
            </span>
          </div>
          {ws.xy_position_modeled === false ? (
            <div className="signoff-ws-foot">
              <span>XY position is not modeled for this architecture — Z sweep only.</span>
            </div>
          ) : (
          <div
            className="signoff-ws-grid"
            style={{ gridTemplateColumns: `repeat(${ws.grid_n}, minmax(0, 1fr))` }}
          >
            {ws.min_axis_grid_hz.map((row, rIdx) =>
              row.map((hz, cIdx) => {
                const cellCls = dropClass(hz);
                return (
                  <div
                    key={`${rIdx}-${cIdx}`}
                    className={`signoff-ws-cell ${cellCls}`}
                    title={`X ${pct(ws.coords[cIdx] ?? 0)}, Y ${pct(ws.coords[rIdx] ?? 0)} · X ${ws.freq_x_grid_hz[rIdx]?.[cIdx]?.toFixed(1)} Hz, Y ${ws.freq_y_grid_hz[rIdx]?.[cIdx]?.toFixed(1)} Hz · Safe accel ${Math.round(ws.safe_accel_grid_mm_s2[rIdx]?.[cIdx] ?? 0).toLocaleString()} mm/s²`}
                  >
                    <span className="signoff-ws-hz">{hz.toFixed(1)}</span>
                    <span className="signoff-ws-sub">Hz</span>
                  </div>
                );
              })
            )}
          </div>
          )}
          {ws.xy_position_modeled !== false && ws.coords.length > 1 && (
            <div className="signoff-ws-foot">
              <span>
                Columns: X {pct(ws.coords[0])}–{pct(ws.coords[ws.coords.length - 1])} · Rows: Y{" "}
                {pct(ws.coords[0])}–{pct(ws.coords[ws.coords.length - 1])}
              </span>
              <span>Mid height</span>
            </div>
          )}
          {ws.z_sweep.length > 0 && (
            <>
              <div className="signoff-ws-foot">
                <span>Height sweep at bed centre</span>
              </div>
              <div className="signoff-z-grid">
                {ws.z_sweep.map((z) => (
                  <div
                    key={z.z_rel}
                    className={`signoff-ws-cell ${dropClass(z.min_axis_hz)}`}
                    title={`Z ${pct(z.z_rel)} · X ${z.shaper_x_hz.toFixed(1)} Hz, Y ${z.shaper_y_hz.toFixed(1)} Hz · Safe accel ${Math.round(z.safe_accel_mm_s2).toLocaleString()} mm/s²`}
                  >
                    <span className="signoff-ws-sub">Z {pct(z.z_rel)}</span>
                    <span className="signoff-ws-hz">{z.min_axis_hz.toFixed(1)}</span>
                    <span className="signoff-ws-sub">Hz</span>
                  </div>
                ))}
              </div>
            </>
          )}
          {ws.worst_xy_z_sweep && ws.worst_xy_z_sweep.length > 0 && (
            <div className="signoff-ws-foot">
              <span>
                Softest cell (X {pct(ws.worst_xy_z_sweep[0].u_rel)}, Y{" "}
                {pct(ws.worst_xy_z_sweep[0].v_rel)}) at{" "}
                {ws.worst_xy_z_sweep.map((z) => `Z ${pct(z.z_rel)}`).join(" / ")}
              </span>
              <span>
                <strong>
                  {ws.worst_xy_z_sweep.map((z) => z.min_axis_hz.toFixed(1)).join(" / ")} Hz
                </strong>
              </span>
            </div>
          )}
          <div className="signoff-ws-foot">
            <span>
              Worst-case safe accel:{" "}
              <strong>{Math.round(ws.worst_case_safe_accel_mm_s2).toLocaleString()} mm/s²</strong> at X{" "}
              {pct(ws.worst_accel_pos[0])} · Y {pct(ws.worst_accel_pos[1])} · Z{" "}
              {pct(ws.worst_accel_pos[2])}
            </span>
          </div>
        </div>
      )}

      {unc && unc.drivers.length > 0 && (
        <div className="signoff-block">
          <div className="signoff-subhead">
            <span>Top Uncertainty Drivers</span>
            <span className="signoff-submeta">Share of resonance / flex variance</span>
          </div>
          <div className="signoff-drivers">
            {unc.drivers.slice(0, 4).map((d) => (
              <div
                key={d.group_id}
                className="signoff-driver-row"
                title={`${d.how_to_tighten} (Trust ${d.trust_pct}%, ±${d.sigma_param_pct}% 1σ input · 90% effect: ±${d.delta_freq_hz.toFixed(1)} Hz, ±${Math.round(d.delta_accel_mm_s2).toLocaleString()} mm/s², ±${d.delta_deflection_um.toFixed(1)} µm)`}
              >
                <span className="signoff-driver-name">
                  {d.label}
                  <span className="signoff-driver-trust">{Math.round(d.trust_pct)}%</span>
                </span>
                <span className="signoff-driver-impact">
                  <strong>{d.variance_share_pct.toFixed(0)}%</strong> Hz ·{" "}
                  <strong>{(d.deflection_variance_share_pct ?? 0).toFixed(0)}%</strong> flex
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {checks.length > 0 && (
        <div className="signoff-checks">
          {checks.map((c) => (
            <div key={c.id} className="signoff-check-row" title={c.detail}>
              <span className={`signoff-check-dot ${c.status}`} aria-hidden="true" />
              <span className="signoff-check-title">{c.title}</span>
              <span className="signoff-check-val">{c.value}</span>
            </div>
          ))}
        </div>
      )}

      <table className="deep-verify-table">
        <thead>
          <tr>
            <th>Metric</th>
            <th className="num">Fast</th>
            <th className="num">Fine</th>
            <th className="num">Δ</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>f₁ Mode</td>
            <td className="num">{fast.fundamental_freq_hz.toFixed(1)} Hz</td>
            <td className="num">
              <strong>{deep.fundamental_freq_hz.toFixed(1)} Hz</strong>
            </td>
            <td className={`num ${deltaClass(convergence.freq_diff_pct)}`}>
              {fmtSignedPct(convergence.freq_diff_pct)}
            </td>
          </tr>
          <tr>
            <td>Nozzle–Bed Flex</td>
            <td className="num">{fast.combined_max_deflection_um.toFixed(1)} µm</td>
            <td className="num">
              <strong>{deep.combined_max_deflection_um.toFixed(1)} µm</strong>
            </td>
            <td className={`num ${deltaClass(convergence.deflection_diff_pct)}`}>
              {fmtSignedPct(convergence.deflection_diff_pct)}
            </td>
          </tr>
          <tr>
            <td>Safe Accel</td>
            <td className="num">{Math.round(fast.recommended_max_accel_mm_s2).toLocaleString()}</td>
            <td className="num">
              <strong>{Math.round(deep.recommended_max_accel_mm_s2).toLocaleString()}</strong>
            </td>
            <td className={`num ${deltaClass(convergence.accel_diff_pct)}`}>
              {fmtSignedPct(convergence.accel_diff_pct)}
            </td>
          </tr>
          <tr>
            <td>Panel Drum Mode</td>
            <td className="num">
              {fast.panel_drum_freq_hz > 0 ? `${fast.panel_drum_freq_hz.toFixed(1)} Hz` : "—"}
            </td>
            <td className={`num ${drumColor}`}>
              <strong>
                {deep.panel_drum_freq_hz > 0 ? `${deep.panel_drum_freq_hz.toFixed(1)} Hz` : "—"}
              </strong>
            </td>
            <td className="num">
              {hasDrum && fast.panel_drum_freq_hz > 0 && deep.panel_drum_freq_hz > 0
                ? `${drumDelta > 0 ? "+" : ""}${drumDelta.toFixed(1)} Hz`
                : "—"}
            </td>
          </tr>
          <tr>
            <td>Peak Panel Stress</td>
            <td className="num">
              {fast.panel_max_von_mises_mpa > 0
                ? `${fast.panel_max_von_mises_mpa.toFixed(2)} MPa`
                : "—"}
            </td>
            <td className="num">
              <strong>
                {deep.panel_max_von_mises_mpa > 0
                  ? `${deep.panel_max_von_mises_mpa.toFixed(2)} MPa`
                  : "—"}
              </strong>
            </td>
            <td className="num">
              {hasPanelStress && fast.panel_max_von_mises_mpa > 0 && deep.panel_max_von_mises_mpa > 0
                ? `${stressDelta > 0 ? "+" : ""}${stressDelta.toFixed(2)}`
                : "—"}
            </td>
          </tr>
        </tbody>
      </table>

      <div className="deep-verify-actions">
        <div className="impact-seg">
          <button
            type="button"
            className={`impact-seg-btn ${!viewingDeepSim ? "active" : ""}`}
 aria-pressed={!viewingDeepSim}
            onClick={() => onToggleViewingDeepSim(false)}
            title={`Show Fast ${fast_mesh_n}×${fast_mesh_n} MITC4 simulation in 3D viewport`}
          >
            Fast {fast_mesh_n}×{fast_mesh_n}
          </button>
          <button
            type="button"
            className={`impact-seg-btn ${viewingDeepSim ? "active" : ""}`}
 aria-pressed={viewingDeepSim}
            onClick={() => onToggleViewingDeepSim(true)}
            title={`Show Fine ${deep_mesh_n}×${deep_mesh_n} MITC4 simulation in 3D viewport`}
          >
            Fine {deep_mesh_n}×{deep_mesh_n}
          </button>
        </div>
        <button
          type="button"
          className="deep-verify-cta"
          onClick={onInspectDeepFea}
          title="Switch 3D viewport to Frame FEA with the fine MITC4 shell stress heatmap"
        >
          View Fine Mesh in Frame FEA
        </button>
      </div>
    </div>
  );
};

const BED_MODE_TYPE_LABELS: Record<BedPlateFeaData["plate_first_mode_type"], string> = {
  elastic_drum: "Elastic Drum",
  mount_bounce: "Mount Bounce",
  cantilever_flap: "Cantilever Flap",
};

const BED_RISK_META: Record<
  BedPlateFeaData["first_layer_risk"],
  { label: string; cls: string }
> = {
  low: { label: "Low", cls: "ok" },
  moderate: { label: "Moderate", cls: "warn" },
  high: { label: "High", cls: "bad" },
  critical: { label: "Critical", cls: "bad" },
};

const BedPlateFemStrip: React.FC<{
  bedFea: BedPlateFeaData;
  cmpBedFea?: BedPlateFeaData | null;
  ts?: ThermalStructuralReport | null;
  showBedWarp?: boolean;
  onInspectBedWarp?: () => void;
}> = ({ bedFea, cmpBedFea, ts, showBedWarp, onInspectBedWarp }) => {
  const riskMeta = BED_RISK_META[bedFea.first_layer_risk] ?? BED_RISK_META.low;
  const modeTypeLabel =
    BED_MODE_TYPE_LABELS[bedFea.plate_first_mode_type] ?? bedFea.plate_first_mode_type;
  const isHighRisk =
    bedFea.first_layer_risk === "high" || bedFea.first_layer_risk === "critical";
  const plateFlatnessUm = bedFea.plate_flatness_um ?? bedFea.total_flatness_error_um;
  const cmpPlateFlatnessUm = cmpBedFea
    ? (cmpBedFea.plate_flatness_um ?? cmpBedFea.total_flatness_error_um)
    : undefined;
  const rigidTiltUm = bedFea.rigid_tilt_um ?? 0;
  const mountSettlementUm = bedFea.mount_settlement_um ?? 0;
  const showTiltOrSettlement = rigidTiltUm >= 1.0 || mountSettlementUm >= 1.0;

  return (
    <div className={`bed-fem-strip risk-${bedFea.first_layer_risk}`}>
      <div className="bed-fem-head">
        <span>2D MITC4 Bed Plate FEM</span>
        <span className="badge-pill-row" style={{ gap: 4 }}>
          <span className={`badge-pill ${riskMeta.cls}`} style={{ padding: "1px 6px", fontSize: 11 }}>
            {riskMeta.label} Risk
          </span>
          {ts && ts.bed_buckling_taco_um > 30 && (
            <span className="badge-pill bad" style={{ padding: "1px 6px", fontSize: 11 }}>
              Taco Buckling!
            </span>
          )}
          {bedFea.mesh_bed_leveling_required && (
            <span className="badge-pill warn" style={{ padding: "1px 6px", fontSize: 11 }}>
              Mesh Leveling Needed
            </span>
          )}
        </span>
      </div>

      <div
        className="metric-kv"
        title={`Plane-removed non-planar plate flatness requiring mesh bed leveling: ${plateFlatnessUm.toFixed(1)} µm (raw peak-to-valley: ${bedFea.total_flatness_error_um.toFixed(1)} µm, rigid tilt: ${rigidTiltUm.toFixed(1)} µm, mount settlement: ${mountSettlementUm.toFixed(1)} µm)`}
      >
        <span>
          Plate Flatness Error
          {ts && (
            <span className="metric-sub-inline">
              Grad {ts.bed_gradient_bow_um.toFixed(1)} µm · Buckling {ts.bed_buckling_taco_um.toFixed(1)} µm
            </span>
          )}
        </span>
        <span
          className={`metric-val metric-val-with-delta ${
            isHighRisk
              ? "tag-bad"
              : bedFea.first_layer_risk === "moderate"
                ? "tag-warn"
                : "tag-ok"
          }`}
        >
          {cmpPlateFlatnessUm != null && (
            <InlineDeltaBadge
              oldVal={cmpPlateFlatnessUm}
              newVal={plateFlatnessUm}
              higherIsBetter={false}
              eps={0.2}
              formatDelta={(d) => `${d > 0 ? "+" : ""}${d.toFixed(1)} µm`}
            />
          )}
          <strong>{plateFlatnessUm.toFixed(1)} µm</strong>
        </span>
      </div>

      {showTiltOrSettlement && (
        <div
          className="metric-kv"
          title="Rigid bed tilt is removed by tramming / QGL / Z-tilt; uniform mount settlement is removed by Z-homing"
        >
          <span>Rigid Tilt / Settlement</span>
          <span className="metric-val">
            <span className={rigidTiltUm >= 120 ? "tag-warn" : ""}>
              Tilt {rigidTiltUm.toFixed(1)} µm
            </span>
            {mountSettlementUm >= 1.0 && ` · Settle ${mountSettlementUm.toFixed(1)} µm`}
          </span>
        </div>
      )}

      <div className="metric-kv">
        <span>2 N Probe Deflection</span>
        <span className="metric-val">
          Center: {bedFea.probe_center_deflection_um.toFixed(1)} µm · Corner:{" "}
          {bedFea.probe_corner_deflection_um.toFixed(1)} µm
        </span>
      </div>

      <div className="metric-kv">
        <span>Plate 1st Mode</span>
        <span className="metric-val">
          {bedFea.plate_first_mode_hz.toFixed(0)} Hz{" "}
          <span style={{ color: "var(--muted)", fontWeight: 500, fontSize: 11 }}>
            ({modeTypeLabel})
          </span>
        </span>
      </div>

      {bedFea.summary && <div className="bed-fem-summary">{bedFea.summary}</div>}

      {onInspectBedWarp && (
        <div className="bed-fem-actions">
          <button
            type="button"
            className={`deep-verify-cta ${showBedWarp ? "active" : ""}`}
 aria-pressed={showBedWarp}
            style={{ padding: "3px 8px", fontSize: 11 }}
            onClick={onInspectBedWarp}
            title="Highlight 3D MITC4 heated bed thermal & gravity warp surface in the viewport"
          >
            {showBedWarp ? "Viewing 3D Bed Warp" : "View 3D Bed Warp"}
          </button>
        </div>
      )}
    </div>
  );
};

const DOMINANT_FLEX_LABELS: Record<
  ToolheadFlexBreakdownData["dominant_flex_source"],
  string
> = {
  rail_xbeam: "X-beam & linear rail bending/torsion",
  carriage_body: "X-carriage backplate & toolhead body compliance",
  rail_carriage: "MGN linear rail & carriage bearing roll",
  toolhead_body: "toolhead body & carriage mount flexure",
  hotend_nozzle: "hotend mount, brace & nozzle stem bending",
};

const ToolheadFlexBreakdownCard: React.FC<{
  breakdown: ToolheadFlexBreakdownData;
  toolheadStructure?: ToolheadStructureGeometryData;
  specToolhead?: PrinterSpecData["toolhead"];
  toolheadPunchIn?: boolean;
  onToggleToolheadPunchIn?: () => void;
}> = ({
  breakdown,
  toolheadStructure,
  specToolhead,
  toolheadPunchIn,
  onToggleToolheadPunchIn,
}) => {
  const railPct = Math.max(0, breakdown.rail_and_xbeam_pct);
  const carBodyPct = Math.max(0, breakdown.carriage_and_body_pct);
  const hotendPct = Math.max(0, breakdown.hotend_and_nozzle_pct);
  const hasFourWay =
    breakdown.rail_carriage_pct != null && breakdown.toolhead_body_pct != null;
  const railCarPct = hasFourWay ? Math.max(0, breakdown.rail_carriage_pct!) : 0;
  const bodyOnlyPct = hasFourWay ? Math.max(0, breakdown.toolhead_body_pct!) : 0;

  const stageK = toolheadStructure?.stage_stiffnesses_n_per_mm;
  const railCarK =
    breakdown.rail_carriage_stiffness_n_per_mm ??
    stageK?.rail_carriage_n_per_mm ??
    breakdown.carriage_mount_stiffness_n_per_mm;
  const bodyK =
    breakdown.body_stiffness_n_per_mm ??
    stageK?.body_n_per_mm ??
    breakdown.carriage_mount_stiffness_n_per_mm;
  const hotendK =
    breakdown.hotend_stiffness_n_per_mm ?? stageK?.hotend_total_n_per_mm;

  const domSource = breakdown.dominant_flex_source;
  const dominantStageId: string = hasFourWay
    ? (() => {
        const candidates: Array<[string, number]> = [
          ["rail_xbeam", railPct],
          ["rail_carriage", railCarPct],
          ["toolhead_body", bodyOnlyPct],
          ["hotend_nozzle", hotendPct],
        ];
        candidates.sort((a, b) => b[1] - a[1]);
        return candidates[0][0];
      })()
    : domSource;

  const domPct =
    domSource === "hotend_nozzle"
      ? hotendPct
      : domSource === "rail_carriage" && hasFourWay
        ? railCarPct
        : domSource === "toolhead_body" && hasFourWay
          ? bodyOnlyPct
          : domSource === "carriage_body"
            ? carBodyPct
            : railPct;

  const fmtK = (k?: number | null): string =>
    k != null && Number.isFinite(k) && k > 0
      ? `${Math.round(k).toLocaleString()} N/mm`
      : "—";
  const fmtNum = (v?: number | null): string => {
    if (v == null || !Number.isFinite(v)) return "—";
    return Math.abs(v - Math.round(v)) < 0.05 ? `${Math.round(v)}` : v.toFixed(1);
  };

  const assembledMassG =
    breakdown.assembled_mass_g ??
    toolheadStructure?.assembled_toolhead_mass_g ??
    (specToolhead
      ? specToolhead.total_mass_g + (specToolhead.hotend.mass_g ?? 45)
      : undefined);
  const cogMm =
    breakdown.effective_cog_offset_mm ??
    toolheadStructure?.effective_cog_offset_mm ??
    specToolhead?.cog_offset_mm;
  const nozzleDropMm =
    breakdown.nozzle_drop_mm ??
    (specToolhead?.nozzle_offset_mm
      ? Math.abs(specToolhead.nozzle_offset_mm[2])
      : undefined);

  const xBeamSub = breakdown.sub_causes?.find((c) => c.id === "x_beam");
  const stageRows = hasFourWay
    ? [
        {
          id: "rail_xbeam",
          label: xBeamSub?.label || "X-Beam & Frame",
          dotClass: "th-flex-dot-beam",
          pct: railPct,
          stiffText: fmtK(xBeamSub?.stiffness_n_per_mm),
        },
        {
          id: "rail_carriage",
          label: "MGN Rail / Carriage",
          dotClass: "th-flex-dot-rail",
          pct: railCarPct,
          stiffText: fmtK(railCarK),
        },
        {
          id: "toolhead_body",
          label: "Toolhead Body",
          dotClass: "th-flex-dot-body",
          pct: bodyOnlyPct,
          stiffText: fmtK(bodyK),
        },
        {
          id: "hotend_nozzle",
          label: "Hotend & Brace",
          dotClass: "th-flex-dot-hotend",
          pct: hotendPct,
          stiffText: fmtK(hotendK),
        },
      ]
    : [
        {
          id: "rail_xbeam",
          label: xBeamSub?.label || "X-Beam & Frame",
          dotClass: "th-flex-dot-beam",
          pct: railPct,
          stiffText: fmtK(xBeamSub?.stiffness_n_per_mm),
        },
        {
          id: "carriage_body",
          label: "MGN Carriage & Body",
          dotClass: "th-flex-dot-body",
          pct: carBodyPct,
          stiffText: fmtK(breakdown.carriage_mount_stiffness_n_per_mm),
        },
        {
          id: "hotend_nozzle",
          label: "Hotend & Brace",
          dotClass: "th-flex-dot-hotend",
          pct: hotendPct,
          stiffText: fmtK(hotendK),
        },
      ];

  const barTitle = hasFourWay
    ? `X-Beam & Rail: ${railPct.toFixed(0)}% · MGN Carriage: ${railCarPct.toFixed(
        0
      )}% · Toolhead Body: ${bodyOnlyPct.toFixed(0)}% · Hotend & Brace: ${hotendPct.toFixed(0)}%`
    : `X-Beam & Rail: ${railPct.toFixed(0)}% · Carriage & Body: ${carBodyPct.toFixed(
        0
      )}% · Hotend & Brace: ${hotendPct.toFixed(0)}%`;

  const topSubCause = breakdown.sub_causes?.[0] ?? null;

  return (
    <div className="sens-breakdown-card th-flex-card">
      <div className="sens-breakdown-head">
        <span>Inside the Toolhead</span>
        <div className="th-flex-head-right">
          <span className="th-flex-tip-pill" title="Effective static XY stiffness at nozzle tip">
            Tip k {Math.round(breakdown.effective_tip_stiffness_n_per_mm).toLocaleString()} N/mm
          </span>
          {onToggleToolheadPunchIn && (
            <button
              type="button"
              className={`deep-verify-cta th-flex-punch-btn ${toolheadPunchIn ? "active" : ""}`}
              aria-pressed={toolheadPunchIn}
              onClick={onToggleToolheadPunchIn}
              title="Zoom camera directly onto the toolhead, carriage, and hotend to inspect local flex"
            >
              {toolheadPunchIn ? "Exit Punch-In" : "Punch-In View"}
            </button>
          )}
        </div>
      </div>

      <div className="subsystem-bar-track" title={barTitle}>
        {railPct > 0.5 && (
          <div
            className="subsystem-bar-seg th-flex-seg-beam"
            style={{ width: `${railPct}%` }}
          />
        )}
        {hasFourWay ? (
          <>
            {railCarPct > 0.5 && (
              <div
                className="subsystem-bar-seg th-flex-seg-rail"
                style={{ width: `${railCarPct}%` }}
              />
            )}
            {bodyOnlyPct > 0.5 && (
              <div
                className="subsystem-bar-seg th-flex-seg-body"
                style={{ width: `${bodyOnlyPct}%` }}
              />
            )}
          </>
        ) : (
          carBodyPct > 0.5 && (
            <div
              className="subsystem-bar-seg th-flex-seg-body"
              style={{ width: `${carBodyPct}%` }}
            />
          )
        )}
        {hotendPct > 0.5 && (
          <div
            className="subsystem-bar-seg th-flex-seg-hotend"
            style={{ width: `${hotendPct}%` }}
          />
        )}
      </div>

      <div className="th-flex-grid" role="table" aria-label="Toolhead compliance breakdown">
        {stageRows.map((row) => (
          <div
            key={row.id}
            role="row"
            className={`th-flex-row ${dominantStageId === row.id ? "is-dominant" : ""}`}
          >
            <span role="cell" className="th-flex-stage">
              <span className={`swatch-dot ${row.dotClass}`} />
              <span>{row.label}</span>
            </span>
            <span role="cell" className="th-flex-pct">
              {Math.round(row.pct)}%
            </span>
            <span role="cell" className="th-flex-stiff">
              {row.stiffText}
            </span>
          </div>
        ))}
      </div>

      <div className="th-flex-meta-strip">
        <span className="th-flex-meta-item">
          Mass (g) <strong>{fmtNum(assembledMassG)}</strong>
        </span>
        <span className="th-flex-meta-item">
          CoG Y / Z (mm){" "}
          <strong>
            {cogMm ? `${fmtNum(Math.abs(cogMm[1]))} / ${fmtNum(Math.abs(cogMm[2]))}` : "—"}
          </strong>
        </span>
        <span className="th-flex-meta-item">
          Nozzle Drop (mm) <strong>{fmtNum(nozzleDropMm)}</strong>
        </span>
      </div>

      {topSubCause ? (
        <div className="th-flex-hint">
          <div>
            <span className="sens-fix-prefix">Top limit:</span>{" "}
            <strong className="th-flex-cause-title">{topSubCause.label}</strong>
            {topSubCause.share_pct != null && (
              <span className="th-flex-cause-share"> ({Math.round(topSubCause.share_pct)}%)</span>
            )}
            {topSubCause.detail && <span> — {topSubCause.detail}</span>}
          </div>
          {topSubCause.action_hint && (
            <div className="th-flex-action">{topSubCause.action_hint}</div>
          )}
        </div>
      ) : (
        <div className="th-flex-hint">
          <span className="sens-fix-prefix">Main limit:</span>{" "}
          {DOMINANT_FLEX_LABELS[domSource] ?? domSource} ({Math.round(domPct)}% of local toolhead
          tip flex).
        </div>
      )}
    </div>
  );
};

const ModalVibrationModesCard: React.FC<{
  modes: ModeShapeData[];
  limitingSummary?: LimitingModalSummaryData;
  worstModeNumber?: number;
  viewMode: ViewMode;
  onViewModeChange: (m: ViewMode) => void;
  onInspectModalMode?: (modeIdx: number) => void;
  changed: boolean;
}> = ({
  modes,
  limitingSummary,
  worstModeNumber,
  viewMode,
  onViewModeChange,
  onInspectModalMode,
  changed,
}) => {
  const limitingModeNum = limitingSummary?.primary_structural_mode_number;

  const baseModes = modes.slice(0, 3);
  const shownMap = new Map<number, ModeShapeData>();
  for (const item of baseModes) {
    shownMap.set(item.mode_number, item);
  }
  const primaryLimitingMode = modes.find(
    (x) =>
      Boolean(x.is_primary_limiting_mode) ||
      (limitingModeNum != null && x.mode_number === limitingModeNum)
  );
  if (primaryLimitingMode && !shownMap.has(primaryLimitingMode.mode_number)) {
    shownMap.set(primaryLimitingMode.mode_number, primaryLimitingMode);
  }
  if (worstModeNumber != null && !shownMap.has(worstModeNumber)) {
    const worstExtra = modes.find((x) => x.mode_number === worstModeNumber);
    if (worstExtra) {
      shownMap.set(worstExtra.mode_number, worstExtra);
    }
  }
  const shownModes = Array.from(shownMap.values()).sort(
    (a, b) => a.mode_number - b.mode_number
  );

  return (
    <div className={`section-card ${changed ? "card-changed" : ""}`}>
      <div className="modal-summary-head">
        <span>Modal Vibration Modes</span>
        <button
          type="button"
          className={`deep-verify-cta ${viewMode === "modal" ? "active" : ""}`}
 aria-pressed={viewMode === "modal"}
          style={{ padding: "1px 7px", fontSize: 11 }}
          onClick={() => onViewModeChange("modal")}
          title="Animate 3D structural vibration modes in the viewport"
        >
          {viewMode === "modal" ? "Viewing Modal" : "View Modal 3D"}
        </button>
      </div>
      {limitingSummary && (
        <div className="modal-summary-belt-note">
          3D modes show pure structural vibration (locked belts); belt stretch (
          <strong>{Math.round(limitingSummary.compliance_share_pct.belt)}%</strong>) is coupled in
          series in the Klipper PSD &amp; Limiting Resonance summary. Click any mode to animate in
          3D.
        </div>
      )}
      <div className="modal-summary-list">
        {shownModes.map((m) => {
          const modeIndex = modes.findIndex((x) => x.mode_number === m.mode_number);
          const axisTag = m.excited_axis ? `${m.excited_axis}-Accel` : m.dominant_axis;
          const sevMeta = modeSeverityMeta(m.severity);
          const isWorst = worstModeNumber === m.mode_number;
          const isPrimaryLimiting =
            Boolean(m.is_primary_limiting_mode) ||
            (limitingModeNum != null && m.mode_number === limitingModeNum);
          const hasImpact = m.print_impact_score != null;
          const impCls = impactLevelCls(m.print_impact_level, m.print_impact_score);
          const share = m.subsystem_share_pct;
          const shareParts = share
            ? SUBSYSTEM_ORDER.map((k) => ({
                key: k,
                label: SUBSYSTEM_DISPLAY[k].short,
                color: SUBSYSTEM_DISPLAY[k].color,
                pct: Math.round(share[k] ?? 0),
              })).filter((p) => p.pct >= 3)
            : [];
          const topCauses = m.top_causes?.slice(0, 3) ?? [];
          const canInspect = Boolean(onInspectModalMode && modeIndex >= 0);

          const handleActivate = () => {
            if (onInspectModalMode && modeIndex >= 0) {
              onInspectModalMode(modeIndex);
            }
          };

          return (
            <div
              key={m.mode_number}
              className={`modal-mode-item ${m.severity ?? ""} ${
                isPrimaryLimiting ? "limiting-mode" : isWorst ? "worst-mode" : ""
              } ${canInspect ? "clickable" : ""}`}
              onClick={canInspect ? handleActivate : undefined}
              role={canInspect ? "button" : undefined}
              tabIndex={canInspect ? 0 : undefined}
              onKeyDown={
                canInspect
                  ? (e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        handleActivate();
                      }
                    }
                  : undefined
              }
              title={
                canInspect
                  ? `Click to animate Mode ${m.mode_number} (${m.frequency_hz.toFixed(1)} Hz) in 3D`
                  : undefined
              }
            >
              <div className="modal-mode-top">
                <span className="modal-mode-title">
                  {isPrimaryLimiting && (
                    <span className="limiting-mode-pill-inline">
                      LIMITING {limitingSummary?.limiting_axis ?? m.excited_axis ?? "XY"}
                    </span>
                  )}
                  {!isPrimaryLimiting && isWorst && (
                    <span className="worst-star-inline" title="Worst print-impact mode">
                      ★{" "}
                    </span>
                  )}
                  <strong>Mode {m.mode_number}</strong>{" "}
                  <span className="modal-mode-freq">
                    ({m.frequency_hz.toFixed(1)} Hz · {axisTag})
                  </span>
                  {m.mode_label && <span className="modal-mode-label"> — {m.mode_label}</span>}
                </span>
                <span className="modal-mode-badges">
                  {hasImpact ? (
                    <span className={`badge-pill modal-sev-pill ${impCls}`}>
                      Impact {Math.round(m.print_impact_score!)}/100
                    </span>
                  ) : (
                    sevMeta && (
                      <span className={`badge-pill modal-sev-pill ${sevMeta.cls}`}>
                        {sevMeta.label}
                      </span>
                    )
                  )}
                </span>
              </div>

              {((m.nozzle_whip_ratio ?? 0) >= 1.25 ||
                shareParts.length > 0 ||
                isWorst ||
                m.limiting_axis_share_pct != null) && (
                <div className="modal-mode-meta-row">
                  {isPrimaryLimiting && m.limiting_axis_share_pct != null && (
                    <span className="sens-badge limiting-share">
                      {Math.round(m.limiting_axis_share_pct)}% of{" "}
                      {limitingSummary?.limiting_axis ?? "axis"} struct flex
                    </span>
                  )}
                  {isWorst && !isPrimaryLimiting && (
                    <span className="sens-badge flex">★ Worst Print Mode</span>
                  )}
                  {(m.nozzle_whip_ratio ?? 0) >= 1.25 && (
                    <span
                      className={`sens-badge ${
                        (m.nozzle_whip_ratio ?? 0) >= 1.8 ? "flex" : "roi"
                      }`}
                      title="Nozzle-to-carriage dynamic whip amplification"
                    >
                      {m.nozzle_whip_ratio!.toFixed(1)}× whip
                    </span>
                  )}
                  {shareParts.length > 0 && (
                    <span className="modal-sub-pills">
                      {shareParts.map((sp, i) => (
                        <React.Fragment key={sp.key}>
                          {i > 0 && <span className="modal-sub-dot">·</span>}
                          <span className="modal-sub-pill-item">
                            <span className="swatch-dot" style={{ background: sp.color }} />
                            {sp.label} {sp.pct}%
                          </span>
                        </React.Fragment>
                      ))}
                    </span>
                  )}
                </div>
              )}

              {topCauses.length > 0 && (
                <div className="mode-causes-list">
                  {topCauses.map((cause) => {
                    const isMass = cause.kind === "mass";
                    return (
                      <div key={`${cause.id}-${cause.kind}`} className="mode-cause-row" title={cause.detail}>
                        <span className={`cause-kind-tag ${isMass ? "mass" : "flex"}`}>
                          {isMass ? "MASS" : "FLEX"}
                        </span>
                        <span className="mode-cause-label">{cause.label}</span>
                        <span className="mode-cause-stats">
                          <strong>{Math.round(cause.share_pct)}%</strong>
                          {cause.est_hz_gain > 0 && (
                            <span className="mode-cause-hz">
                              +{formatHzGain(cause.est_hz_gain)} Hz
                            </span>
                          )}
                        </span>
                      </div>
                    );
                  })}
                </div>
              )}

              {m.how_to_fix ? (
                <div className="modal-mode-fix">
                  <span className="sens-fix-prefix">Main limit:</span>{" "}
                  {m.how_to_fix.replace(/^\s*Main limit:\s*/i, "")}
                </div>
              ) : m.what_is_shaking ? (
                <div className="modal-mode-why">{m.what_is_shaking}</div>
              ) : null}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export const MetricsPanel: React.FC<Props> = ({
  sim,
  deepVerifyResult,
  viewingDeepSim,
  onToggleViewingDeepSim,
  onInspectDeepFea,
  onDismissDeepVerify,
  baselineSim,
  baselineBom,
  baselineLabel,
  prevSim,
  prevBom,
  lastChangeSummary,
  compareMode,
  onCompareModeChange,
  canUndo,
  onUndo,
  onResetToBaseline,
  onSetBaseline,
  solidFea,
  tetFea,
  activeSolidEngine,
  topoResult,
  viewMode,
  onViewModeChange,
  onInspectModalMode,
  bom,
  bomLoading,
  stockLengthMm,
  onStockLengthChange,
  onDownloadBomCsv,
  showBedWarp,
  onInspectBedWarp,
  toolheadPunchIn,
  onToggleToolheadPunchIn,
  onApplyFittedSpec,
  onInjectCustomBracket,
  error,
}) => {
  const [rightTab, setRightTab] = useState<"overview" | "shaper" | "deep" | "bom">("overview");
  const [bomCategoryFilter, setBomCategoryFilter] = useState<string>("all");
  const [sectionOpen, setSectionOpen] = useState<Record<string, boolean>>({
    fea_thermal_struct: true,
  });
  const [showInfoNotes, setShowInfoNotes] = useState<boolean>(false);

  const toggleSection = (key: string) => {
    setSectionOpen((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const isTetActive = Boolean(tetFea) && (activeSolidEngine === "tet10" || !solidFea);

  // Auto-focus Overview tab when a new topology, solid FEA, or deep verify result arrives
  useEffect(() => {
    if (topoResult) {
      setRightTab("overview");
    }
  }, [topoResult]);

  useEffect(() => {
    if (tetFea || solidFea) {
      setRightTab("overview");
    }
  }, [tetFea, solidFea]);

  useEffect(() => {
    if (deepVerifyResult) {
      setRightTab("overview");
    }
  }, [deepVerifyResult]);

  const bomCategories = useMemo(() => {
    if (!bom) return [];
    const seen = new Set<string>();
    const list: string[] = [];
    for (const c of CATEGORY_ORDER) {
      if (bom.items.some((it) => it.category === c)) {
        seen.add(c);
        list.push(c);
      }
    }
    for (const it of bom.items) {
      if (!seen.has(it.category)) {
        seen.add(it.category);
        list.push(it.category);
      }
    }
    return list;
  }, [bom]);

  const categoryBreakdown = useMemo(() => {
    if (!bom) return [];
    const costs = bom.category_costs_usd || {};
    const keys = [
      ...CATEGORY_ORDER.filter((k) => k in costs),
      ...Object.keys(costs).filter((k) => !CATEGORY_ORDER.includes(k)),
    ];
    return keys
      .map((cat) => ({ cat, cost: costs[cat] || 0 }))
      .filter((row) => row.cost > 0);
  }, [bom]);

  const filteredBomItems = useMemo(() => {
    if (!bom) return [];
    if (bomCategoryFilter === "all") return bom.items;
    return bom.items.filter((it) => it.category === bomCategoryFilter);
  }, [bom, bomCategoryFilter]);

  const hasComparison = Boolean(lastChangeSummary && (prevSim || baselineSim));
  const compareSim = hasComparison
    ? compareMode === "last"
      ? prevSim ?? baselineSim
      : baselineSim ?? prevSim
    : null;
  const compareBom = hasComparison
    ? compareMode === "last"
      ? prevSim
        ? prevBom
        : baselineBom
      : baselineSim
        ? baselineBom
        : prevBom
    : null;

  const impactMetrics = useMemo(() => {
    if (!sim || !compareSim) return [];
    return buildImpactMetrics(sim, compareSim, bom, compareBom);
  }, [sim, compareSim, bom, compareBom]);

  const impactSummaryText = useMemo(() => {
    if (impactMetrics.length === 0) return "";
    return buildPlainEnglishSummary(impactMetrics);
  }, [impactMetrics]);

  if (!sim && error) {
    return (
      <aside className="sidebar right">
        <div className="sidebar-scroll">
          <div className="diag-item bad">
            <strong>Simulation Error:</strong> {error}
          </div>
        </div>
      </aside>
    );
  }

  if (!sim) {
    return (
      <aside className="sidebar right">
        <div className="sidebar-scroll">Running multiphysics simulation...</div>
      </aside>
    );
  }

  const { scores, static_fea: fea, modal_fea: mod, kinematics: kin, thermal: th, virtual_print: vp } = sim;
  const curInertialFlex = Math.max(fea.accel_deflection_x_um, fea.accel_deflection_y_um);
  const cmpInertialFlex = compareSim
    ? Math.max(
        compareSim.static_fea.accel_deflection_x_um,
        compareSim.static_fea.accel_deflection_y_um
      )
    : undefined;

  // Determine which detail cards moved vs comparison
  const structuralChanged =
    compareSim != null &&
    (Math.abs(curInertialFlex - (cmpInertialFlex ?? curInertialFlex)) >= 0.05 ||
      Math.abs(fea.combined_max_deflection_um - compareSim.static_fea.combined_max_deflection_um) >= 0.05 ||
      Math.abs(
        (fea.nozzle_absolute_deflection_um ?? 0) -
          (compareSim.static_fea.nozzle_absolute_deflection_um ?? 0)
      ) >= 0.05 ||
      Math.abs(
        (fea.bed_deflection_um ?? 0) - (compareSim.static_fea.bed_deflection_um ?? 0)
      ) >= 0.05 ||
      Math.abs(
        (fea.beam_center_deflection_um ?? 0) - (compareSim.static_fea.beam_center_deflection_um ?? 0)
      ) >= 0.05 ||
      Math.abs(
        (fea.nozzle_twist_amplification_pct ?? 0) -
          (compareSim.static_fea.nozzle_twist_amplification_pct ?? 0)
      ) >= 0.5 ||
      Math.abs(
        (fea.panel_max_von_mises_mpa ?? 0) -
          (compareSim.static_fea.panel_max_von_mises_mpa ?? 0)
      ) >= 0.02 ||
      Math.abs(scores.structural - compareSim.scores.structural) >= 0.05 ||
      Boolean(sim.geometry.panels?.bolted_to_gantry) !==
        Boolean(compareSim.geometry.panels?.bolted_to_gantry) ||
      Boolean(sim.geometry.panels?.is_flying_gantry) !==
        Boolean(compareSim.geometry.panels?.is_flying_gantry) ||
      Math.abs(
        (sim.geometry.panels?.effective_shear_gt_kn_mm ?? 0) -
          (compareSim.geometry.panels?.effective_shear_gt_kn_mm ?? 0)
      ) >= 0.01 ||
      (sim.geometry.extrusion_filling?.id ?? "none") !==
        (compareSim.geometry.extrusion_filling?.id ?? "none") ||
      (sim.geometry.linear_rails?.x_rail_mount ?? "front") !==
        (compareSim.geometry.linear_rails?.x_rail_mount ?? "front") ||
      (sim.geometry.linear_rails?.y_rail_mount ?? "top") !==
        (compareSim.geometry.linear_rails?.y_rail_mount ?? "top") ||
      (sim.geometry.linear_rails?.x_rail_profile ?? "") !==
        (compareSim.geometry.linear_rails?.x_rail_profile ?? "") ||
      (sim.geometry.linear_rails?.y_rail_profile ?? "") !==
        (compareSim.geometry.linear_rails?.y_rail_profile ?? "") ||
      Boolean(sim.geometry.linear_rails?.dual_x_rails) !==
        Boolean(compareSim.geometry.linear_rails?.dual_x_rails) ||
      Math.abs(
        (fea.toolhead_flex_breakdown?.hotend_and_nozzle_pct ?? 0) -
          (compareSim.static_fea.toolhead_flex_breakdown?.hotend_and_nozzle_pct ?? 0)
      ) >= 1 ||
      Math.abs(
        (fea.toolhead_flex_breakdown?.effective_tip_stiffness_n_per_mm ?? 0) -
          (compareSim.static_fea.toolhead_flex_breakdown?.effective_tip_stiffness_n_per_mm ?? 0)
      ) >= 1);
  const modalChanged =
    compareSim != null &&
    (Math.abs(mod.fundamental_freq_hz - compareSim.modal_fea.fundamental_freq_hz) >= 0.05 ||
      Math.abs(
        (mod.nozzle_pitch_twist_freq_hz ?? 0) - (compareSim.modal_fea.nozzle_pitch_twist_freq_hz ?? 0)
      ) >= 0.05 ||
      Math.abs(
        (mod.panel_drum_freq_hz ?? 0) - (compareSim.modal_fea.panel_drum_freq_hz ?? 0)
      ) >= 0.1 ||
      Math.abs(
        (mod.frame_damping_ratio ?? 0) - (compareSim.modal_fea.frame_damping_ratio ?? 0)
      ) >= 0.002 ||
      Math.abs(mod.recommended_max_accel_mm_s2 - compareSim.modal_fea.recommended_max_accel_mm_s2) >= 15 ||
      Math.abs(
        (sim.motion_limits?.achievable_max_accel_mm_s2 ?? mod.recommended_max_accel_mm_s2) -
          (compareSim.motion_limits?.achievable_max_accel_mm_s2 ??
            compareSim.modal_fea.recommended_max_accel_mm_s2)
      ) >= 15 ||
      (sim.motion_limits?.accel_limiting_factor ?? "") !==
        (compareSim.motion_limits?.accel_limiting_factor ?? "") ||
      Math.abs(scores.dynamics - compareSim.scores.dynamics) >= 0.05 ||
      Math.abs(kin.torque_safety_margin - compareSim.kinematics.torque_safety_margin) >= 0.05 ||
      Math.abs(
        (kin.motor_stiffness_n_per_mm ?? 0) - (compareSim.kinematics.motor_stiffness_n_per_mm ?? 0)
      ) >= 0.1 ||
      Math.abs(
        (kin.bracket_stiffness_n_per_mm ?? 0) - (compareSim.kinematics.bracket_stiffness_n_per_mm ?? 0)
      ) >= 0.5 ||
      (kin.bracket_material ?? "") !== (compareSim.kinematics.bracket_material ?? "") ||
      Boolean(kin.bracket_creep_risk) !== Boolean(compareSim.kinematics.bracket_creep_risk) ||
      Math.abs(
        (kin.drive_stiffness_n_per_mm ?? 0) - (compareSim.kinematics.drive_stiffness_n_per_mm ?? 0)
      ) >= 0.1 ||
      Math.abs(
        (kin.motor_voltage_efficiency_pct ?? 0) -
          (compareSim.kinematics.motor_voltage_efficiency_pct ?? 0)
      ) >= 0.5 ||
      Math.abs((kin.motor_heat_w_each ?? 0) - (compareSim.kinematics.motor_heat_w_each ?? 0)) >= 0.1 ||
      (kin.drive_mode ?? "") !== (compareSim.kinematics.drive_mode ?? "") ||
      Math.abs((kin.belt_pluck_freq_150mm_hz ?? 0) - (compareSim.kinematics.belt_pluck_freq_150mm_hz ?? 0)) >= 1 ||
      (mod.accel_bottleneck?.primary_source ?? "") !==
        (compareSim.modal_fea.accel_bottleneck?.primary_source ?? "") ||
      (mod.accel_bottleneck?.limiting_axis ?? "") !==
        (compareSim.modal_fea.accel_bottleneck?.limiting_axis ?? "") ||
      Math.abs(
        (mod.accel_bottleneck?.primary_share_pct ?? 0) -
          (compareSim.modal_fea.accel_bottleneck?.primary_share_pct ?? 0)
      ) >= 1 ||
      Math.abs(
        (sim.weight_shaper_gains?.limiting_moving_mass_g ?? 0) -
          (compareSim.weight_shaper_gains?.limiting_moving_mass_g ?? 0)
      ) >= 1);
  const thermalChanged =
    compareSim != null &&
    (Math.abs(th.max_volumetric_flow_mm3_s - compareSim.thermal.max_volumetric_flow_mm3_s) >= 0.05 ||
      Math.abs(scores.thermal - compareSim.scores.thermal) >= 0.05);
  const ts = sim.thermal_structural;
  const cmpTs = compareSim?.thermal_structural;
  const bedFea = sim.bed_plate_fea ?? null;
  const cmpBedFea = compareSim?.bed_plate_fea ?? null;
  const thermalStructChanged =
    compareSim != null &&
    ((ts != null &&
      (cmpTs == null ||
        Math.abs(ts.gantry_bimetallic_bow_um - cmpTs.gantry_bimetallic_bow_um) >= 0.2 ||
        Math.abs(ts.bed_gradient_bow_um - cmpTs.bed_gradient_bow_um) >= 0.2 ||
        Math.abs(ts.bed_buckling_taco_um - cmpTs.bed_buckling_taco_um) >= 0.2 ||
        Math.abs(ts.frame_z_drift_um - cmpTs.frame_z_drift_um) >= 0.2 ||
        Math.abs(ts.total_thermal_z_error_um - cmpTs.total_thermal_z_error_um) >= 0.2 ||
        Math.abs(ts.bracket_modulus_retention_pct - cmpTs.bracket_modulus_retention_pct) >= 0.5 ||
        Math.abs(ts.rail_newmark_gamma - cmpTs.rail_newmark_gamma) >= 0.005 ||
        Math.abs(ts.rail_carriage_roll_nm_per_rad - cmpTs.rail_carriage_roll_nm_per_rad) >= 10 ||
        Math.abs(
          (kin.tooth_mesh_stiffness_n_per_mm ?? 0) -
            (compareSim.kinematics.tooth_mesh_stiffness_n_per_mm ?? 0)
        ) >= 1 ||
        Math.abs(
          (kin.idler_shaft_deflection_um ?? 0) -
            (compareSim.kinematics.idler_shaft_deflection_um ?? 0)
        ) >= 0.1 ||
        Math.abs(
          (kin.bearing_play_um ?? 0) - (compareSim.kinematics.bearing_play_um ?? 0)
        ) >= 0.1)) ||
      (bedFea != null &&
        (cmpBedFea == null ||
          Math.abs(
            (bedFea.plate_flatness_um ?? bedFea.total_flatness_error_um) -
              (cmpBedFea.plate_flatness_um ?? cmpBedFea.total_flatness_error_um)
          ) >= 0.2 ||
          Math.abs((bedFea.rigid_tilt_um ?? 0) - (cmpBedFea.rigid_tilt_um ?? 0)) >= 0.5 ||
          Math.abs(bedFea.plate_first_mode_hz - cmpBedFea.plate_first_mode_hz) >= 0.5)));
  const printChanged =
    compareSim != null &&
    (Math.abs(vp.total_time_s - compareSim.virtual_print.total_time_s) >= 0.05 ||
      Math.abs(scores.print_quality - compareSim.scores.print_quality) >= 0.05);

  const baselineShortTag = baselineLabel.toLowerCase().startsWith("stock") ? "vs. Stock" : "vs. Base";

  // Mini SVG polyline for hotend heat-up curve
  const pts = th.heatup_curve;
  const maxT = Math.max(...pts.map((p) => p.time_s), 1);
  const maxC = Math.max(...pts.map((p) => p.temp_c), 230);
  const svgPoints = pts
    .map((p) => `${(p.time_s / maxT) * 260},${60 - ((p.temp_c - 20) / (maxC - 20)) * 54}`)
    .join(" ");

  // 1D FVM Axial Hotend Temperature Profile SVG
  const axPts = th.axial_profile || [];
  const maxZ = Math.max(...axPts.map((p) => p.z_mm), 40);
  const axSvgPoints = axPts
    .map((p) => `${(p.z_mm / maxZ) * 260},${62 - ((p.temp_c - 20) / (maxC - 20)) * 54}`)
    .join(" ");
  const tgY = 62 - ((th.glass_transition_c - 20) / (maxC - 20)) * 54;

  const totalCategoryCost = Math.max(
    bom?.total_cost_usd ?? 0,
    categoryBreakdown.reduce((acc, r) => acc + r.cost, 0),
    1
  );

  // Tier 1 & Tier 2 derived summary helpers
  const curCost =
    bom?.total_cost_usd ??
    sim.bom_total_cost_usd ??
    sim.geometry.mass_summary.estimated_material_cost_usd;
  const cmpCost =
    compareBom?.total_cost_usd ??
    compareSim?.bom_total_cost_usd ??
    compareSim?.geometry.mass_summary.estimated_material_cost_usd;

  const limAxis: "X" | "Y" =
    mod.limiting_modal_summary?.limiting_axis ??
    mod.accel_bottleneck?.limiting_axis ??
    (mod.shaper_x && mod.shaper_y
      ? mod.shaper_x.max_accel_mm_s2 <= mod.shaper_y.max_accel_mm_s2
        ? "X"
        : "Y"
      : "Y");
  const limShaper = limAxis === "X" ? mod.shaper_x : mod.shaper_y;
  const limShaperName =
    mod.limiting_modal_summary?.recommended_shaper ??
    limShaper?.recommended_shaper ??
    "ZV";
  const limEffHz =
    mod.limiting_modal_summary?.effective_freq_hz ??
    limShaper?.effective_freq_hz ??
    mod.fundamental_freq_hz;
  const xEffStr = mod.shaper_x ? Math.round(mod.shaper_x.effective_freq_hz).toString() : "—";
  const yEffStr = mod.shaper_y ? Math.round(mod.shaper_y.effective_freq_hz).toString() : "—";
  const ml = sim.motion_limits;
  const cmpMl = compareSim?.motion_limits;
  const achievableAccel = ml?.achievable_max_accel_mm_s2 ?? mod.recommended_max_accel_mm_s2;
  const cmpAchievableAccel = compareSim
    ? (cmpMl?.achievable_max_accel_mm_s2 ?? compareSim.modal_fea.recommended_max_accel_mm_s2)
    : undefined;
  const accelLimFactor = ml?.accel_limiting_factor;
  const accelLimPillLabel =
    accelLimFactor === "motor"
      ? "Motor-Limited"
      : accelLimFactor === "belt"
        ? "Belt-Limited"
        : accelLimFactor === "shaper"
          ? "Shaper-Limited"
          : null;
  const motorSpeedExceeded = Boolean(
    ml && ml.configured_max_velocity_mm_s > ml.motor_max_velocity_mm_s + 0.5
  );
  const totalDefects =
    vp.underextrusion_segments +
    vp.ringing_segments +
    vp.step_skip_segments +
    (vp.smoothing_segments ?? 0) +
    (vp.belt_slip_segments ?? 0);

  const changedImpactMetrics = impactMetrics.filter((m) => m.status !== "same");

  const classifiedDiags = sim.diagnostics.map((d) => {
    const lower = d.toLowerCase();
    const cls: "bad" | "warn" | "info" =
      d.startsWith("COLLISION") ||
      lower.includes("overtension") ||
      lower.includes("overcurrent") ||
      lower.includes("overheat")
        ? "bad"
        : lower.includes("risk") ||
            lower.includes("low") ||
            lower.includes("detected") ||
            lower.includes("bending") ||
            lower.includes("double-shear") ||
            lower.includes("twist")
          ? "warn"
          : "info";
    return { text: prettifyDiagnostic(d), cls };
  });
  // Critical items first, then warnings, each keeping the solver's order.
  const alertDiags = [
    ...classifiedDiags.filter((d) => d.cls === "bad"),
    ...classifiedDiags.filter((d) => d.cls === "warn"),
  ];
  const infoDiags = classifiedDiags.filter((d) => d.cls === "info");

  const effDrive = kin.drive_mode || sim.spec.gantry.drive_mode || "2wd";
  const hasMetrologyData = Boolean(
    sim.time_domain_metrology ||
      (sim.workspace_frf && (sim.workspace_frf.min_axis_freq_grid_hz?.length ?? 0) > 0) ||
      sim.rcsa
  );
  const hasSensitivityData = Boolean(
    sim.sensitivity &&
      ((sim.sensitivity.top_stiffen_targets?.length ?? 0) > 0 ||
        (sim.sensitivity.top_lighten_targets?.length ?? 0) > 0 ||
        Boolean(sim.sensitivity.summary_diagnosis) ||
        Boolean(sim.sensitivity.subsystem_breakdown))
  );
  const hasWeightGains = Boolean(
    sim.weight_shaper_gains && sim.weight_shaper_gains.items.length > 0
  );

  const visibleDeepKeys: string[] = ["fea_thermal_struct", "modes_kinematics"];
  if (hasWeightGains) visibleDeepKeys.push("weight_potential");
  if (hasMetrologyData) visibleDeepKeys.push("metrology_frf");
  if (hasSensitivityData) visibleDeepKeys.push("sensitivity");
  visibleDeepKeys.push("thermal_print");

  const allExpanded =
    visibleDeepKeys.length > 0 && visibleDeepKeys.every((k) => Boolean(sectionOpen[k]));

  const feaAccordionBadge = `${curInertialFlex.toFixed(1)} µm · SF ${fea.min_safety_factor.toFixed(1)}×${
    ts ? ` · ${ts.total_thermal_z_error_um.toFixed(0)} µm Z` : ""
  }`;
  const modesAccordionBadge = `${effDrive.toUpperCase()} · ${kin.torque_safety_margin.toFixed(1)}× torque${
    kin.belt_pluck_freq_150mm_hz != null
      ? ` · ${Math.round(kin.belt_pluck_freq_150mm_hz)} Hz belt`
      : ""
  }`;
  const weightAccordionBadge =
    sim.weight_shaper_gains && hasWeightGains
      ? `Up to +${sim.weight_shaper_gains.total_potential_hz_gain.toFixed(1)} Hz (−${Math.round(sim.weight_shaper_gains.total_potential_cut_g)}g)`
      : undefined;
  const metrologyBadgeParts: string[] = [];
  if (sim.time_domain_metrology) {
    metrologyBadgeParts.push(
      `G_XY ${(sim.time_domain_metrology.circularity_gxy_um ?? 0).toFixed(1)} µm`
    );
  }
  if (sim.rcsa) {
    metrologyBadgeParts.push(`Whip ${sim.rcsa.nozzle_to_sensor_amp_ratio.toFixed(2)}×`);
  } else if (sim.workspace_frf) {
    metrologyBadgeParts.push(
      `${sim.workspace_frf.min_freq_hz.toFixed(0)}–${sim.workspace_frf.max_freq_hz.toFixed(0)} Hz`
    );
  }
  const metrologyAccordionBadge = metrologyBadgeParts.join(" · ") || undefined;

  const sensPrimary = sim.sensitivity?.primary_problem_subsystem;
  const sensPrimaryPct =
    sensPrimary && sim.sensitivity?.subsystem_breakdown?.[sensPrimary]
      ? Math.round(sim.sensitivity.subsystem_breakdown[sensPrimary].compliance_share_pct)
      : null;
  const sensitivityAccordionBadge = sensPrimary
    ? `#1 ${SUBSYSTEM_DISPLAY[sensPrimary]?.short ?? sensPrimary}${
        sensPrimaryPct != null ? ` (${sensPrimaryPct}% flex)` : ""
      }`
    : sim.sensitivity?.top_stiffen_targets?.[0]
      ? `#1 ${sim.sensitivity.top_stiffen_targets[0].label ?? sim.sensitivity.top_stiffen_targets[0].id}`
      : undefined;
  const thermalPrintAccordionBadge = `${th.max_volumetric_flow_mm3_s.toFixed(1)} mm³/s · ${vp.total_time_s.toFixed(0)}s print`;

  return (
    <aside className="sidebar right">
      <div className="tabs" role="tablist" aria-label="Results" onKeyDown={onTabListKeyDown}>
        <button
          type="button"
          role="tab"
          aria-selected={rightTab === "overview"}
          tabIndex={rightTab === "overview" ? 0 : -1}
          className={`tab-btn ${rightTab === "overview" ? "active" : ""}`}
          style={{ whiteSpace: "nowrap", padding: "9px 4px" }}
          onClick={() => setRightTab("overview")}
          title="Composite Score, KPIs, Limiting Resonance & Diagnostics"
        >
          Overview
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={rightTab === "shaper"}
          tabIndex={rightTab === "shaper" ? 0 : -1}
          className={`tab-btn ${rightTab === "shaper" ? "active" : ""}`}
          style={{ whiteSpace: "nowrap", padding: "9px 4px" }}
          onClick={() => setRightTab("shaper")}
          title="Klipper Input Shaper PSD Graphs, ADXL345 Overlay & Modal Modes"
        >
          Shaper
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={rightTab === "deep"}
          tabIndex={rightTab === "deep" ? 0 : -1}
          className={`tab-btn ${rightTab === "deep" ? "active" : ""}`}
          style={{ whiteSpace: "nowrap", padding: "9px 4px" }}
          onClick={() => setRightTab("deep")}
          title="Static FEA, Thermal-Structural, Kinematics, Metrology, Sensitivity & Thermal"
        >
          Deep
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={rightTab === "bom"}
          tabIndex={rightTab === "bom" ? 0 : -1}
          className={`tab-btn ${rightTab === "bom" ? "active" : ""}`}
          style={{ whiteSpace: "nowrap", padding: "9px 4px" }}
          onClick={() => setRightTab("bom")}
          title={
            bom
              ? `BOM & Cut-List ($${Math.round(bom.total_cost_usd).toLocaleString()})`
              : "BOM & Cut-List"
          }
        >
          BOM
          {bom && (
            <span style={{ fontSize: 11, opacity: 0.78, marginLeft: 3, fontWeight: 500 }}>
              ${Math.round(bom.total_cost_usd)}
            </span>
          )}
        </button>
      </div>

      <div className="sidebar-scroll">
        {error && (
          <div className="diag-item bad">
            <strong>Notice:</strong> {error}
          </div>
        )}

        {/* Active transient verification / solid part / topology cards stay visible across tabs */}
        {deepVerifyResult && (
          <DeepVerifyCard
            result={deepVerifyResult}
            viewingDeepSim={viewingDeepSim}
            onToggleViewingDeepSim={onToggleViewingDeepSim}
            onInspectDeepFea={onInspectDeepFea}
            onDismiss={onDismissDeepVerify}
          />
        )}
        {viewMode === "topology" && topoResult && <TopologyOptCard topo={topoResult} />}
        {viewMode === "solid_part" && isTetActive && tetFea && (
          <TetFEACard
            tet={tetFea}
            activeCustomBracket={sim.spec?.gantry?.custom_bracket_override ?? null}
            onInjectCustomBracket={onInjectCustomBracket}
          />
        )}
        {viewMode === "solid_part" && !isTetActive && solidFea && (
          <SolidHexFEACard
            solidFea={solidFea}
            activeCustomBracket={sim.spec?.gantry?.custom_bracket_override ?? null}
            onInjectCustomBracket={onInjectCustomBracket}
          />
        )}

        {rightTab === "overview" && (
          <>

            {/* Live Impact Card (changed metrics only) OR Compact Baseline Tip */}
            {hasComparison && compareSim ? (
              <div className="section-card impact-card">
                <div className="impact-header">
                  <div className="impact-title-row">
                    <span className="section-title" style={{ margin: 0 }}>
                      Change Impact
                    </span>
                    <div className="impact-actions">
                      <div className="impact-seg">
                        <button
                          type="button"
                          className={`impact-seg-btn ${compareMode === "last" ? "active" : ""}`}
 aria-pressed={compareMode === "last"}
                          onClick={() => onCompareModeChange("last")}
                          title="Compare against the design right before your last edit"
                        >
                          vs. Last
                        </button>
                        <button
                          type="button"
                          className={`impact-seg-btn ${compareMode === "baseline" ? "active" : ""}`}
 aria-pressed={compareMode === "baseline"}
                          onClick={() => onCompareModeChange("baseline")}
                          title={`Compare cumulative gains against ${baselineLabel}`}
                        >
                          {baselineShortTag}
                        </button>
                      </div>
                      <button
                        type="button"
                        className="impact-mini-btn"
                        disabled={!canUndo}
                        onClick={onUndo}
                        title="Undo last change"
                      >
                        Undo
                      </button>
                      <button
                        type="button"
                        className="impact-mini-btn"
                        onClick={onResetToBaseline}
                        title={`Reset all changes back to ${baselineLabel}`}
                      >
                        Reset
                      </button>
                    </div>
                  </div>
                  <div className="impact-change-pill">
                    <span className="impact-change-label">
                      {compareMode === "baseline" ? `Total vs. ${baselineLabel}` : lastChangeSummary}
                    </span>
                    <button
                      type="button"
                      className="impact-pin-btn"
                      onClick={onSetBaseline}
                      title="Save current configuration as your new baseline for future comparisons"
                    >
                      Set as Baseline
                    </button>
                  </div>
                  {compareMode === "baseline" && lastChangeSummary && (
                    <div className="impact-sub-note">Latest edit: {lastChangeSummary}</div>
                  )}
                </div>

                <div className="impact-summary">{impactSummaryText}</div>

                {changedImpactMetrics.length > 0 ? (
                  <div className="impact-grid">
                    {changedImpactMetrics.map((m) => (
                      <div key={m.key} className={`impact-tile ${m.status}`}>
                        <div className="impact-tile-top">
                          <span className="impact-tile-label">
                            <span className={`impact-dot ${m.status}`} />
                            {m.label}
                          </span>
                          <span className={`impact-badge ${m.status}`}>{m.deltaText}</span>
                        </div>
                        <div className="impact-tile-sub">
                          {m.oldText} → <strong>{m.newText}</strong>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div style={{ fontSize: 11, color: "var(--muted)" }}>No measurable delta</div>
                )}
              </div>
            ) : (
              <div className="welcome-tip-card">
                <div className="welcome-tip-text">
                  Loaded <strong>{baselineLabel}</strong> — edit <strong>Design Controls</strong> to see live stiffness, ringing, flow &amp; cost impact.
                </div>
              </div>
            )}

            {/* Tier 1: Unified Overview Card (Composite Score + Subsystem Bars + 6 Headline KPIs) */}
            <div className="section-card" style={{ gap: 10 }}>
              <div className="score-hero">
                <div>
                  <div style={{ fontSize: 11, color: "var(--muted)", textTransform: "uppercase", fontWeight: 600 }}>
                    Composite Engineering Score
                  </div>
                  <div style={{ fontSize: 13, fontWeight: 500, marginTop: 2 }}>{sim.printer_name}</div>
                </div>
                <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 3 }}>
                  <div className="score-number" style={{ color: scoreColor(scores.composite) }}>
                    {scores.composite.toFixed(1)}
                  </div>
                  {compareSim && (
                    <InlineDeltaBadge
                      oldVal={compareSim.scores.composite}
                      newVal={scores.composite}
                      higherIsBetter={true}
                      formatDelta={(d) => `${d > 0 ? "+" : ""}${d.toFixed(1)}`}
                    />
                  )}
                </div>
              </div>

              <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
                {(
                  [
                    ["Structural (FEA)", scores.structural, compareSim?.scores.structural],
                    ["Dynamics & Shaper", scores.dynamics, compareSim?.scores.dynamics],
                    ["Hotend Thermal", scores.thermal, compareSim?.scores.thermal],
                    ["Virtual Print Quality", scores.print_quality, compareSim?.scores.print_quality],
                    ["Mass & Packaging", scores.efficiency, compareSim?.scores.efficiency],
                  ] as Array<[string, number, number | undefined]>
                ).map(([label, val, oldVal]) => (
                  <div className="bar-row" key={label}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 11 }}>
                      <span style={{ color: "var(--muted)" }}>{label}</span>
                      <span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
                        {oldVal !== undefined && (
                          <InlineDeltaBadge
                            oldVal={oldVal}
                            newVal={val}
                            higherIsBetter={true}
                            formatDelta={(d) => `${d > 0 ? "+" : ""}${d.toFixed(1)}`}
                          />
                        )}
                        <strong>{val.toFixed(1)}</strong>
                      </span>
                    </div>
                    <div className="bar-track">
                      <div className="bar-fill" style={{ width: `${val}%`, background: scoreColor(val) }} />
                    </div>
                  </div>
                ))}
              </div>

              <div className="kpi-grid headline-kpi-grid">
                <div
                  className="kpi-box"
                  title={
                    ml
                      ? `Achievable Max Accel = min(Shaper ${Math.round(ml.shaper_max_accel_mm_s2).toLocaleString()}, Motor ${Math.round(ml.motor_max_accel_mm_s2).toLocaleString()}, Belt ${Math.round(ml.belt_max_accel_mm_s2).toLocaleString()}) mm/s²\n${limAxis} Shaper: ${limShaperName} @ ${Math.round(limEffHz)} Hz${ml.shaper_l1_accel_factor > 1.01 ? ` (L1 spike factor ${ml.shaper_l1_accel_factor.toFixed(2)}×)` : ""}`
                      : `Safe Max Accel (${limAxis}: ${limShaperName} @ ${Math.round(limEffHz)} Hz)`
                  }
                >
                  <div className="kpi-head-row">
                    <span className="kpi-label">Achievable Max Accel</span>
                    {accelLimPillLabel && accelLimFactor && (
                      <span className={`kpi-limit-pill ${accelLimFactor}`}>
                        {accelLimPillLabel}
                      </span>
                    )}
                  </div>
                  <div className="kpi-val-row">
                    <span
                      className="kpi-value"
                      style={{
                        color:
                          accelLimFactor === "motor"
                            ? "var(--amber)"
                            : accelLimFactor === "belt"
                              ? "var(--red)"
                              : "var(--accent)",
                      }}
                    >
                      {Math.round(achievableAccel).toLocaleString()} mm/s²
                    </span>
                    {cmpAchievableAccel != null && (
                      <InlineDeltaBadge
                        oldVal={cmpAchievableAccel}
                        newVal={achievableAccel}
                        higherIsBetter={true}
                        eps={15}
                        formatDelta={(d) => `${d > 0 ? "+" : ""}${Math.round(d).toLocaleString()}`}
                      />
                    )}
                  </div>
                  {ml ? (
                    <div className="kpi-tri-split">
                      <span className={accelLimFactor === "shaper" ? "lim shaper" : ""}>
                        Shaper {formatCompactAccel(ml.shaper_max_accel_mm_s2)}
                      </span>
                      <span className="kpi-tri-dot">·</span>
                      <span className={accelLimFactor === "motor" ? "lim motor" : ""}>
                        Motor {formatCompactAccel(ml.motor_max_accel_mm_s2)}
                      </span>
                      <span className="kpi-tri-dot">·</span>
                      <span className={accelLimFactor === "belt" ? "lim belt" : ""}>
                        Belt {formatCompactAccel(ml.belt_max_accel_mm_s2)}
                      </span>
                    </div>
                  ) : (
                    <span className="kpi-sub">
                      {limAxis}: {limShaperName} @ {Math.round(limEffHz)} Hz
                    </span>
                  )}
                  <button
                    type="button"
                    onClick={() => setRightTab("shaper")}
                    style={{
                      marginTop: 2,
                      padding: 0,
                      border: "none",
                      background: "transparent",
                      color: "var(--accent)",
                      fontSize: 11,
                      fontWeight: 600,
                      textAlign: "left",
                      cursor: "pointer",
                      display: "inline-flex",
                      alignItems: "center",
                      gap: 3,
                    }}
                    title="Open Klipper Input Shaper PSD graphs, ADXL345 overlay & modal modes"
                  >
                    View Klipper PSD →
                  </button>
                </div>

                <div className="kpi-box">
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 4 }}>
                    <span className="kpi-label">Resonance (f₁)</span>
                    {compareSim && (
                      <InlineDeltaBadge
                        oldVal={compareSim.modal_fea.fundamental_freq_hz}
                        newVal={mod.fundamental_freq_hz}
                        higherIsBetter={true}
                        eps={0.05}
                        formatDelta={(d) => `${d > 0 ? "+" : ""}${d.toFixed(1)} Hz`}
                      />
                    )}
                  </div>
                  <span className="kpi-value">{mod.fundamental_freq_hz.toFixed(1)} Hz</span>
                  <span className="kpi-sub">
                    X {xEffStr} / Y {yEffStr} Hz
                  </span>
                </div>

                <div
                  className="kpi-box"
                  title={`Dynamic inertial nozzle flex: ${curInertialFlex.toFixed(1)} µm (X ${fea.accel_deflection_x_um.toFixed(1)} / Y ${fea.accel_deflection_y_um.toFixed(1)} µm) · Static gravity sag: ${fea.gravity_sag_toolhead_um.toFixed(1)} µm · Combined peak: ${fea.combined_max_deflection_um.toFixed(1)} µm · Stiffness X/Y: ${fea.toolhead_stiffness_x_n_per_mm.toFixed(0)}/${fea.toolhead_stiffness_y_n_per_mm.toFixed(0)} N/mm`}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 4 }}>
                    <span className="kpi-label">Inertial Flex</span>
                    {cmpInertialFlex != null && (
                      <InlineDeltaBadge
                        oldVal={cmpInertialFlex}
                        newVal={curInertialFlex}
                        higherIsBetter={false}
                        eps={0.05}
                        formatDelta={(d) => `${d > 0 ? "+" : ""}${d.toFixed(1)} µm`}
                      />
                    )}
                  </div>
                  <span className="kpi-value">{curInertialFlex.toFixed(1)} µm</span>
                  <span className="kpi-sub">
                    Sag {fea.gravity_sag_toolhead_um.toFixed(1)} µm · {fea.toolhead_stiffness_x_n_per_mm.toFixed(0)}/{fea.toolhead_stiffness_y_n_per_mm.toFixed(0)} N/mm
                  </span>
                </div>

                <div
                  className="kpi-box"
                  title={
                    ml
                      ? `Motor Travel Max Speed (${Math.round(ml.motor_max_velocity_mm_s)} mm/s): stepper back-EMF + ωₑ·L·i bus-voltage ceiling before torque collapse.\nPrint Flow Speed (${th.max_linear_speed_02x04_mm_s.toFixed(0)} mm/s @ 0.2×0.4 mm, ${th.max_volumetric_flow_mm3_s.toFixed(1)} mm³/s): hotend melt-zone volumetric flow limit (${th.filament_id.toUpperCase()}).${motorSpeedExceeded ? `\n⚠ Configured max_velocity (${Math.round(ml.configured_max_velocity_mm_s)} mm/s) exceeds motor/voltage ceiling!` : ""}`
                      : `Hotend melt-zone flow: ${th.max_volumetric_flow_mm3_s.toFixed(1)} mm³/s (${th.max_linear_speed_02x04_mm_s.toFixed(0)} mm/s @ 0.2×0.4 mm, ${th.filament_id.toUpperCase()})`
                  }
                >
                  <div className="kpi-head-row">
                    <span className="kpi-label">Max Speed &amp; Flow</span>
                    {compareSim && (
                      <InlineDeltaBadge
                        oldVal={compareSim.thermal.max_volumetric_flow_mm3_s}
                        newVal={th.max_volumetric_flow_mm3_s}
                        higherIsBetter={true}
                        eps={0.05}
                        formatDelta={(d) => `${d > 0 ? "+" : ""}${d.toFixed(1)}`}
                      />
                    )}
                  </div>
                  {ml ? (
                    <>
                      <div className="kpi-speed-split">
                        <div className="kpi-speed-col">
                          <span className="kpi-speed-tag">Motor Travel</span>
                          <span
                            className={`kpi-speed-num ${motorSpeedExceeded ? "tag-warn" : ""}`}
                          >
                            {Math.round(ml.motor_max_velocity_mm_s)}
                            <span className="kpi-speed-unit"> mm/s</span>
                          </span>
                        </div>
                        <div className="kpi-speed-col">
                          <span className="kpi-speed-tag">Print Flow</span>
                          <span className="kpi-speed-num" style={{ color: "var(--accent)" }}>
                            {th.max_linear_speed_02x04_mm_s.toFixed(0)}
                            <span className="kpi-speed-unit"> mm/s</span>
                          </span>
                        </div>
                      </div>
                      <span className="kpi-sub">
                        {th.max_volumetric_flow_mm3_s.toFixed(1)} mm³/s · {th.filament_id.toUpperCase()}
                        {motorSpeedExceeded && (
                          <span className="tag-warn">
                            {" "}
                            · cfg {Math.round(ml.configured_max_velocity_mm_s)}
                          </span>
                        )}
                      </span>
                    </>
                  ) : (
                    <>
                      <span className="kpi-value">{th.max_volumetric_flow_mm3_s.toFixed(1)} mm³/s</span>
                      <span className="kpi-sub">
                        {th.filament_id.toUpperCase()} · {th.max_linear_speed_02x04_mm_s.toFixed(0)} mm/s
                      </span>
                    </>
                  )}
                </div>

                <div className="kpi-box">
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 4 }}>
                    <span className="kpi-label">Print Time</span>
                    {compareSim && (
                      <InlineDeltaBadge
                        oldVal={compareSim.virtual_print.total_time_s}
                        newVal={vp.total_time_s}
                        higherIsBetter={false}
                        eps={0.05}
                        formatDelta={(d) => `${d > 0 ? "+" : ""}${d.toFixed(1)} s`}
                      />
                    )}
                  </div>
                  <span className="kpi-value">{vp.total_time_s.toFixed(1)} s</span>
                  <span
                    className={`kpi-sub ${totalDefects === 0 ? "tag-ok" : "tag-warn"}`}
                  >
                    {totalDefects === 0 ? "0 defects" : `${totalDefects} defects`}
                  </span>
                </div>

                <div
                  className="kpi-box"
                  title={
                    cmpCost != null
                      ? `Build cost: $${Math.round(cmpCost).toLocaleString()} → $${Math.round(curCost).toLocaleString()}`
                      : `Build cost: $${Math.round(curCost).toLocaleString()}`
                  }
                >
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 4 }}>
                    <span className="kpi-label">Mass &amp; Cost</span>
                    {compareSim && (
                      <InlineDeltaBadge
                        oldVal={compareSim.geometry.mass_summary.total_mass_kg}
                        newVal={sim.geometry.mass_summary.total_mass_kg}
                        higherIsBetter={false}
                        eps={0.01}
                        formatDelta={(d) => `${d > 0 ? "+" : ""}${d.toFixed(2)} kg`}
                      />
                    )}
                  </div>
                  <span className="kpi-value">
                    {sim.geometry.mass_summary.total_mass_kg.toFixed(2)} kg
                  </span>
                  <span className="kpi-sub">
                    ${Math.round(curCost).toLocaleString()} · Y mov{" "}
                    {Math.round(sim.geometry.mass_summary.y_moving_mass_g)}g
                  </span>
                </div>
              </div>
            </div>

            {/* Tier 1: Single Authoritative Limiting Resonance & Top Causes Card */}
            {(mod.accel_bottleneck || mod.limiting_modal_summary) && (
              <AccelBottleneckCard
                bottleneck={mod.accel_bottleneck}
                limitingSummary={mod.limiting_modal_summary}
                shaperX={mod.shaper_x}
                shaperY={mod.shaper_y}
                modes={mod.modes}
                onInspectModalMode={onInspectModalMode}
                changed={modalChanged}
              />
            )}

            {/* Tier 1: Concise Engineering Diagnostics (warnings & errors first) */}
            <div className="section-card">
              <div className="section-title">
                <span>
                  Engineering Diagnostics
                  {alertDiags.length > 0 && (
                    <span
                      className={`badge-pill ${alertDiags.some((d) => d.cls === "bad") ? "bad" : "warn"}`}
                      style={{ marginLeft: 6, padding: "0 6px", fontSize: 11 }}
                    >
                      {alertDiags.length}
                    </span>
                  )}
                </span>
                {infoDiags.length > 0 && (
                  <button
                    type="button"
                    className="impact-mini-btn"
                    onClick={() => setShowInfoNotes((v) => !v)}
                  >
                    {showInfoNotes
                      ? `Hide ${infoDiags.length} notes`
                      : `Show ${infoDiags.length} notes`}
                  </button>
                )}
              </div>
              {alertDiags.length === 0 ? (
                <div
                  className="diag-item"
                  style={{
                    borderLeftColor: "var(--green)",
                    background: "rgba(34, 197, 94, 0.08)",
                    color: "var(--green)",
                  }}
                >
                  ✓ All structural, kinematic, thermal &amp; virtual print checks passed
                </div>
              ) : (
                alertDiags.map((item, idx) => (
                  <div key={`alert-${idx}`} className={`diag-item ${item.cls}`}>
                    <span className="diag-sev">{item.cls === "bad" ? "Critical" : "Warning"}</span>
                    {item.text}
                  </div>
                ))
              )}
              {showInfoNotes &&
                infoDiags.map((item, idx) => (
                  <div key={`info-${idx}`} className="diag-item">
                    <span className="diag-sev">Note</span>
                    {item.text}
                  </div>
                ))}
            </div>
          </>
        )}

        {rightTab === "shaper" && (
          <>
            {mod.shaper_graphs ? (
              <KlipperShaperPsdCard
                graphs={mod.shaper_graphs}
                adxlVsNozzle={mod.adxl_vs_nozzle_diagnostic}
                frameDampingRatio={mod.frame_damping_ratio}
                extrusionFilling={sim.geometry.extrusion_filling}
                spec={sim.spec}
                motionLimits={sim.motion_limits}
                changed={modalChanged}
              />
            ) : (
              <div className="section-card">
                <div className="section-title">
                  <span>Klipper Input Shaper (PSD)</span>
                </div>
                <div style={{ fontSize: 12, color: "var(--muted)", lineHeight: 1.45 }}>
                  Synthetic Klipper ADXL345 PSD curves are unavailable for this simulation result.
                  Run a full modal simulation with XY gantry dynamics to generate X/Y input shaper
                  spectra and candidate tables.
                </div>
              </div>
            )}

            <details className="sub-accordion">
              <summary>
                <span>Measured ADXL345 / Shake&amp;Tune CSV Overlay</span>
                <span className="sub-accordion-tag">Measured vs Sim</span>
              </summary>
              <div className="sub-accordion-body">
                <AdxlCalibrationPanel
                  spec={sim.spec}
                  onApplyFittedSpec={onApplyFittedSpec}
                />
              </div>
            </details>

            {mod.modes && mod.modes.length > 0 && (
              <ModalVibrationModesCard
                modes={mod.modes}
                limitingSummary={mod.limiting_modal_summary}
                worstModeNumber={mod.worst_print_mode_number}
                viewMode={viewMode}
                onViewModeChange={onViewModeChange}
                onInspectModalMode={onInspectModalMode}
                changed={modalChanged}
              />
            )}
          </>
        )}

        {rightTab === "deep" && (
          <>
            {/* Tier 2: Expandable Deep Engineering Data Accordions */}
            <div className="sim-toolbar-row">
              <span className="section-title" style={{ margin: 0 }}>
                Deep Engineering Data
              </span>
              <button
                type="button"
                className="impact-mini-btn"
                onClick={() => {
                  const nextVal = !allExpanded;
                  const next: Record<string, boolean> = {};
                  for (const k of visibleDeepKeys) next[k] = nextVal;
                  setSectionOpen(next);
                }}
              >
                {allExpanded ? "Collapse All" : "Expand All"}
              </button>
            </div>

            {/* Accordion 1: Static FEA, Toolhead & Thermal-Structural */}
            <DeepDiveAccordion
              title="Static FEA, Toolhead & Thermal-Structural"
              badge={feaAccordionBadge}
              updated={structuralChanged || thermalStructChanged}
              open={Boolean(sectionOpen["fea_thermal_struct"])}
              onToggle={() => toggleSection("fea_thermal_struct")}
            >
              {sim.geometry.panels?.enabled && (() => {
                const p = sim.geometry.panels;
                const mountTag =
                  p.mounting === "bolted_structural"
                    ? p.bolted_to_gantry
                      ? "Bolted + Gantry"
                      : p.is_flying_gantry
                        ? "Outer Frame — Flying Gantry"
                        : "Bolted Outer Frame"
                    : "Clip-In";
                const metaParts: string[] = [];
                if (p.buckling_factor != null) {
                  metaParts.push(`ψ=${p.buckling_factor.toFixed(2)}`);
                }
                if (p.areal_mass_kg_m2 != null) {
                  metaParts.push(`${p.areal_mass_kg_m2.toFixed(2)} kg/m²`);
                }
                const hasShearReadout =
                  p.effective_shear_gt_kn_mm != null || metaParts.length > 0;
                const totalMassTitle =
                  p.total_panel_mass_kg != null
                    ? `Total panel mass: ${p.total_panel_mass_kg.toFixed(2)} kg`
                    : undefined;
                return (
                  <>
                    <div className="metric-kv" title={totalMassTitle}>
                      <span>Structural Panels</span>
                      <span
                        className="metric-val"
                        style={{
                          color:
                            p.mounting === "bolted_structural"
                              ? p.bolted_to_gantry
                                ? "var(--accent)"
                                : p.is_flying_gantry
                                  ? "var(--amber)"
                                  : "var(--text)"
                              : "var(--muted)",
                          textAlign: "right",
                        }}
                      >
                        {p.thickness_mm}mm {p.material_name} ({mountTag})
                      </span>
                    </div>
                    {hasShearReadout && (
                      <div className="metric-kv" title={totalMassTitle}>
                        <span>Panel Shear Gt / Buckling</span>
                        <span className="metric-val" style={{ textAlign: "right" }}>
                          {p.effective_shear_gt_kn_mm != null
                            ? `${p.effective_shear_gt_kn_mm.toFixed(2)} kN/mm`
                            : "—"}
                          {metaParts.length > 0 && (
                            <span style={{ color: "var(--muted)", fontWeight: 500, fontSize: 11 }}>
                              {" "}
                              ({metaParts.join(" • ")})
                            </span>
                          )}
                        </span>
                      </div>
                    )}
                  </>
                );
              })()}
              {sim.geometry.linear_rails && (() => {
                const lr = sim.geometry.linear_rails;
                const xMountShort: Record<string, string> = {
                  front: "Front",
                  top: "Top",
                  bottom: "Bottom",
                  dual_top_bottom: "Dual Top+Bot",
                };
                const yMountShort: Record<string, string> = {
                  top: "Top",
                  inner: "Inner",
                  bottom: "Bottom",
                };
                const xProf = lr.x_rail_profile.includes("mgn9") ? "MGN9H" : "MGN12H";
                const yProf = lr.y_rail_profile.includes("mgn9") ? "MGN9H" : "MGN12H";
                return (
                  <div
                    className="metric-kv"
                    title={`Total hardened-steel rail mass: ${Math.round(lr.total_rail_mass_g)} g`}
                  >
                    <span>Linear Rails (Composite EI)</span>
                    <span className="metric-val" style={{ textAlign: "right" }}>
                      {xProf} {xMountShort[lr.x_rail_mount] ?? lr.x_rail_mount}
                      <span style={{ color: "var(--accent)", fontWeight: 500, fontSize: 11 }}>
                        {" "}
                        (+{Math.round(lr.x_beam_ei_y_boost_pct)}% Y / +{Math.round(lr.x_beam_ei_z_boost_pct)}% Z)
                      </span>
                      {sim.architecture === "corexy" && (
                        <span style={{ color: "var(--muted)", fontWeight: 500, fontSize: 11 }}>
                          {" "}
                          · {yProf} {yMountShort[lr.y_rail_mount] ?? lr.y_rail_mount} (+
                          {Math.round(lr.y_rail_ei_x_boost_pct)}% X)
                        </span>
                      )}
                    </span>
                  </div>
                );
              })()}
              <div
                className="metric-kv"
                title={`Dynamic nozzle deflection under XY acceleration (excludes static gravity sag). Combined gravity+accel peak: ${fea.combined_max_deflection_um.toFixed(1)} µm.`}
              >
                <span>
                  Inertial Accel Flex
                  <span className="metric-sub-inline">
                    X {fea.accel_deflection_x_um.toFixed(1)} µm · Y {fea.accel_deflection_y_um.toFixed(1)} µm
                  </span>
                </span>
                <span className="metric-val metric-val-with-delta">
                  {cmpInertialFlex != null && (
                    <InlineDeltaBadge
                      oldVal={cmpInertialFlex}
                      newVal={curInertialFlex}
                      higherIsBetter={false}
                      eps={0.05}
                      formatDelta={(d) => `${d > 0 ? "+" : ""}${d.toFixed(1)} µm`}
                    />
                  )}
                  {curInertialFlex.toFixed(1)} µm
                </span>
              </div>
              <div className="metric-kv">
                <span>Gravity Sag (Nozzle)</span>
                <span className="metric-val">{fea.gravity_sag_toolhead_um.toFixed(1)} µm</span>
              </div>
              {(fea.nozzle_absolute_deflection_um != null || fea.bed_deflection_um != null) && (
                <div className="metric-kv">
                  <span>Abs / Bed Flex Split</span>
                  <span className="metric-val metric-val-with-delta">
                    {compareSim?.static_fea.nozzle_absolute_deflection_um != null &&
                      fea.nozzle_absolute_deflection_um != null && (
                        <InlineDeltaBadge
                          oldVal={compareSim.static_fea.nozzle_absolute_deflection_um}
                          newVal={fea.nozzle_absolute_deflection_um}
                          higherIsBetter={false}
                          formatDelta={(d) => `${d > 0 ? "+" : ""}${d.toFixed(1)} µm`}
                        />
                      )}
                    Abs {(fea.nozzle_absolute_deflection_um ?? fea.combined_max_deflection_um).toFixed(1)} µm · Bed{" "}
                    {(fea.bed_deflection_um ?? 0).toFixed(1)} µm
                  </span>
                </div>
              )}
              {fea.beam_center_deflection_um != null && (
                <div className="metric-kv">
                  <span>X-Beam Center Flex</span>
                  <span className="metric-val metric-val-with-delta">
                    {compareSim?.static_fea.beam_center_deflection_um != null && (
                      <InlineDeltaBadge
                        oldVal={compareSim.static_fea.beam_center_deflection_um}
                        newVal={fea.beam_center_deflection_um}
                        higherIsBetter={false}
                        formatDelta={(d) => `${d > 0 ? "+" : ""}${d.toFixed(1)} µm`}
                      />
                    )}
                    {fea.beam_center_deflection_um.toFixed(1)} µm
                  </span>
                </div>
              )}
              {(fea.nozzle_twist_amplification_pct != null || fea.nozzle_twist_angle_mrad != null) && (
                <div className="metric-kv">
                  <span>Nozzle Lever-Arm Twist</span>
                  <span
                    className={`metric-val metric-val-with-delta ${
                      (fea.nozzle_twist_amplification_pct ?? 0) > 60
                        ? "tag-bad"
                        : (fea.nozzle_twist_amplification_pct ?? 0) > 30
                          ? "tag-warn"
                          : "tag-ok"
                    }`}
                  >
                    {compareSim?.static_fea.nozzle_twist_amplification_pct != null &&
                      fea.nozzle_twist_amplification_pct != null && (
                        <InlineDeltaBadge
                          oldVal={compareSim.static_fea.nozzle_twist_amplification_pct}
                          newVal={fea.nozzle_twist_amplification_pct}
                          higherIsBetter={false}
                          eps={0.5}
                          formatDelta={(d) => `${d > 0 ? "+" : ""}${d.toFixed(0)}%`}
                        />
                      )}
                    {fea.nozzle_twist_amplification_pct != null
                      ? `+${fea.nozzle_twist_amplification_pct.toFixed(0)}%`
                      : ""}
                    {fea.nozzle_twist_amplification_pct != null && fea.nozzle_twist_angle_mrad != null
                      ? " • "
                      : ""}
                    {fea.nozzle_twist_angle_mrad != null
                      ? `${fea.nozzle_twist_angle_mrad.toFixed(2)} mrad`
                      : ""}
                  </span>
                </div>
              )}
              <div className="metric-kv">
                <span>
                  Max von Mises Stress
                  {fea.peak_stress_part_label && (
                    <span className="metric-sub-inline">Peak: {fea.peak_stress_part_label}</span>
                  )}
                </span>
                <span className="metric-val">{fea.max_von_mises_mpa.toFixed(2)} MPa</span>
              </div>
              {(fea.panel_max_von_mises_mpa ?? 0) > 0 && (
                <div className="metric-kv">
                  <span>Panel Peak Stress (MITC4)</span>
                  <span className="metric-val metric-val-with-delta">
                    {compareSim?.static_fea.panel_max_von_mises_mpa != null &&
                      compareSim.static_fea.panel_max_von_mises_mpa > 0 && (
                        <InlineDeltaBadge
                          oldVal={compareSim.static_fea.panel_max_von_mises_mpa}
                          newVal={fea.panel_max_von_mises_mpa!}
                          higherIsBetter={false}
                          eps={0.01}
                          formatDelta={(d) => `${d > 0 ? "+" : ""}${d.toFixed(2)} MPa`}
                        />
                      )}
                    {fea.panel_max_von_mises_mpa!.toFixed(2)} MPa
                  </span>
                </div>
              )}
              <div className="metric-kv">
                <span>Min Safety Factor</span>
                <span className={`metric-val ${fea.min_safety_factor < 3 ? "tag-bad" : "tag-ok"}`}>
                  {fea.min_safety_factor.toFixed(1)}×
                </span>
              </div>
              {fea.top_flex_explanation && (
                <div className="fea-explain-note">{fea.top_flex_explanation}</div>
              )}
              {fea.toolhead_flex_breakdown && (
                <ToolheadFlexBreakdownCard
                  breakdown={fea.toolhead_flex_breakdown}
                  toolheadStructure={sim.geometry.toolhead_structure}
                  specToolhead={sim.spec.toolhead}
                  toolheadPunchIn={toolheadPunchIn}
                  onToggleToolheadPunchIn={onToggleToolheadPunchIn}
                />
              )}

              {ts && (
                <div
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    gap: 6,
                    marginTop: 6,
                    paddingTop: 8,
                    borderTop: "1px dashed rgba(255, 255, 255, 0.09)",
                  }}
                >
                  <div className="section-title">
                    <span>Thermal-Structural &amp; Precision</span>
                    <span className="sub-block-tag">
                      {Math.round(ts.chamber_temp_c)}°C / {Math.round(ts.bed_target_temp_c)}°C
                    </span>
                  </div>

                  <div className="metric-kv">
                    <span>
                      Bimetallic Gantry Bow
                      {(ts.gantry_bow_direction === "toward_rail" ||
                        ts.gantry_bow_direction === "away_from_rail") &&
                        ts.gantry_bimetallic_bow_um > 1 && (
                          <span className="metric-sub-inline">
                            {ts.gantry_bow_direction === "toward_rail"
                              ? "toward rail"
                              : "away from rail"}
                          </span>
                        )}
                    </span>
                    <span
                      className={`metric-val metric-val-with-delta ${
                        ts.gantry_bimetallic_bow_um > 80
                          ? "tag-bad"
                          : ts.gantry_bimetallic_bow_um > 40
                            ? "tag-warn"
                            : "tag-ok"
                      }`}
                    >
                      {cmpTs && (
                        <InlineDeltaBadge
                          oldVal={cmpTs.gantry_bimetallic_bow_um}
                          newVal={ts.gantry_bimetallic_bow_um}
                          higherIsBetter={false}
                          eps={0.2}
                          formatDelta={(d) => `${d > 0 ? "+" : ""}${d.toFixed(1)} µm`}
                        />
                      )}
                      {ts.gantry_bimetallic_bow_um.toFixed(1)} µm
                      {ts.gantry_backer_reduction_pct > 0 && (
                        <span className="badge-pill ok" style={{ padding: "1px 6px", fontSize: 11 }}>
                          -{Math.round(ts.gantry_backer_reduction_pct)}% backer
                        </span>
                      )}
                    </span>
                  </div>

                  {!bedFea && (
                    <div className="metric-kv">
                      <span>
                        Bed Thermal Bow / Buckling
                        <span className="metric-sub-inline">
                          Grad {ts.bed_gradient_bow_um.toFixed(1)} µm · Buckling {ts.bed_buckling_taco_um.toFixed(1)} µm
                        </span>
                      </span>
                      <span
                        className={`metric-val metric-val-with-delta ${
                          ts.bed_buckling_taco_um > 30
                            ? "tag-bad"
                            : ts.bed_gradient_bow_um + ts.bed_buckling_taco_um > 50
                              ? "tag-warn"
                              : ""
                        }`}
                      >
                        {(ts.bed_gradient_bow_um + ts.bed_buckling_taco_um).toFixed(1)} µm
                        {ts.bed_buckling_taco_um > 30 && (
                          <span className="badge-pill bad" style={{ padding: "1px 6px", fontSize: 11 }}>
                            Taco Buckling!
                          </span>
                        )}
                      </span>
                    </div>
                  )}

                <div className="metric-kv" title={`Chamber thermal rise ΔT = ${ts.delta_t_k.toFixed(1)} K`}>
                  <span>Frame Z-Drift</span>
                  <span className="metric-val">
                    {ts.frame_z_drift_um.toFixed(1)} µm
                    <span style={{ color: "var(--muted)", fontWeight: 500, fontSize: 11 }}>
                      {" "}
                      (ΔT {ts.delta_t_k.toFixed(0)} K)
                    </span>
                  </span>
                </div>

                <div className="metric-kv">
                  <span>Total Thermal Z Error</span>
                  <span
                    className={`metric-val metric-val-with-delta ${
                      ts.total_thermal_z_error_um > 200
                        ? "tag-bad"
                        : ts.total_thermal_z_error_um > 100
                          ? "tag-warn"
                          : "tag-ok"
                    }`}
                  >
                    {cmpTs && (
                      <InlineDeltaBadge
                        oldVal={cmpTs.total_thermal_z_error_um}
                        newVal={ts.total_thermal_z_error_um}
                        higherIsBetter={false}
                        eps={0.2}
                        formatDelta={(d) => `${d > 0 ? "+" : ""}${d.toFixed(1)} µm`}
                      />
                    )}
                    <strong>{ts.total_thermal_z_error_um.toFixed(1)} µm</strong>
                  </span>
                </div>

                <div className="metric-kv">
                  <span>Bracket E(T) Retention</span>
                  <span
                    className={`metric-val metric-val-with-delta ${
                      ts.bracket_tg_warning
                        ? ts.bracket_modulus_retention_pct < 75
                          ? "tag-bad"
                          : "tag-warn"
                        : "tag-ok"
                    }`}
                  >
                    {ts.bracket_modulus_retention_pct.toFixed(0)}%
                    {ts.bracket_tg_warning && (
                      <span
                        className={`badge-pill ${ts.bracket_modulus_retention_pct < 75 ? "bad" : "warn"}`}
                        style={{ padding: "1px 6px", fontSize: 11 }}
                      >
                        Tg Softening
                      </span>
                    )}
                  </span>
                </div>

                {ts.x_beam_has_tg && (
                  <div
                    className="metric-kv"
                    title="Temperature-dependent composite laminate flexural (E) and torsional shear (G) modulus retention at chamber temperature"
                  >
                    <span>X-Beam Matrix Retention</span>
                    <span
                      className={`metric-val metric-val-with-delta ${
                        ts.x_beam_tg_warning
                          ? "tag-bad"
                          : (ts.x_beam_shear_retention_pct ?? 100) < 92
                            ? "tag-warn"
                            : "tag-ok"
                      }`}
                    >
                      E {(ts.x_beam_modulus_retention_pct ?? 100).toFixed(0)}% · G{" "}
                      {(ts.x_beam_shear_retention_pct ?? 100).toFixed(0)}%
                      {ts.x_beam_tg_warning && (
                        <span className="badge-pill bad">Matrix Tg!</span>
                      )}
                    </span>
                  </div>
                )}

                {ts.motor_mount_temp_c != null && (
                  <div
                    className="metric-kv"
                    title={
                      sim.spec.gantry.stepper_thermal_gasket === "ptfe_2mm"
                        ? "Estimated stepper motor mounting flange temperature with 2 mm PTFE thermal barrier (−14 °C mount reduction)"
                        : "Estimated stepper motor mounting flange temperature conducting into the gantry bracket (add a 2 mm PTFE gasket to drop ~14 °C)"
                    }
                  >
                    <span>Motor Mount Temp</span>
                    <span
                      className={`metric-val metric-val-with-delta ${
                        ts.motor_mount_temp_c >= 90
                          ? "tag-bad"
                          : ts.motor_mount_temp_c >= 75
                            ? "tag-warn"
                            : "tag-ok"
                      }`}
                    >
                      {ts.motor_mount_temp_c.toFixed(1)} °C
                      {sim.spec.gantry.stepper_thermal_gasket === "ptfe_2mm" && (
                        <span className="badge-pill ok">PTFE −14 °C</span>
                      )}
                    </span>
                  </div>
                )}

                {(ts.bearing_cte_clearance_um != null ||
                  ts.bearing_temp_c != null ||
                  ts.bearing_cte_pressfit_warning) && (
                  <div
                    className="metric-kv"
                    title="F695/695-2RS steel bearing inside aluminum housing: differential CTE expansion at elevated chamber temperature can loosen press-fit or alter radial clearance"
                  >
                    <span>
                      F695 Bearing CTE
                      {ts.bearing_temp_c != null && (
                        <span className="metric-sub-inline">
                          {ts.bearing_temp_c.toFixed(0)} °C
                        </span>
                      )}
                    </span>
                    <span
                      className={`metric-val metric-val-with-delta ${
                        ts.bearing_cte_pressfit_warning ? "tag-bad" : "tag-ok"
                      }`}
                    >
                      {ts.bearing_cte_clearance_um != null
                        ? `${ts.bearing_cte_clearance_um >= 0 ? "+" : ""}${ts.bearing_cte_clearance_um.toFixed(1)} µm`
                        : "OK"}
                      {ts.bearing_cte_pressfit_warning && (
                        <span className="badge-pill bad">Bearing CTE Warning</span>
                      )}
                    </span>
                  </div>
                )}

                {(ts.toolhead_belt_cog_z_offset_mm != null || ts.toolhead_x_roll_warning) && (
                  <div
                    className="metric-kv"
                    title="Vertical Z lever arm between belt drive plane and toolhead CoG; large offsets excite X-axis roll modes under acceleration"
                  >
                    <span>
                      Belt-Plane CoG Offset
                      {ts.toolhead_nozzle_y_lever_mm != null && (
                        <span className="metric-sub-inline">
                          Nozzle Y {ts.toolhead_nozzle_y_lever_mm.toFixed(1)} mm
                        </span>
                      )}
                    </span>
                    <span
                      className={`metric-val metric-val-with-delta ${
                        ts.toolhead_x_roll_warning ? "tag-warn" : "tag-ok"
                      }`}
                    >
                      {ts.toolhead_belt_cog_z_offset_mm != null
                        ? `${Math.abs(ts.toolhead_belt_cog_z_offset_mm).toFixed(1)} mm`
                        : "—"}
                      {ts.toolhead_x_roll_warning && (
                        <span className="badge-pill warn">X-Roll Coupling</span>
                      )}
                    </span>
                  </div>
                )}

                <div className="precision-mini-grid">
                  <div
                    className="precision-cell"
                    title="Newmark partial-composite shear action factor γ (0 = uncoupled, 1 = fully bonded rail-beam)"
                  >
                    <span className="precision-label">Newmark Rail γ</span>
                    <span className="precision-val">{ts.rail_newmark_gamma.toFixed(2)}</span>
                  </div>
                  <div
                    className="precision-cell"
                    title={`Carriage roll stiffness (${Math.round(ts.rail_carriage_roll_nm_per_rad)} N·m/rad) • Radial ${ts.rail_carriage_radial_n_per_um.toFixed(0)} N/µm`}
                  >
                    <span className="precision-label">Carriage Roll</span>
                    <span className="precision-val">
                      {(ts.rail_carriage_roll_nm_per_rad / 1000).toFixed(2)} kNm/rad
                    </span>
                  </div>
                  <div
                    className="precision-cell"
                    title="Belt tooth-mesh engagement stiffness at pulley wrap"
                  >
                    <span className="precision-label">Tooth-Mesh k</span>
                    <span className="precision-val">
                      {kin.tooth_mesh_stiffness_n_per_mm != null
                        ? `${Math.round(kin.tooth_mesh_stiffness_n_per_mm)} N/mm`
                        : "—"}
                    </span>
                  </div>
                  <div
                    className="precision-cell"
                    title={
                      [
                        kin.idler_shaft_stiffness_n_per_mm != null
                          ? `Shaft k = ${Math.round(kin.idler_shaft_stiffness_n_per_mm)} N/mm`
                          : null,
                        kin.bearing_play_um != null
                          ? `Bearing play = ${kin.bearing_play_um.toFixed(1)} µm`
                          : null,
                      ]
                        .filter(Boolean)
                        .join(" • ") || "Idler shaft deflection under belt tension"
                    }
                  >
                    <span className="precision-label">Idler Deflection</span>
                    <span className="precision-val">
                      {kin.idler_shaft_deflection_um != null
                        ? `${kin.idler_shaft_deflection_um.toFixed(1)} µm`
                        : "—"}
                    </span>
                  </div>
                </div>

                  {bedFea && (
                    <BedPlateFemStrip
                      bedFea={bedFea}
                      cmpBedFea={cmpBedFea}
                      ts={ts}
                      showBedWarp={showBedWarp}
                      onInspectBedWarp={onInspectBedWarp}
                    />
                  )}
                </div>
              )}

              {!ts && bedFea && (
                <div style={{ marginTop: 6 }}>
                  <BedPlateFemStrip
                    bedFea={bedFea}
                    cmpBedFea={cmpBedFea}
                    showBedWarp={showBedWarp}
                    onInspectBedWarp={onInspectBedWarp}
                  />
                </div>
              )}
            </DeepDiveAccordion>

            {/* Accordion 2: Kinematics, Belts & Steppers */}
            <DeepDiveAccordion
              title="Kinematics, Belts & Steppers"
              badge={modesAccordionBadge}
              updated={modalChanged}
              open={Boolean(sectionOpen["modes_kinematics"])}
              onToggle={() => toggleSection("modes_kinematics")}
            >
              {(() => {
                const motorCount = kin.xy_motor_count ?? (effDrive === "awd" ? 4 : 2);
                const isAwd = effDrive === "awd";
                const pluckHz = kin.belt_pluck_freq_150mm_hz;
                const prevPluckHz = compareSim?.kinematics.belt_pluck_freq_150mm_hz;
                const pretensionN = sim.spec.gantry.belt.pretension_n;
                const maxWorkingN = kin.belt_max_working_tension_n;

                return (
                  <>
                    <div className="metric-kv">
                      <span>Drive Mode</span>
                      <span className="metric-val" style={{ color: isAwd ? "var(--accent)" : undefined }}>
                        {isAwd ? `AWD (${motorCount} Motors)` : `2WD (${motorCount} Motors)`}
                        {(kin.belt_routing ?? sim.spec.gantry.belt_routing) === "monolith_short"
                          ? " • Monolith Short"
                          : ""}
                        {sim.spec.gantry.double_shear_pulleys ? " • Double-Shear" : ""}
                      </span>
                    </div>
                    {mod.nozzle_pitch_twist_freq_hz != null && (
                      <div className="metric-kv">
                        <span>Nozzle Pitch/Twist Mode</span>
                        <span className="metric-val metric-val-with-delta">
                          {compareSim?.modal_fea.nozzle_pitch_twist_freq_hz != null && (
                            <InlineDeltaBadge
                              oldVal={compareSim.modal_fea.nozzle_pitch_twist_freq_hz}
                              newVal={mod.nozzle_pitch_twist_freq_hz}
                              higherIsBetter={true}
                              formatDelta={(d) => `${d > 0 ? "+" : ""}${d.toFixed(1)} Hz`}
                            />
                          )}
                          {mod.nozzle_pitch_twist_freq_hz.toFixed(1)} Hz
                        </span>
                      </div>
                    )}
                    {(mod.panel_drum_freq_hz ?? 0) > 0 && (
                      <div className="metric-kv">
                        <span>Panel Drum Mode</span>
                        <span
                          className={`metric-val metric-val-with-delta ${
                            mod.panel_drum_freq_hz! < 65
                              ? "tag-warn"
                              : mod.panel_drum_freq_hz! >= 85
                                ? "tag-ok"
                                : ""
                          }`}
                        >
                          {compareSim?.modal_fea.panel_drum_freq_hz != null &&
                            compareSim.modal_fea.panel_drum_freq_hz > 0 && (
                              <InlineDeltaBadge
                                oldVal={compareSim.modal_fea.panel_drum_freq_hz}
                                newVal={mod.panel_drum_freq_hz!}
                                higherIsBetter={true}
                                eps={0.1}
                                formatDelta={(d) => `${d > 0 ? "+" : ""}${d.toFixed(1)} Hz`}
                              />
                            )}
                          {mod.panel_drum_freq_hz!.toFixed(1)} Hz
                        </span>
                      </div>
                    )}
                    <div className="metric-kv">
                      <span>Stepper Torque Margin</span>
                      <span className={`metric-val metric-val-with-delta ${kin.step_skip_risk ? "tag-bad" : "tag-ok"}`}>
                        {compareSim && (
                          <InlineDeltaBadge
                            oldVal={compareSim.kinematics.torque_safety_margin}
                            newVal={kin.torque_safety_margin}
                            higherIsBetter={true}
                            eps={0.05}
                            formatDelta={(d) => `${d > 0 ? "+" : ""}${d.toFixed(2)}×`}
                          />
                        )}
                        {kin.torque_safety_margin.toFixed(2)}×
                      </span>
                    </div>
                    {kin.motor_stiffness_n_per_mm != null && (
                      <div className="metric-kv">
                        <span>Motor Magnetic Stiffness</span>
                        <span className="metric-val metric-val-with-delta">
                          {compareSim?.kinematics.motor_stiffness_n_per_mm != null && (
                            <InlineDeltaBadge
                              oldVal={compareSim.kinematics.motor_stiffness_n_per_mm}
                              newVal={kin.motor_stiffness_n_per_mm}
                              higherIsBetter={true}
                              eps={0.1}
                              formatDelta={(d) => `${d > 0 ? "+" : ""}${d.toFixed(1)}`}
                            />
                          )}
                          {kin.motor_stiffness_n_per_mm.toFixed(1)} N/mm
                        </span>
                      </div>
                    )}
                    {kin.belt_stiffness_n_per_mm != null && (
                      <div className="metric-kv">
                        <span>Belt Stiffness</span>
                        <span className="metric-val metric-val-with-delta">
                          {compareSim?.kinematics.belt_stiffness_n_per_mm != null && (
                            <InlineDeltaBadge
                              oldVal={compareSim.kinematics.belt_stiffness_n_per_mm}
                              newVal={kin.belt_stiffness_n_per_mm}
                              higherIsBetter={true}
                              eps={0.1}
                              formatDelta={(d) => `${d > 0 ? "+" : ""}${d.toFixed(1)}`}
                            />
                          )}
                          {kin.belt_stiffness_n_per_mm.toFixed(1)} N/mm
                        </span>
                      </div>
                    )}
                    {kin.bracket_stiffness_n_per_mm != null && (
                      <div
                        className="metric-kv"
                        title={
                          sim.geometry.gantry_brackets
                            ? `${sim.geometry.gantry_brackets.name} • HDT ${sim.geometry.gantry_brackets.hdt_c}°C • Creep limit ${sim.geometry.gantry_brackets.creep_tension_limit_n} N`
                            : undefined
                        }
                      >
                        <span>
                          Bracket Stiffness
                          {kin.custom_bracket_active || sim.spec?.gantry?.custom_bracket_override ? (
                            <span style={{ color: "#22d3ee", fontSize: 11, fontWeight: 500 }}>
                              {" "}
                              (CAD:{" "}
                              {kin.custom_bracket_label ??
                                sim.spec?.gantry?.custom_bracket_override?.label}
                              )
                            </span>
                          ) : (
                            sim.geometry.gantry_brackets && (
                              <span style={{ color: "var(--muted)", fontSize: 11 }}>
                                {" "}
                                ({sim.geometry.gantry_brackets.short_name})
                              </span>
                            )
                          )}
                        </span>
                        <span
                          className={`metric-val metric-val-with-delta ${
                            kin.bracket_creep_risk ? "tag-bad" : ""
                          }`}
                        >
                          {compareSim?.kinematics.bracket_stiffness_n_per_mm != null && (
                            <InlineDeltaBadge
                              oldVal={compareSim.kinematics.bracket_stiffness_n_per_mm}
                              newVal={kin.bracket_stiffness_n_per_mm}
                              higherIsBetter={true}
                              eps={0.5}
                              formatDelta={(d) => `${d > 0 ? "+" : ""}${d.toFixed(0)}`}
                            />
                          )}
                          {kin.bracket_stiffness_n_per_mm.toFixed(0)} N/mm
                          {kin.bracket_creep_risk && " (CREEP)"}
                        </span>
                      </div>
                    )}
                    {kin.drive_stiffness_n_per_mm != null && (
                      <div className="metric-kv">
                        <span>Combined Drive Stiffness</span>
                        <span className="metric-val metric-val-with-delta">
                          {compareSim?.kinematics.drive_stiffness_n_per_mm != null && (
                            <InlineDeltaBadge
                              oldVal={compareSim.kinematics.drive_stiffness_n_per_mm}
                              newVal={kin.drive_stiffness_n_per_mm}
                              higherIsBetter={true}
                              eps={0.1}
                              formatDelta={(d) => `${d > 0 ? "+" : ""}${d.toFixed(1)}`}
                            />
                          )}
                          {kin.drive_stiffness_n_per_mm.toFixed(1)} N/mm
                        </span>
                      </div>
                    )}
                    {kin.motor_voltage_efficiency_pct != null && (
                      <div className="metric-kv">
                        <span>Voltage Efficiency</span>
                        <span
                          className={`metric-val metric-val-with-delta ${
                            kin.motor_voltage_efficiency_pct < 70
                              ? "tag-bad"
                              : kin.motor_voltage_efficiency_pct < 85
                                ? "tag-warn"
                                : "tag-ok"
                          }`}
                        >
                          {compareSim?.kinematics.motor_voltage_efficiency_pct != null && (
                            <InlineDeltaBadge
                              oldVal={compareSim.kinematics.motor_voltage_efficiency_pct}
                              newVal={kin.motor_voltage_efficiency_pct}
                              higherIsBetter={true}
                              eps={0.5}
                              formatDelta={(d) => `${d > 0 ? "+" : ""}${d.toFixed(0)}%`}
                            />
                          )}
                          {kin.motor_voltage_efficiency_pct.toFixed(0)}%
                        </span>
                      </div>
                    )}
                    {kin.motor_heat_w_each != null && (
                      <div className="metric-kv">
                        <span>Motor Heat</span>
                        <span
                          className={`metric-val ${
                            kin.motor_overcurrent_risk || kin.motor_thermal_risk
                              ? "tag-bad"
                              : kin.motor_heat_w_each > 8.5
                                ? "tag-warn"
                                : ""
                          }`}
                        >
                          {kin.motor_heat_w_each.toFixed(1)} W/motor
                          {kin.motor_est_temp_c != null && (
                            <span style={{ color: "var(--muted)", fontWeight: 500, fontSize: 11 }}>
                              {" "}
                              (~{Math.round(kin.motor_est_temp_c)}°C / {Math.round(kin.motor_max_winding_temp_c ?? 130)}°C)
                            </span>
                          )}
                        </span>
                      </div>
                    )}
                    {pluckHz != null && (
                      <div className="metric-kv">
                        <span>Belt Tune (150mm span)</span>
                        <span
                          className={`metric-val metric-val-with-delta ${
                            kin.belt_overtension_risk ? "tag-bad" : kin.shaft_bending_risk ? "tag-warn" : ""
                          }`}
                        >
                          {prevPluckHz != null && (
                            <InlineDeltaBadge
                              oldVal={prevPluckHz}
                              newVal={pluckHz}
                              higherIsBetter={true}
                              eps={1}
                              formatDelta={(d) => `${d > 0 ? "+" : ""}${Math.round(d)} Hz`}
                            />
                          )}
                          {Math.round(pluckHz)} Hz
                          <span style={{ color: "var(--muted)", fontWeight: 500, fontSize: 11 }}>
                            ({Math.round(pretensionN)} N
                            {maxWorkingN != null ? ` / ${Math.round(maxWorkingN)} N max` : ""})
                          </span>
                        </span>
                      </div>
                    )}
                    <div className="metric-kv">
                      <span>Belt Peak Tension</span>
                      <span
                        className={`metric-val ${
                          kin.belt_slip_risk || kin.belt_overtension_risk ? "tag-bad" : "tag-ok"
                        }`}
                      >
                        {kin.belt_peak_tension_n.toFixed(1)} N {kin.belt_slip_risk ? "(SLIP)" : ""}
                      </span>
                    </div>
                    {kin.frame_belt_preload_n != null && (
                      <div className="metric-kv">
                        <span>Frame Belt Preload</span>
                        <span className="metric-val">{Math.round(kin.frame_belt_preload_n)} N</span>
                      </div>
                    )}
                    {kin.vfa_belt_tooth_speed_range_mm_s && (
                      <div
                        className="metric-kv"
                        title="Speed band where belt-tooth meshing excites stepper resonance (VFAs)"
                      >
                        <span>VFA Tooth Band</span>
                        <span className="metric-val">
                          {Math.round(kin.vfa_belt_tooth_speed_range_mm_s[0])}–
                          {Math.round(kin.vfa_belt_tooth_speed_range_mm_s[1])} mm/s
                          {kin.pulley_circumference_mm != null
                            ? ` · ${
                                Number.isInteger(kin.pulley_circumference_mm)
                                  ? kin.pulley_circumference_mm
                                  : kin.pulley_circumference_mm.toFixed(1)
                              }mm/rev`
                            : ""}
                        </span>
                      </div>
                    )}
                    {(kin.belt_validation_tier === "theoretical" ||
                      kin.belt_validation_tier === "experimental") && (
                      <div
                        className={`belt-theory-callout ${
                          kin.belt_validation_tier === "theoretical" ? "theoretical" : "experimental"
                        }`}
                      >
                        <div className="belt-theory-header">
                          <span
                            className={`belt-tier-badge ${
                              kin.belt_validation_tier === "theoretical" ? "theoretical" : "experimental"
                            }`}
                          >
                            {kin.belt_validation_tier === "theoretical"
                              ? "THEORETICAL BELT MODEL"
                              : "EXPERIMENTAL BELT"}
                          </span>
                          <span className="belt-theory-limits">
                            {kin.belt_label ?? sim.spec.gantry.belt.type}
                          </span>
                        </div>
                        {kin.belt_theoretical_summary && (
                          <div className="belt-theory-summary">{kin.belt_theoretical_summary}</div>
                        )}
                        {kin.belt_practical_issues && kin.belt_practical_issues.length > 0 && (
                          <ul className="belt-theory-issues">
                            {kin.belt_practical_issues.map((issue, idx) => (
                              <li key={idx}>{issue}</li>
                            ))}
                          </ul>
                        )}
                      </div>
                    )}
                    {(kin.motor_overcurrent_risk ||
                      kin.motor_thermal_risk ||
                      kin.belt_overtension_risk ||
                      kin.shaft_bending_risk ||
                      kin.bracket_creep_risk ||
                      kin.steel_core_fatigue_risk ||
                      kin.belt_bend_fatigue_risk ||
                      kin.belt_creep_risk) && (
                      <div className="warning-pill-stack">
                        {kin.steel_core_fatigue_risk && (
                          <div
                            className="warning-pill warn"
                            title={
                              kin.belt_min_pulley_teeth != null &&
                              kin.belt_min_back_idler_mm != null
                                ? `Minimum ${kin.belt_min_pulley_teeth}T pulley & ≥${Math.round(
                                    kin.belt_min_back_idler_mm
                                  )} mm smooth back-idler required`
                                : undefined
                            }
                          >
                            ⚠ Steel-Core Fatigue (min {kin.belt_min_pulley_teeth ?? 24}T / ≥
                            {Math.round(kin.belt_min_back_idler_mm ?? 20)}mm idler)
                          </div>
                        )}
                        {kin.belt_bend_fatigue_risk && !kin.steel_core_fatigue_risk && (
                          <div
                            className="warning-pill warn"
                            title={
                              kin.belt_back_idler_mm != null &&
                              kin.belt_min_back_idler_mm != null
                                ? `Back-idler ${kin.belt_back_idler_mm.toFixed(
                                    1
                                  )} mm vs min ${Math.round(kin.belt_min_back_idler_mm)} mm`
                                : undefined
                            }
                          >
                            ⚠ Belt Bend-Radius Risk (min {kin.belt_min_pulley_teeth ?? 16}T / ≥
                            {Math.round(kin.belt_min_back_idler_mm ?? 12)}mm back-idler)
                          </div>
                        )}
                        {kin.belt_creep_risk && (
                          <div className="warning-pill warn">
                            ⚠{" "}
                            {sim.spec.gantry.belt.core === "aramid"
                              ? "Aramid Belt Creep Risk under sustained preload"
                              : `TPU Belt Softening / Creep Risk${
                                  kin.belt_thermal_retention_pct != null
                                    ? ` (${Math.round(
                                        kin.belt_thermal_retention_pct
                                      )}% tooth stiffness)`
                                    : ""
                                }`}
                          </div>
                        )}
                        {kin.bracket_creep_risk && (
                          <div className="warning-pill bad">
                            ⚠ Belt tension exceeds gantry bracket creep limit
                          </div>
                        )}
                        {kin.motor_overcurrent_risk && (
                          <div className="warning-pill bad">
                            ⚠ Motor run current exceeds rated Amps — overheating &amp; demagnetization risk
                          </div>
                        )}
                        {kin.motor_thermal_risk && !kin.motor_overcurrent_risk && (
                          <div className="warning-pill bad">
                            ⚠ Motor winding temp (~{Math.round(kin.motor_est_temp_c ?? 0)}°C) approaches Class{" "}
                            {sim.spec.kinematics.xy_stepper.insulation_class ?? "B"} (
                            {Math.round(kin.motor_max_winding_temp_c ?? 130)}°C) limit — reduce Amps or use a Class H motor
                          </div>
                        )}
                        {kin.belt_overtension_risk && (
                          <div className="warning-pill bad">
                            ⚠ Peak dynamic tension {kin.belt_peak_tension_n.toFixed(1)} N (static + accel load)
                            {kin.belt_max_working_tension_n != null
                              ? ` exceeds ${Math.round(kin.belt_max_working_tension_n)} N working rating`
                              : " exceeds working tension rating"}
                          </div>
                        )}
                        {kin.shaft_bending_risk && (
                          <div className="warning-pill warn">
                            ⚠ High tension (&gt;45 N) without double-shear pulleys
                          </div>
                        )}
                      </div>
                    )}
                    {mod.modes && mod.modes.length > 0 && (
                      <div
                        style={{
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "space-between",
                          gap: 8,
                          marginTop: 6,
                          paddingTop: 7,
                          borderTop: "1px dashed rgba(255, 255, 255, 0.1)",
                          fontSize: 11,
                          color: "var(--muted)",
                        }}
                      >
                        <span>
                          {mod.modes.length} structural modes solved (f₁ ={" "}
                          <strong style={{ color: "var(--text)" }}>
                            {mod.fundamental_freq_hz.toFixed(1)} Hz
                          </strong>
                          )
                        </span>
                        <button
                          type="button"
                          className="deep-verify-cta"
                          style={{ padding: "2px 8px", fontSize: 11 }}
                          onClick={() => setRightTab("shaper")}
                          title="Open Shaper tab to view Klipper PSD graphs and clickable 3D modal modes"
                        >
                          Shaper &amp; Modes →
                        </button>
                      </div>
                    )}
                  </>
                );
              })()}
            </DeepDiveAccordion>

            {/* Accordion 3: Weight-Cut Shaper Potential */}
            {hasWeightGains && sim.weight_shaper_gains && (
              <DeepDiveAccordion
                title="Weight-Cut Shaper Potential"
                badge={weightAccordionBadge}
                updated={modalChanged}
                open={Boolean(sectionOpen["weight_potential"])}
                onToggle={() => toggleSection("weight_potential")}
              >
                <WeightShaperPotentialCard
                  gains={sim.weight_shaper_gains}
                  changed={modalChanged}
                  embedded
                />
              </DeepDiveAccordion>
            )}

            {/* Accordion 4: Metrology, Circularity & Workspace FRF */}
            {hasMetrologyData && (
              <DeepDiveAccordion
                title="Metrology, Circularity & Workspace FRF"
                badge={metrologyAccordionBadge}
                open={Boolean(sectionOpen["metrology_frf"])}
                onToggle={() => toggleSection("metrology_frf")}
              >
                <CncMetrologyContent
                  metrology={sim.time_domain_metrology}
                  workspaceFrf={sim.workspace_frf}
                  rcsa={sim.rcsa}
                  spec={sim.spec}
                />
              </DeepDiveAccordion>
            )}

            {/* Accordion 5: Adjoint Sensitivity & Upgrade ROI */}
            {hasSensitivityData && sim.sensitivity && (
              <DeepDiveAccordion
                title="Adjoint Sensitivity & Upgrade ROI"
                badge={sensitivityAccordionBadge}
                open={Boolean(sectionOpen["sensitivity"])}
                onToggle={() => toggleSection("sensitivity")}
              >
                <AdjointSensitivityContent
                  sensitivity={sim.sensitivity}
                  viewMode={viewMode}
                  onViewModeChange={onViewModeChange}
                />
              </DeepDiveAccordion>
            )}

            {/* Accordion 7: Hotend Thermal & Virtual Print */}
            <DeepDiveAccordion
              title="Hotend Thermal & Virtual Print"
              badge={thermalPrintAccordionBadge}
              updated={thermalChanged || printChanged}
              open={Boolean(sectionOpen["thermal_print"])}
              onToggle={() => toggleSection("thermal_print")}
            >
              <div className="section-title">
                <span>Hotend FVM Thermal &amp; Cooling ({th.filament_id.toUpperCase()})</span>
              </div>
              <div className="metric-kv">
                <span>Flow Bottleneck</span>
                <span className="metric-val">{th.flow_limiting_factor.replace(/_/g, " ")}</span>
              </div>
              <div className="metric-kv">
                <span>Eq. Speed (0.2×0.42mm)</span>
                <span className="metric-val">{th.max_linear_speed_02x04_mm_s.toFixed(0)} mm/s</span>
              </div>
              <div className="metric-kv">
                <span>Heat-up ({th.target_temp_c}°C) / Bed</span>
                <span className="metric-val">
                  {th.heatup_time_215c_s.toFixed(0)}s / {th.bed_heatup_time_100c_s.toFixed(0)}s
                </span>
              </div>
              <div className="metric-kv">
                <span>Cold Throat / Tg Limit</span>
                <span className={`metric-val ${th.heat_creep_risk ? "tag-bad" : "tag-ok"}`}>
                  {th.cold_side_temp_c.toFixed(1)}°C / {th.glass_transition_c}°C{" "}
                  {th.heat_creep_risk ? "(CREEP)" : "(OK)"}
                </span>
              </div>
              <div className="metric-kv">
                <span>Max Overhang / 40mm Bridge</span>
                <span className="metric-val">
                  {th.max_overhang_deg.toFixed(0)}° • {th.bridge_sag_40mm_mm.toFixed(2)} mm sag
                </span>
              </div>
              {th.chamber_cpap && (
                <div className="sub-block">
                  <div className="sub-block-title">
                    <span>Chamber &amp; CPAP Cooling</span>
                    {th.chamber_cpap.kalico_shared_duct_active && (
                      <span className="badge-pill info">Kalico Shared Duct</span>
                    )}
                  </div>
                  <div
                    className="metric-kv"
                    title={`Enclosure heat loss UA = ${th.chamber_cpap.enclosure_ua_w_per_k.toFixed(2)} W/K · Heater ${Math.round(th.chamber_cpap.chamber_heater_power_w)} W · PIR ${Math.round(th.chamber_cpap.chamber_insulation_mm)} mm`}
                  >
                    <span>
                      Chamber Equilibrium / 60 °C Time
                      {th.chamber_cpap.chamber_heater_power_w > 0 && (
                        <span className="metric-sub-inline">
                          {Math.round(th.chamber_cpap.chamber_heater_power_w)} W PTC
                        </span>
                      )}
                    </span>
                    <span className="metric-val">
                      {th.chamber_cpap.chamber_steady_temp_c.toFixed(1)} °C ·{" "}
                      {th.chamber_cpap.chamber_heatup_time_60c_s != null
                        ? `${(th.chamber_cpap.chamber_heatup_time_60c_s / 60).toFixed(1)} min`
                        : "N/A"}
                    </span>
                  </div>
                  {th.chamber_cpap.cpap_blower_id !== "none" && (
                    <>
                      <div
                        className="metric-kv"
                        title={`Hose ID ${th.chamber_cpap.cpap_hose_id_mm} mm (flow factor ${th.chamber_cpap.cpap_hose_flow_factor.toFixed(2)}, +${th.chamber_cpap.cpap_hose_moving_mass_g.toFixed(0)} g moving mass) @ ${Math.round(th.chamber_cpap.cpap_effective_pwm_pct)}% eff. PWM`}
                      >
                        <span>CPAP Delivered Flow</span>
                        <span className="metric-val">
                          {th.chamber_cpap.cpap_delivered_dm3_s.toFixed(1)} dm³/s ·{" "}
                          {th.chamber_cpap.cpap_delivered_cfm.toFixed(1)} CFM
                        </span>
                      </div>
                      <div
                        className="metric-kv"
                        title="Compression & blower motor heat added to CPAP air stream at current PWM"
                      >
                        <span>CPAP Exhaust ΔT / Outlet</span>
                        <span
                          className={`metric-val ${
                            th.chamber_cpap.cpap_exhaust_temp_rise_c >= 6 ? "tag-warn" : ""
                          }`}
                        >
                          +{th.chamber_cpap.cpap_exhaust_temp_rise_c.toFixed(1)} °C ·{" "}
                          {th.chamber_cpap.cpap_outlet_air_temp_c.toFixed(1)} °C
                        </span>
                      </div>
                    </>
                  )}
                </div>
              )}
              {axPts.length > 0 && (
                <svg viewBox="0 0 260 68" style={{ width: "100%", height: 68, marginTop: 4 }}>
                  <rect x="0" y="0" width="260" height="68" rx="4" fill="#0d131d" />
                  <line x1="0" y1={tgY} x2="260" y2={tgY} stroke="#ef4444" strokeDasharray="3 3" strokeWidth="1" />
                  <polyline fill="none" stroke="#38bdf8" strokeWidth="2" points={axSvgPoints} />
                  <text x="6" y="13" fill="#8b9bb4" fontSize="8.5">
                    FVM Axial T(z): Nozzle → Block → Heatbreak → Sink
                  </text>
                  <text x="198" y={Math.max(22, tgY - 3)} fill="#ef4444" fontSize="8">
                    Tg {th.glass_transition_c}°C
                  </text>
                </svg>
              )}
              <svg viewBox="0 0 260 66" style={{ width: "100%", height: 62, marginTop: 2 }}>
                <rect x="0" y="0" width="260" height="66" rx="4" fill="#0d131d" />
                <polyline fill="none" stroke="#f97316" strokeWidth="2" points={svgPoints} />
                <text x="6" y="14" fill="#8b9bb4" fontSize="9">
                  Hotend Heat-up ({th.target_temp_c}°C in {th.heatup_time_215c_s.toFixed(0)}s)
                </text>
              </svg>

              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  gap: 6,
                  marginTop: 6,
                  paddingTop: 8,
                  borderTop: "1px dashed rgba(255, 255, 255, 0.09)",
                }}
              >
                <div className="section-title">
                  <span>Virtual G-code Print</span>
                </div>
                <div className="metric-kv">
                  <span>Avg / Peak Print Speed</span>
                  <span className="metric-val">
                    {vp.avg_print_speed_mm_s.toFixed(0)} / {vp.peak_print_speed_mm_s.toFixed(0)} mm/s
                  </span>
                </div>
                <div className="metric-kv">
                  <span>Peak Flow / Ringing</span>
                  <span className="metric-val">
                    {vp.peak_volumetric_flow_mm3_s.toFixed(1)} mm³/s | {(vp.max_ringing_severity * 100).toFixed(0)}%
                  </span>
                </div>
                {(vp.max_corner_smoothing_um ?? 0) > 0 && (
                  <div className="metric-kv">
                    <span>Max Corner Smoothing</span>
                    <span className="metric-val">
                      {(vp.max_corner_smoothing_um ?? 0).toFixed(1)} µm
                    </span>
                  </div>
                )}
                <div className="metric-kv">
                  <span>Defect Segments</span>
                  <span className="metric-val">
                    {vp.underextrusion_segments} flow • {vp.ringing_segments} ring • {vp.step_skip_segments} skip
                    {(vp.smoothing_segments ?? 0) > 0 ? ` • ${vp.smoothing_segments} smooth` : ""}
                    {(vp.belt_slip_segments ?? 0) > 0 ? ` • ${vp.belt_slip_segments} slip` : ""}
                  </span>
                </div>
              </div>
            </DeepDiveAccordion>
          </>
        )}

        {rightTab === "bom" && (
          <>
            <div className="section-card">
              <div className="section-title">
                <span>Stock Bar &amp; Export</span>
                {bomLoading && <span style={{ color: "var(--muted)", fontSize: 11 }}>Updating...</span>}
              </div>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8 }}>
                <div className="chip-group">
                  {[1000, 2000].map((len) => (
                    <button
                      key={len}
                      className={`chip-btn ${stockLengthMm === len ? "active" : ""}`}
 aria-pressed={stockLengthMm === len}
                      onClick={() => onStockLengthChange(len)}
                    >
                      {len}mm Stock
                    </button>
                  ))}
                </div>
                <button
                  className="btn primary"
                  style={{ padding: "5px 10px", fontSize: 12 }}
                  disabled={!bom}
                  onClick={onDownloadBomCsv}
                >
                  Download CSV
                </button>
              </div>
            </div>

            {!bom ? (
              <div className="section-card" style={{ color: "var(--muted)", fontSize: 12 }}>
                {bomLoading ? "Generating Bill of Materials & cut list..." : "No BOM data available yet."}
              </div>
            ) : (
              <>
                <div className="kpi-grid">
                  <div className="kpi-box">
                    <span className="kpi-label">Total BOM Cost</span>
                    <span className="kpi-value" style={{ color: "var(--green)" }}>
                      ${bom.total_cost_usd.toFixed(2)}
                    </span>
                    <span className="kpi-sub">{bom.items.length} line items</span>
                  </div>
                  <div className="kpi-box">
                    <span className="kpi-label">Total Mass</span>
                    <span className="kpi-value" style={{ color: "var(--accent)" }}>
                      {bom.total_mass_kg.toFixed(2)} kg
                    </span>
                    <span className="kpi-sub">{bom.architecture}</span>
                  </div>
                </div>

                <div className="kpi-grid-3">
                  <div className="kpi-box">
                    <span className="kpi-label">Extrusion</span>
                    <span className="kpi-value" style={{ fontSize: 13 }}>
                      {bom.total_extrusion_length_m.toFixed(2)} m
                    </span>
                  </div>
                  <div className="kpi-box">
                    <span className="kpi-label">Belt</span>
                    <span className="kpi-value" style={{ fontSize: 13 }}>
                      {bom.total_belt_length_m.toFixed(2)} m
                    </span>
                  </div>
                  <div className="kpi-box">
                    <span className="kpi-label">Fasteners</span>
                    <span className="kpi-value" style={{ fontSize: 13 }}>
                      {bom.total_fasteners}
                    </span>
                  </div>
                </div>

                <div className="section-card">
                  <div className="section-title">Category Cost Breakdown</div>
                  {categoryBreakdown.map(({ cat, cost }) => {
                    const pct = (cost / totalCategoryCost) * 100;
                    const barColor = CATEGORY_COLORS[cat] || "var(--accent)";
                    return (
                      <div className="bar-row" key={cat}>
                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12 }}>
                          <span style={{ color: "var(--muted)", textTransform: "capitalize" }}>{cat}</span>
                          <span style={{ fontVariantNumeric: "tabular-nums" }}>
                            <strong>${cost.toFixed(2)}</strong>{" "}
                            <span style={{ color: "var(--muted)", fontSize: 11 }}>({pct.toFixed(0)}%)</span>
                          </span>
                        </div>
                        <div className="bar-track">
                          <div className="bar-fill" style={{ width: `${Math.min(100, pct)}%`, background: barColor }} />
                        </div>
                      </div>
                    );
                  })}
                </div>

                <div className="section-card">
                  <div className="section-title">
                    <span>Extrusion &amp; Structural Cut-List</span>
                    <span style={{ color: "var(--muted)", fontSize: 11 }}>{bom.cut_list.length} cuts</span>
                  </div>
                  <div className="bom-table-wrap">
                    <table className="bom-table">
                      <thead>
                        <tr>
                          <th>Profile</th>
                          <th className="num">Len</th>
                          <th className="num">Qty</th>
                          <th className="num">Mass</th>
                          <th className="num">Cost</th>
                        </tr>
                      </thead>
                      <tbody>
                        {bom.cut_list.map((c) => (
                          <tr key={`${c.part_code}-${c.length_mm}`} title={`${c.part_code}: ${c.name} (${c.notes || c.role})`}>
                            <td>
                              <div className="bom-item-name">
                                {splitBomName(c.name, c.profile_id).title}
                              </div>
                              <div className="bom-item-sub">
                                {splitBomName(c.name, c.profile_id).sub}
                              </div>
                            </td>
                            <td className="num">{Math.round(c.length_mm)}mm</td>
                            <td className="num">×{c.quantity}</td>
                            <td className="num">{Math.round(c.total_mass_g)}g</td>
                            <td className="num">${c.total_cost_usd.toFixed(2)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                {bom.stock_nesting.length > 0 && (
                  <div className="section-card">
                    <div className="section-title">
                      <span>Stock Bar Nesting</span>
                      <span style={{ color: "var(--muted)", fontSize: 11 }}>{stockLengthMm}mm bars</span>
                    </div>
                    {bom.stock_nesting.map((sn) => (
                      <div className="nesting-group" key={`${sn.profile_id}-${sn.material_id}`}>
                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11 }}>
                          <strong>
                            {
                              splitBomName(
                                bom.cut_list.find((c) => c.profile_id === sn.profile_id)?.name ?? "",
                                sn.profile_id
                              ).sub
                            }
                          </strong>
                          <span style={{ color: "var(--muted)", fontVariantNumeric: "tabular-nums" }}>
                            {sn.bars_required} bar{sn.bars_required === 1 ? "" : "s"} •{" "}
                            <span className={sn.waste_pct > 25 ? "tag-warn" : "tag-ok"}>
                              {sn.waste_pct.toFixed(1)}% scrap
                            </span>
                          </span>
                        </div>
                        {sn.bar_cuts_mm.map((cuts, bIdx) => {
                          const used = cuts.reduce((a, b) => a + b, 0);
                          const waste = Math.max(0, sn.stock_length_mm - used);
                          const wastePct = (waste / Math.max(sn.stock_length_mm, 1)) * 100;
                          return (
                            <div className="nesting-bar-row" key={bIdx}>
                              <span className="nesting-bar-idx">#{bIdx + 1}</span>
                              <div className="nesting-track">
                                {cuts.map((cutLen, cIdx) => {
                                  const segPct = (cutLen / Math.max(sn.stock_length_mm, 1)) * 100;
                                  const bg = NESTING_PALETTE[cIdx % NESTING_PALETTE.length];
                                  return (
                                    <div
                                      key={cIdx}
                                      className="nesting-seg"
                                      style={{ width: `${segPct}%`, background: bg }}
                                      title={`Cut ${cIdx + 1}: ${Math.round(cutLen)} mm`}
                                    >
                                      {segPct >= 14 ? `${Math.round(cutLen)}` : ""}
                                    </div>
                                  );
                                })}
                                {waste > 0.5 && (
                                  <div
                                    className="nesting-waste"
                                    style={{ width: `${wastePct}%` }}
                                    title={`Scrap: ${Math.round(waste)} mm`}
                                  >
                                    {wastePct >= 15 ? `${Math.round(waste)}` : ""}
                                  </div>
                                )}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    ))}
                  </div>
                )}

                <div className="section-card">
                  <div className="section-title">
                    <span>Hardware BOM</span>
                    <span style={{ color: "var(--muted)", fontSize: 11 }}>{filteredBomItems.length} items</span>
                  </div>
                  <div className="chip-group">
                    <button
                      className={`chip-btn ${bomCategoryFilter === "all" ? "active" : ""}`}
 aria-pressed={bomCategoryFilter === "all"}
                      onClick={() => setBomCategoryFilter("all")}
                    >
                      All
                    </button>
                    {bomCategories.map((cat) => (
                      <button
                        key={cat}
                        className={`chip-btn ${bomCategoryFilter === cat ? "active" : ""}`}
 aria-pressed={bomCategoryFilter === cat}
                        onClick={() => setBomCategoryFilter(cat)}
                      >
                        {cat}
                      </button>
                    ))}
                  </div>
                  <div className="bom-table-wrap">
                    <table className="bom-table">
                      <thead>
                        <tr>
                          <th>Component</th>
                          <th className="num">Qty</th>
                          <th className="num">Cost</th>
                        </tr>
                      </thead>
                      <tbody>
                        {filteredBomItems.map((it) => (
                          <tr key={`${it.sku}-${it.name}`} title={`${it.sku} • ${it.specification}`}>
                            <td>
                              <div className="bom-item-name">{it.name}</div>
                              <div className="bom-item-sub">
                                {it.sku} • {it.specification}
                              </div>
                            </td>
                            <td className="num">
                              {it.quantity} {it.unit}
                            </td>
                            <td className="num">${it.total_cost_usd.toFixed(2)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </>
            )}
          </>
        )}
      </div>
    </aside>
  );
};
