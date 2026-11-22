import React, { useEffect, useMemo, useState } from "react";
import {
  AccelBottleneckData,
  AccelBottleneckShares,
  AxisShaperGraph,
  BedPlateFeaData,
  CompareMode,
  DeepVerifyResponse,
  GeometryExtrusionFillingData,
  KlipperShaperCandidate,
  PrinterBOMResponse,
  RCSAData,
  SensitivityData,
  SimulationResponse,
  SolidFeaEngine,
  SolidFEAResponse,
  SubsystemKey,
  TetFEAResponse,
  TimeDomainMetrologyData,
  ToolheadFlexBreakdownData,
  TopologyOptResponse,
  ViewMode,
  WeightShaperGainsSummary,
  WorkspaceFRFData,
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
  bom: PrinterBOMResponse | null;
  bomLoading: boolean;
  stockLengthMm: number;
  onStockLengthChange: (lenMm: number) => void;
  onDownloadBomCsv: () => void;
  showBedWarp?: boolean;
  onInspectBedWarp?: () => void;
  toolheadPunchIn?: boolean;
  onToggleToolheadPunchIn?: () => void;
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
  const curCost = curBom?.total_cost_usd ?? cur.geometry.mass_summary.estimated_material_cost_usd;
  const cmpCost = cmpBom?.total_cost_usd ?? cmp.geometry.mass_summary.estimated_material_cost_usd;

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
      label: "Safe Max Accel",
      oldVal: cmp.modal_fea.recommended_max_accel_mm_s2,
      newVal: cur.modal_fea.recommended_max_accel_mm_s2,
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
      label: "Nozzle–Bed Flex",
      oldVal: cmp.static_fea.combined_max_deflection_um,
      newVal: cur.static_fea.combined_max_deflection_um,
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
        <span style={{ color: "var(--purple)", fontSize: 10.5 }}>
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

const TetFEACard: React.FC<{ tet: TetFEAResponse }> = ({ tet }) => {
  const isSafe = tet.safety_factor >= 1.5 && tet.failure_mode === "safe";
  const backendLabel = formatBackendLabel(tet.backend);
  const elemLabel = formatElementLabel(tet.element_type);

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
        <span>Max Displacement</span>
        <span className="metric-val">{tet.max_displacement_um.toFixed(1)} µm</span>
      </div>
      <div className="metric-kv">
        <span>Stiffness</span>
        <span className="metric-val">{tet.stiffness_n_per_mm.toFixed(1)} N/mm</span>
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
      <div className="metric-kv">
        <span>Safety Factor</span>
        <span className={`metric-val ${isSafe ? "tag-ok" : "tag-bad"}`}>
          {tet.safety_factor.toFixed(2)}× ({tet.failure_mode.replace(/_/g, " ")})
        </span>
      </div>
    </div>
  );
};

const SolidHexFEACard: React.FC<{ solidFea: SolidFEAResponse }> = ({ solidFea }) => (
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
  </div>
);

const KlipperShaperPsdCard: React.FC<{
  graphs: { x: AxisShaperGraph; y: AxisShaperGraph };
  frameDampingRatio?: number;
  extrusionFilling?: GeometryExtrusionFillingData;
  changed: boolean;
}> = ({ graphs, frameDampingRatio, extrusionFilling, changed }) => {
  const [axis, setAxis] = useState<"x" | "y">("x");
  const [selectedShaperName, setSelectedShaperName] = useState<string | null>(null);

  const graph = axis === "x" ? graphs.x : graphs.y;

  // Reset shaper inspection to recommended when axis or simulation graph refreshes
  useEffect(() => {
    setSelectedShaperName(null);
  }, [axis, graph.recommended_shaper, graph.peak_freq_hz]);

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

  return (
    <div className={`section-card psd-card ${changed ? "card-changed" : ""}`}>
      <div className="section-title">
        <span>Klipper ADXL345 Resonance (PSD)</span>
        <span className="psd-peak-pill">
          Peak {graph.peak_freq_hz.toFixed(1)} Hz
        </span>
      </div>
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

const AccelBottleneckCard: React.FC<{
  bottleneck: AccelBottleneckData;
  changed: boolean;
}> = ({ bottleneck, changed }) => {
  const [selectedAxis, setSelectedAxis] = useState<"X" | "Y">(bottleneck.limiting_axis);

  useEffect(() => {
    setSelectedAxis(bottleneck.limiting_axis);
  }, [bottleneck.limiting_axis]);

  const activeShares: AccelBottleneckShares =
    selectedAxis === "X"
      ? bottleneck.x_shares_pct ?? bottleneck.shares_pct
      : bottleneck.y_shares_pct ?? bottleneck.shares_pct;

  const totalShare = Math.max(
    BOTTLENECK_SOURCES.reduce((acc, s) => acc + Math.max(0, activeShares[s.key] ?? 0), 0),
    1e-6
  );

  const primaryMeta =
    BOTTLENECK_SOURCES.find((s) => s.key === bottleneck.primary_source) ?? BOTTLENECK_SOURCES[0];

  // Dominant source for the currently viewed axis (highlights the biggest contributor in the legend)
  let activeDominantKey: keyof AccelBottleneckShares = bottleneck.primary_source;
  let activeMaxVal = -1;
  for (const s of BOTTLENECK_SOURCES) {
    const v = activeShares[s.key] ?? 0;
    if (v > activeMaxVal) {
      activeMaxVal = v;
      activeDominantKey = s.key;
    }
  }

  const formatAxisLine = (shares: AccelBottleneckShares) =>
    BOTTLENECK_SOURCES.map((s) => `${s.shortLabel} ${Math.round(shares[s.key] ?? 0)}%`).join(" · ");

  return (
    <div className={`section-card bottleneck-card ${changed ? "card-changed" : ""}`}>
      <div className="section-title bottleneck-header">
        <span>Accel Limit Breakdown</span>
        <span
          className="bottleneck-pill"
          style={{
            background: `${primaryMeta.color}22`,
            border: `1px solid ${primaryMeta.color}66`,
            color: primaryMeta.color,
          }}
        >
          Limiting: {bottleneck.limiting_axis} · {Math.round(bottleneck.primary_share_pct)}%{" "}
          {bottleneck.primary_label}
        </span>
      </div>

      <div className="bottleneck-axis-row">
        <span>Compliance Share ({selectedAxis}-Axis)</span>
        <div className="bottleneck-axis-tabs">
          {(["X", "Y"] as const).map((ax) => {
            const isLimiting = ax === bottleneck.limiting_axis;
            return (
              <button
                key={ax}
                type="button"
                className={`bottleneck-axis-btn ${selectedAxis === ax ? "active" : ""}`}
                onClick={() => setSelectedAxis(ax)}
                title={isLimiting ? `${ax}-axis is the limiting acceleration axis` : `Inspect ${ax}-axis compliance share`}
              >
                {ax}
                {isLimiting ? " ★" : ""}
              </button>
            );
          })}
        </div>
      </div>

      <div className="bottleneck-bar" role="img" aria-label={`Acceleration compliance share on ${selectedAxis}-axis`}>
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
          return (
            <div key={src.key} className={`bottleneck-legend-item ${isDom ? "primary" : ""}`}>
              <span className="bottleneck-legend-left">
                <span className="swatch-dot" style={{ background: src.color }} />
                <span>{src.label}</span>
              </span>
              <span className="bottleneck-legend-pct" style={{ color: isDom ? src.color : undefined }}>
                {val.toFixed(0)}%
              </span>
            </div>
          );
        })}
      </div>

      <div className="bottleneck-xy-strip">
        <div className={`bottleneck-xy-line ${bottleneck.limiting_axis === "X" ? "is-limiting" : ""}`}>
          <span className="bottleneck-xy-axis">X:</span>
          <span>{formatAxisLine(bottleneck.x_shares_pct ?? bottleneck.shares_pct)}</span>
        </div>
        <div className={`bottleneck-xy-line ${bottleneck.limiting_axis === "Y" ? "is-limiting" : ""}`}>
          <span className="bottleneck-xy-axis">Y:</span>
          <span>{formatAxisLine(bottleneck.y_shares_pct ?? bottleneck.shares_pct)}</span>
        </div>
      </div>

      {bottleneck.recommendation && (
        <div className="bottleneck-tip" style={{ borderLeftColor: primaryMeta.color }}>
          {bottleneck.recommendation}
        </div>
      )}
    </div>
  );
};

const WeightShaperPotentialCard: React.FC<{
  gains: WeightShaperGainsSummary;
  changed: boolean;
}> = ({ gains, changed }) => {
  const axisTagStyle = (
    axis: "X" | "Y" | "XY",
    affectsLimiting: boolean
  ): React.CSSProperties => ({
    display: "inline-flex",
    alignItems: "center",
    padding: "0 4px",
    borderRadius: 4,
    fontSize: 8.5,
    fontWeight: 700,
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
      className={`section-card ${changed ? "card-changed" : ""}`}
      style={{
        gap: 8,
        borderColor: "rgba(56, 189, 248, 0.32)",
      }}
    >
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
        <span style={{ color: "#7dd3fc", fontWeight: 600 }}>
          (+{Math.round(gains.total_potential_accel_gain_mm_s2).toLocaleString()} mm/s²)
        </span>
      </div>

      <div className="psd-table-wrap" style={{ overflowX: "auto" }}>
        <table className="psd-table" style={{ width: "100%", fontSize: 10.5 }}>
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
                    <div style={{ fontSize: 9, color: "var(--muted)", lineHeight: 1.15 }}>
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
                        fontSize: 9,
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

  return (
    <div className={`section-card deep-verify-card ${convergence.converged ? "converged" : "refining"}`}>
      <div className="deep-verify-header">
        <span className="section-title" style={{ margin: 0 }}>
          Deep Verify Convergence
        </span>
        <div className="deep-verify-header-right">
          <span className={`badge-pill ${convergence.converged ? "ok" : "warn"}`}>
            {convergence.converged ? "Converged" : "Mesh Sensitive"}
          </span>
          <button
            type="button"
            className="deep-verify-close"
            onClick={onDismiss}
            title="Dismiss Deep Verify results"
            aria-label="Dismiss Deep Verify results"
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
        </span>
        <span>
          <strong>{deep.num_shell_elements.toLocaleString()}</strong> shells •{" "}
          <strong>{Math.round(deep.solve_time_ms)} ms</strong>
        </span>
      </div>

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
            onClick={() => onToggleViewingDeepSim(false)}
            title={`Show Fast ${fast_mesh_n}×${fast_mesh_n} MITC4 simulation in 3D viewport`}
          >
            Fast {fast_mesh_n}×{fast_mesh_n}
          </button>
          <button
            type="button"
            className={`impact-seg-btn ${viewingDeepSim ? "active" : ""}`}
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

function workspaceHeatColors(hz: number, minHz: number, maxHz: number): { bg: string; border: string } {
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

const CncControlMetrologyCard: React.FC<{
  metrology?: TimeDomainMetrologyData;
  workspaceFrf?: WorkspaceFRFData;
  rcsa?: RCSAData;
  sensitivity?: SensitivityData;
  viewMode: ViewMode;
  onViewModeChange: (m: ViewMode) => void;
}> = ({ metrology, workspaceFrf, rcsa, sensitivity, viewMode, onViewModeChange }) => {
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
    };
  }, [metrology?.corner_trace]);

  const hasMetrology = Boolean(metrology);
  const hasWorkspace = Boolean(
    workspaceFrf && (workspaceFrf.min_axis_freq_grid_hz?.length ?? 0) > 0
  );
  const hasRcsa = Boolean(rcsa);
  const topStiffen = sensitivity?.top_stiffen_targets?.slice(0, 3) ?? [];
  const topLighten = sensitivity?.top_lighten_targets?.slice(0, 2) ?? [];
  const hasSensitivity = Boolean(
    sensitivity &&
      (topStiffen.length > 0 ||
        topLighten.length > 0 ||
        Boolean(sensitivity.summary_diagnosis) ||
        Boolean(sensitivity.subsystem_breakdown))
  );

  if (!hasMetrology && !hasWorkspace && !hasRcsa && !hasSensitivity) {
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
    <div className="section-card cnc-metrology-card">
      <div className="section-title">
        <span>CNC Control &amp; Metrology</span>
        {workspaceFrf?.drive_type && (
          <span className="badge-pill" style={{ fontSize: 9.5, padding: "1px 6px" }}>
            {workspaceFrf.drive_type.replace(/_/g, " ")}
          </span>
        )}
      </div>

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

      {/* 2. 90° Corner Ringing Zoom SVG Plot */}
      {metrology && (
        <div className="metrology-subblock">
          <div className="metrology-subhead">
            <span>90° Corner Ringing Zoom</span>
            {cornerSvg && cornerSvg.devGain > 1 && (
              <span className="metrology-submeta">×{cornerSvg.devGain} error zoom</span>
            )}
          </div>

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
              <line x1="10" y1="11" x2="22" y2="11" stroke="#e2e8f0" strokeWidth="1.2" strokeDasharray="3 2" />
              <text x="25" y="14" fill="#94a3b8" fontSize="8">
                Ideal
              </text>
              <line x1="62" y1="11" x2="74" y2="11" stroke="#ef4444" strokeWidth="1.5" strokeOpacity="0.7" />
              <text x="77" y="14" fill="#fca5a5" fontSize="8">
                Unshaped
              </text>
              <line x1="132" y1="11" x2="144" y2="11" stroke="#10b981" strokeWidth="2" />
              <text x="147" y="14" fill="#6ee7b7" fontSize="8">
                Shaped
              </text>
            </svg>
          )}

          <div className="kpi-grid-3">
            <div
              className="kpi-box"
              title={`Corner rounding ${(metrology.corner_rounding_um ?? 0).toFixed(1)} µm • Overshoot ${(metrology.corner_overshoot_um ?? 0).toFixed(1)} µm`}
            >
              <span className="kpi-label">Rounding</span>
              <span className="kpi-value" style={{ fontSize: 13 }}>
                {(metrology.corner_rounding_um ?? 0).toFixed(1)} µm
              </span>
              <span className="kpi-sub">
                +{(metrology.corner_overshoot_um ?? 0).toFixed(0)}µm over
              </span>
            </div>
            <div
              className="kpi-box"
              title={`Peak-to-peak ringing ${(metrology.ringing_peak_to_peak_um ?? 0).toFixed(1)} µm vs ${(metrology.ringing_unshaped_um ?? 0).toFixed(1)} µm unshaped`}
            >
              <span className="kpi-label">Ringing</span>
              <span
                className={`kpi-value ${
                  (metrology.ringing_peak_to_peak_um ?? 0) > 45
                    ? "tag-bad"
                    : (metrology.ringing_peak_to_peak_um ?? 0) > 20
                      ? "tag-warn"
                      : "tag-ok"
                }`}
                style={{ fontSize: 13 }}
              >
                {(metrology.ringing_peak_to_peak_um ?? 0).toFixed(1)} µm
              </span>
              <span className="kpi-sub" style={{ color: "var(--green)" }}>
                -{Math.round(metrology.ringing_reduction_pct ?? 0)}% vs raw
              </span>
            </div>
            <div className="kpi-box" title="Distance for residual ringing to decay below visible threshold">
              <span className="kpi-label">Decay</span>
              <span className="kpi-value" style={{ fontSize: 13 }}>
                {(metrology.ringing_decay_mm ?? 0).toFixed(1)} mm
              </span>
              <span className="kpi-sub">
                raw {(metrology.ringing_unshaped_um ?? 0).toFixed(0)} µm
              </span>
            </div>
          </div>
        </div>
      )}

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

      {/* 4. Adjoint Sensitivity Hotspots */}
      {hasSensitivity && (
        <div className="metrology-subblock">
          <div className="metrology-subhead">
            <span>Adjoint Sensitivity Hotspots</span>
            <button
              type="button"
              className={`deep-verify-cta ${viewMode === "sensitivity" ? "active" : ""}`}
              style={{ padding: "2px 8px", fontSize: 10 }}
              onClick={() => onViewModeChange("sensitivity")}
              title="Color structural frame & panel bays by adjoint Hz/USD & Hz/100g ROI in the 3D viewport"
            >
              {viewMode === "sensitivity" ? "Viewing in 3D" : "View in 3D"}
            </button>
          </div>

          {sensitivity?.summary_diagnosis && (
            <div className="sens-diagnosis-banner">{sensitivity.summary_diagnosis}</div>
          )}

          {sensitivity?.subsystem_breakdown && (() => {
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
                    <span className="badge-pill bad" style={{ padding: "0 6px", fontSize: 9.5 }}>
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
                    return (
                      <div
                        key={k}
                        className={`sens-sub-chip sev-${sev} ${isPrimary ? "primary" : ""}`}
                        title={
                          entry
                            ? `${entry.label}: ${entry.compliance_share_pct.toFixed(1)}% flex • Stiff sens ${entry.modal_stiff_hz.toFixed(
                                2
                              )} Hz • Mass sens ${entry.modal_mass_hz.toFixed(2)} Hz`
                            : SUBSYSTEM_DISPLAY[k].full
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
                      </div>
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
              {topStiffen.map((t) => (
                <div
                  key={`stiff-${t.id}`}
                  className="sens-row stiffen"
                  title={t.action_hint || `${t.label ?? t.id} (${t.id})`}
                >
                  <div className="sens-row-top">
                    <span className="sens-id">
                      <strong>{t.label ?? t.id}</strong>
                      {!t.label && <span className="sens-role">{t.role}</span>}
                    </span>
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
              ))}
            </div>
          )}

          {topLighten.length > 0 && (
            <div className="sens-group">
              <div className="sens-group-label lighten">Primary Inertia Hotspots (Moving-Mass Limits)</div>
              {topLighten.map((t) => {
                const gainPer100g = t.hz_gain_per_100g_saved ?? Math.abs(t.hz_per_100g);
                return (
                  <div
                    key={`light-${t.id}`}
                    className="sens-row lighten"
                    title={t.action_hint || `${t.label ?? t.id} (${t.id})`}
                  >
                    <div className="sens-row-top">
                      <span className="sens-id">
                        <strong>{t.label ?? t.id}</strong>
                        {!t.label && <span className="sens-role">{t.role}</span>}
                      </span>
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
      )}
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
  showBedWarp?: boolean;
  onInspectBedWarp?: () => void;
}> = ({ bedFea, cmpBedFea, showBedWarp, onInspectBedWarp }) => {
  const riskMeta = BED_RISK_META[bedFea.first_layer_risk] ?? BED_RISK_META.low;
  const modeTypeLabel =
    BED_MODE_TYPE_LABELS[bedFea.plate_first_mode_type] ?? bedFea.plate_first_mode_type;
  const isHighRisk =
    bedFea.first_layer_risk === "high" || bedFea.first_layer_risk === "critical";

  return (
    <div className={`bed-fem-strip risk-${bedFea.first_layer_risk}`}>
      <div className="bed-fem-head">
        <span>2D MITC4 Bed Plate FEM</span>
        <span className="badge-pill-row" style={{ gap: 4 }}>
          <span className={`badge-pill ${riskMeta.cls}`} style={{ padding: "1px 6px", fontSize: 9.5 }}>
            {riskMeta.label} Risk
          </span>
          {bedFea.mesh_bed_leveling_required && (
            <span className="badge-pill warn" style={{ padding: "1px 6px", fontSize: 9.5 }}>
              Mesh Leveling Needed
            </span>
          )}
        </span>
      </div>

      <div className="metric-kv">
        <span>Flatness Error (Total)</span>
        <span
          className={`metric-val metric-val-with-delta ${
            isHighRisk
              ? "tag-bad"
              : bedFea.first_layer_risk === "moderate"
                ? "tag-warn"
                : "tag-ok"
          }`}
        >
          {cmpBedFea && (
            <InlineDeltaBadge
              oldVal={cmpBedFea.total_flatness_error_um}
              newVal={bedFea.total_flatness_error_um}
              higherIsBetter={false}
              eps={0.2}
              formatDelta={(d) => `${d > 0 ? "+" : ""}${d.toFixed(1)} µm`}
            />
          )}
          <strong>{bedFea.total_flatness_error_um.toFixed(1)} µm</strong>
        </span>
      </div>

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
          <span style={{ color: "var(--muted)", fontWeight: 500, fontSize: 10.5 }}>
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
            style={{ padding: "3px 8px", fontSize: 10 }}
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
  hotend_nozzle: "hotend mount & nozzle stem bending",
};

const ToolheadFlexBreakdownStrip: React.FC<{
  breakdown: ToolheadFlexBreakdownData;
  toolheadPunchIn?: boolean;
  onToggleToolheadPunchIn?: () => void;
}> = ({ breakdown, toolheadPunchIn, onToggleToolheadPunchIn }) => {
  const railPct = Math.max(0, breakdown.rail_and_xbeam_pct);
  const bodyPct = Math.max(0, breakdown.carriage_and_body_pct);
  const hotendPct = Math.max(0, breakdown.hotend_and_nozzle_pct);
  const domSource = breakdown.dominant_flex_source;
  const domPct =
    domSource === "hotend_nozzle"
      ? hotendPct
      : domSource === "carriage_body"
        ? bodyPct
        : railPct;

  return (
    <div
      className="sens-breakdown-card"
      style={{
        marginTop: 6,
        borderColor: "rgba(56, 189, 248, 0.35)",
      }}
    >
      <div className="sens-breakdown-head">
        <span>Toolhead Flex Breakdown</span>
        {onToggleToolheadPunchIn && (
          <button
            type="button"
            className={`deep-verify-cta ${toolheadPunchIn ? "active" : ""}`}
            style={{ padding: "1px 7px", fontSize: 9.5 }}
            onClick={onToggleToolheadPunchIn}
            title="Zoom camera directly onto the toolhead, backplate, and hotend to inspect local flex"
          >
            {toolheadPunchIn ? "Exit Punch-In" : "Punch-In View"}
          </button>
        )}
      </div>

      <div
        className="subsystem-bar-track"
        title={`X-Beam & Rail: ${railPct.toFixed(0)}% • Carriage & Body: ${bodyPct.toFixed(
          0
        )}% • Hotend & Nozzle: ${hotendPct.toFixed(0)}%`}
      >
        {railPct > 0.5 && (
          <div
            className="subsystem-bar-seg"
            style={{ width: `${railPct}%`, background: "#38bdf8" }}
          />
        )}
        {bodyPct > 0.5 && (
          <div
            className="subsystem-bar-seg"
            style={{ width: `${bodyPct}%`, background: "#f59e0b" }}
          />
        )}
        {hotendPct > 0.5 && (
          <div
            className="subsystem-bar-seg"
            style={{ width: `${hotendPct}%`, background: "#f43f5e" }}
          />
        )}
      </div>

      <div className="subsystem-bar-legend">
        <span className={`subsystem-legend-item ${domSource === "rail_xbeam" ? "primary" : ""}`}>
          <span className="swatch-dot" style={{ background: "#38bdf8" }} />
          Rail/Beam <strong>{Math.round(railPct)}%</strong>
        </span>
        <span className={`subsystem-legend-item ${domSource === "carriage_body" ? "primary" : ""}`}>
          <span className="swatch-dot" style={{ background: "#f59e0b" }} />
          Carriage/Body <strong>{Math.round(bodyPct)}%</strong>
        </span>
        <span className={`subsystem-legend-item ${domSource === "hotend_nozzle" ? "primary" : ""}`}>
          <span className="swatch-dot" style={{ background: "#f43f5e" }} />
          Hotend/Nozzle <strong>{Math.round(hotendPct)}%</strong>
        </span>
      </div>

      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: "4px 10px",
          fontSize: 10.5,
          color: "var(--muted)",
          fontVariantNumeric: "tabular-nums",
        }}
      >
        <span>
          Tip k: <strong style={{ color: "var(--text)" }}>{Math.round(breakdown.effective_tip_stiffness_n_per_mm)} N/mm</strong>
        </span>
        <span>
          Mount: <strong style={{ color: "var(--text)" }}>{Math.round(breakdown.carriage_mount_stiffness_n_per_mm)} N/mm</strong>
        </span>
        <span>
          Hotend: <strong style={{ color: "var(--text)" }}>{Math.round(breakdown.hotend_stiffness_n_per_mm)} N/mm</strong>
        </span>
      </div>

      <div style={{ fontSize: 10.5, lineHeight: 1.35, color: "#cbd5e1" }}>
        <span className="sens-fix-prefix">Main limit:</span>{" "}
        {DOMINANT_FLEX_LABELS[domSource] ?? domSource} ({Math.round(domPct)}% of local toolhead tip
        flex).
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
  bom,
  bomLoading,
  stockLengthMm,
  onStockLengthChange,
  onDownloadBomCsv,
  showBedWarp,
  onInspectBedWarp,
  toolheadPunchIn,
  onToggleToolheadPunchIn,
  error,
}) => {
  const [rightTab, setRightTab] = useState<"simulation" | "bom">("simulation");
  const [bomCategoryFilter, setBomCategoryFilter] = useState<string>("all");

  const isTetActive = Boolean(tetFea) && (activeSolidEngine === "tet10" || !solidFea);

  // Auto-focus Simulation tab when a new topology or solid FEA result arrives
  useEffect(() => {
    if (topoResult) {
      setRightTab("simulation");
    }
  }, [topoResult]);

  useEffect(() => {
    if (tetFea || solidFea) {
      setRightTab("simulation");
    }
  }, [tetFea, solidFea]);

  useEffect(() => {
    if (deepVerifyResult) {
      setRightTab("simulation");
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

  // Determine which detail cards moved vs comparison
  const structuralChanged =
    compareSim != null &&
    (Math.abs(fea.combined_max_deflection_um - compareSim.static_fea.combined_max_deflection_um) >= 0.05 ||
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
          Math.abs(bedFea.total_flatness_error_um - cmpBedFea.total_flatness_error_um) >= 0.2 ||
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

  return (
    <aside className="sidebar right">
      <div className="tabs">
        <button
          className={`tab-btn ${rightTab === "simulation" ? "active" : ""}`}
          onClick={() => setRightTab("simulation")}
        >
          Simulation
        </button>
        <button
          className={`tab-btn ${rightTab === "bom" ? "active" : ""}`}
          onClick={() => setRightTab("bom")}
        >
          BOM &amp; Cuts {bom ? `($${Math.round(bom.total_cost_usd)})` : ""}
        </button>
      </div>

      <div className="sidebar-scroll">
        {error && (
          <div className="diag-item bad">
            <strong>Notice:</strong> {error}
          </div>
        )}

        {rightTab === "simulation" && (
          <>
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
            {viewMode === "solid_part" && isTetActive && tetFea && <TetFEACard tet={tetFea} />}
            {viewMode === "solid_part" && !isTetActive && solidFea && <SolidHexFEACard solidFea={solidFea} />}

            {/* Live Impact Card OR Welcoming Baseline Tip */}
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
                          onClick={() => onCompareModeChange("last")}
                          title="Compare against the design right before your last edit"
                        >
                          vs. Last
                        </button>
                        <button
                          type="button"
                          className={`impact-seg-btn ${compareMode === "baseline" ? "active" : ""}`}
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

                <div className="impact-grid">
                  {impactMetrics.map((m) => (
                    <div key={m.key} className={`impact-tile ${m.status}`}>
                      <div className="impact-tile-top">
                        <span className="impact-tile-label">
                          {m.status !== "same" && <span className={`impact-dot ${m.status}`} />}
                          {m.label}
                        </span>
                        <span className={`impact-badge ${m.status}`}>
                          {m.status === "same" ? "—" : m.deltaText}
                        </span>
                      </div>
                      <div className="impact-tile-val">{m.newText}</div>
                      <div className="impact-tile-sub">
                        {m.status === "same" ? "No change" : `${m.oldText} → ${m.newText}`}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="welcome-tip-card">
                <div className="welcome-tip-text">
                  Loaded <strong>{baselineLabel}</strong> — adjust any setting in{" "}
                  <strong>Design Controls</strong> on the left to see its exact impact on stiffness, ringing,
                  print speed, and cost.
                </div>
              </div>
            )}

            <div className="score-hero">
              <div>
                <div style={{ fontSize: 11, color: "var(--muted)", textTransform: "uppercase", fontWeight: 700 }}>
                  Composite Engineering Score
                </div>
                <div style={{ fontSize: 13, fontWeight: 600, marginTop: 2 }}>{sim.printer_name}</div>
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

            <div className="section-card">
              <div className="section-title">Subsystem Scorecard</div>
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
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 11.5 }}>
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

            <div className="section-card">
              <div className="section-title">Engineering Diagnostics</div>
              {sim.diagnostics.map((d, idx) => {
                const lower = d.toLowerCase();
                const cls =
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
                      : "";
                return (
                  <div key={idx} className={`diag-item ${cls}`}>
                    {d}
                  </div>
                );
              })}
            </div>

            {viewMode !== "topology" && topoResult && <TopologyOptCard topo={topoResult} />}

            <div className={`section-card ${structuralChanged ? "card-changed" : ""}`}>
              <div className="section-title">
                <span>Static 3D Frame FEA</span>
                {structuralChanged && <span className="changed-dot-label">Updated</span>}
              </div>
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
                            <span style={{ color: "var(--muted)", fontWeight: 500, fontSize: 10.5 }}>
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
                      <span style={{ color: "var(--accent)", fontWeight: 600, fontSize: 10.5 }}>
                        {" "}
                        (+{Math.round(lr.x_beam_ei_y_boost_pct)}% Y / +{Math.round(lr.x_beam_ei_z_boost_pct)}% Z)
                      </span>
                      {sim.architecture === "corexy" && (
                        <span style={{ color: "var(--muted)", fontWeight: 500, fontSize: 10.5 }}>
                          {" "}
                          · {yProf} {yMountShort[lr.y_rail_mount] ?? lr.y_rail_mount} (+
                          {Math.round(lr.y_rail_ei_x_boost_pct)}% X)
                        </span>
                      )}
                    </span>
                  </div>
                );
              })()}
              <div className="metric-kv">
                <span>Gravity Sag (Nozzle)</span>
                <span className="metric-val">{fea.gravity_sag_toolhead_um.toFixed(1)} µm</span>
              </div>
              <div className="metric-kv">
                <span>
                  Nozzle-to-Bed Flex
                  {(fea.nozzle_absolute_deflection_um != null || fea.bed_deflection_um != null) && (
                    <span className="metric-sub-inline">
                      Abs: {(fea.nozzle_absolute_deflection_um ?? fea.combined_max_deflection_um).toFixed(1)} µm · Bed:{" "}
                      {(fea.bed_deflection_um ?? 0).toFixed(1)} µm
                    </span>
                  )}
                </span>
                <span className="metric-val metric-val-with-delta">
                  {compareSim && (
                    <InlineDeltaBadge
                      oldVal={compareSim.static_fea.combined_max_deflection_um}
                      newVal={fea.combined_max_deflection_um}
                      higherIsBetter={false}
                      formatDelta={(d) => `${d > 0 ? "+" : ""}${d.toFixed(1)} µm`}
                    />
                  )}
                  {fea.combined_max_deflection_um.toFixed(1)} µm
                </span>
              </div>
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
              <div className="metric-kv" title="Nozzle-to-Bed structural loop stiffness">
                <span>Nozzle-to-Bed Stiffness X / Y</span>
                <span className="metric-val">
                  {fea.toolhead_stiffness_x_n_per_mm.toFixed(0)} / {fea.toolhead_stiffness_y_n_per_mm.toFixed(0)} N/mm
                </span>
              </div>
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
                <ToolheadFlexBreakdownStrip
                  breakdown={fea.toolhead_flex_breakdown}
                  toolheadPunchIn={toolheadPunchIn}
                  onToggleToolheadPunchIn={onToggleToolheadPunchIn}
                />
              )}
            </div>

            {ts && (
              <div className={`section-card ${thermalStructChanged ? "card-changed" : ""}`}>
                <div className="section-title">
                  <span>Thermal-Structural &amp; Precision</span>
                  <span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
                    <span className="sub-block-tag">
                      {Math.round(ts.chamber_temp_c)}°C / {Math.round(ts.bed_target_temp_c)}°C
                    </span>
                    {thermalStructChanged && <span className="changed-dot-label">Updated</span>}
                  </span>
                </div>

                <div className="metric-kv">
                  <span>Bimetallic Gantry Bow</span>
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
                      <span className="badge-pill ok" style={{ padding: "1px 6px", fontSize: 9.5 }}>
                        -{Math.round(ts.gantry_backer_reduction_pct)}% backer
                      </span>
                    )}
                  </span>
                </div>

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
                      <span className="badge-pill bad" style={{ padding: "1px 6px", fontSize: 9.5 }}>
                        Taco Buckling!
                      </span>
                    )}
                  </span>
                </div>

                <div className="metric-kv" title={`Chamber thermal rise ΔT = ${ts.delta_t_k.toFixed(1)} K`}>
                  <span>Frame Z-Drift</span>
                  <span className="metric-val">
                    {ts.frame_z_drift_um.toFixed(1)} µm
                    <span style={{ color: "var(--muted)", fontWeight: 500, fontSize: 10.5 }}>
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
                        style={{ padding: "1px 6px", fontSize: 9.5 }}
                      >
                        Tg Softening
                      </span>
                    )}
                  </span>
                </div>

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
                    showBedWarp={showBedWarp}
                    onInspectBedWarp={onInspectBedWarp}
                  />
                )}
              </div>
            )}

            {!ts && bedFea && (
              <div className={`section-card ${thermalStructChanged ? "card-changed" : ""}`}>
                <div className="section-title">
                  <span>Heated Bed Plate FEM</span>
                  {thermalStructChanged && <span className="changed-dot-label">Updated</span>}
                </div>
                <BedPlateFemStrip
                  bedFea={bedFea}
                  cmpBedFea={cmpBedFea}
                  showBedWarp={showBedWarp}
                  onInspectBedWarp={onInspectBedWarp}
                />
              </div>
            )}

            {mod.shaper_graphs && (
              <KlipperShaperPsdCard
                graphs={mod.shaper_graphs}
                frameDampingRatio={mod.frame_damping_ratio}
                extrusionFilling={sim.geometry.extrusion_filling}
                changed={modalChanged}
              />
            )}

            {mod.accel_bottleneck && (
              <AccelBottleneckCard bottleneck={mod.accel_bottleneck} changed={modalChanged} />
            )}

            {sim.weight_shaper_gains && sim.weight_shaper_gains.items.length > 0 && (
              <WeightShaperPotentialCard
                gains={sim.weight_shaper_gains}
                changed={modalChanged}
              />
            )}

            <CncControlMetrologyCard
              metrology={sim.time_domain_metrology}
              workspaceFrf={sim.workspace_frf}
              rcsa={sim.rcsa}
              sensitivity={sim.sensitivity}
              viewMode={viewMode}
              onViewModeChange={onViewModeChange}
            />

            <div className={`section-card ${modalChanged ? "card-changed" : ""}`}>
              <div className="section-title">
                <span>Dynamics, Kinematics &amp; Belts</span>
                {modalChanged && <span className="changed-dot-label">Updated</span>}
              </div>
              {(() => {
                const effDrive = kin.drive_mode || sim.spec.gantry.drive_mode || "2wd";
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
                    <div className="metric-kv">
                      <span>Fundamental Mode f₁</span>
                      <span className="metric-val metric-val-with-delta">
                        {compareSim && (
                          <InlineDeltaBadge
                            oldVal={compareSim.modal_fea.fundamental_freq_hz}
                            newVal={mod.fundamental_freq_hz}
                            higherIsBetter={true}
                            formatDelta={(d) => `${d > 0 ? "+" : ""}${d.toFixed(1)} Hz`}
                          />
                        )}
                        {mod.fundamental_freq_hz.toFixed(1)} Hz
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
                    {mod.shaper_x && (
                      <div className="metric-kv">
                        <span>X-Axis Shaper &amp; Accel</span>
                        <span className="metric-val metric-val-with-delta">
                          {compareSim?.modal_fea.shaper_x && (
                            <InlineDeltaBadge
                              oldVal={compareSim.modal_fea.shaper_x.max_accel_mm_s2}
                              newVal={mod.shaper_x.max_accel_mm_s2}
                              higherIsBetter={true}
                              eps={15}
                              formatDelta={(d) => `${d > 0 ? "+" : ""}${Math.round(d).toLocaleString()}`}
                            />
                          )}
                          {mod.shaper_x.recommended_shaper} @ {mod.shaper_x.effective_freq_hz.toFixed(1)} Hz ·{" "}
                          <strong>{Math.round(mod.shaper_x.max_accel_mm_s2).toLocaleString()}</strong> mm/s²
                        </span>
                      </div>
                    )}
                    {mod.shaper_y && (
                      <div className="metric-kv">
                        <span>Y-Axis Shaper &amp; Accel</span>
                        <span className="metric-val metric-val-with-delta">
                          {compareSim?.modal_fea.shaper_y && (
                            <InlineDeltaBadge
                              oldVal={compareSim.modal_fea.shaper_y.max_accel_mm_s2}
                              newVal={mod.shaper_y.max_accel_mm_s2}
                              higherIsBetter={true}
                              eps={15}
                              formatDelta={(d) => `${d > 0 ? "+" : ""}${Math.round(d).toLocaleString()}`}
                            />
                          )}
                          {mod.shaper_y.recommended_shaper} @ {mod.shaper_y.effective_freq_hz.toFixed(1)} Hz ·{" "}
                          <strong>{Math.round(mod.shaper_y.max_accel_mm_s2).toLocaleString()}</strong> mm/s²
                        </span>
                      </div>
                    )}
                    {(mod.shaper_x || mod.shaper_y) && (
                      <>
                        <div className="metric-kv">
                          <span>Structural Freq (X / Y)</span>
                          <span className="metric-val">
                            {mod.shaper_x ? mod.shaper_x.structural_freq_hz.toFixed(0) : "—"} /{" "}
                            {mod.shaper_y ? mod.shaper_y.structural_freq_hz.toFixed(0) : "—"} Hz
                          </span>
                        </div>
                        <div className="metric-kv">
                          <span>Belt Freq (X / Y)</span>
                          <span className="metric-val">
                            {mod.shaper_x ? mod.shaper_x.belt_freq_hz.toFixed(0) : "—"} /{" "}
                            {mod.shaper_y ? mod.shaper_y.belt_freq_hz.toFixed(0) : "—"} Hz
                          </span>
                        </div>
                        {(mod.shaper_x?.motor_freq_hz != null || mod.shaper_y?.motor_freq_hz != null) && (
                          <div className="metric-kv">
                            <span>Motor Stiffness Freq (X / Y)</span>
                            <span className="metric-val">
                              {mod.shaper_x?.motor_freq_hz != null
                                ? mod.shaper_x.motor_freq_hz.toFixed(0)
                                : "—"}{" "}
                              /{" "}
                              {mod.shaper_y?.motor_freq_hz != null
                                ? mod.shaper_y.motor_freq_hz.toFixed(0)
                                : "—"}{" "}
                              Hz
                            </span>
                          </div>
                        )}
                        {(mod.shaper_x?.bracket_freq_hz != null || mod.shaper_y?.bracket_freq_hz != null) && (
                          <div className="metric-kv">
                            <span>Bracket Freq (X / Y)</span>
                            <span className="metric-val">
                              {mod.shaper_x?.bracket_freq_hz != null
                                ? mod.shaper_x.bracket_freq_hz.toFixed(0)
                                : "—"}{" "}
                              /{" "}
                              {mod.shaper_y?.bracket_freq_hz != null
                                ? mod.shaper_y.bracket_freq_hz.toFixed(0)
                                : "—"}{" "}
                              Hz
                            </span>
                          </div>
                        )}
                      </>
                    )}
                    <div className="metric-kv">
                      <span>Rec. Max Accel (Limiting)</span>
                      <span className="metric-val metric-val-with-delta">
                        {compareSim && (
                          <InlineDeltaBadge
                            oldVal={compareSim.modal_fea.recommended_max_accel_mm_s2}
                            newVal={mod.recommended_max_accel_mm_s2}
                            higherIsBetter={true}
                            eps={15}
                            formatDelta={(d) => `${d > 0 ? "+" : ""}${Math.round(d).toLocaleString()}`}
                          />
                        )}
                        <strong>{Math.round(mod.recommended_max_accel_mm_s2).toLocaleString()} mm/s²</strong>
                      </span>
                    </div>
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
                          {sim.geometry.gantry_brackets && (
                            <span style={{ color: "var(--muted)", fontSize: 10.5 }}>
                              {" "}
                              ({sim.geometry.gantry_brackets.short_name})
                            </span>
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
                            kin.motor_overcurrent_risk
                              ? "tag-bad"
                              : kin.motor_heat_w_each > 8.5
                                ? "tag-warn"
                                : ""
                          }`}
                        >
                          {kin.motor_heat_w_each.toFixed(1)} W/motor
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
                          <span style={{ color: "var(--muted)", fontWeight: 500, fontSize: 10.5 }}>
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
                        title="Speed band where 2mm GT2 belt-tooth meshing excites stepper resonance (VFAs)"
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
                    {(kin.motor_overcurrent_risk ||
                      kin.belt_overtension_risk ||
                      kin.shaft_bending_risk ||
                      kin.bracket_creep_risk ||
                      kin.steel_core_fatigue_risk) && (
                      <div className="warning-pill-stack">
                        {kin.steel_core_fatigue_risk && (
                          <div className="warning-pill warn">
                            ⚠ Steel-Core Fatigue on ≤20T Pulley
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
                        {kin.belt_overtension_risk && (
                          <div className="warning-pill bad">
                            ⚠ Belt overtensioned — exceeds working tension rating
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
                      <div className="modal-summary-block">
                        <div className="modal-summary-head">
                          <span>Dominant Vibration Modes &amp; Limits</span>
                          <button
                            type="button"
                            className={`deep-verify-cta ${viewMode === "modal" ? "active" : ""}`}
                            style={{ padding: "1px 7px", fontSize: 9.5 }}
                            onClick={() => onViewModeChange("modal")}
                            title="Animate 3D structural vibration modes in the viewport"
                          >
                            {viewMode === "modal" ? "Viewing Modal" : "View Modal 3D"}
                          </button>
                        </div>
                        <div className="modal-summary-list">
                          {(() => {
                            const baseModes = mod.modes.slice(0, 3);
                            const worstNum = mod.worst_print_mode_number;
                            const worstExtra =
                              worstNum != null && !baseModes.some((x) => x.mode_number === worstNum)
                                ? mod.modes.find((x) => x.mode_number === worstNum)
                                : undefined;
                            const shownModes = worstExtra ? [...baseModes, worstExtra] : baseModes;
                            return shownModes.map((m) => {
                              const axisTag = m.excited_axis
                                ? `${m.excited_axis}-Accel`
                                : m.dominant_axis;
                              const sevMeta = modeSeverityMeta(m.severity);
                              const isWorst = worstNum === m.mode_number;
                              const hasImpact = m.print_impact_score != null;
                              const impCls = impactLevelCls(
                                m.print_impact_level,
                                m.print_impact_score
                              );
                              const share = m.subsystem_share_pct;
                              const shareParts = share
                                ? SUBSYSTEM_ORDER.map((k) => ({
                                    key: k,
                                    label: SUBSYSTEM_DISPLAY[k].short,
                                    color: SUBSYSTEM_DISPLAY[k].color,
                                    pct: Math.round(share[k] ?? 0),
                                  })).filter((p) => p.pct >= 3)
                                : [];

                              return (
                                <div
                                  key={m.mode_number}
                                  className={`modal-mode-item ${m.severity ?? ""} ${
                                    isWorst ? "worst-mode" : ""
                                  }`}
                                >
                                  <div className="modal-mode-top">
                                    <span className="modal-mode-title">
                                      {isWorst && (
                                        <span className="worst-star-inline" title="Worst print-impact mode">
                                          ★{" "}
                                        </span>
                                      )}
                                      <strong>Mode {m.mode_number}</strong>{" "}
                                      <span className="modal-mode-freq">
                                        ({m.frequency_hz.toFixed(1)} Hz · {axisTag})
                                      </span>
                                      {m.mode_label && (
                                        <span className="modal-mode-label"> — {m.mode_label}</span>
                                      )}
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
                                    isWorst) && (
                                    <div className="modal-mode-meta-row">
                                      {isWorst && (
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
                                                <span
                                                  className="swatch-dot"
                                                  style={{ background: sp.color }}
                                                />
                                                {sp.label} {sp.pct}%
                                              </span>
                                            </React.Fragment>
                                          ))}
                                        </span>
                                      )}
                                    </div>
                                  )}

                                  {m.how_to_fix ? (
                                    <div className="modal-mode-fix">
                                      <span className="sens-fix-prefix">Main limit:</span> {m.how_to_fix}
                                    </div>
                                  ) : m.what_is_shaking ? (
                                    <div className="modal-mode-why">{m.what_is_shaking}</div>
                                  ) : null}
                                </div>
                              );
                            });
                          })()}
                        </div>
                      </div>
                    )}
                  </>
                );
              })()}
            </div>

            {viewMode !== "solid_part" && isTetActive && tetFea && <TetFEACard tet={tetFea} />}
            {viewMode !== "solid_part" && !isTetActive && solidFea && <SolidHexFEACard solidFea={solidFea} />}

            <div className={`section-card ${thermalChanged ? "card-changed" : ""}`}>
              <div className="section-title">
                <span>Hotend FVM Thermal &amp; Cooling ({th.filament_id.toUpperCase()})</span>
                {thermalChanged && <span className="changed-dot-label">Updated</span>}
              </div>
              <div className="metric-kv">
                <span>Max Volumetric Flow</span>
                <span className="metric-val metric-val-with-delta">
                  {compareSim && (
                    <InlineDeltaBadge
                      oldVal={compareSim.thermal.max_volumetric_flow_mm3_s}
                      newVal={th.max_volumetric_flow_mm3_s}
                      higherIsBetter={true}
                      formatDelta={(d) => `${d > 0 ? "+" : ""}${d.toFixed(1)}`}
                    />
                  )}
                  {th.max_volumetric_flow_mm3_s.toFixed(1)} mm³/s
                </span>
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
            </div>

            <div className={`section-card ${printChanged ? "card-changed" : ""}`}>
              <div className="section-title">
                <span>Virtual G-code Print</span>
                {printChanged && <span className="changed-dot-label">Updated</span>}
              </div>
              <div className="metric-kv">
                <span>Simulated Print Time</span>
                <span className="metric-val metric-val-with-delta">
                  {compareSim && (
                    <InlineDeltaBadge
                      oldVal={compareSim.virtual_print.total_time_s}
                      newVal={vp.total_time_s}
                      higherIsBetter={false}
                      formatDelta={(d) => `${d > 0 ? "+" : ""}${d.toFixed(1)} s`}
                    />
                  )}
                  {vp.total_time_s.toFixed(1)} s
                </span>
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
              <div className="metric-kv">
                <span>Defect Segments</span>
                <span className="metric-val">
                  {vp.underextrusion_segments} flow • {vp.ringing_segments} ring • {vp.step_skip_segments} skip
                </span>
              </div>
            </div>
          </>
        )}

        {rightTab === "bom" && (
          <>
            <div className="section-card">
              <div className="section-title">
                <span>Stock Bar &amp; Export</span>
                {bomLoading && <span style={{ color: "var(--muted)", fontSize: 10 }}>Updating...</span>}
              </div>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8 }}>
                <div className="chip-group">
                  {[1000, 2000].map((len) => (
                    <button
                      key={len}
                      className={`chip-btn ${stockLengthMm === len ? "active" : ""}`}
                      onClick={() => onStockLengthChange(len)}
                    >
                      {len}mm Stock
                    </button>
                  ))}
                </div>
                <button
                  className="btn primary"
                  style={{ padding: "5px 10px", fontSize: 11.5 }}
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
                    <span className="kpi-value" style={{ fontSize: 13.5 }}>
                      {bom.total_extrusion_length_m.toFixed(2)} m
                    </span>
                  </div>
                  <div className="kpi-box">
                    <span className="kpi-label">Belt</span>
                    <span className="kpi-value" style={{ fontSize: 13.5 }}>
                      {bom.total_belt_length_m.toFixed(2)} m
                    </span>
                  </div>
                  <div className="kpi-box">
                    <span className="kpi-label">Fasteners</span>
                    <span className="kpi-value" style={{ fontSize: 13.5 }}>
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
                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11.5 }}>
                          <span style={{ color: "var(--muted)", textTransform: "capitalize" }}>{cat}</span>
                          <span style={{ fontVariantNumeric: "tabular-nums" }}>
                            <strong>${cost.toFixed(2)}</strong>{" "}
                            <span style={{ color: "var(--muted)", fontSize: 10.5 }}>({pct.toFixed(0)}%)</span>
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
                    <span style={{ color: "var(--muted)", fontSize: 10 }}>{bom.cut_list.length} cuts</span>
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
                              <div className="bom-item-name">{c.profile_id}</div>
                              <div className="bom-item-sub">{c.name}</div>
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
                      <span style={{ color: "var(--muted)", fontSize: 10 }}>{stockLengthMm}mm bars</span>
                    </div>
                    {bom.stock_nesting.map((sn) => (
                      <div className="nesting-group" key={`${sn.profile_id}-${sn.material_id}`}>
                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11 }}>
                          <strong>{sn.profile_id}</strong>
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
                    <span style={{ color: "var(--muted)", fontSize: 10 }}>{filteredBomItems.length} items</span>
                  </div>
                  <div className="chip-group">
                    <button
                      className={`chip-btn ${bomCategoryFilter === "all" ? "active" : ""}`}
                      onClick={() => setBomCategoryFilter("all")}
                    >
                      All
                    </button>
                    {bomCategories.map((cat) => (
                      <button
                        key={cat}
                        className={`chip-btn ${bomCategoryFilter === cat ? "active" : ""}`}
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
