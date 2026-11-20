import React, { useEffect, useMemo, useState } from "react";
import {
  CompareMode,
  PrinterBOMResponse,
  SimulationResponse,
  SolidFeaEngine,
  SolidFEAResponse,
  TetFEAResponse,
  TopologyOptResponse,
  ViewMode,
} from "../types";

interface Props {
  sim: SimulationResponse | null;
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
  bom: PrinterBOMResponse | null;
  bomLoading: boolean;
  stockLengthMm: number;
  onStockLengthChange: (lenMm: number) => void;
  onDownloadBomCsv: () => void;
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
      label: "Frame Flex",
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

export const MetricsPanel: React.FC<Props> = ({
  sim,
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
  bom,
  bomLoading,
  stockLengthMm,
  onStockLengthChange,
  onDownloadBomCsv,
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
      Math.abs(scores.structural - compareSim.scores.structural) >= 0.05);
  const modalChanged =
    compareSim != null &&
    (Math.abs(mod.fundamental_freq_hz - compareSim.modal_fea.fundamental_freq_hz) >= 0.05 ||
      Math.abs(mod.recommended_max_accel_mm_s2 - compareSim.modal_fea.recommended_max_accel_mm_s2) >= 15 ||
      Math.abs(scores.dynamics - compareSim.scores.dynamics) >= 0.05 ||
      Math.abs(kin.torque_safety_margin - compareSim.kinematics.torque_safety_margin) >= 0.05 ||
      (kin.drive_mode ?? "") !== (compareSim.kinematics.drive_mode ?? "") ||
      Math.abs((kin.belt_pluck_freq_150mm_hz ?? 0) - (compareSim.kinematics.belt_pluck_freq_150mm_hz ?? 0)) >= 1);
  const thermalChanged =
    compareSim != null &&
    (Math.abs(th.max_volumetric_flow_mm3_s - compareSim.thermal.max_volumetric_flow_mm3_s) >= 0.05 ||
      Math.abs(scores.thermal - compareSim.scores.thermal) >= 0.05);
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
                  Loaded <strong>{baselineLabel}</strong> — change any setting or click a{" "}
                  <strong>Quick Upgrade</strong> on the left to see its exact impact on stiffness, ringing,
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
                  d.startsWith("COLLISION") || lower.includes("overtension")
                    ? "bad"
                    : lower.includes("risk") ||
                        lower.includes("low") ||
                        lower.includes("detected") ||
                        lower.includes("bending") ||
                        lower.includes("double-shear")
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
              {sim.geometry.panels?.enabled && (
                <div className="metric-kv">
                  <span>Enclosure / Panels</span>
                  <span className="metric-val" style={{ color: "var(--accent)" }}>
                    {sim.geometry.panels.thickness_mm}mm {sim.geometry.panels.material_name} (
                    {sim.geometry.panels.mounting === "bolted_structural" ? "Bolted" : "Clip-In"})
                  </span>
                </div>
              )}
              <div className="metric-kv">
                <span>Gravity Sag (Nozzle)</span>
                <span className="metric-val">{fea.gravity_sag_toolhead_um.toFixed(1)} µm</span>
              </div>
              <div className="metric-kv">
                <span>Peak Accel Deflection</span>
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
              <div className="metric-kv">
                <span>Toolhead Stiffness X / Y</span>
                <span className="metric-val">
                  {fea.toolhead_stiffness_x_n_per_mm.toFixed(0)} / {fea.toolhead_stiffness_y_n_per_mm.toFixed(0)} N/mm
                </span>
              </div>
              <div className="metric-kv">
                <span>Max von Mises Stress</span>
                <span className="metric-val">{fea.max_von_mises_mpa.toFixed(2)} MPa</span>
              </div>
              <div className="metric-kv">
                <span>Min Safety Factor</span>
                <span className={`metric-val ${fea.min_safety_factor < 3 ? "tag-bad" : "tag-ok"}`}>
                  {fea.min_safety_factor.toFixed(1)}×
                </span>
              </div>
            </div>

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
                    {mod.shaper_x && (
                      <div className="metric-kv">
                        <span>X-Axis Shaper</span>
                        <span className="metric-val">
                          {mod.shaper_x.recommended_shaper} @ {mod.shaper_x.effective_freq_hz.toFixed(1)} Hz
                        </span>
                      </div>
                    )}
                    {mod.shaper_y && (
                      <div className="metric-kv">
                        <span>Y-Axis Shaper</span>
                        <span className="metric-val">
                          {mod.shaper_y.recommended_shaper} @ {mod.shaper_y.effective_freq_hz.toFixed(1)} Hz
                        </span>
                      </div>
                    )}
                    <div className="metric-kv">
                      <span>Rec. Max Accel</span>
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
                        {mod.recommended_max_accel_mm_s2.toFixed(0)} mm/s²
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
                    {(kin.belt_overtension_risk || kin.shaft_bending_risk) && (
                      <div className="warning-pill-stack">
                        {kin.belt_overtension_risk && (
                          <div className="warning-pill bad">
                            ⚠ Belt overtensioned — upgrade to 9mm/12mm Aramid
                          </div>
                        )}
                        {kin.shaft_bending_risk && (
                          <div className="warning-pill warn">
                            ⚠ High tension (&gt;45 N) without double-shear pulleys
                          </div>
                        )}
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
