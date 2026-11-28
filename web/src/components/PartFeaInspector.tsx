import React, { useMemo } from "react";
import { IntraPartField, SubsystemKey } from "../types";
import { computeSubpartStrainBreakdown } from "../viewport/toolheadParts";

export type PartFeaMetric = "stress" | "flex";
export type PartFeaClipAxis = "off" | "x" | "y" | "z";

export interface IsolatedCadPartOption {
  id: string;
  label: string;
  subsystem: SubsystemKey;
  assetId: string;
  glbUrl?: string;
  partFieldKey?: string;
  port0Mm?: [number, number, number];
  port1Mm?: [number, number, number];
  stressMpa?: number;
  localFlexUm?: number;
}

export interface PartFeaInspectorState {
  partId: string;
  metric: PartFeaMetric;
  clipAxis: PartFeaClipAxis;
  clipFrac: number;
  clipFlip: boolean;
  showPointCloud: boolean;
  flexExaggeration: number;
}

interface PartFeaInspectorProps {
  options: IsolatedCadPartOption[];
  activeOption: IsolatedCadPartOption;
  state: PartFeaInspectorState;
  field: IntraPartField | null;
  isModalMode: boolean;
  modeLabel?: string;
  boundsMm: { min: [number, number, number]; max: [number, number, number] } | null;
  onChangePart: (partId: string) => void;
  onChangeState: (patch: Partial<PartFeaInspectorState>) => void;
  onResetCamera: () => void;
  onExit: () => void;
}

function formatNum(val: number, digits = 2): string {
  if (!Number.isFinite(val)) return "—";
  if (val === 0) return "0";
  if (Math.abs(val) < 0.01) return val.toFixed(3);
  if (Math.abs(val) < 10) return val.toFixed(digits);
  return val.toFixed(1);
}

export const PartFeaInspector: React.FC<PartFeaInspectorProps> = ({
  options,
  activeOption,
  state,
  field,
  isModalMode,
  modeLabel,
  boundsMm,
  onChangePart,
  onChangeState,
  onResetCamera,
  onExit,
}) => {
  const subparts = useMemo(() => computeSubpartStrainBreakdown(field), [field]);

  const peakStressMpa = useMemo(() => {
    if (field?.max_stress_mpa != null && Number.isFinite(field.max_stress_mpa)) {
      return field.max_stress_mpa;
    }
    if (field?.stress_mpa && field.stress_mpa.length > 0) {
      let mx = 0;
      for (let i = 0; i < field.stress_mpa.length; i++) {
        if (field.stress_mpa[i] > mx) mx = field.stress_mpa[i];
      }
      return mx;
    }
    return activeOption.stressMpa ?? null;
  }, [field, activeOption.stressMpa]);

  const maxFlexUm = useMemo(() => {
    if (field?.max_flex_um != null && Number.isFinite(field.max_flex_um)) {
      return field.max_flex_um;
    }
    if (field?.flex_um && field.flex_um.length > 0) {
      let mx = 0;
      for (let i = 0; i < field.flex_um.length; i++) {
        if (field.flex_um[i] > mx) mx = field.flex_um[i];
      }
      return mx;
    }
    return activeOption.localFlexUm ?? null;
  }, [field, activeOption.localFlexUm]);

  const clipAxisIdx = state.clipAxis === "x" ? 0 : state.clipAxis === "y" ? 1 : 2;
  const clipCoordMm = useMemo(() => {
    if (!boundsMm || state.clipAxis === "off") return null;
    const lo = boundsMm.min[clipAxisIdx];
    const hi = boundsMm.max[clipAxisIdx];
    return lo + (hi - lo) * state.clipFrac;
  }, [boundsMm, state.clipAxis, state.clipFrac, clipAxisIdx]);

  const legendMaxLabel =
    state.metric === "stress"
      ? peakStressMpa != null
        ? `${formatNum(peakStressMpa)} MPa`
        : isModalMode
          ? "Peak Strain"
          : "Peak Stress"
      : maxFlexUm != null
        ? `${formatNum(maxFlexUm)} µm`
        : "Peak Flex";

  const pointCount = field?.points_mm?.length ?? 0;

  return (
    <aside
      className="part-fea-inspector"
      aria-label="Single-Object 3D FEA Inspector"
    >
      <div className="pfi-header">
        <div className="pfi-title-block">
          <span className="pfi-kicker">
            Inside-Part 3D FEA · {isModalMode ? modeLabel || "Modal" : "Static"}
          </span>
          <strong className="pfi-title">{activeOption.label}</strong>
        </div>
        <div className="pfi-header-actions">
          <button
            type="button"
            className="part-inspector-action-btn"
            onClick={onResetCamera}
            title="Re-center 3D camera on this isolated part"
          >
            Frame
          </button>
          <button
            type="button"
            className="part-inspector-action-btn active"
            onClick={onExit}
            title="Exit isolated single-part inspection and return to full printer"
          >
            Exit
          </button>
        </div>
      </div>

      {/* Part Switcher Dropdown */}
      <div className="pfi-row">
        <label htmlFor="pfi-part-select" className="pfi-label">
          CAD Part
        </label>
        <select
          id="pfi-part-select"
          className="pfi-select"
          value={activeOption.id}
          onChange={(e) => onChangePart(e.target.value)}
        >
          {options.map((opt) => (
            <option key={opt.id} value={opt.id}>
              {opt.label}
            </option>
          ))}
        </select>
      </div>

      {/* Metric Toggle: Von Mises Stress vs Elastic Flex */}
      <div className="pfi-row">
        <span id="pfi-metric-label" className="pfi-label">
          Field Metric
        </span>
        <div
          className="pfi-seg-group"
          role="group"
          aria-labelledby="pfi-metric-label"
        >
          <button
            type="button"
            className={`pfi-seg-btn ${state.metric === "stress" ? "active" : ""}`}
            aria-pressed={state.metric === "stress"}
            onClick={() => onChangeState({ metric: "stress" })}
          >
            Von Mises Stress (MPa)
          </button>
          <button
            type="button"
            className={`pfi-seg-btn ${state.metric === "flex" ? "active" : ""}`}
            aria-pressed={state.metric === "flex"}
            onClick={() => onChangeState({ metric: "flex" })}
          >
            Elastic Flex (µm)
          </button>
        </div>
      </div>

      {/* Colour Scale Legend */}
      <div className="pfi-legend">
        <span className="pfi-legend-bound">0</span>
        <div className="legend-gradient pfi-legend-bar" aria-hidden="true" />
        <span className="pfi-legend-bound">{legendMaxLabel}</span>
      </div>

      {/* Internal Cutaway / Clip Plane Controls */}
      <div className="pfi-divider" />
      <div className="pfi-row">
        <span id="pfi-clip-axis-label" className="pfi-label">
          Cutaway Plane
        </span>
        <div
          className="pfi-seg-group"
          role="group"
          aria-labelledby="pfi-clip-axis-label"
        >
          {(["off", "x", "y", "z"] as const).map((axis) => (
            <button
              key={axis}
              type="button"
              className={`pfi-seg-btn ${state.clipAxis === axis ? "active" : ""}`}
              aria-pressed={state.clipAxis === axis}
              onClick={() => onChangeState({ clipAxis: axis })}
            >
              {axis === "off" ? "Off" : axis.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      {state.clipAxis !== "off" && (
        <div className="pfi-slider-block">
          <div className="pfi-slider-head">
            <label htmlFor="pfi-clip-slider" className="pfi-label">
              Slice Position ({state.clipAxis.toUpperCase()} mm)
            </label>
            <div className="pfi-slider-right">
              <span className="pfi-val-pill">
                {clipCoordMm != null ? `${clipCoordMm.toFixed(1)} mm` : `${Math.round(state.clipFrac * 100)}%`}
              </span>
              <button
                type="button"
                className={`pfi-mini-btn ${state.clipFlip ? "active" : ""}`}
                aria-pressed={state.clipFlip}
                onClick={() => onChangeState({ clipFlip: !state.clipFlip })}
                title="Flip which half of the part is cut away"
              >
                Flip
              </button>
            </div>
          </div>
          <input
            id="pfi-clip-slider"
            type="range"
            min={0.05}
            max={0.95}
            step={0.01}
            value={state.clipFrac}
            onChange={(e) => onChangeState({ clipFrac: Number(e.target.value) })}
          />
        </div>
      )}

      {/* Internal 3D Point Cloud Toggle & Elastic Flex Exaggeration */}
      <div className="pfi-checks-row">
        <label className="checkbox-row pfi-check">
          <input
            type="checkbox"
            checked={state.showPointCloud}
            onChange={(e) => onChangeState({ showPointCloud: e.target.checked })}
          />
          <span>Internal 3D Field Cloud</span>
        </label>
      </div>

      <div className="pfi-slider-block">
        <div className="pfi-slider-head">
          <label htmlFor="pfi-flex-slider" className="pfi-label">
            Elastic Flex Preview
          </label>
          <span className="pfi-val-pill">
            {state.flexExaggeration === 0 ? "Off" : `${state.flexExaggeration.toFixed(1)}×`}
          </span>
        </div>
        <input
          id="pfi-flex-slider"
          type="range"
          min={0}
          max={10}
          step={0.5}
          value={state.flexExaggeration}
          onChange={(e) => onChangeState({ flexExaggeration: Number(e.target.value) })}
        />
      </div>

      {/* Peak Telemetry Readout */}
      <div className="pfi-divider" />
      <div className="part-inspector-grid">
        <div className="part-inspector-cell">
          <span className="part-inspector-label">Peak Stress</span>
          <span className="part-inspector-val">
            {peakStressMpa != null ? `${formatNum(peakStressMpa)} MPa` : "Normalized"}
          </span>
        </div>
        <div className="part-inspector-cell">
          <span className="part-inspector-label">Max Local Flex</span>
          <span className="part-inspector-val">
            {maxFlexUm != null ? `${formatNum(maxFlexUm)} µm` : "Normalized"}
          </span>
        </div>
      </div>

      <div className="pfi-source-note">
        {pointCount >= 4
          ? `3D Tet4 recovery (${pointCount.toLocaleString()} internal nodes · ${field?.source ?? "superelement"})`
          : "Analytical cantilever load-path preview (no Tet4 point cloud on this asset yet)"}
      </div>

      {/* Per-Subpart Strain Energy Breakdown */}
      {subparts.length > 0 && (
        <div className="pfi-subparts">
          <span className="pfi-label">Subpart Strain Energy</span>
          <div className="pfi-subpart-list">
            {subparts.slice(0, 6).map((sp) => (
              <div key={sp.index} className="pfi-subpart-row">
                <span className="pfi-subpart-name" title={sp.name}>
                  {sp.name}
                </span>
                <span className="pfi-subpart-metrics">
                  <strong>{sp.strainEnergyPct.toFixed(1)}%</strong>
                  {sp.maxStressMpa > 0 && ` · ${formatNum(sp.maxStressMpa, 1)} MPa`}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </aside>
  );
};
