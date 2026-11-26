import React, { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  ConfidenceTier,
  ModelConfidenceEntry,
  ModelConfidenceOptionItem,
  ModelConfidenceReport,
} from "../types";
import { prettifyUnits } from "../utils/displayText";

export interface ModelConfidenceModalProps {
  open: boolean;
  onClose: () => void;
  report: ModelConfidenceReport | null;
  advancedMode: boolean;
  onEnableAdvancedMode: () => void;
}

type SubsystemFilter = "all" | ModelConfidenceEntry["subsystem"];
type TierFilter = "all" | "high" | "calibrated" | "low" | "overridden";

const SUBSYSTEM_FILTERS: Array<{ id: SubsystemFilter; label: string }> = [
  { id: "all", label: "All" },
  { id: "frame", label: "Frame" },
  { id: "gantry", label: "Gantry" },
  { id: "toolhead", label: "Toolhead" },
  { id: "hotend", label: "Hotend" },
  { id: "bed", label: "Bed / Z" },
  { id: "solver", label: "Solver" },
];

const SUBSYSTEM_ORDER: Record<ModelConfidenceEntry["subsystem"], number> = {
  frame: 0,
  gantry: 1,
  toolhead: 2,
  hotend: 3,
  bed: 4,
  solver: 5,
};

const TIER_LABELS: Record<ConfidenceTier, string> = {
  exact: "Datasheet / Exact",
  calibrated: "Calibrated",
  estimate: "Lumped Estimate",
  heuristic: "Rough Heuristic",
};

const FALLBACK_DISCLAIMER_TITLE = "Work in Progress — Do Not Blindly Trust Every Number";
const FALLBACK_DISCLAIMER_BODY =
  "PrintSim is an active engineering work in progress. Some parameters come directly from verified datasheets and analytical mechanics (90–100%), others are calibrated empirical models (70–89%), and unmodeled hardware assemblies rely on lumped estimates or heuristics (<70%). Use relative A/B trade-offs with higher confidence than absolute single-build figures, and turn on Advanced Mode to inspect or override assumed values.";

function pctTierClass(pct: number): ConfidenceTier {
  if (pct >= 90) return "exact";
  if (pct >= 70) return "calibrated";
  if (pct >= 50) return "estimate";
  return "heuristic";
}

function overallBadgeClass(pct: number): string {
  if (pct >= 90) return "badge-pill ok";
  if (pct >= 70) return "badge-pill info";
  if (pct >= 50) return "badge-pill warn";
  return "badge-pill bad";
}

export const ModelConfidenceModal: React.FC<ModelConfidenceModalProps> = ({
  open,
  onClose,
  report,
  advancedMode,
  onEnableAdvancedMode,
}) => {
  const [subsystemFilter, setSubsystemFilter] = useState<SubsystemFilter>("all");
  const [tierFilter, setTierFilter] = useState<TierFilter>("all");
  const [sortByLowestTrust, setSortByLowestTrust] = useState<boolean>(false);
  const [expandedIds, setExpandedIds] = useState<Record<string, boolean>>({});
  const closeBtnRef = useRef<HTMLButtonElement | null>(null);
  const previousFocusRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!open) {
      setExpandedIds({});
      return;
    }
    previousFocusRef.current = document.activeElement as HTMLElement | null;
    closeBtnRef.current?.focus();

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      previousFocusRef.current?.focus?.();
    };
  }, [open, onClose]);

  const toggleBreakdown = (entryId: string) => {
    setExpandedIds((prev) => ({
      ...prev,
      [entryId]: !prev[entryId],
    }));
  };

  const entries = report?.entries ?? [];

  const overriddenCount = useMemo(
    () => entries.filter((e) => e.overridden).length,
    [entries]
  );

  const effectiveTierFilter: TierFilter =
    tierFilter === "overridden" && overriddenCount === 0 ? "all" : tierFilter;

  const filteredEntries = useMemo(() => {
    const list = entries.filter((entry) => {
      if (subsystemFilter !== "all" && entry.subsystem !== subsystemFilter) {
        return false;
      }
      if (effectiveTierFilter === "high" && entry.trust_pct < 90) return false;
      if (
        effectiveTierFilter === "calibrated" &&
        (entry.trust_pct < 70 || entry.trust_pct >= 90)
      ) {
        return false;
      }
      if (effectiveTierFilter === "low" && entry.trust_pct >= 70) return false;
      if (effectiveTierFilter === "overridden" && !entry.overridden) return false;
      return true;
    });

    if (sortByLowestTrust) {
      return [...list].sort(
        (a, b) =>
          a.trust_pct - b.trust_pct ||
          SUBSYSTEM_ORDER[a.subsystem] - SUBSYSTEM_ORDER[b.subsystem]
      );
    }
    return [...list].sort(
      (a, b) => SUBSYSTEM_ORDER[a.subsystem] - SUBSYSTEM_ORDER[b.subsystem]
    );
  }, [entries, subsystemFilter, effectiveTierFilter, sortByLowestTrust]);

  const expandableFilteredIds = useMemo(
    () =>
      filteredEntries
        .filter((e) => e.options_breakdown && e.options_breakdown.length > 1)
        .map((e) => e.id),
    [filteredEntries]
  );

  const allExpanded =
    expandableFilteredIds.length > 0 &&
    expandableFilteredIds.every((id) => Boolean(expandedIds[id]));

  const toggleAllBreakdowns = () => {
    if (allExpanded) {
      setExpandedIds({});
    } else {
      const next: Record<string, boolean> = {};
      for (const id of expandableFilteredIds) {
        next[id] = true;
      }
      setExpandedIds(next);
    }
  };

  if (!open) return null;

  const disclaimerTitle = report?.disclaimer_title || FALLBACK_DISCLAIMER_TITLE;
  const disclaimerBody = report?.disclaimer_body || FALLBACK_DISCLAIMER_BODY;
  const overallPct =
    report && Number.isFinite(report.overall_trust_pct)
      ? Math.round(report.overall_trust_pct)
      : null;

  const hasActiveFilters = subsystemFilter !== "all" || effectiveTierFilter !== "all";

  const modalContent = (
    <div
      className="confidence-backdrop"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        className="confidence-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="confidence-modal-title"
      >
        <div className="confidence-header">
          <div className="confidence-header-left">
            <div className="confidence-title-row">
              <h2 id="confidence-modal-title" className="confidence-title">
                WIP · Model Confidence &amp; Assumptions
              </h2>
              {overallPct !== null && (
                <span className={overallBadgeClass(overallPct)}>
                  Avg Confidence {overallPct}%
                </span>
              )}
              {overriddenCount > 0 && (
                <span className="badge-pill info">
                  {overriddenCount} Manual {overriddenCount === 1 ? "Override" : "Overrides"}
                </span>
              )}
            </div>
            <p className="confidence-subtitle">
              Live parameter-by-parameter audit of active solver models, empirical calibrations, and lumped estimates
            </p>
          </div>
          <button
            ref={closeBtnRef}
            type="button"
            className="pareto-close-btn"
            onClick={onClose}
            aria-label="Close model confidence modal"
            title="Close (Esc)"
          >
            ×
          </button>
        </div>

        <div className="confidence-body">
          <div className="confidence-disclaimer" role="note" aria-label="Work in progress engineering disclaimer">
            <div className="confidence-disclaimer-title">
              <span className="confidence-disclaimer-tag">WIP Notice</span>
              <strong>{disclaimerTitle}</strong>
            </div>
            <p className="confidence-disclaimer-text">{prettifyUnits(disclaimerBody)}</p>
            <p className="confidence-disclaimer-guide">
              <strong>90–100%</strong> exact datasheet &amp; textbook mechanics ·{" "}
              <strong>70–89%</strong> calibrated empirical models ·{" "}
              <strong>&lt;70%</strong> lumped estimates for unmodeled hardware — inspect or override any lumped group via{" "}
              <strong>Advanced Mode</strong>.
            </p>
          </div>

          {report && entries.length > 0 && (
            <div className="confidence-controls">
              <div className="confidence-filter-row">
                <div
                  className="chip-group"
                  role="group"
                  aria-label="Filter confidence ledger by subsystem"
                >
                  {SUBSYSTEM_FILTERS.map((item) => (
                    <button
                      key={item.id}
                      type="button"
                      className={`chip-btn ${subsystemFilter === item.id ? "active" : ""}`}
                      aria-pressed={subsystemFilter === item.id}
                      onClick={() => setSubsystemFilter(item.id)}
                    >
                      {item.label}
                    </button>
                  ))}
                </div>
              </div>

              <div className="confidence-filter-row confidence-filter-row-secondary">
                <div
                  className="chip-group"
                  role="group"
                  aria-label="Filter confidence ledger by trust tier"
                >
                  <button
                    type="button"
                    className={`chip-btn ${effectiveTierFilter === "all" ? "active" : ""}`}
                    aria-pressed={effectiveTierFilter === "all"}
                    onClick={() => setTierFilter("all")}
                  >
                    All Tiers
                  </button>
                  <button
                    type="button"
                    className={`chip-btn ${effectiveTierFilter === "high" ? "active" : ""}`}
                    aria-pressed={effectiveTierFilter === "high"}
                    onClick={() => setTierFilter("high")}
                  >
                    90–100% High
                  </button>
                  <button
                    type="button"
                    className={`chip-btn ${effectiveTierFilter === "calibrated" ? "active" : ""}`}
                    aria-pressed={effectiveTierFilter === "calibrated"}
                    onClick={() => setTierFilter("calibrated")}
                  >
                    70–89% Calibrated
                  </button>
                  <button
                    type="button"
                    className={`chip-btn ${effectiveTierFilter === "low" ? "active" : ""}`}
                    aria-pressed={effectiveTierFilter === "low"}
                    onClick={() => setTierFilter("low")}
                  >
                    &lt;70% Estimate
                  </button>
                  {overriddenCount > 0 && (
                    <button
                      type="button"
                      className={`chip-btn ${effectiveTierFilter === "overridden" ? "active" : ""}`}
                      aria-pressed={effectiveTierFilter === "overridden"}
                      onClick={() => setTierFilter("overridden")}
                    >
                      Overridden ({overriddenCount})
                    </button>
                  )}
                </div>

                <div className="confidence-sort-group">
                  <span className="confidence-count">
                    {filteredEntries.length} of {entries.length}
                  </span>
                  {expandableFilteredIds.length > 0 && (
                    <button
                      type="button"
                      className={`chip-btn ${allExpanded ? "active" : ""}`}
                      aria-pressed={allExpanded}
                      onClick={toggleAllBreakdowns}
                      title="Expand or collapse per-option confidence breakdowns for all visible catalog components"
                    >
                      {allExpanded ? "Collapse All Options" : "Expand All Options"}
                    </button>
                  )}
                  <button
                    type="button"
                    className={`chip-btn ${sortByLowestTrust ? "active" : ""}`}
                    aria-pressed={sortByLowestTrust}
                    onClick={() => setSortByLowestTrust((v) => !v)}
                    title="Toggle between subsystem order and lowest-confidence-first order"
                  >
                    {sortByLowestTrust ? "Sort: Lowest Trust" : "Sort: Subsystem"}
                  </button>
                </div>
              </div>
            </div>
          )}

          {!report ? (
            <div className="confidence-empty">
              <p className="confidence-empty-text">
                Run a simulation to inspect live parameter confidence for your current build.
              </p>
            </div>
          ) : filteredEntries.length === 0 ? (
            <div className="confidence-empty">
              <p className="confidence-empty-text">
                No model parameters match the selected filters.
              </p>
              {hasActiveFilters && (
                <button
                  type="button"
                  className="btn"
                  onClick={() => {
                    setSubsystemFilter("all");
                    setTierFilter("all");
                  }}
                >
                  Reset Filters
                </button>
              )}
            </div>
          ) : (
            <div className="confidence-list" aria-label="Active model confidence ledger">
              <div className="confidence-list-head" aria-hidden="true">
                <span>Component</span>
                <span>Active Values</span>
                <span>Trust</span>
                <span>Basis &amp; Caveat</span>
              </div>
              {filteredEntries.map((entry) => {
                const valueTier = pctTierClass(entry.trust_pct);
                const clampedPct = Math.max(0, Math.min(100, Math.round(entry.trust_pct)));
                const breakdown: ModelConfidenceOptionItem[] | null =
                  entry.options_breakdown && entry.options_breakdown.length > 1
                    ? entry.options_breakdown
                    : null;
                const isExpanded = Boolean(breakdown && expandedIds[entry.id]);

                let minPct = clampedPct;
                let maxPct = clampedPct;
                let sortedOptions: ModelConfidenceOptionItem[] = [];
                if (breakdown) {
                  const pcts = breakdown.map((opt) =>
                    Math.max(0, Math.min(100, Math.round(opt.trust_pct)))
                  );
                  minPct = Math.min(...pcts);
                  maxPct = Math.max(...pcts);
                  if (isExpanded) {
                    sortedOptions = [...breakdown].sort((a, b) => {
                      const diff = sortByLowestTrust
                        ? a.trust_pct - b.trust_pct
                        : b.trust_pct - a.trust_pct;
                      if (diff !== 0) return diff;
                      if (a.active !== b.active) return a.active ? -1 : 1;
                      return a.label.localeCompare(b.label);
                    });
                  }
                }
                const rangeText =
                  minPct === maxPct ? `${minPct}%` : `${minPct}–${maxPct}%`;
                const breakdownListId = `confidence-breakdown-${entry.id}`;

                return (
                  <div key={entry.id} className="confidence-row">
                    <div className="confidence-col-component">
                      <div className="confidence-row-pills">
                        <span className="badge-pill">{entry.subsystem_label}</span>
                        {entry.overridden && (
                          <span className="badge-pill info">Overridden</span>
                        )}
                      </div>
                      <div className="confidence-name">{entry.name}</div>
                      <div className="confidence-selection">
                        {prettifyUnits(entry.active_selection)}
                      </div>
                    </div>

                    <div className="confidence-col-values">
                      <span className="confidence-mobile-label">Active Values</span>
                      <div className="confidence-values-text">
                        {prettifyUnits(entry.active_values)}
                      </div>
                    </div>

                    <div className="confidence-col-trust">
                      <div className="confidence-trust-top">
                        <span className={`confidence-pct tier-${valueTier}`}>
                          {clampedPct}%
                        </span>
                        <span className="confidence-tier-group">
                          <span className="confidence-tier-label">
                            {TIER_LABELS[entry.tier] ?? entry.tier}
                          </span>
                          {entry.cad_superelement_active && (
                            <span
                              className="badge-pill info"
                              title={
                                entry.cad_superelement_id
                                  ? `Backed by 3D CAD Craig-Bampton FEA superelement (${entry.cad_superelement_id}${entry.cad_trust_pct != null ? ` · ${Math.round(entry.cad_trust_pct)}% trust` : ""})`
                                  : "Backed by 3D CAD Craig-Bampton continuum FEA superelement"
                              }
                            >
                              CAD CMS
                            </span>
                          )}
                        </span>
                      </div>
                      <progress
                        className={`confidence-meter tier-${entry.tier}`}
                        value={clampedPct}
                        max={100}
                        aria-label={`${entry.name} confidence ${clampedPct}%`}
                      />
                    </div>

                    <div className="confidence-col-notes">
                      <div className="confidence-note-line">
                        <span className="confidence-note-key">Basis:</span>{" "}
                        <span className="confidence-note-val">
                          {prettifyUnits(entry.basis)}
                        </span>
                      </div>
                      <div className="confidence-note-line">
                        <span className="confidence-note-key">Caveat:</span>{" "}
                        <span className="confidence-note-val confidence-note-caveat">
                          {prettifyUnits(entry.caveat)}
                        </span>
                      </div>
                      {breakdown && (
                        <button
                          type="button"
                          className={`confidence-breakdown-toggle${isExpanded ? " active" : ""}`}
                          aria-expanded={isExpanded}
                          aria-controls={breakdownListId}
                          onClick={() => toggleBreakdown(entry.id)}
                        >
                          <span className="confidence-breakdown-chevron" aria-hidden="true">
                            {isExpanded ? "▾" : "▸"}
                          </span>
                          {isExpanded
                            ? `Hide ${breakdown.length} options`
                            : `Compare all ${breakdown.length} options (${rangeText})`}
                        </button>
                      )}
                    </div>

                    {isExpanded && sortedOptions.length > 0 && (
                      <div className="confidence-breakdown-wrap">
                        <div
                          id={breakdownListId}
                          className="confidence-breakdown-list"
                          role="region"
                          aria-label={`${entry.name} option confidence comparison`}
                        >
                          {sortedOptions.map((opt) => {
                            const optPct = Math.max(
                              0,
                              Math.min(100, Math.round(opt.trust_pct))
                            );
                            const optValueTier = pctTierClass(opt.trust_pct);
                            const optMeterTier = opt.tier ?? optValueTier;
                            return (
                              <div
                                key={opt.id}
                                className={`confidence-breakdown-row${opt.active ? " is-active" : ""}`}
                              >
                                <div className="confidence-breakdown-main">
                                  <div className="confidence-breakdown-label-row">
                                    <span className="confidence-breakdown-label">
                                      {prettifyUnits(opt.label)}
                                    </span>
                                    {opt.active && (
                                      <span className="badge-pill info">Active</span>
                                    )}
                                  </div>
                                  {opt.summary_values && (
                                    <div className="confidence-breakdown-values">
                                      {prettifyUnits(opt.summary_values)}
                                    </div>
                                  )}
                                </div>

                                <div className="confidence-breakdown-trust">
                                  <div className="confidence-breakdown-trust-top">
                                    <span
                                      className={`confidence-pct confidence-breakdown-pct tier-${optValueTier}`}
                                    >
                                      {optPct}%
                                    </span>
                                    <span className="confidence-tier-label">
                                      {TIER_LABELS[optMeterTier] ?? optMeterTier}
                                    </span>
                                  </div>
                                  <progress
                                    className={`confidence-meter tier-${optMeterTier}`}
                                    value={optPct}
                                    max={100}
                                    aria-label={`${opt.label} confidence ${optPct}%`}
                                  />
                                </div>

                                <div className="confidence-breakdown-notes">
                                  {opt.basis && (
                                    <div className="confidence-note-line">
                                      <span className="confidence-note-key">Basis:</span>{" "}
                                      <span className="confidence-note-val">
                                        {prettifyUnits(opt.basis)}
                                      </span>
                                    </div>
                                  )}
                                  {opt.caveat && (
                                    <div className="confidence-note-line">
                                      <span className="confidence-note-key">Caveat:</span>{" "}
                                      <span className="confidence-note-val confidence-note-caveat">
                                        {prettifyUnits(opt.caveat)}
                                      </span>
                                    </div>
                                  )}
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>

        <div className="confidence-footer">
          <div className="confidence-legend" aria-label="Confidence tier legend">
            <span className="confidence-legend-item">
              <span className="confidence-legend-dot tier-exact" aria-hidden="true" />
              90–100% Datasheet / Exact
            </span>
            <span className="confidence-legend-item">
              <span className="confidence-legend-dot tier-calibrated" aria-hidden="true" />
              70–89% Calibrated
            </span>
            <span className="confidence-legend-item">
              <span className="confidence-legend-dot tier-estimate" aria-hidden="true" />
              50–69% Lumped Estimate
            </span>
            <span className="confidence-legend-item">
              <span className="confidence-legend-dot tier-heuristic" aria-hidden="true" />
              &lt;50% Rough Heuristic
            </span>
          </div>

          <div className="confidence-footer-actions">
            {!advancedMode && (
              <button
                type="button"
                className="btn primary"
                onClick={() => {
                  onEnableAdvancedMode();
                  onClose();
                }}
              >
                Turn On Advanced Mode to Override Values
              </button>
            )}
            <button type="button" className="btn" onClick={onClose}>
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  );

  return createPortal(modalContent, document.body);
};
