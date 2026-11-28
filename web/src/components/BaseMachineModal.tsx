import React, { useEffect, useState } from "react";
import { ModelStatus, PresetItem } from "../types";

export interface BaseMachineModalProps {
  open: boolean;
  presets: PresetItem[];
  selectedPreset: string;
  onSelectPreset: (presetId: string) => void;
  onClose: () => void;
}

const MODEL_TIER_TABS: Array<{
  id: ModelStatus;
  label: string;
  subtitle: string;
  pillClass: string;
}> = [
  {
    id: "complete",
    label: "Complete Models",
    subtitle:
      "Full CAD-backed real machines with dedicated gantry, toolhead, hotend & Z-mount superelements and PrintDB/benchmark validation.",
    pillClass: "ok",
  },
  {
    id: "wip_specific",
    label: "WIP Specific Models",
    subtitle:
      "Named community & commercial machines that are partially modeled (e.g., borrowing gantry brackets or awaiting dedicated toolhead/frame CAD).",
    pillClass: "warn",
  },
  {
    id: "generic_standin",
    label: "Generic Stand-Ins",
    subtitle:
      "Parametric & procedural archetype baselines (CoreXY, Bedslinger, Delta) not tied to a single vendor CAD kit.",
    pillClass: "",
  },
];

const BASE_KIT_BADGE_LABELS: Record<string, string> = {
  siboor: "Siboor Kit",
  ldo: "LDO Kit",
  diy: "DIY",
  generic: "Generic",
};

function resolvePresetTier(p: PresetItem): ModelStatus {
  const raw = p.model_status || p.category;
  if (raw === "complete" || raw === "wip_specific" || raw === "generic_standin") {
    return raw;
  }
  return "generic_standin";
}

export const BaseMachineModal: React.FC<BaseMachineModalProps> = ({
  open,
  presets,
  selectedPreset,
  onSelectPreset,
  onClose,
}) => {
  const initialTab = ((): ModelStatus => {
    const active = presets.find((p) => p.id === selectedPreset);
    if (active) return resolvePresetTier(active);
    return "complete";
  })();

  const [activeTab, setActiveTab] = useState<ModelStatus>(initialTab);

  useEffect(() => {
    if (!open) return;
    const active = presets.find((p) => p.id === selectedPreset);
    if (active) {
      setActiveTab(resolvePresetTier(active));
    }
  }, [open, presets, selectedPreset]);

  useEffect(() => {
    if (!open) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [open, onClose]);

  if (!open) return null;

  const visibleTabs = MODEL_TIER_TABS.filter((tab) =>
    presets.some((p) => resolvePresetTier(p) === tab.id)
  );
  const currentTabId: ModelStatus = visibleTabs.some((t) => t.id === activeTab)
    ? activeTab
    : visibleTabs[0]?.id ?? "complete";
  const currentTabMeta =
    visibleTabs.find((t) => t.id === currentTabId) ?? MODEL_TIER_TABS[0];
  const tabPresets = presets.filter((p) => resolvePresetTier(p) === currentTabId);

  return (
    <div
      className="modal-backdrop base-kit-backdrop"
      role="presentation"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        className="modal-card base-kit-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="base-kit-modal-title"
      >
        <div className="modal-header">
          <div>
            <h2 id="base-kit-modal-title" className="modal-title">
              Choose a Base Machine
            </h2>
            <p className="base-kit-subtitle">
              Browse by model fidelity: <strong>Complete Models</strong> with full CAD
              superelements, <strong>WIP Specific Models</strong> still under active CAD
              development, or <strong>Generic Stand-Ins</strong> for clean parametric baselines.
            </p>
          </div>
          <button
            type="button"
            className="btn modal-close-btn"
            aria-label="Close base machine chooser"
            onClick={onClose}
          >
            ✕
          </button>
        </div>

        <div className="base-kit-tabs" role="tablist" aria-label="Model maturity tier">
          {visibleTabs.map((tab) => {
            const count = presets.filter((p) => resolvePresetTier(p) === tab.id).length;
            const isActive = tab.id === currentTabId;
            return (
              <button
                key={tab.id}
                type="button"
                role="tab"
                aria-selected={isActive}
                className={`base-kit-tab${isActive ? " active" : ""}`}
                onClick={() => setActiveTab(tab.id)}
              >
                <span>{tab.label}</span>
                <span className={`badge-pill${tab.pillClass ? ` ${tab.pillClass}` : ""}`}>
                  {count}
                </span>
              </button>
            );
          })}
        </div>

        <div className="base-kit-tab-desc">{currentTabMeta.subtitle}</div>

        <div className="base-kit-grid" role="tabpanel">
          {tabPresets.map((p) => {
            const isSelected = p.id === selectedPreset;
            const kitBadge = p.base_kit ? BASE_KIT_BADGE_LABELS[p.base_kit] : null;
            return (
              <button
                key={p.id}
                type="button"
                className={`base-kit-card${isSelected ? " active" : ""}`}
                aria-pressed={isSelected}
                onClick={() => {
                  onSelectPreset(p.id);
                  onClose();
                }}
              >
                <div className="base-kit-card-head">
                  <span className="base-kit-card-title">{p.short_name || p.name}</span>
                  <span className="badge-pill-row">
                    {kitBadge && kitBadge !== "Generic" && (
                      <span className="badge-pill">{kitBadge}</span>
                    )}
                    {isSelected && <span className="badge-pill accent">Active</span>}
                  </span>
                </div>
                {p.description && <p className="base-kit-card-desc">{p.description}</p>}
                {p.wip_notes && (
                  <p className="base-kit-card-desc">
                    <strong>WIP:</strong> {p.wip_notes}
                  </p>
                )}
                {p.highlights && p.highlights.length > 0 ? (
                  <div className="base-kit-tags">
                    {p.highlights.map((tag) => (
                      <span key={tag} className="base-kit-tag">
                        {tag}
                      </span>
                    ))}
                  </div>
                ) : p.summary ? (
                  <div className="base-kit-summary">{p.summary}</div>
                ) : null}
              </button>
            );
          })}
        </div>

        <div className="base-kit-footer">
          <span className="base-kit-tip">
            Tip: After starting, switch <strong>Gantry Kit / Upgrade</strong> in the Gantry
            section or <strong>Toolhead Preset</strong> in the Toolhead section — each picker is
            also grouped into Complete, WIP, and Generic options.
          </span>
          <button type="button" className="btn" onClick={onClose}>
            Keep Current Build
          </button>
        </div>
      </div>
    </div>
  );
};
