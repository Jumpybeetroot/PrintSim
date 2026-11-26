import React, { useEffect, useState } from "react";
import { PresetItem } from "../types";

export interface BaseMachineModalProps {
  open: boolean;
  presets: PresetItem[];
  selectedPreset: string;
  onSelectPreset: (presetId: string) => void;
  onClose: () => void;
}

const BASE_KIT_TABS: Array<{
  id: string;
  label: string;
  subtitle: string;
}> = [
  {
    id: "siboor_kit",
    label: "Siboor Kit",
    subtitle: "CNC 6061 gantry, double-shear pulleys, MGN12/MGN9 rails & kit Z geometry",
  },
  {
    id: "ldo_kit",
    label: "LDO Kit",
    subtitle: "LDO Rev D matched rails, 2504AC steppers, umbilical & authentic Z spacing",
  },
  {
    id: "diy_build",
    label: "DIY / Self-Sourced",
    subtitle: "Stock printed Voron, Monolith custom builds, generic CoreXY, Bedslinger & Delta",
  },
];

export const BaseMachineModal: React.FC<BaseMachineModalProps> = ({
  open,
  presets,
  selectedPreset,
  onSelectPreset,
  onClose,
}) => {
  const initialTab = (() => {
    const active = presets.find((p) => p.id === selectedPreset);
    if (active?.category && BASE_KIT_TABS.some((t) => t.id === active.category)) {
      return active.category;
    }
    return "siboor_kit";
  })();

  const [activeTab, setActiveTab] = useState<string>(initialTab);

  useEffect(() => {
    if (!open) return;
    const active = presets.find((p) => p.id === selectedPreset);
    if (active?.category && BASE_KIT_TABS.some((t) => t.id === active.category)) {
      setActiveTab(active.category);
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

  const visibleTabs = BASE_KIT_TABS.filter((tab) =>
    presets.some((p) => (p.category || "diy_build") === tab.id)
  );
  const currentTabId = visibleTabs.some((t) => t.id === activeTab)
    ? activeTab
    : visibleTabs[0]?.id ?? "diy_build";
  const currentTabMeta = visibleTabs.find((t) => t.id === currentTabId) ?? BASE_KIT_TABS[0];
  const tabPresets = presets.filter((p) => (p.category || "diy_build") === currentTabId);

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
              What Is Your Machine Based On?
            </h2>
            <p className="base-kit-subtitle">
              Pick your starting kit or DIY platform. PrintSim auto-configures the matching gantry
              CAD kit, belt routing, rail types &amp; orientation, Z drive, bed mounts, and top
              headroom — and you can upgrade any subsystem (like swapping to a Monolith LDO gantry)
              afterward.
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

        <div className="base-kit-tabs" role="tablist" aria-label="Base machine kit family">
          {visibleTabs.map((tab) => {
            const count = presets.filter((p) => (p.category || "diy_build") === tab.id).length;
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
                <span className="badge-pill">{count}</span>
              </button>
            );
          })}
        </div>

        <div className="base-kit-tab-desc">{currentTabMeta.subtitle}</div>

        <div className="base-kit-grid" role="tabpanel">
          {tabPresets.map((p) => {
            const isSelected = p.id === selectedPreset;
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
                  {isSelected && <span className="badge-pill accent">Active</span>}
                </div>
                {p.description && <p className="base-kit-card-desc">{p.description}</p>}
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
            Tip: After starting, switch <strong>Gantry Kit / Upgrade</strong> in the Gantry section
            to upgrade to Monolith × LDO CNC, Siboor CNC, or custom DIY parts with all coupled
            settings updated automatically.
          </span>
          <button type="button" className="btn" onClick={onClose}>
            Keep Current Build
          </button>
        </div>
      </div>
    </div>
  );
};
