import React, { useEffect, useRef, useState } from "react";
import { ARCH_LABELS, BELT_LABELS } from "../constants/printerLabels";
import { CatalogResponse, PrinterSpecData } from "../types";
import { SavedBuildEntry } from "../utils/studioStorage";

export interface SavedBuildsDrawerProps {
  open: boolean;
  onClose: () => void;
  currentSpec: PrinterSpecData | null;
  currentFilamentId: string;
  catalog: CatalogResponse | null;
  savedBuilds: SavedBuildEntry[];
  activeBuildId: string | null;
  lastAutoSavedAt: string | null;
  statusMessage: string | null;
  statusError: string | null;
  onSaveCurrent: (name: string, overwriteId?: string) => void;
  onLoadBuild: (entry: SavedBuildEntry) => void;
  onDeleteBuild: (id: string) => void;
  onExportSettingsJson: (name: string) => void;
  onImportSettingsFile: (file: File) => void;
  onResetToStock: () => void;
}

function formatSavedTime(iso: string | null): string {
  if (!iso) return "Not saved yet";
  const dt = new Date(iso);
  if (Number.isNaN(dt.getTime())) return iso;
  const diffSec = Math.round((Date.now() - dt.getTime()) / 1000);
  if (diffSec < 10) return "Just now";
  if (diffSec < 60) return `${diffSec}s ago`;
  const diffMin = Math.round(diffSec / 60);
  if (diffMin < 60) return `${diffMin}m ago`;
  const sameDay = dt.toDateString() === new Date().toDateString();
  if (sameDay) {
    return dt.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  }
  return dt.toLocaleDateString([], {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function shortProfileName(
  profileId: string,
  catalog: CatalogResponse | null
): string {
  const found = catalog?.profiles?.find((p) => p.id === profileId);
  if (!found) return profileId.replace(/^extrusion_/, "");
  return found.name.replace(/\s*\([^)]*\)\s*/g, "").trim() || found.name;
}

function shortFilamentName(
  filamentId: string,
  catalog: CatalogResponse | null
): string {
  const found = catalog?.filaments?.find((f) => f.id === filamentId);
  return found?.name || filamentId.toUpperCase();
}

export const SavedBuildsDrawer: React.FC<SavedBuildsDrawerProps> = ({
  open,
  onClose,
  currentSpec,
  currentFilamentId,
  catalog,
  savedBuilds,
  activeBuildId,
  lastAutoSavedAt,
  statusMessage,
  statusError,
  onSaveCurrent,
  onLoadBuild,
  onDeleteBuild,
  onExportSettingsJson,
  onImportSettingsFile,
  onResetToStock,
}) => {
  const [nameInput, setNameInput] = useState<string>("");
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (!open) return;
    setNameInput(currentSpec?.name || "Custom Printer Build");
  }, [open, currentSpec?.name]);

  useEffect(() => {
    if (!open) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open, onClose]);

  if (!open) return null;

  const trimmedName = nameInput.trim();
  const matchingExisting = trimmedName
    ? savedBuilds.find((b) => b.name.toLowerCase() === trimmedName.toLowerCase())
    : undefined;

  const handleSaveSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!currentSpec) return;
    onSaveCurrent(trimmedName || currentSpec.name || "Untitled Build");
  };

  return (
    <div
      className="compare-drawer-backdrop"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        className="compare-drawer saved-builds-drawer"
        role="dialog"
        aria-label="Save and Retrieve Printer Settings"
      >
        <div className="compare-drawer-header">
          <div>
            <div className="compare-drawer-title">Save &amp; Retrieve Settings</div>
            <div className="saved-autosave-sub">
              <span className="saved-autosave-dot" />
              <span>
                Auto-save active ({formatSavedTime(lastAutoSavedAt)}) — settings persist across refresh
              </span>
            </div>
          </div>
          <div className="compare-drawer-header-actions">
            <button
              type="button"
              className="btn"
              onClick={onResetToStock}
              title="Clear browser auto-save and reload stock Voron Trident 300"
            >
              Reset Stock
            </button>
            <button type="button" className="btn" onClick={onClose}>
              Close
            </button>
          </div>
        </div>

        <div className="compare-drawer-body">
          {statusMessage && (
            <div className="saved-feedback ok">✓ {statusMessage}</div>
          )}
          {statusError && (
            <div className="saved-feedback err">⚠ {statusError}</div>
          )}

          {/* Save Current Build */}
          <div className="compare-section">
            <div className="section-title">
              <span>Save Current Configuration</span>
              {currentSpec && (
                <span className="saved-current-pill">
                  {ARCH_LABELS[currentSpec.architecture] ?? currentSpec.architecture} ·{" "}
                  {shortFilamentName(currentFilamentId, catalog)}
                </span>
              )}
            </div>
            <form className="saved-form-row" onSubmit={handleSaveSubmit}>
              <input
                type="text"
                value={nameInput}
                onChange={(e) => setNameInput(e.target.value)}
                placeholder="Build name (e.g. Trident 300 AWD 9mm)"
                aria-label="Saved build name"
              />
              <button
                type="submit"
                className="btn primary"
                disabled={!currentSpec}
              >
                {matchingExisting ? "Overwrite" : "Save Build"}
              </button>
            </form>
            <div className="saved-form-hint">
              {matchingExisting
                ? `Matches existing slot "${matchingExisting.name}" — clicking Overwrite updates it.`
                : "Saves full printer hardware spec, filament, test position, mesh settings & viewport mode in this browser."}
            </div>

            <div className="saved-file-actions">
              <button
                type="button"
                className="btn"
                disabled={!currentSpec}
                onClick={() =>
                  onExportSettingsJson(trimmedName || currentSpec?.name || "printsim_build")
                }
                title="Download complete PrintSim Studio settings as a portable .json file"
              >
                Export Settings (.json)
              </button>
              <button
                type="button"
                className="btn"
                onClick={() => fileInputRef.current?.click()}
                title="Load a previously exported .json settings file or .yaml printer spec"
              >
                Import File (.json / .yaml)
              </button>
              <input aria-label="Import settings file (.json or .yaml)"
                ref={fileInputRef}
                type="file"
                accept=".json,.yaml,.yml,application/json,text/yaml"
                style={{ display: "none" }}
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) {
                    onImportSettingsFile(file);
                  }
                  e.target.value = "";
                }}
              />
            </div>
          </div>

          {/* Saved Builds Library */}
          <div className="compare-section">
            <div className="section-title">
              <span>Saved Builds ({savedBuilds.length})</span>
              <span className="saved-storage-tag">Browser Local Storage</span>
            </div>

            {savedBuilds.length === 0 ? (
              <div className="saved-empty-state">
                <div className="saved-empty-title">No named builds saved yet</div>
                <div className="saved-empty-desc">
                  Your current session is already auto-saved on every edit so refreshing the page will keep your work. Use <strong>Save Build</strong> above to store multiple named configurations you can switch between anytime.
                </div>
              </div>
            ) : (
              <div className="saved-build-list">
                {savedBuilds.map((entry) => {
                  const s = entry.settings.spec;
                  const bv = s.kinematics.build_volume_mm;
                  const isActive = entry.id === activeBuildId;
                  return (
                    <div
                      key={entry.id}
                      className={`saved-build-card ${isActive ? "active" : ""}`}
                    >
                      <div className="saved-build-top">
                        <div className="saved-build-name-wrap">
                          <span className="saved-build-name">{entry.name}</span>
                          {isActive && <span className="saved-active-badge">Loaded</span>}
                        </div>
                        <span className="saved-build-time">
                          {formatSavedTime(entry.savedAt)}
                        </span>
                      </div>

                      <div className="saved-build-chips">
                        <span>{ARCH_LABELS[s.architecture] ?? s.architecture}</span>
                        <span>
                          {bv[0]}×{bv[1]}×{bv[2]} mm
                        </span>
                        <span>{shortProfileName(s.frame.profile, catalog)}</span>
                        <span>
                          {BELT_LABELS[s.gantry.belt.type] ?? s.gantry.belt.type}
                        </span>
                        <span>
                          {shortFilamentName(entry.settings.filamentId, catalog)}
                        </span>
                      </div>

                      <div className="saved-build-actions">
                        <button
                          type="button"
                          className="btn primary saved-card-btn"
                          onClick={() => onLoadBuild(entry)}
                        >
                          Load
                        </button>
                        <button
                          type="button"
                          className="btn saved-card-btn"
                          disabled={!currentSpec}
                          onClick={() => onSaveCurrent(entry.name, entry.id)}
                          title="Overwrite this saved slot with your current studio settings"
                        >
                          Update
                        </button>
                        <button
                          type="button"
                          className="btn saved-card-btn danger"
                          onClick={() => onDeleteBuild(entry.id)}
                          title="Delete this saved build"
                        >
                          Delete
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
