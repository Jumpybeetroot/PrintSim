import React, { useEffect, useRef, useState } from "react";
import { onTabListKeyDown } from "../utils/a11y";
import { DesignAnchoringSection, DesignFrameSection } from "./DesignFrameSection";
import { DesignBedZSection } from "./DesignBedZSection";
import { DesignGantrySection, FALLBACK_GANTRY_BRACKETS } from "./DesignGantrySection";
import { DesignToolheadSection } from "./DesignToolheadSection";
import { DesignThermalSection } from "./DesignThermalSection";
import {
  loadAdvancedModePreference,
  loadShowHintsPreference,
  saveAdvancedModePreference,
  saveShowHintsPreference,
} from "../utils/studioStorage";
import {
  clearAllModelOverrides,
  countActiveModelOverrides,
} from "./ModelOverrideControl";
import { ModelConfidenceModal } from "./ModelConfidenceModal";
import {
  CatalogResponse,
  GantryBracketInfo,
  PrinterSpecData,
  SimulationResponse,
  SolidFeaEngine,
  TopoOptRequestParams,
} from "../types";

/** Top-level sections of the Printer tab, in order, for the jump bar. */
const DESIGN_SECTIONS = [
  { id: "frame", label: "Frame" },
  { id: "gantry", label: "Gantry" },
  { id: "toolhead", label: "Toolhead" },
  { id: "hotend", label: "Hotend" },
  { id: "bed", label: "Bed" },
  { id: "sim", label: "Sim" },
] as const;

interface Props {
  catalog: CatalogResponse | null;
  sim: SimulationResponse | null;
  selectedPreset: string;
  onSelectPreset: (id: string) => void;
  filamentId: string;
  onFilamentChange: (id: string) => void;
  spec: PrinterSpecData | null;
  onSpecChange: (next: PrinterSpecData, changeLabel?: string) => void;
  savedBuildsCount?: number;
  lastAutoSavedAt?: string | null;
  onOpenSavedBuilds?: () => void;
  toolheadRelPos: [number, number, number];
  onToolheadRelPosChange: (pos: [number, number, number], changeLabel?: string) => void;
  panelMeshN: number;
  onPanelMeshNChange: (n: number) => void;
  yamlText: string;
  onYamlChange: (txt: string) => void;
  onSimulateYaml: () => void;
  gcodeText: string;
  onGcodeChange: (txt: string) => void;
  /** Runs the virtual print with the current G-code box contents. */
  onRunGcode?: () => void;
  onRunPartFea: (params: {
    preset: string;
    materialId: string;
    forceN: [number, number, number];
    fdmAnisotropic: boolean;
    engine: SolidFeaEngine;
  }) => void;
  onRunStlFea: (params: {
    file: File;
    materialId: string;
    fixedFace: string;
    loadFace: string;
    forceN: [number, number, number];
    fdmAnisotropic: boolean;
  }) => void;
  onDownloadPartStep: (preset: string) => void;
  exportingStep: boolean;
  onRunTopoOpt: (params: TopoOptRequestParams) => void;
  onDownloadTopoStl: (params: TopoOptRequestParams) => void;
  hasTopoResult: boolean;
  topoLoading: boolean;
  loading: boolean;
}

const PART_FEA_PRINTED_IDS = new Set([
  "abs_printed",
  "pet_cf_printed",
  "pa12_cf_printed",
  "ppa_cf_printed",
  "pps_cf_printed",
]);

const PART_FEA_METAL_IDS = new Set([
  "aluminum_6061_t6",
  "aluminum_7075_t6",
  "carbon_fiber_plate",
  "steel_4140",
  "stainless_304",
  "titanium_grade5",
]);

export const DesignPanel: React.FC<Props> = ({
  catalog,
  sim,
  selectedPreset,
  onSelectPreset,
  filamentId,
  onFilamentChange,
  spec,
  onSpecChange,
  savedBuildsCount = 0,
  lastAutoSavedAt = null,
  onOpenSavedBuilds,
  toolheadRelPos,
  onToolheadRelPosChange,
  panelMeshN,
  onPanelMeshNChange,
  yamlText,
  onYamlChange,
  onSimulateYaml,
  gcodeText,
  onGcodeChange,
  onRunGcode,
  onRunPartFea,
  onRunStlFea,
  onDownloadPartStep,
  exportingStep,
  onRunTopoOpt,
  onDownloadTopoStl,
  hasTopoResult,
  topoLoading,
  loading,
}) => {
  const [tab, setTab] = useState<"params" | "part_fea" | "yaml" | "gcode">("params");

  // Section jump bar: track which Printer-tab section is at the top of the scroll area.
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const [activeSection, setActiveSection] = useState<string>(DESIGN_SECTIONS[0].id);
  // After a chip click, the clicked section stays highlighted until the panel stops scrolling
  // to it. The lock is released by a pause in scroll events, not a fixed time, so a long
  // smooth scroll cannot outlast it.
  const jumpLockedRef = useRef<boolean>(false);
  const jumpLockTimerRef = useRef<number | null>(null);
  const holdJumpLock = (ms: number) => {
    jumpLockedRef.current = true;
    if (jumpLockTimerRef.current !== null) window.clearTimeout(jumpLockTimerRef.current);
    jumpLockTimerRef.current = window.setTimeout(() => {
      jumpLockedRef.current = false;
      jumpLockTimerRef.current = null;
    }, ms);
  };
  const designReady = Boolean(catalog && spec);
  useEffect(() => {
    const el = scrollRef.current;
    if (!el || tab !== "params" || !designReady) return;
    const measure = () => {
      if (jumpLockedRef.current) {
        holdJumpLock(160);
        return;
      }
      // The last sections are short and can never reach the top; at the bottom, pick the last one.
      if (el.scrollTop + el.clientHeight >= el.scrollHeight - 2 && el.scrollTop > 0) {
        setActiveSection(DESIGN_SECTIONS[DESIGN_SECTIONS.length - 1].id);
        return;
      }
      const line = el.getBoundingClientRect().top + 64;
      let current: string = DESIGN_SECTIONS[0].id;
      for (const sec of DESIGN_SECTIONS) {
        const node = el.querySelector<HTMLElement>(`[data-design-section="${sec.id}"]`);
        if (node && node.getBoundingClientRect().top <= line) current = sec.id;
      }
      setActiveSection(current);
    };
    el.addEventListener("scroll", measure, { passive: true });
    measure();
    return () => {
      el.removeEventListener("scroll", measure);
      if (jumpLockTimerRef.current !== null) window.clearTimeout(jumpLockTimerRef.current);
      jumpLockedRef.current = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tab, designReady]);

  const jumpToSection = (id: string) => {
    const node = scrollRef.current?.querySelector<HTMLElement>(`[data-design-section="${id}"]`);
    if (!node) return;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    setActiveSection(id);
    holdJumpLock(600);
    node.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
  };

  const [showHints, setShowHints] = useState<boolean>(() => loadShowHintsPreference());
  const [advancedMode, setAdvancedMode] = useState<boolean>(() => loadAdvancedModePreference());
  const [confidenceModalOpen, setConfidenceModalOpen] = useState<boolean>(false);
  const [feaEngine, setFeaEngine] = useState<SolidFeaEngine>("tet10");
  const [partPreset, setPartPreset] = useState<string>("corner_gusset");
  const [partMat, setPartMat] = useState<string>("pa12_cf_printed");
  const [partFdm, setPartFdm] = useState<boolean>(true);
  const [fx, setFx] = useState<number>(60);
  const [fy, setFy] = useState<number>(0);
  const [fz, setFz] = useState<number>(0);
  const [fixedFace, setFixedFace] = useState<string>("min_z");
  const [loadFace, setLoadFace] = useState<string>("max_z");

  // 3D SIMP Topology Optimization state
  const [targetVolFrac, setTargetVolFrac] = useState<number>(0.45);
  const [topoIterations, setTopoIterations] = useState<number>(18);
  const [preserveBoundaries, setPreserveBoundaries] = useState<boolean>(true);

  if (!spec || !catalog) {
    return (
      <aside className="sidebar">
        <div className="sidebar-scroll">Loading catalog...</div>
      </aside>
    );
  }

  const profileLabel = (id: string) => {
    const found = catalog.profiles.find((p) => p.id === id);
    return found ? found.name.replace(/\s*\([^)]*\)\s*/g, "").trim() : id;
  };

  const materialLabel = (id: string) => {
    const found = catalog.materials.find((m) => m.id === id);
    return found ? found.name.replace(/\s*\([^)]*\)\s*/g, "").trim() : id;
  };

  const update = (mutator: (draft: PrinterSpecData) => void, changeLabel?: string) => {
    const clone: PrinterSpecData = JSON.parse(JSON.stringify(spec));
    mutator(clone);
    onSpecChange(clone, changeLabel);
  };

  const updatePositiveNum = (
    raw: string,
    mutator: (draft: PrinterSpecData, val: number) => void,
    changeLabelBuilder: (val: number) => string
  ) => {
    const val = Number(raw);
    if (!Number.isFinite(val) || val <= 0) return;
    update((d) => mutator(d, val), changeLabelBuilder(val));
  };

  const gantryBrackets: GantryBracketInfo[] =
    catalog.gantry_brackets && catalog.gantry_brackets.length > 0
      ? catalog.gantry_brackets
      : FALLBACK_GANTRY_BRACKETS;

  const partFeaPrintedMaterials = (() => {
    const fromCatalog = catalog.materials
      .filter((m) => PART_FEA_PRINTED_IDS.has(m.id))
      .map((m) => ({ id: m.id, name: m.name }));
    const existingIds = new Set(fromCatalog.map((m) => m.id));
    const extras = gantryBrackets
      .filter((b) => PART_FEA_PRINTED_IDS.has(b.id) && !existingIds.has(b.id))
      .map((b) => ({ id: b.id, name: b.name }));
    return [...fromCatalog, ...extras];
  })();

  const partFeaMetalMaterials = (() => {
    const fromCatalog = catalog.materials
      .filter((m) => PART_FEA_METAL_IDS.has(m.id))
      .map((m) => ({ id: m.id, name: m.name }));
    const existingIds = new Set(fromCatalog.map((m) => m.id));
    const extras = gantryBrackets
      .filter((b) => PART_FEA_METAL_IDS.has(b.id) && !existingIds.has(b.id))
      .map((b) => ({ id: b.id, name: b.name }));
    return [...fromCatalog, ...extras];
  })();

  const useCadSuperelements = spec.use_cad_superelements ?? true;
  const activeCadSuperelements =
    sim?.geometry?.toolhead_structure?.cad_superelements_active ?? [];
  const activeOverrideCount = countActiveModelOverrides(spec.model_overrides);

  return (
    <aside className="sidebar">
      {/* Preset & filament apply to every tab, so they sit above the tabs, once. */}
      <div className="design-header">
        <div className="design-header-row">
          <label htmlFor="design-preset-select">Printer</label>
          <select
            id="design-preset-select"
            value={selectedPreset}
            onChange={(e) => onSelectPreset(e.target.value)}
          >
            {!selectedPreset && <option value="">Custom / Saved Configuration</option>}
            {catalog.presets.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name} ({p.architecture})
              </option>
            ))}
          </select>
        </div>
        <div className="design-header-row">
          <label htmlFor="design-filament-select" title="Filament material (sets rheology and glass transition)">
            Filament
          </label>
          <select
            id="design-filament-select"
            value={filamentId}
            onChange={(e) => onFilamentChange(e.target.value)}
          >
            {catalog.filaments.map((f) => (
              <option key={f.id} value={f.id}>
                {f.name} ({f.print_temp_c}°C)
              </option>
            ))}
          </select>
        </div>
        {onOpenSavedBuilds && (
          <div className="session-save-strip">
            <span
              className="session-autosave-pill"
              title={
                lastAutoSavedAt
                  ? `Last auto-saved at ${new Date(lastAutoSavedAt).toLocaleTimeString()} — your settings persist across page refreshes`
                  : "Active settings auto-save in this browser across page refreshes"
              }
            >
              <span className="saved-autosave-dot" />
              <span>Auto-saved</span>
            </span>
            <div className="session-save-btns">
              <button
                type="button"
                className="btn saved-builds-btn session-mini-btn"
                onClick={onOpenSavedBuilds}
              >
                {savedBuildsCount > 0 ? `Save / Load (${savedBuildsCount})` : "Save / Load"}
              </button>
            </div>
          </div>
        )}
      </div>

      <div className="tabs" role="tablist" aria-label="Design inputs" onKeyDown={onTabListKeyDown}>
        <button role="tab"
          aria-selected={tab === "params"}
          tabIndex={tab === "params" ? 0 : -1}
          className={`tab-btn ${tab === "params" ? "active" : ""}`} onClick={() => setTab("params")}>
          Printer
        </button>
        <button role="tab"
          aria-selected={tab === "part_fea"}
          tabIndex={tab === "part_fea" ? 0 : -1}
          className={`tab-btn ${tab === "part_fea" ? "active" : ""}`} onClick={() => setTab("part_fea")}>
          Parts &amp; Topo
        </button>
        <button role="tab"
          aria-selected={tab === "yaml"}
          tabIndex={tab === "yaml" ? 0 : -1}
          className={`tab-btn ${tab === "yaml" ? "active" : ""}`} onClick={() => setTab("yaml")}>
          YAML
        </button>
        <button role="tab"
          aria-selected={tab === "gcode"}
          tabIndex={tab === "gcode" ? 0 : -1}
          className={`tab-btn ${tab === "gcode" ? "active" : ""}`} onClick={() => setTab("gcode")}>
          G-code
        </button>
      </div>

      <div className="sidebar-scroll" ref={scrollRef}>
        {tab === "params" && (
          <div className="studio-mode-bar">
            <div className="studio-mode-bar-btns">
              <button
                type="button"
                className={`hints-pill ${showHints ? "active" : ""}`}
                aria-pressed={showHints}
                onClick={() =>
                  setShowHints((v) => {
                    const next = !v;
                    saveShowHintsPreference(next);
                    return next;
                  })
                }
                title="Toggle plain-English explanations on printer controls"
              >
                Hints: {showHints ? "ON" : "OFF"}
              </button>
              <button
                type="button"
                className={`hints-pill ${advancedMode ? "active" : ""}${!advancedMode && activeOverrideCount > 0 ? " has-overrides" : ""}`}
                aria-pressed={advancedMode}
                onClick={() =>
                  setAdvancedMode((v) => {
                    const next = !v;
                    saveAdvancedModePreference(next);
                    return next;
                  })
                }
                title="Show underlying physics parameters and allow direct overrides on every lumped model dropdown"
              >
                Advanced Mode: {advancedMode ? "ON" : "OFF"}
                {activeOverrideCount > 0 ? ` · ${activeOverrideCount}` : ""}
              </button>
              <button
                type="button"
                className="mode-info-link"
                onClick={() => setConfidenceModalOpen(true)}
                title="Read the WIP engineering disclaimer and inspect confidence ratings (0–100%) for every active model value"
              >
                WIP · Model Trust?
              </button>
            </div>
            {advancedMode && activeOverrideCount > 0 && (
              <button
                type="button"
                className="override-action-btn"
                onClick={() =>
                  update(
                    (d) => {
                      clearAllModelOverrides(d);
                    },
                    `Reset all ${activeOverrideCount} physics model overrides`
                  )
                }
                title="Revert all manual physics model overrides back to catalog defaults"
              >
                Reset All Overrides
              </button>
            )}
            {advancedMode && (
              <label
                className={`checkbox-row cad-cms-toggle-row${useCadSuperelements ? " active" : ""}`}
                title="Use pre-baked 3D tetrahedral Craig-Bampton (CAD CMS) superelements for supported hotends and toolheads instead of classic lumped analytical models"
              >
                <input
                  id="dp-use-cad-superelements"
                  type="checkbox"
                  checked={useCadSuperelements}
                  onChange={(e) => {
                    const checked = e.target.checked;
                    update(
                      (d) => {
                        d.use_cad_superelements = checked;
                      },
                      `CAD 3D FEA Superelements: ${checked ? "Enabled" : "Disabled (Lumped)"}`
                    );
                  }}
                />
                <span>CAD 3D FEA Superelements</span>
                {useCadSuperelements && activeCadSuperelements.length > 0 && (
                  <span className="badge-pill info cad-cms-toggle-count">
                    {activeCadSuperelements.length} active
                  </span>
                )}
              </label>
            )}
          </div>
        )}

        {tab === "params" && (
          <nav className="section-jump" aria-label="Jump to section">
            {DESIGN_SECTIONS.map((sec) => (
              <button
                key={sec.id}
                type="button"
                className={`chip-btn ${activeSection === sec.id ? "active" : ""}`}
                aria-current={activeSection === sec.id ? "true" : undefined}
                onClick={() => jumpToSection(sec.id)}
              >
                {sec.label}
              </button>
            ))}
          </nav>
        )}

        {tab === "params" && (
          <>
            <DesignFrameSection
              spec={spec}
              catalog={catalog}
              sim={sim}
              showHints={showHints}
              advancedMode={advancedMode}
              update={update}
              updatePositiveNum={updatePositiveNum}
              profileLabel={profileLabel}
              materialLabel={materialLabel}
            />

            <DesignGantrySection
              spec={spec}
              catalog={catalog}
              sim={sim}
              showHints={showHints}
              advancedMode={advancedMode}
              update={update}
              updatePositiveNum={updatePositiveNum}
              profileLabel={profileLabel}
              materialLabel={materialLabel}
            />

            <DesignToolheadSection
              spec={spec}
              catalog={catalog}
              sim={sim}
              showHints={showHints}
              advancedMode={advancedMode}
              update={update}
              updatePositiveNum={updatePositiveNum}
              materialLabel={materialLabel}
            />

            <DesignThermalSection
              spec={spec}
              catalog={catalog}
              sim={sim}
              showHints={showHints}
              advancedMode={advancedMode}
              update={update}
              updatePositiveNum={updatePositiveNum}
              materialLabel={materialLabel}
            />

            <DesignBedZSection
              spec={spec}
              catalog={catalog}
              sim={sim}
              showHints={showHints}
              advancedMode={advancedMode}
              update={update}
              updatePositiveNum={updatePositiveNum}
              materialLabel={materialLabel}
            />

            <DesignAnchoringSection
              spec={spec}
              catalog={catalog}
              sim={sim}
              showHints={showHints}
              advancedMode={advancedMode}
              toolheadRelPos={toolheadRelPos}
              onToolheadRelPosChange={onToolheadRelPosChange}
              panelMeshN={panelMeshN}
              onPanelMeshNChange={onPanelMeshNChange}
              update={update}
            />
          </>
        )}

        {tab === "part_fea" && (
          <>
            <div className="section-card">
              <div className="section-title">
                <span>3D Solid Part FEA</span>
                <span style={{ color: feaEngine === "tet10" ? "var(--accent)" : "var(--muted)", fontSize: 11 }}>
                  {feaEngine === "tet10" ? "Tet10 (P2)" : "Voxel Hex8"}
                </span>
              </div>

              <div className="field-row">
                <label id="dp-solver-engine-54">Solver Engine</label>
                <div role="group" aria-labelledby="dp-solver-engine-54" className="engine-toggle">
                  <button
                    type="button"
                    className={`engine-toggle-btn ${feaEngine === "tet10" ? "active" : ""}`}
 aria-pressed={feaEngine === "tet10"}
                    onClick={() => setFeaEngine("tet10")}
                  >
                    Curved Tet10
                    <span className="engine-toggle-sub">Gmsh + scikit-fem</span>
                  </button>
                  <button
                    type="button"
                    className={`engine-toggle-btn ${feaEngine === "hex8" ? "active" : ""}`}
 aria-pressed={feaEngine === "hex8"}
                    onClick={() => setFeaEngine("hex8")}
                  >
                    Fast Voxel Hex8
                    <span className="engine-toggle-sub">Structured Grid</span>
                  </button>
                </div>
              </div>

              <div className="field-row">
                <label htmlFor="dp-parametric-bracket-mount-55">Parametric Bracket / Mount</label>
                <select id="dp-parametric-bracket-mount-55" value={partPreset} onChange={(e) => setPartPreset(e.target.value)}>
                  {catalog.part_presets.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.name}
                    </option>
                  ))}
                </select>
              </div>
              <div className="field-grid-2 wide-first">
                <div className="field-row">
                  <label htmlFor="dp-material-56">Material</label>
                  <select id="dp-material-56"
                    value={partMat}
                    onChange={(e) => {
                      setPartMat(e.target.value);
                      setPartFdm(e.target.value.includes("printed"));
                    }}
                  >
                    <optgroup label="3D-Printed Engineering Polymers">
                      {partFeaPrintedMaterials.map((m) => (
                        <option key={m.id} value={m.id}>
                          {m.name}
                        </option>
                      ))}
                    </optgroup>
                    <optgroup label="CNC Metals & Carbon Plate">
                      {partFeaMetalMaterials.map((m) => (
                        <option key={m.id} value={m.id}>
                          {m.name}
                        </option>
                      ))}
                    </optgroup>
                  </select>
                </div>
                <div className="field-row">
                  <label htmlFor="dp-fdm-z-anisotropy-57">FDM Z-Anisotropy</label>
                  <select id="dp-fdm-z-anisotropy-57" value={partFdm ? "yes" : "no"} onChange={(e) => setPartFdm(e.target.value === "yes")}>
                    <option value="yes">FDM Layer Model</option>
                    <option value="no">Isotropic (CNC)</option>
                  </select>
                </div>
              </div>
              <div className="field-grid-3">
                <div className="field-row">
                  <label htmlFor="dp-fx-n-58">Fx (N)</label>
                  <input id="dp-fx-n-58" type="number" value={fx} onChange={(e) => setFx(Number(e.target.value))} />
                </div>
                <div className="field-row">
                  <label htmlFor="dp-fy-n-59">Fy (N)</label>
                  <input id="dp-fy-n-59" type="number" value={fy} onChange={(e) => setFy(Number(e.target.value))} />
                </div>
                <div className="field-row">
                  <label htmlFor="dp-fz-n-60">Fz (N)</label>
                  <input id="dp-fz-n-60" type="number" value={fz} onChange={(e) => setFz(Number(e.target.value))} />
                </div>
              </div>
              <div className="btn-row">
                <button
                  className="btn primary"
                  disabled={loading || topoLoading}
                  onClick={() =>
                    onRunPartFea({
                      preset: partPreset,
                      materialId: partMat,
                      forceN: [fx, fy, fz],
                      fdmAnisotropic: partFdm,
                      engine: feaEngine,
                    })
                  }
                >
                  {loading ? "Solving..." : "Run Part FEA"}
                </button>
                <button
                  className="btn"
                  disabled={loading || topoLoading || exportingStep}
                  onClick={() => onDownloadPartStep(partPreset)}
                  title="Download real OpenCASCADE B-Rep STEP CAD file for this parametric bracket"
                >
                  {exportingStep ? "Exporting..." : "Download Bracket STEP"}
                </button>
              </div>

              <div style={{ borderTop: "1px solid var(--border)", paddingTop: 10, marginTop: 4 }}>
                <div className="section-title" style={{ marginBottom: 4 }}>
                  Or Upload Custom STEP / STL / OBJ
                </div>
                <div style={{ fontSize: 11, color: "var(--muted)", marginBottom: 6, lineHeight: 1.3 }}>
                  <code>.step</code>/<code>.stp</code> → Curved Tet10 (Gmsh) • <code>.stl</code>/<code>.obj</code> → Voxel Hex8
                </div>
                <div className="field-grid-2" style={{ marginBottom: 6 }}>
                  <div className="field-row">
                    <label htmlFor="dp-fixed-clamp-face-61">Fixed Clamp Face</label>
                    <select id="dp-fixed-clamp-face-61" value={fixedFace} onChange={(e) => setFixedFace(e.target.value)}>
                      <option value="min_z">Bottom (-Z)</option>
                      <option value="min_x">Left (-X)</option>
                      <option value="min_y">Front (-Y)</option>
                    </select>
                  </div>
                  <div className="field-row">
                    <label htmlFor="dp-loaded-face-62">Loaded Face</label>
                    <select id="dp-loaded-face-62" value={loadFace} onChange={(e) => setLoadFace(e.target.value)}>
                      <option value="max_z">Top (+Z)</option>
                      <option value="max_x">Right (+X)</option>
                      <option value="max_y">Rear (+Y)</option>
                    </select>
                  </div>
                </div>
                <input aria-label="Upload a custom STEP, STL or OBJ part"
                  type="file"
                  accept=".stl,.obj,.step,.stp"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) {
                      onRunStlFea({
                        file,
                        materialId: partMat,
                        fixedFace,
                        loadFace,
                        forceN: [fx, fy, fz],
                        fdmAnisotropic: partFdm,
                      });
                      e.target.value = "";
                    }
                  }}
                />
              </div>
            </div>

            <div className="section-card" style={{ borderColor: "rgba(168, 85, 247, 0.4)" }}>
              <div className="section-title">
                <span>3D Topology Optimization (SIMP)</span>
                <span style={{ color: "var(--purple)", fontSize: 11 }}>Generative</span>
              </div>
              <div style={{ fontSize: 11, color: "var(--muted)", lineHeight: 1.35 }}>
                Carves low-stress voxels from the selected preset &amp; load case above while maximizing
                stiffness-to-weight ratio.
              </div>

              <div className="slider-row">
                <label htmlFor="dp-target-volume-fraction-63">
                  <span>Target Volume Fraction</span>
                  <strong style={{ color: "var(--text)" }}>{Math.round(targetVolFrac * 100)}%</strong>
                </label>
                <input id="dp-target-volume-fraction-63"
                  type="range"
                  min={0.25}
                  max={0.8}
                  step={0.05}
                  value={targetVolFrac}
                  onChange={(e) => setTargetVolFrac(Number(e.target.value))}
                />
              </div>

              <div className="slider-row">
                <label htmlFor="dp-simp-iterations-64">
                  <span>SIMP Iterations</span>
                  <strong style={{ color: "var(--text)" }}>{topoIterations}</strong>
                </label>
                <input id="dp-simp-iterations-64"
                  type="range"
                  min={8}
                  max={30}
                  step={1}
                  value={topoIterations}
                  onChange={(e) => setTopoIterations(Number(e.target.value))}
                />
              </div>

              <label className="checkbox-row">
                <input
                  type="checkbox"
                  checked={preserveBoundaries}
                  onChange={(e) => setPreserveBoundaries(e.target.checked)}
                />
                <span>Preserve Mounting Faces (Fixed &amp; Loaded)</span>
              </label>

              <button
                className="btn primary"
                disabled={loading || topoLoading}
                onClick={() =>
                  onRunTopoOpt({
                    preset: partPreset,
                    materialId: partMat,
                    forceN: [fx, fy, fz],
                    targetVolumeFraction: targetVolFrac,
                    iterations: topoIterations,
                    preserveBoundaries,
                    fdmAnisotropic: partFdm,
                  })
                }
              >
                {topoLoading ? "Running SIMP Optimization..." : "Run Topology Opt"}
              </button>

              <button
                className="btn"
                disabled={loading || topoLoading}
                style={hasTopoResult ? { borderColor: "var(--accent)", color: "var(--accent)" } : undefined}
                onClick={() =>
                  onDownloadTopoStl({
                    preset: partPreset,
                    materialId: partMat,
                    forceN: [fx, fy, fz],
                    targetVolumeFraction: targetVolFrac,
                    iterations: topoIterations,
                    preserveBoundaries,
                    fdmAnisotropic: partFdm,
                  })
                }
              >
                Download Optimized STL
              </button>
            </div>
          </>
        )}

        {tab === "yaml" && (
          <div className="section-card">
            <div className="section-title">
              <span>Declarative Printer YAML</span>
              <button className="btn primary" style={{ padding: "4px 8px" }} disabled={loading} onClick={onSimulateYaml}>
                Apply YAML
              </button>
            </div>
            <textarea
              aria-label="Printer YAML"
              className="code-editor"
              value={yamlText}
              onChange={(e) => onYamlChange(e.target.value)}
              spellCheck={false}
            />
          </div>
        )}

        {tab === "gcode" && (
          <div className="section-card">
            <div className="section-title">
              <span>Custom G-code Input</span>
              {onRunGcode && (
                <button
                  type="button"
                  className="btn primary"
                  style={{ padding: "4px 8px" }}
                  disabled={loading}
                  onClick={onRunGcode}
                  title="Run the virtual print with this G-code and show the result in the Print view"
                >
                  {loading ? "Solving..." : "Run Print Sim"}
                </button>
              )}
            </div>
            <p style={{ margin: 0, color: "var(--muted)", fontSize: 12 }}>
              Leave blank to run the built-in high-speed calibration cube &amp; ringing torture benchmark, or paste
              any Marlin/Klipper/OrcaSlicer G-code below.
            </p>
            <textarea
              aria-label="G-code input"
              className="code-editor"
              placeholder="; Paste G-code here (G0/G1 moves)..."
              value={gcodeText}
              onChange={(e) => onGcodeChange(e.target.value)}
              spellCheck={false}
            />
          </div>
        )}
      </div>

      <ModelConfidenceModal
        open={confidenceModalOpen}
        onClose={() => setConfidenceModalOpen(false)}
        report={sim?.model_confidence ?? null}
        advancedMode={advancedMode}
        onEnableAdvancedMode={() => {
          setAdvancedMode(true);
          saveAdvancedModePreference(true);
        }}
      />
    </aside>
  );
};
