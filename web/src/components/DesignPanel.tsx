import React, { useEffect, useRef, useState } from "react";
import {
  CatalogResponse,
  PrinterSpecData,
  SimulationResponse,
  SolidFeaEngine,
  TopoOptRequestParams,
} from "../types";

interface Props {
  catalog: CatalogResponse | null;
  sim: SimulationResponse | null;
  selectedPreset: string;
  onSelectPreset: (id: string) => void;
  filamentId: string;
  onFilamentChange: (id: string) => void;
  spec: PrinterSpecData | null;
  baselineSpec: PrinterSpecData | null;
  onSpecChange: (next: PrinterSpecData, changeLabel?: string) => void;
  yamlText: string;
  onYamlChange: (txt: string) => void;
  onSimulateYaml: () => void;
  gcodeText: string;
  onGcodeChange: (txt: string) => void;
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

const BRACING_LABELS: Record<PrinterSpecData["frame"]["bracing"], string> = {
  none: "None",
  corner_gussets: "Corner Gussets",
  rear_cross: "Rear Cross + Top",
  full_truss: "Full 3D Truss",
};

const ARCH_LABELS: Record<PrinterSpecData["architecture"], string> = {
  corexy: "CoreXY",
  cartesian_bedslinger: "Cartesian Bedslinger",
  delta: "Delta",
};

const BELT_LABELS: Record<PrinterSpecData["gantry"]["belt"]["type"], string> = {
  gt2_6mm: "GT2 6mm",
  gt2_9mm: "GT2 9mm",
  gt2_12mm: "GT2 12mm",
};

const SHAPER_LABELS: Record<PrinterSpecData["kinematics"]["input_shaper"], string> = {
  auto: "Auto (Klipper)",
  none: "None",
  zv: "ZV",
  mzv: "MZV",
  ei: "EI",
  "2hump_ei": "2HUMP_EI",
};

const HEATBREAK_LABELS: Record<string, string> = {
  titanium_grade5: "Titanium Gr.5",
  stainless_304: "Stainless 304",
  brass_c360: "Brass C360",
};

const PANEL_MATERIALS: Array<{ id: string; label: string; shortLabel: string }> = [
  { id: "acrylic_pmma", label: "Cast Acrylic (PMMA)", shortLabel: "Acrylic" },
  { id: "polycarbonate", label: "Polycarbonate (PC)", shortLabel: "Polycarbonate" },
  { id: "acm_dibond", label: "ACM / Dibond", shortLabel: "ACM / Dibond" },
  { id: "aluminum_5052_sheet", label: "Aluminum 5052 Sheet", shortLabel: "Aluminum 5052" },
  { id: "carbon_fiber_plate", label: "Carbon Fiber 3K Sheet", shortLabel: "Carbon Fiber 3K" },
];

const PANEL_MOUNTING_SHORT: Record<string, string> = {
  none: "Open",
  clip_enclosure: "Clip-In",
  bolted_structural: "Bolted Structural",
};

const PANEL_THICKNESS_OPTIONS = [2, 3, 4, 5, 6];

function panelMaterialShort(id: string): string {
  return PANEL_MATERIALS.find((m) => m.id === id)?.shortLabel ?? id;
}

function formatPanelSummary(frame: PrinterSpecData["frame"]): string {
  const mounting = !frame.enclosed_panels ? "none" : frame.panel_mounting || "bolted_structural";
  if (!frame.enclosed_panels || mounting === "none") return "Open";
  const mat = panelMaterialShort(frame.panel_material || "acrylic_pmma");
  const mnt = PANEL_MOUNTING_SHORT[mounting] ?? mounting;
  return `${frame.panel_thickness_mm}mm ${mat} (${mnt})`;
}

interface QuickUpgradeItem {
  id: string;
  title: string;
  subtitle: string;
  disabled?: (s: PrinterSpecData) => boolean;
  disabledHint?: string;
  isActive: (s: PrinterSpecData) => boolean;
  apply: (d: PrinterSpecData) => void;
  revert: (d: PrinterSpecData, base: PrinterSpecData | null) => void;
}

export const DesignPanel: React.FC<Props> = ({
  catalog,
  sim,
  selectedPreset,
  onSelectPreset,
  filamentId,
  onFilamentChange,
  spec,
  baselineSpec,
  onSpecChange,
  yamlText,
  onYamlChange,
  onSimulateYaml,
  gcodeText,
  onGcodeChange,
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
  const [showHints, setShowHints] = useState<boolean>(true);
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

  // Smooth local draft for belt tension slider (avoids API spam while dragging)
  const [tensionDraft, setTensionDraft] = useState<number>(spec?.gantry.belt.pretension_n ?? 28);
  const tensionTimerRef = useRef<number | null>(null);

  useEffect(() => {
    if (spec?.gantry.belt.pretension_n != null) {
      setTensionDraft(spec.gantry.belt.pretension_n);
    }
  }, [spec?.gantry.belt.pretension_n]);

  useEffect(() => {
    return () => {
      if (tensionTimerRef.current != null) window.clearTimeout(tensionTimerRef.current);
    };
  }, []);

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

  // Resolve real catalog ID for carbon square tube (Voron 20x20 CF beam)
  const cfProfileId =
    catalog.profiles.find((p) => p.id === "cf_tube_20x20x2")?.id ||
    catalog.profiles.find((p) => p.id === "cf_square_20")?.id ||
    catalog.profiles.find((p) => p.id.startsWith("cf_"))?.id ||
    "cf_tube_20x20x2";

  const commitBeltTension = (rawVal: number) => {
    const val = Math.max(15, Math.min(120, Math.round(rawVal)));
    if (tensionTimerRef.current != null) {
      window.clearTimeout(tensionTimerRef.current);
      tensionTimerRef.current = null;
    }
    if (val === spec.gantry.belt.pretension_n) return;
    update(
      (d) => {
        d.gantry.belt.pretension_n = val;
      },
      `Belt Tension: ${spec.gantry.belt.pretension_n} N → ${val} N`
    );
  };

  const handleTensionSliderChange = (nextVal: number) => {
    setTensionDraft(nextVal);
    if (tensionTimerRef.current != null) window.clearTimeout(tensionTimerRef.current);
    tensionTimerRef.current = window.setTimeout(() => {
      commitBeltTension(nextVal);
    }, 220);
  };

  const quickUpgrades: QuickUpgradeItem[] = [
    {
      id: "awd_4motor",
      title: "4-Motor AWD",
      subtitle: "2× torque & half belt span → huge accel",
      disabled: (s) => s.architecture !== "corexy",
      disabledHint: "4-Motor AWD is only available on CoreXY frames",
      isActive: (s) => s.architecture === "corexy" && (s.gantry.drive_mode ?? "2wd") === "awd",
      apply: (d) => {
        d.gantry.drive_mode = "awd";
        d.gantry.double_shear_pulleys = true;
      },
      revert: (d, base) => {
        const useBase = base && (base.gantry.drive_mode ?? "2wd") !== "awd";
        d.gantry.drive_mode = useBase ? base.gantry.drive_mode : "2wd";
        const keepDoubleShear =
          d.gantry.belt.pretension_n >= 46 || (base ? Boolean(base.gantry.double_shear_pulleys) : false);
        d.gantry.double_shear_pulleys = keepDoubleShear;
      },
    },
    {
      id: "ht_9mm_aramid",
      title: "High-Tension 9mm Aramid",
      subtitle: "Stiff high-tension belts for crisp corners",
      isActive: (s) =>
        s.gantry.belt.type === "gt2_9mm" &&
        s.gantry.belt.core === "aramid" &&
        s.gantry.belt.pretension_n >= 60,
      apply: (d) => {
        d.gantry.belt.type = "gt2_9mm";
        d.gantry.belt.core = "aramid";
        d.gantry.belt.pretension_n = 65;
        d.gantry.double_shear_pulleys = true;
      },
      revert: (d, base) => {
        const baseWasSame =
          base &&
          base.gantry.belt.type === "gt2_9mm" &&
          base.gantry.belt.core === "aramid" &&
          base.gantry.belt.pretension_n >= 60;
        const useBase = Boolean(base && !baseWasSame);
        d.gantry.belt.type = useBase && base ? base.gantry.belt.type : "gt2_6mm";
        d.gantry.belt.core = useBase && base ? base.gantry.belt.core : "fiberglass";
        d.gantry.belt.pretension_n = useBase && base ? base.gantry.belt.pretension_n : 28;
        const keepDoubleShear =
          (d.gantry.drive_mode ?? "2wd") === "awd" || (base ? Boolean(base.gantry.double_shear_pulleys) : false);
        d.gantry.double_shear_pulleys = keepDoubleShear;
      },
    },
    {
      id: "acm_4mm_structural",
      title: "4mm ACM Structural",
      subtitle: "Bolted shear panels lock frame square",
      isActive: (s) =>
        s.frame.enclosed_panels &&
        s.frame.panel_mounting === "bolted_structural" &&
        s.frame.panel_material === "acm_dibond" &&
        s.frame.panel_thickness_mm >= 4,
      apply: (d) => {
        d.frame.enclosed_panels = true;
        d.frame.panel_mounting = "bolted_structural";
        d.frame.panel_material = "acm_dibond";
        d.frame.panel_thickness_mm = 4.0;
      },
      revert: (d, base) => {
        const baseWasSame =
          base &&
          base.frame.enclosed_panels &&
          base.frame.panel_mounting === "bolted_structural" &&
          base.frame.panel_material === "acm_dibond" &&
          base.frame.panel_thickness_mm >= 4;
        const useBase = Boolean(base && !baseWasSame);
        d.frame.enclosed_panels = useBase && base ? base.frame.enclosed_panels : false;
        d.frame.panel_mounting = useBase && base ? base.frame.panel_mounting : "none";
        d.frame.panel_material = useBase && base ? base.frame.panel_material : "acrylic_pmma";
        d.frame.panel_thickness_mm = useBase && base ? base.frame.panel_thickness_mm : 3.0;
      },
    },
    {
      id: "cf_x_beam",
      title: "Carbon X Beam",
      subtitle: "Lighter gantry → higher Y resonance",
      isActive: (s) =>
        s.gantry.x_beam_material === "carbon_fiber_tube" &&
        (s.gantry.x_beam_profile === cfProfileId || s.gantry.x_beam_profile.startsWith("cf_")),
      apply: (d) => {
        d.gantry.x_beam_profile = cfProfileId;
        d.gantry.x_beam_material = "carbon_fiber_tube";
      },
      revert: (d, base) => {
        const useBase =
          base &&
          base.gantry.x_beam_material !== "carbon_fiber_tube" &&
          !base.gantry.x_beam_profile.startsWith("cf_");
        d.gantry.x_beam_profile = useBase ? base.gantry.x_beam_profile : "extrusion_2020";
        d.gantry.x_beam_material = useBase ? base.gantry.x_beam_material : "aluminum_6063_t5";
      },
    },
    {
      id: "uhf_hotend",
      title: "UHF High-Flow",
      subtitle: "High-speed extrusion flow (mm³/s)",
      isActive: (s) =>
        s.toolhead.hotend.melt_zone_length_mm >= 26 && s.toolhead.hotend.heater_power_w >= 100,
      apply: (d) => {
        d.toolhead.hotend.melt_zone_length_mm = 26;
        d.toolhead.hotend.heater_power_w = 100;
      },
      revert: (d, base) => {
        const useBaseMelt = base && base.toolhead.hotend.melt_zone_length_mm < 26;
        const useBaseHeat = base && base.toolhead.hotend.heater_power_w < 100;
        d.toolhead.hotend.melt_zone_length_mm = useBaseMelt
          ? base.toolhead.hotend.melt_zone_length_mm
          : 16;
        d.toolhead.hotend.heater_power_w = useBaseHeat ? base.toolhead.hotend.heater_power_w : 60;
      },
    },
    {
      id: "48v_steppers",
      title: "48V Steppers",
      subtitle: "Strong torque at 500–700+ mm/s",
      isActive: (s) => s.kinematics.xy_stepper.supply_voltage_v >= 48,
      apply: (d) => {
        d.kinematics.xy_stepper.supply_voltage_v = 48;
      },
      revert: (d, base) => {
        const useBase = base && base.kinematics.xy_stepper.supply_voltage_v < 48;
        d.kinematics.xy_stepper.supply_voltage_v = useBase
          ? base.kinematics.xy_stepper.supply_voltage_v
          : 24;
      },
    },
  ];

  const handleToggleUpgrade = (upg: QuickUpgradeItem) => {
    if (upg.disabled?.(spec)) return;
    const active = upg.isActive(spec);
    if (active) {
      update((d) => upg.revert(d, baselineSpec), `Removed Upgrade: ${upg.title}`);
    } else {
      update((d) => upg.apply(d), `Added Upgrade: ${upg.title}`);
    }
  };

  const effectivePanelMounting: PrinterSpecData["frame"]["panel_mounting"] = !spec.frame.enclosed_panels
    ? "none"
    : spec.frame.panel_mounting || "bolted_structural";
  const panelsEnabled = spec.frame.enclosed_panels && effectivePanelMounting !== "none";
  const currentPanelMaterial = spec.frame.panel_material || "acrylic_pmma";
  const currentPanelThickness = spec.frame.panel_thickness_mm || 3;
  const thicknessChoices = PANEL_THICKNESS_OPTIONS.includes(currentPanelThickness)
    ? PANEL_THICKNESS_OPTIONS
    : [...PANEL_THICKNESS_OPTIONS, currentPanelThickness].sort((a, b) => a - b);

  const driveMode: "2wd" | "awd" = spec.gantry.drive_mode || "2wd";
  const doubleShearPulleys = Boolean(spec.gantry.double_shear_pulleys);
  const beltPluckHz = sim?.kinematics.belt_pluck_freq_150mm_hz;
  const beltMaxWorkingN = sim?.kinematics.belt_max_working_tension_n;
  const beltOvertension = Boolean(sim?.kinematics.belt_overtension_risk);
  const shaftBendingRisk = Boolean(sim?.kinematics.shaft_bending_risk);

  return (
    <aside className="sidebar">
      <div className="tabs">
        <button className={`tab-btn ${tab === "params" ? "active" : ""}`} onClick={() => setTab("params")}>
          Printer
        </button>
        <button className={`tab-btn ${tab === "part_fea" ? "active" : ""}`} onClick={() => setTab("part_fea")}>
          Parts &amp; Topo
        </button>
        <button className={`tab-btn ${tab === "yaml" ? "active" : ""}`} onClick={() => setTab("yaml")}>
          YAML
        </button>
        <button className={`tab-btn ${tab === "gcode" ? "active" : ""}`} onClick={() => setTab("gcode")}>
          G-code
        </button>
      </div>

      <div className="sidebar-scroll">
        {tab === "params" && (
          <div className="hints-toggle-bar">
            <span className="hints-toggle-label">What each control affects</span>
            <button
              type="button"
              className={`hints-pill ${showHints ? "active" : ""}`}
              onClick={() => setShowHints((v) => !v)}
              title="Toggle plain-English explanations on printer controls"
            >
              Plain-English Hints: {showHints ? "ON" : "OFF"}
            </button>
          </div>
        )}

        {/* Preset & Filament selectors always available */}
        <div className="section-card">
          <div className="section-title">Preset &amp; Filament</div>
          <div className="field-row">
            <label>Printer Template</label>
            <select value={selectedPreset} onChange={(e) => onSelectPreset(e.target.value)}>
              {catalog.presets.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name} ({p.architecture})
                </option>
              ))}
            </select>
          </div>
          <div className="field-row">
            <label>Filament Material (Rheology &amp; Tg)</label>
            <select value={filamentId} onChange={(e) => onFilamentChange(e.target.value)}>
              {catalog.filaments.map((f) => (
                <option key={f.id} value={f.id}>
                  {f.name} ({f.print_temp_c}°C)
                </option>
              ))}
            </select>
          </div>
        </div>

        {tab === "params" && (
          <>
            {/* 1-Click Quick Upgrades */}
            <div className="section-card upgrade-card">
              <div className="section-title">
                <span>Quick Upgrades (Try &amp; Compare)</span>
                <span style={{ color: "var(--muted)", fontSize: 10, fontWeight: 500, textTransform: "none" }}>
                  Click to toggle
                </span>
              </div>
              <div className="upgrade-grid">
                {quickUpgrades.map((upg) => {
                  const active = upg.isActive(spec);
                  const isDisabled = Boolean(upg.disabled?.(spec));
                  return (
                    <button
                      key={upg.id}
                      type="button"
                      disabled={isDisabled}
                      className={`upgrade-btn ${active ? "active" : ""}`}
                      onClick={() => handleToggleUpgrade(upg)}
                      title={
                        isDisabled
                          ? upg.disabledHint || `${upg.title} unavailable`
                          : active
                            ? `Click to revert ${upg.title} to stock`
                            : `Apply ${upg.title}`
                      }
                    >
                      <span className="upgrade-btn-head">
                        <span className="upgrade-btn-title">{upg.title}</span>
                        <span className="upgrade-check">{active ? "✓" : "+"}</span>
                      </span>
                      <span className="upgrade-btn-sub">{upg.subtitle}</span>
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="section-card">
              <div className="section-title">Frame &amp; Structure (FEA)</div>
              <div className="field-row">
                <label>Architecture</label>
                <select
                  value={spec.architecture}
                  onChange={(e) => {
                    const nextVal = e.target.value as PrinterSpecData["architecture"];
                    update(
                      (d) => {
                        d.architecture = nextVal;
                      },
                      `Architecture: ${ARCH_LABELS[spec.architecture]} → ${ARCH_LABELS[nextVal]}`
                    );
                  }}
                >
                  <option value="corexy">CoreXY</option>
                  <option value="cartesian_bedslinger">Cartesian Bedslinger</option>
                  <option value="delta">Delta (3-Tower)</option>
                </select>
              </div>

              {showHints && (
                <div className="hint-line">
                  Bigger/thicker extrusions = stiffer frame &amp; higher resonance, but heavier &amp; pricier
                </div>
              )}

              <div className="field-grid-3">
                <div className="field-row">
                  <label>Width mm</label>
                  <input
                    type="number"
                    value={spec.frame.width_mm}
                    onChange={(e) =>
                      updatePositiveNum(
                        e.target.value,
                        (d, v) => {
                          d.frame.width_mm = v;
                        },
                        (v) => `Frame Width: ${spec.frame.width_mm} → ${v} mm`
                      )
                    }
                  />
                </div>
                <div className="field-row">
                  <label>Depth mm</label>
                  <input
                    type="number"
                    value={spec.frame.depth_mm}
                    onChange={(e) =>
                      updatePositiveNum(
                        e.target.value,
                        (d, v) => {
                          d.frame.depth_mm = v;
                        },
                        (v) => `Frame Depth: ${spec.frame.depth_mm} → ${v} mm`
                      )
                    }
                  />
                </div>
                <div className="field-row">
                  <label>Height mm</label>
                  <input
                    type="number"
                    value={spec.frame.height_mm}
                    onChange={(e) =>
                      updatePositiveNum(
                        e.target.value,
                        (d, v) => {
                          d.frame.height_mm = v;
                        },
                        (v) => `Frame Height: ${spec.frame.height_mm} → ${v} mm`
                      )
                    }
                  />
                </div>
              </div>

              <div className="field-grid-2">
                <div className="field-row">
                  <label>Frame Profile</label>
                  <select
                    value={spec.frame.profile}
                    onChange={(e) => {
                      const nextVal = e.target.value;
                      update(
                        (d) => {
                          d.frame.profile = nextVal;
                        },
                        `Frame Profile: ${profileLabel(spec.frame.profile)} → ${profileLabel(nextVal)}`
                      );
                    }}
                  >
                    {catalog.profiles.map((p) => (
                      <option key={p.id} value={p.id}>
                        {p.name}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="field-row">
                  <label>Frame Material</label>
                  <select
                    value={spec.frame.material}
                    onChange={(e) => {
                      const nextVal = e.target.value;
                      update(
                        (d) => {
                          d.frame.material = nextVal;
                        },
                        `Frame Material: ${materialLabel(spec.frame.material)} → ${materialLabel(nextVal)}`
                      );
                    }}
                  >
                    {catalog.materials.map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.name}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {showHints && (
                <div className="hint-line">Cuts frame wobble &amp; ghosting dramatically for low cost</div>
              )}

              <div className="field-row">
                <label>Bracing</label>
                <select
                  value={spec.frame.bracing}
                  onChange={(e) => {
                    const nextVal = e.target.value as PrinterSpecData["frame"]["bracing"];
                    update(
                      (d) => {
                        d.frame.bracing = nextVal;
                      },
                      `Bracing: ${BRACING_LABELS[spec.frame.bracing]} → ${BRACING_LABELS[nextVal]}`
                    );
                  }}
                >
                  <option value="none">None (Open Frame)</option>
                  <option value="corner_gussets">Corner Gussets</option>
                  <option value="rear_cross">Rear Cross + Top</option>
                  <option value="full_truss">Full 3D Truss</option>
                </select>
              </div>

              <div className="sub-block">
                <div className="sub-block-title">
                  <span>Structural &amp; Enclosure Panels</span>
                  {panelsEnabled && (
                    <span className="sub-block-tag">
                      {currentPanelThickness}mm {panelMaterialShort(currentPanelMaterial)}
                    </span>
                  )}
                </div>

                {showHints && (
                  <div className="hint-line">
                    Bolted ACM, aluminum, or carbon panels act as shear walls that lock the frame square;
                    thicker/stiffer sheets boost resonance &amp; cut flex.
                  </div>
                )}

                <div className="field-row">
                  <label>Panel Mounting</label>
                  <select
                    value={effectivePanelMounting}
                    onChange={(e) => {
                      const nextMount = e.target.value as PrinterSpecData["frame"]["panel_mounting"];
                      const prevDesc = formatPanelSummary(spec.frame);
                      const nextFrame: PrinterSpecData["frame"] = {
                        ...spec.frame,
                        enclosed_panels: nextMount !== "none",
                        panel_mounting: nextMount,
                        panel_material: spec.frame.panel_material || "acrylic_pmma",
                        panel_thickness_mm: spec.frame.panel_thickness_mm || 3,
                      };
                      const nextDesc = formatPanelSummary(nextFrame);
                      update(
                        (d) => {
                          d.frame.enclosed_panels = nextMount !== "none";
                          d.frame.panel_mounting = nextMount;
                          if (!d.frame.panel_material) d.frame.panel_material = "acrylic_pmma";
                          if (!d.frame.panel_thickness_mm) d.frame.panel_thickness_mm = 3;
                        },
                        `Panels: ${prevDesc} → ${nextDesc}`
                      );
                    }}
                  >
                    <option value="none">Open (No Panels)</option>
                    <option value="clip_enclosure">Clip-In Enclosure</option>
                    <option value="bolted_structural">Bolted Structural Panels</option>
                  </select>
                </div>

                {panelsEnabled && (
                  <div className="field-grid-2">
                    <div className="field-row">
                      <label>Panel Material</label>
                      <select
                        value={currentPanelMaterial}
                        onChange={(e) => {
                          const nextMat = e.target.value;
                          const prevDesc = formatPanelSummary(spec.frame);
                          const nextDesc = formatPanelSummary({
                            ...spec.frame,
                            panel_material: nextMat,
                          });
                          update(
                            (d) => {
                              d.frame.panel_material = nextMat;
                            },
                            `Panels: ${prevDesc} → ${nextDesc}`
                          );
                        }}
                      >
                        {PANEL_MATERIALS.map((pm) => (
                          <option key={pm.id} value={pm.id}>
                            {pm.label}
                          </option>
                        ))}
                      </select>
                    </div>
                    <div className="field-row">
                      <label>Thickness (mm)</label>
                      <select
                        value={currentPanelThickness}
                        onChange={(e) => {
                          const nextThick = Number(e.target.value);
                          if (!Number.isFinite(nextThick) || nextThick <= 0) return;
                          const prevDesc = formatPanelSummary(spec.frame);
                          const nextDesc = formatPanelSummary({
                            ...spec.frame,
                            panel_thickness_mm: nextThick,
                          });
                          update(
                            (d) => {
                              d.frame.panel_thickness_mm = nextThick;
                            },
                            `Panels: ${prevDesc} → ${nextDesc}`
                          );
                        }}
                      >
                        {thicknessChoices.map((mm) => (
                          <option key={mm} value={mm}>
                            {mm} mm
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>
                )}
              </div>
            </div>

            <div className="section-card">
              <div className="section-title">Gantry, Belts &amp; Kinematics</div>

              {spec.architecture === "corexy" && (
                <div className="sub-block">
                  <div className="sub-block-title">
                    <span>XY Drive System</span>
                    <span className="sub-block-tag">
                      {driveMode === "awd" ? "4 Motors (AWD)" : "2 Motors (2WD)"}
                    </span>
                  </div>

                  {showHints && (
                    <div className="hint-line">
                      AWD adds 2 front XY motors for 2× torque &amp; half belt span; double-shear bearings
                      prevent pulley shaft bending under high tension.
                    </div>
                  )}

                  <div className="field-row">
                    <label>Drive Mode</label>
                    <div className="engine-toggle">
                      <button
                        type="button"
                        className={`engine-toggle-btn ${driveMode === "2wd" ? "active" : ""}`}
                        onClick={() => {
                          if (driveMode === "2wd") return;
                          update(
                            (d) => {
                              d.gantry.drive_mode = "2wd";
                            },
                            "Drive Mode: AWD (4-Motor) → 2WD (2-Motor)"
                          );
                        }}
                      >
                        2WD CoreXY
                        <span className="engine-toggle-sub">2 Motors</span>
                      </button>
                      <button
                        type="button"
                        className={`engine-toggle-btn ${driveMode === "awd" ? "active" : ""}`}
                        onClick={() => {
                          if (driveMode === "awd") return;
                          update(
                            (d) => {
                              d.gantry.drive_mode = "awd";
                            },
                            "Drive Mode: 2WD → AWD (4-Motor)"
                          );
                        }}
                      >
                        AWD CoreXY
                        <span className="engine-toggle-sub">4 Motors</span>
                      </button>
                    </div>
                  </div>

                  <div className="field-row">
                    <label>Pulley Mounts</label>
                    <div className="engine-toggle">
                      <button
                        type="button"
                        className={`engine-toggle-btn ${!doubleShearPulleys ? "active" : ""}`}
                        onClick={() => {
                          if (!doubleShearPulleys) return;
                          update(
                            (d) => {
                              d.gantry.double_shear_pulleys = false;
                            },
                            "Pulley Mounts: Double-Shear → Single-Shear"
                          );
                        }}
                      >
                        Single-Shear
                        <span className="engine-toggle-sub">Standard</span>
                      </button>
                      <button
                        type="button"
                        className={`engine-toggle-btn ${doubleShearPulleys ? "active" : ""}`}
                        onClick={() => {
                          if (doubleShearPulleys) return;
                          update(
                            (d) => {
                              d.gantry.double_shear_pulleys = true;
                            },
                            "Pulley Mounts: Single-Shear → Double-Shear"
                          );
                        }}
                      >
                        Double-Shear
                        <span className="engine-toggle-sub">Bearings</span>
                      </button>
                    </div>
                  </div>
                </div>
              )}

              {showHints && (
                <div className="hint-line">Lighter X-beam = faster Y acceleration &amp; less ringing</div>
              )}

              <div className="field-grid-2">
                <div className="field-row">
                  <label>X Beam Profile</label>
                  <select
                    value={spec.gantry.x_beam_profile}
                    onChange={(e) => {
                      const nextVal = e.target.value;
                      update(
                        (d) => {
                          d.gantry.x_beam_profile = nextVal;
                        },
                        `X Beam Profile: ${profileLabel(spec.gantry.x_beam_profile)} → ${profileLabel(nextVal)}`
                      );
                    }}
                  >
                    {catalog.profiles.map((p) => (
                      <option key={p.id} value={p.id}>
                        {p.name}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="field-row">
                  <label>X Beam Material</label>
                  <select
                    value={spec.gantry.x_beam_material}
                    onChange={(e) => {
                      const nextVal = e.target.value;
                      update(
                        (d) => {
                          d.gantry.x_beam_material = nextVal;
                        },
                        `X Beam Material: ${materialLabel(spec.gantry.x_beam_material)} → ${materialLabel(nextVal)}`
                      );
                    }}
                  >
                    {catalog.materials.map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.name}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {showHints && (
                <div className="hint-line">Wider/stiffer belts reduce stretch and sharpen corners</div>
              )}

              <div className="field-grid-2">
                <div className="field-row">
                  <label>Belt Width</label>
                  <select
                    value={spec.gantry.belt.type}
                    onChange={(e) => {
                      const nextVal = e.target.value as PrinterSpecData["gantry"]["belt"]["type"];
                      update(
                        (d) => {
                          d.gantry.belt.type = nextVal;
                        },
                        `Belt Width: ${BELT_LABELS[spec.gantry.belt.type]} → ${BELT_LABELS[nextVal]}`
                      );
                    }}
                  >
                    <option value="gt2_6mm">GT2 6mm</option>
                    <option value="gt2_9mm">GT2 9mm</option>
                    <option value="gt2_12mm">GT2 12mm</option>
                  </select>
                </div>
                <div className="field-row">
                  <label>Belt Core</label>
                  <select
                    value={spec.gantry.belt.core}
                    onChange={(e) => {
                      const nextVal = e.target.value as PrinterSpecData["gantry"]["belt"]["core"];
                      update(
                        (d) => {
                          d.gantry.belt.core = nextVal;
                        },
                        `Belt Core: ${spec.gantry.belt.core} → ${nextVal}`
                      );
                    }}
                  >
                    <option value="fiberglass">Fiberglass</option>
                    <option value="aramid">Aramid</option>
                    <option value="steel">Steel</option>
                  </select>
                </div>
              </div>

              <div className="slider-row">
                <label>
                  <span>Belt Tension (N)</span>
                  <span className="tension-readout">
                    <strong>{tensionDraft} N</strong>
                    {beltPluckHz != null && (
                      <span className="tension-hz-pill">~{Math.round(beltPluckHz)} Hz @ 150 mm</span>
                    )}
                  </span>
                </label>
                <div className="tension-control-row">
                  <input
                    type="range"
                    min={15}
                    max={120}
                    step={1}
                    value={Math.max(15, Math.min(120, tensionDraft))}
                    onChange={(e) => handleTensionSliderChange(Number(e.target.value))}
                    onPointerUp={() => commitBeltTension(tensionDraft)}
                    onKeyUp={() => commitBeltTension(tensionDraft)}
                  />
                  <input
                    type="number"
                    className="tension-num-input"
                    min={15}
                    max={120}
                    step={1}
                    value={tensionDraft}
                    onChange={(e) => {
                      const v = Number(e.target.value);
                      if (!Number.isFinite(v)) return;
                      setTensionDraft(v);
                      if (v >= 15 && v <= 120) {
                        commitBeltTension(v);
                      }
                    }}
                    onBlur={() => commitBeltTension(tensionDraft)}
                  />
                </div>
                {(showHints || beltOvertension || shaftBendingRisk) && (
                  <div className={`hint-line ${beltOvertension ? "danger" : shaftBendingRisk ? "warn" : ""}`}>
                    {beltPluckHz != null && (
                      <span>
                        Pluck tune: <strong>~{Math.round(beltPluckHz)} Hz @ 150 mm</strong>
                        {beltMaxWorkingN != null ? ` (max ${Math.round(beltMaxWorkingN)} N)` : ""}.{" "}
                      </span>
                    )}
                    {beltOvertension
                      ? "Belt overtensioned — upgrade to 9mm/12mm Aramid!"
                      : shaftBendingRisk
                        ? "High tension (>45 N) without double-shear bearings risks shaft bending."
                        : "High tension >45 N needs 9mm+ belts & double-shear bearings."}
                  </div>
                )}
              </div>

              {showHints && (
                <div className="hint-line">Higher voltage keeps stepper torque strong at high speed</div>
              )}

              <div className="field-grid-2">
                <div className="field-row">
                  <label>Max Vel (mm/s)</label>
                  <input
                    type="number"
                    value={spec.kinematics.max_velocity_mm_s}
                    onChange={(e) =>
                      updatePositiveNum(
                        e.target.value,
                        (d, v) => {
                          d.kinematics.max_velocity_mm_s = v;
                        },
                        (v) => `Max Vel: ${spec.kinematics.max_velocity_mm_s} → ${v} mm/s`
                      )
                    }
                  />
                </div>
                <div className="field-row">
                  <label>Max Accel (mm/s²)</label>
                  <input
                    type="number"
                    step={500}
                    value={spec.kinematics.max_accel_mm_s2}
                    onChange={(e) =>
                      updatePositiveNum(
                        e.target.value,
                        (d, v) => {
                          d.kinematics.max_accel_mm_s2 = v;
                        },
                        (v) =>
                          `Max Accel: ${spec.kinematics.max_accel_mm_s2.toLocaleString()} → ${v.toLocaleString()} mm/s²`
                      )
                    }
                  />
                </div>
              </div>

              <div className="field-grid-2">
                <div className="field-row">
                  <label>Input Shaper</label>
                  <select
                    value={spec.kinematics.input_shaper}
                    onChange={(e) => {
                      const nextVal = e.target.value as PrinterSpecData["kinematics"]["input_shaper"];
                      update(
                        (d) => {
                          d.kinematics.input_shaper = nextVal;
                        },
                        `Input Shaper: ${SHAPER_LABELS[spec.kinematics.input_shaper]} → ${SHAPER_LABELS[nextVal]}`
                      );
                    }}
                  >
                    <option value="auto">Auto (Klipper)</option>
                    <option value="none">None</option>
                    <option value="zv">ZV</option>
                    <option value="mzv">MZV</option>
                    <option value="ei">EI</option>
                  </select>
                </div>
                <div className="field-row">
                  <label>Stepper Voltage (V)</label>
                  <input
                    type="number"
                    value={spec.kinematics.xy_stepper.supply_voltage_v}
                    onChange={(e) =>
                      updatePositiveNum(
                        e.target.value,
                        (d, v) => {
                          d.kinematics.xy_stepper.supply_voltage_v = v;
                        },
                        (v) => `Stepper Voltage: ${spec.kinematics.xy_stepper.supply_voltage_v}V → ${v}V`
                      )
                    }
                  />
                </div>
              </div>
            </div>

            <div className="section-card">
              <div className="section-title">Toolhead &amp; Hotend Thermal</div>

              {showHints && (
                <div className="hint-line">Lighter toolhead directly boosts X/Y resonance &amp; safe accel</div>
              )}

              <div className="field-grid-2">
                <div className="field-row">
                  <label>Toolhead Mass (g)</label>
                  <input
                    type="number"
                    value={spec.toolhead.total_mass_g}
                    onChange={(e) =>
                      updatePositiveNum(
                        e.target.value,
                        (d, v) => {
                          d.toolhead.total_mass_g = v;
                        },
                        (v) => `Toolhead Mass: ${spec.toolhead.total_mass_g}g → ${v}g`
                      )
                    }
                  />
                </div>
                <div className="field-row">
                  <label>Heater Power (W)</label>
                  <input
                    type="number"
                    value={spec.toolhead.hotend.heater_power_w}
                    onChange={(e) =>
                      updatePositiveNum(
                        e.target.value,
                        (d, v) => {
                          d.toolhead.hotend.heater_power_w = v;
                        },
                        (v) => `Heater Power: ${spec.toolhead.hotend.heater_power_w}W → ${v}W`
                      )
                    }
                  />
                </div>
              </div>

              {showHints && (
                <div className="hint-line">
                  Longer melt zone &amp; higher wattage unlock faster print flow (mm³/s)
                </div>
              )}

              <div className="field-grid-2">
                <div className="field-row">
                  <label>Melt Zone (mm)</label>
                  <input
                    type="number"
                    value={spec.toolhead.hotend.melt_zone_length_mm}
                    onChange={(e) =>
                      updatePositiveNum(
                        e.target.value,
                        (d, v) => {
                          d.toolhead.hotend.melt_zone_length_mm = v;
                        },
                        (v) => `Melt Zone: ${spec.toolhead.hotend.melt_zone_length_mm} → ${v} mm`
                      )
                    }
                  />
                </div>
                <div className="field-row">
                  <label>Nozzle Dia (mm)</label>
                  <input
                    type="number"
                    step={0.1}
                    value={spec.toolhead.hotend.nozzle_diameter_mm}
                    onChange={(e) =>
                      updatePositiveNum(
                        e.target.value,
                        (d, v) => {
                          d.toolhead.hotend.nozzle_diameter_mm = v;
                        },
                        (v) => `Nozzle Dia: ${spec.toolhead.hotend.nozzle_diameter_mm} → ${v} mm`
                      )
                    }
                  />
                </div>
              </div>

              {showHints && (
                <div className="hint-line">
                  Titanium stops heat-creep clogs; higher CFM improves overhangs &amp; bridges
                </div>
              )}

              <div className="field-grid-2">
                <div className="field-row">
                  <label>Heatbreak Material</label>
                  <select
                    value={spec.toolhead.hotend.heatbreak_material}
                    onChange={(e) => {
                      const nextVal = e.target.value;
                      update(
                        (d) => {
                          d.toolhead.hotend.heatbreak_material = nextVal;
                        },
                        `Heatbreak: ${HEATBREAK_LABELS[spec.toolhead.hotend.heatbreak_material] ?? spec.toolhead.hotend.heatbreak_material} → ${HEATBREAK_LABELS[nextVal] ?? nextVal}`
                      );
                    }}
                  >
                    <option value="titanium_grade5">Titanium Gr.5</option>
                    <option value="stainless_304">Stainless 304</option>
                    <option value="brass_c360">Brass C360</option>
                  </select>
                </div>
                <div className="field-row">
                  <label>Cooling Fan (CFM)</label>
                  <input
                    type="number"
                    step={0.5}
                    value={spec.toolhead.hotend.part_cooling_cfm}
                    onChange={(e) =>
                      updatePositiveNum(
                        e.target.value,
                        (d, v) => {
                          d.toolhead.hotend.part_cooling_cfm = v;
                        },
                        (v) => `Cooling Fan: ${spec.toolhead.hotend.part_cooling_cfm} → ${v} CFM`
                      )
                    }
                  />
                </div>
              </div>
            </div>
          </>
        )}

        {tab === "part_fea" && (
          <>
            <div className="section-card">
              <div className="section-title">
                <span>3D Solid Part FEA</span>
                <span style={{ color: feaEngine === "tet10" ? "var(--accent)" : "var(--muted)", fontSize: 10 }}>
                  {feaEngine === "tet10" ? "Tet10 (P2)" : "Voxel Hex8"}
                </span>
              </div>

              <div className="field-row">
                <label>Solver Engine</label>
                <div className="engine-toggle">
                  <button
                    type="button"
                    className={`engine-toggle-btn ${feaEngine === "tet10" ? "active" : ""}`}
                    onClick={() => setFeaEngine("tet10")}
                  >
                    Curved Tet10
                    <span className="engine-toggle-sub">Gmsh + scikit-fem</span>
                  </button>
                  <button
                    type="button"
                    className={`engine-toggle-btn ${feaEngine === "hex8" ? "active" : ""}`}
                    onClick={() => setFeaEngine("hex8")}
                  >
                    Fast Voxel Hex8
                    <span className="engine-toggle-sub">Structured Grid</span>
                  </button>
                </div>
              </div>

              <div className="field-row">
                <label>Parametric Bracket / Mount</label>
                <select value={partPreset} onChange={(e) => setPartPreset(e.target.value)}>
                  {catalog.part_presets.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.name}
                    </option>
                  ))}
                </select>
              </div>
              <div className="field-grid-2">
                <div className="field-row">
                  <label>Material</label>
                  <select
                    value={partMat}
                    onChange={(e) => {
                      setPartMat(e.target.value);
                      setPartFdm(e.target.value.includes("printed"));
                    }}
                  >
                    {catalog.materials.map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.name}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="field-row">
                  <label>FDM Z-Anisotropy</label>
                  <select value={partFdm ? "yes" : "no"} onChange={(e) => setPartFdm(e.target.value === "yes")}>
                    <option value="yes">FDM Layer Model</option>
                    <option value="no">Isotropic (CNC)</option>
                  </select>
                </div>
              </div>
              <div className="field-grid-3">
                <div className="field-row">
                  <label>Fx (N)</label>
                  <input type="number" value={fx} onChange={(e) => setFx(Number(e.target.value))} />
                </div>
                <div className="field-row">
                  <label>Fy (N)</label>
                  <input type="number" value={fy} onChange={(e) => setFy(Number(e.target.value))} />
                </div>
                <div className="field-row">
                  <label>Fz (N)</label>
                  <input type="number" value={fz} onChange={(e) => setFz(Number(e.target.value))} />
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
                <div style={{ fontSize: 10.5, color: "var(--muted)", marginBottom: 6, lineHeight: 1.3 }}>
                  <code>.step</code>/<code>.stp</code> → Curved Tet10 (Gmsh) • <code>.stl</code>/<code>.obj</code> → Voxel Hex8
                </div>
                <div className="field-grid-2" style={{ marginBottom: 6 }}>
                  <div className="field-row">
                    <label>Fixed Clamp Face</label>
                    <select value={fixedFace} onChange={(e) => setFixedFace(e.target.value)}>
                      <option value="min_z">Bottom (-Z)</option>
                      <option value="min_x">Left (-X)</option>
                      <option value="min_y">Front (-Y)</option>
                    </select>
                  </div>
                  <div className="field-row">
                    <label>Loaded Face</label>
                    <select value={loadFace} onChange={(e) => setLoadFace(e.target.value)}>
                      <option value="max_z">Top (+Z)</option>
                      <option value="max_x">Right (+X)</option>
                      <option value="max_y">Rear (+Y)</option>
                    </select>
                  </div>
                </div>
                <input
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
                <span style={{ color: "var(--purple)", fontSize: 10 }}>Generative</span>
              </div>
              <div style={{ fontSize: 11, color: "var(--muted)", lineHeight: 1.35 }}>
                Carves low-stress voxels from the selected preset &amp; load case above while maximizing
                stiffness-to-weight ratio.
              </div>

              <div className="slider-row">
                <label>
                  <span>Target Volume Fraction</span>
                  <strong style={{ color: "var(--text)" }}>{Math.round(targetVolFrac * 100)}%</strong>
                </label>
                <input
                  type="range"
                  min={0.25}
                  max={0.8}
                  step={0.05}
                  value={targetVolFrac}
                  onChange={(e) => setTargetVolFrac(Number(e.target.value))}
                />
              </div>

              <div className="slider-row">
                <label>
                  <span>SIMP Iterations</span>
                  <strong style={{ color: "var(--text)" }}>{topoIterations}</strong>
                </label>
                <input
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
              className="code-editor"
              value={yamlText}
              onChange={(e) => onYamlChange(e.target.value)}
              spellCheck={false}
            />
          </div>
        )}

        {tab === "gcode" && (
          <div className="section-card">
            <div className="section-title">Custom G-code Input</div>
            <p style={{ margin: 0, color: "var(--muted)", fontSize: 11.5 }}>
              Leave blank to run the built-in high-speed calibration cube &amp; ringing torture benchmark, or paste
              any Marlin/Klipper/OrcaSlicer G-code below.
            </p>
            <textarea
              className="code-editor"
              placeholder="; Paste G-code here (G0/G1 moves)..."
              value={gcodeText}
              onChange={(e) => onGcodeChange(e.target.value)}
              spellCheck={false}
            />
          </div>
        )}
      </div>
    </aside>
  );
};
