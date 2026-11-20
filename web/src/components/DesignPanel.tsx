import React, { useState } from "react";
import { CatalogResponse, PrinterSpecData, SolidFeaEngine, TopoOptRequestParams } from "../types";

interface Props {
  catalog: CatalogResponse | null;
  selectedPreset: string;
  onSelectPreset: (id: string) => void;
  filamentId: string;
  onFilamentChange: (id: string) => void;
  spec: PrinterSpecData | null;
  onSpecChange: (next: PrinterSpecData) => void;
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

export const DesignPanel: React.FC<Props> = ({
  catalog,
  selectedPreset,
  onSelectPreset,
  filamentId,
  onFilamentChange,
  spec,
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

  const update = (mutator: (draft: PrinterSpecData) => void) => {
    const clone: PrinterSpecData = JSON.parse(JSON.stringify(spec));
    mutator(clone);
    onSpecChange(clone);
  };

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
        {/* Preset & Filament selectors always available */}
        <div className="section-card">
          <div className="section-title">Preset & Filament</div>
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
            <label>Filament Material (Rheology & Tg)</label>
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
            <div className="section-card">
              <div className="section-title">Frame & Structure (FEA)</div>
              <div className="field-row">
                <label>Architecture</label>
                <select
                  value={spec.architecture}
                  onChange={(e) =>
                    update((d) => {
                      d.architecture = e.target.value as PrinterSpecData["architecture"];
                    })
                  }
                >
                  <option value="corexy">CoreXY</option>
                  <option value="cartesian_bedslinger">Cartesian Bedslinger</option>
                  <option value="delta">Delta (3-Tower)</option>
                </select>
              </div>

              <div className="field-grid-3">
                <div className="field-row">
                  <label>Width mm</label>
                  <input
                    type="number"
                    value={spec.frame.width_mm}
                    onChange={(e) => update((d) => (d.frame.width_mm = Number(e.target.value)))}
                  />
                </div>
                <div className="field-row">
                  <label>Depth mm</label>
                  <input
                    type="number"
                    value={spec.frame.depth_mm}
                    onChange={(e) => update((d) => (d.frame.depth_mm = Number(e.target.value)))}
                  />
                </div>
                <div className="field-row">
                  <label>Height mm</label>
                  <input
                    type="number"
                    value={spec.frame.height_mm}
                    onChange={(e) => update((d) => (d.frame.height_mm = Number(e.target.value)))}
                  />
                </div>
              </div>

              <div className="field-grid-2">
                <div className="field-row">
                  <label>Frame Profile</label>
                  <select
                    value={spec.frame.profile}
                    onChange={(e) => update((d) => (d.frame.profile = e.target.value))}
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
                    onChange={(e) => update((d) => (d.frame.material = e.target.value))}
                  >
                    {catalog.materials.map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.name}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="field-grid-2">
                <div className="field-row">
                  <label>Bracing</label>
                  <select
                    value={spec.frame.bracing}
                    onChange={(e) =>
                      update((d) => {
                        d.frame.bracing = e.target.value as PrinterSpecData["frame"]["bracing"];
                      })
                    }
                  >
                    <option value="none">None (Open Frame)</option>
                    <option value="corner_gussets">Corner Gussets</option>
                    <option value="rear_cross">Rear Cross + Top</option>
                    <option value="full_truss">Full 3D Truss</option>
                  </select>
                </div>
                <div className="field-row">
                  <label>Bolted Panels</label>
                  <select
                    value={spec.frame.enclosed_panels ? "yes" : "no"}
                    onChange={(e) => update((d) => (d.frame.enclosed_panels = e.target.value === "yes"))}
                  >
                    <option value="no">Open</option>
                    <option value="yes">Shear Panels</option>
                  </select>
                </div>
              </div>
            </div>

            <div className="section-card">
              <div className="section-title">Gantry, Belts & Kinematics</div>
              <div className="field-grid-2">
                <div className="field-row">
                  <label>X Beam Profile</label>
                  <select
                    value={spec.gantry.x_beam_profile}
                    onChange={(e) => update((d) => (d.gantry.x_beam_profile = e.target.value))}
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
                    onChange={(e) => update((d) => (d.gantry.x_beam_material = e.target.value))}
                  >
                    {catalog.materials.map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.name}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="field-grid-3">
                <div className="field-row">
                  <label>Belt Width</label>
                  <select
                    value={spec.gantry.belt.type}
                    onChange={(e) =>
                      update((d) => {
                        d.gantry.belt.type = e.target.value as PrinterSpecData["gantry"]["belt"]["type"];
                      })
                    }
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
                    onChange={(e) =>
                      update((d) => {
                        d.gantry.belt.core = e.target.value as PrinterSpecData["gantry"]["belt"]["core"];
                      })
                    }
                  >
                    <option value="fiberglass">Fiberglass</option>
                    <option value="aramid">Aramid</option>
                    <option value="steel">Steel</option>
                  </select>
                </div>
                <div className="field-row">
                  <label>Tension N</label>
                  <input
                    type="number"
                    value={spec.gantry.belt.pretension_n}
                    onChange={(e) => update((d) => (d.gantry.belt.pretension_n = Number(e.target.value)))}
                  />
                </div>
              </div>

              <div className="field-grid-2">
                <div className="field-row">
                  <label>Max Vel (mm/s)</label>
                  <input
                    type="number"
                    value={spec.kinematics.max_velocity_mm_s}
                    onChange={(e) => update((d) => (d.kinematics.max_velocity_mm_s = Number(e.target.value)))}
                  />
                </div>
                <div className="field-row">
                  <label>Max Accel (mm/s²)</label>
                  <input
                    type="number"
                    step={500}
                    value={spec.kinematics.max_accel_mm_s2}
                    onChange={(e) => update((d) => (d.kinematics.max_accel_mm_s2 = Number(e.target.value)))}
                  />
                </div>
              </div>

              <div className="field-grid-2">
                <div className="field-row">
                  <label>Input Shaper</label>
                  <select
                    value={spec.kinematics.input_shaper}
                    onChange={(e) =>
                      update((d) => {
                        d.kinematics.input_shaper = e.target.value as PrinterSpecData["kinematics"]["input_shaper"];
                      })
                    }
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
                      update((d) => (d.kinematics.xy_stepper.supply_voltage_v = Number(e.target.value)))
                    }
                  />
                </div>
              </div>
            </div>

            <div className="section-card">
              <div className="section-title">Toolhead & Hotend Thermal</div>
              <div className="field-grid-2">
                <div className="field-row">
                  <label>Toolhead Mass (g)</label>
                  <input
                    type="number"
                    value={spec.toolhead.total_mass_g}
                    onChange={(e) => update((d) => (d.toolhead.total_mass_g = Number(e.target.value)))}
                  />
                </div>
                <div className="field-row">
                  <label>Heater Power (W)</label>
                  <input
                    type="number"
                    value={spec.toolhead.hotend.heater_power_w}
                    onChange={(e) => update((d) => (d.toolhead.hotend.heater_power_w = Number(e.target.value)))}
                  />
                </div>
              </div>

              <div className="field-grid-2">
                <div className="field-row">
                  <label>Melt Zone (mm)</label>
                  <input
                    type="number"
                    value={spec.toolhead.hotend.melt_zone_length_mm}
                    onChange={(e) =>
                      update((d) => (d.toolhead.hotend.melt_zone_length_mm = Number(e.target.value)))
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
                      update((d) => (d.toolhead.hotend.nozzle_diameter_mm = Number(e.target.value)))
                    }
                  />
                </div>
              </div>

              <div className="field-grid-2">
                <div className="field-row">
                  <label>Heatbreak Material</label>
                  <select
                    value={spec.toolhead.hotend.heatbreak_material}
                    onChange={(e) =>
                      update((d) => (d.toolhead.hotend.heatbreak_material = e.target.value))
                    }
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
                      update((d) => (d.toolhead.hotend.part_cooling_cfm = Number(e.target.value)))
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
              Leave blank to run the built-in high-speed calibration cube & ringing torture benchmark, or paste
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
