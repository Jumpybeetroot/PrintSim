import React, { useState } from "react";
import { CatalogResponse, PrinterSpecData } from "../types";

interface Props {
  catalog: CatalogResponse | null;
  selectedPreset: string;
  onSelectPreset: (id: string) => void;
  spec: PrinterSpecData | null;
  onSpecChange: (next: PrinterSpecData) => void;
  yamlText: string;
  onYamlChange: (txt: string) => void;
  onSimulateYaml: () => void;
  gcodeText: string;
  onGcodeChange: (txt: string) => void;
  loading: boolean;
}

export const DesignPanel: React.FC<Props> = ({
  catalog,
  selectedPreset,
  onSelectPreset,
  spec,
  onSpecChange,
  yamlText,
  onYamlChange,
  onSimulateYaml,
  gcodeText,
  onGcodeChange,
  loading,
}) => {
  const [tab, setTab] = useState<"params" | "yaml" | "gcode">("params");

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
          Parametric
        </button>
        <button className={`tab-btn ${tab === "yaml" ? "active" : ""}`} onClick={() => setTab("yaml")}>
          YAML DSL
        </button>
        <button className={`tab-btn ${tab === "gcode" ? "active" : ""}`} onClick={() => setTab("gcode")}>
          G-code
        </button>
      </div>

      <div className="sidebar-scroll">
        {/* Preset switcher always available */}
        <div className="section-card">
          <div className="section-title">Preset Template</div>
          <select value={selectedPreset} onChange={(e) => onSelectPreset(e.target.value)}>
            {catalog.presets.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name} ({p.architecture})
              </option>
            ))}
          </select>
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
