import React from "react";
import { SimulationResponse, SolidFEAResponse } from "../types";

interface Props {
  sim: SimulationResponse | null;
  solidFea: SolidFEAResponse | null;
  error: string | null;
}

function scoreColor(v: number): string {
  if (v >= 80) return "var(--green)";
  if (v >= 60) return "var(--accent)";
  if (v >= 45) return "var(--amber)";
  return "var(--red)";
}

export const MetricsPanel: React.FC<Props> = ({ sim, solidFea, error }) => {
  if (error) {
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

  return (
    <aside className="sidebar right">
      <div className="sidebar-scroll">
        <div className="score-hero">
          <div>
            <div style={{ fontSize: 11, color: "var(--muted)", textTransform: "uppercase", fontWeight: 700 }}>
              Composite Engineering Score
            </div>
            <div style={{ fontSize: 13, fontWeight: 600, marginTop: 2 }}>{sim.printer_name}</div>
          </div>
          <div className="score-number" style={{ color: scoreColor(scores.composite) }}>
            {scores.composite.toFixed(1)}
          </div>
        </div>

        <div className="section-card">
          <div className="section-title">Subsystem Scorecard</div>
          {(
            [
              ["Structural (FEA)", scores.structural],
              ["Dynamics & Shaper", scores.dynamics],
              ["Hotend Thermal", scores.thermal],
              ["Virtual Print Quality", scores.print_quality],
              ["Mass & Packaging", scores.efficiency],
            ] as Array<[string, number]>
          ).map(([label, val]) => (
            <div className="bar-row" key={label}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11.5 }}>
                <span style={{ color: "var(--muted)" }}>{label}</span>
                <strong>{val.toFixed(1)}</strong>
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
            const cls = d.startsWith("COLLISION")
              ? "bad"
              : d.includes("risk") || d.includes("Low") || d.includes("detected")
                ? "warn"
                : "";
            return (
              <div key={idx} className={`diag-item ${cls}`}>
                {d}
              </div>
            );
          })}
        </div>

        <div className="section-card">
          <div className="section-title">Static 3D Frame FEA</div>
          <div className="metric-kv">
            <span>Gravity Sag (Nozzle)</span>
            <span className="metric-val">{fea.gravity_sag_toolhead_um.toFixed(1)} µm</span>
          </div>
          <div className="metric-kv">
            <span>Peak Accel Deflection</span>
            <span className="metric-val">{fea.combined_max_deflection_um.toFixed(1)} µm</span>
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

        <div className="section-card">
          <div className="section-title">Modal Resonance & Input Shaper</div>
          <div className="metric-kv">
            <span>Fundamental Mode f₁</span>
            <span className="metric-val">{mod.fundamental_freq_hz.toFixed(1)} Hz</span>
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
            <span className="metric-val">{mod.recommended_max_accel_mm_s2.toFixed(0)} mm/s²</span>
          </div>
          <div className="metric-kv">
            <span>Stepper Torque Margin</span>
            <span className={`metric-val ${kin.step_skip_risk ? "tag-bad" : "tag-ok"}`}>
              {kin.torque_safety_margin.toFixed(2)}×
            </span>
          </div>
          <div className="metric-kv">
            <span>Belt Peak Tension</span>
            <span className={`metric-val ${kin.belt_slip_risk ? "tag-bad" : "tag-ok"}`}>
              {kin.belt_peak_tension_n.toFixed(1)} N {kin.belt_slip_risk ? "(SLIP)" : ""}
            </span>
          </div>
        </div>

        {solidFea && (
          <div className="section-card" style={{ borderColor: "rgba(56,189,248,0.45)" }}>
            <div className="section-title">
              <span>3D Solid Hex8 Part FEA</span>
              <span style={{ color: solidFea.safety_factor < 1.5 ? "var(--red)" : "var(--green)" }}>
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
        )}

        <div className="section-card">
          <div className="section-title">Hotend FVM Thermal & Cooling ({th.filament_id.toUpperCase()})</div>
          <div className="metric-kv">
            <span>Max Volumetric Flow</span>
            <span className="metric-val">{th.max_volumetric_flow_mm3_s.toFixed(1)} mm³/s</span>
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
              {th.cold_side_temp_c.toFixed(1)}°C / {th.glass_transition_c}°C {th.heat_creep_risk ? "(CREEP)" : "(OK)"}
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

        <div className="section-card">
          <div className="section-title">Virtual G-code Print</div>
          <div className="metric-kv">
            <span>Simulated Print Time</span>
            <span className="metric-val">{vp.total_time_s.toFixed(1)} s</span>
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
      </div>
    </aside>
  );
};
