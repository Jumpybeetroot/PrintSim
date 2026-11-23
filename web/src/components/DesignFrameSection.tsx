import React, { useEffect, useRef, useState } from "react";
import {
  ARCH_LABELS,
  BRACING_LABELS,
  Z_DRIVE_LABELS,
} from "../constants/printerLabels";
import {
  CatalogResponse,
  CornerJointType,
  ExtrusionFillingInfo,
  ExtrusionFillingType,
  FrameFeetType,
  PrinterSpecData,
  SimulationResponse,
} from "../types";

const FALLBACK_EXTRUSION_FILLINGS: ExtrusionFillingInfo[] = [
  {
    id: "none",
    name: "Hollow (Unfilled)",
    short_name: "Hollow",
    density_kg_m3: 0,
    youngs_modulus_pa: 0,
    bonded: false,
    damping_boost: 0,
    cost_per_kg_usd: 0,
    color_hex: "#94a3b8",
    notes: "Standard hollow aluminum extrusion (low structural damping)",
  },
  {
    id: "pu_foam",
    name: "Expanding PU Acoustic Foam",
    short_name: "PU Foam",
    density_kg_m3: 45,
    youngs_modulus_pa: 0.02e9,
    bonded: false,
    damping_boost: 0.01,
    cost_per_kg_usd: 18,
    color_hex: "#fde68a",
    notes: "Ultra-light acoustic deadening; negligible added mass or stiffness",
  },
  {
    id: "silica_sand",
    name: "Kiln-Dried Silica Sand",
    short_name: "Silica Sand",
    density_kg_m3: 1550,
    youngs_modulus_pa: 0,
    bonded: false,
    damping_boost: 0.028,
    cost_per_kg_usd: 1.5,
    color_hex: "#d6b88d",
    notes: "Low-cost granular Coulomb friction damping; zero bonded bending stiffness",
  },
  {
    id: "lead_shot",
    name: "Fine Lead Shot (#8–#9)",
    short_name: "Lead Shot",
    density_kg_m3: 6800,
    youngs_modulus_pa: 0,
    bonded: false,
    damping_boost: 0.042,
    cost_per_kg_usd: 11,
    color_hex: "#475569",
    notes: "Maximum granular impact damping & base inertia; heavy mass lowers raw Hz",
  },
  {
    id: "epoxy_granite",
    name: "Epoxy Granite (85% Quartz + Epoxy)",
    short_name: "Epoxy Granite",
    density_kg_m3: 2250,
    youngs_modulus_pa: 30e9,
    bonded: true,
    damping_boost: 0.036,
    cost_per_kg_usd: 9.5,
    color_hex: "#64748b",
    notes: "Mineral casting bonds to cavity walls: +10-16% bending stiffness & strong resin damping",
  },
  {
    id: "uhpc_grout",
    name: "Non-Shrink UHPC / CSA Grout",
    short_name: "UHPC Grout",
    density_kg_m3: 2350,
    youngs_modulus_pa: 40e9,
    bonded: true,
    damping_boost: 0.03,
    cost_per_kg_usd: 4.5,
    color_hex: "#9ca3af",
    notes: "High-modulus bonded cementitious grout: highest core EI boost + solid damping",
  },
];

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
  const topSuffix = frame.top_panel === false ? ", No Top" : "";
  return `${frame.panel_thickness_mm}mm ${mat} (${mnt}${topSuffix})`;
}

const CORNER_JOINT_OPTIONS: Array<{ id: CornerJointType; label: string; shortLabel: string }> = [
  { id: "blind_joint", label: "Blind Joint (T-Slot)", shortLabel: "Blind Joint" },
  { id: "l_bracket", label: "L-Bracket (Die-Cast)", shortLabel: "L-Bracket" },
  { id: "gusset_plate", label: "Gusset Plate", shortLabel: "Gusset Plate" },
  { id: "corner_cube", label: "3-Way Corner Cube", shortLabel: "Corner Cube" },
  { id: "rigid", label: "Ideal Rigid", shortLabel: "Ideal Rigid" },
];

const FRAME_FEET_OPTIONS: Array<{
  id: FrameFeetType;
  label: string;
  shortLabel: string;
  hint: string;
}> = [
  {
    id: "rigid_bolted",
    label: "Rigid Bolted",
    shortLabel: "Rigid Bolted",
    hint: "Frame rigidly anchored to heavy benchtop — suppresses whole-frame sway so all modes are internal structural flex.",
  },
  {
    id: "rubber_feet",
    label: "Rubber / TPU Feet",
    shortLabel: "Rubber Feet",
    hint: "Compliant benchtop feet allow low-Hz whole-frame rocking (low nozzle-to-bed error!) while isolating bench vibrations.",
  },
  {
    id: "anti_vib_pads",
    label: "Anti-Vib Pads / HULA",
    shortLabel: "Anti-Vib Pads",
    hint: "Ultra-soft omnidirectional isolation (HULA/sorbothane) lets the whole chassis float together with minimal nozzle-to-bed print error.",
  },
  {
    id: "paver_on_foam",
    label: "Concrete Paver on Foam (18 kg Inert Base)",
    shortLabel: "Paver on Foam",
    hint: "18 kg concrete paver over acoustic foam decouples benchtop noise while adding massive base inertia to resist frame rocking.",
  },
];

const TEST_POS_PRESETS: Array<{
  id: string;
  label: string;
  pos: [number, number, number];
}> = [
  { id: "center", label: "Center", pos: [0.5, 0.5, 0.5] },
  { id: "front_left", label: "Front-Left", pos: [0.15, 0.15, 0.5] },
  { id: "top_z", label: "Top Z", pos: [0.5, 0.5, 0.85] },
  { id: "low_z", label: "Low Z", pos: [0.5, 0.5, 0.15] },
];

export interface DesignFrameSectionProps {
  spec: PrinterSpecData;
  catalog: CatalogResponse;
  sim: SimulationResponse | null;
  showHints: boolean;
  update: (mutator: (draft: PrinterSpecData) => void, changeLabel?: string) => void;
  updatePositiveNum: (
    raw: string,
    mutator: (draft: PrinterSpecData, val: number) => void,
    changeLabelBuilder: (val: number) => string
  ) => void;
  profileLabel: (id: string) => string;
  materialLabel: (id: string) => string;
}

export const DesignFrameSection: React.FC<DesignFrameSectionProps> = ({
  spec,
  catalog,
  sim,
  showHints,
  update,
  updatePositiveNum,
  profileLabel,
  materialLabel,
}) => {
  const [chamberTempDraft, setChamberTempDraft] = useState<number>(
    Math.round(spec.frame.chamber_temp_c ?? 25)
  );
  const chamberTempTimerRef = useRef<number | null>(null);

  const [gantryClearanceDraft, setGantryClearanceDraft] = useState<number>(
    Math.round(spec.frame.gantry_top_clearance_mm ?? 55)
  );
  const gantryClearanceTimerRef = useRef<number | null>(null);

  useEffect(() => {
    setChamberTempDraft(Math.round(spec.frame.chamber_temp_c ?? 25));
  }, [spec.frame.chamber_temp_c]);

  useEffect(() => {
    setGantryClearanceDraft(Math.round(spec.frame.gantry_top_clearance_mm ?? 55));
  }, [spec.frame.gantry_top_clearance_mm]);

  useEffect(() => {
    return () => {
      if (chamberTempTimerRef.current != null) window.clearTimeout(chamberTempTimerRef.current);
      if (gantryClearanceTimerRef.current != null)
        window.clearTimeout(gantryClearanceTimerRef.current);
    };
  }, []);

  const specChamberTempC = Math.round(spec.frame.chamber_temp_c ?? 25);
  const commitChamberTemp = (rawVal: number) => {
    const val = Math.max(25, Math.min(85, Math.round(rawVal)));
    if (chamberTempTimerRef.current != null) {
      window.clearTimeout(chamberTempTimerRef.current);
      chamberTempTimerRef.current = null;
    }
    if (val === specChamberTempC) return;
    update(
      (d) => {
        d.frame.chamber_temp_c = val;
      },
      `Chamber Temp: ${specChamberTempC}°C → ${val}°C`
    );
  };

  const handleChamberTempSliderChange = (nextVal: number) => {
    setChamberTempDraft(nextVal);
    if (chamberTempTimerRef.current != null) window.clearTimeout(chamberTempTimerRef.current);
    chamberTempTimerRef.current = window.setTimeout(() => commitChamberTemp(nextVal), 220);
  };

  const specGantryClearanceMm = Math.round(spec.frame.gantry_top_clearance_mm ?? 55);
  const commitGantryClearance = (rawVal: number) => {
    const val = Math.max(25, Math.min(300, Math.round(rawVal)));
    if (gantryClearanceTimerRef.current != null) {
      window.clearTimeout(gantryClearanceTimerRef.current);
      gantryClearanceTimerRef.current = null;
    }
    if (val === specGantryClearanceMm) return;
    update(
      (d) => {
        d.frame.gantry_top_clearance_mm = val;
      },
      `Top-to-Gantry Clearance: ${specGantryClearanceMm} mm → ${val} mm`
    );
  };

  const handleGantryClearanceSliderChange = (nextVal: number) => {
    setGantryClearanceDraft(nextVal);
    if (gantryClearanceTimerRef.current != null)
      window.clearTimeout(gantryClearanceTimerRef.current);
    gantryClearanceTimerRef.current = window.setTimeout(
      () => commitGantryClearance(nextVal),
      220
    );
  };

  const currentCornerJoint: CornerJointType = spec.frame.corner_joint_type ?? "blind_joint";
  const effectivePanelMounting: PrinterSpecData["frame"]["panel_mounting"] = !spec.frame.enclosed_panels
    ? "none"
    : spec.frame.panel_mounting || "bolted_structural";
  const panelsEnabled = spec.frame.enclosed_panels && effectivePanelMounting !== "none";
  const currentPanelMaterial = spec.frame.panel_material || "acrylic_pmma";
  const currentPanelThickness = spec.frame.panel_thickness_mm || 3;
  const isFlyingGantry = spec.bed.z_drive === "belted_quad_z";
  const boltPanelsToGantry = spec.frame.bolt_panels_to_gantry ?? true;
  const topPanelInstalled = spec.frame.top_panel ?? true;
  const tridentBackboneEnabled = Boolean(spec.frame.trident_backbone);
  const simBackbone = sim?.geometry.trident_backbone ?? null;
  const thicknessChoices = PANEL_THICKNESS_OPTIONS.includes(currentPanelThickness)
    ? PANEL_THICKNESS_OPTIONS
    : [...PANEL_THICKNESS_OPTIONS, currentPanelThickness].sort((a, b) => a - b);

  const extrusionFillings: ExtrusionFillingInfo[] =
    catalog.extrusion_fillings && catalog.extrusion_fillings.length > 0
      ? catalog.extrusion_fillings
      : FALLBACK_EXTRUSION_FILLINGS;
  const currentFillingId: ExtrusionFillingType = spec.frame.extrusion_filling ?? "none";
  const selectedFilling =
    extrusionFillings.find((f) => f.id === currentFillingId) ?? extrusionFillings[0];
  const simFilling =
    sim?.geometry.extrusion_filling && sim.geometry.extrusion_filling.id === currentFillingId
      ? sim.geometry.extrusion_filling
      : null;
  const fillingAddedMassKg = simFilling?.added_mass_kg ?? 0;
  const fillingDampingRatio =
    simFilling?.damping_ratio ??
    sim?.modal_fea.frame_damping_ratio ??
    0.042 + (selectedFilling?.damping_boost ?? 0);

  return (
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

      {spec.architecture === "corexy" && (
        <>
          <div className="field-row">
            <label>
              <span>Gantry &amp; Z Architecture</span>
              <span style={{ color: isFlyingGantry ? "var(--amber)" : "var(--accent)", fontWeight: 600 }}>
                {isFlyingGantry ? "Flying Gantry" : "Fixed Gantry"}
              </span>
            </label>
            <select
              value={spec.bed.z_drive}
              onChange={(e) => {
                const nextZ = e.target.value as PrinterSpecData["bed"]["z_drive"];
                const prevLabel = Z_DRIVE_LABELS[spec.bed.z_drive] ?? spec.bed.z_drive;
                const nextLabel = Z_DRIVE_LABELS[nextZ] ?? nextZ;
                update(
                  (d) => {
                    d.bed.z_drive = nextZ;
                  },
                  `Gantry Type: ${prevLabel} → ${nextLabel}`
                );
              }}
            >
              <option value="leadscrew_triple">Fixed Gantry — Triple-Z Bed (Trident/VzBot)</option>
              <option value="belted_quad_z">Flying Gantry — Quad-Belted Z (Voron 2.4)</option>
              <option value="leadscrew_dual">Fixed Gantry — Dual-Z Bed</option>
              <option value="leadscrew_dual_belt_sync">
                Fixed Gantry — Belt-Synced Dual-Z (VzBot)
              </option>
              {spec.bed.z_drive === "leadscrew_single" && (
                <option value="leadscrew_single">Fixed Gantry — Single-Z Bed</option>
              )}
            </select>
          </div>
          {showHints && (
            <div className="hint-line">
              Fixed gantry (bed moves Z) lets structural panels bolt straight to the Y-rail &amp; motor
              beams; flying gantry (Voron 2.4) moves XY in Z, so panels only stiffen the outer cube.
            </div>
          )}
        </>
      )}

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

      {(() => {
        const extrusionProfiles = catalog.profiles.filter((p) =>
          p.id.startsWith("extrusion_")
        );
        const fallbackUprights = [
          { id: "extrusion_2020", name: "2020 Aluminum Extrusion" },
          { id: "extrusion_2040", name: "2040 Aluminum Extrusion" },
          { id: "extrusion_3030", name: "3030 Aluminum Extrusion" },
          { id: "extrusion_4040", name: "4040 Aluminum Extrusion" },
        ];
        const uprightOptions =
          extrusionProfiles.length > 0 ? extrusionProfiles : fallbackUprights;
        const currentUpright = spec.frame.upright_profile || "";
        const hasCustomUpright =
          Boolean(currentUpright) &&
          !uprightOptions.some((p) => p.id === currentUpright);
        const isMixed =
          Boolean(currentUpright) && currentUpright !== spec.frame.profile;

        return (
          <div className="field-row">
            <label>
              <span>Corner Upright Profile</span>
              {isMixed && (
                <span style={{ color: "var(--accent)", fontWeight: 600 }}>
                  Mixed Frame
                </span>
              )}
            </label>
            <select
              value={currentUpright}
              onChange={(e) => {
                const raw = e.target.value;
                const nextUpright = raw ? raw : null;
                const prevLbl = spec.frame.upright_profile
                  ? profileLabel(spec.frame.upright_profile)
                  : "Same as Frame";
                const nextLbl = nextUpright
                  ? profileLabel(nextUpright)
                  : "Same as Frame";
                update(
                  (d) => {
                    d.frame.upright_profile = nextUpright;
                  },
                  `Corner Uprights: ${prevLbl} → ${nextLbl}`
                );
              }}
            >
              <option value="">
                Same as Frame ({profileLabel(spec.frame.profile)})
              </option>
              {uprightOptions.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
              {hasCustomUpright && (
                <option value={currentUpright}>
                  {profileLabel(currentUpright)}
                </option>
              )}
            </select>
            {showHints && (
              <div className="hint-line">
                Use heavier vertical corner columns (e.g. 4040 uprights + 2020
                horizontals, DoomCube style) for high frame rigidity without full-frame
                weight/cost.
              </div>
            )}
          </div>
        );
      })()}

      {spec.architecture === "corexy" && (
        <div className="slider-row">
          <label>
            <span>Top-to-Gantry Clearance (mm)</span>
            <span className="tension-readout">
              <strong>{gantryClearanceDraft} mm</strong>
              <span className="kpi-sub">
                {gantryClearanceDraft < 50
                  ? "compact"
                  : gantryClearanceDraft <= 60
                    ? "stock"
                    : `+${gantryClearanceDraft - 55} mm tophat`}
              </span>
            </span>
          </label>
          <div className="tension-control-row">
            <input
              type="range"
              min={25}
              max={250}
              step={5}
              value={Math.max(25, Math.min(250, gantryClearanceDraft))}
              onChange={(e) => handleGantryClearanceSliderChange(Number(e.target.value))}
              onPointerUp={() => commitGantryClearance(gantryClearanceDraft)}
              onKeyUp={() => commitGantryClearance(gantryClearanceDraft)}
            />
            <input
              type="number"
              className="tension-num-input"
              min={25}
              max={300}
              step={5}
              value={gantryClearanceDraft}
              onChange={(e) => {
                const v = Number(e.target.value);
                if (!Number.isFinite(v)) return;
                setGantryClearanceDraft(v);
                if (v >= 25 && v <= 300) commitGantryClearance(v);
              }}
              onBlur={() => commitGantryClearance(gantryClearanceDraft)}
            />
          </div>
          {showHints && (
            <div className="hint-line">
              Adds headroom above the gantry for tall toolheads, umbilicals, or a tophat
              while lengthening the upper frame columns.
            </div>
          )}
        </div>
      )}

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
          <option value="top_cross">Top Frame X-Brace</option>
          <option value="rear_cross">Rear X-Brace + Top</option>
          <option value="side_cross">Side X-Braces (Left/Right)</option>
          <option value="rear_and_sides">Rear + Side X-Braces (Open Top)</option>
          <option value="full_truss">Full 3D Truss (All Faces)</option>
        </select>
      </div>

      {spec.architecture === "corexy" && (
        <div className="gantry-bolt-block">
          {isFlyingGantry ? (
            <>
              <label className="checkbox-row disabled">
                <input type="checkbox" checked={false} disabled />
                <span>Trident Backbone (Clee — 3mm Steel)</span>
              </label>
              <div className="hint-line warn">
                Designed for fixed-gantry Voron Trident (replaces 3 bottom Z-motor mounts);
                unavailable on flying gantry.
              </div>
            </>
          ) : (
            <>
              <label className="checkbox-row">
                <input
                  type="checkbox"
                  checked={tridentBackboneEnabled}
                  onChange={(e) => {
                    const nextVal = e.target.checked;
                    update(
                      (d) => {
                        d.frame.trident_backbone = nextVal;
                      },
                      nextVal
                        ? "Trident Backbone (Clee): Off → Installed (3mm Steel)"
                        : "Trident Backbone (Clee): Installed → Off"
                    );
                  }}
                />
                <span>Trident Backbone (Clee — 3mm Mild Steel)</span>
              </label>
              {tridentBackboneEnabled && (
                <div className="inline-readout-strip">
                  <strong>
                    +{(simBackbone && simBackbone.mass_kg > 0 ? simBackbone.mass_kg : 4.6).toFixed(2)} kg steel
                  </strong>
                  <span>·</span>
                  <span>
                    {(simBackbone && simBackbone.thickness_mm > 0 ? simBackbone.thickness_mm : 3).toFixed(0)}mm solid plate
                  </span>
                  {sim?.static_fea?.bed_deflection_um != null && (
                    <>
                      <span>·</span>
                      <span>Bed sag {sim.static_fea.bed_deflection_um.toFixed(1)} µm</span>
                    </>
                  )}
                  <span>·</span>
                  <span style={{ color: "var(--accent)" }}>
                    {(simBackbone && simBackbone.z_thrust_boost_factor > 0 ? simBackbone.z_thrust_boost_factor : 2.4).toFixed(1)}× Z-mount
                  </span>
                  <span>·</span>
                  <span>
                    +${(simBackbone && simBackbone.cost_usd > 0 ? simBackbone.cost_usd : 50).toFixed(0)}
                  </span>
                </div>
              )}
              {showHints && (
                <div className="hint-line">
                  West3D / Clee 3mm laser-cut powder-coated mild steel solid full-footprint
                  deck plate unifying all 3 bottom Z-motor mounts across the lower frame.
                </div>
              )}
            </>
          )}
        </div>
      )}

      <details className="sub-accordion">
        <summary>
          <span>Advanced Frame (Joints &amp; Cavity Fill)</span>
          <span className="sub-accordion-tag">
            {CORNER_JOINT_OPTIONS.find((o) => o.id === currentCornerJoint)?.shortLabel ??
              currentCornerJoint}{" "}
            · {selectedFilling.short_name}
          </span>
        </summary>
        <div className="sub-accordion-body">
          <div className="field-row">
            <label>Corner Joint Type</label>
            <select
              value={currentCornerJoint}
              onChange={(e) => {
                const nextVal = e.target.value as CornerJointType;
                const prevLbl =
                  CORNER_JOINT_OPTIONS.find((o) => o.id === currentCornerJoint)?.shortLabel ??
                  currentCornerJoint;
                const nextLbl =
                  CORNER_JOINT_OPTIONS.find((o) => o.id === nextVal)?.shortLabel ?? nextVal;
                update(
                  (d) => {
                    d.frame.corner_joint_type = nextVal;
                  },
                  `Corner Joint: ${prevLbl} → ${nextLbl}`
                );
              }}
            >
              {CORNER_JOINT_OPTIONS.map((opt) => (
                <option key={opt.id} value={opt.id}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>

          <div className="field-row">
            <label>
              <span>Extrusion Cavity Filling</span>
              <span
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: 4,
                  color: currentFillingId !== "none" ? "var(--accent)" : "var(--muted)",
                  fontWeight: 600,
                }}
              >
                <span
                  className="swatch-dot"
                  style={{ background: selectedFilling.color_hex }}
                />
                {selectedFilling.short_name}
              </span>
            </label>
            <select
              value={currentFillingId}
              onChange={(e) => {
                const nextId = e.target.value as ExtrusionFillingType;
                const nextItem =
                  extrusionFillings.find((f) => f.id === nextId) ?? extrusionFillings[0];
                update(
                  (d) => {
                    d.frame.extrusion_filling = nextId;
                  },
                  `Extrusion Fill: ${selectedFilling.short_name} → ${nextItem.short_name}`
                );
              }}
            >
              {extrusionFillings.map((f) => (
                <option key={f.id} value={f.id}>
                  {f.name}
                </option>
              ))}
            </select>
            <div className="inline-readout-strip">
              {currentFillingId === "none" ? (
                <>
                  <span>Hollow extrusion</span>
                  <span>·</span>
                  <span>ζ = {fillingDampingRatio.toFixed(3)}</span>
                  <span>·</span>
                  <span>Base stiffness</span>
                </>
              ) : (
                <>
                  <strong>+{fillingAddedMassKg.toFixed(2)} kg fill</strong>
                  <span>·</span>
                  <span>ζ = {fillingDampingRatio.toFixed(3)}</span>
                  <span>·</span>
                  <span
                    style={{
                      color: selectedFilling.bonded ? "var(--accent)" : "var(--green)",
                    }}
                  >
                    {selectedFilling.bonded ? "Bonded +EI" : "Granular Damping"}
                  </span>
                </>
              )}
            </div>
            {showHints && (
              <div className="hint-line">
                Fillings apply to stationary frame extrusions only (never the moving X-beam).
                Epoxy granite &amp; UHPC bond to cavity walls (+EI + damping); sand &amp;
                lead shot damp resonance without raising moving mass.
              </div>
            )}
          </div>
        </div>
      </details>

      <div className="sub-block">
        <div className="sub-block-title">
          <span>Structural &amp; Enclosure Panels</span>
          {panelsEnabled && (
            <span className="sub-block-tag">
              {currentPanelThickness}mm {panelMaterialShort(currentPanelMaterial)}
              {spec.architecture === "corexy" && !topPanelInstalled ? " · No Top" : ""}
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

        {spec.architecture === "corexy" && panelsEnabled && (
          <label className="checkbox-row" style={{ marginTop: 4 }}>
            <input
              type="checkbox"
              checked={topPanelInstalled}
              onChange={(e) => {
                const nextVal = e.target.checked;
                update(
                  (d) => {
                    d.frame.top_panel = nextVal;
                  },
                  nextVal
                    ? "Top Panel: Open Top → Installed"
                    : "Top Panel: Installed → Open Top"
                );
              }}
            />
            <span>Include Top Panel</span>
          </label>
        )}

        {spec.architecture === "corexy" &&
          panelsEnabled &&
          effectivePanelMounting === "bolted_structural" && (
            <div className="gantry-bolt-block">
              {isFlyingGantry ? (
                <>
                  <label className="checkbox-row disabled">
                    <input type="checkbox" checked={false} disabled />
                    <span>Bolt Panels Directly to Fixed Gantry Beams</span>
                  </label>
                  <div className="hint-line warn">
                    Unavailable on Flying Gantry (Voron 2.4) — the XY gantry moves up/down in Z, so
                    panels can only bolt to the outer cube.
                  </div>
                </>
              ) : (
                <>
                  <label className="checkbox-row">
                    <input
                      type="checkbox"
                      checked={boltPanelsToGantry}
                      onChange={(e) => {
                        const nextVal = e.target.checked;
                        update(
                          (d) => {
                            d.frame.bolt_panels_to_gantry = nextVal;
                          },
                          nextVal
                            ? "Panel Bolts: Outer Frame Only → Bolted to Fixed Gantry"
                            : "Panel Bolts: Bolted to Fixed Gantry → Outer Frame Only"
                        );
                      }}
                    />
                    <span>Bolt Panels Directly to Fixed Gantry Beams</span>
                  </label>
                  {showHints && (
                    <div className="hint-line">
                      Ties side &amp; rear panels straight into the fixed Y-rail and rear motor
                      extrusions for a major stiffness &amp; resonance boost.
                    </div>
                  )}
                </>
              )}
            </div>
          )}

        <div className="slider-row" style={{ marginTop: 4 }}>
          <label>
            <span>Chamber Temp (°C)</span>
            <span className="tension-readout">
              <strong style={{ color: chamberTempDraft >= 55 ? "var(--amber)" : "var(--text)" }}>
                {chamberTempDraft} °C
              </strong>
              <span className="kpi-sub">
                {chamberTempDraft <= 25 ? "ambient" : `ΔT +${chamberTempDraft - 25} K`}
              </span>
            </span>
          </label>
          <div className="tension-control-row">
            <input
              type="range"
              min={25}
              max={85}
              step={5}
              value={Math.max(25, Math.min(85, chamberTempDraft))}
              onChange={(e) => handleChamberTempSliderChange(Number(e.target.value))}
              onPointerUp={() => commitChamberTemp(chamberTempDraft)}
              onKeyUp={() => commitChamberTemp(chamberTempDraft)}
            />
            <input
              type="number"
              className="tension-num-input"
              min={25}
              max={85}
              step={5}
              value={chamberTempDraft}
              onChange={(e) => {
                const v = Number(e.target.value);
                if (!Number.isFinite(v)) return;
                setChamberTempDraft(v);
                if (v >= 25 && v <= 85) commitChamberTemp(v);
              }}
              onBlur={() => commitChamberTemp(chamberTempDraft)}
            />
          </div>
          {showHints && (
            <div className="hint-line">
              Heated chambers drive aluminum-steel bimetallic gantry bow, frame Z-drift, and
              printed bracket E(T) softening near polymer Tg.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export interface DesignAnchoringSectionProps {
  spec: PrinterSpecData;
  catalog: CatalogResponse;
  showHints: boolean;
  toolheadRelPos: [number, number, number];
  onToolheadRelPosChange: (pos: [number, number, number], changeLabel?: string) => void;
  panelMeshN: number;
  onPanelMeshNChange: (n: number) => void;
  update: (mutator: (draft: PrinterSpecData) => void, changeLabel?: string) => void;
}

export const DesignAnchoringSection: React.FC<DesignAnchoringSectionProps> = ({
  spec,
  catalog,
  showHints,
  toolheadRelPos,
  onToolheadRelPosChange,
  panelMeshN,
  onPanelMeshNChange,
  update,
}) => {
  const [posDraft, setPosDraft] = useState<[number, number, number]>(toolheadRelPos);
  const posTimerRef = useRef<number | null>(null);

  useEffect(() => {
    setPosDraft(toolheadRelPos);
  }, [toolheadRelPos[0], toolheadRelPos[1], toolheadRelPos[2]]);

  useEffect(() => {
    return () => {
      if (posTimerRef.current != null) window.clearTimeout(posTimerRef.current);
    };
  }, []);

  const currentFeetType: FrameFeetType = spec.frame.feet_type ?? "rigid_bolted";
  const frameFeetCatalogEntry = catalog.frame_feet?.find((f) => f.id === currentFeetType);
  const frameFeetOptionMeta =
    FRAME_FEET_OPTIONS.find((o) => o.id === currentFeetType) ?? FRAME_FEET_OPTIONS[0];

  const commitToolheadRelPos = (nextPos: [number, number, number], label?: string) => {
    if (posTimerRef.current != null) {
      window.clearTimeout(posTimerRef.current);
      posTimerRef.current = null;
    }
    if (
      Math.abs(nextPos[0] - toolheadRelPos[0]) < 0.005 &&
      Math.abs(nextPos[1] - toolheadRelPos[1]) < 0.005 &&
      Math.abs(nextPos[2] - toolheadRelPos[2]) < 0.005
    ) {
      return;
    }
    onToolheadRelPosChange(nextPos, label);
  };

  const handlePosAxisSlider = (axisIdx: 0 | 1 | 2, pctVal: number) => {
    const frac = Math.max(0.1, Math.min(0.9, Math.round(pctVal) / 100));
    const next: [number, number, number] = [posDraft[0], posDraft[1], posDraft[2]];
    next[axisIdx] = frac;
    setPosDraft(next);
    if (posTimerRef.current != null) window.clearTimeout(posTimerRef.current);
    posTimerRef.current = window.setTimeout(() => commitToolheadRelPos(next), 240);
  };

  return (
    <div className="section-card">
      <details className="sub-accordion">
        <summary>
          <span>Simulation &amp; Anchoring</span>
          <span className="sub-accordion-tag">
            {frameFeetOptionMeta.shortLabel} · {panelMeshN}×{panelMeshN}
          </span>
        </summary>
        <div className="sub-accordion-body">
          <div className="field-row">
            <label>
              <span>Benchtop Feet</span>
              <span
                style={{
                  color: currentFeetType === "rigid_bolted" ? "var(--muted)" : "var(--accent)",
                  fontWeight: 600,
                }}
              >
                {frameFeetOptionMeta.shortLabel}
              </span>
            </label>
            <select
              value={currentFeetType}
              onChange={(e) => {
                const nextVal = e.target.value as FrameFeetType;
                const prevLbl =
                  FRAME_FEET_OPTIONS.find((o) => o.id === currentFeetType)?.shortLabel ??
                  currentFeetType;
                const nextLbl =
                  FRAME_FEET_OPTIONS.find((o) => o.id === nextVal)?.shortLabel ?? nextVal;
                update(
                  (d) => {
                    d.frame.feet_type = nextVal;
                  },
                  `Benchtop Feet: ${prevLbl} → ${nextLbl}`
                );
              }}
            >
              {FRAME_FEET_OPTIONS.map((opt) => {
                const catItem = catalog.frame_feet?.find((f) => f.id === opt.id);
                return (
                  <option key={opt.id} value={opt.id}>
                    {catItem?.name ?? opt.label}
                  </option>
                );
              })}
            </select>
            {showHints && (
              <div className="hint-line">
                {frameFeetCatalogEntry?.notes || frameFeetOptionMeta.hint}
              </div>
            )}
          </div>

          <div className="sub-block">
            <div className="sub-block-title">
              <span>Test Position (X, Y, Z)</span>
              <span className="sub-block-tag">
                {Math.round(posDraft[0] * 100)} / {Math.round(posDraft[1] * 100)} /{" "}
                {Math.round(posDraft[2] * 100)}%
              </span>
            </div>

            <div className="chip-group">
              {TEST_POS_PRESETS.map((preset) => {
                const isActive =
                  Math.abs(posDraft[0] - preset.pos[0]) < 0.03 &&
                  Math.abs(posDraft[1] - preset.pos[1]) < 0.03 &&
                  Math.abs(posDraft[2] - preset.pos[2]) < 0.03;
                return (
                  <button
                    key={preset.id}
                    type="button"
                    className={`chip-btn ${isActive ? "active" : ""}`}
                    onClick={() => {
                      setPosDraft(preset.pos);
                      commitToolheadRelPos(preset.pos, `Test Position: ${preset.label}`);
                    }}
                  >
                    {preset.label}
                  </button>
                );
              })}
            </div>

            <div className="pos-slider-grid">
              {(
                [
                  [0, "X"],
                  [1, "Y"],
                  [2, "Z"],
                ] as Array<[0 | 1 | 2, string]>
              ).map(([idx, axisName]) => {
                const pct = Math.round(posDraft[idx] * 100);
                return (
                  <div key={axisName} className="pos-slider-row">
                    <span className="pos-axis-lbl">{axisName}</span>
                    <input
                      type="range"
                      min={10}
                      max={90}
                      step={5}
                      value={pct}
                      onChange={(e) => handlePosAxisSlider(idx, Number(e.target.value))}
                      onPointerUp={() => commitToolheadRelPos(posDraft)}
                      onKeyUp={() => commitToolheadRelPos(posDraft)}
                    />
                    <span className="pos-axis-val">{pct}%</span>
                  </div>
                );
              })}
            </div>

            {showHints && (
              <div className="hint-line">
                Probe corners or Top/Low Z to reveal cantilever bed droop, Z-column flex, or
                asymmetric gantry modes.
              </div>
            )}
          </div>

          <div className="field-row">
            <label>Panel Shell Mesh (MITC4)</label>
            <div
              className="engine-toggle engine-toggle-4"
              style={{ gridTemplateColumns: "repeat(4, minmax(0, 1fr))" }}
            >
              {(
                [
                  [2, "2×2 Fast"],
                  [3, "3×3 Balanced"],
                  [4, "4×4 Med"],
                  [8, "8×8 Max"],
                ] as Array<[number, string]>
              ).map(([n, label]) => (
                <button
                  key={n}
                  type="button"
                  className={`engine-toggle-btn ${panelMeshN === n ? "active" : ""}`}
                  onClick={() => onPanelMeshNChange(n)}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
        </div>
      </details>
    </div>
  );
};
