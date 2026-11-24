import React, { useEffect, useRef, useState } from "react";
import {
  BedMountType,
  CatalogResponse,
  PrinterSpecData,
  ZLeadscrewType,
  ZRailProfileType,
} from "../types";
import { BED_MOUNT_STANDOFF_HEIGHT_MM } from "../viewport/bedMounts";

const FALLBACK_Z_RAIL_PROFILES: Array<{ id: ZRailProfileType; name: string }> = [
  { id: "mgn12_rail", name: "MGN12H Linear Rail (12×8mm)" },
  { id: "mgn9_rail", name: "MGN9H Linear Rail (9×6.5mm)" },
  { id: "steel_rod_8mm", name: "8mm Smooth Steel Rods" },
  { id: "steel_rod_10mm", name: "10mm Smooth Steel Rods" },
  { id: "steel_rod_12mm", name: "12mm Smooth Steel Rods" },
  { id: "none", name: "None (Leadscrew Guided)" },
];

const FALLBACK_Z_DRIVE_MECHANISMS: Array<{ id: ZLeadscrewType; name: string }> = [
  { id: "auto", name: "Auto (Match Z Architecture)" },
  { id: "tr8x4_pom", name: "TR8×4 Trident POM Nut" },
  { id: "tr8x8_pom", name: "TR8×8 POM Nut" },
  { id: "tr8x8_brass", name: "TR8×8 Brass Nut" },
  { id: "sfu1204_bk10", name: "SFU1204 C7 Ballscrew + BK10" },
  { id: "gt2_6mm_z_belt", name: "GT2 6mm Z-Belt" },
  { id: "gt2_9mm_z_belt", name: "GT2 9mm Z-Belt" },
];

const BED_MOUNT_OPTIONS: Array<{
  id: BedMountType;
  label: string;
  shortLabel: string;
  hint: string;
  points: 3 | 4;
  tapRated: boolean;
  tapLabel: string;
  tapTitle: string;
  tacoLabel: string;
  tacoTone: "ok" | "info" | "danger";
  tacoTitle: string;
}> = [
  {
    id: "kinematic_3pt",
    label: "3-Pt Kinematic (Maxwell)",
    shortLabel: "3-Pt Kinematic",
    hint: "Zero thermal taco buckling at high bed temps with high axial/lateral stiffness.",
    points: 3,
    tapRated: false,
    tapLabel: "Magnetic — Not Tap-Rated",
    tapTitle: "Magnetic preload can lift or shift under Voron Tap nozzle probing forces",
    tacoLabel: "Zero Taco",
    tacoTone: "ok",
    tacoTitle: "Exact 6-DOF Maxwell radial V-grooves allow unconstrained radial thermal expansion",
  },
  {
    id: "trident_3pt_standoffs",
    label: "3-Pt Trident Bakelite Standoffs + GE5C",
    shortLabel: "Trident 3-Pt Bakelite",
    hint: "Stock Voron/Siboor Trident 3-point Bakelite thermal standoffs on 2020 T-subframe with GE5C spherical Z-joints.",
    points: 3,
    tapRated: true,
    tapLabel: "Tap-Rated ✓",
    tapTitle: "Hard-bolted phenolic Bakelite columns stay rigid under Voron Tap probing",
    tacoLabel: "Low Slotted Relief",
    tacoTone: "info",
    tacoTitle: "Rear-center fixed anchor with two front slotted thermal-relief washers",
  },
  {
    id: "whopping_pochard_kinematic",
    label: "Whopping Pochard Leaf-Spring (Tap-Rated)",
    shortLabel: "Whopping Pochard",
    hint: "Zero-play kinematic leaf-spring mount — eliminates thermal taco while staying rigid under Voron Tap probing.",
    points: 3,
    tapRated: true,
    tapLabel: "Tap-Rated ✓",
    tapTitle: "Bolted 301 spring-steel leaf flexures resist vertical Tap probing with zero play",
    tacoLabel: "Zero Taco",
    tacoTone: "ok",
    tacoTitle: "Radial leaf-spring flexures absorb bed thermal expansion without taco buckling",
  },
  {
    id: "rigid_4pt",
    label: "4-Pt Rigid (Hard Spacers)",
    shortLabel: "4-Pt Rigid",
    hint: "Stiffest mechanically, but over-constrains thermal expansion and tacos above 80°C.",
    points: 4,
    tapRated: true,
    tapLabel: "Tap-Rated ✓",
    tapTitle: "Solid metal hex standoffs provide highest axial probing rigidity",
    tacoLabel: "High Thermal Taco Risk",
    tacoTone: "danger",
    tacoTitle: "Over-constrained 4-corner hard bolting buckles the bed plate above 80°C",
  },
  {
    id: "floating_4pt",
    label: "4-Pt Floating (1 Fixed + 3 Slotted)",
    shortLabel: "4-Pt Floating",
    hint: "Stiff vertically while slotted corners relieve thermal expansion without buckling.",
    points: 4,
    tapRated: true,
    tapLabel: "Tap-Rated ✓",
    tapTitle: "Solid vertical shoulder-bolt support handles nozzle probing loads cleanly",
    tacoLabel: "Low Slotted Relief",
    tacoTone: "info",
    tacoTitle: "1 fixed locator anchor + 3 radially slotted glide tabs relieve expansion",
  },
  {
    id: "silicone_4pt",
    label: "4-Pt Silicone Spacers",
    shortLabel: "4-Pt Silicone",
    hint: "Thermally decoupled with moderate damping; softer than kinematic/rigid mounts.",
    points: 4,
    tapRated: false,
    tapLabel: "Compliant — Not Tap-Rated",
    tapTitle: "70A elastomer compresses under Tap probing force, degrading Z-offset repeatability",
    tacoLabel: "Low Compliant Relief",
    tacoTone: "info",
    tacoTitle: "Compliant silicone columns shear laterally to relieve thermal expansion",
  },
  {
    id: "springs_4pt",
    label: "4-Pt Bed Springs (Yellow)",
    shortLabel: "4-Pt Springs",
    hint: "Low-cost leveling springs; softest mount with highest bed-bounce under Y accel.",
    points: 4,
    tapRated: false,
    tapLabel: "Compliant — Not Tap-Rated",
    tapTitle: "Low axial spring stiffness deflects heavily under nozzle probing forces",
    tacoLabel: "Low Compliant Relief",
    tacoTone: "info",
    tacoTitle: "Compliant coil springs absorb radial expansion with minimal plate buckling",
  },
];

const PLATE_MATERIAL_FALLBACKS: Array<{ id: string; name: string }> = [
  { id: "mic6_aluminum", name: "MIC-6 Cast Aluminum Plate" },
  { id: "atp5_aluminum", name: "ATP-5 Cast Aluminum Plate" },
  { id: "aluminum_6061_t6", name: "Aluminum 6061-T6" },
  { id: "aluminum_7075_t6", name: "Aluminum 7075-T6" },
  { id: "steel_mild", name: "Mild Steel (A36)" },
  { id: "carbon_fiber_quasi", name: "Carbon Fiber (Quasi-Isotropic)" },
];

const PLATE_MATERIAL_IDS = new Set([
  "mic6_aluminum",
  "atp5_aluminum",
  "aluminum_6061_t6",
  "aluminum_7075_t6",
  "aluminum_5052_sheet",
  "steel_mild",
  "steel_4140",
  "stainless_304",
  "titanium_grade5",
  "carbon_fiber_quasi",
  "carbon_fiber_plate",
  "brass_c360",
  "copper_c110",
]);

function bedZTopologyMeta(arch: PrinterSpecData["architecture"], zDrive: string): {
  badge: string;
  caption: string;
} {
  if (arch === "cartesian_bedslinger") {
    return { badge: "Moving Y-Bed", caption: "Y-axis bed carriage on linear rails/rods" };
  }
  if (arch === "delta") {
    return { badge: "Fixed Delta Base", caption: "Stationary circular plate anchored to bottom triangle" };
  }
  if (zDrive === "belted_quad_z") {
    return { badge: "Fixed Base Bed", caption: "Voron 2.4 style — bed fixed to base extrusions, flying XY gantry" };
  }
  if (zDrive === "leadscrew_triple") {
    return { badge: "3-Pt Trident Z", caption: "3 independent Z leadscrews (front-left, front-right, rear-center)" };
  }
  if (zDrive === "leadscrew_dual") {
    return { badge: "2-Pt Left/Right Z", caption: "Dual left/right Z leadscrews & linear guides" };
  }
  if (zDrive === "leadscrew_dual_belt_sync") {
    return {
      badge: "Belt-Synced Dual-Z",
      caption: "VzBot style — dual Z leadscrews mechanically synced via closed-loop timing belt",
    };
  }
  if (zDrive === "leadscrew_single") {
    return { badge: "Cantilever Rear-Z", caption: "Single rear Z rail/screw — test front/low Z for cantilever droop" };
  }
  return { badge: zDrive, caption: "Z-axis bed support" };
}

export interface DesignBedZSectionProps {
  spec: PrinterSpecData;
  catalog: CatalogResponse;
  showHints: boolean;
  update: (mutator: (draft: PrinterSpecData) => void, changeLabel?: string) => void;
  updatePositiveNum: (
    raw: string,
    mutator: (draft: PrinterSpecData, val: number) => void,
    changeLabelBuilder: (val: number) => string
  ) => void;
  materialLabel: (id: string) => string;
}

export const DesignBedZSection: React.FC<DesignBedZSectionProps> = ({
  spec,
  catalog,
  showHints,
  update,
  updatePositiveNum,
  materialLabel,
}) => {
  const [bedTempDraft, setBedTempDraft] = useState<number>(
    Math.round(spec.bed.target_temp_c ?? 60)
  );
  const bedTempTimerRef = useRef<number | null>(null);

  const [bedWidthDraft, setBedWidthDraft] = useState<string>(
    String(Math.round(spec.bed.size_mm[0]))
  );
  const [bedDepthDraft, setBedDepthDraft] = useState<string>(
    String(Math.round(spec.bed.size_mm[1]))
  );
  const [bedHeaterDraft, setBedHeaterDraft] = useState<string>(
    String(Math.round(spec.bed.heater_power_w))
  );

  useEffect(() => {
    setBedTempDraft(Math.round(spec.bed.target_temp_c ?? 60));
  }, [spec.bed.target_temp_c]);

  useEffect(() => {
    setBedWidthDraft(String(Math.round(spec.bed.size_mm[0])));
  }, [spec.bed.size_mm[0]]);

  useEffect(() => {
    setBedDepthDraft(String(Math.round(spec.bed.size_mm[1])));
  }, [spec.bed.size_mm[1]]);

  useEffect(() => {
    setBedHeaterDraft(String(Math.round(spec.bed.heater_power_w)));
  }, [spec.bed.heater_power_w]);

  useEffect(() => {
    return () => {
      if (bedTempTimerRef.current != null) window.clearTimeout(bedTempTimerRef.current);
    };
  }, []);

  const commitBedSizeAxis = (axis: 0 | 1, raw: string) => {
    const current = Math.round(spec.bed.size_mm[axis]);
    const parsed = Number(raw);
    const setDraft = axis === 0 ? setBedWidthDraft : setBedDepthDraft;
    if (raw.trim() === "" || !Number.isFinite(parsed)) {
      setDraft(String(current));
      return;
    }
    const clamped = Math.max(80, Math.min(1300, Math.round(parsed)));
    setDraft(String(clamped));
    if (clamped === current) return;
    const axisLabel = axis === 0 ? "Bed Width" : "Bed Depth";
    update(
      (d) => {
        const nextSize: [number, number] = [d.bed.size_mm[0], d.bed.size_mm[1]];
        nextSize[axis] = clamped;
        d.bed.size_mm = nextSize;
      },
      `${axisLabel}: ${current} → ${clamped} mm`
    );
  };

  const commitBedHeaterW = (raw: string) => {
    const current = Math.round(spec.bed.heater_power_w);
    const parsed = Number(raw);
    if (raw.trim() === "" || !Number.isFinite(parsed)) {
      setBedHeaterDraft(String(current));
      return;
    }
    const clamped = Math.max(50, Math.min(3000, Math.round(parsed)));
    setBedHeaterDraft(String(clamped));
    if (clamped === current) return;
    update(
      (d) => {
        d.bed.heater_power_w = clamped;
      },
      `Bed Heater: ${current}W → ${clamped}W`
    );
  };

  const specBedTempC = Math.round(spec.bed.target_temp_c ?? 60);
  const commitBedTemp = (rawVal: number) => {
    const val = Math.max(25, Math.min(130, Math.round(rawVal)));
    if (bedTempTimerRef.current != null) {
      window.clearTimeout(bedTempTimerRef.current);
      bedTempTimerRef.current = null;
    }
    if (val === specBedTempC) return;
    update(
      (d) => {
        d.bed.target_temp_c = val;
      },
      `Bed Temp: ${specBedTempC}°C → ${val}°C`
    );
  };

  const handleBedTempSliderChange = (nextVal: number) => {
    setBedTempDraft(nextVal);
    if (bedTempTimerRef.current != null) window.clearTimeout(bedTempTimerRef.current);
    bedTempTimerRef.current = window.setTimeout(() => commitBedTemp(nextVal), 220);
  };

  const currentBedMount: BedMountType = spec.bed.mount_type ?? "kinematic_3pt";
  const bedMountCatalogEntry = catalog.bed_mounts?.find((m) => m.id === currentBedMount);
  const bedMountOptionMeta =
    BED_MOUNT_OPTIONS.find((o) => o.id === currentBedMount) ?? BED_MOUNT_OPTIONS[0];

  const plateMaterialOptions = (() => {
    const filtered = catalog.materials
      .filter((m) => PLATE_MATERIAL_IDS.has(m.id) || m.id === spec.bed.plate_material)
      .map((m) => ({ id: m.id, name: m.name }));
    const existingIds = new Set(filtered.map((m) => m.id));
    for (const fb of PLATE_MATERIAL_FALLBACKS) {
      if (!existingIds.has(fb.id)) {
        filtered.push(fb);
        existingIds.add(fb.id);
      }
    }
    if (!existingIds.has(spec.bed.plate_material)) {
      filtered.unshift({ id: spec.bed.plate_material, name: spec.bed.plate_material });
    }
    return filtered;
  })();

  const zTopoMeta = bedZTopologyMeta(spec.architecture, spec.bed.z_drive);

  const zRailProfileOptions =
    catalog.z_rail_profiles && catalog.z_rail_profiles.length > 0
      ? catalog.z_rail_profiles
      : FALLBACK_Z_RAIL_PROFILES;
  const currentZRailProfile: ZRailProfileType = spec.bed.z_rail_profile ?? "mgn12_rail";

  const zDriveMechanismOptions: Array<{ id: string; name: string }> =
    catalog.z_drive_mechanisms && catalog.z_drive_mechanisms.length > 0
      ? [
          { id: "auto", name: "Auto (Match Z Architecture)" },
          ...catalog.z_drive_mechanisms.map((m) => ({ id: m.id, name: m.name })),
        ]
      : FALLBACK_Z_DRIVE_MECHANISMS;
  const currentZLeadscrewType: ZLeadscrewType = spec.bed.z_leadscrew_type ?? "auto";

  return (
    <div className="section-card">
      <div className="section-title">
        <span>Build Plate &amp; Bed Mount</span>
        <span className="sub-block-tag">{bedTempDraft}°C</span>
      </div>

      <div className="slider-row">
        <label>
          <span>Bed Temp (°C)</span>
          <span className="tension-readout">
            <strong style={{ color: bedTempDraft >= 100 ? "var(--amber)" : "var(--text)" }}>
              {bedTempDraft} °C
            </strong>
          </span>
        </label>
        <div className="tension-control-row">
          <input
            type="range"
            min={25}
            max={130}
            step={5}
            value={Math.max(25, Math.min(130, bedTempDraft))}
            onChange={(e) => handleBedTempSliderChange(Number(e.target.value))}
            onPointerUp={() => commitBedTemp(bedTempDraft)}
            onKeyUp={() => commitBedTemp(bedTempDraft)}
          />
          <input
            type="number"
            className="tension-num-input"
            min={25}
            max={130}
            step={5}
            value={bedTempDraft}
            onChange={(e) => {
              const v = Number(e.target.value);
              if (!Number.isFinite(v)) return;
              setBedTempDraft(v);
              if (v >= 25 && v <= 130) commitBedTemp(v);
            }}
            onBlur={() => commitBedTemp(bedTempDraft)}
          />
        </div>
      </div>

      <div className="field-row">
        <label>
          <span>Bed Mount</span>
          {bedMountCatalogEntry?.k_axial_n_per_mm != null && (
            <span style={{ fontSize: 10, color: "var(--muted)" }}>
              k_z {Math.round(bedMountCatalogEntry.k_axial_n_per_mm)} N/mm
            </span>
          )}
        </label>
        <select
          value={currentBedMount}
          onChange={(e) => {
            const nextVal = e.target.value as BedMountType;
            const prevLbl =
              BED_MOUNT_OPTIONS.find((o) => o.id === currentBedMount)?.shortLabel ??
              currentBedMount;
            const nextLbl =
              BED_MOUNT_OPTIONS.find((o) => o.id === nextVal)?.shortLabel ?? nextVal;
            update(
              (d) => {
                d.bed.mount_type = nextVal;
              },
              `Bed Mount: ${prevLbl} → ${nextLbl}`
            );
          }}
        >
          {BED_MOUNT_OPTIONS.map((opt) => {
            const catItem = catalog.bed_mounts?.find((m) => m.id === opt.id);
            return (
              <option key={opt.id} value={opt.id}>
                {catItem?.name ?? opt.label}
              </option>
            );
          })}
        </select>
        <div className="mount-physics-strip" aria-label="Bed mount physics summary">
          <span
            className="mount-pill neutral"
            title={`${bedMountCatalogEntry?.mount_points ?? bedMountOptionMeta.points}-point bed support layout`}
          >
            {bedMountCatalogEntry?.mount_points ?? bedMountOptionMeta.points}-Pt
          </span>
          <span
            className={`mount-pill ${bedMountOptionMeta.tapRated ? "ok" : "warn"}`}
            title={bedMountOptionMeta.tapTitle}
          >
            {bedMountOptionMeta.tapLabel}
          </span>
          <span
            className={`mount-pill ${bedMountOptionMeta.tacoTone}`}
            title={bedMountOptionMeta.tacoTitle}
          >
            {bedMountOptionMeta.tacoLabel}
          </span>
          <span
            className="mount-pill neutral"
            title="Nominal standoff stack height between heated bed plate and subframe"
          >
            {BED_MOUNT_STANDOFF_HEIGHT_MM[currentBedMount]} mm
          </span>
        </div>
        {showHints && (
          <div className="hint-line">
            {bedMountCatalogEntry?.notes || bedMountOptionMeta.hint}
          </div>
        )}
      </div>

      <div className="field-grid-3">
        <div className="field-row">
          <label title="Build Plate Width (mm)">Bed W mm</label>
          <input
            type="number"
            min={80}
            max={1300}
            step={10}
            value={bedWidthDraft}
            onChange={(e) => setBedWidthDraft(e.target.value)}
            onBlur={() => commitBedSizeAxis(0, bedWidthDraft)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                commitBedSizeAxis(0, bedWidthDraft);
                e.currentTarget.blur();
              } else if (e.key === "Escape") {
                setBedWidthDraft(String(Math.round(spec.bed.size_mm[0])));
                e.currentTarget.blur();
              }
            }}
          />
        </div>
        <div className="field-row">
          <label title="Build Plate Depth (mm)">Bed D mm</label>
          <input
            type="number"
            min={80}
            max={1300}
            step={10}
            value={bedDepthDraft}
            onChange={(e) => setBedDepthDraft(e.target.value)}
            onBlur={() => commitBedSizeAxis(1, bedDepthDraft)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                commitBedSizeAxis(1, bedDepthDraft);
                e.currentTarget.blur();
              } else if (e.key === "Escape") {
                setBedDepthDraft(String(Math.round(spec.bed.size_mm[1])));
                e.currentTarget.blur();
              }
            }}
          />
        </div>
        <div className="field-row">
          <label title="Bed Heater Wattage (W)">Heater W</label>
          <input
            type="number"
            min={50}
            max={3000}
            step={25}
            value={bedHeaterDraft}
            onChange={(e) => setBedHeaterDraft(e.target.value)}
            onBlur={() => commitBedHeaterW(bedHeaterDraft)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                commitBedHeaterW(bedHeaterDraft);
                e.currentTarget.blur();
              } else if (e.key === "Escape") {
                setBedHeaterDraft(String(Math.round(spec.bed.heater_power_w)));
                e.currentTarget.blur();
              }
            }}
          />
        </div>
      </div>

      <div className="field-grid-2">
        <div className="field-row">
          <label>Plate Thickness (mm)</label>
          <input
            type="number"
            min={4}
            max={12}
            step={1}
            value={spec.bed.plate_thickness_mm}
            onChange={(e) =>
              updatePositiveNum(
                e.target.value,
                (d, v) => {
                  d.bed.plate_thickness_mm = Math.max(4, Math.min(12, v));
                },
                (v) =>
                  `Plate Thickness: ${spec.bed.plate_thickness_mm} → ${Math.max(
                    4,
                    Math.min(12, v)
                  )} mm`
              )
            }
          />
        </div>
        <div className="field-row">
          <label>Plate Material</label>
          <select
            value={spec.bed.plate_material}
            onChange={(e) => {
              const nextVal = e.target.value;
              update(
                (d) => {
                  d.bed.plate_material = nextVal;
                },
                `Plate Material: ${materialLabel(spec.bed.plate_material)} → ${materialLabel(
                  nextVal
                )}`
              );
            }}
          >
            {plateMaterialOptions.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name}
              </option>
            ))}
          </select>
        </div>
      </div>

      <details className="sub-accordion">
        <summary>
          <span>Z-Axis Hardware &amp; Bed Mass</span>
          <span className="sub-accordion-tag">
            {zTopoMeta.badge} · {spec.bed.mass_g}g
          </span>
        </summary>
        <div className="sub-accordion-body">
          <div className="inline-readout-strip" title={zTopoMeta.caption}>
            <strong style={{ color: "var(--accent)" }}>{zTopoMeta.badge}</strong>
            <span>·</span>
            <span>{zTopoMeta.caption}</span>
          </div>

          <div className="field-grid-2">
            <div className="field-row">
              <label>Z Linear Rails / Guideway</label>
              <select
                value={currentZRailProfile}
                onChange={(e) => {
                  const nextVal = e.target.value as ZRailProfileType;
                  const prevLbl =
                    zRailProfileOptions.find((o) => o.id === currentZRailProfile)?.name ??
                    currentZRailProfile;
                  const nextLbl =
                    zRailProfileOptions.find((o) => o.id === nextVal)?.name ?? nextVal;
                  update(
                    (d) => {
                      d.bed.z_rail_profile = nextVal;
                    },
                    `Z Rails: ${prevLbl} → ${nextLbl}`
                  );
                }}
              >
                {zRailProfileOptions.map((opt) => (
                  <option key={opt.id} value={opt.id}>
                    {opt.name}
                  </option>
                ))}
              </select>
            </div>
            <div className="field-row">
              <label>Z Leadscrew / Drive</label>
              <select
                value={currentZLeadscrewType}
                onChange={(e) => {
                  const nextVal = e.target.value as ZLeadscrewType;
                  const prevLbl =
                    zDriveMechanismOptions.find((o) => o.id === currentZLeadscrewType)
                      ?.name ?? currentZLeadscrewType;
                  const nextLbl =
                    zDriveMechanismOptions.find((o) => o.id === nextVal)?.name ?? nextVal;
                  update(
                    (d) => {
                      d.bed.z_leadscrew_type = nextVal;
                    },
                    `Z Drive: ${prevLbl} → ${nextLbl}`
                  );
                }}
              >
                {zDriveMechanismOptions.map((opt) => (
                  <option key={opt.id} value={opt.id}>
                    {opt.name}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="field-row">
            <label>Bed Mass (g)</label>
            <input
              type="number"
              min={800}
              max={6000}
              step={100}
              value={spec.bed.mass_g}
              onChange={(e) =>
                updatePositiveNum(
                  e.target.value,
                  (d, v) => {
                    d.bed.mass_g = Math.max(800, Math.min(6000, Math.round(v)));
                  },
                  (v) =>
                    `Bed Mass: ${spec.bed.mass_g}g → ${Math.max(
                      800,
                      Math.min(6000, Math.round(v))
                    )}g`
                )
              }
            />
          </div>
        </div>
      </details>
    </div>
  );
};
