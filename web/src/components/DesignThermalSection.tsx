import React, { useEffect, useRef, useState } from "react";
import { CPAP_BLOWER_LABELS } from "../constants/printerLabels";
import { humanizeId } from "../utils/displayText";
import {
  CatalogResponse,
  CpapBlowerType,
  HotendMountStyle,
  HotendPresetItem,
  PrinterSpecData,
  SimulationResponse,
} from "../types";
import {
  ModelOverrideControl,
  PresetAffectsStrip,
  clearModelOverrideGroup,
} from "./ModelOverrideControl";
import {
  cadHarmonicTipStiffness,
  fmtMassG,
  formatCadCmsTooltip,
  formatElemCount,
  useBakedHardwareAssets,
} from "./DesignToolheadSection";

const HEATBREAK_LABELS: Record<string, string> = {
  titanium_grade5: "Titanium Gr.5",
  stainless_304: "Stainless 304",
};

const HOTEND_MOUNT_STYLE_OPTIONS: Array<{
  id: HotendMountStyle;
  label: string;
  shortLabel: string;
}> = [
  { id: "groove_mount", label: "Groove Mount (E3D V6)", shortLabel: "Groove Mount" },
  { id: "rigid_4bolt_m3", label: "4× M3 Rigid Flange", shortLabel: "4× M3 Rigid" },
  { id: "dragon_4bolt_m25", label: "Dragon 4× M2.5 Standoffs", shortLabel: "Dragon 4× M2.5" },
  { id: "chube_6bolt_m3_18pcd", label: "Chube 6× M3 Ti Shell (18mm PCD)", shortLabel: "Chube 6× M3 Ti" },
];

const FALLBACK_HOTEND_CLAMP_N_PER_MM: Record<HotendMountStyle, number> = {
  groove_mount: 800,
  rigid_4bolt_m3: 2200,
  dragon_4bolt_m25: 3200,
  chube_6bolt_m3_18pcd: 6500,
};


const FALLBACK_HOTEND_PRESETS: HotendPresetItem[] = [
  {
    id: "e3d_v6_groove",
    name: "E3D V6 Groove Mount",
    short_name: "E3D V6",
    mount_style: "groove_mount",
    mass_g: 42,
    overall_length_mm: 62,
    heater_power_w: 40,
    block_material: "aluminum_6061_t6",
    block_mass_g: 12,
    nozzle_material: "brass_c360",
    melt_zone_length_mm: 12,
    heatbreak_material: "stainless_304",
    heatbreak_length_mm: 14,
    heatbreak_wall_mm: 0.45,
    heatbreak_id_mm: 2.0,
    heatsink_thermal_resistance_k_w: 8.5,
    structural_stiffness_n_per_mm: 45,
    max_temp_c: 285,
  },
  {
    id: "e3d_revo_voron",
    name: "E3D Revo Voron (4-Bolt M3)",
    short_name: "Revo Voron",
    mount_style: "rigid_4bolt_m3",
    mass_g: 36,
    overall_length_mm: 52,
    heater_power_w: 40,
    block_material: "brass_c360",
    block_mass_g: 8,
    nozzle_material: "brass_c360",
    melt_zone_length_mm: 13,
    heatbreak_material: "titanium_grade5",
    heatbreak_length_mm: 12,
    heatbreak_wall_mm: 0.35,
    heatbreak_id_mm: 2.0,
    heatsink_thermal_resistance_k_w: 7.2,
    structural_stiffness_n_per_mm: 240,
    max_temp_c: 300,
  },
  {
    id: "phaetus_dragon_hf",
    name: "Phaetus Dragon HF (4× M2.5 Standoffs)",
    short_name: "Dragon HF",
    mount_style: "dragon_4bolt_m25",
    mass_g: 52,
    overall_length_mm: 61,
    heater_power_w: 60,
    block_material: "copper_c110",
    block_mass_g: 16,
    nozzle_material: "brass_c360",
    melt_zone_length_mm: 17,
    heatbreak_material: "titanium_grade5",
    heatbreak_length_mm: 14,
    heatbreak_wall_mm: 0.25,
    heatbreak_id_mm: 2.0,
    heatsink_thermal_resistance_k_w: 6.8,
    structural_stiffness_n_per_mm: 520,
    max_temp_c: 500,
  },
  {
    id: "phaetus_rapido_2_uhf",
    name: "Phaetus Rapido 2 UHF (4× M2.5 Standoffs)",
    short_name: "Rapido 2 UHF",
    mount_style: "dragon_4bolt_m25",
    mass_g: 58,
    overall_length_mm: 68,
    heater_power_w: 115,
    block_material: "copper_c110",
    block_mass_g: 19,
    nozzle_material: "copper_c110",
    melt_zone_length_mm: 27,
    heatbreak_material: "titanium_grade5",
    heatbreak_length_mm: 14,
    heatbreak_wall_mm: 0.25,
    heatbreak_id_mm: 2.0,
    heatsink_thermal_resistance_k_w: 6.5,
    structural_stiffness_n_per_mm: 480,
    max_temp_c: 350,
  },
  {
    id: "vz_goliath",
    name: "VzBot Goliath Long (4× M2.5 Rigid)",
    short_name: "Goliath",
    mount_style: "dragon_4bolt_m25",
    mass_g: 74,
    overall_length_mm: 76,
    heater_power_w: 115,
    block_material: "copper_c110",
    block_mass_g: 28,
    nozzle_material: "copper_c110",
    melt_zone_length_mm: 36,
    heatbreak_material: "titanium_grade5",
    heatbreak_length_mm: 15,
    heatbreak_wall_mm: 0.3,
    heatbreak_id_mm: 2.0,
    heatsink_thermal_resistance_k_w: 5.8,
    structural_stiffness_n_per_mm: 780,
    max_temp_c: 450,
  },
  {
    id: "chube_compact",
    name: "Luke's Lab Chube Compact (80W Ti Shell)",
    short_name: "Chube Compact",
    mount_style: "rigid_4bolt_m3",
    mass_g: 63,
    overall_length_mm: 47.2,
    heater_power_w: 80,
    block_material: "copper_c110",
    block_mass_g: 18,
    nozzle_material: "copper_c110",
    melt_zone_length_mm: 33,
    heatbreak_material: "titanium_grade5",
    heatbreak_length_mm: 10,
    heatbreak_wall_mm: 0.25,
    heatbreak_id_mm: 2.0,
    heatsink_thermal_resistance_k_w: 2.6,
    structural_stiffness_n_per_mm: 2975,
    max_temp_c: 500,
  },
  {
    id: "chube_air",
    name: "Luke's Lab Chube Air (100W Ti Shell)",
    short_name: "Chube Air",
    mount_style: "chube_6bolt_m3_18pcd",
    mass_g: 79,
    overall_length_mm: 66,
    heater_power_w: 100,
    block_material: "copper_c110",
    block_mass_g: 26,
    nozzle_material: "copper_c110",
    melt_zone_length_mm: 42,
    heatbreak_material: "titanium_grade5",
    heatbreak_length_mm: 11,
    heatbreak_wall_mm: 0.25,
    heatbreak_id_mm: 2.0,
    heatsink_thermal_resistance_k_w: 2.5,
    structural_stiffness_n_per_mm: 560,
    max_temp_c: 500,
  },
  {
    id: "chube_conduction",
    name: "Luke's Lab Chube Conduction (120W Ti Shell)",
    short_name: "Chube Conduction",
    mount_style: "chube_6bolt_m3_18pcd",
    mass_g: 90,
    overall_length_mm: 66,
    heater_power_w: 120,
    block_material: "copper_c110",
    block_mass_g: 34,
    nozzle_material: "copper_c110",
    melt_zone_length_mm: 52,
    heatbreak_material: "titanium_grade5",
    heatbreak_length_mm: 10,
    heatbreak_wall_mm: 0.25,
    heatbreak_id_mm: 2.0,
    heatsink_thermal_resistance_k_w: 2.2,
    structural_stiffness_n_per_mm: 815,
    max_temp_c: 500,
  },
  {
    id: "tricorn_short",
    name: "Tricorn Short (120W 3×40W Inverted Ti)",
    short_name: "Tricorn Short",
    mount_style: "rigid_4bolt_m3",
    mass_g: 50,
    overall_length_mm: 54,
    heater_power_w: 120,
    block_material: "copper_c110",
    block_mass_g: 23,
    nozzle_material: "copper_c110",
    melt_zone_length_mm: 41,
    heatbreak_material: "titanium_grade5",
    heatbreak_length_mm: 13,
    heatbreak_wall_mm: 0.45,
    heatbreak_id_mm: 1.9,
    heatsink_thermal_resistance_k_w: 2.6,
    structural_stiffness_n_per_mm: 440,
    max_temp_c: 450,
  },
  {
    id: "tricorn_long",
    name: "Tricorn Long (120W 3×40W Inverted Ti)",
    short_name: "Tricorn Long",
    mount_style: "rigid_4bolt_m3",
    mass_g: 62,
    overall_length_mm: 66,
    heater_power_w: 120,
    block_material: "copper_c110",
    block_mass_g: 31.4,
    nozzle_material: "copper_c110",
    melt_zone_length_mm: 53.5,
    heatbreak_material: "titanium_grade5",
    heatbreak_length_mm: 13,
    heatbreak_wall_mm: 0.45,
    heatbreak_id_mm: 1.9,
    heatsink_thermal_resistance_k_w: 2.4,
    structural_stiffness_n_per_mm: 265,
    max_temp_c: 450,
  },
  {
    id: "monolith_50mm_hf",
    name: "Monolith 50mm+V6 HF Hotend (80W PT1000)",
    short_name: "Monolith 50mm HF",
    mount_style: "rigid_4bolt_m3",
    mass_g: 52,
    overall_length_mm: 68,
    heater_power_w: 80,
    block_material: "copper_c110",
    block_mass_g: 22,
    nozzle_material: "copper_c110",
    melt_zone_length_mm: 50,
    heatbreak_material: "titanium_grade5",
    heatbreak_length_mm: 10.5,
    heatbreak_wall_mm: 0.25,
    heatbreak_id_mm: 2.0,
    heatsink_thermal_resistance_k_w: 2.4,
    structural_stiffness_n_per_mm: 920,
    max_temp_c: 350,
  },
  {
    id: "trianglelab_dragon_ace_volcano",
    name: "Trianglelab Dragon Ace Volcano (70W Cu/Ti)",
    short_name: "Dragon Ace Volcano",
    mount_style: "dragon_4bolt_m25",
    mass_g: 45,
    overall_length_mm: 59.4,
    heater_power_w: 70,
    block_material: "copper_c110",
    block_mass_g: 22,
    nozzle_material: "copper_c110",
    melt_zone_length_mm: 34,
    heatbreak_material: "titanium_grade5",
    heatbreak_length_mm: 11,
    heatbreak_wall_mm: 0.45,
    heatbreak_id_mm: 1.9,
    heatsink_thermal_resistance_k_w: 3.1,
    structural_stiffness_n_per_mm: 842,
    max_temp_c: 450,
  },
];


function hotendStiffnessTier(k: number): { label: string; color: string } {
  if (k < 100) return { label: "Floppy", color: "var(--red)" };
  if (k < 350) return { label: "Moderate", color: "var(--amber)" };
  if (k < 1200) return { label: "Rigid", color: "var(--accent)" };
  return { label: "Ultra-Rigid", color: "var(--green)" };
}


export interface DesignThermalSectionProps {
  spec: PrinterSpecData;
  catalog: CatalogResponse;
  sim: SimulationResponse | null;
  showHints: boolean;
  advancedMode: boolean;
  update: (mutator: (draft: PrinterSpecData) => void, changeLabel?: string) => void;
  updatePositiveNum: (
    raw: string,
    mutator: (draft: PrinterSpecData, val: number) => void,
    changeLabelBuilder: (val: number) => string
  ) => void;
  materialLabel: (id: string) => string;
}

export const DesignThermalSection: React.FC<DesignThermalSectionProps> = ({
  spec,
  catalog,
  sim,
  showHints,
  advancedMode,
  update,
  updatePositiveNum,
  materialLabel,
}) => {
  const bakedAssets = useBakedHardwareAssets();

  const [hotendStiffnessDraft, setHotendStiffnessDraft] = useState<number>(
    Math.round(spec.toolhead.hotend.structural_stiffness_n_per_mm ?? 450)
  );
  const hotendStiffnessTimerRef = useRef<number | null>(null);

  const [hotendMassDraft, setHotendMassDraft] = useState<string>(
    String(Math.round(spec.toolhead.hotend.mass_g ?? 45))
  );
  const hotendMassTimerRef = useRef<number | null>(null);

  useEffect(() => {
    setHotendStiffnessDraft(
      Math.round(spec.toolhead.hotend.structural_stiffness_n_per_mm ?? 450)
    );
  }, [spec.toolhead.hotend.structural_stiffness_n_per_mm]);

  useEffect(() => {
    setHotendMassDraft(String(Math.round(spec.toolhead.hotend.mass_g ?? 45)));
  }, [spec.toolhead.hotend.mass_g]);

  useEffect(() => {
    return () => {
      if (hotendStiffnessTimerRef.current != null)
        window.clearTimeout(hotendStiffnessTimerRef.current);
      if (hotendMassTimerRef.current != null) window.clearTimeout(hotendMassTimerRef.current);
    };
  }, []);

  const specHotendStiffness = Math.round(spec.toolhead.hotend.structural_stiffness_n_per_mm ?? 450);
  const commitHotendStiffness = (rawVal: number) => {
    const val = Math.max(25, Math.min(10000, Math.round(rawVal)));
    if (hotendStiffnessTimerRef.current != null) {
      window.clearTimeout(hotendStiffnessTimerRef.current);
      hotendStiffnessTimerRef.current = null;
    }
    if (val === specHotendStiffness) return;
    update(
      (d) => {
        d.toolhead.hotend.structural_stiffness_n_per_mm = val;
        d.toolhead.hotend.preset_id = null;
      },
      `Hotend Stiffness: ${specHotendStiffness} → ${val} N/mm`
    );
  };

  const handleHotendStiffnessSliderChange = (nextVal: number) => {
    setHotendStiffnessDraft(nextVal);
    if (hotendStiffnessTimerRef.current != null) {
      window.clearTimeout(hotendStiffnessTimerRef.current);
    }
    hotendStiffnessTimerRef.current = window.setTimeout(
      () => commitHotendStiffness(nextVal),
      220
    );
  };

  const specHotendMassG = Math.round(spec.toolhead.hotend.mass_g ?? 45);

  const commitHotendMass = (rawVal: number) => {
    const val = Math.max(5, Math.min(400, Math.round(rawVal)));
    if (hotendMassTimerRef.current != null) {
      window.clearTimeout(hotendMassTimerRef.current);
      hotendMassTimerRef.current = null;
    }
    if (val === specHotendMassG) return;
    update(
      (d) => {
        d.toolhead.hotend.mass_g = val;
      },
      `Hotend Weight: ${specHotendMassG}g → ${val}g`
    );
  };

  const useCadSuperelements = spec.use_cad_superelements ?? true;
  const activeCadSuperelements =
    sim?.geometry?.toolhead_structure?.cad_superelements_active ?? [];
  const activeHotendCadCms = useCadSuperelements
    ? activeCadSuperelements.find((se) => se.category === "assemblies") ??
      activeCadSuperelements.find(
        (se) => se.category === "hotends" || se.category.toLowerCase().includes("hotend")
      ) ??
      null
    : null;
  const isBracedHotendCms =
    activeHotendCadCms?.category === "assemblies" ||
    Boolean(sim?.geometry?.toolhead_structure?.cad_hotend_braced);
  const useHotendBottomBrace = spec.toolhead.use_hotend_bottom_brace ?? true;
  const cadHotendBraceAvailable =
    Boolean(sim?.geometry?.toolhead_structure?.cad_hotend_brace_available) ||
    isBracedHotendCms;
  const nozzleShieldOptions =
    sim?.geometry?.toolhead_structure?.nozzle_shield_options ?? [];
  const resolvedNozzleShieldMode: "stock" | "microwaved" | "none" =
    sim?.geometry?.toolhead_structure?.nozzle_shield_mode ??
    (spec.toolhead.nozzle_shield_mode && spec.toolhead.nozzle_shield_mode !== "auto"
      ? spec.toolhead.nozzle_shield_mode
      : nozzleShieldOptions[0] ?? "stock");
  const effectiveHotendMassG =
    sim?.geometry?.toolhead_structure?.hotend_mass_g ?? specHotendMassG;
  const activeHotendCadK = cadHarmonicTipStiffness(activeHotendCadCms);
  const effectiveHotendDisplayK =
    activeHotendCadK != null ? Math.round(activeHotendCadK) : hotendStiffnessDraft;
  const effectiveHotendLenMm =
    sim?.geometry?.toolhead_structure?.hotend_overall_length_mm ??
    spec.toolhead.hotend.overall_length_mm ??
    62;

  const hotendPresets: HotendPresetItem[] =
    catalog.hotend_presets && catalog.hotend_presets.length > 0
      ? catalog.hotend_presets
      : FALLBACK_HOTEND_PRESETS;
  const hotendPresetIds = new Set(hotendPresets.map((hp) => hp.id));
  const extraBakedHotends = Object.entries(bakedAssets.hotends ?? {}).filter(
    ([assetId]) => !hotendPresetIds.has(assetId)
  );
  const currentHotendMountStyle: HotendMountStyle =
    spec.toolhead.hotend.mount_style ?? "rigid_4bolt_m3";
  const matchedHotendPreset = hotendPresets.find(
    (hp) => hp.id === spec.toolhead.hotend.preset_id
  );
  const matchedBakedHotendId =
    !matchedHotendPreset &&
    spec.toolhead.hotend.preset_id &&
    extraBakedHotends.some(([assetId]) => assetId === spec.toolhead.hotend.preset_id)
      ? spec.toolhead.hotend.preset_id
      : null;
  const selectedHotendPresetVal = matchedHotendPreset
    ? matchedHotendPreset.id
    : matchedBakedHotendId ?? "__custom__";
  const hotendTier = hotendStiffnessTier(effectiveHotendDisplayK);
  const currentHotendMountShortLabel =
    HOTEND_MOUNT_STYLE_OPTIONS.find((o) => o.id === currentHotendMountStyle)?.shortLabel ??
    currentHotendMountStyle;

  return (
            <div className="section-card" data-design-section="hotend">
              <div className="section-title">
                <span>Hotend Stiffness, Thermal &amp; Cooling</span>
                <span className="section-title-pills">
                  {activeHotendCadCms && (
                    <span
                      className="badge-pill info"
                      title={formatCadCmsTooltip(activeHotendCadCms)}
                    >
                      {isBracedHotendCms ? "CAD CMS · Ti Brace" : "CAD CMS"}
                    </span>
                  )}
                  <span className="sub-block-tag tabular-nums">
                    {fmtMassG(effectiveHotendMassG)}g
                  </span>
                  <span
                    className="sub-block-tag tabular-nums"
                    style={{ color: hotendTier.color }}
                    title={
                      activeHotendCadK != null
                        ? "CAD 3D Tet4 harmonic lateral tip stiffness (2 / (1/Kx + 1/Ky))"
                        : "Hotend structural bending stiffness"
                    }
                  >
                    {effectiveHotendDisplayK.toLocaleString()} N/mm
                  </span>
                </span>
              </div>

              {showHints && (
                <div className="hint-line">
                  Hotend mount architecture governs nozzle-tip bending stiffness (V6 groove-mount
                  whip vs. rigid 4-bolt standoffs or Chube 6-bolt Ti shell); melt zone &amp; wattage
                  set max volumetric flow.
                </div>
              )}

              <div className="field-row">
                <label htmlFor="dp-hotend-preset-45">Hotend Preset</label>
                <select id="dp-hotend-preset-45"
                  value={selectedHotendPresetVal}
                  onChange={(e) => {
                    const nextId = e.target.value;
                    const chosen = hotendPresets.find((hp) => hp.id === nextId);
                    if (!chosen) {
                      const baked = bakedAssets.hotends?.[nextId];
                      if (!baked) return;
                      const prevName = spec.toolhead.hotend.name;
                      const bakedName = baked.name || humanizeId(nextId);
                      const bakedMassG = Math.max(5, Math.round((baked.mass_kg ?? 0.045) * 1000));
                      const kx = baked.port_stiffness_n_per_mm?.kx;
                      const ky = baked.port_stiffness_n_per_mm?.ky;
                      const bakedK =
                        kx && ky && kx > 0 && ky > 0 ? Math.round(2 / (1 / kx + 1 / ky)) : 1200;
                      const bakedLenMm =
                        baked.port0_mm && baked.port1_mm
                          ? Math.max(
                              25,
                              Math.round(Math.abs(baked.port1_mm[2] - baked.port0_mm[2]))
                            )
                          : spec.toolhead.hotend.overall_length_mm ?? 55;
                      update(
                        (d) => {
                          d.toolhead.hotend.preset_id = nextId;
                          d.toolhead.hotend.name = bakedName;
                          d.toolhead.hotend.mass_g = bakedMassG;
                          d.toolhead.hotend.structural_stiffness_n_per_mm = bakedK;
                          d.toolhead.hotend.overall_length_mm = bakedLenMm;
                          d.toolhead.nozzle_shield_mode = "auto";
                          d.use_cad_superelements = true;
                        },
                        `Hotend: ${prevName} → ${bakedName} (${bakedK} N/mm · ${bakedMassG}g)`
                      );
                      return;
                    }
                    const prevName = spec.toolhead.hotend.name;
                    const chosenMassG = chosen.mass_g ?? 45;
                    update(
                      (d) => {
                        d.toolhead.hotend.preset_id = chosen.id;
                        d.toolhead.hotend.name = chosen.name;
                        d.toolhead.hotend.mass_g = chosenMassG;
                        if (d.toolhead.hotend.mount_style !== chosen.mount_style) {
                          clearModelOverrideGroup(d, "hotend_clamp");
                        }
                        d.toolhead.hotend.mount_style = chosen.mount_style as HotendMountStyle;
                        d.toolhead.hotend.structural_stiffness_n_per_mm =
                          chosen.structural_stiffness_n_per_mm;
                        d.toolhead.hotend.overall_length_mm = chosen.overall_length_mm;
                        d.toolhead.hotend.heater_power_w = chosen.heater_power_w;
                        d.toolhead.hotend.block_material = chosen.block_material;
                        d.toolhead.hotend.block_mass_g = chosen.block_mass_g;
                        d.toolhead.hotend.nozzle_material = chosen.nozzle_material;
                        d.toolhead.hotend.melt_zone_length_mm = chosen.melt_zone_length_mm;
                        d.toolhead.hotend.heatbreak_material = chosen.heatbreak_material;
                        d.toolhead.hotend.heatbreak_length_mm = chosen.heatbreak_length_mm;
                        d.toolhead.hotend.heatbreak_wall_mm = chosen.heatbreak_wall_mm;
                        d.toolhead.hotend.heatbreak_id_mm = chosen.heatbreak_id_mm;
                        d.toolhead.hotend.heatsink_thermal_resistance_k_w =
                          chosen.heatsink_thermal_resistance_k_w;
                        d.toolhead.hotend.max_temp_c = chosen.max_temp_c;
                        d.toolhead.nozzle_shield_mode = "auto";
                      },
                      `Hotend: ${prevName} → ${chosen.short_name || chosen.name} (${Math.round(chosen.structural_stiffness_n_per_mm)} N/mm · ${Math.round(chosenMassG)}g)`
                    );
                  }}
                >
                  {selectedHotendPresetVal === "__custom__" && (
                    <option value="__custom__">
                      {spec.toolhead.hotend.name || "Custom Hotend"} — {specHotendStiffness} N/mm ·{" "}
                      {specHotendMassG}g (Custom)
                    </option>
                  )}
                  {hotendPresets.map((hp) => {
                    const isSelectedPreset = hp.id === spec.toolhead.hotend.preset_id;
                    const displayK =
                      isSelectedPreset && activeHotendCadK != null
                        ? effectiveHotendDisplayK
                        : Math.round(hp.structural_stiffness_n_per_mm);
                    const displayMassG =
                      isSelectedPreset && effectiveHotendMassG != null
                        ? effectiveHotendMassG
                        : hp.mass_g;
                    const tier = hotendStiffnessTier(displayK);
                    const tierLabel =
                      isSelectedPreset && isBracedHotendCms ? "Ti Braced" : tier.label;
                    const wtStr =
                      displayMassG != null ? ` · ${fmtMassG(displayMassG)}g` : "";
                    return (
                      <option key={hp.id} value={hp.id}>
                        {hp.name} — {displayK.toLocaleString()} N/mm{wtStr} ({tierLabel})
                      </option>
                    );
                  })}
                  {extraBakedHotends.length > 0 && (
                    <optgroup label="Baked CAD Assets">
                      {extraBakedHotends.map(([assetId, manifest]) => {
                        const isSelectedPreset = assetId === spec.toolhead.hotend.preset_id;
                        const kx = manifest.port_stiffness_n_per_mm?.kx;
                        const ky = manifest.port_stiffness_n_per_mm?.ky;
                        const manifestK =
                          kx && ky && kx > 0 && ky > 0
                            ? Math.round(2 / (1 / kx + 1 / ky))
                            : 1200;
                        const displayK =
                          isSelectedPreset && activeHotendCadK != null
                            ? effectiveHotendDisplayK
                            : manifestK;
                        const manifestMassG =
                          manifest.mass_kg != null ? Math.round(manifest.mass_kg * 1000) : null;
                        const displayMassG =
                          isSelectedPreset && effectiveHotendMassG != null
                            ? effectiveHotendMassG
                            : manifestMassG;
                        const tier = hotendStiffnessTier(displayK);
                        const tierLabel =
                          isSelectedPreset && isBracedHotendCms ? "Ti Braced" : tier.label;
                        const wtStr =
                          displayMassG != null ? ` · ${fmtMassG(displayMassG)}g` : "";
                        const label = manifest.name || humanizeId(assetId);
                        return (
                          <option key={assetId} value={assetId}>
                            {label} — {displayK.toLocaleString()} N/mm{wtStr} ({tierLabel})
                          </option>
                        );
                      })}
                    </optgroup>
                  )}
                </select>
                {activeHotendCadCms && (
                  <div
                    className="cad-driven-summary"
                    title={formatCadCmsTooltip(activeHotendCadCms)}
                  >
                    3D {activeHotendCadCms.element_type || "Tet4"}
                    {isBracedHotendCms ? " · Ti Bottom Brace" : ""} ·{" "}
                    {formatElemCount(activeHotendCadCms.num_elements)} Tets · Kx{" "}
                    {Number.isFinite(activeHotendCadCms.port_stiffness_n_per_mm?.kx)
                      ? Math.round(activeHotendCadCms.port_stiffness_n_per_mm.kx).toLocaleString()
                      : "—"}{" "}
                    / Ky{" "}
                    {Number.isFinite(activeHotendCadCms.port_stiffness_n_per_mm?.ky)
                      ? Math.round(activeHotendCadCms.port_stiffness_n_per_mm.ky).toLocaleString()
                      : "—"}{" "}
                    N/mm · L {effectiveHotendLenMm.toFixed(1)} mm
                  </div>
                )}
                {matchedHotendPreset?.notes && showHints && (
                  <div className="hint-line">{matchedHotendPreset.notes}</div>
                )}
                <PresetAffectsStrip
                  visible={advancedMode}
                  items={[
                    { label: "Weight", value: `${fmtMassG(effectiveHotendMassG)} g` },
                    { label: "Mount", value: currentHotendMountShortLabel },
                    { label: "Stiffness", value: `${effectiveHotendDisplayK.toLocaleString()} N/mm` },
                    { label: "Heater", value: `${Math.round(spec.toolhead.hotend.heater_power_w)} W` },
                    { label: "Melt Zone", value: `${spec.toolhead.hotend.melt_zone_length_mm} mm` },
                    {
                      label: "Heatbreak",
                      value:
                        HEATBREAK_LABELS[spec.toolhead.hotend.heatbreak_material] ??
                        materialLabel(spec.toolhead.hotend.heatbreak_material),
                    },
                  ]}
                  suffix="(editable below)"
                />
              </div>

              {cadHotendBraceAvailable && (() => {
                const isTrinityBrace = (spec.toolhead.preset_id || "").toLowerCase().startsWith("trinity_");
                const braceLabel = isTrinityBrace ? "Unibody Heatsink Brace" : "Ti Bottom Brace";
                const braceTitle = isTrinityBrace
                  ? "Toggle the Trinity 1.6 monolithic unibody heatsink/shell wrap brace (couples the lower heatsink/Ti shell directly into the chassis for 3.7–8.3× higher lateral nozzle stiffness)"
                  : "Toggle the Microwaved lower Ti-6Al-4V hotend bottom brace (+1.7 g, ties the lower hotend block into the SLM chassis for ~8.5× higher lateral nozzle stiffness)";
                return (
                <label
                  className={`checkbox-row cad-cms-toggle-row${useHotendBottomBrace ? " active" : ""}`}
                  title={braceTitle}
                >
                  <input
                    id="dp-use-hotend-bottom-brace"
                    type="checkbox"
                    checked={useHotendBottomBrace}
                    onChange={(e) => {
                      const checked = e.target.checked;
                      update(
                        (d) => {
                          d.toolhead.use_hotend_bottom_brace = checked;
                        },
                        `${braceLabel}: ${checked ? "On (Braced)" : "Off (Unbraced)"}`
                      );
                    }}
                  />
                  <span>{braceLabel}</span>
                  <span
                    className={`badge-pill ${useHotendBottomBrace ? "info" : "warn"} cad-cms-toggle-count`}
                  >
                    {useHotendBottomBrace ? "ON" : "OFF"} ·{" "}
                    {effectiveHotendDisplayK.toLocaleString()} N/mm
                  </span>
                </label>
                );
              })()}

              {nozzleShieldOptions.length > 0 && (
                <div className="field-row">
                  <label
                    htmlFor="dp-nozzle-shield-mode"
                    title="Swap between the stock Tricorn Ti bucket shield, Microwaved's optional SLM Ti nozzle shield, or bare nozzle (updates total weight, CoG, and 3D model)"
                  >
                    Nozzle Shield
                  </label>
                  <select
                    id="dp-nozzle-shield-mode"
                    value={resolvedNozzleShieldMode}
                    onChange={(e) => {
                      const nextMode = e.target.value as "stock" | "microwaved" | "none";
                      const modeLabels: Record<string, string> = {
                        microwaved: "Microwaved SLM Ti Shield",
                        stock: "Stock Tricorn Ti Bucket",
                        none: "None (Bare Nozzle)",
                      };
                      update(
                        (d) => {
                          d.toolhead.nozzle_shield_mode = nextMode;
                        },
                        `Nozzle Shield: ${modeLabels[nextMode] ?? nextMode}`
                      );
                    }}
                  >
                    {nozzleShieldOptions.includes("microwaved") && (
                      <option value="microwaved">Microwaved SLM Ti Shield (1.3 g)</option>
                    )}
                    {nozzleShieldOptions.includes("stock") && (
                      <option value="stock">Stock Tricorn Ti Bucket (2.7 g)</option>
                    )}
                    {nozzleShieldOptions.includes("none") && (
                      <option value="none">None — Bare Nozzle (0 g)</option>
                    )}
                  </select>
                </div>
              )}

              <div className="field-grid-2">
                <div className="field-row">
                  <label htmlFor="dp-nozzle-dia-mm-46">Nozzle Dia (mm)</label>
                  <input id="dp-nozzle-dia-mm-46"
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
                <div className="field-row">
                  <label htmlFor="dp-cooling-fan-cfm-47">Cooling Fan (CFM)</label>
                  <input id="dp-cooling-fan-cfm-47"
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

              {(() => {
                const currentCpapBlower: CpapBlowerType =
                  spec.toolhead.cpap_blower_id ?? "none";
                const currentHoseIdMm = spec.toolhead.cpap_hose_id_mm ?? 15;
                const currentCpapPwm = spec.toolhead.cpap_pwm_pct ?? 100;
                const cpapActive = currentCpapBlower !== "none";
                return (
                  <div className="sub-block">
                    <div className="sub-block-title">
                      <span>CPAP Blower &amp; Duct</span>
                      <span className="sub-block-tag">
                        {CPAP_BLOWER_LABELS[currentCpapBlower] ?? currentCpapBlower}
                      </span>
                    </div>

                    <div className="field-row">
                      <label htmlFor="dp-cpap-blower-id">CPAP Blower</label>
                      <select
                        id="dp-cpap-blower-id"
                        value={currentCpapBlower}
                        title="Remote brushless CPAP part-cooling blower (reduces toolhead mass and boosts cooling flow; exhaust heating & hose drag modeled)"
                        onChange={(e) => {
                          const nextVal = e.target.value as CpapBlowerType;
                          const prevLbl =
                            CPAP_BLOWER_LABELS[currentCpapBlower] ?? currentCpapBlower;
                          const nextLbl = CPAP_BLOWER_LABELS[nextVal] ?? nextVal;
                          update(
                            (d) => {
                              d.toolhead.cpap_blower_id = nextVal;
                              if (nextVal !== "none") {
                                if (d.toolhead.cpap_hose_id_mm == null) {
                                  d.toolhead.cpap_hose_id_mm = 15;
                                }
                                if (d.toolhead.cpap_pwm_pct == null) {
                                  d.toolhead.cpap_pwm_pct = 100;
                                }
                              }
                            },
                            `CPAP Blower: ${prevLbl} → ${nextLbl}`
                          );
                        }}
                      >
                        <option value="none">Onboard Fans (No CPAP)</option>
                        <option value="ws7040_24v">WS7040 24V (5.8 dm³/s)</option>
                        <option value="wm9290_24v">WM9290 24V (9.2 dm³/s, Quietest)</option>
                        <option value="wm7575_220vac">WM7575 220VAC (11.6 dm³/s)</option>
                      </select>
                    </div>

                    {cpapActive && (
                      <>
                        <div className="field-grid-2">
                          <div className="field-row">
                            <label htmlFor="dp-cpap-hose-id-mm">CPAP Hose ID (mm)</label>
                            <select
                              id="dp-cpap-hose-id-mm"
                              value={currentHoseIdMm}
                              title="Internal diameter of flexible CPAP hose: 15 mm HT silicone is lighter & tighter routing; 18 mm standard gives higher flow"
                              onChange={(e) => {
                                const nextMm = Number(e.target.value);
                                if (!Number.isFinite(nextMm) || nextMm <= 0) return;
                                update(
                                  (d) => {
                                    d.toolhead.cpap_hose_id_mm = nextMm;
                                  },
                                  `CPAP Hose ID: ${currentHoseIdMm} → ${nextMm} mm`
                                );
                              }}
                            >
                              {currentHoseIdMm !== 15 && currentHoseIdMm !== 18 && (
                                <option value={currentHoseIdMm}>
                                  {currentHoseIdMm} mm (Custom)
                                </option>
                              )}
                              <option value={15}>15 mm HT</option>
                              <option value={18}>18 mm Std</option>
                            </select>
                          </div>
                          <div className="field-row">
                            <label htmlFor="dp-cpap-pwm-pct">CPAP Duty (%)</label>
                            <input
                              id="dp-cpap-pwm-pct"
                              type="number"
                              min={0}
                              max={100}
                              step={5}
                              value={currentCpapPwm}
                              title="CPAP blower PWM duty cycle (%)"
                              onChange={(e) => {
                                const parsed = Number(e.target.value);
                                if (!Number.isFinite(parsed)) return;
                                const clamped = Math.max(0, Math.min(100, Math.round(parsed)));
                                if (clamped === currentCpapPwm) return;
                                update(
                                  (d) => {
                                    d.toolhead.cpap_pwm_pct = clamped;
                                  },
                                  `CPAP Duty: ${currentCpapPwm}% → ${clamped}%`
                                );
                              }}
                            />
                          </div>
                        </div>

                        <label className="checkbox-row">
                          <input
                            id="dp-shared-heatsink-cpap-duct"
                            type="checkbox"
                            checked={Boolean(spec.toolhead.shared_heatsink_cpap_duct)}
                            onChange={(e) => {
                              const nextVal = e.target.checked;
                              update(
                                (d) => {
                                  d.toolhead.shared_heatsink_cpap_duct = nextVal;
                                },
                                nextVal
                                  ? "Shared Heatsink CPAP Duct: Off → Enabled (Kalico)"
                                  : "Shared Heatsink CPAP Duct: Enabled → Off"
                              );
                            }}
                          />
                          <span>Shared Heatsink CPAP Duct (Kalico min-PWM bleed)</span>
                        </label>
                      </>
                    )}
                  </div>
                );
              })()}

              <details className="sub-accordion">
                <summary>
                  <span>Custom Hotend Thermal &amp; Stiffness</span>
                  <span className="sub-accordion-tag">
                    {spec.toolhead.hotend.heater_power_w}W ·{" "}
                    {spec.toolhead.hotend.melt_zone_length_mm}mm
                  </span>
                </summary>
                <div className="sub-accordion-body">
                  <div className="field-row">
                    <label htmlFor="dp-hotend-weight-g-48">
                      <span>Hotend Weight (g)</span>
                      <span
                        className="sub-block-tag"
                        style={{ color: "var(--accent)", fontVariantNumeric: "tabular-nums" }}
                        title="Added directly on top of Toolhead Mass (excl. Hotend) to form moving mass"
                      >
                        + {fmtMassG(effectiveHotendMassG)}g to moving mass
                      </span>
                    </label>
                    <input id="dp-hotend-weight-g-48"
                      type="number"
                      min={5}
                      max={400}
                      step={1}
                      value={hotendMassDraft}
                      onChange={(e) => {
                        const raw = e.target.value;
                        setHotendMassDraft(raw);
                        const v = Number(raw);
                        if (raw.trim() !== "" && Number.isFinite(v) && v >= 5 && v <= 400) {
                          if (hotendMassTimerRef.current != null) {
                            window.clearTimeout(hotendMassTimerRef.current);
                          }
                          hotendMassTimerRef.current = window.setTimeout(
                            () => commitHotendMass(v),
                            220
                          );
                        }
                      }}
                      onBlur={() => {
                        const v = Number(hotendMassDraft);
                        const clamped =
                          hotendMassDraft.trim() !== "" && Number.isFinite(v)
                            ? Math.max(5, Math.min(400, Math.round(v)))
                            : specHotendMassG;
                        setHotendMassDraft(String(clamped));
                        commitHotendMass(clamped);
                      }}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") {
                          const v = Number(hotendMassDraft);
                          if (hotendMassDraft.trim() !== "" && Number.isFinite(v)) {
                            const clamped = Math.max(5, Math.min(400, Math.round(v)));
                            setHotendMassDraft(String(clamped));
                            commitHotendMass(clamped);
                          }
                        }
                      }}
                    />
                  </div>

                  {!activeHotendCadCms ? (
                    <>
                      <div className="field-row">
                        <label htmlFor="dp-hotend-mount-style-49">Hotend Mount Style</label>
                        <select id="dp-hotend-mount-style-49"
                          value={currentHotendMountStyle}
                          onChange={(e) => {
                            const nextStyle = e.target.value as HotendMountStyle;
                            const prevLbl =
                              HOTEND_MOUNT_STYLE_OPTIONS.find((o) => o.id === currentHotendMountStyle)
                                ?.shortLabel ?? currentHotendMountStyle;
                            const nextLbl =
                              HOTEND_MOUNT_STYLE_OPTIONS.find((o) => o.id === nextStyle)?.shortLabel ??
                              nextStyle;
                            const hadOv = Boolean(
                              spec.model_overrides?.hotend_clamp &&
                                Object.values(spec.model_overrides.hotend_clamp).some((v) => v != null)
                            );
                            update(
                              (d) => {
                                d.toolhead.hotend.mount_style = nextStyle;
                                d.toolhead.hotend.preset_id = null;
                                clearModelOverrideGroup(d, "hotend_clamp");
                              },
                              `Hotend Mount: ${prevLbl} → ${nextLbl}${hadOv ? " (override reset)" : ""}`
                            );
                          }}
                        >
                          {HOTEND_MOUNT_STYLE_OPTIONS.map((opt) => (
                            <option key={opt.id} value={opt.id}>
                              {opt.label}
                            </option>
                          ))}
                        </select>
                        <ModelOverrideControl
                          visible={advancedMode}
                          idPrefix="ov-hotend-clamp"
                          title="Hotend Clamp"
                          group="hotend_clamp"
                          spec={spec}
                          sim={sim}
                          update={update}
                          fields={[
                            {
                              key: "clamp_stiffness_base_n_per_mm",
                              label: "Clamp Base Stiffness",
                              unit: "N/mm",
                              min: 50,
                              max: 50000,
                              step: 100,
                              decimals: 0,
                              fallbackBaseline:
                                catalog.hotend_mount_styles?.find((m) => m.id === currentHotendMountStyle)
                                  ?.clamp_stiffness_base_n_per_mm ??
                                FALLBACK_HOTEND_CLAMP_N_PER_MM[currentHotendMountStyle],
                              tooltip: "Baseline hotend mounting clamp stiffness before mount material & thickness scaling",
                            },
                          ]}
                        />
                      </div>

                      <div className="slider-row">
                        <label htmlFor="dp-hotend-structural-stiffness-n-mm-50">
                          <span>Hotend Structural Stiffness (N/mm)</span>
                          <span className="tension-readout">
                            <strong>{hotendStiffnessDraft.toLocaleString()} N/mm</strong>
                            <span className="kpi-sub" style={{ color: hotendTier.color }}>
                              {hotendTier.label}
                            </span>
                          </span>
                        </label>
                        <div className="tension-control-row">
                          <input id="dp-hotend-structural-stiffness-n-mm-50"
                            type="range"
                            min={25}
                            max={10000}
                            step={25}
                            value={Math.max(25, Math.min(10000, hotendStiffnessDraft))}
                            onChange={(e) => handleHotendStiffnessSliderChange(Number(e.target.value))}
                            onPointerUp={() => commitHotendStiffness(hotendStiffnessDraft)}
                            onKeyUp={() => commitHotendStiffness(hotendStiffnessDraft)}
                          />
                          <input aria-label="Hotend Structural Stiffness (N/mm) value"
                            type="number"
                            className="tension-num-input"
                            min={25}
                            max={10000}
                            step={25}
                            value={hotendStiffnessDraft}
                            onChange={(e) => {
                              const v = Number(e.target.value);
                              if (!Number.isFinite(v)) return;
                              setHotendStiffnessDraft(v);
                              if (v >= 25 && v <= 10000) commitHotendStiffness(v);
                            }}
                            onBlur={() => {
                              const clamped = Math.max(
                                25,
                                Math.min(10000, Math.round(hotendStiffnessDraft))
                              );
                              setHotendStiffnessDraft(clamped);
                              commitHotendStiffness(clamped);
                            }}
                          />
                        </div>
                      </div>
                    </>
                  ) : (
                    <ModelOverrideControl
                      visible={advancedMode}
                      idPrefix="ov-hotend-clamp"
                      title="Hotend Clamp"
                      group="hotend_clamp"
                      spec={spec}
                      sim={sim}
                      update={update}
                      fields={[
                        {
                          key: "clamp_stiffness_base_n_per_mm",
                          label: "Clamp Base Stiffness",
                          unit: "N/mm",
                          min: 50,
                          max: 50000,
                          step: 100,
                          decimals: 0,
                          fallbackBaseline:
                            catalog.hotend_mount_styles?.find((m) => m.id === currentHotendMountStyle)
                              ?.clamp_stiffness_base_n_per_mm ??
                            FALLBACK_HOTEND_CLAMP_N_PER_MM[currentHotendMountStyle],
                          tooltip: "Baseline hotend mounting clamp stiffness before mount material & thickness scaling",
                        },
                      ]}
                    />
                  )}

                  <div className="field-grid-2">
                    <div className="field-row">
                      <label htmlFor="dp-heater-power-w-51">Heater Power (W)</label>
                      <input id="dp-heater-power-w-51"
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
                    <div className="field-row">
                      <label htmlFor="dp-melt-zone-mm-52">Melt Zone (mm)</label>
                      <input id="dp-melt-zone-mm-52"
                        type="number"
                        value={spec.toolhead.hotend.melt_zone_length_mm}
                        onChange={(e) =>
                          updatePositiveNum(
                            e.target.value,
                            (d, v) => {
                              d.toolhead.hotend.melt_zone_length_mm = v;
                            },
                            (v) =>
                              `Melt Zone: ${spec.toolhead.hotend.melt_zone_length_mm} → ${v} mm`
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

                  <div className="field-row">
                    <label htmlFor="dp-heatbreak-material-53">Heatbreak Material</label>
                    <select id="dp-heatbreak-material-53"
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
                      {spec.toolhead.hotend.heatbreak_material !== "titanium_grade5" &&
                        spec.toolhead.hotend.heatbreak_material !== "stainless_304" && (
                          <option value={spec.toolhead.hotend.heatbreak_material}>
                            {materialLabel(spec.toolhead.hotend.heatbreak_material)} (Custom)
                          </option>
                        )}
                      <option value="titanium_grade5">Titanium Gr.5 (Low Heat Creep)</option>
                      <option value="stainless_304">Stainless 304 (Standard)</option>
                    </select>
                  </div>
                </div>
              </details>
            </div>
  );
};
