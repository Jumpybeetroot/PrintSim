import React, { useCallback, useEffect, useRef, useState } from "react";
import { DesignPanel } from "./components/DesignPanel";
import { MetricsPanel } from "./components/MetricsPanel";
import { PrinterViewport } from "./components/PrinterViewport";
import {
  CatalogResponse,
  CompareMode,
  HistorySnapshot,
  PrinterBOMResponse,
  PrinterSpecData,
  SimulationResponse,
  SolidFeaEngine,
  SolidFEAResponse,
  TetFEAResponse,
  TopoOptRequestParams,
  TopologyOptResponse,
  ViewMode,
} from "./types";

function arrayBufferToBase64(buf: ArrayBuffer): string {
  const bytes = new Uint8Array(buf);
  const chunkSize = 0x8000;
  const parts: string[] = [];
  for (let i = 0; i < bytes.length; i += chunkSize) {
    parts.push(String.fromCharCode(...bytes.subarray(i, i + chunkSize)));
  }
  return btoa(parts.join(""));
}

const BRACING_LABELS: Record<string, string> = {
  none: "None",
  corner_gussets: "Corner Gussets",
  rear_cross: "Rear Cross + Top",
  full_truss: "Full 3D Truss",
};

const BELT_LABELS: Record<string, string> = {
  gt2_6mm: "GT2 6mm",
  gt2_9mm: "GT2 9mm",
  gt2_12mm: "GT2 12mm",
};

const ARCH_LABELS: Record<string, string> = {
  corexy: "CoreXY",
  cartesian_bedslinger: "Cartesian Bedslinger",
  delta: "Delta",
};

const PANEL_MATERIAL_LABELS: Record<string, string> = {
  acrylic_pmma: "Acrylic",
  polycarbonate: "Polycarbonate",
  acm_dibond: "ACM / Dibond",
  aluminum_5052_sheet: "Aluminum 5052",
  carbon_fiber_plate: "Carbon Fiber 3K",
};

const PANEL_MOUNTING_LABELS: Record<string, string> = {
  none: "Open",
  clip_enclosure: "Clip-In",
  bolted_structural: "Bolted Structural",
};

function formatPanelDesc(frame: PrinterSpecData["frame"]): string {
  const mounting = !frame.enclosed_panels ? "none" : frame.panel_mounting || "bolted_structural";
  if (!frame.enclosed_panels || mounting === "none") return "Open";
  const mat = PANEL_MATERIAL_LABELS[frame.panel_material] ?? frame.panel_material ?? "Acrylic";
  const mnt = PANEL_MOUNTING_LABELS[mounting] ?? mounting;
  return `${frame.panel_thickness_mm}mm ${mat} (${mnt})`;
}

function formatDriveMode(mode?: string): string {
  return mode === "awd" ? "AWD (4-Motor)" : "2WD (2-Motor)";
}

function shortCatalogName(
  id: string,
  items?: Array<{ id: string; name: string }>
): string {
  const found = items?.find((it) => it.id === id);
  if (!found) return id;
  return found.name.replace(/\s*\([^)]*\)\s*/g, "").trim() || found.name;
}

function describeSpecDiff(
  prev: PrinterSpecData | null,
  next: PrinterSpecData,
  catalog: CatalogResponse | null
): string {
  if (!prev) return "Updated design";
  const diffs: string[] = [];

  if (prev.architecture !== next.architecture) {
    diffs.push(
      `Architecture: ${ARCH_LABELS[prev.architecture] ?? prev.architecture} → ${
        ARCH_LABELS[next.architecture] ?? next.architecture
      }`
    );
  }
  if (
    prev.frame.width_mm !== next.frame.width_mm ||
    prev.frame.depth_mm !== next.frame.depth_mm ||
    prev.frame.height_mm !== next.frame.height_mm
  ) {
    diffs.push(
      `Frame Size: ${prev.frame.width_mm}×${prev.frame.depth_mm}×${prev.frame.height_mm} → ${next.frame.width_mm}×${next.frame.depth_mm}×${next.frame.height_mm} mm`
    );
  }
  if (prev.frame.profile !== next.frame.profile) {
    diffs.push(
      `Frame Profile: ${shortCatalogName(prev.frame.profile, catalog?.profiles)} → ${shortCatalogName(
        next.frame.profile,
        catalog?.profiles
      )}`
    );
  }
  if (prev.frame.material !== next.frame.material) {
    diffs.push(
      `Frame Material: ${shortCatalogName(prev.frame.material, catalog?.materials)} → ${shortCatalogName(
        next.frame.material,
        catalog?.materials
      )}`
    );
  }
  if (prev.frame.bracing !== next.frame.bracing) {
    diffs.push(
      `Bracing: ${BRACING_LABELS[prev.frame.bracing] ?? prev.frame.bracing} → ${
        BRACING_LABELS[next.frame.bracing] ?? next.frame.bracing
      }`
    );
  }
  if (
    prev.frame.enclosed_panels !== next.frame.enclosed_panels ||
    prev.frame.panel_mounting !== next.frame.panel_mounting ||
    prev.frame.panel_material !== next.frame.panel_material ||
    prev.frame.panel_thickness_mm !== next.frame.panel_thickness_mm
  ) {
    diffs.push(`Panels: ${formatPanelDesc(prev.frame)} → ${formatPanelDesc(next.frame)}`);
  }
  if ((prev.gantry.drive_mode ?? "2wd") !== (next.gantry.drive_mode ?? "2wd")) {
    diffs.push(
      `Drive Mode: ${formatDriveMode(prev.gantry.drive_mode)} → ${formatDriveMode(next.gantry.drive_mode)}`
    );
  }
  if (Boolean(prev.gantry.double_shear_pulleys) !== Boolean(next.gantry.double_shear_pulleys)) {
    diffs.push(
      `Pulley Mounts: ${prev.gantry.double_shear_pulleys ? "Double-Shear" : "Single-Shear"} → ${
        next.gantry.double_shear_pulleys ? "Double-Shear" : "Single-Shear"
      }`
    );
  }
  if (prev.gantry.x_beam_profile !== next.gantry.x_beam_profile) {
    diffs.push(
      `X Beam: ${shortCatalogName(prev.gantry.x_beam_profile, catalog?.profiles)} → ${shortCatalogName(
        next.gantry.x_beam_profile,
        catalog?.profiles
      )}`
    );
  }
  if (prev.gantry.x_beam_material !== next.gantry.x_beam_material) {
    diffs.push(
      `X Beam Material: ${shortCatalogName(prev.gantry.x_beam_material, catalog?.materials)} → ${shortCatalogName(
        next.gantry.x_beam_material,
        catalog?.materials
      )}`
    );
  }
  if (prev.gantry.belt.type !== next.gantry.belt.type) {
    diffs.push(
      `Belt Width: ${BELT_LABELS[prev.gantry.belt.type] ?? prev.gantry.belt.type} → ${
        BELT_LABELS[next.gantry.belt.type] ?? next.gantry.belt.type
      }`
    );
  }
  if (prev.gantry.belt.core !== next.gantry.belt.core) {
    diffs.push(`Belt Core: ${prev.gantry.belt.core} → ${next.gantry.belt.core}`);
  }
  if (prev.gantry.belt.pretension_n !== next.gantry.belt.pretension_n) {
    diffs.push(`Belt Tension: ${prev.gantry.belt.pretension_n} N → ${next.gantry.belt.pretension_n} N`);
  }
  if (prev.kinematics.max_velocity_mm_s !== next.kinematics.max_velocity_mm_s) {
    diffs.push(`Max Vel: ${prev.kinematics.max_velocity_mm_s} → ${next.kinematics.max_velocity_mm_s} mm/s`);
  }
  if (prev.kinematics.max_accel_mm_s2 !== next.kinematics.max_accel_mm_s2) {
    diffs.push(
      `Max Accel: ${prev.kinematics.max_accel_mm_s2.toLocaleString()} → ${next.kinematics.max_accel_mm_s2.toLocaleString()} mm/s²`
    );
  }
  if (prev.kinematics.input_shaper !== next.kinematics.input_shaper) {
    diffs.push(`Input Shaper: ${prev.kinematics.input_shaper.toUpperCase()} → ${next.kinematics.input_shaper.toUpperCase()}`);
  }
  if (prev.kinematics.xy_stepper.supply_voltage_v !== next.kinematics.xy_stepper.supply_voltage_v) {
    diffs.push(
      `Stepper Voltage: ${prev.kinematics.xy_stepper.supply_voltage_v}V → ${next.kinematics.xy_stepper.supply_voltage_v}V`
    );
  }
  if (prev.toolhead.total_mass_g !== next.toolhead.total_mass_g) {
    diffs.push(`Toolhead Mass: ${prev.toolhead.total_mass_g}g → ${next.toolhead.total_mass_g}g`);
  }
  if (prev.toolhead.hotend.heater_power_w !== next.toolhead.hotend.heater_power_w) {
    diffs.push(`Heater Power: ${prev.toolhead.hotend.heater_power_w}W → ${next.toolhead.hotend.heater_power_w}W`);
  }
  if (prev.toolhead.hotend.melt_zone_length_mm !== next.toolhead.hotend.melt_zone_length_mm) {
    diffs.push(
      `Melt Zone: ${prev.toolhead.hotend.melt_zone_length_mm} → ${next.toolhead.hotend.melt_zone_length_mm} mm`
    );
  }
  if (prev.toolhead.hotend.nozzle_diameter_mm !== next.toolhead.hotend.nozzle_diameter_mm) {
    diffs.push(
      `Nozzle: ${prev.toolhead.hotend.nozzle_diameter_mm} → ${next.toolhead.hotend.nozzle_diameter_mm} mm`
    );
  }
  if (prev.toolhead.hotend.heatbreak_material !== next.toolhead.hotend.heatbreak_material) {
    diffs.push(
      `Heatbreak: ${shortCatalogName(prev.toolhead.hotend.heatbreak_material, catalog?.materials)} → ${shortCatalogName(
        next.toolhead.hotend.heatbreak_material,
        catalog?.materials
      )}`
    );
  }
  if (prev.toolhead.hotend.part_cooling_cfm !== next.toolhead.hotend.part_cooling_cfm) {
    diffs.push(`Cooling Fan: ${prev.toolhead.hotend.part_cooling_cfm} → ${next.toolhead.hotend.part_cooling_cfm} CFM`);
  }

  if (diffs.length === 0) return "Updated parameters";
  if (diffs.length <= 2) return diffs.join(" • ");
  return `${diffs[0]} (+${diffs.length - 1} more)`;
}

export default function App() {
  const [catalog, setCatalog] = useState<CatalogResponse | null>(null);
  const [selectedPreset, setSelectedPreset] = useState<string>("voron_trident_300");
  const [filamentId, setFilamentId] = useState<string>("pla");
  const [spec, setSpec] = useState<PrinterSpecData | null>(null);
  const [yamlText, setYamlText] = useState<string>("");
  const [gcodeText, setGcodeText] = useState<string>("");
  const [sim, setSim] = useState<SimulationResponse | null>(null);
  const [solidFea, setSolidFea] = useState<SolidFEAResponse | null>(null);
  const [tetFea, setTetFea] = useState<TetFEAResponse | null>(null);
  const [activeSolidEngine, setActiveSolidEngine] = useState<SolidFeaEngine>("tet10");
  const [topoResult, setTopoResult] = useState<TopologyOptResponse | null>(null);
  const [topoLoading, setTopoLoading] = useState<boolean>(false);
  const [exportingStep, setExportingStep] = useState<boolean>(false);
  const [bom, setBom] = useState<PrinterBOMResponse | null>(null);
  const [bomLoading, setBomLoading] = useState<boolean>(false);
  const [stockLengthMm, setStockLengthMm] = useState<number>(1000);
  const [loading, setLoading] = useState<boolean>(false);
  const [optimizing, setOptimizing] = useState<boolean>(false);
  const [optObjective, setOptObjective] = useState<"balanced" | "speed" | "rigidity" | "budget">("balanced");
  const [optMessage, setOptMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Baseline & change-tracking state
  const [baselineSim, setBaselineSim] = useState<SimulationResponse | null>(null);
  const [baselineBom, setBaselineBom] = useState<PrinterBOMResponse | null>(null);
  const [baselineSpec, setBaselineSpec] = useState<PrinterSpecData | null>(null);
  const [baselineFilamentId, setBaselineFilamentId] = useState<string>("pla");
  const [baselineLabel, setBaselineLabel] = useState<string>("Stock Voron Trident 300");
  const [history, setHistory] = useState<HistorySnapshot[]>([]);
  const [lastChangeSummary, setLastChangeSummary] = useState<string | null>(null);
  const [compareMode, setCompareMode] = useState<CompareMode>("last");

  const [viewMode, setViewMode] = useState<ViewMode>("assembly");
  const [selectedModeIdx, setSelectedModeIdx] = useState<number>(0);
  const [feaScale, setFeaScale] = useState<number>(600);
  const [layerLimit, setLayerLimit] = useState<number>(999);

  // Refs for atomic history capture & stale-request guarding
  const simReqIdRef = useRef<number>(0);
  const simRef = useRef<SimulationResponse | null>(null);
  const bomRef = useRef<PrinterBOMResponse | null>(null);
  const filamentRef = useRef<string>("pla");
  const lastChangeRef = useRef<string | null>(null);
  const catalogRef = useRef<CatalogResponse | null>(null);
  const stockLengthRef = useRef<number>(1000);

  simRef.current = sim;
  bomRef.current = bom;
  filamentRef.current = filamentId;
  lastChangeRef.current = lastChangeSummary;
  catalogRef.current = catalog;
  stockLengthRef.current = stockLengthMm;

  const prevSnapshot = history.length > 0 ? history[history.length - 1] : null;
  const prevSim = prevSnapshot?.sim ?? null;
  const prevBom = prevSnapshot?.bom ?? null;

  const fetchBomForSpec = useCallback(
    async (specData: PrinterSpecData, stockLen: number): Promise<PrinterBOMResponse | null> => {
      try {
        const res = await fetch("/api/bom", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ spec: specData, stock_length_mm: stockLen }),
        });
        if (!res.ok) return null;
        return (await res.json()) as PrinterBOMResponse;
      } catch {
        return null;
      }
    },
    []
  );

  const runSimulate = useCallback(
    async (
      payload: { spec?: PrinterSpecData; yaml_text?: string },
      customGcode?: string,
      filOverride?: string,
      options?: {
        changeLabel?: string;
        isBaselineReset?: boolean;
        baselineName?: string;
      }
    ) => {
      const reqId = ++simReqIdRef.current;
      const activeFil = filOverride ?? filamentRef.current;
      setLoading(true);
      setBomLoading(true);
      setError(null);
      try {
        const res = await fetch("/api/simulate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            ...payload,
            filament_id: activeFil,
            gcode_text: customGcode && customGcode.trim().length > 0 ? customGcode : null,
          }),
        });
        if (!res.ok) {
          const errBody = await res.json().catch(() => ({ detail: res.statusText }));
          throw new Error(errBody.detail || "Simulation failed");
        }
        const data: SimulationResponse = await res.json();
        const bomData = await fetchBomForSpec(data.spec, stockLengthRef.current);

        // Ignore stale out-of-order responses
        if (reqId !== simReqIdRef.current) return;

        if (options?.isBaselineReset) {
          const label = `Stock ${options.baselineName || data.printer_name}`;
          setBaselineSim(data);
          setBaselineBom(bomData);
          setBaselineSpec(data.spec);
          setBaselineFilamentId(activeFil);
          setBaselineLabel(label);
          setHistory([]);
          setLastChangeSummary(null);
          setCompareMode("last");
        } else {
          const currentSim = simRef.current;
          if (currentSim) {
            const snap: HistorySnapshot = {
              sim: currentSim,
              bom: bomRef.current,
              spec: currentSim.spec,
              filamentId: filamentRef.current,
              label: lastChangeRef.current,
            };
            setHistory((prev) => [...prev.slice(-19), snap]);
            const resolvedLabel =
              options?.changeLabel ||
              describeSpecDiff(currentSim.spec, data.spec, catalogRef.current);
            setLastChangeSummary(resolvedLabel);
            setCompareMode("last");
          } else {
            // First simulation ever loaded
            setBaselineSim(data);
            setBaselineBom(bomData);
            setBaselineSpec(data.spec);
            setBaselineFilamentId(activeFil);
            setBaselineLabel(`Stock ${data.printer_name}`);
          }
        }

        setSim(data);
        setBom(bomData);
        setSpec(data.spec);
        setYamlText(data.yaml_text);
        setLayerLimit(Math.max(0, data.virtual_print.layer_count - 1));
      } catch (e) {
        if (reqId === simReqIdRef.current) {
          setError(e instanceof Error ? e.message : String(e));
        }
      } finally {
        if (reqId === simReqIdRef.current) {
          setLoading(false);
          setBomLoading(false);
        }
      }
    },
    [fetchBomForSpec]
  );

  // Refresh BOM when user switches stock bar length (1000mm <-> 2000mm)
  const handleStockLengthChange = useCallback(
    async (lenMm: number) => {
      setStockLengthMm(lenMm);
      stockLengthRef.current = lenMm;
      if (!simRef.current) return;
      setBomLoading(true);
      try {
        const [curBom, baseBom] = await Promise.all([
          fetchBomForSpec(simRef.current.spec, lenMm),
          baselineSpec ? fetchBomForSpec(baselineSpec, lenMm) : Promise.resolve(null),
        ]);
        if (curBom) setBom(curBom);
        if (baseBom) setBaselineBom(baseBom);
      } finally {
        setBomLoading(false);
      }
    },
    [baselineSpec, fetchBomForSpec]
  );

  const runPartFea = useCallback(
    async (params: {
      preset: string;
      materialId: string;
      forceN: [number, number, number];
      fdmAnisotropic: boolean;
      engine: SolidFeaEngine;
    }) => {
      setLoading(true);
      setError(null);
      try {
        if (params.engine === "tet10") {
          const res = await fetch("/api/fea/tet-part", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              preset: params.preset,
              material_id: params.materialId,
              force_n: params.forceN,
              order: 2,
              fdm_anisotropic: params.fdmAnisotropic,
            }),
          });
          if (!res.ok) {
            const errBody = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(errBody.detail || "Curved Tet10 FEA failed");
          }
          const data: TetFEAResponse = await res.json();
          setTetFea(data);
          setActiveSolidEngine("tet10");
          setViewMode("solid_part");
        } else {
          const res = await fetch("/api/fea/solid-part", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              preset: params.preset,
              material_id: params.materialId,
              force_n: params.forceN,
              fdm_anisotropic: params.fdmAnisotropic,
            }),
          });
          if (!res.ok) {
            const errBody = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(errBody.detail || "Solid Hex8 FEA failed");
          }
          const data: SolidFEAResponse = await res.json();
          setSolidFea(data);
          setActiveSolidEngine("hex8");
          setViewMode("solid_part");
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setLoading(false);
      }
    },
    []
  );

  const runStlFea = useCallback(
    async (params: {
      file: File;
      materialId: string;
      fixedFace: string;
      loadFace: string;
      forceN: [number, number, number];
      fdmAnisotropic: boolean;
    }) => {
      setLoading(true);
      setError(null);
      try {
        const buf = await params.file.arrayBuffer();
        const b64 = arrayBufferToBase64(buf);
        const ext = params.file.name.split(".").pop()?.toLowerCase() || "stl";

        if (ext === "step" || ext === "stp") {
          const res = await fetch("/api/fea/tet-step", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              part_name: params.file.name,
              step_base64: b64,
              material_id: params.materialId,
              fixed_face: params.fixedFace,
              load_face: params.loadFace,
              force_n: params.forceN,
              order: 2,
              fdm_anisotropic: params.fdmAnisotropic,
            }),
          });
          if (!res.ok) {
            const errBody = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(errBody.detail || "STEP Tet10 FEA failed");
          }
          const data: TetFEAResponse = await res.json();
          setTetFea(data);
          setActiveSolidEngine("tet10");
          setViewMode("solid_part");
        } else {
          const res = await fetch("/api/fea/solid-stl", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              part_name: params.file.name,
              stl_base64: b64,
              file_type: ext,
              material_id: params.materialId,
              fixed_face: params.fixedFace,
              load_face: params.loadFace,
              force_n: params.forceN,
              fdm_anisotropic: params.fdmAnisotropic,
            }),
          });
          if (!res.ok) {
            const errBody = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(errBody.detail || "STL Solid FEA failed");
          }
          const data: SolidFEAResponse = await res.json();
          setSolidFea(data);
          setActiveSolidEngine("hex8");
          setViewMode("solid_part");
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setLoading(false);
      }
    },
    []
  );

  const handleDownloadPartStep = useCallback(async (preset: string) => {
    setExportingStep(true);
    setError(null);
    try {
      const res = await fetch("/api/export/part-step", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ preset }),
      });
      if (!res.ok) {
        const errBody = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(errBody.detail || "Bracket STEP export failed");
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${preset}.step`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setExportingStep(false);
    }
  }, []);

  const runTopoOpt = useCallback(async (params: TopoOptRequestParams) => {
    setTopoLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/fea/topology-optimize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          preset: params.preset,
          material_id: params.materialId,
          force_n: params.forceN,
          target_volume_fraction: params.targetVolumeFraction,
          iterations: params.iterations,
          preserve_boundaries: params.preserveBoundaries,
          fdm_anisotropic: params.fdmAnisotropic,
        }),
      });
      if (!res.ok) {
        const errBody = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(errBody.detail || "Topology optimization failed");
      }
      const data: TopologyOptResponse = await res.json();
      setTopoResult(data);
      setViewMode("topology");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setTopoLoading(false);
    }
  }, []);

  const downloadTopoStl = useCallback(async (params: TopoOptRequestParams) => {
    setTopoLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/fea/topology-optimize/stl", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          preset: params.preset,
          material_id: params.materialId,
          force_n: params.forceN,
          target_volume_fraction: params.targetVolumeFraction,
          iterations: params.iterations,
          preserve_boundaries: params.preserveBoundaries,
          fdm_anisotropic: params.fdmAnisotropic,
          smooth_iterations: 0,
        }),
      });
      if (!res.ok) {
        const errBody = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(errBody.detail || "Topology STL export failed");
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${params.preset}_topo_${Math.round(params.targetVolumeFraction * 100)}pct.stl`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setTopoLoading(false);
    }
  }, []);

  const loadPreset = useCallback(
    async (presetId: string) => {
      setSelectedPreset(presetId);
      setOptMessage(null);
      const res = await fetch(`/api/presets/${presetId}`);
      if (!res.ok) return;
      const data = await res.json();
      setSpec(data.spec);
      setYamlText(data.yaml_text);
      await runSimulate({ spec: data.spec }, gcodeText, undefined, {
        isBaselineReset: true,
        baselineName: data.spec?.name,
      });
    },
    [gcodeText, runSimulate]
  );

  useEffect(() => {
    (async () => {
      try {
        const res = await fetch("/api/catalog");
        const data: CatalogResponse = await res.json();
        setCatalog(data);
        const firstId =
          data.presets.find((p) => p.id === "voron_trident_300")?.id ||
          data.presets.find((p) => p.id === "corexy_voron_style")?.id ||
          data.presets[0]?.id;
        if (firstId) {
          await loadPreset(firstId);
        }
        // Preload default Curved Tet10 bracket FEA (with Hex8 fallback) so 3D Solid Part tab is instant
        const tetRes = await fetch("/api/fea/tet-part", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            preset: "corner_gusset",
            material_id: "pa12_cf_printed",
            force_n: [60, 0, 0],
            order: 2,
            fdm_anisotropic: true,
          }),
        });
        if (tetRes.ok) {
          setTetFea(await tetRes.json());
          setActiveSolidEngine("tet10");
        } else {
          const pRes = await fetch("/api/fea/solid-part", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ preset: "corner_gusset", material_id: "abs_printed" }),
          });
          if (pRes.ok) {
            setSolidFea(await pRes.json());
            setActiveSolidEngine("hex8");
          }
        }
      } catch {
        setError("Failed to connect to PrintSim backend API on /api. Is `printsim serve` running?");
      }
    })();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleSpecChange = (next: PrinterSpecData, changeLabel?: string) => {
    const label = changeLabel || describeSpecDiff(spec, next, catalog);
    setSpec(next);
    setOptMessage(null);
    void runSimulate({ spec: next }, gcodeText, undefined, { changeLabel: label });
  };

  const handleFilamentChange = (nextFil: string) => {
    if (nextFil === filamentId) return;
    const prevName =
      catalog?.filaments.find((f) => f.id === filamentId)?.name || filamentId.toUpperCase();
    const nextName =
      catalog?.filaments.find((f) => f.id === nextFil)?.name || nextFil.toUpperCase();
    const label = `Filament: ${prevName} → ${nextName}`;
    setFilamentId(nextFil);
    if (spec) {
      void runSimulate({ spec }, gcodeText, nextFil, { changeLabel: label });
    }
  };

  const handleUndo = useCallback(() => {
    if (history.length === 0) return;
    simReqIdRef.current += 1; // cancel any in-flight request
    const last = history[history.length - 1];
    const nextHistory = history.slice(0, -1);
    setHistory(nextHistory);
    setSim(last.sim);
    setBom(last.bom);
    setSpec(last.spec);
    setYamlText(last.sim.yaml_text);
    setFilamentId(last.filamentId);
    setLastChangeSummary(last.label);
    setLayerLimit(Math.max(0, last.sim.virtual_print.layer_count - 1));
    setOptMessage(null);
    setError(null);
    setLoading(false);
    setBomLoading(false);
  }, [history]);

  const handleResetToBaseline = useCallback(() => {
    if (!baselineSim || !baselineSpec) return;
    simReqIdRef.current += 1;
    setHistory([]);
    setSim(baselineSim);
    setBom(baselineBom);
    setSpec(baselineSpec);
    setYamlText(baselineSim.yaml_text);
    setFilamentId(baselineFilamentId);
    setLastChangeSummary(null);
    setCompareMode("last");
    setLayerLimit(Math.max(0, baselineSim.virtual_print.layer_count - 1));
    setOptMessage(null);
    setError(null);
    setLoading(false);
    setBomLoading(false);
  }, [baselineBom, baselineFilamentId, baselineSim, baselineSpec]);

  const handleSetBaseline = useCallback(() => {
    if (!sim || !spec) return;
    setBaselineSim(sim);
    setBaselineBom(bom);
    setBaselineSpec(spec);
    setBaselineFilamentId(filamentId);
    setBaselineLabel("Custom Baseline");
    setHistory([]);
    setLastChangeSummary(null);
    setCompareMode("last");
    setOptMessage(null);
  }, [bom, filamentId, sim, spec]);

  const handleOptimize = async () => {
    if (!spec) return;
    setOptimizing(true);
    setOptMessage(null);
    setError(null);
    try {
      const res = await fetch("/api/optimize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ spec, iterations: 24, objective: optObjective }),
      });
      if (!res.ok) {
        const errBody = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(errBody.detail || "Optimization failed");
      }
      const data = await res.json();
      setOptMessage(
        `AI Search (${optObjective}): ${data.baseline_objective} → ${data.best_objective} (+${data.improvement_delta})`
      );
      await runSimulate({ spec: data.best_spec }, gcodeText, undefined, {
        changeLabel: `AI Optimize (${optObjective})`,
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setOptimizing(false);
    }
  };

  const handleExportStl = async () => {
    if (!spec) return;
    const res = await fetch("/api/export/stl", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(spec),
    });
    if (!res.ok) return;
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${spec.name.toLowerCase().replace(/[^a-z0-9]+/g, "_")}.stl`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleExportStep = async () => {
    if (!spec) return;
    setExportingStep(true);
    setError(null);
    try {
      const res = await fetch("/api/export/step", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(spec),
      });
      if (!res.ok) {
        const errBody = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(errBody.detail || "STEP assembly export failed");
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${spec.name.toLowerCase().replace(/[^a-z0-9]+/g, "_")}.step`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setExportingStep(false);
    }
  };

  const handleExportBomCsv = async () => {
    if (!spec) return;
    setError(null);
    try {
      const res = await fetch("/api/bom/csv", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ spec, stock_length_mm: stockLengthMm }),
      });
      if (!res.ok) {
        const errBody = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(errBody.detail || "BOM CSV export failed");
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${spec.name.toLowerCase().replace(/[^a-z0-9]+/g, "_")}_bom.csv`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-badge">PrintSim 0.5</span>
          <span>Multiphysics 3D Printer Studio</span>
          {optMessage && (
            <span style={{ fontSize: 12, color: "var(--green)", fontWeight: 600, marginLeft: 8 }}>
              ✓ {optMessage}
            </span>
          )}
        </div>

        <div className="topbar-actions">
          <select
            style={{ width: 130 }}
            value={optObjective}
            onChange={(e) => setOptObjective(e.target.value as typeof optObjective)}
            title="AI Optimization Objective"
          >
            <option value="balanced">AI: Balanced</option>
            <option value="speed">AI: Max Speed</option>
            <option value="rigidity">AI: Max Rigidity</option>
            <option value="budget">AI: Budget/Light</option>
          </select>
          <button className="btn ai" disabled={optimizing || !spec} onClick={handleOptimize}>
            {optimizing ? "Optimizing..." : "AI Optimize Design"}
          </button>
          <button className="btn" disabled={!spec} onClick={handleExportStl}>
            Export STL
          </button>
          <button
            className="btn"
            disabled={!spec || exportingStep}
            onClick={handleExportStep}
            title="Export full OpenCASCADE B-Rep STEP CAD assembly"
          >
            {exportingStep ? "Exporting STEP..." : "Export STEP"}
          </button>
          <button className="btn" disabled={!spec} onClick={handleExportBomCsv}>
            Export BOM CSV
          </button>
          <button
            className="btn primary"
            disabled={loading || !spec}
            onClick={() => spec && runSimulate({ spec }, gcodeText, undefined, { changeLabel: "Re-Simulated Design" })}
          >
            {loading ? "Solving..." : "Re-Simulate"}
          </button>
        </div>
      </header>

      <main className="workspace">
        <DesignPanel
          catalog={catalog}
          sim={sim}
          selectedPreset={selectedPreset}
          onSelectPreset={loadPreset}
          filamentId={filamentId}
          onFilamentChange={handleFilamentChange}
          spec={spec}
          baselineSpec={baselineSpec}
          onSpecChange={handleSpecChange}
          yamlText={yamlText}
          onYamlChange={setYamlText}
          onSimulateYaml={() => runSimulate({ yaml_text: yamlText }, gcodeText)}
          gcodeText={gcodeText}
          onGcodeChange={setGcodeText}
          onRunPartFea={runPartFea}
          onRunStlFea={runStlFea}
          onDownloadPartStep={handleDownloadPartStep}
          exportingStep={exportingStep}
          onRunTopoOpt={runTopoOpt}
          onDownloadTopoStl={downloadTopoStl}
          hasTopoResult={Boolean(topoResult)}
          topoLoading={topoLoading}
          loading={loading}
        />

        <PrinterViewport
          sim={sim}
          solidFea={solidFea}
          tetFea={tetFea}
          activeSolidEngine={activeSolidEngine}
          topoResult={topoResult}
          viewMode={viewMode}
          onViewModeChange={setViewMode}
          selectedModeIdx={selectedModeIdx}
          onSelectModeIdx={setSelectedModeIdx}
          feaScale={feaScale}
          onFeaScaleChange={setFeaScale}
          layerLimit={layerLimit}
          onLayerLimitChange={setLayerLimit}
        />

        <MetricsPanel
          sim={sim}
          baselineSim={baselineSim}
          baselineBom={baselineBom}
          baselineLabel={baselineLabel}
          prevSim={prevSim}
          prevBom={prevBom}
          lastChangeSummary={lastChangeSummary}
          compareMode={compareMode}
          onCompareModeChange={setCompareMode}
          canUndo={history.length > 0}
          onUndo={handleUndo}
          onResetToBaseline={handleResetToBaseline}
          onSetBaseline={handleSetBaseline}
          solidFea={solidFea}
          tetFea={tetFea}
          activeSolidEngine={activeSolidEngine}
          topoResult={topoResult}
          viewMode={viewMode}
          bom={bom}
          bomLoading={bomLoading}
          stockLengthMm={stockLengthMm}
          onStockLengthChange={handleStockLengthChange}
          onDownloadBomCsv={handleExportBomCsv}
          error={error}
        />
      </main>
    </div>
  );
}
