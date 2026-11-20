import React, { useCallback, useEffect, useState } from "react";
import { DesignPanel } from "./components/DesignPanel";
import { MetricsPanel } from "./components/MetricsPanel";
import { PrinterViewport } from "./components/PrinterViewport";
import {
  CatalogResponse,
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

export default function App() {
  const [catalog, setCatalog] = useState<CatalogResponse | null>(null);
  const [selectedPreset, setSelectedPreset] = useState<string>("corexy_voron_style");
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

  const [viewMode, setViewMode] = useState<ViewMode>("assembly");
  const [selectedModeIdx, setSelectedModeIdx] = useState<number>(0);
  const [feaScale, setFeaScale] = useState<number>(600);
  const [layerLimit, setLayerLimit] = useState<number>(999);

  const runSimulate = useCallback(
    async (
      payload: { spec?: PrinterSpecData; yaml_text?: string },
      customGcode?: string,
      filOverride?: string
    ) => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch("/api/simulate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            ...payload,
            filament_id: filOverride ?? filamentId,
            gcode_text: customGcode && customGcode.trim().length > 0 ? customGcode : null,
          }),
        });
        if (!res.ok) {
          const errBody = await res.json().catch(() => ({ detail: res.statusText }));
          throw new Error(errBody.detail || "Simulation failed");
        }
        const data: SimulationResponse = await res.json();
        setSim(data);
        setSpec(data.spec);
        setYamlText(data.yaml_text);
        setLayerLimit(Math.max(0, data.virtual_print.layer_count - 1));
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setLoading(false);
      }
    },
    [filamentId]
  );

  // Automatically refresh BOM & Cut-List whenever simulation or stock length changes
  useEffect(() => {
    if (!sim) return;
    const controller = new AbortController();
    (async () => {
      setBomLoading(true);
      try {
        const res = await fetch("/api/bom", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ spec: sim.spec, stock_length_mm: stockLengthMm }),
          signal: controller.signal,
        });
        if (res.ok) {
          const data: PrinterBOMResponse = await res.json();
          setBom(data);
        }
      } catch (e) {
        if ((e as Error).name !== "AbortError") {
          setError(e instanceof Error ? e.message : String(e));
        }
      } finally {
        if (!controller.signal.aborted) {
          setBomLoading(false);
        }
      }
    })();
    return () => controller.abort();
  }, [sim, stockLengthMm]);

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
      await runSimulate({ spec: data.spec }, gcodeText);
    },
    [gcodeText, runSimulate]
  );

  useEffect(() => {
    (async () => {
      try {
        const res = await fetch("/api/catalog");
        const data: CatalogResponse = await res.json();
        setCatalog(data);
        const firstId = data.presets.find((p) => p.id === "corexy_voron_style")?.id || data.presets[0]?.id;
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

  const handleSpecChange = (next: PrinterSpecData) => {
    setSpec(next);
    setOptMessage(null);
    void runSimulate({ spec: next }, gcodeText);
  };

  const handleFilamentChange = (nextFil: string) => {
    setFilamentId(nextFil);
    if (spec) void runSimulate({ spec }, gcodeText, nextFil);
  };

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
      await runSimulate({ spec: data.best_spec }, gcodeText);
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
          <span className="brand-badge">PrintSim 0.4</span>
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
            onClick={() => spec && runSimulate({ spec }, gcodeText)}
          >
            {loading ? "Solving..." : "Re-Simulate"}
          </button>
        </div>
      </header>

      <main className="workspace">
        <DesignPanel
          catalog={catalog}
          selectedPreset={selectedPreset}
          onSelectPreset={loadPreset}
          filamentId={filamentId}
          onFilamentChange={handleFilamentChange}
          spec={spec}
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
          solidFea={solidFea}
          tetFea={tetFea}
          activeSolidEngine={activeSolidEngine}
          topoResult={topoResult}
          viewMode={viewMode}
          bom={bom}
          bomLoading={bomLoading}
          stockLengthMm={stockLengthMm}
          onStockLengthChange={setStockLengthMm}
          onDownloadBomCsv={handleExportBomCsv}
          error={error}
        />
      </main>
    </div>
  );
}
