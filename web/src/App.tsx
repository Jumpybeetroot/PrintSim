import React, { useCallback, useEffect, useState } from "react";
import { DesignPanel } from "./components/DesignPanel";
import { MetricsPanel } from "./components/MetricsPanel";
import { PrinterViewport } from "./components/PrinterViewport";
import { CatalogResponse, PrinterSpecData, SimulationResponse, SolidFEAResponse, ViewMode } from "./types";

export default function App() {
  const [catalog, setCatalog] = useState<CatalogResponse | null>(null);
  const [selectedPreset, setSelectedPreset] = useState<string>("corexy_voron_style");
  const [filamentId, setFilamentId] = useState<string>("pla");
  const [spec, setSpec] = useState<PrinterSpecData | null>(null);
  const [yamlText, setYamlText] = useState<string>("");
  const [gcodeText, setGcodeText] = useState<string>("");
  const [sim, setSim] = useState<SimulationResponse | null>(null);
  const [solidFea, setSolidFea] = useState<SolidFEAResponse | null>(null);
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

  const runPartFea = useCallback(
    async (params: {
      preset: string;
      materialId: string;
      forceN: [number, number, number];
      fdmAnisotropic: boolean;
    }) => {
      setLoading(true);
      setError(null);
      try {
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
          throw new Error(errBody.detail || "Solid FEA failed");
        }
        const data: SolidFEAResponse = await res.json();
        setSolidFea(data);
        setViewMode("solid_part");
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
        const bytes = new Uint8Array(buf);
        let bin = "";
        for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
        const b64 = btoa(bin);
        const ext = params.file.name.split(".").pop()?.toLowerCase() || "stl";

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
        setViewMode("solid_part");
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setLoading(false);
      }
    },
    []
  );

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
        // Preload default solid bracket FEA so switching to 3D Solid Part tab is instant
        const pRes = await fetch("/api/fea/solid-part", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ preset: "corner_gusset", material_id: "abs_printed" }),
        });
        if (pRes.ok) setSolidFea(await pRes.json());
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

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-badge">PrintSim 0.2</span>
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
          loading={loading}
        />

        <PrinterViewport
          sim={sim}
          solidFea={solidFea}
          viewMode={viewMode}
          onViewModeChange={setViewMode}
          selectedModeIdx={selectedModeIdx}
          onSelectModeIdx={setSelectedModeIdx}
          feaScale={feaScale}
          onFeaScaleChange={setFeaScale}
          layerLimit={layerLimit}
          onLayerLimitChange={setLayerLimit}
        />

        <MetricsPanel sim={sim} solidFea={solidFea} error={error} />
      </main>
    </div>
  );
}
