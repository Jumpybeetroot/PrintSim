/// <reference lib="webworker" />

/**
 * Background Pyodide WebAssembly worker for running PrintSim's full Python
 * multiphysics engine (NumPy, SciPy, Pydantic, PyYAML) directly in the browser
 * on GitHub Pages without a backend server.
 */

const PYODIDE_INDEX_URL = "https://cdn.jsdelivr.net/pyodide/v0.27.5/full/";
const PYODIDE_MJS_URL = `${PYODIDE_INDEX_URL}pyodide.mjs`;

interface PyodideInstance {
  loadPackage(names: string | string[]): Promise<void>;
  pyimport(name: string): any;
  runPythonAsync(code: string): Promise<any>;
  globals: { set(key: string, value: unknown): void; get(key: string): any; delete(key: string): void };
}

type WorkerInMessage =
  | { type: "init"; wasmZipUrl: string }
  | { type: "request"; id: number; method: string; path: string; bodyJson: string | null };

let pyodide: PyodideInstance | null = null;
let initPromise: Promise<void> | null = null;
const installedExtras = new Set<string>();

function postStatus(state: "booting" | "ready" | "error", detail: string): void {
  self.postMessage({ type: "status", state, detail });
}

async function ensureInitialized(wasmZipUrl: string): Promise<void> {
  if (pyodide) return;
  if (initPromise) return initPromise;

  initPromise = (async () => {
    try {
      postStatus("booting", "Loading Python WebAssembly runtime…");
      const pyodideModule: any = await import(/* @vite-ignore */ PYODIDE_MJS_URL);
      const py: PyodideInstance = await pyodideModule.loadPyodide({
        indexURL: PYODIDE_INDEX_URL,
      });

      postStatus("booting", "Loading NumPy, SciPy & Pydantic…");
      await py.loadPackage(["numpy", "scipy", "pydantic", "pyyaml", "micropip"]);

      postStatus("booting", "Unpacking PrintSim solver engine…");
      const zipRes = await fetch(wasmZipUrl);
      if (!zipRes.ok) {
        throw new Error(`Failed to fetch PrintSim WASM bundle (${zipRes.status})`);
      }
      const zipBytes = new Uint8Array(await zipRes.arrayBuffer());
      py.globals.set("_printsim_zip_bytes", zipBytes);

      await py.runPythonAsync(`
import io, os, sys, zipfile
_buf = io.BytesIO(bytes(_printsim_zip_bytes.to_py()))
os.makedirs("/printsim", exist_ok=True)
with zipfile.ZipFile(_buf, "r") as _zf:
    _zf.extractall("/printsim")
if "/printsim" not in sys.path:
    sys.path.insert(0, "/printsim")
import printsim.wasm_dispatch as _wasm_dispatch
`);
      py.globals.delete("_printsim_zip_bytes");

      pyodide = py;
      postStatus("ready", "Browser Python FEA engine ready");
    } catch (err) {
      initPromise = null;
      const msg = err instanceof Error ? err.message : String(err);
      postStatus("error", msg);
      throw err;
    }
  })();

  return initPromise;
}

async function ensureOptionalWheel(pkg: "trimesh" | "ezdxf"): Promise<void> {
  if (!pyodide || installedExtras.has(pkg)) return;
  postStatus("booting", `Loading ${pkg} for browser export…`);
  await pyodide.runPythonAsync(`
import micropip
await micropip.install("${pkg}", deps=False)
`);
  installedExtras.add(pkg);
  postStatus("ready", "Browser Python FEA engine ready");
}

function requiredExtraForPath(path: string): "trimesh" | "ezdxf" | null {
  const clean = path.split("?", 1)[0].replace(/\/+$/, "");
  if (
    clean === "/api/export/stl" ||
    clean === "/api/fea/solid-stl" ||
    clean === "/api/fea/topology-optimize/stl"
  ) {
    return "trimesh";
  }
  if (clean === "/api/export/dxf") {
    return "ezdxf";
  }
  return null;
}

let cachedWasmZipUrl = "";

self.onmessage = async (event: MessageEvent<WorkerInMessage>) => {
  const msg = event.data;
  if (msg.type === "init") {
    cachedWasmZipUrl = msg.wasmZipUrl;
    try {
      await ensureInitialized(msg.wasmZipUrl);
    } catch {
      // status already emitted
    }
    return;
  }

  if (msg.type === "request") {
    try {
      if (!pyodide) {
        if (!cachedWasmZipUrl) {
          throw new Error("PrintSim browser worker not initialized");
        }
        await ensureInitialized(cachedWasmZipUrl);
      }
      const extra = requiredExtraForPath(msg.path);
      if (extra) {
        try {
          await ensureOptionalWheel(extra);
        } catch (wheelErr) {
          // If optional wheel install fails offline, let dispatch surface a clean error
          console.warn(`Optional Pyodide wheel '${extra}' failed:`, wheelErr);
        }
      }

      const py = pyodide!;
      py.globals.set("_req_method", msg.method);
      py.globals.set("_req_path", msg.path);
      py.globals.set("_req_body", msg.bodyJson);
      const resultJson: string = await py.runPythonAsync(`
import json as _json
_env = _wasm_dispatch.dispatch_api_request(_req_method, _req_path, _req_body)
_json.dumps(_env)
`);
      py.globals.delete("_req_method");
      py.globals.delete("_req_path");
      py.globals.delete("_req_body");

      self.postMessage({
        type: "response",
        id: msg.id,
        envelope: JSON.parse(resultJson),
      });
    } catch (err) {
      const detail = err instanceof Error ? err.message : String(err);
      self.postMessage({
        type: "response",
        id: msg.id,
        envelope: {
          status: 500,
          content_type: "application/json",
          body_text: JSON.stringify({ detail: `Browser engine error: ${detail}` }),
        },
      });
    }
  }
};
