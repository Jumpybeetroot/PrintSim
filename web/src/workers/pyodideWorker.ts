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
  runPython(code: string): any;
  runPythonAsync(code: string): Promise<any>;
  globals: { set(key: string, value: unknown): void; get(key: string): any; delete(key: string): void };
}

type RequestMessage = {
  type: "request";
  id: number;
  method: string;
  path: string;
  bodyJson: string | null;
};

type WorkerInMessage =
  | { type: "init"; wasmZipUrl: string }
  | RequestMessage
  | { type: "cancel"; id: number };

let pyodide: PyodideInstance | null = null;
let dispatchFn: ((method: string, path: string, bodyJson: string | null) => string) | null = null;
let initPromise: Promise<void> | null = null;
const installedExtras = new Set<string>();
let cachedWasmZipUrl = "";
// Serialize requests cleanly so concurrent /api/simulate + /api/bom calls never race.
// An explicit FIFO (instead of a promise chain) lets a "cancel" message drop a request that is
// still waiting, so superseded simulations from rapid edits never occupy the single Python thread.
const pendingRequests: RequestMessage[] = [];
let pumpScheduled = false;
let pumpRunning = false;

function postStatus(state: "booting" | "ready" | "error", detail: string): void {
  self.postMessage({ type: "status", state, detail });
}

async function ensureInitialized(wasmZipUrl: string): Promise<void> {
  if (pyodide && dispatchFn) return;
  if (initPromise) return initPromise;

  initPromise = (async () => {
    try {
      postStatus("booting", "Loading Python WebAssembly runtime…");
      // Start the PrintSim bundle download immediately so it overlaps the Pyodide runtime and
      // package downloads instead of waiting behind them.
      const zipBytesPromise: Promise<Uint8Array> = fetch(wasmZipUrl).then(async (zipRes) => {
        if (!zipRes.ok) {
          throw new Error(`Failed to fetch PrintSim WASM bundle (${zipRes.status})`);
        }
        return new Uint8Array(await zipRes.arrayBuffer());
      });
      // Avoid an unhandled rejection if the runtime load fails first; the await below rethrows.
      zipBytesPromise.catch(() => undefined);

      const pyodideModule: any = await import(/* @vite-ignore */ PYODIDE_MJS_URL);
      const py: PyodideInstance = await pyodideModule.loadPyodide({
        indexURL: PYODIDE_INDEX_URL,
      });

      postStatus("booting", "Loading NumPy, SciPy & Pydantic…");
      // micropip is only needed for optional export wheels; it loads on demand in
      // ensureOptionalWheel() rather than on every engine boot.
      await py.loadPackage(["numpy", "scipy", "pydantic", "pyyaml"]);

      postStatus("booting", "Unpacking PrintSim solver engine…");
      const zipBytes = await zipBytesPromise;
      py.globals.set("_printsim_zip_bytes", zipBytes);

      await py.runPythonAsync(`
import io, json, os, sys, zipfile
_buf = io.BytesIO(bytes(_printsim_zip_bytes.to_py()))
os.makedirs("/printsim", exist_ok=True)
with zipfile.ZipFile(_buf, "r") as _zf:
    _zf.extractall("/printsim")
if "/printsim" not in sys.path:
    sys.path.insert(0, "/printsim")
import printsim.wasm_dispatch as _wasm_dispatch

def _printsim_dispatch_json(method, path, body_json):
    env = _wasm_dispatch.dispatch_api_request(method, path, body_json)
    return json.dumps(env)
`);
      py.globals.delete("_printsim_zip_bytes");

      dispatchFn = py.globals.get("_printsim_dispatch_json");
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
  await pyodide.loadPackage("micropip");
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

async function handleRequest(msg: RequestMessage): Promise<void> {
  try {
    if (!pyodide || !dispatchFn) {
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
        console.warn(`Optional Pyodide wheel '${extra}' failed:`, wheelErr);
      }
    }

    const resultJson = dispatchFn!(msg.method, msg.path, msg.bodyJson);
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

self.onmessage = (event: MessageEvent<WorkerInMessage>) => {
  const msg = event.data;
  if (msg.type === "init") {
    cachedWasmZipUrl = msg.wasmZipUrl;
    void ensureInitialized(msg.wasmZipUrl).catch(() => {
      // status already emitted
    });
    return;
  }

  if (msg.type === "request") {
    pendingRequests.push(msg);
    schedulePump();
    return;
  }

  if (msg.type === "cancel") {
    // Only requests that have not started can be dropped: Python runs synchronously, so a
    // request that is already executing finishes (its response is ignored by the bridge).
    const idx = pendingRequests.findIndex((req) => req.id === msg.id);
    if (idx >= 0) {
      pendingRequests.splice(idx, 1);
      self.postMessage({ type: "cancelled", id: msg.id });
    }
  }
};

function schedulePump(): void {
  if (pumpScheduled || pumpRunning) return;
  pumpScheduled = true;
  // A macrotask hop (not a microtask) so "cancel" / newer "request" messages that arrived while
  // the previous request held the thread are delivered before the next request is picked.
  setTimeout(() => {
    pumpScheduled = false;
    void pumpOne();
  }, 0);
}

async function pumpOne(): Promise<void> {
  const next = pendingRequests.shift();
  if (!next) return;
  pumpRunning = true;
  try {
    await handleRequest(next);
  } finally {
    pumpRunning = false;
    if (pendingRequests.length > 0) {
      schedulePump();
    }
  }
}
