/**
 * Hybrid GitHub Pages API Bridge.
 *
 * Active ONLY when built with `VITE_STATIC_PAGES=true` (GitHub Pages deployment).
 * Local `printsim serve` and `npm run dev` builds bypass this module completely.
 *
 * Hybrid execution strategy:
 *   1. Instant (<1 s): Catalog, hardware GLB assets, preset specs, pre-baked preset
 *      simulations & BOMs, and default Hex8 solid FEA are served directly from static
 *      files under `${BASE_URL}pages-data/` and `${BASE_URL}api/assets/hardware/`.
 *   2. Live In-Browser Engine: A background Pyodide Web Worker boots NumPy, SciPy,
 *      Pydantic, PyYAML, and PrintSim's Python solver so any parameter edit, custom YAML,
 *      ADXL calibration/fit, Pareto/AI optimization, Solid Hex8 FEA, SIMP topology opt,
 *      and STL/DXF/CSV export runs 100% live in the browser.
 */

export const IS_STATIC_PAGES = import.meta.env.VITE_STATIC_PAGES === "true";

export type PagesEngineState = "idle" | "booting" | "ready" | "error";

export interface PagesEngineStatus {
  state: PagesEngineState;
  detail: string;
}

interface WorkerEnvelope {
  status: number;
  content_type?: string;
  headers?: Record<string, string>;
  body_text?: string;
  body_base64?: string;
}

let currentStatus: PagesEngineStatus = {
  state: IS_STATIC_PAGES ? "booting" : "idle",
  detail: IS_STATIC_PAGES ? "Pre-baked presets instant · booting Python WASM…" : "",
};
const statusListeners = new Set<(s: PagesEngineStatus) => void>();

export function getPagesEngineStatus(): PagesEngineStatus {
  return currentStatus;
}

export function subscribePagesEngineStatus(
  listener: (status: PagesEngineStatus) => void
): () => void {
  statusListeners.add(listener);
  listener(currentStatus);
  return () => {
    statusListeners.delete(listener);
  };
}

function setStatus(next: PagesEngineStatus): void {
  currentStatus = next;
  statusListeners.forEach((fn) => fn(next));
}

function baseUrlPrefix(): string {
  const raw = import.meta.env.BASE_URL || "/";
  return raw.endsWith("/") ? raw : `${raw}/`;
}

function decodeBase64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i += 1) {
    out[i] = bin.charCodeAt(i);
  }
  return out;
}

function envelopeToResponse(env: WorkerEnvelope): Response {
  const headers = new Headers(env.headers || {});
  if (env.content_type && !headers.has("Content-Type")) {
    headers.set("Content-Type", env.content_type);
  }
  if (env.body_base64 !== undefined) {
    const bytes = decodeBase64ToBytes(env.body_base64);
    const ab = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
    return new Response(ab, { status: env.status, headers });
  }
  return new Response(env.body_text ?? "", { status: env.status, headers });
}

function jsonResponse(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

let installed = false;

export function installPagesApiBridge(): void {
  if (!IS_STATIC_PAGES || installed || typeof window === "undefined") {
    return;
  }
  installed = true;

  const origFetch = window.fetch.bind(window);
  const base = baseUrlPrefix();
  const pagesDataUrl = (rel: string) => `${base}pages-data/${rel}`;

  // Canonical spec string -> presetId cache so untouched presets serve in <50ms
  const presetSpecToId = new Map<string, string>();
  const presetYamlToId = new Map<string, string>();

  // Spawn background Pyodide Web Worker immediately
  let worker: Worker | null = null;
  let nextReqId = 1;
  const pending = new Map<
    number,
    { resolve: (env: WorkerEnvelope) => void; reject: (err: Error) => void }
  >();

  const ensureWorker = (): Worker => {
    if (worker) return worker;
    worker = new Worker(new URL("../workers/pyodideWorker.ts", import.meta.url), {
      type: "module",
    });
    worker.onmessage = (event: MessageEvent<any>) => {
      const msg = event.data;
      if (msg?.type === "status") {
        setStatus({ state: msg.state, detail: msg.detail || "" });
        return;
      }
      if (msg?.type === "response") {
        const entry = pending.get(msg.id);
        if (entry) {
          pending.delete(msg.id);
          entry.resolve(msg.envelope);
        }
      }
    };
    worker.onerror = (ev) => {
      const detail = ev.message || "Pyodide worker crashed";
      setStatus({ state: "error", detail });
      pending.forEach((p) => p.reject(new Error(detail)));
      pending.clear();
    };
    const wasmZipAbsUrl = new URL(pagesDataUrl("printsim_wasm.zip"), window.location.href).href;
    worker.postMessage({ type: "init", wasmZipUrl: wasmZipAbsUrl });
    return worker;
  };

  // Start booting Pyodide right away in background
  ensureWorker();

  const callWorker = (method: string, path: string, bodyJson: string | null): Promise<Response> => {
    const w = ensureWorker();
    const id = nextReqId++;
    return new Promise<Response>((resolve, reject) => {
      pending.set(id, {
        resolve: (env) => resolve(envelopeToResponse(env)),
        reject,
      });
      w.postMessage({ type: "request", id, method, path, bodyJson });
    });
  };

  const extractPathname = (input: RequestInfo | URL): string | null => {
    try {
      const rawUrl =
        typeof input === "string"
          ? input
          : input instanceof URL
          ? input.href
          : input.url;
      if (rawUrl.startsWith("/api/")) return rawUrl;
      const parsed = new URL(rawUrl, window.location.href);
      if (parsed.origin === window.location.origin && parsed.pathname.startsWith("/api/")) {
        return `${parsed.pathname}${parsed.search}`;
      }
      return null;
    } catch {
      return null;
    }
  };

  const isDefaultSimulatePayload = (parsed: Record<string, any>): boolean => {
    const fil = parsed.filament_id ?? "pla";
    if (fil !== "pla") return false;
    if (parsed.gcode_text) return false;
    if (parsed.accel_override_mm_s2 != null) return false;
    if (parsed.panel_mesh_n != null && parsed.panel_mesh_n !== 3) return false;
    if (parsed.bed_mesh_n != null && parsed.bed_mesh_n !== 8) return false;
    const pos = parsed.toolhead_rel_pos;
    if (Array.isArray(pos) && pos.length === 3) {
      if (
        Math.abs(Number(pos[0]) - 0.5) > 1e-6 ||
        Math.abs(Number(pos[1]) - 0.5) > 1e-6 ||
        Math.abs(Number(pos[2]) - 0.5) > 1e-6
      ) {
        return false;
      }
    }
    return true;
  };

  const matchPrebakedPresetId = (parsed: Record<string, any>): string | null => {
    if (parsed.spec && typeof parsed.spec === "object") {
      const hit = presetSpecToId.get(JSON.stringify(parsed.spec));
      if (hit) return hit;
    }
    if (typeof parsed.yaml_text === "string" && parsed.yaml_text.trim()) {
      const hit = presetYamlToId.get(parsed.yaml_text.trim());
      if (hit) return hit;
    }
    return null;
  };

  window.fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const apiPath = extractPathname(input);
    if (!apiPath) {
      return origFetch(input, init);
    }

    const cleanPath = apiPath.split("?", 1)[0].replace(/\/+$/, "");
    const method = (
      init?.method ||
      (input instanceof Request ? input.method : "GET")
    ).toUpperCase();

    // 1. Static PBR hardware GLB meshes (Three.js GLTFLoader)
    if (method === "GET" && cleanPath.startsWith("/api/assets/hardware/") && cleanPath.endsWith("/visual.glb")) {
      const rel = cleanPath.replace(/^\/+/, "");
      return origFetch(`${base}${rel}`, init);
    }

    // 2. Live MCP Studio Session Bridge (no-op in static browser deployment)
    if (cleanPath.startsWith("/api/studio/")) {
      return jsonResponse({ ok: true, commands: [], runtime: "static-pages" });
    }

    // 3. Instant pre-baked catalog & hardware asset metadata
    if (method === "GET" && cleanPath === "/api/catalog") {
      return origFetch(pagesDataUrl("catalog.json"));
    }
    if (method === "GET" && cleanPath === "/api/assets/hardware") {
      return origFetch(pagesDataUrl("hardware_assets.json"));
    }
    if (method === "GET" && cleanPath.startsWith("/api/presets/")) {
      const presetId = cleanPath.slice("/api/presets/".length);
      const res = await origFetch(pagesDataUrl(`presets/${presetId}.json`));
      if (res.ok) {
        const text = await res.text();
        try {
          const data = JSON.parse(text);
          if (data?.spec) {
            presetSpecToId.set(JSON.stringify(data.spec), presetId);
          }
          if (typeof data?.yaml_text === "string") {
            presetYamlToId.set(data.yaml_text.trim(), presetId);
          }
        } catch {
          // ignore cache parse errors
        }
        return new Response(text, {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      return res;
    }

    // Read JSON request body if present
    let bodyText: string | null = null;
    if (typeof init?.body === "string") {
      bodyText = init.body;
    } else if (input instanceof Request && method !== "GET" && method !== "HEAD") {
      bodyText = await input.clone().text();
    }

    // 4. Instant pre-baked startup Tet10 fallback -> Hex8 default bracket
    if (method === "POST" && (cleanPath === "/api/fea/tet-part" || cleanPath === "/api/fea/tet-step")) {
      return jsonResponse(
        {
          detail:
            "Curved Tet10 FEA (Gmsh + scikit-fem) requires a local PrintSim backend (`uv run printsim serve`); using in-browser QM6-3D Hex8 FEA.",
        },
        400
      );
    }
    if (method === "POST" && cleanPath === "/api/fea/solid-part" && bodyText) {
      try {
        const parsed = JSON.parse(bodyText);
        if (
          (parsed.preset ?? "corner_gusset") === "corner_gusset" &&
          (parsed.material_id ?? "abs_printed") === "abs_printed" &&
          parsed.force_n == null &&
          parsed.fdm_anisotropic == null
        ) {
          const cached = await origFetch(pagesDataUrl("solid_part_default.json"));
          if (cached.ok) return cached;
        }
      } catch {
        // fall through to live worker
      }
    }

    // 5. Instant pre-baked simulation & BOM for unmodified built-in presets
    if (method === "POST" && cleanPath === "/api/simulate" && bodyText) {
      try {
        const parsed = JSON.parse(bodyText);
        if (isDefaultSimulatePayload(parsed)) {
          const presetId = matchPrebakedPresetId(parsed);
          if (presetId) {
            const cached = await origFetch(pagesDataUrl(`simulations/${presetId}.json`));
            if (cached.ok) return cached;
          }
        }
      } catch {
        // fall through to live worker
      }
    }
    if (method === "POST" && cleanPath === "/api/bom" && bodyText) {
      try {
        const parsed = JSON.parse(bodyText);
        const stock = parsed.stock_length_mm ?? 1000.0;
        const kerf = parsed.kerf_mm ?? 3.0;
        if (Math.abs(Number(stock) - 1000.0) < 1e-6 && Math.abs(Number(kerf) - 3.0) < 1e-6) {
          const presetId = matchPrebakedPresetId(parsed);
          if (presetId) {
            const cached = await origFetch(pagesDataUrl(`boms/${presetId}.json`));
            if (cached.ok) return cached;
          }
        }
      } catch {
        // fall through to live worker
      }
    }

    // 6. All live edits, optimizations, ADXL fits, FEA, and exports -> Pyodide Web Worker
    return callWorker(method, apiPath, bodyText);
  };
}
