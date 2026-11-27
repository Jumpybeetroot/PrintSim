import {
  AdjointLens,
  FeaLoadCase,
  ModalColorMode,
  ModalScaleMode,
  PrinterSpecData,
  ViewMode,
} from "../types";
import { isValidPrinterSpec } from "./shareSpec";

export type OptObjective = "balanced" | "speed" | "rigidity" | "budget";

export interface StudioPersistedSettings {
  version: 1;
  savedAt: string;
  spec: PrinterSpecData;
  selectedPreset: string;
  filamentId: string;
  toolheadRelPos: [number, number, number];
  panelMeshN: number;
  stockLengthMm: number;
  viewMode: ViewMode;
  selectedModeIdx: number;
  feaScale: number;
  feaLoadCase: FeaLoadCase;
  sensitivityLens: AdjointLens;
  modalColorMode: ModalColorMode;
  modalScaleMode: ModalScaleMode;
  showBedWarp: boolean;
  toolheadPunchIn?: boolean;
  layerLimit?: number;
  optObjective: OptObjective;
  gcodeText?: string;
}

export interface SavedBuildEntry {
  id: string;
  name: string;
  savedAt: string;
  settings: StudioPersistedSettings;
}

export interface StorageResult<T = void> {
  ok: boolean;
  data?: T;
  error?: string;
}

const AUTOSAVE_KEY = "printsim_studio_autosave_v1";
const SAVED_BUILDS_KEY = "printsim_studio_saved_builds_v1";
const SHOW_HINTS_KEY = "printsim_studio_show_hints_v1";
const ADVANCED_MODE_KEY = "printsim_studio_advanced_mode_v1";
const BASE_MACHINE_CHOSEN_KEY = "printsim_studio_base_machine_chosen_v1";
const VIEWPORT_CAMERA_KEY = "printsim_studio_viewport_camera_v1";
const VIEWPORT_UI_STATE_KEY = "printsim_studio_viewport_ui_v1";
const MAX_SAVED_BUILDS = 30;
const MAX_GCODE_CHARS = 100_000;

const VALID_VIEW_MODES = new Set<ViewMode>([
  "assembly",
  "fea_stress",
  "modal",
  "sensitivity",
  "toolpath",
  "solid_part",
  "topology",
]);

const VALID_LOAD_CASES = new Set<FeaLoadCase>(["xy", "x", "y", "gravity"]);
const VALID_LENSES = new Set<AdjointLens>([
  "print_impact",
  "compliance",
  "modal",
  "mass",
]);
const VALID_MODAL_COLOR_MODES = new Set<ModalColorMode>(["relative", "global"]);
const VALID_MODAL_SCALE_MODES = new Set<ModalScaleMode>(["impact", "shape"]);
const VALID_OPT_OBJECTIVES = new Set<OptObjective>([
  "balanced",
  "speed",
  "rigidity",
  "budget",
]);

function isPlainObject(val: unknown): val is Record<string, unknown> {
  return typeof val === "object" && val !== null && !Array.isArray(val);
}

function clamp(val: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, val));
}

function sanitizeToolheadPos(val: unknown): [number, number, number] {
  if (
    Array.isArray(val) &&
    val.length === 3 &&
    val.every((n) => typeof n === "number" && Number.isFinite(n))
  ) {
    return [
      Number(clamp(val[0], 0.1, 0.9).toFixed(2)),
      Number(clamp(val[1], 0.1, 0.9).toFixed(2)),
      Number(clamp(val[2], 0.1, 0.9).toFixed(2)),
    ];
  }
  return [0.5, 0.5, 0.5];
}

export function sanitizePersistedSettings(raw: unknown): StudioPersistedSettings | null {
  if (!isPlainObject(raw)) return null;
  if (!isValidPrinterSpec(raw.spec)) return null;

  const rawViewMode = raw.viewMode as ViewMode;
  const viewMode: ViewMode = VALID_VIEW_MODES.has(rawViewMode) ? rawViewMode : "assembly";
  const rawLoadCase = raw.feaLoadCase as FeaLoadCase;
  const feaLoadCase: FeaLoadCase = VALID_LOAD_CASES.has(rawLoadCase) ? rawLoadCase : "xy";
  const rawLens = raw.sensitivityLens as AdjointLens;
  const sensitivityLens: AdjointLens = VALID_LENSES.has(rawLens) ? rawLens : "print_impact";
  const rawModalColor = raw.modalColorMode as ModalColorMode;
  const modalColorMode: ModalColorMode = VALID_MODAL_COLOR_MODES.has(rawModalColor)
    ? rawModalColor
    : "relative";
  const rawModalScale = raw.modalScaleMode as ModalScaleMode;
  const modalScaleMode: ModalScaleMode = VALID_MODAL_SCALE_MODES.has(rawModalScale)
    ? rawModalScale
    : "impact";
  const rawOpt = raw.optObjective as OptObjective;
  const optObjective: OptObjective = VALID_OPT_OBJECTIVES.has(rawOpt) ? rawOpt : "balanced";

  const panelMeshN =
    typeof raw.panelMeshN === "number" && Number.isFinite(raw.panelMeshN)
      ? Math.round(clamp(raw.panelMeshN, 2, 8))
      : 3;

  const stockLengthMm = raw.stockLengthMm === 2000 ? 2000 : 1000;

  const selectedModeIdx =
    typeof raw.selectedModeIdx === "number" && Number.isFinite(raw.selectedModeIdx)
      ? Math.max(0, Math.round(raw.selectedModeIdx))
      : 0;

  const feaScale =
    typeof raw.feaScale === "number" && Number.isFinite(raw.feaScale)
      ? clamp(raw.feaScale, 10, 5000)
      : 600;

  const layerLimit =
    typeof raw.layerLimit === "number" && Number.isFinite(raw.layerLimit)
      ? Math.max(0, Math.round(raw.layerLimit))
      : 999;

  const gcodeText =
    typeof raw.gcodeText === "string" && raw.gcodeText.length <= MAX_GCODE_CHARS
      ? raw.gcodeText
      : "";

  const savedAt =
    typeof raw.savedAt === "string" && raw.savedAt.length > 0
      ? raw.savedAt
      : new Date().toISOString();

  return {
    version: 1,
    savedAt,
    spec: JSON.parse(JSON.stringify(raw.spec)) as PrinterSpecData,
    selectedPreset: typeof raw.selectedPreset === "string" ? raw.selectedPreset : "",
    filamentId:
      typeof raw.filamentId === "string" && raw.filamentId.trim().length > 0
        ? raw.filamentId
        : "pla",
    toolheadRelPos: sanitizeToolheadPos(raw.toolheadRelPos),
    panelMeshN,
    stockLengthMm,
    viewMode,
    selectedModeIdx,
    feaScale,
    feaLoadCase,
    sensitivityLens,
    modalColorMode,
    modalScaleMode,
    showBedWarp: Boolean(raw.showBedWarp),
    toolheadPunchIn: Boolean(raw.toolheadPunchIn),
    layerLimit,
    optObjective,
    gcodeText,
  };
}

function prepareForStorage(settings: StudioPersistedSettings): StudioPersistedSettings {
  const copy: StudioPersistedSettings = {
    ...settings,
    version: 1,
    savedAt: settings.savedAt || new Date().toISOString(),
  };
  if (copy.gcodeText && copy.gcodeText.length > MAX_GCODE_CHARS) {
    delete copy.gcodeText;
  }
  return copy;
}

export function loadAutoSavedSession(): StudioPersistedSettings | null {
  try {
    const raw = window.localStorage.getItem(AUTOSAVE_KEY);
    if (!raw) return null;
    const parsed: unknown = JSON.parse(raw);
    return sanitizePersistedSettings(parsed);
  } catch {
    return null;
  }
}

export function saveAutoSavedSession(settings: StudioPersistedSettings): StorageResult<string> {
  const prepared = prepareForStorage(settings);
  try {
    window.localStorage.setItem(AUTOSAVE_KEY, JSON.stringify(prepared));
    return { ok: true, data: prepared.savedAt };
  } catch {
    // Retry without G-code if quota was hit
    try {
      const compact = { ...prepared, gcodeText: "" };
      window.localStorage.setItem(AUTOSAVE_KEY, JSON.stringify(compact));
      return { ok: true, data: compact.savedAt };
    } catch (e) {
      return {
        ok: false,
        error: e instanceof Error ? e.message : "Browser storage quota exceeded",
      };
    }
  }
}

export function clearAutoSavedSession(): void {
  try {
    window.localStorage.removeItem(AUTOSAVE_KEY);
  } catch {
    // Ignore storage errors in restricted contexts
  }
}

export function loadSavedBuilds(): SavedBuildEntry[] {
  try {
    const raw = window.localStorage.getItem(SAVED_BUILDS_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    const entries: SavedBuildEntry[] = [];
    for (const item of parsed) {
      if (!isPlainObject(item)) continue;
      if (typeof item.id !== "string" || typeof item.name !== "string") continue;
      const sanitized = sanitizePersistedSettings(item.settings);
      if (!sanitized) continue;
      entries.push({
        id: item.id,
        name: item.name.trim() || sanitized.spec.name || "Saved Build",
        savedAt:
          typeof item.savedAt === "string" && item.savedAt.length > 0
            ? item.savedAt
            : sanitized.savedAt,
        settings: sanitized,
      });
    }
    return entries.sort((a, b) => b.savedAt.localeCompare(a.savedAt));
  } catch {
    return [];
  }
}

function writeSavedBuilds(entries: SavedBuildEntry[]): StorageResult<SavedBuildEntry[]> {
  const trimmed = entries.slice(0, MAX_SAVED_BUILDS);
  try {
    window.localStorage.setItem(SAVED_BUILDS_KEY, JSON.stringify(trimmed));
    return { ok: true, data: trimmed };
  } catch {
    try {
      const compact = trimmed.map((entry) => ({
        ...entry,
        settings: { ...entry.settings, gcodeText: "" },
      }));
      window.localStorage.setItem(SAVED_BUILDS_KEY, JSON.stringify(compact));
      return { ok: true, data: compact };
    } catch (e) {
      return {
        ok: false,
        error: e instanceof Error ? e.message : "Browser storage full — delete an old saved build",
      };
    }
  }
}

function createBuildId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return `build-${crypto.randomUUID()}`;
  }
  return `build-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

export function upsertSavedBuild(
  name: string,
  settings: StudioPersistedSettings,
  overwriteId?: string
): StorageResult<{ entry: SavedBuildEntry; builds: SavedBuildEntry[] }> {
  const trimmedName = name.trim() || settings.spec.name || "Untitled Build";
  const now = new Date().toISOString();
  const prepared = prepareForStorage({
    ...settings,
    savedAt: now,
    spec: {
      ...JSON.parse(JSON.stringify(settings.spec)),
      name: trimmedName,
    },
  });

  const existing = loadSavedBuilds();
  let targetIdx = -1;
  if (overwriteId) {
    targetIdx = existing.findIndex((b) => b.id === overwriteId);
  } else {
    targetIdx = existing.findIndex(
      (b) => b.name.toLowerCase() === trimmedName.toLowerCase()
    );
  }

  const entry: SavedBuildEntry = {
    id: targetIdx >= 0 ? existing[targetIdx].id : createBuildId(),
    name: trimmedName,
    savedAt: now,
    settings: prepared,
  };

  const next =
    targetIdx >= 0
      ? [entry, ...existing.filter((_, i) => i !== targetIdx)]
      : [entry, ...existing];

  const writeRes = writeSavedBuilds(next);
  if (!writeRes.ok) {
    return { ok: false, error: writeRes.error };
  }
  return {
    ok: true,
    data: { entry, builds: writeRes.data ?? next },
  };
}

export function deleteSavedBuild(id: string): StorageResult<SavedBuildEntry[]> {
  const next = loadSavedBuilds().filter((b) => b.id !== id);
  return writeSavedBuilds(next);
}

export function serializeSettingsFile(
  name: string,
  settings: StudioPersistedSettings
): string {
  return JSON.stringify(
    {
      format: "printsim_studio_settings",
      version: 1,
      name: name.trim() || settings.spec.name || "PrintSim Build",
      exportedAt: new Date().toISOString(),
      settings: prepareForStorage(settings),
    },
    null,
    2
  );
}

export function parseSettingsFile(
  rawText: string
): { name: string; settings: StudioPersistedSettings } | null {
  try {
    const parsed: unknown = JSON.parse(rawText);
    if (!isPlainObject(parsed)) return null;

    // Wrapped studio settings export format
    if (isPlainObject(parsed.settings)) {
      const sanitized = sanitizePersistedSettings(parsed.settings);
      if (!sanitized) return null;
      const name =
        typeof parsed.name === "string" && parsed.name.trim().length > 0
          ? parsed.name.trim()
          : sanitized.spec.name || "Imported Build";
      return { name, settings: sanitized };
    }

    // Direct StudioPersistedSettings object
    const direct = sanitizePersistedSettings(parsed);
    if (direct) {
      return {
        name: direct.spec.name || "Imported Build",
        settings: direct,
      };
    }

    // Bare PrinterSpecData JSON object
    if (isValidPrinterSpec(parsed)) {
      const specObj: PrinterSpecData = parsed;
      const fallback: StudioPersistedSettings = {
        version: 1,
        savedAt: new Date().toISOString(),
        spec: specObj,
        selectedPreset: "",
        filamentId: "pla",
        toolheadRelPos: [0.5, 0.5, 0.5],
        panelMeshN: 3,
        stockLengthMm: 1000,
        viewMode: "assembly",
        selectedModeIdx: 0,
        feaScale: 600,
        feaLoadCase: "xy",
        sensitivityLens: "print_impact",
        modalColorMode: "relative",
        modalScaleMode: "impact",
        showBedWarp: false,
        optObjective: "balanced",
        gcodeText: "",
      };
      return {
        name: specObj.name || "Imported Printer Spec",
        settings: fallback,
      };
    }

    return null;
  } catch {
    return null;
  }
}

export function loadShowHintsPreference(): boolean {
  try {
    return window.localStorage.getItem(SHOW_HINTS_KEY) === "1";
  } catch {
    return false;
  }
}

export function saveShowHintsPreference(show: boolean): void {
  try {
    window.localStorage.setItem(SHOW_HINTS_KEY, show ? "1" : "0");
  } catch {
    // Ignore storage errors
  }
}

export function loadAdvancedModePreference(): boolean {
  try {
    return window.localStorage.getItem(ADVANCED_MODE_KEY) === "1";
  } catch {
    return false;
  }
}

export function saveAdvancedModePreference(enabled: boolean): void {
  try {
    window.localStorage.setItem(ADVANCED_MODE_KEY, enabled ? "1" : "0");
  } catch {
    // Ignore storage errors
  }
}

export function loadBaseMachineChosenPreference(): boolean {
  try {
    return window.localStorage.getItem(BASE_MACHINE_CHOSEN_KEY) === "1";
  } catch {
    return false;
  }
}

export function saveBaseMachineChosenPreference(chosen: boolean): void {
  try {
    window.localStorage.setItem(BASE_MACHINE_CHOSEN_KEY, chosen ? "1" : "0");
  } catch {
    // Ignore storage errors
  }
}

export interface ViewportCameraPose {
  position: [number, number, number];
  target: [number, number, number];
}

function isValidVec3Tuple(v: unknown): v is [number, number, number] {
  return (
    Array.isArray(v) &&
    v.length === 3 &&
    v.every((n) => typeof n === "number" && Number.isFinite(n) && Math.abs(n) <= 200)
  );
}

function isValidCameraPose(pose: unknown): pose is ViewportCameraPose {
  if (!pose || typeof pose !== "object") return false;
  const obj = pose as Record<string, unknown>;
  if (!isValidVec3Tuple(obj.position) || !isValidVec3Tuple(obj.target)) return false;
  const dx = obj.position[0] - obj.target[0];
  const dy = obj.position[1] - obj.target[1];
  const dz = obj.position[2] - obj.target[2];
  const dist = Math.hypot(dx, dy, dz);
  return Number.isFinite(dist) && dist >= 0.05 && dist <= 180;
}

export function loadViewportCameraPose(): ViewportCameraPose | null {
  try {
    const raw = window.localStorage.getItem(VIEWPORT_CAMERA_KEY);
    if (!raw) return null;
    const parsed: unknown = JSON.parse(raw);
    if (!isValidCameraPose(parsed)) return null;
    return {
      position: [parsed.position[0], parsed.position[1], parsed.position[2]],
      target: [parsed.target[0], parsed.target[1], parsed.target[2]],
    };
  } catch {
    return null;
  }
}

export function saveViewportCameraPose(pose: ViewportCameraPose): void {
  if (!isValidCameraPose(pose)) return;
  try {
    window.localStorage.setItem(VIEWPORT_CAMERA_KEY, JSON.stringify(pose));
  } catch {
    // Ignore storage errors
  }
}

export type ViewportFeaLayerId = "panels" | "frame" | "gantry" | "toolhead" | "bed";

const VALID_VIEWPORT_FEA_LAYERS = new Set<ViewportFeaLayerId>([
  "panels",
  "frame",
  "gantry",
  "toolhead",
  "bed",
]);

export interface ViewportUiState {
  hiddenFeaLayers: ViewportFeaLayerId[];
  hiddenFeaParts: string[];
  isolateToolheadModal: boolean;
  toolheadExplode: boolean;
  modalAmpScale: number;
  modalPaused: boolean;
  ecoDpr: boolean;
  showHudDetails: boolean;
  showCarvedGhost: boolean;
}

const DEFAULT_VIEWPORT_UI_STATE: ViewportUiState = {
  hiddenFeaLayers: [],
  hiddenFeaParts: [],
  isolateToolheadModal: false,
  toolheadExplode: false,
  modalAmpScale: 1.0,
  modalPaused: false,
  ecoDpr: false,
  showHudDetails: false,
  showCarvedGhost: true,
};

export function loadViewportUiState(): ViewportUiState {
  try {
    const raw = window.localStorage.getItem(VIEWPORT_UI_STATE_KEY);
    if (!raw) return { ...DEFAULT_VIEWPORT_UI_STATE };
    const parsed: unknown = JSON.parse(raw);
    if (!isPlainObject(parsed)) return { ...DEFAULT_VIEWPORT_UI_STATE };

    const hiddenFeaLayers = Array.isArray(parsed.hiddenFeaLayers)
      ? Array.from(
          new Set(
            parsed.hiddenFeaLayers.filter(
              (v): v is ViewportFeaLayerId =>
                typeof v === "string" && VALID_VIEWPORT_FEA_LAYERS.has(v as ViewportFeaLayerId)
            )
          )
        )
      : [];

    const hiddenFeaParts = Array.isArray(parsed.hiddenFeaParts)
      ? Array.from(
          new Set(
            parsed.hiddenFeaParts
              .filter((v): v is string => typeof v === "string" && v.length > 0 && v.length <= 200)
              .slice(0, 200)
          )
        )
      : [];

    const modalAmpScale =
      typeof parsed.modalAmpScale === "number" && Number.isFinite(parsed.modalAmpScale)
        ? clamp(parsed.modalAmpScale, 0.1, 5.0)
        : 1.0;

    return {
      hiddenFeaLayers,
      hiddenFeaParts,
      isolateToolheadModal: Boolean(parsed.isolateToolheadModal),
      toolheadExplode: Boolean(parsed.toolheadExplode),
      modalAmpScale,
      modalPaused: Boolean(parsed.modalPaused),
      ecoDpr: Boolean(parsed.ecoDpr),
      showHudDetails: Boolean(parsed.showHudDetails),
      showCarvedGhost:
        typeof parsed.showCarvedGhost === "boolean" ? parsed.showCarvedGhost : true,
    };
  } catch {
    return { ...DEFAULT_VIEWPORT_UI_STATE };
  }
}

export function saveViewportUiState(state: ViewportUiState): void {
  try {
    window.localStorage.setItem(VIEWPORT_UI_STATE_KEY, JSON.stringify(state));
  } catch {
    // Ignore storage errors
  }
}
