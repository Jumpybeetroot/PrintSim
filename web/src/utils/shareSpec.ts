import { PrinterSpecData } from "../types";

const HASH_PREFIX = "#spec=";
const VALID_ARCHITECTURES = new Set(["corexy", "cartesian_bedslinger", "delta"]);

function bytesToBase64Url(bytes: Uint8Array): string {
  const chunkSize = 0x8000;
  const parts: string[] = [];
  for (let i = 0; i < bytes.length; i += chunkSize) {
    parts.push(String.fromCharCode(...bytes.subarray(i, i + chunkSize)));
  }
  return btoa(parts.join(""))
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/g, "");
}

function base64UrlToBytes(b64url: string): Uint8Array {
  const normalized = b64url.replace(/-/g, "+").replace(/_/g, "/");
  const padLen = (4 - (normalized.length % 4)) % 4;
  const padded = normalized + "=".repeat(padLen);
  const bin = atob(padded);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) {
    bytes[i] = bin.charCodeAt(i);
  }
  return bytes;
}

function isPlainObject(val: unknown): val is Record<string, unknown> {
  return typeof val === "object" && val !== null && !Array.isArray(val);
}

function isValidPrinterSpec(val: unknown): val is PrinterSpecData {
  if (!isPlainObject(val)) return false;
  if (typeof val.architecture !== "string" || !VALID_ARCHITECTURES.has(val.architecture)) {
    return false;
  }
  if (!isPlainObject(val.frame)) return false;
  if (
    typeof val.frame.width_mm !== "number" ||
    !Number.isFinite(val.frame.width_mm) ||
    typeof val.frame.depth_mm !== "number" ||
    !Number.isFinite(val.frame.depth_mm) ||
    typeof val.frame.height_mm !== "number" ||
    !Number.isFinite(val.frame.height_mm)
  ) {
    return false;
  }
  if (!isPlainObject(val.gantry) || !isPlainObject(val.gantry.belt)) return false;
  if (!isPlainObject(val.kinematics) || !isPlainObject(val.kinematics.xy_stepper)) return false;
  if (
    !isPlainObject(val.toolhead) ||
    !isPlainObject(val.toolhead.hotend) ||
    !isPlainObject(val.toolhead.extruder)
  ) {
    return false;
  }
  if (!isPlainObject(val.bed)) return false;
  return true;
}

/**
 * Encode a PrinterSpecData object into a UTF-8 safe base64url URL hash prefixed with `#spec=`.
 */
export function encodeSpecToHash(spec: PrinterSpecData): string {
  const json = JSON.stringify(spec);
  const bytes = new TextEncoder().encode(json);
  return `${HASH_PREFIX}${bytesToBase64Url(bytes)}`;
}

/**
 * Decode a `#spec=...` URL hash into a structurally validated PrinterSpecData, or null if invalid.
 */
export function decodeSpecFromHash(hash: string): PrinterSpecData | null {
  if (!hash || typeof hash !== "string") return null;
  const trimmed = hash.trim();
  const normalized = trimmed.startsWith("#") ? trimmed : `#${trimmed}`;
  if (!normalized.startsWith(HASH_PREFIX)) return null;

  const rawPayload = normalized.slice(HASH_PREFIX.length).split("&")[0];
  if (!rawPayload) return null;

  try {
    const bytes = base64UrlToBytes(rawPayload);
    const json = new TextDecoder().decode(bytes);
    const parsed: unknown = JSON.parse(json);
    if (!isValidPrinterSpec(parsed)) return null;
    return parsed;
  } catch {
    return null;
  }
}

/**
 * Build a full shareable URL embedding the given printer spec in the URL hash.
 */
export function buildShareUrl(spec: PrinterSpecData): string {
  const hash = encodeSpecToHash(spec);
  if (typeof window === "undefined") return hash;
  return `${window.location.origin}${window.location.pathname}${window.location.search}${hash}`;
}
