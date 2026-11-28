import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { ModeShapeData, SubsystemKey } from "../types";
import { SCALE } from "./cadGlbCache";

export interface InspectedPartInfo {
  key: string;
  name: string;
  subsystem: SubsystemKey;
  partGroupId: string;
  role: string;
  material?: string;
  profile?: string;
  complianceSharePct?: number;
  stationComplianceSharePct?: number;
  bracketFlexUm?: number;
  bracketTwistMrad?: number;
  transmittedForceN?: number;
  rotStiffnessNmRad?: number;
  torsionStiffnessNmRad?: number;
  unitMassG?: number;
  totalCount?: number;
  nodeDispUm?: number;
  heatScore?: number;
  stressMpa?: number;
  yieldMpa?: number;
  estHzGain?: number;
  hzPer100g?: number;
  whyItHurts?: string;
  actionHint?: string;
  bedFlatnessUm?: number;
  bedTacoUm?: number;
  bedModeHz?: number;
  bedRisk?: "low" | "moderate" | "high" | "critical";
  localFlexUm?: number;
  partStiffnessNPerMm?: number;
  modalFlexPct?: number;
  kineticSharePct?: number;
  partSource?: "cad_3d_recovery" | "stage_fea";
  isStructural?: boolean;
  /** Canonical backend toolhead part key (`th_part_<id>`) for stage-level hide/highlight */
  toolheadPartKey?: string;
  /** CAD asset metadata for single-object in-part 3D FEA inspection */
  cadAssetId?: string;
  cadGlbUrl?: string;
  partFieldKey?: string;
  cadPort0Mm?: [number, number, number];
  cadPort1Mm?: [number, number, number];
}

export interface ModalVisualSeverity {
  score: number;
  impact01: number;
  motionScale: number;
  colorScale: number;
  tier: "critical" | "high" | "moderate" | "low" | "negligible";
  label: "Critical" | "High" | "Moderate" | "Low" | "Negligible";
  segments: number;
  pillClass: "bad" | "warn" | "info" | "neutral";
}

export function getModalVisualSeverity(mode: ModeShapeData | undefined): ModalVisualSeverity {
  let score = 100;
  if (mode?.print_impact_score != null && Number.isFinite(mode.print_impact_score)) {
    score = Math.max(0, Math.min(100, mode.print_impact_score));
  } else if (mode?.print_impact_level === "critical") {
    score = 90;
  } else if (mode?.print_impact_level === "high") {
    score = 65;
  } else if (mode?.print_impact_level === "moderate") {
    score = 40;
  } else if (mode?.print_impact_level === "low") {
    score = 10;
  } else if (mode?.severity === "critical") {
    score = 85;
  } else if (mode?.severity === "moderate") {
    score = 45;
  } else if (mode?.severity === "stiff") {
    score = 10;
  }

  const impact01 = Math.max(0, Math.min(1, score / 100));
  const motionScale = 0.08 + 0.92 * Math.pow(impact01, 0.72);
  const colorScale = 0.14 + 0.86 * Math.pow(impact01, 0.65);

  const level = mode?.print_impact_level;
  if (level === "critical" || score >= 75) {
    return {
      score,
      impact01,
      motionScale,
      colorScale,
      tier: "critical",
      label: "Critical",
      segments: 4,
      pillClass: "bad",
    };
  }
  if (level === "high" || score >= 50) {
    return {
      score,
      impact01,
      motionScale,
      colorScale,
      tier: "high",
      label: "High",
      segments: 3,
      pillClass: "bad",
    };
  }
  if (level === "moderate" || score >= 25) {
    return {
      score,
      impact01,
      motionScale,
      colorScale,
      tier: "moderate",
      label: "Moderate",
      segments: 2,
      pillClass: "warn",
    };
  }
  if (score >= 5) {
    return {
      score,
      impact01,
      motionScale,
      colorScale,
      tier: "low",
      label: "Low",
      segments: 1,
      pillClass: "info",
    };
  }
  return {
    score,
    impact01,
    motionScale,
    colorScale,
    tier: "negligible",
    label: "Negligible",
    segments: 0,
    pillClass: "neutral",
  };
}

export function disposeGroupChildren(group: THREE.Group): void {
  const geometries = new Set<THREE.BufferGeometry>();
  const materials = new Set<THREE.Material>();
  while (group.children.length > 0) {
    const obj = group.children[0];
    group.remove(obj);
    obj.traverse((child) => {
      const geo = (child as THREE.Mesh).geometry;
      if (geo) geometries.add(geo);
      const mat = (child as THREE.Mesh).material;
      if (Array.isArray(mat)) {
        mat.forEach((m) => materials.add(m));
      } else if (mat) {
        materials.add(mat);
      }
    });
  }
  geometries.forEach((g) => g.dispose());
  materials.forEach((m) => m.dispose());
}

export function frameCameraToBox(
  camera: THREE.PerspectiveCamera | undefined,
  controls: OrbitControls | undefined,
  box: THREE.Box3
): void {
  if (!camera || !controls || box.isEmpty()) return;
  const sphere = box.getBoundingSphere(new THREE.Sphere());
  const halfFovRad = THREE.MathUtils.degToRad(camera.fov * 0.5);
  const dist = Math.max((sphere.radius / Math.sin(halfFovRad)) * 1.32, 3.6);
  controls.target.copy(sphere.center);
  camera.position.set(
    sphere.center.x + dist * 0.64,
    sphere.center.y + dist * 0.46,
    sphere.center.z + dist * 0.68
  );
  controls.update();
}

export const INSPECT_HIGHLIGHT_COLOR = new THREE.Color("#38bdf8");

export function applyInspectHighlight(
  mesh: THREE.Mesh | null | undefined,
  mode: "none" | "hover" | "pin",
  pulse = 0
): void {
  if (!mesh) return;
  const mat = mesh.material;
  if (!(mat instanceof THREE.MeshStandardMaterial)) return;
  if (mesh.userData._origEmissiveHex === undefined) {
    mesh.userData._origEmissiveHex = mat.emissive.getHex();
    mesh.userData._origEmissiveIntensity = mat.emissiveIntensity;
  }
  if (mode === "none") {
    mat.emissive.setHex(mesh.userData._origEmissiveHex as number);
    mat.emissiveIntensity = mesh.userData._origEmissiveIntensity as number;
  } else if (mode === "hover") {
    mat.emissive.copy(INSPECT_HIGHLIGHT_COLOR);
    mat.emissiveIntensity = Math.max((mesh.userData._origEmissiveIntensity as number) || 0, 0.28);
  } else {
    mat.emissive.copy(INSPECT_HIGHLIGHT_COLOR);
    mat.emissiveIntensity = 0.36 + 0.16 * pulse;
  }
}

export function toThree(ptMm: [number, number, number], cx: number, cy: number): THREE.Vector3 {
  return new THREE.Vector3((ptMm[0] - cx) * SCALE, ptMm[2] * SCALE, -(ptMm[1] - cy) * SCALE);
}

const _orientDir = new THREE.Vector3();
const _orientUp = new THREE.Vector3(0, 1, 0);
const _orientQuat = new THREE.Quaternion();
const _orientCadY = new THREE.Vector3(0, 0, -1);
const _orientU = new THREE.Vector3();
const _orientV = new THREE.Vector3();
const _orientMat4 = new THREE.Matrix4();

export function orientBeamMesh(mesh: THREE.Mesh, p0: THREE.Vector3, p1: THREE.Vector3): void {
  _orientDir.subVectors(p1, p0);
  const len = Math.max(_orientDir.length(), 1e-4);
  mesh.position.copy(p0).add(p1).multiplyScalar(0.5);
  mesh.scale.set(1, len, 1);
  _orientDir.divideScalar(len);
  _orientQuat.setFromUnitVectors(_orientUp, _orientDir);
  mesh.quaternion.copy(_orientQuat);
}

/**
 * Orient a unit-length box mesh (width along local X, length along local Y, height along local Z)
 * using the exact CAD beam3d.py cross-section frame so `height` stands vertically in CAD Z (Three.js +Y)
 * for horizontal beams and `width` spans horizontally.
 */
export function orientBeamBoxMesh(mesh: THREE.Mesh, p0: THREE.Vector3, p1: THREE.Vector3): void {
  _orientDir.subVectors(p1, p0);
  const len = Math.max(_orientDir.length(), 1e-4);
  mesh.position.copy(p0).add(p1).multiplyScalar(0.5);
  mesh.scale.set(1, len, 1);
  _orientDir.divideScalar(len);
  const refVec = Math.abs(_orientDir.y) > 0.99 ? _orientCadY : _orientUp;
  _orientU.crossVectors(_orientDir, refVec).normalize();
  _orientV.crossVectors(_orientU, _orientDir).normalize();
  // Basis columns: local X = U (width), local Y = T (length), local Z = V (height)
  // Check handedness: U x T = V => det([U, T, V]) = (U x T)·V = +1 -> Wait: in right-handed (X, Y, Z), X x Y = Z!
  // Since U x T = V, [U, T, V] has det = +1!
  _orientMat4.makeBasis(_orientU, _orientDir, _orientV);
  mesh.quaternion.setFromRotationMatrix(_orientMat4);
}

export interface ToolheadMatAppearance {
  hex: number;
  metalness: number;
  roughness: number;
  shortLabel: string;
}

export function getToolheadMaterialAppearance(matId?: string | null): ToolheadMatAppearance {
  switch (matId) {
    case "slm_aluminum":
      // Bead-blasted / as-sintered SLM AlSi10Mg: satin matte metallic vs shiny machined billet
      return { hex: 0xb8c4d0, metalness: 0.72, roughness: 0.42, shortLabel: "SLM AlSi10Mg" };
    case "slm_ti6al4v":
      return { hex: 0x7c7d8c, metalness: 0.74, roughness: 0.4, shortLabel: "SLM Ti64" };
    case "slm_316l_steel":
      return { hex: 0x94a3b8, metalness: 0.78, roughness: 0.38, shortLabel: "SLM 316L" };
    case "slm_17_4ph_steel":
      return { hex: 0x7c8798, metalness: 0.78, roughness: 0.38, shortLabel: "SLM 17-4PH" };
    case "slm_maraging_steel":
      return { hex: 0x6b7280, metalness: 0.76, roughness: 0.4, shortLabel: "SLM MS1" };
    case "slm_cucrzr":
      return { hex: 0xb96d4b, metalness: 0.76, roughness: 0.38, shortLabel: "SLM CuCrZr" };
    case "sheet_metal_steel":
      return { hex: 0x64748b, metalness: 0.82, roughness: 0.32, shortLabel: "Steel Sheet" };
    case "titanium_grade5":
      return { hex: 0x8b8896, metalness: 0.85, roughness: 0.24, shortLabel: "CNC Ti64" };
    case "aluminum_6061_t6":
      return { hex: 0xcbd5e1, metalness: 0.86, roughness: 0.22, shortLabel: "CNC Al 6061" };
    case "aluminum_7075_t6":
      return { hex: 0xdbe4ef, metalness: 0.9, roughness: 0.18, shortLabel: "CNC Al 7075" };
    case "carbon_fiber_plate":
      return { hex: 0x181c24, metalness: 0.32, roughness: 0.34, shortLabel: "3K CF Plate" };
    case "ppa_cf_printed":
      return { hex: 0x1c2128, metalness: 0.22, roughness: 0.5, shortLabel: "PPA-CF" };
    case "pps_cf_printed":
      return { hex: 0x191d24, metalness: 0.24, roughness: 0.48, shortLabel: "PPS-CF" };
    case "pa12_cf_printed":
      return { hex: 0x1f242c, metalness: 0.18, roughness: 0.58, shortLabel: "PA12-CF" };
    case "pet_cf_printed":
      return { hex: 0x28303b, metalness: 0.18, roughness: 0.54, shortLabel: "PET-CF" };
    case "abs_printed":
    default:
      // Realistic Voron dark charcoal/anthracite ABS (paired with subtle Voron-red trim accents)
      return { hex: 0x2d333b, metalness: 0.12, roughness: 0.48, shortLabel: "ABS" };
  }
}
