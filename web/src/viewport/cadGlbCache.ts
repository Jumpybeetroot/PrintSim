import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import {
  AdjointLens,
  FeaLoadCase,
  SubsystemKey,
  ToolheadFlexBreakdownData,
} from "../types";
import type { InspectedPartInfo } from "./geometryHelpers";

export const SCALE = 0.01; // 100 mm = 1 Three.js unit
export const CAD_M_TO_THREE = 1000 * SCALE; // 1.0 m (SI backend GLB) = 10.0 Three.js units

// Module-level CAD GLB cache: each URL is fetched & parsed once, then cloned cheaply per rebuild.
export const cadGltfLoader = new GLTFLoader();
export const cadGlbPromiseCache = new Map<string, Promise<THREE.Group | null>>();
export const cadGlbResolvedCache = new Map<string, THREE.Group | null>();
export const cadGlbBoundsCache = new Map<string, THREE.Box3>();

export function computeCadGlbBounds(root: THREE.Group): THREE.Box3 {
  const wrapper = new THREE.Group();
  const sample = root.clone(true);
  sample.scale.setScalar(CAD_M_TO_THREE);
  sample.rotation.x = -Math.PI * 0.5;
  wrapper.add(sample);
  wrapper.updateMatrixWorld(true);
  return new THREE.Box3().setFromObject(wrapper);
}

export function loadCadGlbCached(url: string): Promise<THREE.Group | null> {
  const existing = cadGlbPromiseCache.get(url);
  if (existing) return existing;
  const promise = new Promise<THREE.Group | null>((resolve) => {
    cadGltfLoader.load(
      url,
      (gltf) => {
        const root = gltf.scene ?? null;
        if (root) {
          const bounds = computeCadGlbBounds(root);
          if (!bounds.isEmpty()) {
            cadGlbBoundsCache.set(url, bounds);
            cadGlbResolvedCache.set(url, root);
            resolve(root);
            return;
          }
        }
        cadGlbResolvedCache.set(url, null);
        resolve(null);
      },
      undefined,
      () => {
        cadGlbResolvedCache.set(url, null);
        resolve(null);
      }
    );
  });
  cadGlbPromiseCache.set(url, promise);
  return promise;
}

export function cloneCachedCadGlb(
  cached: THREE.Group,
  options: {
    isHeatMode: boolean;
    heatColor: THREE.Color;
    heatEmissive: THREE.Color;
    heatEmissiveIntensity: number;
    inspectInfo: InspectedPartInfo;
    tagInspectable: (mesh: THREE.Mesh, info: InspectedPartInfo) => void;
  }
): THREE.Group {
  const clone = cached.clone(true);
  clone.traverse((obj) => {
    if (!(obj instanceof THREE.Mesh)) return;
    if (obj.geometry) {
      obj.geometry = obj.geometry.clone();
      if (!obj.geometry.getAttribute("normal")) {
        obj.geometry.computeVertexNormals();
      }
    }
    const cloneMat = (mat: THREE.Material): THREE.Material => {
      if (options.isHeatMode) {
        const std =
          mat instanceof THREE.MeshStandardMaterial
            ? mat.clone()
            : new THREE.MeshStandardMaterial();
        std.color.copy(options.heatColor);
        std.emissive.copy(options.heatEmissive);
        std.emissiveIntensity = options.heatEmissiveIntensity;
        std.metalness = Math.min(std.metalness, 0.35);
        std.roughness = Math.max(std.roughness, 0.32);
        std.flatShading = false;
        std.side = mat.side;
        std.map = null;
        std.vertexColors = false;
        std.needsUpdate = true;
        return std;
      }
      const cloned = mat.clone();
      if (cloned instanceof THREE.MeshStandardMaterial) {
        cloned.flatShading = false;
        cloned.side = mat.side;
        // Viewport lights are analytical (ambient + hemisphere + 3 directional, no IBL envMap);
        // keep roughness >= 0.24 and cap metalness so brushed Ti/stainless/copper catch highlights cleanly.
        cloned.roughness = Math.max(cloned.roughness, 0.24);
        cloned.metalness = Math.min(cloned.metalness, 0.72);
        cloned.envMapIntensity = Math.max(cloned.envMapIntensity ?? 1.0, 1.25);
      }
      cloned.needsUpdate = true;
      return cloned;
    };
    if (Array.isArray(obj.material)) {
      obj.material = obj.material.map(cloneMat);
    } else if (obj.material) {
      obj.material = cloneMat(obj.material);
    } else {
      obj.material = new THREE.MeshStandardMaterial({
        color: options.heatColor,
        emissive: options.heatEmissive,
        emissiveIntensity: options.heatEmissiveIntensity,
        metalness: options.isHeatMode ? 0.32 : 0.7,
        roughness: options.isHeatMode ? 0.36 : 0.28,
      });
    }
    options.tagInspectable(obj, options.inspectInfo);
  });
  return clone;
}

export const FEA_LOAD_META: Record<FeaLoadCase, { label: string; explanation: string }> = {
  xy: {
    label: "XY Accel",
    explanation:
      "Horizontal X+Y print acceleration (zero gravity droop) — shows real printing flex between nozzle & bed",
  },
  x: {
    label: "X Accel",
    explanation: "Pure horizontal X-axis acceleration from moving toolhead mass",
  },
  y: {
    label: "Y Accel",
    explanation: "Pure horizontal Y-axis acceleration from moving gantry / bed mass",
  },
  gravity: {
    label: "+Gravity",
    explanation: "Full static envelope including vertical gravity sag + belt preload + XY accel",
  },
};

export const ADJOINT_LENS_OPTIONS: Array<{ id: AdjointLens; label: string; title: string }> = [
  {
    id: "print_impact",
    label: "Print Impact",
    title: "Combined nozzle-to-bed flex, XY print-excited modal Hz & moving-mass penalty",
  },
  {
    id: "compliance",
    label: "Static Flex %",
    title: "Pure static nozzle-to-bed compliance share (0 to +1)",
  },
  {
    id: "modal",
    label: "Modal Hz",
    title: "XY print-excited modal frequency sensitivity (-1 to +1)",
  },
  {
    id: "mass",
    label: "Moving Mass",
    title: "Moving-mass frequency penalty — cyan highlights parts to lighten",
  },
];

export const SUBSYSTEM_ORDER: SubsystemKey[] = ["toolhead", "gantry", "frame", "bed"];

export const SUBSYSTEM_META: Record<SubsystemKey, { label: string; color: string }> = {
  toolhead: { label: "Toolhead", color: "#f43f5e" },
  gantry: { label: "Gantry", color: "#f59e0b" },
  frame: { label: "Frame", color: "#38bdf8" },
  bed: { label: "Bed", color: "#a855f7" },
};

export const MODAL_COMPLIANCE_SOURCES: Array<{
  key: "frame" | "belt" | "motor" | "bracket";
  short: string;
  color: string;
}> = [
  { key: "frame", short: "Struct", color: "#38bdf8" },
  { key: "belt", short: "Belt", color: "#f59e0b" },
  { key: "motor", short: "Motor", color: "#a78bfa" },
  { key: "bracket", short: "Bkt", color: "#fb7185" },
];

export const CAUSE_CATEGORY_COLORS: Record<
  "belt" | "motor" | "bracket" | "structural" | "mass",
  string
> = {
  belt: "#f59e0b",
  structural: "#38bdf8",
  motor: "#a78bfa",
  bracket: "#fb7185",
  mass: "#22d3ee",
};

export const PUNCHIN_FLEX_SOURCE_LABELS: Record<
  ToolheadFlexBreakdownData["dominant_flex_source"],
  string
> = {
  rail_xbeam: "X-beam & linear rail bending/torsion",
  carriage_body: "X-carriage backplate & toolhead body compliance",
  rail_carriage: "MGN linear rail & carriage bearing roll",
  toolhead_body: "toolhead body & carriage mount flexure",
  hotend_nozzle: "hotend mount, brace & nozzle stem bending",
};
