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
    resolveSubmesh?: (
      meshName: string,
      centerCadMm: [number, number, number]
    ) => {
      heatColor?: THREE.Color;
      heatEmissive?: THREE.Color;
      heatEmissiveIntensity?: number;
      inspectInfo?: InspectedPartInfo;
      vertexColorFn?: (xMm: number, yMm: number, zMm: number, outColor: THREE.Color) => void;
    } | null;
  }
): THREE.Group {
  const clone = cached.clone(true);
  const tmpCenter = new THREE.Vector3();
  const tmpVCol = new THREE.Color();
  clone.traverse((obj) => {
    if (!(obj instanceof THREE.Mesh)) return;
    if (obj.geometry) {
      obj.geometry = obj.geometry.clone();
      if (!obj.geometry.getAttribute("normal")) {
        obj.geometry.computeVertexNormals();
      }
      if (!obj.geometry.boundingBox) {
        obj.geometry.computeBoundingBox();
      }
    }
    let centerCadMm: [number, number, number] = [0, 0, 0];
    if (obj.geometry?.boundingBox) {
      obj.geometry.boundingBox.getCenter(tmpCenter);
      centerCadMm = [tmpCenter.x * 1000, tmpCenter.y * 1000, tmpCenter.z * 1000];
    }
    const subOverride = options.resolveSubmesh
      ? options.resolveSubmesh(obj.name || "", centerCadMm)
      : null;
    const effHeatColor = subOverride?.heatColor ?? options.heatColor;
    const effHeatEmissive = subOverride?.heatEmissive ?? options.heatEmissive;
    const effHeatEmissiveInt =
      subOverride?.heatEmissiveIntensity ?? options.heatEmissiveIntensity;
    const effInspectInfo = subOverride?.inspectInfo ?? options.inspectInfo;
    const vColorFn = options.isHeatMode ? subOverride?.vertexColorFn : undefined;
    let useVertexColors = false;
    if (vColorFn && obj.geometry) {
      const posAttr = obj.geometry.getAttribute("position");
      if (posAttr && posAttr.count > 0) {
        const colors = new Float32Array(posAttr.count * 3);
        for (let vi = 0; vi < posAttr.count; vi++) {
          vColorFn(
            posAttr.getX(vi) * 1000,
            posAttr.getY(vi) * 1000,
            posAttr.getZ(vi) * 1000,
            tmpVCol
          );
          colors[vi * 3] = tmpVCol.r;
          colors[vi * 3 + 1] = tmpVCol.g;
          colors[vi * 3 + 2] = tmpVCol.b;
        }
        obj.geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
        useVertexColors = true;
      }
    }
    const cloneMat = (mat: THREE.Material): THREE.Material => {
      if (options.isHeatMode) {
        const std =
          mat instanceof THREE.MeshStandardMaterial
            ? mat.clone()
            : new THREE.MeshStandardMaterial();
        if (useVertexColors) {
          std.color.setHex(0xffffff);
          std.emissive.copy(effHeatEmissive);
          std.emissiveIntensity = effHeatEmissiveInt * 0.35;
          std.vertexColors = true;
        } else {
          std.color.copy(effHeatColor);
          std.emissive.copy(effHeatEmissive);
          std.emissiveIntensity = effHeatEmissiveInt;
          std.vertexColors = false;
        }
        std.metalness = Math.min(std.metalness, 0.35);
        std.roughness = Math.max(std.roughness, 0.32);
        std.flatShading = false;
        std.side = mat.side;
        std.map = null;
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
        color: effHeatColor,
        emissive: effHeatEmissive,
        emissiveIntensity: effHeatEmissiveInt,
        metalness: options.isHeatMode ? 0.32 : 0.7,
        roughness: options.isHeatMode ? 0.36 : 0.28,
      });
    }
    options.tagInspectable(obj, effInspectInfo);
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
