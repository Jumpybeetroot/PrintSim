import React, { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import {
  AdjointLens,
  BedMountType,
  FeaLoadCase,
  FrameFeetType,
  GeometryMember,
  ShellPanelGeometryData,
  SimulationResponse,
  SolidFeaEngine,
  SolidFEAResponse,
  SubsystemKey,
  TetFEAResponse,
  TopologyOptResponse,
  ViewMode,
  XRailMountType,
  YRailMountType,
} from "../types";

interface PartGroupClassification {
  id: string;
  label: string;
  subsystem: SubsystemKey;
}

/**
 * Mirror backend `classify_part_group` + `frame_feet` so 3D viewport raycasting
 * maps member/bay IDs to the exact same physical group IDs used by adjoint sensitivity.
 */
function classifyPartGroup(memberId: string, role = ""): PartGroupClassification {
  const mid = (memberId || "").trim();
  if (mid.startsWith("feet_") || mid.startsWith("foot_") || mid.startsWith("ground_") || role === "foot") {
    return { id: "frame_feet", label: "Machine Feet & Benchtop Pads", subsystem: "frame" };
  }
  if (mid.startsWith("col_fl") || mid.startsWith("col_fr")) {
    return { id: "front_frame_columns", label: "Front Frame Columns", subsystem: "frame" };
  }
  if (mid.startsWith("col_rl") || mid.startsWith("col_rr")) {
    return { id: "rear_frame_columns", label: "Rear Frame Columns", subsystem: "frame" };
  }
  if (mid.startsWith("col_")) {
    return { id: "frame_columns", label: "Frame Vertical Columns", subsystem: "frame" };
  }
  if (mid.startsWith("z_col_")) {
    return { id: "z_gantry_uprights", label: "Z-Axis Gantry Uprights", subsystem: "frame" };
  }
  if (mid.startsWith("tower_") || role === "tower") {
    return { id: "delta_towers", label: "Delta Vertical Towers", subsystem: "frame" };
  }
  if (mid.startsWith("arm_")) {
    return { id: "delta_arms", label: "Delta Diagonal Arms", subsystem: "gantry" };
  }
  if (mid.startsWith("x_gantry") || mid.startsWith("x_beam")) {
    return { id: "x_gantry_beam", label: "Moving X-Gantry Beam", subsystem: "gantry" };
  }
  if (mid.startsWith("y_rail") || mid.startsWith("gantry_y")) {
    return { id: "y_axis_rail_beams", label: "Y-Axis Rail Beams", subsystem: "gantry" };
  }
  if (mid.startsWith("toolhead_mount") || mid.startsWith("nozzle_stem") || mid === "toolhead_cog_spring") {
    return { id: "toolhead_mount", label: "Toolhead & Nozzle Mount", subsystem: "toolhead" };
  }
  if (mid === "toolhead_cog" || mid === "toolhead_assembly") {
    return { id: "toolhead_assembly", label: "Toolhead Moving Mass (CoG)", subsystem: "toolhead" };
  }
  if (mid === "bed_plate") {
    return { id: "bed_plate", label: "Heated Bed Plate", subsystem: "bed" };
  }
  if (mid.startsWith("bed_") || mid.startsWith("z_bed")) {
    return { id: "bed_support", label: "Bed Support & Z Arms", subsystem: "bed" };
  }
  if (mid.startsWith("panel_shear_") || mid.startsWith("bay_") || role === "panel") {
    if (mid.includes("left") || mid.includes("right")) {
      return { id: "side_panels", label: "Left & Right Enclosure Panels", subsystem: "frame" };
    }
    if (mid.includes("rear")) {
      return { id: "rear_panel", label: "Rear Enclosure Panel", subsystem: "frame" };
    }
    if (mid.includes("top")) {
      return { id: "top_panel", label: "Top Enclosure Panel", subsystem: "frame" };
    }
    return { id: "enclosure_panels", label: "Enclosure Panels", subsystem: "frame" };
  }
  if (mid.startsWith("brace_") || mid.startsWith("diag_") || role === "brace") {
    return { id: "frame_bracing", label: "Frame Diagonal Bracing", subsystem: "frame" };
  }
  if (
    mid.startsWith("top_") ||
    mid.startsWith("bot_") ||
    mid.startsWith("base_") ||
    mid.startsWith("g_front") ||
    mid.startsWith("g_rear") ||
    mid.startsWith("g_left") ||
    mid.startsWith("g_right") ||
    mid.startsWith("gantry_rear_bar") ||
    mid.startsWith("portal_top")
  ) {
    return { id: "frame_perimeter_beams", label: "Frame Perimeter Beams", subsystem: "frame" };
  }

  if (role === "rail") {
    return { id: "y_axis_rail_beams", label: "Y-Axis Rail Beams", subsystem: "gantry" };
  }
  if (role === "gantry") {
    return { id: "x_gantry_beam", label: "Moving X-Gantry Beam", subsystem: "gantry" };
  }
  if (role === "bed") {
    return { id: "bed_support", label: "Bed Support & Z Arms", subsystem: "bed" };
  }
  if (role === "toolhead") {
    return { id: "toolhead_mount", label: "Toolhead & Nozzle Mount", subsystem: "toolhead" };
  }
  if (role === "frame") {
    return { id: "frame_perimeter_beams", label: "Frame Perimeter Beams", subsystem: "frame" };
  }
  return {
    id: mid || "frame",
    label: mid ? mid.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()) : "Structural Member",
    subsystem: "frame",
  };
}

interface InspectedPartInfo {
  key: string;
  name: string;
  subsystem: SubsystemKey;
  partGroupId: string;
  role: string;
  material?: string;
  profile?: string;
  complianceSharePct?: number;
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
}

interface Props {
  sim: SimulationResponse | null;
  solidFea: SolidFEAResponse | null;
  tetFea: TetFEAResponse | null;
  activeSolidEngine: SolidFeaEngine;
  topoResult: TopologyOptResponse | null;
  viewMode: ViewMode;
  onViewModeChange: (m: ViewMode) => void;
  selectedModeIdx: number;
  onSelectModeIdx: (idx: number) => void;
  feaScale: number;
  onFeaScaleChange: (s: number) => void;
  feaLoadCase: FeaLoadCase;
  onFeaLoadCaseChange: (lc: FeaLoadCase) => void;
  layerLimit: number;
  onLayerLimitChange: (l: number) => void;
  showBedWarp?: boolean;
  onToggleBedWarp?: (show: boolean) => void;
}

const SCALE = 0.01; // 100 mm = 1 Three.js unit

function formatBackendLabel(backend: string): string {
  const low = backend.toLowerCase();
  if (low.includes("gmsh") && (low.includes("skfem") || low.includes("scikit"))) {
    return "Gmsh + scikit-fem";
  }
  return backend;
}

function formatElementLabel(elementType: string): string {
  const low = elementType.toLowerCase();
  if (low === "tet10" || low === "p2" || low === "tet10_p2") return "Tet10 (P2)";
  if (low === "tet4" || low === "p1" || low === "tet4_p1") return "Tet4 (P1)";
  return elementType;
}

const FEA_LOAD_META: Record<FeaLoadCase, { label: string; explanation: string }> = {
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

function modeSeverityMeta(severity?: "critical" | "moderate" | "stiff"): {
  label: string;
  cls: string;
} | null {
  if (severity === "critical") return { label: "Limiting", cls: "bad" };
  if (severity === "moderate") return { label: "Moderate", cls: "warn" };
  if (severity === "stiff") return { label: "Stiff", cls: "ok" };
  return null;
}

function formatHzGain(hz: number): string {
  return hz >= 0.1 ? hz.toFixed(1) : hz.toFixed(2);
}

const ADJOINT_LENS_OPTIONS: Array<{ id: AdjointLens; label: string; title: string }> = [
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

const SUBSYSTEM_ORDER: SubsystemKey[] = ["toolhead", "gantry", "frame", "bed"];

const SUBSYSTEM_META: Record<SubsystemKey, { label: string; color: string }> = {
  toolhead: { label: "Toolhead", color: "#f43f5e" },
  gantry: { label: "Gantry", color: "#f59e0b" },
  frame: { label: "Frame", color: "#38bdf8" },
  bed: { label: "Bed", color: "#a855f7" },
};

function getLensScore(
  entry:
    | {
        heat_score?: number;
        heat_score_compliance?: number;
        heat_score_modal?: number;
        heat_score_mass?: number;
        hz_per_100g?: number;
      }
    | undefined,
  lens: AdjointLens
): number {
  if (!entry) return 0;
  if (lens === "compliance") {
    return entry.heat_score_compliance ?? Math.max(0, entry.heat_score ?? 0);
  }
  if (lens === "modal") {
    return entry.heat_score_modal ?? entry.heat_score ?? 0;
  }
  if (lens === "mass") {
    if (entry.heat_score_mass != null) return entry.heat_score_mass;
    if (entry.heat_score != null && entry.heat_score < 0) return entry.heat_score;
    if (entry.hz_per_100g != null && entry.hz_per_100g < 0) {
      return Math.max(-1, entry.hz_per_100g / 6);
    }
    return 0;
  }
  if (entry.heat_score != null) return entry.heat_score;
  if (entry.hz_per_100g != null && entry.hz_per_100g < 0) {
    return Math.max(-1, entry.hz_per_100g / 6);
  }
  return 0;
}

function impactLevelMeta(level?: "critical" | "high" | "moderate" | "low"): {
  label: string;
  cls: string;
  dotColor: string;
} {
  if (level === "critical") return { label: "Critical", cls: "bad", dotColor: "#ef4444" };
  if (level === "high") return { label: "High", cls: "warn", dotColor: "#f59e0b" };
  if (level === "moderate") return { label: "Moderate", cls: "warn", dotColor: "#eab308" };
  return { label: "Low", cls: "ok", dotColor: "#64748b" };
}

function disposeGroupChildren(group: THREE.Group): void {
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

function frameCameraToBox(
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

const STRESS_STOP_0 = new THREE.Color("#2563eb");
const STRESS_STOP_1 = new THREE.Color("#06b6d4");
const STRESS_STOP_2 = new THREE.Color("#22c55e");
const STRESS_STOP_3 = new THREE.Color("#eab308");
const STRESS_STOP_4 = new THREE.Color("#ef4444");

function stressColorInto(t: number, out: THREE.Color): THREE.Color {
  const c = Math.max(0, Math.min(1, t));
  // Scientific blue -> cyan -> green -> yellow -> red
  if (c < 0.25) {
    out.lerpColors(STRESS_STOP_0, STRESS_STOP_1, c / 0.25);
  } else if (c < 0.5) {
    out.lerpColors(STRESS_STOP_1, STRESS_STOP_2, (c - 0.25) / 0.25);
  } else if (c < 0.75) {
    out.lerpColors(STRESS_STOP_2, STRESS_STOP_3, (c - 0.5) / 0.25);
  } else {
    out.lerpColors(STRESS_STOP_3, STRESS_STOP_4, (c - 0.75) / 0.25);
  }
  return out;
}

function stressColor(t: number): THREE.Color {
  return stressColorInto(t, new THREE.Color());
}

const SENS_NEG_MAX = new THREE.Color("#06b6d4"); // electric cyan (-1.0)
const SENS_NEG_MID = new THREE.Color("#38bdf8"); // sky (-0.5)
const SENS_NEUTRAL = new THREE.Color("#334155"); // slate (0.0)
const SENS_POS_MID = new THREE.Color("#f59e0b"); // amber (+0.5)
const SENS_POS_MAX = new THREE.Color("#ef4444"); // blazing crimson (+1.0)

function sensitivityColorInto(score: number, out: THREE.Color): THREE.Color {
  const s = Math.max(-1, Math.min(1, Number.isFinite(score) ? score : 0));
  if (s >= 0) {
    if (s < 0.5) {
      out.lerpColors(SENS_NEUTRAL, SENS_POS_MID, s / 0.5);
    } else {
      out.lerpColors(SENS_POS_MID, SENS_POS_MAX, (s - 0.5) / 0.5);
    }
  } else {
    const a = -s;
    if (a < 0.5) {
      out.lerpColors(SENS_NEUTRAL, SENS_NEG_MID, a / 0.5);
    } else {
      out.lerpColors(SENS_NEG_MID, SENS_NEG_MAX, (a - 0.5) / 0.5);
    }
  }
  return out;
}

const BED_WARP_VALLEY = new THREE.Color("#3b82f6");
const BED_WARP_FLAT = new THREE.Color("#10b981");
const BED_WARP_CROWN = new THREE.Color("#f59e0b");
const BED_WARP_SEVERE = new THREE.Color("#ef4444");

function bedWarpColorInto(t: number, out: THREE.Color): THREE.Color {
  const c = Math.max(0, Math.min(1, Number.isFinite(t) ? t : 0.5));
  if (c < 0.42) {
    out.lerpColors(BED_WARP_VALLEY, BED_WARP_FLAT, c / 0.42);
  } else if (c < 0.76) {
    out.lerpColors(BED_WARP_FLAT, BED_WARP_CROWN, (c - 0.42) / 0.34);
  } else {
    out.lerpColors(BED_WARP_CROWN, BED_WARP_SEVERE, (c - 0.76) / 0.24);
  }
  return out;
}

const INSPECT_HIGHLIGHT_COLOR = new THREE.Color("#38bdf8");

function applyInspectHighlight(
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

function toThree(ptMm: [number, number, number], cx: number, cy: number): THREE.Vector3 {
  return new THREE.Vector3((ptMm[0] - cx) * SCALE, ptMm[2] * SCALE, -(ptMm[1] - cy) * SCALE);
}

const _orientDir = new THREE.Vector3();
const _orientUp = new THREE.Vector3(0, 1, 0);
const _orientQuat = new THREE.Quaternion();

function orientBeamMesh(mesh: THREE.Mesh, p0: THREE.Vector3, p1: THREE.Vector3): void {
  _orientDir.subVectors(p1, p0);
  const len = Math.max(_orientDir.length(), 1e-4);
  mesh.position.copy(p0).add(p1).multiplyScalar(0.5);
  mesh.scale.set(1, len, 1);
  _orientDir.divideScalar(len);
  _orientQuat.setFromUnitVectors(_orientUp, _orientDir);
  mesh.quaternion.copy(_orientQuat);
}

interface ContinuousBeamChain {
  nodeIds: string[];
  members: GeometryMember[];
  template: GeometryMember;
  isRod: boolean;
}

interface ModalContinuousBeam {
  geometry: THREE.BufferGeometry;
  nodeIds: string[];
  controlPts: THREE.Vector3[];
  controlValues: number[];
  width: number;
  height: number;
  segmentsPerSpan: number;
  undeformedAxis: THREE.Vector3;
  fixedStart: boolean;
  fixedEnd: boolean;
}

interface ModalShellMesh {
  geometry: THREE.BufferGeometry;
  wireGeometry: THREE.BufferGeometry;
  vertNodeIds: string[];
}

/**
 * Merge collinear, matching-spec rectangular extrusions (e.g. col_*_low + col_*_top,
 * y_rail_*_f + y_rail_*_r, x_gantry_l + x_gantry_r, Cartesian Z-columns/base rails,
 * Delta towers) into single continuous multi-span chains.
 */
function groupContinuousBeams(
  members: GeometryMember[],
  basePos: Record<string, THREE.Vector3>,
  fixedNodeIds?: Set<string>
): ContinuousBeamChain[] {
  const isMergeable = (m: GeometryMember): boolean =>
    m.shape !== "rod" &&
    m.role !== "brace" &&
    m.role !== "toolhead" &&
    m.role !== "bed" &&
    Boolean(basePos[m.start_id]) &&
    Boolean(basePos[m.end_id]) &&
    basePos[m.start_id].distanceToSquared(basePos[m.end_id]) > 1e-10;

  const sameExtrusionSpec = (a: GeometryMember, b: GeometryMember): boolean =>
    a.profile === b.profile &&
    a.material === b.material &&
    a.role === b.role &&
    a.shape === b.shape;

  // Map nodeId -> mergeable member indices incident to that node
  const nodeToMergeable = new Map<string, number[]>();
  for (let i = 0; i < members.length; i++) {
    const m = members[i];
    if (!isMergeable(m)) continue;
    let listS = nodeToMergeable.get(m.start_id);
    if (!listS) {
      listS = [];
      nodeToMergeable.set(m.start_id, listS);
    }
    listS.push(i);

    let listE = nodeToMergeable.get(m.end_id);
    if (!listE) {
      listE = [];
      nodeToMergeable.set(m.end_id, listE);
    }
    listE.push(i);
  }

  const otherNode = (m: GeometryMember, sharedId: string): string =>
    m.start_id === sharedId ? m.end_id : m.start_id;

  const dirFromTo = (fromId: string, toId: string, out: THREE.Vector3): THREE.Vector3 => {
    out.subVectors(basePos[toId], basePos[fromId]);
    const len = out.length();
    if (len > 1e-8) out.divideScalar(len);
    return out;
  };

  const _dIn = new THREE.Vector3();
  const _dCand = new THREE.Vector3();

  /**
   * Find the unique collinear continuation of member `mIdx` leaving `sharedId`
   * along the same straight axis (degree-2-for-this-chain collinearity check).
   */
  const findCollinearPartner = (mIdx: number, sharedId: string): number => {
    const m = members[mIdx];
    const incident = nodeToMergeable.get(sharedId);
    if (!incident || incident.length < 2) return -1;

    const prevId = otherNode(m, sharedId);
    dirFromTo(prevId, sharedId, _dIn);

    let forwardPartner = -1;
    let forwardCount = 0;
    let backwardOtherCount = 0;

    for (const cIdx of incident) {
      if (cIdx === mIdx) continue;
      const cand = members[cIdx];
      if (!sameExtrusionSpec(m, cand)) continue;
      const candOther = otherNode(cand, sharedId);
      dirFromTo(sharedId, candOther, _dCand);
      const dot = _dIn.dot(_dCand);
      if (dot > 0.995) {
        forwardCount++;
        forwardPartner = cIdx;
      } else if (dot < -0.995) {
        backwardOtherCount++;
      }
    }

    // Strict degree-2 along this collinear line (exactly one entering, one leaving)
    if (forwardCount === 1 && backwardOtherCount === 0) {
      return forwardPartner;
    }
    return -1;
  };

  const partnerAtStart = new Int32Array(members.length).fill(-1);
  const partnerAtEnd = new Int32Array(members.length).fill(-1);
  for (let i = 0; i < members.length; i++) {
    if (!isMergeable(members[i])) continue;
    partnerAtStart[i] = findCollinearPartner(i, members[i].start_id);
    partnerAtEnd[i] = findCollinearPartner(i, members[i].end_id);
  }

  // Enforce symmetric pairing
  for (let i = 0; i < members.length; i++) {
    const ps = partnerAtStart[i];
    if (ps !== -1 && partnerAtStart[ps] !== i && partnerAtEnd[ps] !== i) {
      partnerAtStart[i] = -1;
    }
    const pe = partnerAtEnd[i];
    if (pe !== -1 && partnerAtStart[pe] !== i && partnerAtEnd[pe] !== i) {
      partnerAtEnd[i] = -1;
    }
  }

  const visited = new Uint8Array(members.length);
  const chains: ContinuousBeamChain[] = [];

  const traceChainFrom = (startMemberIdx: number, startFromEnd: boolean) => {
    const startMember = members[startMemberIdx];
    const chainMembers: GeometryMember[] = [startMember];
    const chainNodes: string[] = startFromEnd
      ? [startMember.end_id, startMember.start_id]
      : [startMember.start_id, startMember.end_id];

    visited[startMemberIdx] = 1;
    let currIdx = startMemberIdx;
    let currNode = chainNodes[1];

    while (true) {
      const m = members[currIdx];
      const nextIdx = currNode === m.end_id ? partnerAtEnd[currIdx] : partnerAtStart[currIdx];
      if (nextIdx === -1 || visited[nextIdx]) break;

      visited[nextIdx] = 1;
      const nextM = members[nextIdx];
      const nextNode = otherNode(nextM, currNode);
      chainMembers.push(nextM);
      chainNodes.push(nextNode);
      currIdx = nextIdx;
      currNode = nextNode;
    }

    // Orient chains so a fixed foot/anchor node (e.g. b_fl) is always at index 0
    if (fixedNodeIds && chainNodes.length >= 2) {
      const firstFixed = fixedNodeIds.has(chainNodes[0]);
      const lastFixed = fixedNodeIds.has(chainNodes[chainNodes.length - 1]);
      if (!firstFixed && lastFixed) {
        chainNodes.reverse();
        chainMembers.reverse();
      }
    }

    chains.push({
      nodeIds: chainNodes,
      members: chainMembers,
      template: chainMembers[0],
      isRod: chainMembers[0].shape === "rod",
    });
  };

  // Pass 1: start from natural chain endpoints (degree <= 1 along the collinear chain)
  for (let i = 0; i < members.length; i++) {
    if (visited[i] || !isMergeable(members[i])) continue;
    const hasStartPartner = partnerAtStart[i] !== -1;
    const hasEndPartner = partnerAtEnd[i] !== -1;
    if (!hasStartPartner) {
      traceChainFrom(i, false);
    } else if (!hasEndPartner) {
      traceChainFrom(i, true);
    }
  }

  // Pass 2: any remaining members (non-mergeable rods/braces/toolhead or degenerate loops)
  for (let i = 0; i < members.length; i++) {
    if (visited[i]) continue;
    if (isMergeable(members[i])) {
      traceChainFrom(i, false);
    } else {
      visited[i] = 1;
      const m = members[i];
      chains.push({
        nodeIds: [m.start_id, m.end_id],
        members: [m],
        template: m,
        isRod: m.shape === "rod",
      });
    }
  }

  return chains;
}

// Module-level zero-allocation scratch buffers for 60 fps continuous beam sweeping
const _beamUp = new THREE.Vector3(0, 1, 0);
const _beamRefAxis = new THREE.Vector3();
const _beamInitQ = new THREE.Quaternion();
const _beamStepQ = new THREE.Quaternion();
const _beamU = new THREE.Vector3();
const _beamV = new THREE.Vector3();
const _beamT = new THREE.Vector3();
const _beamPrevT = new THREE.Vector3();
const _beamP = new THREE.Vector3();
const _beamC0 = new THREE.Vector3();
const _beamC1 = new THREE.Vector3();
const _beamC2 = new THREE.Vector3();
const _beamC3 = new THREE.Vector3();
const _beamCol = new THREE.Color();
const _beamSecantPool: THREE.Vector3[] = [];
const _beamTangentPool: THREE.Vector3[] = [];
let _beamSpanLens = new Float64Array(16);

function ensureSplineScratch(numPts: number): void {
  while (_beamSecantPool.length < numPts) {
    _beamSecantPool.push(new THREE.Vector3());
  }
  while (_beamTangentPool.length < numPts) {
    _beamTangentPool.push(new THREE.Vector3());
  }
  if (_beamSpanLens.length < numPts) {
    _beamSpanLens = new Float64Array(Math.max(numPts * 2, 16));
  }
}

function writeRingVertex(
  posArr: Float32Array,
  normArr: Float32Array,
  colArr: Float32Array | null,
  vIdx: number,
  pt: THREE.Vector3,
  nx: number,
  ny: number,
  nz: number,
  col: THREE.Color | null
): void {
  const o = vIdx * 3;
  posArr[o] = pt.x;
  posArr[o + 1] = pt.y;
  posArr[o + 2] = pt.z;
  normArr[o] = nx;
  normArr[o + 1] = ny;
  normArr[o + 2] = nz;
  if (colArr && col) {
    colArr[o] = col.r;
    colArr[o + 1] = col.g;
    colArr[o + 2] = col.b;
  }
}

/**
 * Update a pre-allocated continuous swept rectangular beam BufferGeometry in-place
 * using chord-length cubic Hermite interpolation, fixed-end tangent clamping,
 * stable Bishop parallel-transport frames, sharp split rectangular face normals,
 * and smooth per-vertex scientific color gradients (zero per-frame heap allocations).
 */
function updateContinuousBeamGeometry(
  geo: THREE.BufferGeometry,
  controlPts: THREE.Vector3[],
  controlValues: number[],
  width: number,
  height: number,
  segmentsPerSpan: number,
  useVertexColors: boolean,
  undeformedAxis?: THREE.Vector3,
  fixedStart = false,
  fixedEnd = false,
  updateColors = true
): void {
  const numPts = controlPts.length;
  if (numPts < 2) return;
  const numSpans = numPts - 1;
  const segs = Math.max(1, Math.floor(segmentsPerSpan));
  const totalSlices = numSpans * segs;
  const ringCount = totalSlices + 1;

  const posAttr = geo.getAttribute("position") as THREE.BufferAttribute;
  const normAttr = geo.getAttribute("normal") as THREE.BufferAttribute;
  const colAttr = useVertexColors
    ? (geo.getAttribute("color") as THREE.BufferAttribute | undefined)
    : undefined;
  const posArr = posAttr.array as Float32Array;
  const normArr = normAttr.array as Float32Array;
  const colArr = useVertexColors && updateColors && colAttr ? (colAttr.array as Float32Array) : null;

  ensureSplineScratch(numPts);

  // 1. Reference undeformed axis for stable Bishop frame anchor
  if (undeformedAxis && undeformedAxis.lengthSq() > 1e-10) {
    _beamRefAxis.copy(undeformedAxis).normalize();
  } else {
    _beamRefAxis.subVectors(controlPts[numPts - 1], controlPts[0]);
    const refLen = _beamRefAxis.length();
    if (refLen > 1e-8) {
      _beamRefAxis.divideScalar(refLen);
    } else {
      _beamRefAxis.set(0, 1, 0);
    }
  }

  // 2. Chord-length parameterization & secant velocities S_i = (P_{i+1} - P_i) / d_i
  for (let i = 0; i < numSpans; i++) {
    const sVec = _beamSecantPool[i];
    sVec.subVectors(controlPts[i + 1], controlPts[i]);
    const d = Math.max(sVec.length(), 1e-5);
    _beamSpanLens[i] = d;
    sVec.divideScalar(d);
  }

  // 3. Smooth cubic Hermite tangents M_i (chord-length Bessel interior + fixed/natural endpoints)
  if (numPts === 2) {
    _beamTangentPool[0].copy(_beamSecantPool[0]);
    _beamTangentPool[1].copy(_beamSecantPool[0]);
  } else {
    // Interior nodes: chord-length weighted parabolic tangent (prevents overshoot on unequal spans)
    for (let i = 1; i < numPts - 1; i++) {
      const dPrev = _beamSpanLens[i - 1];
      const dNext = _beamSpanLens[i];
      const denom = dPrev + dNext;
      const wPrev = dNext / denom;
      const wNext = dPrev / denom;
      const sPrev = _beamSecantPool[i - 1];
      const sNext = _beamSecantPool[i];
      _beamTangentPool[i].set(
        wPrev * sPrev.x + wNext * sNext.x,
        wPrev * sPrev.y + wNext * sNext.y,
        wPrev * sPrev.z + wNext * sNext.z
      );
    }

    // Start node tangent: clamp to undeformed axis if fixed (authentic cantilever S-curve), else natural (P''=0)
    if (fixedStart) {
      _beamTangentPool[0].copy(_beamRefAxis);
    } else {
      const s0 = _beamSecantPool[0];
      const m1 = _beamTangentPool[1];
      _beamTangentPool[0].set(
        1.5 * s0.x - 0.5 * m1.x,
        1.5 * s0.y - 0.5 * m1.y,
        1.5 * s0.z - 0.5 * m1.z
      );
    }

    // End node tangent
    const last = numPts - 1;
    if (fixedEnd) {
      _beamTangentPool[last].copy(_beamRefAxis);
    } else {
      const sLast = _beamSecantPool[numSpans - 1];
      const mPrev = _beamTangentPool[last - 1];
      _beamTangentPool[last].set(
        1.5 * sLast.x - 0.5 * mPrev.x,
        1.5 * sLast.y - 0.5 * mPrev.y,
        1.5 * sLast.z - 0.5 * mPrev.z
      );
    }
  }

  // 4. Initial cross-section frame (U, refAxis, V) matching orientBeamMesh BoxGeometry(w, 1, h)
  _beamInitQ.setFromUnitVectors(_beamUp, _beamRefAxis);
  _beamU.set(1, 0, 0).applyQuaternion(_beamInitQ).normalize();
  _beamV.set(0, 0, 1).applyQuaternion(_beamInitQ).normalize();

  const hw = width * 0.5;
  const hh = height * 0.5;
  const sideVertCount = 4 * ringCount * 2;
  const startCapBase = sideVertCount;
  const endCapBase = sideVertCount + 4;

  // 5. Sweep rings along the spline with Bishop parallel transport
  for (let ring = 0; ring <= totalSlices; ring++) {
    const spanIdx = ring === totalSlices ? numSpans - 1 : Math.floor(ring / segs);
    const localStep = ring === totalSlices ? segs : ring - spanIdx * segs;
    const u = localStep / segs;

    const p0 = controlPts[spanIdx];
    const p1 = controlPts[spanIdx + 1];
    const d = _beamSpanLens[spanIdx];
    const m0 = _beamTangentPool[spanIdx];
    const m1 = _beamTangentPool[spanIdx + 1];

    if (numPts === 2) {
      _beamP.set(
        p0.x + (p1.x - p0.x) * u,
        p0.y + (p1.y - p0.y) * u,
        p0.z + (p1.z - p0.z) * u
      );
      _beamT.copy(_beamSecantPool[0]);
    } else {
      const u2 = u * u;
      const u3 = u2 * u;
      const h00 = 2 * u3 - 3 * u2 + 1;
      const h10 = (u3 - 2 * u2 + u) * d;
      const h01 = -2 * u3 + 3 * u2;
      const h11 = (u3 - u2) * d;

      _beamP.set(
        h00 * p0.x + h10 * m0.x + h01 * p1.x + h11 * m1.x,
        h00 * p0.y + h10 * m0.y + h01 * p1.y + h11 * m1.y,
        h00 * p0.z + h10 * m0.z + h01 * p1.z + h11 * m1.z
      );

      const dh00 = 6 * u2 - 6 * u;
      const dh10 = (3 * u2 - 4 * u + 1) * d;
      const dh01 = -6 * u2 + 6 * u;
      const dh11 = (3 * u2 - 2 * u) * d;

      _beamT.set(
        dh00 * p0.x + dh10 * m0.x + dh01 * p1.x + dh11 * m1.x,
        dh00 * p0.y + dh10 * m0.y + dh01 * p1.y + dh11 * m1.y,
        dh00 * p0.z + dh10 * m0.z + dh01 * p1.z + dh11 * m1.z
      );
      const tLen = _beamT.length();
      if (tLen > 1e-8) {
        _beamT.divideScalar(tLen);
      } else {
        _beamT.copy(_beamSecantPool[spanIdx]);
      }
    }

    // Parallel-transport frame (U, T, V) with U × T = V
    if (ring === 0) {
      const dot0 = _beamRefAxis.dot(_beamT);
      if (dot0 < 0.999999 && dot0 > -0.9999) {
        _beamStepQ.setFromUnitVectors(_beamRefAxis, _beamT);
        _beamU.applyQuaternion(_beamStepQ);
      }
    } else {
      const dotS = _beamPrevT.dot(_beamT);
      if (dotS < 0.999999 && dotS > -0.9999) {
        _beamStepQ.setFromUnitVectors(_beamPrevT, _beamT);
        _beamU.applyQuaternion(_beamStepQ);
      }
    }
    _beamU.addScaledVector(_beamT, -_beamU.dot(_beamT));
    if (_beamU.lengthSq() > 1e-10) {
      _beamU.normalize();
    } else {
      _beamU.set(1, 0, 0).cross(_beamT);
      if (_beamU.lengthSq() < 1e-6) _beamU.set(0, 0, 1).cross(_beamT);
      _beamU.normalize();
    }
    _beamV.crossVectors(_beamU, _beamT).normalize();
    _beamPrevT.copy(_beamT);

    // 4 cross-section corners around _beamP
    const ux = _beamU.x * hw;
    const uy = _beamU.y * hw;
    const uz = _beamU.z * hw;
    const vx = _beamV.x * hh;
    const vy = _beamV.y * hh;
    const vz = _beamV.z * hh;

    _beamC0.set(_beamP.x - ux - vx, _beamP.y - uy - vy, _beamP.z - uz - vz);
    _beamC1.set(_beamP.x + ux - vx, _beamP.y + uy - vy, _beamP.z + uz - vz);
    _beamC2.set(_beamP.x + ux + vx, _beamP.y + uy + vy, _beamP.z + uz + vz);
    _beamC3.set(_beamP.x - ux + vx, _beamP.y - uy + vy, _beamP.z - uz + vz);

    let ringColor: THREE.Color | null = null;
    if (colArr) {
      const v0 = controlValues[spanIdx] ?? 0;
      const v1 = controlValues[spanIdx + 1] ?? v0;
      ringColor = stressColorInto(v0 + (v1 - v0) * u, _beamCol);
    }

    // Side 0: -V face (c0, c1)
    const s0Base = (0 * ringCount + ring) * 2;
    writeRingVertex(posArr, normArr, colArr, s0Base, _beamC0, -_beamV.x, -_beamV.y, -_beamV.z, ringColor);
    writeRingVertex(posArr, normArr, colArr, s0Base + 1, _beamC1, -_beamV.x, -_beamV.y, -_beamV.z, ringColor);

    // Side 1: +U face (c1, c2)
    const s1Base = (1 * ringCount + ring) * 2;
    writeRingVertex(posArr, normArr, colArr, s1Base, _beamC1, _beamU.x, _beamU.y, _beamU.z, ringColor);
    writeRingVertex(posArr, normArr, colArr, s1Base + 1, _beamC2, _beamU.x, _beamU.y, _beamU.z, ringColor);

    // Side 2: +V face (c2, c3)
    const s2Base = (2 * ringCount + ring) * 2;
    writeRingVertex(posArr, normArr, colArr, s2Base, _beamC2, _beamV.x, _beamV.y, _beamV.z, ringColor);
    writeRingVertex(posArr, normArr, colArr, s2Base + 1, _beamC3, _beamV.x, _beamV.y, _beamV.z, ringColor);

    // Side 3: -U face (c3, c0)
    const s3Base = (3 * ringCount + ring) * 2;
    writeRingVertex(posArr, normArr, colArr, s3Base, _beamC3, -_beamU.x, -_beamU.y, -_beamU.z, ringColor);
    writeRingVertex(posArr, normArr, colArr, s3Base + 1, _beamC0, -_beamU.x, -_beamU.y, -_beamU.z, ringColor);

    // Closed end caps with crisp flat face normals
    if (ring === 0) {
      writeRingVertex(posArr, normArr, colArr, startCapBase, _beamC0, -_beamT.x, -_beamT.y, -_beamT.z, ringColor);
      writeRingVertex(posArr, normArr, colArr, startCapBase + 1, _beamC1, -_beamT.x, -_beamT.y, -_beamT.z, ringColor);
      writeRingVertex(posArr, normArr, colArr, startCapBase + 2, _beamC2, -_beamT.x, -_beamT.y, -_beamT.z, ringColor);
      writeRingVertex(posArr, normArr, colArr, startCapBase + 3, _beamC3, -_beamT.x, -_beamT.y, -_beamT.z, ringColor);
    }
    if (ring === totalSlices) {
      writeRingVertex(posArr, normArr, colArr, endCapBase, _beamC0, _beamT.x, _beamT.y, _beamT.z, ringColor);
      writeRingVertex(posArr, normArr, colArr, endCapBase + 1, _beamC1, _beamT.x, _beamT.y, _beamT.z, ringColor);
      writeRingVertex(posArr, normArr, colArr, endCapBase + 2, _beamC2, _beamT.x, _beamT.y, _beamT.z, ringColor);
      writeRingVertex(posArr, normArr, colArr, endCapBase + 3, _beamC3, _beamT.x, _beamT.y, _beamT.z, ringColor);
    }
  }

  posAttr.needsUpdate = true;
  normAttr.needsUpdate = true;
  if (colArr && colAttr) {
    colAttr.needsUpdate = true;
  }
}

function createContinuousBeamGeometry(
  controlPts: THREE.Vector3[],
  controlValues: number[],
  width: number,
  height: number,
  segmentsPerSpan: number,
  useVertexColors: boolean,
  undeformedAxis?: THREE.Vector3,
  fixedStart = false,
  fixedEnd = false
): THREE.BufferGeometry {
  const numSpans = Math.max(1, controlPts.length - 1);
  const segs = Math.max(1, Math.floor(segmentsPerSpan));
  const totalSlices = numSpans * segs;
  const ringCount = totalSlices + 1;

  const sideVertCount = 4 * ringCount * 2;
  const totalVerts = sideVertCount + 8;

  const positions = new Float32Array(totalVerts * 3);
  const normals = new Float32Array(totalVerts * 3);

  const triCount = 4 * totalSlices * 2 + 4;
  const indices =
    totalVerts <= 65535 ? new Uint16Array(triCount * 3) : new Uint32Array(triCount * 3);

  let idxOff = 0;
  for (let side = 0; side < 4; side++) {
    for (let s = 0; s < totalSlices; s++) {
      const a0 = (side * ringCount + s) * 2;
      const a1 = a0 + 1;
      const b0 = a0 + 2;
      const b1 = a0 + 3;
      indices[idxOff++] = a0;
      indices[idxOff++] = b0;
      indices[idxOff++] = b1;
      indices[idxOff++] = a0;
      indices[idxOff++] = b1;
      indices[idxOff++] = a1;
    }
  }

  // Start cap (normal -T_0)
  const startCapBase = sideVertCount;
  indices[idxOff++] = startCapBase;
  indices[idxOff++] = startCapBase + 1;
  indices[idxOff++] = startCapBase + 2;
  indices[idxOff++] = startCapBase;
  indices[idxOff++] = startCapBase + 2;
  indices[idxOff++] = startCapBase + 3;

  // End cap (normal +T_N)
  const endCapBase = sideVertCount + 4;
  indices[idxOff++] = endCapBase;
  indices[idxOff++] = endCapBase + 3;
  indices[idxOff++] = endCapBase + 2;
  indices[idxOff++] = endCapBase;
  indices[idxOff++] = endCapBase + 2;
  indices[idxOff++] = endCapBase + 1;

  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geo.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
  if (useVertexColors) {
    const colors = new Float32Array(totalVerts * 3);
    geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  }
  geo.setIndex(new THREE.BufferAttribute(indices, 1));

  updateContinuousBeamGeometry(
    geo,
    controlPts,
    controlValues,
    width,
    height,
    segs,
    useVertexColors,
    undeformedAxis,
    fixedStart,
    fixedEnd,
    true
  );
  geo.computeBoundingSphere();
  return geo;
}

function applyRodGradientColors(geo: THREE.BufferGeometry, val0: number, val1: number): void {
  const posAttr = geo.getAttribute("position") as THREE.BufferAttribute;
  const colors = new Float32Array(posAttr.count * 3);
  for (let i = 0; i < posAttr.count; i++) {
    const t = Math.max(0, Math.min(1, posAttr.getY(i) + 0.5));
    stressColorInto(val0 + (val1 - val0) * t, _beamCol);
    colors[i * 3] = _beamCol.r;
    colors[i * 3 + 1] = _beamCol.g;
    colors[i * 3 + 2] = _beamCol.b;
  }
  geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
}

export const PrinterViewport: React.FC<Props> = ({
  sim,
  solidFea,
  tetFea,
  activeSolidEngine,
  topoResult,
  viewMode,
  onViewModeChange,
  selectedModeIdx,
  onSelectModeIdx,
  feaScale,
  onFeaScaleChange,
  feaLoadCase,
  onFeaLoadCaseChange,
  layerLimit,
  onLayerLimitChange,
  showBedWarp: showBedWarpProp,
  onToggleBedWarp,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [showCarvedGhost, setShowCarvedGhost] = useState<boolean>(true);
  const [adjointLens, setAdjointLens] = useState<AdjointLens>("print_impact");
  const [modalColorMode, setModalColorMode] = useState<"relative" | "global">("relative");
  const [internalShowBedWarp, setInternalShowBedWarp] = useState<boolean>(false);
  const [hoveredPart, setHoveredPart] = useState<InspectedPartInfo | null>(null);
  const [pinnedPart, setPinnedPart] = useState<InspectedPartInfo | null>(null);

  const showBedWarp = showBedWarpProp ?? internalShowBedWarp;
  const setShowBedWarp = (next: boolean) => {
    if (onToggleBedWarp) {
      onToggleBedWarp(next);
    } else {
      setInternalShowBedWarp(next);
    }
  };

  const stateRef = useRef<{
    renderer?: THREE.WebGLRenderer;
    scene?: THREE.Scene;
    camera?: THREE.PerspectiveCamera;
    controls?: OrbitControls;
    dynamicGroup?: THREE.Group;
    animFrame?: number;
    lastFramedKey?: string;
    inspectableMeshes: THREE.Mesh[];
    hoveredMesh: THREE.Mesh | null;
    pinnedMesh: THREE.Mesh | null;
    pinnedKey: string | null;
    hoveredKey: string | null;
    modalMeshes: Array<{
      mesh: THREE.Mesh;
      startId: string;
      endId: string;
    }>;
    modalContinuousBeams: ModalContinuousBeam[];
    modalShellMeshes: ModalShellMesh[];
    modalToolhead?: THREE.Group;
    modalToolheadNodeId?: string;
    modalCogMarker?: THREE.Group;
    modalCogNodeId?: string;
    modalNozzleMarker?: THREE.Group;
    modalNozzleNodeId?: string;
    modalNozzleRefNodeId?: string;
    modalBedMesh?: THREE.Mesh;
    modalBedNodeId?: string;
    modalBedOffset: THREE.Vector3;
    basePositions: Record<string, THREE.Vector3>;
    modalVectors: Record<string, THREE.Vector3>;
    isModal: boolean;
  }>({
    inspectableMeshes: [],
    hoveredMesh: null,
    pinnedMesh: null,
    pinnedKey: null,
    hoveredKey: null,
    modalMeshes: [],
    modalContinuousBeams: [],
    modalShellMeshes: [],
    modalBedOffset: new THREE.Vector3(0, -0.04, 0),
    basePositions: {},
    modalVectors: {},
    isModal: false,
  });

  const handleCloseInspector = () => {
    if (stateRef.current.pinnedMesh) {
      const prevPinned = stateRef.current.pinnedMesh;
      stateRef.current.pinnedMesh = null;
      stateRef.current.pinnedKey = null;
      applyInspectHighlight(
        prevPinned,
        stateRef.current.hoveredMesh === prevPinned ? "hover" : "none"
      );
    } else {
      stateRef.current.pinnedKey = null;
    }
    setPinnedPart(null);
  };

  const showTetPart = Boolean(tetFea) && (activeSolidEngine === "tet10" || !solidFea);
  const showHexPart = !showTetPart && Boolean(solidFea);

  // Initialize Three.js scene once
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(42, el.clientWidth / el.clientHeight, 0.05, 200);
    camera.position.set(7.5, 6.2, 8.2);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(el.clientWidth, el.clientHeight);
    el.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(0, 2.3, 0);

    // Lighting
    scene.add(new THREE.AmbientLight(0xffffff, 0.65));
    const keyLight = new THREE.DirectionalLight(0xffffff, 1.05);
    keyLight.position.set(8, 14, 10);
    scene.add(keyLight);
    const rimLight = new THREE.DirectionalLight(0x38bdf8, 0.45);
    rimLight.position.set(-10, 8, -8);
    scene.add(rimLight);

    // Floor grid
    const grid = new THREE.GridHelper(14, 28, 0x334155, 0x1e293b);
    scene.add(grid);

    const dynamicGroup = new THREE.Group();
    scene.add(dynamicGroup);

    stateRef.current.renderer = renderer;
    stateRef.current.scene = scene;
    stateRef.current.camera = camera;
    stateRef.current.controls = controls;
    stateRef.current.dynamicGroup = dynamicGroup;

    // Interactive 3D raycasting for Part Inspector (click vs orbit-drag safe)
    const raycaster = new THREE.Raycaster();
    const pointerNdc = new THREE.Vector2();
    let pointerDownPos: { x: number; y: number; button: number } | null = null;
    let lastMoveTs = 0;

    const pickInspectableMesh = (clientX: number, clientY: number): THREE.Mesh | null => {
      const targets = stateRef.current.inspectableMeshes;
      if (!targets.length) return null;
      const rect = renderer.domElement.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) return null;
      pointerNdc.x = ((clientX - rect.left) / rect.width) * 2 - 1;
      pointerNdc.y = -((clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointerNdc, camera);
      const hits = raycaster.intersectObjects(targets, false);
      for (const hit of hits) {
        const obj = hit.object as THREE.Mesh;
        if (obj.userData?.inspectInfo) {
          return obj;
        }
      }
      return null;
    };

    const onPointerDown = (e: PointerEvent) => {
      if (e.button === 0) {
        pointerDownPos = { x: e.clientX, y: e.clientY, button: e.button };
      } else {
        pointerDownPos = null;
      }
    };

    const onPointerMove = (e: PointerEvent) => {
      const now = performance.now();
      if (now - lastMoveTs < 30) return;
      lastMoveTs = now;
      if (e.buttons !== 0) {
        // User is actively orbiting/panning — clear hover indicator cleanly
        if (stateRef.current.hoveredMesh) {
          const oldHover = stateRef.current.hoveredMesh;
          stateRef.current.hoveredMesh = null;
          if (oldHover !== stateRef.current.pinnedMesh) {
            applyInspectHighlight(oldHover, "none");
          }
        }
        if (stateRef.current.hoveredKey !== null) {
          stateRef.current.hoveredKey = null;
          setHoveredPart(null);
        }
        renderer.domElement.style.cursor = "";
        return;
      }

      const hitMesh = pickInspectableMesh(e.clientX, e.clientY);
      const info = (hitMesh?.userData?.inspectInfo as InspectedPartInfo | undefined) ?? null;
      renderer.domElement.style.cursor = info ? "pointer" : "";

      if (hitMesh !== stateRef.current.hoveredMesh) {
        const prevHover = stateRef.current.hoveredMesh;
        stateRef.current.hoveredMesh = hitMesh;
        if (prevHover && prevHover !== stateRef.current.pinnedMesh) {
          applyInspectHighlight(prevHover, "none");
        }
        if (hitMesh && hitMesh !== stateRef.current.pinnedMesh) {
          applyInspectHighlight(hitMesh, "hover");
        }
      }

      const nextKey = info?.key ?? null;
      if (nextKey !== stateRef.current.hoveredKey) {
        stateRef.current.hoveredKey = nextKey;
        setHoveredPart(info);
      }
    };

    const onPointerUp = (e: PointerEvent) => {
      if (!pointerDownPos || e.button !== 0) return;
      const dist = Math.hypot(e.clientX - pointerDownPos.x, e.clientY - pointerDownPos.y);
      pointerDownPos = null;
      if (dist >= 5) return; // orbit drag, not a part click

      const hitMesh = pickInspectableMesh(e.clientX, e.clientY);
      const info = (hitMesh?.userData?.inspectInfo as InspectedPartInfo | undefined) ?? null;
      const prevPinned = stateRef.current.pinnedMesh;

      if (prevPinned && prevPinned !== hitMesh) {
        applyInspectHighlight(
          prevPinned,
          stateRef.current.hoveredMesh === prevPinned ? "hover" : "none"
        );
      }

      if (hitMesh && info) {
        stateRef.current.pinnedMesh = hitMesh;
        stateRef.current.pinnedKey = info.key;
        applyInspectHighlight(hitMesh, "pin", 0.5);
        setPinnedPart(info);
      } else {
        stateRef.current.pinnedMesh = null;
        stateRef.current.pinnedKey = null;
        setPinnedPart(null);
      }
    };

    const onPointerLeave = () => {
      pointerDownPos = null;
      renderer.domElement.style.cursor = "";
      const prevHover = stateRef.current.hoveredMesh;
      stateRef.current.hoveredMesh = null;
      if (prevHover && prevHover !== stateRef.current.pinnedMesh) {
        applyInspectHighlight(prevHover, "none");
      }
      if (stateRef.current.hoveredKey !== null) {
        stateRef.current.hoveredKey = null;
        setHoveredPart(null);
      }
    };

    renderer.domElement.addEventListener("pointerdown", onPointerDown);
    renderer.domElement.addEventListener("pointermove", onPointerMove);
    renderer.domElement.addEventListener("pointerup", onPointerUp);
    renderer.domElement.addEventListener("pointerleave", onPointerLeave);

    const onResize = () => {
      if (!el) return;
      camera.aspect = el.clientWidth / el.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(el.clientWidth, el.clientHeight);
    };
    window.addEventListener("resize", onResize);

    const clock = new THREE.Clock();
    const tmpPosA = new THREE.Vector3();
    const tmpPosB = new THREE.Vector3();
    const tmpNozV0 = new THREE.Vector3();
    const tmpNozV1 = new THREE.Vector3();

    const animate = () => {
      stateRef.current.animFrame = requestAnimationFrame(animate);
      controls.update();

      const t = clock.getElapsedTime();
      if (stateRef.current.pinnedMesh) {
        const pulse = 0.5 + 0.5 * Math.sin(t * 4.2);
        applyInspectHighlight(stateRef.current.pinnedMesh, "pin", pulse);
      }

      if (
        stateRef.current.isModal &&
        (stateRef.current.modalContinuousBeams.length > 0 ||
          stateRef.current.modalMeshes.length > 0 ||
          stateRef.current.modalShellMeshes.length > 0)
      ) {
        const phase = Math.sin(t * 7.5) * 0.38;
        const {
          basePositions,
          modalVectors,
          modalContinuousBeams,
          modalShellMeshes,
          modalMeshes,
          modalToolhead,
          modalToolheadNodeId,
          modalCogMarker,
          modalCogNodeId,
          modalNozzleMarker,
          modalNozzleNodeId,
          modalNozzleRefNodeId,
          modalBedMesh,
          modalBedNodeId,
          modalBedOffset,
        } = stateRef.current;

        const getPosInto = (id: string, out: THREE.Vector3): THREE.Vector3 => {
          const b = basePositions[id];
          const v = modalVectors[id];
          if (!b) return out.set(0, 0, 0);
          out.copy(b);
          if (v) out.addScaledVector(v, phase);
          return out;
        };

        for (const beam of modalContinuousBeams) {
          for (let i = 0; i < beam.nodeIds.length; i++) {
            getPosInto(beam.nodeIds[i], beam.controlPts[i]);
          }
          updateContinuousBeamGeometry(
            beam.geometry,
            beam.controlPts,
            beam.controlValues,
            beam.width,
            beam.height,
            beam.segmentsPerSpan,
            true,
            beam.undeformedAxis,
            beam.fixedStart,
            beam.fixedEnd,
            false
          );
        }

        for (const shell of modalShellMeshes) {
          const posAttr = shell.geometry.getAttribute("position") as THREE.BufferAttribute;
          const posArr = posAttr.array as Float32Array;
          const ids = shell.vertNodeIds;
          for (let i = 0; i < ids.length; i++) {
            getPosInto(ids[i], tmpPosA);
            const o = i * 3;
            posArr[o] = tmpPosA.x;
            posArr[o + 1] = tmpPosA.y;
            posArr[o + 2] = tmpPosA.z;
          }
          posAttr.needsUpdate = true;
          shell.geometry.computeVertexNormals();

          const wirePosAttr = shell.wireGeometry.getAttribute("position") as THREE.BufferAttribute;
          (wirePosAttr.array as Float32Array).set(posArr);
          wirePosAttr.needsUpdate = true;
        }

        for (const item of modalMeshes) {
          orientBeamMesh(
            item.mesh,
            getPosInto(item.startId, tmpPosA),
            getPosInto(item.endId, tmpPosB)
          );
        }
        if (modalToolhead) {
          getPosInto(modalToolheadNodeId || "toolhead", modalToolhead.position);
        }
        if (modalCogMarker && modalCogNodeId) {
          getPosInto(modalCogNodeId, modalCogMarker.position);
        }
        if (modalNozzleMarker && modalNozzleNodeId) {
          const curNoz = getPosInto(modalNozzleNodeId, tmpPosA);
          modalNozzleMarker.position.copy(curNoz);
          if (modalNozzleRefNodeId && basePositions[modalNozzleRefNodeId] && basePositions[modalNozzleNodeId]) {
            tmpNozV0.subVectors(
              basePositions[modalNozzleRefNodeId],
              basePositions[modalNozzleNodeId]
            );
            const curRef = getPosInto(modalNozzleRefNodeId, tmpPosB);
            tmpNozV1.subVectors(curRef, curNoz);
            if (tmpNozV0.lengthSq() > 1e-6 && tmpNozV1.lengthSq() > 1e-6) {
              modalNozzleMarker.quaternion.setFromUnitVectors(
                tmpNozV0.normalize(),
                tmpNozV1.normalize()
              );
            }
          }
        }
        if (modalBedMesh && modalBedNodeId) {
          getPosInto(modalBedNodeId, modalBedMesh.position).add(modalBedOffset);
        }
      }

      renderer.render(scene, camera);
    };
    animate();

    return () => {
      window.removeEventListener("resize", onResize);
      renderer.domElement.removeEventListener("pointerdown", onPointerDown);
      renderer.domElement.removeEventListener("pointermove", onPointerMove);
      renderer.domElement.removeEventListener("pointerup", onPointerUp);
      renderer.domElement.removeEventListener("pointerleave", onPointerLeave);
      if (stateRef.current.animFrame) cancelAnimationFrame(stateRef.current.animFrame);
      disposeGroupChildren(dynamicGroup);
      grid.geometry.dispose();
      if (Array.isArray(grid.material)) grid.material.forEach((m) => m.dispose());
      else grid.material.dispose();
      controls.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === el) el.removeChild(renderer.domElement);
    };
  }, []);

  // Rebuild 3D geometry when simulation results or visualization mode changes
  useEffect(() => {
    const group = stateRef.current.dynamicGroup;
    if (!group) return;

    // Dispose and clear previous meshes cleanly
    disposeGroupChildren(group);

    stateRef.current.inspectableMeshes = [];
    stateRef.current.hoveredMesh = null;
    stateRef.current.pinnedMesh = null;
    stateRef.current.hoveredKey = null;
    setHoveredPart(null);
    if (viewMode === "solid_part" || viewMode === "topology") {
      stateRef.current.pinnedKey = null;
      setPinnedPart(null);
    }

    stateRef.current.modalMeshes = [];
    stateRef.current.modalContinuousBeams = [];
    stateRef.current.modalShellMeshes = [];
    stateRef.current.modalToolhead = undefined;
    stateRef.current.modalToolheadNodeId = undefined;
    stateRef.current.modalCogMarker = undefined;
    stateRef.current.modalCogNodeId = undefined;
    stateRef.current.modalNozzleMarker = undefined;
    stateRef.current.modalNozzleNodeId = undefined;
    stateRef.current.modalNozzleRefNodeId = undefined;
    stateRef.current.modalBedMesh = undefined;
    stateRef.current.modalBedNodeId = undefined;
    stateRef.current.isModal = viewMode === "modal";

    // Render 3D Solid Part FEA mode (Curved Tet10 Gmsh+scikit-fem OR Voxel Hex8)
    if (viewMode === "solid_part") {
      if (showTetPart && tetFea) {
        const sm = tetFea.surface_mesh;
        const verts = sm.vertices_mm;
        const tris = sm.triangles;
        if (verts.length === 0 || tris.length === 0) return;

        let minX = Infinity,
          maxX = -Infinity,
          minY = Infinity,
          maxY = -Infinity,
          minZ = Infinity,
          maxZ = -Infinity;
        for (const v of verts) {
          minX = Math.min(minX, v[0]);
          maxX = Math.max(maxX, v[0]);
          minY = Math.min(minY, v[1]);
          maxY = Math.max(maxY, v[1]);
          minZ = Math.min(minZ, v[2]);
          maxZ = Math.max(maxZ, v[2]);
        }

        const span = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 10);
        const partScale = 4.2 / span;
        const pcx = 0.5 * (minX + maxX);
        const pcy = 0.5 * (minY + maxY);
        const dispMult = feaScale * 0.08;
        const maxVm = Math.max(tetFea.max_von_mises_mpa, 0.05);

        const toPartThree = (vx: number, vy: number, vz: number): [number, number, number] => [
          (vx - pcx) * partScale,
          (vz - minZ) * partScale + 0.4,
          -(vy - pcy) * partScale,
        ];

        const geo = new THREE.BufferGeometry();
        const isPerTriangleStress =
          sm.von_mises_mpa.length === tris.length && tris.length !== verts.length;

        if (!isPerTriangleStress) {
          // Indexed smooth surface mesh with per-vertex stress heatmap
          const positions = new Float32Array(verts.length * 3);
          const colors = new Float32Array(verts.length * 3);
          for (let i = 0; i < verts.length; i++) {
            const v = verts[i];
            const d = sm.displacements_mm[i] || [0, 0, 0];
            const [tx, ty, tz] = toPartThree(
              v[0] + d[0] * dispMult,
              v[1] + d[1] * dispMult,
              v[2] + d[2] * dispMult
            );
            positions[i * 3] = tx;
            positions[i * 3 + 1] = ty;
            positions[i * 3 + 2] = tz;

            const vm = sm.von_mises_mpa[i] ?? 0;
            const col = stressColor(vm / maxVm);
            colors[i * 3] = col.r;
            colors[i * 3 + 1] = col.g;
            colors[i * 3 + 2] = col.b;
          }

          const indices = new Uint32Array(tris.length * 3);
          for (let t = 0; t < tris.length; t++) {
            indices[t * 3] = tris[t][0];
            indices[t * 3 + 1] = tris[t][1];
            indices[t * 3 + 2] = tris[t][2];
          }

          geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
          geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
          geo.setIndex(new THREE.BufferAttribute(indices, 1));
          geo.computeVertexNormals();
        } else {
          // Fallback non-indexed triangles if stress is per-triangle
          const positions = new Float32Array(tris.length * 9);
          const colors = new Float32Array(tris.length * 9);
          for (let t = 0; t < tris.length; t++) {
            const tri = tris[t];
            const col = stressColor((sm.von_mises_mpa[t] ?? 0) / maxVm);
            for (let k = 0; k < 3; k++) {
              const vi = tri[k];
              const v = verts[vi];
              const d = sm.displacements_mm[vi] || [0, 0, 0];
              const [tx, ty, tz] = toPartThree(
                v[0] + d[0] * dispMult,
                v[1] + d[1] * dispMult,
                v[2] + d[2] * dispMult
              );
              const off = t * 9 + k * 3;
              positions[off] = tx;
              positions[off + 1] = ty;
              positions[off + 2] = tz;
              colors[off] = col.r;
              colors[off + 1] = col.g;
              colors[off + 2] = col.b;
            }
          }
          geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
          geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
          geo.computeVertexNormals();
        }

        const surfaceMat = new THREE.MeshStandardMaterial({
          vertexColors: true,
          roughness: 0.32,
          metalness: 0.18,
          side: THREE.DoubleSide,
          polygonOffset: true,
          polygonOffsetFactor: 1,
          polygonOffsetUnits: 1,
        });
        const surfaceMesh = new THREE.Mesh(geo, surfaceMat);
        group.add(surfaceMesh);

        // Subtle conformal Gmsh tetrahedral wireframe overlay
        const wireMat = new THREE.MeshBasicMaterial({
          color: 0x0f172a,
          wireframe: true,
          transparent: true,
          opacity: 0.18,
          depthWrite: false,
        });
        const wireMesh = new THREE.Mesh(geo, wireMat);
        group.add(wireMesh);

        // Auto-frame camera nicely on undeformed part bounding box (CAD Z-up -> Three Y-up)
        const frameKey = `tet:${tetFea.part_name}:${tetFea.num_elements}:${tetFea.num_vertices}`;
        if (stateRef.current.lastFramedKey !== frameKey) {
          const partBox = new THREE.Box3(
            new THREE.Vector3((minX - pcx) * partScale, 0.4, -(maxY - pcy) * partScale),
            new THREE.Vector3(
              (maxX - pcx) * partScale,
              (maxZ - minZ) * partScale + 0.4,
              -(minY - pcy) * partScale
            )
          );
          frameCameraToBox(stateRef.current.camera, stateRef.current.controls, partBox);
          stateRef.current.lastFramedKey = frameKey;
        }
        return;
      }

      if (!solidFea || solidFea.voxels.length === 0) return;
      const voxels = solidFea.voxels;
      let minX = Infinity,
        maxX = -Infinity,
        minY = Infinity,
        maxY = -Infinity,
        minZ = Infinity,
        maxZ = -Infinity;
      for (const v of voxels) {
        minX = Math.min(minX, v.center_mm[0]);
        maxX = Math.max(maxX, v.center_mm[0]);
        minY = Math.min(minY, v.center_mm[1]);
        maxY = Math.max(maxY, v.center_mm[1]);
        minZ = Math.min(minZ, v.center_mm[2]);
        maxZ = Math.max(maxZ, v.center_mm[2]);
      }
      const span = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 10);
      const partScale = 4.2 / span; // Fit part nicely to ~4.2 scene units
      const pcx = 0.5 * (minX + maxX);
      const pcy = 0.5 * (minY + maxY);
      const dispMult = feaScale * 0.08;

      const boxSize = solidFea.voxel_size_mm * partScale * 0.96;
      const boxGeo = new THREE.BoxGeometry(boxSize, boxSize, boxSize);
      const boxMat = new THREE.MeshStandardMaterial({ roughness: 0.35, metalness: 0.25 });
      const inst = new THREE.InstancedMesh(boxGeo, boxMat, voxels.length);

      const maxVm = Math.max(solidFea.max_von_mises_mpa, 0.05);
      const dummy = new THREE.Object3D();
      for (let i = 0; i < voxels.length; i++) {
        const v = voxels[i];
        const x = (v.center_mm[0] + v.disp_mm[0] * dispMult - pcx) * partScale;
        const y = (v.center_mm[2] + v.disp_mm[2] * dispMult - minZ) * partScale + boxSize * 0.5 + 0.4;
        const z = -(v.center_mm[1] + v.disp_mm[1] * dispMult - pcy) * partScale;
        dummy.position.set(x, y, z);
        dummy.updateMatrix();
        inst.setMatrixAt(i, dummy.matrix);
        inst.setColorAt(i, stressColor(v.von_mises_mpa / maxVm));
      }
      inst.instanceMatrix.needsUpdate = true;
      if (inst.instanceColor) inst.instanceColor.needsUpdate = true;
      group.add(inst);

      const frameKey = `hex:${solidFea.part_name}:${solidFea.num_elements}`;
      if (stateRef.current.lastFramedKey !== frameKey) {
        const halfBox = boxSize * 0.5;
        const partBox = new THREE.Box3(
          new THREE.Vector3((minX - pcx) * partScale - halfBox, 0.4, -(maxY - pcy) * partScale - halfBox),
          new THREE.Vector3(
            (maxX - pcx) * partScale + halfBox,
            (maxZ - minZ) * partScale + boxSize + 0.4,
            -(minY - pcy) * partScale + halfBox
          )
        );
        frameCameraToBox(stateRef.current.camera, stateRef.current.controls, partBox);
        stateRef.current.lastFramedKey = frameKey;
      }
      return;
    }

    // Render 3D SIMP Topology Optimization mode
    if (viewMode === "topology") {
      if (!topoResult || topoResult.voxels.length === 0) return;
      const allVoxels = topoResult.voxels;
      let minX = Infinity,
        maxX = -Infinity,
        minY = Infinity,
        maxY = -Infinity,
        minZ = Infinity,
        maxZ = -Infinity;
      for (const v of allVoxels) {
        minX = Math.min(minX, v.center_mm[0]);
        maxX = Math.max(maxX, v.center_mm[0]);
        minY = Math.min(minY, v.center_mm[1]);
        maxY = Math.max(maxY, v.center_mm[1]);
        minZ = Math.min(minZ, v.center_mm[2]);
        maxZ = Math.max(maxZ, v.center_mm[2]);
      }
      const span = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 10);
      const partScale = 4.2 / span;
      const pcx = 0.5 * (minX + maxX);
      const pcy = 0.5 * (minY + maxY);
      const dispMult = feaScale * 0.08;
      const boxSize = topoResult.voxel_size_mm * partScale * 0.96;

      const keptVoxels = allVoxels.filter((v) => v.kept);
      const removedVoxels = allVoxels.filter((v) => !v.kept);
      const dummy = new THREE.Object3D();

      if (keptVoxels.length > 0) {
        const keptGeo = new THREE.BoxGeometry(boxSize, boxSize, boxSize);
        const keptMat = new THREE.MeshStandardMaterial({ roughness: 0.35, metalness: 0.25 });
        const keptInst = new THREE.InstancedMesh(keptGeo, keptMat, keptVoxels.length);
        const maxVm = Math.max(topoResult.max_von_mises_mpa, 0.05);

        for (let i = 0; i < keptVoxels.length; i++) {
          const v = keptVoxels[i];
          const x = (v.center_mm[0] + v.disp_mm[0] * dispMult - pcx) * partScale;
          const y = (v.center_mm[2] + v.disp_mm[2] * dispMult - minZ) * partScale + boxSize * 0.5 + 0.4;
          const z = -(v.center_mm[1] + v.disp_mm[1] * dispMult - pcy) * partScale;
          dummy.position.set(x, y, z);
          dummy.updateMatrix();
          keptInst.setMatrixAt(i, dummy.matrix);
          keptInst.setColorAt(i, stressColor(v.von_mises_mpa / maxVm));
        }
        keptInst.instanceMatrix.needsUpdate = true;
        if (keptInst.instanceColor) keptInst.instanceColor.needsUpdate = true;
        group.add(keptInst);
      }

      if (showCarvedGhost && removedVoxels.length > 0) {
        const ghostSize = topoResult.voxel_size_mm * partScale * 0.94;
        const ghostFillGeo = new THREE.BoxGeometry(ghostSize, ghostSize, ghostSize);
        const ghostFillMat = new THREE.MeshBasicMaterial({
          color: 0x94a3b8,
          transparent: true,
          opacity: 0.14,
          depthWrite: false,
        });
        const ghostFillInst = new THREE.InstancedMesh(ghostFillGeo, ghostFillMat, removedVoxels.length);

        const ghostWireGeo = new THREE.BoxGeometry(ghostSize, ghostSize, ghostSize);
        const ghostWireMat = new THREE.MeshBasicMaterial({
          color: 0x64748b,
          wireframe: true,
          transparent: true,
          opacity: 0.14,
          depthWrite: false,
        });
        const ghostWireInst = new THREE.InstancedMesh(ghostWireGeo, ghostWireMat, removedVoxels.length);

        for (let i = 0; i < removedVoxels.length; i++) {
          const v = removedVoxels[i];
          const x = (v.center_mm[0] - pcx) * partScale;
          const y = (v.center_mm[2] - minZ) * partScale + boxSize * 0.5 + 0.4;
          const z = -(v.center_mm[1] - pcy) * partScale;
          dummy.position.set(x, y, z);
          dummy.updateMatrix();
          ghostFillInst.setMatrixAt(i, dummy.matrix);
          ghostWireInst.setMatrixAt(i, dummy.matrix);
        }
        ghostFillInst.instanceMatrix.needsUpdate = true;
        ghostWireInst.instanceMatrix.needsUpdate = true;
        group.add(ghostFillInst);
        group.add(ghostWireInst);
      }
      return;
    }

    if (!sim) return;

    // Restore full-printer camera framing when returning from a solid part view
    if (stateRef.current.lastFramedKey && stateRef.current.lastFramedKey !== "printer") {
      const cam = stateRef.current.camera;
      const ctrl = stateRef.current.controls;
      if (cam && ctrl) {
        cam.position.set(7.5, 6.2, 8.2);
        ctrl.target.set(0, 2.3, 0);
        ctrl.update();
      }
      stateRef.current.lastFramedKey = "printer";
    }

    const geom = sim.geometry;
    const cx = geom.frame_dims_mm[0] * 0.5;
    const cy = geom.frame_dims_mm[1] * 0.5;

    const basePos: Record<string, THREE.Vector3> = {};
    const deformedPos: Record<string, THREE.Vector3> = {};
    const fixedNodeIds = new Set<string>();

    const activeNodalDisplacements: Record<string, [number, number, number]> =
      feaLoadCase === "x"
        ? sim.static_fea.nodal_displacements_x_mm ?? sim.static_fea.nodal_displacements_mm
        : feaLoadCase === "y"
          ? sim.static_fea.nodal_displacements_y_mm ?? sim.static_fea.nodal_displacements_mm
          : feaLoadCase === "gravity"
            ? sim.static_fea.nodal_displacements_mm
            : sim.static_fea.nodal_displacements_xy_mm ?? sim.static_fea.nodal_displacements_mm;

    for (const n of geom.nodes) {
      const p0 = toThree(n.position_mm, cx, cy);
      basePos[n.id] = p0;
      if (n.fixed) fixedNodeIds.add(n.id);

      const disp = activeNodalDisplacements[n.id] || [0, 0, 0];
      const dispVec = new THREE.Vector3(disp[0] * SCALE, disp[2] * SCALE, -disp[1] * SCALE);
      deformedPos[n.id] = p0.clone().addScaledVector(dispVec, feaScale);
    }
    stateRef.current.basePositions = basePos;

    // Modal displacement vectors
    const modeIdx =
      sim.modal_fea.modes.length > 0
        ? Math.min(selectedModeIdx, sim.modal_fea.modes.length - 1)
        : 0;
    const activeMode = sim.modal_fea.modes[modeIdx];
    const mVecs: Record<string, THREE.Vector3> = {};
    let maxModalAmp = 0;
    if (activeMode) {
      for (const [nid, vec] of Object.entries(activeMode.nodal_vectors)) {
        const mv = new THREE.Vector3(vec[0], vec[2], -vec[1]);
        mVecs[nid] = mv;
        const amp = mv.length();
        if (amp > maxModalAmp) maxModalAmp = amp;
      }
    }
    const modalAmpNorm = maxModalAmp > 1e-6 ? maxModalAmp : 1.0;
    stateRef.current.modalVectors = mVecs;

    const bedNodeIdEarly = geom.bed_node_id || "bed_surface";
    const bedModalVec = mVecs[bedNodeIdEarly] ?? new THREE.Vector3(0, 0, 0);
    let maxRelModalAmp = 0;
    for (const n of geom.nodes) {
      const mv = mVecs[n.id];
      const relAmp = mv ? mv.distanceTo(bedModalVec) : bedModalVec.length();
      if (relAmp > maxRelModalAmp) maxRelModalAmp = relAmp;
    }
    const relModalNorm = Math.max(modalAmpNorm, maxRelModalAmp, 1e-6);
    const getModalNodeScalar = (nid: string): number => {
      const mv = mVecs[nid];
      if (modalColorMode === "relative") {
        const relDist = mv ? mv.distanceTo(bedModalVec) : bedModalVec.length();
        return Math.min(1, relDist / relModalNorm);
      }
      return Math.min(1, (mv?.length() ?? 0) / modalAmpNorm);
    };

    // Stress lookup (member-level and smooth nodal averages for corner blending)
    const stressMap: Record<string, number> = {};
    const memberStressById: Record<string, SimulationResponse["static_fea"]["member_stresses"][number]> = {};
    const maxStress = Math.max(
      sim.static_fea.max_von_mises_mpa,
      sim.static_fea.panel_max_von_mises_mpa ?? 0,
      0.05
    );
    for (const ms of sim.static_fea.member_stresses) {
      stressMap[ms.member_id] = ms.von_mises_mpa / maxStress;
      memberStressById[ms.member_id] = ms;
    }

    const stiffenById: Record<string, NonNullable<SimulationResponse["sensitivity"]>["top_stiffen_targets"][number]> = {};
    for (const t of sim.sensitivity?.top_stiffen_targets ?? []) {
      stiffenById[t.id] = t;
    }
    const lightenById: Record<string, NonNullable<SimulationResponse["sensitivity"]>["top_lighten_targets"][number]> = {};
    for (const t of sim.sensitivity?.top_lighten_targets ?? []) {
      lightenById[t.id] = t;
    }

    const tagInspectable = (mesh: THREE.Mesh, info: InspectedPartInfo) => {
      mesh.userData.inspectInfo = info;
      stateRef.current.inspectableMeshes.push(mesh);
    };

    // Include bed spider arms & Z-rail posts (role === "bed") so real 3D bed supports render cleanly,
    // while excluding toolhead lever-arm members (rendered as sleek carriage->CoG->nozzle struts)
    // and solver-internal equivalent shear-panel X-struts (panel_shear_*).
    const structuralMembers = geom.members.filter(
      (m) => m.role !== "toolhead" && !m.id.startsWith("panel_shear_")
    );

    const nodalStressSum: Record<string, number> = {};
    const nodalStressCount: Record<string, number> = {};
    for (const m of structuralMembers) {
      const s = stressMap[m.id] ?? 0;
      nodalStressSum[m.start_id] = (nodalStressSum[m.start_id] ?? 0) + s;
      nodalStressCount[m.start_id] = (nodalStressCount[m.start_id] ?? 0) + 1;
      nodalStressSum[m.end_id] = (nodalStressSum[m.end_id] ?? 0) + s;
      nodalStressCount[m.end_id] = (nodalStressCount[m.end_id] ?? 0) + 1;
    }
    const globalNodalStress: Record<string, number> = {};
    for (const nid of Object.keys(nodalStressSum)) {
      globalNodalStress[nid] = nodalStressSum[nid] / Math.max(nodalStressCount[nid], 1);
    }

    // Build continuous structural members (merges split columns, Y-rails, X-gantry, etc.)
    const useVertexColors = viewMode === "fea_stress" || viewMode === "modal";
    const frameOpacity = viewMode === "toolpath" ? 0.25 : 1.0;
    const activeFilling =
      viewMode === "assembly" && geom.extrusion_filling && geom.extrusion_filling.id !== "none"
        ? geom.extrusion_filling
        : null;
    const fillingMat = activeFilling
      ? new THREE.MeshStandardMaterial({
          color: new THREE.Color(activeFilling.color_hex),
          metalness: activeFilling.id === "lead_shot" ? 0.55 : 0.12,
          roughness: activeFilling.bonded ? 0.48 : 0.72,
        })
      : null;
    let fillingMatUsed = false;

    const ghostLineMat = useVertexColors
      ? new THREE.LineBasicMaterial({ color: 0x475569, transparent: true, opacity: 0.35 })
      : null;
    let ghostLineMatUsed = false;

    const beamChains: ContinuousBeamChain[] =
      viewMode === "sensitivity"
        ? structuralMembers.map((m) => ({
            nodeIds: [m.start_id, m.end_id],
            members: [m],
            template: m,
            isRod: m.shape === "rod",
          }))
        : groupContinuousBeams(structuralMembers, basePos, fixedNodeIds);

    const memberSensMap = sim.sensitivity?.member_sensitivities ?? {};

    for (const chain of beamChains) {
      const m = chain.template;
      const firstId = chain.nodeIds[0];
      const lastId = chain.nodeIds[chain.nodeIds.length - 1];
      if (!basePos[firstId] || !basePos[lastId]) continue;

      const isBedMember = m.role === "bed";
      const isBedPost =
        isBedMember &&
        (m.id.includes("z_rail") ||
          m.id.includes("z_post") ||
          m.id.includes("leadscrew") ||
          m.id.includes("z_column"));
      const rawW = Math.max(m.width_mm * SCALE, isBedMember ? 0.05 : 0.06);
      const rawH = Math.max(m.height_mm * SCALE, isBedMember ? 0.05 : 0.06);
      const w = isBedMember ? Math.min(rawW, isBedPost ? 0.13 : 0.11) : rawW;
      const h = isBedMember ? Math.min(rawH, isBedPost ? 0.13 : 0.11) : rawH;

      // Compute smooth nodal scalar values for FEA stress & Modal heatmap gradients
      const controlValues: number[] = [];
      if (viewMode === "modal") {
        for (const nid of chain.nodeIds) {
          controlValues.push(getModalNodeScalar(nid));
        }
      } else if (viewMode === "fea_stress") {
        const nCount = chain.nodeIds.length;
        for (let i = 0; i < nCount; i++) {
          const nid = chain.nodeIds[i];
          if (i > 0 && i < nCount - 1) {
            const sPrev = stressMap[chain.members[i - 1].id] ?? 0;
            const sNext = stressMap[chain.members[i].id] ?? 0;
            controlValues.push(0.5 * (sPrev + sNext));
          } else if (i === 0) {
            const sFirst = stressMap[chain.members[0].id] ?? 0;
            controlValues.push(
              fixedNodeIds.has(nid)
                ? sFirst
                : 0.5 * sFirst + 0.5 * (globalNodalStress[nid] ?? sFirst)
            );
          } else {
            const sLast = stressMap[chain.members[chain.members.length - 1].id] ?? 0;
            controlValues.push(
              fixedNodeIds.has(nid)
                ? sLast
                : 0.5 * sLast + 0.5 * (globalNodalStress[nid] ?? sLast)
            );
          }
        }
      } else {
        for (let i = 0; i < chain.nodeIds.length; i++) {
          controlValues.push(0);
        }
      }

      let matColor = new THREE.Color(0xffffff);
      let sensEmissive = new THREE.Color(0x000000);
      let sensEmissiveIntensity = 0;

      if (viewMode === "sensitivity") {
        const heatScore = getLensScore(memberSensMap[m.id], adjointLens);
        sensitivityColorInto(heatScore, matColor);
        const absScore = Math.min(1, Math.abs(heatScore));
        if (absScore > 0.2) {
          sensEmissive.copy(matColor);
          sensEmissiveIntensity = absScore * 0.3;
        }
      } else if (!useVertexColors) {
        matColor = new THREE.Color(m.color_hex);
        if (m.role === "brace") {
          matColor = new THREE.Color("#38bdf8");
        } else if (m.role === "gantry") {
          matColor = m.material.includes("carbon")
            ? new THREE.Color("#1f2937")
            : new THREE.Color("#f59e0b");
        } else if (m.role === "bed") {
          matColor = isBedPost
            ? new THREE.Color("#7c8aa0")
            : m.id.includes("spring")
              ? new THREE.Color("#94a3b8")
              : new THREE.Color("#475569");
        }
      }

      const mat = new THREE.MeshStandardMaterial({
        color: matColor,
        vertexColors: useVertexColors,
        emissive: sensEmissive,
        emissiveIntensity: sensEmissiveIntensity,
        metalness:
          viewMode === "sensitivity"
            ? 0.22
            : useVertexColors
              ? 0.25
              : m.material.includes("carbon")
                ? 0.15
                : 0.55,
        roughness:
          viewMode === "sensitivity"
            ? 0.42
            : useVertexColors
              ? 0.4
              : m.material.includes("carbon")
                ? 0.65
                : 0.35,
        transparent: frameOpacity < 1,
        opacity: frameOpacity,
      });

      const grp = classifyPartGroup(m.id, m.role);
      let chainMaxVm = 0;
      let chainYield: number | undefined;
      let chainCompShare = 0;
      let chainHeatScore = 0;
      let chainHzPer100g = 0;
      let hasSensEntry = false;
      for (const cm of chain.members) {
        const ms = memberStressById[cm.id];
        if (ms && ms.von_mises_mpa >= chainMaxVm) {
          chainMaxVm = ms.von_mises_mpa;
          chainYield = ms.yield_mpa;
        }
        const se = memberSensMap[cm.id];
        if (se) {
          hasSensEntry = true;
          chainCompShare += se.compliance_share_pct ?? 0;
          const sc = getLensScore(se, adjointLens);
          if (Math.abs(sc) > Math.abs(chainHeatScore)) chainHeatScore = sc;
          chainHzPer100g += se.hz_per_100g ?? 0;
        }
      }
      const stiffTarget = stiffenById[grp.id] ?? stiffenById[m.id];
      const lightTarget = lightenById[grp.id] ?? lightenById[m.id];
      const chainInspectInfo: InspectedPartInfo = {
        key: `beam:${chain.members.map((cm) => cm.id).join("+")}`,
        name: stiffTarget?.label ?? lightTarget?.label ?? grp.label,
        subsystem: grp.subsystem,
        partGroupId: grp.id,
        role: m.role,
        material: m.material,
        profile: m.profile,
        complianceSharePct:
          stiffTarget?.compliance_share_pct ?? (hasSensEntry ? chainCompShare : undefined),
        heatScore: hasSensEntry ? chainHeatScore : undefined,
        stressMpa: chainMaxVm > 0 ? chainMaxVm : undefined,
        yieldMpa: chainYield,
        estHzGain: stiffTarget?.est_hz_gain,
        hzPer100g: lightTarget?.hz_per_100g ?? (hasSensEntry ? chainHzPer100g : undefined),
        whyItHurts: stiffTarget?.why_it_hurts ?? lightTarget?.why_it_hurts,
        actionHint: stiffTarget?.action_hint ?? lightTarget?.action_hint,
      };

      if (chain.isRod) {
        const geo = new THREE.CylinderGeometry(w * 0.5, w * 0.5, 1, 10, useVertexColors ? 6 : 1);
        if (useVertexColors) {
          applyRodGradientColors(geo, controlValues[0] ?? 0, controlValues[1] ?? controlValues[0] ?? 0);
        }
        const mesh = new THREE.Mesh(geo, mat);
        const p0 = viewMode === "fea_stress" ? deformedPos[firstId] : basePos[firstId];
        const p1 = viewMode === "fea_stress" ? deformedPos[lastId] : basePos[lastId];
        orientBeamMesh(mesh, p0, p1);
        tagInspectable(mesh, chainInspectInfo);
        group.add(mesh);

        if (viewMode === "modal") {
          stateRef.current.modalMeshes.push({ mesh, startId: firstId, endId: lastId });
        }
      } else {
        const baseFirst = basePos[firstId];
        const baseLast = basePos[lastId];
        const undeformedAxis = new THREE.Vector3().subVectors(baseLast, baseFirst);
        if (undeformedAxis.lengthSq() > 1e-10) {
          undeformedAxis.normalize();
        } else {
          undeformedAxis.set(0, 1, 0);
        }

        const fixedStart = fixedNodeIds.has(firstId);
        const fixedEnd = fixedNodeIds.has(lastId);
        const segmentsPerSpan = useVertexColors ? (chain.nodeIds.length >= 3 ? 8 : 6) : 1;
        const controlPts = chain.nodeIds.map((nid) =>
          (viewMode === "fea_stress" ? deformedPos[nid] || basePos[nid] : basePos[nid]).clone()
        );

        const geo = createContinuousBeamGeometry(
          controlPts,
          controlValues,
          w,
          h,
          segmentsPerSpan,
          useVertexColors,
          undeformedAxis,
          fixedStart,
          fixedEnd
        );
        const mesh = new THREE.Mesh(geo, mat);
        if (viewMode === "modal") {
          mesh.frustumCulled = false;
          stateRef.current.modalContinuousBeams.push({
            geometry: geo,
            nodeIds: chain.nodeIds,
            controlPts,
            controlValues,
            width: w,
            height: h,
            segmentsPerSpan,
            undeformedAxis,
            fixedStart,
            fixedEnd,
          });
        }
        tagInspectable(mesh, chainInspectInfo);
        group.add(mesh);

        // Continuous T-slot core accent strip on filled stationary frame extrusions (Assembly mode only)
        if (fillingMat && m.role === "frame") {
          fillingMatUsed = true;
          const inlayGeoA = new THREE.BoxGeometry(w * 0.32, 0.96, h * 1.03);
          const inlayA = new THREE.Mesh(inlayGeoA, fillingMat);
          orientBeamMesh(inlayA, baseFirst, baseLast);
          group.add(inlayA);

          const inlayGeoB = new THREE.BoxGeometry(w * 1.03, 0.96, h * 0.32);
          const inlayB = new THREE.Mesh(inlayGeoB, fillingMat);
          orientBeamMesh(inlayB, baseFirst, baseLast);
          group.add(inlayB);
        }
      }

      // Single clean undeformed ghost reference line per continuous beam in FEA / Modal mode
      if (ghostLineMat) {
        ghostLineMatUsed = true;
        const lineGeo = new THREE.BufferGeometry().setFromPoints([
          basePos[firstId],
          basePos[lastId],
        ]);
        group.add(new THREE.Line(lineGeo, ghostLineMat));
      }
    }

    if (fillingMat && !fillingMatUsed) {
      fillingMat.dispose();
    }
    if (ghostLineMat && !ghostLineMatUsed) {
      ghostLineMat.dispose();
    }

    // Foot anchor markers & benchtop isolator pads (exclude bed and MITC4 shell panel nodes)
    const feetType: FrameFeetType = geom.feet_type ?? sim.spec.frame.feet_type ?? "rigid_bolted";
    const footCandidates = geom.nodes.filter(
      (n) =>
        (n.fixed || n.id.startsWith("foot_") || n.id.startsWith("ground_")) &&
        n.role !== "bed" &&
        n.role !== "panel" &&
        Boolean(basePos[n.id])
    );
    const footSeen = new Set<string>();
    const footNodes = footCandidates.filter((n) => {
      const p = basePos[n.id];
      const key = `${Math.round(p.x * 10)},${Math.round(p.z * 10)}`;
      if (footSeen.has(key)) return false;
      footSeen.add(key);
      return true;
    });
    if (footNodes.length > 0) {
      const feetStiff = stiffenById["frame_feet"];
      let feetCompShare = 0;
      let feetHeatScore = 0;
      let hasFeetSens = false;
      for (const [mid, entry] of Object.entries(memberSensMap)) {
        if (mid.startsWith("feet_")) {
          hasFeetSens = true;
          feetCompShare += entry.compliance_share_pct ?? 0;
          const sc = getLensScore(entry, adjointLens);
          if (Math.abs(sc) > Math.abs(feetHeatScore)) feetHeatScore = sc;
        }
      }
      const feetInspectInfo: InspectedPartInfo = {
        key: "frame_feet",
        name: feetStiff?.label ?? "Machine Feet & Benchtop Pads",
        subsystem: "frame",
        partGroupId: "frame_feet",
        role: "foot",
        material: feetType.replace(/_/g, " "),
        complianceSharePct:
          feetStiff?.compliance_share_pct ?? (hasFeetSens ? feetCompShare : undefined),
        heatScore: hasFeetSens ? feetHeatScore : undefined,
        estHzGain: feetStiff?.est_hz_gain,
        whyItHurts: feetStiff?.why_it_hurts,
        actionHint:
          feetStiff?.action_hint ??
          (feetType !== "rigid_bolted"
            ? "Compliant benchtop feet allow whole-frame rocking motion"
            : undefined),
      };

      if (feetType === "rubber_feet") {
        const rubberBodyGeo = new THREE.CylinderGeometry(0.15, 0.19, 0.14, 14);
        const rubberBodyMat = new THREE.MeshStandardMaterial({
          color: 0x1e293b,
          roughness: 0.86,
          metalness: 0.08,
        });
        const rubberRingGeo = new THREE.CylinderGeometry(0.17, 0.17, 0.032, 14);
        const rubberRingMat = new THREE.MeshStandardMaterial({
          color: 0xf59e0b,
          roughness: 0.55,
          metalness: 0.15,
        });
        for (const n of footNodes) {
          const p = basePos[n.id];
          const footGrp = new THREE.Group();
          const body = new THREE.Mesh(rubberBodyGeo, rubberBodyMat);
          body.position.y = -0.05;
          tagInspectable(body, feetInspectInfo);
          footGrp.add(body);
          const ring = new THREE.Mesh(rubberRingGeo, rubberRingMat);
          ring.position.y = -0.02;
          footGrp.add(ring);
          footGrp.position.copy(p);
          group.add(footGrp);
        }
      } else if (feetType === "anti_vib_pads") {
        const plateGeo = new THREE.CylinderGeometry(0.19, 0.19, 0.035, 16);
        const plateMat = new THREE.MeshStandardMaterial({
          color: 0x334155,
          metalness: 0.65,
          roughness: 0.35,
        });
        const isoCoreGeo = new THREE.CylinderGeometry(0.165, 0.165, 0.09, 16);
        const isoCoreMat = new THREE.MeshStandardMaterial({
          color: 0x06b6d4,
          emissive: 0x0891b2,
          emissiveIntensity: 0.25,
          roughness: 0.65,
          metalness: 0.1,
        });
        for (const n of footNodes) {
          const p = basePos[n.id];
          const hulaGrp = new THREE.Group();
          const topPlate = new THREE.Mesh(plateGeo, plateMat);
          topPlate.position.y = 0.0;
          hulaGrp.add(topPlate);
          const core = new THREE.Mesh(isoCoreGeo, isoCoreMat);
          core.position.y = -0.06;
          tagInspectable(core, feetInspectInfo);
          hulaGrp.add(core);
          const botPlate = new THREE.Mesh(plateGeo, plateMat);
          botPlate.position.y = -0.12;
          hulaGrp.add(botPlate);
          hulaGrp.position.copy(p);
          group.add(hulaGrp);
        }
      } else {
        const boltPadGeo = new THREE.CylinderGeometry(0.14, 0.17, 0.08, 12);
        const boltPadMat = new THREE.MeshStandardMaterial({ color: 0x22c55e, roughness: 0.5 });
        for (const n of footNodes) {
          const pad = new THREE.Mesh(boltPadGeo, boltPadMat);
          pad.position.copy(basePos[n.id]);
          tagInspectable(pad, feetInspectInfo);
          group.add(pad);
        }
      }
    }

    // TRUE 2D MITC4 Shell Panel 3D Stress Heatmaps (fea_stress), Live 3D Panel Modes (modal),
    // and Adjoint Sensitivity ROI Bay Heatmaps (sensitivity)
    if (
      (viewMode === "fea_stress" || viewMode === "modal" || viewMode === "sensitivity") &&
      geom.shell_panels &&
      geom.shell_panels.length > 0
    ) {
      const validPanels = geom.shell_panels.filter(
        (p) =>
          p.node_ids.length === 4 &&
          Boolean(basePos[p.node_ids[0]]) &&
          Boolean(basePos[p.node_ids[1]]) &&
          Boolean(basePos[p.node_ids[2]]) &&
          Boolean(basePos[p.node_ids[3]])
      );

      if (validPanels.length > 0) {
        // Nodal-averaged MITC4 shell von Mises stress per bay (and global fallback)
        const panelStressSum: Record<string, number> = {};
        const panelStressCount: Record<string, number> = {};
        if (viewMode === "fea_stress" && sim.static_fea.shell_panel_stresses) {
          for (const sps of sim.static_fea.shell_panel_stresses) {
            const vm = sps.von_mises_mpa;
            for (const nid of sps.node_ids) {
              const bayKey = `${sps.bay}:${nid}`;
              panelStressSum[bayKey] = (panelStressSum[bayKey] ?? 0) + vm;
              panelStressCount[bayKey] = (panelStressCount[bayKey] ?? 0) + 1;
              panelStressSum[nid] = (panelStressSum[nid] ?? 0) + vm;
              panelStressCount[nid] = (panelStressCount[nid] ?? 0) + 1;
            }
          }
        }

        const bayMaxVmMap: Record<string, number> = {};
        if (sim.static_fea.shell_panel_stresses) {
          for (const sps of sim.static_fea.shell_panel_stresses) {
            bayMaxVmMap[sps.bay] = Math.max(bayMaxVmMap[sps.bay] ?? 0, sps.von_mises_mpa);
          }
        }

        const addShellLayer = (
          panels: ShellPanelGeometryData[],
          opacity: number,
          bayName: string
        ) => {
          if (panels.length === 0) return;

          // Key vertices by bay:nodeId so interior panel shading is smooth while 90° bay corners stay crisp
          const keyToVertIdx = new Map<string, number>();
          const vertNodeIds: string[] = [];
          const vertBays: string[] = [];

          const getVertIdx = (bay: string, nid: string): number => {
            const key = `${bay}:${nid}`;
            let idx = keyToVertIdx.get(key);
            if (idx === undefined) {
              idx = vertNodeIds.length;
              keyToVertIdx.set(key, idx);
              vertNodeIds.push(nid);
              vertBays.push(bay);
            }
            return idx;
          };

          const quadVerts: Array<[number, number, number, number]> = [];
          for (const panel of panels) {
            const i0 = getVertIdx(panel.bay, panel.node_ids[0]);
            const i1 = getVertIdx(panel.bay, panel.node_ids[1]);
            const i2 = getVertIdx(panel.bay, panel.node_ids[2]);
            const i3 = getVertIdx(panel.bay, panel.node_ids[3]);
            quadVerts.push([i0, i1, i2, i3]);
          }

          const vertCount = vertNodeIds.length;
          const positions = new Float32Array(vertCount * 3);
          const colors = new Float32Array(vertCount * 3);

          for (let i = 0; i < vertCount; i++) {
            const nid = vertNodeIds[i];
            const bay = vertBays[i];
            const pt =
              viewMode === "fea_stress"
                ? deformedPos[nid] || basePos[nid]
                : basePos[nid];
            const o = i * 3;
            positions[o] = pt.x;
            positions[o + 1] = pt.y;
            positions[o + 2] = pt.z;

            if (viewMode === "sensitivity") {
              const bayScore = getLensScore(
                sim.sensitivity?.panel_bay_sensitivities?.[bay],
                adjointLens
              );
              sensitivityColorInto(bayScore, _beamCol);
            } else {
              let scalar = 0;
              if (viewMode === "fea_stress") {
                const bayKey = `${bay}:${nid}`;
                const avgVm = panelStressCount[bayKey]
                  ? panelStressSum[bayKey] / panelStressCount[bayKey]
                  : panelStressCount[nid]
                    ? panelStressSum[nid] / panelStressCount[nid]
                    : 0;
                scalar = avgVm / maxStress;
              } else {
                scalar = getModalNodeScalar(nid);
              }
              stressColorInto(scalar, _beamCol);
            }
            colors[o] = _beamCol.r;
            colors[o + 1] = _beamCol.g;
            colors[o + 2] = _beamCol.b;
          }

          // 2 triangles per MITC4 quad: [n0, n1, n2] and [n0, n2, n3]
          const triIndices =
            vertCount <= 65535
              ? new Uint16Array(quadVerts.length * 6)
              : new Uint32Array(quadVerts.length * 6);
          const edgeSeen = new Set<number>();
          const edgeList: number[] = [];
          const pushEdge = (a: number, b: number) => {
            const lo = a < b ? a : b;
            const hi = a < b ? b : a;
            const k = lo * vertCount + hi;
            if (!edgeSeen.has(k)) {
              edgeSeen.add(k);
              edgeList.push(lo, hi);
            }
          };

          let tOff = 0;
          for (const [i0, i1, i2, i3] of quadVerts) {
            triIndices[tOff++] = i0;
            triIndices[tOff++] = i1;
            triIndices[tOff++] = i2;
            triIndices[tOff++] = i0;
            triIndices[tOff++] = i2;
            triIndices[tOff++] = i3;
            pushEdge(i0, i1);
            pushEdge(i1, i2);
            pushEdge(i2, i3);
            pushEdge(i3, i0);
          }

          const shellGeo = new THREE.BufferGeometry();
          shellGeo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
          shellGeo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
          shellGeo.setIndex(new THREE.BufferAttribute(triIndices, 1));
          shellGeo.computeVertexNormals();

          const shellMat = new THREE.MeshStandardMaterial({
            vertexColors: true,
            transparent: true,
            opacity,
            depthWrite: false,
            roughness: 0.36,
            metalness: 0.14,
            side: THREE.DoubleSide,
            polygonOffset: true,
            polygonOffsetFactor: 1,
            polygonOffsetUnits: 1,
          });
          const shellMesh = new THREE.Mesh(shellGeo, shellMat);
          shellMesh.renderOrder = 1;

          const bayGrp = classifyPartGroup(`bay_${bayName}`, "panel");
          const baySens = sim.sensitivity?.panel_bay_sensitivities?.[bayName];
          const bayStiff = stiffenById[bayGrp.id] ?? stiffenById[`bay_${bayName}`];
          const bayLabel = `${bayName.charAt(0).toUpperCase()}${bayName.slice(1)} Enclosure Panel (MITC4)`;
          tagInspectable(shellMesh, {
            key: `panel:${bayName}`,
            name: bayLabel,
            subsystem: "frame",
            partGroupId: bayGrp.id,
            role: "panel",
            material: geom.panels?.material_name ?? sim.spec.frame.panel_material,
            profile: `${panels[0]?.thickness_mm ?? sim.spec.frame.panel_thickness_mm}mm MITC4 shell`,
            complianceSharePct: baySens?.compliance_share_pct ?? bayStiff?.compliance_share_pct,
            heatScore: baySens ? getLensScore(baySens, adjointLens) : undefined,
            stressMpa: bayMaxVmMap[bayName] > 0 ? bayMaxVmMap[bayName] : undefined,
            estHzGain: bayStiff?.est_hz_gain,
            hzPer100g: baySens?.hz_per_100g,
            whyItHurts: bayStiff?.why_it_hurts,
            actionHint: bayStiff?.action_hint,
          });
          group.add(shellMesh);

          // Subtle dark MITC4 quad wireframe grid overlay (2×2 or 6×6)
          const wirePositions = new Float32Array(positions);
          const wireIndices =
            vertCount <= 65535 ? new Uint16Array(edgeList) : new Uint32Array(edgeList);
          const wireGeo = new THREE.BufferGeometry();
          wireGeo.setAttribute("position", new THREE.BufferAttribute(wirePositions, 3));
          wireGeo.setIndex(new THREE.BufferAttribute(wireIndices, 1));
          const wireMat = new THREE.LineBasicMaterial({
            color: 0x0f172a,
            transparent: true,
            opacity: opacity >= 0.5 ? 0.5 : 0.3,
          });
          const wireLines = new THREE.LineSegments(wireGeo, wireMat);
          wireLines.renderOrder = 2;
          group.add(wireLines);

          if (viewMode === "modal") {
            shellMesh.frustumCulled = false;
            wireLines.frustumCulled = false;
            stateRef.current.modalShellMeshes.push({
              geometry: shellGeo,
              wireGeometry: wireGeo,
              vertNodeIds,
            });
          }
        };

        const panelsByBay = new Map<string, ShellPanelGeometryData[]>();
        for (const p of validPanels) {
          const list = panelsByBay.get(p.bay);
          if (list) list.push(p);
          else panelsByBay.set(p.bay, [p]);
        }
        panelsByBay.forEach((bayPanels, bayName) => {
          const bayOpacity = bayName === "top" ? 0.28 : viewMode === "sensitivity" ? 0.65 : 0.72;
          addShellLayer(bayPanels, bayOpacity, bayName);
        });
      }
    }

    // Assembly-only visual feedback: NEMA17 stepper motors, gantry brackets & 3D structural panels
    if (viewMode === "assembly") {
      const bracketInfo = geom.gantry_brackets;
      const bracketColor = new THREE.Color(bracketInfo?.color_hex ?? "#94a3b8");
      const bracketIsCnc = Boolean(bracketInfo?.cnc_machined);
      const bracketMat = new THREE.MeshStandardMaterial({
        color: bracketColor,
        metalness: bracketIsCnc ? 0.78 : 0.22,
        roughness: bracketIsCnc ? 0.24 : 0.48,
      });

      // NEMA17 stepper motor blocks & tinted motor-mount brackets at every node with role === "mount"
      const mountNodes = geom.nodes.filter((n) => n.role === "mount" && basePos[n.id]);
      if (mountNodes.length > 0) {
        const motorBodyGeo = new THREE.BoxGeometry(0.42, 0.38, 0.42);
        const motorBodyMat = new THREE.MeshStandardMaterial({
          color: 0x1e2532,
          metalness: 0.72,
          roughness: 0.32,
        });
        const mountPlateGeo = new THREE.BoxGeometry(0.56, 0.085, 0.56);
        const motorCapGeo = new THREE.BoxGeometry(0.43, 0.045, 0.43);
        const motorCapMat = new THREE.MeshStandardMaterial({
          color: 0x94a3b8,
          metalness: 0.82,
          roughness: 0.25,
        });
        const pulleyGeo = new THREE.CylinderGeometry(0.085, 0.085, 0.14, 14);
        const pulleyMat = new THREE.MeshStandardMaterial({
          color: 0xe2e8f0,
          metalness: 0.85,
          roughness: 0.2,
        });
        const isDoubleShear = Boolean(geom.double_shear_pulleys ?? sim.spec.gantry.double_shear_pulleys);
        const bearingCapGeo = isDoubleShear ? new THREE.CylinderGeometry(0.115, 0.115, 0.04, 14) : null;
        const bearingCapMat = isDoubleShear
          ? new THREE.MeshStandardMaterial({
              color: 0x38bdf8,
              metalness: 0.65,
              roughness: 0.28,
            })
          : null;

        for (const n of mountNodes) {
          const p = basePos[n.id];
          const inwardX = Math.abs(p.x) > 0.2 ? -Math.sign(p.x) * 0.16 : 0;
          const inwardZ = Math.abs(p.z) > 0.2 ? -Math.sign(p.z) * 0.16 : 0;

          const motorGroup = new THREE.Group();
          motorGroup.position.set(p.x + inwardX, p.y, p.z + inwardZ);

          const bodyMesh = new THREE.Mesh(motorBodyGeo, motorBodyMat);
          bodyMesh.position.y = -0.19;
          motorGroup.add(bodyMesh);

          // Tinted motor mount bracket plate bridging the stepper & corner extrusion
          const plateMesh = new THREE.Mesh(mountPlateGeo, bracketMat);
          plateMesh.position.set(-inwardX * 0.35, 0.035, -inwardZ * 0.35);
          motorGroup.add(plateMesh);

          const capMesh = new THREE.Mesh(motorCapGeo, motorCapMat);
          capMesh.position.y = 0.02;
          motorGroup.add(capMesh);

          const pulleyMesh = new THREE.Mesh(pulleyGeo, pulleyMat);
          pulleyMesh.position.y = 0.13;
          motorGroup.add(pulleyMesh);

          if (bearingCapGeo && bearingCapMat) {
            const topBearing = new THREE.Mesh(bearingCapGeo, bearingCapMat);
            topBearing.position.y = 0.22;
            motorGroup.add(topBearing);
          }

          group.add(motorGroup);
        }
      }

      // XY-joint carriage bracket blocks (y_car_l / y_car_r) tinted with gantry_brackets.color_hex
      const xyJointNodes = geom.nodes.filter(
        (n) => (n.id === "y_car_l" || n.id === "y_car_r") && basePos[n.id]
      );
      if (xyJointNodes.length > 0) {
        const xyJointGeo = new THREE.BoxGeometry(0.32, 0.24, 0.34);
        for (const n of xyJointNodes) {
          const jointMesh = new THREE.Mesh(xyJointGeo, bracketMat);
          jointMesh.position.copy(basePos[n.id]);
          group.add(jointMesh);
        }
      }

      if (mountNodes.length === 0 && xyJointNodes.length === 0) {
        bracketMat.dispose();
      }

      // Hardened-steel 3D linear rails bolted onto the chosen faces of X-gantry & CoreXY Y-rail beams
      {
        const lr = geom.linear_rails;
        const xRailMount: XRailMountType =
          lr?.x_rail_mount ??
          sim.spec.gantry.x_rail_mount ??
          (sim.spec.gantry.dual_x_rails ? "dual_top_bottom" : "front");
        const yRailMount: YRailMountType =
          lr?.y_rail_mount ?? sim.spec.gantry.y_rail_mount ?? "top";
        const xRailProf = lr?.x_rail_profile ?? sim.spec.gantry.x_rail_profile ?? "mgn12_rail";
        const yRailProf = lr?.y_rail_profile ?? sim.spec.gantry.y_rail_profile ?? "mgn9_rail";
        const xRailW = Math.max(
          (lr?.x_rail_width_mm ?? (xRailProf.includes("mgn9") ? 9.0 : 12.0)) * SCALE,
          0.075
        );
        const xRailH = Math.max(
          (lr?.x_rail_height_mm ?? (xRailProf.includes("mgn9") ? 6.5 : 8.0)) * SCALE,
          0.055
        );
        const yRailW = Math.max(
          (lr?.y_rail_width_mm ?? (yRailProf.includes("mgn9") ? 9.0 : 12.0)) * SCALE,
          0.075
        );
        const yRailH = Math.max(
          (lr?.y_rail_height_mm ?? (yRailProf.includes("mgn9") ? 6.5 : 8.0)) * SCALE,
          0.055
        );

        const railSteelMat = new THREE.MeshStandardMaterial({
          color: 0x8e9aa8,
          metalness: 0.75,
          roughness: 0.28,
        });
        const railEdgeMat = new THREE.LineBasicMaterial({
          color: 0xdbeafe,
          transparent: true,
          opacity: 0.36,
        });
        let railMatUsed = false;

        const addRailBar = (
          geo: THREE.BoxGeometry,
          edgesGeo: THREE.EdgesGeometry,
          pos: THREE.Vector3
        ) => {
          railMatUsed = true;
          const barMesh = new THREE.Mesh(geo, railSteelMat);
          barMesh.position.copy(pos);
          group.add(barMesh);
          const outline = new THREE.LineSegments(edgesGeo, railEdgeMat);
          outline.position.copy(pos);
          group.add(outline);
        };

        // X gantry linear rail(s) on x_gantry_l + x_gantry_r (CoreXY & Cartesian)
        const xGantryL = geom.members.find((m) => m.id === "x_gantry_l");
        const xGantryR = geom.members.find((m) => m.id === "x_gantry_r");
        if (xGantryL && xGantryR && basePos[xGantryL.start_id] && basePos[xGantryR.end_id]) {
          const pL = basePos[xGantryL.start_id];
          const pR = basePos[xGantryR.end_id];
          const railLenX = Math.max(Math.abs(pR.x - pL.x) - 0.14, 0.15);
          const beamCX = 0.5 * (pL.x + pR.x);
          const beamCY = 0.5 * (pL.y + pR.y); // Three.js Y = CAD Z
          const beamCZ = 0.5 * (pL.z + pR.z); // Three.js Z = -CAD Y
          const beamHalfCadZ = 0.5 * Math.max(xGantryL.width_mm * SCALE, 0.06);
          const beamHalfCadY = 0.5 * Math.max(xGantryL.height_mm * SCALE, 0.06);

          if (xRailMount === "front") {
            // CAD -Y face -> Three.js +Z face; wide face in CAD Z (Three.js Y), thickness in CAD -Y (Three.js +Z)
            const xGeo = new THREE.BoxGeometry(railLenX, xRailW, xRailH);
            const xEdges = new THREE.EdgesGeometry(xGeo);
            addRailBar(
              xGeo,
              xEdges,
              new THREE.Vector3(beamCX, beamCY, beamCZ + beamHalfCadY + xRailH * 0.5)
            );
          } else if (xRailMount === "top") {
            // CAD +Z face -> Three.js +Y face; wide face in CAD Y (Three.js Z), thickness in CAD +Z (Three.js +Y)
            const xGeo = new THREE.BoxGeometry(railLenX, xRailH, xRailW);
            const xEdges = new THREE.EdgesGeometry(xGeo);
            addRailBar(
              xGeo,
              xEdges,
              new THREE.Vector3(beamCX, beamCY + beamHalfCadZ + xRailH * 0.5, beamCZ)
            );
          } else if (xRailMount === "bottom") {
            // CAD -Z face -> Three.js -Y face
            const xGeo = new THREE.BoxGeometry(railLenX, xRailH, xRailW);
            const xEdges = new THREE.EdgesGeometry(xGeo);
            addRailBar(
              xGeo,
              xEdges,
              new THREE.Vector3(beamCX, beamCY - beamHalfCadZ - xRailH * 0.5, beamCZ)
            );
          } else if (xRailMount === "dual_top_bottom") {
            // Dual rails on both +Z (+Y Three.js) and -Z (-Y Three.js) faces
            const xGeo = new THREE.BoxGeometry(railLenX, xRailH, xRailW);
            const xEdges = new THREE.EdgesGeometry(xGeo);
            addRailBar(
              xGeo,
              xEdges,
              new THREE.Vector3(beamCX, beamCY + beamHalfCadZ + xRailH * 0.5, beamCZ)
            );
            addRailBar(
              xGeo,
              xEdges,
              new THREE.Vector3(beamCX, beamCY - beamHalfCadZ - xRailH * 0.5, beamCZ)
            );
          }
        }

        // CoreXY Y-axis linear rails on left (x=0) and right (x=W) Y-rail extrusions
        if (geom.architecture === "corexy") {
          const yRailLF = geom.members.find((m) => m.id === "y_rail_l_f");
          const yRailRF = geom.members.find((m) => m.id === "y_rail_r_f");
          const pLF = basePos["g_fl"];
          const pLR = basePos["g_rl"];
          const pRF = basePos["g_fr"];
          const pRR = basePos["g_rr"];
          if (yRailLF && yRailRF && pLF && pLR && pRF && pRR) {
            const railLenY = Math.max(Math.abs(pLR.z - pLF.z) - 0.16, 0.15);
            const beamHalfCadX = 0.5 * Math.max(yRailLF.width_mm * SCALE, 0.06);
            const beamHalfCadZ = 0.5 * Math.max(yRailLF.height_mm * SCALE, 0.06);
            const leftCenter = new THREE.Vector3(
              pLF.x,
              0.5 * (pLF.y + pLR.y),
              0.5 * (pLF.z + pLR.z)
            );
            const rightCenter = new THREE.Vector3(
              pRF.x,
              0.5 * (pRF.y + pRR.y),
              0.5 * (pRF.z + pRR.z)
            );

            if (yRailMount === "top") {
              const yGeo = new THREE.BoxGeometry(yRailW, yRailH, railLenY);
              const yEdges = new THREE.EdgesGeometry(yGeo);
              addRailBar(
                yGeo,
                yEdges,
                leftCenter.clone().add(new THREE.Vector3(0, beamHalfCadZ + yRailH * 0.5, 0))
              );
              addRailBar(
                yGeo,
                yEdges,
                rightCenter.clone().add(new THREE.Vector3(0, beamHalfCadZ + yRailH * 0.5, 0))
              );
            } else if (yRailMount === "bottom") {
              const yGeo = new THREE.BoxGeometry(yRailW, yRailH, railLenY);
              const yEdges = new THREE.EdgesGeometry(yGeo);
              addRailBar(
                yGeo,
                yEdges,
                leftCenter.clone().add(new THREE.Vector3(0, -(beamHalfCadZ + yRailH * 0.5), 0))
              );
              addRailBar(
                yGeo,
                yEdges,
                rightCenter.clone().add(new THREE.Vector3(0, -(beamHalfCadZ + yRailH * 0.5), 0))
              );
            } else if (yRailMount === "inner") {
              // +X (inner) face of left Y extrusion and -X (inner) face of right Y extrusion
              const yGeo = new THREE.BoxGeometry(yRailH, yRailW, railLenY);
              const yEdges = new THREE.EdgesGeometry(yGeo);
              addRailBar(
                yGeo,
                yEdges,
                leftCenter.clone().add(new THREE.Vector3(beamHalfCadX + yRailH * 0.5, 0, 0))
              );
              addRailBar(
                yGeo,
                yEdges,
                rightCenter.clone().add(new THREE.Vector3(-(beamHalfCadX + yRailH * 0.5), 0, 0))
              );
            }
          }
        }

        if (!railMatUsed) {
          railSteelMat.dispose();
          railEdgeMat.dispose();
        }
      }

      // Translucent / semi-opaque 3D structural panels mounted flush on outer extrusion faces (CoreXY Left, Right, Rear)
      if (geom.architecture === "corexy" && geom.panels?.enabled) {
        const panels = geom.panels;
        const [fw, fd, fh] = geom.frame_dims_mm;
        const frameMember = geom.members.find((m) => m.role === "frame");
        const extrusionHalfMm = (frameMember?.width_mm ?? 20) * 0.5;
        const extHalf = extrusionHalfMm * SCALE;
        const tScene = Math.max((panels.thickness_mm || 3) * SCALE, 0.024);

        // Full outer-perimeter panel dimensions covering the extrusion T-slot faces
        const spanX = (fw + 2 * extrusionHalfMm) * SCALE;
        const spanY = (fd + 2 * extrusionHalfMm) * SCALE;
        const spanZ = (fh + 2 * extrusionHalfMm) * SCALE;

        const isTransparentSheet =
          panels.material_id === "acrylic_pmma" || panels.material_id === "polycarbonate";
        const isBoltedStructural = panels.mounting === "bolted_structural";

        const panelColor = new THREE.Color(
          panels.color_hex || (isTransparentSheet ? "#93c5fd" : "#475569")
        );
        const panelMat = new THREE.MeshStandardMaterial({
          color: panelColor,
          transparent: true,
          opacity: isTransparentSheet ? 0.16 : 0.32,
          depthWrite: false,
          roughness: isTransparentSheet ? 0.14 : 0.38,
          metalness: isTransparentSheet ? 0.1 : 0.42,
          side: THREE.DoubleSide,
        });

        const edgeMat = new THREE.LineBasicMaterial({
          color: isBoltedStructural ? 0x38bdf8 : 0x94a3b8,
          transparent: true,
          opacity: isBoltedStructural ? 0.65 : 0.26,
        });

        const sidePanelGeo = new THREE.BoxGeometry(tScene, spanZ, spanY);
        const rearPanelGeo = new THREE.BoxGeometry(spanX, spanZ, tScene);
        const sideEdgesGeo = new THREE.EdgesGeometry(sidePanelGeo);
        const rearEdgesGeo = new THREE.EdgesGeometry(rearPanelGeo);

        // Mount flush on the OUTSIDE face of Left (-X), Right (+X), and Rear (+Y CAD = -Z Three.js) extrusions
        const leftCenter = toThree([0, fd * 0.5, fh * 0.5], cx, cy);
        const rightCenter = toThree([fw, fd * 0.5, fh * 0.5], cx, cy);
        const rearCenter = toThree([fw * 0.5, fd, fh * 0.5], cx, cy);

        const leftPos = leftCenter.clone().add(new THREE.Vector3(-(extHalf + tScene * 0.5), 0, 0));
        const rightPos = rightCenter.clone().add(new THREE.Vector3(extHalf + tScene * 0.5, 0, 0));
        const rearPos = rearCenter.clone().add(new THREE.Vector3(0, 0, -(extHalf + tScene * 0.5)));

        const leftBoltX = leftPos.x - tScene * 0.55;
        const rightBoltX = rightPos.x + tScene * 0.55;
        const rearBoltZ = rearPos.z - tScene * 0.55;

        const panelSpecs: Array<{
          bay: string;
          label: string;
          geo: THREE.BoxGeometry;
          edges: THREE.EdgesGeometry;
          pos: THREE.Vector3;
        }> = [
          { bay: "left", label: "Left Enclosure Panel", geo: sidePanelGeo, edges: sideEdgesGeo, pos: leftPos },
          { bay: "right", label: "Right Enclosure Panel", geo: sidePanelGeo, edges: sideEdgesGeo, pos: rightPos },
          { bay: "rear", label: "Rear Enclosure Panel", geo: rearPanelGeo, edges: rearEdgesGeo, pos: rearPos },
        ];

        for (const ps of panelSpecs) {
          const wallMat = panelMat.clone();
          const mesh = new THREE.Mesh(ps.geo, wallMat);
          mesh.position.copy(ps.pos);
          mesh.renderOrder = 2;
          const bayGrp = classifyPartGroup(`bay_${ps.bay}`, "panel");
          const baySens = sim.sensitivity?.panel_bay_sensitivities?.[ps.bay];
          const bayStiff = stiffenById[bayGrp.id] ?? stiffenById[`bay_${ps.bay}`];
          tagInspectable(mesh, {
            key: `panel:${ps.bay}`,
            name: ps.label,
            subsystem: "frame",
            partGroupId: bayGrp.id,
            role: "panel",
            material: panels.material_name,
            profile: `${panels.thickness_mm}mm (${panels.mounting.replace(/_/g, " ")})`,
            complianceSharePct: baySens?.compliance_share_pct ?? bayStiff?.compliance_share_pct,
            heatScore: baySens ? getLensScore(baySens, adjointLens) : undefined,
            estHzGain: bayStiff?.est_hz_gain,
            hzPer100g: baySens?.hz_per_100g,
            whyItHurts: bayStiff?.why_it_hurts,
            actionHint: bayStiff?.action_hint,
          });
          group.add(mesh);

          const outline = new THREE.LineSegments(ps.edges, edgeMat);
          outline.position.copy(ps.pos);
          group.add(outline);
        }
        panelMat.dispose();

        // Metallic perimeter bolt heads aligned directly on the extrusion T-slot centerlines
        if (isBoltedStructural) {
          const boltCoords: THREE.Vector3[] = [];
          const pBot = toThree([0, 0, 0], cx, cy);
          const pTop = toThree([fw, fd, fh], cx, cy);
          const yBot = pBot.y;
          const yTop = pTop.y;
          const xMin = pBot.x;
          const xMax = pTop.x;
          const zFront = pBot.z; // CAD y = 0 (front columns)
          const zRear = pTop.z;  // CAD y = fd (rear columns)
          const steps = 5;

          const addSideBolts = (xFace: number) => {
            for (let i = 0; i <= steps; i++) {
              const t = i / steps;
              const zAlong = zFront + t * (zRear - zFront);
              const yAlong = yBot + t * (yTop - yBot);
              // Bottom & top horizontal extrusion T-slot centerlines
              boltCoords.push(new THREE.Vector3(xFace, yBot, zAlong));
              boltCoords.push(new THREE.Vector3(xFace, yTop, zAlong));
              // Front & rear vertical column T-slot centerlines (skip corners)
              if (i > 0 && i < steps) {
                boltCoords.push(new THREE.Vector3(xFace, yAlong, zFront));
                boltCoords.push(new THREE.Vector3(xFace, yAlong, zRear));
              }
            }
          };

          const addRearBolts = (zFace: number) => {
            for (let i = 0; i <= steps; i++) {
              const t = i / steps;
              const xAlong = xMin + t * (xMax - xMin);
              const yAlong = yBot + t * (yTop - yBot);
              // Base rear & top rear horizontal extrusion centerlines
              boltCoords.push(new THREE.Vector3(xAlong, yBot, zFace));
              boltCoords.push(new THREE.Vector3(xAlong, yTop, zFace));
              // Left & right rear vertical column centerlines (skip corners)
              if (i > 0 && i < steps) {
                boltCoords.push(new THREE.Vector3(xMin, yAlong, zFace));
                boltCoords.push(new THREE.Vector3(xMax, yAlong, zFace));
              }
            }
          };

          addSideBolts(leftBoltX);
          addSideBolts(rightBoltX);
          addRearBolts(rearBoltZ);

          const boltGeo = new THREE.SphereGeometry(0.036, 8, 8);
          const boltMat = new THREE.MeshStandardMaterial({
            color: 0xe2e8f0,
            metalness: 0.85,
            roughness: 0.25,
          });
          const boltInst = new THREE.InstancedMesh(boltGeo, boltMat, boltCoords.length);
          const dummy = new THREE.Object3D();
          for (let i = 0; i < boltCoords.length; i++) {
            dummy.position.copy(boltCoords[i]);
            dummy.updateMatrix();
            boltInst.setMatrixAt(i, dummy.matrix);
          }
          boltInst.instanceMatrix.needsUpdate = true;
          group.add(boltInst);
        }

        // Bright cyan gantry-line bolt heads aligned directly on the fixed Y-rail & rear gantry extrusion T-slots
        if (panels.bolted_to_gantry) {
          const gantryNode =
            geom.nodes.find((n) => n.id === "g_rl") ??
            geom.nodes.find((n) => n.id === "g_rr") ??
            geom.nodes.find((n) => n.role === "gantry");
          if (gantryNode) {
            const gantryY = toThree(gantryNode.position_mm, cx, cy).y;
            const pBot = toThree([0, 0, 0], cx, cy);
            const pTop = toThree([fw, fd, fh], cx, cy);
            const xMin = pBot.x;
            const xMax = pTop.x;
            const zFront = pBot.z;
            const zRear = pTop.z;
            const gantrySteps = 5;
            const gantryBoltCoords: THREE.Vector3[] = [];

            for (let i = 0; i <= gantrySteps; i++) {
              const t = i / gantrySteps;
              const zAlong = zFront + t * (zRear - zFront);
              const xAlong = xMin + t * (xMax - xMin);
              // Left & Right side panels bolted along the fixed Y-rail extrusions
              gantryBoltCoords.push(new THREE.Vector3(leftBoltX, gantryY, zAlong));
              gantryBoltCoords.push(new THREE.Vector3(rightBoltX, gantryY, zAlong));
              // Rear panel bolted along the fixed rear gantry motor extrusion
              gantryBoltCoords.push(new THREE.Vector3(xAlong, gantryY, rearBoltZ));
            }

            const gantryBoltRadius = Math.max(0.044, tScene * 0.65);
            const gantryBoltGeo = new THREE.SphereGeometry(gantryBoltRadius, 10, 10);
            const gantryBoltMat = new THREE.MeshStandardMaterial({
              color: 0x38bdf8,
              emissive: 0x0284c7,
              emissiveIntensity: 0.45,
              metalness: 0.78,
              roughness: 0.2,
            });
            const gantryBoltInst = new THREE.InstancedMesh(
              gantryBoltGeo,
              gantryBoltMat,
              gantryBoltCoords.length
            );
            const dummy = new THREE.Object3D();
            for (let i = 0; i < gantryBoltCoords.length; i++) {
              dummy.position.copy(gantryBoltCoords[i]);
              dummy.updateMatrix();
              gantryBoltInst.setMatrixAt(i, dummy.matrix);
            }
            gantryBoltInst.instanceMatrix.needsUpdate = true;
            group.add(gantryBoltInst);
          }
        }
      }
    }

    // Heated bed plate (deforms with bed_surface in Frame FEA, animates in Modal mode, & tints in Sensitivity)
    const [bedX, bedY] = geom.bed_size_mm;
    const bedThickScene = Math.max(Math.min((sim.spec.bed.plate_thickness_mm ?? 8) * SCALE, 0.14), 0.05);
    const bedGeo = new THREE.BoxGeometry(bedX * SCALE, bedThickScene, bedY * SCALE);

    let bedHeatScore = 0;
    if (viewMode === "sensitivity") {
      const bedSpringScore = getLensScore(memberSensMap["bed_mount_spring"], adjointLens);
      bedHeatScore = bedSpringScore;
      for (const [mid, entry] of Object.entries(memberSensMap)) {
        if (mid.startsWith("bed_arm_") || mid.startsWith("bed_")) {
          const s = getLensScore(entry, adjointLens);
          if (Math.abs(s) > Math.abs(bedHeatScore)) bedHeatScore = s;
        }
      }
    }

    const bedColor = new THREE.Color(0x2a3444);
    const bedEmissive = new THREE.Color(0x000000);
    let bedEmissiveIntensity = 0;
    if (viewMode === "sensitivity") {
      sensitivityColorInto(bedHeatScore, bedColor);
      const absBed = Math.min(1, Math.abs(bedHeatScore));
      if (absBed > 0.15) {
        bedEmissive.copy(bedColor);
        bedEmissiveIntensity = absBed * 0.32;
      }
    }

    const bedPlateFea = sim.bed_plate_fea ?? null;
    const isBedWarpRenderMode =
      viewMode === "assembly" ||
      viewMode === "fea_stress" ||
      viewMode === "modal" ||
      viewMode === "sensitivity";
    const activeBedWarp = Boolean(showBedWarp && bedPlateFea && isBedWarpRenderMode);

    let bedCompShare = 0;
    let bedSensHasData = false;
    const bedSpringEntry = memberSensMap["bed_mount_spring"];
    if (bedSpringEntry) {
      bedSensHasData = true;
      bedCompShare += bedSpringEntry.compliance_share_pct ?? 0;
    }
    for (const [mid, entry] of Object.entries(memberSensMap)) {
      if (mid !== "bed_mount_spring" && (mid.startsWith("bed_arm_") || mid.startsWith("bed_"))) {
        bedSensHasData = true;
        bedCompShare += entry.compliance_share_pct ?? 0;
      }
    }
    const bedStiffTarget = stiffenById["bed_support"] ?? stiffenById["bed_plate"];
    const bedInspectInfo: InspectedPartInfo = {
      key: "bed_plate",
      name: "Heated Bed Plate",
      subsystem: "bed",
      partGroupId: "bed_plate",
      role: "bed",
      material: sim.spec.bed.plate_material,
      profile: `${sim.spec.bed.plate_thickness_mm}mm plate · ${(
        sim.spec.bed.mount_type ?? "kinematic_3pt"
      ).replace(/_/g, " ")}`,
      complianceSharePct:
        bedStiffTarget?.compliance_share_pct ?? (bedSensHasData ? bedCompShare : undefined),
      heatScore: bedSensHasData ? bedHeatScore : undefined,
      estHzGain: bedStiffTarget?.est_hz_gain,
      whyItHurts: bedStiffTarget?.why_it_hurts ?? bedPlateFea?.summary,
      actionHint:
        bedStiffTarget?.action_hint ??
        (bedPlateFea &&
        (bedPlateFea.first_layer_risk === "high" || bedPlateFea.first_layer_risk === "critical")
          ? "Bed mount & Z-axis support stiffness / thermal constraint"
          : undefined),
      bedFlatnessUm: bedPlateFea?.total_flatness_error_um,
      bedTacoUm: bedPlateFea?.taco_buckling_um,
      bedModeHz: bedPlateFea?.plate_first_mode_hz,
      bedRisk: bedPlateFea?.first_layer_risk,
    };

    const bedMat = new THREE.MeshStandardMaterial({
      color: bedColor,
      emissive: bedEmissive,
      emissiveIntensity: bedEmissiveIntensity,
      metalness: viewMode === "sensitivity" ? 0.25 : 0.4,
      roughness: 0.45,
      transparent:
        activeBedWarp ||
        viewMode === "fea_stress" ||
        viewMode === "modal" ||
        viewMode === "sensitivity",
      opacity: activeBedWarp
        ? 0.45
        : viewMode === "sensitivity"
          ? 0.52
          : viewMode === "fea_stress" || viewMode === "modal"
            ? 0.38
            : 0.95,
    });
    const bedMesh = new THREE.Mesh(bedGeo, bedMat);
    const bedBaseCenter = toThree(geom.bed_center_mm, cx, cy);
    bedBaseCenter.y -= bedThickScene * 0.5;
    const bedNodeId = geom.bed_node_id || "bed_surface";
    const bedNodeBase = basePos[bedNodeId];
    const bedVisualOffset = bedNodeBase
      ? bedBaseCenter.clone().sub(bedNodeBase)
      : new THREE.Vector3(0, -bedThickScene * 0.5, 0);

    if (viewMode === "fea_stress" && deformedPos[bedNodeId]) {
      bedMesh.position.copy(deformedPos[bedNodeId]).add(bedVisualOffset);
    } else {
      bedMesh.position.copy(bedBaseCenter);
    }
    if (viewMode === "modal" && bedNodeBase) {
      bedMesh.frustumCulled = false;
      stateRef.current.modalBedMesh = bedMesh;
      stateRef.current.modalBedNodeId = bedNodeId;
      stateRef.current.modalBedOffset.copy(bedVisualOffset);
    }
    tagInspectable(bedMesh, bedInspectInfo);
    group.add(bedMesh);

    // Real 2D MITC4 Heated Bed Plate Surface Mesh & 3D Thermal Taco Warp Overlay
    if (bedPlateFea && isBedWarpRenderMode) {
      const nx = Math.max(2, Math.floor(bedPlateFea.grid_nx || 0));
      const ny = Math.max(2, Math.floor(bedPlateFea.grid_ny || 0));
      const totalGridVerts = nx * ny;
      const gx =
        bedPlateFea.grid_x_rel && bedPlateFea.grid_x_rel.length >= nx
          ? bedPlateFea.grid_x_rel
          : Array.from({ length: nx }, (_, i) => i / Math.max(nx - 1, 1) - 0.5);
      const gy =
        bedPlateFea.grid_y_rel && bedPlateFea.grid_y_rel.length >= ny
          ? bedPlateFea.grid_y_rel
          : Array.from({ length: ny }, (_, i) => i / Math.max(ny - 1, 1) - 0.5);
      const zVals =
        bedPlateFea.z_total_um && bedPlateFea.z_total_um.length >= totalGridVerts
          ? bedPlateFea.z_total_um
          : new Array<number>(totalGridVerts).fill(0);

      let gxMin = Infinity,
        gxMax = -Infinity;
      for (let i = 0; i < nx; i++) {
        if (gx[i] < gxMin) gxMin = gx[i];
        if (gx[i] > gxMax) gxMax = gx[i];
      }
      const gxMid = 0.5 * (gxMin + gxMax);
      const gxSpan = Math.max(gxMax - gxMin, 1e-6);

      let gyMin = Infinity,
        gyMax = -Infinity;
      for (let j = 0; j < ny; j++) {
        if (gy[j] < gyMin) gyMin = gy[j];
        if (gy[j] > gyMax) gyMax = gy[j];
      }
      const gyMid = 0.5 * (gyMin + gyMax);
      const gySpan = Math.max(gyMax - gyMin, 1e-6);

      let zMin = Infinity,
        zMax = -Infinity,
        zSum = 0;
      for (let k = 0; k < totalGridVerts; k++) {
        const zv = Number.isFinite(zVals[k]) ? zVals[k] : 0;
        if (zv < zMin) zMin = zv;
        if (zv > zMax) zMax = zv;
        zSum += zv;
      }
      const zMean = zSum / Math.max(totalGridVerts, 1);
      const zSpan = Math.max(zMax - zMin, 1e-6);
      const rawHalfSpan = Math.max(Math.abs(zMax - zMean), Math.abs(zMin - zMean), 1);
      // 100 µm -> ~0.18 scene units, smoothly clamped at 0.42 units so 400 µm stays beneath gantry
      const clampedAmp = Math.min(0.42, Math.max(zSpan > 1 ? 0.035 : 0, (rawHalfSpan / 100) * 0.18));
      const umScale = clampedAmp / rawHalfSpan;
      const baseTopY = bedThickScene * 0.5 + 0.007;
      const warpLiftY = activeBedWarp ? Math.max(0.016, (zMean - zMin) * umScale + 0.014) : 0;

      const positions = new Float32Array(totalGridVerts * 3);
      const colors = new Float32Array(totalGridVerts * 3);
      const warpCol = new THREE.Color();

      for (let iy = 0; iy < ny; iy++) {
        const vCentered = (gy[iy] - gyMid) / gySpan; // [-0.5, +0.5] CAD Y
        const localZ = -vCentered * bedY * SCALE; // CAD +Y -> Three.js -Z
        for (let ix = 0; ix < nx; ix++) {
          const uCentered = (gx[ix] - gxMid) / gxSpan; // [-0.5, +0.5] CAD X
          const localX = uCentered * bedX * SCALE;
          const idx = iy * nx + ix;
          const zv = Number.isFinite(zVals[idx]) ? zVals[idx] : 0;
          const localY = activeBedWarp ? baseTopY + warpLiftY + (zv - zMean) * umScale : baseTopY;

          const o = idx * 3;
          positions[o] = localX;
          positions[o + 1] = localY;
          positions[o + 2] = localZ;

          if (activeBedWarp) {
            const tNorm = zSpan < 3.5 ? 0.42 : (zv - zMin) / zSpan;
            bedWarpColorInto(tNorm, warpCol);
            colors[o] = warpCol.r;
            colors[o + 1] = warpCol.g;
            colors[o + 2] = warpCol.b;
          } else {
            colors[o] = 0.22;
            colors[o + 1] = 0.28;
            colors[o + 2] = 0.36;
          }
        }
      }

      const quadCount = (nx - 1) * (ny - 1);
      const triIndices = new Uint16Array(quadCount * 6);
      const edgeIndices: number[] = [];
      let tIdx = 0;
      for (let iy = 0; iy < ny - 1; iy++) {
        for (let ix = 0; ix < nx - 1; ix++) {
          const a = iy * nx + ix;
          const b = a + 1;
          const c = (iy + 1) * nx + ix + 1;
          const d = (iy + 1) * nx + ix;
          triIndices[tIdx++] = a;
          triIndices[tIdx++] = b;
          triIndices[tIdx++] = c;
          triIndices[tIdx++] = a;
          triIndices[tIdx++] = c;
          triIndices[tIdx++] = d;
        }
      }
      for (let iy = 0; iy < ny; iy++) {
        for (let ix = 0; ix < nx - 1; ix++) {
          edgeIndices.push(iy * nx + ix, iy * nx + ix + 1);
        }
      }
      for (let iy = 0; iy < ny - 1; iy++) {
        for (let ix = 0; ix < nx; ix++) {
          edgeIndices.push(iy * nx + ix, (iy + 1) * nx + ix);
        }
      }

      const bedSurfGeo = new THREE.BufferGeometry();
      bedSurfGeo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      bedSurfGeo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
      bedSurfGeo.setIndex(new THREE.BufferAttribute(triIndices, 1));
      bedSurfGeo.computeVertexNormals();

      const bedSurfMat = new THREE.MeshStandardMaterial({
        vertexColors: true,
        roughness: activeBedWarp ? 0.32 : 0.55,
        metalness: activeBedWarp ? 0.14 : 0.35,
        side: THREE.DoubleSide,
        transparent: !activeBedWarp,
        opacity: activeBedWarp ? 0.94 : 0.28,
        polygonOffset: true,
        polygonOffsetFactor: -1,
        polygonOffsetUnits: -1,
      });
      const bedSurfMesh = new THREE.Mesh(bedSurfGeo, bedSurfMat);
      bedSurfMesh.renderOrder = activeBedWarp ? 3 : 1;
      tagInspectable(bedSurfMesh, bedInspectInfo);
      bedMesh.add(bedSurfMesh);

      const bedWireGeo = new THREE.BufferGeometry();
      bedWireGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(positions), 3));
      bedWireGeo.setIndex(new THREE.BufferAttribute(new Uint16Array(edgeIndices), 1));
      const bedWireMat = new THREE.LineBasicMaterial({
        color: activeBedWarp ? 0x0f172a : 0x64748b,
        transparent: true,
        opacity: activeBedWarp ? 0.65 : 0.25,
      });
      const bedWireLines = new THREE.LineSegments(bedWireGeo, bedWireMat);
      bedWireLines.renderOrder = activeBedWarp ? 4 : 2;
      bedMesh.add(bedWireLines);

      // Highlight MITC4 bed plate mount constraint pins on warped surface
      if (activeBedWarp && bedPlateFea.mount_points_rel && bedPlateFea.mount_points_rel.length > 0) {
        const pinGeo = new THREE.RingGeometry(0.05, 0.085, 16);
        const pinMat = new THREE.MeshBasicMaterial({
          color: 0xf8fafc,
          side: THREE.DoubleSide,
        });
        for (const [mx, my] of bedPlateFea.mount_points_rel) {
          const uM = (mx - gxMid) / gxSpan;
          const vM = (my - gyMid) / gySpan;
          const pin = new THREE.Mesh(pinGeo, pinMat);
          pin.rotation.x = -Math.PI * 0.5;
          pin.position.set(uM * bedX * SCALE, baseTopY + warpLiftY + 0.015, -vM * bedY * SCALE);
          pin.renderOrder = 5;
          bedMesh.add(pin);
        }
      }
    }

    // 3D Bed Mount Standoffs / Kinematic Pucks / Springs / Silicone Columns at support points
    if (viewMode === "assembly" || viewMode === "sensitivity" || viewMode === "fea_stress") {
      const bedMountType: BedMountType =
        geom.bed_mount_type ?? sim.spec.bed.mount_type ?? "kinematic_3pt";
      const supportPtsThree: THREE.Vector3[] = [];

      if (geom.bed_support_points_mm && geom.bed_support_points_mm.length > 0) {
        for (const ptMm of geom.bed_support_points_mm) {
          supportPtsThree.push(toThree(ptMm, cx, cy));
        }
      } else {
        const [bcx, bcy, bcz] = geom.bed_center_mm;
        const insetX = bedX * 0.36;
        const insetY = bedY * 0.36;
        const mountZ = bcz - (sim.spec.bed.plate_thickness_mm ?? 8) - 8;
        if (bedMountType === "kinematic_3pt") {
          supportPtsThree.push(
            toThree([bcx - insetX, bcy - insetY, mountZ], cx, cy),
            toThree([bcx + insetX, bcy - insetY, mountZ], cx, cy),
            toThree([bcx, bcy + insetY, mountZ], cx, cy)
          );
        } else {
          supportPtsThree.push(
            toThree([bcx - insetX, bcy - insetY, mountZ], cx, cy),
            toThree([bcx + insetX, bcy - insetY, mountZ], cx, cy),
            toThree([bcx - insetX, bcy + insetY, mountZ], cx, cy),
            toThree([bcx + insetX, bcy + insetY, mountZ], cx, cy)
          );
        }
      }

      if (supportPtsThree.length > 0) {
        const bedDispDelta =
          viewMode === "fea_stress" && deformedPos[bedNodeId] && bedNodeBase
            ? deformedPos[bedNodeId].clone().sub(bedNodeBase)
            : new THREE.Vector3(0, 0, 0);
        const bedBottomY = bedBaseCenter.y - bedThickScene * 0.5;
        const useSensBedTint = viewMode === "sensitivity" && Math.abs(bedHeatScore) > 0.15;

        if (bedMountType === "kinematic_3pt") {
          const puckGeo = new THREE.CylinderGeometry(0.11, 0.125, 0.045, 14);
          const puckMat = new THREE.MeshStandardMaterial({
            color: useSensBedTint ? bedColor : 0xcbd5e1,
            emissive: useSensBedTint ? bedEmissive : 0x000000,
            emissiveIntensity: useSensBedTint ? bedEmissiveIntensity : 0,
            metalness: 0.78,
            roughness: 0.28,
          });
          const ballGeo = new THREE.SphereGeometry(0.068, 12, 12);
          const ballMat = new THREE.MeshStandardMaterial({
            color: useSensBedTint ? bedColor : 0xeab308,
            metalness: 0.85,
            roughness: 0.2,
          });
          for (const sp of supportPtsThree) {
            const grp = new THREE.Group();
            const topPuck = new THREE.Mesh(puckGeo, puckMat);
            topPuck.position.y = -0.025;
            grp.add(topPuck);
            const ball = new THREE.Mesh(ballGeo, ballMat);
            ball.position.y = -0.085;
            grp.add(ball);
            const botPuck = new THREE.Mesh(puckGeo, puckMat);
            botPuck.position.y = -0.145;
            grp.add(botPuck);
            grp.position.set(sp.x + bedDispDelta.x, bedBottomY + bedDispDelta.y, sp.z + bedDispDelta.z);
            group.add(grp);
          }
        } else if (bedMountType === "springs_4pt") {
          const coilGeo = new THREE.TorusGeometry(0.068, 0.017, 8, 16);
          const coilMat = new THREE.MeshStandardMaterial({
            color: useSensBedTint ? bedColor : 0xfacc15,
            emissive: useSensBedTint ? bedEmissive : 0x000000,
            emissiveIntensity: useSensBedTint ? bedEmissiveIntensity : 0,
            metalness: 0.45,
            roughness: 0.35,
          });
          const washerGeo = new THREE.CylinderGeometry(0.085, 0.085, 0.02, 12);
          const washerMat = new THREE.MeshStandardMaterial({
            color: 0x334155,
            metalness: 0.7,
            roughness: 0.4,
          });
          for (const sp of supportPtsThree) {
            const grp = new THREE.Group();
            const topW = new THREE.Mesh(washerGeo, washerMat);
            topW.position.y = -0.012;
            grp.add(topW);
            for (let k = 0; k < 4; k++) {
              const coil = new THREE.Mesh(coilGeo, coilMat);
              coil.rotation.x = Math.PI * 0.5;
              coil.position.y = -0.042 - k * 0.038;
              grp.add(coil);
            }
            const botW = new THREE.Mesh(washerGeo, washerMat);
            botW.position.y = -0.19;
            grp.add(botW);
            grp.position.set(sp.x + bedDispDelta.x, bedBottomY + bedDispDelta.y, sp.z + bedDispDelta.z);
            group.add(grp);
          }
        } else if (bedMountType === "silicone_4pt") {
          const silGeo = new THREE.CylinderGeometry(0.082, 0.082, 0.16, 14);
          const silMat = new THREE.MeshStandardMaterial({
            color: useSensBedTint ? bedColor : 0xf97316,
            emissive: useSensBedTint ? bedEmissive : 0x000000,
            emissiveIntensity: useSensBedTint ? bedEmissiveIntensity : 0,
            roughness: 0.75,
            metalness: 0.08,
          });
          const capGeo = new THREE.CylinderGeometry(0.09, 0.09, 0.022, 14);
          const capMat = new THREE.MeshStandardMaterial({
            color: 0x94a3b8,
            metalness: 0.72,
            roughness: 0.32,
          });
          for (const sp of supportPtsThree) {
            const grp = new THREE.Group();
            const col = new THREE.Mesh(silGeo, silMat);
            col.position.y = -0.09;
            grp.add(col);
            const topC = new THREE.Mesh(capGeo, capMat);
            topC.position.y = -0.012;
            grp.add(topC);
            const botC = new THREE.Mesh(capGeo, capMat);
            botC.position.y = -0.17;
            grp.add(botC);
            grp.position.set(sp.x + bedDispDelta.x, bedBottomY + bedDispDelta.y, sp.z + bedDispDelta.z);
            group.add(grp);
          }
        } else if (bedMountType === "floating_4pt") {
          const tabGeo = new THREE.BoxGeometry(0.22, 0.032, 0.11);
          const tabMat = new THREE.MeshStandardMaterial({
            color: useSensBedTint ? bedColor : 0x94a3b8,
            emissive: useSensBedTint ? bedEmissive : 0x000000,
            emissiveIntensity: useSensBedTint ? bedEmissiveIntensity : 0,
            metalness: 0.75,
            roughness: 0.3,
          });
          const pinGeo = new THREE.CylinderGeometry(0.052, 0.052, 0.14, 12);
          const fixedPinMat = new THREE.MeshStandardMaterial({
            color: 0xe2e8f0,
            metalness: 0.85,
            roughness: 0.22,
          });
          const floatPinMat =
            supportPtsThree.length > 1
              ? new THREE.MeshStandardMaterial({
                  color: 0x38bdf8,
                  metalness: 0.65,
                  roughness: 0.28,
                })
              : fixedPinMat;
          supportPtsThree.forEach((sp, idx) => {
            const grp = new THREE.Group();
            const tab = new THREE.Mesh(tabGeo, tabMat);
            tab.position.y = -0.025;
            grp.add(tab);
            const pin = new THREE.Mesh(pinGeo, idx === 0 ? fixedPinMat : floatPinMat);
            pin.position.y = -0.095;
            grp.add(pin);
            grp.position.set(sp.x + bedDispDelta.x, bedBottomY + bedDispDelta.y, sp.z + bedDispDelta.z);
            group.add(grp);
          });
        } else {
          // rigid_4pt
          const standGeo = new THREE.CylinderGeometry(0.072, 0.072, 0.16, 12);
          const standMat = new THREE.MeshStandardMaterial({
            color: useSensBedTint ? bedColor : 0xe2e8f0,
            emissive: useSensBedTint ? bedEmissive : 0x000000,
            emissiveIntensity: useSensBedTint ? bedEmissiveIntensity : 0,
            metalness: 0.86,
            roughness: 0.2,
          });
          const flangeGeo = new THREE.CylinderGeometry(0.105, 0.105, 0.028, 6);
          for (const sp of supportPtsThree) {
            const grp = new THREE.Group();
            const cyl = new THREE.Mesh(standGeo, standMat);
            cyl.position.y = -0.085;
            grp.add(cyl);
            const flg = new THREE.Mesh(flangeGeo, standMat);
            flg.position.y = -0.018;
            grp.add(flg);
            grp.position.set(sp.x + bedDispDelta.x, bedBottomY + bedDispDelta.y, sp.z + bedDispDelta.z);
            group.add(grp);
          }
        }
      }
    }

    // Subtle undeformed bed plate ghost reference in FEA & Modal modes to visualize bed sway/motion
    if ((viewMode === "fea_stress" || viewMode === "modal") && bedNodeBase) {
      const bedGhostEdges = new THREE.EdgesGeometry(bedGeo);
      const bedGhostMat = new THREE.LineBasicMaterial({
        color: 0x475569,
        transparent: true,
        opacity: 0.35,
      });
      const bedGhost = new THREE.LineSegments(bedGhostEdges, bedGhostMat);
      bedGhost.position.copy(bedBaseCenter);
      group.add(bedGhost);
    }

    // Build volume wireframe
    const [bx, by, bz] = geom.build_volume_mm;
    const bvBox = new THREE.BoxGeometry(bx * SCALE, bz * SCALE, by * SCALE);
    const bvEdges = new THREE.EdgesGeometry(bvBox);
    bvBox.dispose();
    const bvLine = new THREE.LineSegments(
      bvEdges,
      new THREE.LineBasicMaterial({
        color: sim.collision.has_collision ? 0xef4444 : 0x38bdf8,
        transparent: true,
        opacity: viewMode === "toolpath" ? 0.45 : 0.25,
      })
    );
    const orig = geom.build_origin_mm;
    bvLine.position.copy(toThree([orig[0] + bx * 0.5, orig[1] + by * 0.5, orig[2] + bz * 0.5], cx, cy));
    group.add(bvLine);

    // Toolhead carriage, CoG marker sphere, and Nozzle-tip lever-arm assembly
    const thNodeId = geom.toolhead_node_id || "toolhead";
    const cogNodeId = geom.toolhead_cog_node_id || "toolhead_cog";
    const nozNodeId = geom.nozzle_node_id || "nozzle_tip";

    // Ensure CoG and Nozzle Tip positions exist even on legacy/fallback specs
    const thBase = basePos[thNodeId];
    if (thBase) {
      const cogOff = geom.cog_offset_mm ?? sim.spec.toolhead.cog_offset_mm ?? [0, -18, -25];
      const nozOff = geom.nozzle_offset_mm ?? sim.spec.toolhead.nozzle_offset_mm ?? [0, -22, -62];
      const cogDelta = new THREE.Vector3(cogOff[0] * SCALE, cogOff[2] * SCALE, -cogOff[1] * SCALE);
      const nozDelta = new THREE.Vector3(nozOff[0] * SCALE, nozOff[2] * SCALE, -nozOff[1] * SCALE);

      if (!basePos[cogNodeId]) {
        basePos[cogNodeId] = thBase.clone().add(cogDelta);
        deformedPos[cogNodeId] = (deformedPos[thNodeId] ?? thBase).clone().add(cogDelta);
        if (mVecs[thNodeId] && !mVecs[cogNodeId]) {
          mVecs[cogNodeId] = mVecs[thNodeId].clone();
        }
      }
      if (!basePos[nozNodeId]) {
        basePos[nozNodeId] = thBase.clone().add(nozDelta);
        deformedPos[nozNodeId] = (deformedPos[thNodeId] ?? thBase).clone().add(nozDelta);
        if (mVecs[thNodeId] && !mVecs[nozNodeId]) {
          mVecs[nozNodeId] = mVecs[thNodeId].clone();
        }
      }
    }

    const thAnchor = viewMode === "fea_stress" ? deformedPos[thNodeId] : basePos[thNodeId];
    const cogAnchor = viewMode === "fea_stress" ? deformedPos[cogNodeId] : basePos[cogNodeId];
    const nozAnchor = viewMode === "fea_stress" ? deformedPos[nozNodeId] : basePos[nozNodeId];

    const isCogNozMode =
      viewMode === "assembly" ||
      viewMode === "fea_stress" ||
      viewMode === "modal" ||
      viewMode === "sensitivity";

    if (isCogNozMode && thAnchor && cogAnchor && nozAnchor) {
      const mountMatId = geom.mount_material ?? sim.spec.toolhead.mount_material ?? "abs_printed";
      const isAluminumMount =
        mountMatId === "aluminum_6061_t6" || mountMatId === "aluminum_7075_t6";
      const mountColorMap: Record<string, number> = {
        abs_printed: 0xe63946,
        pet_cf_printed: 0x3b4252,
        pa12_cf_printed: 0x475569,
        ppa_cf_printed: 0x2d333b,
        pps_cf_printed: 0x23272e,
        aluminum_6061_t6: 0xc6d0db,
        aluminum_7075_t6: 0xd1dae6,
      };

      // Mode-aware colors & emissive glows for carriage, mount strut, nozzle stem, CoG sphere & nozzle tip
      const carriageColorObj = new THREE.Color(mountColorMap[mountMatId] ?? 0xe63946);
      const carriageEmissive = new THREE.Color(0x000000);
      let carriageEmissiveInt = 0;
      let carriageOpacity = viewMode === "assembly" ? 0.92 : 0.65;

      const strut1ColorObj = new THREE.Color(
        viewMode === "assembly"
          ? isAluminumMount
            ? 0xe2e8f0
            : mountMatId.includes("_cf_")
              ? 0x64748b
              : 0x94a3b8
          : 0x38bdf8
      );
      const strut1Emissive = new THREE.Color(0x000000);
      let strut1EmissiveInt = 0;
      let strut1Radius = 0.032;

      const strut2ColorObj = new THREE.Color(0xd97706);
      const strut2Emissive = new THREE.Color(0x000000);
      let strut2EmissiveInt = 0;
      let strut2RadiusTop = 0.026;
      let strut2RadiusBot = 0.022;

      const cogSphereColorObj = new THREE.Color(0xf59e0b);
      const cogSphereEmissive = new THREE.Color(0xd97706);
      let cogSphereEmissiveInt = 0.65;
      let cogRingColorHex = 0xfde68a;

      const nozzleColorObj = new THREE.Color(0xfbbf24);
      const nozzleEmissive = new THREE.Color(0xb45309);
      let nozzleEmissiveInt = 0.28;
      const nozzleCollarColorObj = new THREE.Color(0xd97706);

      if (viewMode === "sensitivity") {
        const thMountEntry =
          memberSensMap["toolhead_mount"] ?? memberSensMap["toolhead_cog_spring"];
        const nozStemEntry =
          memberSensMap["nozzle_stem"] ?? memberSensMap["nozzle_stem_spring"];
        const lumpedMap = sim.sensitivity?.lumped_mass_sensitivities;
        const thMassEntry =
          lumpedMap?.["toolhead_cog"] ?? lumpedMap?.[cogNodeId] ?? lumpedMap?.[thNodeId];

        const mountScore = getLensScore(thMountEntry, adjointLens);
        const stemScore = getLensScore(nozStemEntry, adjointLens);
        const cogMassScore = thMassEntry
          ? getLensScore(thMassEntry, adjointLens)
          : adjointLens === "mass"
            ? getLensScore(thMountEntry, "mass")
            : 0;
        const carriageScore =
          Math.abs(mountScore) >= Math.abs(cogMassScore) ? mountScore : cogMassScore;

        sensitivityColorInto(carriageScore, carriageColorObj);
        carriageOpacity = 0.88;
        if (Math.abs(carriageScore) > 0.15) {
          carriageEmissive.copy(carriageColorObj);
          carriageEmissiveInt = Math.min(1, Math.abs(carriageScore)) * 0.42;
        }

        sensitivityColorInto(mountScore, strut1ColorObj);
        const absMount = Math.min(1, Math.abs(mountScore));
        strut1Radius = 0.036 + absMount * 0.026;
        if (absMount > 0.15) {
          strut1Emissive.copy(strut1ColorObj);
          strut1EmissiveInt = absMount * 0.55;
        }

        sensitivityColorInto(stemScore, strut2ColorObj);
        const absStem = Math.min(1, Math.abs(stemScore));
        strut2RadiusTop = 0.03 + absStem * 0.02;
        strut2RadiusBot = 0.024 + absStem * 0.016;
        if (absStem > 0.15) {
          strut2Emissive.copy(strut2ColorObj);
          strut2EmissiveInt = absStem * 0.55;
        }

        sensitivityColorInto(cogMassScore, cogSphereColorObj);
        const absCog = Math.min(1, Math.abs(cogMassScore));
        cogSphereEmissive.copy(cogSphereColorObj);
        cogSphereEmissiveInt = absCog > 0.12 ? 0.35 + absCog * 0.55 : 0.12;

        if (sim.sensitivity?.primary_problem_subsystem === "toolhead") {
          cogRingColorHex = 0xef4444;
        } else if (cogMassScore < -0.15) {
          cogRingColorHex = 0x22d3ee;
        } else if (mountScore > 0.25) {
          cogRingColorHex = 0xfb923c;
        } else {
          cogRingColorHex = 0x64748b;
        }

        if (absStem > 0.15) {
          sensitivityColorInto(stemScore, nozzleColorObj);
          nozzleEmissive.copy(nozzleColorObj);
          nozzleEmissiveInt = absStem * 0.48;
          nozzleCollarColorObj.copy(nozzleColorObj);
        } else {
          nozzleColorObj.set(0x64748b);
          nozzleEmissive.set(0x000000);
          nozzleEmissiveInt = 0;
          nozzleCollarColorObj.set(0x475569);
        }
      } else if (viewMode === "modal") {
        const thScalar = getModalNodeScalar(thNodeId);
        const cogScalar = getModalNodeScalar(cogNodeId);
        const nozScalar = getModalNodeScalar(nozNodeId);

        stressColorInto(thScalar, carriageColorObj);
        carriageOpacity = 0.78;
        if (thScalar > 0.4) {
          carriageEmissive.copy(carriageColorObj);
          carriageEmissiveInt = (thScalar - 0.3) * 0.4;
        }

        const s1 = 0.5 * (thScalar + cogScalar);
        stressColorInto(s1, strut1ColorObj);
        strut1Radius = 0.038;
        if (s1 > 0.45) {
          strut1Emissive.copy(strut1ColorObj);
          strut1EmissiveInt = (s1 - 0.35) * 0.5;
        }

        const s2 = 0.5 * (cogScalar + nozScalar);
        stressColorInto(s2, strut2ColorObj);
        strut2RadiusTop = 0.032;
        strut2RadiusBot = 0.026;
        if (s2 > 0.45) {
          strut2Emissive.copy(strut2ColorObj);
          strut2EmissiveInt = (s2 - 0.35) * 0.5;
        }

        stressColorInto(cogScalar, cogSphereColorObj);
        cogSphereEmissive.copy(cogSphereColorObj);
        cogSphereEmissiveInt = 0.25 + cogScalar * 0.55;
        cogRingColorHex = cogScalar > 0.7 ? 0xfca5a5 : 0xfde68a;

        stressColorInto(nozScalar, nozzleColorObj);
        nozzleEmissive.copy(nozzleColorObj);
        nozzleEmissiveInt = 0.2 + nozScalar * 0.55;
        nozzleCollarColorObj.copy(nozzleColorObj);
      } else if (viewMode === "fea_stress") {
        const mountStress =
          stressMap["toolhead_mount"] ?? stressMap["toolhead_cog_spring"] ?? 0.35;
        const stemStress =
          stressMap["nozzle_stem"] ?? stressMap["nozzle_stem_spring"] ?? mountStress * 0.8;
        stressColorInto(mountStress, strut1ColorObj);
        stressColorInto(stemStress, strut2ColorObj);
      }

      // Pre-compute toolhead inspection metadata
      const thMountEntry =
        memberSensMap["toolhead_mount"] ?? memberSensMap["toolhead_cog_spring"];
      const nozStemEntry =
        memberSensMap["nozzle_stem"] ?? memberSensMap["nozzle_stem_spring"];
      const lumpedMapInspect = sim.sensitivity?.lumped_mass_sensitivities;
      const thMassEntryInspect =
        lumpedMapInspect?.["toolhead_cog"] ??
        lumpedMapInspect?.[cogNodeId] ??
        lumpedMapInspect?.[thNodeId];
      const thMountStress =
        memberStressById["toolhead_mount"] ?? memberStressById["toolhead_cog_spring"];
      const nozStemStress =
        memberStressById["nozzle_stem"] ?? memberStressById["nozzle_stem_spring"];
      const thMountStiff = stiffenById["toolhead_mount"] ?? stiffenById["toolhead"];
      const thMassLight =
        lightenById["toolhead_assembly"] ?? lightenById["toolhead_cog"] ?? lightenById["toolhead"];
      const thSubShare = sim.sensitivity?.subsystem_breakdown?.toolhead?.compliance_share_pct;

      const carriageInspectInfo: InspectedPartInfo = {
        key: "toolhead_carriage",
        name: "Toolhead Carriage & Mount",
        subsystem: "toolhead",
        partGroupId: "toolhead_mount",
        role: "toolhead",
        material: mountMatId,
        profile: `${Math.round(sim.spec.toolhead.internal_stiffness_n_per_mm ?? 450)} N/mm mount`,
        complianceSharePct:
          thMountStiff?.compliance_share_pct ?? thMountEntry?.compliance_share_pct ?? thSubShare,
        heatScore: thMountEntry ? getLensScore(thMountEntry, adjointLens) : undefined,
        stressMpa: thMountStress?.von_mises_mpa,
        yieldMpa: thMountStress?.yield_mpa,
        estHzGain: thMountStiff?.est_hz_gain,
        hzPer100g: thMassEntryInspect?.hz_per_100g,
        whyItHurts: thMountStiff?.why_it_hurts,
        actionHint: thMountStiff?.action_hint,
      };
      const mountStrutInspectInfo: InspectedPartInfo = {
        ...carriageInspectInfo,
        key: "toolhead_mount",
        name: thMountStiff?.label ?? "Toolhead Mount Lever Arm",
      };
      const nozStemInspectInfo: InspectedPartInfo = {
        key: "nozzle_stem",
        name: "Hotend Nozzle Stem",
        subsystem: "toolhead",
        partGroupId: "toolhead_mount",
        role: "toolhead",
        material: sim.spec.toolhead.hotend.heatbreak_material,
        profile: `Drop Z: ${Math.round(Math.abs(sim.spec.toolhead.nozzle_offset_mm?.[2] ?? 62))} mm`,
        complianceSharePct: nozStemEntry?.compliance_share_pct ?? thMountStiff?.compliance_share_pct,
        heatScore: nozStemEntry ? getLensScore(nozStemEntry, adjointLens) : undefined,
        stressMpa: nozStemStress?.von_mises_mpa ?? thMountStress?.von_mises_mpa,
        yieldMpa: nozStemStress?.yield_mpa ?? thMountStress?.yield_mpa,
        estHzGain: thMountStiff?.est_hz_gain,
        whyItHurts: thMountStiff?.why_it_hurts,
        actionHint: thMountStiff?.action_hint,
      };
      const cogInspectInfo: InspectedPartInfo = {
        key: "toolhead_cog",
        name: thMassLight?.label ?? "Toolhead Moving Mass (CoG)",
        subsystem: "toolhead",
        partGroupId: "toolhead_assembly",
        role: "toolhead",
        material: `${Math.round(sim.spec.toolhead.total_mass_g)} g moving mass`,
        profile: `CoG Y/Z: ${Math.round(Math.abs(sim.spec.toolhead.cog_offset_mm?.[1] ?? 18))}/${Math.round(Math.abs(sim.spec.toolhead.cog_offset_mm?.[2] ?? 25))} mm`,
        complianceSharePct: thSubShare,
        heatScore: thMassEntryInspect ? getLensScore(thMassEntryInspect, adjointLens) : undefined,
        hzPer100g: thMassLight?.hz_per_100g ?? thMassEntryInspect?.hz_per_100g,
        whyItHurts: thMassLight?.why_it_hurts,
        actionHint:
          thMassLight?.action_hint ??
          (sim.spec.toolhead.total_mass_g > 235
            ? "Moving toolhead inertia limits XY acceleration and input-shaper frequency"
            : undefined),
      };
      const nozzleInspectInfo: InspectedPartInfo = {
        ...nozStemInspectInfo,
        key: "nozzle_tip",
        name: `Hotend Nozzle (${sim.spec.toolhead.hotend.nozzle_diameter_mm}mm)`,
      };

      // Compact carriage block mounted on the X-beam centerline
      const thGroup = new THREE.Group();
      const carriageBody = new THREE.Mesh(
        new THREE.BoxGeometry(0.42, 0.28, 0.30),
        new THREE.MeshStandardMaterial({
          color: carriageColorObj,
          emissive: carriageEmissive,
          emissiveIntensity: carriageEmissiveInt,
          metalness: viewMode === "sensitivity" ? 0.24 : isAluminumMount ? 0.72 : 0.28,
          roughness: viewMode === "sensitivity" ? 0.4 : isAluminumMount ? 0.26 : 0.45,
          transparent: viewMode !== "assembly",
          opacity: carriageOpacity,
        })
      );
      tagInspectable(carriageBody, carriageInspectInfo);
      thGroup.add(carriageBody);
      thGroup.position.copy(thAnchor);
      group.add(thGroup);

      // Sleek lever-arm struts connecting X-beam carriage -> CoG -> Nozzle tip
      const strutGeo1 = new THREE.CylinderGeometry(strut1Radius, strut1Radius, 1, 10);
      const strutMat1 = new THREE.MeshStandardMaterial({
        color: strut1ColorObj,
        emissive: strut1Emissive,
        emissiveIntensity: strut1EmissiveInt,
        metalness: viewMode === "sensitivity" ? 0.3 : 0.7,
        roughness: viewMode === "sensitivity" ? 0.36 : 0.28,
      });
      const strutToCog = new THREE.Mesh(strutGeo1, strutMat1);
      orientBeamMesh(strutToCog, thAnchor, cogAnchor);
      tagInspectable(strutToCog, mountStrutInspectInfo);
      group.add(strutToCog);

      const strutGeo2 = new THREE.CylinderGeometry(strut2RadiusTop, strut2RadiusBot, 1, 10);
      const strutMat2 = new THREE.MeshStandardMaterial({
        color: strut2ColorObj,
        emissive: strut2Emissive,
        emissiveIntensity: strut2EmissiveInt,
        metalness: viewMode === "sensitivity" ? 0.3 : 0.75,
        roughness: viewMode === "sensitivity" ? 0.36 : 0.25,
      });
      const strutToNoz = new THREE.Mesh(strutGeo2, strutMat2);
      orientBeamMesh(strutToNoz, cogAnchor, nozAnchor);
      tagInspectable(strutToNoz, nozStemInspectInfo);
      group.add(strutToNoz);

      // Undeformed lever-arm ghost reference in FEA & Modal modes
      if ((viewMode === "fea_stress" || viewMode === "modal") && basePos[thNodeId] && basePos[cogNodeId] && basePos[nozNodeId]) {
        const ghostGeo = new THREE.BufferGeometry().setFromPoints([
          basePos[thNodeId],
          basePos[cogNodeId],
          basePos[nozNodeId],
        ]);
        const ghostMat = new THREE.LineBasicMaterial({
          color: 0x94a3b8,
          transparent: true,
          opacity: 0.45,
        });
        group.add(new THREE.Line(ghostGeo, ghostMat));
      }

      // Glowing CoG marker sphere + equatorial target ring
      const cogGroup = new THREE.Group();
      const cogSphere = new THREE.Mesh(
        new THREE.SphereGeometry(0.13, 18, 18),
        new THREE.MeshStandardMaterial({
          color: cogSphereColorObj,
          emissive: cogSphereEmissive,
          emissiveIntensity: cogSphereEmissiveInt,
          roughness: 0.22,
          metalness: 0.18,
        })
      );
      tagInspectable(cogSphere, cogInspectInfo);
      cogGroup.add(cogSphere);
      const cogRing = new THREE.Mesh(
        new THREE.TorusGeometry(0.19, 0.018, 10, 24),
        new THREE.MeshBasicMaterial({
          color: cogRingColorHex,
          transparent: true,
          opacity: 0.9,
        })
      );
      cogRing.rotation.x = Math.PI * 0.5;
      cogGroup.add(cogRing);
      cogGroup.position.copy(cogAnchor);
      group.add(cogGroup);

      // Sharp inverted cone whose apex sits exactly at nozzle_tip
      const nozzleGroup = new THREE.Group();
      const nozzleCone = new THREE.Mesh(
        new THREE.ConeGeometry(0.09, 0.24, 16),
        new THREE.MeshStandardMaterial({
          color: nozzleColorObj,
          emissive: nozzleEmissive,
          emissiveIntensity: nozzleEmissiveInt,
          metalness: viewMode === "sensitivity" ? 0.35 : 0.88,
          roughness: viewMode === "sensitivity" ? 0.34 : 0.18,
        })
      );
      nozzleCone.rotation.x = Math.PI;
      nozzleCone.position.y = 0.12; // tip at local (0, 0, 0)
      tagInspectable(nozzleCone, nozzleInspectInfo);
      nozzleGroup.add(nozzleCone);

      const nozzleCollar = new THREE.Mesh(
        new THREE.CylinderGeometry(0.095, 0.095, 0.08, 14),
        new THREE.MeshStandardMaterial({
          color: nozzleCollarColorObj,
          metalness: viewMode === "sensitivity" ? 0.35 : 0.82,
          roughness: viewMode === "sensitivity" ? 0.36 : 0.24,
        })
      );
      nozzleCollar.position.y = 0.27;
      nozzleGroup.add(nozzleCollar);

      nozzleGroup.position.copy(nozAnchor);
      if (viewMode === "fea_stress" && basePos[cogNodeId] && basePos[nozNodeId]) {
        const v0 = new THREE.Vector3().subVectors(basePos[cogNodeId], basePos[nozNodeId]);
        const v1 = new THREE.Vector3().subVectors(cogAnchor, nozAnchor);
        if (v0.lengthSq() > 1e-6 && v1.lengthSq() > 1e-6) {
          nozzleGroup.quaternion.setFromUnitVectors(v0.normalize(), v1.normalize());
        }
      }
      group.add(nozzleGroup);

      if (viewMode === "modal") {
        stateRef.current.modalToolhead = thGroup;
        stateRef.current.modalToolheadNodeId = thNodeId;
        stateRef.current.modalCogMarker = cogGroup;
        stateRef.current.modalCogNodeId = cogNodeId;
        stateRef.current.modalNozzleMarker = nozzleGroup;
        stateRef.current.modalNozzleNodeId = nozNodeId;
        stateRef.current.modalNozzleRefNodeId = cogNodeId;
        stateRef.current.modalMeshes.push({ mesh: strutToCog, startId: thNodeId, endId: cogNodeId });
        stateRef.current.modalMeshes.push({ mesh: strutToNoz, startId: cogNodeId, endId: nozNodeId });
      }
    } else if (thAnchor) {
      // Translucent toolhead reference in Virtual Print (toolpath) mode
      const thGroup = new THREE.Group();
      const [ex, ey, ez] = geom.toolhead_envelope_mm;
      const body = new THREE.Mesh(
        new THREE.BoxGeometry(ex * SCALE, ez * SCALE * 0.72, ey * SCALE),
        new THREE.MeshStandardMaterial({
          color: 0xe63946,
          roughness: 0.4,
          transparent: true,
          opacity: 0.35,
        })
      );
      body.position.y = -ez * SCALE * 0.2;
      thGroup.add(body);

      const nozzle = new THREE.Mesh(
        new THREE.ConeGeometry(0.08, 0.22, 10),
        new THREE.MeshStandardMaterial({ color: 0xfbbf24, metalness: 0.85, roughness: 0.2 })
      );
      nozzle.rotation.x = Math.PI;
      nozzle.position.y = -ez * SCALE * 0.65;
      thGroup.add(nozzle);

      thGroup.position.copy(thAnchor);
      group.add(thGroup);
    }

    // Glowing #1 Problem Area hotspot beacon in Adjoint ROI (sensitivity) mode
    if (viewMode === "sensitivity" && sim.sensitivity) {
      let primarySub: SubsystemKey | undefined = sim.sensitivity.primary_problem_subsystem;
      if (!primarySub && sim.sensitivity.subsystem_breakdown) {
        let bestShare = -1;
        for (const key of SUBSYSTEM_ORDER) {
          const share = sim.sensitivity.subsystem_breakdown[key]?.compliance_share_pct ?? -1;
          if (share > bestShare) {
            bestShare = share;
            primarySub = key;
          }
        }
      }

      let beaconCenter: THREE.Vector3 | null = null;
      let beaconRadius = 0.48;
      if (primarySub === "toolhead" && cogAnchor && nozAnchor) {
        beaconCenter = cogAnchor.clone().lerp(nozAnchor, 0.42);
        beaconRadius = 0.44;
      } else if (primarySub === "gantry") {
        if (basePos["y_car_l"] && basePos["y_car_r"]) {
          beaconCenter = basePos["y_car_l"].clone().add(basePos["y_car_r"]).multiplyScalar(0.5);
        } else if (thAnchor) {
          beaconCenter = thAnchor.clone();
        }
        beaconRadius = 0.56;
      } else if (primarySub === "bed") {
        beaconCenter = bedBaseCenter.clone().add(new THREE.Vector3(0, 0.08, 0));
        beaconRadius = 0.72;
      } else if (primarySub === "frame") {
        const topTargetId = sim.sensitivity.top_stiffen_targets?.[0]?.id;
        const targetMember = topTargetId
          ? geom.members.find((mm) => mm.id === topTargetId)
          : undefined;
        if (targetMember && basePos[targetMember.start_id] && basePos[targetMember.end_id]) {
          beaconCenter = basePos[targetMember.start_id]
            .clone()
            .add(basePos[targetMember.end_id])
            .multiplyScalar(0.5);
          beaconRadius = 0.52;
        }
      }

      if (beaconCenter) {
        const beaconGroup = new THREE.Group();
        const outerRing = new THREE.Mesh(
          new THREE.TorusGeometry(beaconRadius, 0.022, 12, 36),
          new THREE.MeshBasicMaterial({
            color: 0xef4444,
            transparent: true,
            opacity: 0.78,
          })
        );
        outerRing.rotation.x = Math.PI * 0.5;
        beaconGroup.add(outerRing);

        const vertRing = new THREE.Mesh(
          new THREE.TorusGeometry(beaconRadius * 0.82, 0.014, 10, 32),
          new THREE.MeshBasicMaterial({
            color: 0xfca5a5,
            transparent: true,
            opacity: 0.52,
          })
        );
        beaconGroup.add(vertRing);

        beaconGroup.position.copy(beaconCenter);
        group.add(beaconGroup);
      }
    }

    // Virtual Print Toolpath overlay
    if (viewMode === "toolpath" && sim.virtual_print.segments.length > 0) {
      const positions: number[] = [];
      const colors: number[] = [];
      const defectHex: Record<string, string> = {
        ok: "#22c55e",
        ringing: "#f59e0b",
        underextrusion: "#ef4444",
        step_skip: "#d946ef",
        cooling: "#38bdf8",
        travel: "#475569",
      };

      for (const seg of sim.virtual_print.segments) {
        if (seg.layer_index > layerLimit) continue;
        if (!seg.is_extrusion) continue; // keep view crisp with extrusion moves
        const p0 = toThree(
          [orig[0] + seg.start_mm[0], orig[1] + seg.start_mm[1], orig[2] + seg.start_mm[2]],
          cx,
          cy
        );
        const p1 = toThree(
          [orig[0] + seg.end_mm[0], orig[1] + seg.end_mm[1], orig[2] + seg.end_mm[2]],
          cx,
          cy
        );
        positions.push(p0.x, p0.y, p0.z, p1.x, p1.y, p1.z);
        const col = new THREE.Color(defectHex[seg.defect] || "#22c55e");
        colors.push(col.r, col.g, col.b, col.r, col.g, col.b);
      }

      if (positions.length > 0) {
        const tpGeo = new THREE.BufferGeometry();
        tpGeo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
        tpGeo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
        const tpMat = new THREE.LineBasicMaterial({ vertexColors: true });
        group.add(new THREE.LineSegments(tpGeo, tpMat));
      }
    }

    // Reconcile pinned Part Inspector selection with newly built meshes after spec/mode change
    if (stateRef.current.pinnedKey) {
      const matched = stateRef.current.inspectableMeshes.find(
        (mesh) => (mesh.userData?.inspectInfo as InspectedPartInfo | undefined)?.key === stateRef.current.pinnedKey
      );
      if (matched && matched.userData?.inspectInfo) {
        stateRef.current.pinnedMesh = matched;
        applyInspectHighlight(matched, "pin", 0.5);
        setPinnedPart(matched.userData.inspectInfo as InspectedPartInfo);
      } else {
        stateRef.current.pinnedMesh = null;
        stateRef.current.pinnedKey = null;
        setPinnedPart(null);
      }
    }
  }, [
    sim,
    solidFea,
    tetFea,
    showTetPart,
    topoResult,
    showCarvedGhost,
    viewMode,
    selectedModeIdx,
    feaScale,
    feaLoadCase,
    layerLimit,
    adjointLens,
    modalColorMode,
    showBedWarp,
  ]);

  const maxLayers = Math.max(0, (sim?.virtual_print.layer_count ?? 1) - 1);
  const clampedModeIdx =
    sim && sim.modal_fea.modes.length > 0
      ? Math.min(selectedModeIdx, sim.modal_fea.modes.length - 1)
      : 0;
  const activeMode = sim?.modal_fea.modes[clampedModeIdx];

  const xRailMountHud: XRailMountType =
    sim?.geometry.linear_rails?.x_rail_mount ??
    sim?.spec.gantry.x_rail_mount ??
    (sim?.spec.gantry.dual_x_rails ? "dual_top_bottom" : "front");
  const yRailMountHud: YRailMountType =
    sim?.geometry.linear_rails?.y_rail_mount ?? sim?.spec.gantry.y_rail_mount ?? "top";
  const isCoreXyHud = sim?.geometry.architecture === "corexy";
  const X_MOUNT_HUD_LABELS: Record<XRailMountType, string> = {
    front: "Front",
    top: "Top",
    bottom: "Bottom",
    dual_top_bottom: "Dual Top+Bot",
  };
  const Y_MOUNT_HUD_LABELS: Record<YRailMountType, string> = {
    top: "Top",
    inner: "Inner",
    bottom: "Bottom",
  };
  const railHudBadgeText =
    sim &&
    sim.geometry.architecture !== "delta" &&
    (xRailMountHud !== "front" || (isCoreXyHud && yRailMountHud !== "top"))
      ? isCoreXyHud
        ? `X ${X_MOUNT_HUD_LABELS[xRailMountHud]} · Y ${Y_MOUNT_HUD_LABELS[yRailMountHud]}`
        : `X ${X_MOUNT_HUD_LABELS[xRailMountHud]}`
      : null;

  const bedPlateFeaHud = sim?.bed_plate_fea ?? null;
  const canShowBedWarpHud =
    Boolean(bedPlateFeaHud) &&
    (viewMode === "assembly" ||
      viewMode === "fea_stress" ||
      viewMode === "modal" ||
      viewMode === "sensitivity");
  const bedWarpHintHot =
    canShowBedWarpHud &&
    !showBedWarp &&
    (bedPlateFeaHud?.total_flatness_error_um ?? 0) >= 50;

  return (
    <div className="viewport-wrap">
      <div ref={containerRef} className="viewport-canvas" />

      <div className="viewport-hud-top">
        <div className="viewport-hud-top-left">
          <div className="hud-pill-group">
            {(
              [
                ["assembly", "Assembly"],
                ["fea_stress", "Frame FEA"],
                ["modal", "Modal"],
                ["sensitivity", "Adjoint ROI"],
                ["toolpath", "Virtual Print"],
                ["solid_part", "3D Solid Part"],
                ["topology", "Topo Opt"],
              ] as Array<[ViewMode, string]>
            ).map(([id, label]) => (
              <button
                key={id}
                className={`hud-pill ${viewMode === id ? "active" : ""}`}
                onClick={() => onViewModeChange(id)}
              >
                {label}
              </button>
            ))}
          </div>

          {canShowBedWarpHud && bedPlateFeaHud && (
            <button
              type="button"
              className={`bed-warp-toggle-pill ${showBedWarp ? "active" : ""} ${
                bedWarpHintHot ? "hint-hot" : ""
              }`}
              onClick={() => setShowBedWarp(!showBedWarp)}
              title={`Toggle 3D MITC4 Heated Bed Thermal/Gravity Warp Mesh (Flatness error: ${bedPlateFeaHud.total_flatness_error_um.toFixed(
                1
              )} µm, Taco: ${bedPlateFeaHud.taco_buckling_um.toFixed(1)} µm)`}
            >
              <span>Bed Thermal/Warp</span>
              {bedWarpHintHot && (
                <span className="bed-warp-hint-tag">
                  {Math.round(bedPlateFeaHud.total_flatness_error_um)} µm
                </span>
              )}
            </button>
          )}
        </div>

        {sim && (
          <div className="hud-card hud-stats">
            <span>
              <strong>{sim.geometry.mass_summary.total_mass_kg} kg</strong> total
            </span>
            <span>
              X/Y moving: <strong>{sim.geometry.mass_summary.x_moving_mass_g}g</strong> /{" "}
              <strong>{sim.geometry.mass_summary.y_moving_mass_g}g</strong>
            </span>
            <span>
              Est. BOM: <strong>${sim.geometry.mass_summary.estimated_material_cost_usd}</strong>
            </span>
          </div>
        )}
      </div>

      {/* Compact Hover Pill for 3D Part Raycasting */}
      {hoveredPart && hoveredPart.key !== pinnedPart?.key && (
        <div className="viewport-hover-pill">
          <span
            className="swatch-dot"
            style={{ background: SUBSYSTEM_META[hoveredPart.subsystem].color }}
          />
          <strong className="viewport-hover-name">{hoveredPart.name}</strong>
          {hoveredPart.complianceSharePct != null && hoveredPart.complianceSharePct >= 1 && (
            <span className="viewport-hover-stat">
              {hoveredPart.complianceSharePct.toFixed(0)}% flex
            </span>
          )}
          {hoveredPart.bedFlatnessUm != null && (
            <span className="viewport-hover-stat">
              {hoveredPart.bedFlatnessUm.toFixed(0)} µm flat err
            </span>
          )}
          <span className="viewport-hover-hint">Click to inspect</span>
        </div>
      )}

      {/* Floating 3D Part Inspector Card */}
      {pinnedPart && sim && (
        <div className="part-inspector-card">
          <div className="part-inspector-head">
            <div className="part-inspector-title-wrap">
              <span
                className="part-inspector-sub-badge"
                style={{
                  borderColor: SUBSYSTEM_META[pinnedPart.subsystem].color,
                  color: SUBSYSTEM_META[pinnedPart.subsystem].color,
                }}
              >
                {SUBSYSTEM_META[pinnedPart.subsystem].label}
              </span>
              <strong className="part-inspector-title">{pinnedPart.name}</strong>
            </div>
            <button
              type="button"
              className="part-inspector-close"
              onClick={handleCloseInspector}
              title="Close Part Inspector"
              aria-label="Close Part Inspector"
            >
              ×
            </button>
          </div>

          {(pinnedPart.material || pinnedPart.profile) && (
            <div className="part-inspector-meta">
              {pinnedPart.profile && <span>{pinnedPart.profile.replace(/_/g, " ")}</span>}
              {pinnedPart.profile && pinnedPart.material && <span>·</span>}
              {pinnedPart.material && <span>{pinnedPart.material.replace(/_/g, " ")}</span>}
            </div>
          )}

          <div className="part-inspector-grid">
            {pinnedPart.complianceSharePct != null && (
              <div className="part-inspector-cell">
                <span className="part-inspector-label">Nozzle Flex</span>
                <span className="part-inspector-val">
                  {pinnedPart.complianceSharePct.toFixed(1)}%
                </span>
              </div>
            )}
            {pinnedPart.stressMpa != null && (
              <div className="part-inspector-cell">
                <span className="part-inspector-label">von Mises</span>
                <span className="part-inspector-val">
                  {pinnedPart.stressMpa.toFixed(2)} MPa
                  {pinnedPart.yieldMpa ? ` / ${Math.round(pinnedPart.yieldMpa)}` : ""}
                </span>
              </div>
            )}
            {pinnedPart.heatScore != null && (
              <div className="part-inspector-cell">
                <span className="part-inspector-label">Adjoint ROI</span>
                <span
                  className={`part-inspector-val ${
                    pinnedPart.heatScore > 0.25
                      ? "tag-warn"
                      : pinnedPart.heatScore < -0.2
                        ? "tag-ok"
                        : ""
                  }`}
                >
                  {pinnedPart.heatScore >= 0 ? "+" : ""}
                  {(pinnedPart.heatScore * 100).toFixed(0)}%
                </span>
              </div>
            )}
            {pinnedPart.estHzGain != null && pinnedPart.estHzGain > 0 && (
              <div className="part-inspector-cell">
                <span className="part-inspector-label">Stiffen Gain</span>
                <span className="part-inspector-val tag-ok">
                  +{formatHzGain(pinnedPart.estHzGain)} Hz
                </span>
              </div>
            )}
            {pinnedPart.hzPer100g != null && Math.abs(pinnedPart.hzPer100g) >= 0.01 && (
              <div className="part-inspector-cell">
                <span className="part-inspector-label">Mass Sens.</span>
                <span className="part-inspector-val">
                  {pinnedPart.hzPer100g > 0 ? "+" : ""}
                  {pinnedPart.hzPer100g.toFixed(2)} Hz/100g
                </span>
              </div>
            )}
            {pinnedPart.bedFlatnessUm != null && (
              <div className="part-inspector-cell">
                <span className="part-inspector-label">Bed Flatness</span>
                <span
                  className={`part-inspector-val ${
                    pinnedPart.bedRisk === "critical" || pinnedPart.bedRisk === "high"
                      ? "tag-bad"
                      : pinnedPart.bedRisk === "moderate"
                        ? "tag-warn"
                        : "tag-ok"
                  }`}
                >
                  {pinnedPart.bedFlatnessUm.toFixed(1)} µm
                </span>
              </div>
            )}
            {pinnedPart.bedTacoUm != null && (
              <div className="part-inspector-cell">
                <span className="part-inspector-label">Taco Warp</span>
                <span className="part-inspector-val">{pinnedPart.bedTacoUm.toFixed(1)} µm</span>
              </div>
            )}
            {pinnedPart.bedModeHz != null && (
              <div className="part-inspector-cell">
                <span className="part-inspector-label">Plate Mode f₁</span>
                <span className="part-inspector-val">{pinnedPart.bedModeHz.toFixed(0)} Hz</span>
              </div>
            )}
          </div>

          {pinnedPart.whyItHurts && (
            <div className="part-inspector-why">{pinnedPart.whyItHurts}</div>
          )}
          {pinnedPart.actionHint && (
            <div className="part-inspector-hint">
              <span className="sens-fix-prefix">Main limit:</span> {pinnedPart.actionHint}
            </div>
          )}
        </div>
      )}

      <div className="viewport-hud-bottom">
        {canShowBedWarpHud && showBedWarp && bedPlateFeaHud && (
          <div className="hud-card bed-warp-hud">
            <span className="badge-pill">2D MITC4 Bed Plate</span>
            <div className="bed-warp-legend">
              <span style={{ color: "#60a5fa" }}>Valley</span>
              <div className="legend-gradient bed-warp" />
              <span style={{ color: "#f87171" }}>Crown/Bow</span>
            </div>
            <span style={{ color: "var(--muted)" }}>|</span>
            <span>
              Flatness:{" "}
              <strong
                className={
                  bedPlateFeaHud.first_layer_risk === "critical" ||
                  bedPlateFeaHud.first_layer_risk === "high"
                    ? "tag-bad"
                    : bedPlateFeaHud.first_layer_risk === "moderate"
                      ? "tag-warn"
                      : "tag-ok"
                }
              >
                {bedPlateFeaHud.total_flatness_error_um.toFixed(1)} µm
              </strong>
            </span>
            <span style={{ color: "var(--muted)" }}>|</span>
            <span>
              Taco: <strong>{bedPlateFeaHud.taco_buckling_um.toFixed(1)} µm</strong>
            </span>
            <span style={{ color: "var(--muted)" }}>|</span>
            <span>
              Plate 1st Mode: <strong>{bedPlateFeaHud.plate_first_mode_hz.toFixed(0)} Hz</strong>
            </span>
          </div>
        )}
        {viewMode === "fea_stress" && sim && (
          <>
            <div className="hud-card">
              <span style={{ color: "var(--muted)", fontWeight: 600 }}>Load:</span>
              {(["xy", "x", "y", "gravity"] as FeaLoadCase[]).map((id) => (
                <button
                  key={id}
                  type="button"
                  className={`hud-pill ${feaLoadCase === id ? "active" : ""}`}
                  style={{ padding: "3px 8px", fontSize: 11 }}
                  title={FEA_LOAD_META[id].explanation}
                  onClick={() => onFeaLoadCaseChange(id)}
                >
                  {FEA_LOAD_META[id].label}
                </button>
              ))}
            </div>
            <div className="hud-card">
              {sim.geometry.shell_panels && sim.geometry.shell_panels.length > 0 && (
                <span className="badge-pill">
                  MITC4 {sim.geometry.panel_mesh_n ? `${sim.geometry.panel_mesh_n}×${sim.geometry.panel_mesh_n}` : "Shell"} ·{" "}
                  {sim.geometry.shell_panels.length} quads
                </span>
              )}
              <span>0 MPa</span>
              <div className="legend-gradient" />
              <span>
                {Math.max(
                  sim.static_fea.max_von_mises_mpa,
                  sim.static_fea.panel_max_von_mises_mpa ?? 0
                ).toFixed(2)}{" "}
                MPa
              </span>
              <span style={{ color: "var(--muted)" }}>|</span>
              <label style={{ display: "flex", alignItems: "center", gap: 6, color: "var(--text)" }}>
                Deflection ×{feaScale}
                <input
                  type="range"
                  min={50}
                  max={2500}
                  step={50}
                  value={feaScale}
                  onChange={(e) => onFeaScaleChange(Number(e.target.value))}
                  style={{ width: 100 }}
                />
              </label>
            </div>
            <div className="hud-card hud-guide-bar">
              <div className="hud-guide-row">
                <span className="badge-pill hud-guide-pill">{FEA_LOAD_META[feaLoadCase].label}</span>
                <span className="hud-guide-lead">{FEA_LOAD_META[feaLoadCase].explanation}</span>
              </div>
              {sim.static_fea.top_flex_explanation ? (
                <div className="hud-guide-sub">
                  {sim.static_fea.top_flex_explanation}
                  {sim.static_fea.peak_stress_part_label &&
                    !sim.static_fea.top_flex_explanation
                      .toLowerCase()
                      .includes(sim.static_fea.peak_stress_part_label.toLowerCase()) && (
                      <>
                        {" "}
                        · Peak stress: <strong>{sim.static_fea.peak_stress_part_label}</strong>
                      </>
                    )}
                </div>
              ) : sim.static_fea.peak_stress_part_label ? (
                <div className="hud-guide-sub">
                  Peak von Mises stress in <strong>{sim.static_fea.peak_stress_part_label}</strong>
                </div>
              ) : null}
            </div>
          </>
        )}

        {viewMode === "modal" && sim && (
          <>
            <div className="hud-card">
              <span style={{ color: "var(--muted)", fontWeight: 600 }}>Mode:</span>
              {sim.modal_fea.modes.slice(0, 6).map((m, idx) => {
                const axisTag = m.excited_axis ? `${m.excited_axis}-Accel` : m.dominant_axis;
                const isWorst = sim.modal_fea.worst_print_mode_number === m.mode_number;
                const impMeta = m.print_impact_level ? impactLevelMeta(m.print_impact_level) : null;
                return (
                  <button
                    key={m.mode_number}
                    type="button"
                    className={`hud-pill modal-mode-pill ${clampedModeIdx === idx ? "active" : ""} ${
                      isWorst ? "worst-mode" : ""
                    }`}
                    style={{ padding: "4px 8px" }}
                    onClick={() => onSelectModeIdx(idx)}
                    title={
                      [
                        `Mode ${m.mode_number} (${m.frequency_hz.toFixed(1)} Hz · ${axisTag})`,
                        m.mode_label ? `— ${m.mode_label}` : "",
                        m.print_impact_score != null
                          ? `• Print Impact ${Math.round(m.print_impact_score)}/100`
                          : "",
                        isWorst ? "★ Worst print-impact mode" : "",
                      ]
                        .filter(Boolean)
                        .join(" ")
                    }
                  >
                    {impMeta && (
                      <span
                        className="impact-dot-mini"
                        style={{ background: impMeta.dotColor }}
                      />
                    )}
                    {isWorst && (
                      <span className="worst-star" title="Worst print-impact mode">
                        ★
                      </span>
                    )}
                    M{m.mode_number} · {m.frequency_hz.toFixed(1)} Hz ({axisTag})
                  </button>
                );
              })}
              <span style={{ color: "var(--muted)" }}>|</span>
              <span style={{ color: "var(--muted)", fontWeight: 600 }}>Color:</span>
              <button
                type="button"
                className={`hud-pill ${modalColorMode === "relative" ? "active" : ""}`}
                style={{ padding: "3px 7px", fontSize: 10.5 }}
                onClick={() => setModalColorMode("relative")}
                title="Color nodes by |u_node - u_bed| — whole-frame rocking stays cool blue while flopping toolheads blaze red"
              >
                Relative to Bed (Print Error)
              </button>
              <button
                type="button"
                className={`hud-pill ${modalColorMode === "global" ? "active" : ""}`}
                style={{ padding: "3px 7px", fontSize: 10.5 }}
                onClick={() => setModalColorMode("global")}
                title="Color nodes by absolute global modal displacement amplitude |u_node|"
              >
                Global Motion
              </button>
            </div>

            {activeMode && (() => {
              const axisTag = activeMode.excited_axis
                ? `${activeMode.excited_axis}-Accel`
                : activeMode.dominant_axis;
              const sevMeta = modeSeverityMeta(activeMode.severity);
              const impMeta =
                activeMode.print_impact_level || activeMode.print_impact_score != null
                  ? impactLevelMeta(
                      activeMode.print_impact_level ??
                        ((activeMode.print_impact_score ?? 0) >= 75
                          ? "critical"
                          : (activeMode.print_impact_score ?? 0) >= 50
                            ? "high"
                            : (activeMode.print_impact_score ?? 0) >= 25
                              ? "moderate"
                              : "low")
                    )
                  : null;
              const isWorstActive =
                sim.modal_fea.worst_print_mode_number === activeMode.mode_number;
              const share = activeMode.subsystem_share_pct;

              return (
                <div className="hud-card hud-guide-bar modal-guide">
                  <div className="hud-guide-row hud-guide-header">
                    <div className="hud-guide-badges">
                      <span className="badge-pill hud-guide-pill">
                        {isWorstActive ? "★ " : ""}Mode {activeMode.mode_number} ·{" "}
                        {activeMode.frequency_hz.toFixed(1)} Hz
                      </span>
                      <span className="badge-pill hud-axis-pill">{axisTag}</span>
                      {impMeta ? (
                        <span className={`badge-pill ${impMeta.cls}`}>
                          {impMeta.label}
                          {activeMode.print_impact_score != null
                            ? ` ${Math.round(activeMode.print_impact_score)}/100`
                            : ""}
                        </span>
                      ) : (
                        sevMeta && <span className={`badge-pill ${sevMeta.cls}`}>{sevMeta.label}</span>
                      )}
                      {(activeMode.nozzle_whip_ratio ?? 0) >= 1.25 && (
                        <span
                          className={`badge-pill ${
                            (activeMode.nozzle_whip_ratio ?? 0) >= 1.8 ? "bad" : "warn"
                          }`}
                          title="Nozzle tip moves farther than the carriage due to rotational pitch/twist whip"
                        >
                          {activeMode.nozzle_whip_ratio!.toFixed(2)}× whip
                        </span>
                      )}
                      {activeMode.toolhead_flop_pct != null && activeMode.toolhead_flop_pct >= 5 && (
                        <span
                          className={`badge-pill ${
                            activeMode.toolhead_flop_pct >= 50 ? "bad" : "warn"
                          }`}
                          title="Fraction of relative nozzle-to-bed motion coming from local toolhead mount/stem flop"
                        >
                          Toolhead Flop: {Math.round(activeMode.toolhead_flop_pct)}%
                        </span>
                      )}
                      {activeMode.mode_label && (
                        <strong className="hud-guide-title">{activeMode.mode_label}</strong>
                      )}
                    </div>
                    <div className="hud-guide-meta">
                      <span>
                        <strong style={{ color: "#f59e0b" }}>●</strong> CoG ·{" "}
                        <strong style={{ color: "#fbbf24" }}>▼</strong> Nozzle
                      </span>
                      <span>·</span>
                      <span>
                        XYZ:{" "}
                        <strong>
                          {activeMode.toolhead_participation_xyz
                            .map((v) => `${Math.round(v * 100)}%`)
                            .join("/")}
                        </strong>
                      </span>
                      {sim.geometry.shell_panels && sim.geometry.shell_panels.length > 0 && (
                        <>
                          <span>·</span>
                          <span>
                            MITC4{" "}
                            {sim.geometry.panel_mesh_n
                              ? `${sim.geometry.panel_mesh_n}×${sim.geometry.panel_mesh_n}`
                              : "Shell"}
                            {(sim.modal_fea.panel_drum_freq_hz ?? 0) > 0
                              ? ` (${sim.modal_fea.panel_drum_freq_hz!.toFixed(0)} Hz)`
                              : ""}
                          </span>
                        </>
                      )}
                      {sim.geometry.extrusion_filling && sim.geometry.extrusion_filling.id !== "none" && (
                        <>
                          <span>·</span>
                          <span className="hud-fill-badge">
                            <span
                              className="swatch-dot"
                              style={{ background: sim.geometry.extrusion_filling.color_hex }}
                            />
                            {sim.geometry.extrusion_filling.short_name} (ζ=
                            {sim.geometry.extrusion_filling.damping_ratio.toFixed(3)})
                          </span>
                        </>
                      )}
                    </div>
                  </div>

                  {share && (
                    <div className="subsystem-bar-wrap">
                      <div className="subsystem-bar-track" title="Modal strain/kinetic energy share by subsystem">
                        {SUBSYSTEM_ORDER.map((k) => {
                          const pct = Math.max(0, share[k] ?? 0);
                          if (pct <= 0.5) return null;
                          return (
                            <div
                              key={k}
                              className="subsystem-bar-seg"
                              style={{ width: `${pct}%`, background: SUBSYSTEM_META[k].color }}
                              title={`${SUBSYSTEM_META[k].label}: ${Math.round(pct)}%`}
                            />
                          );
                        })}
                      </div>
                      <div className="subsystem-bar-legend">
                        {SUBSYSTEM_ORDER.map((k) => (
                          <span key={k} className="subsystem-legend-item">
                            <span
                              className="swatch-dot"
                              style={{ background: SUBSYSTEM_META[k].color }}
                            />
                            {SUBSYSTEM_META[k].label}{" "}
                            <strong>{Math.round(share[k] ?? 0)}%</strong>
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {activeMode.what_is_shaking && (
                    <div className="hud-guide-line">
                      <span className="hud-guide-key">What&apos;s shaking:</span>{" "}
                      <span className="hud-guide-val">{activeMode.what_is_shaking}</span>
                    </div>
                  )}
                  {activeMode.how_to_fix && (
                    <div className="hud-guide-line hud-fix-line">
                      <span className="hud-fix-key">Main limit:</span>{" "}
                      <span className="hud-fix-val">{activeMode.how_to_fix}</span>
                    </div>
                  )}
                </div>
              );
            })()}
          </>
        )}

        {viewMode === "sensitivity" && sim && (() => {
          const topStiff = sim.sensitivity?.top_stiffen_targets?.[0];
          const topLight = sim.sensitivity?.top_lighten_targets?.[0];
          const breakdown = sim.sensitivity?.subsystem_breakdown;
          const primarySub = sim.sensitivity?.primary_problem_subsystem;
          const stiffStatParts: string[] = [];
          if (topStiff) {
            if ((topStiff.est_hz_gain ?? 0) > 0) {
              stiffStatParts.push(`+${formatHzGain(topStiff.est_hz_gain!)} Hz potential`);
            }
            if (topStiff.compliance_share_pct >= 1) {
              stiffStatParts.push(`${topStiff.compliance_share_pct.toFixed(0)}% of flex`);
            }
            if (stiffStatParts.length === 0) {
              stiffStatParts.push(`+${topStiff.hz_per_usd.toFixed(2)} Hz/$`);
            }
          }
          const lightGain = topLight
            ? (topLight.hz_gain_per_100g_saved ?? Math.abs(topLight.hz_per_100g))
            : 0;

          return (
            <div className="hud-card hud-guide-bar sens-guide">
              <div className="hud-guide-row hud-guide-header">
                <div className="hud-sens-lens-row">
                  <span style={{ color: "var(--muted)", fontWeight: 700, fontSize: 10.5 }}>
                    Lens:
                  </span>
                  {ADJOINT_LENS_OPTIONS.map((opt) => (
                    <button
                      key={opt.id}
                      type="button"
                      className={`hud-pill ${adjointLens === opt.id ? "active" : ""}`}
                      style={{ padding: "3px 8px", fontSize: 10.5 }}
                      onClick={() => setAdjointLens(opt.id)}
                      title={opt.title}
                    >
                      {opt.label}
                    </button>
                  ))}
                </div>
                <div className="hud-guide-row hud-sens-legend">
                  <div className="legend-gradient sensitivity" />
                  {adjointLens !== "compliance" && (
                    <span style={{ color: "#22d3ee", fontWeight: 700 }}>Cyan = Inertia Hotspot</span>
                  )}
                  {adjointLens !== "compliance" && adjointLens !== "mass" && (
                    <span style={{ color: "var(--muted)" }}>·</span>
                  )}
                  <span style={{ color: "#94a3b8", fontWeight: 600 }}>Slate = Neutral</span>
                  {adjointLens !== "mass" && (
                    <>
                      <span style={{ color: "var(--muted)" }}>·</span>
                      <span style={{ color: "#fb923c", fontWeight: 700 }}>
                        Red/Orange = Compliance Hotspot
                      </span>
                    </>
                  )}
                </div>
              </div>

              {breakdown && (
                <div className="subsystem-bar-wrap">
                  <div className="subsystem-bar-track" title="Static nozzle-to-bed print error share by subsystem">
                    {SUBSYSTEM_ORDER.map((k) => {
                      const pct = Math.max(0, breakdown[k]?.compliance_share_pct ?? 0);
                      if (pct <= 0.5) return null;
                      return (
                        <div
                          key={k}
                          className="subsystem-bar-seg"
                          style={{ width: `${pct}%`, background: SUBSYSTEM_META[k].color }}
                          title={`${breakdown[k]?.label ?? SUBSYSTEM_META[k].label}: ${Math.round(pct)}% of flex`}
                        />
                      );
                    })}
                  </div>
                  <div className="subsystem-bar-legend">
                    <span style={{ color: "var(--muted)", fontWeight: 600 }}>Print-Error Share:</span>
                    {SUBSYSTEM_ORDER.map((k) => {
                      const entry = breakdown[k];
                      const isPrimary = primarySub === k;
                      return (
                        <span
                          key={k}
                          className={`subsystem-legend-item ${isPrimary ? "primary" : ""}`}
                        >
                          <span
                            className="swatch-dot"
                            style={{ background: SUBSYSTEM_META[k].color }}
                          />
                          {SUBSYSTEM_META[k].label}{" "}
                          <strong>{Math.round(entry?.compliance_share_pct ?? 0)}%</strong>
                          {isPrimary && <span className="subsystem-primary-tag">#1</span>}
                        </span>
                      );
                    })}
                  </div>
                </div>
              )}

              {(topStiff || topLight) && (
                <div className="hud-guide-row hud-sens-takeaway">
                  {topStiff && (
                    <span className="hud-sens-item">
                      <span className="hud-sens-tag stiffen">#1 Compliance:</span>{" "}
                      <strong>{topStiff.label ?? topStiff.id}</strong>{" "}
                      <span className="hud-sens-stat">({stiffStatParts.join(" · ")})</span>
                      {topStiff.action_hint && (
                        <>
                          {" "}
                          <span className="hud-fix-key">Main limit:</span>{" "}
                          <span className="hud-fix-val">{topStiff.action_hint}</span>
                        </>
                      )}
                    </span>
                  )}
                  {topLight && (
                    <span className="hud-sens-item">
                      <span className="hud-sens-tag lighten">#1 Inertia:</span>{" "}
                      <strong>{topLight.label ?? topLight.id}</strong>{" "}
                      <span className="hud-sens-stat">(+{lightGain.toFixed(2)} Hz / 100g)</span>
                    </span>
                  )}
                </div>
              )}
            </div>
          );
        })()}

        {viewMode === "toolpath" && sim && (
          <div className="hud-card">
            <span className="tag-ok">● OK</span>
            <span className="tag-warn">● Ringing</span>
            <span className="tag-bad">● Under-extrusion</span>
            <span style={{ color: "#d946ef" }}>● Step skip</span>
            <span style={{ color: "var(--muted)" }}>|</span>
            <label style={{ display: "flex", alignItems: "center", gap: 6, color: "var(--text)" }}>
              Layer {Math.min(layerLimit, maxLayers) + 1}/{maxLayers + 1}
              <input
                type="range"
                min={0}
                max={maxLayers}
                value={Math.min(layerLimit, maxLayers)}
                onChange={(e) => onLayerLimitChange(Number(e.target.value))}
                style={{ width: 110 }}
              />
            </label>
          </div>
        )}

        {viewMode === "solid_part" && showTetPart && tetFea && (
          <>
            <div className="hud-card">
              <span className="badge-pill tet">
                {formatBackendLabel(tetFea.backend)} • {formatElementLabel(tetFea.element_type)}
              </span>
              <span>0 MPa</span>
              <div className="legend-gradient" />
              <span>{tetFea.max_von_mises_mpa.toFixed(2)} MPa</span>
              <span style={{ color: "var(--muted)" }}>|</span>
              <label style={{ display: "flex", alignItems: "center", gap: 6, color: "var(--text)" }}>
                Deform ×{Math.round(feaScale * 0.08)}
                <input
                  type="range"
                  min={50}
                  max={2500}
                  step={50}
                  value={feaScale}
                  onChange={(e) => onFeaScaleChange(Number(e.target.value))}
                  style={{ width: 85 }}
                />
              </label>
            </div>
            <div className="hud-card">
              <span>
                <strong>{tetFea.part_name}</strong>
              </span>
              <span style={{ color: "var(--muted)" }}>|</span>
              <span>
                <strong>{tetFea.num_elements.toLocaleString()}</strong> els /{" "}
                <strong>{tetFea.num_dofs.toLocaleString()}</strong> DOF
              </span>
              <span style={{ color: "var(--muted)" }}>|</span>
              <span>
                Max σ: <strong>{tetFea.max_von_mises_mpa.toFixed(2)} MPa</strong>
              </span>
              <span style={{ color: "var(--muted)" }}>|</span>
              <span>
                SF:{" "}
                <strong className={tetFea.safety_factor < 1.5 ? "tag-bad" : "tag-ok"}>
                  {tetFea.safety_factor.toFixed(2)}×
                </strong>
              </span>
            </div>
          </>
        )}

        {viewMode === "solid_part" && showHexPart && solidFea && (
          <div className="hud-card">
            <span className="badge-pill">Voxel Hex8</span>
            <span>
              <strong>{solidFea.part_name}</strong> ({solidFea.num_elements} Hex8)
            </span>
            <span>0 MPa</span>
            <div className="legend-gradient" />
            <span>{solidFea.max_von_mises_mpa.toFixed(2)} MPa</span>
            <label style={{ display: "flex", alignItems: "center", gap: 6, color: "var(--text)" }}>
              Deform ×{Math.round(feaScale * 0.08)}
              <input
                type="range"
                min={50}
                max={2500}
                step={50}
                value={feaScale}
                onChange={(e) => onFeaScaleChange(Number(e.target.value))}
                style={{ width: 90 }}
              />
            </label>
          </div>
        )}

        {viewMode === "solid_part" && !showTetPart && !showHexPart && (
          <div className="hud-card" style={{ color: "var(--muted)" }}>
            Run 3D Solid Part FEA from the <strong style={{ color: "var(--text)" }}>Parts &amp; Topo</strong> tab to
            inspect curved Tet10 or Hex8 stress results.
          </div>
        )}

        {viewMode === "topology" && topoResult && (
          <>
            <div className="hud-card">
              <span>0 MPa</span>
              <div className="legend-gradient" />
              <span>{topoResult.max_von_mises_mpa.toFixed(2)} MPa</span>
              <span style={{ color: "var(--muted)" }}>|</span>
              <label style={{ display: "flex", alignItems: "center", gap: 6, color: "var(--text)" }}>
                Deform ×{Math.round(feaScale * 0.08)}
                <input
                  type="range"
                  min={50}
                  max={2500}
                  step={50}
                  value={feaScale}
                  onChange={(e) => onFeaScaleChange(Number(e.target.value))}
                  style={{ width: 85 }}
                />
              </label>
              <span style={{ color: "var(--muted)" }}>|</span>
              <label className="checkbox-row">
                <input
                  type="checkbox"
                  checked={showCarvedGhost}
                  onChange={(e) => setShowCarvedGhost(e.target.checked)}
                />
                <span>Show Carved Ghost</span>
              </label>
            </div>
            <div className="hud-card">
              <span>
                Mass:{" "}
                <strong>
                  {topoResult.baseline_mass_g.toFixed(1)}g → {topoResult.optimized_mass_g.toFixed(1)}g
                </strong>{" "}
                <span className="tag-ok">(-{topoResult.mass_reduction_pct.toFixed(1)}%)</span>
              </span>
              <span style={{ color: "var(--muted)" }}>|</span>
              <span>
                Specific Stiffness:{" "}
                <strong className={topoResult.specific_stiffness_gain_pct >= 0 ? "tag-ok" : "tag-warn"}>
                  {topoResult.specific_stiffness_gain_pct >= 0 ? "+" : ""}
                  {topoResult.specific_stiffness_gain_pct.toFixed(1)}%
                </strong>
              </span>
              <span style={{ color: "var(--muted)" }}>|</span>
              <span>
                SF:{" "}
                <strong className={topoResult.safety_factor < 1.5 ? "tag-bad" : "tag-ok"}>
                  {topoResult.safety_factor.toFixed(2)}×
                </strong>
              </span>
            </div>
          </>
        )}

        {viewMode === "topology" && !topoResult && (
          <div className="hud-card" style={{ color: "var(--muted)" }}>
            Run 3D Topology Optimization from the <strong style={{ color: "var(--text)" }}>Parts &amp; Topo</strong> tab
            to carve and inspect optimal material paths.
          </div>
        )}

        {viewMode === "assembly" && (
          <div className="hud-card" style={{ color: "var(--muted)" }}>
            <span>
              <strong style={{ color: "#f59e0b" }}>●</strong> Toolhead CoG
            </span>
            <span>
              <strong style={{ color: "#fbbf24" }}>▼</strong> Nozzle Tip
            </span>
            {sim?.geometry.gantry_brackets && (
              <span className="hud-inline-badge">
                <span
                  className="swatch-dot"
                  style={{ background: sim.geometry.gantry_brackets.color_hex }}
                />
                Brackets:{" "}
                <strong style={{ color: "var(--text)" }}>
                  {sim.geometry.gantry_brackets.short_name}
                </strong>
              </span>
            )}
            {sim?.geometry.extrusion_filling && sim.geometry.extrusion_filling.id !== "none" && (
              <span className="hud-fill-badge">
                <span
                  className="swatch-dot"
                  style={{ background: sim.geometry.extrusion_filling.color_hex }}
                />
                Fill: <strong>{sim.geometry.extrusion_filling.short_name}</strong> (+
                {sim.geometry.extrusion_filling.added_mass_kg.toFixed(1)} kg, ζ=
                {sim.geometry.extrusion_filling.damping_ratio.toFixed(3)})
              </span>
            )}
            {railHudBadgeText && (
              <span className="hud-fill-badge">
                <span className="swatch-dot" style={{ background: "#8e9aa8" }} />
                Rails: <strong>{railHudBadgeText}</strong>
              </span>
            )}
            <span>|</span>
            <span>Click part to inspect • Drag to orbit • Scroll to zoom</span>
          </div>
        )}
      </div>
    </div>
  );
};
