import React, { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import {
  bedWarpColorInto,
  sensitivityColorInto,
  stressColor,
  stressColorInto,
} from "../viewport/colormaps";
import {
  AdjointLens,
  BedMountType,
  FeaLoadCase,
  FrameFeetType,
  GeometryMember,
  ModalColorMode,
  ShellPanelGeometryData,
  SimulationResponse,
  SolidFeaEngine,
  SolidFEAResponse,
  SubsystemKey,
  TetFEAResponse,
  ToolheadFlexBreakdownData,
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
  if (mid.includes("backbone") || mid === "bay_backbone" || mid === "trident_backbone") {
    return {
      id: "trident_backbone",
      label: "Trident Backbone (Clee 3mm Steel)",
      subsystem: "frame",
    };
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
  toolheadPunchIn?: boolean;
  onToggleToolheadPunchIn?: (active: boolean) => void;
  sensitivityLens?: AdjointLens;
  onSensitivityLensChange?: (lens: AdjointLens) => void;
  modalColorMode?: ModalColorMode;
  onModalColorModeChange?: (mode: ModalColorMode) => void;
  onInspectedPartChange?: (partId: string | null) => void;
  onRegisterCapture?: (fn: (frames?: number, delayMs?: number) => Promise<string[]>) => void;
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

const MODAL_COMPLIANCE_SOURCES: Array<{
  key: "frame" | "belt" | "motor" | "bracket";
  short: string;
  color: string;
}> = [
  { key: "frame", short: "Struct", color: "#38bdf8" },
  { key: "belt", short: "Belt", color: "#f59e0b" },
  { key: "motor", short: "Motor", color: "#a78bfa" },
  { key: "bracket", short: "Bkt", color: "#fb7185" },
];

const CAUSE_CATEGORY_COLORS: Record<
  "belt" | "motor" | "bracket" | "structural" | "mass",
  string
> = {
  belt: "#f59e0b",
  structural: "#38bdf8",
  motor: "#a78bfa",
  bracket: "#fb7185",
  mass: "#22d3ee",
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
const _orientCadY = new THREE.Vector3(0, 0, -1);
const _orientU = new THREE.Vector3();
const _orientV = new THREE.Vector3();
const _orientMat4 = new THREE.Matrix4();

function orientBeamMesh(mesh: THREE.Mesh, p0: THREE.Vector3, p1: THREE.Vector3): void {
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
function orientBeamBoxMesh(mesh: THREE.Mesh, p0: THREE.Vector3, p1: THREE.Vector3): void {
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

interface ToolheadMatAppearance {
  hex: number;
  metalness: number;
  roughness: number;
  shortLabel: string;
}

function getToolheadMaterialAppearance(matId?: string | null): ToolheadMatAppearance {
  switch (matId) {
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

const PUNCHIN_FLEX_SOURCE_LABELS: Record<
  ToolheadFlexBreakdownData["dominant_flex_source"],
  string
> = {
  rail_xbeam: "X-beam & linear rail bending/torsion",
  carriage_body: "X-carriage backplate & toolhead body compliance",
  hotend_nozzle: "hotend mount & nozzle stem bending",
};

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
const _beamCadY = new THREE.Vector3(0, 0, -1);
const _beamRefAxis = new THREE.Vector3();
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

  // 4. Initial cross-section frame (U, refAxis, V) matching beam3d.py:
  //    For non-vertical beams (e.g. X-beam 2040/3060), height stands vertically along CAD +Z (Three.js +Y -> V)
  //    and width spans horizontally (U = refAxis × up). For vertical columns, U spans CAD X and V spans CAD Y.
  const refUpVec = Math.abs(_beamRefAxis.y) > 0.99 ? _beamCadY : _beamUp;
  _beamU.crossVectors(_beamRefAxis, refUpVec).normalize();
  _beamV.crossVectors(_beamU, _beamRefAxis).normalize();

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

interface ElasticDeformMesh {
  mesh: THREE.Mesh;
  geometry: THREE.BufferGeometry;
  restPos: Float32Array;
  restNorm: Float32Array | null;
  /** Precomputed per-vertex normalized flex coordinate s in [0, 1] */
  flexS: Float32Array;
  /** Stage 2 ("body") only: precomputed Stage-1 backplate flex coordinate at rear mounting interface */
  bpFlexS?: Float32Array;
  /** Stage 3 ("hotend") only: precomputed curvature-integrated slope & deflection fractions */
  hotendSlope?: Float32Array;
  hotendDisp?: Float32Array;
  stage: "backplate" | "body" | "hotend";
}

const HOTEND_LUT_N = 64;
const HOTEND_PROFILE_CACHE: Record<string, { slope: Float32Array; disp: Float32Array }> = {};

function getHotendProfileLUT(hotendMountStyle: string): { slope: Float32Array; disp: Float32Array } {
  const key = hotendMountStyle === "chube_6bolt_m3_18pcd" ? "chube" : "necked";
  const cached = HOTEND_PROFILE_CACHE[key];
  if (cached) return cached;

  const slope = new Float32Array(HOTEND_LUT_N + 1);
  const disp = new Float32Array(HOTEND_LUT_N + 1);

  if (key === "chube") {
    // Uniform continuous 25.4 mm satin-titanium structural tube (exact Euler-Bernoulli cantilever)
    for (let k = 0; k <= HOTEND_LUT_N; k++) {
      const s = k / HOTEND_LUT_N;
      slope[k] = s * (2 - s);
      disp[k] = (s * s * (3 - s)) / 3;
    }
  } else {
    // Piece-wise stiffness hotend (Dragon/Goliath 4x Ti standoffs, V6 slender heatbreak neck, Revo stem):
    // Clamped square in collar (s=0), stiff heatsink (s < 0.16), peak curvature across
    // standoffs / heatbreak neck (s in [0.18, 0.52]), rigid heater block + nozzle (s > 0.56).
    const kappa = new Float32Array(HOTEND_LUT_N + 1);
    for (let k = 0; k <= HOTEND_LUT_N; k++) {
      const s = k / HOTEND_LUT_N;
      const neckBump = Math.exp(-Math.pow((s - 0.35) / 0.155, 2));
      const sinkTerm = s < 0.24 ? 0.16 * (1 - s / 0.24) : 0;
      const blockTerm = 0.05;
      kappa[k] = (blockTerm + sinkTerm + 1.0 * neckBump) * (1 - 0.45 * s);
    }
    const ds = 1 / HOTEND_LUT_N;
    slope[0] = 0;
    for (let k = 1; k <= HOTEND_LUT_N; k++) {
      slope[k] = slope[k - 1] + 0.5 * (kappa[k - 1] + kappa[k]) * ds;
    }
    const totalSlope = Math.max(slope[HOTEND_LUT_N], 1e-6);
    for (let k = 0; k <= HOTEND_LUT_N; k++) {
      slope[k] /= totalSlope;
    }
    disp[0] = 0;
    for (let k = 1; k <= HOTEND_LUT_N; k++) {
      disp[k] = disp[k - 1] + 0.5 * (slope[k - 1] + slope[k]) * ds;
    }
  }

  const out = { slope, disp };
  HOTEND_PROFILE_CACHE[key] = out;
  return out;
}

function sampleProfileLUT(lut: Float32Array, s: number): number {
  const clamped = s <= 0 ? 0 : s >= 1 ? 1 : s;
  const f = clamped * HOTEND_LUT_N;
  const i0 = Math.floor(f);
  const i1 = i0 < HOTEND_LUT_N ? i0 + 1 : HOTEND_LUT_N;
  const frac = f - i0;
  return lut[i0] + (lut[i1] - lut[i0]) * frac;
}

function extractRotVecFromQuat(
  q: THREE.Quaternion,
  maxRad: number,
  out: THREE.Vector3
): THREE.Vector3 {
  // Ensure shortest-arc hemisphere (w >= 0)
  const sign = q.w < 0 ? -1 : 1;
  const w = Math.min(1, Math.max(-1, q.w * sign));
  const halfAngle = Math.acos(w);
  const sinHalf = Math.sqrt(Math.max(0, 1 - w * w));
  const angle = Math.min(2 * halfAngle, maxRad);
  if (sinHalf > 1e-6 && angle > 1e-6) {
    const scale = (angle * sign) / sinHalf;
    out.set(q.x * scale, q.y * scale, q.z * scale);
  } else {
    out.set(0, 0, 0);
  }
  return out;
}

function deformBackplateMesh(
  item: ElasticDeformMesh,
  pivotX: number,
  pivotY: number,
  pivotZ: number,
  tpx: number,
  tpy: number,
  tpz: number,
  spx: number,
  spy: number,
  spz: number
): void {
  const posAttr = item.geometry.getAttribute("position") as THREE.BufferAttribute;
  const posArr = posAttr.array as Float32Array;
  const normAttr = item.restNorm
    ? (item.geometry.getAttribute("normal") as THREE.BufferAttribute | undefined)
    : undefined;
  const normArr = normAttr ? (normAttr.array as Float32Array) : null;
  const restPos = item.restPos;
  const restNorm = item.restNorm;
  const flexS = item.flexS;
  const count = flexS.length;

  for (let i = 0; i < count; i++) {
    const o = i * 3;
    const x0 = restPos[o];
    const y0 = restPos[o + 1];
    const z0 = restPos[o + 2];
    const s = flexS[i];
    if (s <= 1e-6) {
      posArr[o] = x0;
      posArr[o + 1] = y0;
      posArr[o + 2] = z0;
      if (normArr && restNorm) {
        normArr[o] = restNorm[o];
        normArr[o + 1] = restNorm[o + 1];
        normArr[o + 2] = restNorm[o + 2];
      }
      continue;
    }

    const w = 0.5 * s * s * (3 - s);
    const ax = tpx * w;
    const ay = tpy * w;
    const az = tpz * w;
    const vx = x0 - pivotX;
    const vy = y0 - pivotY;
    const vz = z0 - pivotZ;

    const angSq = ax * ax + ay * ay + az * az;
    if (angSq > 1e-12) {
      const ang = Math.sqrt(angSq);
      const invAng = 1 / ang;
      const kx = ax * invAng;
      const ky = ay * invAng;
      const kz = az * invAng;
      const cosA = Math.cos(ang);
      const sinA = Math.sin(ang);
      const omc = 1 - cosA;

      const kDotV = kx * vx + ky * vy + kz * vz;
      const cx = ky * vz - kz * vy;
      const cy = kz * vx - kx * vz;
      const cz = kx * vy - ky * vx;
      posArr[o] = pivotX + vx * cosA + cx * sinA + kx * kDotV * omc + spx * w;
      posArr[o + 1] = pivotY + vy * cosA + cy * sinA + ky * kDotV * omc + spy * w;
      posArr[o + 2] = pivotZ + vz * cosA + cz * sinA + kz * kDotV * omc + spz * w;

      if (normArr && restNorm) {
        const nx = restNorm[o];
        const ny = restNorm[o + 1];
        const nz = restNorm[o + 2];
        const kDotN = kx * nx + ky * ny + kz * nz;
        const ncx = ky * nz - kz * ny;
        const ncy = kz * nx - kx * nz;
        const ncz = kx * ny - ky * nx;
        normArr[o] = nx * cosA + ncx * sinA + kx * kDotN * omc;
        normArr[o + 1] = ny * cosA + ncy * sinA + ky * kDotN * omc;
        normArr[o + 2] = nz * cosA + ncz * sinA + kz * kDotN * omc;
      }
    } else {
      posArr[o] = x0 + spx * w;
      posArr[o + 1] = y0 + spy * w;
      posArr[o + 2] = z0 + spz * w;
      if (normArr && restNorm) {
        normArr[o] = restNorm[o];
        normArr[o + 1] = restNorm[o + 1];
        normArr[o + 2] = restNorm[o + 2];
      }
    }
  }
  posAttr.needsUpdate = true;
  if (normAttr) normAttr.needsUpdate = true;
}

function deformBodyMesh(
  item: ElasticDeformMesh,
  pivotX: number,
  pivotY: number,
  pivotZ: number,
  bodyRootX: number,
  bodyRootY: number,
  bodyRootZ: number,
  collarRelX: number,
  collarRelY: number,
  collarRelZ: number,
  tpx: number,
  tpy: number,
  tpz: number,
  spx: number,
  spy: number,
  spz: number,
  tbx: number,
  tby: number,
  tbz: number,
  sbx: number,
  sby: number,
  sbz: number
): void {
  const posAttr = item.geometry.getAttribute("position") as THREE.BufferAttribute;
  const posArr = posAttr.array as Float32Array;
  const normAttr = item.restNorm
    ? (item.geometry.getAttribute("normal") as THREE.BufferAttribute | undefined)
    : undefined;
  const normArr = normAttr ? (normAttr.array as Float32Array) : null;
  const restPos = item.restPos;
  const restNorm = item.restNorm;
  const flexS = item.flexS;
  const bpFlexS = item.bpFlexS;
  const count = flexS.length;

  for (let i = 0; i < count; i++) {
    const o = i * 3;
    let qx = restPos[o];
    let qy = restPos[o + 1];
    let qz = restPos[o + 2];
    let nx = restNorm ? restNorm[o] : 0;
    let ny = restNorm ? restNorm[o + 1] : 1;
    let nz = restNorm ? restNorm[o + 2] : 0;

    // Step 1: Stage-2 elastic body flex relative to rear bolted face (s2 = 0 at rear face)
    const s2 = flexS[i];
    if (s2 > 1e-6) {
      const w2 = 0.5 * s2 * s2 * (3 - s2);
      const ax2 = tbx * w2;
      const ay2 = tby * w2;
      const az2 = tbz * w2;
      const vx2 = qx - bodyRootX;
      const vy2 = qy - bodyRootY;
      const vz2 = qz - bodyRootZ;
      const angSq2 = ax2 * ax2 + ay2 * ay2 + az2 * az2;
      if (angSq2 > 1e-12) {
        const ang2 = Math.sqrt(angSq2);
        const inv2 = 1 / ang2;
        const kx2 = ax2 * inv2;
        const ky2 = ay2 * inv2;
        const kz2 = az2 * inv2;
        const cos2 = Math.cos(ang2);
        const sin2 = Math.sin(ang2);
        const omc2 = 1 - cos2;
        const kDotV2 = kx2 * vx2 + ky2 * vy2 + kz2 * vz2;
        const cx2 = ky2 * vz2 - kz2 * vy2;
        const cy2 = kz2 * vx2 - kx2 * vz2;
        const cz2 = kx2 * vy2 - ky2 * vx2;
        qx = bodyRootX + vx2 * cos2 + cx2 * sin2 + kx2 * kDotV2 * omc2 + sbx * w2;
        qy = bodyRootY + vy2 * cos2 + cy2 * sin2 + ky2 * kDotV2 * omc2 + sby * w2;
        qz = bodyRootZ + vz2 * cos2 + cz2 * sin2 + kz2 * kDotV2 * omc2 + sbz * w2;

        if (normArr && restNorm) {
          const kDotN2 = kx2 * nx + ky2 * ny + kz2 * nz;
          const ncx2 = ky2 * nz - kz2 * ny;
          const ncy2 = kz2 * nx - kx2 * nz;
          const ncz2 = kx2 * ny - ky2 * nx;
          const nnx = nx * cos2 + ncx2 * sin2 + kx2 * kDotN2 * omc2;
          const nny = ny * cos2 + ncy2 * sin2 + ky2 * kDotN2 * omc2;
          const nnz = nz * cos2 + ncz2 * sin2 + kz2 * kDotN2 * omc2;
          nx = nnx;
          ny = nny;
          nz = nnz;
        }
      } else {
        qx += sbx * w2;
        qy += sby * w2;
        qz += sbz * w2;
      }
    }

    // Step 2: Stage-1 backplate continuous deformation (bit-for-bit identical at rear interface!)
    const s1 = bpFlexS ? bpFlexS[i] : 1;
    if (s1 > 1e-6) {
      const w1 = 0.5 * s1 * s1 * (3 - s1);
      const ax1 = tpx * w1;
      const ay1 = tpy * w1;
      const az1 = tpz * w1;
      const vx1 = qx + collarRelX - pivotX;
      const vy1 = qy + collarRelY - pivotY;
      const vz1 = qz + collarRelZ - pivotZ;
      const angSq1 = ax1 * ax1 + ay1 * ay1 + az1 * az1;
      if (angSq1 > 1e-12) {
        const ang1 = Math.sqrt(angSq1);
        const inv1 = 1 / ang1;
        const kx1 = ax1 * inv1;
        const ky1 = ay1 * inv1;
        const kz1 = az1 * inv1;
        const cos1 = Math.cos(ang1);
        const sin1 = Math.sin(ang1);
        const omc1 = 1 - cos1;
        const kDotV1 = kx1 * vx1 + ky1 * vy1 + kz1 * vz1;
        const cx1 = ky1 * vz1 - kz1 * vy1;
        const cy1 = kz1 * vx1 - kx1 * vz1;
        const cz1 = kx1 * vy1 - ky1 * vx1;
        qx = pivotX + vx1 * cos1 + cx1 * sin1 + kx1 * kDotV1 * omc1 + spx * w1 - collarRelX;
        qy = pivotY + vy1 * cos1 + cy1 * sin1 + ky1 * kDotV1 * omc1 + spy * w1 - collarRelY;
        qz = pivotZ + vz1 * cos1 + cz1 * sin1 + kz1 * kDotV1 * omc1 + spz * w1 - collarRelZ;

        if (normArr && restNorm) {
          const kDotN1 = kx1 * nx + ky1 * ny + kz1 * nz;
          const ncx1 = ky1 * nz - kz1 * ny;
          const ncy1 = kz1 * nx - kx1 * nz;
          const ncz1 = kx1 * ny - ky1 * nx;
          const nnx = nx * cos1 + ncx1 * sin1 + kx1 * kDotN1 * omc1;
          const nny = ny * cos1 + ncy1 * sin1 + ky1 * kDotN1 * omc1;
          const nnz = nz * cos1 + ncz1 * sin1 + kz1 * kDotN1 * omc1;
          nx = nnx;
          ny = nny;
          nz = nnz;
        }
      } else {
        qx += spx * w1;
        qy += spy * w1;
        qz += spz * w1;
      }
    }

    posArr[o] = qx;
    posArr[o + 1] = qy;
    posArr[o + 2] = qz;
    if (normArr && restNorm) {
      normArr[o] = nx;
      normArr[o + 1] = ny;
      normArr[o + 2] = nz;
    }
  }

  posAttr.needsUpdate = true;
  if (normAttr) normAttr.needsUpdate = true;
}

function deformHotendMesh(
  item: ElasticDeformMesh,
  hotendLen: number,
  bx: number,
  by: number,
  bz: number
): void {
  const posAttr = item.geometry.getAttribute("position") as THREE.BufferAttribute;
  const posArr = posAttr.array as Float32Array;
  const normAttr = item.restNorm
    ? (item.geometry.getAttribute("normal") as THREE.BufferAttribute | undefined)
    : undefined;
  const normArr = normAttr ? (normAttr.array as Float32Array) : null;
  const restPos = item.restPos;
  const restNorm = item.restNorm;
  const hotendSlope = item.hotendSlope;
  const hotendDisp = item.hotendDisp;
  const count = item.flexS.length;
  const safeLen = Math.max(hotendLen, 0.1);
  const invSafeLen = 0.5 / safeLen;

  if (!hotendSlope || !hotendDisp) return;

  for (let i = 0; i < count; i++) {
    const o = i * 3;
    const x0 = restPos[o];
    const y0 = restPos[o + 1];
    const z0 = restPos[o + 2];

    const sl = hotendSlope[i];
    const dp = hotendDisp[i];
    if (sl <= 1e-6 && dp <= 1e-6) {
      posArr[o] = x0;
      posArr[o + 1] = y0;
      posArr[o + 2] = z0;
      if (normArr && restNorm) {
        normArr[o] = restNorm[o];
        normArr[o + 1] = restNorm[o + 1];
        normArr[o + 2] = restNorm[o + 2];
      }
      continue;
    }

    const ax = bx * sl;
    const ay = by * sl;
    const az = bz * sl;

    // Exact Euler-Bernoulli cantilever centerline deflection (downward -Y beam)
    const dxC = bz * hotendLen * dp;
    const dzC = -bx * hotendLen * dp;
    // Exact geometric arc-length foreshortening (pulls slightly +Y toward collar so rod never stretches)
    const dyC = (dxC * dxC + dzC * dzC) * invSafeLen;

    // Exact orthonormal cross-section rotation Ry(ay) -> Rz(az) -> Rx(ax) on radial (x0, 0, z0)
    const cy = Math.cos(ay);
    const sy = Math.sin(ay);
    const cz = Math.cos(az);
    const sz = Math.sin(az);
    const cx = Math.cos(ax);
    const sx = Math.sin(ax);

    const x1 = x0 * cy + z0 * sy;
    const z1 = -x0 * sy + z0 * cy;
    const rx = x1 * cz;
    const ry = x1 * sz * cx - z1 * sx;
    const rz = x1 * sz * sx + z1 * cx;

    posArr[o] = dxC + rx;
    posArr[o + 1] = y0 + dyC + ry;
    posArr[o + 2] = dzC + rz;

    if (normArr && restNorm) {
      const nx0 = restNorm[o];
      const ny0 = restNorm[o + 1];
      const nz0 = restNorm[o + 2];
      // Apply exact same Ry(ay) -> Rz(az) -> Rx(ax) to rest normal
      const nx1 = nx0 * cy + nz0 * sy;
      const nz1 = -nx0 * sy + nz0 * cy;
      const nx2 = nx1 * cz - ny0 * sz;
      const ny2 = nx1 * sz + ny0 * cz;
      normArr[o] = nx2;
      normArr[o + 1] = ny2 * cx - nz1 * sx;
      normArr[o + 2] = ny2 * sx + nz1 * cx;
    }
  }

  posAttr.needsUpdate = true;
  if (normAttr) normAttr.needsUpdate = true;
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
  toolheadPunchIn: toolheadPunchInProp,
  onToggleToolheadPunchIn,
  sensitivityLens: sensitivityLensProp,
  onSensitivityLensChange,
  modalColorMode: modalColorModeProp,
  onModalColorModeChange,
  onInspectedPartChange,
  onRegisterCapture,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [showCarvedGhost, setShowCarvedGhost] = useState<boolean>(true);
  const [internalAdjointLens, setInternalAdjointLens] = useState<AdjointLens>("print_impact");
  const [internalModalColorMode, setInternalModalColorMode] = useState<ModalColorMode>("relative");
  const [internalShowBedWarp, setInternalShowBedWarp] = useState<boolean>(false);
  const [internalToolheadPunchIn, setInternalToolheadPunchIn] = useState<boolean>(false);
  const [isolateToolheadModal, setIsolateToolheadModal] = useState<boolean>(false);
  const [modalAmpScale, setModalAmpScale] = useState<number>(1.0);
  const [ecoDpr, setEcoDpr] = useState<boolean>(false);
  const [modalPaused, setModalPaused] = useState<boolean>(false);
  const [showHudDetails, setShowHudDetails] = useState<boolean>(false);
  const [hoveredPart, setHoveredPart] = useState<InspectedPartInfo | null>(null);
  const [pinnedPart, setPinnedPart] = useState<InspectedPartInfo | null>(null);

  const adjointLens = sensitivityLensProp ?? internalAdjointLens;
  const setAdjointLens = (next: AdjointLens) => {
    if (onSensitivityLensChange) {
      onSensitivityLensChange(next);
    } else {
      setInternalAdjointLens(next);
    }
  };

  const modalColorMode = modalColorModeProp ?? internalModalColorMode;
  const setModalColorMode = (next: ModalColorMode) => {
    if (onModalColorModeChange) {
      onModalColorModeChange(next);
    } else {
      setInternalModalColorMode(next);
    }
  };

  const showBedWarp = showBedWarpProp ?? internalShowBedWarp;
  const setShowBedWarp = (next: boolean) => {
    if (onToggleBedWarp) {
      onToggleBedWarp(next);
    } else {
      setInternalShowBedWarp(next);
    }
  };

  const canPunchIn =
    Boolean(sim) &&
    (viewMode === "assembly" ||
      viewMode === "fea_stress" ||
      viewMode === "modal" ||
      viewMode === "sensitivity");
  const rawPunchIn = toolheadPunchInProp ?? internalToolheadPunchIn;
  const toolheadPunchIn = canPunchIn && rawPunchIn;
  const isToolheadIsolated = toolheadPunchIn && viewMode === "modal" && isolateToolheadModal;
  const setToolheadPunchIn = (next: boolean) => {
    if (onToggleToolheadPunchIn) {
      onToggleToolheadPunchIn(next);
    } else {
      setInternalToolheadPunchIn(next);
    }
  };

  const savedCameraRef = useRef<{ pos: THREE.Vector3; target: THREE.Vector3 } | null>(null);
  const cameraAnimRef = useRef<{
    targetPos: THREE.Vector3;
    targetLook: THREE.Vector3;
    active: boolean;
  }>({
    targetPos: new THREE.Vector3(),
    targetLook: new THREE.Vector3(),
    active: false,
  });
  const prevPunchInRef = useRef<boolean>(false);
  const lastPunchTargetRef = useRef<THREE.Vector3 | null>(null);

  const onRegisterCaptureRef = useRef(onRegisterCapture);
  onRegisterCaptureRef.current = onRegisterCapture;
  const captureFnRef = useRef<((frames?: number, delayMs?: number) => Promise<string[]>) | null>(null);
  const requestRenderRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    if (onRegisterCapture && captureFnRef.current) {
      onRegisterCapture(captureFnRef.current);
    }
  }, [onRegisterCapture]);

  useEffect(() => {
    onInspectedPartChange?.(pinnedPart ? pinnedPart.partGroupId || pinnedPart.key : null);
  }, [pinnedPart, onInspectedPartChange]);

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
    modalRailShoe?: THREE.Group;
    modalToolheadNodeId?: string;
    modalBackplateGroup?: THREE.Group;
    modalCogMarker?: THREE.Group;
    modalCogNodeId?: string;
    modalCogParentNodeId?: string;
    modalCogOffset: THREE.Vector3;
    modalNozzleMarker?: THREE.Group;
    modalNozzleNodeId?: string;
    modalNozzleRefNodeId?: string;
    modalNozzleOffset: THREE.Vector3;
    modalNozzleBaseDir: THREE.Vector3;
    modalNozzleDispLine?: THREE.Line;
    modalElasticMeshes: ElasticDeformMesh[];
    modalRailFlexFrac: number;
    modalPlateFlexFrac: number;
    modalBodyFlexFrac: number;
    modalCollarBpW: number;
    modalHotendStyle: string;
    modalBackplatePivotLocal: THREE.Vector3;
    modalBodyRootLocal: THREE.Vector3;
    modalCollarRelToTh: THREE.Vector3;
    modalBodyDepthRef: number;
    modalBedMesh?: THREE.Mesh;
    modalBedNodeId?: string;
    modalBedOffset: THREE.Vector3;
    basePositions: Record<string, THREE.Vector3>;
    modalVectors: Record<string, THREE.Vector3>;
    modalRotations: Record<string, THREE.Vector3>;
    modalHotendLen: number;
    isModal: boolean;
    isolateToolheadModal: boolean;
    modalAmpScale: number;
    isToolheadPunchIn: boolean;
    ecoDpr: boolean;
    modalPaused: boolean;
  }>({
    inspectableMeshes: [],
    hoveredMesh: null,
    pinnedMesh: null,
    pinnedKey: null,
    hoveredKey: null,
    modalMeshes: [],
    modalContinuousBeams: [],
    modalShellMeshes: [],
    modalCogOffset: new THREE.Vector3(),
    modalNozzleOffset: new THREE.Vector3(),
    modalNozzleBaseDir: new THREE.Vector3(0, 1, 0),
    modalElasticMeshes: [],
    modalRailFlexFrac: 0,
    modalPlateFlexFrac: 0.5,
    modalBodyFlexFrac: 0.5,
    modalCollarBpW: 1.0,
    modalHotendStyle: "dragon_4bolt_m25",
    modalBackplatePivotLocal: new THREE.Vector3(),
    modalBodyRootLocal: new THREE.Vector3(),
    modalCollarRelToTh: new THREE.Vector3(),
    modalBodyDepthRef: 0.22,
    modalBedOffset: new THREE.Vector3(0, -0.04, 0),
    basePositions: {},
    modalVectors: {},
    modalRotations: {},
    modalHotendLen: 0.45,
    isModal: false,
    isolateToolheadModal: false,
    modalAmpScale: 1.0,
    isToolheadPunchIn: false,
    ecoDpr: false,
    modalPaused: false,
  });

  useEffect(() => {
    stateRef.current.isolateToolheadModal = isToolheadIsolated;
    requestRenderRef.current?.();
  }, [isToolheadIsolated]);

  useEffect(() => {
    stateRef.current.modalAmpScale = modalAmpScale;
    requestRenderRef.current?.();
  }, [modalAmpScale]);

  useEffect(() => {
    stateRef.current.isToolheadPunchIn = toolheadPunchIn;
    requestRenderRef.current?.();
  }, [toolheadPunchIn]);

  useEffect(() => {
    stateRef.current.modalPaused = modalPaused;
    requestRenderRef.current?.();
  }, [modalPaused]);

  useEffect(() => {
    stateRef.current.ecoDpr = ecoDpr;
    const renderer = stateRef.current.renderer;
    const el = containerRef.current;
    if (renderer && el) {
      const nextDpr = ecoDpr ? 1 : Math.min(window.devicePixelRatio || 1, 1.5);
      renderer.setPixelRatio(nextDpr);
      if (el.clientWidth > 0 && el.clientHeight > 0) {
        renderer.setSize(el.clientWidth, el.clientHeight);
      }
      requestRenderRef.current?.();
    }
  }, [ecoDpr]);

  const handleCloseInspector = () => {
    if (stateRef.current.pinnedMesh) {
      const prevPinned = stateRef.current.pinnedMesh;
      stateRef.current.pinnedMesh = null;
      stateRef.current.pinnedKey = null;
      applyInspectHighlight(
        prevPinned,
        stateRef.current.hoveredMesh === prevPinned ? "hover" : "none"
      );
      requestRenderRef.current?.();
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
    const viewportBgColor = new THREE.Color("#0d1422");
    scene.background = viewportBgColor;

    const initW = Math.max(el.clientWidth, 1);
    const initH = Math.max(el.clientHeight, 1);
    const camera = new THREE.PerspectiveCamera(42, initW / initH, 0.05, 200);
    camera.position.set(7.5, 6.2, 8.2);

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false,
      powerPreference: "low-power",
    });
    renderer.setClearColor(viewportBgColor, 1);
    renderer.setPixelRatio(
      stateRef.current.ecoDpr ? 1 : Math.min(window.devicePixelRatio || 1, 1.5)
    );
    renderer.setSize(initW, initH);
    el.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(0, 2.3, 0);

    // Studio CAD Lighting (key + hemisphere + warm front-lower fill + cool rim for crisp metallic speculars)
    scene.add(new THREE.AmbientLight(0xffffff, 0.48));
    scene.add(new THREE.HemisphereLight(0xf1f5f9, 0x1e293b, 0.52));
    const keyLight = new THREE.DirectionalLight(0xffffff, 1.15);
    keyLight.position.set(8, 14, 10);
    scene.add(keyLight);
    const frontFillLight = new THREE.DirectionalLight(0xfff5e8, 0.55);
    frontFillLight.position.set(-5, 2.5, 12);
    scene.add(frontFillLight);
    const rimLight = new THREE.DirectionalLight(0x38bdf8, 0.5);
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

    let disposed = false;
    let rafId: number | null = null;
    let modalTimerId: number | null = null;
    let userOrbiting = false;
    let needsDamping = false;
    let inInternalControlsUpdate = false;
    let settledFrames = 0;
    let dampFrames = 0;
    let modalPhaseSec = 0;
    let lastModalAdvanceMs: number | null = null;
    let prevCamAnimMs: number | null = null;
    const MODAL_FRAME_INTERVAL_MS = 33; // ~30 FPS modal animation cadence
    const prevCamPos = new THREE.Vector3();
    const prevCtrlTarget = new THREE.Vector3();

    const captureBgColor = new THREE.Color("#0e1522");
    const tmpPosA = new THREE.Vector3();
    const tmpPosB = new THREE.Vector3();
    const tmpNozV1 = new THREE.Vector3();
    const tmpCogV0 = new THREE.Vector3();
    const tmpCogV1 = new THREE.Vector3();
    const tmpStage1Quat = new THREE.Quaternion();
    const tmpIdentQuat = new THREE.Quaternion();
    const tmpRotCogVec = new THREE.Vector3();
    const tmpRelRotVec = new THREE.Vector3();
    const tmpTotalRotVec = new THREE.Vector3();
    const tmpRotAxis = new THREE.Vector3();
    const tmpHotendQuat = new THREE.Quaternion();
    const tmpHotendTopVec = new THREE.Vector3();
    const tmpIsoRelCogV = new THREE.Vector3();
    const tmpIsoRelNozV = new THREE.Vector3();
    const tmpIsoBaseRot = new THREE.Vector3();
    const tmpIsoRelCogRot = new THREE.Vector3();
    const tmpIsoRelNozRot = new THREE.Vector3();
    const tmpCurTh = new THREE.Vector3();
    const tmpUStage12 = new THREE.Vector3();
    const tmpThetaRail = new THREE.Vector3();
    const tmpThetaBp = new THREE.Vector3();
    const tmpThetaBpCollar = new THREE.Vector3();
    const tmpThetaBody = new THREE.Vector3();
    const tmpRailQuat = new THREE.Quaternion();
    const tmpInvRailQuat = new THREE.Quaternion();
    const tmpBpQuat = new THREE.Quaternion();
    const tmpBpCollarQuat = new THREE.Quaternion();
    const tmpBodyQuat = new THREE.Quaternion();
    const tmpCollarWorldQuat = new THREE.Quaternion();
    const tmpCollarWorldPos = new THREE.Vector3();
    const tmpInvCollarQuat = new THREE.Quaternion();
    const tmpThetaBendLocal = new THREE.Vector3();
    const tmpUFromRot = new THREE.Vector3();
    const tmpUShearTotal = new THREE.Vector3();
    const tmpUShearLocal = new THREE.Vector3();
    const tmpUBpShear = new THREE.Vector3();
    const tmpUBodyShear = new THREE.Vector3();
    const tmpCollarS2 = new THREE.Vector3();
    const tmpNozTipWorld = new THREE.Vector3();
    const tmpRelQuat = new THREE.Quaternion();

    const hasActiveModalMeshes = (): boolean =>
      stateRef.current.isModal &&
      (stateRef.current.modalContinuousBeams.length > 0 ||
        stateRef.current.modalMeshes.length > 0 ||
        stateRef.current.modalShellMeshes.length > 0);

    const isModalAnimating = (): boolean =>
      hasActiveModalMeshes() && !stateRef.current.modalPaused && !document.hidden;

    const clearModalTimer = () => {
      if (modalTimerId !== null) {
        window.clearTimeout(modalTimerId);
        modalTimerId = null;
      }
    };

    const clearScheduledRaf = () => {
      if (rafId !== null) {
        cancelAnimationFrame(rafId);
        rafId = null;
      }
      stateRef.current.animFrame = undefined;
    };

    const renderFrame = (t: number, stepCamera: boolean) => {
      if (disposed) return;

      if (stepCamera) {
        prevCamPos.copy(camera.position);
        prevCtrlTarget.copy(controls.target);

        if (cameraAnimRef.current.active) {
          const nowMs = performance.now();
          const dt =
            prevCamAnimMs !== null
              ? Math.min(Math.max((nowMs - prevCamAnimMs) / 1000, 0.001), 0.1)
              : 0.016;
          prevCamAnimMs = nowMs;
          const alpha = 1 - Math.exp(-10 * dt);
          camera.position.lerp(cameraAnimRef.current.targetPos, alpha);
          controls.target.lerp(cameraAnimRef.current.targetLook, alpha);
          if (
            camera.position.distanceToSquared(cameraAnimRef.current.targetPos) < 1e-4 &&
            controls.target.distanceToSquared(cameraAnimRef.current.targetLook) < 1e-4
          ) {
            camera.position.copy(cameraAnimRef.current.targetPos);
            controls.target.copy(cameraAnimRef.current.targetLook);
            cameraAnimRef.current.active = false;
            prevCamAnimMs = null;
          }
        }

        let controlsUpdated = false;
        inInternalControlsUpdate = true;
        try {
          controlsUpdated = controls.update();
        } finally {
          inInternalControlsUpdate = false;
        }

        if (needsDamping) {
          if (userOrbiting) {
            settledFrames = 0;
            dampFrames = 0;
          } else {
            dampFrames += 1;
            const camDeltaSq =
              prevCamPos.distanceToSquared(camera.position) +
              prevCtrlTarget.distanceToSquared(controls.target);
            const stillMoving = controlsUpdated && camDeltaSq >= 1e-8;
            if (stillMoving && dampFrames < 240) {
              settledFrames = 0;
            } else {
              settledFrames += 1;
              if (settledFrames >= 2 || dampFrames >= 240) {
                needsDamping = false;
                settledFrames = 0;
                dampFrames = 0;
              }
            }
          }
        }
      }

      if (
        stateRef.current.isModal &&
        (stateRef.current.modalContinuousBeams.length > 0 ||
          stateRef.current.modalMeshes.length > 0 ||
          stateRef.current.modalShellMeshes.length > 0)
      ) {
        const {
          basePositions,
          modalVectors,
          modalRotations,
          modalHotendLen,
          modalContinuousBeams,
          modalShellMeshes,
          modalMeshes,
          modalToolhead,
          modalRailShoe,
          modalToolheadNodeId,
          modalBackplateGroup,
          modalCogMarker,
          modalCogNodeId,
          modalCogParentNodeId,
          modalCogOffset,
          modalNozzleMarker,
          modalNozzleNodeId,
          modalNozzleRefNodeId,
          modalNozzleOffset,
          modalNozzleBaseDir,
          modalNozzleDispLine,
          modalElasticMeshes,
          modalRailFlexFrac,
          modalPlateFlexFrac,
          modalBodyFlexFrac,
          modalCollarBpW,
          modalHotendStyle,
          modalBackplatePivotLocal,
          modalBodyRootLocal,
          modalCollarRelToTh,
          modalBedMesh,
          modalBedNodeId,
          modalBedOffset,
          isolateToolheadModal: isIsoToolhead,
          modalAmpScale: ampScale,
          isToolheadPunchIn: isPunchIn,
        } = stateRef.current;

        // Auto-calibrated modal stroke: 28 mm full view / 14 mm macro punch-in at 1.0x
        const baseStroke = (isPunchIn ? 0.14 : 0.28) * ampScale;
        const phase = Math.sin(t * 7.5) * baseStroke;
        const maxStageAngle = 0.22 * Math.min(ampScale, 1.5);

        const thId = modalToolheadNodeId || "toolhead";
        const thModalVec = isIsoToolhead ? modalVectors[thId] : undefined;
        const thModalRot = isIsoToolhead ? modalRotations[thId] : undefined;

        // When Toolhead Isolation is enabled, freeze the frame, gantry & MGN rail stationary
        // and subtract carriage translation/rotation so ONLY rail-roll + backplate + body + hotend flex animates.
        let isoGain = 1.0;
        if (isIsoToolhead) {
          tmpIsoRelCogV.set(0, 0, 0);
          tmpIsoRelNozV.set(0, 0, 0);
          if (modalCogNodeId && modalVectors[modalCogNodeId]) {
            tmpIsoRelCogV.copy(modalVectors[modalCogNodeId]);
            if (thModalVec) tmpIsoRelCogV.sub(thModalVec);
          }
          if (modalNozzleNodeId && modalVectors[modalNozzleNodeId]) {
            tmpIsoRelNozV.copy(modalVectors[modalNozzleNodeId]);
            if (thModalVec) tmpIsoRelNozV.sub(thModalVec);
          }
          tmpIsoBaseRot.set(0, 0, 0);
          if (thModalRot) tmpIsoBaseRot.copy(thModalRot);
          tmpIsoRelCogRot.set(0, 0, 0);
          if (modalCogNodeId && modalRotations[modalCogNodeId]) {
            tmpIsoRelCogRot.copy(modalRotations[modalCogNodeId]).sub(tmpIsoBaseRot);
          }
          tmpIsoRelNozRot.set(0, 0, 0);
          if (modalNozzleNodeId && modalRotations[modalNozzleNodeId]) {
            tmpIsoRelNozRot.copy(modalRotations[modalNozzleNodeId]).sub(tmpIsoBaseRot);
          }
          const relDispMag = Math.max(tmpIsoRelCogV.length(), tmpIsoRelNozV.length());
          const relRotMag = Math.max(tmpIsoRelCogRot.length(), tmpIsoRelNozRot.length());
          const effectiveRel = Math.max(relDispMag, relRotMag * 0.04);
          // Keep near-zero toolhead flex modes (pure frame sway) visibly rock-solid;
          // gently scale moderate mixed modes to a crisp, restrained local stroke.
          if (effectiveRel > 0.05) {
            isoGain = Math.min(0.55 / Math.max(effectiveRel, 0.25), 1.8);
          }
        }
        const toolPhase = isIsoToolhead ? phase * isoGain : phase;
        const toolPhaseMeters = toolPhase * 0.1;

        const getPosInto = (id: string, out: THREE.Vector3): THREE.Vector3 => {
          const b = basePositions[id];
          const v = modalVectors[id];
          if (!b) return out.set(0, 0, 0);
          out.copy(b);
          if (isIsoToolhead) {
            if (id === modalCogNodeId || id === modalNozzleNodeId) {
              if (v) out.addScaledVector(v, toolPhase);
              if (thModalVec) out.addScaledVector(thModalVec, -toolPhase);
            }
            return out;
          }
          if (v) out.addScaledVector(v, phase);
          return out;
        };

        const setQuatFromClampedRotVec = (
          vec: THREE.Vector3,
          maxAngleRad: number,
          outQuat: THREE.Quaternion
        ): void => {
          const mag = vec.length();
          if (mag > maxAngleRad) {
            vec.multiplyScalar(maxAngleRad / mag);
          }
          const clampedMag = Math.min(mag, maxAngleRad);
          if (clampedMag > 1e-5) {
            tmpRotAxis.copy(vec).divideScalar(clampedMag);
            outQuat.setFromAxisAngle(tmpRotAxis, clampedMag);
          } else {
            outQuat.identity();
          }
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
        const curTh = getPosInto(thId, tmpCurTh);
        if (modalRailShoe) {
          modalRailShoe.position.copy(curTh);
          modalRailShoe.quaternion.identity();
        }
        if (modalToolhead) {
          modalToolhead.position.copy(curTh);
          modalToolhead.quaternion.identity();
        }

        // True decoupled 4-stage toolhead deformation:
        //   Stage 0: Rigid MGN rail-carriage bearing roll (modalToolhead + backplate/body base)
        //   Stage 1: X-carriage backplate elastic bending (only plateFrac in rail-local frame)
        //   Stage 2: Toolhead body elastic flexing (bodyFrac in front of backplate)
        //   Stage 3: Hotend Euler-Bernoulli cantilever bending
        tmpStage1Quat.identity();
        tmpRotCogVec.set(0, 0, 0);
        tmpUStage12.set(0, 0, 0);
        tmpRailQuat.identity();
        tmpInvRailQuat.identity();
        tmpBpQuat.identity();
        tmpBpCollarQuat.identity();
        tmpBodyQuat.identity();
        tmpCollarWorldQuat.identity();
        tmpCollarWorldPos.copy(curTh).add(modalCollarRelToTh);
        tmpUBpShear.set(0, 0, 0);
        tmpUBodyShear.set(0, 0, 0);

        const rawCogRot = modalCogNodeId ? modalRotations[modalCogNodeId] : undefined;
        const effCogRot = isIsoToolhead
          ? rawCogRot
            ? tmpIsoRelCogRot
            : undefined
          : rawCogRot;

        if (modalCogNodeId) {
          const curCog = getPosInto(modalCogNodeId, tmpPosA);
          const baseThPos = basePositions[thId];
          const baseCogPos = basePositions[modalCogNodeId];
          if (baseThPos && baseCogPos) {
            tmpUStage12
              .copy(curCog)
              .sub(baseCogPos)
              .sub(curTh)
              .add(baseThPos);
          }

          if (effCogRot && effCogRot.lengthSq() > 1e-8) {
            tmpRotCogVec.copy(effCogRot).multiplyScalar(toolPhaseMeters);
            setQuatFromClampedRotVec(tmpRotCogVec, maxStageAngle, tmpStage1Quat);
          } else if (
            modalCogParentNodeId &&
            basePositions[modalCogParentNodeId] &&
            basePositions[modalCogNodeId]
          ) {
            tmpCogV0.subVectors(basePositions[modalCogNodeId], basePositions[modalCogParentNodeId]);
            const curParent = getPosInto(modalCogParentNodeId, tmpPosB);
            tmpCogV1.subVectors(curCog, curParent);
            if (tmpCogV0.lengthSq() > 1e-6 && tmpCogV1.lengthSq() > 1e-6) {
              tmpStage1Quat.setFromUnitVectors(tmpCogV0.normalize(), tmpCogV1.normalize());
              const qAngle = 2 * Math.acos(Math.min(1, Math.abs(tmpStage1Quat.w)));
              if (qAngle > maxStageAngle) {
                tmpStage1Quat.slerp(tmpIdentQuat, 1 - maxStageAngle / qAngle);
              }
              extractRotVecFromQuat(tmpStage1Quat, maxStageAngle, tmpRotCogVec);
            }
          }

          if (modalElasticMeshes.length > 0) {
            // Split total carriage-to-body compliance into:
            //   1) Rail bearing roll (rigid rotation of MGN carriage + backplate + body base)
            //   2) Backplate elastic bending (plateFrac; drops to ~2.5% for 5mm Al so plate stays dead flat!)
            //   3) Toolhead body elastic flexing (bodyFrac; ~68.5% for 4mm ABS body!)
            const railFrac = modalRailFlexFrac;
            const plateFrac = modalPlateFlexFrac;
            const bodyFrac = modalBodyFlexFrac;

            tmpThetaRail.copy(tmpRotCogVec).multiplyScalar(railFrac);
            setQuatFromClampedRotVec(tmpThetaRail, maxStageAngle, tmpRailQuat);
            tmpInvRailQuat.copy(tmpRailQuat).invert();

            // Express backplate & body bending vectors in rail-rotated local frame
            tmpThetaBp.copy(tmpRotCogVec).multiplyScalar(plateFrac).applyQuaternion(tmpInvRailQuat);
            tmpThetaBody.copy(tmpRotCogVec).multiplyScalar(bodyFrac).applyQuaternion(tmpInvRailQuat);
            setQuatFromClampedRotVec(tmpThetaBp, maxStageAngle, tmpBpQuat);
            setQuatFromClampedRotVec(tmpThetaBody, maxStageAngle, tmpBodyQuat);

            // Exact backplate bending evaluated at collar mount point (bit-for-bit seam lock!)
            tmpThetaBpCollar.copy(tmpThetaBp).multiplyScalar(modalCollarBpW);
            setQuatFromClampedRotVec(tmpThetaBpCollar, maxStageAngle, tmpBpCollarQuat);

            tmpCollarWorldQuat.copy(tmpRailQuat).multiply(tmpBpCollarQuat).multiply(tmpBodyQuat);

            // Compute collar displacement already produced by pure rotations (rail + backplate + body)
            tmpCollarS2
              .copy(modalBodyRootLocal)
              .negate()
              .applyQuaternion(tmpBodyQuat)
              .add(modalBodyRootLocal);
            tmpUFromRot
              .copy(tmpCollarS2)
              .add(modalCollarRelToTh)
              .sub(modalBackplatePivotLocal)
              .applyQuaternion(tmpBpCollarQuat)
              .add(modalBackplatePivotLocal)
              .applyQuaternion(tmpRailQuat)
              .sub(modalCollarRelToTh);

            // Residual translational shear (clamped so it never opposes bending or bulges backwards)
            tmpUShearTotal.copy(tmpUStage12).sub(tmpUFromRot);
            const rotLenSq = tmpUFromRot.lengthSq();
            if (rotLenSq > 1e-8) {
              const dot = tmpUShearTotal.dot(tmpUFromRot);
              if (dot < 0) {
                tmpUShearTotal.addScaledVector(tmpUFromRot, -dot / rotLenSq);
              }
            }
            const maxShear = Math.max(0.03, baseStroke * 0.42);
            const shearMag = tmpUShearTotal.length();
            if (shearMag > maxShear) {
              tmpUShearTotal.multiplyScalar(maxShear / shearMag);
            }
            tmpUShearLocal.copy(tmpUShearTotal).applyQuaternion(tmpInvRailQuat);
            tmpUBpShear.copy(tmpUShearLocal).multiplyScalar(plateFrac);
            tmpUBodyShear.copy(tmpUShearLocal).multiplyScalar(bodyFrac);

            // Exact deformed world collar center (bit-for-bit identical to deformBodyMesh at local (0,0,0))
            tmpCollarS2.add(tmpUBodyShear);
            tmpCollarWorldPos
              .copy(tmpCollarS2)
              .add(modalCollarRelToTh)
              .sub(modalBackplatePivotLocal)
              .applyQuaternion(tmpBpCollarQuat)
              .add(modalBackplatePivotLocal)
              .addScaledVector(tmpUBpShear, modalCollarBpW)
              .applyQuaternion(tmpRailQuat)
              .add(curTh);

            // Rigid rail-carriage bearing roll applied identically to MGN block, backplate base, and body base
            if (modalToolhead) {
              modalToolhead.position.copy(curTh);
              modalToolhead.quaternion.copy(tmpRailQuat);
            }
            if (modalBackplateGroup) {
              modalBackplateGroup.position.copy(curTh);
              modalBackplateGroup.quaternion.copy(tmpRailQuat);
            }
            if (modalCogMarker) {
              modalCogMarker.position
                .copy(modalCollarRelToTh)
                .applyQuaternion(tmpRailQuat)
                .add(curTh);
              modalCogMarker.quaternion.copy(tmpRailQuat);
            }

            const pvx = modalBackplatePivotLocal.x;
            const pvy = modalBackplatePivotLocal.y;
            const pvz = modalBackplatePivotLocal.z;
            const brx = modalBodyRootLocal.x;
            const bry = modalBodyRootLocal.y;
            const brz = modalBodyRootLocal.z;
            const crx = modalCollarRelToTh.x;
            const cry = modalCollarRelToTh.y;
            const crz = modalCollarRelToTh.z;

            for (let mi = 0; mi < modalElasticMeshes.length; mi++) {
              const em = modalElasticMeshes[mi];
              if (em.stage === "backplate") {
                deformBackplateMesh(
                  em,
                  pvx,
                  pvy,
                  pvz,
                  tmpThetaBp.x,
                  tmpThetaBp.y,
                  tmpThetaBp.z,
                  tmpUBpShear.x,
                  tmpUBpShear.y,
                  tmpUBpShear.z
                );
              } else if (em.stage === "body") {
                deformBodyMesh(
                  em,
                  pvx,
                  pvy,
                  pvz,
                  brx,
                  bry,
                  brz,
                  crx,
                  cry,
                  crz,
                  tmpThetaBp.x,
                  tmpThetaBp.y,
                  tmpThetaBp.z,
                  tmpUBpShear.x,
                  tmpUBpShear.y,
                  tmpUBpShear.z,
                  tmpThetaBody.x,
                  tmpThetaBody.y,
                  tmpThetaBody.z,
                  tmpUBodyShear.x,
                  tmpUBodyShear.y,
                  tmpUBodyShear.z
                );
              }
            }
          } else {
            tmpThetaRail.copy(tmpRotCogVec).multiplyScalar(modalRailFlexFrac);
            setQuatFromClampedRotVec(tmpThetaRail, maxStageAngle, tmpRailQuat);
            if (modalToolhead) {
              modalToolhead.quaternion.copy(tmpRailQuat);
            }
            if (modalBackplateGroup) {
              modalBackplateGroup.position.copy(curTh);
              modalBackplateGroup.quaternion
                .copy(tmpIdentQuat)
                .slerp(tmpStage1Quat, Math.min(1, modalRailFlexFrac + modalPlateFlexFrac));
            }
            if (modalCogMarker) {
              modalCogMarker.position.copy(curCog).add(modalCogOffset);
              modalCogMarker.quaternion.copy(tmpStage1Quat);
            }
          }
        }

        // Stage 3: True Euler-Bernoulli Elastic Hotend Cantilever — clamped 100% square inside collar!
        if (modalNozzleMarker && modalNozzleNodeId) {
          const rawNozRot = modalRotations[modalNozzleNodeId];
          const effNozRot = isIsoToolhead
            ? rawNozRot
              ? tmpIsoRelNozRot
              : undefined
            : rawNozRot;

          if (modalElasticMeshes.length > 0 && modalCogMarker) {
            // Pin hotend top flange (local 0, modalHotendLen, 0) flush & square inside deformed collar
            modalNozzleMarker.quaternion.copy(tmpCollarWorldQuat);
            tmpHotendTopVec.set(0, modalHotendLen, 0).applyQuaternion(tmpCollarWorldQuat);
            modalNozzleMarker.position.copy(tmpCollarWorldPos).sub(tmpHotendTopVec);

            // Compute relative hotend bend in collar-local frame (scaled proportionally to maxStageAngle)
            tmpThetaBendLocal.set(0, 0, 0);
            tmpInvCollarQuat.copy(tmpCollarWorldQuat).invert();
            if (effNozRot && effCogRot) {
              tmpRelRotVec.copy(effNozRot).sub(effCogRot).multiplyScalar(toolPhaseMeters);
              const relMag = tmpRelRotVec.length();
              if (relMag > maxStageAngle) tmpRelRotVec.multiplyScalar(maxStageAngle / relMag);
              tmpThetaBendLocal.copy(tmpRelRotVec).applyQuaternion(tmpInvCollarQuat);
            } else if (
              modalNozzleRefNodeId &&
              basePositions[modalNozzleRefNodeId] &&
              basePositions[modalNozzleNodeId]
            ) {
              const curNoz = getPosInto(modalNozzleNodeId, tmpPosA);
              const curRef = getPosInto(modalNozzleRefNodeId, tmpPosB);
              tmpNozV1.subVectors(curRef, curNoz);
              if (tmpNozV1.lengthSq() > 1e-6 && modalNozzleBaseDir.lengthSq() > 1e-6) {
                tmpHotendQuat.setFromUnitVectors(modalNozzleBaseDir, tmpNozV1.normalize());
                tmpRelQuat.copy(tmpInvCollarQuat).multiply(tmpHotendQuat);
                extractRotVecFromQuat(tmpRelQuat, maxStageAngle, tmpThetaBendLocal);
              }
            }

            for (let mi = 0; mi < modalElasticMeshes.length; mi++) {
              const em = modalElasticMeshes[mi];
              if (em.stage === "hotend") {
                deformHotendMesh(
                  em,
                  modalHotendLen,
                  tmpThetaBendLocal.x,
                  tmpThetaBendLocal.y,
                  tmpThetaBendLocal.z
                );
              }
            }

            if (modalNozzleDispLine) {
              const dispAttr = modalNozzleDispLine.geometry.getAttribute(
                "position"
              ) as THREE.BufferAttribute;
              if (dispAttr && dispAttr.count >= 2) {
                const hotendLut = getHotendProfileLUT(modalHotendStyle);
                const tipDispFrac = hotendLut.disp[HOTEND_LUT_N];
                const tipDx = tmpThetaBendLocal.z * modalHotendLen * tipDispFrac;
                const tipDz = -tmpThetaBendLocal.x * modalHotendLen * tipDispFrac;
                const tipDy =
                  (0.5 * (tipDx * tipDx + tipDz * tipDz)) / Math.max(modalHotendLen, 0.1);
                tmpNozTipWorld
                  .set(tipDx, tipDy, tipDz)
                  .applyQuaternion(tmpCollarWorldQuat)
                  .add(modalNozzleMarker.position);
                dispAttr.setXYZ(1, tmpNozTipWorld.x, tmpNozTipWorld.y, tmpNozTipWorld.z);
                dispAttr.needsUpdate = true;
              }
            }
          } else {
            tmpHotendQuat.identity();
            if (effNozRot && effCogRot) {
              tmpRotCogVec.copy(effCogRot).multiplyScalar(toolPhaseMeters);
              const cogMag = tmpRotCogVec.length();
              if (cogMag > maxStageAngle) tmpRotCogVec.multiplyScalar(maxStageAngle / cogMag);

              tmpRelRotVec.copy(effNozRot).sub(effCogRot).multiplyScalar(toolPhaseMeters);
              const relMag = tmpRelRotVec.length();
              if (relMag > maxStageAngle) tmpRelRotVec.multiplyScalar(maxStageAngle / relMag);

              tmpTotalRotVec.copy(tmpRotCogVec).add(tmpRelRotVec);
              setQuatFromClampedRotVec(tmpTotalRotVec, maxStageAngle * 1.25, tmpHotendQuat);
            } else if (
              modalNozzleRefNodeId &&
              basePositions[modalNozzleRefNodeId] &&
              basePositions[modalNozzleNodeId]
            ) {
              const curNoz = getPosInto(modalNozzleNodeId, tmpPosA);
              const curRef = getPosInto(modalNozzleRefNodeId, tmpPosB);
              tmpNozV1.subVectors(curRef, curNoz);
              if (tmpNozV1.lengthSq() > 1e-6 && modalNozzleBaseDir.lengthSq() > 1e-6) {
                tmpHotendQuat.setFromUnitVectors(modalNozzleBaseDir, tmpNozV1.normalize());
                const qAngle = 2 * Math.acos(Math.min(1, Math.abs(tmpHotendQuat.w)));
                const maxFallbackAngle = maxStageAngle * 1.15;
                if (qAngle > maxFallbackAngle) {
                  tmpHotendQuat.slerp(tmpIdentQuat, 1 - maxFallbackAngle / qAngle);
                }
              }
            }

            modalNozzleMarker.quaternion.copy(tmpHotendQuat);
            if (modalCogMarker) {
              tmpHotendTopVec.set(0, modalHotendLen, 0).applyQuaternion(tmpHotendQuat);
              modalNozzleMarker.position.copy(modalCogMarker.position).sub(tmpHotendTopVec);
            } else {
              const curNoz = getPosInto(modalNozzleNodeId, tmpPosA);
              modalNozzleMarker.position.copy(curNoz).add(modalNozzleOffset);
            }

            if (modalNozzleDispLine) {
              const dispAttr = modalNozzleDispLine.geometry.getAttribute(
                "position"
              ) as THREE.BufferAttribute;
              if (dispAttr && dispAttr.count >= 2) {
                dispAttr.setXYZ(
                  1,
                  modalNozzleMarker.position.x,
                  modalNozzleMarker.position.y,
                  modalNozzleMarker.position.z
                );
                dispAttr.needsUpdate = true;
              }
            }
          }
        }
        if (modalBedMesh && modalBedNodeId) {
          getPosInto(modalBedNodeId, modalBedMesh.position).add(modalBedOffset);
        }
      }

      renderer.render(scene, camera);
    };

    function scheduleModalTick(): void {
      if (disposed || modalTimerId !== null || rafId !== null) return;
      if (!isModalAnimating()) return;
      modalTimerId = window.setTimeout(() => {
        modalTimerId = null;
        if (disposed || !isModalAnimating()) return;
        scheduleRaf();
      }, MODAL_FRAME_INTERVAL_MS);
    }

    function scheduleRaf(): void {
      if (disposed || document.hidden) return;
      clearModalTimer();
      if (rafId !== null) return;
      rafId = requestAnimationFrame(runRafTick);
      stateRef.current.animFrame = rafId;
    }

    function runRafTick(): void {
      rafId = null;
      stateRef.current.animFrame = undefined;
      if (disposed || document.hidden) {
        lastModalAdvanceMs = null;
        prevCamAnimMs = null;
        return;
      }

      if (isModalAnimating()) {
        const nowMs = performance.now();
        const dtModal =
          lastModalAdvanceMs !== null
            ? Math.min(Math.max((nowMs - lastModalAdvanceMs) / 1000, 0), 0.1)
            : 0;
        lastModalAdvanceMs = nowMs;
        modalPhaseSec += dtModal;
      } else {
        lastModalAdvanceMs = null;
      }

      renderFrame(modalPhaseSec, true);

      const needsInteractiveRaf =
        userOrbiting || needsDamping || cameraAnimRef.current.active;
      if (needsInteractiveRaf) {
        scheduleRaf();
      } else if (isModalAnimating()) {
        scheduleModalTick();
      }
    }

    const requestRender = () => {
      if (disposed) return;
      if (document.hidden) {
        clearModalTimer();
        clearScheduledRaf();
        lastModalAdvanceMs = null;
        prevCamAnimMs = null;
        return;
      }
      if (!isModalAnimating()) {
        clearModalTimer();
        lastModalAdvanceMs = null;
      }
      if (cameraAnimRef.current.active) {
        needsDamping = false;
        prevCamAnimMs = null;
      }
      scheduleRaf();
    };

    requestRenderRef.current = requestRender;

    const onControlsStart = () => {
      cameraAnimRef.current.active = false;
      prevCamAnimMs = null;
      userOrbiting = true;
      needsDamping = true;
      settledFrames = 0;
      dampFrames = 0;
      scheduleRaf();
    };

    const onControlsChange = () => {
      if (inInternalControlsUpdate) return;
      needsDamping = true;
      settledFrames = 0;
      dampFrames = 0;
      scheduleRaf();
    };

    const onControlsEnd = () => {
      userOrbiting = false;
      needsDamping = true;
      settledFrames = 0;
      dampFrames = 0;
      scheduleRaf();
    };

    controls.addEventListener("start", onControlsStart);
    controls.addEventListener("change", onControlsChange);
    controls.addEventListener("end", onControlsEnd);

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
            requestRender();
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
        requestRender();
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
        applyInspectHighlight(hitMesh, "pin", 0.65);
        setPinnedPart(info);
        if (prevPinned !== hitMesh) {
          requestRender();
        }
      } else {
        stateRef.current.pinnedMesh = null;
        stateRef.current.pinnedKey = null;
        setPinnedPart(null);
        if (prevPinned) {
          requestRender();
        }
      }
    };

    const onPointerLeave = () => {
      pointerDownPos = null;
      renderer.domElement.style.cursor = "";
      const prevHover = stateRef.current.hoveredMesh;
      stateRef.current.hoveredMesh = null;
      if (prevHover && prevHover !== stateRef.current.pinnedMesh) {
        applyInspectHighlight(prevHover, "none");
        requestRender();
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
      if (disposed || !el) return;
      const w = el.clientWidth;
      const h = el.clientHeight;
      if (w <= 0 || h <= 0) return;
      renderer.setPixelRatio(
        stateRef.current.ecoDpr ? 1 : Math.min(window.devicePixelRatio || 1, 1.5)
      );
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
      requestRender();
    };
    window.addEventListener("resize", onResize);

    let resizeObserver: ResizeObserver | null = null;
    if (typeof ResizeObserver !== "undefined") {
      resizeObserver = new ResizeObserver(() => {
        onResize();
      });
      resizeObserver.observe(el);
    }

    const onVisibilityChange = () => {
      if (disposed) return;
      if (document.hidden) {
        clearModalTimer();
        clearScheduledRaf();
        lastModalAdvanceMs = null;
        prevCamAnimMs = null;
      } else {
        lastModalAdvanceMs = null;
        prevCamAnimMs = null;
        requestRender();
      }
    };
    document.addEventListener("visibilitychange", onVisibilityChange);

    // Render initial frame on demand
    requestRender();

    const captureViewport = async (frames = 2, delayMs = 420): Promise<string[]> => {
      if (disposed) return [];
      const count = Math.max(1, Math.min(4, Math.round(Number.isFinite(frames) ? frames : 2)));
      const waitMs = Math.max(0, Math.min(2000, Math.round(Number.isFinite(delayMs) ? delayMs : 420)));
      const results: string[] = [];

      const isModal = stateRef.current.isModal;
      let t0 = modalPhaseSec;
      // In modal mode, guarantee crisp peak deformation (avoid zero-crossings) and
      // deterministic opposite-extreme half-cycle phase stepping even when paused or in background tabs.
      if (isModal && Math.abs(Math.sin(t0 * 7.5)) < 0.65) {
        const halfCycles = Math.round((t0 * 7.5) / Math.PI - 0.5);
        t0 = ((halfCycles + 0.5) * Math.PI) / 7.5;
      }
      const stepSec = isModal ? Math.PI / 7.5 : waitMs / 1000;

      for (let i = 0; i < count; i++) {
        if (disposed) break;
        const tFrame = t0 + i * stepSec;
        const prevBg = scene.background;
        scene.background = captureBgColor;
        renderFrame(tFrame, false);
        try {
          const dataUrl = renderer.domElement.toDataURL("image/png");
          if (dataUrl && dataUrl.startsWith("data:image/png")) {
            results.push(dataUrl);
          }
        } catch {
          // Ignore transient WebGL context errors
        } finally {
          scene.background = prevBg;
        }
        // Restore current/paused pose immediately so foreground rendering stays seamless
        renderFrame(modalPhaseSec, false);

        if (i < count - 1 && waitMs > 0) {
          await new Promise<void>((resolve) => {
            window.setTimeout(resolve, waitMs);
          });
        }
      }
      return results;
    };

    captureFnRef.current = captureViewport;
    onRegisterCaptureRef.current?.(captureViewport);

    return () => {
      disposed = true;
      requestRenderRef.current = null;
      captureFnRef.current = null;
      clearModalTimer();
      clearScheduledRaf();
      document.removeEventListener("visibilitychange", onVisibilityChange);
      window.removeEventListener("resize", onResize);
      resizeObserver?.disconnect();
      renderer.domElement.removeEventListener("pointerdown", onPointerDown);
      renderer.domElement.removeEventListener("pointermove", onPointerMove);
      renderer.domElement.removeEventListener("pointerup", onPointerUp);
      renderer.domElement.removeEventListener("pointerleave", onPointerLeave);
      disposeGroupChildren(dynamicGroup);
      grid.geometry.dispose();
      if (Array.isArray(grid.material)) grid.material.forEach((m) => m.dispose());
      else grid.material.dispose();
      controls.removeEventListener("start", onControlsStart);
      controls.removeEventListener("change", onControlsChange);
      controls.removeEventListener("end", onControlsEnd);
      controls.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === el) el.removeChild(renderer.domElement);
    };
  }, []);

  // Rebuild 3D geometry when simulation results or visualization mode changes
  useEffect(() => {
    const group = stateRef.current.dynamicGroup;
    if (!group) return;

    try {
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
    stateRef.current.modalRailShoe = undefined;
    stateRef.current.modalToolheadNodeId = undefined;
    stateRef.current.modalBackplateGroup = undefined;
    stateRef.current.modalCogMarker = undefined;
    stateRef.current.modalCogNodeId = undefined;
    stateRef.current.modalCogParentNodeId = undefined;
    stateRef.current.modalCogOffset.set(0, 0, 0);
    stateRef.current.modalNozzleMarker = undefined;
    stateRef.current.modalNozzleNodeId = undefined;
    stateRef.current.modalNozzleRefNodeId = undefined;
    stateRef.current.modalNozzleOffset.set(0, 0, 0);
    stateRef.current.modalNozzleBaseDir.set(0, 1, 0);
    stateRef.current.modalNozzleDispLine = undefined;
    stateRef.current.modalElasticMeshes = [];
    stateRef.current.modalRailFlexFrac = 0;
    stateRef.current.modalPlateFlexFrac = 0.5;
    stateRef.current.modalBodyFlexFrac = 0.5;
    stateRef.current.modalCollarBpW = 1.0;
    stateRef.current.modalHotendStyle = "dragon_4bolt_m25";
    stateRef.current.modalBackplatePivotLocal.set(0, 0, 0);
    stateRef.current.modalBodyRootLocal.set(0, 0, 0);
    stateRef.current.modalCollarRelToTh.set(0, 0, 0);
    stateRef.current.modalBodyDepthRef = 0.22;
    stateRef.current.modalBedMesh = undefined;
    stateRef.current.modalBedNodeId = undefined;
    stateRef.current.modalRotations = {};
    stateRef.current.modalHotendLen = 0.45;
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

    const mRots: Record<string, THREE.Vector3> = {};
    if (activeMode?.nodal_rotations) {
      for (const [nid, rot] of Object.entries(activeMode.nodal_rotations)) {
        // CAD Z-up [rx, ry, rz] (rad/m) -> Three.js Y-up [rx, rz, -ry] (right-handed, det=+1)
        mRots[nid] = new THREE.Vector3(rot[0], rot[2], -rot[1]);
      }
    }
    stateRef.current.modalRotations = mRots;

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
    const thFocusBase = basePos[geom.toolhead_node_id || "toolhead"] ?? null;

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
          m.id.includes("z_column") ||
          m.id.startsWith("bed_rear_z_") ||
          m.id.startsWith("bed_left_z_") ||
          m.id.startsWith("bed_right_z_"));
      const rawW = Math.max(m.width_mm * SCALE, isBedMember ? 0.05 : 0.06);
      const rawH = Math.max(m.height_mm * SCALE, isBedMember ? 0.05 : 0.06);
      const w = isBedMember && !isBedPost ? Math.min(rawW, 0.11) : rawW;
      const h = isBedMember && !isBedPost ? Math.min(rawH, 0.11) : rawH;

      // In Toolhead Punch-In mode, ghost far outer frame / brace / bed members so the X-beam & toolhead stay unobstructed
      const isToolheadContextBeam =
        m.role === "gantry" ||
        m.id.startsWith("x_gantry") ||
        m.id.startsWith("x_beam") ||
        m.id.startsWith("y_rail");
      let memberOpacity = frameOpacity;
      if (toolheadPunchIn && !isToolheadContextBeam) {
        const midPt = basePos[firstId].clone().add(basePos[lastId]).multiplyScalar(0.5);
        const distToTh = thFocusBase ? midPt.distanceTo(thFocusBase) : 99;
        if (m.role === "frame" || m.role === "brace" || m.role === "bed" || distToTh > 2.1) {
          memberOpacity = Math.min(memberOpacity, 0.16);
        }
      }

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
            ? new THREE.Color(m.color_hex || "#9aa8ba")
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
        transparent: memberOpacity < 1,
        opacity: memberOpacity,
        depthWrite: memberOpacity >= 0.5,
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

        // Continuous T-slot core accent strip on filled stationary extrusions (Assembly mode only)
        if (
          fillingMat &&
          !toolheadPunchIn &&
          (m.role === "frame" || m.role === "rail" || m.role === "tower")
        ) {
          fillingMatUsed = true;
          const inlayGeoA = new THREE.BoxGeometry(w * 0.32, 0.96, h * 1.03);
          const inlayA = new THREE.Mesh(inlayGeoA, fillingMat);
          orientBeamBoxMesh(inlayA, baseFirst, baseLast);
          group.add(inlayA);

          const inlayGeoB = new THREE.BoxGeometry(w * 1.03, 0.96, h * 0.32);
          const inlayB = new THREE.Mesh(inlayGeoB, fillingMat);
          orientBeamBoxMesh(inlayB, baseFirst, baseLast);
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

          const isBackboneBay = bayName === "backbone";
          const shellMat = new THREE.MeshStandardMaterial({
            vertexColors: true,
            transparent: true,
            opacity,
            depthWrite: false,
            roughness: isBackboneBay ? 0.34 : 0.36,
            metalness: isBackboneBay ? 0.42 : 0.14,
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
          const bayLabel = isBackboneBay
            ? "Trident Backbone (Clee 3mm Steel, MITC4)"
            : `${bayName.charAt(0).toUpperCase()}${bayName.slice(1)} Enclosure Panel (MITC4)`;
          tagInspectable(shellMesh, {
            key: isBackboneBay ? "frame:trident_backbone" : `panel:${bayName}`,
            name: bayLabel,
            subsystem: "frame",
            partGroupId: bayGrp.id,
            role: "panel",
            material: isBackboneBay
              ? (geom.trident_backbone?.material_name ?? "Powder-Coated Mild Steel")
              : (geom.panels?.material_name ?? sim.spec.frame.panel_material),
            profile: isBackboneBay
              ? `${panels[0]?.thickness_mm ?? geom.trident_backbone?.thickness_mm ?? 3}mm steel plate (MITC4)`
              : `${panels[0]?.thickness_mm ?? sim.spec.frame.panel_thickness_mm}mm MITC4 shell`,
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
          const baseBayOpacity =
            bayName === "top"
              ? 0.28
              : bayName === "backbone"
                ? viewMode === "sensitivity"
                  ? 0.85
                  : 0.88
                : viewMode === "sensitivity"
                  ? 0.65
                  : 0.72;
          const bayOpacity = toolheadPunchIn ? baseBayOpacity * 0.14 : baseBayOpacity;
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
        const xIsMgn9 = xRailProf.includes("mgn9");
        const xIsMgn15 = xRailProf.includes("mgn15");
        const xRailW = Math.max(
          (lr?.x_rail_width_mm ?? (xIsMgn9 ? 9.0 : xIsMgn15 ? 15.0 : 12.0)) * SCALE,
          0.075
        );
        const xRailH = Math.max(
          (lr?.x_rail_height_mm ?? (xIsMgn9 ? 6.5 : xIsMgn15 ? 10.0 : 8.0)) * SCALE,
          0.055
        );
        const yRailW = Math.max(
          (lr?.y_rail_width_mm ?? (yRailProf.includes("mgn9") ? 9.0 : yRailProf.includes("mgn15") ? 15.0 : 12.0)) * SCALE,
          0.075
        );
        const yRailH = Math.max(
          (lr?.y_rail_height_mm ?? (yRailProf.includes("mgn9") ? 6.5 : yRailProf.includes("mgn15") ? 10.0 : 8.0)) * SCALE,
          0.055
        );

        const railSteelMat = new THREE.MeshStandardMaterial({
          color: 0xa6b4c4,
          metalness: 0.86,
          roughness: 0.22,
        });
        const railEdgeMat = new THREE.LineBasicMaterial({
          color: 0xe2e8f0,
          transparent: true,
          opacity: 0.34,
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
          const beamHalfCadY = 0.5 * Math.max(xGantryL.width_mm * SCALE, 0.06);
          const beamHalfCadZ = 0.5 * Math.max(xGantryL.height_mm * SCALE, 0.06);

          // Subtle M3 counterbore bolt holes + dark longitudinal ball-raceway grooves along the rail
          const boltPitch = (xIsMgn9 ? 20.0 : xIsMgn15 ? 40.0 : 25.0) * SCALE;
          const boltRadius = (xIsMgn9 ? 1.8 : xIsMgn15 ? 2.8 : 2.3) * SCALE;
          const boltCounterboreGeo = new THREE.CylinderGeometry(boltRadius, boltRadius, 0.008, 10);
          const boltCounterboreMat = new THREE.MeshStandardMaterial({
            color: 0x1e293b,
            metalness: 0.45,
            roughness: 0.58,
          });
          const grooveMat = new THREE.MeshStandardMaterial({
            color: 0x475569,
            metalness: 0.72,
            roughness: 0.4,
          });
          const halfSpan = Math.max(railLenX * 0.5 - 0.1, 0);
          const boltCount = Math.max(1, Math.floor((halfSpan * 2) / boltPitch) + 1);
          const startBoltX = beamCX - ((boltCount - 1) * boltPitch) * 0.5;

          const addXCounterboresAndGrooves = (
            face: "front" | "top" | "bottom",
            railCenter: THREE.Vector3
          ) => {
            // Two subtle longitudinal ball-race grooves along the sides of the rail
            const grooveGeo =
              face === "front"
                ? new THREE.BoxGeometry(railLenX - 0.01, 0.012, 0.014)
                : new THREE.BoxGeometry(railLenX - 0.01, 0.014, 0.012);
            for (const side of [-1, 1]) {
              const gMesh = new THREE.Mesh(grooveGeo, grooveMat);
              if (face === "front") {
                gMesh.position.set(
                  railCenter.x,
                  railCenter.y + side * (xRailW * 0.5 - 0.004),
                  railCenter.z + xRailH * 0.1
                );
              } else {
                gMesh.position.set(
                  railCenter.x,
                  railCenter.y + (face === "top" ? 1 : -1) * xRailH * 0.1,
                  railCenter.z + side * (xRailW * 0.5 - 0.004)
                );
              }
              group.add(gMesh);
            }
            // M3 mounting counterbore dots along the outer rail face
            for (let bi = 0; bi < boltCount; bi++) {
              const bx = startBoltX + bi * boltPitch;
              const hole = new THREE.Mesh(boltCounterboreGeo, boltCounterboreMat);
              if (face === "front") {
                hole.rotation.x = Math.PI * 0.5;
                hole.position.set(bx, railCenter.y, railCenter.z + xRailH * 0.5 + 0.001);
              } else if (face === "top") {
                hole.position.set(bx, railCenter.y + xRailH * 0.5 + 0.001, railCenter.z);
              } else {
                hole.position.set(bx, railCenter.y - xRailH * 0.5 - 0.001, railCenter.z);
              }
              group.add(hole);
            }
          };

          if (xRailMount === "front") {
            // CAD -Y face -> Three.js +Z face; wide face in CAD Z (Three.js Y), thickness in CAD -Y (Three.js +Z)
            const xGeo = new THREE.BoxGeometry(railLenX, xRailW, xRailH);
            const xEdges = new THREE.EdgesGeometry(xGeo);
            const pos = new THREE.Vector3(beamCX, beamCY, beamCZ + beamHalfCadY + xRailH * 0.5);
            addRailBar(xGeo, xEdges, pos);
            addXCounterboresAndGrooves("front", pos);
          } else if (xRailMount === "top") {
            // CAD +Z face -> Three.js +Y face; wide face in CAD Y (Three.js Z), thickness in CAD +Z (Three.js +Y)
            const xGeo = new THREE.BoxGeometry(railLenX, xRailH, xRailW);
            const xEdges = new THREE.EdgesGeometry(xGeo);
            const pos = new THREE.Vector3(beamCX, beamCY + beamHalfCadZ + xRailH * 0.5, beamCZ);
            addRailBar(xGeo, xEdges, pos);
            addXCounterboresAndGrooves("top", pos);
          } else if (xRailMount === "bottom") {
            // CAD -Z face -> Three.js -Y face
            const xGeo = new THREE.BoxGeometry(railLenX, xRailH, xRailW);
            const xEdges = new THREE.EdgesGeometry(xGeo);
            const pos = new THREE.Vector3(beamCX, beamCY - beamHalfCadZ - xRailH * 0.5, beamCZ);
            addRailBar(xGeo, xEdges, pos);
            addXCounterboresAndGrooves("bottom", pos);
          } else if (xRailMount === "dual_top_bottom") {
            // Dual rails on both +Z (+Y Three.js) and -Z (-Y Three.js) faces
            const xGeo = new THREE.BoxGeometry(railLenX, xRailH, xRailW);
            const xEdges = new THREE.EdgesGeometry(xGeo);
            const posTop = new THREE.Vector3(beamCX, beamCY + beamHalfCadZ + xRailH * 0.5, beamCZ);
            const posBot = new THREE.Vector3(beamCX, beamCY - beamHalfCadZ - xRailH * 0.5, beamCZ);
            addRailBar(xGeo, xEdges, posTop);
            addXCounterboresAndGrooves("top", posTop);
            addRailBar(xGeo, xEdges, posBot);
            addXCounterboresAndGrooves("bottom", posBot);
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

      // Z Linear Rails & MGN Carriages / Smooth Chrome Rods (from geom.z_rails)
      if (geom.z_rails && geom.z_rails.length > 0) {
        const zRailSteelMat = new THREE.MeshStandardMaterial({
          color: 0x9aa6b4,
          metalness: 0.82,
          roughness: 0.24,
          transparent: toolheadPunchIn,
          opacity: toolheadPunchIn ? 0.18 : 1.0,
        });
        const zRodChromeMat = new THREE.MeshStandardMaterial({
          color: 0xe2e8f0,
          metalness: 0.92,
          roughness: 0.14,
          transparent: toolheadPunchIn,
          opacity: toolheadPunchIn ? 0.18 : 1.0,
        });
        const zCarriageBodyMat = new THREE.MeshStandardMaterial({
          color: 0x64748b,
          metalness: 0.72,
          roughness: 0.3,
          transparent: toolheadPunchIn,
          opacity: toolheadPunchIn ? 0.18 : 1.0,
        });
        const zCarriageCapMat = new THREE.MeshStandardMaterial({
          color: 0x14532d,
          metalness: 0.15,
          roughness: 0.55,
          transparent: toolheadPunchIn,
          opacity: toolheadPunchIn ? 0.18 : 1.0,
        });
        const zRailEdgeMat = new THREE.LineBasicMaterial({
          color: 0xdbeafe,
          transparent: true,
          opacity: toolheadPunchIn ? 0.08 : 0.32,
        });

        let zRailUsed = false;
        let zRodUsed = false;
        let zCapUsed = false;

        for (const zr of geom.z_rails) {
          if (zr.profile_id === "none") continue;
          const p0 = toThree(zr.start_mm, cx, cy);
          const p1 = toThree(zr.end_mm, cx, cy);
          const axisVec = new THREE.Vector3().subVectors(p1, p0);
          const railLen = axisVec.length();
          if (railLen < 1e-3) continue;
          const axisDir = axisVec.clone().divideScalar(railLen);

          const zMidX = 0.5 * (p0.x + p1.x);
          const zMidZ = 0.5 * (p0.z + p1.z);
          const carY = zr.carriage_z_mm * SCALE;
          const isSmoothRod = zr.profile_id.startsWith("steel_rod");

          const zRailInspect: InspectedPartInfo = {
            key: `z_rail:${zr.id}`,
            name: isSmoothRod
              ? `Z Smooth Rod (${zr.profile_id.replace("steel_rod_", "")})`
              : `Z Linear Rail (${zr.profile_id.includes("mgn9") ? "MGN9H" : "MGN12H"})`,
            subsystem: "bed",
            partGroupId: "bed_support",
            role: "bed",
            material: "hardened_steel",
            profile: zr.profile_id,
          };
          const zCarInspect: InspectedPartInfo = {
            ...zRailInspect,
            key: `z_carriage:${zr.id}`,
            name: isSmoothRod
              ? `Z Linear Bearing (${zr.profile_id.replace("steel_rod_", "LM")}UU)`
              : `Z Rail Carriage (${zr.profile_id.includes("mgn9") ? "MGN9H" : "MGN12H"})`,
          };

          if (isSmoothRod) {
            zRodUsed = true;
            const diaMm = zr.profile_id.includes("12")
              ? 12
              : zr.profile_id.includes("10")
                ? 10
                : 8;
            const rodR = Math.max(diaMm * 0.5 * SCALE, 0.036);
            const rodGeo = new THREE.CylinderGeometry(rodR, rodR, 1, 14);
            const rodMesh = new THREE.Mesh(rodGeo, zRodChromeMat);
            orientBeamMesh(rodMesh, p0, p1);
            tagInspectable(rodMesh, zRailInspect);
            group.add(rodMesh);

            // Linear ball bearing sleeve (LM*UU) at carriage_z_mm
            const sleeveGeo = new THREE.CylinderGeometry(rodR * 1.85, rodR * 1.85, 0.28, 16);
            const sleeveMesh = new THREE.Mesh(sleeveGeo, zCarriageBodyMat);
            sleeveMesh.position.set(zMidX, carY, zMidZ);
            tagInspectable(sleeveMesh, zCarInspect);
            group.add(sleeveMesh);
          } else {
            zRailUsed = true;
            zCapUsed = true;
            const isMgn9 = zr.profile_id.includes("mgn9");
            const rW = (isMgn9 ? 9.0 : 12.0) * SCALE;
            const rH = (isMgn9 ? 6.5 : 8.0) * SCALE;
            // Convert CAD normal [nx, ny, nz] -> Three.js normal (nx, nz, -ny)
            const normVec = new THREE.Vector3(
              zr.normal[0],
              zr.normal[2],
              -zr.normal[1]
            );
            normVec.addScaledVector(axisDir, -normVec.dot(axisDir));
            if (normVec.lengthSq() < 1e-6) normVec.set(0, 0, 1);
            else normVec.normalize();
            const widthVec = new THREE.Vector3().crossVectors(axisDir, normVec).normalize();
            const basisMat = new THREE.Matrix4().makeBasis(widthVec, axisDir, normVec);
            const railQuat = new THREE.Quaternion().setFromRotationMatrix(basisMat);

            const barGeo = new THREE.BoxGeometry(rW, railLen, rH);
            const barMesh = new THREE.Mesh(barGeo, zRailSteelMat);
            barMesh.position
              .copy(p0)
              .add(p1)
              .multiplyScalar(0.5)
              .addScaledVector(normVec, rH * 0.5);
            barMesh.quaternion.copy(railQuat);
            tagInspectable(barMesh, zRailInspect);
            group.add(barMesh);

            const barOutline = new THREE.LineSegments(new THREE.EdgesGeometry(barGeo), zRailEdgeMat);
            barOutline.position.copy(barMesh.position);
            barOutline.quaternion.copy(railQuat);
            group.add(barOutline);

            // MGN metallic carriage block + wiper end-caps at carriage_z_mm
            const carW = (isMgn9 ? 20 : 27) * SCALE;
            const carL = (isMgn9 ? 36 : 44) * SCALE;
            const carH = (isMgn9 ? 10 : 13) * SCALE;
            const carGroup = new THREE.Group();
            const carBody = new THREE.Mesh(
              new THREE.BoxGeometry(carW, carL * 0.72, carH),
              zCarriageBodyMat
            );
            tagInspectable(carBody, zCarInspect);
            carGroup.add(carBody);

            const capGeo = new THREE.BoxGeometry(carW * 0.96, carL * 0.14, carH * 0.94);
            const topCap = new THREE.Mesh(capGeo, zCarriageCapMat);
            topCap.position.y = carL * 0.43;
            carGroup.add(topCap);
            const botCap = new THREE.Mesh(capGeo, zCarriageCapMat);
            botCap.position.y = -carL * 0.43;
            carGroup.add(botCap);

            carGroup.position
              .set(zMidX, carY, zMidZ)
              .addScaledVector(normVec, rH * 0.5 + carH * 0.42);
            carGroup.quaternion.copy(railQuat);
            group.add(carGroup);
          }
        }

        if (!zRailUsed) {
          zRailSteelMat.dispose();
          zRailEdgeMat.dispose();
        }
        if (!zRodUsed) zRodChromeMat.dispose();
        if (!zCapUsed) zCarriageCapMat.dispose();
        if (!zRailUsed && !zRodUsed) zCarriageBodyMat.dispose();
      }

      // Z Leadscrews / Ballscrews / Z-Belts & Z-Motors (from geom.z_leadscrews)
      if (geom.z_leadscrews && geom.z_leadscrews.length > 0) {
        const leadSteelMat = new THREE.MeshStandardMaterial({
          color: 0xb8c4d4,
          metalness: 0.84,
          roughness: 0.28,
          transparent: toolheadPunchIn,
          opacity: toolheadPunchIn ? 0.16 : 1.0,
        });
        const ballChromeMat = new THREE.MeshStandardMaterial({
          color: 0xe6eef8,
          metalness: 0.94,
          roughness: 0.12,
          transparent: toolheadPunchIn,
          opacity: toolheadPunchIn ? 0.16 : 1.0,
        });
        const brassNutMat = new THREE.MeshStandardMaterial({
          color: 0xd4a017,
          metalness: 0.85,
          roughness: 0.24,
          transparent: toolheadPunchIn,
          opacity: toolheadPunchIn ? 0.16 : 1.0,
        });
        const pomNutMat = new THREE.MeshStandardMaterial({
          color: 0x1e2430,
          metalness: 0.12,
          roughness: 0.72,
          transparent: toolheadPunchIn,
          opacity: toolheadPunchIn ? 0.16 : 1.0,
        });
        const ballNutMat = new THREE.MeshStandardMaterial({
          color: 0x94a3b8,
          metalness: 0.88,
          roughness: 0.2,
          transparent: toolheadPunchIn,
          opacity: toolheadPunchIn ? 0.16 : 1.0,
        });
        const zBeltMat = new THREE.MeshStandardMaterial({
          color: 0x181c24,
          metalness: 0.08,
          roughness: 0.86,
          transparent: toolheadPunchIn,
          opacity: toolheadPunchIn ? 0.16 : 1.0,
        });
        const zPulleyMat = new THREE.MeshStandardMaterial({
          color: 0xdce4ee,
          metalness: 0.84,
          roughness: 0.22,
          transparent: toolheadPunchIn,
          opacity: toolheadPunchIn ? 0.16 : 1.0,
        });
        const couplerMat = new THREE.MeshStandardMaterial({
          color: 0x38bdf8,
          metalness: 0.75,
          roughness: 0.25,
          transparent: toolheadPunchIn,
          opacity: toolheadPunchIn ? 0.16 : 1.0,
        });
        const zMotorBodyMat = new THREE.MeshStandardMaterial({
          color: 0x1e2532,
          metalness: 0.72,
          roughness: 0.34,
          transparent: toolheadPunchIn,
          opacity: toolheadPunchIn ? 0.16 : 1.0,
        });
        const zMotorCapMat = new THREE.MeshStandardMaterial({
          color: 0x94a3b8,
          metalness: 0.82,
          roughness: 0.25,
          transparent: toolheadPunchIn,
          opacity: toolheadPunchIn ? 0.16 : 1.0,
        });

        let leadUsed = false;
        let ballUsed = false;
        let brassUsed = false;
        let pomUsed = false;
        let beltUsed = false;
        let couplerUsed = false;
        let zMotorUsed = false;

        const zMotorBodyGeo = new THREE.BoxGeometry(0.42, 0.38, 0.42);
        const zMotorCapGeo = new THREE.BoxGeometry(0.43, 0.045, 0.43);

        for (const zl of geom.z_leadscrews) {
          const p0 = toThree(zl.start_mm, cx, cy);
          const p1 = toThree(zl.end_mm, cx, cy);
          const mPos = toThree(zl.motor_pos_mm, cx, cy);
          const midX = 0.5 * (p0.x + p1.x);
          const midZ = 0.5 * (p0.z + p1.z);
          const nutY = zl.nut_z_mm * SCALE;
          const driveLen = p0.distanceTo(p1);
          if (driveLen < 1e-3) continue;

          const mechNice = zl.mechanism_id.replace(/_/g, " ").toUpperCase();
          const zDriveInspect: InspectedPartInfo = {
            key: `z_drive:${zl.id}`,
            name:
              zl.kind === "ballscrew"
                ? `Z Ballscrew (${mechNice})`
                : zl.kind === "belt"
                  ? `Z Drive Belt (${mechNice})`
                  : `Z Leadscrew (${mechNice})`,
            subsystem: "bed",
            partGroupId: "bed_support",
            role: "bed",
            material: zl.kind === "belt" ? "gt2_belt" : "steel",
            profile: zl.mechanism_id,
          };
          const zMotorInspect: InspectedPartInfo = {
            key: `z_motor:${zl.id}`,
            name: `NEMA17 Z-Axis Motor (${zl.id})`,
            subsystem: "bed",
            partGroupId: "bed_support",
            role: "bed",
            material: "nema17_stepper",
            profile: zl.mechanism_id,
          };

          // NEMA17 Z-stepper motor at motor_pos_mm
          zMotorUsed = true;
          const zMotGrp = new THREE.Group();
          const zMotBody = new THREE.Mesh(zMotorBodyGeo, zMotorBodyMat);
          zMotBody.position.y = -0.19;
          tagInspectable(zMotBody, zMotorInspect);
          zMotGrp.add(zMotBody);
          const zMotCap = new THREE.Mesh(zMotorCapGeo, zMotorCapMat);
          zMotCap.position.y = 0.02;
          zMotGrp.add(zMotCap);
          zMotGrp.position.copy(mPos);
          group.add(zMotGrp);

          if (zl.kind === "belt") {
            beltUsed = true;
            const beltW = Math.max((zl.diameter_mm || 6) * SCALE, 0.055);
            const beltT = 0.018;
            const pulleyR = 0.09;
            const strandGeo = new THREE.BoxGeometry(beltW, 1, beltT);

            const s0A = p0.clone().add(new THREE.Vector3(0, 0, pulleyR));
            const s1A = p1.clone().add(new THREE.Vector3(0, 0, pulleyR));
            const strandA = new THREE.Mesh(strandGeo, zBeltMat);
            orientBeamMesh(strandA, s0A, s1A);
            tagInspectable(strandA, zDriveInspect);
            group.add(strandA);

            const s0B = p0.clone().add(new THREE.Vector3(0, 0, -pulleyR));
            const s1B = p1.clone().add(new THREE.Vector3(0, 0, -pulleyR));
            const strandB = new THREE.Mesh(strandGeo, zBeltMat);
            orientBeamMesh(strandB, s0B, s1B);
            tagInspectable(strandB, zDriveInspect);
            group.add(strandB);

            // Bottom 80T drive pulley + top idler pulley
            const drive80t = new THREE.Mesh(
              new THREE.CylinderGeometry(0.18, 0.18, beltW * 1.15, 20),
              zPulleyMat
            );
            drive80t.rotation.z = Math.PI * 0.5;
            drive80t.position.set(p0.x, Math.max(p0.y + 0.08, mPos.y + 0.12), p0.z);
            tagInspectable(drive80t, zDriveInspect);
            group.add(drive80t);

            const topIdler = new THREE.Mesh(
              new THREE.CylinderGeometry(pulleyR * 1.15, pulleyR * 1.15, beltW * 1.2, 16),
              zPulleyMat
            );
            topIdler.rotation.z = Math.PI * 0.5;
            topIdler.position.copy(p1);
            group.add(topIdler);

            // Z-belt clamp block at nut_z_mm
            pomUsed = true;
            const clampBlock = new THREE.Mesh(
              new THREE.BoxGeometry(0.18, 0.22, pulleyR * 2.4),
              pomNutMat
            );
            clampBlock.position.set(midX, nutY, midZ);
            tagInspectable(clampBlock, zDriveInspect);
            group.add(clampBlock);
          } else {
            const isBall = zl.kind === "ballscrew" || zl.mechanism_id.includes("sfu");
            const rShaft = Math.max((zl.diameter_mm || (isBall ? 12 : 8)) * 0.5 * SCALE, 0.036);
            const shaftGeo = new THREE.CylinderGeometry(rShaft, rShaft, 1, 14);
            const shaftMat = isBall ? ballChromeMat : leadSteelMat;
            if (isBall) ballUsed = true;
            else leadUsed = true;

            const shaftMesh = new THREE.Mesh(shaftGeo, shaftMat);
            orientBeamMesh(shaftMesh, p0, p1);
            tagInspectable(shaftMesh, zDriveInspect);
            group.add(shaftMesh);

            // Nut block at nut_z_mm
            const nutGrp = new THREE.Group();
            if (isBall) {
              // SFU1204 flanged ballnut cylinder + BK10 angular-contact bearing block
              const nutCyl = new THREE.Mesh(
                new THREE.CylinderGeometry(rShaft * 2.0, rShaft * 2.0, 0.34, 18),
                ballNutMat
              );
              tagInspectable(nutCyl, zDriveInspect);
              nutGrp.add(nutCyl);
              const nutFlange = new THREE.Mesh(
                new THREE.CylinderGeometry(rShaft * 3.1, rShaft * 3.1, 0.06, 18),
                ballNutMat
              );
              nutFlange.position.y = 0.14;
              nutGrp.add(nutFlange);

              // BK10 bottom bearing block
              const bk10 = new THREE.Mesh(new THREE.BoxGeometry(0.32, 0.16, 0.26), pomNutMat);
              pomUsed = true;
              bk10.position.set(p0.x, p0.y + 0.1, p0.z);
              group.add(bk10);
            } else {
              const isBrass = zl.mechanism_id.includes("brass");
              const nutMat = isBrass ? brassNutMat : pomNutMat;
              if (isBrass) brassUsed = true;
              else pomUsed = true;

              const nutCyl = new THREE.Mesh(
                new THREE.CylinderGeometry(rShaft * 1.55, rShaft * 1.55, 0.18, 14),
                nutMat
              );
              tagInspectable(nutCyl, zDriveInspect);
              nutGrp.add(nutCyl);
              const nutFlange = new THREE.Mesh(
                new THREE.CylinderGeometry(rShaft * 2.6, rShaft * 2.6, 0.045, 16),
                nutMat
              );
              nutFlange.position.y = 0.07;
              nutGrp.add(nutFlange);
            }
            nutGrp.position.set(midX, nutY, midZ);
            group.add(nutGrp);

            // Flexible shaft coupler between Z motor and bottom of screw
            couplerUsed = true;
            const coupler = new THREE.Mesh(
              new THREE.CylinderGeometry(rShaft * 2.1, rShaft * 2.1, 0.22, 14),
              couplerMat
            );
            coupler.position.set(p0.x, Math.max(p0.y, mPos.y + 0.14), p0.z);
            group.add(coupler);
          }
        }

        if (!leadUsed) leadSteelMat.dispose();
        if (!ballUsed) {
          ballChromeMat.dispose();
          ballNutMat.dispose();
        }
        if (!brassUsed) brassNutMat.dispose();
        if (!pomUsed) pomNutMat.dispose();
        if (!beltUsed) {
          zBeltMat.dispose();
          zPulleyMat.dispose();
        }
        if (!couplerUsed) couplerMat.dispose();
        if (!zMotorUsed) {
          zMotorBodyGeo.dispose();
          zMotorCapGeo.dispose();
          zMotorBodyMat.dispose();
          zMotorCapMat.dispose();
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
        const punchPanelScale = toolheadPunchIn ? 0.14 : 1.0;
        const panelMat = new THREE.MeshStandardMaterial({
          color: panelColor,
          transparent: true,
          opacity: (isTransparentSheet ? 0.16 : 0.32) * punchPanelScale,
          depthWrite: false,
          roughness: isTransparentSheet ? 0.14 : 0.38,
          metalness: isTransparentSheet ? 0.1 : 0.42,
          side: THREE.DoubleSide,
        });

        const edgeMat = new THREE.LineBasicMaterial({
          color: isBoltedStructural ? 0x38bdf8 : 0x94a3b8,
          transparent: true,
          opacity: (isBoltedStructural ? 0.65 : 0.26) * punchPanelScale,
        });

        const topPanelInstalled =
          panels.top_panel !== false && sim.spec.frame.top_panel !== false;

        const sidePanelGeo = new THREE.BoxGeometry(tScene, spanZ, spanY);
        const rearPanelGeo = new THREE.BoxGeometry(spanX, spanZ, tScene);
        const sideEdgesGeo = new THREE.EdgesGeometry(sidePanelGeo);
        const rearEdgesGeo = new THREE.EdgesGeometry(rearPanelGeo);

        // Mount flush on the OUTSIDE face of Left (-X), Right (+X), Rear (+Y CAD = -Z Three.js), and Top (+Z CAD = +Y Three.js) extrusions
        const leftCenter = toThree([0, fd * 0.5, fh * 0.5], cx, cy);
        const rightCenter = toThree([fw, fd * 0.5, fh * 0.5], cx, cy);
        const rearCenter = toThree([fw * 0.5, fd, fh * 0.5], cx, cy);
        const topCenter = toThree([fw * 0.5, fd * 0.5, fh], cx, cy);

        const leftPos = leftCenter.clone().add(new THREE.Vector3(-(extHalf + tScene * 0.5), 0, 0));
        const rightPos = rightCenter.clone().add(new THREE.Vector3(extHalf + tScene * 0.5, 0, 0));
        const rearPos = rearCenter.clone().add(new THREE.Vector3(0, 0, -(extHalf + tScene * 0.5)));
        const topPos = topCenter.clone().add(new THREE.Vector3(0, extHalf + tScene * 0.5, 0));

        const leftBoltX = leftPos.x - tScene * 0.55;
        const rightBoltX = rightPos.x + tScene * 0.55;
        const rearBoltZ = rearPos.z - tScene * 0.55;
        const topBoltY = topPos.y + tScene * 0.55;

        const panelSpecs: Array<{
          bay: string;
          label: string;
          geo: THREE.BoxGeometry;
          edges: THREE.EdgesGeometry;
          pos: THREE.Vector3;
          opacityMul?: number;
        }> = [
          { bay: "left", label: "Left Enclosure Panel", geo: sidePanelGeo, edges: sideEdgesGeo, pos: leftPos },
          { bay: "right", label: "Right Enclosure Panel", geo: sidePanelGeo, edges: sideEdgesGeo, pos: rightPos },
          { bay: "rear", label: "Rear Enclosure Panel", geo: rearPanelGeo, edges: rearEdgesGeo, pos: rearPos },
        ];
        if (topPanelInstalled) {
          const topPanelGeo = new THREE.BoxGeometry(spanX, tScene, spanY);
          const topEdgesGeo = new THREE.EdgesGeometry(topPanelGeo);
          panelSpecs.push({
            bay: "top",
            label: "Top Enclosure Panel",
            geo: topPanelGeo,
            edges: topEdgesGeo,
            pos: topPos,
            opacityMul: 0.78,
          });
        }

        for (const ps of panelSpecs) {
          const wallMat = panelMat.clone();
          if (ps.opacityMul != null) wallMat.opacity *= ps.opacityMul;
          const mesh = new THREE.Mesh(ps.geo, wallMat);
          mesh.position.copy(ps.pos);
          mesh.renderOrder = 2;
          const bayGrp = classifyPartGroup(`bay_${ps.bay}`, "panel");
          const baySens = sim.sensitivity?.panel_bay_sensitivities?.[ps.bay];
          const bayStiff = stiffenById[bayGrp.id] ?? stiffenById[`bay_${ps.bay}`];
          if (!toolheadPunchIn) {
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
          }
          group.add(mesh);

          const outline = new THREE.LineSegments(ps.edges, edgeMat);
          outline.position.copy(ps.pos);
          group.add(outline);
        }
        panelMat.dispose();

        // Metallic perimeter bolt heads aligned directly on the extrusion T-slot centerlines
        if (isBoltedStructural && !toolheadPunchIn) {
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

          const addTopBolts = (yFace: number) => {
            for (let i = 0; i <= steps; i++) {
              const t = i / steps;
              const xAlong = xMin + t * (xMax - xMin);
              const zAlong = zFront + t * (zRear - zFront);
              // Top front & top rear extrusions
              boltCoords.push(new THREE.Vector3(xAlong, yFace, zFront));
              boltCoords.push(new THREE.Vector3(xAlong, yFace, zRear));
              // Top left & top right extrusions (skip corners)
              if (i > 0 && i < steps) {
                boltCoords.push(new THREE.Vector3(xMin, yFace, zAlong));
                boltCoords.push(new THREE.Vector3(xMax, yFace, zAlong));
              }
            }
          };

          addSideBolts(leftBoltX);
          addSideBolts(rightBoltX);
          addRearBolts(rearBoltZ);
          if (topPanelInstalled) {
            addTopBolts(topBoltY);
          }

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

      // Trident Backbone by Clee: solid full-footprint 3mm laser-cut powder-coated mild steel lower deck plate
      const showTridentBackbone =
        geom.architecture === "corexy" &&
        (Boolean(geom.trident_backbone?.active) ||
          (Boolean(sim.spec.frame.trident_backbone) &&
            !geom.panels?.is_flying_gantry &&
            sim.spec.bed.z_drive !== "belted_quad_z"));
      if (showTridentBackbone) {
        const [fw, fd] = geom.frame_dims_mm;
        const frameMember = geom.members.find((m) => m.role === "frame");
        const extHalfMm = (frameMember?.width_mm ?? 20) * 0.5;
        const plateThickMm = geom.trident_backbone?.thickness_mm ?? 3.0;
        const plateThickScene = Math.max(plateThickMm * SCALE, 0.026);

        const xL = -extHalfMm;
        const xR = fw + extHalfMm;
        const yF = -extHalfMm;
        const yB = fd + extHalfMm;

        // Small rear-center Z-rail clearance notch on the back edge
        const notchHalfW = Math.min(18, Math.max(10, fw * 0.04));
        const notchDepth = Math.min(extHalfMm + 10, Math.max(12, fd * 0.04));

        // 2D shape in (sx, sy) mapped via rotateX(-PI/2) -> (sx, sz, -sy) matching toThree
        const sx = (cadX: number) => (cadX - cx) * SCALE;
        const sy = (cadY: number) => (cadY - cy) * SCALE;

        const bbShape = new THREE.Shape();
        bbShape.moveTo(sx(xL), sy(yF));
        bbShape.lineTo(sx(xR), sy(yF));
        bbShape.lineTo(sx(xR), sy(yB));
        bbShape.lineTo(sx(fw * 0.5 + notchHalfW), sy(yB));
        bbShape.lineTo(sx(fw * 0.5 + notchHalfW), sy(yB - notchDepth));
        bbShape.lineTo(sx(fw * 0.5 - notchHalfW), sy(yB - notchDepth));
        bbShape.lineTo(sx(fw * 0.5 - notchHalfW), sy(yB));
        bbShape.lineTo(sx(xL), sy(yB));
        bbShape.closePath();

        const backboneGeo = new THREE.ExtrudeGeometry(bbShape, {
          depth: plateThickScene,
          bevelEnabled: false,
          curveSegments: 1,
        });
        backboneGeo.rotateX(-Math.PI / 2);
        backboneGeo.computeVertexNormals();

        const plateBaseY = toThree([0, 0, extHalfMm], cx, cy).y;
        const plateTopY = plateBaseY + plateThickScene;

        const backboneMat = new THREE.MeshStandardMaterial({
          color: 0x1e2530,
          metalness: 0.62,
          roughness: 0.36,
          transparent: toolheadPunchIn,
          opacity: toolheadPunchIn ? 0.22 : 1.0,
        });
        const backboneMesh = new THREE.Mesh(backboneGeo, backboneMat);
        backboneMesh.position.y = plateBaseY;

        const bbBaySens = sim.sensitivity?.panel_bay_sensitivities?.["backbone"];
        const bbStiff = stiffenById["trident_backbone"] ?? stiffenById["bay_backbone"];
        if (!toolheadPunchIn) {
          tagInspectable(backboneMesh, {
            key: "frame:trident_backbone",
            name: "Trident Backbone (Clee, 3mm Mild Steel)",
            subsystem: "frame",
            partGroupId: "trident_backbone",
            role: "panel",
            material: geom.trident_backbone?.material_name ?? "Powder-Coated Mild Steel",
            profile: `${plateThickMm.toFixed(1)}mm solid laser-cut plate`,
            complianceSharePct: bbBaySens?.compliance_share_pct ?? bbStiff?.compliance_share_pct,
            heatScore: bbBaySens ? getLensScore(bbBaySens, adjointLens) : undefined,
            estHzGain: bbStiff?.est_hz_gain,
            hzPer100g: bbBaySens?.hz_per_100g,
            whyItHurts: bbStiff?.why_it_hurts,
            actionHint: bbStiff?.action_hint,
          });
        }
        group.add(backboneMesh);

        // Crisp metallic edge highlight around the solid laser-cut steel plate
        const backboneEdgesGeo = new THREE.EdgesGeometry(backboneGeo, 28);
        const backboneEdgesMat = new THREE.LineBasicMaterial({
          color: 0x475569,
          transparent: true,
          opacity: toolheadPunchIn ? 0.2 : 0.85,
        });
        const backboneEdges = new THREE.LineSegments(backboneEdgesGeo, backboneEdgesMat);
        backboneEdges.position.y = plateBaseY;
        group.add(backboneEdges);

        // Surface details on top of the solid steel plate:
        // 3 circular NEMA-17 Z-motor bore/mounting rings + M3 bolts, 8 DIN-rail/wiring slots, and 4-side M5 perimeter bolts
        if (!toolheadPunchIn) {
          const padCentersMm: Array<[number, number]> =
            geom.z_leadscrews && geom.z_leadscrews.length > 0
              ? geom.z_leadscrews.map((zl) => [zl.motor_pos_mm[0], zl.motor_pos_mm[1]])
              : [
                  [extHalfMm + 28, extHalfMm + 28],
                  [fw * 0.5, fd - extHalfMm - 28],
                  [fw - extHalfMm - 28, extHalfMm + 28],
                ];

          // Circular NEMA-17 pilot bore recess (22mm dia) + machined mounting ring (39mm OD)
          const boreDiskGeo = new THREE.CylinderGeometry(0.112, 0.112, 0.008, 24);
          const boreDiskMat = new THREE.MeshStandardMaterial({
            color: 0x0b0f17,
            metalness: 0.2,
            roughness: 0.88,
          });
          const mountRingGeo = new THREE.RingGeometry(0.114, 0.198, 24);
          mountRingGeo.rotateX(-Math.PI / 2);
          const mountRingMat = new THREE.MeshStandardMaterial({
            color: 0x334155,
            metalness: 0.78,
            roughness: 0.3,
          });

          const m3BoltCoords: THREE.Vector3[] = [];
          const m3OffsetScene = 15.5 * SCALE; // NEMA-17 standard 31mm x 31mm M3 bolt pattern
          const m3BoltY = plateTopY + 0.01;

          for (const [pxMm, pyMm] of padCentersMm) {
            const pPos = toThree([pxMm, pyMm, extHalfMm + plateThickMm], cx, cy);
            const boreMesh = new THREE.Mesh(boreDiskGeo, boreDiskMat);
            boreMesh.position.set(pPos.x, plateTopY + 0.003, pPos.z);
            group.add(boreMesh);

            const ringMesh = new THREE.Mesh(mountRingGeo, mountRingMat);
            ringMesh.position.set(pPos.x, plateTopY + 0.005, pPos.z);
            group.add(ringMesh);

            for (const dx of [-m3OffsetScene, m3OffsetScene]) {
              for (const dz of [-m3OffsetScene, m3OffsetScene]) {
                m3BoltCoords.push(new THREE.Vector3(pPos.x + dx, m3BoltY, pPos.z + dz));
              }
            }
          }

          if (m3BoltCoords.length > 0) {
            const m3BoltGeo = new THREE.CylinderGeometry(0.02, 0.02, 0.016, 8);
            const m3BoltMat = new THREE.MeshStandardMaterial({
              color: 0xe2e8f0,
              metalness: 0.88,
              roughness: 0.22,
            });
            const m3BoltInst = new THREE.InstancedMesh(m3BoltGeo, m3BoltMat, m3BoltCoords.length);
            const m3Dummy = new THREE.Object3D();
            for (let i = 0; i < m3BoltCoords.length; i++) {
              m3Dummy.position.copy(m3BoltCoords[i]);
              m3Dummy.updateMatrix();
              m3BoltInst.setMatrixAt(i, m3Dummy.matrix);
            }
            m3BoltInst.instanceMatrix.needsUpdate = true;
            group.add(m3BoltInst);
          }

          // 8 narrow dark recessed DIN-rail / wiring slots matching Clee's West3D Trident Backbone layout:
          // 2 left inner, 2 right inner, 2 front-left/right horizontal, 2 rear-left/right horizontal
          const leftInnerX = Math.min(fw * 0.2, Math.max(extHalfMm + 48, fw * 0.18));
          const rightInnerX = fw - leftInnerX;
          const vSlotLenMm = Math.min(72, Math.max(36, fd * 0.19));
          const hSlotLenFrontMm = Math.min(76, Math.max(36, fw * 0.19));
          const hSlotLenRearMm = Math.min(68, Math.max(34, fw * 0.17));
          const slotNarrowMm = 7.5;
          const frontSlotY = Math.max(extHalfMm + 36, fd * 0.13);
          const rearSlotY = fd - frontSlotY;

          const slotSpecsMm: Array<{ cxMm: number; cyMm: number; wMm: number; dMm: number }> = [
            // 2 left inner vertical slots
            { cxMm: leftInnerX, cyMm: fd * 0.36, wMm: slotNarrowMm, dMm: vSlotLenMm },
            { cxMm: leftInnerX, cyMm: fd * 0.66, wMm: slotNarrowMm, dMm: vSlotLenMm },
            // 2 right inner vertical slots
            { cxMm: rightInnerX, cyMm: fd * 0.36, wMm: slotNarrowMm, dMm: vSlotLenMm },
            { cxMm: rightInnerX, cyMm: fd * 0.66, wMm: slotNarrowMm, dMm: vSlotLenMm },
            // 2 front-left / front-right horizontal slots
            { cxMm: fw * 0.34, cyMm: frontSlotY, wMm: hSlotLenFrontMm, dMm: slotNarrowMm },
            { cxMm: fw * 0.66, cyMm: frontSlotY, wMm: hSlotLenFrontMm, dMm: slotNarrowMm },
            // 2 rear-left / rear-right horizontal slots
            { cxMm: fw * 0.27, cyMm: rearSlotY, wMm: hSlotLenRearMm, dMm: slotNarrowMm },
            { cxMm: fw * 0.73, cyMm: rearSlotY, wMm: hSlotLenRearMm, dMm: slotNarrowMm },
          ];

          const slotRimGeo = new THREE.BoxGeometry(1, 0.004, 1);
          const slotRimMat = new THREE.MeshStandardMaterial({
            color: 0x3b4759,
            metalness: 0.72,
            roughness: 0.35,
          });
          const slotCutGeo = new THREE.BoxGeometry(1, 0.007, 1);
          const slotCutMat = new THREE.MeshStandardMaterial({
            color: 0x090d14,
            metalness: 0.15,
            roughness: 0.92,
          });
          const slotRimInst = new THREE.InstancedMesh(slotRimGeo, slotRimMat, slotSpecsMm.length);
          const slotCutInst = new THREE.InstancedMesh(slotCutGeo, slotCutMat, slotSpecsMm.length);
          const slotDummy = new THREE.Object3D();
          for (let i = 0; i < slotSpecsMm.length; i++) {
            const s = slotSpecsMm[i];
            const sPos = toThree([s.cxMm, s.cyMm, extHalfMm + plateThickMm], cx, cy);
            slotDummy.position.set(sPos.x, plateTopY + 0.002, sPos.z);
            slotDummy.scale.set(s.wMm * SCALE + 0.018, 1, s.dMm * SCALE + 0.018);
            slotDummy.updateMatrix();
            slotRimInst.setMatrixAt(i, slotDummy.matrix);

            slotDummy.position.set(sPos.x, plateTopY + 0.004, sPos.z);
            slotDummy.scale.set(s.wMm * SCALE, 1, s.dMm * SCALE);
            slotDummy.updateMatrix();
            slotCutInst.setMatrixAt(i, slotDummy.matrix);
          }
          slotRimInst.instanceMatrix.needsUpdate = true;
          slotCutInst.instanceMatrix.needsUpdate = true;
          group.add(slotRimInst);
          group.add(slotCutInst);

          // M5 perimeter bolts along ALL 4 bottom 2020 extrusion centerlines (front, rear, left, right)
          const bbBoltCoords: THREE.Vector3[] = [];
          const boltY = plateTopY + 0.014;
          const p00 = toThree([0, 0, 0], cx, cy);
          const p11 = toThree([fw, fd, 0], cx, cy);
          const sideSteps = 5;
          for (let i = 0; i <= sideSteps; i++) {
            const t = i / sideSteps;
            const zAlong = p00.z + t * (p11.z - p00.z);
            bbBoltCoords.push(new THREE.Vector3(p00.x, boltY, zAlong));
            bbBoltCoords.push(new THREE.Vector3(p11.x, boltY, zAlong));
          }
          const spanSteps = 6;
          for (let i = 1; i < spanSteps; i++) {
            const t = i / spanSteps;
            const cadX = t * fw;
            const xAlong = p00.x + t * (p11.x - p00.x);
            // Front bottom extrusion
            bbBoltCoords.push(new THREE.Vector3(xAlong, boltY, p00.z));
            // Rear bottom extrusion (skip rear-center Z-rail clearance notch)
            if (Math.abs(cadX - fw * 0.5) >= notchHalfW + 6) {
              bbBoltCoords.push(new THREE.Vector3(xAlong, boltY, p11.z));
            }
          }
          const bbBoltGeo = new THREE.CylinderGeometry(0.028, 0.028, 0.022, 8);
          const bbBoltMat = new THREE.MeshStandardMaterial({
            color: 0xdbe4ee,
            metalness: 0.86,
            roughness: 0.24,
          });
          const bbBoltInst = new THREE.InstancedMesh(bbBoltGeo, bbBoltMat, bbBoltCoords.length);
          const bbDummy = new THREE.Object3D();
          for (let i = 0; i < bbBoltCoords.length; i++) {
            bbDummy.position.copy(bbBoltCoords[i]);
            bbDummy.updateMatrix();
            bbBoltInst.setMatrixAt(i, bbDummy.matrix);
          }
          bbBoltInst.instanceMatrix.needsUpdate = true;
          group.add(bbBoltInst);
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
        toolheadPunchIn ||
        activeBedWarp ||
        viewMode === "fea_stress" ||
        viewMode === "modal" ||
        viewMode === "sensitivity",
      opacity: toolheadPunchIn
        ? 0.2
        : activeBedWarp
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

    // 3D Bed Mount Standoffs / Kinematic Pucks / Whopping Pochard Leaf-Springs / Springs / Silicone Columns
    // Always positioned directly underneath the heated bed plate via `bed_mount_points_mm`
    if (viewMode === "assembly" || viewMode === "sensitivity" || viewMode === "fea_stress") {
      const bedMountType: BedMountType =
        geom.bed_mount_type ?? sim.spec.bed.mount_type ?? "kinematic_3pt";
      const supportPtsThree: THREE.Vector3[] = [];

      if (geom.bed_mount_points_mm && geom.bed_mount_points_mm.length > 0) {
        for (const ptMm of geom.bed_mount_points_mm) {
          supportPtsThree.push(toThree(ptMm, cx, cy));
        }
      } else {
        const [bcx, bcy, bcz] = geom.bed_center_mm;
        const insetX = bedX * 0.36;
        const insetY = bedY * 0.36;
        const mountZ = bcz - (sim.spec.bed.plate_thickness_mm ?? 8) - 8;
        if (bedMountType === "kinematic_3pt" || bedMountType === "whopping_pochard_kinematic") {
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

        if (bedMountType === "whopping_pochard_kinematic") {
          // Whopping Pochard Titanium/Stainless Leaf-Spring Kinematic Flexure Tabs under the bed
          const leafBladeGeo = new THREE.BoxGeometry(0.32, 0.016, 0.13);
          const leafBladeMat = new THREE.MeshStandardMaterial({
            color: useSensBedTint ? bedColor : 0xd6dde8,
            emissive: useSensBedTint ? bedEmissive : 0x000000,
            emissiveIntensity: useSensBedTint ? bedEmissiveIntensity : 0,
            metalness: 0.88,
            roughness: 0.22,
          });
          const clampGeo = new THREE.BoxGeometry(0.14, 0.09, 0.16);
          const clampMat = new THREE.MeshStandardMaterial({
            color: useSensBedTint ? bedColor : 0x475569,
            metalness: 0.72,
            roughness: 0.34,
          });
          const pinSeatGeo = new THREE.CylinderGeometry(0.048, 0.056, 0.085, 14);
          const pinSeatMat = new THREE.MeshStandardMaterial({
            color: 0x38bdf8,
            metalness: 0.82,
            roughness: 0.2,
          });
          for (const sp of supportPtsThree) {
            const grp = new THREE.Group();
            // Orient leaf-spring blade radially toward bed center (thermal expansion flex axis)
            const angleToCenter = Math.atan2(bedBaseCenter.z - sp.z, bedBaseCenter.x - sp.x);
            grp.rotation.y = -angleToCenter;

            const topClamp = new THREE.Mesh(clampGeo, clampMat);
            topClamp.position.set(0.08, -0.045, 0);
            tagInspectable(topClamp, bedInspectInfo);
            grp.add(topClamp);

            const blade = new THREE.Mesh(leafBladeGeo, leafBladeMat);
            blade.position.set(0, -0.095, 0);
            tagInspectable(blade, bedInspectInfo);
            grp.add(blade);

            const seatPin = new THREE.Mesh(pinSeatGeo, pinSeatMat);
            seatPin.position.set(-0.09, -0.14, 0);
            grp.add(seatPin);

            grp.position.set(sp.x + bedDispDelta.x, bedBottomY + bedDispDelta.y, sp.z + bedDispDelta.z);
            group.add(grp);
          }
        } else if (bedMountType === "kinematic_3pt") {
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
        opacity: toolheadPunchIn ? 0.06 : viewMode === "toolpath" ? 0.45 : 0.25,
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
      const thStruct = geom.toolhead_structure;
      const fallbackMountMatId = geom.mount_material ?? sim.spec.toolhead.mount_material ?? "abs_printed";
      const carriageMountMatId =
        thStruct?.carriage_mount_material ??
        sim.spec.toolhead.carriage_mount_material ??
        fallbackMountMatId;
      const bodyMatId =
        thStruct?.body_material ?? sim.spec.toolhead.body_material ?? fallbackMountMatId;
      const hotendMountMatId =
        thStruct?.hotend_mount_material ??
        sim.spec.toolhead.hotend_mount_material ??
        bodyMatId;
      const hotendMountStyle =
        thStruct?.hotend_mount_style ?? sim.spec.toolhead.hotend.mount_style ?? "rigid_4bolt_m3";
      const hotendPresetId =
        thStruct?.hotend_preset_id ?? sim.spec.toolhead.hotend.preset_id ?? null;
      const hotendStiffness =
        thStruct?.hotend_stiffness_n_per_mm ??
        sim.spec.toolhead.hotend.structural_stiffness_n_per_mm ??
        450;

      // Per-part structural thicknesses (clamped to engineering UI ranges)
      const carPlateMm = Math.max(
        2.0,
        Math.min(
          15.0,
          thStruct?.carriage_plate_thickness_mm ??
            sim.spec.toolhead.carriage_plate_thickness_mm ??
            5.0
        )
      );
      const bodyWallMm = Math.max(
        2.0,
        Math.min(
          12.0,
          thStruct?.body_wall_thickness_mm ?? sim.spec.toolhead.body_wall_thickness_mm ?? 4.0
        )
      );
      const hotendClampMm = Math.max(
        3.0,
        Math.min(
          20.0,
          thStruct?.hotend_mount_thickness_mm ??
            sim.spec.toolhead.hotend_mount_thickness_mm ??
            8.0
        )
      );

      const backplateApp = getToolheadMaterialAppearance(carriageMountMatId);
      const bodyApp = getToolheadMaterialAppearance(bodyMatId);
      const clampApp = getToolheadMaterialAppearance(hotendMountMatId);

      // Mode-aware colors & emissive glows for all 3 articulated structural stages
      const mgnBlockColorObj = new THREE.Color(0xa3b1c2);
      const mgnBlockEmissive = new THREE.Color(0x000000);
      let mgnBlockEmissiveInt = 0;

      const backplateColorObj = new THREE.Color(backplateApp.hex);
      const backplateEmissive = new THREE.Color(0x000000);
      let backplateEmissiveInt = 0;

      const bodyColorObj = new THREE.Color(bodyApp.hex);
      const bodyEmissive = new THREE.Color(0x000000);
      let bodyEmissiveInt = 0;

      const clampColorObj = new THREE.Color(clampApp.hex);
      const clampEmissive = new THREE.Color(0x000000);
      let clampEmissiveInt = 0;

      const hotendShellColorObj = new THREE.Color(
        hotendMountStyle === "chube_6bolt_m3_18pcd"
          ? 0xb4bfcc
          : hotendMountStyle === "rigid_4bolt_m3"
            ? 0xb91c1c
            : 0xc8d2de
      );
      const hotendShellEmissive = new THREE.Color(0x000000);
      let hotendShellEmissiveInt = 0;

      const heaterBlockColorObj = new THREE.Color(
        sim.spec.toolhead.hotend.block_material.includes("copper") ? 0xd97742 : 0xc4cfdc
      );
      const isHardenedNozzle = sim.spec.toolhead.hotend.nozzle_material === "hardened_steel";
      const nozzleColorObj = new THREE.Color(isHardenedNozzle ? 0x4b5563 : 0xf59e0b);
      const nozzleEmissive = new THREE.Color(isHardenedNozzle ? 0x000000 : 0x92400e);
      let nozzleEmissiveInt = isHardenedNozzle ? 0 : 0.08;

      const isHeatMode = viewMode === "modal" || viewMode === "fea_stress" || viewMode === "sensitivity";

      // 3-way physical compliance split: MGN rail bearings vs Backplate elastic bending vs Body elastic flex
      const stageStiffnesses = thStruct?.stage_stiffnesses_n_per_mm;
      const kRailStage = stageStiffnesses?.rail_carriage_n_per_mm;
      const kPlateStage = stageStiffnesses?.backplate_only_n_per_mm;
      const kCarStage = stageStiffnesses?.carriage_mount_n_per_mm ?? 400;
      const kBodyStage = stageStiffnesses?.body_n_per_mm ?? 400;
      let stageRailFrac = 0;
      let stagePlateFrac = 0.5;
      let stageBodyFrac = 0.5;
      if (
        kRailStage != null &&
        Number.isFinite(kRailStage) &&
        kRailStage > 0 &&
        kPlateStage != null &&
        Number.isFinite(kPlateStage) &&
        kPlateStage > 0
      ) {
        const cRail = 1 / Math.max(kRailStage, 1);
        const cPlate = 1 / Math.max(kPlateStage, 1);
        const cBody = 1 / Math.max(kBodyStage, 1);
        const cTot = cRail + cPlate + cBody;
        stageRailFrac = cRail / cTot;
        stagePlateFrac = cPlate / cTot; // No artificial floor: stiff 5mm Al drops to ~2% and stays visibly flat!
        stageBodyFrac = cBody / cTot;
      } else {
        const cCar = 1 / Math.max(kCarStage, 1);
        const cBody = 1 / Math.max(kBodyStage, 1);
        stageRailFrac = 0;
        stagePlateFrac = cCar / (cCar + cBody);
        stageBodyFrac = 1 - stagePlateFrac;
      }

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

        sensitivityColorInto(mountScore, mgnBlockColorObj);
        sensitivityColorInto(mountScore, backplateColorObj);
        const absMount = Math.min(1, Math.abs(mountScore));
        if (absMount > 0.15) {
          backplateEmissive.copy(backplateColorObj);
          backplateEmissiveInt = absMount * 0.45;
          mgnBlockEmissive.copy(mgnBlockColorObj);
          mgnBlockEmissiveInt = absMount * 0.35;
        }

        sensitivityColorInto(cogMassScore, bodyColorObj);
        const absCog = Math.min(1, Math.abs(cogMassScore));
        if (absCog > 0.12) {
          bodyEmissive.copy(bodyColorObj);
          bodyEmissiveInt = absCog * 0.42;
        }

        sensitivityColorInto(stemScore, clampColorObj);
        sensitivityColorInto(stemScore, hotendShellColorObj);
        sensitivityColorInto(stemScore, heaterBlockColorObj);
        const absStem = Math.min(1, Math.abs(stemScore));
        if (absStem > 0.15) {
          clampEmissive.copy(clampColorObj);
          clampEmissiveInt = absStem * 0.45;
          hotendShellEmissive.copy(hotendShellColorObj);
          hotendShellEmissiveInt = absStem * 0.48;
          sensitivityColorInto(stemScore, nozzleColorObj);
          nozzleEmissive.copy(nozzleColorObj);
          nozzleEmissiveInt = absStem * 0.48;
        } else {
          nozzleColorObj.set(0x64748b);
          nozzleEmissive.set(0x000000);
          nozzleEmissiveInt = 0;
        }
      } else if (viewMode === "modal") {
        const thScalar = getModalNodeScalar(thNodeId);
        const cogScalar = getModalNodeScalar(cogNodeId);
        const nozScalar = getModalNodeScalar(nozNodeId);
        // Weight backplate modal color by its true elastic bending share (+ subtle rail-roll cue)
        // so a rock-solid Aluminum backplate stays cool while a soft ABS body glows warm!
        const plateColorWeight = Math.min(1, stagePlateFrac + stageRailFrac * 0.25);
        const plateColorScalar = thScalar + (cogScalar - thScalar) * plateColorWeight;
        const s2 = 0.5 * (cogScalar + nozScalar);

        stressColorInto(thScalar, mgnBlockColorObj);
        if (thScalar > 0.35) {
          mgnBlockEmissive.copy(mgnBlockColorObj);
          mgnBlockEmissiveInt = (thScalar - 0.25) * 0.38;
        }

        stressColorInto(plateColorScalar, backplateColorObj);
        if (plateColorScalar > 0.35) {
          backplateEmissive.copy(backplateColorObj);
          backplateEmissiveInt = (plateColorScalar - 0.25) * 0.42;
        }

        stressColorInto(cogScalar, bodyColorObj);
        if (cogScalar > 0.3) {
          bodyEmissive.copy(bodyColorObj);
          bodyEmissiveInt = (cogScalar - 0.2) * 0.4;
        }

        stressColorInto(s2, clampColorObj);
        stressColorInto(s2, hotendShellColorObj);
        stressColorInto(nozScalar, heaterBlockColorObj);
        if (s2 > 0.35) {
          clampEmissive.copy(clampColorObj);
          clampEmissiveInt = (s2 - 0.25) * 0.42;
          hotendShellEmissive.copy(hotendShellColorObj);
          hotendShellEmissiveInt = (s2 - 0.25) * 0.48;
        }

        stressColorInto(nozScalar, nozzleColorObj);
        nozzleEmissive.copy(nozzleColorObj);
        nozzleEmissiveInt = 0.22 + nozScalar * 0.52;
      } else if (viewMode === "fea_stress") {
        const mountStress =
          stressMap["toolhead_mount"] ?? stressMap["toolhead_cog_spring"] ?? 0.35;
        const stemStress =
          stressMap["nozzle_stem"] ?? stressMap["nozzle_stem_spring"] ?? mountStress * 0.8;
        stressColorInto(mountStress * 0.7, mgnBlockColorObj);
        stressColorInto(mountStress, backplateColorObj);
        stressColorInto(0.5 * (mountStress + stemStress), bodyColorObj);
        stressColorInto(stemStress, clampColorObj);
        stressColorInto(stemStress, hotendShellColorObj);
        stressColorInto(stemStress, heaterBlockColorObj);
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
        name: "X-Carriage MGN Block & Backplate",
        subsystem: "toolhead",
        partGroupId: "toolhead_mount",
        role: "toolhead",
        material: `${carriageMountMatId} (${carPlateMm.toFixed(1)}mm)`,
        profile: `${backplateApp.shortLabel} · ${carPlateMm.toFixed(1)}mm plate · ${Math.round(sim.spec.toolhead.internal_stiffness_n_per_mm ?? 450)} N/mm`,
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
        name: `X-Carriage Backplate (${backplateApp.shortLabel} · ${carPlateMm.toFixed(1)}mm)`,
      };
      const hotendStyleShort =
        hotendMountStyle === "chube_6bolt_m3_18pcd"
          ? "Chube 6× M3 Ti Shell"
          : hotendMountStyle === "dragon_4bolt_m25"
            ? "Dragon 4× M2.5 Standoffs"
            : hotendMountStyle === "groove_mount"
              ? "V6 Groove Mount"
              : "4× M3 Rigid Flange";
      const nozStemInspectInfo: InspectedPartInfo = {
        key: "nozzle_stem",
        name: `${sim.spec.toolhead.hotend.name || "Hotend Assembly"} (${hotendStyleShort})`,
        subsystem: "toolhead",
        partGroupId: "toolhead_mount",
        role: "toolhead",
        material: `${clampApp.shortLabel} ${hotendClampMm.toFixed(1)}mm clamp · ${sim.spec.toolhead.hotend.heatbreak_material}`,
        profile: `${Math.round(hotendStiffness)} N/mm · Clamp ${hotendClampMm.toFixed(1)}mm · Drop Z ${Math.round(Math.abs(sim.spec.toolhead.nozzle_offset_mm?.[2] ?? 62))} mm`,
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
        name: thMassLight?.label ?? `Toolhead Body / Extruder (${bodyApp.shortLabel} · ${bodyWallMm.toFixed(1)}mm)`,
        subsystem: "toolhead",
        partGroupId: "toolhead_assembly",
        role: "toolhead",
        material: `${bodyApp.shortLabel} · ${bodyWallMm.toFixed(1)}mm wall · ${Math.round(sim.spec.toolhead.total_mass_g)} g`,
        profile: `Wall ${bodyWallMm.toFixed(1)}mm · CoG Y/Z: ${Math.round(Math.abs(sim.spec.toolhead.cog_offset_mm?.[1] ?? 18))}/${Math.round(Math.abs(sim.spec.toolhead.cog_offset_mm?.[2] ?? 25))} mm`,
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
        name: `Hotend Nozzle (${sim.spec.toolhead.hotend.nozzle_diameter_mm}mm · ${Math.round(hotendStiffness)} N/mm)`,
      };

      // =====================================================================
      // EXACT MECHANICAL REFERENCE DIMENSIONS (CAD Z-up -> Three.js Y-up, -Y_cad -> +Z_three)
      // =====================================================================
      const isDeltaArch = geom.architecture === "delta";
      const xGantryBeam = geom.members.find((m) => m.id === "x_gantry_l" || m.id === "x_gantry_r");
      const beamDepthMm = xGantryBeam ? Math.max(xGantryBeam.width_mm, 6) : 20;
      const beamHeightMm = xGantryBeam ? Math.max(xGantryBeam.height_mm, 6) : 20;
      const beamHalfZ = 0.5 * beamDepthMm * SCALE; // half depth in CAD Y (Three.js Z)
      const beamHalfY = 0.5 * beamHeightMm * SCALE; // half height in CAD Z (Three.js Y)

      const lrTool = geom.linear_rails;
      const xRailMountTool: XRailMountType =
        lrTool?.x_rail_mount ??
        sim.spec.gantry.x_rail_mount ??
        (sim.spec.gantry.dual_x_rails ? "dual_top_bottom" : "front");
      const xRailProfTool = lrTool?.x_rail_profile ?? sim.spec.gantry.x_rail_profile ?? "mgn12_rail";
      const isMgn9Tool = xRailProfTool.includes("mgn9");
      const isMgn15Tool = xRailProfTool.includes("mgn15");
      const railWidthTool =
        (lrTool?.x_rail_width_mm ?? (isMgn9Tool ? 9.0 : isMgn15Tool ? 15.0 : 12.0)) * SCALE;
      const railThickTool =
        (lrTool?.x_rail_height_mm ?? (isMgn9Tool ? 6.5 : isMgn15Tool ? 10.0 : 8.0)) * SCALE;

      // HIWIN MGN12H / MGN9H / MGN15H authentic datasheet carriage dimensions (ZERO beam clipping!)
      const carLenX = (isMgn9Tool ? 39.9 : isMgn15Tool ? 58.8 : 45.4) * SCALE;
      const carWidth = (isMgn9Tool ? 20.0 : isMgn15Tool ? 32.0 : 27.0) * SCALE;
      const carClear = (isMgn9Tool ? 2.0 : isMgn15Tool ? 4.0 : 3.0) * SCALE;
      const carThick = (isMgn9Tool ? 8.0 : isMgn15Tool ? 12.0 : 10.0) * SCALE;
      const carTotalH = carClear + carThick; // 13.0 mm MGN12H, 10.0 mm MGN9H, 16.0 mm MGN15H
      const carCenterOff = carClear + carThick * 0.5; // 8.0 mm MGN12H, 6.0 mm MGN9H, 10.0 mm MGN15H
      const wiperLen = (isMgn9Tool ? 4.4 : isMgn15Tool ? 6.4 : 5.2) * SCALE;
      const steelBodyLen = carLenX - 2 * wiperLen;
      const boltSpanX = (isMgn9Tool ? 16.0 : isMgn15Tool ? 25.0 : 20.0) * SCALE;
      const boltSpanW = (isMgn9Tool ? 15.0 : isMgn15Tool ? 25.0 : 20.0) * SCALE;

      const baseTh = basePos[thNodeId] ?? thAnchor;
      const baseCog = basePos[cogNodeId] ?? cogAnchor;
      const baseNoz = basePos[nozNodeId] ?? nozAnchor;
      const nozRelY = baseNoz.y - baseTh.y; // negative (mm below X-beam)
      const hotendLen = Math.max(Math.min(Math.abs(nozRelY) * 0.72, 0.56), 0.34);
      const collarYRelToTh = nozRelY + hotendLen;

      // Backplate front face Z coordinate (relative to X-beam centerline) for each x_rail_mount
      const plateThick = carPlateMm * SCALE; // User-tunable 2..15 mm X-carriage backplate
      const clampH = hotendClampMm * SCALE; // User-tunable 3..20 mm hotend mount collar/clamp height
      const plateW = Math.max(0.48, carLenX + 0.054); // >= 48 mm wide, always clears MGN9/12/15 wipers cleanly
      const plateH = 0.38; // 38 mm tall (extended if needed to support hotend mount)
      let backplateFrontZ = 0;
      if (!isDeltaArch) {
        if (xRailMountTool === "front") {
          backplateFrontZ = beamHalfZ + carTotalH + plateThick; // clears MGN carriage even at 15 mm!
        } else {
          const plateInnerZ = Math.max(beamHalfZ + 0.03, carWidth * 0.5 + 0.006);
          backplateFrontZ = plateInnerZ + plateThick; // clears BOTH beam front & top/bot carriage!
        }
      }

      // Compute exact rest positions for Stage 2 (Hotend Mount Collar & Extruder) and Stage 3 (Nozzle Tip)
      // so the toolhead ALWAYS clears the backplate with ZERO clipping while tracking exact FEA/Modal motion!
      const baseNozRelZ = baseNoz.z - baseTh.z;
      const targetHotendRelZ = isDeltaArch
        ? baseNozRelZ
        : Math.max(baseNozRelZ, backplateFrontZ + 0.16);
      const rearBridgeZ = isDeltaArch ? 0 : -(targetHotendRelZ - backplateFrontZ);

      const stage3RestPos = new THREE.Vector3(baseNoz.x, baseNoz.y, baseTh.z + targetHotendRelZ);
      const stage2RestPos = new THREE.Vector3(
        baseNoz.x,
        baseNoz.y + hotendLen,
        baseTh.z + targetHotendRelZ
      );
      const stage2Offset = stage2RestPos.clone().sub(baseCog);
      const stage3Offset = stage3RestPos.clone().sub(baseNoz);

      // Shared mechanical hardware materials
      const mgnBlockMat = new THREE.MeshStandardMaterial({
        color: mgnBlockColorObj,
        emissive: mgnBlockEmissive,
        emissiveIntensity: mgnBlockEmissiveInt,
        metalness: isHeatMode ? 0.35 : 0.86,
        roughness: isHeatMode ? 0.38 : 0.22,
      });
      const darkScrewMat = new THREE.MeshStandardMaterial({
        color: 0x1f2937,
        metalness: 0.78,
        roughness: 0.34,
      });
      const boltHeadMat = new THREE.MeshStandardMaterial({
        color: 0xe2e8f0,
        metalness: 0.92,
        roughness: 0.16,
      });
      const backplateMat = new THREE.MeshStandardMaterial({
        color: backplateColorObj,
        emissive: backplateEmissive,
        emissiveIntensity: backplateEmissiveInt,
        metalness: isHeatMode ? 0.3 : backplateApp.metalness,
        roughness: isHeatMode ? 0.36 : backplateApp.roughness,
      });
      const bodyMat = new THREE.MeshStandardMaterial({
        color: bodyColorObj,
        emissive: bodyEmissive,
        emissiveIntensity: bodyEmissiveInt,
        metalness: isHeatMode ? 0.28 : bodyApp.metalness,
        roughness: isHeatMode ? 0.38 : bodyApp.roughness,
      });
      const clampMat = new THREE.MeshStandardMaterial({
        color: clampColorObj,
        emissive: clampEmissive,
        emissiveIntensity: clampEmissiveInt,
        metalness: isHeatMode ? 0.3 : clampApp.metalness,
        roughness: isHeatMode ? 0.36 : clampApp.roughness,
      });
      const accentTrimColor = isHeatMode
        ? bodyColorObj
        : bodyMatId === "abs_printed"
          ? new THREE.Color(0xdc2626) // Authentic Voron-red accent strip on charcoal ABS
          : new THREE.Color(0x475569);
      const accentTrimMat = new THREE.MeshStandardMaterial({
        color: accentTrimColor,
        emissive: bodyEmissive,
        emissiveIntensity: bodyEmissiveInt * 0.7,
        metalness: isHeatMode ? 0.25 : 0.2,
        roughness: 0.42,
      });

      // =====================================================================
      // STAGE 1A: MGN Carriage Block(s) on Exact Rail Face (or Hex Delta Effector) at thAnchor
      // =====================================================================
      const thGroup = new THREE.Group();
      const railShoeGroup = new THREE.Group();
      const backplateGroup = new THREE.Group();
      const isModalSubdiv = viewMode === "modal";

      const m3SocketGeo = new THREE.CylinderGeometry(0.018, 0.018, 0.018, 8);

      if (isDeltaArch) {
        // Delta architecture: Hexagonal CNC/CF Delta Effector Platform centered at thAnchor
        const effRadius = 0.32;
        const effThick = Math.max(0.035, carPlateMm * SCALE);
        const effPlate = new THREE.Mesh(
          new THREE.CylinderGeometry(effRadius, effRadius, effThick, 6, isModalSubdiv ? 2 : 1),
          backplateMat
        );
        tagInspectable(effPlate, mountStrutInspectInfo);
        backplateGroup.add(effPlate);

        // Central stainless bearing/bore collar
        const effHub = new THREE.Mesh(
          new THREE.CylinderGeometry(0.145, 0.145, effThick + 0.014, 18, isModalSubdiv ? 2 : 1),
          mgnBlockMat
        );
        tagInspectable(effHub, carriageInspectInfo);
        backplateGroup.add(effHub);

        // 3 pairs of ball-joint clevis bosses at 120 deg toward delta arm endpoints
        const clevisGeo = isModalSubdiv
          ? new THREE.BoxGeometry(0.11, 0.05, 0.075, 2, 2, 2)
          : new THREE.BoxGeometry(0.11, 0.05, 0.075);
        const ballGeo = new THREE.SphereGeometry(0.028, 10, 10);
        for (let i = 0; i < 3; i++) {
          const ang = (i / 3) * Math.PI * 2 + Math.PI / 6;
          for (const side of [-1, 1]) {
            const tang = ang + Math.PI * 0.5;
            const cxOff = Math.cos(ang) * (effRadius - 0.02) + Math.cos(tang) * side * 0.095;
            const czOff = Math.sin(ang) * (effRadius - 0.02) + Math.sin(tang) * side * 0.095;
            const clevis = new THREE.Mesh(clevisGeo, backplateMat);
            clevis.position.set(cxOff, 0, czOff);
            clevis.rotation.y = -ang;
            backplateGroup.add(clevis);
            const ball = new THREE.Mesh(ballGeo, boltHeadMat);
            ball.position.set(
              cxOff + Math.cos(ang) * 0.045,
              0,
              czOff + Math.sin(ang) * 0.045
            );
            thGroup.add(ball);
          }
        }
      } else {
        // CoreXY & Cartesian Bedslinger: HIWIN MGN Carriage Block(s) flush on exact X-rail face
        const wiperMat = new THREE.MeshStandardMaterial({
          color: isHeatMode ? mgnBlockColorObj : 0x15803d,
          emissive: mgnBlockEmissive,
          emissiveIntensity: mgnBlockEmissiveInt * 0.8,
          metalness: 0.16,
          roughness: 0.52,
        });
        const m3BossGeo = new THREE.CylinderGeometry(0.019, 0.019, 0.012, 10);
        const railPadMat =
          viewMode !== "assembly"
            ? new THREE.MeshStandardMaterial({
                color: isHeatMode ? mgnBlockColorObj : 0x64748b,
                metalness: 0.72,
                roughness: 0.34,
                transparent: true,
                opacity: 0.82,
              })
            : null;

        const addMgnCarriage = (face: "front" | "top" | "bottom") => {
          const isFrontFace = face === "front";
          const ySign = face === "top" ? 1 : -1;

          const bodyGeo = isFrontFace
            ? new THREE.BoxGeometry(steelBodyLen, carWidth, carThick)
            : new THREE.BoxGeometry(steelBodyLen, carThick, carWidth);
          const wiperGeo = isFrontFace
            ? new THREE.BoxGeometry(wiperLen, carWidth - 0.008, carThick - 0.006)
            : new THREE.BoxGeometry(wiperLen, carThick - 0.006, carWidth - 0.008);

          const cx = 0;
          const cy = isFrontFace ? 0 : ySign * (beamHalfY + carCenterOff);
          const cz = isFrontFace ? beamHalfZ + carCenterOff : 0;

          // Brushed stainless steel central carriage block
          const steelBlock = new THREE.Mesh(bodyGeo, mgnBlockMat);
          steelBlock.position.set(cx, cy, cz);
          tagInspectable(steelBlock, carriageInspectInfo);
          thGroup.add(steelBlock);

          // Left & right authentic HIWIN dark-green wiper end-caps
          for (const sx of [-1, 1]) {
            const wiper = new THREE.Mesh(wiperGeo, wiperMat);
            wiper.position.set(sx * (steelBodyLen * 0.5 + wiperLen * 0.5), cy, cz);
            tagInspectable(wiper, carriageInspectInfo);
            thGroup.add(wiper);
          }

          // 4x dark metallic M3 mounting screw bosses on the carriage mounting face
          for (const sx of [-0.5, 0.5]) {
            for (const sw of [-0.5, 0.5]) {
              const boss = new THREE.Mesh(m3BossGeo, darkScrewMat);
              if (isFrontFace) {
                boss.rotation.x = Math.PI * 0.5;
                boss.position.set(sx * boltSpanX, sw * boltSpanW, beamHalfZ + carTotalH + 0.002);
              } else {
                boss.position.set(
                  sx * boltSpanX,
                  ySign * (beamHalfY + carTotalH + 0.002),
                  sw * boltSpanW
                );
              }
              thGroup.add(boss);
            }
          }

          // In FEA/Modal/Sensitivity (where static full rail is hidden so beam bends cleanly),
          // add a compact local rail-shoe pad under the carriage so it rides flush on the deformed beam!
          if (railPadMat) {
            const padGeo = isFrontFace
              ? new THREE.BoxGeometry(carLenX + 0.14, railWidthTool, railThickTool)
              : new THREE.BoxGeometry(carLenX + 0.14, railThickTool, railWidthTool);
            const pad = new THREE.Mesh(padGeo, railPadMat);
            if (isFrontFace) {
              pad.position.set(0, 0, beamHalfZ + railThickTool * 0.5);
            } else {
              pad.position.set(0, ySign * (beamHalfY + railThickTool * 0.5), 0);
            }
            railShoeGroup.add(pad);
          }
        };

        if (xRailMountTool === "front") {
          addMgnCarriage("front");
        } else if (xRailMountTool === "top") {
          addMgnCarriage("top");
        } else if (xRailMountTool === "bottom") {
          addMgnCarriage("bottom");
        } else {
          // dual_top_bottom: TWO MGN carriage blocks, one on top rail & one on bottom rail!
          addMgnCarriage("top");
          addMgnCarriage("bottom");
        }

        // =====================================================================
        // STAGE 1B: X-Carriage Backplate (flush on MGN carriage face, ZERO beam/rail overlap!)
        // =====================================================================
        const beltTabGeo = isModalSubdiv
          ? new THREE.BoxGeometry(0.088, 0.15, 0.044, 2, 4, 2)
          : new THREE.BoxGeometry(0.088, 0.15, 0.044);
        const beltClampCapGeo = new THREE.BoxGeometry(0.068, 0.12, 0.018);
        const addBeltClampTabs = (zMid: number, yMid: number) => {
          for (const sx of [-1, 1]) {
            const tab = new THREE.Mesh(beltTabGeo, backplateMat);
            tab.position.set(sx * (plateW * 0.5 + 0.042), yMid, zMid);
            tagInspectable(tab, mountStrutInspectInfo);
            backplateGroup.add(tab);

            const cap = new THREE.Mesh(beltClampCapGeo, darkScrewMat);
            cap.position.set(sx * (plateW * 0.5 + 0.044), yMid, zMid + 0.026);
            backplateGroup.add(cap);

            for (const sy of [-0.036, 0.036]) {
              const screw = new THREE.Mesh(m3SocketGeo, boltHeadMat);
              screw.rotation.x = Math.PI * 0.5;
              screw.position.set(sx * (plateW * 0.5 + 0.044), yMid + sy, zMid + 0.038);
              backplateGroup.add(screw);
            }
          }
        };

        const backplateBotMinY = Math.min(-beamHalfY - 0.09, collarYRelToTh - 0.015);
        const backplateTopMinY = Math.max(beamHalfY + 0.09, plateH * 0.5, collarYRelToTh + clampH + 0.02);

        if (xRailMountTool === "front") {
          const zRear = beamHalfZ + carTotalH;
          const zCenter = zRear + plateThick * 0.5;
          const yBot = Math.min(-plateH * 0.5, backplateBotMinY);
          const yTop = backplateTopMinY;
          const hFront = yTop - yBot;
          const mainPlate = new THREE.Mesh(
            isModalSubdiv
              ? new THREE.BoxGeometry(plateW, hFront, plateThick, 8, 12, 2)
              : new THREE.BoxGeometry(plateW, hFront, plateThick),
            backplateMat
          );
          mainPlate.position.set(0, 0.5 * (yTop + yBot), zCenter);
          tagInspectable(mainPlate, mountStrutInspectInfo);
          backplateGroup.add(mainPlate);

          // 4x stainless M3 socket-head mounting bolts into MGN carriage
          for (const sx of [-0.5, 0.5]) {
            for (const sw of [-0.5, 0.5]) {
              const bolt = new THREE.Mesh(m3SocketGeo, boltHeadMat);
              bolt.rotation.x = Math.PI * 0.5;
              bolt.position.set(sx * boltSpanX, sw * boltSpanW, backplateFrontZ + 0.008);
              backplateGroup.add(bolt);
            }
          }
          addBeltClampTabs(zCenter - 0.004, 0);
        } else {
          const plateInnerZ = backplateFrontZ - plateThick;
          const zPlateCenter = plateInnerZ + plateThick * 0.5;
          const topFlangeY0 = beamHalfY + carTotalH;
          const flangeDepthZ = backplateFrontZ + carWidth * 0.5;
          const flangeCenterZ = 0.5 * (backplateFrontZ - carWidth * 0.5);

          const addHorizFlange = (ySign: 1 | -1) => {
            const flange = new THREE.Mesh(
              isModalSubdiv
                ? new THREE.BoxGeometry(plateW, plateThick, flangeDepthZ, 6, 2, 10)
                : new THREE.BoxGeometry(plateW, plateThick, flangeDepthZ),
              backplateMat
            );
            flange.position.set(0, ySign * (topFlangeY0 + plateThick * 0.5), flangeCenterZ);
            tagInspectable(flange, mountStrutInspectInfo);
            backplateGroup.add(flange);

            // 4x M3 mounting bolts through flange into MGN carriage
            for (const sx of [-0.5, 0.5]) {
              for (const sw of [-0.5, 0.5]) {
                const bolt = new THREE.Mesh(m3SocketGeo, boltHeadMat);
                bolt.position.set(
                  sx * boltSpanX,
                  ySign * (topFlangeY0 + plateThick + 0.008),
                  sw * boltSpanW
                );
                backplateGroup.add(bolt);
              }
            }

            // Side stiffening cheeks positioned strictly OUTSIDE the MGN carriage X-width and IN FRONT of the X-beam
            const cheekH = 0.095;
            const cheekThickX = 0.024;
            const cheekRearZ = Math.min(beamHalfZ + 0.008, plateInnerZ - 0.02);
            const cheekDepth = Math.max(plateInnerZ - cheekRearZ, 0.024);
            const cheekGeo = isModalSubdiv
              ? new THREE.BoxGeometry(cheekThickX, cheekH, cheekDepth, 2, 4, 6)
              : new THREE.BoxGeometry(cheekThickX, cheekH, cheekDepth);
            for (const sx of [-1, 1]) {
              const cheek = new THREE.Mesh(cheekGeo, backplateMat);
              cheek.position.set(
                sx * (plateW * 0.5 - cheekThickX * 0.5),
                ySign * (topFlangeY0 - cheekH * 0.5),
                cheekRearZ + cheekDepth * 0.5
              );
              backplateGroup.add(cheek);
            }
          };

          let frontYTop = backplateTopMinY;
          let frontYBot = backplateBotMinY;

          if (xRailMountTool === "top" || xRailMountTool === "dual_top_bottom") {
            addHorizFlange(1);
            frontYTop = Math.max(frontYTop, topFlangeY0 + plateThick);
          }
          if (xRailMountTool === "bottom" || xRailMountTool === "dual_top_bottom") {
            addHorizFlange(-1);
            frontYBot = Math.min(frontYBot, -(topFlangeY0 + plateThick));
          }

          const frontDropH = frontYTop - frontYBot;
          const frontDropPlate = new THREE.Mesh(
            isModalSubdiv
              ? new THREE.BoxGeometry(plateW, frontDropH, plateThick, 8, 10, 2)
              : new THREE.BoxGeometry(plateW, frontDropH, plateThick),
            backplateMat
          );
          frontDropPlate.position.set(0, 0.5 * (frontYTop + frontYBot), zPlateCenter);
          tagInspectable(frontDropPlate, mountStrutInspectInfo);
          backplateGroup.add(frontDropPlate);

          addBeltClampTabs(zPlateCenter - 0.004, 0);
        }
      }

      thGroup.position.copy(thAnchor);
      group.add(thGroup);
      if (railShoeGroup.children.length > 0) {
        railShoeGroup.position.copy(thAnchor);
        group.add(railShoeGroup);
      }

      // Compute Stage 1 FEA rotation quaternion (rest -> deformed th -> cog vector)
      const stage1FeaQuat = new THREE.Quaternion();
      if (viewMode === "fea_stress" && basePos[thNodeId] && basePos[cogNodeId]) {
        const cv0 = new THREE.Vector3().subVectors(basePos[cogNodeId], basePos[thNodeId]);
        const cv1 = new THREE.Vector3().subVectors(cogAnchor, thAnchor);
        if (cv0.lengthSq() > 1e-6 && cv1.lengthSq() > 1e-6) {
          stage1FeaQuat.setFromUnitVectors(cv0.normalize(), cv1.normalize());
          const qAng = 2 * Math.acos(Math.min(1, Math.abs(stage1FeaQuat.w)));
          if (qAng > 0.34) {
            stage1FeaQuat.slerp(new THREE.Quaternion(), 1 - 0.34 / qAng);
          }
        }
      }

      backplateGroup.position.copy(thAnchor);
      if (viewMode === "fea_stress") {
        backplateGroup.quaternion.identity().slerp(stage1FeaQuat, 0.4);
      }
      group.add(backplateGroup);

      // =====================================================================
      // STAGE 2: Realistic Direct-Drive Extruder + Open-Cage Shroud + Dual Cooling Ducts + Hotend Clamp
      // Local origin (0, 0, 0) = bottom center of Hotend Mount Collar (directly above hotend!).
      // Center front below y = 0 is 100% OPEN so the ENTIRE hotend is clearly visible!
      // =====================================================================
      const cogGroup = new THREE.Group();
      const isDirectDrive = sim.spec.toolhead.extruder.type !== "bowden";
      const isCpapToolhead =
        (sim.spec.toolhead.preset_id || "").toLowerCase().includes("cpap") ||
        (sim.spec.toolhead.hotend.part_cooling_cfm ?? 0) >= 16;

      // 2A. Hotend Mount Collar / Clamping Block (y in [0, +clampH], colored by hotend_mount_material)
      const clampCollar = new THREE.Mesh(
        new THREE.CylinderGeometry(0.148, 0.148, clampH, 22, isModalSubdiv ? 4 : 1),
        clampMat
      );
      clampCollar.position.set(0, clampH * 0.5, 0);
      tagInspectable(clampCollar, nozStemInspectInfo);
      cogGroup.add(clampCollar);

      // Rear structural saddle bridging Hotend Mount Collar flush to X-carriage backplate (ZERO gap!)
      if (!isDeltaArch) {
        const bridgeDepth = Math.max(Math.abs(rearBridgeZ), 0.08);
        const clampWeb = new THREE.Mesh(
          isModalSubdiv
            ? new THREE.BoxGeometry(0.32, clampH, bridgeDepth, 4, 3, 8)
            : new THREE.BoxGeometry(0.32, clampH, bridgeDepth),
          clampMat
        );
        clampWeb.position.set(0, clampH * 0.5, rearBridgeZ * 0.5);
        tagInspectable(clampWeb, nozStemInspectInfo);
        cogGroup.add(clampWeb);
      }

      // Twin front clamping M3 bolts on Hotend Mount Collar
      for (const sx of [-1, 1]) {
        const cb = new THREE.Mesh(m3SocketGeo, boltHeadMat);
        cb.rotation.x = Math.PI * 0.5;
        cb.position.set(sx * 0.11, clampH * 0.5, 0.148);
        cogGroup.add(cb);
      }

      // 2B. Upper Direct-Drive Extruder Assembly (or sleek Bowden coupler turret)
      const extBotY = clampH;
      const extWidth = 0.32 + (bodyWallMm - 4.0) * 0.012; // Scales visibly with body_wall_thickness_mm
      if (isDirectDrive) {
        const extH = 0.22;
        const extDepth = isDeltaArch ? 0.22 : Math.max(Math.abs(rearBridgeZ) + 0.12, 0.24);
        const extCenterZ = isDeltaArch ? 0 : rearBridgeZ + extDepth * 0.5;

        // Sculpted main extruder gear housing (bolts flush against backplateFrontZ)
        const extHousing = new THREE.Mesh(
          isModalSubdiv
            ? new THREE.BoxGeometry(extWidth, extH, extDepth, 6, 6, 8)
            : new THREE.BoxGeometry(extWidth, extH, extDepth),
          bodyMat
        );
        extHousing.position.set(0, extBotY + extH * 0.5, extCenterZ);
        tagInspectable(extHousing, cogInspectInfo);
        cogGroup.add(extHousing);

        // Sleek chamfered front face cap + Voron-red/accent trim bar
        const frontFaceTrim = new THREE.Mesh(
          new THREE.BoxGeometry(Math.max(0.24, extWidth - 0.06), 0.032, 0.022),
          accentTrimMat
        );
        frontFaceTrim.position.set(0, extBotY + extH - 0.026, extCenterZ + extDepth * 0.5 + 0.008);
        cogGroup.add(frontFaceTrim);

        // Exposed extruder drive gear hub window on front-right
        const driveGearMat = new THREE.MeshStandardMaterial({
          color: isHeatMode ? bodyColorObj : 0x334155,
          metalness: 0.72,
          roughness: 0.34,
        });
        const driveGear = new THREE.Mesh(
          new THREE.CylinderGeometry(0.068, 0.068, 0.036, 16),
          driveGearMat
        );
        driveGear.rotation.x = Math.PI * 0.5;
        driveGear.position.set(0.065, extBotY + extH * 0.52, extCenterZ + extDepth * 0.5 + 0.004);
        cogGroup.add(driveGear);

        // Compact NEMA14 36 mm Round Pancake Stepper Motor (authentic Orbiter / Sherpa / CW2 silhouette!)
        // Positioned cleanly in front of rearBridgeZ so it NEVER clips the backplate or X-beam!
        const pancakeR = 0.172; // ~34.4-36 mm OD
        const pancakeDepth = 0.135; // 13.5 mm pancake stator
        const pancakeCenterZ = isDeltaArch
          ? -0.02
          : Math.max(rearBridgeZ + pancakeDepth * 0.5 + 0.025, -0.02);
        const pancakePos = new THREE.Vector3(-0.075, extBotY + extH * 0.88, pancakeCenterZ);

        const statorMat = new THREE.MeshStandardMaterial({
          color: isHeatMode ? bodyColorObj : 0x161b22,
          emissive: bodyEmissive,
          emissiveIntensity: bodyEmissiveInt * 0.5,
          metalness: 0.65,
          roughness: 0.36,
        });
        const endbellMat = new THREE.MeshStandardMaterial({
          color: isHeatMode ? bodyColorObj : 0xcbd5e1,
          metalness: 0.88,
          roughness: 0.22,
        });

        const statorCyl = new THREE.Mesh(
          new THREE.CylinderGeometry(pancakeR, pancakeR, pancakeDepth, 24),
          statorMat
        );
        statorCyl.rotation.x = Math.PI * 0.5;
        statorCyl.position.copy(pancakePos);
        tagInspectable(statorCyl, cogInspectInfo);
        cogGroup.add(statorCyl);

        // Brushed CNC silver front & rear NEMA14 endbells
        const endbellGeo = new THREE.CylinderGeometry(pancakeR + 0.004, pancakeR + 0.004, 0.018, 24);
        for (const sz of [-1, 1]) {
          const cap = new THREE.Mesh(endbellGeo, endbellMat);
          cap.rotation.x = Math.PI * 0.5;
          cap.position.set(
            pancakePos.x,
            pancakePos.y,
            pancakePos.z + sz * (pancakeDepth * 0.5 + 0.008)
          );
          cogGroup.add(cap);
        }

        // Center motor boss + 4 perimeter M3 motor bolts on front face
        const motorShaftHub = new THREE.Mesh(
          new THREE.CylinderGeometry(0.036, 0.036, 0.02, 14),
          boltHeadMat
        );
        motorShaftHub.rotation.x = Math.PI * 0.5;
        motorShaftHub.position.set(
          pancakePos.x,
          pancakePos.y,
          pancakePos.z + pancakeDepth * 0.5 + 0.02
        );
        cogGroup.add(motorShaftHub);

        const motorBoltGeo = new THREE.CylinderGeometry(0.013, 0.013, 0.016, 8);
        for (let mi = 0; mi < 4; mi++) {
          const ang = (mi / 4) * Math.PI * 2 + Math.PI * 0.25;
          const mb = new THREE.Mesh(motorBoltGeo, darkScrewMat);
          mb.rotation.x = Math.PI * 0.5;
          mb.position.set(
            pancakePos.x + Math.cos(ang) * 0.135,
            pancakePos.y + Math.sin(ang) * 0.135,
            pancakePos.z + pancakeDepth * 0.5 + 0.02
          );
          cogGroup.add(mb);
        }

        // Stainless knurled filament idler thumbscrew tensioner on +X side
        const thumbPost = new THREE.Mesh(
          new THREE.CylinderGeometry(0.016, 0.016, 0.075, 10),
          boltHeadMat
        );
        thumbPost.rotation.z = Math.PI * 0.5;
        thumbPost.position.set(extWidth * 0.5 + 0.035, extBotY + extH * 0.65, 0.01);
        cogGroup.add(thumbPost);

        const thumbKnob = new THREE.Mesh(
          new THREE.CylinderGeometry(0.036, 0.036, 0.036, 14),
          boltHeadMat
        );
        thumbKnob.rotation.z = Math.PI * 0.5;
        thumbKnob.position.set(extWidth * 0.5 + 0.075, extBotY + extH * 0.65, 0.01);
        cogGroup.add(thumbKnob);

        // Top filament guide chimney + PTFE Bowden collet + short translucent PTFE guide stub
        const chimney = new THREE.Mesh(
          new THREE.CylinderGeometry(0.046, 0.056, 0.055, 14),
          bodyMat
        );
        chimney.position.set(0.02, extBotY + extH + 0.025, 0);
        cogGroup.add(chimney);

        const colletRing = new THREE.Mesh(
          new THREE.CylinderGeometry(0.038, 0.038, 0.022, 12),
          darkScrewMat
        );
        colletRing.position.set(0.02, extBotY + extH + 0.06, 0);
        cogGroup.add(colletRing);

        const ptfeStub = new THREE.Mesh(
          new THREE.CylinderGeometry(0.02, 0.02, 0.085, 10),
          new THREE.MeshStandardMaterial({
            color: 0xe2e8f0,
            roughness: 0.35,
            metalness: 0.05,
            transparent: true,
            opacity: 0.78,
          })
        );
        ptfeStub.position.set(0.02, extBotY + extH + 0.11, 0);
        cogGroup.add(ptfeStub);
      } else {
        // Bowden extruder setup: compact CNC turret + brass pneumatic push-fit coupler + PTFE stub
        const bowdenTurret = new THREE.Mesh(
          new THREE.CylinderGeometry(0.095, 0.115, 0.11, 16),
          bodyMat
        );
        bowdenTurret.position.set(0, extBotY + 0.055, 0);
        tagInspectable(bowdenTurret, cogInspectInfo);
        cogGroup.add(bowdenTurret);

        const brassCoupler = new THREE.Mesh(
          new THREE.CylinderGeometry(0.048, 0.048, 0.045, 6),
          new THREE.MeshStandardMaterial({ color: 0xf59e0b, metalness: 0.88, roughness: 0.2 })
        );
        brassCoupler.position.set(0, extBotY + 0.132, 0);
        cogGroup.add(brassCoupler);

        const ptfeStub = new THREE.Mesh(
          new THREE.CylinderGeometry(0.02, 0.02, 0.12, 10),
          new THREE.MeshStandardMaterial({
            color: 0xe2e8f0,
            roughness: 0.35,
            metalness: 0.05,
            transparent: true,
            opacity: 0.78,
          })
        );
        ptfeStub.position.set(0, extBotY + 0.21, 0);
        cogGroup.add(ptfeStub);
      }

      // 2C. Lower Open-Cage Shroud + Continuous Dual Angled Part-Cooling Fan Ducts
      // Flanking left (-X) and right (+X) sides ONLY — CENTER FRONT IS 100% OPEN so the entire hotend shines!
      const cageDropLen = Math.max(hotendLen - 0.12, 0.2);
      const cageDepth = isDeltaArch ? 0.16 : Math.max(Math.abs(rearBridgeZ) + 0.07, 0.19);
      const cageCenterZ = isDeltaArch ? -0.01 : rearBridgeZ + cageDepth * 0.5;

      // Cheek thickness driven directly by body_wall_thickness_mm (grows outward, never clips hotend!)
      const cheekInnerX = hotendMountStyle === "chube_6bolt_m3_18pcd" ? 0.212 : 0.189;
      const cheekThickX = 0.022 + (bodyWallMm - 2.0) * 0.0055;
      const sideCheekX = cheekInnerX + cheekThickX * 0.5;
      const sideCheekOuterX = cheekInnerX + cheekThickX;

      // Continuous duct geometry anchor Y coordinates (ZERO floating gaps at any hotendLen!)
      const ductElbowY = -(hotendLen - 0.145);
      const ductTipY = -(hotendLen - 0.068);
      const cheekTopY = Math.min(0.035, clampH * 0.55);
      const cheekBotY = ductElbowY - 0.012;
      const cheekH = Math.max(cheekTopY - cheekBotY, 0.14);
      const cheekCenterY = 0.5 * (cheekTopY + cheekBotY);

      const sideCheekGeo = isModalSubdiv
        ? new THREE.BoxGeometry(cheekThickX, cheekH, cageDepth * 0.78, 2, 8, 4)
        : new THREE.BoxGeometry(cheekThickX, cheekH, cageDepth * 0.78);
      const sideTrimGeo = isModalSubdiv
        ? new THREE.BoxGeometry(0.014, cheekH * 0.88, 0.024, 1, 8, 1)
        : new THREE.BoxGeometry(0.014, cheekH * 0.88, 0.024);
      const ductElbowGeo = new THREE.CylinderGeometry(0.031, 0.031, 0.105, 12);
      const ductNozzleOutletGeo = new THREE.BoxGeometry(0.036, 0.014, 0.078);
      const downUnit = new THREE.Vector3(0, -1, 0);

      for (const sx of [-1, 1]) {
        // Side structural cage rail extending seamlessly from collar down to duct elbow
        const cheek = new THREE.Mesh(sideCheekGeo, bodyMat);
        cheek.position.set(sx * sideCheekX, cheekCenterY, cageCenterZ - 0.012);
        tagInspectable(cheek, cogInspectInfo);
        cogGroup.add(cheek);

        // Crisp accent edge trim along the front of each cheek
        const trim = new THREE.Mesh(sideTrimGeo, accentTrimMat);
        trim.position.set(
          sx * (sideCheekOuterX + 0.004),
          cheekCenterY,
          cageCenterZ + cageDepth * 0.37
        );
        cogGroup.add(trim);

        // Upper cooling discharge point pStart (bottom of blower fan OR bottom of CPAP manifold tube)
        let pStart: THREE.Vector3;
        if (isCpapToolhead) {
          // Twin high-flow CPAP side manifold tubes anchored at top collar & sweeping down to duct elbow
          const cpapTop = new THREE.Vector3(sx * (sideCheekOuterX + 0.044), cheekTopY, -0.01);
          pStart = new THREE.Vector3(
            sx * (sideCheekOuterX + 0.036),
            Math.max(ductElbowY + 0.05, cheekTopY - cheekH * 0.65),
            0.002
          );
          const cpapDir = pStart.clone().sub(cpapTop);
          const cpapLen = cpapDir.length() + 0.024;
          const cpapTube = new THREE.Mesh(
            new THREE.CylinderGeometry(0.042, 0.034, cpapLen, 14, isModalSubdiv ? 8 : 1),
            bodyMat
          );
          cpapTube.position.copy(cpapTop).add(pStart).multiplyScalar(0.5);
          if (cpapDir.lengthSq() > 1e-6) {
            cpapTube.quaternion.setFromUnitVectors(downUnit, cpapDir.clone().normalize());
          }
          tagInspectable(cpapTube, cogInspectInfo);
          cogGroup.add(cpapTube);
        } else {
          // Compact dual 4010/5015 angled blower fan housings flush on the outer cheek
          const fanH = 0.19;
          const fanCenterX = sideCheekOuterX + 0.044;
          const fanCenterY = -Math.min(0.125, cageDropLen * 0.34);
          const fanBody = new THREE.Mesh(
            new THREE.BoxGeometry(0.072, fanH, 0.175),
            darkScrewMat
          );
          fanBody.position.set(sx * fanCenterX, fanCenterY, -0.006);
          fanBody.rotation.z = -sx * 0.1; // subtle outward top lean, bottom discharge angles inward
          tagInspectable(fanBody, cogInspectInfo);
          cogGroup.add(fanBody);

          // Subtle fan intake ring on outer face of each blower
          const fanIntake = new THREE.Mesh(
            new THREE.CylinderGeometry(0.065, 0.065, 0.014, 16),
            bodyMat
          );
          fanIntake.rotation.z = Math.PI * 0.5 - sx * 0.1;
          fanIntake.position.set(sx * (fanCenterX + 0.038), fanCenterY, -0.006);
          cogGroup.add(fanIntake);

          pStart = new THREE.Vector3(sx * (fanCenterX - 0.012), fanCenterY - fanH * 0.44, -0.002);
        }

        // Continuous angled duct chute bridging fan/CPAP bottom outlet -> duct elbow (ZERO gap!)
        const pElbow = new THREE.Vector3(sx * (cheekInnerX + 0.016), ductElbowY, 0.004);
        const pTip = new THREE.Vector3(sx * (cheekInnerX - 0.038), ductTipY, 0.006);

        const chuteVec = pElbow.clone().sub(pStart);
        const chuteLen = Math.max(chuteVec.length() + 0.026, 0.04);
        const chuteMesh = new THREE.Mesh(
          isModalSubdiv
            ? new THREE.BoxGeometry(0.058, chuteLen, 0.106, 2, 6, 2)
            : new THREE.BoxGeometry(0.058, chuteLen, 0.106),
          bodyMat
        );
        chuteMesh.position.copy(pStart).add(pElbow).multiplyScalar(0.5);
        if (chuteVec.lengthSq() > 1e-6) {
          chuteMesh.quaternion.setFromUnitVectors(downUnit, chuteVec.clone().normalize());
        }
        tagInspectable(chuteMesh, cogInspectInfo);
        cogGroup.add(chuteMesh);

        // Smooth rounded elbow knuckle sealing chute -> inward air nozzle seamlessly
        const elbowMesh = new THREE.Mesh(ductElbowGeo, bodyMat);
        elbowMesh.rotation.x = Math.PI * 0.5;
        elbowMesh.position.copy(pElbow);
        tagInspectable(elbowMesh, cogInspectInfo);
        cogGroup.add(elbowMesh);

        // Angled lower part-cooling air nozzle tip pointing inward toward nozzle tip (100% open front!)
        const tipVec = pTip.clone().sub(pElbow);
        const tipLen = Math.max(tipVec.length() + 0.018, 0.06);
        const airNozzle = new THREE.Mesh(
          isModalSubdiv
            ? new THREE.BoxGeometry(0.05, tipLen, 0.1, 2, 4, 2)
            : new THREE.BoxGeometry(0.05, tipLen, 0.1),
          bodyMat
        );
        airNozzle.position.copy(pElbow).add(pTip).multiplyScalar(0.5);
        if (tipVec.lengthSq() > 1e-6) {
          airNozzle.quaternion.setFromUnitVectors(downUnit, tipVec.clone().normalize());
        }
        tagInspectable(airNozzle, cogInspectInfo);
        cogGroup.add(airNozzle);

        // Crisp dark inner air-outlet port lip at the duct tip
        const outletLip = new THREE.Mesh(ductNozzleOutletGeo, darkScrewMat);
        outletLip.position.copy(pTip);
        outletLip.quaternion.copy(airNozzle.quaternion);
        cogGroup.add(outletLip);
      }

      // Top-rear CPAP hose coupler manifold + Y-bridge connecting both side CPAP tubes cleanly
      if (isCpapToolhead) {
        const cpapBridgeSpan = (sideCheekOuterX + 0.044) * 2;
        const cpapYBridge = new THREE.Mesh(
          isModalSubdiv
            ? new THREE.BoxGeometry(cpapBridgeSpan, 0.055, 0.068, 6, 2, 2)
            : new THREE.BoxGeometry(cpapBridgeSpan, 0.055, 0.068),
          bodyMat
        );
        cpapYBridge.position.set(0, cheekTopY + 0.01, rearBridgeZ * 0.55 - 0.01);
        tagInspectable(cpapYBridge, cogInspectInfo);
        cogGroup.add(cpapYBridge);

        const cpapInlet = new THREE.Mesh(
          new THREE.CylinderGeometry(0.068, 0.074, 0.11, 16),
          darkScrewMat
        );
        cpapInlet.position.set(0.095, extBotY + 0.24, rearBridgeZ + 0.075);
        cpapInlet.rotation.x = -0.22;
        cogGroup.add(cpapInlet);
      }

      cogGroup.position.copy(cogAnchor).add(stage2Offset);
      if (viewMode === "fea_stress") {
        cogGroup.quaternion.copy(stage1FeaQuat);
      }
      group.add(cogGroup);

      // =====================================================================
      // STAGE 3: High-Detail Articulated Hotend Assembly
      // Local origin (0, 0, 0) = EXACT nozzle tip apex; local +Y runs dead-vertical up to y = hotendLen
      // (bottom of Hotend Mount Collar), articulating cleanly with toolhead_cog -> nozzle_tip!
      // =====================================================================
      const nozzleGroup = new THREE.Group();

      const nozzleMat = new THREE.MeshStandardMaterial({
        color: nozzleColorObj,
        emissive: nozzleEmissive,
        emissiveIntensity: nozzleEmissiveInt,
        metalness: isHeatMode ? 0.35 : 0.9,
        roughness: isHeatMode ? 0.32 : 0.18,
      });
      const hotendShellMat = new THREE.MeshStandardMaterial({
        color: hotendShellColorObj,
        emissive: hotendShellEmissive,
        emissiveIntensity: hotendShellEmissiveInt,
        metalness: isHeatMode ? 0.32 : hotendMountStyle === "chube_6bolt_m3_18pcd" ? 0.86 : 0.84,
        roughness: isHeatMode ? 0.36 : hotendMountStyle === "chube_6bolt_m3_18pcd" ? 0.26 : 0.24,
      });
      const heaterBlockMat = new THREE.MeshStandardMaterial({
        color: heaterBlockColorObj,
        emissive: hotendShellEmissive,
        emissiveIntensity: hotendShellEmissiveInt * 0.8,
        metalness: isHeatMode ? 0.32 : 0.88,
        roughness: isHeatMode ? 0.36 : 0.22,
      });

      // Precision Brass / Hardened Steel Nozzle (tip apex at exact local y = 0)
      const nozzleCone = new THREE.Mesh(
        new THREE.ConeGeometry(0.044, 0.092, 16, isModalSubdiv ? 2 : 1),
        nozzleMat
      );
      nozzleCone.rotation.x = Math.PI;
      nozzleCone.position.y = 0.046;
      tagInspectable(nozzleCone, nozzleInspectInfo);
      nozzleGroup.add(nozzleCone);

      const nozzleHex = new THREE.Mesh(new THREE.CylinderGeometry(0.056, 0.056, 0.038, 6), nozzleMat);
      nozzleHex.position.y = 0.111;
      tagInspectable(nozzleHex, nozzleInspectInfo);
      nozzleGroup.add(nozzleHex);

      const nozzleNeck = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 0.025, 12), nozzleMat);
      nozzleNeck.position.y = 0.14;
      nozzleGroup.add(nozzleNeck);

      const topY = hotendLen;
      const presetLow = (hotendPresetId || "").toLowerCase();

      if (hotendMountStyle === "chube_6bolt_m3_18pcd") {
        // Luke's Lab Chube (Compact / Air / Conduction):
        // Unmistakable 25.4 mm OD brushed satin-titanium outer structural tube + 6x M3 Torx flange on 18 mm PCD
        // + exposed gleaming copper melt core tip (+ CNC Al conduction block OR side 2510 fan module!)
        const shellBotY = 0.178;
        const flangeH = 0.036;
        const shellTopY = Math.max(topY - flangeH, shellBotY + 0.14);
        const shellLen = shellTopY - shellBotY;

        // Exposed gleaming copper melt core peeking from bottom of satin-titanium shell
        const copperCore = new THREE.Mesh(
          new THREE.CylinderGeometry(0.094, 0.082, 0.052, 20, isModalSubdiv ? 2 : 1),
          heaterBlockMat
        );
        copperCore.position.y = 0.16;
        tagInspectable(copperCore, nozStemInspectInfo);
        nozzleGroup.add(copperCore);

        // 25.4 mm OD (r = 0.127) satin-titanium cylindrical outer structural tube
        const tiShell = new THREE.Mesh(
          new THREE.CylinderGeometry(0.127, 0.127, shellLen, 28, isModalSubdiv ? 12 : 1),
          hotendShellMat
        );
        tiShell.position.y = 0.5 * (shellBotY + shellTopY);
        tagInspectable(tiShell, nozStemInspectInfo);
        nozzleGroup.add(tiShell);

        // Machined circumferential CNC accent rings on the titanium shell
        const tiRingGeo = new THREE.TorusGeometry(0.128, 0.004, 8, 28);
        for (const frac of [0.16, 0.84]) {
          const ring = new THREE.Mesh(tiRingGeo, darkScrewMat);
          ring.rotation.x = Math.PI * 0.5;
          ring.position.y = shellBotY + shellLen * frac;
          nozzleGroup.add(ring);
        }

        // Top 6-bolt circular mounting flange + 6 visible M3 Torx bolts on 18 mm PCD (r = 0.090)
        const flange = new THREE.Mesh(
          new THREE.CylinderGeometry(0.152, 0.152, flangeH, 24),
          hotendShellMat
        );
        flange.position.y = shellTopY + flangeH * 0.5;
        tagInspectable(flange, nozStemInspectInfo);
        nozzleGroup.add(flange);

        const m3TorxGeo = new THREE.CylinderGeometry(0.017, 0.017, 0.022, 8);
        for (let bi = 0; bi < 6; bi++) {
          const ang = (bi / 6) * Math.PI * 2;
          const bolt = new THREE.Mesh(m3TorxGeo, boltHeadMat);
          bolt.position.set(Math.cos(ang) * 0.09, shellTopY + flangeH + 0.008, Math.sin(ang) * 0.09);
          nozzleGroup.add(bolt);
        }

        if (presetLow.includes("conduction")) {
          // Chube Conduction: iconic bright machined CNC Aluminum conduction heatsink block
          const condH = Math.min(0.145, shellLen * 0.48);
          const condBlock = new THREE.Mesh(
            isModalSubdiv
              ? new THREE.BoxGeometry(0.31, condH, 0.27, 4, 6, 4)
              : new THREE.BoxGeometry(0.31, condH, 0.27),
            new THREE.MeshStandardMaterial({
              color: isHeatMode ? hotendShellColorObj : 0xd8e2ee,
              emissive: hotendShellEmissive,
              emissiveIntensity: hotendShellEmissiveInt * 0.6,
              metalness: isHeatMode ? 0.35 : 0.9,
              roughness: isHeatMode ? 0.36 : 0.18,
            })
          );
          condBlock.position.y = shellTopY - condH * 0.52;
          tagInspectable(condBlock, nozStemInspectInfo);
          nozzleGroup.add(condBlock);
        } else if (presetLow.includes("air")) {
          // Chube Air: compact 2510 axial cooling fan module mounted neatly on the left (-X) side
          // hugging the 25.4 mm titanium shell cleanly inside the open-cage shroud (ZERO clipping!)
          const chubeFan = new THREE.Mesh(
            isModalSubdiv
              ? new THREE.BoxGeometry(0.062, 0.19, 0.19, 2, 6, 2)
              : new THREE.BoxGeometry(0.062, 0.19, 0.19),
            darkScrewMat
          );
          chubeFan.position.set(-0.158, shellTopY - Math.min(0.1, shellLen * 0.45), 0.01);
          nozzleGroup.add(chubeFan);
          const chubeFanHub = new THREE.Mesh(
            new THREE.CylinderGeometry(0.064, 0.064, 0.016, 16),
            boltHeadMat
          );
          chubeFanHub.rotation.z = Math.PI * 0.5;
          chubeFanHub.position.set(-0.192, shellTopY - Math.min(0.1, shellLen * 0.45), 0.01);
          nozzleGroup.add(chubeFanHub);
        }
      } else if (hotendMountStyle === "dragon_4bolt_m25") {
        // Phaetus Dragon HF/UHF, Rapido 2 HF/UHF, Mellow Goliath:
        // Warm copper-plated heater block (+ extended melt core / sock accent for UHF / Goliath)
        // + 4 clearly visible shiny titanium M2.5 standoff posts + finned heatsink!
        const isUhfBlock =
          presetLow.includes("uhf") ||
          presetLow.includes("goliath") ||
          (sim.spec.toolhead.hotend.melt_zone_length_mm ?? 0) >= 24;
        const siliconeSockMat = isUhfBlock
          ? new THREE.MeshStandardMaterial({
              color: isHeatMode ? heaterBlockColorObj : 0x1e293b,
              metalness: 0.08,
              roughness: 0.72,
            })
          : null;
        const blockH = isUhfBlock ? 0.155 : 0.115;
        const blockBotY = 0.148;
        const blockCenterY = blockBotY + blockH * 0.5;
        const blockTopY = blockBotY + blockH;

        if (isUhfBlock && siliconeSockMat && (presetLow.includes("rapido") || presetLow.includes("goliath"))) {
          // Extended cylindrical/rectangular high-flow copper melt zone + silicone sock lower accent
          const copperCore = new THREE.Mesh(
            new THREE.CylinderGeometry(0.096, 0.096, blockH, 20, isModalSubdiv ? 4 : 1),
            heaterBlockMat
          );
          copperCore.position.y = blockCenterY;
          tagInspectable(copperCore, nozStemInspectInfo);
          nozzleGroup.add(copperCore);

          const sockCollar = new THREE.Mesh(
            new THREE.CylinderGeometry(0.104, 0.102, blockH * 0.42, 20, isModalSubdiv ? 2 : 1),
            siliconeSockMat
          );
          sockCollar.position.y = blockBotY + blockH * 0.24;
          nozzleGroup.add(sockCollar);
        } else {
          // Machined rectangular copper heater block + lower silicone boot strip for UHF
          const blockMesh = new THREE.Mesh(
            isModalSubdiv
              ? new THREE.BoxGeometry(0.215, blockH, 0.16, 3, 4, 3)
              : new THREE.BoxGeometry(0.215, blockH, 0.16),
            heaterBlockMat
          );
          blockMesh.position.y = blockCenterY;
          tagInspectable(blockMesh, nozStemInspectInfo);
          nozzleGroup.add(blockMesh);

          if (isUhfBlock && siliconeSockMat) {
            const sockBand = new THREE.Mesh(
              new THREE.BoxGeometry(0.224, blockH * 0.36, 0.168),
              siliconeSockMat
            );
            sockBand.position.y = blockBotY + blockH * 0.2;
            nozzleGroup.add(sockBand);
          }

          // Visible stainless heater cartridge protruding slightly on -X side
          const cartridge = new THREE.Mesh(
            new THREE.CylinderGeometry(0.028, 0.028, 0.24, 12),
            boltHeadMat
          );
          cartridge.rotation.z = Math.PI * 0.5;
          cartridge.position.set(-0.015, blockCenterY - 0.008, -0.025);
          nozzleGroup.add(cartridge);
        }

        const sinkBotY = Math.max(blockTopY + 0.088, topY * 0.58);
        const sinkTopY = Math.max(topY - 0.022, sinkBotY + 0.105);
        const sinkH = sinkTopY - sinkBotY;

        // Central internal bimetallic throat
        const throatLen = Math.max(sinkBotY - blockTopY, 0.04);
        const throat = new THREE.Mesh(
          new THREE.CylinderGeometry(0.018, 0.018, throatLen, 10, isModalSubdiv ? 10 : 1),
          boltHeadMat
        );
        throat.position.y = blockTopY + throatLen * 0.5;
        nozzleGroup.add(throat);

        // 4 clearly visible shiny titanium M2.5 standoff posts bridging heatsink down to copper block!
        const standoffLen = Math.max(sinkBotY - blockTopY + 0.01, 0.055);
        const standoffGeo = new THREE.CylinderGeometry(
          0.015,
          0.015,
          standoffLen,
          10,
          isModalSubdiv ? 12 : 1
        );
        const standoffCorners: Array<[number, number]> = [
          [-0.072, -0.056],
          [0.072, -0.056],
          [-0.072, 0.056],
          [0.072, 0.056],
        ];
        for (const [sx, sz] of standoffCorners) {
          const standoff = new THREE.Mesh(standoffGeo, boltHeadMat);
          standoff.position.set(sx, blockTopY + standoffLen * 0.5 - 0.005, sz);
          tagInspectable(standoff, nozStemInspectInfo);
          nozzleGroup.add(standoff);
        }

        // Finned cylindrical aluminum heatsink (5 crisp radial fins)
        const sinkCore = new THREE.Mesh(
          new THREE.CylinderGeometry(0.064, 0.064, sinkH, 16, isModalSubdiv ? 6 : 1),
          hotendShellMat
        );
        sinkCore.position.y = 0.5 * (sinkBotY + sinkTopY);
        tagInspectable(sinkCore, nozStemInspectInfo);
        nozzleGroup.add(sinkCore);

        const finGeo = new THREE.CylinderGeometry(0.114, 0.114, 0.012, 20);
        const finCount = 5;
        for (let fi = 0; fi < finCount; fi++) {
          const fin = new THREE.Mesh(finGeo, hotendShellMat);
          fin.position.y = sinkBotY + ((fi + 0.5) / finCount) * sinkH;
          nozzleGroup.add(fin);
        }

        // Top 4-bolt mounting collar
        const topCollar = new THREE.Mesh(
          new THREE.CylinderGeometry(0.115, 0.115, 0.022, 18),
          hotendShellMat
        );
        topCollar.position.y = sinkTopY + 0.011;
        nozzleGroup.add(topCollar);
      } else if (hotendMountStyle === "groove_mount") {
        // E3D V6 Groove Mount:
        // Machined rectangular Al heater block (23 x 11.5 x 16 mm) + stainless heater cartridge
        // + VISIBLY SLENDER 2.8 mm OD titanium/stainless heatbreak neck + 6 radial Al cooling fins + groove collar!
        const blockH = 0.115;
        const blockBotY = 0.148;
        const blockCenterY = blockBotY + blockH * 0.5;
        const blockTopY = blockBotY + blockH;

        const blockMesh = new THREE.Mesh(
          isModalSubdiv
            ? new THREE.BoxGeometry(0.23, blockH, 0.16, 3, 4, 3)
            : new THREE.BoxGeometry(0.23, blockH, 0.16),
          heaterBlockMat
        );
        blockMesh.position.y = blockCenterY;
        tagInspectable(blockMesh, nozStemInspectInfo);
        nozzleGroup.add(blockMesh);

        // Protruding stainless heater cartridge cylinder + red silicone lead stubs
        const cartridge = new THREE.Mesh(
          new THREE.CylinderGeometry(0.03, 0.03, 0.255, 12),
          boltHeadMat
        );
        cartridge.rotation.z = Math.PI * 0.5;
        cartridge.position.set(-0.018, blockCenterY - 0.008, -0.025);
        nozzleGroup.add(cartridge);

        const leadStub = new THREE.Mesh(
          new THREE.CylinderGeometry(0.012, 0.012, 0.055, 8),
          new THREE.MeshStandardMaterial({
            color: isHeatMode ? heaterBlockColorObj : 0xdc2626,
            metalness: 0.1,
            roughness: 0.55,
          })
        );
        leadStub.rotation.z = Math.PI * 0.5;
        leadStub.position.set(-0.168, blockCenterY - 0.008, -0.025);
        nozzleGroup.add(leadStub);

        // Visibly slender 2.8 mm OD (r = 0.0145) thin-walled heatbreak neck!
        const neckTopY = Math.max(blockTopY + 0.095, topY * 0.56);
        const neckLen = neckTopY - blockTopY;
        const thinNeck = new THREE.Mesh(
          new THREE.CylinderGeometry(0.0145, 0.0145, neckLen, 12, isModalSubdiv ? 12 : 1),
          boltHeadMat
        );
        thinNeck.position.y = blockTopY + neckLen * 0.5;
        tagInspectable(thinNeck, nozStemInspectInfo);
        nozzleGroup.add(thinNeck);

        // Stack of 6 thin radial 22 mm OD aluminum cooling fins (r = 0.11)
        const sinkTopY = Math.max(topY - 0.052, neckTopY + 0.11);
        const sinkH = sinkTopY - neckTopY;
        const sinkCore = new THREE.Mesh(
          new THREE.CylinderGeometry(0.048, 0.048, sinkH, 16, isModalSubdiv ? 6 : 1),
          hotendShellMat
        );
        sinkCore.position.y = 0.5 * (neckTopY + sinkTopY);
        tagInspectable(sinkCore, nozStemInspectInfo);
        nozzleGroup.add(sinkCore);

        const v6FinGeo = new THREE.CylinderGeometry(0.11, 0.11, 0.01, 20);
        const v6FinCount = 6;
        for (let fi = 0; fi < v6FinCount; fi++) {
          const fin = new THREE.Mesh(v6FinGeo, hotendShellMat);
          fin.position.y = neckTopY + ((fi + 0.5) / v6FinCount) * sinkH;
          nozzleGroup.add(fin);
        }

        // Machined silver V6 groove-mount collar (16 mm OD lower lip -> 12 mm recessed groove -> 16 mm upper lip)
        const grooveNeck = new THREE.Mesh(
          new THREE.CylinderGeometry(0.058, 0.058, 0.032, 16),
          hotendShellMat
        );
        grooveNeck.position.y = sinkTopY + 0.016;
        nozzleGroup.add(grooveNeck);

        const grooveLip = new THREE.Mesh(
          new THREE.CylinderGeometry(0.08, 0.08, 0.022, 16),
          hotendShellMat
        );
        grooveLip.position.y = sinkTopY + 0.041;
        nozzleGroup.add(grooveLip);
      } else {
        // rigid_4bolt_m3 (E3D Revo Voron):
        // Compact ceramic HeaterCore ring + silicone boot accent + slim stainless transition tube
        // + anodized red/dark-silver 6-fin heatsink + square 4-bolt flange with 4x M3 socket cap screws!
        const ceramicRing = new THREE.Mesh(
          new THREE.CylinderGeometry(0.074, 0.074, 0.072, 18, isModalSubdiv ? 4 : 1),
          heaterBlockMat
        );
        ceramicRing.position.y = 0.184;
        tagInspectable(ceramicRing, nozStemInspectInfo);
        nozzleGroup.add(ceramicRing);

        const revoSock = new THREE.Mesh(
          new THREE.CylinderGeometry(0.082, 0.08, 0.036, 18),
          new THREE.MeshStandardMaterial({
            color: isHeatMode ? heaterBlockColorObj : 0xdc2626,
            metalness: 0.1,
            roughness: 0.58,
          })
        );
        revoSock.position.y = 0.172;
        nozzleGroup.add(revoSock);

        const sinkBotY = Math.max(0.285, topY * 0.52);
        const stemLen = Math.max(sinkBotY - 0.22, 0.05);
        const stemTube = new THREE.Mesh(
          new THREE.CylinderGeometry(0.028, 0.028, stemLen, 14, isModalSubdiv ? 10 : 1),
          boltHeadMat
        );
        stemTube.position.y = 0.22 + stemLen * 0.5;
        nozzleGroup.add(stemTube);

        const sinkTopY = Math.max(topY - 0.034, sinkBotY + 0.11);
        const sinkH = sinkTopY - sinkBotY;
        const revoCore = new THREE.Mesh(
          new THREE.CylinderGeometry(0.062, 0.062, sinkH, 16, isModalSubdiv ? 6 : 1),
          hotendShellMat
        );
        revoCore.position.y = 0.5 * (sinkBotY + sinkTopY);
        tagInspectable(revoCore, nozStemInspectInfo);
        nozzleGroup.add(revoCore);

        const revoFinGeo = new THREE.CylinderGeometry(0.108, 0.108, 0.011, 20);
        const revoFinCount = 6;
        for (let fi = 0; fi < revoFinCount; fi++) {
          const fin = new THREE.Mesh(revoFinGeo, hotendShellMat);
          fin.position.y = sinkBotY + ((fi + 0.5) / revoFinCount) * sinkH;
          nozzleGroup.add(fin);
        }

        const topFlange = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.032, 0.24), hotendShellMat);
        topFlange.position.y = sinkTopY + 0.016;
        tagInspectable(topFlange, nozStemInspectInfo);
        nozzleGroup.add(topFlange);

        const boltGeo = new THREE.CylinderGeometry(0.017, 0.017, 0.024, 8);
        const boltCorners: Array<[number, number]> = [
          [-0.085, -0.085],
          [0.085, -0.085],
          [-0.085, 0.085],
          [0.085, 0.085],
        ];
        for (const [bx, bz] of boltCorners) {
          const b = new THREE.Mesh(boltGeo, boltHeadMat);
          b.position.set(bx, sinkTopY + 0.036, bz);
          nozzleGroup.add(b);
        }
      }

      // Align articulated hotendGroup: dead-vertical at rest, articulating cleanly with cog -> nozzle flex!
      const baseHotendRefDir = new THREE.Vector3().subVectors(baseCog, baseNoz);
      if (baseHotendRefDir.lengthSq() > 1e-6) {
        baseHotendRefDir.normalize();
      } else {
        baseHotendRefDir.set(0, 1, 0);
      }
      nozzleGroup.position.copy(nozAnchor).add(stage3Offset);
      if (viewMode === "fea_stress") {
        const curHotendDir = new THREE.Vector3().subVectors(cogAnchor, nozAnchor);
        if (curHotendDir.lengthSq() > 1e-6) {
          nozzleGroup.quaternion.setFromUnitVectors(baseHotendRefDir, curHotendDir.normalize());
          const qAng = 2 * Math.acos(Math.min(1, Math.abs(nozzleGroup.quaternion.w)));
          if (qAng > 0.42) {
            nozzleGroup.quaternion.slerp(new THREE.Quaternion(), 1 - 0.42 / qAng);
          }
        }
        // Pin hotend top face (local 0, hotendLen, 0) flush inside bottom of green collar!
        const topVec = new THREE.Vector3(0, hotendLen, 0).applyQuaternion(nozzleGroup.quaternion);
        nozzleGroup.position.copy(cogGroup.position).sub(topVec);
      }
      group.add(nozzleGroup);

      // =====================================================================
      // CLEAN PUNCH-IN & FEA/MODAL ENGINEERING OVERLAYS
      // Assembly view = PURE clean CAD (zero floating rings or pips!)
      // FEA & Modal Punch-In = subtle translucent cyan rest ghost + tiny 1.5 mm node pips/crosshairs
      // =====================================================================
      if (
        (viewMode === "fea_stress" || viewMode === "modal") &&
        basePos[thNodeId] &&
        basePos[cogNodeId] &&
        basePos[nozNodeId]
      ) {
        const ghostGeo = new THREE.BufferGeometry().setFromPoints([
          baseTh,
          stage2RestPos,
          stage3RestPos,
        ]);
        const ghostMat = new THREE.LineBasicMaterial({
          color: toolheadPunchIn ? 0x38bdf8 : 0x94a3b8,
          transparent: true,
          opacity: toolheadPunchIn ? 0.45 : 0.35,
        });
        group.add(new THREE.Line(ghostGeo, ghostMat));

        if (toolheadPunchIn) {
          // Faint translucent cyan undeformed ghost silhouette of extruder + hotend + nozzle
          const ghostSilMat = new THREE.MeshBasicMaterial({
            color: 0x38bdf8,
            wireframe: true,
            transparent: true,
            opacity: 0.16,
            depthWrite: false,
          });
          const ghostBody = new THREE.Mesh(
            new THREE.BoxGeometry(extWidth, 0.24, 0.24),
            ghostSilMat
          );
          ghostBody.position.set(stage2RestPos.x, stage2RestPos.y + clampH + 0.11, stage2RestPos.z);
          group.add(ghostBody);

          const ghostHotendGrp = new THREE.Group();
          const ghostNoz = new THREE.Mesh(new THREE.ConeGeometry(0.044, 0.092, 10), ghostSilMat);
          ghostNoz.rotation.x = Math.PI;
          ghostNoz.position.y = 0.046;
          ghostHotendGrp.add(ghostNoz);
          const ghostShellR = hotendMountStyle === "chube_6bolt_m3_18pcd" ? 0.127 : 0.096;
          const ghostShellLen = Math.max(topY - 0.14, 0.16);
          const ghostShell = new THREE.Mesh(
            new THREE.CylinderGeometry(ghostShellR, ghostShellR, ghostShellLen, 12),
            ghostSilMat
          );
          ghostShell.position.y = 0.14 + ghostShellLen * 0.5;
          ghostHotendGrp.add(ghostShell);
          ghostHotendGrp.position.copy(stage3RestPos);
          group.add(ghostHotendGrp);

          // Live nozzle-tip displacement vector line (rest tip -> deflected/animated tip)
          const dispLineGeo = new THREE.BufferGeometry().setFromPoints([
            stage3RestPos.clone(),
            nozzleGroup.position.clone(),
          ]);
          const dispLineMat = new THREE.LineBasicMaterial({
            color: 0xf43f5e,
            transparent: true,
            opacity: 0.88,
          });
          const dispLine = new THREE.Line(dispLineGeo, dispLineMat);
          dispLine.frustumCulled = false;
          group.add(dispLine);
          if (viewMode === "modal") {
            stateRef.current.modalNozzleDispLine = dispLine;
          }

          // Crisp, tiny 1.5 mm translucent engineering node pips + delicate crosshairs (NO clunky torus donuts!)
          const pipGeo = new THREE.SphereGeometry(0.015, 10, 10);
          const addNodePip = (pos: THREE.Vector3, colorHex: number) => {
            const pipMat = new THREE.MeshBasicMaterial({
              color: colorHex,
              transparent: true,
              opacity: 0.72,
              depthWrite: false,
            });
            const pip = new THREE.Mesh(pipGeo, pipMat);
            pip.position.copy(pos);
            group.add(pip);

            const tick = 0.038;
            const crossGeo = new THREE.BufferGeometry().setFromPoints([
              new THREE.Vector3(pos.x - tick, pos.y, pos.z),
              new THREE.Vector3(pos.x + tick, pos.y, pos.z),
              new THREE.Vector3(pos.x, pos.y - tick, pos.z),
              new THREE.Vector3(pos.x, pos.y + tick, pos.z),
              new THREE.Vector3(pos.x, pos.y, pos.z - tick),
              new THREE.Vector3(pos.x, pos.y, pos.z + tick),
            ]);
            const crossMat = new THREE.LineBasicMaterial({
              color: colorHex,
              transparent: true,
              opacity: 0.48,
            });
            group.add(new THREE.LineSegments(crossGeo, crossMat));
          };
          addNodePip(baseTh, 0x38bdf8); // Carriage node on X-beam
          addNodePip(stage2RestPos, 0xfbbf24); // Hotend mount / Stage 2 pivot
          addNodePip(stage3RestPos, 0xf43f5e); // Rest nozzle tip
        }
      }

      if (viewMode === "modal") {
        const collarRelToTh = stage2RestPos.clone().sub(baseTh);
        const xClampHalf = boltSpanX * 0.55;
        const wClampHalf = boltSpanW * 0.55;
        const bpPivotLocal = new THREE.Vector3();
        if (isDeltaArch) {
          bpPivotLocal.set(0, 0, 0);
        } else if (xRailMountTool === "front") {
          // Hinge at bottom edge of the 4-screw hard-clamped bolted pad
          bpPivotLocal.set(0, -wClampHalf, beamHalfZ + carTotalH);
        } else if (xRailMountTool === "top") {
          bpPivotLocal.set(0, beamHalfY + carTotalH, wClampHalf);
        } else if (xRailMountTool === "bottom") {
          bpPivotLocal.set(0, -(beamHalfY + carTotalH), wClampHalf);
        } else {
          bpPivotLocal.set(0, 0, wClampHalf);
        }

        const bodyRootLocal = isDeltaArch
          ? new THREE.Vector3(0, -collarYRelToTh, 0)
          : new THREE.Vector3(0, 0, rearBridgeZ);

        const plateInnerZRef = backplateFrontZ - plateThick;
        const topFlangeY0Ref = beamHalfY + carTotalH;
        const frontBackplateBotMinY = Math.min(-beamHalfY - 0.09, collarYRelToTh - 0.015);
        const frontBackplateTopMinY = Math.max(
          beamHalfY + 0.09,
          plateH * 0.5,
          collarYRelToTh + clampH + 0.02
        );

        const computeBackplateS = (x: number, y: number, z: number): number => {
          if (isDeltaArch) {
            const r = Math.hypot(x, z);
            return THREE.MathUtils.clamp((0.3 - r) / (0.3 - 0.145), 0, 1);
          }
          if (xRailMountTool === "front") {
            // Hard-mounted 4-screw clamped zone: strictly zero elastic bending across the bolted pad
            if (Math.abs(x) <= xClampHalf && Math.abs(y) <= wClampHalf) {
              return 0;
            }
            const yBotSpan = Math.max(
              -wClampHalf - Math.min(collarYRelToTh, frontBackplateBotMinY),
              0.08
            );
            const sYBot =
              y < -wClampHalf ? THREE.MathUtils.clamp((-wClampHalf - y) / yBotSpan, 0, 1) : 0;
            const yTopSpan = Math.max(frontBackplateTopMinY - wClampHalf, 0.12);
            const sYTop =
              y > wClampHalf ? THREE.MathUtils.clamp((y - wClampHalf) / yTopSpan, 0, 1) * 0.35 : 0;
            const xWingSpan = Math.max(plateW * 0.5 - xClampHalf, 0.08);
            const sX = THREE.MathUtils.clamp((Math.abs(x) - xClampHalf) / xWingSpan, 0, 1);
            return THREE.MathUtils.clamp(Math.max(sYBot, 0.55 * sX, sYTop), 0, 1);
          }
          // Hard-mounted 4-screw horizontal bolted pad on top/bottom MGN carriage face
          if (Math.abs(x) <= xClampHalf && z <= wClampHalf) {
            return 0;
          }
          const sZ = THREE.MathUtils.clamp(
            (z - wClampHalf) / Math.max(plateInnerZRef - wClampHalf, 0.08),
            0,
            1
          );
          const xWingSpan = Math.max(plateW * 0.5 - xClampHalf, 0.08);
          const sX = THREE.MathUtils.clamp((Math.abs(x) - xClampHalf) / xWingSpan, 0, 1);
          if (xRailMountTool === "top") {
            const sDrop = THREE.MathUtils.clamp(
              (topFlangeY0Ref - y) / Math.max(topFlangeY0Ref - (collarYRelToTh + clampH), 0.16),
              0,
              1
            );
            return THREE.MathUtils.clamp(Math.max(0.65 * sZ + 0.35 * sZ * sDrop, 0.25 * sX), 0, 1);
          }
          if (xRailMountTool === "bottom") {
            const sRise = THREE.MathUtils.clamp(
              (y + topFlangeY0Ref) / Math.max(topFlangeY0Ref * 1.5, 0.18),
              0,
              1
            );
            return THREE.MathUtils.clamp(Math.max(0.75 * sZ + 0.25 * sZ * sRise, 0.25 * sX), 0, 1);
          }
          return THREE.MathUtils.clamp(Math.max(sZ, 0.25 * sX), 0, 1);
        };

        const collarBpS = isDeltaArch
          ? 1.0
          : computeBackplateS(collarRelToTh.x, collarRelToTh.y, backplateFrontZ);
        const collarBpW = 0.5 * collarBpS * collarBpS * (3 - collarBpS);

        const computeBodyS = (_x: number, y: number, z: number): number => {
          if (isDeltaArch) {
            const yMountInCog = Math.max(-collarYRelToTh, 0.12);
            return THREE.MathUtils.clamp((yMountInCog - y) / yMountInCog, 0, 1);
          }
          const bodySpanZ = Math.max(Math.abs(rearBridgeZ) * 0.65, 0.09);
          return THREE.MathUtils.clamp((z - rearBridgeZ) / bodySpanZ, 0, 1);
        };

        const hotendLut = getHotendProfileLUT(hotendMountStyle);
        const elasticMeshes: ElasticDeformMesh[] = [];
        const replacedGeos = new Set<THREE.BufferGeometry>();

        const registerElasticGroup = (
          targetGroup: THREE.Group,
          stage: "backplate" | "body" | "hotend"
        ) => {
          for (const child of targetGroup.children) {
            if (!(child instanceof THREE.Mesh) || !child.geometry) continue;
            child.updateMatrix();
            const origGeo = child.geometry;
            const bakedGeo = origGeo.clone();
            bakedGeo.applyMatrix4(child.matrix);
            child.geometry = bakedGeo;
            child.position.set(0, 0, 0);
            child.rotation.set(0, 0, 0);
            child.scale.set(1, 1, 1);
            child.updateMatrix();
            replacedGeos.add(origGeo);

            const posAttr = bakedGeo.getAttribute("position") as THREE.BufferAttribute | undefined;
            if (!posAttr || posAttr.count === 0) continue;
            const restPos = new Float32Array(posAttr.array as Float32Array);
            const normAttr = bakedGeo.getAttribute("normal") as THREE.BufferAttribute | undefined;
            const restNorm = normAttr ? new Float32Array(normAttr.array as Float32Array) : null;
            const count = posAttr.count;
            const flexS = new Float32Array(count);

            let bpFlexS: Float32Array | undefined;
            let hotendSlope: Float32Array | undefined;
            let hotendDisp: Float32Array | undefined;

            if (stage === "backplate") {
              for (let i = 0; i < count; i++) {
                const o = i * 3;
                flexS[i] = computeBackplateS(restPos[o], restPos[o + 1], restPos[o + 2]);
              }
            } else if (stage === "body") {
              bpFlexS = new Float32Array(count);
              for (let i = 0; i < count; i++) {
                const o = i * 3;
                const qx = restPos[o];
                const qy = restPos[o + 1];
                const qz = restPos[o + 2];
                flexS[i] = computeBodyS(qx, qy, qz);
                if (isDeltaArch) {
                  bpFlexS[i] = 1.0;
                } else {
                  // Evaluate Stage-1 backplate flex at the projected rear mounting plane (bit-for-bit seam lock!)
                  bpFlexS[i] = computeBackplateS(
                    qx + collarRelToTh.x,
                    qy + collarRelToTh.y,
                    backplateFrontZ
                  );
                }
              }
            } else {
              hotendSlope = new Float32Array(count);
              hotendDisp = new Float32Array(count);
              const invLen = 1 / Math.max(hotendLen, 0.1);
              for (let i = 0; i < count; i++) {
                const y0 = restPos[i * 3 + 1];
                const s = THREE.MathUtils.clamp((hotendLen - y0) * invLen, 0, 1);
                flexS[i] = s;
                hotendSlope[i] = sampleProfileLUT(hotendLut.slope, s);
                hotendDisp[i] = sampleProfileLUT(hotendLut.disp, s);
              }
            }

            bakedGeo.computeBoundingBox();
            bakedGeo.computeBoundingSphere();
            if (bakedGeo.boundingSphere) {
              bakedGeo.boundingSphere.radius += 0.45;
            }

            elasticMeshes.push({
              mesh: child,
              geometry: bakedGeo,
              restPos,
              restNorm,
              flexS,
              bpFlexS,
              hotendSlope,
              hotendDisp,
              stage,
            });
          }
        };

        registerElasticGroup(backplateGroup, "backplate");
        registerElasticGroup(cogGroup, "body");
        registerElasticGroup(nozzleGroup, "hotend");

        // Dispose any original shared/unshared geometries replaced during elastic baking (zero WebGL leaks)
        const liveThGeos = new Set<THREE.BufferGeometry>();
        thGroup.traverse((obj) => {
          if (obj instanceof THREE.Mesh && obj.geometry) {
            liveThGeos.add(obj.geometry);
          }
        });
        replacedGeos.forEach((g) => {
          if (!liveThGeos.has(g)) g.dispose();
        });

        stateRef.current.modalToolhead = thGroup;
        stateRef.current.modalRailShoe =
          railShoeGroup.children.length > 0 ? railShoeGroup : undefined;
        stateRef.current.modalToolheadNodeId = thNodeId;
        stateRef.current.modalBackplateGroup = backplateGroup;
        stateRef.current.modalCogMarker = cogGroup;
        stateRef.current.modalCogNodeId = cogNodeId;
        stateRef.current.modalCogParentNodeId = thNodeId;
        stateRef.current.modalCogOffset.copy(stage2Offset);
        stateRef.current.modalNozzleMarker = nozzleGroup;
        stateRef.current.modalNozzleNodeId = nozNodeId;
        stateRef.current.modalNozzleRefNodeId = cogNodeId;
        stateRef.current.modalNozzleOffset.copy(stage3Offset);
        stateRef.current.modalNozzleBaseDir.copy(baseHotendRefDir);
        stateRef.current.modalHotendLen = hotendLen;
        stateRef.current.modalElasticMeshes = elasticMeshes;
        stateRef.current.modalRailFlexFrac = stageRailFrac;
        stateRef.current.modalPlateFlexFrac = stagePlateFrac;
        stateRef.current.modalBodyFlexFrac = stageBodyFrac;
        stateRef.current.modalCollarBpW = collarBpW;
        stateRef.current.modalHotendStyle = hotendMountStyle;
        stateRef.current.modalBackplatePivotLocal.copy(bpPivotLocal);
        stateRef.current.modalBodyRootLocal.copy(bodyRootLocal);
        stateRef.current.modalCollarRelToTh.copy(collarRelToTh);
        stateRef.current.modalBodyDepthRef = isDeltaArch ? 0.22 : Math.max(Math.abs(rearBridgeZ) + 0.12, 0.24);
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
        applyInspectHighlight(matched, "pin", 0.65);
        setPinnedPart(matched.userData.inspectInfo as InspectedPartInfo);
      } else {
        stateRef.current.pinnedMesh = null;
        stateRef.current.pinnedKey = null;
        setPinnedPart(null);
      }
    }
    } finally {
      requestRenderRef.current?.();
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
    toolheadPunchIn,
  ]);

  // Smooth camera transition into / out of Toolhead Punch-In macro view
  useEffect(() => {
    const cam = stateRef.current.camera;
    const ctrl = stateRef.current.controls;
    if (!cam || !ctrl) return;

    const wasPunchIn = prevPunchInRef.current;
    prevPunchInRef.current = toolheadPunchIn;

    if (toolheadPunchIn && sim) {
      const geom = sim.geometry;
      const thId = geom.toolhead_node_id || "toolhead";
      const cogId = geom.toolhead_cog_node_id || "toolhead_cog";
      const nozId = geom.nozzle_node_id || "nozzle_tip";
      const bp = stateRef.current.basePositions;
      const pTh = bp[thId];
      const pCog = bp[cogId];
      const pNoz = bp[nozId];

      let mid: THREE.Vector3 | null = null;
      const isDeltaCam = geom.architecture === "delta";
      if (pTh && pNoz) {
        // Frame vertically from X-beam/extruder down to nozzle tip, shifted slightly forward into the toolhead center
        mid = new THREE.Vector3(
          pTh.x,
          pTh.y * 0.44 + pNoz.y * 0.56,
          pTh.z + (isDeltaCam ? 0 : 0.24)
        );
      } else if (pCog) {
        mid = pCog.clone();
      } else if (pTh) {
        mid = pTh.clone();
      }

      if (mid) {
        if (!wasPunchIn) {
          // Save current full-printer camera pose to restore smoothly on exit
          savedCameraRef.current = {
            pos: cam.position.clone(),
            target: ctrl.target.clone(),
          };
          // Front-right 3/4 slightly-below-gantry macro view (~220 mm distance) so the X-beam, linear rail,
          // MGN carriage, backplate, extruder, and full exposed hotend down to nozzle tip are framed front-and-center!
          const macroOffset = new THREE.Vector3(1.26, -0.22, 1.78);
          cameraAnimRef.current.targetLook.copy(mid);
          cameraAnimRef.current.targetPos.copy(mid).add(macroOffset);
          cameraAnimRef.current.active = true;
          lastPunchTargetRef.current = mid.clone();
          requestRenderRef.current?.();
        } else if (
          !lastPunchTargetRef.current ||
          lastPunchTargetRef.current.distanceTo(mid) > 0.08
        ) {
          // Toolhead position moved (e.g., user adjusted X/Y/Z test position) — smoothly shift camera by delta
          const delta = lastPunchTargetRef.current
            ? mid.clone().sub(lastPunchTargetRef.current)
            : new THREE.Vector3();
          cameraAnimRef.current.targetLook.copy(mid);
          cameraAnimRef.current.targetPos.copy(cam.position).add(delta);
          cameraAnimRef.current.active = true;
          lastPunchTargetRef.current = mid.clone();
          requestRenderRef.current?.();
        }
      }
    } else if (wasPunchIn && !toolheadPunchIn) {
      // Restore previous camera position & full-printer target smoothly
      const saved = savedCameraRef.current;
      if (saved) {
        cameraAnimRef.current.targetPos.copy(saved.pos);
        cameraAnimRef.current.targetLook.copy(saved.target);
      } else {
        cameraAnimRef.current.targetPos.set(7.5, 6.2, 8.2);
        cameraAnimRef.current.targetLook.set(0, 2.3, 0);
      }
      cameraAnimRef.current.active = true;
      lastPunchTargetRef.current = null;
      requestRenderRef.current?.();
    }
  }, [toolheadPunchIn, sim]);

  const maxLayers = Math.max(0, (sim?.virtual_print.layer_count ?? 1) - 1);
  const clampedModeIdx =
    sim && sim.modal_fea.modes.length > 0
      ? Math.min(selectedModeIdx, sim.modal_fea.modes.length - 1)
      : 0;
  const activeMode = sim?.modal_fea.modes[clampedModeIdx];
  const limitingSummary = sim?.modal_fea.limiting_modal_summary;
  const primaryLimitingModeIdx =
    sim && sim.modal_fea.modes.length > 0
      ? (() => {
          if (limitingSummary?.primary_structural_mode_number != null) {
            const byNum = sim.modal_fea.modes.findIndex(
              (m) => m.mode_number === limitingSummary.primary_structural_mode_number
            );
            if (byNum >= 0) return byNum;
          }
          return sim.modal_fea.modes.findIndex((m) => Boolean(m.is_primary_limiting_mode));
        })()
      : -1;
  const primaryLimitingModeNum =
    limitingSummary?.primary_structural_mode_number ??
    (primaryLimitingModeIdx >= 0 ? sim?.modal_fea.modes[primaryLimitingModeIdx]?.mode_number : undefined);

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

          {canPunchIn && (
            <button
              type="button"
              className={`bed-warp-toggle-pill ${toolheadPunchIn ? "active" : ""}`}
              style={
                toolheadPunchIn
                  ? {
                      background:
                        "linear-gradient(135deg, rgba(14, 165, 233, 0.34), rgba(56, 189, 248, 0.22))",
                      borderColor: "rgba(56, 189, 248, 0.85)",
                      color: "#e0f2fe",
                    }
                  : undefined
              }
              onClick={() => setToolheadPunchIn(!toolheadPunchIn)}
              title="Macro punch-in zoom locked onto the X-rail, carriage backplate, toolhead body, and hotend to inspect local flex and whip"
            >
              <svg
                width="13"
                height="13"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.2"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                <circle cx="11" cy="11" r="7" />
                <line x1="21" y1="21" x2="16.65" y2="16.65" />
                {toolheadPunchIn ? (
                  <line x1="8" y1="11" x2="14" y2="11" />
                ) : (
                  <>
                    <line x1="11" y1="8" x2="11" y2="14" />
                    <line x1="8" y1="11" x2="14" y2="11" />
                  </>
                )}
              </svg>
              <span>{toolheadPunchIn ? "Exit Punch-In" : "Toolhead Punch-In"}</span>
            </button>
          )}

          {toolheadPunchIn && viewMode === "modal" && (
            <button
              type="button"
              className={`bed-warp-toggle-pill ${isToolheadIsolated ? "active" : ""}`}
              style={
                isToolheadIsolated
                  ? {
                      background:
                        "linear-gradient(135deg, rgba(16, 185, 129, 0.34), rgba(5, 150, 105, 0.24))",
                      borderColor: "rgba(52, 211, 153, 0.85)",
                      color: "#d1fae5",
                    }
                  : undefined
              }
              onClick={() => setIsolateToolheadModal((v) => !v)}
              title="Freeze frame, gantry, and X-rail carriage stationary so you only see relative flex inside the backplate, toolhead body, clamp, and hotend"
            >
              <svg
                width="13"
                height="13"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.2"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                <circle cx="12" cy="12" r="7" />
                <line x1="12" y1="2" x2="12" y2="5" />
                <line x1="12" y1="19" x2="12" y2="22" />
                <line x1="2" y1="12" x2="5" y2="12" />
                <line x1="19" y1="12" x2="22" y2="12" />
              </svg>
              <span>{isToolheadIsolated ? "Toolhead Isolated" : "Isolate Toolhead"}</span>
            </button>
          )}

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

          <div className="hud-pill-group">
            <button
              type="button"
              className={`hud-pill ${ecoDpr ? "active" : ""}`}
              onClick={() => setEcoDpr((v) => !v)}
              title="Toggle 1× pixel ratio (Eco) vs 1.5× Retina DPR to reduce GPU fill rate"
            >
              Eco 1×
            </button>
            <button
              type="button"
              className={`hud-pill ${showHudDetails ? "active" : ""}`}
              onClick={() => setShowHudDetails((prev) => !prev)}
              title="Toggle detailed HUD overlays over the 3D viewport (all full details are also in the right panel)"
            >
              {showHudDetails ? "HUD: Full" : "HUD: Clean"}
            </button>
          </div>
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

        {showHudDetails && viewMode === "modal" && sim && limitingSummary && (() => {
          const shares = limitingSummary.compliance_share_pct;
          const totalShare = Math.max(
            MODAL_COMPLIANCE_SOURCES.reduce((acc, s) => acc + Math.max(0, shares[s.key] ?? 0), 0),
            1e-6
          );
          const bnKind = limitingSummary.limiting_kind;
          const bnColor =
            bnKind === "belt"
              ? "#f59e0b"
              : bnKind === "motor"
                ? "#a78bfa"
                : bnKind === "bracket"
                  ? "#fb7185"
                  : (shares.frame ?? 0) >= 60
                    ? "#ef4444"
                    : "#38bdf8";
          const bnText =
            bnKind === "structural"
              ? `Structural Mode M${limitingSummary.primary_structural_mode_number} ${Math.round(
                  shares.frame ?? 0
                )}%`
              : bnKind === "belt"
                ? `Belt Stretch ${Math.round(shares.belt ?? 0)}%`
                : bnKind === "motor"
                  ? `Motor Stiffness ${Math.round(shares.motor ?? 0)}%`
                  : `Gantry Brackets ${Math.round(shares.bracket ?? 0)}%`;

          const freqMap: Record<"frame" | "belt" | "motor" | "bracket", number> = {
            frame: limitingSummary.structural_freq_hz,
            belt: limitingSummary.belt_freq_hz,
            motor: limitingSummary.motor_freq_hz,
            bracket: limitingSummary.bracket_freq_hz,
          };

          const showFocusBtn =
            primaryLimitingModeIdx >= 0 && primaryLimitingModeIdx !== clampedModeIdx;

          return (
            <div className="hud-card hud-limiting-banner">
              <div className="hud-limiting-top">
                <div className="hud-limiting-badges">
                  <span className="hud-limiting-main-pill">
                    LIMITING: {limitingSummary.limiting_axis}-Axis ·{" "}
                    {limitingSummary.effective_freq_hz.toFixed(1)} Hz (
                    {limitingSummary.recommended_shaper} · ≤
                    {Math.round(limitingSummary.max_accel_mm_s2).toLocaleString()} mm/s²)
                  </span>
                  <span className="hud-limiting-compare">
                    ({limitingSummary.other_axis}:{" "}
                    {limitingSummary.other_axis_effective_freq_hz.toFixed(1)} Hz)
                  </span>
                  <span
                    className="hud-limiting-bn-pill"
                    style={{
                      background: `${bnColor}22`,
                      borderColor: `${bnColor}66`,
                      color: bnColor,
                    }}
                    title={limitingSummary.headline}
                  >
                    #1 {bnText}
                  </span>
                </div>

                {showFocusBtn && (
                  <button
                    type="button"
                    className="hud-focus-limiting-btn"
                    onClick={() => onSelectModeIdx(primaryLimitingModeIdx)}
                    title={`Snap 3D modal animation to Mode ${limitingSummary.primary_structural_mode_number} (${limitingSummary.primary_structural_mode_label})`}
                  >
                    Focus Limiting Mode M{limitingSummary.primary_structural_mode_number}
                  </button>
                )}
              </div>

              <div className="hud-limiting-mid">
                <div className="hud-limiting-comp-group">
                  <div
                    className="hud-limiting-bar"
                    title={`${limitingSummary.limiting_axis}-Axis compliance breakdown`}
                  >
                    {MODAL_COMPLIANCE_SOURCES.map((src) => {
                      const rawPct = Math.max(0, shares[src.key] ?? 0);
                      const normPct = (rawPct / totalShare) * 100;
                      if (normPct < 0.5) return null;
                      return (
                        <div
                          key={src.key}
                          className="hud-limiting-bar-seg"
                          style={{ width: `${normPct}%`, background: src.color }}
                          title={`${src.short}: ${rawPct.toFixed(0)}% (${freqMap[src.key].toFixed(0)} Hz)`}
                        />
                      );
                    })}
                  </div>
                  <div className="hud-limiting-freq-strip">
                    {MODAL_COMPLIANCE_SOURCES.map((src, i) => {
                      const f = freqMap[src.key];
                      const showFreq = Number.isFinite(f) && f > 0 && f < 9999;
                      return (
                        <React.Fragment key={src.key}>
                          {i > 0 && <span className="hud-limiting-dot">·</span>}
                          <span className="hud-limiting-freq-item">
                            <span className="swatch-dot" style={{ background: src.color }} />
                            {src.short}
                            {showFreq ? <strong> {Math.round(f)}Hz</strong> : null}
                          </span>
                        </React.Fragment>
                      );
                    })}
                  </div>
                </div>

                {limitingSummary.top_overall_causes &&
                  limitingSummary.top_overall_causes.length > 0 && (
                    <div className="hud-limiting-cause-chips">
                      {limitingSummary.top_overall_causes.slice(0, 3).map((cause, idx) => {
                        const cColor = CAUSE_CATEGORY_COLORS[cause.category] ?? "#38bdf8";
                        return (
                          <span
                            key={cause.id}
                            className="hud-limiting-cause-chip"
                            style={{ borderLeftColor: cColor }}
                            title={cause.detail}
                          >
                            <span className="hud-limiting-rank" style={{ color: cColor }}>
                              #{idx + 1}
                            </span>
                            <span className="hud-limiting-cause-lbl">{cause.label}</span>
                            <strong>{Math.round(cause.share_pct)}%</strong>
                            {cause.est_hz_gain > 0 && (
                              <span className="hud-limiting-cause-gain">
                                (+{formatHzGain(cause.est_hz_gain)} Hz)
                              </span>
                            )}
                          </span>
                        );
                      })}
                    </div>
                  )}
              </div>

              {limitingSummary.belt_included_note && (
                <div
                  className="hud-limiting-note"
                  title={limitingSummary.belt_included_note}
                >
                  {limitingSummary.belt_included_note}
                </div>
              )}
            </div>
          );
        })()}
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
        {toolheadPunchIn && sim && (() => {
          const bd = sim.static_fea.toolhead_flex_breakdown;
          const thStruct = sim.geometry.toolhead_structure;
          const fallbackMat =
            sim.geometry.mount_material ?? sim.spec.toolhead.mount_material ?? "abs_printed";
          const bpMat = getToolheadMaterialAppearance(
            thStruct?.carriage_mount_material ??
              sim.spec.toolhead.carriage_mount_material ??
              fallbackMat
          );
          const bodyMatInfo = getToolheadMaterialAppearance(
            thStruct?.body_material ?? sim.spec.toolhead.body_material ?? fallbackMat
          );
          const clampMatInfo = getToolheadMaterialAppearance(
            thStruct?.hotend_mount_material ??
              sim.spec.toolhead.hotend_mount_material ??
              sim.spec.toolhead.body_material ??
              fallbackMat
          );
          const bpThickMm =
            thStruct?.carriage_plate_thickness_mm ??
            sim.spec.toolhead.carriage_plate_thickness_mm ??
            5.0;
          const bodyThickMm =
            thStruct?.body_wall_thickness_mm ?? sim.spec.toolhead.body_wall_thickness_mm ?? 4.0;
          const clampThickMm =
            thStruct?.hotend_mount_thickness_mm ??
            sim.spec.toolhead.hotend_mount_thickness_mm ??
            8.0;
          const heStiff =
            bd?.hotend_stiffness_n_per_mm ??
            thStruct?.hotend_stiffness_n_per_mm ??
            sim.spec.toolhead.hotend.structural_stiffness_n_per_mm ??
            450;

          const railPct = Math.max(0, Math.min(100, bd?.rail_and_xbeam_pct ?? 0));
          const bodyPct = Math.max(0, Math.min(100, bd?.carriage_and_body_pct ?? 0));
          const hotendPct = Math.max(0, Math.min(100, bd?.hotend_and_nozzle_pct ?? 0));
          const domSource = bd?.dominant_flex_source;
          const domPct =
            domSource === "hotend_nozzle"
              ? hotendPct
              : domSource === "carriage_body"
                ? bodyPct
                : railPct;

          return (
            <div
              className="hud-card"
              style={{
                width: 318,
                flexDirection: "column",
                alignItems: "stretch",
                gap: 6,
                padding: "9px 11px",
                background: "rgba(13, 19, 29, 0.92)",
                borderColor: "rgba(56, 189, 248, 0.55)",
                boxShadow: "0 10px 28px rgba(0, 0, 0, 0.45)",
              }}
            >
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  gap: 8,
                }}
              >
                <span
                  style={{
                    fontSize: 10.5,
                    fontWeight: 800,
                    textTransform: "uppercase",
                    letterSpacing: "0.05em",
                    color: "var(--accent)",
                  }}
                >
                  Toolhead Flex Breakdown
                </span>
                <span
                  style={{
                    fontSize: 10.5,
                    fontWeight: 700,
                    color: "var(--text)",
                    fontVariantNumeric: "tabular-nums",
                  }}
                >
                  {bd
                    ? `Tip ${Math.round(bd.effective_tip_stiffness_n_per_mm)} N/mm`
                    : `${Math.round(heStiff)} N/mm hotend`}
                </span>
              </div>

              {bd ? (
                <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
                  {(
                    [
                      ["X-Beam & Rail Flex", railPct, "#38bdf8", domSource === "rail_xbeam"],
                      [
                        "Carriage Backplate & Body",
                        bodyPct,
                        "#f59e0b",
                        domSource === "carriage_body",
                      ],
                      [
                        "Hotend & Nozzle Stem",
                        hotendPct,
                        "#f43f5e",
                        domSource === "hotend_nozzle",
                      ],
                    ] as Array<[string, number, string, boolean]>
                  ).map(([label, pct, color, isDom]) => (
                    <div key={label} style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                      <div
                        style={{
                          display: "flex",
                          justifyContent: "space-between",
                          alignItems: "center",
                          fontSize: 10.5,
                        }}
                      >
                        <span
                          style={{
                            color: isDom ? "#f8fafc" : "#cbd5e1",
                            fontWeight: isDom ? 700 : 500,
                          }}
                        >
                          {label}
                        </span>
                        <strong style={{ color, fontVariantNumeric: "tabular-nums" }}>
                          {pct.toFixed(0)}%
                        </strong>
                      </div>
                      <div
                        style={{
                          height: 5,
                          borderRadius: 3,
                          background: "rgba(15, 23, 42, 0.95)",
                          border: "1px solid rgba(39, 50, 71, 0.85)",
                          overflow: "hidden",
                        }}
                      >
                        <div
                          style={{
                            width: `${pct}%`,
                            height: "100%",
                            background: color,
                            borderRadius: 3,
                            transition: "width 0.22s ease",
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div style={{ fontSize: 10.5, color: "var(--muted)" }}>
                  Stage breakdown populates from static FEA toolhead compliance model.
                </div>
              )}

              {showHudDetails && (
                <div
                  style={{
                    display: "flex",
                    flexWrap: "wrap",
                    gap: "3px 8px",
                    fontSize: 10,
                    color: "var(--muted)",
                    paddingTop: 4,
                    borderTop: "1px dashed rgba(255, 255, 255, 0.08)",
                    fontVariantNumeric: "tabular-nums",
                  }}
                >
                  <span>
                    Backplate:{" "}
                    <strong style={{ color: "var(--text)" }}>
                      {bpMat.shortLabel} ({bpThickMm.toFixed(1)}mm)
                    </strong>
                  </span>
                  <span>
                    Body:{" "}
                    <strong style={{ color: "var(--text)" }}>
                      {bodyMatInfo.shortLabel} ({bodyThickMm.toFixed(1)}mm)
                    </strong>
                  </span>
                  <span>
                    Clamp:{" "}
                    <strong style={{ color: "var(--text)" }}>
                      {clampMatInfo.shortLabel} ({clampThickMm.toFixed(1)}mm)
                    </strong>
                  </span>
                  <span>
                    Hotend: <strong style={{ color: "var(--text)" }}>{Math.round(heStiff)} N/mm</strong>
                  </span>
                </div>
              )}

              {isToolheadIsolated && (
                <div
                  style={{
                    fontSize: 10.5,
                    lineHeight: 1.35,
                    color: "#a7f3d0",
                    background: "rgba(16, 185, 129, 0.12)",
                    border: "1px solid rgba(52, 211, 153, 0.35)",
                    borderRadius: 6,
                    padding: "4px 7px",
                  }}
                >
                  <strong>Carriage-Locked Isolation:</strong> Frame &amp; gantry sway frozen — showing
                  only relative flex from the MGN rail carriage to the nozzle tip
                  {activeMode?.toolhead_flop_pct != null
                    ? ` (${Math.round(activeMode.toolhead_flop_pct)}% of this mode)`
                    : ""}
                  .
                </div>
              )}

              {showHudDetails && bd && domSource && (
                <div style={{ fontSize: 10.5, lineHeight: 1.35, color: "#cbd5e1" }}>
                  <span className="sens-fix-prefix">Main limit:</span>{" "}
                  {PUNCHIN_FLEX_SOURCE_LABELS[domSource]} ({Math.round(domPct)}% of toolhead tip
                  flex).
                </div>
              )}
            </div>
          );
        })()}

        {canShowBedWarpHud && showBedWarp && bedPlateFeaHud && (
          <div className="hud-card bed-warp-hud">
            {showHudDetails && <span className="badge-pill">2D MITC4 Bed Plate</span>}
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
            {showHudDetails && (
              <>
                <span style={{ color: "var(--muted)" }}>|</span>
                <span>
                  Taco: <strong>{bedPlateFeaHud.taco_buckling_um.toFixed(1)} µm</strong>
                </span>
                <span style={{ color: "var(--muted)" }}>|</span>
                <span>
                  Plate 1st Mode: <strong>{bedPlateFeaHud.plate_first_mode_hz.toFixed(0)} Hz</strong>
                </span>
              </>
            )}
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
            {showHudDetails && (
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
            )}
          </>
        )}

        {viewMode === "modal" && sim && (
          <>
            <div className="hud-card">
              <span style={{ color: "var(--muted)", fontWeight: 600 }}>Mode:</span>
              {(() => {
                const visibleModes = sim.modal_fea.modes.slice(0, 6).map((m, idx) => ({ m, idx }));
                if (primaryLimitingModeIdx >= 6 && sim.modal_fea.modes[primaryLimitingModeIdx]) {
                  visibleModes.push({
                    m: sim.modal_fea.modes[primaryLimitingModeIdx],
                    idx: primaryLimitingModeIdx,
                  });
                }
                return visibleModes.map(({ m, idx }) => {
                  const axisTag = m.excited_axis ? `${m.excited_axis}-Accel` : m.dominant_axis;
                  const isWorst = sim.modal_fea.worst_print_mode_number === m.mode_number;
                  const isPrimaryLimiting =
                    Boolean(m.is_primary_limiting_mode) ||
                    (primaryLimitingModeNum != null && m.mode_number === primaryLimitingModeNum);
                  const impMeta = m.print_impact_level
                    ? impactLevelMeta(m.print_impact_level)
                    : null;
                  const limAxis = limitingSummary?.limiting_axis;
                  return (
                    <button
                      key={m.mode_number}
                      type="button"
                      className={`hud-pill modal-mode-pill ${clampedModeIdx === idx ? "active" : ""} ${
                        isPrimaryLimiting ? "limiting-mode" : isWorst ? "worst-mode" : ""
                      }`}
                      style={{ padding: "4px 8px" }}
                      onClick={() => onSelectModeIdx(idx)}
                      title={
                        [
                          `Mode ${m.mode_number} (${m.frequency_hz.toFixed(1)} Hz · ${axisTag})`,
                          m.mode_label ? `— ${m.mode_label}` : "",
                          m.limiting_axis_share_pct != null && limAxis
                            ? `• Drives ${Math.round(
                                m.limiting_axis_share_pct
                              )}% of ${limAxis}-axis structural compliance`
                            : "",
                          m.print_impact_score != null
                            ? `• Print Impact ${Math.round(m.print_impact_score)}/100`
                            : "",
                          isPrimaryLimiting
                            ? `• Primary limiting structural mode${limAxis ? ` on ${limAxis}-axis` : ""}`
                            : isWorst
                              ? "★ Worst print-impact mode"
                              : "",
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
                      {isPrimaryLimiting && <span className="limit-pill-tag">LIMIT</span>}
                      {!isPrimaryLimiting && isWorst && (
                        <span className="worst-star" title="Worst print-impact mode">
                          ★
                        </span>
                      )}
                      M{m.mode_number} · {m.frequency_hz.toFixed(1)} Hz ({axisTag})
                    </button>
                  );
                });
              })()}
              {!showHudDetails &&
                primaryLimitingModeIdx >= 0 &&
                primaryLimitingModeIdx !== clampedModeIdx &&
                primaryLimitingModeNum != null && (
                  <button
                    type="button"
                    className="hud-focus-limiting-btn"
                    style={{ padding: "3px 8px", fontSize: 10.5 }}
                    onClick={() => onSelectModeIdx(primaryLimitingModeIdx)}
                    title={`Snap 3D modal animation to Limiting Mode M${primaryLimitingModeNum}${
                      limitingSummary?.primary_structural_mode_label
                        ? ` (${limitingSummary.primary_structural_mode_label})`
                        : ""
                    }`}
                  >
                    ★ Focus Limiting M{primaryLimitingModeNum}
                  </button>
                )}
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
              <span style={{ color: "var(--muted)" }}>|</span>
              <label
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: 6,
                  fontSize: 10.5,
                  color: "var(--muted)",
                  whiteSpace: "nowrap",
                  fontVariantNumeric: "tabular-nums",
                }}
                title="Dial modal animation visual scale (0.1× subtle to 2.0× exaggerated)"
              >
                <span style={{ fontWeight: 600 }}>
                  Scale{" "}
                  <strong style={{ color: "var(--text)" }}>{modalAmpScale.toFixed(1)}×</strong>
                </span>
                <input
                  type="range"
                  min={0.1}
                  max={2.0}
                  step={0.1}
                  value={modalAmpScale}
                  onChange={(e) => setModalAmpScale(Number(e.target.value))}
                  style={{ width: 72, accentColor: "var(--accent)", cursor: "pointer" }}
                  aria-label="Modal display scale"
                />
              </label>
              <span style={{ color: "var(--muted)" }}>|</span>
              <button
                type="button"
                className={`hud-pill ${modalPaused ? "active" : ""}`}
                style={{ padding: "3px 8px", fontSize: 10.5 }}
                onClick={() => setModalPaused((p) => !p)}
                title={
                  modalPaused
                    ? "Resume 3D modal vibration animation (~30 FPS)"
                    : "Pause 3D modal vibration animation (0 FPS idle GPU)"
                }
              >
                {modalPaused ? "▶ Play" : "⏸ Pause"}
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
              const isPrimaryLimitingActive =
                Boolean(activeMode.is_primary_limiting_mode) ||
                (primaryLimitingModeNum != null &&
                  activeMode.mode_number === primaryLimitingModeNum);
              const limAxis = limitingSummary?.limiting_axis;
              const share = activeMode.subsystem_share_pct;
              const topCauses = activeMode.top_causes?.slice(0, 4) ?? [];
              const firstCause = topCauses[0];

              return (
                <div
                  className={`hud-card hud-guide-bar modal-guide ${
                    isPrimaryLimitingActive ? "is-limiting-active" : ""
                  }`}
                >
                  <div className="hud-guide-row hud-guide-header">
                    <div className="hud-guide-badges">
                      <span className="badge-pill hud-guide-pill">
                        {isWorstActive ? "★ " : ""}Mode {activeMode.mode_number} ·{" "}
                        {activeMode.frequency_hz.toFixed(1)} Hz
                      </span>
                      <span className="badge-pill hud-axis-pill">{axisTag}</span>
                      {showHudDetails && isPrimaryLimitingActive && (
                        <span className="badge-pill hud-limit-mode-badge">
                          LIMITING {limAxis ?? activeMode.excited_axis ?? "XY"}-AXIS MODE
                          {activeMode.limiting_axis_share_pct != null
                            ? ` (${Math.round(activeMode.limiting_axis_share_pct)}% of ${
                                limAxis ?? "axis"
                              } struct flex)`
                            : ""}
                        </span>
                      )}
                      {showHudDetails &&
                        (impMeta ? (
                          <span className={`badge-pill ${impMeta.cls}`}>
                            {impMeta.label}
                            {activeMode.print_impact_score != null
                              ? ` ${Math.round(activeMode.print_impact_score)}/100`
                              : ""}
                          </span>
                        ) : (
                          sevMeta && <span className={`badge-pill ${sevMeta.cls}`}>{sevMeta.label}</span>
                        ))}
                      {showHudDetails && (activeMode.nozzle_whip_ratio ?? 0) >= 1.25 && (
                        <span
                          className={`badge-pill ${
                            (activeMode.nozzle_whip_ratio ?? 0) >= 1.8 ? "bad" : "warn"
                          }`}
                          title="Nozzle tip moves farther than the carriage due to rotational pitch/twist whip"
                        >
                          {activeMode.nozzle_whip_ratio!.toFixed(2)}× whip
                        </span>
                      )}
                      {showHudDetails &&
                        activeMode.toolhead_flop_pct != null &&
                        activeMode.toolhead_flop_pct >= 5 && (
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
                      {!showHudDetails && firstCause && (
                        <span
                          className={`hud-mode-cause-chip ${
                            firstCause.kind === "mass" ? "mass" : "flex"
                          }`}
                          title={firstCause.detail}
                        >
                          <span style={{ color: "var(--accent)", fontWeight: 800 }}>#1</span>
                          <span className="hud-cause-name">{firstCause.label}</span>
                          <strong className="hud-cause-pct">
                            {Math.round(firstCause.share_pct)}%
                          </strong>
                        </span>
                      )}
                    </div>
                    <div className="hud-guide-meta">
                      {showHudDetails && (
                        <>
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
                          {sim.geometry.extrusion_filling &&
                            sim.geometry.extrusion_filling.id !== "none" && (
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
                          <span>·</span>
                        </>
                      )}
                      <button
                        type="button"
                        className="hud-inline-more-btn"
                        onClick={() => setShowHudDetails((prev) => !prev)}
                        title={
                          showHudDetails
                            ? "Collapse detailed HUD overlays"
                            : "Expand full mode causes, subsystem bars, and explanations"
                        }
                      >
                        {showHudDetails ? "Less ▴" : "More ▾"}
                      </button>
                    </div>
                  </div>

                  {showHudDetails && topCauses.length > 0 && (
                    <div className="hud-mode-causes">
                      <span className="hud-mode-causes-label">
                        Top Causes of Mode {activeMode.mode_number}:
                      </span>
                      <div className="hud-mode-causes-chips">
                        {topCauses.map((cause) => {
                          const isMass = cause.kind === "mass";
                          return (
                            <span
                              key={cause.id}
                              className={`hud-mode-cause-chip ${isMass ? "mass" : "flex"}`}
                              title={cause.detail}
                            >
                              <span className={`cause-kind-tag ${isMass ? "mass" : "flex"}`}>
                                {isMass ? "MASS" : "FLEX"}
                              </span>
                              <span className="hud-cause-name">{cause.label}</span>
                              <strong className="hud-cause-pct">{Math.round(cause.share_pct)}%</strong>
                              {cause.est_hz_gain > 0 && (
                                <span className="hud-cause-hz">
                                  +{formatHzGain(cause.est_hz_gain)} Hz
                                </span>
                              )}
                            </span>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  {showHudDetails && share && (
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

                  {showHudDetails && activeMode.what_is_shaking && (
                    <div className="hud-guide-line hud-guide-compact">
                      <span className="hud-guide-key">What&apos;s shaking:</span>{" "}
                      <span className="hud-guide-val">{activeMode.what_is_shaking}</span>
                    </div>
                  )}
                  {showHudDetails && activeMode.how_to_fix && (
                    <div className="hud-guide-line hud-fix-line hud-guide-compact">
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

              {showHudDetails && breakdown && (
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

              {showHudDetails && (topStiff || topLight) && (
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
              <strong style={{ color: "#f59e0b" }}>●</strong> {showHudDetails ? "Toolhead CoG" : "CoG"}
            </span>
            <span>·</span>
            <span>
              <strong style={{ color: "#fbbf24" }}>▼</strong> {showHudDetails ? "Nozzle Tip" : "Nozzle"}
            </span>
            {showHudDetails && sim?.geometry.gantry_brackets && (
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
            {showHudDetails &&
              sim?.geometry.extrusion_filling &&
              sim.geometry.extrusion_filling.id !== "none" && (
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
            {showHudDetails && railHudBadgeText && (
              <span className="hud-fill-badge">
                <span className="swatch-dot" style={{ background: "#8e9aa8" }} />
                Rails: <strong>{railHudBadgeText}</strong>
              </span>
            )}
            <span>·</span>
            <span>
              {showHudDetails
                ? "Click part to inspect • Drag to orbit • Scroll to zoom"
                : "Click any part to inspect"}
            </span>
          </div>
        )}
      </div>
    </div>
  );
};
