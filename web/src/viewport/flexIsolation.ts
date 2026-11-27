import * as THREE from "three";
import {
  ModeShapeData,
  SensitivityStiffenTarget,
  SimulationResponse,
  SubsystemKey,
  ViewMode,
} from "../types";
import { InspectedPartInfo } from "./geometryHelpers";
import { classifyPartGroup } from "./partClassifier";

export type FlexIsolationStyle = "xray" | "solo";

export type FlexIsolationTarget =
  | { kind: "none" }
  | { kind: "top_flexers"; minSharePct: number }
  | { kind: "part_group"; partGroupId: string; label: string; subsystem?: SubsystemKey }
  | {
      kind: "part_key";
      partKey: string;
      partGroupId?: string;
      toolheadPartKey?: string;
      label: string;
      subsystem?: SubsystemKey;
    }
  | { kind: "subsystem"; subsystem: SubsystemKey; label: string };

export const NO_FLEX_ISOLATION: FlexIsolationTarget = { kind: "none" };

export interface RankedFlexItem {
  id: string;
  label: string;
  subsystem: SubsystemKey;
  complianceSharePct: number;
  estHzGain?: number;
  hzPerUsd?: number;
  localFlexUm?: number;
  kind: "stiffness" | "mass";
  whyItHurts?: string;
  actionHint?: string;
  target: FlexIsolationTarget;
  /** True when this item is an individual toolhead subpart under toolhead_mount */
  isToolheadSubpart?: boolean;
}

export function isFlexIsolationActive(target: FlexIsolationTarget | null | undefined): boolean {
  return Boolean(target && target.kind !== "none");
}

export function isSameFlexIsolation(
  a: FlexIsolationTarget | null | undefined,
  b: FlexIsolationTarget | null | undefined
): boolean {
  if (!a || !b) return false;
  if (a.kind !== b.kind) return false;
  switch (a.kind) {
    case "none":
      return true;
    case "top_flexers":
      return b.kind === "top_flexers" && a.minSharePct === b.minSharePct;
    case "part_group":
      return b.kind === "part_group" && a.partGroupId === b.partGroupId;
    case "part_key":
      return b.kind === "part_key" && a.partKey === b.partKey;
    case "subsystem":
      return b.kind === "subsystem" && a.subsystem === b.subsystem;
  }
}

export function describeFlexIsolation(target: FlexIsolationTarget): string {
  switch (target.kind) {
    case "none":
      return "";
    case "top_flexers":
      return `Flex ≥ ${target.minSharePct}%`;
    case "part_group":
    case "part_key":
    case "subsystem":
      return target.label;
  }
}

export function inferSubsystemForPartGroup(partGroupId: string, role = ""): SubsystemKey {
  const id = (partGroupId || "").toLowerCase();
  const r = (role || "").toLowerCase();
  if (
    id.startsWith("toolhead") ||
    id.includes("nozzle") ||
    id.includes("hotend") ||
    id.includes("carriage_plate") ||
    r === "toolhead"
  ) {
    return "toolhead";
  }
  if (
    id === "x_gantry_beam" ||
    id === "y_axis_rail_beams" ||
    id === "xy_joints" ||
    id === "gantry_brackets" ||
    id === "y_rail_carriages" ||
    id === "z_gantry_joints" ||
    id === "delta_arms" ||
    id === "y_carriages" ||
    id === "z_carriages" ||
    id === "delta_carriages" ||
    r === "gantry" ||
    r === "rail"
  ) {
    return "gantry";
  }
  if (
    id === "bed_plate" ||
    id === "bed_support" ||
    id === "moving_y_bed" ||
    id.startsWith("bed_") ||
    r === "bed"
  ) {
    return "bed";
  }
  return classifyPartGroup(partGroupId, role).subsystem;
}

const SUBSYSTEM_LABELS: Record<SubsystemKey, string> = {
  toolhead: "Toolhead & Mount",
  gantry: "X/Y Gantry & Rails",
  frame: "Outer Frame & Panels",
  bed: "Bed & Z-Mount",
};

export function subsystemLabel(sub: SubsystemKey): string {
  return SUBSYSTEM_LABELS[sub] ?? sub;
}

/**
 * Build a unified, ranked list of parts/groups causing the most flex in the current view.
 * Combines backend `part_group_sensitivities` + `top_stiffen_targets`, active modal `top_causes`
 * (when in modal mode), and individual `toolhead_parts` when toolhead compliance is significant.
 */
export function buildRankedFlexParts(
  sim: SimulationResponse | null,
  viewMode: ViewMode,
  activeMode?: ModeShapeData
): RankedFlexItem[] {
  if (!sim) return [];

  const items: RankedFlexItem[] = [];
  const seenIds = new Set<string>();

  // In Modal view, prioritize the active mode's structural top_causes first (excluding pure belt/motor drive causes with no 3D mesh)
  if (viewMode === "modal" && activeMode?.top_causes && activeMode.top_causes.length > 0) {
    for (const c of activeMode.top_causes) {
      if (c.subsystem === "drive") continue;
      if (c.share_pct < 1.0) continue;
      const sub: SubsystemKey = c.subsystem;
      seenIds.add(c.id);
      items.push({
        id: c.id,
        label: c.label,
        subsystem: sub,
        complianceSharePct: c.share_pct,
        estHzGain: c.est_hz_gain > 0 ? c.est_hz_gain : undefined,
        kind: c.kind === "mass" ? "mass" : "stiffness",
        whyItHurts: c.detail,
        target: {
          kind: "part_group",
          partGroupId: c.id,
          label: c.label,
          subsystem: sub,
        },
      });
    }
  }

  // Merge all physical part groups from sensitivity.part_group_sensitivities (and top_stiffen_targets fallback)
  const pgMap: Record<string, SensitivityStiffenTarget> = {
    ...(sim.sensitivity?.part_group_sensitivities ?? {}),
  };
  for (const t of sim.sensitivity?.top_stiffen_targets ?? []) {
    if (!pgMap[t.id]) pgMap[t.id] = t;
  }

  const pgEntries = Object.values(pgMap)
    .filter((entry) => (entry.compliance_share_pct ?? 0) >= 0.5)
    .sort((a, b) => (b.compliance_share_pct ?? 0) - (a.compliance_share_pct ?? 0));

  for (const entry of pgEntries) {
    if (seenIds.has(entry.id)) continue;
    seenIds.add(entry.id);
    const sub = inferSubsystemForPartGroup(entry.id, entry.role);
    const label = entry.label ?? classifyPartGroup(entry.id, entry.role).label;
    items.push({
      id: entry.id,
      label,
      subsystem: sub,
      complianceSharePct: entry.compliance_share_pct ?? 0,
      estHzGain: entry.est_hz_gain && entry.est_hz_gain > 0 ? entry.est_hz_gain : undefined,
      hzPerUsd: entry.hz_per_usd > 0 ? entry.hz_per_usd : undefined,
      kind: "stiffness",
      whyItHurts: entry.why_it_hurts,
      actionHint: entry.action_hint,
      target: {
        kind: "part_group",
        partGroupId: entry.id,
        label,
        subsystem: sub,
      },
    });
  }

  // Also append top structural toolhead subparts (e.g. Hotend Clamp, Carriage Backplate, MGN Block)
  // when they carry >= 3% of total nozzle flex so users can jump straight to the exact toolhead stage.
  const thParts = sim.static_fea?.toolhead_parts ?? [];
  for (const tp of thParts) {
    if (!tp.is_structural || tp.compliance_share_pct < 3.0) continue;
    const key = `th_part_${tp.id}`;
    if (seenIds.has(key)) continue;
    seenIds.add(key);
    items.push({
      id: key,
      label: tp.name,
      subsystem: "toolhead",
      complianceSharePct: tp.compliance_share_pct,
      localFlexUm: tp.local_flex_um > 0 ? tp.local_flex_um : undefined,
      kind: "stiffness",
      whyItHurts: tp.why_it_hurts,
      actionHint: tp.action_hint,
      isToolheadSubpart: true,
      target: {
        kind: "part_key",
        partKey: key,
        partGroupId: "toolhead_mount",
        toolheadPartKey: key,
        label: tp.name,
        subsystem: "toolhead",
      },
    });
  }

  // Keep modal causes in their mode-share order first, then global compliance share order
  if (viewMode !== "modal") {
    items.sort((a, b) => {
      // Keep parent part groups ahead of subparts at equal %
      const diff = b.complianceSharePct - a.complianceSharePct;
      if (Math.abs(diff) > 0.25) return diff;
      if (Boolean(a.isToolheadSubpart) !== Boolean(b.isToolheadSubpart)) {
        return a.isToolheadSubpart ? 1 : -1;
      }
      return 0;
    });
  }

  return items;
}

/**
 * Walk up the parent chain to find the nearest `inspectInfo` attached to a mesh or its ancestor group.
 */
function resolveInspectInfo(obj: THREE.Object3D): InspectedPartInfo | undefined {
  let cur: THREE.Object3D | null = obj;
  while (cur) {
    if (cur.userData?.inspectInfo) {
      return cur.userData.inspectInfo as InspectedPartInfo;
    }
    cur = cur.parent;
  }
  return undefined;
}

/**
 * Resolve the effective compliance share (%) for a 3D mesh, falling back to `part_group_sensitivities`.
 */
function resolveMeshCompliancePct(
  info: InspectedPartInfo | undefined,
  sim: SimulationResponse | null
): number {
  if (!info) return 0;
  if (info.complianceSharePct != null && Number.isFinite(info.complianceSharePct)) {
    return info.complianceSharePct;
  }
  const pg = sim?.sensitivity?.part_group_sensitivities?.[info.partGroupId];
  if (pg && Number.isFinite(pg.compliance_share_pct)) {
    return pg.compliance_share_pct;
  }
  return 0;
}

/**
 * Check whether an object (usually a Mesh) belongs to the active `FlexIsolationTarget`.
 */
export function doesObjectMatchFlexIsolation(
  obj: THREE.Object3D,
  isolation: FlexIsolationTarget,
  sim: SimulationResponse | null
): boolean {
  if (isolation.kind === "none") return true;
  const info = resolveInspectInfo(obj);
  const partKey = (obj.userData?.partKey as string | undefined) ?? info?.key;
  const thPartKey = info?.toolheadPartKey;

  if (!info) {
    // Non-inspectable helper geometry (build volume wireframe, arrows, axes, beacons):
    // Keep hotspot beacons visible; ghost generic decorative helpers.
    if (obj.userData?.isHotspotBeacon) return true;
    return false;
  }

  switch (isolation.kind) {
    case "subsystem":
      return info.subsystem === isolation.subsystem;

    case "part_group": {
      const targetId = isolation.partGroupId;
      if (info.partGroupId === targetId) return true;
      // Toolhead assembly & toolhead mount share the same physical toolhead cluster
      if (
        (targetId === "toolhead_mount" || targetId === "toolhead_assembly") &&
        (info.partGroupId === "toolhead_mount" || info.partGroupId === "toolhead_assembly")
      ) {
        return true;
      }
      // Enclosure panels catch-all
      if (
        targetId === "enclosure_panels" &&
        (info.partGroupId === "side_panels" ||
          info.partGroupId === "rear_panel" ||
          info.partGroupId === "top_panel")
      ) {
        return true;
      }
      // Frame columns catch-all
      if (
        targetId === "frame_columns" &&
        (info.partGroupId === "front_frame_columns" || info.partGroupId === "rear_frame_columns")
      ) {
        return true;
      }
      // Y carriages catch-all
      if (
        targetId === "y_carriages" &&
        (info.partGroupId === "y_rail_carriages" || info.partGroupId === "xy_joints")
      ) {
        return true;
      }
      return false;
    }

    case "part_key": {
      if (partKey && partKey === isolation.partKey) return true;
      if (thPartKey && (thPartKey === isolation.partKey || thPartKey === isolation.toolheadPartKey)) {
        return true;
      }
      if (info.key === isolation.partKey) return true;
      return false;
    }

    case "top_flexers": {
      const pct = resolveMeshCompliancePct(info, sim);
      return pct >= isolation.minSharePct;
    }
  }
}

interface SavedMatProps {
  transparent: boolean;
  opacity: number;
  depthWrite: boolean;
}

function restoreMeshMaterial(mesh: THREE.Mesh): void {
  if (mesh.userData._isoOrigMaterial) {
    // Dispose cloned ghost material and restore original shared material
    const currentMat = mesh.material;
    mesh.material = mesh.userData._isoOrigMaterial as THREE.Material | THREE.Material[];
    delete mesh.userData._isoOrigMaterial;
    if (Array.isArray(currentMat)) {
      currentMat.forEach((m) => m.dispose());
    } else if (currentMat) {
      currentMat.dispose();
    }
    return;
  }
  const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
  for (const m of mats) {
    if (!m) continue;
    const saved = m.userData?._isoSavedProps as SavedMatProps | undefined;
    if (saved) {
      m.transparent = saved.transparent;
      m.opacity = saved.opacity;
      m.depthWrite = saved.depthWrite;
      m.needsUpdate = true;
      delete m.userData._isoSavedProps;
    }
  }
}

function applyGhostMaterial(mesh: THREE.Mesh): void {
  // Clone material once so we never ghost a material shared with a kept (isolated) mesh
  if (!mesh.userData._isoOrigMaterial) {
    mesh.userData._isoOrigMaterial = mesh.material;
    if (Array.isArray(mesh.material)) {
      mesh.material = mesh.material.map((m) => m.clone());
    } else if (mesh.material) {
      mesh.material = mesh.material.clone();
    }
  }
  const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
  for (const m of mats) {
    if (!m) continue;
    m.transparent = true;
    m.opacity = 0.085;
    m.depthWrite = false;
    m.needsUpdate = true;
  }
}

/**
 * Apply flex isolation (X-Ray ghosting or Solo hiding) across `dynamicGroup` meshes.
 * Must be called AFTER `applyFeaVisibility` so manual layer/part hides are respected.
 */
export function applyFlexIsolationToGroup(
  group: THREE.Group | undefined,
  isolation: FlexIsolationTarget,
  style: FlexIsolationStyle,
  activeMode: ViewMode,
  sim: SimulationResponse | null
): { matchedCount: number; ghostedCount: number } {
  if (!group) return { matchedCount: 0, ghostedCount: 0 };

  const structuralView =
    activeMode === "fea_stress" || activeMode === "modal" || activeMode === "sensitivity";
  const active = structuralView && isolation.kind !== "none";

  let matchedCount = 0;
  let ghostedCount = 0;

  group.traverse((obj) => {
    if (obj === group) return;
    const isMeshLike =
      obj instanceof THREE.Mesh || obj instanceof THREE.Line || obj instanceof THREE.Points;
    if (!isMeshLike) return;

    if (!active) {
      if (obj.userData.isFlexGhosted) {
        delete obj.userData.isFlexGhosted;
      }
      if (obj.userData.isFlexSoloHidden) {
        delete obj.userData.isFlexSoloHidden;
      }
      if (obj instanceof THREE.Mesh) {
        restoreMeshMaterial(obj);
      }
      return;
    }

    // If already hidden by manual layer/part hide, leave it hidden
    if (!obj.visible && !obj.userData.isFlexSoloHidden) {
      return;
    }

    const matched = doesObjectMatchFlexIsolation(obj, isolation, sim);
    if (matched) {
      matchedCount += 1;
      delete obj.userData.isFlexGhosted;
      delete obj.userData.isFlexSoloHidden;
      if (obj instanceof THREE.Mesh) {
        restoreMeshMaterial(obj);
      }
    } else {
      ghostedCount += 1;
      if (style === "solo") {
        obj.userData.isFlexSoloHidden = true;
        obj.userData.isFlexGhosted = true;
        obj.visible = false;
        if (obj instanceof THREE.Mesh) {
          restoreMeshMaterial(obj);
        }
      } else {
        delete obj.userData.isFlexSoloHidden;
        obj.userData.isFlexGhosted = true;
        if (obj instanceof THREE.Mesh) {
          applyGhostMaterial(obj);
        } else {
          // Dim helper lines/points in X-Ray mode
          obj.visible = false;
          obj.userData.isFlexSoloHidden = true;
        }
      }
    }
  });

  return { matchedCount, ghostedCount };
}

/**
 * Compute the world-space bounding box of all meshes matching a `FlexIsolationTarget`,
 * plus the best representative inspectable mesh to pin in the Part Inspector.
 */
export function findFlexTargetBoundsAndMesh(
  group: THREE.Group | undefined,
  inspectableMeshes: THREE.Mesh[],
  isolation: FlexIsolationTarget,
  sim: SimulationResponse | null
): { box: THREE.Box3; bestMesh: THREE.Mesh | null } {
  const box = new THREE.Box3();
  let bestMesh: THREE.Mesh | null = null;
  let bestShare = -1;

  if (!group || isolation.kind === "none") {
    return { box, bestMesh };
  }

  group.updateMatrixWorld(true);

  for (const mesh of inspectableMeshes) {
    if (!doesObjectMatchFlexIsolation(mesh, isolation, sim)) continue;
    box.expandByObject(mesh);
    const info = mesh.userData?.inspectInfo as InspectedPartInfo | undefined;
    const share = resolveMeshCompliancePct(info, sim);
    if (!bestMesh || share > bestShare) {
      bestMesh = mesh;
      bestShare = share;
    }
  }

  return { box, bestMesh };
}
