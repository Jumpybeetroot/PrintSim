import * as THREE from "three";
import {
  AdjointLens,
  CadGantryStationAsset,
  FeaLoadCase,
  SimulationResponse,
  XRailMountType,
  YRailMountType,
} from "../types";
import {
  CAD_M_TO_THREE,
  SCALE,
  cadGlbBoundsCache,
  cadGlbResolvedCache,
  cloneCachedCadGlb,
} from "./cadGlbCache";
import { sensitivityColorInto, stressColorInto } from "./colormaps";
import { InspectedPartInfo, orientBeamMesh, toThree } from "./geometryHelpers";
import { classifyPartGroup, getLensScore } from "./partClassifier";

export interface ModalCadGantryElasticMesh {
  geometry: THREE.BufferGeometry;
  restPos: Float32Array;
  blendS: Float32Array;
  /** Upper-left 3x3 mapping wrapper-local Three.js displacement -> mesh-local vertex delta */
  wrapperToMesh3: THREE.Matrix3;
}

export interface ModalCadGantryItem {
  wrapper: THREE.Group;
  anchorNodeId: string;
  baseRotY: number;
  port1NodeId?: string;
  /** Rest port0->port1 vector expressed in wrapper-local Three.js space */
  baseSpanWrapper?: THREE.Vector3;
  elasticMeshes?: ModalCadGantryElasticMesh[];
}

export interface BuildCadGantryStationsParams {
  group: THREE.Group;
  sim: SimulationResponse;
  basePos: Record<string, THREE.Vector3>;
  deformedPos?: Record<string, THREE.Vector3>;
  viewMode: "assembly" | "fea_stress" | "modal" | "sensitivity";
  feaLoadCase?: FeaLoadCase;
  toolheadPunchIn: boolean;
  adjointLens: AdjointLens;
  stiffenById: Record<
    string,
    NonNullable<SimulationResponse["sensitivity"]>["top_stiffen_targets"][number]
  >;
  tagInspectable: (mesh: THREE.Mesh, info: InspectedPartInfo) => void;
  getModalNodeScalar?: (nodeId: string) => number;
  globalNodalStress?: Record<string, number>;
  memberSensMap?: Record<
    string,
    NonNullable<SimulationResponse["sensitivity"]>["member_sensitivities"][string]
  >;
}

export interface BuildCadGantryStationsResult {
  replacedMountNodeIds: Set<string>;
  replacedXyNodeIds: Set<string>;
  replacedZDriveIds: Set<string>;
  modalItems: ModalCadGantryItem[];
}

const _tmpV = new THREE.Vector3();
const _tmpInvWrapper = new THREE.Matrix4();
const _tmpToWrapper = new THREE.Matrix4();
const _tmpToMesh = new THREE.Matrix4();

/**
 * Place baked 3D CAD gantry hardware stations (CoreXY corner A/B drives & front idlers,
 * XY-joints, Voron 2.4 quad Z-drives/Z-joints, and Trident Z-mounts/bed arms) in
 * Assembly, FEA Stress, Modal, or Sensitivity modes — with mode-aware heatmap coloring,
 * static FEA deformed placement + 2-port elastic XY-joint blending, and 60 fps modal handles.
 */
export function buildCadGantryStationMeshes({
  group,
  sim,
  basePos,
  deformedPos,
  viewMode,
  feaLoadCase = "xy",
  toolheadPunchIn,
  adjointLens,
  stiffenById,
  tagInspectable,
  getModalNodeScalar,
  globalNodalStress,
  memberSensMap,
}: BuildCadGantryStationsParams): BuildCadGantryStationsResult {
  const replacedMountNodeIds = new Set<string>();
  const replacedXyNodeIds = new Set<string>();
  const replacedZDriveIds = new Set<string>();
  const modalItems: ModalCadGantryItem[] = [];

  const geom = sim.geometry;
  const bracketInfo = geom.gantry_brackets;
  const cadGantryAssets =
    bracketInfo?.hardware_kit !== "procedural" ? (bracketInfo?.cad_assets ?? null) : null;
  if (!cadGantryAssets) {
    return { replacedMountNodeIds, replacedXyNodeIds, replacedZDriveIds, modalItems };
  }

  const isHeatMode = viewMode !== "assembly";
  const bracketColor = new THREE.Color(bracketInfo?.color_hex ?? "#94a3b8");
  const zeroEmissive = new THREE.Color(0x000000);
  const gantryKitLabel = bracketInfo?.hardware_kit_label ?? bracketInfo?.name ?? "Gantry Kit";

  const getCachedGantryStation = (
    stationKey: string
  ): { root: THREE.Group; asset: CadGantryStationAsset } | null => {
    const asset = cadGantryAssets[stationKey];
    if (!asset?.visual_glb_url) return null;
    const root = cadGlbResolvedCache.get(asset.visual_glb_url) ?? null;
    if (!root) return null;
    return { root, asset };
  };

  const resolveStationHeat = (
    stationKey: string,
    anchorNodeId: string,
    port1NodeId?: string,
    sensMemberId?: string
  ): { color: THREE.Color; emissive: THREE.Color; emissiveIntensity: number } => {
    if (!isHeatMode) {
      return { color: bracketColor, emissive: zeroEmissive, emissiveIntensity: 0 };
    }
    const color = new THREE.Color(0x475569);
    const emissive = new THREE.Color(0x000000);
    let emissiveIntensity = 0;

    if (viewMode === "modal" && getModalNodeScalar) {
      const s0 = getModalNodeScalar(anchorNodeId);
      const s1 = port1NodeId ? getModalNodeScalar(port1NodeId) : s0;
      const s = Math.max(s0, s1);
      stressColorInto(s, color);
      if (s > 0.35) {
        emissive.copy(color);
        emissiveIntensity = (s - 0.25) * 0.28;
      }
    } else if (viewMode === "fea_stress") {
      const s0 = globalNodalStress?.[anchorNodeId] ?? 0;
      const s1 = port1NodeId ? (globalNodalStress?.[port1NodeId] ?? s0) : s0;
      const s = Math.max(s0, s1);
      stressColorInto(s, color);
      if (s > 0.55) {
        emissive.copy(color);
        emissiveIntensity = (s - 0.5) * 0.24;
      }
    } else if (viewMode === "sensitivity") {
      const sensEntry = sensMemberId ? memberSensMap?.[sensMemberId] : undefined;
      if (sensEntry) {
        const sc = getLensScore(sensEntry, adjointLens);
        sensitivityColorInto(sc, color);
        const absSc = Math.min(1, Math.abs(sc));
        if (absSc > 0.15) {
          emissive.copy(color);
          emissiveIntensity = absSc * 0.32;
        }
      } else {
        // Subtle neutral slate for stations without a separate FEA superelement entry
        const fallbackGroup = stationKey.startsWith("xy_") ? "xy_joints" : "gantry_brackets";
        const gTarget = stiffenById[fallbackGroup];
        if (gTarget && gTarget.compliance_share_pct > 1.0) {
          const approxSc = Math.min(1, gTarget.compliance_share_pct / 25.0);
          sensitivityColorInto(approxSc, color);
          emissive.copy(color);
          emissiveIntensity = approxSc * 0.22;
        }
      }
    }
    return { color, emissive, emissiveIntensity };
  };

  const placeGantryCadStation = (
    stationKey: string,
    anchorNodeId: string,
    displayName: string,
    inspectKey: string,
    rotY = 0,
    port1NodeId?: string,
    sensMemberId?: string,
    cadOriginNodeId?: string
  ): boolean => {
    const baseAnchor = basePos[anchorNodeId];
    if (!baseAnchor) return false;
    const hit = getCachedGantryStation(stationKey);
    if (!hit) return false;

    const stiffness = hit.asset.stiffness_n_per_mm ?? bracketInfo?.cad_stiffness_n_per_mm ?? null;
    const massText =
      hit.asset.mass_g != null && Number.isFinite(hit.asset.mass_g)
        ? `${Math.round(hit.asset.mass_g)} g`
        : undefined;
    const stiffText =
      stiffness != null && Number.isFinite(stiffness) && stiffness > 0
        ? `${Math.round(stiffness)} N/mm CAD`
        : undefined;
    const profileDesc = [gantryKitLabel, massText, stiffText].filter(Boolean).join(" · ");
    const isXyStation = stationKey === "xy_left" || stationKey === "xy_right";
    const classified = classifyPartGroup(
      sensMemberId || (isXyStation ? "xy_joint" : stationKey),
      "gantry"
    );
    const partGroupId = isXyStation
      ? "xy_joints"
      : sensMemberId
        ? classified.id
        : stationKey.startsWith("trident_bed_arm")
          ? "bed_support"
          : stationKey.startsWith("z_joint")
            ? "z_gantry_joints"
            : "gantry_brackets";
    const subsystem =
      partGroupId === "bed_support" ? "bed" : classified.subsystem;
    const bktStiff = stiffenById[partGroupId];
    const sensEntry = sensMemberId ? memberSensMap?.[sensMemberId] : undefined;
    const seMetricsMap = sim.static_fea.superelement_metrics;
    const seMetric = sensMemberId
      ? (seMetricsMap?.[sensMemberId] ?? seMetricsMap?.[`cad_${sensMemberId}`])
      : undefined;
    const activeCase: FeaLoadCase = feaLoadCase ?? "xy";
    const bracketFlexUm = seMetric
      ? activeCase === "x"
        ? seMetric.flex_x_um
        : activeCase === "y"
          ? seMetric.flex_y_um
          : activeCase === "gravity"
            ? seMetric.flex_gravity_um
            : seMetric.flex_xy_um
      : undefined;
    const bracketTwistMrad = seMetric
      ? activeCase === "x"
        ? seMetric.twist_x_mrad
        : activeCase === "y"
          ? seMetric.twist_y_mrad
          : activeCase === "gravity"
            ? seMetric.twist_gravity_mrad
            : seMetric.twist_xy_mrad
      : undefined;
    const transmittedForceN = seMetric
      ? activeCase === "x"
        ? seMetric.force_x_n
        : activeCase === "y"
          ? seMetric.force_y_n
          : activeCase === "gravity"
            ? seMetric.force_gravity_n
            : seMetric.force_xy_n
      : undefined;
    const inspectInfo: InspectedPartInfo = {
      key: inspectKey,
      name: displayName,
      subsystem,
      partGroupId,
      role: subsystem === "bed" ? "bed" : "gantry",
      material: bracketInfo?.name ?? gantryKitLabel,
      profile: profileDesc || gantryKitLabel,
      complianceSharePct:
        bktStiff?.compliance_share_pct ?? sensEntry?.compliance_share_pct,
      stationComplianceSharePct: sensEntry?.compliance_share_pct,
      bracketFlexUm,
      bracketTwistMrad,
      transmittedForceN,
      heatScore: sensEntry ? getLensScore(sensEntry, adjointLens) : undefined,
      estHzGain: bktStiff?.est_hz_gain,
      hzPer100g: sensEntry?.hz_per_100g,
      whyItHurts: bktStiff?.why_it_hurts,
      actionHint: bktStiff?.action_hint,
    };

    const heat = resolveStationHeat(stationKey, anchorNodeId, port1NodeId, sensMemberId);
    const cadObj = cloneCachedCadGlb(hit.root, {
      isHeatMode,
      heatColor: heat.color,
      heatEmissive: heat.emissive,
      heatEmissiveIntensity: heat.emissiveIntensity,
      inspectInfo,
      tagInspectable,
    });
    cadObj.scale.setScalar(CAD_M_TO_THREE);
    cadObj.rotation.x = -Math.PI * 0.5;
    // Station GLBs are baked about a frame datum (e.g. Y-extrusion axis). When the FEA anchor node
    // sits elsewhere (inner-rail carriage on its raceway), shift the mesh back onto that datum.
    const cadOriginBase = cadOriginNodeId ? basePos[cadOriginNodeId] : undefined;
    if (cadOriginBase) {
      cadObj.position
        .copy(cadOriginBase)
        .sub(baseAnchor)
        .applyAxisAngle(new THREE.Vector3(0, 1, 0), -rotY);
    }

    const wrapper = new THREE.Group();
    const placedPos =
      viewMode === "fea_stress" && deformedPos?.[anchorNodeId]
        ? deformedPos[anchorNodeId]
        : baseAnchor;
    wrapper.position.copy(placedPos);
    if (rotY !== 0) wrapper.rotation.y = rotY;
    wrapper.add(cadObj);

    // 2-port elastic bracket deformation (seam-locks port0/anchor at s=0 and port1 at s=1).
    // Supports general 3D spans (XY-joints, corner motor/idler mounts, Trident bed arms) as well
    // as zero-length FEA node pairs (Voron 2.4 col_g_* -> g_* Z-joints) using CAD port0->port1 axis.
    const hasTwoPortSpan = Boolean(
      port1NodeId &&
        port1NodeId !== anchorNodeId &&
        basePos[port1NodeId] &&
        (viewMode === "fea_stress" || viewMode === "modal")
    );
    let elasticMeshes: ModalCadGantryElasticMesh[] | undefined;
    let baseSpanWrapper: THREE.Vector3 | undefined;

    if (hasTwoPortSpan && port1NodeId) {
      const invBaseRotQ = new THREE.Quaternion().setFromAxisAngle(
        new THREE.Vector3(0, 1, 0),
        -rotY
      );
      baseSpanWrapper = basePos[port1NodeId].clone().sub(baseAnchor).applyQuaternion(invBaseRotQ);

      // Determine wrapper-local blend origin (p0) and axis (p1 - p0) for s in [0, 1]
      const p0Blend = new THREE.Vector3(0, 0, 0);
      const axisBlend = baseSpanWrapper.clone();
      if (axisBlend.lengthSq() < 1e-8 && hit.asset.port0_mm && hit.asset.port1_mm) {
        const p0m = hit.asset.port0_mm;
        const p1m = hit.asset.port1_mm;
        // cadObj.rotation.x = -PI/2 maps CAD (x, y, z) -> Three wrapper (x, z, -y) * SCALE
        p0Blend.set(p0m[0] * SCALE, p0m[2] * SCALE, -p0m[1] * SCALE);
        axisBlend.set(
          (p1m[0] - p0m[0]) * SCALE,
          (p1m[2] - p0m[2]) * SCALE,
          -(p1m[1] - p0m[1]) * SCALE
        );
      }
      const axisLenSq = axisBlend.lengthSq();
      if (axisLenSq > 1e-8) {
        wrapper.updateMatrixWorld(true);
        _tmpInvWrapper.copy(wrapper.matrixWorld).invert();
        elasticMeshes = [];

        cadObj.traverse((obj) => {
          if (!(obj instanceof THREE.Mesh) || !obj.geometry) return;
          const posAttr = obj.geometry.getAttribute("position") as THREE.BufferAttribute | undefined;
          if (!posAttr || posAttr.count === 0) return;

          _tmpToWrapper.multiplyMatrices(_tmpInvWrapper, obj.matrixWorld);
          _tmpToMesh.copy(_tmpToWrapper).invert();
          const wrapperToMesh3 = new THREE.Matrix3().setFromMatrix4(_tmpToMesh);

          const count = posAttr.count;
          const restPos = new Float32Array(posAttr.array as Float32Array);
          const blendS = new Float32Array(count);
          for (let i = 0; i < count; i++) {
            _tmpV.set(restPos[i * 3], restPos[i * 3 + 1], restPos[i * 3 + 2]);
            _tmpV.applyMatrix4(_tmpToWrapper);
            const rawS =
              ((_tmpV.x - p0Blend.x) * axisBlend.x +
                (_tmpV.y - p0Blend.y) * axisBlend.y +
                (_tmpV.z - p0Blend.z) * axisBlend.z) /
              axisLenSq;
            blendS[i] = rawS <= 0 ? 0 : rawS >= 1 ? 1 : rawS;
          }

          if (viewMode === "fea_stress" && deformedPos?.[port1NodeId]) {
            const curSpanWrapper = _tmpV
              .copy(deformedPos[port1NodeId])
              .sub(placedPos)
              .applyQuaternion(invBaseRotQ);
            const deltaWrapper = curSpanWrapper.sub(baseSpanWrapper!);
            const deltaMesh = deltaWrapper.clone().applyMatrix3(wrapperToMesh3);
            const arr = posAttr.array as Float32Array;
            for (let i = 0; i < count; i++) {
              const s = blendS[i];
              if (s === 0) continue;
              const o = i * 3;
              arr[o] = restPos[o] + s * deltaMesh.x;
              arr[o + 1] = restPos[o + 1] + s * deltaMesh.y;
              arr[o + 2] = restPos[o + 2] + s * deltaMesh.z;
            }
            posAttr.needsUpdate = true;
            obj.geometry.computeVertexNormals();
          } else if (viewMode === "modal") {
            obj.frustumCulled = false;
            elasticMeshes!.push({
              geometry: obj.geometry,
              restPos,
              blendS,
              wrapperToMesh3,
            });
          }
        });
      }
    }

    // Subtle transparency on non-XY frame stations in Punch-In mode or skip far outer stations
    if (toolheadPunchIn && !isXyStation && isHeatMode) {
      return false;
    }

    group.add(wrapper);

    if (viewMode === "modal") {
      modalItems.push({
        wrapper,
        anchorNodeId,
        baseRotY: rotY,
        port1NodeId: hasTwoPortSpan ? port1NodeId : undefined,
        baseSpanWrapper,
        elasticMeshes,
      });
    }
    return true;
  };

  // 1. CoreXY CAD gantry corner stations (A/B rear motor mounts & front idlers/AWD motors)
  if (geom.architecture === "corexy") {
    const corexyCornerSpecs: Array<{
      stationKey: string;
      nodeIds: string[];
      port1Id: string;
      name: string;
      sensId: string;
    }> = [
      {
        stationKey: "rear_left",
        nodeIds: ["g_rl", "gantry_rl"],
        port1Id: "pulley_rl",
        name: `${gantryKitLabel} — Rear-Left B-Drive Mount`,
        sensId: "corner_mount_rl",
      },
      {
        stationKey: "rear_right",
        nodeIds: ["g_rr", "gantry_rr"],
        port1Id: "pulley_rr",
        name: `${gantryKitLabel} — Rear-Right A-Drive Mount`,
        sensId: "corner_mount_rr",
      },
      {
        stationKey: "front_left",
        nodeIds: ["g_fl", "gantry_fl"],
        port1Id: "pulley_fl",
        name: `${gantryKitLabel} — Front-Left Corner Mount`,
        sensId: "corner_mount_fl",
      },
      {
        stationKey: "front_right",
        nodeIds: ["g_fr", "gantry_fr"],
        port1Id: "pulley_fr",
        name: `${gantryKitLabel} — Front-Right Corner Mount`,
        sensId: "corner_mount_fr",
      },
    ];
    for (const corner of corexyCornerSpecs) {
      const matchedId = corner.nodeIds.find((nid) => Boolean(basePos[nid]));
      if (!matchedId) continue;
      const placed = placeGantryCadStation(
        corner.stationKey,
        matchedId,
        corner.name,
        `gantry_cad:${corner.stationKey}`,
        0,
        basePos[corner.port1Id] ? corner.port1Id : undefined,
        corner.sensId
      );
      if (placed) {
        replacedMountNodeIds.add(matchedId);
        replacedMountNodeIds.add(corner.port1Id);
      }
    }

    // Fallback: any extra role==="mount" nodes not matching canonical g_* IDs
    // (exclude pulley_* port-1 superelement endpoints, which belong to their parent g_* corner mount)
    const mountNodes = geom.nodes.filter(
      (n) => n.role === "mount" && !n.id.startsWith("pulley_") && basePos[n.id]
    );
    for (const n of mountNodes) {
      if (replacedMountNodeIds.has(n.id)) continue;
      const p = basePos[n.id];
      const stationKey =
        p.x < 0 && p.z < 0
          ? "rear_left"
          : p.x > 0 && p.z < 0
            ? "rear_right"
            : p.x < 0 && p.z > 0
              ? "front_left"
              : p.x > 0 && p.z > 0
                ? "front_right"
                : null;
      if (stationKey) {
        const placed = placeGantryCadStation(
          stationKey,
          n.id,
          `${gantryKitLabel} — ${stationKey.replace("_", " ")}`,
          `gantry_cad:${n.id}`
        );
        if (placed) {
          replacedMountNodeIds.add(n.id);
        }
      }
    }
  }

  // 2. XY-joint carriage brackets (y_car_l -> x_beam_l, y_car_r -> x_beam_r)
  const xyJointNodes = geom.nodes.filter(
    (n) => (n.id === "y_car_l" || n.id === "y_car_r") && basePos[n.id]
  );
  for (const n of xyJointNodes) {
    const isLeft = n.id === "y_car_l";
    const stationKey = isLeft ? "xy_left" : "xy_right";
    const port1Id = isLeft ? "x_beam_l" : "x_beam_r";
    const sensId = isLeft ? "xy_joint_l" : "xy_joint_r";
    const label = isLeft
      ? `${gantryKitLabel} — Left XY-Joint`
      : `${gantryKitLabel} — Right XY-Joint`;
    const railNodeId = isLeft ? "y_rail_car_l" : "y_rail_car_r";
    const placed = placeGantryCadStation(
      stationKey,
      n.id,
      label,
      `gantry_cad:${stationKey}`,
      0,
      basePos[port1Id] ? port1Id : undefined,
      sensId,
      basePos[railNodeId] ? railNodeId : undefined
    );
    if (placed) {
      replacedXyNodeIds.add(n.id);
    }
  }

  // 3. Voron 2.4 Belted Quad-Z CAD hardware (bottom Z-drive units & flying gantry Z-joints)
  const activeBedZDrive = geom.bed_z_drive ?? sim.spec.bed.z_drive;
  if (geom.architecture === "corexy" && activeBedZDrive === "belted_quad_z") {
    const zDriveCorners: Array<{
      ids: string[];
      primaryStation: string;
      fallbackStation: string;
      rotY: number;
      label: string;
      tag: string;
    }> = [
      {
        ids: ["b_fl", "bot_fl"],
        primaryStation: "z_drive_left",
        fallbackStation: "z_drive_right",
        rotY: 0,
        label: `${gantryKitLabel} — Front-Left Z-Drive`,
        tag: "z_drive_fl",
      },
      {
        ids: ["b_fr", "bot_fr"],
        primaryStation: "z_drive_right",
        fallbackStation: "z_drive_left",
        rotY: 0,
        label: `${gantryKitLabel} — Front-Right Z-Drive`,
        tag: "z_drive_fr",
      },
      {
        ids: ["b_rl", "bot_rl"],
        primaryStation: "z_drive_right",
        fallbackStation: "z_drive_left",
        rotY: Math.PI,
        label: `${gantryKitLabel} — Rear-Left Z-Drive`,
        tag: "z_drive_rl",
      },
      {
        ids: ["b_rr", "bot_rr"],
        primaryStation: "z_drive_left",
        fallbackStation: "z_drive_right",
        rotY: Math.PI,
        label: `${gantryKitLabel} — Rear-Right Z-Drive`,
        tag: "z_drive_rr",
      },
    ];
    for (const zc of zDriveCorners) {
      const nid = zc.ids.find((id) => Boolean(basePos[id]));
      if (!nid) continue;
      const placed =
        placeGantryCadStation(zc.primaryStation, nid, zc.label, `gantry_cad:${zc.tag}`, zc.rotY) ||
        placeGantryCadStation(zc.fallbackStation, nid, zc.label, `gantry_cad:${zc.tag}`, zc.rotY);
      if (placed) {
        replacedZDriveIds.add(zc.tag);
      }
    }

    const zJointCorners: Array<{
      ids: string[];
      port1Id: string;
      rotY: number;
      label: string;
      tag: string;
    }> = [
      {
        ids: ["col_g_fl", "g_fl", "gantry_fl"],
        port1Id: "g_fl",
        rotY: 0,
        label: `${gantryKitLabel} — Front-Left Z-Joint`,
        tag: "z_joint_fl",
      },
      {
        ids: ["col_g_fr", "g_fr", "gantry_fr"],
        port1Id: "g_fr",
        rotY: 0,
        label: `${gantryKitLabel} — Front-Right Z-Joint`,
        tag: "z_joint_fr",
      },
      {
        ids: ["col_g_rl", "g_rl", "gantry_rl"],
        port1Id: "g_rl",
        rotY: Math.PI,
        label: `${gantryKitLabel} — Rear-Left Z-Joint`,
        tag: "z_joint_rl",
      },
      {
        ids: ["col_g_rr", "g_rr", "gantry_rr"],
        port1Id: "g_rr",
        rotY: Math.PI,
        label: `${gantryKitLabel} — Rear-Right Z-Joint`,
        tag: "z_joint_rr",
      },
    ];
    for (const zj of zJointCorners) {
      const nid = zj.ids.find((id) => Boolean(basePos[id]));
      if (!nid) continue;
      const p1Id = basePos[zj.port1Id] && zj.port1Id !== nid ? zj.port1Id : undefined;
      placeGantryCadStation(
        "z_joint",
        nid,
        zj.label,
        `gantry_cad:${zj.tag}`,
        zj.rotY,
        p1Id,
        zj.tag
      );
    }
  }

  // 4. Voron Trident Leadscrew Triple-Z CAD hardware (bottom Z motor mounts & bed support arms)
  if (geom.architecture === "corexy" && activeBedZDrive === "leadscrew_triple") {
    const tridentMotorMounts: Array<{
      stationKey: string;
      nodeIds: string[];
      driveId: string;
      fallbackRotY: number;
      label: string;
      tag: string;
    }> = [
      {
        stationKey: "trident_z_mount_left",
        nodeIds: ["b_fl", "bot_fl"],
        driveId: "z_drive_fl",
        fallbackRotY: 0,
        label: `${gantryKitLabel} — Front-Left Trident Z-Motor Mount`,
        tag: "trident_z_mount_left",
      },
      {
        stationKey: "trident_z_mount_right",
        nodeIds: ["b_fr", "bot_fr"],
        driveId: "z_drive_fr",
        fallbackRotY: Math.PI,
        label: `${gantryKitLabel} — Front-Right Trident Z-Motor Mount`,
        tag: "trident_z_mount_right",
      },
      {
        stationKey: "trident_z_mount_rear",
        nodeIds: ["b_rear_mid"],
        driveId: "z_drive_rear",
        fallbackRotY: Math.PI * 0.5,
        label: `${gantryKitLabel} — Rear Trident Z-Motor Mount`,
        tag: "trident_z_mount_rear",
      },
    ];
    for (const tm of tridentMotorMounts) {
      const nid = tm.nodeIds.find((id) => Boolean(basePos[id]));
      if (!nid) continue;
      const hasPerSideStation = Boolean(cadGantryAssets[tm.stationKey]);
      const placed = hasPerSideStation
        ? placeGantryCadStation(tm.stationKey, nid, tm.label, `gantry_cad:${tm.tag}`, 0)
        : placeGantryCadStation(
            "trident_z_mount",
            nid,
            tm.label,
            `gantry_cad:${tm.tag}`,
            tm.fallbackRotY
          );
      if (placed) {
        replacedZDriveIds.add(tm.driveId);
      }
    }

    const tridentBedArms: Array<{
      stationKey: string;
      nodeId: string;
      port1Id: string;
      fallbackRotY: number;
      label: string;
      tag: string;
      sensId: string;
    }> = [
      {
        stationKey: "trident_bed_arm_left",
        nodeId: "z_bed_fl",
        port1Id: "z_arm_tip_fl",
        fallbackRotY: 0,
        label: `${gantryKitLabel} — Front-Left Trident Bed Support Arm`,
        tag: "trident_bed_arm_left",
        sensId: "trident_bed_arm_fl",
      },
      {
        stationKey: "trident_bed_arm_right",
        nodeId: "z_bed_fr",
        port1Id: "z_arm_tip_fr",
        fallbackRotY: Math.PI,
        label: `${gantryKitLabel} — Front-Right Trident Bed Support Arm`,
        tag: "trident_bed_arm_right",
        sensId: "trident_bed_arm_fr",
      },
      {
        stationKey: "trident_bed_arm_rear",
        nodeId: "z_bed_r",
        port1Id: "z_arm_tip_r",
        fallbackRotY: Math.PI * 0.5,
        label: `${gantryKitLabel} — Rear Trident Bed Support Arm`,
        tag: "trident_bed_arm_rear",
        sensId: "trident_bed_arm_r",
      },
    ];
    for (const ba of tridentBedArms) {
      if (!basePos[ba.nodeId]) continue;
      const p1Id = basePos[ba.port1Id] ? ba.port1Id : undefined;
      const hasPerSideStation = Boolean(cadGantryAssets[ba.stationKey]);
      if (hasPerSideStation) {
        placeGantryCadStation(
          ba.stationKey,
          ba.nodeId,
          ba.label,
          `gantry_cad:${ba.tag}`,
          0,
          p1Id,
          ba.sensId
        );
      } else {
        placeGantryCadStation(
          "trident_bed_arm",
          ba.nodeId,
          ba.label,
          `gantry_cad:${ba.tag}`,
          ba.fallbackRotY,
          p1Id,
          ba.sensId
        );
      }
    }
  }

  return { replacedMountNodeIds, replacedXyNodeIds, replacedZDriveIds, modalItems };
}

export interface BuildAssemblyHardwareParams {
  group: THREE.Group;
  sim: SimulationResponse;
  basePos: Record<string, THREE.Vector3>;
  cx: number;
  cy: number;
  toolheadPunchIn: boolean;
  adjointLens: AdjointLens;
  feaLoadCase?: FeaLoadCase;
  stiffenById: Record<
    string,
    NonNullable<SimulationResponse["sensitivity"]>["top_stiffen_targets"][number]
  >;
  memberSensMap?: Record<
    string,
    NonNullable<SimulationResponse["sensitivity"]>["member_sensitivities"][string]
  >;
  tagInspectable: (mesh: THREE.Mesh, info: InspectedPartInfo) => void;
}

/**
 * Assembly-mode procedural hardware: NEMA17 stepper motors, gantry brackets,
 * X/Y/Z linear rails, Z leadscrews/ballscrews/belts/motors, and 3D structural panels
 * (including CoreXY enclosures, open-door CNC portal frame, and header/skirt plates).
 */
export function buildAssemblyHardwareMeshes({
  group,
  sim,
  basePos,
  cx,
  cy,
  toolheadPunchIn,
  adjointLens,
  feaLoadCase = "xy",
  stiffenById,
  memberSensMap,
  tagInspectable,
}: BuildAssemblyHardwareParams): void {
  const geom = sim.geometry;
      const bracketInfo = geom.gantry_brackets;
      const bracketColor = new THREE.Color(bracketInfo?.color_hex ?? "#94a3b8");
      const bracketIsCnc = Boolean(bracketInfo?.cnc_machined);
      const bracketMat = new THREE.MeshStandardMaterial({
        color: bracketColor,
        metalness: bracketIsCnc ? 0.78 : 0.22,
        roughness: bracketIsCnc ? 0.24 : 0.48,
      });
      let bracketMatUsed = false;

      const cadResult = buildCadGantryStationMeshes({
        group,
        sim,
        basePos,
        viewMode: "assembly",
        feaLoadCase,
        toolheadPunchIn,
        adjointLens,
        stiffenById,
        memberSensMap,
        tagInspectable,
      });
      const cadReplacedMountNodeIds = cadResult.replacedMountNodeIds;
      const cadReplacedXyNodeIds = cadResult.replacedXyNodeIds;
      const cadReplacedZDriveIds = cadResult.replacedZDriveIds;

      // NEMA17 stepper motor blocks & tinted motor-mount brackets at every node with role === "mount"
      // (skipping CoreXY corner stations already rendered by real CAD GLB assemblies and pulley_* port-1 nodes)
      const mountNodes = geom.nodes.filter(
        (n) => n.role === "mount" && !n.id.startsWith("pulley_") && basePos[n.id]
      );
      const proceduralMountNodes = mountNodes.filter((n) => !cadReplacedMountNodeIds.has(n.id));

      if (proceduralMountNodes.length > 0) {
        bracketMatUsed = true;
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

        for (const n of proceduralMountNodes) {
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

      // Procedural fallback XY-joint carriage bracket blocks when CAD GLB is not loaded
      const xyJointNodes = geom.nodes.filter(
        (n) => (n.id === "y_car_l" || n.id === "y_car_r") && basePos[n.id] && !cadReplacedXyNodeIds.has(n.id)
      );
      if (xyJointNodes.length > 0) {
        bracketMatUsed = true;
        const xyJointGeo = new THREE.BoxGeometry(0.32, 0.24, 0.34);
        const xyStiff = stiffenById["xy_joints"];
        const seMetricsMap = sim.static_fea.superelement_metrics;
        for (const n of xyJointNodes) {
          const isLeft = n.id === "y_car_l";
          const sensId = isLeft ? "xy_joint_l" : "xy_joint_r";
          const sensEntry = memberSensMap?.[sensId];
          const seMetric = seMetricsMap?.[sensId];
          const jointMesh = new THREE.Mesh(xyJointGeo, bracketMat);
          jointMesh.position.copy(basePos[n.id]);
          tagInspectable(jointMesh, {
            key: `xy_joint_proc:${sensId}`,
            name: isLeft ? "Left XY-Joint Bracket" : "Right XY-Joint Bracket",
            subsystem: "gantry",
            partGroupId: "xy_joints",
            role: "gantry",
            material: bracketInfo?.name ?? "Gantry Bracket",
            profile: "Procedural XY-Joint",
            complianceSharePct:
              xyStiff?.compliance_share_pct ?? sensEntry?.compliance_share_pct,
            stationComplianceSharePct: sensEntry?.compliance_share_pct,
            bracketFlexUm: seMetric
              ? feaLoadCase === "x"
                ? seMetric.flex_x_um
                : feaLoadCase === "y"
                  ? seMetric.flex_y_um
                  : feaLoadCase === "gravity"
                    ? seMetric.flex_gravity_um
                    : seMetric.flex_xy_um
              : undefined,
            bracketTwistMrad: seMetric
              ? feaLoadCase === "x"
                ? seMetric.twist_x_mrad
                : feaLoadCase === "y"
                  ? seMetric.twist_y_mrad
                  : feaLoadCase === "gravity"
                    ? seMetric.twist_gravity_mrad
                    : seMetric.twist_xy_mrad
              : undefined,
            transmittedForceN: seMetric
              ? feaLoadCase === "x"
                ? seMetric.force_x_n
                : feaLoadCase === "y"
                  ? seMetric.force_y_n
                  : feaLoadCase === "gravity"
                    ? seMetric.force_gravity_n
                    : seMetric.force_xy_n
              : undefined,
            heatScore: sensEntry ? getLensScore(sensEntry, adjointLens) : undefined,
            estHzGain: xyStiff?.est_hz_gain,
            hzPer100g: sensEntry?.hz_per_100g,
            whyItHurts: xyStiff?.why_it_hurts,
            actionHint: xyStiff?.action_hint,
          });
          group.add(jointMesh);
        }
      }

      if (!bracketMatUsed) {
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

          const xRailVis = geom.x_rail_visual ?? null;
          const cachedXRailGlb =
            xRailVis?.visual_glb_url
              ? (cadGlbResolvedCache.get(xRailVis.visual_glb_url) ?? null)
              : null;

          const addXRailFace = (
            face: "front" | "top" | "bottom",
            pos: THREE.Vector3
          ) => {
            if (cachedXRailGlb && xRailVis) {
              const nativeLenThree = Math.max(xRailVis.length_mm * SCALE, 1e-4);
              const railClone = cachedXRailGlb.clone(true);
              const scaleX = (railLenX / nativeLenThree) * CAD_M_TO_THREE;
              // Native GLB: X=length, Y=width (raceways ±Y), Z=height (top bolt face +Z, pockets -Z)
              if (face === "top") {
                railClone.rotation.x = -Math.PI * 0.5;
              } else if (face === "bottom") {
                railClone.rotation.x = Math.PI * 0.5;
              }
              railClone.scale.set(scaleX, CAD_M_TO_THREE, CAD_M_TO_THREE);
              railClone.position.copy(pos);
              railClone.name = `cad_x_rail_${xRailVis.asset_id}_${face}`;
              group.add(railClone);
              return;
            }
            const xGeo =
              face === "front"
                ? new THREE.BoxGeometry(railLenX, xRailW, xRailH)
                : new THREE.BoxGeometry(railLenX, xRailH, xRailW);
            const xEdges = new THREE.EdgesGeometry(xGeo);
            addRailBar(xGeo, xEdges, pos);
            addXCounterboresAndGrooves(face, pos);
          };

          if (xRailMount === "front") {
            // CAD -Y face -> Three.js +Z face; wide face in CAD Z (Three.js Y), thickness in CAD -Y (Three.js +Z)
            const pos = new THREE.Vector3(beamCX, beamCY, beamCZ + beamHalfCadY + xRailH * 0.5);
            addXRailFace("front", pos);
          } else if (xRailMount === "top") {
            // CAD +Z face -> Three.js +Y face; wide face in CAD Y (Three.js Z), thickness in CAD +Z (Three.js +Y)
            const pos = new THREE.Vector3(beamCX, beamCY + beamHalfCadZ + xRailH * 0.5, beamCZ);
            addXRailFace("top", pos);
          } else if (xRailMount === "bottom") {
            // CAD -Z face -> Three.js -Y face
            const pos = new THREE.Vector3(beamCX, beamCY - beamHalfCadZ - xRailH * 0.5, beamCZ);
            addXRailFace("bottom", pos);
          } else if (xRailMount === "dual_top_bottom") {
            // Dual rails on both +Z (+Y Three.js) and -Z (-Y Three.js) faces
            const posTop = new THREE.Vector3(beamCX, beamCY + beamHalfCadZ + xRailH * 0.5, beamCZ);
            const posBot = new THREE.Vector3(beamCX, beamCY - beamHalfCadZ - xRailH * 0.5, beamCZ);
            addXRailFace("top", posTop);
            addXRailFace("bottom", posBot);
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

            const yRailVis = geom.y_rail_visual ?? null;
            const cachedYRailGlb =
              yRailVis?.visual_glb_url
                ? (cadGlbResolvedCache.get(yRailVis.visual_glb_url) ?? null)
                : null;
            const addYRailBar = (
              pos: THREE.Vector3,
              normVec: THREE.Vector3,
              fallbackGeo: THREE.BoxGeometry,
              fallbackEdges: THREE.EdgesGeometry,
              tag: string
            ) => {
              if (cachedYRailGlb && yRailVis) {
                const nativeLenThree = Math.max(yRailVis.length_mm * SCALE, 1e-4);
                const railClone = cachedYRailGlb.clone(true);
                const lenVec = new THREE.Vector3(0, 0, 1);
                const wVec = new THREE.Vector3().crossVectors(normVec, lenVec).normalize();
                const q = new THREE.Quaternion().setFromRotationMatrix(
                  new THREE.Matrix4().makeBasis(lenVec, wVec, normVec)
                );
                railClone.quaternion.copy(q);
                railClone.scale.set(
                  (railLenY / nativeLenThree) * CAD_M_TO_THREE,
                  CAD_M_TO_THREE,
                  CAD_M_TO_THREE
                );
                railClone.position.copy(pos);
                railClone.name = `cad_y_rail_${yRailVis.asset_id}_${tag}`;
                group.add(railClone);
                return;
              }
              addRailBar(fallbackGeo, fallbackEdges, pos);
            };

            if (yRailMount === "top") {
              const yGeo = new THREE.BoxGeometry(yRailW, yRailH, railLenY);
              const yEdges = new THREE.EdgesGeometry(yGeo);
              const up = new THREE.Vector3(0, 1, 0);
              addYRailBar(
                leftCenter.clone().add(new THREE.Vector3(0, beamHalfCadZ + yRailH * 0.5, 0)),
                up,
                yGeo,
                yEdges,
                "l_top"
              );
              addYRailBar(
                rightCenter.clone().add(new THREE.Vector3(0, beamHalfCadZ + yRailH * 0.5, 0)),
                up,
                yGeo,
                yEdges,
                "r_top"
              );
            } else if (yRailMount === "bottom") {
              const yGeo = new THREE.BoxGeometry(yRailW, yRailH, railLenY);
              const yEdges = new THREE.EdgesGeometry(yGeo);
              const down = new THREE.Vector3(0, -1, 0);
              addYRailBar(
                leftCenter.clone().add(new THREE.Vector3(0, -(beamHalfCadZ + yRailH * 0.5), 0)),
                down,
                yGeo,
                yEdges,
                "l_bot"
              );
              addYRailBar(
                rightCenter.clone().add(new THREE.Vector3(0, -(beamHalfCadZ + yRailH * 0.5), 0)),
                down,
                yGeo,
                yEdges,
                "r_bot"
              );
            } else if (yRailMount === "inner") {
              // +X (inner) face of left Y extrusion and -X (inner) face of right Y extrusion
              const yGeo = new THREE.BoxGeometry(yRailH, yRailW, railLenY);
              const yEdges = new THREE.EdgesGeometry(yGeo);
              addYRailBar(
                leftCenter.clone().add(new THREE.Vector3(beamHalfCadX + yRailH * 0.5, 0, 0)),
                new THREE.Vector3(1, 0, 0),
                yGeo,
                yEdges,
                "l_inner"
              );
              addYRailBar(
                rightCenter.clone().add(new THREE.Vector3(-(beamHalfCadX + yRailH * 0.5), 0, 0)),
                new THREE.Vector3(-1, 0, 0),
                yGeo,
                yEdges,
                "r_inner"
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

            // Match the vertical column member sharing this Z-rail's (x, z) axis so the rail bolts flush onto its outer face
            const colMember =
              geom.members.find((m) => {
                const ms = basePos[m.start_id];
                const me = basePos[m.end_id];
                if (!ms || !me) return false;
                if (Math.abs(me.y - ms.y) < 0.1) return false;
                const mx = 0.5 * (ms.x + me.x);
                const mz = 0.5 * (ms.z + me.z);
                return Math.abs(mx - zMidX) < 0.025 && Math.abs(mz - zMidZ) < 0.025;
              }) ??
              geom.members.find(
                (m) =>
                  m.id.startsWith("col_fl") ||
                  m.id === "bed_rear_z_bot" ||
                  m.id === "z_col_l_low" ||
                  m.id === "tower_0_low"
              );
            const colHalfX = colMember ? Math.max(colMember.width_mm, 10) * 0.5 * SCALE : 10 * SCALE;
            const colHalfZ = colMember ? Math.max(colMember.height_mm, 10) * 0.5 * SCALE : 10 * SCALE;
            const colHalfW =
              Math.abs(normVec.x) * colHalfX + Math.abs(normVec.z) * colHalfZ > 0
                ? Math.hypot(normVec.x * colHalfX, normVec.z * colHalfZ)
                : colHalfX;

            const zRailPos = p0
              .clone()
              .add(p1)
              .multiplyScalar(0.5)
              .addScaledVector(normVec, colHalfW + rH * 0.5);
            const zRailVis = geom.z_rail_visual ?? null;
            const cachedZRailGlb =
              zRailVis?.visual_glb_url && !toolheadPunchIn
                ? (cadGlbResolvedCache.get(zRailVis.visual_glb_url) ?? null)
                : null;
            // Right-handed basis for native GLB (X=length, Y=width, Z=height/normal)
            const glbWVec = new THREE.Vector3().crossVectors(normVec, axisDir).normalize();
            const glbRailQuat = new THREE.Quaternion().setFromRotationMatrix(
              new THREE.Matrix4().makeBasis(axisDir, glbWVec, normVec)
            );
            if (cachedZRailGlb && zRailVis) {
              const nativeLenThree = Math.max(zRailVis.length_mm * SCALE, 1e-4);
              const zRailClone = cachedZRailGlb.clone(true);
              zRailClone.quaternion.copy(glbRailQuat);
              zRailClone.scale.set(
                (railLen / nativeLenThree) * CAD_M_TO_THREE,
                CAD_M_TO_THREE,
                CAD_M_TO_THREE
              );
              zRailClone.position.copy(zRailPos);
              zRailClone.traverse((o) => {
                if (o instanceof THREE.Mesh) tagInspectable(o, zRailInspect);
              });
              group.add(zRailClone);
            } else {
              const barGeo = new THREE.BoxGeometry(rW, railLen, rH);
              const barMesh = new THREE.Mesh(barGeo, zRailSteelMat);
              barMesh.position.copy(zRailPos);
              barMesh.quaternion.copy(railQuat);
              tagInspectable(barMesh, zRailInspect);
              group.add(barMesh);

              const barOutline = new THREE.LineSegments(new THREE.EdgesGeometry(barGeo), zRailEdgeMat);
              barOutline.position.copy(barMesh.position);
              barOutline.quaternion.copy(railQuat);
              group.add(barOutline);
            }

            // MGN metallic carriage block + wiper end-caps at carriage_z_mm
            const isMgn15 = zr.profile_id.includes("mgn15");
            const isHgh15 = zr.profile_id.includes("hgh15");
            const carH = (isHgh15 ? 28 : isMgn15 ? 16 : isMgn9 ? 10 : 13) * SCALE;
            const zCarVis = geom.carriage_visuals?.z ?? null;
            const cachedZCarGlb =
              zCarVis?.visual_glb_url && !toolheadPunchIn
                ? (cadGlbResolvedCache.get(zCarVis.visual_glb_url) ?? null)
                : null;
            if (cachedZCarGlb && zCarVis) {
              const carClone = cachedZCarGlb.clone(true);
              const originOff = carH - zCarVis.port1_z_mm * SCALE;
              carClone.quaternion.copy(glbRailQuat);
              carClone.scale.setScalar(CAD_M_TO_THREE);
              carClone.position
                .set(zMidX, carY, zMidZ)
                .addScaledVector(normVec, colHalfW + originOff);
              carClone.traverse((o) => {
                if (o instanceof THREE.Mesh) tagInspectable(o, zCarInspect);
              });
              group.add(carClone);
            } else {
              const carW = (isMgn9 ? 20 : 27) * SCALE;
              const carL = (isMgn9 ? 36 : 44) * SCALE;
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
                .addScaledVector(normVec, colHalfW + rH * 0.5 + carH * 0.42);
              carGroup.quaternion.copy(railQuat);
              group.add(carGroup);
            }
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
        const wobbleXCopperMat = new THREE.MeshStandardMaterial({
          color: 0xd97736,
          metalness: 0.88,
          roughness: 0.2,
          transparent: toolheadPunchIn,
          opacity: toolheadPunchIn ? 0.16 : 1.0,
        });
        const gearboxMetalMat = new THREE.MeshStandardMaterial({
          color: 0xb8c4d4,
          metalness: 0.9,
          roughness: 0.2,
          transparent: toolheadPunchIn,
          opacity: toolheadPunchIn ? 0.16 : 1.0,
        });
        const gearboxHousingMat = new THREE.MeshStandardMaterial({
          color: 0x334155,
          metalness: 0.8,
          roughness: 0.3,
          transparent: toolheadPunchIn,
          opacity: toolheadPunchIn ? 0.16 : 1.0,
        });

        let leadUsed = false;
        let ballChromeUsed = false;
        let ballNutUsed = false;
        let brassUsed = false;
        let pomUsed = false;
        let beltUsed = false;
        let couplerUsed = false;
        let zMotorUsed = false;
        let wobbleXUsed = false;
        let gearboxUsed = false;

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

          const mechId = zl.mechanism_id || "";
          const isWobbleX = mechId.includes("wobblex");
          const isPlanetary = mechId.includes("planetary");
          const isWorm = mechId.includes("worm");
          const isHeavyBall = mechId.includes("sfu1605");

          const mechNice = mechId.replace(/_/g, " ").toUpperCase();
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
            profile: mechId,
          };
          const zMotorInspect: InspectedPartInfo = {
            key: `z_motor:${zl.id}`,
            name: isPlanetary
              ? `NEMA17 + 5:1 Planetary Z-Motor (${zl.id})`
              : isWorm
                ? `NEMA17 + 20:1 Worm Z-Motor (${zl.id})`
                : `NEMA17 Z-Axis Motor (${zl.id})`,
            subsystem: "bed",
            partGroupId: "bed_support",
            role: "bed",
            material: "nema17_stepper",
            profile: mechId,
          };

          // Skip procedural NEMA17 Z-stepper motor when a real CAD Z-drive / Z-mount GLB already renders it
          const hasCadZDrive = cadReplacedZDriveIds.has(zl.id);
          if (!hasCadZDrive) {
            zMotorUsed = true;
            const zMotGrp = new THREE.Group();
            const zMotBody = new THREE.Mesh(zMotorBodyGeo, zMotorBodyMat);
            zMotBody.position.y = -0.19;
            tagInspectable(zMotBody, zMotorInspect);
            zMotGrp.add(zMotBody);
            const zMotCap = new THREE.Mesh(zMotorCapGeo, zMotorCapMat);
            zMotCap.position.y = 0.02;
            zMotGrp.add(zMotCap);

            // Metallic gearbox collar on Z motor for Planetary / Worm belted-Z mods
            if (isPlanetary) {
              gearboxUsed = true;
              const gCollar = new THREE.Mesh(
                new THREE.CylinderGeometry(0.185, 0.195, 0.22, 20),
                gearboxMetalMat
              );
              gCollar.position.y = 0.15;
              tagInspectable(gCollar, zMotorInspect);
              zMotGrp.add(gCollar);

              const gBaseFlange = new THREE.Mesh(
                new THREE.CylinderGeometry(0.215, 0.215, 0.04, 20),
                gearboxHousingMat
              );
              gBaseFlange.position.y = 0.055;
              zMotGrp.add(gBaseFlange);

              const gTopBoss = new THREE.Mesh(
                new THREE.CylinderGeometry(0.145, 0.165, 0.055, 20),
                gearboxHousingMat
              );
              gTopBoss.position.y = 0.285;
              zMotGrp.add(gTopBoss);
            } else if (isWorm) {
              gearboxUsed = true;
              const wormBlock = new THREE.Mesh(
                new THREE.BoxGeometry(0.36, 0.26, 0.34),
                gearboxHousingMat
              );
              wormBlock.position.y = 0.17;
              tagInspectable(wormBlock, zMotorInspect);
              zMotGrp.add(wormBlock);

              const wormDrum = new THREE.Mesh(
                new THREE.CylinderGeometry(0.14, 0.14, 0.38, 18),
                gearboxMetalMat
              );
              wormDrum.rotation.z = Math.PI * 0.5;
              wormDrum.position.y = 0.18;
              tagInspectable(wormDrum, zMotorInspect);
              zMotGrp.add(wormDrum);

              const wormCollar = new THREE.Mesh(
                new THREE.CylinderGeometry(0.115, 0.125, 0.06, 16),
                gearboxMetalMat
              );
              wormCollar.position.y = 0.33;
              zMotGrp.add(wormCollar);
            }

            zMotGrp.position.copy(mPos);
            group.add(zMotGrp);
          }

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

            // Bottom 80T drive pulley (skip when CAD Z-drive GLB already includes 80T/20T/16T pulleys & belt)
            if (!hasCadZDrive) {
              const drive80t = new THREE.Mesh(
                new THREE.CylinderGeometry(0.18, 0.18, beltW * 1.15, 20),
                zPulleyMat
              );
              drive80t.rotation.z = Math.PI * 0.5;
              drive80t.position.set(p0.x, Math.max(p0.y + 0.08, mPos.y + 0.12), p0.z);
              tagInspectable(drive80t, zDriveInspect);
              group.add(drive80t);
            }

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
            const isBall = zl.kind === "ballscrew" || mechId.includes("sfu");
            const nominalDiamMm = zl.diameter_mm || (isHeavyBall ? 16 : isBall ? 12 : 8);
            const rShaft = Math.max(
              nominalDiamMm * 0.5 * SCALE,
              isHeavyBall ? 0.078 : 0.036
            );
            const shaftGeo = new THREE.CylinderGeometry(rShaft, rShaft, 1, 16);
            const shaftMat = isBall ? ballChromeMat : leadSteelMat;
            if (isBall) {
              ballChromeUsed = true;
              ballNutUsed = true;
            } else {
              leadUsed = true;
            }

            const shaftMesh = new THREE.Mesh(shaftGeo, shaftMat);
            orientBeamMesh(shaftMesh, p0, p1);
            tagInspectable(shaftMesh, zDriveInspect);
            group.add(shaftMesh);

            // Nut block at nut_z_mm
            const nutGrp = new THREE.Group();
            if (isBall) {
              // SFU1204 / SFU1605 flanged ballnut cylinder + BK10 or heavier FK12 bearing support block
              const nutH = isHeavyBall ? 0.42 : 0.34;
              const nutR = rShaft * (isHeavyBall ? 2.15 : 2.0);
              const flangeR = rShaft * (isHeavyBall ? 3.25 : 3.1);
              const flangeH = isHeavyBall ? 0.085 : 0.06;
              const nutCyl = new THREE.Mesh(
                new THREE.CylinderGeometry(nutR, nutR, nutH, 18),
                ballNutMat
              );
              tagInspectable(nutCyl, zDriveInspect);
              nutGrp.add(nutCyl);
              const nutFlange = new THREE.Mesh(
                new THREE.CylinderGeometry(flangeR, flangeR, flangeH, 18),
                ballNutMat
              );
              nutFlange.position.y = nutH * 0.42;
              nutGrp.add(nutFlange);

              if (isHeavyBall) {
                // Heavy FK12 flanged angular-contact bearing support block
                pomUsed = true;
                gearboxUsed = true;
                const fk12Grp = new THREE.Group();
                const fkFlange = new THREE.Mesh(
                  new THREE.BoxGeometry(0.44, 0.12, 0.44),
                  pomNutMat
                );
                fkFlange.position.y = 0.06;
                tagInspectable(fkFlange, zDriveInspect);
                fk12Grp.add(fkFlange);

                const fkBoss = new THREE.Mesh(
                  new THREE.CylinderGeometry(0.165, 0.175, 0.16, 18),
                  gearboxHousingMat
                );
                fkBoss.position.y = 0.18;
                tagInspectable(fkBoss, zDriveInspect);
                fk12Grp.add(fkBoss);

                const fkLockNut = new THREE.Mesh(
                  new THREE.CylinderGeometry(0.125, 0.125, 0.055, 16),
                  gearboxMetalMat
                );
                fkLockNut.position.y = 0.285;
                fk12Grp.add(fkLockNut);

                const boltGeo = new THREE.CylinderGeometry(0.024, 0.024, 0.04, 10);
                for (const bx of [-0.16, 0.16]) {
                  for (const bz of [-0.16, 0.16]) {
                    const bMesh = new THREE.Mesh(boltGeo, gearboxMetalMat);
                    bMesh.position.set(bx, 0.135, bz);
                    fk12Grp.add(bMesh);
                  }
                }
                fk12Grp.position.set(p0.x, p0.y + 0.02, p0.z);
                group.add(fk12Grp);
              } else {
                // BK10 bottom bearing block
                const bk10 = new THREE.Mesh(new THREE.BoxGeometry(0.32, 0.16, 0.26), pomNutMat);
                pomUsed = true;
                bk10.position.set(p0.x, p0.y + 0.1, p0.z);
                group.add(bk10);
              }
            } else {
              const isBrass = mechId.includes("brass");
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

            // Visible copper/brass WobbleX kinematic decoupler ring assembly on the nut carrier
            if (isWobbleX) {
              wobbleXUsed = true;
              ballChromeUsed = true;
              const wxBaseY = isBall ? 0.23 : 0.14;
              const wxRingR = Math.max(rShaft * 3.15, 0.145);

              const wxLowerRing = new THREE.Mesh(
                new THREE.CylinderGeometry(wxRingR, wxRingR, 0.075, 20),
                wobbleXCopperMat
              );
              wxLowerRing.position.y = wxBaseY;
              tagInspectable(wxLowerRing, zDriveInspect);
              nutGrp.add(wxLowerRing);

              const wxUpperRing = new THREE.Mesh(
                new THREE.CylinderGeometry(wxRingR * 0.94, wxRingR * 0.94, 0.065, 20),
                wobbleXCopperMat
              );
              wxUpperRing.position.y = wxBaseY + 0.105;
              tagInspectable(wxUpperRing, zDriveInspect);
              nutGrp.add(wxUpperRing);

              // Three 120°-spaced hardened steel decoupling balls visible between the copper WobbleX plates
              const wxBallGeo = new THREE.SphereGeometry(0.024, 10, 8);
              for (let bi = 0; bi < 3; bi++) {
                const ang = (bi * Math.PI * 2) / 3;
                const wxBall = new THREE.Mesh(wxBallGeo, ballChromeMat);
                wxBall.position.set(
                  Math.cos(ang) * wxRingR * 0.68,
                  wxBaseY + 0.055,
                  Math.sin(ang) * wxRingR * 0.68
                );
                nutGrp.add(wxBall);
              }
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
        if (!ballChromeUsed) ballChromeMat.dispose();
        if (!ballNutUsed) ballNutMat.dispose();
        if (!brassUsed) brassNutMat.dispose();
        if (!pomUsed) pomNutMat.dispose();
        if (!beltUsed) {
          zBeltMat.dispose();
          zPulleyMat.dispose();
        }
        if (!couplerUsed) couplerMat.dispose();
        if (!wobbleXUsed) wobbleXCopperMat.dispose();
        if (!gearboxUsed) {
          gearboxMetalMat.dispose();
          gearboxHousingMat.dispose();
        }
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
        const frameMember = geom.members.find(
          (m) => m.role === "frame" && !m.id.startsWith("col_")
        ) ?? geom.members.find((m) => m.role === "frame");
        const uprightMember =
          geom.members.find((m) => m.id.startsWith("col_")) ?? frameMember;
        const upHalfXMm = (uprightMember?.width_mm ?? frameMember?.width_mm ?? 20) * 0.5;
        const upHalfYMm = (uprightMember?.height_mm ?? frameMember?.width_mm ?? 20) * 0.5;
        const frameHalfZMm = (frameMember?.height_mm ?? frameMember?.width_mm ?? 20) * 0.5;
        const extHalfX = upHalfXMm * SCALE;
        const extHalfY = upHalfYMm * SCALE;
        const extHalfZ = frameHalfZMm * SCALE;
        const tScene = Math.max((panels.thickness_mm || 3) * SCALE, 0.024);

        // Full outer-perimeter panel dimensions matching true outer edge-to-edge frame faces
        const spanX = (fw + 2 * upHalfXMm) * SCALE;
        const spanY = (fd + 2 * upHalfYMm) * SCALE;
        const spanZ = (fh + 2 * frameHalfZMm) * SCALE;

        const isBoltedStructural = panels.mounting === "bolted_structural";
        const punchPanelScale = toolheadPunchIn ? 0.14 : 1.0;

        const resolveFaceVisual = (bay: "left" | "right" | "rear" | "top") => {
          const face = panels.faces?.[bay];
          const matId = face?.material_id ?? panels.material_id;
          const matName = face?.material_name ?? panels.material_name;
          const thickMm = face?.thickness_mm ?? panels.thickness_mm ?? 3;
          const isTransparent = matId === "acrylic_pmma" || matId === "polycarbonate";
          const colorHex = face?.color_hex ?? panels.color_hex ?? (isTransparent ? "#93c5fd" : "#475569");
          const tFaceScene = Math.max(thickMm * SCALE, 0.024);
          return { matId, matName, thickMm, isTransparent, colorHex, tFaceScene };
        };

        const leftVis = resolveFaceVisual("left");
        const rightVis = resolveFaceVisual("right");
        const rearVis = resolveFaceVisual("rear");
        const topVis = resolveFaceVisual("top");

        const edgeMat = new THREE.LineBasicMaterial({
          color: isBoltedStructural ? 0x38bdf8 : 0x94a3b8,
          transparent: true,
          opacity: (isBoltedStructural ? 0.65 : 0.26) * punchPanelScale,
        });

        const topPanelInstalled =
          panels.top_panel !== false && sim.spec.frame.top_panel !== false;

        const leftPanelGeo = new THREE.BoxGeometry(leftVis.tFaceScene, spanZ, spanY);
        const rightPanelGeo = new THREE.BoxGeometry(rightVis.tFaceScene, spanZ, spanY);
        const rearPanelGeo = new THREE.BoxGeometry(spanX, spanZ, rearVis.tFaceScene);
        const leftEdgesGeo = new THREE.EdgesGeometry(leftPanelGeo);
        const rightEdgesGeo = new THREE.EdgesGeometry(rightPanelGeo);
        const rearEdgesGeo = new THREE.EdgesGeometry(rearPanelGeo);

        // Mount flush on the OUTSIDE face of Left (-X), Right (+X), Rear (+Y CAD = -Z Three.js), and Top (+Z CAD = +Y Three.js) extrusions
        const leftCenter = toThree([0, fd * 0.5, fh * 0.5], cx, cy);
        const rightCenter = toThree([fw, fd * 0.5, fh * 0.5], cx, cy);
        const rearCenter = toThree([fw * 0.5, fd, fh * 0.5], cx, cy);
        const topCenter = toThree([fw * 0.5, fd * 0.5, fh], cx, cy);

        const leftPos = leftCenter.clone().add(new THREE.Vector3(-(extHalfX + leftVis.tFaceScene * 0.5), 0, 0));
        const rightPos = rightCenter.clone().add(new THREE.Vector3(extHalfX + rightVis.tFaceScene * 0.5, 0, 0));
        const rearPos = rearCenter.clone().add(new THREE.Vector3(0, 0, -(extHalfY + rearVis.tFaceScene * 0.5)));
        const topPos = topCenter.clone().add(new THREE.Vector3(0, extHalfZ + topVis.tFaceScene * 0.5, 0));

        const leftBoltX = leftPos.x - leftVis.tFaceScene * 0.55;
        const rightBoltX = rightPos.x + rightVis.tFaceScene * 0.55;
        const rearBoltZ = rearPos.z - rearVis.tFaceScene * 0.55;
        const topBoltY = topPos.y + topVis.tFaceScene * 0.55;

        const panelSpecs: Array<{
          bay: "left" | "right" | "rear" | "top";
          label: string;
          geo: THREE.BoxGeometry;
          edges: THREE.EdgesGeometry;
          pos: THREE.Vector3;
          vis: ReturnType<typeof resolveFaceVisual>;
          opacityMul?: number;
        }> = [
          { bay: "left", label: "Left Enclosure Panel", geo: leftPanelGeo, edges: leftEdgesGeo, pos: leftPos, vis: leftVis },
          { bay: "right", label: "Right Enclosure Panel", geo: rightPanelGeo, edges: rightEdgesGeo, pos: rightPos, vis: rightVis },
          { bay: "rear", label: "Rear Enclosure Panel", geo: rearPanelGeo, edges: rearEdgesGeo, pos: rearPos, vis: rearVis },
        ];
        if (topPanelInstalled) {
          const topPanelGeo = new THREE.BoxGeometry(spanX, topVis.tFaceScene, spanY);
          const topEdgesGeo = new THREE.EdgesGeometry(topPanelGeo);
          panelSpecs.push({
            bay: "top",
            label: "Top Enclosure Panel",
            geo: topPanelGeo,
            edges: topEdgesGeo,
            pos: topPos,
            vis: topVis,
            opacityMul: 0.78,
          });
        }

        for (const ps of panelSpecs) {
          const wallMat = new THREE.MeshStandardMaterial({
            color: new THREE.Color(ps.vis.colorHex),
            transparent: true,
            opacity:
              (ps.vis.isTransparent ? 0.16 : 0.32) *
              punchPanelScale *
              (ps.opacityMul ?? 1.0),
            depthWrite: false,
            roughness: ps.vis.isTransparent ? 0.14 : 0.38,
            metalness: ps.vis.isTransparent ? 0.1 : 0.42,
            side: THREE.DoubleSide,
          });
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
              material: ps.vis.matName,
              profile: `${ps.vis.thickMm}mm (${panels.mounting.replace(/_/g, " ")})`,
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

      // Trident Backbone by Clee (github.com/clee/TridentBackbone):
      // Authentic 3.18mm (1/8") laser-cut mild steel open-center perimeter plate with
      // front 20x20 corner post reliefs, recessed rear wings + rear corner ears,
      // 86x20mm split rear Z-motor tab with 20.4x20.1mm U-notch, 25x8mm rear wiring notch,
      // R10mm central cutout, 3x Ø26mm Z-motor bores, and 8x 3.6mm DIN/accessory slots.
      const showTridentBackbone =
        geom.architecture === "corexy" &&
        (Boolean(geom.trident_backbone?.active) ||
          (Boolean(sim.spec.frame.trident_backbone) &&
            !geom.panels?.is_flying_gantry &&
            sim.spec.bed.z_drive !== "belted_quad_z"));
      if (showTridentBackbone) {
        const [fw, fd] = geom.frame_dims_mm;
        const frameMember =
          geom.members.find((m) => m.role === "frame" && !m.id.startsWith("col_")) ??
          geom.members.find((m) => m.role === "frame");
        const uprightMember =
          geom.members.find((m) => m.id.startsWith("col_")) ?? frameMember;
        const extWidthMm = frameMember?.width_mm ?? 20;
        const extHalfMm = extWidthMm * 0.5;
        const upHalfXMm = (uprightMember?.width_mm ?? extWidthMm) * 0.5;
        const upHalfYMm = (uprightMember?.height_mm ?? extWidthMm) * 0.5;
        const bbExtra = geom.trident_backbone as
          | (NonNullable<typeof geom.trident_backbone> & {
              cutout_w_mm?: number;
              cutout_d_mm?: number;
            })
          | undefined;
        const plateThickMm = bbExtra?.thickness_mm ?? 3.18;
        const plateThickScene = Math.max(plateThickMm * SCALE, 0.026);

        const xL = -upHalfXMm;
        const xR = fw + upHalfXMm;
        const yF = -upHalfYMm;
        const yB = fd + upHalfYMm;
        const midX = fw * 0.5;

        // Outer contour features matching clee/TridentBackbone backbone-300.dxf
        const cornerRelief = extWidthMm; // 20x20mm front corner post reliefs for 2020 extrusions
        const rearRecess = extWidthMm; // 20mm recessed rear wings for rear extrusion/wiring clearance
        const yRearWing = yB - rearRecess;

        // Small rear corner ears inboard of rear vertical posts
        const rearEarWidth = Math.min(14.0, Math.max(6.0, fw * 0.035));
        const rearEarDepth = Math.min(7.0, Math.max(3.5, rearRecess * 0.35));

        // Integral protruding rear-center Z-motor tab (86mm wide x 20mm deep)
        const rearTabHalfW = Math.min(
          43.0,
          Math.max(24.0, (fw - 2 * cornerRelief - 2 * rearEarWidth - 36) * 0.35)
        );
        const rearTabL = midX - rearTabHalfW;
        const rearTabR = midX + rearTabHalfW;

        // 20.4 x 20.1mm U-shaped rear Z-rail clearance notch splitting the rear tab
        const zRailNotchHalfW = Math.min(rearTabHalfW - 8.0, Math.max(10.2, extHalfMm + 0.2));
        const zRailNotchDepth = Math.max(20.1, extWidthMm + 0.1);

        // 25 x 8mm rear wiring relief notch on the left side of the rear Z-motor tab
        const wireNotchAvailW = rearTabL - (xL + cornerRelief + rearEarWidth) - 8.0;
        const wireNotchW = Math.min(25.0, Math.max(0.0, wireNotchAvailW));
        const wireNotchDepth = wireNotchW > 4.0 ? 8.0 : 0.0;

        // 2D shape in (sx, sy) mapped via rotateX(-PI/2) -> (sx, sz, -sy) matching toThree
        const sx = (cadX: number) => (cadX - cx) * SCALE;
        const sy = (cadY: number) => (cadY - cy) * SCALE;

        // Outer perimeter polygon traced strictly CCW in (sx, sy)
        const bbShape = new THREE.Shape();
        bbShape.moveTo(sx(xL + cornerRelief), sy(yF));
        bbShape.lineTo(sx(xR - cornerRelief), sy(yF));
        // Front-Right 20x20 corner post relief
        bbShape.lineTo(sx(xR - cornerRelief), sy(yF + cornerRelief));
        bbShape.lineTo(sx(xR), sy(yF + cornerRelief));
        // Right edge up to recessed rear-right corner
        bbShape.lineTo(sx(xR), sy(yRearWing));
        bbShape.lineTo(sx(xR - cornerRelief), sy(yRearWing));
        // Rear-Right small corner ear near RR vertical post
        bbShape.lineTo(sx(xR - cornerRelief), sy(yRearWing + rearEarDepth));
        bbShape.lineTo(sx(xR - cornerRelief - rearEarWidth), sy(yRearWing + rearEarDepth));
        bbShape.lineTo(sx(xR - cornerRelief - rearEarWidth), sy(yRearWing));
        // Recessed rear-right wing -> protruding rear-center Z-motor tab
        bbShape.lineTo(sx(rearTabR), sy(yRearWing));
        bbShape.lineTo(sx(rearTabR), sy(yB));
        // 20.4 x 20.1mm U-shaped rear Z-rail clearance notch
        bbShape.lineTo(sx(midX + zRailNotchHalfW), sy(yB));
        bbShape.lineTo(sx(midX + zRailNotchHalfW), sy(yB - zRailNotchDepth));
        bbShape.lineTo(sx(midX - zRailNotchHalfW), sy(yB - zRailNotchDepth));
        bbShape.lineTo(sx(midX - zRailNotchHalfW), sy(yB));
        // Left half of rear-center Z-motor tab
        bbShape.lineTo(sx(rearTabL), sy(yB));
        // 25 x 8mm rear wiring relief notch on the left side of the rear Z-motor tab
        if (wireNotchW > 4.0) {
          bbShape.lineTo(sx(rearTabL), sy(yRearWing - wireNotchDepth));
          bbShape.lineTo(sx(rearTabL - wireNotchW), sy(yRearWing - wireNotchDepth));
          bbShape.lineTo(sx(rearTabL - wireNotchW), sy(yRearWing));
        } else {
          bbShape.lineTo(sx(rearTabL), sy(yRearWing));
        }
        // Recessed rear-left wing -> small rear-left corner ear near RL vertical post
        bbShape.lineTo(sx(xL + cornerRelief + rearEarWidth), sy(yRearWing));
        bbShape.lineTo(sx(xL + cornerRelief + rearEarWidth), sy(yRearWing + rearEarDepth));
        bbShape.lineTo(sx(xL + cornerRelief), sy(yRearWing + rearEarDepth));
        bbShape.lineTo(sx(xL + cornerRelief), sy(yRearWing));
        bbShape.lineTo(sx(xL), sy(yRearWing));
        // Left edge down to Front-Left 20x20 corner post relief
        bbShape.lineTo(sx(xL), sy(yF + cornerRelief));
        bbShape.lineTo(sx(xL + cornerRelief), sy(yF + cornerRelief));
        bbShape.closePath();

        // Helper for strictly CW rounded-rectangle / stadium-slot holes in (sx, sy)
        const addCwRoundedRectHole = (
          x0Mm: number,
          y0Mm: number,
          x1Mm: number,
          y1Mm: number,
          radiusMm: number
        ) => {
          const x0 = sx(x0Mm);
          const x1 = sx(x1Mm);
          const y0 = sy(y0Mm);
          const y1 = sy(y1Mm);
          const w = x1 - x0;
          const h = y1 - y0;
          if (w <= 1e-4 || h <= 1e-4) return;
          const r = Math.min(radiusMm * SCALE, w * 0.48, h * 0.48);
          const hole = new THREE.Path();
          hole.moveTo(x0, y0 + r);
          hole.lineTo(x0, y1 - r);
          hole.absarc(x0 + r, y1 - r, r, Math.PI, Math.PI * 0.5, true);
          hole.lineTo(x1 - r, y1);
          hole.absarc(x1 - r, y1 - r, r, Math.PI * 0.5, 0, true);
          hole.lineTo(x1, y0 + r);
          hole.absarc(x1 - r, y0 + r, r, 0, -Math.PI * 0.5, true);
          hole.lineTo(x0 + r, y0);
          hole.absarc(x0 + r, y0 + r, r, -Math.PI * 0.5, -Math.PI, true);
          hole.closePath();
          bbShape.holes.push(hole);
        };

        // 3x Ø26.0mm Z-motor pilot/leadscrew bore positions (exact DXF offsets for 2020; clears thicker posts too)
        const boreFrontOffsetMm = Math.max(41.8, cornerRelief + 21.8);
        const boreRearOffsetMm = Math.max(49.9, zRailNotchDepth + 29.8);
        const boreRadiusMm = 13.0; // Ø26.0mm
        const zMotorBoresMm: Array<[number, number]> = [
          [xL + boreFrontOffsetMm, yF + boreFrontOffsetMm],
          [xR - boreFrontOffsetMm, yF + boreFrontOffsetMm],
          [midX, yB - boreRearOffsetMm],
        ];

        // TRUE OPEN CENTRAL CUTOUT with R10mm corner fillets leaving ~80mm wide perimeter steel flanges
        const minSideFlangeMm = boreFrontOffsetMm + 24.0;
        const minFrontFlangeMm = boreFrontOffsetMm + 24.0;
        const minRearFlangeMm = boreRearOffsetMm + 24.0;
        const rawCutW = bbExtra?.cutout_w_mm ?? fw - 160.0;
        const rawCutD = bbExtra?.cutout_d_mm ?? fd - 180.0;
        const maxCutW = Math.max(0, xR - xL - 2 * minSideFlangeMm);
        const maxCutD = Math.max(0, yB - minRearFlangeMm - (yF + minFrontFlangeMm));
        const cutW = Math.min(Math.max(rawCutW, 0), maxCutW);
        const cutD = Math.min(Math.max(rawCutD, 0), maxCutD);

        const cutXMin = midX - cutW * 0.5;
        const cutXMax = midX + cutW * 0.5;
        const yCutCenter = 0.5 * (yF + yRearWing);
        const cutYMin = Math.max(yF + minFrontFlangeMm, yCutCenter - cutD * 0.5);
        const cutYMax = Math.min(yB - minRearFlangeMm, cutYMin + cutD);

        if (cutW >= 24.0 && cutYMax - cutYMin >= 24.0) {
          const cutFilletR = Math.min(10.0, cutW * 0.25, (cutYMax - cutYMin) * 0.25);
          addCwRoundedRectHole(cutXMin, cutYMin, cutXMax, cutYMax, cutFilletR);
        }

        // TRUE PUNCHED 3x Ø26.0mm circular Z-motor pilot/leadscrew bores (strictly CW)
        for (const [bxMm, byMm] of zMotorBoresMm) {
          const boreHole = new THREE.Path();
          boreHole.absarc(sx(bxMm), sy(byMm), boreRadiusMm * SCALE, 0, Math.PI * 2, true);
          bbShape.holes.push(boreHole);
        }

        // TRUE PUNCHED 8x narrow 3.6mm DIN/accessory mounting slots through the perimeter steel flanges:
        // 4x 50x3.6mm vertical slots (2 left flange, 2 right flange) + 4x 40x3.6mm horizontal slots (2 front, 2 rear)
        const slotWidthMm = 3.6;
        const slotHalfW = slotWidthMm * 0.5;
        const slotRadiusMm = slotHalfW;
        const flangeClearMinMm = 12.0;

        // Left & right flange vertical 50 x 3.6mm slots
        const leftOpenSpanX = cutXMin - (xL + extWidthMm);
        const rightOpenSpanX = xR - extWidthMm - cutXMax;
        const leftSlotX = 0.5 * (xL + extWidthMm + cutXMin);
        const rightSlotX = 0.5 * (cutXMax + (xR - extWidthMm));
        const vSpanMin = yF + boreFrontOffsetMm + 30.0;
        const vSpanMax = yRearWing - 22.0;
        const vSpan = vSpanMax - vSpanMin;
        if (leftOpenSpanX >= flangeClearMinMm && rightOpenSpanX >= flangeClearMinMm && vSpan >= 52.0) {
          const vSlotLen = Math.min(50.0, Math.max(16.0, (vSpan - 20.0) * 0.5));
          const vHalf = vSlotLen * 0.5;
          const vCentersY = [vSpanMin + vSpan * 0.27, vSpanMin + vSpan * 0.73];
          for (const cyMm of vCentersY) {
            addCwRoundedRectHole(
              leftSlotX - slotHalfW,
              cyMm - vHalf,
              leftSlotX + slotHalfW,
              cyMm + vHalf,
              slotRadiusMm
            );
            addCwRoundedRectHole(
              rightSlotX - slotHalfW,
              cyMm - vHalf,
              rightSlotX + slotHalfW,
              cyMm + vHalf,
              slotRadiusMm
            );
          }
        }

        // Front flange horizontal 40 x 3.6mm slots (2x)
        const frontOpenSpanY = cutYMin - (yF + extWidthMm);
        const frontSlotY = 0.5 * (yF + extWidthMm + cutYMin);
        const hFrontMin = xL + boreFrontOffsetMm + 30.0;
        const hFrontMax = xR - boreFrontOffsetMm - 30.0;
        const hFrontSpan = hFrontMax - hFrontMin;
        if (frontOpenSpanY >= flangeClearMinMm && hFrontSpan >= 44.0) {
          const hFrontLen = Math.min(40.0, Math.max(14.0, (hFrontSpan - 20.0) * 0.5));
          const hFrontHalf = hFrontLen * 0.5;
          const fCentersX = [hFrontMin + hFrontSpan * 0.27, hFrontMin + hFrontSpan * 0.73];
          for (const cxMm of fCentersX) {
            addCwRoundedRectHole(
              cxMm - hFrontHalf,
              frontSlotY - slotHalfW,
              cxMm + hFrontHalf,
              frontSlotY + slotHalfW,
              slotRadiusMm
            );
          }
        }

        // Rear flange horizontal 40 x 3.6mm slots (2x: rear-left wing & rear-right wing)
        const rearSlotTopLimit = yRearWing - wireNotchDepth;
        const rearOpenSpanY = rearSlotTopLimit - cutYMax;
        const rearSlotY = 0.5 * (cutYMax + rearSlotTopLimit);
        const rearLeftMin = Math.max(leftSlotX + 12.0, xL + 46.0);
        const rearLeftMax = midX - 28.0;
        const rearWingSpan = rearLeftMax - rearLeftMin;
        if (rearOpenSpanY >= flangeClearMinMm && rearWingSpan >= 24.0) {
          const hRearLen = Math.min(40.0, Math.max(12.0, rearWingSpan - 10.0));
          const hRearHalf = hRearLen * 0.5;
          const rearLeftCenterX = 0.5 * (rearLeftMin + rearLeftMax);
          const rearRightCenterX = fw - rearLeftCenterX;
          for (const cxMm of [rearLeftCenterX, rearRightCenterX]) {
            addCwRoundedRectHole(
              cxMm - hRearHalf,
              rearSlotY - slotHalfW,
              cxMm + hRearHalf,
              rearSlotY + slotHalfW,
              slotRadiusMm
            );
          }
        }

        const backboneGeo = new THREE.ExtrudeGeometry(bbShape, {
          depth: plateThickScene,
          bevelEnabled: false,
          curveSegments: 16,
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
            name: "Trident Backbone (Clee, 3.18mm Mild Steel Perimeter Plate)",
            subsystem: "frame",
            partGroupId: "trident_backbone",
            role: "panel",
            material: geom.trident_backbone?.material_name ?? "Powder-Coated Mild Steel",
            profile: `${plateThickMm.toFixed(2)}mm laser-cut open-center perimeter plate`,
            complianceSharePct: bbBaySens?.compliance_share_pct ?? bbStiff?.compliance_share_pct,
            heatScore: bbBaySens ? getLensScore(bbBaySens, adjointLens) : undefined,
            estHzGain: bbStiff?.est_hz_gain,
            hzPer100g: bbBaySens?.hz_per_100g,
            whyItHurts: bbStiff?.why_it_hurts,
            actionHint: bbStiff?.action_hint,
          });
        }
        group.add(backboneMesh);

        // Crisp metallic edge highlight outlining outer perimeter, R10 central cutout, Ø26 bores & 3.6mm slots
        const backboneEdgesGeo = new THREE.EdgesGeometry(backboneGeo, 28);
        const backboneEdgesMat = new THREE.LineBasicMaterial({
          color: 0x475569,
          transparent: true,
          opacity: toolheadPunchIn ? 0.2 : 0.85,
        });
        const backboneEdges = new THREE.LineSegments(backboneEdgesGeo, backboneEdgesMat);
        backboneEdges.position.y = plateBaseY;
        group.add(backboneEdges);

        // Hardware details on top of the plate:
        // 12x M3 stainless socket-head bolts (4 on a 31x31mm square around each of the 3x Ø26mm Z-motor bores)
        // + 10x M5 frame-mounting socket-head bolts (4 FL, 4 FR, 2 on the split rear-center tab)
        if (!toolheadPunchIn) {
          const m3BoltCoords: THREE.Vector3[] = [];
          const m3OffsetMm = 15.5; // NEMA-17 standard 31 x 31mm M3 bolt pattern
          const m3BoltY = plateTopY + 0.01;

          for (const [bxMm, byMm] of zMotorBoresMm) {
            for (const dxMm of [-m3OffsetMm, m3OffsetMm]) {
              for (const dyMm of [-m3OffsetMm, m3OffsetMm]) {
                const bPos = toThree([bxMm + dxMm, byMm + dyMm, extHalfMm + plateThickMm], cx, cy);
                m3BoltCoords.push(new THREE.Vector3(bPos.x, m3BoltY, bPos.z));
              }
            }
          }

          if (m3BoltCoords.length > 0) {
            const m3BoltGeo = new THREE.CylinderGeometry(0.024, 0.024, 0.02, 10);
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

          // 10x M5 frame-mounting socket-head bolts on 2020 deck extrusion centerlines:
          // 4 FL (2 front + 2 left), 4 FR (2 front + 2 right), 2 on the rear-center Z-motor tab
          const m5OffNear = 16.0;
          const m5OffFar = 52.0;
          const rearTabBoltOffsetX = 0.5 * (zRailNotchHalfW + rearTabHalfW);
          const m5PointsMm: Array<[number, number]> = [
            // 4x Front-Left
            [xL + cornerRelief + m5OffNear, 0],
            [xL + cornerRelief + m5OffFar, 0],
            [0, yF + cornerRelief + m5OffNear],
            [0, yF + cornerRelief + m5OffFar],
            // 4x Front-Right
            [xR - cornerRelief - m5OffNear, 0],
            [xR - cornerRelief - m5OffFar, 0],
            [fw, yF + cornerRelief + m5OffNear],
            [fw, yF + cornerRelief + m5OffFar],
            // 2x Rear-center Z-motor tab
            [midX - rearTabBoltOffsetX, fd],
            [midX + rearTabBoltOffsetX, fd],
          ];

          const m5BoltY = plateTopY + 0.015;
          const bbBoltGeo = new THREE.CylinderGeometry(0.036, 0.036, 0.028, 10);
          const bbBoltMat = new THREE.MeshStandardMaterial({
            color: 0xdbe4ee,
            metalness: 0.86,
            roughness: 0.24,
          });
          const bbBoltInst = new THREE.InstancedMesh(bbBoltGeo, bbBoltMat, m5PointsMm.length);
          const bbDummy = new THREE.Object3D();
          for (let i = 0; i < m5PointsMm.length; i++) {
            const [mxMm, myMm] = m5PointsMm[i];
            const bPos = toThree([mxMm, myMm, extHalfMm + plateThickMm], cx, cy);
            bbDummy.position.set(bPos.x, m5BoltY, bPos.z);
            bbDummy.updateMatrix();
            bbBoltInst.setMatrixAt(i, bbDummy.matrix);
          }
          bbBoltInst.instanceMatrix.needsUpdate = true;
          group.add(bbBoltInst);
        }
      }

      // Helper for strictly CW rounded-rectangle cutout holes in any 2D THREE.Shape
      const addCwRoundedRectHole2D = (
        targetShape: THREE.Shape,
        x0: number,
        y0: number,
        x1: number,
        y1: number,
        rRaw: number
      ) => {
        const w = x1 - x0;
        const h = y1 - y0;
        if (w <= 1e-4 || h <= 1e-4) return;
        const r = Math.min(Math.max(rRaw, 0), w * 0.48, h * 0.48);
        const hole = new THREE.Path();
        hole.moveTo(x0, y0 + r);
        hole.lineTo(x0, y1 - r);
        hole.absarc(x0 + r, y1 - r, r, Math.PI, Math.PI * 0.5, true);
        hole.lineTo(x1 - r, y1);
        hole.absarc(x1 - r, y1 - r, r, Math.PI * 0.5, 0, true);
        hole.lineTo(x1, y0 + r);
        hole.absarc(x1 - r, y0 + r, r, 0, -Math.PI * 0.5, true);
        hole.lineTo(x0 + r, y0);
        hole.absarc(x0 + r, y0 + r, r, -Math.PI * 0.5, -Math.PI, true);
        hole.closePath();
        targetShape.holes.push(hole);
      };

      // Open-Center Top CNC Halo Ring (5mm brushed 6061 aluminum plate resting flush on top rails)
      const topBrace = geom.top_brace;
      if (
        geom.architecture === "corexy" &&
        Boolean(topBrace?.active) &&
        topBrace?.id === "cnc_halo_ring"
      ) {
        const [fw, fd, fh] = geom.frame_dims_mm;
        const frameMember =
          geom.members.find((m) => m.role === "frame" && !m.id.startsWith("col_")) ??
          geom.members.find((m) => m.role === "frame");
        const uprightMember =
          geom.members.find((m) => m.id.startsWith("col_")) ?? frameMember;
        const extWidthMm = frameMember?.width_mm ?? 20;
        const extHalfMm = extWidthMm * 0.5;
        const upHalfXMm = (uprightMember?.width_mm ?? extWidthMm) * 0.5;
        const upHalfYMm = (uprightMember?.height_mm ?? extWidthMm) * 0.5;
        const plateThickMm = topBrace.thickness_mm > 0 ? topBrace.thickness_mm : 5.0;
        const plateThickScene = Math.max(plateThickMm * SCALE, 0.032);

        const xL = -upHalfXMm;
        const xR = fw + upHalfXMm;
        const yF = -upHalfYMm;
        const yB = fd + upHalfYMm;
        const sx = (cadX: number) => (cadX - cx) * SCALE;
        const sy = (cadY: number) => (cadY - cy) * SCALE;

        const haloShape = new THREE.Shape();
        haloShape.moveTo(sx(xL), sy(yF));
        haloShape.lineTo(sx(xR), sy(yF));
        haloShape.lineTo(sx(xR), sy(yB));
        haloShape.lineTo(sx(xL), sy(yB));
        haloShape.closePath();

        const flangeMm = topBrace.flange_width_mm > 0 ? topBrace.flange_width_mm : 50.0;
        const rawCutW =
          topBrace.cutout_w_mm > 0 ? topBrace.cutout_w_mm : Math.max(80, fw - 2 * flangeMm);
        const rawCutD =
          topBrace.cutout_d_mm > 0 ? topBrace.cutout_d_mm : Math.max(80, fd - 2 * flangeMm);
        const maxCutW = Math.max(0, xR - xL - 2 * extWidthMm - 16);
        const maxCutD = Math.max(0, yB - yF - 2 * extWidthMm - 16);
        const cutW = Math.min(rawCutW, maxCutW);
        const cutD = Math.min(rawCutD, maxCutD);

        if (cutW >= 24 && cutD >= 24) {
          const midX = fw * 0.5;
          const midY = fd * 0.5;
          const rMm = Math.min(20.0, cutW * 0.25, cutD * 0.25);
          addCwRoundedRectHole2D(
            haloShape,
            sx(midX - cutW * 0.5),
            sy(midY - cutD * 0.5),
            sx(midX + cutW * 0.5),
            sy(midY + cutD * 0.5),
            rMm * SCALE
          );
        }

        const haloGeo = new THREE.ExtrudeGeometry(haloShape, {
          depth: plateThickScene,
          bevelEnabled: false,
          curveSegments: 16,
        });
        haloGeo.rotateX(-Math.PI / 2);
        haloGeo.computeVertexNormals();

        const plateBaseY = toThree([0, 0, fh + extHalfMm], cx, cy).y + 0.002;
        const plateTopY = plateBaseY + plateThickScene;

        const haloMat = new THREE.MeshStandardMaterial({
          color: 0xcbd5e1,
          metalness: 0.84,
          roughness: 0.25,
          transparent: toolheadPunchIn,
          opacity: toolheadPunchIn ? 0.16 : 1.0,
        });
        const haloMesh = new THREE.Mesh(haloGeo, haloMat);
        haloMesh.position.y = plateBaseY;

        const tbBaySens = sim.sensitivity?.panel_bay_sensitivities?.["top_brace"];
        const tbStiff = stiffenById["top_brace"] ?? stiffenById["bay_top_brace"];
        if (!toolheadPunchIn) {
          tagInspectable(haloMesh, {
            key: "frame:top_brace",
            name: `Top CNC Halo Ring (${Math.round(plateThickMm)}mm Al)`,
            subsystem: "frame",
            partGroupId: "top_brace",
            role: "panel",
            material: topBrace.material_name || "6061-T6 Aluminum",
            profile: `${plateThickMm.toFixed(1)}mm CNC halo ring (${Math.round(cutW)}×${Math.round(cutD)}mm open center)`,
            complianceSharePct: tbBaySens?.compliance_share_pct ?? tbStiff?.compliance_share_pct,
            heatScore: tbBaySens ? getLensScore(tbBaySens, adjointLens) : undefined,
            estHzGain: tbStiff?.est_hz_gain,
            hzPer100g: tbBaySens?.hz_per_100g,
            whyItHurts: tbStiff?.why_it_hurts,
            actionHint: tbStiff?.action_hint,
          });
        }
        group.add(haloMesh);

        const haloEdgesGeo = new THREE.EdgesGeometry(haloGeo, 28);
        const haloEdgesMat = new THREE.LineBasicMaterial({
          color: 0x64748b,
          transparent: true,
          opacity: toolheadPunchIn ? 0.16 : 0.85,
        });
        const haloEdges = new THREE.LineSegments(haloEdgesGeo, haloEdgesMat);
        haloEdges.position.y = plateBaseY;
        group.add(haloEdges);

        // Perimeter M5 socket-head bolts seated along top frame extrusion centerlines
        if (!toolheadPunchIn) {
          const steps = 5;
          const boltCoords: THREE.Vector3[] = [];
          const boltY = plateTopY + 0.013;
          for (let i = 0; i <= steps; i++) {
            const t = i / steps;
            const bx = t * fw;
            const by = t * fd;
            const pf = toThree([bx, 0, fh + extHalfMm + plateThickMm], cx, cy);
            const pr = toThree([bx, fd, fh + extHalfMm + plateThickMm], cx, cy);
            boltCoords.push(new THREE.Vector3(pf.x, boltY, pf.z));
            boltCoords.push(new THREE.Vector3(pr.x, boltY, pr.z));
            if (i > 0 && i < steps) {
              const pl = toThree([0, by, fh + extHalfMm + plateThickMm], cx, cy);
              const pri = toThree([fw, by, fh + extHalfMm + plateThickMm], cx, cy);
              boltCoords.push(new THREE.Vector3(pl.x, boltY, pl.z));
              boltCoords.push(new THREE.Vector3(pri.x, boltY, pri.z));
            }
          }
          const boltGeo = new THREE.CylinderGeometry(0.034, 0.034, 0.026, 10);
          const boltMat = new THREE.MeshStandardMaterial({
            color: 0xe2e8f0,
            metalness: 0.88,
            roughness: 0.22,
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
      }

      // Open-Door Front CNC Portal Frame (5mm brushed 6061 aluminum picture-frame plate at y = -extHalf)
      const frontBrace = geom.front_brace;
      if (
        geom.architecture === "corexy" &&
        Boolean(frontBrace?.active) &&
        frontBrace?.id === "cnc_portal_frame"
      ) {
        const [fw, , fh] = geom.frame_dims_mm;
        const frameMember =
          geom.members.find((m) => m.role === "frame" && !m.id.startsWith("col_")) ??
          geom.members.find((m) => m.role === "frame");
        const uprightMember =
          geom.members.find((m) => m.id.startsWith("col_")) ?? frameMember;
        const extWidthMm = frameMember?.width_mm ?? 20;
        const extHalfMm = extWidthMm * 0.5;
        const upHalfXMm = (uprightMember?.width_mm ?? extWidthMm) * 0.5;
        const upHalfYMm = (uprightMember?.height_mm ?? extWidthMm) * 0.5;
        const plateThickMm = frontBrace.thickness_mm > 0 ? frontBrace.thickness_mm : 5.0;
        const plateThickScene = Math.max(plateThickMm * SCALE, 0.032);

        const xL = -upHalfXMm;
        const xR = fw + upHalfXMm;
        const zB = -extHalfMm;
        const zT = fh + extHalfMm;
        const sx = (cadX: number) => (cadX - cx) * SCALE;
        const sz = (cadZ: number) => cadZ * SCALE;

        const portalShape = new THREE.Shape();
        portalShape.moveTo(sx(xL), sz(zB));
        portalShape.lineTo(sx(xR), sz(zB));
        portalShape.lineTo(sx(xR), sz(zT));
        portalShape.lineTo(sx(xL), sz(zT));
        portalShape.closePath();

        const flangeMm = frontBrace.flange_width_mm > 0 ? frontBrace.flange_width_mm : 55.0;
        const rawCutW =
          frontBrace.cutout_w_mm > 0 ? frontBrace.cutout_w_mm : Math.max(80, fw - 2 * flangeMm);
        const rawCutH =
          frontBrace.cutout_h_mm > 0 ? frontBrace.cutout_h_mm : Math.max(80, fh - 2 * flangeMm);
        const maxCutW = Math.max(0, xR - xL - 2 * extWidthMm - 16);
        const maxCutH = Math.max(0, zT - zB - 2 * extWidthMm - 16);
        const cutW = Math.min(rawCutW, maxCutW);
        const cutH = Math.min(rawCutH, maxCutH);

        if (cutW >= 24 && cutH >= 24) {
          const midX = fw * 0.5;
          const midZ = fh * 0.5;
          const rMm = Math.min(20.0, cutW * 0.25, cutH * 0.25);
          addCwRoundedRectHole2D(
            portalShape,
            sx(midX - cutW * 0.5),
            sz(midZ - cutH * 0.5),
            sx(midX + cutW * 0.5),
            sz(midZ + cutH * 0.5),
            rMm * SCALE
          );
        }

        const portalGeo = new THREE.ExtrudeGeometry(portalShape, {
          depth: plateThickScene,
          bevelEnabled: false,
          curveSegments: 16,
        });
        portalGeo.computeVertexNormals();

        // Mount flush on front outer face CAD y = -upHalfYMm (Three.js +Z outward)
        const frontOuterZ = toThree([0, -upHalfYMm, 0], cx, cy).z + 0.002;
        const frontFaceZ = frontOuterZ + plateThickScene;

        const portalMat = new THREE.MeshStandardMaterial({
          color: 0xcbd5e1,
          metalness: 0.84,
          roughness: 0.25,
          transparent: toolheadPunchIn,
          opacity: toolheadPunchIn ? 0.16 : 1.0,
        });
        const portalMesh = new THREE.Mesh(portalGeo, portalMat);
        portalMesh.position.z = frontOuterZ;

        const fbBaySens = sim.sensitivity?.panel_bay_sensitivities?.["front_brace"];
        const fbStiff = stiffenById["front_brace"] ?? stiffenById["bay_front_brace"];
        if (!toolheadPunchIn) {
          tagInspectable(portalMesh, {
            key: "frame:front_brace",
            name: `Front CNC Portal Frame (${Math.round(plateThickMm)}mm Al)`,
            subsystem: "frame",
            partGroupId: "front_brace",
            role: "panel",
            material: frontBrace.material_name || "6061-T6 Aluminum",
            profile: `${plateThickMm.toFixed(1)}mm CNC front portal (${Math.round(cutW)}×${Math.round(cutH)}mm clear door)`,
            complianceSharePct: fbBaySens?.compliance_share_pct ?? fbStiff?.compliance_share_pct,
            heatScore: fbBaySens ? getLensScore(fbBaySens, adjointLens) : undefined,
            estHzGain: fbStiff?.est_hz_gain,
            hzPer100g: fbBaySens?.hz_per_100g,
            whyItHurts: fbStiff?.why_it_hurts,
            actionHint: fbStiff?.action_hint,
          });
        }
        group.add(portalMesh);

        const portalEdgesGeo = new THREE.EdgesGeometry(portalGeo, 28);
        const portalEdgesMat = new THREE.LineBasicMaterial({
          color: 0x64748b,
          transparent: true,
          opacity: toolheadPunchIn ? 0.16 : 0.85,
        });
        const portalEdges = new THREE.LineSegments(portalEdgesGeo, portalEdgesMat);
        portalEdges.position.z = frontOuterZ;
        group.add(portalEdges);

        // Perimeter M5 socket-head bolts seated along front perimeter extrusion centerlines
        if (!toolheadPunchIn) {
          const steps = 5;
          const boltCoords: THREE.Vector3[] = [];
          const boltZ = frontFaceZ + 0.013;
          for (let i = 0; i <= steps; i++) {
            const t = i / steps;
            const bx = t * fw;
            const bz = t * fh;
            boltCoords.push(new THREE.Vector3(sx(bx), sz(0), boltZ));
            boltCoords.push(new THREE.Vector3(sx(bx), sz(fh), boltZ));
            if (i > 0 && i < steps) {
              boltCoords.push(new THREE.Vector3(sx(0), sz(bz), boltZ));
              boltCoords.push(new THREE.Vector3(sx(fw), sz(bz), boltZ));
            }
          }
          const boltGeo = new THREE.CylinderGeometry(0.034, 0.034, 0.026, 10);
          boltGeo.rotateX(Math.PI / 2);
          const boltMat = new THREE.MeshStandardMaterial({
            color: 0xe2e8f0,
            metalness: 0.88,
            roughness: 0.22,
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
      }

      // Open-Door Front Upper Header + Lower Skirt (two 5mm brushed 6061 aluminum plates at y = -extHalf)
      if (
        geom.architecture === "corexy" &&
        Boolean(frontBrace?.active) &&
        frontBrace?.id === "header_and_skirt"
      ) {
        const [fw, , fh] = geom.frame_dims_mm;
        const frameMember =
          geom.members.find((m) => m.role === "frame" && !m.id.startsWith("col_")) ??
          geom.members.find((m) => m.role === "frame");
        const uprightMember =
          geom.members.find((m) => m.id.startsWith("col_")) ?? frameMember;
        const extWidthMm = frameMember?.width_mm ?? 20;
        const extHalfMm = extWidthMm * 0.5;
        const upHalfXMm = (uprightMember?.width_mm ?? extWidthMm) * 0.5;
        const upHalfYMm = (uprightMember?.height_mm ?? extWidthMm) * 0.5;
        const plateThickMm = frontBrace.thickness_mm > 0 ? frontBrace.thickness_mm : 5.0;
        const plateThickScene = Math.max(plateThickMm * SCALE, 0.032);

        const xL = -upHalfXMm;
        const xR = fw + upHalfXMm;
        const sx = (cadX: number) => (cadX - cx) * SCALE;
        const sz = (cadZ: number) => cadZ * SCALE;

        const gantryTopClr = geom.gantry_top_clearance_mm ?? sim.spec.frame.gantry_top_clearance_mm ?? 55;
        const headerHeightMm =
          frontBrace.header_height_mm > 0 ? frontBrace.header_height_mm : gantryTopClr;
        const skirtHeightMm =
          frontBrace.skirt_height_mm > 0 ? frontBrace.skirt_height_mm : 65.0;

        const zHeaderTop = fh + extHalfMm;
        const zHeaderBot = Math.max(extHalfMm + 20, fh - headerHeightMm);
        const zSkirtBot = -extHalfMm;
        const zSkirtTop = Math.min(zHeaderBot - 20, Math.max(extWidthMm, skirtHeightMm));
        const clearDoorH = Math.max(0, zHeaderBot - zSkirtTop);

        const frontOuterZ = toThree([0, -upHalfYMm, 0], cx, cy).z + 0.002;
        const frontFaceZ = frontOuterZ + plateThickScene;

        const hsMat = new THREE.MeshStandardMaterial({
          color: 0xcbd5e1,
          metalness: 0.84,
          roughness: 0.25,
          transparent: toolheadPunchIn,
          opacity: toolheadPunchIn ? 0.16 : 1.0,
        });
        const hsEdgesMat = new THREE.LineBasicMaterial({
          color: 0x64748b,
          transparent: true,
          opacity: toolheadPunchIn ? 0.16 : 0.85,
        });

        const fbBaySens = sim.sensitivity?.panel_bay_sensitivities?.["front_brace"];
        const fbStiff = stiffenById["front_brace"] ?? stiffenById["bay_front_brace"];
        const hsInspectInfo: InspectedPartInfo = {
          key: "frame:front_brace",
          name: `Front Upper Header + Lower Skirt (${Math.round(plateThickMm)}mm Al)`,
          subsystem: "frame",
          partGroupId: "front_brace",
          role: "panel",
          material: frontBrace.material_name || "6061-T6 Aluminum",
          profile: `${plateThickMm.toFixed(1)}mm CNC header (${Math.round(headerHeightMm)}mm) + skirt (${Math.round(skirtHeightMm)}mm) · ${Math.round(clearDoorH)}mm clear door`,
          complianceSharePct: fbBaySens?.compliance_share_pct ?? fbStiff?.compliance_share_pct,
          heatScore: fbBaySens ? getLensScore(fbBaySens, adjointLens) : undefined,
          estHzGain: fbStiff?.est_hz_gain,
          hzPer100g: fbBaySens?.hz_per_100g,
          whyItHurts: fbStiff?.why_it_hurts,
          actionHint: fbStiff?.action_hint,
        };

        const addFrontRectPlate = (z0Mm: number, z1Mm: number) => {
          const rectShape = new THREE.Shape();
          rectShape.moveTo(sx(xL), sz(z0Mm));
          rectShape.lineTo(sx(xR), sz(z0Mm));
          rectShape.lineTo(sx(xR), sz(z1Mm));
          rectShape.lineTo(sx(xL), sz(z1Mm));
          rectShape.closePath();

          const geo = new THREE.ExtrudeGeometry(rectShape, {
            depth: plateThickScene,
            bevelEnabled: false,
            curveSegments: 8,
          });
          geo.computeVertexNormals();
          const mesh = new THREE.Mesh(geo, hsMat);
          mesh.position.z = frontOuterZ;
          if (!toolheadPunchIn) {
            tagInspectable(mesh, hsInspectInfo);
          }
          group.add(mesh);

          const edgesGeo = new THREE.EdgesGeometry(geo, 28);
          const edges = new THREE.LineSegments(edgesGeo, hsEdgesMat);
          edges.position.z = frontOuterZ;
          group.add(edges);
        };

        addFrontRectPlate(zHeaderBot, zHeaderTop);
        addFrontRectPlate(zSkirtBot, zSkirtTop);

        // Perimeter M5 socket-head bolts along header top/sides and skirt bottom/sides
        if (!toolheadPunchIn) {
          const steps = 5;
          const boltCoords: THREE.Vector3[] = [];
          const boltZ = frontFaceZ + 0.013;
          for (let i = 0; i <= steps; i++) {
            const t = i / steps;
            const bx = t * fw;
            boltCoords.push(new THREE.Vector3(sx(bx), sz(fh), boltZ));
            boltCoords.push(new THREE.Vector3(sx(bx), sz(0), boltZ));
          }
          // Side column bolts near lower edge of upper header and upper edge of lower skirt
          const headerMidZ = Math.max(zHeaderBot + 10, fh - Math.min(headerHeightMm * 0.75, headerHeightMm - 10));
          const skirtUpperZ = Math.min(zSkirtTop - 10, Math.max(20, skirtHeightMm * 0.75));
          boltCoords.push(new THREE.Vector3(sx(0), sz(headerMidZ), boltZ));
          boltCoords.push(new THREE.Vector3(sx(fw), sz(headerMidZ), boltZ));
          boltCoords.push(new THREE.Vector3(sx(0), sz(skirtUpperZ), boltZ));
          boltCoords.push(new THREE.Vector3(sx(fw), sz(skirtUpperZ), boltZ));

          const boltGeo = new THREE.CylinderGeometry(0.034, 0.034, 0.026, 10);
          boltGeo.rotateX(Math.PI / 2);
          const boltMat = new THREE.MeshStandardMaterial({
            color: 0xe2e8f0,
            metalness: 0.88,
            roughness: 0.22,
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
      }
}
