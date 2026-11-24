import * as THREE from "three";
import { BedMountType, ViewMode } from "../types";

/**
 * Exact standoff heights matching backend `BED_MOUNT_CATALOG` (mm).
 */
export const BED_MOUNT_STANDOFF_HEIGHT_MM: Record<BedMountType, number> = {
  trident_3pt_standoffs: 10,
  floating_4pt: 12,
  rigid_4pt: 12,
  whopping_pochard_kinematic: 14,
  kinematic_3pt: 16,
  silicone_4pt: 16,
  springs_4pt: 20,
};

export interface ModalBedMountItem {
  topGroup: THREE.Group; // bed-side hardware (moves & tilts with bed_surface)
  bottomGroup: THREE.Group; // subframe-side hardware (moves & tilts with bed_frame)
  flexGroup: THREE.Group; // compliant middle body spanning bottomAnchor -> topAnchor (local Y: 0..restFlexLen)
  mountOffsetFromBed: THREE.Vector3; // (dx, -bedThickScene*0.5, dz) relative to bedMesh center
  mountOffsetFromFrame: THREE.Vector3; // (dx, 0, dz) relative to subframeBasePos
  flexTopInsetY: number; // distance below bedBottomY where flex body meets topGroup
  flexBotInsetY: number; // distance above subframeTopY where flex body meets bottomGroup
  restFlexLen: number; // rest length of flexGroup along Y (> 0)
  radialYaw: number; // radial orientation angle toward bed center
}

export interface ModalGe5cJointItem {
  group: THREE.Group;
  nodeId: string; // "z_bed_fl" | "z_bed_fr" | "z_bed_r"
  offset: THREE.Vector3;
}

export interface BedMountsBuildResult {
  root: THREE.Group;
  subframeGroup: THREE.Group;
  subframeBasePos: THREE.Vector3;
  mountItems: ModalBedMountItem[];
  ge5cItems: ModalGe5cJointItem[];
}

export interface BedMountsBuildParams {
  bedMountType: BedMountType;
  zDriveMode?: string;
  architecture?: string;
  supportPtsThree: THREE.Vector3[];
  zArmNodes?: Array<{ nodeId: string; pos: THREE.Vector3 }>;
  bedBaseCenter: THREE.Vector3;
  bedThickScene: number;
  bedBottomY: number;
  bedDispDelta: THREE.Vector3;
  frameDispDelta?: THREE.Vector3;
  zArmDispDeltas?: Record<string, THREE.Vector3>;
  scale: number;
  viewMode: ViewMode;
  toolheadPunchIn?: boolean;
  useSensBedTint?: boolean;
  bedColor?: THREE.Color;
  bedEmissive?: THREE.Color;
  bedEmissiveIntensity?: number;
  onTagMesh?: (mesh: THREE.Mesh) => void;
}

const EXTRUSION_2020_MM = 20;

/**
 * Compute Y rotation so local +X points radially inward from `sp` toward `center` in the XZ plane.
 */
function radialYawToCenter(sp: THREE.Vector3, center: THREE.Vector3): number {
  const dx = center.x - sp.x;
  const dz = center.z - sp.z;
  if (dx * dx + dz * dz < 1e-8) return 0;
  return Math.atan2(-dz, dx);
}

/**
 * Create a realistic 2020 aluminum T-slot extrusion geometry along local Z (centered in X & Z,
 * with its top face flush at local `y = 0` and bottom face at `y = -20 * s`).
 */
function createExtrusion2020ProfileGeometry(length: number, s: number): THREE.ExtrudeGeometry {
  const h = 10.0 * s;
  const c = 1.15 * s; // corner chamfer
  const sw = 3.0 * s; // half-width of 6 mm T-slot opening
  const sd = 2.4 * s; // recessed slot channel depth

  const shape = new THREE.Shape();
  // Start at top-right chamfer, trace CCW around 2020 profile with 4 recessed T-slots
  shape.moveTo(h - c, h);
  shape.lineTo(sw, h);
  shape.lineTo(sw, h - sd);
  shape.lineTo(-sw, h - sd);
  shape.lineTo(-sw, h);
  shape.lineTo(-h + c, h);
  shape.lineTo(-h, h - c);
  // Left face
  shape.lineTo(-h, sw);
  shape.lineTo(-h + sd, sw);
  shape.lineTo(-h + sd, -sw);
  shape.lineTo(-h, -sw);
  shape.lineTo(-h, -h + c);
  shape.lineTo(-h + c, -h);
  // Bottom face
  shape.lineTo(-sw, -h);
  shape.lineTo(-sw, -h + sd);
  shape.lineTo(sw, -h + sd);
  shape.lineTo(sw, -h);
  shape.lineTo(h - c, -h);
  shape.lineTo(h, -h + c);
  // Right face
  shape.lineTo(h, -sw);
  shape.lineTo(h - sd, -sw);
  shape.lineTo(h - sd, sw);
  shape.lineTo(h, sw);
  shape.lineTo(h, h - c);
  shape.closePath();

  const geo = new THREE.ExtrudeGeometry(shape, {
    depth: length,
    bevelEnabled: false,
    curveSegments: 1,
  });
  // Top face strictly at y = 0, centered along local Z
  geo.translate(0, -h, -length * 0.5);
  return geo;
}

/**
 * Add a 2020 aluminum extrusion bar between two local XZ endpoints inside `parent` (top face at y = 0).
 */
function addExtrusion2020Bar(
  parent: THREE.Group,
  x0: number,
  z0: number,
  x1: number,
  z1: number,
  s: number,
  extrusionMat: THREE.Material,
  slotCoreMat: THREE.Material,
  tag: (m: THREE.Mesh) => THREE.Mesh
): void {
  const dx = x1 - x0;
  const dz = z1 - z0;
  const len = Math.hypot(dx, dz);
  if (len < 4 * s) return;

  const barGroup = new THREE.Group();
  barGroup.position.set((x0 + x1) * 0.5, 0, (z0 + z1) * 0.5);
  // Local +Z aligned along (dx, dz)
  barGroup.rotation.y = Math.atan2(dx, dz);

  const profileGeo = createExtrusion2020ProfileGeometry(len, s);
  const profileMesh = tag(new THREE.Mesh(profileGeo, extrusionMat));
  barGroup.add(profileMesh);

  // Recessed dark anodized inner core accent so the 4 T-slot grooves read crisply
  const coreGeo = new THREE.BoxGeometry(14.8 * s, 14.8 * s, Math.max(len - 0.8 * s, 1.0 * s));
  const coreMesh = new THREE.Mesh(coreGeo, slotCoreMat);
  coreMesh.position.y = -10.0 * s;
  barGroup.add(coreMesh);

  parent.add(barGroup);
}

/**
 * Create a continuous helical coil spring geometry (flat-ground ends) from `yBottom` to `yTop`.
 */
function createHelicalSpringGeometry(
  helixRadius: number,
  wireRadius: number,
  yTop: number,
  yBottom: number,
  turns: number
): THREE.TubeGeometry {
  const sampleCount = 80;
  const flatFrac = 0.35 / turns;
  const span = yTop - yBottom;
  const pts: THREE.Vector3[] = [];
  for (let i = 0; i <= sampleCount; i++) {
    const t = i / sampleCount;
    const theta = t * turns * Math.PI * 2;
    let yNorm: number;
    if (t < flatFrac) {
      const u = t / flatFrac;
      yNorm = 0.5 * flatFrac * u * u;
    } else if (t > 1 - flatFrac) {
      const u = (1 - t) / flatFrac;
      yNorm = 1 - 0.5 * flatFrac * u * u;
    } else {
      yNorm = 0.5 * flatFrac + ((t - flatFrac) * (1 - flatFrac)) / (1 - 2 * flatFrac);
    }
    const y = yBottom + yNorm * span;
    pts.push(new THREE.Vector3(Math.cos(theta) * helixRadius, y, Math.sin(theta) * helixRadius));
  }
  const curve = new THREE.CatmullRomCurve3(pts);
  return new THREE.TubeGeometry(curve, 84, wireRadius, 8, false);
}

/**
 * Create an extruded 18-lobe ribbed/knurled leveling thumbwheel centered at origin in XZ plane.
 */
function createKnurledThumbwheelGeometry(
  baseRadius: number,
  lobeDepth: number,
  lobes: number,
  thickness: number
): THREE.ExtrudeGeometry {
  const shape = new THREE.Shape();
  const steps = lobes * 4;
  for (let i = 0; i <= steps; i++) {
    const a = (i / steps) * Math.PI * 2;
    const r = baseRadius + lobeDepth * Math.cos(lobes * a);
    const x = Math.cos(a) * r;
    const y = Math.sin(a) * r;
    if (i === 0) shape.moveTo(x, y);
    else shape.lineTo(x, y);
  }
  shape.closePath();
  const geo = new THREE.ExtrudeGeometry(shape, {
    depth: thickness,
    bevelEnabled: true,
    bevelThickness: thickness * 0.14,
    bevelSize: lobeDepth * 0.45,
    bevelSegments: 1,
    curveSegments: 1,
  });
  geo.rotateX(Math.PI * 0.5);
  geo.translate(0, thickness * 0.5, 0);
  return geo;
}

/**
 * Create a realistic barreled & subtly ribbed 70A silicone bed isolator column from `yBottom` to `yTop`.
 */
function createSiliconeColumnGeometry(
  endRadius: number,
  midRadius: number,
  yTop: number,
  yBottom: number
): THREE.LatheGeometry {
  const span = yTop - yBottom;
  const profile: THREE.Vector2[] = [];
  const segs = 14;
  for (let i = 0; i <= segs; i++) {
    const t = i / segs; // 0 = bottom, 1 = top (CCW outward normals)
    const y = yBottom + t * span;
    const barrel = Math.sin(t * Math.PI);
    const rib = Math.sin(t * Math.PI * 4) * 0.032 * endRadius * barrel;
    const r = endRadius + (midRadius - endRadius) * Math.pow(barrel, 0.72) + rib;
    profile.push(new THREE.Vector2(Math.max(r, endRadius * 0.6), y));
  }
  return new THREE.LatheGeometry(profile, 20);
}

/**
 * Create a continuous S-stepped 301 spring-steel leaf flexure blade spanning local `y = 0` to `y = restFlexLen`.
 * Local `+X` = radial inward (bed-side upper clamp), local `-X` = radial outward (subframe lower clamp).
 */
function createLeafFlexureBladeGeometry(
  s: number,
  bladeWidth: number,
  bladeThick: number,
  restFlexLen: number
): THREE.ExtrudeGeometry {
  const halfT = bladeThick * 0.5;
  const yBot = halfT;
  const yTop = Math.max(restFlexLen - halfT, yBot + 1.0 * s);
  const yMid1 = yBot + (yTop - yBot) * 0.24;
  const yMid2 = yBot + (yTop - yBot) * 0.76;

  const topPts: Array<[number, number]> = [
    [10.2 * s, yTop + halfT],
    [2.6 * s, yTop + halfT],
    [0.6 * s, yMid2 + halfT],
    [-0.6 * s, yMid1 + halfT],
    [-2.6 * s, yBot + halfT],
    [-10.2 * s, yBot + halfT],
  ];
  const botPts: Array<[number, number]> = [
    [-10.2 * s, yBot - halfT],
    [-2.2 * s, yBot - halfT],
    [-0.2 * s, yMid1 - halfT],
    [1.0 * s, yMid2 - halfT],
    [3.0 * s, yTop - halfT],
    [10.2 * s, yTop - halfT],
  ];

  const shape = new THREE.Shape();
  shape.moveTo(topPts[0][0], topPts[0][1]);
  for (let i = 1; i < topPts.length; i++) {
    shape.lineTo(topPts[i][0], topPts[i][1]);
  }
  for (let i = 0; i < botPts.length; i++) {
    shape.lineTo(botPts[i][0], botPts[i][1]);
  }
  shape.closePath();

  const geo = new THREE.ExtrudeGeometry(shape, {
    depth: bladeWidth,
    bevelEnabled: false,
  });
  geo.translate(0, 0, -bladeWidth * 0.5);
  return geo;
}

/**
 * Align an articulated mount's `topGroup`, `bottomGroup`, and `flexGroup` cleanly between
 * live top/bottom world endpoints (used for static FEA pose & initial build).
 */
function placeArticulatedMountStatic(
  item: ModalBedMountItem,
  topWorldPos: THREE.Vector3,
  botWorldPos: THREE.Vector3
): void {
  const yawQuat = new THREE.Quaternion().setFromAxisAngle(
    new THREE.Vector3(0, 1, 0),
    item.radialYaw
  );

  item.topGroup.position.copy(topWorldPos);
  item.topGroup.quaternion.copy(yawQuat);

  item.bottomGroup.position.copy(botWorldPos);
  item.bottomGroup.quaternion.copy(yawQuat);

  const pFlexBot = new THREE.Vector3(
    botWorldPos.x,
    botWorldPos.y + item.flexBotInsetY,
    botWorldPos.z
  );
  const pFlexTop = new THREE.Vector3(
    topWorldPos.x,
    topWorldPos.y - item.flexTopInsetY,
    topWorldPos.z
  );
  const dir = pFlexTop.clone().sub(pFlexBot);
  const dist = dir.length();

  item.flexGroup.position.copy(pFlexBot);
  if (dist > 1e-5) {
    dir.divideScalar(dist);
    const alignQuat = new THREE.Quaternion();
    if (dir.y < -0.9999) {
      alignQuat.setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI);
    } else {
      alignQuat.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
    }
    item.flexGroup.quaternion.copy(alignQuat).multiply(yawQuat);
    const yScale = Math.min(2.2, Math.max(0.35, dist / Math.max(item.restFlexLen, 1e-5)));
    item.flexGroup.scale.set(1, yScale, 1);
  } else {
    item.flexGroup.quaternion.copy(yawQuat);
    item.flexGroup.scale.set(1, 1, 1);
  }
}

/**
 * Build realistic engineering-accurate 3D CAD bed mount assemblies + 2020 aluminum bed subframe +
 * Trident GE5C spherical Z-joints for all 7 `BedMountType` variants.
 *
 * Every mount is split cleanly into `topGroup`, `bottomGroup`, and `flexGroup` so Modal View and
 * Static FEA View articulate the heated bed, subframe, and compliant standoffs accurately with zero gap.
 *
 * All geometries and materials are attached to meshes inside `result.root` so `disposeGroupChildren`
 * frees every WebGL resource cleanly on scene rebuild.
 */
export function buildBedMountsGroup(params: BedMountsBuildParams): BedMountsBuildResult {
  const {
    bedMountType,
    zDriveMode,
    architecture,
    supportPtsThree,
    zArmNodes,
    bedBaseCenter,
    bedThickScene,
    bedBottomY,
    bedDispDelta,
    frameDispDelta,
    zArmDispDeltas,
    scale: s,
    viewMode,
    toolheadPunchIn = false,
    useSensBedTint = false,
    bedColor,
    bedEmissive,
    bedEmissiveIntensity = 0,
    onTagMesh,
  } = params;

  const root = new THREE.Group();
  root.name = `bed_mounts_${bedMountType}`;

  const standoffMm = BED_MOUNT_STANDOFF_HEIGHT_MM[bedMountType] ?? 14;
  const standoffH = standoffMm * s;
  const subframeTopY = bedBottomY - standoffH;
  const subframeBasePos = new THREE.Vector3(bedBaseCenter.x, subframeTopY, bedBaseCenter.z);
  const effFrameDisp = frameDispDelta ?? bedDispDelta;

  const subframeGroup = new THREE.Group();
  subframeGroup.name = "bed_subframe_2020";
  subframeGroup.position.copy(subframeBasePos).add(effFrameDisp);
  root.add(subframeGroup);

  const mountItems: ModalBedMountItem[] = [];
  const ge5cItems: ModalGe5cJointItem[] = [];

  if (supportPtsThree.length === 0) {
    return {
      root,
      subframeGroup,
      subframeBasePos,
      mountItems,
      ge5cItems,
    };
  }

  const isModal = viewMode === "modal";
  const isFeaStress = viewMode === "fea_stress";
  const isSensitivity = viewMode === "sensitivity";

  // Crisp mount visibility in modal mode (0.92) so elastic mount flex is clearly inspectable
  const mountTransparent = toolheadPunchIn || isModal;
  const mountOpacity = toolheadPunchIn ? 0.14 : isModal ? 0.92 : 1.0;

  // Subtle semi-transparent 2020 subframe in modal & fea_stress so bed_arm_* FEA colormaps remain legible
  const subframeTransparent = toolheadPunchIn || isModal || isFeaStress || isSensitivity;
  const subframeOpacity = toolheadPunchIn
    ? 0.12
    : isModal || isFeaStress
      ? 0.58
      : isSensitivity
        ? 0.72
        : 1.0;

  const makeMat = (opts: {
    color: number;
    metalness: number;
    roughness: number;
    flatShading?: boolean;
    sensTint?: boolean;
  }): THREE.MeshStandardMaterial => {
    const applySens = Boolean(opts.sensTint && useSensBedTint && bedColor);
    return new THREE.MeshStandardMaterial({
      color: applySens ? bedColor : opts.color,
      emissive: applySens && bedEmissive ? bedEmissive : 0x000000,
      emissiveIntensity: applySens ? bedEmissiveIntensity : 0,
      metalness: opts.metalness,
      roughness: opts.roughness,
      flatShading: opts.flatShading ?? false,
      transparent: mountTransparent,
      opacity: mountOpacity,
      depthWrite: !toolheadPunchIn,
    });
  };

  const tag = (mesh: THREE.Mesh): THREE.Mesh => {
    if (isModal) mesh.frustumCulled = false;
    if (onTagMesh) onTagMesh(mesh);
    return mesh;
  };

  // ---------------------------------------------------------------------------
  // 1. Build Realistic 2020 Aluminum Bed Subframe (`subframeGroup`, top at local y = 0)
  // ---------------------------------------------------------------------------
  const extrusionMat = new THREE.MeshStandardMaterial({
    color: 0x5c6b7e, // dark-silver anodized 6063-T5 2020 aluminum extrusion
    metalness: 0.76,
    roughness: 0.3,
    transparent: subframeTransparent,
    opacity: subframeOpacity,
    depthWrite: !toolheadPunchIn && !isModal && !isFeaStress,
  });
  const slotCoreMat = new THREE.MeshStandardMaterial({
    color: 0x131b26, // dark recessed T-slot groove contrast
    metalness: 0.35,
    roughness: 0.72,
    transparent: subframeTransparent,
    opacity: subframeOpacity,
    depthWrite: !toolheadPunchIn && !isModal && !isFeaStress,
  });
  const gussetMat = new THREE.MeshStandardMaterial({
    color: 0x2b3648, // dark anodized 2020 corner T-gusset brackets
    metalness: 0.78,
    roughness: 0.34,
    transparent: subframeTransparent,
    opacity: subframeOpacity,
    depthWrite: !toolheadPunchIn && !isModal && !isFeaStress,
  });

  const localPts = supportPtsThree.map((p) => ({
    x: p.x - subframeBasePos.x,
    z: p.z - subframeBasePos.z,
  }));

  let minMountX = Infinity;
  let maxMountX = -Infinity;
  let minMountZ = Infinity; // rear (-Z in Three.js)
  let maxMountZ = -Infinity; // front (+Z in Three.js)
  for (const lp of localPts) {
    if (lp.x < minMountX) minMountX = lp.x;
    if (lp.x > maxMountX) maxMountX = lp.x;
    if (lp.z < minMountZ) minMountZ = lp.z;
    if (lp.z > maxMountZ) maxMountZ = lp.z;
  }

  const isTridentTriple =
    zDriveMode === "leadscrew_triple" && (architecture === "corexy" || architecture === undefined);

  if (isTridentTriple) {
    // Authentic Voron / Siboor Trident 2020 Aluminum T-Subframe
    const zFlNode = zArmNodes?.find((n) => n.nodeId === "z_bed_fl");
    const zFrNode = zArmNodes?.find((n) => n.nodeId === "z_bed_fr");
    const zRearNode = zArmNodes?.find((n) => n.nodeId === "z_bed_r");

    // Front mount Z coordinate (largest Z in Three.js)
    const frontPts = localPts.filter((lp) => lp.z >= maxMountZ - 15 * s);
    const rearPts = localPts.filter((lp) => lp.z < maxMountZ - 15 * s);
    const zFrontBar =
      frontPts.length > 0
        ? frontPts.reduce((acc, p) => acc + p.z, 0) / frontPts.length
        : maxMountZ;

    // Extend front 2020 crossbar across FL & FR mounts and cleanly toward FL/FR Z-carriage brackets
    const flLimitX = zFlNode ? zFlNode.pos.x - subframeBasePos.x + 22 * s : minMountX - 38 * s;
    const frLimitX = zFrNode ? zFrNode.pos.x - subframeBasePos.x - 22 * s : maxMountX + 38 * s;
    const xFrontLeft = Math.max(flLimitX, minMountX - 36 * s);
    const xFrontRight = Math.min(frLimitX, maxMountX + 36 * s);

    addExtrusion2020Bar(
      subframeGroup,
      xFrontLeft,
      zFrontBar,
      xFrontRight,
      zFrontBar,
      s,
      extrusionMat,
      slotCoreMat,
      tag
    );

    // Central front-to-back 2020 spine running from front crossbar back under Rear-Center mount toward rear Z-carriage
    const centerSpineX =
      rearPts.length === 1 ? rearPts[0].x : (minMountX + maxMountX) * 0.5;
    const zSpineStart = zFrontBar - 10.0 * s; // flush butt-joint against back face of front crossbar
    const rearCarriageLimitZ = zRearNode
      ? zRearNode.pos.z - subframeBasePos.z + 34 * s
      : minMountZ - 32 * s;
    const zSpineEnd = Math.min(minMountZ - 24 * s, rearCarriageLimitZ);

    addExtrusion2020Bar(
      subframeGroup,
      centerSpineX,
      zSpineStart,
      centerSpineX,
      zSpineEnd,
      s,
      extrusionMat,
      slotCoreMat,
      tag
    );

    // Twin 2020 CNC T-junction corner gussets locking the central spine to the front crossbar
    const gussetGeo = new THREE.BoxGeometry(10.0 * s, 18.0 * s, 10.0 * s);
    for (const side of [-1, 1]) {
      const gMesh = new THREE.Mesh(gussetGeo, gussetMat);
      gMesh.position.set(centerSpineX + side * 15.0 * s, -10.0 * s, zFrontBar - 15.0 * s);
      if (isModal) gMesh.frustumCulled = false;
      subframeGroup.add(gMesh);
    }

    // If a 4-point mount kit is installed on a Trident, also add a rear 2020 crossbar under RL & RR mounts
    const hasSplitRearMounts = rearPts.some((p) => Math.abs(p.x - centerSpineX) > 15 * s);
    if (hasSplitRearMounts) {
      const zRearBar = rearPts.reduce((acc, p) => acc + p.z, 0) / rearPts.length;
      const rearMinX = Math.min(...rearPts.map((p) => p.x));
      const rearMaxX = Math.max(...rearPts.map((p) => p.x));
      addExtrusion2020Bar(
        subframeGroup,
        rearMinX - 24 * s,
        zRearBar,
        centerSpineX - 10.0 * s,
        zRearBar,
        s,
        extrusionMat,
        slotCoreMat,
        tag
      );
      addExtrusion2020Bar(
        subframeGroup,
        centerSpineX + 10.0 * s,
        zRearBar,
        rearMaxX + 24 * s,
        zRearBar,
        s,
        extrusionMat,
        slotCoreMat,
        tag
      );
    }
  } else {
    // Fixed-Base Voron 2.4 (`belted_quad_z`), Dual-Z, Single-Z, Bedslinger & Delta:
    // Twin parallel front-to-back 2020 extrusions centered directly under left & right bed mounts
    const zBarFront = maxMountZ + 26 * s;
    const zBarRear = minMountZ - 26 * s;
    const spanX = maxMountX - minMountX;

    if (spanX > 22 * s) {
      addExtrusion2020Bar(
        subframeGroup,
        minMountX,
        zBarFront,
        minMountX,
        zBarRear,
        s,
        extrusionMat,
        slotCoreMat,
        tag
      );
      addExtrusion2020Bar(
        subframeGroup,
        maxMountX,
        zBarFront,
        maxMountX,
        zBarRear,
        s,
        extrusionMat,
        slotCoreMat,
        tag
      );

      // Subtle cross-tie for any 3-point rear-center (or center) mount so it bolts solidly onto aluminum
      const centerMounts = localPts.filter(
        (lp) => Math.abs(lp.x - minMountX) > 12 * s && Math.abs(lp.x - maxMountX) > 12 * s
      );
      if (centerMounts.length > 0) {
        const tieZ = centerMounts.reduce((acc, p) => acc + p.z, 0) / centerMounts.length;
        addExtrusion2020Bar(
          subframeGroup,
          minMountX + 10.0 * s,
          tieZ,
          maxMountX - 10.0 * s,
          tieZ,
          s,
          extrusionMat,
          slotCoreMat,
          tag
        );
        const cornerBracketGeo = new THREE.BoxGeometry(8.0 * s, 18.0 * s, 8.0 * s);
        for (const bx of [minMountX + 14.0 * s, maxMountX - 14.0 * s]) {
          const cb = new THREE.Mesh(cornerBracketGeo, gussetMat);
          cb.position.set(bx, -10.0 * s, tieZ + (tieZ < 0 ? 14.0 * s : -14.0 * s));
          if (isModal) cb.frustumCulled = false;
          subframeGroup.add(cb);
        }
      } else {
        // Subtle central stabilizer tie on 4-point subframes (never colliding with corner mounts)
        const midZ = (minMountZ + maxMountZ) * 0.5;
        if (Math.abs(maxMountZ - minMountZ) > 60 * s) {
          addExtrusion2020Bar(
            subframeGroup,
            minMountX + 10.0 * s,
            midZ,
            maxMountX - 10.0 * s,
            midZ,
            s,
            extrusionMat,
            slotCoreMat,
            tag
          );
        }
      }
    } else {
      // Degenerate single-line mount pattern fallback
      const avgX = (minMountX + maxMountX) * 0.5;
      addExtrusion2020Bar(
        subframeGroup,
        avgX,
        zBarFront,
        avgX,
        zBarRear,
        s,
        extrusionMat,
        slotCoreMat,
        tag
      );
    }
  }

  // Ensure gussetMat is disposed if unused and disable frustum culling in modal animation
  let usesGussetMat = false;
  subframeGroup.traverse((c) => {
    if ((c as THREE.Mesh).isMesh) {
      if (isModal) c.frustumCulled = false;
      if ((c as THREE.Mesh).material === gussetMat) usesGussetMat = true;
    }
  });
  if (!usesGussetMat) gussetMat.dispose();

  // ---------------------------------------------------------------------------
  // Helper to register each articulated 3-part mount item (`topGroup`, `bottomGroup`, `flexGroup`)
  // ---------------------------------------------------------------------------
  const registerMountItem = (
    sp: THREE.Vector3,
    topGroup: THREE.Group,
    bottomGroup: THREE.Group,
    flexGroup: THREE.Group,
    flexTopInsetY: number,
    flexBotInsetY: number,
    radialYaw: number
  ): void => {
    const restFlexLen = Math.max(standoffH - flexTopInsetY - flexBotInsetY, 0.5 * s);
    const mountOffsetFromBed = new THREE.Vector3(
      sp.x - bedBaseCenter.x,
      -bedThickScene * 0.5,
      sp.z - bedBaseCenter.z
    );
    const mountOffsetFromFrame = new THREE.Vector3(
      sp.x - subframeBasePos.x,
      0,
      sp.z - subframeBasePos.z
    );

    const item: ModalBedMountItem = {
      topGroup,
      bottomGroup,
      flexGroup,
      mountOffsetFromBed,
      mountOffsetFromFrame,
      flexTopInsetY,
      flexBotInsetY,
      restFlexLen,
      radialYaw,
    };

    if (isModal) {
      for (const g of [topGroup, bottomGroup, flexGroup]) {
        g.traverse((c) => {
          if ((c as THREE.Mesh).isMesh) c.frustumCulled = false;
        });
      }
    }

    const topPos = new THREE.Vector3(
      sp.x + bedDispDelta.x,
      bedBottomY + bedDispDelta.y,
      sp.z + bedDispDelta.z
    );
    const botPos = new THREE.Vector3(
      sp.x + effFrameDisp.x,
      subframeTopY + effFrameDisp.y,
      sp.z + effFrameDisp.z
    );
    placeArticulatedMountStatic(item, topPos, botPos);

    root.add(topGroup);
    root.add(bottomGroup);
    root.add(flexGroup);
    mountItems.push(item);
  };

  // ---------------------------------------------------------------------------
  // 2. Build Articulated Bed Mount Hardware for all 7 `BedMountType` variants
  // ---------------------------------------------------------------------------
  if (bedMountType === "trident_3pt_standoffs") {
    // 1. Stock Voron / Siboor Trident 3-Pt Phenolic Bakelite Thermal-Isolation Standoffs (10 mm)
    const flexTopInsetY = 2.0 * s;
    const flexBotInsetY = 1.8 * s;
    const restFlexLen = standoffH - flexTopInsetY - flexBotInsetY; // 6.2 mm

    const bakeliteMat = makeMat({
      color: 0x852d12, // phenolic reddish-brown Bakelite thermal insulator
      metalness: 0.08,
      roughness: 0.4,
      sensTint: true,
    });
    const stainlessMat = makeMat({
      color: 0xdce4ee,
      metalness: 0.9,
      roughness: 0.2,
    });
    const socketDarkMat = makeMat({
      color: 0x1e293b,
      metalness: 0.52,
      roughness: 0.5,
    });
    const tnutBossMat = makeMat({
      color: 0x94a3b8,
      metalness: 0.84,
      roughness: 0.28,
    });

    // Identify rear-center fixed anchor (smallest Three.js Z = largest CAD Y)
    let rearFixedIdx = 0;
    let minZ = Infinity;
    supportPtsThree.forEach((pt, idx) => {
      if (pt.z < minZ) {
        minZ = pt.z;
        rearFixedIdx = idx;
      }
    });

    const hasSlottedFront = supportPtsThree.length > 1;
    const topFenderGeo = new THREE.CylinderGeometry(6.4 * s, 6.4 * s, 0.7 * s, 16);
    const m4HeadGeo = new THREE.CylinderGeometry(3.4 * s, 3.4 * s, 1.3 * s, 14);
    const fixedCollarGeo = new THREE.CylinderGeometry(9.0 * s, 8.6 * s, 1.3 * s, 20);
    const fixedCollarMat = makeMat({
      color: 0xe8eef5,
      metalness: 0.92,
      roughness: 0.16,
    });

    const slottedWasherGeo = hasSlottedFront
      ? new THREE.BoxGeometry(20.0 * s, 1.3 * s, 11.4 * s)
      : null;
    const slotRecessGeo = hasSlottedFront
      ? new THREE.BoxGeometry(8.8 * s, 1.45 * s, 3.0 * s)
      : null;
    const slottedWasherMat = hasSlottedFront
      ? makeMat({
          color: 0x38bdf8, // cyan-tinted slotted thermal-expansion washer on front 2 points
          metalness: 0.7,
          roughness: 0.26,
        })
      : null;

    // Bakelite column (flexGroup: y = 0..restFlexLen)
    const shoulderH = 1.1 * s;
    const barrelH = Math.max(restFlexLen - 2 * shoulderH, 1.0 * s);
    const botShoulderGeo = new THREE.CylinderGeometry(7.2 * s, 8.2 * s, shoulderH, 18);
    const barrelGeo = new THREE.CylinderGeometry(6.3 * s, 6.3 * s, barrelH, 18);
    const topShoulderGeo = new THREE.CylinderGeometry(8.2 * s, 7.2 * s, shoulderH, 18);
    const innerShaftGeo = new THREE.CylinderGeometry(2.0 * s, 2.0 * s, restFlexLen, 10);

    // Bottom 2020 T-nut clamping washer/boss (bottomGroup: y = 0..flexBotInsetY)
    const botClampingFlangeGeo = new THREE.CylinderGeometry(8.4 * s, 8.8 * s, 1.0 * s, 18);
    const botBossGeo = new THREE.CylinderGeometry(6.6 * s, 7.2 * s, 0.8 * s, 16);

    supportPtsThree.forEach((sp, idx) => {
      const yaw = radialYawToCenter(sp, bedBaseCenter);
      const isRearFixed = idx === rearFixedIdx;

      const topGroup = new THREE.Group();
      const topFender = new THREE.Mesh(topFenderGeo, stainlessMat);
      topFender.position.y = -0.35 * s;
      topGroup.add(topFender);

      const m4Head = new THREE.Mesh(m4HeadGeo, stainlessMat);
      m4Head.position.y = -0.65 * s;
      topGroup.add(m4Head);

      if (isRearFixed) {
        const fixedCollar = tag(new THREE.Mesh(fixedCollarGeo, fixedCollarMat));
        fixedCollar.position.y = -1.35 * s;
        topGroup.add(fixedCollar);
      } else if (slottedWasherGeo && slotRecessGeo && slottedWasherMat) {
        const slotWasher = tag(new THREE.Mesh(slottedWasherGeo, slottedWasherMat));
        slotWasher.position.y = -1.35 * s;
        topGroup.add(slotWasher);

        const slotRecess = new THREE.Mesh(slotRecessGeo, socketDarkMat);
        slotRecess.position.set(1.2 * s, -1.35 * s, 0);
        topGroup.add(slotRecess);
      }

      const flexGroup = new THREE.Group();
      const botShoulder = tag(new THREE.Mesh(botShoulderGeo, bakeliteMat));
      botShoulder.position.y = shoulderH * 0.5;
      flexGroup.add(botShoulder);

      const barrel = tag(new THREE.Mesh(barrelGeo, bakeliteMat));
      barrel.position.y = shoulderH + barrelH * 0.5;
      flexGroup.add(barrel);

      const topShoulder = tag(new THREE.Mesh(topShoulderGeo, bakeliteMat));
      topShoulder.position.y = restFlexLen - shoulderH * 0.5;
      flexGroup.add(topShoulder);

      const innerShaft = new THREE.Mesh(innerShaftGeo, stainlessMat);
      innerShaft.position.y = restFlexLen * 0.5;
      flexGroup.add(innerShaft);

      const bottomGroup = new THREE.Group();
      const clampFlange = tag(new THREE.Mesh(botClampingFlangeGeo, tnutBossMat));
      clampFlange.position.y = 0.5 * s;
      bottomGroup.add(clampFlange);

      const botBoss = new THREE.Mesh(botBossGeo, stainlessMat);
      botBoss.position.y = 1.4 * s;
      bottomGroup.add(botBoss);

      registerMountItem(sp, topGroup, bottomGroup, flexGroup, flexTopInsetY, flexBotInsetY, yaw);
    });

    if (!hasSlottedFront) {
      socketDarkMat.dispose();
    }
  } else if (bedMountType === "kinematic_3pt") {
    // 2. 3-Point Maxwell Kinematic Ball-in-V-Groove (16 mm)
    const flexTopInsetY = 3.2 * s;
    const flexBotInsetY = 5.2 * s;
    const restFlexLen = standoffH - flexTopInsetY - flexBotInsetY; // 7.6 mm

    const lowerPuckMat = makeMat({
      color: 0x334155, // dark-anodized CNC aluminum lower puck
      metalness: 0.78,
      roughness: 0.3,
      sensTint: true,
    });
    const upperCupMat = makeMat({
      color: 0xd8e0ea, // chamfered magnetic stainless steel upper cup
      metalness: 0.88,
      roughness: 0.2,
    });
    const chromeBallMat = makeMat({
      color: 0xf8fafc, // hardened G10 chrome steel bearing ball
      metalness: 0.96,
      roughness: 0.08,
    });
    const hardenedPinMat = makeMat({
      color: 0xcbd5e1, // hardened steel V-groove dowel pins
      metalness: 0.92,
      roughness: 0.14,
    });
    const neoMagnetMat = makeMat({
      color: 0xbae6fd, // recessed nickel-plated N52 neodymium preload magnet disc
      metalness: 0.9,
      roughness: 0.12,
    });

    // Top magnetic stainless washer/cup (y = 0..-3.2*s)
    const upperFlangeGeo = new THREE.CylinderGeometry(10.8 * s, 10.2 * s, 1.4 * s, 18);
    const upperCupGeo = new THREE.CylinderGeometry(8.2 * s, 6.0 * s, 1.8 * s, 18);

    // G10 hardened chrome bearing ball (flexGroup: y = 0..7.6*s, centered at y = 3.8*s, R = 4.0*s)
    const ballGeo = new THREE.SphereGeometry(4.0 * s, 20, 14);

    // Lower CNC puck + twin radial V-groove dowel pins + N52 preload magnet (bottomGroup: y = 0..+5.2*s)
    const lowerPuckGeo = new THREE.CylinderGeometry(11.4 * s, 12.6 * s, 4.0 * s, 20);
    const vPinGeo = new THREE.CylinderGeometry(2.0 * s, 2.0 * s, 16.0 * s, 12);
    const magnetGeo = new THREE.CylinderGeometry(4.5 * s, 4.5 * s, 1.2 * s, 16);
    const baseBoltGeo = new THREE.CylinderGeometry(1.7 * s, 1.7 * s, 1.6 * s, 10);

    for (const sp of supportPtsThree) {
      const yaw = radialYawToCenter(sp, bedBaseCenter);

      const topGroup = new THREE.Group();
      const upperFlange = tag(new THREE.Mesh(upperFlangeGeo, upperCupMat));
      upperFlange.position.y = -0.7 * s;
      topGroup.add(upperFlange);

      const upperCup = tag(new THREE.Mesh(upperCupGeo, upperCupMat));
      upperCup.position.y = -2.3 * s;
      topGroup.add(upperCup);

      const flexGroup = new THREE.Group();
      const ball = tag(new THREE.Mesh(ballGeo, chromeBallMat));
      ball.position.y = restFlexLen * 0.5;
      flexGroup.add(ball);

      const bottomGroup = new THREE.Group();
      const lowerPuck = tag(new THREE.Mesh(lowerPuckGeo, lowerPuckMat));
      lowerPuck.position.y = 2.0 * s;
      bottomGroup.add(lowerPuck);

      // Recessed N52 neodymium preload magnet disc centered between the V-groove pins
      const magnet = new THREE.Mesh(magnetGeo, neoMagnetMat);
      magnet.position.y = 4.1 * s;
      bottomGroup.add(magnet);

      // Twin parallel hardened steel dowel pins oriented radially along local +X (90 deg V-groove)
      for (const side of [-1, 1]) {
        const pin = tag(new THREE.Mesh(vPinGeo, hardenedPinMat));
        pin.rotation.z = Math.PI * 0.5;
        pin.position.set(0, 4.4 * s, side * 3.8 * s);
        bottomGroup.add(pin);
      }

      for (const side of [-1, 1]) {
        const bBolt = new THREE.Mesh(baseBoltGeo, upperCupMat);
        bBolt.position.set(0, 4.5 * s, side * 8.6 * s);
        bottomGroup.add(bBolt);
      }

      registerMountItem(sp, topGroup, bottomGroup, flexGroup, flexTopInsetY, flexBotInsetY, yaw);
    }
  } else if (bedMountType === "whopping_pochard_kinematic") {
    // 3. Whopping Pochard 301 Spring-Steel Leaf-Spring Kinematic (14 mm, Tap-Rated)
    const flexTopInsetY = 3.8 * s;
    const flexBotInsetY = 4.6 * s;
    const restFlexLen = standoffH - flexTopInsetY - flexBotInsetY; // 5.6 mm

    const leafBladeMat = makeMat({
      color: 0xe2e8f0, // 301 stainless spring-steel leaf flexure blade
      metalness: 0.94,
      roughness: 0.16,
      sensTint: true,
    });
    const upperClampMat = makeMat({
      color: 0xcbd5e1, // machined stainless bed-side clamp block
      metalness: 0.86,
      roughness: 0.24,
    });
    const lowerBlockMat = makeMat({
      color: 0x273244, // stepped dark-anodized CNC lower clamp block
      metalness: 0.78,
      roughness: 0.34,
      sensTint: true,
    });
    const stainlessBoltMat = makeMat({
      color: 0xf1f5f9,
      metalness: 0.92,
      roughness: 0.16,
    });
    const alignPinMat = makeMat({
      color: 0x38bdf8, // hardened radial alignment dowel pin
      metalness: 0.85,
      roughness: 0.2,
    });

    // Upper clamp (y = 0..-3.8*s + lower clamping shim lip)
    const upperClampGeo = new THREE.BoxGeometry(12.6 * s, 3.8 * s, 15.6 * s);
    const upperShimGeo = new THREE.BoxGeometry(10.0 * s, 1.1 * s, 13.4 * s);
    const m3BoltGeo = new THREE.CylinderGeometry(1.9 * s, 1.9 * s, 2.0 * s, 10);
    const alignPinGeo = new THREE.CylinderGeometry(1.7 * s, 1.7 * s, 4.2 * s, 12);

    // S-stepped spring-steel blade (flexGroup: y = 0..5.6*s)
    const leafBladeGeo = createLeafFlexureBladeGeometry(s, 12.4 * s, 1.0 * s, restFlexLen);

    // Lower stepped CNC clamp block + M5 T-nut bolt (bottomGroup: y = 0..+4.6*s + clamp lip)
    const lowerBaseGeo = new THREE.BoxGeometry(14.4 * s, 4.6 * s, 16.4 * s);
    const lowerStepGeo = new THREE.BoxGeometry(9.6 * s, 1.4 * s, 13.4 * s);
    const m5TnutBoltGeo = new THREE.CylinderGeometry(3.2 * s, 3.2 * s, 1.6 * s, 12);

    for (const sp of supportPtsThree) {
      const yaw = radialYawToCenter(sp, bedBaseCenter);

      const topGroup = new THREE.Group();
      const upperClamp = tag(new THREE.Mesh(upperClampGeo, upperClampMat));
      upperClamp.position.set(5.4 * s, -1.9 * s, 0);
      topGroup.add(upperClamp);

      const upperShim = new THREE.Mesh(upperShimGeo, lowerBlockMat);
      upperShim.position.set(5.4 * s, -4.45 * s, 0);
      topGroup.add(upperShim);

      for (const side of [-1, 1]) {
        const bolt = new THREE.Mesh(m3BoltGeo, stainlessBoltMat);
        bolt.position.set(5.4 * s, -5.6 * s, side * 4.4 * s);
        topGroup.add(bolt);
      }

      const alignPin = new THREE.Mesh(alignPinGeo, alignPinMat);
      alignPin.position.set(1.6 * s, -2.5 * s, 0);
      topGroup.add(alignPin);

      const flexGroup = new THREE.Group();
      const blade = tag(new THREE.Mesh(leafBladeGeo, leafBladeMat));
      flexGroup.add(blade);

      const bottomGroup = new THREE.Group();
      const lowerBase = tag(new THREE.Mesh(lowerBaseGeo, lowerBlockMat));
      lowerBase.position.set(-5.4 * s, 2.3 * s, 0);
      bottomGroup.add(lowerBase);

      const lowerStep = new THREE.Mesh(lowerStepGeo, upperClampMat);
      lowerStep.position.set(-5.4 * s, 5.4 * s, 0);
      bottomGroup.add(lowerStep);

      for (const side of [-1, 1]) {
        const lowerBolt = new THREE.Mesh(m3BoltGeo, stainlessBoltMat);
        lowerBolt.position.set(-5.4 * s, 6.7 * s, side * 4.4 * s);
        bottomGroup.add(lowerBolt);
      }

      const m5Bolt = new THREE.Mesh(m5TnutBoltGeo, stainlessBoltMat);
      m5Bolt.position.set(-9.6 * s, 4.9 * s, 0);
      bottomGroup.add(m5Bolt);

      registerMountItem(sp, topGroup, bottomGroup, flexGroup, flexTopInsetY, flexBotInsetY, yaw);
    }
  } else if (bedMountType === "floating_4pt") {
    // 4. 4-Pt Floating — Anchor #0 Fixed Shoulder Bolt + Anchors #1–3 Radially Slotted Tabs (12 mm)
    const flexTopInsetY = 3.2 * s;
    const flexBotInsetY = 3.6 * s;
    const restFlexLen = standoffH - flexTopInsetY - flexBotInsetY; // 5.2 mm

    const fixedBushingMat = makeMat({
      color: 0xcbd5e1,
      metalness: 0.88,
      roughness: 0.2,
      sensTint: true,
    });
    const stainlessMat = makeMat({
      color: 0xe8eef5,
      metalness: 0.92,
      roughness: 0.16,
    });

    const fixedTopFlangeGeo = new THREE.CylinderGeometry(9.8 * s, 9.8 * s, 1.8 * s, 18);
    const fixedLocatorGeo = new THREE.CylinderGeometry(7.8 * s, 7.8 * s, 1.4 * s, 18);
    const fixedSleeveGeo = new THREE.CylinderGeometry(6.8 * s, 7.2 * s, restFlexLen, 18);
    const fixedBotBossGeo = new THREE.CylinderGeometry(8.2 * s, 9.8 * s, 3.6 * s, 18);
    const shoulderPinGeo = new THREE.CylinderGeometry(3.8 * s, 3.8 * s, restFlexLen, 14);

    const hasFloatingCorners = supportPtsThree.length > 1;
    const tabMat = hasFloatingCorners
      ? makeMat({
          color: 0x94a3b8, // slotted anodized aluminum mount tab
          metalness: 0.78,
          roughness: 0.28,
          sensTint: true,
        })
      : null;
    const slotDarkMat = hasFloatingCorners
      ? makeMat({
          color: 0x0f172a, // recessed dark thermal expansion slot
          metalness: 0.2,
          roughness: 0.85,
        })
      : null;
    const brassWasherMat = hasFloatingCorners
      ? makeMat({
          color: 0xd4a338, // brass glide washer
          metalness: 0.84,
          roughness: 0.24,
        })
      : null;
    const ptfeGlideMat = hasFloatingCorners
      ? makeMat({
          color: 0xf8fafc, // low-friction white PTFE glide washer
          metalness: 0.05,
          roughness: 0.35,
        })
      : null;

    const floatTopFlangeGeo = hasFloatingCorners
      ? new THREE.CylinderGeometry(8.4 * s, 8.4 * s, 1.2 * s, 18)
      : null;
    const ptfeRingGeo = hasFloatingCorners
      ? new THREE.CylinderGeometry(7.6 * s, 7.6 * s, 0.9 * s, 18)
      : null;
    const brassWasherGeo = hasFloatingCorners
      ? new THREE.CylinderGeometry(7.0 * s, 7.0 * s, 1.1 * s, 18)
      : null;
    const midGuideSleeveGeo = hasFloatingCorners
      ? new THREE.CylinderGeometry(5.2 * s, 5.2 * s, restFlexLen * 0.48, 16)
      : null;
    const slottedTabGeo = hasFloatingCorners
      ? new THREE.BoxGeometry(24.0 * s, 3.6 * s, 12.4 * s)
      : null;
    const slotRecessGeo = hasFloatingCorners
      ? new THREE.BoxGeometry(12.2 * s, 3.8 * s, 3.8 * s)
      : null;

    supportPtsThree.forEach((sp, idx) => {
      const yaw = radialYawToCenter(sp, bedBaseCenter);
      const topGroup = new THREE.Group();
      const flexGroup = new THREE.Group();
      const bottomGroup = new THREE.Group();

      if (idx === 0) {
        // Anchor #0: rigid fixed stainless locator bushing + steel boss
        const topFlange = tag(new THREE.Mesh(fixedTopFlangeGeo, fixedBushingMat));
        topFlange.position.y = -0.9 * s;
        topGroup.add(topFlange);

        const locator = tag(new THREE.Mesh(fixedLocatorGeo, stainlessMat));
        locator.position.y = -2.5 * s;
        topGroup.add(locator);

        const sleeve = tag(new THREE.Mesh(fixedSleeveGeo, fixedBushingMat));
        sleeve.position.y = restFlexLen * 0.5;
        flexGroup.add(sleeve);

        const pin = new THREE.Mesh(shoulderPinGeo, stainlessMat);
        pin.position.y = restFlexLen * 0.5;
        flexGroup.add(pin);

        const botBoss = tag(new THREE.Mesh(fixedBotBossGeo, fixedBushingMat));
        botBoss.position.y = 1.8 * s;
        bottomGroup.add(botBoss);
      } else if (
        floatTopFlangeGeo &&
        ptfeRingGeo &&
        brassWasherGeo &&
        midGuideSleeveGeo &&
        slottedTabGeo &&
        slotRecessGeo &&
        tabMat &&
        slotDarkMat &&
        brassWasherMat &&
        ptfeGlideMat
      ) {
        // Anchors #1–3: shoulder flange + PTFE/brass glide stack + stainless shoulder shaft + slotted aluminum tab
        const topFlange = new THREE.Mesh(floatTopFlangeGeo, stainlessMat);
        topFlange.position.y = -0.6 * s;
        topGroup.add(topFlange);

        const ptfeWasher = new THREE.Mesh(ptfeRingGeo, ptfeGlideMat);
        ptfeWasher.position.y = -1.65 * s;
        topGroup.add(ptfeWasher);

        const brassWasher = tag(new THREE.Mesh(brassWasherGeo, brassWasherMat));
        brassWasher.position.y = -2.65 * s;
        topGroup.add(brassWasher);

        const pin = tag(new THREE.Mesh(shoulderPinGeo, stainlessMat));
        pin.position.y = restFlexLen * 0.5;
        flexGroup.add(pin);

        const midSleeve = new THREE.Mesh(midGuideSleeveGeo, tabMat);
        midSleeve.position.y = restFlexLen * 0.5;
        flexGroup.add(midSleeve);

        const tab = tag(new THREE.Mesh(slottedTabGeo, tabMat));
        tab.position.y = 1.8 * s;
        bottomGroup.add(tab);

        const slotRecess = new THREE.Mesh(slotRecessGeo, slotDarkMat);
        slotRecess.position.y = 1.8 * s;
        bottomGroup.add(slotRecess);
      }

      registerMountItem(sp, topGroup, bottomGroup, flexGroup, flexTopInsetY, flexBotInsetY, yaw);
    });
  } else if (bedMountType === "silicone_4pt") {
    // 6. 4-Pt 70A Matte Orange Silicone Isolator Columns (16 mm)
    const flexTopInsetY = 1.8 * s;
    const flexBotInsetY = 1.8 * s;
    const restFlexLen = standoffH - flexTopInsetY - flexBotInsetY; // 12.4 mm

    const siliconeMat = makeMat({
      color: 0xea580c, // matte 70A orange silicone rubber
      metalness: 0.04,
      roughness: 0.82,
      sensTint: true,
    });
    const washerMat = makeMat({
      color: 0xcbd5e1, // chamfered stainless compression washers
      metalness: 0.86,
      roughness: 0.24,
    });
    const screwMat = makeMat({
      color: 0x94a3b8,
      metalness: 0.88,
      roughness: 0.22,
    });

    const topWasherGeo = new THREE.CylinderGeometry(8.4 * s, 9.4 * s, 1.8 * s, 20);
    const cskHeadGeo = new THREE.CylinderGeometry(3.8 * s, 2.2 * s, 1.5 * s, 14);
    const columnGeo = createSiliconeColumnGeometry(7.2 * s, 8.8 * s, restFlexLen, 0);
    const innerGuideGeo = new THREE.CylinderGeometry(1.9 * s, 1.9 * s, restFlexLen, 10);
    const botWasherGeo = new THREE.CylinderGeometry(9.4 * s, 8.4 * s, 1.8 * s, 20);

    for (const sp of supportPtsThree) {
      const yaw = radialYawToCenter(sp, bedBaseCenter);

      const topGroup = new THREE.Group();
      const topWasher = new THREE.Mesh(topWasherGeo, washerMat);
      topWasher.position.y = -0.9 * s;
      topGroup.add(topWasher);

      const cskHead = new THREE.Mesh(cskHeadGeo, screwMat);
      cskHead.position.y = -0.8 * s;
      topGroup.add(cskHead);

      const flexGroup = new THREE.Group();
      const column = tag(new THREE.Mesh(columnGeo, siliconeMat));
      flexGroup.add(column);

      const innerGuide = new THREE.Mesh(innerGuideGeo, screwMat);
      innerGuide.position.y = restFlexLen * 0.5;
      flexGroup.add(innerGuide);

      const bottomGroup = new THREE.Group();
      const botWasher = new THREE.Mesh(botWasherGeo, washerMat);
      botWasher.position.y = 0.9 * s;
      bottomGroup.add(botWasher);

      registerMountItem(sp, topGroup, bottomGroup, flexGroup, flexTopInsetY, flexBotInsetY, yaw);
    }
  } else if (bedMountType === "springs_4pt") {
    // 7. 4-Pt Yellow ISO Die Springs (20 mm) — True Helical Coil + Under-Bar Knurled Thumbwheel
    const flexTopInsetY = 2.2 * s;
    const flexBotInsetY = 2.2 * s;
    const restFlexLen = standoffH - flexTopInsetY - flexBotInsetY; // 15.6 mm

    const springMat = makeMat({
      color: 0xfacc15, // matte yellow ISO die-spring steel
      metalness: 0.34,
      roughness: 0.42,
      sensTint: true,
    });
    const cupMat = makeMat({
      color: 0x334155, // upper & lower dark steel spring seat cups
      metalness: 0.74,
      roughness: 0.38,
    });
    const rodMat = makeMat({
      color: 0xb8c4d4,
      metalness: 0.86,
      roughness: 0.24,
    });
    const wheelMat = makeMat({
      color: 0x1e293b, // matte black 18-lobe knurled POM/nylon leveling thumbwheel
      metalness: 0.14,
      roughness: 0.62,
    });
    const brassInsertMat = makeMat({
      color: 0xd4a338, // brass threaded insert hub in thumbwheel
      metalness: 0.84,
      roughness: 0.24,
    });

    const wireR = 1.35 * s;
    const topCupGeo = new THREE.CylinderGeometry(8.4 * s, 8.9 * s, 2.2 * s, 18);
    const m4BossGeo = new THREE.CylinderGeometry(3.2 * s, 3.2 * s, 1.8 * s, 12);
    const helixGeo = createHelicalSpringGeometry(
      5.8 * s,
      wireR,
      restFlexLen - wireR,
      wireR,
      5.5
    );
    const flexGuideGeo = new THREE.CylinderGeometry(1.9 * s, 1.9 * s, restFlexLen, 10);
    const botCupGeo = new THREE.CylinderGeometry(8.9 * s, 8.4 * s, 2.2 * s, 18);
    // Stud passes cleanly through the 20 mm subframe down to the under-bar leveling thumbwheel
    const lowerStudLen = (EXTRUSION_2020_MM + 5.2) * s;
    const lowerStudGeo = new THREE.CylinderGeometry(1.9 * s, 1.9 * s, lowerStudLen, 10);
    const thumbwheelGeo = createKnurledThumbwheelGeometry(13.8 * s, 1.2 * s, 18, 3.6 * s);
    const brassHubGeo = new THREE.CylinderGeometry(4.2 * s, 4.2 * s, 4.4 * s, 14);
    const wheelCenterY = -(EXTRUSION_2020_MM + 2.0) * s;

    for (const sp of supportPtsThree) {
      const yaw = radialYawToCenter(sp, bedBaseCenter);

      const topGroup = new THREE.Group();
      const topCup = new THREE.Mesh(topCupGeo, cupMat);
      topCup.position.y = -1.1 * s;
      topGroup.add(topCup);

      const m4Boss = new THREE.Mesh(m4BossGeo, rodMat);
      m4Boss.position.y = -1.1 * s;
      topGroup.add(m4Boss);

      const flexGroup = new THREE.Group();
      const springCoil = tag(new THREE.Mesh(helixGeo, springMat));
      flexGroup.add(springCoil);

      const guideRod = new THREE.Mesh(flexGuideGeo, rodMat);
      guideRod.position.y = restFlexLen * 0.5;
      flexGroup.add(guideRod);

      const bottomGroup = new THREE.Group();
      const botCup = new THREE.Mesh(botCupGeo, cupMat);
      botCup.position.y = 1.1 * s;
      bottomGroup.add(botCup);

      const lowerStud = new THREE.Mesh(lowerStudGeo, rodMat);
      lowerStud.position.y = -lowerStudLen * 0.5;
      bottomGroup.add(lowerStud);

      // 18-lobe ribbed black knurled leveling thumbwheel + brass threaded hub underneath the 2020 bar
      const wheel = tag(new THREE.Mesh(thumbwheelGeo, wheelMat));
      wheel.position.y = wheelCenterY;
      bottomGroup.add(wheel);

      const brassHub = new THREE.Mesh(brassHubGeo, brassInsertMat);
      brassHub.position.y = wheelCenterY;
      bottomGroup.add(brassHub);

      registerMountItem(sp, topGroup, bottomGroup, flexGroup, flexTopInsetY, flexBotInsetY, yaw);
    }
  } else {
    // 5. rigid_4pt — 4-Pt Rigid Hard-Bolted Machined Hexagonal Standoffs (12 mm)
    const flexTopInsetY = 1.8 * s;
    const flexBotInsetY = 1.8 * s;
    const restFlexLen = standoffH - flexTopInsetY - flexBotInsetY; // 8.4 mm

    const hexStandoffMat = makeMat({
      color: 0xdce4ee, // machined 6-sided hexagonal aluminum/steel standoff
      metalness: 0.88,
      roughness: 0.2,
      flatShading: true,
      sensTint: true,
    });
    const washerMat = makeMat({
      color: 0xb8c5d6,
      metalness: 0.86,
      roughness: 0.25,
    });
    const screwMat = makeMat({
      color: 0xe2e8f0,
      metalness: 0.9,
      roughness: 0.18,
    });

    const topFlangeGeo = new THREE.CylinderGeometry(9.4 * s, 10.2 * s, 1.8 * s, 20);
    const capScrewGeo = new THREE.CylinderGeometry(3.8 * s, 3.8 * s, 1.6 * s, 14);
    const hexBodyGeo = new THREE.CylinderGeometry(7.2 * s, 7.2 * s, restFlexLen, 6);
    const botFlangeGeo = new THREE.CylinderGeometry(10.2 * s, 9.4 * s, 1.8 * s, 20);

    for (const sp of supportPtsThree) {
      const yaw = radialYawToCenter(sp, bedBaseCenter);

      const topGroup = new THREE.Group();
      const topWasher = new THREE.Mesh(topFlangeGeo, washerMat);
      topWasher.position.y = -0.9 * s;
      topGroup.add(topWasher);

      const capScrew = new THREE.Mesh(capScrewGeo, screwMat);
      capScrew.position.y = -0.85 * s;
      topGroup.add(capScrew);

      const flexGroup = new THREE.Group();
      const hexStandoff = tag(new THREE.Mesh(hexBodyGeo, hexStandoffMat));
      hexStandoff.position.y = restFlexLen * 0.5;
      flexGroup.add(hexStandoff);

      const bottomGroup = new THREE.Group();
      const botWasher = new THREE.Mesh(botFlangeGeo, washerMat);
      botWasher.position.y = 0.9 * s;
      bottomGroup.add(botWasher);

      registerMountItem(sp, topGroup, bottomGroup, flexGroup, flexTopInsetY, flexBotInsetY, yaw);
    }
  }

  // ---------------------------------------------------------------------------
  // 3. Build 3× GE5C Spherical Rod-End Bearing Assemblies for ALL `leadscrew_triple` builds
  // ---------------------------------------------------------------------------
  if (isTridentTriple && zArmNodes && zArmNodes.length > 0) {
    const ge5cOuterMat = makeMat({
      color: 0x9aa8ba, // machined steel outer housing
      metalness: 0.88,
      roughness: 0.22,
    });
    const ge5cBronzeMat = makeMat({
      color: 0xc9933b, // bronze/brass spherical inner ball
      metalness: 0.82,
      roughness: 0.24,
    });
    const ge5cPinMat = makeMat({
      color: 0xe8eef5, // stainless M5 clevis pin
      metalness: 0.92,
      roughness: 0.16,
    });
    const ge5cBracketMat = makeMat({
      color: 0x2b3648, // dark anodized Z-carriage clevis bracket
      metalness: 0.74,
      roughness: 0.34,
    });

    const outerRaceGeo = new THREE.CylinderGeometry(8.6 * s, 8.6 * s, 6.0 * s, 20);
    const innerBallGeo = new THREE.SphereGeometry(6.3 * s, 16, 12);
    const clevisPinGeo = new THREE.CylinderGeometry(2.5 * s, 2.5 * s, 11.4 * s, 12);
    const pinCapGeo = new THREE.CylinderGeometry(4.0 * s, 4.0 * s, 2.2 * s, 12);
    const zBracketGeo = new THREE.BoxGeometry(11.0 * s, 12.6 * s, 12.0 * s);
    const rodEndShankGeo = new THREE.BoxGeometry(10.5 * s, 6.6 * s, 10.6 * s);

    const subframeCenterY = subframeTopY - 10.0 * s;

    for (const zNode of zArmNodes) {
      const zPt = zNode.pos;
      const toCenter = new THREE.Vector3(bedBaseCenter.x - zPt.x, 0, bedBaseCenter.z - zPt.z);
      const horizLen = toCenter.length();
      const inwardDist = horizLen > 0.35 ? 25.0 * s : 0;
      if (horizLen > 1e-4) toCenter.divideScalar(horizLen);

      const offset = new THREE.Vector3(
        toCenter.x * inwardDist,
        subframeCenterY - zPt.y,
        toCenter.z * inwardDist
      );

      const bearingGrp = new THREE.Group();
      bearingGrp.name = `ge5c_${zNode.nodeId}`;
      bearingGrp.rotation.y = radialYawToCenter(zPt, bedBaseCenter);

      const outerRace = tag(new THREE.Mesh(outerRaceGeo, ge5cOuterMat));
      bearingGrp.add(outerRace);

      const bronzeBall = tag(new THREE.Mesh(innerBallGeo, ge5cBronzeMat));
      bearingGrp.add(bronzeBall);

      const centerPin = new THREE.Mesh(clevisPinGeo, ge5cPinMat);
      bearingGrp.add(centerPin);

      const pinCap = new THREE.Mesh(pinCapGeo, ge5cPinMat);
      pinCap.position.y = 6.2 * s;
      bearingGrp.add(pinCap);

      // Z-carriage clevis bracket on outward (-X) side toward vertical Z-rail
      const zBracket = tag(new THREE.Mesh(zBracketGeo, ge5cBracketMat));
      zBracket.position.set(-10.5 * s, 0, 0);
      bearingGrp.add(zBracket);

      // Machined rod-end collar shank on inward (+X) side toward T-subframe
      const rodShank = new THREE.Mesh(rodEndShankGeo, ge5cOuterMat);
      rodShank.position.set(10.0 * s, 0, 0);
      bearingGrp.add(rodShank);

      if (isModal) {
        bearingGrp.traverse((c) => {
          if ((c as THREE.Mesh).isMesh) c.frustumCulled = false;
        });
      }

      const zDelta = zArmDispDeltas?.[zNode.nodeId];
      bearingGrp.position.copy(zPt).add(offset);
      if (zDelta) bearingGrp.position.add(zDelta);

      root.add(bearingGrp);
      ge5cItems.push({
        group: bearingGrp,
        nodeId: zNode.nodeId,
        offset,
      });
    }
  }

  return {
    root,
    subframeGroup,
    subframeBasePos,
    mountItems,
    ge5cItems,
  };
}
